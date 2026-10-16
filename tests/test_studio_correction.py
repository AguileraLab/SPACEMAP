"""Studio's batch correction runs classic Tab 6's steps between slides and keeps what differs inside a slide."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import correction, samples

MARKERS = ["A", "B", "C"]


def box(label, x):
    ring = [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10], [x, 0]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"name": label}}


@pytest.fixture
def project(tmp_path):
    """Two slides, two ROIs each; S2 is stained 1.8x brighter; roi_2 (a later timepoint) has twice the A."""
    root = tmp_path / "study"
    shapes = root / "data/boundaries"
    shapes.mkdir(parents=True)
    tables = root / "outputs/segmentation"
    tables.mkdir(parents=True)
    images = root / "outputs/rois"
    images.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for slide, bright in (("S1", 1.0), ("S2", 1.8)):
        (shapes / f"{slide}_roi_bounding_boxes_coords.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": [box("roi_1", 0), box("roi_2", 20)]})
        )
        for roi, more_a in (("roi_1", 1.0), ("roi_2", 2.0)):
            (images / f"{slide}_{roi}.ome.tiff").touch()
            n = 600
            kind = rng.random(n) < 0.5  # two cell types in every ROI
            frame = pd.DataFrame(
                {
                    "label": np.arange(1, n + 1),
                    "area": rng.uniform(40, 60, n),
                    "A": bright * more_a * np.where(kind, 400, 40) * rng.lognormal(0, 0.2, n),
                    "B": bright * np.where(kind, 30, 300) * rng.lognormal(0, 0.2, n),
                    "C": bright * 100 * rng.lognormal(0, 0.2, n),
                }
            )
            frame.to_csv(tables / f"{slide}_{roi}.ome_cell_features.csv", index=False)
    config = {
        "tab_configurations": {
            "tab6_batch_correction": {
                "bc_config": {"som_iterations": 2000, "som_grid_dim": 4},
                "bc_selected_markers": MARKERS,
            }
        }
    }
    samples.link(root, {}, "slide,timepoint\nS1,pre\nS2,pre\n", "slide", "slide", {"timepoint": ""}, apply=True)
    return root, config


def job(root, config, **changes):
    from spacemap_gui.studio import sample_links
    plan = correction.plan(root, {}, config)
    settings = correction.config_of(config)
    settings.update({"batch_unit_col": plan["unit"], **changes})
    return {
        "project_root": str(root),
        "config": settings,
        "markers": plan["markers"],
        "tables": correction.tables(root, {}),
        "samples": samples.values(root),
        "sample_links": sample_links.read(root),
        "output_dir": "",
    }


def arcsinh_median(frame, column, mask):
    return float(np.median(np.arcsinh(frame.loc[mask, column] / 5.0)))


def test_the_plan_offers_slides_as_the_unit(project):
    root, config = project
    plan = correction.plan(root, {}, config)
    assert plan["tables"] == 4 and plan["markers"] == MARKERS and plan["unit"] == "slide"
    units = {item["name"]: item["units"] for item in plan["units"]}
    assert units["ROI"] == 4 and units["slide"] == 2 and plan["covariate_options"] == ["timepoint"]


def test_correcting_between_slides_removes_the_stain_and_keeps_the_timepoints(project):
    root, config = project
    assert correction.plan(root, {}, config)["settings"]["correction_mode"] == "cycombine"  # never chosen: cyCombine
    found = correction.run(job(root, config), root / "job", lambda *args: None)
    assert found["unit"] == "slide" and found["units"] == ["S1", "S2"] and found["mode"] == "cycombine"
    assert found["mad"] < 0.2  # the spread inside each slide is kept
    out = pd.read_parquet(found["output"])
    assert len(out) == 2400 and {"slide", "timepoint", "batch_id", "cell_id", "A_corr"} <= set(out.columns)
    assert sorted(out["batch_id"].unique()) == ["S1_roi_1", "S1_roi_2", "S2_roi_1", "S2_roi_2"]
    s1, s2 = out["slide"] == "S1", out["slide"] == "S2"
    # The slides' C (the same biology everywhere) differed by the stain; after, they meet.
    assert abs(arcsinh_median(out, "C", s1) - arcsinh_median(out, "C", s2)) > 0.4
    assert abs(float(out.loc[s1, "C_corr"].median() - out.loc[s2, "C_corr"].median())) < 0.1
    # Inside a slide the later timepoint keeps its extra A.
    for slide in ("S1", "S2"):
        early = out["batch_id"] == f"{slide}_roi_1"
        late = out["batch_id"] == f"{slide}_roi_2"
        before = arcsinh_median(out, "A", late) - arcsinh_median(out, "A", early)
        after = float(out.loc[late, "A_corr"].median() - out.loc[early, "A_corr"].median())
        assert after == pytest.approx(before, abs=0.1)
    # Studio saves Parquet only: no CSV is written unless the user asks for one.
    assert found["name"].endswith(".parquet") and not list((root / "outputs/batch_corrected").glob("*.csv"))


def test_tab6s_default_mode_also_removes_the_stain(project):
    """SPACEMAP's monotonic mode keeps every cell's order within a slide, and no longer widens the slides (K9)."""
    root, config = project
    found = correction.run(job(root, config, correction_mode="monotonic"), root / "job", lambda *args: None)
    assert found["mad"] < 0.2
    out = pd.read_parquet(found["output"])
    s1, s2 = out["slide"] == "S1", out["slide"] == "S2"
    assert abs(float(out.loc[s1, "C_corr"].median() - out.loc[s2, "C_corr"].median())) < 0.4
    for slide in (s1, s2):  # sorted by A as measured, the corrected A never goes down (capping can only tie)
        ordered = out.loc[slide].dropna(subset=["A_corr"]).sort_values("A")["A_corr"].to_numpy()
        assert (np.diff(ordered) >= -1e-9).all()


def test_the_job_calls_tab6s_corrector_as_classic_does(project):
    """Same filtered cells, same BatchCorrector arguments: the same corrected values as calling it directly."""
    from spacemap.batch_correction_utils import BatchCorrector

    root, config = project
    settings = job(root, config)
    found = correction.run(settings, root / "job", lambda *args: None)
    out = pd.read_parquet(found["output"])
    loaded, with_filter = correction._load(
        settings["tables"], settings["samples"], "batch_id", settings["config"], print
    )
    cleaned, _ = correction._clean(loaded, settings["config"], MARKERS, with_filter)
    direct = BatchCorrector(
        markers=MARKERS,
        som_grid_size=(4, 4),
        som_iterations=2000,
        random_seed=42,
        verbose=False,
        min_cells_per_cluster=10,
        correction_mode="cycombine",
    ).correct_batch_effects(cleaned.rename(columns={"batch_id": "batch"}).assign(batch=cleaned["slide"]))
    merged = out.set_index("cell_id").loc[direct["cell_id"]]
    for marker in MARKERS:
        assert np.allclose(merged[f"{marker}_corr"].to_numpy(), direct[f"{marker}_corr"].to_numpy())
    assert out["A_corr"].isna().sum() == len(out) - len(direct)  # filtered cells keep no corrected value


def test_the_settings_are_shared_with_classic(project):
    root, config = project
    from spacemap_gui.studio.settings import config_path

    config_path(root).parent.mkdir(parents=True, exist_ok=True)
    config_path(root).write_text(json.dumps(config))
    saved = correction.save(root, {"correction_mode": "per_cluster", "batch_unit_col": "slide"}, markers=["A", "B"])
    assert saved["correction_mode"] == "per_cluster" and saved["som_iterations"] == 2000  # earlier settings kept
    stored = json.loads(config_path(root).read_text())["tab_configurations"]["tab6_batch_correction"]
    assert stored["bc_config"]["batch_unit_col"] == "slide" and stored["bc_selected_markers"] == ["A", "B"]
    with pytest.raises(ValueError):
        correction.save(root, {"som_grid_dim": 500})
    with pytest.raises(ValueError):
        correction.save(root, {"tissue_col": "x"})


def test_only_the_ticked_tables_are_corrected_and_the_page_shows_them_corrected(project):
    root, config = project
    plan = correction.plan(root, {}, config)
    assert [row["roi"] for row in plan["rows"]] == ["S1_roi_1", "S1_roi_2", "S2_roi_1", "S2_roi_2"]
    assert {row["slide"] for row in plan["rows"]} == {"S1", "S2"} and not any(row["corrected"] or row["tagged"] for row in plan["rows"])
    with pytest.raises(ValueError, match="at least two"):
        correction.job_settings(root, {}, correction.config_of(config), MARKERS, ["S1_roi_1"])
    settings = correction.job_settings(root, {}, {**correction.config_of(config), "batch_unit_col": "slide"}, MARKERS,
                                       ["S1_roi_1", "S2_roi_1"])
    assert list(settings["tables"]) == ["S1_roi_1", "S2_roi_1"]
    found = correction.run(settings, root / "job", lambda *args: None)
    assert len(pd.read_parquet(found["output"])) == 1200
    after = correction.plan(root, {}, config, found["output"])
    assert [row["roi"] for row in after["rows"] if row["corrected"]] == ["S1_roi_1", "S2_roi_1"] and after["last"]["rois"] == 2


def test_input_folder_is_explicit_even_if_empty_and_accepts_external_parquet(project, tmp_path):
    root, config = project
    external = tmp_path / 'external'
    external.mkdir()
    assert correction.tables(root, {'bc_input_dir': str(external)}) == {}
    source = next(iter(correction.tables(root, {}).values()))
    pd.read_csv(source).to_parquet(external / 'S1_roi_1.ome_cell_features.parquet', index=False)
    plan = correction.plan(root, {'bc_input_dir': str(external)}, config)
    assert plan['tables'] == 1 and plan['rows'][0]['roi'] == 'S1_roi_1'
    assert plan['folders']['input'] == str(external)
    assert 'A' in plan['common']
    stored = correction.save_folders(root, {'input': str(external), 'output': str(tmp_path / 'new_output')})
    assert stored['directory_mappings']['bc_input_dir'] == str(external)
    assert stored['directory_mappings']['bc_output_dir'] == str(tmp_path / 'new_output')
    assert not (tmp_path / 'new_output').exists()  # saving choices does not run or write results


def test_status_tracks_saved_joint_run_before_adoption_and_invalidates_on_input_change(project):
    root, config = project
    found = correction.run(job(root, config), root / 'job', lambda *args: None)
    plan = correction.plan(root, {}, config)
    assert all(row['corrected'] for row in plan['rows'])  # no active table needed
    manifest = correction.last_run(found['output'])
    assert manifest['config']['arcsinh_cofactor'] == 5
    assert manifest['summary']['som']['seed'] == 473
    assert sum(manifest['corrected_per_roi'].values()) == found['corrected_cells']
    path = next(iter(correction.tables(root, {}).values()))
    with open(path, 'a') as handle:
        handle.write('\n')
    stale = correction.plan(root, {}, config)
    assert all(row['status'] == 'stale' and not row['corrected'] for row in stale['rows'])


def test_confounded_biology_is_not_silently_claimed_as_protected(project):
    root, config = project
    settings = job(root, config, covariates=['condition'])
    for roi, row in settings['samples'].items():
        row['condition'] = 'case' if roi.startswith('S1') else 'control'
    with pytest.raises(ValueError, match='confounded'):
        correction.run(settings, root / 'job', lambda *args: None)
    assert not list((root / 'outputs/batch_corrected').glob('*.parquet'))


def test_zero_marker_choice_is_preserved_and_invalid_normalization_rejected(project):
    root, config = project
    correction.save(root, {'norm_method': 'rank', 'som_rlen': 20, 'cycombine_seed': 12}, markers=[])
    from spacemap_gui.studio.settings import read_config
    plan = correction.plan(root, {}, read_config(root))
    assert plan['markers'] == [] and plan['settings']['norm_method'] == 'rank'
    with pytest.raises(ValueError):
        correction.save(root, {'norm_method': 'bogus'})


def test_fast_parquet_twins_are_not_duplicate_inputs(project):
    root, _ = project
    source = next(iter(correction.tables(root, {}).values()))
    pd.read_csv(source).to_parquet(source + '.parquet', index=False)
    assert len(correction.tables(root, {})) == 4


def test_mapped_segmentation_parquet_is_used_without_a_csv_twin(project):
    root, config = project
    folder = root / 'custom_measurements'
    folder.mkdir()
    source = next(iter(correction.tables(root, {}).values()))
    pd.read_csv(source).to_parquet(folder / 'S1_roi_1.ome_cell_features.parquet', index=False)
    plan = correction.plan(root, {'segmentation_output_dir': 'custom_measurements'}, config)
    assert plan['folders']['input'] == str(folder) and plan['tables'] == 1


def test_missing_area_does_not_discard_every_other_cell(project):
    root, config = project
    settings = correction.config_of(config)
    frame = pd.DataFrame({'area': [np.nan, 10, 20, 30], 'A': [1, 2, 3, 4]})
    settings.update(area_min_pct=0, area_max_pct=100, outlier_low=0, outlier_high=100)
    cleaned, _ = correction._clean(frame, settings, ['A'], set())
    assert cleaned.area.tolist() == [10, 20, 30]


def test_slide_clinical_and_roi_metadata_reach_every_output_cell(project):
    root, config = project
    samples.link(root, {}, 'slide,Age,Response\nS1,65,CR\nS2,54,PD\n', 'slide', 'slide', {'Age': 'age', 'Response': 'response'}, apply=True)
    samples.link(root, {}, 'roi,Region\nS1_roi_1,Tumor\nS1_roi_2,Normal\nS2_roi_1,Tumor\nS2_roi_2,Normal\n', 'roi', 'roi', {'Region': 'region'}, apply=True)
    samples.link(root, {}, 'Cell,Review\n1,Checked\n2,Pending\n', 'Cell', 'column:label', {'Review': 'review'}, apply=True)
    found = correction.run(job(root, config), root / 'job', lambda *args: None)
    output = pd.read_parquet(found['output'])
    assert len(output) == 2400
    assert set(output.loc[output.slide == 'S1', 'age']) == {'65'}
    assert set(output.loc[output.slide == 'S2', 'response']) == {'PD'}
    assert set(output.loc[output.batch_id.str.endswith('roi_2'), 'region']) == {'Normal'}
    assert output.loc[output.label == 1, 'review'].tolist() == ['Checked'] * 4
    assert output.loc[output.label == 2, 'review'].tolist() == ['Pending'] * 4
    assert set(output.loc[output.label > 2, 'review']) == {''}
    assert output[['slide', 'age', 'response', 'region']].notna().all().all()


def test_clinical_columns_cannot_replace_marker_values(project):
    root, config = project
    samples.link(root, {}, 'slide,A\nS1,65\nS2,54\n', 'slide', 'slide', {'A': ''}, apply=True)
    with pytest.raises(ValueError, match='Rename these Samples columns'):
        correction.run(job(root, config), root / 'job', lambda *args: None)
