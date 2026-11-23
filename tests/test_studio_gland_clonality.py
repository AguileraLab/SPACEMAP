"""Scientific invariants and non-destructive publication of gland composition types."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import gland_clonality as g


def fixture():
    rows = []
    for person in range(8):
        for visit in ("pre", "post"):
            for region, size in ((1, 30), (2, 35), (3, 3)):
                for i in range(size + 14):
                    inside = i < size
                    rows.append(dict(cell=len(rows)+1, label=len(rows)+1, image=f"s{person}_{visit}", piece=1,
                        region=region if inside else 0, bed=0 if inside else region, flag=int(not inside),
                        subtype=("A" if (i < size*.85 if region == 1 else i < size*.15) else "B") if inside else None,
                        pheno="Epi" if inside else ("T" if i % 3 else "Stroma"),
                        x=region*100+i, y=person*100+i, person=str(person), visit=visit,
                        response="Yes" if person < 4 else "No", treatment="a" if person % 2 else "b"))
    f = pd.DataFrame(rows)
    c = {**g.DEFAULTS, "image_col": "image", "piece_col": "piece", "cell_id_col": "cell", "label_col": "label",
         "x_col": "x", "y_col": "y", "interior_col": "subtype", "exterior_col": "pheno",
         "interior_classes": ["A", "B"], "exterior_classes": ["T", "Stroma"],
         "regions": [dict(name="Glands", roi_col="region", bed_id_col="bed", bed_col="flag", bed_value="1")], "scope": ["Glands"],
         "interior_k": 2, "exterior_k": 1, "sample_col": "person", "visit_col": "visit", "group_col": "response",
         "stratum_col": "treatment", "baseline": "pre", "followup": "post", "group_a": "Yes", "group_b": "No"}
    return f, c


def test_count_isolation_and_small_structures():
    f, c = fixture()
    result = g.analyse(f, c)
    s = result.structures
    assert len(s) == 48 and s.structure.is_unique
    assert s.n_interior.tolist()[:3] == [30, 35, 3]
    assert s.n_exterior.eq(14).all()
    assert result.models["interior"]["fitted"] == 32
    assert result.models["interior"]["predicted_small"] == 16
    assert s.gland_type.notna().all()
    assert result.assignments.loc[f.region.eq(0), c["output_col"]].isna().all()
    assert result.assignments.loc[f.region.gt(0), c["niche_col"]].isna().all()
    assert result.assignments.loc[f.region.eq(0), c["niche_col"]].notna().all()
    model = result.models["interior"]
    raw = s[["interior_count:A", "interior_count:B"]].to_numpy()
    n = raw.sum(1); fit = n >= c["min_gland"]
    smooth = (raw+c["prior"]*np.array(model["prior"]))/(n+c["prior"])[:, None]
    assert np.allclose(model["mean"], np.average(smooth[fit], axis=0, weights=np.log1p(n[fit])))
    assert np.allclose(np.array(result.profiles["interior"]["raw"]).sum(axis=1), 1)


def test_empty_beds_are_missing_and_orphans_cannot_cross_images():
    f, c = fixture()
    f.loc[f.image.eq("s0_pre") & f.region.eq(0), "bed"] = 99
    result = g.analyse(f, c)
    assert result.structures.loc[result.structures.image.eq("s0_pre"), "niche"].isna().all()
    assert any("no matching gland" in n for n in result.notes)
    c["exterior_classes"] = []
    result = g.analyse(f, c)
    assert result.models["exterior"] is None and result.assignments[c["niche_col"]].isna().all()


def test_ids_include_tissue_piece_and_zero_pieces_are_excluded():
    f, c = fixture()
    other = f[f.image.eq("s0_pre")].copy(); other["piece"] = 2; other["cell"] += 100000
    combined = pd.concat([f, other], ignore_index=True)
    assert len(g.analyse(combined, c).structures) == 51
    other["piece"] = 0
    assert len(g.analyse(pd.concat([f, other], ignore_index=True), c).structures) == 48


def test_auto_k_reproducible_and_invalid_fixed_k_rejected():
    f, c = fixture()
    with pytest.raises(ValueError, match="distinct well-sampled"):
        g.analyse(f, {**c, "interior_k": 20})
    a = g.analyse(f, {**c, "auto_k": True})
    b = g.analyse(f, {**c, "auto_k": True})
    assert a.models["interior"]["k"] == 2
    pd.testing.assert_series_equal(a.structures.gland_type, b.structures.gland_type)


def test_overlap_and_metadata_conflicts_rejected():
    f, c = fixture()
    f["other"] = f.region
    c["regions"] = [*c["regions"], dict(name="Other", roi_col="other")]
    with pytest.raises(ValueError, match="several mapped"):
        g.analyse(f, c)
    f, c = fixture(); f.loc[0, "person"] = "mismatch"
    with pytest.raises(ValueError, match="changes within a gland"):
        g.analyse(f, c)


def test_patient_units_visits_complete_pairs_and_fdr():
    f, c = fixture()
    s = g.analyse(f, c).structures
    s = s[~(s["sample"].eq("0") & s.visit.eq("post"))]
    out = g.clinical(s, c, strata=True)
    pooled = [r for r in out["tests"] if r["view"] == "Pooled visits" and r["stratum"] == "All"]
    assert all((r["n_a"], r["n_b"]) == (4, 4) for r in pooled)
    delta = [r for r in out["tests"] if r["view"].startswith("Δ") and r["stratum"] == "All"]
    assert all((r["n_a"], r["n_b"]) == (3, 4) for r in delta)
    assert all(r["q"] >= r["p"] for r in out["tests"] if r["p"] is not None)
    assert all(r["p"] is None for r in out["tests"] if r["stratum"] != "All")
    vals = pd.DataFrame(out["values"])
    assert np.allclose(vals.groupby(["sample", "view", "stratum", "kind"]).value.sum().loc[lambda x: x > 0], 100)


def test_job_publication_preserves_source_and_other_images(tmp_path, monkeypatch):
    from spacemap import dataio
    from spacemap_gui.studio import tasks
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    f, c = fixture(); table = tmp_path / "cells.parquet"; f.to_parquet(table, index=False)
    original = table.read_bytes(); folder = tmp_path / "job"; folder.mkdir()
    cfg = g.prepare(tmp_path, table, c, ["s0_pre", "s1_pre"])
    result = g.run(cfg, folder, lambda *a: None)
    assert table.read_bytes() == original
    joined = dataio.load(table)
    outside = ~f.image.isin(["s0_pre", "s1_pre"])
    assert joined.loc[outside, c["output_col"]].isna().all()
    assert joined.loc[~outside & f.region.gt(0), c["output_col"]].notna().all()
    assert Path(result["folder"], "gland_models.json").is_file()
    assert (folder / "interior_profiles.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="already exists"):
        g.prepare(tmp_path, table, c, ["s0_pre"])
    diagnose = g.prepare(tmp_path, table, c, ["s0_pre"], diagnose=True)
    before = tasks.source_revision(diagnose)
    g.run(diagnose, folder, lambda *a: None)
    assert tasks.source_revision(diagnose) == before
    assert table.read_bytes() == original


def test_output_names_cannot_alias_inputs():
    _, c = fixture()
    with pytest.raises(ValueError, match="input column"):
        g.check({**c, "output_col": "region"})


def test_k_review_recommends_saved_scores_with_boundary_and_missing_cases():
    model = {"k": 8, "diagnostics": [{"k": 4, "silhouette": .18}, {"k": 2, "silhouette": .211}, {"k": 8, "silhouette": .214}], "silhouette_n": 2000}
    review = g.k_review(model)
    assert review["suggested_k"] == 8 and review["score"] == .214
    assert [r["k"] for r in review["rows"]] == [2, 4, 8]
    assert "largest K tested" in review["notes"][0]
    # A manually fitted K or an undefined score does not establish a winner.
    assert g.k_review({"diagnostics": [{"k": 4, "silhouette": .3}]})["suggested_k"] is None
    assert g.k_review({"diagnostics": [{"k": 1, "silhouette": None}]})["suggested_k"] is None
    assert g.k_review(None)["suggested_k"] is None
    model["diagnostics"][1]["silhouette"] = .4
    assert g.k_review(model)["suggested_k"] == 2
    assert g.k_review(model)["notes"] == []
    for m in (model, None, {"diagnostics": [{"k": 2, "silhouette": -.3}, {"k": 3, "silhouette": -.2}]}):
        png = g.k_review_figure(m, "exterior")
        assert png.startswith(b"\x89PNG") and len(png) > 1000


def test_separate_image_tables_keep_local_ids_and_source_files(tmp_path, monkeypatch):
    from spacemap import dataio, table_collection
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    f, c = fixture()
    entries, originals = [], {}
    for image in ("s0_pre", "s1_pre"):
        path = tmp_path / (image + ".parquet")
        one = f[f.image.eq(image)].drop(columns=["image"]).copy()
        one["cell"] = np.arange(len(one)); one["label"] = np.arange(1, len(one)+1)
        one.to_parquet(path, index=False)
        entries.append(dict(image=image, path=str(path), anchor="cell"))
        originals[path] = path.read_bytes()
    table = tmp_path / ("images" + table_collection.SUFFIX)
    table.write_text(json.dumps(dict(format="spacemap-image-tables-v1", tables=entries, unpaired=[])))
    c.update(image_col="batch_id", cell_id_col="cell_id")
    cfg = g.prepare(tmp_path, table, c, [r["image"] for r in entries])
    folder = tmp_path / "job"; folder.mkdir()
    result = g.run(cfg, folder, lambda *a: None)
    assert result["glands"] == 6
    for path, original in originals.items():
        assert path.read_bytes() == original
        joined = dataio.load(path)
        assert joined.loc[joined.region.gt(0), c["output_col"]].notna().all()
    links = pd.read_parquet(folder / "gland_cell_links.parquet")
    assert links.cell_id.is_unique
    assert len(links) == sum(len(dataio.load(p)) for p in originals)


def test_stale_inputs_and_failed_publication_do_not_leave_partial_columns(tmp_path, monkeypatch):
    from spacemap import dataio, pending_columns
    from spacemap_gui.studio import tasks
    f, c = fixture(); table = tmp_path / "cells.parquet"; f.to_parquet(table, index=False)
    cfg = g.prepare(tmp_path, table, c, ["s0_pre"])
    changed = f.copy(); changed.loc[0, "subtype"] = "B"; changed.to_parquet(table, index=False)
    with pytest.raises(ValueError, match="changed during"):
        tasks.save_result(cfg, {c["output_col"]: ["Type"]}, f.cell.iloc[:1], lambda *a: None)
    cfg = g.prepare(tmp_path, table, c, ["s0_pre"])
    original_save = pending_columns.save
    def failing(path, columns, **kwargs):
        original_save(path, {c["output_col"]: columns[c["output_col"]]}, **kwargs)
        raise OSError("simulated second-column write failure")
    monkeypatch.setattr(pending_columns, "save", failing)
    with pytest.raises(OSError, match="simulated"):
        tasks.save_result(cfg, {c["output_col"]: ["Type"], c["niche_col"]: [None]}, f.cell.iloc[:1], lambda *a: None)
    assert not {c["output_col"], c["niche_col"]} & set(dataio.columns(table))


def test_settings_roundtrip_and_conflicting_save(tmp_path):
    f, c = fixture(); path = tmp_path / "cells.parquet"; f.to_parquet(path, index=False)
    before = g.settings(tmp_path, path)
    saved = g.settings(tmp_path, path, c, before["revision"])
    assert g.settings(tmp_path, path)["values"] == c
    with pytest.raises(ValueError, match="another window"):
        g.settings(tmp_path, path, c, before["revision"])
    assert saved["revision"] != before["revision"]


def test_type_selection_stays_in_requested_image_and_uses_exact_cells(tmp_path, monkeypatch):
    from spacemap_gui.studio import flow_gating
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    f, c = fixture()
    # One gland type is absent in this image, but present in the rest of the run.
    f = f[~(f.image.eq("s0_pre") & ((f.region > 1) | (f.bed > 1)))].copy()
    table = tmp_path / "cells.parquet"; f.to_parquet(table, index=False)
    folder = tmp_path / "job"; folder.mkdir()
    result = g.run(g.prepare(tmp_path, table, c, list(f.image.unique())), folder, lambda *a: None)
    monkeypatch.setattr(g, "saved_result", lambda *args: (result, folder))
    s = pd.read_parquet(folder / "gland_structures.parquet")
    name = s.loc[s.image.eq("s0_pre"), "gland_type"].iloc[0]
    shown = g.select_type_cells(tmp_path, table, "run", "s0_pre", "interior", name, "test")
    assert shown["image"] == "s0_pre" and shown["colour_by"] == c["output_col"]
    assert shown["selection"]["count"] == 30 and shown["selection"]["by_image"] == {"s0_pre": 30}
    assert shown["selection"]["colour"] == result["palette"][name]
    members = flow_gating.token_item(shown["selection"]["token"])["frame"]
    assert set(members._flow_id) == set(f.loc[f.image.eq("s0_pre") & f.region.gt(0), "label"])
    from spacemap.viewer import cells as cell_tiles
    image_labels = np.array([[members._flow_id.iloc[0], 0], [members._flow_id.iloc[1], 0]], dtype=np.int64)
    tile_labels, lut = flow_gating.tile(shown["selection"]["token"], str(table), "s0_pre", image_labels)
    rgba = cell_tiles.render_cell_tile(tile_labels, lut, mode="fill")
    expected = [int(result["palette"][name][i:i+2], 16) for i in (1, 3, 5)]
    assert rgba[0, 0, :3].tolist() == expected and rgba[0, 0, 3] > 0
    assert rgba[0, 1, 3] == 0  # Unselected pixels remain transparent.
    assert shown["bounds"][2] < 100 and shown["structures"] == 1
    absent = next(n for n in result["profiles"]["interior"]["rows"] if n != name)
    missing = g.select_type_cells(tmp_path, table, "run", "s0_pre", "interior", absent, "test")
    assert missing["missing"] and "not present" in missing["message"] and "selection" not in missing
    excluded = g.select_type_cells(tmp_path, table, "run", "another-image", "interior", name, "test")
    assert excluded["missing"] and "not included" in excluded["message"]
    niche = s.loc[s.image.eq("s0_pre"), "niche"].iloc[0]
    bed = g.select_type_cells(tmp_path, table, "run", "s0_pre", "exterior", niche, "test")
    assert bed["selection"]["count"] == 14 and bed["colour_by"] == c["niche_col"]
    assert set(flow_gating.token_item(bed["selection"]["token"])["frame"]._flow_id) == set(f.loc[f.image.eq("s0_pre") & f.bed.gt(0), "label"])
    result["diagnose"] = True
    with pytest.raises(ValueError, match="K review"):
        g.select_type_cells(tmp_path, table, "run", "s0_pre", "exterior", niche, "test")


def test_missing_optional_strata_remain_available_for_pooled_comparison():
    f, c = fixture(); f.loc[f.person.eq("0"), "treatment"] = None
    result = g.analyse(f, c)
    compared = g.clinical(result.structures, c, strata=True)
    assert json.dumps(compared, allow_nan=False)
    assert {r["sample"] for r in compared["values"] if r["stratum"] == "All"} == set(map(str, range(8)))


def test_smoothing_cannot_create_fit_variation_from_a_tiny_gland_only_subtype():
    _, c = fixture()
    counts = np.array([[20, 0], [40, 0], [1, 1]])
    for standardize in (True, False):
        _, _, model, profile = g.fit_composition(counts, ["A", "B"], {**c, "auto_k": True, "standardize": standardize}, "interior")
        assert model["k"] == 1 and model["active"] == [False, False]
        assert np.allclose(profile["z"], 0)


def test_numeric_image_identity_is_preserved_and_boolean_bed_flags_work():
    f, c = fixture(); f = f[f.image.eq("s0_pre")].copy()
    f["image"] = 1.0; f["flag"] = f.flag.astype(bool)
    result = g.analyse(f, c)
    assert set(result.structures.image) == {"1.0"}
    assert result.structures.n_exterior.eq(14).all()


def test_individual_silhouettes_match_saved_mean_and_exclude_small_predictions():
    from sklearn.metrics import silhouette_samples
    f, c = fixture(); result = g.analyse(f, {**c, "seed": 77})
    before = result.structures.copy(deep=True)
    data = g.silhouette_details(result.structures, result.models["interior"], "interior")
    assert data["state"] == "ok" and data["scored"] == data["fitted"] == 32
    assert data["predicted_small"] == 16
    assert data["mean"] == pytest.approx(result.models["interior"]["silhouette"])
    assert sum(row["scored"] for row in data["groups"]) == 32
    assert {p["structure"] for p in data["points"]} == set(before.loc[before.interior_status.eq("Fit"), "structure"])
    # Reverse one saved assignment to exercise a poorly fitting structure.
    altered = before.copy(); names = altered.gland_type.unique()
    altered.loc[0, "gland_type"] = next(n for n in names if n != altered.loc[0, "gland_type"])
    scored = g.silhouette_details(altered, result.models["interior"], "interior")
    assert sum(row["negative"] for row in scored["groups"]) >= 1
    assert next(p for p in scored["points"] if p["structure"] == altered.loc[0, "structure"])["silhouette"] < 0
    # For two-feature compositions, positive linear scaling leaves distances proportional.
    fit = altered[altered.interior_status.eq("Fit")].set_index("structure")
    ordered = fit.loc[[p["structure"] for p in scored["points"]]]
    counts = ordered[["interior_count:A", "interior_count:B"]].to_numpy()
    n = counts.sum(1); m = result.models["interior"]
    x = (counts+c["prior"]*np.array(m["prior"]))/(n+c["prior"])[:, None]
    x = np.clip((x-m["mean"])/m["scale"], -m["clip"], m["clip"])
    expected = silhouette_samples(x, ordered.gland_type)
    assert np.allclose([p["silhouette"] for p in scored["points"]], expected)
    pd.testing.assert_frame_equal(before, result.structures)
    assert g.silhouette_figure(data).startswith(b"\x89PNG")
    csv = g.silhouette_csv(data)
    assert "reference_structures" in csv and "selected_cells" in csv
    missing = g.silhouette_details(before, result.models["exterior"], "exterior")
    assert missing["state"] == "unavailable" and missing["mean"] is None and "K=1" in missing["message"]
    assert g.silhouette_details(before, None, "exterior")["points"] == []


@pytest.mark.parametrize("standardize", [True, False])
def test_individual_scores_use_same_bounded_sample_as_k_review(standardize):
    _, c = fixture(); c = {**c, "seed": 123, "standardize": standardize, "interior_k": 3}
    counts = np.random.default_rng(9).multinomial(40, [.4, .4, .2], size=2100)
    assigned, status, model, _ = g.fit_composition(counts, ["a", "b", "c"], c, "interior")
    s = pd.DataFrame({"structure": [f"s{i}" for i in range(len(counts))], "image": "image",
                      "gland_type": assigned, "interior_status": status})
    for j, label in enumerate(model["features"]): s["interior_count:"+label] = counts[:, j]
    data = g.silhouette_details(s, model, "interior")
    assert data["scored"] == 2000 and data["fitted"] == 2100
    assert data["mean"] == pytest.approx(model["silhouette"])
    assert len({p["structure"] for p in data["points"]}) == 2000
    assert json.dumps(data, allow_nan=False)


def test_legacy_scores_use_saved_artifacts_without_refitting(tmp_path, monkeypatch):
    from sklearn.cluster import KMeans
    f, c = fixture(); result = g.analyse(f, c)
    g.export_analysis(result, tmp_path)
    expected = json.loads((tmp_path / "interior_silhouette.json").read_text())
    (tmp_path / "interior_silhouette.json").unlink()  # pre-feature saved run
    monkeypatch.setattr(g, "saved_result", lambda *a: ({}, tmp_path))
    monkeypatch.setattr(KMeans, "fit", lambda *a, **k: pytest.fail("Must not refit"))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    actual = g.saved_silhouette(tmp_path, tmp_path / "missing_original.parquet", "old", "interior")
    assert actual == expected
    assert g.saved_silhouette(tmp_path, "", "old", "interior") is actual
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="interior"):
        g.saved_silhouette(tmp_path, "", "old", "../outside")
