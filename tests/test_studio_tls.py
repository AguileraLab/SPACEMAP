"""TLS candidates: geometry, tissue barriers, scoped tags and safe publication."""
import json

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon, box, mapping

from spacemap import dataio
from spacemap_gui.studio import tls, tasks
from spacemap_gui.studio import tls_review, jobs


def options(**changes):
    return {**tls.DEFAULTS, "phenotype_col": "phenotype_test", "t_labels": ["T"],
            "min_b": 1, "min_t": 1, "min_cells": 2, "max_gap": 4.,
            "min_fraction": 0., "pixel_um": 1., **changes}


def test_boundary_distance_area_and_non_lymphocyte_membership():
    xy = np.array([[0, 0], [1, 0], [.5, .5], [4, 0]])
    regions, tags = tls.detect_region(xy, np.array(["B", "T", "Other", "Other"]), options(), 1)
    assert len(regions) == 1
    assert tags[""].tolist() == [1, 1, 1, 0]
    assert tags["_id"].tolist() == [1, 1, 1, 0]
    assert tags["_size"][0] == pytest.approx(regions[0]["geom"].area)
    assert tags["_size"][3] == 0
    for i in range(3):
        assert tags["_distance"][i] == pytest.approx(-regions[0]["geom"].boundary.distance(shapely.Point(xy[i])))
    assert tags["_distance"][3] == pytest.approx(1.)
    assert regions[0]["cells"] == 3
    assert regions[0]["density_per_mm2"] == pytest.approx(2e6 / regions[0]["area_um2"])


def test_calibration_converts_gap_area_and_distance():
    xy = np.array([[0, 0], [1, 0], [4, 0]])
    regions, tags = tls.detect_region(xy, np.array(["B", "T", "Other"]), options(max_gap=2), .5)
    assert len(regions) == 1
    assert regions[0]["area_um2"] == pytest.approx(regions[0]["geom"].area * .25)
    assert tags["_distance"][2] == pytest.approx(.5)


def test_edge_trim_reduces_margin_without_changing_core_search():
    xy = np.array([[0,0],[1,0],[2.8,0]])
    labels = np.array(["B","T","Other"])
    original, before = tls.detect_region(xy, labels, options(), 1)
    trimmed, after = tls.detect_region(xy, labels, options(edge_trim=.5), 1)
    assert len(original) == len(trimmed) == 1
    assert trimmed[0]["geom"].within(original[0]["geom"])
    assert trimmed[0]["area_um2"] < original[0]["area_um2"]
    assert before[""].tolist() == [1,1,1] and after[""].tolist() == [1,1,0]
    assert after["_distance"][2] > 0 and after["_distance"][0] < 0
    with pytest.raises(ValueError, match="half"):
        tls.check({"max_gap":4, "edge_trim":2})


def test_large_gaps_split_structures_and_ids_are_deterministic():
    xy = np.array([[0, 0], [1, 0], [20, 0], [21, 0], [100, 0]])
    labels = np.array(["B", "T", "B", "T", "Other"])
    regions, tags = tls.detect_region(xy, labels, options(), 1)
    assert [r["id"] for r in regions] == [1, 2]
    assert tags["_id"].tolist() == [1, 1, 2, 2, 0]
    assert np.isnan(tags["_distance"][-1])  # isolated tissue scrap: no clean line
    _, reversed_tags = tls.detect_region(xy[::-1], labels[::-1], options(), 1)
    np.testing.assert_equal(tags["_id"], reversed_tags["_id"][::-1])


def test_ring_keeps_empty_centre_and_does_not_tag_it():
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    xy = np.column_stack((10*np.cos(angles), 10*np.sin(angles)))
    xy = np.vstack([xy, [0, 0]])
    labels = np.array(["B", "T"] * 10 + ["Other"])
    regions, tags = tls.detect_region(xy, labels, options(max_gap=4), 1)
    assert len(regions) == 1 and len(regions[0]["geom"].interiors) == 1
    assert tags[""][-1] == 0 and tags["_id"][-1] == 0
    assert np.isnan(tags["_distance"][-1])


def test_t_cell_chain_does_not_join_distant_b_cores():
    xy = np.array([[0,0],[1,0],[20,0],[21,0], *[[x,0] for x in (3,6,9,12,15,18)]])
    phenotypes = np.array(["B"] * 4 + ["T"] * 6)
    found, tags = tls.detect_region(xy, phenotypes, options(min_b=2, min_cells=3), 1)
    assert len(found) == 2
    assert tags["_id"][:4].tolist() == [1,1,2,2]
    assert tags[""][6:8].tolist() == [0,0]


def test_scattered_b_cells_cannot_borrow_t_cell_connections_to_make_a_core():
    xy = np.array([[0,0],[6,0],[3,0]])
    found, _ = tls.detect_region(xy, np.array(["B","B","T"]), options(min_b=2), 1)
    assert not found


def test_annotations_clip_boundaries_and_block_lines_across_tissue_gaps():
    tissue = Polygon([(0,0),(10,0),(10,10),(8,10),(8,2),(2,2),(2,10),(0,10)])
    xy = np.array([[1,6],[1,7],[9,6],[1,2]])
    regions, tags = tls.detect_region(xy, np.array(["B","T","Other","Other"]), options(), 1, tissue)
    assert len(regions) == 1 and tissue.covers(regions[0]["geom"])
    assert np.isnan(tags["_distance"][2])
    assert tags["_distance"][3] == pytest.approx(2.)


@pytest.mark.parametrize("changes", [{"min_b":2}, {"min_t":2}, {"min_cells":3}, {"min_fraction":.8}, {"min_density":1e6}])
def test_rejects_candidates_failing_each_filter(changes):
    found, tags = tls.detect_region(np.array([[0,0],[1,0],[.5,.5]]), np.array(["B","T","Other"]), options(**changes), 1)
    assert not found and (tags[""] == 0).all() and np.isnan(tags["_distance"]).all()


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    from spacemap_gui.studio import tagging
    monkeypatch.setattr(tagging, "_exports_and_images", lambda root: ({}, {}))
    monkeypatch.setattr(tagging, "annotation_path", lambda *args: (None, False))
    rows = []
    for image in ("one", "two"):
        for region in (1, 2):
            for x, name in [(0,"B"),(1,"T"),(4,"Other")]:
                rows.append(dict(batch_id=image, tissue_roi=region, x_centroid=x, y_centroid=0, phenotype_test=name))
    rows.extend([dict(batch_id="one", tissue_roi=0, x_centroid=0, y_centroid=0, phenotype_test="B"),
                 dict(batch_id="one", tissue_roi=1, x_centroid=np.nan, y_centroid=0, phenotype_test="T"),
                 dict(batch_id="empty", tissue_roi=1, x_centroid=0, y_centroid=0, phenotype_test="Other")])
    frame = pd.DataFrame(rows)
    frame["cell_id"] = [f"c{i}" for i in range(len(frame))]
    path = tmp_path / "combined.parquet"
    frame.to_parquet(path, index=False)
    return tmp_path, path


def test_full_cohort_scoping_local_ids_empty_images_and_cell_tagging_conventions(cohort):
    root, table = cohort
    before = table.read_bytes()
    settings = tls.prepare(root, table, options(), [], False)
    result = tls.run(settings, root / "job", lambda *args: None)
    frame = dataio.load(table)
    assert result["counts"] == {"one":2, "two":2, "empty":0}
    assert result["total"] == 4 and len(result["structures"]) == 4
    assert len({r["key"] for r in result["structures"]}) == 4
    assert all(r["id"] == 1 for r in result["structures"])
    assert frame.loc[frame.TLS.eq(1), "TLS_id"].eq(1).all()
    assert frame.loc[12:13, tls.outputs("TLS")].isna().all().all()
    assert frame.loc[14, "TLS"] == 0 and np.isnan(frame.loc[14, "TLS_distance"])
    assert table.read_bytes() == before
    shapes = json.loads((root / "job/tls_candidates.geojson").read_text())
    assert len(shapes["features"]) == 4
    assert (root / "job/tls_candidates.csv").is_file()


def completed_tls(cohort):
    root, table = cohort
    settings = tls.prepare(root, table, options(), [], False)
    job_id = "a" * 32
    folder = jobs.root(root) / job_id
    result = tls.run(settings, folder, lambda *args: None)
    (folder / "request.json").write_text(json.dumps({"settings":settings}))
    (folder / "status.json").write_text(json.dumps({"id":job_id,"kind":"tls_candidates","status":"done","result":result}))
    return job_id, folder, result


def test_manual_expand_retags_only_its_image_and_region_and_keeps_original(cohort):
    root, table = cohort
    job_id, folder, previous = completed_tls(cohort)
    before = dataio.load(table)
    original = (folder / "tls_candidates.geojson").read_bytes()
    draft = tls_review.context(root, table, job_id, "one", "1")
    draft["features"] = [dict(geometry=mapping(box(-2,-2,6,2)))]
    settings = tls_review.prepare(root, table, draft)
    result = tls.run(settings, root / "review", lambda *args: None)
    after = dataio.load(table)
    assert before.loc[2,"TLS"] == 0 and after.loc[2,"TLS"] == 1
    assert after.loc[:2,"TLS_distance"].lt(0).all()
    pd.testing.assert_frame_equal(before.loc[3:,tls.outputs("TLS")], after.loc[3:,tls.outputs("TLS")])
    assert result["counts"] == previous["counts"] and result["measured"] == previous["measured"]
    reviewed = [s for s in result["structures"] if s.get("reviewed")]
    assert len(reviewed) == 1 and reviewed[0]["cells"] == 3
    assert result["parent_job"] == job_id and result["reviewed"]
    assert (folder / "tls_candidates.geojson").read_bytes() == original
    method = json.loads((root / "review/tls_method.json").read_text())
    assert method["manual_scope"] == {"image":"one","region":"1"}


def test_manual_delete_updates_counts_without_erasing_another_region(cohort):
    root, table = cohort
    job_id, _, _ = completed_tls(cohort)
    draft = tls_review.context(root, table, job_id, "one", "1")
    draft["features"] = []
    result = tls.run(tls_review.prepare(root, table, draft), root / "review", lambda *args: None)
    frame = dataio.load(table)
    assert result["counts"] == {"one":1,"two":2,"empty":0}
    assert frame.loc[:2,["TLS","TLS_id","TLS_size"]].eq(0).all().all()
    assert frame.loc[:2,"TLS_distance"].isna().all()
    assert frame.loc[3,"TLS"] == 1


def test_manual_split_is_a_review_override_with_local_component_ids(cohort):
    root, table = cohort
    job_id, _, _ = completed_tls(cohort)
    draft = tls_review.context(root, table, job_id, "one", "1")
    draft["features"] = [dict(geometry=mapping(box(-.3,-.3,.3,.3))),dict(geometry=mapping(box(.7,-.3,1.3,.3)))]
    result = tls.run(tls_review.prepare(root, table, draft), root / "review", lambda *args: None)
    frame = dataio.load(table)
    assert frame.loc[:1,"TLS_id"].tolist() == [1,2]
    reviewed = [s for s in result["structures"] if s.get("reviewed")]
    assert len(reviewed) == 2 and {s["b_cells"] for s in reviewed} == {0,1}
    assert result["counts"]["one"] == 3


def test_manual_touching_polygons_merge_and_empty_gaps_stay_separate(cohort):
    root, table = cohort
    job_id, _, _ = completed_tls(cohort)
    draft = tls_review.context(root, table, job_id, "one", "1")
    # A far-away polygon has no supporting cells and disappears; touching pieces become one.
    draft["features"] = [dict(geometry=mapping(g)) for g in [box(-2,-2,1,2),box(1,-2,6,2),box(100,0,110,10)]]
    result = tls.run(tls_review.prepare(root, table, draft), root / "review", lambda *args: None)
    assert len([s for s in result["structures"] if s.get("reviewed")]) == 1
    geometries = json.loads((root / "review/tls_candidates.geojson").read_text())["features"]
    reviewed = next(f for f in geometries if f["properties"].get("reviewed"))
    assert shapely.from_geojson(json.dumps(reviewed["geometry"])).bounds[2] <= 6


def test_manual_save_rejects_stale_data_and_unknown_scopes(cohort):
    from spacemap import pending_columns
    root, table = cohort
    job_id, _, _ = completed_tls(cohort)
    draft = tls_review.context(root, table, job_id, "one", "1")
    frame = dataio.load(table)
    pending_columns.save(table, {"phenotype_test":["B"] * len(frame)}, anchor_column="cell_id",anchor_values=frame.cell_id)
    with pytest.raises(ValueError, match="changed while editing"):
        tls_review.prepare(root, table, draft)
    with pytest.raises(ValueError, match="Select a TLS"):
        tls_review.context(root, table, job_id, "two", "999")


def test_manual_publication_failure_restores_all_previous_tags(cohort, monkeypatch):
    from spacemap import pending_columns
    root, table = cohort
    job_id, _, _ = completed_tls(cohort)
    draft = tls_review.context(root, table, job_id, "one", "1")
    draft["features"] = []
    settings = tls_review.prepare(root, table, draft)
    before = dataio.load(table)[tls.outputs("TLS")]
    original_save = pending_columns.save
    def fail(table, columns, **kw):
        original_save(table, {"TLS":columns["TLS"]}, **kw)
        raise OSError("disk full")
    monkeypatch.setattr(pending_columns, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        tls.run(settings, root / "review", lambda *args: None)
    pd.testing.assert_frame_equal(before, dataio.load(table)[tls.outputs("TLS")])


def test_draft_trim_uses_microns_preserves_holes_and_rejects_total_erasure():
    polygon = Polygon([(0,0),(10,0),(10,10),(0,10)], holes=[[(4,4),(6,4),(6,6),(4,6)]])
    features = [dict(geometry=mapping(polygon),properties={"key":"test"})]
    trimmed = tls_review.trim(features, .5, .5)
    geom = shapely.from_geojson(json.dumps(trimmed[0]["geometry"]))
    assert geom.bounds == (1,1,9,9) and len(geom.interiors) == 1
    assert trimmed[0]["properties"]["key"] == "test"
    with pytest.raises(ValueError, match="every boundary"):
        tls_review.trim(features, 20, .5)
    with pytest.raises(ValueError, match="valid polygons"):
        tls_review.polygons([dict(geometry=mapping(shapely.Point(1,1)))])


def test_subset_excludes_other_images_and_whole_column_overwrite_is_explicit(cohort):
    root, table = cohort
    settings = tls.prepare(root, table, options(), ["one"], False)
    tls.run(settings, root / "job", lambda *args: None)
    assert dataio.load(table).loc[6:11, "TLS"].isna().all()
    with pytest.raises(ValueError, match="already exists"):
        tls.prepare(root, table, options(), ["two"], False)
    settings = tls.prepare(root, table, options(), ["two"], True)
    tls.run(settings, root / "job2", lambda *args: None)
    frame = dataio.load(table)
    assert frame.loc[:5, "TLS"].isna().all()
    assert frame.loc[6:11, "TLS"].notna().all()
    assert tasks.overwrite_columns(settings) == set(tls.outputs("TLS"))
    with pytest.raises(ValueError, match="Only the selected"):
        tasks.overwrite_columns({**settings, "overwrite_columns":["phenotype_test"]})


def test_stale_source_does_not_publish(cohort):
    root, table = cohort
    settings = tls.prepare(root, table, options(), [], False)
    frame = pd.read_parquet(table)
    frame.loc[0,"phenotype_test"] = "changed"
    frame.to_parquet(table, index=False)
    with pytest.raises(ValueError, match="changed during this run"):
        tls.run(settings, root / "job", lambda *args: None)
    assert "TLS" not in dataio.columns(table)


def test_failed_column_publication_rolls_back_new_outputs(cohort, monkeypatch):
    from spacemap import pending_columns
    root, table = cohort
    settings = tls.prepare(root, table, options(), [], False)
    real = pending_columns.save
    def fail(table, columns, **kwargs):
        real(table, {"TLS": columns["TLS"]}, **kwargs)
        raise OSError("disk full")
    monkeypatch.setattr(pending_columns, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        tls.run(settings, root / "job", lambda *args: None)
    assert not set(tls.outputs("TLS")) & set(dataio.columns(table))


def test_calibration_and_label_validation(cohort, monkeypatch):
    root, table = cohort
    with pytest.raises(ValueError, match="pixel calibration"):
        tls.prepare(root, table, options(pixel_um=0), [], False)
    with pytest.raises(ValueError, match="different"):
        tls.prepare(root, table, options(b_labels=["T"]), [], False)
    with pytest.raises(ValueError, match="present"):
        tls.prepare(root, table, options(b_labels=["not a phenotype"]), [], False)
    with pytest.raises(ValueError):
        tls.check({"max_gap":float("nan")})
    with pytest.raises(ValueError):
        tls.outputs("phenotype_test")


def test_auto_mpp_uses_each_images_metadata_and_project_fallback(cohort):
    import tifffile
    from spacemap_gui import persistence

    root, table = cohort
    image_dir = root / "outputs/rois"
    image_dir.mkdir(parents=True)
    for name, mpp in [("one", .5), ("two", .75)]:
        tifffile.imwrite(image_dir / (name + ".ome.tiff"), np.zeros((8, 8), dtype=np.uint16),
                         metadata={"axes": "YX", "PhysicalSizeX": mpp, "PhysicalSizeY": mpp})
    persistence.save_project_config(str(root), {"mpp_codex": 1.25})
    found = tls.calibration(root, ["one", "two", "empty"])
    assert {k: v["mpp"] for k, v in found.items()} == {"one": .5, "two": .75, "empty": 1.25}
    assert [v["source"] for v in found.values()] == ["image", "image", "project"]
    settings = tls.prepare(root, table, options(pixel_um=0), [])
    assert settings["pixel_sizes"] == {"one": .5, "two": .75, "empty": 1.25}
    assert settings["pixel_sources"] == found
    manual = tls.prepare(root, table, options(pixel_um=2), [])
    assert set(manual["pixel_sizes"].values()) == {2}
    assert all(v["source"] == "manual" for v in manual["pixel_sources"].values())


def test_auto_mpp_loaded_image_has_priority_without_leaking_to_other_images(cohort):
    import tifffile
    root, _ = cohort
    loaded = root / "loaded.ome.tiff"
    tifffile.imwrite(loaded, np.zeros((8, 8), dtype=np.uint16),
                     metadata={"axes": "YX", "PhysicalSizeX": .42, "PhysicalSizeY": .42})
    found = tls.calibration(root, ["one", "two"], [{"path": str(loaded), "sample": "one"}])
    assert found["one"]["mpp"] == .42 and found["one"]["source"] == "image"
    assert found["two"] == {"mpp": None, "source": "missing"}


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "unknown"])
def test_auto_mpp_ignores_invalid_calibration(cohort, monkeypatch, bad):
    from types import SimpleNamespace
    from spacemap.viewer import sources
    from spacemap_gui.studio import settings
    root, _ = cohort
    monkeypatch.setattr(settings, "read_config", lambda root: {"mpp_codex": bad})
    monkeypatch.setattr(sources, "image_info", lambda path: SimpleNamespace(pixel_size_um=bad))
    assert tls.calibration(root, ["one"], [{"path": "loaded.tiff", "sample": "one"}]) == {
        "one": {"mpp": None, "source": "missing"}}


def test_saved_settings_revision(cohort):
    from spacemap_gui.studio.settings import read_config, revision
    root, _ = cohort
    before = tls.values(read_config(root))
    tls.save(root, {"max_gap":20.}, revision(before))
    assert tls.values(read_config(root))["max_gap"] == 20
    with pytest.raises(ValueError, match="changed in another"):
        tls.save(root, {"max_gap":50.}, revision(before))


def test_cancellation_before_publication_preserves_previous_columns(cohort):
    from spacemap_gui.studio import job_stop
    root, table = cohort
    initial = tls.prepare(root, table, options(), [], False)
    tls.run(initial, root / "job", lambda *args: None)
    before = dataio.load(table)[tls.outputs("TLS")]
    settings = tls.prepare(root, table, options(min_b=100), [], True)
    def report(*args):
        pass
    def cancelled():
        raise job_stop.Cancelled("Stopped by the user")
    report.protect_output = cancelled
    with pytest.raises(job_stop.Cancelled):
        tls.run(settings, root / "job2", report)
    pd.testing.assert_frame_equal(dataio.load(table)[tls.outputs("TLS")], before)
    assert "tls_candidates" in job_stop.INTERRUPTIBLE


def test_annotation_change_invalidates_the_job(cohort, monkeypatch):
    from spacemap_gui.studio import tagging
    root, table = cohort
    path = root / "tissue.geojson"
    feature = {"type":"Feature", "geometry":mapping(box(-10,-10,30,30)), "properties":{"classification":{"name":"tissue"}}}
    path.write_text(json.dumps({"type":"FeatureCollection", "features":[feature]}))
    monkeypatch.setattr(tagging, "annotation_path", lambda *args: (str(path), False))
    settings = tls.prepare(root, table, options(), [], False)
    old = settings["revision"]
    path.write_text(json.dumps({"type":"FeatureCollection", "features":[]}))
    assert tasks.source_revision(settings) != old
    with pytest.raises(ValueError, match="annotations changed"):
        tls.run(settings, root / "job", lambda *args: None)
    assert "TLS" not in dataio.columns(table)


def test_parquet_with_saved_nondefault_index_uses_cell_ids(cohort):
    root, table = cohort
    frame = pd.read_parquet(table)
    frame["x_centroid"] = frame["x_centroid"].astype("Float64")
    frame.index = np.arange(100, 100 + len(frame))
    frame.to_parquet(table)
    result = tls.run(tls.prepare(root, table, options(), [], False), root / "job", lambda *args: None)
    assert result["counts"] == {"one":2, "two":2, "empty":0}
    assert dataio.load(table).loc[100, "TLS"] == 1
    assert pd.isna(dataio.load(table).loc[113, "TLS"])


def test_result_columns_are_not_offered_as_phenotyping_markers():
    from spacemap_gui.studio import markers
    assert [item["name"] for item in markers.panel(["CD20", "TLS", "TLS_id", "TLS_size", "TLS_distance", "TLS_test"], "_corr")] == ["CD20"]


def test_latest_result_only_uses_completed_runs_of_the_current_table(cohort):
    import os
    from spacemap_gui.studio import jobs
    root, table = cohort
    settings = tls.prepare(root, table, options(), [], False)
    result = tls.run(settings, root / "job", lambda *args: None)
    records = [dict(kind="tls_candidates", status="done", result=result),
               dict(kind="tls_candidates", status="done", result={**result, "table":"another table"}),
               dict(kind="tls_candidates", status="failed", result=result)]
    for i, record in enumerate(records):
        folder = jobs.root(root) / f"{i:032x}"
        folder.mkdir(parents=True)
        (folder / "status.json").write_text(json.dumps(record))
        os.utime(folder, (100+i, 100+i))
    assert tls.latest(root, table)["job_id"] == "0" * 32
    assert tls.latest(root, table)["counts"]["one"] == 2
