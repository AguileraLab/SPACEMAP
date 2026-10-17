"""Exact geometry, persistent history and conflicting edits in Studio Annotation."""

import json

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box, mapping, shape
from shapely.ops import unary_union

from spacemap_gui.studio import annotation, annotation_edits as edits


def feature(geometry, **properties):
    return {"type": "Feature", "properties": properties, "geometry": mapping(geometry)}


def write(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def geometry(features):
    return unary_union([shape(f["geometry"]) for f in features])


def test_open_is_read_only_and_components_have_stable_ids(tmp_path):
    path = tmp_path / "tissue.geojson"
    write(path, [feature(unary_union([box(0, 0, 100, 100), box(200, 0, 300, 100)]))])
    original = path.read_bytes(), path.stat().st_mtime_ns
    first, version = edits.read(path, kind="tissue", label="sample")
    second, _ = edits.read(path, kind="tissue", label="sample")
    assert first == second and len(first) == 2
    assert [f["properties"]["name"] for f in first] == ["sample_1", "sample_2"]
    assert all(f["properties"]["classification"]["name"] == "Valid_Tissue" for f in first)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert not edits.history(path)["entries"]
    assert len(list(tmp_path.iterdir())) == 1
    assert version == edits.revision(path)


def test_split_width_and_merge_gap_preserve_holes_and_precision(tmp_path):
    original = unary_union([box(0, 0, 100, 100), box(90, 45, 210, 55), box(200, 0, 300, 100)]).difference(box(20.123, 20.456, 30.789, 30.987))
    parts = edits.edit_geometry([mapping(original)], "split", line=[[150, 30], [150, 70]])
    assert len(parts) == 2
    a, b = map(shape, parts)
    assert a.distance(b) == pytest.approx(2)
    assert a.area + b.area == pytest.approx(original.area - 20)
    assert sum(len(p.interiors) for p in (a, b)) == 1
    with pytest.raises(ValueError, match="apart"):
        edits.edit_geometry(parts, "merge", gap=1)
    joined = shape(edits.edit_geometry(parts, "merge", gap=2)[0])
    assert len(joined.interiors) == 1
    assert Polygon(joined.interiors[0]).equals(box(20.123, 20.456, 30.789, 30.987))
    assert joined.covers(a) and joined.covers(b)
    assert not joined.equals(original)  # Only a narrow bridge is added, not the entire gap.
    path = tmp_path / 'tissue.geojson'
    write(path, [feature(original, id='t1')])
    version = edits.revision(path)
    result = edits.edit(path, ['t1'], 'split', version, line=[[150, 30], [150, 70]], kind='tissue')
    assert len(result['features']) == 2
    with pytest.raises(ValueError, match="another window"):
        edits.edit(path, result['selected'], 'merge', version, gap=3)
    assert geometry(edits.restore(path, version, result['revision'], kind='tissue')['features']).equals(original)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '2'])
def test_polygon_edit_rejects_invalid_width_and_gap(value):
    with pytest.raises(ValueError):
        edits.edit_geometry([mapping(box(0,0,20,20))], 'split', line=[[10,-1],[10,21]], width=value)
    with pytest.raises(ValueError):
        edits.edit_geometry([mapping(box(0,0,20,20)), mapping(box(20,0,40,20))], 'merge', gap=value)


def test_approved_neck_changes_only_cut_area_and_undo_restores(tmp_path):
    original = unary_union([box(0, 0, 100, 100), box(90, 45, 210, 55), box(200, 0, 300, 100)])
    other = box(500, 500, 550, 550)
    path = tmp_path / 'tissue.geojson'
    write(path, [feature(original, id='t1'), feature(other, id='other', name='Keep this', named=True)])
    version = edits.revision(path)
    blade = box(145, 40, 155, 60)
    result = edits.approve_neck(path, mapping(blade), version)
    assert len(result['features']) == 3
    assert geometry(result['features']).equals(unary_union([original.difference(blade), other]))
    assert next(f for f in result['features'] if f['properties']['id'] == 'other')['properties']['name'] == 'Keep this'
    restored = edits.restore(path, version, result['revision'], kind='tissue')
    assert geometry(restored['features']).equals(unary_union([original, other]))


def test_brush_cuts_neck_preserves_holes_and_restores_exactly(tmp_path):
    path = tmp_path / "tissue.geojson"
    original = unary_union([box(0, 0, 100, 100), box(90, 45, 210, 55), box(200, 0, 300, 100)]).difference(
        box(20, 20, 30, 30)
    )
    write(path, [feature(original, id="t1")])
    version = edits.revision(path)
    erased = edits.brush(path, "t1", [[150, 30], [150, 70]], 3, "erase", version, kind="tissue")
    full, after = edits.read(path)
    assert len(full) == 2 and len(erased["selected"]) == 2
    assert sum(len(shape(f["geometry"]).interiors) for f in full) == 1
    restored = edits.restore(path, version, after, kind="tissue")
    assert geometry(restored["features"]).equals(original)
    # Restoration preserves the cut version as well.
    again = edits.restore(path, erased["revision"], restored["revision"], kind="tissue")
    assert geometry(again["features"]).equals(geometry(full))
    assert len(edits.history(path)["entries"]) == 3


def test_simplified_rename_and_undo_use_original_full_geometry(tmp_path):
    path = tmp_path / "tissue.geojson"
    # Over 4000 vertices, including non-circular fine detail that cannot be reconstructed from the screen copy.
    theta = np.linspace(0, 2 * np.pi, 12000)
    radius = 100 + np.sin(theta * 201)
    original = Polygon(np.c_[200 + radius * np.cos(theta), 200 + radius * np.sin(theta)])
    write(path, [feature(original, id="t1")])
    full, version = edits.read(path)
    shown = edits.display(full, version)["features"]
    assert shown[0]["properties"]["simplified"]
    shown[0]["properties"]["name"] = "Tumor"
    named = edits.save(path, shown, version)
    assert geometry(edits.read(path)[0]).equals(original)
    erased = edits.brush(path, "t1", [[200, 50], [200, 350]], 2, "erase", named["revision"])
    # Browser Undo holds the simplified pre-stroke shape, tagged with its earlier version.
    edits.save(path, named["features"], erased["revision"])
    assert geometry(edits.read(path)[0]).equals(original)


def test_tissue_display_keeps_detail_when_holes_exceed_the_point_budget():
    import shapely
    from spacemap.viewer import shapes

    # The holes alone require more than 4000 points. Increasing simplification
    # indefinitely cannot meet that budget and used to deform the boundary.
    holes = [list(box(10 + x * 20, 10 + y * 20, 20 + x * 20, 20 + y * 20).exterior.coords)
             for x in range(40) for y in range(30)]
    original = Polygon([(0, 0), (850, 0), (850, 300), (830, 300), (830, 650), (0, 650)], holes)
    source = feature(original, id="t1", name="tissue_1")
    before = json.dumps(source)
    preview = edits.display_features([source])
    saved_view = edits.display([source], "a" * 64)["features"]
    for shown in (preview, saved_view):
        outline = shape(shown[0]["geometry"])
        assert outline.equals(original), "A point budget must not change these exact tissue edges or holes"
        assert len(outline.interiors) == len(holes)
        assert shapely.get_num_coordinates(outline) > shapes.MAX_DISPLAY_VERTICES
    assert json.dumps(source) == before


def test_fill_small_holes_uses_physical_area_and_keeps_large_holes_and_gaps():
    outer = box(0, 0, 100, 100).difference(box(0, 40, 5, 60))
    small, large = box(10, 10, 12, 12), box(30, 30, 40, 40)
    tissue = outer.difference(small).difference(large)
    separate = box(150, 0, 180, 30)
    original = [feature(tissue), feature(separate)]
    before = json.dumps(original)
    # At 0.5 µm/px the small hole is exactly 1 µm²; the large one is 25 µm².
    for limit in (0, 1):
        unchanged, count = annotation.fill_small_holes(original, limit, 0.5)
        assert unchanged is original and count == 0
    filled, count = annotation.fill_small_holes(original, 2, 0.5)
    assert count == 1
    assert geometry(filled).equals(unary_union([outer.difference(large), separate]))
    assert len(edits.polygons(geometry(filled))) == 2
    assert json.dumps(original) == before
    # An island inside the filled hole is absorbed, never saved twice.
    island = feature(box(10.25, 10.25, 11, 11))
    with_island, count = annotation.fill_small_holes(original + [island], 2, 0.5)
    assert count == 1 and geometry(with_island).equals(geometry(filled))


def test_old_window_and_queued_jobs_cannot_overwrite_new_work(tmp_path):
    path = tmp_path / "tissue.geojson"
    write(path, [feature(box(0, 0, 100, 100), id="t1")])
    before = edits.revision(path)
    edited = edits.brush(path, "t1", [[90, 40], [120, 40]], 8, "add", before)
    expected = path.read_bytes()
    for work in [
        lambda: edits.replace(path, [], before, "Detect"),
        lambda: edits.save(path, [], before),
        lambda: edits.restore(path, before, before),
    ]:
        with pytest.raises(ValueError, match="another window, tab or job"):
            work()
        assert path.read_bytes() == expected
    assert edited["revision"] != before


def test_brush_edges_add_erase_everything_and_invalid_numbers():
    original = box(0, 0, 20, 20)
    added = edits.brush_geometry(mapping(original), [[19, 19], [24, 24]], 4, "add", (25, 25))
    assert geometry([{"geometry": p} for p in added]).bounds == (0, 0, 25, 25)
    assert edits.brush_geometry(mapping(original), [[10, 10]], 30, "erase") == []
    with pytest.raises(ValueError, match="selected polygon"):
        edits.brush_geometry(mapping(original), [[100, 100]], 2, "add")
    for radius in [float("nan"), float("inf"), 0, -1, True]:
        with pytest.raises(ValueError):
            edits.brush_geometry(mapping(original), [[10, 10]], radius, "add")


def test_subtract_reviewed_artifacts_is_repeatable_and_has_history(tmp_path):
    tissue, artifact = tmp_path / "tissue.geojson", tmp_path / "artifacts.geojson"
    write(tissue, [feature(box(0, 0, 100, 100))])
    write(artifact, [feature(box(45, 0, 55, 100))])
    original = edits.revision(tissue)
    first = edits.subtract(tissue, [artifact], original)
    assert len(first["features"]) == 2 and geometry(first["features"]).area == 9000
    second = edits.subtract(tissue, [artifact], first["revision"])
    assert geometry(first["features"]).equals(geometry(second["features"]))
    assert edits.history(tissue)["entries"][0]["label"] == "Subtracted artifacts"


def test_artifact_detection_keeps_tissue_and_drawings(tmp_path, monkeypatch):
    from spacemap_gui.studio import artifacts

    output, sample = tmp_path / "out", "image"
    tissue = annotation.valid_tissue(output, sample)
    drawn = annotation.drawn_artifacts(output, sample)
    write(tissue, [feature(box(0, 0, 100, 100))])
    write(drawn, [feature(box(0, 0, 5, 5), source="manual", id="manual")])
    initial = tissue.read_bytes()

    def detect(image, config, temporary, sample, report):
        path = annotation.clot(temporary, sample)
        write(path, [feature(box(50, 50, 55, 55))])
        return {"written": [str(path)]}

    monkeypatch.setattr(artifacts, "apply", detect)
    artifacts.detect_editable("image", {}, output, sample, edits.revision(drawn))
    assert tissue.read_bytes() == initial
    found, _ = edits.read(drawn)
    assert len(found) == 2 and {f["properties"]["source"] for f in found} == {"manual", "auto_artifact"}


def test_pieces_uses_new_manual_tissue_instead_of_stale_review(tmp_path):
    import os
    from spacemap_gui.studio import pieces

    reviewed = pieces.review(tmp_path, "test")
    tissue = annotation.valid_tissue(tmp_path, "test")
    write(reviewed, [feature(box(0, 0, 100, 100), classification={"name": "Valid_Tissue"})])
    edits.replace(
        tissue, [feature(box(0, 0, 50, 100))], edits.revision(tissue), "Edited by hand", kind="tissue", edited=True
    )
    os.utime(tissue, ns=(reviewed.stat().st_atime_ns, reviewed.stat().st_mtime_ns))
    geom, source = pieces.source(tmp_path, "test")
    assert source == tissue and geom.area == 5000
    edits.replace(tissue, [], edits.revision(tissue), "Edited by hand", kind="tissue", edited=True)
    with pytest.raises(ValueError, match="no tissue"):
        pieces.source(tmp_path, "test")


def test_history_rejects_paths(tmp_path):
    path = tmp_path / "tissue.geojson"
    with pytest.raises(ValueError, match="saved version"):
        edits.restore(path, "../../secret", edits.revision(path))


def test_tissue_preview_counts_polygons_in_repaired_collections(tmp_path, monkeypatch):
    from shapely.geometry import GeometryCollection, LineString, MultiPolygon
    from spacemap import tissue_annotation_utils as classic

    donut = box(0, 0, 100, 100).difference(box(20, 20, 80, 80))
    collection = GeometryCollection(
        [donut, MultiPolygon([box(200, 0, 300, 100), box(400, 0, 500, 100)]), LineString([(0, 0), (20, 0)])]
    )
    monkeypatch.setattr(classic, "PyramidImageReader", lambda path: type("Reader", (), {"channel_names": ["DAPI"]})())
    monkeypatch.setattr(
        classic,
        "detect_tissue_mask",
        lambda *a, **k: {"tissue_mask": np.ones((10, 10), bool), "stats": {"tissue_percent": 6.57}},
    )
    monkeypatch.setattr(classic, "mask_to_geojson_features", lambda *a, **k: [feature(collection)])
    found = annotation.detect("unused", {}, {**annotation.settings({}), "min_tissue_area_um2": 0})
    assert found["stats"]["tissue_percent"] > 0 and len(found["features"]) == 3
    assert len(annotation.rings(found["features"])) == 3
    assert len(annotation.rings(found["features"], include_holes=True)) == 4
    assert len(annotation.rings([feature(collection)])) == 3
    assert geometry(found["features"]).equals(unary_union([donut, box(200, 0, 300, 100), box(400, 0, 500, 100)]))


def test_a_late_save_of_this_page_is_not_mistaken_for_another_window(tmp_path):
    """The page stopped waiting for save A (slow server) but it landed; its next save B names A and goes through.
    A change made by anything else is still refused."""
    from shapely.geometry import box, mapping
    path = tmp_path / "sample_clot_reviewed.geojson"
    feature = lambda x: {"type": "Feature", "geometry": mapping(box(x, 0, x + 5, 5)), "properties": {}}
    start = edits.revision(path)
    edits.save(path, [feature(0)], start, kind="artifacts", save_id="aaaa1111")
    assert edits.last_save_id(path) == "aaaa1111"
    edits.save(path, [feature(10)], start, kind="artifacts", save_id="bbbb2222", after_save="aaaa1111")  # page still holds *start*
    assert len(edits.read(path, kind="artifacts")[0]) == 1 and edits.last_save_id(path) == "bbbb2222"
    other = edits.revision(path)
    edits.replace(path, [feature(20)], other, "Found for the ticked images", kind="artifacts")  # a job, not this page
    assert edits.last_save_id(path) is None
    with pytest.raises(ValueError, match="changed at"):
        edits.save(path, [feature(30)], other, kind="artifacts", save_id="cccc3333", after_save="bbbb2222")


def test_heavy_predictions_read_once_shown_light_and_saved_whole(tmp_path):
    """A clot with 300 one-pixel holes: read once per version (callers get their own copies), shown without the
    invisible holes (marked simplified), and saved with every hole."""
    from shapely.geometry import Polygon
    holes = [[(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)] for x in range(10, 190, 6) for y in range(10, 60, 5)]
    clot = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)], holes)
    path = tmp_path / "s_clot_reviewed.geojson"
    import spacemap.viewer.shapes as shapes_module
    original = shapes_module.MAX_DISPLAY_VERTICES
    shapes_module.MAX_DISPLAY_VERTICES = 100  # a small budget, so this test polygon counts as heavy
    try:
        _heavy(path, clot, holes)
    finally:
        shapes_module.MAX_DISPLAY_VERTICES = original


def _heavy(path, clot, holes):
    edits.replace(path, [{"type": "Feature", "geometry": mapping(clot), "properties": {}}], edits.revision(path), "found", kind="artifacts")
    first, version = edits.read(path, kind="artifacts")
    first[0]["properties"]["name"] = "changed here only"
    again, same = edits.read(path, kind="artifacts")
    assert same == version and again[0]["properties"].get("name") != "changed here only"
    assert len(shape(again[0]["geometry"]).interiors) == len(holes)
    shown = edits.display(again, same)["features"][0]
    assert shown["properties"]["simplified"] and len(shape(shown["geometry"]).interiors) == 0
    saved = edits.save(path, [shown], same, kind="artifacts")  # an unchanged simplified polygon keeps its full detail
    assert len(shape(edits.read(path, kind="artifacts")[0][0]["geometry"]).interiors) == len(holes) and saved["revision"]


def test_fill_holes_fills_only_the_small_ones():
    from shapely.geometry import Polygon
    from spacemap_gui.studio import artifact_learning
    clot = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], [[(10, 10), (12, 10), (12, 12), (10, 12)], [(40, 40), (70, 40), (70, 70), (40, 70)]])
    kept = artifact_learning._tidy([{"type": "Feature", "geometry": mapping(clot), "properties": {}}], 100, 1.0)
    assert len(shape(kept[0]["geometry"]).interiors) == 1  # the 4 px² hole filled, the 900 px² hole kept
    untouched = artifact_learning._tidy([{"type": "Feature", "geometry": mapping(clot), "properties": {}}], 0, 1.0)
    assert len(shape(untouched[0]["geometry"]).interiors) == 2
