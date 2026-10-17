"""Studio's tissue pieces and zones run Tab 4's steps 2b and 4 and write their files."""

import json

import pytest

from spacemap_gui.studio import annotation, pieces, zones
from test_studio_commands import setup  # noqa: F401 (fixture)


def square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def feature(ring, name="Valid_Tissue", colour=-16711936):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"objectType": "annotation", "classification": {"name": name, "colorRGB": colour}, "name": name},
    }


def write(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


@pytest.fixture
def tissue(tmp_path):
    """A large piece, a scrap too small to keep, and two lobes joined by a 20 um bridge."""
    output = tmp_path / "stage0"
    dumbbell = [[0, 1500], [800, 1500], [800, 1880], [1200, 1880], [1200, 1500], [2000, 1500], [2000, 2300]]
    dumbbell += [[1200, 2300], [1200, 1920], [800, 1920], [800, 2300], [0, 2300], [0, 1500]]
    write(
        annotation.valid_tissue(output, "S1"),
        [feature(square(0, 0, 1000, 1000)), feature(square(1500, 0, 1700, 200)), feature(dumbbell)],
    )
    return output


def test_the_reader_is_step_2bs(tissue):
    from spacemap_gui.tabs.tab4_annotation.s2b_tissue_pieces import _polygons_of

    path = str(annotation.valid_tissue(tissue, "S1"))
    assert [p.wkt for p in pieces.polygons(path)] == [p.wkt for p in _polygons_of(path)]
    assert pieces.polygons(path, ("Other",)) == _polygons_of(path, ("Other",)) == []


def test_pieces_preview_keeps_drops_and_finds_the_neck(tissue):
    values = pieces.settings({})
    found = pieces.preview(tissue, "S1", values)
    assert found["counts"] == {"kept": 2, "rejected": 1, "necks": 1}, found["reasons"]
    assert found["source"] == "S1_valid_tissue.geojson" and len(found["kept"]) == 2 and found["necks"]
    cut = pieces.preview(tissue, "S1", {**values, "cut_at_necks": True})
    assert cut["counts"]["kept"] == 3
    loose = pieces.preview(tissue, "S1", {**values, "min_area_mm2": 0.001, "detect_necks": False})
    assert loose["counts"] == {"kept": 3, "rejected": 0, "necks": 0}
    with pytest.raises(ValueError):
        pieces.check({"min_area_mm2": -1})
    with pytest.raises(ValueError):
        pieces.check({"shape": 1})


def test_neck_side_controls_and_preview_tissue(tissue):
    values = pieces.settings({})
    changes = pieces.check({'neck_min_lobe_area_mm2': 0.5, 'neck_min_lobe_fraction': 0.1})
    strict = pieces.preview(tissue, 'S1', {**values, **changes})
    assert strict['counts']['necks'] == 0
    original = annotation.valid_tissue(tissue, 'S1').read_bytes()
    features = json.loads(original)['features']
    found = pieces.preview(tissue, 'S1', values, features=features)
    assert found['source'] == 'current preview' and len(found['neck_regions']) == 1
    neck = found['neck_regions'][0]
    assert min(neck['side_areas_um2']) >= values['neck_min_lobe_area_mm2'] * 1e6
    assert neck['geometry']['type'] in ('Polygon', 'MultiPolygon')
    assert annotation.valid_tissue(tissue, 'S1').read_bytes() == original
    from spacemap_gui.studio import annotation_edits
    prepared = annotation_edits.prepare(features, kind='tissue')
    approved = annotation_edits.neck_features(prepared, neck['geometry'])
    assert len(approved['features']) > len(prepared)


def test_pieces_apply_writes_tab4_files_and_a_rerun_uses_current_tissue(tissue):
    values = pieces.settings({})
    done = pieces.apply(tissue, "S1", values)
    assert (done["kept"], done["rejected"], done["necks"]) == (2, 1, 1)
    review = json.loads((tissue / "intermediate/S1_tissue_pieces.geojson").read_text())["features"]
    assert sorted({f["properties"]["name"] for f in review}) == ["Detected_Neck", "Rejected_Tissue", "Valid_Tissue"]
    valid = json.loads(annotation.valid_tissue(tissue, "S1").read_text())["features"]
    assert len(valid) == 2 and {f["properties"]["name"] for f in valid} == {"tissue_1", "tissue_2"}
    assert {f["properties"]["classification"]["name"] for f in valid} == {"Valid_Tissue"}
    # A looser limit must not resurrect discarded tissue from an older review.
    again = pieces.apply(tissue, "S1", {**values, "min_area_mm2": 0.001})
    assert again["kept"] == 2 and again["source"] == "S1_valid_tissue.geojson"


def test_current_tissue_takes_precedence_over_review_and_detected_boundary(tissue):
    from spacemap_gui.studio import annotation_edits

    values = pieces.settings({})
    pieces.apply(tissue, "S1", {**values, "cut_at_necks": True})
    current = annotation.valid_tissue(tissue, "S1")
    # The old review and raw detection still contain a neck. Current tissue does not.
    write(annotation.boundary(tissue, "S1"), [feature(square(0, 0, 10000, 10000))])
    found = pieces.preview(tissue, "S1", values)
    assert found["counts"]["necks"] == 0
    assert found["source"] == current.name
    version = annotation_edits.revision(current)
    write(current, [])
    with pytest.raises(ValueError, match="Tissue changed"):
        pieces.preview(tissue, "S1", values, expected=version)
    with pytest.raises(ValueError, match="no tissue"):
        pieces.preview(tissue, "S1", values)
    current.unlink()
    with pytest.raises(ValueError, match="Detect tissue first"):
        pieces.preview(tissue, "S1", values)


def test_neck_preview_route_uses_loaded_layer_and_checks_image_and_revision(setup, tissue):  # noqa: F811
    from starlette.testclient import TestClient
    from spacemap_gui.studio import annotation_edits
    from spacemap_gui.viewer import api, registry, ui

    channel, _, root, spec = setup
    image = registry.resolve(spec["layers"][0]["token"])
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    write(annotation.valid_tissue(output, "P11_roi_2"), [feature(square(0, 0, 10, 10))])
    path = annotation.valid_tissue(tissue, "S1")
    layer = ui.shapes_layer(path, image, annotation_kind="tissue", layer_id="tissue-regions")
    values = pieces.settings({})
    changes = {key: values[key] for key in pieces.EDITABLE}
    payload = {"changes": changes, "tissue_token": layer["token"], "image_token": layer["image_token"],
               "revision": annotation_edits.revision(path)}
    client = TestClient(api.app)
    route = f"/studio/{channel}/pieces_preview"
    original = path.read_bytes()
    answer = client.post(route, json=payload)
    assert answer.status_code == 200, answer.text
    found = answer.json()
    assert found["counts"]["necks"] == 1
    assert found["layer"]["saved_file"] == str(path.resolve())
    assert path.read_bytes() == original
    # Unsaved tissue takes precedence even when the saved version has a neck.
    preview = client.post(route, json={"changes": changes, "image_token": layer["image_token"],
                                      "features": [feature(square(0, 0, 1000, 1000))]})
    assert preview.status_code == 200, preview.text
    assert preview.json()["counts"]["necks"] == 0
    empty = client.post(route, json={"changes": changes, "features": []})
    assert empty.status_code == 400 and "Detect tissue first" in empty.json()["error"]
    stale = client.post(route, json={**payload, "revision": "old"})
    assert stale.status_code == 400 and "Tissue changed" in stale.json()["error"]
    from shutil import copyfile
    copyfile(image, root / "other.ome.tiff")
    other = ui.shapes_layer(path, str(root / "other.ome.tiff"), annotation_kind="tissue")
    wrong_layer = client.post(route, json={**payload, "tissue_token": other["token"]})
    assert wrong_layer.status_code == 400 and "this image's Tissue" in wrong_layer.json()["error"]
    wrong_image = client.post(route, json={**payload, "image_token": other["image_token"]})
    assert wrong_image.status_code == 400 and "image changed" in wrong_image.json()["error"]
    # Without a loaded layer, fallback registers current valid tissue, not the raw boundary.
    fallback = client.post(route, json={"changes": changes})
    assert fallback.status_code == 200, fallback.text
    assert fallback.json()["layer"]["saved_file"] == str(annotation.valid_tissue(output, "P11_roi_2"))


def test_neck_settings_error_identifies_unrecognised_keys():
    with pytest.raises(ValueError, match="Unrecognised neck setting: obsolete_control"):
        pieces.check({"obsolete_control": 1})
    with pytest.raises(ValueError, match="could not be read"):
        pieces.check(None)


def test_piece_settings_save_into_tab4(tmp_path):
    from spacemap_gui.studio.settings import read_config, revision

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"tab_configurations": {"tab4_annotation": {}}}))
    before = pieces.settings(read_config(root))
    saved = pieces.save(root, {"min_area_mm2": 0.2, "cut_at_necks": True}, revision(before))
    stored = json.loads((root / "study.json").read_text())["tab_configurations"]["tab4_annotation"]
    assert stored["annot_ta_pieces_config"] == {"min_area_mm2": 0.2, "cut_at_necks": True}
    assert saved["min_area_mm2"] == 0.2 and saved["neck_min_width_um"] == before["neck_min_width_um"]
    with pytest.raises(ValueError, match="another window"):
        pieces.save(root, {"min_area_mm2": 0.3}, revision(before))


@pytest.fixture
def domains(tmp_path):
    output = tmp_path / "stage0"
    write(
        annotation.domains(output, "S1"),
        [feature(square(0, 0, 1000, 1000), "Tumor", -65536), feature(square(1000, 0, 2000, 1000), "Stroma", -16776961)],
    )
    return output


RULES = {
    "pools": [{"name": "tissue", "sources": ["Tumor", "Stroma"], "subtract": []}],
    "zones": [{"label": "Tumor", "width_px": 100, "inner": 1, "outer": 2}],
    "keep_sources": True,
    "clip": "tissue",
}


def test_zones_preview_draws_pools_and_rings_without_writing(domains):
    found = zones.preview(domains, "S1", {}, RULES, (1200, 2200))
    names = {layer["name"]: layer for layer in found["layers"]}
    assert set(names) == {"tissue", "outer_1", "outer_2", "inner_1"}
    assert names["tissue"]["kind"] == "pool" and names["outer_1"]["kind"] == "zone"
    assert sorted(found["classes"]) == ["Stroma", "Tumor"]
    # Outer rings are clipped to the pooled tissue: they stay within x 0..2000, y 0..1000.
    for layer in ("outer_1", "outer_2"):
        for ring in names[layer]["rings"]:
            assert all(-1 <= x <= 2001 and -1 <= y <= 1001 for x, y in ring)
    assert sorted(p.name for p in domains.rglob("*")) == ["S1_macro_domains.geojson"]
    with pytest.raises(ValueError):
        zones.check({**RULES, "zones": [{"label": "Tumor", "width_px": 0, "inner": 1, "outer": 0}]})
    with pytest.raises(ValueError, match="Export the pixel classes"):
        zones.preview(domains, "S2", {}, RULES, (1200, 2200))


@pytest.mark.parametrize("image_shape", [(1200, 2200), (4800, 4800)])
def test_the_raster_preview_agrees_with_tab4s_zones(domains, tmp_path, image_shape):
    from shapely.geometry import shape

    rules = {**RULES, "zones": [{"label": "Tumor", "width_px": 120, "inner": 2, "outer": 2}]}
    found = {
        layer["name"]: layer["area_px"] for layer in zones.preview(domains, "S1", {}, rules, image_shape)["layers"]
    }
    zones.build(annotation.domains(domains, "S1"), zones.check(rules), {}, tmp_path / "vector.geojson")
    vector = {
        f["properties"]["name"]: shape(f["geometry"]).area
        for f in json.loads((tmp_path / "vector.geojson").read_text())["features"]
        if f["properties"]["name"] in found
    }
    assert set(found) == set(vector) == {"tissue", "outer_1", "outer_2", "inner_1", "inner_2"}
    for name, area in vector.items():
        assert found[name] == pytest.approx(area, rel=0.03), name


def test_zones_apply_writes_tab4s_file_in_step_4s_folder(domains, tmp_path):
    from spacemap_gui.studio.settings import read_config, revision

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"tab_configurations": {"tab4_annotation": {}}}))
    assert zones.values(read_config(root)) == {"pools": [], "zones": [], "keep_sources": True, "clip": None}
    saved = zones.save(root, RULES, revision(zones.values(read_config(root))))
    stored = json.loads((root / "study.json").read_text())["tab_configurations"]["tab4_annotation"]
    assert stored["annot_ta_zone_rules"][0] == {"label": "Tumor", "width_px": 100.0, "inner": 1, "outer": 2}
    assert stored["annot_ta_zone_clip_class"] == "tissue" and saved["clip"] == "tissue"
    folder = tmp_path / "stage1b"
    done = zones.apply(domains, folder, "S1", read_config(root))
    assert done["pooled"] == ["tissue"] and done["zones"] == 3
    written = json.loads((folder / "S1_zones.geojson").read_text())["features"]
    assert sorted(f["properties"]["name"] for f in written) == [
        "Stroma",
        "Tumor",
        "inner_1",
        "outer_1",
        "outer_2",
        "tissue",
    ]
