"""Studio's tissue detection writes the files Tab 4's step 1 writes."""

import json
from pathlib import Path

import numpy as np
import pytest

from spacemap_gui.studio import annotation
from test_studio_commands import run, setup  # noqa: F401  (fixture)
from test_viewer_cell_profiles import marker_image


def test_paths_are_tab4s_own(tmp_path, monkeypatch):
    from spacemap_gui.tabs.tab4_annotation import common

    monkeypatch.setattr(common.st, "session_state", {})
    output = str(tmp_path / "stage0")
    assert str(annotation.valid_tissue(output, "P11_roi_1")) == common.geojson_path("P11_roi_1", "valid_tissue", output)
    assert str(annotation.boundary(output, "P11_roi_1")) == common.geojson_path("P11_roi_1", "tissue_boundary", output)
    assert str(annotation.clot(output, "P11_roi_1")) == common.geojson_path("P11_roi_1", "blood_clot", output)
    assert str(annotation.zones(output, "P11_roi_1")) == common.geojson_path("P11_roi_1", "zones", output)
    assert annotation.sample_id("P11_roi_1.ome.tiff") == common.sample_id_from_filename("P11_roi_1.ome.tiff")
    images, out = annotation.folders(tmp_path, {"annot_ta_input_dir": "${PROJECT_ROOT}/outputs/rois"})
    assert (
        images == tmp_path / "outputs/rois"
        and out == tmp_path / "outputs/processed_annotations/stage0_tissue_annotation/working_files"  # Sub-directory, on by default
    )
    out = annotation.main_output(tmp_path, {})
    from spacemap_gui.tabs.tab0_project import get_default_directory_mappings

    defaults = get_default_directory_mappings(str(tmp_path), str(tmp_path / "codex"), str(tmp_path / "he"))
    assert str(out) == defaults["annot_ta_output_dir"]
    assert str(annotation.zones_folder(tmp_path, {})) == defaults["annot_ta_pool_output_dir"]


def test_settings_merge_check_and_save(tmp_path):
    from spacemap_gui.studio.settings import revision

    config = {"tab_configurations": {"tab4_annotation": {"annot_ta_tissue_config": {"tissue_morph_radius": 7}}}}
    values = annotation.settings(config)
    assert values["tissue_morph_radius"] == 7 and values["tissue_mask_mode"] == "he_only"
    for wrong in ({"tissue_mask_mode": "magic"}, {"blur_downsample": 3}, {"boundary_simplify_px": 1}, {"min_tissue_island_px": -1}):
        with pytest.raises(ValueError):
            annotation.check(wrong)
    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps(config))
    saved = annotation.save(root, {"he_od_threshold_factor": 0.7}, revision(values))
    stored = json.loads((root / "study.json").read_text())["tab_configurations"]["tab4_annotation"][
        "annot_ta_tissue_config"
    ]
    assert saved["he_od_threshold_factor"] == 0.7 and stored == {
        "tissue_morph_radius": 7,
        "he_od_threshold_factor": 0.7,
    }


def he_image(path):
    """A white slide with one stained square, as three H&E layers plus DAPI."""
    size = 400
    white = np.full((size, size), 240, np.uint16)
    stained = white.copy()
    stained[100:300, 100:300] = 90
    data = np.stack([stained, stained, white, stained])
    marker_image(path, data, names=["HE_Red", "HE_Green", "HE_Blue", "DAPI"])


def test_detection_writes_tab4_files(tmp_path):
    image = tmp_path / "S1_roi_1.ome.tiff"
    he_image(image)
    values = {**annotation.settings({}), "min_tissue_island_px": 100, "tissue_morph_radius": 3}
    preview = annotation.detect(str(image), {}, values)
    assert preview["written"] == [] and preview["features"], "A preview writes nothing"
    output = tmp_path / "stage0"
    found = annotation.detect(str(image), {}, values, output=output, sample="S1_roi_1")
    assert found["written"] == [
        str(output / "intermediate/S1_roi_1_tissue_boundary.geojson"),
        str(output / "S1_roi_1_valid_tissue.geojson"),
    ]
    valid = json.loads((output / "S1_roi_1_valid_tissue.geojson").read_text())
    assert valid["features"][0]["properties"]["classification"]["name"] == "Valid_Tissue"
    xs = [x for ring in annotation.rings(valid["features"]) for x, _ in ring]
    # On a synthetic slide Tab 4's detector keeps everything inside its border margin;
    # what matters here is that its result is written where Tab 4 writes it.
    assert 0 <= min(xs) <= 100 and 300 <= max(xs) <= 400, "The tissue covers the stained square, inside the image"


def test_preview_lists_polygons_without_writing_and_uses_project_folder(setup, monkeypatch):  # noqa: F811
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api, registry

    channel, _, root, spec = setup
    output = root / "reviewed_tissue"
    registry.publish(channel, spec, project={"root": str(root), "mappings": {"annot_ta_output_dir": str(output)}})
    features = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 0]]]},
                 "properties": {"id": "tissue_1", "name": "tissue_1"}}]
    monkeypatch.setattr(annotation, "detect", lambda *args: {"features": features, "stats": {"tissue_pct": 5}})
    result = TestClient(api.app).post(f"/studio/{channel}/tissue_preview", json={"settings": {}})
    assert result.status_code == 200, result.text
    preview = result.json()
    assert preview["pieces"] == len(preview["features"]) == 1
    assert preview["features"][0]["properties"]["name"] == "tissue_1"
    assert preview["layer"]["id"] == "tissue-regions" and preview["layer"]["regions"]
    assert preview["layer"]["saved_file"] == str(output / "working_files" / "P11_roi_2_valid_tissue.geojson")
    assert not output.exists(), "A preview must not create or replace project annotation files"


def test_pruned_preview_saves_full_detail_and_rejects_stale_saves(setup, monkeypatch):  # noqa: F811
    from shapely.geometry import Point, box, mapping, shape
    from starlette.testclient import TestClient
    from spacemap_gui.studio import annotation_edits
    from spacemap_gui.viewer import api

    channel, _, _, _ = setup
    detailed = Point(200, 200).buffer(100, quad_segs=1500)
    features = annotation_edits.prepare([
        {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
        for geometry in [detailed, box(400, 400, 500, 500), box(600, 600, 700, 700)]
    ], kind="tissue")
    monkeypatch.setattr(annotation, "detect", lambda *args: {"features": features, "stats": {}})
    client = TestClient(api.app)
    preview = client.post(f"/studio/{channel}/tissue_preview", json={"settings": {}}).json()
    token = preview["layer"]["token"]
    before = client.get(f"/shapes/{token}").json()
    assert shape(preview["features"][0]["geometry"]).equals_exact(detailed, 0)
    assert not preview["features"][0]["properties"].get("simplified")
    # Delete two preview pieces; the remaining detailed boundary must be saved exactly.
    saved = client.post(f"/shapes/{token}/save", json={"features": preview["features"][:1], "revision": before["revision"]})
    assert saved.status_code == 200, saved.text
    stored = json.loads(Path(preview["layer"]["saved_file"]).read_text())["features"]
    assert len(stored) == 1 and shape(stored[0]["geometry"]).equals_exact(detailed, 0)
    stale = client.post(f"/shapes/{token}/save", json={"features": [], "revision": before["revision"]})
    assert stale.status_code == 409
    assert len(json.loads(Path(preview["layer"]["saved_file"]).read_text())["features"]) == 1


def test_hole_filling_settings_default_off_and_reject_invalid_sizes(tmp_path):
    from spacemap_gui.studio.settings import read_config, revision

    assert annotation.settings({})["fill_holes_um2"] == 0
    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            annotation.check({"fill_holes_um2": value})
    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text('{}')
    annotation.save(root, {"fill_holes_um2": 25}, revision(annotation.settings({})))
    assert annotation.settings(read_config(root))["fill_holes_um2"] == 25


def test_preview_and_saved_tissue_fill_the_same_small_holes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from shapely.geometry import box, mapping, shape
    from spacemap import tissue_annotation_utils as classic
    from spacemap.viewer import sources

    outer = box(0, 0, 100, 100)
    small, large = box(10, 10, 12, 12), box(30, 30, 40, 40)
    geometry = outer.difference(small).difference(large)
    monkeypatch.setattr(classic, "PyramidImageReader", lambda _: SimpleNamespace(channel_names=["DAPI"], full_shape=(100, 100)))
    monkeypatch.setattr(classic, "detect_tissue_mask", lambda *a, **k: {"tissue_mask": np.ones((100, 100), bool), "stats": {"tissue_percent": 98.96}})
    monkeypatch.setattr(classic, "mask_to_geojson_features", lambda *a, **k: [{"type":"Feature", "geometry":mapping(geometry), "properties":{}}])
    monkeypatch.setattr(sources, "image_info", lambda _: SimpleNamespace(pixel_size_um=0.5))
    values = {**annotation.settings({}), "fill_holes_um2": 2, "microns_per_pixel": 10, "min_tissue_area_um2": 0}
    preview = annotation.detect("sample.ome.tiff", {}, values)
    assert preview["stats"]["holes_filled"] == 1 and preview["written"] == []
    assert shape(preview["features"][0]["geometry"]).equals(outer.difference(large))
    assert preview["stats"]["tissue_percent"] == pytest.approx(99)
    output = tmp_path / "tissue"
    saved = annotation.detect("sample.ome.tiff", {}, values, output=output)
    for path in saved["written"]:
        features = json.loads(Path(path).read_text())["features"]
        assert shape(features[0]["geometry"]).equals(outer.difference(large))


def test_detect_tissue_command_and_work_list(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs
    from spacemap_gui.studio.steps import work_list

    root = setup[2]
    started = {}
    monkeypatch.setattr(
        jobs,
        "start",
        lambda project, kind, settings, request_id=None: started.update(kind=kind, settings=settings) or {"id": "x"},
    )
    result = run(setup, "detect_tissue", {}, confirmed=True)
    assert result["status"] == "done", result
    item = started["settings"]["images"][0]
    assert started["kind"] == "detect_tissue" and item["sample"] == "P11_roi_2"
    assert item["output"].endswith("processed_annotations/stage0_tissue_annotation/working_files")  # new project: Sub-directory on
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    output.mkdir(parents=True)
    (output / "P11_roi_2_valid_tissue.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    rows = {row["name"]: row["state_label"] for row in work_list(setup[0], "tab4")["items"]}
    assert rows["P11_roi_2"] == "Tissue found"


def test_export_classes_needs_names_for_this_image(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs
    from spacemap_gui.studio.steps import work_list

    root = setup[2]
    started = {}
    monkeypatch.setattr(
        jobs,
        "start",
        lambda project, kind, settings, request_id=None: started.update(kind=kind, settings=settings) or {"id": "x"},
    )
    result = run(setup, "export_classes", {}, confirmed=True)
    assert result["status"] != "done" and "Save the class names" in result["message"] and not started
    tab4 = {
        "annot_ta_domain_map_image": {"sample": "P11_roi_2", "n_clusters": 6},
        "annot_ta_domain_map": {"0": "Tumor"},
    }
    (root / "study.json").write_text(json.dumps({"tab_configurations": {"tab4_annotation": tab4}}))
    result = run(setup, "export_classes", {}, confirmed=True)
    assert result["status"] == "done", result
    assert started["kind"] == "export_classes" and started["settings"]["sample"] == "P11_roi_2"
    assert started["settings"]["config"]["tab_configurations"]["tab4_annotation"] == tab4
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    output.mkdir(parents=True)
    (output / "P11_roi_2_raw_macro_clusters.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    assert {row["name"]: row["state_label"] for row in work_list(setup[0], "tab4")["items"]}["P11_roi_2"] == (
        "Classes exported"
    )
    (output / "P11_roi_2_macro_domains.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    assert {row["name"]: row["state_label"] for row in work_list(setup[0], "tab4")["items"]}["P11_roi_2"] == (
        "Domains exported"
    )
    # Tab 4's step 4 writes zones to its own folder.
    zones = root / "outputs/processed_annotations/stage1b_compartments_zones"
    zones.mkdir(parents=True)
    (zones / "P11_roi_2_zones.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    assert {row["name"]: row["state_label"] for row in work_list(setup[0], "tab4")["items"]}["P11_roi_2"] == "Zones"


def test_the_images_list_label_scale_and_drawn_artifacts(tmp_path):
    from spacemap_gui.studio.settings import read_config

    root = tmp_path / "study"
    images = root / "outputs/rois"
    images.mkdir(parents=True)
    (root / "study.json").write_text(json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}))
    (images / "S1.ome.tiff").write_bytes(b"")
    output = root / "outputs/processed_annotations/stage0_tissue_annotation"
    plan = annotation.images_plan(root, {}, read_config(root))
    assert plan["rows"][0]["sample"] == "S1" and not any(plan["rows"][0]["done"].values()) and plan["label"] == "tissue"
    # The tissue is done once its file is there; drawn artifacts are marked, and taken out of the tissue.
    (output / "intermediate").mkdir(parents=True)
    square = lambda x0, y0, x1, y1: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]  # noqa: E731
    annotation.valid_tissue(output, "S1").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"classification": {"name": "Valid_Tissue"}}, "geometry": {"type": "Polygon", "coordinates": square(0, 0, 100, 100)}}]}))
    annotation.drawn_artifacts(output, "S1").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "fold"}, "geometry": {"type": "Polygon", "coordinates": square(0, 0, 50, 100)}}]}))
    plan = annotation.images_plan(root, {}, read_config(root))
    assert plan["rows"][0]["done"]["tissue"] and plan["rows"][0]["drawn"]
    assert annotation.subtract_drawn(output, "S1") == 1
    left = json.loads(annotation.valid_tissue(output, "S1").read_text())["features"][0]
    from shapely.geometry import shape

    assert abs(shape(left["geometry"]).area - 5000) < 1 and left["properties"]["classification"]["name"] == "Valid_Tissue"
    # The label and folders are saved; the scale is one of the pyramid steps.
    annotation.save_step(root, {"label": "Tissue_ROI", "folders": {"output": str(root / "outputs/annotations")}})
    plan = annotation.images_plan(root, {**read_config(root).get("directory_mappings", {})}, read_config(root))
    assert plan["label"] == "Tissue_ROI" and plan["folders"]["output"] == str(root / "outputs/annotations")
    with pytest.raises(ValueError):
        annotation.save_step(root, {"label": "1 bad"})
    with pytest.raises(ValueError):
        annotation.check({"blur_downsample": 3})
    assert annotation.check({"blur_downsample": 4}) == {"blur_downsample": 4}


def test_minimum_area_migrates_old_scale_pixels_and_stays_fixed_when_scale_changes(tmp_path):
    from spacemap_gui.studio.settings import read_config, revision

    config = {'tab_configurations': {'tab4_annotation': {'annot_ta_tissue_config': {
        'microns_per_pixel': 0.5, 'blur_downsample': 2, 'min_tissue_island_px': 5000,
    }}}}
    values = annotation.settings(config)
    assert values['min_tissue_area_um2'] == 5000
    root = tmp_path / 'study'
    root.mkdir()
    (root / 'study.json').write_text(json.dumps(config))
    saved = annotation.save(root, {'blur_downsample': 1}, revision(values))
    assert saved['min_tissue_area_um2'] == 5000 and saved['min_tissue_island_px'] == 20000
    saved = annotation.save(root, {'min_tissue_area_um2': 2500, 'blur_downsample': 4}, revision(saved))
    assert saved['min_tissue_area_um2'] == 2500 and saved['min_tissue_island_px'] == 625
    assert annotation.settings(read_config(root))['min_tissue_area_um2'] == 2500
    for value in [-1, True, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            annotation.check({'min_tissue_area_um2': value})


@pytest.mark.parametrize('scale', [1, 2, 4])
@pytest.mark.parametrize('pixel_size', [0.5, 1.0])
def test_final_minimum_area_uses_square_microns_at_every_scale(tmp_path, monkeypatch, scale, pixel_size):
    from types import SimpleNamespace
    from shapely.geometry import box, mapping, shape
    from spacemap import tissue_annotation_utils as classic
    from spacemap.viewer import sources

    # Match the reported case, plus a piece exactly on the limit and one above.
    polygons = [box(0, index * 3, area / pixel_size ** 2, index * 3 + 1)
                for index, area in enumerate([2569, 5000, 6000])]
    features = [{'type': 'Feature', 'properties': {}, 'geometry': mapping(p)} for p in polygons]
    seen = []
    def detect(reader, settings, **kwargs):
        seen.append(settings['min_tissue_island_px'])
        # Also represents the classic fallback restoring islands when all were
        # too small, or polygon repair splitting an initially connected region.
        return {'tissue_mask': np.ones((10, 10), bool), 'stats': {'tissue_percent': 100}}
    monkeypatch.setattr(classic, 'PyramidImageReader', lambda _: SimpleNamespace(channel_names=['DAPI'], full_shape=(10, 30000)))
    monkeypatch.setattr(classic, 'detect_tissue_mask', detect)
    monkeypatch.setattr(classic, 'mask_to_geojson_features', lambda *a, **k: features)
    monkeypatch.setattr(sources, 'image_info', lambda _: SimpleNamespace(pixel_size_um=pixel_size))
    values = {**annotation.settings({}), 'min_tissue_area_um2': 5000, 'blur_downsample': scale}
    preview = annotation.detect('sample.ome.tiff', {}, values)
    assert [shape(f['geometry']).area * pixel_size ** 2 for f in preview['features']] == [5000, 6000]
    assert preview['stats']['small_pieces_removed'] == 1
    assert seen == [int(5000 / (pixel_size * scale) ** 2)]
    saved = annotation.detect('sample.ome.tiff', {}, values, output=tmp_path / 'tissue')
    for path in saved['written']:
        written = json.loads(Path(path).read_text())['features']
        assert [shape(f['geometry']).area * pixel_size ** 2 for f in written] == [5000, 6000]
    empty = annotation.detect('sample.ome.tiff', {}, {**values, 'min_tissue_area_um2': 10000})
    assert empty['features'] == [] and empty['stats']['tissue_percent'] == 0


def test_the_sub_directory_is_on_for_new_folders_and_moves_existing_files_only_when_asked(tmp_path):
    """New output folders keep the step's files in ./working_files; a folder that already holds them stays as it is
    until Move, which takes the files, their History and every saved reference along; exports stay in the folder."""
    import json

    from spacemap_gui.studio import annotation_edits as edits

    root = tmp_path / "study"
    main = annotation.main_output(root, {})
    main.mkdir(parents=True)
    assert annotation.folders(root, {})[1] == main / "working_files"
    # An existing project: its files are in the output folder itself, so nothing changes until Move.
    tissue = main / "roi_valid_tissue.geojson"
    edits.replace(tissue, [], edits.revision(tissue), "Fixture")
    (main / "studio_export").mkdir()
    (main / "studio_export" / "named.json").write_text('["QC_roi.geojson"]')
    (main / "QC_roi.geojson").write_text("{}")
    record = root / "outputs/studio/annotation_layers/key/index.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({str(tissue.resolve()): {"name": "tissue"}}))
    assert annotation.folders(root, {})[1] == main and annotation.subfolder(root, main)["pending"] == 2  # file + History
    moved = annotation.save_subfolder(root, {}, True, "./working_files", move=True)
    target = main / "working_files"
    assert (target / tissue.name).is_file() and not tissue.exists() and (target / "studio_history" / tissue.name).is_dir()
    assert (main / "QC_roi.geojson").is_file()  # an export stays in the output folder
    assert str((target / tissue.name).resolve()) in json.loads(record.read_text())
    assert annotation.folders(root, {})[1] == target and moved["on"]
    assert edits.read(target / tissue.name)[1] == edits.revision(target / tissue.name)
    import pytest

    with pytest.raises(ValueError):
        annotation.check_subfolder("../outside")
