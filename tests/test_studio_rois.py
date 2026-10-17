"""Studio's ROI step reads and writes Tab 3's own files."""

import json

import numpy as np
import pytest
import tifffile

from spacemap_gui.studio import rois
from test_studio_commands import setup  # noqa: F401  (fixture)
from test_viewer_cell_profiles import marker_image


def square(name, x, y, size):
    ring = [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
    return {"type": "Feature", "properties": {"name": name}, "geometry": {"type": "Polygon", "coordinates": [ring]}}


def test_folders_follow_the_pipeline(tmp_path):
    registered = tmp_path / "outputs/registered"
    registered.mkdir(parents=True)
    (registered / "P11_merged.ome.tiff").touch()
    mappings = {
        "tab1_batch_output_dir": "${PROJECT_ROOT}/outputs/registered",
        "tab3_geojson_dir": "${PROJECT_ROOT}/data/boundaries",
    }
    images, shapes, outputs = rois.folders(tmp_path, mappings)
    assert images == registered and shapes == tmp_path / "data/boundaries" and outputs == tmp_path / "outputs/rois"
    background = tmp_path / "outputs/background_subtracted"
    background.mkdir(parents=True)
    (background / "P11_bg_subtracted.ome.tiff").touch()
    assert rois.folders(tmp_path, {**mappings, "tab2_output_dir": str(background)})[0] == background
    assert rois.roi_file(shapes, "P11_merged.ome.tiff").name == "P11_roi_bounding_boxes_coords.geojson"


def test_roi_names_extraction_state_and_viewer_colours(tmp_path):
    path = tmp_path / "P11_roi_bounding_boxes_coords.geojson"
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": [square("roi_1", 0, 0, 8), square("tumour edge", 8, 8, 8)]}
        )
    )
    assert rois.roi_names(path) == ["roi_1", "tumour edge"]
    (tmp_path / "P11_roi_1.ome.tiff").touch()
    assert rois.extracted(tmp_path, "P11_merged.ome.tiff", ["roi_1", "tumour edge"]) == ["roi_1"]
    display = tmp_path / "display.json"
    display.write_text(json.dumps({"channels": [{"index": 1, "name": "M1", "label": "CD8", "color": "#00ff00"}]}))
    config = rois.channel_config(display, ["M0", "M1"])
    assert (
        config["keep"] == ["M0", "M1"] and config["rename"] == {"M1": "CD8"} and config["colors"]["CD8"] == (0, 255, 0)
    )


def test_extract_cuts_each_roi_with_tab3(tmp_path):
    image = tmp_path / "P11_merged.ome.tiff"
    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    marker_image(image, data)
    shapes = tmp_path / "P11_roi_bounding_boxes_coords.geojson"
    shapes.write_text(json.dumps({"type": "FeatureCollection", "features": [square("roi_1", 4, 4, 10)]}))
    outputs = tmp_path / "rois"
    messages = []
    written = rois.extract(
        [{"image": str(image), "rois": str(shapes), "outputs": str(outputs)}],
        lambda text, progress: messages.append(text),
    )
    assert written == [str(outputs / "P11_roi_1.ome.tiff")]
    with tifffile.TiffFile(written[0]) as tif:
        assert tif.pages[0].shape[0] >= 10
    assert rois.extract([{"image": str(image), "rois": str(shapes), "outputs": str(outputs)}], lambda *_: None) == [], (
        "An ROI already extracted is kept, not written again"
    )


def test_draw_and_extract_through_studio_commands(setup):  # noqa: F811
    import time

    from spacemap.viewer import shapes
    from spacemap_gui.studio import jobs
    from test_studio_commands import run

    root = setup[2]
    opened = run(setup, "draw_rois")
    assert opened["status"] == "done", opened
    layer = next(item for item in opened["spec"]["layers"] if item["id"] == "rois")
    assert layer["editable"] and layer["default_class"] == "ROI"
    assert run(setup, "draw_rois")["message"] == "The ROI layer is already open."
    target = root / "data/boundaries/P11_roi_2_roi_bounding_boxes_coords.geojson"
    early = run(setup, "extract_rois", {}, confirmed=True)
    assert early["status"] == "error" and "Draw ROIs" in early["message"]
    shapes.save(target, [square("edge", 2, 2, 20)])
    started = run(setup, "extract_rois", {}, confirmed=True)
    assert started["status"] == "done", started
    for _ in range(600):
        current = jobs.status(str(root), started["data"]["job"]["id"])
        if current["status"] not in jobs.ACTIVE:
            break
        time.sleep(0.1)
    assert current["status"] == "done", jobs.log(str(root), current["id"])
    assert current["result"]["images"] == [str(root / "outputs/rois/P11_roi_2_edge.ome.tiff")]


def test_tab3s_marker_choice_is_saved_and_extracted(tmp_path):
    from spacemap.background_subtraction_utils import get_channel_info
    from spacemap_gui.studio.settings import read_config

    root = tmp_path / "study"
    image = root / "outputs/registration/P11_merged.ome.tiff"
    image.parent.mkdir(parents=True)
    marker_image(image, np.arange(2 * 60 * 60, dtype=np.uint16).reshape(2, 60, 60))
    (root / "study.json").write_text("{}")
    shown = rois.save_markers(root, image, ["M0"], {"M0": "DNA", "M1": "unused"})
    assert [(item["name"], item["keep"], item["label"]) for item in shown] == [("M0", True, "DNA"), ("M1", False, "unused")]
    pair = rois.pair_settings(read_config(root), image.name)
    assert pair["export_map"] == {"M0": True, "M1": False} and pair["rename_map"] == {"M0": "DNA", "M1": "unused"}
    config = rois.channel_config(None, ["M0", "M1"], pair)
    assert config["keep"] == ["M0"] and config["rename"]["M0"] == "DNA"
    for keep, rename in (([], {}), (["M0", "M1"], {"M1": "M0"}), (["M0"], {"M0": "bad<name>"})):
        with pytest.raises(ValueError):
            rois.save_markers(root, image, keep, rename)
    shapes = tmp_path / "shapes"
    shapes.mkdir()
    rois.roi_file(shapes, image.name).write_text(
        json.dumps({"type": "FeatureCollection", "features": [square("a", 5, 5, 20)]})
    )
    rois.extract(
        [
            {
                "image": str(image),
                "rois": str(rois.roi_file(shapes, image.name)),
                "outputs": str(tmp_path / "out"),
                "pair": pair,
            }
        ],
        lambda *args: None,
    )
    assert list(get_channel_info(str(tmp_path / "out" / "P11_a.ome.tiff"))[0]) == ["DNA"]


def _tma_project(tmp_path):
    """A project with a 2-marker image showing a 2 x 3 grid of round cores (a small TMA)."""
    import json

    import cv2

    root = tmp_path / "study"
    (root / "outputs/registered").mkdir(parents=True)
    (root / "data").mkdir()
    dapi = np.zeros((1400, 2100), np.uint16)
    for row in range(2):
        for column in range(3):
            cv2.circle(dapi, (350 + column * 700, 350 + row * 700), 180, 4000, -1)
    tifffile.imwrite(root / "outputs/registered/T1_merged.ome.tiff", np.stack([dapi, dapi // 2]), metadata={
        "axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
        "Channel": {"Name": ["DAPI", "CD3"], "Fluor": ["DAPI", "ATTO 550"]}})
    (root / "study.json").write_text(json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}))
    return root


def test_the_roi_step_lists_images_finds_cores_and_extracts_with_the_metadata(tmp_path):
    import json

    from spacemap_gui.studio import rois
    from spacemap_gui.studio.settings import read_config

    root = _tma_project(tmp_path)
    shapes = root / "data/boundaries"
    rois.save_step(root, {"folders": {"images": str(root / "outputs/registered"), "shapes": str(shapes), "output": str(root / "outputs/rois")}})
    plan = rois.images_plan(root, {}, read_config(root))
    assert plan["folders"]["images"] == str(root / "outputs/registered")  # kept as typed
    row = plan["rows"][0]
    assert row["image"] == "T1_merged.ome.tiff" and row["file"] == "T1_roi_bounding_boxes_coords.geojson" and row["rois"] == 0
    # Core settings are saved; the detector (classic Tab 3's) names the cores by row and column.
    rois.save_step(root, {"tma": {"min": 60, "max": 200, "prefix": "C_"}})
    assert rois.images_plan(root, {}, read_config(root))["tma"]["min"] == 60
    with pytest.raises(ValueError):
        rois.save_step(root, {"tma": {"min": 300, "max": 200}})
    image = root / "outputs/registered/T1_merged.ome.tiff"
    count = rois.detect_cores(image, shapes / row["file"], rois.images_plan(root, {}, read_config(root))["tma"])
    assert count == 6 and rois.roi_names(shapes / row["file"]) == ["C_A1", "C_A2", "C_A3", "C_B1", "C_B2", "C_B3"]
    first = json.loads((shapes / row["file"]).read_text())["features"][0]["geometry"]["coordinates"][0]
    assert abs(first[0][0] - 170) < 30 and abs(first[2][0] - 530) < 30  # back in full-resolution pixels
    # A picked GeoJSON replaces the one by name.
    (shapes / "other.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "one"},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [100, 0], [100, 80], [0, 0]]]}}]}))
    rois.save_step(root, {"files": {"T1_merged.ome.tiff": "other.geojson"}})
    assert rois.images_plan(root, {}, read_config(root))["rows"][0]["rois"] == 1
    rois.save_step(root, {"files": {"T1_merged.ome.tiff": ""}})  # back to the file by name
    assert rois.images_plan(root, {}, read_config(root))["rows"][0]["rois"] == 6
    # Markers: dye and exposure from the file; name and colour chosen here reach the extracted images.
    found = rois.markers(root, image, None)
    assert [m["name"] for m in found] == ["DAPI", "CD3"] and found[1]["dye"] != "-" and all(m["keep"] for m in found)
    rois.save_markers(root, image, ["DAPI", "CD3"], {"CD3": "CD3e"}, {"CD3": "#FF0000"})
    found = rois.markers(root, image, None)
    assert found[1]["rename"] == "CD3e" and found[1]["new_color"] == "#ff0000"
    assert rois.images_plan(root, {}, read_config(root))["rows"][0]["edited"]
    written = rois.extract([{"image": str(image), "rois": str(shapes / row["file"]), "outputs": str(root / "outputs/rois"),
                             "pair": rois.pair_settings(read_config(root), image.name)}], lambda *args: None)
    assert len(written) == 6
    with tifffile.TiffFile(root / "outputs/rois/T1_C_A1.ome.tiff") as tif:
        xml = tif.ome_metadata
        assert len(tif.series[0].levels) > 1  # a pyramid
    assert 'Name="CD3e"' in xml and 'PhysicalSizeX="0.5"' in xml and "Cropped from" in xml
    assert str(-65536) in xml or "Color=\"-16776961\"" in xml  # red, as OME writes it (RGBA as a signed integer)
