"""Studio's Background step writes Tab 2's own threshold tables and images."""

import time

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacemap_gui.studio import background, jobs
from test_studio_commands import run, setup  # noqa: F401  (fixture)


def table_for(root):
    return root / "data/thresholds/P11_roi_2.ome.tiff_-_Merged_Image.csv"


def test_levels_are_saved_in_tab2_layout_and_undone(setup):  # noqa: F811
    root = setup[2]
    first = run(setup, "save_background_levels", {"levels": {"M0": 1, "M1": 2.5}, "method": "Triangle"}, confirmed=True)
    assert first["status"] == "done", first
    saved = pd.read_csv(table_for(root))
    assert list(saved.columns) == ["Marker", "Polygon_1"]
    assert background.read_levels(table_for(root)) == {"M0": 1.0, "M1": 2.5}
    assert run(setup, "undo", {"command_id": first["command_id"]}, confirmed=True)["status"] == "done"
    assert not table_for(root).exists(), "Undo removes a table that did not exist before"
    run(setup, "save_background_levels", {"levels": {"M0": 1}}, confirmed=True)
    second = run(setup, "save_background_levels", {"levels": {"M0": 4}}, confirmed=True)
    assert background.read_levels(table_for(root)) == {"M0": 4.0}
    assert run(setup, "undo", {"command_id": second["command_id"]}, confirmed=True)["status"] == "done"
    assert background.read_levels(table_for(root)) == {"M0": 1.0}, "Undo restores the earlier table"


def test_unknown_markers_and_missing_levels_are_explained(setup):  # noqa: F811
    wrong = run(setup, "save_background_levels", {"levels": {"CD999": 3}}, confirmed=True)
    assert wrong["status"] == "error" and "not in this image" in wrong["message"]
    early = run(setup, "subtract_background", {}, confirmed=True)
    assert early["status"] == "error" and "Save background levels" in early["message"]


def test_several_polygons_are_averaged(tmp_path):
    path = tmp_path / "levels.csv"
    pd.DataFrame({"Marker": ["A", "B"], "Polygon_1": [2.0, 1.0], "Polygon_2": [4.0, None]}).to_csv(path, index=False)
    assert background.read_levels(path) == {"A": 3.0, "B": 1.0}


def test_background_is_removed_by_a_job(setup):  # noqa: F811
    root = setup[2]
    assert run(setup, "save_background_levels", {"levels": {"M0": 1}}, confirmed=True)["status"] == "done"
    started = run(setup, "subtract_background", {}, confirmed=True)
    assert started["status"] == "done", started
    job = started["data"]["job"]
    for _ in range(600):
        current = jobs.status(str(root), job["id"])
        if current["status"] not in jobs.ACTIVE:
            break
        time.sleep(0.1)
    assert current["status"] == "done", jobs.log(str(root), job["id"])
    output = root / "outputs/background_subtracted/P11_roi_2_bg_subtracted.ome.tiff"
    assert current["result"]["images"] == [str(output)]
    with tifffile.TiffFile(output) as tif:
        planes = [tif.pages[index].asarray() for index in range(2)]
    assert np.all(planes[0] == 0) and np.all(planes[1] == 1), "M0 loses its level, M1 has none"
    assert not list(output.parent.glob("*.partial"))


def test_saved_levels_route_reads_the_open_image(setup):  # noqa: F811
    from starlette.testclient import TestClient

    from spacemap_gui.viewer import api

    with TestClient(api.app) as client:
        route = f"/studio/{setup[0]}/background_levels"
        assert client.get(route).json()["levels"] is None
        run(setup, "save_background_levels", {"levels": {"M0": 7}}, confirmed=True)
        value = client.get(route).json()
        assert value["levels"] == {"M0": 7.0} and value["table"].startswith("P11_roi_2") and not value["removed"]


def _blank_project(tmp_path):
    """A 4-channel ROI with a blank cycle (ATTO550) and a tissue annotation, in a project."""
    import json

    import numpy as np
    import tifffile

    root = tmp_path / "study"
    (root / "outputs/rois").mkdir(parents=True)
    (root / "outputs/processed_annotations/stage0_tissue_annotation").mkdir(parents=True)
    planes = np.zeros((4, 64, 64), np.uint16)
    planes[0] += 1000  # DAPI
    planes[1] += 500  # CD3
    planes[2] += 120  # Blank ATTO550
    planes[3] += 300  # CD8
    tifffile.imwrite(root / "outputs/rois/P1_roi_1.ome.tiff", planes, metadata={
        "axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
        "Channel": {"Name": ["DAPI", "CD3", "Blank_ATTO550", "CD8"], "Fluor": ["DAPI", "ATTO 550", "ATTO 550", "AF 750"]}})
    tissue = {"type": "Feature", "properties": {"classification": {"name": "tissue"}},
              "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [64, 0], [64, 64], [0, 64], [0, 0]]]}}
    (root / "outputs/processed_annotations/stage0_tissue_annotation/P1_roi_1_valid_tissue.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [tissue]}))
    (root / "study.json").write_text(json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}))
    return root


def test_the_background_step_lists_images_blanks_annotations_and_removes_background(tmp_path):
    import numpy as np
    import tifffile

    from spacemap_gui.studio import background
    from spacemap_gui.studio.settings import read_config

    root = _blank_project(tmp_path)
    config = read_config(root)
    plan = background.images_plan(root, {}, config)
    row = plan["rows"][0]
    assert row["image"] == "P1_roi_1.ome.tiff" and row["blanks"] == ["Blank_ATTO550"]
    assert row["annotation"] == "P1_roi_1_valid_tissue.geojson" and row["classes"] == ["tissue"] and row["class"] == "tissue"
    found = background.image_markers(root, {}, config, "P1_roi_1.ome.tiff")
    markers = {m["name"]: m for m in found["markers"]}
    assert markers["CD3"]["subtract"] == "Blank_ATTO550" and markers["CD8"]["subtract"] == ""  # paired by dye, as classic
    assert not markers["Blank_ATTO550"]["keep"] and markers["DAPI"]["keep"]  # blanks left out by default
    background.save_step(root, {"image": "P1_roi_1.ome.tiff", "markers": {"keep": ["DAPI", "CD3", "CD8"], "rename": {"CD8": "CD8a"},
                                                                         "colors": {"CD3": "#ff0000"}, "blanks": {"CD3": "Blank_ATTO550"}},
                                "classes": {"P1_roi_1.ome.tiff": "tissue"}})
    assert background.images_plan(root, {}, read_config(root))["rows"][0]["edited"]
    output = root / "outputs/background_subtracted/P1_roi_1_bg_subtracted.ome.tiff"  # the folder is made by the step
    background.remove([{"image": str(root / "outputs/rois/P1_roi_1.ome.tiff"), "output": str(output), "keep": ["DAPI", "CD3", "CD8"],
                        "rename": {"CD8": "CD8a"}, "colors": {"CD3": "#ff0000"}, "blanks": {"CD3": "Blank_ATTO550"},
                        "levels": {"DAPI": 100.0, "CD3": 50.0}}], lambda *args: None)
    with tifffile.TiffFile(output) as tif:
        assert len(tif.series) == 1 and len(tif.series[0].levels) == 4
        data = tif.series[0].asarray()
        xml = tif.ome_metadata
    assert data.shape == (3, 64, 64) and data[0].max() == 900 and data[1].max() == 330 and data[2].max() == 300  # 500-120-50
    assert 'Name="CD8a"' in xml and "Blank_ATTO550" not in xml.split("<StructuredAnnotations")[0]
    assert "Background removed from" in xml and "CD3 - Blank_ATTO550" in xml
    with pytest.raises(ValueError):
        background.save_step(root, {"folders": {"images": str(tmp_path / "nowhere")}})


def test_auto_levels_for_many_images_inside_their_class(tmp_path):
    from spacemap_gui.studio import background

    root = _blank_project(tmp_path)
    table = root / "data/thresholds/P1_roi_1.ome.tiff_-_Merged_Image.csv"
    table.parent.mkdir(parents=True)
    found = background.detect([{"image": str(root / "outputs/rois/P1_roi_1.ome.tiff"), "root": str(root), "method": "Triangle",
                                 "annotation": str(root / "outputs/processed_annotations/stage0_tissue_annotation/P1_roi_1_valid_tissue.geojson"),
                                 "class": "tissue"}], lambda *args: None)
    # Detection keeps the levels in the project only (the markers' minimums the viewer opens with), no CSV yet.
    levels = background.stored_levels(root, "P1_roi_1.ome.tiff")
    assert found[0]["markers"] == 3 and set(levels) == {"DAPI", "CD3", "CD8"}  # the blank cycle is not levelled
    assert not table.exists() and background.images_plan(root, {}, {})["rows"][0]["detected"]
    background.export([{"image": str(root / "outputs/rois/P1_roi_1.ome.tiff"), "table": str(table), "csv": True, "levels": levels}], lambda *args: None)
    assert background.read_levels(table) == pytest.approx(levels, abs=1e-3)  # Export writes Tab 2's table
    with pytest.raises(ValueError):
        background.class_region(root / "outputs/processed_annotations/stage0_tissue_annotation/P1_roi_1_valid_tissue.geojson", "tumor")
