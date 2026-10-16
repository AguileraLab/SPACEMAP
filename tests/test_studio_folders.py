"""Each Studio step shows and changes its own folders, saved where classic SPACEMAP reads them."""

import json

import pytest

from spacemap_gui.studio import folders
from spacemap_gui.studio.settings import read_config


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "study"
    for folder in ("outputs/registered", "outputs/rois", "outputs/other_rois", "data/boundaries"):
        (root / folder).mkdir(parents=True)
    (root / "outputs/registered/P1_merged.ome.tiff").write_bytes(b"x")
    (root / "outputs/other_rois/P1_roi_1.ome.tiff").write_bytes(b"x")
    (root / "study.json").write_text(json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}))
    return root


def test_each_step_lists_the_folders_it_uses(root):
    found = folders.listing(root, "tab3")
    rows = {row["label"]: row for row in found["folders"]}
    assert rows["Images"]["folder"] == "outputs/registered" and rows["Images"]["role"] == "in"  # follows the pipeline
    assert rows["ROI images"] == {"label": "ROI images", "role": "out", "folder": "outputs/rois", "exists": True, "editable": True, "note": ""}
    assert "outputs/other_rois" in found["choices"] and not any("intermediate" in item for item in found["choices"])
    phenotyping = {row["label"]: row for row in folders.listing(root, "tab7")["folders"]}
    assert not phenotyping["ROI images"]["editable"] and "Regions of interest" in phenotyping["ROI images"]["note"]
    assert folders.listing(root, "tab0")["folders"] == []


def test_a_changed_folder_is_saved_as_classic_saves_it(root):
    found = folders.change(root, "tab6", "Combined measurements", "outputs/combined_new")
    rows = {row["label"]: row for row in found["folders"]}
    assert rows["Combined measurements"]["folder"] == "outputs/combined_new" and not rows["Combined measurements"]["exists"]
    assert read_config(root)["directory_mappings"]["bc_output_dir"] == "${PROJECT_ROOT}/outputs/combined_new"
    folders.change(root, "tab3", "ROI images", "outputs/other_rois")  # both of Tab 3's names for its output
    saved = read_config(root)["directory_mappings"]
    assert saved["tab3_output_dir"] == saved["roi_output_dir"] == "${PROJECT_ROOT}/outputs/other_rois"
    # Tab 3's own input: a chosen folder is kept (as classic Tab 3 keeps it), not replaced by the pipeline's.
    (root / "data/scans").mkdir()
    (root / "data/scans/P1.ome.tiff").write_bytes(b"x")
    folders.change(root, "tab3", "Images", "data/scans")
    assert {row["label"]: row for row in folders.listing(root, "tab3")["folders"]}["Images"]["folder"] == "data/scans"
    # Background's images are Tab 2's own choice.
    folders.change(root, "tab2", "Images", "outputs/other_rois")
    assert {row["label"]: row for row in folders.listing(root, "tab2")["folders"]}["Images"]["folder"] == "outputs/other_rois"


def test_wrong_folders_are_refused(root):
    for step, label, value in (("tab6", "Combined measurements", "../outside"), ("tab6", "Combined measurements", "/tmp/x"),
                               ("tab6", "Cell measurements", "outputs/missing"), ("tab7", "ROI images", "outputs/rois"),
                               ("tab6", "Nothing", "outputs/rois"), ("tab6", "Combined measurements", "")):
        with pytest.raises(ValueError):
            folders.change(root, step, label, value)
