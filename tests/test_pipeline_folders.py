"""Tab 3's input follows the pipeline; a Tab 2 output linked to the ROI folder is repaired on load."""

from __future__ import annotations

import json
import os

from spacemap_gui import persistence
from spacemap_gui.pipeline_folders import follow_pipeline, unlink_step_outputs


def _project(tmp_path):
    folders = {name: tmp_path / "outputs" / name for name in ("registered", "background_subtracted", "rois")}
    for folder in folders.values():
        folder.mkdir(parents=True)
    mappings = {"tab1_batch_output_dir": str(folders["registered"]),
                "tab2_output_dir": str(folders["background_subtracted"]),
                "bg_subtraction_output_dir": str(folders["background_subtracted"]),
                "tab3_output_dir": str(folders["rois"])}
    return folders, mappings


def test_tab3_reads_the_last_step_that_made_images(tmp_path):
    folders, mappings = _project(tmp_path)
    (folders["rois"] / "s1r1.ome.tiff").write_bytes(b"")
    # No background subtraction yet: the registered images, also instead of Tab 3's own output.
    assert follow_pipeline(str(folders["rois"]), mappings)[0] == str(folders["registered"])
    assert follow_pipeline("", mappings)[0] == str(folders["registered"])
    # Background subtraction was run: its images.
    (folders["background_subtracted"] / "slide1.ome.tiff").write_bytes(b"")
    value, note = follow_pipeline(str(folders["registered"]), mappings)
    assert value == str(folders["background_subtracted"]) and "Tab 2" in note
    # A folder chosen by hand is kept.
    mine = tmp_path / "elsewhere"
    mine.mkdir()
    assert follow_pipeline(str(mine), mappings) == (str(mine), "")


def test_a_tab2_output_linked_to_the_roi_folder_is_set_back_on_load(tmp_path):
    root = tmp_path / "PDAC"
    root.mkdir()
    config = {"project_name": "PDAC", "project_root": str(root.resolve()),
              "directory_mappings": {"tab2_output_dir": "${PROJECT_ROOT}/outputs/rois",
                                     "bg_subtraction_output_dir": "${PROJECT_ROOT}/outputs/rois",
                                     "tab3_input_dir": "${PROJECT_ROOT}/outputs/rois",
                                     "tab3_output_dir": "${PROJECT_ROOT}/outputs/rois"},
              "portable_state": {"interface_state": {"tab2_output_dir": "${PROJECT_ROOT}/outputs/rois",
                                                     "tab3_output_dir": "${PROJECT_ROOT}/outputs/rois"}}}
    assert unlink_step_outputs(json.loads(json.dumps(config)), str(root)) == "${PROJECT_ROOT}/outputs/background_subtracted"
    (root / "PDAC.json").write_text(json.dumps(config))
    loaded = persistence.load_project_config(str(root))
    assert loaded["directory_mappings"]["tab2_output_dir"] == "${PROJECT_ROOT}/outputs/background_subtracted"
    assert loaded["directory_mappings"]["bg_subtraction_output_dir"] == "${PROJECT_ROOT}/outputs/background_subtracted"
    assert loaded["portable_state"]["interface_state"]["tab2_output_dir"] == "${PROJECT_ROOT}/outputs/background_subtracted"
    assert loaded["directory_mappings"]["tab3_output_dir"] == "${PROJECT_ROOT}/outputs/rois"  # untouched
    saved = json.loads((root / "PDAC.json").read_text())
    assert saved["directory_mappings"]["tab2_output_dir"].endswith("outputs/background_subtracted")
    assert saved["audit_log"][-1]["parameter_changed"] == "tab2_output_dir"


def test_two_spellings_of_one_folder_are_the_same_setting(tmp_path):
    real = tmp_path / "storage" / "work" / "project"
    real.mkdir(parents=True)
    alias = tmp_path / "work"
    os.symlink(tmp_path / "storage" / "work", alias)  # a storage alias and its resolved path
    assert persistence.same_folder_setting(str(alias / "project"), str(real))
    assert not persistence.same_folder_setting(str(alias / "project"), str(tmp_path))
    assert persistence.same_folder_setting(40, 40.0) and not persistence.same_folder_setting("a", "")


def test_changing_tab3_input_no_longer_moves_tab2_output(monkeypatch, tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace

    folders, mappings = _project(tmp_path)
    config = {"directory_mappings": dict(mappings, tab3_input_dir=mappings["tab1_batch_output_dir"],
                                         tab3_multi_input_dir=mappings["tab1_batch_output_dir"]),
              "tab_configurations": {}, "audit_log": []}
    state = {"active_project": config, "active_project_root": str(tmp_path), "_active_tab_session": "tab3",
             "_currently_rendering_tab": "tab3", "_project_just_hydrated": False,
             "tab3_multi_input_dir": str(folders["rois"])}
    monkeypatch.setattr(persistence, "st", SimpleNamespace(session_state=state))
    saved = []
    monkeypatch.setattr(persistence, "save_project_config", lambda root, cfg: saved.append(deepcopy(cfg)))
    monkeypatch.setattr(persistence, "log_persistence", lambda _message: None)
    persistence.sync_project_state()
    assert config["directory_mappings"]["tab3_input_dir"] == str(folders["rois"])  # the linked Tab 3 key follows
    assert config["directory_mappings"]["tab2_output_dir"] == str(folders["background_subtracted"])  # Tab 2 does not
