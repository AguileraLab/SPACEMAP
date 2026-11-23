"""Opening a project whose folder was moved must not loop, and should find the move."""

from __future__ import annotations

import json

from spacemap_gui import persistence


def _project(parent, name):
    root = parent / name
    root.mkdir(parents=True)
    (root / f"{name}.json").write_text(json.dumps({"project_name": name}))
    return root


def test_settings_file_is_found_or_reported_missing(tmp_path):
    assert persistence.project_config_path(_project(tmp_path, "A")).name == "A.json"
    assert persistence.project_config_path(tmp_path / "gone") is None


def test_a_moved_project_is_found_beside_the_others(tmp_path):
    new_home = tmp_path / "new" / "project"
    example_a = _project(new_home, "ExampleA")
    example_b = _project(new_home, "ExampleB")
    index = {"ExampleA": str(example_a), "ExampleB": str(tmp_path / "old" / "projects" / "ExampleB")}
    assert persistence.find_moved_project("ExampleB", index) == example_b
    assert persistence.find_moved_project("NOWHERE", index) is None


def test_loading_a_missing_project_says_so(tmp_path):
    from spacemap_gui.tabs.tab0_project import load_project_callback

    assert load_project_callback(str(tmp_path / "gone")) is False
