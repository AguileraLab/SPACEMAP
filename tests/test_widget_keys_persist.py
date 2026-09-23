"""Every pipeline setting a user can change must reach the project file.

A widget's value is saved only when the widget has a ``key`` that maps to a tab
and is not classed as transient. A key outside every tab prefix (the old Tab 3
``tma_*`` sliders) is dropped on every save without any error, so the user finds
their settings gone after reloading the project. This scans the tab sources for
setting widgets with literal keys and checks each one would be saved.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import streamlit as st

from spacemap_gui.persistence import (
    SETTING_WIDGET_CALLS,
    get_tab_name_from_key,
    is_transient_key,
    update_stored_config_key,
)
from spacemap_gui.ui_utils import seed_widget_state

TABS_DIR = Path(__file__).resolve().parents[1] / "src" / "spacemap_gui" / "tabs"

# Keys that are deliberately session-only; each has a canonical saved record.
INTENTIONALLY_TRANSIENT = {
    "tab3_load_existing_canvas",          # canvas scratch state
    "annot_ta_s3_preset_choice",          # preset picker; the chosen labels are saved
    "bc_vis_file_in",                     # viewer path
    "bc_process_all_checkbox",            # copied into bc_config
    "bc_unified_batch_col_select",        # copied into bc_config
    "bc_marker_multiselect",              # copied into bc_config
    "tab7_shared_panel_name",             # tab7_marker_workspace is authoritative
    "tab7_shared_panel_editor_name",
    "tab7_shared_panel_editor",
    "global_spatial_file_empty",          # placeholder when no CSV exists
    "global_visualization_file_empty",
}
# Project-level keys stored outside tab_configurations by dedicated code.
PROJECT_LEVEL = {"threshold_dir"}


def _literal_setting_keys() -> list[tuple[str, int, str]]:
    found = []
    for path in sorted(TABS_DIR.rglob("*.py")):
        if path.name in {"legacy_cell_annotation.py", "tab0_project.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in SETTING_WIDGET_CALLS or node.func.attr in {"file_uploader", "data_editor"}:
                continue
            for keyword in node.keywords:
                if keyword.arg != "key":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.append((path.name, node.lineno, value.value))
                elif isinstance(value, ast.JoinedStr) and value.values:
                    head = value.values[0]
                    if isinstance(head, ast.Constant) and isinstance(head.value, str) and len(head.value) >= 4:
                        # Probe the literal prefix with a placeholder suffix.
                        found.append((path.name, node.lineno, head.value + "x1"))
    return found


def test_every_literal_setting_key_is_saved_with_the_project():
    lost = []
    for filename, line, key in _literal_setting_keys():
        if key in INTENTIONALLY_TRANSIENT or key in PROJECT_LEVEL:
            continue
        if get_tab_name_from_key(key) is None or is_transient_key(key):
            lost.append(f"{filename}:{line} {key}")
    assert not lost, "Setting widgets whose values are never saved:\n" + "\n".join(lost)


@pytest.mark.parametrize(
    "key, value",
    [
        ("tab3_tma_min_slide1.ome.tif", 30),
        ("tab3_tma_row_fmt_slide1.ome.tif", "1-N"),
        ("tab1_batch_codex_prefix", "CODEX_"),
        ("tab1_batch_he_suffix", "-HE"),
        ("tab2_blank_input_source", "Raw CODEX Images (QPTIFF)"),
        ("tools_edit_csv_import_src_anchors", ["cell_id", "batch_id"]),
        ("tools_edit_csv_import_tolerance", "1.0%"),
        ("seg_qc_hist_bins", 80),
    ],
)
def test_newly_keyed_settings_are_written_to_the_project(key, value):
    config = {"tab_configurations": {}}
    update_stored_config_key(config, key, value)
    stored = [settings for settings in config["tab_configurations"].values() if key in settings]
    assert stored and stored[0][key] == value


@pytest.fixture
def clean_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_seed_sets_a_first_value_and_keeps_a_valid_restored_one(clean_state):
    assert seed_widget_state("k", "a", options=["a", "b"]) == "a"
    st.session_state["k"] = "b"
    assert seed_widget_state("k", "a", options=["a", "b"]) == "b"


def test_seed_replaces_a_restored_choice_that_no_longer_exists(clean_state):
    st.session_state["k"] = "gone"
    assert seed_widget_state("k", "a", options=["a", "b"]) == "a"


def test_seed_drops_multiselect_entries_missing_from_the_new_csv(clean_state):
    st.session_state["cols"] = ["CD3", "CD999", "CD8"]
    assert seed_widget_state("cols", [], options=["CD3", "CD8"], multi=True) == ["CD3", "CD8"]


def test_logging_switches_start_on_when_a_project_is_loaded(tmp_path, monkeypatch):
    """A log box switched off in an earlier session must not stay off for good."""
    import json

    from spacemap_gui import persistence

    monkeypatch.setattr(persistence, "INTERFACE_STATE_FILE", tmp_path / "state.json")
    st.session_state.clear()
    root = tmp_path / "Proj"
    (root / "outputs").mkdir(parents=True)
    config = {
        "project_name": "Proj",
        "project_root": str(root),
        "directory_mappings": {},
        "tab_configurations": {
            "tab1_registration": {"tab1_create_log": False, "tab1_process_log_enabled": False},
            "tab7_phenotyping": {"leiden_create_log": False, "leiden_resolution": 0.8},
        },
        "audit_log": [],
    }
    (root / "Proj.json").write_text(json.dumps(config))

    persistence.hydrate_project_state(config, str(root))

    assert st.session_state["tab1_create_log"] is True
    assert st.session_state["tab1_process_log_enabled"] is True
    assert st.session_state["leiden_create_log"] is True
    assert st.session_state["leiden_resolution"] == 0.8   # ordinary settings are untouched
    st.session_state.clear()
