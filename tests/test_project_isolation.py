"""Project switching must not leak one project's paths into another."""

import json

import pytest
import streamlit as st

from spacemap_gui import persistence


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """Point the interface-state cache at a throwaway file."""
    path = tmp_path / "interface_state.json"
    monkeypatch.setattr(persistence, "INTERFACE_STATE_FILE", path)
    st.session_state.clear()
    yield path
    st.session_state.clear()


def _activate(root: str | None):
    if root is None:
        st.session_state.pop("active_project_root", None)
    else:
        st.session_state["active_project_root"] = root


def test_each_project_gets_its_own_cache_section(state_file):
    _activate("/projects/A")
    persistence.update_interface_state({"global_pheno_file": "A.csv"})

    _activate("/projects/B")
    persistence.update_interface_state({"global_pheno_file": "B.csv"})

    _activate("/projects/A")
    assert persistence.load_interface_state()["global_pheno_file"] == "A.csv"
    _activate("/projects/B")
    assert persistence.load_interface_state()["global_pheno_file"] == "B.csv"


def test_a_new_project_starts_empty_rather_than_inheriting(state_file):
    """The reported bug: project B opened showing project A's unified CSV."""
    _activate("/projects/A")
    persistence.update_interface_state(
        {"global_pheno_file": "A.csv", "global_pheno_dir": "/projects/A/outputs"}
    )

    _activate("/projects/B")
    state = persistence.load_interface_state()

    assert state == {}, "a project must never see another project's cached selections"


def test_no_project_mode_keeps_its_own_shared_cache(state_file):
    _activate(None)
    persistence.update_interface_state({"global_pheno_file": "loose.csv"})

    _activate("/projects/A")
    assert persistence.load_interface_state() == {}

    _activate(None)
    assert persistence.load_interface_state()["global_pheno_file"] == "loose.csv"


def test_project_writes_never_touch_the_shared_section(state_file):
    _activate(None)
    persistence.update_interface_state({"shared": "keep me"})

    _activate("/projects/A")
    persistence.update_interface_state({"global_pheno_file": "A.csv"})

    raw = json.loads(state_file.read_text())
    assert raw["shared"] == "keep me"
    assert "global_pheno_file" not in raw, "project state must live under its own section"
    assert raw[persistence.PROJECT_STATE_SECTION]["/projects/A"]["global_pheno_file"] == "A.csv"


def test_updates_merge_within_a_project_without_dropping_keys(state_file):
    _activate("/projects/A")
    persistence.update_interface_state({"one": 1})
    persistence.update_interface_state({"two": 2})

    assert persistence.load_interface_state() == {"one": 1, "two": 2}


def test_switching_back_and_forth_is_stable(state_file):
    for _ in range(3):
        _activate("/projects/A")
        persistence.update_interface_state({"marker": "A"})
        _activate("/projects/B")
        persistence.update_interface_state({"marker": "B"})

    _activate("/projects/A")
    assert persistence.load_interface_state()["marker"] == "A"


def test_legacy_flat_state_is_not_served_to_a_loaded_project(state_file):
    """Existing installs have a flat file full of the last project's paths."""
    state_file.write_text(json.dumps({
        "global_pheno_dir": "/projects/OLD/outputs/batch_corrected",
        "global_pheno_file": "OLD.csv",
    }))

    _activate("/projects/NEW")
    assert persistence.load_interface_state() == {}

    _activate(None)
    assert persistence.load_interface_state()["global_pheno_file"] == "OLD.csv"


def test_hydrate_does_not_preserve_the_previous_unified_csv():
    """`hydrate_project_state` used to keep these keys across a project switch."""
    import inspect

    source = inspect.getsource(persistence.hydrate_project_state)
    preserved = source.split("if key in [")[1].split("]:")[0]

    for leaked in ("global_pheno_file", "global_pheno_dir", "global_spatial_file",
                   "global_visualization_file"):
        assert leaked not in preserved, f"{leaked} must be re-derived from the new project"


def test_sync_restores_the_unified_csv_filename_from_project_config():
    import inspect

    source = inspect.getsource(persistence.sync_project_state)

    assert 'config.get("unified_csv_filename"' in source
    for file_key in ("global_pheno_file", "global_spatial_file", "global_visualization_file"):
        assert file_key in source, f"{file_key} must be restored on project load"


def test_switching_projects_swaps_the_unified_csv(tmp_path, monkeypatch):
    """End-to-end reproduction of the reported bug."""
    import json

    monkeypatch.setattr(persistence, "INTERFACE_STATE_FILE", tmp_path / "state.json")
    st.session_state.clear()

    def make(name, csv_name):
        root = tmp_path / name
        (root / "outputs" / "batch_corrected").mkdir(parents=True)
        config = {
            "project_name": name,
            "project_root": str(root),
            "unified_csv_filename": csv_name,
            "directory_mappings": {
                "global_pheno_dir": str(root / "outputs" / "batch_corrected"),
            },
            "tab_configurations": {},
            "audit_log": [],
        }
        (root / f"{name}.json").write_text(json.dumps(config))
        return str(root), config

    root_a, config_a = make("ProjA", "A_unified.csv")
    root_b, config_b = make("ProjB", "B_unified.csv")

    persistence.hydrate_project_state(config_a, root_a)
    assert st.session_state["global_pheno_file"] == "A_unified.csv"
    persistence.update_interface_state({"global_pheno_file": "A_unified.csv"})

    persistence.hydrate_project_state(config_b, root_b)
    assert st.session_state["global_pheno_file"] == "B_unified.csv", "B must not show A's CSV"
    assert str(st.session_state["global_pheno_dir"]).startswith(root_b)
    assert "ProjA" not in str(st.session_state["global_pheno_dir"])

    persistence.hydrate_project_state(config_a, root_a)
    assert st.session_state["global_pheno_file"] == "A_unified.csv"
    st.session_state.clear()


def test_sidebar_saves_the_outgoing_project_before_switching():
    """Switching projects used to discard unsaved edits from the current one."""
    from pathlib import Path

    source = Path("src/spacemap_gui/app.py").read_text()
    block = source.split("# If user changed the project, load it instantly")[1].split("st.markdown(\"---\")")[0]

    save_at = block.index("save_project_callback()")
    load_at = block.index("load_project_callback(")
    assert save_at < load_at, "the outgoing project must be saved before the new one loads"
