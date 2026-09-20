from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import spacemap_gui.persistence as persistence
from spacemap_gui.persistence import (
    ensure_threshold_directory_mapping,
    get_tab_affected_from_key,
    update_stored_config_key,
)
from spacemap_gui.tabs.tab0_project import (
    DIRECTORY_KEYS,
    get_default_directory_mappings,
)


def _install_fake_streamlit(monkeypatch, session_state):
    """Run persistence synchronization without a Streamlit runtime or disk writes."""
    saved = []
    monkeypatch.setattr(
        persistence,
        "st",
        SimpleNamespace(session_state=session_state),
    )
    monkeypatch.setattr(
        persistence,
        "save_project_config",
        lambda root, config: saved.append((str(root), deepcopy(config))),
    )
    monkeypatch.setattr(persistence, "log_persistence", lambda _message: None)
    return saved


def _active_project_config(threshold_dir):
    return {
        "directory_mappings": {"threshold_dir": str(threshold_dir)},
        "tab_configurations": {},
        "audit_log": [],
    }


def test_threshold_directory_is_a_project_directory_mapping(tmp_path):
    mappings = get_default_directory_mappings(
        str(tmp_path),
        str(tmp_path / "data" / "codex"),
        str(tmp_path / "data" / "he"),
    )

    assert "threshold_dir" in DIRECTORY_KEYS
    assert mappings["threshold_dir"] == str(tmp_path / "data" / "thresholds")


def test_threshold_directory_mapping_defaults_inside_project(tmp_path):
    config = {}

    value, changed = ensure_threshold_directory_mapping(config, str(tmp_path))

    assert changed is True
    assert value == str(tmp_path / "data" / "thresholds")
    assert config["directory_mappings"]["threshold_dir"] == value


def test_threshold_directory_mapping_preserves_populated_legacy_folder(tmp_path):
    legacy = tmp_path / "data" / "threshold"
    legacy.mkdir(parents=True)
    (legacy / "sample.csv").write_text("Marker,Polygon_1\nCD8,1\n")

    value, changed = ensure_threshold_directory_mapping({}, str(tmp_path))

    assert changed is True
    assert value == str(legacy)


def test_threshold_directory_mapping_migrates_legacy_batch_widget(tmp_path):
    legacy = tmp_path / "custom" / "thresholds"
    config = {
        "directory_mappings": {},
        "tab_configurations": {
            "tab2_background_subtraction": {
                "tab2_bgdet_batch_threshold_dir": str(legacy),
            },
            "tab7_phenotyping": {
                "resolve_threshold_dir": str(tmp_path / "older-resolve-choice"),
            },
        },
    }

    value, changed = ensure_threshold_directory_mapping(config, str(tmp_path))

    assert changed is True
    assert value == str(legacy)
    assert config["directory_mappings"]["threshold_dir"] == str(legacy)
    assert "tab2_bgdet_batch_threshold_dir" not in config["tab_configurations"]["tab2_background_subtraction"]
    assert "resolve_threshold_dir" not in config["tab_configurations"]["tab7_phenotyping"]


def test_canonical_threshold_directory_wins_over_legacy_values(tmp_path):
    canonical = tmp_path / "data" / "thresholds"
    config = {
        "directory_mappings": {"threshold_dir": str(canonical)},
        "tab_configurations": {
            "tab2_background_subtraction": {
                "tab2_bgdet_batch_threshold_dir": str(tmp_path / "legacy"),
            },
        },
    }

    value, changed = ensure_threshold_directory_mapping(config, str(tmp_path))

    assert changed is True
    assert value == str(canonical)
    assert "tab2_bgdet_batch_threshold_dir" not in config["tab_configurations"]["tab2_background_subtraction"]


def test_migration_removes_hot_session_aliases(monkeypatch, tmp_path):
    aliases = {
        "tab2_bgdet_batch_threshold_dir": "batch-old",
        "tab2_bgdet_threshold_dir": "single-old",
        "resolve_threshold_dir": "resolve-old",
    }
    monkeypatch.setattr(
        persistence,
        "st",
        SimpleNamespace(session_state=dict(aliases)),
    )
    config = {
        "directory_mappings": {
            "threshold_dir": str(tmp_path / "data" / "thresholds")
        }
    }

    ensure_threshold_directory_mapping(config, str(tmp_path))

    assert not (set(aliases) & set(persistence.st.session_state))


def test_threshold_directory_is_cross_tab_directory_setting():
    assert get_tab_affected_from_key("threshold_dir") == "Directories"


def test_threshold_directory_is_serialized_under_directory_mappings(tmp_path):
    config = {}
    threshold_dir = Path(tmp_path) / "data" / "thresholds"

    update_stored_config_key(config, "threshold_dir", str(threshold_dir))

    assert config == {
        "directory_mappings": {"threshold_dir": str(threshold_dir)}
    }


def _assert_global_threshold_edit_syncs(monkeypatch, tmp_path, active_tab):
    original = tmp_path / "data" / "thresholds"
    changed = tmp_path / "chosen" / active_tab / "thresholds"
    config = _active_project_config(original)
    session_state = {
        "active_project": config,
        "active_project_root": str(tmp_path),
        "_active_tab_session": active_tab,
        "_currently_rendering_tab": active_tab,
        "_project_just_hydrated": False,
        "threshold_dir": str(changed),
        "proj_dir_threshold_dir": str(original),
    }
    saved = _install_fake_streamlit(monkeypatch, session_state)

    persistence.sync_project_state()

    assert config["directory_mappings"]["threshold_dir"] == str(changed)
    assert session_state["threshold_dir"] == str(changed)
    assert session_state["proj_dir_threshold_dir"] == str(changed)
    assert saved
    assert saved[-1][1]["directory_mappings"]["threshold_dir"] == str(changed)


def test_tab2_threshold_directory_edit_syncs_to_project_mapping(monkeypatch, tmp_path):
    _assert_global_threshold_edit_syncs(monkeypatch, tmp_path, "tab2")


def test_resolve_threshold_directory_edit_syncs_to_project_mapping(monkeypatch, tmp_path):
    _assert_global_threshold_edit_syncs(monkeypatch, tmp_path, "tab7")


def test_workspace_threshold_directory_edit_syncs_to_global_key(monkeypatch, tmp_path):
    original = tmp_path / "data" / "thresholds"
    changed = tmp_path / "workspace-choice" / "thresholds"
    config = _active_project_config(original)
    session_state = {
        "active_project": config,
        "active_project_root": str(tmp_path),
        "_active_tab_session": "tab0",
        "_currently_rendering_tab": "tab0",
        "_project_just_hydrated": False,
        "threshold_dir": str(original),
        "proj_dir_threshold_dir": str(changed),
    }
    saved = _install_fake_streamlit(monkeypatch, session_state)

    persistence.sync_project_state()

    assert config["directory_mappings"]["threshold_dir"] == str(changed)
    assert session_state["threshold_dir"] == str(changed)
    assert session_state["proj_dir_threshold_dir"] == str(changed)
    assert saved
    assert saved[-1][1]["directory_mappings"]["threshold_dir"] == str(changed)


def test_project_hydration_restores_each_projects_threshold_directory(monkeypatch, tmp_path):
    project_a_root = tmp_path / "project-a"
    project_b_root = tmp_path / "project-b"
    threshold_a = project_a_root / "data" / "thresholds-a"
    threshold_b = project_b_root / "data" / "thresholds-b"
    config_a = _active_project_config(threshold_a)
    config_a["project_name"] = "project-a"
    config_b = _active_project_config(threshold_b)
    config_b["project_name"] = "project-b"
    session_state = {
        "active_project": config_a,
        "active_project_root": str(project_a_root),
        "active_project_name": "project-a",
        "threshold_dir": str(threshold_a),
        "proj_dir_threshold_dir": str(threshold_a),
    }
    _install_fake_streamlit(monkeypatch, session_state)

    persistence.hydrate_project_state(config_b, str(project_b_root))

    assert session_state["active_project"] is config_b
    assert session_state["threshold_dir"] == str(threshold_b)
    assert session_state["proj_dir_threshold_dir"] == str(threshold_b)

    persistence.hydrate_project_state(config_a, str(project_a_root))

    assert session_state["active_project"] is config_a
    assert session_state["threshold_dir"] == str(threshold_a)
    assert session_state["proj_dir_threshold_dir"] == str(threshold_a)
