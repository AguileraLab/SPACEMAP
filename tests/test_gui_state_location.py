"""Project discovery must persist inside the configured application state root."""

import json
from pathlib import Path

from spacemap_gui import persistence


def test_explicit_home_keeps_project_registry_isolated(tmp_path, monkeypatch):
    home, state = tmp_path / "home", tmp_path / "chosen"
    legacy = home / ".spacemap_gui"
    legacy.mkdir(parents=True)
    (legacy / "projects.json").write_text('{"OldStudy": "/old/study"}')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SPACEMAP_HOME", str(state))
    monkeypatch.delenv("SPACEMAP_STATE_DIR", raising=False)
    assert persistence.load_projects_index() == {}
    persistence.save_project_to_index("NewStudy", str(tmp_path / "project"))
    persistence.save_active_session("NewStudy")
    assert persistence.load_active_session() == "NewStudy"
    assert persistence.get_projects_index_path() == state / "state" / "projects.json"
    assert persistence.get_active_session_path() == state / "state" / "active_session.json"
    assert json.loads((legacy / "projects.json").read_text()) == {"OldStudy": "/old/study"}
    assert not (legacy / "active_session.json").exists()


def test_default_home_preserves_legacy_registry_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("SPACEMAP_HOME", "SPACEMAP_STATE_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)
    legacy = tmp_path / ".spacemap_gui"
    legacy.mkdir()
    (legacy / "projects.json").write_text('{"OldStudy": "/old/study"}')
    (legacy / "active_session.json").write_text('{"active_project_name": "OldStudy"}')
    assert persistence.load_projects_index() == {"OldStudy": "/old/study"}
    assert persistence.load_active_session() == "OldStudy"
    persistence.save_project_to_index("NewStudy", str(tmp_path / "project"))
    assert set(persistence.load_projects_index()) == {"OldStudy", "NewStudy"}
    assert (legacy / "projects.json").exists()
