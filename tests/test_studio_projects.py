"""The Project step's projects: listing, opening from Studio (classic follows), a new project, notes."""

import json

import pytest

from spacemap_gui.studio import projects
from spacemap_gui.viewer import registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway home, so the user's real project list is never touched."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _project(base, name, **extra):
    root = base / name
    (root / "outputs" / "batch_corrected").mkdir(parents=True)
    (root / f"{name}.json").write_text(json.dumps({"project_name": name, "created_at": "2026-09-26T10:00:00",
                                                   "directory_mappings": {}, "tab_configurations": {}, **extra}))
    return root


def test_open_a_listed_project_and_classic_follows_once(home):
    from spacemap_gui.persistence import load_active_session, save_project_to_index

    first = _project(home, "First", unified_csv_filename="cells.csv")
    (first / "outputs/batch_corrected/cells.csv").write_text("x\n1\n")
    second = _project(home, "Second")
    save_project_to_index("First", str(first))
    save_project_to_index("Second", str(second))
    save_project_to_index("Gone", str(home / "nowhere"))
    channel = "projects-test"
    registry.publish(channel, {"layers": []}, project={"root": str(second)})
    listed = {item["name"]: item for item in projects.known(second)}
    assert listed["Second"]["open"] and not listed["First"]["open"] and listed["Gone"]["missing"]

    done = projects.open_project(channel, name="First")
    project = registry.session_project(channel)
    assert done["root"] == str(first.resolve()) and project["root"] == str(first.resolve())
    assert project["unified"] == str((first / "outputs/batch_corrected/cells.csv").resolve())
    assert registry.session_view(channel, mark_seen=False)["spec"]["layers"] == []  # the old image is closed
    assert load_active_session() == "First"
    assert registry.take_choice(channel) == str(first.resolve()) and registry.take_choice(channel) is None

    with pytest.raises(ValueError, match="not found"):
        projects.open_project(channel, name="Gone")
    with pytest.raises(ValueError, match="no project settings"):
        projects.open_project(channel, folder=str(home))
    projects.forget("Gone")
    assert "Gone" not in {item["name"] for item in projects.known()}


def test_open_a_folder_adds_it_and_a_new_project_opens(home):
    channel = "projects-test-2"
    registry.publish(channel, {"layers": []})
    colleague = _project(home / "colleague", "Theirs")
    projects.open_project(channel, folder=str(colleague))
    assert "Theirs" in {item["name"] for item in projects.known()}

    with pytest.raises(ValueError, match="letters"):
        projects.create(channel, "has space", str(home))
    made = projects.create(channel, "Fresh", str(home))
    assert made["root"].endswith("project/Fresh") and registry.session_project(channel)["root"] == made["root"]
    with pytest.raises(ValueError, match="already exists"):
        projects.create(channel, "Fresh", str(home))


def test_notes_are_kept_where_classic_keeps_them(home):
    root = _project(home, "Noted")
    saved = projects.save_notes(root, {"description": "  Liver cohort  ", "pi": "Dr A"})
    config = json.loads((root / "Noted.json").read_text())
    assert config["description"] == "Liver cohort" and config["pi"] == "Dr A"
    assert saved["created"] == "2026-09-26" and saved["institution"] == ""
    with pytest.raises(ValueError):
        projects.save_notes(root, {"project_root": "/elsewhere"})
