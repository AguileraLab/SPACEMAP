"""The shell links existing tools without running pipeline code or replacing views."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from spacemap_gui.studio import shell
from spacemap_gui.viewer import api, registry


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    channel = "shell-" + tmp_path.name
    registry.publish(
        channel, {"id": "image-open", "layers": []}, project={"root": str(tmp_path / "study"), "tab": "tab7"}
    )
    return channel


def test_shell_metadata_is_safe_and_does_not_mark_a_window_open(session):
    result = shell.context(session, "/prefix")
    assert result["project"] == "study" and result["step"] == "tab7"
    assert len(result["steps"]) == 11
    assert result["steps"][8]["classic"] == "/prefix/phenotyping"  # Cell tagging (6) comes before Batch correction (7)
    assert not registry.window_open(session)
    assert "root" not in result and "mappings" not in result
    assert registry.session_view(session, mark_seen=False)["spec"]["id"] == "image-open"


def test_context_route_requires_registered_session(session):
    with TestClient(api.app) as client:
        assert client.get("/studio/missing/context").status_code == 404
        response = client.get(f"/studio/{session}/context")
        assert response.status_code == 200
        assert response.json()["steps"][0]["classic"] == "/project"


def test_empty_workspace_has_a_classic_project_link(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    registry.publish("empty-shell", {"id": "empty", "layers": []})
    result = shell.context("empty-shell")
    assert not result["has_project"] and result["steps"][0]["classic"] == "/project"


def test_launcher_preserves_existing_view_and_follows_project_changes(session, monkeypatch, tmp_path):
    from spacemap_gui.viewer import ui

    state = {ui._SESSION_KEY: session}
    monkeypatch.setattr(ui.st, "session_state", state)
    monkeypatch.setattr(api, "MOUNTED", True)
    project = {"root": str(tmp_path / "study"), "tab": "tab8"}
    monkeypatch.setattr(ui, "project_context", lambda tab: {**project, "tab": tab})
    link = ui.studio_link("tab8")
    assert "studio=1" in link and "step=tab8" in link
    assert registry.session_view(session, mark_seen=False)["spec"]["id"] == "image-open"
    assert not registry.window_open(session)
    project["root"] = str(tmp_path / "next-project")
    ui.studio_link("tab3")
    assert registry.session_view(session, mark_seen=False)["spec"]["layers"] == []
    assert registry.session_project(session)["root"] == project["root"]


def test_every_classic_step_still_exists():
    app = (Path(__file__).resolve().parents[1] / "src/spacemap_gui/app.py").read_text()
    for _, _, route, _ in shell.STEPS:
        assert f'url_path="{route}"' in app
