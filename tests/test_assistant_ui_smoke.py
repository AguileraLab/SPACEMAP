from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from spacemap_gui.components.assistant_panel import (
    _online_prompt_safety_issue,
    _online_safe_context,
)


AI_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assistant_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the optional assistant on for tests that render it.

    The suite runs with SPACEMAP_ASSISTANT=off, which is what a new user gets.
    These tests are specifically about the assistant, so they opt in.
    """
    monkeypatch.setenv("SPACEMAP_ASSISTANT", "on")


def test_assistant_workspace_renders_without_exception(assistant_on: None) -> None:
    app = AppTest.from_file(
        str(AI_ROOT / "src" / "spacemap_gui" / "tabs" / "tab_ai_assistant.py"),
        default_timeout=20,
    )
    app.run()
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    # One Setup tab replaced Model & Compute, Admin Access and Settings.
    assert labels[:5] == ["Chat", "Actions", "Instructions & Safety", "Memory", "Setup"]
    for removed in ("Model & Compute", "Admin Access", "Settings", "Local model", "Provider and API key"):
        assert removed not in labels
    for removed_action_subtab in (
        "Registration",
        "ROI Extraction",
        "Artifact Tests",
        "All Other Tools",
        "Tool Coverage",
    ):
        assert removed_action_subtab not in labels
    for inside_setup in ("Answers", "Cluster", "Folder access for AI tools", "Activity and model log"):
        assert inside_setup in labels
    expanders = [expander.label for expander in app.expander]
    for section in ("Find more models on Hugging Face", "Answer with an online model instead (optional)",
                    "Storage and access", "Advanced"):
        assert section in expanders
    button_labels = [button.label for button in app.button]
    assert "New Chat" in button_labels
    assert "Retry Last Request" in button_labels
    assert "Find Useful Lessons in All Chats" in button_labels


def test_main_app_has_pipeline_and_assistant_launcher(assistant_on: None) -> None:
    app = AppTest.from_file(
        str(AI_ROOT / "src" / "spacemap_gui" / "app.py"),
        default_timeout=30,
    )
    app.run(timeout=30)
    assert not app.exception
    labels = [link.label for link in app.get("page_link")]
    assert "Workspace" in labels
    assert "Visualization" in labels
    assert "Open Full Assistant Workspace" in labels
    assert len(app.get("popover")) == 1


def test_online_context_removes_project_state_and_paths() -> None:
    safe = _online_safe_context(
        {
            "page_id": "tab3",
            "page_title": "ROI Extraction",
            "registered_sections": ["ROI extraction"],
            "purpose": "Crop regions.",
            "before_run": "Check inputs.",
            "next_step": "Segment cells.",
            "active_project_root": "/private/project",
            "current_settings": {"input": "/private/file.tiff"},
        }
    )
    assert safe["page_title"] == "ROI Extraction"
    assert "active_project_root" not in safe
    assert "current_settings" not in safe
    assert "/private" not in str(safe)


def test_online_prompt_gate_leaves_paths_for_owner_access_checks() -> None:
    assert _online_prompt_safety_issue("Review /data/study/public/file.tiff") is None
    issue = _online_prompt_safety_issue("Review this protected health information")
    assert issue is not None
    assert _online_prompt_safety_issue("Explain Leiden clustering using synthetic data") is None


def test_main_app_hides_the_assistant_when_it_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default installation shows the pipeline and no AI entry point."""
    monkeypatch.setenv("SPACEMAP_ASSISTANT", "off")
    app = AppTest.from_file(
        str(AI_ROOT / "src" / "spacemap_gui" / "app.py"),
        default_timeout=30,
    )
    app.run(timeout=30)
    assert not app.exception
    labels = [link.label for link in app.get("page_link")]
    assert "Workspace" in labels
    assert "Visualization" in labels
    assert "Open Full Assistant Workspace" not in labels
