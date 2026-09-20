from pathlib import Path

import pytest

from spacemap_assistant.paths import (
    AI_ROOT,
    ASSISTANT_DATA_DIR,
    AssistantPathError,
    action_workspace_root,
    grant_admin_access,
    require_assistant_read_path,
    require_assistant_write_path,
    require_inside_ai_root,
    revoke_admin_access,
)


def test_read_boundary_rejects_parent_tree() -> None:
    with pytest.raises(AssistantPathError):
        require_inside_ai_root(AI_ROOT.parent / "src" / "spacemap_gui" / "app.py")


def test_read_boundary_blocks_traversal() -> None:
    with pytest.raises(AssistantPathError):
        require_inside_ai_root(AI_ROOT / "assistant_data" / ".." / ".." / "outside.txt")


def test_write_boundary_accepts_assistant_data() -> None:
    target = require_assistant_write_path(ASSISTANT_DATA_DIR / "exports" / "chat.md")
    assert target == (ASSISTANT_DATA_DIR / "exports" / "chat.md").resolve()


@pytest.mark.parametrize(
    "relative",
    ["src/spacemap_gui/app.py", "tests/test_example.py", "docs/plan.md", "models/model.gguf"],
)
def test_write_boundary_rejects_code_and_admin_areas(relative: str) -> None:
    with pytest.raises(AssistantPathError):
        require_assistant_write_path(AI_ROOT / Path(relative))


def test_write_boundary_rejects_the_installed_source_tree() -> None:
    from spacemap_assistant.paths import SOURCE_ROOT

    with pytest.raises(AssistantPathError):
        require_assistant_write_path(SOURCE_ROOT / "spacemap_gui" / "app.py")


def test_admin_can_grant_and_revoke_specific_folder(tmp_path: Path, monkeypatch) -> None:
    import spacemap_assistant.paths as paths

    policy = tmp_path / "admin_access.json"
    monkeypatch.setattr(paths, "ADMIN_ACCESS_PATH", policy)
    external = tmp_path / "external_project"
    inputs = external / "data"
    outputs = external / "outputs" / "run_1"
    inputs.mkdir(parents=True)

    with pytest.raises(AssistantPathError):
        require_assistant_read_path(inputs)
    read_grant = grant_admin_access(inputs, "read")
    assert require_assistant_read_path(inputs) == inputs.resolve()

    write_grant = grant_admin_access(external, "read_write")
    assert require_assistant_write_path(outputs) == outputs.resolve()
    assert action_workspace_root(outputs) == external.resolve()

    revoke_admin_access(read_grant["id"])
    revoke_admin_access(write_grant["id"])
    with pytest.raises(AssistantPathError):
        require_assistant_read_path(inputs)


def test_admin_cannot_grant_write_to_folder_containing_ai_source(tmp_path: Path, monkeypatch) -> None:
    import spacemap_assistant.paths as paths

    monkeypatch.setattr(paths, "ADMIN_ACCESS_PATH", tmp_path / "admin_access.json")
    with pytest.raises(AssistantPathError, match="containing the SPACEMAP"):
        grant_admin_access(AI_ROOT.parent, "read_write")
