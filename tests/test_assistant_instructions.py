import pytest

import spacemap_assistant.instructions as instruction_module
from spacemap_assistant.instructions import (
    CORE_SAFETY_POLICY,
    InstructionProfile,
    build_system_prompt,
)


def test_prompt_keeps_fixed_policy_above_editable_instructions() -> None:
    profile = InstructionProfile(
        positive="Use simple words.",
        negative="Do not guess.",
        updated_at="2026-01-01T00:00:00Z",
    )
    prompt = build_system_prompt(
        profile,
        page_context='{"page_title": "Phenotyping"}',
        active_visual_context='{"channels": ["DAPI", "CD45"]}',
        verified_memories="- CD3 is a T-cell marker in panel A.",
    )
    assert prompt.startswith("FIXED SPACEMAP SAFETY POLICY")
    assert CORE_SAFETY_POLICY in prompt
    assert "Use simple words." in prompt
    assert "Do not guess." in prompt
    assert "Phenotyping" in prompt
    assert "CD3" in prompt
    assert "DAPI" in prompt
    assert "Preserve every unspecified field" in prompt
    assert prompt.index(CORE_SAFETY_POLICY) < prompt.index("Use simple words.")


def test_instruction_length_is_bounded() -> None:
    with pytest.raises(ValueError):
        InstructionProfile(positive="x" * 20_001).normalized()


def test_advisory_prompt_does_not_invite_tools_or_jobs() -> None:
    profile = InstructionProfile(updated_at="2026-01-01T00:00:00Z")
    prompt = build_system_prompt(
        profile,
        page_context='{"section": "pixel_classification"}',
        source_context="irrelevant_source_function()",
        registered_tools="irrelevant.figure.tool",
        allow_action_plan=False,
    )
    assert "CURRENT REQUEST MODE" in prompt
    assert "Do not emit a job" in prompt
    assert "irrelevant_source_function" not in prompt
    assert "irrelevant.figure.tool" not in prompt
    assert "PROTECTED GENERAL JOB FORMAT" not in prompt


def test_instruction_profile_is_saved_with_version_history(monkeypatch) -> None:
    # Use a temporary folder beneath assistant_data so the real user profile is untouched.
    local_dir = instruction_module.ASSISTANT_DATA_DIR / "test_instruction_storage"
    local_dir.mkdir(parents=True, exist_ok=True)
    profile_path = local_dir / "instructions.json"
    history_path = local_dir / "history.jsonl"
    monkeypatch.setattr(instruction_module, "INSTRUCTIONS_PATH", profile_path)
    monkeypatch.setattr(instruction_module, "INSTRUCTION_HISTORY_PATH", history_path)
    try:
        saved = instruction_module.save_instruction_profile("Be clear.", "Do not guess.")
        loaded = instruction_module.load_instruction_profile()
        assert loaded == saved
        assert loaded.positive == "Be clear."
        assert loaded.negative == "Do not guess."
        assert history_path.read_text(encoding="utf-8").count("\n") == 1
    finally:
        history_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)
        local_dir.rmdir()
