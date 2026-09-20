from spacemap_assistant.memory import suggest_memories


def test_explicit_memory_instruction_becomes_proposed_candidate() -> None:
    suggestions = suggest_memories("Please remember that CD3 is our T-cell marker")
    assert suggestions
    assert suggestions[0].content == "CD3 is our T-cell marker."
    assert suggestions[0].kind == "marker_alias"


def test_temporary_paths_are_not_suggested_as_memory() -> None:
    suggestions = suggest_memories("I prefer this output: /tmp/one.csv")
    assert suggestions == []


def test_normal_task_request_does_not_pollute_memory() -> None:
    assert suggest_memories("Run ROI extraction on these two files.") == []
