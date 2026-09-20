from pathlib import Path

import pytest

from spacemap_assistant.paths import ASSISTANT_PROJECTS_DIR, is_within
from spacemap_assistant.registration_action import _validate_inputs, parse_registration_request


AI_TEST = ASSISTANT_PROJECTS_DIR / "AI_test"
CODEX = AI_TEST / "data" / "codex" / "P11_codex.qptiff"
HE = AI_TEST / "data" / "he" / "P11_he.qptiff"
OUTPUT = AI_TEST / "outputs" / "AI_registered"


def test_a_symlinked_alias_resolves_into_the_same_project_area(tmp_path: Path) -> None:
    """A site that exposes one share under two prefixes must not confuse the guard.

    The original version of this test relied on a symlink that existed only on
    one cluster. Creating the alias here keeps the behaviour covered anywhere.
    """
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(ASSISTANT_PROJECTS_DIR, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This filesystem does not support symbolic links.")
    work_form = alias / CODEX.relative_to(ASSISTANT_PROJECTS_DIR)
    assert work_form.resolve() == CODEX.resolve()
    assert is_within(work_form, ASSISTANT_PROJECTS_DIR)
    codex, he, output, project = _validate_inputs(work_form, HE, OUTPUT)
    assert codex == CODEX.resolve()
    assert he == HE.resolve()
    assert output == OUTPUT.resolve()
    assert project == AI_TEST.resolve()


def test_natural_language_registration_request_extracts_exact_paths() -> None:
    prompt = f"Register {CODEX} along with {HE} and save it within {OUTPUT}"
    assert parse_registration_request(prompt) == {
        "codex_path": str(CODEX),
        "he_path": str(HE),
        "output_dir": str(OUTPUT),
        "overwrite": False,
    }


def test_overwrite_requires_clear_words_in_the_request() -> None:
    prompt = f"Register {CODEX} with {HE}; save in {OUTPUT} and overwrite the existing result"
    parsed = parse_registration_request(prompt)
    assert parsed is not None
    assert parsed["overwrite"] is True
