"""Annotation-tab button keys must never be written into a project file.

Streamlit refuses to create a widget whose key already holds a value assigned
from code. A button key that gets saved with the project and restored on load
therefore does not merely add noise: the tab stops rendering, with an error that
names the widget rather than the persistence layer that broke it. A button that
has since been removed is the worst case, because the key outlives the widget.

If this test fails after you add a button, extend the rule in
``persistence.is_transient_key`` rather than renaming the button.
"""

import ast
import pathlib
import re

import pytest

from spacemap_gui.persistence import is_transient_key

TAB = pathlib.Path(__file__).resolve().parents[1] / "src/spacemap_gui/tabs/tab4_annotation"

# Widget keys that are genuine reproducibility settings and must survive a save.
SETTINGS_KEYS = [
    "annot_ta_input_dir",
    "annot_ta_output_dir",
    "annot_ta_s1_codex_markers",
    "annot_ta_s1_form",
    "annot_ta_s1_overlay",
    "annot_ta_s1_overlay_clot",
    "annot_ta_s2_form",
    "annot_ta_s2_layers",
    "annot_ta_s2_markers",
    "annot_ta_s2b_form",
    "annot_ta_s2b_source",
    "annot_ta_s3_form",
    "annot_ta_s3_markers",
    "annot_ta_s3_layers",
    "annot_ta_artifact_config",
    "annot_ta_tissue_config",
]

# Buttons that existed in earlier builds. Their keys are still sitting in saved
# projects, so the rule has to keep matching them long after the widget is gone.
RETIRED_BUTTON_KEYS = [
    "annot_ta_s2_preset_Cautious",
    "annot_ta_s2_preset_Balanced",
    "annot_ta_s2_preset_Thorough",
]


def _button_keys() -> set[str]:
    """Every key passed to a .button(...) call anywhere in the tab."""
    keys: set[str] = set()
    for path in TAB.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (isinstance(function, ast.Attribute) and function.attr == "button"):
                continue
            for keyword in node.keywords:
                if keyword.arg == "key" and isinstance(keyword.value, ast.Constant):
                    keys.add(str(keyword.value.value))
                elif keyword.arg == "key" and isinstance(keyword.value, ast.JoinedStr):
                    # An f-string key such as a per-preset button: keep the
                    # literal prefix, which is what the rule has to match.
                    literal = "".join(
                        part.value for part in keyword.value.values
                        if isinstance(part, ast.Constant)
                    )
                    keys.add(literal + "X")
    return keys


def test_the_tab_actually_defines_buttons():
    assert len(_button_keys()) >= 10, "the scanner found no buttons, so it is not testing anything"


@pytest.mark.parametrize("key", sorted(_button_keys()))
def test_every_button_key_is_treated_as_transient(key):
    assert is_transient_key(key), (
        f"{key} would be saved into the project file and restored into session state, "
        "which makes Streamlit refuse to draw the button and takes the tab down with it"
    )


@pytest.mark.parametrize("key", RETIRED_BUTTON_KEYS)
def test_retired_button_keys_are_still_pruned(key):
    assert is_transient_key(key)


@pytest.mark.parametrize("key", SETTINGS_KEYS)
def test_real_settings_are_not_pruned(key):
    assert not is_transient_key(key), f"{key} is a setting and must persist with the project"
