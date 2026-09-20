"""Shared Leiden/SOM/Vortex marker rules and the empty-signature UI regression."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from spacemap.ph_accel_utils import (
    parse_phenotype_signatures,
    recommend_phenotypes,
    signatures_to_text,
)
from spacemap_gui.tabs.tab7_phenotyping.common import DEFAULT_SIGNATURE_TEMPLATE


def test_default_parses_all_eleven_user_rules_without_flattening():
    rules, problems = parse_phenotype_signatures(
        DEFAULT_SIGNATURE_TEMPLATE, return_problems=True
    )
    assert problems == []
    assert list(rules) == [
        "Fibroblast",
        "IMC",
        "DC",
        "Macrophage",
        "Endothelial",
        "B",
        "Epithelial",
        "Plasma",
        "Treg",
        "CD8",
        "CD4",
    ]
    assert rules["Fibroblast"]["alternatives"] == [
        ["FN1"],
        ["Vimentin"],
        ["PDGFRa"],
        ["Podoplanin"],
        ["aSMA"],
    ]
    assert len(rules["Macrophage"]["alternatives"]) == 7
    assert rules["Treg"]["alternatives"][0] == ["FoxP3", "CD4", "CD3e", "CD45"]
    assert all(rule["support"] == rule["excluded"] == [] for rule in rules.values())
    assert parse_phenotype_signatures(signatures_to_text(rules)) == rules


def test_legacy_and_grouped_rules_can_be_mixed_and_roundtripped():
    rules, problems = parse_phenotype_signatures(
        "# comment\nB: CD20+, CD3e-, !CD8\nEpithelial = [Pan-Cytokeratin] OR [E-cadherin]; NOT=[CD45]",
        return_problems=True,
    )
    assert problems == []
    assert rules["B"] == (["CD20"], ["CD3e", "CD8"])
    assert rules["Epithelial"]["excluded"] == ["CD45"]
    assert parse_phenotype_signatures(signatures_to_text(rules)) == rules


def test_bad_lines_are_reported_without_dropping_valid_lines():
    rules, problems = parse_phenotype_signatures(
        "# comment\nA = [CD20\nB: CD20\nno separator\nb = [CD8]",
        return_problems=True,
    )
    assert list(rules) == ["B"]
    assert len(problems) == 3
    assert problems[0].startswith("line 2:")
    assert problems[1].startswith("line 4:")
    assert "duplicate phenotype" in problems[2]


def test_or_uses_best_alternative_and_and_requires_every_marker():
    profiles = pd.DataFrame(
        {"A_corr_norm": [3, 3, -2], "b_norm": [-2, 2, -1], "C": [1, -5, -3]}
    )
    rules = parse_phenotype_signatures("Type = [A, B] OR [C]")
    result = recommend_phenotypes(profiles, rules, propagate=False)
    assert result.loc[0, "phenotype"] == "Type"  # Alternative C wins.
    assert result.loc[0, "confidence"] == 1
    assert result.loc[1, "confidence"] == 2  # AND uses weaker marker B.
    assert pd.isna(result.loc[2, "phenotype"])


def test_support_and_not_do_not_rescue_failed_required_groups():
    profiles = pd.DataFrame({"A": [2, 2, -1], "S": [4, 100, 100], "N": [-1, 3, -1]})
    rules = parse_phenotype_signatures("Type = [A]; SUPPORT=[S]; NOT=[N]")
    result = recommend_phenotypes(profiles, rules, propagate=False)
    assert result.loc[0, "confidence"] == 3
    assert pd.isna(result.loc[1, "phenotype"])
    assert pd.isna(result.loc[2, "phenotype"])


def test_missing_markers_never_weaken_required_or_excluded_rules():
    profiles = pd.DataFrame({"A": [5, 5], "C": [-1, 2]})
    rules = parse_phenotype_signatures(
        "Type = [A, Missing] OR [C]; SUPPORT=[Optional]\nExcluded = [A]; NOT=[Missing]"
    )
    result = recommend_phenotypes(profiles, rules, propagate=False)
    assert pd.isna(result.loc[0, "phenotype"])
    assert result.loc[1, "phenotype"] == "Type"
    assert result.attrs["unresolved_markers"] == ["Missing", "Optional"]
    assert result.attrs["scored_phenotypes"] == ["Type"]


def test_legacy_mean_and_negative_scores_are_unchanged():
    result = recommend_phenotypes(
        pd.DataFrame({"CD20": [3], "CD45": [1], "CD3e": [-1]}),
        parse_phenotype_signatures("B: CD20, CD45, CD3e-"),
        propagate=False,
    )
    assert result.loc[0, "phenotype"] == "B"
    assert result.loc[0, "confidence"] == 3


def panel_app(prefix, reference=False):
    script = f"""
import numpy as np
from spacemap_gui.tabs.tab7_phenotyping.common import render_phenotype_recommendation_panel
scores = {{0: {{'CD20': 3., 'CD3e': -1., 'CD45': 2.}},
          1: {{'CD20': -1., 'CD3e': 3., 'CD45': 2.}}}}
extra = {{}}
if {reference!r}:
    extra = dict(reference_matrix=np.array([[3., -1., 2.], [-1., 3., 2.]]),
                 reference_marker_names=['CD20', 'CD3e', 'CD45'],
                 reference_label_options={{'manual': np.array(['B', 'T'])}})
render_phenotype_recommendation_panel({prefix!r}, scores, **extra)
"""
    return AppTest.from_string(script, default_timeout=30).run()


@pytest.mark.parametrize("prefix", ["leiden", "som", "vortex"])
def test_all_three_panels_accept_defaults_and_handle_empty_or_invalid_text(prefix):
    app = panel_app(prefix)
    assert not app.exception
    assert app.text_area[0].value == DEFAULT_SIGNATURE_TEMPLATE
    assert any("11 signature(s) parsed" in caption.value for caption in app.caption)
    app.text_area[0].set_value("").run()
    assert not app.exception  # Previously use_profile was referenced before assignment.
    assert any("Add at least one valid marker rule" in info.value for info in app.info)
    app.text_area[0].set_value("Type = [CD20").run()
    assert not app.exception
    assert any(
        "Some marker rules were skipped" in warning.value for warning in app.warning
    )
    app.button(key=f"{prefix}_reco_example").click().run()
    assert not app.exception
    assert app.text_area[0].value == DEFAULT_SIGNATURE_TEMPLATE
    app.button(key=f"{prefix}_reco_run").click().run()
    assert not app.exception
    table = app.session_state[f"{prefix}_reco_table"]
    assert table.loc[0, "phenotype"] == "B"


@pytest.mark.parametrize("prefix", ["leiden", "som", "vortex"])
def test_empty_signatures_do_not_block_learning_from_a_reference(prefix):
    app = panel_app(prefix, reference=True)
    assert not app.exception
    app.text_area[0].set_value("").run()
    assert not app.exception
    assert app.radio[0].value.startswith("Learn from my annotated cells")
    assert app.button(key=f"{prefix}_reco_run") is not None
    assert not any(
        "Add at least one valid marker rule" in info.value for info in app.info
    )


def test_saved_custom_text_is_not_replaced_by_the_new_default():
    app = panel_app("leiden")
    app.text_area[0].set_value("Custom: CD20, CD3e-").run()
    app.run()
    assert not app.exception
    assert app.text_area[0].value == "Custom: CD20, CD3e-"
