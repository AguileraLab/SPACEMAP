from spacemap_gui.resolve_rule_parser import parse_resolve_rule_text


MARKERS = [
    "CD3e", "CD4", "CD8", "CD11b", "CD11c", "CD15", "CD20", "CD31",
    "CD38", "CD45", "CD68", "CD163", "E-cadherin", "FN1", "FoxP3",
    "HLA-DR", "Pan-Cytokeratin", "PDGFRa", "Podoplanin", "S100A9",
    "Vimentin", "aSMA",
]


def test_bracket_syntax_preserves_and_or_support_and_not():
    rows, problems = parse_resolve_rule_text(
        "CD8 = ['CD8', 'CD3e', 'CD45'] OR ['CD8', 'CD3e']; "
        "SUPPORT=[]; NOT=['E-cadherin']",
        MARKERS,
    )

    assert problems == []
    assert rows == [{
        "Phenotype": "CD8",
        "alternatives": [["CD8", "CD3e", "CD45"], ["CD8", "CD3e"]],
        "support": [],
        "excluded": ["E-cadherin"],
    }]


def test_comma_separated_bracket_groups_and_parenthesized_fields_work():
    rows, problems = parse_resolve_rule_text(
        "CD8 = [CD8, CD3e, CD45], [CD8, CD3e], "
        "(SUPPORT = ['']), (NOT = [E-cadherin])",
        MARKERS,
    )

    assert problems == []
    assert rows[0]["alternatives"] == [["CD8", "CD3e", "CD45"], ["CD8", "CD3e"]]
    assert rows[0]["support"] == []
    assert rows[0]["excluded"] == ["E-cadherin"]


def test_optional_fields_work_in_either_order():
    rows, problems = parse_resolve_rule_text(
        "CD8 = [CD8]; NOT=[CD4]; SUPPORT=[CD45]",
        MARKERS,
    )

    assert problems == []
    assert rows[0]["support"] == ["CD45"]
    assert rows[0]["excluded"] == ["CD4"]


def test_old_and_or_syntax_remains_supported():
    rows, problems = parse_resolve_rule_text(
        "CD8 = CD8 AND CD3e OR CD8 AND CD45 SUPPORT CD45 NOT CD4",
        MARKERS,
    )

    assert problems == []
    assert rows[0] == {
        "Phenotype": "CD8",
        "alternatives": [["CD8", "CD3e"], ["CD8", "CD45"]],
        "support": ["CD45"],
        "excluded": ["CD4"],
    }


def test_colon_separator_is_not_confused_by_optional_equals_signs():
    rows, problems = parse_resolve_rule_text(
        "Epithelial: [Pan-Cytokeratin] OR [E-cadherin]; SUPPORT=[]; NOT=[CD45]",
        MARKERS,
    )

    assert problems == []
    assert rows[0]["Phenotype"] == "Epithelial"
    assert rows[0]["alternatives"] == [["Pan-Cytokeratin"], ["E-cadherin"]]


def test_marker_matching_is_case_insensitive_and_returns_canonical_spelling():
    rows, problems = parse_resolve_rule_text("Treg = [foxp3, cd4, cd3E]", MARKERS)

    assert problems == []
    assert rows[0]["alternatives"] == [["FoxP3", "CD4", "CD3e"]]


def test_unknown_marker_rejects_the_entire_line_without_weakening_it():
    rows, problems = parse_resolve_rule_text("CD8 = [CD8, typo]", MARKERS)

    assert rows == []
    assert problems == ["line 1: 'CD8' skipped - unknown marker(s) typo"]


def test_malformed_bracket_is_reported():
    rows, problems = parse_resolve_rule_text("CD8 = [CD8, CD3e)", MARKERS)

    assert rows == []
    assert "without a matching ']'" in problems[0]


def test_duplicate_phenotype_is_rejected_instead_of_silently_overwriting():
    rows, problems = parse_resolve_rule_text("CD8 = [CD8]\ncd8 = [CD8, CD3e]", MARKERS)

    assert len(rows) == 1
    assert problems == ["line 2: duplicate phenotype 'cd8' skipped"]


def test_attachment_legacy_rules_create_eleven_rows_and_seven_alternatives():
    rules = """
Fibroblast = [FN1] OR [Vimentin] OR [PDGFRa] OR [Podoplanin] OR [aSMA]
IMC = [CD11b, S100A9, CD15] OR [CD11b, S100A9] OR [CD15, S100A9] OR [CD11b, CD15]
DC = [CD11c, HLA-DR, CD45] OR [CD11c, HLA-DR] OR [CD11c, CD45]
Macrophage = [CD68, CD163, CD45] OR [CD68, CD45] OR [CD68, HLA-DR] OR [CD163, HLA-DR] OR [CD163, CD45] OR [CD68, Vimentin] OR [CD163, Vimentin]
Endothelial = [CD31]
B = [CD20, HLA-DR, CD45] OR [CD20, HLA-DR] OR [CD20, CD45]
Epithelial = [Pan-Cytokeratin] OR [E-cadherin]
Plasma = [CD38]
Treg = [FoxP3, CD4, CD3e, CD45] OR [FoxP3, CD4, CD3e] OR [FoxP3, CD4, CD45] OR [FoxP3, CD3e, CD45]
CD8 = [CD8, CD45, CD3e] OR [CD8, CD45] OR [CD8, CD3e]
CD4 = [CD4, CD45, CD3e] OR [CD4, CD45] OR [CD4, CD3e]
"""
    rows, problems = parse_resolve_rule_text(rules, MARKERS)

    assert problems == []
    assert len(rows) == 11
    assert max(len(row["alternatives"]) for row in rows) == 7
