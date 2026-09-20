"""Turning the on-screen phenotype choices into write instructions.

This is the step between the dropdowns and the CSV writer. It decides whether a cell is
written, cleared, or skipped — the decision that determines whether a wrong label can
actually be taken off a cell.
"""

import pytest

from spacemap_gui.tabs.tab7_phenotyping.tools import (
    _build_manual_reference_pending_assignments,
    _build_manual_reference_save_receipt,
)


def _result(key, current_phenotype):
    return {
        "result_key": key,
        "batch_col": "batch_id",
        "batch_value": "reg055_A",
        "cell_id_col": "cell_id",
        "cell_id_value": f"cell_{key}",
        "default_x_col": "x_centroid",
        "x_value": 494.09,
        "default_y_col": "y_centroid",
        "y_value": 76.80,
        "default_output_col": "phenotype_ref1",
        "current_phenotype": current_phenotype,
    }


def test_unassigned_on_a_labelled_cell_becomes_a_clear():
    results = [_result("a", "Treg")]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": "Unassigned"}, "phenotype_ref1"
    )

    assert counts["clearing"] == 1
    assert counts["unassigned"] == 0
    assert len(assignments) == 1
    assert assignments[0]["clear_existing"] is True


@pytest.mark.parametrize("clear_choice", ["Unclassified", "Unknown", "un-labeled"])
def test_every_no_label_choice_on_a_labelled_cell_becomes_a_clear(clear_choice):
    results = [_result("a", "Treg")]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": clear_choice}, "phenotype_ref1"
    )

    assert counts["clearing"] == 1
    assert len(assignments) == 1
    assert assignments[0]["clear_existing"] is True


def test_unassigned_on_an_unlabelled_cell_is_skipped():
    results = [_result("a", None)]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": "Unassigned"}, "phenotype_ref1"
    )

    assert assignments == []
    assert counts["unassigned"] == 1


@pytest.mark.parametrize("legacy_value", ["Unassigned", "unclassified", "Unknown"])
def test_a_stored_no_label_string_is_physically_cleared(legacy_value):
    results = [_result("a", legacy_value)]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": legacy_value}, "phenotype_ref1"
    )

    assert counts["clearing"] == 1
    assert assignments[0]["clear_existing"] is True


def test_a_concrete_choice_is_written_normally():
    results = [_result("a", "Treg")]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": "CD8"}, "phenotype_ref1"
    )

    assert counts == {"clearing": 0, "unassigned": 0, "pending_new": 0}
    assert assignments[0]["phenotype_value"] == "CD8"
    assert assignments[0]["clear_existing"] is False


def test_add_new_phenotype_is_held_back():
    results = [_result("a", "Treg")]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": "Add New Phenotype"}, "phenotype_ref1"
    )

    assert assignments == []
    assert counts["pending_new"] == 1


def test_a_missing_selection_defaults_to_unassigned():
    """A cell whose dropdown was never touched must not be silently written."""
    results = [_result("a", None)]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {}, "phenotype_ref1"
    )

    assert assignments == []
    assert counts["unassigned"] == 1


def test_a_mixed_batch_is_split_correctly():
    """The shape of a real review pass: some cleared, some kept, some untouched."""
    results = [_result(str(i), "Treg") for i in range(5)] + [_result("new", None)]
    selections = {
        "0": "Unassigned", "1": "Unassigned",   # remove these two
        "2": "Treg", "3": "CD8",                # keep / relabel
        "4": "Add New Phenotype",
        "new": "Unassigned",                    # never had a label
    }

    assignments, counts = _build_manual_reference_pending_assignments(
        results, selections, "phenotype_ref1"
    )

    assert counts == {"clearing": 2, "unassigned": 1, "pending_new": 1}
    assert len(assignments) == 4
    assert sum(1 for a in assignments if a["clear_existing"]) == 2


def test_every_assignment_carries_its_result_key():
    """The diagnostics panel and the snapshot refresh both key off this."""
    results = [_result("a", "Treg"), _result("b", "Treg")]

    assignments, _counts = _build_manual_reference_pending_assignments(
        results, {"a": "Unassigned", "b": "CD8"}, "phenotype_ref1"
    )

    assert {a["result_key"] for a in assignments} == {"a", "b"}


def test_save_receipt_reports_exact_actions_and_cells():
    results = [_result("a", "Treg"), _result("b", "Treg"), _result("c", None)]
    selections = {"a": "Unclassified", "b": "Treg", "c": "CD8"}
    assignments, _counts = _build_manual_reference_pending_assignments(
        results, selections, "phenotype_ref1"
    )

    receipt = _build_manual_reference_save_receipt(
        assignments,
        results,
        verified_count=3,
        target_path="/data/unified.csv",
        timestamp="2026-08-26 21:10:18",
    )

    assert receipt["verified_count"] == 3
    assert receipt["cleared_count"] == 1
    assert receipt["changed_count"] == 1
    assert receipt["unchanged_count"] == 1
    assert [row["Action"] for row in receipt["details"]] == [
        "Removed label",
        "Verified unchanged",
        "Assigned",
    ]
    assert receipt["details"][0]["Cell ID"] == "cell_a"
    assert receipt["details"][0]["Saved value"] == "<blank>"
    assert receipt["target_path"] == "/data/unified.csv"


@pytest.mark.parametrize("blank", [None, "", "nan", "NaN", "<NA>"])
def test_blank_current_labels_count_as_no_label(blank):
    results = [_result("a", blank)]

    assignments, counts = _build_manual_reference_pending_assignments(
        results, {"a": "Unassigned"}, "phenotype_ref1"
    )

    assert assignments == []
    assert counts["unassigned"] == 1
