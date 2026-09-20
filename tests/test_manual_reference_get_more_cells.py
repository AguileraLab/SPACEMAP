"""Get More Cells must never repeat a cell within one review run.

Clicking "Select and Visualize Cells" starts a run and clears the record of what has been
shown; every "Get More Cells" after that adds to it, so each batch is new. Starting a
fresh selection is free to return the same cells again.
"""

import pytest

from spacemap_gui.tabs.tab7_phenotyping.tools import (
    _accumulate_shown_cell_keys,
    _sanitize_filename_token,
)


def _key(batch, cell_id, x, y):
    """The identity the fetch loop builds for each candidate."""
    return _sanitize_filename_token(f"{batch}_{cell_id}_{x}_{y}")


def test_cell_keys_are_stable_and_distinct():
    """The skip only works if the same cell always produces the same key."""
    first = _key("reg055_A", "cell_170913", 494.092872570194, 76.8077753779698)
    again = _key("reg055_A", "cell_170913", 494.092872570194, 76.8077753779698)
    other = _key("reg048_A", "cell_152286", 1012.85239852399, 179.944649446494)

    assert first == again
    assert first != other


def test_two_cells_in_the_same_batch_do_not_collide():
    a = _key("reg055_A", "cell_170913", 494.09, 76.80)
    b = _key("reg055_A", "cell_170914", 494.09, 76.80)

    assert a != b


def test_successive_batches_accumulate():
    shown = _accumulate_shown_cell_keys(None, ["a", "b"])
    shown = _accumulate_shown_cell_keys(shown, ["c", "d"])
    shown = _accumulate_shown_cell_keys(shown, ["e"])

    assert shown == ["a", "b", "c", "d", "e"]


def test_a_repeated_key_is_not_double_counted():
    shown = _accumulate_shown_cell_keys(["a", "b"], ["b", "c"])

    assert shown == ["a", "b", "c"]


@pytest.mark.parametrize("previous", [None, [], ["a"]])
def test_handles_an_empty_or_missing_history(previous):
    assert _accumulate_shown_cell_keys(previous, []) == sorted(previous or [])


def test_a_fresh_selection_starts_over():
    """Select and Visualize Cells resets the list, so earlier cells may return."""
    shown = _accumulate_shown_cell_keys(["a", "b", "c"], ["d"])
    assert "a" in shown

    restarted = _accumulate_shown_cell_keys([], ["a"])

    assert restarted == ["a"]
