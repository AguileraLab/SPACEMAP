"""The loader must hand back what ``pandas.read_csv`` would.

These exist because it once did not. ``spacemap.dataio.load`` converted text
columns to ``category`` to save memory, which broke two Manual Reference
screens at runtime while the whole suite stayed green: assigning a phenotype to
an unlabelled cell raised "Cannot setitem on a Categorical with a new
category", and inverting a mask built from such a column raised "bad operand
type for unary ~".

Nothing else in the codebase asks for categorical dtypes, so the loader must
not impose them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spacemap import dataio


@pytest.fixture
def table(tmp_path):
    """A small CSV with the column shapes that caused the failures."""
    path = tmp_path / "cells.csv"
    pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "batch_id": ["P1", "P1", "P2", "P2"],
            "phenotype_leiden": ["Tumor", "CD8 T", None, "Stroma"],
            "CD45": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_csv(path, index=False)
    return path


def test_text_columns_are_not_categorical_by_default(table):
    frame = dataio.load(table)
    for column in ("batch_id", "phenotype_leiden"):
        assert frame[column].dtype == object, (
            f"{column} came back as {frame[column].dtype}; categorical dtypes "
            "break callers that assign new values"
        )


def test_dtypes_match_pandas(table):
    expected = pd.read_csv(table, low_memory=False)
    actual = dataio.load(table)
    assert list(actual.columns) == list(expected.columns)
    for column in expected.columns:
        assert str(actual[column].dtype) == str(expected[column].dtype), column


def test_a_label_outside_the_existing_values_can_be_assigned(table):
    """Manual Reference assigns phenotypes that are not yet in the column."""
    frame = dataio.load(table, columns=["phenotype_leiden"])
    frame.loc[frame.index[2], "phenotype_leiden"] = "a label never seen before"
    assert frame.loc[frame.index[2], "phenotype_leiden"] == "a label never seen before"


def test_a_mask_over_a_text_column_can_be_inverted(table):
    """The exact shape of _exclude_existing_visual_manual_reference_assignments."""
    frame = dataio.load(table, columns=["phenotype_leiden"])
    mask = frame["phenotype_leiden"].apply(lambda value: bool(str(value).strip()) and value is not None)
    assert mask.dtype == bool
    assert not frame.loc[~mask].empty or frame.loc[~mask].empty  # must not raise


def test_categoricals_remain_available_when_asked_for(table):
    """Read-only callers may still opt in for the memory saving."""
    frame = dataio.load(table, columns=["batch_id"], categoricals=True)
    assert str(frame["batch_id"].dtype) == "category"


def test_named_columns_are_the_only_ones_returned(table):
    frame = dataio.load(table, columns=["cell_id", "CD45"])
    assert list(frame.columns) == ["cell_id", "CD45"]


def test_an_unknown_column_is_reported_clearly(table):
    with pytest.raises(dataio.DataIOError, match="no column"):
        dataio.load(table, columns=["not_a_column"])
