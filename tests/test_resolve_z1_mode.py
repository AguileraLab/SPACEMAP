"""RESOLVE Z1 score mode: positive where the per-batch Z1 score exceeds a cutoff."""

import numpy as np
import pandas as pd

from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import (
    DEFAULT_RESOLVE_Z1_CUTOFF,
    _apply_resolve_z1_positivity,
)


def _frame():
    rng = np.random.default_rng(0)
    background = rng.normal(100.0, 10.0, size=190)
    bright = np.full(10, 5000.0)
    values = np.concatenate([background, bright])
    return pd.DataFrame({
        "CD3": np.concatenate([values, values * 3.0]),   # batch B is 3x brighter overall
        "batch_id": ["A"] * 200 + ["B"] * 200,
    })


def test_default_cutoff_is_one_and_a_half():
    assert DEFAULT_RESOLVE_Z1_CUTOFF == 1.5


def test_only_the_bright_cells_are_positive_in_each_batch():
    df = _frame()
    binary, scores = _apply_resolve_z1_positivity(df, ["CD3"], "batch_id")

    positive = binary["CD3_bin_resolve"].to_numpy().astype(bool)
    expected = np.zeros(400, dtype=bool)
    expected[190:200] = True
    expected[390:400] = True
    np.testing.assert_array_equal(positive, expected)
    assert list(scores.columns) == ["CD3_norm_resolve"]


def test_z1_is_computed_within_each_batch():
    # Batch B is the same pattern, three times brighter. Per-batch Z1 makes the
    # two batches identical; a global Z1 would call most of batch B positive.
    df = _frame()
    _, scores = _apply_resolve_z1_positivity(df, ["CD3"], "batch_id")
    a = scores["CD3_norm_resolve"].to_numpy()[:200]
    b = scores["CD3_norm_resolve"].to_numpy()[200:]
    np.testing.assert_allclose(a, b, atol=0.05)


def test_the_cutoff_controls_how_many_cells_are_positive():
    df = _frame()
    loose, _ = _apply_resolve_z1_positivity(df, ["CD3"], "batch_id", cutoff=-1.0)
    strict, _ = _apply_resolve_z1_positivity(df, ["CD3"], "batch_id", cutoff=1.5)
    assert loose["CD3_bin_resolve"].sum() > strict["CD3_bin_resolve"].sum()


def test_missing_values_are_never_positive_and_absent_markers_are_skipped():
    df = _frame()
    df.loc[0, "CD3"] = np.nan
    binary, _ = _apply_resolve_z1_positivity(df, ["CD3", "NotAMarker"], None)
    assert binary.loc[0, "CD3_bin_resolve"] == 0
    assert list(binary.columns) == ["CD3_bin_resolve"]


def test_cells_without_a_batch_value_are_kept_and_not_positive():
    df = _frame()
    df.loc[5, "batch_id"] = None
    binary, scores = _apply_resolve_z1_positivity(df, ["CD3"], "batch_id")
    assert len(binary) == len(df) and len(scores) == len(df)
    assert binary.loc[5, "CD3_bin_resolve"] == 0
    assert binary["CD3_bin_resolve"].notna().all()
