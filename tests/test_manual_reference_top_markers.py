"""Ranking the markers shown beside each manually-reviewed cell.

The card lists a cell's top 5 markers so the reviewer can judge the label. Ranking runs
on the *normalised* columns, and normalisation (z-score, CDF) routinely produces negative
values — so a marker with no value must be left out of the ranking rather than scored as
zero, which used to let missing markers displace real ones.
"""

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.tabs.tab7_phenotyping.tools import (
    _apply_visual_normalization,
    _compute_top_markers_for_row,
)


MARKERS = ["HLA-DR", "aSMA", "Synaptophysin", "MUC-1", "CD3", "CD8", "FoxP3"]
NORM_COLS = [f"{m}_norm" for m in MARKERS]


def _row(values):
    return pd.Series(dict(zip(NORM_COLS, values)))


def test_ranks_by_normalised_value_highest_first():
    row = _row([2.20, 1.94, 1.50, 1.49, 1.38, 0.10, -0.40])

    top = _compute_top_markers_for_row(row, MARKERS, NORM_COLS, top_n=5)

    assert [name for _raw, name, _score in top] == ["HLA-DR", "aSMA", "Synaptophysin", "MUC-1", "CD3"]
    assert [round(score, 2) for _raw, _name, score in top] == [2.20, 1.94, 1.50, 1.49, 1.38]


def test_negative_values_still_rank_when_few_markers_are_positive():
    """z-scored data is often mostly negative; the top 5 must still be the 5 largest."""
    row = _row([-0.10, -0.50, -0.90, -1.30, -1.70, -2.10, -2.50])

    top = _compute_top_markers_for_row(row, MARKERS, NORM_COLS, top_n=5)

    assert [name for _raw, name, _score in top] == ["HLA-DR", "aSMA", "Synaptophysin", "MUC-1", "CD3"]


def test_missing_markers_do_not_displace_real_negative_ones():
    """The bug: NaN scored as 0.0 outranked genuinely-measured negative markers."""
    row = _row([np.nan, np.nan, -0.20, -0.30, -0.40, -0.50, -0.60])

    top = _compute_top_markers_for_row(row, MARKERS, NORM_COLS, top_n=5)
    names = [name for _raw, name, _score in top]

    assert "HLA-DR" not in names and "aSMA" not in names
    assert names == ["Synaptophysin", "MUC-1", "CD3", "CD8", "FoxP3"]


def test_a_row_with_no_measured_markers_yields_nothing():
    row = _row([np.nan] * len(MARKERS))

    assert _compute_top_markers_for_row(row, MARKERS, NORM_COLS, top_n=5) == []


def test_non_numeric_values_are_treated_as_missing():
    row = _row(["n/a", 1.0, 2.0, "", None, 0.5, 0.25])

    top = _compute_top_markers_for_row(row, MARKERS, NORM_COLS, top_n=5)
    names = [name for _raw, name, _score in top]

    assert "HLA-DR" not in names
    assert names[:2] == ["Synaptophysin", "aSMA"]


def test_duplicate_clean_names_collapse_to_their_strongest_column():
    """"CD3_cellRing" and "CD3" both clean to CD3; the stronger column represents it."""
    markers = ["CD3_cellRing", "CD3", "CD8_cellRing"]
    norm_cols = [f"{m}_norm" for m in markers]
    row = pd.Series({norm_cols[0]: 0.5, norm_cols[1]: 2.5, norm_cols[2]: 1.0})

    top = _compute_top_markers_for_row(
        row, markers, norm_cols, top_n=5, affix_position="suffix", correction_affix="_cellRing"
    )

    names = [name for _raw, name, _score in top]
    assert names.count("CD3") == 1
    assert dict((n, s) for _r, n, s in top)["CD3"] == 2.5


def test_a_missing_duplicate_does_not_hide_its_measured_twin():
    """If one of two columns for a marker is NaN, the measured one must still count."""
    markers = ["CD3_cellRing", "CD3"]
    norm_cols = [f"{m}_norm" for m in markers]
    row = pd.Series({norm_cols[0]: np.nan, norm_cols[1]: 1.75})

    top = _compute_top_markers_for_row(
        row, markers, norm_cols, top_n=5, affix_position="suffix", correction_affix="_cellRing"
    )

    assert [(name, score) for _raw, name, score in top] == [("CD3", 1.75)]


def test_top_n_is_respected():
    row = _row([7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    assert len(_compute_top_markers_for_row(row, MARKERS, NORM_COLS, top_n=3)) == 3


def test_missing_raw_measurements_stay_missing_through_normalization():
    raw = pd.DataFrame(
        {
            "A": [np.nan, 1.0, 2.0],
            "B": [2.0, 3.0, 4.0],
            "C": [4.0, 5.0, 6.0],
        }
    )

    prepared, norm_cols = _apply_visual_normalization(raw, ["A", "B", "C"], ["Z1"], None)

    assert pd.isna(prepared.loc[0, norm_cols[0]])
    assert prepared.loc[0, norm_cols[1:]].notna().all()


def test_top_markers_are_ranked_from_the_completed_transform_pipeline():
    markers = ["A", "B", "C", "D", "E", "F"]
    raw = pd.DataFrame(
        {
            "A": [1.0, 5.0, 9.0],
            "B": [8.0, 4.0, 0.0],
            "C": [3.0, 3.5, 4.0],
            "D": [9.0, 2.0, 1.0],
            "E": [2.0, 8.0, 3.0],
            "F": [4.0, 4.0, 7.0],
        }
    )
    prepared, norm_cols = _apply_visual_normalization(
        raw,
        markers,
        ["Z1", "Z2", "CDF", "NLog"],
        None,
        apply_arcsinh=True,
        arcsinh_cofactor=5.0,
    )

    top = _compute_top_markers_for_row(prepared.loc[0], markers, norm_cols, top_n=5)
    expected_order = (
        prepared.loc[0, norm_cols]
        .sort_values(ascending=False)
        .index.to_series()
        .map(dict(zip(norm_cols, markers)))
        .head(5)
        .tolist()
    )

    assert [name for _raw, name, _score in top] == expected_order
