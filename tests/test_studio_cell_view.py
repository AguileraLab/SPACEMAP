"""Which tables belong to a segmentation's cells, and which cells pass a layer's filters."""

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import cell_view


def _table(path, labels, xs, extra=None):
    frame = pd.DataFrame({"batch_id": "roi", "label": labels, "x_centroid": xs, "y_centroid": [10.0] * len(labels), **(extra or {})})
    frame.to_parquet(path, index=False)
    return str(path)


def test_a_table_from_another_segmentation_is_not_offered(tmp_path):
    xs = np.arange(50) * 20.0
    own = _table(tmp_path / "own.parquet", range(1, 51), xs)
    same = _table(tmp_path / "combined.parquet", range(1, 51), xs + 1, {"phenotype": ["T"] * 50})
    other = _table(tmp_path / "old.parquet", range(1, 51), xs[::-1].copy())  # the same numbers, other cells
    assert cell_view.matches(own, same, "roi")[0]
    fits, reason = cell_view.matches(own, other, "roi")
    assert not fits and "another segmentation" in reason
    no_keys = tmp_path / "plain.csv"
    pd.DataFrame({"a": [1]}).to_csv(no_keys, index=False)
    assert not cell_view.matches(own, str(no_keys), "roi")[0]


def test_filters_keep_groups_and_ranges(tmp_path):
    table = _table(tmp_path / "cells.parquet", range(1, 11), np.arange(10.0),
                   {"kind": list("AABBAABBAB"), "score": np.arange(10.0)})
    kinds = cell_view.column_kinds(table)
    assert kinds["kind"] == "group" and kinds["score"] == "value"
    lookup, shown, total = cell_view.keep(table, "roi", cell_view.parse_filters('[{"col": "kind", "values": ["A"]}, {"col": "score", "lo": 2, "hi": 9}]'))
    assert total == 10 and shown == 3 and sorted(np.flatnonzero(lookup)) == [5, 6, 9]
    assert cell_view.keep(table, "roi", [])[0] is None


def test_each_linked_cell_is_checked_even_in_large_rois(tmp_path):
    xs = np.arange(1000) * 20.0
    own = _table(tmp_path / "own.parquet", range(1, 1001), xs)
    changed = xs.copy()
    changed[677] += 20
    table = _table(tmp_path / "changed.parquet", range(1, 1001), changed)
    fits, reason = cell_view.matches(own, table, "roi")
    assert not fits and reason.startswith("1 cells")


@pytest.mark.parametrize("invalid", [np.nan, 1.5, 0, -1])
def test_invalid_local_label_cannot_be_dropped_or_truncated(tmp_path, invalid):
    labels = np.arange(1, 11, dtype=float)
    labels[5] = invalid
    table = _table(tmp_path / "cells.parquet", labels, np.arange(10.0))
    fits, reason = cell_view.matches(None, table, "roi", cells_count=10)
    assert not fits and "positive whole numbers" in reason


def test_saved_numeric_scores_have_value_controls(tmp_path):
    from spacemap import pending_columns

    table = _table(tmp_path / "cells.parquet", [1, 2], [10, 20], {"cell_id":[101, 102]})
    assert "prediction_score" not in cell_view.column_kinds(table)
    pending_columns.save(table, {"phenotype":["A", "B"], "prediction_score":[0.4, 0.9]},
                         anchor_column="cell_id", anchor_values=[101, 102])
    kinds = cell_view.column_kinds(table)
    assert kinds["phenotype"] == "group" and kinds["prediction_score"] == "value"
