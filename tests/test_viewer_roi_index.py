"""Fast ROI lookup preserves cell identity without reading unrelated row groups."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacemap import dataio
from spacemap.viewer import cells


def write_table(tmp_path, frame, monkeypatch):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    path = tmp_path / "cells.csv"
    frame.to_csv(path, index=False)
    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    arrow = arrow.replace_schema_metadata({dataio._METADATA_KEY: dataio._source_fingerprint(path).encode()})
    pq.write_table(arrow, dataio.fast_copy_path(path), row_group_size=3)
    return str(path)


def test_mixed_row_groups_keep_global_positions_and_skip_unrelated_groups(tmp_path, monkeypatch):
    frame = pd.DataFrame({"roi": ["A"] * 3 + ["A", "B", "A"] + ["B"] * 3, "cell": [1, 2, 3, 4, 10, 5, 11, 12, 13]})
    path = write_table(tmp_path, frame, monkeypatch)
    real = pq.ParquetFile.read_row_group
    groups = []

    def read(self, number, *args, **kwargs):
        groups.append(number)
        return real(self, number, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read)
    positions, labels = cells.roi_index(path, "B", roi_column="roi", label_column="cell")
    np.testing.assert_array_equal(positions, [4, 6, 7, 8])
    np.testing.assert_array_equal(labels, [10, 11, 12, 13])
    assert groups == [1, 2]
    cells.roi_index(path, "B", roi_column="roi", label_column="cell")
    assert groups == [1, 2], "Repeated selection must reuse the cached index."


@pytest.mark.parametrize("roi", ["A", "B", "missing"])
def test_invalid_labels_and_missing_rois_match_normal_reader(tmp_path, monkeypatch, roi):
    frame = pd.DataFrame({"batch_id": ["A", "B", "A", "B", "A", "B"], "label": ["2", "3", "invalid", None, "5", "6"]})
    path = write_table(tmp_path, frame, monkeypatch)
    positions, labels = cells.roi_index(path, roi)
    numbers = pd.to_numeric(frame.label, errors="coerce").to_numpy()
    wanted = (frame.batch_id == roi) & np.isfinite(numbers)
    np.testing.assert_array_equal(positions, np.flatnonzero(wanted))
    np.testing.assert_array_equal(labels, numbers[wanted].astype(np.int64))


def test_numeric_roi_names_keep_existing_string_conversion(tmp_path, monkeypatch):
    path = write_table(tmp_path, pd.DataFrame({"batch_id": [1.0, 2.0, 1.0], "label": [5, 6, 7]}), monkeypatch)
    assert cells._parquet_roi_index(path, "1.0", "batch_id", "label") is None
    positions, labels = cells.roi_index(path, "1.0")
    np.testing.assert_array_equal(positions, [0, 2])
    np.testing.assert_array_equal(labels, [5, 7])


def test_stale_copy_is_not_used(tmp_path, monkeypatch):
    path = write_table(tmp_path, pd.DataFrame({"batch_id": ["A"], "label": [1]}), monkeypatch)
    pd.DataFrame({"batch_id": ["A", "A"], "label": [3, 4]}).to_csv(path, index=False)
    assert cells._parquet_roi_index(path, "A", "batch_id", "label") is None
    positions, labels = cells.roi_index(path, "A")
    np.testing.assert_array_equal(positions, [0, 1])
    np.testing.assert_array_equal(labels, [3, 4])
