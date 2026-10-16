"""Tables Studio saves as Parquet: read like any table, saved columns kept beside them, written in as Parquet."""

import pandas as pd

from spacemap import dataio, pending_columns


def test_a_parquet_table_is_its_own_fast_copy_and_stays_parquet(tmp_path):
    path = tmp_path / "batch_corrected_merged_20260927_1950.parquet"
    pd.DataFrame({"cell_id": ["c0", "c1", "c2"], "CD3": [1.0, 2.0, 3.0], "batch_id": ["R1", "R1", "R2"]}).to_parquet(path)
    assert dataio.is_parquet_table(path) and dataio.fast_copy_path(path) == path.resolve()
    assert dataio.fast_copy_info(path)["current"] and dataio.ensure_fast_copy(path) == path.resolve()
    assert dataio.columns(path) == ["cell_id", "CD3", "batch_id"]
    assert dataio.load(path, columns=["CD3"])["CD3"].tolist() == [1.0, 2.0, 3.0]
    pending_columns.save(path, {"phenotype_x": ["T", "B", "T"]}, anchor_column="cell_id", anchor_values=["c0", "c1", "c2"])
    assert "phenotype_x" in dataio.columns(path)
    assert dataio.load(path, columns=["phenotype_x"])["phenotype_x"].tolist() == ["T", "B", "T"]
    report = pending_columns.commit(path)
    assert report["written"] and pending_columns.pending(path) == []
    written = pd.read_parquet(path)
    assert written["phenotype_x"].tolist() == ["T", "B", "T"] and not list(tmp_path.glob("*.csv"))
