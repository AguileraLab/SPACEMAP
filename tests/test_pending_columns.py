"""Phenotype saves stored beside the CSV, then written into it on request."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels, pending_columns
from spacemap_gui.tabs.tab7_phenotyping.common import (
    PendingSaveBlocked,
    _save_assignments_as_pending,
)
from spacemap_gui.ui_utils import file_signature


@pytest.fixture(autouse=True)
def no_fast_copy(monkeypatch):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")


def _table(tmp_path, n=60):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(n)],
        "CD3": rng.gamma(2, 50, n).round(3),
        "batch_id": ["S1"] * (n // 2) + ["S2"] * (n - n // 2),
        "phenotype_leiden": ["Old"] * 10 + [None] * (n - 10),
    })
    path = tmp_path / "unified.csv"
    frame.to_csv(path, index=False)
    return path, frame


def _results(frame):
    # A filtered analysis: every third cell, clusters 0-2.
    subset = frame.iloc[::3].copy()
    subset["leiden_cluster"] = np.arange(len(subset)) % 3
    return subset


def _get_pheno(mapping):
    return lambda cluster: mapping.get(cluster, "Unassigned")


def _old_full_rewrite(path, results, mapping, output_col):
    """The previous save: read the whole CSV, update analysed cells, rewrite it."""
    df_in = pd.read_csv(path)
    mask = df_in["cell_id"].isin(results["cell_id"].values)
    id_to_pheno = dict(zip(results["cell_id"], results["leiden_cluster"].apply(_get_pheno(mapping))))
    df_in.loc[mask, output_col] = df_in.loc[mask, "cell_id"].map(id_to_pheno)
    id_to_cluster = dict(zip(results["cell_id"], results["leiden_cluster"]))
    df_in.loc[mask, "leiden_cluster"] = df_in.loc[mask, "cell_id"].map(id_to_cluster)
    return df_in


def _fast_save(path, results, mapping, output_col):
    return _save_assignments_as_pending(
        path, results, "cell_id", False, ["leiden_cluster"], "leiden_cluster",
        output_col, _get_pheno(mapping), source_note="test",
    )


def test_saved_columns_are_read_back_before_being_written(tmp_path):
    path, frame = _table(tmp_path)
    before = path.read_bytes()
    saved = _fast_save(path, _results(frame), {0: "T", 1: "B", 2: "Mac"}, "phenotype_leiden")

    assert set(saved) == {"phenotype_leiden", "leiden_cluster"}
    assert path.read_bytes() == before, "the CSV itself must not be rewritten on save"
    assert "leiden_cluster" in dataio.columns(path)
    assert "leiden_cluster" not in dataio.columns(path, include_pending=False)
    shown = dataio.load(path, columns=["cell_id", "phenotype_leiden"])
    assert shown.loc[0, "phenotype_leiden"] == "T"
    on_disk = dataio.load(path, columns=["phenotype_leiden"], apply_labels=False)
    assert on_disk.loc[0, "phenotype_leiden"] == "Old"


def test_save_then_write_matches_the_old_full_rewrite(tmp_path):
    path, frame = _table(tmp_path)
    results = _results(frame)
    mapping = {0: "T", 1: "B", 2: "Mac"}
    expected = _old_full_rewrite(path, results, mapping, "phenotype_leiden")

    _fast_save(path, results, mapping, "phenotype_leiden")
    report = pending_columns.commit(path, rebuild_fast_copy=False)

    assert report["written"]
    written = pd.read_csv(path)
    pd.testing.assert_frame_equal(
        written[expected.columns].reset_index(drop=True), expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert pending_columns.pending(path) == []


def test_saves_stack_and_a_second_method_keeps_the_first(tmp_path):
    path, frame = _table(tmp_path)
    _fast_save(path, _results(frame), {0: "T", 1: "B", 2: "Mac"}, "phenotype_leiden")
    _save_assignments_as_pending(
        path, _results(frame), "cell_id", False, [], "leiden_cluster", "phenotype_som",
        _get_pheno({0: "X", 1: "Y", 2: "Z"}),
    )
    shown = dataio.load(path, columns=["phenotype_leiden", "phenotype_som"])
    assert shown.loc[0, "phenotype_leiden"] == "T" and shown.loc[0, "phenotype_som"] == "X"
    assert pd.isna(shown.loc[1, "phenotype_som"])  # not analysed, no previous value


def test_a_rewritten_csv_is_never_patched_by_position(tmp_path):
    path, frame = _table(tmp_path)
    _fast_save(path, _results(frame), {0: "T", 1: "B", 2: "Mac"}, "phenotype_leiden")

    # Something else rewrites the CSV with the rows in reverse order.
    frame.iloc[::-1].to_csv(path, index=False)
    os.utime(path, ns=(1, 1))

    assert pending_columns.has_stale(path)
    reversed_view = dataio.load(path, columns=["cell_id", "phenotype_leiden"])
    assert reversed_view.loc[reversed_view["cell_id"] == "c0", "phenotype_leiden"].item() == "Old"
    with pytest.raises(PendingSaveBlocked):
        _fast_save(path, _results(frame), {0: "T"}, "phenotype_leiden")

    assert pending_columns.reapply_by_id(path) == 2
    fixed = dataio.load(path, columns=["cell_id", "phenotype_leiden"]).set_index("cell_id")
    assert fixed.loc["c0", "phenotype_leiden"] == "T"      # analysed cell, new label
    assert fixed.loc["c1", "phenotype_leiden"] == "Old"    # not analysed, kept
    assert not pending_columns.has_stale(path)


def test_a_failed_write_leaves_the_csv_and_the_saved_changes(tmp_path, monkeypatch):
    path, frame = _table(tmp_path)
    _fast_save(path, _results(frame), {0: "T"}, "phenotype_leiden")
    before = path.read_bytes()

    def broken(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", broken)
    with pytest.raises(pending_columns.PendingColumnsError):
        pending_columns.commit(path, rebuild_fast_copy=False)
    monkeypatch.undo()

    assert path.read_bytes() == before
    assert set(pending_columns.pending_names(path)) == {"phenotype_leiden", "leiden_cluster"}


def test_labels_are_written_in_the_same_pass(tmp_path):
    path, frame = _table(tmp_path)
    _fast_save(path, _results(frame), {0: "T"}, "phenotype_leiden")
    labels.append(path, [{"row": 5, "column": "phenotype_manual", "value": "CD8"}])

    pending_columns.commit(path, rebuild_fast_copy=False)

    written = pd.read_csv(path)
    assert written.loc[5, "phenotype_manual"] == "CD8"
    assert written.loc[0, "phenotype_leiden"] == "T"
    assert labels.pending_count(path) == 0


def test_the_background_write_reports_its_progress(tmp_path):
    path, frame = _table(tmp_path)
    _fast_save(path, _results(frame), {0: "T"}, "phenotype_leiden")
    assert pending_columns._run_commit_process(str(path)) == 0
    status = pending_columns.commit_status(path)
    assert status["state"] == "done" and "2 column" in status["message"]


def test_a_save_changes_the_cache_key(tmp_path):
    path, frame = _table(tmp_path)
    before = file_signature(path)
    _fast_save(path, _results(frame), {0: "T"}, "phenotype_leiden")
    assert file_signature(path) != before
