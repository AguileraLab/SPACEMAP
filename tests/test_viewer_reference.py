"""Fast reference edits must retain identity, Undo and durable review flags."""

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.viewer import cells, reference


def test_presets_include_consensus_from_other_loaded_tables(table, tmp_path):
    other = tmp_path / "other.csv"
    pd.DataFrame({"phenotype_ml_consensus": ["T", "B", "T", None]}).to_csv(other, index=False)
    result = reference.options([table, str(other)], table)
    assert result["columns"] == ["phenotype_ref1"]
    assert result["phenotypes"] == ["B", "T"]
    assert result["sources"] == ["other.csv"]


def test_presets_include_pending_consensus_and_ignore_unrelated_label_edits(table):
    reference.save(table, "r1", "phenotype_ml_consensus", {10: "T", 30: "B"}, "consensus-examples")
    result = reference.phenotype_names(table)
    assert result == ["B", "T"]
    reference.save(table, "r1", "phenotype_ref1", {10: "NK"}, "unrelated-column")
    assert reference.phenotype_names(table) is result
    reference.save(table, "r1", "phenotype_ml_consensus", {10: "NK"}, "changed-consensus")
    assert reference.phenotype_names(table) == ["B", "NK"]


def test_presets_drop_names_cleared_by_pending_changes(tmp_path):
    path = tmp_path / "consensus.csv"
    pd.DataFrame({"batch_id": ["r1"] * 2, "label": [1, 2], "phenotype_ml_consensus": ["T", "B"]}).to_csv(
        path, index=False
    )
    labels.append(str(path), [{"row": 0, "column": "phenotype_ml_consensus", "value": ""}])
    assert reference.phenotype_names(str(path)) == ["B"]


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    path = tmp_path / "cells.csv"
    pd.DataFrame(
        {
            "batch_id": ["r1", "r2", "r1", "r1"],
            "label": [10, 10, 30, 70],
            "x_centroid": [1, 9, 2, 3],
            "y_centroid": [3, 9, 2, 1],
            "phenotype_ref1": ["T", "B", "B", None],
        }
    ).to_csv(path, index=False)
    dataio.ensure_fast_copy(path, force=True)
    return str(path)


def test_assignment_reads_selected_rows_and_includes_pending_labels(table, monkeypatch):
    labels.append(table, [{"row": 2, "column": "phenotype_ref1", "value": "Myeloid"}])

    def no_full_load(*args, **kwargs):
        raise AssertionError("A reference edit must not load the whole project table")

    monkeypatch.setattr(dataio, "load", no_full_load)
    result = cells.assign(table, "r1", [30, 70], "phenotype_ref1", "T")
    assert result["previous"] == {"30": "Myeloid", "70": None}
    ids, values = cells.roi_values(table, "r1", "phenotype_ref1")
    assert dict(zip(ids, values)) == {10: "T", 30: "T", 70: "T"}
    assert cells.roi_values(table, "r2", "phenotype_ref1")[1].tolist() == ["B"]


def test_new_column_can_be_cleared_by_undo(table):
    first = cells.assign(table, "r1", [10], "my_reference", "T")
    assert first["previous"] == {"10": None}
    cells.assign_many(table, "r1", first["previous"], "my_reference")
    assert cells.values_at_rows(table, [0], "my_reference").tolist() == [None]


def test_retry_keeps_original_undo_and_does_not_reapply_over_newer_edit(table):
    first = reference.save(table, "r1", "phenotype_ref1", {10: "Myeloid", 30: "T"}, "request-one")
    assert first["previous"] == {"10": "T", "30": "B"}
    reference.save(table, "r1", "phenotype_ref1", {10: "NK"}, "request-two")
    assert reference.save(table, "r1", "phenotype_ref1", {10: "Myeloid", 30: "T"}, "request-one") == first
    assert cells.values_at_rows(table, [0], "phenotype_ref1")[0] == "NK"
    with pytest.raises(ValueError, match="different changes"):
        reference.save(table, "r1", "phenotype_ref1", {10: "B"}, "request-one")


def test_retry_after_failed_write_retains_original_undo(table, monkeypatch):
    append = labels.append

    def fail(*args, **kwargs):
        raise OSError("Disk is full")

    monkeypatch.setattr(labels, "append", fail)
    with pytest.raises(OSError):
        reference.save(table, "r1", "phenotype_ref1", {30: "T"}, "failed-save")
    monkeypatch.setattr(labels, "append", append)
    saved = reference.save(table, "r1", "phenotype_ref1", {30: "T"}, "failed-save")
    assert saved["previous"] == {"30": "B"}
    assert cells.values_at_rows(table, [2], "phenotype_ref1")[0] == "T"


def test_undo_cannot_overwrite_a_later_edit_from_another_window(table):
    original = reference.save(table, "r1", "phenotype_ref1", {10: "NK"}, "first-window")
    reference.save(table, "r1", "phenotype_ref1", {10: "Myeloid"}, "other-window")
    with pytest.raises(ValueError, match="changed after"):
        reference.save(
            table,
            "r1",
            "phenotype_ref1",
            original["previous"],
            "undo-first-window",
            source="undo",
            expected={"10": "NK"},
        )
    assert cells.values_at_rows(table, [0], "phenotype_ref1")[0] == "Myeloid"


def test_recovery_after_interrupted_receipt_does_not_overwrite_newer_labels(table, monkeypatch):
    append = labels.append

    def interrupted(*args, **kwargs):
        append(*args, **kwargs)
        raise OSError("Connection interrupted after save")

    monkeypatch.setattr(labels, "append", interrupted)
    with pytest.raises(OSError):
        reference.save(table, "r1", "phenotype_ref1", {30: "T"}, "interrupted-save")
    monkeypatch.setattr(labels, "append", append)
    labels.append(table, [{"row": 2, "column": "phenotype_ref1", "value": "NK"}])
    result = reference.save(table, "r1", "phenotype_ref1", {30: "T"}, "interrupted-save")
    assert result["previous"] == {"30": "B"}
    assert cells.values_at_rows(table, [2], "phenotype_ref1")[0] == "NK"


def test_flags_are_separate_and_survive_csv_materialization(table):
    before = cells.values_at_rows(table, [0], "phenotype_ref1")[0]
    reference.set_flag(table, "r1", "phenotype_ref1", 10, True)
    assert cells.values_at_rows(table, [0], "phenotype_ref1")[0] == before
    assert reference.review(table, "r1", "phenotype_ref1", mode="flagged")["items"][0]["label"] == 10
    assert reference.review(table, "r2", "phenotype_ref1", mode="flagged")["total"] == 0
    reference.save(table, "r1", "phenotype_ref1", {10: "NK"}, "new-phenotype")
    labels.commit(table, rebuild_fast_copy=True)
    assert reference.get_flag(table, "r1", "phenotype_ref1", 10)["flagged"]
    item = reference.review(table, "r1", "phenotype_ref1", mode="flagged")["items"][0]
    assert item["value"] == "NK"
    reference.set_flag(table, "r1", "phenotype_ref1", 10, False)
    assert reference.review(table, "r1", "phenotype_ref1", mode="flagged")["total"] == 0


def test_review_shows_existing_labels_and_current_classic_edits(table):
    assert reference.review(table, "r1", "phenotype_ref1", mode="all", limit=1)["items"][0]["label"] == 10
    assert reference.review(table, "r1", "phenotype_ref1", mode="all", offset=1)["items"][0]["label"] == 30
    assert not reference.path_for(table).exists()  # reading does not create review storage
    reference.save(table, "r1", "phenotype_ref1", {30: "T"}, "review-update")
    labels.append(table, [{"row": 2, "column": "phenotype_ref1", "value": "B"}])
    assert reference.review(table, "r1", "phenotype_ref1")["items"][0]["value"] == "B"


def test_simultaneous_saves_do_not_lose_labels(table, monkeypatch):
    original = labels.load

    def slower(path):
        value = original(path)
        time.sleep(0.02)
        return value

    monkeypatch.setattr(labels, "load", slower)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(lambda row: labels.append(table, [{"row": row, "column": "new_ref", "value": str(row)}]), range(4))
        )
    assert len(labels.load(table)) == 4


def test_duplicate_identity_and_identity_columns_are_rejected(table):
    with pytest.raises(ValueError, match="cell number or position"):
        reference.save(table, "r1", "label", {10: "T"}, "bad-column")
    frame = pd.read_csv(table)
    frame.loc[2, "label"] = 10
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    with pytest.raises(ValueError, match="repeated"):
        cells.assign(table, "r1", [10], "phenotype_ref1", "T")


def test_selected_rows_keep_the_requested_order(table):
    labels.append(table, [{"row": 0, "column": "phenotype_ref1", "value": "NK"}])
    assert cells.values_at_rows(table, np.array([2, 0, 1]), "phenotype_ref1").tolist() == ["B", "NK", "B"]


def test_selected_rows_include_full_length_pending_columns(table, monkeypatch):
    from spacemap import pending_columns

    pending_columns.save(
        table, {"phenotype_ref2": ["B", "NK", "T", "B"]}, anchor_column="x_centroid", anchor_values=[1, 9, 2, 3]
    )
    labels.append(table, [{"row": 2, "column": "phenotype_ref2", "value": "Myeloid"}])
    monkeypatch.setattr(dataio, "load", lambda *args, **kwargs: pytest.fail("Read only selected rows"))
    assert cells.values_at_rows(table, [0, 2], "phenotype_ref2").tolist() == ["B", "Myeloid"]


def test_csv_commit_and_background_save_do_not_drop_a_new_label(table, monkeypatch):
    import threading
    from spacemap import pending_columns

    labels.append(table, [{"row": 0, "column": "phenotype_ref1", "value": "NK"}])
    writing, release = threading.Event(), threading.Event()
    original = pd.DataFrame.to_csv

    def slow_write(frame, *args, **kwargs):
        writing.set()
        assert release.wait(10)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", slow_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        commit = pool.submit(pending_columns.commit, table, rebuild_fast_copy=False)
        assert writing.wait(10)
        saved = pool.submit(labels.append, table, [{"row": 2, "column": "phenotype_ref1", "value": "Myeloid"}])
        time.sleep(0.05)
        assert not saved.done()
        release.set()
        commit.result(timeout=10)
        assert saved.result(timeout=10) == 1
    frame = dataio.load(table)
    assert frame.loc[0, "phenotype_ref1"] == "NK"
    assert frame.loc[2, "phenotype_ref1"] == "Myeloid"


def test_assignments_cannot_replace_unique_cell_identity(tmp_path):
    import pandas as pd

    table = tmp_path / "cells.csv"
    pd.DataFrame({"batch_id": ["roi"], "label": [1], "cell_id": ["roi-1"]}).to_csv(table, index=False)
    with pytest.raises(ValueError, match="phenotype column"):
        reference.save(table, "roi", "cell_id", {1: "T"}, "reject-identity-edit")
    assert pd.read_csv(table)["cell_id"].tolist() == ["roi-1"]


def test_only_whole_word_reference_columns_are_reference_columns():
    # "ref" inside a word made phenotype_ml_consensus_refined and file_prefix the
    # default column for manual labels (clicks would have changed them).
    from spacemap.viewer.reference import is_reference_column

    for name in ("phenotype_ref1", "phenotype_ref2", "phenotype_manual", "manual_reference", "phenotype_reference_v2"):
        assert is_reference_column(name), name
    for name in ("phenotype_ml_consensus_refined", "file_prefix", "phenotype_leiden", "preference_score"):
        assert not is_reference_column(name), name


def test_options_mark_which_phenotype_columns_are_reference_columns(tmp_path):
    table = tmp_path / "cells.csv"
    pd.DataFrame({"batch_id": ["a"], "label": [1], "file_prefix": ["x"], "phenotype_ml_consensus": ["T"],
                  "phenotype_ml_consensus_refined": ["T"], "phenotype_ref1": ["B"]}).to_csv(table, index=False)
    found = reference.options([str(table)], str(table))
    assert found["reference_columns"] == ["phenotype_ref1"]
    assert "phenotype_ml_consensus_refined" in found["columns"]  # still offered, with a confirmation in the viewer
