"""Cancellation and failed publication cannot leave partial tagged tables."""

import os
import pandas as pd
import pytest

from spacemap_gui.studio import job_stop, tagging
from spacemap_gui.studio.tagging_outputs import StagedTables, cleanup_staging


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_cancel_before_publish_keeps_previous_output_and_input(tmp_path, suffix):
    folder = tmp_path / "job"
    folder.mkdir()
    target = tmp_path / ("tagged" + suffix)
    previous = pd.DataFrame({"cell": [1, 2], "tag": [0, 0]})
    tagging.write_table(previous, target)
    report = lambda *a: None
    guard = job_stop.OutputGuard(folder)
    report.protect_output = guard.protect
    with pytest.raises(job_stop.Cancelled):
        with StagedTables(tmp_path, folder) as staged:
            staged.stage(previous.assign(tag=1), target, tagging.write_table)
            pd.testing.assert_frame_equal(tagging.read_table(target), previous)
            (folder / "cancel").touch()
            staged.publish(report)
    guard.close()
    pd.testing.assert_frame_equal(tagging.read_table(target), previous)
    assert not list(tmp_path.glob(".studio-tagging-*"))


def test_failed_second_replacement_rolls_back_first(tmp_path, monkeypatch):
    targets = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for target in targets:
        target.write_text("original")
    replace = os.replace

    def fail_second(source, target):
        if str(source).endswith("1.csv"):
            raise OSError("simulated publication failure")
        return replace(source, target)

    with StagedTables(tmp_path) as staged:
        for target in targets:
            staged.stage(pd.DataFrame({"value": [1]}), target, tagging.write_table)
        monkeypatch.setattr(os, "replace", fail_second)
        with pytest.raises(OSError, match="publication failure"):
            staged.publish(lambda *a: None)
        assert [p.read_text() for p in targets] == ["original", "original"]


def test_serialization_failure_never_changes_destination(tmp_path):
    target = tmp_path / "tagged.csv"
    target.write_text("original")

    def interrupted_write(frame, path):
        path.write_text("incomplete")
        raise job_stop.ComputationCancelled("stopped")

    with pytest.raises(job_stop.ComputationCancelled):
        with StagedTables(tmp_path) as staged:
            staged.stage(None, target, interrupted_write)
    assert target.read_text() == "original"
    assert not list(tmp_path.glob(".studio-tagging-*"))


def test_reaper_only_removes_owned_scratch(tmp_path):
    folder = tmp_path / "job"
    staged = StagedTables(tmp_path, folder)
    (staged.directory / ".owner").write_text("another job")
    cleanup_staging(folder)
    assert staged.directory.exists()
    (staged.directory / ".owner").write_text(str(folder.resolve()))
    cleanup_staging(folder)
    assert not staged.directory.exists()
    assert not (folder / "tagging-staging.json").exists()
