"""Jobs use the real RESOLVE path and publish only complete, still-current data."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, pending_columns
from spacemap.resolve_live import LiveResolve, Panel, Settings
from spacemap_gui.studio import jobs, tasks


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    project = tmp_path / "study"
    project.mkdir()
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.gamma(2, 30, (120, 3)), columns=["CD3", "CD8", "CD20"])
    frame["batch_id"] = np.repeat(["roi1", "roi2"], 60)
    frame["label"] = list(range(1, 61)) * 2
    frame["cell_id"] = range(120)
    frame.loc[3, "CD8"] = np.nan
    table = project / "cells.csv"
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    panel = {
        "criteria_cols": ["Criteria 1"],
        "table": [
            {"Phenotype": "T", "Criteria 1": ["CD3"], "Supportive": ["CD8"]},
            {"Phenotype": "CD8", "Criteria 1": ["CD3", "CD8"], "Must not have": ["CD20"]},
            {"Phenotype": "B", "Criteria 1": ["CD20"]},
        ],
    }
    panel_file = project / "panel.json"
    panel_file.write_text(json.dumps(panel))
    thresholds = project / "thresholds.csv"
    pd.DataFrame(
        [{"batch_id": "roi1", "CD3": 40, "CD8": 60, "CD20": 50}, {"batch_id": "roi2", "CD3": 60, "CD8": 30, "CD20": 70}]
    ).to_csv(thresholds, index=False)
    settings = {
        "table": str(table),
        "panel": str(panel_file),
        "thresholds": str(thresholds),
        "markers": ["CD3", "CD8", "CD20"],
        "roi_column": "batch_id",
        "label_column": "label",
        "cell_id_col": "cell_id",
        "output_col": "phenotype_studio",
        "options": {
            "tie_score_transform": "asinh_difference",
            "norm_z1": True,
            "norm_z2": True,
            "norm_cdf": True,
            "norm_nlog": True,
        },
    }
    settings["revision"] = tasks.source_revision(settings)
    return project, settings, frame, Panel.from_payload(panel)


def test_all_rois_equal_live_resolve_without_rewriting_csv(inputs):
    project, settings, frame, panel = inputs
    original = Path(settings["table"]).read_bytes()
    events = []
    result = tasks.resolve_all(settings, project, lambda message, progress=None: events.append(message))
    assert result["rois"] == 2 and result["left_out"] == 1
    output = dataio.load(settings["table"], columns=["phenotype_studio"])["phenotype_studio"]
    thresholds = pd.read_csv(settings["thresholds"]).set_index("batch_id")
    for roi, rows in frame.groupby("batch_id", sort=False):
        expected = (
            LiveResolve(
                rows[settings["markers"]].to_numpy(),
                settings["markers"],
                panel,
                Settings.from_config(settings["options"]),
            )
            .assign(thresholds.loc[roi].to_dict())
            .phenotypes
        )
        assert output.loc[rows.index].fillna("missing").tolist() == pd.Series(expected).fillna("missing").tolist()
    assert Path(settings["table"]).read_bytes() == original
    assert pending_columns.pending_names(settings["table"]) == ["phenotype_studio"]
    assert any("roi2" in event for event in events)


def test_missing_thresholds_fail_without_partial_result(inputs):
    project, settings, _, _ = inputs
    pd.read_csv(settings["thresholds"]).iloc[:1].to_csv(settings["thresholds"], index=False)
    settings["revision"] = tasks.source_revision(settings)
    with pytest.raises(ValueError, match="roi2"):
        tasks.resolve_all(settings, project, lambda *args: None)
    assert not pending_columns.pending_names(settings["table"])


def test_changed_source_is_not_applied(inputs):
    project, settings, _, _ = inputs
    changed = False

    def report(message, progress=None):
        nonlocal changed
        if "roi2" in message and not changed:
            changed = True
            Path(settings["thresholds"]).write_text(Path(settings["thresholds"]).read_text() + "\n")

    with pytest.raises(ValueError, match="changed during"):
        tasks.resolve_all(settings, project, report)
    assert not pending_columns.pending_names(settings["table"])


def wait_job(project, job_id, timeout=180):
    # Scientific imports can be slow on shared filesystems. This verifies the
    # result and lifecycle, not a machine-specific startup performance budget.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = jobs.status(project, job_id)
        if result["status"] not in jobs.ACTIVE:
            return result
        time.sleep(0.1)
    pytest.fail("Worker did not finish in time: " + jobs.log(project, job_id)[-4000:])


def test_background_worker_result_progress_and_resource_log(inputs):
    project, settings, _, _ = inputs
    job = jobs.start(project, "resolve_all", settings, request_id="same-start")
    repeated = jobs.start(project, "resolve_all", settings, request_id="same-start")
    assert repeated["id"] == job["id"]
    result = wait_job(project, job["id"])
    assert result["status"] == "done", jobs.log(project, job["id"])
    assert result["progress"] == 1 and result["result"]["column"] == "phenotype_studio"
    assert {"duration_s", "cpu_time_s", "peak_ram_gb", "peak_gpu_memory_gb"} <= set(result["usage"])
    assert "roi2" in jobs.log(project, job["id"])


def test_cancel_before_work_saves_no_result(inputs):
    project, settings, _, _ = inputs
    job = jobs.start(project, "resolve_all", settings)
    jobs.cancel(project, job["id"])
    result = wait_job(project, job["id"])
    assert result["status"] == "cancelled", jobs.log(project, job["id"])
    assert not pending_columns.pending_names(settings["table"])


def test_worker_failure_and_job_paths_are_checked(inputs):
    project, settings, _, _ = inputs
    job = jobs.start(project, "resolve_all", {**settings, "output_col": "label"})
    result = wait_job(project, job["id"])
    assert result["status"] == "failed" and "phenotype_" in result["message"]
    with pytest.raises(ValueError):
        jobs.status(project, "../../outside")
    with pytest.raises(ValueError):
        jobs.start(project, "run_shell", {})


def test_a_job_folder_without_its_status_yet_is_not_listed(tmp_path):
    (jobs.root(tmp_path) / ("a" * 32)).mkdir(parents=True)
    assert jobs.recent(tmp_path) == []


@pytest.mark.parametrize("same_timestamp", [False, True])
def test_recent_order_is_unchanged_when_old_job_gains_cached_files(tmp_path, same_timestamp):
    import os
    from spacemap_workflows.leiden_job import write_json

    older, newer = "1" * 32, "2" * 32
    for i, job_id in enumerate((older, newer), start=1):
        folder = jobs.root(tmp_path) / job_id
        folder.mkdir(parents=True)
        write_json(folder / "request.json", {"kind": "name_clusters", "settings": {}})
        write_json(folder / "status.json", {"id": job_id, "kind": "name_clusters", "status": "failed",
                                           "started": f"2026-10-01T00:00:00.00000{i}+00:00"})
        mtime = 1_000_000_000 if same_timestamp else i * 1_000_000_000
        os.utime(folder / "request.json", ns=(mtime, mtime))
        os.utime(folder, ns=(i * 1_000_000_000, i * 1_000_000_000))
    (jobs.root(tmp_path) / older / "cached_profile.json").write_text("{}")
    assert [job["id"] for job in jobs.recent(tmp_path)] == [newer, older]
