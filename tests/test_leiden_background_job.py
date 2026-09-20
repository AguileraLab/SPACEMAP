"""Leiden must survive page reruns and retain results without a UI session."""

import json
import logging
import os
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
import psutil
import pytest

from spacemap_workflows import leiden_job


def settings_for(input_path):
    return dict(
        full_input_path=str(input_path),
        markers_selected=["a", "b", "c", "d"],
        cell_id_col="cell_id",
        batch_id_col="",
        enable_filter=True,
        filter_col="corrected",
        filter_val="1",
        norm_z1=True,
        norm_z2=True,
        norm_cdf=True,
        norm_nlog=True,
        norm_sinh=False,
        cofactor=5.0,
        low_expr_metric="raw_sum",
        enable_low_expr_filter=True,
        low_expr_threshold=0.01,
        resolution=1.0,
        n_neighbors=5,
        n_pcs=3,
        use_intelligent=True,
        subsample_size=60,
        random_seed=42,
        compute_umap=False,
        meta_clusters=0,
        use_gpu=False,
        fresh_assignment_mode=False,
        output_col_name="phenotype_leiden",
    )


def dataset(path, n=120):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(1, 5, size=(n, 4)), columns=list("abcd"))
    frame["cell_id"] = np.arange(n)
    frame["corrected"] = 1
    frame.loc[0, "corrected"] = 0
    frame.loc[1, list("abcd")] = 0  # Preserved low-expression bucket.
    frame.loc[2, "a"] = np.nan
    frame.to_csv(path, index=False)
    return frame


def audit_for(project_root, settings):
    return dict(
        run_id="test-leiden",
        event_type="analysis_run",
        started_at="2026-09-18T00:00:00",
        started_perf=time.perf_counter(),
        input_csv=settings["full_input_path"],
        markers=settings["markers_selected"],
        parameters=settings,
        method_key="leiden",
        project_root=str(project_root),
        project_name="Test project",
    )


def wait_for_job(project_root, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = leiden_job.latest_job(project_root)
        if job["status"] not in {"starting", "running"}:
            return job
        time.sleep(0.1)
    pytest.fail("Worker timed out: " + leiden_job.read_log(job["job_dir"]))


def test_real_worker_survives_ui_interrupt_and_reconnect(tmp_path, monkeypatch):
    from streamlit.runtime.scriptrunner_utils.exceptions import StopException

    # Keep this tiny scientific run cheap on large HPC hosts.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(key, "2")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    input_path = tmp_path / "input.csv"
    source = dataset(input_path)
    settings = settings_for(input_path)
    with pytest.raises(StopException):
        launched = leiden_job.start_job(
            tmp_path, settings, audit_for(tmp_path, settings)
        )
        raise StopException()  # What previously unwound the clustering itself.
    same = leiden_job.start_job(tmp_path, settings)
    assert same["job_id"] == launched["job_id"]  # No duplicate while running.
    assert leiden_job.latest_job(tmp_path / "other-project") is None
    settings["resolution"] = 9.0  # Later widget edits cannot alter the request.
    request = json.loads((Path(launched["job_dir"]) / "request.json").read_text())
    assert request["settings"]["resolution"] == 1.0
    job = wait_for_job(tmp_path)
    assert job["status"] == "success", leiden_job.read_log(job["job_dir"])
    result = json.loads((Path(job["job_dir"]) / "result.json").read_text())
    frame = pd.read_parquet(result["leiden_temp_file"])
    assert frame.cell_id.tolist() == source.cell_id.drop(index=[0, 2]).tolist()
    assert frame.loc[frame.cell_id == 1, "leiden_cluster"].item() == -1
    assert frame.leiden_cluster.notna().all()
    assert result["leiden_low_expr_count"] == 1
    assert "umap_x" not in frame
    assert set(result["leiden_norm_cols"]).issubset(frame.columns)
    audit = pd.read_csv(tmp_path / "outputs/phenotyping/logs/leiden_runs.csv")
    assert audit.status.tolist() == ["success"]
    assert audit.input_rows.item() == len(source)
    assert audit.output_rows.item() == len(frame)
    assert "COMPLETED: Total Leiden Pipeline" in leiden_job.read_log(job["job_dir"])


def test_intelligent_pipeline_has_no_session_state_access(tmp_path, monkeypatch):
    from spacemap_workflows import leiden_pipeline

    class NoSession:
        def __getattr__(self, name):
            raise AssertionError("The worker accessed Streamlit session state")

        def __contains__(self, key):
            raise AssertionError("The worker accessed Streamlit session state")

    import streamlit as st

    monkeypatch.setattr(st, "session_state", NoSession())
    input_path = tmp_path / "input.csv"
    dataset(input_path, n=50005)
    settings = settings_for(input_path)
    calls = []

    def cluster(frame, norm_cols, parameters, logger):
        calls.append((len(frame), list(norm_cols), parameters.copy()))
        return np.arange(len(frame)) % 3, None

    monkeypatch.setattr(
        leiden_pipeline.ph_leiden_utils, "intelligent_leiden_clustering", cluster
    )
    job_dir = tmp_path / "run"
    job_dir.mkdir()
    result = leiden_pipeline.run_pipeline(
        dict(
            project_root=str(tmp_path),
            settings=settings,
            audit=audit_for(tmp_path, settings),
        ),
        job_dir,
        logging.getLogger("test-leiden"),
    )
    assert calls[0][0] == 50002
    assert calls[0][1] == [f"{marker}_norm" for marker in "abcd"]
    assert calls[0][2]["subsample_size"] == 60
    assert result["leiden_n_clusters"] == 3
    frame = pd.read_parquet(result["leiden_temp_file"])
    assert len(frame) == 50003
    assert frame.cell_id.is_monotonic_increasing


def test_worker_saves_exception_and_failed_audit(tmp_path):
    input_path = tmp_path / "input.csv"
    dataset(input_path)
    settings = settings_for(input_path)
    settings["markers_selected"] = ["missing_marker"]
    request = dict(
        project_root=str(tmp_path),
        settings=settings,
        audit=audit_for(tmp_path, settings),
    )
    job_dir = tmp_path / "run"
    job_dir.mkdir()
    leiden_job.write_json(job_dir / "request.json", request)
    assert leiden_job.run_job(job_dir) == 1
    job = leiden_job.read_job(job_dir)
    assert job["status"] == "failed"
    assert "missing_marker" in job["error"]
    assert "ValueError" in job["traceback"]
    assert not (job_dir / "result.json").exists()
    audit = pd.read_csv(tmp_path / "outputs/phenotyping/logs/leiden_runs.csv")
    assert audit.status.tolist() == ["failed"]
    assert "missing_marker" in audit.error.item()


def test_dead_worker_is_interrupted_not_reported_as_running(tmp_path):
    leiden_job.write_json(
        tmp_path / "status.json", {"job_id": "run", "status": "running"}
    )
    leiden_job.write_json(
        tmp_path / "process.json",
        {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "create_time": psutil.Process().create_time() - 100,  # Reused PID.
        },
    )
    assert leiden_job.read_job(tmp_path)["status"] == "interrupted"
    # A committed result status wins even when its process has exited.
    leiden_job.write_json(
        tmp_path / "status.json", {"job_id": "run", "status": "success"}
    )
    assert leiden_job.read_job(tmp_path)["status"] == "success"


def test_reconnect_loads_result_once_and_preserves_phenotype_edits(
    tmp_path, monkeypatch
):
    from spacemap_gui.tabs.tab7_phenotyping import leiden_clustering

    state = {}
    monkeypatch.setattr(leiden_clustering.st, "session_state", state)
    result = {
        "leiden_run_id": "run",
        "leiden_run_input_path": str(tmp_path / "input.csv"),
        "leiden_temp_file": str(tmp_path / "results.parquet"),
        "leiden_phenotype_map": {"0": "Unassigned"},
    }
    Path(result["leiden_temp_file"]).touch()
    leiden_job.write_json(tmp_path / "result.json", result)
    job = {"status": "success", "job_dir": str(tmp_path)}
    restore = leiden_clustering._restore_leiden_job_result
    restore(job, result["leiden_run_input_path"])
    assert state["leiden_phenotype_map"] == {0: "Unassigned"}
    state["leiden_phenotype_map"][0] = "T cells"
    state.pop("_leiden_applied_job")  # Browser reconnect with persisted project state.
    restore(job, result["leiden_run_input_path"])
    assert state["leiden_phenotype_map"] == {0: "T cells"}
    state.clear()
    restore(job, str(tmp_path / "different.csv"))
    assert "leiden_temp_file" not in state


def test_timer_reports_rerun_without_touching_ui(monkeypatch, caplog):
    from spacemap_gui.tabs.tab7_phenotyping import common
    from streamlit.runtime.scriptrunner_utils.exceptions import StopException

    def no_ui(*args, **kwargs):
        pytest.fail("Timer tried to update Streamlit during interruption")

    monkeypatch.setattr(common.logger, "info", no_ui)
    timer = common.Timer("Leiden Clustering")
    timer.start_time = time.time()
    with caplog.at_level(logging.ERROR, logger="SPACEMAP_UI"):
        timer.__exit__(StopException, StopException(), None)
    assert "INTERRUPTED: Leiden Clustering" in caplog.text
    assert "StopException" in caplog.text
