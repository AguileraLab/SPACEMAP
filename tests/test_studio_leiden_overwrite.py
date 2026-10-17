"""Explicit Leiden replacement keeps cluster membership, names and saved overlays consistent."""

import json
from pathlib import Path

import pandas as pd
import pytest

from spacemap import dataio, labels, pending_columns
from spacemap_gui.studio import chat, cluster_names, jobs, leiden, tasks
from test_studio_chat import _job
from test_studio_commands import run, setup  # noqa: F401


@pytest.fixture(params=["csv", "parquet"])
def existing(tmp_path, monkeypatch, request):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    frame = pd.DataFrame({
        "cell_id": ["a", "b", "c"], "CD3": [20., 40., 60.],
        "phenotype_leiden": ["T", "T", "B"],
        "phenotype_leiden_clusters": ["Cluster 8", "Cluster 8", "Cluster 9"],
        "phenotype_ref1": ["T", None, "B"],
    })
    table = tmp_path / ("cells." + request.param)
    if request.param == "csv":
        frame.to_csv(table, index=False)
    else:
        frame.to_parquet(table, index=False)
    pending_columns.save(table, {"phenotype_leiden": ["CD4", "CD4", "B"]},
                         anchor_column="cell_id", anchor_values=frame.cell_id)
    labels.append(table, [{"row": 0, "column": "phenotype_leiden", "value": "Old edit"},
                          {"row": 1, "column": "phenotype_leiden_clusters", "value": "Cluster 99"},
                          {"row": 2, "column": "phenotype_ref1", "value": "Manual B"}])
    values = {**leiden.DEFAULTS, "markers": ["CD3"], "use_gpu": False}
    settings = {
        "table": str(table), "markers": ["CD3"], "cell_id_col": "cell_id",
        "output_col": "phenotype_leiden_clusters", "phenotype_col": "phenotype_leiden",
        "project_root": str(tmp_path), "pipeline": leiden.pipeline_settings(values, table, 5.0),
        "overwrite_columns": ["phenotype_leiden", "phenotype_leiden_clusters"],
    }
    settings["revision"] = tasks.source_revision(settings)
    return table, settings


def test_replacement_requires_explicit_choice(existing):
    table, settings = existing
    before = dataio.load(table)
    with pytest.raises(ValueError, match="Overwrite existing results"):
        tasks.validate_table({**settings, "overwrite_columns": []})
    assert tasks.validate_table(settings)
    pd.testing.assert_frame_equal(dataio.load(table), before)
    for forbidden in (["CD3"], ["cell_id"], ["phenotype_ref1"], "phenotype_leiden"):
        with pytest.raises(ValueError, match="Only the selected"):
            tasks.validate_table({**settings, "overwrite_columns": forbidden})


def test_rerun_resets_names_and_removes_only_replaced_label_overrides(existing, monkeypatch, tmp_path):
    table, settings = existing
    original_bytes = table.read_bytes()
    from spacemap_workflows import leiden_pipeline

    def compute(*args):
        result = tmp_path / "computed.parquet"
        # Out-of-order results exercise cell-ID matching rather than row positions.
        pd.DataFrame({"cell_id": ["c", "a", "b"], "leiden_cluster": [1, 0, 0],
                      "CD3": [60., 20., 40.], "CD3_norm": [1., -1., 0.]}).to_parquet(result)
        return {"leiden_temp_file": str(result), "leiden_n_clusters": 2, "leiden_norm_cols": ["CD3_norm"]}

    monkeypatch.setattr(leiden_pipeline, "_compute", compute)
    found = leiden.run(settings, tmp_path, lambda *args: None)
    assert found["overwrite"] and set(found["columns"]) == set(settings["overwrite_columns"])
    result = dataio.load(table)
    assert result.phenotype_leiden.tolist() == ["Cluster 0", "Cluster 0", "Cluster 1"]
    assert result.phenotype_leiden.equals(result.phenotype_leiden_clusters)
    assert result.phenotype_ref1.iloc[2] == "Manual B"
    assert labels.pending_columns(table) == ["phenotype_ref1"]
    assert table.read_bytes() == original_bytes  # uses the same fast pending-column save path
    assert result.CD3.tolist() == [20., 40., 60.]
    pending_columns.commit(table, rebuild_fast_copy=False)
    persisted = pd.read_parquet(table) if table.suffix == ".parquet" else pd.read_csv(table)
    assert persisted.phenotype_leiden.tolist() == result.phenotype_leiden.tolist()
    assert persisted.phenotype_ref1.iloc[2] == "Manual B"
    assert not pending_columns.pending_names(table)


@pytest.mark.parametrize("failure", ["changed_source", "second_column", "label_cleanup"])
def test_failed_replacement_keeps_previous_results(existing, monkeypatch, failure):
    table, settings = existing
    if failure == "changed_source":
        labels.append(table, [{"row": 1, "column": "phenotype_ref1", "value": "New reference"}])
    before, revision = dataio.load(table), tasks.source_revision(settings)
    if failure == "second_column":
        write = pending_columns._write_parquet
        def fail(frame, target, meta):
            if meta.get("column") == "phenotype_leiden":
                raise OSError("disk write failed")
            return write(frame, target, meta)
        monkeypatch.setattr(pending_columns, "_write_parquet", fail)
    elif failure == "label_cleanup":
        clear = labels.clear_columns
        def fail(*args):
            clear(*args)
            raise OSError("label cleanup failed")
        monkeypatch.setattr(labels, "clear_columns", fail)
    with pytest.raises((ValueError, OSError), match="changed during|failed"):
        tasks.save_result(settings, {settings["output_col"]: ["Cluster 0"] * 3,
                                    settings["phenotype_col"]: ["Cluster 0"] * 3},
                          before.cell_id, lambda *args: None)
    pd.testing.assert_frame_equal(dataio.load(table), before)
    assert tasks.source_revision(settings) == revision


def test_command_uses_checkbox_and_names_only_the_current_run(setup, monkeypatch):  # noqa: F811
    _, table, root, _ = setup
    frame = pd.read_csv(table)
    frame["cell_id"] = range(len(frame))
    frame["phenotype_leiden"] = "Old name"
    frame["phenotype_leiden_clusters"] = "Cluster 0"
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    cache = leiden.cache_path(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"markers": ["M0", "M1"]}))
    old = _job(root, "leiden", {"table": str(table), "column": "phenotype_leiden_clusters",
        "phenotype_column": "phenotype_leiden", "markers": ["M0", "M1"],
        "clusters": [{"name": "Cluster 0", "cells": 5}]})
    _job(root, "name_clusters", {"column": "phenotype_leiden"}, {
        "table": str(table), "clusters_col": "phenotype_leiden_clusters", "cluster_job_id": old,
        "names": {"Cluster 0": "Old name"}})
    starts = []
    def start(root, kind, settings, **kwargs):
        root = Path(root)
        starts.append(settings)
        if kind == "leiden":
            result = {"table": str(table), "column": settings["output_col"],
                      "phenotype_column": settings["phenotype_col"], "markers": settings["markers"],
                      "clusters": [{"name": "Cluster 0", "cells": 5}], "overwrite": True}
        else:
            result = leiden.name_clusters(settings, root, lambda *args: None)
        job_id = _job(root, kind, result, settings)
        return jobs.status(root, job_id)
    monkeypatch.setattr(jobs, "start", start)
    blocked = run(setup, "run_leiden", confirmed=True)
    assert blocked["status"] == "error" and "Overwrite existing results" in blocked["message"]
    assert not starts
    assert run(setup, "run_leiden", {"overwrite": "true"}, confirmed=True)["status"] == "error"
    started = run(setup, "run_leiden", {"overwrite": True}, confirmed=True)
    assert started["status"] == "done", started
    assert set(starts[0]["overwrite_columns"]) == {"phenotype_leiden", "phenotype_leiden_clusters"}
    latest = jobs.recent(root)[0]
    assert cluster_names.saved_names(root, table, "phenotype_leiden_clusters")["names"] == {}
    assert chat._saved_names(root, jobs.recent(root)).get("phenotype_leiden_clusters", {}) == {}
    assert run(setup, "name_clusters", {"job_id": old, "names": {"Cluster 0": "Wrong run"}}, confirmed=True)["status"] == "error"
    named = run(setup, "name_clusters", {"job_id": latest["id"], "names": {"Cluster 0": "New name"}}, confirmed=True)
    assert named["status"] == "done", named
    assert starts[-1]["output_col"] == "phenotype_leiden"
    assert dataio.load(table).phenotype_leiden.eq("New name").all()
    assert "phenotype_leiden_2" not in dataio.columns(table)
    assert cluster_names.saved_names(root, table, "phenotype_leiden_clusters", cluster_job_id=latest["id"])["names"] == {"Cluster 0": "New name"}
