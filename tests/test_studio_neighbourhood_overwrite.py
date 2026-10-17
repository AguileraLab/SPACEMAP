"""Neighbourhood replacement updates groups and names together, with explicit consent."""

from pathlib import Path

import pandas as pd
import pytest

from spacemap import dataio, labels, pending_columns
from spacemap_gui.studio import cluster_names, jobs, leiden, neighbourhoods, tasks
from spacemap_gui.studio.settings import read_config, revision
from test_studio_chat import _job
from test_studio_commands import run, setup  # noqa: F401
from test_studio_neighbourhoods import job, table  # noqa: F401


@pytest.fixture(params=["csv", "parquet"])
def existing(table, request, monkeypatch):  # noqa: F811
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    frame = pd.read_csv(table)
    frame["neighborhood"] = "Old name"
    frame["neighborhood_cn"] = "CN 99"
    frame["phenotype_ref1"] = None
    table = table.with_suffix("." + request.param)
    if request.param == "csv":
        frame.to_csv(table, index=False)
    else:
        frame.to_parquet(table, index=False)
    pending_columns.save(table, {"neighborhood": ["Saved name"] * len(frame)},
                         anchor_column="cell_id", anchor_values=frame.cell_id)
    labels.append(table, [{"row": 0, "column": "neighborhood", "value": "Edited name"},
                          {"row": 1, "column": "neighborhood_cn", "value": "CN 98"},
                          {"row": 2, "column": "phenotype_ref1", "value": "Manual T"}])
    settings = {**job(table), "overwrite_columns": ["neighborhood", "neighborhood_cn"]}
    return table, settings


def test_replace_is_explicit_and_inputs_are_protected(existing):
    table, settings = existing
    with pytest.raises(ValueError, match="Overwrite existing results"):
        tasks.validate_table({**settings, "overwrite_columns": []})
    assert tasks.validate_table(settings)
    for names in (["phenotype_ref1"], ["neighborhood_other"], ["x_centroid"], "neighborhood"):
        with pytest.raises(ValueError, match="Only the selected"):
            tasks.validate_table({**settings, "overwrite_columns": names})
    # A user can choose a neighbourhood column as an input region, but cannot replace it in that run.
    with pytest.raises(ValueError, match="Only the selected"):
        tasks.validate_table({**settings, "values": {**settings["values"], "roi_col": "neighborhood"}})


def test_success_replaces_both_outputs_and_resets_names(existing, tmp_path):
    table, settings = existing
    original = table.read_bytes()
    result = neighbourhoods.run({**settings, "images": ["R2"], "image_col": "batch_id"}, tmp_path, lambda *a: None)
    assert result["overwrite"] and set(result["columns"]) == {"neighborhood", "neighborhood_cn"}
    found = dataio.load(table)
    assert found.neighborhood.equals(found.neighborhood_cn)
    assert found.loc[found.batch_id == "R1", "neighborhood"].isna().all()
    assert set(found.loc[found.batch_id == "R2", "neighborhood"]) == {"CN 0", "CN 1"}
    assert found.phenotype_ref1.iloc[2] == "Manual T"
    assert labels.pending_columns(table) == ["phenotype_ref1"]
    assert table.read_bytes() == original
    pending_columns.commit(table, rebuild_fast_copy=False)
    persisted = pd.read_csv(table) if table.suffix == ".csv" else pd.read_parquet(table)
    pd.testing.assert_series_equal(persisted.neighborhood.astype("string"), found.neighborhood.astype("string"))


@pytest.mark.parametrize("failure", ["changed_source", "second_column", "label_cleanup"])
def test_failed_replacement_keeps_both_previous_outputs(existing, monkeypatch, failure):
    table, settings = existing
    if failure == "changed_source":
        labels.append(table, [{"row": 3, "column": "phenotype_ref1", "value": "New reference"}])
    before, token = dataio.load(table), tasks.source_revision(settings)
    if failure == "second_column":
        write = pending_columns._write_parquet
        def fail(frame, target, meta):
            if meta.get("column") == "neighborhood":
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
        tasks.save_result(settings, {"neighborhood_cn": ["CN 0"] * len(before),
                                    "neighborhood": ["CN 0"] * len(before)}, before.cell_id, lambda *a: None)
    pd.testing.assert_frame_equal(dataio.load(table), before)
    assert tasks.source_revision(settings) == token


def test_command_overwrite_and_names_stay_with_current_run(setup, monkeypatch):  # noqa: F811
    _, table, root, _ = setup
    frame = pd.read_csv(table)
    frame["cell_id"] = range(len(frame))
    frame["neighborhood"] = "Earlier name"
    frame["neighborhood_cn"] = "CN 0"
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    chosen = neighbourhoods.save(root, {"phenotype_col": "phenotype_ml_consensus"},
                                revision(neighbourhoods.values(read_config(root))))
    old = _job(root, "neighbourhoods", {"table": str(table), "column": "neighborhood_cn",
               "phenotype_column": "neighborhood", "clusters": [{"name": "CN 0", "cells": 5}]})
    starts = []
    def start(root, kind, settings, **kwargs):
        root = Path(root)
        starts.append(settings)
        if kind == "neighbourhoods":
            result = tasks.save_result(settings, {"neighborhood_cn": ["CN 0"] * len(frame),
                                                  "neighborhood": ["CN 0"] * len(frame)}, frame.cell_id, lambda *a: None)
            result.update(phenotype_column=settings["names_col"], markers=settings["markers"],
                          clusters=[{"name": "CN 0", "cells": 5}], overwrite=True)
        else:
            result = leiden.name_clusters(settings, root, lambda *a: None)
        return jobs.status(root, _job(root, kind, result, settings))
    monkeypatch.setattr(jobs, "start", start)
    blocked = run(setup, "run_neighbourhoods", confirmed=True)
    assert blocked["status"] == "error" and "Overwrite existing results" in blocked["message"]
    assert run(setup, "run_neighbourhoods", {"overwrite": "true"}, confirmed=True)["status"] == "error"
    assert run(setup, "run_neighbourhoods", {"overwrite": True, "settings_revision": "stale"}, confirmed=True)["status"] == "error"
    assert not starts
    answer = run(setup, "run_neighbourhoods", {"overwrite": True, "settings_revision": revision(chosen)}, confirmed=True)
    assert answer["status"] == "done", answer
    assert set(starts[0]["overwrite_columns"]) == {"neighborhood", "neighborhood_cn"}
    latest = jobs.recent(root)[0]["id"]
    assert cluster_names.saved_names(root, table, "neighborhood_cn")["names"] == {}
    assert run(setup, "name_clusters", {"job_id": old, "names": {"CN 0": "Stale"}}, confirmed=True)["status"] == "error"
    for name in ("Immune zone", "Updated zone"):
        named = run(setup, "name_clusters", {"job_id": latest, "names": {"CN 0": name}}, confirmed=True)
        assert named["status"] == "done", named
        assert starts[-1]["output_col"] == "neighborhood"
        assert dataio.load(table).neighborhood.eq(name).all()
    assert "neighborhood_2" not in dataio.columns(table)
