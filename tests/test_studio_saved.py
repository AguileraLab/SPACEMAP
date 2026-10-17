"""Jobs > Saved results: Studio's columns wait beside the table; written in only when asked; CSV copies on request."""

import time

import pandas as pd
import pytest

from spacemap import dataio, pending_columns
from spacemap_gui.studio import saved


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    root = tmp_path / "study"
    folder = root / "outputs/batch_corrected"
    folder.mkdir(parents=True)
    table = folder / "combined.csv"
    pd.DataFrame({"cell_id": ["c0", "c1"], "CD3": [1.0, 2.0]}).to_csv(table, index=False)
    return {"root": str(root), "unified": str(table), "mappings": {}}


def test_saved_columns_are_listed_discarded_and_written_in_when_asked(project):
    table = project["unified"]
    for column in ("phenotype_a", "phenotype_b"):
        pending_columns.save(table, {column: ["T", "B"]}, anchor_column="cell_id", anchor_values=["c0", "c1"])
    found = saved.state(project)
    assert [item["column"] for item in found["pending"]] == ["phenotype_a", "phenotype_b"] and not found["parquet"]
    assert "phenotype_a" not in pd.read_csv(table).columns  # the CSV is untouched until asked
    found = saved.act(project, "discard", {"columns": ["phenotype_b"]})
    assert [item["column"] for item in found["pending"]] == ["phenotype_a"]
    with pytest.raises(ValueError):
        saved.act(project, "discard", {"columns": ["phenotype_nope"]})
    saved.act(project, "write", {})
    for _ in range(300):
        if pending_columns.commit_status(table).get("state") in {"done", "failed"}:
            break
        time.sleep(0.2)
    assert pending_columns.commit_status(table)["state"] == "done"
    assert pd.read_csv(table)["phenotype_a"].tolist() == ["T", "B"] and saved.state(project)["pending"] == []


def test_a_parquet_table_gets_a_csv_copy_only_on_request(project, tmp_path):
    folder = tmp_path / "study/outputs/batch_corrected"
    table = folder / "batch_corrected_merged_20260927_1950.parquet"
    pd.DataFrame({"cell_id": ["c0", "c1"], "CD3_corr": [0.5, 0.7]}).to_parquet(table)
    pending_columns.save(table, {"phenotype_x": ["T", "B"]}, anchor_column="cell_id", anchor_values=["c0", "c1"])
    listed = saved.state({**project, "unified": str(table)})
    assert listed["parquet"] and [item["name"] for item in listed["tables"]] == [table.name]
    assert listed["tables"][0]["in_use"] and not listed["tables"][0]["csv"] and not list(folder.glob("batch_*.csv"))
    with pytest.raises(ValueError):
        saved.act(project, "export", {"name": "../combined.csv"})
    result = saved.export({"table": str(table)}, tmp_path, lambda *args: None)
    copy = pd.read_csv(result["output"])
    assert (
        copy["phenotype_x"].tolist() == ["T", "B"]
        and saved.state(project)["tables"][0]["csv"] == table.with_suffix(".csv").name
    )
    assert dataio.columns(table)[:2] == ["cell_id", "CD3_corr"]


def test_loaded_parquet_outside_batch_folder_exports_predictions_and_scores(project, tmp_path, monkeypatch):
    from spacemap_gui.studio import jobs
    table = tmp_path / "imported.parquet"
    pd.DataFrame({"cell_id": ["c0", "c1"]}).to_parquet(table)
    pending_columns.save(table, {"phenotype_mr": ["T", "Others"], "mr_score": [None, .4]},
                         anchor_column="cell_id", anchor_values=["c0", "c1"])
    loaded = {**project, "unified": str(table)}
    found = saved.state(loaded)
    assert found["tables"][0]["in_use"] and found["tables"][0]["name"] == table.name
    calls = []
    monkeypatch.setattr(jobs, "start", lambda root, kind, args: calls.append((kind, args)) or {"id":"job"})
    saved.act(loaded, "export", {"name":table.name, "table":str(table)})
    assert calls == [("export_table", {"table":str(table)})]
    with pytest.raises(ValueError):
        saved.act(loaded, "export", {"name":table.name, "table":str(tmp_path / "unlisted.parquet")})
    result = saved.export(calls[0][1], tmp_path, lambda *args: None)
    csv = pd.read_csv(result["output"])
    assert csv.phenotype_mr.tolist() == ["T", "Others"]
    assert pd.isna(csv.mr_score[0]) and csv.mr_score[1] == .4
    assert saved.state(loaded)["tables"][0]["csv"] == "imported.csv"
