"""Reference-only prediction, optional morphology, confidence and durable saves."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, ml_mr_utils as ml, pending_columns
from spacemap_gui.studio import schemas, tasks


@pytest.fixture
def measurements(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    # These tests exercise fitting, validation and persistence; plotting has
    # separate coverage in the existing Studio ensemble integration test.
    for name in ("evaluate_ml_mr_performance", "generate_performance_heatmaps",
                 "generate_expression_heatmap", "generate_confidence_stratified_heatmaps"):
        monkeypatch.setattr(ml, name, lambda *a, **kw: None)
    frame = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(100)], "batch_id": "roi",
        "M0": 1.0, "M1": 1.0,
        "area": np.r_[np.arange(50) + 5., np.arange(50) + 150.],
        "eccentricity": [0.1] * 50 + [0.9] * 50,
        "phenotype_ref1": ["A"] * 20 + [None] * 30 + ["B"] * 20 + ["Others"] * 30,
    })
    frame.loc[25, "phenotype_ref1"] = "Unassigned"
    path = tmp_path / "cells.parquet"
    frame.to_parquet(path, index=False)
    settings = {"table": str(path), "markers": ["M0", "M1"],
                "reference_column": "phenotype_ref1", "methods": [],
                "output_col": "phenotype_my_predictions", "confidence_column": "my_score",
                "refined_column": "phenotype_my_filtered", "confidence_threshold": .8,
                "low_confidence_label": "Others", "excluded_phenotypes": ["Others"],
                "morphology_cols": ["area", "eccentricity"]}
    settings["revision"] = tasks.source_revision(settings)
    return frame, path, settings


def test_reference_only_learns_morphology_and_saves_numeric_scores(measurements, tmp_path):
    frame, path, settings = measurements
    before = path.read_bytes()
    result = tasks.mr_ml(settings, tmp_path, lambda *args: None)
    assert path.read_bytes() == before
    assert result["scored_cells"] == 60 and result["reference_only"]
    assert result["features"]["morphology"] == ["area", "eccentricity"]
    assert result["column"] == "phenotype_my_filtered"
    got = dataio.load(path)
    known = frame.phenotype_ref1.isin(["A", "B"])
    assert got.loc[known, "phenotype_my_filtered"].equals(frame.loc[known, "phenotype_ref1"].rename("phenotype_my_filtered"))
    assert got.loc[known, "my_score"].isna().all()
    assert got.loc[~known, "my_score"].between(0, 1).all()
    expected = np.where(got.area < 100, "A", "B")
    # Markers are constant: morphology must actually be used for this to pass.
    assert (got.phenotype_my_predictions == expected).mean() > .95
    entries = pending_columns.pending(path)
    assert len(entries) == 3 and all(Path(entry["file"]).suffix == ".parquet" for entry in entries)
    pending_columns.commit(path)
    reread = pd.read_parquet(path)
    pd.testing.assert_series_equal(got.my_score, reread.my_score)
    assert pd.api.types.is_numeric_dtype(reread.my_score)
    assert reread.phenotype_ref1.equals(frame.phenotype_ref1)


def test_low_score_filter_preserves_references_without_inflating_validation(measurements, tmp_path, monkeypatch):
    frame, path, settings = measurements
    # Deliberately incorrect low-support predictions, including known B cells.
    monkeypatch.setattr(ml, "predict_weighted_ensemble", lambda x, *a, **kw: (np.full(len(x), "A"), np.full(len(x), .4)))
    tasks.mr_ml(settings, tmp_path, lambda *args: None)
    got = dataio.load(path)
    known = frame.phenotype_ref1.isin(["A", "B"])
    assert (got.loc[~known, "phenotype_my_predictions"] == "A").all()
    assert (got.loc[~known, "phenotype_my_filtered"] == "Others").all()
    assert (got.loc[~known, "my_score"] == .4).all()
    assert got.loc[known, "phenotype_my_filtered"].tolist() == frame.loc[known, "phenotype_ref1"].tolist()
    summary = (tmp_path / "results/ml_mr_summary.txt").read_text()
    assert "Validation accuracy: 0.500" in summary


def test_missing_measurements_preserve_known_labels_and_leave_unknowns_unscored(measurements, tmp_path):
    frame, path, settings = measurements
    frame.loc[[0, 26], "area"] = np.nan
    frame.to_parquet(path, index=False)
    settings["revision"] = tasks.source_revision(settings)
    tasks.mr_ml(settings, tmp_path, lambda *args: None)
    got = dataio.load(path)
    assert got.loc[0, "phenotype_my_filtered"] == "A"
    assert pd.isna(got.loc[26, "phenotype_my_filtered"])
    assert got.loc[[0, 26], "my_score"].isna().all()
    assert got.cell_id.tolist() == frame.cell_id.tolist()


@pytest.mark.parametrize("changes", [{"confidence_column":"area"}, {"refined_column":"phenotype_ref1"},
    {"confidence_column":"phenotype_my_predictions"}, {"confidence_threshold":float('nan')},
    {"morphology_cols":["cell_id"]}, {"morphology_cols":["phenotype_ref1"]}])
def test_invalid_outputs_and_leaking_features_rejected_before_training(measurements, changes):
    _, _, settings = measurements
    with pytest.raises(ValueError):
        tasks.validate_mr_ml({**settings, **changes})


def test_non_numeric_morphology_is_actionable(measurements, tmp_path):
    frame, path, settings = measurements
    frame["area"] = "not a measurement"
    frame.to_parquet(path, index=False)
    settings["revision"] = tasks.source_revision(settings)
    messages = []
    with pytest.raises(ValueError, match="must contain numeric"):
        tasks.mr_ml(settings, tmp_path, lambda message, *a: messages.append(message))
    assert any("'area' must contain numeric measurements" in m for m in messages)
    assert pending_columns.pending(path) == []


def test_empty_methods_allowed_without_weakening_other_commands():
    value = schemas.COMMANDS["run_mr_ml"].validate({"column":"phenotype_new", "reference_column":"phenotype_ref1", "methods":[], "morphology_cols":["area"]})
    assert value["methods"] == [] and value["morphology_cols"] == ["area"]
    with pytest.raises(ValueError):
        schemas.COMMANDS["show_markers"].validate({"markers":[]})
