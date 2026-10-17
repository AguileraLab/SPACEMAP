"""Saved consensus models must be reusable, correctly keyed and independent of image work."""

import time

import joblib
import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.viewer import consensus_model as m, reference


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    table = tmp_path / "cells.csv"
    rng = np.random.default_rng(8)
    values = np.r_[rng.normal(10, 1, (80, 2)), rng.normal(100, 1, (80, 2))]
    pd.DataFrame(
        {
            "batch_id": ["r1"] * 80 + ["r2"] * 80,
            "label": list(range(1, 81)) * 2,
            "M0": values[:, 0],
            "M1": values[:, 1],
            m.COLUMN: ["T"] * 80 + ["B"] * 80,
            "phenotype_ref1": [None] * 160,
        }
    ).to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    return m.settings_for({"roi": "r1"}, str(table), ["M1", "M0"], "phenotype_ref1")


def wait(settings, **kwargs):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        result = m.request(settings, **kwargs)
        if result["status"] != "preparing":
            return result
        time.sleep(0.03)
    raise AssertionError("Model worker did not finish")


def test_training_is_bounded_and_uses_all_rois(settings, monkeypatch):
    monkeypatch.setattr(m, "PER_CLASS", 12)
    artifact = m.train(settings)
    assert len(artifact["rows"]) == 24
    assert artifact["provenance"]["available_examples"] == 160
    assert artifact["provenance"]["training_counts"] == {"T": 12, "B": 12}
    assert artifact["provenance"]["oob_balanced_accuracy"] > 0.95
    assert artifact["provenance"]["source"] == "table"
    assert not reference.path_for(settings["table"]).exists()


def test_saved_model_reload_does_not_train_again(settings, monkeypatch):
    first = wait(settings, label=1)
    assert first["status"] == "ready", first
    assert first["choices"][0]["phenotype"] == "T"
    target = m.path_for(settings)
    assert target.exists()
    with m._LOCK:
        m._JOBS.pop(str(target))

    def no_training(*args, **kwargs):
        raise AssertionError("A saved model must load without retraining")

    monkeypatch.setattr(m.subprocess, "Popen", no_training)
    again = wait(settings, label=1)
    assert again == first
    other = m.request({**settings, "roi": "r2"}, label=1)
    assert other["choices"][0]["phenotype"] == "B"
    assert other["provenance"]["revision"] == first["provenance"]["revision"]


def test_manual_edits_flags_and_image_settings_do_not_retrain(settings):
    stamp = m.signature(settings)
    reference.save(settings["table"], "r1", "phenotype_ref1", {1: "Manual"}, "manual-example")
    reference.set_flag(settings["table"], "r1", "phenotype_ref1", 1, True)
    assert m.signature(settings) == stamp
    assert m.signature({**settings, "roi": "r2", "destination": "other", "backgrounds": {"M0": 500}}) == stamp
    reference.save(settings["table"], "r1", m.COLUMN, {1: "B"}, "consensus-change")
    assert m.signature(settings) != stamp


def test_marker_edits_and_consensus_flags_invalidate(settings):
    stamp = m.signature(settings)
    labels.append(settings["table"], [{"row": 1, "column": "M0", "value": "22"}])
    assert m.signature(settings) != stamp
    assert m.selected_values(settings, [1])[0, 0] == 22
    stamp = m.signature(settings)
    reference.set_flag(settings["table"], "r1", m.COLUMN, 1, True)
    assert m.signature(settings) != stamp
    artifact = m.train(settings)
    assert 0 not in artifact["rows"]
    assert artifact["provenance"]["flagged_excluded"] == 1


def test_training_cells_use_only_out_of_bag_votes(settings):
    artifact = m.train(settings)
    rows = artifact["rows"][[0, 90]]
    probabilities = m.predict(artifact, m.selected_values(settings, rows), rows)
    np.testing.assert_array_equal(probabilities, artifact["oob"][[0, 90]])


def test_missing_measurements_and_rare_classes_disclosed(settings):
    labels.append(
        settings["table"], [{"row": 0, "column": "M0", "value": "nan"}, {"row": 1, "column": m.COLUMN, "value": "Rare"}]
    )
    artifact = m.train(settings)
    assert artifact["provenance"]["missing_excluded"] == 1
    assert artifact["provenance"]["needs_examples"] == ["Rare"]
    assert np.isnan(m.predict(artifact, m.selected_values(settings, [0]), np.array([0]))).all()


def test_csv_fallback_and_row_order_match_fast_copy(settings, monkeypatch):
    rows = np.array([99, 0, 80, 3])
    expected = m.selected_values(settings, rows)
    monkeypatch.setattr(dataio, "fast_copy_info", lambda table: {"current": False})
    np.testing.assert_array_equal(expected, m.selected_values(settings, rows, streaming=True))
    assert m.train(settings)["provenance"]["examples"] == 160


def test_scan_reuses_model_and_excludes_assigned_cells(settings):
    assert wait(settings, prepare=True)["status"] == "ready"
    reference.save(settings["table"], "r1", "phenotype_ref1", {1: "T"}, "already-labelled")
    first = wait(settings, scan=True)
    assert first["status"] == "ready", first
    assert first["label"] != 1 and first["remaining"] == 79
    next_cell = m.request(settings, scan=True, exclude=[first["label"]])
    assert next_cell["label"] not in [1, first["label"]]
    assert next_cell["remaining"] == 78


def test_source_change_during_training_is_rejected(settings, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    original = RandomForestClassifier.fit

    def fit(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        reference.save(settings["table"], "r1", m.COLUMN, {1: "B"}, "changed-while-training")
        return result

    monkeypatch.setattr(RandomForestClassifier, "fit", fit)
    with pytest.raises(ValueError, match="changed during training"):
        m.train(settings)


def test_corrupt_cache_rebuilt(settings):
    target = m.path_for(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"incomplete")
    assert wait(settings, prepare=True)["status"] == "ready"
    assert joblib.load(target)["revision"] == m.signature(settings)


def test_api_prepares_without_a_cell_and_checks_registered_table(settings, monkeypatch):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api, registry
    from test_viewer_cell_profiles import marker_image
    from pathlib import Path

    image = Path(settings["table"]).with_suffix(".ome.tiff")
    marker_image(image, np.ones((2, 16, 16), np.uint16))
    token = registry.register_object("cells", image=str(image), roi="r1", tables={"t0": settings["table"]})
    monkeypatch.setattr(api, "_markers", lambda *args, **kwargs: ["M0", "M1"])
    with TestClient(api.app) as client:
        result = client.post(f"/cells/{token}/suggestions", json={"mode": "consensus_rf", "t": "t0", "prepare": True})
        assert result.status_code == 200, result.text
        assert result.json()["status"] in {"preparing", "ready"}
        assert wait(settings, prepare=True)["status"] == "ready"
        result = client.post(f"/cells/{token}/suggestions", json={"mode": "consensus_rf", "t": "t0", "label": 1})
        assert result.status_code == 200 and result.json()["choices"][0]["phenotype"] == "T"
        assert client.post(f"/cells/{token}/suggestions", json={"mode": "consensus_rf", "t": "t0"}).status_code == 400
        assert (
            client.post(
                f"/cells/{token}/suggestions", json={"mode": "consensus_rf", "t": "/other.csv", "prepare": True}
            ).status_code
            == 400
        )


def test_training_column_is_required(settings, monkeypatch):
    with pytest.raises(ValueError, match="phenotype_ml_consensus"):
        monkeypatch.setattr(dataio, "columns", lambda *args, **kwargs: ["M0", "M1"])
        m.settings_for({"roi": "r1"}, settings["table"], ["M0", "M1"], "phenotype_ref1")
