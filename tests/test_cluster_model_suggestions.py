"""Saved manual-reference models suggest cluster names without AI or relabelling cells."""
import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.viewer import consensus_model
from spacemap_gui.studio import chat, cluster_names, reference_models, subphenotype_guidance
from test_studio_reference_models import training  # noqa: F401
from test_studio_chat import _job


@pytest.fixture
def named(training):
    table, settings, root = training
    identifier = reference_models.train(settings, root / "trained", lambda *_: None)["models"][0]["model_id"]
    frame = dataio.load(table)
    frame["clusters"] = ["Cluster 0"] * 60 + ["Cluster 1"] * 60
    target = root / "target.parquet"
    frame.to_parquet(target, index=False)
    return root, target, identifier, frame


def test_names_come_from_saved_model_and_keep_manual_evidence_separate(named, monkeypatch):
    root, table, identifier, frame = named
    frame["phenotype_ref1"] = "Different manual label"
    frame.to_parquet(table, index=False)
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    for model in (RandomForestClassifier, ExtraTreesClassifier):
        monkeypatch.setattr(model, "fit", lambda *_a, **_k: pytest.fail("Do not retrain a complete-panel saved model"))
    before = table.read_bytes()
    result = cluster_names.suggest_model(table, "clusters", identifier)
    assert [result["clusters"][f"Cluster {i}"]["suggestions"][0]["label"] for i in range(2)] == ["A", "B"]
    assert all(group["top"][0]["label"] == "Different manual label" for group in result["clusters"].values())
    assert result["validation"]["phenotypes"] and result["features"] == ["M0", "M1"]
    assert table.read_bytes() == before
    json.dumps(cluster_names.evidence(result), allow_nan=False)


def test_model_without_target_labels_reads_all_cluster_samples_once(named, monkeypatch):
    _, table, identifier, frame = named
    frame.drop(columns=["phenotype_ref1", "batch_id"]).to_parquet(table, index=False)
    monkeypatch.setattr(subphenotype_guidance, "SAMPLE", 9)
    reads = []
    original = consensus_model.selected_values
    def take(settings, rows, **kwargs):
        reads.append(len(rows))
        return original(settings, rows, **kwargs)
    monkeypatch.setattr(consensus_model, "selected_values", take)
    found = cluster_names.suggest_model(table, "clusters", identifier)
    assert reads == [18]
    assert found["reference_column"] == "" and found["labelled"] == 0
    assert all(c["model_sampled"] == c["model_scored"] == 9 for c in found["clusters"].values())
    assert found["clusters"]["Cluster 0"]["suggestions"][0]["label"] == "A"


@pytest.mark.parametrize("marker", ["M0", "M0_corr"])
def test_fewer_cluster_features_use_saved_training_examples_and_original_scale(named, marker):
    root, table, identifier, frame = named
    # Clustering may use a corrected measurement; the model must still read its
    # own measured column and fitted transform, never these display/run values.
    frame["M0_corr"] = -999
    frame.to_parquet(table, index=False)
    weights = reference_models.model_path(identifier).read_bytes()
    choices = cluster_names.suggestion_sources(root, table, [marker])
    selected = next(m for m in choices["models"] if m["id"] == identifier)
    assert selected["compatible"] and selected["features"] == ["M0"] and selected["missing"] == ["M1"]
    found = cluster_names.suggest_model(table, "clusters", identifier, features=[marker])
    assert found["features"] == ["M0"] and found["missing"] == ["M1"]
    assert found["validation"]["features"] == ["M0"]
    assert [found["clusters"][f"Cluster {i}"]["suggestions"][0]["label"] for i in range(2)] == ["A", "B"]
    assert reference_models.model_path(identifier).read_bytes() == weights
    assert cluster_names.suggest_model(table, "clusters", identifier, features=[marker], cached_only=True) is found
    assert cluster_names.suggest_model(table, "clusters", identifier, cached_only=True) is None


def test_missing_target_features_are_supported_but_no_overlap_has_no_prediction(named):
    root, table, identifier, frame = named
    frame.drop(columns=["M1", "phenotype_ref1", "batch_id"]).to_parquet(table, index=False)
    found = cluster_names.suggest_model(table, "clusters", identifier, features=["M0", "M1"])
    assert found["features"] == ["M0"] and found["clusters"]["Cluster 1"]["suggestions"][0]["label"] == "B"
    choices = cluster_names.suggestion_sources(root, table, ["Unrelated marker"])
    assert not any(m["compatible"] for m in choices["models"])
    with pytest.raises(ValueError, match="No model features match"):
        cluster_names.suggest_model(table, "clusters", identifier, features=["Unrelated marker"])


def test_listing_and_cached_prompt_never_load_model_or_cell_rows(named, monkeypatch):
    root, table, identifier, _ = named
    found = cluster_names.suggest_model(table, "clusters", identifier)
    job = _job(root, "leiden", {"table": str(table), "column": "clusters", "markers": ["M0", "M1"],
        "clusters": [{"name": "Cluster 0", "cells": 60}, {"name": "Cluster 1", "cells": 60}],
        "profile": {"version": 2, "columns": ["M0", "M1"], "rows": ["Cluster 0", "Cluster 1"], "values": [[-1, 0], [1, 0]]}})
    def unexpected(*_args, **_kwargs):
        pytest.fail("Metadata listing and cached prompt preparation must not read rows or weights")
    monkeypatch.setattr(reference_models, "_source", unexpected)
    monkeypatch.setattr(dataio, "load", unexpected)
    import joblib
    monkeypatch.setattr(joblib, "load", unexpected)
    options = cluster_names.suggestion_sources(root, table)
    assert any(m["id"] == identifier and m["compatible"] for m in options["models"])
    assert cluster_names.suggest_model(table, "clusters", identifier, cached_only=True) is found
    result = chat.results({"root": str(root), "unified": str(table)}, "Suggest cluster names",
                          {"job_id": job, "model_id": identifier}, cached_only=True)[0]
    assert result["profile"] and result["reference"]["model"]["id"] == identifier
    assert result["reference"]["per_cluster"]["Cluster 0"]["suggested"][0]["label"] == "A"
    assert "not independent accuracy" in result["reference"]["score_meaning"]


def test_pending_manual_labels_invalidate_model_evidence(named):
    _, table, identifier, _ = named
    first = cluster_names.suggest_model(table, "clusters", identifier)
    labels.append(table, [{"row": 2, "column": "phenotype_ref1", "value": "A"}])
    assert cluster_names.suggest_model(table, "clusters", identifier, cached_only=True) is None
    second = cluster_names.suggest_model(table, "clusters", identifier)
    assert second["labelled"] == first["labelled"] + 1


def test_ai_can_request_selected_model_evidence_for_one_cluster(named):
    from spacemap_gui.studio import investigation
    from spacemap_gui.viewer import registry
    root, table, identifier, _ = named
    channel = "cluster-model-investigation"
    project = {"root": str(root), "unified": str(table), "tab": "tab7"}
    registry.publish(channel, {"id": channel, "layers": []}, project=project)
    job = _job(root, "som", {"table": str(table), "column": "clusters", "markers": ["M0", "M1"],
        "clusters": [{"name": "Cluster 0", "cells": 60}, {"name": "Cluster 1", "cells": 60}]})
    helper = investigation.Investigation(channel, project, {"explorer": {"job_id": job, "model_id": identifier}})
    record, images = helper.run("cluster_reference", {"clusters": ["Cluster 1"]})
    found = record["data"]
    assert not images and list(found["per_cluster"]) == ["Cluster 1"]
    assert found["per_cluster"]["Cluster 1"]["suggested"][0]["label"] == "B"
    assert found["model"]["id"] == identifier and found["coverage"].startswith("1 of 2")
    with pytest.raises(ValueError, match="cluster"):
        helper.run("cluster_reference", {"clusters": ["Cluster 404"]})


def test_ai_evidence_is_bounded_without_mutating_cached_predictions():
    hints = {"model_id": "saved", "clusters": {f"Cluster {i}": {"suggestions": [{"label": "A", "score": .9}]}
                                               for i in range(100)},
             "validation": {"phenotypes": [{"phenotype": "A"}, {"phenotype": "Unused"}]}}
    found = cluster_names.evidence(hints)
    assert len(found["per_cluster"]) == 80 and found["coverage"].startswith("80 of 100")
    selected = cluster_names.evidence(hints, ["Cluster 99"])
    assert list(selected["per_cluster"]) == ["Cluster 99"]
    assert selected["validation"]["phenotypes"] == [{"phenotype": "A"}]
    assert len(hints["clusters"]) == 100 and len(hints["validation"]["phenotypes"]) == 2


@pytest.mark.parametrize("method", ["leiden", "som", "vortex"])
def test_api_saved_model_suggestions_for_every_clustering_without_ai(named, method):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import registry, api
    root, table, identifier, frame = named
    frame.drop(columns="phenotype_ref1").to_parquet(table, index=False)
    before = dataio.load(table)
    channel = "model-cluster-" + method
    registry.publish(channel, {"id": channel, "layers": []}, project={"root": str(root), "unified": str(table), "tab": "tab7"})
    job = _job(root, method, {"table": str(table), "column": "clusters", "markers": ["M0", "M1"],
               "clusters": [{"name": "Cluster 0", "cells": 60}, {"name": "Cluster 1", "cells": 60}]})
    client = TestClient(api.app)
    response = client.post(f"/studio/{channel}/cluster_suggestions", json={"job_id": job, "model_id": identifier})
    assert response.status_code == 200, response.text
    assert response.json()["clusters"]["Cluster 1"]["suggestions"][0]["label"] == "B"
    pd.testing.assert_frame_equal(dataio.load(table), before)
