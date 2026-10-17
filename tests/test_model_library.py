"""Trained models are shared through a model library and imported into other projects."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A disposable SPACEMAP home (settings and models), so nothing real is touched."""
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SPACEMAP_MODEL_LIBRARY", raising=False)
    monkeypatch.delenv("SPACEMAP_MODELS_DIR", raising=False)
    return tmp_path


def _project(base, name):
    root = base / name
    (root / "outputs").mkdir(parents=True)
    (root / f"{name}.json").write_text(json.dumps({"project_name": name, "directory_mappings": {}, "tab_configurations": {}}))
    return root


def test_models_are_shared_and_imported_into_another_project(home):
    from sklearn.ensemble import RandomForestClassifier

    from spacemap_gui.studio import artifact_learning, model_library, reference_models, training

    first, second = _project(home, "first"), _project(home, "second")
    # A Cellpose model, an artifact model version and a phenotyping model trained in the first project.
    _, cellpose = training.folders(first)
    (cellpose / "models").mkdir(parents=True)
    (cellpose / "models" / "tonsil_v1").write_bytes(b"weights")
    (cellpose / "models" / "tonsil_v1.json").write_text(json.dumps({"base": "cyto3", "examples": 12}))
    clots = artifact_learning.folder(first, "clot")
    clots.mkdir(parents=True)
    joblib.dump({"model": "forest"}, clots / "clot_v1.joblib")
    (clots / "versions.json").write_text(json.dumps([{"id": "clot_v1", "name": "Blood clots 1", "created": "2026-10-01", "images": 3}]))
    table = first / "outputs" / "cells.csv"
    pd.DataFrame({"CD3": [1.0, 2.0], "CD8": [3.0, 4.0]}).to_csv(table, index=False)
    model = RandomForestClassifier(n_estimators=4, random_state=0).fit(np.array([[0, 0], [1, 1], [0, 1], [1, 0]]), ["T", "B", "T", "B"])
    reference_models.directory().mkdir(parents=True)
    joblib.dump({"model": model, "rows": np.array([0, 1]), "oob": np.zeros((2, 2))}, reference_models.model_path("a" * 32))
    reference_models.model_path("a" * 32).with_suffix(".json").write_text(json.dumps(
        {"method": "Random Forest", "column": "phenotype_ref", "markers": ["CD3", "CD8"], "table_path": str(table.resolve()),
         "table": "cells.csv", "trained": "2026-10-01", "sklearn": "x"}))

    kinds = sorted(m["kind"] for m in model_library.project_models(first))
    assert kinds == ["artifact", "cellpose", "phenotyping"]
    assert model_library.project_models(second) == []
    # Share all three; the library describes each.
    for item in model_library.project_models(first):
        model_library.share(first, item["kind"], item["id"], "first")
    shared = model_library.entries()
    assert {e["kind"] for e in shared} == {"cellpose", "artifact", "phenotyping"}
    assert all(e["source_project"] == "first" for e in shared)
    with pytest.raises(ValueError):
        model_library.share(first, "cellpose", "tonsil_v1")  # already there
    # The second project imports them; each step lists them there.
    other = second / "outputs" / "cells.csv"
    pd.DataFrame({"CD3": [5.0], "CD8": [6.0], "CD20": [1.0]}).to_csv(other, index=False)
    assert reference_models.available(other) == []  # trained on another table, not imported yet
    from spacemap_gui.viewer.api import _library_models
    assert [entry["model_id"] for entry in _library_models(other)] == ["a" * 32]
    for entry in shared:
        model_library.import_model(second, entry["kind"], entry["entry"])
    assert [m["name"] for m in training.examples(second)["models"]] == ["tonsil_v1"]
    versions = json.loads((artifact_learning.folder(second, "clot") / "versions.json").read_text())
    assert versions[0]["id"] == "clot_v1" and versions[0]["imported_from"]["project"] == "first"
    assert [(m["model_id"], m["column"]) for m in reference_models.available(other)] == [("a" * 32, "phenotype_ref")]
    pd.DataFrame({"CD3": [5.0]}).to_csv(other, index=False)
    assert reference_models.available(other) == []  # a table without CD8 cannot use it
    with pytest.raises(ValueError):
        model_library.import_model(second, "cellpose", shared[0]["entry"] if shared[0]["kind"] == "cellpose" else
                                   next(e["entry"] for e in shared if e["kind"] == "cellpose"))  # already imported


def test_the_library_can_live_in_any_folder(home):
    from spacemap_gui.studio import model_library

    assert model_library.location().name == "library"
    lab = home / "lab_drive"
    lab.mkdir()
    model_library.set_location(str(lab / "spacemap_models"))
    assert model_library.location() == lab / "spacemap_models" and (lab / "spacemap_models").is_dir()
    with pytest.raises(ValueError):
        model_library.set_location("relative/path")
    with pytest.raises(ValueError):
        model_library.set_location(str(home / "missing" / "deeper"))


def test_an_imported_model_predicts_without_its_training_cells():
    from sklearn.ensemble import RandomForestClassifier

    from spacemap.viewer import consensus_model

    model = RandomForestClassifier(n_estimators=4, random_state=0).fit(np.array([[0, 0], [1, 1], [0, 1], [1, 0]]), ["T", "B", "T", "B"])
    artifact = {"model": model, "rows": np.array([-1]), "oob": np.zeros((1, 2))}  # as request() prepares it for another table
    probabilities = consensus_model.predict(artifact, np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))
    assert np.isfinite(probabilities).all() and probabilities.shape == (2, 2)


def test_an_imported_model_adapts_to_missing_markers(home):
    """A phenotype decided by one marker (an H&E-like signal) survives losing a noise marker; losing the deciding
    marker is reported as a clear drop in held-out agreement."""
    from sklearn.ensemble import RandomForestClassifier

    from spacemap_gui.studio import reference_models

    rng = np.random.default_rng(0)
    he = np.concatenate([rng.normal(0, 1, 300), rng.normal(5, 1, 300)])
    noise = rng.normal(0, 1, 600)
    targets = np.array(["Other"] * 300 + ["RBC"] * 300)
    values = np.column_stack([he, noise]).astype(np.float32)
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(values, targets)
    sample = reference_models.training_sample(values, targets, limit=400)
    assert len(sample["targets"]) <= 400 and set(sample["targets"]) == {"Other", "RBC"}
    model_id = "b" * 32
    reference_models.directory().mkdir(parents=True)
    joblib.dump({"model": model, "rows": np.array([0]), "oob": np.zeros((1, 2)), "provenance": {"method": "Random Forest"},
                 "training_sample": {**sample, "markers": ["HE_eosin", "CD99"]}}, reference_models.model_path(model_id))
    meta = {"method": "Random Forest", "column": "phenotype_ref", "markers": ["HE_eosin", "CD99"], "table_path": "/elsewhere/cells.csv",
            "imported": True, "adaptable": True, "sklearn": "x", "model_id": model_id}
    reference_models.model_path(model_id).with_suffix(".json").write_text(json.dumps(meta))
    table = home / "cells.csv"
    pd.DataFrame({"HE_eosin": [0.0, 5.0]}).to_csv(table, index=False)  # CD99 was not measured here
    listed = reference_models.available(table)
    assert len(listed) == 1 and listed[0]["missing_markers"] == ["CD99"]
    artifact, provenance = reference_models.adapt(model_id, ["HE_eosin"])
    assert provenance["markers"] == ["HE_eosin"] and provenance["missing_markers"] == ["CD99"]
    assert provenance["held_out_balanced_accuracy"] > 0.95  # the deciding marker is still there
    assert list(artifact["model"].predict(np.array([[0.0], [5.0]]))) == ["Other", "RBC"]
    _, without_signal = reference_models.adapt(model_id, ["CD99"])
    assert without_signal["held_out_balanced_accuracy"] < 0.7 < without_signal["full_model_accuracy_same_cells"]
    assert reference_models.adapt(model_id, ["HE_eosin"])[1] == provenance  # cached
    pd.DataFrame({"CD3": [1.0]}).to_csv(table, index=False)
    assert reference_models.available(table) == []  # none of its markers here
