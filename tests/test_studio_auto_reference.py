import copy
import pytest

from spacemap_gui.studio import auto_reference as auto, tasks, reference_models
from test_studio_reference_models import training  # noqa: F401


@pytest.fixture
def queue(training, monkeypatch):  # noqa: F811
    table, settings, root = training
    project = {"root":str(root)}
    monkeypatch.setattr(auto, "_context", lambda channel: (project, "table"))
    monkeypatch.setattr(auto.manual_reference, "training_settings", lambda channel,column,checked: {**settings,"training_column":column,"checked_only":checked})
    monkeypatch.setattr(auto.time, "time", lambda: 100.)
    stamp = ["first"]
    monkeypatch.setattr(auto, "_stamp", lambda settings: stamp[0])
    runs = []
    monkeypatch.setattr(auto.jobs, "recent", lambda root: copy.deepcopy(runs))
    def start(root, kind, settings):
        item = {"id":str(len(runs)+1), "kind":kind, "status":"running", "settings":settings}
        runs.append(item)
        return copy.deepcopy(item)
    monkeypatch.setattr(auto.jobs, "start", start)
    return project, stamp, runs, settings


def test_auto_training_debounces_changes_and_keeps_latest_update(queue):
    _, stamp, runs, _ = queue
    assert not auto.status("x")["enabled"]
    auto.configure("x", "phenotype_ref1", True)
    assert auto.tick("x", 100)["state"] == "waiting"
    stamp[0] = "edit"
    auto.tick("x", 114)
    assert not runs
    auto.tick("x", 128)
    assert not runs
    assert auto.tick("x", 129)["state"] == "training"
    assert len(runs) == 1 and runs[0]["settings"]["auto_train"]
    # Many polls / windows cannot create duplicate jobs. New edits queue one update.
    for now in (130, 140, 150):
        stamp[0] = str(now)
        auto.tick("x", now)
    assert len(runs) == 1
    runs[0].update(status="done", result={"models":[{"model_id":"a"}], "auto_train":True})
    result = auto.tick("x", 170)
    assert result["completed"]["id"] == "1" and result["state"] == "waiting"
    auto.tick("x", 189)
    assert len(runs) == 2
    runs[1].update(status="done", result={"models":[{"model_id":"b"}], "auto_train":True})
    assert auto.tick("x", 200)["state"] == "ready"
    auto.tick("x", 500)
    assert len(runs) == 2
    auto.configure("x", None, False)
    stamp[0] = "new edit"
    assert auto.tick("x", 600)["state"] == "off"
    assert len(runs) == 2


def test_auto_training_waits_for_other_jobs_and_retries_only_after_changes(queue):
    project, stamp, runs, _ = queue
    auto.configure("x", "phenotype_ref1", True)
    runs.append({"id":"other", "kind":"leiden", "status":"running"})
    auto.tick("x", 100)
    assert auto.tick("x", 130)["state"] == "waiting" and len(runs) == 1
    runs[0]["status"] = "done"
    auto.tick("x", 131)
    runs[1].update(status="done", result={"models":[], "waiting":"Add more labelled cells."})
    assert auto.tick("x", 200)["message"] == "Add more labelled cells."
    auto.tick("x", 250)
    assert len(runs) == 2
    stamp[0] = "more labels"
    auto.tick("x", 251)
    project["read_only"] = True
    auto.tick("x", 280)
    assert len(runs) == 2
    with pytest.raises(ValueError, match="read-only"):
        auto.configure("x", "phenotype_ref1", True)
    project["read_only"] = False
    auto.tick("x", 281)
    assert len(runs) == 3


def test_auto_worker_waits_with_few_labels_and_fits_without_writing_labels(training):  # noqa: F811
    from spacemap.viewer import reference
    table, settings, root = training
    original = table.read_bytes()
    result = tasks.train_reference_model({**settings, "auto_train":True}, root, lambda *args: None)
    assert len(result["models"]) == 1 and result["models"][0]["auto_train"]
    assert result["models"][0]["held_out_balanced_accuracy"] > .9
    assert table.read_bytes() == original
    # Sparse checked examples wait; subsequent approved labels can trigger another run.
    reference.set_flag(table, "roi1", "phenotype_ref1", 1, False)
    waiting = tasks.train_reference_model({**settings, "auto_train":True, "checked_only":True}, root, lambda *args: None)
    assert not waiting["models"] and "five" in waiting["waiting"]


def test_switching_training_source_does_not_activate_an_old_run(queue):
    _, _, runs, _ = queue
    auto.configure("x", "phenotype_ref1", True)
    auto.tick("x", 100)
    auto.tick("x", 116)
    auto.configure("x", "phenotype_leiden", True)
    assert auto.tick("x", 120)["state"] == "waiting"
    runs[0].update(status="done", result={"models":[{"column":"phenotype_ref1"}]})
    result = auto.tick("x", 150)
    assert result["completed"] is None and result["column"] == "phenotype_leiden"
    auto.tick("x", 180)
    assert runs[-1]["settings"]["training_column"] == "phenotype_leiden"


def test_auto_model_survives_new_labels_during_fit(training):  # noqa: F811
    from spacemap.viewer import reference
    table, settings, root = training
    changed = []
    def report(message, *args):
        if "Comparing" in message and not changed:
            reference.save(table, "roi1", "phenotype_ref1", {"3":"A"}, "during-training")
            changed.append(True)
    result = reference_models.train({**settings, "auto_train":True}, root, report)
    assert changed and result["models"]
    assert result["models"][0]["label_revision"] == settings["revision"]


def test_small_training_budget_retains_rare_labelled_classes(training, monkeypatch):  # noqa: F811
    import numpy as np
    from spacemap.viewer import consensus_model
    _, settings, _ = training
    values = np.array(["A"]*1000+["B"]*1000+["Rare"]*5, dtype=object)
    monkeypatch.setattr(consensus_model, "_label_batches", lambda settings: iter([(0, values)]))
    rows, targets, _, _ = consensus_model._sample(settings, lambda message: None, limit=100)
    assert len(rows) <= 100 and sum(targets == "Rare") == 5


def test_auto_replaces_previous_fit_and_caches_but_manual_runs_accumulate(training):  # noqa: F811
    import json
    table, settings, root = training
    original = table.read_bytes()
    fit = lambda options: reference_models.train(options, root, lambda *args: None)["models"]
    manual = fit(settings)
    manual_bytes = {m["model_id"]: reference_models.model_path(m["model_id"]).read_bytes() for m in manual}
    first = fit({**settings, "markers": ["M0"], "auto_train": True})[0]
    second = fit({**settings, "auto_train": True})[0]
    assert second["markers"] == ["M0", "M1"]
    reference_models.adapt(second["model_id"], ["M0"])
    cache_files = list((reference_models.directory() / "adapted").glob("*"))
    assert cache_files
    # Simulate duplicates from the previous implementation; all should retire.
    old_id = "e" * 32
    reference_models.model_path(old_id).write_bytes(reference_models.model_path(second["model_id"]).read_bytes())
    reference_models.model_path(old_id).with_suffix(".json").write_text(json.dumps({**second, "model_id": old_id}))
    third = fit({**settings, "auto_train": True})[0]
    listed = reference_models.available(table)
    assert [m["model_id"] for m in listed if m["auto_train"]] == [third["model_id"]]
    assert set(next(m for m in listed if m["auto_train"])["replaces"]) == {first["model_id"], second["model_id"], old_id}
    assert len(list(reference_models.directory().glob("*.joblib"))) == 3
    assert not any(path.exists() for path in cache_files)
    for model in (first, second, {"model_id": old_id}):
        assert not reference_models.model_path(model["model_id"]).exists()
        # A selection saved in another ROI/window follows the same source.
        found = reference_models.request({**settings, "roi": "roi1"}, model["model_id"], label=3)
        assert found["choices"][0]["phenotype"] == "A"
        assert found["provenance"]["model_id"] == third["model_id"]
        assert found["provenance"]["features_total"] == 2
    for key, content in manual_bytes.items():
        assert reference_models.model_path(key).read_bytes() == content
    fit(settings)
    assert len(reference_models.available(table)) == 5  # Four manual, one automatic.
    assert table.read_bytes() == original


def test_auto_replacement_keeps_other_sources_and_imported_models(training):  # noqa: F811
    import json
    table, settings, root = training
    fit = lambda options: reference_models.train({**options, "auto_train": True}, root, lambda *args: None)["models"][0]
    first = fit(settings)
    other_column = {**settings, "training_column": "phenotype_leiden"}
    other_column["revision"] = tasks.source_revision(other_column)
    keep = [fit(other_column)["model_id"]]
    other_table = root / "other.csv"
    other_table.write_bytes(table.read_bytes())
    options = {**settings, "table": str(other_table)}
    options["revision"] = tasks.source_revision(options)
    keep.append(fit(options)["model_id"])
    # Importing an automatic library snapshot pins it as an explicitly kept model.
    meta_path = reference_models.model_path(first["model_id"]).with_suffix(".json")
    meta_path.write_text(json.dumps({**first, "imported": True, "imported_from": "library"}))
    keep.append(first["model_id"])
    saved_bytes = {key: reference_models.model_path(key).read_bytes() for key in keep}
    fit(settings)
    fit(settings)
    assert len(list(reference_models.directory().glob("*.joblib"))) == 4
    for key, content in saved_bytes.items():
        assert reference_models.model_path(key).read_bytes() == content
        assert reference_models.current_model_id(key) == key


@pytest.mark.parametrize("failure", ["artifact", "metadata", "index"])
def test_failed_auto_save_keeps_working_model(training, monkeypatch, failure):  # noqa: F811
    import joblib
    table, settings, root = training
    options = {**settings, "auto_train": True}
    old = reference_models.train(options, root, lambda *args: None)["models"][0]
    path = reference_models.model_path(old["model_id"])
    before = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("Simulated save failure")
    with monkeypatch.context() as patch:
        if failure == "artifact":
            patch.setattr(joblib, "dump", fail)
        elif failure == "index":
            write = reference_models.write_json
            def fail_index(path, value):
                if path.name == ".auto_models.json":
                    fail()
                return write(path, value)
            patch.setattr(reference_models, "write_json", fail_index)
        else:
            patch.setattr(reference_models, "write_json", fail)
        with pytest.raises(OSError, match="Simulated"):
            reference_models.train(options, root, lambda *args: None)
    assert path.read_bytes() == before
    assert len(list(reference_models.directory().glob("*.joblib"))) == 1
    assert not list(reference_models.directory().glob("*.tmp"))
    assert reference_models.available(table)[0]["model_id"] == old["model_id"]
    assert reference_models.request({**settings, "roi": "roi1"}, old["model_id"], label=3)["status"] == "ready"


def test_older_concurrent_auto_fit_cannot_replace_a_newer_success(training):  # noqa: F811
    _, settings, root = training
    nested = []
    def finish_newer(message, *args):
        if "Saving models" in message and not nested:
            nested.append(reference_models.train({**settings, "auto_train": True}, root, lambda *args: None))
    older = reference_models.train({**settings, "auto_train": True}, root, finish_newer)
    winner = nested[0]["models"][0]["model_id"]
    assert older["models"][0]["model_id"] == winner
    assert [path.stem for path in reference_models.directory().glob("*.joblib")] == [winner]


def test_pending_reduced_feature_fit_does_not_recreate_retired_files(training):  # noqa: F811
    _, settings, root = training
    first = reference_models.train({**settings, "auto_train": True}, root, lambda *args: None)["models"][0]
    # A scan/cell request pins the source it started with while fitting in the background.
    snapshot = reference_models._source(first["model_id"])
    newer = reference_models.train({**settings, "auto_train": True, "markers": ["M1"]}, root, lambda *args: None)["models"][0]
    artifact, provenance = reference_models.adapt(first["model_id"], ["M0"], _snapshot=snapshot)
    assert artifact["model"].n_features_in_ == 1 and provenance["markers"] == ["M0"]
    assert not list((reference_models.directory() / "adapted").glob("*"))
    assert reference_models.current_model_id(first["model_id"]) == newer["model_id"]
