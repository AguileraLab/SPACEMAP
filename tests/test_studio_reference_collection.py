"""Pool only reviewed project examples; keep per-image IDs and outputs separate."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, pending_columns
from spacemap_gui.studio import phenotyping_inputs as inputs, reference_collection as collection, reference_features, reference_models
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def study(setup, monkeypatch):  # noqa: F811
    channel, _, root, _ = setup
    monkeypatch.setenv("SPACEMAP_MODELS_DIR", str(root / "models"))
    monkeypatch.setattr(reference_features, "options", lambda channel: {
        "selected": ["M0", "M1"], "missing": [], "allow_missing": True, "features": [{"name": name, "group": "Markers"} for name in ("M0", "M1")]})
    (root / "outputs/rois/second.ome.tiff").touch()
    tables = root / "tables"
    tables.mkdir()
    paths = []
    for name, phenotype, value in [("P11_roi_2", "T", 1.), ("second", "B", 10.)]:
        path = tables / (name + "_cell_features_tagged.csv")
        pd.DataFrame({"label": np.arange(1, 21), "M0": np.linspace(value, value + .1, 20),
                      "M1": np.linspace(value, value + .2, 20), "phenotype_ref1": [phenotype] * 15 + [None] * 5}).to_csv(path, index=False)
        dataio.ensure_fast_copy(path, force=True)
        paths.append(str(path))
    inputs.use_table(channel, paths[0])
    return channel, root, paths


def body(study, **updates):
    return {"action": "train", "tables": study[2], "reference_column": "phenotype_ref1", **updates}


def train(study, **updates):
    settings = collection.prepare(study[0], body(study, **updates))
    folder = study[1] / ("training-" + str(len(list(study[1].glob("training-*")))))
    folder.mkdir()
    return collection.run(settings, folder, lambda *args: None)


def test_one_class_per_image_pools_and_predictions_write_to_each_table(study):
    original = [Path(path).read_bytes() for path in study[2]]
    result = train(study)
    assert result["examples"] == 30
    model_id = result["models"][0]["model_id"]
    _, meta, artifact = reference_models._source(model_id)
    assert set(artifact["model"].classes_) == {"T", "B"}
    assert len(meta["training_sources"]) == 2
    for path, phenotype in zip(study[2], ["T", "B"]):
        local = reference_models._for_table(artifact, path)
        origin = artifact['training_origin']
        held_out = {origin['rows'][i] for i in artifact['validation_sample']['rows'] if origin['tables'][i] == path}
        assert set(local['rows']) | held_out == set(range(15))
        assert not set(local['rows']) & held_out
        assert local['oob'].shape == (len(local['rows']), 2)
        query = {"table": path, "roi": inputs.single_image(study[1], path), "roi_column": "", "label_column": "label", "destination": "phenotype_ref1"}
        assert reference_models.request(query, model_id, label=20)["choices"][0]["phenotype"] == phenotype
    settings = collection.prepare(study[0], body(study, action="predict", model_id=model_id))
    prediction = study[1] / "prediction"
    prediction.mkdir()
    found = collection.run(settings, prediction, lambda *args: None)
    assert found["images"] == 2 and found["cells"] == 40
    for path, before, expected in zip(study[2], original, ["T", "B"]):
        assert Path(path).read_bytes() == before
        frame = dataio.load(path, columns=["phenotype_ref1", "phenotype_mr_ml"])
        assert frame.phenotype_ref1.notna().sum() == 15
        assert frame.phenotype_mr_ml.tolist() == [expected] * 20
        pending_columns.commit(path)
        assert pd.read_csv(path).phenotype_mr_ml.tolist() == [expected] * 20


def test_retraining_clones_settings_but_never_old_examples_or_classes(study, monkeypatch):
    first = train(study)
    parent = first["models"][0]["model_id"]
    model_path = reference_models.model_path(parent)
    digest = hashlib.sha256(model_path.read_bytes()).digest()
    for path in study[2]:
        frame = pd.read_csv(path)
        frame["phenotype_ref1"] = frame.phenotype_ref1.replace({"T": "New T", "B": "New B"})
        frame['M2'] = frame.M0 * 2
        frame.to_csv(path, index=False)
        dataio.ensure_fast_copy(path, force=True)
    monkeypatch.setattr(reference_features, 'options', lambda channel: {
        'selected': ['M0', 'M1', 'M2'], 'allow_missing': True,
        'features': [{'name': name, 'group': 'Markers'} for name in ('M0', 'M1', 'M2')]})
    from spacemap_gui.studio import reference_setup
    reference_setup.configure(study[1], {'action':'training', 'include_saved_examples':False})
    result = train(study, model_id=parent)
    assert len(result["models"]) == 1
    child_id = result["models"][0]["model_id"]
    assert child_id != parent
    _, meta, child = reference_models._source(child_id)
    assert set(child["model"].classes_) == {"New T", "New B"}
    assert meta["base_model_id"] == parent and meta["examples"] == 30
    assert meta['markers'] == ['M0', 'M1', 'M2'] and child['model'].n_features_in_ == 3
    assert hashlib.sha256(model_path.read_bytes()).digest() == digest


def test_changed_labels_and_ambiguous_table_selection_are_rejected(study):
    settings = collection.prepare(study[0], body(study))
    path = study[2][0]
    frame = pd.read_csv(path)
    frame.loc[0, "phenotype_ref1"] = "Changed"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="changed"):
        collection.check_sources(settings["sources"])
    with pytest.raises(ValueError, match="listed"):
        collection.prepare(study[0], body(study, tables=[str(study[1] / "unrelated.csv")]))


def test_missing_training_markers_require_an_explicit_feature_choice(study):
    path = study[2][1]
    pd.read_csv(path).drop(columns="M1").to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing training features: M1"):
        collection.prepare(study[0], body(study))


def test_shared_collection_keeps_examples_but_removes_private_table_paths(study):
    import joblib
    from spacemap_gui.studio import model_library
    result = train(study)
    identifier = result['models'][0]['model_id']
    # Manual training already publishes to Models. Test an explicit second export.
    model_library.set_location(str(study[1] / 'other-library'))
    # Normal jobs store their pooled training table inside outputs/studio/jobs.
    # This fixture uses a short temporary path, so list its known project tables explicitly.
    from unittest.mock import patch
    with patch.object(model_library, '_project_tables', return_value=set(study[2])):
        shared = model_library.share(study[1], 'phenotyping', identifier)
    folder = model_library.location() / 'phenotyping' / shared['entry']
    artifact = joblib.load(folder / (identifier + '.joblib'))
    assert 'training_origin' not in artifact
    assert 'training_sample' in artifact
    assert 'training_sources' not in shared['record'] and 'table_path' not in shared['record']
    assert str(study[1]) not in (folder / 'model.json').read_text()
    assert reference_models._source(identifier)[2]['training_origin']


def test_prediction_failure_keeps_completed_image_receipt(study, monkeypatch):
    import json
    from spacemap_gui.studio import mr_ml
    result = train(study)
    settings = collection.prepare(study[0], body(study, action='predict', model_id=result['models'][0]['model_id']))
    folder = study[1] / 'partial-predictions'
    folder.mkdir()
    actual, calls, released = mr_ml.predict_saved, [], []
    def predict(job, path, report):
        calls.append(job['table'])
        if len(calls) == 2:
            raise ValueError('Second image failed')
        return actual(job, path, report)
    monkeypatch.setattr(mr_ml, 'predict_saved', predict)
    def report(*args):
        pass
    report.finish_output = lambda: released.append(True)
    with pytest.raises(ValueError, match='Second image failed'):
        collection.run(settings, folder, report)
    assert len(json.loads((folder / 'completed_tables.json').read_text())) == 1
    assert released == [True]
    assert 'phenotype_mr_ml' in dataio.columns(study[2][0])
    assert 'phenotype_mr_ml' not in dataio.columns(study[2][1])


def test_mrml_selected_images_use_saved_output_and_class_settings(study):
    from spacemap_gui.studio import mr_ml
    from spacemap_gui.studio.settings import revision
    fitted = train(study)
    changes = {'workflow': 'saved_model', 'model_id': fitted['models'][0]['model_id'],
               'enable_filter': False, 'preserve_reference_labels': False,
               'limit_phenotypes': True, 'prediction_phenotypes': ['T'],
               'output_phenotype_col': 'phenotype_selected', 'confidence_col': 'selected_score',
               'refined_output_phenotype_col': 'phenotype_selected_filtered'}
    mr_ml.save(study[1], changes, revision(mr_ml.values(study[1])))
    settings = collection.prepare(study[0], body(study, action='predict', use_mrml_settings=True, model_id=changes['model_id']))
    assert all(s['output_col']=='phenotype_selected' and s['config']['prediction_phenotypes']==['T'] for s in settings['prediction_settings'])
    folder=study[1]/'selected-predictions';folder.mkdir()
    found=collection.run(settings,folder,lambda *a:None)
    assert found['images']==2
    first,second=[dataio.load(path) for path in study[2]]
    assert first.phenotype_selected.eq('T').all() and second.phenotype_selected.isna().all()
