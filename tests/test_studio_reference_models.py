from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap.viewer import reference
from spacemap_gui.studio import reference_models, tasks


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "2")
    rng = np.random.default_rng(4)
    count = 120
    frame = pd.DataFrame(
        {
            "cell_id": range(count),
            "batch_id": ["roi1"] * 60 + ["roi2"] * 60,
            "label": list(range(1, 61)) * 2,
            "M0": np.r_[rng.normal(2, 0.1, 60), rng.normal(12, 0.1, 60)],
            "M1": rng.normal(3, 0.1, count),
            "phenotype_ref1": ["A"] * 60 + ["B"] * 60,
            "phenotype_threshold": ["A"] * 60 + ["B"] * 60,
            "phenotype_leiden": ["A"] * 58 + ["B"] * 62,
        }
    )
    frame.loc[[2, 64], "phenotype_ref1"] = None
    table = tmp_path / "cells.csv"
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    settings = {
        "table": str(table),
        "roi_column": "batch_id",
        "label_column": "label",
        "markers": ["M0", "M1"],
        "training_column": "phenotype_ref1",
        "checked_only": False,
    }
    settings["revision"] = tasks.source_revision(settings)
    return table, settings, tmp_path


def test_train_compare_reload_and_fast_suggestions_without_retraining(training, monkeypatch):
    table, settings, folder = training
    original = table.read_bytes()
    result = reference_models.train(settings, folder, lambda *args: None)
    assert len(result["models"]) == 2 and result["examples"] == 118
    assert all(model["held_out_balanced_accuracy"] > 0.9 for model in result["models"])
    assert len(reference_models.available(table)) == 2
    model_id = result["models"][0]["model_id"]
    query = {
        "table": str(table),
        "roi": "roi1",
        "roi_column": "batch_id",
        "label_column": "label",
        "destination": "phenotype_ref1",
    }
    monkeypatch.setattr(reference_models, "train", lambda *args: pytest.fail("Must not train on a cell click"))
    first = reference_models.request(query, model_id, label=3)
    assert first["choices"][0]["phenotype"] == "A" and first["choices"][0]["probability"] > 0.9
    # A new manual label does not invalidate a deliberately saved model.
    reference.save(table, "roi1", "phenotype_ref1", {3: "A"}, "new-manual-label")
    assert reference_models.request(query, model_id, label=3) == first
    assert table.read_bytes() == original
    with pytest.raises(ValueError):
        reference_models.request(query, "../../unsafe", label=3)
    with pytest.raises(ValueError):
        reference_models.request({**query, "table": str(folder / "other.csv")}, model_id, label=3)
    for _ in range(100):
        scan = reference_models.request({**query, "roi": "roi2"}, model_id, scan=True)
        if scan["status"] != "preparing":
            break
        time.sleep(0.02)
    assert scan["status"] == "ready" and scan["label"] == 5, scan


def test_checked_only_training_leaves_out_flagged_cells(training):
    table, settings, folder = training
    for roi in ("roi1", "roi2"):
        for label in range(10, 20):
            reference.set_flag(table, roi, "phenotype_ref1", label, False)
    reference.set_flag(table, "roi1", "phenotype_ref1", 11, True)
    settings.update(checked_only=True, revision=tasks.source_revision(settings))
    result = reference_models.train(settings, folder, lambda *args: None)
    assert result["examples"] == 19
    assert all(model["checked_only"] and model["flagged_excluded"] == 1 for model in result["models"])


def test_imported_reference_without_original_table_path_is_listed_and_predicts(training):
    import json
    table, settings, folder = training
    trained = reference_models.train(settings, folder, lambda *args: None)
    model_id = trained["models"][0]["model_id"]
    path = reference_models.model_path(model_id).with_suffix(".json")
    meta = json.loads(path.read_text())
    meta.pop("table_path")
    meta["imported"] = True
    path.write_text(json.dumps(meta))
    other = folder / "other.csv"
    pd.read_csv(table).to_csv(other, index=False)
    assert model_id in [m["model_id"] for m in reference_models.available(other)]
    found = reference_models.request({"table":str(other),"roi":"roi1","roi_column":"batch_id",
        "label_column":"label","destination":"phenotype_ref1","match_scale":False}, model_id, label=3)
    assert found["status"] == "ready" and found["choices"][0]["phenotype"] == "A"


def test_mr_ml_uses_classic_code_and_saves_new_columns(training):
    table, settings, folder = training
    options = {
        **settings,
        "output_col": "phenotype_mr_job",
        "reference_column": "phenotype_ref1",
        "methods": ["phenotype_threshold", "phenotype_leiden"],
    }
    original = table.read_bytes()
    messages = []
    result = tasks.mr_ml(options, folder, lambda message, *args: messages.append(message))
    assert table.read_bytes() == original
    assert result["cells"] == 120 and set(result["agreement"]) == set(options["methods"])
    values = dataio.load(table, columns=result["columns"])
    assert values["phenotype_mr_job"].notna().all()
    assert values["phenotype_mr_job_confidence"].between(0, 1).all()
    assert all(v["agreement"] > 0.9 for v in result["agreement"].values())


def wait_prediction(query, model_id, label):
    for _ in range(300):
        found = reference_models.request(query,model_id,label=label)
        if found['status']!='preparing':
            return found
        time.sleep(.05)
    pytest.fail('Reduced-feature model did not finish')


def test_missing_values_in_another_roi_reuse_available_features(training):
    table, settings, folder = training
    frame = pd.read_csv(table)
    extra = frame.iloc[[2,64]].copy()
    extra['batch_id']='roi3';extra['label']=[1,2];extra['cell_id']=[120,121];extra['M1']=np.nan
    pd.concat([frame,extra],ignore_index=True).to_csv(table,index=False)
    dataio.ensure_fast_copy(table,force=True)
    settings['revision']=tasks.source_revision(settings)
    result=reference_models.train(settings,folder,lambda *args:None)
    model_id=result['models'][0]['model_id']
    query={'table':str(table),'roi':'roi3','roi_column':'batch_id','label_column':'label','destination':'phenotype_ref1'}
    found=wait_prediction(query,model_id,1)
    assert found['choices'][0]['phenotype']=='A'
    assert found['provenance']['missing_markers']==['M1']
    assert found['provenance']['features_used']==1 and found['provenance']['features_total']==2
    found=wait_prediction(query,model_id,2)
    assert found['choices'][0]['phenotype']=='B'
    # Known complete measurements also work when moving back to the first ROI.
    assert wait_prediction({**query,'roi':'roi1'},model_id,3)['choices'][0]['phenotype']=='A'
    assert reference_models.request(query,model_id,label=1)['status']=='ready'
    reduced,_=reference_models.adapt(model_id,['M0'])
    assert len(reduced['rows'])==82 and reduced['oob'].shape==(82,2)
    for _ in range(200):
        scan=reference_models.request(query,model_id,scan=True)
        if scan['status']!='preparing':
            break
        time.sleep(.05)
    assert scan['status']=='ready' and scan['remaining']==2 and scan['omitted']==0


def test_partial_panel_in_another_dataset_does_not_require_import_or_reference_labels(training):
    table,settings,folder=training
    model_id=reference_models.train(settings,folder,lambda *args:None)['models'][0]['model_id']
    other=folder/'new.parquet'
    pd.DataFrame({'batch_id':['new'],'label':[1],'M0':[12.]}).to_parquet(other,index=False)
    assert model_id in [m['model_id'] for m in reference_models.available(other)]
    found=wait_prediction({'table':str(other),'roi':'new','roi_column':'batch_id','label_column':'label',
        'destination':'new labels','match_scale':False},model_id,1)
    assert found['choices'][0]['phenotype']=='B' and found['provenance']['missing_markers']==['M1']
    import json
    meta_path=reference_models.model_path(model_id).with_suffix('.json')
    meta=json.loads(meta_path.read_text());meta['allow_missing']=False;meta_path.write_text(json.dumps(meta))
    assert model_id not in [m['model_id'] for m in reference_models.available(other)]


def test_validation_cells_never_enter_saved_or_reduced_model(training, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    table, settings, folder = training
    fit = RandomForestClassifier.fit
    seen = []
    def record(self, values, targets, *args, **kwargs):
        seen.append(np.asarray(values).copy())
        return fit(self, values, targets, *args, **kwargs)
    monkeypatch.setattr(RandomForestClassifier, 'fit', record)
    result = reference_models.train({**settings, 'training_percent': 70}, folder, lambda *args: None)
    meta = result['models'][0]
    _, _, artifact = reference_models._source(meta['model_id'])
    training_rows = set(artifact['rows'])
    validation = artifact['validation_sample']
    assert len(training_rows) == meta['training_cells'] == 82
    assert len(validation['rows']) == meta['held_out_cells'] == 36
    assert not training_rows.intersection(validation['rows'])
    assert set(artifact['training_sample']['rows']) == training_rows
    assert len(seen) == 1 and len(seen[0]) == 82  # no final all-cell refit
    assert set(validation['targets']) == {'A', 'B'}
    reduced, reduced_meta = reference_models.adapt(meta['model_id'], ['M0'])
    assert not set(reduced['rows']).intersection(validation['rows'])
    assert all(len(values) == 82 for values in seen)
    assert reduced_meta['validation_held_out']
    assert meta['held_out_macro_f1'] > .9
    again = reference_models.train({**settings, 'training_percent': 70}, folder, lambda *args: None)
    assert np.array_equal(reference_models._source(again['models'][0]['model_id'])[2]['validation_sample']['rows'], validation['rows'])
    eighty = reference_models.train({**settings, 'training_percent': 80}, folder, lambda *args: None)
    assert eighty['models'][0]['training_cells'] == 94


@pytest.mark.parametrize('value', [0, 100, 49, 91, 70.5, True, '70', None])
def test_invalid_training_split_is_rejected(training, value):
    _, settings, folder = training
    with pytest.raises(ValueError, match='training percentage'):
        reference_models.train({**settings, 'training_percent': value}, folder, lambda *args: None)
