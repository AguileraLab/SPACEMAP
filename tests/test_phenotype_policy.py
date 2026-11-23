"""Class switches exclude examples and preserve honest, unrenormalized scores."""
import json
import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import reference_setup, reference_models, reference_examples, reference_collection, mr_ml, tasks
from test_studio_reference_models import training  # noqa: F401
from test_studio_mr_ml import model_job


def test_policy_roundtrip_validation_and_job_snapshot(training):
    table, settings, root = training
    saved = reference_setup.configure(root, {'action': 'phenotypes', 'excluded_training_phenotypes': [' Others '],
                                           'excluded_prediction_phenotypes': ['Others', 'B']}, table)
    assert saved['excluded_training_phenotypes'] == ['Others']
    assert reference_setup.training(root)['excluded_prediction_phenotypes'] == ['B', 'Others']
    with pytest.raises(ValueError, match='names'):
        reference_setup.configure(root, {'action': 'phenotypes', 'excluded_training_phenotypes': 'Others'}, table)
    model = reference_models.train(settings, root/'fit', lambda *a: None)['models'][0]
    job = model_job(root, table, model['model_id'])
    assert job['excluded_prediction_phenotypes'] == ['B', 'Others']
    reference_setup.configure(root, {'action': 'phenotypes', 'excluded_training_phenotypes': [],
                                    'excluded_prediction_phenotypes': []}, table)
    assert job['excluded_prediction_phenotypes'] == ['B', 'Others']  # queued jobs keep their recipe


def test_training_and_replayed_examples_exclude_switched_off_class(training):
    table, settings, root = training
    frame = pd.read_csv(table)
    frame.loc[10:24, 'phenotype_ref1'] = 'Others'
    frame.loc[10:24, 'M1'] = 30.
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    settings['revision'] = tasks.source_revision(settings)
    original = table.read_bytes()
    parent = reference_models.train(settings, root/'parent', lambda *a: None)['models'][0]
    path = reference_models.model_path(parent['model_id']); checksum = path.read_bytes()
    assert 'Others' in parent['classes']
    # Another table forces replay of the old examples, including the blocked class.
    target = root/'new.csv'; frame.to_csv(target, index=False); dataio.ensure_fast_copy(target, force=True)
    filtered = {**settings, 'table': str(target), 'base_model_id': parent['model_id'],
                'include_saved_examples': True, 'excluded_training_phenotypes': ['Others']}
    filtered['revision'] = tasks.source_revision(filtered)
    result = reference_models.train(filtered, root/'child', lambda *a: None)['models'][0]
    _, meta, artifact = reference_models._source(result['model_id'])
    assert meta['classes'] == ['A', 'B'] and meta['excluded_training_phenotypes'] == ['Others']
    for archive in ('example_archive', 'validation_archive'):
        assert all('Others' not in set(g['targets']) for g in artifact[archive])
    source = {**settings, 'excluded_training_phenotypes': ['Others']}
    _, labels = reference_collection._examples(source, lambda *a: None)
    assert set(labels) == {'A', 'B'}
    assert path.read_bytes() == checksum and table.read_bytes() == original
    from spacemap_gui.studio import model_review
    refit = model_review.run({'action':'refit', 'model_ids':[parent['model_id']], 'project_root':str(root),
        'models_folder':str(root/'refitted-models'), 'model_timezone':'UTC', 'name':'Without Others',
        'phenotype_groups':{'Others':'Background'}, 'keep_detailed_predictions':True, 'excluded_training_phenotypes':['Others']},
        root/'refit', lambda *a: None)['models'][0]
    assert refit['classes'] == ['A', 'B']
    assert refit['phenotype_groups'] == {}
    assert path.read_bytes() == checksum


def test_saved_prediction_block_does_not_inflate_confidence_or_erase_reference(training):
    table, settings, root = training
    model = reference_models.train(settings, root/'fit', lambda *a: None)['models'][0]
    _, _, artifact = reference_models._source(model['model_id'])
    original = table.read_bytes(); model_bytes = reference_models.model_path(model['model_id']).read_bytes()
    frame = dataio.load(table)
    expected = artifact['model'].predict_proba(frame[['M0', 'M1']].to_numpy(np.float32))[:, 0]
    reference_setup.configure(root, {'action': 'phenotypes', 'excluded_training_phenotypes': [],
                                    'excluded_prediction_phenotypes': ['B', 'Others']}, table)
    job = model_job(root, table, model['model_id'], refined_confidence_threshold=.75,
                    preserve_reference_labels=True)
    result = tasks.mr_ml(job, root/'predict', lambda *a: None)
    found = dataio.load(table)
    assert set(found[job['output_col']].dropna()) == {'A'}
    assert (found.loc[found.batch_id == 'roi2', job['refined_column']] == 'Others').all()
    scored = found[job['confidence_column']].notna()
    np.testing.assert_allclose(found.loc[scored, job['confidence_column']], expected[scored])
    assert result['excluded_prediction_phenotypes'] == ['B', 'Others']
    assert found['phenotype_ref1'].fillna('').equals(frame['phenotype_ref1'].fillna(''))
    assert original == table.read_bytes() and model_bytes == reference_models.model_path(model['model_id']).read_bytes()
    # Cell-card suggestions obey the same switch, with their original confidence.
    query = {**settings, 'roi':'roi2', 'destination':'phenotype_ref1', 'excluded_prediction_phenotypes':['B']}
    reply = reference_models.request(query, model['model_id'], label=5)
    assert {p['phenotype'] for p in reply['choices']} == {'A'}
    assert reply['choices'][0]['probability'] < .75
    with pytest.raises(ValueError, match='All model phenotypes'):
        reference_models.request({**query, 'excluded_prediction_phenotypes':['A','B']}, model['model_id'], prepare=True)


def test_grouped_labels_cannot_reintroduce_blocked_name():
    names, columns = mr_ml.prediction_groups(['A','B','Others'], {'B':'Others'}, blocked=['Others'])
    assert names == ['A'] and columns == [[0]]
    with pytest.raises(ValueError, match='All model phenotypes'):
        mr_ml.prediction_groups(['A','B'], {}, blocked=['A','B'])
    # Existing output-subset behavior remains distinct from an explicit block.
    assert mr_ml.prediction_groups(['A','B'], {}, selected=['A'])[0] == ['A', None]


def test_classic_prediction_keeps_denominator():
    from sklearn.preprocessing import LabelEncoder
    from spacemap.ml_mr_utils import predict_weighted_ensemble
    class Model:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, values):
            return np.array([[.1, .2, .7], [.8, .1, .1]])
    labels, score = predict_weighted_ensemble(pd.DataFrame({'x':[1,2]}), Model(), pd.DataFrame(), pd.DataFrame(),
        LabelEncoder().fit(['A','B','Others']), excluded_prediction_phenotypes=['Others'])
    assert labels.tolist() == ['B','A']
    np.testing.assert_allclose(score, [.2,.8])


def test_explicit_training_switches_replace_legacy_classic_exclusions(training):
    table, settings, root = training
    chosen = {**mr_ml.DEFAULTS, 'markers':['M0','M1']}
    assert mr_ml.job_settings(root, table, {'cell_id':'cell_id'}, chosen)['excluded_phenotypes'] == ['Others']
    reference_setup.configure(root, {'action':'phenotypes', 'excluded_training_phenotypes':[],
                                    'excluded_prediction_phenotypes':[]}, table)
    assert mr_ml.job_settings(root, table, {'cell_id':'cell_id'}, chosen)['excluded_phenotypes'] == []
