"""Immutable model merging, feature edits, grouped outputs and honest validation."""
import hashlib
import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import model_review as review, reference_models as models, reference_setup as setup, tasks, mr_ml
from test_studio_reference_models import training  # noqa: F401


def prepare(training, **changes):
    table, settings, root = training
    saved = models.train(settings, root, lambda *a: None)['models']
    setup.configure(root, {'action': 'folder', 'folder': str(models.directory())}, table)
    listed = setup.options(root)['models']
    body = {'keys': [next(m['key'] for m in listed if m['id'] == item['model_id']) for item in saved],
            'action': 'merge', 'timezone': 'America/Chicago', **changes}
    return saved, review.prepare(root, body), body


def test_merge_deduplicates_preserves_classes_and_originals(training):
    saved, settings, _ = prepare(training, name='Lab reference')
    before = {m['model_id']: hashlib.sha256(models.model_path(m['model_id']).read_bytes()).hexdigest() for m in saved}
    inspected = review.run({**settings, 'action': 'inspect'}, None, lambda *a: None)
    assert inspected['can_merge'] and inspected['training_cells'] == 82
    assert inspected['duplicate_training_cells'] == 82 and inspected['validation_cells'] == 36
    assert inspected['shared_features'] == ['M0', 'M1']
    profile = inspected['models'][0]['profile']
    assert profile['rows'] == ['A', 'B'] and profile['columns'] == ['M0', 'M1']
    assert profile['values'][0][0] < 0 < profile['values'][1][0]
    result = review.run(settings, training[2], lambda *a: None)
    child = result['models'][0]
    assert child['name'] == 'Lab reference' and child['classes'] == ['A', 'B']
    assert child['total_fit_cells'] == 82 and child['held_out_cells'] == 36 and child['held_out_macro_f1'] > .9
    assert any(zone in child['shared_path'] for zone in ('CDT', 'CST'))
    assert 'Lab_reference' in child['shared_path']
    artifact = models._source(child['model_id'])[2]
    fit_ids = {(s, p) for g in artifact['example_archive'] for s, p in zip(g['sources'], g['positions'])}
    test_ids = {(s, p) for g in artifact['validation_archive'] for s, p in zip(g['sources'], g['positions'])}
    assert not fit_ids & test_ids
    assert all(hashlib.sha256(models.model_path(i).read_bytes()).hexdigest() == digest for i, digest in before.items())
    assert len(setup.options(training[2])['models']) == 3


def test_merge_different_panels_retains_unique_archived_features(training):
    table, settings, root = training
    first = models.train(settings, root, lambda *a: None)['models'][0]
    other = root / 'other.csv'
    frame = pd.read_csv(table).rename(columns={'M1': 'M2'})
    frame['phenotype_ref1'] = frame.phenotype_ref1.replace({'B': 'C'})
    frame.to_csv(other, index=False)
    options = {**settings, 'table': str(other), 'markers': ['M0', 'M2']}
    options['revision'] = tasks.source_revision(options)
    second = models.train(options, root, lambda *a: None)['models'][0]
    combined = {'action': 'merge', 'model_ids': [first['model_id'], second['model_id']], 'models_folder': str(root / 'library'),
                'project_root': str(root), 'name': 'Combined', 'model_timezone': 'America/Chicago',
                'phenotype_groups': {}, 'keep_detailed_predictions': True}
    compared = review.run({**combined, 'action': 'inspect'}, None, lambda *a: None)
    assert compared['models'][0]['unique_features'] == ['M1']
    assert compared['models'][1]['unique_features'] == ['M2']
    child = review.run(combined, root, lambda *a: None)['models'][0]
    assert child['markers'] == ['M0'] and child['classes'] == ['A', 'B', 'C']
    assert child['archived_features'] == ['M0', 'M1', 'M2']
    assert child['total_fit_cells'] == 164 and child['held_out_cells'] == 72
    with pytest.raises(ValueError, match='No shared validation'):
        review.run({**combined, 'action': 'validate'}, root, lambda *a: None)


def test_validation_compares_same_heldout_cells(training):
    _, settings, _ = prepare(training)
    result = review.run({**settings, 'action': 'validate'}, None, lambda *a: None)
    assert [m['validation']['cells'] for m in result['models']] == [36, 36]
    assert all(m['validation']['f1'] > .9 for m in result['models'])
    assert sum(map(sum, result['models'][0]['validation']['confusion'])) == 36


def test_conflicting_labels_block_merge_without_writing(training):
    saved, settings, _ = prepare(training)
    identifier = saved[1]['model_id']
    _, meta, artifact = models._source(identifier)
    artifact['example_archive'][0]['targets'] = np.repeat('Conflict', len(artifact['example_archive'][0]['targets']))
    models._save_model(identifier, artifact, meta)
    with pytest.raises(ValueError, match='conflicting labels'):
        review.run(settings, training[2], lambda *a: None)
    assert len(list(models.directory().glob('*.joblib'))) == 2


def test_missing_examples_can_inspect_and_copy_but_cannot_merge(training):
    saved, settings, _ = prepare(training)
    identifier = saved[0]['model_id']
    _, meta, artifact = models._source(identifier)
    artifact.pop('example_archive')
    artifact.pop('training_sample')
    models._save_model(identifier, artifact, meta)
    found = review.run({**settings, 'action': 'inspect'}, None, lambda *a: None)
    assert not found['can_merge'] and found['models'][0]['profile'] is None
    with pytest.raises(ValueError, match='no reusable examples'):
        review.run(settings, training[2], lambda *a: None)
    copy = review.run({**settings, 'action': 'copy', 'model_ids': [identifier]}, training[2], lambda *a: None)
    assert copy['models'][0]['classes'] == ['A', 'B']


def test_feature_refit_and_grouped_prediction_keep_detail(training):
    saved, settings, _ = prepare(training)
    changed = {**settings, 'model_ids': [saved[0]['model_id']], 'action': 'refit', 'features': ['M0'],
               'phenotype_groups': {'A': 'Epithelia', 'B': 'Epithelia'}}
    child = review.run(changed, training[2], lambda *a: None)['models'][0]
    assert child['markers'] == ['M0'] and child['classes'] == ['A', 'B']
    assert child['archived_features'] == ['M0', 'M1']
    table, _, root = training
    chosen = {**mr_ml.DEFAULTS, 'workflow': 'saved_model', 'model_id': child['model_id'], 'enable_filter': False,
              'cpu_workers': 1, 'output_phenotype_col': 'phenotype_broad'}
    config = mr_ml.job_settings(root, table, {'cell_id': 'cell_id', 'roi': 'batch_id'}, chosen)
    result = mr_ml.predict_saved(config, root / 'predicted', lambda *a: None)
    assert result['detail_column'] == 'phenotype_broad_detail'
    frame = dataio.load(table)
    assert set(frame.phenotype_broad) == {'Epithelia'}
    assert set(frame.phenotype_broad_detail) == {'A', 'B'}
    query = {'table': str(table), 'roi': 'roi1', 'roi_column': 'batch_id', 'label_column': 'label', 'match_scale': False}
    suggestions = models.request(query, child['model_id'], label=3)
    assert suggestions['choices'][0]['phenotype'] == 'Epithelia'
    assert {p['phenotype'] for p in suggestions['detailed_probabilities']} == {'A', 'B'}
    with pytest.raises(ValueError, match='Use features measured'):
        review.run({**changed, 'features': ['Missing']}, root, lambda *a: None)


def test_empty_model_folder_cannot_use_previous_review_selection(training):
    _, _, body = prepare(training)
    empty = training[2] / 'empty'
    empty.mkdir()
    setup.configure(training[2], {'action': 'folder', 'folder': str(empty)})
    with pytest.raises(ValueError, match='current Models folder'):
        review.prepare(training[2], body)


def test_cells_used_by_another_model_are_never_reported_as_validation(training):
    saved, settings, _ = prepare(training)
    identifier = saved[1]['model_id']
    _, meta, artifact = models._source(identifier)
    held = artifact['validation_archive'][0]
    from spacemap_gui.studio.reference_examples import take
    extra = take(held, np.arange(2))
    artifact['example_archive'].append(extra)
    models._save_model(identifier, artifact, meta)
    result = review.run({**settings, 'action': 'validate'}, None, lambda *a: None)
    assert result['validation_cells'] == 34 and result['reused_validation_cells'] == 2
    assert all(m['validation']['cells'] == 34 for m in result['models'])
