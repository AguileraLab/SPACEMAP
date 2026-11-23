"""Cross-project replay, immutable exports and conservative class selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import reference_models as models, reference_setup as setup, reference_examples, tasks, mr_ml
from test_studio_reference_models import training  # noqa: F401
from test_studio_mr_ml import model_job


def child_project(training):
    table, settings, root = training
    parent = models.train(settings, root / 'first', lambda *a: None)['models'][0]
    frame = pd.read_csv(table)
    frame['M2'] = frame.M0 * 2
    frame['phenotype_ref1'] = frame.phenotype_ref1.replace({'B': 'New B'})
    path = root / 'other.csv'
    frame.to_csv(path, index=False)
    dataio.ensure_fast_copy(path, force=True)
    child = {**settings, 'table': str(path), 'markers': ['M0', 'M1', 'M2'], 'base_model_id': parent['model_id'],
             'include_saved_examples': True, 'model_name': 'Cross project version'}
    child['revision'] = tasks.source_revision(child)
    return parent, child, root


def test_replay_retains_panels_and_excludes_validation_and_repeated_project(training):
    parent, settings, root = child_project(training)
    before = hashlib.sha256(models.model_path(parent['model_id']).read_bytes()).hexdigest()
    child = models.train(settings, root / 'second', lambda *a: None)['models'][0]
    _, _, artifact = models._source(child['model_id'])
    assert child['classes'] == ['A', 'B', 'New B']
    assert child['markers'] == ['M0', 'M1'] and child['unused_features'] == ['M2']
    assert child['archived_features'] == ['M0', 'M1', 'M2']
    assert child['saved_example_cells'] == 82 and child['total_fit_cells'] == 164
    assert artifact['oob'].shape == (82, 3)
    assert len(artifact['example_archive']) == 2
    assert len(artifact['validation_archive']) == 2
    assert {s['phenotype'] for s in mr_ml.panel_support(child['model_id'], settings['table'])['phenotypes'] if s['cells']} == {'A', 'B', 'New B'}
    new = artifact['example_archive'][1]
    expected = dataio.load(settings['table']).iloc[artifact['rows']]
    assert np.array_equal(new['values'], expected[['M0', 'M1', 'M2']].to_numpy(np.float32))
    assert not set(artifact['rows']) & set(artifact['validation_sample']['rows'])
    # A later split replaces this project's prior examples, rather than leaking
    # old training rows into its current validation partition or counting twice.
    again = models.train({**settings, 'base_model_id': child['model_id'], 'training_percent': 50}, root / 'third', lambda *a: None)['models'][0]
    _, _, updated = models._source(again['model_id'])
    assert again['saved_example_cells'] == 82 and again['total_fit_cells'] == 141
    key = reference_examples.source_key(settings['table'])
    assert sum((g['sources'] == key).sum() for g in updated['example_archive']) == 59
    new = next(g for g in updated['example_archive'] if (g['sources'] == key).all())
    assert np.array_equal(new['values'], dataio.load(settings['table']).iloc[updated['rows']][['M0', 'M1', 'M2']].to_numpy(np.float32))
    held = next(g for g in updated['validation_archive'] if (g['sources'] == key).all())
    assert len(held['targets']) == again['held_out_cells']
    assert not set(map(tuple, new['values'])) & set(map(tuple, held['values']))
    # Feature adaptation must retain the same training-only OOB row mapping.
    reduced, _ = models.adapt(child['model_id'], ['M0'])
    assert set(reduced['rows']) == set(artifact['rows']) and reduced['oob'].shape == (82, 3)
    assert hashlib.sha256(models.model_path(parent['model_id']).read_bytes()).hexdigest() == before
    current = models.train({**settings, 'include_saved_examples': False}, root / 'current', lambda *a: None)['models'][0]
    assert current['markers'] == ['M0', 'M1', 'M2'] and current['classes'] == ['A', 'New B']


def test_training_exports_named_child_and_loading_keeps_archive(training, monkeypatch):
    parent, settings, root = child_project(training)
    folder = root / 'shared'
    folder.mkdir()
    setup.configure(root, {'action': 'folder', 'folder': str(folder)}, settings['table'])
    options = setup.options(root, settings['table'])
    entry = next(x for x in options['models'] if x['id'] == parent['model_id']) if any(x['id'] == parent['model_id'] for x in options['models']) else None
    # An arbitrary account model is absent until it is put in the selected directory.
    assert entry is None
    setup.publish(root, parent['model_id'])
    options = setup.options(root, settings['table'])
    entry = next(x for x in options['models'] if x['id'] == parent['model_id'])
    setup.configure(root, {'action': 'load', 'key': entry['key']}, settings['table'])
    child = models.train({**settings, 'project_root': str(root), 'models_folder': str(folder)}, root / 'fit', lambda *a: None)['models'][0]
    assert child['shared_path'] and Path(child['shared_path'], 'model.json').is_file()
    record = json.loads(Path(child['shared_path'], 'model.json').read_text())
    assert record['name'] == 'Cross project version'
    assert str(root) not in json.dumps(record)
    assert setup.options(root, settings['table'])['base_model_id'] == child['model_id']
    originals = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    destination = root / 'another-project'; destination.mkdir()
    local = destination / 'local'; local.mkdir()
    monkeypatch.setattr(models, 'directory', lambda: local)
    options = setup.configure(destination, {'action': 'folder', 'folder': str(folder)}, settings['table'])
    selected = next(m for m in options['models'] if m['id'] == child['model_id'])
    loaded = setup.configure(destination, {'action': 'load', 'key': selected['key']}, settings['table'])
    _, _, imported = models._source(loaded['base_model_id'])
    assert len(imported['example_archive']) == 2 and 'training_origin' not in imported
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert len(loaded['models']) == 2  # original + loaded child, no duplicate shared copy


def test_selected_phenotypes_never_force_other_cells_into_selected_class(training):
    table, settings, root = training
    model = models.train(settings, root/'fit', lambda *a: None)['models'][0]
    support = mr_ml.panel_support(model['model_id'], table)
    assert support['features'] == ['M0', 'M1']
    assert all(row['suggested'] and row['f1'] > .9 for row in support['phenotypes'])
    job = model_job(root, table, model['model_id'], preserve_reference_labels=False,
                    limit_phenotypes=True, prediction_phenotypes=['A'])
    result = mr_ml.predict_saved(job, root/'prediction', lambda *a: None)
    found = dataio.load(table)
    assert found.loc[found.batch_id=='roi1',job['output_col']].eq('A').all()
    assert found.loc[found.batch_id=='roi2',job['output_col']].isna().all()
    assert result['outside_selected_phenotypes'] == 60
    assert found.loc[found.batch_id=='roi2',job['refined_column']].isna().all()
    with pytest.raises(ValueError, match='Select at least'):
        model_job(root, table, model['model_id'], limit_phenotypes=True, prediction_phenotypes=['Imaginary'])


def test_legacy_archive_excludes_explicit_validation_and_needs_row_identity(training):
    parent, settings, root = child_project(training)
    with pytest.raises(ValueError, match='No shared features'):
        models.train({**settings, 'markers': ['M2'], 'revision': tasks.source_revision({**settings, 'markers': ['M2']})}, root/'bad', lambda *a: None)
    _, meta, artifact = models._source(parent['model_id'])
    legacy = {k: v for k, v in artifact.items() if k != 'example_archive'}
    # Even a legacy artifact with mixed samples must not reuse explicit holdout.
    sample, held = artifact['training_sample'], artifact['validation_sample']
    legacy['training_sample'] = {**sample, **{k: np.concatenate([sample[k], held[k]]) for k in ('values', 'targets', 'rows')}}
    groups = reference_examples.archive({**meta, 'validation_held_out': False}, legacy)
    assert len(groups) == 1
    assert np.array_equal(groups[0]['values'], sample['values'])
    assert not reference_examples.archive({**meta, 'validation_held_out': False}, {**legacy, 'rows': []})


def test_legacy_final_fit_examples_reused_and_original_is_unchanged(training):
    import joblib
    parent, settings, root = child_project(training)
    path = models.model_path(parent['model_id'])
    meta = json.loads(path.with_suffix('.json').read_text())
    artifact = joblib.load(path)
    # Match the shipped older writer: score a split, then fit all saved rows.
    sample, held = artifact['training_sample'], artifact['validation_sample']
    full = {k: np.concatenate([sample[k], held[k]]) for k in ('values', 'targets', 'rows')}
    order = np.argsort(full['rows'])
    full = {k: v[order] for k, v in full.items()}
    artifact['model'].fit(full['values'], full['targets'])
    artifact.update(rows=full['rows'], oob=artifact['model'].oob_decision_function_,
                    training_sample={**full, 'markers': meta['markers']})
    for key in ('validation_sample', 'example_archive', 'validation_archive'):
        artifact.pop(key)
    for key in ('classes', 'validation_held_out', 'held_out_per_class'):
        meta.pop(key)
    meta['needs_examples'] = ['From original project']
    joblib.dump(artifact, path)
    path.with_suffix('.json').write_text(json.dumps(meta))
    before = path.read_bytes()
    setup.publish(root, parent['model_id'])
    choice = next(m for m in setup.options(root, settings['table'])['models'] if m['id'] == parent['model_id'])
    result = setup.configure(root, {'action': 'load', 'key': choice['key']}, settings['table'])
    entry = next(m for m in result['models'] if m['id'] == parent['model_id'])
    assert entry['classes'] == ['A', 'B'] and entry['reusable_examples'] == 118
    assert entry['needs_examples'] == []
    child = models.train({**settings, 'project_root': str(root)}, root/'legacy-child', lambda *a: None)['models'][0]
    assert child['classes'] == ['A', 'B', 'New B'] and child['saved_example_cells'] == 118
    assert setup.training(root)['base_model_id'] == child['model_id']
    assert path.read_bytes() == before
    assert Path(child['shared_path'], 'model.json').is_file()


def test_missing_examples_stop_before_table_scan_and_singletons_are_reported(training, monkeypatch):
    import joblib
    from spacemap.viewer import consensus_model
    parent, settings, root = child_project(training)
    path = models.model_path(parent['model_id'])
    artifact = joblib.load(path)
    artifact.pop('training_sample'); artifact.pop('example_archive')
    joblib.dump(artifact, path)
    with monkeypatch.context() as patch:
        patch.setattr(consensus_model, '_sample', lambda *a, **kw: pytest.fail('No table scan before preflight'))
        with pytest.raises(reference_examples.MissingTrainingExamples, match='Details & options'):
            models.train(settings, root/'bad', lambda *a: None)
    frame = pd.read_csv(settings['table'])
    frame.loc[0, 'phenotype_ref1'] = 'One cell'
    frame.to_csv(settings['table'], index=False)
    dataio.ensure_fast_copy(settings['table'], force=True)
    settings.update(include_saved_examples=False, revision=tasks.source_revision(settings))
    child = models.train(settings, root/'current-only', lambda *a: None)['models'][0]
    assert child['needs_examples'] == ['One cell'] and child['classes'] == ['A', 'New B']


def test_learn_more_than_eleven_types_and_check_the_available_marker_panel(training):
    table, settings, root = training
    rng = np.random.default_rng(12)
    labels = np.repeat([f'Type {n:02}' for n in range(13)], 30)
    frame = pd.DataFrame({'cell_id': np.arange(len(labels)), 'label': np.arange(len(labels)), 'batch_id': 'image',
                          'phenotype_ref1': labels, 'M0': np.repeat(np.arange(13)*10, 30) + rng.normal(0, .01, len(labels)),
                          'M1': np.ones(len(labels))})
    frame.to_csv(table, index=False); dataio.ensure_fast_copy(table, force=True)
    settings['revision'] = tasks.source_revision(settings)
    parent = models.train(settings, root/'many', lambda *a: None)['models'][0]
    more = frame.iloc[:30].copy()
    more['phenotype_ref1'] = 'New type'; more['M0'] = 140 + rng.normal(0, .01, 30)
    child_table = root/'another.csv'
    more.to_csv(child_table, index=False); dataio.ensure_fast_copy(child_table, force=True)
    child_settings = {**settings, 'table': str(child_table), 'base_model_id': parent['model_id'], 'include_saved_examples': True}
    child_settings['revision'] = tasks.source_revision(child_settings)
    child = models.train(child_settings, root/'extended', lambda *a: None)['models'][0]
    assert len(child['classes']) == 14 and 'New type' in child['classes']
    support = mr_ml.panel_support(child['model_id'], child_table)
    assert len(support['phenotypes']) == 14 and all(p['suggested'] for p in support['phenotypes'])
    reduced_table = root/'missing-marker.parquet'
    more.drop(columns='M0').to_parquet(reduced_table, index=False)
    reduced = mr_ml.panel_support(child['model_id'], reduced_table)
    assert reduced['missing'] == ['M0']
    assert len(reduced['phenotypes']) == 14 and not any(p['suggested'] for p in reduced['phenotypes'])
    query = {'table': str(reduced_table), 'roi': 'image', 'roi_column': 'batch_id', 'label_column': 'label',
             'destination': 'phenotype_ref1', 'match_scale': False}
    from test_studio_reference_models import wait_prediction
    found = wait_prediction(query, child['model_id'], 1)
    assert found['provenance']['panel_check']['suggested'] == []
    assert set(found['provenance']['panel_check']['needs_checking']) == set(child['classes'])
    # A smaller target panel changes recommendations, not the saved vocabulary.
    assert models.metadata(child['model_id'])['classes'] == child['classes']


def test_archive_budget_retains_rare_phenotypes(monkeypatch):
    monkeypatch.setattr(reference_examples, 'LIMIT', 50)
    group={'markers':['M'], 'values':np.arange(205,dtype=float)[:,None],
           'targets':np.array(['Common']*200+['Rare']*5), 'sources':np.repeat('old-project',205)}
    settings={'markers':['M'],'table':'/tmp/new-project.csv','include_saved_examples':True}
    value=np.arange(10,dtype=float)[:,None]
    _,_,labels,_,bank,old=reference_examples.combine(settings,value,np.array(['New']*10),np.arange(10),np.arange(10),({}, {'example_archive':[group]}))
    assert len(labels)==50 and old==40 and (labels=='Rare').sum()==5
    assert sum(len(g['targets']) for g in bank)==50


def test_a_single_new_phenotype_can_extend_a_saved_model(training):
    parent, settings, root=child_project(training)
    frame=pd.read_csv(settings['table'])
    frame['phenotype_ref1']='New type'
    frame.to_csv(settings['table'],index=False)
    dataio.ensure_fast_copy(settings['table'],force=True)
    settings['revision']=tasks.source_revision(settings)
    child=models.train(settings,root/'extend',lambda *a:None)['models'][0]
    assert child['classes']==['A','B','New type']
    assert child['saved_example_cells']==82 and child['training_cells']==84
    assert child['held_out_cells']==36


def test_automatic_versions_keep_learning_old_and_new_classes_across_images(training):
    parent, settings, root = child_project(training)
    setup.publish(root, parent['model_id'])
    choice = next(m for m in setup.options(root, settings['table'])['models'] if m['id'] == parent['model_id'])
    setup.configure(root, {'action': 'load', 'key': choice['key']}, settings['table'])
    first = models.train({**settings, **setup.training(root), 'auto_train': True, 'project_root': str(root)},
                         root/'auto-first', lambda *a: None)['models'][0]
    assert first['classes'] == ['A', 'B', 'New B']
    before = models.model_path(first['model_id']).read_bytes()
    frame = pd.read_csv(settings['table'])
    frame['phenotype_ref1'] = 'Another new type'
    table = root/'third-image.csv'
    frame.to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    options = {**settings, **setup.training(root), 'table': str(table),
               'auto_train': True, 'project_root': str(root)}
    options['revision'] = tasks.source_revision(options)
    second = models.train(options, root/'auto-second', lambda *a: None)['models'][0]
    assert second['classes'] == ['A', 'Another new type', 'B', 'New B']
    assert second['base_model_id'] == first['model_id']
    assert setup.training(root)['base_model_id'] == second['model_id']
    assert Path(first['shared_path'], 'model.json').is_file()
    assert Path(second['shared_path'], 'model.json').is_file()
    assert models.model_path(first['model_id']).read_bytes() == before
    _, _, artifact = models._source(second['model_id'])
    assert len(artifact['example_archive']) == 3
    assert not set(artifact['rows']) & set(artifact['validation_sample']['rows'])


def test_returning_to_same_table_with_another_reference_keeps_previous_types(training):
    import joblib
    parent, settings, root = child_project(training)
    child = models.train(settings, root/'second', lambda *a:None)['models'][0]
    # Simulate previously exported versions that did not archive row identities.
    child_path = models.model_path(child['model_id'])
    artifact = joblib.load(child_path)
    for key in ('example_archive', 'validation_archive'):
        for group in artifact[key]:
            group.pop('positions'); group.pop('references')
    joblib.dump(artifact, child_path)
    before = child_path.read_bytes()
    table = training[0]
    frame = pd.read_csv(table)
    frame['phenotype_ref2'] = None
    frame.loc[:19, 'phenotype_ref2'] = 'Different type'
    frame.to_csv(table, index=False); dataio.ensure_fast_copy(table, force=True)
    returned = {**training[1], 'base_model_id':child['model_id'], 'include_saved_examples':True,
                'training_column':'phenotype_ref2'}
    returned['revision'] = tasks.source_revision(returned)
    fitted = models.train(returned, root/'return', lambda *a:None)['models'][0]
    assert fitted['classes'] == ['A', 'B', 'Different type', 'New B']
    _, _, final = models._source(fitted['model_id'])
    key = reference_examples.source_key(table)
    old_rows = set()
    for group in final['example_archive']:
        old_rows.update(group['positions'][(group['sources'] == key) & (group['references'] == 'phenotype_ref1')])
    assert old_rows and not old_rows.intersection(range(20))
    fit_ids = {(source, int(row)) for group in final['example_archive'] for source,row in zip(group['sources'],group['positions'])}
    held_ids = {(source, int(row)) for group in final['validation_archive'] for source,row in zip(group['sources'],group['positions'])}
    assert not fit_ids.intersection(held_ids)
    assert child_path.read_bytes() == before


def test_cumulative_training_cannot_silently_drop_an_existing_type(training):
    table, settings, root = training
    parent = models.train(settings, root/'first', lambda *a:None)['models'][0]
    frame = pd.read_csv(table); frame.loc[frame.phenotype_ref1=='B', 'phenotype_ref1'] = 'Replacement'
    frame.to_csv(table,index=False); dataio.ensure_fast_copy(table,force=True)
    changed = {**settings,'base_model_id':parent['model_id'],'include_saved_examples':True}
    changed['revision'] = tasks.source_revision(changed)
    with pytest.raises(ValueError, match='would lose saved phenotypes: B'):
        models.train(changed, root/'bad', lambda *a:None)
