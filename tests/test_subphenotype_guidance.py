"""Filtered subsets keep sources intact; model votes assist naming without forcing labels."""
import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import subphenotypes, subphenotype_guidance, reference_models, tasks, mr_ml
from test_studio_reference_models import training  # noqa: F401
from test_studio_subphenotypes import parents, project, job, MARKERS  # noqa: F401


def test_region_filters_intersect_numeric_text_and_missing_cells(tmp_path):
    frame = pd.DataFrame({'phenotype_parent': ['Epithelia'] * 5 + [None],
                          'tumor': [1., 1., 1., 0., None, 1.],
                          'review': ['keep', 'skip', None, 'keep', 'keep', 'keep']})
    path = tmp_path / 'cells.parquet'
    frame.to_parquet(path)
    chosen = {'parent_col': 'phenotype_parent', 'parent_values': ['Epithelia'], 'filters': [
        {'column': 'tumor', 'op': 'in', 'values': ['1']},
        {'column': 'review', 'op': 'not_in', 'values': ['skip']}]}
    assert subphenotypes.selection(path, chosen) == {'cells': 1, 'total': 6, 'parents': {'Epithelia': 1}}
    chosen['filters'][0]['values'] = ['1.0']
    assert subphenotypes.selection(path, chosen)['cells'] == 1
    chosen['filters'][0]['column'] = 'missing'
    with pytest.raises(ValueError, match='existing columns'):
        subphenotypes.selection(path, chosen)


def test_filter_applies_to_discovery_run_and_review_without_changing_inputs(parents, tmp_path):
    root, path, indices = parents
    frame = pd.read_csv(path)
    frame['tumor'] = 1
    frame.loc[indices[:100], 'tumor'] = 0
    frame['phenotype_ref1'] = np.where(frame.CD20 > 500, 'Activated T', 'Resting T')
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    filters = [{'column': 'tumor', 'op': 'in', 'values': ['1']}]
    args = dict(filters=filters, chosen_markers={'T cell': ['CD3', 'CD20']}, reference_column='phenotype_ref1')
    found = subphenotypes.run(job(root, path, 'discover', **args), tmp_path / 'find', lambda *args: None)
    assert found['discovery']['T cell']['cells'] == len(indices) - 100
    result = subphenotypes.run(job(root, path, 'run', **args), tmp_path / 'run', lambda *args: None)
    table = dataio.load(path)
    eligible = table.phenotype_parent.eq('T cell') & table.tumor.eq(1)
    assert table.phenotype_sub_clusters.notna().equals(eligible)
    assert result['guidance']['labelled'] == result['cells'] == int(eligible.sum())
    assert result['selection']['filters'] == filters
    assert all(c['reference_suggestions'] and not c['model_votes'] for c in result['guidance']['clusters'].values())
    assert path.read_bytes() == before and table.phenotype_parent.tolist() == frame.phenotype_parent.tolist()
    json.dumps(result, allow_nan=False)


def test_inputs_protected_and_empty_filter_or_explicit_empty_markers_fail(parents, tmp_path):
    root, path, _ = parents
    selected = job(root, path, 'run')['config']
    for output in ('phenotype_parent', 'phenotype_ref1'):
        with pytest.raises(ValueError, match='separate from the inputs'):
            subphenotypes.check_inputs({**selected, 'reference_column': 'phenotype_ref1', 'studio_output_col': output},
                                       [*dataio.columns(path), 'phenotype_ref1'])
    with pytest.raises(ValueError, match='No cells match'):
        subphenotypes.run(job(root, path, 'run', filters=[{'column': 'phenotype_parent', 'op': 'in', 'values': ['absent']}]),
                          tmp_path / 'none', lambda *args: None)
    with pytest.raises(ValueError, match='fewer than two markers'):
        subphenotypes.run(job(root, path, 'run', chosen_markers={'T cell': []}), tmp_path / 'empty', lambda *args: None)


def test_saved_model_and_manual_evidence_stay_separate_and_exclusions_are_honest(training, monkeypatch):
    table, settings, root = training
    model = reference_models.train(settings, root / 'trained', lambda *args: None)['models'][0]['model_id']
    target = root / 'target.parquet'
    frame = dataio.load(table)
    # Deliberately contradict model evidence: manual reference must remain visible.
    frame['phenotype_ref1'] = np.where(frame.batch_id.eq('roi1'), 'Novel type', 'B')
    frame.to_parquet(target, index=False)
    groups = pd.Series(['first'] * 60 + ['second'] * 60)
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    for cls in (RandomForestClassifier, ExtraTreesClassifier):
        monkeypatch.setattr(cls, 'fit', lambda *a, **k: pytest.fail('Full-panel guidance must not retrain'))
    chosen = {'model_id': model, 'reference_column': 'phenotype_ref1', 'limit_model_phenotypes': True, 'model_phenotypes': ['B']}
    result = subphenotype_guidance.build(target, groups, chosen, lambda *args: None)
    first, second = result['clusters']['first'], result['clusters']['second']
    assert first['excluded_winner'] == 'A' and not first['suggestions']
    assert first['top'][0]['label'] == 'Novel type' and first['top'][0]['cells'] == 60
    assert second['suggestions'][0]['label'] == 'B' and second['suggestions'][0]['source'] == 'model'
    assert first['model_scored'] == second['model_scored'] == 60
    assert dataio.load(target).phenotype_ref1.tolist() == frame.phenotype_ref1.tolist()
    assert not any(c.endswith('_clusters') for c in dataio.columns(target))
    json.dumps(result, allow_nan=False)


def test_guidance_adapts_missing_markers_and_bounds_sample(training):
    table, settings, root = training
    model = reference_models.train(settings, root / 'trained', lambda *args: None)['models'][0]['model_id']
    target = root / 'partial.parquet'
    pd.concat([dataio.load(table)] * 10, ignore_index=True).drop(columns='M1').to_parquet(target)
    result = subphenotype_guidance.build(target, pd.Series(['subset'] * 1200), {'model_id': model}, lambda *args: None)
    assert result['features'] == ['M0'] and result['missing'] == ['M1']
    assert result['clusters']['subset']['model_sampled'] == result['clusters']['subset']['model_scored'] == 400
    assert result['clusters']['subset']['model_votes']


def test_shared_model_loads_for_subset_names_without_changing_training_selection(training, monkeypatch):
    from pathlib import Path
    from spacemap_gui.studio import reference_setup
    table, settings, root = training
    identifier = reference_models.train(settings, root / 'trained', lambda *args: None)['models'][0]['model_id']
    shared = root / 'shared'
    exported = Path(reference_setup.publish(root, identifier, shared))
    artifact = exported / (identifier + '.joblib')
    original = artifact.read_bytes()
    local_models = root / 'recipient-models'
    local_models.mkdir()
    monkeypatch.setattr(reference_models, 'directory', lambda: local_models)
    destination = root / 'next-project'
    destination.mkdir()
    target = destination / 'cells.parquet'
    dataio.load(table).to_parquet(target, index=False)
    listed = reference_setup.configure(destination, {'action': 'folder', 'folder': str(shared)}, target)
    before = reference_setup.saved(destination)
    imported = reference_setup.prepare_model(destination, listed['models'][0]['key'], target)
    assert imported != identifier
    assert reference_setup.saved(destination) == before
    assert artifact.read_bytes() == original
    assert any(m['id'] == imported for m in reference_setup.options(destination, target)['models'])
    result = subphenotype_guidance.build(target, pd.Series(['A-cells'] * 60 + ['B-cells'] * 60),
                                         {'model_id': imported}, lambda *args: None)
    assert result['clusters']['A-cells']['suggestions'][0]['label'] == 'A'


def test_mrml_grouping_sums_probabilities_before_choosing_and_does_not_force_excluded(training, monkeypatch):
    table, settings, root = training
    model = reference_models.train(settings, root / 'trained', lambda *args: None)['models'][0]['model_id']
    # A third class tests that two smaller fine-type scores can win together.
    snapshot = reference_models._source(model)
    artifact = {**snapshot[2]}
    from copy import copy
    artifact['model'] = copy(artifact['model'])
    artifact['model'].classes_ = np.array(['Tumor', 'Sarcomatoid', 'Immune'])
    meta = {**snapshot[1], 'classes': artifact['model'].classes_.tolist()}
    monkeypatch.setattr(reference_models, '_source', lambda *a: (model, meta, artifact))
    monkeypatch.setattr(reference_models, 'metadata', lambda *a: meta)
    monkeypatch.setattr(reference_models, '_scan_predictor', lambda *a: lambda art, values, rows: np.tile([.3, .3, .4], (len(rows), 1)))
    chosen = {**mr_ml.DEFAULTS, 'workflow': 'saved_model', 'model_id': model, 'enable_filter': False,
              'phenotype_groups': {'Tumor': 'Epithelia', 'Sarcomatoid': 'Epithelia'}}
    configured = mr_ml.job_settings(root, table, {'cell_id': 'cell_id'}, chosen)
    result = mr_ml.predict_saved(configured, root / 'predicted', lambda *a: None)
    found = dataio.load(table)
    assert set(found[configured['output_col']]) == {'Epithelia'}
    np.testing.assert_allclose(found[configured['confidence_column']], .6)
    assert result['phenotype_groups'] == chosen['phenotype_groups']
    chosen = {**chosen, 'limit_phenotypes': True, 'prediction_phenotypes': ['Tumor'],
              'output_phenotype_col': 'phenotype_selected', 'confidence_col': 'selected_confidence',
              'refined_output_phenotype_col': 'phenotype_selected_refined'}
    result = mr_ml.predict_saved(mr_ml.job_settings(root, table, {'cell_id': 'cell_id'}, chosen), root / 'limited', lambda *a: None)
    assert result['outside_selected_phenotypes'] == 120
    assert dataio.load(table).phenotype_selected.isna().all()


def test_grouping_leaves_original_model_and_manual_reference_unchanged(training):
    table, settings, root = training
    model = reference_models.train(settings, root / 'trained', lambda *args: None)['models'][0]['model_id']
    original = reference_models.model_path(model).read_bytes()
    chosen = {**mr_ml.DEFAULTS, 'workflow': 'saved_model', 'model_id': model, 'enable_filter': False,
              'preserve_reference_labels': True, 'phenotype_groups': {'A': 'Broad', 'B': 'Broad'}}
    configured = mr_ml.job_settings(root, table, {'cell_id': 'cell_id'}, chosen)
    mr_ml.predict_saved(configured, root / 'predicted', lambda *a: None)
    found = dataio.load(table)
    assert set(found[configured['output_col']]) == {'Broad'}
    assert set(found.phenotype_ref1.dropna()) == {'A', 'B'}
    assert reference_models.model_path(model).read_bytes() == original
