"""Reusable cell splits and fair multi-method figures for per-image inputs."""
import base64
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from spacemap import dataio, pending_columns
from spacemap_gui.studio import phenotype_assessment as assessment, reference_splits, reference_models
from spacemap_gui.viewer import api, registry
from test_studio_commands import setup  # noqa: F401
from test_studio_reference_collection import study, train  # noqa: F401


@pytest.fixture
def scored(study):  # noqa: F811
    fitted = train(study)
    for i, path in enumerate(study[2]):
        frame = pd.read_csv(path)
        frame['phenotype_good'] = 'T' if i == 0 else 'B'
        frame['phenotype_missing'] = None
        if i == 0:
            frame['phenotype_missing'] = 'T'
        frame.to_csv(path, index=False)
        dataio.ensure_fast_copy(path, force=True)
    return study, fitted


def request(fitted, subset='validation', **changes):
    return {'reference': 'phenotype_ref1', 'methods': ['phenotype_good', 'phenotype_missing'],
            'split_id': fitted['split_id'], 'subset': subset, **changes}


def test_split_is_reusable_across_methods_with_missing_predictions(scored):
    study, fitted = scored
    channel, root, paths = study
    before = [Path(path).read_bytes() for path in paths]
    results = {subset: assessment.compare(channel, request(fitted, subset)) for subset in ('training', 'validation', 'both', 'all')}
    assert [results[key]['cells'] for key in ('training', 'validation', 'both', 'all')] == [21, 9, 30, 30]
    for found in results.values():
        good, missing = found['summary']
        assert good['macro_f1'] == 1
        assert missing['n_test'] == good['n_test'] == found['cells']
        assert missing['n_rejected'] > 0 and missing['macro_f1'] < 1
    members = {key: assessment.result(root, result['id'])['membership'] for key, result in results.items()}
    ids = lambda key: set(zip(members[key].table, members[key].row))
    assert not ids('training') & ids('validation')
    assert ids('training') | ids('validation') == ids('both')
    repeated = assessment.compare(channel, request(fitted))
    assert repeated['summary'] == results['validation']['summary']
    assert [Path(path).read_bytes() for path in paths] == before
    # Two fitted algorithms share one immutable split, independent of weights.
    assert len(reference_splits.options(root, paths)) == 1
    for model in fitted['models']:
        reference_models.model_path(model['model_id']).unlink()
    assert assessment.compare(channel, request(fitted))['cells'] == 9


def test_all_labels_can_be_assessed_without_training_or_correction(study):  # noqa: F811
    channel, _, paths = study
    for path in paths:
        frame = pd.read_csv(path)
        frame['other_method'] = frame.phenotype_ref1
        frame.to_csv(path, index=False)
        dataio.ensure_fast_copy(path, force=True)
    assert assessment.options(channel)['splits'] == []
    found = assessment.compare(channel, {'reference': 'phenotype_ref1', 'methods': ['other_method'], 'subset': 'all'})
    assert found['cells'] == 30 and found['summary'][0]['macro_f1'] == 1


def test_split_survives_result_column_commit_but_rejects_changed_cells_and_labels(scored):
    study, fitted = scored
    channel, _, paths = study
    pending_columns.save(paths[0], {'new_prediction': ['T'] * 20}, anchor_column='label', anchor_values=np.arange(1, 21))
    pending_columns.commit(paths[0])
    assert assessment.compare(channel, request(fitted))['cells'] == 9
    original = pd.read_csv(paths[0])
    changed = original.iloc[::-1].reset_index(drop=True)
    changed.to_csv(paths[0], index=False)
    dataio.ensure_fast_copy(paths[0], force=True)
    with pytest.raises(ValueError, match='identities changed'):
        assessment.compare(channel, request(fitted))
    original['phenotype_ref1'] = 'Changed'
    original.to_csv(paths[0], index=False)
    dataio.ensure_fast_copy(paths[0], force=True)
    with pytest.raises(ValueError, match='Reference labels.*changed'):
        assessment.compare(channel, request(fitted))


def test_methods_missing_from_one_image_count_as_unassigned(scored):
    study, fitted = scored
    channel, _, paths = study
    frame = pd.read_csv(paths[1]).drop(columns=['phenotype_missing'])
    frame.to_csv(paths[1], index=False)
    dataio.ensure_fast_copy(paths[1], force=True)
    found = assessment.compare(channel, request(fitted, 'both'))
    assert found['summary'][1]['n_rejected'] == 15
    assert len(found['notices']) == 1
    with pytest.raises(ValueError, match='not found'):
        assessment.compare(channel, request(fitted, methods=['nonexistent']))
    with pytest.raises(ValueError, match='different from the reference'):
        assessment.compare(channel, request(fitted, methods=['phenotype_ref1']))
    with pytest.raises(ValueError, match='paired'):
        assessment.compare(channel, request(fitted, tables=['/unrelated.csv']))


def test_classic_figures_and_export_match_the_scored_cells(scored):
    from spacemap_gui.tabs.tab9_visualization import f1_validation_utils as classic
    study, fitted = scored
    channel, root, _ = study
    found = assessment.compare(channel, request(fitted))
    record = assessment.result(root, found['id'])
    # The same Classic engine produces summary/per-class/confusion results.
    _, frame, _, methods, _, _, _ = assessment._frame(channel, request(fitted))
    expected = classic.analyze_validation(frame, reference_columns=['phenotype_ref1'], method_columns=methods,
        excluded_labels=[], group_column='__assessment_image', prediction_design='Validation cells')
    pd.testing.assert_frame_equal(expected['summary'], record['bundle']['summary'])
    for kind in assessment.PLOTS:
        data, mime = assessment.figure(root, found['id'], kind, 'phenotype_good', {'dpi': 72})
        assert data.startswith(b'\x89PNG') and mime == 'image/png'
    for fmt, marker in [('svg', b'<svg'), ('pdf', b'%PDF')]:
        data, _ = assessment.figure(root, found['id'], options={'format': fmt, 'dpi': 72})
        assert marker in data
    with zipfile.ZipFile(io.BytesIO(assessment.export(root, found['id']))) as archive:
        members = pd.read_csv(archive.open('cells.csv'))
        scores = pd.read_csv(archive.open('summary.csv'))
        assert len(members) == 9 and set(members.subset) == {'validation'}
        assert scores.query("group == 'Pooled'").n_test.tolist() == [9, 9]
        assert json.loads(archive.read('assessment.json'))['split_id'] == fitted['split_id']


def test_api_read_only_comparisons_and_project_scoping(scored):
    study, fitted = scored
    channel, root, _ = study
    project = registry.session_project(channel)
    view = registry.session_view(channel, mark_seen=False)['spec']
    registry.publish(channel, view, project={**project, 'read_only': True})
    url = f'/studio/{channel}/phenotype_assessment'
    with TestClient(api.app) as client:
        options = client.get(url)
        assert options.status_code == 200 and len(options.json()['splits']) == 1
        result = client.post(url, json=request(fitted))
        assert result.status_code == 200, result.text
        identifier = result.json()['id']
        picture = client.post(url, json={'action': 'figure', 'id': identifier, 'options': {'dpi': 72}})
        assert picture.status_code == 200 and base64.b64decode(picture.json()['data']).startswith(b'\x89PNG')
        assert client.post(url, json=request(fitted, split_id='../bad')).status_code == 400
        assert client.post(url, json={'action': 'export', 'id': 'wrong'}).status_code == 400
    with pytest.raises(ValueError, match='Compare again'):
        assessment.result(root / 'other_project', identifier)


def test_recent_legacy_models_keep_their_original_split(study):  # noqa: F811
    fitted = train(study)
    channel, root, paths = study
    original = reference_splits.read(root, fitted['split_id'], paths)
    (reference_splits.directory(root) / (fitted['split_id'] + '.json')).unlink()
    for model in fitted['models']:
        path = reference_models.model_path(model['model_id']).with_suffix('.json')
        meta = json.loads(path.read_text())
        meta.pop('split_id')
        path.write_text(json.dumps(meta))
    choices = assessment.options(channel)['splits']
    restored = reference_splits.read(root, choices[0]['id'], paths)
    assert restored['tables'] == original['tables']
    assert restored['training_cells'] == 21 and restored['validation_cells'] == 9
