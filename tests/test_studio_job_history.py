"""Clearing job cards preserves durable Parquet data and provenance."""
import json

import pandas as pd
import pytest

from spacemap import dataio, pending_columns
from spacemap_gui.studio import jobs
from test_studio_commands import setup  # noqa: F401


def record(project, name, state='done', settings=None):
    job_id = name * 32
    folder = jobs.root(project) / job_id
    folder.mkdir(parents=True)
    (folder / 'status.json').write_text(json.dumps({
        'id':job_id, 'kind':'mr_ml', 'status':state, 'started':'2026-10-19T20:00:00+00:00',
        'message':'Finished', 'progress':1 if state == 'done' else .2,
    }))
    (folder / 'request.json').write_text(json.dumps({'kind':'mr_ml', 'settings':settings or {}}))
    (folder / 'terminal.log').write_text('Saved predictions\n')
    return job_id


def test_clear_selected_and_finished_keep_parquet_outputs_and_running_jobs(tmp_path):
    table = tmp_path / 'cells.parquet'
    pd.DataFrame({'cell_id':['a','b']}).to_parquet(table)
    pending_columns.save(table, {'phenotype_mr_ml':['T','B']}, anchor_column='cell_id', anchor_values=['a','b'])
    first = record(tmp_path, 'a', settings={'table':str(table), 'output_col':'phenotype_mr_ml'})
    failed = record(tmp_path, 'b', 'failed')
    active = record(tmp_path, 'c', 'running')
    original = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    cleared = jobs.clear_history(tmp_path, [first, active])
    assert cleared == {'cleared':[first], 'active':[active]}
    assert jobs.status(tmp_path, first)['hidden']
    assert not jobs.status(tmp_path, failed)['hidden']
    jobs.clear_history(tmp_path)
    assert jobs.status(tmp_path, failed)['hidden']
    assert not jobs.status(tmp_path, active)['hidden']
    assert jobs.status(tmp_path, active)['status'] == 'running'
    assert all(p.read_bytes() == content for p, content in original.items())
    assert dataio.load(table).phenotype_mr_ml.tolist() == ['T','B']
    assert not list(tmp_path.glob('*.csv'))
    assert jobs.log(tmp_path, first) == 'Saved predictions\n'
    assert {item['id'] for item in jobs.recent(tmp_path)} == {first, failed, active}


def test_invalid_selection_does_not_clear_other_jobs(tmp_path):
    first = record(tmp_path, 'a')
    with pytest.raises(ValueError):
        jobs.clear_history(tmp_path, [first, '../outside'])
    assert not jobs.status(tmp_path, first)['hidden']


def test_summaries_explain_prediction_and_training_scope():
    settings = {'table':'/project/cells.parquet', 'selected_images':['image1','image2'],
                'reference_column':'phenotype_ref1', 'output_col':'phenotype_mr_ml', 'markers':['CD3','CD8']}
    assert jobs.describe('mr_ml', settings) == '2 images · phenotype_ref1 → phenotype_mr_ml · 2 features'
    collection = {'sources':[{'table':'one.csv','selected_images':['image1']},
                            {'table':'two.csv','selected_images':['image2','image3']}],
                  'training_column':'phenotype_ref1', 'markers':['CD8']}
    assert jobs.describe('reference_collection', collection) == '3 images · Train from phenotype_ref1 · 1 feature'
    assert jobs.describe('phenotyping_images', {'work':[{'kind':'mr_ml','settings':settings}]}).startswith('MR ML · 2 images')


def test_training_status_keeps_target_version_across_reconnects(tmp_path):
    job_id = record(tmp_path, 'f', 'running')
    folder = jobs.root(tmp_path) / job_id
    record_path = folder / 'status.json'
    state = json.loads(record_path.read_text())
    state.update(kind='train_cellpose', message='Scoring validation tiles…')
    record_path.write_text(json.dumps(state))
    request_path = folder / 'request.json'
    request_path.write_text(json.dumps({'kind':'train_cellpose', 'settings':{
        'training_session':{'name':'studio_example', 'versions':[{}, {}]}}}))
    before = {p:p.read_bytes() for p in (record_path, request_path)}
    assert jobs.status(tmp_path, job_id)['model_name'] == 'studio_example_v3'
    assert all(p.read_bytes() == value for p, value in before.items())
    state.update(status='done', result={'name':'actual_saved_version'})
    record_path.write_text(json.dumps(state))
    assert jobs.status(tmp_path, job_id)['model_name'] == 'actual_saved_version'
    state.update(status='running', result=None)
    record_path.write_text(json.dumps(state))
    request_path.write_text(json.dumps({'kind':'train_cellpose', 'settings':{}}))
    assert 'model_name' not in jobs.status(tmp_path, job_id)  # legacy job: do not invent a version


def test_clear_history_api_preserves_direct_job_access(setup):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api

    channel, table, _, _ = setup
    job_id = record(table.parent, 'a', settings={'table':str(table),'output_col':'phenotype_mr_ml'})
    with TestClient(api.app) as client:
        route = f'/studio/{channel}/jobs'
        assert client.post(route+'/clear', json={}).status_code == 400
        response = client.post(route+'/clear', json={'ids':[job_id]})
        assert response.status_code == 200 and response.json()['cleared'] == [job_id]
        found = client.get(route+'/'+job_id).json()
        assert found['job']['hidden'] and found['log'] == 'Saved predictions\n'
        assert 'phenotype_mr_ml' in found['job']['summary']


