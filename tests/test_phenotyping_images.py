"""A shared image selection must limit both computation and saved outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap_gui.studio import phenotyping_images as scope, phenotyping_inputs as inputs
from spacemap_gui.studio import reference_collection as collection, reference_models, tasks, mr_ml, leiden, clustering
from test_studio_commands import setup  # noqa: F401
from test_studio_reference_collection import study  # noqa: F401
from test_studio_reference_models import training  # noqa: F401
from test_studio_mr_ml import model_job


def test_merged_images_are_selectable_and_training_uses_only_checked_images(study):
    channel, root, paths = study
    frames = []
    for name, path in zip(['P11_roi_2', 'second'], paths):
        frame = pd.read_csv(path); frame['image'] = name; frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame['cell_id'] = np.arange(len(frame))
    path = root/'merged.parquet'; frame.to_parquet(path, index=False)
    inputs.use_table(channel, str(path))
    inputs.save_roles(root, path, {'roi':'image','slide':'','cell_id':'cell_id','label':'label'})
    found = collection.plan(root, path)
    assert {item['image'] for item in found['tables']} == {'P11_roi_2', 'second'}
    assert {item['path'] for item in found['tables']} == {str(path)}
    scope.save(root, path, ['second'])
    from spacemap_gui.studio import manual_reference
    single = manual_reference.training_settings(channel, 'phenotype_ref1')
    assert single['selected_images'] == ['P11_roi_2']  # Train model follows the open image.
    settings = collection.prepare(channel, {'action':'train','reference_column':'phenotype_ref1'})
    assert len(settings['sources']) == 1 and settings['sources'][0]['selected_images'] == ['second']
    rows, truth = collection._examples(settings['sources'][0], lambda *a:None)
    assert len(rows) == 15 and set(truth) == {'B'} and min(rows) >= 20
    scope.save(root, path, [])
    with pytest.raises(ValueError, match='Select at least one image'):
        collection.prepare(channel, {'action':'train','reference_column':'phenotype_ref1'})


def test_training_one_image_retains_other_images_in_the_same_table(training):
    table, settings, root = training
    parent = reference_models.train(settings, root/'parent', lambda *a:None)['models'][0]
    child = reference_models.train({**settings,'base_model_id':parent['model_id'],'include_saved_examples':True,
        'selected_images':['roi1']}, root/'child', lambda *a:None)['models'][0]
    assert child['classes'] == ['A', 'B'] and child['examples'] == 59
    assert child['saved_example_cells'] > 0
    _, _, artifact = reference_models._source(child['model_id'])
    assert all(row < 60 for row in artifact['rows'])
    assert any((group['positions'] >= 60).any() for group in artifact['example_archive'])


def test_saved_prediction_preserves_unselected_image_and_manual_edits(training):
    table, settings, root = training
    model = reference_models.train(settings, root/'fit', lambda *a:None)['models'][0]
    job = model_job(root, table, model['model_id'], preserve_reference_labels=False)
    job.update(selected_images=['roi1'])
    result = mr_ml.predict_saved(job, root/'first', lambda *a:None)
    assert result['cells'] == 60
    found = dataio.load(table)
    assert found.loc[found.batch_id=='roi1',job['output_col']].eq('A').all()
    assert found.loc[found.batch_id=='roi2',job['output_col']].isna().all()
    labels.append(table,[{'row':0,'column':job['output_col'],'value':'My label'}])
    rerun = {**job,'selected_images':['roi2'],'overwrite_columns':[job['output_col'],*job['auxiliary_columns']]}
    rerun['revision'] = tasks.source_revision(rerun)
    mr_ml.predict_saved(rerun, root/'second', lambda *a:None)
    found = dataio.load(table)
    assert found.loc[0,job['output_col']] == 'My label'
    assert found.loc[found.batch_id=='roi2',job['output_col']].eq('B').all()
    assert job['output_col'] not in pd.read_csv(table).columns


@pytest.mark.parametrize('method',['leiden','som','vortex'])
def test_clustering_only_receives_selected_rows(training, monkeypatch, method):
    table, _, root = training
    values = {**leiden.DEFAULTS, 'markers':['M0','M1'],'use_gpu':False,'enable_low_expr_filter':False}
    settings = {'table':str(table),'markers':['M0','M1'],'cell_id_col':'cell_id','roi_col':'batch_id',
                'roi_column':'batch_id','selected_images':['roi2'],'output_col':'phenotype_scope_clusters',
                'phenotype_col':'phenotype_scope','project_root':str(root),'method':method,'cofactor':5,
                'config':values,'pipeline':leiden.pipeline_settings(values,table,5)}
    settings['revision'] = tasks.source_revision(settings)
    def compute(frame, *args, **kwargs):
        assert set(frame.batch_id) == {'roi2'} and len(frame) == 60
        frame = frame.copy(); frame['meta_cluster' if method=='som' else 'vortex_cluster'] = 0
        return frame, ['M0','M1'], 'cell_id', 2
    if method == 'leiden':
        from spacemap_workflows import leiden_pipeline
        def calculate(request, folder, *args):
            frame = pd.read_csv(request['settings']['full_input_path'])
            assert set(frame.batch_id) == {'roi2'} and len(frame) == 60
            frame = frame.copy(); frame['leiden_cluster'] = 0
            path = Path(folder)/'clusters.parquet'; path.parent.mkdir(parents=True); frame.to_parquet(path,index=False)
            return {'leiden_temp_file':str(path),'leiden_norm_cols':['M0','M1'],'leiden_n_clusters':1}
        monkeypatch.setattr(leiden_pipeline,'_compute',calculate)
        monkeypatch.setattr(leiden,'describe',lambda *a:([],{}))
        result = leiden.run(settings, root/'cluster', lambda *a:None)
    else:
        from spacemap_workflows import som_pipeline, vortex_pipeline
        monkeypatch.setattr(som_pipeline if method=='som' else vortex_pipeline,'compute',compute)
        monkeypatch.setattr(clustering,'describe',lambda *a:([],{}))
        result = clustering.run(settings, root/'cluster', lambda *a:None)
    assert result['cells'] == 60
    found = dataio.load(table)
    assert found.loc[found.batch_id=='roi1',settings['output_col']].isna().all()
    assert found.loc[found.batch_id=='roi2',settings['output_col']].notna().all()


def test_image_selection_api_persists_and_rejects_unknown_images(study):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    channel, root, paths = study
    route = '/studio/'+channel+'/phenotyping_inputs'
    with TestClient(app) as client:
        result = client.post(route, json={'selected_images':['second']})
        assert result.status_code == 200, result.text
        assert client.get(route).json()['selected_images'] == ['second']
        assert client.post(route,json={'selected_images':['missing']}).status_code == 400
        selection = collection.plan(root, paths[0])
        assert selection['selected_images'] == ['second']
        assert collection.prepare(channel, {'action':'train','reference_column':'phenotype_ref1'})['sources'][0]['table'] == paths[1]


def test_multiple_image_jobs_are_preflighted_and_publish_individual_results(study, monkeypatch):
    from spacemap_gui.studio import commands, jobs
    from spacemap_gui.viewer import registry
    from spacemap_gui.studio.settings import revision
    channel, root, paths = study
    cache = leiden.cache_path(root); cache.parent.mkdir(parents=True,exist_ok=True)
    import json
    cache.write_text(json.dumps({'markers':['M0','M1']}))
    launched=[]
    monkeypatch.setattr(jobs,'start',lambda root,kind,settings,**kw: launched.append((kind,settings)) or {'id':'a'*32})
    args={'settings_revision':revision(leiden.values(root))}
    answer = scope.start(channel,'run_leiden',args,{},'selection')
    assert answer[2]['job']['id']=='a'*32 and launched[0][0]=='phenotyping_images'
    work = launched[0][1]
    assert {item['settings']['table'] for item in work['work']} == set(paths)
    assert all(item['settings']['cell_id_col']=='label' for item in work['work'])
    calls=[]
    def compute(settings, folder, report):
        calls.append(settings['table'])
        frame=dataio.load(settings['table'])
        return tasks.save_result(settings,{settings['output_col']:['Cluster 0']*len(frame)},frame.label,report)
    monkeypatch.setattr(tasks,'leiden',compute)
    folder=root/'batch';folder.mkdir()
    result=scope.run(work,folder,lambda *a:None)
    assert result['images']==2 and result['cells']==40
    assert len([job for job in jobs.recent(root) if job['kind']=='leiden' and job['status']=='done'])==2
    assert calls == paths
    # A pre-existing output in one table blocks the entire next batch.
    with pytest.raises(ValueError,match='already exists'):
        scope.start(channel,'run_leiden',args,registry.session_view(channel), 'another')
    assert len(launched)==1
