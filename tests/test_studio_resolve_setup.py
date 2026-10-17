"""Studio RESOLVE shares classic gates/rules and preserves exact ROI identity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap.resolve_live import Panel
from spacemap_gui.studio import resolve_setup as setup, settings, tasks


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('SPACEMAP_HOME', str(tmp_path / 'home'))
    root = tmp_path / 'project'
    root.mkdir()
    rng = np.random.default_rng(52)
    frame = pd.DataFrame({'batch_id': ['Slide_roi_1']*40 + ['Slide_roi_2']*40,
                          'cell_id': np.arange(80), 'label': list(range(40))*2,
                          'CD3e': np.tile(np.r_[rng.uniform(100, 120, 20), rng.uniform(1, 5, 20)], 2),
                          'CD20': np.tile(np.r_[rng.uniform(1, 5, 20), rng.uniform(100, 120, 20)], 2),
                          'phenotype_ref1': ['T']*20 + ['B']*20 + [None]*40})
    table = root / 'cells.parquet'
    frame.to_parquet(table, index=False)
    return root, str(table), frame


def body(root, table, mode='Predefined (CSV)'):
    state = setup.get(root, table)
    return {**state, 'values': {**state['values'], 'markers':['CD3e','CD20'], 'threshold_mode':mode},
            'rules':'T = [CD3e]\nB = [CD20]',
            'thresholds':[{'batch_id':name,'CD3e':30.,'CD20':30.} for name in ('Slide_roi_1','Slide_roi_2')]}


def test_save_roundtrip_classic_files_and_whole_table_run(workspace):
    root, table, frame = workspace
    saved = setup.save(root, table, body(root, table))
    assert saved['rules'] == 'T = [CD3e]\nB = [CD20]'
    cache = root / 'outputs/phenotyping/cache'
    assert json.loads((cache/'resolve_params.json').read_text())['markers'] == ['CD3e','CD20']
    config = setup.job_settings(root, table, {'column':'phenotype_test','settings_revision':saved['revision']})
    result = tasks.resolve_all(config, root, lambda *args:None)
    assert result['cells'] == 80 and result['rois'] == 2
    labels = dataio.load(table, columns=['phenotype_test']).phenotype_test
    assert labels.tolist() == (['T']*20 + ['B']*20)*2
    pd.testing.assert_frame_equal(pd.read_parquet(table), frame)
    with pytest.raises(ValueError, match='changed'):
        setup.save(root, table, {**saved, 'rules':'B=[CD20]'})


@pytest.mark.parametrize('mode', setup.MODES[1:])
def test_each_positivity_path_equals_classic(workspace, mode):
    from spacemap_gui.tabs.tab7_phenotyping.common import compute_auto_thresholds, _THRESHOLD_MODE_TO_KEY
    from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import _apply_thresholds_from_table, _apply_resolve_without_thresholds, _apply_resolve_z1_positivity
    from spacemap.resolve_live import Settings
    root, table, frame = workspace
    options = body(root, table, mode)['values']
    conf = Settings.from_config(options)
    markers = options['markers']
    binary, scores = tasks.resolve_scores(frame, pd.DataFrame(), markers, options, 'batch_id')
    if mode.startswith('Z1'):
        expected = _apply_resolve_z1_positivity(frame, markers, 'batch_id')
    elif mode.startswith('None'):
        expected = _apply_resolve_without_thresholds(frame, markers, 'batch_id', options, options['norm_order'])
    else:
        gates = compute_auto_thresholds(frame, markers, _THRESHOLD_MODE_TO_KEY[mode], 'batch_id')
        expected = _apply_thresholds_from_table(frame, gates, markers, 'batch_id', conf.norm_flags, list(conf.norm_order), 'resolve', conf.tie_score_transform, conf.tie_score_cofactor)
    pd.testing.assert_frame_equal(binary, expected[0])
    pd.testing.assert_frame_equal(scores, expected[1])


def test_background_polygon_aggregation_and_corrected_scale_guard(workspace):
    root, table, _ = workspace
    folder = root/'background'; folder.mkdir()
    for roi in ('Slide_roi_1','Slide_roi_2'):
        pd.DataFrame({'Marker':['CD3e','CD20'], 'Polygon_1':[10,20], 'Polygon_2':[20,40], 'Polygon_3':[90,90]}).to_csv(folder/(roi+'.ome.tiff_-_Merged_Image.csv'), index=False)
    options = body(root, table)['values']
    gates = setup.calculate(root, table, options, background_folder=str(folder))['thresholds']
    assert len(gates) == 2 and gates[0]['CD3e'] == 20 and gates[1]['CD20'] == 40
    with pytest.raises(ValueError, match='Background thresholds use raw'):
        setup.calculate(root, table, {**options, 'input_space':'arcsinh'}, background_folder=str(folder))
    options.update(threshold_mode='Z1 score (no threshold files)',input_space='arcsinh')
    frame = pd.read_parquet(table)
    transformed = frame.copy(); transformed[['CD3e','CD20']] = np.arcsinh(frame[['CD3e','CD20']]/5)
    _, a = tasks.resolve_scores(frame, pd.DataFrame(), options['markers'], {**options,'input_space':'raw'}, 'batch_id')
    _, b = tasks.resolve_scores(transformed, pd.DataFrame(), options['markers'], options, 'batch_id')
    pd.testing.assert_frame_equal(a,b)


def test_bad_rules_and_ambiguous_thresholds_never_replace_saved_files(workspace):
    root, table, _ = workspace
    saved = setup.save(root, table, body(root, table))
    previous = Path(setup.paths(root, table)['panel']).read_bytes()
    with pytest.raises(ValueError, match='unknown marker'):
        setup.save(root, table, {**saved,'rules':'T=[MissingMarker]'})
    assert Path(setup.paths(root, table)['panel']).read_bytes() == previous
    duplicate = [*saved['thresholds'], {**saved['thresholds'][0], 'batch_id':'Slide_roi_1.ome.tiff'}]
    with pytest.raises(ValueError, match='same ROI'):
        setup.save(root, table, {**saved,'thresholds':duplicate})
    with pytest.raises(ValueError, match='No thresholds'):
        setup.save(root, table, {**saved,'thresholds':saved['thresholds'][:1]})
    assert Path(setup.paths(root, table)['panel']).read_bytes() == previous


def test_reference_drafts_are_measured_review_only_rules(workspace):
    root, table, _ = workspace
    request = body(root, table)
    result = setup.draft_reference(root, table, {**request,'reference':'phenotype_ref1'})
    assert result['rules'] == 'T = [CD3e]\nB = [CD20]'
    assert all(row['coverage'] == row['purity'] == 1 for row in result['evidence'])
    assert not Path(setup.paths(root, table)['panel']).exists()
    assert 'phenotype_threshold' not in dataio.columns(table)
    panel = Panel.from_payload(setup.rules_payload('T=[CD3e]; SUPPORT=[CD20]; NOT=[CD20]', ['CD3e','CD20']))
    assert panel.supportive == panel.exclusions == {'T':['CD20']}


def test_filter_is_used_for_thresholds_drafts_and_assignments(workspace):
    root, table, _ = workspace
    request = body(root,table,'Otsu')
    request['values'].update(enable_filter=True,filter_col='batch_id',filter_val='Slide_roi_1')
    gates = setup.calculate(root,table,request['values'])['thresholds']
    assert len(gates) == 1 and gates[0]['batch_id'] == 'Slide_roi_1'
    request['values']['threshold_mode'] = 'Predefined (CSV)'; request['thresholds'] = gates
    saved = setup.save(root,table,request)
    config = setup.job_settings(root,table,{'column':'phenotype_filtered'})
    tasks.resolve_all(config,root,lambda *args:None)
    result = dataio.load(table,columns=['phenotype_filtered']).phenotype_filtered
    assert result.iloc[:40].notna().all() and result.iloc[40:].isna().all()


def test_methods_include_custom_reference_and_job_output_without_counting_blank(workspace):
    from spacemap_gui.studio.phenotyping_inputs import plan
    from spacemap.viewer.reference import register_column
    root, table, frame = workspace
    frame['review_labels'] = ['T','',None]+[None]*77
    frame['phenotype_custom'] = 'A'
    frame.to_parquet(table,index=False)
    register_column(table,'review_labels')
    folder=root/'outputs/studio/jobs'/('a'*32);folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps({'status':'done','result':{'table':table,'column':'phenotype_custom'}}))
    (folder/'request.json').write_text(json.dumps({'kind':'som','settings':{'table':table}}))
    found=plan(root,{},table)
    assert next(m for m in found['methods'] if m['id']=='som')['columns']==['phenotype_custom']
    assert 'review_labels' in next(m for m in found['methods'] if m['id']=='reference')['columns']
    assert found['rows'][0]['results']['review_labels']==1


def test_run_does_not_need_an_open_image(workspace, monkeypatch):
    from spacemap_gui.studio import commands, jobs
    from spacemap_gui.viewer import registry
    root, table, _ = workspace
    saved=setup.save(root,table,body(root,table))
    spec={'id':'resolve-no-image','layers':[]}
    registry.publish('resolve-no-image',spec,project={'root':str(root),'unified':table,'mappings':{}})
    captured=[]
    monkeypatch.setattr(jobs,'start',lambda root,kind,values,**kw: captured.append(values) or {'id':'a'*32,'status':'starting'})
    result=commands._execute('resolve-no-image','run_resolve_all',{'column':'phenotype_empty_view','settings_revision':saved['revision']},spec,[],'request')
    assert result[2]['job']['status']=='starting'
    assert captured[0]['table']==table and captured[0]['roi_column']=='batch_id'


def test_save_failure_restores_rules_thresholds_and_settings(workspace, monkeypatch):
    from spacemap_gui import persistence
    root,table,_=workspace
    saved=setup.save(root,table,body(root,table))
    files=setup.paths(root,table)
    before={key:Path(path).read_bytes() for key,path in files.items()}
    monkeypatch.setattr(persistence,'save_project_config',lambda *args,**kwargs:None)
    saved['values']['unclassified_label']='Other'
    saved['rules']='Changed=[CD3e]'
    with pytest.raises(ValueError,match='did not save'):
        setup.save(root,table,saved)
    assert before=={key:Path(path).read_bytes() for key,path in files.items()}


def test_ambiguous_reference_patterns_are_not_presented_as_rules(workspace):
    root,table,frame=workspace
    frame['phenotype_ref1']=['T','B']*40
    frame.to_parquet(table,index=False)
    draft=setup.draft_reference(root,table,{**body(root,table),'reference':'phenotype_ref1'})
    assert not draft['rules'] and len(draft['skipped'])==2


def test_adaptive_parent_child_matches_classic_across_small_rois(workspace):
    """Classic pools parent candidates: two 10-cell ROIs must not skip its 20-cell gate."""
    from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import _adaptive_resolve_hierarchy
    from spacemap.resolve_live import hierarchy
    from spacemap.ph_thres_utils import assign_phenotypes_binary_tiebreak
    root, table, frame = workspace
    frame=frame.iloc[list(range(10))+list(range(40,50))].reset_index(drop=True)
    frame['CD3e']=100.
    frame['CD20']=[1.]*15+[200.]*5
    frame.to_parquet(table,index=False)
    request=body(root,table,'None (No Thresholds)')
    request['rules']='T=[CD3e]\nChild=[CD3e, CD20]'
    request['values']['adaptive_hierarchy']=True
    saved=setup.save(root,table,request)
    config=setup.job_settings(root,table,{'column':'phenotype_adaptive'})
    result=tasks.resolve_all(config,root,lambda *args:None)
    assert result['adaptive_hierarchy'] and result['counts']=={'T':15,'Child':5}
    panel=Panel.from_payload(setup.rules_payload(request['rules'],request['values']['markers']))
    binary,scores=tasks.resolve_scores(frame,pd.DataFrame(),config['markers'],config['options'],'batch_id')
    refined=_adaptive_resolve_hierarchy(frame,binary,panel.definitions,hierarchy(panel.definitions))
    expected=assign_phenotypes_binary_tiebreak(pd.concat([refined,scores],axis=1),list(refined),list(scores),'phenotype',panel.definitions,hierarchy(panel.definitions))['phenotype']
    actual=dataio.load(table,columns=['phenotype_adaptive']).phenotype_adaptive
    assert actual.tolist()==expected.tolist()
    pure=tasks.resolve_batch(frame,pd.DataFrame(),config['markers'],panel,{**config['options'],'adaptive_hierarchy':False},'batch_id')
    assert pure.eq('Child').all()


def test_parquet_index_does_not_change_cell_assignment_positions(workspace):
    root,table,frame=workspace
    frame.index=pd.Index(range(500,580),name='stored_index')
    frame.to_parquet(table,index=True)
    setup.save(root,table,body(root,table))
    config=setup.job_settings(root,table,{'column':'phenotype_indexed'})
    tasks.resolve_all(config,root,lambda *args:None)
    result=dataio.load(table,columns=['cell_id','phenotype_indexed'])
    assert result.cell_id.tolist()==list(range(80))
    assert result.phenotype_indexed.tolist()==(['T']*20+['B']*20)*2
