import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import pytest
from spacemap import dataio
from spacemap_gui.studio import flow_gating as flow, flow_cache
from test_studio_commands import setup


def test_category_count_projection_cache_and_exported_zero_units(setup):
    channel,table,root,_=setup
    frame=pd.read_csv(table);frame['patient']=['A','B','A','B','C'];frame['phenotype']=['Treg','CD8','Treg','CD8','Treg']
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    gates=[{'id':name,'kind':'range','name':name,'colour':'#123456','x':{'column':'M0'},'bounds':bounds}
           for name,bounds in [('a',[1,2]),('b',[3,4])]]
    ws=flow.save(channel,{'revision':0,'changes':{'gates':gates}})['workspace']
    design={'groups':[{'id':g,'population':g} for g in ('a','b')],
            'features':[{'id':metric,'metric':metric,'target_rule':{'column':'phenotype','values':['Treg']},'denominator':'@group'} for metric in ('count','proportion')],
            'group_by':'patient','min_cells':1,'test':'paired_t'}
    body={'action':'stats_run','revision':ws['revision'],'design':design}
    r=flow.dispatch(channel,body)
    assert [s['value'] for s in r['samples']]==[1,0,1,0,100,0,100,0]
    assert all(s['N']==2 for s in r['summaries'])
    assert flow.dispatch(channel,body)['cached']
    exported=flow.dispatch(channel,{**body,'action':'stats_export','format':'csv'})
    text=base64.b64decode(exported['download'].split(',')[1]).decode('utf-8-sig')
    assert 'n_target' in text and 'denominator_cells' in text and 'target_rule' in text
    body['design']['features'][0]['target_rule']['values']=['CD8']
    changed=flow.dispatch(channel,body)
    assert not changed['cached'] and changed['samples'][0]['value']==0


def test_stats_api_cache_exports_and_filtered_spatial_selection(setup,monkeypatch):
    channel,table,root,_=setup
    frame=pd.read_csv(table);frame['patient']=['A','B','A','B','C'];frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    gates=[{'id':'a','kind':'range','name':'A','colour':'#123456','x':{'column':'M0'},'bounds':[1,2]},
           {'id':'b','kind':'range','name':'B','colour':'#987654','x':{'column':'M0'},'bounds':[3,4]},
           {'id':'high','kind':'threshold','name':'M positive','colour':'#234567','x':{'column':'M0'},'bounds':[2]}]
    ws=flow.save(channel,{'revision':0,'changes':{'gates':gates}})['workspace']
    design={'groups':[{'id':'a','population':'a'},{'id':'b','population':'b'}],
            'features':[{'id':'m','column':'M0','metric':'mean'},{'id':'p','column':'M0','metric':'positive','threshold_id':'high'}],
            'group_by':'patient','min_cells':1,'test':'paired_t'}
    body={'revision':ws['revision'],'design':design,'action':'stats_run'}
    result=flow.dispatch(channel,body)
    assert len(result['tests'])==2 and all(r['matched_N']==2 for r in result['tests'])
    again=flow.dispatch(channel,body);assert again['cached']
    assert flow_cache.state(root)['files']>=2
    # Disk cache survives a process-memory reset.
    flow_cache._MEMORY.clear()
    assert flow.dispatch(channel,body)['cached']
    for kind in ('csv','xlsx'):
        export=flow.dispatch(channel,{**body,'action':'stats_export','format':kind})
        raw=base64.b64decode(export['download'].split(',')[1]);assert Path(export['file']).exists()
        if kind=='xlsx':
            with zipfile.ZipFile(BytesIO(raw)) as z:
                assert len([s for s in z.namelist() if 'worksheets/sheet' in s])==5
                for name in z.namelist():ET.fromstring(z.read(name))
        else:
            text=raw.decode('utf-8-sig');assert 'n_cells_a' in text and 'dataset_revision' in text
    subset=flow.select(channel,{'revision':ws['revision'],'populations':['b'],'rule':{'column':'patient','values':['A']}})
    assert subset['count']==1
    assert flow.token_rows(subset['token'],channel,str(table)).tolist()==[2]
    with pytest.raises(ValueError,match='changed'):flow.dispatch(channel,{**body,'revision':0})
    flow_cache.clear(root)
    assert flow_cache.state(root)['files']==0
    assert Path(export['file']).exists()


def test_threshold_auto_and_global_restore_with_undo(setup):
    channel,*_=setup
    spec={'parent':'root','x':{'column':'M0'},'bins':4,'sample':100,'kind':'histogram'}
    gate={'id':'high','kind':'threshold','parent':'root','name':'High','colour':'#123456','x':{'column':'M0'},'bounds':[2]}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    hist=flow.plot(channel,{'revision':1,'plot':spec})['histogram']
    assert hist['series'][0]['above_threshold']==2
    global_t={k:gate[k] for k in ('id','name','x','bounds')};global_t['id']='global'
    flow.save(channel,{'revision':1,'changes':{'thresholds':[global_t]}})
    assert flow.load(channel)['thresholds'][0]['id']=='global'
    flow.save(channel,{'revision':2,'changes':{'gates':[]}})
    assert flow.plot(channel,{'revision':3,'plot':{**spec,'threshold_id':'global'}})['histogram']['series'][0]['above_percent']==50
    flow.save(channel,{'revision':3,'action':'undo'})
    assert flow.load(channel)['thresholds'][0]['id']=='global'


def test_each_gate_parent_denominators_and_cache_invalidates_for_edits(setup):
    channel,table,_,_=setup
    frame=pd.read_csv(table);frame['patient']=['A','B','A','B','C'];frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    gates=[{'id':'parent','kind':'range','name':'Parent','colour':'#123456','x':{'column':'M0'},'bounds':[1,4]},
           {'id':'a','parent':'parent','kind':'range','name':'A','colour':'#123456','x':{'column':'M0'},'bounds':[1,2]},
           {'id':'b','parent':'parent','kind':'range','name':'B','colour':'#123456','x':{'column':'M0'},'bounds':[3,4]}]
    flow.save(channel,{'revision':0,'changes':{'gates':gates}})
    d={'groups':[{'id':'a','population':'a'},{'id':'b','population':'b'}],
       'features':[{'id':'p','metric':'proportion','denominator':'@parent'}], 'group_by':'patient','min_cells':1}
    body={'action':'stats_run','revision':1,'design':d};result=flow.dispatch(channel,body)
    assert all(s['value']==50 for s in result['samples'])
    assert result['design']['features'][0]['denominators']=={'a':'parent','b':'parent'}
    assert flow.dispatch(channel,body)['cached']
    gates[-1]['bounds']=[4,4];flow.save(channel,{'revision':1,'changes':{'gates':gates}})
    new=flow.dispatch(channel,{**body,'revision':2});assert not new['cached']
    assert result['result_id']!=new['result_id']


def test_data_changes_block_stale_figure_report_and_dot_selection(setup):
    from spacemap_gui.studio import flow_statistics
    channel,table,_,_=setup
    frame=pd.read_csv(table);frame['patient']=['A','B','A','B','C'];frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    d={'groups':[{'id':'a','population':'root'},{'id':'b','population':'root'}],
       'features':[{'id':'m','metric':'mean','column':'M0'}], 'group_by':'patient','min_cells':1}
    body={'action':'stats_run','revision':0,'design':d};result=flow.dispatch(channel,body)
    flow_statistics.validate_figure(flow.load(channel),result)
    frame['M0']+=1;frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    with pytest.raises(ValueError,match='changed'):
        flow_statistics.validate_figure(flow.load(channel),result)
    with pytest.raises(ValueError,match='changed'):
        flow.dispatch(channel,{**body,'action':'stats_export','format':'csv','expected_dataset_revision':result['dataset_revision']})
    with pytest.raises(ValueError,match='changed'):
        flow.select(channel,{'revision':0,'populations':['root'],'expected_dataset_revision':result['dataset_revision']})
