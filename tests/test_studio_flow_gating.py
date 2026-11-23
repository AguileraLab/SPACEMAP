import copy
import pandas as pd
import pytest
from spacemap import dataio
from spacemap_gui.studio import flow_gating as flow, data_editor
from test_studio_commands import setup  # noqa: F401


def test_workspace_membership_labels_text_and_restore(setup):
    channel, table, root, _ = setup
    info=flow.options(channel)
    assert info['n']==4 # only checked/linked image, not the other ROI
    assert info['workspace']['output_column']=='phenotype_flow'
    gate={'id':'gate_a','kind':'rectangle','name':'Treg','colour':'#ff3355','parent':'root',
          'x':{'column':'M0','scale':'linear'},'y':{'column':'M1','scale':'linear'},'bounds':[2,4,1,2]}
    saved=flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    assert next(s for s in saved['stats'] if s['id']=='gate_a')['n']==2
    restored=flow.load(channel)
    assert restored['gates']==[gate]
    assert pd.read_parquet(restored['membership_file']).gate_a.sum()==2
    with pytest.raises(ValueError,match='another window'):
        flow.save(channel,{'revision':0,'changes':{}})
    selection=flow.select(channel,{'revision':1,'populations':['gate_a']})
    assert flow.select(channel,{'revision':1,'populations':['gate_a']})['token']==selection['token']
    assert flow.token_rows(selection['token'],channel,str(table)).tolist()==[1,2]
    with data_editor.locked(table) as e:
        e.register_rows(selection['token'],[1,2])
    assert data_editor.query(table,filters=[{'_row_selection':selection['token']}])['matched']==2
    original=table.read_bytes()
    request={'revision':1,'populations':['gate_a'],'column':'phenotype_ref1','preserve':True}
    preview=flow.label_plan(channel,request)[-1]
    assert preview['write']==2
    flow.apply_labels(channel,{**request,'receipt':preview['receipt']})
    assert dataio.load(table,columns=['phenotype_ref1']).phenotype_ref1.fillna('').tolist()==['','Treg','Treg','','']
    assert table.read_bytes()==original
    assert flow.load(channel)['applied']['column']=='phenotype_ref1'


def test_overlaps_preserve_existing_and_stale_apply(setup):
    channel,table,_,_=setup
    gate={'id':'one','kind':'range','name':'One','colour':'#112233','x':{'column':'M0'},'bounds':[1,3]}
    other={**copy.deepcopy(gate),'id':'two','name':'Two','bounds':[2,4]}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate,other]}})
    request={'revision':1,'populations':['one','two'],'column':'phenotype_flow'}
    preview=flow.label_plan(channel,request)[-1]
    assert preview['overlap']==2 and preview['write']==2
    flow.apply_labels(channel,{**request,'receipt':preview['receipt']})
    assert dataio.load(table,columns=['phenotype_flow']).phenotype_flow.fillna('').tolist()==['One','','','Two','']
    with pytest.raises(ValueError,match='changed'):
        flow.apply_labels(channel,{**request,'receipt':preview['receipt']})
    assert flow.label_plan(channel,{**request,'revision':2})[-1]['write']==0


def test_plot_sample_and_snapshot_independent_of_later_gates(setup):
    channel,*_=setup
    spec={'x':{'column':'M0','scale':'arcsinh','cofactor':5},'y':{'column':'M1'},'bins':4,'sample':100}
    plotted=flow.plot(channel,{'revision':0,'plot':spec})
    assert sum(plotted['histogram']['series'][0]['counts'])==4
    assert plotted['points'][0]['measured']==4
    saved=flow.dispatch(channel,{'action':'freeze','revision':0,'plot':spec})
    snapshot=copy.deepcopy(saved['workspace']['layouts'][0]['panels'][0]['snapshot'])
    gate={'id':'one','kind':'range','name':'One','colour':'#112233','x':{'column':'M0'},'bounds':[1,2]}
    flow.save(channel,{'revision':1,'changes':{'gates':[gate]}})
    assert flow.load(channel)['layouts'][0]['panels'][0]['snapshot']==snapshot
    flow.save(channel,{'action':'undo','revision':2})
    assert flow.load(channel)['gates']==[]


def test_rename_after_selection_does_not_serialize_cached_numpy_arrays(setup):
    channel,*_=setup
    gate={'id':'one','kind':'range','name':'One','colour':'#112233','x':{'column':'M0'},'bounds':[1,2]}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    flow.select(channel,{'revision':1,'populations':['one']})
    flow.save(channel,{'revision':1,'changes':{'gates':[{**gate,'name':'Renamed'}]}})
    assert flow.load(channel)['gates'][0]['name']=='Renamed'


def test_independent_recipes_and_filter_populations(setup):
    channel,*_=setup
    gate={'id':'b','kind':'filter','name':'B','colour':'#112233','rule':{'column':'phenotype_ml_consensus','values':['B']}}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    plot={'x':{'column':'M0','reference_population':'b','pipeline':[{'op':'zscore'},{'op':'cdf'},{'op':'nlog'}]},
          'parent':'b','bins':4,'sample':100}
    result=flow.plot(channel,{'revision':1,'plot':plot})
    assert result['plot']['x']['reference_n']==2
    assert result['histogram']['series'][0]['n']==2
    other=flow.plot(channel,{'revision':1,'plot':{**plot,'parent':'root','x':{'column':'M0','scale':'linear'}}})
    assert other['histogram']['series'][0]['n']==4
    assert result['histogram']['edges']!=other['histogram']['edges']


def test_gate_keeps_the_filters_of_its_authoring_plot(setup):
    channel,*_=setup
    rule={'column':'phenotype_ml_consensus','values':['B']}
    gate={'id':'filtered','name':'B high','colour':'#123456','kind':'range','x':{'column':'M0'},'bounds':[1,4],'rule':rule}
    saved=flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    assert next(s for s in saved['stats'] if s['id']=='filtered')['n']==2
    plotted=flow.plot(channel,{'revision':1,'plot':{'x':{'column':'M0'},'rule':rule,'bins':4,'sample':100}})
    assert plotted['histogram']['series'][0]['n']==2


def test_workspace_identity_and_data_drafts_are_guarded(setup):
    channel,table,*_=setup
    flow.options(channel) # warm data cache before an edit
    with pytest.raises(ValueError,match='project or cell table'):
        flow.dispatch(channel,{'action':'save','workspace':'another-project','revision':0,'changes':{}})
    state=data_editor.query(table)
    data_editor.mutate(table,{'kind':'cells','changes':[{'col':'M0','row':0,'value':99}]},revision=state['revision'])
    with pytest.raises(ValueError,match='Save the Data edits'):
        flow.options(channel)


def test_gate_edit_reuses_only_independent_branches(setup):
    channel,*_=setup
    first={'id':'a','kind':'range','name':'A','colour':'#112233','x':{'column':'M0'},'bounds':[1,3]}
    second={**first,'id':'b','name':'B','bounds':[2,4]}
    child={**first,'id':'child','parent':'a','name':'Child','bounds':[2,4]}
    saved=flow.save(channel,{'revision':0,'changes':{'gates':[first,second,child]}})
    _,_,before=flow.evaluate(saved['workspace'])
    changed=flow.save(channel,{'revision':1,'changes':{'gates':[{**first,'bounds':[3,4]},second,child]}})
    _,_,after=flow.evaluate(changed['workspace'])
    assert after['b'] is before['b']
    assert after['a'] is not before['a'] and after['child'] is not before['child']
    assert after['child'][0].tolist()==[False,False,True,True]


def test_plot_style_cache_and_live_invalidation(setup):
    channel,table,*_=setup
    spec={'x':{'column':'M0'},'bins':4,'sample':100,'kind':'histogram'}
    first=flow.plot(channel,{'revision':0,'plot':spec})
    second=flow.plot(channel,{'revision':0,'plot':{**spec,'kind':'ridgeline','style':{'font':24}}})
    assert second['cached'] and second['histogram']==first['histogram']
    live=flow.plot(channel,{'revision':0,'plot':spec,'context':{'kind':'rows','table':str(table),'rows':[]}})
    assert live['histogram']['series'][0]['n']==0


def test_create_column_persists_choice_without_changing_measurements(setup):
    channel,table,*_=setup
    original=table.read_bytes()
    found=flow.dispatch(channel,{'action':'create_column','revision':0,'column':'my_flow_subtypes'})
    assert found['workspace']['output_column']=='my_flow_subtypes'
    assert dataio.load(table,columns=['my_flow_subtypes']).my_flow_subtypes.isna().all()
    assert table.read_bytes()==original
    with pytest.raises(ValueError,match='measurements'):
        flow.dispatch(channel,{'action':'create_column','revision':1,'column':'M0'})


def test_failed_label_write_restores_journal(setup,monkeypatch):
    channel,table,*_=setup
    gate={'id':'one','kind':'range','name':'One','colour':'#112233','x':{'column':'M0'},'bounds':[1,3]}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    request={'revision':1,'populations':['one'],'column':'phenotype_flow'}
    receipt=flow.label_plan(channel,request)[-1]['receipt']
    append=flow.labels.append
    def fail_after_write(*args,**kwargs):
        append(*args,**kwargs)
        raise OSError('Injected disk failure')
    monkeypatch.setattr(flow.labels,'append',fail_after_write)
    with pytest.raises(OSError,match='Injected'):
        flow.apply_labels(channel,{**request,'receipt':receipt})
    assert flow.load(channel)['revision']==1
    assert not flow.labels.sidecar_path(table).exists()


def test_repeated_segmentation_ids_keep_their_image_identity(setup):
    channel,table,*_=setup
    workspace=flow.load(channel)
    workspace['sources'][0]['selected_images'].append('other')
    workspace['gates']=[{'id':'high','kind':'range','name':'Other image','colour':'#112233',
                         'x':{'column':'M0'},'bounds':[50,150]}]
    frame,_,result=flow.evaluate(workspace)
    picked=frame.loc[result['high'][0]]
    assert picked['_flow_id'].tolist()==[1]
    assert picked['_flow_roi'].tolist()==['other']
    assert frame['_flow_id'].eq(1).sum()==2


def test_spatial_gate_lookup_is_limited_to_open_roi_even_with_reused_labels(setup):
    import shutil
    import numpy as np
    channel,table,root,_=setup
    shutil.copyfile(root/'outputs/rois/P11_roi_2.ome.tiff', root/'outputs/rois/other.ome.tiff')
    gate={'id':'roi_gate','kind':'range','name':'High','colour':'#112233','parent':'root',
          'x':{'column':'M0','scale':'linear'},'bounds':[90,110]}
    flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    selection=flow.select(channel,{'revision':1,'populations':['roi_gate']})
    assert selection['count']==1 and selection['by_image']=={'other':1}
    # Both images contain label 1. The selected cell in other must not leak
    # into P11_roi_2 merely because their local segmentation IDs match.
    image_labels=np.array([[0,1,2,3,4]])
    positions,lut=flow.tile(selection['token'],str(table),'P11_roi_2',image_labels)
    assert not lut[positions][...,3].any()
    positions,lut=flow.tile(selection['token'],str(table),'other',image_labels)
    assert lut[positions][0,1,3]>0 and not lut[positions][0,2:,3].any()


def test_gate_edits_reuse_plot_columns_and_persist_exact_memberships(setup,monkeypatch):
    import pyarrow.parquet as pq
    from spacemap_gui.studio import data_view
    channel,table,root,_=setup
    axis={'column':'M0','scale':'linear'}
    flow.plot(channel,{'revision':0,'plot':{'x':axis,'y':{'column':'M1'},'sample':100,'bins':5}})
    def forbidden(*args,**kwargs):
        raise AssertionError('Gate edits must reuse measurements already loaded by the plot')
    monkeypatch.setattr(data_view._Table,'_whole',forbidden)
    gate={'id':'fast','kind':'range','name':'Fast','colour':'#112233','parent':'root','x':axis,'bounds':[2,3]}
    result=flow.save(channel,{'revision':0,'changes':{'gates':[gate]}})
    saved=pq.read_table(result['workspace']['membership_file']).to_pandas()
    assert saved['_flow_table'].astype(str).eq(str(table)).all()
    assert saved['_flow_row'].tolist()==[0,1,2,3]
    assert saved['_flow_id'].tolist()==[1,2,3,4]
    assert saved['fast'].tolist()==[False,True,True,False]
    renamed=flow.save(channel,{'revision':1,'changes':{'gates':[{**gate,'name':'Renamed'}]}})
    assert renamed['workspace']['membership_file']==result['workspace']['membership_file']
    deleted=flow.save(channel,{'revision':2,'changes':{'gates':[]}})
    assert pq.read_table(deleted['workspace']['membership_file']).num_rows==4
