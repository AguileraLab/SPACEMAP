"""Split-by branches use exact parent membership, content types and shared colours."""
import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, phenotype_presets
from spacemap_gui.studio import flow_gating as flow
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def study(setup):
    channel, table, root, _ = setup
    n = 60
    frame = pd.DataFrame({'batch_id':['P11_roi_2']*n,'label':np.arange(n)+1,
                          'M0':np.arange(n,dtype=float),'M1':np.arange(n,dtype=float)*2,
                          'assigned':['Epithelial']*40+['CD8']*20,
                          'timepoint':['pre','post']*(n//2), 'binary':np.arange(n)%2,
                          'twenty_four':np.arange(n)%24,'twenty_five':np.arange(n)%25,
                          'text_many':['sample_'+str(i%30) for i in range(n)]})
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    return channel,table,root,frame


def request_split(channel, column, parent='root', **extra):
    return flow.dispatch(channel, {'action':'split','revision':flow.load(channel)['revision'],
                                   'column':column,'parent':parent,**extra})


def split(channel, column, parent='root', **extra):
    """Confirm all categories, unless the test supplies an explicit selection."""
    result = request_split(channel, column, parent, **extra)
    if result['kind'] == 'categorical' and 'workspace' not in result:
        return request_split(channel, column, parent,
                             **{**extra, 'categories':[c['value'] for c in result['categories']],
                                'dataset_revision':result['dataset_revision'], 'revision':result['revision']})
    return result


def test_category_preview_and_selected_subgroups_preserve_parent(study, monkeypatch):
    channel,table,_,frame=study
    before=table.read_bytes();workspace=flow.load(channel)
    with monkeypatch.context() as patch:
        patch.setattr(phenotype_presets, 'ensure', lambda *_:pytest.fail('Preview must not write colours'))
        description=request_split(channel,'assigned')
    assert 'workspace' not in description
    assert description['categories']==[{'label':'CD8','value':'CD8','count':20},
                                       {'label':'Epithelial','value':'Epithelial','count':40}]
    assert flow.load(channel)['revision']==workspace['revision'] and flow.load(channel)['gates']==[]
    result=request_split(channel,'assigned',categories=['Epithelial'],
                         dataset_revision=description['dataset_revision'], colours={'Epithelial':'#123456'})
    assert result['added']==1
    gate=result['workspace']['gates'][0]
    assert gate['name']=='assigned_Epithelial' and gate['colour']=='#123456'
    _,_,masks=flow.evaluate(flow.load(channel))
    assert masks['root'][0].sum()==60
    assert masks[gate['id']][0].tolist()==frame.assigned.eq('Epithelial').tolist()
    nested=request_split(channel,'binary',gate['id'])
    assert {float(c['value']):c['count'] for c in nested['categories']}=={0:20,1:20}
    positive=next(c for c in nested['categories'] if float(c['value'])==1)
    child=request_split(channel,'binary',gate['id'],categories=[positive['value']],dataset_revision=nested['dataset_revision'])
    assert child['added']==1 and child['workspace']['gates'][-1]['name']=='binary_'+positive['label']
    _,_,masks=flow.evaluate(flow.load(channel))
    assert masks[child['workspace']['gates'][-1]['id']][0].tolist()==(frame.assigned.eq('Epithelial')&frame.binary.eq(1)).tolist()
    assert table.read_bytes()==before


def test_category_selection_validation_and_stale_previews(study):
    channel,table,_,frame=study
    description=request_split(channel,'assigned')
    for selected in ([], 'CD8', [True], [20], [{}], ['gone']):
        with pytest.raises(ValueError,match='category|Categories'):
            request_split(channel,'assigned',categories=selected,dataset_revision=description['dataset_revision'])
    with pytest.raises(ValueError,match='Measurements changed'):
        request_split(channel,'assigned',categories=['CD8'])
    assert flow.load(channel)['gates']==[] and flow.load(channel)['revision']==description['revision']
    request_split(channel,'assigned',categories=['CD8'],dataset_revision=description['dataset_revision'])
    with pytest.raises(ValueError,match='populations changed'):
        request_split(channel,'assigned',categories=['Epithelial'],revision=description['revision'],dataset_revision=description['dataset_revision'])
    frame['assigned']='Changed'
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    with pytest.raises(ValueError,match='Measurements changed'):
        request_split(channel,'assigned',categories=['Epithelial'],dataset_revision=description['dataset_revision'])
    assert len(flow.load(channel)['gates'])==1


def test_missing_category_can_be_selected_independently(study):
    channel,table,_,frame=study
    frame['assigned']=['Missing','Treg',None,' ']*15
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    description=request_split(channel,'assigned')
    assert description['categories']==[{'label':'Missing','value':'Missing','count':15},
                                       {'label':'Treg','value':'Treg','count':15},
                                       {'label':'Missing (empty)','value':None,'count':30}]
    result=request_split(channel,'assigned',categories=[None],dataset_revision=description['dataset_revision'])
    assert [g['name'] for g in result['workspace']['gates']]==['assigned_Missing (empty)']
    _,_,masks=flow.evaluate(flow.load(channel))
    assert masks[result['workspace']['gates'][0]['id']][0].sum()==30
    # Adding a different selection does not remove the previous branch.
    result=request_split(channel,'assigned',categories=['Missing'],dataset_revision=description['dataset_revision'])
    assert [g['name'] for g in result['workspace']['gates']]==['assigned_Missing (empty)','assigned_Missing']


def test_nested_category_splits_shared_colours_and_idempotent_undo(study):
    channel,table,_,frame=study
    original=table.read_bytes()
    phenotype_presets.update({'Epithelial':'#123456','CD8':'#987654'})
    first=split(channel,'assigned')
    assert first['kind']=='categorical' and first['added']==2
    gates={g['name']:g for g in first['workspace']['gates']}
    assert gates['assigned_Epithelial']['colour']=='#123456' and gates['assigned_CD8']['colour']=='#987654'
    parent=gates['assigned_Epithelial']['id']
    second=split(channel,'timepoint',parent)
    children=[g for g in second['workspace']['gates'] if g.get('parent')==parent]
    assert {g['name'] for g in children}=={'timepoint_pre','timepoint_post'}
    current,_,masks=flow.evaluate(flow.load(channel))
    for child in children:
        expected=frame.assigned.eq('Epithelial') & frame.timepoint.eq(child['split']['value'])
        assert masks[child['id']][0].tolist()==expected.tolist()
        selection=flow.select(channel,{'revision':2,'populations':[child['id']]})
        assert flow.token_rows(selection['token'],channel,str(table)).tolist()==np.flatnonzero(expected).tolist()
        assert selection['colour']==child['colour']
    again=split(channel,'timepoint',parent)
    assert again['added']==0 and again['workspace']['revision']==2
    assert table.read_bytes()==original
    flow.save(channel,{'action':'undo','revision':2})
    assert len(flow.load(channel)['gates'])==2
    flow.save(channel,{'action':'redo','revision':3})
    assert len(flow.load(channel)['gates'])==4


def test_numeric_cardinality_boundary_is_analysis_wide(study):
    channel,_,_,_=study
    assert split(channel,'twenty_four')['added']==24
    description=split(channel,'twenty_five')
    assert description['kind']=='numeric' and 'workspace' not in description
    assert description['distinct']==25
    parent=flow.load(channel)['gates'][0]['id']
    # A tiny child does not turn a continuous marker into a categorical axis.
    assert split(channel,'M0',parent)['kind']=='numeric'
    assert split(channel,'text_many')['added']==30  # nonnumeric text remains categorical


def test_threshold_range_boundaries_missing_and_stale_requests(study):
    channel,table,_,frame=study
    frame.loc[0,'M0']=np.nan
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    description=split(channel,'M0')
    assert description['missing']==1 and description['minimum']==1
    result=split(channel,'M0',condition={'mode':'threshold','value':20},dataset_revision=description['dataset_revision'])
    _,_,masks=flow.evaluate(flow.load(channel))
    gates={g['name']:g for g in result['workspace']['gates']}
    assert masks[gates['M0 < 20']['id']][0].sum()==19
    assert masks[gates['M0 ≥ 20']['id']][0].sum()==40
    assert masks[gates['M0: missing']['id']][0].sum()==1
    assert np.stack([masks[g['id']][0] for g in gates.values()]).sum(axis=0).tolist()==[1]*60
    ranged=split(channel,'M0',condition={'mode':'range','low':10,'high':20})
    _,_,masks=flow.evaluate(flow.load(channel))
    inside=next(g for g in ranged['workspace']['gates'] if g['name']=='M0: 10–20')
    assert masks[inside['id']][0].sum()==11  # inclusive endpoints
    with pytest.raises(ValueError,match='minimum'):
        split(channel,'M0',condition={'mode':'range','low':20,'high':10})
    with pytest.raises(ValueError,match='finite'):
        split(channel,'M0',condition={'mode':'threshold','value':''})
    with pytest.raises(ValueError,match='changed'):
        split(channel,'M0',condition={'mode':'threshold','value':5},dataset_revision='old')


def test_missing_labels_are_separate_from_literal_missing_and_draft_colours(study):
    channel,table,_,frame=study
    frame['assigned']=['Missing','Treg',None,' ']*15
    frame.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    result=split(channel,'assigned',colours={'Treg':'#abcdef'})
    groups={g['name']:g for g in result['workspace']['gates']}
    assert set(groups)=={'assigned_Missing','assigned_Treg','assigned_Missing (empty)'}
    assert groups['assigned_Treg']['colour']=='#abcdef'
    _,_,masks=flow.evaluate(flow.load(channel))
    assert masks[groups['assigned_Missing']['id']][0].sum()==15
    assert masks[groups['assigned_Missing (empty)']['id']][0].sum()==30
    assert masks[groups['assigned_Treg']['id']][0].sum()==15
    # The selection-local colour does not overwrite the user's shared palette.
    assert phenotype_presets.ensure(['Treg'])['Treg']!='#abcdef'


def test_split_rejects_missing_parent_and_unbounded_categories(study):
    channel,table,_,frame=study
    with pytest.raises(ValueError,match='parent'):
        split(channel,'assigned','not_a_population')
    large=pd.concat([frame]*5,ignore_index=True)
    large['label']=np.arange(len(large))+1
    large['assigned']=['distinct_'+str(i) for i in range(len(large))]
    large.to_csv(table,index=False);dataio.ensure_fast_copy(table,force=True)
    with pytest.raises(ValueError,match='Too many categories'):
        split(channel,'assigned')
    assert flow.load(channel)['gates']==[]
