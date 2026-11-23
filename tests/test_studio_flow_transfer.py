"""Branch transfers are atomic, exact, reversible and preserve gate transforms."""
import copy
import numpy as np
import pytest

from spacemap_gui.studio import flow_gating as flow
from test_studio_commands import setup  # noqa: F401
from test_studio_flow_split import study, split  # noqa: F401


def transfer(channel, ids, target=None, mode='copy', **extra):
    return flow.dispatch(channel,{'action':'transfer','revision':flow.load(channel)['revision'],
                                  'populations':ids,'target':target,'mode':mode,**extra})


def branches(channel):
    found=split(channel,'assigned')
    gates={g['name']:g for g in found['workspace']['gates']}
    split(channel,'timepoint',gates['assigned_Epithelial']['id'])
    return gates['assigned_Epithelial']['id'],gates['assigned_CD8']['id']


def test_copy_selected_siblings_recomputes_against_new_parent_and_undo(study):
    channel,table,_,frame=study
    original=table.read_bytes();a,b=branches(channel)
    before=copy.deepcopy(flow.load(channel))
    children=[g for g in before['gates'] if g.get('parent')==a]
    copied=transfer(channel,[g['id'] for g in children],b)
    assert copied['affected']==2 and copied['workspace']['revision']==before['revision']+1
    _,_,masks=flow.evaluate(flow.load(channel))
    for identifier in copied['selected']:
        g=next(g for g in copied['workspace']['gates'] if g['id']==identifier)
        source=next(c for c in children if c['name']==g['name'])
        assert g['colour']==source['colour'] and g['rule']==source['rule']
        assert masks[identifier][0].tolist()==(frame.assigned.eq('CD8')&frame.timepoint.eq(g['split']['value'])).tolist()
        assert int(masks[source['id']][0].sum())==20
    flow.save(channel,{'action':'undo','revision':copied['workspace']['revision']})
    assert flow.load(channel)['gates']==before['gates']
    assert table.read_bytes()==original


def test_branch_duplicate_deduplicates_selected_descendants_and_remaps_boolean(study):
    channel,_,_,_=study;a,b=branches(channel)
    w=flow.load(channel);children=[g for g in w['gates'] if g.get('parent')==a]
    both={'id':'both','kind':'boolean','parent':a,'name':'Both','colour':'#123456','op':'or','refs':[g['id'] for g in children]}
    flow.save(channel,{'revision':w['revision'],'changes':{'gates':[*w['gates'],both]}})
    result=transfer(channel,[a,children[0]['id']])
    assert result['affected']==4 and len(result['selected'])==1
    clone=result['selected'][0];gates=result['workspace']['gates'];root=next(g for g in gates if g['id']==clone)
    assert root['name']=='assigned_Epithelial copy' and root['parent']=='root'
    copied=[g for g in gates if g.get('parent')==clone]
    boolean=next(g for g in copied if g['kind']=='boolean')
    assert set(boolean['refs'])=={g['id'] for g in copied if g['kind']=='filter'}
    _,_,masks=flow.evaluate(flow.load(channel));assert masks[boolean['id']][0].sum()==40


def test_move_keeps_ids_descendants_and_fitted_coordinates(study):
    channel,_,_,_=study;a,b=branches(channel);w=flow.load(channel)
    gate={'id':'high','kind':'range','parent':a,'name':'High','colour':'#abcdef','x':{'column':'M0','normalize':'zscore'},'bounds':[0,2]}
    flow.save(channel,{'revision':w['revision'],'changes':{'gates':[*w['gates'],gate]}})
    before=copy.deepcopy(flow.load(channel));fitted=next(g for g in before['gates'] if g['id']=='high')['x']
    moved=transfer(channel,['high'],b,'move');current=next(g for g in moved['workspace']['gates'] if g['id']=='high')
    assert current['x']==fitted and current['parent']==b
    _,_,masks=flow.evaluate(flow.load(channel));assert masks['high'][0].sum()==20
    assert transfer(channel,['high'],b,'move')['affected']==0
    flow.save(channel,{'action':'undo','revision':flow.load(channel)['revision']})
    assert flow.load(channel)['gates']==before['gates']


def test_invalid_transfers_never_change_workspace(study):
    channel,_,_,_=study;a,b=branches(channel);w=flow.load(channel)
    child=next(g['id'] for g in w['gates'] if g.get('parent')==a)
    for ids,target,mode in [([a],child,'move'),(['root'],b,'copy'),(['gone'],b,'copy'),([a],'gone','move')]:
        with pytest.raises(ValueError):transfer(channel,ids,target,mode)
        assert flow.load(channel)['gates']==w['gates'] and flow.load(channel)['revision']==w['revision']
    with pytest.raises(ValueError,match='changed'):transfer(channel,[a],b,revision=-1)


def test_cross_branch_boolean_requires_references_and_cycles_rejected(study):
    channel,_,_,_=study;a,b=branches(channel);w=flow.load(channel)
    gate={'id':'ref','kind':'boolean','parent':'root','name':'Reference','colour':'#abcdef','op':'or','refs':[a]}
    flow.save(channel,{'revision':w['revision'],'changes':{'gates':[*w['gates'],gate]}})
    with pytest.raises(ValueError,match='referenced'):transfer(channel,['ref'],b)
    # Target depends on the moved source, even without being its tree child.
    with pytest.raises(ValueError,match='cycle'):transfer(channel,[a],'ref','move')
    result=transfer(channel,[a,'ref'],b)
    _,_,masks=flow.evaluate(flow.load(channel));assert all(masks[i][0].sum()==0 for i in result['selected'])


def test_transfer_limit_is_checked_before_saving(study):
    channel,_,_,_=study
    gates=[{'id':f'g{i}','kind':'filter','parent':'root','name':str(i),'colour':'#112233','rule':{'column':'M0','op':'>=','value':0}} for i in range(129)]
    saved=flow.save(channel,{'revision':0,'changes':{'gates':gates}})
    with pytest.raises(ValueError,match='256'):transfer(channel,[g['id'] for g in gates])
    assert flow.load(channel)['revision']==saved['workspace']['revision']
