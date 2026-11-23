"""Result reviews use frozen image/ID membership and staged, guarded decisions."""
import json
from pathlib import Path

import pandas as pd
import pytest
from spacemap import dataio, labels
from spacemap_gui.studio import jobs, review_results as rr, review_store as store
from test_studio_chat import _job


@pytest.fixture
def result(tmp_path):
    path=tmp_path/'cells.parquet'
    pd.DataFrame({'batch_id':['a']*4+['b']*4,'cell_id':list(range(1,5))*2,
        'label':list(range(1,5))*2,'clusters':['C0','C0','C1','C1']*2,
        'phenotype':['Old']*8,'x_centroid':range(8),'y_centroid':range(8)}).to_parquet(path,index=False)
    rows=[{'roi':roi,'image_path':str(tmp_path/(roi+'.tif')),'table_path':str(path)} for roi in ('a','b')]
    result={'table':str(path),'column':'clusters','phenotype_column':'phenotype','images':['a','b'],
        'clusters':[{'name':'C0'},{'name':'C1'}],
        'profile':{'version':2,'rows':['C0','C1'],'columns':['M'],'values':[[1],[-1]]}}
    return tmp_path,path,rows,result


@pytest.mark.parametrize('kind',list(rr.KINDS))
def test_all_cluster_types_keep_ids_images_and_full_context(result,kind):
    root,path,rows,res=result; original=path.read_bytes()
    job=_job(root,kind,res)
    items=rr.flag(root,{'job_id':job,'groups':['C0','C1']},rows)['items']
    assert len(items)==4
    item=items[0]; context=rr.context(root,item)
    assert context['profile']['rows']==['C0','C1']
    assert context['groups'][0]['cells']==2
    assert context['groups'][0]['label']=='Old'
    assert path.read_bytes()==original
    item['decision']={'action':'relabel','label':'New'}
    rr.apply(root,item)
    current=dataio.load(path,build_fast_copy=False)
    assert current.phenotype.tolist()==['New','New','Old','Old','Old','Old','Old','Old']
    assert current.clusters.tolist()==['C0','C0','C1','C1']*2
    assert path.read_bytes()==original
    with pytest.raises(ValueError,match='labels changed'):rr.apply(root,item)


def test_newer_run_and_changed_membership_are_rejected(result):
    root,path,rows,res=result;job=_job(root,'subphenotype',res)
    item=rr.flag(root,{'job_id':job,'groups':['C0']},rows)['items'][0]
    labels.append(path,[{'row':0,'column':'clusters','value':'C1'}])
    item['decision']={'action':'relabel','label':'X'}
    with pytest.raises(ValueError,match='membership changed'):rr.apply(root,item)
    _job(root,'subphenotype',res)
    with pytest.raises(ValueError,match='newer run'):rr.flag(root,{'job_id':job,'groups':['C0']},rows)


def test_nomination_is_pending_owner_approval(result):
    root,path,rows,res=result;job=_job(root,'leiden',res)
    item=rr.flag(root,{'job_id':job,'groups':['C0']},rows)['items'][0]
    nomination=rr.nominate(root,item,['C1'],'reviewer-id')['items'][0]
    assert nomination['state']=='requested'
    assert nomination['target']['image']==item['target']['image']
    with pytest.raises(ValueError):rr.nominate(root,item,['unknown'],'reviewer-id')


@pytest.mark.parametrize('mode,kind,column',[('interior','gland_type','gland_clone'),('exterior','niche_type','gland_bed_niche'),('gland','gland','gland_clone'),('bed','niche','gland_bed_niche')])
def test_gland_and_niche_reviews(result,mode,kind,column):
    root,path,rows,res=result
    frame=pd.read_parquet(path);frame['gland_clone']=['G1','G1','G2','G2']*2;frame['gland_bed_niche']=['N1','N1','N2','N2']*2;frame.to_parquet(path,index=False)
    res.update(values={'output_col':'gland_clone','niche_col':'gland_bed_niche'},column='gland_clone',columns=['gland_clone','gland_bed_niche'],
        profiles={'interior':{'rows':['G1','G2']},'exterior':{'rows':['N1','N2']}})
    job=_job(root,'gland_types',res);folder=jobs.directory(root,job)
    pd.DataFrame({'structure':['s0','s1'],'local_id':[1,2],'gland_type':['G1','G2'],'niche':['N1','N2']}).to_parquet(folder/'gland_structures.parquet')
    pd.DataFrame({'cell_id':list('1234')*2,'image':['a']*4+['b']*4,'gland':[0,0,1,1]*2,'bed':[0,0,1,1]*2}).to_parquet(folder/'gland_cell_links.parquet')
    group={'interior':'G1','exterior':'N1','gland':'s0','bed':'s0'}[mode]
    item=rr.flag(root,{'job_id':job,'mode':mode,'groups':[group]},rows)['items'][0]
    assert item['kind']==kind and item['target']['count']==2
    context=rr.context(root,item)
    assert context['individual']==(mode in ('gland','bed'))
    assert len(context['profile']['rows'])==2
    assert context['groups'][0]['label']==('N1' if mode in ('bed','exterior') else 'G1')
    item['decision']={'action':'relabel','label':'Reviewed'};rr.apply(root,item)
    frame=dataio.load(path,build_fast_copy=False)
    assert frame[column].tolist()[:2]==['Reviewed']*2
    assert 'Reviewed' not in frame[column].tolist()[2:]


def test_manual_cells_match_image_and_keep_source(result,monkeypatch):
    from spacemap_gui.studio import manual_reference
    root,path,rows,res=result
    monkeypatch.setattr(manual_reference,'context',lambda c:(path,{'roi_column':'batch_id','label_column':'label'},{}))
    monkeypatch.setattr(manual_reference,'_positions',lambda *a:[{'roi':'b','ids':[1,3]}])
    found=rr.flag_cells(root,'fixture',{'table':str(path),'column':'phenotype','selection':{}},rows)
    assert len(found['items'])==2
    item=found['items'][0];item['decision']={'action':'relabel','label':'Checked'};rr.apply(root,item)
    assert dataio.load(path,build_fast_copy=False).phenotype.tolist()==['Old']*4+['Checked','Old','Old','Old']


def test_separate_tables_match_local_ids_without_merging(result):
    root,path,rows,res=result
    frame=pd.read_parquet(path)
    for row in rows:
        table=root/(row['roi']+'.parquet')
        frame.loc[frame.batch_id.eq(row['roi'])].drop(columns='batch_id').to_parquet(table,index=False)
        row['table_path']=str(table)
    manifest=root/'images.spatial-tables.json'
    manifest.write_text(json.dumps({'format':'spacemap-image-tables-v1','tables':[
        {'image':r['roi'],'path':r['table_path'],'anchor':'cell_id'} for r in rows]}))
    res['table']=str(manifest);job=_job(root,'subphenotype',res)
    found=rr.flag(root,{'job_id':job,'groups':['C0']},rows)['items']
    assert len(found)==2 and found[1]['target']['roi_column'] is None
    found[1]['decision']={'action':'relabel','label':'New'};rr.apply(root,found[1])
    assert dataio.load(rows[0]['table_path'],build_fast_copy=False).phenotype.tolist()==['Old']*4
    assert dataio.load(rows[1]['table_path'],build_fast_copy=False).phenotype.tolist()==['New','New','Old','Old']


def test_tls_local_ids_and_rejection_use_guarded_job(result,monkeypatch):
    from spacemap_gui.studio import tls_review
    from spacemap_gui.viewer.review_api import apply_decision
    root,path,rows,res=result
    frame=pd.read_parquet(path);frame['TLS_id']=1;frame['tissue_roi']=[1,1,2,2]*2;frame.to_parquet(path,index=False)
    structures=[{'image':r,'region':str(region),'id':1,'key':json.dumps([r,str(region),1])} for r in ('a','b') for region in (1,2)]
    res.update(column='TLS',columns=['TLS','TLS_id'],structures=structures,settings={'roi_col':'tissue_roi'})
    job=_job(root,'tls_candidates',res)
    found=rr.flag(root,{'job_id':job,'groups':[s['key'] for s in structures]},rows)['items']
    assert len(found)==4 and all(i['target']['count']==2 for i in found)
    item=found[0]
    monkeypatch.setattr(tls_review,'context',lambda *args:{'features':[{'properties':{'key':s['key']}} for s in structures[:2]]})
    submitted=[]
    monkeypatch.setattr(tls_review,'prepare',lambda *args:submitted.append(args[-1]) or {})
    monkeypatch.setattr(jobs,'start',lambda *args,**kwargs:{'id':'tls-review-job'})
    item=store.decide(root,item['id'],version=item['version'],action='reject')
    applying=apply_decision(root,item['id'],item['version'])
    assert applying['state']=='applying' and applying['applied']['job_id']=='tls-review-job'
    assert [f['properties']['key'] for f in submitted[0]['features']]==[structures[1]['key']]
    monkeypatch.setattr(jobs,'status',lambda *args:{'status':'done'})
    rr.reconcile(root)
    assert store.get(root,item['id'])['state']=='applied'
    assert dataio.load(path,build_fast_copy=False).TLS_id.tolist()==[1]*8  # worker was stubbed; no partial clearing


def test_shared_resources_prepare_only_the_opened_result(result,monkeypatch):
    from spacemap_gui.viewer import review_api, registry
    from test_viewer import _write_channel_pages
    root,path,rows,res=result;job=_job(root,'subphenotype',res)
    for row in rows:
        _write_channel_pages(Path(row['image_path']),channels=1,height=20,width=20)
    items=rr.flag(root,{'job_id':job,'groups':['C0']},rows)['items']
    scope=json.dumps({'items':[i['id'] for i in items]})
    opened=[]
    def prepare(root,binding):
        target=json.loads(binding);opened.append(target['image'])
        return {'token':registry.register_object('cells',image=target['image'],tables={'review':str(path)})}
    monkeypatch.setattr(rr,'_layer',prepare)
    assert not review_api._resources(str(root),'fixture',scope)['cells']
    assert opened==[]
    layer=rr.layer(root,items[0])
    resources=review_api._resources(str(root),'fixture',scope)
    assert list(resources['cells'].values())==[layer['token']]
    assert opened==[items[0]['target']['image']]
    rr.layer(root,items[0])
    assert len(opened)==1
