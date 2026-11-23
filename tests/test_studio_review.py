"""Review contracts against disposable projects; no research files are touched."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from spacemap_gui.studio import annotation_components as components, annotation_edits as edits, review_store as store


@pytest.fixture
def source(tmp_path):
    from shapely.geometry import Polygon, MultiPolygon, mapping
    first = Polygon([(0,0),(10,0),(10,10),(0,10)], holes=[[(2,2),(4,2),(4,4),(2,4)]])
    second = Polygon([(20,0),(30,0),(30,10),(20,10)])
    path = tmp_path / 'original.geojson'
    path.write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature',
        'geometry':mapping(MultiPolygon([first,second])), 'properties':{'classification':{'name':'tumor','color':[255,0,0]}}}]}))
    return path


def flag(root, source, index=0):
    value = components.index(source)
    row = value['rows'][index]
    return store.add(root, kind='annotation', image='image.tif', title='Tumor '+row['id'],
        target={'source':str(source),'image':str(root/'image.tif'),'ids':[row['id']],
                'revision':value['revision'],'fingerprints':{row['id']:row['fingerprint']}})


def test_share_images_includes_all_pages_and_deleting_link_keeps_decisions(tmp_path, source, monkeypatch):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import registry, review_links
    from spacemap_gui.viewer.api import app
    from spacemap_gui.studio import review_images
    from test_viewer import _write_channel_pages
    rows=[]
    for roi in ('a', 'b'):
        path=tmp_path/(roi+'.tif'); _write_channel_pages(path,channels=2,height=32,width=32)
        rows.append({'roi':roi,'image':path.name,'image_path':str(path)})
    monkeypatch.setattr(review_images,'plan',lambda channel:{'rows':rows})
    monkeypatch.setattr(review_links.registry,'_state_folder',lambda:tmp_path/'state')
    registry.publish('review-share-fixture',{'layers':[]},project={'root':str(tmp_path)})
    registry.session_remember_images('review-share-fixture',[{'path':r['image_path'],'name':r['image']} for r in rows])
    items=store.add_many(tmp_path,[dict(kind='annotation',image=rows[i%2]['image'],title='tumor',
        target={'source':str(source),'image':rows[i%2]['image_path'],'ids':[str(i)],'revision':'fixture'},area=i+1)
        for i in range(225)])
    same_name=store.add(tmp_path,kind='annotation',image='a.tif',title='Other source',
        target={'source':str(source),'image':str(tmp_path/'elsewhere/a.tif'),'ids':['outside'],'revision':'fixture'})
    cell=store.add(tmp_path,kind='cells',image='a.tif',title='Cells',
        target={'source':str(source),'image':rows[0]['image_path'],'ids':['cell'],'revision':'fixture'})
    nomination=store.add(tmp_path,kind='annotation',image='a.tif',title='Nomination',requested_by='reviewer',
        target={'source':str(source),'image':rows[0]['image_path'],'ids':['nomination'],'revision':'fixture'})
    route='/studio/review-share-fixture/review'
    with TestClient(app) as client:
        found=client.post(route,json={'action':'share_selection','images':['a','b'],'kind':'annotation'})
        assert found.status_code==200,found.text
        ids=found.json()['ids']; assert len(ids)==225
        assert not {same_name['id'],cell['id'],nomination['id']} & set(ids)
        assert client.post(route,json={'action':'share_selection','images':['a','b']}).json()['total']==226
        assert client.post(route,json={'action':'share_selection','images':['not-listed']}).status_code==400
        created=client.post(route,json={'action':'share','name':'All image flags','items':ids[:1],'images':['a','b']}).json()
        base=created['url'].replace('http://testserver','').rstrip('/')
        answered=client.post(base+'/decision',json={'id':ids[0],'version':1,'decision':'relabel','label':'normal'})
        assert answered.status_code==200,answered.text
        updated=client.post(route,json={'action':'update_link','id':created['id'],'name':'All image flags',
            'access':'internal','items':ids,'images':['a','b']})
        assert updated.status_code==200,updated.text
        assert updated.json()['url']==created['url'] and updated.json()['total']==225
        assert client.get(base+'/queue').json()['total']==225
        public_images=client.get(base+'/images').json()['images']
        token=next(r['roi'] for r in public_images if r['image']=='a.tif')
        queue=client.get(base+'/queue',params={'image':token,'sort':'size_down'}).json()
        assert queue['total']==113 and len(queue['items'])==100
        assert queue['items'][0]['id']==items[-1]['id']
        assert len(client.get(base+'/selection',params={'image':token}).json()['ids'])==113
        assert client.get(base+'/queue',params={'image':''}).json()['total']==0
        assert client.get(base+'/queue',params={'image':'not-shared'}).status_code==403
        assert client.get(route,params={'image':registry.register(rows[1]['image_path'])}).json()['total']==112
        assert client.get(route,params={'image':''}).json()['total']==0
        another=client.post(route,json={'action':'share','name':'New all flags','items':ids,'images':['a','b']}).json()
        second_base=another['url'].replace('http://testserver','').rstrip('/')
        assert client.get(second_base+'/queue').json()['total']==225
        store.destination(tmp_path,created['id'])
        assert client.post(route,json={'action':'delete_link','id':created['id']}).status_code==200
        assert client.post(route,json={'action':'delete_link','id':created['id']}).status_code==200  # retry safe
        assert client.get(base+'/queue').status_code==403
        assert store.destination(tmp_path)==''
        assert [g['id'] for g in store.grants(tmp_path)]==[another['id']]
        assert store.get(tmp_path,ids[0])['decision']==answered.json()['decision']
        assert client.get(second_base+'/queue').json()['total']==225
        store.revoke(tmp_path,another['id'])
        store.delete_grant(tmp_path,another['id'])
        assert store.grants(tmp_path)==[] and store.list_items(tmp_path)['total']==228


def test_pending_geojson_write_preserves_original_until_explicit_commit(tmp_path, source):
    from spacemap_gui.studio import annotation_pending
    original = source.read_bytes()
    indexed = components.index(source)
    changed = components.edit(tmp_path, tmp_path/'image.tif', source,
        ids=[indexed['rows'][0]['id']], expected=indexed['revision'], action='rename', name='normal')
    copy = Path(changed['path'])
    identity = components.display_identity(copy)
    assert identity['name'] == source.name and identity['annotation_origin'] == 'copy'
    rows = annotation_pending.pending(tmp_path)
    assert all(isinstance(value, str) for value in rows[0]['stamp'])  # lossless browser round trip
    assert len(rows) == 1 and rows[0]['name'] == source.name
    assert source.read_bytes() == original
    messages = []
    result = annotation_pending.write({'root':str(tmp_path), 'items':rows}, tmp_path,
        lambda *args:messages.append(args))
    assert result['files'] == [str(source)] and result['count'] == 1
    assert annotation_pending.pending(tmp_path) == []
    assert source.name == 'original.geojson'
    assert sorted(r['area'] for r in components.index(source)['rows']) == [96, 100]
    assert {r['name'] for r in components.index(source)['rows']} == {'normal', 'tumor'}
    assert len(edits.history(source)) >= 2
    current = components.index(copy)
    components.edit(tmp_path, tmp_path/'image.tif', copy,
        ids=[current['rows'][0]['id']], expected=current['revision'], action='rename', name='stroma')
    assert len(annotation_pending.pending(tmp_path)) == 1
    annotation_pending.write({'root':str(tmp_path), 'items':annotation_pending.pending(tmp_path)}, tmp_path, lambda *args:None)
    assert annotation_pending.pending(tmp_path) == []


def test_pending_geojson_write_rejects_external_changes_and_stale_choices(tmp_path, source):
    from spacemap_gui.studio import annotation_pending
    indexed = components.index(source)
    result = components.edit(tmp_path, tmp_path/'image.tif', source,
        ids=[indexed['rows'][0]['id']], expected=indexed['revision'], action='rename', name='normal')
    stale = annotation_pending.pending(tmp_path)
    copy = Path(result['path'])
    current = components.index(copy)
    components.edit(tmp_path, tmp_path/'image.tif', copy,
        ids=[current['rows'][0]['id']], expected=current['revision'], action='rename', name='stroma')
    with pytest.raises(ValueError, match='changed'):
        annotation_pending.write({'root':str(tmp_path), 'items':stale}, tmp_path, lambda *args:None)
    source.write_text(source.read_text()+'\n')
    outside = source.read_bytes()
    with pytest.raises(ValueError, match='changed'):
        annotation_pending.write({'root':str(tmp_path), 'items':annotation_pending.pending(tmp_path)}, tmp_path, lambda *args:None)
    assert source.read_bytes() == outside
    assert len(annotation_pending.pending(tmp_path)) == 1


def test_review_polygon_sizes_sort_across_pages_without_changing_decisions(tmp_path, source):
    one, two = flag(tmp_path, source, 0), flag(tmp_path, source, 1)
    from spacemap_gui.viewer.review_api import ensure_polygon_sizes
    ensure_polygon_sizes(tmp_path)
    assert store.list_items(tmp_path, sort='size_down', limit=1)['items'][0]['id'] == two['id']
    assert store.list_items(tmp_path, sort='size_down', offset=1, limit=1)['items'][0]['id'] == one['id']
    assert store.list_items(tmp_path, sort='size_up', limit=1)['items'][0]['id'] == one['id']
    assert store.list_items(tmp_path, ids=[one['id']], sort='size_down')['total'] == 1
    assert store.get(tmp_path, one['id'])['version'] == one['version']
    assert store.missing_polygon_sizes(tmp_path) == []
    with pytest.raises(ValueError, match='sort'):
        store.list_items(tmp_path, sort='invalid')


def test_components_preserve_holes_and_linked_original(tmp_path, source):
    original = source.read_bytes()
    result = components.page(source, limit=1)
    assert result['total'] == 2 and len(result['rows']) == 1
    assert sorted(r['area'] for r in components.index(source)['rows']) == [96,100]
    ids = [result['rows'][0]['id']]
    changed = components.edit(tmp_path,tmp_path/'image.tif',source,ids=ids,expected=result['revision'],action='rename',name='normal')
    copy = Path(changed['path'])
    assert source.read_bytes() == original
    assert {c['name'] for c in components.page(copy)['classes']} == {'normal','tumor'}
    assert len(edits.read(copy)[0]) == 2
    assert sorted(r['area'] for r in components.index(copy)['rows']) == [96,100]
    assert any(len(f['geometry']['coordinates'])==2 for f in edits.read(copy)[0])
    assert len(edits.history(copy)) >= 2


def test_exact_hits_paging_and_selection_tiles_leave_geometry_untouched(source):
    from types import SimpleNamespace
    import numpy as np
    original = source.read_bytes()
    first = components.hit(source, 1, 1)['feature']
    assert first is not None
    assert components.hit(source, 3, 3)['feature'] is None  # polygon hole
    assert components.hit(source, 1, 1, hidden=['tumor'])['feature'] is None
    assert components.hit(source, 15, 5)['feature'] is None  # between components
    second = components.hit(source, 25, 5)['feature']
    assert first['properties']['id'] != second['properties']['id']
    found = components.page(source, limit=1, locate=first['properties']['id'])
    assert found['rows'][0]['id'] == first['properties']['id']
    chosen = components.selection(source, [first['properties']['id']])
    assert 'features' not in chosen and chosen['count'] == 1
    image = SimpleNamespace(width=64, height=64)
    tile = components.selection_tile(source, chosen['selection'], image, 0, 0, 0)
    assert np.count_nonzero(tile[:, :, 3]) > 0
    assert tile[5, 25, 3] == 0  # unselected neighbouring component
    hidden = components.selection_tile(source, chosen['selection'], image, 0, 0, 0, ['tumor'])
    assert np.count_nonzero(hidden[:, :, 3]) == 0
    assert components.selection(source, [first['properties']['id']]) == chosen
    with pytest.raises(ValueError, match='changed'):
        components.selection(source, ['missing'])
    assert source.read_bytes() == original


def test_review_decisions_do_not_mutate_sources_and_conflicts(tmp_path, source):
    original=source.read_bytes()
    item=flag(tmp_path,source)
    assert flag(tmp_path,source)['id']==item['id']
    answered=store.decide(tmp_path,item['id'],version=1,action='relabel',label='normal')
    assert source.read_bytes()==original and answered['state']=='answered'
    with pytest.raises(ValueError,match='changed'):
        store.decide(tmp_path,item['id'],version=1,action='reject')
    from spacemap_gui.viewer.review_api import apply_decision
    applied=apply_decision(tmp_path,item['id'],answered['version'])
    assert applied['state']=='applied' and source.read_bytes()==original
    assert store.step_state(tmp_path)=='done'
    assert len(store.history(tmp_path,item['id']))==4


def test_bulk_review_decisions_are_atomic_and_scoped(tmp_path, source):
    before = source.read_bytes()
    one, two = flag(tmp_path, source, 0), flag(tmp_path, source, 1)
    shared = store.create_grant(tmp_path, name='One polygon', items=[one['id']])
    versions = {one['id']:one['version'], two['id']:two['version']}
    with pytest.raises(PermissionError):
        store.decide_many(tmp_path, versions, action='relabel', label='normal', grant=(shared['id'],shared['secret']))
    assert store.get(tmp_path,one['id'])['version']==1
    store.decide(tmp_path, two['id'], version=1, action='keep', note='Preserve this comment')
    with pytest.raises(ValueError, match='changed'):
        store.decide_many(tmp_path, versions, action='relabel', label='normal')
    assert store.get(tmp_path,one['id'])['version']==1
    versions[two['id']]=2
    results=store.decide_many(tmp_path,versions,action='relabel',label='normal')
    assert all(item['decision']['label']=='normal' for item in results)
    assert results[1]['decision']['note']=='Preserve this comment'
    assert source.read_bytes()==before


def test_accepting_two_independent_polygons_survives_copy_revision(tmp_path, source):
    from spacemap_gui.viewer.review_api import apply_decision
    items=[flag(tmp_path,source,i) for i in range(2)]
    for item in items:
        answered=store.decide(tmp_path,item['id'],version=1,action='relabel',label='normal')
        assert apply_decision(tmp_path,item['id'],answered['version'])['state']=='applied'
    copy=components.effective_path(tmp_path,tmp_path/'image.tif',source)
    assert components.page(copy)['classes']==[{'name':'normal','count':2}]


def test_stale_polygon_decision_is_blocked(tmp_path, source):
    from spacemap_gui.viewer.review_api import apply_decision
    item=flag(tmp_path,source)
    answered=store.decide(tmp_path,item['id'],version=1,action='reject')
    components.edit(tmp_path,tmp_path/'image.tif',source,ids=item['target']['ids'],expected=item['target']['revision'],action='rename',name='changed')
    with pytest.raises(ValueError,match='changed'):
        apply_decision(tmp_path,item['id'],answered['version'])
    assert store.get(tmp_path,item['id'])['state']=='conflict'


def test_scoped_revoked_grants_and_concurrent_decisions(tmp_path,source):
    one,two=[flag(tmp_path,source,i) for i in range(2)]
    grant=store.create_grant(tmp_path,name='Pathologist',items=[one['id']])
    auth=(grant['id'],grant['secret'])
    assert 'secret_hash' not in store.authorize(tmp_path,*auth)
    with pytest.raises(PermissionError):
        store.decide(tmp_path,two['id'],version=1,action='keep',grant=auth)
    def attempt(_):
        try: return store.decide(tmp_path,one['id'],version=1,action='keep',grant=auth)['state']
        except ValueError: return 'conflict'
    with ThreadPoolExecutor(2) as executor:
        assert sorted(executor.map(attempt,range(2)))==['answered','conflict']
    store.revoke(tmp_path,grant['id'])
    with pytest.raises(PermissionError): store.authorize(tmp_path,*auth)
    assert all('secret' not in json.dumps(g) for g in store.grants(tmp_path))


def test_queue_pages_and_literal_search(tmp_path,source):
    flag(tmp_path,source,0);flag(tmp_path,source,1)
    first=store.list_items(tmp_path,limit=1)
    second=store.list_items(tmp_path,limit=1,offset=1)
    assert first['total']==2 and first['items'][0]['id']!=second['items'][0]['id']
    assert not store.list_items(tmp_path,search="%' OR 1=1 --")['items']
    assert len(store.list_items(tmp_path,ids_only=True)['ids']) == 2
    assert store.list_items(tmp_path,ids=[first['items'][0]['id']],ids_only=True)['ids'] == [first['items'][0]['id']]
    extra=store.add_many(tmp_path,[{'kind':'cells','image':'another.tif','title':'Cell '+str(i),
        'target':{'source':'table','ids':[i],'revision':'fixture'}} for i in range(205)])
    assert len(store.list_items(tmp_path,kind='cells')['items']) == 100
    assert len(store.list_items(tmp_path,kind='cells',ids_only=True)['ids']) == 205
    assert store.list_items(tmp_path,kind='annotation',ids_only=True)['ids'] == store.list_items(tmp_path,ids=[first['items'][0]['id'],second['items'][0]['id']],ids_only=True)['ids']


def test_review_metadata_merged_and_separate_tables(tmp_path, monkeypatch):
    import pandas as pd
    from spacemap_gui.studio import review_images as images
    table=tmp_path/'combined.parquet'
    pd.DataFrame({'batch_id':['a','a','b','b'], 'visit':[1,1,2,2], 'response':['CR','PR','CR','CR'],
                  'private':['secret']*4}).to_parquet(table)
    before=table.read_bytes()
    rows=[{'roi':r,'image_path':str(tmp_path/(r+'.tif')),'image':r+'.tif','table_path':str(table)} for r in ('a','b')]
    result=images.values(tmp_path,rows,['visit','response'])
    assert result['a']['visit']['text']=='1' and result['b']['visit']['text']=='2'
    assert result['a']['response']['text']=='Mixed (2)' and result['b']['response']['text']=='CR'
    assert all('private' not in v for v in result.values())
    load=images.dataio.load
    monkeypatch.setattr(images.dataio,'load',lambda *a,**kw:pytest.fail('Unchanged metadata should use cached summaries'))
    assert images.values(tmp_path,rows,['visit','response'])==result
    monkeypatch.setattr(images.dataio,'load',load)
    per=[]
    for roi,visit in [('a',1),('b',2)]:
        path=tmp_path/(roi+'.csv');pd.DataFrame({'visit':[visit,None],'group':['X','X']}).to_csv(path,index=False)
        per.append({'roi':roi,'image_path':str(tmp_path/(roi+'.tif')),'table_path':str(path)})
    separate=images.values(tmp_path,per,['visit','group'],per_image=True)
    assert separate['a']['visit']['text']=='1.0' and 'no value' in separate['a']['visit']['detail']
    assert separate['b']['visit']['text']=='2.0'
    assert images.snapshot(tmp_path,{'rows':rows},['a'],['visit']).keys()=={str(tmp_path/'a.tif')}
    with pytest.raises(ValueError):images.snapshot(tmp_path,{'rows':rows},['not-listed'],['visit'])
    assert table.read_bytes()==before


def test_image_only_sharing_and_metadata_permissions(tmp_path, monkeypatch):
    import pandas as pd
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import registry, review_links
    from spacemap_gui.viewer.api import app
    from spacemap_gui.studio import review_images
    from test_viewer import _write_channel_pages
    rows=[]
    table=tmp_path/'cells.parquet'
    pd.DataFrame({'batch_id':['a','b'],'timepoint':[1,2],'private':['secret','secret']}).to_parquet(table)
    for roi in ('a','b'):
        path=tmp_path/(roi+'.tif');_write_channel_pages(path,channels=2,height=64,width=64)
        rows.append({'roi':roi,'image':path.name,'image_path':str(path),'table_path':str(table)})
    monkeypatch.setattr(review_images,'plan',lambda channel:{'rows':rows,'per_image':False})
    monkeypatch.setattr(review_links.registry,'_state_folder',lambda:tmp_path/'state')
    registry.publish('review-images-fixture',{'layers':[]},project={'root':str(tmp_path),'unified':str(table)})
    route='/studio/review-images-fixture/review'
    with TestClient(app) as client:
        shared=client.post(route,json={'action':'share','name':'Images only','images':['a'],'metadata_columns':['timepoint']})
        assert shared.status_code==200,shared.text
        base=shared.json()['url'].replace('http://testserver','').rstrip('/')
        assert client.get(base+'/queue').json()['total']==0
        result=client.get(base+'/images').json()
        assert len(result['images'])==1 and result['metadata_columns']==['timepoint']
        assert result['images'][0]['metadata']['timepoint']['text']=='1'
        assert 'private' not in json.dumps(result) and str(tmp_path) not in json.dumps(result)
        image=result['images'][0]['roi']
        spec=client.get(base+'/view_image',params={'id':image}).json()['spec']
        assert client.get(base+'/image/'+image+'/info').status_code==200
        assert client.get(base+'/view_image',params={'id':registry.register(rows[1]['image_path'])}).status_code==403
        assert client.get(base+'/image/'+registry.register(rows[1]['image_path'])+'/info').status_code==403
        assert not spec.get('session') and not spec.get('workspace')
        assert client.post(base+'/decision',json={'id':'fake','version':1,'decision':'keep'}).status_code==403
        updated=client.post(route,json={'action':'update_link','id':shared.json()['id'],'name':'Images only','access':'internal','images':['b'],'metadata_columns':[]})
        assert updated.status_code==200,updated.text
        assert updated.json()['url']==shared.json()['url']
        found=client.get(base+'/images').json()
        assert len(found['images'])==2 and found['metadata_columns']==[] and all(not r['metadata'] for r in found['images'])
        assert store.list_items(tmp_path)['total']==0  # No artificial review tasks.
        store.revoke(tmp_path,shared.json()['id'])
        assert client.get(base+'/images').status_code==403


def test_bulk_suggestions_remembered_by_type_with_atomic_conflicts(tmp_path, source):
    original = source.read_bytes()
    one, two = [flag(tmp_path, source, i) for i in range(2)]
    versions = {item['id']:item['version'] for item in (one, two)}
    result = store.set_suggestions(tmp_path, versions, ['tumor','normal','Other'], remember=True)
    assert result['updated'] == 2
    assert all(store.get(tmp_path, item['id'])['suggestions'] == ['tumor','normal','Other'] for item in (one,two))
    assert source.read_bytes() == original
    assert store.remembered_suggestions(tmp_path)[store.suggestion_context(one)] == ['tumor','normal','Other']
    assert store.suggestion_context({'kind':'annotation','title':'normal','target':{}}) not in store.remembered_suggestions(tmp_path)
    with pytest.raises(ValueError, match='changed'):
        store.set_suggestions(tmp_path, versions, ['stale'])
    assert all(store.get(tmp_path, item['id'])['suggestions'] == ['tumor','normal','Other'] for item in (one,two))


def test_components_follow_project_images_without_republishing_view(tmp_path, source):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import registry
    from spacemap_gui.viewer.api import app
    from spacemap_gui.studio.phenotyping_inputs import save_folders
    from test_viewer import _write_channel_pages
    images=tmp_path/'assigned'; images.mkdir()
    first, second=images/'first.tif',images/'second.tif'
    outside=tmp_path/'unassigned'/'second.tif';outside.parent.mkdir()
    for image in (first,second,outside):
        _write_channel_pages(image,channels=2,height=64,width=64)
    save_folders(tmp_path,{'images':str(images),'tables':str(tmp_path)})
    channel='review-image-navigation'
    registry.publish(channel,{'layers':[{'type':'image','token':registry.register(first)}]},project={'root':str(tmp_path)})
    annotation=registry.register_object('annotations',path=str(source),image=str(second))
    unassigned=registry.register_object('annotations',path=str(source),image=str(outside))
    before=source.read_bytes()
    with TestClient(app) as client:
        prefix='/studio/'+channel+'/review'
        registry.session_remember_images(channel,[{'path':str(second),'name':second.name}])
        found=client.get(prefix+'/components/'+annotation)
        assert found.status_code==200,found.text
        registry.session_remember_images(channel,[])
        # After a server reload, exact project image assignments still qualify.
        again=client.get(prefix+'/components/'+annotation)
        assert again.status_code==200,again.text
        ids=[again.json()['rows'][0]['id']]
        flagged=client.post(prefix,json={'action':'flag','sources':[{'token':annotation,'ids':ids}]})
        assert flagged.status_code==200,flagged.text
        assert len(flagged.json()['items'])==1
        assert client.get(prefix+'/components/'+unassigned).status_code==400
        assert client.post(prefix,json={'action':'flag','token':unassigned,'ids':ids}).status_code==400
        assert registry.session_view(channel,mark_seen=False)['spec']['layers'][0]['token']==registry.register(first)
    assert source.read_bytes()==before


def test_owner_and_public_routes(tmp_path, source):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import registry
    from spacemap_gui.viewer.api import app
    image=tmp_path/'image.tif'
    from test_viewer import _write_channel_pages
    _write_channel_pages(image, channels=2, height=64, width=64)
    token=registry.register(str(image))
    annotation=registry.register_object('annotations',path=str(source),image=str(image))
    registry.publish('review-fixture',{'layers':[{'type':'image','token':token}]},project={'root':str(tmp_path)})
    with TestClient(app) as client:
        prefix='/studio/review-fixture/review'
        response=client.get(prefix+'/components/'+annotation)
        assert response.status_code==200,response.text
        first=response.json()['rows'][0]['id']
        flagged=client.post(prefix,json={'action':'flag','token':annotation,'ids':[first]})
        assert flagged.status_code==200,flagged.text
        item=flagged.json()['items'][0]
        options=client.post(prefix,json={'action':'suggestion_options','ids':[item['id']]}).json()
        assert options['options']==['Other','tumor']
        assert client.post(prefix,json={'action':'suggestions','items':{item['id']:item['version']},'choices':['tumor','normal'],'remember':True}).status_code==200
        other=next(r['id'] for r in components.index(source)['rows'] if r['id']!=first)
        second=client.post(prefix,json={'action':'flag','token':annotation,'ids':[other]}).json()['items'][0]
        assert second['suggestions']==['tumor','normal']
        shared=client.post(prefix,json={'action':'share','name':'External reviewer','items':[item['id']]})
        assert shared.status_code==200,shared.text
        base=shared.json()['url'].split('http://testserver')[-1].rstrip('/')
        assert client.get(base+'/queue').status_code==200
        assert client.post(base+'/studio/review-fixture/review',json={'action':'apply'}).status_code==403
        assert client.get(base+'/image/alien/info').status_code==403
        viewed=client.get(base+'/view',params={'id':item['id']})
        assert viewed.status_code==200,viewed.text
        assert 'session' not in viewed.json()['spec']
        assert client.get(base+'/components',params={'item':item['id']}).json()['total']==2
        clicked=client.get(base+'/polygon',params={'item':item['id'],'id':first}).json()['items']
        assert len(clicked)==1 and clicked[0]['suggestions']==['tumor','normal']
        assert 'target' not in clicked[0]
        assert client.get(base+'/polygon',params={'item':item['id'],'id':other}).json()['items']==[]
        assert client.get(base+'/polygon',params={'item':second['id'],'id':other}).status_code==403
        preview=client.post(base+'/preview',json={'focus':item['id'],'items':[item['id']]}).json()
        assert len(preview['features'])==1
        alias=viewed.json()['spec']['layers'][0]['token']
        image_preview=client.get(base+'/image_preview',params={'image':alias}).json()
        assert len(image_preview['features'])==1 and image_preview['revision']>0
        assert image_preview['features'][0]['properties']['review_suggestions']==['tumor','normal']
        assert len(client.get(prefix,params={'action':'image_preview','image':token}).json()['features'])==2
        assert client.get(base+'/image_preview',params={'image':token}).status_code==403
        choices=client.post(base+'/label_options',json={'ids':[item['id']]}).json()
        assert 'tumor' in choices['options'] and 'target' not in choices['items'][0]
        assert client.post(base+'/label_options',json={'ids':[second['id']]}).status_code==403
        assert client.post(base+'/decisions',json={'items':{second['id']:second['version']},'decision':'keep'}).status_code==403
        decided=client.post(base+'/decisions',json={'items':{item['id']:choices['items'][0]['version']},'decision':'relabel','label':'normal'})
        assert decided.status_code==200,decided.text
        assert 'target' not in decided.json()['items'][0]
        assert client.post(base+'/preview',json={'focus':item['id'],'items':[second['id']]}).status_code==403
        other=next(r['id'] for r in components.index(source)['rows'] if r['id']!=first)
        nomination=client.post(base+'/nominate',json={'item':item['id'],'ids':[other]})
        assert nomination.status_code==200,nomination.text
        assert nomination.json()['items'][0]['state']=='pending'
        assert nomination.json()['items'][0]['id']==second['id']  # Existing flags are not duplicated or demoted.
        before=client.get(base+'/queue').json()['revision']
        assert client.post(base+'/remove',json={'items':[item['id']]}).status_code==403
        original=source.read_bytes()
        removed=client.post(prefix,json={'action':'remove','items':[item['id']]})
        assert removed.status_code==200 and removed.json()['removed']==[item['id']]
        updated=client.get(base+'/queue',params={'since':before}).json()
        assert updated['total']==0 and updated['removed']==[item['id']]
        assert client.get(base+'/view',params={'id':item['id']}).status_code==403
        assert source.read_bytes()==original
        assert client.post(prefix,json={'action':'revoke','id':shared.json()['id']}).status_code==200
        assert client.get(base+'/queue').status_code==403


def test_live_link_destinations_do_not_expand_other_grants(tmp_path, source):
    one=flag(tmp_path,source,0)
    first=store.create_grant(tmp_path,name='First',items=[one['id']])
    second=store.create_grant(tmp_path,name='Second',items=[one['id']])
    store.destination(tmp_path,first['id'])
    two=flag(tmp_path,source,1)
    assert set(store.authorize(tmp_path,first['id'],first['secret'])['scope']['items'])=={one['id'],two['id']}
    assert store.authorize(tmp_path,second['id'],second['secret'])['scope']['items']==[one['id']]
    store.append_to_grant(tmp_path,second['id'],[two['id']])
    assert store.append_to_grant(tmp_path,second['id'],[two['id']])['added']==0
    before=store.notifications(tmp_path)['revision']
    store.decide(tmp_path,two['id'],version=two['version'],action='keep',grant=(first['id'],first['secret']))
    note=store.notifications(tmp_path,before)
    assert note['pending']==1 and note['new'][0]['name']=='First'
    store.revoke(tmp_path,first['id'])
    assert store.destination(tmp_path)==''
    with pytest.raises(ValueError,match='closed'):
        store.append_to_grant(tmp_path,first['id'],[one['id']])


def test_lightweight_overview_does_not_prepare_geometry(source,monkeypatch):
    monkeypatch.setattr(edits,'read',lambda *a:pytest.fail('Overview must not prepare components'))
    overview=components.overview(source)
    assert overview['classes']==[{'name':'tumor','count':2,'color':'#ff0000'}]


def test_prepared_geometry_fingerprint_matches_legacy_mapping():
    from shapely.geometry import shape,mapping
    data={'type':'FeatureCollection','features':[{'type':'Feature','properties':{'classification':{'name':'tumor'}},
        'geometry':{'type':'MultiPolygon','coordinates':[[[[0,0],[4,0],[4,4],[0,4],[0,0]]],[[[8,0],[9,0],[9,1],[8,1],[8,0]]]]}}]}
    prepared=edits.prepare(data["features"])
    for feature in prepared:
        canonical={**feature,'geometry':mapping(shape(feature['geometry']))}
        assert components.fingerprint(feature)==components.fingerprint(canonical)
        assert edits.prepared_geometry(feature).equals_exact(shape(feature['geometry']),0)


def test_area_picks_preserve_holes_hidden_classes_and_exact_enclosure(source):
    before=source.read_bytes()
    all_points=[[-1,-1],[31,-1],[31,11],[-1,11]]
    assert len(components.area_selection(source,all_points)['ids'])==2
    assert components.area_selection(source,all_points,['tumor'])['ids']==[]
    assert components.area_selection(source,[[1,1],[9,1],[9,9],[1,9]])['ids']==[]
    assert components.area_selection(source,[[2.1,2.1],[3.9,2.1],[3.9,3.9],[2.1,3.9]])['ids']==[]
    assert len(components.area_selection(source,[[0,0],[10,0],[10,10],[0,10]])['ids'])==1
    assert source.read_bytes()==before


def test_area_cells_use_shown_classes_and_one_roi(tmp_path):
    import pandas as pd
    from spacemap_gui.studio.view_selection import cells_in_area
    table=tmp_path/'cells.csv'
    pd.DataFrame({'batch_id':['one','one','one','two'],'label':[1,2,3,4],
        'x_centroid':[5,15,25,5],'y_centroid':[5,5,5,5],
        'phenotype_ref1':['A','B','A','A']}).to_csv(table,index=False)
    binding={'tables':{'t':str(table)},'roi':'one','roi_column':'batch_id','label_column':'label'}
    body={'points':[[0,0],[20,0],[20,10],[0,10]],'view':{'t':'t','col':'phenotype_ref1','hide':'B'}}
    result=cells_in_area(binding,body)
    assert result['ids']==[1]
    assert cells_in_area(binding,{**body,'add':[3,4]})['ids']==[1,3]
    assert cells_in_area(binding,{**body,'only_ids':[2,3,4]})['ids']==[3]
    assert len(pd.read_csv(table))==4


def test_review_preview_and_applied_rename_use_destination_colour(tmp_path,source):
    from spacemap_gui.viewer.review_api import review_previews,apply_decision
    data=json.loads(source.read_text())
    data['features'][0]['geometry']['type']='Polygon'
    shapes=data['features'][0]['geometry']['coordinates']
    data['features'][0]['geometry']['coordinates']=shapes[0]
    data['features'].append({'type':'Feature','geometry':{'type':'Polygon','coordinates':shapes[1]},
        'properties':{'classification':{'name':'normal','color':[0,255,0]}}})
    source.write_text(json.dumps(data))
    before=source.read_bytes();item=flag(tmp_path,source)
    answer=store.decide(tmp_path,item['id'],version=item['version'],action='relabel',label='normal')
    preview=review_previews(tmp_path,[item['id']],item['id'])['features'][0]
    assert preview['properties']['classification']=={'name':'normal','color':[0,255,0]}
    assert source.read_bytes()==before
    apply_decision(tmp_path,item['id'],answer['version'])
    path=components.effective_path(tmp_path,item['target']['image'],source)
    actual=components.index(path)['by_id'][item['target']['ids'][0]]
    assert actual['properties']['classification']['color']==[0,255,0]
    assert actual['properties']['colour']=='#00ff00'


def test_complete_review_preview_is_image_and_grant_scoped(tmp_path, source):
    from spacemap_gui.viewer.review_api import review_previews, _project_spec
    one,two=flag(tmp_path,source,0),flag(tmp_path,source,1)
    other=store.add(tmp_path,kind='annotation',image='other.tif',title='Other image',
        target={**one['target'],'image':str(tmp_path/'other.tif')})
    original=source.read_bytes()
    full=review_previews(tmp_path,[],one['id'],all_in_image=True)['features']
    assert {f['properties']['review_item'] for f in full}=={one['id'],two['id']}
    scoped=review_previews(tmp_path,[],one['id'],all_in_image=True,allowed=[one['id'],other['id']])['features']
    assert len(scoped)==1 and scoped[0]['properties']['review_item']==one['id']
    token=scoped[0]['properties']['review_source']
    public=_project_spec({'id':'fixture','layers':[{'features':scoped}]},'grant')
    assert public['layers'][0]['features'][0]['properties']['review_source']!=token
    assert source.read_bytes()==original


def test_saved_link_edit_reuses_scope_and_credential(tmp_path, source, monkeypatch):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import registry, review_links
    from spacemap_gui.viewer.api import app
    image=tmp_path/'image.tif'
    registry.publish('share-edit-fixture',{'layers':[]},project={'root':str(tmp_path)})
    monkeypatch.setattr(review_links.registry, '_state_folder', lambda:tmp_path/'state')
    monkeypatch.delenv('SPACEMAP_REVIEW_PUBLIC_BASE_URL', raising=False)
    one,two=flag(tmp_path,source,0),flag(tmp_path,source,1)
    with TestClient(app) as client:
        route='/studio/share-edit-fixture/review'
        created=client.post(route,json={'action':'share','name':'Pathologist','items':[one['id']]}).json()
        credential=created['url'].split('/review-public/')[1]
        listed=client.get(route+'?action=links').json()['links'][0]
        assert listed['url_saved'] and listed['access']=='internal'
        recalled=client.get(route,params={'action':'link','id':created['id']})
        assert recalled.json()['url']==created['url'] and recalled.headers['cache-control']=='no-store'
        assert 'secret' not in json.dumps(listed) and credential not in json.dumps(listed)
        _,path=review_links._path(tmp_path,created['id'])
        assert path.stat().st_mode & 0o777 == 0o600
        body={'action':'update_link','id':created['id'],'name':'Dr Example','access':'external','items':[two['id']]}
        assert client.post(route,json=body).status_code==400
        assert store.grants(tmp_path)[0]['scope']['items']==[one['id']]
        monkeypatch.setenv('SPACEMAP_REVIEW_PUBLIC_BASE_URL','https://review.example.org/spacemap-viewer/api')
        updated=client.post(route,json=body)
        assert updated.status_code==200,updated.text
        assert updated.json()['url']=='https://review.example.org/spacemap-viewer/api/review-public/'+credential
        assert updated.json()['added']==1
        assert len(store.grants(tmp_path))==1 and store.grants(tmp_path)[0]['name']=='Dr Example'
        oldbase=created['url'].replace('http://testserver','').rstrip('/')
        assert client.get(oldbase+'/queue').json()['total']==2
        assert client.post(oldbase+'/update_link',json=body).status_code==403
        path.unlink()  # Older links stored no recoverable key.
        assert client.post(route,json=body).status_code==400
        assert client.get(oldbase+'/queue').status_code==200  # A reviewer revisiting recovers its credential automatically.
        assert client.get(route,params={'action':'link','id':created['id']}).json()['url'].endswith(credential)
        path.unlink()
        recovered=client.post(route,json={**body,'existing_url':created['url']})
        assert recovered.status_code==200 and recovered.json()['url'].endswith(credential)
        previous=recovered.json()['url']
        new_host={'host':'new-node.example:8503'}
        assert client.get(route,params={'action':'link','id':created['id']},headers=new_host).json()['url']==previous
        moved=client.post(route,json={**body,'access':'internal'},headers=new_host)
        assert moved.status_code==200 and moved.json()['url']=='http://new-node.example:8503/review-public/'+credential
        assert client.get(route,params={'action':'link','id':created['id']}).json()['url']==moved.json()['url']
        assert len(store.grants(tmp_path))==1 and len(store.grants(tmp_path)[0]['scope']['items'])==2
        store.revoke(tmp_path,created['id'])
        assert client.post(route,json=body).status_code==403
        assert client.get(oldbase+'/queue').status_code==403


def test_remove_review_preserves_polygons_edits_history_and_scopes(tmp_path, source):
    from spacemap_gui.viewer.review_api import apply_decision
    original = source.read_bytes()
    one, two = flag(tmp_path, source, 0), flag(tmp_path, source, 1)
    shared = store.create_grant(tmp_path, name='Both', items=[one['id'], two['id']])
    other = store.create_grant(tmp_path, name='Only second', items=[two['id']])
    answered = store.decide(tmp_path, one['id'], version=one['version'], action='relabel', label='normal')
    applied = apply_decision(tmp_path, one['id'], answered['version'])
    saved = {p: p.read_bytes() for p in tmp_path.rglob('*.geojson') if p.is_file()}
    before = store.list_items(tmp_path)['revision']
    assert store.remove(tmp_path, [one['id'], one['id']]) == {'removed': [one['id']]}
    assert source.read_bytes() == original
    assert all(p.read_bytes() == content for p, content in saved.items())
    assert store.list_items(tmp_path)['total'] == 1
    assert store.authorize(tmp_path, shared['id'], shared['secret'])['scope']['items'] == [two['id']]
    assert store.list_items(tmp_path, since=before)['removed'] == [one['id']]
    assert store.list_items(tmp_path, since=before, grant_id=shared['id'])['removed'] == [one['id']]
    assert store.list_items(tmp_path, since=before, grant_id=other['id'])['removed'] == []
    history = store.history(tmp_path, one['id'])
    assert history[-1]['action'] == 'remove' and history[-1]['payload']['entry'] == applied
    assert any(e['action'] == 'decision' for e in history)
    again = flag(tmp_path, source, 0)
    assert again['id'] != one['id'] and again['state'] == 'pending'


def test_remove_review_is_atomic_while_approval_in_progress(tmp_path, source):
    one, two = flag(tmp_path, source, 0), flag(tmp_path, source, 1)
    grant = store.create_grant(tmp_path, name='Both', items=[one['id'], two['id']])
    with pytest.raises(ValueError, match='changed'):
        store.remove(tmp_path, [one['id'], 'missing'])
    answered = store.decide(tmp_path, two['id'], version=two['version'], action='keep')
    store.transition(tmp_path, two['id'], version=answered['version'], state='applying')
    with pytest.raises(ValueError, match='approval'):
        store.remove(tmp_path, [one['id'], two['id']])
    assert store.list_items(tmp_path)['total'] == 2
    assert len(store.authorize(tmp_path, grant['id'], grant['secret'])['scope']['items']) == 2


def test_suggestions_across_selected_images_are_unpaged_and_preserve_decisions(tmp_path, monkeypatch):
    from spacemap_gui.viewer.review_api import suggestion_options
    images = [str(tmp_path / name / 'same.tif') for name in ('a', 'b', 'c')]
    rows = store.add_many(tmp_path, [dict(kind='annotation', image='same.tif', title='tumor',
        target={'image':images[i % 2], 'source':'not-read.geojson', 'revision':'fixture', 'ids':[str(i)], 'annotation_name':'tumor'}) for i in range(205)])
    other = store.add(tmp_path, kind='annotation', image='same.tif', title='tumor', target={'image':images[2], 'source':'not-read.geojson', 'revision':'fixture', 'ids':['outside']})
    cells = store.add(tmp_path, kind='cells', image='same.tif', title='A', target={'image':images[0], 'source':'not-read.csv', 'revision':'fixture', 'ids':['cell']})
    answered = store.decide(tmp_path, rows[0]['id'], version=rows[0]['version'], action='relabel', label='normal', actor='reviewer')
    def forbid_geometry(*args, **kwargs):
        raise AssertionError('Bulk suggestions must not load whole-image polygon geometry')
    monkeypatch.setattr(components, 'page', forbid_geometry)
    preview = suggestion_options(tmp_path, None, images[:2], 'annotation')
    assert len(preview['items']) == 204 and preview['skipped'] == 1
    assert {'tumor','Other'} <= set(preview['options'])
    versions = {item['id']:item['version'] for item in preview['items']}
    assert store.set_suggestions(tmp_path, versions, ['tumor','normal','Other'])['updated'] == 204
    assert store.get(tmp_path, rows[0]['id']) == answered
    assert not store.get(tmp_path, other['id'])['suggestions']
    assert not store.get(tmp_path, cells['id'])['suggestions']
    assert all(store.get(tmp_path, item['id'])['suggestions'] == ['tumor','normal','Other'] for item in preview['items'])
    # A decision arriving after the preview must block the stale bulk write.
    before = store.get(tmp_path, rows[2]['id'])
    updated = suggestion_options(tmp_path, None, images[:2], 'annotation')['items']
    store.decide(tmp_path, rows[1]['id'], version=next(i['version'] for i in updated if i['id']==rows[1]['id']), action='keep', actor='reviewer')
    with pytest.raises(ValueError, match='changed'):
        store.set_suggestions(tmp_path, {i['id']:i['version'] for i in updated}, ['different'])
    assert store.get(tmp_path, rows[2]['id']) == before
