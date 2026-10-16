import json
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from starlette.testclient import TestClient

from spacemap_gui.studio import classification as c, annotation_edits as edits, annotation_inputs as inputs, annotation
from spacemap_gui.viewer import registry
from spacemap_gui.viewer.api import app
from test_studio_pixelclasses import regions
from test_studio_commands import setup


def source(tmp_path):
    path=tmp_path/'Tumor.geojson'
    path.write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'classification':{'name':'Tumor'}},'geometry':mapping(box(64.5,64.5,560.5,560.5))}]}))
    return path


def test_scale_channels_features_and_cache(regions,tmp_path,monkeypatch):
    scope=source(tmp_path)
    prepared,found,chosen,_=c.compute(regions,{},scope,{'scale':2,'n_clusters':4})
    assert prepared['factor']==2 and len(prepared['stack']['marker_names'])==8
    assert prepared['stack']['shape']==(320,320)
    modern={**chosen,'selected_only':True,'markers':['DAPI','CD3'],'features':['texture','edges']}
    selected,result,_,_=c.compute(regions,{},scope,modern)
    assert selected['stack']['marker_names']==['DAPI','CD3']
    assert selected['stack']['features'].shape[1]==6 and np.isfinite(selected['stack']['features']).all()
    assert found['run_id']!=result['run_id']
    def refuse(*a,**k):raise AssertionError('Redraw must not retrain')
    monkeypatch.setattr('spacemap.tissue_annotation_utils.run_pixel_clustering',refuse)
    assert c.compute(regions,{},scope,modern)[1] is result
    with pytest.raises(ValueError,match='at least one marker'):c.options({'selected_only':True})
    with pytest.raises(ValueError,match='not in this image'):c.compute(regions,{},scope,{**modern,'markers':['NotAMarker']})


def test_merge_weighted_profiles_exact_polygon_partition_and_history(regions,tmp_path):
    scope=source(tmp_path);prepared,found,chosen,_=c.compute(regions,{},scope,{'scale':4,'n_clusters':4})
    before=c.render(prepared,found,None); ids=[i['id'] for i in before['classes']]
    a,b=ids[:2]; state={**before['review'],'groups':[[a,b]],'names':{str(a):'Tumor proliferative'},'aggregation':'mean'}
    merged=c.render(prepared,found,state)
    row=next(i for i in merged['classes'] if i['id']==a)
    assert row['members']==[a,b] and row['count']==int(found['counts'][[a,b]].sum())
    expected=found['z_sums'][[a,b]].sum(0)/found['counts'][[a,b]].sum()
    index=merged['profile']['rows'].index(str(a))
    np.testing.assert_allclose(merged['profile']['values'][index],expected,atol=.00006)
    output=tmp_path/'output'; target=output/(annotation.sample_id(regions)+'_raw_macro_clusters.geojson')
    result=c.save_polygons(tmp_path,regions,output,scope,prepared,found,state,edits.revision(target))
    features=edits.read(target)[0]
    assert features and any(f['properties']['classification']['name']=='Tumor proliferative' for f in features)
    areas=[shape(f['geometry']) for f in features];union=unary_union(areas)
    assert sum(a.area for a in areas)==pytest.approx(union.area)  # no overlap after merging
    assert union.difference(box(64.5,64.5,560.5,560.5)).area==0
    assert all(f['properties']['classification_run']==found['run_id'] for f in features)
    changed={**state,'excluded':[a,b]}
    c.save_polygons(tmp_path,regions,output,scope,prepared,found,changed,result['revision'])
    assert len(edits.history(target)['entries'])==2
    restored=edits.restore(target,result['revision'],edits.revision(target))
    assert unary_union([shape(f['geometry']) for f in restored['features']]).area==pytest.approx(union.area)


def test_drafts_isolated_and_concurrent_edits_checked(regions,tmp_path):
    scope=source(tmp_path);_,found,chosen,_=c.compute(regions,{},scope,{'n_clusters':4})
    state=c.review({'names':{'0':'Tumor'}},range(4))
    version=c.save_review(tmp_path,regions,found,chosen,state)
    assert c.initial(tmp_path,regions)['run_id']==found['run_id']
    assert c.initial(tmp_path,'different.ome.tiff')['options']==c.DEFAULTS
    with pytest.raises(ValueError,match='another window'):c.save_review(tmp_path,regions,found,chosen,state,'stale')
    assert c.save_review(tmp_path,regions,found,chosen,state,version)==version


def test_assistant_receives_matrix_and_rejects_invented_ids():
    class Fake:
        def complete(self,messages,**kwargs):
            assert 'Tumor' in messages[-1]['content'] and 'PCNA' in messages[-1]['content']
            return json.dumps({'names':[{'id':0,'name':'Proliferative pattern','reason':'PCNA enriched'}, {'id':99,'name':'invented'}],
                               'merges':[{'ids':[0,1],'reason':'Similar profiles'},{'ids':[0,99]}]})
    context={'classes':[{'id':0},{'id':1}],'annotation_name':'Tumor','profile':{'columns':['PCNA'],'values':[[2],[1]]}}
    result=c.suggest(context,client=Fake())
    assert len(result['names'])==1 and result['merges']==[{'ids':[0,1],'reason':'Similar profiles'}]


def test_api_honors_imported_class_and_rejects_stale_input(setup):
    channel,_,root,spec=setup;image=registry.resolve(spec['layers'][0]['token'])
    output=annotation.all_folders(root,{})['output']
    imported=inputs.import_polygons(root,image,'Other annotations',[{'type':'Feature','properties':{'classification':{'name':name}},'geometry':mapping(geom)}
        for name,geom in [('Tumor',box(0,0,32,32)),('Stroma',box(32,32,64,64))]])
    inputs.save_selection(root,{},image,output,'classes',{'mode':'file','path':str(imported.resolve()),'class_name':'Tumor'})
    client=TestClient(app);url='/studio/'+channel+'/classification'
    body={'image_token':spec['layers'][0]['token'],'options':{'scale':2,'n_clusters':2}}
    response=client.post(url,json=body);assert response.status_code==200,response.text
    result=response.json();assert result['pixels']<32*32
    body.update(run_id=result['run_id'],revision=result['revision'],review=result['review'],action='review')
    response=client.post(url,json=body);assert response.status_code==200,response.text
    body['run_id']='old';assert client.post(url,json=body).status_code==400
    assert inputs.selection(root,image,'clot')['class_name']=='Tumor'



def test_saved_classes_give_zones_their_domains(tmp_path):
    from spacemap_gui.studio import classification, zones

    def square(name, x, cluster):
        return {'type': 'Feature', 'properties': {'name': name, 'classification': {'name': name, 'color': [200, 0, 0]},
                                                    'source_clusters': [cluster]},
                'geometry': {'type': 'Polygon', 'coordinates': [[[x, 0], [x + 10, 0], [x + 10, 10], [x, 10], [x, 0]]]}}

    classification.write_domains(tmp_path, 'S1', [square('Tumor', 0, 1), square('Stroma', 20, 2), square('Tumor', 40, 3)])
    path = zones.source(tmp_path, 'S1')  # no longer "Export the pixel classes first"
    assert sorted(zones.classes(path)) == ['Stroma', 'Tumor']
