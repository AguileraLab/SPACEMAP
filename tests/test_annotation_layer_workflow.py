"""Threshold regions → split/name → subclassify → export, with recoverable parents."""
import json
from pathlib import Path
import pytest
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from starlette.testclient import TestClient

from spacemap_gui.studio import annotation, annotation_inputs as inputs, annotation_layers as layers
from spacemap_gui.studio import annotation_edits as edits, classification, regions, commands
from spacemap_gui.viewer import registry
from spacemap_gui.viewer.api import app
from test_studio_regions import _image
from test_studio_commands import setup


def test_regions_split_rename_subclassify_export_without_copying_parent(tmp_path):
    root=tmp_path/'study';root.mkdir()
    config={'project_name':'study','directory_mappings':{},'tab_configurations':{}}
    (root/'study.json').write_text(json.dumps(config))
    image=_image(root/'sample.ome.tiff');output=root/'outputs/annotations'
    values=regions.check({'rules':[{'marker':'PanCK','above':100}], 'sigma':0,
                          'within':'image','scale':4,'name':'Epithelia','min_area_um2':0})
    parent={'mode':'image','class_name':''}
    target=regions.target(output,image,values,parent)
    found,stats=regions.find(image,values)
    assert stats['grid_shape']==[100,100] and stats['scale']==4
    assert stats['level_microns']==4
    regions.write(target,found)
    regions.register_result({'root':root,'image':image,'target':target,'selection':parent,'values':values})
    first=edits.read(target)
    cut=edits.edit(target,[first[0][0]['properties']['id']],'split',first[1],line=[[0,200],[205,200]],width=2)
    assert len(cut['features'])==2
    for f in cut['features']:
        name='Tumor' if shape(f['geometry']).centroid.y<200 else 'Normal'
        f['properties'].update(name=name,label=name,named=True,classification={'name':name,'color':[11,33,99]})
    edits.save(target,cut['features'],cut['revision'])
    selection={'mode':'file','path':str(target.resolve()),'class_name':'Tumor'}
    inputs.save_selection(root,config,image,output,'classes',selection)
    assert inputs.prepare_working(root,config,image,output,'classes')==selection
    assert not inputs.imported(root,image)  # generated layers stay live; no disconnected copy
    assert inputs.working_layer(root,config,image,output,'classes')['saved_file']==str(target.resolve())
    scope=inputs.scope_path(root,image,output,'classes')
    tumor=unary_union([shape(f['geometry']) for f in edits.read(scope)[0]])
    before=target.read_bytes()
    prepared,found,chosen,_=classification.compute(str(image),config,scope,{'scale':4,'n_clusters':2,'smoothing':0})
    child=layers.classification_target(output,image,selection)
    saved=classification.save_polygons(root,image,output,scope,prepared,found,None,edits.revision(child),target=child)
    assert child!=target and target.read_bytes()==before
    assert saved['layer']['id']!='pixel-classes' and saved['layer']['editable']
    classified=edits.read(child)[0]
    assert classified and all(shape(f['geometry']).difference(tumor).area==0 for f in classified)
    assert inputs.sources(root,config,image,output)[str(child.resolve())]=='Classification · Tumor'
    reopened=commands._annotation_layers(config,{'output':output},image,annotation.sample_id(image),root)
    assert {str(target.resolve()),str(child.resolve())}.issubset({l['saved_file'] for l in reopened})
    choices=[i for i in layers.export_options(root,config,image,output) if i['path'] in (str(target.resolve()),str(child.resolve()))]
    exported=layers.export(root,config,image,output,choices)
    assert exported['layers']==2
    assert exported['count']==2+len(classified)
    assert {'Tumor','Normal'}.issubset({f['properties']['classification']['name'] for f in exported['geojson']['features']})
    assert all(f['properties']['source_layer_revision'] for f in exported['geojson']['features'])
    assert target.read_bytes()==before
    assert str(annotation.exported(output,annotation.sample_id(image))) not in [i['path'] for i in layers.export_options(root,config,image,output)]
    edits.save(target,cut['features'][:1],edits.revision(target))
    with pytest.raises(ValueError,match='changed'):layers.export(root,config,image,output,choices)


def test_marker_regions_clip_exact_scope_and_keep_holes(tmp_path):
    image=_image(tmp_path/'sample.ome.tiff');scope=tmp_path/'scope.geojson'
    polygon=box(.3,.7,105.8,306.9).difference(box(45.5,60.5,65.5,80.5))
    edits.replace(scope,[{'type':'Feature','properties':{},'geometry':mapping(polygon)}],edits.revision(scope),'Scope')
    found,_=regions.find(image,{'rules':[{'marker':'PanCK','above':100}],'scale':4,'sigma':0,'min_area_um2':0},scope)
    union=unary_union([shape(f['geometry']) for f in found])
    assert union.area>0 and union.difference(polygon).area==0
    assert union.intersection(box(45.5,60.5,65.5,80.5)).area==0


def test_export_route_validates_image_sources_and_revisions(setup):
    channel,_,root,spec=setup;image=registry.resolve(spec['layers'][0]['token'])
    output=annotation.all_folders(root,{})['output'];path=annotation.classes(output,annotation.sample_id(image))
    edits.replace(path,[{'type':'Feature','properties':{'classification':{'name':'Tumor'}},'geometry':mapping(box(1,1,20,20))}],edits.revision(path),'Test')
    client=TestClient(app);url='/studio/'+channel+'/annotation_export'
    listing=client.get(url);assert listing.status_code==200,listing.text
    assert any(i['name']=='Tumor' and not i['export'] for i in listing.json()['names'])  # the names, as Layers shows them; none ticked yet
    chosen=next(i for i in layers.export_options(root,{},image,output) if i['path']==str(path.resolve()))
    assert chosen['selected'] and chosen['classes']=={'Tumor':1}
    body={'image_token':spec['layers'][0]['token'],'layers':[chosen]}
    result=client.post(url,json=body);assert result.status_code==200,result.text
    assert result.json()['count']==1 and Path(result.json()['path']).is_file()
    assert client.post(url,json={**body,'image_token':'wrong'}).status_code==400
    assert client.post(url,json={**body,'layers':[{'path':'/tmp/another-study.geojson','revision':'x'}]}).status_code==400
    assert client.post(url,json={**body,'layers':[{**chosen,'revision':'stale'}]}).status_code==400
    ticked=client.post(url,json={'action':'settings','changes':{'rules':{'local:Tumor':{'export':True}}}});assert ticked.status_code==200,ticked.text
    by_name=client.post(url,json={'image_token':spec['layers'][0]['token']})  # by the export rules
    assert by_name.status_code==200 and 'Tumor' in by_name.json()['names'],by_name.text


def test_subclassification_save_route_keeps_generated_parent(setup):
    channel,_,root,spec=setup;image=registry.resolve(spec['layers'][0]['token'])
    output=annotation.all_folders(root,{})['output'];path=annotation.marker_regions(output,annotation.sample_id(image))
    edits.replace(path,[{'type':'Feature','properties':{'classification':{'name':name}},'geometry':mapping(geom)}
        for name,geom in [('Tumor',box(1,1,30,30)),('Normal',box(31,31,60,60))]],edits.revision(path),'Test')
    selection={'mode':'file','path':str(path.resolve()),'class_name':'Tumor'}
    inputs.save_selection(root,{},image,output,'classes',selection)
    before=path.read_bytes();client=TestClient(app);url='/studio/'+channel+'/classification'
    body={'image_token':spec['layers'][0]['token'],'options':{'scale':4,'n_clusters':2}}
    response=client.post(url,json=body);assert response.status_code==200,response.text
    result=response.json()
    body.update({key:result[key] for key in ('run_id','revision','scope_revision','polygon_revision','review')},action='save')
    saved=client.post(url,json=body);assert saved.status_code==200,saved.text
    assert saved.json()['saved_polygons']['layer']['name']=='Classification · Tumor'
    assert path.read_bytes()==before
    reloaded=client.get('/studio/'+channel+'/annotation_sources?layers=1').json()['layers']
    child=saved.json()['saved_polygons']['path']
    assert len([l for l in reloaded if l.get('saved_file')==child and l.get('editable')])==1
    assert len({l['id'] for l in reloaded})==len(reloaded)


def test_export_merges_polygons_with_the_same_name():
    from shapely.geometry import shape

    from spacemap_gui.studio.annotation_layers import merge_same_names

    def square(name, x, source):
        properties = {'name': name, 'source_feature_id': source, 'source_layer': 'layer.geojson'}
        if name:
            properties['classification'] = {'name': name}
        return {'type': 'Feature', 'properties': properties,
                'geometry': {'type': 'Polygon', 'coordinates': [[[x, 0], [x + 10, 0], [x + 10, 10], [x, 10], [x, 0]]]}}

    merged = merge_same_names([square('Tumor', 0, 'a'), square('Stroma', 20, 'b'), square('Tumor', 40, 'c'), square('', 60, 'd')])
    names = [feature['properties'].get('name') for feature in merged]
    assert names == ['Tumor', 'Stroma', ''], names  # first-seen order; the unnamed polygon stays on its own
    tumour = merged[0]
    assert shape(tumour['geometry']).geom_type == 'MultiPolygon' and abs(shape(tumour['geometry']).area - 200) < 1e-6
    assert tumour['properties']['merged_count'] == 2 and 'merged_from' not in tumour['properties']  # small, whatever the count
