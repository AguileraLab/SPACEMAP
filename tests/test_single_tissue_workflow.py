"""One saved Tissue source, precise class-scoped subtraction and recoverable migration."""
import json
from pathlib import Path

import pytest
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from starlette.testclient import TestClient

from spacemap_gui.studio import annotation, annotation_edits as edits, annotation_inputs as inputs, artifact_learning as learning
from spacemap_gui.viewer.api import app
from spacemap_gui.viewer import registry
from test_studio_commands import setup


def feature(geom, name='tissue', identifier='one'):
    return {'type':'Feature','properties':{'id':identifier,'classification':{'name':name}},'geometry':mapping(geom)}


def area(path):
    return unary_union([shape(f['geometry']) for f in edits.read(path)[0]])


def put(path, features, kind='regions'):
    return edits.replace(path,features,edits.revision(path),'Prepared',kind=kind)


def test_legacy_clean_continues_once_and_original_remains_in_history(tmp_path):
    tissue=annotation.valid_tissue(tmp_path,'s');clean=annotation.cleaned_tissue(tmp_path,'s');artifact=tmp_path/'artifact.geojson'
    original=put(tissue,[feature(box(0,0,100,100))],'tissue')
    put(artifact,[feature(box(0,0,40,100))])
    edits.subtract(tissue,[artifact],original['revision'],target=clean,target_revision=edits.revision(clean))
    old_clean=clean.read_bytes();clean_area=area(clean)
    annotation.unify_tissue(tmp_path,'s')
    assert area(tissue).equals(clean_area) and clean.read_bytes()==old_clean
    assert annotation.analysis_tissue(tmp_path,'s')==tissue
    assert edits.history(tissue)['entries'][0]['label']=='Continued from saved Clean Tissue'
    edits.restore(tissue,original['revision'],edits.revision(tissue),kind='tissue')
    assert area(tissue).area==10000
    restored=tissue.read_bytes();annotation.unify_tissue(tmp_path,'s')
    assert tissue.read_bytes()==restored  # Undo must not silently re-adopt Clean.


def test_stale_legacy_clean_is_archived_without_replacing_recent_edits(tmp_path):
    tissue=annotation.valid_tissue(tmp_path,'s');clean=annotation.cleaned_tissue(tmp_path,'s')
    put(tissue,[feature(box(0,0,100,100))],'tissue');put(clean,[feature(box(0,0,50,100))],'tissue')
    before=tissue.read_bytes();annotation.unify_tissue(tmp_path,'s')
    assert tissue.read_bytes()==before
    saved=edits.history(tissue)['entries'][0]
    assert saved['label']=='Earlier Clean Tissue (archived)'
    edits.restore(tissue,saved['after'],edits.revision(tissue),kind='tissue')
    assert area(tissue).area==5000


def test_subtraction_updates_chosen_class_only_and_keeps_exact_history(tmp_path):
    target=tmp_path/'imported.geojson';red=tmp_path/'clot.geojson'
    tissue=box(.125,.375,100,100).difference(box(70,70,80,80));other=box(10,10,90,90)
    before=put(target,[feature(tissue,'tissue'),feature(other,'tumor','other')])
    artifact=put(red,[feature(box(0,0,30,100),'Blood_Clot')])
    result=edits.subtract(target,[red],before['revision'],kind='regions',class_name='tissue',artifact_revisions={str(red):artifact['revision']})
    found=edits.read(target)[0]
    assert shape(next(f for f in found if f['properties']['id']=='other')['geometry']).equals(other)
    kept=next(f for f in found if f['properties']['id']=='one')
    assert shape(kept['geometry']).equals(tissue.difference(box(0,0,30,100)))
    assert kept['properties']['classification']['name']=='tissue'
    edits.restore(target,before['revision'],result['revision'])
    assert area(target).equals(unary_union([tissue,other]))
    with pytest.raises(ValueError,match='changed'):
        edits.subtract(target,[red],edits.revision(target),artifact_revisions={str(red):'stale'})


def test_shared_scope_follows_image_across_steps_without_crossing_images(tmp_path):
    image=tmp_path/'one.ome.tiff';other=tmp_path/'two.ome.tiff'
    value=inputs.save_selection(tmp_path,{},image,tmp_path,'clot',{'mode':'tissue','class_name':''})
    for step in ('tissue','debris','classes','regions'): assert inputs.selection(tmp_path,image,step)==value
    inputs.save_selection(tmp_path,{},image,tmp_path,'debris',{'mode':'image','class_name':''})
    assert inputs.selection(tmp_path,image,'clot')['mode']=='image'
    assert inputs.selection(tmp_path,other,'clot')['mode']=='tissue'
    with pytest.raises(ValueError,match='Choose a Tissue'):
        inputs.working_path(tmp_path,image,tmp_path)


def test_subtraction_route_uses_same_target_and_only_active_artifact(setup):
    channel,_,root,spec=setup
    image=registry.resolve(spec['layers'][0]['token']);output=annotation.all_folders(root,{})['output'];sample=annotation.sample_id(image)
    target=annotation.valid_tissue(output,sample);initial=put(target,[feature(box(0,0,64,64))],'tissue')
    clot=learning.polygons_path(output,sample,'clot');quality=learning.polygons_path(output,sample,'debris')
    red=put(clot,[feature(box(0,0,20,64))],'artifacts');put(quality,[feature(box(50,0,64,64))],'artifacts')
    client=TestClient(app);prefix='/studio/'+channel
    working=client.post(prefix+'/tissue_versions',json={'stage':'clot','image_token':spec['layers'][0]['token']}).json()
    body={'image_token':spec['layers'][0]['token'],'selection':working['selection'],'tissue_token':working['layer']['token'],
          'tissue_revision':initial['revision'],'reviewed_revision':red['revision']}
    response=client.post(prefix+'/artifact_review/clot/subtract',json=body)
    assert response.status_code==200,response.text
    result=response.json()
    assert result['layer']['saved_file']==str(target.resolve())
    assert area(target).equals(box(20,0,64,64))
    assert not annotation.cleaned_tissue(output,sample).exists()
    assert inputs.scope_path(root,image,output,'debris')==target
    assert inputs.scope_path(root,image,output,'classes')==target
    assert client.post(prefix+'/artifact_review/clot/subtract',json=body).status_code==400
    restored=client.post('/shapes/'+working['layer']['token']+'/history',json={'version':initial['revision'],'revision':result['revision']})
    assert restored.status_code==200 and area(target).area==64*64
