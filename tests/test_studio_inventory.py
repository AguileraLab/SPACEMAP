"""Lists uses current workflow sources, preserves duplicates and removes exact files."""
import json
from pathlib import Path
import zipfile

import pytest
from starlette.testclient import TestClient

from spacemap_gui.studio import inventory, annotation_inputs, annotation_names, segmentation, project_files
from spacemap_gui.studio.settings import read_config
from spacemap_gui.viewer import api
from test_studio_commands import setup  # noqa: F401


def polygons(name='Tissue'):
    return [dict(type='Feature', properties={'classification': {'name':name, 'color':[50,180,110]}},
                 geometry={'type':'Polygon','coordinates':[[[0,0],[20,0],[20,20],[0,20],[0,0]]]})]


def test_lists_index_is_lazy_and_duplicates_remain_distinct(setup, monkeypatch):
    channel, table, root, spec = setup
    image=root/'outputs/rois/P11_roi_2.ome.tiff'
    first=annotation_inputs.import_polygons(root,image,'review.geojson',polygons())
    second=annotation_inputs.import_polygons(root,image,'review.geojson',polygons('Stroma'))
    with TestClient(api.app) as client:
        result=client.get(f'/studio/{channel}/lists')
        assert result.status_code == 200
        assert [s['name'] for s in result.json()['steps']][:3] == ['Project','Registration','Background']
        saved=annotation_names._scan
        monkeypatch.setattr(annotation_names,'_scan',lambda *a:pytest.fail('Closed lists must not scan polygon content'))
        assert client.get(f'/studio/{channel}/lists?step=tab4').status_code==200
        monkeypatch.setattr(annotation_names,'_scan',saved)
        response=client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()
        rows=response['items']
        assert len(rows)==2 and len({r['id'] for r in rows})==2
        assert {r['path'] for r in rows}=={str(first),str(second)}
        assert {r['classes'][0]['name'] for r in rows}=={'Tissue','Stroma'}
        assert all(r['name']=='review.geojson' and r['count']==1 for r in rows)
        removed=client.post(f'/studio/{channel}/lists',json=dict(action='remove',step='tab4',group='imports',ids=[rows[0]['id']]))
        assert removed.status_code==200,removed.text
        assert Path(rows[0]['path']).exists() and Path(rows[1]['path']).exists()
        assert project_files.excluded(rows[0]['path'],root)
        assert Path(rows[0]['path']) not in annotation_inputs.imported(root,image)
        sources=client.get(f'/studio/{channel}/annotation_sources?layers=1')
        assert sources.status_code==200,sources.text
        assert rows[0]['path'] not in {l.get('saved_file') for l in sources.json()['layers']}
        assert len(client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items'])==1
        areas=segmentation.saved_annotations(root,read_config(root),{},[image.name])[image.name]
        assert rows[0]['path'] not in json.dumps(areas) and rows[1]['path'] in json.dumps(areas)
        assert client.post(f'/studio/{channel}/lists',json=dict(action='remove',step='tab4',group='imports',ids=[rows[0]['id']])).status_code==400


def test_upload_matches_images_without_open_image_and_keeps_duplicates(setup):
    channel, table, root, spec=setup
    with TestClient(api.app) as client:
        for _ in range(2):
            response=client.post(f'/studio/{channel}/lists/upload?step=tab4&group=imports&name=P11_roi_2.geojson',content=json.dumps({'features':polygons()}))
            assert response.status_code==200,response.text
        rows=client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items']
        assert len(rows)==2
        result=client.post(f'/studio/{channel}/lists',json=dict(action='open',step='tab4',group='imports',ids=[rows[0]['id']]))
        assert result.status_code==200,result.text
        assert result.json()['item']['viewer_image']['name']=='P11_roi_2.ome.tiff'
        assert result.json()['item']['annotation_layer']['saved_file']==rows[0]['path']
        response=client.post(f'/studio/{channel}/lists/upload?step=tab4&group=imports&name=unmatched.geojson',content=json.dumps({'features':polygons()}))
        assert response.status_code==200 and response.json()['unmatched']==['unmatched.geojson']
        unmatched=next(r for r in client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items'] if not r['image'])
        response=client.post(f'/studio/{channel}/lists',json=dict(action='match',step='tab4',group='imports',ids=[unmatched['id']],image='P11_roi_2.ome.tiff'))
        assert response.status_code==200,response.text
        assert all(r['image'] for r in client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items'])


def test_csv_opens_exact_table_with_image_and_archive_includes_sidecars(setup):
    channel,table,root,spec=setup
    path=root/'outputs/segmentation/P11_roi_2.ome_cell_features.csv'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('label,x,y\n1,2,3\n')
    Path(str(path)+'.labels.parquet').write_bytes(b'labels')
    with TestClient(api.app) as client:
        rows=client.get(f'/studio/{channel}/lists?step=tab5&group=measurements').json()['items']
        assert len(rows)==1 and rows[0]['image'].endswith('P11_roi_2.ome.tiff')
        result=client.post(f'/studio/{channel}/lists',json=dict(action='open',step='tab5',group='measurements',ids=[rows[0]['id']])).json()
        assert result['item']['file_token'] and result['item']['viewer_image']['name']=='P11_roi_2.ome.tiff'
        export=client.post(f'/studio/{channel}/lists',json=dict(action='export',step='tab5',group='measurements',ids=[rows[0]['id']])).json()
        response=client.get(export['download']);assert response.status_code==200
        import io
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            assert set(bundle.namelist())=={path.name,path.name+'.labels.parquet'}
        assert client.post(f'/studio/{channel}/lists',json=dict(action='remove',step='tab5',group='measurements',ids=[rows[0]['id']])).status_code==200
        assert path.read_text()=='label,x,y\n1,2,3\n'
        assert Path(str(path)+'.labels.parquet').read_bytes()==b'labels'
        assert not client.get(f'/studio/{channel}/lists?step=tab5&group=measurements').json()['items']


def test_linked_removal_keeps_original_and_is_project_local(setup,tmp_path):
    root=setup[2]
    source=tmp_path/'linked.csv';source.write_text('a\n1\n')
    row=inventory._row({'id':'tables','role':'input'},source)
    inventory.remove(root,read_config(root),[row])
    assert source.is_file() and project_files.excluded(source,root)
    assert not project_files.excluded(source,tmp_path/'another-project')


def test_archive_rejects_traversal_and_wrong_type_before_writing(setup,tmp_path):
    root=setup[2];archive=tmp_path/'model.zip'
    with zipfile.ZipFile(archive,'w') as bundle:
        bundle.writestr('model.json',json.dumps({'kind':'cellpose','files':['../escaped']}))
        bundle.writestr('../escaped',b'weights')
    with pytest.raises(ValueError,match='invalid filenames'):
        inventory.import_files(root,read_config(root),'tab5','seg_models',[(archive,'model.zip')])
    assert not (root/'outputs/segmentation/cellpose_models/escaped').exists()
    source=tmp_path/'bad.csv';source.write_text('a\n1')
    with pytest.raises(ValueError,match='wrong file type'):
        inventory.import_files(root,read_config(root),'tab1','codex',[(source,'bad.csv')])


def test_model_export_import_and_project_options_match(setup,tmp_path):
    from spacemap_gui.studio import reference_setup
    root=setup[2];model=root/'data/reference_models'/('a'*32+'.joblib')
    model.parent.mkdir(parents=True);model.write_bytes(b'not deserialized during listing')
    model.with_suffix('.json').write_text(json.dumps(dict(method='Random Forest',markers=['M0'],name='Immune model',classes=['T','B'])))
    rows=inventory.listing(root,read_config(root),'tab7','ref_models')['items']
    assert len(rows)==1 and rows[0]['name']=='Immune model'
    assert reference_setup.options(root)['models'][0]['name']=='Immune model'
    target=tmp_path/'export.zip';inventory.export_bundle(rows,target)
    other=tmp_path/'other';other.mkdir()
    inventory.import_files(other,{},'tab7','ref_models',[(target,'export.zip')])
    assert reference_setup.options(other)['models'][0]['name']=='Immune model'
    inventory.remove(root,read_config(root),rows)
    assert not reference_setup.options(root)['models']
    assert reference_setup.options(other)['models']


def test_orphan_imports_are_listed_and_removable(setup):
    root=setup[2];path=root/'outputs/studio/imported_annotations/orphan/abcd_old.geojson'
    path.parent.mkdir(parents=True);path.write_text(json.dumps({'features':polygons()}))
    rows=inventory.listing(root,read_config(root),'tab4','imports')['items']
    assert len(rows)==1 and rows[0]['image']==''
    inventory.remove(root,read_config(root),rows)
    assert path.is_file() and project_files.excluded(path,root)
    assert not inventory.listing(root,read_config(root),'tab4','imports')['items']


def test_folder_import_links_all_images_unmatched_and_duplicates_without_copying(setup,tmp_path):
    import shutil
    from spacemap_gui.studio import annotation_links
    channel,table,root,spec=setup
    image=root/'outputs/rois/P11_roi_2.ome.tiff'
    second=image.with_name('P12_roi_1.ome.tiff');shutil.copy2(image,second)
    folder=tmp_path/'geojson';folder.mkdir();(folder/'nested').mkdir()
    first=folder/'P11_roi_2.geojson';first.write_text(json.dumps({'features':polygons()}))
    other=folder/'P12_roi_1.geojson';other.write_text(json.dumps({'features':polygons('Stroma')}))
    nested=folder/'nested/P12_roi_1.geojson';nested.write_text(json.dumps({'features':polygons('Nested')}))
    unassigned=folder/'unknown.geojson';unassigned.write_text(json.dumps({'features':polygons()}))
    (folder/'broken.geojson').write_text('not JSON')
    (folder/'notes.txt').write_text('not an annotation')
    (folder/'named.json').write_text('{"settings":true}')
    history=folder/'studio_history/old.geojson/states';history.mkdir(parents=True)
    (history/'snapshot.geojson').write_text(json.dumps({'features':polygons()}))
    (history/'snapshot.json').write_text(json.dumps({'features':polygons()}))
    originals={p:p.read_bytes() for p in [first,other,unassigned]}
    with TestClient(api.app) as client:
        response=client.post(f'/studio/{channel}/lists',json=dict(action='import',step='tab4',group='imports',paths=[str(folder)],image=image.name))
        assert response.status_code==200,response.text
        data=response.json()
        assert len(data['added'])==3 and data['unmatched']==['unknown.geojson'] and len(data['errors'])==1
        assert set(data['added'])=={str(p) for p in originals}
        assert not list((root/'outputs/studio/imported_annotations').glob('*/*.geojson'))
        rows=client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items']
        assert len(rows)==3 and all(r['linked'] for r in rows)
        assert next(r for r in rows if r['path']==str(other))['image']==str(second)
        sources=annotation_inputs.sources(root,read_config(root),second,root/'outputs/processed_annotations/stage0_tissue_annotation')
        assert str(other) in sources and str(first) not in sources
        opened=client.post(f'/studio/{channel}/lists',json=dict(action='open',step='tab4',group='imports',ids=[next(r['id'] for r in rows if r['path']==str(other))])).json()['item']
        assert opened['viewer_image']['name']==second.name
        assert opened['annotation_layer']['type']=='annotations'  # Explicit Edit a copy, no editable original.
        assert client.post(f'/studio/{channel}/lists',json=dict(action='import',step='tab4',group='imports',paths=[str(first)])).status_code==200
        rows=client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items']
        duplicates=[r for r in rows if r['path']==str(first)]
        assert len(duplicates)==2 and duplicates[0]['id']!=duplicates[1]['id']
        response=client.post(f'/studio/{channel}/lists',json=dict(action='remove',step='tab4',group='imports',ids=[duplicates[0]['id']]))
        assert response.json()['paths']==[]  # Remaining reference still supplies the source.
        rows=client.get(f'/studio/{channel}/lists?step=tab4&group=imports').json()['items']
        assert client.post(f'/studio/{channel}/lists',json=dict(action='remove',step='tab4',group='imports',ids=[r['id'] for r in rows])).status_code==200
        assert not annotation_links.records(root)
    assert all(p.read_bytes()==content for p,content in originals.items())


def test_link_removal_preserves_original_inside_project(setup):
    from spacemap_gui.studio import annotation_links
    root=setup[2];path=root/'outputs/processed_annotations/P11_roi_2.geojson'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'features':polygons()}))
    original=path.read_bytes()
    inventory.import_paths(root,read_config(root),'tab4','imports',[str(path)])
    rows=inventory.listing(root,read_config(root),'tab4','imports')['items']
    inventory.remove(root,read_config(root),rows)
    assert path.read_bytes()==original and not annotation_links.records(root)


def test_discovered_import_removal_preserves_original_inside_outputs(setup):
    """A discovered original has no link_id, but is still not an owned copy."""
    root=setup[2];image=root/'outputs/rois/P11_roi_2.ome.tiff'
    folder=root/'outputs/processed_annotations/new'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'P11_roi_2.geojson';path.write_text(json.dumps({'features':polygons()}))
    original=path.read_bytes()
    config=read_config(root)
    rows=inventory.listing(root,config,'tab4','imports')['items']
    row=next(r for r in rows if r['path']==str(path))
    assert row['linked'] and not row.get('link_id')
    from spacemap_gui.studio import annotation
    output=annotation.all_folders(root,config.get('directory_mappings',{}))['output']
    annotation_inputs.save_selection(root,config,image,output,'clot',dict(mode='file',path=str(path)))
    result=inventory.remove(root,config,[row])
    assert result['paths']==[str(path)]
    assert path.read_bytes()==original
    assert project_files.excluded(path,root)
    assert str(path) in annotation_names.hidden_files(root,image)
    assert not inventory.listing(root,config,'tab4','imports')['items']
    assert not list((root/'outputs/studio/deleted').glob('lists_*'))
    with pytest.raises(ValueError,match='removed from the project'):
        annotation_inputs.scope_path(root,image,output,'clot')
    with pytest.raises(ValueError,match='removed from the project'):
        annotation_inputs.working_path(root,image,output,'clot')


def test_input_inside_project_is_unlisted_without_moving_file(setup):
    root=setup[2];source=root/'data/input.csv';source.parent.mkdir(exist_ok=True)
    source.write_text('a\n1\n');original=source.read_bytes()
    inventory.remove(root,read_config(root),[inventory._row({'id':'tables','role':'input'},source)])
    assert source.read_bytes()==original and project_files.excluded(source,root)


def test_artifact_model_unlisted_from_workflow_keeps_weights_and_metadata(setup):
    from spacemap_gui.studio import artifact_learning
    root=setup[2];home=artifact_learning.folder(root,'clot');home.mkdir(parents=True,exist_ok=True)
    weights=home/'fixture.joblib';weights.write_bytes(b'original weights')
    metadata=home/'versions.json';metadata.write_text(json.dumps([{'id':'fixture','created':'2026-10-23','image_names':[]}]))
    before={p:p.read_bytes() for p in (weights,metadata)}
    rows=inventory.listing(root,read_config(root),'tab4','artifact_models')['items']
    inventory.remove(root,read_config(root),rows)
    assert all(p.read_bytes()==content for p,content in before.items())
    assert not artifact_learning.state(root,'clot')['versions']
    with pytest.raises(ValueError,match='removed from this project'):
        artifact_learning.model(root,'clot','fixture')


def test_display_provenance_distinguishes_generated_outputs_from_explicit_imports(setup):
    from spacemap_gui.studio import annotation_links
    root=setup[2];image=root/'outputs/rois/P11_roi_2.ome.tiff'
    path=root/'outputs/processed_annotations/working_files/intermediate/P11_roi_2_clot_reviewed.geojson'
    path.parent.mkdir(parents=True);path.write_text(json.dumps({'features':polygons()}))
    assert annotation_inputs.display_metadata(root,image,path)['annotation_category']=='Blood clots'
    annotation_links.register(root,path,image)
    metadata = annotation_inputs.display_metadata(root, image, path)
    assert metadata['annotation_origin'] == 'imported'
    assert 'annotation_category' not in metadata
    assert metadata['annotation_display_family']
