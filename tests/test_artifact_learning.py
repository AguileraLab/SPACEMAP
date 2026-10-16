"""Reviewed artifact models, independent clean tissue, and imported analysis areas."""
import json
import shutil

import numpy as np
import pytest
from shapely.geometry import box, mapping, shape

from spacemap_gui.studio import annotation, annotation_edits as edits, artifact_learning as learning, annotation_inputs as inputs
from test_studio_artifacts import marker_image


def polygon(x0=0,y0=0,x1=128,y1=128,name='Tissue'):
    return {'type':'Feature','properties':{'classification':{'name':name,'colorRGB':0xD04F65}},'geometry':mapping(box(x0,y0,x1,y1))}


@pytest.fixture
def slide(tmp_path):
    rng=np.random.default_rng(4)
    data=np.full((5,128,128),220,dtype=np.uint16)
    data[0,:,:64]=230;data[1,:,:64]=70;data[2,:,:64]=170
    data[3,:,:64]=1200;data[4,:,64:]=1700
    data+=rng.integers(0,15,data.shape,dtype=np.uint16)
    path=tmp_path/'first.ome.tiff'
    marker_image(path,data,['HE_Red','HE_Green','HE_Blue','CD3','PanCK'])
    return path,data


def test_refit_versions_missing_markers_and_reuse(tmp_path,slide,monkeypatch):
    image,data=slide
    tissue=[polygon()];positive=[polygon(0,0,64,128,'Blood_Clot')]
    one=learning.snapshot(tmp_path,'clot',image,2,positive,[],tissue,True)
    assert len(one['examples'])==1
    two=learning.snapshot(tmp_path,'clot',image,2,positive,[],tissue,True)
    assert len(two['examples'])==1 and two['examples'][0]['id']!=one['examples'][0]['id']
    other=tmp_path/'second.ome.tiff';shutil.copyfile(image,other)
    learning.snapshot(tmp_path,'clot',other,2,positive,[],tissue,True)
    report=lambda *args:None
    version=learning.train(learning.training_request(tmp_path,'clot'),report)['version']
    assert version['images']==2 and learning.state(tmp_path,'blur')['versions']==[]
    second=learning.train(learning.training_request(tmp_path,'clot'),report)['version']
    assert second['id']!=version['id'] and len(learning.state(tmp_path,'clot')['versions'])==2
    from datetime import datetime
    stamp = datetime.fromisoformat(second['created'])
    assert second['id'].startswith('clot_' + stamp.strftime('%Y%m%dT%H%M%S_'))
    assert second['name'].endswith(stamp.strftime('%Y-%m-%d_%H:%M:%S.%f_UTC'))
    assert (learning.folder(tmp_path,'clot') / (version['id'] + '.joblib')).is_file()
    saved,_=learning.model(tmp_path,'clot',version['id'])
    # Prediction reuses trained weights, and must never call fit.
    monkeypatch.setattr(type(saved['model']),'fit',lambda *a,**k:pytest.fail('Prediction retrained the forest'))
    found=learning.predict(tmp_path,'clot',version['id'],image,2,tissue,minimum_um2=0)
    assert found['features'] and set(found['matched_markers'])=={'cd3','panck'}
    area=learning._geometry(found['features'])
    assert area.intersection(box(0,0,64,128)).area/area.area>.85
    partial=tmp_path/'partial.ome.tiff';marker_image(partial,data[:4],['HE_Red','HE_Green','HE_Blue','CD3'])
    found=learning.predict(tmp_path,'clot',version['id'],partial,2,tissue,minimum_um2=0)
    assert found['missing_markers']==['panck'] and found['matched_markers']==['cd3']
    he=tmp_path/'he.ome.tiff';marker_image(he,data[:3],['HE_Red','HE_Green','HE_Blue'])
    assert learning.predict(tmp_path,'clot',version['id'],he,2,tissue,minimum_um2=0)['used']=='H&E fallback'
    assert learning.model(tmp_path,'clot',version['id'])[0] is saved


def test_examples_require_clear_labels_and_cancel_before_publish(tmp_path,slide):
    image,_=slide
    with pytest.raises(ValueError,match='overlap'):
        learning.snapshot(tmp_path,'clot',image,2,[polygon()],[polygon()],[polygon()],False)
    with pytest.raises(ValueError,match='Draw an artifact'):
        learning.snapshot(tmp_path,'clot',image,2,[],[],[polygon()],False)
    learning.snapshot(tmp_path,'clot',image,2,[polygon(0,0,64,128)],[],[polygon()],True)
    def stop(*args): raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):learning.train(learning.training_request(tmp_path,'clot'),stop)
    assert learning.state(tmp_path,'clot')['versions']==[]


def test_tile_features_have_no_seams(slide):
    image,_=slide
    reader=learning.Features(image,2)
    full=reader.tile(0,0,64,64)
    np.testing.assert_allclose(reader.tile(10,12,30,35),full[12:47,10:40],atol=1e-6)


def test_clot_exact_requested_scale_and_classic_detector(tmp_path,slide,monkeypatch):
    image,_=slide
    import spacemap.tissue_annotation_utils as utils
    seen={}
    def detector(rgb,tissue,config,mpp,downsample):
        seen.update(shape=rgb.shape,scale=downsample,ratio=config['clot_eosin_ratio'],minimum=config['clot_min_area_mm2'])
        return {'clot_mask':tissue}
    monkeypatch.setattr(utils,'detect_blood_clot',detector)
    result=learning.detect(image,{},'clot',2,2.,[polygon()],100)
    assert seen=={'shape':(64,64,3),'scale':2,'ratio':2.,'minimum':.0001}
    assert learning._geometry(result['features']).area>15000
    assert not list(tmp_path.rglob('*reviewed.geojson'))


def test_legacy_clean_files_retain_history_without_overriding_working_tissue(tmp_path):
    original=annotation.valid_tissue(tmp_path,'a');clean=annotation.cleaned_tissue(tmp_path,'a');artifact=tmp_path/'clot.geojson'
    edits.replace(original,[polygon()],edits.revision(original),'Detected tissue',kind='tissue')
    edits.replace(artifact,[polygon(0,0,64,128,'Blood_Clot')],edits.revision(artifact),'Reviewed',kind='artifacts')
    before=original.read_bytes();version=edits.revision(original)
    result=edits.subtract(original,[artifact],version,target=clean,target_revision=edits.revision(clean))
    assert original.read_bytes()==before and sum(shape(f['geometry']).area for f in result['features'])==8192
    assert annotation.analysis_tissue(tmp_path,'a')==original
    first=edits.revision(clean)
    edits.save(clean,[],first,kind='tissue')
    assert annotation.analysis_tissue(tmp_path,'a')==original
    edits.restore(clean,first,edits.revision(clean),kind='tissue')
    assert json.loads(clean.read_text())['studio']['source_revision']==version
    edits.replace(original,[polygon(0,0,100,100)],version,'Redetect',kind='tissue')
    assert annotation.analysis_tissue(tmp_path,'a')==original
    edits.subtract(original,[artifact],edits.revision(original),target=clean,target_revision=edits.revision(clean))
    assert annotation.analysis_tissue(tmp_path,'a')==original
    edits.restore(clean,first,edits.revision(clean),kind='tissue')
    assert annotation.analysis_tissue(tmp_path,'a')==original


def test_import_keeps_names_colours_holes_and_image_binding(tmp_path,slide):
    image,_=slide
    feature=polygon(name='Tumour');feature['geometry']=mapping(box(10,10,110,110).difference(box(40,40,60,60)))
    target=inputs.import_polygons(tmp_path,image,'QuPath regions.geojson',[feature])
    loaded,_=edits.read(target)
    assert loaded[0]['properties']['classification']['name']=='Tumour'
    assert loaded[0]['properties']['classification']['color']==[208,79,101]
    assert len(loaded[0]['geometry']['coordinates'])==2
    assert inputs.imported(tmp_path,image)==[target]
    assert inputs.imported(tmp_path,tmp_path/'different.tiff')==[]
    original=target.read_bytes()
    choice={'mode':'file','path':str(target.resolve()),'class_name':'Tumour'}
    inputs.save_selection(tmp_path,{},image,tmp_path,'classes',choice)
    scope=inputs.scope_path(tmp_path,image,tmp_path,'classes')
    assert shape(json.loads(scope.read_text())['features'][0]['geometry']).area==9600
    assert target.read_bytes()==original
    with pytest.raises(ValueError,match='belonging'):
        inputs.save_selection(tmp_path,{},tmp_path/'different.tiff',tmp_path,'classes',choice)
    with pytest.raises(ValueError,match='outside'):
        inputs.import_polygons(tmp_path,image,'wrong',[polygon(1000,1000,2000,2000)])


def test_settings_revision_and_independent_artifacts(tmp_path):
    state=learning.state(tmp_path,'clot')
    learning.save_options(tmp_path,'clot',{'scale':2,'value':1.5},state['revision'])
    assert learning.options(tmp_path,'clot')['scale']==2
    assert learning.options(tmp_path,'blur')['scale']==2
    with pytest.raises(ValueError,match='changed'):
        learning.save_options(tmp_path,'clot',{'scale':4},state['revision'])


def test_import_one_annotation_class_as_tissue_preserves_geometry_and_source(slide):
    import copy
    from shapely.geometry import MultiPolygon, Polygon
    image, _ = slide
    tissue = polygon(name='tissue')
    geometry = MultiPolygon([Polygon([(1.25,2.5),(60,2.5),(60,60),(1.25,60)],
                                    holes=[[(10,10),(20,10),(20,20),(10,20)]]),
                             box(80,80,110,110)])
    tissue['geometry'] = mapping(geometry)
    original = [polygon(name='stroma'), tissue, polygon(name='tumor')]
    before = copy.deepcopy(original)
    result = inputs.tissue_import(image, original, 'tissue', source='external.geojson')
    assert len(result) == 2
    from shapely.ops import unary_union
    assert unary_union([shape(f['geometry']) for f in result]).equals(geometry)
    assert len(result[0]['geometry']['coordinates']) == 2
    assert all(f['properties']['classification']['name'] == 'Valid_Tissue' for f in result)
    assert all(f['properties']['tissue_source_class'] == 'tissue' for f in result)
    assert original == before
    assert len(inputs.tissue_import(image, original, None)) == 4


def test_import_tissue_rejects_missing_class_and_wrong_coordinates(slide):
    image, _ = slide
    with pytest.raises(ValueError, match='annotation name'):
        inputs.tissue_import(image, [polygon(name='stroma')], 'tissue')
    with pytest.raises(ValueError, match='outside this image'):
        inputs.tissue_import(image, [polygon(1000,1000,1100,1100,name='tissue')], 'tissue')


@pytest.mark.parametrize('label', ['tissue', 'My tissue'])
def test_scope_menu_names_sources_once_without_changing_saved_scopes(tmp_path, slide, label):
    image, _ = slide
    config = {'tab_configurations': {'tab4_annotation': {annotation.LABEL_KEY: label}}}
    original = annotation.valid_tissue(tmp_path, 'first')
    clean = annotation.cleaned_tissue(tmp_path, 'first')
    edits.replace(original, [polygon()], edits.revision(original), 'Tissue', kind='tissue', label=label)
    imported = inputs.import_polygons(tmp_path, image, 'QuPath tissue.geojson', [polygon(name='tissue')])
    entries = inputs.sources(tmp_path, config, image, tmp_path)
    menu = inputs.source_options(tmp_path, config, image, tmp_path, entries)
    assert [item['name'] for item in menu['tissue']] == [label]
    assert menu['automatic'] == 'tissue' and menu['tissue'][0]['available']
    assert [item['path'] for item in menu['files']] == [str(imported)]
    assert menu['files'][0]['kind'] == 'imported'
    # A previously saved file choice still uses precisely that file/class.
    legacy = {'mode': 'file', 'path': str(original.resolve()), 'class_name': 'Valid_Tissue'}
    inputs.save_selection(tmp_path, config, image, tmp_path, 'clot', legacy)
    scoped = inputs.scope_path(tmp_path, image, tmp_path, 'clot')
    assert shape(json.loads(scoped.read_text())['features'][0]['geometry']).area == 128 * 128
    assert inputs.selection(tmp_path, image, 'clot') == legacy
    edits.subtract(original, [imported], edits.revision(original), target=clean, target_revision=edits.revision(clean))
    menu = inputs.source_options(tmp_path, config, image, tmp_path, inputs.sources(tmp_path, config, image, tmp_path))
    assert menu['automatic'] == 'tissue' and len(menu['tissue']) == 1
    assert inputs.scope_path(tmp_path, image, tmp_path, 'classes') == scoped
    assert str(original) not in [item['path'] for item in menu['files']]
    assert str(clean) not in [item['path'] for item in menu['files']]


def test_forget_examples_keeps_saved_versions(tmp_path,slide):
    image,_=slide
    learning.snapshot(tmp_path,'clot',image,2,[polygon(0,0,64,128)],[],[polygon()],True)
    result=learning.train(learning.training_request(tmp_path,'clot'),lambda *args:None)
    learning.forget_image(tmp_path,'clot',image)
    assert learning.state(tmp_path,'clot')['examples']==[]
    assert learning.model(tmp_path,'clot',result['version']['id'])[0]['model'] is not None


def test_pieces_can_edit_cleaned_without_touching_original(tmp_path):
    from spacemap_gui.studio import pieces
    original=annotation.valid_tissue(tmp_path,'a');clean=annotation.cleaned_tissue(tmp_path,'a');artifact=tmp_path/'artifact.geojson'
    edits.replace(original,[polygon()],edits.revision(original),'Tissue',kind='tissue')
    edits.replace(artifact,[polygon(55,0,65,128)],edits.revision(artifact),'Artifact',kind='artifacts')
    before=original.read_bytes()
    edits.subtract(original,[artifact],edits.revision(original),target=clean,target_revision=edits.revision(clean))
    values={**pieces.settings({}),'filter_by_area':False,'filter_by_inscribed_circle':False,'detect_necks':False,'cut_at_necks':False}
    result=pieces.apply(tmp_path,'a',values,expected=edits.revision(clean),tissue_path=clean)
    assert result['kept']==2 and original.read_bytes()==before
    assert annotation.analysis_tissue(tmp_path,'a')==original


def test_empty_cleaned_is_not_replaced_by_redetected_tissue(tmp_path,slide):
    from spacemap_gui.studio import pixelclasses
    image,_=slide
    original=annotation.valid_tissue(tmp_path,'a');clean=annotation.cleaned_tissue(tmp_path,'a');artifact=tmp_path/'artifact.geojson'
    edits.replace(original,[polygon()],edits.revision(original),'Tissue',kind='tissue')
    edits.replace(artifact,[polygon()],edits.revision(artifact),'Artifact',kind='artifacts')
    edits.subtract(original,[artifact],edits.revision(original),target=clean,target_revision=edits.revision(clean))
    with pytest.raises(ValueError,match='no pixels'):
        pixelclasses.prepare(str(image),{},clean)


def test_geometry_repairs_crossed_outlines_and_names_unusable_polygons():
    from shapely.geometry import Polygon
    crossed = polygon()
    crossed['geometry'] = mapping(Polygon([(0,0),(100,100),(0,100),(100,0),(0,0)]))
    before = json.dumps(crossed)
    repaired = learning._geometry([crossed], 'Work within')
    assert repaired.is_valid and repaired.area == 5000 and json.dumps(crossed) == before
    broken = polygon()
    broken['properties']['name'] = 'green 3'
    broken['geometry'] = mapping(Polygon([(0,0),(10,0),(20,0),(0,0)]))
    with pytest.raises(ValueError, match='Green example polygon.*green 3.*Delete it or redraw'):
        learning._geometry([broken], 'Green example')


@pytest.mark.parametrize('part', ['clot', 'debris'])
@pytest.mark.parametrize('role', ['positive', 'negative'])
@pytest.mark.parametrize('touch', ['point', 'line'])
def test_training_clips_boundary_contacts_without_empty_overlay(tmp_path, slide, part, role, touch):
    """A reviewed region can touch Tissue at a point/edge besides its usable area."""
    from shapely.geometry import Polygon
    image, _ = slide
    contact = (Polygon([(128,128),(140,128),(128,140)]) if touch == 'point'
               else box(128,80,140,120))
    red = [polygon(0,0,64,128), {'type':'Feature', 'properties':{}, 'geometry':mapping(contact)}]
    green = []
    if role == 'negative':
        red, green = green, red
    tissue = [polygon()]
    before = json.dumps([red, green, tissue])
    captured = learning.snapshot(tmp_path, part, image, 2, red, green, tissue, False)
    home = learning.folder(tmp_path, part)
    with np.load(home / (captured['snapshot_id'] + '.npz')) as samples:
        assert len(samples['y']) >= 20
        assert np.all(samples['y'] == (role == 'positive'))
        assert np.all(samples['xy'][:, 0] < 32)
    saved = json.loads((home / (captured['snapshot_id'] + '.json')).read_text())
    assert saved['annotations'] == json.loads(json.dumps(
        {'positive':red, 'negative':green, 'tissue':tissue}))
    assert json.dumps([red, green, tissue]) == before


def test_retained_examples_restore_once_into_editable_coloured_layer(tmp_path, slide):
    image, _ = slide
    learning.snapshot(tmp_path, 'clot', image, 2, [polygon(0,0,64,128)], [polygon(64,0,128,128)], [polygon()], False)
    path = learning.open_examples(tmp_path, 'clot', image)
    found, version = edits.read(path, kind='examples')
    assert len(found) == 2
    assert {f['properties']['example_role'] for f in found} == {'positive', 'negative'}
    assert {tuple(f['properties']['classification']['color']) for f in found} == {(230,25,75),(46,204,113)}
    with pytest.raises(ValueError, match='keep the two labels separate'):
        edits.edit(path, [f['properties']['id'] for f in found], 'merge', version, kind='examples')
    edits.save(path, [], version, kind='examples')
    learning.open_examples(tmp_path, 'clot', image)
    assert not edits.read(path, kind='examples')[0]
    assert not learning.example_path(tmp_path, 'blur', image).exists()
    assert learning.example_path(tmp_path, 'clot', tmp_path/'other.tiff') != path


def test_artifact_only_training_then_supervised_from_another_image(tmp_path, slide):
    image, data = slide
    red = [polygon(0, 0, 64, 128, 'Artifact')]
    learning.snapshot(tmp_path, 'clot', image, 2, red, [], [polygon()], False)
    first = learning.train(learning.training_request(tmp_path, 'clot'), lambda *a: None)['version']
    assert first['mode'] == 'artifact_only'
    assert 'marker:cd3' in first['feature_names'] and 'marker:panck' in first['feature_names']
    result = learning.predict(tmp_path, 'clot', first['id'], image, 2, [polygon()], minimum_um2=0)
    assert result['mode'] == 'artifact_only' and result['features']
    other = tmp_path / 'good.ome.tiff'; shutil.copyfile(image, other)
    # A good-only image is also a useful contribution to the shared library.
    learning.snapshot(tmp_path, 'clot', other, 2, [], [polygon(70, 0, 128, 128)], [polygon()], False)
    second = learning.train(learning.training_request(tmp_path, 'clot'), lambda *a: None)['version']
    assert second['mode'] == 'supervised' and second['images'] == 2
    assert learning.state(tmp_path, 'clot')['versions'][0]['id'] == first['id']
    snapshot = json.loads((learning.folder(tmp_path, 'clot') / (first['examples'][0] + '.json')).read_text())
    assert snapshot['preprocessing']['pixel_size_um'] > 0 and snapshot['software']['packages']['scikit-learn']
    with np.load(learning.folder(tmp_path, 'clot') / (first['examples'][0] + '.npz')) as values:
        assert len(values['xy']) == len(values['y']) and set(values['y']) == {1}


def test_marker_only_artifact_training(tmp_path, slide):
    image, data = slide
    marker_image(image, data[3:], ['CD3', 'PanCK'])
    learning.snapshot(tmp_path, 'clot', image, 2, [polygon(0, 0, 64, 128)], [], [polygon()], False)
    version = learning.train(learning.training_request(tmp_path, 'clot'), lambda *a: None)['version']
    assert not version['he'] and version['markers'] == ['cd3', 'panck']
    assert learning.predict(tmp_path, 'clot', version['id'], image, 2, [polygon()], minimum_um2=0)['features']


def test_combine_legacy_examples_and_convert_prediction(tmp_path, slide):
    image, _ = slide
    output = tmp_path / 'output'
    red = {**polygon(0, 0, 64, 128, 'Artifact'), 'properties': {'example_role': 'positive'}}
    green = {**polygon(70, 0, 128, 128, 'Good tissue'), 'properties': {'example_role': 'negative'}}
    path = learning.example_path(tmp_path, 'clot', image)
    edits.replace(path, [red, green], edits.revision(path), 'Examples', kind='examples')
    target = learning.polygons_path(output, 'first', 'clot')
    with pytest.raises(ValueError, match='Save the artifact preview'):
        learning.combine_examples(tmp_path, 'clot', image, output, preview_active=True)
    assert learning.combine_examples(tmp_path, 'clot', image, output)
    assert not learning.combine_examples(tmp_path, 'clot', image, output)
    features, revision = edits.read(target, kind='artifacts')
    examples, green_revision = edits.read(path, kind='examples')
    assert len(features) == len(examples) == 1 and examples[0]['properties']['example_role'] == 'negative'
    before = target.read_bytes()
    result = learning.mark_good(tmp_path, 'clot', image, output, features[0]['properties']['id'], revision, green_revision, preview=features)
    assert result['artifact']['features'] == [] and target.read_bytes() == before
    assert len(result['examples']['features']) == 2
    assert shape(edits.read(path)[0][-1]['geometry']).equals(shape(features[0]['geometry']))
    with pytest.raises(ValueError, match='changed'):
        learning.mark_good(tmp_path, 'clot', image, output, features[0]['properties']['id'], revision, green_revision)


def test_empty_good_collection_never_restores_predictions_as_manual_drawings(tmp_path, slide):
    image, _ = slide
    path = learning.open_examples(tmp_path, 'clot', image)
    assert path.is_file() and edits.read(path)[0] == []
    learning.snapshot(tmp_path, 'clot', image, 2, [polygon(0, 0, 64, 128)], [], [polygon()], False)
    before = path.read_bytes()
    learning.open_examples(tmp_path, 'clot', image)
    assert path.read_bytes() == before and edits.read(path)[0] == []


def test_a_moved_polygon_takes_the_colour_of_the_side_it_joins(tmp_path, slide):
    """Red clots coloured #ff00ff in Layers, good tissue #00ffff: marking a clot as good makes it #00ffff, and back."""
    image, _ = slide
    output = tmp_path / 'output'
    clots = learning.polygons_path(output, 'first', 'clot')
    clots.parent.mkdir(parents=True, exist_ok=True)
    edits.replace(clots, [{**polygon(0, 0, 30, 30, 'Blood_Clot'), 'properties': {'colour': '#ff00ff'}},
                          {**polygon(40, 0, 70, 30, 'Blood_Clot'), 'properties': {'colour': '#ff00ff'}}], edits.revision(clots), 'found', kind='artifacts')
    examples = learning.example_path(tmp_path, 'clot', image)
    edits.replace(examples, [{**polygon(80, 0, 120, 30, 'Good tissue'), 'properties': {'example_role': 'negative', 'colour': '#00ffff'}}],
                  edits.revision(examples), 'drawn', kind='examples')
    red, revision = edits.read(clots, kind='artifacts')
    green_revision = edits.revision(examples)
    result = learning.mark_good(tmp_path, 'clot', image, output, red[0]['properties']['id'], revision, green_revision)
    moved = next(f for f in result['examples']['features'] if f['properties'].get('corrected_artifact_id') == red[0]['properties']['id'])
    assert moved['properties']['colour'] == '#00ffff'
    back = learning.convert_examples(tmp_path, 'clot', image, output, [moved['properties']['id']], 'positive',
                                     edits.revision(clots), edits.revision(examples))
    assert next(f for f in back['artifact']['features'] if f['properties'].get('corrected_example_id') == moved['properties']['id'])['properties']['colour'] == '#ff00ff'
