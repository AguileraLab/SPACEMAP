"""Exact multi-source subtraction, recoverable labels, and immutable parents."""
import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union
from starlette.testclient import TestClient

from spacemap_gui.studio import annotation, annotation_difference as difference
from spacemap_gui.studio import annotation_edits as edits, annotation_inputs as inputs, annotation_layers as layers
from spacemap_gui.viewer import registry
from spacemap_gui.viewer.api import app
from test_studio_commands import setup


def sources(setup):
    channel, _, root, spec = setup
    image = registry.resolve(spec['layers'][0]['token'])
    output = annotation.all_folders(root, {})['output']
    tissue = annotation.valid_tissue(output, annotation.sample_id(image))
    regions = annotation.marker_regions(output, annotation.sample_id(image))
    extra = output / (annotation.sample_id(image) + '_marker_regions_extra.geojson')
    for path, items in ((tissue, [('tissue', box(0, 0, 60, 60).difference(box(2, 2, 4, 4)))]),
                        (regions, [('Tumor', box(10, 10, 20, 20)), ('Epithelia', box(15, 15, 35, 35)),
                                   ('Keep', box(40, 40, 50, 50))]),
                        (extra, [('', box(50, 0, 64, 10))])):
        edits.replace(path, [{'type': 'Feature', 'geometry': mapping(geom),
                              'properties': {'classification': {'name': name}}} for name, geom in items],
                      edits.revision(path), 'Fixture')
    body = {'image_token': spec['layers'][0]['token'], 'label': 'Stroma',
            'base': [pick(root, image, output, tissue)],
            'subtract': [pick(root, image, output, regions, 'Tumor'), pick(root, image, output, regions, 'Epithelia'),
                         pick(root, image, output, extra)]}
    return root, image, output, [tissue, regions, extra], body


def pick(root, image, output, path, name=None):
    """A name of the catalog (as Layers lists it) held by *path*, as the page sends it."""
    found = [i for i in difference.catalog(root, {}, image, output) if str(Path(path).resolve()) in i['revisions']
             and (name is None or i['name'] == name)]
    assert len(found) == 1, found
    return {'key': found[0]['key'], 'revisions': found[0]['revisions']}


def test_difference_preserves_holes_labels_parents_and_recipe(setup):
    root, image, output, paths, body = sources(setup)
    before = {p: p.read_bytes() for p in paths}
    preview = difference.create(root, {}, image, output, {**body, 'action': 'preview'})
    assert preview['count'] == 1 and preview['outlines'] and not Path(preview['path']).exists()
    result = difference.create(root, {}, image, output, {**body, 'target_revision': preview['target_revision']})
    polygons = edits.read(result['path'])[0]
    expected = box(0, 0, 60, 60).difference(unary_union([box(2, 2, 4, 4), box(10, 10, 20, 20),
                                                      box(15, 15, 35, 35), box(50, 0, 64, 10)]))
    assert unary_union([shape(f['geometry']) for f in polygons]).equals(expected)
    assert polygons[0]['properties']['name'] == 'Stroma'
    assert polygons[0]['properties']['classification']['name'] == 'Stroma'
    assert result['before'] == [] and result['layer']['editable'] and result['layer']['name'] == 'Stroma'
    assert all(p.read_bytes() == before[p] for p in paths)
    recipe = layers.records(root, image)[result['path']]['parent']
    assert recipe['operation'] == 'difference' and len(recipe['subtract']) == 3
    assert all(i['revision'] for side in ('base', 'subtract') for i in recipe[side])
    assert result['path'] in inputs.sources(root, {}, image, output)
    selection = {'mode': 'file', 'path': result['path'], 'class_name': 'Stroma'}
    inputs.save_selection(root, {}, image, output, 'classes', selection)
    assert inputs.prepare_working(root, {}, image, output, 'classes') == selection
    scope = unary_union([shape(f['geometry']) for f in edits.read(inputs.scope_path(root, image, output, 'classes'))[0]])
    assert scope.equals(expected)
    choice = next(item for item in layers.export_options(root, {}, image, output) if item['path'] == result['path'])
    assert layers.export(root, {}, image, output, [choice])['count'] == 1
    edits.restore(result['path'], preview['target_revision'], result['revision'])
    assert not edits.read(result['path'])[0]  # creation can be undone to empty, then restored from History
    edits.restore(result['path'], result['revision'], edits.revision(result['path']))
    assert edits.read(result['path'])[0][0]['properties']['name'] == 'Stroma'


def test_multi_base_overlap_and_invalid_input_are_normalised(setup):
    root, image, output, paths, body = sources(setup)
    # Keep every class on the source side, even overlapping polygons; duplicate choices don't double-count.
    body.update(base=[body['base'][0], body['subtract'][2], body['base'][0]], subtract=[body['subtract'][0]])
    result = difference.create(root, {}, image, output, body)
    assert len(result['recipe']['base']) == 2
    expected = box(0, 0, 60, 60).difference(box(2, 2, 4, 4)).union(box(50, 0, 64, 10)).difference(box(10, 10, 20, 20))
    assert unary_union([shape(f['geometry']) for f in edits.read(result['path'])[0]]).equals(expected)
    invalid = Polygon([(1, 1), (8, 8), (8, 1), (1, 8), (1, 1)])
    # An externally imported invalid polygon is repaired without changing its file.
    paths[2].write_text(json.dumps({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {}, 'geometry': mapping(invalid)}]}))
    body['subtract'] = [pick(root, image, output, paths[2])]
    body['base'] = body['base'][:1]
    result = difference.create(root, {}, image, output, body)
    assert all(shape(f['geometry']).is_valid for f in result['features'])


def test_difference_guards_stale_sources_missing_labels_and_empty_results(setup):
    root, image, output, _, body = sources(setup)
    call = lambda value: difference.create(root, {}, image, output, value)
    with pytest.raises(ValueError, match='changed'):
        call({**body, 'subtract': [{**body['subtract'][0], 'revisions': {p: 'stale' for p in body['subtract'][0]['revisions']}}]})
    with pytest.raises(ValueError, match='no longer'):
        call({**body, 'subtract': [{**body['subtract'][0], 'key': '::Missing::'}]})
    with pytest.raises(ValueError, match='no longer'):
        call({**body, 'base': [{'key': '/another-project/tissue.geojson'}]})
    with pytest.raises(ValueError, match='Nothing remains'):
        call({**body, 'subtract': body['base']})
    with pytest.raises(ValueError, match='each side'):
        call({**body, 'subtract': []})
    preview = call({**body, 'action': 'preview'})
    result = call(body)
    with pytest.raises(ValueError, match='result layer changed'):
        call({**body, 'target_revision': preview['target_revision']})
    # Alter a source during geometry computation: don't commit a stale result.
    from unittest.mock import patch
    old_union = difference.unary_union
    def mutate(geoms):
        path = Path(next(iter(body['base'][0]['revisions'])))
        if path.read_text().endswith('\n') is False:
            path.write_text(path.read_text() + '\n')
        return old_union(geoms)
    with patch.object(difference, 'unary_union', mutate), pytest.raises(ValueError, match='during subtraction'):
        call(body)
    assert edits.revision(result['path']) == result['revision']


def test_difference_api_records_and_rejects_wrong_image_or_readonly(setup):
    root, image, output, _, body = sources(setup)
    channel, _, _, spec = setup
    client = TestClient(app); url = '/studio/' + channel + '/annotation_difference'
    assert client.get(url).status_code == 200
    assert client.post(url, json={**body, 'image_token': 'wrong'}).status_code == 400
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
    from spacemap_gui.studio.annotation_workflow import RELATIVE_PATH
    assert Path(root / RELATIVE_PATH).is_file()
    assert any('subtract_annotations' in path.read_text() for path in
               (root / 'outputs/studio/annotation_workflow/events').glob('*.json'))
    registry.publish(channel, spec, project={'root': str(root), 'mappings': {}, 'read_only': True})
    response = client.post(url, json=body)
    assert response.status_code == 400 and 'viewing only' in response.text
    assert client.post(url, json={**body, 'action': 'preview'}).status_code == 200


def test_catalog_lists_names_as_layers_and_a_renamed_polygon_moves(setup):
    root, image, output, paths, _ = sources(setup)
    names = {i['name']: i['count'] for i in difference.catalog(root, {}, image, output)}
    assert {'Tumor', 'Epithelia', 'Keep'} <= set(names) and names['Tumor'] == 1
    features, version = edits.read(paths[1])
    tumor = next(f for f in features if f['properties']['classification']['name'] == 'Tumor')
    tumor['properties'].update(name='Normal', label='Normal', named=True)  # renamed by hand; its class still says Tumor
    edits.save(paths[1], features, version)
    names = {i['name']: i['count'] for i in difference.catalog(root, {}, image, output)}
    assert 'Tumor' not in names and names['Normal'] == 1  # one name per polygon: it moved to Normal


def test_the_same_subtraction_by_name_on_another_image(setup):
    """Selected: each image uses its own annotations of the chosen names; a minus name it lacks takes nothing away."""
    root, image, output, paths, body = sources(setup)
    names = {item['key']: item for item in difference.catalog(root, {}, image, output)}
    base = [{'name': names[b['key']]['name'], 'role': names[b['key']]['role']} for b in body['base']]
    subtract = [{'name': 'Tumor', 'role': ''}, {'name': 'Absent', 'role': ''}]
    made = difference.create_by_names(root, {}, image, output, 'Stroma', base, subtract)
    polygons = edits.read(made['path'])[0]
    assert unary_union([shape(f['geometry']) for f in polygons]).equals(box(0, 0, 60, 60).difference(unary_union([box(2, 2, 4, 4), box(10, 10, 20, 20)])))
    assert made['missing'] == ['Absent']
    only = difference.create_by_names(root, {}, image, output, 'Copy', base, [{'name': 'Absent', 'role': ''}])
    assert unary_union([shape(f['geometry']) for f in edits.read(only['path'])[0]]).equals(box(0, 0, 60, 60).difference(box(2, 2, 4, 4)))
    with pytest.raises(ValueError, match='it has no Nothing'):
        difference.create_by_names(root, {}, image, output, 'X', [{'name': 'Nothing', 'role': ''}], subtract)
    from spacemap_gui.studio import tasks
    found = tasks.subtract_annotations({'root': str(root), 'images': [image], 'label': 'Stroma2', 'base': base, 'subtract': subtract},
                                       None, lambda *a: None)
    assert found['count'] == 1 and found['images'][0]['missing'] == ['Absent']
