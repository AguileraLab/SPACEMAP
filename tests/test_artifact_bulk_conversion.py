"""A selected batch moves classes without losing geometry or changing Tissue."""
from copy import deepcopy
import json

import pytest
from shapely.geometry import Point, box, mapping, shape

from spacemap_gui.studio import annotation_edits as edits, artifact_learning as learning


def feature(geometry, identifier):
    return {'type': 'Feature', 'geometry': mapping(geometry), 'properties': {
        'id': identifier, 'source': 'model', 'model_version': 'historical-model',
        'classification': {'name': 'Blood_Clot', 'color': [230, 25, 75]}}}


@pytest.mark.parametrize('part', ['clot', 'blur', 'debris'])
@pytest.mark.parametrize('preview', [False, True])
def test_batch_both_directions_preserves_holes_detail_and_provenance(tmp_path, part, preview):
    image = tmp_path/'sample.ome.tiff'
    target = learning.polygons_path(tmp_path/'out', 'sample', part)
    green = learning.example_path(tmp_path, part, image)
    # The display outlines really are simplified, including an unselected one.
    geometries = [Point(x, 200).buffer(75, resolution=1200).difference(box(x-10, 190, x+10, 210)) for x in (100, 300, 500)]
    saved = edits.replace(target, [feature(g, str(i)) for i, g in enumerate(geometries)], edits.revision(target), 'Source', kind='artifacts')
    edits.replace(green, [], edits.revision(green), 'Opened', kind='examples')
    original = target.read_bytes()
    display = saved['features']
    assert all(f['properties'].get('simplified') for f in display)
    result = learning.convert_examples(tmp_path, part, image, tmp_path/'out', ['0', '1', '1'], 'negative',
        saved['revision'], edits.revision(green), preview=display if preview else None)
    assert result['converted'] == 2
    remaining = result['artifact']['features']
    positives = edits.read(target, kind='artifacts')[0] if not preview else remaining
    assert len(positives) == 1 and shape(positives[0]['geometry']).equals(geometries[2])
    negatives = edits.read(green, kind='examples')[0]
    assert len(negatives) == 2
    for f, g in zip(negatives, geometries):
        assert shape(f['geometry']).equals(g) and len(shape(f['geometry']).interiors) == 1
        assert f['properties']['model_version'] == 'historical-model'
        assert f['properties']['example_role'] == 'negative'
    if preview:
        assert target.read_bytes() == original
    reverse = learning.convert_examples(tmp_path, part, image, tmp_path/'out', result['selected'], 'positive',
        result['artifact']['revision'], result['examples']['revision'], preview=remaining if preview else None)
    assert reverse['converted'] == 2 and edits.read(green, kind='examples')[0] == []
    restored = reverse['artifact']['features'] if preview else edits.read(target, kind='artifacts')[0]
    assert len(restored) == 3
    for f, g in zip(restored[1:], geometries):
        assert shape(f['geometry']).equals(g)
        assert f['properties']['source'] == 'manual_artifact'
        assert f['properties']['classification']['name'] == learning.CLASSES[part]
        assert f['properties']['model_version'] == 'historical-model'
    if preview:
        assert target.read_bytes() == original


def test_invalid_batch_or_stale_revision_changes_neither_file(tmp_path):
    image = tmp_path/'sample.ome.tiff'
    target = learning.polygons_path(tmp_path/'out', 'sample', 'clot')
    green = learning.example_path(tmp_path, 'clot', image)
    saved = edits.replace(target, [feature(box(0, 0, 10, 10), 'a'), feature(box(20, 0, 30, 10), 'b')], edits.revision(target), 'Source', kind='artifacts')
    edits.replace(green, [], edits.revision(green), 'Opened', kind='examples')
    original = target.read_bytes(), green.read_bytes()
    for ids, revision in [(['a', 'missing'], saved['revision']), ([], saved['revision']), (['a'], 'stale')]:
        with pytest.raises(ValueError):
            learning.convert_examples(tmp_path, 'clot', image, tmp_path/'out', ids, 'negative', revision, edits.revision(green))
        assert (target.read_bytes(), green.read_bytes()) == original
    broken = deepcopy(saved['features'])
    broken[0]['properties'].update(simplified=True, _studio_geometry='0'*64)
    with pytest.raises(ValueError, match='Reload'):
        learning.convert_examples(tmp_path, 'clot', image, tmp_path/'out', ['a'], 'negative', saved['revision'], edits.revision(green), preview=broken)
    assert (target.read_bytes(), green.read_bytes()) == original
    # A conversion retains an exact source snapshot for recovery, including previews.
    result = learning.convert_examples(tmp_path, 'clot', image, tmp_path/'out', ['a'], 'negative', saved['revision'], edits.revision(green))
    assert json.loads(edits._state_path(target, saved['revision']).read_text())[0]['properties']['id'] == 'a'
    assert result['selected']
