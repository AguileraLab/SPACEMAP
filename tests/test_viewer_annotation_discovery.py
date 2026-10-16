"""History directories named *.geojson must never become viewer annotations."""
import json

import pytest

from spacemap.viewer import catalog, view_state
from spacemap_gui.viewer import registry
from test_viewer import client, _square, _write_channel_pages


@pytest.fixture
def project(tmp_path):
    output = tmp_path / 'outputs/annotations'
    output.mkdir(parents=True)
    real = output / 'HCC_SARC_1_tissue_boundary.geojson'
    real.write_text(json.dumps({'type': 'FeatureCollection', 'features': [_square(0, 0, 0)]}))
    history = output / 'intermediate/studio_history' / real.name
    history.mkdir(parents=True)
    archived = history / 'states' / real.name
    archived.parent.mkdir()
    archived.write_bytes(real.read_bytes())
    cache = output / 'viewer_cache' / real.name
    cache.parent.mkdir()
    cache.write_bytes(real.read_bytes())
    # Not every folder ending in .geojson is history, but none is a data file.
    (output / 'other' / real.name).mkdir(parents=True)
    return tmp_path, real, history, archived


def test_discovery_returns_current_file_not_history_folder_or_snapshots(project):
    root, real, history, archived = project
    found = catalog.find_annotations(root, {}, 'HCC_SARC_1')
    assert [path for _, path in found] == [str(real)]
    # Explicitly mapped history roots must not expose historical versions either.
    assert catalog.find_annotations(root / 'empty', {'annot_output_dir': str(history)}, 'HCC_SARC_1') == []
    assert archived.is_file() and history.is_dir()


def test_project_file_picker_lists_files_and_preserves_archives(client, project):
    root, real, history, archived = project
    token = registry.register_object('workspace', root=str(root), image=str(root / 'image.tif'))
    assert client.get(f'/workspace/{token}/files').json()['files'] == [str(real.relative_to(root))]
    rejected = client.post(f'/workspace/{token}/add', json={'relative': str(history.relative_to(root))})
    assert rejected.status_code == 400
    assert archived.read_bytes() == real.read_bytes()


@pytest.mark.parametrize('endpoint', ['legend', 'tile/0/0/0.png', 'at?x=1&y=1'])
def test_stale_directory_layer_returns_clear_error_instead_of_server_exception(client, project, endpoint):
    root, real, history, _ = project
    image = root / 'image.ome.tiff'
    _write_channel_pages(image, channels=1, width=64, height=64)
    token = registry.register_object('annotations', path=str(history), image=str(image))
    answer = client.get(f'/annotations/{token}/{endpoint}')
    assert answer.status_code == 400 and 'folder' in answer.json()['error'].lower()
    valid = registry.register_object('annotations', path=str(real), image=str(image))
    assert client.get(f'/annotations/{valid}/legend').status_code == 200


def test_restoring_extras_ignores_directory_layers_without_rewriting_saved_history(client, project):
    root, real, history, archived = project
    image = root / 'image.tif'
    workspace = registry.register_object('workspace', root=str(root), image=str(image))
    good = registry.register_object('annotations', path=str(real), image=str(image))
    bad = registry.register_object('annotations', path=str(history), image=str(image))
    extras = [{'id': name, 'type': 'annotations', 'token': token} for name, token in [('real', good), ('history', bad)]]
    view_state.save(root, image, {'extras': extras})
    saved_path = view_state.path_for(root, image)
    before = saved_path.read_bytes()
    restored = client.get(f'/workspace/{workspace}/view').json()['view']['extras']
    assert [layer['id'] for layer in restored] == ['real']
    assert saved_path.read_bytes() == before and archived.is_file()
