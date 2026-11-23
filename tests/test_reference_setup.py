"""Directory loading, child versions and training/validation settings."""

import hashlib
import json

import pytest

from spacemap_gui.studio import reference_setup, reference_models, manual_reference, auto_reference
from spacemap_gui.viewer import registry
from test_studio_reference_models import training  # noqa: F401
from test_studio_manual_reference import study  # noqa: F401
from test_studio_commands import setup  # noqa: F401


def test_directory_load_preserves_source_and_retraining_creates_child(training, monkeypatch):
    table, settings, root = training
    original = reference_models.train(settings, root, lambda *args: None)['models'][0]
    source = reference_models.model_path(original['model_id'])
    source_bytes = source.read_bytes()
    manifest_bytes = source.with_suffix('.json').read_bytes()
    library = root / 'shared' / 'phenotyping' / ('phenotyping_' + original['model_id'])
    library.mkdir(parents=True)
    (library / source.name).write_bytes(source_bytes)
    (library / 'model.json').write_text(json.dumps({'kind': 'phenotyping', 'id': original['model_id'], 'record': original}))
    imported = root / 'local-models'
    imported.mkdir()
    monkeypatch.setattr(reference_models, 'directory', lambda: imported)
    choices = reference_setup.configure(root, {'action': 'folder', 'folder': str(root / 'shared')}, table)
    assert len(choices['models']) == 1
    loaded = reference_setup.configure(root, {'action': 'load', 'key': choices['models'][0]['key']}, table)
    parent = loaded['base_model_id']
    assert parent != original['model_id']
    parent_path = reference_models.model_path(parent)
    checksum = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    split = reference_setup.configure(root, {'action': 'split', 'training_percent': 80}, table)
    assert split['training_percent'] == 80
    child = reference_models.train({**settings, **reference_setup.training(root)}, root, lambda *args: None)['models'][0]
    assert child['base_model_id'] == parent and child['model_id'] != parent
    assert child['training_percent'] == 80 and child['training_cells'] == 94
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == checksum
    assert source.read_bytes() == source_bytes and source.with_suffix('.json').read_bytes() == manifest_bytes
    assert (library / source.name).read_bytes() == source_bytes
    assert reference_setup.configure(root, {'action': 'load', 'key': ''}, table)['base_model_id'] == ''
    with pytest.raises(ValueError, match='listed'):
        reference_setup.configure(root, {'action': 'load', 'key': '../../bad'}, table)


def test_setup_api_persists_split_for_manual_and_auto_training(study):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    channel, _, spec = study
    auto_reference.configure(channel, 'prediction', True)
    route = '/studio/' + channel + '/reference_setup'
    with TestClient(app) as client:
        assert client.get(route).json()['training_percent'] == 70
        result = client.post(route, json={'action': 'split', 'training_percent': 80})
        assert result.status_code == 200, result.text
        assert manual_reference.training_settings(channel, 'prediction')['training_percent'] == 80
        project, key = auto_reference._context(channel)
        assert auto_reference._read(project['root'])[key]['settings']['training_percent'] == 80
        assert client.post(route, json={'action': 'split', 'training_percent': 100}).status_code == 400
        registry.publish(channel, spec, project={**registry.session_project(channel), 'read_only': True})
        assert client.get(route).status_code == 200
        assert client.post(route, json={'action': 'split', 'training_percent': 70}).status_code == 400


def test_auto_child_does_not_retire_the_selected_parent(training):
    _, settings, root = training
    parent = reference_models.train({**settings, 'auto_train': True}, root, lambda *args: None)['models'][0]
    path = reference_models.model_path(parent['model_id'])
    before = path.read_bytes()
    for _ in range(2):
        child = reference_models.train({**settings, 'auto_train': True, 'base_model_id': parent['model_id']}, root, lambda *args: None)['models'][0]
        assert child['base_model_id'] == parent['model_id']
        assert path.read_bytes() == before


def test_auto_saves_each_version_and_uses_the_latest_for_suggestions(training):
    _, settings, root = training
    options = {**settings, 'project_root': str(root), 'auto_train': True}
    first = reference_models.train(options, root/'auto1', lambda *a: None)['models'][0]
    assert reference_setup.saved(root)['model_id'] == first['model_id']
    first_bytes = reference_models.model_path(first['model_id']).read_bytes()
    second = reference_models.train({**options, **reference_setup.training(root)}, root/'auto2', lambda *a: None)['models'][0]
    assert reference_setup.saved(root)['model_id'] == second['model_id']
    assert second['base_model_id'] == first['model_id']
    assert reference_models.model_path(first['model_id']).read_bytes() == first_bytes
    assert len(list((reference_setup.model_library.location()/'phenotyping').glob('*/model.json'))) == 2
    assert first['trained'] != second['trained']
    result=reference_setup.configure(root, {'action':'share'}, settings['table'])
    selected=next(m for m in result['models'] if m['id']==second['model_id'])
    assert selected['shared_path']


def test_share_failure_keeps_the_trained_model_and_reports_destination_error(training, monkeypatch):
    _, settings, root=training
    def fail(*a, **k):
        raise PermissionError('read-only sharing folder')
    monkeypatch.setattr(reference_setup, 'publish', fail)
    result=reference_models.train({**settings,'project_root':str(root)}, root/'fit', lambda *a:None)
    assert all('saved locally' in m['export_error'] and reference_models.model_path(m['model_id']).is_file() for m in result['models'])


def test_project_library_lists_versions_trained_on_external_image_tables(training):
    from spacemap_gui.studio import model_library
    table, settings, root = training
    project = root/'project'
    project.mkdir()
    fitted = reference_models.train({**settings, 'project_root': str(project)}, root/'fit', lambda *a: None)['models']
    assert table.parent != project and not table.is_relative_to(project/'outputs')
    listed = model_library.project_models(project)
    assert {m['id'] for m in listed if m['kind'] == 'phenotyping'} == {m['model_id'] for m in fitted}


def test_folder_is_exclusive_and_empty_folder_clears_cached_model(training):
    table, settings, root = training
    model = reference_models.train({**settings, 'project_root': str(root)}, root/'fit', lambda *a: None)['models'][0]
    original = reference_models.model_path(model['model_id']).read_bytes()
    assert reference_setup.options(root, table)['base_model_id'] == model['model_id']
    empty = root/'empty'; empty.mkdir()
    changed = reference_setup.configure(root, {'action':'folder', 'folder':str(empty)}, table)
    assert changed['models'] == [] and changed['base_model_id'] == ''
    assert reference_setup.training(root)['base_model_id'] == ''
    assert reference_models.model_path(model['model_id']).read_bytes() == original
    assert not list(empty.iterdir())


def test_readable_version_folder_and_local_time(training):
    from pathlib import Path
    table, settings, root = training
    reference_setup.configure(root, {'action':'training', 'model_name':'Immune panel',
        'include_saved_examples':True, 'timezone':'America/Chicago'}, table)
    model = reference_models.train({**settings, **reference_setup.training(root), 'project_root':str(root)}, root/'fit', lambda *a:None)['models'][0]
    folder = Path(model['shared_path'])
    assert 'Immune_panel__RF__' in folder.name and ('CDT' in folder.name or 'CST' in folder.name)
    assert model['model_id'] not in folder.name
    assert reference_setup.publish(root, model['model_id']) == str(folder)
    # Users may rename or copy the folder; its manifest still identifies it.
    renamed = folder.with_name('My favourite model'); folder.rename(renamed)
    listed = reference_setup.options(root, table)
    item = next(m for m in listed['models'] if m['id'] == model['model_id'])
    assert item['shared_path'] == str(renamed)
