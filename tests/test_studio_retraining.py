"""Selected saved labels are retrained with fresh image inputs, without altering tiles."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacemap_gui.studio import segmentation, training
from test_studio_artifacts import marker_image


@pytest.fixture
def scene(tmp_path):
    image = tmp_path / 'outputs/rois/roi.ome.tiff'
    image.parent.mkdir(parents=True)
    planes = np.stack([np.full((96, 96), value, np.uint16) for value in (10, 20, 70)])
    names = ['DAPI', 'Border', 'Hematoxylin']
    marker_image(image, planes, names)
    old = {**segmentation.DEFAULTS, 'tab4_last_nuclear_names': ['DAPI']}
    training.start(tmp_path, 'reuse', 'nuclei')
    tiles = []
    for x, y, role in ((0, 0, 'train'), (32, 0, 'train'), (0, 32, 'train'), (32, 32, 'validate')):
        bounds = [x, y, 32, 32]
        cells = [[[x+4, y+4], [x+20, y+4], [x+20, y+20], [x+4, y+20]]]
        tiles.append(training.save_example(tmp_path, image, names, bounds, cells, old, role=role)['saved'])
    new = {**old, 'tab4_last_nuclear_names': ['DAPI', 'Hematoxylin'], 'tab4_last_membrane_names': ['Border']}
    config = {'tab_configurations': {'tab5_segmentation': new}}
    return tmp_path, image, tiles, config


def test_selected_round_uses_fresh_channels_latest_weights_and_unchanged_labels(scene, monkeypatch):
    from cellpose import train as engine
    root, image, tiles, config = scene
    source, models = training.folders(root)
    # Simulate the most recent completed round. Segmentation's active model is independent.
    latest = models / 'models/reuse_v1'
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b'fixture weights')
    session = training.session(root)
    session.update(current=str(latest), versions=[{'name': 'reuse_v1', 'path': str(latest)}])
    training._write(source / 'session.json', session)
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.name != 'session.json'}
    # Legacy records had no source path: they must still resolve across all saved tiles.
    entries = training._read(source / 'examples.json', [])
    entries[0].pop('image_path')
    training._write(source / 'examples.json', entries)
    before['examples.json'] = (source / 'examples.json').read_bytes()
    settings = training.retrain_settings(root, [tiles[0], tiles[2]], config, {}, 2)
    # Later UI changes must not change the submitted feature settings.
    settings = copy.deepcopy(settings)
    config['tab_configurations']['tab5_segmentation']['tab4_last_nuclear_names'] = ['DAPI']
    calls, bases, scores, messages = [], [], [], []
    net = SimpleNamespace(diam_labels=SimpleNamespace(item=lambda: 12))
    monkeypatch.setattr(training, '_model', lambda base, gpu: bases.append(base) or SimpleNamespace(net=net))
    def capture(net, **kw):
        calls.append(kw)
        path = Path(kw['save_path']) / 'models' / kw['model_name']
        path.write_bytes(b'new weights')
        return str(path), [0.2], []
    monkeypatch.setattr(engine, 'train_seg', capture)
    monkeypatch.setattr(training, 'score', lambda model, pictures, truths: scores.append((pictures, truths)) or 0.8)
    result = training.train(settings, root / 'job', lambda message, *args: messages.append(message))
    assert result['name'] == 'reuse_v2' and result['from'] == str(latest) and bases[0] == str(latest)
    assert result['training_tile_names'] == sorted([tiles[0], tiles[2]])
    assert result['validation_tile_names'] == [tiles[3]]
    assert len(calls[0]['train_data']) == 2 and len(scores) == 2
    for picture in calls[0]['train_data'] + scores[0][0]:
        np.testing.assert_array_equal(picture[2], 70)  # newly selected H&E stain participates
        np.testing.assert_array_equal(picture[0], 20)  # newly selected border channel
    for name, labels in zip(result['training_tile_names'], calls[0]['train_labels']):
        np.testing.assert_array_equal(labels, tifffile.imread(source / f'{name}_masks.tif'))
    assert all((source / name).read_bytes() == content for name, content in before.items())
    assert training.session(root)['current'] == result['model']
    assert latest.read_bytes() == b'fixture weights'
    manifest = json.loads(Path(result['input_manifest']).read_text())
    assert len(manifest['tiles']) == 3 and all(item['mask_sha256'] for item in manifest['tiles'])
    assert any(str(image) in message for message in messages)


@pytest.mark.parametrize('selection', [[], ['missing'], [None], 'not-a-list'])
def test_invalid_selection_rejected(scene, selection):
    root, _, _, config = scene
    with pytest.raises(ValueError):
        training.retrain_settings(root, selection, config, {})


def test_validation_is_never_training_and_missing_source_is_not_cached_fallback(scene):
    root, image, tiles, config = scene
    with pytest.raises(ValueError, match='validation'):
        training.retrain_settings(root, [tiles[-1]], config, {})
    image.unlink()
    with pytest.raises(ValueError, match='Source image missing'):
        training.retrain_settings(root, [tiles[0]], config, {})


def test_missing_channel_or_changed_mask_stops_before_training(scene, monkeypatch):
    root, _, tiles, config = scene
    source, _ = training.folders(root)
    monkeypatch.setattr(training, '_model', lambda *a: pytest.fail('Must fail before loading any model'))
    config['tab_configurations']['tab5_segmentation']['tab4_last_membrane_names'] = ['Absent H&E']
    settings = training.retrain_settings(root, [tiles[0]], config, {})
    with pytest.raises(ValueError, match='selected input channels missing: Absent H&E'):
        training.train(settings, root / 'job', lambda *a: None)
    config['tab_configurations']['tab5_segmentation']['tab4_last_membrane_names'] = ['Border']
    settings = training.retrain_settings(root, [tiles[0]], config, {})
    (source / f'{tiles[0]}_masks.tif').write_bytes(b'changed after request')
    with pytest.raises(ValueError, match='corrected cells changed'):
        training.train(settings, root / 'job', lambda *a: None)


def test_changed_training_session_stops_old_request(scene):
    root, _, tiles, config = scene
    settings = training.retrain_settings(root, [tiles[0]], config, {})
    training.start(root, 'different', 'cyto')
    with pytest.raises(ValueError, match='training model changed'):
        training.train(settings, root / 'job', lambda *a: None)


def test_retrain_api_snapshots_selection_and_saved_inputs(setup, monkeypatch):
    from starlette.testclient import TestClient
    from spacemap_gui.studio import jobs
    from spacemap_gui.viewer import api
    channel, _, root = setup[:3]
    image = root / 'outputs/rois/P11_roi_2.ome.tiff'
    training.start(root, 'api_check', 'nuclei')
    saved = training.save_example(root, image, ['M0', 'M1'], [0, 0, 32, 32],
        [[[4, 4], [20, 4], [20, 20], [4, 20]]],
        {**segmentation.DEFAULTS, 'tab4_last_nuclear_names': ['M0']})
    (root / f'{root.name}.json').write_text(json.dumps({'tab_configurations': {'tab5_segmentation': {
        'tab4_last_nuclear_names': ['M0', 'M1'], 'tab4_gpu': False}}}))
    captured = []
    monkeypatch.setattr(jobs, 'start', lambda root, kind, settings: captured.append((kind, settings)) or {'id': 'mock', 'status': 'starting'})
    with TestClient(api.app) as client:
        route = f'/studio/{channel}/training/retrain'
        assert client.post(route, json={'names': [], 'epochs': 1}).status_code == 400
        response = client.post(route, json={'names': [saved['saved']], 'epochs': 4})
    assert response.status_code == 200, response.text
    assert len(captured) == 1
    kind, request = captured[0]
    assert kind == 'train_cellpose' and request['refresh_inputs'] and request['n_epochs'] == 4
    assert request['input_settings']['tab4_last_nuclear_names'] == ['M0', 'M1']
    assert [item['name'] for item in request['tiles']] == [saved['saved']]


from test_studio_commands import setup  # noqa: E402, F401 (fixture)


def test_missing_marker_training_preserves_dapi_labels_and_separate_scores(scene, monkeypatch):
    from cellpose import train as engine
    root, image, tiles, config = scene
    source, _ = training.folders(root)
    config['tab_configurations']['tab5_segmentation']['tab4_last_membrane_names'].append('Absent marker')
    settings = training.retrain_settings(root, [tiles[0]], config, {}, 1, missing_markers=True)
    masks_before = {path.name: path.read_bytes() for path in source.glob('*.tif')}
    calls, scored = [], []
    net = SimpleNamespace(diam_labels=SimpleNamespace(item=lambda: 12))
    monkeypatch.setattr(training, '_model', lambda *args: SimpleNamespace(net=net))
    def capture(net, **kw):
        calls.append(kw)
        path = Path(kw['save_path']) / 'models' / kw['model_name']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'new weights')
        return str(path), [0.2], []
    monkeypatch.setattr(engine, 'train_seg', capture)
    monkeypatch.setattr(training, 'score', lambda model, pictures, truths: scored.append((pictures, truths)) or len(scored)/10)
    result = training.train(settings, root/'job', lambda *args: None)
    assert result['missing_marker_training'] and result['required_nuclear'] == 'DAPI'
    assert result['training_tiles'] == result['validation_tiles'] == 1 and result['input_examples'] == 4
    pictures = calls[0]['train_data']
    np.testing.assert_array_equal(pictures[0][2], 70)  # full: includes H&E stain
    np.testing.assert_array_equal(pictures[1][2], 10)  # no H&E: retains DAPI
    np.testing.assert_array_equal(pictures[3][2], 10)  # DAPI only
    np.testing.assert_array_equal(pictures[3][0], 10)  # same nuclear fallback as inference
    mask = tifffile.imread(source/f'{tiles[0]}_masks.tif')
    assert len(calls[0]['train_labels']) == 4
    for label in calls[0]['train_labels']:
        np.testing.assert_array_equal(label, mask)
    assert len(scored) == 8
    assert result['subset_validation']['full'] == {'before': 0.1, 'after': 0.5}
    assert result['subset_validation']['dapi_only'] == {'before': 0.4, 'after': 0.8}
    assert all((source/name).read_bytes() == value for name, value in masks_before.items())
    # The actual newly saved metadata enforces DAPI and permits absent optional channels.
    settings = {**settings['input_settings'], 'tab4_model_type': 'custom', 'tab4_custom_model': result['model']}
    assert segmentation.channels(['DAPI'], settings) == (['DAPI'], ['DAPI'])
    assert segmentation.channels(['DAPI', 'Border'], settings) == (['DAPI'], ['Border'])
    with pytest.raises(ValueError, match='requires DAPI'):
        segmentation.channels(['Hematoxylin', 'Border'], settings)
    with pytest.raises(ValueError, match='requires DAPI'):
        segmentation.channels(['DAPI', 'Hematoxylin'], {**settings, 'tab4_last_nuclear_names': ['Hematoxylin']})


def test_missing_marker_mode_requires_dapi_choice_and_actual_dapi(scene):
    root, _, tiles, config = scene
    config['tab_configurations']['tab5_segmentation']['tab4_last_nuclear_names'] = ['Hematoxylin']
    with pytest.raises(ValueError, match='requires DAPI selected'):
        training.retrain_settings(root, [tiles[0]], config, {}, missing_markers=True)
    # A different image has the optional stain, but no DAPI. Never invent an anchor.
    other = root/'no_dapi.ome.tiff'
    marker_image(other, np.ones((2, 96, 96), np.uint16), ['Hematoxylin', 'Border'])
    source, _ = training.folders(root)
    entries = training._read(source/'examples.json', [])
    entries[0]['image_path'] = str(other)
    training._write(source/'examples.json', entries)
    config['tab_configurations']['tab5_segmentation']['tab4_last_nuclear_names'] = ['DAPI', 'Hematoxylin']
    settings = training.retrain_settings(root, [tiles[0]], config, {}, missing_markers=True)
    with pytest.raises(ValueError, match='required DAPI channel missing'):
        training.train(settings, root/'job', lambda *args: None)


def test_subset_recipes_group_he_and_are_deterministic():
    from spacemap.segmentation_inputs import marker_subsets
    nuclear = ['DAPI', 'Ki67', 'HE_Red', 'HE_Green', 'HE_Blue']
    border = ['CD31', 'E-cadherin', 'CD45', 'HLA-A']
    seen = set()
    for seed in range(20):
        recipes = marker_subsets(nuclear, border, seed)
        assert recipes == marker_subsets(nuclear, border, seed)
        assert recipes['full'] == (nuclear, border)
        for mode, (nuc, mem) in recipes.items():
            assert 'DAPI' in nuc
            if mode != 'full':
                assert not any(n.startswith('HE_') for n in nuc+mem)
        assert recipes['dapi_only'] == (['DAPI'], [])
        seen.add(tuple(recipes['fewer_markers'][1]))
    assert len(seen) > 4


def test_no_selected_nuclear_channel_never_falls_back_to_first_arbitrary_marker():
    with pytest.raises(ValueError, match='None of the selected nuclear markers'):
        segmentation.channels(['CD45', 'HE_Red'], {**segmentation.DEFAULTS, 'tab4_last_nuclear_names': ['DAPI']})


def test_missing_required_channel_preserves_existing_segmentation_results(setup, monkeypatch):
    from test_studio_commands import run
    from spacemap_gui.studio import jobs
    root = setup[2]
    model = root/'model_with_dapi'
    model.write_bytes(b'fixture')
    Path(str(model)+'.json').write_text(json.dumps({'missing_marker_training': True, 'required_nuclear': 'DAPI'}))
    (root/f'{root.name}.json').write_text(json.dumps({'tab_configurations': {'tab5_segmentation': {
        'tab4_model_type': 'custom', 'tab4_custom_model': str(model), 'tab4_last_nuclear_names': ['DAPI']}}}))
    output = root/'outputs/segmentation/P11_roi_2.ome_cell_features.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('label\n1\n')
    monkeypatch.setattr(jobs, 'start', lambda *a, **kw: pytest.fail('Cannot launch incompatible image'))
    answer = run(setup, 'segment_images', {'again': True, 'tag': False}, confirmed=True)
    assert 'requires DAPI' in answer['message']
    assert output.read_text() == 'label\n1\n' and not (output.parent/'previous').exists()


def test_real_cellpose_round_and_prediction_without_optional_channels(scene):
    """One CPU epoch checks the actual trainer/model contract, not accuracy."""
    root, _, tiles, config = scene
    config['tab_configurations']['tab5_segmentation']['tab4_gpu'] = False
    settings = training.retrain_settings(root, [tiles[0], tiles[1]], config, {}, 1, missing_markers=True)
    result = training.train(settings, root/'job', lambda *args: None)
    assert Path(result['model']).is_file() and result['input_examples'] == 8
    assert set(result['subset_validation']) == {'full', 'without_he', 'fewer_markers', 'dapi_only'}
    assert all(0 <= value['after'] <= 1 for value in result['subset_validation'].values())
    # A genuinely different channel panel; retain the training choices to test omission.
    image = root/'only_dapi.ome.tiff'
    marker_image(image, np.full((1, 64, 64), 10, np.uint16), ['DAPI'])
    current = {**settings['input_settings'], 'tab4_model_type': 'custom', 'tab4_custom_model': result['model']}
    found = segmentation.preview(str(image), ['DAPI'], [0, 0, 64, 64], current)
    assert found['nuclear'] == found['membranes'] == ['DAPI']
    assert found['missing_markers'] == ['Border', 'Hematoxylin']
