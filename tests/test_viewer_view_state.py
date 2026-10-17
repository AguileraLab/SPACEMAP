"""Durable image-scoped display preferences, including competing browser saves."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from spacemap.viewer import display, view_state
from spacemap_gui.viewer import registry
from test_viewer import client  # noqa: F401


def record(visible=False):
    return {'layers': {'image': {'visible': visible, 'opacity': .4, 'features': ['must not be copied'],
        'channels': [{'index': 1, 'name': 'DAPI', 'visible': False, 'color': '#123456', 'low': 12.25}]}},
        'extras': [], 'center': [.3, .6], 'zoom': 2.5, 'order': ['image'],
        'layerGroups': {'annotations': True}, 'removed': ['drawings']}


def test_preferences_are_small_json_scoped_to_project_and_full_image_path(tmp_path):
    image = tmp_path / 'data' / 'one' / 'same.ome.tiff'
    saved = view_state.save(tmp_path, image, record())
    path = view_state.path_for(tmp_path, image)
    assert path.is_file() and json.loads(path.read_text()) == saved
    assert saved['image'] == 'data/one/same.ome.tiff'
    assert saved['layers']['image']['channels'][0]['low'] == 12.25
    assert 'features' not in saved['layers']['image']
    assert saved['removed'] == ['drawings'] and saved['center'] == [.3, .6]
    assert view_state.path_for(tmp_path, tmp_path / 'data/two/same.ome.tiff') != path
    assert view_state.load(view_state.path_for(tmp_path / 'another_project', image)) is None
    assert path.stat().st_size < 2000


def test_marker_drawer_settings_survive_json_without_cell_pixels(tmp_path):
    image = tmp_path / 'one.tiff'
    prefs = {'open': True, 'all': False, 'zoom': 4, 'size': 180, 'height': 320,
             'floating': True, 'x': 50, 'y': 100, 'width': 720, 'pixels': 'do not store'}
    view_state.save(tmp_path, image, {**record(), 'splitMarkers': prefs})
    restored = view_state.load(view_state.path_for(tmp_path, image))['splitMarkers']
    assert restored == {k: v for k, v in prefs.items() if k != 'pixels'}
    assert view_state.clean({'splitMarkers': {'height': -30, 'zoom': 400}})['splitMarkers'] == {'height': 140, 'zoom': 8}


@pytest.mark.parametrize('target', ['view', 'display'])
def test_late_save_cannot_undo_newer_unload_save(tmp_path, target):
    image = tmp_path / 'one.tiff'
    def save(sequence, visible):
        stamp = {'writer': 'browser-A', 'sequence': sequence}
        if target == 'view':
            return view_state.save(tmp_path, image, record(visible), stamp)
        return display.save(tmp_path / 'display.json', [{'index': 0, 'name': 'DAPI', 'visible': visible}], stamp=stamp)
    save(2, False)
    save(1, True)
    result = save(2, True)  # duplicate keepalive delivery is harmless
    assert (result['layers']['image']['visible'] if target == 'view' else result['channels'][0]['visible']) is False
    result = save(3, True)
    assert (result['layers']['image']['visible'] if target == 'view' else result['channels'][0]['visible']) is True


def test_concurrent_saves_leave_complete_json(tmp_path):
    image = tmp_path / 'one.tiff'
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda n: view_state.save(tmp_path, image, record(bool(n % 2)),
                      {'writer': 'same-browser', 'sequence': n}), range(30)))
    saved = view_state.load(view_state.path_for(tmp_path, image))
    assert saved['_writers']['same-browser'] == 29 and saved['layers']['image']['visible']
    assert not list(tmp_path.rglob('*.tmp'))


def test_corrupt_file_is_not_overwritten(tmp_path):
    path = view_state.path_for(tmp_path, tmp_path / 'one.tiff')
    path.parent.mkdir(parents=True)
    path.write_text('broken JSON')
    with pytest.raises(ValueError):
        view_state.save(tmp_path, tmp_path / 'one.tiff', record())
    assert path.read_text() == 'broken JSON'


def test_settings_routes_work_while_default_image_worker_pool_is_full(client, tmp_path):
    """Real HTTP handlers remain responsive with the normal thread pool occupied."""
    import threading
    import anyio
    import httpx
    from spacemap_gui.viewer import api

    image = tmp_path / 'busy.tiff'
    token = registry.register_object('workspace', root=str(tmp_path), image=str(image))
    display_token = registry.register_object('display', path=str(tmp_path / 'display.json'), image=str(image))
    entered, release = threading.Event(), threading.Event()
    def busy_image():
        entered.set()
        assert release.wait(10), 'Test did not release the image worker'
    async def scenario():
        limiter = anyio.to_thread.current_default_thread_limiter()
        previous = limiter.total_tokens
        limiter.total_tokens = 1
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(anyio.to_thread.run_sync, busy_image)
                with anyio.fail_after(3):
                    while not entered.is_set():
                        await anyio.sleep(.01)
                try:
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as session:
                        with anyio.fail_after(3):
                            saved = await session.post(f'/workspace/{token}/view', json={'view': record()})
                            assert saved.status_code == 200, saved.text
                            assert (await session.get(f'/workspace/{token}/view')).json()['view']['zoom'] == 2.5
                            saved = await session.post(f'/display/{display_token}/save', json={
                                'channels': [{'index': 0, 'name': 'DAPI', 'low': 17.25}]})
                            assert saved.status_code == 200, saved.text
                            assert (await session.get(f'/display/{display_token}')).json()['channels'][0]['low'] == 17.25
                finally:
                    release.set()
        finally:
            limiter.total_tokens = previous
            release.set()
    anyio.run(scenario)


def test_marker_paths_preserve_legacy_settings_and_separate_new_same_names(tmp_path):
    a, b = tmp_path / 'a/same.tiff', tmp_path / 'b/same.tiff'
    assert display.settings_path(tmp_path, a) != display.settings_path(tmp_path, b)
    legacy = tmp_path / 'outputs/viewer/display/same.tiff.json'
    display.save(legacy, [{'index': 0, 'name': 'DAPI', 'low': 12.5}])
    assert display.settings_path(tmp_path, a) == legacy
    display.save(legacy, display.load(legacy)['channels'], image=a)
    assert display.settings_path(tmp_path, a) == legacy
    assert display.settings_path(tmp_path, b) != legacy


def test_all_markers_off_renders_black_without_reading_pixels(client, tmp_path, monkeypatch):  # noqa: F811
    import io
    import numpy as np
    from PIL import Image
    from spacemap.viewer import render
    from test_viewer import _write_channel_pages

    path = tmp_path / 'image.ome.tiff'
    _write_channel_pages(path, channels=2, height=120, width=160)
    token = registry.register(path)
    route = f'/image/{token}/tile/0/0/0.jpg'
    default = np.asarray(Image.open(io.BytesIO(client.get(route).content)))
    assert default.max() > 0
    def unexpected(*args, **kwargs):
        raise AssertionError('All-off must not read any channel pixels')
    monkeypatch.setattr(render, 'raw_tile', unexpected)
    result = client.get(route + '?s=')
    assert result.status_code == 200
    pixels = np.asarray(Image.open(io.BytesIO(result.content)))
    assert pixels.shape == (120, 160, 3) and pixels.max() == 0


def test_routes_use_registered_image_and_reject_cross_image_annotations(client, tmp_path):  # noqa: F811
    image = tmp_path / 'image.tiff'
    token = registry.register_object('workspace', root=str(tmp_path), image=str(image))
    url = f'/workspace/{token}/view'
    assert client.get(url).json() == {'view': None}
    assert client.post(url, json={'view': record()}).status_code == 200
    assert client.get(url).json()['view']['layers']['image']['opacity'] == .4
    assert client.get('/workspace/not-registered/view').status_code == 404
    bad = record()
    wrong = registry.register_object('annotations', image=str(tmp_path / 'other.tiff'), path=str(tmp_path / 'labels.geojson'))
    bad['extras'] = [{'id': 'wrong', 'type': 'annotations', 'token': wrong}]
    assert client.post(url, json={'view': bad}).status_code == 400
    assert client.get(url).json()['view']['extras'] == []
    assert client.post(url, json={'view': {'layers': []}}).status_code == 400
