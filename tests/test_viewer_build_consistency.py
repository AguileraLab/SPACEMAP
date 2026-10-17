from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient
from spacemap_gui.viewer import api


def test_running_server_keeps_matching_frontend_until_restart(tmp_path,monkeypatch):
    folder=tmp_path/'frontend';folder.mkdir()
    (folder/'index.html').write_text('<script src="test.js"></script>')
    script=folder/'test.js';script.write_text('const version=1;')
    monkeypatch.setattr(api,'__file__',str(tmp_path/'api.py'))
    client=TestClient(Starlette(routes=[Mount('/window',app=api._window_files())]))
    first=client.get('/window/test.js');assert 'version=1' in first.text
    script.write_text('const version=2;')
    assert client.get('/window/test.js').content==first.content
    assert client.get('/window/test.js',headers={'If-None-Match':first.headers['etag']}).status_code==304
    fresh=TestClient(Starlette(routes=[Mount('/window',app=api._window_files())]))
    assert 'version=2' in fresh.get('/window/test.js').text
