"""Opt-in live HTTP check: fixed gateway, two backend processes/ports, same link.

SPACEMAP_NETWORK_TESTS=1 PYTHONPATH=src:tests python -m pytest -q \
    tests/test_review_link_migration.py

The gateway below is a disposable test fixture, not a deployment server.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import httpx
import pytest


pytestmark = pytest.mark.skipif(os.environ.get('SPACEMAP_NETWORK_TESTS') != '1',
                                reason='Opt-in loopback HTTP migration test')
API = '/spacemap-viewer/api'
WORKER = '''
import sys
from starlette.applications import Starlette
from starlette.routing import Mount
import uvicorn
from spacemap_gui.viewer import api, registry
registry.publish('migration-fixture', {'layers': []}, project={'root': sys.argv[1]})
uvicorn.run(Starlette(routes=[Mount('/spacemap-viewer/api', app=api.app)]),
            host='127.0.0.1', port=int(sys.argv[2]), access_log=False, log_level='error')
'''


@contextmanager
def backend(root, state_home, gateway, index):
    with socket.socket() as candidate:
        candidate.bind(('127.0.0.1', 0))
        port = candidate.getsockname()[1]
    env = {**os.environ, 'SPACEMAP_HOME': str(state_home),
           'SPACEMAP_STATE_DIR': str(state_home / 'state'),
           'SPACEMAP_REVIEW_INTERNAL_BASE_URL': gateway + API,
           'SPACEMAP_ASSISTANT': 'off', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    with (root / f'backend-{index}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', WORKER, str(root), str(port)],
                                   env=env, stdout=log, stderr=log)
        try:
            url = f'http://127.0.0.1:{port}'
            with httpx.Client(trust_env=False, timeout=1) as client:
                deadline = time.monotonic() + 55
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        pytest.fail(f'Backend exited; see {log.name}')
                    try:
                        if client.get(url + API + '/window/index.html').status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(.1)
                else:
                    pytest.fail(f'Backend did not start; see {log.name}')
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@contextmanager
def gateway():
    destination = {'url': None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Reviewer credentials must not enter access logs.

        def forward(self):
            path = urlsplit(self.path).path
            allowed = (path.startswith(API + '/review-public/') or
                       (self.command == 'GET' and path.startswith(API + '/window/')))
            if not allowed:
                self.send_error(404)
                return
            if not destination['url']:
                self.send_error(503)
                return
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            with httpx.Client(trust_env=False, timeout=30) as client:
                response = client.request(self.command, destination['url'] + self.path,
                                          content=raw, headers={'Content-Type': self.headers.get('Content-Type', '')})
            self.send_response(response.status_code)
            for key in ('Content-Type', 'Location', 'Cache-Control', 'Referrer-Policy'):
                if key in response.headers:
                    self.send_header(key, response.headers[key])
            self.send_header('Content-Length', str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = do_POST = forward

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', destination
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_same_link_and_decisions_survive_backend_restart_on_new_port(tmp_path):
    from spacemap_gui.studio import annotation_components, review_store as store
    from test_viewer import _write_channel_pages
    source = tmp_path / 'source.geojson'
    source.write_text(json.dumps({'type': 'FeatureCollection', 'features': [{
        'type': 'Feature', 'properties': {'classification': {'name': 'tumor'}},
        'geometry': {'type': 'Polygon', 'coordinates': [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}}]}))
    image = tmp_path / 'roi.tif'
    _write_channel_pages(image, channels=2, height=32, width=32)
    indexed = annotation_components.index(source)
    item = store.add(tmp_path, kind='annotation', image=image.name, title='tumor',
                     target={'image': str(image), 'source': str(source), 'revision': indexed['revision'],
                             'ids': [indexed['rows'][0]['id']]})
    route = API + '/studio/migration-fixture/review'
    with gateway() as (fixed, destination), httpx.Client(trust_env=False, timeout=30) as client:
        with backend(tmp_path, tmp_path / 'home', fixed, 1) as first:
            destination['url'] = first
            result = client.post(first + route, json={'action': 'share', 'name': 'Disposable reviewer', 'items': [item['id']]})
            assert result.status_code == 200, result.text
            link = result.json()['url']
            grant_id = result.json()['id']
            assert link.startswith(fixed + API + '/review-public/')
            assert client.get(link, follow_redirects=True).status_code == 200
            assert client.get(fixed + route).status_code == 404
            assert client.get(link + 'queue').json()['total'] == 1
            answered = client.post(link + 'decision', json={'id': item['id'], 'version': item['version'],
                                                          'decision': 'relabel', 'label': 'normal'})
            assert answered.status_code == 200, answered.text
        destination['url'] = None
        assert client.get(link + 'queue').status_code == 503  # Stable address is not continuous service.
        with backend(tmp_path, tmp_path / 'home', fixed, 2) as second:
            assert second != first
            destination['url'] = second
            queue = client.get(link + 'queue')
            assert queue.status_code == 200, queue.text
            record = queue.json()['items'][0]
            assert record['decision']['label'] == 'normal' and record['state'] == 'answered'
            assert client.get(link, follow_redirects=True).status_code == 200
            view = client.get(link + 'view', params={'id': item['id']})
            assert view.status_code == 200, view.text
            recalled = client.get(second + route, params={'action': 'link', 'id': grant_id})
            assert recalled.json()['url'] == link
            updated = client.post(second + route, json={'action': 'update_link', 'id': grant_id,
                                  'name': 'Disposable reviewer', 'access': 'internal', 'items': [item['id']]})
            assert updated.status_code == 200 and updated.json()['url'] == link
            assert len(store.grants(tmp_path)) == 1
            assert client.post(second + route, json={'action': 'revoke', 'id': grant_id}).status_code == 200
            assert client.get(link + 'queue').status_code == 403
