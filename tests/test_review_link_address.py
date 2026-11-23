"""Fixed internal addresses must not change review credentials or stored decisions."""
import pytest
from starlette.requests import Request

from spacemap_gui.viewer import review_links


def request(host='node-a:8503'):
    return Request({'type': 'http', 'scheme': 'http', 'server': ('node-a', 8503),
                    'path': '/spacemap-viewer/api/studio/session/review',
                    'root_path': '', 'headers': [(b'host', host.encode())],
                    'query_string': b'', 'method': 'POST'})


@pytest.mark.parametrize('scheme', ['http', 'https'])
def test_internal_gateway_does_not_follow_compute_node(monkeypatch, scheme):
    base = f'{scheme}://review.internal.example:9080/spacemap-viewer/api'
    monkeypatch.setenv('SPACEMAP_REVIEW_INTERNAL_BASE_URL', base + '/')
    assert review_links.base_url(request('node-a:8503'), 'internal') == base
    assert review_links.base_url(request('node-b:8502'), 'internal') == base


def test_without_gateway_internal_links_keep_current_node(monkeypatch):
    monkeypatch.delenv('SPACEMAP_REVIEW_INTERNAL_BASE_URL', raising=False)
    assert review_links.base_url(request('node-b:8502'), 'internal') == 'http://node-b:8502/spacemap-viewer/api'


def test_external_address_is_independent(monkeypatch):
    monkeypatch.setenv('SPACEMAP_REVIEW_INTERNAL_BASE_URL', 'http://vpn.example/spacemap-viewer/api')
    monkeypatch.setenv('SPACEMAP_REVIEW_PUBLIC_BASE_URL', 'https://public.example/spacemap-viewer/api')
    assert review_links.base_url(request(), 'external') == 'https://public.example/spacemap-viewer/api'


@pytest.mark.parametrize('base', [
    'review.example', 'ftp://review.example', 'http://',
    'http://user:secret@review.example', 'http://@review.example',
    'http://review.example/?key=secret', 'http://review.example/#fragment',
    'http://review.example:wrong', 'http://review.example:65536', 'http://review.example:0',
    'http://bad host', 'http://review.example/\npath', 'http://review.example\\other',
])
def test_invalid_internal_service_is_rejected(monkeypatch, base):
    monkeypatch.setenv('SPACEMAP_REVIEW_INTERNAL_BASE_URL', base)
    with pytest.raises(ValueError, match='SPACEMAP_REVIEW_INTERNAL_BASE_URL'):
        review_links.base_url(request(), 'internal')

