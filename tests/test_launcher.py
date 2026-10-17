import socket

import pytest

from spacemap_gui.launch import available_port, command


def test_busy_port_uses_next_available_and_preserves_listener():
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0)); server.listen()
        busy = server.getsockname()[1]
        assert available_port(busy, '127.0.0.1') > busy
        with socket.create_connection(('127.0.0.1', busy), timeout=1):
            pass


@pytest.mark.parametrize('args', [['--server.port=8502'], ['--server.port','8502']])
def test_port_override_replaced_once_other_options_preserved(monkeypatch, args):
    monkeypatch.setattr('spacemap_gui.launch.available_port', lambda port, address: port + 1)
    found = command('server.py', args + ['--server.address=127.0.0.1', '--server.headless=true'])
    assert found[4:] == ['server.py','--server.port=8503','--server.address=127.0.0.1','--server.headless=true']
