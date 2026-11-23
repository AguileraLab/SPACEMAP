"""The local model server on a cluster: one record per node, leftovers found and stopped, a free port, a guard."""

import os
import socket
import subprocess
import sys
import time

import pytest

from spacemap_assistant import model_runtime


def fake_server(folder, *args):
    """A process run like a model server (Python running .../model_service/server.py) that only sleeps."""
    script = folder / "spacemap_assistant" / "model_service" / "server.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import time\ntime.sleep(120)\n")
    return subprocess.Popen([sys.executable, str(script), *args], start_new_session=True)


def gone(process, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.2)
    return False


def test_the_record_belongs_to_this_node_and_only_servers_match(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path))
    assert model_runtime._state_path().name == f"model_server.{socket.gethostname()}.json"
    assert not model_runtime._pid_matches(os.getpid())  # SPACEMAP itself is not a model server
    python = sys.executable
    assert model_runtime._is_model_server([python, "/x/spacemap_assistant/model_service/server.py", "--port", "1"])
    assert model_runtime._is_model_server([python, "-m", "llama_cpp.server", "--model", "m.gguf"])
    assert not model_runtime._is_model_server(["bash", "-c", "python -m llama_cpp.server"])
    assert not model_runtime._is_model_server([python, "-c", "print('spacemap_assistant/model_service/server.py')"])


def test_a_server_left_by_an_earlier_session_is_found_and_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path))
    left = fake_server(tmp_path, "--model-id", "tiny", "--port", "18765")
    mention = subprocess.Popen(  # only mentions a server: never taken for one
        [sys.executable, "-c", "import time; time.sleep(120)", "spacemap_assistant/model_service/server.py"],
        start_new_session=True,
    )
    try:
        time.sleep(0.5)
        servers = model_runtime.leftover_servers()
        found = [item for item in servers if item["pid"] == left.pid]
        assert found and found[0]["port"] == 18765 and found[0]["model"] == "tiny"
        assert mention.pid not in {item["pid"] for item in servers}
        stopped = model_runtime.stop_leftover_servers([left.pid])  # only this test's server
        assert [item["pid"] for item in stopped] == [left.pid] and gone(left, 10)
    finally:
        left.kill()
        mention.kill()


def test_a_port_held_by_another_program_moves_the_model_to_a_free_one(monkeypatch):
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    saved = {"server_port": port}
    monkeypatch.setattr(model_runtime.ai_settings, "load_settings", lambda: dict(saved))
    monkeypatch.setattr(model_runtime.ai_settings, "update_settings", lambda **changes: saved.update(changes) or saved)
    monkeypatch.setattr(model_runtime, "leftover_servers", lambda: [])
    try:
        note = model_runtime._make_room()
        assert saved["server_port"] > port and not model_runtime._port_in_use(saved["server_port"])
        assert f"port {saved['server_port']}" in note
    finally:
        holder.close()


def test_the_guard_stops_the_server_when_spacemap_stops():
    import psutil

    spacemap = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
    started = psutil.Process(spacemap.pid).create_time()
    guard = subprocess.Popen(
        [sys.executable, str(model_runtime.GUARD_PATH), str(spacemap.pid), repr(started), "--"]
        + [sys.executable, "-c", "import time; time.sleep(120)"],
        start_new_session=True,
    )
    try:
        time.sleep(1)
        server = psutil.Process(guard.pid).children()[0]
        assert not gone(guard, 1.0)  # SPACEMAP still runs: so does the server
        spacemap.wait()
        assert gone(guard, 15) and not server.is_running()
    finally:
        guard.kill()
        spacemap.kill()


@pytest.mark.parametrize("pid", [0, 1])
def test_no_pid_is_not_a_server(pid):
    assert not model_runtime._pid_matches(pid)


def test_stopping_with_spacemap_can_be_turned_off(tmp_path, monkeypatch):
    from spacemap import ai_settings

    monkeypatch.setattr(ai_settings, "config_dir", lambda: tmp_path)
    assert ai_settings.load_settings()["stop_with_spacemap"] is True
    ai_settings.update_settings(stop_with_spacemap=False)
    assert ai_settings.load_settings()["stop_with_spacemap"] is False  # unknown keys used to be dropped


def test_slow_stop_is_a_state_and_eventually_releases_process(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(model_runtime, '_state_path', lambda: tmp_path / 'runtime.json')
    monkeypatch.setattr(model_runtime, 'STOP_GRACE_SECONDS', .4)
    monkeypatch.setattr(model_runtime, '_port_in_use', lambda *a: False)
    monkeypatch.setattr(model_runtime, 'active_entry', lambda: SimpleNamespace(
        id='test', name='Test model', backend='transformers', is_installed=lambda: True))
    script = tmp_path / 'spacemap_assistant/model_service/server.py'
    script.parent.mkdir(parents=True)
    ready = tmp_path / 'ready'
    script.write_text('import signal,time\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        f'Path({str(ready)!r}).touch()\n'
        'time.sleep(120)\n')
    process = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(.02)
        assert ready.exists()
        record = {'host': socket.gethostname(), 'pid': process.pid,
                  'process_started': model_runtime._process_start_time(process.pid)}
        model_runtime._state_path().write_text(json.dumps(record))
        before = time.monotonic()
        status = model_runtime.stop_runtime()
        assert status.running and status.stopping and not status.ready
        assert time.monotonic() - before < .3
        assert model_runtime.stop_runtime().stopping  # idempotent while waiting
        with pytest.raises(model_runtime.ModelRuntimeError, match='stopping'):
            model_runtime.ensure_running()
        with pytest.raises(model_runtime.ModelRuntimeError, match='stopping'):
            model_runtime.start_runtime()
        assert gone(process, 5)
        for _ in range(100):
            if not model_runtime._state_path().exists():
                break
            time.sleep(.02)
        assert not model_runtime._state_path().exists()
        assert not model_runtime.runtime_status().running
    finally:
        process.kill()
        process.wait()


def test_stopping_never_signals_a_shared_application_group(monkeypatch):
    calls = []
    monkeypatch.setattr(model_runtime.os, 'getpgid', lambda pid: 100)
    monkeypatch.setattr(model_runtime.os, 'killpg', lambda *args: calls.append(('group', *args)))
    monkeypatch.setattr(model_runtime.os, 'kill', lambda *args: calls.append(('pid', *args)))
    model_runtime._signal_server(200, model_runtime.signal.SIGTERM)
    assert calls == [('pid', 200, model_runtime.signal.SIGTERM)]


def test_foreign_and_reused_process_records_are_not_stopped(tmp_path, monkeypatch):
    import json
    path = tmp_path / 'runtime.json'
    monkeypatch.setattr(model_runtime, '_state_path', lambda: path)
    monkeypatch.setattr(model_runtime, '_pid_matches', lambda pid: True)
    monkeypatch.setattr(model_runtime, '_process_start_time', lambda pid: 20.)
    path.write_text(json.dumps({'host': 'another-host', 'pid': 200}))
    with pytest.raises(model_runtime.ModelRuntimeError, match='another computer'):
        model_runtime.stop_runtime()
    assert not model_runtime._record_matches({'host':socket.gethostname(),'pid':200,'process_started':10.})
