"""SPACEMAP servers left running can be listed and stopped; a refused save says what changed the file."""

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from spacemap_gui.studio import annotation_edits as edits, servers


def test_servers_are_listed_and_another_one_here_can_be_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "_folder", lambda: tmp_path / "servers")
    servers.register(9999)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)  # spacemap"])
    try:
        (tmp_path / "servers").mkdir(exist_ok=True)
        (tmp_path / "servers" / f"{socket.gethostname()}-9998.json").write_text(json.dumps(
            {"host": socket.gethostname(), "port": 9998, "pid": other.pid, "started": time.time(), "seen": time.time()}))
        (tmp_path / "servers" / "elsewhere-8502.json").write_text(json.dumps(
            {"host": "elsewhere", "port": 8502, "pid": 1, "started": time.time(), "seen": time.time(), "slurm_job": "42"}))
        (tmp_path / "servers" / "stale-8503.json").write_text(json.dumps({"host": "stale", "port": 8503, "pid": 1, "seen": 0}))
        time.sleep(0.2)
        found = {(e["host"], e["port"]): e for e in servers.listing()}
        assert found[(socket.gethostname(), 9999)]["this"] and ("stale", 8503) not in found and len(found) == 3
        with pytest.raises(ValueError, match="This is the server"):
            servers.stop(socket.gethostname(), 9999)
        with pytest.raises(ValueError, match="scancel 42"):
            servers.stop("elsewhere", 8502)
        assert "Stopped" in servers.stop(socket.gethostname(), 9998)
        other.wait(timeout=10)
    finally:
        if other.poll() is None:
            other.kill()


def test_a_refused_save_says_what_changed_the_file(tmp_path):
    path = tmp_path / "roi_valid_tissue.geojson"
    first = edits.replace(path, [], edits.revision(path), "Detected tissue")
    square = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]]]}}
    edits.replace(path, [square], first["revision"], "Detected tissue")  # a job writes it meanwhile
    with pytest.raises(ValueError, match=r"changed at \d\d:\d\d \(Detected tissue\)"):
        edits.replace(path, [], first["revision"], "Edited by hand")


def test_current_server_survives_a_generic_python_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "_folder", lambda: tmp_path)
    monkeypatch.setattr(servers.Path, "read_bytes", lambda self: b"python\0-m\0pytest\0-q\0")
    entry = {"host": socket.gethostname(), "port": 9999, "pid": os.getpid(),
             "started": time.time(), "seen": time.time()}
    record = tmp_path / "current.json"
    record.write_text(json.dumps(entry))

    assert servers.listing()[0]["this"] is True
    assert record.exists()
    with pytest.raises(ValueError, match="This is the server"):
        servers.stop(entry["host"], entry["port"])


def test_an_unrelated_python_process_is_not_offered_or_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "_folder", lambda: tmp_path)
    monkeypatch.setattr(servers.Path, "read_bytes", lambda self: b"python\0-m\0pytest\0-q\0")
    signals = []
    monkeypatch.setattr(servers.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    other_pid = os.getpid() + 1
    entry = {"host": socket.gethostname(), "port": 9998, "pid": other_pid,
             "started": time.time(), "seen": time.time()}
    record = tmp_path / "unrelated.json"
    record.write_text(json.dumps(entry))

    with pytest.raises(ValueError, match="not running any more"):
        servers.stop(entry["host"], entry["port"])
    assert signals == [(other_pid, 0)]  # Only probe liveness; never terminate it.
    assert not record.exists()
