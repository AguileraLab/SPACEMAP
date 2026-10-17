"""The SPACEMAP servers this user has running, wherever they run, so an old one left behind can be found and stopped.

Each server writes ``servers/<host>-<port>.json`` in the user's SPACEMAP folder (shared across the cluster's nodes)
when it starts listening, refreshes it every minute and removes it when it stops. A server not heard from for three
minutes is gone. One on this machine can be stopped from Studio; one on another machine is named so it can be
stopped there (or its Slurm job ended).
"""

import atexit
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time

BEAT = 60
GONE = 180


def _folder():
    from spacemap.userpaths import config_dir

    return Path(config_dir()) / "servers"


def _record(port):
    return _folder() / f"{socket.gethostname()}-{int(port)}.json"


def register(port, source=""):
    """Announce this server and keep the announcement fresh until it stops."""
    path = _record(port)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"host": socket.gethostname(), "port": int(port), "pid": os.getpid(), "started": time.time(),
             "source": str(source), "slurm_job": os.environ.get("SLURM_JOB_ID", "")}

    def beat():
        while True:
            try:
                path.write_text(json.dumps({**entry, "seen": time.time()}))
            except OSError:
                pass
            time.sleep(BEAT)

    threading.Thread(target=beat, name="spacemap-server-beat", daemon=True).start()

    def forget():
        try:
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()
        except (OSError, ValueError):
            pass

    atexit.register(forget)
    return entry


def _alive_here(pid):
    try:
        pid = int(pid)
        os.kill(pid, 0)
    except (OSError, TypeError, ValueError):
        return False
    # This process registered itself; its launcher need not name SPACEMAP.
    # Keep the command-line check for other PIDs before offering to stop them.
    if pid == os.getpid():
        return True
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return True
    return "streamlit" in command or "spacemap" in command


def listing(this_port=None):
    """The servers running now, newest first; this one marked."""
    found = []
    folder = _folder()
    if not folder.is_dir():
        return found
    host = socket.gethostname()
    for path in folder.glob("*.json"):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        here = entry.get("host") == host
        gone = (here and not _alive_here(entry.get("pid"))) or time.time() - float(entry.get("seen", 0)) > GONE
        if gone:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        entry["this"] = here and entry.get("pid") == os.getpid()
        entry["here"] = here
        found.append(entry)
    return sorted(found, key=lambda e: -float(e.get("started", 0)))


def stop(host, port):
    """Stop another server on this machine (never this one). Returns what was done."""
    entry = next((e for e in listing() if e.get("host") == host and int(e.get("port", -1)) == int(port)), None)
    if entry is None:
        raise ValueError("That server is not running any more.")
    if entry["this"]:
        raise ValueError("This is the server you are using: stop it in its terminal (Ctrl+C).")
    if not entry["here"]:
        job = f" (Slurm job {entry['slurm_job']}: scancel {entry['slurm_job']})" if entry.get("slurm_job") else ""
        raise ValueError(f"It runs on {entry['host']}: stop it there with Ctrl+C or kill {entry['pid']}{job}.")
    os.kill(int(entry["pid"]), signal.SIGTERM)
    return f"Stopped the SPACEMAP server on {host}:{port}."
