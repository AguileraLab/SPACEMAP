"""Stop isolated compute workers without interrupting publication of results.

Only audited jobs whose shared outputs use a protected publication opt in.
A small independent watcher lives on the worker's host, so Stop also works
across a shared project and while Python is blocked inside native training.
"""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

from spacemap.sharing import open_lock
from spacemap_workflows.leiden_job import write_json

ACTIVE = {"starting", "running", "stopping"}
INTERRUPTIBLE = {"consensus_ml", "mr_ml", "resolve_all", "leiden", "som", "vortex",
                 "subphenotype", "neighbourhoods", "tls_candidates", "cell_tagging"}
PROTOCOL = 1


class Cancelled(Exception):
    """Ordinary checkpoint cancellation, preserving other jobs' cleanup paths."""


class ComputationCancelled(BaseException):
    """Verified compute callbacks must not swallow cancellation as a fit error."""


class OutputGuard:
    """Once publication starts, finish it and final bookkeeping without killing.

    Retaining the lock through completion also prevents a completed result from
    becoming an orphan if Stop is pressed during its final UI summaries.
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.handle = None

    def protect(self, *, allow_cancel=False, saving=True):
        if self.handle is not None:
            return
        handle = open_lock(self.folder / ".output.lock")
        fcntl.flock(handle, fcntl.LOCK_SH)
        if not allow_cancel and (self.folder / "cancel").exists():
            handle.close()
            raise Cancelled("Stopped by the user before saving results.")
        self.handle = handle
        if saving:
            (self.folder / "saving").touch()

    def close(self):
        if self.handle is not None:
            (self.folder / "saving").unlink(missing_ok=True)
            self.handle.close()
            self.handle = None


def _alive(record):
    import psutil

    try:
        process = psutil.Process(record["pid"])
        return process.create_time() == record["created"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True  # Lack of permission is not evidence of exit.


def reconcile(folder, current):
    """Persist an exited worker's status; keep blocked until helpers exit too."""
    folder = Path(folder)
    record_path = folder / "process.json"
    if current["status"] not in ACTIVE or not record_path.exists():
        return current
    record = json.loads(record_path.read_text())
    if record["host"] != socket.gethostname():
        return current
    records = [record]
    stopping = folder / "terminating.json"
    if stopping.exists():
        records.extend(json.loads(stopping.read_text()).get("processes", []))
    if any(_alive(item) for item in records):
        return current
    from .tagging_outputs import cleanup_staging

    cleanup_staging(folder)
    current = json.loads((folder / "status.json").read_text())
    if current["status"] in ACTIVE:
        cancelled = (folder / "cancel").exists()
        current.update(status="cancelled" if cancelled else "failed",
                       message="Stopped by the user." if cancelled else
                       "The worker stopped before saving its result. Check the job log.",
                       finished=datetime.now(timezone.utc).isoformat())
        write_json(folder / "status.json", current)
    return current


def message(folder, record):
    folder = Path(folder)
    if record.get("stop_protocol") != PROTOCOL:
        return "Stop requested; waiting for the current safe checkpoint."
    if (folder / "saving").exists():
        return "Finishing the result save safely…"
    if (folder / "stop_error.json").exists():
        return json.loads((folder / "stop_error.json").read_text())["message"]
    return "Stopping computation…"


def _verified_worker(folder, record):
    import psutil

    if record["host"] != socket.gethostname():
        raise ValueError("The stop watcher must run on the worker's host.")
    process = psutil.Process(record["pid"])
    if process.create_time() != record["created"]:
        raise psutil.NoSuchProcess(record["pid"])
    args = process.cmdline()
    if (process.uids().real != os.getuid() or len(args) != 4 or
            args[1:3] != ["-m", "spacemap_gui.studio.jobs"] or
            Path(args[3]).resolve() != folder.resolve() or
            os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid):
        raise ValueError("Worker identity could not be verified; no process was terminated.")
    return process


def stop_helpers(folder):
    """Also clean up helpers when a progress callback cancels before the watcher."""
    import psutil

    children = psutil.Process().children(recursive=True)
    identities = []
    for child in children:
        try:
            identities.append({"pid": child.pid, "created": child.create_time()})
        except psutil.NoSuchProcess:
            pass
    if identities:
        write_json(Path(folder) / "terminating.json", {"processes": identities})
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=2)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=2)
    if any(_alive(item) for item in identities):
        raise RuntimeError("Stop requested; waiting for the operating system to release helper processes.")


def _terminate(folder, record, grace=2.0):
    import psutil

    with open_lock(folder / ".stop.lock") as single:
        try:
            fcntl.flock(single, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        with open_lock(folder / ".output.lock") as output:
            # Nonblocking: a result transaction can finish while the UI polls.
            while True:
                current = reconcile(folder, json.loads((folder / "status.json").read_text()))
                if current["status"] not in ACTIVE:
                    return
                try:
                    fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            current = json.loads((folder / "status.json").read_text())
            if current["status"] not in ACTIVE:
                return
            process = _verified_worker(folder, record)
            processes = [process, *process.children(recursive=True)]
            identities = []
            for item in processes:
                try:
                    identities.append({"pid": item.pid, "created": item.create_time()})
                except psutil.NoSuchProcess:
                    pass
            # Record helpers before signaling: parent exit alone must not unlock
            # the project while an ignored TERM is still consuming CPU/GPU.
            write_json(folder / "terminating.json", {"processes": identities})
            os.killpg(process.pid, signal.SIGTERM)
            # Helpers may have created their own session; psutil protects each
            # saved Process object against PID reuse when sending signals.
            for item in processes[1:]:
                try:
                    item.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(processes, timeout=grace)
            forced = []
            for item in alive:
                try:
                    if item.status() == psutil.STATUS_ZOMBIE:
                        continue
                    item.kill()
                    forced.append(item.pid)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=grace)
            if any(_alive(item) for item in identities):
                raise RuntimeError("Stop requested; waiting for the operating system to release the worker.")
            current = reconcile(folder, current)
            if current["status"] == "cancelled":
                current.update(force_terminated=True, termination_signal="SIGKILL" if forced else "SIGTERM")
                write_json(folder / "status.json", current)
                with (folder / "terminal.log").open("a") as log:
                    log.write("\nStopped by the user; worker and helper processes exited.\n")


def watch(folder):
    """Independent process; never executes commands supplied by a client."""
    import psutil

    folder = Path(folder)
    record = json.loads((folder / "process.json").read_text())
    request = json.loads((folder / "request.json").read_text())
    if (record.get("stop_protocol") != PROTOCOL or request["kind"] not in INTERRUPTIBLE
            or record["host"] != socket.gethostname()):
        return
    while True:
        current = reconcile(folder, json.loads((folder / "status.json").read_text()))
        if current["status"] not in ACTIVE:
            return
        if (folder / "cancel").exists():
            try:
                _terminate(folder, record)
            except psutil.NoSuchProcess:
                reconcile(folder, current)
            except (OSError, ValueError, RuntimeError, psutil.Error) as exc:
                write_json(folder / "stop_error.json", {"message": str(exc)})
            return
        time.sleep(0.2)


if __name__ == "__main__":
    watch(sys.argv[1])
