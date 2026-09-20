"""Disk-backed Leiden jobs, independent of Streamlit's script/session lifetime.

Only JSON paths/settings cross the process boundary. Each project has one active
job; logs and completed results remain available after a browser reconnect.
"""

from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _job_root(project_root):
    return Path(project_root).resolve() / "outputs" / "phenotyping" / "leiden_jobs"


def read_job(job_dir):
    """Read an atomic status snapshot, detecting workers that exited abruptly."""
    job_dir = Path(job_dir)
    status = json.loads((job_dir / "status.json").read_text())
    status["job_dir"] = str(job_dir)
    if status["status"] in {"starting", "running"}:
        process_path = job_dir / "process.json"
        if process_path.exists():
            process = json.loads(process_path.read_text())
            if process["hostname"] == socket.gethostname():
                import psutil

                try:
                    worker = psutil.Process(process["pid"])
                    alive = (
                        worker.create_time() == process["create_time"]
                        and worker.status() != psutil.STATUS_ZOMBIE
                    )
                except psutil.NoSuchProcess:
                    alive = False
                except psutil.AccessDenied:
                    alive = True
                if not alive:
                    # Re-read: the worker may have saved completion between
                    # our first status read and the process-liveness check.
                    status.update(json.loads((job_dir / "status.json").read_text()))
                    if status["status"] in {"starting", "running"}:
                        status.update(
                            status="interrupted",
                            error=(
                                "The Leiden worker exited before saving a result. "
                                "Check the saved terminal log and node/job memory limits."
                            ),
                        )
    return status


def latest_job(project_root):
    root = _job_root(project_root)
    pointer = root / "latest.json"
    if not pointer.exists():
        return None
    return read_job(root / json.loads(pointer.read_text())["job_id"])


def start_job(project_root, settings, audit=None):
    """Start or reconnect to this project's existing active process."""
    import fcntl
    import psutil

    root = _job_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        active = latest_job(project_root)
        if active and active["status"] in {"starting", "running"}:
            return active
        job_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        job_dir = root / job_id
        job_dir.mkdir()
        write_json(
            job_dir / "request.json",
            {
                "project_root": str(Path(project_root).resolve()),
                "settings": settings,
                "audit": audit,
            },
        )
        write_json(job_dir / "status.json", {"job_id": job_id, "status": "starting"})
        env = os.environ.copy()
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        with (job_dir / "terminal.log").open("a") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "spacemap_workflows.leiden_job", str(job_dir)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        write_json(
            job_dir / "process.json",
            {
                "pid": process.pid,
                "hostname": socket.gethostname(),
                "create_time": psutil.Process(process.pid).create_time(),
            },
        )
        write_json(root / "latest.json", {"job_id": job_id})
        # Reap the child eventually without tying its lifetime to this thread.
        from threading import Thread

        Thread(target=process.wait, daemon=True).start()
        return read_job(job_dir)


def read_log(job_dir, max_bytes=80000):
    path = Path(job_dir) / "terminal.log"
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - max_bytes))
        text = handle.read().decode("utf-8", errors="replace")
    return text.split("\n", 1)[-1] if size > max_bytes else text


@contextmanager
def stage(logger, name):
    start = time.monotonic()
    logger.info("STARTED: %s", name)
    try:
        yield
    except BaseException as exc:
        label = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
        logger.error(
            "%s: %s after %.2fs (%s: %s)",
            label,
            name,
            time.monotonic() - start,
            type(exc).__name__,
            exc,
        )
        raise
    else:
        logger.info("COMPLETED: %s in %.2fs", name, time.monotonic() - start)


def run_job(job_dir):
    job_dir = Path(job_dir)
    logger = logging.getLogger("SPACEMAP_LEIDEN_WORKER")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    status = {"job_id": job_dir.name, "status": "running"}
    write_json(job_dir / "status.json", status)
    try:
        request = json.loads((job_dir / "request.json").read_text())
        logger.info("Loading Leiden dependencies in the background worker...")
        from spacemap_workflows.leiden_pipeline import run_pipeline

        with stage(logger, "Total Leiden Pipeline"):
            result = run_pipeline(request, job_dir, logger)
        write_json(job_dir / "result.json", result)
        status.update(status="success")
    except BaseException as exc:
        import traceback

        logger.exception("Leiden pipeline stopped")
        status.update(
            status="failed" if isinstance(exc, Exception) else "interrupted",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        write_json(job_dir / "status.json", status)
        logger.removeHandler(handler)
        handler.close()
    return 0 if status["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(run_job(sys.argv[1]))
