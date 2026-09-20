"""Download, import and remove offline models from inside the GUI.

Installing a model is slow and the Streamlit script reruns constantly, so a
download cannot simply block in a callback.  Each install runs on a background
thread and reports through a small status file, which means progress survives
reruns, page changes and a browser reload.

Nothing in this module is imported until the user asks to install something, so
``huggingface_hub`` stays an optional dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from spacemap.model_catalog import ModelEntry
from spacemap.userpaths import assistant_data_dir

STATUS_DIRNAME = "model_installs"

#: Live threads for this process, keyed by model id.  The status file is the
#: durable record; this only lets us tell "running" from "died without
#: writing", and lets a cancel request reach the worker.
_THREADS: dict[str, threading.Thread] = {}
_CANCEL: dict[str, threading.Event] = {}
_LOCK = threading.Lock()


class ModelInstallError(RuntimeError):
    """Raised when an install cannot start or cannot finish."""


@dataclass
class InstallStatus:
    """A snapshot of one install, as shown in the UI."""

    model_id: str
    state: str = "idle"  # idle | running | done | error | cancelled
    message: str = ""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, 100.0 * self.downloaded_bytes / self.total_bytes)

    @property
    def is_active(self) -> bool:
        return self.state == "running"


def _status_dir() -> Path:
    path = assistant_data_dir() / STATUS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _status_path(model_id: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in model_id)
    return _status_dir() / f"{safe}.json"


def _write_status(status: InstallStatus) -> None:
    target = _status_path(status.model_id)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
        os.replace(temporary, target)
    except OSError:
        pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def read_status(model_id: str) -> InstallStatus:
    """Return the last recorded status for *model_id*."""
    try:
        payload = json.loads(_status_path(model_id).read_text(encoding="utf-8"))
        status = InstallStatus(**{**asdict(InstallStatus(model_id=model_id)), **payload})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return InstallStatus(model_id=model_id)
    # A thread that died without writing a terminal state would otherwise show
    # "running" forever.
    if status.state == "running":
        with _LOCK:
            thread = _THREADS.get(model_id)
        if thread is None or not thread.is_alive():
            status.state = "error"
            status.message = "The download stopped unexpectedly. Try again."
    return status


def clear_status(model_id: str) -> None:
    """Forget a finished or failed install so the UI returns to idle."""
    try:
        _status_path(model_id).unlink(missing_ok=True)
    except OSError:
        pass


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _require_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ModelInstallError(
            "Downloading a model needs the 'huggingface-hub' package. Install the AI "
            "extras with: pip install 'spacemap-ai[ai]'"
        ) from exc
    return __import__("huggingface_hub")


def _progress_watcher(entry: ModelEntry, status: InstallStatus, stop: threading.Event) -> None:
    """Poll the target directory so the UI can show a progress bar.

    ``huggingface_hub`` reports progress through tqdm, which does not survive a
    Streamlit rerun.  Measuring bytes on disk is coarser but needs no plumbing
    and is accurate enough for a multi-gigabyte download.
    """
    target = entry.target_dir()
    while not stop.wait(2.0):
        status.downloaded_bytes = _directory_size(target)
        _write_status(status)


def _run_download(entry: ModelEntry, token: str, status: InstallStatus, cancel: threading.Event) -> None:
    hub = _require_hub()
    target = entry.target_dir()
    target.mkdir(parents=True, exist_ok=True)

    stop_watcher = threading.Event()
    watcher = threading.Thread(
        target=_progress_watcher, args=(entry, status, stop_watcher), daemon=True
    )
    watcher.start()

    try:
        if cancel.is_set():
            raise ModelInstallError("Cancelled before the download started.")
        common: dict[str, Any] = {
            "repo_id": entry.repo_id,
            "revision": entry.revision or "main",
            "token": token or None,
        }
        if entry.backend == "gguf" and entry.filename:
            hub.hf_hub_download(
                filename=entry.filename,
                local_dir=str(target),
                **common,
            )
        else:
            hub.snapshot_download(
                local_dir=str(target),
                allow_patterns=list(entry.allow_patterns) or None,
                # Weight shards plus the tokenizer are enough; the repository's
                # own git metadata and duplicate formats only waste disk.
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", ".git*"],
                **common,
            )
    finally:
        stop_watcher.set()
        watcher.join(timeout=5.0)

    status.downloaded_bytes = _directory_size(target)


def _worker(entry: ModelEntry, token: str, status: InstallStatus, cancel: threading.Event) -> None:
    try:
        _run_download(entry, token, status, cancel)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        status.state = "cancelled" if cancel.is_set() else "error"
        status.message = _friendly_error(entry, exc)
    else:
        if entry.is_installed():
            status.state = "done"
            status.message = f"{entry.name} is installed."
        else:
            status.state = "error"
            status.message = (
                "The download finished but the model files are incomplete. "
                "Remove the folder and try again."
            )
    finally:
        status.finished_at = time.time()
        _write_status(status)
        with _LOCK:
            _THREADS.pop(entry.id, None)
            _CANCEL.pop(entry.id, None)


def _friendly_error(entry: ModelEntry, exc: Exception) -> str:
    """Translate the common download failures into an actionable sentence."""
    text = str(exc)
    lowered = text.lower()
    if "401" in text or "403" in text or "gated" in lowered or "awaiting a review" in lowered:
        return (
            f"Access to {entry.repo_id} was refused. Open {entry.license_url or entry.repo_id} "
            "while signed in to Hugging Face, accept the model terms, then paste a token that "
            "has read access."
        )
    if "404" in text or "not found" in lowered:
        return (
            f"{entry.repo_id} was not found. Check the repository name in the model library; "
            "published names do change."
        )
    if "no space" in lowered or "errno 28" in lowered:
        return "The disk filled up during the download. Free space or choose another models folder."
    if "proxy" in lowered or "connection" in lowered or "timed out" in lowered or "resolve" in lowered:
        return (
            "The download could not reach Hugging Face. Check the network connection, or any "
            "proxy settings, and try again."
        )
    return f"The download failed: {text}"


def start_install(entry: ModelEntry, *, token: str = "") -> InstallStatus:
    """Begin downloading *entry* in the background and return its status."""
    if not entry.repo_id:
        raise ModelInstallError(
            f"{entry.name} has no repository name, so it cannot be downloaded. "
            "Point the catalog entry at a local folder instead."
        )
    with _LOCK:
        existing = _THREADS.get(entry.id)
        if existing is not None and existing.is_alive():
            return read_status(entry.id)

    status = InstallStatus(
        model_id=entry.id,
        state="running",
        message=f"Downloading {entry.name} from {entry.repo_id}.",
        total_bytes=int(entry.approx_size_gb * 1e9),
        started_at=time.time(),
    )
    _write_status(status)

    cancel = threading.Event()
    thread = threading.Thread(
        target=_worker, args=(entry, token, status, cancel), name=f"install-{entry.id}", daemon=True
    )
    with _LOCK:
        _THREADS[entry.id] = thread
        _CANCEL[entry.id] = cancel
    thread.start()
    return status


def cancel_install(model_id: str) -> bool:
    """Ask a running install to stop.  Returns True when a request was sent."""
    with _LOCK:
        cancel = _CANCEL.get(model_id)
    if cancel is None:
        return False
    cancel.set()
    return True


def import_local_model(entry: ModelEntry, source: str | Path, *, move: bool = False) -> Path:
    """Register weights that are already on disk, for machines with no network.

    The files are copied (or moved) into the models folder so that the rest of
    the application has one place to look.  A directory that is already in the
    right place is accepted as-is.
    """
    origin = Path(source).expanduser().resolve()
    if not origin.exists():
        raise ModelInstallError(f"{origin} does not exist.")
    target = entry.target_dir()
    if origin == target.resolve():
        if not entry.is_installed():
            raise ModelInstallError(
                f"{origin} does not look like a complete model folder "
                "(no config.json and no weight files)."
            )
        return target

    if origin.is_file():
        target.mkdir(parents=True, exist_ok=True)
        destination = target / (entry.filename or origin.name)
    else:
        if target.exists() and any(target.iterdir()):
            raise ModelInstallError(
                f"{target} already contains files. Remove the existing copy first."
            )
        destination = target
        target.parent.mkdir(parents=True, exist_ok=True)

    try:
        if origin.is_file():
            shutil.move(str(origin), destination) if move else shutil.copy2(origin, destination)
        else:
            shutil.move(str(origin), destination) if move else shutil.copytree(origin, destination)
    except OSError as exc:
        raise ModelInstallError(f"Could not place the model files: {exc}") from exc

    if not entry.is_installed():
        raise ModelInstallError(
            f"The files were placed in {target} but do not look like a usable "
            f"{entry.backend} model."
        )
    return target


def remove_model(entry: ModelEntry) -> bool:
    """Delete a model's files.  Returns False when nothing was there."""
    target = entry.target_dir()
    if not target.exists():
        return False
    shutil.rmtree(target)
    clear_status(entry.id)
    return True


__all__ = [
    "InstallStatus",
    "ModelInstallError",
    "cancel_install",
    "clear_status",
    "import_local_model",
    "read_status",
    "remove_model",
    "start_install",
]
