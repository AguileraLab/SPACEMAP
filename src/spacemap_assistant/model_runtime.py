"""Start, stop and inspect the local model server.

Earlier versions managed one fixed model, in one fixed conda environment, on
one fixed port, and refused to start without a CUDA GPU.  This version resolves
all of that from settings:

* the **model** comes from the catalog entry named in the assistant settings;
* the **interpreter** is the one running SPACEMAP, unless a dedicated
  environment is configured for sites that keep AI dependencies separate;
* the **port** is configurable, so two copies can run side by side;
* **CPU is a supported target**.  A machine with no GPU loads the model in
  float32 and answers slowly, which is far better than refusing to run.

Two backends are supported.  ``transformers`` launches the bundled FastAPI
server; ``gguf`` launches ``llama_cpp.server``, which speaks the same
OpenAI-style protocol and is the practical option on a laptop.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spacemap import ai_settings
from spacemap.hardware import detect_hardware, recommended_precision
from spacemap.model_catalog import ModelCatalogError, ModelEntry, get_model
from spacemap.userpaths import cache_dir, logs_dir

from .model_client import MODEL_KEY_PATH, LocalModelClient, ModelUnavailable

#: The bundled model server, shipped inside the package so that it is present
#: in a normal installation and not only in a git checkout.
SERVER_PATH = Path(__file__).resolve().parent / "model_service" / "server.py"

#: Precisions the UI may request.  ``auto`` asks the hardware probe to choose.
PRECISIONS = ("auto", "float32", "bfloat16", "int8", "int4")

#: How long to wait for the server process to stay up before reporting failure.
STARTUP_GRACE_SECONDS = 1.0


class ModelRuntimeError(RuntimeError):
    """Raised when the local model runtime cannot be managed."""


@dataclass(frozen=True)
class RuntimeStatus:
    """What the UI needs to know about the local model server."""

    installed: bool
    running: bool
    ready: bool
    pid: int | None
    message: str
    precision: str = ""
    gpu: str = ""
    model_id: str = ""
    backend: str = ""


def _state_path() -> Path:
    """Where the running server's process record is kept."""
    from spacemap.userpaths import config_dir

    return config_dir() / "model_server.json"


def _log_path() -> Path:
    return logs_dir() / "model-server.log"


def active_entry() -> ModelEntry | None:
    """Return the selected catalog entry, or None when nothing is selected."""
    model_id = str(ai_settings.load_settings().get("active_model") or "").strip()
    if not model_id:
        return None
    try:
        return get_model(model_id)
    except ModelCatalogError:
        return None


def _server_port() -> int:
    try:
        return int(ai_settings.load_settings().get("server_port") or 8091)
    except (TypeError, ValueError):
        return 8091


def _server_host() -> str:
    # Binding anywhere but loopback would expose an unauthenticated-by-default
    # model server to the network, so the host is clamped here.
    host = str(ai_settings.load_settings().get("server_host") or "127.0.0.1")
    return host if host in {"127.0.0.1", "localhost", "::1"} else "127.0.0.1"


def _interpreter() -> Path:
    """The Python used to run the model server.

    A site that keeps heavy AI dependencies in a separate environment can set
    ``SPACEMAP_MODEL_PYTHON``.  Everyone else uses the interpreter already
    running SPACEMAP, which is what a plain ``pip install`` produces.
    """
    configured = os.environ.get("SPACEMAP_MODEL_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise ModelRuntimeError(
                f"SPACEMAP_MODEL_PYTHON points at {candidate}, which does not exist."
            )
        return candidate
    return Path(sys.executable)


def _read_pid_record() -> dict[str, Any]:
    try:
        payload = json.loads(_state_path().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _pid_matches(pid: int) -> bool:
    """True when *pid* is still a model server this application started."""
    if pid <= 1:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        # /proc is Linux-only; fall back to asking whether the process exists.
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    return "spacemap" in cmdline.lower() or "llama_cpp.server" in cmdline or str(SERVER_PATH) in cmdline


def _write_pid_record(pid: int, precision: str, entry: ModelEntry) -> None:
    target = _state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "precision": precision,
        "model": entry.id,
        "backend": entry.backend,
        "port": _server_port(),
        "started_at": int(time.time()),
    }
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_api_key() -> None:
    """Create the shared secret the GUI uses to talk to the local server."""
    target = MODEL_KEY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if len(target.read_text(encoding="utf-8").strip()) >= 24:
            return
    except OSError:
        pass
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(secrets.token_urlsafe(48) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _port_in_use() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", _server_port())) == 0
    except OSError:
        return False


def gpu_status() -> dict[str, Any]:
    """Describe the accelerator, if any.  Never raises."""
    report = detect_hardware()
    if report.gpus:
        first = report.gpus[0]
        return {
            "available": True,
            "count": len(report.gpus),
            "name": first.name,
            "memory_gib": round(first.total_memory_gb * 1e9 / 1024**3, 1),
        }
    if report.mps_available:
        return {"available": True, "count": 1, "name": "Apple Metal", "memory_gib": 0}
    reason = (
        "PyTorch is not installed, so no GPU can be used."
        if not report.torch_available
        else "No CUDA or Metal device is visible. The model will run on CPU."
    )
    return {"available": False, "count": 0, "name": "", "memory_gib": 0, "reason": reason}


def resolve_precision(requested: str, entry: ModelEntry) -> str:
    """Turn a requested precision into one this machine can actually load."""
    if requested not in PRECISIONS:
        raise ModelRuntimeError(f"Precision must be one of {', '.join(PRECISIONS)}.")
    report = detect_hardware()
    if requested == "auto":
        return recommended_precision(report, entry.approx_size_gb)
    if requested in {"int8", "int4"} and not report.cuda_available:
        raise ModelRuntimeError(
            "Quantized loading needs a CUDA GPU because bitsandbytes has no CPU build. "
            "Choose float32 to run on the CPU, or install a GGUF model instead."
        )
    if requested == "bfloat16" and not report.cuda_available:
        # CPU bfloat16 is supported by torch but is slower than float32 on most
        # hardware and silently degrades quality on older chips.
        return "float32"
    return requested


def runtime_status() -> RuntimeStatus:
    """Inspect the configured model and any running server."""
    entry = active_entry()
    if entry is None:
        return RuntimeStatus(
            False, False, False, None,
            "No model is selected. Choose one in Tab 0 under AI Assistant.",
        )
    if not entry.is_installed():
        return RuntimeStatus(
            False, False, False, None,
            f"{entry.name} is not installed yet. Download it in Tab 0 under AI Assistant.",
            model_id=entry.id, backend=entry.backend,
        )

    record = _read_pid_record()
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        pid = None
    running = bool(pid and _pid_matches(pid))

    if running:
        try:
            health = LocalModelClient(timeout_seconds=3).health()
        except ModelUnavailable:
            return RuntimeStatus(
                True, True, False, pid,
                f"{entry.name} is starting and loading its weights.",
                str(record.get("precision") or ""), model_id=entry.id, backend=entry.backend,
            )
        ready = health.get("status") == "ready"
        return RuntimeStatus(
            True, True, ready, pid,
            f"{entry.name} is ready." if ready else f"{entry.name} is loading.",
            str(health.get("precision") or record.get("precision") or ""),
            str(health.get("gpu") or ""),
            model_id=entry.id, backend=entry.backend,
        )

    if _port_in_use():
        return RuntimeStatus(
            True, False, False, None,
            f"Port {_server_port()} is in use by a process SPACEMAP did not start. "
            "Stop it, or change the port in the assistant settings.",
            model_id=entry.id, backend=entry.backend,
        )
    return RuntimeStatus(
        True, False, False, None, f"{entry.name} is installed but stopped.",
        model_id=entry.id, backend=entry.backend,
    )


def _build_command(entry: ModelEntry, precision: str) -> list[str]:
    """The command line that launches the right server for this backend."""
    interpreter = str(_interpreter())
    if entry.backend == "gguf":
        target = entry.target_file()
        if target is None or not target.is_file():
            raise ModelRuntimeError(f"{entry.name} has no GGUF file on disk.")
        return [
            interpreter, "-m", "llama_cpp.server",
            "--model", str(target),
            "--host", _server_host(),
            "--port", str(_server_port()),
            "--api_key", _read_api_key(),
        ]
    return [
        interpreter, str(SERVER_PATH),
        "--model-dir", str(entry.target_dir()),
        "--model-id", entry.repo_id or entry.id,
        "--host", _server_host(),
        "--port", str(_server_port()),
        "--precision", precision,
        "--api-key-file", str(MODEL_KEY_PATH),
    ]


def _read_api_key() -> str:
    try:
        return MODEL_KEY_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _server_environment() -> dict[str, str]:
    """Environment for the server process."""
    env = os.environ.copy()
    # A local server must never be reached through an institutional proxy: the
    # request would leave the machine along with its API key.
    for proxy_name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        env.pop(proxy_name, None)
    hf_home = cache_dir() / "huggingface"
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            # The weights are already local; refusing network access here means
            # a stale reference can never turn into a silent download.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    return env


def start_runtime(precision: str = "auto") -> RuntimeStatus:
    """Launch the local model server for the selected model."""
    current = runtime_status()
    if current.running:
        return current
    entry = active_entry()
    if entry is None or not entry.is_installed():
        raise ModelRuntimeError(current.message)
    if _port_in_use():
        raise ModelRuntimeError(
            f"Port {_server_port()} is already in use. Stop the other process or "
            "change the port in the assistant settings."
        )

    resolved = resolve_precision(precision, entry)
    _ensure_api_key()
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(entry, resolved)

    try:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {entry.id} "
                f"({entry.backend}) in {resolved} mode on port {_server_port()}\n"
            )
            log.flush()
            process = subprocess.Popen(
                command,
                env=_server_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except FileNotFoundError as exc:
        raise ModelRuntimeError(
            f"Could not start the model server: {exc}. "
            + "Install the missing packages into SPACEMAP's environment from Tab 0 -> AI Assistant (optional), which "
            "shows what is missing and the exact command"
            + (" (a GGUF model needs llama-cpp-python)." if entry.backend == "gguf" else ".")
        ) from exc
    except OSError as exc:
        raise ModelRuntimeError(f"Could not start the model server: {exc}") from exc

    _write_pid_record(process.pid, resolved, entry)
    time.sleep(STARTUP_GRACE_SECONDS)
    if process.poll() is not None:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        except OSError:
            pass
        raise ModelRuntimeError(f"{entry.name} stopped during startup.\n{tail}")
    return runtime_status()


def stop_runtime() -> RuntimeStatus:
    """Stop the local model server if this application started it."""
    record = _read_pid_record()
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        pid = 0
    if pid and _pid_matches(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                raise ModelRuntimeError(f"Could not stop the model server: {exc}") from exc
        for _ in range(50):
            if not _pid_matches(pid):
                break
            time.sleep(0.1)
        if _pid_matches(pid):
            raise ModelRuntimeError(
                "The model server is still stopping. Wait a few seconds and try again."
            )
    try:
        _state_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ModelRuntimeError(f"Could not remove the process record: {exc}") from exc
    return runtime_status()


__all__ = [
    "PRECISIONS",
    "ModelRuntimeError",
    "RuntimeStatus",
    "active_entry",
    "gpu_status",
    "resolve_precision",
    "runtime_status",
    "start_runtime",
    "stop_runtime",
]
