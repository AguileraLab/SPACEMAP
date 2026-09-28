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
#: Starts the server and stops it when SPACEMAP stops (see model_service/guard.py).
GUARD_PATH = Path(__file__).resolve().parent / "model_service" / "guard.py"
#: The scripts a model server runs, whichever copy of SPACEMAP started it.
SERVER_SCRIPTS = ("spacemap_assistant/model_service/server.py", "spacemap_assistant/model_service/guard.py")

#: Precisions the UI may request.  ``auto`` asks the hardware probe to choose.
PRECISIONS = ("auto", "float32", "bfloat16", "int8", "int4")

#: How long to wait for the server process to stay up before reporting failure.
STARTUP_GRACE_SECONDS = 1.0


#: Where the assistant's settings are, for messages.
SETUP_PLACE = "AI setup (Tab 0 > AI Assistant, or Setup on the assistant page)"


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
    port: int = 0


def _state_path() -> Path:
    """Where the running server's process record is kept: one per node.

    SPACEMAP's settings folder is shared by every node of a cluster, but a
    process number means something only on the node that started it. With one
    record, starting the model on a second node made the first node's server
    look like "a process SPACEMAP did not start", holding the port.
    """
    from spacemap.userpaths import config_dir

    return config_dir() / f"model_server.{socket.gethostname()}.json"


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


def _is_model_server(argv: list[str]) -> bool:
    """A model server as SPACEMAP starts one: Python running the server or its guard, or ``-m llama_cpp.server``.

    Only the program and its script count: a shell or editor whose command merely mentions them is not a server.
    """
    if len(argv) < 2 or not Path(argv[0]).name.startswith("python"):
        return False
    return argv[1].replace("\\", "/").endswith(SERVER_SCRIPTS) or argv[1:3] == ["-m", "llama_cpp.server"]


def _pid_matches(pid: int) -> bool:
    """True when *pid* is still one of this user's model servers on this node.

    Any SPACEMAP process used to match (its command line names spacemap), so a
    record from another node could point at this node's SPACEMAP app itself.
    """
    if pid <= 1:
        return False
    try:
        import psutil

        process = psutil.Process(pid)
        if process.uids().real != os.getuid():
            return False
        return _is_model_server(process.cmdline())
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - psutil.NoSuchProcess, AccessDenied, ZombieProcess
        return False
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\x00")
    except OSError:
        return False
    return _is_model_server([part for part in argv if part])


def _option(cmdline: list[str], name: str) -> str:
    try:
        return cmdline[cmdline.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def model_servers() -> list[dict[str, Any]]:
    """This user's model servers on this node: pid, port, model and start time (the guard, not its child)."""
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for process in psutil.process_iter(["pid", "ppid", "uids", "cmdline", "create_time"]):
        info = process.info
        cmdline = info.get("cmdline") or []
        if not cmdline or not info.get("uids") or info["uids"].real != os.getuid() or not _is_model_server(cmdline):
            continue
        found.append({"pid": info["pid"], "ppid": info["ppid"], "cmdline": cmdline, "started": info["create_time"]})
    guards = {item["pid"] for item in found if item["cmdline"][1].endswith("guard.py")}
    servers = []
    for item in found:
        if item["ppid"] in guards:
            continue  # the server a guard runs is listed through its guard
        cmdline = item["cmdline"]
        model = _option(cmdline, "--model-id") or Path(_option(cmdline, "--model") or "").name
        port = _option(cmdline, "--port")
        servers.append(
            {"pid": item["pid"], "port": int(port) if port.isdigit() else 0, "model": model, "started": int(item["started"])}
        )
    return sorted(servers, key=lambda item: item["started"])


def leftover_servers() -> list[dict[str, Any]]:
    """Model servers of this user on this node that no SPACEMAP is using (left by an earlier session)."""
    record = _read_pid_record()
    current = record.get("pid") if record.get("host", socket.gethostname()) == socket.gethostname() else None
    return [item for item in model_servers() if item["pid"] != current]


def _stop_pid(pid: int) -> None:
    """Stop a server and what it runs: its whole group when it leads one (as SPACEMAP starts it), else itself."""
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
            return
    except OSError:
        pass
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def stop_leftover_servers(pids: list[int] | None = None) -> list[dict[str, Any]]:
    """Stop the model servers earlier SPACEMAP sessions left on this node (only *pids*, if given)."""
    stopped = [item for item in leftover_servers() if pids is None or item["pid"] in pids]
    for item in stopped:
        _stop_pid(item["pid"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and any(_pid_matches(item["pid"]) for item in stopped):
        time.sleep(0.2)
    return stopped


def _write_pid_record(pid: int, precision: str, entry: ModelEntry) -> None:
    target = _state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid,
        "host": socket.gethostname(),
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


def _port_in_use(port: int | None = None) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port or _server_port())) == 0
    except OSError:
        return False


def _free_port(start: int) -> int:
    """The first port after *start* nothing listens on and this process can take."""
    for port in range(start + 1, start + 200):
        if _port_in_use(port):
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        return port
    raise ModelRuntimeError(f"No free port found after {start}. Change the port in the assistant settings.")


def _make_room() -> str:
    """Before starting: stop a server an earlier session left on the port, or move to a free port.

    Returns a note for the log (empty when the port was free).
    """
    port = _server_port()
    if not _port_in_use(port):
        return ""
    ours = [item for item in leftover_servers() if item["port"] == port]
    if ours:
        stop_leftover_servers([item["pid"] for item in ours])
        for _ in range(50):
            if not _port_in_use(port):
                return f"Stopped a model server an earlier SPACEMAP session left on port {port}."
            time.sleep(0.2)
    moved = _free_port(port)
    ai_settings.update_settings(server_port=moved)
    return f"Port {port} is used by another program on this node; the model uses port {moved} instead."


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
            f"No model is chosen yet. Choose one in {SETUP_PLACE}.",
        )
    if not entry.is_installed():
        return RuntimeStatus(
            False, False, False, None,
            f"{entry.name} is not downloaded yet. Download it in {SETUP_PLACE}.",
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
            model_id=entry.id, backend=entry.backend, port=_server_port(),
        )

    if _port_in_use():
        left = [item for item in leftover_servers() if item["port"] == _server_port()]
        return RuntimeStatus(
            True, False, False, None,
            f"An earlier SPACEMAP session left a model server on port {_server_port()}; starting the model stops it."
            if left
            else f"Port {_server_port()} is used by another program on this node; starting the model picks a free port.",
            model_id=entry.id, backend=entry.backend, port=_server_port(),
        )
    return RuntimeStatus(
        True, False, False, None, f"{entry.name} is installed but stopped.",
        model_id=entry.id, backend=entry.backend, port=_server_port(),
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
    room = _make_room()

    resolved = resolve_precision(precision, entry)
    _ensure_api_key()
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(entry, resolved)
    if ai_settings.load_settings().get("stop_with_spacemap", True):
        started = _process_start_time(os.getpid())
        if started is not None:
            command = [str(_interpreter()), str(GUARD_PATH), str(os.getpid()), repr(started), "--", *command]

    try:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {entry.id} "
                f"({entry.backend}) in {resolved} mode on port {_server_port()} on {socket.gethostname()}\n"
                + (room + "\n" if room else "")
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
            + f"Install the missing packages into SPACEMAP's environment: {SETUP_PLACE} "
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


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 - without psutil the server is simply not guarded
        return None


def _log_tail(lines: int = 12) -> str:
    try:
        return "\n".join(_log_path().read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def ensure_running(*, precision: str | None = None, timeout: float = 900.0, on_wait=None) -> RuntimeStatus:
    """Start the chosen local model if it is stopped, and wait until it answers.

    Lets the chat start the model by itself when a question is asked, instead
    of sending you to a settings page first. *on_wait(status, seconds)* is
    called every few seconds while it loads. Raises ModelRuntimeError when the
    model is not downloaded, stops while loading, or is not ready in time.
    """
    status = runtime_status()
    if status.ready:
        return status
    if not status.installed:
        raise ModelRuntimeError(status.message)
    if not status.running:
        if precision is None:
            from spacemap.ai_settings import load_settings

            precision = str(load_settings().get("precision") or "auto")
        start_runtime(precision)
    started = time.monotonic()
    while True:
        status = runtime_status()
        if status.ready:
            return status
        waited = time.monotonic() - started
        if not status.running:
            raise ModelRuntimeError(f"The model stopped while loading.\n{_log_tail()}")
        if waited > timeout:
            raise ModelRuntimeError(f"The model is still loading after {waited:.0f} s.\n{_log_tail()}")
        if on_wait is not None:
            on_wait(status, waited)
        time.sleep(3)


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
