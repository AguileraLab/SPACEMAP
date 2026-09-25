"""One run log for every tab.

Each tab has a single "Record run log" box, switched on by default. While it
is on, every action on the tab adds one row to ``<project>/logs/run_log.csv``:
when it started, how long it took, the CPU time, peak RAM and peak GPU memory
it used, the tab's settings and which of them changed since the tab's previous
run, and the machine and software it ran on.

Actions are recorded around the page run in ``app.py``, so no tab has to wire
it in; long jobs that run in their own process (the Leiden worker) call
:func:`record_run` themselves. Tab-specific detail logs (per-sample
registration logs, the per-method phenotyping logs) follow the same switch.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

LABEL = "Record run log"
HELP = (
    "Keeps a record of every run on this tab in the project's logs/run_log.csv: when it "
    "started, how long it took, the CPU time, peak memory (RAM) and GPU memory it used, "
    "the settings it used and which of them changed since the previous run. "
    "On by default. Every settings change is also kept, with old and new values, in the "
    "project file's audit log."
)
RUN_LOG_FILENAME = "run_log.csv"
#: A page update with no button pressed is recorded only when it took this long.
MIN_UNPROMPTED_SECONDS = 2.0
_MAX_SETTINGS_CHARS = 200_000

COLUMNS = [
    "run_id",
    "started_at",
    "finished_at",
    "duration_s",
    "tab",
    "action",
    "status",
    "error",
    "cpu_time_s",
    "cpu_cores_used",
    "peak_ram_gb",
    "ram_change_gb",
    "gpu_name",
    "peak_gpu_memory_gb",
    "settings_changed_json",
    "settings_json",
    "details_json",
    "hostname",
    "cpu_model",
    "cpu_logical_cores",
    "ram_total_gb",
    "python_version",
    "spacemap_version",
    "git_commit",
]

TAB_LABELS = {
    "tab0": "Tab 0 Project",
    "tab1": "Tab 1 Registration",
    "tab2": "Tab 2 Background Subtraction",
    "tab3": "Tab 3 ROI Extraction",
    "tab4": "Tab 4 Annotation",
    "tab5": "Tab 5 Segmentation",
    "tab6": "Tab 6 Batch Correction",
    "tab7": "Tab 7 Phenotyping",
    "tab8": "Tab 8 Spatial Analysis",
    "tab9": "Tab 9 Visualization",
}
#: Project-file settings group for each page (see persistence.get_tab_name_from_key).
TAB_SETTINGS_GROUPS = {
    "tab1": "tab1_registration",
    "tab2": "tab2_background_subtraction",
    "tab3": "tab3_roi_extraction",
    "tab4": "tab4_annotation",
    "tab5": "tab5_segmentation",
    "tab6": "tab6_batch_correction",
    "tab7": "tab7_phenotyping",
    "tab8": "tab8_analysis",
    "tab9": "tab9_visualization",
}


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def page_of(context: str) -> str:
    """``"tab7:leiden"`` -> ``"tab7"``."""
    return str(context).split(":", 1)[0]


def switch_key(context: str) -> str:
    return f"{page_of(context)}_create_log"


def enabled(context: str) -> bool:
    """Whether the tab's run log is on. It is on unless the user switched it off."""
    try:
        return bool(st.session_state.get(switch_key(context), True))
    except Exception:
        return True


def render_switch(context: str) -> bool:
    """The one log box every tab shows, in the same place and with the same name."""
    key = switch_key(context)
    if st.session_state.get(key) is None:
        st.session_state[key] = True

    def _save() -> None:
        try:
            from spacemap_gui.persistence import save_active_project_state

            save_active_project_state()
        except Exception:
            pass

    return st.checkbox(LABEL, key=key, help=HELP, on_change=_save)


def log_path(project_root: str | Path) -> Path:
    return Path(project_root) / "logs" / RUN_LOG_FILENAME


# ---------------------------------------------------------------------------
# Measuring what a run used
# ---------------------------------------------------------------------------

def _process_tree_rss(process) -> int:
    total = 0
    try:
        total += process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                continue
    except Exception:
        pass
    return total


def _process_tree_cpu_seconds(process) -> float:
    try:
        times = process.cpu_times()
        return float(times.user + times.system + times.children_user + times.children_system)
    except Exception:
        return 0.0


def _gpu_memory_mb_for_pids(pids: set[int]) -> float | None:
    """GPU memory held by these processes, from nvidia-smi (None when unavailable)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=3,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    used = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids:
            try:
                used += float(parts[1])
            except ValueError:
                continue
    return used


def _torch_peak_gpu_gb() -> float | None:
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            return float(torch.cuda.max_memory_allocated()) / 1024 ** 3
    except Exception:
        pass
    return None


def _reset_torch_peak() -> None:
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class ResourceMonitor:
    """Samples RAM (and GPU memory, when nvidia-smi exists) while a run is going."""

    def __init__(self, interval_s: float = 0.5, gpu_interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.gpu_interval_s = gpu_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None
        self.ram_start = 0
        self.ram_peak = 0
        self.gpu_peak_mb: float | None = None
        self.cpu_start = 0.0
        self.started = 0.0

    def start(self) -> "ResourceMonitor":
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
        except Exception:
            self._process = None
        self.started = time.perf_counter()
        if self._process is not None:
            self.ram_start = self.ram_peak = _process_tree_rss(self._process)
            self.cpu_start = _process_tree_cpu_seconds(self._process)
        _reset_torch_peak()
        self._thread = threading.Thread(target=self._sample, name="spacemap-run-log", daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        last_gpu = 0.0
        while not self._stop.wait(self.interval_s):
            if self._process is None:
                continue
            self.ram_peak = max(self.ram_peak, _process_tree_rss(self._process))
            now = time.perf_counter()
            if now - last_gpu >= self.gpu_interval_s:
                last_gpu = now
                self._sample_gpu()

    def _sample_gpu(self) -> None:
        if self._process is None:
            return
        try:
            pids = {self._process.pid} | {child.pid for child in self._process.children(recursive=True)}
        except Exception:
            pids = {self._process.pid}
        used = _gpu_memory_mb_for_pids(pids)
        if used is not None:
            self.gpu_peak_mb = max(self.gpu_peak_mb or 0.0, used)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        duration = time.perf_counter() - self.started
        ram_end = _process_tree_rss(self._process) if self._process is not None else 0
        self.ram_peak = max(self.ram_peak, ram_end)
        self._sample_gpu()
        cpu = _process_tree_cpu_seconds(self._process) - self.cpu_start if self._process is not None else 0.0
        gpu_gb = self.gpu_peak_mb / 1024 if self.gpu_peak_mb else None
        torch_peak = _torch_peak_gpu_gb()
        if torch_peak is not None:
            gpu_gb = max(gpu_gb or 0.0, torch_peak)
        return {
            "duration_s": duration,
            "cpu_time_s": max(cpu, 0.0),
            "peak_ram_gb": self.ram_peak / 1024 ** 3 if self.ram_peak else None,
            "ram_change_gb": (ram_end - self.ram_start) / 1024 ** 3 if self._process is not None else None,
            "peak_gpu_memory_gb": gpu_gb,
        }


# ---------------------------------------------------------------------------
# Machine and software
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def machine_info() -> dict[str, Any]:
    cpu_model = platform.processor() or ""
    if not cpu_model:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name") and ":" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    ram_total = ""
    try:
        import psutil

        ram_total = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    gpu_name = ""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=False, capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                gpu_name = "; ".join(line.strip() for line in result.stdout.splitlines() if line.strip())
        except Exception:
            pass
    try:
        from importlib import metadata

        version = metadata.version("spacemap")
    except Exception:
        version = ""
    git_commit = ""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            check=False, capture_output=True, text=True, timeout=3,
        )
        git_commit = result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        pass
    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_logical_cores": os.cpu_count() or "",
        "ram_total_gb": ram_total,
        "gpu_name": gpu_name,
        "python_version": platform.python_version(),
        "spacemap_version": version,
        "git_commit": git_commit,
    }


# ---------------------------------------------------------------------------
# Settings and what changed
# ---------------------------------------------------------------------------

def _text_keys(value: Any) -> Any:
    """The same value with every dict key as text, so keys can be sorted.

    Some settings are dicts keyed by numbers (per-row visibility, for example)
    inside dicts keyed by names; sorting such mixed keys raises TypeError.
    """
    if isinstance(value, dict):
        return {str(key): _text_keys(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_text_keys(item) for item in value]
    return value


def _json(value: Any) -> str:
    """Compact, sorted JSON for the log (long values are cut short)."""
    try:
        text = json.dumps(_text_keys(value), default=str, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = json.dumps(str(value))
    return text if len(text) <= _MAX_SETTINGS_CHARS else text[: _MAX_SETTINGS_CHARS] + "...(truncated)"


def settings_snapshot(context: str) -> dict[str, Any]:
    """The tab's saved settings: the same keys the project file stores for it."""
    group = TAB_SETTINGS_GROUPS.get(page_of(context))
    if not group:
        return {}
    try:
        from spacemap_gui.persistence import _to_json_compatible, get_tab_name_from_key, is_transient_key
    except Exception:
        return {}
    snapshot: dict[str, Any] = {}
    try:
        items = list(st.session_state.items())
    except Exception:
        return {}
    for key, value in items:
        if not isinstance(key, str) or is_transient_key(key) or get_tab_name_from_key(key) != group:
            continue
        try:
            converted = _to_json_compatible(value)
            text = json.dumps(converted, default=str)
        except Exception:
            continue
        if len(text) > 20_000:
            converted = f"<{len(text):,} characters, not logged>"
        snapshot[key] = converted
    return snapshot


def _state_path(project_root: Path) -> Path:
    return Path(project_root) / "logs" / ".run_log_last_settings.json"


def settings_changes(project_root: Path, context: str, current: dict[str, Any]) -> dict[str, Any]:
    """Settings that differ from the tab's previous logged run, as {key: [old, new]}."""
    path = _state_path(project_root)
    try:
        previous_all = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        previous_all = {}
    if not isinstance(previous_all, dict):  # written by an older build as one string
        previous_all = {}
    previous = previous_all.get(page_of(context))
    changes: dict[str, Any] = {}
    if isinstance(previous, dict):
        for key in sorted(set(previous) | set(current)):
            old, new = previous.get(key), current.get(key)
            if _json(old) != _json(new):
                changes[key] = [old, new]
    previous_all[page_of(context)] = current
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Complete JSON, never cut short, in a temporary file of this process's own.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(_text_keys(previous_all), default=str, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass
    return changes


# ---------------------------------------------------------------------------
# Writing a row
# ---------------------------------------------------------------------------

def _round(value, digits=3):
    return "" if value is None else round(float(value), digits)


def record_run(
    project_root: str | Path,
    context: str,
    action: str,
    *,
    started_at: datetime,
    usage: dict[str, Any],
    status: str = "completed",
    error: str = "",
    settings: dict[str, Any] | None = None,
    settings_changed: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> Path | None:
    """Append one row to the project run log. Never raises: logging must not stop work."""
    try:
        machine = machine_info()
        duration = float(usage.get("duration_s") or 0.0)
        cpu = float(usage.get("cpu_time_s") or 0.0)
        row = {column: "" for column in COLUMNS}
        row.update({key: machine.get(key, "") for key in COLUMNS if key in machine})
        row.update(
            {
                "run_id": f"{page_of(context)}-{started_at.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}",
                "started_at": started_at.isoformat(timespec="seconds"),
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_s": _round(duration),
                "tab": TAB_LABELS.get(page_of(context), page_of(context)),
                "action": action,
                "status": status,
                "error": error,
                "cpu_time_s": _round(cpu),
                "cpu_cores_used": _round(cpu / duration, 2) if duration > 0 else "",
                "peak_ram_gb": _round(usage.get("peak_ram_gb")),
                "ram_change_gb": _round(usage.get("ram_change_gb")),
                "peak_gpu_memory_gb": _round(usage.get("peak_gpu_memory_gb")),
                "settings_changed_json": _json(settings_changed or {}),
                "settings_json": _json(settings or {}),
                "details_json": _json(details or {}),
            }
        )
        path = log_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+", newline="", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        return path
    except Exception:
        return None


def read_rows(project_root: str | Path, context: str | None = None, limit: int = 50) -> list[dict[str, str]]:
    """The most recent rows, optionally for one tab, newest last."""
    path = log_path(project_root)
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return []
    if context:
        label = TAB_LABELS.get(page_of(context), page_of(context))
        rows = [row for row in rows if row.get("tab") == label]
    return rows[-limit:]


# ---------------------------------------------------------------------------
# Recording page runs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _button_labels() -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    """Map button keys to their labels, from the interface source."""
    exact: dict[str, str] = {}
    prefixes: list[tuple[str, str]] = []
    root = Path(__file__).resolve().parent
    for source in root.rglob("*.py"):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in ("button", "form_submit_button", "download_button"):
                continue
            label = ""
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                label = node.args[0].value
            elif node.args and isinstance(node.args[0], ast.JoinedStr):
                label = "".join(
                    part.value if isinstance(part, ast.Constant) else "..." for part in node.args[0].values
                )
            for keyword in node.keywords:
                if keyword.arg != "key":
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    exact.setdefault(keyword.value.value, label)
                elif isinstance(keyword.value, ast.JoinedStr) and keyword.value.values:
                    head = keyword.value.values[0]
                    if isinstance(head, ast.Constant) and isinstance(head.value, str) and len(head.value) >= 4:
                        prefixes.append((head.value, label))
    prefixes.sort(key=lambda item: -len(item[0]))
    return exact, tuple(prefixes)


def clicked_action() -> tuple[str, str]:
    """The button pressed in this rerun, as (label, key); ("", "") when none is known."""
    try:
        from spacemap_gui.persistence import is_action_widget_key
    except Exception:
        return "", ""
    try:
        items = list(st.session_state.items())
    except Exception:
        return "", ""
    for key, value in items:
        if value is not True or not isinstance(key, str) or not is_action_widget_key(key):
            continue
        exact, prefixes = _button_labels()  # scanned once, only when a button was pressed
        label = exact.get(key)
        if label is None:
            label = next((text for prefix, text in prefixes if key.startswith(prefix)), "")
        return (label or key), key
    return "", ""


@contextmanager
def page_run(context: str):
    """Record one page run: which button started it, the time and resources it used."""
    project_root = None
    try:
        project_root = st.session_state.get("active_project_root")
    except Exception:
        pass
    if not project_root or not enabled(context):
        yield
        return

    action, action_key = clicked_action()
    # The settings a button ran with are the ones in place when it was pressed.
    # A page update with no button is snapshotted only if it ends up recorded.
    settings = settings_snapshot(context) if action else None
    started_at = datetime.now().astimezone()
    monitor = ResourceMonitor().start()
    status, error = "completed", ""
    try:
        yield
    except BaseException as exc:
        # st.rerun() and st.stop() end a run normally.
        if type(exc).__name__ not in ("RerunException", "StopException"):
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        usage = monitor.stop()
        if action or usage["duration_s"] >= MIN_UNPROMPTED_SECONDS:
            if settings is None:
                settings = settings_snapshot(context)
            changed = settings_changes(Path(project_root), context, settings)
            record_run(
                project_root,
                context,
                action or "Page update",
                started_at=started_at,
                usage=usage,
                status=status,
                error=error,
                settings=settings,
                settings_changed=changed,
                details={"button_key": action_key} if action_key else None,
            )
