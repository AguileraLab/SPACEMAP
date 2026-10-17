"""Small, shared settings using the classic persistence keys and size guards."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from contextlib import contextmanager
from functools import wraps
from spacemap.sharing import open_lock

_LOCKS = {}
_GUARD = threading.Lock()
_HELD = threading.local()


def locked_save(function):
    @wraps(function)
    def call(project, *args, **kwargs):
        with project_lock(project, tolerate_read_only=True):
            return function(project, *args, **kwargs)

    return call


def locked_sync(function):
    @wraps(function)
    def call(*args, **kwargs):
        import streamlit as st

        project = st.session_state.get("active_project_root")
        if not project:
            return function(*args, **kwargs)
        with project_lock(project, tolerate_read_only=True):
            return function(*args, **kwargs)

    return call


@contextmanager
def project_lock(project, *, tolerate_read_only=False):
    import fcntl

    key = str(Path(project).resolve())
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_HELD, "projects", set())
        if key in held:
            yield
            return
        path = Path(project) / ".settings.lock"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open_lock(path)
        except OSError:
            if not tolerate_read_only:
                raise
            yield
            return
        with handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            _HELD.projects = held | {key}
            try:
                yield
            finally:
                _HELD.projects = held


def config_path(project):
    root = Path(project)
    path = root / (root.name + ".json")
    return path if path.exists() or not (root / "experiment.json").exists() else root / "experiment.json"


def read_config(project):
    path = config_path(project)
    if not path.exists():
        return {}
    if path.stat().st_size > 1_000_000:
        raise ValueError("Open this project in classic SPACEMAP to reduce its saved settings first.")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("The project settings could not be read.")
    return value


def revision(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _resolve_path(project):
    return Path(project) / "outputs" / "phenotyping" / "cache" / "resolve_params.json"


def resolve_values(project):
    config = read_config(project)
    value = deepcopy(config.get("tab_configurations", {}).get("tab7_phenotyping", {}).get("resolve_config", {}))
    # Tab 7 also saves the same options for independent/background readers.
    path = _resolve_path(project)
    if path.exists():
        cached = json.loads(path.read_text())
        if isinstance(cached, dict):
            value = (
                (
                    {**value, **cached}
                    if path.stat().st_mtime_ns >= config_path(project).stat().st_mtime_ns
                    else {**cached, **value}
                )
                if config_path(project).exists()
                else cached
            )
    defaults = {
        "output_col": "phenotype_threshold",
        "unclassified_label": "Others",
        "tie_score_transform": "identity_residual",
        "tie_score_cofactor": 5.0,
        "norm_z1": True,
        "norm_z2": False,
        "norm_cdf": False,
        "norm_nlog": False,
        "norm_order": ["Z1"],
        "threshold_statistic": "median",
        "threshold_source": "Auto",
        "threshold_mode": "Predefined (CSV)",
        "threshold_percentile": 95.0, "threshold_manual": 0.0,
        "z1_cutoff": 1.5,
        "transformation": "None",
        "normalization_slots": ["Z1", "None", "None", "None"],
        "enable_filtering": False,
        "filter_column": "batch_id",
        "filter_value": "",
        "batch_col": "batch_id",
        "cell_id_col": "cell_id",
        "threshold_marker_col": "",
        "threshold_value_col": "",
        "polygon_prefix": "Polygon",
        "cluster_col": "resolve_cluster",
    }
    return {**defaults, **value}


def get(project):
    values = resolve_values(project)
    return {"values": values, "revision": revision(values)}


def save(project, changes, expected):
    from spacemap_gui import persistence
    from spacemap_workflows.leiden_job import write_json

    allowed = {
        "output_col",
        "unclassified_label",
        "tie_score_transform",
        "tie_score_cofactor",
        "norm_z1",
        "norm_z2",
        "norm_cdf",
        "norm_nlog",
    }
    if not isinstance(changes, dict) or set(changes) - allowed:
        raise ValueError("Only the listed RESOLVE settings can be changed here.")
    if persistence.estimate_json_bytes(changes, limit=64000) > 64000:
        raise ValueError("Settings must stay small; do not include cell tables.")
    for key, value in changes.items():
        if key.startswith("norm_") and not isinstance(value, bool):
            raise ValueError("Choose on or off for normalization.")
        if key == "tie_score_cofactor" and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 65535
        ):
            raise ValueError("The Arcsinh cofactor must be between 0 and 65,535.")
        if key in {"output_col", "unclassified_label"} and (
            not isinstance(value, str) or not 0 < len(value.strip()) <= 128
        ):
            raise ValueError("Enter a short name.")
        if key == "output_col" and not value.startswith("phenotype_"):
            raise ValueError("Start the output column name with phenotype_.")
        if key == "tie_score_transform" and value not in {"identity_residual", "asinh_difference", "log1p_difference"}:
            raise ValueError("Choose Raw, Arcsinh or Log1p for the scores.")
    with project_lock(project):
        current = resolve_values(project)
        if expected != revision(current):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        current.update(changes)
        current["transformation"] = {
            "identity_residual": "None",
            "asinh_difference": "ArcSinh",
            "log1p_difference": "Log",
        }[current["tie_score_transform"]]
        if any(key.startswith("norm_") for key in changes):
            current["norm_order"] = [
                name
                for name, key in [("Z1", "norm_z1"), ("Z2", "norm_z2"), ("CDF", "norm_cdf"), ("NLog", "norm_nlog")]
                if current.get(key)
            ]
            current["normalization_slots"] = (current["norm_order"] + ["None"] * 4)[:4]
        if current["norm_nlog"] and not current["norm_cdf"]:
            raise ValueError("Turn on CDF before using NLog, as in classic RESOLVE.")
        config = read_config(project)
        persistence.update_stored_config_key(config, "resolve_config", current)
        path = _resolve_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.read_text() if path.exists() else None
        write_json(path, current)
        persistence.save_project_config(str(project), config)
        saved = read_config(project)
        if saved.get("tab_configurations", {}).get("tab7_phenotyping", {}).get("resolve_config") != current:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                write_json(path, json.loads(previous))
            raise ValueError("The project settings were not saved. Check the project folder permissions.")
        from spacemap.viewer.live_resolve import invalidate_settings

        invalidate_settings(path)
    return get(project)


def pull_into_classic(project, config, session):
    """Three-way merge: a changed widget keeps the user's in-progress edit."""
    disk = read_config(project)
    saved = disk.get("tab_configurations", {})
    original = config.setdefault("tab_configurations", {})
    changed = []
    from spacemap_gui.persistence import values_equal

    from .segmentation import EDITABLE

    shared = {"resolve_config"} | EDITABLE  # the keys Studio saves
    for tab, fields in saved.items():
        if not isinstance(fields, dict):
            continue
        target = original.setdefault(tab, {})
        for key, value in fields.items():
            if key not in shared:
                continue
            old = target.get(key)
            if values_equal(old, value):
                continue
            live = session.get(key, old)
            if isinstance(old, dict) and isinstance(value, dict) and isinstance(live, dict):
                merged = deepcopy(live)
                for field, remote in value.items():
                    if values_equal(live.get(field), old.get(field)):
                        merged[field] = deepcopy(remote)
                session[key] = merged
                target[key] = deepcopy(value)
                changed.append(key)
                widgets = {
                    "output_col": "resolve_output_col",
                    "unclassified_label": "resolve_unclassified_label",
                    "tie_score_cofactor": "resolve_arcsinh_cofactor",
                    "transformation": "resolve_transform_choice",
                    "input_space": "resolve_input_space",
                    "adaptive_hierarchy": "resolve_adaptive_hierarchy",
                    "threshold_mode": "resolve_threshold_mode",
                    "threshold_percentile": "resolve_threshold_percentile",
                    "threshold_manual": "resolve_threshold_manual",
                    "threshold_statistic": "resolve_threshold_stat",
                    "threshold_source": "resolve_threshold_source",
                    "z1_cutoff": "resolve_z1_cutoff",
                    "batch_col": "resolve_batch_col",
                }
                for field, widget in widgets.items():
                    if widget in session and values_equal(session[widget], old.get(field)):
                        session[widget] = deepcopy(merged.get(field))
                for i in range(4):
                    widget = f"resolve_normalization_{i + 1}"
                    before = (old.get("normalization_slots", []) + ["None"] * 4)[i]
                    if widget in session and values_equal(session[widget], before):
                        session[widget] = (merged.get("normalization_slots", []) + ["None"] * 4)[i]
            elif values_equal(live, old):
                session[key] = deepcopy(value)
                target[key] = deepcopy(value)
                changed.append(key)
            # A changed widget is an active edit and is saved by normal sync.
    if changed:
        session["_studio_settings_notice"] = (
            "Settings updated from Studio: " + ", ".join(key.replace("_", " ") for key in changed) + "."
        )
    return changed
