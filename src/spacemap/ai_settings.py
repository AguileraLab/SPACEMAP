"""Settings that decide whether the optional AI assistant runs at all.

The assistant is an optional component.  SPACEMAP must start, and every
pipeline tab must work, on a machine that has none of the AI dependencies
installed, no model weights on disk, no GPU and no network access.  This module
is the single place that answers "is the assistant switched on?", and it is
deliberately free of heavy imports so that the question can be asked before any
assistant code is loaded.

The settings file lives in the user configuration directory and is written by
the GUI.  An administrator deploying SPACEMAP for a group can also set
``SPACEMAP_ASSISTANT=off`` to force it off for everyone regardless of the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .userpaths import config_dir, models_dir

SETTINGS_FILENAME = "assistant.json"
SCHEMA_VERSION = 1

#: Backends the UI can select.  ``local`` runs a model on this machine,
#: ``api`` calls a remote provider, ``none`` keeps the assistant switched on but
#: unconfigured (it will explain what to set up).
BACKENDS = ("local", "api", "none")

#: The assistant ships switched off.  A first-time user gets the scientific
#: pipeline and nothing else until they opt in.
DEFAULT_SETTINGS: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "enabled": False,
    "backend": "none",
    "active_model": "",
    "server_host": "127.0.0.1",
    "server_port": 8091,
    "models_dir": "",
    "allow_downloads": True,
    "api": {
        "provider": "",
        "base_url": "",
        "model": "",
    },
    "cluster": {
        "enabled": False,
        "scheduler": "slurm",
        "partition": "",
        "account": "",
        "time_limit": "04:00:00",
    },
}

_TRUTHY = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSEY = {"0", "false", "no", "off", "disable", "disabled"}


def settings_path() -> Path:
    """Return the location of the assistant settings file."""
    return config_dir() / SETTINGS_FILENAME


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Overlay stored values on the defaults, one level deep."""
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in overlay.items():
        if key not in merged:
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_settings() -> dict[str, Any]:
    """Return the effective settings, falling back to the defaults."""
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return _merge(DEFAULT_SETTINGS, {})
    if not isinstance(raw, dict):
        return _merge(DEFAULT_SETTINGS, {})
    return _merge(DEFAULT_SETTINGS, raw)


def save_settings(settings: dict[str, Any]) -> Path:
    """Write settings atomically and return the file path."""
    target = settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _merge(DEFAULT_SETTINGS, settings)
    payload["schema_version"] = SCHEMA_VERSION
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def update_settings(**changes: Any) -> dict[str, Any]:
    """Apply *changes* to the stored settings and return the new values."""
    settings = _merge(load_settings(), changes)
    save_settings(settings)
    return settings


def _environment_override() -> bool | None:
    """Return a forced on/off decision from the environment, if any."""
    raw = os.environ.get("SPACEMAP_ASSISTANT", "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSEY:
        return False
    return None


def is_enabled() -> bool:
    """Return True when the assistant should be loaded and shown."""
    override = _environment_override()
    if override is not None:
        return override
    return bool(load_settings().get("enabled", False))


def set_enabled(enabled: bool) -> dict[str, Any]:
    """Switch the assistant on or off and persist the choice."""
    return update_settings(enabled=bool(enabled))


def is_forced_off() -> bool:
    """True when an administrator disabled the assistant via the environment."""
    return _environment_override() is False


def resolved_models_dir() -> Path:
    """Where offline model weights are stored for this installation."""
    configured = str(load_settings().get("models_dir") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return models_dir()


TOKEN_FILENAME = "huggingface_token"


def token_path() -> Path:
    """Where the Hugging Face access token is stored for this user."""
    return config_dir() / TOKEN_FILENAME


def read_token() -> str:
    """Return the saved Hugging Face token, or the value from the environment.

    Gated models need a token that has accepted the model terms.  The
    environment wins so that a container or CI run can supply one without
    writing it to disk.
    """
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    try:
        return token_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def write_token(token: str) -> Path:
    """Save a Hugging Face token with owner-only permissions."""
    target = token_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(token.strip() + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def clear_token() -> None:
    """Delete the saved token."""
    try:
        token_path().unlink(missing_ok=True)
    except OSError:
        pass


#: Modules that must import for a local model to run.  Kept as names rather
#: than imports so that checking availability costs nothing.
LOCAL_BACKEND_MODULES = ("torch", "transformers", "accelerate", "fastapi", "uvicorn")


def missing_local_dependencies() -> list[str]:
    """Return the names of AI dependencies that are not importable."""
    from importlib.util import find_spec

    missing = []
    for name in LOCAL_BACKEND_MODULES:
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def local_backend_ready() -> bool:
    """True when this interpreter can host a local model in-process."""
    return not missing_local_dependencies()


__all__ = [
    "BACKENDS",
    "DEFAULT_SETTINGS",
    "is_enabled",
    "is_forced_off",
    "load_settings",
    "local_backend_ready",
    "missing_local_dependencies",
    "resolved_models_dir",
    "save_settings",
    "set_enabled",
    "settings_path",
    "clear_token",
    "read_token",
    "token_path",
    "update_settings",
    "write_token",
]
