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
    # GPU memory mode for the local model: auto, bfloat16, float32, int8 or int4 (model_runtime.PRECISIONS).
    "precision": "auto",
    # Stop the local model server when the SPACEMAP that started it stops (model_service/guard.py).
    "stop_with_spacemap": True,
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


def is_forced_on() -> bool:
    """True when the environment pins the assistant on.

    The in-app switch cannot override the environment, so the UI needs to say
    so rather than offering a control that does nothing.
    """
    return _environment_override() is True


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
#: What running a model on this machine needs, as declared in pyproject.toml
#: (the ``ai`` extra, plus torch from SPACEMAP's core dependencies):
#: (pip name, version, import name, required). bitsandbytes is optional: it
#: only lets large models fit a smaller NVIDIA GPU (8- or 4-bit).
AI_PACKAGES: tuple[tuple[str, str, str, bool], ...] = (
    ("torch", ">=2.0.0", "torch", True),
    ("transformers", ">=5.0", "transformers", True),
    ("accelerate", ">=0.30", "accelerate", True),
    ("safetensors", ">=0.4", "safetensors", True),
    ("huggingface-hub", ">=0.24", "huggingface_hub", True),
    ("fastapi", ">=0.110", "fastapi", True),
    ("uvicorn", ">=0.27", "uvicorn", True),
    ("pydantic", ">=2.0", "pydantic", True),
    ("bitsandbytes", ">=0.43", "bitsandbytes", False),
)
#: A small quantized (GGUF) model on a machine without a GPU (the ``gguf`` extra).
GGUF_PACKAGES: tuple[tuple[str, str, str, bool], ...] = (("llama-cpp-python", ">=0.3.0", "llama_cpp", True),)

LOCAL_BACKEND_MODULES = tuple(module for _, _, module, required in AI_PACKAGES if required)


def dependency_report(packages=AI_PACKAGES) -> list[dict]:
    """Each package: its pip name, the version wanted, whether it is needed, and the version installed (or None)."""
    import importlib.metadata
    from importlib.util import find_spec

    report = []
    for name, spec, module, required in packages:
        version = None
        try:
            if find_spec(module) is not None:
                try:
                    version = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    version = "installed"
        except (ImportError, ValueError):
            version = None
        report.append({"name": name, "wanted": spec, "required": required, "installed": version})
    return report


def install_command(packages: list[tuple[str, str]]) -> str:
    """The pip command that installs *packages* into the environment running SPACEMAP.

    It names this interpreter by its full path, so the packages go into the
    right environment whatever it is called, and it installs only what is
    listed - never SPACEMAP itself.
    """
    import sys

    wanted = " ".join(f'"{name}{spec}"' for name, spec in packages)
    return f'"{sys.executable}" -m pip install {wanted}'


def environment_name() -> str:
    """A readable name for the environment running SPACEMAP (the conda environment's name if any)."""
    import os
    import sys
    from pathlib import Path

    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name


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
    "AI_PACKAGES",
    "BACKENDS",
    "GGUF_PACKAGES",
    "dependency_report",
    "environment_name",
    "install_command",
    "DEFAULT_SETTINGS",
    "is_enabled",
    "is_forced_off",
    "is_forced_on",
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
