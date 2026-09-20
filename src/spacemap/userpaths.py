"""User-writable locations for SPACEMAP application state.

Everything the application writes while it runs -- interface state, the AI
assistant's database, downloaded models, logs -- lives under a single root so
that an installed copy never writes into its own package directory.  A user who
installs SPACEMAP with ``pip`` normally has no write access there, and a user
who installs it twice would otherwise have the two copies fight over the same
state file.

Resolution order for the root:

1. ``$SPACEMAP_HOME`` when it is set.
2. ``$XDG_DATA_HOME/spacemap`` when ``XDG_DATA_HOME`` is set.
3. ``~/.spacemap``

A development checkout that prefers repository-local state can export
``SPACEMAP_HOME=<checkout>/.spacemap_home``.  Individual directories can also be
redirected on their own -- see the ``SPACEMAP_*_DIR`` variables below -- which is
what the container image does to keep models on a mounted volume.
"""

from __future__ import annotations

import os
from pathlib import Path


#: Environment variable overrides for individual directories.  Each maps to the
#: accessor of the same name below.
_DIR_ENV_VARS = {
    "config": "SPACEMAP_CONFIG_DIR",
    "state": "SPACEMAP_STATE_DIR",
    "cache": "SPACEMAP_CACHE_DIR",
    "logs": "SPACEMAP_LOGS_DIR",
    "models": "SPACEMAP_MODELS_DIR",
    "projects": "SPACEMAP_PROJECTS_DIR",
    "assistant_data": "SPACEMAP_ASSISTANT_DATA_DIR",
}


def _env_path(name: str) -> Path | None:
    """Return an expanded path from environment variable *name*, if set."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def spacemap_home() -> Path:
    """Return the root directory for all user-writable SPACEMAP state."""
    explicit = _env_path("SPACEMAP_HOME")
    if explicit is not None:
        return explicit
    xdg = _env_path("XDG_DATA_HOME")
    if xdg is not None:
        return xdg / "spacemap"
    return Path.home() / ".spacemap"


def _subdir(kind: str, *parts: str) -> Path:
    """Return the directory for *kind*, honouring its dedicated override."""
    override = _env_path(_DIR_ENV_VARS[kind])
    if override is not None:
        return override
    return spacemap_home().joinpath(*parts)


def config_dir() -> Path:
    """Small JSON/TOML settings the user edits or the UI writes."""
    return _subdir("config", "config")


def state_dir() -> Path:
    """Interface state that survives restarts but is not user-editable."""
    return _subdir("state", "state")


def cache_dir() -> Path:
    """Regenerable scratch data (Numba, Cellpose, Hugging Face)."""
    return _subdir("cache", "cache")


def logs_dir() -> Path:
    """Application and model-server logs."""
    return _subdir("logs", "logs")


def models_dir() -> Path:
    """Downloaded or user-placed offline model weights."""
    return _subdir("models", "models")


def projects_dir() -> Path:
    """Default parent for analysis projects created from the GUI."""
    return _subdir("projects", "projects")


def assistant_data_dir() -> Path:
    """The AI assistant's conversation database, actions and exports."""
    return _subdir("assistant_data", "assistant_data")


#: Directories created by :func:`ensure_layout`.  ``projects_dir`` is deliberate-
#: ly absent: the user chooses where projects live and an empty directory in the
#: home folder would only be confusing.
_MANAGED_DIRS = (config_dir, state_dir, cache_dir, logs_dir, models_dir)


def ensure_layout() -> Path:
    """Create the user-writable directory layout and return its root."""
    for accessor in _MANAGED_DIRS:
        accessor().mkdir(parents=True, exist_ok=True)
    return spacemap_home()


def describe_layout() -> dict[str, str]:
    """Return the resolved locations, for display in the UI and bug reports."""
    return {
        "home": str(spacemap_home()),
        "config": str(config_dir()),
        "state": str(state_dir()),
        "cache": str(cache_dir()),
        "logs": str(logs_dir()),
        "models": str(models_dir()),
        "projects": str(projects_dir()),
        "assistant_data": str(assistant_data_dir()),
    }


__all__ = [
    "spacemap_home",
    "config_dir",
    "state_dir",
    "cache_dir",
    "logs_dir",
    "models_dir",
    "projects_dir",
    "assistant_data_dir",
    "ensure_layout",
    "describe_layout",
]
