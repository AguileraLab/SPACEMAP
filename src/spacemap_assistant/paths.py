"""Filesystem boundaries for the SPACEMAP AI assistant.

These rules apply to assistant tools, not to the existing manual SPACEMAP UI.
The scientific UI keeps its current behavior. The assistant starts inside
SPACEMAP_AI; an administrator may add exact read-only or read/write data roots.
Application source remains protected.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spacemap import userpaths


#: The directory holding the installed SPACEMAP packages -- ``src/`` in a git
#: checkout, the site-packages directory in a normal installation.  The
#: assistant may *read* here (the code index explains the pipeline to the user)
#: but must never write.
PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent

#: The user-writable application root.  Earlier versions anchored everything to
#: the repository directory, which only works for an editable checkout owned by
#: the person running it.
APP_ROOT = userpaths.spacemap_home()

#: Retained under its original name because the assistant's prompts, tests and
#: saved actions refer to it.
AI_ROOT = APP_ROOT

ASSISTANT_DATA_DIR = userpaths.assistant_data_dir()
MODELS_DIR = userpaths.models_dir()
RUNTIME_DIR = APP_ROOT / "runtime"
ASSISTANT_PROJECTS_DIR = userpaths.projects_dir()
ADMIN_ACCESS_PATH = ASSISTANT_DATA_DIR / "config" / "admin_access.json"

ASSISTANT_WRITABLE_DIRS = (
    ASSISTANT_DATA_DIR,
    ASSISTANT_PROJECTS_DIR,
)

ASSISTANT_PROTECTED_DIRS = (
    SOURCE_ROOT,
    MODELS_DIR,
    RUNTIME_DIR,
    userpaths.config_dir(),
    userpaths.state_dir(),
)


class AssistantPathError(PermissionError):
    """Raised when an assistant tool requests a forbidden filesystem path."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def is_within(path: str | Path, parent: str | Path) -> bool:
    """Return True when *path* resolves to *parent* or one of its children."""
    candidate = _resolved(path)
    boundary = _resolved(parent)
    try:
        candidate.relative_to(boundary)
        return True
    except ValueError:
        return False


def _is_broad_system_root(candidate: Path) -> bool:
    """Reject whole filesystems, mount points and home parents as data roots.

    A shallow directory is almost never the folder someone means to hand to the
    assistant, and naming specific site mounts here would only cover one
    institution's layout.
    """
    if candidate == candidate.parent:
        return True
    parts = [part for part in candidate.parts if part not in ("/", "\\")]
    if len(parts) <= 1:
        return True
    return candidate in {Path.home(), Path.home().parent}


def _load_admin_access() -> list[dict[str, Any]]:
    try:
        payload = json.loads(ADMIN_ACCESS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    grants = payload.get("grants") if isinstance(payload, dict) else None
    if not isinstance(grants, list):
        return []
    clean: list[dict[str, Any]] = []
    for item in grants:
        if not isinstance(item, dict) or item.get("mode") not in {"read", "read_write"}:
            continue
        raw_path = str(item.get("path") or "").strip()
        if not raw_path:
            continue
        path = _resolved(raw_path)
        clean.append(
            {
                "id": str(item.get("id") or ""),
                "path": str(path),
                "mode": str(item["mode"]),
                "created_at": str(item.get("created_at") or ""),
            }
        )
    return clean


def list_admin_access_grants() -> list[dict[str, Any]]:
    """Return administrator-added roots. Default roots are implicit."""
    return _load_admin_access()


def _write_admin_access(grants: list[dict[str, Any]]) -> None:
    ADMIN_ACCESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = ADMIN_ACCESS_PATH.with_name(
        f".{ADMIN_ACCESS_PATH.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    payload = {"schema_version": 1, "grants": grants}
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, ADMIN_ACCESS_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def grant_admin_access(path: str | Path, mode: str) -> dict[str, Any]:
    """Add an exact UI-administered read or read/write root."""
    if not str(path).strip():
        raise AssistantPathError("Enter an exact folder path.")
    candidate = _resolved(path)
    if mode not in {"read", "read_write"}:
        raise AssistantPathError("Choose Read only or Read + write.")
    if not candidate.is_dir():
        raise AssistantPathError("The administrator access path must be an existing folder.")
    # The containment check runs first: when a folder is refused for both
    # reasons, "this contains the application" is the more useful message.
    if mode == "read_write":
        if is_within(APP_ROOT, candidate) or is_within(SOURCE_ROOT, candidate):
            raise AssistantPathError(
                "Do not grant write access to a folder containing the SPACEMAP "
                "application. Choose a specific data project instead."
            )
    if _is_broad_system_root(candidate):
        raise AssistantPathError("Choose a specific project or data folder, not a whole system root.")
    if mode == "read_write":
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            raise AssistantPathError(
                "This looks like an application source folder. Grant a specific data folder instead."
            )
    grants = _load_admin_access()
    canonical = str(candidate)
    for item in grants:
        if item["path"] == canonical and item["mode"] == mode:
            return item
    record = {
        "id": str(uuid.uuid4()),
        "path": canonical,
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    grants.append(record)
    _write_admin_access(grants)
    return record


def revoke_admin_access(grant_id: str) -> dict[str, Any]:
    grants = _load_admin_access()
    removed = next((item for item in grants if item["id"] == str(grant_id)), None)
    if removed is None:
        raise AssistantPathError("That administrator access grant no longer exists.")
    _write_admin_access([item for item in grants if item["id"] != str(grant_id)])
    return removed


def _configured_roots(*, write: bool) -> list[Path]:
    modes = {"read_write"} if write else {"read", "read_write"}
    return [Path(item["path"]).resolve() for item in _load_admin_access() if item["mode"] in modes]


def is_assistant_read_allowed(path: str | Path) -> bool:
    candidate = _resolved(path)
    if is_within(candidate, APP_ROOT) or is_within(candidate, SOURCE_ROOT):
        return True
    return any(is_within(candidate, root) for root in _configured_roots(write=False))


def is_assistant_write_allowed(path: str | Path) -> bool:
    candidate = _resolved(path)
    if any(is_within(candidate, protected) for protected in ASSISTANT_PROTECTED_DIRS):
        return False
    if any(part in {".git", ".codex", ".agents"} for part in candidate.parts):
        return False
    return any(is_within(candidate, root) for root in ASSISTANT_WRITABLE_DIRS) or any(
        is_within(candidate, root) for root in _configured_roots(write=True)
    )


def require_inside_ai_root(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve an assistant path and reject traversal outside ``SPACEMAP_AI``."""
    candidate = _resolved(path)
    if not (is_within(candidate, APP_ROOT) or is_within(candidate, SOURCE_ROOT)):
        raise AssistantPathError(
            f"Assistant access is limited to {APP_ROOT} and the installed SPACEMAP "
            f"source; requested path was {candidate}."
        )
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def require_assistant_read_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path using the default boundary plus administrator read grants."""
    candidate = _resolved(path)
    if not is_assistant_read_allowed(candidate):
        raise AssistantPathError(
            "Assistant read access is limited to SPACEMAP_AI and administrator-approved folders; "
            f"requested path was {candidate}."
        )
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def require_assistant_write_path(path: str | Path) -> Path:
    """Allow assistant-created data only in dedicated data/project directories."""
    candidate = _resolved(path)
    if not is_assistant_write_allowed(candidate):
        raise AssistantPathError(
            "Assistant writes are limited to assistant_data/, projects/, and exact folders granted "
            "Read + write by an administrator. Application source remains protected. "
            f"Requested path was {candidate}."
        )
    return candidate


def action_workspace_root(path: str | Path) -> Path:
    """Return the named project or exact admin root containing an action output."""
    candidate = _resolved(path)
    if is_within(candidate, ASSISTANT_PROJECTS_DIR):
        relative = candidate.relative_to(ASSISTANT_PROJECTS_DIR.resolve())
        if not relative.parts:
            raise AssistantPathError("Choose a named project folder.")
        return (ASSISTANT_PROJECTS_DIR / relative.parts[0]).resolve()
    roots = [root for root in _configured_roots(write=True) if is_within(candidate, root)]
    if not roots:
        raise AssistantPathError("The action output is not inside an administrator-approved write root.")
    return max(roots, key=lambda item: len(item.parts))


def ensure_assistant_layout() -> None:
    """Create the local runtime layout without touching the scientific source."""
    for directory in (
        ASSISTANT_DATA_DIR / "config",
        ASSISTANT_DATA_DIR / "database",
        ASSISTANT_DATA_DIR / "exports",
        ASSISTANT_DATA_DIR / "actions",
        ASSISTANT_DATA_DIR / "logs",
        ASSISTANT_DATA_DIR / "memory",
        MODELS_DIR,
        RUNTIME_DIR,
        ASSISTANT_PROJECTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
