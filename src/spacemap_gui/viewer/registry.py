"""Which files the viewer may serve.

The browser never sends a file path. The interface registers each image it
shows and hands the viewer an opaque token; the tile server only answers for
registered tokens. A request cannot reach any other file on the machine.

Tokens are signed digests of what they stand for (with a secret kept in the
user's SPACEMAP state folder), and the table of tokens is saved there too.
So the same image always gets the same token, and links held by an open
viewer window, or by remembered layers, still work after SPACEMAP restarts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_BY_TOKEN: dict[str, str] = {}
_BY_PATH: dict[str, str] = {}
_OBJECTS: dict[str, dict] = {}
_OBJECT_KEYS: dict[str, str] = {}
_SESSIONS: dict[str, dict] = {}
_STATE = {"loaded": False, "secret": None}
_MAX_ENTRIES = 5000  # tokens kept on disk; the oldest go first


def _state_folder() -> Path:
    from spacemap.userpaths import state_dir

    return Path(state_dir())


def _store_path() -> Path:
    return _state_folder() / "viewer_registry.json"


def _secret() -> bytes:
    """The key tokens are signed with: made once per user, readable only by them."""
    if _STATE["secret"] is None:
        path = _state_folder() / "viewer_secret"
        try:
            secret = path.read_bytes()
        except OSError:
            secret = b""
        if len(secret) < 32:
            secret = os.urandom(32)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(secret)
        _STATE["secret"] = secret
    return _STATE["secret"]


def _token(kind: str, material: str) -> str:
    digest = hmac.new(_secret(), f"{kind}\0{material}".encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest[:12]).decode("ascii")


def _load() -> None:
    """Read the saved tokens once (called with _LOCK held)."""
    if _STATE["loaded"]:
        return
    _STATE["loaded"] = True
    try:
        data = json.loads(_store_path().read_text())
    except (OSError, ValueError):
        return
    for token, path in (data.get("paths") or {}).items():
        _BY_TOKEN[token] = path
        _BY_PATH[path] = token
    for token, payload in (data.get("objects") or {}).items():
        if isinstance(payload, dict) and payload.get("kind"):
            _OBJECTS[token] = payload
            _OBJECT_KEYS[json.dumps(payload, sort_keys=True, default=str)] = token
    for session, entry in (data.get("sessions") or {}).items():
        if isinstance(entry, dict):
            _SESSIONS[session] = {
                "version": int(entry.get("version", 0)),
                "spec": entry.get("spec"),
                "project": entry.get("project"),
                "seen": 0.0,
                "published": float(entry.get("published", 0.0)),
            }


def _save() -> None:
    """Write the tokens and window views to disk (called with _LOCK held)."""
    for table in (_BY_TOKEN, _OBJECTS):
        while len(table) > _MAX_ENTRIES:
            table.pop(next(iter(table)))
    sessions = dict(sorted(_SESSIONS.items(), key=lambda item: item[1].get("published", 0.0))[-50:])
    data = {
        "paths": dict(_BY_TOKEN),
        "objects": dict(_OBJECTS),
        "sessions": {
            sid: {
                "version": e["version"],
                "spec": e["spec"],
                "project": e.get("project"),
                "published": e.get("published", 0.0),
            }
            for sid, e in sessions.items()
        },
    }
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, default=str))
        os.replace(temporary, path)
    except OSError:
        pass  # the viewer still works for this run; links just will not survive a restart


def register(path: str | Path) -> str:
    """Allow the viewer to serve *path*; returns its token (the same across restarts)."""
    from spacemap.viewer.sources import is_viewable

    resolved = str(Path(path).expanduser().resolve())
    if not Path(resolved).is_file():
        raise FileNotFoundError(resolved)
    if not is_viewable(resolved):
        raise ValueError(f"Not an image the viewer can open: {Path(resolved).name}")
    with _LOCK:
        _load()
        token = _BY_PATH.get(resolved)
        if token is None:
            token = _token("image", resolved)
            _BY_PATH[resolved] = token
            _BY_TOKEN[token] = resolved
            _save()
        return token


def resolve(token: str) -> str | None:
    with _LOCK:
        _load()
        return _BY_TOKEN.get(str(token))


def clear() -> None:
    """Forget every token and window view (in memory and on disk)."""
    with _LOCK:
        _BY_TOKEN.clear()
        _BY_PATH.clear()
        _OBJECTS.clear()
        _OBJECT_KEYS.clear()
        _SESSIONS.clear()
        _STATE["loaded"] = True
        try:
            _store_path().unlink()
        except OSError:
            pass


def register_object(kind: str, **payload) -> str:
    """Allow the viewer to use a layer that is not a single image file.

    ``cells`` layers carry the image, its label pyramid, the ROI name and the
    per-cell tables they may be coloured from; ``annotations`` layers carry one
    GeoJSON file. The same payload always gets the same token.
    """
    key = json.dumps({"kind": kind, **payload}, sort_keys=True, default=str)
    with _LOCK:
        _load()
        token = _OBJECT_KEYS.get(key)
        if token is None:
            token = _token(kind, key)
            _OBJECT_KEYS[key] = token
            _OBJECTS[token] = json.loads(key)
            _save()
        return token


def resolve_object(token: str, kind: str) -> dict | None:
    with _LOCK:
        _load()
        found = _OBJECTS.get(str(token))
    return found if found and found.get("kind") == kind else None


def connect_cell_table(token, table_id, path, keys):
    """Add a checked table binding without replacing existing bindings in other windows."""
    with _LOCK:
        _load()
        layer = _OBJECTS.get(str(token))
        if not layer or layer.get("kind") != "cells":
            raise ValueError("Reopen the cell layer.")
        if layer.get("tables", {}).get(table_id) == str(path) and layer.get("table_roles", {}).get(table_id) == keys:
            return
        layer.setdefault("tables", {})[table_id] = str(path)
        layer.setdefault("table_roles", {})[table_id] = dict(keys)
        _save()


# ---------------------------------------------------------------------------
# Viewer windows: a viewer opened in its own browser window follows what the
# interface shows. The interface publishes each view under a channel id; the
# window asks for it every few seconds. Views are saved with the tokens, so a
# window picks up where it was after a restart.
# ---------------------------------------------------------------------------


def publish(session_id: str, spec: dict, project: dict | None = None) -> int:
    """Make *spec* the view for *session_id*; returns its version number.

    *project* (root, folder settings, cell table, colours, tab) lets the window
    open the project's other images itself.
    """
    with _LOCK:
        _load()
        entry = _SESSIONS.setdefault(str(session_id), {"version": 0, "spec": None, "seen": 0.0})
        changed = entry["spec"] != spec or (project is not None and entry.get("project") != project)
        if entry["spec"] != spec:
            entry["version"] += 1
            entry["spec"] = spec
        if project is not None:
            entry["project"] = project
        entry["published"] = time.time()
        if changed:
            _save()
        return entry["version"]


def session_view(session_id: str, *, mark_seen: bool = True) -> dict | None:
    """The current view for a window, and a note that the window is open."""
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        if entry is None:
            return None
        if mark_seen:
            entry["seen"] = time.time()
        return {"version": entry["version"], "spec": entry["spec"]}


def window_open(session_id: str, within: float = 8.0) -> bool:
    """True while a viewer window for *session_id* has asked for its view recently."""
    with _LOCK:
        entry = _SESSIONS.get(str(session_id))
        return bool(entry) and time.time() - entry.get("seen", 0.0) < within


def session_project(session_id: str) -> dict | None:
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        return dict(entry["project"]) if entry and entry.get("project") else None


def choose_project(session_id: str, root: str) -> None:
    """Studio opened the project in *root*: classic SPACEMAP loads it the next time its page runs (take_choice)."""
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        if entry is not None:
            entry["chosen_project"] = str(root)
            _save()


def take_choice(session_id: str) -> str | None:
    """The project Studio opened since classic SPACEMAP last looked, once."""
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        root = entry.pop("chosen_project", None) if entry else None
        if root:
            _save()
        return root


def annotation_projects(image: str) -> list[dict]:
    """Writable project contexts associated with a registered annotation image."""
    path = Path(image).resolve()
    with _LOCK:
        _load()
        found = {}
        for entry in _SESSIONS.values():
            project = entry.get('project') or {}
            if not project.get('root') or project.get('read_only'):
                continue
            layers = (entry.get('spec') or {}).get('layers', [])
            if path.is_relative_to(Path(project['root']).resolve()) or any(
                layer.get('type') == 'image' and _BY_TOKEN.get(layer.get('token')) == str(path) for layer in layers
            ):
                found[project['root']] = dict(project)
        return list(found.values())


def session_use_table(session_id: str, path: str) -> dict | None:
    """Make *path* the session's combined table (a new batch correction's output, say)."""
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        if not entry or not entry.get("project"):
            return None
        entry["project"] = {**entry["project"], "unified": str(path)}
        entry["chosen_table"] = str(path)
        _save()
        return dict(entry["project"])


def take_table_choice(session_id: str) -> dict | None:
    """A combined input selected in Studio, consumed once by its classic session."""
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        table = entry.pop("chosen_table", None) if entry else None
        if table:
            _save()
            return {"root": entry.get("project", {}).get("root"), "table": table}
    return None


def use_colours(session_id: str, colours: dict, root: str) -> None:
    """New phenotype colours for the session and for the cell layers of its project (keys as the viewer keys them)."""
    with _LOCK:
        _load()
        entry = _SESSIONS.get(str(session_id))
        if entry and entry.get("project"):
            entry["project"] = {**entry["project"], "colors": {**(entry["project"].get("colors") or {}), **colours}}
        inside = str(Path(root).resolve())
        for payload in _OBJECTS.values():
            if payload.get("kind") == "cells" and str(Path(str(payload.get("image", ""))).resolve()).startswith(inside):
                payload["colors"] = {**(payload.get("colors") or {}), **colours}
        _save()


def session_remember_images(session_id: str, entries: list[dict]) -> None:
    with _LOCK:
        if str(session_id) in _SESSIONS:
            _SESSIONS[str(session_id)]["images"] = entries


def session_images(session_id: str) -> list[dict]:
    with _LOCK:
        entry = _SESSIONS.get(str(session_id))
        return list(entry.get("images") or []) if entry else []
