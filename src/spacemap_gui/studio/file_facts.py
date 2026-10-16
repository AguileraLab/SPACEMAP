"""Small facts read from big files (an annotation file's names, a cell table's number of cells), kept on disk until
the file changes, so a page listing 40 images opens at once after SPACEMAP restarts instead of re-reading gigabytes.

Kept per user in SPACEMAP's state folder (``file_facts.json``), keyed by the file's path, size and time.
"""

import json
import os
from pathlib import Path
import threading

_LOCK = threading.Lock()
_FACTS = None
_DIRTY = False
LIMIT = 20000


def _path():
    from spacemap.userpaths import state_dir

    return Path(state_dir()) / "file_facts.json"


def _load():
    global _FACTS
    if _FACTS is None:
        try:
            _FACTS = json.loads(_path().read_text())
        except (OSError, ValueError):
            _FACTS = {}
    return _FACTS


def _key(path, kind):
    stat = os.stat(path)
    return f"{kind}|{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def get(path, kind):
    """The saved fact for this file as it is now, or None."""
    try:
        key = _key(path, kind)
    except OSError:
        return None
    with _LOCK:
        return _load().get(key)


def put(path, kind, value):
    global _DIRTY
    try:
        key = _key(path, kind)
    except OSError:
        return
    with _LOCK:
        facts = _load()
        if len(facts) > LIMIT:
            facts.clear()
        facts[key] = value
        _DIRTY = True


def save():
    """Write the facts gathered since the last save (call after a listing)."""
    global _DIRTY
    with _LOCK:
        if not _DIRTY:
            return
        target = _path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            partial.write_text(json.dumps(_FACTS))
            os.replace(partial, target)
            _DIRTY = False
        except OSError:
            pass  # the facts are only a shortcut: reading the files again still works
