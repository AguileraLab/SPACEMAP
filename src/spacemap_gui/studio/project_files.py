"""Project-local removal of linked files; originals stay intact at any location."""
import json
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

_CONTEXT = ContextVar('spacemap_file_project', default=None)

def use(root):
    return _CONTEXT.set(str(root))


def index(root):
    return Path(root) / 'outputs/studio/removed_files.json'


@lru_cache(maxsize=64)
def _read(path, stamp):
    try:
        return set(json.loads(Path(path).read_text()))
    except (OSError, ValueError):
        return set()


def removed(root):
    path = index(root)
    return _read(str(path), path.stat().st_mtime_ns) if path.is_file() else set()


def excluded(path, root=None):
    if not path:
        return False
    path = Path(path).expanduser().resolve()
    if root is not None:
        return str(path) in removed(root)
    # Server requests propagate their project context into background reads.
    active = _CONTEXT.get()
    if active:
        return str(path) in removed(active)
    return any(str(path) in removed(parent) for parent in path.parents if index(parent).is_file())



def set_removed(root, paths, value=True):
    from .annotation_edits import _atomic
    from .settings import project_lock
    with project_lock(root):
        saved = set(removed(root))
        for path in paths:
            key = str(Path(path).expanduser().resolve())
            saved.add(key) if value else saved.discard(key)
        _atomic(index(root), json.dumps(sorted(saved)).encode())
        _read.cache_clear()


def visible(paths, root=None):
    return [path for path in paths if not excluded(path, root)]
