"""Bounded, disposable Flow plot results and membership files.

Only these generated files may be cleared. Workspace/history, exports, source
tables, and label journals are never cache-cleanup targets.
"""
from collections import OrderedDict
import gzip
import json
from pathlib import Path
import re
import threading
import uuid

_LOCK = threading.RLock()
_MEMORY = OrderedDict()
MAX_PLOTS = 16
MAX_BYTES = 64 * 1024 * 1024
MEMBERSHIPS = 2
_PLOT = re.compile(r'[0-9a-f]{64}\.json\.gz')
_MEMBERSHIP = re.compile(r'membership-[0-9a-f]{24}\.parquet')


def _files(folder, pattern):
    return [p for p in Path(folder).iterdir() if pattern.fullmatch(p.name) and p.is_file() and not p.is_symlink()] if Path(folder).is_dir() and not Path(folder).is_symlink() else []


def get(folder, key):
    identity = (str(folder), key)
    with _LOCK:
        value = _MEMORY.get(identity)
        if value is not None:
            _MEMORY.move_to_end(identity)
            return value
        path = Path(folder) / 'plot-cache' / (key + '.json.gz')
        try:
            with gzip.open(path, 'rt') as stream:
                record = json.load(stream)
            if not isinstance(record, dict) or record.get('key') != key or record.get('version') != 1:
                return None
            value = record['value']
            if not isinstance(value, dict) or not ('statistics' in value or {'plot', 'histogram', 'points', 'stats', 'gates'} <= value.keys()):
                return None
            path.touch()  # recently used results survive the size/count limit
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            return None
        _remember(identity, value)
        return value


def _remember(identity, value):
    _MEMORY[identity] = value
    _MEMORY.move_to_end(identity)
    while len(_MEMORY) > MAX_PLOTS:
        _MEMORY.popitem(last=False)


def put(folder, key, value):
    with _LOCK:
        _remember((str(folder), key), value)
        target = Path(folder) / 'plot-cache' / (key + '.json.gz')
        temporary = target.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(temporary, 'wt', compresslevel=1) as stream:
                json.dump({'version': 1, 'key': key, 'value': value}, stream, allow_nan=False, separators=(',', ':'))
            temporary.replace(target)
            size = 0
            for index, path in enumerate(sorted(_files(target.parent, _PLOT), key=lambda p: p.stat().st_mtime_ns, reverse=True)):
                size += path.stat().st_size
                if index >= MAX_PLOTS or size > MAX_BYTES:
                    path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass  # a cache write must not prevent analysis on read-only storage
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def prune_memberships(folder, current):
    """Keep the published membership plus one prior result; Undo can rebuild."""
    with _LOCK:
        try:
            older = sorted((p for p in _files(folder, _MEMBERSHIP) if p != Path(current)),
                           key=lambda p: p.stat().st_mtime_ns, reverse=True)
            for path in older[MEMBERSHIPS - 1:]:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def _workspaces(root):
    base = Path(root) / 'outputs/studio/flow_gating'
    return [p for p in base.iterdir() if re.fullmatch(r'[0-9a-f]{20}', p.name) and p.is_dir() and not p.is_symlink()] if base.is_dir() else []


def state(root):
    files = [p for folder in _workspaces(root) for p in [*_files(folder, _MEMBERSHIP), *_files(folder / 'plot-cache', _PLOT)]]
    size = 0
    for path in files:
        try:
            size += path.stat().st_size
        except FileNotFoundError:
            pass
    return {'files': len(files), 'bytes': size, 'plot_limit': MAX_PLOTS, 'plot_bytes_limit': MAX_BYTES, 'membership_limit': MEMBERSHIPS}


def clear(root):
    """Clear only generated Flow files in this project, preserving active views."""
    from .settings import project_lock
    from .flow_gating import atomic_json
    with project_lock(root), _LOCK:
        folders = _workspaces(root)
        for folder in folders:
            for path in [*_files(folder, _MEMBERSHIP), *_files(folder / 'plot-cache', _PLOT)]:
                path.unlink(missing_ok=True)
            pointer = folder / 'workspace.json'
            if pointer.is_file():
                workspace = json.loads(pointer.read_text())
                workspace['membership_file'] = None
                workspace.pop('analysis_stats',None)
                atomic_json(pointer, workspace)
        names = {str(p) for p in folders}
        for key in list(_MEMORY):
            if key[0] in names:
                del _MEMORY[key]
    return state(root)
