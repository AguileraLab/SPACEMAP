"""Small, regenerable summaries that survive Studio restarts.

Cache only JSON, never cell data or model artifacts. Source and journal stamps
are checked on every lookup. Per-key locks coalesce concurrent startup reads.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading

_MEMORY = OrderedDict()
_LOCK = threading.RLock()
_BUILD_LOCKS = [threading.RLock() for _ in range(64)]


def file_stamp(path):
    try:
        stat = Path(path).stat()
        return [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    except FileNotFoundError:
        return [str(path), None]


def table_stamp(table):
    from spacemap import labels, pending_columns, table_collection
    if table_collection.is_collection(table):
        return [file_stamp(table), *[table_stamp(r['path']) for r in table_collection.entries(table)]]
    files = [Path(table), labels.sidecar_path(table)]
    files += sorted(pending_columns.store_dir(table).glob('*.parquet'))
    return [file_stamp(path) for path in files]


def cached(namespace, identity, stamp, build):
    """Return a fresh summary; unwritable/corrupt caches fall back to computing."""
    from spacemap.userpaths import cache_dir
    key = hashlib.sha256(json.dumps([namespace, identity], sort_keys=True).encode()).hexdigest()
    try:
        folder = cache_dir() / 'studio-summaries-v1'
        target = folder / (key + '.json')
    except OSError:
        target = None
    memory_key = (str(target), key)
    with _BUILD_LOCKS[int(key[:4], 16) % len(_BUILD_LOCKS)]:
        version = stamp()
        with _LOCK:
            item = _MEMORY.get(memory_key)
        if item is None and target is not None:
            try:
                item = json.loads(target.read_text())
            except (OSError, ValueError):
                pass
        if not isinstance(item, dict) or item.get('stamp') != version or not isinstance(item.get('value'), dict):
            item = {'stamp': version, 'value': build()}
            # Never retain a summary assembled while its inputs changed.
            if version != stamp():
                return item['value']
            if target is not None:
                temporary = None
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode='w', dir=folder, suffix='.tmp', delete=False) as handle:
                        temporary = Path(handle.name)
                        json.dump(item, handle, allow_nan=False)
                    temporary.replace(target)
                except (OSError, ValueError):
                    pass
                finally:
                    if temporary is not None:
                        try:
                            temporary.unlink(missing_ok=True)
                        except OSError:
                            pass
        with _LOCK:
            _MEMORY[memory_key] = item
            _MEMORY.move_to_end(memory_key)
            while len(_MEMORY) > 128:
                _MEMORY.popitem(last=False)
        return deepcopy(item['value'])
