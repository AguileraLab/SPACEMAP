"""Small, durable per-image viewer preferences; never polygon or pixel data."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
from spacemap.sharing import open_lock

FIELDS = {
    'visible', 'opacity', 'mode', 'color_by', 'table', 'colormap', 'outline', 'open', 'selection',
    'fcol', 'fvals', 'vcol', 'radius', 'weighted', 'cmap', 'cmapAuto', 'profileSource',
    'suggestionsOn', 'learnColumn', 'suggestionMode', 'suggestionModelId', 'markerTransforms',
    'hidden', 'channels', 'autoMethod', 'autoScope', 'activeSet',
}
EXTRA_FIELDS = FIELDS | {
    'id', 'type', 'name', 'token', 'image_token', 'tables', 'added', 'editable', 'regions',
    'annotation_kind', 'artifact_part', 'training_part', 'tissue_version', 'annotation_group',
    'classes', 'default_class', 'names', 'saved_file', 'display_token', 'remember',
}


def path_for(root, image) -> Path:
    root, image = Path(root).resolve(), Path(image).resolve()
    identity = str(image.relative_to(root)) if image.is_relative_to(root) else str(image)
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', image.name)[:100]
    return root / 'outputs' / 'viewer' / 'views' / f'{name}.{suffix}.json'


def load(path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    if not isinstance(data, dict):
        raise ValueError('The saved viewer settings are not a JSON object.')
    return data


def clean(record) -> dict:
    from .display import clean as clean_channels

    if not isinstance(record, dict):
        raise ValueError('Viewer settings must be a JSON object.')
    if len(json.dumps(record, allow_nan=False).encode()) > 256_000:
        raise ValueError('Viewer settings are too large; only display preferences can be saved.')

    def layer(item, fields):
        if not isinstance(item, dict):
            raise ValueError('Invalid layer settings.')
        out = {k: v for k, v in item.items() if k in fields}
        if 'channels' in out:
            out['channels'] = clean_channels(out['channels'])
        return out

    layers, extras = record.get('layers', {}), record.get('extras', [])
    if not isinstance(layers, dict) or not isinstance(extras, list) or len(layers) > 256 or len(extras) > 128:
        raise ValueError('Invalid viewer layers.')
    out = {key: record[key] for key in ('order', 'layerGroups', 'shownOnly', 'removed') if key in record}
    if isinstance(record.get('pixelSmoothing'), bool):
        out['pixelSmoothing'] = record['pixelSmoothing']
    split = record.get('splitMarkers')
    if isinstance(split, dict):
        kept = {k: split[k] for k in ('open', 'all', 'floating') if isinstance(split.get(k), bool)}
        for key, low, high in [('zoom', .25, 8), ('size', 96, 280), ('height', 140, 900),
                               ('width', 320, 1800), ('x', 0, 10000), ('y', 0, 10000)]:
            value = split.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                kept[key] = max(low, min(high, value))
        out['splitMarkers'] = kept
    out.update(schema=1, panelLayout=1, keyedLayers=True,
               layers={str(key): layer(item, FIELDS) for key, item in layers.items()},
               extras=[layer(item, EXTRA_FIELDS) for item in extras])
    center, zoom = record.get('center'), record.get('zoom')
    if (isinstance(center, list) and len(center) == 2 and
            all(isinstance(v, (int, float)) and math.isfinite(v) for v in center) and
            isinstance(zoom, (int, float)) and math.isfinite(zoom) and zoom > 0):
        out.update(center=center, zoom=zoom)
    return out


@contextmanager
def locked(path):
    """Serialize writers from both browser windows and separate Studio processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_lock(path.with_suffix(path.suffix + '.lock')) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def write(path, data, stamp=None) -> dict:
    """Caller holds locked(path); late unload requests cannot replace newer edits."""
    path = Path(path)
    previous = load(path) or {}
    writers = previous.get('_writers', {})
    if isinstance(stamp, dict) and stamp.get('writer') and isinstance(stamp.get('sequence'), int):
        writer, sequence = str(stamp['writer'])[:128], stamp['sequence']
        if sequence <= writers.get(writer, -1):
            return previous
        writers = {**writers, writer: sequence}
        data['_writers'] = dict(list(writers.items())[-16:])
    data['saved_at'] = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp')
    try:
        temporary.write_text(json.dumps(data, indent=1, allow_nan=False))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return data


def save(root, image, record, stamp=None) -> dict:
    data = clean(record)
    root, image = Path(root).resolve(), Path(image).resolve()
    data['image'] = str(image.relative_to(root)) if image.is_relative_to(root) else str(image)
    path = path_for(root, image)
    with locked(path):
        return write(path, data, stamp)
