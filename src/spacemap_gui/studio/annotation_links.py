"""GeoJSON imports registered by reference; original files are never rewritten."""
from functools import lru_cache
import json
from pathlib import Path
import uuid

from . import project_files
from .annotation_edits import _atomic
from .settings import project_lock


def index(root):
    return Path(root) / 'outputs/studio/annotation_links.json'


@lru_cache(maxsize=32)
def _read(path, stamp):
    return json.loads(Path(path).read_text())


def records(root, image=None):
    path = index(root)
    stat = path.stat() if path.is_file() else None
    rows = _read(str(path), (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_size)) if stat else []
    return [dict(row) for row in rows if image is None or row['image'] == str(Path(image).resolve())]


def features(path):
    raw = json.loads(Path(path).read_text())
    items = raw if isinstance(raw, list) else raw.get('features', [raw]) if isinstance(raw, dict) else []
    if not isinstance(items, list) or not any(isinstance(f, dict) and
            (f.get('geometry') or {}).get('type') in ('Polygon', 'MultiPolygon') for f in items):
        raise ValueError('This file contains no polygon annotations.')
    return items


def validate_image(path, image):
    items = features(path)
    if image:
        from shapely.geometry import box, shape
        from spacemap.viewer.sources import image_info
        info = image_info(str(image))
        bounds = box(0, 0, info.width, info.height)
        if not any(shape(f['geometry']).intersects(bounds) for f in items if f.get('geometry')):
            raise ValueError('These polygons are outside the matching image.')


def register(root, path, image=None):
    """Each import has an identity, including repeated references to the same file."""
    path = Path(path).resolve()
    validate_image(path, image)
    row = dict(id=uuid.uuid4().hex, path=str(path), image=str(Path(image).resolve()) if image else '')
    with project_lock(root):
        _atomic(index(root), json.dumps([*records(root), row], indent=2).encode())
        _read.cache_clear()
        project_files.set_removed(root, [path], False)
        if image:
            from .annotation_names import hide_file
            hide_file(root, image, path, hidden=False)
    return row


def remove(root, ids):
    with project_lock(root):
        before = records(root)
        kept = [row for row in before if row['id'] not in ids]
        removed = [row for row in before if row['id'] in ids]
        _atomic(index(root), json.dumps(kept, indent=2).encode())
        _read.cache_clear()
        detached = []
        excluded = set()
        for row in removed:
            if not any(r['path'] == row['path'] and r['image'] == row['image'] for r in kept):
                if row['image']:
                    from .annotation_names import hide_file
                    hide_file(root, row['image'], row['path'])
                detached.append(row['path'])
            if not any(r['path'] == row['path'] for r in kept):
                excluded.add(row['path'])
        if excluded:
            project_files.set_removed(root, excluded)
        return detached
