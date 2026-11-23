"""Bounded metadata pages and lossless component edits for linked annotations."""
from collections import Counter, OrderedDict
from concurrent.futures import Future
from hashlib import sha256
import json
from pathlib import Path
import threading

from . import annotation_edits as edits

_CACHE = OrderedDict()
_LOCK = threading.RLock()
_LOADING = {}
_OVERVIEWS = OrderedDict()


def _key(path):
    path = Path(path)
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size, stat.st_ino)


def overview(path):
    """Class names/colours and stored component counts, without geometry preparation."""
    from spacemap.viewer.annotations import _feature_class
    key = _key(path)
    with _LOCK:
        if key in _OVERVIEWS:
            _OVERVIEWS.move_to_end(key)
            return _OVERVIEWS[key]
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    features = data.get('features', []) if isinstance(data, dict) else data
    counts, colours = Counter(), {}
    def count(geometry):
        if not geometry:
            return 0
        if geometry.get('type') == 'Polygon':
            return int(bool(geometry.get('coordinates')))
        if geometry.get('type') == 'MultiPolygon':
            return sum(bool(rings) for rings in geometry.get('coordinates', []))
        return sum(count(g) for g in geometry.get('geometries', []))
    for feature in features:
        name, colour = _feature_class(feature)
        counts[name] += count(feature.get('geometry'))
        if colour is not None:
            colours.setdefault(name, '#%02x%02x%02x' % colour)
    palette = colours
    if any(name not in palette for name in counts):
        from spacemap.acc_utils import build_color_mapping
        palette = build_color_mapping(sorted(counts, key=str.casefold), existing=colours)
    result = {'revision': sha256(raw).hexdigest(),
              'classes': [{'name': name, 'count': n, 'color': palette[name]} for name, n in sorted(counts.items())]}
    with _LOCK:
        _OVERVIEWS[key] = result
        while len(_OVERVIEWS) > 16:
            _OVERVIEWS.popitem(last=False)
    return result


def fingerprint(feature):
    return sha256(json.dumps(feature, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def managed_path(root, image, source):
    from .annotation_inputs import import_folder
    source = Path(source).resolve()
    folder = import_folder(root, image)
    if source.parent == folder.resolve():
        return source
    key = sha256(str(source).encode()).hexdigest()[:16]
    return folder / (key + '_' + source.name)


def effective_path(root, image, source):
    copy = managed_path(root, image, source)
    return copy if copy.is_file() else Path(source)


def display_identity(path):
    """Keep a managed copy's original filename visible, without renaming files."""
    path = Path(path)
    try:
        provenance = json.loads(path.with_suffix('.source.json').read_text())
        source = Path(provenance['source'])
        prefix = sha256(str(source.resolve()).encode()).hexdigest()[:16]
        if path.name == prefix + '_' + source.name:
            return {'name': source.name, 'source_name': source.name, 'annotation_origin': 'copy'}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


def display_style(path):
    """A stable, non-path display family shared by an original and its edit copy."""
    identity = display_identity(path)
    source = Path(path)
    if identity:
        source = Path(json.loads(source.with_suffix('.source.json').read_text())['source'])
    family = sha256(str(source.resolve().parent).encode()).hexdigest()[:24]
    return {**identity, 'annotation_display_family': family}


def index(path):
    from shapely.strtree import STRtree
    path = Path(path)
    key = _key(path)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        pending = _LOADING.get(key)
        owner = pending is None
        if owner:
            pending = _LOADING[key] = Future()
    if not owner:
        return pending.result()
    try:
        features, version = edits.read(path)
        rows, geometries = [], []
        for feature in features:
            props = feature['properties']
            geometry = edits.prepared_geometry(feature)
            geometries.append(geometry)
            rows.append({'id': str(props['id']), 'name': (props.get('classification') or {}).get('name') or props.get('name') or 'Unlabelled',
                         'area': geometry.area, 'bounds': list(geometry.bounds), 'fingerprint': fingerprint(feature)})
        result = {'revision': version, 'rows': rows, 'features': features,
                  'geometries': geometries, 'tree': STRtree(geometries),
                  'by_id': {str(f['properties']['id']): f for f in features}}
        with _LOCK:
            _CACHE[key] = result
            while len(_CACHE) > 8:
                _CACHE.popitem(last=False)
        pending.set_result(result)
        return result
    except BaseException as exc:
        pending.set_exception(exc)
        raise
    finally:
        with _LOCK:
            _LOADING.pop(key, None)


def hit(path, x, y, hidden=()):
    """Exact component hit, including holes, with the same IDs as lists/edits."""
    import math
    from shapely.geometry import Point
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError('Choose a point on the image.')
    value = index(path)
    point, hidden = Point(x, y), set(hidden)
    hits = [int(i) for i in value['tree'].query(point)
            if value['rows'][int(i)]['name'] not in hidden
            and value['geometries'][int(i)].covers(point)]
    i = min(hits, key=lambda i: value['rows'][i]['area']) if hits else None
    return {'feature': value['features'][i] if i is not None else None,
            'revision': value['revision']}


def selection_area(points):
    import numpy as np
    from shapely.geometry import Polygon
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not 3 <= len(coords) <= 20000 or not np.isfinite(coords).all():
        raise ValueError('Draw a finite, closed selection area.')
    area = Polygon(coords)
    if area.is_empty or not area.is_valid or area.area <= 0:
        raise ValueError('Draw an area without crossing its own outline.')
    return area


def area_selection(path, points, hidden=()):
    """Only fully enclosed visible components; use the cached spatial index."""
    value = index(path)
    area, hidden = selection_area(points), set(hidden)
    ids = [value['rows'][int(i)]['id'] for i in value['tree'].query(area, predicate='covers')
           if value['rows'][int(i)]['name'] not in hidden]
    return {'ids': ids, 'revision': value['revision']}


def local_area_selection(features, points):
    from shapely.geometry import shape
    area = selection_area(points)
    return {'ids': [str(f['properties']['id']) for f in features if area.covers(shape(f['geometry']))]}


_SELECTIONS = OrderedDict()


def selection(path, ids):
    """In-memory tiled outline. Never expand thousands of geometries in the browser."""
    from spacemap.viewer.annotations import AnnotationSet
    from shapely.strtree import STRtree
    value = index(path)
    wanted = set(map(str, ids))
    rows = [i for i, row in enumerate(value['rows']) if row['id'] in wanted]
    if len(rows) != len(wanted):
        raise ValueError('Selected polygons changed. Reload the source.')
    key = sha256((str(Path(path).resolve()) + value['revision'] + '\0'.join(sorted(wanted))).encode()).hexdigest()
    with _LOCK:
        if key not in _SELECTIONS:
            geometries = [value['geometries'][i] for i in rows]
            classes = [value['rows'][i]['name'] for i in rows]
            names = sorted(set(classes))
            _SELECTIONS[key] = AnnotationSet(str(Path(path).resolve()), value['revision'], geometries,
                classes, names, {n: (255, 255, 255) for n in names}, STRtree(geometries))
        _SELECTIONS.move_to_end(key)
        while len(_SELECTIONS) > 64:
            _SELECTIONS.popitem(last=False)
    return {'selection': key, 'count': len(rows)}


def selection_tile(path, key, image, z, x, y, hidden=()):
    from spacemap.viewer.annotations import render_annotation_tile
    import cv2
    import numpy as np
    with _LOCK:
        source = _SELECTIONS.get(key)
    if source is None or source.path != str(Path(path).resolve()):
        raise ValueError('Reselect these polygons to refresh their outline.')
    # White border with a black halo remains visible on H&E and fluorescence.
    rgba = render_annotation_tile(source, image, z, x, y, hidden=set(hidden), fill_alpha=0, line_px=2)
    edge = rgba[:, :, 3].copy()
    halo = cv2.dilate(edge, np.ones((5, 5), dtype=np.uint8))
    rgba[:, :, :3] = 0
    rgba[edge > 0, :3] = 255
    rgba[:, :, 3] = halo
    return rgba


def page(path, *, name=None, search='', sort='size_down', offset=0, limit=100, locate=None):
    value = index(path)
    rows = [r for r in value['rows'] if (name is None or r['name'] == name)
            and str(search).lower() in (r['name'] + ' ' + r['id']).lower()]
    rows = sorted(rows, key=(lambda r: (r['bounds'][1], r['bounds'][0], r['id'])) if sort == 'place'
                  else lambda r: (r['area'], r['id']), reverse=sort == 'size_down')
    start, limit = max(0, int(offset)), max(1, min(200, int(limit)))
    if locate:
        rank = next((i for i,r in enumerate(rows) if r['id'] == locate), None)
        if rank is not None:
            start = (rank // limit) * limit
    return {'revision': value['revision'], 'total': len(rows), 'rows': rows[start:start + limit],
            'offset': start,
            'classes': [{'name': k, 'count': n} for k, n in sorted(Counter(r['name'] for r in value['rows']).items())]}


def selected(path, *, ids=None, name=None, search='', below=None):
    value = index(path)
    wanted = None if ids is None else set(map(str, ids))
    rows = [r for r in value['rows'] if (wanted is None or r['id'] in wanted)
            and (name is None or r['name'] == name)
            and str(search).lower() in (r['name'] + ' ' + r['id']).lower()
            and (below is None or r['area'] < float(below))]
    if wanted is not None and len(rows) != len(wanted):
        raise ValueError('Some polygons changed or are no longer in this selection.')
    return rows


def edit(root, image, source, *, ids, expected, action, name='', fingerprints=None):
    """Create one managed copy on first edit; never touch a linked source."""
    from copy import deepcopy
    from .annotation_inputs import import_folder
    target = managed_path(root, image, source)
    # One lock covers checking/copying/editing; replace owns the target's history lock.
    guard = import_folder(root, image) / '.component-actions'
    with edits.locked(guard):
        current = effective_path(root, image, source)
        before, revision = edits.read(current)
        wanted = set(map(str, ids))
        found = {str(f['properties']['id']): f for f in before if str(f['properties']['id']) in wanted}
        if len(found) != len(wanted) or not wanted:
            raise ValueError('Some selected polygons are missing. Reload the source.')
        if revision != expected:
            if not fingerprints or any(fingerprints.get(key) != fingerprint(f) for key, f in found.items()):
                raise ValueError('These polygons changed since they were selected. Reload before editing.')
        if action not in ('rename', 'delete'):
            raise ValueError('Choose Rename or Delete.')
        name = str(name).strip()
        if action == 'rename' and (not name or len(name) > 200):
            raise ValueError('Enter an annotation name (up to 200 characters).')
        palette = {(f['properties'].get('classification') or {}).get('name'): f['properties'].get('classification') for f in before}
        after = []
        for feature in before:
            if str(feature['properties']['id']) in wanted:
                if action == 'delete':
                    continue
                feature = deepcopy(feature)
                classification = dict(palette.get(name) or feature['properties'].get('classification') or {})
                classification['name'] = name
                feature['properties']['classification'] = classification
                colour = classification.get('color')
                if isinstance(colour, (list, tuple)) and len(colour) >= 3:
                    feature['properties']['colour'] = '#%02x%02x%02x' % tuple(colour[:3])
                if 'name' in feature['properties']:
                    feature['properties']['name'] = name
            after.append(feature)
        if target != current:
            # Preserve the original state as the copy's first undo entry.
            edits.replace(target, before, edits.revision(target), 'Imported for component editing')
            provenance = {'source': str(Path(source).resolve()), 'source_revision': revision,
                          'image': str(Path(image).resolve())}
            edits._atomic(target.with_suffix('.source.json'), json.dumps(provenance).encode())
        result = edits.replace(target, after, edits.revision(target) if target != current else revision,
                               ('Renamed' if action == 'rename' else 'Removed') + f' {len(wanted)} polygons', edited=True)
        return {'path': str(target), 'revision': edits.revision(target), 'count': len(wanted), 'result': result}
