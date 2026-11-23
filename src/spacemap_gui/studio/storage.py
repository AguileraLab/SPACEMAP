"""Explicit, verified cleanup of regenerable job files and discarded import history.

Never classify an entire job folder as temporary: models, masks, pending columns,
figures and uncommitted results may be its only outputs.
"""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from .annotation_edits import _atomic
from .settings import project_lock

NAMES = {'leiden': 'Saved Leiden working tables', 'training': 'Saved training requests',
         'history': 'Imported annotation history'}


def caches(root):
    """Read-only, on-demand inventory. None of these files enter a cleanup plan.

    Inspect only known generated files. In particular, phenotyping/cache contains
    saved settings, Flow workspaces contain gates, and Parquet may be source data.
    Size scans do not load tables, model weights or image pixels.
    """
    from spacemap.userpaths import cache_dir
    from . import flow_cache

    root = Path(root).resolve()
    shared = cache_dir().expanduser().resolve()
    groups = []

    def files(folder, pattern='*', recursive=False):
        folder = Path(folder)
        if not folder.is_dir() or folder.is_symlink():
            return []
        if not recursive:
            return [p for p in folder.glob(pattern) if p.is_file() and not p.is_symlink()]
        result = []
        for base, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
            result.extend(p for n in names if (p := Path(base) / n).match(pattern) and not p.is_symlink())
        return result

    def add(key, name, folder, paths, purpose, impact, scope='Project'):
        items = []
        for path in sorted(set(paths)):
            try:
                items.append({'name': str(path.relative_to(folder)), 'path': str(path), 'bytes': path.stat().st_size})
            except OSError:
                continue  # A running cache may expire between listing and stat.
        if items or Path(folder).is_dir():
            groups.append(dict(id=key, name=name, path=str(folder), purpose=purpose, impact=impact,
                               scope=scope, items=items, count=len(items), bytes=sum(i['bytes'] for i in items)))

    flow = root / 'outputs/studio/flow_gating'
    workspaces = flow_cache._workspaces(root)
    add('flow_plots', 'Flow plots', flow,
        [p for f in workspaces for p in flow_cache._files(f / 'plot-cache', flow_cache._PLOT)],
        'Cached Flow plots and statistics for this project.',
        'Kept for faster repeat plots. Clearing rebuilds plots; saved gates, labels and figures remain.')
    add('flow_memberships', 'Flow memberships', flow,
        [p for f in workspaces for p in flow_cache._files(f, flow_cache._MEMBERSHIP)],
        'Cached cell membership in Flow gates.',
        'Kept for faster filtering and exact counts. Clearing recalculates membership from saved gates.')
    viewer = root / 'outputs/viewer_cache/cell_labels'
    add('cell_labels', 'Viewer cell outlines', viewer, files(viewer, '*.cell_labels.tif'),
        'Image pyramids used to draw segmentation outlines quickly at each zoom level.',
        'Kept for fast image opening and navigation. Rebuilding can take time; source outlines are separate.')
    add('roi_tables', 'Viewer image tables', viewer / 'roi_tables',
        [p for p in files(viewer / 'roi_tables', '*.parquet') if not p.name.endswith('.labels.parquet')],
        'Per-image table copies with the image identifier used by linked cell overlays.',
        'Kept for fast cell overlays. Rebuilding reads the original measurement table.')
    leiden = root / 'outputs/phenotyping/leiden_cache'
    add('leiden_reload', 'Leiden reload files', leiden,
        files(leiden, 'res_*.parquet') + files(leiden, 'meta_*.json'),
        'Normalized markers, clusters and metadata for reopening the last Leiden result in Classic SPACEMAP.',
        'Kept. Removing these can require rerunning Leiden to recover normalized per-cell values.')

    # Only metadata-marked CSV twins are caches; never infer from .parquet alone.
    def generated_twin(path):
        try:
            import pyarrow.parquet as pq
            return b'spacemap_source' in (pq.read_schema(path).metadata or {})
        except (OSError, ValueError):
            return False

    twins = []
    skip = {'.git', 'jobs', 'studio_history', 'deleted', 'imported_annotations', 'annotation_workflow',
            'viewer_cache', 'cellpose_training', 'firefox', '__pycache__'}
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.')
                   and not d.endswith('.pending') and not (Path(base) / d).is_symlink()]
        for name in names:
            path = Path(base) / name
            if not name.endswith('.csv.parquet') or path.is_symlink() or not path.with_suffix('').is_file():
                continue
            if generated_twin(path):
                twins.append(path)
    add('table_copies', 'Fast table copies', root, twins,
        'Verified generated Parquet copies of CSV files within this project.',
        'Kept for faster table reads. Original tables, saved labels and pending columns are separate.')
    for key, name, folder, pattern, recursive, purpose, impact in [
        ('shared_tables', 'Shared fast table copies', shared / 'fast_copies', '*.parquet', False,
         'Fast table copies for read-only source directories, shared across projects.',
         'Kept. Removing them slows subsequent table reads and rebuilding.'),
        ('cell_bounds', 'Cell boundary indexes', shared / 'viewer/cell_bounds', '*', False,
         'Disk indexes for locating cell boundaries, shared across projects.',
         'Kept for quick cell selection and profiles. Rebuilding scans source polygons.'),
        ('summaries', 'Studio summaries', shared / 'studio-summaries-v1', '*.json', False,
         'Small saved table and project summaries, shared across projects.',
         'Kept for fast startup and panels. Rebuilding may scan full tables.'),
        ('numba', 'Compiled calculations', Path(os.environ.get('NUMBA_CACHE_DIR') or shared / 'numba_cache').expanduser().resolve(), '*', True,
         'Compiled numerical routines used by Studio and analysis jobs.',
         'Kept. Removing these makes the first subsequent calculation recompile code.'),
        ('cellpose', 'Cellpose downloads', Path(os.environ.get('CELLPOSE_LOCAL_MODELS_PATH') or shared / 'cellpose_cache').expanduser().resolve(), '*', True,
         'Downloaded Cellpose model resources shared across projects.',
         'Kept. Removing these may require downloading models again and affect offline segmentation.'),
        ('huggingface', 'Assistant downloads', shared / 'huggingface', '*', True,
         'Downloaded assistant model resources shared across projects.',
         'Kept. Removing these may require downloading models again and affect offline use.'),
    ]:
        found = files(folder, pattern, recursive)
        if key == 'shared_tables':
            found = [p for p in found if not p.name.endswith('.labels.parquet') and generated_twin(p)]
        add(key, name, folder, found, purpose, impact, 'Shared')
    return {'groups': groups, 'bytes': sum(g['bytes'] for g in groups), 'preserved': True}


def _json(path):
    return json.loads(Path(path).read_text())


def _stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def _hash(path):
    result = sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _saved_table(result, job, known=None):
    from spacemap import dataio
    table = Path(result.get('table') or '')
    if not table.is_file() or table.resolve().is_relative_to(job.resolve()):
        return False
    columns = result.get('columns') or [result.get('column')]
    if not columns or not all(isinstance(c, str) and c for c in columns):
        return False
    key = str(table.resolve())
    if known is not None and key in known:
        available = known[key]
    else:
        available = set(dataio.columns(table))
        if known is not None:
            known[key] = available
    return set(columns).issubset(available)


def _saved_model(root, request, status):
    from . import artifact_learning
    version = (status.get('result') or {}).get('version') or {}
    part = request.get('settings', {}).get('part')
    if part not in artifact_learning.NAMES or not version.get('id'):
        return False
    folder = artifact_learning.folder(root, part)
    identifier = version['id']
    metadata = folder / 'versions.json'
    return (Path(identifier).name == identifier and (folder / (identifier + '.joblib')).is_file()
            and metadata.is_file() and any(v.get('id') == identifier for v in _json(metadata)))


def _compact(request):
    """Keep settings and example identities, discard only embedded geometry copies."""
    result = json.loads(json.dumps(request))
    removed = Counter()
    def visit(value):
        if isinstance(value, dict):
            for key in list(value):
                item = value[key]
                if key in ('annotations', 'positives', 'negatives', 'tissue') and isinstance(item, (dict, list)):
                    removed[key] += len(item)
                    value[key + '_saved_count'] = len(item)
                    del value[key]
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(result.get('settings', {}))
    if removed:
        result['storage_cleanup'] = {'discarded_training_geometry': dict(removed), 'model_preserved': True}
    return result, bool(removed)


def preview(root):
    root = Path(root).resolve()
    jobs = root / 'outputs/studio/jobs'
    items, retained, active, columns = [], Counter(), False, {}
    for job in sorted(jobs.iterdir()) if jobs.is_dir() else []:
        if not job.is_dir() or job.is_symlink() or not (job / 'status.json').is_file():
            continue
        try:
            status = _json(job / 'status.json')
            if status.get('status') in ('running', 'starting', 'stopping'):
                active = True
                retained['Active jobs'] += 1
                continue
            if status.get('status') != 'done':
                continue
            request = _json(job / 'request.json')
            kind = request.get('kind')
            if kind == 'leiden' and (job / 'results.parquet').is_file():
                path = job / 'results.parquet'
                if _saved_table(status.get('result') or {}, job, columns):
                    items.append(dict(category='leiden', path=str(path), stamp=_stamp(path), bytes=path.stat().st_size,
                                      saved=status['result']['table']))
                else:
                    retained['Leiden results without verified saved columns'] += 1
            if kind == 'train_artifact_model' and (job / 'request.json').stat().st_size > 1024 * 1024:
                if _saved_model(root, request, status):
                    compact, changed = _compact(request)
                    path = job / 'request.json'
                    saved = path.stat().st_size - len(json.dumps(compact, indent=2).encode())
                    if changed and saved > 0:
                        items.append(dict(category='training', path=str(path), stamp=_stamp(path), bytes=saved))
                else:
                    retained['Training requests without verified saved models'] += 1
        except (ValueError, OSError, KeyError, TypeError):
            retained['Unreadable job records'] += 1
    imports = root / 'outputs/studio/imported_annotations'
    archived = {}
    deleted = root / 'outputs/studio/deleted'
    if deleted.is_dir():
        for path in deleted.rglob('*.geojson'):
            if path.is_file() and not path.is_symlink():
                archived.setdefault(path.name, []).append(path)
                original = re.sub(r'^\d{8}_\d{6}_', '', path.name)
                if original != path.name:
                    archived.setdefault(original, []).append(path)
    for history in sorted(imports.glob('*/studio_history/*')):
        if not history.is_dir() or history.is_symlink():
            continue
        if active:
            retained['Annotation history while jobs are active'] += 1
            continue
        try:
            events = sorted((history / 'events').glob('*.json'))
            if not events:
                continue
            digest = _json(events[-1]).get('after')
            current = history.parent.parent / history.name
            choices = ([current] if current.is_file() else []) + archived.get(history.name, [])
            saved = next((p for p in choices if not p.is_symlink() and _hash(p) == digest), None)
            if saved is None:
                retained['Annotation history without a verified saved final version'] += 1
                continue
            paths = [p for sub in ('events', 'states') for p in (history / sub).iterdir() if p.is_file() and not p.is_symlink()]
            items.append(dict(category='history', path=str(history), saved=str(saved), digest=digest,
                              files=[dict(path=str(p), stamp=_stamp(p)) for p in paths], bytes=sum(p.stat().st_size for p in paths)))
        except (ValueError, OSError, KeyError, TypeError):
            retained['Unreadable annotation history'] += 1
    for item in items:
        item['id'] = sha256((item['category'] + ':' + item['path']).encode()).hexdigest()[:24]
        path = Path(item['path'])
        item['name'] = path.name if item['category'] == 'history' else path.parent.name[:8] + ' · ' + path.name
        item['purpose'], item['impact'] = {
            'leiden': ('Per-cell transformed markers and clusters from a completed Leiden run.',
                       'Saved assignments and Mean/Median heatmaps are preserved. Per-cell transformed data would require rerunning Leiden. Saved table: ' + item.get('saved', '')),
            'training': ('Geometry snapshots embedded in an artifact model training request.',
                         'The trained model, saved examples and settings are preserved. Exact old input polygons are discarded.'),
            'history': ('Undo states and events for an imported annotation.',
                        'Undo history is discarded. The verified saved final version is preserved: ' + item.get('saved', '')),
        }[item['category']]
    groups = [dict(id=key, name=name, count=sum(i['category'] == key for i in items),
                   bytes=sum(i['bytes'] for i in items if i['category'] == key)) for key, name in NAMES.items()]
    return {'groups': groups, 'items': items, 'retained': dict(retained)}


def clean(root, plan, categories, ids=None):
    """Hold the same launch lock as Jobs while checking and cleaning finished work."""
    import fcntl
    jobs = Path(root) / 'outputs/studio/jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    with (jobs / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _clean(root, plan, categories, ids)


def _clean(root, plan, categories, ids=None):
    """Use a server-held preview, then recheck each dependency immediately before removal."""
    import fcntl
    from . import cluster_profiles
    if not categories or not set(categories).issubset(NAMES):
        raise ValueError('Select cleanup categories.')
    if ids is not None and (not isinstance(ids, list) or not ids or not set(ids).issubset(i['id'] for i in plan['items'])):
        raise ValueError('Select files from the current cleanup preview.')
    root = Path(root).resolve()
    freed, cleaned, skipped = 0, 0, []
    active = any(_json(p).get('status') in ('starting', 'running', 'stopping')
                 for p in (root / 'outputs/studio/jobs').glob('*/status.json'))
    for item in plan['items']:
        if item['category'] not in categories or (ids is not None and item['id'] not in ids):
            continue
        path = Path(item['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root / 'outputs/studio'):
            skipped.append(str(path)); continue
        try:
            with project_lock(root):
                if item['category'] in ('leiden', 'training'):
                    job = path.parent
                    status = _json(job / 'status.json')
                    if status.get('status') != 'done' or _stamp(path) != item['stamp']:
                        skipped.append(str(path)); continue
                    if item['category'] == 'leiden':
                        if not _saved_table(status.get('result') or {}, job):
                            skipped.append(str(path)); continue
                        profile = status.get('result', {}).get('profile') or {}
                        if profile.get('version') != cluster_profiles.VERSION or not {'mean', 'median'}.issubset(profile.get('summaries', {})):
                            status['result'] = cluster_profiles._upgrade(job, status['result'])
                            _atomic(job / 'status.json', json.dumps(status, indent=2).encode())
                        size = path.stat().st_size
                        path.unlink(); freed += size
                    else:
                        request = _json(path)
                        if not _saved_model(root, request, status):
                            skipped.append(str(path)); continue
                        compact, changed = _compact(request)
                        if not changed:
                            continue
                        size = path.stat().st_size
                        _atomic(path, json.dumps(compact, indent=2).encode())
                        freed += size - path.stat().st_size
                else:
                    if active:
                        skipped.append(str(path)); continue
                    # Preserve the lock inode; never replace a lock another editor may hold.
                    with (path / 'edit.lock').open('a') as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        events = sorted((path / 'events').glob('*.json'))
                        if not events or _json(events[-1]).get('after') != item['digest'] or _hash(item['saved']) != item['digest']:
                            skipped.append(str(path)); continue
                        if any(not Path(f['path']).is_file() or _stamp(f['path']) != f['stamp'] for f in item['files']):
                            skipped.append(str(path)); continue
                        for file in item['files']:
                            target = Path(file['path'])
                            size = target.stat().st_size
                            target.unlink(); freed += size
                cleaned += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            skipped.append(f'{path}: {error}')
    return {'bytes': freed, 'cleaned': cleaned, 'skipped': skipped}
