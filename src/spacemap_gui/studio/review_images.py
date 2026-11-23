"""Image metadata for Review, using the same table assignments as Phenotyping.

Only requested columns are read, once per table/revision. Image listing itself
does not scan cell values. Shared metadata is an explicit, bounded snapshot.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from spacemap import dataio
from spacemap.viewer.catalog import sample_of
from . import phenotyping_inputs, review_store


def plan(channel):
    from .spatial_sources import input_plan
    return input_plan(channel)


def options(rows):
    names = set()
    for path in {r.get('table_path') for r in rows if r.get('table_path')}:
        names.update(dataio.columns(path))
    return sorted(names, key=str.casefold)


def _summary(series):
    values = series.dropna()
    values = values[values.astype(str).str.strip().ne('')]
    unique = values.astype(str).unique()
    if not len(unique):
        return {'text': '—', 'detail': 'No value'}
    partial = len(values) < len(series)
    if len(unique) == 1:
        return {'text': str(unique[0]), 'detail': str(unique[0]) + (' · Some cells have no value' if partial else '')}
    return {'text': f'Mixed ({len(unique):,})',
            'detail': ', '.join(sorted(unique[:6])) + (' …' if len(unique) > 6 else '') + (' · Some values missing' if partial else '')}


@lru_cache(maxsize=64)
def _table_values(path, stamp, columns, image_column, single_image):
    available = set(dataio.columns(path))
    chosen = [c for c in columns if c in available]
    if not chosen:
        return {}
    frame = dataio.load(path, columns=list(dict.fromkeys([*chosen, *([image_column] if image_column else [])])), build_fast_copy=False)
    if image_column:
        # Normalize the small set of distinct identifiers, not millions of strings.
        raw = frame[image_column].astype('string').fillna('')
        groups = raw.map({v: sample_of(v) for v in raw.unique()})
        parts = frame.groupby(groups, sort=False, observed=True)
    else:
        parts = [(single_image, frame)]
    return {str(image): {c: _summary(part[c]) for c in chosen} for image, part in parts}


def values(root, rows, columns, *, per_image=False):
    if not isinstance(columns, list) or len(columns) > 30 or any(not isinstance(c, str) for c in columns):
        raise ValueError('Choose up to 30 metadata columns.')
    result = {r['roi']: {} for r in rows}
    if not columns:
        return result
    from spacemap.viewer.sources import fingerprint
    from spacemap.viewer.cells import _edits_token
    tables = defaultdict(list)
    for row in rows:
        if row.get('table_path'):
            tables[row['table_path']].append(row)
    for path, images in tables.items():
        roles = phenotyping_inputs.column_roles(root, path)
        image_column = '' if per_image else roles['roi']
        if not image_column and len(images) != 1:
            raise ValueError('Assign an image column for the shared table in Phenotyping before using its metadata.')
        stamp = (fingerprint(path), _edits_token(path))
        found = _table_values(path, stamp, tuple(columns), image_column, images[0]['roi'])
        for row in images:
            result[row['roi']] = found.get(sample_of(row['roi']), {})
    return result


def selected_rows(found, image_ids):
    """Resolve listed image IDs only; never accept arbitrary client paths."""
    if not isinstance(image_ids, list) or len(image_ids) > 2000:
        raise ValueError('Choose up to 2,000 listed images.')
    rows = found.get('rows', [])
    by_id = {r['roi']: r for r in rows if r.get('image_path')}
    if set(image_ids) - set(by_id):
        raise ValueError('Some images are no longer listed. Refresh Review before sharing.')
    return [r for r in rows if r['roi'] in image_ids and r.get('image_path')]


def snapshot(root, found, image_ids, columns, item_images=(), existing=()):
    """Resolve only listed image IDs. Never accept arbitrary paths from a client."""
    selected_rows(found, image_ids)
    rows = found.get('rows', [])
    paths = {str(Path(p).resolve()) for p in [*item_images, *existing]}
    chosen = [r for r in rows if r['roi'] in image_ids or str(Path(r.get('image_path', '')).resolve()) in paths]
    metadata = values(root, chosen, columns, per_image=found.get('per_image', False))
    return {str(Path(r['image_path']).resolve()): {
        'roi': r['roi'], 'image': r.get('image') or Path(r['image_path']).name,
        'metadata': metadata.get(r['roi'], {})} for r in chosen}
