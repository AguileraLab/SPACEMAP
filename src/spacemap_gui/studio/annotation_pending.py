"""Explicit Jobs write-back of edited annotation copies, with conflict checks."""
import json
from pathlib import Path

from . import annotation_components as components, annotation_edits as edits


def _stamp(path):
    stat = Path(path).stat()
    # Nanosecond timestamps exceed JavaScript's exact integer range. Keep the
    # browser round trip lossless, or every legitimate write looks stale.
    return [str(stat.st_size), str(stat.st_mtime_ns), str(stat.st_ino)]


def pending(root):
    # Only small provenance records and stats; never parse polygon geometry in
    # Jobs polling. Existing editable copies are discovered without migration.
    base = Path(root) / 'outputs/studio/imported_annotations'
    rows = []
    for meta in sorted(base.glob('*/*.source.json')):
        try:
            record = json.loads(meta.read_text())
            copy = meta.with_name(meta.name[:-len('.source.json')] + '.geojson')
            stamp = _stamp(copy)
            if list(map(str, record.get('written_copy_stamp') or [])) == stamp:
                continue
            identity = components.display_identity(copy)
            if not identity:
                continue
            rows.append({'id': str(copy.relative_to(base)), 'name': identity['name'],
                         'image': Path(record['image']).name, 'source': record['source'],
                         'stamp': stamp, 'missing': not Path(record['source']).is_file()})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def write(settings, folder, report):
    root = Path(settings['root'])
    requested = settings.get('items') or []
    if not requested:
        raise ValueError('Choose annotation files to write.')
    # Resolve IDs only through the pending inventory, never a client-supplied path.
    known = {row['id']: row for row in pending(root)}
    rows = []
    for item in requested:
        row = known.get(item.get('id'))
        if row is None or row['stamp'] != item.get('stamp'):
            raise ValueError('The annotation edits changed. Refresh Jobs before writing.')
        rows.append(row)
    if len({row['source'] for row in rows}) != len(rows):
        raise ValueError('Several copies target the same source. Write one selected copy at a time.')
    written = []
    for number, row in enumerate(rows):
        copy = root / 'outputs/studio/imported_annotations' / row['id']
        meta = copy.with_suffix('.source.json')
        record = json.loads(meta.read_text())
        # Same lock as component edits. Never commit a newer, unrequested edit.
        from .annotation_inputs import import_folder
        with edits.locked(import_folder(root, record['image']) / '.component-actions'):
            with edits.locked(copy):
                if _stamp(copy) != row['stamp']:
                    raise ValueError('The annotation edits changed. Refresh Jobs before writing.')
                target = Path(record['source'])
                if not target.is_file():
                    raise ValueError('Original annotation file is missing: ' + target.name)
                report('Writing ' + target.name + '…', number / len(rows))
                features, _ = edits.read(copy)
                # replace checks the original revision under its own lock and
                # archives its previous version before atomic replacement.
                edits.replace(target, features, record['source_revision'], 'Wrote reviewed annotation copy', edited=True)
                record.update(source_revision=edits.revision(target), written_copy_stamp=_stamp(copy))
                edits._atomic(meta, json.dumps(record).encode())
                written.append(str(target))
    report('Annotation files written.', 1)
    return {'files': written, 'count': len(written)}
