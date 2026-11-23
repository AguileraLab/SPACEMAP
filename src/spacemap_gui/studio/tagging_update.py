"""Refresh selected annotation fields without rebuilding a unified dataset.

Only coordinates for selected images enter the tagging engine. Original tables
are streamed, preserving other rows/columns, and replaced together after checks.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from spacemap import labels, pending_columns
from spacemap.sharing import open_lock
from spacemap_gui.tabs import cell_tagging_ui as tab5
from .tagging_outputs import StagedTables

BATCH = 65536


def stamp(path):
    s = Path(path).stat()
    return [s.st_ino, s.st_size, s.st_mtime_ns]


def manifest_path(path):
    return Path(str(path) + '.tagging.json')


def clean_edits(paths):
    for path in paths:
        if labels.sidecar_path(path).exists() or list(pending_columns.store_dir(path).glob('*.parquet')):
            raise ValueError(f'{path.name}: write pending results and labels into the table before updating its tags.')


def _header(path):
    if path.suffix.lower() == '.parquet':
        return pq.read_schema(path).names
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return next(csv.reader(handle))


def _selected(path, batch, x, y, wanted, context_columns=()):
    """Keep stable original row positions; do not load the complete feature table."""
    columns = list(dict.fromkeys([batch, x, y, *context_columns]))
    if path.suffix.lower() == '.parquet':
        frames = (b.to_pandas() for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=columns))
    else:
        frames = pd.read_csv(path, usecols=columns, dtype={batch: str}, chunksize=BATCH)
    selected, offset, seen = [], 0, set()
    for frame in frames:
        frame.index = pd.RangeIndex(offset, offset + len(frame))
        offset += len(frame)
        values = frame[batch].astype(str)
        keep = values.isin(wanted)
        seen.update(values[keep])
        if keep.any():
            selected.append(frame.loc[keep])
    if seen != wanted:
        raise ValueError('Selected images missing from the table: ' + ', '.join(sorted(wanted - seen)))
    return pd.concat(selected), offset


def _equal_text(text, value):
    if value is None:
        return text.strip().lower() in ('', 'nan', 'none', 'null')
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return Decimal(text) == value
        except InvalidOperation:
            return False
    if isinstance(value, float):
        try:
            number = float(text) if text.strip() else math.nan
            return (math.isnan(number) and math.isnan(float(value))) or math.isclose(number, value, rel_tol=1e-12, abs_tol=1e-10)
        except ValueError:
            return False
    return text == str(value)


def _text(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ''
    return format(value, '.17g') if isinstance(value, float) else str(value)


def _write_parquet(source, target, patch, positions, units, report):
    reader = pq.ParquetFile(source)
    schema = reader.schema_arrow
    metadata = dict(schema.metadata or {})
    stored_units = json.loads(metadata.get(b'spacemap_column_units', b'{}'))
    for name, unit in units.items():
        previous = stored_units.get(name)
        if name in schema.names and previous and previous != unit:
            raise ValueError(f'{name} uses {previous}; keep those units when updating selected images (selected: {unit}).')
    metadata[b'spacemap_column_units'] = json.dumps({**stored_units, **units}).encode()
    fields = list(schema)
    for name in patch.column_names:
        if name not in schema.names:
            fields.append(patch.schema.field(name))
    output_schema = pa.schema(fields, metadata=metadata)
    patch_arrays = {name: pc.cast(patch[name].combine_chunks(), output_schema.field(name).type, safe=True) for name in patch.column_names}
    offset = 0
    with pq.ParquetWriter(target, output_schema, compression='snappy') as writer:
        for batch in reader.iter_batches(batch_size=BATCH):
            indices = pc.index_in(pa.array(np.arange(offset, offset + len(batch))), value_set=positions)
            selected = pc.is_valid(indices)
            arrays = []
            for field in output_schema:
                old = batch[field.name] if field.name in schema.names else pa.nulls(len(batch), type=field.type)
                arrays.append(pc.if_else(selected, pc.take(patch_arrays[field.name], indices), old)
                              if field.name in patch_arrays else old)
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=output_schema))
            offset += len(batch)
            report(f'Preparing Parquet: {offset:,} cells')
    if pq.ParquetFile(target).metadata.num_rows != offset:
        raise ValueError('Could not verify the updated Parquet row count.')


def _write_csv(source, target, patch, positions, count, parquet, identity, report):
    original_header = _header(source)
    header = original_header + [n for n in patch.column_names if n not in original_header]
    indices = {n: header.index(n) for n in patch.column_names}
    values = patch.to_pydict()
    lookup = {int(row): i for i, row in enumerate(positions.to_pylist())}
    identity_indices = [original_header.index(name) for name in identity]
    # A same-named companion must have exactly the same rows and cell identities.
    def expected_rows():
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=BATCH, columns=identity):
            data = batch.to_pydict()
            yield from zip(*(data[name] for name in identity))
    expected = iter(expected_rows()) if parquet else None
    total = 0
    with source.open(newline='', encoding='utf-8-sig') as inp, target.open('w', newline='', encoding='utf-8') as out:
        reader, writer = csv.reader(inp), csv.writer(out, lineterminator='\n')
        if next(reader) != original_header:
            raise ValueError('CSV header changed during the update.')
        writer.writerow(header)
        for row in reader:
            if len(row) != len(original_header):
                raise ValueError(f'CSV row {total + 1} has an unexpected number of fields.')
            if expected is not None:
                record = next(expected, None)
                if record is None or not all(_equal_text(row[column], value) for column, value in zip(identity_indices, record)):
                    raise ValueError(f'CSV and Parquet cells do not match at row {total + 1}. Refresh the CSV copy in Jobs first.')
            row += [''] * (len(header) - len(original_header))
            if total in lookup:
                index = lookup[total]
                for name, column in indices.items():
                    row[column] = _text(values[name][index])
            writer.writerow(row)
            total += 1
            if total % BATCH == 0:
                report(f'Preparing CSV: {total:,} cells')
        if total != count or (expected is not None and next(expected, None) is not None):
            raise ValueError('CSV and Parquet row counts differ. Refresh the CSV copy in Jobs first.')


def run(settings, folder, report):
    from . import tagging
    from .tag_columns import column_name

    state, found = settings['state'], settings['found']
    source = Path(state[tab5.MERGED_DIR_KEY]) / found['table']
    other = source.with_suffix('.csv' if source.suffix.lower() == '.parquet' else '.parquet')
    paths = [source] + ([other] if other.exists() else [])
    clean_edits(paths)
    before = {p: stamp(p) for p in paths}
    headers = {p: _header(p) for p in paths}
    if any(header != headers[source] for header in headers.values()):
        raise ValueError('The CSV and Parquet columns differ. Refresh the CSV copy in Jobs before updating tags.')
    parquet = next((p for p in paths if p.suffix.lower() == '.parquet'), None)
    wanted = set(settings.get('tables') or [])
    if not wanted:
        raise ValueError('Select the images to update.')
    rows = [r for r in found['rows'] if r['source'] in wanted]
    if {r['source'] for r in rows} != wanted:
        raise ValueError('One or more selected images are no longer in this table.')
    batch = found['batch_column']
    x, y = found['coordinates']['x'], found['coordinates']['y']
    planned = tagging.planned_columns(state)
    protected = {batch, x, y, 'cell_id', 'label', 'original_label'}
    if not planned or protected.intersection(planned):
        raise ValueError('Choose annotation columns with names different from cell IDs, image IDs and coordinates.')
    parent = state[tab5.TISSUE_LABEL_KEY]
    parent_fields = {flag: column_name(state.get(tagging.NAMES_KEY), parent, kind)
                     for flag, kind in [('Binary (0/1)', 'inside'), ('ROI ID', 'roi')]}
    context_columns = [n for n in parent_fields.values() if n in headers[source] and n not in planned]
    report('Reading selected cells from the unified table', 0)
    selected, total = _selected(source, batch, x, y, wanted, context_columns)
    groups = selected.groupby(selected[batch].astype(str), sort=False)
    exports = {tagging._sample_key(k): v for k, v in (settings.get('exports') or {}).items()}
    frames, records, geometry_stamps, units = [], [], {}, {}
    for number, row in enumerate(rows):
        geometry, _ = tagging.annotation_path(state, row['sample'], row.get('chosen') or None, exports)
        if not geometry:
            raise ValueError(f"{row['source']}: choose an annotation file before updating.")
        geometry_stamps[Path(geometry)] = stamp(geometry)
        # No existing annotation or marker fields enter finish(): unchecked fields
        # cannot be renamed, scaled again, or inadvertently included in the patch.
        frame = groups.get_group(row['source'])[[x, y]].copy()
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{row['source']}: cell coordinates contain missing or invalid values.")
        microns = tagging.microns_for(settings.get('root'), row['sample']) if settings.get('root') else None
        def progress(message, fraction=None):
            report(f"{row['source']}: {message}", None if fraction is None else .7 * (number + fraction) / len(rows))
        measured_state = {**state, tab5.OVERWRITE_COLUMNS_KEY: True}
        if context_columns:
            checks = [dict(r) for r in state.get(tab5.RULES_KEY, [])]
            parent_rule = next((r for r in checks if r['Annotation Label'] == parent), None)
            if parent_rule is None:
                parent_rule = {'Annotation Label': parent, 'Binary (0/1)': False}
                checks.append(parent_rule)
            for flag, name in parent_fields.items():
                if name in context_columns:
                    parent_rule[flag] = True
            measured_state[tab5.RULES_KEY] = checks
        result, generated, warnings, _ = tagging.tag_frame(frame, geometry, measured_state, x, y, microns, progress)
        for name in context_columns:
            old = pd.to_numeric(groups.get_group(row['source'])[name], errors='coerce')
            changed = old.notna() & ~np.isclose(old, result[name], equal_nan=True)
            if changed.any():
                raise ValueError(f"{row['source']}: parent polygons differ from saved {name} for {int(changed.sum()):,} cells. "
                                 'Use the original parent polygons, or also select that parent column and review analyses using it.')
        if set(generated) - set(context_columns) != set(planned):
            raise ValueError('The generated annotation columns differ from the selected columns.')
        frames.append(result[planned])
        actual_units = 'um' if state.get(tagging.UNITS_KEY) == 'um' and microns else 'px'
        for rule in state.get(tab5.RULES_KEY, []):
            label = rule['Annotation Label']
            for flag, kind, unit in [('Distance', 'distance', actual_units), ('ROI Size', 'roi_size', actual_units + '2')]:
                name = column_name(state.get(tagging.NAMES_KEY), label, kind)
                if rule.get(flag) and name in planned:
                    if name in units and units[name] != unit:
                        raise ValueError(f'{name}: pixel sizes are missing for some images. Choose pixels for a consistent column.')
                    units[name] = unit
        records.append({'Source': row['source'], 'Status': 'ok', 'Cells': len(frame), 'Columns': len(planned),
                        'Annotation': str(geometry), 'Warnings': '; '.join(warnings)})
    frame = pd.concat(frames).sort_index()
    if frame.index.has_duplicates:
        raise ValueError('A cell was included more than once; no table was changed.')
    positions = pa.array(frame.index.to_numpy(dtype=np.int64))
    patch = pa.Table.from_pandas(frame, preserve_index=False)
    prior_receipts = {}
    for path in paths:
        saved = manifest_path(path)
        previous = json.loads(saved.read_text()) if saved.is_file() else {}
        prior_receipts[path] = previous
        for name, unit in units.items():
            old_unit = previous.get('units', {}).get(name)
            if old_unit and old_unit != unit:
                raise ValueError(f'{name} uses {old_unit}; keep those units when updating selected images (selected: {unit}).')
    identity = list(dict.fromkeys([batch, x, y] + [n for n in ['cell_id', 'label'] if n in headers[source]]))
    backup = source.parent / '.cell_tagging_backups' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    with StagedTables(source.parent, folder) as staged:
        for path in paths:
            report(f'Preparing updated {path.name}', .75 if path == paths[0] else .85)
            if path.suffix.lower() == '.parquet':
                staged.stage(None, path, lambda _, out: _write_parquet(path, out, patch, positions, units, report))
            else:
                staged.stage(None, path, lambda _, out: _write_csv(path, out, patch, positions, total, parquet, identity, report))
            temporary = staged.pending[-1][0]
            os.chmod(temporary, path.stat().st_mode & 0o777)
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
        with ExitStack() as locks:
            for path in sorted(paths):
                handle = locks.enter_context(open_lock(labels.sidecar_path(path).with_suffix('.lock')))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError('This table is being saved by another job. Try again after it finishes.') from None
            clean_edits(paths)
            if any(stamp(p) != version for p, version in {**before, **geometry_stamps}.items()):
                raise ValueError('The table or annotation changed during tagging. No updates were saved; run again.')
            # The receipt is published in the same rollback-protected transaction.
            for path in paths:
                previous = prior_receipts[path]
                history = {r['Source']: r for r in previous.get('records', [])}
                history.update({r['Source']: r for r in records})
                receipt = {'records': list(history.values()), 'columns': planned,
                           'images': sorted(set(previous.get('images', [])) | wanted), 'rows': total,
                           'backup': str(backup), 'units': {**previous.get('units', {}), **units},
                           'annotations': {str(p): s for p, s in geometry_stamps.items()}}
                staged.stage(receipt, manifest_path(path), lambda data, out: out.write_text(json.dumps(data, indent=2)))
            written = staged.publish(report, backup_directory=backup)
    return {'records': records, 'outputs': [p for p in written if Path(p).suffix.lower() in {'.csv', '.parquet'}],
            'count': len(paths), 'failed': 0, 'updated': True, 'backup': str(backup), 'columns': planned,
            'cells': len(frame), 'rows': total, 'images': sorted(wanted)}
