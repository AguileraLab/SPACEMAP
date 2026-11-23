"""Reusable, immutable cell membership for a reference model's training split.

Stored separately from model weights, so comparisons survive model retirement.
Only sampled rows are recorded; a split is never silently regenerated at scoring.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from spacemap_workflows.leiden_job import write_json


def directory(root):
    return Path(root) / 'outputs' / 'phenotyping' / 'reference_splits'


def token(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def identities(source, rows):
    from .data_editor import Source
    reader = Source(source['table'])
    columns = list(dict.fromkeys(c for c in [source.get('roi_column'), source.get('cell_id_col'),
        source.get('label_column'), 'x_centroid', 'y_centroid'] if c and c in reader.fields))
    if not any(c in columns for c in (source.get('cell_id_col'), source.get('label_column')) if c):
        raise ValueError('Choose a Cell ID column before saving a training split.')
    values = reader.take(rows, columns)
    keys = [[token(values[c][i]) for c in columns] for i in range(len(rows))]
    return reader.length, columns, keys


def make(settings, rows, targets, training, validation):
    """Capture identities and truth for the exact examples fitted/evaluated."""
    origin = settings.get('training_origin')
    sources = settings.get('training_sources') or [settings]
    membership = np.full(len(rows), 'training', object)
    membership[validation] = 'validation'
    entries = []
    for source in sources:
        path = str(Path(source['table']).resolve())
        if origin:
            keep = np.asarray(origin['tables'])[rows] == path
            local = np.asarray(origin['rows'], np.int64)[rows[keep]]
        else:
            keep = np.ones(len(rows), bool)
            local = rows
        if not keep.any():
            continue
        order = np.argsort(local)
        local = local[order]
        length, columns, keys = identities(source, local)
        entries.append({'table': path, 'length': length, 'identity_columns': columns, 'identity': keys,
                        'rows': local.tolist(), 'truth': targets[keep][order].tolist(), 'subset': membership[keep][order].tolist()})
    record = {'version': 1, 'reference': settings['training_column'], 'training_percent': settings.get('training_percent', 70),
              'seed': 42, 'training_cells': len(training), 'validation_cells': len(validation), 'tables': entries}
    # Identical fits (e.g. Auto-train without new labels) share one membership record.
    record['id'] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:32]
    return record


def save(root, record, models):
    path = directory(root) / (record['id'] + '.json')
    from .settings import project_lock
    with project_lock(root):
        from spacemap.sharing import shared_directory
        shared_directory(path.parent)
        existing = json.loads(path.read_text()) if path.is_file() else {}
        by_id = {m['id']: m for m in existing.get('models', [])}
        by_id.update({m['model_id']: {'id': m['model_id'], 'method': m['method'], 'trained': m['trained']}
                      for m in models if m.get('split_id') == record['id']})
        if not by_id:
            return
        # Bound display metadata while keeping the split itself permanently available.
        versions = sorted(by_id.values(), key=lambda m: m['trained'], reverse=True)[:20]
        write_json(path, {**record, 'models': versions, 'trained': versions[0]['trained']})


def options(root, allowed):
    allowed = {str(Path(p).resolve()) for p in allowed}
    found = []
    for path in directory(root).glob('*.json'):
        try:
            item = json.loads(path.read_text())
            if set(t['table'] for t in item['tables']) <= allowed:
                found.append({k: item[k] for k in ('id', 'reference', 'training_percent', 'training_cells', 'validation_cells', 'trained')})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    # Recent pre-manifest models already have explicit, disjoint holdouts.
    # Offer those too, but check their original input revisions before use.
    from . import reference_models as models
    seen = {x['id'] for x in found}
    for path in models.directory().glob('*.json'):
        if not re.fullmatch(r'[0-9a-f]{32}', path.stem):
            continue
        try:
            meta = json.loads(path.read_text())
            tables = {s['table'] for s in meta.get('training_sources', [])} or {meta.get('table_path')}
            if (not meta.get('split_id') and not meta.get('imported') and meta.get('validation_held_out')
                    and tables <= allowed and path.with_suffix('.joblib').is_file()):
                found.append({'id': 'model:' + path.stem, 'reference': meta['column'], 'trained': meta['trained'],
                              'training_percent': meta['training_percent'], 'training_cells': meta['training_cells'],
                              'validation_cells': meta['held_out_cells']})
            elif meta.get('split_id') in seen:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(found, key=lambda item: item['trained'], reverse=True)


def read(root, identifier, allowed):
    if identifier not in {x['id'] for x in options(root, allowed)}:
        raise ValueError('Choose a saved split from this project.')
    if not identifier.startswith('model:'):
        if not re.fullmatch(r'[0-9a-f]{32}', identifier):
            raise ValueError('Choose a listed split.')
        return json.loads((directory(root) / (identifier + '.json')).read_text())
    from . import reference_models as models
    _, meta, artifact = models._source(identifier[6:])
    training, validation = artifact.get('training_sample', {}), artifact.get('validation_sample', {})
    rows = np.asarray(artifact['rows'], np.int64)
    if set(rows) != set(training.get('rows', [])) or not len(validation.get('rows', [])):
        raise ValueError('This older model lacks a complete saved split. Train a new version once.')
    sources = meta.get('training_sources') or [{'table': meta['table_path'], 'markers': meta['markers'],
        'training_column': meta['column'], 'roi_column': 'batch_id', 'label_column': 'label'}]
    for source in sources:
        expected = source.get('marker_revision', meta.get('revision'))
        if models.marker_revision(source) != expected:
            raise ValueError('This older model’s input table changed. Train a new version to save a reusable split.')
    order = np.argsort(training['rows'])
    targets = np.r_[np.asarray(training['targets'])[order], validation['targets']]
    combined = np.r_[np.sort(rows), validation['rows']]
    settings = {**sources[0], 'training_percent': meta['training_percent'],
                'training_sources': meta.get('training_sources'), 'training_origin': artifact.get('training_origin')}
    return make(settings, combined, targets, np.arange(len(rows)), np.arange(len(rows), len(combined)))


def selected_rows(entry, subset, reference):
    """Check identity/truth before reusing positions, including after CSV commits."""
    from .data_editor import Source
    from spacemap.viewer.cells import values_at_rows
    rows = np.asarray(entry['rows'], np.int64)
    keep = np.ones(len(rows), bool) if subset == 'both' else np.asarray(entry['subset']) == subset
    rows = rows[keep]
    reader = Source(entry['table'])
    if reader.length != entry['length'] or not set(entry['identity_columns']) <= set(reader.fields):
        raise ValueError('Cells changed since this split. Train a new version before using it.')
    values = reader.take(rows, entry['identity_columns'])
    keys = [[token(values[c][i]) for c in entry['identity_columns']] for i in range(len(rows))]
    expected = [item for item, yes in zip(entry['identity'], keep) if yes]
    if keys != expected:
        raise ValueError('Cell order or identities changed since this split. Train a new version before using it.')
    truth = np.asarray(entry['truth'])[keep]
    current = np.asarray([str(v).strip() if pd.notna(v) else '' for v in values_at_rows(entry['table'], rows, reference)])
    if not np.array_equal(current, truth):
        raise ValueError('Reference labels in this split changed. Train a new version, or assess All labelled cells.')
    return rows
