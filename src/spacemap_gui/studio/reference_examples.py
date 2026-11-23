"""Retained training examples for explicit, reproducible cross-project refits.

Panels stay separate in the archive. A combined classifier uses their shared
features: an unmeasured marker is never imputed as zero. Validation examples
never enter this archive. Source hashes permit replacement without publishing
private table paths.
"""
import hashlib
from pathlib import Path

import numpy as np

LIMIT = 50000


class MissingTrainingExamples(ValueError):
    """The selected model cannot contribute identifiable old examples."""


MISSING_EXAMPLES = ('Old training examples unavailable. In Details & options, turn off '
                    'Keep previous training examples to use only your labels.')


def source_key(table):
    return hashlib.sha256(str(Path(table).resolve()).encode()).hexdigest()


def sources(settings, rows):
    origin = settings.get('training_origin')
    if origin:
        return np.asarray([source_key(origin['tables'][int(row)]) for row in rows])
    return np.repeat(source_key(settings['table']), len(rows))


def positions(settings, rows):
    origin = settings.get('training_origin')
    return np.asarray(origin['rows'], np.int64)[rows] if origin else np.asarray(rows, np.int64)


def take(group, keep):
    return {**group, **{k: np.asarray(group[k])[keep]
                       for k in ('values', 'targets', 'sources', 'positions', 'references') if k in group}}


def _enrich(meta, artifact, bank, validation, seen):
    """Recover row/column provenance for versions written before it was archived.

    Own examples have saved row IDs. Inherited examples are matched exactly to
    immutable ancestors; unknown identities are never guessed.
    """
    if all('positions' in g and 'references' in g for g in bank):
        return bank
    result = [dict(g) for g in bank]
    own_rows = np.asarray((artifact.get('validation_sample') or {}).get('rows', []) if validation else artifact.get('rows', []), np.int64)
    if result and len(own_rows) == len(result[-1]['targets']):
        last = result[-1]
        origin = artifact.get('training_origin')
        own_sources = (sources({'training_origin': origin}, own_rows) if origin else
                       np.repeat(source_key(meta['table_path']), len(own_rows)) if meta.get('table_path') else [])
        if np.array_equal(own_sources, last['sources']):
            last.setdefault('positions', positions({'training_origin': origin}, own_rows))
            last.setdefault('references', np.repeat(meta.get('column', ''), len(own_rows)))
    parent = meta.get('base_model_id')
    if parent and parent not in seen and any('positions' not in g or 'references' not in g for g in result):
        from . import reference_models
        try:
            _, older_meta, older = reference_models._source(parent)
        except (ValueError, OSError):
            older = None
        if older is not None:
            getter = validation_archive if validation else archive
            previous = getter(older_meta, older, _seen=seen | {parent})
            lookup = {}
            for group in previous:
                ids = group.get('positions', np.full(len(group['targets']), -1))
                refs = group.get('references', np.repeat('', len(group['targets'])))
                for i, value in enumerate(group['values']):
                    key = (tuple(group['markers']), str(group['sources'][i]), str(group['targets'][i]),
                           np.asarray(value, np.float32).tobytes())
                    identity = (int(ids[i]), str(refs[i]))
                    lookup[key] = identity if key not in lookup or lookup[key] == identity else (-1, '')
            for group in result:
                if 'positions' in group and 'references' in group:
                    continue
                matched = [lookup.get((tuple(group['markers']), str(source), str(target),
                                      np.asarray(value, np.float32).tobytes()), (-1, ''))
                           for source, target, value in zip(group['sources'], group['targets'], group['values'])]
                group['positions'] = np.asarray([v[0] for v in matched], np.int64)
                group['references'] = np.asarray([v[1] for v in matched])
    return result


def archive(meta, artifact, _seen=None):
    if artifact.get('example_archive'):
        return _enrich(meta, artifact, artifact['example_archive'], False, _seen or set())
    sample = artifact.get('training_sample')
    if not sample:
        return []
    markers = sample.get('markers', meta.get('markers', []))
    values, targets = np.asarray(sample.get('values')), np.asarray(sample.get('targets'))
    if values.ndim != 2 or targets.ndim != 1 or values.shape != (len(targets), len(markers)) or not len(targets):
        return []
    rows = np.asarray(sample.get('rows', []))
    if rows.shape != targets.shape:
        return []
    keep = np.ones(len(targets), dtype=bool)
    if not meta.get('validation_held_out'):
        # Older Studio evaluated an internal split, then fitted ALL examples.
        # Its saved sample is reusable when row IDs prove membership in that
        # final fit. They are historical training data, not held-out validation.
        fitted = np.asarray(artifact.get('rows', []))
        if rows.shape != targets.shape or fitted.ndim != 1 or not len(fitted):
            return []
        keep &= np.isin(rows, fitted)
    validation = artifact.get('validation_sample')
    if validation is not None:
        held_out = np.asarray(validation.get('rows', []))
        if len(held_out):
            if rows.shape != targets.shape:
                return []
            keep &= ~np.isin(rows, held_out)
    if not keep.any():
        return []
    origin = artifact.get('training_origin')
    if origin and 'rows' in sample:
        if rows.shape != targets.shape or np.any(rows < 0) or np.any(rows >= len(origin['tables'])):
            return []
        keys = sources({'training_origin': origin}, rows)
    elif meta.get('table_path'):
        keys = np.repeat(source_key(meta['table_path']), len(sample['targets']))
    else:
        return []  # Cannot safely exclude this project's previous labels.
    return [{'markers': list(markers), 'values': values[keep],
             'targets': targets[keep], 'sources': keys[keep],
             'positions': positions({'training_origin': origin}, rows[keep]),
             'references': np.repeat(meta.get('column', ''), int(keep.sum()))}]


def require_reusable(settings):
    """Fail before launching a job or scanning a large cell table."""
    if settings.get('base_model_id') and settings.get('include_saved_examples'):
        from . import reference_models
        _, meta, artifact = reference_models._source(settings['base_model_id'])
        if not archive(meta, artifact):
            raise MissingTrainingExamples(MISSING_EXAMPLES)


def validation_archive(meta, artifact, _seen=None):
    """Preserve independent checks across projects, separately from fit data."""
    if artifact.get('validation_archive'):
        return _enrich(meta, artifact, artifact['validation_archive'], True, _seen or set())
    sample = artifact.get('validation_sample')
    if not meta.get('validation_held_out') or not sample or not len(sample['targets']):
        return []
    if artifact.get('training_origin'):
        keys = sources({'training_origin': artifact['training_origin']}, sample['rows'])
    elif meta.get('table_path'):
        keys = np.repeat(source_key(meta['table_path']), len(sample['targets']))
    else:
        return []
    return [{'markers': list(meta['markers']), 'values': sample['values'],
             'targets': sample['targets'], 'sources': keys,
             'positions': positions({'training_origin': artifact.get('training_origin')}, sample['rows']),
             'references': np.repeat(meta.get('column', ''), len(sample['rows']))}]


def retain_previous(group, settings, rows):
    """Replace this reference's examples, not every reference from the same table.

    Cells labelled in the current reference take precedence across columns. In
    particular, a new validation cell cannot remain among old training examples.
    """
    current = {source_key(s['table']) for s in settings.get('training_sources', [settings])}
    same_table = np.isin(group['sources'], list(current))
    from spacemap.phenotype_policy import allowed
    permitted = allowed(group['targets'], settings.get('excluded_training_phenotypes', []))
    keep = ~same_table
    if not same_table.any():
        return keep & permitted
    refs = np.asarray(group.get('references', np.repeat('', len(keep))))
    old_rows = np.asarray(group.get('positions', np.full(len(keep), -1)), np.int64)
    if np.any(same_table & ((refs == '') | (old_rows < 0))):
        raise ValueError('Old cell identities are unavailable. Load an original model before adding labels from this table.')
    source_ids, local_rows = sources(settings, rows), positions(settings, rows)
    from . import phenotyping_images
    for source in settings.get('training_sources', [settings]):
        group_rows = np.flatnonzero(np.asarray(group['sources']) == source_key(source['table']))
        if len(group_rows) and source.get('selected_images') is not None:
            keep[group_rows] |= ~phenotyping_images.rows_mask(source, old_rows[group_rows])
    for source in current:
        candidate = (np.asarray(group['sources']) == source) & (refs != settings['training_column'])
        keep |= candidate & ~np.isin(old_rows, local_rows[source_ids == source])
    return keep & permitted


def validation_bank(settings, values, targets, rows, testing, parent=None):
    bank = []
    if parent is not None and settings.get('include_saved_examples'):
        for group in validation_archive(*parent):
            keep = retain_previous(group, settings, rows)
            if keep.any():
                bank.append(take(group, keep))
    bank.append({'markers': list(settings['markers']), 'values': values[testing], 'targets': targets[testing],
                 'sources': sources(settings, rows[testing]), 'positions': positions(settings, rows[testing]),
                 'references': np.repeat(settings['training_column'], len(testing))})
    # Bound validation storage independently; these rows never enter combine().
    count = sum(len(group['targets']) for group in bank)
    if count > LIMIT:
        chosen = np.random.default_rng(42).choice(count, LIMIT, replace=False)
        offset, reduced = 0, []
        for group in bank:
            keep = np.isin(np.arange(len(group['targets'])) + offset, chosen)
            offset += len(keep)
            if keep.any():
                reduced.append(take(group, keep))
        bank = reduced
    return bank


def combine(settings, values, targets, rows, training, parent=None):
    markers = list(settings['markers'])
    new = {'markers': markers, 'values': values[training], 'targets': targets[training],
           'sources': sources(settings, rows[training]), 'positions': positions(settings, rows[training]),
           'references': np.repeat(settings.get('training_column', ''), len(training))}
    bank = []
    if parent is not None and settings.get('include_saved_examples'):
        old = archive(*parent)
        if not old:
            raise MissingTrainingExamples(MISSING_EXAMPLES)
        # Refresh this reference column while retaining other references/ROIs.
        # New labelled rows override the same cells in any older reference.
        for group in old:
            keep = retain_previous(group, settings, rows)
            if keep.any():
                bank.append(take(group, keep))
    # Bound retained examples across generations. Keep current examples and
    # a reproducible proportional sample of the older projects.
    budget = max(0, LIMIT - len(training))
    count = sum(len(group['targets']) for group in bank)
    if count > budget:
        rng = np.random.default_rng(42)
        labels = np.concatenate([g['targets'] for g in bank])
        required = np.concatenate([rng.choice(np.flatnonzero(labels == label), min(5, (labels == label).sum()), replace=False)
                                   for label in np.unique(labels)])
        if len(required) > budget:
            raise ValueError('Too many saved phenotypes for the example budget. Use a smaller source model or train on this project only.')
        remaining = np.setdiff1d(np.arange(count), required)
        chosen = set(np.r_[required, rng.choice(remaining, budget-len(required), replace=False)].tolist())
        offset, reduced = 0, []
        for group in bank:
            keep = np.asarray([i + offset in chosen for i in range(len(group['targets']))])
            offset += len(keep)
            if keep.any():
                reduced.append(take(group, keep))
        bank = reduced
    shared = [name for name in markers if all(name in group['markers'] for group in bank)]
    if not shared:
        raise ValueError('No shared features with the saved examples. Turn off Keep previous training examples to train with this project’s features.')
    bank.append(new)
    matrices = [np.asarray(g['values'])[:, [g['markers'].index(m) for m in shared]] for g in bank]
    fit_values = np.concatenate(matrices)
    fit_targets = np.concatenate([g['targets'] for g in bank])
    old_count = len(fit_targets) - len(training)
    fit_rows = np.r_[np.full(old_count, -1, np.int64), rows[training]]
    return shared, fit_values, fit_targets, fit_rows, bank, old_count
