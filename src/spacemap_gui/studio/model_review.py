"""Inspect, compare and combine reference models without changing their sources."""
from datetime import datetime, timezone
import uuid
from zoneinfo import ZoneInfo

import numpy as np

from spacemap.viewer import consensus_model
from . import reference_examples as examples, reference_models as models, reference_setup as setup


def prepare(root, body):
    keys = body.get('keys')
    action = body.get('action', 'inspect')
    if action not in {'inspect', 'validate', 'merge', 'copy', 'refit'}:
        raise ValueError('Choose Inspect, Validate, Merge or Save a copy.')
    if not isinstance(keys, list) or not 1 <= len(keys) <= 12 or len(set(keys)) != len(keys):
        raise ValueError('Select 1–12 models.')
    listed = setup.options(root)['models']
    if any(key not in {m['key'] for m in listed} for key in keys):
        raise ValueError('Choose models from the current Models folder.')
    if action == 'merge' and len(keys) < 2:
        raise ValueError('Select at least two models to merge.')
    if action in {'copy', 'refit'} and len(keys) != 1:
        raise ValueError('Select one model to save a custom copy.')
    name = body.get('name', '').strip()
    if len(name) > 100 or any(ord(c) < 32 for c in name):
        raise ValueError('Use a model name of up to 100 characters.')
    zone = body.get('timezone', 'UTC')
    try:
        ZoneInfo(zone)
    except (ValueError, KeyError, TypeError):
        raise ValueError('Choose a valid time zone.') from None
    # Only explicit selection loads a trusted joblib artifact.
    identifiers = [setup.prepare_model(root, key) for key in keys]
    mapping = body.get('phenotype_groups', {})
    from .mr_ml import check
    check({'phenotype_groups': mapping})
    classes = {str(c) for identifier in identifiers for c in models._source(identifier)[2]['model'].classes_}
    if set(mapping) - classes:
        raise ValueError('Group only phenotypes from the selected models.')
    features = body.get('features')
    if features is not None and (not isinstance(features, list) or not features or
            len(features) != len(set(features)) or any(not isinstance(f, str) for f in features)):
        raise ValueError('Keep at least one measured feature.')
    return {'project_root': str(root), 'model_ids': identifiers, 'action': action, 'features': features,
            **setup.phenotype_policy(root),
            'models_folder': setup.training(root)['models_folder'], 'name': name,
            'model_timezone': zone, 'phenotype_groups': mapping,
            'keep_detailed_predictions': body.get('keep_detailed_predictions', True) is not False}


def _records(bank):
    """One cell identity across reference columns; never count ancestry twice."""
    records = {}
    for group in bank:
        count = len(group['targets'])
        positions = np.asarray(group.get('positions', np.full(count, -1)))
        if len(positions) != count or np.any(positions < 0):
            raise ValueError('Cell IDs are missing from saved examples. Re-export this model from its original project.')
        for i, target in enumerate(group['targets']):
            identity = (str(group['sources'][i]), int(positions[i]))
            row = {'target': str(target), 'values': dict(zip(group['markers'], map(float, group['values'][i]))),
                   'reference': str(group.get('references', [''] * count)[i])}
            if not all(np.isfinite(v) for v in row['values'].values()):
                raise ValueError('Saved examples contain missing measurements. Retrain that model first.')
            old = records.get(identity)
            if old:
                if old['target'] != row['target']:
                    raise ValueError('The same cell has conflicting labels: ' + old['target'] + ' / ' + row['target'] + '. Select models with matching labels.')
                if any(not np.isclose(value, old['values'][key], rtol=1e-6, atol=1e-7)
                       for key, value in row['values'].items() if key in old['values']):
                    raise ValueError('The same cell has different measurements. Select models trained on the same measurements.')
                old['values'].update(row['values'])
            else:
                records[identity] = row
    return records


def _bank(records):
    grouped = {}
    for identity, row in records.items():
        grouped.setdefault(tuple(sorted(row['values'])), []).append((identity, row))
    return [{'markers': list(markers), 'values': np.asarray([[r['values'][m] for m in markers] for _, r in rows], np.float32),
             'targets': np.asarray([r['target'] for _, r in rows]), 'sources': np.asarray([key[0] for key, _ in rows]),
             'positions': np.asarray([key[1] for key, _ in rows], np.int64),
             'references': np.asarray([r['reference'] for _, r in rows])} for markers, rows in grouped.items()]


def _matrix(bank, markers):
    eligible = [g for g in bank if set(markers) <= set(g['markers'])]
    if not eligible:
        return np.empty((0, len(markers)), np.float32), np.asarray([], dtype=str)
    return (np.concatenate([np.asarray(g['values'])[:, [g['markers'].index(m) for m in markers]] for g in eligible]),
            np.concatenate([g['targets'] for g in eligible]))


def _score(model, values, targets, mapping=None):
    from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
    if not len(targets):
        return {'cells': 0, 'f1': None, 'per_class': [], 'confusion': []}
    probabilities = model.predict_proba(values)
    from .mr_ml import prediction_groups
    names, positions = prediction_groups(model.classes_, mapping or {})
    predicted = np.asarray(names)[np.column_stack([probabilities[:, p].sum(axis=1) for p in positions]).argmax(axis=1)]
    truth = np.asarray([(mapping or {}).get(str(t), str(t)) for t in targets])
    labels = sorted(set(truth) | set(predicted))
    precision, recall, f1, count = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    return {'cells': len(targets), 'f1': float(f1_score(truth, predicted, average='macro', zero_division=0)),
            'labels': labels, 'confusion': confusion_matrix(truth, predicted, labels=labels).tolist(),
            'per_class': [{'phenotype': name, 'precision': float(p), 'recall': float(r), 'f1': float(f), 'cells': int(n)}
                          for name, p, r, f, n in zip(labels, precision, recall, f1, count)]}


def _profile(bank, markers, classes):
    """Mean measured profiles, standardized per feature across stored examples.

    These describe reference cells, not class-specific model coefficients.
    Missing measurements remain blank, never zero-filled.
    """
    raw = np.full((len(classes), len(markers)), np.nan)
    scaled = raw.copy()
    for col, marker in enumerate(markers):
        values, targets = _matrix(bank, [marker])
        if not len(targets):
            continue
        mean, sd = values[:, 0].mean(), values[:, 0].std()
        for row, label in enumerate(classes):
            used = values[targets == label, 0]
            if len(used):
                raw[row, col] = used.mean()
                scaled[row, col] = (used.mean() - mean) / sd if sd > 0 else 0
    clean = lambda matrix: [[float(v) if np.isfinite(v) else None for v in row] for row in matrix]
    return {'rows': classes, 'columns': markers, 'values': clean(scaled), 'raw': clean(raw)}


def inspect(settings):
    snapshots = [models._source(identifier) for identifier in settings['model_ids']]
    features = [set(meta['markers']) for _, meta, _ in snapshots]
    shared = sorted(set.intersection(*features))
    union = sorted(set.union(*features))
    items, training, validation, issues = [], [], [], []
    for index, (identifier, meta, artifact) in enumerate(snapshots):
        bank, holdout = examples.archive(meta, artifact), examples.validation_archive(meta, artifact)
        training.extend(bank)
        validation.extend(holdout)
        item = {'id': identifier, 'name': meta.get('name') or meta['method'], 'method': meta['method'],
                'trained': meta.get('trained'), 'features': meta['markers'],
                'unique_features': sorted(features[index] - set().union(*(features[:index] + features[index+1:]))),
                'classes': [str(c) for c in artifact['model'].classes_],
                'training_cells': sum(len(g['targets']) for g in bank),
                'validation_cells': sum(len(g['targets']) for g in holdout),
                'phenotype_groups': meta.get('phenotype_groups', {}),
                'keep_detailed_predictions': meta.get('keep_detailed_predictions', True)}
        item['profile'] = _profile(bank, meta['markers'], item['classes']) if bank else None
        item['importance'] = dict(zip(meta['markers'], map(float, artifact['model'].feature_importances_)))
        if not bank:
            issues.append(item['name'] + ': no reusable examples. Prediction and custom copies still work.')
        items.append(item)
    fit, held, reused_validation = {}, {}, 0
    if not issues:
        try:
            fit, held = _records(training), _records(validation)
            # Check conflicts across partitions, too. Any cell previously used
            # for fitting stays training-only, even if another model held it out.
            _records(training + validation)
            reused_validation = len(held.keys() & fit.keys())
            held = {key: value for key, value in held.items() if key not in fit}
        except ValueError as exc:
            issues.append(str(exc))
    fit_bank, held_bank = _bank(fit), _bank(held)
    fit_features = sorted(set.intersection(*(set(g['markers']) for g in fit_bank))) if fit_bank else []
    if not fit_features and not issues:
        issues.append('No shared measured features. These models cannot be merged.')
    result = {'models': items, 'shared_features': shared, 'all_features': union,
              'merge_features': fit_features, 'issues': issues, 'can_merge': len(items) > 1 and not issues,
              'training_cells': len(fit), 'validation_cells': len(held),
              'reused_validation_cells': reused_validation,
              'duplicate_training_cells': sum(len(g['targets']) for g in training) - len(fit)}
    return result, snapshots, fit_bank, held_bank


def run(settings, folder, report):
    report('Reading selected models…', .05)
    result, snapshots, bank, holdout = inspect(settings)
    action = settings['action']
    if action == 'inspect':
        return result
    if action == 'validate':
        if result['issues']:
            raise ValueError(result['issues'][0])
        # Compare on exactly the same cells, excluding any cell used by ANY
        # selected model for training. Full measured panels are required here.
        required = result['all_features']
        values, targets = _matrix(holdout, required)
        if not len(targets):
            raise ValueError('No shared validation cells with all required features. Use Compare predictions (F1) with your reference labels.')
        for item, (_, meta, artifact) in zip(result['models'], snapshots):
            item['validation'] = _score(artifact['model'], values[:, [required.index(m) for m in meta['markers']]], targets,
                                        settings.get('phenotype_groups'))
        return result
    if action in {'merge', 'refit'} and result['issues']:
        raise ValueError(result['issues'][0])
    report('Training new model…' if action in {'merge', 'refit'} else 'Saving a custom copy…', .25)
    identifier = uuid.uuid4().hex
    mapping = settings['phenotype_groups']
    now = datetime.now(timezone.utc).isoformat()
    if action in {'merge', 'refit'}:
        from sklearn.ensemble import RandomForestClassifier
        from spacemap.phenotype_policy import allowed
        blocked = settings.get('excluded_training_phenotypes', [])
        bank = [examples.take(g, allowed(g['targets'], blocked)) for g in bank]
        holdout = [examples.take(g, allowed(g['targets'], blocked)) for g in holdout]
        markers = settings.get('features') or result['merge_features']
        if set(markers) - set(result['merge_features']):
            raise ValueError('Use features measured in all saved training examples.')
        values, targets = _matrix(bank, markers)
        if len(targets) > examples.LIMIT:
            raise ValueError('More than 50,000 unique saved cells. Merge fewer models at a time.')
        expected = {c for item in result['models'] for c in item['classes']}
        expected = {c for c in expected if allowed([c], blocked)[0]}
        if expected - set(targets):
            raise ValueError('Saved examples are missing phenotypes: ' + ', '.join(sorted(expected - set(targets))) + '.')
        if len(set(targets)) < 2:
            raise ValueError('Merging needs examples of at least two phenotypes.')
        model = RandomForestClassifier(n_estimators=96, max_depth=20, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=2, bootstrap=True, oob_score=True)
        if action == 'refit':
            from sklearn.base import clone
            model = clone(snapshots[0][2]['model']).set_params(n_jobs=2, warm_start=False)
        model.fit(values, targets)
        model.n_jobs = 1
        validation_values, validation_targets = _matrix(holdout, markers)
        score = _score(model, validation_values, validation_targets)
        meta = {'method': 'Random Forest' if action == 'merge' else snapshots[0][1]['method'],
                'excluded_training_phenotypes': blocked,
                'column': 'Merged labels' if action == 'merge' else snapshots[0][1].get('column', ''), 'markers': markers,
                'classes': list(map(str, model.classes_)), 'total_fit_cells': len(targets), 'training_cells': len(targets),
                'saved_example_cells': len(targets), 'examples': len(targets), 'adaptable': True, 'allow_missing': True,
                'archived_features': sorted({m for g in bank for m in g['markers']}),
                'validation_held_out': bool(score['cells']), 'held_out_cells': score['cells'],
                'held_out_macro_f1': score['f1'], 'validation_result': score,
                'held_out_per_class': {p['phenotype']: p['recall'] for p in score['per_class']},
                'marker_importance': dict(zip(markers, map(float, model.feature_importances_)))}
        sample = models.training_sample(values, targets, rows=np.full(len(targets), -1, np.int64))
        artifact = {'model': model, 'rows': np.array([-1]), 'oob': np.zeros((1, len(model.classes_))),
                    'training_sample': {**sample, 'markers': markers}, 'example_archive': bank,
                    'validation_archive': holdout, 'validation_sample': {'values': validation_values,
                        'targets': validation_targets, 'rows': np.full(len(validation_targets), -1, np.int64)}}
    else:
        _, original, source = snapshots[0]
        meta = {k: v for k, v in original.items() if k not in {'table_path', 'training_sources', 'imported_from', 'auto_train'}}
        artifact = {k: v for k, v in source.items() if k not in {'_file_stamp', 'training_origin'}}
        artifact.update(example_archive=examples.archive(original, source), validation_archive=examples.validation_archive(original, source))
    mapping = {key: value for key, value in mapping.items() if key in set(map(str, artifact['model'].classes_))}
    meta.update(model_id=identifier, revision=identifier, sklearn=consensus_model._version(), trained=now,
                classes=list(map(str, artifact['model'].classes_)),
                training_started=now, model_timezone=settings['model_timezone'],
                name=settings['name'] or ('Merged model' if action == 'merge' else snapshots[0][1].get('name', 'Model') + ' · custom'),
                parent_model_ids=settings['model_ids'], phenotype_groups=mapping,
                keep_detailed_predictions=settings['keep_detailed_predictions'], persistent_version=True,
                project_key=examples.source_key(settings['project_root']))
    if action in {'copy', 'refit'}:
        meta['base_model_id'] = settings['model_ids'][0]
    artifact.update(revision=identifier, provenance=meta)
    report('Saving the new version…', .9)
    saved = models._save_model(identifier, artifact, meta)
    setup.publish_training(settings['project_root'], [saved], None, settings['models_folder'])
    if saved.get('export_error'):
        raise ValueError(saved['export_error'])
    return {'models': [saved], 'message': 'Saved in Models. Choose it and click Use.',
            'review': result, 'table': '', 'model_folder': saved['shared_path']}
