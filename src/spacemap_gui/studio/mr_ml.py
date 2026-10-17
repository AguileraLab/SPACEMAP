"""Classic MR-ML settings and cohort prediction with saved reference models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spacemap import dataio
from . import consensus

DEFAULTS = {
    'workflow': 'classic', 'model_id': '', 'preserve_reference_labels': True, 'match_scale': False,
    'markers': [], 'ground_truth_col': 'phenotype_ref1', 'validation_ground_truth_col': '',
    'phenotyping_methods': [], 'exclude_phenotypes': ['Others'],
    'morphology_cols': [], 'compartment_cols': [],
    'feature_weights': {'markers': 1.0, 'morphology': 0.0, 'compartments': 0.0},
    'ml_method': 'random_forest', 'n_estimators': 1500, 'test_size': .3, 'random_state': 42,
    'cpu_workers': 0, 'use_z_normalization': False, 'batch_normalization_col': 'batch_id',
    'drop_overlapping_validation': True,
    'enable_filter': True, 'filter_col': 'tissue', 'filter_val': '1',
    'output_phenotype_col': 'phenotype_ml_mr', 'confidence_col': 'ml_mr_confidence',
    'refined_output_phenotype_col': 'phenotype_ml_mr_refined',
    'refined_confidence_threshold': .75, 'refined_low_confidence_label': 'Others',
}


def cache_path(root):
    return Path(root) / 'outputs/phenotyping/cache/mr_ml_params.json'


def values(root):
    path = cache_path(root)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS and v is not None}}


def check(changes):
    from .leiden import COLUMN
    if not isinstance(changes, dict) or set(changes) - set(DEFAULTS):
        raise ValueError('Unknown MR-ML setting.')
    for key, value in changes.items():
        default = DEFAULTS[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, list):
            valid = isinstance(value, list) and len(value) <= 200 and all(isinstance(v, str) and v.strip() for v in value)
        elif key == 'feature_weights':
            valid = isinstance(value, dict) and set(value) == set(default) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and 0 <= v <= 100 for v in value.values())
        elif key in ('n_estimators', 'test_size', 'random_state', 'cpu_workers', 'refined_confidence_threshold'):
            limits = {'n_estimators': (10, 5000), 'test_size': (.05, .5), 'random_state': (0, 2**31-1),
                      'cpu_workers': (0, 4096), 'refined_confidence_threshold': (0, 1)}
            low, high = limits[key]
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high
            if isinstance(default, int):
                valid = valid and int(value) == value
        else:
            valid = isinstance(value, str) and len(value) <= 200
            if key == 'workflow':
                valid = value in ('classic', 'saved_model')
            elif key == 'ml_method':
                valid = value in consensus.METHODS
            elif key in ('output_phenotype_col', 'refined_output_phenotype_col'):
                valid = valid and bool(consensus.OUTPUT.fullmatch(value))
            elif key == 'confidence_col':
                valid = valid and bool(COLUMN.fullmatch(value))
            elif key == 'refined_low_confidence_label':
                valid = valid and bool(value.strip())
        if not valid:
            raise ValueError('Check the MR-ML ' + key.replace('_', ' ') + '.')


def save(root, changes, expected):
    from .settings import project_lock, revision
    check(changes)
    with project_lock(root):
        if expected != revision(values(root)):
            raise ValueError('MR-ML settings changed in another window. Reload before saving.')
        path = cache_path(root)
        saved = json.loads(path.read_text()) if path.is_file() else {}
        saved.update(changes)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(saved, indent=2))
        temp.replace(path)
    return values(root)


def job_settings(root, table, roles, chosen, overwrite=False):
    """Snapshot the saved configuration against the entire combined table."""
    from . import tasks, reference_models
    check(chosen)
    columns = dataio.columns(table)
    saved_model = chosen['workflow'] == 'saved_model'
    if saved_model:
        meta = reference_models.metadata(chosen['model_id'])
        markers = [name for name in meta['markers'] if name in columns]
        if not markers:
            raise ValueError('The table has none of the saved model features.')
        if len(markers) < len(meta['markers']) and not (meta.get('adaptable') and meta.get('allow_missing', True)):
            raise ValueError('This saved model requires all its features. Choose a compatible model.')
        reference = chosen['ground_truth_col'] if chosen['ground_truth_col'] in columns else None
    else:
        from spacemap.ml_mr_utils import _looks_like_reference_column, _looks_like_annotation_metadata
        from spacemap.viewer.reference import columns as reference_columns
        markers, reference = chosen['markers'], chosen['ground_truth_col']
        references = set(reference_columns(table))
        if any(name in references or name in (reference, chosen['validation_ground_truth_col']) or _looks_like_reference_column(name)
               or _looks_like_annotation_metadata(name) for name in chosen['phenotyping_methods']):
            raise ValueError('Choose independent phenotype columns for method votes, excluding reference and annotation metadata.')
        features = [*markers, *chosen['morphology_cols'], *chosen['compartment_cols']]
        if any(name.startswith('phenotype_') or name in references or name in (
                reference, chosen['validation_ground_truth_col'], roles.get('cell_id'), 'cell_id', 'label') for name in features):
            raise ValueError('Choose measured features, excluding cell identities and phenotype/reference labels.')
        required = [reference, *markers, *chosen['morphology_cols'], *chosen['compartment_cols'], *chosen['phenotyping_methods']]
        if chosen['validation_ground_truth_col']:
            required.append(chosen['validation_ground_truth_col'])
        if chosen['use_z_normalization']:
            required.append(chosen['batch_normalization_col'])
        missing = set(required) - set(columns)
        if missing or not markers:
            raise ValueError('Choose available reference, marker and method columns. Missing: ' + ', '.join(sorted(missing)))
    if chosen['enable_filter'] and chosen['filter_col'] not in columns:
        raise ValueError('Choose an existing row-filter column, or turn off the row filter.')
    settings = {'table': str(table), 'project_root': str(root), 'config': chosen,
                'markers': markers, 'reference_column': reference,
                'methods': [] if saved_model else chosen['phenotyping_methods'],
                'morphology_cols': [] if saved_model else chosen['morphology_cols'],
                'compartment_cols': [] if saved_model else chosen['compartment_cols'],
                'output_col': chosen['output_phenotype_col'], 'confidence_column': chosen['confidence_col'],
                'refined_column': chosen['refined_output_phenotype_col'],
                'confidence_threshold': chosen['refined_confidence_threshold'],
                'low_confidence_label': chosen['refined_low_confidence_label'],
                'excluded_phenotypes': chosen['exclude_phenotypes'],
                'cell_id_col': roles.get('cell_id') or 'cell_id', 'roi_column': roles.get('roi') or 'batch_id',
                'label_column': 'label' if 'label' in columns else roles.get('cell_id') or 'cell_id'}
    settings['auxiliary_columns'] = [settings['confidence_column'], settings['refined_column']]
    settings['overwrite_columns'] = [settings['output_col'], *settings['auxiliary_columns']] if overwrite else []
    tasks.validate_mr_ml(settings)
    settings['revision'] = tasks.source_revision(settings)
    return settings


def predict_saved(settings, folder, report):
    """Use a fixed saved artifact across all ROIs; bound each probability matrix."""
    from spacemap.viewer import consensus_model
    from spacemap.ml_mr_utils import _valid_reference_mask
    from . import reference_models as models, tasks

    chosen, table = settings['config'], settings['table']
    snapshot = models._source(chosen['model_id'])
    model_id, meta, artifact = snapshot
    if meta['sklearn'] != consensus_model._version():
        raise ValueError('The model software version changed. Train a fresh reference model.')
    own = meta.get('table_path') == str(Path(table).resolve())
    present = models._present(meta, table)
    missing = [name for name in meta['markers'] if name not in present]
    if not present or (missing and not (meta.get('adaptable') and meta.get('allow_missing', True))):
        raise ValueError('The selected saved model is not compatible with this table.')
    report(f"Using {meta['method']} · {len(present)}/{len(meta['markers'])} saved features across the combined table…", .1)
    if missing:
        report('Preparing the available-feature model. Missing: ' + ', '.join(missing), .15)
        artifact, _ = models.adapt(model_id, present, _snapshot=snapshot)
    from copy import copy
    from spacemap.consensus_execution import cpu_budget
    predictor_model = copy(artifact['model'])
    if hasattr(predictor_model, 'n_jobs'):
        predictor_model.n_jobs = cpu_budget(chosen['cpu_workers'])['workers']
    artifact = {**artifact, 'model': predictor_model}
    if not own and chosen['match_scale']:
        maps = models.scale_maps(settings, model_id, present, _snapshot=snapshot)
        if maps:
            artifact = {**artifact, 'model': models.ScaleMatched(artifact['model'], maps)}
    # Batch predictions are applications of a trained model, not validation.
    # Manual labels, when retained, receive no invented confidence score.
    artifact = {**artifact, 'rows': np.array([-1]), 'oob': np.zeros((1, len(artifact['model'].classes_)))}
    predictor = models._scan_predictor(model_id, present, False, meta.get('adaptable') and meta.get('allow_missing', True), snapshot)
    reference = settings.get('reference_column')
    wanted = [settings['cell_id_col'], *present, *([reference] if reference else [])]
    if chosen['enable_filter']:
        wanted.append(chosen['filter_col'])
    frame = dataio.load(table, columns=list(dict.fromkeys(wanted)))
    anchor = frame[settings['cell_id_col']]
    if anchor.isna().any() or anchor.astype(str).duplicated().any():
        raise ValueError('Cell IDs must be present and unique.')
    count = len(frame)
    labels = np.full(count, None, object)
    confidence = np.full(count, np.nan)
    eligible = np.ones(count, bool)
    if chosen['enable_filter']:
        eligible = frame[chosen['filter_col']].astype('string').str.strip().eq(str(chosen['filter_val']).strip()).fillna(False).to_numpy(bool)
    if not eligible.any():
        raise ValueError('The row filter selected no cells. Check its column/value, or turn it off.')
    report(f'Predicting {int(eligible.sum()):,} cells in batches…', .25)
    for start in range(0, count, 65536):
        stop = min(count, start + 65536)
        rows = np.arange(start, stop)[eligible[start:stop]]
        if not len(rows):
            continue
        values = frame.iloc[rows][present].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
        probabilities = predictor(artifact, values, rows)
        valid = np.isfinite(probabilities).all(axis=1) & (probabilities.sum(axis=1) > 0)
        labels[rows[valid]] = artifact['model'].classes_[probabilities[valid].argmax(axis=1)]
        confidence[rows[valid]] = probabilities[valid].max(axis=1)
    retained = np.zeros(count, bool)
    if chosen['preserve_reference_labels'] and reference:
        retained = _valid_reference_mask(frame[reference], chosen['exclude_phenotypes']).to_numpy(bool) & eligible
        labels[retained] = frame.loc[retained, reference].astype(str).str.strip()
        confidence[retained] = np.nan
    filtered = labels.copy()
    filtered[np.isfinite(confidence) & (confidence < chosen['refined_confidence_threshold'])] = chosen['refined_low_confidence_label']
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / 'prediction_model.json').write_text(json.dumps({'model_id': model_id, 'trained': meta.get('trained'),
        'features': present, 'missing_features': missing, 'match_scale': chosen['match_scale']}, indent=2))
    result = tasks.save_result(settings, {settings['output_col']: labels, settings['confidence_column']: confidence,
                                        settings['refined_column']: filtered}, anchor, report)
    result.update(column=settings['refined_column'], prediction_column=settings['output_col'],
                  confidence_column=settings['confidence_column'], workflow='saved_model', model_id=model_id,
                  model_method=meta['method'], features=present, missing_features=missing,
                  scored_cells=int(np.isfinite(confidence).sum()), retained_reference_cells=int(retained.sum()),
                  unscored_cells=int((eligible & pd.isna(labels)).sum()))
    return result
