"""Review evidence for subclusters: manual counts and saved-model votes stay separate."""
import numpy as np
import pandas as pd

from spacemap import dataio
from spacemap.viewer import consensus_model
from . import reference_models, mr_ml

SAMPLE = 400


def build(table, groups, chosen, report):
    column = chosen.get('reference_column', '')
    model_id = chosen.get('model_id', '')
    names = pd.Series(groups.to_numpy(), dtype='string')
    manual = pd.Series('', index=names.index, dtype='string')
    if column:
        manual = dataio.load(table, columns=[column])[column].reset_index(drop=True).astype('string').fillna('').str.strip()
        manual = manual.mask(manual.str.lower().isin(consensus_model._EMPTY), '')
    selected = names.notna()
    totals = manual[selected & manual.ne('')].value_counts()
    allowed = set(chosen['model_phenotypes']) if model_id and chosen.get('limit_model_phenotypes') else None
    artifact = snapshot = None
    support, present, missing, assessed = {}, [], [], None
    if model_id:
        snapshot = reference_models._source(model_id)
        identifier, meta, artifact = snapshot
        if meta['sklearn'] != consensus_model._version():
            raise ValueError('This model needs its original software version. Choose another model or No model.')
        present = reference_models._present(meta, table)
        if 'model_features' in chosen:
            present = [name for name in present if name in chosen['model_features']]
        assessed = mr_ml.panel_assessment(snapshot, present)
        support = {p['phenotype']: p['suggested'] for p in assessed['phenotypes']}
        missing = assessed['missing']
        if missing:
            artifact = reference_models.adapt(identifier, present, _snapshot=snapshot)[0]
        # These are descriptive cluster votes, never presented as independent
        # validation or model accuracy; keep validation evidence in its own field.
        artifact = {**artifact, 'rows': np.array([-1]), 'oob': np.zeros((1, len(artifact['model'].classes_)))}
        predictor = reference_models._scan_predictor(identifier, present, False,
            bool(meta.get('adaptable') and meta.get('allow_missing', True)), snapshot)
    rng = np.random.default_rng(42)
    group_rows = {name: np.asarray(rows) for name, rows in names.groupby(names, sort=False).groups.items()}
    samples = {name: np.sort(rng.choice(rows, min(len(rows), SAMPLE), replace=False)) for name, rows in group_rows.items()} if artifact else {}
    measured_rows = np.unique(np.concatenate(list(samples.values()))) if samples else np.array([], dtype=int)
    measured = None
    if artifact:
        report('Checking clusters against the saved model…', None)
        # One projected read for all bounded samples; do not rescan a large CSV
        # separately for every cluster. Prediction matrices stay per-cluster.
        measured = consensus_model.selected_values({'table': table, 'markers': present,
            'label_column': dataio.columns(table, include_pending=False)[0]}, measured_rows)
    found = {}
    for name, rows in group_rows.items():
        here = manual.iloc[rows]
        counts = here[here.ne('')].value_counts()
        top = [{'label': str(label), 'cells': int(n), 'of_label': float(n / totals[label]),
                'of_cluster': float(n / counts.sum())} for label, n in counts.items()]
        suggestions, scored, sampled, excluded, model_votes = [], 0, 0, None, {}
        if artifact:
            picked = samples[name]
            sampled = len(picked)
            values = measured[np.searchsorted(measured_rows, picked)]
            probabilities = predictor(artifact, values, picked)
            valid = np.isfinite(probabilities).all(axis=1) & (probabilities.sum(axis=1) > 0)
            scored = int(valid.sum())
            if scored:
                votes = probabilities[valid].mean(axis=0)
                classes = artifact['model'].classes_
                model_votes = {str(c): float(v) for c, v in zip(classes, votes)}
                winner = str(classes[int(votes.argmax())])
                if allowed is not None and winner not in allowed:
                    excluded = winner
                else:
                    for i in np.argsort(-votes, kind='stable'):
                        label = str(classes[i])
                        if (allowed is None or label in allowed) and votes[i] > .05:
                            suggestions.append({'label': label, 'score': float(votes[i]), 'by_markers': float(votes[i]),
                                'reference_cells': int(counts.get(label, 0)),
                                'of_label': float(counts.get(label, 0) / totals[label]) if label in totals else 0.,
                                'needs_checking': not support.get(label, False) or not np.isfinite(values).all(), 'source': 'model'})
                        if len(suggestions) == 3:
                            break
        reference_suggestions = [{'label': item['label'], 'score': item['of_cluster'], 'by_markers': None,
                                  'reference_cells': item['cells'], 'of_label': item['of_label'], 'source': 'reference'}
                                 for item in top if allowed is None or item['label'] in allowed][:3]
        found[str(name)] = {'labelled': int(counts.sum()), 'top': top[:3],
                           'suggestions': suggestions if model_id else reference_suggestions,
                           'reference_suggestions': reference_suggestions, 'model_scored': scored,
                           'model_sampled': sampled, 'excluded_winner': excluded, 'model_votes': model_votes,
                           'reference_counts': {str(c): int(n) for c, n in counts.items()}}
    return {'model_id': model_id, 'reference_column': column, 'clusters': found,
            'labels': [{'label': str(c), 'cells': int(n)} for c, n in totals.items()],
            'labelled': int(totals.sum()), 'by_markers': bool(model_id), 'features': present, 'missing': missing,
            'allowed': sorted(allowed) if allowed is not None else None, 'validation': assessed,
            'note': 'Model votes describe sampled cells. Review marker profiles and manual labels before naming.'}
