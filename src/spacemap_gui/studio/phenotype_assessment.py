"""Classic F1 assessments in Studio, using explicit reusable cell subsets."""
from collections import OrderedDict
import io
import json
from pathlib import Path
import threading
import uuid
import zipfile

import numpy as np
import pandas as pd

from . import data_view, phenotyping_inputs as inputs, reference_splits

RESULTS = OrderedDict()
LOCK = threading.RLock()
PLOT_LOCK = threading.RLock()
PARTS = {'validation': 'Validation cells', 'training': 'Training cells',
         'both': 'Training + validation', 'all': 'All labelled cells'}
PLOTS = {'comparison': 'F1 comparison', 'per_class': 'F1 by phenotype',
         'precision_recall': 'Precision, recall and F1', 'confusion': 'Confusion matrix',
         'by_image': 'F1 by image'}


def context(channel):
    from spacemap_gui.viewer import registry
    project = registry.session_project(channel) or {}
    root = project['root']
    active = inputs.combined_table(channel)
    if inputs.column_roles(root, active)['roi']:
        tables = {str(Path(active).resolve()): None}
    else:
        tables = {str(Path(path).resolve()): image for image, path in inputs.image_tables(root, active).items() if path}
        tables.setdefault(str(Path(active).resolve()), inputs.single_image(root, active) or Path(active).stem)
    return root, tables


def options(channel):
    from spacemap_gui.tabs.tab9_visualization import f1_validation_utils as classic
    root, tables = context(channel)
    columns = set()
    for path in tables:
        table = data_view._table(path)
        identities = set(inputs.column_roles(root, path).values()) | {'label', 'cell_id'}
        columns.update(name for name in table.names if table.kinds[name] == 'group' and name not in identities)
    columns = sorted(columns)
    return {'columns': columns, 'references': classic.reference_candidates(columns),
            'methods': classic.prediction_candidates(columns), 'defaults': classic.default_method_columns(columns),
            'splits': reference_splits.options(root, tables), 'plots': PLOTS,
            'tables': [{'path': path, 'image': image or Path(path).name} for path, image in tables.items()]}


def _frame(channel, body):
    from .data_editor import stamp
    root, tables = context(channel)
    column, methods = body.get('reference'), body.get('methods')
    if (not isinstance(column, str) or not isinstance(methods, list) or not methods
            or len(methods) > 30 or any(not isinstance(m, str) for m in methods)):
        raise ValueError('Choose a reference column and one or more methods (up to 30).')
    methods = list(dict.fromkeys(methods))
    if column in methods:
        raise ValueError('Choose prediction columns different from the reference.')
    subset = body.get('subset', 'all')
    if subset not in PARTS:
        raise ValueError('Choose Validation, Training, Both or All labelled cells.')
    split = reference_splits.read(root, body.get('split_id', ''), tables) if subset != 'all' else None
    if split and split['reference'] != column:
        raise ValueError('Choose the reference column used for this saved split.')
    selected = body.get('tables', list(tables))
    if not isinstance(selected, list) or not selected or any(path not in tables for path in selected):
        raise ValueError('Choose tables paired with this project’s images.')
    selected = list(dict.fromkeys(selected))
    parts, members, missing = [], [], []
    matched_methods = set()
    for path in selected:
        before = stamp(path)
        table = data_view._table(path)
        if column not in table.names:
            continue
        if table.kinds[column] != 'group':
            if pd.isna(np.asarray(table.column(column))).all():
                continue
            raise ValueError('Use phenotype names as the reference, not cell numbers or measurements.')
        if split:
            entry = next((entry for entry in split['tables'] if entry['table'] == path), None)
            if entry is None:
                continue
            rows = reference_splits.selected_rows(entry, subset, column)
            membership = dict(zip(entry['rows'], entry['subset']))
        else:
            truth = pd.Series(np.asarray(table.column(column)), dtype='string').str.strip()
            rows = np.flatnonzero((truth.notna() & truth.ne('')).fillna(False).to_numpy())
            membership = {}
        if not len(rows):
            continue
        roles = inputs.column_roles(root, path)
        present = [m for m in methods if m in table.names]
        matched_methods.update(present)
        names = list(dict.fromkeys([column, *present, *([roles['roi']] if roles['roi'] else [])]))
        frame = pd.DataFrame(table.take(rows, names))
        numeric = [m for m in present if table.kinds[m] != 'group' and frame[m].notna().any()]
        if numeric:
            raise ValueError('Name the phenotypes before comparing: ' + ', '.join(numeric))
        for method in set(methods) - set(present):
            frame[method] = None
            missing.append(f'{Path(path).name}: {method}')
        frame['__assessment_image'] = frame[roles['roi']].astype(str) if roles['roi'] else tables[path]
        if stamp(path) != before:
            raise ValueError('Labels or predictions changed during comparison. Compare again.')
        parts.append(frame[[column, *methods, '__assessment_image']])
        members.extend({'table': Path(path).name, 'image': image, 'row': int(row),
                        'subset': membership.get(int(row), 'all'), 'reference': str(truth)}
                       for row, image, truth in zip(rows, frame['__assessment_image'], frame[column]))
    absent = set(methods) - matched_methods
    if absent:
        raise ValueError('Prediction columns not found: ' + ', '.join(sorted(absent)))
    if not parts:
        raise ValueError('No labelled cells in this selection. Choose another subset or label cells first.')
    frame = pd.concat(parts, ignore_index=True)
    if frame[column].nunique() > 200:
        raise ValueError('Choose a phenotype column with no more than 200 categories.')
    return root, frame, pd.DataFrame(members), methods, subset, split, missing


def compare(channel, body):
    from spacemap_gui.tabs.tab9_visualization import f1_validation_utils as classic
    root, frame, members, methods, subset, split, missing = _frame(channel, body)
    reference = body['reference']
    excluded = body.get('exclude_labels', [])
    if not isinstance(excluded, list) or any(not isinstance(v, str) for v in excluded):
        raise ValueError('Enter phenotype names to exclude.')
    bundle = classic.analyze_validation(frame, reference_columns=[reference], method_columns=methods,
        excluded_labels=excluded, group_column='__assessment_image', prediction_design=PARTS[subset])
    if bundle['summary'].empty:
        raise ValueError('No reference cells remain after excluding labels.')
    keep = classic.valid_truth_mask(frame[reference], excluded).to_numpy()
    members = members.loc[keep].reset_index(drop=True)
    note = 'Independent validation only when the assessed method did not use these cells or their labels.'
    notices = (['Missing columns count as unassigned predictions: ' + '; '.join(missing)] if missing else [])
    identifier = uuid.uuid4().hex
    record = {'root': str(Path(root).resolve()), 'bundle': bundle, 'membership': members,
              'reference': reference, 'subset': subset, 'split_id': split['id'] if split else None,
              'cells': len(members), 'note': note, 'notices': notices}
    with LOCK:
        RESULTS[identifier] = record
        while len(RESULTS) > 8:
            RESULTS.popitem(last=False)
    pooled = bundle['summary'].query("group == 'Pooled' and route == 'End-to-end'")
    return {'id': identifier, 'reference': reference, 'subset': PARTS[subset], 'cells': len(members),
            'summary': json.loads(pooled.to_json(orient='records')),
            'per_class': json.loads(bundle['per_class'].query("group == 'Pooled'").to_json(orient='records')),
            'methods': bundle['metadata']['method_display_names'], 'note': note, 'notices': notices, 'plots': PLOTS}


def result(root, identifier):
    with LOCK:
        record = RESULTS.get(identifier)
        if not record or record['root'] != str(Path(root).resolve()):
            raise ValueError('Compare again to refresh these results.')
        RESULTS.move_to_end(identifier)
        return record


def figure(root, identifier, kind='comparison', method=None, options=None):
    from spacemap_gui.tabs.tab9_visualization import f1_validation_utils as classic
    from . import figures
    record = result(root, identifier)
    fmt, dpi, options = figures._options(options)
    if kind not in PLOTS:
        raise ValueError('Choose a listed assessment figure.')
    bundle = record['bundle']
    reference = bundle['metadata']['reference_names'][record['reference']]
    title = str(options.get('title') or f"{PARTS[record['subset']]} · {record['cells']:,} cells")[:200]
    style = classic.ValidationStyle(heatmap_vmin=0, title_prefix=title, dpi=dpi,
                                    figure_width=8, figure_height=4.5)
    common = {'reference': reference, 'route': 'End-to-end', 'group': 'Pooled', 'style': style}
    with PLOT_LOCK:
        if kind == 'comparison':
            plot = classic.plot_f1_metric_comparison(bundle, **common)
        elif kind == 'per_class':
            plot = classic.plot_per_class_heatmap(bundle, **common)
        elif kind == 'precision_recall':
            plot = classic.plot_precision_recall_f1_heatmaps(bundle, **common)
        elif kind == 'by_image':
            plot = classic.plot_grouped_macro_f1(bundle, reference=reference, style=style)
        else:
            names = bundle['metadata']['method_display_names']
            if method not in names:
                raise ValueError('Choose a compared method for its confusion matrix.')
            plot = classic.plot_normalized_confusion(bundle, method=names[method], **common)
        plot.text(.01, -.015, record['note'], fontsize=8)
        return figures._save(plot, fmt, dpi), figures.FORMATS[fmt]


def export(root, identifier):
    record = result(root, identifier)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for key in ('summary', 'per_class', 'confusion', 'reference_audit'):
            archive.writestr(key + '.csv', record['bundle'][key].to_csv(index=False))
        archive.writestr('cells.csv', record['membership'].to_csv(index=False))
        archive.writestr('assessment.json', json.dumps({k: v for k, v in record.items()
            if k not in {'bundle', 'membership', 'root'}}, indent=2))
    return buffer.getvalue()
