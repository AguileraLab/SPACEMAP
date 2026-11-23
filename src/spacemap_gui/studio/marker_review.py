"""Saved review evidence and explicit, revision-checked links to cells/settings."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from spacemap import dataio
from spacemap.marker_preprocessing import transform
from . import marker_figures, subphenotypes, tasks


def saved(root, job_id, phenotype):
    result = marker_figures.saved_result(root, job_id)
    if not any(e['kind'] == 'review' and e['phenotype'] == phenotype for e in result['figures']):
        raise ValueError('Generate Marker review for this phenotype first.')
    return result


def evidence(root, body):
    result = saved(root, body.get('job_id'), body.get('phenotype'))
    p = body['phenotype']
    bundle = {k: pd.read_parquet(Path(result['output_dir']) / (k + '.parquet')) for k in marker_figures.TABLES}
    bundle['metadata'] = result['metadata']
    stored = json.loads((Path(result['output_dir']) / 'review.json').read_text())
    def rows(key, phenotype=True):
        frame = bundle[key]
        if phenotype:
            frame = frame[frame.phenotype.eq(p)]
        return json.loads(frame.to_json(orient='records'))
    return dict(rows=stored['phenotypes'][p], histograms=rows('histograms'), samples=rows('patients'),
                context=rows('summary', False), pairs=rows('coexpression'), rule=stored['rule'],
                grouping=result['metadata']['patient_column'], table=result['table'])


def current(root, table, body):
    result = saved(root, body.get('job_id'), body.get('phenotype'))
    meta = result['metadata']
    if Path(result['table']).resolve() != Path(table).resolve():
        raise ValueError('This review belongs to another cell table.')
    if tasks.source_revision({'table': str(table)}) != meta['source_revision']:
        raise ValueError('The cells or saved labels changed. Generate Marker review again.')
    return result


def apply(root, table, body):
    from .settings import revision, project_lock
    with project_lock(root):
        result = current(root, table, body)
        config = subphenotypes.values(root)
        meta = result['metadata']
        scope = subphenotypes.image_scope(root, table)
        if (body.get('revision') != revision(config) or
                any(config[k] != v for k, v in meta['selection'].items()) or
                any(config[k] != meta[k] for k in ('transform', 'cofactor')) or
                set(scope.get('selected_images', [])) != set(meta.get('selected_images') or [])):
            raise ValueError('The selection or settings changed. Generate Marker review again.')
        markers = body.get('markers')
        if (not isinstance(markers, list) or not markers or
                any(not isinstance(m, str) or m not in meta['markers'] for m in markers)):
            raise ValueError('Choose at least one reviewed marker.')
        choices = {**config['chosen_markers'], body['phenotype']: list(dict.fromkeys(markers))}
        subphenotypes.save(root, {'chosen_markers': choices}, revision(config))
        return {'chosen_markers': choices}


def select(root, channel, table, body):
    from . import data_editor, flow_gating, phenotyping_inputs
    from .phenotyping_images import mask as image_mask
    result = current(root, table, body)
    meta, config = result['metadata'], result['metadata']['selection']
    marker = body.get('marker')
    if marker not in meta['markers']:
        raise ValueError('Choose a reviewed marker.')
    bounds = [body.get('lo'), body.get('hi')]
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)) for v in bounds):
        raise ValueError('Enter finite analysis values, or leave the range blank.')
    lo, hi = bounds
    if lo is not None and hi is not None and lo > hi:
        raise ValueError('The range start must not exceed its end.')
    label = phenotyping_inputs.segmentation_column(root, table)
    if not label or label not in dataio.columns(table):
        raise ValueError('Set the segmentation label in Column roles to link cells to the image.')
    roi = meta.get('image_column')
    columns = list(dict.fromkeys([label, marker, config['parent_col'],
        *(f['column'] for f in config['filters']), *([roi] if roi else [])]))
    stamp = data_editor.stamp(table)
    frame = dataio.load(table, columns=columns)
    settings = {'roi_column': roi, 'selected_images': meta.get('selected_images')}
    keep = image_mask(frame, settings) & subphenotypes.region_mask(frame, config['filters'])
    keep &= frame[config['parent_col']].astype('string').eq(body['phenotype']).fillna(False).to_numpy(bool)
    x = transform(frame, [marker], meta['transform'], meta['cofactor'])[marker].to_numpy()
    keep &= np.isfinite(x)
    if lo is not None:
        keep &= x >= lo
    if hi is not None:
        keep &= x <= hi
    # Store only selected row identities. Data and image tiles consume this same membership.
    ids = pd.to_numeric(frame.loc[keep, label], errors='coerce').to_numpy(float)
    if not np.all(np.isfinite(ids) & (ids > 0) & (ids == np.floor(ids)) & (ids <= 20000000)):
        raise ValueError('Cell labels must be positive segmentation IDs. Check Column roles.')
    rois = (frame.loc[keep, roi].astype('string').fillna('') if roi else
            pd.Series((meta.get('selected_images') or [''])[0], index=frame.index[keep]))
    selected = pd.DataFrame({'_flow_table': str(table), '_flow_row': np.flatnonzero(keep),
        '_flow_id': ids.astype(np.int64), '_flow_roi': rois.to_numpy()})
    if selected.duplicated(['_flow_roi', '_flow_id']).any():
        raise ValueError('Cell IDs are repeated within an image. Check Column roles before linking.')
    current(root, table, body)
    if data_editor.stamp(table) != stamp:
        raise ValueError('The table changed while selecting. Generate Marker review again.')
    summary = flow_gating.register_selection(channel, str(table), stamp, selected,
        f"{body['phenotype']} · {marker}")
    return {'selection': summary, 'table': str(table)}
