"""Read-only area picks in the viewer's exact current cell/table binding."""
from types import SimpleNamespace

import numpy as np
from shapely import intersects_xy

from spacemap.viewer import cells
from . import annotation_components, selection
from .cell_links import table_layer


def cells_in_area(binding, body):
    from spacemap_gui.viewer.api import _density_params
    params = {str(k): str(v) for k, v in body.get('view', {}).items()}
    params['viewer'] = '1'
    bound = table_layer(binding, params.get('t', ''))
    table, options = _density_params(SimpleNamespace(query_params=params), bound)
    xy = cells.centroid_columns(table)
    if not xy:
        raise ValueError('This cell table needs linked X and Y centroid columns for area selection.')
    keys = {k: options[k] for k in ('roi_column', 'label_column')}
    ids, x = cells.roi_values(table, bound['roi'], xy[0], **keys)
    other, y = cells.roi_values(table, bound['roi'], xy[1], **keys)
    if not np.array_equal(ids, other):
        raise ValueError('Cell data changed. Draw the selection again.')
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y) & (ids > 0) & np.isin(ids, options['allowed_labels'])
    inside = np.zeros(len(ids), bool)
    if 'only_ids' in body:
        inside[keep] = np.isin(ids[keep], body['only_ids'])
    else:
        area = annotation_components.selection_area(body.get('points'))
        inside[keep] = intersects_xy(area, x[keep], y[keep])
    # Shift adds previous picks only from the same binding and still-visible cells.
    wanted = ids[inside]
    if body.get('add'):
        wanted = np.union1d(wanted, ids[keep & np.isin(ids, body['add'])])
    wanted = np.unique(wanted)
    settings = {'table': table, 'markers': []}
    token = selection.remember({'channel': None, 'settings': settings, 'revision': selection._revision(settings),
        'groups': [{'roi': bound['roi'], 'ids': wanted}]})
    return {'ids': wanted.tolist(), 'selection': token, 'count': len(wanted)}
