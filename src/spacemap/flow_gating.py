"""Cell-measurement gates. Geometry is saved in its authoring coordinates.

Membership is exact; display sampling never participates in evaluation. Boolean
rules use three-valued logic so missing measurements cannot become negatives.
"""
from copy import deepcopy
from collections import OrderedDict
import json

import numpy as np
import pandas as pd
from .flow_frame_cache import frame_cache


def axis_values(values, spec, inverse=False):
    values = np.asarray(values, dtype=float)
    mode = spec.get('scale', 'linear')
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        if mode == 'linear':
            result = values.copy()
        elif mode == 'arcsinh':
            c = float(spec.get('cofactor', 5))
            if not np.isfinite(c) or c <= 0:
                raise ValueError('Cofactor must be positive.')
            result = np.sinh(values) * c if inverse else np.arcsinh(values / c)
        elif mode == 'log10':
            result = np.power(10., values) if inverse else np.where(values > 0, np.log10(values), np.nan)
        elif mode == 'logicle':
            try:
                from flowutils import transforms
            except ImportError as exc:
                raise ValueError('Logicle requires the optional flowutils package. Choose Linear or Arcsinh.') from exc
            params = {k: float(spec.get(k, v)) for k, v in dict(t=262144, m=4.5, w=.5, a=0).items()}
            if not (params['t'] > 0 and params['m'] > 0 and 0 <= 2*params['w'] <= params['m'] and
                    -params['w'] <= params['a'] <= params['m'] - 2*params['w']):
                raise ValueError('Invalid Logicle T/M/W/A parameters.')
            result = values.copy()
            valid = np.isfinite(values)
            function = transforms.logicle_inverse if inverse else transforms.logicle
            result[valid] = function(values[valid], None, **params)
        else:
            raise ValueError('Unknown axis scale.')
    result[~np.isfinite(result)] = np.nan
    return result


def fit_axis(frame, spec, reference=None):
    """Fit once on a fixed reference, ddof=0 for both pooled and grouped SD."""
    if 'pipeline' in spec:
        from .flow_transforms import fit
        return fit(frame,spec,reference)
    spec = deepcopy(spec)
    if spec['column'] not in frame:
        raise ValueError('Choose an available measurement column.')
    values = axis_values(pd.to_numeric(frame[spec['column']], errors='coerce'), spec)
    mode = spec.get('normalize', 'none')
    if mode not in ('none', 'zscore', 'percentile'):
        raise ValueError('Choose no normalization, Z-score or percentile scaling.')
    if mode == 'none':
        return spec
    group = spec.get('group', '')
    if group and group not in frame:
        raise ValueError('The normalization grouping column is missing.')
    groups = frame[group].astype('string') if group else pd.Series(['all'] * len(frame))
    reference = np.ones(len(frame), bool) if reference is None else np.asarray(reference, bool)
    params = {}
    for key in groups.dropna().unique():
        x = values[reference & groups.eq(key).fillna(False).to_numpy(bool) & np.isfinite(values)]
        if not len(x):
            continue
        if mode == 'zscore':
            low, width = float(x.mean()), float(x.std(ddof=0))
        else:
            percentiles = spec.get('percentiles', [1, 99])
            if not (len(percentiles) == 2 and 0 <= percentiles[0] < percentiles[1] <= 100):
                raise ValueError('Percentiles must satisfy 0 ≤ low < high ≤ 100.')
            low, high = np.percentile(x, percentiles)
            low, width = float(low), float(high-low)
        params[str(key)] = {'offset': low, 'scale': width, 'n': len(x), 'constant': width == 0}
    spec['fitted'] = params
    return spec


def measurement(frame, spec):
    if 'pipeline' in spec:
        from .flow_transforms import transform
        return transform(frame,spec)
    if spec.get('column') not in frame:
        raise ValueError('Missing measurement: ' + str(spec.get('column')))
    x = axis_values(pd.to_numeric(frame[spec['column']], errors='coerce'), spec)
    if spec.get('normalize', 'none') == 'none':
        return x
    if 'fitted' not in spec:
        raise ValueError('Fit normalization on the analysis cells first.')
    groups = frame[spec['group']].astype('string') if spec.get('group') else pd.Series(['all'] * len(frame))
    out = np.full(len(x), np.nan)
    for key, param in spec['fitted'].items():
        keep = groups.eq(key).fillna(False).to_numpy(bool)
        out[keep] = (x[keep] - param['offset']) / (param['scale'] or 1.)
    if spec.get('normalize') == 'percentile' and spec.get('clip', False):
        out = np.clip(out, 0, 1)
    return out


def _categories(frame, column):
    # Studio frames are immutable and scoped to a source revision. Encode a
    # low-cardinality column once instead of converting every cell to text for
    # every sibling population. Ordinary caller-owned frames are not cached.
    cache = frame_cache(frame).setdefault('categories', OrderedDict()) if frame.attrs.get('flow_scope') else {}
    if column in cache:
        return cache[column]
    values=frame[column]
    if pd.api.types.is_numeric_dtype(values.dtype):
        # Encode numbers first, then format only distinct category values.
        # Binary annotation columns must not stringify millions of cells.
        codes, unique = pd.factorize(values)
        names = pd.Index(pd.Series(unique).astype('string'))
    else:
        codes, names = pd.factorize(values.astype('string[pyarrow]'))
    if isinstance(cache, OrderedDict) and len(names) <= 512:
        cache[column] = (codes, names)
        while len(cache) > 4:
            cache.popitem(last=False)
    return codes, names


def filter_rule(frame, rule):
    """Explicit null operators; AND/OR/NOT preserve unknowns."""
    size = len(frame)
    if not rule:
        return np.ones(size, bool), np.ones(size, bool)
    if 'rules' in rule:
        children = [filter_rule(frame, child) for child in rule['rules']]
        if not children:
            raise ValueError('Add a filter condition.')
        return combine(children, rule.get('join', 'and'))
    column = rule.get('column')
    if column not in frame:
        raise ValueError('Missing filter column: ' + str(column))
    op = rule.get('op', 'in')
    if op == 'blank':
        codes, names = _categories(frame,column)
        blank = np.array([not str(name).strip() for name in names] + [True])
        return blank[codes], np.ones(size, bool)
    if op in ('in', 'not in'):
        codes, names = _categories(frame,column)
        known = codes >= 0
        accepted = set(str(v) for v in rule.get('values', []))
        matches = np.array([str(name) in accepted for name in names] + [False])
        match = matches[codes]
        return (match if op == 'in' else ~match) & known, known
    series = frame[column]
    if op == 'nonfinite':
        return ~np.isfinite(pd.to_numeric(series, errors='coerce').to_numpy(dtype=float,na_value=np.nan)), np.ones(size, bool)
    if op in ('missing', 'present'):
        known = series.notna().to_numpy()
        return (~known if op == 'missing' else known), np.ones(size, bool)
    x = pd.to_numeric(series, errors='coerce').to_numpy(float)
    known = np.isfinite(x)
    funcs = {'>': np.greater, '>=': np.greater_equal, '<': np.less, '<=': np.less_equal, '==': np.equal}
    if op not in funcs or not np.isfinite(float(rule.get('value', np.nan))):
        raise ValueError('Choose a comparison and finite value.')
    return funcs[op](x, float(rule['value'])) & known, known


def combine(children, op):
    matches = np.stack([x[0] for x in children])
    known = np.stack([x[1] for x in children])
    if op == 'not' and len(children) == 1:
        return ~matches[0] & known[0], known[0]
    if op == 'and':
        yes = matches.all(axis=0)
        return yes, yes | (known & ~matches).any(axis=0)
    if op == 'or':
        yes = matches.any(axis=0)
        return yes, yes | known.all(axis=0)
    raise ValueError('Choose AND, OR, or NOT of one population.')


def polygon_mask(x, y, vertices):
    """Ray crossing, including the boundary (overlaps between arbitrary gates allowed)."""
    v = np.asarray(vertices, float)
    if v.ndim != 2 or v.shape[1] != 2 or not 3 <= len(v) <= 4096 or not np.isfinite(v).all():
        raise ValueError('A polygon needs 3–4096 finite vertices.')
    inside = np.zeros(len(x), bool)
    boundary = np.zeros(len(x), bool)
    for (x1, y1), (x2, y2) in zip(v, np.roll(v, -1, axis=0)):
        dx, dy = x2-x1, y2-y1
        cross = (x-x1)*dy - (y-y1)*dx
        tol = np.finfo(float).eps * 16 * max(1, abs(dx), abs(dy))
        boundary |= (np.abs(cross) <= tol) & (x >= min(x1,x2)) & (x <= max(x1,x2)) & (y >= min(y1,y2)) & (y <= max(y1,y2))
        if dy:
            inside ^= ((y1 > y) != (y2 > y)) & (x < dx*(y-y1)/dy+x1)
    return inside | boundary


def evaluate(frame, gates, root=None, *, reuse=None):
    if len(gates) > 256:
        raise ValueError('Use at most 256 populations per workspace.')
    root = np.ones(len(frame), bool) if root is None else np.asarray(root, bool)
    results = {'root': (root, root)}
    results.update(reuse or {})
    by_id = {g['id']: g for g in gates}
    if len(by_id) != len(gates) or 'root' in by_id:
        raise ValueError('Population IDs must be unique.')
    visiting, transformed = set(), {}

    def axis(spec):
        key = json.dumps(spec, sort_keys=True)
        if key not in transformed:
            transformed[key] = measurement(frame, spec)
        return transformed[key]

    def visit(identifier):
        if identifier in results:
            return results[identifier]
        if identifier in visiting or identifier not in by_id:
            raise ValueError('Population dependencies contain a cycle or missing parent.')
        visiting.add(identifier)
        gate = by_id[identifier]
        parent, _ = visit(gate.get('parent', 'root'))
        kind = gate.get('kind')
        if kind == 'filter':
            match, known = filter_rule(frame, gate.get('rule'))
        elif kind == 'boolean':
            refs = gate.get('refs', [])
            if not refs:
                raise ValueError('Choose populations for the Boolean rule.')
            match, known = combine([visit(r) for r in refs], gate.get('op'))
        else:
            x = axis(gate['x'])
            known = np.isfinite(x)
            bounds = gate.get('bounds', [])
            if not np.isfinite(np.asarray(bounds, float)).all():
                raise ValueError('Gate bounds must be finite.')
            if kind == 'range':
                if len(bounds) != 2 or bounds[0] > bounds[1]:
                    raise ValueError('Enter increasing range bounds.')
                match = (x >= bounds[0]) & (x <= bounds[1])
            elif kind == 'threshold':
                if len(bounds) != 1:
                    raise ValueError('Enter one threshold.')
                match = x >= bounds[0] if gate.get('above', True) else x < bounds[0]
            else:
                y = axis(gate['y'])
                known &= np.isfinite(y)
                if kind == 'rectangle' and len(bounds) == 4:
                    if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
                        raise ValueError('Enter increasing bounds.')
                    match = (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
                elif kind == 'polygon':
                    match = polygon_mask(x, y, gate['vertices'])
                elif kind == 'quadrant' and len(bounds) == 2:
                    match = (x >= bounds[0] if gate.get('right') else x < bounds[0]) & (y >= bounds[1] if gate.get('above') else y < bounds[1])
                else:
                    raise ValueError('Choose a supported gate type.')
        if not gate.get('enabled', True):
            match, known = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
        if kind != 'filter' and gate.get('rule'):
            match,known=combine([(match,known),filter_rule(frame,gate['rule'])],'and')
        results[identifier] = (match & known & parent, known & parent)
        visiting.remove(identifier)
        return results[identifier]
    for identifier in by_id:
        visit(identifier)
    return results


def distributions(frame, results, spec, live=None):
    """One exact shared-bin computation for overlay, ridgeline and waterfall."""
    x = measurement(frame, spec['x'])
    parent = results.get(spec.get('parent', 'root'), results['root'])[0]
    finite = parent & np.isfinite(x)
    limits = spec.get('x_range')
    if not limits:
        limits = [float(x[finite].min()), float(x[finite].max())] if finite.any() else [0., 1.]
    if limits[0] == limits[1]:
        limits = [limits[0]-.5, limits[1]+.5]
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError('Enter increasing finite axis limits.')
    bins = int(spec.get('bins', 80))
    if not 4 <= bins <= 512:
        raise ValueError('Use 4–512 histogram bins.')
    edges = np.linspace(*limits, bins+1)
    series = []
    threshold = spec.get('threshold')
    cutoff = float(threshold['bounds'][0]) if threshold else None
    if cutoff is not None and not np.isfinite(cutoff):
        raise ValueError('Threshold must be finite.')
    for item in spec.get('series') or [{'population': spec.get('parent', 'root')}]:
        population = item.get('population', 'root')
        mask = results[population][0].copy()
        mask &= filter_rule(frame,spec.get('rule'))[0]
        mask &= filter_rule(frame, item.get('rule'))[0]
        if live is not None and not item.get('pinned', False):
            mask &= live
        valid = mask & np.isfinite(x)
        counts = np.histogram(x[valid], edges)[0]
        values = counts.astype(float)
        mode = spec.get('frequency', 'percent')
        denominator = int(valid.sum())
        group = spec.get('patient', '')
        patients = None
        if group:
            groups = frame[group].astype('string')
            curves = []
            for key in groups[valid].dropna().unique():
                selected = valid & groups.eq(key).fillna(False).to_numpy(bool)
                n = int(selected.sum())
                if n >= int(spec.get('min_patient_cells', 1)):
                    curves.append(np.histogram(x[selected], edges)[0] / n)
            patients = len(curves)
            values = np.mean(curves, axis=0) if curves else np.zeros(bins)
            denominator = 1
            if mode == 'count':
                raise ValueError('Equal-patient curves require proportion, percent, density or peak normalization.')
        if mode == 'percent':
            values = values / max(1, denominator) * 100
        elif mode == 'proportion':
            values = values / max(1, denominator)
        elif mode == 'density':
            values = values / max(1, denominator) / np.diff(edges)
        elif mode == 'peak':
            values = values / max(float(values.max()), 1e-300)
        elif mode != 'count':
            raise ValueError('Choose a histogram frequency mode.')
        series.append({**item, 'counts': counts.tolist(), 'values': values.tolist(), 'n': int(mask.sum()),
            'above_threshold': int((valid & (x > cutoff)).sum()) if cutoff is not None else None,
            'above_percent': 100 * int((valid & (x > cutoff)).sum()) / int(mask.sum()) if cutoff is not None and mask.any() else (0. if cutoff is not None else None),
            'measured': int(valid.sum()), 'invalid': int((mask & ~np.isfinite(x)).sum()),
            'outside': int(valid.sum()-counts.sum()), 'patients': patients,
            'quantiles': np.percentile(x[valid], [25,50,75]).tolist() if valid.any() else None})
    return {'edges': edges.tolist(), 'series': series, 'threshold': threshold, 'frequency': spec.get('frequency', 'percent')}


def reproject_boundary(gate, x_display, y_display=None, samples=128):
    """Sample edges BEFORE nonlinear projection; evaluation always uses saved geometry."""
    if gate.get('kind') not in ('rectangle', 'polygon'):
        return []
    if any(s.get('normalize', 'none') != 'none' or 'pipeline' in s for s in [gate['x'],gate['y'],x_display,y_display or {}]):
        if gate['x'] != x_display or gate['y'] != y_display:
            return []  # group-specific transforms have no single global boundary
        convert = False
    else:
        convert = True
    b = gate.get('bounds', [])
    vertices = gate.get('vertices') if gate['kind'] == 'polygon' else [[b[0],b[2]],[b[1],b[2]],[b[1],b[3]],[b[0],b[3]]]
    v = np.asarray(vertices)
    points = np.concatenate([np.linspace(a,b,samples) for a,b in zip(v,np.roll(v,-1,axis=0))])
    points = np.vstack([points, points[0]])
    if convert:
        for i, original, target in [(0,gate['x'],x_display),(1,gate['y'],y_display)]:
            points[:,i] = axis_values(axis_values(points[:,i], original, inverse=True), target)
    return [[float(a) if np.isfinite(a) else None, float(b) if np.isfinite(b) else None] for a,b in points]
