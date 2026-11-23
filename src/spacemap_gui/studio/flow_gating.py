"""Revisioned Flow gating workspaces shared by Studio and its detached window."""
from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np
import pandas as pd

from spacemap import flow_gating as engine
from spacemap import labels
from spacemap.viewer.sources import fingerprint
from . import data_editor, data_view, flow_cache, phenotyping_images, phenotyping_inputs
from .settings import project_lock

_CACHE = OrderedDict()
_TOKENS = OrderedDict()
_LOCK = threading.RLock()
_COMPUTE = threading.BoundedSemaphore(2)
PALETTE = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc949','#af7aa1','#ff9da7','#9c755f','#bab0ab']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def gate_meaning(gates):
    return [{k:v for k,v in gate.items() if k not in ('name','colour','label','label_position')} for gate in gates]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, allow_nan=False, indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def context(channel):
    from spacemap_gui.viewer import registry
    project = registry.session_project(channel)
    if not project:
        raise ValueError('Open a Studio project first.')
    table = phenotyping_inputs.combined_table(channel)
    folder = Path(project['root']) / 'outputs/studio/flow_gating' / digest(str(Path(table).resolve()))[:20]
    return project, table, folder


def sources(channel):
    project, table, _ = context(channel)
    result = phenotyping_images.sources(project['root'], table)
    for source in result:
        roles = phenotyping_inputs.column_roles(project['root'], source['table'])
        source.update(label_column=phenotyping_inputs.segmentation_column(project['root'], source['table']),
                      cell_id_column=roles['cell_id'])
    return result


def load(channel):
    project, table, folder = context(channel)
    pointer = folder / 'workspace.json'
    identifier=digest([str(Path(project['root']).resolve()),str(Path(table).resolve())])
    if pointer.exists():
        workspace=json.loads(pointer.read_text())
        workspace['id']=identifier
        workspace.setdefault('root_colour','#a8b4c4')
        workspace.setdefault('thresholds',[])
        workspace.setdefault('stats_design',{})
        return workspace
    return {'version': 1, 'id':identifier, 'revision': 0, 'table': table, 'project': Path(project['root']).name,
            'sources': sources(channel), 'gates': [], 'plots': [], 'layouts': [], 'root_colour':'#a8b4c4',
            'output_column': 'phenotype_flow', 'output_dir': str(folder / 'figures'),
            'active': 'root', 'overlays': [], 'live': {'enabled': False, 'source': 'cells'},
            'thresholds': [], 'stats_design': {}, 'recipes': [], 'history': [], 'future': [], 'applied': None}


def required(workspace, extra=()):
    names = set(extra)
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('column', 'group', 'patient') and isinstance(item, str) and item:
                    names.add(item)
                elif key == 'markers' and isinstance(item,list):
                    names.update(item)
                elif key not in ('fitted', 'snapshot'):
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(workspace['gates'])
    return sorted(names)


def dataset(workspace, extra=()):
    """Projected columns; cache immutable data revisions, shared by both windows."""
    columns = required(workspace, extra)
    for source in workspace['sources']:
        with data_editor.locked(source['table']) as editor:
            if editor.cursor:
                raise ValueError('Save the Data edits before gating this table.')
    stamps = [(s['table'], data_editor.stamp(s['table'])) for s in workspace['sources']]
    scope = digest([workspace['sources'], stamps])
    key = digest([workspace['sources'], stamps, columns])
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        # A plot often already loaded the axes needed by its new gate. Deleting
        # a gate needs fewer columns, not a second copy of every cell identity.
        for existing_key, item in reversed(_CACHE.items()):
            frame, _ = item
            if frame.attrs.get('flow_scope') == scope and set(columns).issubset(frame.columns):
                _CACHE.move_to_end(existing_key)
                return item
    parts = []
    for source in workspace['sources']:
        table = data_view._table(source['table'])
        needed = list(dict.fromkeys([*columns, source['label_column'], *([source['roi_column']] if source['roi_column'] else [])]))
        missing = set(needed)-set(table.names)
        if missing:
            raise ValueError('Missing columns in ' + Path(source['table']).name + ': ' + ', '.join(sorted(missing)))
        # Keep staged Data edits explicit rather than analysing stale values.
        with data_editor.locked(source['table']) as editor:
            if editor.cursor:
                raise ValueError('Save the Data edits before gating this table.')
        projected=table.project(needed)
        # Keep text in Arrow buffers instead of constructing millions of Python
        # strings, only to factorize the same category columns immediately.
        for name in needed:
            if table.kinds[name]=='group':
                projected[name]=pd.arrays.ArrowStringArray(projected[name])
        frame = pd.DataFrame(projected,copy=False)
        keep = phenotyping_images.mask(frame, source)
        frame['_flow_row'] = np.arange(len(frame), dtype=np.int64)
        if not keep.all():
            frame = frame.loc[keep].copy()
        frame['_flow_table'] = source['table']
        frame['_flow_roi'] = frame[source['roi_column']].astype('string[pyarrow]') if source['roi_column'] else source['selected_images'][0]
        ids = pd.to_numeric(frame[source['label_column']], errors='coerce')
        if not np.isfinite(ids).all() or (ids < 1).any() or (ids != np.floor(ids)).any() or frame['_flow_roi'].isna().any():
            raise ValueError('Choose valid image and segmentation-label roles in Phenotyping.')
        frame['_flow_id'] = ids.astype(np.int64)
        if frame.duplicated(['_flow_roi','_flow_id']).any():
            raise ValueError('Cell labels must be unique within each image. Check Column roles.')
        parts.append(frame)
    frame = parts[0] if len(parts)==1 else pd.concat(parts, ignore_index=True)
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        frame = frame.reset_index(drop=True)
    frame.attrs['flow_scope'] = scope
    if stamps != [(s['table'], data_editor.stamp(s['table'])) for s in workspace['sources']]:
        raise ValueError('Measurements changed while loading. Refresh the plots.')
    item = (frame, digest(stamps))
    with _LOCK:
        _CACHE[key] = item
        while len(_CACHE) > 3:
            _CACHE.popitem(last=False)
    return item


def evaluate(workspace, extra=()):
    frame, revision = dataset(workspace, extra)
    key = digest([revision, gate_meaning(workspace['gates'])])
    # Attached to the cached frame, never exposed in serialized responses.
    cached = engine.frame_cache(frame).get('evaluation')
    if cached and cached[0] == key:
        return frame, revision, cached[1]
    # Editing a gate invalidates only its descendants and Boolean dependants.
    definitions={g['id']:g for g in gate_meaning(workspace['gates'])}
    signatures={'root':revision};visiting=set()
    def signature(identifier):
        if identifier in signatures:
            return signatures[identifier]
        if identifier in visiting or identifier not in definitions:
            raise ValueError('Population dependencies contain a cycle or missing parent.')
        visiting.add(identifier)
        gate=definitions[identifier]
        dependencies=[gate.get('parent','root'),*gate.get('refs',[])]
        signatures[identifier]=digest([gate,[signature(d) for d in dependencies]])
        visiting.remove(identifier)
        return signatures[identifier]
    for identifier in definitions:
        signature(identifier)
    reused={identifier:cached[1][identifier] for identifier in definitions
            if cached and len(cached)>2 and cached[2].get(identifier)==signatures[identifier]}
    result = engine.evaluate(frame, workspace['gates'],reuse=reused)
    engine.frame_cache(frame)['evaluation'] = (key, result, signatures)
    return frame, revision, result


def stats(workspace, result):
    output = []
    for gate in [{'id':'root','name':'Analysis cells','colour':'#a8b4c4'}, *workspace['gates']]:
        selected, known = result[gate['id']]
        total = int(result[gate.get('parent','root')][0].sum())
        output.append({'id':gate['id'], 'n':int(selected.sum()), 'evaluable':int(known.sum()),
                       'parent_n':total, 'percent':100*int(selected.sum())/total if total else 0.})
    return output


def public(workspace):
    return {k: v for k, v in workspace.items() if k not in ('history', 'future')}


def saved_stats(workspace, revision=None):
    record=workspace.get('analysis_stats') or {}
    if revision is None:
        revision=digest([(s['table'],data_editor.stamp(s['table'])) for s in workspace['sources']])
    if (record.get('dataset')==revision and record.get('sources')==digest(workspace['sources']) and
            record.get('gates')==digest(gate_meaning(workspace['gates']))):
        return record['stats']
    return None


def options(channel, *, metadata_only=False):
    workspace = load(channel)
    # Let controls open from the schema. The first plot supplies exact counts;
    # do not load millions of identity rows once here and again for its axes.
    frame, revision, result = (None, None, None) if metadata_only else evaluate(workspace)
    columns, numeric = None, None
    for source in workspace['sources']:
        table = data_view._table(source['table'])
        names = set(table.names)
        numbers = {n for n in names if table.kinds[n] == 'value'}
        columns = names if columns is None else columns & names
        numeric = numbers if numeric is None else numeric & numbers
    return {'workspace':public(workspace), 'stats':stats(workspace,result) if result is not None else saved_stats(workspace) or [], 'dataset_revision':revision,
            'columns':sorted(columns or []), 'numeric':sorted(numeric or []),
            'logicle':bool(importlib.util.find_spec('flowutils')), 'n':len(frame) if frame is not None else None}


def validate_gate(gate):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', str(gate.get('id',''))):
        raise ValueError('Invalid population ID.')
    if not isinstance(gate.get('name'),str) or not 1 <= len(gate['name'].strip()) <= 128:
        raise ValueError('Enter a population name (up to 128 characters).')
    if not re.fullmatch(r'#[0-9a-fA-F]{6}', gate.get('colour','')):
        raise ValueError('Choose a population colour.')


def save(channel, body):
    project, _, folder = context(channel)
    with project_lock(project['root']), _COMPUTE:
        before = load(channel)
        if body.get('revision') != before['revision']:
            raise ValueError('Changed in another window. Refresh before editing.')
        workspace = deepcopy(before)
        action = body.get('action','save')
        editable = ('gates','plots','layouts','output_column','output_dir','active','overlays','live','recipes','root_colour','thresholds','stats_design')
        fields = (*editable, 'sources')
        before.setdefault('recipes',[])
        for record in (*before.get('history',[]), *before.get('future',[])):
            record.setdefault('thresholds',[])
            record.setdefault('stats_design',{})
        workspace.setdefault('recipes',[])
        if action in ('undo','redo'):
            origin, target = ('history','future') if action == 'undo' else ('future','history')
            if not workspace[origin]:
                raise ValueError('Nothing to ' + action + '.')
            workspace[target].append({k:deepcopy(before[k]) for k in fields})
            workspace.update(workspace[origin].pop())
        else:
            changes = body.get('changes',{})
            if set(changes)-set(editable):
                raise ValueError('Unknown workspace setting.')
            workspace['history'] = [*before['history'][-29:], {k:deepcopy(before[k]) for k in fields}]
            workspace['future'] = []
            workspace.update(changes)
            if action == 'scope':
                workspace['sources'] = sources(channel)
                # Previously fitted gates may not be silently transferred to a new reference.
                if any(a.get('normalize','none') != 'none' or a.get('pipeline') for g in workspace['gates'] for a in [g.get('x',{}),g.get('y',{})]):
                    raise ValueError('Normalized gates retain their reference scope. Start a new workspace or remove them before changing scope.')
        for gate in workspace['gates']:
            validate_gate(gate)
        if not re.fullmatch(r'#[0-9a-fA-F]{6}', str(workspace['root_colour'])):
            raise ValueError('Choose a population colour.')
        colour_only = (before['gates'] != workspace['gates'] and
                       [{k:v for k,v in g.items() if k != 'colour'} for g in before['gates']] ==
                       [{k:v for k,v in g.items() if k != 'colour'} for g in workspace['gates']])
        if action == 'save' and ('gates' not in body.get('changes', {}) or colour_only):
            # Saving a new plot/layout/camera does not require cell membership.
            # In particular, do not hold a new plot behind a million-row save.
            workspace['revision'] += 1
            workspace['saved_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(folder/'workspace.json', workspace)
            return {'workspace':public(workspace)}
        frame, revision = dataset(workspace)
        for gate in workspace['gates']:
            for axis in ('x','y'):
                if axis in gate and ((gate[axis].get('normalize','none') != 'none' and 'fitted' not in gate[axis]) or
                                     ('pipeline' in gate[axis] and not gate[axis].get('pipeline_fitted'))):
                    gate[axis] = engine.fit_axis(frame,gate[axis])
        frame, revision, result = evaluate(workspace)
        workspace['revision'] += 1
        workspace['dataset_revision'] = revision
        workspace['saved_at'] = datetime.now(timezone.utc).isoformat()
        if before['gates'] != workspace['gates'] and workspace.get('applied'):
            workspace['applied']['stale'] = True
        # Only persist memberships when meaning/scope/data changes, not fonts or camera.
        membership_key = digest([revision,gate_meaning(workspace['gates']),workspace['sources']])
        folder.mkdir(parents=True,exist_ok=True)
        target = folder / ('membership-'+membership_key[:24]+'.parquet')
        if not target.exists():
            import pyarrow as pa
            import pyarrow.parquet as pq
            identity = engine.frame_cache(frame).get('identity_arrow')
            if identity is None:
                # Encode repeated table paths and ROI names once. Going through
                # a new pandas object frame on every edit converted millions of
                # identical strings and started extra Arrow conversion threads.
                identity = {c:pa.array(pd.Categorical(frame[c])) if c in ('_flow_table','_flow_roi') else pa.array(frame[c])
                            for c in ('_flow_table','_flow_roi','_flow_id','_flow_row')}
                engine.frame_cache(frame)['identity_arrow'] = identity
            membership = dict(identity)
            for key,(mask,known) in result.items():
                membership[key] = pa.array(mask, mask=~known)
            temporary = target.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            try:
                pq.write_table(pa.table(membership), temporary, compression='zstd')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        workspace['membership_file'] = str(target)
        summary=stats(workspace,result)
        workspace['analysis_stats']={'dataset':revision,'sources':digest(workspace['sources']),
                                    'gates':digest(gate_meaning(workspace['gates'])),'stats':summary}
        atomic_json(folder/'workspace.json',workspace)  # publish only a complete revision
        flow_cache.prune_memberships(folder, target)
        return {'workspace':public(workspace),'stats':summary}


def fit_plot(workspace, spec):
    needed = required({'gates':[spec]})
    frame, revision, result = evaluate(workspace,needed)
    spec = deepcopy(spec)
    for key in ('x','y'):
        if spec.get(key):
            axis=spec[key]
            reference=result.get(axis.get('reference_population','root'))
            if reference is None:
                raise ValueError('The normalization reference population is missing.')
            spec[key] = engine.fit_axis(frame,axis,reference[0]) if 'fitted' not in axis and not axis.get('pipeline_fitted') else axis
    return frame, revision, result, spec


def live_mask(frame, context):
    if context is None:
        return None
    source = context.get('kind')
    mask = frame['_flow_table'].eq(context.get('table')).to_numpy()
    if source == 'cells':
        return mask & frame['_flow_roi'].eq(str(context.get('roi'))).to_numpy() & frame['_flow_id'].isin(context.get('ids',[])).to_numpy()
    if source == 'rows':
        return mask & frame['_flow_row'].isin(context.get('rows',[])).to_numpy()
    if source == 'filters':
        for rule in context.get('filters',[]):
            if 'values' in rule:
                parsed = {'column':rule['col'],'op':'in','values':rule['values']}
                mask &= engine.filter_rule(frame,parsed)[0]
            else:
                for key,op in [('lo','>='),('hi','<=')]:
                    if rule.get(key) is not None:
                        mask &= engine.filter_rule(frame,{'column':rule['col'],'op':op,'value':rule[key]})[0]
        return mask
    if source == 'region':
        from shapely.geometry import shape
        from shapely import contains_xy
        x = pd.to_numeric(frame[context['x']],errors='coerce').to_numpy()
        y = pd.to_numeric(frame[context['y']],errors='coerce').to_numpy()
        polygons = [contains_xy(shape(g),x,y) for g in context.get('geometries',[])]
        inside = (np.logical_and.reduce(polygons) if context.get('join') == 'and' else np.logical_or.reduce(polygons)) if polygons else np.zeros(len(frame),bool)
        return mask & frame['_flow_roi'].eq(str(context.get('roi'))).to_numpy() & inside
    if source == 'image':
        return mask & frame['_flow_roi'].eq(str(context.get('roi'))).to_numpy()
    raise ValueError('Choose a live selection source.')


def plot_dependencies(workspace, spec):
    """Only populations affecting the plotted values invalidate those values."""
    by_id={g['id']:g for g in workspace['gates']}
    selected={s.get('population','root') for s in spec.get('series',[])} or {spec.get('parent','root')}
    selected.update(spec[a].get('reference_population','root') for a in ('x','y') if spec.get(a))
    visited={'root'}
    def visit(identifier):
        if identifier in visited:
            return
        if identifier not in by_id:
            raise ValueError('This plot population was removed. Choose another population.')
        visited.add(identifier)
        gate=by_id[identifier]
        for parent in [gate.get('parent','root'),*gate.get('refs',[])]:
            visit(parent)
    for identifier in selected:
        visit(identifier)
    return gate_meaning([g for g in workspace['gates'] if g['id'] in visited])


def plot_gates(workspace, spec):
    output=[]
    for gate in workspace['gates']:
        if gate.get('parent','root') != spec.get('parent','root') or gate.get('x',{}).get('column') != spec['x']['column']:
            continue
        if gate.get('y',{}).get('column') != (spec.get('y') or {}).get('column') and gate.get('kind') not in ('range','threshold'):
            continue
        output.append({**gate,'boundary':engine.reproject_boundary(gate,spec['x'],spec.get('y'))})
    return output


def plot(channel, body):
    start = time.perf_counter()
    workspace = load(channel)
    if body.get('revision') is not None and body['revision'] != workspace['revision']:
        raise ValueError('The workspace changed. Refresh the plot.')
    live = body.get('context')
    spec = deepcopy(body['plot'])
    # A threshold is a fitted quantitative feature, independent of its parent mask.
    candidates = workspace.get('thresholds',[]) + [g for g in workspace['gates'] if g.get('kind') == 'threshold']
    selected = next((g for g in candidates if g['id'] == spec.get('threshold_id')), None)
    if not selected and spec.get('threshold_id') != 'none':
        selected = next((g for g in reversed(workspace['gates']) if g.get('kind') == 'threshold' and
                         g.get('parent','root') == spec.get('parent','root') and g.get('x') == spec.get('x')), None)
    if selected and selected.get('x') == spec.get('x'):
        spec['threshold'] = {k:deepcopy(selected[k]) for k in ('id','name','x','bounds')}
    else:
        spec.pop('threshold',None)
    needs_density = spec.get('kind') in ('density_scatter','density','contour')
    _, _, folder = context(channel)
    # Check the small cache before loading/fitting any cell columns. Stamps,
    # scope, gates, filters, recipes and Live inputs are part of its identity.
    for source in workspace['sources']:
        with data_editor.locked(source['table']) as editor:
            if editor.cursor:
                raise ValueError('Save the Data edits before gating this table.')
    stamps = [(s['table'], data_editor.stamp(s['table'])) for s in workspace['sources']]
    def request_key(plot_spec):
        numerical = {k:v for k,v in plot_spec.items() if k not in ('id','name','style','camera','kind','tool','live_columns')}
        return digest(['flow-plot-v6', workspace['id'], workspace['sources'], stamps, plot_dependencies(workspace,plot_spec), numerical, needs_density, live])
    cache_key = request_key(spec)
    cached = flow_cache.get(folder, cache_key)
    signature=digest(gate_meaning(workspace['gates']))
    current_stats=saved_stats(workspace,digest(stamps))
    if cached is not None and (current_stats is not None or cached.get('gate_signature')==signature):
        fitted = {k:cached['plot'][k] for k in ('x','y') if k in cached['plot']}
        palette = {g['id']:g for g in workspace['gates']}
        requested = spec.get('series') or [{'population':spec.get('parent','root')}]
        series = []
        for item, values in zip(requested, cached['histogram']['series']):
            population = palette.get(item.get('population'), {'name':'Analysis cells','colour':workspace['root_colour']})
            series.append({**values, 'name':item.get('name',population['name']), 'colour':item.get('colour',population['colour'])})
        gates = plot_gates(workspace,{**spec,**fitted})
        return {**cached, 'histogram':{**cached['histogram'], 'series':series}, 'gates':gates,
                'stats':current_stats if current_stats is not None else cached['stats'],'gate_signature':signature,
                'plot':{**spec, **fitted}, 'revision':workspace['revision'],
                'elapsed_ms':round((time.perf_counter()-start)*1000,1), 'cached':True}
    extra = [r['col'] for r in (live or {}).get('filters',[])] + [v for k,v in (live or {}).items() if k in ('x','y')]
    frame, revision, result, spec = fit_plot(workspace,{**spec, 'live_columns':[{'column':c} for c in extra]})
    context_mask = live_mask(frame,live)
    hist = engine.distributions(frame,result,spec,context_mask)
    palette={g['id']:g for g in workspace['gates']}
    for series in hist['series']:
        population=palette.get(series.get('population'),{'name':'Analysis cells','colour':workspace['root_colour']})
        series.setdefault('name',population['name'])
        series.setdefault('colour',population['colour'])
    masks = []
    points = []
    sample = int(spec.get('sample',20000))
    if not 100 <= sample <= 200000:
        raise ValueError('Display sample must be 100–200,000 cells.')
    x = engine.measurement(frame,spec['x'])
    y = engine.measurement(frame,spec['y']) if spec.get('y') else None
    for series in hist['series']:
        mask = result[series.get('population','root')][0] & engine.filter_rule(frame,series.get('rule'))[0]
        mask &= engine.filter_rule(frame,spec.get('rule'))[0]
        if context_mask is not None and not series.get('pinned'):
            mask &= context_mask
        if y is not None:
            valid = mask & np.isfinite(x) & np.isfinite(y)
            ids = np.flatnonzero(valid)
            chosen = ids if len(ids) <= sample else np.random.default_rng(0).choice(ids,sample,replace=False)
            values = {'x':x[chosen].tolist(), 'y':y[chosen].tolist(), 'rows':chosen.tolist(), 'n':int(mask.sum()),
                      'measured':int(valid.sum()),'sampled':len(chosen), 'invalid':int((mask & ~valid).sum())}
            if needs_density and valid.any():
                density,xe,ye = np.histogram2d(x[valid],y[valid],bins=96)
                values['density'] = {'z':density.T.tolist(),'x':((xe[:-1]+xe[1:])/2).tolist(),'y':((ye[:-1]+ye[1:])/2).tolist()}
            points.append(values)
        masks.append(mask)
    found={'revision':workspace['revision'], 'dataset_revision':revision, 'plot':spec, 'histogram':hist,
            'points':points, 'gates':plot_gates(workspace,spec), 'gate_signature':signature, 'stats':stats(workspace,result), 'elapsed_ms':round((time.perf_counter()-start)*1000,1),
            'live_n':int(context_mask.sum()) if context_mask is not None else None}
    if stamps != [(s['table'], data_editor.stamp(s['table'])) for s in workspace['sources']]:
        raise ValueError('Measurements changed while plotting. Refresh the plots.')
    flow_cache.put(folder, cache_key, found)
    fitted_key = request_key(spec)
    if fitted_key != cache_key:
        flow_cache.put(folder, fitted_key, found)
    return found


def select(channel, body):
    workspace = load(channel)
    if body.get('revision') != workspace['revision']:
        raise ValueError('The populations changed. Refresh before selecting.')
    frame, revision, results = evaluate(workspace, required({'gates':[body.get('rule')]}))
    if body.get('expected_dataset_revision',revision) != revision:
        raise ValueError('The measurements changed. Run the analysis again before selecting cells.')
    chosen = body.get('populations') or ['root']
    selection_key=digest([channel,workspace['id'],workspace['revision'],revision,chosen,body.get('join','or'),body.get('rule')])
    with _LOCK:
        cached=next(((token,item) for token,item in _TOKENS.items() if item.get('key')==selection_key),None)
        if cached:
            _TOKENS.move_to_end(cached[0])
            return cached[1]['summary']
    masks = [results[key][0] for key in chosen]
    mask = np.logical_and.reduce(masks) if body.get('join') == 'and' else np.logical_or.reduce(masks)
    mask &= engine.filter_rule(frame,body.get('rule'))[0]
    colours = {g['id']:g['colour'] for g in workspace['gates']}
    token = uuid.uuid4().hex
    item = {'channel':channel,'stamps':[(s['table'],data_editor.stamp(s['table'])) for s in workspace['sources']],
            'frame':frame, 'mask':mask, 'colour':colours.get(chosen[0],workspace['root_colour']),'revision':revision, 'luts':{},'rows':{},'key':selection_key}
    summary={'token':token,'count':int(mask.sum()),'colour':item['colour'],'name':next((g['name'] for g in workspace['gates'] if g['id']==chosen[0]),'Analysis cells'),
             'by_image':frame.loc[mask,'_flow_roi'].value_counts().to_dict(), 'revision':workspace['revision']}
    item['summary']=summary
    with _LOCK:
        _TOKENS[token] = item
        while len(_TOKENS) > 32:
            _TOKENS.popitem(last=False)
    return summary


def register_selection(channel, table, stamp, frame, name, colour=None):
    """Publish explicit reviewed membership using the shared Data/image protocol."""
    token = uuid.uuid4().hex
    colour = colour if isinstance(colour, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', colour) else '#f3b34d'
    summary = {'token': token, 'count': len(frame), 'name': name, 'colour': colour,
               'by_image': frame['_flow_roi'].value_counts().to_dict()}
    item = {'channel': channel, 'stamps': [(table, stamp)], 'frame': frame,
            'mask': np.ones(len(frame), bool), 'colour': summary['colour'],
            'luts': {}, 'rows': {}, 'summary': summary}
    with _LOCK:
        _TOKENS[token] = item
        while len(_TOKENS) > 32:
            _TOKENS.popitem(last=False)
    return summary


def token_item(token, channel=None, table=None):
    with _LOCK:
        item = _TOKENS.get(token)
        if item is None or channel is not None and item['channel'] != channel:
            raise ValueError('This gate view expired. Select the population again.')
        _TOKENS.move_to_end(token)
    if table is not None and str(table) not in {s[0] for s in item['stamps']}:
        raise ValueError('This gate belongs to another table.')
    if any(data_editor.stamp(path) != stamp for path,stamp in item['stamps']):
        raise ValueError('The cell measurements changed. Select the gate again.')
    return item


def token_rows(token, channel, table):
    item = token_item(token,channel,str(table))
    frame = item['frame']
    with _LOCK:
        if str(table) not in item['rows']:
            item['rows'][str(table)]=frame.loc[item['mask'] & frame['_flow_table'].eq(str(table)).to_numpy(),'_flow_row'].to_numpy(np.int64)
        return item['rows'][str(table)]


def tile(token, table, roi, image_labels, highlight=False):
    item = token_item(token,table=table)
    frame = item['frame']
    key = (str(table),str(roi),highlight)
    with _LOCK:
        lut = item['luts'].get(key)
    if lut is None:
        current = frame['_flow_table'].eq(str(table)).to_numpy() & frame['_flow_roi'].eq(str(roi)).to_numpy()
        ids = frame.loc[current,'_flow_id'].to_numpy(np.int64)
        maximum = int(ids.max()) if len(ids) else 0
        if maximum > 20000000:
            raise ValueError('Segmentation labels are too large for the image lookup.')
        lut = np.zeros((maximum+1,4),np.uint8)
        if highlight:
            lut[ids] = [170,180,190,35]
        selected = frame.loc[current & item['mask'],'_flow_id'].to_numpy(np.int64)
        colour = item['colour'].lstrip('#')
        lut[selected] = [int(colour[i:i+2],16) for i in (0,2,4)]+[230]
        with _LOCK:
            item['luts'][key] = lut
    return np.where(image_labels < len(lut),image_labels,0), lut


def label_plan(channel, body):
    workspace = load(channel)
    if body.get('revision') != workspace['revision']:
        raise ValueError('Populations changed. Preview the labels again.')
    # Scope changes in the image checkboxes must never silently apply old gates.
    if sources(channel) != workspace['sources']:
        raise ValueError('Selected images changed. Use Sync selected images before applying labels.')
    frame, revision, results = evaluate(workspace)
    ids = body.get('populations',[])
    if not ids or 'root' in ids or len(set(ids)) != len(ids):
        raise ValueError('Choose named populations to label.')
    gates = {g['id']:g for g in workspace['gates']}
    column = body.get('column','phenotype_flow').strip()
    if not column or len(column)>128 or re.search(r'[\x00-\x1f/\\]',column):
        raise ValueError('Enter a valid output column name.')
    if any(column in {s['label_column'],s['cell_id_column'],s['roi_column']} for s in workspace['sources']):
        raise ValueError('Use a label column, not a cell identity.')
    masks = np.stack([results[key][0] for key in ids])
    count = masks.sum(axis=0)
    valid = np.logical_and.reduce([results[key][1] for key in ids])
    assigned = np.full(len(frame),None,object)
    conflict = body.get('conflict','skip')
    if conflict not in ('skip','priority'):
        raise ValueError('Choose Skip overlaps or explicit priority.')
    if conflict == 'priority' and body.get('priority') != ids:
        raise ValueError('Confirm the ordered population priority before assigning overlaps.')
    for key,mask in reversed(list(zip(ids,masks))):
        chosen = mask & ((count == 1) if conflict == 'skip' else (count > 0))
        assigned[chosen] = gates[key].get('label') or gates[key]['name']
    existing = np.full(len(frame),None,object)
    for source in workspace['sources']:
        path = source['table']
        chosen = frame['_flow_table'].eq(path).to_numpy()
        table = data_view._table(path)
        if column in table.names:
            if table.kinds[column] == 'value' and pd.notna(table.column(column)).any():
                raise ValueError('That column contains numeric measurements. Choose a label column.')
            existing[chosen] = table.take(frame.loc[chosen,'_flow_row'].to_numpy(),[column])[column]
    preserve = body.get('preserve',True)
    occupied = pd.notna(existing) & (np.asarray([str(v).strip() if v is not None else '' for v in existing]) != '')
    kept = int(((assigned != None) & occupied).sum()) if preserve else 0  # noqa: E711
    if preserve:
        assigned[occupied] = None
    summary = {'unique':int((count==1).sum()),'overlap':int((count>1).sum()),'unassigned':int((count==0).sum()),
               'unevaluable':int((~valid).sum()), 'preserved':kept,'write':int(pd.notna(assigned).sum()),
               'column':column,'dataset_revision':revision,'revision':workspace['revision']}
    summary['receipt'] = digest([summary,ids,body.get('priority'),preserve,conflict])
    return workspace,frame,assigned,summary


def apply_labels(channel, body):
    project,_,folder = context(channel)
    with project_lock(project['root']), ExitStack() as stack:
        workspace = load(channel)
        for path in sorted(s['table'] for s in workspace['sources']):
            stack.enter_context(labels.write_lock(path))
        workspace,frame,assigned,summary = label_plan(channel,body)
        if body.get('receipt') != summary['receipt']:
            raise ValueError('Labels or gates changed. Preview again before applying.')
        stamp = datetime.now(timezone.utc).isoformat()
        # Preflight every table before writes. Each table uses the existing atomic Parquet label journal.
        writes = []
        for path,part in frame.groupby('_flow_table',sort=False):
            positions = part.index.to_numpy()
            positions = positions[pd.notna(assigned[positions])]
            updates = pd.DataFrame({'row':frame.loc[positions,'_flow_row'].to_numpy(),
                                    'column':summary['column'],'value':assigned[positions],'saved_at':stamp})
            if len(updates):
                writes.append((path,updates))
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='apply-',dir=folder) as temporary:
            backups=[]
            for i,(path,_) in enumerate(writes):
                target=labels.sidecar_path(path);backup=Path(temporary)/str(i)
                if target.exists():
                    shutil.copyfile(target,backup)
                backups.append((target,backup))
            try:
                for path,updates in writes:
                    labels.append(path,updates.to_dict('records'))
                workspace['revision'] += 1
                workspace['output_column'] = summary['column']
                workspace['applied'] = {**summary,'stale':False,'saved_at':stamp}
                atomic_json(folder/'workspace.json',workspace)
            except Exception:
                for target,backup in backups:
                    if backup.exists():
                        staging=target.with_name(target.name+'.flow-rollback')
                        shutil.copyfile(backup,staging);staging.replace(target)
                    else:
                        target.unlink(missing_ok=True)
                raise
    return {'workspace':public(workspace),'summary':summary}


def create_column(channel,body):
    project,_,folder=context(channel)
    with project_lock(project['root']),ExitStack() as stack:
        workspace=load(channel)
        if body.get('revision')!=workspace['revision']:
            raise ValueError('The workspace changed. Refresh before creating a column.')
        column=body.get('column','').strip()
        if not column or len(column)>128 or re.search(r'[\x00-\x1f/\\]',column):
            raise ValueError('Enter a valid output column name.')
        paths=sorted({s['table'] for s in workspace['sources']})
        for path in paths:
            stack.enter_context(labels.write_lock(path))
        dataset(workspace)  # Reject unsaved Data edits before touching any table.
        for source in workspace['sources']:
            if column in {source['label_column'],source['cell_id_column'],source['roi_column']}:
                raise ValueError('Use a label column, not a cell identity.')
            table=data_view._table(source['table'])
            if column in table.names and table.kinds[column]=='value' and pd.notna(table.column(column)).any():
                raise ValueError('That column contains measurements. Choose a label column.')
        folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='column-',dir=folder) as temporary:
            backups=[]
            for i,path in enumerate(paths):
                target=labels.sidecar_path(path);backup=Path(temporary)/str(i)
                if target.exists():
                    shutil.copyfile(target,backup)
                backups.append((target,backup))
            try:
                for path in paths:
                    table=data_view._table(path)
                    if column not in table.names and table.count:
                        labels.append(path,[{'row':0,'column':column,'value':None}])
                workspace['revision']+=1
                workspace['output_column']=column
                atomic_json(folder/'workspace.json',workspace)
            except Exception:
                for target,backup in backups:
                    if backup.exists():
                        shutil.copyfile(backup,target)
                    else:
                        target.unlink(missing_ok=True)
                raise
        return {'column':column,'workspace':public(workspace)}


def dispatch(channel, body):
    action = body.get('action')
    if body.get('workspace') is not None and body['workspace'] != load(channel)['id']:
        raise ValueError('The project or cell table changed. Reopen Flow gating.')
    if action in ('stats_run','stats_export'):
        from . import flow_statistics
        with _COMPUTE:
            return (flow_statistics.run if action == 'stats_run' else flow_statistics.export)(channel,body)
    if action == 'transfer':
        from .flow_transfer import transfer
        return transfer(channel, body)
    if action in ('save','undo','redo','scope'):
        return save(channel,body)
    if action == 'split':
        from .flow_split import split
        return split(channel,body)
    if action == 'plot':
        with _COMPUTE:
            return plot(channel,body)
    if action == 'select':
        return select(channel,body)
    if action == 'preview_labels':
        return label_plan(channel,body)[-1]
    if action == 'apply_labels':
        return apply_labels(channel,body)
    if action == 'create_column':
        return create_column(channel,body)
    if action == 'export':
        from .flow_export import export
        workspace=load(channel)
        if body.get('manifest',{}).get('statistics'):
            from .flow_statistics import validate_figure
            validate_figure(workspace,body['manifest']['statistics'])
        return export(workspace,body)
    if action == 'values':
        workspace = load(channel)
        frame,_ = dataset(workspace,[body['column']])
        return {'values':frame[body['column']].astype('string').dropna().value_counts().head(500).to_dict()}
    if action == 'freeze':
        # Snapshot contains exact plotted aggregates and sampled scatter, independent of subsequent gates.
        snapshot = plot(channel,body)
        workspace = load(channel)
        layouts = deepcopy(workspace['layouts'])
        layouts.append({'id':uuid.uuid4().hex,'name':body.get('name','Publication page'),
                        'panels':[{'id':uuid.uuid4().hex,'x':0,'y':0,'w':780,'h':520,'snapshot':snapshot}],
                        'width':1100,'height':780,'frozen_at':datetime.now(timezone.utc).isoformat()})
        return save(channel,{'revision':body['revision'],'changes':{'layouts':layouts}})
    raise ValueError('Unknown Flow gating action.')
