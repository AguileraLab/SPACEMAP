"""Content-based population splits; all memberships stay within their parent."""
import re
import uuid

import numpy as np
import pandas as pd

from spacemap import phenotype_presets
from . import flow_gating as flow
from .settings import project_lock

CATEGORY_LIMIT = 25  # numeric columns with fewer distinct values are categories


def _kind(frame, column):
    cache = flow.engine.frame_cache(frame).setdefault('split_kinds', {})
    if column not in cache:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values.dtype):
            present = values.notna()
            numeric = True
        else:
            present = values.notna() & values.astype('string[pyarrow]').str.strip().ne('').fillna(False)
            numeric = bool(present.any()) and pd.to_numeric(values[present], errors='coerce').notna().all()
        distinct = int(values[present].nunique())
        cache[column] = {'kind':'numeric' if numeric and distinct >= CATEGORY_LIMIT else 'categorical', 'distinct':distinct}
    return cache[column]


def _name(value):
    text = str(value)
    return text if len(text) <= 128 else text[:115] + '…' + flow.digest(text)[:12]


def _categories(values, column):
    text = values.astype('string[pyarrow]')
    empty = text.isna() | text.str.strip().eq('').fillna(False)
    counts = text[~empty].value_counts()
    names = counts.index.tolist()
    if len(names) > 256:
        raise ValueError('Too many categories. Filter this population first or choose another column.')
    names = sorted(names, key=lambda s: (s.casefold(), s))
    if names and pd.to_numeric(pd.Series(names),errors='coerce').notna().all():
        names.sort(key=lambda s:(float(s),s))
    groups = [(_name(name), {'column':column,'op':'in','values':[name]}, name) for name in names]
    if empty.any():
        label = 'Missing' if 'Missing' not in names else 'Missing (empty)'
        groups.append((label, {'column':column,'op':'blank'}, None))
    categories = [{'label':name, 'value':value,
                   'count':int(empty.sum()) if value is None else int(counts[value])}
                  for name, _, value in groups]
    return groups, categories


def _numeric_groups(column, condition, missing):
    mode = condition.get('mode')
    def bound(key):
        value = condition.get(key)
        if isinstance(value, bool) or value is None or not str(value).strip():
            raise ValueError('Enter a finite numeric cutoff.')
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('Enter a finite numeric cutoff.')
        return number
    def rule(op, value):
        return {'column':column,'op':op,'value':value}
    if mode == 'threshold':
        value = bound('value')
        groups = [(f'{column} < {value:g}', rule('<',value), None),
                  (f'{column} ≥ {value:g}', rule('>=',value), None)]
    elif mode == 'range':
        low, high = bound('low'), bound('high')
        if low >= high:
            raise ValueError('Enter a minimum below the maximum.')
        groups = [(f'{column}: {low:g}–{high:g}', {'join':'and','rules':[rule('>=',low),rule('<=',high)]}, None),
                  (f'{column}: outside {low:g}–{high:g}', {'join':'or','rules':[rule('<',low),rule('>',high)]}, None)]
    else:
        raise ValueError('Choose Threshold or Range.')
    if missing:
        groups.append((f'{column}: missing', {'column':column,'op':'nonfinite'}, None))
    return [(_name(name),rule,value) for name,rule,value in groups]


def split(channel, body):
    """Preview choices before committing selected categories or numeric bounds."""
    project, _, _ = flow.context(channel)
    with project_lock(project['root']):
        workspace = flow.load(channel)
        if body.get('revision') != workspace['revision']:
            raise ValueError('The populations changed. Reopen Split by.')
        parent, column = body.get('parent','root'), body.get('column')
        if not isinstance(column, str) or not column or column.startswith('_flow_'):
            raise ValueError('Choose a dataset column.')
        with flow._COMPUTE:
            frame, revision, results = flow.evaluate(workspace, [column])
            if parent not in results:
                raise ValueError('Choose an existing parent population.')
            mask = results[parent][0]
            if not mask.any():
                raise ValueError('No cells in this population.')
            if body.get('dataset_revision') not in (None, revision):
                raise ValueError('Measurements changed. Reopen Split by.')
            description = {**_kind(frame,column), 'column':column, 'parent':parent,
                           'dataset_revision':revision, 'revision':workspace['revision']}
            values = frame.loc[mask, column]
            if description['kind'] == 'numeric':
                numbers = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float,na_value=np.nan)
                finite = numbers[np.isfinite(numbers)]
                description.update(missing=int(len(numbers)-len(finite)),
                                   minimum=float(finite.min()) if len(finite) else None,
                                   maximum=float(finite.max()) if len(finite) else None,
                                   median=float(np.median(finite)) if len(finite) else None)
                if not body.get('condition'):
                    return description
                groups = _numeric_groups(column, body['condition'], description['missing'])
            else:
                groups, categories = _categories(values,column)
                if 'categories' not in body:
                    return {**description, 'categories':categories}
                selected = body['categories']
                if not isinstance(selected, list) or not selected:
                    raise ValueError('Choose at least one category.')
                if any(value is not None and not isinstance(value, str) for value in selected):
                    raise ValueError('Invalid category selection. Reopen Split by.')
                selected = set(selected)
                if not selected.issubset({value for _, _, value in groups}):
                    raise ValueError('Categories changed. Reopen Split by.')
                if body.get('dataset_revision') != revision:
                    raise ValueError('Measurements changed. Reopen Split by.')
                groups = [group for group in groups if group[2] in selected]
        existing = {flow.digest([g.get('parent','root'),g.get('rule')]):g for g in workspace['gates'] if g['kind']=='filter'}
        fresh = [(name,rule,value) for name,rule,value in groups if flow.digest([parent,rule]) not in existing]
        if len(workspace['gates']) + len(fresh) > 256:
            raise ValueError('This split exceeds 256 populations. Remove unused branches or filter first.')
        palette = phenotype_presets.ensure([name for name,_,_ in fresh])
        overrides = body.get('colours') or {}
        if not isinstance(overrides,dict):
            raise ValueError('Invalid population colours.')
        created = []
        for name, rule, value in fresh:
            colour = overrides.get(str(value)) if value is not None else None
            if not isinstance(colour,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',colour):
                colour = palette[name.strip()]
            # Keep colour lookup tied to the category, independently of its gate label.
            gate_name = _name(f'{column}_{value if value is not None else name}') if description['kind'] == 'categorical' else name
            created.append({'id':uuid.uuid4().hex,'kind':'filter','parent':parent,'name':gate_name,'colour':colour,'rule':rule,
                            'split':{'column':column,'kind':description['kind'],'value':value}})
        if not created:
            return {**description, 'workspace':flow.public(workspace),'stats':flow.stats(workspace,results),'added':0}
        # Measurements must still be the same as when the categories were read.
        if revision != flow.digest([(s['table'],flow.data_editor.stamp(s['table'])) for s in workspace['sources']]):
            raise ValueError('Measurements changed. Reopen Split by.')
        saved = flow.save(channel, {'revision':workspace['revision'],'changes':{'gates':[*workspace['gates'],*created]}})
        return {**description, **saved, 'added':len(created)}
