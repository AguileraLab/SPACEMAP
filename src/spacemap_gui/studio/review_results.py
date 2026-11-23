"""Review saved analysis assignments without changing membership or source files.

Each item has an immutable, image-scoped membership snapshot. Decisions edit the
naming column only, after checking IDs, membership and previous names. Large
memberships stay in Parquet, not in the queue or browser request.
"""
from pathlib import Path
import hashlib
import json
from collections import OrderedDict
from threading import RLock
import pandas as pd
from spacemap import dataio, labels, table_collection
from spacemap.viewer.catalog import sample_of
from . import jobs, review_store as store, phenotyping_inputs

KINDS = {'leiden':'leiden', 'som':'som', 'vortex':'vortex',
         'subphenotype':'subcluster', 'neighbourhoods':'neighbourhood'}
_LAYERS = OrderedDict()
_LAYER_LOCK = RLock()
layer_epoch = 0


def normal(values):
    return values.astype('string').fillna('').str.replace(r'\.0$', '', regex=True)


def stamp(path):
    s = Path(path).stat()
    return f'{s.st_size}:{s.st_mtime_ns}'


def _save(root, frame):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()
                            + json.dumps(list(frame.columns)).encode()).hexdigest()
    folder = store.home(root) / 'memberships'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (digest + '.parquet')
    if not path.exists():
        from .annotation_edits import _atomic
        import io
        content = io.BytesIO(); frame.to_parquet(content, index=False)
        _atomic(path, content.getvalue())
    return digest


def snapshot(root, target):
    key = target['membership']
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Invalid saved review membership.')
    return pd.read_parquet(store.home(root) / 'memberships' / (key + '.parquet'))


def _request(root, job_id):
    path = jobs.directory(root, job_id) / 'request.json'
    return json.loads(path.read_text()).get('settings', {}) if path.exists() else {}


def _items(root, target, frame, groups, kind, suggestions=None, requested_by=None):
    defaults = store.remembered_suggestions(root)
    found = []
    for group in groups:
        selected = frame.loc[frame['__group'].eq(group)]
        if selected.empty:
            continue
        exact = {**target, 'ids':[group], 'count':len(selected)}
        if {'__x','__y'} <= set(selected):
            xy = selected[['__x','__y']].apply(pd.to_numeric, errors='coerce').dropna()
            if not xy.empty:
                exact['bounds'] = [float(xy.__x.min()),float(xy.__y.min()),float(xy.__x.max()),float(xy.__y.max())]
        title = target.get('titles', {}).get(group, group)
        choices = suggestions if suggestions is not None else defaults.get(store.suggestion_context({'kind':kind,'target':exact}), [])
        found.append(dict(kind=kind, image=Path(target['image']).name, title=title,
                          target=exact, suggestions=choices, requested_by=requested_by))
    return store.add_many(root, found)


def flag(root, body, rows):
    job_id = str(body.get('job_id') or '')
    job = jobs.status(root, job_id)
    result = job.get('result') or {}
    if job.get('status') != 'done':
        raise ValueError('Choose a completed result before flagging it.')
    settings = _request(root, job_id)
    groups = list(dict.fromkeys(str(v) for v in body.get('groups', [])))
    if not groups or len(groups) > 2000:
        raise ValueError('Select the groups or structures to flag (up to 2,000).')
    mode = body.get('mode', '')
    profile = result.get('profile') or {}
    columns = result.get('values') or {}
    structs = links = None
    if job['kind'] in KINDS:
        kind = KINDS[job['kind']]
        member = result['column']; output = result.get('phenotype_column') or member + '_reviewed'
        allowed = [str(c['name']) for c in result['clusters']]
        from .cluster_names import saved_names
        naming=saved_names(root,result['table'],member,cluster_job_id=job_id)
        output=naming.get('column') or output
        guard = [member]
    elif job['kind'] == 'gland_types':
        if result.get('diagnose'):
            raise ValueError('Fit gland types first. K review does not assign cells.')
        if mode not in ('interior','exterior','gland','bed'):
            raise ValueError('Choose gland types, niche types, glands or beds.')
        exterior = mode in ('exterior','bed')
        kind = {'interior':'gland_type','exterior':'niche_type','gland':'gland','bed':'niche'}[mode]
        member = output = columns['niche_col' if exterior else 'output_col']
        profile = result['profiles']['exterior' if exterior else 'interior']
        guard = [member]
        if mode in ('gland','bed'):
            folder = jobs.directory(root, job_id)
            structs = pd.read_parquet(folder/'gland_structures.parquet')
            links = pd.read_parquet(folder/'gland_cell_links.parquet')
            allowed = structs.structure.astype(str).tolist()
        else:
            allowed = profile['rows']
    elif job['kind'] == 'tls_candidates':
        kind = 'tls'; member = result['column'] + '_id'; output = result['column'] + '_review'
        allowed = [str(s['key']) for s in result['structures']]
        region_col = result.get('settings', settings.get('values', {})).get('roi_col')
        guard = [member, *([region_col] if region_col else [])]
    else:
        raise ValueError('This saved result does not support review.')
    if set(groups) - set(allowed):
        raise ValueError('Some selected groups are not in this saved run.')
    if member == output and job['kind'] in KINDS:
        raise ValueError('Choose a naming column separate from cluster membership.')
    # A historical heatmap must not be attached to a newer clustering that reused
    # the same output column. Resolve the latest writer before snapshotting.
    for newer in jobs.history(root):
        if newer['id'] == job_id:
            break
        other = newer.get('result') or {}
        if (newer.get('status') == 'done' and newer.get('kind') != 'name_clusters'
                and other.get('table') == result.get('table')
                and member in (other.get('columns') or [other.get('column')])):
            raise ValueError('A newer run replaced this result. Flag groups in the latest run.')
    tables = {str(Path(p).resolve()) for p in [result.get('table'), *result.get('tables', [])] if p}
    collection = {r['image']:r for r in table_collection.entries(result['table'])} if table_collection.is_collection(result.get('table','')) else {}
    tables.update(str(Path(r['path']).resolve()) for r in collection.values())
    image_tables = result.get('image_tables') or {}
    found = []; cache = {}; titles = {}
    if structs is not None:
        titles = {str(r.structure):f'{"Bed" if mode=="bed" else "Gland"} {r.local_id} · {r.niche if mode=="bed" else r.gland_type}' for r in structs.itertuples()}
    for row in rows:
        image = row.get('image_path'); table = image_tables.get(row['roi']) or row.get('table_path')
        if not image or not table or str(Path(table).resolve()) not in tables:
            continue
        if result.get('images') and row['roi'] not in result['images']:
            continue
        available = dataio.columns(table)
        roles = phenotyping_inputs.column_roles(root, table, available)
        entry = collection.get(row['roi'])
        roi_col = columns.get('image_col') or settings.get('image_col') or settings.get('roi_column') or roles['roi']
        anchor = columns.get('cell_id_col') or result.get('cell_id_column') or roles['cell_id'] or 'label'
        if entry:
            roi_col = None
            anchor = entry['anchor']
        label = columns.get('label_col') or phenotyping_inputs.segmentation_column(root, table)
        x = columns.get('x_col') or 'x_centroid'; y = columns.get('y_col') or 'y_centroid'
        revision = stamp(table)
        required = list(dict.fromkeys([anchor, *guard, *([roi_col] if roi_col else [])]))
        if not set(required) <= set(available):
            raise ValueError('The saved result columns changed. Reopen the current result before flagging.')
        wanted = list(dict.fromkeys(required + [c for c in [output,label,x,y] if c in available]))
        key = (table, tuple(wanted))
        if key not in cache:
            cache[key] = dataio.load(table, columns=wanted, build_fast_copy=False)
        part = cache[key]
        if roi_col:
            values = normal(part[roi_col]); aliases = {v:sample_of(v) for v in values.unique()}
            part = part.loc[values.map(aliases).eq(sample_of(row['roi']))]
        if part.empty:
            continue
        if normal(part[anchor]).eq('').any() or normal(part[anchor]).duplicated().any():
            raise ValueError('Cell IDs must be present and unique within each image before review.')
        saved = pd.DataFrame({'__id':normal(part[anchor]), '__group':normal(part[member]),
                              '__before':normal(part[output]) if output in part else ''})
        for c in guard:
            saved['guard:'+c] = normal(part[c])
        if label in part: saved['__label'] = normal(part[label])
        if x in part and y in part:
            saved['__x'] = part[x]; saved['__y'] = part[y]
        if structs is not None:
            chosen_links = links.loc[links.image.astype(str).eq(str(row['roi']))]
            by_id = chosen_links.set_index('cell_id')['bed' if mode=='bed' else 'gland']
            link_ids = json.dumps(row['roi'],ensure_ascii=False)+':'+saved.__id if entry else saved.__id
            indices = link_ids.map(by_id)
            saved['__group'] = indices.map(structs.structure.astype(str).to_dict()).fillna('')
        elif kind == 'tls':
            saved['__group'] = ''
            for s in result['structures']:
                if s['image'] == row['roi']:
                    m = normal(part[member]).eq(str(s['id']))
                    if len(guard)>1: m &= normal(part[guard[1]]).eq(str(s['region']))
                    saved.loc[m, '__group'] = str(s['key'])
        if not saved.__group.isin(groups).any():
            continue
        if stamp(table) != revision:
            raise ValueError('The table changed while preparing review. Try again.')
        target = dict(source=str(table), image=str(Path(image).resolve()), revision=revision, column=output,
                      roi=str(row['roi']), roi_column=roi_col, label_column=label, anchor=anchor,
                      membership=_save(root,saved), job_id=job_id, mode=mode,
                      result_kind=job['kind'], titles=titles, groups=allowed,
                      options=sorted(set(saved.__before)-{''}) or allowed,
                      profile=profile, member_column=member)
        if structs is not None or kind=='tls':
            # Individual structure IDs are in the membership snapshot. Repeating
            # the whole cohort's list in every item makes large review queues huge.
            target.pop('groups',None)
            image_groups=set(saved.__group)
            target['titles']={g:t for g,t in titles.items() if g in image_groups}
        found.extend(_items(root,target,saved,groups,kind,body.get('suggestions')))
    if not found:
        raise ValueError('No selected cells match the assigned project images and this run. Check the image/table assignments.')
    return {'items':found}


def context(root, item):
    target = item['target']; frame = snapshot(root, target)
    counts = frame.__group.value_counts()
    names = frame.groupby('__group')['__before'].agg(lambda v:', '.join(sorted(set(v)-{''}))).to_dict()
    with store.database(root) as db:
        items=[store._item(row) for row in db.execute("SELECT * FROM items WHERE kind=? AND json_extract(target,'$.membership')=?",
            (item['kind'],target['membership']))]
    states={g:i['state'] for i in items for g in i['target']['ids']}
    return {'profile':target.get('profile',{}), 'selected':target['ids'], 'bounds':target.get('bounds'),
            'individual':item['kind'] in ('gland','niche','tls'),
            'groups':[{'name':g,'title':target.get('titles',{}).get(g,g),'cells':int(counts.get(g,0)),
                       'label':names.get(g,''),
                       'state':states.get(g,'Not flagged')}
                      for g in target.get('groups', sorted(set(frame.__group)-{''}))], 'kind':item['kind']}


def nominate(root, item, groups, grant_id):
    target = item['target']
    frame=snapshot(root,target)
    if not isinstance(groups,list) or not groups or len(groups)>2000 or set(groups)-set(frame.__group):
        raise ValueError('Choose groups from the shared result.')
    return {'items':_items(root,target,frame,groups,item['kind'],requested_by=grant_id)}


def apply(root, item):
    target = item['target']; path = Path(target['source'])
    with labels.write_lock(path):
        if stamp(path) != target['revision']:
            raise ValueError('The source table changed. Reflag this result before applying.')
        saved = snapshot(root,target); saved = saved.loc[saved.__group.isin(target['ids'])]
        available = dataio.columns(path)
        guards = [c[6:] for c in saved if c.startswith('guard:')]
        cols = [target['anchor'], *guards, *([target['roi_column']] if target['roi_column'] else [])]
        if target['column'] in available: cols.append(target['column'])
        frame = dataio.load(path,columns=list(dict.fromkeys(cols)),build_fast_copy=False)
        if target['roi_column']:
            values = normal(frame[target['roi_column']]); aliases={v:sample_of(v) for v in values.unique()}
            frame = frame.loc[values.map(aliases).eq(sample_of(target['roi']))]
        keys = normal(frame[target['anchor']])
        if keys.duplicated().any(): raise ValueError('Cell IDs are no longer unique.')
        indexed = frame.set_index(keys,drop=False)
        if not set(saved.__id)<=set(indexed.index): raise ValueError('Reviewed cells are missing. Reflag this result.')
        aligned = indexed.loc[saved.__id]
        for c in guards:
            if normal(aligned[c]).tolist()!=saved['guard:'+c].tolist():
                raise ValueError('Cluster or structure membership changed. Reflag this result.')
        before = normal(aligned[target['column']]).tolist() if target['column'] in aligned else ['']*len(aligned)
        if before!=saved.__before.tolist(): raise ValueError('These labels changed after flagging. Review the newer labels first.')
        if item['decision']['action']=='keep': return {'message':'Approved; assignments retained.'}
        if item['kind'] == 'tls' and item['decision']['action'] == 'reject':
            # Never leave TLS IDs, area and distances inconsistent by clearing only
            # the presence label. Recompute the affected region with normal TLS code.
            from . import tls_review
            result = jobs.status(root,target['job_id'])['result']
            structure = next(s for s in result['structures'] if str(s['key'])==target['ids'][0])
            draft = tls_review.context(root,str(path),target['job_id'],structure['image'],structure['region'])
            draft['features'] = [f for f in draft['features'] if str(f['properties']['key'])!=target['ids'][0]]
            settings = tls_review.prepare(root,str(path),draft)
            job = jobs.start(root, 'tls_candidates', settings, request_id='review-'+item['id'])
            return {'job_id':job['id'], 'message':'TLS boundary update queued in Jobs; presence, IDs, sizes and distances will be recomputed together.'}

        value = item['decision'].get('label') if item['decision']['action']=='relabel' else None
        mask = keys.isin(saved.__id)
        count = labels.append(path,[{'row':int(i),'column':target['column'],'value':value} for i in frame.index[mask]])
    return {'count':count,'message':'Labels staged. Write into the table in Jobs to update the data file.'}


def layer(root, item):
    """A read-only image-scoped table and the project's existing cell outlines."""
    global layer_epoch
    key = _layer_key(root,item)
    with _LAYER_LOCK:
        if key in _LAYERS:
            _LAYERS.move_to_end(key)
            return _LAYERS[key]
    value = _layer(*key)
    with _LAYER_LOCK:
        if value:
            _LAYERS[key] = value
            while len(_LAYERS)>512: _LAYERS.popitem(last=False)
            layer_epoch += 1
        return value


def _layer_key(root,item):
    binding = {k:item['target'].get(k) for k in ('membership','image','roi','source','roi_column','label_column')}
    return str(root), json.dumps(binding,sort_keys=True)


def registered_layer(root,item):
    """Read an already opened binding; grant checks never rasterize other images."""
    with _LAYER_LOCK:
        return _LAYERS.get(_layer_key(root,item))


def _layer(root, target_json):
    from spacemap_gui.viewer import registry
    from spacemap.viewer import cells
    from spacemap.viewer.sources import image_info
    from . import cell_layers, settings, cell_view
    target=json.loads(target_json); frame=snapshot(root,target)
    if '__label' not in frame:
        return None
    config=settings.read_config(root); mappings=config.get('directory_mappings',{})
    image=target['image']; roi=target['roi']
    own=cell_layers.own(root,config,mappings,image,roi)
    segmentation=None
    def matches(candidate):
        reference=cell_layers.table_of(root,image,candidate)
        return reference and cell_view.matches(reference,target['source'],roi,
            roi_column=target.get('roi_column'),label_column=target.get('label_column') or 'label')[0]
    if own and matches(own):
        segmentation=own
    else:
        for entry in cell_layers.candidates(root,config,mappings,image,roi):
            if entry['path']!=own and matches(entry['path']):
                segmentation=entry['path'];break
    if not segmentation:return None
    info=image_info(image)
    pyramid,_=cells.prepare_label_pyramid(segmentation,info.width,info.height,Path(root)/'outputs/studio/cache')
    view=frame.rename(columns={'__label':'label','__id':'cell_id','__group':'Review group','__before':'Saved label','__x':'x_centroid','__y':'y_centroid'})
    view=view.drop(columns=[c for c in view if c.startswith('guard:')])
    view['batch_id']=roi
    digest=_save(root,view); path=store.home(root)/'memberships'/(digest+'.parquet')
    token=registry.register_object('cells',image=image,pyramid=str(pyramid),segmentation=segmentation,
        roi=roi,roi_column='batch_id',label_column='label',tables={'review':str(path)})
    return {'id':'review-result-'+token,'type':'cells','name':'Review cells','token':token,
        'image_token':registry.register(image),'visible':True,'opacity':.7,'table':'review',
        'color_by':'Review group','tables':[{'id':'review','name':'Reviewed result'}]}


def flag_cells(root, channel, body, rows):
    from . import manual_reference
    path,keys,_=manual_reference.context(channel)
    if str(path)!=body.get('table'):
        raise ValueError('The active table changed. Select the cells again.')
    column=str(body.get('column') or '')
    available=dataio.columns(path)
    if column not in available:
        raise ValueError('Create or choose a manual-label column first.')
    groups=manual_reference._positions(channel,path,keys,body.get('selection'))
    if sum(len(g['ids']) for g in groups)>5000:
        raise ValueError('Flag up to 5,000 individual cells at a time; use cluster review for whole groups.')
    wanted=list(dict.fromkeys([column,*[v for v in keys.values() if v],*[v for v in ('x_centroid','y_centroid') if v in available]]))
    frame=dataio.load(path,columns=wanted,build_fast_copy=False); found=[]
    for group in groups:
        candidates=[r for r in rows if r.get('image_path') and sample_of(r['roi'])==sample_of(group['roi'])
                    and r.get('table_path') and Path(r['table_path']).resolve()==Path(path).resolve()]
        if len(candidates)!=1:
            raise ValueError('Assign a unique image to this table before flagging cells.')
        part=frame
        if keys['roi_column']:
            part=part.loc[normal(part[keys['roi_column']]).eq(str(group['roi']))]
        ids=normal(part[keys['label_column']])
        if ids.duplicated().any():raise ValueError('Cell IDs are not unique within the image.')
        chosen=[str(i) for i in group['ids']]
        if not set(chosen)<=set(ids):raise ValueError('Some selected cells are no longer available.')
        saved=pd.DataFrame({'__id':ids,'__label':ids,'__group':ids,'__before':normal(part[column])})
        if 'x_centroid' in part and 'y_centroid' in part:
            saved['__x']=part.x_centroid;saved['__y']=part.y_centroid
        target=dict(source=str(path),image=str(Path(candidates[0]['image_path']).resolve()),revision=stamp(path),
            column=column,roi=group['roi'],roi_column=keys['roi_column'],label_column=keys['label_column'],
            anchor=keys['label_column'],membership=_save(root,saved),options=sorted(set(normal(frame[column]))-{''}),
            titles={i:f'Cell {i}' for i in chosen})
        found.extend(_items(root,target,saved,chosen,'cells',body.get('suggestions')))
    return {'items':found}


def reconcile(root):
    """Finish owner-approved TLS applications only after their durable job ends."""
    for item in store.targets(root, store.list_items(root,state='applying',ids_only=True)['ids']):
        job_id=(item.get('applied') or {}).get('job_id')
        if not job_id:
            continue
        job=jobs.status(root,job_id)
        if job['status'] not in ('done','failed','cancelled'):
            continue
        state='applied' if job['status']=='done' else 'conflict'
        result={'job_id':job_id, 'message':'TLS fields updated together. Write into the table in Jobs to update the data file.'} if state=='applied' else {'job_id':job_id,'error':job.get('message') or 'TLS update stopped; review Jobs before retrying.'}
        try:
            store.transition(root,item['id'],version=item['version'],state=state,result=result)
        except ValueError:
            pass  # Another queue poll already reconciled the same version.
