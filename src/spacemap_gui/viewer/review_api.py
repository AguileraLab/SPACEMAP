"""Owner review API and a narrowly scoped, revocable reviewer entry point.

External gateways must expose only /review-public and /window static assets,
not the ordinary Studio API. The public proxy rechecks grants on every request.
"""
from pathlib import Path
import copy
import json
import os
import re
from functools import lru_cache
from hashlib import sha256

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from . import registry
from spacemap_gui.studio import review_store as store, annotation_components as components


def _source(token):
    for kind in ('annotations', 'shapes', 'cells'):
        value = registry.resolve_object(token, kind)
        if value:
            return value
    raise ValueError('Reopen the source layer before reviewing it.')


def _root(request, write=False):
    from .api import _studio_project
    return _studio_project(request, write=write)


def _view(request):
    value = registry.session_view(request.path_params['channel'], mark_seen=False) or {}
    return value.get('spec', value)


def _check_image(request, source):
    images = [registry.resolve(l.get('token')) for l in _view(request).get('layers', []) if l.get('type') == 'image']
    channel = request.path_params['channel']
    images.extend(entry.get('path') for entry in registry.session_images(channel))
    image = Path(source['image']).resolve()
    if image in {Path(p).resolve() for p in images if p}:
        return
    # Opening an image in Studio's browser does not always republish the Classic
    # session spec. Check the project's exact assigned image paths as well, never
    # just the last published image or a similarly named raw/processed file.
    from spacemap_gui.studio.spatial_sources import image_entries
    project = registry.session_project(channel) or {}
    if project.get('root') and image in {Path(entry['path']).resolve() for entry in image_entries(project['root'], project.get('unified'))}:
        return
    raise ValueError('This annotation belongs to an image outside this project’s loaded image list. Open or add its source image first.')


def _public_item(item):
    # Never send file paths, table IDs, or owner session credentials to a link.
    result = {k: v for k, v in item.items() if k not in ('target', 'applied')}
    if item.get('applied'):
        result['applied'] = {'message': 'Applied by the project owner.'}
    if item['target'].get('job_id') and item['target'].get('membership'):
        result['has_result_context'] = True
    if item['kind'] == 'annotation':
        result['polygon_ids'] = item['target']['ids']
    return result


def component_read(path, q, image):
    """Shared read-only component operations for the owner and scoped reviewer."""
    if q.get('overview') == '1':
        return JSONResponse({**components.overview(path), 'display': components.display_style(path)})
    if q.get('selection'):
        from .api import _png
        from spacemap.viewer.sources import image_info
        return _png(components.selection_tile(path, q['selection'], image_info(image),
            int(q['z']), int(q['x']), int(q['y']), q.get('hide', '').split('\x1f')))
    if 'x' in q and 'y' in q:
        return JSONResponse(components.hit(path, float(q['x']), float(q['y']), q.get('hide', '').split('\x1f')))
    if q.get('id'):
        value = components.index(path)
        feature = value['by_id'].get(q['id'])
        if feature is None:
            raise ValueError('This polygon is no longer available.')
        if q.get('profile') == '1':
            from spacemap_gui.studio import regions
            if q.get('revision') != value['revision']:
                raise ValueError('The annotation changed. Click it again.')
            return JSONResponse(regions.profile(image, feature['geometry'], None))
        return JSONResponse({'feature': feature, 'revision': value['revision']})
    return JSONResponse(components.page(path, name=q.get('name'), search=q.get('search', ''),
        sort=q.get('sort', 'size_down'), offset=q.get('offset', 0), locate=q.get('locate')))


async def component_route(request):
    try:
        root = _root(request, write=request.method == 'POST')
        source = _source(request.path_params['token'])
        if source['kind'] not in ('annotations', 'shapes'):
            raise ValueError('Choose an annotation source.')
        await run_in_threadpool(_check_image, request, source)
        path = components.effective_path(root, source['image'], source['path'])
        if request.method == 'GET':
            return await run_in_threadpool(component_read, path, request.query_params, source['image'])
        body = await request.json()
        if body.get('action') == 'outline':
            return JSONResponse(await run_in_threadpool(components.selection, path, body.get('ids', [])))
        if body.get('action') == 'area':
            return JSONResponse(await run_in_threadpool(components.area_selection, path, body.get('points'), body.get('hidden', [])))
        rows = await run_in_threadpool(components.selected, path, ids=body.get('ids'),
                                      name=body.get('name_filter'), search=body.get('search', ''), below=body.get('below'))
        if body.get('action') == 'select':
            return JSONResponse({'ids': [r['id'] for r in rows]})
        result = await run_in_threadpool(components.edit, root, source['image'], source['path'],
            ids=[r['id'] for r in rows], expected=body.get('revision'), action=body.get('action'), name=body.get('name', ''))
        # Component editing keeps the new source tiled, including after a rename.
        # Reading every component into a shapes layer defeats paging on large files.
        from .api import _annotation_layer
        layer = _annotation_layer(result['path'], source['image'], registry.register(source['image']), Path(result['path']).name)
        layer.update(annotation_group='imported', annotation_origin='copy', saved_file=result['path'])
        return JSONResponse({**result, 'layer': layer})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


def flag(request, root, body):
    source = _source(body.get('token'))
    _check_image(request, source)
    image = source['image']
    cell_context = capture_cells(request, body, image)
    suggestions, note = body.get('suggestions'), body.get('note') or ''
    defaults = store.remembered_suggestions(root) if suggestions is None else {}
    if source['kind'] in ('annotations', 'shapes'):
        path = components.effective_path(root, image, source['path'])
        revision = components.index(path)['revision']
        rows = components.selected(path, ids=body.get('ids'))
        if not body.get('ids'):
            raise ValueError('Select the polygons to flag.')
        found = []
        for r in rows:
            choices = suggestions if suggestions is not None else defaults.get(store.suggestion_context({'kind':'annotation','target':{'annotation_name':r['name']}}), [])
            found.append(dict(kind='annotation', image=Path(image).name,
                title=r['name'], suggestions=choices, note=note, area=r['area'],
                target={'source': source['path'], 'image': image, 'ids': [r['id']],
                        'revision': revision, 'fingerprints': {r['id']: r['fingerprint']}, 'bounds': r['bounds'],
                        'cell_context': cell_context}))
        return {'items': store.add_many(root, found)}
    from spacemap_gui.studio.cell_links import table_layer
    from spacemap import dataio
    table_id = str(body.get('table', ''))
    bound = table_layer(source, table_id)
    table = (bound.get('tables') or {}).get(table_id)
    column = str(body.get('column') or '')
    if not table or column not in dataio.columns(table):
        raise ValueError('Choose the cell table and the label column to review.')
    # Cell labels are scoped by the registered image; never match row positions
    # or reuse cell numbers across different images.
    ids = list(dict.fromkeys(str(x) for x in body.get('ids', [])))
    if not ids:
        raise ValueError('Select exact cells or a saved structure first.')
    path = Path(table)
    stat = path.stat()
    target = {'source': table, 'image': image, 'ids': ids, 'revision': f'{stat.st_size}:{stat.st_mtime_ns}',
              'column': column, 'roi': bound.get('roi'), 'roi_column': bound.get('roi_column', 'batch_id'),
              'label_column': bound.get('label_column', 'label'), 'cell_token': body['token'], 'table': table_id}
    kind = body.get('kind', 'cells')
    if suggestions is None:
        suggestions = defaults.get(store.suggestion_context({'kind':kind,'target':target}), [])
    return {'items': [store.add(root, kind=kind, image=Path(image).name,
        title=body.get('title') or f'{column} · {len(ids)} cells', target=target, suggestions=suggestions, note=note)]}


def ensure_polygon_sizes(root, allowed=None):
    """Lazily index legacy flags for exact area sorting, once per saved source."""
    from collections import defaultdict
    groups = defaultdict(list)
    for item in store.missing_polygon_sizes(root, allowed):
        target = item['target']
        groups[(target['image'], target['source'])].append(item)
    sizes = {}
    for (image, source), items in groups.items():
        try:
            rows = {r['id']: r['area'] for r in components.index(components.effective_path(root, image, source))['rows']}
        except (OSError, ValueError):
            continue  # Missing sources sort last; keep their review entries.
        for item in items:
            ids = item['target']['ids']
            sizes[item['id']] = sum(rows[i] for i in ids) if all(i in rows for i in ids) else None
    if sizes:
        store.save_polygon_sizes(root, sizes)


def suggestion_options(root, ids, images=None, kind=None):
    skipped = 0
    if images is not None:
        preview = store.suggestion_items_in_images(root, images, kind)
        items, skipped = preview['items'], preview['skipped']
    else:
        if not isinstance(ids, list) or not ids:
            raise ValueError('Select review items first.')
        items = store.targets(root, ids)
        if len(items) != len(set(ids)):
            raise ValueError('Some review items are no longer available.')
    names = {'Other'}
    seen = set()
    for item in items:
        names.update(item['suggestions'])
        names.update(item['target'].get('options', []))
        target = item['target']
        if item['kind'] == 'annotation':
            names.add(target.get('annotation_name') or item['title'])
            # Whole-image bulk choices use queue metadata; never parse every
            # image's polygon geometry just to open a suggestion dialog.
            if images is not None:
                continue
            path = components.effective_path(root, target['image'], target['source'])
            if str(path) not in seen:
                names.update(row['name'] for row in components.page(path)['classes'])
                seen.add(str(path))
    return {'items':items, 'options':sorted(names, key=str.casefold), 'skipped':skipped}


def apply_decision(root, item_id, version):
    item = store.transition(root, item_id, version=version, state='applying')
    target, decision = item['target'], item['decision']
    try:
        if target.get('membership'):
            from spacemap_gui.studio import review_results
            result = review_results.apply(root, item)
        elif decision['action'] == 'keep':
            result = {'message': 'Approved; original assignment retained.'}
        elif item['kind'] == 'annotation':
            result = components.edit(root, target['image'], target['source'], ids=target['ids'],
                expected=target['revision'], fingerprints=target.get('fingerprints'),
                action='rename' if decision['action'] == 'relabel' else 'delete', name=decision.get('label', ''))
            result.pop('result', None)
            result['message'] = 'Saved to the editable annotation copy. Refresh cell tagging for affected regions when needed.'
        else:
            from spacemap import dataio, labels
            import pandas as pd
            path = Path(target['source'])
            with labels.write_lock(path):
                stat = path.stat()
                if f'{stat.st_size}:{stat.st_mtime_ns}' != target['revision']:
                    raise ValueError('The cell table changed. Reflag the current cells before applying this decision.')
                cols = list(dict.fromkeys([target['roi_column'], target['label_column'], target['column']]))
                frame = dataio.load(path, columns=cols)
                normal = lambda s: s.astype(str).str.replace(r'\.0$', '', regex=True)
                mask = normal(frame[target['roi_column']]).eq(str(target['roi'])) & normal(frame[target['label_column']]).isin(target['ids'])
                if mask.sum() != len(target['ids']):
                    raise ValueError('The reviewed cells do not match uniquely. No labels were changed.')
                value = decision.get('label', '') if decision['action'] == 'relabel' else None
                count = labels.append(path, [{'row': int(i), 'column': target['column'], 'value': value}
                                             for i in frame.index[mask]])
                result = {'count': count, 'message': 'Applied to saved cell labels. Write into the table in Jobs to update the data file.'}
        if result.get('job_id'):
            return store.attach_application_job(root,item_id,item['version'],result)
        return store.transition(root, item_id, version=item['version'], state='applied', result=result)
    except Exception as exc:
        store.transition(root, item_id, version=item['version'], state='conflict', result={'error': str(exc)})
        raise


def capture_cells(request, body, image):
    """Snapshot only the displayed cell/table binding for this exact image."""
    result=[]
    candidates=body.get('cell_context')
    if candidates is None:
        candidates=[{'token':l.get('token'),'table':l.get('table')} for l in _view(request).get('layers',[]) if l.get('type')=='cells']
    for candidate in candidates[:20]:
        bound=registry.resolve_object(candidate.get('token'),'cells')
        if not bound or Path(bound['image']).resolve()!=Path(image).resolve():
            continue
        table=candidate.get('table')
        if table in bound.get('tables',{}):
            result.append({'token':candidate['token'],'table':table})
    return result


def shared_cell_layer(candidate):
    from spacemap_gui.studio.cell_links import table_layer
    original=registry.resolve_object(candidate.get('token'),'cells')
    if not original:
        return None
    table=candidate['table'];bound=copy.deepcopy(table_layer(original,table))
    if table not in bound.get('tables',{}):
        return None
    bound.pop('kind',None)
    bound['tables']={table:bound['tables'][table]}
    bound.pop('table_roles',None)
    token=registry.register_object('cells',**bound)
    return {'id':'review-cells-'+token,'type':'cells','token':token,'visible':False,
            'name':'Shared cells','opacity':.7,'table':table,'roi':bound.get('roi'),
            'image_token':registry.register(bound['image']),'color_by':'',
            'tables':[{'id':table,'name':'Shared cell data'}]}


def view_spec(root, item, *, reviewer=False, cell_context=()):
    from . import ui
    from spacemap.viewer.sources import image_info
    target = item['target']
    image = ui.image_layer(target['image'], save_display=False)
    layers = [image]
    he = ui.he_overlay(target['image'], list(image_info(target['image']).channel_names))
    if he:
        layers.append(he)
    if item['kind'] == 'annotation':
        path = components.effective_path(root, target['image'], target['source'])
        token = registry.register_object('annotations', path=str(path), image=target['image'])
        layers.append({'id': 'review-annotations', 'type': 'annotations', 'token': token, 'review_item': item['id'],
                       'name': Path(target['source']).name, 'opacity': .5, 'visible': True, 'image_token': image['token'],
                       **components.display_style(path)})
        value = components.index(path)
        features = review_previews(root, [item['id']], item['id'])['features']
        # Selection is small and uses full geometry, while context remains tiled.
        layers.append({'id': 'review-focus', 'type': 'shapes', 'name': 'Needs review', 'editable': False,
                       'features': features, 'review_inline': True, 'image_token': image['token'], 'visible': True, 'opacity': 1, 'regions': True})
    elif target.get('membership'):
        from spacemap_gui.studio import review_results
        layer = review_results.layer(root, item)
        if layer:
            layers.append(layer)
    elif target.get('cell_token'):
        original = registry.resolve_object(target['cell_token'], 'cells')
        if original:
            # Only the reviewed table is shared, under a separate registered binding.
            bound = copy.deepcopy(original)
            bound.pop('kind', None)
            bound['tables'] = {target['table']: target['source']}
            token = registry.register_object('cells', **bound)
            layers.append({'id': 'review-cells', 'type': 'cells', 'token': token, 'visible': True,
                           'name': 'Reviewed cells', 'opacity': .7, 'table': target['table'],
                           'image_token': image['token'], 'color_by': target['column'], 'tables': [{'id': target['table'], 'name': 'Shared cell data'}]})
    for candidate in target.get('cell_context',cell_context):
        layer=shared_cell_layer(candidate)
        if layer and not any(l.get('token')==layer['token'] for l in layers):
            layers.append(layer)
    return {'id': 'review-' + str(registry.register(target['image'])) + '-' + item['id'],
            'title': item['image'], 'layers': layers, 'height': 780, 'read_only': reviewer,
            **({'review_selection': {'ids':target['ids'], 'bounds':target.get('bounds')}} if target.get('membership') else {})}


def review_previews(root, ids, focus_id, *, all_in_image=False, allowed=None):
    """Read-only previews in one image, with proposal colours and stable IDs."""
    focus = store.get(root, focus_id)
    if all_in_image:
        ids=store.list_items(root,kind='annotation',search=focus['image'],ids=allowed,ids_only=True)['ids']
    features, palettes, areas = [], {}, {}
    for item in store.targets(root, ids):
        target = item['target']
        if item['kind'] != 'annotation' or target['image'] != focus['target']['image']:
            continue
        path = components.effective_path(root, target['image'], target['source'])
        value = components.index(path)
        if str(path) not in palettes:
            palettes[str(path)] = {r['name']: r['color'] for r in components.overview(path)['classes']}
            areas[str(path)] = {r['id']: r['area'] for r in value['rows']}
        for polygon in target['ids']:
            source = value['by_id'].get(polygon)
            if source is None:
                continue
            props = copy.deepcopy(source['properties'])
            name = (props.get('classification') or {}).get('name') or props.get('name') or 'Unlabelled'
            decision = item.get('decision') or {}
            if decision.get('action') == 'relabel' and item['state'] != 'conflict':
                name = decision['label']
            colour = palettes[str(path)].get(name) or palettes[str(path)].get((props.get('classification') or {}).get('name')) or '#ffd600'
            props.update(id=item['id'] + ':' + str(polygon), name=name,
                review_area=areas[str(path)].get(str(polygon)),
                classification={'name': name, 'color': [int(colour[j:j+2],16) for j in (1,3,5)]},
                colour=colour, review_item=item['id'], review_polygon=str(polygon), review_state=item['state'],
                review_version=item['version'], review_suggestions=item['suggestions'], review_decision=decision,
                review_source=registry.register_object('annotations',path=str(path),image=target['image']))
            features.append({'type': 'Feature', 'properties': props, 'geometry': source['geometry']})
    features.sort(key=lambda f: (f['properties']['classification']['name'], f['properties']['id']))
    return {'features': features}


def image_review_previews(root, image, allowed=None):
    found=store.list_items(root,kind='annotation',search=Path(image).name,ids=allowed,ids_only=True)
    items=[i for i in store.targets(root,found['ids']) if i['target']['image']==image]
    preview=review_previews(root,[i['id'] for i in items],items[0]['id']) if items else {'features':[]}
    return {**preview,'revision':found['revision']}


async def owner(request):
    from . import review_links
    try:
        root = _root(request, write=request.method == 'POST')
        q = request.query_params
        if request.method == 'GET':
            from spacemap_gui.studio.review_results import reconcile
            await run_in_threadpool(reconcile,root)
            queue_image = None
            if 'image' in q and q.get('action', '') in ('', 'selection'):
                queue_image = registry.resolve(q['image']) if q['image'] else ''
                if queue_image is None:
                    raise ValueError('Open a listed image before viewing its review items.')
                if queue_image:
                    await run_in_threadpool(_check_image, request, {'image': queue_image})
                    queue_image = str(Path(queue_image).resolve())
            if q.get('action') == 'image_preview':
                image=registry.resolve(q.get('image',''))
                if not image:
                    raise ValueError('Open an image first.')
                await run_in_threadpool(_check_image, request, {'image':image})
                return JSONResponse(await run_in_threadpool(image_review_previews,root,str(image)))
            if q.get('action') == 'images':
                from spacemap_gui.studio import review_images
                found=await run_in_threadpool(review_images.plan,request.path_params['channel'])
                return JSONResponse({'images':found.get('rows',[]),'counts':store.list_items(root,limit=1)['counts'],
                                     'metadata_columns':store.metadata_columns(root)})
            if q.get('action') == 'metadata_options':
                from spacemap_gui.studio import review_images
                found=await run_in_threadpool(review_images.plan,request.path_params['channel'])
                return JSONResponse({'columns':await run_in_threadpool(review_images.options,found.get('rows',[]))})
            if q.get('action') == 'selection':
                return JSONResponse(store.list_items(root, kind=q.get('kind',''), state=q.get('state',''), search=q.get('search',''), ids_only=True, image=queue_image))
            if q.get('action') == 'links':
                return JSONResponse({'links': [review_links.link_info(root, link) for link in store.grants(root)], 'destination': store.destination(root), 'external_available': os.environ.get('SPACEMAP_REVIEW_PUBLIC_BASE_URL', '').startswith('https://')})
            if q.get('action') == 'link':
                return JSONResponse({'id': q['id'], 'url': review_links.saved_url(request, root, q['id'])}, headers={'Cache-Control':'no-store'})
            if q.get('action') == 'notifications':
                return JSONResponse(store.notifications(root,q.get('since',0)))
            if q.get('action') == 'history':
                return JSONResponse({'events': store.history(root, q['id'])})
            if q.get('action') == 'result_context':
                from spacemap_gui.studio import review_results
                return JSONResponse(await run_in_threadpool(review_results.context, root, store.get(root, q['id'])))
            if q.get('action') == 'view':
                item = store.get(root, q['id'])
                return JSONResponse({'spec': await run_in_threadpool(view_spec, root, item), 'item': item})
            if q.get('sort') in ('size_down', 'size_up'):
                ids = store.list_items(root, image=queue_image, kind='annotation', ids_only=True)['ids']
                await run_in_threadpool(ensure_polygon_sizes, root, ids)
            return JSONResponse(store.list_items(root, kind=q.get('kind', ''), state=q.get('state', ''),
                                                 search=q.get('search', ''), offset=q.get('offset', 0), since=q.get('since', -1), sort=q.get('sort','created'), image=queue_image))
        body = await request.json()
        action = body.get('action')
        if action == 'share_selection':
            from spacemap_gui.studio import review_images
            found = await run_in_threadpool(review_images.plan, request.path_params['channel'])
            rows = review_images.selected_rows(found, body.get('images', []))
            ids = await run_in_threadpool(store.item_ids_in_images, root,
                                         [r['image_path'] for r in rows], body.get('kind', ''))
            return JSONResponse({'ids': ids, 'total': len(ids)})
        if action == 'metadata':
            from spacemap_gui.studio import review_images
            found=await run_in_threadpool(review_images.plan,request.path_params['channel'])
            columns=body.get('columns',[])
            result=await run_in_threadpool(review_images.values,root,found.get('rows',[]),columns,per_image=found.get('per_image',False))
            if body.get('save'):
                store.metadata_columns(root,columns)
            return JSONResponse({'values':result,'columns':columns})
        if action == 'preview':
            return JSONResponse(await run_in_threadpool(review_previews, root, body.get('items', []), body['focus'],all_in_image=body.get('all_in_image',False)))
        if action == 'select_shapes':
            return JSONResponse(await run_in_threadpool(components.local_area_selection, body.get('features', []), body.get('points')))
        if action == 'select_cells':
            from spacemap_gui.studio.view_selection import cells_in_area
            source = _source(body['token'])
            if source['kind'] != 'cells':
                raise ValueError('Choose a cell layer.')
            await run_in_threadpool(_check_image, request, source)
            return JSONResponse(await run_in_threadpool(cells_in_area, source, body))
        if action == 'destination':
            return JSONResponse({'destination':store.destination(root,body.get('id',''))})
        if action == 'add_to_link':
            return JSONResponse(store.append_to_grant(root,body['id'],body.get('items',[])))
        if action == 'update_link':
            base = review_links.base_url(request, body.get('access', 'internal'))
            key = review_links.access_key(root, body['id'], body.get('existing_url', ''))
            context=await run_in_threadpool(share_context,request,root,body,body['id'])
            result = store.update_grant(root, body['id'], name=body.get('name', ''),
                access=body.get('access', 'internal'), items=body.get('items', []), days=body.get('days'),context=context)
            review_links.remember_address(root, body['id'], base)
            return JSONResponse({**result, 'url': base + '/review-public/' + key + '/'})
        if action == 'suggestion_options':
            return JSONResponse(await run_in_threadpool(suggestion_options, root, body.get('ids'), body.get('images'), body.get('kind')))
        if action == 'suggestions':
            return JSONResponse(await run_in_threadpool(store.set_suggestions, root, body.get('items'), body.get('choices'),
                                                       note=body.get('note'), remember=body.get('remember', False)))
        if action == 'flag_cells':
            from spacemap_gui.studio import review_results, review_images
            channel=request.path_params['channel']
            plan=await run_in_threadpool(review_images.plan,channel)
            return JSONResponse(await run_in_threadpool(review_results.flag_cells,root,channel,body,plan.get('rows',[])))
        if action == 'flag_result':
            from spacemap_gui.studio import review_results, review_images
            plan = await run_in_threadpool(review_images.plan, request.path_params['channel'])
            return JSONResponse(await run_in_threadpool(review_results.flag, root, body, plan.get('rows', [])))
        if action == 'flag':
            if body.get('sources'):
                sources=body['sources']
                if not isinstance(sources,list) or len(sources)>100:
                    raise ValueError('Select up to 100 annotation sources.')
                for source in sources:
                    await run_in_threadpool(_check_image, request, _source(source['token']))
                items=[]
                for source in sources:
                    result=await run_in_threadpool(flag,request,root,{**body,**source,'sources':None})
                    items.extend(result['items'])
                return JSONResponse({'items':items})
            return JSONResponse(await run_in_threadpool(flag, request, root, body))
        if action == 'decide':
            return JSONResponse(store.decide(root, body['id'], version=body['version'],
                action=body['decision'], label=body.get('label', ''), note=body.get('note', '')))
        if action == 'remove':
            return JSONResponse(await run_in_threadpool(store.remove, root, body.get('items')))
        if action == 'decide_many':
            return JSONResponse({'items': store.decide_many(root, body.get('items'), action=body['decision'],
                label=body.get('label', ''), note=body.get('note'))})
        if action == 'apply':
            return JSONResponse(await run_in_threadpool(apply_decision, root, body['id'], body['version']))
        if action == 'reopen':
            return JSONResponse(store.transition(root, body['id'], version=body['version'], state='pending'))
        if action == 'revoke':
            store.revoke(root, body['id']); return JSONResponse({'ok': True})
        if action == 'delete_link':
            store.delete_grant(root, body['id']); return JSONResponse({'ok': True})
        if action == 'share':
            access = body.get('access', 'internal')
            base = review_links.base_url(request, access)
            context=await run_in_threadpool(share_context,request,root,body)
            value = store.create_grant(root, name=body.get('name', ''), items=body.get('items', []), days=body.get('days', 14),context=context)
            key = review_links.remember(root, value['id'], value['secret'])
            store.update_grant(root, value['id'], name=body.get('name', ''), access=access)
            review_links.remember_address(root, value['id'], base)
            return JSONResponse({'id': value['id'], 'total': len(set(body.get('items', []))),
                                 'url': base + '/review-public/' + key + '/'})
        raise ValueError('Choose a review action.')
    except PermissionError as exc:
        return JSONResponse({'error': str(exc)}, status_code=403)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


def share_context(request, root, body, grant_id=None):
    from spacemap_gui.studio import review_images
    previous=next((g['scope'] for g in store.grants(root) if g['id']==grant_id),{})
    context=previous.get('context',{})
    ids=list(dict.fromkeys([*previous.get('items',[]),*body.get('items',[])]))
    targets=store.targets(root,ids)
    columns=body.get('metadata_columns',context.get('metadata_columns',[]))
    images={}
    if body.get('images') or columns or context.get('images'):
        found=review_images.plan(request.path_params['channel'])
        images=review_images.snapshot(root,found,body.get('images',[]),columns,
            item_images=[i['target']['image'] for i in targets],existing=context.get('images',{}))
    cells={}
    for image in {*images,*(i['target']['image'] for i in targets)}:
        captured=capture_cells(request,body,image)
        if captured:
            cells[image]=captured
    return {'images':images,'metadata_columns':columns,'cells':cells}


def shared_images(root, grant):
    images=copy.deepcopy(grant['scope'].get('context',{}).get('images',{}))
    for item in store.targets(root,grant['scope']['items']):
        images.setdefault(item['target']['image'],{'image':item['image'],'metadata':{}})
    return images


def _grant(request):
    handle, secret = request.path_params['access'].split('.', 1)
    link = registry.resolve_object(handle, 'review_link')
    if not link:
        raise PermissionError('This review link is unavailable.')
    grant = store.authorize(link['root'], link['grant'], secret)
    return link['root'], grant, secret


def _resources(root, grant_id, scope_json):
    from spacemap_gui.studio import review_results
    return _resource_cache(root,grant_id,scope_json,review_results.layer_epoch)


@lru_cache(maxsize=64)
def _resource_cache(root, grant_id, scope_json, result_epoch):
    scope = json.loads(scope_json)
    resources = {'image': set(), 'annotations': set(), 'cells': set()}
    for image in scope.get('context',{}).get('images',{}):
        resources['image'].add(registry.register(image))
        for candidate in scope.get('context',{}).get('cells',{}).get(image,[]):
            layer=shared_cell_layer(candidate)
            if layer:
                resources['cells'].add(layer['token'])
    for item in store.targets(root, scope['items']):
        target = item['target']
        for candidate in target.get('cell_context',scope.get('context',{}).get('cells',{}).get(target['image'],[])):
            layer=shared_cell_layer(candidate)
            if layer:
                resources['cells'].add(layer['token'])
        resources['image'].add(registry.register(target['image']))
        if item['kind'] == 'annotation':
            # Both original and managed-copy paths are explicitly bound to the
            # same authorized source; creating the copy doesn't break the link.
            for path in (Path(target['source']), components.managed_path(root,target['image'],target['source'])):
                resources['annotations'].add(registry.register_object('annotations', path=str(path), image=target['image']))
        elif target.get('membership'):
            from spacemap_gui.studio import review_results
            layer=review_results.registered_layer(root,item)
            if layer: resources['cells'].add(layer['token'])
        elif target.get('cell_token'):
            original=registry.resolve_object(target['cell_token'],'cells')
            if original:
                bound=copy.deepcopy(original);bound.pop('kind',None);bound['tables']={target['table']:target['source']}
                resources['cells'].add(registry.register_object('cells',**bound))
    return {kind: {_alias(grant_id, token): token for token in tokens} for kind,tokens in resources.items()}


def _alias(grant_id, token):
    """Grant-local opaque IDs must not resolve at the ordinary owner API."""
    return 'r_' + sha256((grant_id + '\0' + token).encode()).hexdigest()[:32]


def _project_spec(spec, grant_id):
    result=copy.deepcopy(spec)
    result['id']=_alias(grant_id,result['id'])
    for layer in result['layers']:
        for key in ('token','image_token'):
            if layer.get(key): layer[key]=_alias(grant_id,layer[key])
        for key in ('display_token','saved_file','sources','workspace','path'):
            layer.pop(key,None)
        if layer.get('remember'):
            layer['remember']=_alias(grant_id,str(layer['remember']))
        for feature in layer.get('features',[]):
            if feature.get('properties',{}).get('review_source'):
                feature['properties']['review_source']=_alias(grant_id,feature['properties']['review_source'])
    return result


async def public(request):
    try:
        root, grant, secret = _grant(request)
        action = request.path_params.get('action', '')
        queue_image = None
        if action in ('queue', 'selection') and 'image' in request.query_params:
            token = request.query_params['image']
            queue_image = next((path for path in shared_images(root, grant)
                                if _alias(grant['id'], registry.register(path)) == token), None) if token else ''
            if queue_image is None:
                raise PermissionError('This image was not shared.')
        if not action or action == 'queue':
            from . import review_links
            try:
                await run_in_threadpool(review_links.recover_visited_link, request, root, grant['id'], secret)
            except OSError:
                pass  # URL recall must not prevent a valid reviewer from reading their queue.
        if not action:
            # The reviewer page receives no owner session or project credentials.
            base = request.url.path.split('/review-public/')[0]
            return RedirectResponse(base + '/window/index.html?review=' + request.path_params['access'],
                                    headers={'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store'})
        if action == 'queue' and request.method == 'GET':
            from spacemap_gui.studio.review_results import reconcile
            await run_in_threadpool(reconcile,root)
            if request.query_params.get('sort') in ('size_down', 'size_up'):
                ids = store.list_items(root, image=queue_image, kind='annotation', ids=grant['scope']['items'], ids_only=True)['ids']
                await run_in_threadpool(ensure_polygon_sizes, root, ids)
            result = store.list_items(root, ids=grant['scope']['items'], offset=request.query_params.get('offset', 0),
                                      kind=request.query_params.get('kind',''), state=request.query_params.get('state', ''), search=request.query_params.get('search', ''),
                                      since=request.query_params.get('since', -1), grant_id=grant['id'], sort=request.query_params.get('sort','created'), image=queue_image)
            result['items'] = [_public_item(i) for i in result['items']]
            return JSONResponse({**result, 'name': grant['name'], 'shared_count':len(grant['scope']['items'])})
        if action == 'selection' and request.method == 'GET':
            return JSONResponse(store.list_items(root, ids=grant['scope']['items'], kind=request.query_params.get('kind',''),
                state=request.query_params.get('state',''),search=request.query_params.get('search',''),ids_only=True,image=queue_image))
        if action == 'images' and request.method == 'GET':
            columns=grant['scope'].get('context',{}).get('metadata_columns',[])
            images=[{'roi':_alias(grant['id'],registry.register(path)), 'image':value['image'],
                     'metadata':{k:v for k,v in value.get('metadata',{}).items() if k in columns}}
                    for path,value in shared_images(root,grant).items()]
            return JSONResponse({'images':images,'metadata_columns':columns,
                                 'counts':store.list_items(root,ids=grant['scope']['items'],limit=1)['counts']})
        if action == 'image_preview' and request.method == 'GET':
            image=next((path for path in shared_images(root,grant)
                if _alias(grant['id'],registry.register(path))==request.query_params.get('image')),None)
            if not image:
                raise PermissionError('This image was not shared.')
            result=await run_in_threadpool(image_review_previews,root,image,grant['scope']['items'])
            for feature in result['features']:
                feature['properties']['review_source']=_alias(grant['id'],feature['properties']['review_source'])
            return JSONResponse(result)
        if action == 'view_image' and request.method == 'GET':
            chosen=next(((path,value) for path,value in shared_images(root,grant).items()
                if _alias(grant['id'],registry.register(path))==request.query_params.get('id')),None)
            if not chosen:
                raise PermissionError('This image was not shared.')
            path,value=chosen
            spec=await run_in_threadpool(view_spec,root,{'id':registry.register(path),'kind':'image',
                'image':value['image'],'target':{'image':path}},reviewer=True,
                cell_context=grant['scope'].get('context',{}).get('cells',{}).get(path,[]))
            return JSONResponse({'spec':_project_spec(spec,grant['id'])})
        if action == 'decision' and request.method == 'POST':
            body = await request.json()
            result = store.decide(root, body['id'], version=body['version'], action=body['decision'],
                label=body.get('label', ''), note=body.get('note', ''), grant=(grant['id'], secret))
            return JSONResponse(_public_item(result))
        if action == 'decisions' and request.method == 'POST':
            body = await request.json()
            results = store.decide_many(root, body.get('items'), action=body['decision'],
                label=body.get('label', ''), note=body.get('note'), grant=(grant['id'], secret))
            return JSONResponse({'items': [_public_item(item) for item in results]})
        if action == 'label_options' and request.method == 'POST':
            body = await request.json()
            ids = body.get('ids', [])
            if not isinstance(ids, list) or not ids or not set(ids).issubset(grant['scope']['items']):
                raise PermissionError('These items were not shared with this link.')
            result = await run_in_threadpool(suggestion_options, root, ids)
            result['items'] = [_public_item(item) for item in result['items']]
            return JSONResponse(result)
        if action in ('preview', 'select_review_polygons') and request.method == 'POST':
            body = await request.json()
            if body.get('focus') not in grant['scope']['items'] or not set(body.get('items', [])).issubset(grant['scope']['items']):
                raise PermissionError('These review items were not shared.')
            preview = await run_in_threadpool(review_previews, root, body.get('items', []), body['focus'],
                all_in_image=action=='select_review_polygons' or body.get('all_in_image',False),allowed=grant['scope']['items'])
            if action == 'select_review_polygons':
                return JSONResponse(await run_in_threadpool(components.local_area_selection, preview['features'], body.get('points')))
            for feature in preview['features']:
                feature['properties']['review_source']=_alias(grant['id'],feature['properties']['review_source'])
            return JSONResponse(preview)
        if action in ('result_context', 'nominate_result'):
            from spacemap_gui.studio import review_results
            body = await request.json() if request.method == 'POST' else request.query_params
            if body.get('id') not in grant['scope']['items']:
                raise PermissionError('This result was not shared.')
            item = store.get(root, body['id'])
            if not item['target'].get('membership'):
                raise ValueError('Choose a saved result review.')
            if action == 'result_context' and request.method == 'GET':
                return JSONResponse(await run_in_threadpool(review_results.context, root, item))
            if action == 'nominate_result' and request.method == 'POST':
                found = await run_in_threadpool(review_results.nominate, root, item, body.get('groups'), grant['id'])
                return JSONResponse({'items': [_public_item(i) for i in found['items']]})
        if action in ('components', 'nominate', 'polygon'):
            body = await request.json() if request.method == 'POST' else request.query_params
            item_id = body.get('item') or request.query_params.get('item')
            if item_id not in grant['scope']['items']:
                raise PermissionError('This annotation source was not shared.')
            item = store.get(root, item_id)
            if item['kind'] != 'annotation':
                raise ValueError('Choose an annotation review item.')
            target = item['target']; path = components.effective_path(root,target['image'],target['source'])
            if action == 'polygon' and request.method == 'GET':
                matches = await run_in_threadpool(store.annotation_items, root, grant['scope']['items'],
                    image=target['image'], source=target['source'], polygon=body.get('id', ''))
                return JSONResponse({'items': [_public_item(found) for found in matches]})
            if action == 'components' and request.method == 'GET':
                return await run_in_threadpool(component_read, path, body, target['image'])
            if action == 'components' and request.method == 'POST':
                if body.get('action') == 'area':
                    return JSONResponse(await run_in_threadpool(components.area_selection, path, body.get('points'), body.get('hidden', [])))
                if body.get('action') == 'outline':
                    return JSONResponse(await run_in_threadpool(components.selection,path,body.get('ids',[])))
                if body.get('action') == 'select':
                    rows=await run_in_threadpool(components.selected,path,name=body.get('name_filter'),search=body.get('search',''))
                    return JSONResponse({'ids':[r['id'] for r in rows]})
                raise PermissionError('Reviewers can select polygons and suggest decisions; they cannot edit this source.')
            if action == 'nominate' and request.method == 'POST':
                rows=components.selected(path,ids=body.get('ids') or [])
                if not rows or len(rows)>100:
                    raise ValueError('Select between one and 100 polygons to suggest for review.')
                revision=components.index(path)['revision']; nominations=[]
                for row in rows:
                    nominations.append(store.add(root,kind='annotation',image=item['image'],title=row['name']+' · '+row['id'][:12],
                        target={**target,'ids':[row['id']],'revision':revision,'fingerprints':{row['id']:row['fingerprint']},'bounds':row['bounds']},
                        suggestions=body.get('suggestions') or [],note=body.get('note') or '',requested_by=grant['id']))
                return JSONResponse({'items':[_public_item(i) for i in nominations]})
        if action == 'view' and request.method == 'GET':
            item_id = request.query_params['id']
            if item_id not in grant['scope']['items']:
                raise PermissionError('This item was not shared.')
            item=store.get(root,item_id)
            spec=await run_in_threadpool(view_spec, root, item, reviewer=True,
                cell_context=grant['scope'].get('context',{}).get('cells',{}).get(item['target']['image'],[]))
            return JSONResponse({'spec':_project_spec(spec,grant['id'])})
        if action.startswith('data/') and request.method=='POST':
            return await run_in_threadpool(review_data,root,grant,action[5:],await request.json())
        if action == 'select_cells' and request.method == 'POST':
            from spacemap_gui.studio.view_selection import cells_in_area
            body = await request.json()
            permitted = _resources(root, grant['id'], json.dumps(grant['scope'], sort_keys=True))
            token = permitted['cells'].get(body.get('token'))
            if not token:
                raise PermissionError('This cell layer was not shared.')
            return JSONResponse(await run_in_threadpool(cells_in_area, registry.resolve_object(token, 'cells'), body))
        # No write-through proxy. Only specific read operations, bound to images
        # and source layers belonging to the explicit grant's item snapshot.
        match = re.fullmatch(r'(image|annotations|cells)/([^/]+)/(.*)', action)
        if request.method != 'GET' or not match:
            raise PermissionError('This action is not available in a review link.')
        kind, token, suffix = match.groups()
        allow = {'image': r'(info|tile/\d+/\d+/\d+\.(png|jpg)|histogram|limits|pixels|display_range)',
                 'annotations': r'(legend|tile/\d+/\d+/\d+\.png|at)',
                 'cells': r'(info|columns|filter|legend|tile/\d+/\d+/\d+\.png|at|outline|ready)'}
        if not re.fullmatch(allow[kind], suffix):
            raise PermissionError('This resource was not shared.')
        permitted = _resources(root, grant['id'], json.dumps(grant['scope'],sort_keys=True))
        if token not in permitted[kind]:
            raise PermissionError('This layer was not shared with this link.')
        action=kind+'/'+permitted[kind][token]+'/'+suffix
        from .api import routes
        from starlette.routing import Match
        from starlette.requests import Request
        scope = {**request.scope, 'path': '/' + action, 'root_path': '', 'raw_path': ('/' + action).encode(), 'path_params': {}}
        for route in routes:
            matched, child = route.matches(scope)
            if matched == Match.FULL and hasattr(route, 'endpoint'):
                forwarded = Request({**scope, **child}, receive=request.receive)
                import inspect
                if inspect.iscoroutinefunction(route.endpoint):
                    return await route.endpoint(forwarded)
                return await run_in_threadpool(route.endpoint, forwarded)
        raise PermissionError('This review resource is unavailable.')
    except PermissionError as exc:
        return JSONResponse({'error': str(exc)}, status_code=403)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


def install(routes):
    routes.extend([
        Route('/studio/{channel}/review', owner, methods=['GET', 'POST']),
        Route('/studio/{channel}/review/components/{token}', component_route, methods=['GET', 'POST']),
        Route('/review-public/{access}/', public, methods=['GET']),
        Route('/review-public/{access}/{action:path}', public, methods=['GET', 'POST', 'PUT', 'DELETE']),
    ])


def review_data(root, grant, action, body):
    """Read the authorized image slice; never accept a caller's path or ROI scope."""
    from spacemap_gui.studio import data_view as data, data_editor
    from spacemap_gui.studio.cell_links import table_layer
    if action not in ('info','query','summary','find','labels'):
        raise PermissionError('Shared cell data is read-only.')
    source=body.get('source') or {}
    permitted=_resources(root,grant['id'],json.dumps(grant['scope'],sort_keys=True))
    token=permitted['cells'].get(source.get('cells'))
    if not token:
        raise PermissionError('This cell table was not shared.')
    bound=table_layer(registry.resolve_object(token,'cells'),str(source.get('table','')))
    path=bound.get('tables',{}).get(source.get('table'))
    if not path:
        raise PermissionError('This cell table was not shared.')
    column,value=bound.get('roi_column','batch_id'),bound.get('roi')
    # A missing ROI is only safe for explicitly single-image tables.
    from spacemap import dataio
    names=dataio.columns(path)
    if column not in names:
        column=''
    elif value is None or str(value)=='':
        raise PermissionError('The shared cell table needs an image assignment.')
    filters=body.get('filters') or []
    if not isinstance(filters,list) or any(not isinstance(f,dict) or any(k.startswith('_') for k in f) for f in filters):
        raise ValueError('Choose ordinary column filters.')
    sort=body.get('sort')
    label=bound.get('label_column','label')
    try:
        if action=='info':
            found=data.info(path,column)
            count=data.query(path,column,value,[],None,0,1,[label] if label in names else names[:1])['total']
            found.update(rows=count,values=[str(value)] if value is not None else [],default=str(value or ''),
                         image_column=column,roi_column=column,label_column=label,cell_id_column=bound.get('cell_id_column','cell_id'))
        elif action=='query':
            found=data.query(path,column,value,filters,sort,body.get('offset',0),min(200,int(body.get('limit',200))),body.get('columns'))
            ids=[r[0] for r in found['rows']]
            labels=data.values_at_rows(path,ids,label) if label in names else [None]*len(ids)
            found['identity']={str(i):{'label':data._cell(v),'roi':value} for i,v in zip(ids,labels)}
        elif action=='summary':
            found=data.summary(path,column,value,filters,str(body.get('column','')),min(5000,int(body.get('limit',30))),min(200,int(body.get('bins',24))))
        elif action=='find':
            found={'offset':data.find(path,column,value,filters,sort,int(body.get('label',-1)),label,column,value)}
        else:
            found={'labels':data.labels_of(path,column,value,filters,label)}
        # Local paths/editor metadata and unrelated image identifiers stay private.
        for key in ('path','source','progress','undo','redo'):
            found.pop(key,None)
        found.update(read_only=True,edited=0,can_undo=False,can_redo=False)
        return JSONResponse(found)
    except data.Preparing as exc:
        return JSONResponse({'preparing':True,'message':'Preparing shared cell data…'})
    except data_editor.ExternalChange:
        return JSONResponse({'error':'The shared table changed. Reopen Data.'},status_code=409)
