"""Lazy project inventory and file actions for Studio's Lists tab."""
import json
from pathlib import Path
import tempfile

from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse

from . import registry


def context(request, write=False):
    from .api import _studio_project
    from spacemap_gui.studio.settings import read_config
    root = Path(_studio_project(request, write=write))
    project = registry.session_project(request.path_params['channel']) or {}
    config = read_config(root)
    config['directory_mappings'] = {**project.get('mappings', {}), **config.get('directory_mappings', {})}
    return root, config, project


async def lists(request):
    from .api import _json_body
    from spacemap_gui.studio import inventory
    try:
        if request.method == 'GET':
            root, config, project = context(request)
            if not request.query_params.get('step'):
                from spacemap_gui.studio.shell import STEPS
                return JSONResponse({'steps': [{'id':s[0], 'name':s[1]} for s in STEPS], 'read_only': bool(project.get('read_only'))})
            return JSONResponse(await run_in_threadpool(inventory.listing, root, config,
                request.query_params['step'], request.query_params.get('group'), project.get('unified')))
        body = await _json_body(request)
        action = body.get('action')
        root, config, project = context(request, write=action in ('remove', 'import', 'match'))
        step, group = body.get('step'), body.get('group')
        if action == 'import':
            paths = body.get('paths')
            if not isinstance(paths, list) or not paths or len(paths) > 5000:
                raise ValueError('Enter file paths, one per line.')
            return JSONResponse(await run_in_threadpool(inventory.import_paths, root, config, step, group, paths, project.get('unified'), body.get('image', '')))
        rows = await run_in_threadpool(inventory.selected, root, config, step, group, body.get('ids'), project.get('unified'))
        if action == 'match':
            if group != 'imports' or len(rows) != 1:
                raise ValueError('Choose one unmatched import.')
            return JSONResponse(await run_in_threadpool(inventory.assign_import, root, config, rows[0], body.get('image')))
        if action == 'remove':
            result = await run_in_threadpool(inventory.remove, root, config, rows)
            return JSONResponse(result)
        if action == 'export':
            # Downloads are streamed; large images are not copied into RAM.
            token = registry.register_object('list_download', root=str(root), rows=rows)
            return JSONResponse({'download': '/studio/' + request.path_params['channel'] + '/lists/download/' + token})
        if action == 'open':
            if len(rows) != 1:
                raise ValueError('Open one item at a time.')
            row = rows[0]
            if not Path(row['path']).is_file():
                raise ValueError('This linked file is no longer at its saved path.')
            from . import ui
            if row.get('image') and Path(row['image']).is_file():
                images = await run_in_threadpool(ui.window_images, request.path_params['channel'])
                entries = registry.session_images(request.path_params['channel']) or []
                image_id = next((i for i,e in enumerate(entries) if Path(e['path']).resolve() == Path(row['image']).resolve()), None)
                if image_id is None:
                    added = await run_in_threadpool(ui.window_add_image, request.path_params['channel'], row['image'])
                    images, image_id = added['images'], added['added']
                row = {**row, 'viewer_image': next(i for i in images if i['id'] == image_id)}
            if row['kind'] == 'table':
                row = {**row, 'file_token': registry.register_object('datafile', path=row['path'])}
                if row.get('method'):
                    registry.session_use_table(request.path_params['channel'], row['path'])
            if Path(row['path']).suffix.lower() in inventory.FIGURE:
                token = registry.register_object('list_download', root=str(root), rows=[row])
                row = {**row, 'preview': '/studio/' + request.path_params['channel'] + '/lists/download/' + token + '?inline=1'}
            if row['kind'] == 'annotation' and row.get('image'):
                from .api import _annotation_layer
                if row.get('annotation_kind') == 'imported':
                    from spacemap_gui.studio import annotation_inputs
                    layer = annotation_inputs.layer(row['path'], row['image'], bool(project.get('read_only')))
                else:
                    from spacemap_gui.studio import annotation_inputs
                    layer = _annotation_layer(row['path'], row['image'], registry.register(row['image']), row['name'])
                    layer.update(annotation_inputs.display_metadata(root, row['image'], row['path']))
                row = {**row, 'annotation_layer': layer}
            return JSONResponse({'item': row})
        raise ValueError('Choose a Lists action.')
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def upload(request):
    from spacemap_gui.studio import inventory
    temporary = None
    try:
        root, config, project = context(request, write=True)
        name = request.query_params.get('name', '')
        with tempfile.NamedTemporaryFile(prefix='spacemap-list-', delete=False) as handle:
            temporary = Path(handle.name)
            async for chunk in request.stream():
                handle.write(chunk)
        result = await run_in_threadpool(inventory.import_files, root, config, request.query_params.get('step'),
            request.query_params.get('group'), [(temporary, name)], project.get('unified'), request.query_params.get('image', ''))
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def download(request):
    from spacemap_gui.studio import inventory
    try:
        root, config, project = context(request)
        found = registry.resolve_object(request.path_params['token'], 'list_download')
        if not found or found['root'] != str(root):
            raise ValueError('Select files to export again.')
        rows = found['rows']
        if any(not Path(row['path']).is_file() for row in rows):
            raise ValueError('A selected file has been removed. Refresh Lists.')
        if len(rows) == 1 and not rows[0].get('model_kind') and not any(
            Path(rows[0]['path'] + suffix).exists() for suffix in ('.pending', '.labels.parquet')):
            return FileResponse(rows[0]['path'], filename=Path(rows[0]['path']).name,
                                content_disposition_type='inline' if request.query_params.get('inline') else 'attachment')
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as handle:
            target = Path(handle.name)
        try:
            await run_in_threadpool(inventory.export_bundle, rows, target)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return FileResponse(target, filename=(Path(rows[0]['path']).stem if len(rows)==1 else 'Studio-files') + '.zip',
                            background=BackgroundTask(target.unlink, missing_ok=True))
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def storage(request):
    from spacemap_gui.studio import storage as disk
    from .api import _json_body
    try:
        root, config, project = context(request, write=request.method == 'POST')
        if request.method == 'GET':
            if request.query_params.get('caches') == '1':
                value = await run_in_threadpool(disk.caches, root)
                return JSONResponse({**value, 'read_only': bool(project.get('read_only'))})
            plan = await run_in_threadpool(disk.preview, root)
            token = registry.register_object('storage_plan', root=str(root), plan=plan)
            return JSONResponse({'token': token, 'groups': plan['groups'], 'retained': plan['retained'],
                                 'items': [{k:i[k] for k in ('id', 'category', 'name', 'path', 'bytes', 'purpose', 'impact')} for i in plan['items']],
                                 'read_only': bool(project.get('read_only'))})
        body = await _json_body(request)
        stored = registry.resolve_object(body.get('token', ''), 'storage_plan')
        if not stored or stored['root'] != str(root):
            raise ValueError('Review temporary files again before cleaning.')
        return JSONResponse(await run_in_threadpool(disk.clean, root, stored['plan'], body.get('categories'), body.get('ids')))
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)
