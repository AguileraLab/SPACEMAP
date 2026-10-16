"""Export the open image's chosen saved annotation layers."""
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from . import registry


async def annotation_export(request):
    from .api import _open_annotation_image, _studio_project, _json_body
    from spacemap_gui.studio import annotation_layers, annotation_workflow
    try:
        root, config, image, boundary = _open_annotation_image(request)
        output = boundary.parent.parent
        from spacemap_gui.studio import annotation_names
        if request.method == 'GET':
            settings = annotation_names.export_settings(root, config)
            return JSONResponse({'names': await run_in_threadpool(annotation_names.groups, root, config, image, output),
                                 'settings': settings, 'file_name': annotation_names.file_name(image, settings),
                                 'folder': str(annotation_names.export_folder(root, settings, config))})
        _studio_project(request, write=True)
        body = await _json_body(request)
        action = body.get('action', 'export')
        if action == 'settings':
            settings = await run_in_threadpool(annotation_names.save_export_settings, root, body.get('changes'))
            return JSONResponse({'settings': settings, 'file_name': annotation_names.file_name(image, settings),
                                 'folder': str(annotation_names.export_folder(root, settings, config))})
        from spacemap_gui.studio import annotation, jobs
        mappings = {**(registry.session_project(request.path_params['channel']) or {}).get('mappings', {}), **config.get('directory_mappings', {})}
        folder = annotation.all_folders(root, mappings)['images']

        def ticked():
            images = body.get('images')
            if not isinstance(images, list) or not images or not all(isinstance(n, str) and '/' not in n and (folder / n).is_file() for n in images):
                raise ValueError('Tick images of the images folder first.')
            return [str(folder / n) for n in dict.fromkeys(images)]
        if action == 'batch':
            job = jobs.start(str(root), 'export_annotations', {'root': str(root), 'images': ticked(),
                                                                'merge': bool(body.get('merge_same_names', True))})
            return JSONResponse({'job': job})
        if action == 'import_matched':  # one of several GeoJSON files: to the image it names, or the one chosen for it
            from spacemap_gui.studio import annotation_inputs
            images = sorted(p for p in folder.iterdir() if p.is_file() and p.name.lower().endswith(annotation.IMAGE_EXTENSIONS))
            if body.get('image'):
                target_image = next((p for p in images if p.name == body['image']), None)
                if target_image is None:
                    raise ValueError('Choose one of the images.')
            else:
                target_image = annotation_names.match_image(images, str(body.get('name', '')))
            if target_image is None:
                return JSONResponse({'matched': None, 'images': [p.name for p in images]})
            if any(p.stem.split('_', 1)[-1] == annotation_names._stem(body.get('name', '')) for p in annotation_inputs.imported(root, target_image)):
                return JSONResponse({'matched': Path(target_image).name, 'already': True, 'open': False, 'layer': None})
            target = await run_in_threadpool(annotation_inputs.import_polygons, root, target_image, body.get('name', 'Imported'), body.get('features'))
            here = Path(target_image).resolve() == Path(image).resolve()
            readonly = bool((registry.session_project(request.path_params['channel']) or {}).get('read_only'))
            return JSONResponse({'matched': Path(target_image).name, 'open': here,
                                 'layer': annotation_inputs.layer(target, target_image, readonly) if here else None})
        if action == 'delete_results':  # a stage's results for the open image, or for each ticked image
            stage = str(body.get('stage', ''))
            targets = ticked() if body.get('images') else [str(image)]
            moved = {}
            for target in targets:
                moved[Path(target).name] = await run_in_threadpool(annotation_names.delete_results, root, config, target, stage)
                await run_in_threadpool(annotation_workflow.record, root, image=target,
                                        event={'operation': 'delete_results', 'parameters': {'stage': stage, 'set_aside': moved[Path(target).name]}})
            return JSONResponse({'moved': moved, 'files': sum(len(v) for v in moved.values()),
                                 'open': any(Path(t).resolve() == Path(image).resolve() for t in targets)})
        if action == 'summary':
            return JSONResponse({'names': await run_in_threadpool(annotation_names.summary, root, config, ticked(), output)})
        if registry.resolve(body.get('image_token', '')) != image:
            raise ValueError('The image changed. Refresh the Export list.')
        if action == 'rename':
            found = await run_in_threadpool(annotation_names.rename, root, config, image, output, body.get('key', ''),
                                            body.get('name'), body.get('colour'))
            from spacemap_gui.studio import annotation_inputs
            found['layers'] = [annotation_inputs.layer(path, image) for path in found['copies']]
            return JSONResponse(found)
        merge = bool(body.get('merge_same_names', True))
        if body.get('layers'):  # the earlier way: chosen files
            result = await run_in_threadpool(annotation_layers.export, root, config, image, output, body['layers'], merge=merge)
        else:  # the names ticked for Export (Layers), also saved under the chosen file name
            naming = annotation_names.export_settings(root, config)
            result = await run_in_threadpool(annotation_names.export, root, config, image, output, merge, naming)
        await run_in_threadpool(annotation_workflow.record, root, image=image, target=result['path'],
            event={'operation': 'export_annotations', 'parameters': {'names': result.get('names'), 'layers': body.get('layers') or result.get('sources'), 'merge_same_names': merge}})
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def annotation_names_route(request):
    """The names ticked for Export for the open image; POST {image_token, choices: {key: bool}} ticks or unticks."""
    from .api import _open_annotation_image, _studio_project, _json_body
    from spacemap_gui.studio import annotation_names
    try:
        root, config, image, boundary = _open_annotation_image(request)
        token = request.query_params.get('image_token')
        if request.method == 'POST':
            _studio_project(request, write=True)
            body = await _json_body(request)
            token = body.get('image_token')
            if registry.resolve(token or '') != image:
                raise ValueError('The image changed. Open it again.')
            return JSONResponse({'choices': await run_in_threadpool(annotation_names.save_choices, root, config, image, boundary.parent.parent, body.get('choices'))})
        if token and registry.resolve(token) != image:
            raise ValueError('The image changed. Open it again.')
        return JSONResponse({'choices': await run_in_threadpool(annotation_names.choices, root, config, image, boundary.parent.parent),
                             'rules': annotation_names.export_settings(root, config)['rules']})
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def annotation_palette(request):
    """The project's hand-given names with their colours (GET); POST {names: {name: colour}, forget: [name]}."""
    from .api import _studio_project, _json_body
    from spacemap_gui.studio import annotation_names
    try:
        root = _studio_project(request, write=request.method == 'POST')
        if request.method == 'POST':
            body = await _json_body(request)
            names = await run_in_threadpool(annotation_names.remember, root, body.get('names'), body.get('forget'))
        else:
            names = await run_in_threadpool(annotation_names.palette, root)
        return JSONResponse({'names': names})
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)


async def annotation_imports(request):
    """Imported GeoJSON files of the Annotation step's images (GET); POST {action: "folder", folder} imports a folder
    in the background, {action: "remove", items: [{image, path}]} takes files off their images."""
    from .api import _studio_project, _json_body
    from spacemap_gui.studio import annotation_names, jobs
    from spacemap_gui.studio.settings import read_config
    try:
        root = Path(_studio_project(request, write=request.method == 'POST'))
        config = read_config(root)
        # The folders as the Annotation step sees them: the session's, with the project's own on top.
        mappings = {**(registry.session_project(request.path_params['channel']) or {}).get('mappings', {}), **(config.get('directory_mappings') or {})}
        config = {**config, 'directory_mappings': mappings}
        if request.method == 'GET':
            return JSONResponse({'files': await run_in_threadpool(annotation_names.imports, root, config)})
        body = await _json_body(request)
        if body.get('action') == 'folder':
            folder = str(body.get('folder') or '').strip()
            if not folder or len(folder) > 1024:
                raise ValueError('Type the folder of GeoJSON files.')
            return JSONResponse({'job': jobs.start(str(root), 'import_annotations', {'root': str(root), 'folder': folder, 'mappings': mappings})})
        if body.get('action') == 'remove':
            removed = await run_in_threadpool(annotation_names.remove_imports, root, config, body.get('items'))
            return JSONResponse({'removed': removed})
        raise ValueError('Choose what to do with the imported files.')
    except (ValueError, OSError, TypeError, KeyError) as error:
        return JSONResponse({'error': str(error)}, status_code=400)
