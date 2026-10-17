"""Create a new annotation from the difference of saved layers or labels."""
import logging

from shapely.errors import GEOSException
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from . import registry


def _batch(request, root, config, body):
    from spacemap_gui.studio import annotation, jobs

    images, label = body.get('images'), body.get('label', '')
    if not isinstance(images, list) or not images or not all(isinstance(name, str) for name in images):
        raise ValueError('Tick the images first.')
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > 60:
        raise ValueError('Enter a result label (up to 60 characters), for example Stroma.')
    sides = {}
    for side in ('base', 'subtract'):
        items = body.get(side)
        if not isinstance(items, list) or not items or not all(isinstance(i, dict) and isinstance(i.get('name'), str) for i in items):
            raise ValueError('Choose at least one annotation on each side of the minus sign.')
        sides[side] = [{'name': i['name'], 'role': str(i.get('role', ''))} for i in items]
    mappings = {**(registry.session_project(request.path_params['channel']) or {}).get('mappings', {}), **config.get('directory_mappings', {})}
    folder = annotation.all_folders(root, mappings)['images']
    missing = [name for name in images if '/' in name or not (folder / name).is_file()]
    if missing:
        raise ValueError('Not in the images folder: ' + ', '.join(missing[:4]))
    return {'job': jobs.start(str(root), 'subtract_annotations', {'root': str(root), 'label': label.strip(), **sides,
                                                                   'images': [str(folder / name) for name in dict.fromkeys(images)]})}


async def annotation_difference(request):
    from .api import _open_annotation_image, _studio_project, _json_body
    from spacemap_gui.studio import annotation_difference as difference, annotation_workflow
    try:
        root, config, image, boundary = _open_annotation_image(request)
        output = boundary.parent.parent
        if request.method == 'GET':
            return JSONResponse({'names': await run_in_threadpool(difference.catalog, root, config, image, output)})
        body = await _json_body(request)
        if body.get('action') == 'batch':  # the same subtraction, by name, on each ticked image (a job)
            _studio_project(request, write=True)
            return JSONResponse(await run_in_threadpool(_batch, request, root, config, body))
        if body.get('action', 'save') != 'preview':
            _studio_project(request, write=True)
        if registry.resolve(body.get('image_token', '')) != image:
            raise ValueError('The image changed. Reopen the subtraction section.')
        result = await run_in_threadpool(difference.create, root, config, image, output, body)
        if body.get('action', 'save') == 'save':
            try:
                await run_in_threadpool(annotation_workflow.record, root, image=image, target=result['path'],
                    event={'operation': 'subtract_annotations', 'parameters': result['recipe']})
            except Exception:
                logging.exception('Could not record annotation subtraction')
                result['workflow_warning'] = 'Layer saved; the workflow index could not be updated. Its recipe is in the annotation layer index.'
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError, GEOSException) as error:
        return JSONResponse({'error': str(error)}, status_code=400)
