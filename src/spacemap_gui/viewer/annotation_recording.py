"""Record successful annotation operations without changing their save semantics."""
from functools import wraps
import json
import logging
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import request_response

from . import registry
from spacemap_gui.studio import annotation_workflow


ENDPOINTS = frozenset({'classification', 'review', 'annotation_sources', 'tissue_versions', 'shapes_save', 'shapes_brush',
    'shapes_polygon_edit', 'shapes_edit', 'shapes_history', 'shapes_import', 'studio_tissue_settings',
    'studio_tissue_preview', 'studio_artifact_settings', 'studio_artifact_preview',
    'studio_artifact_example', 'studio_pieces_settings', 'studio_pieces_preview',
    'studio_annotation_images', 'studio_marker_regions', 'studio_zone_rules', 'studio_zones_preview'})
# Scalar settings and geometric operation parameters only. Saved geometry is in
# the GeoJSON/history; preview output is not silently committed by this index.
PARAMETERS = frozenset({'action', 'op', 'part', 'version', 'base_version', 'revision', 'example_revision',
    'source_revision', 'class_name', 'stage', 'selection', 'scale', 'minimum_um2', 'cut',
    'value', 'values', 'changes', 'detector_values', 'detectors', 'use_reviewed', 'remaining_good', 'capture_current',
    'width', 'gap', 'radius', 'mode', 'preview', 'ids', 'id', 'line', 'points', 'cutout', 'name', 'label'})


def recorded(endpoint):
    @wraps(endpoint)
    async def call(request):
        if request.method != 'POST':
            return await endpoint(request)
        # Resolve before awaiting the request: another window can change the
        # session's active image while a prediction is running.
        project = registry.session_project(request.path_params.get('channel', ''))
        layer = registry.resolve_object(request.path_params.get('token', ''), 'shapes')
        try:
            body = await request.json()
        except (ValueError, TypeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        image = registry.resolve(body.get('image_token', '')) or (layer or {}).get('image')
        if not image and project:
            view = registry.session_view(request.path_params['channel'], mark_seen=False) or {}
            image = next((registry.resolve(item['token']) for item in (view.get('spec') or {}).get('layers', [])
                          if item.get('type') == 'image'), None)
        projects = [project] if project else registry.annotation_projects(image) if image else []
        response = await endpoint(request)
        if 200 <= response.status_code < 300 and not (endpoint.__name__=='classification' and body.get('action') not in ('review','save')):
            parameters = {key: body[key] for key in PARAMETERS if key in body}
            source = registry.resolve_object(body.get('source_token', ''), 'annotations') or registry.resolve_object(body.get('source_token', ''), 'shapes')
            if source:
                parameters['source_file'] = source['path']
            result = json.loads(response.body) if isinstance(response, JSONResponse) else {}
            event = {'operation': endpoint.__name__, 'parameters': parameters,
                     'part': request.path_params.get('part'), 'action': request.path_params.get('action'),
                     'result': {key: result[key] for key in ('revision', 'source', 'count', 'version', 'job') if key in result}}
            for context in projects:
                if not context.get('root') or context.get('read_only'):
                    continue
                try:
                    await run_in_threadpool(annotation_workflow.record, context['root'], event=event,
                        image=image, target=(layer or {}).get('path'), mappings=context.get('mappings'))
                except Exception:
                    # Do not claim the polygon save failed after it committed.
                    # Surface the separate recording failure and retain a log.
                    logging.exception('Could not update annotation workflow record')
                    if isinstance(response, JSONResponse):
                        result['workflow_warning'] = 'Changes saved, but the workflow record could not be updated. Retry from Export → Workflow record.'
                        response = JSONResponse(result, status_code=response.status_code)
        return response
    return call


def install(routes):
    for route in routes:
        if getattr(getattr(route, 'endpoint', None), '__name__', '') in ENDPOINTS:
            route.app = request_response(recorded(route.endpoint))


async def workflow_record(request):
    from .api import _studio_project
    try:
        root = _studio_project(request, write=request.method == 'POST')
        project = registry.session_project(request.path_params['channel']) or {}
        path = Path(root) / annotation_workflow.RELATIVE_PATH
        if request.method == 'POST':
            from .api import _json_body, _open_annotation_image
            body = await _json_body(request)
            event = body.get('event')
            image = None
            if event:
                if not isinstance(event, dict) or event.get('operation') not in ('reject_neck', 'restore_necks'):
                    raise ValueError('Choose a neck review action.')
                _, _, image, _ = _open_annotation_image(request)
                if registry.resolve(body.get('image_token', '')) != image:
                    raise ValueError('The image changed. Review its necks again.')
            await run_in_threadpool(annotation_workflow.record, root, mappings=project.get('mappings'), event=event, image=image)
            if event:
                return JSONResponse({'recorded': True})
        if not path.is_file():
            return JSONResponse({'error': 'Save annotation changes or create the workflow record first.'}, status_code=404)
        return JSONResponse(json.loads(path.read_text()), headers={
            'Content-Disposition': 'attachment; filename="annotation_workflow.json"', 'Cache-Control': 'no-store'})
    except (ValueError, OSError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)
