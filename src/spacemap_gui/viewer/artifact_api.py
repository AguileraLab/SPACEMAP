"""Artifact review and model routes, always bound to the open image/project."""
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from . import registry


def _layer(output, sample, part, image, readonly=False):
    from .ui import shapes_layer
    from spacemap_gui.studio import annotation, artifact_learning as learning
    result = shapes_layer(learning.polygons_path(output, sample, part), image,
                          name=learning.NAMES[part], layer_id="artifact-" + part, regions=True,
                          annotation_kind="artifacts", editable=not readonly,
                          classes=[{"name": learning.CLASSES[part], "color": learning.COLOURS[part]}],
                          default_class=learning.CLASSES[part], reference=annotation.valid_tissue(output, sample))
    result["artifact_part"] = part
    return result


def _examples_layer(root, part, image, readonly=False):
    from .ui import shapes_layer
    from spacemap_gui.studio import annotation_edits, artifact_learning as learning
    classes = [{"name": v["name"], "color": '#' + ''.join(f'{c:02x}' for c in v['color'])}
               for v in [annotation_edits.EXAMPLE_CLASSES['negative']]]
    result = shapes_layer(learning.example_path(root, part, image), image,
        name=learning.NAMES[part] + (' · Sharp tissue' if part == 'blur' else ' · Good glass' if part == 'lint' else ' · Good tissue'), layer_id='artifact-examples-' + part,
        regions=True, annotation_kind='examples', editable=not readonly, classes=classes,
        default_class='Good tissue', visible=False)
    result.update(training_part=part, annotation_group='artifacts', added=True)
    return result


def _batch(root, mappings, part, body):
    """Find (and/or subtract) this part's artifacts in each ticked image as a job."""
    from spacemap_gui.studio import annotation, artifact_learning as learning, jobs

    images = body.get("images")
    if body.get("open_image") and isinstance(body.get("open_image"), str):
        images = [Path(body["open_image"]).name]
    if not isinstance(images, list) or not images or not all(isinstance(name, str) for name in images):
        raise ValueError("Tick the images first.")
    find, subtract = body.get("find") is True, body.get("subtract") is True
    parts = body.get("parts") or [part]
    if not isinstance(parts, list) or not parts or any(p not in learning.NAMES for p in parts) or (find and parts != [part]):
        raise ValueError("Choose Blood clots and/or Image quality.")
    if not (find or subtract):
        raise ValueError("Choose Find or Subtract.")
    folder = annotation.all_folders(root, mappings)["images"]
    missing = [name for name in images if "/" in name or not (folder / name).is_file()]
    if missing:
        raise ValueError("Not in the images folder: " + ", ".join(missing[:4]))
    version = body.get("version") or ""
    if version and version not in {v["id"] for v in learning.state(root, part)["versions"] if not v.get("deleted_at")}:
        raise ValueError("That model is not available any more. Choose another.")
    if find and part == "clot" and not version:
        raise ValueError("Choose a Blood clots model first.")
    settings = {"root": str(root), "part": part, "parts": list(dict.fromkeys(parts)), "version": version, "find": find, "subtract": subtract,
                "replace": body.get("replace") is True, "images": [str(folder / name) for name in dict.fromkeys(images)]}
    return {"job": jobs.start(str(root), "artifact_batch", settings)}


def _train_batch(root, mappings, part, body):
    """Train one model on the reviewed labels of each ticked image (and the base model's other images), as a job."""
    from spacemap_gui.studio import annotation, artifact_learning as learning, jobs

    images = body.get("images")
    if not isinstance(images, list) or not all(isinstance(name, str) for name in images):
        raise ValueError("Tick the images first.")
    folders = annotation.all_folders(root, mappings)
    missing = [name for name in images if "/" in name or not (folders["images"] / name).is_file()]
    if missing:
        raise ValueError("Not in the images folder: " + ", ".join(missing[:4]))
    saved = learning.options(root, part)
    scale = body.get("scale", saved["scale"])
    if type(scale) is not int or scale not in learning.SCALES:
        raise ValueError("Choose one of the detection scales.")
    remaining = body.get("remaining_good", saved["remaining_good"])
    settings = learning.training_request(root, part, base_version=body.get("base_version") or None, allow_empty=True)
    captures, skipped = learning.batch_captures(root, part, [folders["images"] / name for name in dict.fromkeys(images)],
                                                folders["output"], scale, remaining)
    if not captures:
        raise ValueError("None of the ticked images has been reviewed: " + "; ".join(skipped[:4]))
    settings.update(captures=captures, skipped=skipped)
    return {"job": jobs.start(str(root), "train_artifact_model", settings), "images": len(captures), "skipped": skipped,
            "examples": learning.state(root, part)["examples"]}


async def review(request):
    from .api import _open_annotation_image, _json_body, _studio_project
    from spacemap_gui.studio import annotation, annotation_edits, artifact_learning as learning, jobs
    try:
        part = learning.part_name(request.path_params["part"])
        action = request.path_params["action"]
        root, config, image, boundary = _open_annotation_image(request)
        output, sample = boundary.parent.parent, annotation.sample_id(image)
        readonly = bool((registry.session_project(request.path_params["channel"]) or {}).get("read_only"))
        descriptor = _layer(output, sample, part, image, readonly)
        if request.method == "GET" and action == "state":
            from spacemap.viewer.sources import image_info
            info = await run_in_threadpool(image_info, image)
            return JSONResponse({**learning.state(root, part), "layer": descriptor, "read_only": readonly,
                                 "example_layer": _examples_layer(root, part, image, readonly),
                                 "microns_per_pixel": info.pixel_size_um or annotation.settings(config).get("microns_per_pixel", .5083)})
        body = await _json_body(request)
        if registry.resolve(body.get("image_token", "")) != image:
            raise ValueError("The image changed. Reopen the artifact section and try again.")
        if action == 'batch':
            _studio_project(request, write=True)
            mappings = {**(registry.session_project(request.path_params["channel"]) or {}).get("mappings", {}),
                        **config.get("directory_mappings", {})}
            return JSONResponse(await run_in_threadpool(_batch, root, mappings, part, {**body, 'open_image': str(image)} if body.get('open_image') else body))
        if action == 'train' and body.get('images'):
            _studio_project(request, write=True)
            mappings = {**(registry.session_project(request.path_params["channel"]) or {}).get("mappings", {}),
                        **config.get("directory_mappings", {})}
            return JSONResponse(await run_in_threadpool(_train_batch, root, mappings, part, body))
        if action == 'subtract':
            _studio_project(request, write=True)
            from spacemap_gui.studio import annotation_inputs as inputs
            target = inputs.working_layer(root, config, image, output, part)
            if target['token'] != body.get('tissue_token'):
                raise ValueError('Work within changed. Review the selected annotation and subtract again.')
            selected = inputs.selection(root, image, part)
            if selected != body.get('selection'):
                raise ValueError('Work within changed. Review the selected annotation and subtract again.')
            def subtract():
                before, version = annotation_edits.read(target['saved_file'], kind=target['annotation_kind'],
                                                       label=annotation.tissue_label(config))
                if version != body.get('tissue_revision'):
                    raise ValueError('Tissue changed. Wait for its edits to save, then subtract again.')
                # Only this tab's reviewed polygons are applied. Later tabs operate
                # on the saved result; inactive detector outputs are not subtracted.
                path = descriptor['saved_file']
                result = annotation_edits.subtract(target['saved_file'], [path], version,
                    kind=target['annotation_kind'], class_name=selected.get('class_name',''),
                    label=annotation.tissue_label(config), artifact_revisions={path:body.get('reviewed_revision')})
                return {**result, 'before':annotation_edits.display(before,version)['features'], 'layer':target}
            return JSONResponse(await run_in_threadpool(subtract))
        if action in ('select_model', 'delete_model', 'restore_model'):
            _studio_project(request, write=True)
            return JSONResponse(await run_in_threadpool(learning.manage_model, root, part, action, body.get('version')))
        if action == "settings":
            _studio_project(request, write=True)
            return JSONResponse(await run_in_threadpool(learning.save_options, root, part, body.get("values"), body.get("revision")))
        if action == "forget":
            _studio_project(request, write=True)
            return JSONResponse(await run_in_threadpool(learning.forget_image, root, part, image))
        if action == "open_examples":
            _studio_project(request, write=True)
            await run_in_threadpool(learning.open_examples, root, part, image)
            moved = await run_in_threadpool(learning.combine_examples, root, part, image, output,
                                           preview_active=bool(body.get('artifact_preview')))
            return JSONResponse({'layer': _examples_layer(root, part, image), 'combined_red': moved})
        if action in ('mark_good', 'mark_artifact'):
            _studio_project(request, write=True)
            result = await run_in_threadpool(learning.convert_examples, root, part, image, output,
                body.get('ids', [body.get('id')]), 'negative' if action == 'mark_good' else 'positive',
                body.get('revision'), body.get('example_revision'),
                preview=body.get('preview_features'))
            return JSONResponse({**result, 'example_layer': _examples_layer(root, part, image), 'artifact_layer': descriptor})
        if action == "train" and not body.get('capture_current'):
            _studio_project(request, write=True)
            settings = await run_in_threadpool(learning.training_request, root, part, base_version=body.get('base_version'))
            return JSONResponse({"job": await run_in_threadpool(jobs.start, root, "train_artifact_model", settings),
                                 'examples': learning.state(root, part)['examples']})
        if action == 'train' and body.get('base_version'):
            # Fail before sampling or updating the current library if the base
            # version was deleted or its archived training files are missing.
            await run_in_threadpool(learning.training_request, root, part, base_version=body['base_version'])
        from spacemap_gui.studio import annotation_inputs
        scope = await run_in_threadpool(annotation_inputs.scope_path, root, image, output, part)
        tissue, tissue_revision = await run_in_threadpool(annotation_edits.read, scope, kind="tissue")
        if not tissue:
            raise ValueError("Save the selected Tissue or annotation polygons first.")
        saved = learning.options(root, part)
        scale = body.get("scale", saved["scale"])
        if type(scale) is not int or scale not in learning.SCALES:
            raise ValueError("Choose one of the detection scales.")
        minimum = body.get("minimum_um2", saved["minimum_um2"])
        if isinstance(minimum, bool) or not isinstance(minimum, (float, int)) or not 0 <= minimum <= 1e8:
            raise ValueError("Smallest region must be between 0 and 100,000,000 µm².")
        if action in ("detect", "predict"):
            if action == "detect":
                result = await run_in_threadpool(learning.detect, image, config, part, scale, body.get("value", saved.get('value')), tissue, minimum, scope_path=scope, fill_holes_um2=float(body.get("fill_holes_um2", saved.get("fill_holes_um2", 0)) or 0),
                    detector_values=body.get('detector_values', saved.get('detector_values')),
                    detectors=body.get('detectors', saved.get('detectors')))
            else:
                cut = body.get("cut", saved["cut"])
                if isinstance(cut, bool) or not isinstance(cut, (int, float)) or not .05 <= cut <= .95:
                    raise ValueError("Choose a model cut from 0.05 to 0.95.")
                result = await run_in_threadpool(learning.predict, root, part, body.get("version"), image, scale, tissue, cut, minimum,
                                                fill_holes_um2=float(body.get("fill_holes_um2", saved.get("fill_holes_um2", 0)) or 0))
            manual = body.get('manual_features', [])
            if manual:
                exact, _ = await run_in_threadpool(annotation_edits.read, descriptor['saved_file'], kind='artifacts')
                by_id = {f['properties']['id']: f for f in exact}
                manual = [by_id.get(f.get('properties', {}).get('id'), f) if f.get('properties', {}).get('simplified') else f for f in manual]
                result['features'] = annotation_edits.prepare([*result['features'], *manual], kind='artifacts')
            if annotation_edits.revision(scope) != tissue_revision:
                raise ValueError("The selected annotation changed. Detect again to use the new outline.")
            return JSONResponse({**result, "layer": descriptor})
        if action in ("examples", "train"):
            _studio_project(request, write=True)
            positive = body.get("positive", [])
            negative = body.get("negative", [])
            if body.get('capture_current'):
                drawings, version = await run_in_threadpool(annotation_edits.read,
                    learning.example_path(root, part, image), kind='examples')
                if version != body.get('example_revision'):
                    raise ValueError('Examples changed. Wait for them to save, then Train again.')
                positive = [f for f in drawings if f['properties']['example_role'] == 'positive']
                negative = [f for f in drawings if f['properties']['example_role'] == 'negative']
            if body.get("use_reviewed"):
                if "reviewed_features" in body:
                    reviewed = body["reviewed_features"]
                else:
                    reviewed, version = await run_in_threadpool(annotation_edits.read, descriptor["saved_file"], kind="artifacts")
                    if version != body.get("reviewed_revision"):
                        raise ValueError("Artifact polygons changed. Review them before training.")
                # Green corrections override detections; explicit red/green
                # contradictions are still reported by snapshot.
                from shapely.geometry import mapping
                reviewed_area = learning._geometry(reviewed, 'Reviewed artifact').difference(learning._geometry(negative, 'Green example'))
                positive = [*positive, *[{'type':'Feature', 'properties':{}, 'geometry':mapping(g)}
                            for g in annotation_edits.polygons(reviewed_area)]]
            remaining_good = body.get("remaining_good", saved["remaining_good"])
            if type(remaining_good) is not bool:
                raise ValueError("Choose whether remaining tissue is good tissue.")
            if action == 'train':
                settings = await run_in_threadpool(learning.training_request, root, part,
                    base_version=body.get('base_version'), allow_empty=True)
                # Freeze the reviewed geometry and historical training base now.
                # Image reads, feature extraction and fitting all run in the
                # cancellable worker, so Train responds before expensive work.
                settings['capture'] = {'image': str(image), 'scale': scale,
                    'positives': positive, 'negatives': negative, 'tissue': tissue,
                    'remaining_good': remaining_good}
                return JSONResponse({'job': await run_in_threadpool(jobs.start, root, 'train_artifact_model', settings),
                                     'examples': learning.state(root, part)['examples']})
            result = await run_in_threadpool(learning.snapshot, root, part, image, scale, positive,
                                            negative, tissue, remaining_good)
            return JSONResponse(result)
        raise ValueError("Unknown artifact action.")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


async def tissue_versions(request):
    """Open the one editable annotation selected for this image's workflow."""
    from .api import _open_annotation_image, _json_body, _studio_project
    from spacemap_gui.studio import annotation_inputs as inputs, annotation_edits
    try:
        root, config, image, boundary = _open_annotation_image(request)
        body = await _json_body(request) if request.method == 'POST' else {}
        if body.get('image_token') and registry.resolve(body['image_token']) != image:
            raise ValueError('The image changed. Open its Tissue annotation again.')
        stage = body.get('stage', request.query_params.get('stage', 'tissue'))
        readonly = bool((registry.session_project(request.path_params['channel']) or {}).get('read_only'))
        if request.method == 'POST' and not readonly:
            _studio_project(request, write=True)
            if body.get('canonical'):
                await run_in_threadpool(inputs.save_selection, root, config, image, boundary.parent.parent, stage, {'mode':'tissue','class_name':''})
            await run_in_threadpool(inputs.prepare_working, root, config, image, boundary.parent.parent, stage)
        layer = inputs.working_layer(root, config, image, boundary.parent.parent, stage, readonly)
        return JSONResponse({'layer':layer, 'selection':inputs.selection(root,image,stage),
                             'revision':annotation_edits.revision(layer['saved_file'])})
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({'error': str(exc)}, status_code=400)


async def annotation_sources(request):
    from .api import _open_annotation_image, _json_body, _studio_project
    from spacemap_gui.studio import annotation_inputs as inputs, annotation_edits
    try:
        root, config, image, boundary = _open_annotation_image(request)
        output = boundary.parent.parent
        readonly = bool((registry.session_project(request.path_params['channel']) or {}).get('read_only'))
        if request.method == 'POST':
            _studio_project(request, write=True)
            body = await _json_body(request)
            if registry.resolve(body.get('image_token','')) != image:
                raise ValueError('The image changed. Open its annotations again.')
            if body.get('action') == 'use_as_tissue':
                from .ui import shapes_layer
                from spacemap_gui.studio import annotation
                source = _annotation_source(body.get('source_token'), image)
                features, version = await run_in_threadpool(annotation_edits.read, source['path'])
                if body.get('source_revision') != version:
                    raise ValueError('The source annotations changed. Choose the source again before importing.')
                if 'class_name' not in body:
                    raise ValueError('Choose the annotation to use as Tissue.')
                features = await run_in_threadpool(inputs.tissue_import, image, features, body['class_name'],
                    label=annotation.tissue_label(config), source=source['path'])
                for feature in features:
                    feature['properties']['tissue_source_revision'] = version
                target = annotation.valid_tissue(output, annotation.sample_id(image))
                descriptor = shapes_layer(target, image, name=annotation.tissue_label(config), layer_id='tissue-regions',
                    regions=True, annotation_kind='tissue', tissue_label=annotation.tissue_label(config),
                    editable=True, reference=target)
                descriptor.update(working_tissue=True, added=True)
                await run_in_threadpool(inputs.save_selection, root, config, image, output, 'tissue', {'mode':'tissue','class_name':''})
                return JSONResponse({'layer':descriptor, 'features':features,
                    'source':{'path':source['path'], 'class_name':body['class_name']}})
            if body.get('action') in ('remove_layer', 'hide_file'):
                from spacemap_gui.studio import annotation_names
                source = _annotation_source(body.get('layer_token'), image)
                if body['action'] == 'remove_layer':
                    moved = await run_in_threadpool(annotation_names.remove_layer, root, image, output, source['path'])
                    return JSONResponse({'removed': True, 'kept_in': moved})
                await run_in_threadpool(annotation_names.hide_file, root, image, source['path'])
                return JSONResponse({'hidden': True})
            if body.get('action') == 'duplicate':
                from spacemap_gui.studio import annotation_names
                layer = await run_in_threadpool(annotation_names.duplicate, root, image, output, body.get('name'), body.get('from'), body.get('features'))
                return JSONResponse({'layer': layer})
            if body.get('action') == 'import':
                features = body.get('features')
                if body.get('source_token'):
                    source = registry.resolve_object(body['source_token'],'annotations') or registry.resolve_object(body['source_token'],'shapes')
                    if not source or Path(source.get('image','')).resolve() != Path(image).resolve():
                        raise ValueError('Choose a file belonging to this image.')
                    features = (await run_in_threadpool(annotation_edits.read,source['path']))[0]
                target = await run_in_threadpool(inputs.import_polygons, root, image, body.get('name','Imported'),features)
                return JSONResponse({'layer':inputs.layer(target,image,readonly)})
            value = await run_in_threadpool(inputs.save_selection, root, config, image, output, body.get('stage'),body.get('selection'))
            return JSONResponse({'selection':value})
        if request.query_params.get('source_token'):
            from collections import Counter
            if registry.resolve(request.query_params.get('image_token', '')) != image:
                raise ValueError('The image changed. Open its annotations again.')
            source = _annotation_source(request.query_params['source_token'], image)
            features, version = await run_in_threadpool(annotation_edits.read, source['path'])
            counts = Counter(inputs.annotation_class(f) for f in features)
            return JSONResponse({'classes':[{'name':name,'count':count} for name,count in sorted(counts.items())],
                                 'revision':version,'count':len(features),'read_only':readonly})
        if request.query_params.get('layers') == '1':
            from .api import _annotation_layer
            from .ui import shapes_layer
            from spacemap_gui.studio import annotation, artifact_learning
            sample = annotation.sample_id(image)
            token = registry.register(image)
            layers = []
            for part in artifact_learning.NAMES:
                if artifact_learning.polygons_path(output, sample, part).is_file():
                    item = _layer(output, sample, part, image, readonly); item.update(visible=False, added=True); layers.append(item)
                if artifact_learning.example_path(root, part, image).is_file():
                    layers.append(_examples_layer(root, part, image, readonly))
            mappings = {**(registry.session_project(request.path_params['channel']) or {}).get('mappings', {}), **config.get('directory_mappings', {})}
            zones = annotation.zones_folder(root, mappings)
            from spacemap_gui.studio import annotation_layers
            layers.extend(annotation_layers.layer(root,image,output,path,readonly)
                          for path in annotation_layers.generated(root,image,output))
            from spacemap_gui.studio.annotation_names import hidden_files, stale_domains
            hidden = hidden_files(root, image)
            for group, name, path in [*([] if stale_domains(root, image, output) else [('classes','Named classes',annotation.domains(output,sample))]),
                                      ('zones','Zones',annotation.zones(zones,sample))]:
                if path.is_file() and str(path.resolve()) not in hidden:
                    item = _annotation_layer(str(path), image, token, name)
                    item.update(annotation_group=group, visible=False, added=True); layers.append(item)
            layers.extend(inputs.layer(path,image,readonly) for path in inputs.imported(root,image))
            return JSONResponse({'layers':layers})
        project = registry.session_project(request.path_params['channel']) or {}
        config = {**config, 'directory_mappings':{**project.get('mappings', {}), **config.get('directory_mappings', {})}}
        choices = await run_in_threadpool(inputs.sources,root,config,image,output)
        stage = request.query_params.get('stage','classes')
        from spacemap_gui.studio import annotation_names
        result = {**inputs.source_options(root,config,image,output,choices),
                  'selection':inputs.selection(root,image,stage),'read_only':readonly,
                  # Every annotation name, as Layers shows them (training examples and artifacts are not areas).
                  'names':[{k:g[k] for k in ('source','name','files','count')}
                           for g in await run_in_threadpool(annotation_names.groups,root,config,image,output) if not g['role']]}
        path = request.query_params.get('path')
        if path:
            if path not in choices: raise ValueError('Choose a file belonging to this image.')
            features,_ = await run_in_threadpool(annotation_edits.read,path)
            result['classes'] = sorted({inputs.annotation_class(f) for f in features if inputs.annotation_class(f)})
        return JSONResponse(result)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return JSONResponse({'error':str(exc)},status_code=400)


def _annotation_source(token, image):
    source = registry.resolve_object(token, 'annotations') or registry.resolve_object(token, 'shapes')
    if not source or Path(source.get('image', '')).resolve() != Path(image).resolve():
        raise ValueError('Choose an annotation layer belonging to this image.')
    return source


async def remove_small_pieces(request):
    """Remove small pieces from the Tissue of the open image or of each ticked image, as a job."""
    from .api import _open_annotation_image, _json_body, _studio_project
    from spacemap_gui.studio import annotation, jobs
    try:
        root, config, image, _ = _open_annotation_image(request)
        _studio_project(request, write=True)
        body = await _json_body(request)
        smallest = body.get("min_um2")
        if isinstance(smallest, bool) or not isinstance(smallest, (int, float)) or not 0 < smallest <= 1e9:
            raise ValueError("Enter the smallest piece to keep, in µm² (more than 0).")
        mappings = {**(registry.session_project(request.path_params["channel"]) or {}).get("mappings", {}), **config.get("directory_mappings", {})}
        folder = annotation.all_folders(root, mappings)["images"]
        names = [Path(image).name] if body.get("open_image") else body.get("images")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError("Tick the images first.")
        missing = [n for n in names if "/" in n or not (folder / n).is_file()]
        if missing:
            raise ValueError("Not in the images folder: " + ", ".join(missing[:4]))
        job = await run_in_threadpool(jobs.start, str(root), "remove_small_pieces",
                                      {"root": str(root), "min_um2": float(smallest), "images": [str(folder / n) for n in dict.fromkeys(names)]})
        return JSONResponse({"job": job})
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
