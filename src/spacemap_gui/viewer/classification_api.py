"""Classification actions share one checked image/scope and an immutable run identity."""
from pathlib import Path
import numpy as np
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from . import registry


def _batch(root, mappings, body):
    """Classify the ticked images one by one in the background, each inside its own Work within area."""
    from spacemap_gui.studio import annotation, classification as c, jobs

    images = body.get('images')
    if not isinstance(images, list) or not images or not all(isinstance(name, str) for name in images):
        raise ValueError('Tick the images first.')
    folder = annotation.all_folders(root, mappings)['images']
    missing = [name for name in images if '/' in name or not (folder / name).is_file()]
    if missing:
        raise ValueError('Not in the images folder: ' + ', '.join(missing[:4]))
    chosen = c.options(body.get('options', {}))
    job = jobs.start(str(root), 'classify_images', {'root': str(root), 'images': [str(folder / name) for name in dict.fromkeys(images)],
                                                   'options': chosen})
    return {'job': job}


def _images_of(root, mappings, body):
    from spacemap_gui.studio import annotation

    images = body.get('images')
    if not isinstance(images, list) or not images or not all(isinstance(name, str) for name in images):
        raise ValueError('Tick the images first.')
    folder = annotation.all_folders(root, mappings)['images']
    missing = [name for name in images if '/' in name or not (folder / name).is_file()]
    if missing:
        raise ValueError('Not in the images folder: ' + ', '.join(missing[:4]))
    return [str(folder / name) for name in dict.fromkeys(images)]


def _shared(root, mappings, body):
    """Shared classes for the ticked images (a job): a new class model fitted on them (fit), or the chosen one applied."""
    from spacemap_gui.studio import classification as c, class_models, jobs

    images = _images_of(root, mappings, body)
    settings = {'root': str(root), 'images': images}
    if body.get('action') == 'fit':
        settings.update(options=c.options(body.get('options', {})), name=str(body.get('name') or '')[:80],
                        scale_by='slide' if body.get('scale_by') == 'slide' else 'image')
    else:
        class_models.read(root, body.get('class_model'))
        settings['model'] = body['class_model']
    return {'job': jobs.start(str(root), 'classify_shared', settings)}


def _save_batch(root, mappings, body):
    from spacemap_gui.studio import classification as c, jobs

    return {'job': jobs.start(str(root), 'save_class_polygons', {'root': str(root), 'images': _images_of(root, mappings, body),
                                                                  'tidy': c.cleanup(body.get('tidy'))})}


async def classification(request):
    from .api import _json_body, _open_annotation_image, _studio_project
    from spacemap_gui.studio import classification as c, annotation, annotation_edits as edits, annotation_inputs as inputs
    try:
        root,config,image,boundary=_open_annotation_image(request)
        if request.method=='GET':
            from spacemap.viewer.sources import image_info
            info=await run_in_threadpool(image_info,image)
            from spacemap_gui.studio import class_models
            return JSONResponse({**c.initial(root,image),'class_models':class_models.listing(root),'microns_per_pixel':info.pixel_size_um or annotation.settings(config).get('microns_per_pixel'),
                'read_only':bool((registry.session_project(request.path_params['channel']) or {}).get('read_only'))})
        body=await _json_body(request)
        if registry.resolve(body.get('image_token',''))!=image: raise ValueError('The image changed. Classify the open image again.')
        action=body.get('action','run');output=boundary.parent.parent
        if action not in ('run','review','save','suggest','load','batch','fit','select_model','save_batch'): raise ValueError('Choose a classification action.')
        read_only=bool((registry.session_project(request.path_params['channel']) or {}).get('read_only'))
        if action in ('review','save','batch','fit','select_model','save_batch'): _studio_project(request,write=True)
        from spacemap_gui.studio import class_models
        model=body.get('class_model') or ''
        mappings={**(registry.session_project(request.path_params['channel']) or {}).get('mappings',{}),**config.get('directory_mappings',{})}
        if action=='select_model':
            return JSONResponse(await run_in_threadpool(class_models.select,root,model))
        if action=='fit' or (action=='batch' and model):
            return JSONResponse(await run_in_threadpool(_shared,root,mappings,body))
        if action=='save_batch':
            return JSONResponse(await run_in_threadpool(_save_batch,root,mappings,body))
        if action=='batch':
            return JSONResponse(await run_in_threadpool(_batch,root,mappings,body))
        scope=inputs.scope_path(root,image,output,'classes')
        elapsed=0.0
        if action=='load':
            # The image's last run, shown without clustering again; nothing when there is none for this area.
            last=c.latest(root,image)
            loaded=c.load(root,image,last['run_id']) if last else None
            if not loaded or loaded[1]['provenance'].get('scope')!=str(scope):
                return JSONResponse({'none':True,'options':(last or {}).get('options') or c.initial(root,image)['options']})
            prepared,found=loaded;chosen=c.options(last['options'])
        elif action=='run':
            if model:  # the chosen class model's classes, not new ones for this image
                prepared,found,chosen,elapsed=await run_in_threadpool(class_models.apply,root,config,image,output,model)
            else:
                prepared,found,chosen,elapsed=await run_in_threadpool(c.compute,image,config,scope,body.get('options',{}))
            if not read_only: await run_in_threadpool(c.persist,root,image,prepared,found)
        else:
            loaded=c.load(root,image,body.get('run_id'))
            if loaded: prepared,found=loaded;chosen=c.options(body.get('options',{}))
            elif model: prepared,found,chosen,elapsed=await run_in_threadpool(class_models.apply,root,config,image,output,model)
            else: prepared,found,chosen,elapsed=await run_in_threadpool(c.compute,image,config,scope,body.get('options',{}))
        if action!='run' and action!='load' and body.get('run_id')!=found['run_id']: raise ValueError('The image, annotation or settings changed. Classify again before reviewing these classes.')
        target=c.directory(root,image)/(found['run_id']+'.json')
        selection=inputs.selection(root,image,'classes')
        from spacemap_gui.studio import annotation_layers
        polygon_target=annotation_layers.classification_target(output,image,selection)
        saved=c.read_json(target,{})
        shared=found['provenance'].get('class_model')  # names, colours and merges belong to the model
        if shared:
            try: chosen=c.options(class_models.read(root,shared)[0]['options'])
            except ValueError: shared=None
        base=class_models.review_of(root,shared) if shared else saved.get('review')
        state=c.review(body.get('review') if body.get('review') is not None and action!='load' else base,np.flatnonzero(found['counts']).tolist())
        result=await run_in_threadpool(c.render,prepared,found,state,body.get('hidden',[]))
        if action=='suggest':
            context={k:result[k] for k in ('classes','profile','markers','feature_names','scale')}
            selection=inputs.selection(root,image,'classes')
            context.update(image=Path(image).name,annotation=selection,
                annotation_name=selection.get('class_name') or (Path(selection['path']).stem if selection.get('path') else annotation.tissue_label(config)),question=str(body.get('question','Suggest names and useful merges.'))[:2000],
                selected=body.get('selected',[]))
            result['suggestions']=await run_in_threadpool(c.suggest,context,body.get('images',[]))
        if action=='save':
            def commit():
                from spacemap_gui.studio.settings import project_lock
                selected=inputs.selection(root,image,'classes')
                parent=scope if selected['mode']=='image' else inputs.working_path(root,image,output,'classes')
                with project_lock(root), edits.locked(parent):
                    if body.get('revision')!=edits.revision(target): raise ValueError('Classification changed in another window. Reload before saving.')
                    current_scope=inputs.scope_path(root,image,output,'classes')
                    if current_scope!=scope or edits.revision(current_scope)!=body.get('scope_revision'):
                        raise ValueError('The source annotation changed. Classify again before saving.')
                    polygons=c.save_polygons(root,image,output,scope,prepared,found,state,body.get('polygon_revision'),target=polygon_target,tidy=body.get('tidy'))
                    state['sealed']=True
                    return polygons,c.save_review(root,image,found,chosen,state,body.get('revision'))
            result['saved_polygons'],result['revision']=await run_in_threadpool(commit)
        elif action=='review':
            result['revision']=await run_in_threadpool(c.save_review,root,image,found,chosen,state,body.get('revision'))
            if shared and not read_only: await run_in_threadpool(class_models.save_review,root,shared,state)
        else: result['revision']=edits.revision(target)
        if action in ('run','load') or body.get('index'): result['index']=await run_in_threadpool(c.index_png,found['cluster_map'])
        if action=='run' and not read_only and not target.is_file():
            result['revision']=await run_in_threadpool(c.save_review,root,image,found,chosen,state,None)
        if action=='save' and shared and not read_only: await run_in_threadpool(class_models.save_review,root,shared,state)
        if shared:
            check=found.get('check') or {}
            result['class_model']={'id':shared,'name':found['provenance'].get('class_model_name',''),'far_percent':check.get('far_percent'),
                'differ':[{**item,'name':next((c_['name'] for c_ in result['classes'] if item['id'] in c_['members']),str(item['id']))} for item in class_models.flagged(check)]}
        result.update(options=chosen,review=state,seconds=round(elapsed,2),scope_revision=edits.revision(scope),
            polygon_revision=edits.revision(polygon_target))
        return JSONResponse(result)
    except (ValueError,OSError,TypeError,KeyError,RuntimeError) as error:
        return JSONResponse({'error':str(error)},status_code=400)
