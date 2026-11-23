"""One image selection for merged and per-image phenotyping tables."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spacemap import dataio
from spacemap.viewer.catalog import sample_of
from spacemap_workflows.leiden_job import write_json
from .settings import project_lock


def _path(root):
    return Path(root) / 'outputs/studio/phenotyping_images.json'


def _key(table, per_image=False):
    path = Path(table).resolve()
    return hashlib.sha256(str(path.parent if per_image else path).encode()).hexdigest()


def selected(root, table, rows, per_image=False):
    available = [row['roi'] for row in rows if row.get('table_path')]
    saved = json.loads(_path(root).read_text()) if _path(root).exists() else {}
    return [name for name in saved.get(_key(table, per_image), available) if name in available]


def save(root, table, images):
    from . import phenotyping_inputs
    plan = phenotyping_inputs.plan(root, {}, table, summaries=False)
    available = {row['roi'] for row in plan['rows'] if row.get('table_path')}
    if not isinstance(images, list) or any(not isinstance(v, str) or v not in available for v in images):
        raise ValueError('Select images listed in the table.')
    with project_lock(root):
        path = _path(root)
        saved = json.loads(path.read_text()) if path.exists() else {}
        saved[_key(table, plan['per_image'])] = list(dict.fromkeys(images))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, saved)
    return {'selected_images': list(dict.fromkeys(images))}


def sources(root, table, images=None):
    from . import phenotyping_inputs
    plan = phenotyping_inputs.plan(root, {}, table, summaries=False)
    chosen = plan['selected_images'] if images is None else images
    available = {row['roi']: row for row in plan['rows'] if row.get('table_path')}
    if not isinstance(chosen, list) or not chosen or any(name not in available for name in chosen):
        raise ValueError('Select at least one image in the table above.')
    grouped = {}
    for name in dict.fromkeys(chosen):
        row = available[name]
        path = row['table_path']
        if path not in grouped:
            roles = phenotyping_inputs.column_roles(root, path)
            grouped[path] = {'table':path, 'roi_column':roles['roi'], 'selected_images':[]}
        grouped[path]['selected_images'].append(name)
    return list(grouped.values())


def mask(frame, settings):
    column = settings.get('roi_column') or settings.get('roi_col')
    selected_images = settings.get('selected_images')
    if selected_images is None or not column:
        return np.ones(len(frame), bool)
    if column not in frame:
        raise ValueError('The image column is missing. Check Column roles.')
    names = frame[column].astype('string').fillna('')
    lookup = {name: sample_of(name) in selected_images for name in names.unique()}
    return names.map(lookup).to_numpy(bool)


def rows_mask(settings, rows):
    column = settings.get('roi_column') or settings.get('roi_col')
    if settings.get('selected_images') is None or not column:
        return np.ones(len(rows), bool)
    from .data_editor import Source
    return mask(pd.DataFrame(Source(settings['table']).take(rows, [column])), settings)


def subset(frame, settings):
    keep = mask(frame, settings)
    if not keep.any():
        raise ValueError('No cells in the selected images.')
    return frame.loc[keep].copy()


def publish_columns(settings, columns, anchor):
    """Fill new columns outside this run; preserve existing values there."""
    if settings.get('selected_images') is None and not settings.get('row_scoped'):
        return columns, anchor
    key = settings.get('cell_id_col', 'cell_id')
    table = settings['table']
    available = dataio.columns(table)
    replacing = set(settings.get('overwrite_columns', []))
    old_columns = [c for c in columns if c in available and c in replacing]
    full = dataio.load(table, columns=[key, *old_columns])
    positions = pd.Index(full[key].astype(str)).get_indexer(pd.Series(anchor).astype(str))
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise ValueError('Cell identities changed. Run phenotyping again.')
    result = {}
    for name, values in columns.items():
        merged = full[name].to_numpy(copy=True).astype(object) if name in old_columns else np.full(len(full), None, object)
        merged[positions] = np.asarray(values)
        result[name] = merged
    return result, full[key]


def start(channel, name, args, spec, request_id):
    """Preflight all selected tables before launching one worker."""
    from spacemap_gui.viewer import registry
    from . import commands, jobs, mr_ml, phenotyping_inputs
    from .settings import revision
    project = registry.session_project(channel)
    root = project['root']
    selected = sources(root, phenotyping_inputs.combined_table(channel))
    work = []
    for source in selected:
        table = source['table']
        if name == 'run_mr_ml':
            chosen = mr_ml.values(root)
            if args.get('settings_revision') != revision(chosen):
                raise ValueError('MR-ML settings changed. Reload before running.')
            if chosen.get('workflow') == 'saved_model':
                from . import reference_setup
                allowed = reference_setup.options(root, table)['models']
                if chosen.get('model_id') not in {item['id'] for item in allowed}:
                    raise ValueError('Choose a model from the Models folder.')
            roles = phenotyping_inputs.column_roles(root, table)
            kind, settings = 'mr_ml', mr_ml.job_settings(root, table, roles, chosen, args.get('overwrite', False))
        else:
            kind, settings = commands._leiden(channel, name, args, spec, table, request_id, prepare_only=True)
        work.append({'kind':kind, 'settings':{**settings, **source}})
    if len(work) == 1:
        job = jobs.start(root, work[0]['kind'], work[0]['settings'], request_id=request_id)
    else:
        job = jobs.start(root, 'phenotyping_images', {'work':work, 'project_root':str(root)}, request_id=request_id)
    return spec, [], {'job':job}, {'kind':'none'}, f"Running on {sum(len(s['selected_images']) for s in selected)} selected images. See Jobs for results."


def run(settings, folder, report):
    """Separate per-image results remain inspectable through ordinary Jobs."""
    from datetime import datetime, timezone
    import uuid
    from . import jobs, tasks
    work = settings['work']
    results = []
    for item in work:
        tasks.validate_table(item['settings'])
        if tasks.source_revision(item['settings']) != item['settings']['revision']:
            raise ValueError('An input table changed. Start phenotyping again.')
    for index, item in enumerate(work):
        identifier = uuid.uuid4().hex
        child = jobs.root(settings['project_root']) / identifier
        child.mkdir(parents=True)
        write_json(child/'request.json', {'project':settings['project_root'], **item})
        started = datetime.now(timezone.utc).isoformat()
        def progress(message, fraction=0):
            report(f"{Path(item['settings']['table']).name} ({index+1}/{len(work)}): {message}",
                   (index + (fraction or 0))/len(work))
        if hasattr(report, 'protect_output'):
            progress.protect_output = report.protect_output
        result = getattr(tasks, item['kind'])(item['settings'], child, progress)
        write_json(child/'status.json', {'id':identifier, 'kind':item['kind'], 'status':'done', 'progress':1,
            'message':'Finished', 'started':started, 'finished':datetime.now(timezone.utc).isoformat(),
            'result':result, 'parent_job_id':Path(folder).name})
        results.append(result)
        write_json(Path(folder)/'completed_tables.json', results)
        if index+1 < len(work) and hasattr(report, 'finish_output'):
            report.finish_output()
    return {'tables':results, 'images':sum(len(i['settings']['selected_images']) for i in work),
            'cells':sum(r['cells'] for r in results)}
