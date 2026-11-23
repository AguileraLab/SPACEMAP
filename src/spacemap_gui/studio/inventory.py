"""Studio Lists: exact project files, lazy summaries and reversible project removal.

No image pixels, model weights or table statistics are read by the index. Only an
expanded file group asks for its annotation/phenotype summaries.
"""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

from . import project_files
from .settings import project_lock

IMAGE = {'.tif', '.tiff', '.qptiff', '.svs'}
TABLE = {'.csv', '.tsv', '.parquet'}
FIGURE = {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}
METHODS = [('leiden', 'Leiden'), ('som', 'SOM'), ('vortex', 'Vortex'),
           ('resolve', 'Threshold / RESOLVE'), ('reference', 'Manual reference'),
           ('mr_ml', 'MR-ML'), ('consensus', 'Consensus ML'), ('subphenotyping', 'Subphenotyping')]


def _path(root, value):
    p = Path(str(value).replace('${PROJECT_ROOT}', str(root))).expanduser()
    return (p if p.is_absolute() else Path(root) / p).resolve()


def _files(folder, suffixes, depth=0):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = []
    for base, dirs, files in os.walk(folder):
        relative = Path(base).relative_to(folder)
        dirs[:] = [d for d in sorted(dirs) if not d.startswith('.') and d not in
                   {'previous', 'deleted', 'studio_history', 'intermediate', 'cellpose_training'}
                   and len(relative.parts) < depth]
        found.extend(Path(base) / name for name in sorted(files) if Path(name).suffix.lower() in suffixes
                     and not name.startswith('.') and not name.endswith(('.csv.parquet', '.labels.parquet')))
    return found


def _id(group, path, image=''):
    return sha256(f'{group}\0{Path(path).resolve()}\0{image}'.encode()).hexdigest()[:24]


def groups(root, config, step, table=None):
    """Inventory definitions from the same folder resolvers as each workflow."""
    from . import annotation, background, correction, registration, rois, segmentation, tagging, training
    from spacemap_gui.tabs import cell_tagging_ui
    root = Path(root)
    mappings = config.get('directory_mappings') or {}
    items = []

    def add(key, name, folder=None, suffixes=(), role='output', **extra):
        items.append(dict(id=key, name=name, role=role, folder=str(folder) if folder else '',
                          suffixes=sorted(suffixes), **extra))

    if step == 'tab0':
        add('project', 'Project files', root, {'.json'}, role='input', readonly=True)
    elif step == 'tab1':
        codex, he, output = registration.folders(root, mappings)
        add('codex', 'CODEX images', codex, IMAGE, 'input')
        for i, folder in enumerate(registration.extra_folders(root, config), 2):
            add('codex' + str(i), f'CODEX images ({i})', folder, IMAGE, 'input')
        add('he', 'H&E images', he, IMAGE, 'input')
        add('registered', 'Registered images', output, IMAGE)
    elif step == 'tab2':
        images, levels, output = background.folders(root, mappings, config)
        add('levels', 'Threshold tables', levels, TABLE)
        add('background', 'Background removed', output, IMAGE)
    elif step == 'tab3':
        images, shapes, output = rois.folders(root, mappings, config)
        add('roi_shapes', 'ROI outlines', shapes, {'.geojson'})
        add('rois', 'ROI images', output, IMAGE)
    elif step == 'tab4':
        from . import artifact_learning
        add('imports', 'Imported GeoJSON', role='import', special='annotations', stage='export', importable=True)
        add('tissue', 'Tissue', special='annotations', stage='tissue')
        for part, name in artifact_learning.NAMES.items():
            add('artifact_' + part, name, special='annotations', stage='artifacts', part=part)
        for key, name, stage in [('drawn', 'Drawn artifacts', 'artifacts'), ('classes', 'Classification', 'classes'),
                                 ('regions', 'Marker regions', 'regions'), ('zones', 'Zones', 'zones'),
                                 ('exports', 'Exported GeoJSON', 'export')]:
            add(key, name, special='annotations', stage=stage)
        add('artifact_models', 'Artifact models', role='model', special='artifact_models')
    elif step == 'tab5':
        places = segmentation.step_folders(root, config, mappings)
        add('measurements', 'Cell measurements', places['output'], TABLE)
        add('outlines', 'Cell outlines', places['geojson'], {'.geojson'})
        add('seg_models', 'Models', training.folders(root)[1] / 'models', role='model', special='seg_models', importable=True)
    elif step == 'tagging':
        saved = tagging.values(root, config)
        add('tagged', 'Cell tagging results', saved[cell_tagging_ui.OUTPUT_DIR_KEY], TABLE)
    elif step == 'tab6':
        add('combined', 'Combined measurements', correction.output_dir(root, mappings), TABLE)
    elif step == 'tab7':
        from . import phenotyping_inputs
        place = phenotyping_inputs.folders(root, mappings, table)
        # Only an independent input directory; previous steps' outputs remain in their own section.
        previous = {_path(root, g['folder']) for s in ('tab5', 'tagging', 'tab6') for g in groups(root, config, s)
                    if g['folder'] and g.get('suffixes') == sorted(TABLE)}
        if _path(root, place['tables']) not in previous:
            add('tables', 'Cell data', place['tables'], TABLE, 'input')
        for method, name in METHODS:
            add(method, name, special='phenotypes', method=method)
        add('ref_models', 'Models', root / 'data/reference_models', role='model', special='ref_models', method='reference', importable=True, parent='reference')
    elif step in ('tab8', 'tab9'):
        folder = _path(root, mappings.get('spatial_output_dir') or 'outputs/spatial_analysis')
        if step == 'tab8':
            for key, name in [('density', 'Density'), ('neighbourhoods', 'Neighbourhoods'), ('interactions', 'Interactions'),
                              ('triads', 'Triads'), ('feature_screening', 'Feature screening'), ('tls', 'TLS')]:
                add(key, name, folder / key, TABLE | {'.json', '.geojson'}, depth=3)
            add('spatial_files', 'Other results', folder, TABLE | {'.json', '.geojson'})
        else:
            add('figures', 'Figures', _path(root, mappings.get('tab9_actual_output_dir') or folder), FIGURE, depth=4)
            add('layouts', 'Layouts', root / 'outputs/studio', {'.svg', '.pdf', '.png'}, depth=3)
    else:
        raise ValueError('Choose a Studio step.')
    # Independent inputs are listed once; pipeline outputs already have a home above.
    known_images = {p.resolve() for p in (registration.folders(root, mappings)[2],
                    background.folders(root, mappings, config)[2], rois.folders(root, mappings, config)[2])}
    image_inputs = {
        'tab2': lambda: background.folders(root, mappings, config)[0],
        'tab3': lambda: rois.folders(root, mappings, config)[0],
        'tab4': lambda: annotation.folders(root, mappings)[0],
        'tab5': lambda: segmentation.step_folders(root, config, mappings)['images'],
    }
    if step in image_inputs:
        folder = image_inputs[step]()
        if Path(folder).resolve() not in known_images:
            add('images', 'Images', folder, IMAGE, 'input')
            items.insert(0, items.pop())
    if step in ('tagging', 'tab6'):
        from .correction import input_folder
        folder = (tagging.values(root, config)[cell_tagging_ui.CSV_DIR_KEY] if step == 'tagging'
                  else input_folder(root, mappings))
        previous = {segmentation.step_folders(root, config, mappings)['output'].resolve()}
        if step == 'tab6': previous.add(Path(tagging.values(root, config)[cell_tagging_ui.OUTPUT_DIR_KEY]).resolve())
        if Path(folder).resolve() not in previous:
            add('tables', 'Cell data', folder, TABLE, 'input')
            items.insert(0, items.pop())
    for group in items:
        group.setdefault('importable', bool(group['folder']) and not group.get('readonly') and not group.get('special'))
    return items


def _image_for(root, config, path):
    from . import annotation_names
    return annotation_names.match_image(annotation_names._annotation_images(root, config), Path(path).name)


def _row(group, path, **extra):
    path = Path(path).resolve()
    return dict(id=_id(group['id'], path, extra.get('image', '')), path=str(path), name=path.name,
                kind='image' if path.suffix.lower() in IMAGE else 'table' if path.suffix.lower() in TABLE else
                'annotation' if path.suffix.lower() == '.geojson' else 'file',
                role=group['role'], removable=not group.get('readonly', False), **extra)


def _annotation_rows(root, config, group, details):
    from . import annotation, annotation_names, annotation_links, artifact_learning
    output = annotation.all_folders(root, config.get('directory_mappings') or {})['output']
    rows = []
    seen = set()
    links = annotation_links.records(root) if group['id'] == 'imports' else []
    linked_paths = {r['path'] for r in links}
    for image in annotation_names._annotation_images(root, config):
        entries = annotation_names.files(root, config, image, output)
        if group['id'] == 'exports':
            entries = [dict(path=str(p.resolve()), kind='file', layer='Annotations', source='', readonly=False, origin='local')
                       for p in annotation_names.result_files(root, config, image, 'export')]
        for entry in entries:
            path, kind = Path(entry['path']), entry['kind']
            if str(path.resolve()) in linked_paths:
                continue  # Explicit imports, including duplicates, are listed below.
            matches = (group['id'] == 'imports' and (kind == 'imported' or entry['origin'] == 'imported' or
                       (kind == 'file' and path != annotation.zones(annotation.zones_folder(root, config.get('directory_mappings', {})), annotation.sample_id(image)))))
            matches |= group['id'] == 'tissue' and kind == 'tissue'
            matches |= group['id'] == 'drawn' and kind == 'drawn'
            matches |= bool(group.get('part')) and kind == 'artifact' and path == artifact_learning.polygons_path(output, annotation.sample_id(image), group['part'])
            matches |= group['id'] == 'classes' and kind == 'regions' and ('classification' in path.name or 'classes' in path.name)
            matches |= group['id'] == 'regions' and kind == 'regions' and not ('classification' in path.name or 'classes' in path.name)
            matches |= group['id'] == 'zones' and path == annotation.zones(annotation.zones_folder(root, config.get('directory_mappings', {})), annotation.sample_id(image))
            matches |= group['id'] == 'exports'
            if not matches or project_files.excluded(path, root):
                continue
            key = (str(path), str(image))
            if key in seen:
                continue
            seen.add(key)
            row = _row(group, path, image=str(image), stage=group.get('stage'), part=group.get('part'),
                       annotation_kind=kind, source_name=entry['layer'], linked=entry['readonly'])
            row['name'] = path.name.split('_', 1)[-1] if kind == 'imported' else path.name
            row['copy'] = path.stem.split('_', 1)[0] if kind == 'imported' else ''
            if details:
                counts, colours = Counter(), {}
                for name, colour, count in annotation_names._scan(entry):
                    counts[name] += count
                    if colour:
                        colours[name] = colour
                row['classes'] = [dict(name=name, count=count, colour=colours.get(name, '#7eb3ef')) for name, count in counts.items()]
                row['count'] = sum(counts.values())
            rows.append(row)
    # Retain orphaned copies too, even if the image folder has since changed.
    if group['id'] == 'imports':
        for link in links:
            path = Path(link['path'])
            row = _row(group, path, image=link['image'], stage='export', annotation_kind='file', linked=True,
                       link_id=link['id'], missing=not path.is_file())
            row.update(id=link['id'], copy=link['id'][:8], kind='annotation')
            if details:
                counts, colours = Counter(), {}
                for name, colour, count in annotation_names._scan({'path': str(path), 'kind': 'file', 'layer': path.stem}):
                    counts[name] += count
                    colours[name] = colour
                row.update(classes=[dict(name=n, count=c, colour=colours.get(n)) for n,c in counts.items()], count=sum(counts.values()))
            rows.append(row)
        known = {row['path'] for row in rows}
        for path in sorted((Path(root) / 'outputs/studio/imported_annotations').glob('*/*.geojson')):
            if str(path.resolve()) not in known and not project_files.excluded(path, root):
                row = _row(group, path, image='', stage='export', annotation_kind='imported', linked=False)
                row.update(name=path.name.split('_', 1)[-1], copy=path.stem.split('_', 1)[0])
                if details:
                    counts = Counter()
                    for name, colour, count in annotation_names._scan({'path': str(path), 'kind': 'imported', 'layer': row['name']}):
                        counts[name] += count
                    row.update(classes=[dict(name=n, count=c) for n,c in counts.items()], count=sum(counts.values()))
                rows.append(row)
    return rows


def _models(root, config, group):
    from . import segmentation, reference_setup, artifact_learning
    rows = []
    if group['id'] == 'seg_models':
        folder = segmentation.step_folders(root, config, config.get('directory_mappings', {}))['models']
        for item in segmentation.model_choices(root, folder):
            if item['path'] and Path(item['path']).is_file():
                rows.append(_row(group, item['path'], model_id=item['id'], model_kind='cellpose'))
    elif group['id'] == 'ref_models':
        listed = reference_setup.project_models(root)
        for item in listed.values():
            meta = item['meta']
            row = _row(group, item['path'], model_id=item['id'], model_kind='phenotyping', method='reference')
            row.update(name=meta.get('name') or meta.get('method', 'Model') + ' · ' + meta.get('column', ''),
                       count=meta.get('training_cells') or meta.get('examples'), classes=[dict(name=n) for n in meta.get('classes', [])])
            rows.append(row)
    else:
        for part, name in artifact_learning.NAMES.items():
            home = artifact_learning.folder(root, part)
            path = home / 'versions.json'
            versions = json.loads(path.read_text()) if path.is_file() else []
            for item in versions:
                artifact = home / (item['id'] + '.joblib')
                if artifact.is_file() and not item.get('deleted_at'):
                    row = _row(group, artifact, model_id=item['id'], model_kind='artifact', part=part, stage='artifacts')
                    row['name'] = name + ' · ' + (item.get('name') or item['id'])
                    rows.append(row)
    return [r for r in rows if not project_files.excluded(r['path'], root)]


def _phenotypes(root, config, group, table, details):
    from . import phenotyping_inputs, phenotyping_overview, summary_cache
    from spacemap import dataio
    locations = phenotyping_inputs.folders(root, config.get('directory_mappings', {}), table)
    paths = phenotyping_inputs.table_files(locations['tables'])
    if table and Path(table).is_file() and table not in paths:
        paths.insert(0, table)
    rows = []
    for path in paths:
        if project_files.excluded(path, root):
            continue
        columns = dataio.columns(path)
        methods = phenotyping_overview.method_columns(root, path, columns)
        names = next((m['columns'] for m in methods if m['id'] == group['method']), [])
        if group['method'] == 'subphenotyping':
            names = [c for c in columns if c.startswith('phenotype_sub')]
        # Clusters remain counts of clusters; phenotype counts never include companion cluster columns twice.
        names = [c for c in names if not c.endswith('_clusters') or c[:-9] not in names]
        for column in names:
            row = _row(group, path, method=group['method'], column=column, image=str(_image_for(root, config, path) or ''))
            row['id'] = _id(group['id'], path, column)
            row.update(name=column, removable=False)
            if details:
                roles = phenotyping_inputs.column_roles(root, path, columns)
                def build():
                    frame = dataio.load(path, columns=list(dict.fromkeys([column] + ([roles['roi']] if roles['roi'] else []))), build_fast_copy=False)
                    values = frame[column].fillna('').astype(str)
                    valid = ~values.str.strip().str.lower().isin(['', 'nan', 'none', 'unassigned', 'unknown', '-1'])
                    counts = values[valid].value_counts()
                    return dict(count=int(valid.sum()), images=int(frame.loc[valid, roles['roi']].nunique()) if roles['roi'] else int(valid.any()),
                                classes=[dict(name=str(n), count=int(c)) for n, c in counts.items()])
                row.update(summary_cache.cached('inventory-phenotypes', [path, column, roles], lambda: summary_cache.table_stamp(path), build))
            rows.append(row)
    return rows


def listing(root, config, step, group_id=None, table=None):
    definitions = groups(root, config, step, table)
    if group_id is None:
        return {'groups': definitions}
    group = next((g for g in definitions if g['id'] == group_id), None)
    if not group:
        raise ValueError('Choose a listed group.')
    special = group.get('special')
    if special == 'annotations':
        rows = _annotation_rows(root, config, group, True)
    elif special == 'phenotypes':
        rows = _phenotypes(root, config, group, table, True)
    elif special and special.endswith('models'):
        rows = _models(root, config, group)
    else:
        rows = [_row(group, p, image=str(p if p.suffix.lower() in IMAGE else _image_for(root, config, p) or ''))
                for p in _files(group['folder'], set(group['suffixes']), group.get('depth', 0)) if not project_files.excluded(p, root)]
    return {'group': group, 'items': rows}


def selected(root, config, step, group_id, ids, table=None):
    if not isinstance(ids, list) or not ids or len(ids) > 10000 or not all(isinstance(i, str) for i in ids):
        raise ValueError('Select files first.')
    # Resolve the identities against the current project, never trust a client path.
    rows = listing(root, config, step, group_id, table)['items']
    wanted = set(ids)
    found = [r for r in rows if r['id'] in wanted]
    if len(found) != len(wanted):
        raise ValueError('The list changed. Refresh and select the files again.')
    return found


def remove(root, config, rows):
    """Unlist project references only. Files and all sidecars stay at their paths."""
    from . import annotation_names, annotation_links
    root = Path(root).resolve()
    if any(not r['removable'] for r in rows):
        raise ValueError('This entry is a result column. Manage its values in Data.')
    paths = []
    with project_lock(root):
        for row in rows:
            path = Path(row['path']).resolve()
            if row.get('link_id'):
                paths.extend(annotation_links.remove(root, {row['link_id']}))
                continue
            if row.get('model_kind') == 'phenotyping':
                from . import reference_models
                project_files.set_removed(root, [reference_models.model_path(row['model_id'])])
            project_files.set_removed(root, [path])
            if row.get('annotation_kind') and row.get('image'):
                annotation_names.hide_file(root, row['image'], path)
            paths.append(str(path))
    return dict(removed=len(rows), paths=paths)


def import_paths(root, config, step, group_id, paths, table=None, image=''):
    """Import compatible direct children only, independently of the active image."""
    group = next((g for g in groups(root, config, step, table) if g['id'] == group_id and g['importable']), None)
    if not group:
        raise ValueError('This group accepts no imports.')
    # Generic JSON inside a folder is usually settings, model metadata or history.
    # Explicit .json paths/uploads remain supported for actual GeoJSON content.
    suffixes = {'.geojson'} if group_id == 'imports' else set(group['suffixes'])
    sources, has_folder = [], False
    for value in paths:
        path = _path(root, value)
        if path.is_dir():
            has_folder = True
            sources.extend((child, child.name) for child in sorted(path.iterdir())
                           if child.is_file() and not child.name.startswith('.')
                           and (not suffixes or child.suffix.lower() in suffixes))
        else:
            sources.append((path, path.name))
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError('No compatible files found in this folder.')
    return import_files(root, config, step, group_id, sources, table, '' if has_folder else image, link=True, ignore_empty=has_folder)


def import_files(root, config, step, group_id, sources, table=None, image='', *, link=False, ignore_empty=False):
    from . import annotation_inputs, annotation_names, annotation_links
    definitions = groups(root, config, step, table)
    group = next((g for g in definitions if g['id'] == group_id and g['importable']), None)
    if not group:
        raise ValueError('This group accepts no imports.')
    added, unmatched, errors, skipped = [], [], [], []
    images = annotation_names._annotation_images(root, config) if group_id == 'imports' else []
    explicit = next((p for p in images if str(p) == image or p.name == image), None) if image else None
    if image and explicit is None:
        raise ValueError('Choose an available image.')
    for source, name in sources:
        source = Path(source)
        name = Path(name).name
        try:
            if not source.is_file() or not name or name.startswith('.'):
                raise ValueError('Choose an existing file.')
            if group_id == 'imports':
                if Path(name).suffix.lower() not in {'.geojson', '.json'}:
                    raise ValueError('Choose a GeoJSON file.')
                target_image = explicit or annotation_names.match_image(images, name, fuzzy=False)
                if link:
                    target = annotation_links.register(root, source, target_image)['path']
                else:
                    target = annotation_inputs.import_polygons(root, target_image, name, annotation_links.features(source))
                if target_image is None:
                    unmatched.append(name)
            elif group_id in ('seg_models', 'ref_models'):
                target = import_model(root, group, source, name)
            else:
                if Path(name).suffix.lower() not in group['suffixes']:
                    raise ValueError(f'wrong file type for {group["name"]}.')
                target = _path(root, group['folder']) / name
                if source.resolve() == target.resolve():
                    project_files.set_removed(root, [target], False)
                    added.append(str(target))
                    continue
                if target.exists():
                    raise ValueError('Already exists. Rename it before importing another version.')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.partial')
                try:
                    shutil.copy2(source, temporary)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                project_files.set_removed(root, [target], False)
            added.append(str(target))
        except (ValueError, OSError, KeyError, TypeError) as error:
            if ignore_empty and str(error) == 'This file contains no polygon annotations.':
                skipped.append(name)
            else:
                errors.append({'name': name, 'error': str(error)})
    if not added and errors:
        raise ValueError('; '.join(f'{e["name"]}: {e["error"]}' for e in errors))
    return {'added': added, 'unmatched': unmatched, 'errors': errors, 'skipped': skipped}


def assign_import(root, config, row, image):
    from . import annotation_links, annotation_names, annotation_inputs
    from .annotation_edits import _atomic
    target = next((p for p in annotation_names._annotation_images(root, config) if str(p) == image or p.name == image), None)
    if target is None:
        raise ValueError('Choose an available image.')
    if row.get('image') or row.get('annotation_kind') not in ('imported', 'file'):
        raise ValueError('Choose an unmatched import.')
    annotation_links.validate_image(row['path'], target)
    with project_lock(root):
        if row.get('link_id'):
            records = annotation_links.records(root)
            for record in records:
                if record['id'] == row['link_id']:
                    record['image'] = str(target.resolve())
            _atomic(annotation_links.index(root), json.dumps(records, indent=2).encode())
            annotation_links._read.cache_clear()
            destination = row['path']
        else:
            destination = annotation_inputs.import_folder(root, target) / Path(row['path']).name
            if destination.exists():
                raise ValueError('This imported copy is already assigned.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            Path(row['path']).replace(destination)
        annotation_names.hide_file(root, target, destination, hidden=False)
    return {'assigned': str(destination)}


def import_model(root, group, source, name):
    """Portable archives are inspected without unpickling; weights load only when used."""
    target_dir = Path(group['folder'])
    if not zipfile.is_zipfile(source):
        if group['id'] != 'seg_models' or Path(name).suffix.lower() in {'.json', '.csv', '.geojson'}:
            raise ValueError('Import a model ZIP exported from Lists (or Cellpose weights).')
        target = target_dir / name
        if target.exists():
            raise ValueError('A model with this name already exists.')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
    with zipfile.ZipFile(source) as archive:
        if 'model.json' not in archive.namelist():
            raise ValueError('The archive has no model.json manifest.')
        manifest = json.loads(archive.read('model.json'))
        expected = 'phenotyping' if group['id'] == 'ref_models' else 'cellpose'
        if manifest.get('kind') != expected:
            raise ValueError('Choose a ' + expected + ' model archive.')
        names = manifest.get('files')
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or Path(n).name != n or n.startswith('.') for n in names):
            raise ValueError('The model archive contains invalid filenames.')
        for name in names:
            if name not in archive.namelist() or (target_dir / name).exists():
                raise ValueError(f'{name} is missing in the archive or already exists in this project.')
        artifact = next((target_dir / n for n in names if not n.endswith('.json')), None)
        if artifact is None:
            raise ValueError('The archive contains no model weights.')
        # Validate every member before publishing any. No archive paths are extracted verbatim.
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target_dir, prefix='.import_') as temporary:
            for name in names:
                with archive.open(name) as handle, open(Path(temporary) / name, 'wb') as out:
                    shutil.copyfileobj(handle, out)
            for name in names:
                (Path(temporary) / name).replace(target_dir / name)
        if expected == 'phenotyping':
            record = manifest.get('record') or {}
            record['imported_from'] = {'path': str(source), 'project': str(root)}
            artifact.with_suffix('.json').write_text(json.dumps(record, indent=2))
        project_files.set_removed(root, [artifact], False)
        return artifact


def export_bundle(rows, target):
    """One download, collision-safe names; model archives retain portable metadata."""
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        used = set()
        for row in rows:
            path = Path(row['path'])
            name = path.name
            if name in used:
                name = row['id'][:8] + '_' + name
            used.add(name)
            if row.get('model_kind') and len(rows) == 1:
                metadata = path.with_suffix('.json') if path.suffix == '.joblib' else Path(str(path) + '.json')
                record = json.loads(metadata.read_text()) if metadata.is_file() else {}
                files = [name]
                if metadata.is_file():
                    archive.write(metadata, metadata.name); files.append(metadata.name)
                archive.writestr('model.json', json.dumps(dict(kind=row['model_kind'], id=row.get('model_id'), name=row['name'], files=files, record=record)))
            archive.write(path, name)
            # A table's pending phenotype values travel with the source; no silent loss.
            for suffix in ('.labels.parquet', '.pending'):
                beside = Path(str(path) + suffix)
                if beside.is_file():
                    archive.write(beside, name + suffix)
                elif beside.is_dir():
                    for p in beside.rglob('*'):
                        if p.is_file(): archive.write(p, name + suffix + '/' + str(p.relative_to(beside)))
