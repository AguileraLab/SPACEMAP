"""Spatial inputs share Phenotyping's image/table assignments, including unmerged files."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from spacemap import dataio, table_collection as collection


OUTPUT_KINDS = {'neighbourhoods': 'neighbourhoods', 'tls_candidates': 'tls_candidates',
                'cell_interactions': 'cell_interactions', 'spatial_density': 'density',
                'spatial_triads': 'triads', 'spatial_screening': 'feature_screening', 'gland_types': 'gland_types'}


def output_folder(root):
    from .settings import read_config
    from .phenotyping_inputs import _path
    return _path(root, read_config(root).get('directory_mappings', {}).get('spatial_output_dir') or 'outputs/spatial_analysis')


def image_entries(root, active=None):
    from . import phenotyping_inputs as inputs, cell_inputs
    images = cell_inputs.images(root, {'bc_images_dir': inputs.folders(root, {}, active)['images']})
    return [{'sample': roi, 'name': Path(path).name, 'path': path} for roi, path in images.items()]


def input_plan(channel, changes=None):
    """Use the same sources, assignments and column roles as Phenotyping."""
    from spacemap_gui.viewer import registry
    from . import phenotyping_inputs as inputs
    project = registry.session_project(channel) or {}
    root = project['root']
    if changes is not None:
        if not isinstance(changes, dict) or set(changes) - {'folders', 'table', 'pair', 'roles'}:
            raise ValueError('Choose folders, a cell table or an image/table assignment.')
        if 'folders' in changes:
            inputs.save_folders(root, changes['folders'], output_key='spatial_output_dir')
        if 'table' in changes:
            inputs.use_table(channel, changes['table'])
        project = registry.session_project(channel) or {}
        if 'roles' in changes:
            inputs.save_roles(root, project.get('unified'), changes['roles'])
        if 'pair' in changes:
            pair = changes['pair']
            if not isinstance(pair, dict) or set(pair) != {'image', 'table'} or not all(isinstance(v, str) for v in pair.values()):
                raise ValueError('Choose a listed image and its cell table.')
            inputs.save_pair(root, project.get('unified'), pair['image'], pair['table'])
    project = registry.session_project(channel) or {}
    found = inputs.plan(root, project.get('mappings', {}), project.get('unified'), summaries=False)
    found['folders']['output'] = str(output_folder(root))
    found['output_subfolders'] = list(OUTPUT_KINDS.values())
    return found


def resolve(root, active):
    from . import phenotyping_inputs as inputs
    if active and collection.is_collection(active):
        return str(active)
    if active and Path(active).is_file() and inputs.column_roles(root, active)["roi"]:
        return str(active)
    pairs = inputs.image_tables(root, active)
    rows = [{"image": image, "path": path, "anchor": inputs.column_roles(root, path)["cell_id"]}
            for image, path in sorted(pairs.items()) if path]
    if not pairs:
        raise ValueError("Choose Images and Tables, then assign a cell table to each image.")
    value = {"format": "spacemap-image-tables-v1", "tables": rows,
             "unpaired": sorted(image for image, path in pairs.items() if not path)}
    text = json.dumps(value, sort_keys=True, indent=2)
    name = hashlib.sha256(text.encode()).hexdigest()[:20] + collection.SUFFIX
    target = Path(root) / "outputs/studio/spatial_sources" / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        from .settings import project_lock
        with project_lock(root):
            if not target.exists():
                with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as stream:
                    stream.write(text)
                    temporary = Path(stream.name)
                try:
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
    return str(target)


def overview(root, table, *, summaries=False):
    from . import phenotyping_overview, tls
    from . import gland_clonality
    gland_summary = gland_clonality.latest(root, table) if summaries else None
    if not collection.is_collection(table):
        from .phenotyping_inputs import column_roles
        roles = column_roles(root, table)
        info = phenotyping_overview.overview(table, roi_column=roles['roi'], slide_column=roles['slide'])
        return {**info, "results": phenotyping_overview.results(table, roi_column=roles['roi']), "tls": tls.latest(root, table), "glands": gland_summary}
    rows, results = [], {}
    for item in collection.entries(table):
        path, image = item["path"], item["image"]
        cells = None
        parquet = Path(path) if path.endswith('.parquet') else None
        if parquet is None:
            twin = dataio.fast_copy_info(path)
            if twin['current']:
                parquet = Path(twin['twin'])
        if parquet:
            import pyarrow.parquet as pq
            cells = pq.ParquetFile(parquet).metadata.num_rows
        elif summaries:
            cells = phenotyping_overview.overview(path, roi_column='', single_roi=image)['cells']
        rows.append({"roi": image, "cells": cells, "table_path": path, "ready": True})
        if summaries:
            for name, counts in phenotyping_overview.results(path, roi_column='', single_roi=image).items():
                results.setdefault(name, {}).update(counts)
    cells = sum(r['cells'] for r in rows) if all(r['cells'] is not None for r in rows) else None
    rows.extend({'roi': image, 'cells': None, 'table_path': '', 'ready': False}
                for image in json.loads(Path(table).read_text()).get('unpaired', []))
    return {"table": "Per-image cell tables", "per_image": True, "path": str(table),
            "cells": cells, "rois": sorted(rows, key=lambda r:r['roi']), "results": results,
            "tls": tls.latest(root, table) if summaries else None, "glands": gland_summary}


def save_result(settings, columns, anchor, report):
    """Publish per-file journals together under locks, rolling back any failed write."""
    from spacemap import labels, pending_columns
    from . import tasks
    table = settings['table']
    selected = settings.get('images') or settings.get('selected_images')
    rows = collection.entries(table, selected)
    replacing = tasks.overwrite_columns(settings)
    collection.validate(table, [], selected, anchors=True)
    report("Saving results with each image's cell table…", .98)
    protect = getattr(report, 'protect_output', None)
    if protect:
        protect()
    ids = np.asarray(anchor).astype(str)
    if len(set(ids)) != len(ids) or any(len(v) != len(ids) for v in columns.values()):
        raise ValueError("Spatial result row identities do not match.")
    written = []
    with ExitStack() as stack:
        for row in sorted(rows, key=lambda r: r['path']):
            stack.enter_context(labels.write_lock(row['path']))
        if tasks.source_revision(settings) != settings['revision']:
            raise ValueError("Source cell tables or labels changed during this run. Start again with current data.")
        plans = []
        for row in rows:
            path, key = row['path'], row['anchor']
            if set(columns) & set(dataio.columns(path)) - replacing:
                raise ValueError(f"{row['image']}: an output column already exists. Earlier results were kept.")
            physical = dataio.load(path, columns=[key], build_fast_copy=False)[key]
            prefix = json.dumps(row['image'], ensure_ascii=False) + ':'
            virtual = prefix + physical.astype('string')
            import pandas as pd
            indices = pd.Index(ids).get_indexer(virtual)
            good = indices >= 0
            if not good.any():
                continue
            # Fill only supplied rows; preserve prior values outside a partial edit.
            values = {}
            for name, data in columns.items():
                old = (dataio.load(path, columns=[name], build_fast_copy=False)[name].to_numpy(copy=True)
                       if name in dataio.columns(path) else np.full(len(physical), None, dtype=object))
                old = old.astype(object)
                old[good] = np.asarray(data)[indices[good]]
                values[name] = pd.Series(old).infer_objects()
            stack.enter_context(tasks._restore_on_failure(path, columns))
            plans.append((path, key, physical, values))
        for path, key, physical, values in plans:
            pending_columns.save(path, values, anchor_column=key, anchor_values=physical,
                                 source_note='SPACEMAP Studio spatial analysis')
            if replacing:
                labels.clear_columns(path, list(replacing.intersection(columns)))
            written.append(path)
    return {"table": table, "tables": written, "image_tables": {r['image']: r['path'] for r in rows}, "column": settings['output_col'],
            "columns": list(columns), "cells": sum(len(p[2]) for p in plans), "cell_id_column": "cell_id"}


def isolate_images(frame, chosen, table):
    """Coordinate spaces never connect across files, even with shared tissue IDs."""
    if not collection.is_collection(table):
        return chosen
    extra = list(dict.fromkeys(chosen[k] for k in ('batch_col', 'roi_col')
                              if chosen.get(k) and chosen[k] != 'batch_id' and chosen[k] in frame))
    region = extra[0] if len(extra) == 1 else ''
    if len(extra) > 1:
        import pandas as pd
        frame['_spatial_region'] = pd.factorize(pd.MultiIndex.from_frame(frame[extra]))[0].astype(float)
        frame.loc[frame[extra].isna().any(axis=1), '_spatial_region'] = np.nan
        region = '_spatial_region'
    return {**chosen, 'batch_col': 'batch_id', 'roi_col': region}
