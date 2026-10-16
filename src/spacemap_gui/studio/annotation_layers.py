"""Saved derived annotations: editable layers, reusable parents and export sources."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from . import annotation, annotation_edits as edits


def record_path(root, image):
    from .annotation_inputs import image_key
    return Path(root) / 'outputs/studio/annotation_layers' / image_key(image) / 'index.json'


def records(root, image):
    path = record_path(root, image)
    return json.loads(path.read_text()) if path.is_file() else {}


def register(root, image, path, *, name, kind, parent=None, settings=None):
    from .settings import project_lock
    with project_lock(root):
        data = records(root, image)
        data[str(Path(path).resolve())] = {
            'name': name, 'kind': kind, 'parent': parent or {},
            'settings': settings or {}, 'revision': edits.revision(path),
        }
        edits._atomic(record_path(root, image), json.dumps(data, indent=2).encode())


def generated(root, image, output):
    """Include pre-index outputs too; never treat History or exports as current layers."""
    sample = annotation.sample_id(image)
    known = {
        str(annotation.classes(output, sample).resolve()): {'name': 'Classification', 'kind': 'classes'},
        str(annotation.marker_regions(output, sample).resolve()): {'name': 'Marker regions', 'kind': 'regions'},
    }
    for pattern, name, kind in ((sample+'_classification_*.geojson', 'Classification', 'classes'),
                                 (sample+'_marker_regions_*.geojson', 'Marker regions', 'regions')):
        for path in Path(output).glob(pattern):
            known[str(path.resolve())] = {'name': name, 'kind': kind}
    known.update(records(root, image))
    return {p: info for p, info in known.items() if Path(p).is_file()}


def layer(root, image, output, path, readonly=False):
    from spacemap_gui.viewer.ui import shapes_layer
    path = Path(path).resolve()
    info = generated(root, image, output).get(str(path), {'name': path.stem, 'kind': 'regions'})
    if path == annotation.classes(output, annotation.sample_id(image)).resolve():
        identifier = 'pixel-classes'
    elif path == annotation.marker_regions(output, annotation.sample_id(image)).resolve():
        identifier = 'marker-regions'
    else:
        identifier = 'annotation-' + sha256(str(path).encode()).hexdigest()[:16]
    result = shapes_layer(path, str(image), name=info['name'], layer_id=identifier,
        regions=True, annotation_kind='regions', editable=not readonly,
        reference=annotation.valid_tissue(output, annotation.sample_id(image)))
    result.update(added=True, annotation_group=info['kind'], origin=origin(info))
    return result


def origin(info):
    """How a saved layer was made, in words (shown under its names in Layers)."""
    settings, parent = info.get('settings') or {}, info.get('parent') or {}
    within = parent.get('class_name') or (Path(parent['path']).stem if parent.get('mode') == 'file' and parent.get('path') else '')
    if settings.get('copied_from'):
        return 'Copy of ' + settings['copied_from']
    if info.get('kind') == 'classes':
        return 'Classification' + (' within ' + within if within else '')
    rules = settings.get('rules') or []
    if rules:
        cutoff = settings.get('cutoff', 'raw')
        how = {'z': 'z ', 'auto': '× ' + str(settings.get('method', 'Auto')) + ' level ', 'saved': '× saved level '}.get(cutoff, '')
        described = ', '.join(f"{rule['marker']} > {how}{rule['above']:g}" for rule in rules)
        return 'Marker regions: ' + described + (' within ' + within if within else '')
    return {'regions': 'Marker regions', 'zones': 'Zones'}.get(info.get('kind'), 'Drawn')


def classification_target(output, image, selection):
    """A subclassification is a child layer; its parent and sibling results survive."""
    if selection.get('mode') != 'file':
        return annotation.classes(output, annotation.sample_id(image))
    identity = json.dumps({k: selection.get(k, '') for k in ('path', 'class_name')}, sort_keys=True)
    return Path(output) / (annotation.sample_id(image)+'_classification_'+sha256(identity.encode()).hexdigest()[:12]+'.geojson')


def export_options(root, config, image, output):
    from . import annotation_inputs as inputs
    generated_paths = generated(root, image, output)
    result = []
    for path, name in inputs.sources(root, config, image, output).items():
        if 'studio_export' in Path(path).parts:
            continue
        features, version = edits.read(path)
        if not features:
            continue
        classes = {}
        for feature in features:
            key = inputs.annotation_class(feature) or feature['properties'].get('name') or 'Unnamed'
            classes[key] = classes.get(key, 0) + 1
        default = path in generated_paths or Path(path) == annotation.valid_tissue(output, annotation.sample_id(image)).resolve()
        result.append({'path': path, 'name': name, 'count': len(features), 'classes': classes,
                       'revision': version, 'selected': default})
    return result


def merge_same_names(features):
    """One polygon (or multipolygon) per annotation name, in first-seen order, so QuPath shows each name as one
    annotation; the merged feature lists the features it came from. Unnamed polygons stay as they are."""
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    from .annotation_inputs import annotation_class
    groups, order, kept = {}, [], []
    for feature in features:
        name = annotation_class(feature) or feature['properties'].get('name') or ''
        if not name:
            kept.append(('', [feature]))
            continue
        if name not in groups:
            groups[name] = []
            order.append(name)
            kept.append((name, groups[name]))
        groups[name].append(feature)
    merged = []
    for name, members in kept:
        if len(members) == 1:
            merged.append(members[0])
            continue
        geometry = make_valid(unary_union([make_valid(shape(f['geometry'])) for f in members]))
        properties = deepcopy(members[0]['properties'])
        # The count and the files, not every source id: a name of 14,000 polygons stays one small annotation.
        properties.pop('source_feature_id', None)
        properties.update(name=name, merged_count=len(members),
                          source_layers=sorted({f['properties'].get('source_layer', '') for f in members}),
                          id=sha256(('merged|' + name).encode()).hexdigest()[:24])
        merged.append({'type': 'Feature', 'geometry': mapping(geometry), 'properties': properties})
    return merged


def export(root, config, image, output, choices, merge=True):
    """Export checked saved layers with names, colours and lineage; with *merge*, polygons with the same name
    become one annotation (exact geometry otherwise)."""
    from .settings import project_lock
    if not isinstance(choices, list) or not choices:
        raise ValueError('Select at least one saved annotation layer.')
    with project_lock(root):
        available = {item['path']: item for item in export_options(root, config, image, output)}
        features, sources, seen = [], [], set()
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('path') not in available:
                raise ValueError('Choose an annotation layer belonging to this image.')
            path = choice['path']
            if path in seen:
                continue
            seen.add(path)
            with edits.locked(path):
                saved, version = edits.read(path)
                if version != choice.get('revision'):
                    raise ValueError('An annotation changed. Refresh the Export list and try again.')
                sources.append({**available[path], 'revision': version})
                for feature in saved:
                    item = deepcopy(feature)
                    properties = item['properties']
                    identifier = properties.get('id', '')
                    properties.update(source_layer=path, source_layer_revision=version, source_feature_id=identifier,
                        id=sha256((path+'|'+str(identifier)).encode()).hexdigest()[:24])
                    features.append(item)
        if merge:
            features = merge_same_names(features)
        target = annotation.exported(annotation.main_output(root, config.get('directory_mappings') or {}), annotation.sample_id(image))
        result = edits.replace(target, features, edits.revision(target), 'Exported annotation layers', kind='regions')
        manifest = target.with_suffix('.sources.json')
        edits._atomic(manifest, json.dumps({'image': str(image), 'layers': sources, 'merged_same_names': bool(merge), 'revision': result['revision']}, indent=2).encode())
        return {'path': str(target), 'filename': target.name, 'count': len(result['features']),
                'layers': len(sources), 'revision': result['revision'],
                'geojson': {'type': 'FeatureCollection', 'features': result['features']}}
