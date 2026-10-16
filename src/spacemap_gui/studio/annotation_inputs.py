"""Image-bound editable imports and saved analysis-area choices."""
from hashlib import sha256
import json
from pathlib import Path
import re
import uuid

from . import annotation, annotation_edits


def image_key(image):
    return sha256(str(Path(image).resolve()).encode()).hexdigest()[:24]


def import_folder(root, image):
    return Path(root) / 'outputs/studio/imported_annotations' / image_key(image)


def imported(root, image):
    return sorted(import_folder(root, image).glob('*.geojson'))


def import_polygons(root, image, name, features):
    from spacemap.viewer.sources import image_info
    from shapely.geometry import box, shape
    clean = annotation_edits.prepare(features)
    if not clean:
        raise ValueError('This file contains no polygon annotations.')
    info = image_info(str(image))
    bounds = box(0, 0, info.width, info.height)
    if not any(shape(f['geometry']).intersects(bounds) for f in clean):
        raise ValueError('These polygons are outside this image. Import GeoJSON in this image’s pixel coordinates.')
    stem = re.sub(r'[^\w .-]+', '_', Path(str(name)).stem)[:70] or 'Imported annotations'
    target = import_folder(root, image) / (uuid.uuid4().hex[:8] + '_' + stem + '.geojson')
    annotation_edits.replace(target, clean, annotation_edits.revision(target), 'Imported annotations', edited=True)
    return target


def annotation_class(feature):
    value = feature.get('properties', {}).get('classification')
    return str(value.get('name') or '') if isinstance(value, dict) else str(value or '')


def tissue_import(image, features, class_name, *, label='tissue', source=''):
    """Prepare one source class as editable Tissue without writing either file."""
    from shapely.geometry import box, shape
    from spacemap.viewer.sources import image_info
    if class_name is not None and not isinstance(class_name, str):
        raise ValueError('Choose the annotation to use as Tissue.')
    selected = [f for f in features if class_name is None or annotation_class(f) == class_name]
    if not selected:
        raise ValueError('No polygons have that annotation name. Choose another annotation.')
    clean = annotation_edits.prepare(selected, kind='tissue', label=label)
    if not clean:
        raise ValueError('This annotation contains no polygon areas.')
    info = image_info(str(image))
    bounds = box(0, 0, info.width, info.height)
    if not any(shape(f['geometry']).intersection(bounds).area > 0 for f in clean):
        raise ValueError('These polygons are outside this image. Use annotations in this image’s pixel coordinates.')
    for feature in clean:
        feature['properties']['tissue_source'] = str(source)
        feature['properties']['tissue_source_class'] = class_name
    return clean


def layer(path, image, readonly=False):
    from spacemap_gui.viewer.ui import shapes_layer
    result = shapes_layer(path, str(image), name=Path(path).stem.split('_', 1)[-1],
        layer_id='imported-' + sha256(str(path).encode()).hexdigest()[:16], regions=True,
        annotation_kind='regions', editable=not readonly)
    result.update(annotation_group='imported', added=True, source_name=Path(path).stem.split('_', 1)[-1])
    return result


def sources(root, config, image, output):
    from . import annotation_layers
    from spacemap.viewer.catalog import find_annotations, sample_of
    from .annotation_names import hidden_files, stale_domains
    hidden = hidden_files(root, image)  # files taken off this image's list in Layers
    entries = {str(Path(path).resolve()): label for label, path in find_annotations(root, config.get('directory_mappings', {}), sample_of(Path(image).name))
               if Path(path).is_file() and 'studio_history' not in Path(path).parts and str(Path(path).resolve()) not in hidden}
    for path in imported(root, image):
        entries[str(path.resolve())] = path.stem.split('_', 1)[-1]
    entries.update({path: info['name'] for path, info in annotation_layers.generated(root, image, output).items()})
    entries.pop(str(annotation.cleaned_tissue(output, annotation.sample_id(image)).resolve()), None)
    if stale_domains(root, image, output):  # left from a deleted Classification
        entries.pop(str(annotation.domains(output, annotation.sample_id(image)).resolve()), None)
    sample=annotation.sample_id(image)
    for name, path in [(annotation.tissue_label(config), annotation.valid_tissue(output, sample)),
                       *([] if stale_domains(root, image, output) else [('Named classes', annotation.domains(output, sample))]),
                       ('Zones', annotation.zones(annotation.zones_folder(root,config.get('directory_mappings',{})), sample))]:
        if path.is_file(): entries[str(path.resolve())] = name
    return entries


def source_options(root, config, image, output, entries):
    """Describe each source once, retaining saved labels and its origin."""
    sample = annotation.sample_id(image)
    label = annotation.tissue_label(config)
    tissue = [
        {'mode': mode, 'name': label, 'path': str(path.resolve()),
         'available': path.is_file()}
        for mode, version, path in (
            ('tissue', 'Tissue', annotation.valid_tissue(output, sample)),
        )
    ]
    builtins = {item['path'] for item in tissue}
    import_paths = {str(path.resolve()) for path in imported(root, image)}
    files = [{'path': path, 'name': name,
              'kind': 'imported' if path in import_paths else 'file'}
             for path, name in entries.items() if path not in builtins]
    return {'tissue': tissue, 'files': files,
            'automatic': 'tissue'}


def _choice_file(root, image, stage):
    if stage not in ('tissue', 'working', 'classes', 'regions', 'clot', 'blur', 'bright', 'lint', 'debris'):
        raise ValueError('Choose an annotation step.')
    return Path(root) / 'outputs/studio/annotation_inputs' / image_key(image) / (stage + '.json')


def selection(root, image, stage):
    shared = _choice_file(root, image, 'working')
    if shared.is_file(): return json.loads(shared.read_text())
    path = _choice_file(root, image, stage)
    if path.is_file():
        value = json.loads(path.read_text())
        if value.get('mode') in ('auto', 'cleaned'): value = {'mode':'tissue','class_name':''}
        return value
    if stage == 'regions':
        from . import regions
        from .settings import read_config
        if regions.settings(read_config(root))['within'] == 'image': return {'mode':'image','class_name':''}
    return {'mode': 'tissue', 'class_name': ''}


def save_selection(root, config, image, output, stage, value):
    if not isinstance(value, dict) or value.get('mode') not in ('auto','image','tissue','cleaned','file'):
        raise ValueError('Choose the area to analyse.')
    value = {k:v for k,v in value.items() if k in ('mode','path','class_name')}
    if value['mode'] in ('auto', 'cleaned'): value = {'mode':'tissue','class_name':''}
    if not isinstance(value.get('class_name',''), str):
        raise ValueError('Choose an annotation name.')
    if value['mode'] == 'file' and value.get('path') not in sources(root, config, image, output):
        raise ValueError('Choose an annotation belonging to this image.')
    from .settings import project_lock
    with project_lock(root):
        annotation_edits._atomic(_choice_file(root,image,stage), json.dumps(value).encode())
        annotation_edits._atomic(_choice_file(root,image,'working'), json.dumps(value).encode())
    return value


def working_path(root, image, output, stage='tissue'):
    """Editable source, never the derived GeoJSON used to filter a class for ML."""
    value = selection(root, image, stage)
    if value['mode'] == 'image':
        raise ValueError('Choose a Tissue or annotation layer in Work within before subtracting.')
    return (Path(value['path']) if value['mode'] == 'file'
            else annotation.valid_tissue(output, annotation.sample_id(image)))


def working_layer(root, config, image, output, stage='tissue', readonly=False):
    from . import annotation_layers
    from spacemap_gui.viewer.ui import shapes_layer
    target = working_path(root, image, output, stage)
    canonical = target.resolve() == annotation.valid_tissue(output, annotation.sample_id(image)).resolve()
    if not canonical:
        result = (annotation_layers.layer(root, image, output, target, readonly)
                  if str(target.resolve()) in annotation_layers.generated(root, image, output)
                  else layer(target, image, readonly))
        result['working_tissue'] = True
        result['working_class'] = selection(root,image,stage).get('class_name','')
        if result['working_class']:
            result['classes'] = [{'name':result['working_class'],'color':'#2ecc71'}]
            result['default_class'] = result['working_class']
        return result
    result = shapes_layer(target, str(image), name=annotation.tissue_label(config), layer_id='tissue-regions',
        regions=True, annotation_kind='tissue', reference=target, tissue_label=annotation.tissue_label(config),
        editable=not readonly, classes=[{'name':'Valid_Tissue','color':'#2ecc71'}], default_class='Valid_Tissue')
    result.update(added=True, working_tissue=True)
    return result


def prepare_working(root, config, image, output, stage='tissue'):
    """Use an editable project copy when the chosen source is external."""
    annotation.unify_tissue(output, annotation.sample_id(image), label=annotation.tissue_label(config))
    value = selection(root, image, stage)
    target = working_path(root, image, output, stage)
    canonical = annotation.valid_tissue(output, annotation.sample_id(image))
    from . import annotation_layers
    managed = str(target.resolve()) in annotation_layers.generated(root, image, output)
    if value['mode'] == 'file' and target.resolve() != canonical.resolve() and target not in imported(root, image) and not managed:
        if str(target.resolve()) not in sources(root, config, image, output):
            raise ValueError('Choose an annotation belonging to this image.')
        features, version = annotation_edits.read(target)
        for feature in features:
            feature['properties'].update(tissue_source=str(target), tissue_source_revision=version)
        copied = import_polygons(root, image, target.stem, features)
        value = {**value, 'path':str(copied.resolve())}
    return save_selection(root, config, image, output, stage, value)


def scope_path(root, image, output, stage):
    """The saved analysis choice; layer visibility never changes it."""
    from spacemap.viewer.sources import image_info
    from shapely.geometry import box, mapping
    value = selection(root,image,stage)
    mode = value['mode']; sample = annotation.sample_id(image)
    if mode == 'auto': return annotation.analysis_tissue(output,sample)
    if mode == 'tissue': return annotation.valid_tissue(output,sample)
    if mode == 'cleaned':
        return annotation.valid_tissue(output,sample)
    if mode == 'image':
        info = image_info(str(image))
        features = [{'type':'Feature','properties':{},'geometry':mapping(box(0,0,info.width,info.height))}]
    else:
        path = Path(value['path'])
        if not path.is_file(): raise ValueError('The chosen annotation file is missing. Choose another Work within area.')
        if not value.get('class_name'): return path
        features, _ = annotation_edits.read(path)
        from . import annotation_names
        from .settings import read_config
        entry = next((e for e in annotation_names.files(root, read_config(root), image, output)
                      if e['path'] == str(path.resolve())), None)
        # The name as Layers shows it (a renamed tissue piece keeps the tissue class), or the class itself.
        features = [f for f in features if annotation_class(f) == value['class_name'] or (entry and
                    annotation_names.name_of(f, entry['kind'], entry['layer']) == value['class_name'])]
        if not features: raise ValueError('No polygons have that annotation name.')
    raw = json.dumps({'type':'FeatureCollection','features':features}).encode()
    target = _choice_file(root,image,stage).parent / (sha256(raw).hexdigest()+'.geojson')
    if not target.is_file(): annotation_edits._atomic(target,raw)
    return target
