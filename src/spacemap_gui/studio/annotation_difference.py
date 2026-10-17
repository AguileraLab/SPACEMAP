"""Exact annotation differences, saved as independent, reusable polygon layers."""
from hashlib import sha256
from functools import lru_cache
import json
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from . import annotation, annotation_edits as edits, annotation_layers as layers
from .settings import project_lock


@lru_cache(maxsize=256)
def _revision(path, modified_ns, changed_ns, size):
    return edits.revision(path)


def revision_of(path):
    stat = Path(path).stat()
    return _revision(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def catalog(root, config, image, output):
    """The image's annotations by name, as Layers lists them (the image's own, then each imported file): one entry
    per name with its polygon count and the files (with their revisions) holding it. Training examples are left out."""
    from . import annotation_names

    result = []
    for group in annotation_names.groups(root, config, image, output):
        if group["role"] == "good":
            continue
        files = [path for path in group["files"] if "studio_export" not in Path(path).parts]
        if not files or not group["count"]:
            continue
        result.append({"key": group["key"], "source": group["source"], "name": group["name"], "role": group["role"],
                       "count": group["count"], "colour": group["colour"],
                       "revisions": {path: revision_of(path) for path in files}})
    return result


def create(root, config, image, output, body):
    """Union the bases, subtract the exclusions and retain exact source provenance.

    Work on immutable snapshots; check their revisions again at commit. The
    sources are never rewritten, including externally loaded read-only files.
    """
    label = body.get('label', '')
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > 60:
        raise ValueError('Enter a result label (up to 60 characters), for example Stroma.')
    label = label.strip()
    action = body.get('action', 'save')
    if action not in ('preview', 'save'):
        raise ValueError('Choose Preview or Create layer.')
    from . import annotation_names

    entries = {entry["path"]: entry for entry in annotation_names.files(root, config, image, output)}
    groups = {item["key"]: item for item in catalog(root, config, image, output)}
    label_of_tissue = annotation.tissue_label(config)
    snapshots = {}

    def operand(choices, may_be_empty=False):
        if may_be_empty and choices == []:
            return [], unary_union([])
        if not isinstance(choices, list) or not choices:
            raise ValueError('Choose at least one annotation on each side of the minus sign.')
        references, geometry, seen = [], [], set()
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('key') not in groups:
                raise ValueError('That annotation is no longer on this image. Refresh the subtraction choices.')
            group = groups[choice['key']]
            if choice['key'] in seen:
                continue
            seen.add(choice['key'])
            role = group['role']
            selected = []
            for path in group['revisions']:
                if path not in snapshots:
                    snapshots[path] = annotation_names._read(entries[path], label_of_tissue)
                features, version = snapshots[path]
                if version != (choice.get('revisions') or {}).get(path):
                    raise ValueError('An annotation changed. Refresh the subtraction choices and preview again.')
                entry = entries[path]
                selected += [f for f in features if annotation_names.name_of(f, entry['kind'], entry['layer']) == group['name']]
                references.append({'path': path, 'name': group['name'], 'source': group['source'], 'role': role,
                                   'revision': version, 'layer_name': entry['layer']})
            if not selected:
                raise ValueError(group['name'] + ' no longer has polygons. Refresh the subtraction choices.')
            geometry.extend(shape(f['geometry']) for f in selected)
        references.sort(key=lambda item: (item['path'], item['name']))
        return references, unary_union(geometry)

    base, base_geometry = operand(body.get('base'))
    subtract, subtract_geometry = operand(body.get('subtract'), may_be_empty=bool(body.get('by_name')))
    recipe = {'operation': 'difference', 'label': label, 'base': base, 'subtract': subtract}
    identity = {'label': label, **{side: [{k: item[k] for k in ('path', 'name')} for item in items]
                                 for side, items in (('base', base), ('subtract', subtract))}}
    key = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    target = Path(output) / (annotation.sample_id(image) + '_marker_regions_difference_' + key + '.geojson')
    if str(target.resolve()) in snapshots:
        raise ValueError('Choose a new result label so the input annotation is preserved.')
    before, expected = edits.read(target)
    if body.get('target_revision') is not None and body['target_revision'] != expected:
        raise ValueError('The result layer changed. Preview again before replacing it.')
    difference = base_geometry.difference(subtract_geometry)
    if not difference.is_valid:
        difference = make_valid(difference)
    features = [{'type': 'Feature', 'geometry': mapping(polygon), 'properties': {
        'name': label, 'label': label, 'named': True,
        'classification': {'name': label, 'color': [86, 180, 166]},
        'annotation_operation': 'difference', 'annotation_recipe': key,
    }} for polygon in edits.polygons(difference)]
    result = {'count': len(features), 'area_px2': difference.area, 'path': str(target.resolve()),
              'target_revision': expected, 'recipe': recipe}
    if action == 'preview':
        display = edits.display_features(edits.prepare(features))
        return {**result, 'outlines': [ring for f in display for polygon in edits.polygons(shape(f['geometry']))
                                     for ring in mapping(polygon)['coordinates']]}
    if not features:
        raise ValueError('Nothing remains after subtraction. Choose different annotations; no layer was changed.')
    with project_lock(root):
        for path, (_, version) in snapshots.items():
            if edits.revision(path) != version:
                raise ValueError('An annotation changed during subtraction. Refresh and preview again.')
        saved = edits.replace(target, features, expected, 'Created ' + label + ' by subtraction')
        layers.register(root, image, target, name=label, kind='regions', parent=recipe)
    return {**result, **saved, 'before': edits.display(before, expected)['features'],
            'layer': layers.layer(root, image, output, target)}


def create_by_names(root, config, image, output, label, base, subtract):
    """The same subtraction on another image, by name: *base* and *subtract* are [{"name", "role"}]; each finds this
    image's annotations of that name (its own and its imported files'). A name it does not have is left out of the
    minus side (nothing to take away); without any base name the image is skipped."""
    groups = catalog(root, config, image, output)

    def picks(items):
        wanted = {(item.get("name"), item.get("role", "")) for item in items}
        return [{"key": g["key"], "revisions": g["revisions"]} for g in groups if (g["name"], g["role"]) in wanted]

    chosen_base = picks(base)
    if not chosen_base:
        raise ValueError("it has no " + " or ".join(item["name"] for item in base))
    chosen_subtract = picks(subtract)
    result = create(root, config, image, output, {"label": label, "action": "save", "by_name": True,
                                                 "base": chosen_base, "subtract": chosen_subtract})
    present = {g["name"] for g in groups}
    return {"count": result["count"], "path": result["path"], "recipe": result["recipe"],
            "missing": [item["name"] for item in subtract if item["name"] not in present]}
