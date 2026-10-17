"""Durable catalogue of annotation assets for a future clean-tissue workflow.

This records saved data and provenance, not a claim that every saved polygon
has been reviewed. Large images stay in place; JSON references their paths.
"""
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import uuid

from . import annotation, annotation_edits, annotation_inputs, artifact_learning, artifact_quality, artifact_sampling, artifact_forest, pieces
from spacemap.sharing import open_lock

RELATIVE_PATH = 'outputs/studio/annotation_workflow.json'
ANNOTATION_JOBS = frozenset({'detect_tissue', 'clean_tissue', 'detect_artifacts', 'artifact_batch', 'subtract_annotations', 'classify_shared', 'save_class_polygons', 'remove_small_pieces',
    'train_artifact_model', 'subtract_artifacts', 'apply_pieces', 'export_classes',
    'make_zones', 'marker_regions', 'name_clusters', 'export_annotations'})


def _read(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path, data):
    annotation_edits._atomic(path, json.dumps(data, indent=2, allow_nan=False).encode())


def _relative(root, path):
    path = Path(path).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


@lru_cache(maxsize=1)
def software():
    packages = {}
    for name in ('numpy', 'scikit-learn', 'scipy', 'scikit-image', 'shapely', 'joblib', 'tifffile', 'torch'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Capture the actual implementation, including uncommitted local fixes.
    sources = {}
    for path in [Path(module.__file__) for module in (annotation, annotation_edits, artifact_learning, artifact_quality, artifact_sampling, artifact_forest, pieces)]:
        sources[path.name] = sha256(path.read_bytes()).hexdigest()
    return {'python': platform.python_version(), 'packages': packages, 'source_sha256': sources}


def record(root, *, event=None, image=None, target=None, mappings=None):
    """Atomically refresh the index and append an immutable operation record.

    Annotation calls run this after their own file locks have been released.
    A separate lock serializes API and job writers. Unchanged assets reuse
    fingerprints; TIFF pixel data and model objects are never loaded here.
    """
    import fcntl
    from .settings import read_config

    root = Path(root).resolve()
    path = root / RELATIVE_PATH
    home = path.parent / 'annotation_workflow'
    home.mkdir(parents=True, exist_ok=True)
    with open_lock(home / 'record.lock') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        old = _read(path, {})
        previous = old.get('assets', {})
        assets = dict(previous)  # Keep links even if folders/images are later changed.
        warnings = []

        def asset(location, role, *, image_file=False):
            location = Path(location).resolve()
            key = _relative(root, location)
            entry = {'path': key, 'role': role, 'exists': location.is_file()}
            if entry['exists']:
                stat = location.stat()
                entry.update(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)
                before = previous.get(key, {})
                same = all(before.get(k) == entry[k] for k in ('bytes', 'modified_ns'))
                if image_file:
                    if same and before.get('image_info'):
                        entry['image_info'] = before['image_info']
                    else:
                        try:
                            from spacemap.viewer.sources import image_info
                            info = image_info(str(location))
                            entry['image_info'] = {'width': info.width, 'height': info.height,
                                'pixel_size_um': info.pixel_size_um, 'channels': list(info.channel_names)}
                        except (ValueError, OSError, KeyError) as exc:
                            warnings.append(f'Image metadata unavailable for {key}: {exc}')
                if not image_file:
                    entry['sha256'] = before['sha256'] if same and before.get('sha256') else sha256(location.read_bytes()).hexdigest()
                if location.suffix == '.geojson':
                    if same and 'polygon_count' in before:
                        for field in ('polygon_count', 'classes', 'metadata', 'sources'):
                            if field in before:
                                entry[field] = before[field]
                    else:
                        try:
                            data = _read(location, {})
                            features = data.get('features', []) if isinstance(data, dict) else data
                            entry.update(polygon_count=len(features), classes=sorted({annotation_inputs.annotation_class(f) for f in features}))
                            entry['metadata'] = data.get('studio', {}) if isinstance(data, dict) else {}
                            entry['sources'] = list({json.dumps({k: v for k, v in f.get('properties', {}).items()
                                if k in ('tissue_source', 'tissue_source_class', 'tissue_source_revision')}, sort_keys=True)
                                for f in features if f.get('properties', {}).get('tissue_source')})
                            entry['sources'] = [json.loads(item) for item in sorted(entry['sources'])]
                        except (ValueError, TypeError, AttributeError) as exc:
                            warnings.append(f'Could not index {key}: {exc}')
                    history = annotation_edits._home(location)
                    if history.is_dir():
                        entry['history'] = _relative(root, history)
                    entry['review_status'] = 'not_declared'  # Saved is not ground truth.
            assets[key] = entry
            return key

        config = read_config(root)
        mappings = {**old.get('directory_mappings', {}), **(mappings or {}), **config.get('directory_mappings', {})}
        folders = annotation.all_folders(root, mappings)
        output = folders['output']
        # Only known annotation settings are exported, never the full project config.
        settings = {
            'tissue': annotation.settings(config), 'tissue_label': annotation.tissue_label(config),
            'pieces': pieces.settings(config),
            'artifacts': {part: artifact_learning.options(root, part) for part in artifact_learning.NAMES},
            'folders': {key: _relative(root, value) for key, value in folders.items()},
        }
        from . import pixelclasses, regions
        settings.update(pixel_classes=pixelclasses.settings(config), marker_regions=regions.settings(config))
        settings_id = sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        settings_path = home / 'settings' / (settings_id + '.json')
        if not settings_path.exists():
            _write(settings_path, settings)

        images = dict(old.get('images', {}))
        incremental = bool(old and event and image)
        candidates = set() if incremental else set(Path(key) if Path(key).is_absolute() else root / key for key in images)
        if not incremental and folders['images'].is_dir():
            candidates.update(p for p in folders['images'].iterdir() if p.is_file() and p.name.lower().endswith(annotation.IMAGE_EXTENSIONS))
        if image:
            candidates.add(Path(image))
        for candidate in sorted(candidates):
            key = asset(candidate, 'image', image_file=True)
            sample = annotation.sample_id(candidate)
            saved = dict(images.get(key, {}))
            layers = dict(saved.get('layers', {}))
            paths = {
                'tissue_original': annotation.valid_tissue(output, sample),
                'tissue_clean': annotation.cleaned_tissue(output, sample),
                'tissue_pieces': pieces.review(output, sample),
                'artifacts_drawn': annotation.drawn_artifacts(output, sample),
                'artifacts_detected': annotation.detected_artifacts(output, sample),
                'pixel_classes': annotation.classes(output, sample),
                'named_classes': annotation.domains(output, sample),
                'marker_regions': annotation.marker_regions(output, sample),
                'zones': annotation.zones(folders['zones'], sample),
            }
            for part in artifact_learning.NAMES:
                paths['artifact_' + part] = artifact_learning.polygons_path(output, sample, part)
                paths['examples_' + part] = artifact_learning.example_path(root, part, candidate)
            for role, location in paths.items():
                if location.is_file():
                    layers[role] = asset(location, role)
            inputs = {stage: annotation_inputs.selection(root, candidate, stage)
                      for stage in ('clot', 'blur', 'bright', 'lint', 'debris', 'classes', 'regions')}
            external = {asset(location, 'external_annotation'): label for location, label in
                        annotation_inputs.sources(root, config, str(candidate), output).items()
                        if Path(location).is_file()}
            from . import classification
            classification_records={p.stem:asset(p,'classification_review') for p in classification.directory(root,candidate).glob('*.json')}
            from . import annotation_layers
            derived = {asset(p, info['kind']): info for p, info in annotation_layers.generated(root,candidate,output).items()}
            images[key] = {**saved, 'sample': sample, 'layers': layers, 'analysis_inputs': inputs, 'classification_records':classification_records,
                'derived_annotations':derived,
                'external_annotations': external, 'final_review': saved.get('final_review', 'not_declared')}

        libraries = {}
        for part in artifact_learning.NAMES:
            library = artifact_learning.folder(root, part)
            current = _read(library / 'examples.json', {})
            versions = _read(library / 'versions.json', [])
            snapshots = {}
            ids = {item['id'] for item in current.values()}
            ids.update(item for model in versions for item in model.get('examples', []))
            # Preserve superseded submissions too, including ones never used in a model.
            ids.update(p.stem for p in library.glob('*.npz'))
            sources = {identifier: part for identifier in ids}
            for model in versions:
                sources.update(model.get('example_sources', {}))
            for identifier in sorted(ids):
                source = sources[identifier]
                origin = artifact_learning.folder(root, source)
                meta = origin / (identifier + '.json')
                old_snapshot = old.get('artifact_libraries', {}).get(part, {}).get('snapshots', {}).get(identifier, {})
                previous_meta = previous.get(_relative(root, meta), {})
                unchanged = meta.is_file() and previous_meta.get('modified_ns') == meta.stat().st_mtime_ns
                snapshot = old_snapshot if unchanged else _read(meta, {})
                snapshots[identifier] = {k: v for k, v in snapshot.items() if k != 'annotations'}
                snapshots[identifier].update(annotations=asset(meta, 'training_submission'),
                    samples=asset(origin / (identifier + '.npz'), 'training_samples'), source_part=source,
                    label_status='explicit_training_submission')
            libraries[part] = {
                'folder': _relative(root, library), 'active_examples': current,
                'model_selection': _read(library / 'selection.json', {}),
                'snapshots': snapshots,
                'models': [{**model, 'file': asset(library / (model['id'] + '.joblib'), 'artifact_model')}
                           for model in versions],
            }
        if target:
            asset(target, 'edited_annotation')
        now = datetime.now(timezone.utc).isoformat()
        event_path = None
        if event:
            entry = {**event, 'recorded_at': now, 'settings': _relative(root, settings_path), 'software': software()}
            if image:
                entry['image'] = _relative(root, image)
            if target:
                entry.update(target=_relative(root, target), saved_revision=annotation_edits.revision(target))
            event_path = home / 'events' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ_') + uuid.uuid4().hex[:8] + '.json')
            _write(event_path, entry)
        data = {
            'schema_version': 1, 'project_root': str(root), 'created_at': old.get('created_at', now), 'updated_at': now,
            'path_base': 'project_root; absolute paths refer to external files',
            'directory_mappings': mappings, 'settings': settings, 'settings_snapshot': _relative(root, settings_path),
            'images': images, 'artifact_libraries': libraries, 'assets': assets,
            'events_directory': _relative(root, home / 'events'),
            'latest_event': _relative(root, event_path) if event_path else old.get('latest_event'),
            'software_at_indexing': software(), 'warnings': warnings,
            'label_policy': {'saved_polygons': 'Review status is not inferred from saving or prediction.',
                             'training_submissions': 'Explicit examples supplied for the named artifact, not final clean-tissue ground truth.',
                             'unsaved_previews': 'Not promoted to saved polygons; operation settings are recorded.',
                             'historical_provenance': 'Earlier files are indexed as found; unrecorded past tool settings cannot be reconstructed.'},
            'future_workflow': {'status': 'roadmap_not_implemented',
                'stages': ['detect_or_import_tissue', 'predict_each_artifact', 'subtract_artifacts', 'cut_necks', 'filter_fragments', 'review_clean_tissue'],
                'geometry_units': 'GeoJSON coordinates, brush radius, split width and merge gap: full-resolution pixels; tissue/neck settings retain their named physical units.',
                'training_preparation': ['Explicit final review and ignored regions', 'Separate artifact labels from size/neck cleanup decisions',
                    'Patient/image/acquisition grouping and held-out validation images', 'Validate missing markers and detection scales']},
        }
        _write(path, data)
        return data
