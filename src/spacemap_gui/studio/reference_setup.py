"""Project model selection and validation settings; originals are never overwritten."""

import hashlib
import json
from pathlib import Path
import re
import uuid

from spacemap import dataio
from spacemap_workflows.leiden_job import write_json
from . import model_library, reference_models
from .settings import project_lock

# Populated only by explicit loading of legacy models, never catalogue polling.
_LOADED_DETAILS = {}


def _path(root):
    return Path(root) / 'outputs' / 'studio' / 'reference_setup.json'


def saved(root):
    path = _path(root)
    return json.loads(path.read_text()) if path.exists() else {}


def training(root):
    value = saved(root)
    listed = project_models(root, value.get('folder') or model_library.location())
    identifier = value.get('model_id', '')
    return {**phenotype_policy(root), 'training_percent': reference_models.training_percent(value.get('training_percent', 70)),
            'base_model_id': identifier if identifier in listed else '', 'model_timezone': value.get('timezone', 'UTC'), 'include_saved_examples': value.get('include_saved_examples', True),
            'model_name': value.get('model_name', ''), 'models_folder': value.get('folder') or str(model_library.location())}


def phenotype_policy(root):
    value = saved(root)
    return {**{key: value.get(key, []) for key in ('excluded_training_phenotypes', 'excluded_prediction_phenotypes')},
            'phenotype_policy_configured': 'excluded_training_phenotypes' in value}


def _entries(folder):
    """Read metadata only. Loading a selected, trusted model is a separate action."""
    folder = Path(folder)
    found = []
    candidates = list(folder.glob('*.json'))
    candidates += list(folder.glob('*/model.json')) + list(folder.glob('phenotyping/*/model.json'))
    for path in candidates:
        try:
            raw = json.loads(path.read_text())
            if path.name == 'model.json':
                if raw.get('kind') != 'phenotyping':
                    continue
                meta, identifier = raw['record'], raw['id']
            else:
                meta, identifier = raw, path.stem
            if not re.fullmatch(r'[0-9a-f]{32}', identifier):
                continue
            artifact = path.parent / (identifier + '.joblib')
            if not artifact.is_file() or not meta.get('markers') or meta.get('method') not in {'Random Forest', 'Extra Trees'}:
                continue
            found.append({'key': hashlib.sha256(str(artifact.resolve()).encode()).hexdigest(),
                          'path': str(artifact.resolve()), 'meta': meta, 'id': identifier})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def _listed(folder):
    entries = _entries(folder)
    by_id = {item['id']: item for item in entries}
    # A cached copy is an alias only while its source is in this directory.
    for item in _entries(reference_models.directory()):
        origin = item['meta'].get('imported_from') or {}
        source = origin.get('path') if isinstance(origin, dict) else None
        shared = next((entry for entry in entries if entry['path'] == source), None)
        if shared:
            by_id.pop(shared['id'], None)
            by_id[item['id']] = {**item, 'key': shared['key'], 'shared_path': str(Path(shared['path']).parent)}
        elif item['id'] in by_id:
            by_id[item['id']] = {**item, 'shared_path': str(Path(by_id[item['id']]['path']).parent)}
    return by_id


def project_models(root, folder=None):
    """One catalogue for Manual reference, cluster suggestions and Lists."""
    from . import project_files
    folder = folder or saved(root).get('folder') or model_library.location()
    found = _listed(folder)
    if not saved(root).get('folder'):
        for item in _entries(Path(root) / 'data/reference_models'):
            found[item['id']] = item
    return {identifier:item for identifier,item in found.items() if not project_files.excluded(item['path'], root)
            and not project_files.excluded(reference_models.model_path(identifier), root)}


def options(root, table=None):
    value = saved(root)
    folder = Path(value.get('folder') or model_library.location()).expanduser()
    by_id = project_models(root, folder)
    from . import phenotyping_inputs, reference_examples
    tables = {str(table)} if table else set()
    if table:
        tables.update(p for p in phenotyping_inputs.image_tables(root, table).values() if p)
    columns = set(dataio.columns(table)) if table else set()
    models = []
    for item in sorted(by_id.values(), key=lambda item: item['meta'].get('trained', ''), reverse=True):
        meta = item['meta']
        classes = meta.get('classes') or list(meta.get('held_out_per_class', {}))
        stamp = Path(item['path']).stat()
        loaded = _LOADED_DETAILS.get((item['id'], stamp.st_mtime_ns, stamp.st_size), {})
        classes = classes or loaded.get('classes', [])
        # Listing choices must remain metadata-only, including the active model.
        # Legacy manifests can omit replay counts/classes; prediction and the
        # explicit model inspection action read the artifact when needed.
        reusable = meta.get('reusable_examples', loaded.get('reusable_examples'))
        current_project = (meta.get('project_key') == reference_examples.source_key(root) or
                           meta.get('table_path') in tables)
        missing = [name for name in meta['markers'] if name not in columns]
        compatible = bool(columns.intersection(meta['markers'])) and (not missing or
                         (meta.get('adaptable') and meta.get('allow_missing', True)))
        models.append({key: item[key] for key in ('key', 'path', 'id')} | {
            'method': meta['method'], 'name': meta.get('name', ''), 'column': meta.get('column', ''), 'trained': meta.get('trained', ''),
            'parent': meta.get('base_model_id', ''), 'features': len(meta['markers']),
            'markers': meta['markers'], 'adaptable': bool(meta.get('adaptable')), 'allow_missing': meta.get('allow_missing', True),
            'missing': missing, 'compatible': bool(compatible), 'training_cells': meta.get('training_cells'),
            'validation_cells': meta.get('held_out_cells'), 'validation_held_out': bool(meta.get('validation_held_out')),
            'accuracy': meta.get('held_out_balanced_accuracy'), 'f1': meta.get('held_out_macro_f1'),
            'saved_example_cells': meta.get('saved_example_cells', 0), 'unused_features': meta.get('unused_features', []),
            'classes': classes, 'reusable_examples': reusable,
            'needs_examples': meta.get('needs_examples', []) if current_project else [],
            'shared_path': item.get('shared_path') or shared_path(folder, item['id'])})
    return {**training(root), 'folder': str(folder), 'models': models}


def configure(root, body, table=None):
    action = body.get('action')
    updates = {}
    if body.get('timezone'):
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(body['timezone'])
        except (ValueError, TypeError, ZoneInfoNotFoundError) as exc:
            raise ValueError('Unknown time zone.') from exc
        updates['timezone'] = body['timezone']
    if action == 'folder':
        text = body.get('folder')
        if not isinstance(text, str) or not text.strip() or len(text) > 2048:
            raise ValueError('Enter a model directory.')
        folder = Path(text.strip()).expanduser()
        if not folder.is_absolute():
            folder = Path(root) / folder
        if not folder.is_dir():
            raise ValueError('Choose an existing model directory.')
        updates['folder'] = str(folder.resolve())
        if saved(root).get('model_id') not in _listed(folder):
            updates['model_id'] = ''
        from . import auto_reference
        auto_reference.pause_for_folder(root)
    elif action == 'split':
        updates['training_percent'] = reference_models.training_percent(body.get('training_percent'))
    elif action == 'training':
        if not isinstance(body.get('include_saved_examples'), bool):
            raise ValueError('Choose whether to include saved examples.')
        name = body.get('model_name', '')
        if not isinstance(name, str) or len(name) > 120 or any(ord(c) < 32 for c in name):
            raise ValueError('Use a short model name (up to 120 characters).')
        updates.update(include_saved_examples=body['include_saved_examples'], model_name=name.strip())
    elif action == 'phenotypes':
        from spacemap.phenotype_policy import validate_names
        updates.update({key: validate_names(body.get(key)) for key in
                        ('excluded_training_phenotypes', 'excluded_prediction_phenotypes')})
    elif action == 'share':
        identifier = saved(root).get('model_id')
        if not identifier:
            raise ValueError('Load a model first.')
        publish(root, identifier)
    elif action == 'load':
        key = body.get('key')
        if key == '':
            updates['model_id'] = ''
        else:
            identifier = prepare_model(root, key, table)
            updates.update(model_id=identifier, include_saved_examples=True)
    else:
        raise ValueError('Choose a model folder, model or training split.')
    with project_lock(root):
        path = _path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {**saved(root), **updates})
    return options(root, table)


def prepare_model(root, key, table=None):
    """Load a chosen library model without changing training or auto-train settings."""
    item = next((entry for entry in options(root, table)['models'] if entry['key'] == key), None)
    if item is None:
        raise ValueError('Choose a listed phenotyping model.')
    source = Path(item['path'])
    if source.parent == reference_models.directory().resolve():
        identifier = item['id']
        if not item['classes'] or item['reusable_examples'] is None:
            from . import reference_examples
            _, meta, artifact = reference_models._source(identifier)
            stamp = source.stat()
            if len(_LOADED_DETAILS) >= 64:
                _LOADED_DETAILS.clear()
            _LOADED_DETAILS[(identifier, stamp.st_mtime_ns, stamp.st_size)] = {
                'classes': list(map(str, artifact['model'].classes_)),
                'reusable_examples': sum(len(g['targets']) for g in reference_examples.archive(meta, artifact))}
    else:
        # Copy to a new identity, even if an older model has the same ID.
        # The trained or shared source stays unchanged.
        import joblib
        entry = next(entry for entry in _entries(source.parent) if entry['path'] == str(source))
        artifact = joblib.load(source)
        if not isinstance(artifact, dict) or 'model' not in artifact or 'provenance' not in artifact:
            raise ValueError('This is not a SPACEMAP phenotyping model.')
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from spacemap.viewer import consensus_model
        if not isinstance(artifact['model'], (RandomForestClassifier, ExtraTreesClassifier)):
            raise ValueError('Choose a Random Forest or Extra Trees model.')
        if entry['meta'].get('sklearn') != consensus_model._version():
            raise ValueError('This model needs the same scikit-learn version used for training.')
        identifier = uuid.uuid4().hex
        from . import reference_examples
        artifact['example_archive'] = reference_examples.archive(entry['meta'], artifact)
        artifact['validation_archive'] = reference_examples.validation_archive(entry['meta'], artifact)
        origin = {'path': str(source), 'model_id': item['id']}
        meta = {k: v for k, v in entry['meta'].items() if k not in {'table_path', 'training_sources'}}
        meta.update(model_id=identifier, imported=True, imported_from=origin)
        artifact = {k: v for k, v in artifact.items() if k not in {'training_origin', '_file_stamp'}}
        artifact.update(revision=identifier, provenance={**artifact['provenance'], 'model_id': identifier})
        reference_models._save_model(identifier, artifact, meta)
    return identifier


def shared_path(folder, identifier):
    for item in _entries(folder):
        if item['id'] == identifier:
            return str(Path(item['path']).parent)
    return ''


def publish(root, identifier, folder=None):
    """Publish a new immutable version including examples, without private paths."""
    import joblib
    import os
    import tempfile
    from spacemap.sharing import shared_directory, share
    from . import reference_examples
    folder = Path(folder or training(root)['models_folder'])
    identifier, meta, artifact = reference_models._source(identifier)
    with project_lock(folder):
        existing = shared_path(folder, identifier)
        if existing and (Path(existing) / 'model.json').is_file():
            return existing
        target = model_library.new_entry(folder, 'phenotyping', root, {**meta, 'created': meta['trained']})
        shared_directory(target)
        artifact = {**artifact, 'example_archive': reference_examples.archive(meta, artifact),
                    'validation_archive': reference_examples.validation_archive(meta, artifact)}
        for key in ('training_origin', '_file_stamp'):
            artifact.pop(key, None)
        record = {k: v for k, v in meta.items() if k not in {'table_path', 'training_sources', 'imported_from'}}
        artifact['provenance'] = {k: v for k, v in artifact['provenance'].items() if k not in {'training_sources', 'imported_from'}}
        fd, temporary = tempfile.mkstemp(dir=target, prefix='.model-', suffix='.tmp')
        os.close(fd)
        try:
            joblib.dump(artifact, temporary, compress=3)
            os.replace(temporary, target / (identifier + '.joblib'))
        finally:
            Path(temporary).unlink(missing_ok=True)
        share(target / (identifier + '.joblib'))
        write_json(target / 'model.json', {'kind': 'phenotyping', 'id': identifier, 'name': meta.get('name', meta['method']),
            'kind_name': 'Phenotyping', 'source_project': Path(root).name, 'shared': meta['trained'],
            'details': {k: meta.get(k) for k in ('markers', 'examples', 'held_out_balanced_accuracy', 'adaptable')},
            'record': record, 'created': meta['trained'], 'files': [identifier + '.joblib']})
    return str(target)


def publish_training(root, models, parent, folder=None, automatic=False):
    # Automatic and manual training both publish immutable, dated versions.
    # The private copy remains usable if the destination is temporarily full
    # or read-only; callers must surface export_error rather than claim success.
    for meta in models:
        try:
            meta['shared_path'] = publish(root, meta['model_id'], folder)
        except OSError as exc:
            meta['export_error'] = 'Model saved locally. Could not copy to Models folder: ' + str(exc)
    # RF is the default new model; choosing the best on holdout would itself use
    # validation for model selection. Both versions remain in the dropdown.
    with project_lock(root):
        value = saved(root)
        current = value.get('model_id', '')
        if current == parent or (current and reference_models.current_model_id(current) == models[0]['model_id']):
            _path(root).parent.mkdir(parents=True, exist_ok=True)
            write_json(_path(root), {**value, 'model_id': models[0]['model_id']})
