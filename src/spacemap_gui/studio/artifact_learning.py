"""Reviewed artifact examples and versioned, reusable random forests.

One independent library per artifact. A training round snapshots reviewed
polygons, samples pixels deterministically, and refits on accumulated images.
Unlabelled tissue is negative only when the user explicitly chooses that rule.
Prediction reads bounded tiles and never writes tissue or artifact polygons.
"""
from collections import OrderedDict
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
import threading
import uuid

import numpy as np

from . import annotation, annotation_edits, artifacts

# Keep the on-disk debris key: existing bright/lint models and reviewed edits
# become the starting point for the broader image-quality workflow, unchanged.
NAMES = {"clot": "Blood clots", "blur": "Blur", "bright": "Bright spots", "lint": "Lint", "debris": "Image quality"}
CLASSES = {"clot": "Blood_Clot", "blur": "Detected_Blur", "bright": "Detected_Bright", "lint": "Detected_Lint", "debris": "Image_quality"}
QUALITY_DETECTORS = ('blur', 'bright', 'lint')
COLOURS = {part: '#e6194b' for part in NAMES}
SCALES = (1, 2, 4, 8, 16, 32)
TILE = 384
SAMPLES_PER_CLASS = 3000
MAX_TRAIN = 60000
_cache = OrderedDict()
_lock = threading.Lock()


def part_name(part):
    if part not in NAMES:
        raise ValueError("Choose an artifact type.")
    return part


def folder(root, part):
    return Path(root) / "outputs/studio/artifact_models" / part_name(part)


def polygons_path(output, sample, part):
    return Path(output) / annotation.INTERMEDIATE / f"{sample}_{part_name(part)}_reviewed.geojson"


def example_path(root, part, image):
    from .annotation_inputs import image_key
    return folder(root, part) / "drawings" / (image_key(image) + ".geojson")


def open_examples(root, part, image):
    """Restore previously retained drawings once; never resurrect deleted examples."""
    from .settings import project_lock
    target = example_path(root, part, image)
    with project_lock(root):
        if target.exists():
            return target
        key = sha256(str(Path(image).resolve()).encode()).hexdigest()
        old = _read(folder(root, part) / "examples.json", {}).get(key)
        if not old:
            # An empty, explicitly opened green collection must stay empty on
            # later visits; do not restore a future snapshot's predictions as
            # old manual red drawings merely because no green was ever drawn.
            annotation_edits.replace(target, [], annotation_edits.revision(target),
                                     'Opened good-tissue examples', kind='examples')
            return target
        drawings = _read(folder(root, part) / (old["id"] + ".json"), {}).get("annotations", {})
        features = []
        for role in ("positive", "negative"):
            for feature in drawings.get(role, []):
                features.append({**feature, "properties": {**(feature.get("properties") or {}),
                    "example_role": role, "classification": annotation_edits.EXAMPLE_CLASSES[role]}})
        annotation_edits.replace(target, features, annotation_edits.revision(target), "Restored training examples", kind="examples")
    return target


def combine_examples(root, part, image, output, *, preview_active=False):
    """Move legacy red drawings into the artifact collection, preserving history.

    Write the receiving layer first and deduplicate by id so an interrupted
    migration can be repeated safely. Green drawings keep their separate file.
    """
    from contextlib import ExitStack
    examples = example_path(root, part, image)
    target = polygons_path(output, annotation.sample_id(image), part)
    with ExitStack() as stack:
        for path in sorted((examples, target)):
            stack.enter_context(annotation_edits.locked(path))
        features, old = annotation_edits.read(examples, kind='examples')
        red = [f for f in features if f['properties']['example_role'] == 'positive']
        if not red:
            return False
        if preview_active:
            raise ValueError('Save the artifact preview first to combine your earlier red drawings with it.')
        saved, revision = annotation_edits.read(target, kind='artifacts')
        ids = {f['properties']['id'] for f in saved}
        moved = [{**f, 'properties': {**f['properties'], 'source': 'manual_artifact',
                  'classification': {'name': CLASSES[part], 'color': [230, 25, 75]}}}
                 for f in red if f['properties']['id'] not in ids]
        if moved:
            annotation_edits._write(target, [*saved, *moved], saved, revision,
                'Combined red training drawings with artifacts', edited=True, kind='artifacts')
        green = [f for f in features if f['properties']['example_role'] == 'negative']
        annotation_edits._write(examples, green, features, old,
            'Moved red drawings to artifact polygons', edited=True, kind='examples')
    return True


def _read(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def mark_good(root, part, image, output, identifier, expected, example_expected, *, preview=None):
    """Compatibility entry point for one or several red-to-green corrections."""
    return convert_examples(root, part, image, output,
        identifier if isinstance(identifier, list) else [identifier], 'negative', expected, example_expected, preview=preview)


def _chosen_colour(features):
    """The colour chosen in Layers for these polygons (the most common one), if any."""
    from collections import Counter

    found = Counter(f['properties'].get('colour') for f in features if isinstance(f['properties'].get('colour'), str))
    return found.most_common(1)[0][0] if found else None


def convert_examples(root, part, image, output, identifiers, role, expected, example_expected, *, preview=None):
    """Move a complete selection, preserving full geometry and source metadata."""
    from contextlib import ExitStack
    if role not in ('negative', 'positive'):
        raise ValueError('Choose Artifact or Good tissue.')
    if not isinstance(identifiers, list) or not identifiers or len(identifiers) > 100000 or any(not isinstance(i, str) for i in identifiers):
        raise ValueError('Select polygons to convert.')
    identifiers = set(identifiers)
    target = polygons_path(output, annotation.sample_id(image), part)
    examples = example_path(root, part, image)
    with ExitStack() as stack:
        for path in sorted((examples, target)):
            stack.enter_context(annotation_edits.locked(path))
        annotation_edits._check(target, expected)
        annotation_edits._check(examples, example_expected)
        saved, _ = annotation_edits.read(target, kind='artifacts')
        if preview is not None:
            if not isinstance(preview, list):
                raise ValueError('Send a list of artifact polygons.')
            # Resolve display-only outlines before prepare removes their source
            # references. Unselected preview polygons must retain detail too.
            versions = {expected: {f['properties']['id']: f for f in saved}}
            full = []
            for feature in preview:
                props = feature.get('properties') or {}
                if props.get('simplified'):
                    version = props.get('_studio_geometry', expected)
                    if version not in versions:
                        versions[version] = {f['properties']['id']: f for f in
                            _read(annotation_edits._state_path(target, version), [])}
                    exact = versions[version].get(props.get('id'))
                    if exact is None:
                        raise ValueError('Reload these polygons before changing their class.')
                    feature = {**feature, 'geometry': exact['geometry']}
                full.append(feature)
            features = annotation_edits.prepare(full, kind='artifacts')
        else:
            features = saved
        greens, green_revision = annotation_edits.read(examples, kind='examples')
        source, source_path, source_revision = (features, target, expected) if role == 'negative' else (greens, examples, green_revision)
        chosen = [f for f in source if f['properties']['id'] in identifiers]
        if len(chosen) != len(identifiers):
            raise ValueError('Some selected polygons changed. Select them again.')
        moved = []
        for selected in chosen:
            identifier = selected['properties']['id']
            if selected['properties'].get('simplified'):
                version = selected['properties'].get('_studio_geometry', source_revision)
                exact = next((f for f in _read(annotation_edits._state_path(source_path, version), [])
                              if f['properties']['id'] == identifier), None)
                if exact is None:
                    raise ValueError('Reload these polygons before changing their class.')
                selected = {**selected, 'geometry': exact['geometry']}
            # A colour chosen in Layers belongs to the side: a moved polygon takes the colour of the side it joins.
            props = {k: v for k, v in selected['properties'].items()
                     if k not in ('simplified', '_studio_geometry', 'name', 'named', 'label', 'colour')}
            joined = _chosen_colour(greens if role == 'negative' else features)
            if joined:
                props['colour'] = joined
            props.update(id=uuid.uuid4().hex, example_role=role,
                source='corrected_prediction' if role == 'negative' else 'manual_artifact',
                **({'corrected_artifact_id': identifier} if role == 'negative' else {'corrected_example_id': identifier}),
                classification=annotation_edits.EXAMPLE_CLASSES['negative'] if role == 'negative'
                    else {'name': CLASSES[part], 'color': [230, 25, 75]})
            moved.append({**selected, 'properties': props})
        red_after = [f for f in features if f['properties']['id'] not in identifiers] if role == 'negative' else [*features, *moved]
        green_after = [*greens, *moved] if role == 'negative' else [f for f in greens if f['properties']['id'] not in identifiers]
        # Validate the whole batch before either collection changes.
        red_after = annotation_edits.prepare(red_after, kind='artifacts')
        green_after = annotation_edits.prepare(green_after, kind='examples')
        label = f"{len(moved)} polygon(s) marked as " + ('good tissue' if role == 'negative' else 'artifact')
        def write_red():
            return {'features': red_after, 'revision': expected} if preview is not None else annotation_edits._write(
                target, red_after, saved, expected, label, edited=True, kind='artifacts')
        def write_green():
            return annotation_edits._write(examples, green_after, greens, green_revision, label, edited=True, kind='examples')
        # Write a persisted receiving layer first. Red previews stay in the
        # browser; source history retains the pre-conversion green collection.
        if role == 'negative':
            green = write_green(); red = write_red()
        else:
            red = write_red(); green = write_green()
        return {'artifact': red, 'examples': green, 'converted': len(moved), 'selected': [f['properties']['id'] for f in moved]}


def _write(path, value):
    annotation_edits._atomic(path, json.dumps(value, allow_nan=False).encode())


def options(root, part):
    from .settings import read_config
    classic = artifacts.values(read_config(root))
    defaults = {"scale": 2 if part in ('blur', 'debris') else 8,
                "remaining_good": part not in ('blur', 'debris'), "use_reviewed": False, "minimum_um2": 100.0, "fill_holes_um2": 0.0, "cut": 0.5,
                "include_current": True, "training_source": "selected"}
    if part == 'debris':
        defaults['detector_values'] = {p: classic[p]['value'] for p in QUALITY_DETECTORS}
        defaults['detectors'] = ['bright', 'lint']
    else:
        defaults['value'] = classic[part]['value']
    saved = _read(folder(root, part) / "settings.json", {})
    result = {**defaults, **saved}
    if part == 'debris':
        result['detector_values'] = {**defaults['detector_values'], **saved.get('detector_values', {})}
    return result


def save_options(root, part, values, expected):
    from .settings import project_lock, revision
    if not isinstance(values, dict) or set(values) - {"scale", "remaining_good", "use_reviewed", "minimum_um2", "fill_holes_um2", "cut", "value", "detector_values", "detectors", "include_current", "training_source"}:
        raise ValueError("Unknown artifact review setting.")
    if 'include_current' in values and type(values['include_current']) is not bool:
        raise ValueError('Choose whether to include current image edits.')
    if 'training_source' in values and values['training_source'] not in ('selected', 'library'):
        raise ValueError('Train from the selected model or the reviewed library.')
    if part == 'debris':
        _debris_settings(values.get('detector_values', options(root, part)['detector_values']),
                         values.get('detectors', options(root, part)['detectors']))
    elif set(values) & {'detector_values', 'detectors'}:
        raise ValueError('These detector settings belong to Image quality.')
    if "value" in values:
        artifacts.check({part: {"value": values["value"]}})
    if "scale" in values and (type(values["scale"]) is not int or values["scale"] not in SCALES):
        raise ValueError("Choose one of the detection scales.")
    if "remaining_good" in values and type(values["remaining_good"]) is not bool:
        raise ValueError("Choose whether remaining tissue is good tissue.")
    if "use_reviewed" in values and type(values["use_reviewed"]) is not bool:
        raise ValueError("Choose whether to include reviewed polygons.")
    for key, low, high in (("minimum_um2", 0, 1e8), ("fill_holes_um2", 0, 1e8), ("cut", .05, .95)):
        if key in values and (isinstance(values[key], bool) or not isinstance(values[key], (float, int)) or not low <= values[key] <= high):
            raise ValueError(f"{key} must be between {low} and {high}.")
    with project_lock(root):
        current = options(root, part)
        if revision(current) != expected:
            raise ValueError("Artifact settings changed in another window. Reopen this section.")
        _write(folder(root, part) / "settings.json", {**current, **values})
    return state(root, part)


def _model_reference(part, version):
    """Only the shared quality picker may refer to the retained Blur library."""
    if isinstance(version, str) and ':' in version:
        source, identifier = version.split(':', 1)
        if part != 'debris' or source != 'blur' or not identifier or ':' in identifier:
            raise ValueError('Choose a saved model version.')
        return source, identifier
    return part, version


def _versions(root, part):
    versions = _read(folder(root, part) / 'versions.json', [])
    if part == 'debris':
        versions = [*versions, *[{**v, 'id': 'blur:' + v['id'], 'source_part': 'blur'}
                     for v in _read(folder(root, 'blur') / 'versions.json', [])]]
    for version in versions:
        if "image_names" not in version:  # models made before their image names were kept: read them from the examples
            version["image_names"] = _image_names(root, version, version.get("source_part") or part)
    return sorted(versions, key=lambda v: v['created'])


_NAMES = {}


def _image_names(root, version, part):
    """The images a model was trained on, from its saved example records (once per model)."""
    key = (str(root), part, version.get("id"))
    if key not in _NAMES:
        names = set()
        for identifier in version.get("examples", []):
            source = version.get("example_sources", {}).get(identifier, part)
            try:
                with (folder(root, source) / (identifier + ".json")).open() as handle:
                    head = handle.read(4096)  # "image" is among the first keys; the drawings that follow can be large
                found = re.search(r'"image":\s*"([^"]+)"', head)
                if found:
                    names.add(found.group(1))
            except OSError:
                continue
        _NAMES[key] = sorted(names)
    return _NAMES[key]


def state(root, part):
    from .settings import revision
    home = folder(root, part)
    examples = _read(home / "examples.json", {})
    versions = _versions(root, part)
    active = [v for v in versions if not v.get('deleted_at')]
    selected = _read(home / 'selection.json', {}).get('version')
    if selected not in {v['id'] for v in active}:
        own = [v for v in active if v.get('source_part', part) == part]
        selected = (own or active)[-1]['id'] if active else ''
    values = options(root, part)
    return {"settings": values, "revision": revision(values), "versions": active,
            'deleted_versions': [v for v in versions if v.get('deleted_at')], 'selected_version': selected,
            "range": _slider_range(part) if part != 'debris' else None,
            "detector_ranges": {p: _slider_range(p) for p in QUALITY_DETECTORS} if part == 'debris' else {},
            "examples": [{k: value[k] for k in ("id", "image", "positive", "negative", "remaining_good", "created")}
                         for value in examples.values()], "scales": SCALES}


def manage_model(root, part, action, version):
    """Select, recoverably delete, or restore one immutable model version."""
    from .settings import project_lock
    home = folder(root, part)
    source, identifier = _model_reference(part, version)
    if source != part:
        # Preserve the original files and deletion/restore semantics. The shared
        # picker has its own selection; inspecting it does not rewrite Blur's.
        with project_lock(root):
            if action == 'select_model':
                record = next((v for v in _read(folder(root, source) / 'versions.json', []) if v['id'] == identifier), None)
                if record is None:
                    raise ValueError('This model is no longer available. Refresh the model list.')
                if record.get('deleted_at'):
                    raise ValueError('This model was deleted. Restore it before using it.')
            elif action in ('delete_model', 'restore_model'):
                manage_model(root, source, action, identifier)
            else:
                raise ValueError('Choose Select, Delete, or Restore model.')
            if action != 'delete_model':
                _write(home / 'selection.json', {'version': version})
        return state(root, part)
    with project_lock(root):
        versions = _read(home / 'versions.json', [])
        record = next((v for v in versions if v['id'] == version), None)
        if record is None:
            raise ValueError('This model is no longer available. Refresh the model list.')
        if action == 'restore_model':
            if not (home / (record['id'] + '.joblib')).is_file():
                raise ValueError('The saved model file is missing and cannot be restored.')
            record.pop('deleted_at', None)
            record['restored_at'] = datetime.now(timezone.utc).isoformat()
            _write(home / 'versions.json', versions)
        elif action == 'delete_model':
            if not record.get('deleted_at'):
                record['deleted_at'] = datetime.now(timezone.utc).isoformat()
                _write(home / 'versions.json', versions)
            active = [v for v in versions if not v.get('deleted_at')]
            chosen = _read(home / 'selection.json', {}).get('version')
            if chosen not in {v['id'] for v in active}:
                _write(home / 'selection.json', {'version': active[-1]['id'] if active else ''})
        elif action != 'select_model':
            raise ValueError('Choose Select, Delete, or Restore model.')
        if action != 'delete_model':
            if record.get('deleted_at'):
                raise ValueError('This model was deleted. Restore it before using it.')
            _write(home / 'selection.json', {'version': record['id']})
    return state(root, part)


def _slider_range(part):
    definition = artifacts.PARTS[part]
    return {'min': definition[3][0], 'max': definition[3][1], 'step': definition[4]}


def _debris_settings(values, detectors):
    if not isinstance(values, dict) or not {'bright', 'lint'} <= set(values) <= set(QUALITY_DETECTORS):
        raise ValueError('Set the Blur, Bright spots and Lint thresholds.')
    artifacts.check({p: {'value': value} for p, value in values.items()})
    if not isinstance(detectors, list) or not detectors or any(p not in values or p not in QUALITY_DETECTORS for p in detectors):
        raise ValueError('Choose Blur, Bright spots, Lint, or a combination for the preview.')


def _extractor(image, scale, schema='legacy', **kwargs):
    if schema == 'quality_v2':
        from .artifact_quality import extractor_v2
        return extractor_v2(image, scale)
    if schema == 'quality_v1':
        from .artifact_quality import extractor
        return extractor(image, scale)
    if schema != 'legacy':
        raise ValueError('This model uses an unsupported feature schema.')
    return Features(image, scale, **kwargs)


def _geometry(features, context="Annotation"):
    from shapely import get_coordinates
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    if not isinstance(features, list) or len(features) > 100000:
        raise ValueError(f"{context}: choose polygon annotations.")
    geometries = []
    for index, feature in enumerate(features):
        name = (feature.get("properties") or {}).get("name") or str(index + 1)
        try:
            geom = shape(feature["geometry"])
            if geom.geom_type not in ("Polygon", "MultiPolygon") or not np.isfinite(get_coordinates(geom)).all():
                raise ValueError("Not a finite polygon")
            if not geom.is_valid:
                geom = make_valid(geom)
            parts = annotation_edits.polygons(geom)
            if not parts and not geom.is_empty:
                raise ValueError("No polygon area")
            geometries.extend(parts)
        except Exception as exc:
            raise ValueError(f"{context} polygon ‘{name}’ has no usable area. Delete it or redraw its outline.") from exc
    return unary_union(geometries)


def _polygon_area(geometry):
    """Keep sampling area after clipping; point/line contacts carry no labels.

    GEOS may return a mixed GeometryCollection where examples just touch the
    tissue boundary. Overlaying that collection with an empty class can raise
    an assertion even though every input polygon is valid. Preserve all polygon
    area and holes without rounding or modifying the saved annotations.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry
    parts = annotation_edits.polygons(geometry)
    return unary_union(parts) if parts else Polygon()


def _mask(geometry, height, width, scale, x=0, y=0):
    from rasterio.features import rasterize
    from affine import Affine
    if geometry.is_empty:
        return np.zeros((height, width), bool)
    return rasterize([(geometry, 1)], out_shape=(height, width),
                     transform=Affine.translation(x * scale, y * scale) * Affine.scale(scale, scale),
                     dtype="uint8").astype(bool)


class Features:
    """Small windows at the requested scale, with physical-size texture windows."""
    def __init__(self, image, scale, *, markers=None):
        from spacemap.viewer.sources import open_source
        from spacemap.tissue_annotation_utils import split_he_and_codex_channels
        if type(scale) is not int or scale not in SCALES:
            raise ValueError("Choose one of the detection scales.")
        self.source = open_source(image)
        self.info = self.source.info
        self.scale = scale
        self.read_order = 'channel-major'
        self.h = int(np.ceil(self.info.height / scale))
        self.w = int(np.ceil(self.info.width / scale))
        self.he, self.markers = split_he_and_codex_channels(self.info.channel_names)
        self.he = list(self.he[:3])
        if markers is not None:
            self.markers = [i for i in self.markers if str(self.info.channel_names[i]).strip().casefold() in markers]
        self.mpp = self.info.pixel_size_um or .5083
        self.stats = {}
        # Read a small pyramid sample once, rather than every channel at full resolution.
        thumb_scale = max(scale, 2 ** int(np.ceil(np.log2(max(self.info.width, self.info.height) / 512))))
        for index in [*self.he, *self.markers]:
            sample = self.read(index, 0, 0, int(np.ceil(self.info.width / thumb_scale)),
                               int(np.ceil(self.info.height / thumb_scale)), scale=thumb_scale)
            values = sample[np.isfinite(sample) & (sample > 0)]
            self.stats[index] = max(float(np.percentile(values, 99.5)) if values.size else 1., 1.)
        self.names = (["he:r", "he:g", "he:b", "he:h", "he:e", "he:h_mean", "he:e_mean", "he:e_texture"] if len(self.he) == 3 else [])
        self.marker_names = [str(self.info.channel_names[i]).strip().casefold() for i in self.markers]
        if len(set(self.marker_names)) != len(self.marker_names):
            raise ValueError("Marker names repeat in this image. Give each channel a distinct name.")
        for name in self.marker_names:
            self.names.extend(["marker:" + name, "texture:" + name])
        if not self.names:
            raise ValueError("No H&E or named marker channels were found.")

    def read(self, channel, x0, y0, x1, y1, *, scale=None):
        import cv2
        scale = scale or self.scale
        levels = self.info.levels
        level = max((i for i, entry in enumerate(levels) if max(entry.downsample_x, entry.downsample_y) <= scale * 1.01), default=0)
        entry = levels[level]
        # In prediction each tile visits every marker. A panel larger than the
        # whole-plane cache would evict itself and decompress every plane again.
        from spacemap.viewer.sources import _WHOLE_CACHE_BYTES
        panel_bytes = entry.width * entry.height * np.dtype(self.info.dtype).itemsize * len([*self.he, *self.markers])
        cache_whole = self.read_order == 'channel-major' or panel_bytes <= _WHOLE_CACHE_BYTES
        raw = self.source.read_region(level, channel, int(x0 * scale / entry.downsample_x), int(y0 * scale / entry.downsample_y),
                                      int(np.ceil(x1 * scale / entry.downsample_x)), int(np.ceil(y1 * scale / entry.downsample_y)), cache_whole=cache_whole)
        if raw.shape != (y1 - y0, x1 - x0):
            raw = cv2.resize(raw.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        return np.nan_to_num(raw.astype(np.float32), nan=0, posinf=0, neginf=0)

    def tile(self, x, y, width, height):
        from scipy.ndimage import uniform_filter
        from skimage.color import rgb2hed
        window = max(3, int(round(6 / (self.mpp * self.scale))) | 1)
        halo = window // 2 + 1
        x0, y0 = max(0, x - halo), max(0, y - halo)
        x1, y1 = min(self.w, x + width + halo), min(self.h, y + height + halo)
        crop = (slice(y - y0, y - y0 + height), slice(x - x0, x - x0 + width))
        def mean(a): return uniform_filter(a, size=window, mode="reflect")
        def texture(a): return np.sqrt(np.maximum(0, mean(a * a) - mean(a) ** 2))
        fields = []
        if len(self.he) == 3:
            rgb = np.stack([np.clip(self.read(i, x0, y0, x1, y1) / self.stats[i], 0, 1) for i in self.he], axis=-1)
            stains = rgb2hed(np.clip(rgb, 1e-4, 1)).astype(np.float32)
            h, e = stains[..., 0], stains[..., 1]
            fields.extend([rgb[..., i] for i in range(3)] + [h, e, mean(h), mean(e), texture(e)])
        for i in self.markers:
            signal = np.arcsinh(5 * self.read(i, x0, y0, x1, y1) / self.stats[i])
            fields.extend([signal, texture(signal)])
        return np.stack([field[crop] for field in fields], axis=-1).astype(np.float32)


def snapshot(root, part, image, scale, positives, negatives, tissue, remaining_good, *, report=lambda *a: None, device="auto"):
    """Sample the explicitly reviewed labels; one current example set per image."""
    from shapely.geometry import box
    from .settings import project_lock
    import time
    started = time.perf_counter()
    report('Reading image metadata and channel normalization', 0)
    schema = 'quality_v2' if part in ('blur', 'debris') else 'legacy'
    extractor = _extractor(image, scale, schema)
    domain = _polygon_area(_geometry(tissue, "Work within").intersection(box(0, 0, extractor.info.width, extractor.info.height)))
    if part == "lint":
        domain = box(0, 0, extractor.info.width, extractor.info.height).difference(domain)
    positive = _polygon_area(_geometry(positives, "Red example").intersection(domain))
    negative = _polygon_area(_geometry(negatives, "Green example").intersection(domain))
    if remaining_good:
        negative = negative.union(domain.difference(positive))
    if not positive.is_empty and not negative.is_empty and positive.intersection(negative).area > 1e-6:
        raise ValueError("Red and green examples overlap. Erase the overlap with Alt, or delete one of those examples.")
    if positive.is_empty and negative.is_empty:
        raise ValueError("Draw an artifact or good-tissue example inside Work within, then Train.")
    rng = np.random.default_rng(42)
    # Sample coordinates from polygon bounding boxes without rasterising the full slide.
    from shapely import contains_xy
    points, labels = [], []
    for label, geom in ((1, positive), (0, negative)):
        if geom.is_empty:
            continue
        x0, y0, x1, y1 = geom.bounds
        collected = []
        for _ in range(100):
            xs = rng.integers(max(0, int(x0 / scale)), min(extractor.w, int(np.ceil(x1 / scale))), SAMPLES_PER_CLASS * 3)
            ys = rng.integers(max(0, int(y0 / scale)), min(extractor.h, int(np.ceil(y1 / scale))), SAMPLES_PER_CLASS * 3)
            keep = contains_xy(geom, (xs + .5) * scale, (ys + .5) * scale)
            collected.extend(zip(xs[keep], ys[keep]))
            if len(set(collected)) >= SAMPLES_PER_CLASS:
                break
        coords = np.asarray(list(dict.fromkeys(collected))[:SAMPLES_PER_CLASS], dtype=np.int32)
        if len(coords) < 20:
            raise ValueError("An example is too small at this scale. Draw a larger example or choose a finer scale.")
        points.extend(coords.tolist()); labels.extend([label] * len(coords))
    points, labels = np.asarray(points), np.asarray(labels, dtype=np.uint8)
    from .artifact_sampling import sample
    matrix, execution = sample(extractor, points, report=lambda message, progress: report(message, .05+.9*progress), device=device)
    execution['preparation_seconds'] = round(time.perf_counter()-started,3)
    report('Saving training examples', .97)
    identifier = uuid.uuid4().hex
    home = folder(root, part)
    home.mkdir(parents=True, exist_ok=True)
    # Immutable samples let a running training round keep its exact input snapshot.
    with (home / f"{identifier}.npz").open("wb") as handle:
        np.savez_compressed(handle, x=matrix, y=labels, xy=points)
    from .annotation_workflow import software
    record = {"id": identifier, "image": Path(image).name, "image_key": str(Path(image).resolve()),
              "names": extractor.names, "feature_schema": schema,
              "positive": int(labels.sum()), "negative": int((labels == 0).sum()),
              "scale": scale, "remaining_good": bool(remaining_good), "created": datetime.now(timezone.utc).isoformat(),
              "annotations": {"positive": positives, "negative": negatives, "tissue": tissue},
              "preprocessing": {"pixel_size_um": extractor.mpp, "scale": scale,
                  "normalization_p995": {str(k): v for k, v in extractor.stats.items()},
                  "channels": list(extractor.info.channel_names), "sample_xy_units": "detection-scale pixels", "random_seed": 42},
              "software": software(), "execution": execution}
    _write(home / f"{identifier}.json", record)
    with project_lock(root):
        entries = _read(home / "examples.json", {})
        entries[sha256(record["image_key"].encode()).hexdigest()] = {k: v for k, v in record.items() if k != "annotations"}
        _write(home / "examples.json", entries)
    report('Examples saved', 1)
    return {**state(root, part), 'snapshot_id': identifier}


def forget_image(root, part, image):
    """Remove this image from future training; retained versions keep their snapshots."""
    from .settings import project_lock
    home = folder(root, part)
    with project_lock(root):
        entries = _read(home / "examples.json", {})
        entries.pop(sha256(str(Path(image).resolve()).encode()).hexdigest(), None)
        _write(home / "examples.json", entries)
    return state(root, part)


def training_request(root, part, *, base_version=None, current_snapshot=None, allow_empty=False):
    """Refit a chosen version's exact submissions, optionally replacing one image.

    Never fall back to the current library if an archived submission is missing.
    The returned job owns its immutable snapshot list even if models are deleted
    or another image is reviewed while it waits to run.
    """
    home = folder(root, part)

    def snapshot_record(identifier, source=None):
        if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
            raise ValueError('The selected model has an invalid training snapshot.')
        source = source or part
        if source != part and not (part == 'debris' and source == 'blur'):
            raise ValueError('The selected model has an invalid training source.')
        origin = folder(root, source)
        value = _read(origin / (identifier + '.json'), {})
        if value.get('id') != identifier or not value.get('image_key') or not (origin / (identifier + '.npz')).is_file():
            raise ValueError('Training examples for this model are missing. Restore its snapshot files or choose another model.')
        return {**{k: v for k, v in value.items() if k != 'annotations'}, 'source_part': source}

    if base_version:
        source, identifier = _model_reference(part, base_version)
        versions = _read(folder(root, source) / 'versions.json', [])
        base = next((v for v in versions if v['id'] == identifier and not v.get('deleted_at')), None)
        if base is None:
            raise ValueError('The selected model is unavailable. Restore it or choose another model.')
        examples = [snapshot_record(identifier, base.get('example_sources', {}).get(identifier, source))
                    for identifier in base.get('examples', [])]
    else:
        examples = list(_read(home / 'examples.json', {}).values())
    if current_snapshot:
        current = snapshot_record(current_snapshot)
        examples = [e for e in examples if e['image_key'] != current['image_key']]
        examples.append(current)
    if not examples and not allow_empty:
        raise ValueError("Keep reviewed examples before training.")
    return {"root": str(root), "part": part_name(part), "examples": examples,
            'parent_version': base_version, 'current_snapshot': current_snapshot}


def batch_captures(root, part, images, output, scale, remaining_good):
    """Each ticked image's reviewed labels, frozen when Train is pressed: red is its saved artifact polygons (as fixed
    by hand) with any red drawings, green its good-tissue examples, all inside its Work within. An image with no saved
    artifact polygons and no examples was never reviewed, and is left out."""
    from shapely.geometry import mapping
    from . import annotation_inputs as inputs

    captures, skipped = [], []
    for image in images:
        name, sample = Path(image).name, annotation.sample_id(Path(image).name)
        saved = polygons_path(output, sample, part)
        reviewed, _ = annotation_edits.read(saved, kind="artifacts")
        drawings, _ = annotation_edits.read(example_path(root, part, image), kind="examples")
        green = [f for f in drawings if f["properties"].get("example_role") == "negative"]
        red = [f for f in drawings if f["properties"].get("example_role") == "positive"]
        if not saved.is_file() and not green and not red:
            skipped.append(f"{name}: no saved {NAMES[part]} polygons or examples (not reviewed)")
            continue
        tissue, _ = annotation_edits.read(inputs.scope_path(root, image, output, part), kind="tissue")
        if not tissue:
            skipped.append(f"{name}: no saved Tissue or Work within annotation")
            continue
        area = _geometry(reviewed, "Reviewed artifact") if reviewed else None
        if area is not None and green:
            area = area.difference(_geometry(green, "Green example"))  # green corrections override red
        positives = red + ([{"type": "Feature", "properties": {}, "geometry": mapping(g)} for g in annotation_edits.polygons(area)]
                           if area is not None else [])
        captures.append({"image": str(image), "scale": scale, "positives": positives, "negatives": green,
                         "tissue": tissue, "remaining_good": bool(remaining_good)})
    return captures, skipped


def _aligned(x, present, names):
    # Missing channel values are accompanied by explicit availability indicators.
    found = {name: i for i, name in enumerate(present)}
    values = np.zeros((len(x), len(names) * 2), dtype=np.float32)
    for i, name in enumerate(names):
        if name in found:
            values[:, i] = x[:, found[name]]
            values[:, i + len(names)] = 1
    return values


def _upgrade_examples(root, part, examples, report):
    """Image quality examples measured with the earlier per-marker set are measured again, from their saved drawings,
    with the cross-marker set (one model never mixes the two). An image no longer on disk is left out."""
    upgraded, dropped, done = [], [], 0
    old = [e for e in examples if e.get('feature_schema') != 'quality_v2']

    def possible(entry):
        record = _read(folder(root, entry.get('source_part', part)) / (entry['id'] + '.json'), {})
        image, drawn = record.get('image_key') or entry.get('image_key'), record.get('annotations') or {}
        return bool(drawn.get('tissue')) and bool(drawn.get('positive') or drawn.get('negative')) and bool(image) and Path(image).is_file()

    if old and len(old) == len(examples) and not any(possible(e) for e in old):
        return examples, []  # nothing can be measured again (images gone): train as before, with the earlier set
    for entry in examples:
        if entry.get('feature_schema') == 'quality_v2':
            upgraded.append(entry)
            continue
        record = _read(folder(root, entry.get('source_part', part)) / (entry['id'] + '.json'), {})
        drawn, image = record.get('annotations') or {}, record.get('image_key') or entry.get('image_key')
        if not possible(entry):
            dropped.append(str(entry.get('image')))
            continue
        done += 1
        report(f"{entry.get('image')}: measuring again across markers ({done} of {len(old)})", .25 * done / max(len(old), 1))
        try:
            made = snapshot(root, part, image, int(record.get('scale', entry.get('scale', 2))), drawn.get('positive', []), drawn.get('negative', []),
                            drawn.get('tissue', []), bool(record.get('remaining_good', entry.get('remaining_good', False))))
        except ValueError as error:
            dropped.append(f"{entry.get('image')} ({error})")
            continue
        fresh = _read(folder(root, part) / (made['snapshot_id'] + '.json'), {})
        fresh.pop('annotations', None)
        upgraded.append({**fresh, 'source_part': part})
    if not upgraded:
        return examples, []  # nothing could be measured again: train as before, with the earlier set
    return upgraded, dropped


def train(settings, report):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from .settings import project_lock
    root, part, examples = settings["root"], settings["part"], settings["examples"]
    home = folder(root, part)
    home.mkdir(parents=True, exist_ok=True)
    dropped = []
    if part in ('blur', 'debris'):
        examples, dropped = _upgrade_examples(root, part, examples, report)
        if not examples:
            raise ValueError("None of the training images could be measured again: " + ", ".join(dropped[:4]))
    names = sorted(set(name for entry in examples for name in entry["names"]))
    schemas = {e.get('feature_schema', 'legacy') for e in examples}
    schema = 'quality_v2' if 'quality_v2' in schemas else 'quality_v1' if 'quality_v1' in schemas else 'legacy'
    matrices, labels = [], []
    rng = np.random.default_rng(42)
    per_image = max(40, MAX_TRAIN // max(1, len(examples)))
    for i, entry in enumerate(examples):
        report(f"Reading reviewed image {i + 1}/{len(examples)}", .25 * i / len(examples))
        origin = folder(root, entry.get('source_part', part))
        with np.load(origin / (entry["id"] + ".npz"), allow_pickle=False) as sample:
            idx = np.concatenate([rng.choice(np.flatnonzero(sample["y"] == value),
                   min(per_image // 2, int((sample["y"] == value).sum())), replace=False) for value in (0, 1)])
            matrices.append(_aligned(sample["x"][idx], entry["names"], names)); labels.append(sample["y"][idx])
    x, y = np.concatenate(matrices), np.concatenate(labels)
    # Include missing-marker examples during training, never pretend a missing marker measured zero.
    augmented = x.copy()
    groups = sorted(set(n.split(":", 1)[1] for n in names if n.startswith("marker:")))
    for group in groups:
        rows = rng.random(len(x)) < .5
        for i, name in enumerate(names):
            if name in ("marker:" + group, "texture:" + group):
                augmented[rows, i] = 0; augmented[rows, i + len(names)] = 0
    x, y = np.concatenate([x, augmented]), np.tile(y, 2)
    report("Training the artifact model…", .3)
    artifact_only = not np.any(y == 0)
    if not np.any(y == 1):
        raise ValueError("Add at least one artifact example to the training library first.")
    if artifact_only:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=80, n_jobs=2, random_state=42).fit(x)
    else:
        model = RandomForestClassifier(n_estimators=0, warm_start=True, n_jobs=2, max_depth=18,
                                       min_samples_leaf=3, class_weight="balanced", random_state=42)
        for count in (20, 40, 60, 80):
            report(f"Training trees {count}/80", .3 + .5 * (count - 20) / 80)
            model.n_estimators = count; model.fit(x, y)
    # A separately trained H&E model is an honest fallback for absent marker panels.
    he_names = [name for name in names if name.startswith("he:")]
    base = None
    he_entries = [i for i, entry in enumerate(examples) if all(name in entry["names"] for name in he_names)] if he_names and schema == 'legacy' else []
    if he_entries:
        report("Training the H&E fallback…", .85)
        indices = [names.index(name) for name in he_names]
        bx = np.concatenate([matrices[i][:, indices] for i in he_entries])
        by = np.concatenate([labels[i] for i in he_entries])
        if artifact_only:
            base = IsolationForest(n_estimators=40, n_jobs=2, random_state=42).fit(bx)
        elif len(np.unique(by)) == 2:
            base = RandomForestClassifier(n_estimators=40, n_jobs=2, max_depth=18, min_samples_leaf=3,
                                          class_weight="balanced", random_state=42).fit(bx, by)
    created = datetime.now(timezone.utc)
    version = part + '_' + created.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    report("Saving the model version…", .95)
    payload = {"model": model, "he_model": base, "names": names, "he_names": he_names,
               "feature_schema": schema, "mode": "artifact_only" if artifact_only else "supervised"}
    if artifact_only:
        payload['reference_scores'] = np.sort(model.score_samples(x))
        payload['he_reference_scores'] = np.sort(base.score_samples(bx)) if base else None
    temporary = home / f".{version}.tmp"
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(home / f"{version}.joblib")
    record = {"id": version, "created": created.isoformat(), "images": len(examples),
              "name": NAMES[part] + '_' + created.strftime('%Y-%m-%d_%H:%M:%S.%f_UTC'),
              "samples": sum(len(v) for v in labels), "markers": groups, "he": bool(base),
              "examples": [entry["id"] for entry in examples], "image_names": sorted({entry["image"] for entry in examples})}
    record['example_sources'] = {entry['id']: entry.get('source_part', part) for entry in examples}
    if part == 'debris':
        record['artifact_targets'] = list(QUALITY_DETECTORS)
    from .annotation_workflow import software
    record.update(mode=payload['mode'], feature_schema=schema, feature_names=names, estimator_parameters=model.get_params(),
                  he_estimator_parameters=base.get_params() if base else None,
                  software=software(), sample_limit=MAX_TRAIN, sampling_seed=42,
                  missing_marker_augmentation=0.5)
    if schema == 'quality_v1':
        from .artifact_quality import WINDOWS_UM
        record['quality_parameters'] = {'windows_um': list(WINDOWS_UM), 'signal_floor': .05,
                                        'channel_pooling': ['min', 'max', 'mean'], 'tile_size': 256}
    if schema == 'quality_v2':
        from .artifact_quality import WINDOWS_UM
        record['quality_parameters'] = {'windows_um': list(WINDOWS_UM), 'signal_floor': .05, 'tile_size': 256,
                                        'channel_pooling': ['min', 'max', 'mean', 'z max', 'z second', 'z mean', 'z unusual share'],
                                        'per_marker_scaling': 'robust z within the image (median, 1.4826 MAD of signal pixels)'}
    if dropped:
        record['left_out_images'] = dropped
    record['training_scales'] = sorted({entry['scale'] for entry in examples})
    record['parent_version'] = settings.get('parent_version')
    record['current_snapshot'] = settings.get('current_snapshot')
    with project_lock(root):
        versions = _read(home / "versions.json", [])
        _write(home / "versions.json", [*versions, record])
        _write(home / 'selection.json', {'version': version})
    return {"artifact": NAMES[part], "version": record, "count": len(examples), "left_out": dropped}


def model(root, part, version):
    import joblib
    part, version = _model_reference(part, version)
    versions = _read(folder(root, part) / "versions.json", [])
    record = next((v for v in versions if v["id"] == version and not v.get('deleted_at')), None)
    if record is None:
        raise ValueError("Choose a saved model version.")
    path = folder(root, part) / (record["id"] + ".joblib")
    key = str(path.resolve())
    with _lock:
        if key not in _cache:
            _cache[key] = joblib.load(path)
            while len(_cache) > 3:
                _cache.popitem(last=False)
        _cache.move_to_end(key)
        return _cache[key], record


def predict(root, part, version, image, scale, tissue_features, cut=.5, minimum_um2=100, report=None, *, device="auto", fill_holes_um2=0.0):
    from shapely.geometry import box
    from spacemap.tissue_annotation_utils import mask_to_geojson_features
    saved, record = model(root, part, version)
    schema = saved.get('feature_schema', 'legacy')
    extractor = _extractor(image, scale, schema, markers=record["markers"])
    extractor.read_order = "tile-major"
    if hasattr(extractor, "enable_acceleration"):
        extractor.enable_acceleration(device)
    tissue = _geometry(tissue_features, "Work within")
    if tissue.is_empty:
        raise ValueError("Load or detect Tissue first.")
    if part == "lint":
        tissue = box(0, 0, extractor.info.width, extractor.info.height).difference(tissue)
    matched = sorted(set(extractor.marker_names) & set(record["markers"]))
    use_he = bool(saved["he_model"] and len(extractor.he) == 3 and len(matched) < max(1, len(record["markers"]) / 2))
    shared_quality = (schema == 'quality_v1' and any(n in saved['names'] for n in extractor.names if n.startswith('quality:'))) or \
        (schema == 'quality_v2' and any(n in saved['names'] for n in extractor.names if n.startswith('quality2:')))
    if not use_he and not matched and not shared_quality and not (len(extractor.he) == 3 and record["he"]):
        raise ValueError("This image has no measurements shared with the model. Add reviewed examples from this panel first.")
    gpu_forest = None
    if saved.get('mode') != 'artifact_only' and getattr(extractor, 'accelerator', None):
        from .artifact_forest import accelerator
        gpu_forest = accelerator(saved['he_model'] if use_he else saved['model'], extractor.accelerator.device)
    if hasattr(extractor, 'execution'):
        extractor.execution['forest_device'] = 'cuda' if gpu_forest else 'cpu'
    # Keep only a binary full-slide mask; feature matrices and probabilities stay tile-sized.
    mask = np.zeros((extractor.h, extractor.w), dtype=np.uint8)
    # Larger prediction blocks amortize TIFF reads and GPU launches. Training
    # keeps its original sampling grid; detection scale is unchanged.
    tile_size = 512 if schema in ('quality_v1', 'quality_v2') else TILE
    total = int(np.ceil(extractor.h / tile_size) * np.ceil(extractor.w / tile_size))
    count = 0
    for y in range(0, extractor.h, tile_size):
        for x in range(0, extractor.w, tile_size):
            count += 1
            if report: report(f"Predicting {count}/{total}", count / total * .9)
            height, width = min(tile_size, extractor.h - y), min(tile_size, extractor.w - x)
            if not tissue.intersects(box(x * scale, y * scale, (x + width) * scale, (y + height) * scale)):
                continue
            inside = _mask(tissue, height, width, scale, x, y)
            if not inside.any(): continue
            values = extractor.tile(x, y, width, height)[inside]
            if use_he:
                values = values[:, [extractor.names.index(n) for n in saved["he_names"]]]
                forest = saved["he_model"]
            else:
                values = _aligned(values, extractor.names, saved["names"]); forest = saved["model"]
            if saved.get('mode') == 'artifact_only':
                reference = saved['he_reference_scores' if use_he else 'reference_scores']
                # Similarity rank within known artifacts, not a class probability.
                scores = np.searchsorted(reference, forest.score_samples(values), side='right') / len(reference)
                result = scores >= cut * .2
            elif gpu_forest:
                try:
                    result = gpu_forest.predict(values, cut)
                except (RuntimeError, MemoryError) as exc:
                    import logging
                    logging.warning('Artifact forest GPU unavailable; continuing on CPU: %s',exc)
                    gpu_forest=None
                    extractor.execution.update(forest_device='cpu',forest_fallback=str(exc))
                    result = forest.predict_proba(values)[:, list(forest.classes_).index(1)] >= cut
            else:
                result = forest.predict_proba(values)[:, list(forest.classes_).index(1)] >= cut
            mask[y:y + height, x:x + width][inside] = result
    features = mask_to_geojson_features(mask, scale, CLASSES[part], color_rgb=int(COLOURS[part][1:], 16), simplify_px=0, merge=False,
                                       extra_properties={"artifact": part, "source": "reviewed_model", "model_version": record['id'],
                                                         "model_part": _model_reference(part, version)[0]})
    features = _tidy(annotation_edits.prepare(features, kind="artifacts"), float(fill_holes_um2) / extractor.mpp ** 2, 0.5 * scale)
    features = [f for f in features if _geometry([f]).area * extractor.mpp ** 2 >= minimum_um2]
    return {"features": annotation_edits.prepare(features, kind="artifacts"), "matched_markers": matched,
            "missing_markers": sorted(set(record["markers"]) - set(matched)),
            "used": "Cross-marker quality model" if schema == 'quality_v2' else "All-channel texture model" if schema == 'quality_v1' else "H&E fallback" if use_he else "H&E and shared markers" if len(extractor.he) == 3 else "Shared markers",
            "feature_schema": schema, "read_markers": extractor.marker_names,
            "execution": getattr(extractor, "execution", {"device": "cpu"}),
            "scale": scale, "version": version, "mode": saved.get("mode", "supervised")}


def _tidy(features, smallest_px2, step_px):
    """Found polygons as one would draw them: the pixel staircase smoothed within half a detection pixel, and (Fill holes)
    holes smaller than the area given filled; thousands of tiny gaps made files huge and every edit slow."""
    from shapely.geometry import Polygon, mapping, shape
    from shapely.validation import make_valid

    out = []
    for feature in features:
        geometry = shape(feature["geometry"])
        if geometry.geom_type == "Polygon" and geometry.interiors:
            geometry = Polygon(geometry.exterior, [ring for ring in geometry.interiors if Polygon(ring).area >= smallest_px2])
        geometry = geometry.simplify(step_px, preserve_topology=True)
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        if not geometry.is_empty:
            out.append({**feature, "geometry": mapping(geometry)})
    return out


def detect(image, config, part, scale, value, tissue_features, minimum_um2=100, *, scope_path=None, detector_values=None, detectors=None, fill_holes_um2=0.0):
    """Classic detector with an explicit scale; full polygons replace preview rings."""
    from spacemap.tissue_annotation_utils import detect_blood_clot, mask_to_geojson_features
    from spacemap.viewer.sources import fingerprint
    if part == 'debris':
        _debris_settings(detector_values, detectors)
    else:
        artifacts.check({part: {"value": value}})
    domain = _geometry(tissue_features, "Work within")
    if domain.is_empty:
        raise ValueError("Load or detect Tissue first.")
    if part == "clot":
        # Blood-clot thresholding needs H&E only; CODEX is optional for learning.
        settings = annotation.settings(config)
        key = ("clot-input", str(image), str(fingerprint(image)), scale, sha256(domain.wkb).hexdigest(),
               settings.get("he_white_percentile", 99.5), settings.get("he_valid_min_intensity", 0))
        with _lock:
            prepared = _cache.get(key)
        if prepared is None:
            extractor = Features(image, scale, markers=[])
            if len(extractor.he) != 3:
                raise ValueError("The pinkness detector needs the three H&E channels. Use a trained marker model for this image.")
            # Explicit failure is preferable to silently analysing a coarser scale.
            if extractor.h * extractor.w > 90_000_000:
                raise ValueError("This scale is too large for an interactive preview. Choose 1/2 or a coarser scale.")
            channels = [extractor.read(i, 0, 0, extractor.w, extractor.h) for i in extractor.he]
            # Match classic H&E normalisation at the explicitly selected scale.
            normalized = []
            for channel in channels:
                sample = channel[::4, ::4]
                positive = sample[np.isfinite(sample) & (sample > settings.get("he_valid_min_intensity", 0))]
                white = max(float(np.percentile(positive, settings.get("he_white_percentile", 99.5))) if positive.size else 1., 1.)
                normalized.append(np.clip(channel / white, 0, 1))
            rgb = np.stack(normalized, axis=-1)
            prepared = (rgb, _mask(domain, extractor.h, extractor.w, scale), extractor.mpp)
            # Do not retain full-resolution, multi-gigabyte arrays between requests.
            if rgb.nbytes < 160_000_000:
                with _lock:
                    _cache[key] = prepared
                    while len(_cache) > 3: _cache.popitem(last=False)
        rgb, tissue, mpp = prepared
        found = detect_blood_clot(rgb, tissue, {**annotation.settings(config), "clot_eosin_ratio": value,
                                  "clot_min_area_mm2": minimum_um2 / 1e6}, mpp, scale)
        mask = found["clot_mask"]
    else:
        # Other artifact detectors retain their own measured scores and settings.
        prepared = artifacts.prepare(image, config, boundary=scope_path, scale=scale)
        if part == 'debris':
            inside = _mask(domain, *prepared['shape'], scale)
            mask = np.zeros(prepared['shape'], dtype=bool)
            if 'blur' in detectors:
                mask |= artifacts.blur_mask(prepared, detector_values['blur'])
            if 'bright' in detectors:
                mask |= artifacts.bright_mask(prepared, detector_values['bright'])
            if 'lint' in detectors:
                mask |= artifacts.lint_mask(prepared, detector_values['lint'], domain=inside)
        else:
            mask = {"blur": artifacts.blur_mask, "bright": artifacts.bright_mask, "lint": artifacts.lint_mask}[part](prepared, value)
        if part != "lint": mask &= _mask(domain, *mask.shape, scale)
        mpp = prepared["inputs"]["microns_per_pixel"]
    features = mask_to_geojson_features(mask, scale, CLASSES[part], color_rgb=int(COLOURS[part][1:], 16), simplify_px=0, merge=False,
                                       extra_properties={"artifact": part, "source": "reviewed_detector"})
    features = _tidy(annotation_edits.prepare(features, kind="artifacts"), float(fill_holes_um2) / mpp ** 2, 0.5 * scale)
    features = [f for f in features if _geometry([f]).area * mpp ** 2 >= minimum_um2]
    return {"features": annotation_edits.prepare(features, kind="artifacts"), "scale": scale,
            "used": "Pinkness detector" if part == "clot" else "Classic detector"}
