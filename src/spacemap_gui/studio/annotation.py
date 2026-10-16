"""Studio's Annotation step, part 1: Tab 4's tissue detection.

It reads the settings and folders classic Tab 4 saves and writes the same
files its step 1 writes: ``<output>/<sample>_valid_tissue.geojson`` (the
annotation the later steps build on) and
``<output>/intermediate/<sample>_tissue_boundary.geojson``, plus
``..._blood_clot.geojson`` when clot detection is on. The paths mirror
``spacemap_gui.tabs.tab4_annotation.common``, which needs a Streamlit
session; tests pin the names.
"""

from pathlib import Path
from functools import lru_cache
import math

IMAGE_EXTENSIONS = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".qptiff")
INTERMEDIATE = "intermediate"
# The tissue settings Studio shows; the rest keep Tab 4's saved values.
EDITABLE = {
    "tissue_mask_mode",
    "tissue_threshold",
    "tissue_morph_radius",
    "min_tissue_island_px",
    "min_tissue_area_um2",
    "he_od_threshold_factor",
    "detect_blood_clot",
    "blur_downsample",
    "fill_holes_um2",
}
DETAIL_CHOICES = (1, 2, 4, 8, 16)  # pyramid steps: 1 is full resolution
FOLDER_KEYS = {"images": ("annot_ta_input_dir",), "output": ("annot_ta_output_dir",)}
LABEL_KEY = "annot_studio_tissue_label"  # the name the tissue pieces get in Studio and its export
DRAWN_CLASS = "Drawn_Artifact"


def _path(root, value):
    return Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()


def _saved(config):
    return (config.get("tab_configurations") or {}).get("tab4_annotation") or {}


def main_output(root, mappings):
    """The Annotation output folder as Tab 4 sets it; the export is written here."""
    output = mappings.get("annot_ta_output_dir")
    return _path(root, output) if output else Path(root) / "outputs/processed_annotations/stage0_tissue_annotation"


def folders(root, mappings):
    """(images, output): output is where the step's files are kept, the Sub-directory of the Annotation output when it
    is on (tissue, artifacts, pieces, classes, regions and their History), else the output folder itself."""
    images = mappings.get("annot_ta_input_dir") or mappings.get("tab3_output_dir")
    main = main_output(root, mappings)
    return (_path(root, images) if images else Path(root) / "outputs/rois", working_folder(root, main))


# ---------------------------------------------------------------- the Sub-directory for the step's own files
SUBFOLDER_KEY = "annot_studio_subfolder"
SUBFOLDER_DEFAULT = "./working_files"
_SUBFOLDER = {}


def _subfolder_setting(root):
    """The saved Sub-directory choice ({"on", "path"}), or None when never chosen; read once per settings change."""
    from .settings import config_path

    path = config_path(root)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    key = str(path)
    if _SUBFOLDER.get(key, (None,))[0] != stamp:
        from .settings import read_config

        saved = _saved(read_config(root)).get(SUBFOLDER_KEY)
        _SUBFOLDER[key] = (stamp, saved if isinstance(saved, dict) else None)
    return _SUBFOLDER[key][1]


def check_subfolder(value):
    import re

    text = str(value or "").strip()
    if not re.fullmatch(r"\./[A-Za-z0-9_][A-Za-z0-9_.-]{0,59}(/[A-Za-z0-9_][A-Za-z0-9_.-]{0,59}){0,3}", text):
        raise ValueError("Type the sub-directory as ./name (letters, numbers, _ . -), for example ./working_files.")
    return text


def legacy_files(main):
    """The step's files already in the output folder itself (made before the Sub-directory): what Move takes along."""
    main = Path(main)
    if not main.is_dir():
        return []
    import json

    try:  # files Export wrote here stay (studio_export/named.json lists them)
        exports = set(json.loads((main / "studio_export" / "named.json").read_text()))
    except (OSError, ValueError, TypeError):
        exports = set()
    found = [p for p in main.iterdir() if p.is_file() and p.name.endswith((".geojson", ".json")) and not p.name.startswith(".")
             and p.name not in exports]
    found += [main / name for name in (INTERMEDIATE, "studio_history") if (main / name).is_dir()]
    return sorted(found)


def subfolder(root, main):
    """{"on", "path", "pending"}: on by default for a new output folder; an output folder that already holds the step's
    files keeps them where they are until the user moves them (pending counts them)."""
    saved = _subfolder_setting(root)
    if saved is not None:
        return {"on": bool(saved.get("on")), "path": saved.get("path") or SUBFOLDER_DEFAULT, "pending": 0}
    if (Path(main) / SUBFOLDER_DEFAULT[2:]).is_dir():  # already in use: it stays on
        return {"on": True, "path": SUBFOLDER_DEFAULT, "pending": 0}
    pending = len(legacy_files(main))
    return {"on": not pending, "path": SUBFOLDER_DEFAULT, "pending": pending}


def working_folder(root, main):
    chosen = subfolder(root, main)
    return Path(main) / chosen["path"][2:] if chosen["on"] else Path(main)


def save_subfolder(root, mappings, on, path, move=False):
    """Turn the Sub-directory on or off (and name it). With *move*, the step's files in the output folder go into it,
    and every saved reference to them (layer index, Work within, classification runs, hidden files) follows."""
    import json
    import shutil

    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    path = check_subfolder(path)
    if not isinstance(on, bool):
        raise ValueError("Sub-directory is on or off.")
    main = main_output(root, mappings)
    moved = []
    with project_lock(root):
        if move and on:
            target = main / path[2:]
            target.mkdir(parents=True, exist_ok=True)
            pairs = []  # (old, new, is a folder)
            for item in legacy_files(main):
                if item.resolve() == target.resolve() or target.resolve() in item.resolve().parents:
                    continue
                destination = target / item.name
                if item.is_dir():  # intermediate/, studio_history/: file by file, merged with what is there
                    for inner in sorted(item.rglob("*")):
                        goal = destination / inner.relative_to(item)
                        if inner.is_file() and not goal.exists():
                            goal.parent.mkdir(parents=True, exist_ok=True)
                            inner.replace(goal)
                    if not any(f.is_file() for f in item.rglob("*")):
                        shutil.rmtree(item, ignore_errors=True)
                    pairs.append((str(item.resolve()), str(destination.resolve()), True))
                    continue
                if destination.exists():
                    raise ValueError(f"{destination} already exists: move or rename it first.")
                old = str(item.resolve())
                item.replace(destination)
                pairs.append((old, str(destination.resolve()), False))
            moved = [new for _, new, _ in pairs]
            _repoint(root, pairs)
        config = read_config(root)
        persistence.update_stored_config_key(config, SUBFOLDER_KEY, {"on": on, "path": path})
        persistence.save_project_config(str(root), config)
    _SUBFOLDER.clear()
    return {"moved": moved, **subfolder(root, main)}


def _repoint(root, pairs):
    """Saved references to moved files point to their new place: every small JSON record under outputs/studio
    (layer index, Work within choices, classification runs, hidden files...)."""
    import json

    from .annotation_edits import _atomic

    if not pairs:
        return
    swaps = []
    for old, new, folder in pairs:
        quoted_old, quoted_new = json.dumps(old)[1:-1], json.dumps(new)[1:-1]
        if folder:
            swaps.append(('"' + quoted_old + "/", '"' + quoted_new + "/"))  # files inside a moved folder
        else:
            swaps.append(('"' + quoted_old + '"', '"' + quoted_new + '"'))
    base = Path(root) / "outputs" / "studio"
    for record in base.rglob("*.json") if base.is_dir() else []:
        try:
            if record.stat().st_size > 5_000_000:
                continue
            text = record.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        changed = text
        for old, new in swaps:
            changed = changed.replace(old, new)
        if changed != text:
            _atomic(record, changed.encode())


def zones_folder(root, mappings):
    """Where Tab 4's step 4 writes the pooled classes and zones (its own folder, not step 1-3's)."""
    folder = mappings.get("annot_ta_pool_output_dir")
    return _path(root, folder) if folder else Path(root) / "outputs/processed_annotations/stage1b_compartments_zones"


def sample_id(filename):
    """Tab 4's sample name: the file name without its image extension."""
    name = Path(filename).name
    for extension in IMAGE_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[: -len(extension)]
    return Path(name).stem


def valid_tissue(output, sample):
    return Path(output) / f"{sample}_valid_tissue.geojson"


def cleaned_tissue(output, sample):
    return Path(output) / f"{sample}_cleaned_tissue.geojson"


def analysis_tissue(output, sample):
    """The working Tissue file is the input to each following step."""
    return valid_tissue(output, sample)


def unify_tissue(output, sample, *, label='tissue'):
    """Bring a prior Clean Tissue result into the single working file once.

    Both old geometries remain restorable in Tissue History, and the legacy
    file stays on disk. A stale clean result is archived but never replaces
    newer tissue edits. The marker prevents Undo from triggering migration again.
    """
    import json
    from contextlib import ExitStack
    from datetime import datetime, timezone
    import uuid
    from . import annotation_edits as edits
    target, clean = valid_tissue(output, sample), cleaned_tissue(output, sample)
    if not clean.is_file(): return
    marker = edits._home(target) / 'single-tissue.json'
    with ExitStack() as stack:
        for path in sorted((target, clean)): stack.enter_context(edits.locked(path))
        if marker.is_file(): return
        before, expected = edits.read(target, kind='tissue', label=label)
        earlier, version = edits.read(clean, kind='tissue', label=label)
        meta = json.loads(clean.read_text()).get('studio', {})
        if meta.get('source_revision') == expected:
            edits._write(target, earlier, before, expected, 'Continued from saved Clean Tissue',
                         kind='tissue', label=label, metadata={'legacy_cleaned_file':str(clean.resolve())})
        else:
            edits._archive(target, expected, before)
            edits._archive(target, version, earlier, meta)
            now = datetime.now(timezone.utc)
            event = {'id':now.strftime('%Y%m%dT%H%M%S%f')+'-'+uuid.uuid4().hex[:8],
                     'before':expected, 'after':version, 'label':'Earlier Clean Tissue (archived)', 'when':now.isoformat()}
            edits._atomic(edits._home(target)/'events'/(event['id']+'.json'), json.dumps(event).encode())
        edits._atomic(marker, json.dumps({'legacy_revision':version}).encode())


def boundary(output, sample):
    return Path(output) / INTERMEDIATE / f"{sample}_tissue_boundary.geojson"


def clot(output, sample):
    return Path(output) / INTERMEDIATE / f"{sample}_blood_clot.geojson"


def zones(output, sample):
    return Path(output) / f"{sample}_zones.geojson"


def domains(output, sample):
    return Path(output) / f"{sample}_macro_domains.geojson"


def classes(output, sample):
    return Path(output) / f"{sample}_raw_macro_clusters.geojson"


def drawn_artifacts(output, sample):
    """Artifacts drawn by hand on the image; they are taken out of the tissue like the detected ones."""
    return Path(output) / INTERMEDIATE / f"{sample}_drawn_artifacts.geojson"


def detected_artifacts(output, sample):
    return Path(output) / INTERMEDIATE / f"{sample}_detected_artifacts.geojson"


def marker_regions(output, sample):
    return Path(output) / f"{sample}_marker_regions.geojson"


def exported(output, sample):
    return Path(output) / "studio_export" / f"{sample}_annotations.geojson"


def tissue_label(config):
    return str(_saved(config).get(LABEL_KEY) or "tissue")


def all_folders(root, mappings):
    images, output = folders(root, mappings)
    return {"images": images, "output": output, "main": main_output(root, mappings), "zones": zones_folder(root, mappings)}


def _has_features(path):
    import json

    try:
        return bool(json.loads(Path(path).read_text()).get("features")) if Path(path).is_file() else False
    except (OSError, ValueError, AttributeError):
        return False


STAGES = ("tissue", "clean", "pieces", "classes", "regions", "zones", "export")


@lru_cache(maxsize=512)
def _file_status(path, modified, size):
    import json
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            return False, False
        manual = any((f.get("properties") or {}).get("source") != "auto_artifact" for f in data.get("features", []))
        return bool((data.get("studio") or {}).get("edited")), manual
    except (ValueError, OSError):
        return False, False


def file_status(path):
    try:
        stat = Path(path).stat()
        return _file_status(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False, False


def _classified(root, image):
    """Whether Classification has a saved run for the image (classes to review), saved as polygons or not."""
    from . import classification

    last = classification.latest(root, image)
    return bool(last) and (classification.directory(root, image) / (last["run_id"] + ".npz")).is_file()


def images_plan(root, mappings, config):
    """Every image in Tab 4's image folder with the stages already done for it (their files are there)."""
    from . import pieces, artifact_learning

    places = all_folders(root, mappings)
    names = sorted(p.name for p in places["images"].iterdir() if p.is_file() and p.name.lower().endswith(IMAGE_EXTENSIONS)) \
        if places["images"].is_dir() else []
    output, rows = places["output"], []
    for name in names:
        sample = sample_id(name)
        done = {
            "tissue": valid_tissue(output, sample).is_file() or boundary(output, sample).is_file(),
            "clean": detected_artifacts(output, sample).is_file() or drawn_artifacts(output, sample).is_file() or any(artifact_learning.polygons_path(output, sample, part).is_file() for part in artifact_learning.NAMES),
            "pieces": pieces.review(output, sample).is_file(),
            "classes": domains(output, sample).is_file() or classes(output, sample).is_file() or any(output.glob(sample+'_classification_*.geojson')),
            "regions": _has_features(marker_regions(output, sample)) or any(_has_features(p) for p in output.glob(sample+'_marker_regions_*.geojson')),
            "zones": zones(places["zones"], sample).is_file(),
            "export": exported(places["main"], sample).is_file(),
        }
        edited = any(file_status(path)[0] for path in (valid_tissue(output, sample), drawn_artifacts(output, sample), marker_regions(output, sample), cleaned_tissue(output, sample)))
        from .annotation_inputs import imported
        rows.append({"image": name, "sample": sample, "done": done, "classified": _classified(root, places["images"] / name),
                     "imported": [p.stem.split("_", 1)[-1] for p in imported(root, places["images"] / name)],
                     "drawn": file_status(drawn_artifacts(output, sample))[1], "edited": edited})
    return {"folders": {"images": str(places["images"]), "output": str(places["main"])},
            "subfolder": subfolder(root, places["main"]), "working": str(output), "rows": rows,
            "label": tissue_label(config), "detail": DETAIL_CHOICES,
            "microns_per_pixel": float(settings(config).get("microns_per_pixel", 0.5083))}


def save_step(root, changes):
    """Save the step's folders (full paths) and the tissue label."""
    import re

    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if isinstance(changes, dict) and set(changes) == {"subfolder"}:
        value = changes["subfolder"] if isinstance(changes["subfolder"], dict) else {}
        from .settings import read_config

        return save_subfolder(root, read_config(root).get("directory_mappings") or {}, value.get("on"), value.get("path"),
                              move=bool(value.get("move")))
    if not isinstance(changes, dict) or set(changes) - {"folders", "label"}:
        raise ValueError("Only the Annotation step's folders and tissue label can be changed here.")
    base = Path(root).resolve()
    resolved = {}
    for kind, value in (changes.get("folders") or {}).items():
        if kind not in FOLDER_KEYS or not isinstance(value, str) or not value.strip() or len(value) > 1024:
            raise ValueError("Type the folder's path.")
        target = Path(value.strip()).expanduser()
        target = (target if target.is_absolute() else base / target).resolve()
        if kind == "images" and not target.is_dir():
            raise ValueError(f"{target} is not a folder.")
        if not target.parent.is_dir():
            raise ValueError(f"{target.parent} is not a folder.")
        resolved[kind] = "${PROJECT_ROOT}/" + target.relative_to(base).as_posix() if target == base or base in target.parents else str(target)
    label = changes.get("label")
    if label is not None and (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]{0,39}", label.strip())):
        raise ValueError("The tissue label starts with a letter: letters, numbers, spaces, _ and - (up to 40).")
    with project_lock(root):
        config = read_config(root)
        for kind, value in resolved.items():
            for key in FOLDER_KEYS[kind]:
                persistence.update_stored_config_key(config, key, value)
        if label is not None:
            persistence.update_stored_config_key(config, LABEL_KEY, label.strip())
        persistence.save_project_config(str(root), config)


def subtract_drawn(output, sample):
    """Take the artifacts drawn by hand out of the valid tissue file (its shapes and classes otherwise kept)."""
    import json

    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    drawn_path, tissue_path = drawn_artifacts(output, sample), valid_tissue(output, sample)
    if not _has_features(drawn_path) or not tissue_path.is_file():
        return 0
    drawn = unary_union([shape(f["geometry"]).buffer(0) for f in json.loads(drawn_path.read_text())["features"] if f.get("geometry")])
    data = json.loads(tissue_path.read_text())
    features = data.get("features", []) if isinstance(data, dict) else data
    kept = []
    for feature in features:
        left = shape(feature["geometry"]).buffer(0).difference(drawn)
        if not left.is_empty and left.area > 0:
            kept.append({**feature, "geometry": mapping(left)})
    tissue_path.write_text(json.dumps({"type": "FeatureCollection", "features": kept} if isinstance(data, dict) else kept))
    return len(features)


def settings(config):
    from spacemap.tissue_annotation_utils import DEFAULT_TISSUE_CONFIG

    saved = _saved(config).get("annot_ta_tissue_config") or {}
    values = {**DEFAULT_TISSUE_CONFIG, "fill_holes_um2": 0.0, **(saved if isinstance(saved, dict) else {})}
    # Older projects stored detection-level pixels. Convert that physical area
    # once in the form; subsequent scale changes keep the area in µm² fixed.
    values.setdefault("min_tissue_area_um2", round(values["min_tissue_island_px"] *
                      (values["microns_per_pixel"] * values["blur_downsample"]) ** 2, 6))
    return values


def check(changes):
    from spacemap.tissue_annotation_utils import TISSUE_MASK_MODES

    if not isinstance(changes, dict) or set(changes) - EDITABLE:
        raise ValueError("Only the listed tissue settings can be changed here.")
    for key, value in changes.items():
        if key == "tissue_mask_mode":
            if value not in TISSUE_MASK_MODES:
                raise ValueError("Choose one of Tab 4's tissue modes.")
        elif key == "blur_downsample":
            if value not in DETAIL_CHOICES:
                raise ValueError("Choose one of the image's detail levels: 1, 2, 4, 8 or 16.")
        elif key == "detect_blood_clot":
            if not isinstance(value, bool):
                raise ValueError("Choose on or off for blood clots.")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 1e7:
            raise ValueError(f"Enter a positive number for {key.replace('_', ' ')}.")
    return changes


def save(root, changes, expected):
    """Save tissue settings into Tab 4's own saved config."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    check(changes)
    with project_lock(root):
        config = read_config(root)
        current = settings(config)
        if expected != revision(current):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        stored = dict(_saved(config).get("annot_ta_tissue_config") or {})
        stored.update(changes)
        if {"min_tissue_area_um2", "blur_downsample"} & changes.keys():
            combined = {**current, **changes}
            stored["min_tissue_area_um2"] = combined["min_tissue_area_um2"]
            # Keep the classic setting in its original unit for interchange.
            stored["min_tissue_island_px"] = int(math.floor(combined["min_tissue_area_um2"] /
                (combined["microns_per_pixel"] * combined["blur_downsample"]) ** 2))
        persistence.update_stored_config_key(config, "annot_ta_tissue_config", stored)
        persistence.save_project_config(str(root), config)
        return settings(read_config(root))


def channels(names, config):
    """(H&E indices, marker indices): Tab 4's saved choice, else detected by name."""
    from spacemap.tissue_annotation_utils import split_he_and_codex_channels

    detected_he, detected_codex = split_he_and_codex_channels(names)
    saved = _saved(config)
    he = [names.index(name) for name in saved.get("annot_ta_he_channels") or [] if name in names]
    markers = [names.index(name) for name in saved.get("annot_ta_tissue_codex_markers") or [] if name in names]
    return he or list(detected_he), markers or list(detected_codex)


def fill_small_holes(features, threshold_um2, pixel_size_um):
    """Fill enclosed polygon holes strictly below the physical-area threshold.

    Works on full-resolution outlines, so changing detection scale does not
    change the size unit. Zero returns the original result unchanged.
    """
    from shapely.geometry import Polygon, mapping, shape
    from shapely.ops import unary_union
    from .annotation_edits import polygons

    if not math.isfinite(threshold_um2) or threshold_um2 < 0:
        raise ValueError("Enter 0 to keep holes, or a positive hole size in µm².")
    if not threshold_um2 or not features:
        return features, 0
    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("A valid pixel size is needed to measure holes in µm².")
    limit = threshold_um2 / (pixel_size_um * pixel_size_um)
    filled, count = [], 0
    for feature in features:
        for polygon in polygons(shape(feature["geometry"])):
            keep = []
            for hole in polygon.interiors:
                if Polygon(hole).area < limit:
                    count += 1
                else:
                    keep.append(hole.coords)
            filled.append(Polygon(polygon.exterior.coords, keep))
    if not count:
        return features, 0
    # An island inside a filled hole becomes part of that tissue, not a second
    # overlapping polygon. Gaps outside polygon boundaries remain untouched.
    return [{**features[0], "geometry": mapping(unary_union(filled))}], count


def filter_small_pieces(features, minimum_um2, pixel_size_um):
    """Apply the cutoff to each final polygon, including pieces split by repair."""
    from shapely.geometry import mapping, shape
    from .annotation_edits import polygons

    if not minimum_um2:
        return features, 0
    if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("A valid pixel size is needed to measure tissue in µm².")
    kept, removed = [], 0
    cutoff = minimum_um2 / pixel_size_um ** 2
    for feature in features:
        for polygon in polygons(shape(feature["geometry"])):
            if polygon.area >= cutoff:
                kept.append({**feature, "geometry": mapping(polygon)})
            else:
                removed += 1
    return kept, removed


def detect(image, config, values, *, output=None, sample=None, expected=None):
    """Tab 4's tissue detection; with *output*, write its files as Tab 4 does."""
    from spacemap.tissue_annotation_utils import (
        PyramidImageReader,
        detect_tissue_mask,
        mask_to_geojson_features,
        qupath_color_from_hex,
    )

    reader = PyramidImageReader(image)
    he, markers = channels(list(reader.channel_names), config)
    name = sample or sample_id(image)
    from . import annotation_edits
    if output is not None and expected is None:
        expected = annotation_edits.revision(valid_tissue(output, name))
    scale = float(values.get("blur_downsample", 2))
    holes_limit = float(values.get("fill_holes_um2", 0))
    area_limit = values.get("min_tissue_area_um2")
    pixel_size = float(values.get("microns_per_pixel", 0.5083))
    if holes_limit > 0 or (area_limit or 0) > 0:
        from spacemap.viewer.sources import image_info
        pixel_size = image_info(str(image)).pixel_size_um or pixel_size
    detector_values = dict(values)
    if area_limit is not None:
        check({"min_tissue_area_um2": area_limit})
        detector_values["min_tissue_island_px"] = int(math.floor(area_limit / (pixel_size * scale) ** 2))
    # Artifacts are reviewed separately, then subtracted explicitly in Studio.
    result = detect_tissue_mask(reader, {**detector_values, "sample_id": name, "detect_blood_clot": False}, he_indices=he, codex_indices=markers)
    simplify = float(values.get("boundary_simplify_px", 2.0))
    valid = mask_to_geojson_features(
        result["tissue_mask"],
        scale_factor=scale,
        class_name="Valid_Tissue",
        color_rgb=int(qupath_color_from_hex("#2ecc71")),
        simplify_px=simplify,
        merge=True,
    )
    valid, holes_filled = fill_small_holes(valid, holes_limit, pixel_size)
    # Repairing traced boundaries can return a GeometryCollection (polygons
    # plus zero-area lines). Keep every polygon, including its holes, in both
    # the preview and saved result; a collection is not "zero pieces".
    valid = annotation_edits.prepare(valid, kind="tissue", label=tissue_label(config))
    if result["tissue_mask"].any() and not valid:
        raise ValueError("Tissue was found, but its outline could not be made. Try a finer Scale or less smoothing.")
    valid, small_removed = filter_small_pieces(valid, area_limit or 0, pixel_size)
    stats = _plain(result.get("stats") or {})
    stats["holes_filled"] = holes_filled
    stats["small_pieces_removed"] = small_removed
    if holes_filled or small_removed:
        from shapely.geometry import shape
        stats["tissue_percent"] = 100 * sum(shape(f["geometry"]).area for f in valid) / math.prod(reader.full_shape)
    found = {"features": valid, "stats": stats, "written": []}
    if output is None:
        return found
    Path(output, INTERMEDIATE).mkdir(parents=True, exist_ok=True)
    edge = mask_to_geojson_features(
        result["tissue_mask"],
        scale_factor=scale,
        class_name="Tissue_Boundary",
        color_rgb=-16711936,
        simplify_px=simplify,
        merge=False,
        locked=True,
    )
    edge, _ = fill_small_holes(edge, holes_limit, pixel_size)
    edge, _ = filter_small_pieces(edge, area_limit or 0, pixel_size)
    target = valid_tissue(output, name)
    saved = annotation_edits.replace(target, valid, expected if expected is not None else annotation_edits.revision(target),
                                    "Detected tissue", kind="tissue", label=tissue_label(config))
    found['revision'] = saved['revision']
    edge_path = boundary(output, name)
    annotation_edits.replace(edge_path, edge, annotation_edits.revision(edge_path), "Detected boundary")
    found["written"] = [str(boundary(output, name)), str(valid_tissue(output, name))]
    return found


def _plain(stats):
    """Numbers and text only, for a JSON answer."""
    return {key: value for key, value in stats.items() if isinstance(value, (int, float, str, bool))}


def rings(features, limit=2000, *, include_holes=False):
    """Polygon rings, including nested GeometryCollections; optional holes use even-odd filling."""
    polygons = []
    def visit(geometry):
        if geometry.get("type") == "GeometryCollection":
            for child in geometry.get("geometries") or []:
                visit(child)
            return
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            return
        parts = (
            [geometry.get("coordinates")] if geometry.get("type") == "Polygon" else geometry.get("coordinates") or []
        )
        for part in parts:
            if part and len(polygons) < limit:
                for ring in part if include_holes else part[:1]:
                    polygons.append([[round(x, 1), round(y, 1)] for x, y in ring])
    for feature in features:
        visit(feature.get("geometry") or {})
    return polygons


def run(items, config, values, report):
    """Detect tissue and write Tab 4's files for each listed image."""
    written = []
    for index, item in enumerate(items):
        report(f"{Path(item['image']).name}: detecting tissue", index / len(items))
        written.extend(detect(item["image"], config, values, output=item["output"], sample=item["sample"], expected=item.get("tissue_revision"))["written"])
    return written
