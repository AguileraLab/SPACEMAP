"""Studio's Segmentation step: Tab 5's settings, a fast preview and whole-ROI runs.

Settings are Tab 5's own project keys (``tab4_*`` in ``tab5_segmentation``)
with its defaults. The preview runs Tab 5's Cellpose segmenter on a crop in
this process and keeps the model loaded, so after the first preview another
takes seconds. Whole ROIs run as a job through ``segment_image_cellpose``.
"""

from pathlib import Path
import threading
import time

# Tab 5's keys Studio shows and saves, with classic Tab 5's defaults.
DEFAULTS = {
    "tab4_model_type": "cyto",
    "tab4_custom_model": "",
    "tab4_diameter": 0,
    "tab4_flow": 0.8,
    "tab4_gpu": True,
    "tab4_tile_size": 2048,
    "tab4_tile_overlap": 0.1,
    "tab4_cellpose_internal_tile_size": 224,
    "tab4_cellpose_batch_size": 8,
    "tab4_cellpose_batch_auto": True,
    "tab4_gpu_device": 0,
    "tab4_adaptive_gpu_memory": True,
    "tab4_min_cellpose_internal_tile_size": 128,
    "tab4_save_geojson": True,
    "tab4_last_nuclear_names": [],
    "tab4_last_membrane_names": [],
    # Where to segment (Tab 5's mask): the whole image, inside classes of each ROI's annotation, or Tab 5's drawn mask.
    "tab4_mask_source": "No mask (whole image)",
    "tab4_mask_classes": [],
    "tab4_annot_geojson_matches": {},
    # Studio: segment inside an annotation saved in the Annotation step (its name, such as "tissue"), all of its polygons
    # together, as it is now (cleaned or not). Empty: the choice above (classic Tab 5's) decides.
    "tab4_mask_layer": "",
    # Per image (Studio): "" follows the step's choice, WHOLE the whole image, or a saved annotation's name.
    "tab4_mask_layer_matches": {},
}
WHOLE = "__whole__"
# Tab 5's spatial columns during segmentation: set on classic Tab 5; Studio's runs use them as saved.
SPATIAL_DEFAULTS = {
    "tab4_spatial_enabled": True,
    "tab4_spatial_rules": [],
    "tab4_spatial_tissue_label": "tissue",
    "tab4_spatial_edge_enabled": False,
    "tab4_spatial_edge_reference": "tissue",
    "tab4_spatial_edge_buffer": 100.0,
    "tab4_spatial_edge_column": "edge_cell",
    "tab4_spatial_min_cells_per_bed_target": 10,
    "tab4_spatial_require_clean_path": False,
}
FOLDERS = {
    "tab4_annot_geojson_dir": "outputs/processed_annotations/stage1b_compartments_zones",
    "tab4_seg_mask_save_dir": "data/boundaries",
}
EDITABLE = {
    "tab4_mask_source",
    "tab4_mask_classes",
    "tab4_annot_geojson_matches",
    "tab4_annot_geojson_dir",
    "tab4_mask_layer",
    "tab4_mask_layer_matches",
    "tab4_cellpose_batch_size",
    "tab4_cellpose_batch_auto",
    "tab4_gpu_device",
    "tab4_model_type",
    "tab4_custom_model",
    "tab4_diameter",
    "tab4_flow",
    "tab4_gpu",
    "tab4_last_nuclear_names",
    "tab4_last_membrane_names",
}
MODELS = ("cyto", "nuclei", "cyto2", "cp", "custom")
DEFAULTS_SOURCES = ("No mask (whole image)", "Annotation GeoJSON (Annotation tab)", "Drawn tissue mask (this tab)")
PREVIEW_LIMIT = 1024

_models = {}
_lock = threading.Lock()


def values(root, config):
    saved = (config.get("tab_configurations") or {}).get("tab5_segmentation") or {}
    result = {key: saved.get(key, default) for key, default in DEFAULTS.items()}
    # The custom model's path is a folder setting: the project keeps it in directory_mappings, as classic Tab 5 does.
    mapped = (config.get("directory_mappings") or {}).get("tab4_custom_model")
    if mapped:
        result["tab4_custom_model"] = str(mapped).replace("${PROJECT_ROOT}", str(root))
    if not result["tab4_custom_model"]:
        result["tab4_custom_model"] = str(Path(root) / "data/seg_model/CP_20241014_094617")
    result.update({key: saved.get(key, default) for key, default in SPATIAL_DEFAULTS.items()})
    mappings = config.get("directory_mappings") or {}
    for key, fallback in FOLDERS.items():
        value = mappings.get(key) or saved.get(key)
        result[key] = str(value).replace("${PROJECT_ROOT}", str(root)) if value else str(Path(root) / fallback)
    return result


def state_for(settings):
    """Tab 5's saved settings as its segmentation-region code reads them."""
    from spacemap_gui.tabs import segmentation_spatial_ui as spatial

    return {
        spatial.GEOJSON_DIR_KEY: settings["tab4_annot_geojson_dir"],
        spatial.MASK_SOURCE_KEY: settings["tab4_mask_source"],
        spatial.MASK_CLASSES_KEY: list(settings["tab4_mask_classes"]),
        spatial.GEOJSON_MATCHES_KEY: dict(settings["tab4_annot_geojson_matches"]),
        **{key: settings[key] for key in SPATIAL_DEFAULTS},
    }


def regions(settings, image_names):
    """Each ROI's annotation file (matched by name, or chosen by hand) and its classes; all classes; the folder's files."""
    from spacemap_gui.tabs import segmentation_spatial_ui as spatial

    from .tagging import classes_of

    state = state_for(settings)
    folder = settings["tab4_annot_geojson_dir"]
    rows, classes = [], []
    for name in image_names:
        chosen = spatial.chosen_geojson(name, state)
        path = spatial.matched_geojson(name, folder, chosen)
        names = classes_of(path)
        classes += [item for item in names if item not in classes]
        rows.append({"image": name, "annotation": Path(path).name if path else None, "chosen": chosen or "", "classes": names})
    files = sorted(item.name for item in Path(folder).glob("*.geojson")) if Path(folder).is_dir() else []
    return {"rows": rows, "classes": classes, "files": files, "sources": list(spatial.MASK_SOURCES)}


def _annotation_image(images, name):
    """The Annotation step's copy of an image (its annotations are kept by image): the same file name, else the same
    sample (a background-removed ROI shares its ROI's annotations), else the name itself."""
    from . import annotation

    same = Path(images) / name
    if same.is_file():
        return same
    sample = annotation.sample_id(name)
    for candidate in sorted(Path(images).glob(sample + "*")) if Path(images).is_dir() else []:
        if annotation.sample_id(candidate.name) == sample:
            return candidate
    for candidate in sorted(Path(images).glob("*")) if Path(images).is_dir() else []:
        if sample.startswith(annotation.sample_id(candidate.name) + "_"):
            return candidate
    return same


def saved_annotations(root, config, mappings, image_names):
    """For each image, the areas it can be segmented in, by name as Layers shows them ({image: {name: area}}): its
    tissue (all pieces), every other name of its own, and "<file> › <name>" for imported files. An area is a file's
    path, or {"path", "name"} for the polygons of one name in it (cut out by area_path when the job starts)."""
    from . import annotation, annotation_names

    images, output = annotation.folders(root, mappings)
    found = {}
    for name in image_names:
        image = _annotation_image(images, name)
        by_name = {}
        try:
            tissue = annotation.valid_tissue(output, annotation.sample_id(image.name))
            if tissue.is_file():
                by_name[annotation.tissue_label(config)] = str(tissue.resolve())
            for group in annotation_names.groups(root, config, str(image), output):
                if group["role"]:
                    continue  # artifacts and training examples are not areas to segment in
                key = group["name"] if not group["source"] else f"{group['source']} › {group['name']}"
                by_name.setdefault(key, {"path": group["files"][0], "paths": group["files"], "name": group["name"]})
                # The same for every image: "imported:tissue" is the tissue of whatever file was imported for it.
                rule = by_name.setdefault(group["rule"], {"path": group["files"][0], "paths": [], "name": group["name"]})
                rule["paths"] += [path for path in group["files"] if path not in rule["paths"]]
        except (OSError, ValueError):
            pass
        found[name] = by_name
    return found


def _shown(value):
    """A choice saved by its plain name (tissue) is the Local one of that name, as the list shows it."""
    if value and " › " not in value and value != WHOLE and not value.startswith(("local:", "imported:")):
        return "local:" + value
    return value


def _area_groups(saved, current=""):
    """An image's areas as Annotation's Export names them: [[group, [[value, label], ...]], ...]: Local (made in
    Studio) and Imported (from GeoJSON files brought in), each by name, the same values for every image, so one
    choice serves all. A choice saved the earlier way (a file's name) is listed as it is."""
    groups = {"Local": [], "Imported": []}
    for key in sorted(saved, key=str.lower):
        origin, _, name = key.partition(":")
        if origin in ("local", "imported") and name:
            groups["Local" if origin == "local" else "Imported"].append([key, name])
    found = [[group, items] for group, items in groups.items() if items]
    if current and current in saved and " › " in current:
        found.append(["Earlier choice", [[current, current.replace(" › ", ": ")]]])
    return found


def area_path(root, config, mappings, image_name, area):
    """The GeoJSON to segment in: a file as it is, or the polygons of one name cut out of it."""
    import json

    from . import annotation, annotation_names
    from .annotation_edits import _atomic, read

    if isinstance(area, str):
        return area
    images, output = annotation.folders(root, mappings)
    image = _annotation_image(images, image_name)
    entries = {e["path"]: e for e in annotation_names.files(root, config, str(image), output)}
    wanted = area.get("paths") or [area["path"]]  # every file holding the name (a tissue piece renamed in Tissue, say)
    if not any(path in entries for path in wanted):
        raise ValueError(f"{image_name}: the annotation file of {area['name']} is gone. Choose the area again.")
    chosen = []
    for path in wanted:
        entry = entries.get(path)
        if entry is None:
            continue
        features, _ = read(path, kind="tissue" if entry["kind"] == "tissue" else "regions", label=annotation.tissue_label(config))
        chosen += [f for f in features if annotation_names.name_of(f, entry["kind"], entry["layer"]) == area["name"]]
    if not chosen:
        raise ValueError(f"{image_name}: no polygons are named {area['name']} any more. Choose the area again.")
    for feature in chosen:
        feature["properties"]["classification"] = {"name": area["name"]}  # every polygon counts in the mask
    target = Path(root) / "outputs/segmentation/areas" / f"{annotation.sample_id(image_name)}__{_safe(area['name'])}.geojson"
    _atomic(target, json.dumps({"type": "FeatureCollection", "features": chosen}).encode())
    return str(target)


def _safe(text):
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")[:80] or "area"


def set_aside(output, name, geojson=None):
    """Move an image's earlier results (table, outlines, mask, picture) into previous/<time>/ before it is redone."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = _stem(name)
    target = Path(output) / "previous" / stamp
    moved = []
    outlines = results(output, name, geojson)[1]
    for path in (Path(output) / f"{stem}_cell_features.csv", outlines, Path(output) / "geojson" / f"{stem}_cell_boundaries.geojson",
                 Path(output) / "npy" / f"{stem}_mask.npy", Path(output) / "png" / f"{stem}_segmentation.png"):
        if path.is_file():
            destination = target / (path.relative_to(output) if Path(output) in path.parents else Path("geojson") / path.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
            moved.append(str(destination))
    return moved


def selected_mask(root, image, features):
    """The polygons picked on the image, written as this image's segmentation area (outputs/segmentation/areas)."""
    import json

    from .annotation_edits import _atomic, prepare

    clean = prepare(features or [])
    if not clean:
        raise ValueError("Select polygons on the image first: click one, Ctrl-click to add more.")
    from . import annotation

    target = Path(root) / "outputs/segmentation/areas" / (annotation.sample_id(Path(image).name) + "_selected.geojson")
    _atomic(target, json.dumps({"type": "FeatureCollection", "features": clean}).encode())
    return target


def check(changes):
    if not isinstance(changes, dict) or set(changes) - EDITABLE:
        raise ValueError("Only the listed segmentation settings can be changed here.")
    for key, value in changes.items():
        if key == "tab4_model_type" and value not in MODELS:
            raise ValueError("Choose cyto, nuclei, cyto2, cp or custom.")
        if key == "tab4_diameter" and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1000
        ):
            raise ValueError("The diameter must be 0 (automatic) to 1000 pixels.")
        if key == "tab4_flow" and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 3
        ):
            raise ValueError("The flow threshold must be between 0 and 3.")
        if key == "tab4_cellpose_batch_size" and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256):
            raise ValueError("The GPU batch is 1 to 256.")
        if key == "tab4_gpu_device" and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255):
            raise ValueError("The GPU index must be an integer from 0 to 255.")
        if key in {"tab4_gpu", "tab4_cellpose_batch_auto"} and not isinstance(value, bool):
            raise ValueError("Choose on or off for the GPU.")
        if key == "tab4_custom_model" and (not isinstance(value, str) or len(value) > 1024):
            raise ValueError("Enter the custom model's path.")
        if key == "tab4_mask_source" and value not in DEFAULTS_SOURCES:
            raise ValueError("Choose the whole image, inside annotation classes, or the drawn tissue mask.")
        if key == "tab4_mask_layer" and (not isinstance(value, str) or len(value) > 200):
            raise ValueError("Choose a saved annotation by name.")
        if key == "tab4_mask_layer_matches" and (
            not isinstance(value, dict) or len(value) > 5000
            or not all(isinstance(k, str) and isinstance(v, str) and len(k) <= 512 and len(v) <= 200 for k, v in value.items())
        ):
            raise ValueError("Choose where to segment each image by name.")
        if key == "tab4_mask_classes" and (
            not isinstance(value, list) or len(value) > 64 or not all(isinstance(v, str) and 0 < len(v) <= 128 for v in value)
        ):
            raise ValueError("Choose the annotation classes by name.")
        if key == "tab4_annot_geojson_matches" and (
            not isinstance(value, dict) or len(value) > 5000 or not all(
                isinstance(k, str) and isinstance(v, str) and len(k) <= 512 and len(v) <= 256 and "/" not in v and "\\" not in v
                for k, v in value.items())
        ):
            raise ValueError("Choose an annotation file from the folder for each ROI.")
        if key == "tab4_annot_geojson_dir" and (
            not isinstance(value, str) or not value or len(value) > 512 or Path(value).is_absolute() or ".." in Path(value).parts
        ):
            raise ValueError("Choose a folder inside the project.")
        if key in {"tab4_last_nuclear_names", "tab4_last_membrane_names"} and (
            not isinstance(value, list) or len(value) > 64 or not all(isinstance(v, str) and v for v in value)
        ):
            raise ValueError("Choose markers by name.")
    return changes


def save(root, changes, expected):
    """Save Tab 5 settings in the project, as classic Tab 5 would."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    check(changes)
    with project_lock(root):
        config = read_config(root)
        current = values(root, config)
        if expected != revision(current):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        for key, value in changes.items():
            if key in FOLDERS:
                value = "${PROJECT_ROOT}/" + Path(value).as_posix()  # a folder: kept with the project's folders
            persistence.update_stored_config_key(config, key, value)
        persistence.save_project_config(str(root), config)
        return values(root, read_config(root))


def config_for(settings):
    from spacemap.segmentation_cellpose import SegmentationConfig

    config = SegmentationConfig()
    config.diameter = float(settings["tab4_diameter"])
    config.flow_threshold = float(settings["tab4_flow"])
    config.use_gpu = bool(settings["tab4_gpu"])
    config.tile_size = int(settings["tab4_tile_size"])
    config.tile_overlap = float(settings["tab4_tile_overlap"])
    config.cellpose_internal_tile_size = int(settings["tab4_cellpose_internal_tile_size"])
    config.cellpose_batch_size = int(settings["tab4_cellpose_batch_size"])
    config.cellpose_batch_auto = bool(settings.get("tab4_cellpose_batch_auto", True))
    config.gpu_device = int(settings.get("tab4_gpu_device", 0))
    config.adaptive_gpu_memory = bool(settings["tab4_adaptive_gpu_memory"])
    config.min_cellpose_internal_tile_size = int(settings["tab4_min_cellpose_internal_tile_size"])
    config.save_geojson = bool(settings["tab4_save_geojson"])
    if settings["tab4_model_type"] == "custom":
        config.cellpose_model_type = "custom"
        config.cellpose_model_path = settings["tab4_custom_model"]
    else:
        config.cellpose_model_type = settings["tab4_model_type"]
        config.cellpose_model_path = ""  # a missing path selects the built-in model
    return config


def channels(names, settings):
    """(nuclear, cell-border markers) for an image: the saved choices, else DAPI (as Tab 5).

    Tab 5 gives Cellpose (border markers combined, empty, nuclear); with no
    border marker that first channel is blank and no cell is found. Studio
    then uses the nuclear marker there, in previews and whole runs alike.
    """
    nuclear = [name for name in settings["tab4_last_nuclear_names"] if name in names]
    if not nuclear:
        nuclear = [name for name in names if "dapi" in name.lower()] or names[:1]
    membranes = [name for name in settings["tab4_last_membrane_names"] if name in names and name not in nuclear]
    return nuclear[0], membranes or [nuclear[0]]


def read_crop(image, names, wanted, bounds):
    """Full-resolution planes of *wanted* markers inside *bounds* (x, y, w, h), read as the viewer reads."""
    from spacemap.viewer.sources import open_source

    x, y, width, height = (int(round(v)) for v in bounds)
    source = open_source(image)
    return [source.read_region(0, names.index(name), x, y, x + width, y + height) for name in wanted]


def _segmenter(config):
    from spacemap.segmentation_cellpose import CellposeSegmenter

    key = (config.cellpose_model_type, config.cellpose_model_path, config.use_gpu, config.gpu_device)
    if key not in _models:
        _models.clear()  # one model at a time keeps GPU memory small
        _models[key] = CellposeSegmenter(config)
    segmenter = _models[key]
    segmenter.config = config
    return segmenter


def outlines(masks, x0, y0, limit=5000):
    """Cell outlines as polygons in image pixels."""
    import cv2
    import numpy as np
    from scipy import ndimage

    polygons = []
    for label, found in enumerate(ndimage.find_objects(masks), start=1):
        if found is None or len(polygons) >= limit:
            continue
        rows, cols = found
        cell = (masks[found] == label).astype(np.uint8)
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        ring = max(contours, key=cv2.contourArea)[:, 0, :]
        polygons.append([[int(px + cols.start + x0), int(py + rows.start + y0)] for px, py in ring])
    return polygons


def preview(image, names, bounds, settings):
    """Segment the crop *bounds* with the current settings; nothing is saved."""
    import numpy as np

    x, y, width, height = bounds
    width, height = min(int(width), PREVIEW_LIMIT), min(int(height), PREVIEW_LIMIT)
    if width < 16 or height < 16:
        raise ValueError("Choose a larger area to test.")
    nuclear, membranes = channels(names, settings)
    started = time.perf_counter()
    with _lock:
        planes = read_crop(image, names, [nuclear, *membranes], (x, y, width, height))
        dapi = planes[0].astype(np.float32)
        merged = np.max(np.stack(planes[1:]), axis=0).astype(np.float32)
        config = config_for(settings)
        loading = not _models
        segmenter = _segmenter(config)
        masks = segmenter.segment(
            np.stack((merged, np.zeros_like(merged), dapi), axis=-1), tile=True, tile_overlap=config.tile_overlap
        )[0]
    polygons = outlines(np.asarray(masks), int(x), int(y))
    return {
        "count": int(np.asarray(masks).max()),
        "outlines": polygons,
        "bounds": [int(x), int(y), width, height],
        "seconds": round(time.perf_counter() - started, 2),
        "model_loaded": loading,
        "device": "GPU" if config.use_gpu and _gpu() else "CPU",
        "nuclear": nuclear,
        "membranes": membranes,
    }


def _gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def run(items, settings, report):
    """Segment whole images with Tab 5's function; images already segmented are kept."""
    from spacemap.segmentation_cellpose import get_channel_names, segment_image_cellpose

    done = []
    for index, item in enumerate(items):
        names = get_channel_names(item["image"])
        nuclear, membranes = channels(names, settings)

        def progress(done, total=None, message="", index=index, name=Path(item["image"]).name):
            fraction = done / total if total else float(done)
            report(f"{name}: {message or 'segmenting'}", (index + min(max(fraction, 0), 1)) / len(items))

        report(f"{Path(item['image']).name}: starting", index / len(items))
        config = config_for(settings)
        for note in region(config, Path(item["image"]).name, settings, item.get("area")):
            report(f"{Path(item['image']).name}: {note}", index / len(items))
        found = segment_image_cellpose(
            ome_tiff_path=item["image"],
            output_dir=item["outputs"],
            config=config,
            markers=membranes,
            dapi_channel=nuclear,
            overwrite=False,
            progress_callback=progress,
        )
        # Tab 5 writes the outlines into <output>/geojson; Studio's Outlines folder can be another one.
        made, wanted = results(item["outputs"], Path(item["image"]).name)[1], results(item["outputs"], Path(item["image"]).name, item.get("geojson"))[1]
        if made != wanted and made.is_file():
            wanted.parent.mkdir(parents=True, exist_ok=True)
            made.replace(wanted)
        cells = len(found.get("features_df", [])) if found else 0
        done.append({"image": item["image"], "cells": cells, "features": (found or {}).get("features_csv", "")})
    return done


def region(config, image_name, settings, area=None):
    """Set where Cellpose segments one ROI (and its spatial columns), exactly as classic Tab 5 does; *area* (a saved
    annotation or the selected polygons, resolved when the job was started) replaces classic's region."""
    from spacemap_gui.tabs import segmentation_spatial_ui as spatial

    kind, wanted = area_of(settings, image_name)
    if area is None and kind == "layer":
        # Asked for inside a saved annotation, but this image has none: never segment the whole image instead.
        raise ValueError(f"{image_name} has no saved {wanted} annotation. Save it in the Annotation "
                         "step, or choose where to segment again.")
    source = spatial.MASK_SOURCE_NONE  # Studio: the whole image, or the area given
    if area and area.get("whole"):
        area = None
    notes = spatial.apply_region_and_tags(config, image_name, source, settings["tab4_seg_mask_save_dir"], state=state_for(settings))
    if area:
        config.use_input_geojson = True
        config.input_geojson_path = str(area["path"])
        config.mask_class_names = []  # every polygon of it, together
        # Classic cuts artifact, clot and necrosis polygons out of a tissue; not when they are what was chosen.
        from spacemap.segmentation_cellpose import DEFAULT_MASK_EXCLUDE_CLASSES, get_geojson_class_names

        own = {str(name).lower() for name in get_geojson_class_names(str(area["path"]))}
        if area.get("exact") or (own and own <= set(DEFAULT_MASK_EXCLUDE_CLASSES)):
            config.mask_exclude_class_names = []
        notes.append(f"Segmenting inside {area['name']} ({Path(area['path']).name})")
    return notes


# ---------------------------------------------------------------- the step's page: folders, images, models
IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".qptiff", ".svs")
STEP_FOLDERS = {
    "images": (("tab4_input_dir", "tab3_output_dir"), "outputs/rois"),
    "output": (("tab4_output_dir", "cell_seg_dir"), "outputs/segmentation"),
    "models": (("tab4_model_dir",), "data/seg_model"),
}
BUILTIN = {"cyto": "Cellpose cyto", "cyto2": "Cellpose cyto2", "nuclei": "Cellpose nuclei", "cp": "Cellpose cp"}
_NOT_MODELS = {".json", ".npy", ".txt", ".csv", ".md", ".py", ".png", ".jpg", ".pdf", ".log", ".yaml", ".yml", ".partial", ".tmp"}


GEOJSON_KEY = "tab4_studio_geojson_dir"  # the cell outlines' folder; empty: <output>/geojson, as Tab 5 writes it


def step_folders(root, config, mappings):
    """Where the images are, where results go (tables in output, outlines in geojson), and the models (full paths)."""
    from .steps import _folder

    saved = (config.get("tab_configurations") or {}).get("tab5_segmentation") or {}
    places = {**mappings, **{key: saved[key] for key in ("tab4_model_dir",) if saved.get(key)}}
    found = {kind: _folder(Path(root), places, keys, fallback) for kind, (keys, fallback) in STEP_FOLDERS.items()}
    chosen = str(saved.get(GEOJSON_KEY) or "").strip()
    found["geojson"] = (Path(chosen.replace("${PROJECT_ROOT}", str(root))).expanduser() if chosen else found["output"] / "geojson")
    return found


def _stem(name):
    """The name Tab 5 gives an image's results: the file name without its last extension (P11_roi_1.ome)."""
    return Path(name).stem


def results(output, name, geojson=None):
    """An image's cell table (output folder) and outlines (*geojson*, else <output>/geojson), written by Tab 5 or Studio."""
    stem = _stem(name)
    return Path(output) / f"{stem}_cell_features.csv", Path(geojson or Path(output) / "geojson") / f"{stem}_cell_boundaries.geojson"


_counts = {}


def cell_count(table):
    """Cells in a features table (its lines, without reading it as a table); kept until the file changes."""
    try:
        info = Path(table).stat()
    except OSError:
        return None
    key = (str(table), info.st_size, info.st_mtime)
    if key not in _counts:
        from . import file_facts

        kept = file_facts.get(table, "cells")
        if kept is None:
            lines = 0
            with open(table, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    lines += block.count(b"\n")
            kept = max(lines - 1, 0)
            file_facts.put(table, "cells", kept)
        _counts[key] = kept
    return _counts[key]


def _model_files(folder):
    """Cellpose model files in a folder (a few levels down, so a model library's cellpose/<entry>/ is found)."""
    found = []
    base = Path(folder)
    if not base.is_dir():
        return found
    for path in sorted(base.rglob("*")):
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        if len(relative.parts) > 3 or any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() not in _NOT_MODELS and path.stat().st_size > 1_000_000:
            found.append(path)
    return found


def model_choices(root, models_folder):
    """Every model the step can use, in one list: Cellpose's own, this project's trained versions, the Models folder's."""
    import json

    from . import training

    choices = [{"id": key, "name": name, "group": "Cellpose", "path": ""} for key, name in BUILTIN.items()]
    seen = set()
    try:
        trained = training.examples(root)["models"]
    except Exception:
        trained = []
    for item in sorted(trained, key=lambda m: m.get("created") or m["name"], reverse=True):
        seen.add(str(Path(item["path"]).resolve()))
        detail = [item.get("created", "")[:16].replace("T", " "), "from " + Path(str(item.get("base") or "")).name if item.get("base") else ""]
        choices.append({"id": "project:" + item["name"], "name": item["name"], "group": "Trained in this project",
                        "path": item["path"], "title": " · ".join(filter(None, detail))})
    for path in _model_files(models_folder):
        if str(path.resolve()) in seen:
            continue
        meta = {}
        manifest = path.parent / "model.json"
        if manifest.is_file():
            try:
                meta = json.loads(manifest.read_text())
            except ValueError:
                meta = {}
        relative = path.relative_to(models_folder).as_posix()
        title = (f"From {meta['source_project']} (library)" if meta.get("source_project") else relative)
        choices.append({"id": "folder:" + relative, "name": meta.get("name") or path.name, "group": "Models folder",
                        "path": str(path), "title": title})
    return choices


def current_model(values, choices):
    """The id of the model the saved settings use (an "other" entry when it is outside both lists)."""
    if values["tab4_model_type"] != "custom":
        return values["tab4_model_type"], None
    path = values.get("tab4_custom_model") or ""
    for item in choices:
        if item["path"] and path and Path(item["path"]).resolve() == Path(path).resolve():
            return item["id"], None
    if not path:
        return "", None
    return "other", {"id": "other", "name": Path(path).name + (" (missing)" if not Path(path).exists() else ""),
                     "group": "Other", "path": path, "title": path}


def area_of(values, name):
    """Where one image is segmented in Studio: (kind, name), kind "whole" or "layer". The image's own choice, else
    the one set for all images, else the whole image. (Classic Tab 5's annotation-file and drawn-mask choices are
    classic Tab 5's; Studio picks one of each image's annotations instead.)"""
    chosen = (values.get("tab4_mask_layer_matches") or {}).get(name, "")
    if chosen == WHOLE:
        return "whole", None
    if chosen:
        return "layer", chosen
    if values.get("tab4_mask_layer"):
        return "layer", values["tab4_mask_layer"]
    return "whole", None


def plan(root, config, mappings):
    """The step's page: folders, each image with where it is segmented and whether it is done, and the models."""
    from .settings import revision

    values_now = values(root, config)
    places = step_folders(root, config, mappings)
    names = []
    if places["images"].is_dir():
        names = sorted(path.name for path in places["images"].iterdir()
                       if path.is_file() and path.name.lower().endswith(IMAGE_SUFFIXES) and not path.name.startswith("."))
    saved = saved_annotations(root, config, mappings, names)
    rows = []
    for name in names:
        table, outlines = results(places["output"], name, places["geojson"])
        kind, area = area_of(values_now, name)
        rows.append({"image": name, "segmented": table.is_file(), "cells": cell_count(table) if table.is_file() else None,
                     "outlines": outlines.is_file(), "stamp": outlines.stat().st_mtime_ns if outlines.is_file() else 0,
                     "saved": sorted(saved.get(name, {})),
                     "saved_groups": _area_groups(saved.get(name, {}), (values_now.get("tab4_mask_layer_matches") or {}).get(name, "") or values_now.get("tab4_mask_layer") or ""),
                     "area": _shown((values_now.get("tab4_mask_layer_matches") or {}).get(name, "")),
                     "effective": area if kind == "layer" else ("Whole image" if kind == "whole" else "Annotation file"),
                     "missing": kind == "layer" and area not in saved.get(name, {})})
    from . import file_facts

    file_facts.save()  # names and cell counts read this time are kept for the next start
    choices = model_choices(root, places["models"])
    model, other = current_model(values_now, choices)
    kind, layer = area_of({**values_now, "tab4_mask_layer_matches": {}}, "")
    default = "inside " + layer if kind == "layer" else "whole image"
    custom = bool(str(((config.get("tab_configurations") or {}).get("tab5_segmentation") or {}).get(GEOJSON_KEY) or "").strip())
    return {"folders": {kind: str(path) for kind, path in places.items()}, "geojson_custom": custom, "rows": rows,
            "models": choices + ([other] if other else []), "model": model,
            "layer": _shown(values_now["tab4_mask_layer"]), "default_area": default, "revision": revision(values_now),
            "tagging_rules": _tagging_columns(root, config)}


def _tagging_columns(root, config):
    """How many things the Cell tagging step is set to add (annotation columns, edge, beds): what Then tag uses."""
    try:
        from spacemap_gui.tabs import cell_tagging_ui as tab5

        from . import tagging

        state = tagging.values(root, config)
        rules, edge, *_ = tab5.build_engine_inputs(state)
        return sum(1 for rule in rules if rule.is_active()) + (edge is not None) + len(tab5.selected_bed_labels(state=state))
    except Exception:  # noqa: BLE001 - a page count must never stop the list
        return 0


def save_plan(root, changes):
    """Save the step's folders (full paths), the model (by its id), or where one image is segmented."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - {"folders", "model", "areas"}:
        raise ValueError("Only the folders, the model and each image's area can be changed here.")
    base = Path(root).resolve()
    with project_lock(root):
        config = read_config(root)
        mappings = config.get("directory_mappings") or {}
        for kind, value in (changes.get("folders") or {}).items():
            if kind == "geojson" and isinstance(value, str) and len(value) <= 1024:  # empty: back to <output>/geojson
                if value.strip():
                    target = Path(value.strip()).expanduser()
                    target = (target if target.is_absolute() else base / target).resolve()
                    if not target.parent.is_dir():
                        raise ValueError(f"{target.parent} is not a folder.")
                    value = "${PROJECT_ROOT}/" + target.relative_to(base).as_posix() if base in target.parents else str(target)
                persistence.update_stored_config_key(config, GEOJSON_KEY, value.strip())
                continue
            if kind not in STEP_FOLDERS or not isinstance(value, str) or not value.strip() or len(value) > 1024:
                raise ValueError("Type the folder's full path.")
            target = Path(value.strip()).expanduser()
            target = (target if target.is_absolute() else base / target).resolve()
            if kind in ("images", "models") and not target.is_dir():
                raise ValueError(f"{target} is not a folder.")
            if not target.parent.is_dir():
                raise ValueError(f"{target.parent} is not a folder.")
            stored = "${PROJECT_ROOT}/" + target.relative_to(base).as_posix() if target == base or base in target.parents else str(target)
            for key in STEP_FOLDERS[kind][0][:1]:
                persistence.update_stored_config_key(config, key, stored)
        if "model" in changes:
            places = step_folders(root, config, mappings)
            chosen = next((item for item in model_choices(root, places["models"]) if item["id"] == changes["model"]), None)
            if chosen is None:
                raise ValueError("That model is not in the list any more. Choose another.")
            if chosen["group"] == "Cellpose":
                persistence.update_stored_config_key(config, "tab4_model_type", chosen["id"])
            else:
                persistence.update_stored_config_key(config, "tab4_model_type", "custom")
                persistence.update_stored_config_key(config, "tab4_custom_model", chosen["path"])
        if "areas" in changes:
            areas = changes["areas"]
            check({"tab4_mask_layer_matches": areas})
            merged = {**(values(root, config).get("tab4_mask_layer_matches") or {}), **areas}
            persistence.update_stored_config_key(config, "tab4_mask_layer_matches", {k: v for k, v in merged.items() if v})
        persistence.save_project_config(str(root), config)
