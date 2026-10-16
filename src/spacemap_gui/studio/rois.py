"""Studio's Regions of interest step: Tab 3's ROI files and extraction.

ROIs are drawn into the file Tab 3 reads
(``<tab3_geojson_dir>/<image>_roi_bounding_boxes_coords.geojson``) and cut
out by Tab 3's own ``process_single_image_rois``; an extracted ROI is
``<rois>/<image>_<ROI name>.ome.tiff``.
"""

from pathlib import Path
import re

IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".qptiff", ".svs")


def _path(root, value):
    return Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()


IMAGES_KEY = "tab3_studio_image_dir"  # the images folder typed in Studio (kept as typed)
FILES_KEY = "tab3_studio_roi_files"  # {image: the GeoJSON picked for its ROIs}
TMA_KEY = "tab3_studio_tma"  # the core detection settings
TMA_DEFAULTS = {"min": 30, "max": 150, "threshold": 0, "gap": 0.5, "format": "A-Z", "prefix": ""}
FOLDER_KEYS = {"images": (IMAGES_KEY,), "shapes": ("tab3_geojson_dir",), "output": ("tab3_output_dir", "roi_output_dir")}


def _saved(config, key, default=None):
    for fields in ((config or {}).get("tab_configurations") or {}).values():
        if isinstance(fields, dict) and key in fields:
            return fields[key]
    return default


def folders(root, mappings, config=None):
    """(images, ROI files, extracted ROIs). The images folder typed in Studio is used as typed; else as classic
    Tab 3 chooses it: its own folder, or when empty or an earlier step's, the pipeline's (Tab 2's, else Tab 1's)."""
    from spacemap_gui.pipeline_folders import follow_pipeline

    resolved = {
        key: str(_path(root, value)) for key, value in (mappings or {}).items() if isinstance(value, str) and value
    }
    typed = _saved(config, IMAGES_KEY)
    chosen = str(_path(root, typed)) if typed else follow_pipeline(resolved.get("tab3_input_dir"), resolved)[0]
    images = Path(chosen) if chosen else Path(root) / "outputs/registered"
    shapes = Path(resolved.get("tab3_geojson_dir") or Path(root) / "data/boundaries")
    outputs = Path(resolved.get("tab3_output_dir") or resolved.get("roi_output_dir") or Path(root) / "outputs/rois")
    return images, shapes, outputs


def stem(image):
    """The image's name as Tab 3 uses it (no .ome.tiff, no _merged)."""
    from spacemap.background_subtraction_utils import normalize_image_stem

    value = normalize_image_stem(image)
    return value[: -len("_merged")] if value.endswith("_merged") else value


def roi_file(shapes_dir, image):
    return Path(shapes_dir) / f"{stem(image)}_roi_bounding_boxes_coords.geojson"


def shapes_file(shapes_dir, image, picked=None):
    """The GeoJSON with an image's ROIs: the one picked for it, else Tab 3's file by name."""
    if picked and (Path(shapes_dir) / picked).is_file():
        return Path(shapes_dir) / picked
    return roi_file(shapes_dir, image)


def all_folders(root, mappings, config):
    images, shapes, outputs = folders(root, mappings, config)
    return {"images": images, "shapes": shapes, "output": outputs}


def images_plan(root, mappings, config):
    """Every image in the images folder: its ROI file (by name or picked), how many ROIs it has and how many are
    extracted, and whether its markers were chosen."""
    from .background import _is_image

    places = all_folders(root, mappings, config)
    picks = _saved(config, FILES_KEY, {}) or {}
    pairs = next((tab[PAIR_KEY] for tab in (config.get("tab_configurations") or {}).values()
                  if isinstance(tab, dict) and isinstance(tab.get(PAIR_KEY), dict)), {})
    names = sorted(p.name for p in places["images"].iterdir() if p.is_file() and _is_image(p.name)) if places["images"].is_dir() else []
    rows = []
    for name in names:
        file = shapes_file(places["shapes"], name, picks.get(name))
        try:
            found = roi_names(file)
        except Exception:  # an unreadable file is still listed
            found = []
        rows.append({
            "image": name, "file": file.name, "by_name": roi_file(places["shapes"], name).name,
            "chosen": picks.get(name, "") if picks.get(name) and (places["shapes"] / picks[name]).is_file() else "",
            "rois": len(found), "extracted": len(extracted(places["output"], name, found)),
            "edited": bool((pairs.get(name) or {}).get("export_map")),
        })
    geojson = sorted(p.name for p in places["shapes"].glob("*.geojson")) if places["shapes"].is_dir() else []
    tma = {**TMA_DEFAULTS, **(_saved(config, TMA_KEY, {}) or {})}
    return {"folders": {kind: str(folder) for kind, folder in places.items()}, "rows": rows, "geojson_files": geojson, "tma": tma}


def save_step(root, changes):
    """Save the step's folders (full paths), the ROI file picked per image, and the core detection settings."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - {"folders", "files", "tma"}:
        raise ValueError("Only the ROI step's folders, ROI files and core settings can be changed here.")
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
    files = changes.get("files") or {}
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) and len(v) <= 256 and "/" not in v for k, v in files.items()):
        raise ValueError("Pick a GeoJSON file for each image.")
    tma = changes.get("tma")
    if tma is not None:
        if not isinstance(tma, dict) or set(tma) - set(TMA_DEFAULTS):
            raise ValueError("Unknown core setting.")
        tma = {**TMA_DEFAULTS, **tma}
        try:
            tma.update(min=int(tma["min"]), max=int(tma["max"]), threshold=int(tma["threshold"]), gap=float(tma["gap"]), prefix=str(tma["prefix"])[:32])
        except (TypeError, ValueError):
            raise ValueError("Core sizes, threshold and row gap are numbers.") from None
        if not (5 <= tma["min"] < tma["max"] <= 700) or not (0 <= tma["threshold"] <= 255) or not (0.1 <= tma["gap"] <= 2) or tma["format"] not in ("A-Z", "1-N"):
            raise ValueError("Core size 5 to 700 px (smallest below largest), threshold 0 to 255, row gap 0.1 to 2, rows A-Z or 1-N.")
    with project_lock(root):
        config = read_config(root)
        for kind, value in resolved.items():
            for key in FOLDER_KEYS[kind]:
                persistence.update_stored_config_key(config, key, value)
        if files:
            merged = {**(_saved(config, FILES_KEY, {}) or {}), **files}
            persistence.update_stored_config_key(config, FILES_KEY, {k: v for k, v in merged.items() if v})
        if tma is not None:
            persistence.update_stored_config_key(config, TMA_KEY, tma)
        persistence.save_project_config(str(root), config)


def roi_names(path):
    """The ROI names saved in a Tab 3 file, in order."""
    from spacemap.utils import extract_rois_with_geometry_from_geojson, load_geojson

    if not Path(path).is_file():
        return []
    data = load_geojson(str(path))
    return [label for _, label, _ in extract_rois_with_geometry_from_geojson(data)] if data else []


def safe_label(label):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)


def extracted(outputs_dir, image, names):
    """Which of *names* already have an extracted image."""
    return [name for name in names if (Path(outputs_dir) / f"{stem(image)}_{safe_label(name)}.ome.tiff").is_file()]


PAIR_KEY = "tab3_multi_pair_settings"


def pair_settings(config, image_name):
    """Classic Tab 3's settings for one image (which markers to export, their new names), from the project file."""
    for tab in (config.get("tab_configurations") or {}).values():
        if isinstance(tab, dict) and isinstance(tab.get(PAIR_KEY), dict):
            return dict(tab[PAIR_KEY].get(image_name) or {})
    return {}


def channel_config(display_path, channel_names, pair=None):
    """Tab 3's channel settings: the markers Tab 3 exports for this image (all unless unticked), each renamed as
    Tab 3 says, else as the viewer names it, with the viewer's colours."""
    from spacemap.utils import hex_to_rgb
    from spacemap.viewer import display

    pair = pair or {}
    exported, renamed, recoloured = pair.get("export_map") or {}, pair.get("rename_map") or {}, pair.get("color_map") or {}
    saved = {entry["index"]: entry for entry in display.load(display_path)["channels"]} if display_path else {}
    rename, colours = {}, {}
    for index, name in enumerate(channel_names):
        entry = saved.get(index)
        viewer = entry if entry and entry.get("name") == name else {}
        label = renamed.get(name) or viewer.get("label") or name
        if label != name:
            rename[name] = label
        if recoloured.get(name) or viewer.get("color"):
            colours[label] = hex_to_rgb(recoloured.get(name) or viewer["color"])
    keep = [name for name in channel_names if exported.get(name, True)]
    return {"keep": keep, "rename": rename, "colors": colours}


def markers(root, image_path, display_path):
    """The image's markers as Tab 3 would extract them: kept or not, name, colour (the viewer's, else the file's),
    dye and exposure, with the new name and colour chosen for this image."""
    from spacemap.viewer import display

    from .background import channels_of
    from .settings import read_config

    items = channels_of(Path(image_path))
    names = [item["name"] for item in items]
    pair = pair_settings(read_config(root), Path(image_path).name)
    chosen = channel_config(display_path, names, pair)
    saved = {entry["index"]: entry for entry in display.load(display_path)["channels"]} if display_path else {}
    return [
        {
            "index": index, "name": item["name"], "keep": item["name"] in chosen["keep"],
            "rename": (pair.get("rename_map") or {}).get(item["name"], ""), "label": chosen["rename"].get(item["name"], item["name"]),
            "color": ((saved.get(index) or {}).get("color") if (saved.get(index) or {}).get("name") == item["name"] else None) or item["color"],
            "new_color": (pair.get("color_map") or {}).get(item["name"], ""), "dye": item["dye"], "exposure": item["exposure"],
        }
        for index, item in enumerate(items)
    ]


def save_markers(root, image_path, keep, rename, colors=None):
    """Save which markers Tab 3 extracts for this image, their new names and colours, where classic Tab 3 keeps them."""
    from spacemap.background_subtraction_utils import get_channel_info
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    names = list(get_channel_info(str(image_path))[0])
    colors = colors or {}
    if not isinstance(keep, list) or not keep or set(keep) - set(names):
        raise ValueError("Keep at least one of the image's markers.")
    if not isinstance(rename, dict) or set(rename) - set(names):
        raise ValueError("Rename only the image's markers.")
    if not isinstance(colors, dict) or set(colors) - set(names) or any(not re.fullmatch(r"#[0-9a-fA-F]{6}", str(v)) for v in colors.values()):
        raise ValueError("Colours are #rrggbb, for the image's markers.")
    rename = {name: str(new).strip() for name, new in rename.items() if str(new).strip() and str(new).strip() != name}
    for new in rename.values():
        if len(new) > 64 or not re.fullmatch(r"[A-Za-z0-9 _.+\-/]+", new):
            raise ValueError(f"'{new}': use letters, numbers, spaces and . _ - + / (up to 64).")
    final = [rename.get(name, name) for name in names if name in keep]
    if len(set(final)) != len(final):
        raise ValueError("Two kept markers would have the same name.")
    with project_lock(root):
        config = read_config(root)
        everything = {}
        for tab in (config.get("tab_configurations") or {}).values():
            if isinstance(tab, dict) and isinstance(tab.get(PAIR_KEY), dict):
                everything = dict(tab[PAIR_KEY])
        settings = dict(everything.get(Path(image_path).name) or {})
        settings.update(export_map={name: name in keep for name in names}, rename_map=rename, color_map={k: v.lower() for k, v in colors.items()})
        everything[Path(image_path).name] = settings
        persistence.update_stored_config_key(config, PAIR_KEY, everything)
        persistence.save_project_config(str(root), config)
    return markers(root, image_path, None)


def _overview(path, width=700):
    """The image's DAPI (else first marker) as an 8-bit picture *width* px wide, as classic Tab 3 detects cores on,
    and the scale back to full-resolution pixels."""
    import numpy as np
    from PIL import Image
    from spacemap.viewer.render import _resize
    from spacemap.viewer.sources import image_info, open_source

    info = image_info(str(path))
    channel = next((i for i, name in enumerate(info.channel_names) if "dapi" in name.lower()), 0)
    level = next((i for i in range(len(info.levels) - 1, -1, -1) if info.levels[i].width >= width), 0)
    data = open_source(str(path)).read_region(level, channel, 0, 0, info.levels[level].width, info.levels[level].height)
    if data.ndim == 3:
        data = data.mean(axis=2)
    height = max(1, round(info.height * width / info.width))
    data = _resize(np.asarray(data, np.float32), width, height)
    low, high = np.percentile(data, (1, 99.5))
    picture = np.clip((data - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(picture, "L"), info.width / width, info.height / height


def detect_cores(image_path, target, settings):
    """Find the TMA cores of an image with classic Tab 3's detector and save them as its ROIs (named by row and
    column: A1, A2, B1…). Returns how many."""
    from spacemap.viewer import shapes
    from spacemap_gui.tabs.tab3_roi_extraction import detect_tma_cores

    settings = {**TMA_DEFAULTS, **(settings or {})}
    picture, sx, sy = _overview(image_path)
    found, labels = detect_tma_cores(picture, settings["min"], settings["max"], settings["threshold"], settings["gap"],
                                     8, 12, settings["format"], "1-N")
    features = []
    for box, label in zip(found, labels):
        x0, y0 = box["left"] * sx, box["top"] * sy
        x1, y1 = (box["left"] + box["width"]) * sx, (box["top"] + box["height"]) * sy
        features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
                         "properties": {"name": settings["prefix"] + label, "classification": {"name": "ROI", "color": [255, 214, 0]}}})
    shapes.save(target, features)
    return len(features)


def detect_many(items, report):
    written = []
    for number, item in enumerate(items):
        report(f"{Path(item['image']).name}: finding cores", number / len(items))
        written.append({"image": Path(item["image"]).name, "cores": detect_cores(item["image"], item["target"], item["settings"])})
    return written


def extract(items, report):
    """Cut every listed image's ROIs with Tab 3's function; ROIs already extracted are kept."""
    from spacemap.background_subtraction_utils import get_channel_info
    from spacemap.registration import process_single_image_rois

    written = []
    for index, item in enumerate(items):
        image = item["image"]
        names, _, _, detected = get_channel_info(image)
        mpp = detected if isinstance(detected, dict) and detected.get("codex") else {"codex": item.get("mpp") or 1.0}
        before = (
            set(Path(item["outputs"]).glob(f"{stem(image)}_*.ome.tiff")) if Path(item["outputs"]).is_dir() else set()
        )

        def progress(done, total, index=index, name=Path(image).name):
            report(f"{name}: ROI {done} of {total}", (index + done / max(total, 1)) / len(items))

        Path(item["outputs"]).mkdir(parents=True, exist_ok=True)
        process_single_image_rois(
            image,
            item["rois"],
            item["outputs"],
            mpp,
            progress_callback=progress,
            channel_config=channel_config(item.get("display"), names, item.get("pair")),
            keep_outside=bool(item.get("keep_outside")),
        )
        after = set(Path(item["outputs"]).glob(f"{stem(image)}_*.ome.tiff"))
        written.extend(sorted(str(path) for path in after - before))
    return written
