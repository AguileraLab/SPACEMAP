"""Studio's Background step: Tab 2's threshold tables and background-removed images.

Levels are saved in the file and layout Tab 2 writes and RESOLVE reads
(``<threshold_dir>/<image>.ome.tiff_-_Merged_Image.csv`` with ``Marker`` and
``Polygon_1``), and images are written by Tab 2's own
``subtract_background_image``.
"""

from pathlib import Path

IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")


def _path(root, value):
    return Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()


def _mapped(root, mappings, keys, fallback):
    value = next((mappings[key] for key in keys if (mappings or {}).get(key)), None)
    return _path(root, value) if value else Path(root) / fallback


def folders(root, mappings, config=None):
    """(images, threshold tables, background-removed images) as Tab 2 uses them."""
    chosen = next(
        (
            fields["tab2_bgdet_image_dir"]
            for fields in ((config or {}).get("tab_configurations") or {}).values()
            if isinstance(fields, dict) and fields.get("tab2_bgdet_image_dir")
        ),
        None,
    )
    images = (
        _path(root, chosen)
        if chosen
        else _mapped(root, mappings, ("tab3_output_dir", "roi_output_dir"), "outputs/rois")
    )
    return (
        images,
        _mapped(root, mappings, ("threshold_dir",), "data/thresholds"),
        _mapped(root, mappings, ("tab2_output_dir", "bg_subtraction_output_dir"), "outputs/background_subtracted"),
    )


def stem(filename):
    """Tab 2's image stem: the name without its TIFF suffix."""
    name = Path(filename).name
    for suffix in IMAGE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def table_path(threshold_dir, image):
    from spacemap.background_detection import threshold_csv_name

    return Path(threshold_dir) / threshold_csv_name(stem(image))


def output_path(output_dir, image):
    return Path(output_dir) / f"{stem(image)}_bg_subtracted.ome.tiff"


def read_levels(path):
    """Marker -> level from a Tab 2 table; several polygons are averaged, as Tab 2 detects them."""
    import pandas as pd

    frame = pd.read_csv(path)
    if "Marker" not in frame:
        raise ValueError(f"{Path(path).name} has no Marker column.")
    columns = [column for column in frame.columns if str(column).startswith("Polygon_")] or (
        ["threshold"] if "threshold" in frame else []
    )
    if not columns:
        raise ValueError(f"{Path(path).name} has no threshold values.")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return {str(marker): float(value) for marker, value in zip(frame["Marker"], values) if pd.notna(value)}


def save_levels(path, levels, method):
    """Write *levels* as a pooled Tab 2 table (one Polygon_1 per marker)."""
    import pandas as pd

    from spacemap.background_detection import write_threshold_csv

    table = pd.DataFrame(
        {
            "Marker": list(levels),
            "threshold": [float(value) for value in levels.values()],
            "method": method or "Studio",
        }
    )
    return write_threshold_csv(table, path, precision=15)


def subtract(images, report):
    """Write each image's background-removed copy; a partial file is never left as the result."""
    import os

    from spacemap.background_detection import subtract_background_image

    written = []
    for index, item in enumerate(images):
        target = Path(item["output"])
        partial = target.with_name(target.name + ".partial")
        name = Path(item["image"]).name

        def progress(done, total, marker, index=index, name=name):
            report(f"{name}: {marker} ({done} of {total})", (index + done / max(total, 1)) / len(images))

        try:
            subtract_background_image(item["image"], read_levels(item["table"]), partial, progress=progress)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        written.append(str(target))
    return written


# ---------------------------------------------------------------- the Background step (Studio)
# Tab 2's two routes in one step: blank cycles subtracted from the markers they serve (classic Tab 2's
# "Subtract blank channels": blanks named Blank/Empty/Control, paired by dye), and/or each marker's
# background level (Tab 2's Auto methods, on the whole image or inside an annotation class), then kept
# markers, new names and colours. Settings are saved with the project under Tab 2's prefix.
import re
from functools import lru_cache

ANNOTATION_KEY = "tab2_bg_annotation_dir"
MATCHES_KEY = "tab2_bg_annotation_matches"  # {image: annotation file}, picked by hand
CLASSES_KEY = "tab2_bg_annotation_classes"  # {image: annotation class to measure in}
CHANNELS_KEY = "tab2_bg_channels"  # {image: {"keep": [...], "rename": {...}, "colors": {...}, "blanks": {marker: blank}}}
BLANK = re.compile(r"blank|empty|control", re.IGNORECASE)
FOLDER_KEYS = {
    "images": ("tab2_bgdet_image_dir",),
    "annotations": (ANNOTATION_KEY,),
    "tables": ("threshold_dir",),
    "output": ("tab2_output_dir", "bg_subtraction_output_dir"),
}


def _saved(config, key, default=None):
    for fields in ((config or {}).get("tab_configurations") or {}).values():
        if isinstance(fields, dict) and key in fields:
            return fields[key]
    return default


def all_folders(root, mappings, config):
    """The step's four folders as paths: images, annotations, threshold tables, background-removed images."""
    images, tables, output = folders(root, mappings, config)
    chosen = _saved(config, ANNOTATION_KEY) or (mappings or {}).get("annot_ta_output_dir")
    annotations = _path(root, chosen) if chosen else Path(root) / "outputs/processed_annotations/stage0_tissue_annotation"
    return {"images": images, "annotations": annotations, "tables": tables, "output": output}


def _is_image(name):
    from spacemap.background_subtraction_utils import is_supported_image_file

    return is_supported_image_file(name)


@lru_cache(maxsize=128)
def _channels(path, stamp):
    from spacemap.background_subtraction_utils import get_channel_info

    names, colours, meta, _ = get_channel_info(path)
    as_hex = lambda c: c if isinstance(c, str) else "#%02x%02x%02x" % tuple(int(v) for v in c[:3])  # noqa: E731
    return [
        {"name": name, "color": as_hex(colour).lower(), "dye": str((info or {}).get("dye") or "-"),
         "exposure": str((info or {}).get("exposure") or "-"), "blank": bool(BLANK.search(name))}
        for name, colour, info in zip(names, colours, meta)
    ]


def channels_of(path):
    stamp = Path(path).stat()
    return [dict(item) for item in _channels(str(path), (stamp.st_size, stamp.st_mtime_ns))]


def default_blank(dye, blanks):
    """The blank for a marker by its dye, as classic Tab 2 pairs them (ATTO550, AF750, Cy5/AF647)."""
    squeeze = lambda text: re.sub(r"[\s_-]+", "", str(text).lower())  # noqa: E731  ("ATTO 550" and "Blank_ATTO550" alike)
    dye = squeeze(dye)
    for blank in blanks:
        name = squeeze(blank)
        if ("atto550" in dye and "atto550" in name) or ("af750" in dye and "af750" in name) or (
                ("cy5" in dye or "af647" in dye) and ("cy5" in name or "af647" in name)):
            return blank
    return ""


def annotation_for(folder, image, chosen=None):
    import os

    from spacemap.segmentation_cellpose import find_annotation_geojson

    if not folder or not Path(folder).is_dir():
        return None
    if chosen:
        path = Path(folder) / os.path.basename(chosen)
        if path.is_file():
            return path  # a file picked by hand; one no longer in this folder falls back to the name
    found = find_annotation_geojson(str(image), str(folder))
    if found:
        return Path(found)
    # Else the one file naming the image in words and numbers (P11-ROI1 old.geojson for P11_roi_1), as imports match.
    from .annotation_names import match_image, studio_export

    named = [p for p in Path(folder).glob("*.geojson") if match_image([str(image)], p.name) and not studio_export(p)]
    return named[0] if len(named) == 1 else None


def images_plan(root, mappings, config):
    """Every image in the images folder: its annotation (matched by name or picked), its classes, its blank
    cycles, and whether its levels are saved and its background removed."""
    from .tagging import classes_of

    places = all_folders(root, mappings, config)
    matches = _saved(config, MATCHES_KEY, {}) or {}
    chosen_classes = _saved(config, CLASSES_KEY, {}) or {}
    rows = []
    names = sorted(p.name for p in places["images"].iterdir() if p.is_file() and _is_image(p.name)) if places["images"].is_dir() else []
    for name in names:
        annotation = annotation_for(places["annotations"], name, matches.get(name))
        classes = classes_of(annotation) if annotation else []
        try:
            blanks = [item["name"] for item in channels_of(places["images"] / name) if item["blank"]]
        except Exception:  # an unreadable image is still listed
            blanks = []
        default_class = next((c for c in classes if c.lower() in ("valid_tissue", "tissue")), classes[0] if classes else "")
        rows.append({
            "image": name, "annotation": annotation.name if annotation else None,
            "chosen": matches.get(name, "") if matches.get(name) and (places["annotations"] / matches[name]).is_file() else "",
            "classes": classes, "class": chosen_classes.get(name, default_class) if classes else "",
            "blanks": blanks, "levels": table_path(places["tables"], name).is_file(), "detected": bool(stored_levels(root, places['images'] / name)),
            "removed": output_path(places["output"], name).is_file(), "edited": name in (_saved(config, CHANNELS_KEY, {}) or {}),
        })
    geojson = sorted(p.name for p in places["annotations"].glob("*.geojson")) if places["annotations"].is_dir() else []
    return {"folders": {kind: str(folder) for kind, folder in places.items()}, "rows": rows, "geojson_files": geojson}


def image_markers(root, mappings, config, image):
    """One image's markers: dye, exposure, colour, whether it is a blank, its blank (saved, else by dye),
    kept or not, new name and colour; and its saved levels."""
    places = all_folders(root, mappings, config)
    path = places["images"] / image
    items = channels_of(path)
    blanks = [item["name"] for item in items if item["blank"]]
    saved = (_saved(config, CHANNELS_KEY, {}) or {}).get(image) or {}
    keep = set(saved.get("keep", [item["name"] for item in items if not item["blank"]]))  # blanks left out, as classic Tab 2 does
    for item in items:
        item["keep"] = item["name"] in keep
        item["rename"] = (saved.get("rename") or {}).get(item["name"], "")
        item["new_color"] = (saved.get("colors") or {}).get(item["name"], "")
        item["subtract"] = "" if item["blank"] else (saved.get("blanks") or {}).get(item["name"], default_blank(item["dye"], blanks))
    table = table_path(places["tables"], image)
    return {"image": image, "markers": items, "blanks": blanks, "edited": bool(saved),
            "levels": read_levels(table) if table.is_file() else None, "table": table.name}


def save_step(root, changes):
    """Save the step's folders (full paths), annotation picks and classes, and one image's marker choices."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - {"folders", "matches", "classes", "image", "markers"}:
        raise ValueError("Only the Background step's folders, annotations and markers can be changed here.")
    base = Path(root).resolve()
    resolved = {}
    for kind, value in (changes.get("folders") or {}).items():
        if kind not in FOLDER_KEYS or not isinstance(value, str) or not value.strip() or len(value) > 1024:
            raise ValueError("Type the folder's path.")
        target = Path(value.strip()).expanduser()
        target = (target if target.is_absolute() else base / target).resolve()
        if kind in ("images", "annotations") and not target.is_dir():
            raise ValueError(f"{target} is not a folder.")
        if not target.parent.is_dir():
            raise ValueError(f"{target.parent} is not a folder.")
        resolved[kind] = "${PROJECT_ROOT}/" + target.relative_to(base).as_posix() if target == base or base in target.parents else str(target)
    for key in ("matches", "classes"):
        value = changes.get(key) or {}
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) and len(v) <= 256 and "/" not in v for k, v in value.items()):
            raise ValueError("Pick an annotation file and class for each image.")
    markers = changes.get("markers")
    if markers is not None:
        if not isinstance(changes.get("image"), str) or not isinstance(markers, dict) or set(markers) - {"keep", "rename", "colors", "blanks"}:
            raise ValueError("Send one image's marker choices.")
        if not markers.get("keep"):
            raise ValueError("Keep at least one marker.")
        if any(not re.fullmatch(r"#[0-9a-fA-F]{6}", str(v)) for v in (markers.get("colors") or {}).values()):
            raise ValueError("Colours are #rrggbb.")
    with project_lock(root):
        config = read_config(root)
        for kind, value in resolved.items():
            for key in FOLDER_KEYS[kind]:
                persistence.update_stored_config_key(config, key, value)
        for key, name in (("matches", MATCHES_KEY), ("classes", CLASSES_KEY)):
            if changes.get(key):
                merged = {**(_saved(config, name, {}) or {}), **changes[key]}
                persistence.update_stored_config_key(config, name, {k: v for k, v in merged.items() if v})
        if markers is not None:
            stored = dict(_saved(config, CHANNELS_KEY, {}) or {})
            stored[changes["image"]] = {"keep": list(markers.get("keep") or []), "rename": {k: v for k, v in (markers.get("rename") or {}).items() if v},
                                        "colors": dict(markers.get("colors") or {}), "blanks": dict(markers.get("blanks") or {})}
            persistence.update_stored_config_key(config, CHANNELS_KEY, stored)
        persistence.save_project_config(str(root), config)


def remove(items, report):
    """Each image with its background removed: its blank subtracted from each marker (classic Tab 2), then its
    level (Tab 2's thresholds), never below zero; kept markers, new names and colours; the parent's metadata."""
    import os

    import numpy as np
    import tifffile

    from spacemap.background_subtraction_utils import get_image_mpp, read_channel_from_open_tif
    from spacemap.image_metadata import inherit_ome
    from spacemap.registration import _pack_cached_plane, _unpack_cached_plane, _write_ome_pyramid_from_plane_factory

    written = []
    for number, item in enumerate(items):
        path, target = Path(item["image"]), Path(item["output"])
        markers = channels_of(path)
        names = [m["name"] for m in markers]
        kept = [i for i, m in enumerate(markers) if m["name"] in set(item["keep"])]
        levels = item.get("levels") or {}
        blanks = {marker: blank for marker, blank in (item.get("blanks") or {}).items() if blank in names and marker in names}
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(stem(target.name) + ".partial.ome.tiff")  # still .ome.tiff: tifffile writes OME for that name
        cache = {}
        from spacemap.viewer.sources import image_info

        info = image_info(str(path))
        shape, dtype = (info.height, info.width), np.dtype(info.dtype)
        with tifffile.TiffFile(str(path)) as tif:

            def factory(level, level_shape, tif=tif):
                import cv2

                for position, index in enumerate(kept):
                    if level and (position, level) in cache:
                        yield _unpack_cached_plane(*cache.pop((position, level)))
                        continue
                    name = names[index]
                    report(f"{path.name}: {name} ({position + 1} of {len(kept)})", (number + (position + 1) / len(kept)) / len(items))
                    plane = read_channel_from_open_tif(tif, index).astype(np.float32)
                    if name in blanks:
                        plane -= read_channel_from_open_tif(tif, names.index(blanks[name])).astype(np.float32)
                    if levels.get(name):
                        plane -= float(levels[name])
                    plane = np.clip(plane, 0, np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else None).astype(dtype)
                    if level == 0:
                        for deeper in range(1, 4):
                            small = cv2.resize(plane, (max(1, shape[1] // 2 ** deeper), max(1, shape[0] // 2 ** deeper)), interpolation=cv2.INTER_AREA)
                            codec, payload, size = _pack_cached_plane(small)
                            cache[(position, deeper)] = (codec, payload, size, small.shape, small.dtype)
                        yield plane
                    else:
                        yield cv2.resize(plane, (level_shape[1], level_shape[0]), interpolation=cv2.INTER_AREA)

            try:
                mpp = get_image_mpp(str(path)).get("codex") or 0.5
                _write_ome_pyramid_from_plane_factory(
                    str(partial), factory, [item["rename"].get(names[i]) or names[i] for i in kept],
                    [item["colors"].get(names[i]) or markers[i]["color"] for i in kept], mpp, shape,
                    [{"dye": markers[i]["dye"], "exposure": markers[i]["exposure"]} for i in kept], n_pyramid_levels=3)
                inherit_ome(str(path), str(partial), kept, notes={
                    "Background removed from": path.name,
                    "Blanks subtracted": "; ".join(f"{m} - {b}" for m, b in blanks.items() if m in [names[i] for i in kept]) or "none",
                    "Levels subtracted": "; ".join(f"{n}: {v:g}" for n, v in levels.items() if n in [names[i] for i in kept]) or "none",
                })
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
        written.append(str(target))
    return written


def class_region(annotation, class_name):
    """The polygons of one class of an annotation GeoJSON, as Shapely shapes (full-resolution pixels)."""
    import json

    from shapely.geometry import shape

    features = json.loads(Path(annotation).read_text()).get("features") or []
    found = []
    for feature in features:
        props = feature.get("properties") or {}
        name = (props.get("classification") or {}).get("name") or props.get("name") or ""
        if (not class_name or str(name) == class_name) and feature.get("geometry"):
            found.append(shape(feature["geometry"]).buffer(0))
    if not found:
        raise ValueError(f"{Path(annotation).name} has no '{class_name}' area.")
    return found


def display_path(root, image):
    """An image's saved marker settings in the project (colours, ranges; the minimum is its background level)."""
    from spacemap.viewer.display import settings_path
    if Path(image).name == str(image):
        from .settings import read_config
        config = read_config(root)
        image = all_folders(root, config.get('directory_mappings', {}), config)['images'] / image
    return settings_path(root, image)


def stored_levels(root, image):
    """The levels kept in the project for an image (its markers' minimums), or {}."""
    from spacemap.viewer.display import load

    return {c["name"]: c["low"] for c in load(display_path(root, image))["channels"] if c["name"] and c["low"] is not None}


def store_levels(root, image_path, levels, method, *, parameters=None, native=False):
    """Keep *levels* in the project as the image's markers' minimums (where the viewer reads them when it opens)."""
    from spacemap.viewer.display import load, save
    from spacemap.viewer.sources import image_info

    path = display_path(root, image_path)
    saved = load(path)
    by_index = {c["index"]: c for c in saved["channels"]}
    info = image_info(str(image_path))
    colours = info.channel_colors or [None] * len(info.channel_names)
    channels = []
    for index, name in enumerate(info.channel_names):
        channel = dict(by_index.get(index) or {"index": index, "name": name, "label": name, "color": colours[index] or "#ffffff",
                                                "low": None, "high": None, "visible": index == 0})
        if name in levels:
            channel["low"] = float(levels[name])
        channels.append(channel)
    save(path, channels, {**saved.get("auto", {}), "method": method, "parameters": parameters, "native": native})


def detect(items, report):
    """Each image's background level per marker (not the blank cycles), with Tab 2's method, on the whole image or
    inside its annotation class; kept in the project as its markers' minimums (Export writes CSVs and images)."""
    from spacemap.viewer.render import background_level
    from spacemap.viewer.sources import image_info

    written = []
    for number, item in enumerate(items):
        path = Path(item["image"])
        info = image_info(str(path))
        region = class_region(item["annotation"], item["class"]) if item.get("annotation") else None
        markers = [(index, name) for index, name in enumerate(info.channel_names) if not BLANK.search(name)]
        levels = {}
        for position, (index, name) in enumerate(markers):
            report(f"{path.name}: {name} ({position + 1} of {len(markers)})", (number + (position + 1) / len(markers)) / len(items))
            levels[name] = background_level(str(path), info, index, item["method"], region=region, params=item.get("parameters"), native=item.get("native", False))["threshold"]
        store_levels(item["root"], path, levels, item["method"], parameters=item.get("parameters"), native=item.get("native", False))
        written.append({"image": path.name, "markers": len(levels), "where": item.get("class") if region is not None else "whole image"})
    return written


def export(items, report):
    """Each image's thresholds as its CSV table and/or its background-removed image (blanks and/or levels)."""
    done = []
    for number, item in enumerate(items):
        name = Path(item["image"]).name
        if item.get("csv"):
            report(f"{name}: thresholds", number / len(items))
            save_levels(Path(item["table"]), item["levels"], item.get("method") or "Studio")
        if item.get("write_image"):
            remove([item], lambda text, fraction, number=number: report(text, (number + fraction) / len(items)))
        done.append(name)
    return done
