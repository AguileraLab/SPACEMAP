"""Studio's Annotation part for pooled compartments and zones (Tab 4 step 4), live on the image.

It starts from the domains exported in Pixel classes (Tab 4's macro_domains
file). Pools join domains into a larger compartment (for example Epithelia and
Stroma into tissue); zones are rings of a set width inside and outside chosen
classes. The preview draws them on a raster of the domains (about 2,400
pixels across) with exact distance transforms, in well under a second; Make
zones writes Tab 4's zones file (in step 4's own folder) with Tab 4's own
``pool_geojson_classes`` and ``generate_zone_features``, which take tens of
seconds on a whole ROI. ``tests/test_studio_pieces_zones.py`` checks the two
agree. The rules are saved where Tab 4 keeps them; widths are shown in
micrometres and kept in pixels.
"""

from collections import OrderedDict
from pathlib import Path
import threading
import time

import numpy as np

from . import annotation

PREVIEW_SIDE = 2400
_rasters = OrderedDict()
_lock = threading.Lock()

KEYS = ("annot_ta_pool_rules", "annot_ta_zone_rules", "annot_ta_pool_keep_sources", "annot_ta_zone_clip_class")
NAME_LIMIT = 60


def _tab4(config):
    return (config.get("tab_configurations") or {}).get("tab4_annotation") or {}


def microns_per_pixel(config):
    tissue = _tab4(config).get("annot_ta_tissue_config")
    return float((tissue if isinstance(tissue, dict) else {}).get("microns_per_pixel", 0.5083))


def values(config):
    """Tab 4's pool and zone rules as step 4 reads them."""
    saved = _tab4(config)
    pools = [
        {
            "name": str(rule.get("name", "")),
            "sources": [str(v) for v in rule.get("sources", [])],
            "subtract": [str(v) for v in rule.get("subtract", [])],
        }
        for rule in saved.get("annot_ta_pool_rules") or []
        if isinstance(rule, dict)
    ]
    zones = [
        {
            "label": str(rule.get("label", "")),
            "width_px": float(rule.get("width_px", 100)),
            "inner": int(rule.get("inner", 0)),
            "outer": int(rule.get("outer", 0)),
        }
        for rule in saved.get("annot_ta_zone_rules") or []
        if isinstance(rule, dict)
    ]
    clip = saved.get("annot_ta_zone_clip_class")
    return {
        "pools": pools,
        "zones": zones,
        "keep_sources": bool(saved.get("annot_ta_pool_keep_sources", True)),
        "clip": None if clip in (None, "", "(none)") else str(clip),
    }


def _name(value, what):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > NAME_LIMIT:
        raise ValueError(f"Give each {what} a name of up to {NAME_LIMIT} characters.")
    return value.strip()


def check(rules):
    """The rules from Studio, checked and in Tab 4's form."""
    if not isinstance(rules, dict) or set(rules) - {"pools", "zones", "keep_sources", "clip"}:
        raise ValueError("Send pools, zones, keep_sources and clip.")
    pools = []
    for rule in rules.get("pools") or []:
        if not isinstance(rule, dict):
            raise ValueError("Each pool is a name and its classes.")
        sources, subtract = rule.get("sources") or [], rule.get("subtract") or []
        if not all(isinstance(v, str) for v in [*sources, *subtract]):
            raise ValueError("Pool classes are names.")
        pools.append({"name": _name(rule.get("name"), "pool"), "sources": sources, "subtract": subtract})
    zones = []
    for rule in rules.get("zones") or []:
        if not isinstance(rule, dict):
            raise ValueError("Each zone rule is a class, a width and ring counts.")
        width, inner, outer = rule.get("width_px"), rule.get("inner", 0), rule.get("outer", 0)
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not 1 <= width <= 100000:
            raise ValueError("Zone widths are between 1 and 100,000 pixels.")
        for count in (inner, outer):
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 20:
                raise ValueError("Use 0 to 20 rings.")
        zones.append(
            {"label": _name(rule.get("label"), "zone class"), "width_px": float(width), "inner": inner, "outer": outer}
        )
    keep, clip = rules.get("keep_sources", True), rules.get("clip")
    if not isinstance(keep, bool) or not (clip is None or isinstance(clip, str)):
        raise ValueError("Choose whether to keep the domains, and a class to clip to (or none).")
    return {"pools": pools, "zones": zones, "keep_sources": keep, "clip": clip or None}


def save(root, rules, expected):
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    rules = check(rules)
    with project_lock(root):
        config = read_config(root)
        if expected != revision(values(config)):
            raise ValueError("These rules changed in another window. Reload them before saving.")
        stored = {
            "annot_ta_pool_rules": rules["pools"],
            "annot_ta_zone_rules": rules["zones"],
            "annot_ta_pool_keep_sources": rules["keep_sources"],
            "annot_ta_zone_clip_class": rules["clip"] or "(none)",
        }
        for key, value in stored.items():
            persistence.update_stored_config_key(config, key, value)
        persistence.save_project_config(str(root), config)
        return values(read_config(root))


def source(output, sample):
    path = annotation.domains(output, sample)
    if not path.is_file():
        raise ValueError("Export the pixel classes first: zones start from the domains.")
    return path


def classes(path):
    from spacemap.tissue_annotation_utils import read_geojson_class_names

    return read_geojson_class_names(str(path))


def build(path, rules, config, target):
    """Tab 4 step 4 from *path* into *target*: the pools, then the zones added (as step 4's run)."""
    from spacemap.tissue_annotation_utils import (
        _iter_geojson_features,
        generate_zone_features,
        pool_geojson_classes,
        write_feature_collection,
    )

    saved = _tab4(config)
    pooled = pool_geojson_classes(
        str(path),
        rules["pools"],
        str(target),
        keep_source_classes=rules["keep_sources"],
        config=saved.get("annot_ta_pool_config") or {},
        class_colors=saved.get("annot_ta_domain_colors") or {},
    )
    active = [rule for rule in rules["zones"] if rule["inner"] > 0 or rule["outer"] > 0]
    zones = []
    if active:
        # Zones are built from the pooled file so a pool can be a zone's class.
        zones = generate_zone_features(
            str(target), active, clip_class=rules["clip"], config=saved.get("annot_ta_pool_config") or {}
        )
        if zones:
            features = _iter_geojson_features(str(target))
            features.extend(zones)
            write_feature_collection(features, str(target))
    return {"pooled": pooled["created"], "zones": zones}


def _masks(path, image_shape):
    """Each class of the domains file on a raster about 2,400 pixels across; cached per file."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    from spacemap.tissue_annotation_utils import _load_class_geometries

    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, tuple(image_shape))
    with _lock:
        if key in _rasters:
            _rasters.move_to_end(key)
            return _rasters[key]
    height, width = image_shape
    scale = max(1.0, max(height, width) / PREVIEW_SIDE)
    shape = (max(1, int(np.ceil(height / scale))), max(1, int(np.ceil(width / scale))))
    transform = Affine(scale, 0, 0, 0, scale, 0)
    masks = {
        name.lower(): rasterize(geometries, out_shape=shape, transform=transform, dtype="uint8").astype(bool)
        for name, geometries in _load_class_geometries(str(path)).items()
    }
    found = {"masks": masks, "scale": scale, "shape": shape}
    with _lock:
        _rasters[key] = found
        while len(_rasters) > 2:
            _rasters.popitem(last=False)
    return found


def raster_layers(path, rules, image_shape):
    """Tab 4 step 4's pools and zones on the raster, by exact distance transforms.

    The same rules as ``pool_geojson_classes`` (sources joined, subtracted classes
    cut out) and ``generate_zone_features`` (ring k of a class is the band from
    (k-1) to k widths from its edge, outer rings cut to the clip class, rings of
    one level joined across classes); returns (name, kind, mask) and the scale.
    """
    import cv2

    raster = _masks(path, image_shape)
    masks, scale = dict(raster["masks"]), raster["scale"]
    empty = np.zeros(raster["shape"], bool)
    layers = []
    for rule in rules["pools"]:
        joined = np.zeros_like(empty)
        for name in rule["sources"]:
            joined |= masks.get(name.lower(), empty)
        for name in rule["subtract"]:
            joined &= ~masks.get(name.lower(), empty)
        if joined.any():
            masks[rule["name"].lower()] = joined
            layers.append((rule["name"], "pool", joined))
    clip = masks.get(str(rules["clip"]).lower()) if rules["clip"] else None
    outer, inner = {}, {}
    for rule in rules["zones"]:
        source = masks.get(rule["label"].lower())
        if source is None or not source.any() or not (rule["inner"] or rule["outer"]):
            continue
        width = rule["width_px"] / scale
        if rule["outer"]:
            away = cv2.distanceTransform((~source).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            for level in range(1, rule["outer"] + 1):
                band = (away > (level - 1) * width) & (away <= level * width) & ~source
                if clip is not None:
                    band &= clip
                outer[level] = outer.get(level, empty) | band
        if rule["inner"]:
            # Beyond the image is outside the class, as the polygon's own edge there is in Tab 4.
            padded = np.pad(source, 1).astype(np.uint8)
            depth = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
            for level in range(1, rule["inner"] + 1):
                band = source & (depth > (level - 1) * width) & (depth <= level * width)
                inner[level] = inner.get(level, empty) | band
    layers += [(f"outer_{level}", "zone", mask) for level, mask in sorted(outer.items()) if mask.any()]
    layers += [(f"inner_{level}", "zone", mask) for level, mask in sorted(inner.items()) if mask.any()]
    return layers, scale


def _outlines(mask, scale, limit=3000):
    """The mask's outlines, holes included, in full-resolution pixels (drawn even-odd)."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for contour in sorted(contours, key=len, reverse=True)[:limit]:
        contour = cv2.approxPolyDP(contour, 0.8, True)
        if len(contour) >= 3:
            rings.append([[round((x + 0.5) * scale, 1), round((y + 0.5) * scale, 1)] for x, y in contour[:, 0, :]])
    return rings


def preview(output, sample, config, rules, image_shape):
    """The pools and zones for *rules*, drawn on a raster of the domains; nothing is written.

    Tab 4's vector functions take tens of seconds on a whole ROI's domains, so
    the preview uses the raster; Make zones writes the file with Tab 4's own.
    """
    from spacemap.tissue_annotation_utils import deterministic_class_hex_color

    rules = check(rules)
    path = source(output, sample)
    started = time.perf_counter()
    layers, scale = raster_layers(path, rules, image_shape)
    colours = _tab4(config).get("annot_ta_domain_colors") or {}
    drawn = [
        {
            "name": name,
            "kind": kind,
            "colour": colours.get(name) or deterministic_class_hex_color(name),
            "rings": _outlines(mask, scale),
            "area_px": round(float(mask.sum()) * scale * scale),
        }
        for name, kind, mask in layers
    ]
    return {
        "layers": drawn,
        "classes": classes(path),
        "source": path.name,
        "ms": round(1000 * (time.perf_counter() - started)),
        "microns_per_pixel": microns_per_pixel(config),
    }


def apply(output, folder, sample, config):
    """Tab 4 step 4 for one image with the saved rules: its zones file, in step 4's folder."""
    rules = values(config)
    path = source(output, sample)
    Path(folder).mkdir(parents=True, exist_ok=True)
    target = annotation.zones(folder, sample)
    made = build(path, rules, config, target)
    return {
        "sample": sample,
        "pooled": [item["name"] for item in made["pooled"] if item.get("status") == "created"],
        "zones": len(made["zones"]),
        "files": [str(target)],
    }
