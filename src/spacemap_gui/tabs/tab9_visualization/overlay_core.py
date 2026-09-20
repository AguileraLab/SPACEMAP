"""Pure rendering helpers shared by the cell-overlay and actual-tissue panels.

Adapted from ``test_codes/render_segmentation_overlays.py`` and generalised for
real project data, which differs from the reference script in two ways that
matter:

* Images are a single ``CYX`` OME series carrying an internal pyramid, not one
  series per channel, so the pyramid is read from ``series[0].levels``.
* There are no ``HE_*`` channels in a CODEX-only run, so the base layer has to
  fall back to composited markers (or to nothing at all).

Nothing here imports Streamlit: these are ordinary functions so they can be
unit-tested and reused by any panel.
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".qptiff")
GEOJSON_SUFFIXES = (".geojson", ".json")

# Suffixes appended by producers to the shared ROI key. Ordered longest-first so
# that "reg001_A.ome_cell_boundaries" reduces to "reg001_A" rather than
# "reg001_A.ome".
_KEY_STRIP_SUFFIXES = (
    "_cell_boundaries",
    "_cell_segmentation",
    "_boundaries",
    "_segmentation",
    "_merged",
    "_valid_tissue",
    "_tissue_boundary",
    "_tissue",
    "_annotations",
    "_annotation",
)


def roi_key(path_or_name: str | Path) -> str:
    """Return the ROI key shared by an image, its cells and its annotation."""
    name = Path(str(path_or_name)).name
    lowered = name.lower()
    for suffix in (*IMAGE_SUFFIXES, *GEOJSON_SUFFIXES):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            lowered = name.lower()
            break
    changed = True
    while changed:
        changed = False
        for suffix in _KEY_STRIP_SUFFIXES:
            if lowered.endswith(suffix):
                name = name[: -len(suffix)]
                lowered = name.lower()
                changed = True
        if lowered.endswith(".ome"):
            name = name[:-4]
            lowered = name.lower()
            changed = True
    return name


def _list_by_key(directory: str | Path | None, suffixes: Sequence[str]) -> dict[str, Path]:
    if not directory:
        return {}
    root = Path(directory)
    if not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if not path.name.lower().endswith(tuple(suffixes)):
            continue
        found.setdefault(roi_key(path), path)
    return found


def discover_overlay_sets(
    roi_dir: str | Path | None,
    seg_dir: str | Path | None,
    ann_dir: str | Path | None,
) -> list[dict[str, Any]]:
    """Pair image / cell / annotation files by ROI key.

    Every key found in any directory is reported, including partial matches, so
    the panel can say plainly which ROIs have all three layers and which have
    only one or two -- a figure is still worth rendering from cells alone.
    """
    images = _list_by_key(roi_dir, IMAGE_SUFFIXES)
    cells = _list_by_key(seg_dir, GEOJSON_SUFFIXES)
    annots = _list_by_key(ann_dir, GEOJSON_SUFFIXES)

    records: list[dict[str, Any]] = []
    for key in sorted(set(images) | set(cells) | set(annots)):
        image_path = images.get(key)
        cell_path = cells.get(key)
        ann_path = annots.get(key)
        present = sum(p is not None for p in (image_path, cell_path, ann_path))
        records.append(
            {
                "roi": key,
                "image": str(image_path) if image_path else "",
                "cells": str(cell_path) if cell_path else "",
                "tissue": str(ann_path) if ann_path else "",
                "has_image": image_path is not None,
                "has_cells": cell_path is not None,
                "has_tissue": ann_path is not None,
                "n_layers": present,
                "status": {3: "complete", 2: "partial", 1: "single"}.get(present, "none"),
            }
        )
    return records


# ── GeoJSON ──────────────────────────────────────────────────────────────────

def load_geojson(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):  # some writers emit a bare feature array
        return {"type": "FeatureCollection", "features": data}
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError(f"Expected a GeoJSON FeatureCollection: {Path(path).name}")
    return data


def feature_name(properties: dict[str, Any]) -> str:
    classification = properties.get("classification")
    if isinstance(classification, dict) and classification.get("name") is not None:
        return str(classification["name"])
    for key in ("name", "label", "class", "classification"):
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def feature_color(properties: dict[str, Any], fallback: str = "#00D9FF") -> str:
    """Hex colour carried by a QuPath-style annotation, if present.

    QuPath writes ``classification.color`` as an ``[R, G, B]`` list, so honouring
    it means the figure reproduces the colours the annotator actually chose
    instead of inventing new ones.
    """
    classification = properties.get("classification")
    for source in (classification if isinstance(classification, dict) else {}, properties):
        if not isinstance(source, dict):
            continue
        color = source.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            try:
                r, g, b = (max(0, min(255, int(round(float(c))))) for c in color[:3])
                return f"#{r:02X}{g:02X}{b:02X}"
            except Exception:
                pass
        if isinstance(color, str) and color.strip():
            try:
                return "#%02X%02X%02X" % parse_hex_color(color)
            except Exception:
                pass
        # QuPath also writes a packed signed 32-bit ARGB integer; masking to the
        # low 24 bits drops the alpha byte and leaves RGB.
        packed = source.get("colorRGB")
        if isinstance(packed, (int, float)) and not isinstance(packed, bool):
            try:
                value = int(packed) & 0xFFFFFF
                return f"#{(value >> 16) & 0xFF:02X}{(value >> 8) & 0xFF:02X}{value & 0xFF:02X}"
            except Exception:
                pass
    return fallback


def annotation_classes(path: str | Path) -> list[dict[str, Any]]:
    """Per-class summary of a tissue GeoJSON: name, colour and polygon counts.

    Drives the "which polygons to display" picker. Reported per class rather
    than per feature because a single class is routinely one MultiPolygon
    holding many disjoint pieces -- a TMA core's background can be 20+ polygons.
    """
    try:
        data = load_geojson(path)
    except Exception:
        return []
    summary: dict[str, dict[str, Any]] = {}
    for feature in data.get("features", []):
        properties = feature.get("properties", {}) or {}
        name = feature_name(properties).strip() or "(unnamed)"
        entry = summary.setdefault(
            name,
            {"name": name, "color": feature_color(properties, ""), "n_features": 0, "n_polygons": 0},
        )
        entry["n_features"] += 1
        entry["n_polygons"] += len(list(iter_polygons(feature.get("geometry"))))

    ordered = sorted(summary.values(), key=lambda item: item["name"].lower())
    # QuPath writes colorRGB: null for classes the annotator never styled.
    # Giving all of them the same fallback would render five different tissue
    # compartments identically, so uncoloured classes get distinct colours
    # chosen away from whichever classes do carry one.
    assigned = [entry["color"] for entry in ordered if entry["color"]]
    missing = [entry for entry in ordered if not entry["color"]]
    if missing:
        for entry, colour in zip(missing, _distinct_colors(len(missing), assigned)):
            entry["color"] = colour
    return ordered


def _distinct_colors(n: int, avoid: Sequence[str]) -> list[str]:
    """Colours unlike ``avoid`` and each other, shared with the phenotype maps."""
    try:
        from spacemap.acc_utils import distinct_colors as _house

        return _house(n, avoid=list(avoid))
    except Exception:
        pool = [c for c in DEFAULT_MARKER_COLORS if c not in set(avoid)]
        return [pool[i % len(pool)] for i in range(n)] if pool else ["#00D9FF"] * n


def features_for_classes(
    path: str | Path,
    class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Annotation features restricted to the selected class names."""
    try:
        data = load_geojson(path)
    except Exception:
        return []
    if not class_names:
        return list(data.get("features", []))
    wanted = {str(name).strip().casefold() for name in class_names}
    return [
        feature
        for feature in data.get("features", [])
        if feature_name(feature.get("properties", {}) or {}).strip().casefold() in wanted
    ]


def annotation_class_names(path: str | Path) -> list[str]:
    """Distinct annotation class names present in a tissue GeoJSON."""
    try:
        data = load_geojson(path)
    except Exception:
        return []
    names = {
        feature_name(feature.get("properties", {}) or {}).strip()
        for feature in data.get("features", [])
    }
    return sorted(name for name in names if name)


def iter_polygons(geometry: dict[str, Any] | None) -> Iterator[list[list[list[float]]]]:
    if not geometry:
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        yield coordinates
    elif geometry_type == "MultiPolygon":
        yield from coordinates
    elif geometry_type == "GeometryCollection":
        for member in geometry.get("geometries", []):
            yield from iter_polygons(member)


def transform_ring(
    ring: Sequence[Sequence[float]],
    scale_x: float,
    scale_y: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[tuple[float, float]]:
    """Map full-resolution ring coordinates into the rendered image."""
    # Keep sub-pixel precision. Rounding each vertex to the nearest device pixel
    # costs up to 0.5px at the rendered scale, which is 4 full-resolution pixels
    # at a 1/8 pyramid level -- enough to visibly shift an overlay against the
    # image in a low-resolution preview while the full-size export looks fine.
    # PIL draws polygons from float coordinates, so there is nothing to gain by
    # quantising here.
    points = [
        (
            (float(point[0]) - offset_x) * scale_x,
            (float(point[1]) - offset_y) * scale_y,
        )
        for point in ring
    ]
    # Drop only genuinely coincident vertices. The old integer comparison merged
    # points that were merely close, which at a low pyramid level collapsed an
    # 11-vertex cell to 2 vertices -- a degenerate line rather than a cell.
    epsilon = 1e-6
    deduplicated: list[tuple[float, float]] = []
    for point in points:
        if not deduplicated or (
            abs(point[0] - deduplicated[-1][0]) > epsilon
            or abs(point[1] - deduplicated[-1][1]) > epsilon
        ):
            deduplicated.append(point)
    if len(deduplicated) > 1 and (
        abs(deduplicated[0][0] - deduplicated[-1][0]) <= epsilon
        and abs(deduplicated[0][1] - deduplicated[-1][1]) <= epsilon
    ):
        deduplicated.pop()
    return deduplicated


def draw_geometry(
    draw: ImageDraw.ImageDraw,
    geometry: dict[str, Any] | None,
    scale_x: float,
    scale_y: float,
    fill: tuple[int, int, int, int] | None,
    outline: tuple[int, int, int, int] | None,
    width: int = 1,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    hole_fill: Any = (0, 0, 0, 0),
) -> int:
    """Draw all polygons of a geometry; returns how many were drawn."""
    drawn = 0
    for polygon in iter_polygons(geometry):
        if not polygon:
            continue
        exterior = transform_ring(polygon[0], scale_x, scale_y, offset_x, offset_y)
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=fill, outline=outline, width=width)
            drawn += 1
        for hole in polygon[1:]:
            interior = transform_ring(hole, scale_x, scale_y, offset_x, offset_y)
            if len(interior) >= 3:
                # The hole fill must match the target image mode: an RGBA layer
                # wants a 4-tuple to clear alpha, while a single-band "L" mask
                # (used for outside-annotation masking) only accepts an int.
                draw.polygon(interior, fill=hole_fill)
    return drawn


def geometry_bounds(geometry: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    """Full-resolution (minx, miny, maxx, maxy) of a geometry, or None."""
    xs: list[float] = []
    ys: list[float] = []
    for polygon in iter_polygons(geometry):
        for ring in polygon:
            for point in ring:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def clip_cells_to_containing_tissue(
    cell_features: Sequence[dict[str, Any]],
    tissue_features: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clip every cell to the tissue annotation containing its centroid.

    Assignment and clipping happen in the shared level-0 coordinate system,
    before any preview scaling. If annotations overlap, the annotation covering
    the greatest portion of the cell wins. Cells whose centroids are outside all
    selected annotations are omitted.
    """
    from shapely.geometry import mapping, shape
    from shapely.strtree import STRtree

    stats = {
        "input": len(cell_features),
        "assigned": 0,
        "geometry_clipped": 0,
        "outside": 0,
    }
    indexed: list[tuple[dict[str, Any], Any]] = []
    for tissue_feature in tissue_features:
        try:
            geometry = shape(tissue_feature.get("geometry"))
        except Exception:
            continue
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if not geometry.is_empty:
            indexed.append((tissue_feature, geometry))

    if not indexed:
        return list(cell_features), stats

    tissue_geometries = [entry[1] for entry in indexed]
    tree = STRtree(tissue_geometries)
    geometry_indices = {id(geometry): index for index, geometry in enumerate(tissue_geometries)}
    clipped_features: list[dict[str, Any]] = []

    for cell_feature in cell_features:
        try:
            cell_geometry = shape(cell_feature.get("geometry"))
        except Exception:
            stats["outside"] += 1
            continue
        if cell_geometry.is_empty:
            stats["outside"] += 1
            continue
        if not cell_geometry.is_valid:
            cell_geometry = cell_geometry.buffer(0)
        if cell_geometry.is_empty:
            stats["outside"] += 1
            continue

        centroid = cell_geometry.centroid
        raw_candidates = list(tree.query(centroid))
        candidate_indices: list[int] = []
        for candidate in raw_candidates:
            if isinstance(candidate, (int, np.integer)):
                candidate_indices.append(int(candidate))
            else:  # Shapely < 2 returns geometry objects instead of indices.
                candidate_index = geometry_indices.get(id(candidate))
                if candidate_index is not None:
                    candidate_indices.append(candidate_index)

        containing = [
            index for index in candidate_indices if tissue_geometries[index].covers(centroid)
        ]
        if not containing:
            stats["outside"] += 1
            continue

        # Overlapping annotation classes are resolved by the largest shared cell
        # area, making assignment deterministic at nested or touching borders.
        tissue_index = max(
            containing,
            key=lambda index: cell_geometry.intersection(tissue_geometries[index]).area,
        )
        clipped_geometry = cell_geometry.intersection(tissue_geometries[tissue_index])
        if clipped_geometry.is_empty:
            stats["outside"] += 1
            continue

        tissue_feature, _tissue_geometry = indexed[tissue_index]
        updated = dict(cell_feature)
        updated["geometry"] = dict(mapping(clipped_geometry))
        updated_properties = dict(cell_feature.get("properties", {}) or {})
        updated_properties["tissue_annotation"] = feature_name(
            tissue_feature.get("properties", {}) or {}
        )
        updated["properties"] = updated_properties
        clipped_features.append(updated)
        stats["assigned"] += 1
        if not clipped_geometry.equals(cell_geometry):
            stats["geometry_clipped"] += 1

    return clipped_features, stats


# ── Colour ───────────────────────────────────────────────────────────────────

def parse_hex_color(value: str) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"Color must be #RRGGBB, got {value!r}.")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def cell_color(label: Any, fallback_index: int = 0) -> tuple[int, int, int]:
    """Bright, stable per-cell colour; neighbouring labels get distinct hues."""
    try:
        numeric = int(label)
    except (TypeError, ValueError):
        numeric = fallback_index
    # Golden-ratio hue stepping keeps adjacent labels far apart in hue.
    hue = (numeric * 0.618033988749895) % 1.0
    saturation = 0.68 + 0.20 * ((numeric * 37) % 11) / 10.0
    value = 0.92 + 0.08 * ((numeric * 53) % 7) / 6.0
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def darken(rgb: Sequence[int], factor: float = 0.62) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in rgb)  # type: ignore[return-value]


# ── Base image ───────────────────────────────────────────────────────────────

HE_CHANNEL_SETS = (("HE_Red", "HE_Green", "HE_Blue"), ("R", "G", "B"))


def _channel_names_from_tiff(handle: Any) -> list[str]:
    """Channel names from an open TIFF handle."""
    import xml.etree.ElementTree as ET

    namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    metadata = handle.ome_metadata
    n_channels = 0
    series = handle.series[0]
    axes = getattr(series, "axes", "") or ""
    if "C" in axes:
        n_channels = int(series.shape[axes.index("C")])
    if metadata:
        try:
            root = ET.fromstring(metadata)
            channels = root.findall(".//ome:Channel", namespace) or root.findall(".//Channel")
            names = [
                channel.get("Name") or f"Channel_{index}"
                for index, channel in enumerate(channels)
            ]
            if names:
                return names
        except Exception:
            pass
    return [f"Channel_{index}" for index in range(max(n_channels, 1))]


def read_ome_channel_names(path: str | Path) -> list[str]:
    """Channel names from OME metadata, falling back to positional names."""
    import tifffile

    with tifffile.TiffFile(str(path)) as handle:
        return _channel_names_from_tiff(handle)


def _series_hw(series: Any) -> tuple[int, int]:
    axes = getattr(series, "axes", "") or ""
    shape = series.shape
    if "Y" in axes and "X" in axes:
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])
    return int(shape[-2]), int(shape[-1])


def _layout_from_tiff(handle: Any) -> dict[str, Any]:
    """Work out how an open pyramidal OME-TIFF stores channels and levels.

    Two layouts occur in this pipeline and they are not interchangeable:

    * ``cyx``  - one series with a C axis carrying an internal pyramid
      (``series[0].levels``).
    * ``per_channel`` - one series per (channel, level), each a plain YX plane,
      so ``series[channel * n_levels + level]`` addresses a plane. This is what
      the registration step writes for large H&E + CODEX ROIs.
    """
    series_list = list(handle.series)
    first = series_list[0]
    axes = getattr(first, "axes", "") or ""

    if "C" in axes:
        levels = list(getattr(first, "levels", []) or [first])
        return {
            "layout": "cyx",
            "n_channels": int(first.shape[axes.index("C")]),
            "levels_per_channel": len(levels),
            "level_shapes": [_series_hw(level) for level in levels],
        }

    # Per-channel layout: series repeat as a decreasing pyramid per channel,
    # so the run length before the size resets is the level count.
    shapes = [_series_hw(s) for s in series_list]
    levels_per_channel = 1
    for index in range(1, len(shapes)):
        if shapes[index][0] >= shapes[index - 1][0]:
            break
        levels_per_channel += 1
    return {
        "layout": "per_channel",
        "n_channels": max(1, len(series_list) // max(levels_per_channel, 1)),
        "levels_per_channel": levels_per_channel,
        "level_shapes": shapes[:levels_per_channel],
    }


def describe_image_layout(path: str | Path) -> dict[str, Any]:
    """Work out how a pyramidal OME-TIFF stores its channels and levels."""
    import tifffile

    with tifffile.TiffFile(str(path)) as handle:
        return _layout_from_tiff(handle)


def pyramid_level_shapes(path: str | Path) -> list[tuple[int, int]]:
    """(height, width) of each pyramid level, for either storage layout."""
    return describe_image_layout(path)["level_shapes"]


def choose_pyramid_level(shapes: Sequence[tuple[int, int]], max_size: int) -> int:
    """Highest-resolution level whose longest side fits ``max_size``."""
    for level, shape in enumerate(shapes):
        if max(shape) <= int(max_size):
            return level
    return max(len(shapes) - 1, 0)


def _to_uint8(plane: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Contrast-stretch one plane to uint8 for display."""
    plane = np.squeeze(np.asarray(plane)).astype(np.float32)
    finite = np.isfinite(plane)
    if not finite.any():
        return np.zeros(plane.shape, dtype=np.uint8)
    low = float(np.percentile(plane[finite], low_pct))
    high = float(np.percentile(plane[finite], high_pct))
    if not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(plane)), float(np.nanmax(plane))
    if high <= low:
        return np.zeros(plane.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(plane) - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def load_base_image(
    image_path: str | Path,
    channels: Sequence[str] | None = None,
    channel_colors: dict[str, str] | None = None,
    level: int | None = None,
    max_size: int = 2000,
    low_pct: float = 1.0,
    high_pct: float = 99.5,
    require_he: bool = False,
) -> tuple[Image.Image, tuple[int, int], int]:
    """Composite selected channels into an RGB base image.

    Returns ``(image, (full_height, full_width), level_used)``. ``full_*`` is the
    level-0 size, which is the coordinate space cell and annotation GeoJSONs use,
    so callers scale overlays by ``image.size / full_size``.

    When the file carries H&E channels and none are explicitly requested, the
    H&E triplet is used directly as RGB. Otherwise the named marker channels are
    each tinted and screened together, which is the only option for a CODEX-only
    acquisition.
    """
    import tifffile

    with tifffile.TiffFile(str(image_path)) as handle:
        # Read metadata, pyramid layout, and the requested planes through one
        # handle. Reopening a multi-gigabyte TIFF on network storage several
        # times made small previews unnecessarily slow.
        names = _channel_names_from_tiff(handle)
        layout = _layout_from_tiff(handle)
        shapes = layout["level_shapes"]
        if not shapes:
            raise ValueError(f"No image data found in {Path(image_path).name}")
        full_shape = shapes[0]
        use_level = int(level) if level is not None else choose_pyramid_level(shapes, max_size)
        use_level = max(0, min(use_level, len(shapes) - 1))

        requested = [str(c) for c in (channels or []) if str(c)]
        he_set = next((s for s in HE_CHANNEL_SETS if all(n in names for n in s)), None)
        if require_he and he_set is None:
            raise ValueError(
                f"{Path(image_path).name} does not contain HE_Red, HE_Green, and HE_Blue channels."
            )
        use_he = bool(he_set) and not requested

        # Read only the planes actually needed. In the per-channel layout each
        # (channel, level) is its own series, so pulling one marker does not
        # decode the other forty.
        wanted_names = list(he_set) if (use_he and he_set) else list(requested or names[:1])
        planes: dict[str, np.ndarray] = {}
        if layout["layout"] == "per_channel":
            per = int(layout.get("levels_per_channel", 1))
            for name in wanted_names:
                if name not in names:
                    continue
                index = names.index(name)
                series_index = index * per + use_level
                if series_index < len(handle.series):
                    planes[name] = np.asarray(handle.series[series_index].asarray())
        else:
            series = handle.series[0]
            levels = list(getattr(series, "levels", []) or [series])
            data = np.asarray(levels[use_level].asarray())
            axes = getattr(levels[use_level], "axes", "") or ""
            if "C" in axes and data.ndim == 3:
                data = np.moveaxis(data, axes.index("C"), 0)
            elif data.ndim == 2:
                data = data[None, ...]
            for name in wanted_names:
                if name in names:
                    index = names.index(name)
                    if index < data.shape[0]:
                        planes[name] = data[index]

    def plane_for(name: str) -> np.ndarray | None:
        return planes.get(name)

    if use_he and he_set is not None:
        stack = [_to_uint8(plane_for(name), low_pct, high_pct) for name in he_set]
        rgb = np.stack(stack, axis=-1)
    else:
        target = requested or names[:1]
        first_plane = next((p for p in planes.values() if p is not None), None)
        if first_plane is None:
            raise ValueError(
                f"None of the requested channels were found in {Path(image_path).name}."
            )
        height, width = first_plane.shape[-2], first_plane.shape[-1]
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        palette = channel_colors or {}
        for position, name in enumerate(target):
            plane = plane_for(name)
            if plane is None:
                continue
            gray = _to_uint8(plane, low_pct, high_pct).astype(np.float32) / 255.0
            tint = parse_hex_color(palette.get(name, DEFAULT_MARKER_COLORS[position % len(DEFAULT_MARKER_COLORS)]))
            for channel_index in range(3):
                # Screen blending keeps overlapping markers visible instead of
                # letting the last one drawn hide the others.
                contribution = gray * (tint[channel_index] / 255.0)
                rgb[..., channel_index] = 1.0 - (1.0 - rgb[..., channel_index]) * (1.0 - contribution)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    if max(image.size) > int(max_size):
        ratio = int(max_size) / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return image, full_shape, use_level


DEFAULT_MARKER_COLORS = ("#00D9FF", "#FF5B40", "#7CFF6B", "#FFD166", "#C084FC", "#FF7AC6")


def crop_base_to_fullres_box(
    image: Image.Image,
    full_shape: tuple[int, int],
    crop_box: tuple[float, float, float, float],
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Crop a rendered ROI using a box expressed in level-0 coordinates.

    Cell and annotation GeoJSON coordinates refer to the full-resolution ROI,
    while ``image`` may be a pyramid level or an additionally resized preview.
    Cropping therefore has to use the image-to-level-0 scale on each axis.

    The returned box is the *effective* level-0 crop represented by the integer
    pixel bounds.  Passing that box to :func:`render_overlay` keeps its scale and
    origin exactly consistent with the cropped pixels, including at pyramid
    levels where a requested boundary falls between two preview pixels.
    """
    full_h, full_w = (float(full_shape[0]), float(full_shape[1]))
    if full_h <= 0 or full_w <= 0 or image.width <= 0 or image.height <= 0:
        raise ValueError("Image and full-resolution dimensions must be positive.")

    min_x, min_y, max_x, max_y = (float(value) for value in crop_box)
    min_x = min(max(min_x, 0.0), full_w)
    max_x = min(max(max_x, 0.0), full_w)
    min_y = min(max(min_y, 0.0), full_h)
    max_y = min(max(max_y, 0.0), full_h)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Crop box does not overlap the ROI image.")

    scale_x = image.width / full_w
    scale_y = image.height / full_h

    # Use floor/ceil so the selected full-resolution geometry is not clipped at
    # a sub-pixel boundary in a downsampled preview.
    left = max(0, min(image.width - 1, int(np.floor(min_x * scale_x))))
    top = max(0, min(image.height - 1, int(np.floor(min_y * scale_y))))
    right = max(left + 1, min(image.width, int(np.ceil(max_x * scale_x))))
    bottom = max(top + 1, min(image.height, int(np.ceil(max_y * scale_y))))

    effective_box = (
        left / scale_x,
        top / scale_y,
        right / scale_x,
        bottom / scale_y,
    )
    return image.crop((left, top, right, bottom)), effective_box


# ── Composite renderer ───────────────────────────────────────────────────────

def render_overlay(
    *,
    base: Image.Image | None,
    canvas_size: tuple[int, int] | None = None,
    background: str = "#000000",
    full_shape: tuple[int, int] | None = None,
    crop_box: tuple[float, float, float, float] | None = None,
    tissue_features: Sequence[dict[str, Any]] = (),
    tissue_color: str = "#00D9FF",
    tissue_color_lookup: dict[str, str] | None = None,
    tissue_alpha: int = 55,
    tissue_outline_alpha: int = 205,
    tissue_outline_width: int = 2,
    cell_features: Sequence[dict[str, Any]] = (),
    cell_alpha: int = 170,
    cell_outline_alpha: int = 245,
    cell_color_mode: str = "unique",
    cell_fixed_color: str = "#FF5B40",
    cell_color_lookup: dict[str, str] | None = None,
    cell_label_key: str = "label",
    cell_explicit_colors: Sequence[tuple[int, int, int]] | None = None,
    layer_order: Sequence[str] = ("tissue", "cells"),
) -> tuple[Image.Image, dict[str, int]]:
    """Compose base image + tissue polygons + cell polygons into one RGBA figure.

    Coordinates in ``tissue_features`` and ``cell_features`` are full-resolution
    GeoJSON coordinates; they are scaled to the rendered canvas using
    ``full_shape``, and shifted first when ``crop_box`` restricts the view.

    ``base`` may be None, in which case a flat ``background`` canvas of
    ``canvas_size`` is used -- that is what makes a figure possible from cell
    outlines alone when no ROI image exists.
    """
    if base is None:
        if not canvas_size:
            raise ValueError("canvas_size is required when no base image is supplied.")
        base = Image.new("RGB", canvas_size, parse_hex_color(background))
    canvas = base.convert("RGBA")

    if full_shape:
        full_h, full_w = float(full_shape[0]), float(full_shape[1])
    else:
        full_h, full_w = float(canvas.height), float(canvas.width)

    if crop_box:
        min_x, min_y, max_x, max_y = (float(v) for v in crop_box)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        offset_x, offset_y = min_x, min_y
        scale_x = canvas.width / span_x
        scale_y = canvas.height / span_y
    else:
        offset_x = offset_y = 0.0
        scale_x = canvas.width / max(full_w, 1e-6)
        scale_y = canvas.height / max(full_h, 1e-6)

    stats = {"tissue_polygons": 0, "cell_polygons": 0, "cells_considered": 0, "cells_skipped_offscreen": 0}
    order = [str(name).lower() for name in (layer_order or ("tissue", "cells"))]
    for required in ("tissue", "cells"):
        if required not in order:
            order.append(required)

    def _visible(geometry: dict[str, Any] | None) -> bool:
        """Cheap reject for features entirely outside a crop window."""
        if not crop_box:
            return True
        bounds = geometry_bounds(geometry)
        if bounds is None:
            return False
        bminx, bminy, bmaxx, bmaxy = bounds
        return not (bmaxx < min_x or bminx > max_x or bmaxy < min_y or bminy > max_y)

    def _draw_tissue() -> None:
        nonlocal canvas
        if not tissue_features:
            return
        default_rgb = parse_hex_color(tissue_color)
        lookup = {str(k).strip().casefold(): v for k, v in (tissue_color_lookup or {}).items()}
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        # Default draw mode replaces alpha, so interior rings genuinely punch
        # holes rather than compositing a second translucent fill over them.
        draw = ImageDraw.Draw(layer)
        for feature in tissue_features:
            if not _visible(feature.get("geometry")):
                continue
            # Each annotation class can carry its own colour, so a figure can
            # distinguish tumour from stroma rather than tinting both alike.
            name = feature_name(feature.get("properties", {}) or {}).strip().casefold()
            chosen = lookup.get(name)
            rgb = parse_hex_color(chosen) if chosen else default_rgb
            stats["tissue_polygons"] += draw_geometry(
                draw,
                feature.get("geometry"),
                scale_x,
                scale_y,
                fill=(*rgb, int(tissue_alpha)),
                outline=(*rgb, int(tissue_outline_alpha)),
                width=int(max(1, tissue_outline_width)),
                offset_x=offset_x,
                offset_y=offset_y,
            )
        canvas = Image.alpha_composite(canvas, layer)

    def _draw_cells() -> None:
        nonlocal canvas
        if not cell_features:
            return
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        lookup = cell_color_lookup or {}
        fixed = parse_hex_color(cell_fixed_color)
        for index, feature in enumerate(cell_features):
            stats["cells_considered"] += 1
            if not _visible(feature.get("geometry")):
                stats["cells_skipped_offscreen"] += 1
                continue
            properties = feature.get("properties", {}) or {}
            if cell_explicit_colors is not None and index < len(cell_explicit_colors):
                rgb = tuple(cell_explicit_colors[index])
            elif cell_color_mode == "unique":
                rgb = cell_color(properties.get(cell_label_key), index)
            elif cell_color_mode == "lookup":
                key = str(properties.get(cell_label_key, ""))
                rgb = parse_hex_color(lookup.get(key, cell_fixed_color))
            else:
                rgb = fixed
            stats["cell_polygons"] += draw_geometry(
                draw,
                feature.get("geometry"),
                scale_x,
                scale_y,
                fill=(*rgb, int(cell_alpha)),
                outline=(*darken(rgb), int(cell_outline_alpha)),
                width=1,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        canvas = Image.alpha_composite(canvas, layer)

    # Draw order decides what sits on top. The default puts the image at the
    # back, annotations over it and cells in front, which is what a reader
    # expects; reversing it lets a translucent annotation tint the cells instead.
    for name in order:
        if name == "tissue":
            _draw_tissue()
        elif name == "cells":
            _draw_cells()

    return canvas, stats


def mask_outside_geometry(
    image: Image.Image,
    geometries: Sequence[dict[str, Any]],
    full_shape: tuple[int, int],
    crop_box: tuple[float, float, float, float] | None = None,
    dim_alpha: int = 255,
    dim_color: str = "#000000",
) -> Image.Image:
    """Hide everything outside the given annotation polygons.

    Used to crop cells to a tissue region even when the tissue outline itself is
    not drawn, so a figure can show one connected tissue piece in isolation.
    """
    if not geometries:
        return image
    if crop_box:
        min_x, min_y, max_x, max_y = (float(v) for v in crop_box)
        scale_x = image.width / max(max_x - min_x, 1e-6)
        scale_y = image.height / max(max_y - min_y, 1e-6)
        offset_x, offset_y = min_x, min_y
    else:
        scale_x = image.width / max(float(full_shape[1]), 1e-6)
        scale_y = image.height / max(float(full_shape[0]), 1e-6)
        offset_x = offset_y = 0.0

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for geometry in geometries:
        draw_geometry(draw, geometry, scale_x, scale_y, fill=255, outline=255,
                      offset_x=offset_x, offset_y=offset_y, hole_fill=0)
    veil = Image.new("RGBA", image.size, (*parse_hex_color(dim_color), int(dim_alpha)))
    return Image.composite(image.convert("RGBA"), Image.alpha_composite(image.convert("RGBA"), veil), mask)


# ── Cell-geometry sources ────────────────────────────────────────────────────

def shapely_to_geojson(geom: Any) -> dict[str, Any] | None:
    """Convert a shapely geometry to a GeoJSON geometry mapping."""
    if geom is None:
        return None
    try:
        from shapely.geometry import mapping

        if getattr(geom, "is_empty", False):
            return None
        return dict(mapping(geom))
    except Exception:
        return None


def features_from_voronoi(
    cells: Sequence[dict[str, Any]],
    label_key: str = "cell_id",
    property_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Adapt ``actual_tissue._generate_voronoi_cells`` output to render features."""
    features: list[dict[str, Any]] = []
    for cell in cells:
        geometry = shapely_to_geojson(cell.get("geometry"))
        if geometry is None:
            continue
        properties: dict[str, Any] = {"label": cell.get(label_key, cell.get("cell_id"))}
        csv_data = cell.get("csv_data") or {}
        for column in property_columns:
            if column in csv_data:
                properties[column] = csv_data[column]
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return features


def features_from_circles(
    frame: "Any",
    x_col: str,
    y_col: str,
    radius: float,
    label_col: str | None = None,
    n_vertices: int = 16,
    property_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Approximate each cell as a polygon circle of ``radius`` full-res pixels.

    The simplest representation, and the only one available when there is no
    segmentation and too few cells for a meaningful Voronoi tessellation.
    """
    import numpy as _np

    features: list[dict[str, Any]] = []
    angles = _np.linspace(0.0, 2.0 * _np.pi, int(max(6, n_vertices)), endpoint=False)
    unit = _np.stack([_np.cos(angles), _np.sin(angles)], axis=1)
    for position, (_, row) in enumerate(frame.iterrows()):
        try:
            cx = float(row[x_col])
            cy = float(row[y_col])
        except Exception:
            continue
        if not (_np.isfinite(cx) and _np.isfinite(cy)):
            continue
        ring = [[float(cx + px * radius), float(cy + py * radius)] for px, py in unit]
        ring.append(ring[0])
        properties: dict[str, Any] = {
            "label": row[label_col] if label_col and label_col in row else position
        }
        for column in property_columns:
            if column in row:
                properties[column] = row[column]
        features.append(
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
             "properties": properties}
        )
    return features


def build_color_lookup(values: Iterable[Any], palette: Sequence[str] | None = None) -> dict[str, str]:
    """Stable category -> hex colour map for phenotype-coloured cells."""
    colors = list(palette or (
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#2D6CDF", "#C43D3D",
    ))
    lookup: dict[str, str] = {}
    for index, value in enumerate(sorted({str(v) for v in values if str(v) and str(v) != "nan"})):
        lookup[value] = colors[index % len(colors)]
    return lookup


def default_marker_selection(channel_names: Sequence[str]) -> list[str]:
    """Markers pre-selected for a new figure: DAPI, Vimentin and any keratin.

    Keratins are matched fuzzily and case-insensitively because panels name them
    inconsistently -- ``Cytokeratin``, ``Pan-Keratin``, ``PanCK``, ``KRT5``.
    Matching on the substring ``keratin`` catches the spelled-out forms, and the
    ``krt``/``ck`` prefixes catch the gene-symbol and shorthand forms, while a
    trailing-digit guard keeps ``CK`` from swallowing unrelated ``CD``-style
    names. Selection order follows the channel order in the image so the
    composite is reproducible.
    """
    selected: list[str] = []
    for name in channel_names:
        text = str(name).strip()
        lowered = text.casefold()
        is_dapi = "dapi" in lowered
        is_vimentin = "vimentin" in lowered
        is_keratin = (
            "keratin" in lowered
            or lowered.startswith("krt")
            or (lowered.startswith("ck") and lowered[2:3].isdigit())
            or lowered in {"panck", "pan-ck"}
        )
        if is_dapi or is_vimentin or is_keratin:
            selected.append(text)
    return selected


# ── continuous colour ────────────────────────────────────────────────────────

def _lerp(c1: Sequence[int], c2: Sequence[int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))  # type: ignore[return-value]


def continuous_ramp(
    value: float,
    vmin: float,
    vmax: float,
    low_color: str,
    high_color: str,
    mid_color: str | None = None,
    vmid: float | None = None,
) -> tuple[int, int, int]:
    """Map a numeric value onto a two- or three-stop colour ramp.

    With ``mid_color`` the ramp has three stops, which is the right choice when
    the middle of the range is meaningful (a neutral level with depletion below
    and enrichment above). Without it the ramp runs straight from low to high.
    """
    low = parse_hex_color(low_color)
    high = parse_hex_color(high_color)
    if not np.isfinite(value):
        return low
    span = float(vmax) - float(vmin)
    if span <= 0:
        return high
    position = (float(value) - float(vmin)) / span
    position = max(0.0, min(1.0, position))
    if not mid_color:
        return _lerp(low, high, position)
    mid = parse_hex_color(mid_color)
    if vmid is None:
        split = 0.5
    else:
        split = (float(vmid) - float(vmin)) / span
        split = max(1e-6, min(1.0 - 1e-6, split))
    if position <= split:
        return _lerp(low, mid, position / split)
    return _lerp(mid, high, (position - split) / (1.0 - split))


def continuous_color_map(
    values: Sequence[Any],
    low_color: str,
    high_color: str,
    mid_color: str | None = None,
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[list[tuple[int, int, int]], float, float]:
    """Colours for a numeric series plus the (vmin, vmax) actually used.

    Percentile clipping keeps a handful of extreme cells from flattening the
    whole ramp; the returned limits are reported so the legend can state the
    range the colours actually represent.
    """
    numeric = np.asarray(
        [float(v) if v is not None and str(v) != "" else np.nan for v in values], dtype=float
    )
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return [parse_hex_color(low_color)] * len(numeric), 0.0, 1.0
    low_pct, high_pct = clip_percentiles
    vmin = float(np.percentile(finite, max(0.0, low_pct)))
    vmax = float(np.percentile(finite, min(100.0, high_pct)))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    vmid = (vmin + vmax) / 2.0 if mid_color else None
    colors = [
        continuous_ramp(v, vmin, vmax, low_color, high_color, mid_color, vmid) for v in numeric
    ]
    return colors, vmin, vmax


# ── cell colour resolution ───────────────────────────────────────────────────

NEUTRAL_GREY = "#777777"


def resolve_cell_colors(
    features: Sequence[dict[str, Any]],
    mode: str,
    *,
    fixed_color: str = "#FF5B40",
    category_column: str = "",
    category_colors: dict[str, str] | None = None,
    continuous_column: str = "",
    low_color: str = "#2C7BB6",
    mid_color: str | None = None,
    high_color: str = "#D7191C",
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
    reference_column: str = "",
    compare_column: str = "",
    match_color: str = "#00B050",
    mismatch_color: str = "#E03131",
    ignore_values: Sequence[str] = (),
    ignored_color: str = NEUTRAL_GREY,
    missing_color: str = NEUTRAL_GREY,
    label_key: str = "label",
) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
    """One RGB colour per feature, plus a legend describing the mapping.

    Modes: ``unique`` (stable per-cell hue), ``fixed``, ``categorical`` (a column
    of labels), ``continuous`` (a numeric column on a two- or three-stop ramp),
    and ``comparison`` (does a method column agree with a reference column).
    Colour resolution is kept out of the renderer so every mode is a pure
    function that can be checked without drawing anything.
    """
    n = len(features)
    legend: dict[str, Any] = {"mode": mode}

    def prop(feature: dict[str, Any], key: str) -> Any:
        return (feature.get("properties", {}) or {}).get(key)

    if mode == "categorical" and category_column:
        palette = dict(category_colors or {})
        values = [prop(f, category_column) for f in features]
        missing = sorted(
            {str(v) for v in values if v is not None and str(v) not in palette and str(v) != "nan"}
        )
        if missing:
            fallback = build_color_lookup(missing)
            for key, colour in fallback.items():
                palette.setdefault(key, colour)
        colors = []
        used: dict[str, str] = {}
        for value in values:
            if value is None or str(value) == "nan":
                colors.append(parse_hex_color(missing_color))
                continue
            hexcolor = palette.get(str(value), missing_color)
            used[str(value)] = hexcolor
            colors.append(parse_hex_color(hexcolor))
        legend["categories"] = dict(sorted(used.items()))
        return colors, legend

    if mode == "continuous" and continuous_column:
        values = [prop(f, continuous_column) for f in features]
        colors, vmin, vmax = continuous_color_map(
            values, low_color, high_color, mid_color, clip_percentiles
        )
        legend.update(
            {
                "column": continuous_column,
                "vmin": round(float(vmin), 4),
                "vmax": round(float(vmax), 4),
                "stops": 3 if mid_color else 2,
            }
        )
        return colors, legend

    if mode == "comparison" and reference_column and compare_column:
        ignored = {str(v).strip().casefold() for v in ignore_values if str(v).strip()}
        colors: list[tuple[int, int, int]] = []
        counts = {"match": 0, "mismatch": 0, "ignored": 0, "missing": 0}
        for feature in features:
            left = prop(feature, reference_column)
            right = prop(feature, compare_column)
            left_text = "" if left is None else str(left).strip()
            right_text = "" if right is None else str(right).strip()
            if not left_text or not right_text or left_text.lower() == "nan" or right_text.lower() == "nan":
                counts["missing"] += 1
                colors.append(parse_hex_color(missing_color))
            elif left_text.casefold() in ignored or right_text.casefold() in ignored:
                # An excluded label such as "Others" is not evidence of
                # agreement or disagreement, so it gets its own colour rather
                # than being scored as a mismatch.
                counts["ignored"] += 1
                colors.append(parse_hex_color(ignored_color))
            elif left_text.casefold() == right_text.casefold():
                counts["match"] += 1
                colors.append(parse_hex_color(match_color))
            else:
                counts["mismatch"] += 1
                colors.append(parse_hex_color(mismatch_color))
        scored = counts["match"] + counts["mismatch"]
        legend.update(
            {
                "reference": reference_column,
                "compared": compare_column,
                **counts,
                "agreement": round(counts["match"] / scored, 4) if scored else None,
            }
        )
        return colors, legend

    if mode == "fixed":
        return [parse_hex_color(fixed_color)] * n, legend

    return [cell_color((features[i].get("properties", {}) or {}).get(label_key), i) for i in range(n)], legend


# ── in-figure legend ─────────────────────────────────────────────────────────

def legend_entries_from(
    legend: dict[str, Any],
    *,
    base_label: str | None = None,
    tissue_classes: Sequence[tuple[str, str]] = (),
    fixed_color: str = "#FF5B40",
    max_categories: int = 14,
) -> list[tuple[str, str]]:
    """Build ``(label, hex colour)`` rows describing what the figure shows.

    Mirrors the reference script's compact legend but reports the colour mode
    actually in use, so a continuous ramp shows its range and a comparison shows
    its agreement rate rather than a generic "cells" swatch.
    """
    rows: list[tuple[str, str]] = []
    if base_label:
        rows.append((base_label, "#B0B0B0"))
    for name, color in tissue_classes:
        rows.append((f"Tissue: {name}", color))

    mode = str(legend.get("mode", ""))
    if mode == "categorical":
        for name, color in list(legend.get("categories", {}).items())[:max_categories]:
            rows.append((str(name), str(color)))
        remaining = len(legend.get("categories", {})) - max_categories
        if remaining > 0:
            rows.append((f"+{remaining} more", NEUTRAL_GREY))
    elif mode == "continuous":
        rows.append((f"{legend.get('column','value')} low ({legend.get('vmin')})", "__RAMP_LOW__"))
        rows.append((f"{legend.get('column','value')} high ({legend.get('vmax')})", "__RAMP_HIGH__"))
    elif mode == "comparison":
        agreement = legend.get("agreement")
        suffix = f" ({agreement * 100:.1f}%)" if isinstance(agreement, (int, float)) else ""
        rows.append((f"Match{suffix}  n={legend.get('match', 0):,}", "__MATCH__"))
        rows.append((f"Mismatch  n={legend.get('mismatch', 0):,}", "__MISMATCH__"))
        if legend.get("ignored"):
            rows.append((f"Ignored  n={legend.get('ignored', 0):,}", "__IGNORED__"))
    elif mode == "fixed":
        rows.append(("Cells", fixed_color))
    else:
        rows.append(("Cells (unique colours)", "#FF5B40"))
    return rows


def draw_legend(
    image: Image.Image,
    entries: Sequence[tuple[str, str]],
    *,
    placement: str = "top-left",
    background: str = "#FFFFFF",
    background_alpha: int = 215,
    text_color: str = "#141414",
    margin_scale: float = 1.0,
) -> Image.Image:
    """Draw a compact swatch legend onto a copy of ``image``."""
    from PIL import ImageDraw as _ImageDraw, ImageFont as _ImageFont

    rows = [(str(label), str(color)) for label, color in entries if str(label)]
    if not rows:
        return image

    canvas = image.convert("RGBA").copy()
    draw = _ImageDraw.Draw(canvas, "RGBA")
    scale = max(1, round(max(canvas.size) / 2500)) * float(margin_scale)
    try:
        font = _ImageFont.load_default(size=int(11 * scale))
    except TypeError:  # Pillow < 10.1
        font = _ImageFont.load_default()

    pad = int(10 * scale)
    row_h = int(19 * scale)
    swatch = int(12 * scale)
    widest = max((draw.textlength(label, font=font) for label, _ in rows), default=80)
    box_w = int(widest + swatch + 4 * pad)
    box_h = int(len(rows) * row_h + 2 * pad)

    if "right" in placement:
        x0 = canvas.width - box_w - pad
    else:
        x0 = pad
    if "bottom" in placement:
        y0 = canvas.height - box_h - pad
    else:
        y0 = pad

    draw.rounded_rectangle(
        (x0, y0, x0 + box_w, y0 + box_h),
        radius=int(6 * scale),
        fill=(*parse_hex_color(background), int(background_alpha)),
        outline=(20, 20, 20, 220),
        width=max(1, int(scale)),
    )
    for index, (label, color) in enumerate(rows):
        y = y0 + pad + index * row_h
        box = (x0 + pad, y, x0 + pad + swatch, y + swatch)
        if color.startswith("__"):  # placeholder resolved by the caller
            color = {"__RAMP_LOW__": "#2C7BB6", "__RAMP_HIGH__": "#D7191C",
                     "__MATCH__": "#00B050", "__MISMATCH__": "#E03131",
                     "__IGNORED__": NEUTRAL_GREY}.get(color, NEUTRAL_GREY)
        draw.rectangle(box, fill=(*parse_hex_color(color), 255), outline=(0, 0, 0, 230), width=max(1, int(scale)))
        draw.text((x0 + pad * 2 + swatch, y - int(scale)), label,
                  fill=(*parse_hex_color(text_color), 255), font=font)
    return canvas


# ── legend panels placed outside the figure ──────────────────────────────────

def _render_legend_block(
    entries: Sequence[tuple[str, str]],
    title: str,
    *,
    scale: float,
    background: str,
    text_color: str,
    max_width: int | None = None,
) -> Image.Image:
    """One titled legend card sized to its contents."""
    from PIL import ImageDraw as _ImageDraw, ImageFont as _ImageFont

    try:
        font = _ImageFont.load_default(size=int(11 * scale))
        title_font = _ImageFont.load_default(size=int(12 * scale))
    except TypeError:  # Pillow < 10.1
        font = title_font = _ImageFont.load_default()

    probe = _ImageDraw.Draw(Image.new("RGB", (8, 8)))
    pad = int(9 * scale)
    row_h = int(18 * scale)
    swatch = int(11 * scale)
    widths = [probe.textlength(str(label), font=font) for label, _ in entries]
    widths.append(probe.textlength(title, font=title_font))
    width = int(max(widths, default=60) + swatch + 4 * pad)
    if max_width:
        width = min(width, int(max_width))
    height = int(len(entries) * row_h + row_h + 2 * pad)

    card = Image.new("RGB", (max(width, 1), max(height, 1)), parse_hex_color(background))
    draw = _ImageDraw.Draw(card)
    draw.text((pad, pad), title, fill=parse_hex_color(text_color), font=title_font)
    for index, (label, color) in enumerate(entries):
        y = pad + row_h + index * row_h
        draw.rectangle((pad, y, pad + swatch, y + swatch),
                       fill=parse_hex_color(color), outline=(0, 0, 0), width=max(1, int(scale)))
        draw.text((pad * 2 + swatch, y - int(scale)), str(label),
                  fill=parse_hex_color(text_color), font=font)
    return card


def attach_legend_panels(
    image: Image.Image,
    panels: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    *,
    placement: str = "right",
    background: str = "#FFFFFF",
    text_color: str = "#141414",
    gap: int = 10,
) -> Image.Image:
    """Place legend cards in a margin beside the figure instead of over it.

    Overlaying a legend hides tissue, which is exactly what you do not want in a
    figure that exists to show the tissue. Adding a margin keeps every pixel of
    the image visible and gives each legend -- markers, annotation classes and
    cell colouring -- its own titled block.
    """
    blocks = [(title, list(entries)) for title, entries in panels if entries]
    if not blocks:
        return image

    scale = max(1.0, max(image.size) / 1400.0)
    gap_px = int(gap * scale)
    horizontal = placement in {"left", "right"}
    cards = [
        _render_legend_block(
            entries, title, scale=scale, background=background, text_color=text_color,
            max_width=None if horizontal else image.width,
        )
        for title, entries in blocks
    ]

    if horizontal:
        strip_w = max(card.width for card in cards)
        strip_h = sum(card.height for card in cards) + gap_px * (len(cards) - 1)
        strip = Image.new("RGB", (strip_w, max(strip_h, 1)), parse_hex_color(background))
        y = 0
        for card in cards:
            strip.paste(card, (0, y))
            y += card.height + gap_px
        canvas = Image.new(
            "RGB", (image.width + strip_w + gap_px, max(image.height, strip.height)),
            parse_hex_color(background),
        )
        if placement == "right":
            canvas.paste(image.convert("RGB"), (0, 0))
            canvas.paste(strip, (image.width + gap_px, 0))
        else:
            canvas.paste(strip, (0, 0))
            canvas.paste(image.convert("RGB"), (strip_w + gap_px, 0))
        return canvas

    # Stacked above or below: lay the cards out in a row.
    strip_w = sum(card.width for card in cards) + gap_px * (len(cards) - 1)
    strip_h = max(card.height for card in cards)
    strip = Image.new("RGB", (max(strip_w, 1), max(strip_h, 1)), parse_hex_color(background))
    x = 0
    for card in cards:
        strip.paste(card, (x, 0))
        x += card.width + gap_px
    canvas = Image.new(
        "RGB", (max(image.width, strip.width), image.height + strip_h + gap_px),
        parse_hex_color(background),
    )
    if placement == "bottom":
        canvas.paste(image.convert("RGB"), (0, 0))
        canvas.paste(strip, (0, image.height + gap_px))
    else:
        canvas.paste(strip, (0, 0))
        canvas.paste(image.convert("RGB"), (0, strip_h + gap_px))
    return canvas


def attach_csv_properties(
    features: Sequence[dict[str, Any]],
    frame: "Any",
    columns: Sequence[str],
    feature_label_key: str = "label",
    frame_label_column: str = "label",
) -> int:
    """Join unified-CSV columns onto segmentation features by cell label.

    A cell-boundary GeoJSON carries only geometry plus ``label`` and ``area``;
    the phenotype, marker intensities and every other measurement live in the
    unified CSV. Without this join, colouring segmentation polygons by a CSV
    column silently paints every cell the "missing value" colour. Returns the
    number of features that matched a CSV row.
    """
    wanted = [c for c in columns if c]
    if frame is None or not wanted or not len(features):
        return 0
    available = [c for c in wanted if c in getattr(frame, "columns", [])]
    if not available or frame_label_column not in getattr(frame, "columns", []):
        return 0

    lookup: dict[str, dict[str, Any]] = {}
    for record in frame[[frame_label_column, *available]].to_dict("records"):
        lookup[str(record[frame_label_column])] = record

    matched = 0
    for feature in features:
        properties = feature.setdefault("properties", {})
        key = properties.get(feature_label_key)
        if key is None:
            continue
        # Labels arrive as ints from GeoJSON and may be floats in the CSV, so
        # normalise through str(int(...)) where possible before comparing.
        candidates = [str(key)]
        try:
            candidates.insert(0, str(int(float(key))))
        except (TypeError, ValueError):
            pass
        record = next((lookup[c] for c in candidates if c in lookup), None)
        if record is None:
            continue
        for column in available:
            properties[column] = record[column]
        matched += 1
    return matched
