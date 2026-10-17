"""The annotation layer: QuPath-style GeoJSON regions drawn as tiles.

Tissue compartments, pathology annotations, tissue classes and ROI boundaries
are all GeoJSON features with a ``classification`` (a name and, sometimes, a
colour). Files reach tens of megabytes, so they are not sent to the browser:
the server keeps the shapes in a spatial index and draws only the shapes that
touch each tile, filled translucently and outlined in the class colour.

A single tissue region can have hundreds of thousands of vertices. Each shape
is therefore simplified once per zoom level (to half a screen pixel, which
cannot be seen) and clipped to the tile before it is drawn.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .render import tile_bounds
from .sources import ImageInfo, fingerprint


@dataclass
class AnnotationSet:
    path: str
    fingerprint: str
    geometries: list
    classes: list[str]
    class_names: list[str]
    class_colors: dict[str, tuple[int, int, int]]
    tree: object
    _levels: dict = field(default_factory=dict, repr=False)  # zoom -> {shape index: (area, outline)}
    _levels_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def at_zoom(self, index: int, z: int):
        """Shape *index* simplified for zoom *z*: (area or None, outline or None). Cached."""
        cached = self._levels.get(z, {}).get(index)
        if cached is not None:
            return cached
        with self._levels_lock:  # tiles arrive together; simplify each shape once
            level = self._levels.setdefault(z, {})
            cached = level.get(index)
            if cached is None:
                geom = self.geometries[index]
                if z > 0:
                    geom = geom.simplify(0.5 * 2**z, preserve_topology=False)
                area = geom if geom.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection") else None
                cached = level[index] = (area, _outline(geom))
        return cached

    def legend(self) -> list[dict]:
        counts: dict[str, int] = {}
        for name in self.classes:
            counts[name] = counts.get(name, 0) + 1
        return [
            {"value": name, "color": "#%02x%02x%02x" % self.class_colors[name], "count": counts.get(name, 0)}
            for name in self.class_names
        ]


def _feature_class(feature: dict) -> tuple[str, tuple[int, int, int] | None]:
    props = feature.get("properties") or {}
    classification = props.get("classification")
    color = None
    if isinstance(classification, dict):
        name = classification.get("name")
        rgb = classification.get("color")
        if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            color = tuple(int(v) for v in rgb[:3])
    else:
        name = classification
    name = name or props.get("name") or props.get("class") or props.get("objectType") or "Annotation"
    return str(name), color


_CACHE: OrderedDict = OrderedDict()
_LOCK = threading.Lock()
_LOADING: dict[tuple[str, str], Future] = {}


def load_annotations(path: str | Path) -> AnnotationSet:
    """Parse once per file version, including simultaneous tile/legend requests."""
    key = str(Path(path).resolve())
    stamp = fingerprint(key)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None and cached.fingerprint == stamp:
            _CACHE.move_to_end(key)
            return cached
        pending = _LOADING.get((key, stamp))
        owner = pending is None
        if owner:
            pending = _LOADING[key, stamp] = Future()
    if not owner:
        return pending.result()
    try:
        result = _read_annotations(key, stamp)
        with _LOCK:
            _CACHE[key] = result
            while len(_CACHE) > 16:
                _CACHE.popitem(last=False)
        pending.set_result(result)
        return result
    except BaseException as exc:
        pending.set_exception(exc)
        raise
    finally:
        with _LOCK:
            _LOADING.pop((key, stamp), None)


def _read_annotations(key: str, stamp: str) -> AnnotationSet:
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    from spacemap.acc_utils import build_color_mapping

    data = json.loads(Path(key).read_text())
    features = data["features"] if isinstance(data, dict) and "features" in data else data
    if isinstance(features, dict):
        features = [features]
    geometries, classes, given = [], [], {}
    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry:
            continue
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if geom.is_empty:
            continue
        name, color = _feature_class(feature)
        geometries.append(geom)
        classes.append(name)
        if color is not None:
            given.setdefault(name, color)
    names = sorted(set(classes), key=str.casefold)
    fallback = build_color_mapping(names, existing={n: "#%02x%02x%02x" % c for n, c in given.items()})
    colors = {}
    for name in names:
        hex_color = str(fallback.get(name, "#ffff00")).lstrip("#")
        colors[name] = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return AnnotationSet(key, stamp, geometries, classes, names, colors, STRtree(geometries) if geometries else None)


def _outline(geom):
    """Boundary lines of any geometry; collections (which have none in Shapely 2) are split."""
    from shapely.ops import unary_union

    if geom.geom_type == "GeometryCollection":
        parts = [_outline(part) for part in geom.geoms]
        parts = [part for part in parts if part is not None and not part.is_empty]
        return unary_union(parts) if parts else None
    try:
        boundary = geom.boundary
    except Exception:
        return None
    return boundary if boundary is not None and not boundary.is_empty else geom


def annotation_at(path, x, y, hidden=()):
    """Find the smallest visible polygon at a point using the cached spatial index."""
    import math
    from shapely.geometry import Point, mapping

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Choose a point on the image.")
    loaded = load_annotations(path)
    point = Point(x, y)
    hits = [] if loaded.tree is None else loaded.tree.query(point)
    candidates = [int(i) for i in hits if loaded.classes[i] not in hidden and
                  loaded.geometries[i].area > 0 and loaded.geometries[i].covers(point)]
    if not candidates:
        return {"feature": None}
    index = min(candidates, key=lambda i: loaded.geometries[i].area)
    name = loaded.classes[index]
    return {"index": index, "stamp": loaded.fingerprint, "feature": {
        "type": "Feature", "geometry": mapping(loaded.geometries[index]),
        "properties": {"id": str(index), "name": name,
                       "classification": {"name": name, "color": list(loaded.class_colors[name])}}}}


def render_annotation_tile(
    annotations: AnnotationSet,
    image: ImageInfo,
    z: int,
    x: int,
    y: int,
    *,
    fill_alpha: int = 90,
    hidden: set[str] | None = None,
    line_px: int = 2,
) -> np.ndarray:
    """RGBA tile of the annotations touching one viewer tile."""
    import cv2
    import rasterio.features
    import shapely
    from affine import Affine
    from shapely.geometry import box

    x0, y0, x1, y1, out_w, out_h = tile_bounds(image, z, x, y)
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    if out_w == 0 or out_h == 0 or annotations.tree is None:
        return rgba
    scale = 2**z
    hits = annotations.tree.query(box(x0, y0, x1, y1))
    if len(hits) == 0:
        return rgba
    hidden = hidden or set()
    transform = Affine(scale, 0, x0, 0, scale, y0)
    margin = 2 * scale  # so outlines just outside the tile still reach its edge
    clip = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    index_of = {name: i + 1 for i, name in enumerate(annotations.class_names)}
    fills, lines = [], []
    for hit in sorted(int(h) for h in hits):
        name = annotations.classes[hit]
        if name in hidden:
            continue
        value = index_of[name]
        area, outline = annotations.at_zoom(hit, z)
        if area is not None:
            clipped = shapely.clip_by_rect(area, *clip)
            if not clipped.is_empty:
                fills.append((clipped, value))
        if outline is not None:
            clipped = shapely.clip_by_rect(outline, *clip)
            if not clipped.is_empty:
                lines.append((clipped, value))
    if not fills and not lines:
        return rgba
    palette = np.zeros((len(annotations.class_names) + 1, 3), dtype=np.uint8)
    for name, index in index_of.items():
        palette[index] = annotations.class_colors[name]
    if fills:
        fill = rasterio.features.rasterize(fills, out_shape=(out_h, out_w), transform=transform, fill=0, dtype="uint16")
        inside = fill > 0
        rgba[inside, :3] = palette[fill[inside]]
        rgba[inside, 3] = fill_alpha
    if lines:
        edge = rasterio.features.rasterize(
            lines, out_shape=(out_h, out_w), transform=transform, fill=0, dtype="uint16", all_touched=True
        )
        if line_px > 1:
            edge = cv2.dilate(edge, np.ones((line_px, line_px), np.uint8))
        on_edge = edge > 0
        rgba[on_edge, :3] = palette[edge[on_edge]]
        rgba[on_edge, 3] = 255
    return rgba
