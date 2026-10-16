"""Shapes drawn in the viewer: ROIs, regions to segment, background areas.

Shapes are kept as a GeoJSON FeatureCollection in full-resolution image pixels,
the format Tab 3 extracts ROIs from and QuPath reads:

    properties: {"id", "name", "label", "classification": {"name", "color"}}

``name`` and ``label`` carry the same text (Tab 3 reads ``label``; QuPath and
the viewer read ``name``). Files are replaced atomically, so a crash never
leaves half a file.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import numpy as np

MAX_DISPLAY_VERTICES = 4000  # per shape sent to the browser; enough to look exact on screen


def _hex(color) -> str | None:
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return color.lower()
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in color[:3])
    return None


def _rgb(color: str | None) -> list[int] | None:
    if not color:
        return None
    color = color.lstrip("#")
    return [int(color[i : i + 2], 16) for i in (0, 2, 4)]


def normalise(feature: dict, index: int = 0) -> dict | None:
    """One feature in the viewer's form, or None when it has no usable area or line."""
    from shapely.geometry import mapping, shape
    from shapely.validation import make_valid

    if not isinstance(feature, dict) or not feature.get("geometry"):
        return None
    try:
        geom = shape(feature["geometry"])
    except Exception:
        return None
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    props = dict(feature.get("properties") or {})
    classification = props.get("classification")
    class_name = classification.get("name") if isinstance(classification, dict) else classification
    color = _hex(classification.get("color")) if isinstance(classification, dict) else None
    color = color or _hex(props.get("color"))
    name = str(props.get("name") or props.get("label") or class_name or f"Shape {index + 1}")
    props.update(
        {
            "id": str(props.get("id") or secrets.token_hex(4)),
            "name": name,
            "label": name,
        }
    )
    if class_name or color:
        props["classification"] = {"name": str(class_name or name), "color": _rgb(color) or [255, 214, 0]}
    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def load(path: str | Path) -> list[dict]:
    """Features of a shapes file; an absent file has none."""
    path = Path(path)
    if not path.is_file():
        return []
    data = json.loads(path.read_text() or "{}")
    features = data.get("features", []) if isinstance(data, dict) else data
    if isinstance(features, dict):
        features = [features]
    return [f for f in (normalise(feature, i) for i, feature in enumerate(features or [])) if f]


def save(path: str | Path, features: list[dict]) -> list[dict]:
    """Replace the file with *features* (normalised). Returns what was written."""
    path = Path(path)
    clean = [f for f in (normalise(feature, i) for i, feature in enumerate(features or [])) if f]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"type": "FeatureCollection", "features": clean}))
    os.replace(temporary, path)
    return clean


def for_display(features: list[dict], max_vertices: int = MAX_DISPLAY_VERTICES, *, max_error_px: float | None = None) -> list[dict]:
    """Features light enough to draw in the browser: large shapes are simplified.

    Only what is sent is simplified; the file keeps every vertex. A simplified
    shape is marked ``"simplified": true`` so the viewer does not save it back
    over the detailed one unless it was edited.

    ``max_error_px`` limits simplification in full-resolution image pixels.
    With this bound the vertex budget is a preference: a polygon with many
    holes may need more points, and accuracy takes precedence over its size.
    """
    import shapely
    from shapely.geometry import mapping, shape

    if max_error_px is not None and (not np.isfinite(max_error_px) or max_error_px <= 0):
        raise ValueError("The display error limit must be a positive finite number.")

    out = []
    for feature in features:
        geom = shape(feature["geometry"])
        count = int(shapely.get_num_coordinates(geom))
        simplified = False
        if count > max_vertices:
            minx, miny, maxx, maxy = geom.bounds
            tolerance = max(maxx - minx, maxy - miny) / 4000.0
            while count > max_vertices and tolerance < 1e6:
                if max_error_px is not None:
                    tolerance = min(tolerance, max_error_px)
                candidate = geom.simplify(tolerance, preserve_topology=True)
                count = int(shapely.get_num_coordinates(candidate))
                if max_error_px is not None and tolerance >= max_error_px:
                    break
                tolerance *= 2
            geom, simplified = candidate, True
        item = {"type": "Feature", "geometry": mapping(geom), "properties": dict(feature["properties"])}
        if simplified:
            item["properties"]["simplified"] = True
        out.append(item)
    return out


def merge_edits(stored: list[dict], edited: list[dict]) -> list[dict]:
    """The viewer's list of shapes, keeping full detail for shapes it only saw simplified.

    A shape sent back unchanged with ``simplified`` set takes its geometry from
    the stored file; everything else (names, colours, new and edited shapes) is
    taken from the viewer. Shapes the viewer no longer lists are deleted.
    """
    by_id = {f["properties"]["id"]: f for f in stored}
    result = []
    for feature in edited:
        props = dict(feature.get("properties") or {})
        original = by_id.get(str(props.get("id")))
        if props.pop("simplified", False) and original is not None:
            feature = {"type": "Feature", "geometry": original["geometry"], "properties": props}
        else:
            feature = {"type": "Feature", "geometry": feature.get("geometry"), "properties": props}
        result.append(feature)
    return result


def area_px(feature: dict) -> float:
    from shapely.geometry import shape

    return float(shape(feature["geometry"]).area)


MAX_EDIT_POINTS = 20000  # per Merge or Split; a cell has tens to hundreds


def _points(value, least: int, what: str) -> list[tuple[float, float]]:
    import math

    if not isinstance(value, list) or len(value) < least:
        raise ValueError(f"{what} needs at least {least} points.")
    points = []
    for point in value:
        if not (
            isinstance(point, (list, tuple))
            and len(point) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point)
        ):
            raise ValueError(f"{what} has a point that is not two numbers.")
        points.append((float(point[0]), float(point[1])))
    return points


def _polygon(coordinates):
    """A valid shapely polygon from GeoJSON Polygon coordinates (outline, then holes)."""
    from shapely.geometry import Polygon

    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("A shape needs an outline.")
    rings = [_points(ring, 3, "An outline") for ring in coordinates]
    if sum(len(ring) for ring in rings) > MAX_EDIT_POINTS:
        raise ValueError("That shape has too many points to edit here.")
    polygon = Polygon(rings[0], rings[1:])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda part: part.area)
    if polygon.is_empty or polygon.area < 1:
        raise ValueError("That shape is too small to edit.")
    return polygon


def _coordinates(polygon, smallest_hole: float = 0.0) -> list:
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    polygon = orient(polygon)
    rings = [polygon.exterior] + [ring for ring in polygon.interiors if Polygon(ring).area > smallest_hole]
    return [[[round(x, 1), round(y, 1)] for x, y in ring.coords] for ring in rings]


def join(first, second, *, gap: float = 3.0) -> list:
    """One outline from two touching shapes, such as two pieces of one cell.

    Pieces up to *gap* pixels apart are joined across the gap (cell outlines
    traced from neighbouring masks leave about one pixel between them); pieces
    further apart are refused, since joining them would invent tissue.
    """
    a, b = _polygon(first), _polygon(second)
    apart = a.distance(b)
    if apart > gap:
        raise ValueError(f"These two shapes are {apart:.0f} px apart; Merge joins shapes that touch.")
    whole = a.union(b)
    if whole.geom_type != "Polygon":
        reach = apart / 2 + 0.5
        whole = whole.buffer(reach).buffer(-reach).union(a).union(b)
    if whole.geom_type != "Polygon":
        raise ValueError("These two shapes could not be joined into one outline.")
    return _coordinates(whole, smallest_hole=4 * gap * gap)


def cut(coordinates, line) -> list:
    """The pieces of a shape cut along a straight line dragged across it, largest first.

    A line that stops a little short of the edge is carried on to it, so the cut
    goes right across; only the part of the line's path through the shape that
    the drawn line touches cuts, so other lobes of a curved shape stay whole.
    """
    import math

    from shapely.geometry import LineString
    from shapely.ops import split

    polygon = _polygon(coordinates)
    (x0, y0), (x1, y1) = _points(line, 2, "The cutting line")[:2]
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 2:
        raise ValueError("Drag a longer line across the shape.")
    minx, miny, maxx, maxy = polygon.bounds
    reach = math.hypot(maxx - minx, maxy - miny) + length
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    drawn = LineString([(x0, y0), (x1, y1)])
    path = LineString([(x0 - dx * reach, y0 - dy * reach), (x1 + dx * reach, y1 + dy * reach)])
    inside = polygon.intersection(path)
    parts = [inside] if inside.geom_type == "LineString" else list(getattr(inside, "geoms", []))
    cutters = [part for part in parts if part.geom_type == "LineString" and part.distance(drawn) < 1e-6]
    if not cutters:
        raise ValueError("Drag the line across the shape to cut it.")
    pieces = []
    for cutter in cutters:
        # Carry each cut a pixel past the edge so it crosses the outline cleanly.
        (ax, ay), (bx, by) = cutter.coords[0], cutter.coords[-1]
        span = max(math.hypot(bx - ax, by - ay), 1e-9)
        ux, uy = (bx - ax) / span, (by - ay) / span
        blade = LineString([(ax - ux, ay - uy), (bx + ux, by + uy)])
        pieces = [piece for piece in (pieces or [polygon]) for piece in split(piece, blade).geoms]
    pieces = sorted((piece for piece in pieces if piece.area >= 4), key=lambda piece: -piece.area)
    if len(pieces) < 2:
        raise ValueError("The line does not cut the shape in two; drag it right across.")
    return [_coordinates(piece) for piece in pieces]


def mask(features: list[dict], width: int, height: int, *, downsample: float = 1.0) -> np.ndarray:
    """Boolean mask of the shapes on an image of *width* x *height* (optionally downsampled)."""
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape

    out_w, out_h = max(1, int(round(width / downsample))), max(1, int(round(height / downsample)))
    geoms = [shape(f["geometry"]) for f in features]
    if not geoms:
        return np.zeros((out_h, out_w), dtype=bool)
    burned = rasterio.features.rasterize(
        ((g, 1) for g in geoms),
        out_shape=(out_h, out_w),
        transform=Affine(downsample, 0, 0, 0, downsample, 0),
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)
