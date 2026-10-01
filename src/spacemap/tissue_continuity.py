"""Tissue-piece geometry for spatially fragmented sections.

A section commonly holds several physically separate scraps of biopsy. Any
measurement that crosses the bare glass between them is meaningless, and so is a
straight line that leaves the tissue inside a single U-shaped scrap. This module
supplies the geometry those corrections need:

* :func:`split_pieces` turns one valid-tissue polygon into its separate pieces.
* :func:`piece_metrics` measures each piece, including the largest circle that
  fits inside it - the honest upper bound on the analysis radius that piece can
  support.
* :func:`find_necks` detects every narrow bridge that separates two
  substantial sections of one piece, and measures each bridge's own width.
* :func:`cut_at_necks` optionally splits such a piece along the narrowest point.
* :func:`select_pieces` applies the size filters and reports what was dropped.

Everything works directly on shapely geometry in full-resolution image pixels,
the same space the GeoJSON annotations live in. Nothing here looks at cell
positions: a sparsely populated region is still tissue, and only the tissue
polygon decides where a gap is.
"""

from __future__ import annotations

from typing import Any, Sequence

import shapely
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

__all__ = [
    "DEFAULT_PIECE_CONFIG",
    "split_pieces",
    "piece_metrics",
    "find_necks",
    "find_necks_raster",
    "cut_at_necks",
    "select_pieces",
    "analyse_tissue_pieces",
    "mm2_to_px2",
    "px2_to_mm2",
]


# Every threshold the user can set for turning one valid-tissue polygon into a
# reviewed set of analysis pieces. Filters are opt-in so that switching the step
# on does not silently change an existing project's tissue definition.
DEFAULT_PIECE_CONFIG: dict[str, Any] = {
    "microns_per_pixel": 0.5083,
    # Neck handling
    "detect_necks": True,
    "neck_min_width_um": 100.0,
    # A neck joins two real sections, so both sides must be tissue in their own
    # right. Without this every bleb and shred hanging off a large mass detaches
    # under opening and gets reported as a neck. 0.02 mm2 is a disc about 160 um
    # across: an island, not a sliver. The fraction is a light backstop for the
    # case where a piece is itself small.
    "neck_min_lobe_area_mm2": 0.02,
    "neck_min_lobe_fraction": 0.02,
    "cut_at_necks": False,
    # "raster" measures the bridges on a small binary image, which is what
    # makes the step usable on a hand-drawn outline; "polygon" keeps the
    # older all-vector search.
    "neck_method": "raster",
    # Size filters
    "filter_by_area": True,
    "min_area_mm2": 0.05,
    "filter_by_inscribed_circle": False,
    "min_inscribed_diameter_um": 100.0,
    # Geometry hygiene
    "fill_holes_below_um2": 0.0,
    "simplify_px": 0.0,
}

# Bisection bounds when measuring a bridge, in microns. A bridge wider than this
# is not a neck by any reasonable definition.
_MAX_PROBE_WIDTH_UM = 2000.0


# ==========================================================================
# Unit helpers
# ==========================================================================

def mm2_to_px2(area_mm2: float, microns_per_pixel: float) -> float:
    """Convert mm^2 to image pixels^2 at the given calibration."""
    return float(area_mm2) * 1e6 / max(float(microns_per_pixel), 1e-9) ** 2


def px2_to_mm2(area_px2: float, microns_per_pixel: float) -> float:
    """Convert image pixels^2 to mm^2 at the given calibration."""
    return float(area_px2) * float(microns_per_pixel) ** 2 / 1e6


# ==========================================================================
# Pieces
# ==========================================================================

def _as_polygons(geometry) -> list[Polygon]:
    """Flatten any geometry into its polygon parts, dropping empties."""
    if geometry is None or geometry.is_empty:
        return []
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    parts: list[Polygon] = []
    candidates = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    for candidate in candidates:
        if getattr(candidate, "geom_type", "") == "Polygon" and not candidate.is_empty:
            parts.append(candidate)
        elif getattr(candidate, "geom_type", "") in ("MultiPolygon", "GeometryCollection"):
            parts.extend(_as_polygons(candidate))
    return parts


def split_pieces(geometry, simplify_px: float = 0.0, fill_holes_below_px2: float = 0.0) -> list[Polygon]:
    """Separate one tissue annotation into its physically disconnected pieces.

    Polygon parts of a union are already the connected components, so no
    rasterization is involved and no pixel is lost to resampling.
    """
    pieces = _as_polygons(geometry)
    if fill_holes_below_px2 > 0:
        pieces = [_fill_small_holes(piece, fill_holes_below_px2) for piece in pieces]
    if simplify_px > 0:
        simplified = []
        for piece in pieces:
            candidate = piece.simplify(float(simplify_px), preserve_topology=True)
            simplified.append(candidate if not candidate.is_empty else piece)
        pieces = simplified
    # Largest first, so piece 1 is the dominant scrap on every image.
    return sorted(pieces, key=lambda part: part.area, reverse=True)


def _fill_small_holes(polygon: Polygon, min_area_px2: float) -> Polygon:
    """Drop interior rings smaller than ``min_area_px2``.

    A tiny hole is usually a segmentation speck rather than a real lumen. A large
    one is a genuine void that a straight line has no business crossing, so it is
    always kept.
    """
    keep = [ring for ring in polygon.interiors if Polygon(ring).area >= float(min_area_px2)]
    if len(keep) == len(polygon.interiors):
        return polygon
    return Polygon(polygon.exterior, keep)


def piece_metrics(polygon: Polygon, microns_per_pixel: float) -> dict[str, float]:
    """Size and shape of one tissue piece.

    ``inscribed_r_um`` is the radius of the largest circle that fits inside the
    piece without leaving tissue. It is the largest neighbourhood radius the
    piece can support for even one cell, so it is the honest ceiling on any
    radius-based statistic computed there.
    """
    area_px2 = float(polygon.area)
    area_um2 = area_px2 * float(microns_per_pixel) ** 2
    perimeter_um = float(polygon.exterior.length) * float(microns_per_pixel)
    try:
        # GEOS returns the centre-to-boundary segment; its length is the radius.
        inscribed_px = float(shapely.maximum_inscribed_circle(polygon).length)
    except Exception:
        inscribed_px = 0.0
    circularity = (4.0 * 3.141592653589793 * area_um2 / perimeter_um ** 2) if perimeter_um > 0 else 0.0
    return {
        "area_px2": area_px2,
        "area_mm2": area_um2 / 1e6,
        "inscribed_r_um": inscribed_px * float(microns_per_pixel),
        "inscribed_d_um": 2.0 * inscribed_px * float(microns_per_pixel),
        "effective_r_um": (area_um2 / 3.141592653589793) ** 0.5,
        "perimeter_mm": perimeter_um / 1e3,
        "circularity": min(max(circularity, 0.0), 1.0),
        "n_holes": int(len(polygon.interiors)),
    }


# ==========================================================================
# Necks
# ==========================================================================

def _opening(polygon: Polygon, radius_px: float):
    """Morphological opening by a disc, done exactly in polygon space."""
    if radius_px <= 0:
        return polygon
    eroded = polygon.buffer(-float(radius_px))
    if eroded.is_empty:
        return eroded
    return eroded.buffer(float(radius_px))


def _lobes(polygon: Polygon, radius_px: float) -> list[Polygon]:
    return _as_polygons(_opening(polygon, radius_px))


def _substantial_lobes(
    polygon: Polygon,
    radius_px: float,
    min_area_px2: float,
    min_fraction: float,
) -> list[Polygon]:
    """Lobes big enough to count as a section of tissue in their own right.

    A neck is a narrow region *between two larger sections*, so a lobe only
    counts when it holds a real share of the piece. Without this, opening also
    detaches every small bleb and shred hanging off a large mass, and each one
    would be reported as a neck.
    """
    parent_area = float(polygon.area)
    keep: list[Polygon] = []
    for lobe in _lobes(polygon, radius_px):
        area = float(lobe.area)
        if area < float(min_area_px2):
            continue
        if parent_area > 0 and (area / parent_area) < float(min_fraction):
            continue
        keep.append(lobe)
    return keep


def _narrowest_substantial_split(
    polygon: Polygon,
    max_width_um: float,
    microns_per_pixel: float,
    min_area_px2: float,
    min_fraction: float,
    tolerance_um: float,
) -> float | None:
    """Width of the narrowest bridge that separates two substantial sections.

    Think of flooding one side: water crosses a wide bridge freely and a narrow
    one barely at all. Opening the piece with a disc of radius ``w/2`` severs
    every bridge thinner than ``w``, so the smallest ``w`` at which two
    substantial sections come apart *is* the width of the bridge joining them.

    Bisecting for that width is what makes the number meaningful. Measuring at a
    single fixed probe instead reports the narrowest sliver anywhere in the
    piece, which is usually the stalk of some irrelevant bleb.
    """
    def separates(width_um: float) -> bool:
        radius_px = (float(width_um) / 2.0) / float(microns_per_pixel)
        return len(_substantial_lobes(polygon, radius_px, min_area_px2, min_fraction)) >= 2

    if not separates(max_width_um):
        return None
    low_um, high_um = 0.0, float(max_width_um)
    while (high_um - low_um) > float(tolerance_um):
        mid_um = 0.5 * (low_um + high_um)
        if separates(mid_um):
            high_um = mid_um
        else:
            low_um = mid_um
    return float(high_um)


def _joining_strands(polygon: Polygon, lobes: Sequence[Polygon], tolerance_px: float = 1.0) -> list[Polygon]:
    """The strips of tissue that hold two different sections together.

    Opening removes two kinds of material: the strands bridging the sections and
    the ragged fringe around every outline. Only a strip that reaches two
    different sections is load-bearing, so only those are the neck.
    """
    if len(lobes) < 2:
        return []
    from shapely.ops import unary_union

    residue = _as_polygons(polygon.difference(unary_union(list(lobes))))
    strands: list[Polygon] = []
    for part in residue:
        probe = part.buffer(float(tolerance_px))
        if sum(1 for lobe in lobes if probe.intersects(lobe)) >= 2:
            strands.append(part)
    return strands


def _separate(
    polygon: Polygon,
    strands: Sequence[Polygon],
    radius_px: float,
    lobes: Sequence[Polygon],
) -> list[Polygon]:
    """Cut every joining strand at once and return the sections that fall apart."""
    from shapely.ops import unary_union

    # A single strand is parted most tidily across its narrowest point, which
    # costs only the blade's own footprint and leaves both sides intact.
    centre = _strand_centre(strands, lobes, polygon)
    if len(lobes) >= 2:
        # The bridge is what is left of the piece once the sections it joins are
        # taken out, and near the centre is where it is narrow. Both bounds are
        # needed: the residue on real tissue sprawls for hundreds of microns, so
        # confining the blade to the residue alone still lets it rule a line
        # across the piece.
        bridge = polygon.difference(unary_union(list(lobes)))
        limit = None
        if not bridge.is_empty:
            limit = bridge.buffer(max(radius_px * 0.75, 1.0)).intersection(
                centre.buffer(max(radius_px * 4.0, 16.0)))
            if limit.is_empty:
                limit = None
        single = _cut_blade(lobes[0], lobes[1], radius_px, limit=limit)
        if single is not None:
            parts = _as_polygons(polygon.difference(single))
            if len(parts) >= 2:
                return sorted(parts, key=lambda part: part.area, reverse=True)
    # Several strands at once cannot be parted by one blade, so every joining
    # strand is removed together. That costs the strands themselves, which is
    # the price of separating sections held at more than one point.
    if strands:
        # Every joining strand has to give way at once, but each is cut where it
        # passes closest to the constriction rather than removed along its whole
        # length. Taking strands whole does part the sections, and it is what the
        # step used to do, but a joining strand on real tissue can run for
        # hundreds of microns before it reaches anything narrow; removing all of
        # it leaves a track across the piece that reads as a cut nobody asked
        # for. Two strands on opposite sides of a slit are still both cut, which
        # a single region around the centre could not do.
        from shapely.ops import nearest_points

        strand_union = unary_union(list(strands))
        for reach in (3.0, 6.0, 10.0):
            radius = max(radius_px * reach, 12.0)
            discs = unary_union([
                nearest_points(strand, centre)[0].buffer(radius) for strand in strands
            ])
            local = strand_union.intersection(discs)
            if local.is_empty:
                continue
            parts = _as_polygons(polygon.difference(local))
            if len(parts) >= 2:
                return sorted(parts, key=lambda part: part.area, reverse=True)
    return []


def _strand_centre(strands: Sequence[Polygon], lobes: Sequence[Polygon], polygon: Polygon):
    """A representative point for the neck as a whole."""
    if strands:
        widest = max(strands, key=lambda strand: strand.area)
        return widest.representative_point()
    if len(lobes) >= 2:
        line = shapely.shortest_line(lobes[0], lobes[1])
        if line is not None and not line.is_empty:
            return line.interpolate(0.5, normalized=True)
    return polygon.representative_point()


def _strand_marker(strands: Sequence[Polygon], centre, width_um: float, microns_per_pixel: float):
    """Drawable outline of the neck, so a detection can be checked on the image.

    A joining strand on real tissue is rarely a short bridge. It is whatever is
    left once the two sections are opened away, and that residue can wander for
    hundreds of microns along a fringe before it reaches the constriction. Drawn
    whole it appears on the image as a line running out across a piece, far from
    anything narrow, which reads as a cut that was never made. The neck is the
    constriction, so the marker is kept to the part of the strand that lies
    around the measured centre.
    """
    from shapely.ops import unary_union

    radius_px = max(width_um / (2.0 * microns_per_pixel), 10.0)
    if not strands:
        return centre.buffer(radius_px)
    outline = unary_union([strand.buffer(max(width_um / (4.0 * microns_per_pixel), 3.0))
                           for strand in strands])
    # Roughly two bridge widths across: enough to show the constriction and
    # what crosses it, small enough that it never reads as a line drawn over
    # the tissue.
    local = outline.intersection(centre.buffer(radius_px * 2.0))
    return local if not local.is_empty else centre.buffer(radius_px)


def find_necks(
    polygon: Polygon,
    microns_per_pixel: float,
    max_width_um: float = 100.0,
    min_lobe_area_mm2: float = 0.02,
    min_lobe_fraction: float = 0.02,
    tolerance_um: float = 2.0,
    max_necks: int = 12,
    cut: bool = False,
) -> tuple[list[dict[str, Any]], list[Polygon]]:
    """Find every neck in one piece, narrowest first.

    A piece with three constrictions has three necks, so each one is found, cut
    away, and the two halves searched again. Two sections may also hang together
    by several strands at once - a slit that stays joined at both ends - which is
    why the neck is defined by the sections it separates rather than by any
    single strand. ``bridge_width_um`` is then the width at which all of those
    strands give way, so it describes the constriction as a whole.

    Returns ``(necks, parts)``; when ``cut`` is False ``parts`` is just
    ``[polygon]``, because detection on its own must never alter the tissue.
    """
    min_area_px2 = mm2_to_px2(min_lobe_area_mm2, microns_per_pixel)
    necks: list[dict[str, Any]] = []
    pending = [polygon]
    finished: list[Polygon] = []

    while pending and len(necks) < int(max_necks):
        current = pending.pop()
        width_um = _narrowest_substantial_split(
            current, max_width_um, microns_per_pixel, min_area_px2, min_lobe_fraction, tolerance_um,
        )
        if width_um is None:
            finished.append(current)
            continue

        radius_px = (width_um / 2.0) / float(microns_per_pixel)
        lobes = sorted(
            _substantial_lobes(current, radius_px, min_area_px2, min_lobe_fraction),
            key=lambda part: part.area, reverse=True,
        )
        # Two sections can hang together by several strands at once - a slit
        # through a piece that stays joined at both ends is the common case. One
        # blade cannot part that, so every joining strand is cut together.
        strands = _joining_strands(current, lobes)
        parts = _separate(current, strands, radius_px, lobes)

        centre = _strand_centre(strands, lobes, current)
        side_areas = (
            [px2_to_mm2(part.area, microns_per_pixel) for part in parts[:2]]
            if len(parts) >= 2
            else [px2_to_mm2(lobe.area, microns_per_pixel) for lobe in lobes[:2]]
        )
        necks.append({
            "bridge_width_um": float(width_um),
            "n_strands": int(len(strands)),
            "side_areas_mm2": side_areas,
            "smaller_side_fraction": float(min(side_areas) / max(sum(side_areas), 1e-9)),
            "x": float(centre.x),
            "y": float(centre.y),
            "marker": _strand_marker(strands, centre, width_um, microns_per_pixel),
            "separated": bool(len(parts) >= 2),
            "searched_to_um": float(max_width_um),
        })
        # Reporting never depends on the cut succeeding: a neck that could not be
        # parted is still a neck, and silently dropping it hides exactly the
        # weakly joined pieces this step exists to surface.
        if len(parts) >= 2:
            pending.extend(parts)
        else:
            finished.append(current)

    finished.extend(pending)
    necks.sort(key=lambda neck: neck["bridge_width_um"])
    parts_out = sorted(finished, key=lambda part: part.area, reverse=True) if cut else [polygon]
    return necks, parts_out


def cut_at_necks(
    polygon: Polygon,
    microns_per_pixel: float,
    max_width_um: float = 100.0,
    min_lobe_area_mm2: float = 0.02,
    min_lobe_fraction: float = 0.02,
) -> list[Polygon]:
    """Split a piece across every neck it contains.

    The shortest line between two sections runs *along* the bridge, so the cut is
    taken perpendicular to it: that is the narrowest place to divide a dumbbell,
    and any other cut removes more tissue.
    """
    _necks, parts = find_necks(
        polygon, microns_per_pixel,
        max_width_um=max_width_um,
        min_lobe_area_mm2=min_lobe_area_mm2,
        min_lobe_fraction=min_lobe_fraction,
        cut=True,
    )
    return parts


def _cut_blade(lobe_a: Polygon, lobe_b: Polygon, radius_px: float, limit=None):
    """A thin polygon spanning the bridge, perpendicular to the lobe-to-lobe line.

    ``limit`` confines the blade to the bridge and a margin around it. Without
    it the blade keeps its full search length on both sides of the midpoint, and
    on a narrow bridge between two broad sections that length reaches well past
    the constriction: the cut then leaves a slit running deep into solid tissue
    and dead-ending there, which is visible on the image as a line drawn across
    a piece that was never separated at that point.
    """
    line = shapely.shortest_line(lobe_a, lobe_b)
    if line is None or line.is_empty:
        return None
    (x0, y0), (x1, y1) = list(line.coords)[0], list(line.coords)[-1]
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 0:
        return None
    # Unit vector along the bridge, then its perpendicular.
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    mid_x, mid_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    # The bridge cannot be wider than the detection width, so a span of a few
    # opening radii always crosses it completely.
    span = 6.0 * max(radius_px, 1.0)
    across = LineString([
        (mid_x - px * span, mid_y - py * span),
        (mid_x + px * span, mid_y + py * span),
    ])
    blade = across.buffer(max(radius_px * 0.02, 0.25))
    if limit is not None and not limit.is_empty:
        clipped = blade.intersection(limit)
        # An empty clip would mean the bridge and the blade do not meet, which
        # only happens when the geometry is degenerate; the unclipped blade is
        # still the better answer there than no cut at all.
        if not clipped.is_empty:
            blade = clipped
    return blade


# ==========================================================================
# Necks, measured on a raster
# ==========================================================================

# Neck detection asks the same question many times over: open the piece with a
# disc of radius r and see whether it falls into two substantial sections. In
# polygon space each of those openings is a pair of GEOS buffers over every
# vertex of the outline, and a hand-drawn annotation carries tens of thousands
# of them, so one piece can take a minute. On a raster the same opening is a
# single morphology call on a small array, which is thousands of times cheaper
# and gives the same answer to within the raster's own step.
#
# Only the *search* moves to the raster. What is finally subtracted from the
# tissue is the cut footprint alone, vectorised and taken out of the original
# polygon, so the outline that reaches the GeoJSON is the one that was drawn -
# exact everywhere except across the few microns of each cut.
RASTER_NECK_UM = 2.0
RASTER_MAX_SIDE = 4000


def _neck_raster_scale(polygon: Polygon, microns_per_pixel: float, raster_um: float) -> float:
    """Full-resolution pixels per raster pixel, capped so the array stays small."""
    scale = max(float(raster_um) / max(float(microns_per_pixel), 1e-9), 1.0)
    minx, miny, maxx, maxy = polygon.bounds
    longest = max(maxx - minx, maxy - miny, 1.0)
    return max(scale, longest / float(RASTER_MAX_SIDE))


def _disc(radius_px: int):
    import cv2

    size = 2 * int(max(radius_px, 1)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _rasterize_piece(polygon: Polygon, scale: float):
    """Binary mask of one piece, plus the offset that maps it back."""
    import numpy as np
    import rasterio.features
    from rasterio.transform import Affine

    minx, miny, maxx, maxy = polygon.bounds
    width = max(int(np.ceil((maxx - minx) / scale)) + 4, 8)
    height = max(int(np.ceil((maxy - miny) / scale)) + 4, 8)
    origin_x, origin_y = minx - 2.0 * scale, miny - 2.0 * scale
    transform = Affine.translation(origin_x, origin_y) * Affine.scale(scale, scale)
    mask = rasterio.features.rasterize(
        [(polygon, 1)], out_shape=(height, width), transform=transform,
        fill=0, all_touched=False, dtype="uint8",
    )
    return mask.astype(bool), (origin_x, origin_y)


def _mask_to_polygons(mask, origin: tuple[float, float], scale: float) -> list[Polygon]:
    """Vectorise a raster footprint back into full-resolution polygon space."""
    import numpy as np
    import rasterio.features
    from rasterio.transform import Affine
    from shapely.geometry import shape as shapely_shape

    if not mask.any():
        return []
    transform = Affine.translation(origin[0], origin[1]) * Affine.scale(scale, scale)
    parts: list[Polygon] = []
    for geometry, value in rasterio.features.shapes(
        mask.astype(np.uint8), mask=mask, transform=transform
    ):
        if not value:
            continue
        parts.extend(_as_polygons(shapely_shape(geometry)))
    return parts


def _raster_lobes(mask, radius_px: int, min_area_px: float, min_fraction: float):
    """Sections that survive an opening and are substantial in their own right."""
    import cv2
    import numpy as np

    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _disc(radius_px))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    parent = float(mask.sum())
    keep = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < float(min_area_px):
            continue
        if parent > 0 and (area / parent) < float(min_fraction):
            continue
        keep.append(index)
    return labels, keep


def _raster_narrowest_split(mask, scale_um: float, max_width_um: float,
                            min_area_px: float, min_fraction: float, tolerance_um: float):
    """Bisect for the width at which two substantial sections come apart."""
    def separates(width_um: float) -> bool:
        radius = int(round((float(width_um) / 2.0) / max(scale_um, 1e-9)))
        if radius < 1:
            return False
        _, keep = _raster_lobes(mask, radius, min_area_px, min_fraction)
        return len(keep) >= 2

    if not separates(max_width_um):
        return None
    low, high = 0.0, float(max_width_um)
    while (high - low) > float(tolerance_um):
        middle = 0.5 * (low + high)
        if separates(middle):
            high = middle
        else:
            low = middle
    return float(high)


def find_necks_raster(
    polygon: Polygon,
    microns_per_pixel: float,
    max_width_um: float = 100.0,
    min_lobe_area_mm2: float = 0.02,
    min_lobe_fraction: float = 0.02,
    tolerance_um: float = 2.0,
    max_necks: int = 12,
    cut: bool = False,
    raster_um: float = RASTER_NECK_UM,
) -> tuple[list[dict[str, Any]], list[Polygon]]:
    """:func:`find_necks` with the search done on a raster. Same return shape."""
    import cv2
    import numpy as np
    from shapely.geometry import Point
    from shapely.ops import unary_union

    scale = _neck_raster_scale(polygon, microns_per_pixel, raster_um)
    scale_um = scale * float(microns_per_pixel)
    mask, origin = _rasterize_piece(polygon, scale)
    if not mask.any():
        return [], [polygon]

    min_area_px = mm2_to_px2(min_lobe_area_mm2, scale_um)
    pixel_mm2 = (scale_um ** 2) / 1e6

    necks: list[dict[str, Any]] = []
    cut_masks = []
    pending = [mask]
    while pending and len(necks) < int(max_necks):
        current = pending.pop()
        width_um = _raster_narrowest_split(
            current, scale_um, max_width_um, min_area_px, min_lobe_fraction, tolerance_um,
        )
        if width_um is None:
            continue

        radius = max(int(round((width_um / 2.0) / max(scale_um, 1e-9))), 1)
        labels, keep = _raster_lobes(current, radius, min_area_px, min_lobe_fraction)
        if len(keep) < 2:
            continue
        keep = sorted(keep, key=lambda index: -(labels == index).sum())
        lobe_mask = np.isin(labels, keep)

        # What holds the sections together is whatever the opening removed and
        # that still touches two different sections.
        residue = current & ~lobe_mask
        strand_count, strand_labels, _, _ = cv2.connectedComponentsWithStats(
            residue.astype(np.uint8), connectivity=8)
        dilate = _disc(1)
        joining = []
        for index in range(1, strand_count):
            strand = strand_labels == index
            probe = cv2.dilate(strand.astype(np.uint8), dilate) > 0
            touched = {value for value in np.unique(labels[probe]) if value in keep}
            if len(touched) >= 2:
                joining.append(strand)

        rows, columns = np.nonzero(residue if joining else current)
        centre_row, centre_col = float(rows.mean()), float(columns.mean())

        # Each strand is cut where it runs closest to the constriction, never
        # along its whole length, which is what keeps the footprint small.
        footprint = np.zeros_like(current)
        for strand in (joining or []):
            strand_rows, strand_cols = np.nonzero(strand)
            distances = (strand_rows - centre_row) ** 2 + (strand_cols - centre_col) ** 2
            nearest = int(np.argmin(distances))
            disc = np.zeros_like(current, dtype=np.uint8)
            cv2.circle(disc, (int(strand_cols[nearest]), int(strand_rows[nearest])),
                       max(int(radius * 3), 2), 1, -1)
            footprint |= strand & (disc > 0)
        if not footprint.any():
            continue

        remainder = current & ~footprint
        part_count, part_labels, part_stats, _ = cv2.connectedComponentsWithStats(
            remainder.astype(np.uint8), connectivity=8)
        areas = sorted((float(part_stats[i, cv2.CC_STAT_AREA]) for i in range(1, part_count)),
                       reverse=True)
        separated = len(areas) >= 2
        side_areas = [area * pixel_mm2 for area in areas[:2]] or [0.0, 0.0]
        if len(side_areas) < 2:
            side_areas.append(0.0)

        necks.append({
            "bridge_width_um": float(width_um),
            "n_strands": int(len(joining)),
            "side_areas_mm2": side_areas,
            "smaller_side_fraction": float(min(side_areas) / max(sum(side_areas), 1e-9)),
            "x": float(origin[0] + centre_col * scale),
            "y": float(origin[1] + centre_row * scale),
            "marker": unary_union(_mask_to_polygons(footprint, origin, scale))
            or Point(origin[0] + centre_col * scale, origin[1] + centre_row * scale).buffer(radius * scale),
            "separated": bool(separated),
            "searched_to_um": float(max_width_um),
        })
        cut_masks.append(_mask_to_polygons(footprint, origin, scale))
        if separated:
            for index in range(1, part_count):
                pending.append(part_labels == index)

    if not cut:
        return necks, [polygon]

    cuts = [part for group in cut_masks for part in group]
    if not cuts:
        return necks, [polygon]
    # The cut is taken out of the polygon that was drawn, so every edge except
    # the cut itself keeps the precision it came in with. The footprint is
    # widened by a pixel and smoothed first: it comes off a raster, so its edge
    # is a staircase, and a staircase laid along a hand-drawn outline grazes it
    # in dozens of places and shaves a sliver off at each one instead of cutting
    # cleanly through.
    blade = unary_union(cuts).buffer(scale).simplify(scale * 0.5)
    parts = _as_polygons(polygon.difference(blade))
    # Anything left at the scale of the raster itself is one of those grazes,
    # not tissue. The user's own size filter runs later and is untouched by this.
    crumb = 4.0 * scale * scale
    parts = [part for part in parts if part.area > crumb]
    return necks, sorted(parts, key=lambda part: part.area, reverse=True) or [polygon]

# ==========================================================================
# Selection
# ==========================================================================

def select_pieces(
    pieces: Sequence[Polygon],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the size filters. Returns ``(kept, rejected)`` records.

    Both lists carry the full metrics plus a ``reason`` on the rejected side, so
    the step can always show why a piece was dropped.
    """
    settings = {**DEFAULT_PIECE_CONFIG, **(config or {})}
    microns_per_pixel = float(settings["microns_per_pixel"])
    min_area_mm2 = float(settings["min_area_mm2"])
    min_diameter_um = float(settings["min_inscribed_diameter_um"])
    use_area = bool(settings["filter_by_area"])
    use_circle = bool(settings["filter_by_inscribed_circle"])

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for polygon in pieces:
        metrics = piece_metrics(polygon, microns_per_pixel)
        reasons = []
        if use_area and metrics["area_mm2"] < min_area_mm2:
            reasons.append(f"area {metrics['area_mm2']:.4f} mm2 < {min_area_mm2:g} mm2")
        if use_circle and metrics["inscribed_d_um"] < min_diameter_um:
            reasons.append(
                f"widest circle {metrics['inscribed_d_um']:.0f} um < {min_diameter_um:g} um"
            )
        record = {"geom": polygon, **metrics}
        if reasons:
            record["reason"] = "; ".join(reasons)
            rejected.append(record)
        else:
            kept.append(record)

    for index, record in enumerate(kept, start=1):
        record["piece_id"] = index
    for index, record in enumerate(rejected, start=1):
        record["piece_id"] = -index
    return kept, rejected


def analyse_tissue_pieces(geometry, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full pass: split, optionally cut necks, measure, filter.

    Returns ``{"kept", "rejected", "necks", "settings"}``. ``kept`` and
    ``rejected`` are metric records carrying their ``geom``; ``necks`` lists the
    bridges found, whether or not cutting was enabled.
    """
    settings = {**DEFAULT_PIECE_CONFIG, **(config or {})}
    microns_per_pixel = float(settings["microns_per_pixel"])
    fill_px2 = float(settings.get("fill_holes_below_um2", 0.0)) / max(microns_per_pixel ** 2, 1e-9)

    pieces = split_pieces(
        geometry,
        simplify_px=float(settings.get("simplify_px", 0.0)),
        fill_holes_below_px2=fill_px2,
    )

    necks: list[dict[str, Any]] = []
    if bool(settings.get("detect_necks", True)):
        should_cut = bool(settings.get("cut_at_necks", False))
        resolved: list[Polygon] = []
        for source_index, polygon in enumerate(pieces, start=1):
            finder = (
                find_necks_raster
                if str(settings.get("neck_method", "raster")).lower() == "raster"
                else find_necks
            )
            found, parts = finder(
                polygon, microns_per_pixel,
                max_width_um=float(settings["neck_min_width_um"]),
                min_lobe_area_mm2=float(settings["neck_min_lobe_area_mm2"]),
                min_lobe_fraction=float(settings["neck_min_lobe_fraction"]),
                cut=should_cut,
            )
            for neck in found:
                neck["source_piece"] = source_index
                neck["area_mm2"] = px2_to_mm2(polygon.area, microns_per_pixel)
            necks.extend(found)
            resolved.extend(parts)
        pieces = sorted(resolved, key=lambda part: part.area, reverse=True)

    kept, rejected = select_pieces(pieces, settings)
    return {"kept": kept, "rejected": rejected, "necks": necks, "settings": settings}
