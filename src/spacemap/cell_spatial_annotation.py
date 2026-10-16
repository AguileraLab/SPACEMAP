"""Tag segmented cells with the tissue annotations they fall in.

Given a cell table with centroid coordinates and a QuPath GeoJSON holding tissue
domains (``tissue``, ``Epithelia``, ``Stroma``, ``Tumor``, ``TLS``, zones, ...),
this module adds the spatial columns the downstream pipeline expects:

``<label>``
    1 when the cell centroid falls inside that annotation, else 0.
``<label>_roi``
    1-based index of the individual polygon of that annotation containing the
    cell, so cells can be grouped by tissue piece rather than by class. Under
    "group by tissue ROI" the numbering is split per tissue piece, so one
    polygon spanning two pieces yields a different id on each side and cells
    from different pieces are never pooled under a shared id.
``<label>_roi_size``
    Area of that containing polygon in square pixels, clipped to the tissue
    piece whenever the numbering is split.
``<label>_distance``
    Signed distance from the cell to the nearest polygon of that annotation:
    negative inside, positive outside.
``<label>_bed`` / ``<target>_bed`` (+ ``_bed_id``)
    Proximity competition between annotations: each cell is assigned to the
    closest qualifying class. Cells outside the containing tissue, or in a
    tissue piece without a qualifying target, receive ``NaN`` for all beds.
``edge_cell``
    1 when the cell lies within a buffer of a reference annotation's boundary.

The same implementation backs three callers, so a cell table produced by the
Segmentation tab, by the legacy cell-annotation subtab, or by a notebook is
annotated identically:

* :func:`annotate_cells` - operates on an in-memory DataFrame (used inline by
  the segmentation pipeline, right after feature extraction).
* :func:`annotate_cell_csv` - file-in/file-out wrapper.
* :func:`rules_from_records` - builds rules from the tabular UI editors.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

__all__ = [
    "SpatialFeatureRule",
    "EdgeTagConfig",
    "DEFAULT_TISSUE_LABEL",
    "DEFAULT_MIN_CELLS_PER_BED_TARGET",
    "DEFAULT_CLEAN_PATH_TOLERANCE_PX",
    "DEFAULT_CLEAN_PATH_MAX_CANDIDATES",
    "list_annotation_classes",
    "load_annotation_polygons",
    "default_rules_for_classes",
    "rules_from_records",
    "rules_to_records",
    "annotate_cells",
    "annotate_cell_csv",
    "resolve_coordinate_columns",
    "spatial_columns_for_rules",
]

DEFAULT_TISSUE_LABEL = "tissue"
DEFAULT_MIN_CELLS_PER_BED_TARGET = 1

# Clean-path defaults. The tolerance absorbs floating-point noise on lines that
# graze the tissue outline; one pixel is well below any biological scale. The
# candidate cap bounds the widening search for cells whose nearest annotation
# is hidden behind a gap.
DEFAULT_CLEAN_PATH_TOLERANCE_PX = 1.0
DEFAULT_CLEAN_PATH_MAX_CANDIDATES = 8
CLEAN_PATH_ENGINE = "Triangle visibility (CPU)"

# Cap on candidate pairs returned by an indexed fallback query. Even if every
# region lies within the search radius, one query stays within this budget.
_CLEAN_PATH_MATRIX_CELLS = 4_000_000

# Zone layers produced by the annotation step. They are useful as binary
# membership columns but rarely as distance or bed targets.
ZONE_PREFIXES = ("inner_", "outer_")


@dataclass
class SpatialFeatureRule:
    """One row of the spatial feature configuration matrix."""

    label: str
    binary: bool = True
    roi_id: bool = False
    roi_size: bool = False
    distance: bool = False
    bed: bool = False
    bed_target: str = ""
    group_by_roi: bool = False

    def is_active(self) -> bool:
        return bool(self.binary or self.roi_id or self.roi_size or self.distance or self.bed)

    def required_labels(self) -> set[str]:
        labels = {self.label.lower()}
        if self.bed and self.bed_target:
            labels.add(str(self.bed_target).lower())
        return labels


@dataclass
class EdgeTagConfig:
    """Flag cells lying within ``buffer_px`` of a reference annotation's edge."""

    reference: str = DEFAULT_TISSUE_LABEL
    buffer_px: float = 100.0
    column: str = "edge_cell"


# ==========================================================================
# GeoJSON reading
# ==========================================================================

def _feature_class_name(feature: dict) -> str | None:
    properties = feature.get("properties", {}) or {}
    classification = properties.get("classification")
    if isinstance(classification, dict) and classification.get("name"):
        return str(classification["name"])
    if isinstance(classification, str) and classification:
        return classification
    for key in ("label", "name"):
        if properties.get(key):
            return str(properties[key])
    return None


def _iter_features(geojson_path: str | os.PathLike) -> list[dict]:
    with open(str(geojson_path), "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    features = data.get("features", [])
    if not features and data.get("type") == "Feature":
        features = [data]
    return features


def list_annotation_classes(geojson_path: str | os.PathLike) -> list[str]:
    """Return the classification names present in an annotation GeoJSON."""
    try:
        names = {_feature_class_name(feature) for feature in _iter_features(geojson_path)}
    except Exception:
        return []
    return sorted(name for name in names if name)


def load_annotation_polygons(geojson_path: str | os.PathLike) -> dict[str, list]:
    """Group valid polygons by lower-cased classification name."""
    grouped: dict[str, list] = {}
    for feature in _iter_features(geojson_path):
        name = _feature_class_name(feature)
        geometry_dict = feature.get("geometry")
        if not name or not geometry_dict:
            continue
        try:
            geometry = shape(geometry_dict)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        grouped.setdefault(name.lower(), []).append(geometry)
    return grouped


def _split_and_sort(geometry) -> list[dict[str, Any]]:
    """Split a (Multi)Polygon into parts ordered by descending area.

    The 1-based position in this list becomes the ``_roi`` identifier, so the
    largest tissue piece is always ROI 1 and the numbering is reproducible.
    """
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        parts = [geometry]
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for part in geometry.geoms:
            if part.geom_type == "Polygon":
                parts.append(part)
            elif part.geom_type == "MultiPolygon":
                parts.extend(list(part.geoms))
    else:
        return []

    parts = [part for part in parts if not part.is_empty and part.area > 0]
    parts.sort(key=lambda part: part.area, reverse=True)
    return [
        {"id": index + 1, "geom": part, "area_px2": float(part.area)}
        for index, part in enumerate(parts)
    ]


def _prepare_targets(polygons_by_name: dict[str, list], labels: Iterable[str], progress=None) -> dict[str, dict]:
    """Merge, split and index the polygons for every label a rule references."""
    prepared: dict[str, dict] = {}
    names = sorted({str(value).lower() for value in labels})
    for number, label in enumerate(names):
        polygons = polygons_by_name.get(label, [])
        if progress is not None:
            progress(f"Preparing annotation {number + 1}/{len(names)}: {label} ({len(polygons):,} features)",
                     .02 + .08 * number / max(len(names), 1))
        if not polygons:
            prepared[label] = {"parts": [], "tree": None}
            continue
        merged = polygons[0] if len(polygons) == 1 else unary_union(polygons)
        parts = _split_and_sort(merged)
        geometries = [part["geom"] for part in parts]
        prepared[label] = {
            "parts": parts,
            "tree": STRtree(geometries) if geometries else None,
        }
    return prepared


# ==========================================================================
# Rules
# ==========================================================================

def default_rules_for_classes(
    classes: Sequence[str],
    tissue_label: str = DEFAULT_TISSUE_LABEL,
) -> list[SpatialFeatureRule]:
    """Sensible starting rules: membership for everything, ROI/size/distance for tissue."""
    rules: list[SpatialFeatureRule] = []
    for name in classes:
        is_tissue = str(name).lower() == str(tissue_label).lower()
        rules.append(SpatialFeatureRule(
            label=str(name),
            binary=True,
            roi_id=is_tissue,
            roi_size=is_tissue,
            distance=is_tissue,
        ))
    return rules


_RECORD_FIELDS = {
    "Annotation Label": "label",
    "Binary (0/1)": "binary",
    "ROI ID": "roi_id",
    "ROI Size": "roi_size",
    "Distance": "distance",
    "Bed (Use 2 Annotations)": "bed",
    "Bed Comp. Target": "bed_target",
    "Group by Tissue ROI": "group_by_roi",
}


def rules_from_records(records: Iterable[dict[str, Any]]) -> list[SpatialFeatureRule]:
    """Build rules from the column names used by the tabular UI editors."""
    rules: list[SpatialFeatureRule] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        label = record.get("Annotation Label", record.get("label", ""))
        if not label or (isinstance(label, float) and pd.isna(label)):
            continue
        values: dict[str, Any] = {"label": str(label)}
        for column, attribute in _RECORD_FIELDS.items():
            if attribute == "label":
                continue
            if column in record:
                raw = record[column]
            elif attribute in record:
                raw = record[attribute]
            else:
                continue
            if attribute == "bed_target":
                values[attribute] = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)
            else:
                values[attribute] = bool(raw)
        rules.append(SpatialFeatureRule(**values))
    return rules


def rules_to_records(rules: Sequence[SpatialFeatureRule]) -> list[dict[str, Any]]:
    """Inverse of :func:`rules_from_records`, for round-tripping through the UI."""
    return [
        {
            "Annotation Label": rule.label,
            "Binary (0/1)": rule.binary,
            "ROI ID": rule.roi_id,
            "ROI Size": rule.roi_size,
            "Distance": rule.distance,
            "Bed (Use 2 Annotations)": rule.bed,
            "Bed Comp. Target": rule.bed_target,
            "Group by Tissue ROI": rule.group_by_roi,
        }
        for rule in rules
    ]


def spatial_columns_for_rules(
    rules: Sequence[SpatialFeatureRule],
    edge_config: EdgeTagConfig | None = None,
    bed_labels: Sequence[str] | None = None,
) -> list[str]:
    """Column names :func:`annotate_cells` will add, for previewing in the UI."""
    columns: list[str] = []
    for rule in rules:
        if rule.binary:
            columns.append(rule.label)
        if rule.roi_id:
            columns.append(f"{rule.label}_roi")
        if rule.roi_size:
            columns.append(f"{rule.label}_roi_size")
        if rule.distance:
            columns.append(f"{rule.label}_distance")
        if bed_labels is None and rule.bed and rule.bed_target:
            columns.extend([
                f"{rule.label}_bed", f"{rule.bed_target}_bed",
                f"{rule.label}_bed_id", f"{rule.bed_target}_bed_id",
            ])
    if bed_labels is not None:
        for label in _normalize_bed_labels(bed_labels):
            columns.extend([f"{label}_bed", f"{label}_bed_id"])
    if edge_config is not None:
        columns.append(edge_config.column)
    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            unique.append(column)
    return unique


def _normalize_bed_labels(labels: Sequence[str]) -> list[str]:
    """Deduplicate class names without changing their output-column spelling."""
    names: dict[str, str] = {}
    for label in labels:
        name = str(label).strip()
        if name:
            names.setdefault(name.lower(), name)
    return [names[key] for key in sorted(names)]


# ==========================================================================
# Coordinate resolution
# ==========================================================================

def resolve_coordinate_columns(
    dataframe: pd.DataFrame,
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[str | None, str | None]:
    """Find the centroid columns, preferring an explicit choice."""
    columns = list(dataframe.columns)

    def _find(preferred: str | None, axis: str) -> str | None:
        if preferred and preferred in columns:
            return preferred
        exact = [f"{axis}_centroid", f"centroid_{axis}", f"center_{axis}", axis]
        for candidate in exact:
            for column in columns:
                if str(column).lower() == candidate:
                    return column
        for column in columns:
            lowered = str(column).lower()
            if lowered.startswith(axis) and ("centroid" in lowered or "center" in lowered):
                return column
        return None

    return _find(x_col, "x"), _find(y_col, "y")


# ==========================================================================
# Core annotation
# ==========================================================================

def _signed_distance_to_regions(
    points,
    regions: Sequence[dict[str, Any]],
    boundary_engine=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest signed distance and owning ROI ID for polygon regions."""
    if not regions or len(points) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    target_geometries = [region["geom"] for region in regions]
    target_ids = np.asarray([int(region["id"]) for region in regions], dtype=int)
    tree = STRtree(target_geometries)
    nearest, nearest_distances = tree.query_nearest(points, return_distance=True)
    if nearest.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    point_index, target_index = nearest[0], nearest[1]
    # query_nearest can return several ties per point; keep the first.
    _, first_unique = np.unique(point_index, return_index=True)
    point_index = point_index[first_unique]
    target_index = target_index[first_unique]

    matched_points = points[point_index]
    geometry_array = np.array(target_geometries, dtype=object)
    shapely.prepare(geometry_array)
    matched_geometries = geometry_array[target_index]

    inside = shapely.contains(matched_geometries, matched_points)
    distances = nearest_distances[first_unique].copy()
    boundaries = np.array([geometry.boundary for geometry in target_geometries], dtype=object)
    distances[inside] = -_boundary_distances(
        boundaries, target_index[inside], matched_points[inside], boundary_engine
    )
    return point_index, distances, target_ids[target_index]


def _boundary_distances(boundaries, owners, points, engine=None):
    if engine is not None:
        return engine.distances(boundaries, owners, points)
    return shapely.distance(boundaries[owners], points)


def _nearest_region_candidates(points, geometries, tree, nearest_distances, limit):
    """Find the exact nearest K regions without an all-pairs distance matrix.

    Expand indexed distance queries until each point has K candidates. All
    regions outside that radius are farther away, so ranking the returned
    candidates gives the same nearest distances as a full search. Equal
    distances are ordered by region index for reproducibility.
    """
    limit = min(max(int(limit), 1), len(geometries))
    ranking = np.full((len(points), limit), -1, dtype=int)
    if not len(points) or not limit:
        return ranking
    xmin, ymin, xmax, ymax = shapely.total_bounds(geometries)
    xy = shapely.get_coordinates(points)
    # A radius reaching every corner of the target bounds guarantees completion
    # even for a point very far from the annotations.
    maximum = np.hypot(
        np.maximum(abs(xy[:, 0] - xmin), abs(xy[:, 0] - xmax)),
        np.maximum(abs(xy[:, 1] - ymin), abs(xy[:, 1] - ymax)),
    ) + 1.0
    radius = np.minimum(np.maximum(nearest_distances * 2.0, 1.0), maximum)
    pending = np.arange(len(points))
    while pending.size:
        hits = tree.query(points[pending], predicate="dwithin", distance=radius[pending])
        counts = np.bincount(hits[0], minlength=len(pending))
        ready = (counts >= limit) | (radius[pending] >= maximum[pending])
        if np.any(ready):
            selected = ready[hits[0]]
            point_indices = pending[hits[0][selected]]
            region_indices = hits[1][selected]
            distances = shapely.distance(geometries[region_indices], points[point_indices])
            order = np.lexsort((region_indices, distances, point_indices))
            point_indices, region_indices = point_indices[order], region_indices[order]
            if point_indices.size:
                starts = np.r_[0, np.flatnonzero(np.diff(point_indices)) + 1]
                ranks = np.arange(len(point_indices)) - np.repeat(
                    starts, np.diff(np.r_[starts, len(point_indices)])
                )
                keep = ranks < limit
                ranking[point_indices[keep], ranks[keep]] = region_indices[keep]
        pending = pending[~ready]
        radius[pending] = np.minimum(radius[pending] * 2.0, maximum[pending])
    return ranking


def _nearest_visible_distance(
    points,
    regions: Sequence[dict[str, Any]],
    container_geometry,
    tolerance_px: float,
    max_candidates: int,
    probe_cache: dict | None = None,
    boundary_engine=None,
):
    """Signed distance to the nearest region reachable by a clean straight line.

    A distance is only reported when the segment joining the cell to the
    annotation stays inside ``container_geometry`` - the tissue piece the cell
    sits in. That rejects the two ways a straight line lies about proximity: it
    cannot hop the glass to another scrap, and it cannot cut across the mouth of
    a U-shaped piece. Cells with no clean line to any region are left unset,
    because the honest answer there is "no measurement", not a longer one.

    The nearest region is checked first, which settles most cells; only the
    cells it fails for pay for a wider search over the remaining regions.

    Returns ``(positions, distances)`` for the cells that got a clean line.
    """
    total = len(points)
    if total == 0 or not regions or container_geometry is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    geometries = np.array([region["geom"] for region in regions], dtype=object)
    boundaries = np.array([geometry.boundary for geometry in geometries], dtype=object)
    shapely.prepare(geometries)
    tree = STRtree(geometries)
    # Buffering a detailed tissue piece is costly. Reuse it across annotation
    # classes within this run, and do not build it for cells already inside a
    # target. A run-local cache cannot become stale after a GeoJSON edit.
    if probe_cache is None:
        probe_cache = {}

    def get_probe():
        key = (id(container_geometry), float(tolerance_px))
        if key not in probe_cache:
            probe = (container_geometry.buffer(float(tolerance_px))
                     if tolerance_px > 0 else container_geometry)
            shapely.prepare(probe)
            probe_cache[key] = probe
        return probe_cache[key]

    def evaluate(point_positions, region_positions):
        """Signed distance plus whether the connecting line stays in tissue."""
        selected_points = points[point_positions]
        selected_geometries = geometries[region_positions]
        inside = shapely.contains(selected_geometries, selected_points)
        distances = np.empty(len(selected_points), dtype=float)
        clean = inside.copy()
        distances[inside] = -_boundary_distances(
            boundaries, region_positions[inside], selected_points[inside], boundary_engine
        )
        outside = ~inside
        if np.any(outside):
            distances[outside] = shapely.distance(selected_geometries[outside], selected_points[outside])
            lines = shapely.shortest_line(selected_geometries[outside], selected_points[outside])
            clean[outside] = shapely.covers(get_probe(), lines)
        return distances, clean

    resolved = np.full(total, np.nan, dtype=float)

    # Pass 1: the nearest region, which is the answer for the great majority.
    nearest_distances = np.full(total, np.nan)
    nearest, queried_distances = tree.query_nearest(points, return_distance=True)
    if nearest.size:
        point_index, target_index = nearest[0], nearest[1]
        _, first_unique = np.unique(point_index, return_index=True)
        point_index = point_index[first_unique]
        target_index = target_index[first_unique]
        nearest_distances[point_index] = queried_distances[first_unique]
        distances, clean = evaluate(point_index, target_index)
        resolved[point_index[clean]] = distances[clean]

    # Pass 2: only the cells whose nearest region was blocked. Walk outwards
    # through the remaining regions and take the first one in clear view.
    blocked = np.where(np.isnan(resolved) & np.isfinite(nearest_distances))[0]
    limit = max(int(max_candidates), 1)
    if blocked.size and len(regions) > 1 and limit > 1:
        chunk = min(4096, max(int(_CLEAN_PATH_MATRIX_CELLS // len(regions)), 1))
        for start in range(0, blocked.size, chunk):
            block = blocked[start:start + chunk]
            ranking = _nearest_region_candidates(
                points[block], geometries, tree, nearest_distances[block], limit
            )
            unsettled = np.ones(len(block), dtype=bool)
            for rank in range(ranking.shape[1]):
                candidates = np.where(unsettled & (ranking[:, rank] >= 0))[0]
                if candidates.size == 0:
                    break
                distances, clean = evaluate(block[candidates], ranking[candidates, rank])
                accepted = candidates[clean]
                resolved[block[accepted]] = distances[clean]
                unsettled[accepted] = False

    positions = np.where(~np.isnan(resolved))[0]
    return positions, resolved[positions]


def _clipped_parts(prepared: dict, container_geometry) -> list[dict[str, Any]]:
    """Clip annotation polygons to one tissue piece and split disconnected parts.

    ``container_geometry=None`` means that a cell is outside the selected
    containing-tissue annotation. It deliberately returns no targets: grouped
    rules must never fall back to the full slide for those cells.
    """
    parts = prepared.get("parts", [])
    if not parts or container_geometry is None or prepared.get("tree") is None:
        return []

    selected: list[dict[str, Any]] = []
    shapely.prepare(container_geometry)
    # Source-part order is stable because ``_prepare_targets`` area-sorts it.
    for candidate_index in sorted(int(value) for value in prepared["tree"].query(container_geometry)):
        part = parts[candidate_index]
        geometry = part["geom"]
        if container_geometry.covers(geometry):
            # Already inside: retain the exact polygon, including every hole.
            # Only crossing polygons need the expensive intersection operation.
            if float(part["area_px2"]) > 1.0:
                selected.append({
                    "geom": geometry, "area_px2": float(part["area_px2"]),
                    "source_part_id": int(part["id"]),
                })
            continue
        if not container_geometry.intersects(geometry):
            continue
        try:
            clipped = container_geometry.intersection(geometry)
        except Exception:
            continue
        if clipped.is_empty or clipped.area <= 1.0:
            continue
        if not clipped.is_valid:
            clipped = make_valid(clipped)
        for clipped_part in _split_and_sort(clipped):
            selected.append({
                "geom": clipped_part["geom"],
                "area_px2": float(clipped_part["area_px2"]),
                "source_part_id": int(part["id"]),
            })
    return selected


def _scoped_regions(
    prepared: dict,
    containment: Sequence[tuple[int, Any, list[int]]],
    group_by_roi: bool,
) -> dict[int, list[dict[str, Any]]]:
    """Return stable annotation regions for every containing tissue piece.

    Grouped regions are clipped and receive IDs unique across the image. Thus
    one GeoJSON polygon crossing two tissue pieces becomes two independent ROI
    and bed IDs. Ungrouped rules retain the original full-slide polygon IDs.
    """
    if not group_by_roi:
        regions = [
            {
                "id": int(part["id"]),
                "geom": part["geom"],
                "area_px2": float(part["area_px2"]),
                "source_part_id": int(part["id"]),
            }
            for part in prepared.get("parts", [])
        ]
        return {int(tissue_id): regions for tissue_id, _geometry, _positions in containment}

    scoped: dict[int, list[dict[str, Any]]] = {}
    next_id = 1
    for tissue_id, container_geometry, _positions in containment:
        fragments = _clipped_parts(prepared, container_geometry)
        regions: list[dict[str, Any]] = []
        for fragment in fragments:
            region = dict(fragment)
            region["id"] = next_id
            next_id += 1
            regions.append(region)
        scoped[int(tissue_id)] = regions
    return scoped


def _point_region_pairs(points, geometries, tree=None):
    """Boundary-inclusive hits using prepared polygons, rather than points.

    Querying an STRtree with predicate='intersects' prepares the query points.
    Here the tree selects bounding-box candidates and the expensive predicate
    uses the prepared polygon on the left. This reuses its internal index for
    thousands of cells in a complex polygon with many holes.
    """
    if not len(points) or not len(geometries):
        return np.empty((2, 0), dtype=int)
    geometries = np.asarray(geometries, dtype=object)
    shapely.prepare(geometries)
    if tree is None:
        tree = STRtree(geometries)
    chunk = min(8192, max(_CLEAN_PATH_MATRIX_CELLS // len(geometries), 1))
    pieces = []
    for start in range(0, len(points), chunk):
        subset = points[start:start + chunk]
        hits = tree.query(subset)
        if hits.size:
            keep = shapely.intersects(geometries[hits[1]], subset[hits[0]])
            hits = hits[:, keep]
            hits[0] += start
            pieces.append(hits)
    return np.concatenate(pieces, axis=1) if pieces else np.empty((2, 0), dtype=int)


def _region_hits(points, regions: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Unique point/region matches, including points exactly on a boundary."""
    if not regions or len(points) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    hits = _point_region_pairs(points, [region["geom"] for region in regions])
    if hits.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    point_positions, first = np.unique(hits[0], return_index=True)
    return point_positions.astype(int), hits[1][first].astype(int)


def _regions_with_min_cells(
    points,
    regions: Sequence[dict[str, Any]],
    minimum_cells: int,
) -> tuple[list[dict[str, Any]], int]:
    """Filter polygon fragments by contained cell-centroid count."""
    if not regions:
        return [], 0
    hits = _point_region_pairs(points, [region["geom"] for region in regions])
    counts = np.zeros(len(regions), dtype=int)
    if hits.size:
        counts = np.bincount(hits[1], minlength=len(regions))
    threshold = max(int(minimum_cells), 1)
    kept = [region for index, region in enumerate(regions) if counts[index] >= threshold]
    return kept, int(len(regions) - len(kept))


def annotate_cells(
    dataframe: pd.DataFrame,
    geojson_path: str | os.PathLike,
    rules: Sequence[SpatialFeatureRule],
    x_col: str | None = None,
    y_col: str | None = None,
    tissue_label: str = DEFAULT_TISSUE_LABEL,
    edge_config: EdgeTagConfig | None = None,
    min_cells_per_bed_target: int = DEFAULT_MIN_CELLS_PER_BED_TARGET,
    require_clean_path: bool = False,
    clean_path_tolerance_px: float = DEFAULT_CLEAN_PATH_TOLERANCE_PX,
    clean_path_max_candidates: int = DEFAULT_CLEAN_PATH_MAX_CANDIDATES,
    bed_labels: Sequence[str] | None = None,
    overwrite_existing: bool = True,
    use_gpu: bool = False,
    cpu_workers: int = 0,
    independent_distances: bool = False,
    progress=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add spatial annotation columns to a cell table.

    Coordinates are interpreted in full-resolution image pixels, the same space
    the GeoJSON polygons live in.

    ``require_clean_path`` uses continuous triangle visibility for grouped
    distances and beds. Selected distance/bed annotations and tissue boundaries
    block the line; a cell inside a selected annotation stops at its own border
    and receives a negative distance. Hidden classes remain blank. Geometry is
    prepared once per tissue ROI and reused across classes and QC thresholds.
    Ungrouped rules retain their global Euclidean behavior. The legacy
    ``clean_path_tolerance_px`` and ``clean_path_max_candidates`` arguments are
    retained for caller compatibility; the triangle engine uses exact tissue
    boundaries and has no target-candidate cap.

    Returns ``(annotated_dataframe, summary)`` where ``summary`` reports the
    columns that were added and how many cells fell in each annotation.

    ``bed_labels`` selects a single competition between two or more classes,
    always scoped to the containing tissue piece. An empty list disables beds;
    None retains the legacy pair rules. Equal distances favor the last class
    alphabetically (case-insensitive), matching the existing two-class rule.
    ``overwrite_existing`` replaces complete generated columns, including NaNs.
    When False, existing columns are retained in full and only new ones added.
    ``independent_distances`` measures each class independently of the other
    annotation classes, while retaining their shared blocking topology for beds.
    ``cpu_workers`` controls exact CPU visibility traversal (0: Auto, up to 8).
    ``progress(message, fraction=None)`` reports work stages and completed
    batches. Fractions track stages, not a prediction of remaining time.
    """
    dataframe = dataframe.copy()
    tell = progress if progress is not None else lambda message, fraction=None: None
    summary: dict[str, Any] = {"columns": [], "counts": {}, "warnings": []}
    workers = 1
    if require_clean_path:
        from spacemap.spatial_visibility import cpu_workers as resolve_workers

        workers = resolve_workers(cpu_workers)
    summary["cpu_workers"] = workers
    boundary_engine = None
    if use_gpu:
        from spacemap.spatial_distance_acceleration import BoundaryDistanceEngine

        boundary_engine = BoundaryDistanceEngine(requested=True)
    minimum_bed_cells = max(int(min_cells_per_bed_target), 1)
    summary["min_cells_per_bed_target"] = minimum_bed_cells
    if bed_labels is not None:
        bed_labels = _normalize_bed_labels(bed_labels)
        if len(bed_labels) == 1:
            raise ValueError("Select at least two bed annotations, or none to disable beds.")
    output_columns = spatial_columns_for_rules(rules, edge_config, bed_labels)
    existing_columns = [name for name in output_columns if name in dataframe.columns]
    preserved = dataframe[existing_columns].copy() if not overwrite_existing else None
    summary["columns_overwritten"] = existing_columns if overwrite_existing else []
    summary["columns_preserved"] = [] if overwrite_existing else existing_columns
    if dataframe.empty:
        summary["warnings"].append("The cell table is empty.")
        return dataframe, summary

    resolved_x, resolved_y = resolve_coordinate_columns(dataframe, x_col, y_col)
    if not resolved_x or not resolved_y:
        raise ValueError(
            "Could not find centroid columns in the cell table. "
            f"Looked for x/y centroids among: {list(dataframe.columns)[:12]}"
        )
    summary["x_col"] = resolved_x
    summary["y_col"] = resolved_y

    active_rules = [rule for rule in rules if rule.is_active()]
    required: set[str] = {str(tissue_label).lower()}
    for rule in active_rules:
        required |= rule.required_labels()
    if bed_labels is not None:
        required.update(label.lower() for label in bed_labels)
    if edge_config is not None:
        required.add(str(edge_config.reference).lower())

    tell(f"Reading annotation polygons for {len(dataframe):,} cells", 0)
    polygons_by_name = load_annotation_polygons(geojson_path)
    missing = sorted(label for label in required if not polygons_by_name.get(label))
    if missing:
        summary["warnings"].append(
            "Annotation(s) not found in the GeoJSON: " + ", ".join(missing)
        )
    prepared = _prepare_targets(polygons_by_name, required, progress=progress)
    tell(f"Assigning {len(dataframe):,} cells to tissue pieces", .10)

    points = shapely.points(
        pd.to_numeric(dataframe[resolved_x], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(dataframe[resolved_y], errors="coerce").to_numpy(dtype=float),
    )
    row_index = dataframe.index

    # Group cells by the tissue piece that contains them. Rules with
    # "group by tissue ROI" are evaluated within each group. The outside group
    # is retained so ungrouped rules can still operate globally; grouped rules
    # deliberately see no targets there.
    containment: list[tuple[int, Any, list[int]]] = []
    tissue_prepared = prepared.get(str(tissue_label).lower(), {"parts": [], "tree": None})
    assigned: set[int] = set()
    if tissue_prepared["parts"] and tissue_prepared["tree"] is not None:
        hits = _point_region_pairs(
            points, [part["geom"] for part in tissue_prepared["parts"]], tissue_prepared["tree"]
        )
        if hits.size > 0:
            grouped: dict[int, list[int]] = {}
            for point_position, part_position in zip(hits[0], hits[1]):
                position = int(point_position)
                if position in assigned:
                    continue  # a cell belongs to one tissue piece only
                assigned.add(position)
                grouped.setdefault(int(part_position), []).append(position)
            for part_position, positions in grouped.items():
                part = tissue_prepared["parts"][part_position]
                containment.append((int(part["id"]), part["geom"], positions))
    leftover = [position for position in range(len(points)) if position not in assigned]
    if leftover:
        containment.append((0, None, leftover))
    # Tissue piece 1 first, cells outside every piece last, so the per-piece ROI
    # numbering below depends only on the geometry, not on row order.
    containment.sort(key=lambda group: (group[0] == 0, group[0]))
    tell(f"Tissue assignment ready: {sum(group[0] != 0 for group in containment):,} pieces; {len(leftover):,} cells outside tissue", .15)

    scoped_cache: dict[tuple[str, bool], dict[int, list[dict[str, Any]]]] = {}
    visibility_probes: dict = {}

    def regions_for(label: str, group_by_roi: bool) -> dict[int, list[dict[str, Any]]]:
        key = (str(label).lower(), bool(group_by_roi))
        if key not in scoped_cache:
            scoped_cache[key] = _scoped_regions(
                prepared.get(key[0], {"parts": [], "tree": None}),
                containment,
                bool(group_by_roi),
            )
        return scoped_cache[key]

    # All selected classes share one first-hit topology. Only requested
    # distance/bed targets block; binary-only parent and stroma layers do not.
    visibility_labels: set[str] = set()
    visibility_beds_requested = False
    if require_clean_path:
        if not independent_distances:
            visibility_labels.update(
                rule.label.lower() for rule in active_rules
                if rule.distance and rule.group_by_roi and rule.label.lower() != str(tissue_label).lower()
            )
        if bed_labels is not None:
            visibility_labels.update(label.lower() for label in bed_labels)
            visibility_beds_requested = bool(bed_labels)
        else:
            for rule in active_rules:
                if rule.bed and rule.bed_target and rule.group_by_roi:
                    visibility_labels.update((rule.label.lower(), str(rule.bed_target).lower()))
                    visibility_beds_requested = True
    visibility_results: dict[tuple, Any] = {}
    summary["distance_method"] = "Euclidean / legacy"

    def visible_values(tissue_id, container, positions, label, threshold, independent=False):
        """Return one class from a shared, run-local tissue visibility result."""
        if container is None:
            return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int), 0, False
        names = (label,) if independent else tuple(sorted(visibility_labels))
        key = (tissue_id, names)
        if key not in visibility_results:
            from spacemap.spatial_visibility import TissueVisibility, ENGINE_NAME

            started = time.perf_counter()
            context = f"{', '.join(names)} · tissue {tissue_id}"
            detailed = progress is not None and len(positions) >= 2048
            def scene_progress(message):
                tell(f"{context}: {message}")

            if detailed:
                scene_progress(f"Preparing boundary index for {len(positions):,} cells")
            targets = {name: regions_for(name, True).get(int(tissue_id), [])
                       for name in names}
            scene = TissueVisibility(container, targets, workers=workers)
            measurements, skipped = scene.measure(
                points[np.asarray(positions)], (1, minimum_bed_cells) if visibility_beds_requested and not independent else (1,),
                hit_query=_point_region_pairs, progress=scene_progress if detailed else None
            )
            if detailed:
                scene_progress(f"Distances finished ({time.perf_counter() - started:.1f}s)")
            visibility_results[key] = (scene.labels, measurements, skipped,
                                             {name: len(regions) for name, regions in targets.items()})
            summary["distance_method"] = ENGINE_NAME
            summary.setdefault("visibility_tissue_rois", []).append({"tissue_roi": int(tissue_id), **scene.metadata})
            # Keep only the cell results. Large mesh/index arrays are released
            # before preparing the next tissue piece.
        labels, measurements, skipped, sizes = visibility_results[key]
        index = labels.index(label)
        distances, owners = measurements[threshold]
        matched = np.flatnonzero(np.isfinite(distances[:, index]))
        excluded = skipped[threshold][label]
        return matched, distances[matched, index], owners[matched, index], excluded, sizes[label] > excluded

    # Membership, ROI and distance columns are generated once per rule. Beds
    # are handled in a separate de-duplicated pass below.
    for rule_number, rule in enumerate(active_rules):
        started = time.perf_counter()
        tell(f"Annotation {rule_number + 1}/{len(active_rules)}: {rule.label} · membership and region measurements",
             .15 + .60 * rule_number / max(len(active_rules), 1))
        label_lower = rule.label.lower()
        target = prepared.get(label_lower, {"parts": [], "tree": None})
        has_target = bool(target["parts"]) and target["tree"] is not None
        scoped = regions_for(label_lower, rule.group_by_roi)

        if rule.binary:
            column = rule.label
            dataframe[column] = 0
            if has_target and rule.group_by_roi:
                for tissue_id, _container_geometry, positions in containment:
                    if not positions:
                        continue
                    position_array = np.asarray(positions)
                    point_positions, _region_positions = _region_hits(
                        points[position_array], scoped.get(int(tissue_id), [])
                    )
                    if point_positions.size:
                        dataframe.loc[row_index[position_array[point_positions]], column] = 1
            elif has_target:
                hits = _point_region_pairs(
                    points, [part["geom"] for part in target["parts"]], target["tree"]
                )
                if hits.size > 0:
                    dataframe.loc[row_index[np.unique(hits[0])], column] = 1
            summary["columns"].append(column)
            summary["counts"][column] = int(dataframe[column].sum())

        # The ROI id and its size are produced together so both always describe
        # the same region, whether or not the numbering is split per tissue piece.
        if rule.roi_id or rule.roi_size:
            id_column = f"{rule.label}_roi"
            size_column = f"{rule.label}_roi_size"
            if rule.roi_id:
                dataframe[id_column] = 0
            if rule.roi_size:
                dataframe[size_column] = 0.0

            if has_target:
                for tissue_id, _container_geometry, positions in containment:
                    if not positions:
                        continue
                    position_array = np.asarray(positions)
                    regions = scoped.get(int(tissue_id), [])
                    point_positions, region_positions = _region_hits(
                        points[position_array], regions
                    )
                    if point_positions.size == 0:
                        continue
                    rows = row_index[position_array[point_positions]]
                    if rule.roi_id:
                        dataframe.loc[rows, id_column] = [
                            int(regions[int(position)]["id"]) for position in region_positions
                        ]
                    if rule.roi_size:
                        dataframe.loc[rows, size_column] = [
                            float(regions[int(position)]["area_px2"])
                            for position in region_positions
                        ]

            if rule.roi_id:
                summary["columns"].append(id_column)
            if rule.roi_size:
                summary["columns"].append(size_column)

        if rule.distance:
            tell(f"{rule.label}: calculating {'gap-aware' if require_clean_path and rule.group_by_roi else 'ordinary boundary'} distances")
            column = f"{rule.label}_distance"
            dataframe[column] = np.nan
            measurable = 0
            if target["parts"]:
                if rule.group_by_roi:
                    distance_groups = containment
                else:
                    distance_groups = [(0, None, list(range(len(points))))]
                # Only a grouped rule has a tissue piece to stay inside, so the
                # clean-path restriction is meaningless without one.
                clean_path = bool(require_clean_path) and rule.group_by_roi
                for tissue_id, container_geometry, positions in distance_groups:
                    if not positions:
                        continue
                    subset = points[positions]
                    if clean_path and (label_lower in visibility_labels or
                                       (independent_distances and label_lower != str(tissue_label).lower())):
                        matched, distances, _owners, _skipped, has_eligible = visible_values(
                            tissue_id, container_geometry, positions, label_lower, 1, independent=independent_distances
                        )
                        if has_eligible:
                            measurable += len(positions)
                        if matched.size:
                            dataframe.loc[row_index[np.asarray(positions)[matched]], column] = distances
                        continue
                    if rule.group_by_roi:
                        candidates = scoped.get(int(tissue_id), [])
                    else:
                        candidates = next(iter(scoped.values()), [])
                    # A geometric sliver with no cell centroid is not a local
                    # cellular annotation and must not seed a distance.
                    targets, _skipped = _regions_with_min_cells(subset, candidates, 1)
                    if targets:
                        measurable += len(positions)
                    if clean_path:
                        matched, distances = _nearest_visible_distance(
                            subset, targets, container_geometry,
                            float(clean_path_tolerance_px), int(clean_path_max_candidates),
                            probe_cache=visibility_probes,
                            boundary_engine=boundary_engine,
                        )
                    else:
                        matched, distances, _owners = _signed_distance_to_regions(
                            subset, targets, boundary_engine=boundary_engine
                        )
                    if matched.size:
                        dataframe.loc[row_index[np.asarray(positions)[matched]], column] = distances
            summary["columns"].append(column)
            # Cells that had a reachable annotation in their own piece but no
            # clean line to it. Reported so the drop is never silent.
            if require_clean_path and rule.group_by_roi:
                summary.setdefault("clean_path_blanked", {})[column] = int(
                    max(measurable - int(dataframe[column].notna().sum()), 0)
                )
        tell(f"{rule.label}: finished ({time.perf_counter() - started:.1f}s)",
             .15 + .60 * (rule_number + 1) / max(len(active_rules), 1))

    # Process each unordered bed pair once. Reciprocal rules previously reset
    # and overwrote the same four columns; canonical ordering also makes ties
    # deterministic and independent of rule-table row order.
    display_names = {rule.label.lower(): rule.label for rule in active_rules}
    for rule in active_rules:
        if rule.bed_target:
            display_names.setdefault(str(rule.bed_target).lower(), str(rule.bed_target))
    bed_pairs: set[tuple[str, str]] = set()
    for rule in active_rules:
        if rule.bed and rule.bed_target:
            pair = tuple(sorted((rule.label.lower(), str(rule.bed_target).lower())))
            if pair[0] != pair[1]:
                bed_pairs.add(pair)

    # An explicit selection is one multi-class competition. Legacy callers keep
    # their independent pairs and the same tissue grouping and tie behavior.
    competitions = []
    if bed_labels is not None:
        display_names.update({name.lower(): name for name in bed_labels})
        if bed_labels:
            competitions.append((tuple(name.lower() for name in bed_labels), True))
    else:
        for pair in sorted(bed_pairs):
            group_pair = any(
                rule.group_by_roi for rule in active_rules
                if rule.bed and rule.bed_target
                and {rule.label.lower(), str(rule.bed_target).lower()} == set(pair)
            )
            competitions.append((pair, group_pair))

    filtered_bed_targets: Counter[str] = Counter()
    for competition_number, (labels, group_by_roi) in enumerate(competitions):
        tell(f"Assigning beds: {', '.join(display_names[name] for name in labels)} (minimum {minimum_bed_cells} cells per target)",
             .75 + .19 * competition_number / max(len(competitions), 1))
        bed_columns = [f"{display_names[name]}_bed" for name in labels]
        id_columns = [f"{display_names[name]}_bed_id" for name in labels]
        for column in bed_columns + id_columns:
            dataframe[column] = np.nan
        scoped_targets = {name: regions_for(name, group_by_roi) for name in labels}
        bed_groups = containment if group_by_roi else [(0, None, list(range(len(points))))]

        for tissue_id, container_geometry, positions in bed_groups:
            if not positions:
                continue
            subset = points[positions]
            position_array = np.asarray(positions)
            best_distance = np.full(len(subset), np.inf)
            winner = np.full(len(subset), -1, dtype=int)
            winning_owner = np.zeros(len(subset), dtype=int)
            for class_index, name in enumerate(labels):
                if require_clean_path and group_by_roi:
                    matched, distances, owners, skipped, _has_eligible = visible_values(
                        tissue_id, container_geometry, positions, name, minimum_bed_cells
                    )
                else:
                    scoped = scoped_targets[name]
                    candidates = (scoped.get(int(tissue_id), []) if group_by_roi
                                  else next(iter(scoped.values()), []))
                    targets, skipped = _regions_with_min_cells(
                        subset, candidates, minimum_bed_cells
                    )
                    matched, distances, owners = _signed_distance_to_regions(
                        subset, targets, boundary_engine=boundary_engine
                    )
                filtered_bed_targets[name] += skipped
                # <= preserves the legacy pair rule: the last alphabetical
                # class wins exact ties, independent of selection/row order.
                wins = np.isfinite(distances) & (distances <= best_distance[matched])
                winning_positions = matched[wins]
                best_distance[winning_positions] = distances[wins]
                winner[winning_positions] = class_index
                winning_owner[winning_positions] = owners[wins]

            assigned_rows = row_index[position_array[winner >= 0]]
            dataframe.loc[assigned_rows, bed_columns] = 0.0
            for class_index, (bed_column, id_column) in enumerate(zip(bed_columns, id_columns)):
                won = winner == class_index
                rows = row_index[position_array[won]]
                dataframe.loc[rows, bed_column] = 1.0
                dataframe.loc[rows, id_column] = winning_owner[won]
        summary["columns"].extend(bed_columns + id_columns)

    filtered_bed_targets = Counter({
        label: count for label, count in filtered_bed_targets.items() if count
    })
    summary["bed_target_fragments_below_threshold"] = dict(filtered_bed_targets)
    if filtered_bed_targets:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(filtered_bed_targets.items())
        )
        summary["warnings"].append(
            f"Bed target fragments skipped (<{minimum_bed_cells} cells): {details}."
        )

    if edge_config is not None:
        tell(f"Calculating edge cells near {edge_config.reference}", .95)
        column = edge_config.column
        dataframe[column] = 0
        reference = prepared.get(str(edge_config.reference).lower(), {"parts": [], "tree": None})
        if reference["parts"] and require_clean_path:
            from spacemap.spatial_visibility import distance_to_polygon_boundaries

            distances = distance_to_polygon_boundaries(points, [part["geom"] for part in reference["parts"]])
            dataframe.loc[row_index[np.flatnonzero(distances <= float(edge_config.buffer_px))], column] = 1
        elif reference["parts"] and reference["tree"] is not None:
            nearest = reference["tree"].query_nearest(points)
            if nearest.size:
                point_index, part_index = nearest[0], nearest[1]
                _, first = np.unique(point_index, return_index=True)
                point_index, part_index = point_index[first], part_index[first]
                boundaries = np.array(
                    [part["geom"].boundary for part in reference["parts"]], dtype=object
                )
                distances = _boundary_distances(
                    boundaries, part_index, points[point_index], boundary_engine
                )
                within = distances <= float(edge_config.buffer_px)
                dataframe.loc[row_index[point_index[within]], column] = 1
        summary["columns"].append(column)
        summary["counts"][column] = int(dataframe[column].sum())

    if preserved is not None:
        for column in preserved:
            dataframe[column] = preserved[column]
            summary["counts"].pop(column, None)
            summary.get("clean_path_blanked", {}).pop(column, None)
        summary["columns"] = [name for name in summary["columns"] if name not in preserved]
    summary["columns"] = list(dict.fromkeys(summary["columns"]))
    if boundary_engine is not None:
        summary["distance_acceleration"] = boundary_engine.summary()
        if boundary_engine.fallback_reason:
            summary["warnings"].append(boundary_engine.fallback_reason)
    else:
        summary["distance_acceleration"] = {"requested": False, "backend": "CPU", "gpu_groups": 0}
    tell(f"Measurements ready: {len(summary['columns'])} columns for {len(dataframe):,} cells", 1)
    return dataframe, summary


def annotate_cell_csv(
    csv_path: str | os.PathLike,
    geojson_path: str | os.PathLike,
    output_path: str | os.PathLike,
    rules: Sequence[SpatialFeatureRule],
    x_col: str | None = None,
    y_col: str | None = None,
    tissue_label: str = DEFAULT_TISSUE_LABEL,
    edge_config: EdgeTagConfig | None = None,
    min_cells_per_bed_target: int = DEFAULT_MIN_CELLS_PER_BED_TARGET,
    require_clean_path: bool = False,
    clean_path_tolerance_px: float = DEFAULT_CLEAN_PATH_TOLERANCE_PX,
    clean_path_max_candidates: int = DEFAULT_CLEAN_PATH_MAX_CANDIDATES,
    bed_labels: Sequence[str] | None = None,
    overwrite_existing: bool = True,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """File-in/file-out wrapper around :func:`annotate_cells`."""
    dataframe = pd.read_csv(str(csv_path))
    annotated, summary = annotate_cells(
        dataframe, geojson_path, rules,
        x_col=x_col, y_col=y_col, tissue_label=tissue_label, edge_config=edge_config,
        min_cells_per_bed_target=min_cells_per_bed_target,
        require_clean_path=require_clean_path,
        clean_path_tolerance_px=clean_path_tolerance_px,
        clean_path_max_candidates=clean_path_max_candidates,
        bed_labels=bed_labels,
        overwrite_existing=overwrite_existing,
        use_gpu=use_gpu,
    )
    output_directory = os.path.dirname(str(output_path))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    annotated.to_csv(str(output_path), index=False)
    summary["output_path"] = str(output_path)
    summary["cells"] = int(len(annotated))
    return summary
