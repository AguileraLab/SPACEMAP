"""Continuous first-hit distances within one tissue ROI.

A constrained triangulation of tissue minus all selected annotations is built
once. Visibility expands through triangle portals, stopping at annotation and
tissue boundaries. Original polygon edges are used; no raster or angular-ray
sampling is involved. ROI IDs come from the calling annotation engine.
"""
from __future__ import annotations
import heapq
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import shapely
from shapely.strtree import STRtree
from numba import njit

ENGINE_NAME = "Triangle visibility (CPU)"


def cpu_workers(requested=0):
    """Bound exact geometry workers by the process's CPU allocation."""
    from spacemap.consensus_execution import cpu_budget

    return cpu_budget(requested or 8)["workers"]


def measure_visibility(xy, roots, triangles, neighbors, edge_rois, counts, threshold, workers=1, progress=None):
    """Share the read-only mesh; each cell retains its original root/visit order.

    The compiled traversal releases the GIL. Chunking across cells adds no
    approximation and avoids copying the mesh into separate worker processes.
    """
    if (workers <= 1 and progress is None) or len(xy) < 2048:
        result = expand_visibility(xy, roots, triangles, neighbors, edge_rois, counts, threshold)
        if progress is not None:
            progress(len(xy), len(xy))
        return result
    order = np.argsort(roots[0], kind="stable")
    roots = roots[:, order]
    chunk = max(512, math.ceil(len(xy) / max(workers * 4, 20)))

    def measure(start):
        end = min(start + chunk, len(xy))
        first, last = np.searchsorted(roots[0], [start, end])
        local_roots = roots[:, first:last].copy()
        local_roots[0] -= start
        return start, end, expand_visibility(xy[start:end], local_roots, triangles, neighbors,
                                             edge_rois, counts, threshold)

    distances = np.full((len(xy), edge_rois.shape[2]), np.inf)
    owners = np.full(distances.shape, -1, dtype=np.int64)
    completed, milestone = 0, 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tag-distance") as pool:
        pending = [pool.submit(measure, start) for start in range(0, len(xy), chunk)]
        for future in as_completed(pending):
            start, end, (values, ids) = future.result()
            distances[start:end], owners[start:end] = values, ids
            completed += end - start
            # Work milestones, never elapsed-time heartbeats. Run callbacks on
            # the caller thread so status writes and cancellation stay ordered.
            if progress is not None and completed * 4 >= milestone * len(xy):
                progress(completed, len(xy))
                milestone = completed * 4 // len(xy) + 1
    return distances, owners

def segments_of(geometry):
    result = []
    if geometry.geom_type == "Polygon":
        for ring in [geometry.exterior, *geometry.interiors]:
            xy = np.asarray(ring.coords, dtype=np.float64)[:, :2]
            result.append(np.column_stack((xy[:-1], xy[1:])))
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            result.extend([segments_of(part)])
    return np.concatenate(result) if result else np.empty((0, 4), dtype=np.float64)


def make_bvh(segments, ids=None, leaf_size=12):
    ids = np.arange(len(segments), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return (np.zeros((0, 4)), np.zeros((0, 4), dtype=np.int64), ids)
    low = np.minimum(segments[:, :2], segments[:, 2:])
    high = np.maximum(segments[:, :2], segments[:, 2:])
    centers = (low + high) * .5
    boxes, nodes = [], []

    def build(start, end):
        index = len(nodes)
        selected = ids[start:end]
        lo, hi = low[selected].min(axis=0), high[selected].max(axis=0)
        boxes.append([*lo, *hi])
        nodes.append([-1, -1, start, end])
        if end - start > leaf_size:
            axis = int(np.argmax(np.ptp(centers[selected], axis=0)))
            middle = (end - start) // 2
            ids[start:end] = selected[np.argpartition(centers[selected, axis], middle)]
            left = build(start, start + middle)
            right = build(start + middle, end)
            nodes[index][:2] = [left, right]
        return index

    build(0, len(ids))
    return np.asarray(boxes), np.asarray(nodes, dtype=np.int64), ids


@njit(cache=True)
def cross(ax, ay, bx, by):
    return ax * by - ay * bx


@njit(cache=True)
def bbox_distance(x, y, box):
    dx = max(box[0] - x, 0., x - box[2])
    dy = max(box[1] - y, 0., y - box[3])
    return dx * dx + dy * dy


@njit(cache=True)
def point_segment(x, y, s):
    vx, vy = s[2] - s[0], s[3] - s[1]
    norm = vx * vx + vy * vy
    t = 0. if norm == 0 else min(1., max(0., ((x - s[0]) * vx + (y - s[1]) * vy) / norm))
    qx, qy = s[0] + t * vx, s[1] + t * vy
    return (qx - x) ** 2 + (qy - y) ** 2, t


@njit(cache=True)
def clip_portal(x, y, ax, ay, bx, by, ux, uy, vx, vy):
    dx, dy = bx - ax, by - ay
    lo, hi = 0., 1.
    for f0, df in ((cross(ux, uy, ax - x, ay - y), cross(ux, uy, dx, dy)),
                   (cross(ax - x, ay - y, vx, vy), cross(dx, dy, vx, vy))):
        if abs(df) < 1e-20:
            if f0 < -1e-12:
                return 0., 0., 0., 0., False
        elif df > 0:
            lo = max(lo, -f0 / df)
        else:
            hi = min(hi, -f0 / df)
        if lo >= hi:
            return 0., 0., 0., 0., False
    return ax + lo * dx, ay + lo * dy, ax + hi * dx, ay + hi * dy, True


@njit(cache=True, nogil=True)
def boundary_distances(xy, segments, boxes, nodes, order):
    """Indexed nearest boundary distance, also used for negative inside values."""
    result = np.full(len(xy), np.nan)
    if not len(nodes):
        return result
    for row in range(len(xy)):
        x, y = xy[row]
        best = np.inf
        heap = [(bbox_distance(x, y, boxes[0]), 0)]
        while heap:
            lower, node = heapq.heappop(heap)
            if lower >= best:
                break
            left, right, start, end = nodes[node]
            if left >= 0:
                for child in (left, right):
                    bound = bbox_distance(x, y, boxes[child])
                    if bound < best:
                        heapq.heappush(heap, (bound, child))
            else:
                for i in range(start, end):
                    distance, _ = point_segment(x, y, segments[order[i]])
                    best = min(best, distance)
        result[row] = math.sqrt(best)
    return result


def distance_to_polygon_boundaries(points, geometries):
    """Unsigned nearest boundary distance for disjoint merged polygon parts."""
    arrays = [segments_of(geometry) for geometry in geometries]
    out = np.full(len(points), np.nan)
    if not arrays:
        return out
    segments = np.concatenate(arrays)
    xy = np.column_stack((shapely.get_x(points), shapely.get_y(points)))
    valid = np.isfinite(xy).all(axis=1)
    out[valid] = boundary_distances(xy[valid], segments, *make_bvh(segments))
    return out


@njit(cache=True, nogil=True)
def expand_visibility(xy, roots, triangles, neighbors, edge_rois, counts, threshold):
    """All visible classes and their winning ROI, with target QC at endpoints.

    Ineligible annotations still block visibility. This prevents a bed from
    reaching through a small excluded annotation into a more distant target.
    """
    class_count = edge_rois.shape[2]
    distances = np.full((len(xy), class_count), np.inf)
    owners = np.full((len(xy), class_count), -1, dtype=np.int64)
    visits = np.zeros(len(xy), dtype=np.int64)
    for root_index in range(roots.shape[1]):
        row, root = roots[0, root_index], roots[1, root_index]
        x, y = xy[row]
        stack = [(root, -1, 0., 0., 0., 0.)]
        while stack:
            triangle, previous, ux, uy, vx, vy = stack.pop()
            visits[row] += 1
            if visits[row] > 10_000_000:
                raise RuntimeError("Triangle visibility exceeded its topology guard")
            for edge in range(3):
                neighbor = neighbors[triangle, edge]
                if neighbor == previous and previous >= 0:
                    continue
                ax, ay = triangles[triangle, edge]
                bx, by = triangles[triangle, (edge + 1) % 3]
                if previous >= 0:
                    ax, ay, bx, by, visible = clip_portal(x, y, ax, ay, bx, by, ux, uy, vx, vy)
                    if not visible:
                        continue
                if cross(ax - x, ay - y, bx - x, by - y) <= 1e-14:
                    continue
                if neighbor >= 0:
                    stack.append((neighbor, triangle, ax - x, ay - y, bx - x, by - y))
                else:
                    value, _ = point_segment(x, y, np.array([ax, ay, bx, by]))
                    for c in range(class_count):
                        roi = edge_rois[triangle, edge, c]
                        if roi >= 0 and counts[c, roi] >= threshold:
                            if value < distances[row, c] or (value == distances[row, c] and roi < owners[row, c]):
                                distances[row, c], owners[row, c] = value, roi
    return np.sqrt(distances), owners


class TissueVisibility:
    """One run-local index; callers discard it after measuring a tissue piece."""

    def __init__(self, tissue, targets, tile_size=1024., workers=1):
        if not hasattr(shapely, "constrained_delaunay_triangles"):
            raise RuntimeError("Fast tissue visibility requires Shapely 2.1 or newer.")
        self.tissue = tissue
        self.labels = sorted(targets)
        self.targets = targets
        self.tile_size = tile_size
        self.workers = workers
        self.metadata = {"engine": ENGINE_NAME, "tile_size_px": tile_size, "cpu_workers": workers}
        arrays = [segments_of(tissue)]
        classes = [np.full(len(arrays[0]), -1, dtype=np.int64)]
        owners = [np.full(len(arrays[0]), -1, dtype=np.int64)]
        self.geometries = {}
        self.region_ids = {}
        for c, label in enumerate(self.labels):
            regions = targets[label]
            self.geometries[label] = np.asarray([r["geom"] for r in regions], dtype=object)
            self.region_ids[label] = np.asarray([r["id"] for r in regions], dtype=np.int64)
            shapely.prepare(self.geometries[label])
            for region in regions:
                edges = segments_of(region["geom"])
                arrays.append(edges)
                classes.append(np.full(len(edges), c, dtype=np.int64))
                owners.append(np.full(len(edges), int(region["id"]), dtype=np.int64))
        self.segments = np.concatenate(arrays)
        self.classes, self.rois = np.concatenate(classes), np.concatenate(owners)
        self.target_bvhs = [make_bvh(self.segments, np.flatnonzero(self.classes == c)) for c in range(len(self.labels))]
        self.triangle_tree = None
        shapely.prepare(tissue)

    def _direct_distances(self, points, outside, counts):
        """Exact shortcut for a single class with an unobstructed nearest border.

        A line covered by tissue attains the global Euclidean lower bound;
        no other visible part can be closer. Blocked lines still use the full
        triangle search, including visible parts of the same target polygon.
        Empty targets still block and cannot seed distances. No sampling.
        """
        positions = np.flatnonzero(outside)
        label = self.labels[0]
        geometries = self.geometries[label]
        if not len(positions) or not len(geometries):
            return positions[:0], np.empty(0), np.empty(0, dtype=np.int64)
        pairs = STRtree(geometries).query_nearest(points[positions], all_matches=True)
        ids = self.region_ids[label]
        # Equal distances retain the lowest ROI ID, as the traversal does.
        order = np.lexsort((ids[pairs[1]], pairs[0]))
        pairs = pairs[:, order]
        _, first = np.unique(pairs[0], return_index=True)
        positions, targets = positions[pairs[0, first]], pairs[1, first]
        lines = shapely.shortest_line(points[positions], geometries[targets])
        visible = shapely.covers(self.tissue, lines) & (counts[0, ids[targets]] >= 1)
        return positions[visible], shapely.length(lines[visible]), ids[targets[visible]]

    def _build_triangles(self, tiled=True, progress=None):
        start = time.perf_counter()
        tell = progress if progress is not None else lambda message: None
        tell("Preparing tissue geometry for gap-aware distances")
        geometries = [g for label in self.labels for g in self.geometries[label]]
        free = self.tissue.difference(shapely.union_all(geometries))
        source = free
        if tiled and self.tile_size and not free.is_empty:
            width = self.tile_size
            xmin, ymin, xmax, ymax = free.bounds
            xs = np.arange(np.floor(xmin / width) * width, xmax, width)
            ys = np.arange(np.floor(ymin / width) * width, ymax, width)
            # Avoid allocating a huge empty grid for extreme coordinate ranges.
            if len(xs) * len(ys) <= 100_000:
                xx, yy = np.meshgrid(xs, ys)
                tiles = shapely.box(xx.ravel(), yy.ravel(), xx.ravel() + width, yy.ravel() + width)
                source = shapely.intersection(free, tiles)
        tell("Triangulating tissue around gaps and annotation boundaries")
        polygons = shapely.get_parts(shapely.constrained_delaunay_triangles(source))
        tell(f"Indexing {len(polygons):,} tissue triangles")
        self.triangle_tree = STRtree(polygons)
        if not len(polygons):
            self.triangles = np.empty((0, 3, 2))
            self.neighbors = np.empty((0, 3), dtype=np.int64)
            self.edge_rois = np.empty((0, 3, len(self.labels)), dtype=np.int64)
            return
        coords = shapely.get_coordinates(polygons).reshape(-1, 4, 2)[:, :3].copy()
        v1, v2 = coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0]
        reverse = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0] < 0
        coords[reverse] = coords[reverse, ::-1]
        _, vertex_ids = np.unique(coords.reshape(-1, 2), axis=0, return_inverse=True)
        vertex_ids = vertex_ids.reshape(-1, 3)
        edges = np.stack([vertex_ids, np.roll(vertex_ids, -1, axis=1)], axis=-1).reshape(-1, 2)
        edges.sort(axis=1)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        same = np.all(edges[order][1:] == edges[order][:-1], axis=1)
        if np.any(same[1:] & same[:-1]):
            raise RuntimeError("Non-manifold tissue triangulation")
        paired = np.flatnonzero(same)
        neighbors = np.full(len(edges), -1, dtype=np.int64)
        neighbors[order[paired]] = order[paired + 1] // 3
        neighbors[order[paired + 1]] = order[paired] // 3
        boundary_ids = np.flatnonzero(neighbors < 0)
        a = coords.reshape(-1, 2)[boundary_ids]
        b = np.roll(coords, -1, axis=1).reshape(-1, 2)[boundary_ids]
        source_tree = STRtree(shapely.linestrings(self.segments.reshape(-1, 2, 2)))
        matched, distances = source_tree.query_nearest(shapely.points((a + b) * .5), return_distance=True)
        errors = np.full(len(boundary_ids), np.inf)
        np.minimum.at(errors, matched[0], distances)
        if np.any(errors > 1e-6):
            if tiled:
                self.metadata["untiled_retry"] = True
                tell("Rebuilding the mesh without tiles to preserve boundary accuracy")
                return self._build_triangles(tiled=False, progress=progress)
            raise RuntimeError("Tissue triangulation has an unmatched internal boundary")
        matched = matched[:, distances <= 1e-6]
        owners = np.full((len(edges), len(self.labels)), -1, dtype=np.int64)
        for c in range(len(self.labels)):
            selected = self.classes[matched[1]] == c
            indices, sources = boundary_ids[matched[0, selected]], matched[1, selected]
            # Source order / explicit ROI IDs are retained, not re-numbered.
            owners[indices, c] = self.rois[sources]
        self.triangles = coords
        self.neighbors = neighbors.reshape(-1, 3)
        self.edge_rois = owners.reshape(-1, 3, len(self.labels))
        self.metadata.update(triangles=len(coords), preparation_seconds=time.perf_counter() - start)
        tell(f"Tissue mesh ready: {len(coords):,} triangles ({time.perf_counter() - start:.1f}s)")

    def measure(self, points, thresholds=(1,), hit_query=None, progress=None):
        """Measure each distinct QC threshold, sharing topology and membership."""
        if hit_query is None:
            from spacemap.cell_spatial_annotation import _point_region_pairs
            hit_query = _point_region_pairs

        thresholds = sorted({max(1, int(t)) for t in thresholds})
        n, k = len(points), len(self.labels)
        tell = progress if progress is not None else lambda message: None
        tell(f"Checking annotation membership and inside distances for {n:,} cells")
        xy = np.column_stack((shapely.get_x(points), shapely.get_y(points)))
        valid = np.isfinite(xy).all(axis=1) & shapely.covers(self.tissue, points)
        inside_owner = np.full((n, k), -1, dtype=np.int64)
        maximum_id = max((int(ids.max()) for ids in self.region_ids.values() if len(ids)), default=0)
        counts = np.zeros((k, maximum_id + 1), dtype=np.int64)
        negative = np.full((n, k), np.nan)
        for c, label in enumerate(self.labels):
            ids = self.region_ids[label]
            hits = hit_query(points, self.geometries[label])
            if hits.size:
                counts[c, ids] = np.bincount(hits[1], minlength=len(ids))
                positions, first = np.unique(hits[0], return_index=True)
                inside_owner[positions, c] = ids[hits[1, first]]
                negative[positions, c] = -boundary_distances(xy[positions], self.segments, *self.target_bvhs[c])
        occupied = (inside_owner >= 0).any(axis=1)
        outside = valid & ~occupied
        direct_positions, direct_values, direct_ids = (np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int))
        if k == 1:
            tell(f"Checking direct boundary paths for {int(outside.sum()):,} outside cells")
            direct_positions, direct_values, direct_ids = self._direct_distances(points, outside, counts)
        self.metadata["direct_cells"] = len(direct_positions)
        # A target excluded at a stricter QC threshold still blocks the line.
        # Keep those cells in the visibility search for that threshold.
        direct_all_thresholds = direct_positions[counts[0, direct_ids] >= max(thresholds)] if k == 1 else direct_positions
        search = outside.copy()
        search[direct_all_thresholds] = False
        roots = np.empty((2, 0), dtype=np.int64)
        if search.any() and any(len(g) for g in self.geometries.values()):
            tell(f"{len(direct_all_thresholds):,} direct paths resolved; {int(search.sum()):,} cells need gap-aware search")
            if self.triangle_tree is None:
                self._build_triangles(progress=progress)
            roots = self.triangle_tree.query(points[search], predicate="intersects")
            if len(np.unique(roots[0])) != int(search.sum()):
                raise RuntimeError("A tissue cell could not be located in the visibility triangulation")
        results, skipped, signatures = {}, {}, {}
        for threshold in thresholds:
            eligibility = (counts >= threshold).tobytes()
            if eligibility in signatures:
                results[threshold] = results[signatures[eligibility]]
            else:
                distances = np.full((n, k), np.nan)
                owners = np.full((n, k), -1, dtype=np.int64)
                for c in range(k):
                    selected = valid & (inside_owner[:, c] >= 0)
                    selected &= counts[c, np.maximum(inside_owner[:, c], 0)] >= threshold
                    distances[selected, c] = negative[selected, c]
                    owners[selected, c] = inside_owner[selected, c]
                if roots.size:
                    tell(f"Gap-aware search: {int(search.sum()):,} cells, {self.workers} CPU workers, minimum {threshold} cells per target")
                    def completed(done, total):
                        tell(f"Gap-aware search: {done:,}/{total:,} cells completed (minimum {threshold} per target)")

                    measured, found_owners = measure_visibility(xy[search], roots, self.triangles, self.neighbors,
                                                               self.edge_rois, counts, threshold, self.workers,
                                                               progress=completed if progress is not None else None)
                    measured[~np.isfinite(measured)] = np.nan
                    distances[search], owners[search] = measured, found_owners
                if k == 1 and len(direct_positions):
                    eligible = counts[0, direct_ids] >= threshold
                    distances[direct_positions[eligible], 0] = direct_values[eligible]
                    owners[direct_positions[eligible], 0] = direct_ids[eligible]
                results[threshold] = distances, owners
                signatures[eligibility] = threshold
            skipped[threshold] = {label: int((counts[c, self.region_ids[label]] < threshold).sum())
                                  for c, label in enumerate(self.labels)}
        self.metadata["thresholds"] = thresholds
        return results, skipped
