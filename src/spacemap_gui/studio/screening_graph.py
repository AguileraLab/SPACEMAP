"""Exact, bounded-memory spatial motif counts for screening.

One radius graph per image/region; all requested zones share it. Triangles are
closed 3-cliques, counted once by intersecting sorted forward adjacency lists.
No triangle-instance array, cell-by-cell distance matrix, or random subsampling.
"""
from itertools import combinations_with_replacement
import math

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


def radius_edges(xy, low, high, max_edges=20_000_000):
    tree = cKDTree(xy)
    estimate = (int(tree.count_neighbors(tree, high)) - len(xy)) // 2
    if estimate > max_edges:
        raise ValueError(f"The neighbour radius creates {estimate:,} pairs. Reduce the radius or narrow the region (limit {max_edges:,}); cells were not subsampled.")
    edges = tree.query_pairs(high, output_type="ndarray")
    if low > 0 and len(edges):
        delta = xy[edges[:, 0]] - xy[edges[:, 1]]
        edges = edges[np.einsum("ij,ij->i", delta, delta) >= low * low]
    return edges.astype(np.int64, copy=False)


@njit(cache=True)
def _triangle_instances(ptr, nbr, count):
    result = np.empty((count, 3), np.int32)
    found = 0
    for a in range(len(ptr) - 1):
        for ab in range(ptr[a], ptr[a + 1]):
            b = nbr[ab]
            i, j = ab + 1, ptr[b]
            while i < ptr[a + 1] and j < ptr[b + 1]:
                c, other = nbr[i], nbr[j]
                if c < other:
                    i += 1
                elif c > other:
                    j += 1
                else:
                    result[found, 0], result[found, 1], result[found, 2] = a, b, c
                    found += 1
                    i += 1
                    j += 1
    return result


def triangle_array(edges, n_vertices, max_triangles=20_000_000):
    """Exact sorted triples for Classic's shared permutation engine; no Python tuple list."""
    forward = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    ptr = np.r_[0, np.cumsum(np.bincount(forward[:, 0], minlength=n_vertices))]
    if np.sum(np.diff(ptr).astype(float) ** 2) > 500_000_000:
        raise ValueError("The triad graph is too dense. Reduce the radius or restrict the region.")
    _, totals = _triangles(ptr, forward[:, 1], np.zeros(n_vertices, np.int64),
                          np.zeros(n_vertices, np.int64), 1, 1, False)
    count = int(totals[0])
    if count > max_triangles:
        raise ValueError(f"The graph has {count:,} triplets; reduce the radius or restrict the region (limit {max_triangles:,}).")
    return _triangle_instances(ptr, forward[:, 1], count)


@njit(cache=True)
def _triangles(ptr, nbr, labels, zones, n_labels, n_zones, rings):
    counts = np.zeros((n_zones, n_labels ** 3), np.int64)
    totals = np.zeros(n_zones, np.int64)
    for a in range(len(labels)):
        za = zones[a]
        if za < 0:
            continue
        for ab in range(ptr[a], ptr[a + 1]):
            b = nbr[ab]
            zb = zones[b]
            if zb < 0 or (rings and za != zb):
                continue
            i, j = ab + 1, ptr[b]
            while i < ptr[a + 1] and j < ptr[b + 1]:
                c, other = nbr[i], nbr[j]
                if c < other:
                    i += 1
                elif c > other:
                    j += 1
                else:
                    zc = zones[c]
                    if zc >= 0 and (not rings or zc == za):
                        zone = max(za, zb, zc)
                        totals[zone] += 1
                        x, y, z = labels[a], labels[b], labels[c]
                        lo, hi = min(x, y, z), max(x, y, z)
                        mid = x + y + z - lo - hi
                        counts[zone, (lo * n_labels + mid) * n_labels + hi] += 1
                    i += 1
                    j += 1
    return counts, totals


def motif_counts(edges, labels, zones, n_labels, n_zones, *, triangles=False, rings=False):
    """Histograms by first included zone (cumulative) or disjoint ring.

Negative zones exclude vertices. All labels, including non-target types, stay
in the graph so changing targets never changes a feature's denominator.
"""
    if triangles:
        order = np.lexsort((edges[:, 1], edges[:, 0])) if len(edges) else np.array([], dtype=int)
        forward = edges[order]
        ptr = np.r_[0, np.cumsum(np.bincount(forward[:, 0], minlength=len(labels)))]
        # The sum of forward degrees squared bounds intersection work. Fail
        # explicitly rather than freeze or silently return an approximate count.
        work = int(np.sum(np.diff(ptr).astype(float) ** 2))
        if work > 500_000_000:
            raise ValueError("The triad graph is too dense. Reduce the triad radius or restrict the region; no approximate motifs were reported.")
        counts, totals = _triangles(ptr, forward[:, 1], labels, zones, n_labels, n_zones, rings)
    else:
        a, b = edges.T
        valid = (zones[a] >= 0) & (zones[b] >= 0)
        if rings:
            valid &= zones[a] == zones[b]
        a, b = a[valid], b[valid]
        zone = np.maximum(zones[a], zones[b])
        code = np.minimum(labels[a], labels[b]) * n_labels + np.maximum(labels[a], labels[b])
        counts = np.bincount(zone * n_labels ** 2 + code, minlength=n_zones * n_labels ** 2).reshape(n_zones, -1)
        totals = np.bincount(zone, minlength=n_zones)
    if not rings:
        counts, totals = np.cumsum(counts, axis=0), np.cumsum(totals)
    return counts, totals


def label_probability(counts, motif):
    """Exact random-label probability on distinct vertices, without replacement.

This is an expected frequency, not a permutation Z-score or a p-value.
"""
    size, total = len(motif), int(sum(counts))
    if total < size:
        return 0.0
    probability = math.factorial(size)
    used = {}
    for label in motif:
        n = used.get(label, 0)
        probability *= max(0, int(counts[label]) - n) / (total - sum(used.values()))
        used[label] = n + 1
    for n in used.values():
        probability /= math.factorial(n)
    return float(probability)


def definitions(targets, size, repeated=False):
    return [item for item in combinations_with_replacement(targets, size)
            if repeated or len(set(item)) == size]
