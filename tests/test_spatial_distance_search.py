"""Indexed searches retain exact distances and do not miss narrow tissue gaps."""

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree

from test_cell_spatial_annotation import spatial


def test_indexed_candidates_equal_full_distance_ranking(monkeypatch):
    geometries = np.array([box(x * 30, y * 30, x * 30 + 2, y * 30 + 3)
                           for x in range(32) for y in range(32)], dtype=object)
    points = shapely.points(np.random.default_rng(13).uniform(-100, 1100, size=(40, 2)))
    tree = STRtree(geometries)
    distances = shapely.distance(points[:, None], geometries[None, :])
    expected = np.argsort(distances, axis=1, kind="stable")[:, :8]
    original_distance = shapely.distance
    measured_pairs = []

    def count_distance_pairs(a, b, **kwargs):
        measured_pairs.append(np.prod(np.broadcast_shapes(np.shape(a), np.shape(b))))
        return original_distance(a, b, **kwargs)

    monkeypatch.setattr(spatial.shapely, "distance", count_distance_pairs)
    actual = spatial._nearest_region_candidates(points, geometries, tree, distances.min(axis=1), 8)
    np.testing.assert_array_equal(actual, expected)
    assert sum(measured_pairs) < distances.size // 4


def test_clean_path_matches_brute_force_reference():
    tissue = Polygon([(0, 0), (500, 0), (500, 500), (0, 500)],
                     holes=[[(210, 80), (290, 80), (290, 420), (210, 420)]])
    geometries = np.array([box(x, y, x + 15, y + 17)
                           for x in (30, 100, 160, 320, 400)
                           for y in (30, 150, 300, 450)], dtype=object)
    points = shapely.points(np.random.default_rng(4).uniform(0, 500, size=(300, 2)))
    points = points[shapely.covers(tissue, points)]
    regions = [{"geom": geometry, "id": i + 1} for i, geometry in enumerate(geometries)]
    distances = shapely.distance(points[:, None], geometries[None, :])
    rankings = np.argsort(distances, axis=1, kind="stable")[:, :8]
    expected = np.full(len(points), np.nan)
    for row, point in enumerate(points):
        for candidate in rankings[row]:
            geometry = geometries[candidate]
            if geometry.contains(point):
                expected[row] = -geometry.boundary.distance(point)
                break
            if tissue.covers(shapely.shortest_line(geometry, point)):
                expected[row] = geometry.distance(point)
                break
    positions, values = spatial._nearest_visible_distance(points, regions, tissue, 0, 8)
    actual = np.full(len(points), np.nan)
    actual[positions] = values
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True)


def test_subpixel_gap_is_not_skipped_and_tolerance_is_preserved():
    tissue = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                     holes=[[(4.999, 0.5), (5.001, 0.5), (5.001, 9.5), (4.999, 9.5)]])
    points = shapely.points([[1, 5]])
    regions = [{"geom": box(8, 4, 9, 6), "id": 1}]
    positions, _ = spatial._nearest_visible_distance(points, regions, tissue, 0, 8)
    assert positions.size == 0
    positions, distances = spatial._nearest_visible_distance(points, regions, tissue, 0.01, 8)
    assert positions.tolist() == [0]
    np.testing.assert_allclose(distances, [7])


def test_prepared_tissue_probe_reused_between_classes(monkeypatch):
    tissue = box(0, 0, 100, 100)
    points = shapely.points([[5, 5], [50, 50]])
    regions = [{"geom": box(70, 70, 90, 90), "id": 1}]
    calls = []
    original = shapely.buffer

    def counted_buffer(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(spatial.shapely, "buffer", counted_buffer)
    cache = {}
    spatial._nearest_visible_distance(points, regions, tissue, 1, 8, probe_cache=cache)
    spatial._nearest_visible_distance(points, regions, tissue, 1, 8, probe_cache=cache)
    assert len(calls) == 1


def test_contained_polygon_keeps_all_holes_without_intersection(monkeypatch):
    tissue = box(0, 0, 100, 100)
    geometry = Polygon([(10, 10), (90, 10), (90, 90), (10, 90)],
                       holes=[[(30, 30), (70, 30), (70, 70), (30, 70)]])
    prepared = spatial._prepare_targets({"tumor": [geometry]}, ["tumor"])["tumor"]

    def fail(*args, **kwargs):
        raise AssertionError("Contained geometry should not be intersected")

    monkeypatch.setattr(spatial.shapely, "intersection", fail)
    fragments = spatial._clipped_parts(prepared, tissue)
    assert len(fragments) == 1
    assert fragments[0]["geom"] is geometry
    assert len(fragments[0]["geom"].interiors) == 1


def test_prepared_polygon_hits_match_original_including_holes_and_boundaries():
    polygons = np.array([
        Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                holes=[[(30, 30), (70, 30), (70, 70), (30, 70)]]),
        box(100, 0, 200, 100), box(20, 20, 40, 40),
    ], dtype=object)
    points = shapely.points([[0, 0], [100, 50], [30, 30], [50, 50], [25, 25],
                            [150, 50], [250, 50], [float('nan'), float('nan')]])
    tree = STRtree(polygons)
    expected = tree.query(points, predicate="intersects")
    actual = spatial._point_region_pairs(points, polygons, tree)
    np.testing.assert_array_equal(actual, expected)
