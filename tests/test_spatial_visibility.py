"""Production triangle visibility, QC, bed owners, and GUI engine integration."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
from test_cell_spatial_annotation import spatial, _feature, _geojson

SOURCE = Path(__file__).resolve().parents[1] / "src/spacemap/spatial_visibility.py"
SPEC = importlib.util.spec_from_file_location("spacemap.spatial_visibility", SOURCE)
visibility = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visibility
SPEC.loader.exec_module(visibility)


def rules(*labels):
    return [spatial.SpatialFeatureRule(label, roi_id=True, roi_size=True, distance=True, group_by_roi=True)
            for label in labels]


def test_selected_annotations_block_distances_and_beds(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(70, 40, 80, 60)), _feature("normal", box(40, 0, 50, 100))])
    cells = pd.DataFrame({"x_centroid": [10., 45., 75.], "y_centroid": [50., 50., 50.]})
    result, summary = spatial.annotate_cells(cells, path, rules("tumor", "normal"),
                                            bed_labels=["tumor", "normal"], require_clean_path=True)
    assert summary["distance_method"] == visibility.ENGINE_NAME
    np.testing.assert_allclose(result.tumor_distance, [np.nan, np.nan, -5], equal_nan=True)
    np.testing.assert_allclose(result.normal_distance, [30, -5, np.nan], equal_nan=True)
    assert result.normal_bed.tolist() == [1, 1, 0]
    assert result.tumor_bed.tolist() == [0, 0, 1]
    assert result.loc[2, "tumor_bed_id"] == result.loc[2, "tumor_roi"]


def test_qc_excluded_roi_blocks_but_cannot_seed_a_bed(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(40, 0, 50, 100)), _feature("normal", box(70, 30, 90, 70))])
    cells = pd.DataFrame({"x_centroid": [10., 45., 75., 80.], "y_centroid": [50., 50., 50., 50.]})
    result, summary = spatial.annotate_cells(cells, path, rules("tumor", "normal"),
        bed_labels=["tumor", "normal"], require_clean_path=True, min_cells_per_bed_target=2)
    assert result.loc[0, "tumor_distance"] == 30
    assert result.loc[1, "tumor_distance"] == -5
    assert result.loc[:1, ["tumor_bed", "normal_bed"]].isna().all().all()
    assert result.loc[2:, "normal_bed"].eq(1).all()
    assert summary["bed_target_fragments_below_threshold"] == {"tumor": 1}
    assert len(summary["visibility_tissue_rois"]) == 1
    assert summary["visibility_tissue_rois"][0]["thresholds"] == [1, 2]


def test_cross_tissue_roi_ids_remain_unique_and_winning_owners_match(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100)),
        _feature("tissue", box(100.0001, 0, 200, 100)),
        _feature("tumor", box(20, 20, 180, 80))])
    cells = pd.DataFrame({"x_centroid": [30., 190., 170., -10.], "y_centroid": [50.] * 4})
    result, _ = spatial.annotate_cells(cells, path, rules("tumor", "normal"),
                                      bed_labels=["tumor", "normal"], require_clean_path=True)
    assert result.loc[0, "tumor_roi"] != result.loc[2, "tumor_roi"]
    assert result.loc[0, "tumor_bed_id"] == result.loc[0, "tumor_roi"]
    assert result.loc[1, "tumor_bed_id"] == result.loc[2, "tumor_roi"]
    assert result.loc[2, "tumor_bed_id"] == result.loc[2, "tumor_roi"]
    assert pd.isna(result.loc[3, "tumor_distance"])


def test_no_cross_roi_distance_even_when_other_roi_target_is_nearer(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 10, 10)),
        _feature("tissue", box(10.0001, 0, 20, 10)),
        _feature("tumor", box(10.5, 4, 12, 6)), _feature("normal", box(1, 4, 2, 6))])
    cells = pd.DataFrame({"x_centroid": [9., 11., 1.5], "y_centroid": [5.] * 3})
    result, _ = spatial.annotate_cells(cells, path, rules("tumor", "normal"),
                                      bed_labels=["tumor", "normal"], require_clean_path=True)
    assert pd.isna(result.loc[0, "tumor_distance"])
    assert result.loc[0, "normal_distance"] == 7
    assert result.loc[0, "normal_bed"] == 1


def test_visible_part_of_same_roi_beats_blocked_nearest_point(tmp_path):
    tissue = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                     holes=[[(40, 35), (50, 35), (50, 65), (40, 65)]])
    path = _geojson(tmp_path, [_feature("tissue", tissue), _feature("tumor", box(70, 0, 80, 100))])
    cells = pd.DataFrame({"x_centroid": [10., 75.], "y_centroid": [50., 50.]})
    result, _ = spatial.annotate_cells(cells, path, rules("tumor"), require_clean_path=True)
    assert result.loc[0, "tumor_distance"] == pytest.approx(np.hypot(60, 30))


def test_overwrite_replaces_whole_columns_including_blanks(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(20, 20, 40, 40))])
    cells = pd.DataFrame({"x_centroid": [30., 130.], "y_centroid": [30., 30.],
                          "tumor_distance": [999., 999.], "tumor_bed_id": [999., 999.]})
    result, _ = spatial.annotate_cells(cells, path, rules("tumor"), bed_labels=["tumor", "normal"],
                                      require_clean_path=True, overwrite_existing=True)
    assert result.loc[0, "tumor_distance"] == -10
    assert result.loc[1, ["tumor_distance", "tumor_bed_id"]].isna().all()
    preserved, _ = spatial.annotate_cells(cells, path, rules("tumor"), bed_labels=["tumor", "normal"],
                                         require_clean_path=True, overwrite_existing=False)
    pd.testing.assert_series_equal(preserved.tumor_distance, cells.tumor_distance)
    pd.testing.assert_series_equal(preserved.tumor_bed_id, cells.tumor_bed_id)


def test_ties_keep_last_alphabetical_class_and_bed_only_selection_uses_visibility(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100)),
        _feature("normal", box(10, 40, 30, 60)), _feature("tumor", box(70, 40, 90, 60))])
    cells = pd.DataFrame({"x_centroid": [50., 20., 80.], "y_centroid": [50.] * 3})
    result, summary = spatial.annotate_cells(cells, path, [], bed_labels=["tumor", "normal"], require_clean_path=True)
    assert result.loc[0, "tumor_bed"] == 1
    assert result.loc[0, "normal_bed"] == 0
    assert summary["distance_method"] == visibility.ENGINE_NAME


def test_fast_edge_distances_equal_geos_including_holes_and_other_pieces():
    geometries = [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)],
                         holes=[[(20, 20), (40, 20), (40, 40), (20, 40)]]), box(200, 0, 230, 80)]
    xy = np.random.default_rng(24).uniform([-50, -50], [250, 150], (200, 2))
    points = shapely.points(xy)
    expected = shapely.distance(MultiPolygon(geometries).boundary, points)
    np.testing.assert_allclose(visibility.distance_to_polygon_boundaries(points, geometries), expected, atol=1e-9)


def test_missing_targets_and_invalid_coordinates_remain_blank(tmp_path):
    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100))])
    cells = pd.DataFrame({"x_centroid": [20., np.nan, np.inf], "y_centroid": [20., 20., 20.]})
    result, summary = spatial.annotate_cells(cells, path, rules("tumor"),
        bed_labels=["tumor", "normal"], require_clean_path=True)
    assert result[["tumor_distance", "tumor_bed", "normal_bed"]].isna().all().all()


def test_direct_shortcut_matches_full_visibility_with_holes_ties_and_qc(monkeypatch):
    tissue = box(0, 0, 100, 100).difference(box(40, 35, 50, 65))
    targets = {"tumor": [{"id": 1, "geom": box(70, 5, 80, 95)},
                         {"id": 2, "geom": box(20, 5, 30, 15)},
                         {"id": 3, "geom": box(1, 97, 2, 98)}]}
    xy = np.random.default_rng(171).uniform(0, 100, (1800, 2))
    # Include blocked nearest points and a tie between two annotations.
    points = shapely.points(np.vstack((xy, [[10, 50], [50, 10], [75, 50], [np.nan, 3], [110, 50]])))
    scene = visibility.TissueVisibility(tissue, targets)
    actual, skipped = scene.measure(points, thresholds=(1, 40))
    assert scene.metadata["direct_cells"] > 500
    with monkeypatch.context() as patch:
        patch.setattr(visibility.TissueVisibility, "_direct_distances", lambda *a: (
            np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int)))
        expected, expected_skipped = visibility.TissueVisibility(tissue, targets).measure(points, thresholds=(1, 40))
    assert skipped == expected_skipped
    for threshold in (1, 40):
        np.testing.assert_allclose(actual[threshold][0], expected[threshold][0], atol=1e-10, equal_nan=True)
        np.testing.assert_array_equal(actual[threshold][1], expected[threshold][1])


@pytest.mark.parametrize("workers", [1, 4])
def test_parallel_traversal_matches_serial_with_multiple_roots_and_qc(workers):
    tissue = box(0, 0, 100, 100).difference(box(40, 35, 50, 65))
    targets = {"tumor": [{"id": 1, "geom": box(70, 5, 80, 95)}],
               "normal": [{"id": 2, "geom": box(20, 5, 30, 15)}]}
    xy = np.random.default_rng(71).uniform(0, 100, (4200, 2))
    points = shapely.points(np.vstack((xy, np.tile([[50, 0], [70, 5], [80, 95], [0, 0]], (20, 1)))))
    serial, _ = visibility.TissueVisibility(tissue, targets, workers=1).measure(points, thresholds=(1, 100))
    events = []
    parallel, _ = visibility.TissueVisibility(tissue, targets, workers=workers).measure(points, thresholds=(1, 100), progress=events.append)
    for threshold in (1, 100):
        np.testing.assert_array_equal(serial[threshold][0], parallel[threshold][0])
        np.testing.assert_array_equal(serial[threshold][1], parallel[threshold][1])
    assert any("Triangulating tissue" in message for message in events)
    assert any("cells completed" in message for message in events)
    assert len([message for message in events if "cells completed" in message]) <= 8


@pytest.mark.parametrize("overwrite", [True, False])
def test_independent_distances_share_preparation_but_preserve_studio_rules(tmp_path, overwrite):
    from dataclasses import replace

    path = _geojson(tmp_path, [_feature("tissue", box(0, 0, 100, 100).difference(box(40, 35, 50, 65))),
        _feature("tumor", box(70, 5, 80, 95)), _feature("normal", box(30, 0, 35, 100))])
    cells = pd.DataFrame({"x_centroid": [10., 32., 75., 75., np.nan, 120.], "y_centroid": [50.] * 6,
                          "normal_distance": [999.] * 6}, index=[10, 20, 30, 40, 50, 60])
    chosen = rules("tumor", "normal", "tissue")
    options = dict(require_clean_path=True, min_cells_per_bed_target=2, overwrite_existing=overwrite)
    expected, _ = spatial.annotate_cells(cells, path, [replace(r, distance=False) for r in chosen],
                                        bed_labels=["tumor", "normal"], **options)
    for rule in chosen:
        part, _ = spatial.annotate_cells(cells, path, [replace(rule, binary=False, roi_id=False, roi_size=False)],
                                        bed_labels=[], **options)
        expected[f"{rule.label}_distance"] = part[f"{rule.label}_distance"]
    events = []
    actual, _ = spatial.annotate_cells(cells, path, chosen, bed_labels=["tumor", "normal"],
                                      independent_distances=True, progress=lambda message, fraction=None: events.append((message, fraction)), **options)
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    fractions = [fraction for _, fraction in events if fraction is not None]
    assert fractions == sorted(fractions) and fractions[-1] == 1
