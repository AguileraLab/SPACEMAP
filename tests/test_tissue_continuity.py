"""Regression tests for fragmented-tissue geometry and clean-path distances."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping

SRC = Path(__file__).resolve().parents[1] / "src" / "spacemap"


def _load(name: str, path: Path):
    """Import one backend module without pulling in spacemap.__init__.

    The package initialiser eagerly imports GPU and image stacks that these
    geometry tests do not need.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


continuity = _load("spacemap_tissue_continuity_test", SRC / "tissue_continuity.py")
_load("spacemap.spatial_visibility", SRC / "spatial_visibility.py")
spatial = _load("spacemap_cell_spatial_annotation_continuity_test", SRC / "cell_spatial_annotation.py")

SpatialFeatureRule = spatial.SpatialFeatureRule
annotate_cells = spatial.annotate_cells

MPP = 0.5083


def _dumbbell(neck_px: float = 100.0, side_px: float = 800.0) -> Polygon:
    """Two square masses joined by a bridge ``neck_px`` tall."""
    half = neck_px / 2.0
    mid = side_px / 2.0
    gap = 300.0
    right = side_px + gap
    return Polygon([
        (0, 0), (side_px, 0), (side_px, mid - half), (right, mid - half), (right, 0),
        (right + side_px, 0), (right + side_px, side_px), (right, side_px),
        (right, mid + half), (side_px, mid + half), (side_px, side_px), (0, side_px),
    ])


def _mass_with_bleb() -> Polygon:
    """A large mass with a small head on a thin stalk.

    The head detaches under opening exactly as a real bridge would, so this is
    the shape that a neck test based on separation alone gets wrong. Its area is
    0.017 mm2 at this calibration, below the default floor for a real section.
    """
    from shapely.ops import unary_union

    return unary_union([
        box(0, 0, 2000, 2000),          # the mass
        box(-200, 970, 0, 1030),        # a 60 px stalk, severed by opening
        box(-460, 870, -200, 1130),     # a 260 x 260 head
    ])


def _feature(label: str, geometry) -> dict:
    return {
        "type": "Feature",
        "properties": {"classification": {"name": label}},
        "geometry": mapping(geometry),
    }


def _geojson(tmp_path: Path, features: list[dict]) -> Path:
    path = tmp_path / "annotations.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


# ==========================================================================
# Geometry
# ==========================================================================

def test_split_pieces_orders_by_area_and_keeps_them_separate():
    small, large = box(0, 0, 10, 10), box(100, 100, 300, 300)
    pieces = continuity.split_pieces(MultiPolygon([small, large]))
    assert [round(piece.area) for piece in pieces] == [40000, 100]


def test_piece_metrics_reports_the_widest_circle_that_fits():
    # A 100 x 40 pixel strip admits a circle of radius 20 px, never more.
    metrics = continuity.piece_metrics(box(0, 0, 100, 40), MPP)
    assert metrics["inscribed_r_um"] == pytest.approx(20 * MPP, rel=1e-3)
    assert metrics["area_mm2"] == pytest.approx(4000 * MPP ** 2 / 1e6, rel=1e-6)


def test_find_necks_measures_the_real_bridge_width_not_the_search_width():
    # Searching at 100 um must still report the bridge's own width, or every
    # neck would appear to be exactly as wide as the threshold used to find it.
    necks, _ = continuity.find_necks(_dumbbell(neck_px=100.0), MPP, max_width_um=100.0)
    assert len(necks) == 1
    assert necks[0]["bridge_width_um"] == pytest.approx(100.0 * MPP, abs=4.0)
    assert necks[0]["smaller_side_fraction"] > 0.4


def test_find_necks_ignores_a_piece_with_no_narrow_bridge():
    necks, _ = continuity.find_necks(box(0, 0, 2000, 2000), MPP, max_width_um=100.0)
    assert necks == []


def test_find_necks_rejects_a_bleb_hanging_off_a_large_mass():
    """The bug this replaced: a bump on a stalk is not a neck.

    Opening detaches it just like a real bridge, so without a size test on both
    sides every shred around a mass was reported.
    """
    necks, _ = continuity.find_necks(_mass_with_bleb(), MPP, max_width_um=100.0)
    assert necks == []
    # With no size floor at all the same shape does separate, which is exactly
    # why the floor is what distinguishes a neck from a bleb.
    permissive, _ = continuity.find_necks(
        _mass_with_bleb(), MPP, max_width_um=100.0,
        min_lobe_area_mm2=0.0, min_lobe_fraction=0.0,
    )
    assert len(permissive) >= 1


def test_find_necks_reports_every_neck_in_a_chain():
    """Three masses in a row are joined by two bridges, so two necks."""
    a, b, c = box(0, 0, 800, 800), box(1100, 0, 1900, 800), box(2200, 0, 3000, 800)
    bridges = [box(800, 375, 1100, 425), box(1900, 375, 2200, 425)]
    from shapely.ops import unary_union

    chain = unary_union([a, b, c] + bridges)
    necks, _ = continuity.find_necks(chain, MPP, max_width_um=100.0)
    assert len(necks) == 2
    for neck in necks:
        assert neck["bridge_width_um"] == pytest.approx(50 * MPP, abs=4.0)


def test_raster_reports_equal_width_necks_as_independent_cuts():
    from shapely.ops import unary_union
    chain = unary_union([box(10, 40, 115, 320), box(145, 40, 250, 320), box(280, 40, 385, 320),
                         box(110, 140, 150, 160), box(245, 220, 285, 240)])
    necks, original = continuity.find_necks_raster(chain, 1., max_width_um=30., min_lobe_area_mm2=.001)
    assert len(necks) == 2 and original == [chain]
    cuts = [n['cut_geometry'] for n in necks]
    assert not cuts[0].intersects(cuts[1])
    for neck in necks:
        assert len(continuity.split_pieces(chain.difference(neck['cut_geometry']))) == 2
        assert neck['marker'].distance(Point(neck['x'], neck['y'])) < 1
    assert len(continuity.split_pieces(chain.difference(unary_union(cuts)))) == 3


def test_raster_keeps_parallel_strands_of_the_same_neck_together():
    from shapely.ops import unary_union
    ring = unary_union([box(0, 0, 3000, 300), box(0, 900, 3000, 1200),
                        box(0, 300, 120, 900), box(2880, 300, 3000, 900)])
    necks, _ = continuity.find_necks_raster(ring, MPP, max_width_um=100.)
    assert len(necks) == 1 and necks[0]['n_strands'] == 2 and necks[0]['separated']


def test_find_necks_detects_sections_held_together_by_two_strands():
    """A slit through a piece that stays joined at both ends.

    Cutting one strand leaves the sections attached by the other, so a detector
    that reports only what a single cut can part misses this entirely - which is
    the common case in fragmented tissue, not an edge case.
    """
    from shapely.ops import unary_union

    ring = unary_union([
        box(0, 0, 3000, 300),        # upper mass
        box(0, 900, 3000, 1200),     # lower mass
        box(0, 300, 120, 900),       # strand on the left
        box(2880, 300, 3000, 900),   # strand on the right
    ])
    necks, _ = continuity.find_necks(ring, MPP, max_width_um=100.0)
    assert len(necks) == 1
    neck = necks[0]
    assert neck["n_strands"] == 2
    assert neck["separated"] is True
    assert neck["bridge_width_um"] == pytest.approx(120 * MPP, abs=5.0)


def test_a_neck_is_reported_even_when_it_cannot_be_parted():
    """Reporting must never depend on the cut succeeding.

    Dropping what cannot be divided hides exactly the weakly joined pieces this
    step exists to surface.
    """
    from shapely.ops import unary_union

    ring = unary_union([
        box(0, 0, 3000, 300), box(0, 900, 3000, 1200),
        box(0, 300, 120, 900), box(2880, 300, 3000, 900),
    ])
    necks, parts = continuity.find_necks(ring, MPP, max_width_um=100.0, cut=False)
    assert len(necks) == 1
    assert parts == [ring], "detection alone must not alter the tissue"


def test_find_necks_gives_each_neck_a_drawable_marker():
    necks, _ = continuity.find_necks(_dumbbell(), MPP, max_width_um=100.0)
    marker = necks[0]["marker"]
    assert not marker.is_empty and marker.area > 0


def test_cut_at_necks_divides_across_the_bridge_not_along_it():
    piece = _dumbbell(neck_px=100.0)
    parts = continuity.cut_at_necks(piece, MPP, max_width_um=100.0)
    assert len(parts) == 2
    # A cut across the narrowest point loses only the blade's own footprint.
    assert sum(part.area for part in parts) == pytest.approx(piece.area, rel=0.01)


def test_cut_at_necks_leaves_a_piece_without_a_neck_alone():
    assert len(continuity.cut_at_necks(box(0, 0, 2000, 2000), MPP, max_width_um=100.0)) == 1


def test_select_pieces_filters_by_area_and_records_the_reason():
    tiny, big = box(0, 0, 60, 60), box(1000, 1000, 2000, 2000)
    kept, rejected = continuity.select_pieces(
        continuity.split_pieces(MultiPolygon([tiny, big])),
        {"microns_per_pixel": MPP, "filter_by_area": True, "min_area_mm2": 0.05},
    )
    assert [record["piece_id"] for record in kept] == [1]
    assert len(rejected) == 1 and "area" in rejected[0]["reason"]


def test_select_pieces_can_filter_by_the_inscribed_circle_instead():
    # A long thin shred passes an area test but cannot host one full circle.
    shred, blob = box(0, 0, 4000, 40), box(10000, 10000, 10600, 10600)
    kept, rejected = continuity.select_pieces(
        continuity.split_pieces(MultiPolygon([shred, blob])),
        {
            "microns_per_pixel": MPP,
            "filter_by_area": False,
            "filter_by_inscribed_circle": True,
            "min_inscribed_diameter_um": 100.0,
        },
    )
    assert len(kept) == 1 and len(rejected) == 1
    assert "circle" in rejected[0]["reason"]


def test_analyse_tissue_pieces_reports_necks_without_cutting_by_default():
    result = continuity.analyse_tissue_pieces(
        _dumbbell(), {"microns_per_pixel": MPP, "filter_by_area": False}
    )
    assert len(result["necks"]) == 1
    assert len(result["kept"]) == 1, "detection alone must not change the tissue"

    cut = continuity.analyse_tissue_pieces(
        _dumbbell(), {"microns_per_pixel": MPP, "filter_by_area": False, "cut_at_necks": True}
    )
    assert len(cut["kept"]) == 2


def test_unit_conversions_round_trip():
    assert continuity.px2_to_mm2(continuity.mm2_to_px2(0.05, MPP), MPP) == pytest.approx(0.05)


# ==========================================================================
# Clean-path distances
# ==========================================================================

def _u_shaped_annotation(tmp_path: Path) -> Path:
    """A U-shaped scrap plus a physically separate one, each with a target."""
    u_shape = Polygon([(0, 0), (100, 0), (100, 700), (500, 700), (500, 0), (600, 0), (600, 800), (0, 800)])
    return _geojson(tmp_path, [
        _feature("tissue", u_shape),
        _feature("tissue", box(900, 0, 1200, 300)),
        _feature("tumor", box(20, 20, 80, 80)),        # inside the left arm
        _feature("tumor", box(950, 50, 1150, 250)),    # on the separate scrap
    ])


def _cells() -> pd.DataFrame:
    return pd.DataFrame({
        "x_centroid": [50, 550, 1050, 50],
        "y_centroid": [300, 300, 150, 60],
    })


def _rules() -> list:
    return [
        SpatialFeatureRule("tissue", binary=True),
        SpatialFeatureRule("tumor", binary=True, distance=True, group_by_roi=True),
    ]


def test_clean_path_blanks_a_distance_that_would_cut_across_a_folded_piece(tmp_path: Path):
    annotation = _u_shaped_annotation(tmp_path)
    loose, _ = annotate_cells(_cells(), annotation, _rules(), tissue_label="tissue")
    strict, _ = annotate_cells(
        _cells(), annotation, _rules(), tissue_label="tissue", require_clean_path=True
    )

    # The cell in the far arm is only reachable the long way round the U.
    assert loose["tumor_distance"].notna().all()
    assert pd.isna(strict.loc[1, "tumor_distance"])
    # Every other cell keeps the measurement it already had.
    for position in (0, 2, 3):
        assert strict.loc[position, "tumor_distance"] == pytest.approx(
            loose.loc[position, "tumor_distance"]
        )


def test_clean_path_keeps_cells_inside_the_annotation(tmp_path: Path):
    strict, _ = annotate_cells(
        _cells(), _u_shaped_annotation(tmp_path), _rules(),
        tissue_label="tissue", require_clean_path=True,
    )
    # A cell within the target has no gap to cross, so it stays negative.
    assert strict.loc[3, "tumor_distance"] < 0


def test_clean_path_reports_how_many_cells_were_blanked(tmp_path: Path):
    _, summary = annotate_cells(
        _cells(), _u_shaped_annotation(tmp_path), _rules(),
        tissue_label="tissue", require_clean_path=True,
    )
    assert summary["clean_path_blanked"]["tumor_distance"] == 1


def test_clean_path_falls_back_to_a_target_it_can_actually_see(tmp_path: Path):
    # The nearest target is hidden behind a hole; a farther one is in clear view,
    # so the cell must be re-pointed rather than blanked.
    ring = Polygon(
        [(0, 0), (600, 0), (600, 900), (0, 900)],
        [[(240, 100), (360, 100), (360, 300), (240, 300)]],
    )
    annotation = _geojson(tmp_path, [
        _feature("tissue", ring),
        _feature("tumor", box(380, 180, 420, 220)),   # nearer, but behind the hole
        _feature("tumor", box(100, 600, 200, 700)),   # farther, in plain sight
    ])
    # Row 0 is the cell under test; the other two occupy the targets, because a
    # polygon with no cell centroid is treated as a sliver and never seeds a
    # distance.
    cells = pd.DataFrame({
        "x_centroid": [150.0, 400.0, 150.0],
        "y_centroid": [200.0, 200.0, 650.0],
    })
    loose, _ = annotate_cells(cells.copy(), annotation, _rules(), tissue_label="tissue")
    strict, _ = annotate_cells(
        cells.copy(), annotation, _rules(), tissue_label="tissue", require_clean_path=True
    )
    assert pd.notna(strict.loc[0, "tumor_distance"])
    assert strict.loc[0, "tumor_distance"] > loose.loc[0, "tumor_distance"]


def test_clean_path_is_inert_for_rules_that_are_not_grouped_by_tissue(tmp_path: Path):
    # Without a tissue piece to stay inside there is nothing to test against, so
    # the option must not silently drop measurements.
    rules = [
        SpatialFeatureRule("tissue", binary=True),
        SpatialFeatureRule("tumor", binary=True, distance=True, group_by_roi=False),
    ]
    annotation = _u_shaped_annotation(tmp_path)
    loose, _ = annotate_cells(_cells(), annotation, rules, tissue_label="tissue")
    strict, _ = annotate_cells(
        _cells(), annotation, rules, tissue_label="tissue", require_clean_path=True
    )
    pd.testing.assert_series_equal(loose["tumor_distance"], strict["tumor_distance"])


def test_default_run_is_unchanged_by_the_new_option(tmp_path: Path):
    """The feature must be opt-in: an existing project reproduces exactly."""
    annotation = _u_shaped_annotation(tmp_path)
    baseline, _ = annotate_cells(_cells(), annotation, _rules(), tissue_label="tissue")
    default, _ = annotate_cells(
        _cells(), annotation, _rules(), tissue_label="tissue", require_clean_path=False
    )
    pd.testing.assert_frame_equal(baseline, default)
