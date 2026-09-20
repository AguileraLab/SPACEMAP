"""Regression tests for tissue-scoped spatial cell tagging."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, mapping


# Load the focused backend without importing spacemap.__init__, which eagerly
# imports optional GPU/image stacks unrelated to these geometry tests.
MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "spacemap" / "cell_spatial_annotation.py"
SPEC = importlib.util.spec_from_file_location("spacemap_cell_spatial_annotation_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
spatial = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spatial
SPEC.loader.exec_module(spatial)

SpatialFeatureRule = spatial.SpatialFeatureRule
annotate_cells = spatial.annotate_cells


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


def _paired_rules() -> list[SpatialFeatureRule]:
    return [
        SpatialFeatureRule("tissue", binary=True, roi_id=True, roi_size=True),
        SpatialFeatureRule(
            "tumor", binary=True, roi_id=True, roi_size=True, distance=True,
            bed=True, bed_target="epithelia", group_by_roi=True,
        ),
        SpatialFeatureRule(
            "epithelia", binary=True, roi_id=True, roi_size=True, distance=True,
            bed=True, bed_target="tumor", group_by_roi=True,
        ),
    ]


def test_empty_tissue_piece_and_outside_cells_are_not_dragged_into_beds(tmp_path: Path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tissue", box(200, 0, 280, 100)),
        _feature("tumor", box(20, 20, 40, 40)),
        _feature("epithelia", box(60, 20, 80, 40)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [25, 65, 50, 220, 250, 150],
        "y_centroid": [25, 25, 70, 20, 70, 50],
    })

    tagged, _summary = annotate_cells(
        cells, annotation, _paired_rules(), min_cells_per_bed_target=1
    )

    empty_piece = tagged["tissue_roi"].eq(2)
    outside = tagged["tissue"].eq(0)
    for mask in (empty_piece, outside):
        assert tagged.loc[mask, ["tumor", "epithelia"]].eq(0).all().all()
        assert tagged.loc[mask, ["tumor_roi", "epithelia_roi"]].eq(0).all().all()
        assert tagged.loc[mask, ["tumor_distance", "epithelia_distance"]].isna().all().all()
        assert tagged.loc[mask, ["tumor_bed", "epithelia_bed"]].isna().all().all()
        assert tagged.loc[mask, ["tumor_bed_id", "epithelia_bed_id"]].isna().all().all()


def test_polygon_crossing_tissue_pieces_is_clipped_and_gets_unique_ids(tmp_path: Path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tissue", box(200, 0, 280, 100)),
        _feature("tumor", box(50, 10, 240, 90)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [60, 20, 220, 270, 150],
        "y_centroid": [20, 50, 20, 50, 50],
    })
    rules = [
        SpatialFeatureRule("tissue", binary=True, roi_id=True, roi_size=True),
        SpatialFeatureRule(
            "tumor", binary=True, roi_id=True, roi_size=True, distance=True,
            bed=True, bed_target="epithelia", group_by_roi=True,
        ),
    ]

    tagged, _summary = annotate_cells(
        cells, annotation, rules, min_cells_per_bed_target=1
    )

    left_source = tagged.index[tagged["x_centroid"].eq(60)][0]
    right_source = tagged.index[tagged["x_centroid"].eq(220)][0]
    assert tagged.loc[left_source, "tumor"] == 1
    assert tagged.loc[right_source, "tumor"] == 1
    assert tagged.loc[left_source, "tumor_roi"] != tagged.loc[right_source, "tumor_roi"]
    assert tagged.loc[left_source, "tumor_roi_size"] == 4000.0
    assert tagged.loc[right_source, "tumor_roi_size"] == 3200.0
    assert tagged.loc[left_source, "tumor_bed_id"] == tagged.loc[left_source, "tumor_roi"]
    assert tagged.loc[right_source, "tumor_bed_id"] == tagged.loc[right_source, "tumor_roi"]
    assert tagged.loc[tagged["x_centroid"].eq(150), "tumor_bed"].isna().all()


def test_crossing_polygon_fragment_with_zero_local_cells_cannot_seed_bed(tmp_path: Path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tissue", box(200, 0, 280, 100)),
        # One polygon crosses both pieces, but only the left clipped fragment
        # contains a cell centroid.
        _feature("tumor", box(50, 10, 240, 90)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [60, 20, 260, 270],
        "y_centroid": [20, 50, 20, 70],
    })
    rules = [
        SpatialFeatureRule("tissue", binary=True, roi_id=True),
        SpatialFeatureRule(
            "tumor", binary=True, roi_id=True, distance=True,
            bed=True, bed_target="epithelia", group_by_roi=True,
        ),
    ]

    tagged, _summary = annotate_cells(
        cells, annotation, rules, min_cells_per_bed_target=1
    )

    right_piece = tagged["tissue_roi"].eq(2)
    assert tagged.loc[right_piece, "tumor"].eq(0).all()
    assert tagged.loc[right_piece, "tumor_distance"].isna().all()
    assert tagged.loc[right_piece, ["tumor_bed", "epithelia_bed"]].isna().all().all()


def test_one_sided_bed_populates_winning_id_and_nan_losing_id(tmp_path: Path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [15, 20, 70],
        "y_centroid": [15, 20, 70],
    })
    rules = [
        SpatialFeatureRule("tissue", binary=True, roi_id=True),
        SpatialFeatureRule(
            "tumor", binary=True, roi_id=True, distance=True,
            bed=True, bed_target="epithelia", group_by_roi=True,
        ),
    ]

    tagged, _summary = annotate_cells(
        cells, annotation, rules, min_cells_per_bed_target=1
    )

    assert tagged["tumor_bed"].eq(1).all()
    assert tagged["epithelia_bed"].eq(0).all()
    assert tagged["tumor_bed_id"].eq(1).all()
    assert tagged["epithelia_bed_id"].isna().all()


def test_sparse_annotation_is_retained_but_does_not_seed_bed(tmp_path: Path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
        _feature("epithelia", box(70, 70, 90, 90)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [15, 20, 75, 80, 85, 50],
        "y_centroid": [15, 20, 75, 80, 85, 50],
    })

    filtered, summary = annotate_cells(
        cells, annotation, _paired_rules(), min_cells_per_bed_target=3
    )
    assert filtered["tumor"].sum() == 2  # binary annotation is preserved
    assert filtered["tumor_roi"].gt(0).sum() == 2
    assert filtered["tumor_distance"].notna().all()  # distance needs >=1, not bed threshold
    assert filtered["epithelia_bed"].eq(1).all()
    assert filtered["tumor_bed"].eq(0).all()
    assert summary["bed_target_fragments_below_threshold"] == {"tumor": 1}

    admitted, _summary = annotate_cells(
        cells, annotation, _paired_rules(), min_cells_per_bed_target=2
    )
    tumor_cells = admitted["tumor"].eq(1)
    assert admitted.loc[tumor_cells, "tumor_bed"].eq(1).all()


def test_reciprocal_bed_rule_order_does_not_change_output(tmp_path: Path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
        _feature("epithelia", box(70, 10, 90, 30)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [15, 75, 50],
        "y_centroid": [15, 15, 20],
    })
    rules = _paired_rules()

    first, _ = annotate_cells(cells, annotation, rules, min_cells_per_bed_target=1)
    second, _ = annotate_cells(
        cells, annotation, list(reversed(rules)), min_cells_per_bed_target=1
    )
    bed_columns = [
        "tumor_bed", "epithelia_bed", "tumor_bed_id", "epithelia_bed_id"
    ]
    pd.testing.assert_frame_equal(first[bed_columns], second[bed_columns])
    assert np.all(first[["tumor_bed", "epithelia_bed"]].sum(axis=1) == 1)


def test_three_beds_compete_once_with_stable_ties_and_local_ids(tmp_path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tissue", box(200, 0, 280, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
        _feature("normal", box(70, 10, 90, 30)),
        _feature("reactive", box(40, 70, 60, 90)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [20, 80, 50, 35, 65, 50, 50, 220, 150, np.nan],
        "y_centroid": [20, 20, 80, 20, 20, 65, 20, 50, 50, np.nan],
    }, index=np.arange(10) * 3 + 5)
    rules = [SpatialFeatureRule(name, roi_id=True, group_by_roi=True)
             for name in ("tumor", "normal", "reactive")]
    beds = ["tumor", "normal", "reactive"]
    tagged, summary = annotate_cells(cells, annotation, rules, bed_labels=beds)
    reversed_tagged, _ = annotate_cells(cells, annotation, rules, bed_labels=beds[::-1])
    columns = [f"{name}_bed" for name in beds]
    ids = [f"{name}_bed_id" for name in beds]
    pd.testing.assert_frame_equal(tagged, reversed_tagged)
    expected = ["tumor", "normal", "reactive", "tumor", "normal", "reactive", "tumor"]
    for row, name in zip(cells.index, expected):
        assert tagged.loc[row, f"{name}_bed"] == 1
        assert tagged.loc[row, columns].sum() == 1
        assert tagged.loc[row, ids].notna().sum() == 1
        assert tagged.loc[row, f"{name}_bed_id"] == 1
    assert tagged.loc[cells.index[7:], columns + ids].isna().all().all()
    assert set(summary["columns"]) == set(spatial.spatial_columns_for_rules(rules, bed_labels=beds))


def test_multi_beds_missing_and_sparse_classes_do_not_seed_targets(tmp_path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
        _feature("reactive", box(70, 70, 90, 90)),
    ])
    cells = pd.DataFrame({"x_centroid": [15, 20, 75, 50], "y_centroid": [15, 20, 75, 50]})
    tagged, summary = annotate_cells(
        cells, annotation, [], bed_labels=["tumor", "normal", "reactive"],
        min_cells_per_bed_target=2,
    )
    assert tagged["tumor_bed"].eq(1).all()
    assert tagged[["normal_bed", "reactive_bed"]].eq(0).all().all()
    assert tagged[["normal_bed_id", "reactive_bed_id"]].isna().all().all()
    assert summary["bed_target_fragments_below_threshold"] == {"reactive": 1}
    empty, _ = annotate_cells(
        cells, annotation, [], bed_labels=["tumor", "normal", "reactive"],
        min_cells_per_bed_target=3,
    )
    assert empty.filter(regex="_bed").isna().all().all()


@pytest.mark.parametrize("overwrite", [True, False])
def test_overwrite_replaces_whole_columns_including_blank_cells(tmp_path, overwrite):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
    ])
    cells = pd.DataFrame({
        "x_centroid": [20, 150], "y_centroid": [20, 20],
        "tumor_bed": [999, 999], "tumor_bed_id": [999, 999],
        "tumor_distance": [999, 999], "tumor": [999, 999],
        "untouched": ["first", "second"],
    })
    original = cells.copy(deep=True)
    tagged, summary = annotate_cells(
        cells, annotation, [SpatialFeatureRule("tumor", distance=True, group_by_roi=True)],
        bed_labels=["tumor", "normal", "reactive"], overwrite_existing=overwrite,
    )
    pd.testing.assert_frame_equal(cells, original)
    pd.testing.assert_series_equal(tagged["untouched"], original["untouched"])
    if overwrite:
        assert tagged.loc[0, "tumor_bed"] == 1
        assert tagged.loc[1, ["tumor_bed", "tumor_bed_id", "tumor_distance"]].isna().all()
        assert tagged["tumor"].tolist() == [1, 0]
        assert len(summary["columns_overwritten"]) == 4
    else:
        pd.testing.assert_frame_equal(tagged[original.columns], original)
        assert len(summary["columns_preserved"]) == 4
        assert "tumor_bed" not in summary["columns"]
    assert tagged.loc[0, "normal_bed"] == 0
    assert pd.isna(tagged.loc[1, "normal_bed"])


def test_explicit_two_class_beds_match_legacy_and_empty_selection_disables_beds(tmp_path):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
        _feature("epithelia", box(70, 10, 90, 30)),
    ])
    cells = pd.DataFrame({"x_centroid": [20, 80, 50, 150], "y_centroid": [20] * 4})
    rules = _paired_rules()
    legacy, _ = annotate_cells(cells, annotation, rules)
    selected, _ = annotate_cells(cells, annotation, rules, bed_labels=["tumor", "epithelia"])
    pd.testing.assert_frame_equal(legacy, selected)
    no_beds, _ = annotate_cells(cells, annotation, rules, bed_labels=[])
    assert not any("_bed" in name for name in no_beds)
    with pytest.raises(ValueError, match="at least two"):
        annotate_cells(cells, annotation, rules, bed_labels=["tumor"])
