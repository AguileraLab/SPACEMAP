import json

import numpy as np
import pandas as pd
import pytest

from spacemap.background_detection import (
    THRESHOLD_SCHEMA_VERSION,
    align_region_aggregate,
    find_background,
    geojson_component_summary,
    masks_from_geojson,
    prepare_threshold_export,
    write_threshold_csv,
)
from spacemap.tissue_annotation_utils import PyramidImageReader


def test_geojson_region_unit_preserves_feature_or_splits_components(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"classification": {"name": "background"}},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]],
                    [[[10, 10], [14, 10], [14, 14], [10, 14], [10, 10]]],
                    # No pixel centre falls inside this component, so component
                    # mode must not export it as an empty sampling region.
                    [[[18.1, 18.1], [18.2, 18.1], [18.2, 18.2], [18.1, 18.2], [18.1, 18.1]]],
                ],
            },
        }],
    }
    path = tmp_path / "regions.geojson"
    path.write_text(json.dumps(geojson))

    feature_masks = masks_from_geojson(path, (20, 20), 1, ["background"])
    component_masks = masks_from_geojson(
        path, (20, 20), 1, ["background"], region_unit="component"
    )

    assert list(feature_masks) == ["background_1"]
    assert len(component_masks) == 2
    assert list(component_masks) == [
        "background_1_component_1",
        "background_1_component_2",
    ]
    assert sum(int(mask.sum()) for mask in component_masks.values()) == int(
        feature_masks["background_1"].sum()
    )

    summary = geojson_component_summary(
        path, ["background"], min_component_area_px=1.0
    )
    filtered_feature_masks = masks_from_geojson(
        path, (20, 20), 1, ["background"],
        min_component_area_px=1.0,
    )
    assert summary["components"] == 3
    assert summary["kept_components"] == 2
    assert summary["ignored_components"] == 1
    assert int(filtered_feature_masks["background_1"].sum()) == int(
        feature_masks["background_1"].sum()
    )


def test_triangle_matches_imagej_histogram_and_does_not_floor_or_fallback():
    # A right-skewed 8-bit histogram whose ImageJ Triangle split is bin 32.
    counts = np.concatenate([
        np.zeros(10, dtype=int),
        np.arange(210, 0, -10, dtype=int),
        np.zeros(225, dtype=int),
    ])
    pixels = np.repeat(np.arange(256, dtype=np.uint8), counts)

    detected = find_background(
        pixels, method="Triangle", value_range=(0, 255), imagej_dtype=np.uint8
    )
    constant = find_background(
        np.zeros(100, dtype=np.uint16), method="Triangle",
        value_range=(0, 0), imagej_dtype=np.uint16,
    )

    assert detected["threshold"] == 32.0
    assert detected["fallback_status"] == "none"
    # The Groovy script has thresholdFloor=null; the old GUI incorrectly
    # forced every automatic result to at least 5.
    assert constant["threshold"] == 0.0
    assert constant["fallback_status"] == "none"


def test_non_power_of_two_downsample_uses_exact_target_shape():
    reader = PyramidImageReader.__new__(PyramidImageReader)
    reader._levels_per_channel = 4
    reader._level_shapes = [(100, 200), (50, 100), (25, 50), (13, 25)]
    reader.full_shape = (100, 200)

    assert reader.target_shape(4) == (25, 50)
    # The former implementation silently served the 4x pyramid for a 5x
    # request, unlike the Groovy script's thresholdDownsample=5 setting.
    assert reader.target_shape(5) == (20, 40)


def _region_table():
    return pd.DataFrame({
        "Marker": ["A", "B"],
        "threshold": [20.0, 80.0],
        "manual_override": [np.nan, 99.0],
        "detected_aggregation": ["median_across_region_thresholds"] * 2,
        "source_mode": ["geojson_region_thresholds"] * 2,
        "fallback_status": ["none"] * 2,
        "region_unit": ["feature"] * 2,
        "region::background_1": [10.0, 20.0],
        "region::background_2": [20.0, 80.0],
        "region::background_3": [60.0, 100.0],
    })


def test_individual_export_preserves_regions_and_manual_override(tmp_path):
    prepared = prepare_threshold_export(_region_table())

    assert prepared.attrs["polygon_columns"] == ["Polygon_1", "Polygon_2", "Polygon_3"]
    assert prepared.loc[0, ["Polygon_1", "Polygon_2", "Polygon_3"]].tolist() == [10.0, 20.0, 60.0]
    assert prepared.loc[1, ["Polygon_1", "Polygon_2", "Polygon_3"]].tolist() == [20.0, 80.0, 100.0]
    assert prepared["threshold_detected"].tolist() == [20.0, 80.0]
    assert prepared["threshold"].tolist() == [20.0, 99.0]
    assert prepared["threshold_final"].tolist() == [20.0, 99.0]
    assert prepared["manual_override"].isna().tolist() == [True, False]
    assert prepared["schema_version"].eq(THRESHOLD_SCHEMA_VERSION).all()
    assert prepared["export_mode"].eq("individual").all()
    assert prepared["aggregation"].eq("individual_region_thresholds").all()

    output = tmp_path / "thresholds.csv"
    write_threshold_csv(_region_table(), output)
    written = pd.read_csv(output)
    assert written.columns.tolist() == [
        "Marker",
        "Polygon_1", "Polygon_2", "Polygon_3"
    ]
    assert written.loc[0, ["Polygon_1", "Polygon_2", "Polygon_3"]].tolist() == [
        10.0, 20.0, 60.0
    ]
    # A final manual gate must give the same result no matter which RESOLVE
    # polygon statistic is selected later.
    assert written.loc[1, ["Polygon_1", "Polygon_2", "Polygon_3"]].tolist() == [
        99.0, 99.0, 99.0
    ]
    assert output.read_text().splitlines()[1] == "A,10.00,20.00,60.00"


def test_collapsed_exports_use_region_statistics_and_keep_provenance():
    median = prepare_threshold_export(_region_table(), export_mode="median")
    mean = prepare_threshold_export(_region_table(), export_mode="mean")

    assert median.attrs["polygon_columns"] == ["Polygon_1"]
    assert median.loc[0, "Polygon_1"] == 20.0
    assert median.loc[1, "Polygon_1"] == 80.0
    assert median["threshold_final"].tolist() == [20.0, 99.0]
    assert median["aggregation"].eq("median_across_region_thresholds").all()

    assert mean.loc[0, "Polygon_1"] == 30.0
    assert mean.loc[1, "Polygon_1"] == round(200.0 / 3.0, 2)
    assert mean["threshold_final"].round(4).tolist() == [30.0, 99.0]
    # Scalar-first consumers see exactly the selected mean (or manual
    # override), never the stale median detector aggregate.
    assert mean["threshold"].round(4).tolist() == [30.0, 99.0]
    assert mean.loc[0, "threshold"] == mean.loc[0, "Polygon_1"]
    assert mean["aggregation"].eq("mean_across_region_thresholds").all()
    assert mean.attrs["region_columns"] == [
        "region::background_1", "region::background_2", "region::background_3"
    ]


@pytest.mark.parametrize(
    ("mode", "expected", "aggregation"),
    [
        ("p05", [11.0, 26.0], "5th_percentile_across_region_thresholds"),
        ("p25", [15.0, 50.0], "25th_percentile_across_region_thresholds"),
        ("p75", [40.0, 90.0], "75th_percentile_across_region_thresholds"),
    ],
)
def test_collapsed_percentile_exports(mode, expected, aggregation):
    prepared = prepare_threshold_export(_region_table(), export_mode=mode)

    assert prepared.attrs["polygon_columns"] == ["Polygon_1"]
    assert prepared["Polygon_1"].tolist() == expected
    assert prepared["aggregation"].eq(aggregation).all()
    # The manual override remains authoritative in the scalar columns.
    assert prepared["threshold_final"].tolist() == [expected[0], 99.0]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("median", [20.0, 99.0]),
        ("mean", [30.0, 99.0]),
        ("p05", [11.0, 99.0]),
        ("p25", [15.0, 99.0]),
        ("p75", [40.0, 99.0]),
    ],
)
def test_one_value_csv_uses_selected_statistic_and_legacy_header(tmp_path, mode, expected):
    output = tmp_path / f"{mode}.csv"

    write_threshold_csv(_region_table(), output, export_mode=mode)
    written = pd.read_csv(output)

    assert written.columns.tolist() == ["Marker", "Polygon_1"]
    assert written["Polygon_1"].tolist() == expected


def test_batch_groups_keep_their_own_polygon_column_count(tmp_path):
    three_polygons = _region_table().assign(image="image_a")
    four_polygons = _region_table().assign(
        image="image_b",
        **{"region::background_4": [80.0, 120.0]},
    )
    combined = pd.concat([three_polygons, four_polygons], ignore_index=True)

    expected_counts = {"image_a": 3, "image_b": 4}
    for image, group in combined.groupby("image"):
        output = tmp_path / f"{image}.ome.tiff_-_Merged_Image.csv"
        write_threshold_csv(group, output, export_mode="individual")
        written = pd.read_csv(output)
        assert written.columns.tolist() == ["Marker"] + [
            f"Polygon_{position}"
            for position in range(1, expected_counts[image] + 1)
        ]


def test_changing_export_selector_realigns_review_and_export_without_redetection():
    # Detection initially ran under the default individual/median choice.
    detected = _region_table()
    assert detected["threshold"].tolist() == [20.0, 80.0]

    review = align_region_aggregate(detected, export_mode="mean")
    exported = prepare_threshold_export(review, export_mode="mean")

    assert review["threshold"].round(4).tolist() == [30.0, round(200.0 / 3.0, 4)]
    assert exported["threshold_final"].round(4).tolist() == [30.0, 99.0]
    assert exported["threshold"].equals(exported["threshold_final"])


def test_single_region_and_pooled_exports_preserve_override():
    single = pd.DataFrame({
        "Marker": ["A"],
        "threshold": [12.0],
        "manual_override": [55.0],
        "region::background_1": [12.0],
    })
    individual = prepare_threshold_export(single)
    assert individual.loc[0, "Polygon_1"] == 12.0
    assert individual.loc[0, "threshold_final"] == 55.0

    pooled = prepare_threshold_export(pd.DataFrame({
        "Marker": ["A"], "threshold": [15.0], "manual_override": [21.0]
    }))
    assert pooled.loc[0, "Polygon_1"] == 21.0
    assert pooled.loc[0, "threshold_final"] == 21.0
    assert pooled.loc[0, "source_mode"] == "pooled_pixels"
    assert pooled.loc[0, "export_mode"] == "pooled"


def test_detector_fallback_blocks_export_until_manually_overridden():
    detected = find_background(np.array([], dtype=float), method="Triangle")
    assert detected["threshold"] == 150.0
    assert detected["fallback_status"] == "empty_pixels_fallback_150"

    fallback = pd.DataFrame({
        "Marker": ["A"],
        "threshold": [detected["threshold"]],
        "threshold_fallback_status": [detected["fallback_status"]],
        "threshold_error": [detected["threshold_error"]],
        "manual_override": [np.nan],
    })
    with pytest.raises(ValueError, match="manual override"):
        prepare_threshold_export(fallback)

    fallback.loc[0, "manual_override"] = 42.0
    prepared = prepare_threshold_export(fallback)
    assert prepared.loc[0, "threshold_detected"] == 150.0
    assert prepared.loc[0, "threshold_final"] == 42.0
    assert prepared.loc[0, "threshold"] == 42.0
    assert prepared.loc[0, "Polygon_1"] == 42.0
    assert prepared.loc[0, "threshold_fallback_status"] == "empty_pixels_fallback_150"
