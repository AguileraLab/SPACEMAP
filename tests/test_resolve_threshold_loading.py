import numpy as np
import pandas as pd

from spacemap_gui.tabs.tab7_phenotyping.common import (
    _extract_thresholds_from_csv,
    _load_thresholds_table,
    _match_threshold_stem_to_batches,
    _numbered_polygon_columns,
    _preferred_threshold_directory,
    _threshold_settings_fingerprint,
    _threshold_table_coverage,
)
from spacemap.ph_thres_utils import (
    _extract_thresholds_v2,
    _numbered_threshold_columns,
    apply_thresholds_per_batch,
)
from spacemap.background_detection import write_threshold_csv


def _extract(frame, statistic="median", source="Auto", value_col=None, metadata=False):
    return _extract_thresholds_from_csv(
        frame,
        ["CD8"],
        "Marker",
        value_col,
        "Polygon_",
        statistic,
        source,
        return_metadata=metadata,
    )


def test_auto_source_precedence_explicit_then_override_then_polygons_then_scalar():
    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "chosen": [900.0],
        "manual_override": [800.0],
        "Polygon_1": [10.0],
        "Polygon_2": [20.0],
        "threshold_final": [700.0],
        "threshold": [600.0],
    })

    assert _extract(frame, value_col="chosen")["CD8"] == 900.0
    assert _extract(frame)["CD8"] == 800.0

    frame["manual_override"] = np.nan
    assert _extract(frame)["CD8"] == 15.0

    frame = frame.drop(columns=["Polygon_1", "Polygon_2"])
    assert _extract(frame)["CD8"] == 700.0


def test_auto_true_override_flag_uses_final_scalar():
    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "override": [True],
        "Polygon_1": [10.0],
        "Polygon_2": [20.0],
        "threshold_final": [123.0],
        "threshold": [99.0],
    })

    thresholds, metadata = _extract(frame, metadata=True)

    assert thresholds["CD8"] == 123.0
    assert metadata["CD8"]["source"] == "Manual override"


def test_explicit_polygon_source_ignores_overrides_and_scalars():
    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "manual_override": [800.0],
        "Polygon_1": [10.0],
        "Polygon_2": [30.0],
        "threshold_final": [700.0],
    })

    thresholds, metadata = _extract(frame, source="Polygon values", metadata=True)

    assert thresholds["CD8"] == 20.0
    assert metadata["CD8"] == {
        "source": "Polygon values",
        "aggregation": "Median",
        "polygon_count": 2,
    }


def test_explicit_final_scalar_prefers_threshold_final_and_ignores_polygons():
    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "manual_override": [800.0],
        "Polygon_1": [10.0],
        "threshold_final": [700.0],
        "threshold": [600.0],
    })

    thresholds, metadata = _extract(frame, source="Final scalar", metadata=True)

    assert thresholds["CD8"] == 700.0
    assert metadata["CD8"]["source"] == "Final scalar (threshold_final)"
    assert metadata["CD8"]["aggregation"] == "Direct scalar"

    frame["chosen"] = 321.0
    assert _extract(
        frame, source="Final scalar", value_col="chosen"
    )["CD8"] == 321.0


def test_polygon_statistics_include_p05_p25_and_p75():
    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "Polygon_1": [0.0],
        "Polygon_2": [100.0],
    })

    assert _extract(frame, "p05", "Polygon values")["CD8"] == 5.0
    assert _extract(frame, "percentile_25", "Polygon values")["CD8"] == 25.0
    assert _extract(frame, "p75", "Polygon values")["CD8"] == 75.0


def test_one_polygon_is_loaded_directly_for_every_statistic():
    frame = pd.DataFrame({"Marker": ["CD8"], "Polygon_1": [42.0]})

    for statistic in ("median", "mean", "min", "p05", "p25", "p75"):
        thresholds, metadata = _extract(
            frame, statistic, "Polygon values", metadata=True
        )
        assert thresholds["CD8"] == 42.0
        assert metadata["CD8"]["aggregation"] == "Direct (one polygon)"
        assert metadata["CD8"]["polygon_count"] == 1


def test_tab2_legacy_csv_round_trip_supports_resolve_statistics(tmp_path):
    detected = pd.DataFrame({
        "Marker": ["CD8"],
        "threshold": [50.0],
        "manual_override": [np.nan],
        "region::tissue_1": [0.0],
        "region::tissue_2": [100.0],
    })
    separate_path = tmp_path / "separate.csv"
    write_threshold_csv(detected, separate_path, export_mode="individual")
    separate = pd.read_csv(separate_path)

    assert separate.columns.tolist() == ["Marker", "Polygon_1", "Polygon_2"]
    assert _extract(separate, "mean", "Polygon values")["CD8"] == 50.0
    assert _extract(separate, "p05", "Polygon values")["CD8"] == 5.0

    combined_path = tmp_path / "combined.csv"
    write_threshold_csv(detected, combined_path, export_mode="p05")
    combined = pd.read_csv(combined_path)

    assert combined.columns.tolist() == ["Marker", "Polygon_1"]
    for statistic in ("median", "mean", "min", "p05", "p25", "p75"):
        assert _extract(combined, statistic, "Polygon values")["CD8"] == 5.0


def test_only_numbered_polygon_columns_are_parsed_with_literal_custom_prefix():
    columns = ["Polygon_10", "Polygon_Count", "Polygon_2", "Polygon_note"]
    assert _numbered_polygon_columns(columns, "Polygon_") == ["Polygon_2", "Polygon_10"]
    assert _numbered_threshold_columns(columns, "Polygon") == ["Polygon_2", "Polygon_10"]

    custom = ["Gate+_2", "GateX_1", "Gate+_Count", "Gate+_1"]
    assert _numbered_polygon_columns(custom, "Gate+") == ["Gate+_1", "Gate+_2"]

    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "Polygon_1": [42.0],
        "Polygon_Count": [999.0],
    })
    thresholds, metadata = _extract(frame, "mean", "Polygon values", metadata=True)
    assert thresholds["CD8"] == 42.0
    assert metadata["CD8"]["polygon_count"] == 1


def test_core_v2_extractor_ignores_polygon_count_and_supports_auto_and_p25():
    frame = pd.DataFrame({
        "Marker": ["CD8"],
        "manual_override": [np.nan],
        "Polygon_1": [0.0],
        "Polygon_2": [100.0],
        "Polygon_Count": [999.0],
        "threshold_final": [80.0],
    })
    assert _extract_thresholds_v2(frame, ["CD8"], "p25")["CD8"] == 25.0
    frame["manual_override"] = 70.0
    assert _extract_thresholds_v2(frame, ["CD8"], "p25")["CD8"] == 70.0


def test_threshold_settings_fingerprint_changes_with_source_settings(tmp_path):
    base = _threshold_settings_fingerprint(
        tmp_path, "Auto", "median", "Marker", None, "Polygon", ["CD8"]
    )
    assert base == _threshold_settings_fingerprint(
        tmp_path, "Auto", "median", "Marker", None, "Polygon_", ["CD8"]
    )
    assert base != _threshold_settings_fingerprint(
        tmp_path, "Polygon values", "median", "Marker", None, "Polygon", ["CD8"]
    )
    assert base != _threshold_settings_fingerprint(
        tmp_path, "Auto", "p25", "Marker", None, "Polygon", ["CD8"]
    )


def test_loaded_table_exposes_source_aggregation_and_polygon_count(tmp_path):
    pd.DataFrame({
        "Marker": ["CD8"],
        "Polygon_1": [10.0],
        "Polygon_2": [20.0],
    }).to_csv(tmp_path / "sample.ome.tiff_-_Merged_Image.csv", index=False)

    table = _load_thresholds_table(
        str(tmp_path), ["CD8"], ["sample"], "Marker", None,
        "Polygon_", "p25", "Auto",
    )

    assert table.loc[0, "CD8"] == 12.5
    assert table.loc[0, "Threshold_Source"] == "Polygon values"
    assert table.loc[0, "Threshold_Aggregation"] == "25th percentile (p25)"
    assert table.loc[0, "Polygon_Count"] == "2"


def test_batch_matching_does_not_alias_alphanumeric_prefixes(tmp_path):
    pd.DataFrame({"Marker": ["CD8"], "Polygon_1": [10.0]}).to_csv(
        tmp_path / "sample.csv", index=False
    )
    pd.DataFrame({"Marker": ["CD8"], "Polygon_1": [20.0]}).to_csv(
        tmp_path / "sample2.csv", index=False
    )
    pd.DataFrame({"Marker": ["CD8"], "Polygon_1": [30.0]}).to_csv(
        tmp_path / "P1.csv", index=False
    )
    pd.DataFrame({"Marker": ["CD8"], "Polygon_1": [40.0]}).to_csv(
        tmp_path / "P11.csv", index=False
    )

    table = _load_thresholds_table(
        str(tmp_path), ["CD8"], ["sample", "sample2", "P1", "P11"],
        "Marker", None, "Polygon", "median", "Auto",
    ).set_index("batch_id")

    assert table.index.is_unique
    assert table["CD8"].to_dict() == {
        "P1": 30.0, "P11": 40.0, "sample": 10.0, "sample2": 20.0,
    }


def test_qupath_suffixes_still_normalize_to_exact_batch():
    assert _match_threshold_stem_to_batches(
        "sample.ome.tiff_-_Merged_Image",
        ["sample.ome.tiff_-_Fixed_Nolan_Image", "sample2"],
    ) == ["sample.ome.tiff_-_Fixed_Nolan_Image"]


def test_preferred_threshold_directory_ignores_empty_legacy_folder(tmp_path):
    preferred = tmp_path / "data" / "thresholds"
    singular = tmp_path / "data" / "threshold"
    legacy = tmp_path / "data" / "boundries"
    preferred.mkdir(parents=True)
    singular.mkdir(parents=True)
    legacy.mkdir(parents=True)

    assert _preferred_threshold_directory(tmp_path) == preferred

    (legacy / "legacy.csv").write_text("Marker,Polygon_1\nCD8,1\n")
    assert _preferred_threshold_directory(tmp_path) == legacy

    (singular / "singular.csv").write_text("Marker,Polygon_1\nCD8,1.5\n")
    assert _preferred_threshold_directory(tmp_path) == singular

    (preferred / "current.csv").write_text("Marker,Polygon_1\nCD8,2\n")
    assert _preferred_threshold_directory(tmp_path) == preferred


def test_threshold_coverage_requires_every_batch_marker_and_finite_value():
    table = pd.DataFrame({
        "batch_id": ["A", "B"],
        "CD8": [10.0, 20.0],
        "CD4": [30.0, np.nan],
    })

    report = _threshold_table_coverage(table, ["CD8", "CD4"], ["A", "B"])

    assert not report["complete"]
    assert report["missing_by_batch"] == {"B": ["CD4"]}


def test_threshold_coverage_rejects_missing_duplicate_and_ambiguous_global_rows():
    duplicate = pd.DataFrame({"batch_id": ["A", "A"], "CD8": [1.0, 2.0]})
    report = _threshold_table_coverage(duplicate, ["CD8"], ["A", "B"])
    assert report["duplicate_batches"] == ["A"]
    assert report["missing_batches"] == ["B"]

    global_report = _threshold_table_coverage(duplicate, ["CD8"], None)
    assert global_report["ambiguous_global"]
    assert not global_report["complete"]


def test_direct_batch_utility_uses_v2_values_and_fails_closed(tmp_path):
    data = pd.DataFrame({"batch_id": ["A", "B"], "CD8": [30.0, 30.0]})
    pd.DataFrame({
        "Marker": ["CD8"], "Polygon_1": [0.0], "Polygon_2": [100.0],
        "Polygon_Count": [2],
    }).to_csv(tmp_path / "A.csv", index=False)
    pd.DataFrame({
        "Marker": ["CD8"], "manual_override": [40.0], "Polygon_1": [0.0],
        "Polygon_2": [100.0], "threshold_final": [40.0],
    }).to_csv(tmp_path / "B.csv", index=False)

    binary, _ = apply_thresholds_per_batch(
        data, "resolve", tmp_path, ["CD8"], "p25", None,
        (False, False, False, False), "batch_id", threshold_source="Auto",
    )
    assert binary["CD8_bin_resolve"].tolist() == [1, 0]

    (tmp_path / "B.csv").unlink()
    try:
        apply_thresholds_per_batch(
            data, "resolve", tmp_path, ["CD8"], "p25", None,
            (False, False, False, False), "batch_id", threshold_source="Auto",
        )
    except ValueError as error:
        assert "No threshold CSV matched batch 'B'" in str(error)
    else:
        raise AssertionError("missing batch threshold did not fail closed")
