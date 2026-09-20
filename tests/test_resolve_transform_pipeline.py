import numpy as np
import pandas as pd
import pytest

from spacemap_gui.tabs.tab7_phenotyping.resolve_clustering import (
    RESOLVE_NORMALIZATION_OPTIONS,
    RESOLVE_TRANSFORM_OPTIONS,
    _apply_resolve_without_thresholds,
    _apply_thresholds_from_table,
    _canonical_resolve_transform,
    _migrate_resolve_pipeline_config,
    _normalize_resolve_pipeline,
    _resolve_pipeline_errors,
    _resolve_pipeline_preview,
)
from spacemap.ph_thres_utils import apply_thresholds_per_batch


def test_resolve_pipeline_exposes_only_the_simple_transform_and_step_choices():
    assert tuple(RESOLVE_TRANSFORM_OPTIONS) == ("None", "ArcSinh", "Log")
    assert tuple(RESOLVE_NORMALIZATION_OPTIONS) == (
        "None",
        "Z1",
        "Z2",
        "CDF",
        "NLog",
    )


@pytest.mark.parametrize(
    ("stored_value", "expected"),
    [
        (None, "None"),
        ("", "None"),
        ("None", "None"),
        ("identity_residual", "None"),
        ("ArcSinh", "ArcSinh"),
        ("arcsinh", "ArcSinh"),
        ("asinh", "ArcSinh"),
        ("asinh_difference", "ArcSinh"),
        ("Log", "Log"),
        ("log1p", "Log"),
        ("log1p_difference", "Log"),
    ],
)
def test_transform_names_and_legacy_values_are_canonicalized(stored_value, expected):
    assert _canonical_resolve_transform(stored_value) == expected


def test_unknown_transform_is_rejected_clearly():
    with pytest.raises(ValueError, match="transformation"):
        _canonical_resolve_transform("square-root")


@pytest.mark.parametrize(
    ("transform", "runtime_name"),
    [
        ("None", "identity_residual"),
        ("ArcSinh", "asinh_difference"),
        ("Log", "log1p_difference"),
    ],
)
def test_transform_choice_maps_to_the_runtime_method(transform, runtime_name):
    pipeline = _normalize_resolve_pipeline(
        transform,
        5.0,
        ["Z1", "None", "None", "None"],
    )

    assert pipeline["transformation"] == transform
    assert pipeline["tie_score_transform"] == runtime_name


def test_arcsinh_cofactor_defaults_to_five_and_must_be_positive():
    pipeline = _normalize_resolve_pipeline(
        "ArcSinh",
        None,
        ["Z1", "None", "None", "None"],
    )
    assert pipeline["tie_score_cofactor"] == 5.0

    with pytest.raises(ValueError, match="cofactor"):
        _normalize_resolve_pipeline(
            "ArcSinh",
            0,
            ["Z1", "None", "None", "None"],
        )


@pytest.mark.parametrize(
    ("slots", "expected_order", "expected_flags"),
    [
        (
            ["Z1", "None", "None", "None"],
            ["Z1"],
            (True, False, False, False),
        ),
        (
            ["Z1", "Z2", "CDF", "NLog"],
            ["Z1", "Z2", "CDF", "NLog"],
            (True, True, True, True),
        ),
        (
            ["CDF", "None", "Z1", "NLog"],
            ["CDF", "Z1", "NLog"],
            (True, False, True, True),
        ),
        (
            ["None", "None", "None", "None"],
            [],
            (False, False, False, False),
        ),
    ],
)
def test_four_slots_directly_define_order_and_enabled_flags(
    slots,
    expected_order,
    expected_flags,
):
    pipeline = _normalize_resolve_pipeline("None", 5.0, slots)

    assert pipeline["norm_order"] == expected_order
    assert (
        pipeline["norm_z1"],
        pipeline["norm_z2"],
        pipeline["norm_cdf"],
        pipeline["norm_nlog"],
    ) == expected_flags


def test_duplicate_normalization_steps_are_rejected():
    with pytest.raises(ValueError, match="more than once"):
        _normalize_resolve_pipeline(
            "None",
            5.0,
            ["Z1", "Z1", "None", "None"],
        )


def test_nlog_requires_cdf_before_it():
    invalid = _normalize_resolve_pipeline(
        "ArcSinh", 5.0, ["Z1", "NLog", "None", "None"]
    )
    valid = _normalize_resolve_pipeline(
        "ArcSinh", 5.0, ["Z1", "CDF", "NLog", "None"]
    )

    assert _resolve_pipeline_errors(invalid) == ["NLog must come after CDF."]
    assert _resolve_pipeline_errors(valid) == []


@pytest.mark.parametrize(
    ("transform", "slots", "expected"),
    [
        ("None", ["Z1", "None", "None", "None"], "Z1"),
        (
            "ArcSinh",
            ["Z1", "Z2", "CDF", "NLog"],
            "ArcSinh → Z1 → Z2 → CDF → NLog",
        ),
        ("Log", ["None", "None", "None", "None"], "Log"),
        ("None", ["None", "None", "None", "None"], "None"),
    ],
)
def test_pipeline_preview_uses_short_labels(transform, slots, expected):
    assert _resolve_pipeline_preview(transform, slots) == expected


def test_legacy_config_migrates_to_four_slots_and_ignores_stale_flags():
    migrated = _migrate_resolve_pipeline_config({
        "tie_score_transform": "asinh_difference",
        "tie_score_cofactor": 500.0,
        "norm_order": ["Z1"],
        # These old checkboxes can disagree with the selected order. The four
        # slots are now the only source of truth.
        "norm_z1": True,
        "norm_z2": True,
        "norm_cdf": True,
        "norm_nlog": True,
    })

    assert migrated["transformation"] == "ArcSinh"
    assert migrated["tie_score_cofactor"] == 500.0
    assert migrated["normalization_slots"] == ["Z1", "None", "None", "None"]
    assert migrated["norm_order"] == ["Z1"]
    assert (
        migrated["norm_z1"],
        migrated["norm_z2"],
        migrated["norm_cdf"],
        migrated["norm_nlog"],
    ) == (True, False, False, False)


def test_legacy_flags_are_used_only_when_no_saved_order_exists():
    migrated = _migrate_resolve_pipeline_config({
        "norm_z1": True,
        "norm_z2": False,
        "norm_cdf": True,
        "norm_nlog": False,
    })

    assert migrated["normalization_slots"] == ["Z1", "CDF", "None", "None"]
    assert migrated["norm_order"] == ["Z1", "CDF"]
    assert migrated["tie_score_cofactor"] == 5.0


def _runtime_input_and_thresholds():
    data = pd.DataFrame({
        "batch_id": ["A"] * 5,
        "Marker_A": [0.0, 5.0, 10.0, 15.0, 50.0],
        "Marker_B": [4.0, 5.0, 6.0, 25.0, 100.0],
    })
    thresholds = {"Marker_A": 10.0, "Marker_B": 5.0}
    wide_thresholds = pd.DataFrame([{**thresholds, "batch_id": "A"}])
    return data, thresholds, wide_thresholds


def _write_direct_threshold_table(directory, thresholds):
    pd.DataFrame({
        "Marker": list(thresholds),
        "Polygon_1": list(thresholds.values()),
    }).to_csv(directory / "A.csv", index=False)


@pytest.mark.parametrize(
    ("runtime_transform", "expected_transform"),
    [
        ("identity_residual", lambda values: values),
        # No explicit cofactor is passed below: ArcSinh must use the new
        # introductory-transformation default of five in both runtime paths.
        ("asinh_difference", lambda values: np.arcsinh(values / 5.0)),
        ("log1p_difference", np.log1p),
    ],
)
def test_gui_and_direct_runtime_paths_apply_the_same_transform_with_no_normalization(
    tmp_path,
    runtime_transform,
    expected_transform,
):
    data, thresholds, wide_thresholds = _runtime_input_and_thresholds()
    markers = list(thresholds)
    _write_direct_threshold_table(tmp_path, thresholds)

    gui_binary, gui_scores = _apply_thresholds_from_table(
        data,
        wide_thresholds,
        markers,
        "batch_id",
        (False, False, False, False),
        [],
        "runtime",
        tie_score_transform=runtime_transform,
    )
    direct_binary, direct_scores = apply_thresholds_per_batch(
        data,
        "runtime",
        tmp_path,
        markers,
        "median",
        None,
        (False, False, False, False),
        "batch_id",
        norm_order=[],
        tie_score_transform=runtime_transform,
    )

    threshold_vector = np.array([thresholds[marker] for marker in markers])
    raw_values = data[markers].to_numpy(dtype=float)
    expected_scores = np.clip(
        expected_transform(raw_values) - expected_transform(threshold_vector),
        0,
        None,
    )
    expected_binary = (raw_values > threshold_vector).astype(int)

    np.testing.assert_array_equal(gui_binary.to_numpy(), expected_binary)
    np.testing.assert_array_equal(direct_binary.to_numpy(), expected_binary)
    np.testing.assert_allclose(gui_scores.to_numpy(), expected_scores)
    np.testing.assert_allclose(direct_scores.to_numpy(), expected_scores)
    np.testing.assert_array_equal(gui_binary.to_numpy(), direct_binary.to_numpy())
    np.testing.assert_allclose(gui_scores.to_numpy(), direct_scores.to_numpy())


def test_transform_choice_never_changes_binary_positive_gates(tmp_path):
    data, thresholds, wide_thresholds = _runtime_input_and_thresholds()
    markers = list(thresholds)
    _write_direct_threshold_table(tmp_path, thresholds)
    gui_gates = []
    direct_gates = []

    for transform in ("identity_residual", "asinh_difference", "log1p_difference"):
        gui_binary, _ = _apply_thresholds_from_table(
            data,
            wide_thresholds,
            markers,
            "batch_id",
            (False, False, False, False),
            [],
            "gate",
            tie_score_transform=transform,
        )
        direct_binary, _ = apply_thresholds_per_batch(
            data,
            "gate",
            tmp_path,
            markers,
            "median",
            None,
            (False, False, False, False),
            "batch_id",
            norm_order=[],
            tie_score_transform=transform,
        )
        gui_gates.append(gui_binary)
        direct_gates.append(direct_binary)

    for gates in gui_gates[1:] + direct_gates:
        pd.testing.assert_frame_equal(gates, gui_gates[0], check_dtype=False)


def test_explicit_arcsinh_cofactor_is_honored_by_both_runtime_paths(tmp_path):
    data, thresholds, wide_thresholds = _runtime_input_and_thresholds()
    markers = list(thresholds)
    _write_direct_threshold_table(tmp_path, thresholds)

    _, gui_scores = _apply_thresholds_from_table(
        data,
        wide_thresholds,
        markers,
        "batch_id",
        (False, False, False, False),
        [],
        "cofactor",
        tie_score_transform="asinh_difference",
        tie_score_cofactor=2.0,
    )
    _, direct_scores = apply_thresholds_per_batch(
        data,
        "cofactor",
        tmp_path,
        markers,
        "median",
        None,
        (False, False, False, False),
        "batch_id",
        norm_order=[],
        tie_score_transform="asinh_difference",
        tie_score_cofactor=2.0,
    )

    raw_values = data[markers].to_numpy(dtype=float)
    threshold_vector = np.array([thresholds[marker] for marker in markers])
    expected = np.clip(
        np.arcsinh(raw_values / 2.0) - np.arcsinh(threshold_vector / 2.0),
        0,
        None,
    )
    np.testing.assert_allclose(gui_scores.to_numpy(), expected)
    np.testing.assert_allclose(direct_scores.to_numpy(), expected)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        ("identity_residual", lambda values: values),
        ("asinh_difference", lambda values: np.arcsinh(values / 5.0)),
        ("log1p_difference", np.log1p),
    ],
)
def test_no_threshold_mode_uses_the_same_selected_transformation(
    tmp_path, transform, expected
):
    data, thresholds, _ = _runtime_input_and_thresholds()
    markers = list(thresholds)
    config = _migrate_resolve_pipeline_config({
        "tie_score_transform": transform,
        "tie_score_cofactor": 5.0,
        "norm_order": [],
    })

    gui_binary, gui_scores = _apply_resolve_without_thresholds(
        data, markers, "batch_id", config, []
    )
    direct_binary, direct_scores = apply_thresholds_per_batch(
        data,
        "raw",
        tmp_path,
        markers,
        "median",
        None,
        (False, False, False, False),
        "batch_id",
        norm_order=[],
        apply_thresholds=False,
        tie_score_transform=transform,
    )

    raw_values = data[markers].to_numpy(dtype=float)
    np.testing.assert_array_equal(gui_binary.to_numpy(), (raw_values > 0).astype(int))
    np.testing.assert_array_equal(direct_binary.to_numpy(), (raw_values > 0).astype(int))
    np.testing.assert_allclose(gui_scores.to_numpy(), expected(raw_values))
    np.testing.assert_allclose(direct_scores.to_numpy(), expected(raw_values))
