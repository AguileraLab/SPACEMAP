from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from spacemap_gui.tabs.tab9_visualization.f1_paper_figures import (
    PANEL_BY_ID,
    PAPER_PANEL_SPECS,
    build_generic_panel_tables,
    build_generic_paper_figure,
    build_paper_figure,
    default_common_settings,
    default_panel_settings,
    generic_panel_availability,
    missing_panel_tables,
    save_paper_figure,
    _method_default,
    _phenotype_default_color,
    _reference_default,
)
from spacemap_gui.tabs.tab9_visualization.f1_paper_analysis import (
    cached_publication_analysis,
    publication_analysis_fingerprint,
    run_publication_analysis,
    save_publication_analysis,
)
from spacemap_gui.tabs.tab9_visualization.f1_strict_validation_ui import (
    _phenotyping_marker_candidates,
)
from spacemap_gui.tabs.tab9_visualization.f1_strict_runner import (
    build_strict_validation_bundle_from_cached_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RESULTS = PROJECT_ROOT / "test_codes" / "Fig3_code_results" / "results_v2"


def test_option2_defaults_recover_reference_code_visual_preset() -> None:
    common = default_common_settings()
    expected_dimensions = {
        "main_workflow": (14.2, 4.8),
        "main_heldout_macro": (10.0, 6.0),
        "main_spatial": (12.2, 4.7),
        "supp_ref1_f1": (7.8, 9.2),
        "supp_leave_one_out": (12.2, 4.8),
        "supp_confusion": (12.0, 5.2),
        "supp_learning": (8.1, 5.1),
        "supp_bins_ref2": (10.4, 5.3),
        "supp_pairwise": (6.8, 5.8),
    }

    assert common["background_color"] == "#FFFFFF"
    assert common["subplot_title_font_size"] == 11.0
    for panel_id, dimensions in expected_dimensions.items():
        settings = default_panel_settings(PANEL_BY_ID[panel_id])
        assert (settings["width"], settings["height"]) == dimensions


def test_option2_png_svg_exports_have_an_opaque_white_canvas(tmp_path: Path) -> None:
    figure = build_paper_figure("main_workflow", tmp_path)
    try:
        assert to_hex(figure.get_facecolor()) == "#ffffff"
        assert all(to_hex(axis.get_facecolor()) == "#ffffff" for axis in figure.axes)
        paths, preview = save_paper_figure(
            figure,
            tmp_path,
            "white canvas regression",
            ["PNG", "SVG"],
            dpi=100,
        )
    finally:
        plt.close(figure)

    image = Image.open(BytesIO(preview)).convert("RGBA")
    red, green, blue, alpha = image.getpixel((0, 0))
    assert min(red, green, blue) >= 250
    assert alpha == 255
    svg_path = next(path for path in paths if path.suffix == ".svg")
    assert "#ffffff" in svg_path.read_text(encoding="utf-8").lower()


def test_custom_dark_canvas_automatically_uses_readable_foreground() -> None:
    figure = build_paper_figure(
        "main_workflow",
        REFERENCE_RESULTS,
        common={"background_color": "#000000"},
    )
    try:
        figure.canvas.draw()
        assert to_hex(figure.get_facecolor()) == "#000000"
        assert all(to_hex(axis.get_facecolor()) == "#000000" for axis in figure.axes)
        assert all(to_hex(text.get_color()) != "#000000" for text in figure.axes[0].texts)
    finally:
        plt.close(figure)


def test_reference_and_fallback_colors_are_reproducible() -> None:
    assert _method_default("phenotype_leiden", 0) == ("Leiden", "#1f77b4")
    assert _method_default("new_vortex_labels", 9) == ("Vortex", "#9467bd")
    assert _method_default("unknown_algorithm", 0) == _method_default(
        "unknown_algorithm", 8
    )
    assert _reference_default("phenotype_ref1", 0) == (
        "Ref1: expert-selected cells",
        "#1f77b4",
    )
    assert _reference_default("custom_manual_truth", 0) == _reference_default(
        "custom_manual_truth", 4
    )
    assert _phenotype_default_color("CD4") == _phenotype_default_color("cd4")
    assert _phenotype_default_color("Novel phenotype") == _phenotype_default_color(
        "Novel phenotype"
    )


def test_training_markers_use_tab7_common_list_and_training_safety() -> None:
    sample = pd.DataFrame(
        {
            "cell_id": [1, 2],
            "x_centroid": [10.0, 20.0],
            "phenotype_leiden": ["A", "B"],
            "CD3": [0.2, 0.4],
            "CD3_corr": [0.1, 0.3],
            "CD8_norm": [1.0, 2.0],
            "area": [40.0, 45.0],
            "free_text": ["x", "y"],
        }
    )
    markers, morphology, compartments = _phenotyping_marker_candidates(
        list(sample.columns),
        sample,
    )
    assert markers == ["CD3", "CD3_corr"]
    assert morphology == ["area"]
    assert compartments == []


@pytest.fixture
def generic_frame_and_settings() -> tuple[pd.DataFrame, dict[str, object]]:
    truth = np.resize(np.array(["Alpha", "Beta", "Gamma"], dtype=object), 72)
    frame = pd.DataFrame(
        {
            "curator_gold": truth,
            "algorithm_red": truth.copy(),
            "algorithm_blue": np.roll(truth, 1),
            "algorithm_green": np.where(np.arange(72) % 5, truth, "Beta"),
            "red_probability": np.linspace(0.2, 0.99, 72),
            "sample_block": np.resize(["Cohort A", "Cohort B", "Cohort C"], 72),
            "horizontal_location": np.arange(72) % 12,
            "vertical_location": np.arange(72) // 12,
        }
    )
    frame.loc[[4, 19], "curator_gold"] = np.nan
    settings: dict[str, object] = {
        "method_order": ["algorithm_red", "algorithm_blue", "algorithm_green"],
        "method_labels": {
            "algorithm_red": "Red model",
            "algorithm_blue": "Blue model",
            "algorithm_green": "Green model",
        },
        "method_colors": {
            "algorithm_red": "#A00000",
            "algorithm_blue": "#0040A0",
            "algorithm_green": "#008040",
        },
        "reference_order": ["curator_gold"],
        "reference_labels": {"curator_gold": "Curator truth"},
        "reference_colors": {"curator_gold": "#7A4EAB"},
        "component_methods": [
            "algorithm_red",
            "algorithm_blue",
            "algorithm_green",
        ],
        "confidence_columns": {"algorithm_red": "red_probability"},
        "confidence_thresholds": [0.0, 0.5, 0.75, 0.9],
        "excluded_labels": {"", "nan", "none", "unknown", "unclassified", "others"},
        "consensus_min_agree": 2,
        "primary_method": "algorithm_red",
        "secondary_method": "algorithm_green",
        "primary_reference": "curator_gold",
        "secondary_reference": "curator_gold",
        "group_column": "sample_block",
        "group_order": ["Cohort A", "Cohort B", "Cohort C"],
        "group_labels": {
            "Cohort A": "Training site A",
            "Cohort B": "Training site B",
            "Cohort C": "Training site C",
        },
        "group_colors": {
            "Cohort A": "#D8A328",
            "Cohort B": "#EEDD44",
            "Cohort C": "#51A83E",
        },
        "phenotype_colors": {
            "Alpha": "#AA3377",
            "Beta": "#33AA77",
            "Gamma": "#3377AA",
        },
        "x_column": "horizontal_location",
        "y_column": "vertical_location",
    }
    return frame, settings


def test_option2_exposes_complete_main_and_supplementary_panel_set() -> None:
    panel_ids = {spec.panel_id for spec in PAPER_PANEL_SPECS}

    assert len(panel_ids) == 27
    assert {
        "main_workflow",
        "main_heldout_macro",
        "main_consensus_tradeoff",
        "main_ref2_metrics",
        "main_per_phenotype_f1",
        "main_spatial",
        "supp_ref1_precision",
        "supp_ref2_recall",
        "supp_radar",
        "supp_pairwise",
    }.issubset(panel_ids)


def test_panel_source_can_point_to_results_root_or_source_data() -> None:
    source_data = REFERENCE_RESULTS / "source_data"

    assert missing_panel_tables("main_ref2_metrics", REFERENCE_RESULTS) == []
    assert missing_panel_tables("main_ref2_metrics", source_data) == []


@pytest.mark.parametrize(
    "panel_id",
    [
        "main_workflow",
        "main_ref2_metrics",
        "main_per_phenotype_f1",
        "main_spatial",
        "supp_human",
        "supp_radar",
        "supp_pairwise",
    ],
)
def test_representative_paper_panels_render_from_reference_tables(
    panel_id: str,
) -> None:
    figure = build_paper_figure(panel_id, REFERENCE_RESULTS)
    try:
        figure.canvas.draw()
        assert figure.axes
    finally:
        plt.close(figure)


def test_pairwise_panel_uses_percent_scale_by_default() -> None:
    settings = default_panel_settings(PANEL_BY_ID["supp_pairwise"])

    assert settings["heatmap_vmin"] == 50.0
    assert settings["heatmap_vmax"] == 100.0


def test_custom_method_and_colorbar_labels_reach_heatmap() -> None:
    common = {"method_labels": {"Leiden": "Custom Leiden"}}
    panel = {"colorbar_label": "Custom F1 scale"}
    figure = build_paper_figure(
        "supp_ref1_f1",
        REFERENCE_RESULTS,
        common=common,
        panel=panel,
    )
    try:
        figure.canvas.draw()
        column_labels = [item.get_text() for item in figure.axes[0].get_xticklabels()]
        colorbar_labels = [axis.get_ylabel() for axis in figure.axes[1:]]
    finally:
        plt.close(figure)

    assert "Custom Leiden" in column_labels
    assert "Custom F1 scale" in colorbar_labels


def test_saved_panels_have_unique_option2_png_svg_and_settings_names(
    tmp_path: Path,
) -> None:
    figure = build_paper_figure("main_workflow", tmp_path)
    try:
        first_paths, first_preview = save_paper_figure(
            figure,
            tmp_path,
            "Fig 3A custom",
            ["PNG", "SVG"],
            dpi=100,
            settings={"title": "Custom"},
        )
        second_paths, _ = save_paper_figure(
            figure,
            tmp_path,
            "Fig 3A custom",
            ["PNG", "SVG"],
            dpi=100,
            settings={"title": "Custom"},
        )
    finally:
        plt.close(figure)

    assert first_preview.startswith(b"\x89PNG")
    assert {path.suffix for path in first_paths} == {".png", ".svg", ".json"}
    assert all(path.is_file() for path in first_paths)
    assert all(
        path.name.startswith("f1__option2__Fig-3A-custom__") for path in first_paths
    )
    assert {path.name for path in first_paths}.isdisjoint(
        path.name for path in second_paths
    )


def test_generic_tables_use_arbitrary_unified_csv_columns(
    generic_frame_and_settings: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    frame, common = generic_frame_and_settings

    tables = build_generic_panel_tables(frame, common, "main_ref2_metrics")
    summary = tables["T10_performance_summary.csv"]

    assert set(summary["reference"]) == {"curator_gold"}
    assert set(summary["method"]) == {
        "algorithm_red",
        "algorithm_blue",
        "algorithm_green",
    }
    assert set(summary["reference_name"]) == {"Curator truth"}
    assert set(summary["fold"]) == {"Pooled"}


@pytest.mark.parametrize(
    "panel_id",
    [
        "main_workflow",
        "main_heldout_macro",
        "main_consensus_tradeoff",
        "main_ref2_metrics",
        "main_per_phenotype_f1",
        "main_spatial",
        "supp_reference_distribution",
        "supp_weights",
        "supp_ref1_f1",
        "supp_leave_one_out",
        "supp_routes",
        "supp_confidence_all",
        "supp_nonconsensus_threshold",
        "supp_confusion",
        "supp_radar",
        "supp_heldout_variation",
        "supp_bins_ref2",
        "supp_pairwise",
    ],
)
def test_generic_figures_render_with_one_renamed_reference(
    panel_id: str,
    generic_frame_and_settings: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    frame, common = generic_frame_and_settings
    panel = default_panel_settings(PANEL_BY_ID[panel_id])
    panel.update(
        {
            "focus_reference": "curator_gold",
            "spatial_reference": "curator_gold",
            "spatial_methods": ["algorithm_red", "algorithm_green"],
            "confusion_methods": ["algorithm_red", "algorithm_blue"],
            "confidence_method": "algorithm_red",
            "width": 6.0,
            "height": 4.0,
        }
    )
    figure = build_generic_paper_figure(panel_id, frame, common, panel)
    try:
        figure.canvas.draw()
        assert figure.axes
    finally:
        plt.close(figure)


def test_generic_ui_reports_only_genuine_panel_prerequisites(
    generic_frame_and_settings: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    _, common = generic_frame_and_settings

    available, message = generic_panel_availability("supp_human", common)
    confidence_available, _ = generic_panel_availability("supp_confidence_all", common)

    assert not available
    assert "two manual-reference" in message
    assert confidence_available


def test_generic_human_panel_accepts_arbitrarily_named_references(
    generic_frame_and_settings: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    frame, common = generic_frame_and_settings
    frame = frame.copy()
    frame["second_pathologist"] = frame["curator_gold"]
    frame.loc[frame.index[::7], "second_pathologist"] = "Gamma"
    common = {
        **common,
        "reference_order": ["curator_gold", "second_pathologist"],
        "reference_labels": {
            "curator_gold": "Curator truth",
            "second_pathologist": "Pathologist review",
        },
    }
    panel = default_panel_settings(PANEL_BY_ID["supp_human"])

    figure = build_generic_paper_figure("supp_human", frame, common, panel)
    try:
        figure.canvas.draw()
        assert figure.axes
    finally:
        plt.close(figure)


def test_generic_learning_panel_uses_optional_user_table(
    tmp_path: Path,
    generic_frame_and_settings: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    frame, common = generic_frame_and_settings
    learning_path = tmp_path / "custom_learning.csv"
    pd.DataFrame(
        {
            "target_cells_per_class": [8, 16, 32, 8, 16, 32],
            "fold": ["Pooled"] * 6,
            "macro_f1": [0.55, 0.63, 0.71, 0.58, 0.66, 0.74],
        }
    ).to_csv(learning_path, index=False)
    common = {**common, "learning_curve_csv": str(learning_path)}
    panel = default_panel_settings(PANEL_BY_ID["supp_learning"])
    panel.update(
        {
            "focus_reference": "curator_gold",
            "learning_method": "algorithm_red",
        }
    )

    figure = build_generic_paper_figure("supp_learning", frame, common, panel)
    try:
        figure.canvas.draw()
        assert figure.axes
    finally:
        plt.close(figure)


def test_generic_learning_panel_uses_shared_cached_experiment(
    generic_frame_and_settings: tuple[pd.DataFrame, dict[str, object]],
) -> None:
    frame, common = generic_frame_and_settings
    common = {
        **common,
        "learning_curve_csv": "",
        "cached_learning_curve_available": True,
    }
    learning = pd.DataFrame(
        {
            "target_cells_per_class": [4, 8, 4, 8],
            "fold": ["Pooled"] * 4,
            "macro_f1": [0.51, 0.62, 0.54, 0.66],
        }
    )
    panel = default_panel_settings(PANEL_BY_ID["supp_learning"])
    panel.update(
        {
            "focus_reference": "curator_gold",
            "learning_method": "algorithm_red",
        }
    )

    figure = build_generic_paper_figure(
        "supp_learning",
        frame,
        common,
        panel,
        analysis_tables={"learning_curve": learning},
    )
    try:
        figure.canvas.draw()
        assert figure.axes
    finally:
        plt.close(figure)


def test_publication_analysis_retrains_and_reuses_out_of_fold_predictions(
    tmp_path: Path,
) -> None:
    cells = 90
    truth = np.resize(np.array(["Alpha", "Beta", "Gamma"], dtype=object), cells)
    groups = np.repeat(["Image A", "Image B", "Image C"], cells // 3)
    frame = pd.DataFrame(
        {
            "cell_id": [f"cell-{index}" for index in range(cells)],
            "image_name": groups,
            "manual_truth": truth,
            "component_one": truth,
            "component_two": np.where(np.arange(cells) % 7, truth, np.roll(truth, 1)),
            "component_three": np.where(
                np.arange(cells) % 11, truth, np.roll(truth, 2)
            ),
            "marker_a": np.arange(cells) % 3,
            "marker_b": (np.arange(cells) % 3) + np.linspace(0, 0.1, cells),
            # Deliberately wrong saved output: strict publication mode must replace it.
            "prediction_mr_ml_strict": "Wrong saved call",
        }
    )
    input_csv = tmp_path / "arbitrary_dataset.csv"
    frame.to_csv(input_csv, index=False)
    settings = {
        "enabled": True,
        "primary_reference": "manual_truth",
        "component_methods": [
            "component_one",
            "component_two",
            "component_three",
        ],
        "validation_group_column": "image_name",
        "group_mode": "As stored",
        "group_mapping": {},
        "run_grouped_mr": True,
        "run_grouped_consensus": True,
        "run_internal": False,
        "development_reference": None,
        "internal_test_reference": None,
        "markers": ["marker_a", "marker_b"],
        "morphology_columns": [],
        "compartment_columns": [],
        "feature_weights": {
            "markers": 1.0,
            "morphology": 1.0,
            "compartments": 0.5,
        },
        "use_z_normalization": False,
        "batch_normalization_column": "image_name",
        "drop_invalid_marker_rows": True,
        "filter_column": None,
        "filter_value": None,
        "ml_method": "random_forest",
        "mr_n_estimators": 10,
        "consensus_n_estimators": 10,
        "quick_mode": True,
        "quick_n_estimators": 5,
        "random_state": 42,
        "consensus_min_methods": 2,
        "minimum_consensus_training": 2,
        "confidence_threshold": 0.75,
        "low_confidence_label": "Others",
        "include_refined": False,
        "run_learning_curve": True,
        "learning_targets": [1, 2],
        "learning_repeats": 1,
        "learning_n_estimators": 5,
    }
    thresholds = [0.0, 0.5, 0.75]
    fingerprint, payload = publication_analysis_fingerprint(
        input_csv,
        settings,
        ["manual_truth"],
        ["Others"],
        thresholds,
    )

    analysis = run_publication_analysis(
        frame,
        settings=settings,
        reference_columns=["manual_truth"],
        excluded_labels=["Others"],
        method_display_names={
            "component_one": "One",
            "component_two": "Two",
            "component_three": "Three",
        },
        confidence_thresholds=thresholds,
    )
    cached = save_publication_analysis(
        analysis,
        tmp_path,
        fingerprint,
        payload,
    )

    oof = cached["oof_predictions"]
    assert cached_publication_analysis(tmp_path, fingerprint)
    assert set(oof["fold"]) == {"Image A", "Image B", "Image C"}
    assert not oof["prediction_mr_ml_strict"].eq("Wrong saved call").any()
    assert not cached["fold_audit"].empty
    assert set(cached["learning_curve"]["target_cells_per_class"]) == {1, 2}
    assert set(cached["learning_curve"]["fold"]) == {
        "Image A",
        "Image B",
        "Image C",
        "Pooled",
    }
    assert cached["manifest"]["rows"]["oof_predictions"] == cells
    assert cached["manifest"]["rows"]["learning_curve"] == 8

    option1_bundle = build_strict_validation_bundle_from_cached_analysis(
        cached,
        settings=settings,
        selected_reference_columns=["manual_truth"],
        excluded_labels=["Others"],
        method_display_names={
            "component_one": "One",
            "component_two": "Two",
            "component_three": "Three",
        },
        remove_primary_overlap=False,
        confidence_thresholds=thresholds,
        annotator_columns=None,
    )
    assert option1_bundle["metadata"]["shared_ml_cache_reused"] is True
    assert not option1_bundle["summary"].empty
    assert (
        option1_bundle["strict_validation"]["oof_predictions"]
        .sort_values("source_csv_row")
        .reset_index(drop=True)
        .equals(oof.sort_values("source_csv_row").reset_index(drop=True))
    )
