"""Adapter between the grouped-validation engine and Tab 9 result bundles."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from spacemap.grouped_validation import (
    GroupedValidationConfig,
    run_grouped_validation,
)

from .f1_validation_utils import (
    GROUP_POOLED,
    analyze_validation,
    prettify_column_name,
)


OUT_OF_FOLD_DESIGN = "Out-of-fold or held-out predictions (generalization)"


def _unique(values: Sequence[str | None]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _strict_method_columns(
    settings: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    methods = list(settings["component_methods"])
    display = {
        column: prettify_column_name(column)
        for column in methods
    }
    confidence: dict[str, str] = {}

    # Score the raw consensus label itself, not just the models trained from it.
    # This is what tests whether "n methods agreeing" is a usable stand-in for a
    # manual reference: its macro-F1 says how right the agreement is, and its
    # coverage says how much of the slide it can speak for. Without this row the
    # consensus premise underlying Consensus-ML is never measured directly.
    if settings.get("score_direct_consensus", True):
        methods.append("consensus_phenotype_recomputed")
        display["consensus_phenotype_recomputed"] = (
            f"Direct consensus (>={int(settings.get('consensus_min_methods', 3))} methods)"
        )

    if settings.get("run_grouped_mr"):
        methods.append("prediction_mr_ml_strict")
        display["prediction_mr_ml_strict"] = "MR-ML"
        confidence["prediction_mr_ml_strict"] = "mr_ml_confidence_strict"
        if settings.get("include_refined"):
            methods.append("prediction_mr_ml_strict_refined")
            display["prediction_mr_ml_strict_refined"] = "MR-ML refined"
            confidence["prediction_mr_ml_strict_refined"] = (
                "mr_ml_confidence_strict"
            )

    if settings.get("run_grouped_consensus"):
        methods.append("prediction_consensus_ml_strict")
        display["prediction_consensus_ml_strict"] = "Consensus-ML"
        confidence["prediction_consensus_ml_strict"] = (
            "consensus_ml_confidence_strict"
        )
        if settings.get("include_refined"):
            methods.append("prediction_consensus_ml_strict_refined")
            display["prediction_consensus_ml_strict_refined"] = (
                "Consensus-ML refined"
            )
            confidence["prediction_consensus_ml_strict_refined"] = (
                "consensus_ml_confidence_strict"
            )
    return methods, display, confidence


def _paired_macro_f1_deltas(
    summary: pd.DataFrame,
    *,
    primary_reference: str,
    component_names: Sequence[str],
) -> pd.DataFrame:
    """Compare each ML workflow with the best component on each held-out group."""

    reference_match = summary["reference"].eq(primary_reference)
    if "reference_column" in summary.columns:
        reference_match |= summary["reference_column"].eq(primary_reference)
    subset = summary.loc[
        reference_match & summary["route"].eq("End-to-end")
    ].copy()
    rows: list[dict[str, Any]] = []
    for group in list(dict.fromkeys(subset["group"].astype(str))):
        group_rows = subset.loc[subset["group"].eq(group)]
        components = group_rows.loc[
            group_rows["method"].isin(component_names)
        ].sort_values("macro_f1", ascending=False)
        if components.empty:
            continue
        best = components.iloc[0]
        row: dict[str, Any] = {
            "group": group,
            "best_component": str(best["method"]),
            "best_component_macro_f1": float(best["macro_f1"]),
        }
        for method in ("MR-ML", "Consensus-ML"):
            target = group_rows.loc[group_rows["method"].eq(method)]
            if target.empty:
                continue
            score = float(target.iloc[0]["macro_f1"])
            key = method.lower().replace("-", "_")
            row[f"{key}_macro_f1"] = score
            row[f"{key}_delta_vs_best_component"] = (
                score - float(best["macro_f1"])
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        pooled_first = pd.Categorical(
            result["group"],
            categories=[
                GROUP_POOLED,
                *[
                    group
                    for group in result["group"]
                    if group != GROUP_POOLED
                ],
            ],
            ordered=True,
        )
        result = (
            result.assign(__order=pooled_first)
            .sort_values("__order")
            .drop(columns="__order")
            .reset_index(drop=True)
        )
    return result


def strict_grouped_config(
    settings: Mapping[str, Any],
    selected_reference_columns: Sequence[str],
    excluded_labels: Sequence[str],
    annotator_columns: tuple[str, str] | None = None,
) -> GroupedValidationConfig:
    """Translate shared Tab 9 controls into the production engine config."""
    primary_reference = str(settings["primary_reference"])
    engine_references = _unique(
        [
            primary_reference,
            *selected_reference_columns,
            *(annotator_columns or ()),
        ]
    )
    return GroupedValidationConfig(
        primary_reference_column=primary_reference,
        evaluation_reference_columns=engine_references,
        component_method_columns=list(settings["component_methods"]),
        marker_columns=list(settings["markers"]),
        validation_group_column=str(settings["validation_group_column"]),
        group_mode=str(settings["group_mode"]),
        group_mapping=dict(settings.get("group_mapping", {})),
        development_reference_column=settings.get("development_reference"),
        internal_test_reference_column=settings.get("internal_test_reference"),
        morphology_columns=list(settings.get("morphology_columns", [])),
        compartment_columns=list(settings.get("compartment_columns", [])),
        feature_weights=dict(settings["feature_weights"]),
        use_z_normalization=bool(settings["use_z_normalization"]),
        batch_normalization_column=str(settings["batch_normalization_column"]),
        drop_invalid_marker_rows=bool(settings["drop_invalid_marker_rows"]),
        mr_ml_method=str(settings["ml_method"]),
        consensus_ml_method=str(settings["ml_method"]),
        mr_n_estimators=int(settings["mr_n_estimators"]),
        consensus_n_estimators=int(settings["consensus_n_estimators"]),
        quick_mode=bool(settings["quick_mode"]),
        quick_n_estimators=int(settings["quick_n_estimators"]),
        random_state=int(settings["random_state"]),
        consensus_min_methods=int(settings["consensus_min_methods"]),
        consensus_min_training_cells=int(settings["minimum_consensus_training"]),
        refined_confidence_threshold=float(settings["confidence_threshold"]),
        refined_low_confidence_label=str(settings["low_confidence_label"]),
        excluded_phenotypes=list(excluded_labels),
        run_grouped_mr_ml=bool(settings["run_grouped_mr"]),
        run_grouped_consensus_ml=bool(settings["run_grouped_consensus"]),
        run_internal_split=bool(settings["run_internal"]),
        filter_column=settings.get("filter_column"),
        filter_value=settings.get("filter_value"),
    )


def build_strict_validation_bundle(
    frame: pd.DataFrame,
    *,
    settings: Mapping[str, Any],
    selected_reference_columns: Sequence[str],
    excluded_labels: Sequence[str],
    method_display_names: Mapping[str, str] | None,
    remove_primary_overlap: bool,
    confidence_thresholds: Sequence[float],
    annotator_columns: tuple[str, str] | None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run strict folds and create a normal Tab 9 validation bundle."""

    primary_reference = str(settings["primary_reference"])
    config = strict_grouped_config(
        settings,
        selected_reference_columns,
        excluded_labels,
        annotator_columns,
    )
    result = run_grouped_validation(
        frame,
        config,
        progress_callback=progress_callback,
    )
    if result.oof_predictions.empty:
        raise ValueError(
            "Strict retraining did not produce grouped out-of-fold predictions."
        )

    method_columns, default_names, confidence_mapping = (
        _strict_method_columns(settings)
    )
    display_names = dict(default_names)
    display_names.update(
        {
            str(column): str(name)
            for column, name in (method_display_names or {}).items()
            if str(column) in method_columns and str(name).strip()
        }
    )
    references_to_score = _unique(
        [
            primary_reference,
            *selected_reference_columns,
        ]
    )

    bundle = analyze_validation(
        result.oof_predictions,
        reference_columns=references_to_score,
        method_columns=method_columns,
        excluded_labels=list(excluded_labels),
        method_display_names=display_names,
        route_column="consensus_route",
        group_column="validation_group",
        group_mode="As stored",
        group_mapping=None,
        remove_primary_overlap=bool(remove_primary_overlap),
        confidence_mapping=confidence_mapping,
        confidence_thresholds=list(confidence_thresholds),
        annotator_columns=annotator_columns,
        prediction_design=OUT_OF_FOLD_DESIGN,
    )

    component_names = [
        display_names.get(column, prettify_column_name(column))
        for column in settings["component_methods"]
    ]
    paired = _paired_macro_f1_deltas(
        bundle["summary"],
        primary_reference=primary_reference,
        component_names=component_names,
    )

    strict_tables: dict[str, Any] = {
        "oof_predictions": result.oof_predictions,
        "fold_audit": result.fold_audit,
        "f1_weights": result.f1_weights,
        "consensus_training_counts": result.consensus_training_counts,
        "feature_audit": result.feature_audit,
        "configuration": result.configuration,
        "paired_macro_f1_deltas": paired,
        "internal_predictions": result.internal_predictions,
        "internal_summary": pd.DataFrame(),
        "internal_per_class": pd.DataFrame(),
    }

    if not result.internal_predictions.empty:
        internal_methods = list(settings["component_methods"]) + [
            "prediction_mr_ml_internal"
        ]
        internal_names = {
            column: display_names.get(
                column,
                prettify_column_name(column),
            )
            for column in settings["component_methods"]
        }
        internal_names["prediction_mr_ml_internal"] = "MR-ML"
        if settings.get("include_refined"):
            internal_methods.append("prediction_mr_ml_internal_refined")
            internal_names["prediction_mr_ml_internal_refined"] = (
                "MR-ML refined"
            )

        internal_bundle = analyze_validation(
            result.internal_predictions,
            reference_columns=[
                str(settings["internal_test_reference"])
            ],
            method_columns=internal_methods,
            excluded_labels=list(excluded_labels),
            method_display_names=internal_names,
            confidence_mapping={
                "prediction_mr_ml_internal": "mr_ml_confidence_internal",
                "prediction_mr_ml_internal_refined": (
                    "mr_ml_confidence_internal"
                ),
            },
            confidence_thresholds=list(confidence_thresholds),
            prediction_design=(
                "Row-disjoint development / internal-test predictions"
            ),
        )
        strict_tables["internal_summary"] = internal_bundle["summary"]
        strict_tables["internal_per_class"] = internal_bundle["per_class"]

    bundle["strict_validation"] = strict_tables
    bundle["metadata"].update(
        {
            "prediction_design": OUT_OF_FOLD_DESIGN,
            "strict_grouped_retraining": True,
            "strict_group_separation_verified": True,
            "strict_primary_reference": primary_reference,
            "strict_component_methods": list(settings["component_methods"]),
            "strict_component_display_names": component_names,
            "strict_validation_groups": result.feature_audit.get(
                "validation_groups",
                [],
            ),
            "strict_quick_mode": bool(settings["quick_mode"]),
            "strict_mr_n_estimators_used": result.feature_audit.get(
                "mr_n_estimators_used"
            ),
            "strict_consensus_n_estimators_used": (
                result.feature_audit.get("consensus_n_estimators_used")
            ),
        }
    )
    return bundle


def build_strict_validation_bundle_from_cached_analysis(
    analysis: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    selected_reference_columns: Sequence[str],
    excluded_labels: Sequence[str],
    method_display_names: Mapping[str, str] | None,
    remove_primary_overlap: bool,
    confidence_thresholds: Sequence[float],
    annotator_columns: tuple[str, str] | None,
) -> dict[str, Any]:
    """Build Option 1 metrics from saved strict OOF tables without refitting."""
    oof_predictions = analysis.get("oof_predictions")
    if not isinstance(oof_predictions, pd.DataFrame) or oof_predictions.empty:
        raise ValueError("The shared ML cache contains no OOF predictions.")

    primary_reference = str(settings["primary_reference"])
    method_columns, default_names, confidence_mapping = _strict_method_columns(
        settings
    )
    method_columns = [
        column for column in method_columns if column in oof_predictions.columns
    ]
    confidence_mapping = {
        method: column
        for method, column in confidence_mapping.items()
        if method in method_columns and column in oof_predictions.columns
    }
    display_names = dict(default_names)
    display_names.update(
        {
            str(column): str(name)
            for column, name in (method_display_names or {}).items()
            if str(column) in method_columns and str(name).strip()
        }
    )
    references_to_score = [
        column
        for column in _unique([primary_reference, *selected_reference_columns])
        if column in oof_predictions.columns
    ]
    if not references_to_score:
        raise ValueError(
            "None of the selected manual-reference columns are present in the "
            "shared OOF prediction table."
        )

    bundle = analyze_validation(
        oof_predictions,
        reference_columns=references_to_score,
        method_columns=method_columns,
        excluded_labels=list(excluded_labels),
        method_display_names=display_names,
        route_column="consensus_route",
        group_column="validation_group",
        group_mode="As stored",
        group_mapping=None,
        remove_primary_overlap=bool(remove_primary_overlap),
        confidence_mapping=confidence_mapping,
        confidence_thresholds=list(confidence_thresholds),
        annotator_columns=annotator_columns,
        prediction_design=OUT_OF_FOLD_DESIGN,
    )
    component_names = [
        display_names.get(column, prettify_column_name(column))
        for column in settings["component_methods"]
    ]
    paired = _paired_macro_f1_deltas(
        bundle["summary"],
        primary_reference=primary_reference,
        component_names=component_names,
    )
    groups = list(
        dict.fromkeys(
            oof_predictions.get(
                "validation_group", pd.Series(dtype="string")
            )
            .dropna()
            .astype(str)
        )
    )
    bundle["strict_validation"] = {
        "oof_predictions": oof_predictions,
        "fold_audit": analysis.get("fold_audit", pd.DataFrame()),
        "f1_weights": analysis.get("f1_weights", pd.DataFrame()),
        "consensus_training_counts": analysis.get(
            "consensus_training_counts", pd.DataFrame()
        ),
        "learning_curve": analysis.get("learning_curve", pd.DataFrame()),
        "feature_audit": {},
        "configuration": dict(settings),
        "paired_macro_f1_deltas": paired,
        "internal_predictions": pd.DataFrame(),
        "internal_summary": pd.DataFrame(),
        "internal_per_class": pd.DataFrame(),
    }
    bundle["metadata"].update(
        {
            "prediction_design": OUT_OF_FOLD_DESIGN,
            "strict_grouped_retraining": True,
            "strict_group_separation_verified": True,
            "strict_primary_reference": primary_reference,
            "strict_component_methods": list(settings["component_methods"]),
            "strict_component_display_names": component_names,
            "strict_validation_groups": groups,
            "strict_quick_mode": bool(settings["quick_mode"]),
            "shared_ml_cache_reused": True,
            "shared_ml_cache_fingerprint": analysis.get("fingerprint"),
        }
    )
    return bundle
