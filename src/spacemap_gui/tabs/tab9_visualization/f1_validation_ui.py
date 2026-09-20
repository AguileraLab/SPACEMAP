"""Streamlit interface for comprehensive phenotyping validation."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like, to_hex
import pandas as pd
import streamlit as st

from .common import (
    get_cached_unified_df,
    log_console,
    logger,
    save_cached_params,
    timestamp,
)
from .f1_strict_runner import build_strict_validation_bundle_from_cached_analysis
from .f1_paper_figures import (
    method_style_default,
    render_paper_figure_panels,
    render_shared_ml_prediction_zone,
)
from .f1_validation_utils import (
    GROUP_POOLED,
    PALETTES,
    ValidationStyle,
    analyze_validation,
    available_result_values,
    confidence_candidates,
    default_confidence_mapping,
    default_method_columns,
    figure_to_png_bytes,
    group_candidates,
    interpretation_lines,
    mapping_to_text,
    parse_mapping_text,
    plot_confidence_coverage,
    plot_f1_metric_comparison,
    plot_grouped_macro_f1,
    plot_human_agreement,
    plot_normalized_confusion,
    plot_per_class_heatmap,
    plot_precision_recall_f1_heatmaps,
    plot_radar_summary,
    plot_reference_route_comparison,
    prediction_candidates,
    prettify_column_name,
    reference_candidates,
    route_candidates,
    sanitize_filename,
    style_to_dict,
)


CORE_FIGURES = [
    "Per-class F1 heatmap",
    "Micro / macro / weighted-F1 comparison",
    "References × assignment routes",
    "Grouped macro-F1",
]
COMPLEMENTARY_FIGURES = [
    "Precision / recall / F1 heatmaps",
    "Normalized confusion matrix",
    "Confidence versus coverage",
    "Inter-annotator agreement",
    "Radar performance summary",
]
ALL_FIGURES = CORE_FIGURES + COMPLEMENTARY_FIGURES
DEFAULT_FIGURES = [
    "Per-class F1 heatmap",
    "Micro / macro / weighted-F1 comparison",
    "References × assignment routes",
    "Grouped macro-F1",
]

PREDICTION_DESIGNS = [
    "Saved final predictions (descriptive/internal)",
    "Out-of-fold or held-out predictions (generalization)",
]
GROUP_MODES = [
    "As stored",
    "Remove ROI suffix",
    "Manual mapping",
]
HEATMAP_CMAPS = [
    "Blues",
    "Reds",
    "Greens",
    "viridis",
    "mako",
    "rocket",
    "crest",
    "vlag",
]
FONT_FAMILIES = [
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "DejaVu Serif",
]


def _cached(cached_params: Mapping[str, Any], key: str, default: Any) -> Any:
    if key in st.session_state:
        return st.session_state[key]
    return cached_params.get(key, default)


def _safe_choice(options: list[str], value: Any, fallback: str | None = None) -> str:
    if value in options:
        return str(value)
    if fallback in options:
        return str(fallback)
    return options[0] if options else ""


def _safe_defaults(options: list[str], values: Any, fallback: list[str]) -> list[str]:
    if not isinstance(values, list):
        values = fallback
    selected = [str(value) for value in values if str(value) in options]
    if selected:
        return selected
    return [value for value in fallback if value in options]


def _parse_label_list(value: str) -> list[str]:
    parts = str(value or "").replace("\n", ",").split(",")
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def _parse_thresholds(value: str) -> list[float]:
    thresholds: list[float] = []
    for part in str(value or "").replace("\n", ",").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            number = float(text)
        except ValueError:
            continue
        if 0 <= number <= 1:
            thresholds.append(number)
    thresholds = sorted(set(thresholds))
    return thresholds or [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]


def _default_display_mapping(methods: list[str]) -> dict[str, str]:
    return {method: prettify_column_name(method) for method in methods}


def _method_color_key(method: str) -> str:
    digest = hashlib.sha1(str(method).encode("utf-8")).hexdigest()[:10]
    return f"viz_f1_method_color_{digest}"


def _method_color_targets(
    methods: list[str],
    display_mapping: Mapping[str, str],
    *,
    strict_settings: Mapping[str, Any] | None = None,
) -> list[tuple[str, str]]:
    targets = list(dict.fromkeys(str(method) for method in methods))
    strict = dict(strict_settings or {})
    if strict.get("run_grouped_mr"):
        targets.append("prediction_mr_ml_strict")
    if strict.get("run_grouped_consensus"):
        targets.append("prediction_consensus_ml_strict")
    targets = list(dict.fromkeys(targets))
    return [
        (
            method,
            str(
                display_mapping.get(
                    method,
                    prettify_column_name(method),
                )
            ),
        )
        for method in targets
    ]


def _method_colors_from_widgets() -> dict[str, str]:
    registry = st.session_state.get("viz_f1_method_color_registry", [])
    if not isinstance(registry, list):
        return {}
    colors: dict[str, str] = {}
    for item in registry:
        if not isinstance(item, Mapping):
            continue
        method = str(item.get("method", "")).strip()
        label = str(item.get("label", "")).strip()
        if not method:
            continue
        default_label, default_color = method_style_default(method)
        color = str(st.session_state.get(_method_color_key(method), default_color))
        aliases = {
            method,
            label,
            default_label,
            prettify_column_name(method),
        }
        if default_label == "ML-MR":
            aliases.add("MR-ML")
        for alias in aliases:
            if alias:
                colors[alias] = color
    return colors


def _safe_hex_color(value: Any, fallback: str) -> str:
    candidate = str(value).strip()
    return str(to_hex(candidate)) if is_color_like(candidate) else str(fallback)


def _default_reference_columns(columns: list[str]) -> list[str]:
    references = reference_candidates(columns)
    if "phenotype_ref1" in references:
        return ["phenotype_ref1"]
    return references[:1]


def _default_prediction_design(columns: list[str]) -> str:
    if any(
        str(column).startswith("prediction_") and "_strict" in str(column)
        for column in columns
    ):
        return PREDICTION_DESIGNS[1]
    return PREDICTION_DESIGNS[0]


def _style_from_widgets() -> ValidationStyle:
    color_range = st.session_state.get("viz_f1_color_range", (0.5, 1.0))
    if not isinstance(color_range, (list, tuple)) or len(color_range) != 2:
        color_range = (0.5, 1.0)
    return ValidationStyle(
        font_family=str(st.session_state.get("viz_f1_font_family", "DejaVu Sans")),
        base_font_size=float(st.session_state.get("viz_f1_base_fontsize", 10)),
        title_font_size=float(st.session_state.get("viz_f1_title_fontsize", 14)),
        axis_font_size=float(st.session_state.get("viz_f1_axis_fontsize", 11)),
        tick_font_size=float(st.session_state.get("viz_f1_tick_fontsize", 9)),
        annotation_font_size=float(st.session_state.get("viz_f1_annot_fontsize", 8)),
        heatmap_cmap=str(st.session_state.get("viz_f1_heatmap_cmap", "Blues")),
        heatmap_vmin=float(color_range[0]),
        heatmap_vmax=float(color_range[1]),
        show_annotations=bool(st.session_state.get("viz_f1_show_annot", True)),
        square_cells=bool(st.session_state.get("viz_f1_square_cells", False)),
        figure_width=float(st.session_state.get("viz_f1_figure_width", 10.0)),
        figure_height=float(st.session_state.get("viz_f1_figure_height", 6.0)),
        row_height=float(st.session_state.get("viz_f1_row_height", 0.45)),
        palette_name=str(st.session_state.get("viz_f1_palette", "GUI classic")),
        method_colors=_method_colors_from_widgets(),
        dpi=int(st.session_state.get("viz_f1_dpi", 300)),
        title_prefix=str(st.session_state.get("viz_f1_dataset_title", "") or ""),
    )


def _persist_widget_configuration(
    project_root: Path,
    cached_params: dict[str, Any],
) -> None:
    for key, value in st.session_state.items():
        if isinstance(key, str) and key.startswith("viz_f1_"):
            if isinstance(value, pd.DataFrame):
                continue
            cached_params[key] = value
    save_cached_params(project_root, cached_params)
    try:
        from spacemap_gui.persistence import save_active_project_state

        save_active_project_state("tab9_visualization")
    except Exception:
        pass


def _render_concept_guide() -> None:
    with st.expander("What each validation option means", expanded=False):
        st.markdown(
            """
- **Saved predictions**: compares columns already present in the CSV. Group-specific
  scores show where performance changes, but do not prove that the model was trained
  without those groups.
- **Out-of-fold predictions**: every evaluated cell was predicted by a model that
  excluded its complete patient or slide group. Use this design for a manuscript
  generalization claim.
- **End-to-end**: all evaluable reference cells.
- **Direct consensus**: cells where the component phenotyping methods agreed.
- **Non-consensus**: disputed cells that require ML resolution. This is usually the
  hardest and most informative stratum.
- **Ref2 overlap**: keeping overlap is suitable for a regional head-to-head method
  comparison. Removing overlap is needed only when Ref2 is presented as a
  row-independent validation set.
"""
        )

    with st.expander("Why report micro-, macro-, and weighted-F1", expanded=False):
        st.markdown(
            """
- **Micro-F1** summarizes correctness across cells. In single-label multiclass
  classification it equals accuracy and is dominated by common phenotypes.
- **Macro-F1** gives every phenotype equal weight. It is the recommended primary
  SPACEMAP metric because rare cell populations remain biologically important.
- **Weighted-F1** weights each phenotype by its observed abundance and provides the
  prevalence-aware view.

A high micro-F1 with a lower macro-F1 means that abundant phenotypes are performing
better than rarer phenotypes. The three values are complementary rather than
interchangeable.
"""
        )


def _summary_display_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    columns = [
        "reference",
        "route",
        "group",
        "method",
        "n_test",
        "n_active_classes",
        "accuracy",
        "balanced_accuracy",
        "micro_f1",
        "macro_f1",
        "weighted_f1",
        "cohen_kappa",
        "prediction_coverage",
    ]
    display = summary.loc[
        :, [column for column in columns if column in summary.columns]
    ].copy()
    display = display.rename(
        columns={
            "reference": "Reference",
            "route": "Cells included",
            "group": "Group",
            "method": "Method",
            "n_test": "Cells",
            "n_active_classes": "Phenotypes",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced accuracy",
            "micro_f1": "Micro-F1",
            "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted-F1",
            "cohen_kappa": "Cohen kappa",
            "prediction_coverage": "Coverage",
        }
    )
    return display


def _render_analysis_summary(bundle: Mapping[str, Any]) -> None:
    summary = bundle["summary"]
    audit = bundle["reference_audit"]
    metadata = bundle["metadata"]
    values = available_result_values(bundle)
    primary_reference = values["references"][0]
    primary = summary.loc[
        summary["reference"].eq(primary_reference)
        & summary["route"].eq("End-to-end")
        & summary["group"].eq(GROUP_POOLED)
    ].sort_values("macro_f1", ascending=False)

    st.markdown("### Validation summary")
    if metadata.get("strict_group_separation_verified"):
        groups = metadata.get("strict_validation_groups", [])
        st.success(
            "Strict grouped retraining verified that every scored cell was "
            "predicted without fitting MR-ML weights, MR-ML, or Consensus-ML "
            f"on its complete held-out group ({len(groups)} groups)."
        )
        if metadata.get("strict_quick_mode"):
            st.warning(
                "This is a quick-preview run with capped tree counts. Use the "
                "fixed full configuration for manuscript-reported values."
            )
    elif metadata.get("prediction_design", "").startswith("Saved"):
        st.warning(
            "These are scores of saved prediction columns. Group panels are useful "
            "descriptive checks, but they are not strict grouped cross-validation "
            "unless the selected columns contain genuine out-of-fold predictions."
        )
    else:
        st.success(
            "Out-of-fold / held-out interpretation selected. Confirm that every "
            "prediction was generated without fitting on its complete evaluation group."
        )

    overlap = bundle.get("reference_overlap")
    has_overlap = isinstance(overlap, pd.DataFrame) and not overlap.empty
    if len(values["references"]) > 1 and not metadata.get("remove_primary_overlap"):
        if has_overlap:
            worst = overlap.iloc[0]
            st.warning(
                f"**{int(overlap['n_shared_cells'].sum()):,} cells carry labels in more than one "
                f"selected reference** (largest: {worst['reference_a']} / {worst['reference_b']} "
                f"share {int(worst['n_shared_cells']):,} cells = {worst['pct_of_b']:.1%} of "
                f"{worst['reference_b']}). Those rows are the same cell scored twice, so a model "
                "trained on one reference is being tested on cells it already saw. Enable "
                "'Remove primary-reference overlap' above to drop them. Note this removes only "
                "shared *rows* - cells from the same tissue section remain related, which needs "
                "a group hold-out to address."
            )
        else:
            st.info(
                "Additional references retain cells that overlap the primary reference. "
                "This is appropriate for a connected-region head-to-head comparison, but "
                "the additional reference should not be called fully independent."
            )
    elif has_overlap and metadata.get("remove_primary_overlap"):
        st.success(
            f"Overlap removal is on: {int(overlap['n_shared_cells'].sum()):,} multiply-labelled "
            "cells were dropped from the later references, leaving disjoint evaluation sets. "
            "See the reference audit for per-reference counts."
        )

    if not primary.empty:
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Evaluable cells",
            f"{int(primary.iloc[0]['n_test']):,}",
        )
        metric_cols[1].metric(
            "Phenotypes",
            f"{int(primary.iloc[0]['n_active_classes']):,}",
        )
        metric_cols[2].metric(
            "Best macro-F1",
            f"{float(primary.iloc[0]['macro_f1']):.3f}",
        )
        metric_cols[3].metric(
            "Best method",
            str(primary.iloc[0]["method"]),
        )

    st.markdown("#### Plain-language interpretation")
    for line in interpretation_lines(bundle):
        st.markdown(f"- {line}")

    with st.expander("Suggested reporting language", expanded=False):
        st.markdown(
            """
Report the validation design before the scores. State whether predictions were
ordinary saved outputs or truly out-of-fold. Identify macro-F1 as the primary
class-balanced comparison, then report micro- and weighted-F1 alongside it.

For Consensus-ML, report direct-consensus, non-consensus, and end-to-end results
separately. A close end-to-end result supports an annotation-sparing workflow;
superior MR-ML performance on non-consensus cells supports retaining manual
references when difficult-cell resolution is important.
"""
        )

    result_tabs = st.tabs(
        [
            "Summary metrics",
            "Per-class metrics",
            "Reference audit",
            "Confidence / coverage",
            "Annotator agreement",
        ]
    )
    with result_tabs[0]:
        summary_display = _summary_display_table(summary)
        st.dataframe(
            summary_display.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Balanced accuracy": "{:.3f}",
                    "Micro-F1": "{:.3f}",
                    "Macro-F1": "{:.3f}",
                    "Weighted-F1": "{:.3f}",
                    "Cohen kappa": "{:.3f}",
                    "Coverage": "{:.1%}",
                },
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
    with result_tabs[1]:
        st.dataframe(bundle["per_class"], width="stretch", hide_index=True)
    with result_tabs[2]:
        st.dataframe(audit, width="stretch", hide_index=True)
        if has_overlap:
            st.markdown("##### Cells shared between reference columns")
            st.dataframe(
                overlap.style.format(
                    {
                        "n_shared_cells": "{:,}",
                        "pct_of_a": "{:.1%}",
                        "pct_of_b": "{:.1%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Every pair listed here annotates some of the same cells. Independence "
                "claims between two references require this table to be empty."
            )
        grouped = summary.loc[summary["group"].ne(GROUP_POOLED)]
        if not grouped.empty:
            counts = grouped.pivot_table(
                index=["reference", "route", "group"],
                columns="method",
                values="n_test",
                aggfunc="first",
            )
            st.markdown("##### Evaluated cells by group")
            st.dataframe(counts, width="stretch")
    with result_tabs[3]:
        confidence = bundle["confidence"]
        if confidence.empty:
            st.info("No confidence-score mapping was analyzed.")
        else:
            st.dataframe(confidence, width="stretch", hide_index=True)
            st.caption(
                "Confidence is treated as a ranking/support score. Do not call it a "
                "calibrated probability unless a separate calibration analysis was performed."
            )
    with result_tabs[4]:
        human = bundle["human_summary"]
        if human.empty:
            st.info("No annotator pair was selected.")
        else:
            st.dataframe(human, width="stretch", hide_index=True)
            st.caption(
                "Human agreement contextualizes difficult cells; it is not an estimate "
                "of whole-dataset algorithm accuracy when the cells were deliberately enriched."
            )

    _render_strict_validation_details(bundle)


def _as_display_text(value: Any) -> str:
    """Render any audit value as text for display.

    Settings tables hold numbers, strings, lists and None in the same column.
    Arrow requires one type per column, so a mixed column raises during
    serialization and Streamlit falls back with a noisy traceback. Formatting
    everything as text keeps those tables readable and quiet.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _render_strict_validation_details(bundle: Mapping[str, Any]) -> None:
    strict = bundle.get("strict_validation")
    if not isinstance(strict, Mapping):
        return

    feature_audit = strict.get("feature_audit", {})
    paired = strict.get("paired_macro_f1_deltas", pd.DataFrame())
    fold_audit = strict.get("fold_audit", pd.DataFrame())
    internal_summary = strict.get("internal_summary", pd.DataFrame())
    weights = strict.get("f1_weights", pd.DataFrame())
    consensus_counts = strict.get(
        "consensus_training_counts",
        pd.DataFrame(),
    )

    st.markdown("### Strict grouped-validation audit")
    audit_cols = st.columns(4)
    audit_cols[0].metric(
        "Independent groups",
        f"{int(feature_audit.get('n_validation_groups', 0)):,}",
    )
    audit_cols[1].metric(
        "OOF reference rows",
        f"{int(feature_audit.get('n_mapped_reference_rows', 0)):,}",
    )
    audit_cols[2].metric(
        "Direct consensus",
        f"{float(feature_audit.get('direct_consensus_fraction', 0.0)):.1%}",
    )
    audit_cols[3].metric(
        "Selected markers",
        f"{int(feature_audit.get('n_markers', 0)):,}",
    )

    strict_tabs = st.tabs(
        [
            "Paired improvement",
            "Fold leakage audit",
            "Internal Ref1a / Ref1b",
            "Training details",
        ]
    )
    with strict_tabs[0]:
        if isinstance(paired, pd.DataFrame) and not paired.empty:
            st.dataframe(
                paired.style.format(
                    {
                        column: "{:.3f}"
                        for column in paired.columns
                        if "f1" in str(column)
                    },
                    na_rep="—",
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Positive deltas mean the ML workflow exceeded the best "
                "component method on the same held-out cells. Show every group; "
                "do not summarize three groups as a precise confidence interval."
            )
        else:
            st.info("No paired ML-versus-component comparison was available.")
    with strict_tabs[1]:
        if isinstance(fold_audit, pd.DataFrame) and not fold_audit.empty:
            st.dataframe(fold_audit, width="stretch", hide_index=True)
            leakage = fold_audit.get(
                "train_prediction_group_overlap",
                pd.Series(dtype=object),
            )
            no_leakage = bool(
                len(leakage)
                and leakage.map(
                    lambda values: not bool(values)
                    if isinstance(values, (list, tuple, set))
                    else False
                ).all()
            )
            if no_leakage:
                st.success(
                    "Audit passed: no held-out group appears in a grouped "
                    "model's training groups."
                )
        else:
            st.info("No fold audit was returned.")
    with strict_tabs[2]:
        if isinstance(internal_summary, pd.DataFrame) and not internal_summary.empty:
            st.dataframe(
                _summary_display_table(internal_summary).style.format(
                    {
                        "Accuracy": "{:.3f}",
                        "Balanced accuracy": "{:.3f}",
                        "Micro-F1": "{:.3f}",
                        "Macro-F1": "{:.3f}",
                        "Weighted-F1": "{:.3f}",
                        "Cohen kappa": "{:.3f}",
                        "Coverage": "{:.1%}",
                    },
                    na_rep="—",
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Ref1a to Ref1b removes exact-cell reuse, but remains an "
                "internal same-dataset check. The grouped OOF result above is "
                "the stronger generalization analysis."
            )
        else:
            st.info("The optional Ref1a to Ref1b analysis was not run.")
    with strict_tabs[3]:
        detail_tabs = st.tabs(
            ["Feature audit", "Fold-specific F1 weights", "Consensus training"]
        )
        with detail_tabs[0]:
            # The audit mixes numbers, text and lists in one column. Arrow infers a
            # single type per column and fails on the mixture, so render the values
            # as text -- this table is read, not computed on.
            feature_table = pd.DataFrame(
                [
                    {"setting": str(key), "value": _as_display_text(value)}
                    for key, value in feature_audit.items()
                ]
            )
            st.dataframe(feature_table, width="stretch", hide_index=True)
        with detail_tabs[1]:
            if isinstance(weights, pd.DataFrame) and not weights.empty:
                st.dataframe(weights, width="stretch", hide_index=True)
            else:
                st.info("No MR-ML weighting table was returned.")
        with detail_tabs[2]:
            if (
                isinstance(consensus_counts, pd.DataFrame)
                and not consensus_counts.empty
            ):
                st.dataframe(
                    consensus_counts,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No Consensus-ML training-count table was returned.")


def _build_figures(
    bundle: Mapping[str, Any],
    *,
    selected_figures: list[str],
    reference: str,
    route: str,
    group: str,
    method_order: list[str],
    confusion_method: str,
    style: ValidationStyle,
    highlight_threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    figures: dict[str, Any] = {}
    skipped: list[str] = []

    builders: dict[str, Any] = {
        "Per-class F1 heatmap": lambda: plot_per_class_heatmap(
            bundle,
            reference=reference,
            route=route,
            group=group,
            style=style,
            method_order=method_order,
        ),
        "Micro / macro / weighted-F1 comparison": lambda: plot_f1_metric_comparison(
            bundle,
            reference=reference,
            route=route,
            group=group,
            style=style,
            method_order=method_order,
        ),
        "References × assignment routes": lambda: plot_reference_route_comparison(
            bundle,
            style=style,
            method_order=method_order,
        ),
        "Grouped macro-F1": lambda: plot_grouped_macro_f1(
            bundle,
            reference=reference,
            style=style,
            method_order=method_order,
        ),
        "Precision / recall / F1 heatmaps": lambda: plot_precision_recall_f1_heatmaps(
            bundle,
            reference=reference,
            route=route,
            group=group,
            style=style,
            method_order=method_order,
        ),
        "Normalized confusion matrix": lambda: plot_normalized_confusion(
            bundle,
            reference=reference,
            route=route,
            group=group,
            method=confusion_method,
            style=style,
        ),
        "Confidence versus coverage": lambda: plot_confidence_coverage(
            bundle,
            highlight_threshold=highlight_threshold,
            style=style,
            method_order=method_order,
        ),
        "Inter-annotator agreement": lambda: plot_human_agreement(
            bundle,
            style=style,
        ),
        "Radar performance summary": lambda: plot_radar_summary(
            bundle,
            reference=reference,
            route=route,
            group=group,
            style=style,
            method_order=method_order,
        ),
    }

    for figure_name in selected_figures:
        builder = builders.get(figure_name)
        if builder is None:
            continue
        try:
            figures[figure_name] = builder()
        except ValueError as exc:
            skipped.append(f"{figure_name}: {exc}")
        except Exception as exc:
            logger.exception("Failed to create F1 figure %s", figure_name)
            skipped.append(f"{figure_name}: {exc}")
    return figures, skipped


def _render_figure_and_export_panel(
    project_root: Path,
    output_dir_default: str,
    bundle: Mapping[str, Any],
    figures: dict[str, Any],
    style: ValidationStyle,
    focus: Mapping[str, str],
) -> None:
    st.markdown("### Figure previews")
    if not figures:
        st.info("No selected figure could be generated from the current analysis.")
        return

    for index, (name, figure) in enumerate(figures.items()):
        with st.expander(name, expanded=index < 2):
            st.pyplot(figure)
            png_bytes = figure_to_png_bytes(figure, dpi=style.dpi)
            st.download_button(
                f"Download {name} as PNG",
                data=png_bytes,
                file_name=f"{sanitize_filename(name)}.png",
                mime="image/png",
            )

    st.markdown("### Save selected outputs")
    st.caption(
        "Nothing is written during analysis or preview. Files are created only when "
        "you press the save button below. PNG is the only figure format."
    )
    default_output = str(Path(output_dir_default) / "F1_validation")
    output_dir = st.text_input(
        "Output directory",
        value=str(
            st.session_state.get(
                "viz_f1_output_dir",
                default_output,
            )
        ),
        key="viz_f1_output_dir",
    )
    save_figure_names = st.multiselect(
        "PNG figures to save",
        options=list(figures),
        default=list(figures),
        key="viz_f1_save_figure_names",
    )
    table_map = {
        "Summary metrics": bundle["summary"],
        "Per-class metrics": bundle["per_class"],
        "Reference audit": bundle["reference_audit"],
        "Confidence coverage": bundle["confidence"],
        "Inter-annotator summary": bundle["human_summary"],
        "Confusion matrix data": bundle["confusion"],
        "Inter-annotator confusion data": bundle["human_confusion"],
        "Reference overlap": bundle.get("reference_overlap", pd.DataFrame()),
    }
    strict = bundle.get("strict_validation")
    if isinstance(strict, Mapping):
        table_map.update(
            {
                "Strict OOF predictions": strict.get(
                    "oof_predictions",
                    pd.DataFrame(),
                ),
                "Strict fold leakage audit": strict.get(
                    "fold_audit",
                    pd.DataFrame(),
                ),
                "Paired macro-F1 deltas": strict.get(
                    "paired_macro_f1_deltas",
                    pd.DataFrame(),
                ),
                "Fold-specific MR-ML F1 weights": strict.get(
                    "f1_weights",
                    pd.DataFrame(),
                ),
                "Consensus training counts": strict.get(
                    "consensus_training_counts",
                    pd.DataFrame(),
                ),
                "Internal split predictions": strict.get(
                    "internal_predictions",
                    pd.DataFrame(),
                ),
                "Internal split summary": strict.get(
                    "internal_summary",
                    pd.DataFrame(),
                ),
                "Internal split per-class metrics": strict.get(
                    "internal_per_class",
                    pd.DataFrame(),
                ),
            }
        )
    available_tables = [
        name
        for name, table in table_map.items()
        if isinstance(table, pd.DataFrame) and not table.empty
    ]
    default_tables = [
        "Summary metrics",
        "Per-class metrics",
        "Reference audit",
    ]
    if isinstance(strict, Mapping):
        default_tables.extend(["Strict fold leakage audit", "Paired macro-F1 deltas"])
    default_tables = [name for name in default_tables if name in available_tables]
    save_table_names = st.multiselect(
        "CSV tables to save",
        options=available_tables,
        default=default_tables,
        key="viz_f1_save_table_names",
    )
    save_settings = st.checkbox(
        "Save analysis settings JSON",
        value=True,
        key="viz_f1_save_settings_json",
    )

    if st.button(
        "Save Selected Validation Outputs",
        type="primary",
    ):
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        run_id = str(
            bundle["metadata"].get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        context_slug = sanitize_filename(
            f"ref-{focus['reference']}__route-{focus['route']}__group-{focus['group']}"
        )
        saved: list[Path] = []
        for name in save_figure_names:
            figure = figures[name]
            path = target / (
                f"f1__{sanitize_filename(name)}__{context_slug}__{run_id}.png"
            )
            figure.savefig(
                path,
                format="png",
                dpi=style.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            saved.append(path)
        for name in save_table_names:
            path = target / (f"f1__{sanitize_filename(name)}__{run_id}.csv")
            table_map[name].to_csv(path, index=False)
            saved.append(path)
        if save_settings:
            path = target / f"f1__analysis-settings__{run_id}.json"
            payload = {
                "metadata": bundle["metadata"],
                "style": style_to_dict(style),
                "figure_focus": dict(focus),
                "saved_figures": save_figure_names,
                "saved_tables": save_table_names,
                "strict_configuration": (
                    strict.get("configuration") if isinstance(strict, Mapping) else None
                ),
                "strict_feature_audit": (
                    strict.get("feature_audit") if isinstance(strict, Mapping) else None
                ),
            }
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            saved.append(path)

        st.success(f"Saved {len(saved)} selected output file(s).")
        st.code("\n".join(str(path) for path in saved))
        log_console(
            f"[{timestamp()}] Saved {len(saved)} F1 validation outputs to {target}"
        )

    for figure in figures.values():
        plt.close(figure)


def _render_option1_figures(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
    output_dir_default: str,
    *,
    use_shared_predictions: bool,
    shared_analysis: Mapping[str, Any] | None,
    shared_strict_settings: Mapping[str, Any] | None,
) -> None:
    """Render the comprehensive, generic phenotyping-validation workflow."""

    st.markdown("### Phenotyping Validation")
    st.caption(
        "Compare any phenotype columns against one or more manual references, "
        "with optional group, consensus-route, confidence, and annotator analyses."
    )
    _render_concept_guide()

    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.error("Select an existing unified or out-of-fold validation CSV above.")
        return

    columns = [str(column) for column in available_columns]
    detected_methods = prediction_candidates(columns)
    detected_references = reference_candidates(columns)
    detected_groups = group_candidates(columns)
    if use_shared_predictions:
        reference_options = [
            str(column)
            for column in st.session_state.get(
                "viz_f1_shared_reference_columns", detected_references
            )
            if str(column) in columns
        ]
        method_options = list(shared_strict_settings.get("component_methods", [])) if isinstance(shared_strict_settings, Mapping) else detected_methods
        group_options = [
            str(column)
            for column in st.session_state.get(
                "viz_f1_shared_group_columns", detected_groups
            )
            if str(column) in columns
        ]
    else:
        reference_options = [
            str(column)
            for column in st.session_state.get(
                "viz_f1_shared_existing_reference_columns", detected_references
            )
            if str(column) in columns
        ]
        method_options = [
            str(column)
            for column in st.session_state.get(
                "viz_f1_shared_existing_method_columns", detected_methods
            )
            if str(column) in columns
        ]
        existing_group = str(
            st.session_state.get("viz_f1_shared_existing_group_column", "(None)")
        )
        group_options = (
            [existing_group]
            if existing_group in columns and existing_group != "(None)"
            else list(detected_groups)
        )
    route_options = route_candidates(columns)
    confidence_options = confidence_candidates(columns)

    if not reference_options or not method_options:
        st.error("The selected prediction source does not expose usable columns.")
        return

    default_methods = (
        list(method_options)
        if use_shared_predictions
        else (default_method_columns(columns) or detected_methods)
    )
    default_references = (
        list(reference_options)
        if use_shared_predictions
        else (_default_reference_columns(columns) or detected_references)
    )
    strict_input = _default_prediction_design(columns) == PREDICTION_DESIGNS[1]
    default_group = (
        "validation_group"
        if "validation_group" in group_options
        else (
            "batch_id"
            if "batch_id" in group_options
            else (group_options[0] if group_options else "")
        )
    )
    default_route = route_options[0] if route_options else ""
    confidence_defaults = default_confidence_mapping(columns)

    strict_settings = dict(shared_strict_settings or {})
    strict_enabled = bool(use_shared_predictions)
    if strict_enabled:
        st.success(
            "Option 1 is using the shared held-out prediction data configured above. "
            "Running this analysis recalculates summary tables and figures only; it "
            "does not refit MR-ML or Consensus-ML."
        )

    configured_methods = _safe_defaults(
        method_options,
        _cached(
            cached_params,
            "viz_f1_methods_select",
            default_methods,
        ),
        default_methods,
    )
    configured_display_mapping = parse_mapping_text(
        str(
            _cached(
                cached_params,
                "viz_f1_display_mapping",
                mapping_to_text(_default_display_mapping(configured_methods)),
            )
        )
    )
    reset_color_targets = _method_color_targets(
        configured_methods,
        configured_display_mapping,
        strict_settings=strict_settings if strict_enabled else None,
    )
    if st.button(
        "Restore Option 2 method colors in Option 1",
        key="viz_f1_restore_option2_method_colors",
        help=(
            "Restores the Figure 3 defaults for Leiden, SOM, Vortex, RESOLVE, "
            "ML-MR, and Consensus-ML. Unknown method names receive the same stable "
            "name-based default used by Option 2. Analysis and ML settings are unchanged."
        ),
    ):
        for index, (method, _label) in enumerate(reset_color_targets):
            _default_label, default_color = method_style_default(method, index)
            st.session_state[_method_color_key(method)] = default_color
        _persist_widget_configuration(project_root, cached_params)
        st.success(
            "Option 1 method colors now match the Option 2 defaults. They remain editable below."
        )

    st.markdown("## 1. Analysis design")
    with st.form("viz_f1_validation_form", clear_on_submit=False):
        if strict_enabled:
            st.info(
                "Strict mode supplies prediction provenance, validation groups, "
                "consensus routes, and ML confidence columns automatically. "
                "Use this form to choose references, exclusions, and figure style."
            )
        prediction_design = st.selectbox(
            "Where these predictions came from",
            options=PREDICTION_DESIGNS,
            index=PREDICTION_DESIGNS.index(
                _safe_choice(
                    PREDICTION_DESIGNS,
                    _cached(
                        cached_params,
                        "viz_f1_prediction_design",
                        _default_prediction_design(columns),
                    ),
                    _default_prediction_design(columns),
                )
            ),
            key="viz_f1_prediction_design",
            disabled=strict_enabled,
            help=(
                "This controls interpretation, not the arithmetic. Select out-of-fold "
                "only when each cell was predicted by a model that excluded its group."
            ),
        )
        if strict_enabled:
            prediction_design = PREDICTION_DESIGNS[1]

        design_cols = st.columns(2)
        with design_cols[0]:
            references = st.multiselect(
                "Score the result against these manual labels",
                options=reference_options,
                default=_safe_defaults(
                    reference_options,
                    _cached(
                        cached_params,
                        "viz_f1_reference_columns",
                        default_references,
                    ),
                    default_references,
                ),
                key="viz_f1_reference_columns",
                help=(
                    "Select Ref1, Ref2, BD/AP, or any other manually assigned columns. "
                    "Each reference is scored separately."
                ),
            )
        with design_cols[1]:
            methods = st.multiselect(
                "Which phenotype columns to compare",
                options=method_options,
                default=_safe_defaults(
                    method_options,
                    _cached(
                        cached_params,
                        "viz_f1_methods_select",
                        default_methods,
                    ),
                    default_methods,
                ),
                key="viz_f1_methods_select",
                disabled=strict_enabled,
                help=(
                    "Strict mode uses the component methods selected in the "
                    "advanced panel. Otherwise, all selected saved methods are "
                    "evaluated on the same reference rows."
                ),
            )

        display_mapping_default = mapping_to_text(_default_display_mapping(methods))
        display_mapping_text = st.text_area(
            "Rename columns for the figures (optional)",
            value=str(
                _cached(
                    cached_params,
                    "viz_f1_display_mapping",
                    display_mapping_default,
                )
            ),
            key="viz_f1_display_mapping",
            height=110,
        )

        current_display_mapping = parse_mapping_text(display_mapping_text)
        color_targets = _method_color_targets(
            methods,
            current_display_mapping,
            strict_settings=strict_settings if strict_enabled else None,
        )
        st.session_state["viz_f1_method_color_registry"] = [
            {"method": method, "label": label}
            for method, label in color_targets
        ]
        with st.expander(
            "Method colors — matched to Option 2 defaults",
            expanded=True,
        ):
            st.info(
                "These colors identify phenotyping methods in Option 1 bars, points, "
                "lines, radar traces, and legends. Known methods start with exactly "
                "the Option 2 Figure 3 colors and remain stable if methods are reordered. "
                "Heatmap fill colors still represent numeric scores."
            )
            color_columns = st.columns(3)
            for index, (method, label) in enumerate(color_targets):
                default_label, default_color = method_style_default(method, index)
                color_key = _method_color_key(method)
                if color_key not in st.session_state:
                    st.session_state[color_key] = _safe_hex_color(
                        _cached(cached_params, color_key, default_color),
                        default_color,
                    )
                with color_columns[index % 3]:
                    st.color_picker(
                        f"{label or default_label} color",
                        key=color_key,
                        help=(
                            f"Color for `{method}` wherever this method is encoded by "
                            "color. Its Option 2 default is "
                            f"{default_label} = {default_color}."
                        ),
                    )

        filter_cols = st.columns(2)
        with filter_cols[0]:
            excluded_text = st.text_input(
                "Labels to treat as not-a-phenotype",
                value=str(
                    _cached(
                        cached_params,
                        "viz_f1_excluded_labels",
                        "Others, Unclassified, Unknown",
                    )
                ),
                key="viz_f1_excluded_labels",
                help=(
                    "These labels are not treated as biological targets. Predictions "
                    "with these labels count as rejected/unassigned."
                ),
            )
        with filter_cols[1]:
            remove_overlap = st.checkbox(
                "Drop cells labelled in more than one reference",
                value=bool(
                    _cached(
                        cached_params,
                        "viz_f1_remove_reference_overlap",
                        False,
                    )
                ),
                key="viz_f1_remove_reference_overlap",
                help=(
                    "Drops any cell already labelled by an earlier reference in the list, "
                    "so the evaluation sets become mutually disjoint. Turn this on whenever "
                    "a later reference is presented as validation for a model trained on an "
                    "earlier one. It removes shared rows only; cells from the same tissue "
                    "section stay related, so also hold out that patient or slide."
                ),
            )

        st.markdown("#### Group-wise analysis")
        group_enabled = st.checkbox(
            "Report scores separately per patient or slide",
            value=bool(
                _cached(
                    cached_params,
                    "viz_f1_group_enabled",
                    strict_input,
                )
            ),
            key="viz_f1_group_enabled",
            disabled=strict_enabled,
        )
        if strict_enabled:
            group_enabled = False
        if group_enabled and group_options:
            group_cols = st.columns(2)
            with group_cols[0]:
                group_column = st.selectbox(
                    "Column identifying the patient or slide",
                    options=group_options,
                    index=group_options.index(
                        _safe_choice(
                            group_options,
                            _cached(
                                cached_params,
                                "viz_f1_group_column",
                                default_group,
                            ),
                            default_group,
                        )
                    ),
                    key="viz_f1_group_column",
                )
            with group_cols[1]:
                group_mode = st.selectbox(
                    "How to turn that column into groups",
                    options=GROUP_MODES,
                    index=GROUP_MODES.index(
                        _safe_choice(
                            GROUP_MODES,
                            _cached(
                                cached_params,
                                "viz_f1_group_mode",
                                "As stored" if strict_input else "Remove ROI suffix",
                            ),
                            "As stored",
                        )
                    ),
                    key="viz_f1_group_mode",
                    help=(
                        "Remove ROI suffix combines values such as P28_roi_1 and "
                        "P28_roi_2 into P28. Confirm this reflects the real metadata."
                    ),
                )
            group_mapping_text = st.text_area(
                "Manual grouping (raw value = group name)",
                value=str(
                    _cached(
                        cached_params,
                        "viz_f1_group_mapping",
                        "",
                    )
                ),
                key="viz_f1_group_mapping",
                height=100,
                disabled=group_mode != "Manual mapping",
            )
        else:
            group_column = None
            group_mode = "As stored"
            group_mapping_text = ""

        st.markdown("#### Consensus assignment routes")
        route_enabled = st.checkbox(
            "Split results by whether the methods agreed",
            value=bool(
                _cached(
                    cached_params,
                    "viz_f1_route_enabled",
                    bool(route_options),
                )
            ),
            key="viz_f1_route_enabled",
            disabled=strict_enabled or not bool(route_options),
        )
        if strict_enabled:
            route_enabled = False
        if route_enabled and route_options:
            route_column = st.selectbox(
                "Column marking where the methods agreed",
                options=route_options,
                index=route_options.index(
                    _safe_choice(
                        route_options,
                        _cached(
                            cached_params,
                            "viz_f1_route_column",
                            default_route,
                        ),
                        default_route,
                    )
                ),
                key="viz_f1_route_column",
                help=(
                    "Accepted direct values include True, 1, consensus, direct, "
                    "and direct consensus. All other values are non-consensus."
                ),
            )
        else:
            route_column = None

        optional_tabs = st.tabs(
            [
                "Confidence threshold analysis",
                "Inter-annotator analysis",
            ]
        )
        with optional_tabs[0]:
            confidence_enabled = st.checkbox(
                "Show how scores improve if low-confidence calls are dropped",
                value=bool(
                    _cached(
                        cached_params,
                        "viz_f1_confidence_enabled",
                        bool(confidence_defaults),
                    )
                ),
                key="viz_f1_confidence_enabled",
                disabled=strict_enabled or not bool(confidence_options),
            )
            if strict_enabled:
                confidence_enabled = True
            confidence_mapping_text = st.text_area(
                "Which confidence column belongs to which prediction",
                value=str(
                    _cached(
                        cached_params,
                        "viz_f1_confidence_mapping",
                        mapping_to_text(confidence_defaults),
                    )
                ),
                key="viz_f1_confidence_mapping",
                height=100,
                disabled=strict_enabled or not confidence_enabled,
            )
            threshold_text = st.text_input(
                "Confidence thresholds",
                value=str(
                    _cached(
                        cached_params,
                        "viz_f1_confidence_thresholds",
                        "0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95",
                    )
                ),
                key="viz_f1_confidence_thresholds",
                disabled=not confidence_enabled,
            )
            highlight_threshold = st.number_input(
                "Confidence level to highlight",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    _cached(
                        cached_params,
                        "viz_f1_highlight_threshold",
                        0.75,
                    )
                ),
                step=0.05,
                key="viz_f1_highlight_threshold",
                disabled=not confidence_enabled,
            )
        with optional_tabs[1]:
            annotator_enabled = st.checkbox(
                "Compare two human annotators with each other",
                value=bool(
                    _cached(
                        cached_params,
                        "viz_f1_annotator_enabled",
                        "phenotype_bd_y" in reference_options
                        and "phenotype_ap" in reference_options,
                    )
                ),
                key="viz_f1_annotator_enabled",
            )
            annotation_cols = st.columns(2)
            default_a = (
                "phenotype_bd_y"
                if "phenotype_bd_y" in reference_options
                else reference_options[0]
            )
            default_b = (
                "phenotype_ap"
                if "phenotype_ap" in reference_options
                else reference_options[-1]
            )
            with annotation_cols[0]:
                annotator_a = st.selectbox(
                    "Annotator A column",
                    options=reference_options,
                    index=reference_options.index(
                        _safe_choice(
                            reference_options,
                            _cached(
                                cached_params,
                                "viz_f1_annotator_a",
                                default_a,
                            ),
                            default_a,
                        )
                    ),
                    key="viz_f1_annotator_a",
                    disabled=not annotator_enabled,
                )
            with annotation_cols[1]:
                annotator_b = st.selectbox(
                    "Annotator B column",
                    options=reference_options,
                    index=reference_options.index(
                        _safe_choice(
                            reference_options,
                            _cached(
                                cached_params,
                                "viz_f1_annotator_b",
                                default_b,
                            ),
                            default_b,
                        )
                    ),
                    key="viz_f1_annotator_b",
                    disabled=not annotator_enabled,
                )

        st.markdown("## 2. Figures and publication style")
        st.multiselect(
            "Figures to prepare",
            options=ALL_FIGURES,
            default=_safe_defaults(
                ALL_FIGURES,
                _cached(
                    cached_params,
                    "viz_f1_figures_v2",
                    DEFAULT_FIGURES,
                ),
                DEFAULT_FIGURES,
            ),
            key="viz_f1_figures_v2",
            help="Only selected figures are previewed. Saving remains a separate action.",
        )
        st.text_input(
            "Title prefix for the figures",
            value=str(
                _cached(
                    cached_params,
                    "viz_f1_dataset_title",
                    "",
                )
            ),
            key="viz_f1_dataset_title",
        )

        style_cols = st.columns(3)
        with style_cols[0]:
            st.selectbox(
                "Font family",
                options=FONT_FAMILIES,
                index=FONT_FAMILIES.index(
                    _safe_choice(
                        FONT_FAMILIES,
                        _cached(
                            cached_params,
                            "viz_f1_font_family",
                            "DejaVu Sans",
                        ),
                        "DejaVu Sans",
                    )
                ),
                key="viz_f1_font_family",
            )
            st.selectbox(
                "Fallback palette for references/unmapped series",
                options=list(PALETTES),
                index=list(PALETTES).index(
                    _safe_choice(
                        list(PALETTES),
                        _cached(
                            cached_params,
                            "viz_f1_palette",
                            "GUI classic",
                        ),
                        "GUI classic",
                    )
                ),
                key="viz_f1_palette",
                help=(
                    "Used only when a plotted series has no method-specific color "
                    "above, for example manual references in confidence plots."
                ),
            )
            st.selectbox(
                "F1 heatmap color map",
                options=HEATMAP_CMAPS,
                index=HEATMAP_CMAPS.index(
                    _safe_choice(
                        HEATMAP_CMAPS,
                        _cached(
                            cached_params,
                            "viz_f1_heatmap_cmap",
                            "Blues",
                        ),
                        "Blues",
                    )
                ),
                key="viz_f1_heatmap_cmap",
            )
        with style_cols[1]:
            st.number_input(
                "Base figure width (inches)",
                min_value=4.0,
                max_value=30.0,
                value=float(_cached(cached_params, "viz_f1_figure_width", 10.0)),
                step=0.5,
                key="viz_f1_figure_width",
            )
            st.number_input(
                "Base figure height (inches)",
                min_value=3.0,
                max_value=24.0,
                value=float(_cached(cached_params, "viz_f1_figure_height", 6.0)),
                step=0.5,
                key="viz_f1_figure_height",
            )
            st.number_input(
                "Heatmap row height (inches)",
                min_value=0.2,
                max_value=1.5,
                value=float(_cached(cached_params, "viz_f1_row_height", 0.45)),
                step=0.05,
                key="viz_f1_row_height",
            )
            st.number_input(
                "PNG resolution (DPI)",
                min_value=150,
                max_value=600,
                value=int(_cached(cached_params, "viz_f1_dpi", 300)),
                step=50,
                key="viz_f1_dpi",
            )
        with style_cols[2]:
            st.number_input(
                "Base font size",
                min_value=6,
                max_value=24,
                value=int(_cached(cached_params, "viz_f1_base_fontsize", 10)),
                step=1,
                key="viz_f1_base_fontsize",
            )
            st.number_input(
                "Title font size",
                min_value=8,
                max_value=32,
                value=int(_cached(cached_params, "viz_f1_title_fontsize", 14)),
                step=1,
                key="viz_f1_title_fontsize",
            )
            st.number_input(
                "Axis-label font size",
                min_value=6,
                max_value=28,
                value=int(_cached(cached_params, "viz_f1_axis_fontsize", 11)),
                step=1,
                key="viz_f1_axis_fontsize",
            )
            st.number_input(
                "Tick-label font size",
                min_value=5,
                max_value=24,
                value=int(_cached(cached_params, "viz_f1_tick_fontsize", 9)),
                step=1,
                key="viz_f1_tick_fontsize",
            )

        annotation_cols = st.columns(3)
        with annotation_cols[0]:
            show_annotations = st.checkbox(
                "Show heatmap values",
                value=bool(_cached(cached_params, "viz_f1_show_annot", True)),
                key="viz_f1_show_annot",
            )
            st.checkbox(
                "Square heatmap cells",
                value=bool(_cached(cached_params, "viz_f1_square_cells", False)),
                key="viz_f1_square_cells",
                help=(
                    "Draw each cell as a perfect square instead of stretching it to "
                    "fill the figure. The figure grows or shrinks to fit."
                ),
            )
        with annotation_cols[1]:
            st.number_input(
                "Heatmap annotation size",
                min_value=4,
                max_value=24,
                value=int(_cached(cached_params, "viz_f1_annot_fontsize", 8)),
                step=1,
                key="viz_f1_annot_fontsize",
                disabled=not show_annotations,
            )
        with annotation_cols[2]:
            st.slider(
                "Heatmap color scale",
                min_value=0.0,
                max_value=1.0,
                value=tuple(
                    _cached(
                        cached_params,
                        "viz_f1_color_range",
                        (0.5, 1.0),
                    )
                ),
                step=0.05,
                key="viz_f1_color_range",
            )

        submit_cols = st.columns(2)
        run_analysis = submit_cols[0].form_submit_button(
            (
                "Prepare Option 1 from shared predictions"
                if strict_enabled
                else "Run Validation Analysis on existing columns"
            ),
            type="primary",
            width="stretch",
        )
        save_configuration = submit_cols[1].form_submit_button(
            "Save Settings Only",
            width="stretch",
        )

    if save_configuration:
        _persist_widget_configuration(project_root, cached_params)
        st.success("F1 validation settings saved to the project configuration.")

    if run_analysis:
        if not references:
            st.error("Select at least one manual-reference column.")
        elif strict_enabled and strict_settings.get("configuration_error"):
            st.error(str(strict_settings["configuration_error"]))
        elif strict_enabled and not isinstance(shared_analysis, Mapping):
            st.error(
                "Run or load the shared held-out ML prediction data in the common "
                "zone above before preparing Option 1 figures."
            )
        elif strict_enabled and not (
            strict_settings.get("run_grouped_mr")
            or strict_settings.get("run_grouped_consensus")
        ):
            st.error("Strict mode needs grouped MR-ML, grouped Consensus-ML, or both.")
        elif strict_enabled and len(strict_settings.get("component_methods", [])) < 2:
            st.error("Select at least two component phenotyping methods.")
        elif strict_enabled and not strict_settings.get("markers"):
            st.error("Select at least one training marker column.")
        elif not strict_enabled and not methods:
            st.error("Select at least one prediction column.")
        elif annotator_enabled and annotator_a == annotator_b:
            st.error("Select two different annotator columns.")
        else:
            log_lines: list[str] = []

            def _log(message: str) -> None:
                line = f"[{timestamp()}] {message}"
                log_lines.append(line)
                log_console(line)
                logger.info(line)

            try:
                with st.status(
                    "Running phenotyping validation...",
                    expanded=True,
                ) as status:
                    status.write("Loading the selected CSV")
                    frame = get_cached_unified_df(unified_csv_path)
                    _log(f"Loaded {len(frame):,} rows")

                    display_mapping = parse_mapping_text(display_mapping_text)
                    annotator_pair = (
                        (annotator_a, annotator_b) if annotator_enabled else None
                    )
                    excluded_labels = _parse_label_list(excluded_text)
                    confidence_thresholds = _parse_thresholds(threshold_text)

                    if strict_enabled:
                        status.write(
                            "Reusing saved held-out predictions; no ML models are being fitted"
                        )
                        bundle = build_strict_validation_bundle_from_cached_analysis(
                            shared_analysis,
                            settings=strict_settings,
                            selected_reference_columns=references,
                            excluded_labels=excluded_labels,
                            method_display_names=display_mapping,
                            remove_primary_overlap=remove_overlap,
                            confidence_thresholds=confidence_thresholds,
                            annotator_columns=annotator_pair,
                        )
                    else:
                        confidence_mapping = (
                            parse_mapping_text(confidence_mapping_text)
                            if confidence_enabled
                            else {}
                        )
                        confidence_mapping = {
                            method: confidence
                            for method, confidence in confidence_mapping.items()
                            if method in methods and confidence in frame.columns
                        }
                        status.write("Calculating pooled, route, and group metrics")
                        bundle = analyze_validation(
                            frame,
                            reference_columns=references,
                            method_columns=methods,
                            excluded_labels=excluded_labels,
                            method_display_names=display_mapping,
                            route_column=route_column,
                            group_column=(group_column if group_enabled else None),
                            group_mode=group_mode,
                            group_mapping=parse_mapping_text(group_mapping_text),
                            remove_primary_overlap=remove_overlap,
                            confidence_mapping=confidence_mapping,
                            confidence_thresholds=confidence_thresholds,
                            annotator_columns=annotator_pair,
                            prediction_design=prediction_design,
                        )
                    bundle["metadata"].update(
                        {
                            "input_csv": str(unified_csv_path),
                            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            "highlight_threshold": float(highlight_threshold),
                            "prediction_source_mode": (
                                "shared_heldout"
                                if strict_enabled
                                else "existing_columns"
                            ),
                        }
                    )
                    st.session_state["f1_runtime_bundle"] = bundle
                    st.session_state["f1_runtime_log"] = log_lines
                    _persist_widget_configuration(project_root, cached_params)
                    status.update(
                        label="Phenotyping validation complete",
                        state="complete",
                    )
                    _log(
                        f"Created {len(bundle['summary']):,} summary rows and "
                        f"{len(bundle['per_class']):,} per-class rows"
                    )
            except Exception as exc:
                logger.exception("F1 validation failed")
                st.error(f"Validation failed: {exc}")

    bundle = st.session_state.get("f1_runtime_bundle")
    if not isinstance(bundle, dict):
        st.info(
            (
                "Configure Option 1 and press Prepare Option 1 from shared "
                "predictions. The saved ML outputs are reused; no model is refitted."
                if use_shared_predictions
                else "Configure Option 1 and run validation on the selected existing columns."
            )
        )
        return

    required_tables = ("summary", "per_class", "reference_audit")
    incompatible_tables = [
        name
        for name in required_tables
        if not isinstance(bundle.get(name), pd.DataFrame)
    ]
    metadata = bundle.get("metadata")
    if incompatible_tables or not isinstance(metadata, Mapping):
        st.session_state.pop("f1_runtime_bundle", None)
        st.session_state.pop("f1_runtime_log", None)
        st.info(
            "Cached F1 results came from an earlier version of this tab and were "
            "cleared safely. Press Run Validation Analysis to rebuild them."
        )
        return

    if metadata.get("input_csv") != str(unified_csv_path):
        st.warning(
            "The displayed results came from a different CSV. Run the analysis again "
            "for the currently selected input."
        )
    expected_source_mode = (
        "shared_heldout" if use_shared_predictions else "existing_columns"
    )
    if metadata.get("prediction_source_mode") not in {None, expected_source_mode}:
        st.warning(
            "The displayed Option 1 results use the other prediction source. Press "
            "Run Validation Analysis to rebuild the lightweight summaries from the "
            "currently selected source."
        )

    _render_analysis_summary(bundle)

    values = available_result_values(bundle)
    if not values["references"] or not values["methods"]:
        return

    st.markdown("## 3. Figure focus")
    focus_cols = st.columns(4)
    with focus_cols[0]:
        reference_focus = st.selectbox(
            "Manual labels shown in the detailed figures",
            options=values["references"],
            index=values["references"].index(
                _safe_choice(
                    values["references"],
                    st.session_state.get(
                        "viz_f1_focus_reference",
                        values["references"][0],
                    ),
                    values["references"][0],
                )
            ),
            key="viz_f1_focus_reference",
        )
    with focus_cols[1]:
        route_focus = st.selectbox(
            "Which cells to show",
            options=values["routes"],
            index=values["routes"].index(
                _safe_choice(
                    values["routes"],
                    st.session_state.get(
                        "viz_f1_focus_route",
                        "End-to-end",
                    ),
                    "End-to-end",
                )
            ),
            key="viz_f1_focus_route",
        )
    available_groups = [
        group
        for group in values["groups"]
        if (
            (
                bundle["summary"]["reference"].eq(reference_focus)
                & bundle["summary"]["route"].eq(route_focus)
                & bundle["summary"]["group"].eq(group)
            ).any()
        )
    ]
    with focus_cols[2]:
        group_focus = st.selectbox(
            "Patient or slide shown in the detailed figures",
            options=available_groups,
            index=available_groups.index(
                _safe_choice(
                    available_groups,
                    st.session_state.get(
                        "viz_f1_focus_group",
                        GROUP_POOLED,
                    ),
                    GROUP_POOLED,
                )
            ),
            key="viz_f1_focus_group",
        )
    with focus_cols[3]:
        confusion_method = st.selectbox(
            "Method for the confusion matrix",
            options=values["methods"],
            index=values["methods"].index(
                _safe_choice(
                    values["methods"],
                    st.session_state.get(
                        "viz_f1_confusion_method",
                        values["methods"][0],
                    ),
                    values["methods"][0],
                )
            ),
            key="viz_f1_confusion_method",
        )

    method_order = st.multiselect(
        "Methods shown and their figure order",
        options=values["methods"],
        default=_safe_defaults(
            values["methods"],
            st.session_state.get(
                "viz_f1_figure_method_order",
                values["methods"],
            ),
            values["methods"],
        ),
        key="viz_f1_figure_method_order",
    )
    if not method_order:
        method_order = values["methods"]

    style = _style_from_widgets()
    selected_figure_names = list(
        st.session_state.get("viz_f1_figures_v2", DEFAULT_FIGURES)
    )
    highlight = float(
        bundle["metadata"].get(
            "highlight_threshold",
            st.session_state.get("viz_f1_highlight_threshold", 0.75),
        )
    )
    figures, skipped = _build_figures(
        bundle,
        selected_figures=selected_figure_names,
        reference=reference_focus,
        route=route_focus,
        group=group_focus,
        method_order=method_order,
        confusion_method=confusion_method,
        style=style,
        highlight_threshold=highlight,
    )
    for message in skipped:
        st.info(message)

    _render_figure_and_export_panel(
        project_root,
        output_dir_default,
        bundle,
        figures,
        style,
        {
            "reference": reference_focus,
            "route": route_focus,
            "group": group_focus,
        },
    )

    log_lines = st.session_state.get("f1_runtime_log", [])
    if log_lines:
        with st.expander("Validation log", expanded=False):
            st.code("\n".join(str(line) for line in log_lines))


def render_f1_score(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
    output_dir_default: str,
) -> None:
    """Render the generic validation workflow or deferred paper panels."""
    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.error("Select an existing unified CSV before configuring F1 figures.")
        return

    columns = [str(column) for column in available_columns]
    reference_options = reference_candidates(columns)
    method_options = prediction_candidates(columns)
    group_options = group_candidates(columns)
    st.markdown("## Common prediction-data zone")
    st.caption(
        "Choose the prediction source once. The same choice feeds Option 1 and "
        "Option 2; switching figure workflows never changes or reruns the source data."
    )
    source_key = "viz_f1_shared_prediction_source"
    source_options = [
        "Use existing prediction columns from the unified CSV",
        "Use shared held-out ML predictions",
    ]
    if source_key not in st.session_state:
        cached_source = cached_params.get(source_key, source_options[0])
        st.session_state[source_key] = (
            cached_source if cached_source in source_options else source_options[0]
        )
    prediction_source = st.radio(
        "Prediction data used by both figure options",
        source_options,
        horizontal=True,
        key=source_key,
        help=(
            "Existing columns performs no model fitting. Shared held-out ML lets "
            "you train or load MR-ML and/or Consensus-ML with complete groups "
            "excluded, then reuse those saved predictions everywhere below."
        ),
    )
    cached_params[source_key] = prediction_source
    use_shared_predictions = prediction_source == source_options[1]
    shared_settings: Mapping[str, Any] | None = None
    shared_analysis: Mapping[str, Any] | None = None
    source_references: list[str] = []
    source_methods: list[str] = []
    source_group_column = ""
    if use_shared_predictions:
        shared_settings, shared_analysis = render_shared_ml_prediction_zone(
            project_root,
            cached_params,
            output_dir_default,
            str(unified_csv_path),
            columns,
            reference_options,
            method_options,
            group_options,
        )
        source_references = [
            str(column)
            for column in st.session_state.get(
                "viz_f1_shared_reference_columns", reference_options
            )
            if str(column) in columns
        ]
        source_methods = [
            str(column)
            for column in (shared_settings or {}).get("component_methods", [])
            if str(column) in columns
        ]
        source_group_column = str(
            (shared_settings or {}).get("validation_group_column", "")
        )
    else:
        st.info(
            "No ML will run. In either figure option, select the existing phenotype "
            "or prediction columns you want to compare."
        )
        existing_reference_key = "viz_f1_shared_existing_reference_columns"
        existing_method_key = "viz_f1_shared_existing_method_columns"
        existing_group_key = "viz_f1_shared_existing_group_column"
        for key, defaults in (
            (existing_reference_key, reference_options),
            (existing_method_key, method_options),
        ):
            if key not in st.session_state:
                cached_values = cached_params.get(key, defaults)
                values = cached_values if isinstance(cached_values, list) else defaults
                st.session_state[key] = [
                    str(value) for value in values if str(value) in columns
                ]
            else:
                st.session_state[key] = [
                    str(value)
                    for value in st.session_state.get(key, [])
                    if str(value) in columns
                ]
        optional_groups = ["(None)", *columns]
        if existing_group_key not in st.session_state:
            default_existing_group = group_options[0] if group_options else "(None)"
            cached_group = str(
                cached_params.get(existing_group_key, default_existing_group)
            )
            st.session_state[existing_group_key] = (
                cached_group if cached_group in optional_groups else "(None)"
            )
        elif st.session_state.get(existing_group_key) not in optional_groups:
            st.session_state[existing_group_key] = "(None)"
        with st.expander("Select existing columns used by both figure options", expanded=True):
            existing_columns = st.columns(3)
            with existing_columns[0]:
                source_references = st.multiselect(
                    "Existing manual-reference columns",
                    columns,
                    key=existing_reference_key,
                    help="Manual ground-truth columns scored by both Option 1 and Option 2.",
                )
            with existing_columns[1]:
                source_methods = st.multiselect(
                    "Existing prediction/phenotype columns",
                    columns,
                    key=existing_method_key,
                    help="Previously generated phenotype-call columns to use without fitting new ML models.",
                )
            with existing_columns[2]:
                selected_existing_group = st.selectbox(
                    "Existing grouping column (optional)",
                    optional_groups,
                    key=existing_group_key,
                    help="Optional patient, slide, image, or batch column used for group-wise figures; it does not retrain anything.",
                )
                source_group_column = (
                    "" if selected_existing_group == "(None)" else selected_existing_group
                )
        if not source_references:
            st.warning("Select at least one existing manual-reference column.")
        if not source_methods:
            st.warning("Select at least one existing prediction/phenotype column.")

    source_token = json.dumps(
        {
            "mode": "shared" if use_shared_predictions else "existing",
            "references": source_references,
            "methods": source_methods,
            "group": source_group_column,
        },
        sort_keys=True,
    )
    if st.session_state.get("_viz_f1_paper_source_sync_token") != source_token:
        st.session_state["viz_f1_paper_reference_columns"] = list(source_references)
        st.session_state["viz_f1_paper_method_columns"] = list(source_methods)
        st.session_state["viz_f1_paper_component_methods"] = list(source_methods)
        st.session_state["viz_f1_paper_group_column"] = (
            source_group_column or "(None)"
        )
        st.session_state["_viz_f1_paper_source_sync_token"] = source_token

    option_key = "viz_f1_figure_option"
    options = [
        "Option 1 — General F1 validation figures",
        "Option 2 — Customizable paper-style figures",
    ]
    cached_option = st.session_state.get(
        option_key, cached_params.get(option_key, options[0])
    )
    if str(cached_option).startswith("Option 2"):
        cached_option = options[1]
    if cached_option not in options:
        cached_option = options[0]
    if st.session_state.get(option_key) != cached_option:
        st.session_state[option_key] = cached_option
    selected = st.radio(
        "F1 figure workflow",
        options,
        horizontal=True,
        key=option_key,
        help=(
            "Option 1 prepares the general validation summaries. Option 2 prepares "
            "the reference-style customizable panels. Both consume the prediction "
            "source selected in the common zone above."
        ),
    )
    cached_params[option_key] = selected
    if selected == options[0]:
        _render_option1_figures(
            project_root,
            unified_csv_path,
            available_columns,
            cached_params,
            output_dir_default,
            use_shared_predictions=use_shared_predictions,
            shared_analysis=shared_analysis,
            shared_strict_settings=shared_settings,
        )
    else:
        render_paper_figure_panels(
            project_root,
            cached_params,
            output_dir_default,
            unified_csv_path,
            available_columns,
            use_shared_predictions=use_shared_predictions,
            shared_analysis=shared_analysis,
            shared_strict_settings=shared_settings,
        )
