"""Offline, customizable paper-style rendering for the F1 Score tab.

The plotting functions mirror the reference Figure 3 layouts while accepting
universal column mappings. Publication models are fitted by the shared strict
analysis engine, cached once, and reused here; figures render only when their
individual Generate button is pressed. Legacy source-table builders remain
available for reproducibility tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from spacemap_gui.ui_utils import clamp_numeric_widget_state
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .common import log_console, save_cached_params, timestamp
from .f1_paper_analysis import (
    TABLE_FILES,
    cached_publication_analysis,
    load_publication_analysis,
    publication_analysis_fingerprint,
    run_publication_analysis,
    save_publication_analysis,
)
from .f1_strict_validation_ui import render_strict_retraining_panel
from .f1_validation_utils import sanitize_filename


METHOD_ORDER = ["Leiden", "SOM", "Vortex", "RESOLVE", "ML-MR", "Consensus-ML"]
METHOD_COLORS = {
    "Leiden": "#1f77b4",
    "SOM": "#d62728",
    "Vortex": "#9467bd",
    "RESOLVE": "#2ca02c",
    "ML-MR": "#7B2CBF",
    "Consensus-ML": "#008C8C",
}
METHOD_LABELS = {method: method for method in METHOD_ORDER}
REFERENCE_ORDER = ["phenotype_ref1", "phenotype_ref2"]
REFERENCE_LABELS = {
    "phenotype_ref1": "Ref1: expert-selected cells",
    "phenotype_ref2": "Ref2: exhaustive difficult Tissue ID",
    "phenotype_bd": "Annotator BD",
    "phenotype_ap": "Annotator AP",
}
REFERENCE_COLORS = {
    "phenotype_ref1": "#1f77b4",
    "phenotype_ref2": "#d95f02",
    "phenotype_bd": "#008C8C",
    "phenotype_ap": "#9C755F",
}
HELDOUT_IMAGE_ORDER = ["P11", "P2_P10", "P28"]
HELDOUT_IMAGE_LABELS = {"P11": "P11", "P2_P10": "P2/P10", "P28": "P28"}
HELDOUT_IMAGE_COLORS = {
    "P11": "#C49A2C",
    "P2_P10": "#FFE53B",
    "P28": "#59C83A",
}
PHENOTYPE_ORDER = [
    "B",
    "CD4",
    "CD8",
    "Treg",
    "DC",
    "Macrophage",
    "IMC",
    "Endothelial",
    "Epithelial",
    "Fibroblast",
    "Plasma",
    "Others",
]
PHENOTYPE_COLORS = {
    "B": "#24306e",
    "CD4": "#e78ac3",
    "CD8": "#e31a1c",
    "Treg": "#ff9800",
    "DC": "#2ca02c",
    "Macrophage": "#9ecae1",
    "IMC": "#348abd",
    "Endothelial": "#006d6f",
    "Epithelial": "#7b3294",
    "Fibroblast": "#737373",
    "Plasma": "#b65cff",
    "Others": "#d9d9d9",
}
STABLE_FALLBACK_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

# ``fig3_code_v2.py`` is the visual source of truth for Option 2.  Keep these
# values centralized so the UI starts from the manuscript rendering while
# still exposing every value through editable controls.
PUBLICATION_BACKGROUND = "#FFFFFF"
PUBLICATION_FOREGROUND = "#222222"
PUBLICATION_DARK_FOREGROUND = "#F5F5F5"
OPTION2_VISUAL_SCHEMA_VERSION = 2

REFERENCE_PANEL_DIMENSIONS: dict[str, tuple[float, float]] = {
    "main_workflow": (14.2, 4.8),
    "main_heldout_macro": (10.0, 6.0),
    "main_consensus_tradeoff": (8.2, 5.1),
    "main_ref2_metrics": (10.0, 5.1),
    "main_per_phenotype_f1": (13.0, 8.2),
    "main_spatial": (12.2, 4.7),
    "supp_reference_distribution": (15.8, 6.3),
    "supp_weights": (7.3, 6.5),
    "supp_ref1_f1": (7.8, 9.2),
    "supp_ref2_f1": (7.8, 9.2),
    "supp_leave_one_out": (12.2, 4.8),
    "supp_routes": (11.7, 4.9),
    "supp_human": (11.5, 5.2),
    "supp_spatial": (12.2, 4.7),
    "supp_confidence_all": (12.0, 4.8),
    "supp_nonconsensus_threshold": (8.6, 5.1),
    "supp_confusion": (12.0, 5.2),
    "supp_ref1_precision": (7.8, 9.2),
    "supp_ref2_precision": (7.8, 9.2),
    "supp_ref1_recall": (7.8, 9.2),
    "supp_ref2_recall": (7.8, 9.2),
    "supp_radar": (13.6, 6.5),
    "supp_heldout_variation": (13.5, 5.8),
    "supp_learning": (8.1, 5.1),
    "supp_bins_ref2": (10.4, 5.3),
    "supp_bins_ref1": (10.4, 5.3),
    "supp_pairwise": (6.8, 5.8),
}

def _stable_named_color(
    value: Any,
    palette: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the same fallback color for the same semantic name everywhere."""
    colors = list(palette or STABLE_FALLBACK_COLORS)
    normalized = str(value).strip().casefold() or "unnamed"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return str(colors[int(digest[:12], 16) % len(colors)])


def _phenotype_default_color(phenotype: str) -> str:
    normalized = str(phenotype).strip().casefold()
    known = {str(name).casefold(): color for name, color in PHENOTYPE_COLORS.items()}
    return str(known.get(normalized, _stable_named_color(normalized)))
INVALID_LABELS = {"", "nan", "none", "unknown", "unclassified", "others"}


@dataclass(frozen=True)
class PaperPanelSpec:
    """Metadata used by the independent Streamlit panel forms."""

    panel_id: str
    section: str
    label: str
    default_stem: str
    default_title: str
    default_panel_label: str
    required_tables: tuple[str, ...] = ()
    has_legend: bool = True
    method_axis: bool = False
    heatmap: bool = False
    bar_grouping: bool = False


PAPER_PANEL_SPECS = [
    PaperPanelSpec(
        "main_workflow",
        "Main Figure 3",
        "Fig. 3A — Two ML workflows",
        "Fig3A_two_ML_workflows",
        "Two routes to refine phenotyping, tested on unseen images",
        "A",
        has_legend=False,
    ),
    PaperPanelSpec(
        "main_heldout_macro",
        "Main Figure 3",
        "Fig. 3B — Held-out-image macro-F1",
        "Fig3B_leave_one_image_out_macro_f1",
        "Ensemble performance is evaluated on one completely held-out image at a time",
        "B",
        ("T10_performance_summary.csv",),
        method_axis=True,
    ),
    PaperPanelSpec(
        "main_consensus_tradeoff",
        "Main Figure 3",
        "Fig. 3C — Consensus quality–coverage",
        "Fig3C_consensus_quality_coverage",
        "Three-of-four agreement balances label quality and coverage",
        "C",
        ("T05_consensus_and_leave_one_method_out.csv",),
    ),
    PaperPanelSpec(
        "main_ref2_metrics",
        "Main Figure 3",
        "Fig. 3D — Ref2 F1 comparison",
        "Fig3D_ref2_macro_micro_weighted_f1",
        "Ensembles are stress-tested in one difficult Tissue ID",
        "D",
        ("T10_performance_summary.csv",),
        method_axis=True,
        bar_grouping=True,
    ),
    PaperPanelSpec(
        "main_per_phenotype_f1",
        "Main Figure 3",
        "Fig. 3E — Per-phenotype F1",
        "Fig3E_per_phenotype_f1",
        "Pooled averages conceal larger phenotype-specific differences",
        "E",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "main_spatial",
        "Main Figure 3",
        "Fig. 3F — Ref2 spatial validation",
        "Fig3F_ref2_spatial_validation",
        "Spatial validation shows where predictions succeed and fail",
        "F",
        ("T06_strict_oof_predictions.csv",),
    ),
    PaperPanelSpec(
        "supp_reference_distribution",
        "Supplementary Figure 3",
        "Supp. 3.1 — Reference distribution",
        "Supp_Fig3_1_reference_distribution",
        "Manual-reference distribution across image ROIs",
        "",
        ("T03_reference_phenotype_distribution.csv", "T04_reference_locations.csv"),
        bar_grouping=True,
    ),
    PaperPanelSpec(
        "supp_weights",
        "Supplementary Figure 3",
        "Supp. 3.2 — Phenotype-specific weights",
        "Supp_Fig3_2_phenotype_specific_weights",
        "The most reliable component differs by phenotype",
        "",
        ("T08_fold_specific_f1_weights.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_ref1_f1",
        "Supplementary Figure 3",
        "Supp. 3.3 — Ref1 per-phenotype F1",
        "Supp_Fig3_3_ref1_per_class_f1",
        "Per-phenotype performance — Ref1: expert-selected cells",
        "",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_ref2_f1",
        "Supplementary Figure 3",
        "Supp. 3.4 — Ref2 per-phenotype F1",
        "Supp_Fig3_4_ref2_per_class_f1",
        "Per-phenotype performance — Ref2: exhaustive difficult Tissue ID",
        "",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_leave_one_out",
        "Supplementary Figure 3",
        "Supp. 3.5 — Leave-one-method-out sensitivity",
        "Supp_Fig3_5_leave_one_method_out",
        "Removing one method changes the quality–coverage balance",
        "",
        ("T05_consensus_and_leave_one_method_out.csv",),
    ),
    PaperPanelSpec(
        "supp_routes",
        "Supplementary Figure 3",
        "Supp. 3.6 — Direct versus non-consensus routes",
        "Supp_Fig3_6_direct_vs_nonconsensus",
        "Direct-consensus and ML-predicted cells have different difficulty",
        "",
        ("T12_direct_vs_nonconsensus_routes.csv",),
        bar_grouping=True,
    ),
    PaperPanelSpec(
        "supp_human",
        "Supplementary Figure 3",
        "Supp. 3.7 — Human variation",
        "Supp_Fig3_7_human_variation",
        "Manual phenotyping also varies on difficult cells",
        "",
        (
            "T17_human_agreement_summary.csv",
            "T18_human_confusion_counts.csv",
            "T19_human_consensus_enrichment.csv",
        ),
    ),
    PaperPanelSpec(
        "supp_spatial",
        "Supplementary Figure 3",
        "Supp. 3.8 — Ref2 spatial validation",
        "Supp_Fig3_8_ref2_spatial_validation",
        "Spatial validation shows where predictions succeed and fail",
        "S8",
        ("T06_strict_oof_predictions.csv",),
    ),
    PaperPanelSpec(
        "supp_confidence_all",
        "Supplementary Figure 3",
        "Supp. 3.9 — Confidence across all routes",
        "Supp_Fig3_9_confidence_all_routes",
        "Confidence thresholds trade retained coverage for performance",
        "",
        ("T14_confidence_all_routes.csv",),
    ),
    PaperPanelSpec(
        "supp_nonconsensus_threshold",
        "Supplementary Figure 3",
        "Supp. 3.9b — Non-consensus threshold tradeoff",
        "Supp_Fig3_9b_nonconsensus_threshold_tradeoff",
        "Confidence filtering trades retained coverage for performance",
        "",
        ("T15_confidence_nonconsensus_only.csv",),
    ),
    PaperPanelSpec(
        "supp_confusion",
        "Supplementary Figure 3",
        "Supp. 3.10 — Ref2 confusion matrices",
        "Supp_Fig3_10_ref2_confusion_matrices",
        "Normalized confusion matrices — Ref2",
        "",
        ("T13_confusion_counts.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_ref1_precision",
        "Supplementary Figure 3",
        "Supp. 3.11 — Ref1 per-phenotype precision",
        "Supp_Fig3_11_ref1_per_class_precision",
        "Per-phenotype precision — Ref1: expert-selected cells",
        "",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_ref2_precision",
        "Supplementary Figure 3",
        "Supp. 3.12 — Ref2 per-phenotype precision",
        "Supp_Fig3_12_ref2_per_class_precision",
        "Per-phenotype precision — Ref2: exhaustive difficult Tissue ID",
        "",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_ref1_recall",
        "Supplementary Figure 3",
        "Supp. 3.13 — Ref1 per-phenotype recall",
        "Supp_Fig3_13_ref1_per_class_recall",
        "Per-phenotype recall — Ref1: expert-selected cells",
        "",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_ref2_recall",
        "Supplementary Figure 3",
        "Supp. 3.14 — Ref2 per-phenotype recall",
        "Supp_Fig3_14_ref2_per_class_recall",
        "Per-phenotype recall — Ref2: exhaustive difficult Tissue ID",
        "",
        ("T11_per_class_performance.csv",),
        has_legend=False,
        heatmap=True,
    ),
    PaperPanelSpec(
        "supp_radar",
        "Supplementary Figure 3",
        "Supp. 3.15 — Multiple-performance radar",
        "Supp_Fig3_15_multiple_performance_comparison",
        "Multiple performance measures show method- and reference-dependent differences",
        "",
        ("T21_multiple_performance_comparison.csv",),
    ),
    PaperPanelSpec(
        "supp_heldout_variation",
        "Supplementary Figure 3",
        "Supp. 3.16 — Held-out-image variation",
        "Supp_Fig3_16_heldout_image_metric_variation",
        "Pooled performance masks image-to-image variation",
        "",
        ("T10_performance_summary.csv",),
        method_axis=True,
    ),
    PaperPanelSpec(
        "supp_learning",
        "Supplementary Figure 3",
        "Supp. 3.17 — Expert-annotation learning curve",
        "Supp_Fig3_17_manual_annotation_curve",
        "Annotation-learning curve for this dataset",
        "",
        ("T16_annotation_learning_curve.csv", "T10_performance_summary.csv"),
    ),
    PaperPanelSpec(
        "supp_bins_ref2",
        "Supplementary Figure 3",
        "Supp. 3.18 — Ref2 confidence bins",
        "Supp_Fig3_18_nonconsensus_confidence_bins",
        "Ref2 non-consensus cells across Consensus-ML confidence bins",
        "",
        ("T15b_confidence_bins_same_cells.csv",),
    ),
    PaperPanelSpec(
        "supp_bins_ref1",
        "Supplementary Figure 3",
        "Supp. 3.19 — Ref1 confidence bins",
        "Supp_Fig3_19_ref1_nonconsensus_confidence_bins",
        "Ref1 non-consensus cells across Consensus-ML confidence bins",
        "",
        ("T15b_confidence_bins_same_cells.csv",),
    ),
    PaperPanelSpec(
        "supp_pairwise",
        "Supplementary Figure 3",
        "Supp. 3.20 — Ref2 pairwise component agreement",
        "Supp_Fig3_20_Ref2_pairwise_component_agreement",
        "Pairwise similarity among four phenotyping methods",
        "",
        ("T23_Ref2_pairwise_component_agreement_matrix.csv",),
        has_legend=False,
        heatmap=True,
    ),
]
PANEL_BY_ID = {spec.panel_id: spec for spec in PAPER_PANEL_SPECS}

# A universal UI presents each visual once by chart purpose.  Some publication
# outputs differed only by reference/metric and are now controlled inside the
# same independent panel.
GENERIC_PANEL_IDS = [
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
    "supp_human",
    "supp_confidence_all",
    "supp_nonconsensus_threshold",
    "supp_confusion",
    "supp_radar",
    "supp_heldout_variation",
    "supp_learning",
    "supp_bins_ref2",
    "supp_pairwise",
]
GENERIC_PANEL_LABELS = {
    "main_workflow": "Workflow diagram",
    "main_heldout_macro": "Group-wise macro-F1 trajectories",
    "main_consensus_tradeoff": "Consensus quality–coverage curve",
    "main_ref2_metrics": "Grouped performance-metric bars",
    "main_per_phenotype_f1": "Multi-reference per-class heatmap",
    "main_spatial": "Spatial match/mismatch validation",
    "supp_reference_distribution": "Reference distribution and composition",
    "supp_weights": "Phenotype-specific method-reliability heatmap",
    "supp_ref1_f1": "Single-reference per-class metric heatmap",
    "supp_leave_one_out": "Leave-one-method-out consensus sensitivity",
    "supp_routes": "Direct-consensus versus resolved-route comparison",
    "supp_human": "Manual-reference agreement comparison",
    "supp_confidence_all": "Confidence versus retained-coverage curves",
    "supp_nonconsensus_threshold": "Non-consensus confidence tradeoff",
    "supp_confusion": "Normalized confusion matrices",
    "supp_radar": "Multi-metric radar comparison",
    "supp_heldout_variation": "Group-to-group performance variation",
    "supp_learning": "Training-sample learning curve",
    "supp_bins_ref2": "Confidence-bin performance comparison",
    "supp_pairwise": "Pairwise assignment agreement heatmap",
}
GENERIC_PANEL_DESCRIPTIONS = {
    "main_workflow": "Summarizes the selected phenotyping methods, reference sources, and validation design as an editable workflow diagram.",
    "main_heldout_macro": "Shows how each method's macro-F1 changes across the selected patient, slide, batch, or ROI grouping, with the pooled score overlaid.",
    "main_consensus_tradeoff": "Tests how requiring stronger agreement among selected component methods changes direct-label coverage and macro-F1.",
    "main_ref2_metrics": "Compares pooled macro-, micro-, and weighted-F1 for the selected methods against one chosen manual reference.",
    "main_per_phenotype_f1": "Displays phenotype-level F1 across all selected manual references so pooled performance does not hide class-specific differences.",
    "main_spatial": "Maps manual phenotypes and cell-level matches or mismatches using the selected X/Y coordinate columns.",
    "supp_reference_distribution": "Audits where manual labels occur across groups and how the reference phenotype composition is distributed.",
    "supp_weights": "Uses phenotype-level F1 across groups as a generic reliability weight for each selected component method.",
    "supp_ref1_f1": "Shows precision, recall, or F1 by phenotype for one selectable manual reference.",
    "supp_leave_one_out": "Recalculates the consensus quality–coverage relationship after removing each selected component method in turn.",
    "supp_routes": "Compares directly agreed cells with cells assigned by the selected primary and secondary resolution methods.",
    "supp_human": "Compares two selected manual-reference columns and relates their agreement to algorithmic consensus strength.",
    "supp_confidence_all": "Shows the tradeoff between confidence thresholds, accepted-cell coverage, and macro-F1 for methods with mapped confidence columns.",
    "supp_nonconsensus_threshold": "Focuses confidence filtering on cells that do not meet the selected component-method consensus rule.",
    "supp_confusion": "Shows row-normalized phenotype confusion for one reference and up to two selected methods.",
    "supp_radar": "Compares accuracy, macro precision, macro recall, macro-F1, and Cohen's kappa on a common scale.",
    "supp_heldout_variation": "Shows complementary F1 measures across the selected grouping variable to reveal between-group variation.",
    "supp_learning": "Fits balanced training-size experiments with complete groups held out, then plots the cached repeated-learning curve beside full-data OOF performance.",
    "supp_bins_ref2": "Compares all selected methods on the same cells within bins of one mapped confidence score.",
    "supp_pairwise": "Measures exact phenotype-assignment agreement between every selected component-method pair.",
}


LEGEND_LOCATIONS = [
    "Outside right",
    "Outside right (lower)",
    "Inside upper right",
    "Inside upper left",
    "Inside lower right",
    "Inside lower left",
    "Above",
    "Below",
    "Best",
    "Hidden",
]
HEATMAP_CMAPS = [
    "Blues",
    "Reds",
    "Greens",
    "YlGnBu",
    "viridis",
    "mako",
    "rocket",
    "crest",
    "vlag",
]
FONT_FAMILIES = ["DejaVu Sans", "Liberation Sans", "Arial", "DejaVu Serif"]


def _source_data_dir(value: str | Path) -> Path:
    root = Path(value).expanduser()
    return root / "source_data" if (root / "source_data").is_dir() else root


def missing_panel_tables(panel_id: str, source_root: str | Path) -> list[Path]:
    """Return missing source tables for a paper panel."""
    spec = PANEL_BY_ID[panel_id]
    source = _source_data_dir(source_root)
    return [
        source / name for name in spec.required_tables if not (source / name).is_file()
    ]


def load_panel_tables(
    panel_id: str, source_root: str | Path
) -> dict[str, pd.DataFrame]:
    """Load only the source tables needed by one panel."""
    missing = missing_panel_tables(panel_id, source_root)
    if missing:
        raise FileNotFoundError(
            "Missing source table(s): " + ", ".join(str(path) for path in missing)
        )
    source = _source_data_dir(source_root)
    return {
        name: pd.read_csv(source / name, low_memory=False)
        for name in PANEL_BY_ID[panel_id].required_tables
    }


def _valid_generic_labels(series: pd.Series, excluded: set[str]) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return (
        series.notna() & cleaned.ne("") & ~cleaned.str.lower().isin(excluded)
    ).fillna(False)


def _generic_predictions(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = series.astype("string").str.strip()
    assigned = (
        series.notna() & cleaned.ne("") & ~cleaned.str.lower().isin(INVALID_LABELS)
    ).fillna(False)
    return cleaned.where(assigned, "Unresolved").astype(str), assigned


def _generic_scores(
    truth: pd.Series,
    prediction: pd.Series,
    assigned: pd.Series,
) -> dict[str, float | int]:
    labels = _ordered_phenotypes(truth.astype(str).unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "n": int(len(truth)),
            "accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "micro_f1": float(
                f1_score(
                    truth, prediction, labels=labels, average="micro", zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(
                    truth, prediction, labels=labels, average="macro", zero_division=0
                )
            ),
            "weighted_f1": float(
                f1_score(
                    truth,
                    prediction,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "macro_precision": float(
                precision_score(
                    truth, prediction, labels=labels, average="macro", zero_division=0
                )
            ),
            "macro_recall": float(
                recall_score(
                    truth, prediction, labels=labels, average="macro", zero_division=0
                )
            ),
            "cohen_kappa": float(cohen_kappa_score(truth, prediction)),
            "prediction_coverage": float(assigned.mean()) if len(assigned) else 0.0,
        }


def _generic_per_class_rows(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    reference: str,
    fold: str,
    method: str,
) -> list[dict[str, Any]]:
    labels = _ordered_phenotypes(truth.astype(str).unique())
    precision, recall, f1_values, support = precision_recall_fscore_support(
        truth,
        prediction,
        labels=labels,
        zero_division=0,
    )
    return [
        {
            "reference": reference,
            "fold": fold,
            "method": method,
            "phenotype": phenotype,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1_values[index]),
            "support": int(support[index]),
        }
        for index, phenotype in enumerate(labels)
    ]


def _selected_groups(frame: pd.DataFrame, common: Mapping[str, Any]) -> list[str]:
    group_column = str(common.get("group_column") or "")
    if not group_column or group_column not in frame.columns:
        return []
    configured = [str(value) for value in common.get("group_order", [])]
    available = list(dict.fromkeys(frame[group_column].dropna().astype(str)))
    selected = [value for value in configured if value in available]
    return selected or available


def _build_generic_performance_tables(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
    *,
    include_groups: bool,
) -> dict[str, pd.DataFrame]:
    references = [
        str(value)
        for value in common.get("reference_order", [])
        if str(value) in frame.columns
    ]
    methods = [
        str(value)
        for value in common.get("method_order", [])
        if str(value) in frame.columns
    ]
    excluded = {
        str(value).strip().lower()
        for value in common.get("excluded_labels", INVALID_LABELS)
    }
    group_column = str(common.get("group_column") or "")
    group_values = _selected_groups(frame, common) if include_groups else []
    folds: list[tuple[str, pd.Series]] = [
        ("Pooled", pd.Series(True, index=frame.index))
    ]
    if group_column and group_column in frame.columns:
        folds = [
            (group, frame[group_column].astype(str).eq(group)) for group in group_values
        ] + folds
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    multiple_rows: list[dict[str, Any]] = []
    for reference in references:
        reference_mask = _valid_generic_labels(frame[reference], excluded)
        for fold, fold_mask in folds:
            mask = reference_mask & fold_mask
            truth = frame.loc[mask, reference].astype(str).str.strip()
            if truth.empty:
                continue
            for method in methods:
                prediction, assigned = _generic_predictions(frame.loc[mask, method])
                scores = _generic_scores(truth, prediction, assigned)
                summary_rows.append(
                    {
                        "reference": reference,
                        "reference_name": _reference_label(common, reference),
                        "fold": fold,
                        "method_column": method,
                        "method": method,
                        **{
                            key: value
                            for key, value in scores.items()
                            if key
                            not in {"macro_precision", "macro_recall", "cohen_kappa"}
                        },
                    }
                )
                class_rows.extend(
                    _generic_per_class_rows(
                        truth,
                        prediction,
                        reference=reference,
                        fold=fold,
                        method=method,
                    )
                )
                if fold == "Pooled":
                    counts = (
                        pd.DataFrame({"truth": truth, "prediction": prediction})
                        .value_counts()
                        .rename("n")
                        .reset_index()
                    )
                    counts.insert(0, "method", method)
                    counts.insert(0, "reference", reference)
                    confusion_rows.extend(counts.to_dict("records"))
                    multiple_rows.append(
                        {
                            "reference": reference,
                            "reference_name": _reference_label(common, reference),
                            "method": method,
                            **scores,
                        }
                    )
    return {
        "T10_performance_summary.csv": pd.DataFrame(summary_rows),
        "T11_per_class_performance.csv": pd.DataFrame(class_rows),
        "T13_confusion_counts.csv": pd.DataFrame(confusion_rows),
        "T21_multiple_performance_comparison.csv": pd.DataFrame(multiple_rows),
    }


def _consensus_calls(
    frame: pd.DataFrame,
    components: list[str],
    minimum_agreement: int,
) -> pd.Series:
    if not components:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    values = frame[components].astype("string")

    def choose(row: pd.Series) -> Any:
        cleaned = row.dropna().astype(str).str.strip()
        cleaned = cleaned.loc[~cleaned.str.lower().isin(INVALID_LABELS)]
        if cleaned.empty:
            return pd.NA
        counts = cleaned.value_counts()
        return (
            str(counts.index[0]) if int(counts.iloc[0]) >= minimum_agreement else pd.NA
        )

    return values.apply(choose, axis=1).astype("string")


def _build_generic_consensus_tables(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    references = [
        str(value)
        for value in common.get("reference_order", [])
        if str(value) in frame.columns
    ]
    methods = [
        str(value)
        for value in common.get("method_order", [])
        if str(value) in frame.columns
    ]
    components = [
        str(value)
        for value in common.get("component_methods", [])
        if str(value) in methods
    ]
    excluded = {
        str(value).strip().lower()
        for value in common.get("excluded_labels", INVALID_LABELS)
    }
    scenario_rows: list[dict[str, Any]] = []
    scenario_sets: list[tuple[str | None, list[str]]] = [(None, components)]
    scenario_sets += [
        (component, [value for value in components if value != component])
        for component in components
    ]
    for dropped, selected_components in scenario_sets:
        n_methods = len(selected_components)
        if n_methods < 2:
            continue
        for minimum in range(2, n_methods + 1):
            calls = _consensus_calls(frame, selected_components, minimum)
            all_direct = calls.notna()
            for reference in references:
                reference_mask = _valid_generic_labels(frame[reference], excluded)
                covered = reference_mask & all_direct
                if not covered.any():
                    continue
                truth = frame.loc[covered, reference].astype(str).str.strip()
                prediction = calls.loc[covered].astype(str)
                scores = _generic_scores(
                    truth, prediction, pd.Series(True, index=truth.index)
                )
                scenario_rows.append(
                    {
                        "methods": ", ".join(selected_components),
                        "n_methods": n_methods,
                        "min_agree": minimum,
                        "dropped_method": dropped or "None",
                        "all_cells": int(len(frame)),
                        "all_direct_consensus": int(all_direct.sum()),
                        "all_cell_coverage": float(all_direct.mean()),
                        "reference": reference,
                        "reference_name": _reference_label(common, reference),
                        "reference_cells": int(reference_mask.sum()),
                        "reference_cells_covered": int(covered.sum()),
                        "reference_coverage": float(
                            covered.sum() / max(1, reference_mask.sum())
                        ),
                        **{
                            key: value
                            for key, value in scores.items()
                            if key
                            not in {"macro_precision", "macro_recall", "cohen_kappa"}
                        },
                    }
                )

    minimum = int(common.get("consensus_min_agree", max(2, len(components) - 1)))
    minimum = min(max(1, minimum), max(1, len(components)))
    direct_calls = _consensus_calls(frame, components, minimum)
    direct_mask = direct_calls.notna()
    primary_method = str(
        common.get("primary_method") or (methods[0] if methods else "")
    )
    secondary_method = str(
        common.get("secondary_method")
        or (methods[1] if len(methods) > 1 else primary_method)
    )
    route_rows: list[dict[str, Any]] = []
    for reference in references:
        reference_mask = _valid_generic_labels(frame[reference], excluded)
        for route, route_mask, route_methods in [
            ("Direct consensus", direct_mask, [("Direct consensus", direct_calls)]),
            (
                "Non-consensus",
                ~direct_mask,
                [
                    (method, frame[method])
                    for method in [primary_method, secondary_method]
                    if method in frame.columns
                ],
            ),
        ]:
            mask = reference_mask & route_mask
            if not mask.any():
                continue
            truth = frame.loc[mask, reference].astype(str).str.strip()
            for method, predictions in route_methods:
                prediction, assigned = _generic_predictions(predictions.loc[mask])
                scores = _generic_scores(truth, prediction, assigned)
                route_rows.append(
                    {
                        "reference": reference,
                        "reference_name": _reference_label(common, reference),
                        "route": route,
                        "method": method,
                        **{
                            key: value
                            for key, value in scores.items()
                            if key
                            not in {"macro_precision", "macro_recall", "cohen_kappa"}
                        },
                    }
                )
    return {
        "T05_consensus_and_leave_one_method_out.csv": pd.DataFrame(scenario_rows),
        "T12_direct_vs_nonconsensus_routes.csv": pd.DataFrame(route_rows),
        "_direct_consensus_calls": direct_calls.to_frame("direct_consensus"),
    }


def _build_generic_reference_tables(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    references = [
        str(value)
        for value in common.get("reference_order", [])
        if str(value) in frame.columns
    ]
    excluded = {
        str(value).strip().lower()
        for value in common.get("excluded_labels", INVALID_LABELS)
    }
    group_column = str(common.get("group_column") or "")
    groups = _selected_groups(frame, common)
    if not groups:
        groups = ["All cells"]
    location_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for reference in references:
        reference_mask = _valid_generic_labels(frame[reference], excluded)
        values = (
            frame.loc[reference_mask, reference].astype(str).str.strip().value_counts()
        )
        total = int(values.sum())
        for phenotype, count in values.items():
            distribution_rows.append(
                {
                    "reference": reference,
                    "phenotype": phenotype,
                    "cells": int(count),
                    "fraction_of_annotated": float(count / max(1, total)),
                }
            )
        for group in groups:
            group_mask = (
                frame[group_column].astype(str).eq(group)
                if group_column and group_column in frame.columns
                else pd.Series(True, index=frame.index)
            )
            location_rows.append(
                {
                    "reference": reference,
                    "slide_id": group,
                    "batch_id": group,
                    "tissue_roi": group,
                    "annotated_cells": int((reference_mask & group_mask).sum()),
                }
            )
    return {
        "T03_reference_phenotype_distribution.csv": pd.DataFrame(distribution_rows),
        "T04_reference_locations.csv": pd.DataFrame(location_rows),
    }


def _build_generic_confidence_tables(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
    direct_calls: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    references = [
        str(value)
        for value in common.get("reference_order", [])
        if str(value) in frame.columns
    ]
    methods = [
        str(value)
        for value in common.get("method_order", [])
        if str(value) in frame.columns
    ]
    confidence_columns = {
        str(method): str(column)
        for method, column in dict(common.get("confidence_columns", {})).items()
        if str(method) in methods and str(column) in frame.columns
    }
    thresholds = sorted(
        {
            float(value)
            for value in common.get(
                "confidence_thresholds", [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
            )
            if 0.0 <= float(value) <= 1.0
        }
    )
    excluded = {
        str(value).strip().lower()
        for value in common.get("excluded_labels", INVALID_LABELS)
    }
    all_rows: list[dict[str, Any]] = []
    nonconsensus_rows: list[dict[str, Any]] = []
    if direct_calls is None:
        direct_calls = pd.Series(pd.NA, index=frame.index, dtype="string")
    nonconsensus_mask = direct_calls.isna()
    for reference in references:
        reference_mask = _valid_generic_labels(frame[reference], excluded)
        for method, confidence_column in confidence_columns.items():
            confidence = pd.to_numeric(frame[confidence_column], errors="coerce")
            for route, base_mask in [
                ("End-to-end", reference_mask),
                ("Non-consensus", reference_mask & nonconsensus_mask),
            ]:
                base_n = int(base_mask.sum())
                for threshold in thresholds:
                    accepted_mask = base_mask & confidence.ge(threshold)
                    if not accepted_mask.any():
                        continue
                    truth = frame.loc[accepted_mask, reference].astype(str).str.strip()
                    prediction, assigned = _generic_predictions(
                        frame.loc[accepted_mask, method]
                    )
                    scores = _generic_scores(truth, prediction, assigned)
                    row = {
                        "reference": reference,
                        "reference_name": _reference_label(common, reference),
                        "route": route,
                        "method": method,
                        "threshold": threshold,
                        "reference_cells": base_n,
                        "nonconsensus_reference_cells": base_n,
                        "accepted_cells": int(accepted_mask.sum()),
                        "coverage": float(accepted_mask.sum() / max(1, base_n)),
                        "coverage_within_nonconsensus": float(
                            accepted_mask.sum() / max(1, base_n)
                        ),
                        **{
                            key: value
                            for key, value in scores.items()
                            if key
                            not in {"macro_precision", "macro_recall", "cohen_kappa"}
                        },
                    }
                    (all_rows if route == "End-to-end" else nonconsensus_rows).append(
                        row
                    )

    focus_method = str(common.get("confidence_focus_method") or "")
    if focus_method not in confidence_columns and confidence_columns:
        focus_method = next(iter(confidence_columns))
    binned_rows: list[dict[str, Any]] = []
    if focus_method in confidence_columns:
        confidence = pd.to_numeric(
            frame[confidence_columns[focus_method]], errors="coerce"
        )
        edges = np.linspace(0.0, 1.0, 11)
        bins = pd.cut(confidence, bins=edges, include_lowest=True, right=True)
        for reference in references:
            reference_mask = (
                _valid_generic_labels(frame[reference], excluded) & nonconsensus_mask
            )
            for interval in bins.dropna().cat.categories:
                mask = reference_mask & bins.eq(interval)
                if not mask.any():
                    continue
                lower = float(interval.left)
                upper = float(interval.right)
                midpoint = (lower + upper) / 2
                label = f"{max(0.0, lower):.1f}–{upper:.1f}"
                truth = frame.loc[mask, reference].astype(str).str.strip()
                for method in methods:
                    prediction, assigned = _generic_predictions(frame.loc[mask, method])
                    scores = _generic_scores(truth, prediction, assigned)
                    binned_rows.append(
                        {
                            "reference": reference,
                            "reference_name": _reference_label(common, reference),
                            "route": "Non-consensus",
                            "confidence_bin": label,
                            "confidence_lower": lower,
                            "confidence_upper": upper,
                            "confidence_midpoint": midpoint,
                            "method": method,
                            "cells_in_bin": int(mask.sum()),
                            **{
                                key: value
                                for key, value in scores.items()
                                if key
                                not in {
                                    "macro_precision",
                                    "macro_recall",
                                    "cohen_kappa",
                                }
                            },
                        }
                    )
    return {
        "T14_confidence_all_routes.csv": pd.DataFrame(all_rows),
        "T15_confidence_nonconsensus_only.csv": pd.DataFrame(nonconsensus_rows),
        "T15b_confidence_bins_same_cells.csv": pd.DataFrame(binned_rows),
    }


def _build_generic_human_tables(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
    direct_calls: pd.Series,
) -> dict[str, pd.DataFrame]:
    references = [
        str(value)
        for value in common.get("reference_order", [])
        if str(value) in frame.columns
    ]
    if len(references) < 2:
        return {
            "T17_human_agreement_summary.csv": pd.DataFrame(),
            "T18_human_confusion_counts.csv": pd.DataFrame(),
            "T19_human_consensus_enrichment.csv": pd.DataFrame(),
        }
    first, second = references[:2]
    excluded = {
        str(value).strip().lower()
        for value in common.get("excluded_labels", INVALID_LABELS)
    }
    overlap = _valid_generic_labels(frame[first], excluded) & _valid_generic_labels(
        frame[second], excluded
    )
    left = frame.loc[overlap, first].astype(str).str.strip()
    right = frame.loc[overlap, second].astype(str).str.strip()
    agreement = left.eq(right)
    matrix = pd.crosstab(left, right)
    summary = pd.DataFrame(
        [
            {
                "n_shared_cells": int(overlap.sum()),
                "agreement": float(agreement.mean()) if len(agreement) else np.nan,
                "cohen_kappa": float(cohen_kappa_score(left, right))
                if len(agreement)
                else np.nan,
                "symmetric_macro_f1": float(
                    (
                        f1_score(left, right, average="macro", zero_division=0)
                        + f1_score(right, left, average="macro", zero_division=0)
                    )
                    / 2
                )
                if len(agreement)
                else np.nan,
            }
        ]
    )
    enrichment_rows: list[dict[str, Any]] = []
    components = [
        str(value)
        for value in common.get("component_methods", [])
        if str(value) in frame.columns
    ]
    for minimum in range(2, len(components) + 1):
        consensus = _consensus_calls(frame, components, minimum).notna().loc[overlap]
        enrichment_rows.append(
            {
                "min_agree": minimum,
                "n_shared": int(overlap.sum()),
                "human_agreement_rate": float(agreement.mean())
                if len(agreement)
                else np.nan,
                "consensus_fraction_when_humans_agree": float(
                    consensus.loc[agreement].mean()
                )
                if agreement.any()
                else np.nan,
                "consensus_fraction_when_humans_disagree": float(
                    consensus.loc[~agreement].mean()
                )
                if (~agreement).any()
                else np.nan,
            }
        )
    return {
        "T17_human_agreement_summary.csv": summary,
        "T18_human_confusion_counts.csv": matrix.reset_index(),
        "T19_human_consensus_enrichment.csv": pd.DataFrame(enrichment_rows),
    }


def _build_generic_pairwise_table(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
) -> pd.DataFrame:
    methods = [
        str(value)
        for value in common.get("component_methods", [])
        if str(value) in frame.columns
    ]
    matrix = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    for first in methods:
        for second in methods:
            first_values = frame[first]
            second_values = frame[second]
            paired = first_values.notna() & second_values.notna()
            matrix.loc[first, second] = (
                100.0
                * first_values.loc[paired]
                .astype(str)
                .eq(second_values.loc[paired].astype(str))
                .mean()
                if paired.any()
                else np.nan
            )
    matrix.index.name = "Method"
    return matrix.reset_index()


def generic_panel_availability(
    panel_id: str,
    common: Mapping[str, Any],
) -> tuple[bool, str]:
    references = list(common.get("reference_order", []))
    methods = list(common.get("method_order", []))
    components = list(common.get("component_methods", []))
    confidence = dict(common.get("confidence_columns", {}))
    if panel_id == "main_workflow":
        return True, ""
    if not references:
        return False, "Select at least one manual-reference column in Shared Settings."
    if panel_id == "supp_reference_distribution":
        return True, ""
    if not methods:
        return (
            False,
            "Select at least one phenotyping/prediction column in Shared Settings.",
        )
    if panel_id in {"main_heldout_macro", "supp_heldout_variation"} and not common.get(
        "group_column"
    ):
        return (
            False,
            "Select a patient, slide, batch, or ROI grouping column for this figure.",
        )
    if (
        panel_id in {"main_consensus_tradeoff", "supp_leave_one_out", "supp_routes"}
        and len(components) < 2
    ):
        return False, "Select at least two component methods for consensus analysis."
    if panel_id == "supp_weights" and not components:
        return (
            False,
            "Select at least one component method for the reliability heatmap.",
        )
    if panel_id == "supp_human" and len(references) < 2:
        return (
            False,
            "This figure needs two manual-reference columns; one-reference datasets can use the other panels.",
        )
    if (
        panel_id in {"supp_human", "supp_nonconsensus_threshold", "supp_bins_ref2"}
        and len(components) < 2
    ):
        return (
            False,
            "Select at least two component methods to define consensus and non-consensus cells.",
        )
    if (
        panel_id
        in {"supp_confidence_all", "supp_nonconsensus_threshold", "supp_bins_ref2"}
        and not confidence
    ):
        return (
            False,
            "Map at least one method to a continuous confidence/probability column.",
        )
    if panel_id == "main_spatial" and not (
        common.get("x_column") and common.get("y_column")
    ):
        return False, "Select both X- and Y-coordinate columns for spatial validation."
    if panel_id == "supp_pairwise" and len(components) < 2:
        return False, "Select at least two component methods for pairwise agreement."
    if panel_id == "supp_learning":
        if bool(common.get("cached_learning_curve_available", False)):
            return True, ""
        learning_curve_csv = str(common.get("learning_curve_csv") or "").strip()
        if not learning_curve_csv:
            return (
                False,
                "Enable the learning-curve experiment above and rebuild the shared publication analysis.",
            )
        if not Path(learning_curve_csv).expanduser().is_file():
            return False, f"The learning-curve CSV does not exist: {learning_curve_csv}"
    return True, ""


def build_generic_panel_tables(
    frame: pd.DataFrame,
    common: Mapping[str, Any],
    panel_id: str,
    analysis_tables: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Derive only the tables required by one figure from a unified CSV."""
    available, reason = generic_panel_availability(panel_id, common)
    if not available:
        raise ValueError(reason)
    tables: dict[str, pd.DataFrame] = {}
    performance_panels = {
        "main_heldout_macro",
        "main_ref2_metrics",
        "main_per_phenotype_f1",
        "supp_weights",
        "supp_ref1_f1",
        "supp_confusion",
        "supp_radar",
        "supp_heldout_variation",
        "supp_learning",
    }
    if panel_id in performance_panels:
        tables.update(
            _build_generic_performance_tables(
                frame,
                common,
                include_groups=panel_id
                in {"main_heldout_macro", "supp_weights", "supp_heldout_variation"},
            )
        )
    consensus_panels = {
        "main_consensus_tradeoff",
        "supp_leave_one_out",
        "supp_routes",
        "supp_human",
        "supp_nonconsensus_threshold",
        "supp_bins_ref2",
    }
    direct_calls: pd.Series | None = None
    if panel_id in consensus_panels:
        consensus_tables = _build_generic_consensus_tables(frame, common)
        direct_frame = consensus_tables.pop("_direct_consensus_calls")
        direct_calls = direct_frame["direct_consensus"]
        tables.update(consensus_tables)
    if panel_id == "supp_reference_distribution":
        tables.update(_build_generic_reference_tables(frame, common))
    if panel_id in {
        "supp_confidence_all",
        "supp_nonconsensus_threshold",
        "supp_bins_ref2",
    }:
        tables.update(_build_generic_confidence_tables(frame, common, direct_calls))
    if panel_id == "supp_human":
        if direct_calls is None:
            direct_calls = pd.Series(pd.NA, index=frame.index, dtype="string")
        tables.update(_build_generic_human_tables(frame, common, direct_calls))
    if panel_id == "supp_pairwise":
        tables["T23_Ref2_pairwise_component_agreement_matrix.csv"] = (
            _build_generic_pairwise_table(frame, common)
        )
    if panel_id == "main_spatial":
        x_column = str(common["x_column"])
        y_column = str(common["y_column"])
        spatial = frame.copy()
        spatial["x_centroid"] = pd.to_numeric(spatial[x_column], errors="coerce")
        spatial["y_centroid"] = pd.to_numeric(spatial[y_column], errors="coerce")
        tables["T06_strict_oof_predictions.csv"] = spatial
    if panel_id == "supp_weights":
        cached_weights = (analysis_tables or {}).get("f1_weights")
        if isinstance(cached_weights, pd.DataFrame) and not cached_weights.empty:
            weights = cached_weights.copy()
        else:
            per_class = tables["T11_per_class_performance.csv"]
            component_methods = set(
                str(value) for value in common.get("component_methods", [])
            )
            weights = per_class.loc[
                per_class["method"].isin(component_methods)
                & per_class["fold"].ne("Pooled"),
                ["fold", "phenotype", "method", "f1"],
            ].rename(columns={"method": "component_method", "f1": "f1_weight"})
            if weights.empty:
                weights = per_class.loc[
                    per_class["method"].isin(component_methods)
                    & per_class["fold"].eq("Pooled"),
                    ["fold", "phenotype", "method", "f1"],
                ].rename(columns={"method": "component_method", "f1": "f1_weight"})
        tables["T08_fold_specific_f1_weights.csv"] = weights
    if panel_id == "supp_learning":
        cached_learning = (analysis_tables or {}).get("learning_curve")
        if isinstance(cached_learning, pd.DataFrame) and not cached_learning.empty:
            learning = cached_learning.copy()
        else:
            learning_path = Path(str(common["learning_curve_csv"])).expanduser()
            learning = pd.read_csv(learning_path, low_memory=False)
        required = {"target_cells_per_class", "fold", "macro_f1"}
        missing = required.difference(learning.columns)
        if missing:
            raise ValueError(
                "Learning-curve CSV is missing: " + ", ".join(sorted(missing))
            )
        tables["T16_annotation_learning_curve.csv"] = learning
    return tables


def default_common_settings() -> dict[str, Any]:
    return {
        "font_family": "DejaVu Sans",
        "base_font_size": 9.5,
        "title_font_size": 13.0,
        "subplot_title_font_size": 11.0,
        "axis_font_size": 9.5,
        "tick_font_size": 8.0,
        "annotation_font_size": 8.0,
        "dpi": 300,
        "background_color": PUBLICATION_BACKGROUND,
        "grid_color": "#B8B8B8",
        "grid_alpha": 0.35,
        "method_order": list(METHOD_ORDER),
        "method_colors": dict(METHOD_COLORS),
        "method_labels": dict(METHOD_LABELS),
        "reference_colors": dict(REFERENCE_COLORS),
        "reference_labels": dict(REFERENCE_LABELS),
        "reference_order": list(REFERENCE_ORDER),
        "component_methods": list(METHOD_ORDER[:4]),
        "confidence_columns": {},
        "confidence_thresholds": [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95],
        "excluded_labels": sorted(INVALID_LABELS),
        "consensus_min_agree": 3,
        "group_column": "",
        "group_order": [],
        "group_labels": {},
        "group_colors": {},
        "x_column": "",
        "y_column": "",
        "primary_method": METHOD_ORDER[-2],
        "secondary_method": METHOD_ORDER[-1],
        "primary_reference": REFERENCE_ORDER[0],
        "secondary_reference": REFERENCE_ORDER[1],
        "learning_curve_csv": "",
        "heldout_colors": dict(HELDOUT_IMAGE_COLORS),
        "phenotype_colors": dict(PHENOTYPE_COLORS),
        "formats": ["PNG", "SVG"],
    }


def default_panel_settings(spec: PaperPanelSpec) -> dict[str, Any]:
    width_height = REFERENCE_PANEL_DIMENSIONS.get(spec.panel_id, (10.0, 5.3))
    cmap = "Blues"
    if "precision" in spec.panel_id:
        cmap = "Reds"
    elif "recall" in spec.panel_id:
        cmap = "Greens"
    elif spec.panel_id == "supp_pairwise":
        cmap = "YlGnBu"
    settings = {
        "title": spec.default_title,
        "panel_label": spec.default_panel_label,
        "output_stem": spec.default_stem,
        "width": width_height[0],
        "height": width_height[1],
        "legend_location": "Outside right" if spec.has_legend else "Hidden",
        "legend_title": "",
        "legend_columns": 1,
        "x_label": "",
        "y_label": "",
        "shade_methods": ["ML-MR", "Consensus-ML"] if spec.method_axis else [],
        "shade_color": "#F2F2F2",
        "shade_alpha": 1.0,
        "bar_grouping": "Series within categories",
        "bar_width": 0.23,
        "line_width": 2.0,
        "marker_size": 7.0,
        "title_font_size": 13.0,
        "grid_alpha": 0.35,
        "box_aspect": 0.0,
        "point_size": 5.2,
        "heatmap_cmap": cmap,
        "heatmap_vmin": (
            50.0
            if spec.panel_id == "supp_pairwise"
            else (
                0.0
                if spec.panel_id
                in {
                    "supp_weights",
                    "supp_confusion",
                    "supp_human",
                    "supp_reference_distribution",
                }
                else 0.5
            )
        ),
        "heatmap_vmax": (
            100.0
            if spec.panel_id == "supp_pairwise"
            else (5.0 if spec.panel_id == "supp_reference_distribution" else 1.0)
        ),
        "heatmap_auto_range": spec.panel_id == "supp_reference_distribution",
        "heatmap_annotations": spec.panel_id != "supp_confusion",
        "heatmap_annotation_size": 9.5,
        "heatmap_square": True,
        "heatmap_linewidth": 0.5,
        "colorbar_label": "",
        "minimum_bin_cells": 20,
        "selected_confidence": 0.75,
        "match_color": "#2CA25F",
        "mismatch_color": "#DE2D26",
        "unscored_color": "#BDBDBD",
        "spatial_reference": "",
        "focus_reference": "",
        "class_metric": "f1",
        "spatial_methods": [],
        "confusion_methods": [],
        "confidence_method": "",
        "learning_method": "",
    }

    # Panel-specific values mirror the corresponding plotting calls in
    # ``test_codes/Fig3_code_results/fig3_code_v2.py`` (and the v1/legacy
    # functions that v2 deliberately reuses).
    reference_overrides: dict[str, dict[str, Any]] = {
        "main_workflow": {"line_width": 1.4, "title_font_size": 13.2},
        "main_heldout_macro": {
            "line_width": 1.8,
            "marker_size": 6.9,
            "grid_alpha": 0.35,
            "box_aspect": 0.6,
            "legend_title": "Held-out image group",
        },
        "main_consensus_tradeoff": {
            "line_width": 2.0,
            "marker_size": 8.0,
            "grid_alpha": 0.30,
        },
        "main_ref2_metrics": {
            "marker_size": 6.8,
            "bar_width": 0.23,
            "grid_alpha": 0.30,
        },
        "main_spatial": {
            "point_size": 5.2,
            "legend_location": "Outside right (lower)",
            "title_font_size": 12.0,
        },
        "supp_reference_distribution": {
            "bar_width": 0.19,
            "grid_alpha": 0.25,
            "heatmap_square": False,
        },
        "supp_weights": {"heatmap_square": False},
        "supp_leave_one_out": {
            "line_width": 2.0,
            "marker_size": 6.0,
            "grid_alpha": 0.30,
            "title_font_size": 12.0,
        },
        "supp_routes": {
            "bar_width": 0.23,
            "grid_alpha": 0.30,
            "title_font_size": 12.0,
        },
        "supp_human": {
            "grid_alpha": 0.30,
            "title_font_size": 12.0,
            "heatmap_annotations": False,
            "heatmap_square": False,
            "heatmap_linewidth": 0.0,
        },
        "supp_spatial": {
            "point_size": 5.2,
            "legend_location": "Outside right (lower)",
            "title_font_size": 12.0,
        },
        "supp_confidence_all": {
            "line_width": 2.0,
            "marker_size": 5.5,
            "grid_alpha": 0.32,
            "title_font_size": 12.0,
        },
        "supp_nonconsensus_threshold": {
            "line_width": 2.3,
            "marker_size": 7.0,
            "grid_alpha": 0.30,
        },
        "supp_confusion": {
            "title_font_size": 12.0,
            "heatmap_linewidth": 0.0,
        },
        "supp_radar": {
            "line_width": 1.8,
            "marker_size": 7.0,
            "grid_alpha": 0.32,
        },
        "supp_heldout_variation": {
            "line_width": 1.8,
            "marker_size": 6.8,
            "grid_alpha": 0.35,
            "box_aspect": 0.6,
            "legend_title": "Held-out image group",
        },
        "supp_learning": {
            "line_width": 2.3,
            "marker_size": 7.0,
            "grid_alpha": 0.35,
        },
        "supp_bins_ref1": {
            "line_width": 2.0,
            "marker_size": 7.0,
            "grid_alpha": 0.32,
        },
        "supp_bins_ref2": {
            "line_width": 2.0,
            "marker_size": 7.0,
            "grid_alpha": 0.32,
        },
        "supp_pairwise": {
            "heatmap_linewidth": 0.6,
            "heatmap_annotation_size": 11.0,
        },
    }
    settings.update(reference_overrides.get(spec.panel_id, {}))
    return settings


def _normalized_color(value: Any, fallback: str) -> str:
    """Return a Matplotlib-safe hex color without leaking ambient UI styles."""
    try:
        return str(to_hex(to_rgb(str(value))))
    except (TypeError, ValueError):
        return str(to_hex(to_rgb(fallback)))


def _canvas_background(common: Mapping[str, Any]) -> str:
    return _normalized_color(
        common.get("background_color", PUBLICATION_BACKGROUND),
        PUBLICATION_BACKGROUND,
    )


def _canvas_foreground(common: Mapping[str, Any]) -> str:
    red, green, blue = to_rgb(_canvas_background(common))

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)
    )
    return PUBLICATION_DARK_FOREGROUND if luminance < 0.40 else PUBLICATION_FOREGROUND


def _muted_canvas_foreground(common: Mapping[str, Any]) -> str:
    return "#D0D0D0" if _canvas_foreground(common) == PUBLICATION_DARK_FOREGROUND else "#555555"


def _figure_title_size(
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> float:
    return float(panel.get("title_font_size", common["title_font_size"]))


def _style_context(common: Mapping[str, Any]) -> dict[str, Any]:
    background = _canvas_background(common)
    foreground = _canvas_foreground(common)
    return {
        "font.family": str(common["font_family"]),
        "font.size": float(common["base_font_size"]),
        "axes.titlesize": float(
            common.get("subplot_title_font_size", common["title_font_size"])
        ),
        "axes.labelsize": float(common["axis_font_size"]),
        "xtick.labelsize": float(common["tick_font_size"]),
        "ytick.labelsize": float(common["tick_font_size"]),
        "figure.facecolor": background,
        "figure.edgecolor": background,
        "axes.facecolor": background,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "axes.titlecolor": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "text.color": foreground,
        "legend.facecolor": background,
        "legend.edgecolor": foreground,
        "legend.labelcolor": foreground,
        "grid.color": _normalized_color(common.get("grid_color"), "#B8B8B8"),
        "savefig.facecolor": background,
        "savefig.edgecolor": background,
        "savefig.transparent": False,
    }


def _apply_publication_canvas(
    figure: plt.Figure,
    common: Mapping[str, Any],
) -> None:
    """Make previews and PNG/SVG exports opaque and internally consistent."""
    background = _canvas_background(common)
    foreground = _canvas_foreground(common)
    figure.patch.set_facecolor(background)
    figure.patch.set_edgecolor(background)
    figure.patch.set_alpha(1.0)
    for text_item in figure.texts:
        text_item.set_color(foreground)
    for ax in figure.axes:
        ax.set_facecolor(background)
        ax.patch.set_alpha(1.0)
        ax.title.set_color(foreground)
        ax.xaxis.label.set_color(foreground)
        ax.yaxis.label.set_color(foreground)
        ax.tick_params(axis="both", colors=foreground)
        for spine in ax.spines.values():
            spine.set_edgecolor(foreground)
        legend = ax.get_legend()
        if legend is not None:
            for text_item in legend.get_texts():
                text_item.set_color(foreground)
            if legend.get_title() is not None:
                legend.get_title().set_color(foreground)
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(foreground)
    for legend in figure.legends:
        for text_item in legend.get_texts():
            text_item.set_color(foreground)
        if legend.get_title() is not None:
            legend.get_title().set_color(foreground)
        legend.get_frame().set_facecolor(background)
        legend.get_frame().set_edgecolor(foreground)


def _ordered_methods(common: Mapping[str, Any]) -> list[str]:
    values = common.get("method_order")
    if values is None:
        return list(METHOD_ORDER)
    return list(dict.fromkeys(str(value) for value in values))


def _ordered_references(common: Mapping[str, Any]) -> list[str]:
    values = common.get("reference_order")
    if values is None:
        return list(REFERENCE_ORDER)
    return list(dict.fromkeys(str(value) for value in values))


def _primary_reference(common: Mapping[str, Any]) -> str:
    references = _ordered_references(common)
    if not references:
        return ""
    requested = str(common.get("primary_reference") or "")
    return requested if requested in references else references[0]


def _secondary_reference(common: Mapping[str, Any]) -> str:
    references = _ordered_references(common)
    if not references:
        return ""
    requested = str(common.get("secondary_reference") or "")
    if requested in references:
        return requested
    return references[1] if len(references) > 1 else references[0]


def _primary_method(common: Mapping[str, Any]) -> str:
    methods = _ordered_methods(common)
    if not methods:
        return ""
    requested = str(common.get("primary_method") or "")
    return requested if requested in methods else methods[0]


def _secondary_method(common: Mapping[str, Any]) -> str:
    methods = _ordered_methods(common)
    if not methods:
        return ""
    requested = str(common.get("secondary_method") or "")
    if requested in methods:
        return requested
    return methods[1] if len(methods) > 1 else methods[0]


def _group_values(common: Mapping[str, Any], frame: pd.DataFrame) -> list[str]:
    configured = [str(value) for value in common.get("group_order", [])]
    available = [
        str(value)
        for value in frame.get("fold", pd.Series(dtype="string")).dropna().unique()
        if str(value) != "Pooled"
    ]
    selected = [value for value in configured if value in available]
    return selected or available


def _group_label(common: Mapping[str, Any], group: str) -> str:
    return str(common.get("group_labels", {}).get(group, group.replace("_", "/")))


def _group_color(common: Mapping[str, Any], group: str, index: int = 0) -> str:
    configured = common.get("group_colors", {}).get(group)
    if configured:
        return str(configured)
    del index
    if group in HELDOUT_IMAGE_COLORS:
        return HELDOUT_IMAGE_COLORS[group]
    return _stable_named_color(f"group::{group}")


def _method_color(common: Mapping[str, Any], method: str) -> str:
    return str(
        common.get("method_colors", {}).get(
            method, METHOD_COLORS.get(method, "#555555")
        )
    )


def _method_label(common: Mapping[str, Any], method: str) -> str:
    return str(common.get("method_labels", {}).get(method, method))


def _reference_label(common: Mapping[str, Any], reference: str) -> str:
    return str(
        common.get("reference_labels", {}).get(
            reference, REFERENCE_LABELS.get(reference, reference)
        )
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    if label:
        ax.text(
            -0.10,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
        )


def _shade_methods(ax: plt.Axes, methods: list[str], panel: Mapping[str, Any]) -> None:
    shaded = set(str(value) for value in panel.get("shade_methods", []))
    for index, method in enumerate(methods):
        if method in shaded:
            ax.axvspan(
                index - 0.5,
                index + 0.5,
                color=str(panel.get("shade_color", "#F2F2F2")),
                alpha=float(panel.get("shade_alpha", 1.0)),
                zorder=-2,
            )


def _legend_kwargs(location: str, columns: int = 1) -> dict[str, Any] | None:
    if location == "Hidden":
        return None
    values: dict[str, dict[str, Any]] = {
        "Outside right": {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)},
        "Outside right (lower)": {
            "loc": "lower left",
            "bbox_to_anchor": (1.0, 0.0),
        },
        "Inside upper right": {"loc": "upper right"},
        "Inside upper left": {"loc": "upper left"},
        "Inside lower right": {"loc": "lower right"},
        "Inside lower left": {"loc": "lower left"},
        "Above": {"loc": "lower center", "bbox_to_anchor": (0.5, 1.02)},
        "Below": {"loc": "upper center", "bbox_to_anchor": (0.5, -0.18)},
        "Best": {"loc": "best"},
    }
    return {
        **values.get(location, values["Outside right"]),
        "ncol": max(1, int(columns)),
        "frameon": False,
    }


def _apply_legend(
    ax: plt.Axes,
    panel: Mapping[str, Any],
    *,
    handles: list[Any] | None = None,
    labels: list[str] | None = None,
) -> None:
    kwargs = _legend_kwargs(
        str(panel.get("legend_location", "Outside right")),
        int(panel.get("legend_columns", 1)),
    )
    current = ax.get_legend()
    if kwargs is None:
        if current is not None:
            current.remove()
        return
    title = str(panel.get("legend_title", "") or "")
    if handles is None:
        ax.legend(title=title or None, **kwargs)
    else:
        ax.legend(handles=handles, labels=labels, title=title or None, **kwargs)


def _finish_axis(
    ax: plt.Axes,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    default_x: str = "",
    default_y: str = "",
    grid_axis: str | None = "y",
) -> None:
    ax.set_xlabel(str(panel.get("x_label") or default_x))
    ax.set_ylabel(str(panel.get("y_label") or default_y))
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            linestyle=":",
            color=str(common["grid_color"]),
            alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
        )


def _ordered_phenotypes(values: Any, *, include_others: bool = False) -> list[str]:
    present = {str(value) for value in values}
    ordered = [
        value
        for value in PHENOTYPE_ORDER
        if value in present and (include_others or value != "Others")
    ]
    return ordered + sorted(
        present.difference(ordered).difference(
            {"Others"} if not include_others else set()
        )
    )


def _pooled_per_class(
    frame: pd.DataFrame,
    reference: str,
    metric: str,
    methods: list[str],
) -> pd.DataFrame:
    selected = frame.loc[
        frame["reference"].eq(reference)
        & frame["fold"].eq("Pooled")
        & frame["method"].isin(methods)
    ]
    pivot = selected.pivot(index="phenotype", columns="method", values=metric)
    return pivot.reindex(_ordered_phenotypes(pivot.index)).reindex(columns=methods)


def _plot_workflow(
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str,
        size: float = 8.5,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            edgecolor=color,
            facecolor=color + "18",
            linewidth=float(panel["line_width"]),
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            linespacing=1.25,
        )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=float(panel["line_width"]),
                color=_muted_canvas_foreground(common),
            )
        )

    workflow_text = panel.get("workflow_text", {})
    box(
        0.20,
        1.72,
        2.25,
        1.55,
        str(
            workflow_text.get(
                "components",
                "Four complementary\nphenotype calls\nLeiden · SOM · Vortex · RESOLVE",
            )
        ),
        "#4E79A7",
        8.1,
    )
    box(
        3.00,
        3.20,
        2.35,
        1.22,
        str(
            workflow_text.get(
                "mr_source", "Expert-labelled Ref1 cells\nfrom the two training images"
            )
        ),
        _method_color(common, "ML-MR"),
    )
    box(
        3.00,
        0.48,
        2.35,
        1.22,
        str(
            workflow_text.get(
                "consensus_source", "Cells with ≥3-of-4\nmatching phenotype calls"
            )
        ),
        _method_color(common, "Consensus-ML"),
    )
    box(
        5.95,
        3.20,
        2.35,
        1.22,
        str(
            workflow_text.get(
                "mr_training",
                "Learn phenotype-specific\nmethod weights and\nmarker patterns",
            )
        ),
        _method_color(common, "ML-MR"),
    )
    box(
        5.95,
        0.48,
        2.35,
        1.22,
        str(
            workflow_text.get(
                "consensus_training",
                "Train on consensus cells;\npredict non-consensus cells",
            )
        ),
        _method_color(common, "Consensus-ML"),
    )
    box(
        8.90,
        1.67,
        2.30,
        1.60,
        str(
            workflow_text.get(
                "holdout", "Hold out one complete image\nP11 · P28 · P2/P10"
            )
        ),
        "#333333",
    )
    box(
        11.75,
        1.67,
        2.40,
        1.60,
        str(
            workflow_text.get(
                "comparison", "Compare predictions with\nindependent expert labels"
            )
        ),
        "#333333",
    )
    for coordinates in [
        (2.45, 2.68, 3.00, 3.66),
        (2.45, 2.30, 3.00, 1.10),
        (5.35, 3.81, 5.95, 3.81),
        (5.35, 1.09, 5.95, 1.09),
        (8.30, 3.67, 8.90, 2.78),
        (8.30, 1.22, 8.90, 2.16),
        (11.20, 2.47, 11.75, 2.47),
    ]:
        arrow(*coordinates)
    ax.text(
        7.25,
        4.84,
        str(panel["title"]),
        ha="center",
        va="top",
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
    )
    footer = str(
        workflow_text.get(
            "footer",
            "No cells from the held-out image enter method weighting, expert-label "
            "training, or consensus-label training.",
        )
    )
    ax.text(
        7.25,
        0.03,
        footer,
        ha="center",
        fontsize=8.5,
        color=_muted_canvas_foreground(common),
    )
    if panel.get("panel_label"):
        ax.text(
            -0.02,
            4.93,
            str(panel["panel_label"]),
            fontsize=15,
            fontweight="bold",
            va="top",
        )
    return fig


def _heldout_legend_handles(
    common: Mapping[str, Any],
    pooled: pd.DataFrame,
    metrics: list[tuple[str, str]],
    groups: list[str],
) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=_group_color(common, group, index),
            linewidth=2.0,
            label=_group_label(common, group),
        )
        for index, group in enumerate(groups)
    ]
    header = "Pooled " + " / ".join(label for _, label in metrics)
    handles.append(Line2D([], [], linestyle="None", label=header))
    for method in _ordered_methods(common):
        if method not in pooled.index:
            continue
        values = " / ".join(
            f"{float(pooled.loc[method, column]):.3f}" for column, _ in metrics
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=_method_color(common, method),
                markeredgecolor="black",
                markersize=7,
                label=f"{_method_label(common, method)}: {values}",
            )
        )
    return handles


def _plot_heldout_macro(
    summary: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    methods = _ordered_methods(common)
    reference = str(panel.get("focus_reference") or _primary_reference(common))
    groups = _group_values(common, summary)
    selected = summary.loc[
        summary["reference"].eq(reference)
        & summary["method"].isin(methods)
        & summary["fold"].isin([*groups, "Pooled"])
    ].copy()
    if selected.empty:
        raise ValueError(
            "No grouped performance rows were found for the selected reference."
        )
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    x = np.arange(len(methods))
    _shade_methods(ax, methods, panel)
    for group_index, group in enumerate(groups):
        rows = selected.loc[selected["fold"].eq(group)].set_index("method")
        ax.plot(
            x,
            rows.reindex(methods)["macro_f1"].to_numpy(dtype=float),
            color=_group_color(common, group, group_index),
            linewidth=float(panel["line_width"]),
            alpha=0.95,
            zorder=1,
        )
    pooled = selected.loc[selected["fold"].eq("Pooled")].set_index("method")
    for index, method in enumerate(methods):
        if method in pooled.index:
            ax.scatter(
                index,
                float(pooled.loc[method, "macro_f1"]),
                s=float(panel["marker_size"]) ** 2 * 2,
                color=_method_color(common, method),
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
            )
    pooled_n = int(selected.loc[selected["fold"].eq("Pooled"), "n"].max())
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_method_label(common, method) for method in methods], rotation=45, ha="right"
    )
    ax.set_xlim(-0.25, len(methods) - 0.75)
    if float(panel.get("box_aspect", 0.0)) > 0:
        ax.set_box_aspect(float(panel["box_aspect"]))
    ax.set_ylim(float(panel.get("y_min", 0.70)), float(panel.get("y_max", 1.0)))
    ax.set_title(
        f"{panel['title']}\n{_reference_label(common, reference)}, n={pooled_n:,}",
        pad=12,
        fontweight="bold",
    )
    _finish_axis(ax, common, panel, default_y="Macro-F1 on held-out reference cells")
    _apply_legend(
        ax,
        panel,
        handles=_heldout_legend_handles(
            common,
            pooled,
            [("macro_f1", "macro-F1")],
            groups,
        ),
    )
    _panel_label(ax, str(panel.get("panel_label", "")))
    return fig


def _plot_consensus_tradeoff(
    scenarios: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    no_drop = scenarios["dropped_method"].isna() | scenarios["dropped_method"].astype(
        str
    ).eq("None")
    selected_all = scenarios.loc[no_drop]
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    markers = ["o", "s", "^", "D", "P", "X"]
    references = _ordered_references(common)
    for reference_index, reference in enumerate(references):
        selected = selected_all.loc[
            selected_all["reference"].eq(reference)
        ].sort_values("min_agree")
        if selected.empty:
            continue
        color = str(common["reference_colors"].get(reference, "#777777"))
        ax.plot(
            selected["all_cell_coverage"] * 100,
            selected["macro_f1"],
            marker=markers[reference_index % len(markers)],
            markersize=float(panel["marker_size"]),
            linewidth=float(panel["line_width"]),
            color=color,
            label=_reference_label(common, reference),
        )
        for _, row in selected.iterrows():
            denominator = int(row.get("n_methods", 4))
            ax.annotate(
                f"{int(row['min_agree'])}/{denominator}",
                (row["all_cell_coverage"] * 100, row["macro_f1"]),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=float(common["annotation_font_size"]),
                fontweight="bold"
                if int(row["min_agree"]) == int(common.get("consensus_min_agree", 3))
                else "normal",
            )
    chosen = selected_all.loc[
        selected_all["reference"].eq(_primary_reference(common))
        & selected_all["min_agree"].eq(int(common.get("consensus_min_agree", 3)))
    ]
    if not chosen.empty:
        ax.axvline(
            chosen.iloc[0]["all_cell_coverage"] * 100,
            color=_muted_canvas_foreground(common),
            linestyle=":",
            linewidth=1.2,
        )
    ax.set_title(str(panel["title"]), pad=12, fontweight="bold")
    _finish_axis(
        ax,
        common,
        panel,
        default_x="All usable cells receiving a direct label (%)",
        default_y="Macro-F1 among directly labelled cells",
        grid_axis=None,
    )
    ax.grid(
        linestyle=":",
        color=str(common["grid_color"]),
        alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
    )
    _apply_legend(ax, panel)
    _panel_label(ax, str(panel.get("panel_label", "")))
    return fig


def _plot_ref2_metrics(
    summary: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    methods = _ordered_methods(common)
    reference = str(panel.get("focus_reference") or _primary_reference(common))
    metrics = [
        ("macro_f1", "Macro-F1"),
        ("micro_f1", "Micro-F1"),
        ("weighted_f1", "Weighted-F1"),
    ]
    metric_colors = list(panel.get("metric_colors", ["#24445C", "#6D8FA6", "#B5CBD8"]))
    selected = summary.loc[
        summary["reference"].eq(reference)
        & summary["fold"].eq("Pooled")
        & summary["method"].isin(methods)
    ]
    if selected.empty:
        raise ValueError(
            "No pooled performance rows were found for the selected reference."
        )
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    width = float(panel["bar_width"])
    grouping = str(panel.get("bar_grouping", "Series within categories"))
    if grouping == "Categories within series":
        x = np.arange(len(metrics))
        width = min(width, 0.8 / max(1, len(methods)))
        for offset, method in enumerate(methods):
            row = selected.loc[selected["method"].eq(method)]
            if row.empty:
                continue
            values = [float(row[column].iloc[0]) for column, _ in metrics]
            ax.bar(
                x + (offset - (len(methods) - 1) / 2) * width,
                values,
                width=width,
                color=_method_color(common, method),
                label=_method_label(common, method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics])
    else:
        x = np.arange(len(methods))
        _shade_methods(ax, methods, panel)
        for offset, ((column, label), color) in enumerate(zip(metrics, metric_colors)):
            values = [
                float(selected.loc[selected["method"].eq(method), column].iloc[0])
                if selected["method"].eq(method).any()
                else np.nan
                for method in methods
            ]
            ax.bar(
                x + (offset - 1) * width, values, width=width, color=color, label=label
            )
        for index, method in enumerate(methods):
            row = selected.loc[selected["method"].eq(method)]
            if not row.empty:
                ax.scatter(
                    index - width,
                    float(row["macro_f1"].iloc[0]),
                    s=float(panel["marker_size"]) ** 2 * 0.6,
                    color=_method_color(common, method),
                    edgecolor="black",
                    linewidth=0.4,
                    zorder=4,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_method_label(common, method) for method in methods],
            rotation=20,
            ha="right",
        )
    n = int(selected["n"].max())
    ax.set_ylim(float(panel.get("y_min", 0.0)), float(panel.get("y_max", 1.0)))
    ax.set_title(f"{panel['title']} (n={n:,})", pad=12, fontweight="bold")
    _finish_axis(
        ax,
        common,
        panel,
        default_y=f"F1 score against {_reference_label(common, reference)}",
    )
    _apply_legend(ax, panel)
    _panel_label(ax, str(panel.get("panel_label", "")))
    return fig


def _heatmap_settings(panel: Mapping[str, Any]) -> dict[str, Any]:
    auto_range = bool(panel.get("heatmap_auto_range", False))
    return {
        "cmap": str(panel["heatmap_cmap"]),
        "vmin": None if auto_range else float(panel["heatmap_vmin"]),
        "vmax": None if auto_range else float(panel["heatmap_vmax"]),
        "square": bool(panel["heatmap_square"]),
        "annot": bool(panel["heatmap_annotations"]),
        "fmt": ".2f",
        "linewidths": float(panel["heatmap_linewidth"]),
    }


def _heatmap_annotation_size(
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> float:
    return float(panel.get("heatmap_annotation_size", common["annotation_font_size"]))


def _colorbar_label(panel: Mapping[str, Any], default: str) -> str:
    return str(panel.get("colorbar_label") or default)


def _plot_combined_per_class(
    per_class: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    methods = _ordered_methods(common)
    references = _ordered_references(common)
    fig = plt.figure(figsize=(float(panel["width"]), float(panel["height"])))
    grid = fig.add_gridspec(
        1,
        len(references) + 1,
        width_ratios=[*[1] * len(references), 0.055],
        wspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(len(references))]
    cax = fig.add_subplot(grid[0, len(references)])
    for index, (ax, reference) in enumerate(zip(axes, references)):
        pivot = _pooled_per_class(per_class, reference, "f1", methods)
        pivot = pivot.rename(
            columns={method: _method_label(common, method) for method in methods}
        )
        sns.heatmap(
            pivot,
            **_heatmap_settings(panel),
            cbar=index == len(references) - 1,
            cbar_ax=cax if index == len(references) - 1 else None,
            cbar_kws={"label": _colorbar_label(panel, "F1")},
            annot_kws={"size": _heatmap_annotation_size(common, panel)},
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(_reference_label(common, reference), pad=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
    _panel_label(axes[0], str(panel.get("panel_label", "")))
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
        y=0.99,
    )
    return fig


def _plot_single_per_class(
    per_class: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    reference: str,
    metric: str,
) -> plt.Figure:
    methods = _ordered_methods(common)
    pivot = _pooled_per_class(per_class, reference, metric, methods)
    pivot = pivot.rename(
        columns={method: _method_label(common, method) for method in methods}
    )
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    sns.heatmap(
        pivot,
        **_heatmap_settings(panel),
        cbar_kws={
            "label": _colorbar_label(
                panel,
                metric.replace("_", " ").title(),
            )
        },
        annot_kws={"size": _heatmap_annotation_size(common, panel)},
        ax=ax,
    )
    ax.set_xlabel(str(panel.get("x_label") or ""))
    ax.set_ylabel(str(panel.get("y_label") or ""))
    ax.set_title(str(panel["title"]), pad=12, fontweight="bold")
    _panel_label(ax, str(panel.get("panel_label", "")))
    return fig


def _spatial_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    colors: Any,
    title: str,
    point_size: float,
) -> None:
    ax.scatter(
        pd.to_numeric(frame["x_centroid"], errors="coerce"),
        pd.to_numeric(frame["y_centroid"], errors="coerce"),
        c=colors,
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_spatial_validation(
    oof: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    references = _ordered_references(common)
    reference = str(
        panel.get("spatial_reference")
        or panel.get("focus_reference")
        or _primary_reference(common)
    )
    if reference not in references:
        reference = references[0]
    if reference not in oof.columns:
        raise ValueError(
            f"Spatial reference column {reference!r} is absent from the OOF table."
        )
    cleaned = oof[reference].astype("string").str.strip()
    region = oof.loc[oof[reference].notna() & cleaned.ne("")].copy()
    if region.empty:
        raise ValueError(f"No labelled cells were found in {reference}.")
    truth = region[reference].astype(str)
    biological = ~truth.str.strip().str.lower().isin(INVALID_LABELS)
    requested_methods = panel.get("spatial_methods") or [
        _primary_method(common),
        _secondary_method(common),
    ]
    legacy_columns = {
        "Leiden": "phenotype_leiden",
        "SOM": "phenotype_som",
        "Vortex": "phenotype_vortex",
        "RESOLVE": "phenotype_threshold",
        "ML-MR": "prediction_mr_ml_strict",
        "Consensus-ML": "prediction_consensus_ml_strict",
    }
    method_columns: list[tuple[str, str]] = []
    for value in requested_methods:
        method = str(value)
        column = method if method in region.columns else legacy_columns.get(method, "")
        if column in region.columns and method not in {
            item[0] for item in method_columns
        }:
            method_columns.append((method, column))
    method_columns = method_columns[:2]
    if not method_columns:
        raise ValueError("Select at least one method for spatial validation.")
    matches = {
        method: biological & truth.eq(region[column].astype(str))
        for method, column in method_columns
    }
    accuracies = {
        method: float(match.sum() / max(1, biological.sum()))
        for method, match in matches.items()
    }
    fig, axes = plt.subplots(
        1,
        1 + len(method_columns),
        figsize=(float(panel["width"]), float(panel["height"])),
        squeeze=False,
    )
    axes = list(axes.flat)
    phenotype_colors = common.get("phenotype_colors", PHENOTYPE_COLORS)
    _spatial_scatter(
        axes[0],
        region,
        truth.map(phenotype_colors).fillna(str(panel["unscored_color"])),
        f"Manual {reference.replace('phenotype_', '').upper()}\n{len(region):,} annotated cells",
        float(panel["point_size"]),
    )
    unresolved = truth.str.lower().eq("others")
    for axis, (method, _column) in zip(axes[1:], method_columns):
        colors = np.where(
            unresolved,
            panel["unscored_color"],
            np.where(matches[method], panel["match_color"], panel["mismatch_color"]),
        )
        _spatial_scatter(
            axis,
            region,
            colors,
            f"{_method_label(common, method)}\n{accuracies[method]:.1%} cell agreement",
            float(panel["point_size"]),
        )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=str(panel["match_color"]),
            label="Match",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=str(panel["mismatch_color"]),
            label="Mismatch",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=str(panel["unscored_color"]),
            label="Manual Others (not scored)",
            markersize=7,
        ),
    ]
    _apply_legend(axes[-1], panel, handles=legend)
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
        y=1.02,
    )
    _panel_label(axes[0], str(panel.get("panel_label", "")))
    fig.subplots_adjust(wspace=0.08)
    return fig


def _plot_reference_distribution(
    distribution: pd.DataFrame,
    locations: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    reference_values = [
        value for value in locations["reference"].dropna().astype(str).unique()
    ]
    preferred = _ordered_references(common)
    references = [value for value in preferred if value in reference_values]
    references += sorted(set(reference_values).difference(references))
    rois = list(dict.fromkeys(locations["batch_id"].dropna().astype(str)))
    counts = locations.pivot_table(
        index="batch_id",
        columns="reference",
        values="annotated_cells",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=rois, columns=references, fill_value=0)
    composition_source = distribution.loc[distribution["reference"].isin(references)]
    pivot = composition_source.pivot(
        index="phenotype", columns="reference", values="cells"
    )
    pivot = pivot.reindex(
        _ordered_phenotypes(pivot.index, include_others=True)
    ).reindex(columns=references)
    annotations = pivot.fillna(0).astype(int).map(lambda value: f"{value:,}")
    composition = np.log10(pivot.fillna(0) + 1)

    fig = plt.figure(figsize=(float(panel["width"]), float(panel["height"])))
    grid = fig.add_gridspec(1, 3, width_ratios=[2.35, 1.25, 0.07], wspace=0.42)
    ax_counts = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    x = np.arange(len(rois))
    grouping = str(panel.get("bar_grouping", "Series within categories"))
    if grouping == "Categories within series":
        x = np.arange(len(references))
        width = min(float(panel["bar_width"]), 0.8 / max(1, len(rois)))
        for index, roi in enumerate(rois):
            values = counts.loc[roi, references].to_numpy(dtype=float)
            ax_counts.bar(
                x + (index - (len(rois) - 1) / 2) * width,
                values,
                width=width,
                label=roi,
            )
        ax_counts.set_xticks(x)
        ax_counts.set_xticklabels(
            [_reference_label(common, ref) for ref in references],
            rotation=35,
            ha="right",
        )
    else:
        width = min(float(panel["bar_width"]), 0.8 / max(1, len(references)))
        for index, reference in enumerate(references):
            color = str(common["reference_colors"].get(reference, "#777777"))
            ax_counts.bar(
                x + (index - (len(references) - 1) / 2) * width,
                counts[reference].to_numpy(dtype=float),
                width=width,
                color=color,
                label=_reference_label(common, reference),
            )
        ax_counts.set_xticks(x)
        ax_counts.set_xticklabels(rois, rotation=35, ha="right")
    ax_counts.set_yscale("symlog", linthresh=1)
    ax_counts.set_title(str(panel["title"]), pad=10, fontweight="bold")
    _finish_axis(
        ax_counts,
        common,
        panel,
        default_x="Image ROI",
        default_y="Annotated cells (symlog scale)",
    )
    _apply_legend(ax_counts, panel)
    sns.heatmap(
        composition,
        annot=annotations if bool(panel["heatmap_annotations"]) else False,
        fmt="",
        cmap=str(panel["heatmap_cmap"]),
        vmin=(
            None
            if bool(panel.get("heatmap_auto_range", True))
            else float(panel["heatmap_vmin"])
        ),
        vmax=(
            None
            if bool(panel.get("heatmap_auto_range", True))
            else float(panel["heatmap_vmax"])
        ),
        linewidths=float(panel["heatmap_linewidth"]),
        cbar_ax=cax,
        cbar_kws={"label": _colorbar_label(panel, "log10(n + 1)")},
        annot_kws={"size": _heatmap_annotation_size(common, panel)},
        ax=ax_heat,
    )
    ax_heat.set_xticklabels(
        [_reference_label(common, reference) for reference in references],
        rotation=25,
        ha="right",
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("Phenotype")
    ax_heat.set_title("Reference composition\n(cell counts)", pad=10, fontweight="bold")
    return fig


def _plot_weight_heatmap(
    weights: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    methods = [
        method
        for method in common.get("component_methods", _ordered_methods(common))
        if method in set(weights["component_method"].astype(str))
    ]
    component_mapping = {
        "phenotype_leiden": "Leiden",
        "phenotype_som": "SOM",
        "phenotype_vortex": "Vortex",
        "phenotype_threshold": "RESOLVE",
        "Leiden": "Leiden",
        "SOM": "SOM",
        "Vortex": "Vortex",
        "RESOLVE": "RESOLVE",
    }
    pooled = weights.groupby(["phenotype", "component_method"], as_index=False)[
        "f1_weight"
    ].mean()
    pooled["component_method"] = (
        pooled["component_method"]
        .map(component_mapping)
        .fillna(pooled["component_method"])
    )
    pivot = pooled.pivot(
        index="phenotype", columns="component_method", values="f1_weight"
    )
    pivot = pivot.reindex(_ordered_phenotypes(pivot.index)).reindex(columns=methods)
    pivot = pivot.rename(
        columns={method: _method_label(common, method) for method in methods}
    )
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    sns.heatmap(
        pivot,
        **_heatmap_settings(panel),
        cbar_kws={"label": _colorbar_label(panel, "Mean training-fold F1 weight")},
        annot_kws={"size": _heatmap_annotation_size(common, panel)},
        ax=ax,
    )
    ax.set_title(str(panel["title"]), pad=12, fontweight="bold")
    ax.set_xlabel(str(panel.get("x_label") or "Component method"))
    ax.set_ylabel(str(panel.get("y_label") or "Phenotype"))
    return fig


def _plot_leave_one_out(
    scenarios: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    references = _ordered_references(common)
    selected = scenarios.loc[scenarios["reference"].isin(references)]
    fig, axes_array = plt.subplots(
        1,
        len(references),
        figsize=(float(panel["width"]), float(panel["height"])),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array.flat)
    dropped_order = ["None", *list(common.get("component_methods", []))]
    for ax, reference in zip(axes, references):
        subset = selected.loc[selected["reference"].eq(reference)]
        for dropped in dropped_order:
            rows = subset.loc[
                subset["dropped_method"].fillna("None").astype(str).eq(dropped)
            ].sort_values("reference_coverage")
            if rows.empty:
                continue
            color = "#222222" if dropped == "None" else _method_color(common, dropped)
            label = (
                "All selected methods"
                if dropped == "None"
                else f"Without {_method_label(common, dropped)}"
            )
            ax.plot(
                rows["reference_coverage"] * 100,
                rows["macro_f1"],
                "o-",
                color=color,
                linewidth=float(panel["line_width"])
                if dropped == "None"
                else float(panel["line_width"]) * 0.6,
                markersize=float(panel["marker_size"]),
                label=label,
            )
        ax.set_title(_reference_label(common, reference))
        ax.set_xlabel(
            str(panel.get("x_label") or "Reference cells directly labelled (%)")
        )
        ax.grid(
            linestyle=":",
            color=str(common["grid_color"]),
            alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
        )
    axes[0].set_ylabel(str(panel.get("y_label") or "Direct-consensus macro-F1"))
    _apply_legend(axes[-1], panel)
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
    )
    return fig


def _plot_routes(
    routes: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    route_order = ["Direct consensus", "Non-consensus"]
    primary_method = _primary_method(common)
    secondary_method = _secondary_method(common)
    method_order = [
        method
        for method in ["Direct consensus", primary_method, secondary_method]
        if method in set(routes["method"].astype(str))
    ]
    method_order = list(dict.fromkeys(method_order))
    references = _ordered_references(common)
    fig, axes_array = plt.subplots(
        1,
        len(references),
        figsize=(float(panel["width"]), float(panel["height"])),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array.flat)
    width = min(float(panel["bar_width"]), 0.8 / max(1, len(method_order)))
    method_colors = {"Direct consensus": "#4E79A7", **dict(common["method_colors"])}
    for ax, reference in zip(axes, references):
        selected = routes.loc[routes["reference"].eq(reference)]
        grouping = str(panel.get("bar_grouping", "Series within categories"))
        if grouping == "Categories within series":
            x = np.arange(len(method_order))
            for route_index, route in enumerate(route_order):
                values = []
                for method in method_order:
                    row = selected.loc[
                        selected["route"].eq(route) & selected["method"].eq(method)
                    ]
                    values.append(
                        float(row["macro_f1"].iloc[0]) if not row.empty else np.nan
                    )
                ax.bar(
                    x + (route_index - 0.5) * width,
                    values,
                    width=width,
                    color=[
                        method_colors.get(method, "#777777") for method in method_order
                    ],
                    alpha=1.0 if route_index == 0 else 0.55,
                    label=route,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_method_label(common, method) for method in method_order],
                rotation=20,
                ha="right",
            )
        else:
            x = np.arange(len(route_order))
            for method_index, method in enumerate(method_order):
                values = []
                for route in route_order:
                    row = selected.loc[
                        selected["route"].eq(route) & selected["method"].eq(method)
                    ]
                    values.append(
                        float(row["macro_f1"].iloc[0]) if not row.empty else np.nan
                    )
                ax.bar(
                    x + (method_index - (len(method_order) - 1) / 2) * width,
                    values,
                    width=width,
                    color=method_colors.get(method, "#777777"),
                    label=_method_label(common, method),
                )
            counts = [
                int(selected.loc[selected["route"].eq(route), "n"].max())
                if selected["route"].eq(route).any()
                else 0
                for route in route_order
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(
                [
                    f"Direct ≥{int(common.get('consensus_min_agree', 3))}-of-"
                    f"{len(common.get('component_methods', []))}\nn={counts[0]:,}",
                    f"Resolved remainder\nn={counts[1]:,}",
                ]
            )
        ax.set_title(_reference_label(common, reference))
        ax.set_ylim(0, 1.0)
        ax.grid(
            axis="y",
            linestyle=":",
            color=str(common["grid_color"]),
            alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
        )
    axes[0].set_ylabel(str(panel.get("y_label") or "Macro-F1"))
    _apply_legend(axes[-1], panel)
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
    )
    return fig


def _plot_human_variation(
    summary: pd.DataFrame,
    matrix_frame: pd.DataFrame,
    enrichment: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    matrix = matrix_frame.copy()
    unnamed = [column for column in matrix.columns if str(column).startswith("Unnamed")]
    if unnamed:
        matrix = matrix.set_index(unnamed[0])
    elif len(matrix.columns):
        matrix = matrix.set_index(matrix.columns[0])
    normalized = matrix.apply(pd.to_numeric, errors="coerce").div(
        matrix.apply(pd.to_numeric, errors="coerce").sum(axis=1).replace(0, np.nan),
        axis=0,
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(float(panel["width"]), float(panel["height"]))
    )
    sns.heatmap(
        normalized,
        cmap=str(panel["heatmap_cmap"]),
        vmin=(
            None
            if bool(panel.get("heatmap_auto_range", False))
            else float(panel["heatmap_vmin"])
        ),
        vmax=(
            None
            if bool(panel.get("heatmap_auto_range", False))
            else float(panel["heatmap_vmax"])
        ),
        annot=bool(panel["heatmap_annotations"]),
        fmt=".2f",
        linewidths=float(panel["heatmap_linewidth"]),
        ax=axes[0],
        cbar_kws={"label": _colorbar_label(panel, "Row fraction")},
    )
    row = summary.iloc[0]
    references = _ordered_references(common)
    axes[0].set_xlabel(_reference_label(common, references[1]))
    axes[0].set_ylabel(_reference_label(common, references[0]))
    axes[0].set_title(
        f"Two annotators: {row['agreement']:.1%} agreement; κ={row['cohen_kappa']:.2f}"
    )
    denominator = len(common.get("component_methods", []))
    x = enrichment["min_agree"].astype(int).astype(str) + f"/{denominator}"
    axes[1].plot(
        x,
        enrichment["consensus_fraction_when_humans_agree"] * 100,
        "o-",
        color="#2CA25F",
        label="Humans agreed",
    )
    axes[1].plot(
        x,
        enrichment["consensus_fraction_when_humans_disagree"] * 100,
        "s--",
        color="#DE2D26",
        label="Humans disagreed",
    )
    axes[1].set_xlabel(str(panel.get("x_label") or "Algorithmic agreement rule"))
    axes[1].set_ylabel(str(panel.get("y_label") or "Cells meeting the rule (%)"))
    axes[1].set_title("Algorithmic agreement is enriched where humans agree")
    axes[1].grid(
        axis="y",
        linestyle=":",
        color=str(common["grid_color"]),
        alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
    )
    _apply_legend(axes[1], panel)
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
    )
    return fig


def _plot_confidence_all(
    confidence: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    methods = [
        method
        for method in _ordered_methods(common)
        if method in set(confidence["method"].astype(str))
    ]
    references = _ordered_references(common)
    fig, axes_array = plt.subplots(
        1,
        len(references),
        figsize=(float(panel["width"]), float(panel["height"])),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_array.flat)
    for ax, reference in zip(axes, references):
        selected = confidence.loc[confidence["reference"].eq(reference)]
        for method in methods:
            data = selected.loc[selected["method"].eq(method)].sort_values("threshold")
            ax.plot(
                data["coverage"] * 100,
                data["macro_f1"],
                "o-",
                color=_method_color(common, method),
                linewidth=float(panel["line_width"]),
                markersize=float(panel["marker_size"]),
                label=_method_label(common, method),
            )
            for _, row in data.iterrows():
                if np.isclose(float(row["threshold"]), [0.0, 0.75, 0.95]).any():
                    ax.annotate(
                        f"{float(row['threshold']):.2g}",
                        (row["coverage"] * 100, row["macro_f1"]),
                        xytext=(3, 5),
                        textcoords="offset points",
                        fontsize=float(common["annotation_font_size"]),
                    )
        ax.set_xlabel(str(panel.get("x_label") or "Accepted reference cells (%)"))
        ax.set_title(_reference_label(common, reference))
        ax.grid(
            linestyle=":",
            color=str(common["grid_color"]),
            alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
        )
    axes[0].set_ylabel(str(panel.get("y_label") or "Macro-F1 among accepted cells"))
    _apply_legend(axes[-1], panel)
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
    )
    return fig


def _plot_nonconsensus_threshold(
    confidence: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    reference = str(panel.get("focus_reference") or _primary_reference(common))
    method = str(
        panel.get("confidence_method")
        or common.get("confidence_focus_method")
        or _primary_method(common)
    )
    selected = (
        confidence.loc[
            confidence["reference"].eq(reference) & confidence["method"].eq(method)
        ]
        .sort_values("threshold")
        .copy()
    )
    selected.loc[
        selected["accepted_cells"].lt(int(panel["minimum_bin_cells"])), "macro_f1"
    ] = np.nan
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    second = ax.twinx()
    line_one = ax.plot(
        selected["threshold"],
        selected["macro_f1"],
        "o-",
        color=_method_color(common, method),
        linewidth=float(panel["line_width"]),
        markersize=float(panel["marker_size"]),
        label="Macro-F1 of accepted cells",
    )
    line_two = second.plot(
        selected["threshold"],
        selected["coverage_within_nonconsensus"] * 100,
        "s--",
        color="#6B7280",
        linewidth=float(panel["line_width"]) * 0.85,
        label="Non-consensus cells retained",
    )
    confidence_value = float(panel["selected_confidence"])
    ax.axvline(confidence_value, color="#D62728", linestyle=":", linewidth=1.4)
    ax.text(
        confidence_value + 0.012,
        ax.get_ylim()[0] + 0.02,
        f"selected {confidence_value:g}",
        rotation=90,
        color="#B22222",
        fontsize=float(common["annotation_font_size"]),
        va="bottom",
    )
    ax.set_xlabel(str(panel.get("x_label") or "Minimum model confidence"))
    ax.set_ylabel(
        str(panel.get("y_label") or "Macro-F1 among accepted predictions"),
        color=_method_color(common, method),
    )
    second.set_ylabel("Non-consensus cells retained (%)", color="#6B7280")
    ax.set_title(str(panel["title"]), pad=12, fontweight="bold")
    ax.grid(
        axis="y",
        linestyle=":",
        color=str(common["grid_color"]),
        alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
    )
    _apply_legend(
        ax,
        panel,
        handles=line_one + line_two,
        labels=[line.get_label() for line in line_one + line_two],
    )
    return fig


def _plot_confusion(
    confusion: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    reference = str(panel.get("focus_reference") or _primary_reference(common))
    requested = panel.get(
        "confusion_methods",
        [_primary_method(common), _secondary_method(common)],
    )
    methods = [
        method
        for method in list(dict.fromkeys(str(value) for value in requested))[:2]
        if method in set(confusion["method"].astype(str))
    ]
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(float(panel["width"]), float(panel["height"])),
        squeeze=False,
    )
    axes_list = list(axes.flat)
    for index, (ax, method) in enumerate(zip(axes_list, methods)):
        selected = confusion.loc[
            confusion["reference"].eq(reference) & confusion["method"].eq(method)
        ]
        pivot = selected.pivot(index="truth", columns="prediction", values="n").fillna(
            0
        )
        labels = _ordered_phenotypes(
            set(pivot.index) | set(pivot.columns), include_others=True
        )
        pivot = pivot.reindex(index=labels, columns=labels, fill_value=0)
        normalized = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0)
        sns.heatmap(
            normalized,
            cmap=str(panel["heatmap_cmap"]),
            vmin=float(panel["heatmap_vmin"]),
            vmax=float(panel["heatmap_vmax"]),
            square=bool(panel["heatmap_square"]),
            annot=bool(panel["heatmap_annotations"]),
            fmt=".2f",
            linewidths=float(panel["heatmap_linewidth"]),
            cbar=index == len(methods) - 1,
            cbar_kws=(
                {"label": _colorbar_label(panel, "Row fraction")}
                if index == len(methods) - 1
                else None
            ),
            ax=ax,
        )
        ax.set_title(_method_label(common, method))
        ax.set_xlabel(str(panel.get("x_label") or "Predicted"))
        ax.set_ylabel(str(panel.get("y_label") or "Manual") if index == 0 else "")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
    )
    return fig


def _plot_radar(
    multiple: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "cohen_kappa",
    ]
    metric_labels = list(
        panel.get(
            "radar_metric_labels",
            ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Cohen's κ"],
        )
    )
    methods = _ordered_methods(common)
    references = _ordered_references(common)
    angles = np.linspace(0, 2 * np.pi, len(metric_columns), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    fig, axes = plt.subplots(
        1,
        len(references),
        figsize=(float(panel["width"]), float(panel["height"])),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    axes = list(axes.flat)
    for ax, reference in zip(axes, references):
        subset = multiple.loc[multiple["reference"].eq(reference)].set_index("method")
        for method in methods:
            if method not in subset.index:
                continue
            values = subset.loc[method, metric_columns].astype(float).to_numpy()
            closed = np.concatenate([values, values[:1]])
            color = _method_color(common, method)
            ax.plot(
                closed_angles,
                closed,
                color=color,
                linewidth=float(panel["line_width"]),
                marker="o",
                markersize=float(panel["marker_size"]) * 0.5,
                label=_method_label(common, method),
            )
            ax.fill(
                closed_angles,
                closed,
                color=color,
                alpha=float(panel.get("radar_fill_alpha", 0.035)),
            )
        ax.set_xticks(angles)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_rlabel_position(18)
        ax.grid(
            alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
            linestyle=":",
        )
        n = int(subset["n"].iloc[0]) if not subset.empty else 0
        ax.set_title(
            f"{_reference_label(common, reference)}\nn={n:,}", pad=22, fontweight="bold"
        )
    kwargs = _legend_kwargs(str(panel["legend_location"]), int(panel["legend_columns"]))
    if kwargs is not None:
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            title=str(panel.get("legend_title") or "Phenotyping method"),
            **kwargs,
        )
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.42, top=0.82, bottom=0.08)
    return fig


def _plot_heldout_variation(
    summary: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    methods = _ordered_methods(common)
    reference = str(panel.get("focus_reference") or _primary_reference(common))
    selected = summary.loc[
        summary["reference"].eq(reference) & summary["method"].isin(methods)
    ].copy()
    groups = _group_values(common, selected)
    metrics = [("micro_f1", "Micro-F1"), ("weighted_f1", "Weighted-F1")]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(
        1, 2, figsize=(float(panel["width"]), float(panel["height"])), sharey=True
    )
    for ax, (metric, label) in zip(axes, metrics):
        _shade_methods(ax, methods, panel)
        for group_index, fold in enumerate(groups):
            rows = selected.loc[selected["fold"].eq(fold)].set_index("method")
            ax.plot(
                x,
                rows.reindex(methods)[metric].to_numpy(dtype=float),
                color=_group_color(common, fold, group_index),
                linewidth=float(panel["line_width"]),
                alpha=0.95,
            )
        pooled = selected.loc[selected["fold"].eq("Pooled")].set_index("method")
        for index, method in enumerate(methods):
            if method in pooled.index:
                ax.scatter(
                    index,
                    float(pooled.loc[method, metric]),
                    s=float(panel["marker_size"]) ** 2 * 1.4,
                    color=_method_color(common, method),
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_method_label(common, method) for method in methods],
            rotation=45,
            ha="right",
        )
        ax.set_xlim(-0.25, len(methods) - 0.75)
        if float(panel.get("box_aspect", 0.0)) > 0:
            ax.set_box_aspect(float(panel["box_aspect"]))
        ax.set_ylim(float(panel.get("y_min", 0.70)), float(panel.get("y_max", 1.0)))
        ax.set_ylabel(label if ax is axes[0] else "")
        ax.set_title(f"{label} across held-out images", fontweight="bold")
        ax.grid(
            axis="y",
            linestyle=":",
            color=str(common["grid_color"]),
            alpha=float(panel.get("grid_alpha", common["grid_alpha"])),
        )
    pooled_rows = selected.loc[selected["fold"].eq("Pooled")].set_index("method")
    kwargs = _legend_kwargs(str(panel["legend_location"]), int(panel["legend_columns"]))
    if kwargs is not None:
        fig.legend(
            handles=_heldout_legend_handles(common, pooled_rows, metrics, groups),
            title=str(panel.get("legend_title") or "Selected group"),
            **kwargs,
        )
    fig.suptitle(
        str(panel["title"]),
        fontsize=_figure_title_size(common, panel),
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.12, top=0.82, bottom=0.22)
    return fig


def _plot_learning_curve(
    learning: pd.DataFrame,
    summary: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    pooled = learning.loc[learning["fold"].eq("Pooled")]
    if pooled.empty:
        pooled = learning
    aggregate = (
        pooled.groupby("target_cells_per_class")["macro_f1"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("target_cells_per_class")
    )
    focus_reference = str(panel.get("focus_reference") or _primary_reference(common))
    focus_method = str(panel.get("learning_method") or _primary_method(common))
    curve_method = (
        "prediction_mr_ml_strict"
        if "prediction_mr_ml_strict" in _ordered_methods(common)
        else _primary_method(common)
    )
    reference = summary.loc[
        summary["reference"].eq(focus_reference) & summary["fold"].eq("Pooled")
    ]
    components = reference.loc[
        reference["method"].isin(common.get("component_methods", []))
    ]
    if components.empty:
        components = reference.loc[reference["method"].ne(focus_method)]
    focus_rows = reference.loc[reference["method"].eq(focus_method)]
    if focus_rows.empty:
        raise ValueError(
            "The selected learning-curve comparison method has no pooled score "
            "for the selected reference."
        )
    best = (
        components.sort_values("macro_f1", ascending=False).iloc[0]
        if not components.empty
        else None
    )
    full = focus_rows.iloc[0]
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    x = aggregate["target_cells_per_class"].to_numpy(float)
    color = _method_color(common, curve_method)
    ax.fill_between(
        x,
        aggregate["min"],
        aggregate["max"],
        color=color,
        alpha=float(panel.get("range_alpha", 0.16)),
    )
    ax.plot(
        x,
        aggregate["mean"],
        "o-",
        color=color,
        linewidth=float(panel["line_width"]),
        markersize=float(panel["marker_size"]),
        label=f"{_method_label(common, curve_method)} balanced-training mean",
    )
    if best is not None:
        ax.axhline(
            best["macro_f1"],
            color=_method_color(common, str(best["method"])),
            linestyle="--",
            linewidth=1.7,
            label=f"best component ({_method_label(common, str(best['method']))})",
        )
    ax.axhline(
        full["macro_f1"],
        color=_method_color(common, focus_method),
        linestyle=":",
        linewidth=1.8,
        label=f"{_method_label(common, focus_method)} with all available reference cells",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in x])
    ax.set_title(str(panel["title"]), pad=12, fontweight="bold")
    _finish_axis(
        ax,
        common,
        panel,
        default_x="Expert-labelled training cells targeted per phenotype",
        default_y="Held-out macro-F1",
    )
    _apply_legend(ax, panel)
    return fig


def _plot_confidence_bins(
    binned: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    reference: str,
) -> plt.Figure:
    methods = _ordered_methods(common)
    focus_method = str(
        panel.get("confidence_method")
        or common.get("confidence_focus_method")
        or _primary_method(common)
    )
    secondary_method = _primary_method(common)
    selected = binned.loc[binned["reference"].eq(reference)].copy()
    bin_table = (
        selected[["confidence_bin", "confidence_midpoint", "cells_in_bin"]]
        .drop_duplicates()
        .sort_values("confidence_midpoint")
    )
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    minimum = int(panel["minimum_bin_cells"])
    for method in methods:
        rows = (
            selected.loc[selected["method"].eq(method)]
            .sort_values("confidence_midpoint")
            .copy()
        )
        if rows.empty:
            continue
        rows.loc[rows["cells_in_bin"].lt(minimum), "macro_f1"] = np.nan
        ax.plot(
            rows["confidence_midpoint"],
            rows["macro_f1"],
            marker="o",
            markersize=float(panel["marker_size"])
            if method == focus_method
            else float(panel["marker_size"]) * 0.72,
            linewidth=float(panel["line_width"])
            * (
                1.4
                if method == focus_method
                else (1.0 if method == secondary_method else 0.68)
            ),
            linestyle="--"
            if method == secondary_method and method != focus_method
            else "-",
            color=_method_color(common, method),
            alpha=1.0 if method in {secondary_method, focus_method} else 0.82,
            label=_method_label(common, method),
        )
    observed = bin_table.loc[bin_table["cells_in_bin"].gt(0)]
    ax.set_xticks(observed["confidence_midpoint"])
    ax.set_xticklabels(
        [
            f"{row.confidence_bin}\nn={int(row.cells_in_bin):,}{'*' if int(row.cells_in_bin) < minimum else ''}"
            for row in observed.itertuples(index=False)
        ]
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    reference_short = _reference_label(common, reference)
    ax.set_title(str(panel["title"]), pad=12, fontweight="bold")
    _finish_axis(
        ax,
        common,
        panel,
        default_x=(
            f"{_method_label(common, focus_method)} confidence bin "
            f"(same {reference_short} cells for every line)"
        ),
        default_y="Macro-F1 within confidence bin",
    )
    _apply_legend(ax, panel)
    ax.text(
        0.0,
        -0.24,
        f"Only non-consensus cells are included; n is shared across methods. *F1 not plotted for bins with n<{minimum}.",
        transform=ax.transAxes,
        fontsize=float(common["annotation_font_size"]),
        color=_muted_canvas_foreground(common),
        va="top",
    )
    return fig


def _plot_pairwise(
    matrix_frame: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> plt.Figure:
    matrix = matrix_frame.copy()
    if "Method" in matrix.columns:
        matrix = matrix.set_index("Method")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    methods = [method for method in _ordered_methods(common) if method in matrix.index]
    matrix = matrix.reindex(index=methods, columns=methods)
    display_labels = {method: _method_label(common, method) for method in methods}
    matrix = matrix.rename(index=display_labels, columns=display_labels)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(float(panel["width"]), float(panel["height"])))
    sns.heatmap(
        matrix,
        mask=mask,
        annot=bool(panel["heatmap_annotations"]),
        fmt=".1f",
        annot_kws={"size": _heatmap_annotation_size(common, panel)},
        cmap=str(panel["heatmap_cmap"]),
        linewidths=float(panel["heatmap_linewidth"]),
        linecolor=_canvas_background(common),
        vmin=float(panel.get("heatmap_vmin", 50.0)),
        vmax=float(panel.get("heatmap_vmax", 100.0)),
        square=bool(panel["heatmap_square"]),
        cbar_kws={
            "label": _colorbar_label(panel, "Matching phenotype assignments (%)")
        },
        ax=ax,
    )
    ax.set_title(str(panel["title"]), fontweight="bold", pad=12)
    ax.set_xlabel(str(panel.get("x_label") or ""))
    ax.set_ylabel(str(panel.get("y_label") or ""))
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    return fig


def _build_paper_figure_from_tables(
    panel_id: str,
    tables: Mapping[str, pd.DataFrame],
    common: Mapping[str, Any] | None = None,
    panel: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Build one panel from either generic or publication source tables."""
    if panel_id not in PANEL_BY_ID:
        raise KeyError(f"Unknown Figure 3 panel: {panel_id}")
    spec = PANEL_BY_ID[panel_id]
    common_settings = {**default_common_settings(), **dict(common or {})}
    panel_settings = {**default_panel_settings(spec), **dict(panel or {})}

    def table(name: str) -> pd.DataFrame:
        return tables[name]

    with plt.rc_context(_style_context(common_settings)):
        if panel_id == "main_workflow":
            figure = _plot_workflow(common_settings, panel_settings)
        elif panel_id == "main_heldout_macro":
            figure = _plot_heldout_macro(
                table("T10_performance_summary.csv"), common_settings, panel_settings
            )
        elif panel_id == "main_consensus_tradeoff":
            figure = _plot_consensus_tradeoff(
                table("T05_consensus_and_leave_one_method_out.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "main_ref2_metrics":
            figure = _plot_ref2_metrics(
                table("T10_performance_summary.csv"), common_settings, panel_settings
            )
        elif panel_id == "main_per_phenotype_f1":
            figure = _plot_combined_per_class(
                table("T11_per_class_performance.csv"), common_settings, panel_settings
            )
        elif panel_id in {"main_spatial", "supp_spatial"}:
            figure = _plot_spatial_validation(
                table("T06_strict_oof_predictions.csv"), common_settings, panel_settings
            )
        elif panel_id == "supp_reference_distribution":
            figure = _plot_reference_distribution(
                table("T03_reference_phenotype_distribution.csv"),
                table("T04_reference_locations.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "supp_weights":
            figure = _plot_weight_heatmap(
                table("T08_fold_specific_f1_weights.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id in {
            "supp_ref1_f1",
            "supp_ref2_f1",
            "supp_ref1_precision",
            "supp_ref2_precision",
            "supp_ref1_recall",
            "supp_ref2_recall",
        }:
            reference = str(
                panel_settings.get("focus_reference")
                or _primary_reference(common_settings)
            )
            metric = str(
                panel_settings.get("class_metric")
                or (
                    "precision"
                    if "precision" in panel_id
                    else ("recall" if "recall" in panel_id else "f1")
                )
            )
            figure = _plot_single_per_class(
                table("T11_per_class_performance.csv"),
                common_settings,
                panel_settings,
                reference=reference,
                metric=metric,
            )
        elif panel_id == "supp_leave_one_out":
            figure = _plot_leave_one_out(
                table("T05_consensus_and_leave_one_method_out.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "supp_routes":
            figure = _plot_routes(
                table("T12_direct_vs_nonconsensus_routes.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "supp_human":
            figure = _plot_human_variation(
                table("T17_human_agreement_summary.csv"),
                table("T18_human_confusion_counts.csv"),
                table("T19_human_consensus_enrichment.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "supp_confidence_all":
            figure = _plot_confidence_all(
                table("T14_confidence_all_routes.csv"), common_settings, panel_settings
            )
        elif panel_id == "supp_nonconsensus_threshold":
            figure = _plot_nonconsensus_threshold(
                table("T15_confidence_nonconsensus_only.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "supp_confusion":
            figure = _plot_confusion(
                table("T13_confusion_counts.csv"), common_settings, panel_settings
            )
        elif panel_id == "supp_radar":
            figure = _plot_radar(
                table("T21_multiple_performance_comparison.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id == "supp_heldout_variation":
            figure = _plot_heldout_variation(
                table("T10_performance_summary.csv"), common_settings, panel_settings
            )
        elif panel_id == "supp_learning":
            figure = _plot_learning_curve(
                table("T16_annotation_learning_curve.csv"),
                table("T10_performance_summary.csv"),
                common_settings,
                panel_settings,
            )
        elif panel_id in {"supp_bins_ref1", "supp_bins_ref2"}:
            reference = str(
                panel_settings.get("focus_reference")
                or _primary_reference(common_settings)
            )
            figure = _plot_confidence_bins(
                table("T15b_confidence_bins_same_cells.csv"),
                common_settings,
                panel_settings,
                reference=reference,
            )
        elif panel_id == "supp_pairwise":
            figure = _plot_pairwise(
                table("T23_Ref2_pairwise_component_agreement_matrix.csv"),
                common_settings,
                panel_settings,
            )
        else:  # pragma: no cover - PANEL_BY_ID and dispatcher are kept in sync.
            raise KeyError(panel_id)
    _apply_publication_canvas(figure, common_settings)
    return figure


def build_paper_figure(
    panel_id: str,
    source_root: str | Path,
    common: Mapping[str, Any] | None = None,
    panel: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Build a panel from a legacy Figure 3 source-data package."""
    tables = load_panel_tables(panel_id, source_root)
    return _build_paper_figure_from_tables(panel_id, tables, common, panel)


def build_generic_paper_figure(
    panel_id: str,
    frame: pd.DataFrame,
    common: Mapping[str, Any],
    panel: Mapping[str, Any] | None = None,
    analysis_tables: Mapping[str, Any] | None = None,
) -> plt.Figure:
    """Build one universal panel directly from mapped unified-CSV columns."""
    tables = build_generic_panel_tables(
        frame,
        common,
        panel_id,
        analysis_tables=analysis_tables,
    )
    return _build_paper_figure_from_tables(panel_id, tables, common, panel)


def save_paper_figure(
    figure: plt.Figure,
    output_dir: str | Path,
    stem: str,
    formats: list[str],
    *,
    dpi: int = 300,
    settings: Mapping[str, Any] | None = None,
) -> tuple[list[Path], bytes]:
    """Save one panel with a collision-resistant Option 2 filename."""
    target = Path(output_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_stem = sanitize_filename(stem) or "figure3_panel"
    base = target / f"f1__option2__{safe_stem}__{run_id}"
    preview = BytesIO()
    figure.savefig(
        preview,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
        transparent=False,
    )
    preview_bytes = preview.getvalue()
    saved: list[Path] = []
    normalized_formats = {str(value).upper() for value in formats}
    if "PNG" in normalized_formats:
        path = base.with_suffix(".png")
        path.write_bytes(preview_bytes)
        saved.append(path)
    if "SVG" in normalized_formats:
        path = base.with_suffix(".svg")
        figure.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
            transparent=False,
        )
        saved.append(path)
    if settings is not None:
        path = base.with_suffix(".json")
        path.write_text(
            json.dumps(dict(settings), indent=2, default=str) + "\n", encoding="utf-8"
        )
        saved.append(path)
    return saved, preview_bytes


def _initialize_state(
    key: str,
    default: Any,
    cached_params: Mapping[str, Any],
) -> None:
    if key not in st.session_state:
        st.session_state[key] = cached_params.get(key, default)


def _persist_paper_settings(
    project_root: Path,
    cached_params: dict[str, Any],
) -> None:
    for key, value in st.session_state.items():
        if isinstance(key, str) and key.startswith(
            ("viz_f1_paper_", "viz_f1_strict_", "viz_f1_shared_")
        ):
            cached_params[key] = value
    save_cached_params(project_root, cached_params)
    try:
        from spacemap_gui.persistence import save_active_project_state

        save_active_project_state("tab9_visualization")
    except Exception:
        pass


def _column_state_key(kind: str, column: str, setting: str) -> str:
    """Return a stable widget key even when two columns sanitize identically."""
    slug = sanitize_filename(str(column)) or "column"
    digest = hashlib.sha1(str(column).encode("utf-8")).hexdigest()[:8]
    return f"viz_f1_paper_{kind}_{slug}_{digest}_{setting}"


def _value_state_key(kind: str, value: str, setting: str) -> str:
    slug = sanitize_filename(str(value)) or "value"
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
    return f"viz_f1_paper_{kind}_{slug}_{digest}_{setting}"


def _display_name(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def _parse_float_values(value: Any, fallback: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = str(value).replace(";", ",").split(",")
    parsed: list[float] = []
    for candidate in candidates:
        try:
            number = float(candidate)
        except (TypeError, ValueError):
            continue
        if 0.0 <= number <= 1.0:
            parsed.append(number)
    return sorted(set(parsed)) or fallback


def _parse_positive_int_values(value: Any, fallback: list[int]) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = str(value).replace(";", ",").split(",")
    parsed: list[int] = []
    for candidate in candidates:
        try:
            number = int(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if number > 0:
            parsed.append(number)
    return sorted(set(parsed)) or list(fallback)


def _parse_labels(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = str(value).replace(";", ",").split(",")
    return {str(item).strip().lower() for item in candidates if str(item).strip()}


def _common_settings_from_state() -> dict[str, Any]:
    defaults = default_common_settings()
    methods = [
        str(value) for value in st.session_state.get("viz_f1_paper_method_columns", [])
    ]
    references = [
        str(value)
        for value in st.session_state.get("viz_f1_paper_reference_columns", [])
    ]
    components = [
        str(value)
        for value in st.session_state.get("viz_f1_paper_component_methods", methods)
        if str(value) in methods
    ]
    method_colors = {
        method: str(
            st.session_state.get(
                _column_state_key("method", method, "color"),
                _method_default(method, 0)[1],
            )
        )
        for method in methods
    }
    method_labels = {
        method: str(
            st.session_state.get(
                _column_state_key("method", method, "label"), _display_name(method)
            )
        )
        for method in methods
    }
    reference_colors = {
        reference: str(
            st.session_state.get(
                _column_state_key("reference", reference, "color"),
                _reference_default(reference, 0)[1],
            )
        )
        for reference in references
    }
    reference_labels = {
        reference: str(
            st.session_state.get(
                _column_state_key("reference", reference, "label"),
                _reference_default(reference, 0)[0],
            )
        )
        for reference in references
    }
    confidence_columns: dict[str, str] = {}
    for method in methods:
        selected = str(
            st.session_state.get(
                _column_state_key("method", method, "confidence"), "(None)"
            )
        )
        if selected and selected != "(None)":
            confidence_columns[method] = selected
    groups = [
        str(value) for value in st.session_state.get("viz_f1_paper_group_values", [])
    ]
    groups = [
        *[value for value in HELDOUT_IMAGE_ORDER if value in groups],
        *[value for value in groups if value not in HELDOUT_IMAGE_ORDER],
    ]
    phenotypes = [
        str(value)
        for value in st.session_state.get("viz_f1_paper_phenotype_values", [])
    ]
    group_column = str(st.session_state.get("viz_f1_paper_group_column", "(None)"))
    x_column = str(st.session_state.get("viz_f1_paper_x_column", "(None)"))
    y_column = str(st.session_state.get("viz_f1_paper_y_column", "(None)"))
    primary_method = str(st.session_state.get("viz_f1_paper_primary_method", ""))
    secondary_method = str(st.session_state.get("viz_f1_paper_secondary_method", ""))
    primary_reference = str(st.session_state.get("viz_f1_paper_primary_reference", ""))
    secondary_reference = str(
        st.session_state.get("viz_f1_paper_secondary_reference", "")
    )
    return {
        "font_family": st.session_state.get(
            "viz_f1_paper_font_family", defaults["font_family"]
        ),
        "base_font_size": st.session_state.get(
            "viz_f1_paper_base_font_size", defaults["base_font_size"]
        ),
        "title_font_size": st.session_state.get(
            "viz_f1_paper_title_font_size", defaults["title_font_size"]
        ),
        "subplot_title_font_size": st.session_state.get(
            "viz_f1_paper_subplot_title_font_size",
            defaults["subplot_title_font_size"],
        ),
        "axis_font_size": st.session_state.get(
            "viz_f1_paper_axis_font_size", defaults["axis_font_size"]
        ),
        "tick_font_size": st.session_state.get(
            "viz_f1_paper_tick_font_size", defaults["tick_font_size"]
        ),
        "annotation_font_size": st.session_state.get(
            "viz_f1_paper_annotation_font_size", defaults["annotation_font_size"]
        ),
        "dpi": st.session_state.get("viz_f1_paper_dpi", defaults["dpi"]),
        "background_color": st.session_state.get(
            "viz_f1_paper_background_color", defaults["background_color"]
        ),
        "grid_color": st.session_state.get(
            "viz_f1_paper_grid_color", defaults["grid_color"]
        ),
        "grid_alpha": st.session_state.get(
            "viz_f1_paper_grid_alpha", defaults["grid_alpha"]
        ),
        "method_order": methods,
        "method_colors": method_colors,
        "method_labels": method_labels,
        "reference_order": references,
        "reference_colors": reference_colors,
        "reference_labels": reference_labels,
        "component_methods": components,
        "confidence_columns": confidence_columns,
        "confidence_thresholds": _parse_float_values(
            st.session_state.get(
                "viz_f1_paper_confidence_thresholds",
                "0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95",
            ),
            defaults["confidence_thresholds"],
        ),
        "excluded_labels": _parse_labels(
            st.session_state.get(
                "viz_f1_paper_excluded_labels", ", ".join(sorted(INVALID_LABELS))
            )
        ),
        "consensus_min_agree": int(
            st.session_state.get("viz_f1_paper_consensus_min_agree", 3)
        ),
        "primary_method": primary_method
        if primary_method in methods
        else (methods[0] if methods else ""),
        "secondary_method": secondary_method
        if secondary_method in methods
        else (methods[1] if len(methods) > 1 else (methods[0] if methods else "")),
        "primary_reference": primary_reference
        if primary_reference in references
        else (references[0] if references else ""),
        "secondary_reference": secondary_reference
        if secondary_reference in references
        else (
            references[1]
            if len(references) > 1
            else (references[0] if references else "")
        ),
        "group_column": "" if group_column == "(None)" else group_column,
        "group_order": groups,
        "group_labels": {
            group: str(
                st.session_state.get(_value_state_key("group", group, "label"), group)
            )
            for group in groups
        },
        "group_colors": {
            group: str(
                st.session_state.get(
                    _value_state_key("group", group, "color"),
                    _group_color(defaults, group, index),
                )
            )
            for index, group in enumerate(groups)
        },
        "phenotype_colors": {
            phenotype: str(
                st.session_state.get(
                    _value_state_key("phenotype", phenotype, "color"),
                    _phenotype_default_color(phenotype),
                )
            )
            for phenotype in phenotypes
        },
        "x_column": "" if x_column == "(None)" else x_column,
        "y_column": "" if y_column == "(None)" else y_column,
        "learning_curve_csv": str(
            st.session_state.get("viz_f1_paper_learning_curve_csv", "")
        ).strip(),
        "generated_method_labels": {
            "prediction_mr_ml_strict": str(
                st.session_state.get("viz_f1_paper_generated_mr_label", "ML-MR")
            ),
            "prediction_consensus_ml_strict": str(
                st.session_state.get(
                    "viz_f1_paper_generated_consensus_label", "Consensus-ML"
                )
            ),
        },
        "generated_method_colors": {
            "prediction_mr_ml_strict": str(
                st.session_state.get("viz_f1_paper_generated_mr_color", "#7B2CBF")
            ),
            "prediction_consensus_ml_strict": str(
                st.session_state.get(
                    "viz_f1_paper_generated_consensus_color", "#008C8C"
                )
            ),
        },
        "formats": st.session_state.get("viz_f1_paper_formats", defaults["formats"]),
    }


def _panel_key(panel_id: str, setting: str) -> str:
    return f"viz_f1_paper_panel_{panel_id}_{setting}"


def _panel_settings_from_state(spec: PaperPanelSpec) -> dict[str, Any]:
    defaults = default_panel_settings(spec)
    result = {
        setting: st.session_state.get(_panel_key(spec.panel_id, setting), default)
        for setting, default in defaults.items()
    }
    if spec.panel_id == "main_ref2_metrics":
        result["metric_colors"] = [
            st.session_state.get(
                _panel_key(spec.panel_id, f"metric_color_{index}"), color
            )
            for index, color in enumerate(["#24445C", "#6D8FA6", "#B5CBD8"])
        ]
    if spec.panel_id == "main_workflow":
        workflow_defaults = {
            "components": "Selected complementary\ncomponent phenotype calls",
            "mr_source": "Manual-reference cells\nfrom non-held-out groups",
            "consensus_source": "Agreement-derived labels\nfrom non-held-out groups",
            "mr_training": "Learn phenotype-specific\ncomponent weights and\nmarker patterns",
            "consensus_training": "Train on consensus cells;\npredict the remaining cells",
            "holdout": "Hold out one complete\nmapped image group at a time",
            "comparison": "Compare out-of-fold calls\nwith manual references",
            "footer": "No cells from the held-out group enter method weighting, manual-reference training, consensus pseudo-label training, or model fitting.",
        }
        result["workflow_text"] = {
            name: st.session_state.get(
                _panel_key(spec.panel_id, f"workflow_{name}"), value
            )
            for name, value in workflow_defaults.items()
        }
    return result


def _render_shared_settings_form(
    project_root: Path,
    cached_params: dict[str, Any],
    output_dir_default: str,
) -> None:
    defaults = default_common_settings()
    default_source = project_root / "test_codes" / "Fig3_code_results" / "results_v2"
    option1_output = st.session_state.get(
        "viz_f1_output_dir",
        cached_params.get("viz_f1_output_dir"),
    )
    default_output = (
        Path(str(option1_output)).expanduser()
        if option1_output
        else Path(output_dir_default) / "F1_validation"
    )
    simple_defaults = {
        "viz_f1_paper_source_dir": str(default_source),
        "viz_f1_paper_output_dir": str(default_output),
        "viz_f1_paper_formats": defaults["formats"],
        "viz_f1_paper_save_settings_json": True,
        "viz_f1_paper_font_family": defaults["font_family"],
        "viz_f1_paper_base_font_size": defaults["base_font_size"],
        "viz_f1_paper_title_font_size": defaults["title_font_size"],
        "viz_f1_paper_subplot_title_font_size": defaults[
            "subplot_title_font_size"
        ],
        "viz_f1_paper_axis_font_size": defaults["axis_font_size"],
        "viz_f1_paper_tick_font_size": defaults["tick_font_size"],
        "viz_f1_paper_annotation_font_size": defaults["annotation_font_size"],
        "viz_f1_paper_dpi": defaults["dpi"],
        "viz_f1_paper_background_color": defaults["background_color"],
        "viz_f1_paper_grid_color": defaults["grid_color"],
        "viz_f1_paper_grid_alpha": defaults["grid_alpha"],
        "viz_f1_paper_method_order": defaults["method_order"],
    }
    for key, value in simple_defaults.items():
        _initialize_state(key, value, cached_params)
    for method in METHOD_ORDER:
        slug = sanitize_filename(method)
        _initialize_state(
            f"viz_f1_paper_method_color_{slug}",
            defaults["method_colors"][method],
            cached_params,
        )
        _initialize_state(
            f"viz_f1_paper_method_label_{slug}",
            defaults["method_labels"][method],
            cached_params,
        )
    for reference, color in defaults["reference_colors"].items():
        slug = sanitize_filename(reference)
        _initialize_state(f"viz_f1_paper_reference_color_{slug}", color, cached_params)
    for reference, label in defaults["reference_labels"].items():
        slug = sanitize_filename(reference)
        _initialize_state(f"viz_f1_paper_reference_label_{slug}", label, cached_params)
    for group, color in defaults["heldout_colors"].items():
        _initialize_state(
            f"viz_f1_paper_heldout_color_{sanitize_filename(group)}",
            color,
            cached_params,
        )
    for phenotype, color in defaults["phenotype_colors"].items():
        _initialize_state(
            f"viz_f1_paper_phenotype_color_{sanitize_filename(phenotype)}",
            color,
            cached_params,
        )

    st.markdown("### Shared Option 2 settings")
    st.caption(
        "These values are shared by every paper panel. They are committed only when "
        "you press Save Shared Settings; no figure is rendered by this form."
    )
    with st.form("viz_f1_paper_shared_form", clear_on_submit=False):
        path_columns = st.columns(2)
        with path_columns[0]:
            st.text_input(
                "Figure 3 results or source_data directory",
                key="viz_f1_paper_source_dir",
                help="Point to either results_v2 or its source_data subdirectory.",
            )
        with path_columns[1]:
            st.text_input("F1 output directory", key="viz_f1_paper_output_dir")
        export_columns = st.columns(3)
        with export_columns[0]:
            st.multiselect("Output formats", ["PNG", "SVG"], key="viz_f1_paper_formats")
        with export_columns[1]:
            st.number_input(
                "PNG resolution (DPI)",
                min_value=72,
                max_value=1200,
                step=25,
                key="viz_f1_paper_dpi",
            )
        with export_columns[2]:
            st.checkbox(
                "Save settings JSON beside each panel",
                key="viz_f1_paper_save_settings_json",
            )

        with st.expander("Typography and canvas", expanded=True):
            columns = st.columns(4)
            with columns[0]:
                st.selectbox(
                    "Font family",
                    FONT_FAMILIES,
                    key="viz_f1_paper_font_family",
                    help="Typeface applied to all generated Option 2 figures.",
                )
                st.color_picker(
                    "Canvas color",
                    key="viz_f1_paper_background_color",
                    help="Background color of both the figure and plotting axes.",
                )
            with columns[1]:
                st.number_input(
                    "Base font size",
                    min_value=5.0,
                    max_value=30.0,
                    step=0.5,
                    key="viz_f1_paper_base_font_size",
                    help="Default text size used when a more specific title, axis, tick, or annotation size is not set.",
                )
                st.number_input(
                    "Figure/suptitle size",
                    min_value=6.0,
                    max_value=40.0,
                    step=0.5,
                    key="viz_f1_paper_title_font_size",
                    help="Font size used for figure-level titles above multi-panel plots.",
                )
                st.number_input(
                    "Axes/subplot title size",
                    min_value=6.0,
                    max_value=40.0,
                    step=0.5,
                    key="viz_f1_paper_subplot_title_font_size",
                    help="Reference-code default is 11 pt for titles inside plotting axes.",
                )
            with columns[2]:
                st.number_input(
                    "Axis-label size",
                    min_value=5.0,
                    max_value=30.0,
                    step=0.5,
                    key="viz_f1_paper_axis_font_size",
                    help="Font size for X- and Y-axis labels.",
                )
                st.number_input(
                    "Tick-label size",
                    min_value=4.0,
                    max_value=24.0,
                    step=0.5,
                    key="viz_f1_paper_tick_font_size",
                    help="Font size for category names and numeric tick labels.",
                )
            with columns[3]:
                st.number_input(
                    "Annotation size",
                    min_value=4.0,
                    max_value=24.0,
                    step=0.5,
                    key="viz_f1_paper_annotation_font_size",
                    help="Font size for values, notes, sample counts, and other labels drawn inside a figure.",
                )
                st.color_picker(
                    "Grid color",
                    key="viz_f1_paper_grid_color",
                    help="Color of guide/grid lines behind plotted data.",
                )
                st.slider(
                    "Grid opacity",
                    0.0,
                    1.0,
                    step=0.05,
                    key="viz_f1_paper_grid_alpha",
                    help="0 hides grid lines; 1 makes them fully opaque.",
                )

        with st.expander("Method order, labels, and colors", expanded=True):
            st.multiselect(
                "Methods shown and their order",
                METHOD_ORDER,
                key="viz_f1_paper_method_order",
                help="Removing a method removes it from all compatible Option 2 panels.",
            )
            columns = st.columns(3)
            for index, method in enumerate(METHOD_ORDER):
                slug = sanitize_filename(method)
                with columns[index % 3]:
                    st.text_input(
                        f"{method} display label",
                        key=f"viz_f1_paper_method_label_{slug}",
                    )
                    st.color_picker(
                        f"{method} color", key=f"viz_f1_paper_method_color_{slug}"
                    )

        with st.expander("Reference and held-out-image colors", expanded=False):
            reference_columns = st.columns(2)
            for index, reference in enumerate(defaults["reference_colors"]):
                slug = sanitize_filename(reference)
                with reference_columns[index % 2]:
                    if reference in defaults["reference_labels"]:
                        st.text_input(
                            f"{reference} display label",
                            key=f"viz_f1_paper_reference_label_{slug}",
                        )
                    st.color_picker(
                        f"{reference} color", key=f"viz_f1_paper_reference_color_{slug}"
                    )
            heldout_columns = st.columns(3)
            for index, group in enumerate(HELDOUT_IMAGE_ORDER):
                with heldout_columns[index]:
                    st.color_picker(
                        f"Held-out {HELDOUT_IMAGE_LABELS[group]}",
                        key=f"viz_f1_paper_heldout_color_{sanitize_filename(group)}",
                    )

        with st.expander("Spatial phenotype colors", expanded=False):
            phenotype_columns = st.columns(4)
            for index, phenotype in enumerate(PHENOTYPE_ORDER):
                with phenotype_columns[index % 4]:
                    st.color_picker(
                        phenotype,
                        key=f"viz_f1_paper_phenotype_color_{sanitize_filename(phenotype)}",
                    )

        save_shared = st.form_submit_button(
            "Save Shared Option 2 Settings", type="primary"
        )
    if save_shared:
        if not st.session_state.get("viz_f1_paper_formats"):
            st.error("Select PNG, SVG, or both before generating panels.")
        else:
            _persist_paper_settings(project_root, cached_params)
            st.success("Shared Option 2 settings saved. No figure was rendered.")


KNOWN_METHOD_COLUMN_SETTINGS = {
    "phenotype_leiden": ("Leiden", "#1f77b4"),
    "phenotype_som": ("SOM", "#d62728"),
    "phenotype_vortex": ("Vortex", "#9467bd"),
    "phenotype_threshold": ("RESOLVE", "#2ca02c"),
    "prediction_mr_ml_strict": ("ML-MR", "#7B2CBF"),
    "prediction_consensus_ml_strict": ("Consensus-ML", "#008C8C"),
}


def _known_method_column_setting(column: str) -> tuple[str, str] | None:
    """Match reference-code methods despite harmless column prefixes/suffixes."""
    lowered = str(column).strip().lower()
    exact = {
        name.lower(): setting for name, setting in KNOWN_METHOD_COLUMN_SETTINGS.items()
    }
    if lowered in exact:
        return exact[lowered]
    tokens = {
        token
        for token in lowered.replace("-", "_").replace(" ", "_").split("_")
        if token
    }
    compact = "".join(character for character in lowered if character.isalnum())
    if "consensusml" in compact or "mlconsensus" in compact:
        return "Consensus-ML", METHOD_COLORS["Consensus-ML"]
    if "mrml" in compact or "mlmr" in compact:
        return "ML-MR", METHOD_COLORS["ML-MR"]
    if "leiden" in tokens:
        return "Leiden", METHOD_COLORS["Leiden"]
    if "som" in tokens:
        return "SOM", METHOD_COLORS["SOM"]
    if "vortex" in tokens:
        return "Vortex", METHOD_COLORS["Vortex"]
    if "resolve" in tokens or (
        "threshold" in tokens and "phenotype" in tokens
    ):
        return "RESOLVE", METHOD_COLORS["RESOLVE"]
    return None


def _default_method_columns(columns: list[str]) -> list[str]:
    known = [column for column in columns if _known_method_column_setting(column)]
    if known:
        return known
    candidates = []
    for column in columns:
        lowered = column.lower()
        is_prediction = any(
            token in lowered
            for token in ("phenotype", "prediction", "cluster", "class")
        )
        is_reference = any(
            token in lowered for token in ("reference", "ref", "manual", "truth")
        )
        is_metadata = any(
            token in lowered for token in ("confidence", "probability", "score")
        )
        if is_prediction and not is_reference and not is_metadata:
            candidates.append(column)
    return candidates[:8]


def _default_reference_columns(columns: list[str]) -> list[str]:
    known_order = ["phenotype_ref1", "phenotype_ref2", "phenotype_bd", "phenotype_ap"]
    known = [column for column in known_order if column in columns]
    if known:
        return known
    candidates = []
    for column in columns:
        lowered = column.lower()
        if (
            any(
                token in lowered
                for token in ("reference", "manual", "ground_truth", "groundtruth")
            )
            or lowered.startswith("ref")
            or "_ref" in lowered
        ):
            if not any(token in lowered for token in ("confidence", "score")):
                candidates.append(column)
    return candidates[:4]


def _default_column(columns: list[str], candidates: list[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return "(None)"


def _method_default(column: str, index: int) -> tuple[str, str]:
    del index
    known = _known_method_column_setting(column)
    if known is not None:
        return known
    return _display_name(column), _stable_named_color(column)


def method_style_default(column: str, index: int = 0) -> tuple[str, str]:
    """Public shared default used by both F1 figure options."""
    return _method_default(column, index)


def _known_reference_setting(column: str) -> tuple[str, str] | None:
    lowered = str(column).strip().lower()
    if lowered in REFERENCE_LABELS:
        return REFERENCE_LABELS[lowered], REFERENCE_COLORS[lowered]
    tokens = {
        token
        for token in lowered.replace("-", "_").replace(" ", "_").split("_")
        if token
    }
    compact = "".join(character for character in lowered if character.isalnum())
    if "ref1" in compact:
        return REFERENCE_LABELS["phenotype_ref1"], REFERENCE_COLORS["phenotype_ref1"]
    if "ref2" in compact:
        return REFERENCE_LABELS["phenotype_ref2"], REFERENCE_COLORS["phenotype_ref2"]
    if "bd" in tokens:
        return REFERENCE_LABELS["phenotype_bd"], REFERENCE_COLORS["phenotype_bd"]
    if "ap" in tokens:
        return REFERENCE_LABELS["phenotype_ap"], REFERENCE_COLORS["phenotype_ap"]
    return None


def _reference_default(column: str, index: int) -> tuple[str, str]:
    del index
    known = _known_reference_setting(column)
    if known is not None:
        return known
    return _display_name(column), _stable_named_color(f"reference::{column}")


def _valid_multiselect_state(
    key: str,
    options: list[str],
    default: list[str],
    cached_params: Mapping[str, Any],
) -> None:
    _initialize_state(key, default, cached_params)
    value = st.session_state.get(key, default)
    if not isinstance(value, (list, tuple)):
        value = default
    st.session_state[key] = [str(item) for item in value if str(item) in options]


def _valid_select_state(
    key: str,
    options: list[str],
    default: str,
    cached_params: Mapping[str, Any],
) -> None:
    _initialize_state(key, default, cached_params)
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]


# The file signature is part of each cache key, so an edited or re-exported
# unified table is re-read instead of serving the previous file's rows. Column
# lookups go through dataio, which uses the Parquet fast copy when it is current.
@st.cache_data(show_spinner=False, max_entries=4)
def _read_unified_columns_cached(csv_path: str, columns: tuple[str, ...], signature) -> pd.DataFrame:
    from spacemap import dataio

    return dataio.load(csv_path, columns=list(columns))


@st.cache_data(show_spinner=False, max_entries=1)
def _read_unified_frame_cached(csv_path: str, signature) -> pd.DataFrame:
    # The whole numeric table feeds model training, so it keeps the pandas
    # reader: pyarrow can type an all-blank column as text where pandas reads
    # float, and parses a few values differently in the last binary digit.
    return pd.read_csv(csv_path, low_memory=False)


def _read_unified_columns(csv_path: str, columns: tuple[str, ...]) -> pd.DataFrame:
    from spacemap_gui.ui_utils import file_signature

    return _read_unified_columns_cached(csv_path, columns, file_signature(csv_path))


def _read_unified_frame(csv_path: str) -> pd.DataFrame:
    from spacemap_gui.ui_utils import file_signature

    return _read_unified_frame_cached(csv_path, file_signature(csv_path))


def _unique_text_values(
    csv_path: str,
    columns: list[str],
    *,
    maximum: int = 80,
) -> tuple[list[str], bool]:
    if not columns:
        return [], False
    try:
        selected = _read_unified_columns(csv_path, tuple(dict.fromkeys(columns)))
    except Exception:
        return [], False
    values: list[str] = []
    for column in columns:
        if column not in selected.columns:
            continue
        values.extend(selected[column].dropna().astype(str).str.strip().tolist())
    unique = [value for value in dict.fromkeys(values) if value]
    return unique[:maximum], len(unique) > maximum


def _guess_confidence_column(method: str, columns: list[str]) -> str:
    exact = [
        f"{method}_confidence",
        f"confidence_{method}",
        method.replace("prediction_", "confidence_"),
        method.replace("phenotype_", "confidence_"),
    ]
    lowered = {column.lower(): column for column in columns}
    for candidate in exact:
        if candidate.lower() in lowered and lowered[candidate.lower()] != method:
            return lowered[candidate.lower()]
    tokens = [
        token
        for token in method.lower()
        .replace("prediction", "")
        .replace("phenotype", "")
        .split("_")
        if token
    ]
    for column in columns:
        name = column.lower()
        if (
            any(
                word in name
                for word in ("confidence", "probability", "prob", "certainty")
            )
            and tokens
            and all(token in name for token in tokens[:2])
        ):
            return column
    return "(None)"


def _restore_reproducible_style_defaults(
    methods: list[str],
    references: list[str],
    phenotypes: list[str],
    groups: list[str],
) -> None:
    """Restore the complete ``fig3_code_v2.py`` visual preset."""
    common_defaults = default_common_settings()
    common_state_keys = {
        "font_family": "viz_f1_paper_font_family",
        "base_font_size": "viz_f1_paper_base_font_size",
        "title_font_size": "viz_f1_paper_title_font_size",
        "subplot_title_font_size": "viz_f1_paper_subplot_title_font_size",
        "axis_font_size": "viz_f1_paper_axis_font_size",
        "tick_font_size": "viz_f1_paper_tick_font_size",
        "annotation_font_size": "viz_f1_paper_annotation_font_size",
        "dpi": "viz_f1_paper_dpi",
        "background_color": "viz_f1_paper_background_color",
        "grid_color": "viz_f1_paper_grid_color",
        "grid_alpha": "viz_f1_paper_grid_alpha",
    }
    for setting, state_key in common_state_keys.items():
        st.session_state[state_key] = common_defaults[setting]
    for index, method in enumerate(methods):
        label, color = _method_default(method, index)
        st.session_state[_column_state_key("method", method, "label")] = label
        st.session_state[_column_state_key("method", method, "color")] = color
    for index, reference in enumerate(references):
        label, color = _reference_default(reference, index)
        st.session_state[_column_state_key("reference", reference, "label")] = label
        st.session_state[_column_state_key("reference", reference, "color")] = color
    for phenotype in phenotypes:
        st.session_state[_value_state_key("phenotype", phenotype, "color")] = (
            _phenotype_default_color(phenotype)
        )
    for group in groups:
        st.session_state[_value_state_key("group", group, "label")] = (
            HELDOUT_IMAGE_LABELS.get(group, group)
        )
        st.session_state[_value_state_key("group", group, "color")] = (
            HELDOUT_IMAGE_COLORS.get(group, _stable_named_color(f"group::{group}"))
        )
    st.session_state["viz_f1_paper_generated_mr_label"] = "ML-MR"
    st.session_state["viz_f1_paper_generated_mr_color"] = METHOD_COLORS["ML-MR"]
    st.session_state["viz_f1_paper_generated_consensus_label"] = "Consensus-ML"
    st.session_state["viz_f1_paper_generated_consensus_color"] = METHOD_COLORS[
        "Consensus-ML"
    ]
    nonvisual_panel_settings = {
        "output_stem",
        "focus_reference",
        "spatial_reference",
        "spatial_methods",
        "confusion_methods",
        "confidence_method",
        "learning_method",
        "selected_confidence",
        "minimum_bin_cells",
    }
    for spec in PAPER_PANEL_SPECS:
        for setting, value in default_panel_settings(spec).items():
            if setting not in nonvisual_panel_settings:
                st.session_state[_panel_key(spec.panel_id, setting)] = value
        if spec.panel_id == "main_ref2_metrics":
            for index, color in enumerate(["#24445C", "#6D8FA6", "#B5CBD8"]):
                st.session_state[
                    _panel_key(spec.panel_id, f"metric_color_{index}")
                ] = color


def _option2_visual_schema_version(cached_params: Mapping[str, Any]) -> int:
    value = st.session_state.get(
        "viz_f1_paper_visual_schema_version",
        cached_params.get("viz_f1_paper_visual_schema_version", 0),
    )
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _render_universal_shared_settings_form(
    project_root: Path,
    cached_params: dict[str, Any],
    output_dir_default: str,
    unified_csv_path: str,
    available_columns: list[str],
    *,
    use_shared_predictions: bool,
) -> None:
    defaults = default_common_settings()
    columns = [str(column) for column in available_columns]
    option1_output = st.session_state.get(
        "viz_f1_output_dir", cached_params.get("viz_f1_output_dir")
    )
    default_output = (
        Path(str(option1_output)).expanduser()
        if option1_output
        else Path(output_dir_default) / "F1_validation"
    )
    simple_defaults = {
        "viz_f1_paper_output_dir": str(default_output),
        "viz_f1_paper_formats": defaults["formats"],
        "viz_f1_paper_save_settings_json": True,
        "viz_f1_paper_font_family": defaults["font_family"],
        "viz_f1_paper_base_font_size": defaults["base_font_size"],
        "viz_f1_paper_title_font_size": defaults["title_font_size"],
        "viz_f1_paper_subplot_title_font_size": defaults[
            "subplot_title_font_size"
        ],
        "viz_f1_paper_axis_font_size": defaults["axis_font_size"],
        "viz_f1_paper_tick_font_size": defaults["tick_font_size"],
        "viz_f1_paper_annotation_font_size": defaults["annotation_font_size"],
        "viz_f1_paper_dpi": defaults["dpi"],
        "viz_f1_paper_background_color": defaults["background_color"],
        "viz_f1_paper_grid_color": defaults["grid_color"],
        "viz_f1_paper_grid_alpha": defaults["grid_alpha"],
        "viz_f1_paper_confidence_thresholds": "0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95",
        "viz_f1_paper_excluded_labels": ", ".join(sorted(INVALID_LABELS)),
        "viz_f1_paper_generated_mr_label": "ML-MR",
        "viz_f1_paper_generated_mr_color": "#7B2CBF",
        "viz_f1_paper_generated_consensus_label": "Consensus-ML",
        "viz_f1_paper_generated_consensus_color": "#008C8C",
    }
    for key, value in simple_defaults.items():
        _initialize_state(key, value, cached_params)
    _valid_multiselect_state(
        "viz_f1_paper_method_columns",
        columns,
        _default_method_columns(columns),
        cached_params,
    )
    _valid_multiselect_state(
        "viz_f1_paper_reference_columns",
        columns,
        _default_reference_columns(columns),
        cached_params,
    )
    optional_columns = ["(None)", *columns]
    _valid_select_state(
        "viz_f1_paper_group_column",
        optional_columns,
        _default_column(
            columns,
            [
                "validation_group",
                "patient_id",
                "slide_id",
                "batch_id",
                "tissue_roi",
                "roi",
            ],
        ),
        cached_params,
    )
    _valid_select_state(
        "viz_f1_paper_x_column",
        optional_columns,
        _default_column(columns, ["x_centroid", "centroid_x", "x", "X"]),
        cached_params,
    )
    _valid_select_state(
        "viz_f1_paper_y_column",
        optional_columns,
        _default_column(columns, ["y_centroid", "centroid_y", "y", "Y"]),
        cached_params,
    )

    st.markdown("### Step 1 — Shared data mappings and reproducible appearance")
    st.caption(f"Unified CSV: {unified_csv_path}")
    st.caption(
        "Save this form to commit mappings and appearance. Figures remain unchanged until an individual Generate and Save button is pressed."
    )
    current_methods = [
        str(value)
        for value in st.session_state.get("viz_f1_paper_method_columns", [])
    ]
    current_references = [
        str(value)
        for value in st.session_state.get("viz_f1_paper_reference_columns", [])
    ]
    current_group_column = str(
        st.session_state.get("viz_f1_paper_group_column", "(None)")
    )
    current_groups, _ = _unique_text_values(
        unified_csv_path,
        [] if current_group_column == "(None)" else [current_group_column],
    )
    current_phenotypes, _ = _unique_text_values(
        unified_csv_path,
        [*current_references, *current_methods],
    )
    current_phenotypes = [
        value
        for value in current_phenotypes
        if value.lower() not in INVALID_LABELS
    ]
    for index, method in enumerate(current_methods):
        label, color = _method_default(method, index)
        label_key = _column_state_key("method", method, "label")
        color_key = _column_state_key("method", method, "color")
        _initialize_state(label_key, label, cached_params)
        _initialize_state(color_key, color, cached_params)
        if not str(st.session_state.get(label_key, "")).strip():
            st.session_state[label_key] = label
            st.session_state[color_key] = color
    for index, reference in enumerate(current_references):
        label, color = _reference_default(reference, index)
        label_key = _column_state_key("reference", reference, "label")
        color_key = _column_state_key("reference", reference, "color")
        _initialize_state(label_key, label, cached_params)
        _initialize_state(color_key, color, cached_params)
        if not str(st.session_state.get(label_key, "")).strip():
            st.session_state[label_key] = label
            st.session_state[color_key] = color
    phenotype_color_keys = []
    for phenotype in current_phenotypes:
        color_key = _value_state_key("phenotype", phenotype, "color")
        _initialize_state(
            color_key,
            _phenotype_default_color(phenotype),
            cached_params,
        )
        phenotype_color_keys.append(color_key)
    if len(phenotype_color_keys) > 1 and all(
        str(st.session_state.get(key, "")).lower() in {"", "#000000", "black"}
        for key in phenotype_color_keys
    ):
        for phenotype, color_key in zip(current_phenotypes, phenotype_color_keys):
            st.session_state[color_key] = _phenotype_default_color(phenotype)
    if _option2_visual_schema_version(cached_params) < OPTION2_VISUAL_SCHEMA_VERSION:
        _restore_reproducible_style_defaults(
            current_methods,
            current_references,
            current_phenotypes,
            current_groups,
        )
        st.session_state["viz_f1_paper_visual_schema_version"] = (
            OPTION2_VISUAL_SCHEMA_VERSION
        )
        _persist_paper_settings(project_root, cached_params)
        st.info(
            "Option 2 visual defaults were recovered from fig3_code_v2.py: white "
            "publication canvas, reference dimensions, typography, colors, line/marker "
            "sizes, legends, heatmaps, shading, and axis styling. These values remain "
            "editable below and within each figure panel."
        )
    if st.button(
        "Restore all fig3_code_v2 visual defaults",
        key="viz_f1_paper_restore_reproducible_style",
        help=(
            "Restores the white publication canvas, reference-code dimensions, "
            "typography, titles, axes, legends, lines, markers, bars, heatmap settings, "
            "shading, Leiden/SOM/Vortex/RESOLVE and generated-ML colors, known manual "
            "references, and phenotype colors. Data mappings and ML settings are not "
            "changed. Unknown names receive stable deterministic colors."
        ),
    ):
        _restore_reproducible_style_defaults(
            current_methods,
            current_references,
            current_phenotypes,
            current_groups,
        )
        st.session_state["viz_f1_paper_visual_schema_version"] = (
            OPTION2_VISUAL_SCHEMA_VERSION
        )
        _persist_paper_settings(project_root, cached_params)
        st.success(
            "The fig3_code_v2 visual preset was restored across all Option 2 panels. "
            "You can still customize every exposed value."
        )
    with st.form("viz_f1_paper_universal_shared_form", clear_on_submit=False):
        st.text_input(
            "F1 output directory",
            key="viz_f1_paper_output_dir",
            help="Folder where uniquely named PNG, SVG, and optional settings JSON files are saved.",
        )
        export_columns = st.columns(3)
        with export_columns[0]:
            st.multiselect(
                "Output formats",
                ["PNG", "SVG"],
                key="viz_f1_paper_formats",
                help="PNG is a raster image; SVG is editable vector artwork. Select both for the most flexible publication workflow.",
            )
        with export_columns[1]:
            st.number_input(
                "PNG resolution (DPI)",
                min_value=72,
                max_value=1200,
                step=25,
                key="viz_f1_paper_dpi",
                help="Raster resolution used for PNG export. 300 DPI is the standard default; some journals request 600 DPI.",
            )
        with export_columns[2]:
            st.checkbox(
                "Save settings JSON beside each figure",
                key="viz_f1_paper_save_settings_json",
                help="Saves the column mappings, labels, colors, dimensions, and scientific settings used to reproduce each figure.",
            )

        with st.expander("Phenotyping/prediction methods", expanded=True):
            st.info(
                "For each selected phenotype column, set only its figure label and "
                "color. The confidence mapping is optional: choose a separate numeric "
                "column containing one confidence/probability value per cell for the "
                "label assigned by that method. Use (None) if your dataset has no such "
                "score. Confidence is used only by confidence/coverage figures and is "
                "never used as a training feature."
            )
            selected_methods = st.multiselect(
                "Columns to compare",
                columns,
                key="viz_f1_paper_method_columns",
                help=(
                    "Select categorical columns whose values are phenotype calls, "
                    "for example phenotype_leiden, phenotype_som, or phenotype_vortex."
                ),
            )
            method_palette = st.columns(3)
            for index, method in enumerate(selected_methods):
                label_default, color_default = _method_default(method, index)
                label_key = _column_state_key("method", method, "label")
                color_key = _column_state_key("method", method, "color")
                confidence_key = _column_state_key("method", method, "confidence")
                _initialize_state(label_key, label_default, cached_params)
                _initialize_state(color_key, color_default, cached_params)
                if not str(st.session_state.get(label_key, "")).strip():
                    st.session_state[label_key] = label_default
                _valid_select_state(
                    confidence_key,
                    optional_columns,
                    _guess_confidence_column(method, columns),
                    cached_params,
                )
                with method_palette[index % 3]:
                    st.text_input(
                        f"{method} display label",
                        key=label_key,
                        help=(
                            "The human-readable method name printed on axes, titles, "
                            "and legends. It does not rename the CSV column."
                        ),
                    )
                    st.color_picker(
                        f"{method} color",
                        key=color_key,
                        help=(
                            "The method-level color used for lines, bars, points, and "
                            "legend entries. Phenotype-category colors are configured "
                            "separately below."
                        ),
                    )
                    st.selectbox(
                        f"Optional confidence/probability column for {method}",
                        optional_columns,
                        key=confidence_key,
                        format_func=lambda value: (
                            "(None — no per-cell confidence available)"
                            if value == "(None)"
                            else value
                        ),
                        help=(
                            "Choose a separate numeric column with one score per cell "
                            "for this method's assigned phenotype, normally from 0 to 1. "
                            "Examples: confidence_leiden, leiden_confidence, or the "
                            "maximum predicted-class probability. Do not select the "
                            "categorical phenotype column or a marker-intensity column. "
                            "Leave this as None when the method did not produce confidence "
                            "scores; normal F1 figures will still work."
                        ),
                    )
            _valid_multiselect_state(
                "viz_f1_paper_component_methods",
                selected_methods,
                selected_methods[: min(4, len(selected_methods))],
                cached_params,
            )
            st.multiselect(
                "Component methods used to calculate consensus",
                selected_methods,
                key="viz_f1_paper_component_methods",
                help=(
                    "These categorical methods vote to form direct consensus labels "
                    "and are also the inputs whose phenotype-specific reliability "
                    "weights are recalculated inside every held-out fold."
                ),
            )
            st.caption(
                "Appearance of newly trained held-out predictions. Existing saved "
                "ML columns are not used by publication validation."
            )
            generated_columns = st.columns(2)
            with generated_columns[0]:
                st.text_input(
                    "Generated ML-MR display label",
                    key="viz_f1_paper_generated_mr_label",
                    help="Legend/axis label for the newly retrained held-out ML-MR predictions.",
                )
                st.color_picker(
                    "Generated ML-MR color",
                    key="viz_f1_paper_generated_mr_color",
                    help="Method-level plotting color for the newly generated ML-MR results.",
                )
            with generated_columns[1]:
                st.text_input(
                    "Generated Consensus-ML display label",
                    key="viz_f1_paper_generated_consensus_label",
                    help="Legend/axis label for the newly retrained held-out Consensus-ML predictions.",
                )
                st.color_picker(
                    "Generated Consensus-ML color",
                    key="viz_f1_paper_generated_consensus_color",
                    help="Method-level plotting color for the newly generated Consensus-ML results.",
                )
            if selected_methods:
                primary_columns = st.columns(2)
                _valid_select_state(
                    "viz_f1_paper_primary_method",
                    selected_methods,
                    selected_methods[0],
                    cached_params,
                )
                _valid_select_state(
                    "viz_f1_paper_secondary_method",
                    selected_methods,
                    selected_methods[1]
                    if len(selected_methods) > 1
                    else selected_methods[0],
                    cached_params,
                )
                with primary_columns[0]:
                    st.selectbox(
                        "Primary resolution method",
                        selected_methods,
                        key="viz_f1_paper_primary_method",
                        help="Default component method used where a figure needs one primary comparison method.",
                    )
                with primary_columns[1]:
                    st.selectbox(
                        "Secondary resolution method",
                        selected_methods,
                        key="viz_f1_paper_secondary_method",
                        help="Default second component method used by figures that compare two methods.",
                    )
            if use_shared_predictions:
                st.caption(
                    "The scientific agreement threshold is configured once in the "
                    "shared held-out ML zone above and is then used consistently by "
                    "the analysis and figure panels."
                )
            else:
                maximum_agreement = max(1, len(selected_methods))
                minimum_agreement = 2 if maximum_agreement >= 2 else 1
                _initialize_state(
                    "viz_f1_paper_consensus_min_agree",
                    min(3, maximum_agreement),
                    cached_params,
                )
                current_agreement = int(
                    st.session_state.get(
                        "viz_f1_paper_consensus_min_agree", minimum_agreement
                    )
                )
                if not minimum_agreement <= current_agreement <= maximum_agreement:
                    st.session_state["viz_f1_paper_consensus_min_agree"] = min(
                        max(current_agreement, minimum_agreement), maximum_agreement
                    )
                clamp_numeric_widget_state(
                    "viz_f1_paper_consensus_min_agree",
                    int(minimum_agreement), int(maximum_agreement),
                )
                st.number_input(
                    "Consensus minimum agreeing methods",
                    min_value=minimum_agreement,
                    max_value=maximum_agreement,
                    step=1,
                    key="viz_f1_paper_consensus_min_agree",
                    help=(
                        "Used when deriving consensus from existing columns. The "
                        "reference setup uses 3 of 4 methods. Lower values give more "
                        "direct calls but allow weaker agreement; higher values are "
                        "stricter. With two methods, use 2."
                    ),
                )

        with st.expander("Manual-reference columns", expanded=True):
            st.info(
                "Select columns containing human/manual phenotype labels. Known "
                "reference columns from the reference analysis receive the same "
                "labels and colors automatically; unknown names receive a stable "
                "name-based color that is reproducible across datasets."
            )
            selected_references = st.multiselect(
                "Manual-reference columns",
                columns,
                key="viz_f1_paper_reference_columns",
                help="One reference is sufficient for most figures; figures that compare annotators clearly state when two are required.",
            )
            reference_palette = st.columns(2)
            for index, reference in enumerate(selected_references):
                label_default, color_default = _reference_default(reference, index)
                label_key = _column_state_key("reference", reference, "label")
                color_key = _column_state_key("reference", reference, "color")
                _initialize_state(label_key, label_default, cached_params)
                _initialize_state(color_key, color_default, cached_params)
                if not str(st.session_state.get(label_key, "")).strip():
                    st.session_state[label_key] = label_default
                with reference_palette[index % 2]:
                    st.text_input(
                        f"{reference} display label",
                        key=label_key,
                        help="Human-readable name printed for this manual-reference column; the CSV column itself is unchanged.",
                    )
                    st.color_picker(
                        f"{reference} color",
                        key=color_key,
                        help="Plotting color used when this manual reference appears as a series or legend entry.",
                    )
            if selected_references:
                reference_focus_columns = st.columns(2)
                _valid_select_state(
                    "viz_f1_paper_primary_reference",
                    selected_references,
                    selected_references[0],
                    cached_params,
                )
                _valid_select_state(
                    "viz_f1_paper_secondary_reference",
                    selected_references,
                    selected_references[1]
                    if len(selected_references) > 1
                    else selected_references[0],
                    cached_params,
                )
                with reference_focus_columns[0]:
                    st.selectbox(
                        "Primary reference",
                        selected_references,
                        key="viz_f1_paper_primary_reference",
                        help="Main manual ground truth used for training ML-MR and for the default performance figures.",
                    )
                with reference_focus_columns[1]:
                    st.selectbox(
                        "Secondary reference",
                        selected_references,
                        key="viz_f1_paper_secondary_reference",
                        help="Optional second manual ground truth used by reference-comparison figures. With one reference, select the same column.",
                    )

        with st.expander("Grouping, coordinates, and label handling", expanded=False):
            mapping_columns = st.columns(3)
            with mapping_columns[0]:
                group_column = st.selectbox(
                    "Optional patient/slide/batch/ROI group",
                    optional_columns,
                    key="viz_f1_paper_group_column",
                    help="Column identifying the biological unit shown as a group in descriptive figures. The actual held-out fold mapping is configured in the publication-training section below.",
                )
            with mapping_columns[1]:
                st.selectbox(
                    "Optional X-coordinate column",
                    optional_columns,
                    key="viz_f1_paper_x_column",
                    help="Per-cell horizontal tissue coordinate used only by spatial validation figures.",
                )
            with mapping_columns[2]:
                st.selectbox(
                    "Optional Y-coordinate column",
                    optional_columns,
                    key="viz_f1_paper_y_column",
                    help="Per-cell vertical tissue coordinate paired with X for spatial validation figures.",
                )
            st.text_input(
                "Labels excluded from scoring (comma-separated)",
                key="viz_f1_paper_excluded_labels",
                help="Phenotype values treated as missing/unscored rather than biological classes, for example Others, Unclassified, or Unknown.",
            )
            st.text_input(
                "Confidence thresholds from 0 to 1 (comma-separated)",
                key="viz_f1_paper_confidence_thresholds",
                help="Cutoffs evaluated by confidence-versus-coverage figures. A cell is accepted at a cutoff when its mapped confidence score is at least that value.",
            )
            group_values, group_truncated = _unique_text_values(
                unified_csv_path, [] if group_column == "(None)" else [group_column]
            )
            st.session_state["viz_f1_paper_group_values"] = group_values
            if group_values:
                st.caption(
                    "Optional display labels and colors for the detected group values"
                )
                group_widgets = st.columns(3)
                for index, group in enumerate(group_values):
                    label_key = _value_state_key("group", group, "label")
                    color_key = _value_state_key("group", group, "color")
                    _initialize_state(label_key, group, cached_params)
                    _initialize_state(
                        color_key, _group_color(defaults, group, index), cached_params
                    )
                    with group_widgets[index % 3]:
                        st.text_input(
                            f"{group} display label",
                            key=label_key,
                            help="Text printed for this group value in legends and axes.",
                        )
                        st.color_picker(
                            f"{group} color",
                            key=color_key,
                            help="Stable plotting color for this group value; it does not affect fold assignment.",
                        )
            if group_truncated:
                st.warning("Only the first 80 group values are exposed for styling.")

        phenotype_values, phenotype_truncated = _unique_text_values(
            unified_csv_path, [*selected_references, *selected_methods]
        )
        phenotype_values = [
            value for value in phenotype_values if value.lower() not in INVALID_LABELS
        ]
        st.session_state["viz_f1_paper_phenotype_values"] = phenotype_values
        with st.expander("Phenotype colors", expanded=False):
            st.info(
                "These colors belong to biological phenotype values such as B, CD4, "
                "CD8, or Tumor—not to the phenotyping methods. Known reference-code "
                "phenotypes use the house palette; every unknown phenotype gets a "
                "stable name-based color so it remains reproducible across datasets."
            )
            if not phenotype_values:
                st.caption(
                    "Select method/reference columns to discover phenotype values."
                )
            phenotype_widgets = st.columns(4)
            for index, phenotype in enumerate(phenotype_values):
                key = _value_state_key("phenotype", phenotype, "color")
                _initialize_state(
                    key,
                    _phenotype_default_color(phenotype),
                    cached_params,
                )
                with phenotype_widgets[index % 4]:
                    st.color_picker(
                        phenotype,
                        key=key,
                        help="Category color used whenever this phenotype value is plotted spatially or in a phenotype legend.",
                    )
            if phenotype_truncated:
                st.warning(
                    "Only the first 80 detected phenotype values are exposed for styling."
                )

        with st.expander("Typography and canvas", expanded=False):
            typography = st.columns(4)
            with typography[0]:
                st.selectbox(
                    "Font family", FONT_FAMILIES, key="viz_f1_paper_font_family"
                )
                st.color_picker(
                    "Canvas color",
                    key="viz_f1_paper_background_color",
                    help=(
                        "Defaults to the opaque white publication canvas in "
                        "fig3_code_v2.py. If you intentionally choose a dark canvas, "
                        "axis, tick, title, and legend text switch to a light foreground."
                    ),
                )
            with typography[1]:
                st.number_input(
                    "Base font size",
                    min_value=5.0,
                    max_value=30.0,
                    step=0.5,
                    key="viz_f1_paper_base_font_size",
                )
                st.number_input(
                    "Figure/suptitle size",
                    min_value=6.0,
                    max_value=40.0,
                    step=0.5,
                    key="viz_f1_paper_title_font_size",
                )
                st.number_input(
                    "Axes/subplot title size",
                    min_value=6.0,
                    max_value=40.0,
                    step=0.5,
                    key="viz_f1_paper_subplot_title_font_size",
                    help="Reference-code default is 11 pt for titles inside plotting axes.",
                )
            with typography[2]:
                st.number_input(
                    "Axis-label size",
                    min_value=5.0,
                    max_value=30.0,
                    step=0.5,
                    key="viz_f1_paper_axis_font_size",
                )
                st.number_input(
                    "Tick-label size",
                    min_value=4.0,
                    max_value=24.0,
                    step=0.5,
                    key="viz_f1_paper_tick_font_size",
                )
            with typography[3]:
                st.number_input(
                    "Annotation size",
                    min_value=4.0,
                    max_value=24.0,
                    step=0.5,
                    key="viz_f1_paper_annotation_font_size",
                )
                st.color_picker("Grid color", key="viz_f1_paper_grid_color")
                st.slider(
                    "Grid opacity", 0.0, 1.0, step=0.05, key="viz_f1_paper_grid_alpha"
                )
        save_shared = st.form_submit_button(
            "Save Shared Figure Settings", type="primary"
        )
    if save_shared:
        if not st.session_state.get("viz_f1_paper_formats"):
            st.error("Select PNG, SVG, or both before generating figures.")
        else:
            _persist_paper_settings(project_root, cached_params)
            st.success("Shared mappings and settings saved. No figure was rendered.")


def _initialize_panel_state(
    spec: PaperPanelSpec,
    cached_params: Mapping[str, Any],
) -> None:
    for setting, default in default_panel_settings(spec).items():
        _initialize_state(_panel_key(spec.panel_id, setting), default, cached_params)
    if spec.panel_id == "main_ref2_metrics":
        for index, color in enumerate(["#24445C", "#6D8FA6", "#B5CBD8"]):
            _initialize_state(
                _panel_key(spec.panel_id, f"metric_color_{index}"), color, cached_params
            )
    if spec.panel_id == "main_workflow":
        workflow_defaults = _panel_settings_from_state(spec).get("workflow_text", {})
        for name, value in workflow_defaults.items():
            _initialize_state(
                _panel_key(spec.panel_id, f"workflow_{name}"), value, cached_params
            )


def _render_panel_form(
    spec: PaperPanelSpec,
    project_root: Path,
    cached_params: dict[str, Any],
) -> None:
    _initialize_panel_state(spec, cached_params)
    preview_key = f"_f1_paper_preview_{spec.panel_id}"
    paths_key = f"_f1_paper_paths_{spec.panel_id}"
    error_key = f"_f1_paper_error_{spec.panel_id}"
    with st.expander(spec.label, expanded=False):
        st.caption(
            "This panel is independent. Editing controls does not render; the source "
            "table is read only after Generate and Save is pressed."
        )
        with st.form(f"viz_f1_paper_form_{spec.panel_id}", clear_on_submit=False):
            heading_columns = st.columns([2, 1, 1])
            with heading_columns[0]:
                st.text_area(
                    "Figure title", height=85, key=_panel_key(spec.panel_id, "title")
                )
                st.text_input(
                    "Output filename label",
                    key=_panel_key(spec.panel_id, "output_stem"),
                )
            with heading_columns[1]:
                st.text_input(
                    "Panel label", key=_panel_key(spec.panel_id, "panel_label")
                )
                st.number_input(
                    "Width (inches)",
                    min_value=3.0,
                    max_value=40.0,
                    step=0.2,
                    key=_panel_key(spec.panel_id, "width"),
                )
            with heading_columns[2]:
                st.number_input(
                    "Height (inches)",
                    min_value=2.0,
                    max_value=30.0,
                    step=0.2,
                    key=_panel_key(spec.panel_id, "height"),
                )
                st.number_input(
                    "Line width",
                    min_value=0.2,
                    max_value=8.0,
                    step=0.1,
                    key=_panel_key(spec.panel_id, "line_width"),
                )
                st.number_input(
                    "Figure-title size",
                    min_value=6.0,
                    max_value=40.0,
                    step=0.2,
                    key=_panel_key(spec.panel_id, "title_font_size"),
                    help="Panel-level figure/suptitle size from the reference code; remains editable for this figure only.",
                )

            text_columns = st.columns(2)
            with text_columns[0]:
                st.text_input(
                    "X-axis label (blank keeps reference text)",
                    key=_panel_key(spec.panel_id, "x_label"),
                )
            with text_columns[1]:
                st.text_input(
                    "Y-axis label (blank keeps reference text)",
                    key=_panel_key(spec.panel_id, "y_label"),
                )

            if spec.has_legend:
                legend_columns = st.columns(3)
                with legend_columns[0]:
                    st.selectbox(
                        "Legend placement",
                        LEGEND_LOCATIONS,
                        key=_panel_key(spec.panel_id, "legend_location"),
                    )
                with legend_columns[1]:
                    st.text_input(
                        "Legend title", key=_panel_key(spec.panel_id, "legend_title")
                    )
                with legend_columns[2]:
                    st.number_input(
                        "Legend columns",
                        min_value=1,
                        max_value=8,
                        step=1,
                        key=_panel_key(spec.panel_id, "legend_columns"),
                    )

            geometry_columns = st.columns(3)
            with geometry_columns[0]:
                st.number_input(
                    "Marker size",
                    min_value=1.0,
                    max_value=30.0,
                    step=0.5,
                    key=_panel_key(spec.panel_id, "marker_size"),
                )
                st.slider(
                    "Grid opacity",
                    0.0,
                    1.0,
                    step=0.01,
                    key=_panel_key(spec.panel_id, "grid_alpha"),
                    help="Panel-specific reference-code grid opacity.",
                )
                st.number_input(
                    "Plot-box aspect (0 = automatic)",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.05,
                    key=_panel_key(spec.panel_id, "box_aspect"),
                    help="Reference held-out plots use height/width = 0.6; set 0 to let Matplotlib choose from the figure dimensions.",
                )
            if spec.method_axis:
                with geometry_columns[1]:
                    st.multiselect(
                        "Shade method columns",
                        METHOD_ORDER,
                        key=_panel_key(spec.panel_id, "shade_methods"),
                    )
                with geometry_columns[2]:
                    st.color_picker(
                        "Shade color", key=_panel_key(spec.panel_id, "shade_color")
                    )
                    st.slider(
                        "Shade opacity",
                        0.0,
                        1.0,
                        step=0.05,
                        key=_panel_key(spec.panel_id, "shade_alpha"),
                    )
            if spec.bar_grouping:
                with geometry_columns[1]:
                    st.selectbox(
                        "Bar grouping",
                        ["Series within categories", "Categories within series"],
                        key=_panel_key(spec.panel_id, "bar_grouping"),
                    )
                with geometry_columns[2]:
                    st.number_input(
                        "Bar width",
                        min_value=0.03,
                        max_value=0.8,
                        step=0.01,
                        key=_panel_key(spec.panel_id, "bar_width"),
                    )

            if spec.heatmap or spec.panel_id in {
                "supp_reference_distribution",
                "supp_human",
            }:
                heatmap_columns = st.columns(4)
                with heatmap_columns[0]:
                    st.selectbox(
                        "Color map",
                        HEATMAP_CMAPS,
                        key=_panel_key(spec.panel_id, "heatmap_cmap"),
                    )
                with heatmap_columns[1]:
                    st.checkbox(
                        "Automatic color range",
                        key=_panel_key(spec.panel_id, "heatmap_auto_range"),
                    )
                    st.number_input(
                        "Color minimum",
                        step=0.05,
                        key=_panel_key(spec.panel_id, "heatmap_vmin"),
                    )
                    st.number_input(
                        "Color maximum",
                        step=0.05,
                        key=_panel_key(spec.panel_id, "heatmap_vmax"),
                    )
                with heatmap_columns[2]:
                    st.checkbox(
                        "Show cell values",
                        key=_panel_key(spec.panel_id, "heatmap_annotations"),
                    )
                    st.checkbox(
                        "Square cells", key=_panel_key(spec.panel_id, "heatmap_square")
                    )
                    st.number_input(
                        "Heatmap-value size",
                        min_value=4.0,
                        max_value=24.0,
                        step=0.5,
                        key=_panel_key(spec.panel_id, "heatmap_annotation_size"),
                        help="Font size of numbers printed inside heatmap cells; the reference-code default is 9.5 pt (11 pt for the pairwise matrix).",
                    )
                with heatmap_columns[3]:
                    st.text_input(
                        "Colorbar label (blank keeps reference text)",
                        key=_panel_key(spec.panel_id, "colorbar_label"),
                    )
                    st.number_input(
                        "Cell border width",
                        min_value=0.0,
                        max_value=5.0,
                        step=0.1,
                        key=_panel_key(spec.panel_id, "heatmap_linewidth"),
                    )

            if spec.panel_id in {"main_spatial", "supp_spatial"}:
                spatial_columns = st.columns(4)
                with spatial_columns[0]:
                    st.selectbox(
                        "Spatial reference",
                        REFERENCE_ORDER,
                        key=_panel_key(spec.panel_id, "spatial_reference"),
                    )
                    st.number_input(
                        "Cell point size",
                        min_value=0.1,
                        max_value=100.0,
                        step=0.5,
                        key=_panel_key(spec.panel_id, "point_size"),
                    )
                with spatial_columns[1]:
                    st.color_picker(
                        "Match color", key=_panel_key(spec.panel_id, "match_color")
                    )
                with spatial_columns[2]:
                    st.color_picker(
                        "Mismatch color",
                        key=_panel_key(spec.panel_id, "mismatch_color"),
                    )
                with spatial_columns[3]:
                    st.color_picker(
                        "Unscored/Others color",
                        key=_panel_key(spec.panel_id, "unscored_color"),
                    )

            if spec.panel_id in {
                "supp_nonconsensus_threshold",
                "supp_bins_ref1",
                "supp_bins_ref2",
            }:
                sensitivity_columns = st.columns(2)
                with sensitivity_columns[0]:
                    st.number_input(
                        "Minimum cells required to plot F1",
                        min_value=1,
                        max_value=10000,
                        step=1,
                        key=_panel_key(spec.panel_id, "minimum_bin_cells"),
                    )
                if spec.panel_id == "supp_nonconsensus_threshold":
                    with sensitivity_columns[1]:
                        st.slider(
                            "Selected confidence marker",
                            0.0,
                            1.0,
                            step=0.01,
                            key=_panel_key(spec.panel_id, "selected_confidence"),
                        )

            if spec.panel_id == "main_ref2_metrics":
                metric_columns = st.columns(3)
                for index, label in enumerate(["Macro-F1", "Micro-F1", "Weighted-F1"]):
                    with metric_columns[index]:
                        st.color_picker(
                            label,
                            key=_panel_key(spec.panel_id, f"metric_color_{index}"),
                        )

            if spec.panel_id == "main_workflow":
                with st.expander("Workflow box and footer text", expanded=False):
                    for name, label in [
                        ("components", "Component-method box"),
                        ("mr_source", "ML-MR source box"),
                        ("consensus_source", "Consensus-ML source box"),
                        ("mr_training", "ML-MR training box"),
                        ("consensus_training", "Consensus-ML training box"),
                        ("holdout", "Hold-out box"),
                        ("comparison", "Comparison box"),
                        ("footer", "Footer"),
                    ]:
                        st.text_area(
                            label,
                            height=70,
                            key=_panel_key(spec.panel_id, f"workflow_{name}"),
                        )

            submitted = st.form_submit_button(
                "Generate and Save PNG/SVG", type="primary"
            )

        if submitted:
            formats = list(st.session_state.get("viz_f1_paper_formats", []))
            if not formats:
                st.session_state[error_key] = (
                    "Select PNG, SVG, or both in Shared Option 2 settings."
                )
            else:
                source_dir = str(st.session_state["viz_f1_paper_source_dir"])
                missing = missing_panel_tables(spec.panel_id, source_dir)
                if missing:
                    st.session_state[error_key] = (
                        "Missing source table(s): "
                        + ", ".join(str(path) for path in missing)
                    )
                else:
                    figure: plt.Figure | None = None
                    try:
                        common = _common_settings_from_state()
                        panel = _panel_settings_from_state(spec)
                        figure = build_paper_figure(
                            spec.panel_id, source_dir, common, panel
                        )
                        settings_payload = {
                            "panel_id": spec.panel_id,
                            "source_data": str(_source_data_dir(source_dir)),
                            "common": common,
                            "panel": panel,
                        }
                        saved, preview = save_paper_figure(
                            figure,
                            st.session_state["viz_f1_paper_output_dir"],
                            str(panel["output_stem"]),
                            formats,
                            dpi=int(common["dpi"]),
                            settings=settings_payload
                            if st.session_state.get(
                                "viz_f1_paper_save_settings_json", True
                            )
                            else None,
                        )
                        st.session_state[preview_key] = preview
                        st.session_state[paths_key] = [str(path) for path in saved]
                        st.session_state.pop(error_key, None)
                        _persist_paper_settings(project_root, cached_params)
                        log_console(
                            f"[{timestamp()}] Saved Option 2 panel {spec.panel_id} to {saved[0].parent if saved else st.session_state['viz_f1_paper_output_dir']}"
                        )
                    except Exception as exc:
                        st.session_state[error_key] = str(exc)
                    finally:
                        if figure is not None:
                            plt.close(figure)

        if error_key in st.session_state:
            st.error(st.session_state[error_key])
        preview = st.session_state.get(preview_key)
        if isinstance(preview, bytes):
            st.image(
                preview,
                caption=f"Last generated preview — {spec.label}",
                width="stretch",
            )
        saved_paths = st.session_state.get(paths_key)
        if isinstance(saved_paths, list) and saved_paths:
            st.success("Saved the latest panel with a unique label.")
            st.code("\n".join(str(path) for path in saved_paths))


PUBLICATION_TRAINED_PANEL_IDS = {
    "main_heldout_macro",
    "main_ref2_metrics",
    "main_per_phenotype_f1",
    "main_spatial",
    "supp_weights",
    "supp_ref1_f1",
    "supp_routes",
    "supp_confidence_all",
    "supp_nonconsensus_threshold",
    "supp_confusion",
    "supp_radar",
    "supp_heldout_variation",
    "supp_learning",
    "supp_bins_ref2",
}


def _publication_common_settings(
    common: Mapping[str, Any],
    strict_settings: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one strict OOF run into the universal figure configuration."""
    oof = analysis.get("oof_predictions")
    if not isinstance(oof, pd.DataFrame):
        return dict(common)
    components = [
        str(value)
        for value in strict_settings.get("component_methods", [])
        if str(value) in oof.columns
    ]
    methods = list(components)
    generated = [
        (
            "prediction_mr_ml_strict",
            "ML-MR",
            "#7B2CBF",
            "mr_ml_confidence_strict",
        ),
        (
            "prediction_consensus_ml_strict",
            "Consensus-ML",
            "#008C8C",
            "consensus_ml_confidence_strict",
        ),
    ]
    method_labels = dict(common.get("method_labels", {}))
    method_colors = dict(common.get("method_colors", {}))
    generated_labels = dict(common.get("generated_method_labels", {}))
    generated_colors = dict(common.get("generated_method_colors", {}))
    mapped_component_confidence = dict(common.get("confidence_columns", {}))
    confidence_columns: dict[str, str] = {}
    for method in components:
        method_labels.setdefault(method, _display_name(method))
        method_colors.setdefault(method, _method_default(method, len(methods))[1])
        mapped_confidence = str(mapped_component_confidence.get(method, "")).strip()
        if mapped_confidence:
            confidence_columns[method] = mapped_confidence
    for column, label, color, confidence in generated:
        if column not in oof.columns:
            continue
        methods.append(column)
        method_labels[column] = str(generated_labels.get(column, label))
        method_colors[column] = str(generated_colors.get(column, color))
        if confidence in oof.columns:
            confidence_columns[column] = confidence
    method_colors["ML-MR"] = str(
        generated_colors.get("prediction_mr_ml_strict", "#7B2CBF")
    )
    method_colors["Consensus-ML"] = str(
        generated_colors.get("prediction_consensus_ml_strict", "#008C8C")
    )
    primary_reference = str(strict_settings.get("primary_reference") or "")
    references = [
        str(value)
        for value in common.get("reference_order", [])
        if str(value) in oof.columns
    ]
    if primary_reference in oof.columns:
        references = [
            primary_reference,
            *[value for value in references if value != primary_reference],
        ]
    groups = list(
        dict.fromkeys(
            oof.get("validation_group", pd.Series(dtype="string")).dropna().astype(str)
        )
    )
    configured_group_colors = dict(common.get("group_colors", {}))
    primary_method = (
        "prediction_mr_ml_strict"
        if "prediction_mr_ml_strict" in methods
        else (methods[0] if methods else "")
    )
    secondary_method = (
        "prediction_consensus_ml_strict"
        if "prediction_consensus_ml_strict" in methods
        else primary_method
    )
    result = dict(common)
    result.update(
        {
            "method_order": methods,
            "method_labels": method_labels,
            "method_colors": method_colors,
            "component_methods": components,
            "confidence_columns": confidence_columns,
            "confidence_focus_method": secondary_method,
            "reference_order": references,
            "primary_reference": primary_reference
            if primary_reference in references
            else (references[0] if references else ""),
            "secondary_reference": references[1]
            if len(references) > 1
            else (references[0] if references else ""),
            "primary_method": primary_method,
            "secondary_method": secondary_method,
            "group_column": "validation_group",
            "group_order": groups,
            "group_labels": {
                group: str(common.get("group_labels", {}).get(group, group))
                for group in groups
            },
            "group_colors": {
                group: str(
                    configured_group_colors.get(
                        group,
                        _stable_named_color(f"group::{group}"),
                    )
                )
                for group in groups
            },
            "consensus_min_agree": int(
                strict_settings.get(
                    "consensus_min_methods", common.get("consensus_min_agree", 3)
                )
            ),
            "learning_curve_csv": "",
            "cached_learning_curve_available": bool(
                isinstance(analysis.get("learning_curve"), pd.DataFrame)
                and not analysis["learning_curve"].empty
            ),
        }
    )
    return result


def _publication_analysis_frame(
    analysis: Mapping[str, Any],
    common: Mapping[str, Any],
    unified_csv_path: str,
) -> pd.DataFrame:
    oof = analysis.get("oof_predictions")
    if not isinstance(oof, pd.DataFrame):
        raise ValueError(
            "Run or load the shared ML prediction data before generating this figure."
        )
    frame = oof.copy()
    source_columns = [
        str(common.get(setting) or "")
        for setting in ("x_column", "y_column")
        if common.get(setting) and str(common.get(setting)) not in frame.columns
    ]
    source_columns.extend(
        str(column)
        for column in dict(common.get("confidence_columns", {})).values()
        if str(column) and str(column) not in frame.columns
    )
    source_columns = list(dict.fromkeys(source_columns))
    if source_columns and "source_csv_row" in frame.columns:
        source = _read_unified_frame(unified_csv_path)
        rows = pd.to_numeric(frame["source_csv_row"], errors="coerce")
        valid = rows.notna() & rows.ge(0) & rows.lt(len(source))
        for column in source_columns:
            if column not in source.columns:
                continue
            frame[column] = pd.NA
            frame.loc[valid, column] = source.iloc[rows.loc[valid].astype(int)][
                column
            ].to_numpy()
    return frame


def _render_publication_learning_settings(
    cached_params: Mapping[str, Any],
    strict_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Add optional, fingerprinted learning-curve work to the shared run."""
    defaults = {
        "viz_f1_paper_run_learning_curve": False,
        "viz_f1_paper_learning_targets": "5, 10, 20, 40, 80, 160",
        "viz_f1_paper_learning_repeats": 10,
        "viz_f1_paper_learning_estimators": 500,
    }
    for key, value in defaults.items():
        _initialize_state(key, value, cached_params)

    with st.expander("Optional training-sample learning-curve experiment"):
        enabled = st.checkbox(
            "Build a balanced held-out learning curve in the shared run",
            key="viz_f1_paper_run_learning_curve",
            help=(
                "For every size and repeat, new ML-MR models are trained only on "
                "manual-reference cells from non-held-out groups."
            ),
        )
        columns = st.columns(3)
        with columns[0]:
            st.text_input(
                "Training cells per phenotype",
                key="viz_f1_paper_learning_targets",
                disabled=not enabled,
                help="Comma-separated balanced training-size targets.",
            )
        with columns[1]:
            st.number_input(
                "Independent repeats",
                min_value=1,
                max_value=100,
                step=1,
                key="viz_f1_paper_learning_repeats",
                disabled=not enabled,
            )
        with columns[2]:
            st.number_input(
                "Trees per learning model",
                min_value=5,
                max_value=10000,
                step=25,
                key="viz_f1_paper_learning_estimators",
                disabled=not enabled,
            )
        st.caption(
            "This experiment is cached with the main OOF analysis and reused by "
            "the learning-curve figure. Changing these values creates a new cache "
            "fingerprint; computation starts only after Run / rebuild."
        )
        if enabled and bool(strict_settings.get("quick_mode", False)):
            st.info(
                "Fast preview caps the learning curve at 3 repeats and the shared "
                "preview tree count. Turn Fast preview off for publication output."
            )

    result = dict(strict_settings)
    result.update(
        {
            "run_learning_curve": bool(enabled),
            "learning_targets": _parse_positive_int_values(
                st.session_state.get("viz_f1_paper_learning_targets", ""),
                [5, 10, 20, 40, 80, 160],
            ),
            "learning_repeats": int(
                st.session_state.get("viz_f1_paper_learning_repeats", 10)
            ),
            "learning_n_estimators": int(
                st.session_state.get("viz_f1_paper_learning_estimators", 500)
            ),
        }
    )
    return result


def _render_publication_analysis_controls(
    project_root: Path,
    cached_params: dict[str, Any],
    unified_csv_path: str,
    common: Mapping[str, Any],
    strict_settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    st.markdown("#### Build or reuse the shared ML prediction outputs")
    st.markdown(
        "For each validation fold, one complete patient/slide/image group is hidden. "
        "ML-MR learns manual-reference labels and phenotype-specific component weights "
        "only from the remaining groups. Consensus-ML recomputes direct consensus, "
        "trains only on consensus cells from the remaining groups, and predicts the "
        "non-consensus cells in the hidden group. The hidden group never contributes "
        "labels, weights, pseudo-labels, or model fitting. At least two mapped groups "
        "are required; three or more give a more informative leave-one-group-out analysis."
    )
    if strict_settings.get("configuration_error"):
        st.error(str(strict_settings["configuration_error"]))
        return None
    if not (
        strict_settings.get("run_grouped_mr")
        or strict_settings.get("run_grouped_consensus")
    ):
        st.error("Enable grouped ML-MR, grouped Consensus-ML, or both.")
        return None
    if len(strict_settings.get("component_methods", [])) < 2:
        st.error("Select at least two component phenotyping methods.")
        return None
    if not strict_settings.get("markers"):
        st.error("Select at least one marker feature used to train the ML workflows.")
        return None
    references = [str(value) for value in common.get("reference_order", [])]
    primary = str(strict_settings.get("primary_reference") or "")
    if primary:
        references = [primary, *[value for value in references if value != primary]]
    if not references:
        st.error("Select at least one manual-reference column.")
        return None
    output_dir = str(st.session_state.get("viz_f1_paper_output_dir", ""))
    excluded = sorted(
        str(value) for value in common.get("excluded_labels", INVALID_LABELS)
    )
    thresholds = list(common.get("confidence_thresholds", []))
    try:
        fingerprint, payload = publication_analysis_fingerprint(
            unified_csv_path,
            strict_settings,
            references,
            excluded,
            thresholds,
        )
    except Exception as exc:
        st.error(f"Could not fingerprint the publication analysis: {exc}")
        return None

    session_analysis = st.session_state.get("_f1_paper_publication_analysis")
    if (
        not isinstance(session_analysis, dict)
        or session_analysis.get("fingerprint") != fingerprint
    ):
        session_analysis = None
    cache_available = cached_publication_analysis(output_dir, fingerprint)
    if session_analysis is None and cache_available:
        try:
            session_analysis = load_publication_analysis(output_dir, fingerprint)
            st.session_state["_f1_paper_publication_analysis"] = session_analysis
        except Exception as exc:
            st.warning(f"Could not automatically load the matching analysis: {exc}")
    button_columns = st.columns(2)
    run_requested = button_columns[0].button(
        "Run / rebuild shared ML prediction data",
        type="primary",
        width="stretch",
        help="Fits every requested model inside every held-out-group fold, then saves reusable OOF tables.",
    )
    load_requested = button_columns[1].button(
        "Reload matching saved analysis",
        width="stretch",
        disabled=not cache_available,
    )
    if run_requested:
        progress_lines: list[str] = []
        try:
            with st.status(
                "Training leakage-resistant held-out models...", expanded=True
            ) as status:
                frame = _read_unified_frame(unified_csv_path)

                def progress(message: str) -> None:
                    progress_lines.append(str(message))
                    status.write(str(message))
                    log_console(f"[{timestamp()}] {message}")

                analysis = run_publication_analysis(
                    frame,
                    settings=strict_settings,
                    reference_columns=references,
                    excluded_labels=excluded,
                    method_display_names=common.get("method_labels", {}),
                    confidence_thresholds=thresholds,
                    progress_callback=progress,
                )
                session_analysis = save_publication_analysis(
                    analysis,
                    output_dir,
                    fingerprint,
                    payload,
                )
                st.session_state["_f1_paper_publication_analysis"] = session_analysis
                st.session_state["_f1_paper_publication_log"] = progress_lines
                _persist_paper_settings(project_root, cached_params)
                status.update(label="Publication validation complete", state="complete")
        except Exception as exc:
            st.error(f"Publication validation failed: {exc}")
            session_analysis = None
    elif load_requested:
        try:
            session_analysis = load_publication_analysis(output_dir, fingerprint)
            st.session_state["_f1_paper_publication_analysis"] = session_analysis
        except Exception as exc:
            st.error(f"Could not load the saved analysis: {exc}")
            session_analysis = None

    if session_analysis is None:
        if cache_available:
            st.info(
                "A matching completed analysis is available on disk. Load it, or rebuild it with the current settings."
            )
        else:
            st.warning(
                "No matching shared OOF prediction run exists yet. ML-dependent figure buttons remain disabled until it is run."
            )
        if strict_settings.get("quick_mode"):
            st.warning(
                "Fast preview is enabled. Preview results are implementation checks and should not be reported as final publication values."
            )
        return None

    oof = session_analysis["oof_predictions"]
    groups = (
        list(dict.fromkeys(oof["validation_group"].dropna().astype(str)))
        if "validation_group" in oof
        else []
    )
    st.success(
        f"Shared ML data are ready from cache `{fingerprint}`: {len(oof):,} scored cells across {len(groups)} held-out group(s). All compatible figure panels will reuse these outputs."
    )
    if strict_settings.get("quick_mode"):
        st.warning(
            "This cache was configured as a fast preview. Rebuild with Fast preview off before reporting final values."
        )
    with st.expander("Leakage audit and reusable analysis files", expanded=False):
        cache_directory = str(session_analysis.get("cache_directory", ""))
        st.caption(f"Cache directory: {cache_directory}")
        if cache_directory:
            st.code(
                "\n".join(
                    str(Path(cache_directory) / filename)
                    for filename in TABLE_FILES.values()
                )
            )
        audit = session_analysis.get("fold_audit")
        if isinstance(audit, pd.DataFrame):
            st.dataframe(audit, width="stretch", hide_index=True)
    return session_analysis


def _sanitize_panel_state(spec: PaperPanelSpec, common: Mapping[str, Any]) -> None:
    methods = _ordered_methods(common)
    references = _ordered_references(common)
    confidence_methods = list(common.get("confidence_columns", {}))
    for setting in ("focus_reference", "spatial_reference"):
        key = _panel_key(spec.panel_id, setting)
        if references and st.session_state.get(key) not in references:
            st.session_state[key] = references[0]
    for setting in ("shade_methods", "spatial_methods", "confusion_methods"):
        key = _panel_key(spec.panel_id, setting)
        current = st.session_state.get(key, [])
        if not isinstance(current, (list, tuple)):
            current = []
        label_to_method = {
            _method_label(common, method): method for method in methods
        }
        resolved = [
            str(value)
            if str(value) in methods
            else label_to_method.get(str(value), "")
            for value in current
        ]
        st.session_state[key] = list(
            dict.fromkeys(value for value in resolved if value in methods)
        )
    for setting in ("confidence_method", "learning_method"):
        key = _panel_key(spec.panel_id, setting)
        options = confidence_methods if setting == "confidence_method" else methods
        if options and st.session_state.get(key) not in options:
            st.session_state[key] = options[0]


def _render_universal_panel_form(
    spec: PaperPanelSpec,
    project_root: Path,
    cached_params: dict[str, Any],
    unified_csv_path: str,
    analysis: Mapping[str, Any] | None,
    strict_settings: Mapping[str, Any],
    *,
    use_shared_predictions: bool,
) -> None:
    _initialize_panel_state(spec, cached_params)
    common = _common_settings_from_state()
    common["consensus_min_agree"] = int(
        strict_settings.get(
            "consensus_min_methods",
            common.get("consensus_min_agree", 3),
        )
    )
    uses_shared_data = use_shared_predictions and spec.panel_id != "main_workflow"
    if uses_shared_data and analysis is not None:
        common = _publication_common_settings(common, strict_settings, analysis)
    _sanitize_panel_state(spec, common)
    methods = _ordered_methods(common)
    references = _ordered_references(common)
    confidence_methods = list(common.get("confidence_columns", {}))
    available, unavailable_reason = generic_panel_availability(spec.panel_id, common)
    if uses_shared_data and analysis is None:
        available = False
        unavailable_reason = (
            "Run or load the shared ML prediction data above. This "
            "figure requires newly trained out-of-fold ML predictions and will not "
            "use saved ML columns from the unified CSV."
        )
    preview_key = f"_f1_paper_preview_{spec.panel_id}"
    paths_key = f"_f1_paper_paths_{spec.panel_id}"
    error_key = f"_f1_paper_error_{spec.panel_id}"
    with st.expander(GENERIC_PANEL_LABELS[spec.panel_id], expanded=False):
        st.info(GENERIC_PANEL_DESCRIPTIONS[spec.panel_id])
        if not available:
            st.warning(unavailable_reason)
        elif uses_shared_data and analysis is not None:
            st.success(
                "Reusable data are ready from shared OOF run "
                f"`{analysis.get('fingerprint', '')}`. Generating this figure will "
                "reuse those held-out predictions without fitting the models again. "
                "Use Run / rebuild shared ML prediction data above only when you want "
                "to retrain the shared data."
            )
        else:
            st.success(
                "This figure's data can be derived directly from the mapped manual "
                "references and component-method columns; no ML refitting is needed."
            )
        st.caption(
            "Controls are deferred: changing them does not render a figure. Generation reads the unified CSV only after you press the button below."
        )
        with st.form(
            f"viz_f1_paper_universal_form_{spec.panel_id}", clear_on_submit=False
        ):
            heading_columns = st.columns([2, 1, 1])
            with heading_columns[0]:
                st.text_area(
                    "Figure title", height=85, key=_panel_key(spec.panel_id, "title")
                )
                st.text_input(
                    "Output filename label",
                    key=_panel_key(spec.panel_id, "output_stem"),
                )
            with heading_columns[1]:
                st.text_input(
                    "Panel label", key=_panel_key(spec.panel_id, "panel_label")
                )
                st.number_input(
                    "Width (inches)",
                    min_value=3.0,
                    max_value=40.0,
                    step=0.2,
                    key=_panel_key(spec.panel_id, "width"),
                )
            with heading_columns[2]:
                st.number_input(
                    "Height (inches)",
                    min_value=2.0,
                    max_value=30.0,
                    step=0.2,
                    key=_panel_key(spec.panel_id, "height"),
                )
                st.number_input(
                    "Line width",
                    min_value=0.2,
                    max_value=8.0,
                    step=0.1,
                    key=_panel_key(spec.panel_id, "line_width"),
                )

                st.number_input(
                    "Figure-title size",
                    min_value=6.0,
                    max_value=40.0,
                    step=0.2,
                    key=_panel_key(spec.panel_id, "title_font_size"),
                    help="Panel-level figure/suptitle size from the reference code; remains editable for this figure only.",
                )

            text_columns = st.columns(2)
            with text_columns[0]:
                st.text_input(
                    "X-axis label (blank keeps default)",
                    key=_panel_key(spec.panel_id, "x_label"),
                )
            with text_columns[1]:
                st.text_input(
                    "Y-axis label (blank keeps default)",
                    key=_panel_key(spec.panel_id, "y_label"),
                )

            if spec.has_legend:
                legend_columns = st.columns(3)
                with legend_columns[0]:
                    st.selectbox(
                        "Legend placement",
                        LEGEND_LOCATIONS,
                        key=_panel_key(spec.panel_id, "legend_location"),
                    )
                with legend_columns[1]:
                    st.text_input(
                        "Legend title", key=_panel_key(spec.panel_id, "legend_title")
                    )
                with legend_columns[2]:
                    st.number_input(
                        "Legend columns",
                        min_value=1,
                        max_value=8,
                        step=1,
                        key=_panel_key(spec.panel_id, "legend_columns"),
                    )

            geometry_columns = st.columns(3)
            with geometry_columns[0]:
                st.number_input(
                    "Marker size",
                    min_value=1.0,
                    max_value=30.0,
                    step=0.5,
                    key=_panel_key(spec.panel_id, "marker_size"),
                )
                st.slider(
                    "Grid opacity",
                    0.0,
                    1.0,
                    step=0.01,
                    key=_panel_key(spec.panel_id, "grid_alpha"),
                    help="Panel-specific reference-code grid opacity.",
                )
                st.number_input(
                    "Plot-box aspect (0 = automatic)",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.05,
                    key=_panel_key(spec.panel_id, "box_aspect"),
                    help="Reference held-out plots use height/width = 0.6; set 0 to let Matplotlib choose from the figure dimensions.",
                )
            if spec.method_axis:
                with geometry_columns[1]:
                    st.multiselect(
                        "Shade method columns",
                        methods,
                        key=_panel_key(spec.panel_id, "shade_methods"),
                    )
                with geometry_columns[2]:
                    st.color_picker(
                        "Shade color", key=_panel_key(spec.panel_id, "shade_color")
                    )
                    st.slider(
                        "Shade opacity",
                        0.0,
                        1.0,
                        step=0.05,
                        key=_panel_key(spec.panel_id, "shade_alpha"),
                    )
            if spec.bar_grouping:
                with geometry_columns[1]:
                    st.selectbox(
                        "Bar grouping",
                        ["Series within categories", "Categories within series"],
                        key=_panel_key(spec.panel_id, "bar_grouping"),
                    )
                with geometry_columns[2]:
                    st.number_input(
                        "Bar width",
                        min_value=0.03,
                        max_value=0.8,
                        step=0.01,
                        key=_panel_key(spec.panel_id, "bar_width"),
                    )

            if spec.heatmap or spec.panel_id in {
                "supp_reference_distribution",
                "supp_human",
            }:
                heatmap_columns = st.columns(4)
                with heatmap_columns[0]:
                    st.selectbox(
                        "Color map",
                        HEATMAP_CMAPS,
                        key=_panel_key(spec.panel_id, "heatmap_cmap"),
                    )
                with heatmap_columns[1]:
                    st.checkbox(
                        "Automatic color range",
                        key=_panel_key(spec.panel_id, "heatmap_auto_range"),
                    )
                    st.number_input(
                        "Color minimum",
                        step=0.05,
                        key=_panel_key(spec.panel_id, "heatmap_vmin"),
                    )
                    st.number_input(
                        "Color maximum",
                        step=0.05,
                        key=_panel_key(spec.panel_id, "heatmap_vmax"),
                    )
                with heatmap_columns[2]:
                    st.checkbox(
                        "Show cell values",
                        key=_panel_key(spec.panel_id, "heatmap_annotations"),
                    )
                    st.checkbox(
                        "Square cells", key=_panel_key(spec.panel_id, "heatmap_square")
                    )
                    st.number_input(
                        "Heatmap-value size",
                        min_value=4.0,
                        max_value=24.0,
                        step=0.5,
                        key=_panel_key(spec.panel_id, "heatmap_annotation_size"),
                        help="Font size of numbers printed inside heatmap cells; the reference-code default is 9.5 pt (11 pt for the pairwise matrix).",
                    )
                with heatmap_columns[3]:
                    st.text_input(
                        "Colorbar label (blank keeps default)",
                        key=_panel_key(spec.panel_id, "colorbar_label"),
                    )
                    st.number_input(
                        "Cell border width",
                        min_value=0.0,
                        max_value=5.0,
                        step=0.1,
                        key=_panel_key(spec.panel_id, "heatmap_linewidth"),
                    )

            reference_panels = {
                "main_ref2_metrics",
                "supp_ref1_f1",
                "supp_confusion",
                "supp_nonconsensus_threshold",
                "supp_bins_ref2",
                "supp_learning",
            }
            if spec.panel_id in reference_panels and references:
                st.selectbox(
                    "Reference used by this figure",
                    references,
                    format_func=lambda value: _reference_label(common, value),
                    key=_panel_key(spec.panel_id, "focus_reference"),
                )
            if spec.panel_id == "supp_ref1_f1":
                st.selectbox(
                    "Per-class metric",
                    ["f1", "precision", "recall"],
                    format_func=lambda value: value.replace("_", " ").title(),
                    key=_panel_key(spec.panel_id, "class_metric"),
                )

            if spec.panel_id == "main_spatial" and references:
                spatial_columns = st.columns(4)
                with spatial_columns[0]:
                    st.selectbox(
                        "Spatial reference",
                        references,
                        format_func=lambda value: _reference_label(common, value),
                        key=_panel_key(spec.panel_id, "spatial_reference"),
                    )
                    st.number_input(
                        "Cell point size",
                        min_value=0.1,
                        max_value=100.0,
                        step=0.5,
                        key=_panel_key(spec.panel_id, "point_size"),
                    )
                with spatial_columns[1]:
                    st.multiselect(
                        "Methods (first two shown)",
                        methods,
                        key=_panel_key(spec.panel_id, "spatial_methods"),
                    )
                with spatial_columns[2]:
                    st.color_picker(
                        "Match color", key=_panel_key(spec.panel_id, "match_color")
                    )
                    st.color_picker(
                        "Mismatch color",
                        key=_panel_key(spec.panel_id, "mismatch_color"),
                    )
                with spatial_columns[3]:
                    st.color_picker(
                        "Unscored color",
                        key=_panel_key(spec.panel_id, "unscored_color"),
                    )

            if spec.panel_id == "supp_confusion" and methods:
                st.multiselect(
                    "Methods (first two shown)",
                    methods,
                    key=_panel_key(spec.panel_id, "confusion_methods"),
                )
            if (
                spec.panel_id in {"supp_nonconsensus_threshold", "supp_bins_ref2"}
                and confidence_methods
            ):
                st.selectbox(
                    "Confidence source method",
                    confidence_methods,
                    format_func=lambda value: _method_label(common, value),
                    key=_panel_key(spec.panel_id, "confidence_method"),
                )
            if spec.panel_id == "supp_learning" and methods:
                st.selectbox(
                    "Full-data horizontal comparison method",
                    methods,
                    format_func=lambda value: _method_label(common, value),
                    key=_panel_key(spec.panel_id, "learning_method"),
                )

            if spec.panel_id in {"supp_nonconsensus_threshold", "supp_bins_ref2"}:
                sensitivity_columns = st.columns(2)
                with sensitivity_columns[0]:
                    st.number_input(
                        "Minimum cells required to plot F1",
                        min_value=1,
                        max_value=10000,
                        step=1,
                        key=_panel_key(spec.panel_id, "minimum_bin_cells"),
                    )
                if spec.panel_id == "supp_nonconsensus_threshold":
                    with sensitivity_columns[1]:
                        st.slider(
                            "Selected confidence marker",
                            0.0,
                            1.0,
                            step=0.01,
                            key=_panel_key(spec.panel_id, "selected_confidence"),
                        )

            if spec.panel_id == "main_ref2_metrics":
                metric_columns = st.columns(3)
                for index, label in enumerate(["Macro-F1", "Micro-F1", "Weighted-F1"]):
                    with metric_columns[index]:
                        st.color_picker(
                            label,
                            key=_panel_key(spec.panel_id, f"metric_color_{index}"),
                        )

            if spec.panel_id == "main_workflow":
                with st.expander("Workflow box and footer text", expanded=False):
                    for name, label in [
                        ("components", "Component-method box"),
                        ("mr_source", "Reference source box"),
                        ("consensus_source", "Consensus source box"),
                        ("mr_training", "Primary route box"),
                        ("consensus_training", "Secondary route box"),
                        ("holdout", "Validation/grouping box"),
                        ("comparison", "Comparison box"),
                        ("footer", "Footer"),
                    ]:
                        st.text_area(
                            label,
                            height=70,
                            key=_panel_key(spec.panel_id, f"workflow_{name}"),
                        )

            submitted = st.form_submit_button(
                (
                    "Generate figure from reusable OOF data"
                    if uses_shared_data
                    else "Generate figure from mapped source data"
                ),
                type="primary",
                disabled=not available,
            )

        if submitted:
            formats = list(st.session_state.get("viz_f1_paper_formats", []))
            if not formats:
                st.session_state[error_key] = (
                    "Select PNG, SVG, or both in Shared Figure Settings."
                )
            else:
                figure: plt.Figure | None = None
                try:
                    common = _common_settings_from_state()
                    common["consensus_min_agree"] = int(
                        strict_settings.get(
                            "consensus_min_methods",
                            common.get("consensus_min_agree", 3),
                        )
                    )
                    if uses_shared_data:
                        if analysis is None:
                            raise ValueError(
                                "Run or load the shared ML prediction data before generating this figure."
                            )
                        common = _publication_common_settings(
                            common,
                            strict_settings,
                            analysis,
                        )
                    available_now, reason_now = generic_panel_availability(
                        spec.panel_id, common
                    )
                    if not available_now:
                        raise ValueError(reason_now)
                    frame = (
                        _publication_analysis_frame(
                            analysis,
                            common,
                            unified_csv_path,
                        )
                        if uses_shared_data and analysis is not None
                        else _read_unified_frame(unified_csv_path)
                    )
                    panel = _panel_settings_from_state(spec)
                    figure = build_generic_paper_figure(
                        spec.panel_id,
                        frame,
                        common,
                        panel,
                        analysis_tables=analysis,
                    )
                    settings_payload = {
                        "panel_id": spec.panel_id,
                        "input_csv": unified_csv_path,
                        "publication_analysis_fingerprint": (
                            analysis.get("fingerprint")
                            if isinstance(analysis, Mapping)
                            else None
                        ),
                        "common": common,
                        "panel": panel,
                    }
                    saved, preview = save_paper_figure(
                        figure,
                        st.session_state["viz_f1_paper_output_dir"],
                        str(panel["output_stem"]),
                        formats,
                        dpi=int(common["dpi"]),
                        settings=settings_payload
                        if st.session_state.get("viz_f1_paper_save_settings_json", True)
                        else None,
                    )
                    st.session_state[preview_key] = preview
                    st.session_state[paths_key] = [str(path) for path in saved]
                    st.session_state.pop(error_key, None)
                    _persist_paper_settings(project_root, cached_params)
                    log_console(
                        f"[{timestamp()}] Saved universal F1 figure {spec.panel_id} to {saved[0].parent if saved else st.session_state['viz_f1_paper_output_dir']}"
                    )
                except Exception as exc:
                    st.session_state[error_key] = str(exc)
                finally:
                    if figure is not None:
                        plt.close(figure)

        if error_key in st.session_state:
            st.error(st.session_state[error_key])
        preview = st.session_state.get(preview_key)
        if isinstance(preview, bytes):
            st.image(
                preview,
                caption=f"Last generated preview — {GENERIC_PANEL_LABELS[spec.panel_id]}",
                width="stretch",
            )
        saved_paths = st.session_state.get(paths_key)
        if isinstance(saved_paths, list) and saved_paths:
            st.success("Saved the latest figure with a unique label.")
            st.code("\n".join(str(path) for path in saved_paths))


def _render_generated_figure_gallery() -> None:
    generated: list[tuple[str, bytes, list[str]]] = []
    for panel_id in GENERIC_PANEL_IDS:
        preview = st.session_state.get(f"_f1_paper_preview_{panel_id}")
        paths = st.session_state.get(f"_f1_paper_paths_{panel_id}", [])
        if isinstance(preview, bytes):
            generated.append(
                (
                    GENERIC_PANEL_LABELS[panel_id],
                    preview,
                    [str(path) for path in paths] if isinstance(paths, list) else [],
                )
            )
    if not generated:
        return
    st.markdown("### Generated figure gallery")
    st.caption(
        "Every figure generated during this UI session remains visible here while "
        "you work in other panels. Regenerating the same figure type replaces only "
        "that type's previous preview; figures from other panels remain."
    )
    columns = st.columns(2)
    for index, (label, preview, paths) in enumerate(generated):
        with columns[index % 2]:
            st.image(preview, caption=label, width="stretch")
            if paths:
                st.caption(Path(paths[0]).name)


def render_shared_ml_prediction_zone(
    project_root: Path,
    cached_params: dict[str, Any],
    output_dir_default: str,
    unified_csv_path: str,
    available_columns: list[str],
    reference_options: list[str],
    method_options: list[str],
    group_options: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Render the single ML configuration/run shared by both figure workflows."""
    default_output = Path(output_dir_default) / "F1_validation"
    _initialize_state(
        "viz_f1_paper_output_dir",
        str(
            st.session_state.get(
                "viz_f1_output_dir",
                cached_params.get("viz_f1_output_dir", default_output),
            )
        ),
        cached_params,
    )
    st.markdown("### Shared held-out ML settings and saved prediction data")
    st.info(
        "Choose MR-ML, Consensus-ML, or both here. This is the only model-training "
        "zone used by Option 1 and Option 2. A completed run is saved under the F1 "
        "output folder and reused by either figure workflow until a scientific "
        "setting or the input CSV changes."
    )
    all_columns = [str(column) for column in available_columns]
    _valid_multiselect_state(
        "viz_f1_shared_reference_columns",
        all_columns,
        list(reference_options),
        cached_params,
    )
    _valid_multiselect_state(
        "viz_f1_shared_component_columns",
        all_columns,
        list(method_options),
        cached_params,
    )
    _valid_multiselect_state(
        "viz_f1_shared_group_columns",
        all_columns,
        list(group_options),
        cached_params,
    )
    with st.expander("Map arbitrary CSV columns to shared ML roles", expanded=True):
        role_columns = st.columns(3)
        with role_columns[0]:
            selected_references = st.multiselect(
                "Available manual-reference columns",
                all_columns,
                key="viz_f1_shared_reference_columns",
                help="Select every human/manual phenotype column that may be scored. The primary training reference is chosen below.",
            )
        with role_columns[1]:
            selected_components = st.multiselect(
                "Available component phenotype-call columns",
                all_columns,
                key="viz_f1_shared_component_columns",
                help="Select categorical phenotype outputs such as Leiden, SOM, Vortex, or any arbitrarily named method. Do not select marker-intensity columns.",
            )
        with role_columns[2]:
            selected_groups = st.multiselect(
                "Available patient/slide/image grouping columns",
                all_columns,
                key="viz_f1_shared_group_columns",
                help="Select columns that can define complete held-out biological groups. The actual column and fold mapping are chosen below.",
            )
    if not selected_references:
        message = "Select at least one manual-reference column for shared ML."
        st.error(message)
        return {"enabled": True, "configuration_error": message}, None
    if len(selected_components) < 2:
        message = "Select at least two component phenotype-call columns for shared ML."
        st.error(message)
        return {"enabled": True, "configuration_error": message}, None
    if not selected_groups:
        message = "Select at least one grouping column for held-out validation."
        st.error(message)
        return {"enabled": True, "configuration_error": message}, None
    strict_settings = render_strict_retraining_panel(
        unified_csv_path=unified_csv_path,
        available_columns=[str(column) for column in available_columns],
        reference_options=list(selected_references),
        method_options=list(selected_components),
        group_options=list(selected_groups),
        cached_params=cached_params,
        force_enabled=True,
        title="Configure the shared held-out ML run",
        expanded=True,
    )
    strict_settings = _render_publication_learning_settings(
        cached_params,
        strict_settings,
    )
    shared_common = {
        "reference_order": list(selected_references),
        "excluded_labels": set(INVALID_LABELS),
        "confidence_thresholds": [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95],
        "method_labels": {
            method: _method_default(method, index)[0]
            for index, method in enumerate(selected_components)
        },
    }
    analysis = _render_publication_analysis_controls(
        project_root,
        cached_params,
        unified_csv_path,
        shared_common,
        strict_settings,
    )
    return strict_settings, analysis


def render_paper_figure_panels(
    project_root: Path,
    cached_params: dict[str, Any],
    output_dir_default: str,
    unified_csv_path: str | None,
    available_columns: list[str],
    *,
    use_shared_predictions: bool,
    shared_analysis: Mapping[str, Any] | None = None,
    shared_strict_settings: Mapping[str, Any] | None = None,
) -> None:
    """Render universal, deferred paper-style figures from the unified CSV."""
    st.markdown("### Customizable paper-style figures")
    st.caption(
        "Each figure preserves the publication layout while using column mappings from the active unified CSV. Figure types are independent and are generated only when requested."
    )
    if not unified_csv_path or not Path(unified_csv_path).is_file():
        st.error("A valid unified CSV is required for Option 2 figures.")
        return
    _render_universal_shared_settings_form(
        project_root,
        cached_params,
        output_dir_default,
        str(unified_csv_path),
        available_columns,
        use_shared_predictions=use_shared_predictions,
    )
    common = _common_settings_from_state()
    reference_options = list(common.get("reference_order", []))
    method_options = list(common.get("method_order", []))
    if not reference_options:
        st.error(
            "Select at least one manual-reference column in Shared Figure Settings, "
            "then save the form."
        )
        return
    if len(method_options) < 2:
        st.error(
            "Select at least two component phenotype columns in Shared Figure "
            "Settings, then save the form."
        )
        return
    strict_settings = dict(shared_strict_settings or {})
    analysis = shared_analysis if use_shared_predictions else None
    st.markdown("### Step 2 — Independent figure panels")
    st.caption(
        "Each panel has its own title, dimensions, colors, legend, shading, and "
        "output filename. ML-dependent panels read the shared saved outputs above; "
        "their Generate button renders and saves only that figure."
    )
    for panel_id in GENERIC_PANEL_IDS:
        _render_universal_panel_form(
            PANEL_BY_ID[panel_id],
            project_root,
            cached_params,
            str(unified_csv_path),
            analysis,
            strict_settings,
            use_shared_predictions=use_shared_predictions,
        )
    _render_generated_figure_gallery()
