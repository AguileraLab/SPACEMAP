"""Pure analysis and plotting helpers for the Tab 9 phenotyping-validation UI.

The functions in this module never mutate the input table and never write files.
This keeps statistical calculations testable and lets the Streamlit layer require
an explicit save action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
import re
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


INVALID_LABELS = {
    "",
    "nan",
    "none",
    "null",
    "unknown",
    "unclassified",
    "<na>",
}
UNASSIGNED_LABEL = "Other / unassigned"
ROUTE_ORDER = ["End-to-end", "Direct consensus", "Non-consensus"]
GROUP_POOLED = "Pooled"
ROI_SUFFIX_PATTERN = re.compile(r"(?i)(?:[_\-\s]+roi[_\-\s]*\d+)$")
EXCLUDED_GROUP_TOKEN = "<exclude>"

GUI_CLASSIC_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
COLORBLIND_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
PALETTES = {
    "GUI classic": GUI_CLASSIC_COLORS,
    "Colorblind": COLORBLIND_COLORS,
    "Deep": list(sns.color_palette("deep", 10).as_hex()),
    "Muted": list(sns.color_palette("muted", 10).as_hex()),
}

# Semantic method colors used by the reference Figure 3 / Option 2 figures.
# Option 1 resolves these by method identity rather than list position, so
# reordering or hiding methods never changes their established colors.
OPTION2_METHOD_COLORS = {
    "Leiden": "#1f77b4",
    "SOM": "#d62728",
    "Vortex": "#9467bd",
    "RESOLVE": "#2ca02c",
    "ML-MR": "#7B2CBF",
    "Consensus-ML": "#008C8C",
}


@dataclass(frozen=True)
class ValidationStyle:
    """Editable publication styling shared by every validation figure."""

    font_family: str = "DejaVu Sans"
    base_font_size: float = 10.0
    title_font_size: float = 14.0
    axis_font_size: float = 11.0
    tick_font_size: float = 9.0
    annotation_font_size: float = 8.0
    heatmap_cmap: str = "Blues"
    heatmap_vmin: float = 0.5
    heatmap_vmax: float = 1.0
    show_annotations: bool = True
    square_cells: bool = False  # force each heatmap cell to a perfect square
    figure_width: float = 10.0
    figure_height: float = 6.0
    row_height: float = 0.45
    palette_name: str = "GUI classic"
    method_colors: Mapping[str, str] = field(default_factory=dict)
    dpi: int = 300
    title_prefix: str = ""

    def matplotlib_context(self) -> dict[str, Any]:
        return {
            "font.family": self.font_family,
            "font.size": self.base_font_size,
            "axes.titlesize": self.title_font_size,
            "axes.labelsize": self.axis_font_size,
            "xtick.labelsize": self.tick_font_size,
            "ytick.labelsize": self.tick_font_size,
            "legend.fontsize": self.tick_font_size,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }


def style_to_dict(style: ValidationStyle) -> dict[str, Any]:
    return asdict(style)


def clean_label_series(series: pd.Series) -> pd.Series:
    """Return trimmed nullable labels while preserving the input index."""

    cleaned = series.astype("string").str.strip()
    lowered = cleaned.str.lower()
    return cleaned.mask(lowered.isin(INVALID_LABELS))


def normalized_exclusions(labels: Iterable[str]) -> set[str]:
    return {
        str(value).strip().lower()
        for value in labels
        if str(value).strip()
    }


def valid_truth_mask(series: pd.Series, excluded_labels: Iterable[str]) -> pd.Series:
    cleaned = clean_label_series(series)
    excluded = normalized_exclusions(excluded_labels)
    mask = cleaned.notna()
    if excluded:
        mask &= ~cleaned.str.lower().isin(excluded)
    return mask.fillna(False)


def labeled_mask_including_exclusions(series: pd.Series) -> pd.Series:
    return clean_label_series(series).notna().fillna(False)


def normalize_predictions(
    series: pd.Series,
    excluded_labels: Iterable[str],
) -> tuple[pd.Series, pd.Series]:
    """Normalize predictions and return labels plus an assigned-cell mask."""

    cleaned = clean_label_series(series)
    excluded = normalized_exclusions(excluded_labels)
    assigned = cleaned.notna()
    if excluded:
        assigned &= ~cleaned.str.lower().isin(excluded)
    normalized = cleaned.where(assigned, UNASSIGNED_LABEL).astype(str)
    return normalized, assigned.fillna(False)


def consensus_direct_mask(series: pd.Series) -> pd.Series:
    """Interpret common Boolean, numeric, or route-label consensus columns."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_known = numeric.notna()
    result = pd.Series(False, index=series.index, dtype=bool)
    if numeric_known.any():
        result.loc[numeric_known] = numeric.loc[numeric_known].gt(0)

    text = series.astype("string").str.strip().str.lower()
    direct_values = {
        "true",
        "yes",
        "y",
        "consensus",
        "direct",
        "direct consensus",
        "direct_consensus",
        "1",
        "1.0",
    }
    result.loc[~numeric_known] = text.loc[~numeric_known].isin(direct_values)
    return result.fillna(False)


def prettify_column_name(column: str) -> str:
    """Convert common SPACEMAP column names to concise figure labels."""

    exact = {
        "phenotype_leiden": "Leiden",
        "phenotype_som": "SOM",
        "phenotype_vortex": "Vortex",
        "phenotype_threshold": "RESOLVE",
        "phenotype_resolve": "RESOLVE",
        "phenotype_scimap": "SCIMAP",
        "phenotype_ml_mr": "MR-ML",
        "phenotype_ml_mr_refined": "MR-ML refined",
        "phenotype_ml_consensus": "Consensus-ML",
        "phenotype_ml_consensus_refined": "Consensus-ML refined",
        "prediction_mr_ml_strict": "MR-ML",
        "prediction_mr_ml_strict_refined": "MR-ML refined",
        "prediction_consensus_ml_strict": "Consensus-ML",
        "prediction_consensus_ml_strict_refined": "Consensus-ML refined",
        "consensus_phenotype_recomputed": "Direct consensus",
        "phenotype_ref1": "Ref1",
        "phenotype_ref1a": "Ref1a",
        "phenotype_ref1b": "Ref1b",
        "phenotype_ref2": "Ref2",
        "phenotype_bd_x": "BD-x",
        "phenotype_bd_y": "BD",
        "phenotype_ap": "AP",
    }
    if column in exact:
        return exact[column]

    value = str(column)
    for prefix in ("prediction_", "phenotype_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.replace("_strict_refined", " refined")
    value = value.replace("_strict", "")
    value = value.replace("_", " ").strip()
    substitutions = {
        "ml mr": "MR-ML",
        "ml consensus": "Consensus-ML",
        "mr ml": "MR-ML",
        "ap": "AP",
        "bd": "BD",
        "som": "SOM",
    }
    return substitutions.get(value.lower(), value.title())


def parse_mapping_text(text: str) -> dict[str, str]:
    """Parse one column = value mapping per line."""

    mapping: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            left, right = line.split("=", 1)
        elif ":" in line:
            left, right = line.split(":", 1)
        else:
            continue
        left = left.strip()
        right = right.strip()
        if left and right:
            mapping[left] = right
    return mapping


def mapping_to_text(mapping: Mapping[str, str]) -> str:
    return "\n".join(f"{key} = {value}" for key, value in mapping.items())


def _looks_like_reference(column: str) -> bool:
    value = column.lower()
    return (
        "phenotype_ref" in value
        or value.startswith("reference_")
        or "ground_truth" in value
        or value.endswith("_ground_truth")
        or value in {
            "phenotype_bd",
            "phenotype_bd_x",
            "phenotype_bd_y",
            "phenotype_ap",
            "phenotype_ref3",
            "phenotype_ref3_adjudicated",
        }
    )


def reference_candidates(columns: Sequence[str]) -> list[str]:
    preferred = [
        "phenotype_ref1",
        "phenotype_ref1a",
        "phenotype_ref1b",
        "phenotype_ref2",
        "phenotype_bd_y",
        "phenotype_ap",
        "phenotype_ref3",
        "phenotype_ref3_adjudicated",
        "phenotype_bd_x",
    ]
    candidates = [column for column in columns if _looks_like_reference(str(column))]
    ordered = [column for column in preferred if column in candidates]
    ordered.extend(column for column in candidates if column not in ordered)
    return ordered


def prediction_candidates(columns: Sequence[str]) -> list[str]:
    preferred = [
        "phenotype_leiden",
        "phenotype_som",
        "phenotype_vortex",
        "phenotype_threshold",
        "phenotype_scimap",
        "phenotype_ml_mr",
        "phenotype_ml_consensus",
        "prediction_mr_ml_strict",
        "prediction_consensus_ml_strict",
        "consensus_phenotype_recomputed",
        "phenotype_ml_mr_refined",
        "phenotype_ml_consensus_refined",
        "prediction_mr_ml_strict_refined",
        "prediction_consensus_ml_strict_refined",
    ]
    references = set(reference_candidates(columns))
    candidates = [
        str(column)
        for column in columns
        if (
            str(column).startswith("phenotype_")
            or str(column).startswith("prediction_")
            or str(column) == "consensus_phenotype_recomputed"
        )
        and str(column) not in references
        and "confidence" not in str(column).lower()
        and not str(column).lower().endswith("_score")
    ]
    ordered = [column for column in preferred if column in candidates]
    ordered.extend(column for column in candidates if column not in ordered)
    return ordered


def confidence_candidates(columns: Sequence[str]) -> list[str]:
    preferred = [
        "mr_ml_confidence_strict",
        "consensus_ml_confidence_strict",
        "ml_mr_confidence",
        "ml_consensus_score",
    ]
    candidates = [
        str(column)
        for column in columns
        if "confidence" in str(column).lower()
        or str(column).lower().endswith("_score")
        or "support" in str(column).lower()
    ]
    ordered = [column for column in preferred if column in candidates]
    ordered.extend(column for column in candidates if column not in ordered)
    return ordered


def group_candidates(columns: Sequence[str]) -> list[str]:
    preferred = [
        "validation_group",
        "patient_id",
        "patient",
        "slide_id",
        "slide",
        "batch_id",
        "tissue_roi",
        "original_filename",
        "fold",
    ]
    candidates: list[str] = []
    for column in columns:
        value = str(column)
        lower = value.lower()
        if (
            value in preferred
            or "patient" in lower
            or "slide" in lower
            or "batch" in lower
            or lower.endswith("_group")
        ):
            candidates.append(value)
    ordered = [column for column in preferred if column in candidates]
    ordered.extend(column for column in candidates if column not in ordered)
    return ordered


def route_candidates(columns: Sequence[str]) -> list[str]:
    preferred = [
        "is_consensus_recomputed",
        "consensus",
        "is_consensus",
        "consensus_route",
    ]
    candidates = [
        str(column)
        for column in columns
        if (
            "consensus" in str(column).lower()
            and not str(column).startswith("phenotype_")
            and not str(column).startswith("prediction_")
            and "confidence" not in str(column).lower()
            and "count" not in str(column).lower()
        )
    ]
    ordered = [column for column in preferred if column in candidates]
    ordered.extend(column for column in candidates if column not in ordered)
    return ordered


def default_method_columns(columns: Sequence[str]) -> list[str]:
    candidates = prediction_candidates(columns)
    if "prediction_mr_ml_strict" in candidates:
        preferred = [
            "phenotype_leiden",
            "phenotype_som",
            "phenotype_vortex",
            "phenotype_threshold",
            "prediction_mr_ml_strict",
            "prediction_consensus_ml_strict",
        ]
    else:
        preferred = [
            "phenotype_leiden",
            "phenotype_som",
            "phenotype_vortex",
            "phenotype_threshold",
            "phenotype_ml_mr",
            "phenotype_ml_consensus",
        ]
    selected = [column for column in preferred if column in candidates]
    return selected or candidates[:6]


def default_confidence_mapping(columns: Sequence[str]) -> dict[str, str]:
    available = set(str(column) for column in columns)
    pairs = [
        ("prediction_mr_ml_strict", "mr_ml_confidence_strict"),
        ("prediction_mr_ml_strict_refined", "mr_ml_confidence_strict"),
        ("prediction_consensus_ml_strict", "consensus_ml_confidence_strict"),
        (
            "prediction_consensus_ml_strict_refined",
            "consensus_ml_confidence_strict",
        ),
        ("phenotype_ml_mr", "ml_mr_confidence"),
        ("phenotype_ml_mr_refined", "ml_mr_confidence"),
        ("phenotype_ml_consensus", "ml_consensus_score"),
        ("phenotype_ml_consensus_refined", "ml_consensus_score"),
    ]
    return {
        method: confidence
        for method, confidence in pairs
        if method in available and confidence in available
    }


def build_validation_groups(
    series: pd.Series,
    mode: str,
    manual_mapping: Mapping[str, str] | None = None,
) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    if mode == "Remove ROI suffix":
        return cleaned.map(
            lambda value: ROI_SUFFIX_PATTERN.sub("", str(value)).strip()
            if pd.notna(value)
            else pd.NA
        ).astype("string")
    if mode == "Manual mapping":
        mapping = {str(key): str(value) for key, value in (manual_mapping or {}).items()}

        def _resolve(value):
            if pd.isna(value):
                return pd.NA
            target = mapping.get(str(value), str(value))
            # Matches the strict runner: this token removes the group entirely.
            if str(target).strip().lower() == EXCLUDED_GROUP_TOKEN:
                return pd.NA
            return target

        return cleaned.map(_resolve)
    return cleaned


def _safe_kappa(y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        value = float(cohen_kappa_score(y_true, y_pred))
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _score_method(
    y_true: pd.Series,
    prediction_series: pd.Series,
    excluded_labels: Iterable[str],
    active_labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    y_pred, assigned = normalize_predictions(prediction_series, excluded_labels)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_labels,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_labels,
        average="weighted",
        zero_division=0,
    )

    accepted_accuracy = np.nan
    accepted_macro_f1 = np.nan
    if assigned.any():
        accepted_accuracy = float(
            accuracy_score(y_true.loc[assigned], y_pred.loc[assigned])
        )
        accepted_macro_f1 = float(
            f1_score(
                y_true.loc[assigned],
                y_pred.loc[assigned],
                labels=active_labels,
                average="macro",
                zero_division=0,
            )
        )

    summary = {
        "n_test": int(len(y_true)),
        "n_active_classes": int(len(active_labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(recall_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=active_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=active_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "cohen_kappa": _safe_kappa(y_true, y_pred),
        "prediction_coverage": float(assigned.mean()),
        "n_assigned": int(assigned.sum()),
        "n_rejected": int((~assigned).sum()),
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_f1": accepted_macro_f1,
    }

    per_class_rows = [
        {
            "phenotype": str(label),
            "precision": float(p_value),
            "recall": float(r_value),
            "f1": float(f_value),
            "support": int(n_value),
        }
        for label, p_value, r_value, f_value, n_value in zip(
            active_labels,
            precision,
            recall,
            class_f1,
            support,
            strict=False,
        )
    ]

    confusion_pred = y_pred.where(y_pred.isin(active_labels), UNASSIGNED_LABEL)
    confusion_labels = list(active_labels)
    if confusion_pred.eq(UNASSIGNED_LABEL).any():
        confusion_labels.append(UNASSIGNED_LABEL)
    counts = confusion_matrix(y_true, confusion_pred, labels=confusion_labels)
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )
    confusion_rows: list[dict[str, Any]] = []
    for row_index, truth in enumerate(confusion_labels):
        for column_index, prediction in enumerate(confusion_labels):
            confusion_rows.append(
                {
                    "true_label": str(truth),
                    "predicted_label": str(prediction),
                    "count": int(counts[row_index, column_index]),
                    "fraction_of_true_class": float(
                        normalized[row_index, column_index]
                    ),
                }
            )
    return summary, per_class_rows, confusion_rows


def reference_overlap_matrix(
    frame: pd.DataFrame,
    references: Sequence[str],
    reference_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Count cells carrying labels in more than one reference column.

    A cell annotated in two references is the same cell scored twice. When one
    of those references trained a model and the other is presented as its
    validation set, those rows are not independent evidence.
    """
    names = dict(reference_names or {})
    labeled = {
        reference: labeled_mask_including_exclusions(frame[reference])
        for reference in references
        if reference in frame.columns
    }
    rows: list[dict[str, Any]] = []
    for a, b in combinations(list(labeled), 2):
        shared = int((labeled[a] & labeled[b]).sum())
        if shared == 0:
            continue
        rows.append(
            {
                "reference_a": names.get(a, a),
                "reference_b": names.get(b, b),
                "n_shared_cells": shared,
                "pct_of_a": float(shared / max(int(labeled[a].sum()), 1)),
                "pct_of_b": float(shared / max(int(labeled[b].sum()), 1)),
            }
        )
    return pd.DataFrame(rows).sort_values("n_shared_cells", ascending=False) if rows else pd.DataFrame(
        columns=["reference_a", "reference_b", "n_shared_cells", "pct_of_a", "pct_of_b"]
    )


def _reference_masks(
    frame: pd.DataFrame,
    references: Sequence[str],
    excluded_labels: Sequence[str],
    remove_primary_overlap: bool,
) -> dict[str, tuple[pd.Series, int, bool]]:
    """Build the evaluable-cell mask for each reference.

    With ``remove_primary_overlap`` on, each reference drops any cell already
    labelled by an *earlier* reference in the list, not only by the primary.
    That makes the evaluation sets mutually disjoint, so selecting three or
    more references cannot leave the second and third sharing rows.
    """
    masks: dict[str, tuple[pd.Series, int, bool]] = {}
    if not references:
        return masks
    seen_labeled: pd.Series | None = None
    for reference in references:
        mask = valid_truth_mask(frame[reference], excluded_labels)
        overlap_removed = 0
        skipped = False
        if remove_primary_overlap and seen_labeled is not None:
            trimmed = mask & ~seen_labeled
            if mask.any() and not trimmed.any():
                # One reference fully contained in an earlier one - two annotators
                # scoring the same cells, for instance. Silently emptying it would
                # make the reference vanish from every figure with no explanation,
                # so keep it whole and let the audit report the containment.
                skipped = True
            else:
                overlap_removed = int((mask & seen_labeled).sum())
                mask = trimmed
        masks[reference] = (mask, overlap_removed, skipped)
        this_labeled = labeled_mask_including_exclusions(frame[reference])
        seen_labeled = this_labeled if seen_labeled is None else (seen_labeled | this_labeled)
    return masks


def analyze_validation(
    frame: pd.DataFrame,
    *,
    reference_columns: Sequence[str],
    method_columns: Sequence[str],
    excluded_labels: Sequence[str] = ("Others",),
    method_display_names: Mapping[str, str] | None = None,
    route_column: str | None = None,
    group_column: str | None = None,
    group_mode: str = "As stored",
    group_mapping: Mapping[str, str] | None = None,
    remove_primary_overlap: bool = False,
    confidence_mapping: Mapping[str, str] | None = None,
    confidence_thresholds: Sequence[float] = (
        0.0,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.9,
        0.95,
    ),
    annotator_columns: tuple[str, str] | None = None,
    prediction_design: str = "Saved predictions",
) -> dict[str, Any]:
    """Score arbitrary phenotype columns across references, routes, and groups."""

    references = [str(column) for column in reference_columns]
    methods = [str(column) for column in method_columns]
    required = set(references) | set(methods)
    for optional_column in (route_column, group_column):
        if optional_column:
            required.add(str(optional_column))
    for method, confidence in (confidence_mapping or {}).items():
        required.add(str(method))
        required.add(str(confidence))
    if annotator_columns:
        required.update(str(column) for column in annotator_columns)

    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"Required validation columns were not found: {missing}")
    if not references:
        raise ValueError("Select at least one manual-reference column.")
    if not methods:
        raise ValueError("Select at least one phenotype prediction column.")

    display_names = {
        method: (
            str(method_display_names.get(method))
            if method_display_names and method in method_display_names
            else prettify_column_name(method)
        )
        for method in methods
    }
    used_names: dict[str, int] = {}
    for method in methods:
        base = display_names[method]
        used_names[base] = used_names.get(base, 0) + 1
        if used_names[base] > 1:
            display_names[method] = f"{base} ({used_names[base]})"

    reference_names = {
        reference: prettify_column_name(reference) for reference in references
    }
    direct = (
        consensus_direct_mask(frame[route_column])
        if route_column
        else pd.Series(False, index=frame.index, dtype=bool)
    )
    routes: dict[str, pd.Series] = {
        "End-to-end": pd.Series(True, index=frame.index, dtype=bool)
    }
    if route_column:
        routes["Direct consensus"] = direct
        routes["Non-consensus"] = ~direct

    if group_column:
        validation_groups = build_validation_groups(
            frame[group_column],
            group_mode,
            group_mapping,
        )
    else:
        validation_groups = pd.Series(
            pd.NA,
            index=frame.index,
            dtype="string",
        )

    masks = _reference_masks(
        frame,
        references,
        excluded_labels,
        remove_primary_overlap,
    )
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for reference_index, reference in enumerate(references):
        base_mask, overlap_removed, overlap_skipped = masks[reference]
        raw_nonnull = frame[reference].notna()
        cleaned = clean_label_series(frame[reference])
        labeled = cleaned.notna()
        valid_before_overlap = valid_truth_mask(frame[reference], excluded_labels)
        audit_rows.append(
            {
                "reference_column": reference,
                "reference": reference_names[reference],
                "primary_reference": bool(reference_index == 0),
                "n_nonnull_raw": int(raw_nonnull.sum()),
                "n_invalid_or_unknown": int((raw_nonnull & ~labeled).sum()),
                "n_labeled_including_excluded": int(labeled.sum()),
                "n_excluded_labels": int((labeled & ~valid_before_overlap).sum()),
                "n_overlap_removed": int(overlap_removed),
                "overlap_removal_skipped": bool(overlap_skipped),
                "n_evaluable": int(base_mask.sum()),
                "n_direct_consensus": int((base_mask & direct).sum())
                if route_column
                else np.nan,
                "n_non_consensus": int((base_mask & ~direct).sum())
                if route_column
                else np.nan,
            }
        )

        for route, route_mask in routes.items():
            pooled_mask = base_mask & route_mask
            group_masks: list[tuple[str, pd.Series]] = [
                (GROUP_POOLED, pooled_mask)
            ]
            if group_column:
                present_groups = (
                    validation_groups.loc[pooled_mask]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                for group in sorted(present_groups, key=str.lower):
                    group_masks.append(
                        (
                            str(group),
                            pooled_mask & validation_groups.astype(str).eq(str(group)),
                        )
                    )

            for group, evaluation_mask in group_masks:
                if not evaluation_mask.any():
                    continue
                y_true = clean_label_series(
                    frame.loc[evaluation_mask, reference]
                ).astype(str)
                active_labels = sorted(y_true.unique().tolist(), key=str.lower)
                if not active_labels:
                    continue

                for method in methods:
                    summary, classes, confusion = _score_method(
                        y_true,
                        frame.loc[evaluation_mask, method],
                        excluded_labels,
                        active_labels,
                    )
                    common = {
                        "reference_column": reference,
                        "reference": reference_names[reference],
                        "route": route,
                        "group": group,
                        "method_column": method,
                        "method": display_names[method],
                    }
                    summary_rows.append({**common, **summary})
                    per_class_rows.extend(
                        {**common, **row} for row in classes
                    )
                    confusion_rows.extend(
                        {**common, **row} for row in confusion
                    )

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    audit_df = pd.DataFrame(audit_rows)

    confidence_df = analyze_confidence_coverage(
        frame,
        masks=masks,
        reference_names=reference_names,
        methods=methods,
        method_display_names=display_names,
        excluded_labels=excluded_labels,
        route_column=route_column,
        direct_mask=direct,
        confidence_mapping=confidence_mapping or {},
        thresholds=confidence_thresholds,
    )

    human_summary, human_confusion = analyze_annotator_agreement(
        frame,
        annotator_columns=annotator_columns,
        route_column=route_column,
        direct_mask=direct,
    )

    return {
        "summary": summary_df,
        "per_class": per_class_df,
        "confusion": confusion_df,
        "reference_audit": audit_df,
        "reference_overlap": reference_overlap_matrix(frame, references, reference_names),
        "confidence": confidence_df,
        "human_summary": human_summary,
        "human_confusion": human_confusion,
        "metadata": {
            "reference_columns": references,
            "reference_names": reference_names,
            "method_columns": methods,
            "method_display_names": display_names,
            "excluded_labels": list(excluded_labels),
            "route_column": route_column,
            "group_column": group_column,
            "group_mode": group_mode,
            "remove_primary_overlap": bool(remove_primary_overlap),
            "prediction_design": prediction_design,
            "n_input_rows": int(len(frame)),
        },
    }


def analyze_confidence_coverage(
    frame: pd.DataFrame,
    *,
    masks: Mapping[str, tuple[pd.Series, int]],
    reference_names: Mapping[str, str],
    methods: Sequence[str],
    method_display_names: Mapping[str, str],
    excluded_labels: Sequence[str],
    route_column: str | None,
    direct_mask: pd.Series,
    confidence_mapping: Mapping[str, str],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not confidence_mapping:
        return pd.DataFrame()

    for reference, (base_mask, _, _) in masks.items():
        route = "Non-consensus" if route_column else "End-to-end"
        evaluation_mask = base_mask & (~direct_mask if route_column else True)
        if not evaluation_mask.any():
            continue
        y_true = clean_label_series(
            frame.loc[evaluation_mask, reference]
        ).astype(str)
        active_labels = sorted(y_true.unique().tolist(), key=str.lower)

        for method in methods:
            confidence_column = confidence_mapping.get(method)
            if not confidence_column or confidence_column not in frame.columns:
                continue
            predictions, inherently_assigned = normalize_predictions(
                frame.loc[evaluation_mask, method],
                excluded_labels,
            )
            confidence = pd.to_numeric(
                frame.loc[evaluation_mask, confidence_column],
                errors="coerce",
            )
            for threshold in sorted({float(value) for value in thresholds}):
                accepted = (
                    confidence.ge(threshold)
                    & confidence.notna()
                    & inherently_assigned
                )
                if accepted.any():
                    accepted_truth = y_true.loc[accepted]
                    accepted_predictions = predictions.loc[accepted]
                    accepted_accuracy = float(
                        accuracy_score(accepted_truth, accepted_predictions)
                    )
                    accepted_micro = float(
                        f1_score(
                            accepted_truth,
                            accepted_predictions,
                            average="micro",
                            zero_division=0,
                        )
                    )
                    accepted_macro = float(
                        f1_score(
                            accepted_truth,
                            accepted_predictions,
                            labels=active_labels,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    accepted_weighted = float(
                        f1_score(
                            accepted_truth,
                            accepted_predictions,
                            labels=active_labels,
                            average="weighted",
                            zero_division=0,
                        )
                    )
                else:
                    accepted_accuracy = np.nan
                    accepted_micro = np.nan
                    accepted_macro = np.nan
                    accepted_weighted = np.nan
                rows.append(
                    {
                        "reference_column": reference,
                        "reference": reference_names[reference],
                        "route": route,
                        "method_column": method,
                        "method": method_display_names[method],
                        "confidence_column": confidence_column,
                        "threshold": threshold,
                        "n_test": int(len(y_true)),
                        "n_assigned": int(accepted.sum()),
                        "n_rejected": int((~accepted).sum()),
                        "coverage": float(accepted.mean()),
                        "accepted_accuracy": accepted_accuracy,
                        "accepted_micro_f1": accepted_micro,
                        "accepted_macro_f1": accepted_macro,
                        "accepted_weighted_f1": accepted_weighted,
                    }
                )
    return pd.DataFrame(rows)


def analyze_annotator_agreement(
    frame: pd.DataFrame,
    *,
    annotator_columns: tuple[str, str] | None,
    route_column: str | None,
    direct_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not annotator_columns:
        return pd.DataFrame(), pd.DataFrame()
    annotator_a, annotator_b = annotator_columns
    a = clean_label_series(frame[annotator_a])
    b = clean_label_series(frame[annotator_b])
    shared = a.notna() & b.notna()
    routes = {"End-to-end": shared}
    if route_column:
        routes["Direct consensus"] = shared & direct_mask
        routes["Non-consensus"] = shared & ~direct_mask

    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for route, mask in routes.items():
        if not mask.any():
            continue
        y_a = a.loc[mask].astype(str)
        y_b = b.loc[mask].astype(str)
        labels = sorted(set(y_a) | set(y_b), key=str.lower)
        agreement = float(y_a.eq(y_b).mean())
        summary_rows.append(
            {
                "annotator_a_column": annotator_a,
                "annotator_b_column": annotator_b,
                "annotator_a": prettify_column_name(annotator_a),
                "annotator_b": prettify_column_name(annotator_b),
                "route": route,
                "n_shared": int(len(y_a)),
                "n_agree": int(y_a.eq(y_b).sum()),
                "n_disagree": int(y_a.ne(y_b).sum()),
                "observed_agreement": agreement,
                "cohen_kappa": _safe_kappa(y_a, y_b),
                "micro_f1": float(
                    f1_score(y_a, y_b, average="micro", zero_division=0)
                ),
                "macro_f1": float(
                    f1_score(
                        y_a,
                        y_b,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        y_a,
                        y_b,
                        labels=labels,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        )
        counts = confusion_matrix(y_a, y_b, labels=labels)
        totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals != 0,
        )
        for row_index, label_a in enumerate(labels):
            for column_index, label_b in enumerate(labels):
                confusion_rows.append(
                    {
                        "route": route,
                        "annotator_a": prettify_column_name(annotator_a),
                        "annotator_b": prettify_column_name(annotator_b),
                        "label_a": label_a,
                        "label_b": label_b,
                        "count": int(counts[row_index, column_index]),
                        "fraction_of_annotator_a": float(
                            normalized[row_index, column_index]
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(confusion_rows)


def _canonical_method_name(value: Any) -> str | None:
    lowered = str(value).strip().casefold()
    tokens = {
        token
        for token in lowered.replace("-", "_").replace(" ", "_").split("_")
        if token
    }
    compact = "".join(character for character in lowered if character.isalnum())
    if "consensusml" in compact or "mlconsensus" in compact:
        return "Consensus-ML"
    if "mrml" in compact or "mlmr" in compact:
        return "ML-MR"
    if "leiden" in tokens:
        return "Leiden"
    if "som" in tokens:
        return "SOM"
    if "vortex" in tokens:
        return "Vortex"
    if "resolve" in tokens or (
        "threshold" in tokens and "phenotype" in tokens
    ):
        return "RESOLVE"
    return None


def _method_palette(methods: Sequence[str], style: ValidationStyle) -> dict[str, str]:
    fallback_colors = PALETTES.get(style.palette_name, GUI_CLASSIC_COLORS)
    configured = {str(name): str(color) for name, color in style.method_colors.items()}
    configured_casefold = {name.casefold(): color for name, color in configured.items()}
    palette: dict[str, str] = {}
    for index, method in enumerate(methods):
        method_name = str(method)
        canonical = _canonical_method_name(method_name)
        palette[method_name] = configured.get(
            method_name,
            configured_casefold.get(
                method_name.casefold(),
                configured.get(
                    canonical or "",
                    OPTION2_METHOD_COLORS.get(
                        canonical or "",
                        fallback_colors[index % len(fallback_colors)],
                    ),
                ),
            ),
        )
    return palette


def _title(style: ValidationStyle, title: str) -> str:
    prefix = str(style.title_prefix or "").strip()
    return f"{prefix}: {title}" if prefix else title


def _ordered_available(
    values: Iterable[str],
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Restrict and order the methods a figure shows.

    An explicit request is a selection, not merely a sort: the control is
    labelled "Methods shown and their figure order", so anything left out must
    stay out. Appending the unselected methods back on the end - as this used
    to do - made deselection silently do nothing.

    An empty or missing request still means "show everything".
    """
    available = list(dict.fromkeys(str(value) for value in values))
    if not requested:
        return available
    ordered = [str(value) for value in requested if str(value) in available]
    return ordered or available


def _selected_summary(
    bundle: Mapping[str, Any],
    reference: str,
    route: str,
    group: str,
) -> pd.DataFrame:
    summary = bundle["summary"]
    return summary.loc[
        summary["reference"].eq(reference)
        & summary["route"].eq(route)
        & summary["group"].eq(group)
    ].copy()


def plot_per_class_heatmap(
    bundle: Mapping[str, Any],
    *,
    reference: str,
    route: str,
    group: str = GROUP_POOLED,
    metric: str = "f1",
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
    phenotype_order: str = "Reference support",
) -> Figure:
    per_class = bundle["per_class"]
    subset = per_class.loc[
        per_class["reference"].eq(reference)
        & per_class["route"].eq(route)
        & per_class["group"].eq(group)
    ].copy()
    if subset.empty:
        raise ValueError("No per-class values are available for this selection.")

    matrix = subset.pivot_table(
        index="phenotype",
        columns="method",
        values=metric,
        aggfunc="first",
    )
    methods = _ordered_available(matrix.columns, method_order)
    matrix = matrix.reindex(columns=methods)
    if phenotype_order == "Alphabetical":
        phenotypes = sorted(matrix.index.astype(str), key=str.lower)
    else:
        support = (
            subset.groupby("phenotype", observed=True)["support"]
            .max()
            .sort_values(ascending=False)
        )
        phenotypes = support.index.astype(str).tolist()
    matrix = matrix.reindex(phenotypes)
    height = max(style.figure_height, 1.8 + style.row_height * len(matrix))
    width = max(style.figure_width, 2.6 + 1.25 * len(matrix.columns))
    if style.square_cells:
        # Size the canvas to the grid instead of stretching cells to fill it,
        # otherwise square cells leave a wide band of empty figure beside them.
        cell = style.row_height if style.row_height > 0 else 0.45
        height = 1.8 + cell * len(matrix)
        width = 3.4 + cell * len(matrix.columns)

    with plt.rc_context(style.matplotlib_context()):
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(
            matrix,
            annot=style.show_annotations,
            fmt=".2f",
            cmap=style.heatmap_cmap,
            vmin=style.heatmap_vmin,
            vmax=style.heatmap_vmax,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": f"{metric.replace('_', ' ').title()} score"},
            annot_kws={"size": style.annotation_font_size},
            ax=ax,
            square=style.square_cells,
        )
        metric_label = {
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
        }.get(metric, metric.title())
        ax.set_title(_title(style, f"Per-class {metric_label} scores"), pad=12, weight="bold")
        ax.set_xlabel("Phenotyping method", weight="bold")
        ax.set_ylabel("Cell type", weight="bold")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
    return fig


def plot_precision_recall_f1_heatmaps(
    bundle: Mapping[str, Any],
    *,
    reference: str,
    route: str,
    group: str = GROUP_POOLED,
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
) -> Figure:
    per_class = bundle["per_class"]
    subset = per_class.loc[
        per_class["reference"].eq(reference)
        & per_class["route"].eq(route)
        & per_class["group"].eq(group)
    ].copy()
    if subset.empty:
        raise ValueError("No per-class values are available for this selection.")

    supports = (
        subset.groupby("phenotype", observed=True)["support"]
        .max()
        .sort_values(ascending=False)
    )
    phenotypes = supports.index.astype(str).tolist()
    methods = _ordered_available(subset["method"], method_order)
    height = max(style.figure_height, 1.8 + style.row_height * len(phenotypes))
    width = max(style.figure_width * 1.8, 5.2 + 1.1 * len(methods) * 3)

    with plt.rc_context(style.matplotlib_context()):
        fig, axes = plt.subplots(1, 3, figsize=(width, height), sharey=True)
        for axis, metric, title, cmap in zip(
            axes,
            ("precision", "recall", "f1"),
            ("Precision by cell type", "Recall by cell type", "F1 by cell type"),
            ("Reds", "Greens", "Blues"),
            strict=False,
        ):
            matrix = subset.pivot_table(
                index="phenotype",
                columns="method",
                values=metric,
                aggfunc="first",
            ).reindex(index=phenotypes, columns=methods)
            sns.heatmap(
                matrix,
                annot=style.show_annotations,
                fmt=".2f",
                cmap=cmap,
                vmin=style.heatmap_vmin,
                vmax=style.heatmap_vmax,
                linewidths=0.4,
                linecolor="white",
                cbar_kws={"label": "Score", "shrink": 0.82},
                annot_kws={"size": style.annotation_font_size},
                ax=axis,
                square=style.square_cells,
            )
            axis.set_title(title, weight="bold")
            axis.set_xlabel("Method", weight="bold")
            axis.set_ylabel("Cell type" if axis is axes[0] else "", weight="bold")
            axis.tick_params(axis="x", rotation=45)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
            axis.tick_params(axis="y", rotation=0)
        fig.suptitle(_title(style, "Per-class phenotyping performance"), y=1.01, weight="bold")
        fig.tight_layout()
    return fig


def plot_f1_metric_comparison(
    bundle: Mapping[str, Any],
    *,
    reference: str,
    route: str,
    group: str = GROUP_POOLED,
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
) -> Figure:
    subset = _selected_summary(bundle, reference, route, group)
    if subset.empty:
        raise ValueError("No summary metrics are available for this selection.")
    methods = _ordered_available(subset["method"], method_order)
    subset = subset.set_index("method").reindex(methods)
    palette = _method_palette(methods, style)
    metrics = [
        ("micro_f1", "Micro-F1\n(cell-level)"),
        ("macro_f1", "Macro-F1\n(equal phenotype weight)"),
        ("weighted_f1", "Weighted-F1\n(prevalence-weighted)"),
    ]

    with plt.rc_context(style.matplotlib_context()):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(max(style.figure_width * 1.35, 11.0), style.figure_height),
            sharey=True,
        )
        x = np.arange(len(methods), dtype=float)
        for axis, (metric, title) in zip(axes, metrics, strict=False):
            values = subset[metric].to_numpy(dtype=float)
            bars = axis.bar(
                x,
                values,
                color=[palette[method] for method in methods],
                width=0.72,
            )
            axis.set_title(title, weight="bold")
            axis.set_xticks(x, methods, rotation=45, ha="right")
            axis.set_ylim(0, 1.03)
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
            for bar, value in zip(bars, values, strict=False):
                if np.isfinite(value):
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.015,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=style.annotation_font_size,
                    )
            sns.despine(ax=axis)
        axes[0].set_ylabel("F1 score", weight="bold")
        fig.suptitle(
            _title(style, f"Complementary F1 summaries — {reference}, {route}"),
            y=1.02,
            weight="bold",
        )
        fig.tight_layout()
    return fig


def plot_reference_route_comparison(
    bundle: Mapping[str, Any],
    *,
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
) -> Figure:
    summary = bundle["summary"]
    pooled = summary.loc[summary["group"].eq(GROUP_POOLED)].copy()
    if pooled.empty:
        raise ValueError("No pooled validation results are available.")
    references = list(dict.fromkeys(pooled["reference"].astype(str)))
    routes = [route for route in ROUTE_ORDER if route in set(pooled["route"])]
    methods = _ordered_available(pooled["method"], method_order)
    palette = _method_palette(methods, style)
    width = 0.8 / max(len(methods), 1)

    with plt.rc_context(style.matplotlib_context()):
        fig, axes = plt.subplots(
            1,
            len(routes),
            figsize=(max(style.figure_width * 1.45, 4.4 * len(routes)), style.figure_height),
            sharey=True,
            squeeze=False,
        )
        x = np.arange(len(references), dtype=float)
        for axis, route in zip(axes[0], routes, strict=False):
            route_rows = pooled.loc[pooled["route"].eq(route)]
            for index, method in enumerate(methods):
                values = (
                    route_rows.loc[route_rows["method"].eq(method)]
                    .set_index("reference")
                    .reindex(references)["macro_f1"]
                    .to_numpy(dtype=float)
                )
                positions = x + (index - (len(methods) - 1) / 2) * width
                axis.bar(
                    positions,
                    values,
                    width=width * 0.92,
                    label=method,
                    color=palette[method],
                )
            axis.set_xticks(x, references, rotation=0)
            axis.set_title(route, weight="bold")
            axis.set_xlabel("Manual reference", weight="bold")
            axis.set_ylim(0, 1.03)
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
            sns.despine(ax=axis)
        axes[0][0].set_ylabel("Macro-F1", weight="bold")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(5, len(methods)),
            frameon=False,
        )
        fig.suptitle(
            _title(style, "Validation across references and assignment routes"),
            y=1.02,
            weight="bold",
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def plot_grouped_macro_f1(
    bundle: Mapping[str, Any],
    *,
    reference: str,
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
) -> Figure:
    summary = bundle["summary"]
    subset = summary.loc[
        summary["reference"].eq(reference)
        & summary["group"].ne(GROUP_POOLED)
    ].copy()
    if subset.empty:
        raise ValueError("No group-level values are available.")
    groups = sorted(subset["group"].astype(str).unique(), key=str.lower)
    routes = [route for route in ROUTE_ORDER if route in set(subset["route"])]
    methods = _ordered_available(subset["method"], method_order)
    palette = _method_palette(methods, style)
    offsets = np.linspace(-0.24, 0.24, max(len(methods), 1))

    with plt.rc_context(style.matplotlib_context()):
        fig, axes = plt.subplots(
            1,
            len(routes),
            figsize=(max(style.figure_width * 1.45, 4.5 * len(routes)), style.figure_height),
            sharey=True,
            squeeze=False,
        )
        base_x = np.arange(len(groups), dtype=float)
        for axis, route in zip(axes[0], routes, strict=False):
            route_rows = subset.loc[subset["route"].eq(route)]
            for index, method in enumerate(methods):
                rows = (
                    route_rows.loc[route_rows["method"].eq(method)]
                    .set_index("group")
                    .reindex(groups)
                )
                axis.scatter(
                    base_x + offsets[index],
                    rows["macro_f1"].to_numpy(dtype=float),
                    s=48,
                    color=palette[method],
                    label=method,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
            axis.set_xticks(base_x, groups)
            axis.set_title(route, weight="bold")
            axis.set_xlabel("Held-out or evaluation group", weight="bold")
            axis.set_ylim(0, 1.03)
            axis.grid(axis="y", alpha=0.22, linewidth=0.6)
            sns.despine(ax=axis)
        axes[0][0].set_ylabel("Macro-F1", weight="bold")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(5, len(methods)),
            frameon=False,
        )
        fig.suptitle(
            _title(style, f"Group-wise phenotyping performance — {reference}"),
            y=1.02,
            weight="bold",
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def plot_confidence_coverage(
    bundle: Mapping[str, Any],
    *,
    highlight_threshold: float = 0.75,
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
) -> Figure:
    confidence = bundle["confidence"]
    if confidence.empty:
        raise ValueError("No confidence-score mappings were analyzed.")
    methods = _ordered_available(confidence["method"], method_order)
    references = list(dict.fromkeys(confidence["reference"].astype(str)))
    palette = _method_palette(references, style)

    with plt.rc_context(style.matplotlib_context()):
        fig, axes = plt.subplots(
            1,
            len(methods),
            figsize=(max(style.figure_width * 1.3, 5.0 * len(methods)), style.figure_height),
            sharey=True,
            squeeze=False,
        )
        for axis, method in zip(axes[0], methods, strict=False):
            method_rows = confidence.loc[confidence["method"].eq(method)]
            for reference in references:
                rows = method_rows.loc[
                    method_rows["reference"].eq(reference)
                ].sort_values("threshold")
                if rows.empty:
                    continue
                axis.plot(
                    rows["coverage"],
                    rows["accepted_accuracy"],
                    marker="o",
                    linewidth=1.7,
                    color=palette[reference],
                    label=reference,
                )
                highlighted = rows.loc[
                    np.isclose(rows["threshold"], highlight_threshold)
                ]
                if not highlighted.empty:
                    axis.scatter(
                        highlighted["coverage"],
                        highlighted["accepted_accuracy"],
                        s=90,
                        facecolors="white",
                        edgecolors=palette[reference],
                        linewidths=1.8,
                        zorder=4,
                    )
            axis.set_title(method, weight="bold")
            axis.set_xlabel("Assigned fraction", weight="bold")
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(0, 1.03)
            axis.grid(alpha=0.2, linewidth=0.6)
            sns.despine(ax=axis)
        axes[0][0].set_ylabel("Accuracy among assigned cells", weight="bold")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(references)),
            frameon=False,
        )
        fig.suptitle(
            _title(
                style,
                f"Confidence-based abstention (open marker = {highlight_threshold:.2f})",
            ),
            y=1.02,
            weight="bold",
        )
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def plot_human_agreement(
    bundle: Mapping[str, Any],
    *,
    style: ValidationStyle = ValidationStyle(),
) -> Figure:
    summary = bundle["human_summary"]
    if summary.empty:
        raise ValueError("No inter-annotator analysis was requested.")
    routes = [route for route in ROUTE_ORDER if route in set(summary["route"])]
    rows = summary.set_index("route").reindex(routes)
    x = np.arange(len(routes), dtype=float)
    width = 0.36

    with plt.rc_context(style.matplotlib_context()):
        fig, ax = plt.subplots(figsize=(style.figure_width, style.figure_height))
        agreement_bars = ax.bar(
            x - width / 2,
            rows["observed_agreement"],
            width,
            color="#1f77b4",
            label="Observed agreement",
        )
        kappa_bars = ax.bar(
            x + width / 2,
            rows["cohen_kappa"],
            width,
            color="#9e9e9e",
            label="Cohen kappa",
        )
        for bars in (agreement_bars, kappa_bars):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=style.annotation_font_size,
                )
        labels = [
            f"{route}\n(n={int(rows.loc[route, 'n_shared']):,})"
            for route in routes
        ]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Agreement statistic", weight="bold")
        ax.set_title(_title(style, "Independent annotator reproducibility"), weight="bold")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        sns.despine(ax=ax)
        fig.tight_layout()
    return fig


def plot_normalized_confusion(
    bundle: Mapping[str, Any],
    *,
    reference: str,
    route: str,
    group: str,
    method: str,
    style: ValidationStyle = ValidationStyle(),
) -> Figure:
    confusion = bundle["confusion"]
    subset = confusion.loc[
        confusion["reference"].eq(reference)
        & confusion["route"].eq(route)
        & confusion["group"].eq(group)
        & confusion["method"].eq(method)
    ].copy()
    if subset.empty:
        raise ValueError("No confusion matrix is available for this selection.")
    labels = list(
        dict.fromkeys(
            [
                *subset["true_label"].astype(str).tolist(),
                *subset["predicted_label"].astype(str).tolist(),
            ]
        )
    )
    matrix = subset.pivot_table(
        index="true_label",
        columns="predicted_label",
        values="fraction_of_true_class",
        aggfunc="first",
        fill_value=0.0,
    ).reindex(index=labels, columns=labels, fill_value=0.0)
    size = max(style.figure_height, 5.5, 0.48 * len(labels) + 2.5)

    with plt.rc_context(style.matplotlib_context()):
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(
            matrix,
            annot=style.show_annotations,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.4,
            cbar_kws={"label": "Fraction of true class"},
            annot_kws={"size": style.annotation_font_size},
            ax=ax,
        )
        ax.set_title(
            _title(style, f"{method} normalized confusion matrix"),
            weight="bold",
        )
        ax.set_xlabel("Predicted phenotype", weight="bold")
        ax.set_ylabel("Manual-reference phenotype", weight="bold")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
    return fig


def plot_radar_summary(
    bundle: Mapping[str, Any],
    *,
    reference: str,
    route: str,
    group: str = GROUP_POOLED,
    style: ValidationStyle = ValidationStyle(),
    method_order: Sequence[str] | None = None,
) -> Figure:
    subset = _selected_summary(bundle, reference, route, group)
    if subset.empty:
        raise ValueError("No summary metrics are available for this selection.")
    methods = _ordered_available(subset["method"], method_order)
    rows = subset.set_index("method").reindex(methods)
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
        ("cohen_kappa", "Cohen kappa"),
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    palette = _method_palette(methods, style)

    with plt.rc_context(style.matplotlib_context()):
        fig, ax = plt.subplots(
            figsize=(style.figure_width, style.figure_height),
            subplot_kw={"projection": "polar"},
        )
        for method in methods:
            values = [
                float(np.clip(rows.loc[method, metric], 0, 1))
                for metric, _ in metrics
            ]
            closed_values = values + values[:1]
            ax.plot(
                closed_angles,
                closed_values,
                marker="o",
                linewidth=1.8,
                color=palette[method],
                label=method,
            )
            ax.fill(closed_angles, closed_values, color=palette[method], alpha=0.10)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles, [label for _, label in metrics])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.set_title(_title(style, "Multiple performance comparison"), pad=20, weight="bold")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.08),
            frameon=False,
        )
        fig.tight_layout()
    return fig


def figure_to_png_bytes(figure: Figure, dpi: int = 300) -> bytes:
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
    )
    buffer.seek(0)
    return buffer.getvalue()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "validation"


def available_result_values(bundle: Mapping[str, Any]) -> dict[str, list[str]]:
    summary = bundle.get("summary", pd.DataFrame())
    required_columns = {"reference", "route", "group", "method"}
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return {"references": [], "routes": [], "groups": [], "methods": []}
    if not required_columns.issubset(summary.columns):
        return {"references": [], "routes": [], "groups": [], "methods": []}
    return {
        "references": list(dict.fromkeys(summary["reference"].astype(str))),
        "routes": [
            route
            for route in ROUTE_ORDER
            if route in set(summary["route"].astype(str))
        ],
        "groups": list(dict.fromkeys(summary["group"].astype(str))),
        "methods": list(dict.fromkeys(summary["method"].astype(str))),
    }


def interpretation_lines(bundle: Mapping[str, Any]) -> list[str]:
    """Generate restrained, plain-language interpretation prompts."""

    summary = bundle.get("summary", pd.DataFrame())
    metadata = bundle.get("metadata", {})
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return ["No evaluable result rows were produced."]

    values = available_result_values(bundle)
    primary_reference = values["references"][0]
    pooled = summary.loc[
        summary["reference"].eq(primary_reference)
        & summary["route"].eq("End-to-end")
        & summary["group"].eq(GROUP_POOLED)
    ].sort_values("macro_f1", ascending=False)
    lines: list[str] = []
    if not pooled.empty:
        best = pooled.iloc[0]
        lines.append(
            f"Primary result: {best['method']} had the highest macro-F1 "
            f"({best['macro_f1']:.3f}) on {primary_reference}."
        )
        widest_gap = pooled.assign(
            gap=pooled["micro_f1"] - pooled["macro_f1"]
        ).sort_values("gap", ascending=False).iloc[0]
        if widest_gap["gap"] > 0.03:
            lines.append(
                f"{widest_gap['method']} had micro-F1 {widest_gap['micro_f1']:.3f} "
                f"but macro-F1 {widest_gap['macro_f1']:.3f}; the lower macro-F1 "
                "shows that rarer phenotypes performed less uniformly."
            )

    method_names = set(pooled["method"].astype(str))
    if {"MR-ML", "Consensus-ML"}.issubset(method_names):
        pair = pooled.set_index("method")
        difference = float(
            pair.loc["MR-ML", "macro_f1"]
            - pair.loc["Consensus-ML", "macro_f1"]
        )
        lines.append(
            "MR-ML minus Consensus-ML macro-F1 on the primary end-to-end "
            f"comparison was {difference:+.3f}."
        )

    if "Non-consensus" in values["routes"]:
        difficult = summary.loc[
            summary["reference"].eq(primary_reference)
            & summary["route"].eq("Non-consensus")
            & summary["group"].eq(GROUP_POOLED)
        ]
        if not difficult.empty:
            best_difficult = difficult.sort_values(
                "macro_f1",
                ascending=False,
            ).iloc[0]
            lines.append(
                f"On non-consensus cells, {best_difficult['method']} had the "
                f"highest macro-F1 ({best_difficult['macro_f1']:.3f}); this is "
                "the route that most directly tests resolution of disputed cells."
            )

    if metadata.get("prediction_design", "").lower().startswith("saved"):
        lines.append(
            "Group-specific scores of saved final predictions are descriptive. "
            "A generalization claim requires genuinely out-of-fold or held-out predictions."
        )
    else:
        lines.append(
            "The grouped interpretation is valid only if each prediction truly came "
            "from a model that excluded its complete evaluation group during fitting."
        )
    return lines
