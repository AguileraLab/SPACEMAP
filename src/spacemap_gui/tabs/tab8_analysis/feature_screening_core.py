from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import warnings
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index as _lifelines_concordance_index
from lifelines.statistics import logrank_test, proportional_hazard_test
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from .virtual_tissue import _build_region_geometry, _compute_buffer_distance

# GPU acceleration for the motif permutation null. Detection is delegated to the
# same helper the phenotyping methods use, so Tab 7 and Tab 8 agree on whether a
# GPU is usable and there is no second detection rule to keep in sync.
try:
    from spacemap.ph_accel_utils import GPU_AVAILABLE, gpu_device_name, resolve_use_gpu
except Exception:  # spacemap not importable (e.g. bare unit-test context)
    GPU_AVAILABLE = False

    def gpu_device_name():
        return None

    def resolve_use_gpu(use_gpu):
        return False


LogFn = Callable[[str], None] | None
ProgressFn = Callable[[dict[str, Any]], None] | None

SURVIVAL_RISK_COLOR = "#c43d3d"
SURVIVAL_PROTECTIVE_COLOR = "#2d6cdf"
NEUTRAL_COLOR = "#7f8c8d"



def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        f = float(value)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default

import datetime

def _log(log_fn: LogFn, message: str) -> None:
    if log_fn is not None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_fn(f"[{timestamp}] {message}")


def _emit_progress(progress_fn: ProgressFn, **payload: Any) -> None:
    if progress_fn is not None:
        if "message" in payload:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            payload["message"] = f"[{timestamp}] {payload['message']}"
        progress_fn(payload)


def _progress_significance_rank(label: Any) -> int:
    label_text = str(label or "")
    if label_text == "significant_fdr":
        return 3
    if label_text == "discovery_fdr":
        return 2
    if label_text == "nominal_p":
        return 1
    return 0


def _summarize_progress_hits(stats_df: pd.DataFrame, top_n: int | None = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if stats_df.empty:
        return {}, []

    ordered = stats_df.copy()
    ordered["_progress_rank"] = ordered["significance_label"].map(_progress_significance_rank).fillna(0)
    ordered["_progress_fdr_q"] = pd.to_numeric(ordered["fdr_q"], errors="coerce")
    ordered["_progress_p_value"] = pd.to_numeric(ordered["p_value"], errors="coerce")
    ordered = ordered.sort_values(
        ["_progress_rank", "_progress_fdr_q", "_progress_p_value"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if ordered.empty:
        return {}, []

    strongest_row = ordered.iloc[0]
    strongest_summary = {
        "strongest_feature_family": str(strongest_row.get("feature_family") or ""),
        "strongest_feature_label": str(strongest_row.get("feature_label") or ""),
        "strongest_endpoint_label": str(strongest_row.get("endpoint_label") or ""),
        "strongest_stratum_label": str(strongest_row.get("stratum_label") or ""),
        "strongest_significance_label": str(strongest_row.get("significance_label") or ""),
        "strongest_direction": str(strongest_row.get("direction") or ""),
        "strongest_p_value": (
            float(_to_float(strongest_row.get("p_value")))
            if np.isfinite(_to_float(strongest_row.get("p_value")))
            else None
        ),
        "strongest_fdr_q": (
            float(_to_float(strongest_row.get("fdr_q")))
            if np.isfinite(_to_float(strongest_row.get("fdr_q")))
            else None
        ),
    }

    hits = ordered[ordered["significance_label"].isin(["significant_fdr", "discovery_fdr"])].copy()
    if hits.empty and not ordered.empty:
        hits = ordered.head(1).copy()
    if top_n is not None:
        hits = hits.head(max(1, int(top_n))).copy()
    hit_rows: list[dict[str, Any]] = []
    for _, row in hits.iterrows():
        p_value = _to_float(row.get("p_value"))
        fdr_q = _to_float(row.get("fdr_q"))
        hazard_ratio = _to_float(row.get("hazard_ratio"))
        effect_size = _to_float(row.get("effect_size"))
        zone_width = _to_float(row.get("zone_width"))
        hit_rows.append(
            {
                "endpoint_label": str(row.get("endpoint_label") or ""),
                "endpoint_type": str(row.get("endpoint_type") or ""),
                "stratum_label": str(row.get("stratum_label") or ""),
                "feature_family": str(row.get("feature_family") or ""),
                "feature_name": str(row.get("feature_name") or ""),
                "feature_label": str(row.get("feature_label") or ""),
                "significance_label": str(row.get("significance_label") or ""),
                "direction": str(row.get("direction") or ""),
                "zone_width": float(zone_width) if np.isfinite(zone_width) else None,
                "hazard_ratio": float(hazard_ratio) if np.isfinite(hazard_ratio) else None,
                "effect_size": float(effect_size) if np.isfinite(effect_size) else None,
                "p_value": float(p_value) if np.isfinite(p_value) else None,
                "fdr_q": float(fdr_q) if np.isfinite(fdr_q) else None,
            }
        )
    return strongest_summary, hit_rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _slugify(text: Any) -> str:
    chars = []
    for char in str(text or ""):
        if char.isalnum():
            chars.append(char.lower())
        elif char in {"-", "_"}:
            chars.append(char)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


def _first_valid(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _safe_unique(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique().tolist()
    try:
        return sorted(values)
    except Exception:
        return values


def _safe_log2(value: Any) -> float:
    numeric = _safe_float(value, float("nan"))
    if not np.isfinite(numeric) or numeric <= 0:
        return float("nan")
    return float(np.log2(numeric))


def format_zone_label(width: Any, decimals: int = 1) -> str:
    numeric = _to_float(width)
    if not np.isfinite(numeric):
        return "-"
    return f"0 to {float(numeric):.{int(decimals)}f}"


def zone_distance_mask(distances: pd.Series, width: Any) -> pd.Series:
    """Return a boolean mask selecting cells within a signed zone.

    Boundary behaviour
    ------------------
    - **Positive width** (outward from boundary):  ``0 <= distance < width``
      The boundary itself (distance == 0) is **included**.
    - **Negative width** (inward / tumour-interior): ``width < distance <= 0``
      The boundary itself (distance == 0) is **included**.

    In both cases the interval is *half-open*, excluding the far edge
    (the outer shell at ``distance == width`` for positive, the inner
    shell at ``distance == width`` for negative).  This avoids double-
    counting cells that sit exactly on a zone boundary when two
    adjacent zones share a boundary value.
    """
    distance_values = pd.to_numeric(distances, errors="coerce")
    width_value = _to_float(width)
    if not np.isfinite(width_value):
        return pd.Series(False, index=distance_values.index)
    if width_value >= 0:
        return (distance_values >= 0) & (distance_values < width_value)
    return (distance_values <= 0) & (distance_values > width_value)


def format_zone_interval_label(start: Any, end: Any, decimals: int = 1) -> str:
    start_value = _to_float(start)
    end_value = _to_float(end)
    if not np.isfinite(start_value) or not np.isfinite(end_value):
        return "-"
    return f"{float(start_value):.{int(decimals)}f} to {float(end_value):.{int(decimals)}f}"


def zone_distance_interval_mask(distances: pd.Series, start: Any, end: Any) -> pd.Series:
    """Return a half-open interval mask ``start <= distance < end``.

    This is used by the validation runner when the user specifies an explicit
    start and end distance rather than the discovery-style zero-anchored width.
    """
    distance_values = pd.to_numeric(distances, errors="coerce")
    start_value = _to_float(start)
    end_value = _to_float(end)
    if not np.isfinite(start_value) or not np.isfinite(end_value) or start_value == end_value:
        return pd.Series(False, index=distances.index)
    lower = min(float(start_value), float(end_value))
    upper = max(float(start_value), float(end_value))
    return (distance_values >= lower) & (distance_values < upper)


def _survival_direction(hazard_ratio: Any) -> str:
    numeric = _safe_float(hazard_ratio, float("nan"))
    if not np.isfinite(numeric) or numeric <= 0:
        return "neutral"
    if numeric > 1.0:
        return "risk"
    if numeric < 1.0:
        return "protective"
    return "neutral"


def _survival_group_colors(hazard_ratio: Any) -> dict[str, str]:
    direction = _survival_direction(hazard_ratio)
    if direction == "risk":
        return {"High": SURVIVAL_RISK_COLOR, "Low": SURVIVAL_PROTECTIVE_COLOR}
    if direction == "protective":
        return {"High": SURVIVAL_PROTECTIVE_COLOR, "Low": SURVIVAL_RISK_COLOR}
    return {"High": SURVIVAL_RISK_COLOR, "Low": SURVIVAL_PROTECTIVE_COLOR}


def _significance_marker_size(label: Any) -> float:
    text = str(label or "")
    if text == "significant_fdr":
        return 44.0
    if text == "discovery_fdr":
        return 30.0
    return 18.0


def _nominal_significance_status(row: pd.Series) -> str:
    label = str(row.get("significance_label") or "")
    if label == "significant_fdr":
        return "significant_fdr"
    if label == "discovery_fdr":
        return "discovery_fdr"
    p_value = _safe_float(row.get("p_value"), float("nan"))
    if np.isfinite(p_value) and p_value <= 0.05:
        return "nominal_p"
    return "not_significant"


def _width_lookup_key(width_value: Any) -> str:
    numeric = _to_float(width_value)
    if not np.isfinite(numeric):
        return ""
    return f"{float(numeric):.6f}"


def _feature_significance_rank(label: Any) -> int:
    label_text = str(label or "")
    if label_text == "significant_fdr":
        return 2
    if label_text == "discovery_fdr":
        return 1
    return 0


def _significance_meets_threshold(label: Any, threshold: str) -> bool:
    label_text = str(label or "")
    threshold_text = str(threshold or "")
    if threshold_text == "significant_fdr":
        return label_text == "significant_fdr"
    if threshold_text == "discovery_fdr":
        return label_text in {"significant_fdr", "discovery_fdr"}
    return False


def _longest_consecutive_indices(indices: list[int]) -> list[int]:
    ordered = sorted({int(value) for value in indices})
    if not ordered:
        return []
    best = [ordered[0]]
    current = [ordered[0]]
    for value in ordered[1:]:
        if int(value) == int(current[-1]) + 1:
            current.append(int(value))
        else:
            if len(current) > len(best):
                best = current.copy()
            current = [int(value)]
    if len(current) > len(best):
        best = current.copy()
    return best


def _indices_to_range_text(indices: list[int], ordered_widths: list[float]) -> str:
    labels = [format_zone_label(ordered_widths[index]) for index in indices if 0 <= int(index) < len(ordered_widths)]
    return "; ".join(labels)


def _band_label_from_indices(indices: list[int], ordered_widths: list[float]) -> str:
    if not indices:
        return ""
    start_width = ordered_widths[int(indices[0])]
    end_width = ordered_widths[int(indices[-1])]
    if int(indices[0]) == int(indices[-1]):
        return format_zone_label(start_width)
    return f"{format_zone_label(start_width)} .. {format_zone_label(end_width)}"


def _direction_consistency_label(feature_df: pd.DataFrame) -> str:
    endpoint_types = feature_df.get("endpoint_type", pd.Series(dtype=str)).astype(str).str.lower()
    if endpoint_types.empty or not bool((endpoint_types == "survival").any()):
        return "N/A"
    stable_rows = feature_df[feature_df["significance_label"].isin(["significant_fdr", "discovery_fdr"])].copy()
    if stable_rows.empty:
        stable_rows = feature_df.copy()
    directions = [
        str(value)
        for value in stable_rows.get("direction", pd.Series(dtype=str)).astype(str).tolist()
        if str(value) in {"risk", "protective"}
    ]
    if not directions:
        return "Unclear"
    return "Consistent" if len(set(directions)) == 1 else "Mixed"


def _directional_significance_color(direction: Any, status: str) -> str:
    direction_text = str(direction or "neutral")
    palette = {
        ("risk", "significant_fdr"): SURVIVAL_RISK_COLOR,
        ("protective", "significant_fdr"): SURVIVAL_PROTECTIVE_COLOR,
        ("risk", "discovery_fdr"): "#df7a7a",
        ("protective", "discovery_fdr"): "#75a4ea",
        ("risk", "nominal_p"): "#efb1b1",
        ("protective", "nominal_p"): "#bfd4f7",
        ("neutral", "significant_fdr"): "#7c3aed",
        ("neutral", "discovery_fdr"): "#a78bfa",
        ("neutral", "nominal_p"): "#d6bcfa",
    }
    if status == "not_significant":
        return NEUTRAL_COLOR
    return palette.get((direction_text, status), NEUTRAL_COLOR)


def _direction_arrow_text(direction: Any) -> str:
    direction_text = str(direction or "").strip().lower()
    if direction_text == "risk":
        return r"$\uparrow$"
    if direction_text == "protective":
        return r"$\downarrow$"
    return ""


def _underline_texts(ax: Axes, text_objects: list[Any], color_lookup: dict[Any, str], pad_px: float = 1.5) -> None:
    if not text_objects:
        return
    fig = ax.figure
    fig.canvas.draw()
    get_renderer = getattr(fig.canvas, "get_renderer", None)
    if not callable(get_renderer):
        return
    renderer = get_renderer()
    inverse = ax.transData.inverted()
    for text in text_objects:
        bbox = text.get_window_extent(renderer=renderer)
        y_display = bbox.y0 - float(pad_px)
        left = inverse.transform((bbox.x0, y_display))
        right = inverse.transform((bbox.x1, y_display))
        ax.plot(
            [left[0], right[0]],
            [left[1], right[1]],
            color=color_lookup.get(text, text.get_color()),
            linewidth=0.9,
            solid_capstyle="butt",
            clip_on=True,
            zorder=text.get_zorder(),
        )


def _underline_figure_texts(fig: Figure, text_objects: list[Any], color_lookup: dict[Any, str], pad_px: float = 1.5) -> None:
    if not text_objects:
        return
    fig.canvas.draw()
    get_renderer = getattr(fig.canvas, "get_renderer", None)
    if not callable(get_renderer):
        return
    renderer = get_renderer()
    inverse = fig.transFigure.inverted()
    for text in text_objects:
        bbox = text.get_window_extent(renderer=renderer)
        y_display = bbox.y0 - float(pad_px)
        left = inverse.transform((bbox.x0, y_display))
        right = inverse.transform((bbox.x1, y_display))
        fig.add_artist(
            Line2D(
                [left[0], right[0]],
                [left[1], right[1]],
                transform=fig.transFigure,
                color=color_lookup.get(text, text.get_color()),
                linewidth=0.9,
                solid_capstyle="butt",
                zorder=10,
            )
        )


def _get_endpoint_config(config: dict[str, Any], endpoint_label: str) -> dict[str, Any]:
    for endpoint in _prepare_enabled_endpoints(config):
        if endpoint["label"] == str(endpoint_label):
            return dict(endpoint)
    return {}


def _patient_merge_columns(config: dict[str, Any], patient_outcomes_df: pd.DataFrame) -> list[str]:
    return ["patient", "stratum_label"] + [col for col in config["columns"].get("stratify", []) if col in patient_outcomes_df.columns]


def _prepare_enabled_endpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for endpoint in config.get("endpoints", []):
        if not endpoint.get("enabled", False):
            continue
        
        if endpoint.get("type") == "binary" and endpoint.get("multi_mode", False):
            import itertools
            multi_cols = [str(c) for c in endpoint.get("multi_cols", []) if c]
            col_vals = endpoint.get("multi_col_values", {})
            
            for col in multi_cols:
                vals = col_vals.get(col, [])
                if len(vals) < 2:
                    continue
                
                pairs = list(itertools.combinations(vals, 2))
                for valA, valB in pairs:
                    strat_cols = [c for c in multi_cols if c != col]
                    strat_combos = [[]]
                    for r in range(1, min(3, len(strat_cols) + 1)):
                        strat_combos.extend(list(itertools.combinations(strat_cols, r)))
                        
                    for combo in strat_combos:
                        if not combo:
                            label = f"{col}: {valA} vs {valB}"
                            virtual_ep = {
                                "enabled": True,
                                "type": "binary",
                                "label": label,
                                "group_col": col,
                                "positive_value": valA,
                                "negative_value": valB,
                                "filter": {},
                                "config": config,
                                "_group_key": f"{label}__group"
                            }
                            endpoints.append(virtual_ep)
                        else:
                            combo_val_lists = [col_vals.get(sc, []) for sc in combo]
                            if any(not vl for vl in combo_val_lists):
                                continue
                            for val_combo in itertools.product(*combo_val_lists):
                                filter_dict = dict(zip(combo, val_combo))
                                filter_str = ", ".join(f"{sc}={sv}" for sc, sv in filter_dict.items())
                                label = f"{col}: {valA} vs {valB} ({filter_str})"
                                virtual_ep = {
                                    "enabled": True,
                                    "type": "binary",
                                    "label": label,
                                    "group_col": col,
                                    "positive_value": valA,
                                    "negative_value": valB,
                                    "filter": filter_dict,
                                    "config": config,
                                    "_group_key": f"{label}__group"
                                }
                                endpoints.append(virtual_ep)
        else:
            endpoint_copy = dict(endpoint)
            endpoint_copy["config"] = config
            label = str(endpoint.get("label") or endpoint.get("name") or endpoint.get("type") or "endpoint")
            endpoint_copy["label"] = label
            if endpoint.get("type") == "survival":
                endpoint_copy["_time_key"] = f"{label}__time"
                endpoint_copy["_censor_key"] = f"{label}__censor"
            else:
                endpoint_copy["_group_key"] = f"{label}__group"
            endpoints.append(endpoint_copy)
    return endpoints


def _screen_patient_features_from_merged(
    merged: pd.DataFrame,
    config: dict[str, Any],
    endpoints: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()

    prepared_endpoints = endpoints or _prepare_enabled_endpoints(config)
    if not prepared_endpoints:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    group_cols = ["stratum_label", "zone_width", "zone_label", "feature_family", "feature_name", "feature_label"]
    for group_values, feature_df in merged.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_record = dict(zip(group_cols, group_values))
        for endpoint in prepared_endpoints:
            if endpoint.get("type") == "survival":
                stats_row = _screen_survival_feature(feature_df, endpoint)
            else:
                stats_row = _screen_binary_feature(feature_df, endpoint)
            if stats_row is None:
                continue
            results.append(
                {
                    **group_record,
                    "endpoint_label": endpoint["label"],
                    "endpoint_type": endpoint.get("type"),
                    **stats_row,
                }
            )
    stats_df = pd.DataFrame(results)
    if stats_df.empty:
        return stats_df

    fdr_mode = str(config["screening"].get("fdr_mode", "per_family"))
    fdr_alpha = _safe_float(config["screening"].get("fdr_alpha", 0.05), 0.05)
    discovery_alpha = _safe_float(config["screening"].get("fdr_alpha_discovery", 0.10), 0.10)
    q_values = np.full(len(stats_df), np.nan, dtype=float)

    if fdr_mode == "per_family":
        groupby_cols = ["endpoint_label", "stratum_label", "feature_family"]
    else:
        groupby_cols = ["endpoint_label", "stratum_label"]

    for _, idx in stats_df.groupby(groupby_cols, dropna=False).groups.items():
        index_list = list(idx)
        pvals = pd.to_numeric(stats_df.loc[index_list, "p_value"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pvals)
        if mask.sum() == 0:
            continue
        _, corrected, _, _ = multipletests(pvals[mask], alpha=fdr_alpha, method="fdr_bh")
        local = np.full(mask.shape[0], np.nan, dtype=float)
        local[mask] = corrected
        q_values[np.array(index_list, dtype=int)] = local

    stats_df["fdr_q"] = q_values
    stats_df["significance_label"] = "not_significant"
    stats_df.loc[stats_df["fdr_q"] <= discovery_alpha, "significance_label"] = "discovery_fdr"
    stats_df.loc[stats_df["fdr_q"] <= fdr_alpha, "significance_label"] = "significant_fdr"
    stats_df["neg_log10_p"] = -np.log10(pd.to_numeric(stats_df["p_value"], errors="coerce").clip(lower=float(np.finfo(float).tiny)))
    hazard_ratio = pd.to_numeric(stats_df["hazard_ratio"], errors="coerce")
    ci_low = pd.to_numeric(stats_df["ci_low"], errors="coerce")
    ci_high = pd.to_numeric(stats_df["ci_high"], errors="coerce")
    stats_df["direction"] = hazard_ratio.map(_survival_direction)
    stats_df["log2_hazard_ratio"] = hazard_ratio.map(_safe_log2)
    stats_df["log2_ci_low"] = ci_low.map(_safe_log2)
    stats_df["log2_ci_high"] = ci_high.map(_safe_log2)
    return stats_df


def _make_hit_keys(stats_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if stats_df.empty:
        return set()
    return {
        (str(row.endpoint_label), str(row.feature_name), str(row.stratum_label))
        for row in stats_df[["endpoint_label", "feature_name", "stratum_label"]].itertuples(index=False)
    }


def _select_reference_hits(stats_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()
    top_n = max(1, _safe_int(config["screening"].get("reference_top_n"), 10))
    ordered = stats_df.sort_values(["fdr_q", "p_value"], na_position="last").copy()
    hits = ordered[ordered["significance_label"].isin(["significant_fdr", "discovery_fdr"])].copy()
    if hits.empty:
        return ordered.head(top_n).copy()
    seeded = pd.concat([hits, ordered.head(top_n)], ignore_index=True)
    return seeded.drop_duplicates(subset=["endpoint_label", "feature_name", "stratum_label"]).head(top_n).copy()


def _bootstrap_sampling_unit(config: dict[str, Any]) -> str:
    screening = config.get("screening", {}) if isinstance(config, dict) else {}
    unit = str(screening.get("bootstrap_sampling_unit", "patient") or "patient").strip().lower()
    return "roi" if unit == "roi" else "patient"


def _build_bootstrap_patient_units(patient_outcomes: pd.DataFrame) -> pd.DataFrame:
    if patient_outcomes.empty or not {"patient", "stratum_label"}.issubset(patient_outcomes.columns):
        return pd.DataFrame(columns=["patient", "stratum_label", "__bootstrap_key__"])
    units = patient_outcomes[["patient", "stratum_label"]].drop_duplicates().copy()
    units["__bootstrap_key__"] = units["patient"].astype(str) + "\0" + units["stratum_label"].astype(str)
    return units


def _prepare_roi_bootstrap_frame(combined_features: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["patient", "batch", "roi", "stratum_label"]
    if combined_features.empty or any(column not in combined_features.columns for column in required_cols):
        return pd.DataFrame()
    work = combined_features.copy()
    work["__bootstrap_key__"] = (
        work["patient"].astype(str)
        + "\0"
        + work["batch"].astype(str)
        + "\0"
        + work["roi"].astype(str)
        + "\0"
        + work["stratum_label"].astype(str)
    )
    return work


def _build_bootstrap_roi_units(combined_features: pd.DataFrame) -> pd.DataFrame:
    work = _prepare_roi_bootstrap_frame(combined_features)
    if work.empty:
        return pd.DataFrame(columns=["patient", "batch", "roi", "stratum_label", "__bootstrap_key__"])
    return work[["patient", "batch", "roi", "stratum_label", "__bootstrap_key__"]].drop_duplicates().copy()


def _build_bootstrap_roi_units_from_widths(combined_features_by_width: dict[float, pd.DataFrame]) -> pd.DataFrame:
    unit_frames: list[pd.DataFrame] = []
    for combined_features in combined_features_by_width.values():
        units = _build_bootstrap_roi_units(combined_features)
        if not units.empty:
            unit_frames.append(units)
    if not unit_frames:
        return pd.DataFrame(columns=["patient", "batch", "roi", "stratum_label", "__bootstrap_key__"])
    return pd.concat(unit_frames, ignore_index=True).drop_duplicates(subset=["__bootstrap_key__"]).copy()


def _sample_bootstrap_unit_keys(
    unit_df: pd.DataFrame,
    sample_fraction: float,
    rng: np.random.Generator,
    *,
    min_units_per_stratum: int,
) -> list[str]:
    if unit_df.empty or "__bootstrap_key__" not in unit_df.columns or "stratum_label" not in unit_df.columns:
        return []

    min_units = max(1, int(min_units_per_stratum))
    sampled_keys: list[str] = []
    for _, stratum_df in unit_df.groupby("stratum_label", dropna=False, sort=False):
        unit_keys = stratum_df["__bootstrap_key__"].dropna().astype(str).unique().tolist()
        if len(unit_keys) < min_units:
            continue
        sample_size = min(len(unit_keys), max(min_units, int(math.ceil(len(unit_keys) * sample_fraction))))
        sampled = np.atleast_1d(rng.choice(unit_keys, size=sample_size, replace=False)).tolist()
        sampled_keys.extend(str(value) for value in sampled)
    return list(dict.fromkeys(sampled_keys))


def _run_bootstrap_iteration(
    seed: int,
    patient_units: pd.DataFrame,
    patient_fraction: float,
    merged_all: pd.DataFrame,
    config: dict[str, Any],
    endpoints: list[dict[str, Any]],
    reference_keys: set[tuple[str, str, str]],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sampled_keys = _sample_bootstrap_unit_keys(patient_units, patient_fraction, rng, min_units_per_stratum=2)
    if not sampled_keys:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    sampled_merged = merged_all[merged_all["__bootstrap_key__"].isin(sampled_keys)].drop(columns=["__bootstrap_key__"], errors="ignore").copy()
    if sampled_merged.empty:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    bootstrap_stats = _screen_patient_features_from_merged(sampled_merged, config, endpoints=endpoints)
    if bootstrap_stats.empty:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    sampled_hits = _select_reference_hits(bootstrap_stats, config)
    sampled_keys_set = _make_hit_keys(sampled_hits)
    recovery = float(len(reference_keys.intersection(sampled_keys_set))) / float(len(reference_keys)) if reference_keys else 0.0
    significant = float((bootstrap_stats["significance_label"] == "significant_fdr").sum())
    discovery = float(bootstrap_stats["significance_label"].isin(["significant_fdr", "discovery_fdr"]).sum())
    best_p = float(_to_float(bootstrap_stats["p_value"].min()))

    return {"recovery": recovery, "significant": significant, "discovery": discovery, "best_p": best_p}


def _run_bootstrap_iteration_roi(
    seed: int,
    roi_units: pd.DataFrame,
    roi_fraction: float,
    combined_features: pd.DataFrame,
    patient_outcomes: pd.DataFrame,
    merge_cols: list[str],
    config: dict[str, Any],
    endpoints: list[dict[str, Any]],
    reference_keys: set[tuple[str, str, str]],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sampled_keys = _sample_bootstrap_unit_keys(roi_units, roi_fraction, rng, min_units_per_stratum=1)
    if not sampled_keys:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    sampled_combined = combined_features[combined_features["__bootstrap_key__"].isin(sampled_keys)].drop(columns=["__bootstrap_key__"], errors="ignore").copy()
    if sampled_combined.empty:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    sampled_patient_features = aggregate_to_patient(sampled_combined, config)
    if sampled_patient_features.empty:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    sampled_merged = sampled_patient_features.merge(patient_outcomes, on=merge_cols, how="left")
    if sampled_merged.empty:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    bootstrap_stats = _screen_patient_features_from_merged(sampled_merged, config, endpoints=endpoints)
    if bootstrap_stats.empty:
        return {"recovery": 0.0, "significant": 0.0, "discovery": 0.0, "best_p": float("nan")}

    sampled_hits = _select_reference_hits(bootstrap_stats, config)
    sampled_keys_set = _make_hit_keys(sampled_hits)
    recovery = float(len(reference_keys.intersection(sampled_keys_set))) / float(len(reference_keys)) if reference_keys else 0.0
    significant = float((bootstrap_stats["significance_label"] == "significant_fdr").sum())
    discovery = float(bootstrap_stats["significance_label"].isin(["significant_fdr", "discovery_fdr"]).sum())
    best_p = float(_to_float(bootstrap_stats["p_value"].min()))

    return {"recovery": recovery, "significant": significant, "discovery": discovery, "best_p": best_p}


def _run_feature_bootstrap_iteration(
    seed: int,
    patient_units: pd.DataFrame,
    patient_fraction: float,
    merged_by_width: dict[str, pd.DataFrame],
    config: dict[str, Any],
    endpoints: list[dict[str, Any]],
    reference_info: dict[tuple[str, str, str], dict[str, Any]],
    width_key_lookup: dict[str, int],
    neighbor_steps: int,
) -> dict[tuple[str, str, str], dict[str, float]]:
    rng = np.random.default_rng(seed)
    sampled_keys = _sample_bootstrap_unit_keys(patient_units, patient_fraction, rng, min_units_per_stratum=2)

    if not sampled_keys:
        return {}

    bootstrap_frames: list[pd.DataFrame] = []
    for merged in merged_by_width.values():
        sampled_merged = merged[merged["__bootstrap_key__"].isin(sampled_keys)].drop(columns=["__bootstrap_key__"], errors="ignore").copy()
        if sampled_merged.empty:
            continue
        bootstrap_stats = _screen_patient_features_from_merged(sampled_merged, config, endpoints=endpoints)
        if bootstrap_stats.empty:
            continue
        bootstrap_stats = bootstrap_stats.copy()
        bootstrap_stats["__width_index__"] = bootstrap_stats["zone_width"].map(lambda value: width_key_lookup.get(_width_lookup_key(value), -1))
        bootstrap_frames.append(bootstrap_stats)
        
    bootstrap_all = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    bootstrap_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    if not bootstrap_all.empty:
        for key, group_df in bootstrap_all.groupby(["endpoint_label", "stratum_label", "feature_name"], dropna=False, sort=False):
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            endpoint_value, stratum_value, feature_name_value = key
            bootstrap_groups[(str(endpoint_value), str(stratum_value), str(feature_name_value))] = group_df.copy()
            
    iteration_metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, reference in reference_info.items():
        feature_df = bootstrap_groups.get(key, pd.DataFrame()).copy()
        if feature_df.empty:
            iteration_metrics[key] = {"band": 0.0, "exact": 0.0, "best_q": float("nan")}
            continue
            
        threshold_mask = feature_df["significance_label"].map(lambda value: _significance_meets_threshold(value, reference["required_label"]))
        recovered_df = feature_df.loc[threshold_mask].copy()
        if recovered_df.empty:
            iteration_metrics[key] = {"band": 0.0, "exact": 0.0, "best_q": float("nan")}
            continue
            
        recovered_indices = sorted({int(value) for value in pd.to_numeric(recovered_df["__width_index__"], errors="coerce").dropna().astype(int).tolist() if int(value) >= 0})
        exact_recovery = any(index in reference["reference_indices"] for index in recovered_indices)
        band_recovery = any(abs(index - ref_index) <= neighbor_steps for index in recovered_indices for ref_index in reference["reference_indices"])
        
        nearby_df = recovered_df[
            recovered_df["__width_index__"].map(
                lambda value: any(abs(int(value) - ref_index) <= neighbor_steps for ref_index in reference["reference_indices"]) if pd.notna(value) else False
            )
        ].copy()
        nearby_best_q = pd.to_numeric(nearby_df.get("fdr_q", pd.Series(dtype=float)), errors="coerce").min() if not nearby_df.empty else float("nan")
        
        iteration_metrics[key] = {
            "band": 1.0 if band_recovery else 0.0,
            "exact": 1.0 if exact_recovery else 0.0,
            "best_q": float(nearby_best_q) if np.isfinite(nearby_best_q) else float("nan")
        }
        
    return iteration_metrics


def _run_feature_bootstrap_iteration_roi(
    seed: int,
    roi_units: pd.DataFrame,
    roi_fraction: float,
    combined_by_width: dict[str, pd.DataFrame],
    patient_outcomes: pd.DataFrame,
    merge_cols: list[str],
    config: dict[str, Any],
    endpoints: list[dict[str, Any]],
    reference_info: dict[tuple[str, str, str], dict[str, Any]],
    width_key_lookup: dict[str, int],
    neighbor_steps: int,
) -> dict[tuple[str, str, str], dict[str, float]]:
    rng = np.random.default_rng(seed)
    sampled_keys = _sample_bootstrap_unit_keys(roi_units, roi_fraction, rng, min_units_per_stratum=1)
    if not sampled_keys:
        return {}

    bootstrap_frames: list[pd.DataFrame] = []
    for combined_features in combined_by_width.values():
        sampled_combined = combined_features[combined_features["__bootstrap_key__"].isin(sampled_keys)].drop(columns=["__bootstrap_key__"], errors="ignore").copy()
        if sampled_combined.empty:
            continue
        sampled_patient_features = aggregate_to_patient(sampled_combined, config)
        if sampled_patient_features.empty:
            continue
        sampled_merged = sampled_patient_features.merge(patient_outcomes, on=merge_cols, how="left")
        if sampled_merged.empty:
            continue
        bootstrap_stats = _screen_patient_features_from_merged(sampled_merged, config, endpoints=endpoints)
        if bootstrap_stats.empty:
            continue
        bootstrap_stats = bootstrap_stats.copy()
        bootstrap_stats["__width_index__"] = bootstrap_stats["zone_width"].map(lambda value: width_key_lookup.get(_width_lookup_key(value), -1))
        bootstrap_frames.append(bootstrap_stats)

    bootstrap_all = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame()
    bootstrap_groups: dict[tuple[str, str, str], pd.DataFrame] = {}
    if not bootstrap_all.empty:
        for key, group_df in bootstrap_all.groupby(["endpoint_label", "stratum_label", "feature_name"], dropna=False, sort=False):
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            endpoint_value, stratum_value, feature_name_value = key
            bootstrap_groups[(str(endpoint_value), str(stratum_value), str(feature_name_value))] = group_df.copy()

    iteration_metrics: dict[tuple[str, str, str], dict[str, float]] = {}
    for key, reference in reference_info.items():
        feature_df = bootstrap_groups.get(key, pd.DataFrame()).copy()
        if feature_df.empty:
            iteration_metrics[key] = {"band": 0.0, "exact": 0.0, "best_q": float("nan")}
            continue

        threshold_mask = feature_df["significance_label"].map(lambda value: _significance_meets_threshold(value, reference["required_label"]))
        recovered_df = feature_df.loc[threshold_mask].copy()
        if recovered_df.empty:
            iteration_metrics[key] = {"band": 0.0, "exact": 0.0, "best_q": float("nan")}
            continue

        recovered_indices = sorted({int(value) for value in pd.to_numeric(recovered_df["__width_index__"], errors="coerce").dropna().astype(int).tolist() if int(value) >= 0})
        exact_recovery = any(index in reference["reference_indices"] for index in recovered_indices)
        band_recovery = any(abs(index - ref_index) <= neighbor_steps for index in recovered_indices for ref_index in reference["reference_indices"])

        nearby_df = recovered_df[
            recovered_df["__width_index__"].map(
                lambda value: any(abs(int(value) - ref_index) <= neighbor_steps for ref_index in reference["reference_indices"]) if pd.notna(value) else False
            )
        ].copy()
        nearby_best_q = pd.to_numeric(nearby_df.get("fdr_q", pd.Series(dtype=float)), errors="coerce").min() if not nearby_df.empty else float("nan")

        iteration_metrics[key] = {
            "band": 1.0 if band_recovery else 0.0,
            "exact": 1.0 if exact_recovery else 0.0,
            "best_q": float(nearby_best_q) if np.isfinite(nearby_best_q) else float("nan")
        }

    return iteration_metrics


def bootstrap_candidate_stability(
    patient_features: pd.DataFrame,
    patient_outcomes: pd.DataFrame,
    base_stats: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
    log_fn: LogFn = None,
    target_width: float = None,
    combined_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if patient_features.empty or patient_outcomes.empty or base_stats.empty:
        return {
            "bootstrap_reference_recovery": float("nan"),
            "bootstrap_n_significant_mean": float("nan"),
            "bootstrap_n_discovery_mean": float("nan"),
            "bootstrap_best_p_median": float("nan"),
            "bootstrap_valid_iterations": 0,
        }

    reference_hits = _select_reference_hits(base_stats, config)
    reference_keys = _make_hit_keys(reference_hits)
    if not reference_keys:
        return {
            "bootstrap_reference_recovery": float("nan"),
            "bootstrap_n_significant_mean": float("nan"),
            "bootstrap_n_discovery_mean": float("nan"),
            "bootstrap_best_p_median": float("nan"),
            "bootstrap_valid_iterations": 0,
        }

    # SMART FEATURE FAMILY FILTERING: Only bootstrap the feature families that actually had an FDR hit.
    sig_hits = base_stats[base_stats["significance_label"].isin(["significant_fdr", "discovery_fdr"])]
    sig_families: list[str] = []
    if not sig_hits.empty:
        sig_families = sig_hits["feature_family"].unique().tolist()
        patient_features = patient_features[patient_features["feature_family"].isin(sig_families)].copy()

    iterations = max(1, _safe_int(config["screening"].get("bootstrap_iterations"), 30))
    patient_fraction = min(1.0, max(0.1, _safe_float(config["screening"].get("bootstrap_patient_fraction"), 0.80)))
    sampling_unit = _bootstrap_sampling_unit(config)
    patient_units = _build_bootstrap_patient_units(patient_outcomes)
    endpoints = _prepare_enabled_endpoints(config)
    merge_cols = _patient_merge_columns(config, patient_outcomes)
    merged_all = pd.DataFrame()
    roi_units = pd.DataFrame()
    combined_with_keys = pd.DataFrame()
    if sampling_unit == "roi":
        combined_source = combined_features if isinstance(combined_features, pd.DataFrame) else pd.DataFrame()
        if sig_families:
            combined_source = combined_source[combined_source["feature_family"].isin(sig_families)].copy() if not combined_source.empty and "feature_family" in combined_source.columns else combined_source
        combined_with_keys = _prepare_roi_bootstrap_frame(combined_source)
        roi_units = _build_bootstrap_roi_units(combined_with_keys)
        if combined_with_keys.empty or roi_units.empty:
            return {
                "bootstrap_reference_recovery": float("nan"),
                "bootstrap_n_significant_mean": float("nan"),
                "bootstrap_n_discovery_mean": float("nan"),
                "bootstrap_best_p_median": float("nan"),
                "bootstrap_valid_iterations": 0,
            }
    else:
        merged_all = patient_features.merge(patient_outcomes, on=merge_cols, how="left")
        if merged_all.empty:
            return {
                "bootstrap_reference_recovery": float("nan"),
                "bootstrap_n_significant_mean": float("nan"),
                "bootstrap_n_discovery_mean": float("nan"),
                "bootstrap_best_p_median": float("nan"),
                "bootstrap_valid_iterations": 0,
            }
        merged_all = merged_all.copy()
        merged_all["__bootstrap_key__"] = merged_all["patient"].astype(str) + "\0" + merged_all["stratum_label"].astype(str)

    import concurrent.futures
    import os

    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(iterations)]
    max_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if sampling_unit == "roi":
            futures = [
                executor.submit(
                    _run_bootstrap_iteration_roi,
                    seed,
                    roi_units,
                    patient_fraction,
                    combined_with_keys,
                    patient_outcomes,
                    merge_cols,
                    config,
                    endpoints,
                    reference_keys,
                )
                for seed in seeds
            ]
        else:
            futures = [
                executor.submit(
                    _run_bootstrap_iteration,
                    seed,
                    patient_units,
                    patient_fraction,
                    merged_all,
                    config,
                    endpoints,
                    reference_keys,
                )
                for seed in seeds
            ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            if log_fn and target_width is not None and completed % max(1, iterations // 5) == 0:
                _log(log_fn, f"  Candidate stability width {target_width:.2f}: {completed}/{iterations} iterations completed")

    recoveries = [r["recovery"] for r in results]
    significant_counts = [r["significant"] for r in results]
    discovery_counts = [r["discovery"] for r in results]
    best_ps = [r["best_p"] for r in results]

    valid_iterations = len(recoveries)
    if valid_iterations == 0:
        return {
            "bootstrap_reference_recovery": float("nan"),
            "bootstrap_n_significant_mean": float("nan"),
            "bootstrap_n_discovery_mean": float("nan"),
            "bootstrap_best_p_median": float("nan"),
            "bootstrap_valid_iterations": 0,
        }

    return {
        "bootstrap_reference_recovery": float(np.nanmean(recoveries)),
        "bootstrap_n_significant_mean": float(np.nanmean(significant_counts)),
        "bootstrap_n_discovery_mean": float(np.nanmean(discovery_counts)),
        "bootstrap_best_p_median": float(np.nanmedian(best_ps)) if np.isfinite(np.nanmedian(best_ps)) else float("nan"),
        "bootstrap_valid_iterations": int(valid_iterations),
    }


def _prepare_feature_endpoint_payload(
    hit_row: pd.Series,
    patient_features_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    feature_filter = (
        (patient_features_df["feature_name"].astype(str) == str(hit_row["feature_name"]))
        & (patient_features_df["zone_width"].astype(float) == float(hit_row["zone_width"]))
        & (patient_features_df["stratum_label"].astype(str) == str(hit_row["stratum_label"]))
    )
    feature_df = patient_features_df.loc[feature_filter].copy()
    endpoint_label = str(hit_row["endpoint_label"])
    endpoint_type = str(hit_row["endpoint_type"])
    merge_cols = _patient_merge_columns(config, patient_outcomes_df)

    payload: dict[str, Any] = {
        "endpoint_label": endpoint_label,
        "endpoint_type": endpoint_type,
        "feature_df": feature_df,
        "plot_df": pd.DataFrame(),
        "export_df": pd.DataFrame(),
        "audit_df": pd.DataFrame(),
        "group_colors": {},
        "attrition_summary": {},
    }
    if feature_df.empty:
        return payload

    merged = feature_df.merge(patient_outcomes_df, on=merge_cols, how="left")
    payload["merged"] = merged
    feature_patient_ids = set(feature_df["patient"].dropna().astype(str).unique().tolist()) if "patient" in feature_df.columns else set()

    if endpoint_type == "survival":
        endpoint_config = _get_endpoint_config(config, endpoint_label)
        time_key = f"{endpoint_label}__time"
        censor_key = f"{endpoint_label}__censor"
        use_censor_as_censored = bool(endpoint_config.get("censor_flag_means_censored", True))
        merged[time_key] = pd.to_numeric(merged[time_key], errors="coerce")
        merged[censor_key] = pd.to_numeric(merged[censor_key], errors="coerce")

        universe_df = patient_outcomes_df.copy()
        if "stratum_label" in universe_df.columns:
            universe_df = universe_df[universe_df["stratum_label"].astype(str) == str(hit_row.get("stratum_label") or "")].copy()
        for column in merge_cols:
            if column in {"patient", "stratum_label"}:
                continue
            if column in universe_df.columns:
                universe_df = universe_df[universe_df[column].astype(str) == str(hit_row.get(column) or "")].copy()
        feature_cols = [column for column in [*merge_cols, "feature_family", "feature_name", "feature_label", "zone_width", "zone_label", "value", "n_regions", "mean_region_value"] if column in feature_df.columns]
        audit_df = universe_df.merge(feature_df.loc[:, feature_cols], on=merge_cols, how="left") if not universe_df.empty else merged.copy()
        audit_df["feature_value"] = pd.to_numeric(audit_df["value"], errors="coerce")
        audit_df["endpoint_time"] = pd.to_numeric(audit_df[time_key], errors="coerce")
        audit_df["endpoint_censor"] = pd.to_numeric(audit_df[censor_key], errors="coerce")
        missing_feature_mask = ~np.isfinite(audit_df["feature_value"])
        missing_time_mask = ~np.isfinite(audit_df["endpoint_time"])
        missing_censor_mask = ~np.isfinite(audit_df["endpoint_censor"])
        complete_mask = ~(missing_feature_mask | missing_time_mask | missing_censor_mask)

        merged_complete = audit_df.loc[complete_mask].copy()
        complete_patient_ids = set(merged_complete["patient"].dropna().astype(str).unique().tolist()) if "patient" in merged_complete.columns else set()
        missing_endpoint_patients = max(0, len(feature_patient_ids - complete_patient_ids))
        merged_complete["group"], cutpoint = _split_high_low(merged_complete["feature_value"], config, cutpoint_override=hit_row.get("cutpoint"))
        merged = merged_complete[merged_complete["group"].isin(["High", "Low"])].copy()
        split_patient_ids = set(merged["patient"].dropna().astype(str).unique().tolist()) if "patient" in merged.columns else set()
        split_excluded_patients = max(0, len(complete_patient_ids - split_patient_ids))
        counts = merged["group"].value_counts() if not merged.empty else pd.Series(dtype=int)

        audit_df["screen_group"] = pd.Series(index=audit_df.index, dtype="object")
        if not merged_complete.empty:
            audit_df.loc[merged_complete.index, "screen_group"] = merged_complete["group"].astype(object)
        included_mask = complete_mask & audit_df["screen_group"].isin(["High", "Low"])
        split_method = str(config.get("screening", {}).get("split_method") or "median")
        audit_df["used_in_test"] = included_mask
        audit_df["km_status"] = np.where(included_mask, "Included in KM", "Excluded from KM")
        audit_df["exclusion_reason"] = "Included in KM"
        audit_df.loc[missing_feature_mask, "exclusion_reason"] = "Missing feature value"
        audit_df.loc[~missing_feature_mask & missing_time_mask & missing_censor_mask, "exclusion_reason"] = "Missing endpoint time and censor"
        audit_df.loc[~missing_feature_mask & missing_time_mask & ~missing_censor_mask, "exclusion_reason"] = "Missing endpoint time"
        audit_df.loc[~missing_feature_mask & ~missing_time_mask & missing_censor_mask, "exclusion_reason"] = "Missing endpoint censor"
        if split_method in {"quartiles", "tertiles"}:
            audit_df.loc[complete_mask & ~included_mask, "exclusion_reason"] = f"Excluded by {split_method} middle group"
        else:
            audit_df.loc[complete_mask & ~included_mask, "exclusion_reason"] = "Excluded after High/Low split"
        audit_df["endpoint_label"] = endpoint_label
        audit_df["feature_name"] = str(hit_row["feature_name"])
        audit_df["feature_label"] = str(hit_row["feature_label"])
        audit_df["zone_width"] = float(hit_row["zone_width"])
        audit_df["cutpoint"] = float(cutpoint) if cutpoint is not None else float("nan")
        audit_df = audit_df.sort_values(["used_in_test", "patient"], ascending=[True, True], na_position="last").reset_index(drop=True)
        payload["attrition_summary"] = {
            "n_feature_patients": int(len(feature_patient_ids)),
            "n_missing_endpoint_patients": int(missing_endpoint_patients),
            "n_split_excluded_patients": int(split_excluded_patients),
            "n_final_patients": int(len(split_patient_ids)),
            "group_high_n": int(counts.get("High", 0)) if not counts.empty else 0,
            "group_low_n": int(counts.get("Low", 0)) if not counts.empty else 0,
            "endpoint_type": "survival",
        }
        payload["audit_df"] = audit_df
        payload["cutpoint"] = cutpoint
        payload["time_key"] = time_key
        payload["censor_key"] = censor_key
        payload["use_censor_as_censored"] = use_censor_as_censored
        payload["group_colors"] = _survival_group_colors(hit_row.get("hazard_ratio"))
        if merged.empty or merged["group"].nunique() < 2:
            return payload

        merged["event"] = 1 - merged[censor_key] if use_censor_as_censored else merged[censor_key]
        group_colors = payload.get("group_colors", _survival_group_colors(hit_row.get("hazard_ratio")))
        time_source = str(endpoint_config.get("time_col") or "endpoint_time")
        censor_source = str(endpoint_config.get("censor_col") or "endpoint_censor")
        export_cols = merge_cols + ["value", "group", "event", time_key, censor_key]
        export_df = merged[export_cols].copy().rename(
            columns={
                "value": "feature_value",
                "group": "screen_group",
                "event": "event_flag",
                time_key: "endpoint_time",
                censor_key: "endpoint_censor",
            }
        )
        export_df[time_source] = export_df["endpoint_time"]
        export_df[censor_source] = export_df["endpoint_censor"]
        export_df["endpoint_label"] = endpoint_label
        export_df["feature_name"] = str(hit_row["feature_name"])
        export_df["feature_label"] = str(hit_row["feature_label"])
        export_df["zone_width"] = float(hit_row["zone_width"])
        export_df["cutpoint"] = float(cutpoint) if cutpoint is not None else float("nan")
        export_df["hazard_ratio"] = _safe_float(hit_row.get("hazard_ratio"), float("nan"))
        export_df["p_value"] = _safe_float(hit_row.get("p_value"), float("nan"))
        export_df["fdr_q"] = _safe_float(hit_row.get("fdr_q"), float("nan"))
        export_df["risk_direction"] = _survival_direction(hit_row.get("hazard_ratio"))
        export_df["screen_group_color"] = export_df["screen_group"].map(group_colors)
        export_df = export_df.sort_values(["screen_group", "endpoint_time", "patient"], ascending=[True, True, True]).reset_index(drop=True)
        payload.update(
            {
                "plot_df": merged,
                "export_df": export_df,
                "group_colors": group_colors,
            }
        )
        return payload

    group_key = f"{endpoint_label}__group"
    audit_df = merged.copy()
    audit_df["feature_value"] = pd.to_numeric(audit_df["value"], errors="coerce")
    audit_df["endpoint_group"] = audit_df[group_key].astype(object)
    missing_feature_mask = ~np.isfinite(audit_df["feature_value"])
    missing_group_mask = audit_df["endpoint_group"].isna()
    complete_mask = ~(missing_feature_mask | missing_group_mask)
    merged_complete = audit_df.loc[complete_mask].copy()
    complete_patient_ids = set(merged_complete["patient"].dropna().astype(str).unique().tolist()) if "patient" in merged_complete.columns else set()
    missing_endpoint_patients = max(0, len(feature_patient_ids - complete_patient_ids))
    audit_df["used_in_test"] = complete_mask
    audit_df["km_status"] = np.where(complete_mask, "Included in test", "Excluded from test")
    audit_df["exclusion_reason"] = "Included in test"
    audit_df.loc[missing_feature_mask, "exclusion_reason"] = "Missing feature value"
    audit_df.loc[~missing_feature_mask & missing_group_mask, "exclusion_reason"] = "Missing endpoint group"
    audit_df["endpoint_label"] = endpoint_label
    audit_df["feature_name"] = str(hit_row["feature_name"])
    audit_df["feature_label"] = str(hit_row["feature_label"])
    audit_df["zone_width"] = float(hit_row["zone_width"])
    audit_df = audit_df.sort_values(["used_in_test", "patient"], ascending=[True, True], na_position="last").reset_index(drop=True)
    payload["attrition_summary"] = {
        "n_feature_patients": int(len(feature_patient_ids)),
        "n_missing_endpoint_patients": int(missing_endpoint_patients),
        "n_split_excluded_patients": 0,
        "n_final_patients": int(len(complete_patient_ids)),
        "endpoint_type": "binary",
    }
    payload["audit_df"] = audit_df
    merged = merged_complete
    if merged.empty:
        return payload
    export_cols = merge_cols + ["value", group_key]
    export_df = merged[export_cols].copy().rename(columns={"value": "feature_value", group_key: "endpoint_group"})
    export_df["endpoint_label"] = endpoint_label
    export_df["feature_name"] = str(hit_row["feature_name"])
    export_df["feature_label"] = str(hit_row["feature_label"])
    export_df["zone_width"] = float(hit_row["zone_width"])
    export_df["p_value"] = _safe_float(hit_row.get("p_value"), float("nan"))
    export_df["fdr_q"] = _safe_float(hit_row.get("fdr_q"), float("nan"))
    payload.update({"plot_df": merged, "export_df": export_df.reset_index(drop=True)})
    return payload


def build_feature_endpoint_export_df(
    hit_row: pd.Series,
    patient_features_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    return _prepare_feature_endpoint_payload(hit_row, patient_features_df, patient_outcomes_df, config).get("export_df", pd.DataFrame()).copy()


def build_feature_endpoint_attrition_summary(
    hit_row: pd.Series,
    patient_features_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    summary = _prepare_feature_endpoint_payload(hit_row, patient_features_df, patient_outcomes_df, config).get("attrition_summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def _build_missing_feature_diagnostics_map(
    hit_row: pd.Series,
    analysis_df: pd.DataFrame,
    config: dict[str, Any],
    patient_ids: list[str],
    zone_start: float | None = None,
    zone_end: float | None = None,
) -> dict[str, dict[str, Any]]:
    if analysis_df.empty or not patient_ids:
        return {}

    feature_family = str(hit_row.get("feature_family") or "")
    feature_name = str(hit_row.get("feature_name") or "")
    if feature_family != "triad":
        return {}

    feature_parts = [str(value) for value in feature_name.split("::") if str(value)]
    if len(feature_parts) != 4:
        return {}
    triad_parts = sorted(feature_parts[1:4])
    target_triad = (triad_parts[0], triad_parts[1], triad_parts[2])
    required_component_counts: dict[str, int] = {}
    for component in target_triad:
        required_component_counts[component] = required_component_counts.get(component, 0) + 1

    screening = config["screening"]
    columns = config["columns"]
    x_col = str(columns["x"])
    y_col = str(columns["y"])
    phenotype_col = str(columns["phenotype"])
    distance_col = str(columns["distance"])
    patient_col = str(columns["patient"])
    batch_col = str(columns["batch"])
    roi_col = str(columns["roi"])

    use_micron_per_pixel = bool(screening.get("use_micron_per_pixel", True))
    micron_per_pixel = max(_safe_float(screening.get("micron_per_pixel"), 0.5), 1e-6) if use_micron_per_pixel else 1.0
    min_cells_per_zone = max(1, _safe_int(screening.get("min_cells_per_zone"), 20))
    dyad_min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
    dyad_max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel
    triad_min_px = _safe_float(screening.get("triad_min_dist_um"), 0.0) / micron_per_pixel
    triad_max_px = _safe_float(screening.get("triad_max_dist_um"), 30.0) / micron_per_pixel
    use_adaptive_radius = bool(screening.get("use_adaptive_interaction_radius", True))
    adaptive_direct_neighbours_only = bool(screening.get("adaptive_radius_direct_neighbours_only", True))
    adaptive_min_pct = _safe_float(screening.get("adaptive_radius_min_pct", 0.1), 0.1)
    adaptive_max_pct = _safe_float(screening.get("adaptive_radius_max_pct", 99.0), 99.0)
    adaptive_max_sample = max(50, _safe_int(screening.get("adaptive_radius_max_sample", 500), 500))
    permutation_n_iter = max(0, _safe_int(screening.get("dyad_triad_permutation_n_iter", 1000), 1000))
    min_obs_motif_count = max(0, _safe_int(screening.get("min_obs_motif_count", 3), 3))
    perm_seed = _safe_int(screening.get("bootstrap_random_seed"), 20260407)
    interval_mode = zone_start is not None and zone_end is not None and np.isfinite(_safe_float(zone_start, float("nan"))) and np.isfinite(_safe_float(zone_end, float("nan")))
    zone_start_value = _safe_float(zone_start, 0.0)
    zone_end_value = _safe_float(zone_end, _safe_float(hit_row.get("zone_width"), 0.0))
    zone_width_value = _safe_float(hit_row.get("zone_width"), 0.0)

    tracked_patients = {str(patient_id) for patient_id in patient_ids if str(patient_id)}
    state_map = {
        patient_id: {
            "has_components": False,
            "has_triangle": False,
            "component_roi_count": 0,
            "triangle_roi_count": 0,
            "max_obs_count": 0,
        }
        for patient_id in tracked_patients
    }
    valid_zone_index = 0

    for _, group_df in _iter_regions(analysis_df, batch_col, roi_col):
        patient_id = str(_first_valid(group_df[patient_col]) or "")
        if interval_mode:
            zone_df = _zone_interval_subset(group_df, distance_col, zone_start_value, zone_end_value)
        else:
            zone_df = _zone_subset(group_df, distance_col, zone_width_value)
        if len(zone_df) < min_cells_per_zone:
            continue

        roi_seed = perm_seed + valid_zone_index
        valid_zone_index += 1
        if patient_id not in state_map:
            continue

        phenotype_values = zone_df[phenotype_col].astype(str)
        phenotype_counts = phenotype_values.value_counts()
        if any(int(phenotype_counts.get(component, 0)) < needed for component, needed in required_component_counts.items()):
            continue
        state_map[patient_id]["has_components"] = True
        state_map[patient_id]["component_roi_count"] = int(state_map[patient_id]["component_roi_count"]) + 1

        coords = zone_df[[x_col, y_col]].to_numpy(dtype=float)
        if use_adaptive_radius:
            adaptive_rng = np.random.default_rng(roi_seed)
            adaptive_min_px, adaptive_max_px = _compute_roi_adaptive_radius(
                coords,
                adaptive_min_pct,
                adaptive_max_pct,
                adaptive_max_sample,
                adaptive_rng,
                direct_neighbours_only=adaptive_direct_neighbours_only,
            )
            if np.isfinite(adaptive_max_px) and adaptive_max_px > 0:
                min_px, max_px = adaptive_min_px, adaptive_max_px
            else:
                min_px = min(dyad_min_px, triad_min_px)
                max_px = max(dyad_max_px, triad_max_px)
        else:
            min_px = min(dyad_min_px, triad_min_px)
            max_px = max(dyad_max_px, triad_max_px)

        adjacency = _build_adjacency(coords, min_px, max_px)
        triad_counts = _collect_unordered_triads(phenotype_values.to_numpy(dtype=str), adjacency)
        observed_count = int(triad_counts.get(target_triad, 0))
        if observed_count <= 0:
            continue

        state_map[patient_id]["has_triangle"] = True
        state_map[patient_id]["triangle_roi_count"] = int(state_map[patient_id]["triangle_roi_count"]) + 1
        state_map[patient_id]["max_obs_count"] = max(int(state_map[patient_id]["max_obs_count"]), observed_count)

    diagnostics_map: dict[str, dict[str, Any]] = {}
    for patient_id, state in state_map.items():
        feature_compute_detail = ""
        feature_compute_stage = ""
        if not bool(state["has_components"]):
            feature_compute_detail = "Missing one required phenotype component"
            feature_compute_stage = "Feature components"
        elif not bool(state["has_triangle"]):
            feature_compute_detail = "Required phenotypes present, but no qualifying triangle formed"
            feature_compute_stage = "Triangle geometry"
        elif permutation_n_iter > 0 and int(state["max_obs_count"]) < min_obs_motif_count:
            feature_compute_detail = "Too few observed motifs for permutation score"
            feature_compute_stage = "Permutation support"
        diagnostics_map[patient_id] = {
            "feature_compute_detail": feature_compute_detail,
            "feature_compute_stage": feature_compute_stage,
            "triad_required_components": " + ".join(target_triad),
            "triad_components_present_in_valid_roi": bool(state["has_components"]),
            "triad_qualifying_triangle_found": bool(state["has_triangle"]),
            "triad_component_roi_count": int(state["component_roi_count"]),
            "triad_triangle_roi_count": int(state["triangle_roi_count"]),
            "triad_max_observed_motif_count": int(state["max_obs_count"]),
            "triad_min_obs_required": int(min_obs_motif_count) if permutation_n_iter > 0 else 0,
        }
    return diagnostics_map


def build_feature_endpoint_audit_df(
    hit_row: pd.Series,
    analysis_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
    zone_start: float | None = None,
    zone_end: float | None = None,
) -> pd.DataFrame:
    if patient_outcomes_df.empty:
        return pd.DataFrame()

    endpoint_label = str(hit_row.get("endpoint_label") or "")
    endpoint_type = str(hit_row.get("endpoint_type") or "")
    stratum_label = str(hit_row.get("stratum_label") or "")
    zone_width = _safe_float(hit_row.get("zone_width"), float("nan"))
    merge_cols = _patient_merge_columns(config, patient_outcomes_df)

    universe_df = patient_outcomes_df.copy()
    if "stratum_label" in universe_df.columns:
        universe_df = universe_df[universe_df["stratum_label"].astype(str) == stratum_label].copy()
    for column in merge_cols:
        if column in {"patient", "stratum_label"}:
            continue
        if column in universe_df.columns:
            universe_df = universe_df[universe_df[column].astype(str) == str(hit_row.get(column) or "")].copy()
    if universe_df.empty:
        return pd.DataFrame()

    feature_filter = (
        (patient_features_df["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
        & (patient_features_df["zone_width"].astype(float) == float(zone_width))
        & (patient_features_df["stratum_label"].astype(str) == stratum_label)
    )
    feature_df = patient_features_df.loc[feature_filter].copy()
    for column in merge_cols:
        if column in feature_df.columns and column not in {"patient", "stratum_label"}:
            feature_df = feature_df[feature_df[column].astype(str) == str(hit_row.get(column) or "")].copy()

    feature_cols = [column for column in [*merge_cols, "feature_family", "feature_name", "feature_label", "zone_width", "zone_label", "value", "n_regions", "mean_region_value"] if column in feature_df.columns]
    audit_df = universe_df.merge(feature_df.loc[:, feature_cols], on=merge_cols, how="left")

    audit_df["feature_family"] = str(hit_row.get("feature_family") or audit_df.get("feature_family", ""))
    audit_df["feature_name"] = str(hit_row.get("feature_name") or audit_df.get("feature_name", ""))
    audit_df["feature_label"] = str(hit_row.get("feature_label") or audit_df.get("feature_label", ""))
    audit_df["zone_width"] = float(zone_width) if np.isfinite(zone_width) else float("nan")
    audit_df["zone_label"] = str(hit_row.get("zone_label") or (format_zone_label(zone_width) if np.isfinite(zone_width) else ""))
    audit_df["feature_value"] = pd.to_numeric(audit_df.get("value", pd.Series(np.nan, index=audit_df.index)), errors="coerce")

    patient_ids = audit_df["patient"].dropna().astype(str).tolist() if "patient" in audit_df.columns else []
    has_zone_cells_map = {patient_id: False for patient_id in patient_ids}
    has_valid_zone_roi_map = {patient_id: False for patient_id in patient_ids}
    n_zone_rois_map = {patient_id: 0 for patient_id in patient_ids}
    n_valid_zone_rois_map = {patient_id: 0 for patient_id in patient_ids}
    if isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
        scoped_analysis = analysis_df.copy()
        if "__stratum_label__" in scoped_analysis.columns:
            scoped_analysis = scoped_analysis[scoped_analysis["__stratum_label__"].astype(str) == stratum_label].copy()
        patient_col = str(config["columns"]["patient"])
        batch_col = str(config["columns"]["batch"])
        roi_col = str(config["columns"]["roi"])
        distance_col = str(config["columns"]["distance"])
        min_cells_per_zone = max(1, _safe_int(config["screening"].get("min_cells_per_zone"), 20))
        interval_mode = zone_start is not None and zone_end is not None and np.isfinite(_safe_float(zone_start, float("nan"))) and np.isfinite(_safe_float(zone_end, float("nan")))
        zone_start_value = _safe_float(zone_start, 0.0)
        zone_end_value = _safe_float(zone_end, zone_width)
        for _, group_df in _iter_regions(scoped_analysis, batch_col, roi_col):
            patient_id = str(_first_valid(group_df[patient_col]) or "")
            if patient_id not in has_zone_cells_map:
                continue
            if interval_mode:
                zone_df = _zone_interval_subset(group_df, distance_col, zone_start_value, zone_end_value)
            else:
                zone_df = _zone_subset(group_df, distance_col, float(zone_width))
            if zone_df.empty:
                continue
            has_zone_cells_map[patient_id] = True
            n_zone_rois_map[patient_id] = int(n_zone_rois_map[patient_id]) + 1
            if len(zone_df) >= min_cells_per_zone:
                has_valid_zone_roi_map[patient_id] = True
                n_valid_zone_rois_map[patient_id] = int(n_valid_zone_rois_map[patient_id]) + 1
        audit_df["has_zone_cells"] = audit_df["patient"].astype(str).map(has_zone_cells_map).fillna(False)
        audit_df["has_valid_zone_roi"] = audit_df["patient"].astype(str).map(has_valid_zone_roi_map).fillna(False)
        audit_df["n_zone_rois"] = pd.to_numeric(audit_df["patient"].astype(str).map(n_zone_rois_map), errors="coerce").fillna(0).astype(int)
        audit_df["n_valid_zone_rois"] = pd.to_numeric(audit_df["patient"].astype(str).map(n_valid_zone_rois_map), errors="coerce").fillna(0).astype(int)
    else:
        audit_df["has_zone_cells"] = False
        audit_df["has_valid_zone_roi"] = False
        audit_df["n_zone_rois"] = 0
        audit_df["n_valid_zone_rois"] = 0

    missing_feature_patients = audit_df.loc[
        (~np.isfinite(audit_df["feature_value"])) & audit_df["has_valid_zone_roi"],
        "patient",
    ].dropna().astype(str).tolist()
    missing_feature_diagnostics = _build_missing_feature_diagnostics_map(
        hit_row,
        analysis_df,
        config,
        missing_feature_patients,
        zone_start=zone_start,
        zone_end=zone_end,
    )
    audit_df["zone_status"] = "Has valid zone ROI"
    audit_df.loc[~audit_df["has_zone_cells"], "zone_status"] = "No cells in selected range"
    audit_df.loc[audit_df["has_zone_cells"] & ~audit_df["has_valid_zone_roi"], "zone_status"] = "Cells in range but below min_cells_per_zone"
    audit_df["feature_status"] = pd.Series(index=audit_df.index, dtype="object")
    audit_df["feature_compute_stage"] = pd.Series(index=audit_df.index, dtype="object")
    audit_df["feature_compute_detail"] = pd.Series(index=audit_df.index, dtype="object")
    if str(hit_row.get("feature_family") or "") == "triad":
        patient_key = audit_df["patient"].astype(str)
        audit_df["triad_required_components"] = patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_required_components"))
        audit_df["triad_components_present_in_valid_roi"] = patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_components_present_in_valid_roi", False)).fillna(False).astype(bool)
        audit_df["triad_qualifying_triangle_found"] = patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_qualifying_triangle_found", False)).fillna(False).astype(bool)
        audit_df["triad_component_roi_count"] = pd.to_numeric(patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_component_roi_count", 0)), errors="coerce").fillna(0).astype(int)
        audit_df["triad_triangle_roi_count"] = pd.to_numeric(patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_triangle_roi_count", 0)), errors="coerce").fillna(0).astype(int)
        audit_df["triad_max_observed_motif_count"] = pd.to_numeric(patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_max_observed_motif_count", 0)), errors="coerce").fillna(0).astype(int)
        audit_df["triad_min_obs_required"] = pd.to_numeric(patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("triad_min_obs_required", 0)), errors="coerce").fillna(0).astype(int)
        audit_df["feature_compute_stage"] = patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("feature_compute_stage"))
        audit_df["feature_compute_detail"] = patient_key.map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("feature_compute_detail"))

    if endpoint_type == "survival":
        time_key = f"{endpoint_label}__time"
        censor_key = f"{endpoint_label}__censor"
        audit_df["endpoint_time"] = pd.to_numeric(audit_df.get(time_key, pd.Series(np.nan, index=audit_df.index)), errors="coerce")
        audit_df["endpoint_censor"] = pd.to_numeric(audit_df.get(censor_key, pd.Series(np.nan, index=audit_df.index)), errors="coerce")

        missing_feature_mask = ~np.isfinite(audit_df["feature_value"])
        missing_time_mask = ~np.isfinite(audit_df["endpoint_time"])
        missing_censor_mask = ~np.isfinite(audit_df["endpoint_censor"])
        complete_mask = ~(missing_feature_mask | missing_time_mask | missing_censor_mask)

        audit_df["screen_group"] = pd.Series(index=audit_df.index, dtype="object")
        complete_values = audit_df.loc[complete_mask, "feature_value"]
        if not complete_values.empty:
            group_labels, cutpoint = _split_high_low(complete_values, config, cutpoint_override=hit_row.get("cutpoint"))
            audit_df.loc[complete_mask, "screen_group"] = group_labels.astype(object)
        else:
            cutpoint = hit_row.get("cutpoint")

        included_mask = complete_mask & audit_df["screen_group"].isin(["High", "Low"])
        split_method = str(config.get("screening", {}).get("split_method") or "median")
        audit_df["endpoint_status"] = "Endpoint complete"
        audit_df.loc[missing_time_mask & missing_censor_mask, "endpoint_status"] = "Missing endpoint time and censor"
        audit_df.loc[missing_time_mask & ~missing_censor_mask, "endpoint_status"] = "Missing endpoint time"
        audit_df.loc[~missing_time_mask & missing_censor_mask, "endpoint_status"] = "Missing endpoint censor"
        audit_df["split_status"] = "Not eligible for split"
        audit_df.loc[complete_mask & (audit_df["screen_group"].astype(str) == "High"), "split_status"] = "Assigned to High"
        audit_df.loc[complete_mask & (audit_df["screen_group"].astype(str) == "Low"), "split_status"] = "Assigned to Low"

        audit_df["used_in_test"] = included_mask
        audit_df["km_status"] = np.where(included_mask, "Included in KM", "Excluded from KM")
        audit_df["exclusion_reason"] = "Included in KM"
        audit_df["feature_status"] = "Feature computed"
        audit_df.loc[missing_feature_mask & ~audit_df["has_valid_zone_roi"], "feature_status"] = "Feature not attempted"
        audit_df.loc[missing_feature_mask & ~audit_df["has_zone_cells"], "exclusion_reason"] = "No cells in selected range"
        audit_df.loc[missing_feature_mask & audit_df["has_zone_cells"] & ~audit_df["has_valid_zone_roi"], "exclusion_reason"] = "Cells in range but below min_cells_per_zone"
        audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"], "exclusion_reason"] = "Selected feature not computable at this range"
        if missing_feature_diagnostics:
            mapped_reason = audit_df["patient"].astype(str).map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("feature_compute_detail"))
            audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"] & mapped_reason.notna(), "exclusion_reason"] = mapped_reason[mapped_reason.notna()]
        audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"], "feature_status"] = audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"], "exclusion_reason"]
        audit_df.loc[~missing_feature_mask & missing_time_mask & missing_censor_mask, "exclusion_reason"] = "Missing endpoint time and censor"
        audit_df.loc[~missing_feature_mask & missing_time_mask & ~missing_censor_mask, "exclusion_reason"] = "Missing endpoint time"
        audit_df.loc[~missing_feature_mask & ~missing_time_mask & missing_censor_mask, "exclusion_reason"] = "Missing endpoint censor"
        if split_method in {"quartiles", "tertiles"}:
            audit_df.loc[complete_mask & ~included_mask, "exclusion_reason"] = f"Excluded by {split_method} middle group"
            audit_df.loc[complete_mask & ~included_mask, "split_status"] = f"Excluded by {split_method} middle group"
        else:
            audit_df.loc[complete_mask & ~included_mask, "exclusion_reason"] = "Excluded after High/Low split"
            audit_df.loc[complete_mask & ~included_mask, "split_status"] = "Excluded after High/Low split"
        audit_df["audit_stage"] = "Included in KM"
        audit_df.loc[~included_mask & ~audit_df["has_zone_cells"], "audit_stage"] = "Zone selection"
        audit_df.loc[~included_mask & audit_df["has_zone_cells"] & ~audit_df["has_valid_zone_roi"], "audit_stage"] = "Minimum cells per ROI"
        audit_df.loc[~included_mask & missing_feature_mask & audit_df["has_valid_zone_roi"], "audit_stage"] = "Feature computation"
        audit_df.loc[~included_mask & ~missing_feature_mask & (missing_time_mask | missing_censor_mask), "audit_stage"] = "Endpoint availability"
        audit_df.loc[~included_mask & complete_mask, "audit_stage"] = "Split assignment"
        audit_df["audit_detail"] = audit_df["exclusion_reason"]
        audit_df["endpoint_label"] = endpoint_label
        audit_df["cutpoint"] = float(cutpoint) if cutpoint is not None and np.isfinite(_safe_float(cutpoint, float("nan"))) else float("nan")
    else:
        group_key = f"{endpoint_label}__group"
        audit_df["endpoint_group"] = audit_df.get(group_key, pd.Series(index=audit_df.index, dtype="object")).astype(object)
        missing_feature_mask = ~np.isfinite(audit_df["feature_value"])
        missing_group_mask = audit_df["endpoint_group"].isna()
        included_mask = ~(missing_feature_mask | missing_group_mask)
        audit_df["used_in_test"] = included_mask
        audit_df["km_status"] = np.where(included_mask, "Included in test", "Excluded from test")
        audit_df["exclusion_reason"] = "Included in test"
        audit_df["feature_status"] = "Feature computed"
        audit_df.loc[missing_feature_mask & ~audit_df["has_valid_zone_roi"], "feature_status"] = "Feature not attempted"
        audit_df["endpoint_status"] = "Endpoint group available"
        audit_df.loc[missing_group_mask, "endpoint_status"] = "Missing endpoint group"
        audit_df["split_status"] = "Not applicable for binary endpoint"
        audit_df.loc[missing_feature_mask & ~audit_df["has_zone_cells"], "exclusion_reason"] = "No cells in selected range"
        audit_df.loc[missing_feature_mask & audit_df["has_zone_cells"] & ~audit_df["has_valid_zone_roi"], "exclusion_reason"] = "Cells in range but below min_cells_per_zone"
        audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"], "exclusion_reason"] = "Selected feature not computable at this range"
        if missing_feature_diagnostics:
            mapped_reason = audit_df["patient"].astype(str).map(lambda patient_id: (missing_feature_diagnostics.get(patient_id) or {}).get("feature_compute_detail"))
            audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"] & mapped_reason.notna(), "exclusion_reason"] = mapped_reason[mapped_reason.notna()]
        audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"], "feature_status"] = audit_df.loc[missing_feature_mask & audit_df["has_valid_zone_roi"], "exclusion_reason"]
        audit_df.loc[~missing_feature_mask & missing_group_mask, "exclusion_reason"] = "Missing endpoint group"
        audit_df["audit_stage"] = "Included in test"
        audit_df.loc[~included_mask & ~audit_df["has_zone_cells"], "audit_stage"] = "Zone selection"
        audit_df.loc[~included_mask & audit_df["has_zone_cells"] & ~audit_df["has_valid_zone_roi"], "audit_stage"] = "Minimum cells per ROI"
        audit_df.loc[~included_mask & missing_feature_mask & audit_df["has_valid_zone_roi"], "audit_stage"] = "Feature computation"
        audit_df.loc[~included_mask & ~missing_feature_mask & missing_group_mask, "audit_stage"] = "Endpoint availability"
        audit_df["audit_detail"] = audit_df["exclusion_reason"]
        audit_df["endpoint_label"] = endpoint_label
        audit_df["cutpoint"] = float("nan")

    audit_df = audit_df.sort_values(["used_in_test", "patient"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return audit_df


def _continuous_cox_feature_summary(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_col: str,
    penalizer: float = 0.01,
    log_fn: LogFn = None,
) -> dict[str, Any]:
    summary = {
        "n_patients": 0,
        "n_events": 0,
        "hazard_ratio_per_sd": float("nan"),
        "ci_low_per_sd": float("nan"),
        "ci_high_per_sd": float("nan"),
        "p_value": float("nan"),
        "concordance_index": float("nan"),
        "ph_test_p": float("nan"),
        "feature_mean": float("nan"),
        "feature_std": float("nan"),
    }

    model_df = df[[time_col, event_col, feature_col]].dropna().copy()
    if model_df.empty:
        return summary

    model_df[time_col] = pd.to_numeric(model_df[time_col], errors="coerce")
    model_df[event_col] = pd.to_numeric(model_df[event_col], errors="coerce")
    model_df[feature_col] = pd.to_numeric(model_df[feature_col], errors="coerce")
    model_df = model_df[
        np.isfinite(model_df[time_col])
        & np.isfinite(model_df[event_col])
        & np.isfinite(model_df[feature_col])
    ].copy()
    if model_df.empty:
        return summary

    summary["n_patients"] = int(len(model_df))
    summary["n_events"] = int(pd.to_numeric(model_df[event_col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).sum())

    feature_mean = float(model_df[feature_col].mean())
    feature_std = float(model_df[feature_col].std(ddof=0))
    summary["feature_mean"] = feature_mean if np.isfinite(feature_mean) else float("nan")
    summary["feature_std"] = feature_std if np.isfinite(feature_std) else float("nan")
    if not np.isfinite(feature_std) or feature_std < 1e-9 or int(model_df[feature_col].nunique()) < 2:
        return summary

    model_df = model_df.copy()
    model_df["feature_z"] = (model_df[feature_col] - feature_mean) / feature_std
    if int(model_df["feature_z"].nunique()) < 2:
        return summary

    try:
        fitter = CoxPHFitter(penalizer=float(penalizer))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter.fit(model_df[[time_col, event_col, "feature_z"]], duration_col=time_col, event_col=event_col)
            for warning in caught:
                if "convergence" in str(warning.message).lower() or "newton" in str(warning.message).lower():
                    _log(log_fn, f"Continuous Cox convergence warning: {warning.message}")
        fitter_summary = fitter.summary.loc["feature_z"]
        coef = float(np.clip(fitter_summary["coef"], -20.0, 20.0))
        ci_lo = float(np.clip(fitter_summary["coef lower 95%"], -20.0, 20.0))
        ci_hi = float(np.clip(fitter_summary["coef upper 95%"], -20.0, 20.0))
        summary.update(
            {
                "hazard_ratio_per_sd": float(math.exp(coef)),
                "ci_low_per_sd": float(math.exp(ci_lo)),
                "ci_high_per_sd": float(math.exp(ci_hi)),
                "p_value": float(fitter_summary.get("p", float("nan"))),
                "concordance_index": float(fitter.concordance_index_) if hasattr(fitter, "concordance_index_") else float("nan"),
            }
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ph_test = proportional_hazard_test(fitter, model_df[[time_col, event_col, "feature_z"]], time_fit="km")
                summary["ph_test_p"] = float(ph_test.p_value.iloc[0])
        except Exception:
            summary["ph_test_p"] = float("nan")
    except Exception as exc:
        _log(log_fn, f"Continuous Cox fitting failed: {exc}")
    return summary


def build_triad_secondary_analysis(
    hit_row: pd.Series,
    analysis_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
    zone_start: float | None = None,
    zone_end: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "patient_df": pd.DataFrame(),
        "proportion_summary": {},
        "density_summary": {},
        "status": "unsupported",
        "message": "Secondary analysis is currently available only for survival triad features.",
    }

    if patient_outcomes_df.empty or analysis_df.empty:
        result["message"] = "No validation data are available for secondary analysis."
        return result
    if str(hit_row.get("feature_family") or "") != "triad" or str(hit_row.get("endpoint_type") or "") != "survival":
        return result

    feature_name = str(hit_row.get("feature_name") or "")
    feature_parts = [str(value) for value in feature_name.split("::") if str(value)]
    if len(feature_parts) != 4:
        result["message"] = "The selected triad could not be parsed for secondary analysis."
        return result
    triad_parts = sorted(feature_parts[1:4])
    target_triad = (triad_parts[0], triad_parts[1], triad_parts[2])

    endpoint_label = str(hit_row.get("endpoint_label") or "")
    stratum_label = str(hit_row.get("stratum_label") or "")
    merge_cols = _patient_merge_columns(config, patient_outcomes_df)
    universe_df = patient_outcomes_df.copy()
    if "stratum_label" in universe_df.columns:
        universe_df = universe_df[universe_df["stratum_label"].astype(str) == stratum_label].copy()
    for column in merge_cols:
        if column in {"patient", "stratum_label"}:
            continue
        if column in universe_df.columns:
            universe_df = universe_df[universe_df[column].astype(str) == str(hit_row.get(column) or "")].copy()
    if universe_df.empty:
        result["message"] = "No patients matched the selected stratum for secondary analysis."
        return result

    endpoint_config = _get_endpoint_config(config, endpoint_label)
    time_key = f"{endpoint_label}__time"
    censor_key = f"{endpoint_label}__censor"
    if time_key not in universe_df.columns or censor_key not in universe_df.columns:
        result["message"] = "The selected survival endpoint is missing required columns for secondary analysis."
        return result

    screening = config["screening"]
    columns = config["columns"]
    x_col = str(columns["x"])
    y_col = str(columns["y"])
    phenotype_col = str(columns["phenotype"])
    distance_col = str(columns["distance"])
    patient_col = str(columns["patient"])
    batch_col = str(columns["batch"])
    roi_col = str(columns["roi"])
    use_micron_per_pixel = bool(screening.get("use_micron_per_pixel", True))
    micron_per_pixel = max(_safe_float(screening.get("micron_per_pixel"), 0.5), 1e-6) if use_micron_per_pixel else 1.0
    min_cells_per_zone = max(1, _safe_int(screening.get("min_cells_per_zone"), 20))
    triad_min_px = _safe_float(screening.get("triad_min_dist_um"), 0.0) / micron_per_pixel
    triad_max_px = _safe_float(screening.get("triad_max_dist_um"), 30.0) / micron_per_pixel
    dyad_min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
    dyad_max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel
    use_adaptive_radius = bool(screening.get("use_adaptive_interaction_radius", True))
    adaptive_direct_neighbours_only = bool(screening.get("adaptive_radius_direct_neighbours_only", True))
    adaptive_min_pct = _safe_float(screening.get("adaptive_radius_min_pct", 0.1), 0.1)
    adaptive_max_pct = _safe_float(screening.get("adaptive_radius_max_pct", 99.0), 99.0)
    adaptive_max_sample = max(50, _safe_int(screening.get("adaptive_radius_max_sample", 500), 500))
    permutation_n_iter = max(0, _safe_int(screening.get("dyad_triad_permutation_n_iter", 1000), 1000))
    min_obs_motif_count = max(0, _safe_int(screening.get("min_obs_motif_count", 3), 3))
    allow_self_interaction_features = bool(screening.get("allow_self_interaction_features", False))
    area_use_median = bool(screening.get("area_use_median", False))
    area_percentile = _safe_float(screening.get("area_percentile", 95.0), 95.0)
    min_hole_size_px2 = _safe_float(screening.get("min_hole_size_px2", 0.0), 0.0)
    perm_seed = _safe_int(screening.get("bootstrap_random_seed"), 20260407)
    use_censor_as_censored = bool(endpoint_config.get("censor_flag_means_censored", True))

    tracked_patients = universe_df["patient"].dropna().astype(str).tolist() if "patient" in universe_df.columns else []
    has_zone_cells_map = {patient_id: False for patient_id in tracked_patients}
    has_valid_zone_roi_map = {patient_id: False for patient_id in tracked_patients}
    zone_roi_count_map = {patient_id: 0 for patient_id in tracked_patients}
    valid_zone_roi_count_map = {patient_id: 0 for patient_id in tracked_patients}
    total_target_triads_map = {patient_id: 0 for patient_id in tracked_patients}
    total_observed_triads_map = {patient_id: 0 for patient_id in tracked_patients}
    total_zone_area_map = {patient_id: 0.0 for patient_id in tracked_patients}
    max_target_count_map = {patient_id: 0 for patient_id in tracked_patients}

    scoped_analysis = analysis_df.copy()
    if "__stratum_label__" in scoped_analysis.columns:
        scoped_analysis = scoped_analysis[scoped_analysis["__stratum_label__"].astype(str) == stratum_label].copy()

    interval_mode = zone_start is not None and zone_end is not None and np.isfinite(_safe_float(zone_start, float("nan"))) and np.isfinite(_safe_float(zone_end, float("nan")))
    zone_width = _safe_float(hit_row.get("zone_width"), float("nan"))
    zone_start_value = _safe_float(zone_start, 0.0)
    zone_end_value = _safe_float(zone_end, zone_width)
    valid_zone_index = 0
    for _, group_df in _iter_regions(scoped_analysis, batch_col, roi_col):
        patient_id = str(_first_valid(group_df[patient_col]) or "")
        if patient_id not in has_zone_cells_map:
            continue
        if interval_mode:
            zone_df = _zone_interval_subset(group_df, distance_col, zone_start_value, zone_end_value)
        else:
            zone_df = _zone_subset(group_df, distance_col, float(zone_width))
        if zone_df.empty:
            continue
        has_zone_cells_map[patient_id] = True
        zone_roi_count_map[patient_id] = int(zone_roi_count_map[patient_id]) + 1
        if len(zone_df) < min_cells_per_zone:
            continue

        roi_seed = perm_seed + valid_zone_index
        valid_zone_index += 1
        has_valid_zone_roi_map[patient_id] = True
        valid_zone_roi_count_map[patient_id] = int(valid_zone_roi_count_map[patient_id]) + 1

        coords = zone_df[[x_col, y_col]].to_numpy(dtype=float)
        area_mm2, _ = _compute_virtual_area_mm2(
            coords=coords,
            micron_per_pixel=micron_per_pixel,
            hole_size_px2=min_hole_size_px2,
            use_median=area_use_median,
            percentile=area_percentile,
        )
        if np.isfinite(area_mm2) and area_mm2 > 0:
            total_zone_area_map[patient_id] = float(total_zone_area_map[patient_id]) + float(area_mm2)

        if use_adaptive_radius:
            adaptive_rng = np.random.default_rng(roi_seed)
            adaptive_min_px, adaptive_max_px = _compute_roi_adaptive_radius(
                coords,
                adaptive_min_pct,
                adaptive_max_pct,
                adaptive_max_sample,
                adaptive_rng,
                direct_neighbours_only=adaptive_direct_neighbours_only,
            )
            if np.isfinite(adaptive_max_px) and adaptive_max_px > 0:
                min_px, max_px = adaptive_min_px, adaptive_max_px
            else:
                min_px = min(dyad_min_px, triad_min_px)
                max_px = max(dyad_max_px, triad_max_px)
        else:
            min_px = min(dyad_min_px, triad_min_px)
            max_px = max(dyad_max_px, triad_max_px)

        adjacency = _build_adjacency(coords, min_px, max_px)
        triad_counts = _collect_unordered_triads(zone_df[phenotype_col].astype(str).to_numpy(dtype=str), adjacency)
        if not allow_self_interaction_features:
            triad_counts = {key: value for key, value in triad_counts.items() if len(set(key)) == 3}

        target_count = int(triad_counts.get(target_triad, 0))
        total_triads = int(sum(int(value) for value in triad_counts.values()))
        total_target_triads_map[patient_id] = int(total_target_triads_map[patient_id]) + target_count
        total_observed_triads_map[patient_id] = int(total_observed_triads_map[patient_id]) + total_triads
        max_target_count_map[patient_id] = max(int(max_target_count_map[patient_id]), target_count)

    patient_df = universe_df.copy()
    patient_key = patient_df["patient"].astype(str)
    patient_df["endpoint_time"] = pd.to_numeric(patient_df.get(time_key, pd.Series(np.nan, index=patient_df.index)), errors="coerce")
    patient_df["endpoint_censor"] = pd.to_numeric(patient_df.get(censor_key, pd.Series(np.nan, index=patient_df.index)), errors="coerce")
    patient_df["has_zone_cells"] = patient_key.map(has_zone_cells_map).fillna(False).astype(bool)
    patient_df["has_valid_zone_roi"] = patient_key.map(has_valid_zone_roi_map).fillna(False).astype(bool)
    patient_df["n_zone_rois"] = pd.to_numeric(patient_key.map(zone_roi_count_map), errors="coerce").fillna(0).astype(int)
    patient_df["n_valid_zone_rois"] = pd.to_numeric(patient_key.map(valid_zone_roi_count_map), errors="coerce").fillna(0).astype(int)
    patient_df["total_target_triads"] = pd.to_numeric(patient_key.map(total_target_triads_map), errors="coerce").fillna(0).astype(int)
    patient_df["total_observed_triads"] = pd.to_numeric(patient_key.map(total_observed_triads_map), errors="coerce").fillna(0).astype(int)
    patient_df["total_zone_area_mm2"] = pd.to_numeric(patient_key.map(total_zone_area_map), errors="coerce")
    patient_df["triad_present"] = patient_df["total_target_triads"] > 0
    patient_df["supports_primary_permutation_score"] = (
        pd.to_numeric(patient_key.map(max_target_count_map), errors="coerce").fillna(0).astype(int) >= int(min_obs_motif_count)
        if permutation_n_iter > 0
        else patient_df["triad_present"]
    )
    patient_df["raw_triad_proportion"] = np.where(
        patient_df["has_valid_zone_roi"],
        np.where(
            patient_df["total_observed_triads"] > 0,
            patient_df["total_target_triads"].astype(float) / patient_df["total_observed_triads"].astype(float),
            0.0,
        ),
        np.nan,
    )
    patient_df["raw_triad_density"] = np.where(
        patient_df["has_valid_zone_roi"] & np.isfinite(pd.to_numeric(patient_df["total_zone_area_mm2"], errors="coerce")) & (pd.to_numeric(patient_df["total_zone_area_mm2"], errors="coerce") > 0),
        patient_df["total_target_triads"].astype(float) / pd.to_numeric(patient_df["total_zone_area_mm2"], errors="coerce").astype(float),
        np.where(patient_df["has_valid_zone_roi"] & (patient_df["total_target_triads"] <= 0), 0.0, np.nan),
    )
    patient_df["endpoint_complete"] = np.isfinite(patient_df["endpoint_time"]) & np.isfinite(patient_df["endpoint_censor"])
    patient_df["secondary_base_eligible"] = patient_df["has_valid_zone_roi"] & patient_df["endpoint_complete"]
    patient_df["secondary_exclusion_reason"] = "Included"
    patient_df.loc[~patient_df["has_zone_cells"], "secondary_exclusion_reason"] = "No cells in selected range"
    patient_df.loc[patient_df["has_zone_cells"] & ~patient_df["has_valid_zone_roi"], "secondary_exclusion_reason"] = "Cells in range but below min_cells_per_zone"
    patient_df.loc[patient_df["has_valid_zone_roi"] & ~patient_df["endpoint_complete"], "secondary_exclusion_reason"] = "Missing survival endpoint data"
    patient_df["event"] = np.where(
        np.isfinite(patient_df["endpoint_censor"]),
        (1.0 - patient_df["endpoint_censor"].astype(float)) if use_censor_as_censored else patient_df["endpoint_censor"].astype(float),
        np.nan,
    )
    patient_df["secondary_metric_label"] = f"Triad | {' + '.join(target_triad)}"
    patient_df["secondary_zone_label"] = str(
        hit_row.get("zone_label")
        or (format_zone_interval_label(zone_start_value, zone_end_value) if interval_mode else format_zone_label(zone_width))
    )

    def _metric_summary(metric_col: str, metric_label: str) -> dict[str, Any]:
        eligible_mask = patient_df["secondary_base_eligible"] & np.isfinite(pd.to_numeric(patient_df[metric_col], errors="coerce"))
        eligible_df = patient_df.loc[eligible_mask].copy()
        metric_values = pd.to_numeric(eligible_df.get(metric_col, pd.Series(dtype=float)), errors="coerce")
        summary = {
            "metric_col": metric_col,
            "metric_label": metric_label,
            "n_included": int(len(eligible_df)),
            "n_events": int(pd.to_numeric(eligible_df.get("event", pd.Series(dtype=float)), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).sum()),
            "n_zero_patients": int((metric_values <= 0).sum()) if not metric_values.empty else 0,
            "n_present_patients": int(eligible_df.get("triad_present", pd.Series(dtype=bool)).fillna(False).sum()) if not eligible_df.empty else 0,
            "n_absent_patients": int((~eligible_df.get("triad_present", pd.Series(dtype=bool)).fillna(False)).sum()) if not eligible_df.empty else 0,
            "n_primary_scoreable": int(eligible_df.get("supports_primary_permutation_score", pd.Series(dtype=bool)).fillna(False).sum()) if not eligible_df.empty else 0,
            "median_value": float(metric_values.median()) if not metric_values.empty else float("nan"),
            "min_value": float(metric_values.min()) if not metric_values.empty else float("nan"),
            "max_value": float(metric_values.max()) if not metric_values.empty else float("nan"),
        }
        summary.update(
            _continuous_cox_feature_summary(
                eligible_df,
                time_col="endpoint_time",
                event_col="event",
                feature_col=metric_col,
            )
        )
        return summary

    result["patient_df"] = patient_df.sort_values(["secondary_base_eligible", "patient"], ascending=[False, True], na_position="last").reset_index(drop=True)
    result["proportion_summary"] = _metric_summary("raw_triad_proportion", "Raw triad proportion (selected triad / all observed triads)")
    result["density_summary"] = _metric_summary("raw_triad_density", "Raw triad density (selected triads per mm^2)")
    result["status"] = "ok"
    result["message"] = ""
    result["target_triad"] = target_triad
    result["min_obs_motif_count"] = int(min_obs_motif_count)
    result["uses_permutation_threshold"] = bool(permutation_n_iter > 0)
    return result


def plot_secondary_metric_survival_km(
    patient_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    endpoint_label: str,
    config: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if patient_df.empty or metric_col not in patient_df.columns:
        ax.text(0.5, 0.5, "No secondary-analysis data are available for this metric.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    endpoint_config = _get_endpoint_config(config, endpoint_label)
    use_censor_as_censored = bool(endpoint_config.get("censor_flag_means_censored", True))
    work = patient_df.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work["endpoint_time"] = pd.to_numeric(work.get("endpoint_time", pd.Series(np.nan, index=work.index)), errors="coerce")
    work["endpoint_censor"] = pd.to_numeric(work.get("endpoint_censor", pd.Series(np.nan, index=work.index)), errors="coerce")
    work = work[
        work.get("secondary_base_eligible", False).astype(bool)
        & np.isfinite(work[metric_col])
        & np.isfinite(work["endpoint_time"])
        & np.isfinite(work["endpoint_censor"])
    ].copy()
    if work.empty or int(work[metric_col].nunique()) < 2:
        ax.text(0.5, 0.5, "Not enough variation is available to draw a visualization-only KM curve.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    cutpoint = float(work[metric_col].median())
    work["group"] = np.where(work[metric_col] >= cutpoint, "High", "Low")
    if int(work["group"].nunique()) < 2:
        ax.text(0.5, 0.5, "Display grouping produced only one group for this metric.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    work["event"] = (1.0 - work["endpoint_censor"]).clip(lower=0.0, upper=1.0) if use_censor_as_censored else work["endpoint_censor"].clip(lower=0.0, upper=1.0)
    summary_dict = summary if isinstance(summary, dict) else {}
    group_colors = _survival_group_colors(summary_dict.get("hazard_ratio_per_sd"))
    show_ci = bool(config.get("screening", {}).get("show_ci", False))
    time_unit = str(config.get("screening", {}).get("time_unit", "Time"))
    ticks = _build_survival_time_ticks(work["endpoint_time"], time_unit)
    max_tick = max((float(tick) for tick in ticks), default=float(pd.to_numeric(work["endpoint_time"], errors="coerce").max()))
    max_tick = max(max_tick, 1.0)

    plt.close(fig)
    fig = plt.figure(figsize=(13.2, 9.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[4.3, 1.5], wspace=0.25, hspace=0.12)
    ax_km = fig.add_subplot(grid[0, 0])
    ax_risk = fig.add_subplot(grid[1, 0])
    ax_text = fig.add_subplot(grid[:, 1])
    ax_km.set_box_aspect(1.0)

    legend_handles: list[Line2D] = []
    median_survivals: dict[str, float] = {}
    risk_row_labels: list[str] = []
    risk_row_counts: list[list[str]] = []

    for group_name in ["High", "Low"]:
        group_df = work[work["group"].astype(str) == group_name].copy()
        if group_df.empty:
            continue
        durations = pd.to_numeric(group_df["endpoint_time"], errors="coerce")
        events = pd.to_numeric(group_df["event"], errors="coerce").fillna(0.0)
        valid_mask = np.isfinite(durations) & np.isfinite(events)
        durations = durations.loc[valid_mask]
        events = events.loc[valid_mask].clip(lower=0.0, upper=1.0)
        if durations.empty:
            continue
        kmf = KaplanMeierFitter()
        line_color = group_colors.get(group_name, NEUTRAL_COLOR)
        legend_label = f"{group_name} (n={len(durations)})"
        kmf.fit(durations, event_observed=events, label=legend_label)
        survival_df = kmf.survival_function_.reset_index()
        time_values = pd.to_numeric(survival_df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        survival_values = pd.to_numeric(survival_df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
        ax_km.step(time_values, survival_values, where="post", color=line_color, linewidth=2.2)

        if show_ci:
            try:
                ci_df = kmf.confidence_interval_survival_function_
                if ci_df is not None and not ci_df.empty:
                    ci_time = ci_df.index.to_numpy(dtype=float)
                    ci_lower = ci_df.iloc[:, 0].to_numpy(dtype=float)
                    ci_upper = ci_df.iloc[:, 1].to_numpy(dtype=float)
                    ax_km.fill_between(ci_time, ci_lower, ci_upper, step="post", alpha=0.15, color=line_color)
            except Exception:
                pass

        median_time = kmf.median_survival_time_
        if np.isfinite(float(median_time)):
            median_survivals[group_name] = float(median_time)

        censor_times = durations.loc[events <= 0.0]
        if not censor_times.empty:
            censor_x = np.sort(censor_times.to_numpy(dtype=float))
            censor_y = np.asarray(kmf.predict(censor_x), dtype=float)
            ax_km.scatter(censor_x, censor_y, marker="|", s=90, linewidths=1.6, color=line_color, zorder=3)

        legend_handles.append(Line2D([0], [0], color=line_color, linewidth=2.2, label=legend_label))
        risk_row_labels.append(group_name)
        risk_row_counts.append([str(value) for value in _at_risk_counts(durations, ticks)])

    high_df = work[work["group"].astype(str) == "High"]
    low_df = work[work["group"].astype(str) == "Low"]
    try:
        logrank = logrank_test(
            high_df["endpoint_time"],
            low_df["endpoint_time"],
            event_observed_A=high_df["event"],
            event_observed_B=low_df["event"],
        )
        p_value = float(logrank.p_value)
    except Exception:
        p_value = float("nan")

    ax_km.set_title(metric_label)
    ax_km.set_xlabel("")
    ax_km.set_ylabel("Survival probability")
    ax_km.set_xticks(ticks)
    ax_km.set_xticklabels([_format_survival_tick(value, hide_zero=False) for value in ticks])
    ax_km.tick_params(axis="x", labelbottom=True, pad=2)
    ax_km.set_xlim(0.0, max_tick)
    ax_km.set_ylim(0.0, 1.02)
    ax_km.grid(False)
    ax_km.spines["top"].set_visible(False)
    ax_km.spines["right"].set_visible(False)

    ax_risk.axis("off")
    ax_risk.text(0.0, 0.56, "No. at Risk", transform=ax_risk.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    risk_table = ax_risk.table(
        cellText=risk_row_counts,
        rowLabels=risk_row_labels,
        colLabels=[_format_survival_tick(value, hide_zero=True) for value in ticks],
        cellLoc="center",
        rowLoc="center",
        bbox=Bbox.from_bounds(0.0, -0.16, 1.0, 0.50),
    )
    risk_table.auto_set_font_size(False)
    risk_table.set_fontsize(10)
    risk_table.scale(1.0, 1.2)
    for (row_idx, col_idx), cell in risk_table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("white")
        if row_idx == 0:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_color("#111827")
        if col_idx == -1 and row_idx > 0:
            group_name = risk_row_labels[row_idx - 1]
            cell.get_text().set_color(group_colors.get(group_name, NEUTRAL_COLOR))
            cell.get_text().set_fontweight("bold")

    stratum_label = str(_first_valid(work.get("stratum_label", pd.Series(dtype=object))) or "-")
    zone_label = str(_first_valid(work.get("secondary_zone_label", pd.Series(dtype=object))) or "-")
    feature_label = str(_first_valid(work.get("secondary_metric_label", pd.Series(dtype=object))) or "Secondary triad")
    zero_burden_patients = int((pd.to_numeric(work[metric_col], errors="coerce") <= 0).sum()) if not work.empty else 0
    primary_scoreable_patients = int(
        work.get("supports_primary_permutation_score", pd.Series(False, index=work.index)).fillna(False).astype(bool).sum()
    )

    ax_text.axis("off")
    if legend_handles:
        ax_text.legend(
            handles=legend_handles,
            title=f"{metric_label} display groups",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.95,
            fontsize=9,
            title_fontsize=10,
        )
    lines = [
        f"Selected triad: {feature_label}",
        f"Metric: {metric_label}",
        "Primary test: continuous Cox PH (per 1 SD increase)",
        f"Endpoint: {endpoint_label}",
        f"Range: {zone_label}",
        f"Stratum: {stratum_label}",
        f"Display cutpoint (median, not used for inference): {cutpoint:.3g}",
        f"Log-rank p: {p_value:.3g}" if np.isfinite(p_value) else "Log-rank p: NA",
        f"Patients in KM: {int(len(work))}",
        f"Zero-burden patients: {int(zero_burden_patients)}",
        f"Patients meeting primary Z-score threshold: {int(primary_scoreable_patients)}",
    ]
    hr = _safe_float(summary_dict.get("hazard_ratio_per_sd"), float("nan"))
    ci_low = _safe_float(summary_dict.get("ci_low_per_sd"), float("nan"))
    ci_high = _safe_float(summary_dict.get("ci_high_per_sd"), float("nan"))
    lines.append(f"Continuous Cox HR / 1 SD: {hr:.3g}" if np.isfinite(hr) else "Continuous Cox HR / 1 SD: NA")
    lines.append(
        f"95% CI: [{ci_low:.3g}, {ci_high:.3g}]"
        if np.isfinite(ci_low) and np.isfinite(ci_high)
        else "95% CI: NA"
    )
    cox_p = _safe_float(summary_dict.get("p_value"), float("nan"))
    lines.append(f"Continuous Cox p: {cox_p:.3g}" if np.isfinite(cox_p) else "Continuous Cox p: NA")
    c_idx = _safe_float(summary_dict.get("concordance_index"), float("nan"))
    if np.isfinite(c_idx):
        lines.append(f"C-index: {c_idx:.3f}")
    ph_p = _safe_float(summary_dict.get("ph_test_p"), float("nan"))
    if np.isfinite(ph_p):
        lines.append(f"PH test p: {ph_p:.3g}")
    for group_name in ["High", "Low"]:
        if group_name in median_survivals:
            lines.append(f"Median ({group_name}): {median_survivals[group_name]:.1f}")
    ax_text.text(0.02, 0.73, "\n".join(lines), va="top", ha="left", fontsize=10)

    note_lines = []
    if show_ci:
        note_lines.append("Shaded bands show 95% confidence intervals.")
    note_lines.append("Curve colors follow the continuous Cox direction: warm = worse prognosis, cool = better prognosis.")
    note_lines.append("High and Low groups are defined only for display from the median of the pooled, zero-retained patient-level burden metric.")
    note_lines.append("Vertical tick marks on the curves indicate censoring.")
    note_lines.append("The continuous Cox model was fit on the unsplit pooled patient-level burden metric.")
    note_lines.append("KM shown for visualization only; the grouping is not the inferential secondary-analysis test.")
    ax_text.text(
        0.02,
        0.18,
        "\n".join(note_lines),
        va="top",
        ha="left",
        fontsize=9,
        color="#475569",
    )
    fig.subplots_adjust(top=0.93, bottom=0.22, left=0.07, right=0.97, wspace=0.25, hspace=0.18)
    risk_bbox = ax_risk.get_position()
    fig.text((risk_bbox.x0 + risk_bbox.x1) / 2.0, risk_bbox.y0 - 0.06, time_unit, ha="center", va="top", fontsize=10.5, color="#111827")
    return fig


def _format_survival_tick(value: float, hide_zero: bool = False) -> str:
    numeric = _safe_float(value, float("nan"))
    if not np.isfinite(numeric):
        return ""
    if hide_zero and abs(numeric) < 1e-9:
        return ""
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.1f}"


def _build_survival_time_ticks(durations: pd.Series, time_unit: str) -> list[float]:
    clean = pd.to_numeric(durations, errors="coerce")
    clean = clean[np.isfinite(clean) & (clean >= 0)]
    if clean.empty:
        return [0.0]
    max_time = float(clean.max())
    if max_time <= 0:
        return [0.0]

    if "month" in str(time_unit or "").lower():
        ticks = [0.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0]
        while ticks[-1] < max_time:
            ticks.append(ticks[-1] + 12.0)
        filtered = [tick for tick in ticks if tick <= max_time + 1e-9]
        if filtered[-1] < max_time * 0.9:
            filtered.append(float(math.ceil(max_time / 6.0) * 6.0))
        return filtered

    raw = np.linspace(0.0, max_time, num=min(7, max(3, int(clean.nunique()))))
    ticks = sorted({float(round(value, 2)) for value in raw})
    if 0.0 not in ticks:
        ticks.insert(0, 0.0)
    return ticks


def _at_risk_counts(durations: pd.Series, ticks: list[float]) -> list[int]:
    clean = pd.to_numeric(durations, errors="coerce")
    clean = clean[np.isfinite(clean)]
    counts: list[int] = []
    for tick in ticks:
        tick_value = float(tick)
        if abs(tick_value) < 1e-9:
            counts.append(int((clean >= 0.0).sum()))
        else:
            counts.append(int((clean > tick_value).sum()))
    return counts


def _build_stratum_label(row: pd.Series, stratify_cols: list[str]) -> str:
    if not stratify_cols:
        return "All"
    parts = []
    for column in stratify_cols:
        parts.append(f"{column}={row.get(column, 'NA')}")
    return " | ".join(parts)


def _iter_regions(df: pd.DataFrame, batch_col: str, roi_col: str):
    group_cols = [batch_col, roi_col]
    for group_values, group_df in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        yield group_values, group_df


def _zone_subset(group_df: pd.DataFrame, distance_col: str, width: float) -> pd.DataFrame:
    mask = zone_distance_mask(group_df[distance_col], width)
    return group_df.loc[mask].copy()


def _zone_interval_subset(group_df: pd.DataFrame, distance_col: str, start: float, end: float) -> pd.DataFrame:
    mask = zone_distance_interval_mask(group_df[distance_col], start, end)
    return group_df.loc[mask].copy()


def _compute_virtual_area_mm2(
    coords: np.ndarray,
    micron_per_pixel: float,
    hole_size_px2: float,
    use_median: bool,
    percentile: float,
) -> tuple[float, float]:
    if len(coords) < 2:
        return float("nan"), float("nan")

    # Downsample points for area and buffer calculation to prevent Shapely unary_union bottleneck on massive datasets
    if len(coords) > 10000:
        rng = np.random.default_rng(20260621)
        sample_indices = rng.choice(len(coords), size=10000, replace=False)
        calc_coords = coords[sample_indices]
    else:
        calc_coords = coords

    buffer_distance = _compute_buffer_distance(
        coords=calc_coords,
        use_median=bool(use_median),
        percentile=float(percentile),
    )
    polygon = _build_region_geometry(
        coords=calc_coords,
        method="buffer_union",
        alpha=100.0,
        buffer_distance=float(buffer_distance),
        min_hole_size=float(hole_size_px2),
    )
    if polygon is None:
        return float("nan"), float(buffer_distance)

    area_px2 = float(polygon.area)
    area_mm2 = area_px2 * ((float(micron_per_pixel) / 1000.0) ** 2)
    return float(area_mm2), float(buffer_distance)


def _build_adjacency(coords: np.ndarray, min_dist_px: float, max_dist_px: float) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(len(coords))]
    if len(coords) < 2:
        return adjacency

    pairs = cKDTree(coords).query_pairs(r=float(max_dist_px), output_type="ndarray")
    if pairs is None or len(pairs) == 0:
        return adjacency

    min_d2 = float(min_dist_px) ** 2
    for idx_a, idx_b in pairs:
        dx = float(coords[int(idx_a), 0] - coords[int(idx_b), 0])
        dy = float(coords[int(idx_a), 1] - coords[int(idx_b), 1])
        if (dx * dx + dy * dy) < min_d2:
            continue
        adjacency[int(idx_a)].add(int(idx_b))
        adjacency[int(idx_b)].add(int(idx_a))
    return adjacency


def _count_dyads(phenotypes: np.ndarray, adjacency: list[set[int]], left: str, right: str) -> int:
    count = 0
    for idx_left, neighbors in enumerate(adjacency):
        if phenotypes[idx_left] != str(left):
            continue
        for idx_right in neighbors:
            if idx_right <= idx_left:
                continue
            pair = {str(phenotypes[idx_left]), str(phenotypes[idx_right])}
            if pair == {str(left), str(right)}:
                count += 1
    return int(count)


def _collect_unordered_triads(phenotypes: np.ndarray, adjacency: list[set[int]]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    seen: set[tuple[int, int, int]] = set()
    for idx_a, neighbors in enumerate(adjacency):
        neighbor_list = sorted(int(value) for value in neighbors if int(value) > idx_a)
        for pos, idx_b in enumerate(neighbor_list):
            common = adjacency[idx_b].intersection(neighbor_list[pos + 1 :])
            for idx_c in common:
                sorted_triple = sorted((int(idx_a), int(idx_b), int(idx_c)))
                triple: tuple[int, int, int] = (sorted_triple[0], sorted_triple[1], sorted_triple[2])
                if triple in seen:
                    continue
                seen.add(triple)
                sorted_key = sorted([str(phenotypes[triple[0]]), str(phenotypes[triple[1]]), str(phenotypes[triple[2]])])
                key: tuple[str, str, str] = (sorted_key[0], sorted_key[1], sorted_key[2])
                counts[key] = counts.get(key, 0) + 1
    return counts


def _compute_roi_adaptive_radius(
    coords: np.ndarray,
    min_pct: float,
    max_pct: float,
    max_sample: int,
    rng: np.random.Generator,
    direct_neighbours_only: bool = False,
) -> tuple[float, float]:
    """Compute per-ROI interaction radius from the pairwise cell-cell distance distribution.

    Two modes:

    * **Default** (``direct_neighbours_only=False``): Scans all pairs within
      3× the 95th-percentile NN spacing, then returns the ``min_pct``-th and
      ``max_pct``-th percentile of those distances.  Captures cells up to ~2-3
      typical cell spacings apart.

    * **Direct neighbours** (``direct_neighbours_only=True``): Uses the
      95th-percentile NN distance directly as ``max_d``.  Only immediate
      neighbours (one cell spacing) are considered, giving a stricter and more
      biologically conservative interaction window.

    Returns (min_px, max_px).  If computation fails, returns (0.0, nan).
    """
    n = len(coords)
    if n < 2:
        return 0.0, float("nan")

    sample_coords = coords
    if n > int(max_sample):
        idx = rng.choice(n, size=int(max_sample), replace=False)
        sample_coords = coords[idx]

    tree = cKDTree(sample_coords)
    # First pass: nearest-neighbour distances to characterise local tissue scale
    nn_raw, _ = tree.query(sample_coords, k=min(2, len(sample_coords)))
    nn_raw = np.atleast_2d(nn_raw)
    if nn_raw.shape[1] < 2:
        return 0.0, float("nan")
    nn_dists = nn_raw[:, 1]
    nn_dists = nn_dists[np.isfinite(nn_dists) & (nn_dists > 0)]
    if nn_dists.size == 0:
        return 0.0, float("nan")

    p95_nn = float(np.percentile(nn_dists, 95))

    if direct_neighbours_only:
        # Use the 95th-pct NN distance directly — only direct neighbours qualify
        min_d = 0.0
        max_d = p95_nn
        return max(0.0, min_d), max(min_d + 1e-6, max_d)

    # Search radius: up to 3× the 95th-percentile NN gap to capture local pairs
    search_r = p95_nn * 3.0

    # Second pass: collect all pairwise distances within the search radius
    pairs = tree.query_pairs(r=search_r, output_type="ndarray")
    if pairs is not None and len(pairs) > 0:
        dx = sample_coords[pairs[:, 0], 0] - sample_coords[pairs[:, 1], 0]
        dy = sample_coords[pairs[:, 0], 1] - sample_coords[pairs[:, 1], 1]
        pair_dists = np.sqrt(dx * dx + dy * dy)
        pair_dists = pair_dists[pair_dists > 0]
    else:
        pair_dists = nn_dists  # fallback to NN distances

    if pair_dists.size == 0:
        return 0.0, float("nan")

    min_d = float(np.percentile(pair_dists, float(min_pct)))
    max_d = float(np.percentile(pair_dists, float(max_pct)))
    return max(0.0, min_d), max(min_d + 1e-6, max_d)


def _build_edge_array(adjacency: list[set[int]]) -> np.ndarray:
    """Return all edges from an adjacency list as an (M, 2) int32 array with i < j."""
    edges: list[tuple[int, int]] = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            if j > i:
                edges.append((i, j))
    return np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)


def _find_triangle_array(adjacency: list[set[int]]) -> np.ndarray:
    """Return all closed 3-cliques as an (T, 3) int32 array with sorted indices."""
    triangles: list[tuple[int, int, int]] = []
    for i, neighbors_i in enumerate(adjacency):
        for j in sorted(int(v) for v in neighbors_i if int(v) > i):
            common = adjacency[j].intersection(neighbors_i)
            for k in sorted(int(v) for v in common if int(v) > j):
                triangles.append((i, j, k))
    return np.array(triangles, dtype=np.int32) if triangles else np.empty((0, 3), dtype=np.int32)


def _count_motifs_vectorized(
    pheno_ints: np.ndarray,
    edges: np.ndarray,
    triangles: np.ndarray,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int, int], int]]:
    """Vectorised count of all dyad and triad motifs.

    Uses integer-encoded phenotype labels and numpy ``unique`` for speed.
    Returns dicts keyed by sorted integer-tuples.
    """
    dyad_counts: dict[tuple[int, int], int] = {}
    if edges.shape[0] > 0:
        left = pheno_ints[edges[:, 0]]
        right = pheno_ints[edges[:, 1]]
        pair_sorted = np.sort(np.stack([left, right], axis=1), axis=1)
        unique_pairs, cnts = np.unique(pair_sorted, axis=0, return_counts=True)
        for row_i in range(len(unique_pairs)):
            dyad_counts[(int(unique_pairs[row_i, 0]), int(unique_pairs[row_i, 1]))] = int(cnts[row_i])

    triad_counts: dict[tuple[int, int, int], int] = {}
    if triangles.shape[0] > 0:
        a = pheno_ints[triangles[:, 0]]
        b = pheno_ints[triangles[:, 1]]
        c = pheno_ints[triangles[:, 2]]
        tri_sorted = np.sort(np.stack([a, b, c], axis=1), axis=1)
        unique_tris, cnts = np.unique(tri_sorted, axis=0, return_counts=True)
        for row_i in range(len(unique_tris)):
            triad_counts[(int(unique_tris[row_i, 0]), int(unique_tris[row_i, 1]), int(unique_tris[row_i, 2]))] = int(cnts[row_i])

    return dyad_counts, triad_counts


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise-preserving (dyad-constrained) motif null
#
# The plain label-shuffling null destroys pairwise and higher-order structure at
# the same time, so a triad enriched purely because two of its pairs like each
# other scores just as highly as a genuine three-cell unit. This null instead
# holds the observed dyad composition fixed and asks whether the triad still
# occurs more often than that pairwise structure alone accounts for.
#
# Implemented as constrained randomisation (Milo et al. 2002): Metropolis label
# swaps on the fixed adjacency graph, with energy
#     E = sum_k (D_k - D_k^obs)^2
# over dyad-type counts D. A small temperature keeps the walk near the
# constraint surface (E ~ 0, i.e. dyad counts preserved) while letting the
# three-way arrangement decorrelate from the observed one.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from numba import njit as _njit

    NUMBA_AVAILABLE = True
except Exception:  # numba missing -> pure-Python fallback below
    NUMBA_AVAILABLE = False

    def _njit(*args, **kwargs):
        def _wrap(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return _wrap


@_njit(cache=True, fastmath=True)
def _constrained_swap_chain(
    labels, nbr, ptr, dyad_now, dyad_obs, n_pheno,
    n_steps, temperature, scratch, touched, seed,
):
    """One Metropolis chain of dyad-count-preserving label swaps (in place).

    ``labels`` is mutated. ``dyad_now`` tracks the current dyad-type counts and
    is kept exact. Returns the number of accepted swaps.
    """
    np.random.seed(seed)
    n_cells = labels.shape[0]
    accepted = 0

    for _step in range(n_steps):
        u = np.random.randint(0, n_cells)
        v = np.random.randint(0, n_cells)
        lu = labels[u]
        lv = labels[v]
        if lu == lv:
            continue

        # Build the exact change to the dyad-count vector without touching it,
        # so a rejected proposal costs nothing to undo.
        n_touched = 0
        for k in range(ptr[u], ptr[u + 1]):
            w = nbr[k]
            if w == v:
                continue
            lw = labels[w]
            a = lu if lu < lw else lw
            b = lw if lu < lw else lu
            c_old = a * n_pheno + b
            a = lv if lv < lw else lw
            b = lw if lv < lw else lv
            c_new = a * n_pheno + b
            if scratch[c_old] == 0:
                touched[n_touched] = c_old
                n_touched += 1
            scratch[c_old] -= 1
            if scratch[c_new] == 0:
                touched[n_touched] = c_new
                n_touched += 1
            scratch[c_new] += 1

        for k in range(ptr[v], ptr[v + 1]):
            w = nbr[k]
            if w == u:
                continue
            lw = labels[w]
            a = lv if lv < lw else lw
            b = lw if lv < lw else lv
            c_old = a * n_pheno + b
            a = lu if lu < lw else lw
            b = lw if lu < lw else lu
            c_new = a * n_pheno + b
            if scratch[c_old] == 0:
                touched[n_touched] = c_old
                n_touched += 1
            scratch[c_old] -= 1
            if scratch[c_new] == 0:
                touched[n_touched] = c_new
                n_touched += 1
            scratch[c_new] += 1

        # Energy change over only the codes this swap actually moved.
        delta_e = 0.0
        for i in range(n_touched):
            code = touched[i]
            d = scratch[code]
            if d == 0:
                continue
            cur = dyad_now[code] - dyad_obs[code]
            delta_e += float((cur + d) * (cur + d) - cur * cur)

        accept = delta_e <= 0.0
        if not accept and temperature > 0.0:
            accept = np.random.random() < np.exp(-delta_e / temperature)

        if accept:
            for i in range(n_touched):
                code = touched[i]
                dyad_now[code] += scratch[code]
            labels[u] = lv
            labels[v] = lu
            accepted += 1

        for i in range(n_touched):
            scratch[touched[i]] = 0

    return accepted


def _triad_counts_from_labels(labels, triangles, n_pheno):
    """Sorted-triple code counts for one labelling."""
    if triangles.shape[0] == 0:
        return np.zeros(n_pheno ** 3, dtype=np.int64)
    a = labels[triangles[:, 0]].astype(np.int64)
    b = labels[triangles[:, 1]].astype(np.int64)
    c = labels[triangles[:, 2]].astype(np.int64)
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    mid = a + b + c - lo - hi
    return np.bincount(lo * (n_pheno ** 2) + mid * n_pheno + hi, minlength=n_pheno ** 3)


def _dyad_counts_from_labels(labels, edges, n_pheno):
    if edges.shape[0] == 0:
        return np.zeros(n_pheno * n_pheno, dtype=np.int64)
    e0 = labels[edges[:, 0]].astype(np.int64)
    e1 = labels[edges[:, 1]].astype(np.int64)
    return np.bincount(np.minimum(e0, e1) * n_pheno + np.maximum(e0, e1),
                       minlength=n_pheno * n_pheno)


def _build_csr_adjacency(edges, n_cells):
    """Undirected CSR neighbour lists from an (M, 2) edge array."""
    if edges.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(n_cells + 1, dtype=np.int32)
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    order = np.argsort(both[:, 0], kind="stable")
    both = both[order]
    nbr = np.ascontiguousarray(both[:, 1].astype(np.int32))
    counts = np.bincount(both[:, 0], minlength=n_cells)
    ptr = np.zeros(n_cells + 1, dtype=np.int32)
    ptr[1:] = np.cumsum(counts)
    return nbr, ptr


def pairwise_preserving_triad_zscores(
    phenotype_array: np.ndarray,
    edges: np.ndarray,
    triangles: np.ndarray,
    n_samples: int = 200,
    sweeps_per_sample: int = 4,
    burn_in_sweeps: int = 10,
    temperature: float = 1.0,
    seed: int = 20260407,
    min_obs_count: int = 3,
    max_dyad_drift: float = 0.02,
) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    """Triad Z-scores against a null that holds the observed dyad counts fixed.

    Answers "is this trio more common than its own pairwise affinities already
    explain", as opposed to the plain permutation null's much weaker "is this
    trio more common than randomly scattered labels".

    A sweep is ``n_cells`` proposed swaps. The chain burns in, then a triad
    count is taken every ``sweeps_per_sample`` sweeps. ``max_dyad_drift`` is a
    guard, not a tuning knob: if the sampled configurations drift further than
    this (relative L1) from the observed dyad composition, the constraint was
    not actually held and the diagnostics say so rather than returning numbers
    that look fine.

    Returns ``(zscores, diagnostics)``.
    """
    diagnostics: dict[str, Any] = {
        "status": "ok", "n_samples": 0, "acceptance_rate": float("nan"),
        "dyad_drift": float("nan"), "constraint_held": False,
        "numba": bool(NUMBA_AVAILABLE),
    }
    if edges.shape[0] == 0 or triangles.shape[0] == 0 or n_samples <= 0:
        diagnostics["status"] = "no_motifs"
        return {}, diagnostics

    unique_labels, pheno_ints = np.unique(phenotype_array, return_inverse=True)
    n_pheno = int(len(unique_labels))
    if n_pheno < 3:
        diagnostics["status"] = "too_few_phenotypes"
        return {}, diagnostics

    labels0 = pheno_ints.astype(np.int32)
    n_cells = int(labels0.shape[0])
    obs_tri = _triad_counts_from_labels(labels0, triangles, n_pheno)
    dyad_obs = _dyad_counts_from_labels(labels0, edges, n_pheno).astype(np.int64)
    nbr, ptr = _build_csr_adjacency(edges, n_cells)

    labels = labels0.copy()
    dyad_now = dyad_obs.copy()
    scratch = np.zeros(n_pheno * n_pheno, dtype=np.int64)
    touched = np.zeros(2 * (int(nbr.shape[0]) + 4), dtype=np.int64)

    total_steps = 0
    accepted = 0
    burn_steps = int(max(0, burn_in_sweeps) * n_cells)
    if burn_steps:
        accepted += int(_constrained_swap_chain(
            labels, nbr, ptr, dyad_now, dyad_obs, n_pheno,
            burn_steps, float(temperature), scratch, touched, int(seed),
        ))
        total_steps += burn_steps

    steps_per_sample = int(max(1, sweeps_per_sample) * n_cells)
    null_sum = np.zeros(n_pheno ** 3, dtype=np.float64)
    null_sum2 = np.zeros(n_pheno ** 3, dtype=np.float64)
    drift_acc = 0.0
    dyad_total = float(max(int(dyad_obs.sum()), 1))

    for sample_index in range(int(n_samples)):
        accepted += int(_constrained_swap_chain(
            labels, nbr, ptr, dyad_now, dyad_obs, n_pheno,
            steps_per_sample, float(temperature), scratch, touched,
            int(seed) + sample_index + 1,
        ))
        total_steps += steps_per_sample
        counts = _triad_counts_from_labels(labels, triangles, n_pheno).astype(np.float64)
        null_sum += counts
        null_sum2 += counts ** 2
        drift_acc += float(np.abs(dyad_now - dyad_obs).sum()) / dyad_total

    n_eff = int(n_samples)
    mean = null_sum / n_eff
    var = (null_sum2 - null_sum ** 2 / n_eff) / float(max(n_eff - 1, 1))
    std = np.sqrt(np.maximum(var, 0.0))

    drift = drift_acc / float(max(n_eff, 1))
    diagnostics.update({
        "n_samples": n_eff,
        "acceptance_rate": float(accepted) / float(max(total_steps, 1)),
        "dyad_drift": float(drift),
        "constraint_held": bool(drift <= float(max_dyad_drift)),
    })
    if not diagnostics["constraint_held"]:
        diagnostics["status"] = "constraint_drifted"

    zscores: dict[tuple[str, str, str], float] = {}
    for code in np.where(obs_tri >= int(max(0, min_obs_count)))[0]:
        sd = float(std[code])
        if sd < 1e-9:
            continue
        i = int(code) // (n_pheno ** 2)
        j = (int(code) % (n_pheno ** 2)) // n_pheno
        k = int(code) % n_pheno
        if len({i, j, k}) < 3:
            # Homotypic and part-homotypic triples are the pairwise structure
            # itself; this null holds that fixed, so they carry no information.
            continue
        zscores[(str(unique_labels[i]), str(unique_labels[j]), str(unique_labels[k]))] = float(
            (float(obs_tri[code]) - mean[code]) / sd
        )
    return zscores, diagnostics


def _permutation_null_accumulators_gpu(
    pheno_ints: np.ndarray,
    edges: np.ndarray,
    triangles: np.ndarray,
    n_iter: int,
    n_pheno: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Accumulate the permutation null on the GPU; ``None`` if CUDA is unusable.

    Mirrors the CPU inner loop exactly -- same label permutation, same motif
    encoding, same running sum / sum-of-squares accumulation -- so the only
    difference in the resulting Z-scores is Monte-Carlo noise from a different
    random stream, exactly as between two CPU runs with different seeds.

    Returns ``(dyad_sum, dyad_sum2, triad_sum, triad_sum2)`` as float64 arrays.
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None

    try:
        dev = torch.device("cuda")
        pit = torch.as_tensor(np.asarray(pheno_ints, dtype=np.int64), device=dev)
        n_cells = int(pit.numel())
        d_size = n_pheno * n_pheno
        t_size = n_pheno ** 3
        has_tri = triangles.shape[0] > 0

        E = torch.as_tensor(edges.astype(np.int64), device=dev)
        T = torch.as_tensor(triangles.astype(np.int64), device=dev) if has_tri else None

        d_sum = torch.zeros(d_size, dtype=torch.float64, device=dev)
        d_sum2 = torch.zeros(d_size, dtype=torch.float64, device=dev)
        t_sum = torch.zeros(t_size, dtype=torch.float64, device=dev) if has_tri else None
        t_sum2 = torch.zeros(t_size, dtype=torch.float64, device=dev) if has_tri else None

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        # The GPU has far more headroom than the CPU path, but a chunk still
        # scales with (permutations x motifs); cap it against free VRAM so a
        # dense ROI cannot abort the run with an out-of-memory error.
        work_rows = int(max(int(edges.shape[0]), int(triangles.shape[0]), 1))
        free_bytes, _total = torch.cuda.mem_get_info()
        budget = max(1, int(free_bytes * 0.25) // (8 * 6))
        eff_chunk = int(np.clip(budget // work_rows, 1, max(int(chunk_size), 1)))

        done = 0
        while done < n_iter:
            c = min(eff_chunk, n_iter - done)
            perm = torch.rand(c, n_cells, device=dev, generator=gen).argsort(dim=1)
            ch = pit[perm]

            left = ch[:, E[:, 0]]
            right = ch[:, E[:, 1]]
            codes_d = torch.minimum(left, right) * n_pheno + torch.maximum(left, right)
            off_d = torch.arange(c, device=dev).unsqueeze(1) * d_size
            cnt_d = torch.bincount(
                (codes_d + off_d).reshape(-1), minlength=c * d_size
            ).reshape(c, d_size).double()
            d_sum += cnt_d.sum(0)
            d_sum2 += (cnt_d ** 2).sum(0)
            del left, right, codes_d, off_d, cnt_d

            if has_tri and T is not None and t_sum is not None and t_sum2 is not None:
                pa, pb, pc = ch[:, T[:, 0]], ch[:, T[:, 1]], ch[:, T[:, 2]]
                lo = torch.minimum(torch.minimum(pa, pb), pc)
                hi = torch.maximum(torch.maximum(pa, pb), pc)
                mid = pa + pb + pc - lo - hi
                codes_t = lo * (n_pheno ** 2) + mid * n_pheno + hi
                off_t = torch.arange(c, device=dev).unsqueeze(1) * t_size
                cnt_t = torch.bincount(
                    (codes_t + off_t).reshape(-1), minlength=c * t_size
                ).reshape(c, t_size).double()
                t_sum += cnt_t.sum(0)
                t_sum2 += (cnt_t ** 2).sum(0)
                del pa, pb, pc, lo, hi, mid, codes_t, off_t, cnt_t
            del ch, perm
            done += c

        out = (
            d_sum.cpu().numpy(),
            d_sum2.cpu().numpy(),
            (t_sum.cpu().numpy() if t_sum is not None else np.zeros(t_size, dtype=np.float64)),
            (t_sum2.cpu().numpy() if t_sum2 is not None else np.zeros(t_size, dtype=np.float64)),
        )
        torch.cuda.empty_cache()
        return out
    except Exception:
        # Any CUDA failure (OOM, driver mismatch) falls back to the CPU path
        # rather than aborting a long screening run.
        try:
            import torch as _t

            _t.cuda.empty_cache()
        except Exception:
            pass
        return None


def _permutation_zscore_dyad_triad(
    phenotype_array: np.ndarray,
    edges: np.ndarray,
    triangles: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    chunk_size: int = 64,
    min_obs_count: int = 3,
    use_gpu: bool = False,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str, str], float]]:
    """Vectorised permutation Z-scores — no Python loop over iterations.

    Replaces the previous per-iteration Python loop with chunked numpy matrix
    operations:

    * ``rng.permuted(matrix, axis=1)`` generates ``chunk_size`` independent
      phenotype-label permutations in one NumPy call.
    * An offset-and-flatten trick lets a single ``np.bincount`` count all motifs
      across an entire chunk of permutations simultaneously.
    * Running sum / sum-of-squares accumulation avoids storing all null counts in
      memory.

    Typical speed-up over the previous loop: **50–200×** depending on ROI size
    and number of phenotypes.

    Args:
        phenotype_array: String array of cell phenotypes (length = n_cells).
        edges:      (M, 2) int32 array — edge index pairs (i < j).
        triangles:  (T, 3) int32 array — triangle index triples (i < j < k).
        n_iter:     Number of permutation iterations.
        rng:        Seeded NumPy random generator (per-ROI, thread-safe).
        chunk_size: Permutations processed per batch.  Tune to balance
                    memory vs. Python-loop overhead (default 64).
    """
    if n_iter <= 0 or edges.shape[0] == 0:
        return {}, {}

    unique_labels, pheno_ints = np.unique(phenotype_array, return_inverse=True)
    pheno_ints = pheno_ints.astype(np.int32)
    n_pheno = int(len(unique_labels))

    # ── Observed counts (integer-encoded) ────────────────────────────────────
    e0 = pheno_ints[edges[:, 0]]
    e1 = pheno_ints[edges[:, 1]]
    obs_dyad_codes = (np.minimum(e0, e1).astype(np.int64) * n_pheno
                      + np.maximum(e0, e1).astype(np.int64))
    d_null_size = n_pheno * n_pheno
    obs_dyad_counts = np.bincount(obs_dyad_codes, minlength=d_null_size)

    has_tri = triangles.shape[0] > 0
    t_null_size = n_pheno ** 3
    if has_tri:
        ta = pheno_ints[triangles[:, 0]]
        tb = pheno_ints[triangles[:, 1]]
        tc = pheno_ints[triangles[:, 2]]
        tri_s = np.sort(np.stack([ta, tb, tc], axis=1), axis=1)
        obs_tri_codes = (tri_s[:, 0].astype(np.int64) * (n_pheno ** 2)
                         + tri_s[:, 1].astype(np.int64) * n_pheno
                         + tri_s[:, 2].astype(np.int64))
        obs_tri_counts = np.bincount(obs_tri_codes, minlength=t_null_size)
    else:
        obs_tri_counts = np.zeros(t_null_size, dtype=np.int64)

    # ── Null distribution: chunked, fully vectorised ──────────────────────────
    # Accumulate sum and sum-of-squares for online mean/variance (no storage of
    # all n_iter counts needed).
    null_dyad_sum = np.zeros(d_null_size, dtype=np.float64)
    null_dyad_sum2 = np.zeros(d_null_size, dtype=np.float64)
    null_tri_sum = np.zeros(t_null_size, dtype=np.float64) if has_tri else None
    null_tri_sum2 = np.zeros(t_null_size, dtype=np.float64) if has_tri else None

    # Cap the working set instead of trusting a fixed chunk size. Every chunk
    # allocates on the order of c_size * max(n_edges, n_tri) elements, and a
    # dense ROI can carry hundreds of thousands of triangles -- at chunk_size=64
    # that is several GB, which is what makes triads fail on large datasets
    # (and fail hardest when ROIs are processed in parallel). Shrinking the
    # chunk changes only how many permutations are batched per NumPy call, never
    # the result: the accumulators below are exact running sums.
    _work_rows = int(max(int(edges.shape[0]), int(triangles.shape[0]), 1))
    # ~52 bytes per (permutation x motif) across the int32/int64 temporaries,
    # so 8M keeps one ROI's peak near 0.4 GB and leaves headroom when several
    # ROIs are screened in parallel.
    _max_cells_per_chunk = 8_000_000
    _effective_chunk = int(np.clip(_max_cells_per_chunk // _work_rows, 1, int(chunk_size)))

    # GPU path: identical accumulators, ~60x faster on an A100. Falls through to
    # the CPU loop below whenever CUDA is unavailable or the transfer fails, so
    # enabling the checkbox on a CPU-only node changes nothing but a log line.
    _gpu_done = False
    if use_gpu and resolve_use_gpu(True):
        _gpu_out = _permutation_null_accumulators_gpu(
            pheno_ints, edges, triangles, int(n_iter), n_pheno,
            int(rng.integers(0, 2**31 - 1)), int(chunk_size),
        )
        if _gpu_out is not None:
            null_dyad_sum, null_dyad_sum2, _tri_s, _tri_s2 = _gpu_out
            if has_tri:
                null_tri_sum, null_tri_sum2 = _tri_s, _tri_s2
            _gpu_done = True

    iter_done = n_iter if _gpu_done else 0
    while iter_done < n_iter:
        c_size = min(_effective_chunk, n_iter - iter_done)
        # Generate c_size independent permutations in one NumPy call
        chunk = np.tile(pheno_ints, (c_size, 1))      # (c_size, n_cells)
        rng.permuted(chunk, axis=1, out=chunk)         # in-place row-wise permutation

        # ── Dyad chunk ──────────────────────────────────────────────────────
        left = chunk[:, edges[:, 0]]                   # (c_size, n_edges)
        right = chunk[:, edges[:, 1]]
        codes_d = (np.minimum(left, right).astype(np.int64) * n_pheno
                   + np.maximum(left, right).astype(np.int64))  # (c_size, n_edges)
        # Offset trick: flatten all c_size × n_edges codes into one bincount
        off_d = np.arange(c_size, dtype=np.int64)[:, None] * d_null_size
        flat_d = np.bincount((codes_d + off_d).ravel(), minlength=c_size * d_null_size)
        chunk_d = flat_d.reshape(c_size, d_null_size).astype(np.float64)  # (c_size, n_pheno²)
        null_dyad_sum += chunk_d.sum(axis=0)
        null_dyad_sum2 += (chunk_d ** 2).sum(axis=0)

        # ── Triad chunk ─────────────────────────────────────────────────────
        if has_tri and null_tri_sum is not None and null_tri_sum2 is not None:
            pa = chunk[:, triangles[:, 0]]             # (c_size, n_tri)
            pb = chunk[:, triangles[:, 1]]
            pc = chunk[:, triangles[:, 2]]
            # Sort the three labels by arithmetic rather than np.sort on a
            # stacked (c_size, n_tri, 3) array. The stack plus its sorted copy
            # are the two largest allocations in this loop and the sort itself
            # dominates the runtime; min/max/sum reproduces the ordered triple
            # exactly (mid = a+b+c-lo-hi) at ~5x the speed and without ever
            # materialising the 3-deep array.
            lo = np.minimum(np.minimum(pa, pb), pc)
            hi = np.maximum(np.maximum(pa, pb), pc)
            mid = (pa.astype(np.int32) + pb + pc) - lo - hi
            codes_t = (lo.astype(np.int64) * (n_pheno ** 2)
                       + mid.astype(np.int64) * n_pheno
                       + hi.astype(np.int64))
            del lo, hi, mid, pa, pb, pc
            off_t = np.arange(c_size, dtype=np.int64)[:, None] * t_null_size
            flat_t = np.bincount((codes_t + off_t).ravel(), minlength=c_size * t_null_size)
            chunk_t = flat_t.reshape(c_size, t_null_size).astype(np.float64)
            null_tri_sum += chunk_t.sum(axis=0)
            null_tri_sum2 += (chunk_t ** 2).sum(axis=0)

        iter_done += c_size

    # ── Z-scores from accumulated statistics ─────────────────────────────────
    denom = float(max(n_iter - 1, 1))
    null_dyad_mean = null_dyad_sum / n_iter
    # Bessel-corrected variance: (Σx² − (Σx)²/n) / (n−1)
    null_dyad_var = (null_dyad_sum2 - null_dyad_sum ** 2 / n_iter) / denom
    null_dyad_std = np.sqrt(np.maximum(null_dyad_var, 0.0))

    _min_obs = max(0, int(min_obs_count))
    dyad_zscores: dict[tuple[str, str], float] = {}
    for code_idx in np.where(obs_dyad_counts >= _min_obs)[0]:
        std = float(null_dyad_std[code_idx])
        if std < 1e-9:
            continue
        ii = int(code_idx) // n_pheno
        jj = int(code_idx) % n_pheno
        z = float((int(obs_dyad_counts[code_idx]) - null_dyad_mean[code_idx]) / std)
        dyad_zscores[(str(unique_labels[ii]), str(unique_labels[jj]))] = z

    triad_zscores: dict[tuple[str, str, str], float] = {}
    if has_tri and null_tri_sum is not None and null_tri_sum2 is not None:
        null_tri_mean = null_tri_sum / n_iter
        null_tri_var = (null_tri_sum2 - null_tri_sum ** 2 / n_iter) / denom
        null_tri_std = np.sqrt(np.maximum(null_tri_var, 0.0))
        for code_idx in np.where(obs_tri_counts >= _min_obs)[0]:
            std_t = float(null_tri_std[code_idx])
            if std_t < 1e-9:
                continue
            ii = int(code_idx) // (n_pheno ** 2)
            jj = (int(code_idx) % (n_pheno ** 2)) // n_pheno
            kk = int(code_idx) % n_pheno
            z = float((int(obs_tri_counts[code_idx]) - null_tri_mean[code_idx]) / std_t)
            triad_zscores[(str(unique_labels[ii]), str(unique_labels[jj]), str(unique_labels[kk]))] = z

    return dyad_zscores, triad_zscores


def _mean_knn_distance(source_coords: np.ndarray, target_coords: np.ndarray, k: int, same_population: bool) -> float:
    if len(source_coords) == 0 or len(target_coords) == 0:
        return float("nan")
    tree = cKDTree(target_coords)
    query_k = int(max(1, k + 1 if same_population else k))
    distances, _ = tree.query(source_coords, k=query_k)
    distances = np.atleast_2d(distances)
    if same_population:
        usable = distances[:, 1:]
    else:
        usable = distances[:, : int(max(1, k))]
    usable = usable[np.isfinite(usable)]
    if usable.size == 0:
        return float("nan")
    return float(np.nanmean(usable))


# ─────────────────────────────────────────────────────────────────────────────
# Motif visualization helpers (used by the representative-ROI panel)
# ─────────────────────────────────────────────────────────────────────────────

def compute_roi_motif_visualization_data(
    zone_df: pd.DataFrame,
    config: dict[str, Any],
    target_phenotypes: list[str],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict[str, Any]:
    """Return all spatial objects needed for the three motif-visualization figures.

    Finds every occurrence of the target dyad/triad in the ROI using the same
    adaptive-radius adjacency as the screening pipeline, then converts coordinates
    to image-display space using ``scale_x``/``scale_y``.

    Returns a dict with:
    ``centroids``        – (N, 2) float array of per-motif centroids in plot space
    ``instances``        – list of cell-index tuples (pairs or triples)
    ``plot_coords``      – (M, 2) float array, all cells in plot space
    ``pheno_arr``        – (M,) str array of phenotype labels (same ordering as zone_df)
    ``max_px_scaled``    – interaction radius in plot-space pixels
    ``n_found``          – number of motif instances found
    ``target_sorted``    – sorted tuple of target phenotype names
    """
    screening = config["screening"]
    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])
    phenotype_col = str(config["columns"]["phenotype"])
    use_micron_per_pixel = bool(screening.get("use_micron_per_pixel", True))
    micron_per_pixel = max(_safe_float(screening.get("micron_per_pixel"), 0.5083), 1e-6) if use_micron_per_pixel else 1.0

    coords_px = zone_df[[x_col, y_col]].to_numpy(dtype=float)
    pheno_arr = zone_df[phenotype_col].astype(str).to_numpy()
    plot_coords = np.stack([coords_px[:, 0] * scale_x, coords_px[:, 1] * scale_y], axis=1)

    # ── Determine interaction radius ─────────────────────────────────────────
    use_adaptive = bool(screening.get("use_adaptive_interaction_radius", True))
    _direct_nn = bool(screening.get("adaptive_radius_direct_neighbours_only", False))
    _rng = np.random.default_rng(42)
    if use_adaptive:
        adap_min, adap_max = _compute_roi_adaptive_radius(
            coords_px,
            _safe_float(screening.get("adaptive_radius_min_pct"), 0.1),
            _safe_float(screening.get("adaptive_radius_max_pct"), 99.0),
            max(50, _safe_int(screening.get("adaptive_radius_max_sample"), 500)),
            _rng,
            direct_neighbours_only=_direct_nn,
        )
        if np.isfinite(adap_max) and adap_max > 0:
            min_px, max_px = adap_min, adap_max
        else:
            min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
            max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel
    else:
        min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
        max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel

    adj = _build_adjacency(coords_px, min_px, max_px)
    edges = _build_edge_array(adj)
    triangles = _find_triangle_array(adj)

    target_sorted: tuple[str, ...] = tuple(sorted(str(p) for p in target_phenotypes))
    n_target = len(target_sorted)
    centroids: list[list[float]] = []
    instances: list[tuple[int, ...]] = []

    if n_target == 2 and edges.shape[0] > 0:
        p0, p1 = target_sorted
        for row in edges:
            i, j = int(row[0]), int(row[1])
            if tuple(sorted([str(pheno_arr[i]), str(pheno_arr[j])])) == (p0, p1):
                centroids.append([(plot_coords[i, 0] + plot_coords[j, 0]) / 2.0,
                                   (plot_coords[i, 1] + plot_coords[j, 1]) / 2.0])
                instances.append((i, j))

    elif n_target == 3 and triangles.shape[0] > 0:
        p0, p1, p2 = target_sorted
        for row in triangles:
            i, j, k = int(row[0]), int(row[1]), int(row[2])
            if tuple(sorted([str(pheno_arr[i]), str(pheno_arr[j]), str(pheno_arr[k])])) == (p0, p1, p2):
                centroids.append([(plot_coords[i, 0] + plot_coords[j, 0] + plot_coords[k, 0]) / 3.0,
                                   (plot_coords[i, 1] + plot_coords[j, 1] + plot_coords[k, 1]) / 3.0])
                instances.append((i, j, k))

    centroids_arr = np.array(centroids, dtype=float) if centroids else np.empty((0, 2), dtype=float)
    return {
        "centroids": centroids_arr,
        "instances": instances,
        "plot_coords": plot_coords,
        "pheno_arr": pheno_arr,
        "max_px_scaled": max_px * max(scale_x, scale_y),
        "n_found": len(instances),
        "target_sorted": target_sorted,
    }


def plot_motif_density_cloud(
    zone_df: pd.DataFrame,
    motif_data: dict[str, Any],
    target_phenotypes: list[str],
    color_mapping: dict[str, str],
    image: "np.ndarray | None",
    scale_x: float,
    scale_y: float,
    title: str = "",
) -> Figure:
    """Kernel density cloud showing where dyad/triad motifs are spatially concentrated.

    The cloud is drawn as a filled contour over the ROI.  Denser areas (more
    motif instances) appear in darker/more opaque colour.  All ROI cells are
    shown as a dimmed background so tissue architecture stays visible.
    """
    from scipy.stats import gaussian_kde  # lazy import

    centroids = motif_data["centroids"]            # (N, 2) in plot space
    plot_coords = motif_data["plot_coords"]        # (M, 2) all cells in plot space
    pheno_arr = motif_data["pheno_arr"]
    n_found = int(motif_data["n_found"])

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    if image is not None:
        ax.imshow(image, aspect="auto")

    # Background: all cells dimmed
    if plot_coords.shape[0] > 0:
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
                   s=6, alpha=0.22, c="#b0b0b0", edgecolors="none", zorder=1)

    # Target-phenotype cells on top
    for ph in sorted(set(str(p) for p in target_phenotypes)):
        mask = pheno_arr == ph
        if mask.any():
            ax.scatter(plot_coords[mask, 0], plot_coords[mask, 1],
                       s=18, alpha=0.65, c=color_mapping.get(ph, "#555555"),
                       edgecolors="white", linewidths=0.3, zorder=2,
                       label=ph)

    # KDE density cloud
    if n_found >= 3:
        # Bandwidth selection: Scott's rule on centroid scatter
        kde = gaussian_kde(centroids.T)
        # Grid spanning the cell bounding box
        x_min, x_max = float(plot_coords[:, 0].min()), float(plot_coords[:, 0].max())
        y_min, y_max = float(plot_coords[:, 1].min()), float(plot_coords[:, 1].max())
        pad_x = max((x_max - x_min) * 0.04, 5.0)
        pad_y = max((y_max - y_min) * 0.04, 5.0)
        grid_x, grid_y = np.mgrid[x_min - pad_x: x_max + pad_x: 80j,
                                   y_min - pad_y: y_max + pad_y: 80j]
        kde_vals = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        # Transparent where density is very low
        kde_norm = (kde_vals - kde_vals.min()) / max(kde_vals.max() - kde_vals.min(), 1e-12)
        ax.contourf(grid_x, grid_y, kde_norm,
                    levels=8, cmap="YlOrRd", alpha=0.55, zorder=3)
        # Centroid scatter on top of cloud
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   s=28, marker="x", c="#7b0000", linewidths=1.1,
                   alpha=0.75, zorder=4, label=f"Motif centroids (n={n_found})")
    elif n_found > 0:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   s=40, marker="*", c="#c0392b", zorder=4,
                   label=f"Motif centroid(s) (n={n_found})")
    else:
        ax.text(0.5, 0.5, "No motif instances found in this ROI",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#555555")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8,
                  framealpha=0.75, markerscale=1.2)

    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    return fig


def plot_motif_zoom(
    zone_df: pd.DataFrame,
    motif_data: dict[str, Any],
    target_phenotypes: list[str],
    color_mapping: dict[str, str],
    config: dict[str, Any],
    image: "np.ndarray | None",
    scale_x: float,
    scale_y: float,
    title: str = "",
) -> Figure:
    """Zoomed-in figure on the most representative single motif instance.

    Selects the motif instance whose centroid is closest to the overall mean
    centroid (i.e., the most spatially typical example), zooms in with a
    padding proportional to the interaction distance, and draws labelled edges
    connecting the constituent cells.
    """
    centroids = motif_data["centroids"]
    instances = motif_data["instances"]
    plot_coords = motif_data["plot_coords"]
    pheno_arr = motif_data["pheno_arr"]
    max_r = float(motif_data["max_px_scaled"])
    n_found = int(motif_data["n_found"])

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    if image is not None:
        ax.imshow(image, aspect="auto")

    if n_found == 0 or len(instances) == 0:
        ax.text(0.5, 0.5, "No motif instance found in this ROI",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#555555")
        ax.set_axis_off()
        if title:
            ax.set_title(title, loc="left", fontsize=10)
        return fig

    # Pick the instance closest to the mean centroid
    mean_c = centroids.mean(axis=0)
    dists = np.linalg.norm(centroids - mean_c, axis=1)
    best_idx = int(np.argmin(dists))
    best_instance = instances[best_idx]
    best_centroid = centroids[best_idx]

    # All cells in the full ROI: show dimmed background in a generous crop
    pad = max(max_r * 2.5, 40.0)
    cx, cy = best_centroid
    x_lo, x_hi = cx - pad, cx + pad
    y_lo, y_hi = cy - pad, cy + pad

    all_x = plot_coords[:, 0]
    all_y = plot_coords[:, 1]
    in_crop = ((all_x >= x_lo) & (all_x <= x_hi) &
               (all_y >= y_lo) & (all_y <= y_hi))
    if in_crop.any():
        ax.scatter(all_x[in_crop], all_y[in_crop],
                   s=22, alpha=0.25, c="#c0c0c0", edgecolors="none", zorder=1)

    # Draw edges connecting motif cells
    motif_pts = plot_coords[list(best_instance), :]
    n_motif = len(best_instance)
    for ii in range(n_motif):
        for jj in range(ii + 1, n_motif):
            ax.plot([motif_pts[ii, 0], motif_pts[jj, 0]],
                    [motif_pts[ii, 1], motif_pts[jj, 1]],
                    color="#2c3e50", lw=1.8, alpha=0.7, zorder=2,
                    linestyle="--")

    # Draw motif cells with per-phenotype colour + labels
    for ci_idx, cell_idx in enumerate(best_instance):
        ph = str(pheno_arr[cell_idx])
        col = color_mapping.get(ph, "#555555")
        ax.scatter(plot_coords[cell_idx, 0], plot_coords[cell_idx, 1],
                   s=180, c=col, edgecolors="#2c3e50", linewidths=1.4,
                   zorder=5)
        ax.annotate(
            ph,
            xy=(plot_coords[cell_idx, 0], plot_coords[cell_idx, 1]),
            xytext=(plot_coords[cell_idx, 0], plot_coords[cell_idx, 1] - max(max_r * 0.18, 6.0)),
            fontsize=7.5, ha="center", va="top", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"),
            zorder=6,
        )

    # Draw interaction-radius circle around the centroid
    circle = plt.Circle((cx, cy), max_r, color="#2980b9",
                         fill=False, linestyle=":", lw=1.0, alpha=0.5, zorder=3)
    ax.add_patch(circle)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)  # flipped y for image coords
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    ax.text(0.99, 0.01, f"n motifs in ROI = {n_found}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    return fig


def plot_motif_null_histogram(
    zone_df: pd.DataFrame,
    config: dict[str, Any],
    target_phenotypes: list[str],
    n_iter: int = 500,
    seed: int = 20260407,
    title: str = "",
) -> Figure:
    """Histogram of permutation null distribution vs. observed count.

    Runs a compact (n_iter, default 500) permutation directly on the ROI zone
    DataFrame to build the null distribution for the specific target motif.
    The observed count is shown as a red vertical line; the Z-score and
    empirical p-value are annotated.

    This is a standalone display-only permutation — it does NOT replace the
    full vectorised pipeline permutation used for feature scoring.
    """
    screening = config["screening"]
    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])
    phenotype_col = str(config["columns"]["phenotype"])
    use_micron_per_pixel = bool(screening.get("use_micron_per_pixel", True))
    micron_per_pixel = max(_safe_float(screening.get("micron_per_pixel"), 0.5083), 1e-6) if use_micron_per_pixel else 1.0

    coords_px = zone_df[[x_col, y_col]].to_numpy(dtype=float)
    pheno_arr = zone_df[phenotype_col].astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))

    # ── Build adjacency (same adaptive-radius logic) ─────────────────────────
    use_adaptive = bool(screening.get("use_adaptive_interaction_radius", True))
    _direct_nn_hist = bool(screening.get("adaptive_radius_direct_neighbours_only", False))
    if use_adaptive:
        adap_min, adap_max = _compute_roi_adaptive_radius(
            coords_px,
            _safe_float(screening.get("adaptive_radius_min_pct"), 0.1),
            _safe_float(screening.get("adaptive_radius_max_pct"), 99.0),
            max(50, _safe_int(screening.get("adaptive_radius_max_sample"), 500)),
            rng,
            direct_neighbours_only=_direct_nn_hist,
        )
        if np.isfinite(adap_max) and adap_max > 0:
            min_px, max_px = adap_min, adap_max
        else:
            min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
            max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel
    else:
        min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
        max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel

    adj = _build_adjacency(coords_px, min_px, max_px)
    edges = _build_edge_array(adj)
    triangles = _find_triangle_array(adj)

    target_sorted = tuple(sorted(str(p) for p in target_phenotypes))
    n_target = len(target_sorted)

    def _count_target(ph: np.ndarray) -> int:
        if n_target == 2 and edges.shape[0] > 0:
            p0, p1 = target_sorted
            left = ph[edges[:, 0]]
            right = ph[edges[:, 1]]
            mask = (
                ((left == p0) & (right == p1)) |
                ((left == p1) & (right == p0))
            )
            return int(mask.sum())
        elif n_target == 3 and triangles.shape[0] > 0:
            p0, p1, p2 = target_sorted
            ta = ph[triangles[:, 0]]
            tb = ph[triangles[:, 1]]
            tc = ph[triangles[:, 2]]
            count = 0
            for r in range(triangles.shape[0]):
                key = tuple(sorted([str(ta[r]), str(tb[r]), str(tc[r])]))
                if key == (p0, p1, p2):
                    count += 1
            return count
        return 0

    # Observed count
    obs_count = _count_target(pheno_arr)

    # Null distribution: shuffle phenotype labels, keep coords fixed
    null_counts: list[int] = []
    perm = pheno_arr.copy()
    for _ in range(int(n_iter)):
        rng.shuffle(perm)
        null_counts.append(_count_target(perm))
    null_arr = np.array(null_counts, dtype=float)
    mean_null = float(np.mean(null_arr))
    std_null = float(np.std(null_arr, ddof=1)) if n_iter > 1 else 0.0

    # Z-score and empirical p-value (one-sided, enrichment)
    if std_null > 1e-9:
        z_score = (obs_count - mean_null) / std_null
    else:
        z_score = float("nan")
    empirical_p = float(np.mean(null_arr >= obs_count))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    n_bins = max(10, min(40, int(np.sqrt(n_iter))))
    ax.hist(null_arr, bins=n_bins, color="#2980b9", alpha=0.65,
            edgecolor="white", linewidth=0.4, label="Null distribution")

    ax.axvline(obs_count, color="#c0392b", lw=2.2, linestyle="--",
               label=f"Observed (n = {obs_count})")
    ax.axvline(mean_null, color="#7f8c8d", lw=1.2, linestyle=":",
               label=f"Null mean ({mean_null:.1f})")

    y_top = ax.get_ylim()[1]
    z_text = f"Z = {z_score:.2f}" if np.isfinite(z_score) else "Z = N/A"
    sig_text = "significant (Z > 1.96)" if (np.isfinite(z_score) and z_score > 1.96) else "not significant"
    p_text = f"empirical p = {empirical_p:.3f}"
    ax.text(
        obs_count, y_top * 0.97,
        f" {z_text}  {p_text}\n ({sig_text})",
        ha="left" if obs_count >= mean_null else "right",
        va="top", fontsize=8.5, color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#c0392b"),
    )

    motif_label = " ↔ ".join(target_sorted) if n_target == 2 else " + ".join(target_sorted)
    ax.set_xlabel(f"Number of {motif_label} motifs after phenotype shuffle", fontsize=9)
    ax.set_ylabel(f"Count (of {n_iter} permutations)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    else:
        ax.set_title(f"Permutation null model: {motif_label}", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_triad_patient_dotplot(
    patient_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    triad_label: str = "",
    zone_label: str = "",
    endpoint_label: str = "",
    split_method: str = "median",
    custom_cutpoint: float | None = None,
    show_events: bool = True,
    figsize: tuple[float, float] = (5.0, 5.0),
    seed: int = 20260407,
) -> tuple[Figure, dict[str, Any]]:
    """One dot per patient, split into the same high/low groups the KM curve uses.

    The KM curve shows *what happened* to the two groups; this shows *what the
    two groups actually are* -- the spread of the underlying triad measurement,
    how far apart the groups really sit, whether the split is driven by a
    handful of extreme patients, and how many patients sit at zero. A split that
    looks decisive on a KM curve but has both groups overlapping here is one to
    distrust.

    Returns ``(figure, summary)`` where ``summary`` carries the group sizes,
    the cutpoint and the group medians for the caption and the export.
    """
    summary: dict[str, Any] = {"status": "empty", "n_high": 0, "n_low": 0, "cutpoint": float("nan")}
    fig, ax = plt.subplots(figsize=figsize)

    if patient_df.empty or metric_col not in patient_df.columns:
        ax.text(0.5, 0.5, "No patient-level data for this triad.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color=NEUTRAL_COLOR)
        ax.set_axis_off()
        return fig, summary

    work = patient_df.copy()
    values = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.loc[values.notna()].copy()
    work["_value"] = values.loc[values.notna()]
    if work.empty:
        ax.text(0.5, 0.5, "No finite values for this triad metric.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color=NEUTRAL_COLOR)
        ax.set_axis_off()
        return fig, summary

    method = str(split_method or "median").lower()
    if method == "custom" and custom_cutpoint is not None and np.isfinite(float(custom_cutpoint)):
        cutpoint = float(custom_cutpoint)
    elif method in {"tertile", "upper_tertile"}:
        cutpoint = float(work["_value"].quantile(2.0 / 3.0))
    elif method in {"quartile", "upper_quartile"}:
        cutpoint = float(work["_value"].quantile(0.75))
    else:
        cutpoint = float(work["_value"].median())

    work["_group"] = np.where(work["_value"] > cutpoint, "High", "Low")
    high = work[work["_group"] == "High"]
    low = work[work["_group"] == "Low"]

    rng = np.random.default_rng(int(seed))
    colors = {"Low": SURVIVAL_PROTECTIVE_COLOR, "High": SURVIVAL_RISK_COLOR}
    for x_pos, (name, group) in enumerate([("Low", low), ("High", high)]):
        if group.empty:
            continue
        jitter = rng.uniform(-0.16, 0.16, size=len(group))
        event_flags = None
        if show_events and "endpoint_censor" in group.columns:
            event_flags = pd.to_numeric(group["endpoint_censor"], errors="coerce").fillna(0.0) > 0
        if event_flags is None:
            ax.scatter(x_pos + jitter, group["_value"], s=46, alpha=0.85,
                       color=colors[name], edgecolor="white", linewidth=0.7, zorder=3)
        else:
            ev = event_flags.to_numpy()
            ax.scatter(x_pos + jitter[ev], group["_value"].to_numpy()[ev], s=52, alpha=0.9,
                       color=colors[name], edgecolor="black", linewidth=0.9, zorder=3, label=None)
            ax.scatter(x_pos + jitter[~ev], group["_value"].to_numpy()[~ev], s=46, alpha=0.55,
                       facecolor="white", edgecolor=colors[name], linewidth=1.2, zorder=3, label=None)
        median_value = float(group["_value"].median())
        ax.plot([x_pos - 0.28, x_pos + 0.28], [median_value, median_value],
                color=colors[name], linewidth=2.4, zorder=4)

    ax.axhline(cutpoint, color=NEUTRAL_COLOR, linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    # Anchored to the axes' left edge rather than to a data x-position, so the
    # label cannot land on top of the High group's points.
    ax.annotate(
        f"cut = {cutpoint:.4g}",
        xy=(0.012, cutpoint), xycoords=("axes fraction", "data"),
        fontsize=8, color=NEUTRAL_COLOR, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.75),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Low\n(n={len(low)})", f"High\n(n={len(high)})"], fontsize=10)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(metric_label or metric_col, fontsize=10)
    title_bits = [bit for bit in [triad_label, zone_label, endpoint_label] if bit]
    if title_bits:
        ax.set_title("\n".join([" + ".join(title_bits[:1]), " | ".join(title_bits[1:])]).strip("\n"),
                     fontsize=11, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if show_events and "endpoint_censor" in work.columns:
        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                   markerfacecolor=NEUTRAL_COLOR, markeredgecolor="black", label="Event"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                   markerfacecolor="white", markeredgecolor=NEUTRAL_COLOR, label="Censored"),
        ]
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")

    n_zero = int((work["_value"] <= 0).sum())
    summary.update({
        "status": "ok",
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "cutpoint": float(cutpoint),
        "median_high": float(high["_value"].median()) if not high.empty else float("nan"),
        "median_low": float(low["_value"].median()) if not low.empty else float("nan"),
        "n_zero_value_patients": n_zero,
        "n_total": int(len(work)),
    })
    fig.tight_layout()
    return fig, summary


def build_motif_enrichment_summary(
    combined_by_width: dict[Any, pd.DataFrame],
    stats_all: pd.DataFrame,
    families: tuple[str, ...] = ("triad", "dyad"),
    enrichment_z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Cross spatial enrichment with outcome association, per motif and width.

    These are two different questions and a motif can answer one without the
    other:

    * **Enrichment** (from the permutation null) asks whether the motif occurs
      more often than random labelling of the same cells on the same graph
      would give -- i.e. is it spatially organised at all.
    * **Outcome association** (from the survival screen) asks whether how much
      of it a patient has tracks their outcome.

    A motif can be strongly organised in every patient and carry no prognostic
    signal, or track outcome while never exceeding chance arrangement. The
    interesting motifs are the ones that do both, and only this joined view
    shows that. Requires permutation testing to have been enabled, since the
    motif feature value is the permutation Z-score.
    """
    if not combined_by_width:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for width_key, frame in combined_by_width.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        if "feature_family" not in frame.columns:
            continue
        subset = frame[frame["feature_family"].astype(str).isin(families)]
        if subset.empty:
            continue
        frames.append(subset)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    combined = combined[np.isfinite(combined["value"])]
    if combined.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["feature_family", "feature_name", "feature_label", "zone_width", "zone_label"]
                  if c in combined.columns]
    # Compute the enriched fraction as the mean of a precomputed boolean column
    # rather than groupby.apply with a Python lambda: apply calls back into
    # Python once per group, and with thousands of motif x width groups that
    # single line dominated the render.
    combined["_is_enriched"] = combined["value"] >= float(enrichment_z_threshold)
    grouped = combined.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        mean_enrichment_z=("value", "mean"),
        median_enrichment_z=("value", "median"),
        n_regions=("value", "size"),
        frac_regions_enriched=("_is_enriched", "mean"),
    ).reset_index()

    # Attach the outcome association already computed by the survival screen.
    if not stats_all.empty and {"feature_name", "zone_width"}.issubset(stats_all.columns):
        keep = [c for c in ["feature_name", "zone_width", "endpoint_label", "p_value", "fdr_q",
                            "significance_label", "hazard_ratio", "direction"] if c in stats_all.columns]
        summary = summary.merge(stats_all[keep], on=["feature_name", "zone_width"], how="left")

    summary["is_spatially_enriched"] = summary["mean_enrichment_z"] >= float(enrichment_z_threshold)
    summary["is_outcome_linked"] = (
        summary["significance_label"].isin(["significant_fdr", "discovery_fdr"])
        if "significance_label" in summary.columns else False
    )
    summary["both"] = summary["is_spatially_enriched"] & summary["is_outcome_linked"]
    sort_cols = [c for c in ["both", "mean_enrichment_z"] if c in summary.columns]
    return summary.sort_values(sort_cols, ascending=False).reset_index(drop=True)


def summarize_distance_distribution(df: pd.DataFrame, distance_col: str) -> pd.DataFrame:
    distances = pd.to_numeric(df[distance_col], errors="coerce")
    distances = distances[np.isfinite(distances)]
    if distances.empty:
        return pd.DataFrame(columns=["statistic", "value"])

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    rows = [
        {"statistic": "count", "value": float(distances.shape[0])},
        {"statistic": "min", "value": float(distances.min())},
        {"statistic": "max", "value": float(distances.max())},
        {"statistic": "mean", "value": float(distances.mean())},
        {"statistic": "median", "value": float(distances.median())},
    ]
    for percentile in percentiles:
        rows.append(
            {
                "statistic": f"p{percentile}",
                "value": float(np.percentile(distances.to_numpy(dtype=float), percentile)),
            }
        )
    return pd.DataFrame(rows)


def build_distance_histogram(df: pd.DataFrame, distance_col: str, bins: int = 30) -> pd.DataFrame:
    distances = pd.to_numeric(df[distance_col], errors="coerce")
    distances = distances[np.isfinite(distances)]
    if distances.empty:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"])

    values = distances.to_numpy(dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"])
    if abs(upper - lower) < 1e-9:
        padding = max(abs(lower) * 0.05, 0.5)
        edges = np.array([lower - padding, upper + padding], dtype=float)
    else:
        bin_count = max(10, min(int(bins), 60))
        edges = np.linspace(lower, upper, num=bin_count + 1, dtype=float)
    counts, bin_edges = np.histogram(values, bins=edges)
    return pd.DataFrame(
        {
            "bin_left": bin_edges[:-1].astype(float),
            "bin_right": bin_edges[1:].astype(float),
            "bin_center": ((bin_edges[:-1] + bin_edges[1:]) / 2.0).astype(float),
            "count": counts.astype(int),
        }
    )


def estimate_interaction_distance(df: pd.DataFrame, config: dict[str, Any], log_fn: LogFn = None) -> tuple[float, str]:
    columns = config["columns"]
    x_col = str(columns["x"])
    y_col = str(columns["y"])
    batch_col = str(columns["batch"])
    roi_col = str(columns["roi"])
    screening = config["screening"]
    percentile = _safe_float(screening.get("auto_interaction_percentile", 95.0), 95.0)
    use_micron_per_pixel = bool(screening.get("use_micron_per_pixel", True))
    micron_per_pixel = max(_safe_float(screening.get("micron_per_pixel"), 0.5083), 1e-6) if use_micron_per_pixel else 1.0
    sample_per_region = max(100, _safe_int(screening.get("auto_interaction_sample_per_region"), 2000))
    rng = np.random.default_rng(_safe_int(screening.get("bootstrap_random_seed"), 20260407))

    nn_distances: list[np.ndarray] = []
    for _, group_df in _iter_regions(df, batch_col, roi_col):
        coords = group_df[[x_col, y_col]].to_numpy(dtype=float)
        if len(coords) < 2:
            continue
        if len(coords) > sample_per_region:
            sample_idx = rng.choice(len(coords), size=sample_per_region, replace=False)
            coords = coords[sample_idx]
        distances, _ = cKDTree(coords).query(coords, k=2)
        nearest = np.asarray(distances)[:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if nearest.size > 0:
            nn_distances.append(nearest)

    if not nn_distances:
        return float("nan"), "um" if use_micron_per_pixel else "px"

    pooled = np.concatenate(nn_distances)
    threshold_px = float(np.percentile(pooled, percentile))
    if use_micron_per_pixel:
        threshold = float(threshold_px * micron_per_pixel)
        unit = "um"
    else:
        threshold = float(threshold_px)
        unit = "px"
    _log(log_fn, f"Auto interaction distance estimated at {threshold:.2f} {unit} using the {percentile:.0f}th percentile of within-region nearest-neighbor spacing")
    return threshold, unit


def build_candidate_widths(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[float], float, pd.DataFrame, pd.DataFrame]:
    distance_col = str(config["columns"]["distance"])
    distances = pd.to_numeric(df[distance_col], errors="coerce")
    exclude_negative = bool(config["screening"].get("exclude_negative_distances", True))
    if exclude_negative:
        distances = distances[distances >= 0]
    distances = distances[np.isfinite(distances)]
    if distances.empty:
        return [], float("nan"), pd.DataFrame(columns=["statistic", "value"]), pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"])

    outer_mode = str(config["screening"].get("outer_cap_mode", "percentile"))
    max_width = _safe_float(config["screening"].get("max_width"), 0.0)

    if outer_mode == "value":
        outer_cap_value = abs(_safe_float(config["screening"].get("outer_cap_value"), float(np.max(np.abs(distances.to_numpy(dtype=float))))))
        if outer_cap_value > 0:
            distances = distances[np.abs(distances) <= float(outer_cap_value)]
            distances = distances[np.isfinite(distances)]
    if max_width > 0:
        distances = distances[np.abs(distances) <= float(max_width)]
        distances = distances[np.isfinite(distances)]
    if distances.empty:
        return [], float("nan"), pd.DataFrame(columns=["statistic", "value"]), pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"])

    n_candidate_widths = max(1, _safe_int(config["screening"].get("n_candidate_widths"), 12))
    round_width_to = max(0.1, _safe_float(config["screening"].get("round_width_to"), 5.0))
    inner_percentile = _safe_float(config["screening"].get("inner_cap_percentile", 5.0), 5.0)
    outer_percentile = _safe_float(config["screening"].get("outer_cap_percentile", 95.0), 95.0)

    if exclude_negative:
        if outer_mode == "value":
            outer_cap = float(distances.max()) if distances.size else float("nan")
        else:
            outer_cap = float(np.percentile(distances.to_numpy(dtype=float), outer_percentile))
        start_width = max(round_width_to, float(np.percentile(distances.to_numpy(dtype=float), inner_percentile)))
        raw_widths = np.linspace(start_width, outer_cap, n_candidate_widths)
        candidate_widths = sorted({float(round(value / round_width_to) * round_width_to) for value in raw_widths if value > 0})
        candidate_widths = [value for value in candidate_widths if value > 0]
        scan_cap = float(outer_cap)
    else:
        lower_percentile = max(0.0, min(float(inner_percentile), 50.0))
        upper_percentile = min(100.0, max(float(outer_percentile), 50.0))
        if upper_percentile <= lower_percentile:
            lower_percentile = 5.0
            upper_percentile = 95.0
        quantile_levels = np.linspace(lower_percentile, upper_percentile, n_candidate_widths)
        raw_widths = np.percentile(distances.to_numpy(dtype=float), quantile_levels)
        candidate_widths = sorted(
            {
                float(round(value / round_width_to) * round_width_to)
                for value in raw_widths
                if np.isfinite(value) and abs(float(value)) >= round_width_to
            }
        )
        scan_cap = float(np.max(np.abs(raw_widths))) if len(raw_widths) > 0 else float(np.max(np.abs(distances.to_numpy(dtype=float))))

    histogram_df = build_distance_histogram(pd.DataFrame({distance_col: distances}), distance_col)
    summary_df = summarize_distance_distribution(pd.DataFrame({distance_col: distances}), distance_col)
    return candidate_widths, float(scan_cap), summary_df, histogram_df


def prepare_analysis_dataframe(
    df: pd.DataFrame,
    config: dict[str, Any],
    log_fn: LogFn = None,
) -> pd.DataFrame:
    columns = config["columns"]
    x_col = str(columns["x"])
    y_col = str(columns["y"])
    phenotype_col = str(columns["phenotype"])
    distance_col = str(columns["distance"])
    patient_col = str(columns["patient"])
    batch_col = str(columns.get("batch") or "")
    roi_col = str(columns.get("roi") or "")
    stratify_cols = [str(value) for value in columns.get("stratify", []) if str(value)]

    required_cols = [x_col, y_col, phenotype_col, distance_col, patient_col]
    if batch_col:
        required_cols.append(batch_col)
    if roi_col:
        required_cols.append(roi_col)
    for column in stratify_cols:
        required_cols.append(column)

    for endpoint in config.get("endpoints", []):
        if not endpoint.get("enabled", False):
            continue
        if endpoint.get("type") == "survival":
            required_cols.extend([str(endpoint.get("time_col") or ""), str(endpoint.get("censor_col") or "")])
        elif endpoint.get("type") == "binary":
            if endpoint.get("multi_mode", False):
                for col in endpoint.get("multi_cols", []):
                    if col:
                        required_cols.append(str(col))
            else:
                required_cols.append(str(endpoint.get("group_col") or ""))

    unique_required = [column for column in required_cols if column]
    missing = [column for column in unique_required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    work = df.copy()
    work = work.dropna(subset=[x_col, y_col, phenotype_col, patient_col])
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work[distance_col] = pd.to_numeric(work[distance_col], errors="coerce")
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col]) & np.isfinite(work[distance_col])].copy()

    for column, values in config.get("row_filters", {}).items():
        selected_values = [str(value) for value in values if str(value)]
        if not selected_values or column not in work.columns:
            continue
        work = work[work[column].astype(str).isin(selected_values)].copy()
        _log(log_fn, f"Filter {column}: kept {len(work)} rows")

    context_column = str(columns.get("context_column") or "")
    context_values = [str(value) for value in columns.get("context_values", []) if str(value)]
    if context_column and context_values and context_column in work.columns:
        work = work[work[context_column].astype(str).isin(context_values)].copy()
        _log(log_fn, f"Context filter {context_column}: kept {len(work)} rows")

    selected_phenotypes = [str(value) for value in config.get("selected_phenotypes", []) if str(value)]
    if selected_phenotypes:
        work = work[work[phenotype_col].astype(str).isin(selected_phenotypes)].copy()
        _log(log_fn, f"Phenotype selection: kept {len(work)} rows")

    excluded_phenotypes = [str(value) for value in config.get("excluded_phenotypes", []) if str(value)]
    if excluded_phenotypes:
        work = work[~work[phenotype_col].astype(str).isin(excluded_phenotypes)].copy()
        _log(log_fn, f"Phenotype exclusion: kept {len(work)} rows")

    if bool(config["screening"].get("exclude_negative_distances", True)):
        work = work[work[distance_col] >= 0].copy()

    if not batch_col:
        work["__tab8_fs_batch__"] = "ALL"
        batch_col = "__tab8_fs_batch__"
        columns["batch"] = batch_col
    if not roi_col:
        work["__tab8_fs_roi__"] = "ALL"
        roi_col = "__tab8_fs_roi__"
        columns["roi"] = roi_col

    for stratify_col in stratify_cols:
        work[stratify_col] = work[stratify_col].astype(str).fillna("NA")
    work[patient_col] = work[patient_col].astype(str)
    work[phenotype_col] = work[phenotype_col].astype(str)
    work["__stratum_label__"] = work.apply(lambda row: _build_stratum_label(row, stratify_cols), axis=1)
    work["__batch_value__"] = work[batch_col].astype(str)
    work["__roi_value__"] = work[roi_col].astype(str)
    work["__patient_value__"] = work[patient_col].astype(str)
    return work


def _feature_base_record(
    group_df: pd.DataFrame,
    config: dict[str, Any],
    width: float,
    area_mm2: float,
    zone_label: str | None = None,
    zone_start: float | None = None,
    zone_end: float | None = None,
) -> dict[str, Any]:
    columns = config["columns"]
    base: dict[str, Any] = {
        "patient": str(_first_valid(group_df[columns["patient"]]) or "NA"),
        "batch": str(_first_valid(group_df[columns["batch"]]) or "NA"),
        "roi": str(_first_valid(group_df[columns["roi"]]) or "NA"),
        "stratum_label": str(_first_valid(group_df["__stratum_label__"]) or "All"),
        "zone_width": float(width),
        "zone_label": str(zone_label or format_zone_label(width)),
        "n_zone_cells": int(len(group_df)),
        "area_mm2": float(area_mm2) if np.isfinite(area_mm2) else float("nan"),
    }
    if zone_start is not None and np.isfinite(float(zone_start)):
        base["zone_start"] = float(zone_start)
    if zone_end is not None and np.isfinite(float(zone_end)):
        base["zone_end"] = float(zone_end)
    for stratify_col in columns.get("stratify", []):
        if stratify_col in group_df.columns:
            base[str(stratify_col)] = str(_first_valid(group_df[stratify_col]) or "NA")
    return base


def compute_all_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    width: float,
    zone_start: float | None = None,
    zone_end: float | None = None,
    log_fn: LogFn = None,
) -> pd.DataFrame:
    columns = config["columns"]
    phenotype_col = str(columns["phenotype"])
    distance_col = str(columns["distance"])
    x_col = str(columns["x"])
    y_col = str(columns["y"])
    batch_col = str(columns["batch"])
    roi_col = str(columns["roi"])

    enabled_families = set(str(value) for value in config.get("feature_families", []))
    phenotypes = [str(value) for value in config.get("selected_phenotypes", []) if str(value)]
    if not phenotypes:
        phenotypes = _safe_unique(df[phenotype_col])

    screening = config["screening"]
    use_micron_per_pixel = bool(screening.get("use_micron_per_pixel", True))
    micron_per_pixel = max(_safe_float(screening.get("micron_per_pixel"), 0.5), 1e-6) if use_micron_per_pixel else 1.0
    min_cells_per_zone = max(1, _safe_int(screening.get("min_cells_per_zone"), 20))
    min_cells_for_feature = max(1, _safe_int(screening.get("min_cells_for_feature"), 10))
    ratio_pseudocount = _safe_float(screening.get("ratio_pseudocount"), 1.0)
    log_transform_ratios = bool(screening.get("log_transform_ratios", True))
    k_neighbors = max(1, _safe_int(screening.get("nn_k"), 3))
    dyad_min_px = _safe_float(screening.get("dyad_min_dist_um"), 0.0) / micron_per_pixel
    dyad_max_px = _safe_float(screening.get("dyad_max_dist_um"), 30.0) / micron_per_pixel
    triad_min_px = _safe_float(screening.get("triad_min_dist_um"), 0.0) / micron_per_pixel
    triad_max_px = _safe_float(screening.get("triad_max_dist_um"), 30.0) / micron_per_pixel
    triad_mode = str(screening.get("triad_mode", "all"))
    allow_self_interaction_features = bool(screening.get("allow_self_interaction_features", False))
    area_use_median = bool(screening.get("area_use_median", False))
    area_percentile = _safe_float(screening.get("area_percentile", 95.0), 95.0)
    min_hole_size_px2 = _safe_float(screening.get("min_hole_size_px2", 0.0), 0.0)

    # ── Permutation-based significance for dyad / triad features ──────────────
    use_adaptive_radius = bool(screening.get("use_adaptive_interaction_radius", True))
    adaptive_direct_neighbours_only = bool(screening.get("adaptive_radius_direct_neighbours_only", True))
    adaptive_min_pct = _safe_float(screening.get("adaptive_radius_min_pct", 0.1), 0.1)
    adaptive_max_pct = _safe_float(screening.get("adaptive_radius_max_pct", 99.0), 99.0)
    adaptive_max_sample = max(50, _safe_int(screening.get("adaptive_radius_max_sample", 500), 500))
    permutation_n_iter = max(0, _safe_int(screening.get("dyad_triad_permutation_n_iter", 1000), 1000))
    min_obs_motif_count = max(0, _safe_int(screening.get("min_obs_motif_count", 3), 3))
    perm_seed = _safe_int(screening.get("bootstrap_random_seed"), 20260407)
    perm_chunk_size = max(8, _safe_int(screening.get("permutation_chunk_size", 64), 64))
    permutation_use_gpu = bool(screening.get("permutation_use_gpu", False))
    triad_null_mode = str(screening.get("triad_null_mode", "random_label")).strip().lower()
    if triad_null_mode not in {"random_label", "pairwise_preserving"}:
        triad_null_mode = "random_label"
    pp_samples = max(20, _safe_int(screening.get("pairwise_null_samples", 200), 200))
    pp_sweeps = max(1, _safe_int(screening.get("pairwise_null_sweeps_per_sample", 3), 3))
    pp_burn = max(1, _safe_int(screening.get("pairwise_null_burn_in_sweeps", 15), 15))
    pp_temp = max(1e-6, _safe_float(screening.get("pairwise_null_temperature", 2.0), 2.0))
    # n_jobs: number of threads for ROI-level parallelism.  0 = auto (all CPUs).
    _raw_n_jobs = _safe_int(screening.get("n_jobs", 1), 1)
    if _raw_n_jobs <= 0:
        import os as _os
        n_jobs = _os.cpu_count() or 1
    else:
        n_jobs = int(_raw_n_jobs)

    triad_defs: set[tuple[str, str, str]] = set()
    if triad_mode == "all":
        for idx_a, left in enumerate(phenotypes):
            for idx_b, middle in enumerate(phenotypes[idx_a:], start=idx_a):
                for right in phenotypes[idx_b:]:
                    if not allow_self_interaction_features and len({str(left), str(middle), str(right)}) < 3:
                        continue
                    sorted_triad = sorted((str(left), str(middle), str(right)))
                    triad_defs.add((sorted_triad[0], sorted_triad[1], sorted_triad[2]))
    else:
        for item in screening.get("triad_definitions", []):
            if len(item) == 3:
                if not allow_self_interaction_features and len({str(item[0]), str(item[1]), str(item[2])}) < 3:
                    continue
                sorted_triad = sorted([str(item[0]), str(item[1]), str(item[2])])
                triad_defs.add((sorted_triad[0], sorted_triad[1], sorted_triad[2]))

    interval_start_value = float(zone_start) if zone_start is not None and np.isfinite(float(zone_start)) else None
    interval_end_value = float(zone_end) if zone_end is not None and np.isfinite(float(zone_end)) else None
    interval_mode = interval_start_value is not None and interval_end_value is not None
    interval_start_bound = 0.0
    interval_end_bound = float(width)
    if interval_mode:
        assert interval_start_value is not None and interval_end_value is not None
        interval_start_bound = interval_start_value
        interval_end_bound = interval_end_value
        base_zone_width = interval_end_bound
        base_zone_label = format_zone_interval_label(interval_start_bound, interval_end_bound)
    else:
        base_zone_width = float(width)
        base_zone_label = format_zone_label(width)

    # ── Per-ROI worker (closes over all scalar params above) ─────────────────
    def _process_roi(zone_df: pd.DataFrame, roi_seed: int) -> list[dict[str, Any]]:
        """Compute all features for one zone DataFrame.

        Each ROI gets its own RNG derived from ``roi_seed`` so that:
        * results are fully reproducible regardless of execution order, and
        * concurrent threads never share state.
        """
        _rng = np.random.default_rng(roi_seed)
        _rec: list[dict[str, Any]] = []

        _coords = zone_df[[x_col, y_col]].to_numpy(dtype=float)
        _area_mm2, _buf_dist = _compute_virtual_area_mm2(
            coords=_coords,
            micron_per_pixel=micron_per_pixel,
            hole_size_px2=min_hole_size_px2,
            use_median=area_use_median,
            percentile=area_percentile,
        )
        _base = _feature_base_record(
            zone_df,
            config,
            base_zone_width,
            _area_mm2,
            zone_label=base_zone_label,
            zone_start=interval_start_bound if interval_mode else None,
            zone_end=interval_end_bound if interval_mode else None,
        )
        _base["buffer_distance_px"] = float(_buf_dist) if np.isfinite(_buf_dist) else float("nan")

        _pheno_vals = zone_df[phenotype_col].astype(str)
        _counts = _pheno_vals.value_counts()
        _total = max(1, int(len(zone_df)))

        if "density" in enabled_families:
            for _ph in phenotypes:
                _cnt = int(_counts.get(_ph, 0))
                if _cnt < min_cells_for_feature:
                    continue
                _val = float(_cnt) / _area_mm2 if np.isfinite(_area_mm2) and _area_mm2 > 0 else float("nan")
                _rec.append({**_base, "feature_family": "density",
                              "feature_name": f"density::{_ph}",
                              "feature_label": f"Density | {_ph}", "value": _val})

        if "proportion" in enabled_families:
            for _ph in phenotypes:
                _cnt = int(_counts.get(_ph, 0))
                if _cnt < min_cells_for_feature:
                    continue
                _rec.append({**_base, "feature_family": "proportion",
                              "feature_name": f"proportion::{_ph}",
                              "feature_label": f"Proportion | {_ph}",
                              "value": float(_cnt) / float(_total)})

        if "ratio" in enabled_families:
            for _il, _l in enumerate(phenotypes):
                _lc = int(_counts.get(_l, 0))
                if _lc < min_cells_for_feature:
                    continue
                for _r in phenotypes[_il + 1:]:
                    _rc = int(_counts.get(_r, 0))
                    if _rc < min_cells_for_feature:
                        continue
                    _rv = (_lc + ratio_pseudocount) / (_rc + ratio_pseudocount)
                    if log_transform_ratios:
                        _rv = float(np.log2(_rv))
                    _ratio_label = f"Log2 Ratio | {_l}/{_r}" if log_transform_ratios else f"Ratio | {_l}/{_r}"
                    _rec.append({**_base, "feature_family": "ratio",
                                  "feature_name": f"ratio::{_l}::{_r}",
                                  "feature_label": _ratio_label,
                                  "value": float(_rv)})

        # ── Dyad / Triad ─────────────────────────────────────────────────────
        _need_adj = "dyad" in enabled_families or "triad" in enabled_families
        _adj: list[set[int]] | None = None
        _ph_arr = _pheno_vals.to_numpy(dtype=str)
        _dyad_z: dict[tuple[str, str], float] = {}
        _triad_z: dict[tuple[str, str, str], float] = {}
        if _need_adj:
            if use_adaptive_radius:
                _adap_min, _adap_max = _compute_roi_adaptive_radius(
                    _coords, adaptive_min_pct, adaptive_max_pct,
                    adaptive_max_sample, _rng,
                    direct_neighbours_only=adaptive_direct_neighbours_only,
                )
                if np.isfinite(_adap_max) and _adap_max > 0:
                    _min_px, _max_px = _adap_min, _adap_max
                else:
                    _min_px = min(dyad_min_px, triad_min_px)
                    _max_px = max(dyad_max_px, triad_max_px)
            else:
                _min_px = min(dyad_min_px, triad_min_px)
                _max_px = max(dyad_max_px, triad_max_px)
            _adj = _build_adjacency(_coords, _min_px, _max_px)

            if permutation_n_iter > 0:
                _edges_arr = _build_edge_array(_adj)
                _tri_arr = _find_triangle_array(_adj)
                _dyad_z, _triad_z = _permutation_zscore_dyad_triad(
                    _ph_arr, _edges_arr, _tri_arr,
                    permutation_n_iter, _rng, perm_chunk_size,
                    min_obs_count=min_obs_motif_count,
                    use_gpu=permutation_use_gpu,
                )
                # Optionally replace the triad Z-scores with the stricter,
                # dyad-constrained null. Dyad Z-scores are left untouched: for a
                # pair there is no lower-order structure to hold fixed, so the
                # random-label null is already the right question there.
                if triad_null_mode == "pairwise_preserving" and _tri_arr.shape[0] > 0:
                    _pp_z, _pp_diag = pairwise_preserving_triad_zscores(
                        _ph_arr, _edges_arr, _tri_arr,
                        n_samples=pp_samples,
                        sweeps_per_sample=pp_sweeps,
                        burn_in_sweeps=pp_burn,
                        temperature=pp_temp,
                        seed=int(roi_seed),
                        min_obs_count=min_obs_motif_count,
                    )
                    if _pp_diag.get("constraint_held", False):
                        _triad_z = _pp_z
                    else:
                        # The chain failed to hold the dyad composition, so its
                        # Z-scores do not mean what the mode claims. Drop the
                        # triads for this region rather than silently reporting
                        # numbers from a null that did not do its job.
                        _triad_z = {}

        if "dyad" in enabled_families and _adj is not None:
            _use_perm = permutation_n_iter > 0 and bool(_dyad_z)
            _tot_edges = sum(len(v) for v in _adj) / 2.0
            for _il, _l in enumerate(phenotypes):
                if int(_counts.get(_l, 0)) < min_cells_for_feature:
                    continue
                for _r in phenotypes[_il:]:
                    if not allow_self_interaction_features and _l == _r:
                        continue
                    if int(_counts.get(_r, 0)) < min_cells_for_feature:
                        continue
                    if _use_perm:
                        _pk: tuple[str, str] = tuple(sorted([_l, _r]))  # type: ignore[assignment]
                        _val = _dyad_z.get(_pk, float("nan"))
                        if not np.isfinite(_val):
                            continue
                    else:
                        _dc = _count_dyads(_ph_arr, _adj, _l, _r)
                        _val = float(_dc) / float(_tot_edges) if _tot_edges > 0 else float("nan")
                    _rec.append({**_base, "feature_family": "dyad",
                                  "feature_name": f"dyad::{_l}::{_r}",
                                  "feature_label": f"Dyad | {_l} <-> {_r}",
                                  "value": _val})

        if "triad" in enabled_families and _adj is not None:
            _use_perm_t = permutation_n_iter > 0 and bool(_triad_z)
            if not _use_perm_t:
                _tc_raw = _collect_unordered_triads(_ph_arr, _adj)
                _tot_tri = sum(_tc_raw.values())
            for _td in sorted(triad_defs):
                if _use_perm_t:
                    _ptk: tuple[str, str, str] = tuple(sorted(_td))  # type: ignore[assignment]
                    _val = _triad_z.get(_ptk, float("nan"))
                    if not np.isfinite(_val):
                        continue
                else:
                    _tct = int(_tc_raw.get(_td, 0))
                    if _tct <= 0:
                        continue
                    _val = float(_tct) / float(_tot_tri) if _tot_tri > 0 else float("nan")
                _rec.append({**_base, "feature_family": "triad",
                              "feature_name": f"triad::{_td[0]}::{_td[1]}::{_td[2]}",
                              "feature_label": f"Triad | {' + '.join(_td)}",
                              "value": _val})

        if "diversity" in enabled_families:
            _props = np.array([float(_counts.get(_ph, 0)) / float(_total) for _ph in phenotypes], dtype=float)
            _props = _props[_props > 0]
            if _props.size > 0:
                _shannon = float(-(_props * np.log(_props)).sum())
                _rec.append({**_base, "feature_family": "diversity",
                              "feature_name": "diversity::shannon",
                              "feature_label": "Diversity | Shannon index",
                              "value": _shannon})
                _richness = int(_props.size)
                _rec.append({**_base, "feature_family": "diversity",
                              "feature_name": "diversity::richness",
                              "feature_label": "Diversity | Richness",
                              "value": float(_richness)})
                _simpson = float(1.0 - float((_props ** 2).sum()))
                _rec.append({**_base, "feature_family": "diversity",
                              "feature_name": "diversity::simpson",
                              "feature_label": "Diversity | Simpson index",
                              "value": _simpson})
                if _richness > 1:
                    _evenn = float(_shannon / float(np.log(_richness)))
                    _rec.append({**_base, "feature_family": "diversity",
                                  "feature_name": "diversity::evenness",
                                  "feature_label": "Diversity | Evenness (Pielou J)",
                                  "value": _evenn})

        if "nn_distance" in enabled_families:
            _by_ph = {
                _ph: zone_df.loc[zone_df[phenotype_col].astype(str) == _ph, [x_col, y_col]].to_numpy(dtype=float)
                for _ph in phenotypes
            }
            for _l in phenotypes:
                if len(_by_ph[_l]) < min_cells_for_feature:
                    continue
                for _r in phenotypes:
                    if not allow_self_interaction_features and _l == _r:
                        continue
                    if len(_by_ph[_r]) < min_cells_for_feature:
                        continue
                    _nn_val = _mean_knn_distance(_by_ph[_l], _by_ph[_r], k_neighbors, _l == _r)
                    if not np.isfinite(_nn_val):
                        continue
                    _rec.append({**_base, "feature_family": "nn_distance",
                                  "feature_name": f"nn_distance::{_l}::{_r}",
                                  "feature_label": f"NN distance | {_l} -> {_r}",
                                  "value": float(_nn_val) * micron_per_pixel if use_micron_per_pixel else float(_nn_val)})

        return _rec

    # ── Collect valid zone DataFrames, then execute (sequential or parallel) ──
    valid_zones: list[pd.DataFrame] = []
    for _, _grp_df in _iter_regions(df, batch_col, roi_col):
        if interval_mode:
            _zdf = _zone_interval_subset(_grp_df, distance_col, interval_start_bound, interval_end_bound)
        else:
            _zdf = _zone_subset(_grp_df, distance_col, width)
        if len(_zdf) >= min_cells_per_zone:
            valid_zones.append(_zdf)

    records: list[dict[str, Any]] = []
    if n_jobs == 1 or not valid_zones:
        for _i, _zdf in enumerate(valid_zones):
            records.extend(_process_roi(_zdf, perm_seed + _i))
    else:
        # ThreadPoolExecutor: numpy releases the GIL so threads run truly in
        # parallel even on the same process — no pickling needed.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as _exe:
            _futures = [_exe.submit(_process_roi, _zdf, perm_seed + _i)
                        for _i, _zdf in enumerate(valid_zones)]
            for _fut in _futures:
                records.extend(_fut.result())

    _log(log_fn, f"Computed features for width={width:.2f} across {len(valid_zones)} regions"
                 + (f" using {n_jobs} threads" if n_jobs > 1 else ""))
    import gc as _gc
    _gc.collect()
    return pd.DataFrame(records)


def _aggregate_values(series: pd.Series, method: str) -> float:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return float("nan")
    if method == "mean":
        return float(clean.mean())
    if method == "max":
        return float(clean.max())
    if method == "p75":
        return float(clean.quantile(0.75))
    return float(clean.median())


def aggregate_to_patient(combined_features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if combined_features.empty:
        return pd.DataFrame()

    group_cols = ["patient", "stratum_label", "feature_family", "feature_name", "feature_label", "zone_width", "zone_label"]
    for stratify_col in config["columns"].get("stratify", []):
        if stratify_col in combined_features.columns:
            group_cols.append(str(stratify_col))
    aggregation_method = str(config["screening"].get("patient_aggregation", "median"))

    rows = []
    for group_values, group_df in combined_features.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = dict(zip(group_cols, group_values))
        record["value"] = _aggregate_values(group_df["value"], aggregation_method)
        record["n_regions"] = int(group_df[["batch", "roi"]].drop_duplicates().shape[0])
        record["mean_region_value"] = float(pd.to_numeric(group_df["value"], errors="coerce").mean())
        rows.append(record)
    return pd.DataFrame(rows)


def build_zone_patient_coverage_row(
    analysis_df: pd.DataFrame,
    config: dict[str, Any],
    width: float,
    *,
    patient_outcomes: pd.DataFrame | None = None,
    combined_features: pd.DataFrame | None = None,
    zone_start: float | None = None,
    zone_end: float | None = None,
) -> dict[str, Any]:
    if analysis_df.empty:
        return {}

    columns = config["columns"]
    patient_col = str(columns["patient"])
    batch_col = str(columns["batch"])
    roi_col = str(columns["roi"])
    distance_col = str(columns["distance"])
    min_cells_per_zone = max(1, _safe_int(config["screening"].get("min_cells_per_zone"), 20))

    interval_mode = zone_start is not None and zone_end is not None
    zone_width = float(zone_end) if interval_mode and zone_end is not None else float(width)
    zone_label = (
        format_zone_interval_label(float(zone_start), float(zone_end))
        if interval_mode and zone_start is not None and zone_end is not None
        else format_zone_label(zone_width)
    )

    if isinstance(patient_outcomes, pd.DataFrame) and not patient_outcomes.empty and "patient" in patient_outcomes.columns:
        total_patient_ids = sorted(patient_outcomes["patient"].dropna().astype(str).unique().tolist())
    else:
        total_patient_ids = sorted(analysis_df[patient_col].dropna().astype(str).unique().tolist())

    patient_state: dict[str, dict[str, bool]] = {
        patient_id: {"has_zone_cells": False, "has_valid_zone_roi": False}
        for patient_id in total_patient_ids
    }

    for _, group_df in _iter_regions(analysis_df, batch_col, roi_col):
        patient_id = str(_first_valid(group_df[patient_col]) or "")
        if not patient_id:
            continue
        if interval_mode and zone_start is not None and zone_end is not None:
            zone_df = _zone_interval_subset(group_df, distance_col, float(zone_start), float(zone_end))
        else:
            zone_df = _zone_subset(group_df, distance_col, zone_width)
        if zone_df.empty:
            continue
        patient_state.setdefault(patient_id, {"has_zone_cells": False, "has_valid_zone_roi": False})
        patient_state[patient_id]["has_zone_cells"] = True
        if len(zone_df) >= min_cells_per_zone:
            patient_state[patient_id]["has_valid_zone_roi"] = True

    feature_patient_ids: set[str] = set()
    if isinstance(combined_features, pd.DataFrame) and not combined_features.empty and "patient" in combined_features.columns:
        feature_patient_ids = set(combined_features["patient"].dropna().astype(str).unique().tolist())

    patient_keys = list(patient_state.keys())
    total_patients = int(len(patient_keys))
    n_patients_with_zone_cells = int(sum(1 for state in patient_state.values() if state["has_zone_cells"]))
    n_patients_with_valid_zone_roi = int(sum(1 for state in patient_state.values() if state["has_valid_zone_roi"]))
    n_patients_with_zone = int(sum(1 for patient_id in patient_keys if patient_id in feature_patient_ids))

    n_patients_without_zone_cells = max(0, total_patients - n_patients_with_zone_cells)
    n_patients_below_min_cells_per_zone = max(0, n_patients_with_zone_cells - n_patients_with_valid_zone_roi)
    n_patients_without_computable_features = max(0, n_patients_with_valid_zone_roi - n_patients_with_zone)

    return {
        "zone_width": zone_width,
        "zone_label": zone_label,
        "total_patients": total_patients,
        "n_patients_with_zone_cells": n_patients_with_zone_cells,
        "n_patients_with_valid_zone_roi": n_patients_with_valid_zone_roi,
        "n_patients_with_zone": n_patients_with_zone,
        "patient_coverage_fraction": float(n_patients_with_zone) / float(total_patients) if total_patients > 0 else float("nan"),
        "n_patients_without_zone_cells": n_patients_without_zone_cells,
        "n_patients_below_min_cells_per_zone": n_patients_below_min_cells_per_zone,
        "n_patients_without_computable_features": n_patients_without_computable_features,
        "n_patients_dropped_total": max(0, total_patients - n_patients_with_zone),
    }


def build_patient_outcomes(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    patient_col = str(config["columns"]["patient"])
    stratify_cols = [str(value) for value in config["columns"].get("stratify", []) if str(value)]
    group_cols = [patient_col] + stratify_cols
    rows = []

    prepared_endpoints = _prepare_enabled_endpoints(config)

    for group_values, group_df in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        record = {"patient": str(group_values[0]), "stratum_label": _build_stratum_label(group_df.iloc[0], stratify_cols)}
        for index, column in enumerate(stratify_cols, start=1):
            record[column] = str(group_values[index])

        for endpoint in prepared_endpoints:
            label = endpoint["label"]
            if endpoint.get("type") == "survival":
                time_col = str(endpoint.get("time_col") or "")
                censor_col = str(endpoint.get("censor_col") or "")
                if time_col:
                    record[f"{label}__time"] = _first_valid(group_df[time_col])
                if censor_col:
                    record[f"{label}__censor"] = _first_valid(group_df[censor_col])
            elif endpoint.get("type") == "binary":
                group_col = str(endpoint.get("group_col") or "")
                if group_col:
                    patient_matches = True
                    for filter_col, filter_val in endpoint.get("filter", {}).items():
                        if str(_first_valid(group_df[filter_col])) != str(filter_val):
                            patient_matches = False
                            break
                    
                    if patient_matches:
                        val = _first_valid(group_df[group_col])
                        if "positive_value" in endpoint and "negative_value" in endpoint:
                            val_str = str(val)
                            pos_str = str(endpoint["positive_value"])
                            neg_str = str(endpoint["negative_value"])
                            if val_str == pos_str:
                                record[f"{label}__group"] = pos_str
                            elif val_str == neg_str:
                                record[f"{label}__group"] = neg_str
                            else:
                                record[f"{label}__group"] = None
                        else:
                            record[f"{label}__group"] = val
                    else:
                        record[f"{label}__group"] = None
        rows.append(record)

    return pd.DataFrame(rows)


def _split_high_low(
    values: pd.Series,
    config: dict[str, Any],
    cutpoint_override: float | None = None,
    survival_df: pd.DataFrame | None = None,
    time_col: str | None = None,
    event_col: str | None = None,
) -> tuple[pd.Series, float | None]:
    """Split a continuous feature into High / Low groups.

    Returns ``(labels, cutpoint)`` where *labels* contains ``"High"``,
    ``"Low"`` or ``NaN``.  The median split uses ``>= median`` for High
    and ``< median`` for Low.  When all values are identical, every
    patient ends up in "High" and the function returns early.
    """
    clean = pd.to_numeric(values, errors="coerce")
    method = str(config["screening"].get("split_method", "median"))
    labels = pd.Series(index=clean.index, dtype="object")

    if clean.dropna().empty:
        return labels, None

    if cutpoint_override is not None and np.isfinite(float(cutpoint_override)):
        cutpoint = float(cutpoint_override)
        labels.loc[clean >= cutpoint] = "High"
        labels.loc[clean < cutpoint] = "Low"
        return labels, float(cutpoint)

    if method == "optimal" and survival_df is not None and time_col and event_col:
        best_p = float("inf")
        best_cut = float(clean.median())
        for q in np.arange(0.20, 0.85, 0.05):
            cut = float(clean.quantile(q))
            high_mask = clean >= cut
            low_mask = clean < cut
            min_patients = max(2, _safe_int(config["screening"].get("min_patients_per_group"), 5))
            if high_mask.sum() < min_patients or low_mask.sum() < min_patients:
                continue
            try:
                test = logrank_test(
                    survival_df.loc[high_mask, time_col],
                    survival_df.loc[low_mask, time_col],
                    event_observed_A=survival_df.loc[high_mask, event_col],
                    event_observed_B=survival_df.loc[low_mask, event_col]
                )
                if test.p_value < best_p:
                    best_p = float(test.p_value)
                    best_cut = cut
            except Exception:
                pass
        labels.loc[clean >= best_cut] = "High"
        labels.loc[clean < best_cut] = "Low"
        return labels, float(best_cut)

    if method == "custom_cutpoint":
        cutpoint = _safe_float(config["screening"].get("custom_cutpoint"), float(clean.median()))
        labels.loc[clean >= cutpoint] = "High"
        labels.loc[clean < cutpoint] = "Low"
        return labels, float(cutpoint)

    if method == "quartiles":
        low_q = _safe_float(config["screening"].get("quantile_low", 0.25), 0.25)
        high_q = _safe_float(config["screening"].get("quantile_high", 0.75), 0.75)
        low_cut = float(clean.quantile(low_q))
        high_cut = float(clean.quantile(high_q))
        labels.loc[clean <= low_cut] = "Low"
        labels.loc[clean >= high_cut] = "High"
        return labels, float(high_cut)

    if method == "tertiles":
        low_cut = float(clean.quantile(1.0 / 3.0))
        high_cut = float(clean.quantile(2.0 / 3.0))
        labels.loc[clean <= low_cut] = "Low"
        labels.loc[clean >= high_cut] = "High"
        return labels, float(high_cut)

    # Default: median split
    cutpoint = float(clean.median())
    labels.loc[clean >= cutpoint] = "High"
    labels.loc[clean < cutpoint] = "Low"

    # --- Balance guard ---
    n_high = int((labels == "High").sum())
    n_low = int((labels == "Low").sum())
    n_total = n_high + n_low
    if n_total > 0 and min(n_high, n_low) / max(n_total, 1) < 0.20:
        _log(None, f"WARNING  median split is highly unbalanced (High={n_high}, Low={n_low}). "
             f"Consider using 'quartiles' or 'tertiles' split_method for features with many ties.")

    return labels, cutpoint


def _cox_hr_and_ci(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    penalizer: float = 0.01,
    log_fn: LogFn = None,
) -> tuple[float, float, float, float, float]:
    """Fit a univariate Cox PH model and return ``(HR, CI_low, CI_high, concordance, ph_test_p)``.

    Uses an L2 penalizer to stabilise convergence on sparse groups.
    Coefficient bounds are clipped to ``[-20, 20]`` on the log-HR scale
    (HR range ≈ 2e-9 to 5e8) to prevent extreme confidence intervals.
    Returns ``(nan, nan, nan, nan, nan)`` on failure.
    """
    model_df = df[[time_col, event_col, group_col]].dropna().copy()
    if model_df[group_col].nunique() < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    try:
        fitter = CoxPHFitter(penalizer=float(penalizer))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitter.fit(model_df, duration_col=time_col, event_col=event_col)
            for w in caught:
                if "convergence" in str(w.message).lower() or "newton" in str(w.message).lower():
                    _log(log_fn, f"Cox PH convergence warning: {w.message}")
        summary = fitter.summary.loc[group_col]
        coef = float(np.clip(summary["coef"], -20.0, 20.0))
        ci_lo = float(np.clip(summary["coef lower 95%"], -20.0, 20.0))
        ci_hi = float(np.clip(summary["coef upper 95%"], -20.0, 20.0))
        c_index = float(fitter.concordance_index_) if hasattr(fitter, "concordance_index_") else float("nan")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ph_test = proportional_hazard_test(fitter, model_df, time_fit="km")
                ph_p = float(ph_test.p_value.iloc[0])
        except Exception:
            ph_p = float("nan")

        return (
            float(math.exp(coef)),
            float(math.exp(ci_lo)),
            float(math.exp(ci_hi)),
            c_index,
            ph_p,
        )
    except Exception as exc:
        _log(log_fn, f"Cox PH fitting failed: {exc}")
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")


def _screen_survival_feature(feature_df: pd.DataFrame, endpoint: dict[str, Any]) -> dict[str, Any] | None:
    time_col = str(endpoint["_time_key"])
    censor_col = str(endpoint["_censor_key"])
    use_censor_as_censored = bool(endpoint.get("censor_flag_means_censored", True))
    merged = feature_df[["patient", "stratum_label", "value", time_col, censor_col]].dropna().copy()
    if merged.empty:
        return None

    merged[time_col] = pd.to_numeric(merged[time_col], errors="coerce")
    merged[censor_col] = pd.to_numeric(merged[censor_col], errors="coerce")
    merged = merged[np.isfinite(merged[time_col]) & np.isfinite(merged[censor_col])].copy()
    if merged.empty:
        return None

    # --- Validate censor column is binary (0 / 1) ---
    censor_values = merged[censor_col].unique()
    if not set(censor_values).issubset({0, 1, 0.0, 1.0}):
        # Clip non-binary values to {0, 1} and warn
        _log(None, f"WARNING  censor column '{censor_col}' contains non-binary values "
             f"{sorted(set(censor_values))}; clipping to {{0, 1}}.")
        merged[censor_col] = merged[censor_col].clip(lower=0.0, upper=1.0).round().astype(float)

    merged["event"] = (1 - merged[censor_col]).clip(lower=0.0, upper=1.0) if use_censor_as_censored else merged[censor_col].clip(lower=0.0, upper=1.0)

    merged["group"] , cutpoint = _split_high_low(
        merged["value"], 
        endpoint["config"],
        survival_df=merged,
        time_col=time_col,
        event_col="event"
    )
    merged = merged[merged["group"].isin(["High", "Low"])].copy()
    if merged.empty or merged["group"].nunique() < 2:
        return None

    min_patients = max(2, _safe_int(endpoint["config"]["screening"].get("min_patients_per_group"), 5))
    counts = merged["group"].value_counts()
    if int(counts.min()) < min_patients:
        return None

    merged["group_binary"] = (merged["group"] == "High").astype(int)

    high = merged[merged["group"] == "High"]
    low = merged[merged["group"] == "Low"]
    try:
        test = logrank_test(
            high[time_col],
            low[time_col],
            event_observed_A=high["event"],
            event_observed_B=low["event"],
        )
        p_value = float(test.p_value)
    except Exception:
        p_value = float("nan")

    hazard_ratio, ci_low, ci_high, concordance, ph_p = _cox_hr_and_ci(merged, time_col, "event", "group_binary")

    # --- Standardised effect size (rank-biserial correlation) ---
    high_values = pd.to_numeric(high["value"], errors="coerce").dropna()
    low_values = pd.to_numeric(low["value"], errors="coerce").dropna()
    try:
        _u_stat, _u_p = stats.mannwhitneyu(high_values, low_values, alternative="two-sided")
        rank_biserial = 1.0 - (2.0 * float(_u_stat)) / (len(high_values) * len(low_values)) if (len(high_values) * len(low_values)) > 0 else float("nan")
    except Exception:
        rank_biserial = float("nan")

    return {
        "n_patients": int(len(merged)),
        "group_high_n": int(len(high)),
        "group_low_n": int(len(low)),
        "p_value": p_value,
        "effect_size": float(high_values.median() - low_values.median()),
        "standardized_effect_size": float(rank_biserial),
        "hazard_ratio": hazard_ratio,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "concordance_index": concordance,
        "ph_test_p": ph_p,
        "cutpoint": cutpoint,
        "endpoint_mode": "survival",
    }


def _screen_binary_feature(feature_df: pd.DataFrame, endpoint: dict[str, Any]) -> dict[str, Any] | None:
    group_col = str(endpoint["_group_key"])
    merged = feature_df[["patient", "stratum_label", "value", group_col]].dropna().copy()
    if merged.empty:
        return None

    positive_value = str(endpoint.get("positive_value") or "")
    if positive_value:
        merged["group"] = np.where(merged[group_col].astype(str) == positive_value, "Positive", "Negative")
    else:
        unique_values = _safe_unique(merged[group_col])
        if len(unique_values) != 2:
            return None
        merged["group"] = merged[group_col].astype(str)

    counts = merged["group"].value_counts()
    if counts.shape[0] < 2:
        return None
    min_patients = max(2, _safe_int(endpoint["config"]["screening"].get("min_patients_per_group"), 5))
    if int(counts.min()) < min_patients:
        return None

    groups = list(counts.index)
    left = merged[merged["group"] == groups[0]]["value"].astype(float)
    right = merged[merged["group"] == groups[1]]["value"].astype(float)
    if left.empty or right.empty:
        return None

    try:
        _stat, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
        p_value = float(p_value)
    except Exception:
        _stat = float("nan")
        p_value = float("nan")

    # --- Standardised effect size (rank-biserial correlation) ---
    try:
        if np.isfinite(float(_stat)) and (len(left) * len(right)) > 0:
            rank_biserial = 1.0 - (2.0 * float(_stat)) / (len(left) * len(right))
        else:
            rank_biserial = float("nan")
    except Exception:
        rank_biserial = float("nan")

    group_binary = (merged["group"] == groups[0]).astype(int)
    try:
        auc = float(roc_auc_score(group_binary, merged["value"].astype(float)))
        auc = max(auc, 1.0 - auc)  # normalise to [0.5, 1] — AUC is symmetric
    except Exception:
        auc = float("nan")

    return {
        "n_patients": int(len(merged)),
        "group_high_n": int(counts.iloc[0]),
        "group_low_n": int(counts.iloc[1]),
        "p_value": p_value,
        "effect_size": float(left.median() - right.median()),
        "standardized_effect_size": float(rank_biserial),
        "hazard_ratio": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "concordance_index": float("nan"),
        "ph_test_p": float("nan"),
        "cutpoint": None,
        "auc": auc,
        "positive_group": str(groups[0]),
        "negative_group": str(groups[1]),
        "endpoint_mode": "binary",
    }


def screen_patient_features(
    patient_features: pd.DataFrame,
    patient_outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if patient_features.empty or patient_outcomes.empty:
        return pd.DataFrame()

    merged = patient_features.merge(patient_outcomes, on=["patient", "stratum_label"] + [col for col in config["columns"].get("stratify", []) if col in patient_outcomes.columns], how="left")
    if merged.empty:
        return pd.DataFrame()
    return _screen_patient_features_from_merged(merged, config)


def run_feature_validation_pipeline(
    raw_df: pd.DataFrame,
    config: dict[str, Any],
    width: float,
    zone_start: float | None = None,
    zone_end: float | None = None,
    log_fn: LogFn = None,
) -> dict[str, Any]:
    """Run a targeted validation at one fixed zone width.

    This reuses the exact same preprocessing, feature construction, patient
    aggregation, and endpoint screening code as the full discovery scan, but it
    evaluates only a single user-chosen width.
    """
    target_width = _safe_float(width, float("nan"))
    explicit_interval_mode = zone_start is not None and zone_end is not None
    if explicit_interval_mode:
        target_start = _safe_float(zone_start, float("nan"))
        target_end = _safe_float(zone_end, float("nan"))
        if not np.isfinite(target_start) or not np.isfinite(target_end) or float(target_start) == float(target_end):
            raise ValueError("Validation interval must have finite start and end values with different boundaries.")
        interval_label = format_zone_interval_label(target_start, target_end)
        validation_zone_start = float(target_start)
        validation_zone_end = float(target_end)
        validation_width_value = float(target_end)
    else:
        if not np.isfinite(target_width) or abs(float(target_width)) <= 0.0:
            raise ValueError("Validation width must be a finite non-zero value.")
        interval_label = format_zone_label(float(target_width))
        validation_zone_start = float(min(0.0, float(target_width)))
        validation_zone_end = float(max(0.0, float(target_width)))
        validation_width_value = float(target_width)

    for endpoint in config.get("endpoints", []):
        if endpoint.get("type") == "binary" and endpoint.get("multi_mode", False):
            multi_cols = [str(c) for c in endpoint.get("multi_cols", []) if c]
            endpoint["multi_col_values"] = {}
            for col in multi_cols:
                if col in raw_df.columns:
                    vals = [v for v in _safe_unique(raw_df[col]) if v not in {None, "", "nan", "NaN"}]
                    endpoint["multi_col_values"][col] = vals

    analysis_df = prepare_analysis_dataframe(raw_df, config, log_fn=log_fn)
    if analysis_df.empty:
        raise ValueError("No rows remain after applying the selected filters.")

    if bool(config["screening"].get("auto_interaction_distance", True)):
        auto_distance_value, auto_distance_unit = estimate_interaction_distance(analysis_df, config, log_fn=log_fn)
        if np.isfinite(auto_distance_value):
            config["screening"]["dyad_min_dist_um"] = 0.0
            config["screening"]["triad_min_dist_um"] = 0.0
            config["screening"]["dyad_max_dist_um"] = float(auto_distance_value)
            config["screening"]["triad_max_dist_um"] = float(auto_distance_value)
            config["screening"]["auto_interaction_distance_value"] = float(auto_distance_value)
            config["screening"]["interaction_distance_unit"] = str(auto_distance_unit)

    patient_outcomes = build_patient_outcomes(analysis_df, config)
    if patient_outcomes.empty:
        raise ValueError("No patient-level outcomes could be constructed from the selected endpoint columns.")

    if explicit_interval_mode:
        combined = compute_all_features(
            analysis_df,
            config,
            float(validation_width_value),
            zone_start=float(validation_zone_start),
            zone_end=float(validation_zone_end),
            log_fn=log_fn,
        )
    else:
        combined = compute_all_features(analysis_df, config, float(validation_width_value), log_fn=log_fn)
    if combined.empty:
        raise ValueError(
            f"No valid feature rows were produced at range {interval_label}. "
            "Try a different width, phenotype column, or phenotype subset."
        )

    patient_features = aggregate_to_patient(combined, config)
    if patient_features.empty:
        raise ValueError(
            f"No patient-level feature values were produced at range {interval_label}."
        )

    stats_df = screen_patient_features(patient_features, patient_outcomes, config)

    return {
        "analysis_df": analysis_df,
        "combined": combined,
        "patient_features": patient_features,
        "patient_outcomes": patient_outcomes,
        "stats_all": stats_df,
        "validation_width": float(validation_width_value),
        "validation_zone_start": float(validation_zone_start),
        "validation_zone_end": float(validation_zone_end),
        "validation_zone_label": interval_label,
        "config": config,
    }


def build_motif_threshold_sensitivity(
    hit_row: pd.Series,
    analysis_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
    zone_start: float | None = None,
    zone_end: float | None = None,
    thresholds: list[int] | None = None,
    log_fn: LogFn = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "unsupported",
        "message": "Motif-support sensitivity is currently available only for survival dyad/triad features.",
        "sensitivity_df": pd.DataFrame(),
    }

    if analysis_df.empty or patient_outcomes_df.empty:
        result["message"] = "No validation data are available for the motif-support sensitivity analysis."
        return result

    feature_family = str(hit_row.get("feature_family") or "")
    endpoint_type = str(hit_row.get("endpoint_type") or "")
    if feature_family not in {"dyad", "triad"} or endpoint_type != "survival":
        return result

    screening = config.get("screening", {}) if isinstance(config.get("screening", {}), dict) else {}
    if max(0, _safe_int(screening.get("dyad_triad_permutation_n_iter", 1000), 1000)) <= 0:
        result["message"] = "Permutation testing is disabled, so min_obs_motif_count sensitivity is not applicable."
        return result

    threshold_values = sorted({max(0, _safe_int(value, 0)) for value in (thresholds or list(range(0, 5)))})
    if not threshold_values:
        result["message"] = "No valid thresholds were provided for the motif-support sensitivity analysis."
        return result

    zone_width = _safe_float(hit_row.get("zone_width"), float("nan"))
    interval_mode = zone_start is not None and zone_end is not None and np.isfinite(_safe_float(zone_start, float("nan"))) and np.isfinite(_safe_float(zone_end, float("nan")))
    zone_start_value = _safe_float(zone_start, 0.0)
    zone_end_value = _safe_float(zone_end, zone_width)
    feature_name = str(hit_row.get("feature_name") or "")
    endpoint_label = str(hit_row.get("endpoint_label") or "")
    stratum_label = str(hit_row.get("stratum_label") or "")
    template_hit_row = hit_row.copy()

    rows: list[dict[str, Any]] = []
    for threshold in threshold_values:
        threshold_config = dict(config)
        threshold_config["screening"] = dict(screening)
        threshold_config["screening"]["min_obs_motif_count"] = int(threshold)
        row_payload: dict[str, Any] = {
            "min_obs_motif_count": int(threshold),
            "feature_emitted": False,
            "result_available": False,
            "n_feature_patients": 0,
            "n_missing_endpoint_patients": 0,
            "n_split_excluded_patients": 0,
            "n_final_patients": 0,
            "group_high_n": 0,
            "group_low_n": 0,
            "p_value": float("nan"),
            "hazard_ratio": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "concordance_index": float("nan"),
            "ph_test_p": float("nan"),
            "cutpoint": float("nan"),
            "significance_label": "not_testable",
            "direction": "neutral",
            "error": "",
        }
        try:
            if interval_mode:
                combined = compute_all_features(
                    analysis_df,
                    threshold_config,
                    float(zone_width),
                    zone_start=zone_start_value,
                    zone_end=zone_end_value,
                    log_fn=log_fn,
                )
            else:
                combined = compute_all_features(analysis_df, threshold_config, float(zone_width), log_fn=log_fn)
            patient_features = aggregate_to_patient(combined, threshold_config) if not combined.empty else pd.DataFrame()

            feature_mask = pd.Series(False, index=patient_features.index)
            if not patient_features.empty:
                feature_mask = (
                    (patient_features.get("feature_name", pd.Series("", index=patient_features.index)).astype(str) == feature_name)
                    & (patient_features.get("feature_family", pd.Series("", index=patient_features.index)).astype(str) == feature_family)
                    & (patient_features.get("stratum_label", pd.Series("", index=patient_features.index)).astype(str) == stratum_label)
                )
                if np.isfinite(zone_width) and "zone_width" in patient_features.columns:
                    width_series = pd.to_numeric(patient_features["zone_width"], errors="coerce")
                    feature_mask = feature_mask & width_series.map(
                        lambda value: bool(
                            np.isfinite(_safe_float(value, float("nan")))
                            and np.isclose(_safe_float(value, float("nan")), float(zone_width), atol=1e-6)
                        )
                    )
            row_payload["n_feature_patients"] = int(
                patient_features.loc[feature_mask, "patient"].dropna().astype(str).nunique()
                if (not patient_features.empty and bool(feature_mask.any()) and "patient" in patient_features.columns)
                else 0
            )
            row_payload["feature_emitted"] = bool(row_payload["n_feature_patients"] > 0)

            attrition_summary = build_feature_endpoint_attrition_summary(template_hit_row, patient_features, patient_outcomes_df, threshold_config) if row_payload["feature_emitted"] else {}
            row_payload["n_missing_endpoint_patients"] = int(_safe_int(attrition_summary.get("n_missing_endpoint_patients"), 0))
            row_payload["n_split_excluded_patients"] = int(_safe_int(attrition_summary.get("n_split_excluded_patients"), 0))
            row_payload["n_final_patients"] = int(_safe_int(attrition_summary.get("n_final_patients"), 0))

            stats_df = screen_patient_features(patient_features, patient_outcomes_df, threshold_config) if not patient_features.empty else pd.DataFrame()
            selected_row = None
            if not stats_df.empty:
                stats_mask = (
                    (stats_df.get("feature_name", pd.Series("", index=stats_df.index)).astype(str) == feature_name)
                    & (stats_df.get("feature_family", pd.Series("", index=stats_df.index)).astype(str) == feature_family)
                    & (stats_df.get("endpoint_label", pd.Series("", index=stats_df.index)).astype(str) == endpoint_label)
                    & (stats_df.get("stratum_label", pd.Series("", index=stats_df.index)).astype(str) == stratum_label)
                )
                if np.isfinite(zone_width) and "zone_width" in stats_df.columns:
                    stats_width = pd.to_numeric(stats_df["zone_width"], errors="coerce")
                    stats_mask = stats_mask & stats_width.map(
                        lambda value: bool(
                            np.isfinite(_safe_float(value, float("nan")))
                            and np.isclose(_safe_float(value, float("nan")), float(zone_width), atol=1e-6)
                        )
                    )
                if bool(stats_mask.any()):
                    selected_row = stats_df.loc[stats_mask].iloc[0]

            if selected_row is not None:
                row_payload.update(
                    {
                        "result_available": True,
                        "group_high_n": int(_safe_int(selected_row.get("group_high_n"), 0)),
                        "group_low_n": int(_safe_int(selected_row.get("group_low_n"), 0)),
                        "p_value": _safe_float(selected_row.get("p_value"), float("nan")),
                        "hazard_ratio": _safe_float(selected_row.get("hazard_ratio"), float("nan")),
                        "ci_low": _safe_float(selected_row.get("ci_low"), float("nan")),
                        "ci_high": _safe_float(selected_row.get("ci_high"), float("nan")),
                        "concordance_index": _safe_float(selected_row.get("concordance_index"), float("nan")),
                        "ph_test_p": _safe_float(selected_row.get("ph_test_p"), float("nan")),
                        "cutpoint": _safe_float(selected_row.get("cutpoint"), float("nan")),
                        "significance_label": str(selected_row.get("significance_label") or "not_significant"),
                    }
                )
                row_payload["direction"] = _survival_direction(row_payload["hazard_ratio"])
        except Exception as exc:
            row_payload["error"] = str(exc)
        rows.append(row_payload)

    sensitivity_df = pd.DataFrame(rows).sort_values("min_obs_motif_count").reset_index(drop=True)
    result.update(
        {
            "status": "ok",
            "message": "",
            "sensitivity_df": sensitivity_df,
            "default_threshold": int(max(0, _safe_int(screening.get("min_obs_motif_count", 3), 3))),
            "feature_name": feature_name,
            "feature_label": str(hit_row.get("feature_label") or feature_name),
            "endpoint_label": endpoint_label,
            "stratum_label": stratum_label,
            "thresholds": threshold_values,
        }
    )
    return result


def plot_motif_threshold_sensitivity(
    sensitivity_df: pd.DataFrame,
    hit_row: pd.Series,
    default_threshold: int,
) -> Figure:
    fig = plt.figure(figsize=(11.8, 9.0))
    if sensitivity_df.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No threshold-sensitivity results are available for this feature.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    plot_df = sensitivity_df.copy().sort_values("min_obs_motif_count").reset_index(drop=True)
    thresholds = pd.to_numeric(plot_df.get("min_obs_motif_count", pd.Series(dtype=float)), errors="coerce")
    default_threshold_value = int(max(0, _safe_int(default_threshold, 0)))

    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 1.0], hspace=0.28)
    ax_counts = fig.add_subplot(grid[0])
    ax_hr = fig.add_subplot(grid[1])
    ax_p = fig.add_subplot(grid[2])

    n_feature = pd.to_numeric(plot_df.get("n_feature_patients", pd.Series(dtype=float)), errors="coerce")
    n_final = pd.to_numeric(plot_df.get("n_final_patients", pd.Series(dtype=float)), errors="coerce")
    ax_counts.plot(thresholds, n_feature, marker="o", linewidth=2.0, color="#475569", label="Patients with emitted primary motif score")
    ax_counts.plot(thresholds, n_final, marker="o", linewidth=2.0, color="#111827", label="Patients used in the survival test")
    ax_counts.axvline(default_threshold_value, color="#64748b", linestyle="--", linewidth=1.5, alpha=0.9)
    ax_counts.set_ylabel("Patients")
    ax_counts.set_xticks(thresholds.dropna().astype(int).tolist())
    ax_counts.grid(axis="y", alpha=0.15)
    ax_counts.spines["top"].set_visible(False)
    ax_counts.spines["right"].set_visible(False)
    ax_counts.legend(loc="best", frameon=True, framealpha=0.95, fontsize=9)

    hr_values = pd.to_numeric(plot_df.get("hazard_ratio", pd.Series(dtype=float)), errors="coerce")
    ci_low = pd.to_numeric(plot_df.get("ci_low", pd.Series(dtype=float)), errors="coerce")
    ci_high = pd.to_numeric(plot_df.get("ci_high", pd.Series(dtype=float)), errors="coerce")
    result_available = plot_df.get("result_available", pd.Series(False, index=plot_df.index)).fillna(False).astype(bool)
    hr_axis_bounds: list[float] = []
    ax_hr.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.5)
    ax_hr.axvline(default_threshold_value, color="#64748b", linestyle="--", linewidth=1.5, alpha=0.9)
    for idx, threshold in enumerate(thresholds):
        if not np.isfinite(float(threshold)):
            continue
        if bool(result_available.iloc[idx]) and np.isfinite(hr_values.iloc[idx]) and float(hr_values.iloc[idx]) > 0.0:
            hr_value = float(hr_values.iloc[idx])
            ci_low_value = float(ci_low.iloc[idx]) if np.isfinite(ci_low.iloc[idx]) and float(ci_low.iloc[idx]) > 0.0 else hr_value
            ci_high_value = float(ci_high.iloc[idx]) if np.isfinite(ci_high.iloc[idx]) and float(ci_high.iloc[idx]) > 0.0 else hr_value
            ci_low_value = min(ci_low_value, hr_value)
            ci_high_value = max(ci_high_value, hr_value)
            hr_axis_bounds.extend([ci_low_value, hr_value, ci_high_value])
            direction_color = SURVIVAL_RISK_COLOR if hr_value > 1.0 else SURVIVAL_PROTECTIVE_COLOR if hr_value < 1.0 else NEUTRAL_COLOR
            lower = float(max(0.0, hr_value - ci_low_value))
            upper = float(max(0.0, ci_high_value - hr_value))
            ax_hr.errorbar(
                [float(threshold)],
                [hr_value],
                yerr=np.array([[lower], [upper]], dtype=float),
                fmt="o",
                color=direction_color,
                ecolor=direction_color,
                elinewidth=1.8,
                capsize=4,
                markersize=7,
                zorder=3,
            )
        else:
            ax_hr.scatter([float(threshold)], [1.0], marker="x", s=56, linewidths=1.8, color="#94a3b8", zorder=3)
    finite_hr_mask = result_available & np.isfinite(hr_values) & (hr_values > 0)
    if bool(finite_hr_mask.any()):
        ax_hr.plot(thresholds[finite_hr_mask], hr_values[finite_hr_mask], color="#64748b", linewidth=1.2, alpha=0.75, zorder=2)
    ax_hr.set_yscale("log", base=2)
    if hr_axis_bounds:
        axis_min = max(min(hr_axis_bounds) / 1.35, 1e-4)
        axis_max = max(hr_axis_bounds) * 1.35
        if axis_min < axis_max:
            ax_hr.set_ylim(axis_min, axis_max)
    ax_hr.set_ylabel("Hazard ratio (log2 scale)")
    ax_hr.set_xticks(thresholds.dropna().astype(int).tolist())
    ax_hr.grid(axis="y", alpha=0.15)
    ax_hr.spines["top"].set_visible(False)
    ax_hr.spines["right"].set_visible(False)

    p_values = pd.to_numeric(plot_df.get("p_value", pd.Series(dtype=float)), errors="coerce")
    neg_log10_p = np.where(np.isfinite(p_values) & (p_values > 0), -np.log10(p_values), np.nan)
    ax_p.axvline(default_threshold_value, color="#64748b", linestyle="--", linewidth=1.5, alpha=0.9)
    ax_p.axhline(-math.log10(0.05), color="#cbd5e1", linestyle=":", linewidth=1.5)
    ax_p.plot(thresholds, neg_log10_p, marker="o", linewidth=2.0, color="#0f172a")
    no_result_mask = ~result_available
    if bool(no_result_mask.any()):
        ax_p.scatter(thresholds[no_result_mask], np.zeros(int(no_result_mask.sum())), marker="x", s=56, linewidths=1.8, color="#94a3b8", label="No valid survival test")
        ax_p.legend(loc="best", frameon=True, framealpha=0.95, fontsize=9)
    ax_p.set_ylabel(r"$-\log_{10}(p)$")
    ax_p.set_xlabel("min_obs_motif_count threshold")
    ax_p.set_xticks(thresholds.dropna().astype(int).tolist())
    ax_p.grid(axis="y", alpha=0.15)
    ax_p.spines["top"].set_visible(False)
    ax_p.spines["right"].set_visible(False)

    fig.suptitle(
        f"Motif-support sensitivity | {str(hit_row.get('feature_label') or hit_row.get('feature_name') or 'Selected motif')}\n"
        f"Endpoint: {str(hit_row.get('endpoint_label') or '')} | Stratum: {str(hit_row.get('stratum_label') or '')}",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Threshold 0 allows a primary permutation Z-score whenever the null variance is finite; larger thresholds require more observed motif instances per ROI before the primary score is emitted. Use this as a stability diagnostic, not as a p-value optimization step.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#475569",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    return fig


def _weighted_rank_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Convert a metric series into a 0..1 rank score for weighted candidate ranking."""
    numeric = pd.to_numeric(values, errors="coerce")
    scores = pd.Series(0.0, index=numeric.index, dtype=float)
    valid = numeric.notna()
    if not bool(valid.any()):
        return scores

    valid_values = numeric.loc[valid]
    if len(valid_values) == 1:
        scores.loc[valid] = 1.0
        return scores

    ranks = valid_values.rank(method="average", ascending=not higher_is_better)
    denominator = float(max(1, len(valid_values) - 1))
    scores.loc[valid] = 1.0 - ((ranks - 1.0) / denominator)
    return scores


def _candidate_priority_scores(candidate_df: pd.DataFrame, *, use_bootstrap: bool) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build reproducibility-first ranking scores for candidate widths.

    Bootstrap recovery and range-stability should dominate width selection when
    available. Raw hit counts and p/FDR remain secondary tie-break signals.
    """
    zero = pd.Series(0.0, index=candidate_df.index, dtype=float)
    reproducibility_score = zero.copy()
    significance_score = zero.copy()

    if "feature_stability_score" in candidate_df.columns:
        reproducibility_score += 5.0 * _weighted_rank_score(candidate_df["feature_stability_score"], higher_is_better=True)
        reproducibility_score += 2.5 * _weighted_rank_score(candidate_df.get("feature_stability_exact_score", zero), higher_is_better=True)
        reproducibility_score += 1.5 * _weighted_rank_score(candidate_df.get("feature_stability_band_hits", zero), higher_is_better=True)
        reproducibility_score += 1.0 * _weighted_rank_score(candidate_df.get("feature_stability_exact_hits", zero), higher_is_better=True)
        reproducibility_score += 1.5 * _weighted_rank_score(candidate_df.get("feature_stability_top_recovery", zero), higher_is_better=True)

    if use_bootstrap:
        reproducibility_score += 8.0 * _weighted_rank_score(candidate_df.get("bootstrap_reference_recovery", zero), higher_is_better=True)
        reproducibility_score += 2.5 * _weighted_rank_score(candidate_df.get("bootstrap_valid_iterations", zero), higher_is_better=True)
        reproducibility_score += 1.5 * _weighted_rank_score(candidate_df.get("bootstrap_n_significant_mean", zero), higher_is_better=True)
        reproducibility_score += 1.0 * _weighted_rank_score(candidate_df.get("bootstrap_n_discovery_mean", zero), higher_is_better=True)

    reproducibility_score += 1.5 * _weighted_rank_score(candidate_df.get("patient_coverage_fraction", zero), higher_is_better=True)
    reproducibility_score += 0.5 * _weighted_rank_score(candidate_df.get("roi_coverage_fraction", zero), higher_is_better=True)

    significance_score += 2.0 * _weighted_rank_score(candidate_df.get("n_significant", zero), higher_is_better=True)
    significance_score += 1.5 * _weighted_rank_score(candidate_df.get("n_discovery", zero), higher_is_better=True)
    significance_score += 0.75 * _weighted_rank_score(candidate_df.get("best_fdr_q", zero), higher_is_better=False)
    significance_score += 0.35 * _weighted_rank_score(candidate_df.get("best_p_value", zero), higher_is_better=False)

    priority_score = reproducibility_score + significance_score
    return reproducibility_score, significance_score, priority_score


def rank_candidates(candidate_df: pd.DataFrame, use_bootstrap: bool = True) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    ranked = candidate_df.copy()
    reproducibility_score, significance_score, priority_score = _candidate_priority_scores(ranked, use_bootstrap=use_bootstrap)
    ranked["candidate_reproducibility_score"] = reproducibility_score
    ranked["candidate_significance_score"] = significance_score
    ranked["candidate_priority_score"] = priority_score
    sort_defaults: dict[str, float] = {
        "bootstrap_reference_recovery": float("nan"),
        "feature_stability_score": float("nan"),
        "patient_coverage_fraction": float("nan"),
        "n_significant": float("nan"),
        "n_discovery": float("nan"),
        "best_fdr_q": float("nan"),
        "best_p_value": float("nan"),
        "zone_width": float("nan"),
    }
    for column, default_value in sort_defaults.items():
        if column not in ranked.columns:
            ranked[column] = default_value
    ranked = ranked.sort_values(
        by=[
            "candidate_priority_score",
            "candidate_reproducibility_score",
            "bootstrap_reference_recovery",
            "feature_stability_score",
            "patient_coverage_fraction",
            "candidate_significance_score",
            "n_significant",
            "n_discovery",
            "best_fdr_q",
            "best_p_value",
            "zone_width",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_feature_stability_summary(
    stats_df: pd.DataFrame,
    candidate_summary: pd.DataFrame,
) -> pd.DataFrame:
    if stats_df.empty:
        return pd.DataFrame()

    if not candidate_summary.empty and "zone_width" in candidate_summary.columns:
        ordered_widths = pd.to_numeric(candidate_summary["zone_width"], errors="coerce").dropna().astype(float).sort_values().unique().tolist()
    else:
        ordered_widths = pd.to_numeric(stats_df.get("zone_width", pd.Series(dtype=float)), errors="coerce").dropna().astype(float).sort_values().unique().tolist()
    ordered_widths = [float(value) for value in ordered_widths]
    width_index_lookup = {_width_lookup_key(width): index for index, width in enumerate(ordered_widths)}

    coverage_lookup: dict[str, float] = {}
    if not candidate_summary.empty and "patient_coverage_fraction" in candidate_summary.columns:
        for row in candidate_summary.itertuples(index=False):
            width_key = _width_lookup_key(getattr(row, "zone_width", np.nan))
            coverage_value = _to_float(getattr(row, "patient_coverage_fraction", np.nan))
            if width_key and np.isfinite(coverage_value):
                coverage_lookup[width_key] = float(coverage_value)

    summary_rows: list[dict[str, Any]] = []
    group_cols = ["endpoint_label", "endpoint_type", "stratum_label", "feature_family", "feature_name", "feature_label"]
    for group_values, feature_df in stats_df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_record = dict(zip(group_cols, group_values))
        work = feature_df.copy()
        work["__width_index__"] = work["zone_width"].map(lambda value: width_index_lookup.get(_width_lookup_key(value), -1))
        work = work[work["__width_index__"] >= 0].copy()
        if work.empty:
            continue
        work["__significance_rank__"] = work["significance_label"].map(_feature_significance_rank).fillna(0)
        work["__fdr_q__"] = pd.to_numeric(work["fdr_q"], errors="coerce")
        work["__p_value__"] = pd.to_numeric(work["p_value"], errors="coerce")
        work = work.sort_values(["__significance_rank__", "__fdr_q__", "__p_value__", "__width_index__"], ascending=[False, True, True, True], na_position="last").reset_index(drop=True)
        best_row = work.iloc[0]

        significant_df = work[work["significance_label"].astype(str) == "significant_fdr"].copy()
        discovery_df = work[work["significance_label"].astype(str).isin(["significant_fdr", "discovery_fdr"])].copy()
        significant_indices = sorted({int(value) for value in significant_df["__width_index__"].tolist()})
        discovery_indices = sorted({int(value) for value in discovery_df["__width_index__"].tolist()})
        significant_band = _longest_consecutive_indices(significant_indices)
        discovery_band = _longest_consecutive_indices(discovery_indices)
        reference_indices = significant_indices if significant_indices else discovery_indices
        reference_band = significant_band if significant_band else discovery_band
        reference_significance_label = "significant_fdr" if significant_indices else "discovery_fdr" if discovery_indices else "not_significant"

        best_width = _to_float(best_row.get("zone_width"))
        best_width_key = _width_lookup_key(best_width)
        best_fdr_q = _to_float(best_row.get("fdr_q"))
        best_p_value = _to_float(best_row.get("p_value"))
        best_hazard_ratio = _to_float(best_row.get("hazard_ratio"))
        best_effect_size = _to_float(best_row.get("effect_size"))

        discovery_coverages = [coverage_lookup.get(_width_lookup_key(width), float("nan")) for width in discovery_df["zone_width"].tolist()]
        discovery_coverages = [float(value) for value in discovery_coverages if np.isfinite(value)]

        summary_rows.append(
            {
                **group_record,
                "best_zone_width": float(best_width) if np.isfinite(best_width) else float("nan"),
                "best_zone_label": format_zone_label(best_width),
                "best_significance_label": str(best_row.get("significance_label") or ""),
                "best_fdr_q": float(best_fdr_q) if np.isfinite(best_fdr_q) else float("nan"),
                "best_p_value": float(best_p_value) if np.isfinite(best_p_value) else float("nan"),
                "best_direction": str(best_row.get("direction") or ""),
                "best_hazard_ratio": float(best_hazard_ratio) if np.isfinite(best_hazard_ratio) else float("nan"),
                "best_effect_size": float(best_effect_size) if np.isfinite(best_effect_size) else float("nan"),
                "n_significant_ranges": int(len(significant_indices)),
                "n_discovery_ranges": int(len(discovery_indices)),
                "significant_ranges_text": _indices_to_range_text(significant_indices, ordered_widths),
                "discovery_ranges_text": _indices_to_range_text(discovery_indices, ordered_widths),
                "significant_band_size": int(len(significant_band)),
                "discovery_band_size": int(len(discovery_band)),
                "significant_band_label": _band_label_from_indices(significant_band, ordered_widths),
                "discovery_band_label": _band_label_from_indices(discovery_band, ordered_widths),
                "reference_significance_label": reference_significance_label,
                "reference_band_label": _band_label_from_indices(reference_band, ordered_widths),
                "best_range_coverage": coverage_lookup.get(best_width_key, float("nan")),
                "mean_discovery_coverage": float(np.nanmean(discovery_coverages)) if discovery_coverages else float("nan"),
                "direction_consistency": _direction_consistency_label(work),
                "reference_width_indices": reference_indices,
                "reference_band_indices": reference_band,
            }
        )

    return pd.DataFrame(summary_rows)


def bootstrap_feature_stability(
    feature_summary: pd.DataFrame,
    patient_features_per_width: dict[float, pd.DataFrame],
    patient_outcomes: pd.DataFrame,
    config: dict[str, Any],
    candidate_widths: list[float],
    rng: np.random.Generator,
    log_fn: LogFn = None,
    combined_features_per_width: dict[float, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if feature_summary.empty or patient_outcomes.empty or not patient_features_per_width:
        return feature_summary.copy()

    feature_results = feature_summary.copy()
    feature_results["feature_bootstrap_recovery"] = float("nan")
    feature_results["feature_bootstrap_exact_recovery"] = float("nan")
    feature_results["feature_bootstrap_best_fdr_q_median"] = float("nan")
    feature_results["feature_bootstrap_valid_iterations"] = 0

    reference_mask = feature_results["reference_significance_label"].astype(str).isin(["significant_fdr", "discovery_fdr"])
    if not bool(reference_mask.any()):
        return feature_results

    width_order = [float(value) for value in candidate_widths if np.isfinite(_to_float(value))]
    if not width_order:
        return feature_results
    width_key_lookup = {_width_lookup_key(width): index for index, width in enumerate(width_order)}
    neighbor_steps = max(0, _safe_int(config["screening"].get("feature_bootstrap_neighbor_steps"), 1))
    iterations = max(1, _safe_int(config["screening"].get("bootstrap_iterations"), 30))
    patient_fraction = min(1.0, max(0.1, _safe_float(config["screening"].get("bootstrap_patient_fraction"), 0.80)))
    sampling_unit = _bootstrap_sampling_unit(config)
    patient_units = _build_bootstrap_patient_units(patient_outcomes)
    endpoints = _prepare_enabled_endpoints(config)
    merge_cols = _patient_merge_columns(config, patient_outcomes)

    sig_families = feature_results.loc[reference_mask, "feature_family"].unique().tolist() if "feature_family" in feature_results.columns else []

    merged_by_width: dict[str, pd.DataFrame] = {}
    combined_by_width: dict[str, pd.DataFrame] = {}
    roi_units = pd.DataFrame()
    for width in width_order:
        patient_features = patient_features_per_width.get(float(width), pd.DataFrame())
        if patient_features.empty:
            continue
        if sig_families:
            patient_features = patient_features[patient_features["feature_family"].isin(sig_families)].copy()
        if sampling_unit == "roi":
            combined_features = combined_features_per_width.get(float(width), pd.DataFrame()) if isinstance(combined_features_per_width, dict) else pd.DataFrame()
            if sig_families and not combined_features.empty and "feature_family" in combined_features.columns:
                combined_features = combined_features[combined_features["feature_family"].isin(sig_families)].copy()
            combined_with_keys = _prepare_roi_bootstrap_frame(combined_features)
            if combined_with_keys.empty:
                continue
            combined_by_width[_width_lookup_key(width)] = combined_with_keys
        else:
            merged = patient_features.merge(patient_outcomes, on=merge_cols, how="left")
            if merged.empty:
                continue
            merged = merged.copy()
            merged["__bootstrap_key__"] = merged["patient"].astype(str) + "\0" + merged["stratum_label"].astype(str)
            merged_by_width[_width_lookup_key(width)] = merged
    if sampling_unit == "roi":
        roi_units = _build_bootstrap_roi_units_from_widths({float(width): df for width, df in zip(width_order, [combined_by_width.get(_width_lookup_key(width), pd.DataFrame()) for width in width_order])})
        if not combined_by_width or roi_units.empty:
            return feature_results
    elif not merged_by_width:
        return feature_results

    reference_info: dict[tuple[str, str, str], dict[str, Any]] = {}
    feature_metrics: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for _, row in feature_results.loc[reference_mask].iterrows():
        key = (str(row.get("endpoint_label") or ""), str(row.get("stratum_label") or ""), str(row.get("feature_name") or ""))
        reference_indices = [int(value) for value in (row.get("reference_width_indices") or []) if pd.notna(value)]
        if not reference_indices:
            continue
        reference_info[key] = {
            "required_label": str(row.get("reference_significance_label") or "discovery_fdr"),
            "reference_indices": reference_indices,
        }
        feature_metrics[key] = {"band": [], "exact": [], "best_q": []}
    if not reference_info:
        return feature_results

    import concurrent.futures
    import os

    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(iterations)]
    max_workers = min(8, max(1, (os.cpu_count() or 2) - 1))

    completed_iterations = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if sampling_unit == "roi":
            futures = [
                executor.submit(
                    _run_feature_bootstrap_iteration_roi,
                    seed,
                    roi_units,
                    patient_fraction,
                    combined_by_width,
                    patient_outcomes,
                    merge_cols,
                    config,
                    endpoints,
                    reference_info,
                    width_key_lookup,
                    neighbor_steps,
                )
                for seed in seeds
            ]
        else:
            futures = [
                executor.submit(
                    _run_feature_bootstrap_iteration,
                    seed,
                    patient_units,
                    patient_fraction,
                    merged_by_width,
                    config,
                    endpoints,
                    reference_info,
                    width_key_lookup,
                    neighbor_steps,
                )
                for seed in seeds
            ]
        
        for future in concurrent.futures.as_completed(futures):
            iteration_metrics = future.result()
            if iteration_metrics:
                completed_iterations += 1
                for key, metrics in iteration_metrics.items():
                    feature_metrics[key]["band"].append(metrics["band"])
                    feature_metrics[key]["exact"].append(metrics["exact"])
                    feature_metrics[key]["best_q"].append(metrics["best_q"])
                    
            if log_fn and completed_iterations % max(1, iterations // 5) == 0:
                _log(log_fn, f"  Feature stability across all widths: {completed_iterations}/{iterations} iterations completed")

    if completed_iterations == 0:
        return feature_results

    for row_index, row in feature_results.loc[reference_mask].iterrows():
        key = (str(row.get("endpoint_label") or ""), str(row.get("stratum_label") or ""), str(row.get("feature_name") or ""))
        metrics = feature_metrics.get(key)
        if not metrics:
            continue
        feature_results.at[row_index, "feature_bootstrap_recovery"] = float(np.nanmean(metrics["band"])) if metrics["band"] else float("nan")
        feature_results.at[row_index, "feature_bootstrap_exact_recovery"] = float(np.nanmean(metrics["exact"])) if metrics["exact"] else float("nan")
        best_q_values = [value for value in metrics["best_q"] if np.isfinite(value)]
        feature_results.at[row_index, "feature_bootstrap_best_fdr_q_median"] = float(np.nanmedian(best_q_values)) if best_q_values else float("nan")
        feature_results.at[row_index, "feature_bootstrap_valid_iterations"] = int(completed_iterations)

    return feature_results


def rank_feature_stability(feature_summary: pd.DataFrame) -> pd.DataFrame:
    if feature_summary.empty:
        return feature_summary
    ranked = feature_summary.copy()
    ranked["_sort_bootstrap_band"] = pd.to_numeric(ranked.get("feature_bootstrap_recovery", pd.Series(np.nan, index=ranked.index)), errors="coerce").fillna(-1.0)
    ranked["_sort_bootstrap_exact"] = pd.to_numeric(ranked.get("feature_bootstrap_exact_recovery", pd.Series(np.nan, index=ranked.index)), errors="coerce").fillna(-1.0)
    ranked["_sort_sig_band"] = pd.to_numeric(ranked.get("significant_band_size", pd.Series(0, index=ranked.index)), errors="coerce").fillna(0)
    ranked["_sort_sig_n"] = pd.to_numeric(ranked.get("n_significant_ranges", pd.Series(0, index=ranked.index)), errors="coerce").fillna(0)
    ranked["_sort_disc_band"] = pd.to_numeric(ranked.get("discovery_band_size", pd.Series(0, index=ranked.index)), errors="coerce").fillna(0)
    ranked["_sort_disc_n"] = pd.to_numeric(ranked.get("n_discovery_ranges", pd.Series(0, index=ranked.index)), errors="coerce").fillna(0)
    ranked["_sort_best_q"] = pd.to_numeric(ranked.get("best_fdr_q", pd.Series(np.nan, index=ranked.index)), errors="coerce").fillna(np.inf)
    ranked["_sort_best_p"] = pd.to_numeric(ranked.get("best_p_value", pd.Series(np.nan, index=ranked.index)), errors="coerce").fillna(np.inf)
    ranked["_sort_best_width"] = pd.to_numeric(ranked.get("best_zone_width", pd.Series(np.nan, index=ranked.index)), errors="coerce").fillna(np.inf)
    ranked = ranked.sort_values(
        by=[
            "_sort_bootstrap_band",
            "_sort_bootstrap_exact",
            "_sort_sig_band",
            "_sort_sig_n",
            "_sort_disc_band",
            "_sort_disc_n",
            "_sort_best_q",
            "_sort_best_p",
            "_sort_best_width",
        ],
        ascending=[False, False, False, False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["feature_rank"] = np.arange(1, len(ranked) + 1)
    return ranked.drop(
        columns=[
            "_sort_bootstrap_band",
            "_sort_bootstrap_exact",
            "_sort_sig_band",
            "_sort_sig_n",
            "_sort_disc_band",
            "_sort_disc_n",
            "_sort_best_q",
            "_sort_best_p",
            "_sort_best_width",
        ],
        errors="ignore",
    )


def _feature_width_priority_score(feature_row: pd.Series) -> float:
    reference_label = str(feature_row.get("reference_significance_label") or "")
    if reference_label == "significant_fdr":
        significance_weight = 2.0
        band_size = max(1, _safe_int(feature_row.get("significant_band_size"), 0))
        range_count = max(1, _safe_int(feature_row.get("n_significant_ranges"), 0))
    elif reference_label == "discovery_fdr":
        significance_weight = 1.0
        band_size = max(1, _safe_int(feature_row.get("discovery_band_size"), 0))
        range_count = max(1, _safe_int(feature_row.get("n_discovery_ranges"), 0))
    else:
        return 0.0

    band_recovery = _safe_float(feature_row.get("feature_bootstrap_recovery"), float("nan"))
    exact_recovery = _safe_float(feature_row.get("feature_bootstrap_exact_recovery"), float("nan"))
    band_recovery = float(band_recovery) if np.isfinite(band_recovery) else 0.0
    exact_recovery = float(exact_recovery) if np.isfinite(exact_recovery) else 0.0
    return float(significance_weight + 0.5 * float(band_size) + 0.25 * float(range_count) + band_recovery + 0.5 * exact_recovery)


def annotate_candidate_feature_stability(candidate_summary: pd.DataFrame, feature_summary: pd.DataFrame) -> pd.DataFrame:
    if candidate_summary.empty:
        return candidate_summary
    annotated = candidate_summary.copy()
    ordered_widths = pd.to_numeric(annotated.get("zone_width", pd.Series(dtype=float)), errors="coerce").dropna().astype(float).sort_values().unique().tolist()
    if not ordered_widths or feature_summary.empty:
        annotated["feature_stability_score"] = 0.0
        annotated["feature_stability_exact_score"] = 0.0
        annotated["feature_stability_band_hits"] = 0
        annotated["feature_stability_exact_hits"] = 0
        annotated["feature_stability_top_recovery"] = float("nan")
        annotated["feature_stability_top_feature_label"] = ""
        return annotated

    width_records: dict[str, dict[str, Any]] = {
        _width_lookup_key(width): {
            "zone_width": float(width),
            "feature_stability_score": 0.0,
            "feature_stability_exact_score": 0.0,
            "feature_stability_band_hits": 0,
            "feature_stability_exact_hits": 0,
            "feature_stability_top_recovery": float("nan"),
            "feature_stability_top_feature_label": "",
            "_top_feature_score": -1.0,
        }
        for width in ordered_widths
    }

    for _, feature_row in feature_summary.iterrows():
        feature_score = _feature_width_priority_score(feature_row)
        if feature_score <= 0.0:
            continue
        band_indices = [
            int(value)
            for value in (feature_row.get("reference_band_indices") or [])
            if pd.notna(value) and 0 <= int(value) < len(ordered_widths)
        ]
        exact_indices = [
            int(value)
            for value in (feature_row.get("reference_width_indices") or [])
            if pd.notna(value) and 0 <= int(value) < len(ordered_widths)
        ]
        if not band_indices and not exact_indices:
            continue

        top_recovery = _safe_float(feature_row.get("feature_bootstrap_recovery"), float("nan"))
        if not np.isfinite(top_recovery):
            top_recovery = _safe_float(feature_row.get("feature_bootstrap_exact_recovery"), float("nan"))
        top_feature_label = str(feature_row.get("feature_label") or feature_row.get("feature_name") or "")

        for width_index in band_indices:
            width_key = _width_lookup_key(ordered_widths[int(width_index)])
            width_record = width_records.get(width_key)
            if width_record is None:
                continue
            width_record["feature_stability_score"] += float(feature_score)
            width_record["feature_stability_band_hits"] += 1
            if np.isfinite(top_recovery):
                current_recovery = _safe_float(width_record.get("feature_stability_top_recovery"), float("nan"))
                if not np.isfinite(current_recovery) or float(top_recovery) > float(current_recovery):
                    width_record["feature_stability_top_recovery"] = float(top_recovery)
            if float(feature_score) > float(width_record.get("_top_feature_score", -1.0)):
                width_record["_top_feature_score"] = float(feature_score)
                width_record["feature_stability_top_feature_label"] = top_feature_label

        for width_index in exact_indices:
            width_key = _width_lookup_key(ordered_widths[int(width_index)])
            width_record = width_records.get(width_key)
            if width_record is None:
                continue
            width_record["feature_stability_exact_score"] += float(feature_score)
            width_record["feature_stability_exact_hits"] += 1
            if float(feature_score) > float(width_record.get("_top_feature_score", -1.0)):
                width_record["_top_feature_score"] = float(feature_score)
                width_record["feature_stability_top_feature_label"] = top_feature_label

    feature_support_df = pd.DataFrame(width_records.values()).drop(columns=["_top_feature_score"], errors="ignore")
    annotated = annotated.merge(feature_support_df, on="zone_width", how="left")
    annotated["feature_stability_score"] = pd.to_numeric(annotated.get("feature_stability_score", pd.Series(0.0, index=annotated.index)), errors="coerce").fillna(0.0)
    annotated["feature_stability_exact_score"] = pd.to_numeric(annotated.get("feature_stability_exact_score", pd.Series(0.0, index=annotated.index)), errors="coerce").fillna(0.0)
    annotated["feature_stability_band_hits"] = pd.to_numeric(annotated.get("feature_stability_band_hits", pd.Series(0, index=annotated.index)), errors="coerce").fillna(0).astype(int)
    annotated["feature_stability_exact_hits"] = pd.to_numeric(annotated.get("feature_stability_exact_hits", pd.Series(0, index=annotated.index)), errors="coerce").fillna(0).astype(int)
    annotated["feature_stability_top_recovery"] = pd.to_numeric(annotated.get("feature_stability_top_recovery", pd.Series(np.nan, index=annotated.index)), errors="coerce")
    annotated["feature_stability_top_feature_label"] = annotated.get("feature_stability_top_feature_label", pd.Series("", index=annotated.index)).fillna("").astype(str)
    return annotated


def _select_bootstrap_candidates(candidate_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.iloc[0:0].copy()

    bootstrap_pool = candidate_df.copy()
    n_significant = pd.to_numeric(bootstrap_pool.get("n_significant", 0), errors="coerce").fillna(0)
    n_discovery = pd.to_numeric(bootstrap_pool.get("n_discovery", 0), errors="coerce").fillna(0)
    best_p_value = pd.to_numeric(bootstrap_pool.get("best_p_value", np.nan), errors="coerce")
    zone_width = pd.to_numeric(bootstrap_pool.get("zone_width", np.nan), errors="coerce")

    bootstrap_pool = bootstrap_pool.assign(
        _bootstrap_n_significant=n_significant,
        _bootstrap_n_discovery=n_discovery,
        _bootstrap_best_p=best_p_value,
        _bootstrap_zone_width=zone_width,
    )
    bootstrap_pool = bootstrap_pool[bootstrap_pool["_bootstrap_n_significant"] >= 1].copy()
    if bootstrap_pool.empty:
        return bootstrap_pool

    bootstrap_pool = bootstrap_pool.sort_values(
        by=["_bootstrap_n_significant", "_bootstrap_n_discovery", "_bootstrap_best_p", "_bootstrap_zone_width"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    return bootstrap_pool.drop(columns=["_bootstrap_n_significant", "_bootstrap_n_discovery", "_bootstrap_best_p", "_bootstrap_zone_width"], errors="ignore")


def annotate_redundant_features(
    stats_df: pd.DataFrame,
    patient_features_by_width: dict[float, pd.DataFrame],
    correlation_threshold: float = 0.85
) -> pd.DataFrame:
    """Identify features that are highly correlated with a more significant feature."""
    if stats_df.empty:
        return stats_df

    stats_df = stats_df.copy()
    stats_df["is_redundant"] = False
    stats_df["redundancy_cluster"] = ""

    for (width, endpoint, stratum), group_df in stats_df.groupby(["zone_width", "endpoint_label", "stratum_label"]):
        width_float = float(width)
        patient_features = patient_features_by_width.get(width_float)
        if patient_features is None or patient_features.empty:
            continue
            
        sig_mask = group_df["significance_label"].isin(["significant_fdr", "discovery_fdr"])
        if not sig_mask.any():
            continue
            
        sig_df = group_df[sig_mask].sort_values(["fdr_q", "p_value"], na_position="last")
        feature_names = sig_df["feature_name"].unique().tolist()
        
        relevant = patient_features[patient_features["feature_name"].isin(feature_names)].copy()
        if relevant.empty:
            continue
        
        pivot = relevant.pivot_table(index="patient", columns="feature_name", values="value", aggfunc="first")
        pivot = pivot.dropna(axis=1, thresh=max(3, int(0.5 * len(pivot))))
        
        if pivot.shape[1] < 2:
            continue
            
        corr = pivot.corr(method="spearman").abs()
        
        kept_features = []
        cluster_map = {}
        for feature in sig_df["feature_name"]:
            if feature not in corr.columns:
                continue
                
            redundant_to = None
            for kept in kept_features:
                if kept in corr.columns and corr.loc[feature, kept] > correlation_threshold:
                    redundant_to = cluster_map.get(kept, kept)
                    break
                    
            if redundant_to:
                idx = group_df[group_df["feature_name"] == feature].index
                stats_df.loc[idx, "is_redundant"] = True
                stats_df.loc[idx, "redundancy_cluster"] = redundant_to
            else:
                kept_features.append(feature)
                cluster_map[feature] = feature

    return stats_df


def run_feature_screening_pipeline(
    raw_df: pd.DataFrame,
    config: dict[str, Any],
    log_fn: LogFn = None,
    progress_fn: ProgressFn = None,
) -> dict[str, Any]:
    for endpoint in config.get("endpoints", []):
        if endpoint.get("type") == "binary" and endpoint.get("multi_mode", False):
            multi_cols = [str(c) for c in endpoint.get("multi_cols", []) if c]
            endpoint["multi_col_values"] = {}
            for col in multi_cols:
                if col in raw_df.columns:
                    vals = [v for v in _safe_unique(raw_df[col]) if v not in {None, "", "nan", "NaN"}]
                    endpoint["multi_col_values"][col] = vals

    analysis_df = prepare_analysis_dataframe(raw_df, config, log_fn=log_fn)
    if analysis_df.empty:
        raise ValueError("No rows remain after applying the selected filters.")

    if bool(config["screening"].get("auto_interaction_distance", True)):
        auto_distance_value, auto_distance_unit = estimate_interaction_distance(analysis_df, config, log_fn=log_fn)
        if np.isfinite(auto_distance_value):
            config["screening"]["dyad_min_dist_um"] = 0.0
            config["screening"]["triad_min_dist_um"] = 0.0
            config["screening"]["dyad_max_dist_um"] = float(auto_distance_value)
            config["screening"]["triad_max_dist_um"] = float(auto_distance_value)
            config["screening"]["auto_interaction_distance_value"] = float(auto_distance_value)
            config["screening"]["interaction_distance_unit"] = str(auto_distance_unit)

    widths, outer_cap, distance_summary, distance_histogram = build_candidate_widths(analysis_df, config)
    if not widths:
        raise ValueError("No candidate widths could be generated from the selected distance column.")

    _zone_geometry = str(config["screening"].get("zone_geometry", "cumulative")).strip().lower()
    if _zone_geometry not in {"cumulative", "ring"}:
        _zone_geometry = "cumulative"
    config["screening"]["zone_geometry"] = _zone_geometry

    patient_outcomes = build_patient_outcomes(analysis_df, config)
    if patient_outcomes.empty:
        raise ValueError("No patient-level outcomes could be constructed from the selected endpoint columns.")

    combined_per_width: dict[float, pd.DataFrame] = {}
    patient_features_per_width: dict[float, pd.DataFrame] = {}
    stats_by_width: dict[float, pd.DataFrame] = {}
    stats_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []
    total_patients = int(patient_outcomes["patient"].astype(str).nunique())
    batch_col = str(config["columns"].get("batch") or "batch")
    roi_col = str(config["columns"].get("roi") or "roi")
    total_rois = int(analysis_df[[batch_col, roi_col]].drop_duplicates().shape[0]) if {batch_col, roi_col}.issubset(analysis_df.columns) else 0
    bootstrap_sampling_unit = _bootstrap_sampling_unit(config)
    min_patients_with_zone = max(2, _safe_int(config["screening"].get("min_patients_with_zone"), 12))
    total_widths = len(widths)

    _emit_progress(
        progress_fn,
        stage="setup",
        message=f"Prepared {total_widths} candidate widths across {total_patients} patients",
        current=0,
        total=total_widths,
        total_widths=total_widths,
        total_patients=total_patients,
        outer_cap=float(outer_cap),
        scan_min_width=float(min(widths)) if widths else None,
        scan_max_width=float(max(widths)) if widths else None,
        candidate_widths=[float(width) for width in widths],
        distance_summary=distance_summary.copy(),
        distance_histogram=distance_histogram.copy(),
        signed_width_scan=not bool(config["screening"].get("exclude_negative_distances", True)),
        auto_interaction_distance_value=config["screening"].get("auto_interaction_distance_value"),
        interaction_distance_unit=config["screening"].get("interaction_distance_unit"),
    )

    for width_index, width in enumerate(widths, start=1):
        _emit_progress(
            progress_fn,
            stage="width_start",
            message=f"Scanning width {width:.2f} ({width_index}/{total_widths})",
            current=width_index,
            total=total_widths,
            width=float(width),
            width_index=width_index,
            total_widths=total_widths,
        )
        # Zone geometry decides what "width" selects.
        #
        #   cumulative (default): 0 -> width. Widths are nested, so a signal
        #       confined to the innermost band keeps re-appearing (progressively
        #       diluted) at every larger width. Right for "how wide does the
        #       band have to be before the effect emerges, and when does it wash
        #       out", which is what the stability-across-ranges heatmap reads.
        #
        #   ring: previous width -> width, i.e. disjoint annuli. Right for
        #       "at which distance does the effect actually live", because each
        #       column is independent tissue rather than a superset of the one
        #       before it.
        #
        # zone_width stays the outer edge in both modes, so it remains the key
        # used for ordering, lookup and bootstrap alignment everywhere
        # downstream; only the selected cells and the label change.
        if _zone_geometry == "ring":
            _ring_start = float(widths[width_index - 2]) if width_index > 1 else 0.0
            _ring_end = float(width)
            _log(log_fn, f"Screening ring {_ring_start:.2f} to {_ring_end:.2f}")
        else:
            _ring_start = None
            _ring_end = None
            _log(log_fn, f"Screening width {width:.2f}")
        combined = compute_all_features(
            analysis_df, config, width,
            zone_start=_ring_start, zone_end=_ring_end, log_fn=log_fn,
        )
        zone_coverage_row = build_zone_patient_coverage_row(
            analysis_df,
            config,
            float(width),
            patient_outcomes=patient_outcomes,
            combined_features=combined,
            zone_start=_ring_start,
            zone_end=_ring_end,
        )
        n_patients_total = int(zone_coverage_row.get("total_patients", total_patients))
        n_patients_with_zone = int(zone_coverage_row.get("n_patients_with_zone", 0))
        n_patients_with_zone_cells = int(zone_coverage_row.get("n_patients_with_zone_cells", 0))
        n_patients_with_valid_zone_roi = int(zone_coverage_row.get("n_patients_with_valid_zone_roi", 0))
        n_patients_without_zone_cells = int(zone_coverage_row.get("n_patients_without_zone_cells", 0))
        n_patients_below_min_cells_per_zone = int(zone_coverage_row.get("n_patients_below_min_cells_per_zone", 0))
        n_patients_without_computable_features = int(zone_coverage_row.get("n_patients_without_computable_features", 0))
        n_patients_dropped_total = int(zone_coverage_row.get("n_patients_dropped_total", max(0, n_patients_total - n_patients_with_zone)))
        patient_coverage_fraction = _to_float(zone_coverage_row.get("patient_coverage_fraction"), float("nan"))
        if combined.empty:
            candidate_rows.append(
                {
                    "zone_width": float(width),
                    "zone_label": format_zone_label(width),
                    "total_patients": n_patients_total,
                    "n_combined_rows": 0,
                    "n_patient_rows": 0,
                    "n_patients_with_zone": n_patients_with_zone,
                    "n_patients_with_zone_cells": n_patients_with_zone_cells,
                    "n_patients_with_valid_zone_roi": n_patients_with_valid_zone_roi,
                    "n_patients_without_zone_cells": n_patients_without_zone_cells,
                    "n_patients_below_min_cells_per_zone": n_patients_below_min_cells_per_zone,
                    "n_patients_without_computable_features": n_patients_without_computable_features,
                    "n_patients_dropped_total": n_patients_dropped_total,
                    "n_rois_with_zone": 0,
                    "patient_coverage_fraction": patient_coverage_fraction if np.isfinite(patient_coverage_fraction) else 0.0,
                    "roi_coverage_fraction": 0.0,
                    "bootstrap_sampling_unit": bootstrap_sampling_unit,
                    "meets_min_patient_coverage": False,
                    "n_significant": 0,
                    "n_discovery": 0,
                    "best_p_value": float("nan"),
                    "best_fdr_q": float("nan"),
                    "strongest_feature_family": "",
                    "strongest_feature_label": "",
                    "strongest_endpoint_label": "",
                    "strongest_significance_label": "",
                }
            )
            _emit_progress(
                progress_fn,
                stage="width_complete",
                message=f"Width {width:.2f} produced no valid feature rows",
                current=width_index,
                total=total_widths,
                width=float(width),
                width_index=width_index,
                total_widths=total_widths,
                n_combined_rows=0,
                n_patient_rows=0,
                n_patients_with_zone=n_patients_with_zone,
                n_patients_with_zone_cells=n_patients_with_zone_cells,
                n_patients_with_valid_zone_roi=n_patients_with_valid_zone_roi,
                n_patients_without_zone_cells=n_patients_without_zone_cells,
                n_patients_below_min_cells_per_zone=n_patients_below_min_cells_per_zone,
                n_patients_without_computable_features=n_patients_without_computable_features,
                n_patients_dropped_total=n_patients_dropped_total,
                n_rois_with_zone=0,
                patient_coverage_fraction=patient_coverage_fraction if np.isfinite(patient_coverage_fraction) else 0.0,
                roi_coverage_fraction=0.0,
                bootstrap_sampling_unit=bootstrap_sampling_unit,
                meets_min_patient_coverage=False,
                n_significant=0,
                n_discovery=0,
                best_p_value=None,
                best_fdr_q=None,
                strongest_feature_family="",
                strongest_feature_label="",
                strongest_endpoint_label="",
                strongest_significance_label="",
                live_hit_rows=[],
            )
            continue

        patient_features = aggregate_to_patient(combined, config)
        stats_df = screen_patient_features(patient_features, patient_outcomes, config)

        combined_per_width[float(width)] = combined
        patient_features_per_width[float(width)] = patient_features
        stats_by_width[float(width)] = stats_df
        if not stats_df.empty:
            stats_frames.append(stats_df)
            p_value_hits_mask = pd.to_numeric(stats_df["p_value"], errors="coerce") <= 0.05
            significant_mask = stats_df["significance_label"] == "significant_fdr"
            discovery_mask = stats_df["significance_label"].isin(["significant_fdr", "discovery_fdr"])
            best_p = pd.to_numeric(stats_df["p_value"], errors="coerce").min()
            best_q = pd.to_numeric(stats_df["fdr_q"], errors="coerce").min()
        else:
            p_value_hits_mask = pd.Series(dtype=bool)
            significant_mask = pd.Series(dtype=bool)
            discovery_mask = pd.Series(dtype=bool)
            best_p = float("nan")
            best_q = float("nan")

        strongest_summary, live_hit_rows = _summarize_progress_hits(stats_df, top_n=None)

        n_rois_with_zone = int(combined[["batch", "roi"]].drop_duplicates().shape[0]) if not combined.empty else 0
        roi_coverage_fraction = float(n_rois_with_zone) / float(total_rois) if total_rois > 0 else float("nan")
        n_p_value_hits = int(p_value_hits_mask.sum()) if not p_value_hits_mask.empty else 0
        n_significant = int(significant_mask.sum()) if not significant_mask.empty else 0
        n_discovery = int(discovery_mask.sum()) if not discovery_mask.empty else 0
        meets_min_patient_coverage = bool(n_patients_with_zone >= min_patients_with_zone)

        candidate_rows.append(
            {
                "zone_width": float(width),
                "zone_label": format_zone_label(width),
                "total_patients": n_patients_total,
                "n_combined_rows": int(len(combined)),
                "n_patient_rows": int(len(patient_features)),
                "n_patients_with_zone": n_patients_with_zone,
                "n_patients_with_zone_cells": n_patients_with_zone_cells,
                "n_patients_with_valid_zone_roi": n_patients_with_valid_zone_roi,
                "n_patients_without_zone_cells": n_patients_without_zone_cells,
                "n_patients_below_min_cells_per_zone": n_patients_below_min_cells_per_zone,
                "n_patients_without_computable_features": n_patients_without_computable_features,
                "n_patients_dropped_total": n_patients_dropped_total,
                "n_rois_with_zone": n_rois_with_zone,
                "patient_coverage_fraction": patient_coverage_fraction,
                "roi_coverage_fraction": roi_coverage_fraction,
                "bootstrap_sampling_unit": bootstrap_sampling_unit,
                "meets_min_patient_coverage": meets_min_patient_coverage,
                "n_p_value_hits": n_p_value_hits,
                "n_significant": n_significant,
                "n_discovery": n_discovery,
                "best_p_value": float(best_p) if np.isfinite(best_p) else float("nan"),
                "best_fdr_q": float(best_q) if np.isfinite(best_q) else float("nan"),
                "strongest_feature_family": strongest_summary.get("strongest_feature_family", ""),
                "strongest_feature_label": strongest_summary.get("strongest_feature_label", ""),
                "strongest_endpoint_label": strongest_summary.get("strongest_endpoint_label", ""),
                "strongest_significance_label": strongest_summary.get("strongest_significance_label", ""),
            }
        )
        _emit_progress(
            progress_fn,
            stage="width_complete",
            message=(
                f"Width {width:.2f} complete: {n_significant} FDR hits, {n_discovery} discovery hits"
            ),
            current=width_index,
            total=total_widths,
            width=float(width),
            width_index=width_index,
            total_widths=total_widths,
            n_combined_rows=int(len(combined)),
            n_patient_rows=int(len(patient_features)),
            n_patients_with_zone=n_patients_with_zone,
            n_patients_with_zone_cells=n_patients_with_zone_cells,
            n_patients_with_valid_zone_roi=n_patients_with_valid_zone_roi,
            n_patients_without_zone_cells=n_patients_without_zone_cells,
            n_patients_below_min_cells_per_zone=n_patients_below_min_cells_per_zone,
            n_patients_without_computable_features=n_patients_without_computable_features,
            n_patients_dropped_total=n_patients_dropped_total,
            n_rois_with_zone=n_rois_with_zone,
            patient_coverage_fraction=patient_coverage_fraction,
            roi_coverage_fraction=roi_coverage_fraction,
            bootstrap_sampling_unit=bootstrap_sampling_unit,
            meets_min_patient_coverage=meets_min_patient_coverage,
            n_p_value_hits=n_p_value_hits,
            n_significant=n_significant,
            n_discovery=n_discovery,
            best_p_value=float(best_p) if np.isfinite(best_p) else None,
            best_fdr_q=float(best_q) if np.isfinite(best_q) else None,
            live_hit_rows=live_hit_rows,
            **strongest_summary,
        )
        import gc as _gc
        _gc.collect()

    stats_all = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    candidate_summary = rank_candidates(pd.DataFrame(candidate_rows), use_bootstrap=False)
    if candidate_summary.empty:
        raise ValueError("The selected configuration produced no candidate results.")

    bootstrap_top_k = max(1, _safe_int(config["screening"].get("top_k_for_bootstrap"), 5))
    bootstrap_seed = _safe_int(config["screening"].get("bootstrap_random_seed"), 20260407)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_candidates = _select_bootstrap_candidates(candidate_summary, bootstrap_top_k)
    bootstrap_widths = bootstrap_candidates["zone_width"].astype(float).tolist() if not bootstrap_candidates.empty else []
    if bootstrap_widths:
        _emit_progress(
            progress_fn,
            stage="bootstrap_setup",
            message=f"Running bootstrap stability on {len(bootstrap_widths)} widths with at least one FDR-significant hit",
            current=0,
            total=len(bootstrap_widths),
            total_bootstrap_widths=len(bootstrap_widths),
        )
    else:
        _emit_progress(
            progress_fn,
            stage="bootstrap_skipped",
            message="Skipped bootstrap because no width had at least one FDR-significant hit",
            current=0,
            total=0,
            total_bootstrap_widths=0,
        )
    for bootstrap_index, width in enumerate(bootstrap_widths, start=1):
        candidate_mask = candidate_summary["zone_width"].astype(float) == float(width)
        if not bool(candidate_summary.loc[candidate_mask, "meets_min_patient_coverage"].iloc[0]):
            _emit_progress(
                progress_fn,
                stage="bootstrap_skipped",
                message=f"Skipped bootstrap for width {width:.2f} because patient coverage was too low",
                current=bootstrap_index,
                total=len(bootstrap_widths),
                width=float(width),
                bootstrap_index=bootstrap_index,
                total_bootstrap_widths=len(bootstrap_widths),
            )
            continue
        _emit_progress(
            progress_fn,
            stage="bootstrap_start",
            message=f"Bootstrapping width {width:.2f} ({bootstrap_index}/{len(bootstrap_widths)})",
            current=bootstrap_index,
            total=len(bootstrap_widths),
            width=float(width),
            bootstrap_index=bootstrap_index,
            total_bootstrap_widths=len(bootstrap_widths),
        )
        bootstrap_metrics = bootstrap_candidate_stability(
            patient_features=patient_features_per_width.get(float(width), pd.DataFrame()),
            patient_outcomes=patient_outcomes,
            base_stats=stats_by_width.get(float(width), pd.DataFrame()),
            config=config,
            rng=rng,
            log_fn=log_fn,
            target_width=float(width),
            combined_features=combined_per_width.get(float(width), pd.DataFrame()),
        )
        for key, value in bootstrap_metrics.items():
            candidate_summary.loc[candidate_mask, key] = value
        _log(log_fn, f"Bootstrap stability computed for width {width:.2f}")
        _emit_progress(
            progress_fn,
            stage="bootstrap_complete",
            message=f"Bootstrap complete for width {width:.2f}",
            current=bootstrap_index,
            total=len(bootstrap_widths),
            width=float(width),
            bootstrap_index=bootstrap_index,
            total_bootstrap_widths=len(bootstrap_widths),
            bootstrap_reference_recovery=bootstrap_metrics.get("bootstrap_reference_recovery"),
            bootstrap_valid_iterations=bootstrap_metrics.get("bootstrap_valid_iterations"),
            bootstrap_n_significant_mean=bootstrap_metrics.get("bootstrap_n_significant_mean"),
        )

    candidate_summary = rank_candidates(candidate_summary, use_bootstrap=True)
    feature_bootstrap_rng = np.random.default_rng(bootstrap_seed + 137)
    feature_summary = build_feature_stability_summary(stats_all, candidate_summary)
    feature_summary = bootstrap_feature_stability(
        feature_summary=feature_summary,
        patient_features_per_width=patient_features_per_width,
        patient_outcomes=patient_outcomes,
        config=config,
        candidate_widths=widths,
        rng=feature_bootstrap_rng,
        log_fn=log_fn,
        combined_features_per_width=combined_per_width,
    )
    feature_summary = rank_feature_stability(feature_summary)
    candidate_summary = annotate_candidate_feature_stability(candidate_summary, feature_summary)
    candidate_summary = rank_candidates(candidate_summary, use_bootstrap=True)

    selected_width = float(candidate_summary.iloc[0]["zone_width"])
    selected_zone_label = str(candidate_summary.iloc[0]["zone_label"])
    selected_row = candidate_summary.iloc[0]
    selected_stats = stats_by_width.get(selected_width, pd.DataFrame())
    selected_strongest_summary, selected_live_hit_rows = _summarize_progress_hits(selected_stats, top_n=5)

    _emit_progress(
        progress_fn,
        stage="complete",
        message=f"Selected range {format_zone_label(selected_width)} as the best candidate",
        current=total_widths,
        total=total_widths,
        selected_width=selected_width,
        selected_zone_label=selected_zone_label,
        n_significant=int(_to_float(selected_row.get("n_significant"), 0.0)),
        n_discovery=int(_to_float(selected_row.get("n_discovery"), 0.0)),
        best_p_value=(
            float(_to_float(selected_row.get("best_p_value")))
            if np.isfinite(_to_float(selected_row.get("best_p_value")))
            else None
        ),
        best_fdr_q=(
            float(_to_float(selected_row.get("best_fdr_q")))
            if np.isfinite(_to_float(selected_row.get("best_fdr_q")))
            else None
        ),
        live_hit_rows=selected_live_hit_rows,
        **selected_strongest_summary,
    )

    stats_all = annotate_redundant_features(stats_all, patient_features_per_width)

    return {
        "analysis_df": analysis_df,
        "distance_summary": distance_summary,
        "distance_histogram": distance_histogram,
        "outer_cap": float(outer_cap),
        "candidate_summary": candidate_summary,
        "feature_stability_summary": feature_summary,
        "stats_all": stats_all,
        "patient_outcomes": patient_outcomes,
        "combined_by_width": combined_per_width,
        "patient_features_by_width": patient_features_per_width,
        "selected_width": selected_width,
        "selected_zone_label": selected_zone_label,
    }


def _figure_bytes(fig: Figure, dpi: int = 300, fmt: str = "png") -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    return buffer.getvalue()


def plot_candidate_scan(
    candidate_df: pd.DataFrame,
    selected_width: float,
    *,
    top_panel_order: tuple[str, str, str] = ("nominal", "discovery", "significant"),
    title: str = "Candidate window scan",
) -> Figure:
    plot_df = candidate_df.sort_values("zone_width").copy()
    sampling_unit = "patient"
    if "bootstrap_sampling_unit" in plot_df.columns and not plot_df["bootstrap_sampling_unit"].dropna().empty:
        sampling_unit = str(plot_df["bootstrap_sampling_unit"].dropna().astype(str).iloc[0]).strip().lower()
    coverage_column = "patient_coverage_fraction"
    coverage_label = "Patient coverage"
    if sampling_unit == "roi" and "roi_coverage_fraction" in plot_df.columns and pd.to_numeric(plot_df["roi_coverage_fraction"], errors="coerce").notna().any():
        coverage_column = "roi_coverage_fraction"
        coverage_label = "ROI coverage"

    fig = plt.figure(figsize=(10.4, 9.2))
    grid = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.25], hspace=0.10)
    ax_p = fig.add_subplot(grid[0])
    ax_d = fig.add_subplot(grid[1], sharex=ax_p)
    ax_s = fig.add_subplot(grid[2], sharex=ax_p)
    ax_b = fig.add_subplot(grid[3], sharex=ax_p)
    top_axes = [ax_p, ax_d, ax_s]

    plot_df["n_p_value_hits"] = pd.to_numeric(plot_df.get("n_p_value_hits", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0)
    plot_df["n_discovery"] = pd.to_numeric(plot_df.get("n_discovery", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0)
    plot_df["n_significant"] = pd.to_numeric(plot_df.get("n_significant", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0)
    plot_df["bootstrap_reference_recovery"] = pd.to_numeric(plot_df.get("bootstrap_reference_recovery", pd.Series(np.nan, index=plot_df.index)), errors="coerce")
    plot_df["patient_coverage_fraction"] = pd.to_numeric(plot_df.get("patient_coverage_fraction", pd.Series(np.nan, index=plot_df.index)), errors="coerce")
    plot_df["roi_coverage_fraction"] = pd.to_numeric(plot_df.get("roi_coverage_fraction", pd.Series(np.nan, index=plot_df.index)), errors="coerce")

    panel_specs = {
        "nominal": {
            "column": "n_p_value_hits",
            "color": "#2a9d8f",
            "legend": "Raw p <= 0.05 hits",
            "ylabel": "Raw p\nhits",
        },
        "discovery": {
            "column": "n_discovery",
            "color": "#f4a261",
            "legend": "FDR discovery hits",
            "ylabel": "FDR\ndiscovery",
        },
        "significant": {
            "column": "n_significant",
            "color": "#1f5a91",
            "legend": "FDR significant hits",
            "ylabel": "FDR\nsignificant",
        },
    }
    ordered_metrics = [metric for metric in top_panel_order if metric in panel_specs]
    if len(ordered_metrics) != 3:
        ordered_metrics = ["nominal", "discovery", "significant"]

    for axis, metric_key in zip(top_axes, ordered_metrics):
        spec = panel_specs[metric_key]
        axis.plot(plot_df["zone_width"], plot_df[spec["column"]], marker="o", color=spec["color"], label=spec["legend"])
        axis.set_ylabel(str(spec["ylabel"]))

    for axis in [*top_axes, ax_b]:
        axis.axvline(float(selected_width), linestyle="--", color="#9f1d35", linewidth=2, label="Selected range")
        axis.grid(True, alpha=0.2)

    right_side_lines: list[str] = []
    if plot_df["bootstrap_reference_recovery"].notna().any():
        ax_b.plot(plot_df["zone_width"], plot_df["bootstrap_reference_recovery"], marker="o", color="#2db6a3", label="Bootstrap recovery")
    else:
        right_side_lines.append("Bootstrap stability is computed for the top candidate widths.")

    if plot_df[coverage_column].notna().any():
        ax_b.plot(plot_df["zone_width"], plot_df[coverage_column], marker="s", linestyle="--", color="#6b7280", label=coverage_label)
    ax_b.set_xlabel("Signed zone boundary from 0")
    ax_b.set_ylabel("Bootstrap /\ncoverage")
    ax_b.set_ylim(0.0, 1.05)

    selected_rows = plot_df.loc[plot_df["zone_width"].astype(float) == float(selected_width)]
    if not selected_rows.empty:
        selected = selected_rows.iloc[0]
        note_parts = [f"Selected range: {format_zone_label(selected.get('zone_width'))}"]
        note_parts.append(f"Bootstrap unit: {'ROI' if sampling_unit == 'roi' else 'Patient'}")
        note_labels = {
            "nominal": ("n_p_value_hits", "Raw p hits"),
            "discovery": ("n_discovery", "Discovery hits"),
            "significant": ("n_significant", "FDR hits"),
        }
        for metric_key in ordered_metrics:
            column_name, label = note_labels[metric_key]
            note_parts.append(f"{label}: {int(_safe_int(selected.get(column_name), 0))}")
        if np.isfinite(_safe_float(selected.get("bootstrap_reference_recovery"), float("nan"))):
            note_parts.append(f"Bootstrap recovery: {_safe_float(selected.get('bootstrap_reference_recovery')):.2f}")
        selected_coverage = _safe_float(selected.get(coverage_column), float("nan"))
        if np.isfinite(selected_coverage):
            note_parts.append(f"{coverage_label}: {selected_coverage:.2f}")
        if sampling_unit == "roi":
            patient_coverage = _safe_float(selected.get("patient_coverage_fraction"), float("nan"))
            if np.isfinite(patient_coverage):
                note_parts.append(f"Patient coverage: {patient_coverage:.2f}")
        right_side_lines.extend(note_parts)

    handles: list[Any] = []
    labels: list[str] = []
    for axis in [ax_p, ax_d, ax_s, ax_b]:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.79, 0.93),
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=8,
        title="Series",
        title_fontsize=9,
    )
    if right_side_lines:
        fig.text(0.79, 0.46, "\n".join(right_side_lines), ha="left", va="top", fontsize=8.5, color="#334155")
    ax_p.set_title(str(title))
    for axis in top_axes:
        plt.setp(axis.get_xticklabels(), visible=False)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.10, right=0.74, hspace=0.12)
    return fig


def plot_zone_patient_coverage(
    candidate_df: pd.DataFrame,
    selected_width: float,
    *,
    title: str = "Patients contributing per scanned zone",
) -> Figure:
    plot_df = candidate_df.copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No zone coverage data available.", ha="center", va="center", fontsize=12)
        return fig

    plot_df["zone_width"] = pd.to_numeric(plot_df.get("zone_width", pd.Series(dtype=float)), errors="coerce")
    plot_df["n_patients_with_zone"] = pd.to_numeric(plot_df.get("n_patients_with_zone", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    plot_df["total_patients"] = pd.to_numeric(plot_df.get("total_patients", pd.Series(dtype=float)), errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["zone_width"])].sort_values("zone_width", ascending=True).reset_index(drop=True)
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No zone coverage data available.", ha="center", va="center", fontsize=12)
        return fig

    total_patients = int(plot_df["total_patients"].dropna().iloc[0]) if not plot_df["total_patients"].dropna().empty else 0
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x_positions = np.arange(len(plot_df), dtype=float)
    is_selected = plot_df["zone_width"].map(lambda value: bool(np.isclose(float(value), float(selected_width), atol=1e-6)))
    bar_colors = np.where(is_selected.to_numpy(), "#B85C38", "#BFC8D6")
    edge_colors = np.where(is_selected.to_numpy(), "#7D341C", "#8090A8")

    ax.bar(
        x_positions,
        plot_df["n_patients_with_zone"].to_numpy(dtype=float),
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        width=0.72,
        zorder=2,
    )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_ylabel("Patients with analyzable zone", fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df.get("zone_label", pd.Series(plot_df["zone_width"].map(format_zone_label))).astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    max_count = float(plot_df["n_patients_with_zone"].max()) if not plot_df.empty else 0.0
    ax.set_ylim(0.0, max(1.0, max_count * 1.18))

    ax2 = ax.twinx()
    ax2.set_ylabel("% of total patients", fontsize=10)
    if total_patients > 0:
        percentage_values = (plot_df["n_patients_with_zone"].to_numpy(dtype=float) / float(total_patients)) * 100.0
        ax2.plot(x_positions, percentage_values, color="#244E66", linewidth=2.1, marker="o", markersize=4.8, zorder=3)
        if bool(is_selected.any()):
            selected_index = int(np.flatnonzero(is_selected.to_numpy())[0])
            ax2.scatter([x_positions[selected_index]], [percentage_values[selected_index]], color="#B85C38", s=36, zorder=4)
            ax.annotate(
                f"{int(plot_df.iloc[selected_index]['n_patients_with_zone'])}/{total_patients} ({percentage_values[selected_index]:.1f}%)",
                xy=(x_positions[selected_index], float(plot_df.iloc[selected_index]["n_patients_with_zone"])),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#7D341C",
            )
        ax2.set_ylim(0.0, max(5.0, float(np.nanmax(percentage_values)) * 1.18))
    else:
        ax2.set_ylim(0.0, 100.0)

    fig.tight_layout()
    return fig


def _volcano_label_levels(targets: list[float], lower: float, upper: float, preferred_gap: float) -> list[float]:
    if not targets:
        return []
    lower_bound = float(lower)
    upper_bound = max(float(upper), lower_bound)
    gap = max(float(preferred_gap), 1e-6)
    base_upper = upper_bound
    base_lower = max(lower_bound, base_upper - gap * max(len(targets) - 1, 0))
    slots = np.linspace(base_upper, base_lower, num=len(targets)).tolist()
    assigned: list[float] = []
    for target in targets:
        if not slots:
            assigned.append(min(max(float(target), lower_bound), upper_bound))
            continue
        best_idx = min(range(len(slots)), key=lambda idx: abs(slots[idx] - float(target)))
        assigned.append(float(slots.pop(best_idx)))
    return assigned


def _annotate_volcano_hits(
    ax: Axes,
    plot_df: pd.DataFrame,
    threshold_y: float,
    *,
    include_discovery_labels: bool = False,
    include_nominal_labels: bool = False,
    max_other_labels: int = 3,
    force_feature_names: set[str] | None = None,
    force_feature_labels: set[str] | None = None,
) -> tuple[list[Any], dict[Any, str]]:
    if plot_df.empty:
        return [], {}

    x_values = pd.to_numeric(plot_df["x_value"], errors="coerce")
    y_values = pd.to_numeric(plot_df["neg_log10_p"], errors="coerce")
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if not bool(finite_mask.any()):
        return [], {}

    x_values = x_values.loc[finite_mask]
    y_values = y_values.loc[finite_mask]
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    y_max = float(y_values.max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - min(float(y_values.min()), 0.0), 1.0)

    inner_x_pad = 0.26 * x_span + 0.65
    outer_x_pad = 0.34 * x_span + 1.10
    x_left = x_min - outer_x_pad
    x_right = x_max + outer_x_pad
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(min(-0.05, -0.04 * y_span), y_max + 0.07 * y_span)
    left_label_x = x_left + 0.18 * (x_right - x_left)
    right_label_x = x_right - 0.18 * (x_right - x_left)

    max_other = max(0, int(max_other_labels))
    forced_name_set = {str(value) for value in (force_feature_names or set()) if str(value)}
    forced_label_set = {str(value) for value in (force_feature_labels or set()) if str(value)}

    significant_df = plot_df.loc[plot_df["volcano_status"].astype(str) == "significant_fdr"].copy()
    discovery_df = plot_df.loc[plot_df["volcano_status"].astype(str) == "discovery_fdr"].copy()
    nominal_df = plot_df.loc[plot_df["volcano_status"].astype(str) == "nominal_p"].copy()
    for subset in [significant_df, discovery_df, nominal_df]:
        subset.sort_values(["p_value", "neg_log10_p"], ascending=[True, False], na_position="last", inplace=True)

    label_frames = [significant_df]
    extra_frames: list[pd.DataFrame] = []
    if include_discovery_labels:
        extra_frames.append(discovery_df)
    if include_nominal_labels:
        extra_frames.append(nominal_df)
    if extra_frames and max_other > 0:
        extra_df = pd.concat(extra_frames, ignore_index=False)
        extra_df = extra_df.sort_values(["status_rank", "p_value", "neg_log10_p"], ascending=[False, True, False], na_position="last")
        label_frames.append(extra_df.head(max_other).copy())

    if forced_name_set or forced_label_set:
        forced_df = plot_df.loc[
            plot_df["feature_name"].astype(str).isin(forced_name_set)
            | plot_df["feature_label"].astype(str).isin(forced_label_set)
        ].copy()
        forced_df = forced_df.loc[forced_df["volcano_status"].astype(str).isin(["significant_fdr", "discovery_fdr", "nominal_p"])].copy()
        if not forced_df.empty:
            label_frames.append(forced_df)

    labeled = pd.concat(label_frames, ignore_index=False) if label_frames else pd.DataFrame(columns=plot_df.columns)
    if labeled.empty:
        return [], {}
    labeled = labeled.sort_values(["status_rank", "p_value", "neg_log10_p"], ascending=[False, True, False], na_position="last")
    labeled = labeled.drop_duplicates(subset=["feature_name", "feature_label", "x_value", "neg_log10_p"], keep="first").copy()
    labeled["__side__"] = np.where(pd.to_numeric(labeled["x_value"], errors="coerce") >= 0, "right", "left")

    discovery_texts: list[Any] = []
    underline_colors: dict[Any, str] = {}
    label_lower = max(threshold_y + 0.05 * y_span, ax.get_ylim()[0] + 0.18 * y_span)
    label_upper = ax.get_ylim()[1] - 0.04 * y_span
    label_gap = max(0.08 * y_span, 0.12)
    connectors: list[dict[str, Any]] = []
    title_fontsize = float(ax.title.get_fontsize() or 11.0)
    prominent_fontsize = max(11.0, title_fontsize)
    secondary_fontsize = max(7.2, title_fontsize - 3.0)

    for side_name in ["left", "right"]:
        side_df = labeled[labeled["__side__"] == side_name].sort_values(["neg_log10_p", "p_value"], ascending=[False, True], na_position="last").copy()
        if side_df.empty:
            continue
        slot_top = min(label_upper, float(side_df["neg_log10_p"].max()) + 0.03 * y_span)
        slot_bottom = max(label_lower, slot_top - label_gap * max(len(side_df) - 1, 0))
        assigned_y = np.linspace(slot_top, slot_bottom, num=len(side_df)).tolist()
        text_x = left_label_x if side_name == "left" else right_label_x
        horizontal_alignment = "left" if side_name == "left" else "right"
        for label_y, row in zip(assigned_y, side_df.itertuples(index=False)):
            label_text = str(getattr(row, "feature_label", "") or "").strip()
            if not label_text or label_text.lower() in {"nan", "none", "-"}:
                continue
            text = ax.text(
                text_x,
                float(label_y),
                label_text,
                ha=horizontal_alignment,
                va="center",
                fontsize=prominent_fontsize if str(row.volcano_status) == "significant_fdr" else secondary_fontsize,
                fontweight="bold" if str(row.volcano_status) == "significant_fdr" else "normal",
                color=str(row.point_color),
                zorder=6,
                clip_on=True,
            )
            connectors.append(
                {
                    "text": text,
                    "side": side_name,
                    "point_x": float(row.x_value),
                    "point_y": float(row.neg_log10_p),
                    "color": str(row.point_color),
                }
            )
            if str(row.volcano_status) == "discovery_fdr" and include_discovery_labels:
                discovery_texts.append(text)
                underline_colors[text] = str(row.point_color)

    if connectors:
        ax.figure.canvas.draw()
        get_renderer = getattr(ax.figure.canvas, "get_renderer", None)
        if callable(get_renderer):
            renderer = get_renderer()
            inverse = ax.transData.inverted()
            for connector in connectors:
                text = connector["text"]
                bbox = text.get_window_extent(renderer=renderer)
                mid_y = 0.5 * (bbox.y0 + bbox.y1)
                end_x_display = bbox.x1 + 4.0 if connector["side"] == "left" else bbox.x0 - 4.0
                end_point = inverse.transform((end_x_display, mid_y))
                ax.plot(
                    [float(connector["point_x"]), float(end_point[0])],
                    [float(connector["point_y"]), float(end_point[1])],
                    color=str(connector["color"]),
                    linewidth=0.65,
                    alpha=0.8,
                    solid_capstyle="round",
                    clip_on=True,
                    zorder=5,
                )
    return discovery_texts, underline_colors


def plot_volcano(
    stats_df: pd.DataFrame,
    endpoint_label: str,
    zone_width: float,
    stratum_label: str = "All",
    highlight_feature_names: list[str] | None = None,
    highlight_feature_labels: list[str] | None = None,
    highlight_feature_name: str | None = None,
    highlight_feature_label: str | None = None,
    label_settings: dict[str, Any] | None = None,
) -> Figure:
    plot_df = stats_df[
        (stats_df["endpoint_label"].astype(str) == str(endpoint_label))
        & (stats_df["zone_width"].astype(float) == float(zone_width))
        & (stats_df["stratum_label"].astype(str) == str(stratum_label))
    ].copy()
    fig, ax = plt.subplots(figsize=(12.2, 8.8))
    ax.set_box_aspect(2.0 / 2.5)
    if plot_df.empty:
        ax.text(0.5, 0.5, "No results for this endpoint / stratum.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    endpoint_type = str(plot_df["endpoint_type"].astype(str).mode().iloc[0])
    if endpoint_type == "survival":
        plot_df["x_value"] = pd.to_numeric(plot_df["log2_hazard_ratio"], errors="coerce")
        x_label = "log2(Hazard Ratio)"
    else:
        plot_df["x_value"] = pd.to_numeric(plot_df["effect_size"], errors="coerce")
        x_label = "Effect size"
    plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["x_value"], errors="coerce"))].copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "No plottable values for this endpoint / stratum.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    plot_df["volcano_status"] = plot_df.apply(_nominal_significance_status, axis=1)
    plot_df["direction"] = plot_df["direction"].fillna("neutral")
    plot_df["point_color"] = plot_df.apply(
        lambda row: _directional_significance_color(row.get("direction"), str(row.get("volcano_status") or "not_significant")),
        axis=1,
    )
    plot_df["marker_size"] = plot_df["significance_label"].map(_significance_marker_size).fillna(18.0)
    plot_df["status_rank"] = plot_df["volcano_status"].map(
        {
            "not_significant": 0,
            "nominal_p": 1,
            "discovery_fdr": 2,
            "significant_fdr": 3,
        }
    ).fillna(0)

    for status in ["not_significant", "nominal_p", "discovery_fdr", "significant_fdr"]:
        subset = plot_df[plot_df["volcano_status"] == status].copy()
        if subset.empty:
            continue
        ax.scatter(
            subset["x_value"],
            subset["neg_log10_p"],
            s=subset["marker_size"],
            c=subset["point_color"],
            alpha=0.9 if status != "not_significant" else 0.55,
            edgecolors="white" if status != "not_significant" else "none",
            linewidths=0.45 if status != "not_significant" else 0.0,
            zorder=2 + int(subset["status_rank"].max()),
        )

    threshold_y = -np.log10(0.05)
    ax.axhline(threshold_y, linestyle="--", color="#4a4a4a", linewidth=1.1)
    ax.text(
        0.99,
        threshold_y + 0.03,
        "p = 0.05",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4a4a4a",
    )
    ax.axvline(0.0, linestyle=":", color="#6b7280", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(f"Volcano | {endpoint_label} | {stratum_label} | range={format_zone_label(zone_width)}")
    settings = label_settings or {}
    highlight_name_set = {str(value) for value in highlight_feature_names or [] if str(value)}
    highlight_label_set = {str(value) for value in highlight_feature_labels or [] if str(value)}
    if highlight_feature_name:
        highlight_name_set.add(str(highlight_feature_name))
    if highlight_feature_label:
        highlight_label_set.add(str(highlight_feature_label))
    discovery_texts, underline_colors = _annotate_volcano_hits(
        ax,
        plot_df,
        threshold_y,
        include_discovery_labels=bool(settings.get("volcano_label_discovery", False)),
        include_nominal_labels=bool(settings.get("volcano_label_nominal", False)),
        max_other_labels=int(settings.get("volcano_label_other_top_n", 3) or 3),
        force_feature_names=highlight_name_set,
        force_feature_labels=highlight_label_set,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SURVIVAL_RISK_COLOR, markeredgecolor="white", markersize=7, label="Risk, FDR significant"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SURVIVAL_PROTECTIVE_COLOR, markeredgecolor="white", markersize=7, label="Protective, FDR significant"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#df7a7a", markeredgecolor="white", markersize=7, label="Discovery FDR"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#efb1b1", markeredgecolor="white", markersize=7, label="Nominal p < 0.05"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL_COLOR, markeredgecolor="none", markersize=6, label="p >= 0.05"),
    ]
    legend = ax.legend(handles=legend_handles, bbox_to_anchor=(1.28, 1), loc="upper left", frameon=True, framealpha=0.95, fontsize=8)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.67, bottom=0.10, top=0.93)
    _underline_texts(ax, discovery_texts, underline_colors)
    legend_texts = [text for text in legend.get_texts() if "Discovery" in str(text.get_text() or "")]
    _underline_figure_texts(fig, legend_texts, {text: "#df7a7a" for text in legend_texts}, pad_px=1.0)
    return fig


def plot_forest(stats_df: pd.DataFrame, endpoint_label: str, zone_width: float, stratum_label: str = "All", top_n: int = 20) -> Figure:
    plot_df = stats_df[
        (stats_df["endpoint_label"].astype(str) == str(endpoint_label))
        & (stats_df["zone_width"].astype(float) == float(zone_width))
        & (stats_df["stratum_label"].astype(str) == str(stratum_label))
    ].copy()
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.35 * top_n + 1.5)))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No results for this endpoint / stratum.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    plot_df = plot_df.sort_values(["fdr_q", "p_value"], na_position="last").copy()
    if (plot_df["endpoint_type"] == "survival").any():
        plot_df["forest_center"] = pd.to_numeric(plot_df["log2_hazard_ratio"], errors="coerce")
        plot_df["forest_lower"] = pd.to_numeric(plot_df["log2_ci_low"], errors="coerce")
        plot_df["forest_upper"] = pd.to_numeric(plot_df["log2_ci_high"], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df["forest_center"])].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No finite hazard-ratio estimates are available for this endpoint / stratum.", ha="center", va="center")
            ax.set_axis_off()
            return fig
        plot_df["forest_has_ci"] = np.isfinite(plot_df["forest_lower"]) & np.isfinite(plot_df["forest_upper"])
        plot_df = plot_df.head(int(top_n)).copy()
        labels = plot_df["feature_label"].astype(str).tolist()
        y_positions = np.arange(len(plot_df))
        colors = plot_df["direction"].map({"risk": SURVIVAL_RISK_COLOR, "protective": SURVIVAL_PROTECTIVE_COLOR}).fillna(NEUTRAL_COLOR)
        omitted_ci = int((~plot_df["forest_has_ci"]).sum())
        for y_value, row in enumerate(plot_df.itertuples(index=False)):
            center_value = float(row.forest_center)
            if bool(row.forest_has_ci):
                left_err = center_value - float(row.forest_lower)
                right_err = float(row.forest_upper) - center_value
                ax.errorbar(
                    center_value,
                    y_value,
                    xerr=np.array([[left_err], [right_err]]),
                    fmt="o",
                    color=colors.iloc[y_value],
                    ecolor=colors.iloc[y_value],
                    elinewidth=1.1,
                    capsize=3,
                    markersize=5.5,
                )
            else:
                ax.plot(
                    center_value,
                    y_value,
                    marker="o",
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=colors.iloc[y_value],
                    markeredgewidth=1.1,
                    linestyle="none",
                )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("log2(Hazard Ratio)")
        if omitted_ci > 0:
            fig.text(0.79, 0.94, f"Open circles: finite estimate but CI unavailable ({omitted_ci})", ha="left", va="top", fontsize=8, color="#475569")
    else:
        plot_df = plot_df.head(int(top_n)).copy()
        labels = plot_df["feature_label"].astype(str).tolist()
        center = pd.to_numeric(plot_df["effect_size"], errors="coerce")
        y_positions = np.arange(len(plot_df))
        ax.barh(y_positions, center, color="#1f5a91", alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Effect size")

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Top hits | {endpoint_label} | {stratum_label} | range={format_zone_label(zone_width)}")
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.78 if (plot_df["endpoint_type"] == "survival").any() else 0.97, top=0.94, bottom=0.07)
    return fig


def plot_family_summary(stats_df: pd.DataFrame, endpoint_label: str, zone_width: float, stratum_label: str = "All") -> Figure:
    plot_df = stats_df[
        (stats_df["endpoint_label"].astype(str) == str(endpoint_label))
        & (stats_df["zone_width"].astype(float) == float(zone_width))
        & (stats_df["stratum_label"].astype(str) == str(stratum_label))
    ].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No results for this endpoint / stratum.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    family_summary = (
        plot_df.groupby("feature_family", dropna=False)["significance_label"]
        .apply(lambda series: int(series.isin(["significant_fdr", "discovery_fdr"]).sum()))
        .reset_index(name="hit_count")
        .sort_values("hit_count", ascending=False)
    )
    ax.bar(family_summary["feature_family"].astype(str), family_summary["hit_count"].astype(int), color="#2a9d8f")
    ax.set_ylabel("Hits")
    ax.set_xlabel("Feature family")
    ax.set_title(f"Family summary | {endpoint_label} | {stratum_label}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    return fig


def plot_feature_stability_heatmap(stats_df: pd.DataFrame, feature_summary: pd.DataFrame, top_n: int = 25) -> Figure:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    if stats_df.empty or feature_summary.empty:
        ax.text(0.5, 0.5, "No feature stability results available.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    width_values = pd.to_numeric(stats_df.get("zone_width", pd.Series(dtype=float)), errors="coerce")
    ordered_widths = sorted({float(value) for value in width_values.dropna().tolist()})
    if not ordered_widths:
        ax.text(0.5, 0.5, "No range grid is available for the feature stability heatmap.", ha="center", va="center")
        ax.set_axis_off()
        return fig
    width_lookup = {_width_lookup_key(width): index for index, width in enumerate(ordered_widths)}

    ranked_features = feature_summary.copy()
    if "feature_rank" in ranked_features.columns:
        ranked_features = ranked_features.sort_values("feature_rank", ascending=True, na_position="last")
    ranked_features = ranked_features.head(max(1, int(top_n))).reset_index(drop=True)
    if ranked_features.empty:
        ax.text(0.5, 0.5, "No ranked features are available for the heatmap.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    heatmap = np.zeros((len(ranked_features), len(ordered_widths)), dtype=float)
    direction_grid = np.full(heatmap.shape, "", dtype=object)
    for row_index, feature_row in ranked_features.iterrows():
        feature_df = stats_df[
            (stats_df["endpoint_label"].astype(str) == str(feature_row.get("endpoint_label") or ""))
            & (stats_df["stratum_label"].astype(str) == str(feature_row.get("stratum_label") or ""))
            & (stats_df["feature_name"].astype(str) == str(feature_row.get("feature_name") or ""))
        ].copy()
        for _, stat_row in feature_df.iterrows():
            width_index = width_lookup.get(_width_lookup_key(stat_row.get("zone_width")), None)
            if width_index is None:
                continue
            status = _nominal_significance_status(stat_row)
            status_value = {
                "not_significant": 0.0,
                "nominal_p": 1.0,
                "discovery_fdr": 2.0,
                "significant_fdr": 3.0,
            }.get(status, 0.0)
            current_value = float(heatmap[row_index, int(width_index)])
            if float(status_value) > current_value:
                heatmap[row_index, int(width_index)] = float(status_value)
                direction_grid[row_index, int(width_index)] = str(stat_row.get("direction") or "")
            elif float(status_value) == current_value and not str(direction_grid[row_index, int(width_index)] or ""):
                direction_grid[row_index, int(width_index)] = str(stat_row.get("direction") or "")

    cmap = ListedColormap(["#f8fafc", "#fde68a", "#f4a261", "#1f5a91"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(heatmap, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(ordered_widths)))
    ax.set_xticklabels([format_zone_label(width) for width in ordered_widths], rotation=45, ha="right", fontsize=8)
    y_labels = [
        f"{str(row.get('feature_label') or row.get('feature_name') or 'Feature')} | {str(row.get('endpoint_label') or '')} | {str(row.get('stratum_label') or '')}"
        for _, row in ranked_features.iterrows()
    ]
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Range")
    ax.set_ylabel("Feature")
    ax.set_title("Feature stability across ranges")
    ax.set_xticks(np.arange(-0.5, len(ordered_widths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    arrow_fontsize = 9.0 if len(ordered_widths) <= 12 else 8.0 if len(ordered_widths) <= 18 else 7.0
    for row_index in range(heatmap.shape[0]):
        for col_index in range(heatmap.shape[1]):
            if float(heatmap[row_index, col_index]) <= 0.0:
                continue
            arrow_text = _direction_arrow_text(direction_grid[row_index, col_index])
            if not arrow_text:
                continue
            text_color = "white" if float(heatmap[row_index, col_index]) >= 2.5 else "#111827"
            ax.text(
                col_index,
                row_index,
                arrow_text,
                ha="center",
                va="center",
                fontsize=arrow_fontsize,
                color=text_color,
                zorder=3,
            )
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#f8fafc", markeredgecolor="#cbd5e1", markersize=9, label="Not significant"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#fde68a", markeredgecolor="#cbd5e1", markersize=9, label="Nominal p < 0.05"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#f4a261", markeredgecolor="#cbd5e1", markersize=9, label="FDR discovery"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#1f5a91", markeredgecolor="#cbd5e1", markersize=9, label="FDR significant"),
        Line2D([0], [0], linestyle="none", marker=r"$\uparrow$", color="#111827", markersize=10, label="Risk direction"),
        Line2D([0], [0], linestyle="none", marker=r"$\downarrow$", color="#111827", markersize=10, label="Protective direction"),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True, framealpha=0.95, fontsize=8)
    fig.subplots_adjust(left=0.30, right=0.79, bottom=0.24, top=0.90)
    return fig


def plot_feature_range_profile(stats_df: pd.DataFrame, feature_row: pd.Series) -> Figure:
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    if stats_df.empty or feature_row.empty:
        ax.text(0.5, 0.5, "No feature profile is available.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    feature_df = stats_df[
        (stats_df["endpoint_label"].astype(str) == str(feature_row.get("endpoint_label") or ""))
        & (stats_df["stratum_label"].astype(str) == str(feature_row.get("stratum_label") or ""))
        & (stats_df["feature_name"].astype(str) == str(feature_row.get("feature_name") or ""))
    ].copy()
    if feature_df.empty:
        ax.text(0.5, 0.5, "No feature profile is available.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    feature_df["zone_width"] = pd.to_numeric(feature_df["zone_width"], errors="coerce")
    feature_df["fdr_q"] = pd.to_numeric(feature_df["fdr_q"], errors="coerce")
    feature_df["p_value"] = pd.to_numeric(feature_df["p_value"], errors="coerce")
    feature_df = feature_df.dropna(subset=["zone_width"]).sort_values("zone_width").reset_index(drop=True)
    if feature_df.empty:
        ax.text(0.5, 0.5, "No plottable range profile is available.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    q_values = feature_df["fdr_q"].clip(lower=float(np.finfo(float).tiny))
    y_values = -np.log10(q_values)
    point_colors = feature_df.apply(
        lambda row: _directional_significance_color(row.get("direction"), _nominal_significance_status(row)),
        axis=1,
    )
    marker_sizes = feature_df["significance_label"].map(_significance_marker_size).fillna(22.0)

    ax.plot(feature_df["zone_width"], y_values, color="#64748b", linewidth=1.4, alpha=0.9)
    ax.scatter(feature_df["zone_width"], y_values, c=point_colors, s=marker_sizes, edgecolors="white", linewidths=0.5, zorder=3)

    ax.axhline(-np.log10(0.10), linestyle="--", color="#f4a261", linewidth=1.0, alpha=0.9)
    ax.axhline(-np.log10(0.05), linestyle="--", color="#1f5a91", linewidth=1.0, alpha=0.9)
    best_zone_width = _to_float(feature_row.get("best_zone_width"))
    if np.isfinite(best_zone_width):
        ax.axvline(float(best_zone_width), linestyle="--", color="#9f1d35", linewidth=1.6)

    ax.set_xticks(feature_df["zone_width"].tolist())
    ax.set_xticklabels([format_zone_label(value) for value in feature_df["zone_width"].tolist()], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("-log10(FDR q)")
    ax.set_xlabel("Range")
    ax.set_title(f"{str(feature_row.get('feature_label') or feature_row.get('feature_name') or 'Feature')} | {str(feature_row.get('endpoint_label') or '')} | {str(feature_row.get('stratum_label') or '')}")
    ax.grid(True, axis="y", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.26, top=0.90)
    return fig


def plot_feature_stability_publication_panel(
    stats_df: pd.DataFrame,
    feature_summary: pd.DataFrame,
    feature_row: pd.Series,
    top_n: int = 25,
) -> Figure:
    fig = plt.figure(figsize=(12.6, 9.6))
    if stats_df.empty or feature_summary.empty or feature_row.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No feature stability panel is available.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    grid = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.18)
    ax_heatmap = fig.add_subplot(grid[0])
    ax_profile = fig.add_subplot(grid[1])

    heatmap_fig = plot_feature_stability_heatmap(stats_df, feature_summary, top_n=top_n)
    heatmap_buffer = io.BytesIO(_figure_bytes(heatmap_fig, dpi=240))
    heatmap_buffer.seek(0)
    ax_heatmap.imshow(plt.imread(heatmap_buffer))
    ax_heatmap.set_axis_off()
    plt.close(heatmap_fig)

    profile_fig = plot_feature_range_profile(stats_df, feature_row)
    profile_buffer = io.BytesIO(_figure_bytes(profile_fig, dpi=240))
    profile_buffer.seek(0)
    ax_profile.imshow(plt.imread(profile_buffer))
    ax_profile.set_axis_off()
    plt.close(profile_fig)

    selected_label = str(feature_row.get("feature_label") or feature_row.get("feature_name") or "Feature")
    endpoint_label = str(feature_row.get("endpoint_label") or "")
    stratum_label = str(feature_row.get("stratum_label") or "")
    summary_parts = [
        f"Best range: {str(feature_row.get('best_zone_label') or '-')}",
        f"Reference band: {str(feature_row.get('reference_band_label') or '-')}",
        f"FDR band: {str(feature_row.get('significant_band_label') or '-')}",
    ]
    recovery_value = _safe_float(feature_row.get("feature_bootstrap_recovery"), float("nan"))
    if np.isfinite(recovery_value):
        summary_parts.append(f"Bootstrap recovery: {float(recovery_value):.2f}")
    fig.suptitle(f"Feature Stability Panel | {selected_label} | {endpoint_label} | {stratum_label}", fontsize=14, y=0.985)
    fig.text(0.5, 0.955, " | ".join(summary_parts), ha="center", va="top", fontsize=10, color="#334155")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.04)
    return fig


def plot_feature_endpoint(
    hit_row: pd.Series,
    patient_features_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
) -> Figure:
    payload = _prepare_feature_endpoint_payload(hit_row, patient_features_df, patient_outcomes_df, config)
    feature_df = payload.get("feature_df", pd.DataFrame())
    if feature_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No patient-level values available for this hit.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    endpoint_label = str(hit_row["endpoint_label"])
    endpoint_type = str(hit_row["endpoint_type"])
    merged = payload.get("plot_df", pd.DataFrame()).copy()
    attrition_summary = payload.get("attrition_summary", {}) if isinstance(payload.get("attrition_summary", {}), dict) else {}

    if endpoint_type == "survival":
        time_key = str(payload.get("time_key") or f"{endpoint_label}__time")
        cutpoint = payload.get("cutpoint")
        group_colors = payload.get("group_colors", {"High": SURVIVAL_RISK_COLOR, "Low": SURVIVAL_PROTECTIVE_COLOR})
        audit_df = payload.get("audit_df", pd.DataFrame()).copy()
        use_censor_as_censored = bool(payload.get("use_censor_as_censored", True))
        show_excluded_km_curve = bool(config.get("screening", {}).get("show_excluded_km_curve", False))
        group_order = [group for group in ["High", "Low"] if group in merged["group"].astype(str).unique().tolist()]
        time_unit = str(config["screening"].get("time_unit", "Time"))
        tick_series_parts: list[pd.Series] = []
        if time_key in merged.columns:
            tick_series_parts.append(pd.to_numeric(merged[time_key], errors="coerce"))
        if show_excluded_km_curve and "endpoint_time" in audit_df.columns:
            tick_series_parts.append(pd.to_numeric(audit_df["endpoint_time"], errors="coerce"))
        tick_source = pd.concat(tick_series_parts, ignore_index=True) if tick_series_parts else pd.Series(dtype=float)
        if merged.empty and tick_source.dropna().empty:
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            ax.text(0.5, 0.5, "No survival data available for this hit.", ha="center", va="center")
            ax.set_axis_off()
            return fig
        ticks = _build_survival_time_ticks(tick_source, time_unit)

        show_ci = bool(config.get("screening", {}).get("show_ci", False))

        fig = plt.figure(figsize=(13.2, 9.2))
        grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[4.3, 1.5], wspace=0.25, hspace=0.12)
        ax_km = fig.add_subplot(grid[0, 0])
        ax_risk = fig.add_subplot(grid[1, 0])
        ax_text = fig.add_subplot(grid[:, 1])
        ax_km.set_box_aspect(1.0)

        legend_handles: list[Line2D] = []
        median_survivals: dict[str, float] = {}
        risk_row_labels: list[str] = []
        risk_row_counts: list[list[str]] = []
        risk_row_colors: dict[str, str] = {}
        for group_name in group_order:
            group_df = merged[merged["group"].astype(str) == group_name].copy()
            if group_df.empty:
                continue
            durations = pd.to_numeric(group_df[time_key], errors="coerce")
            events = pd.to_numeric(group_df["event"], errors="coerce").fillna(0.0)
            valid_mask = np.isfinite(durations) & np.isfinite(events)
            durations = durations.loc[valid_mask]
            events = events.loc[valid_mask].clip(lower=0.0, upper=1.0)
            if durations.empty:
                continue
            kmf = KaplanMeierFitter()
            line_color = group_colors.get(group_name, NEUTRAL_COLOR)
            legend_label = f"{group_name} (n={len(durations)})"
            kmf.fit(durations, event_observed=events, label=legend_label)
            survival_df = kmf.survival_function_.reset_index()
            time_values = pd.to_numeric(survival_df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
            survival_values = pd.to_numeric(survival_df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
            ax_km.step(time_values, survival_values, where="post", color=line_color, linewidth=2.2)

            if show_ci:
                try:
                    ci_df = kmf.confidence_interval_survival_function_
                    if ci_df is not None and not ci_df.empty:
                        ci_time = ci_df.index.to_numpy(dtype=float)
                        ci_lower = ci_df.iloc[:, 0].to_numpy(dtype=float)
                        ci_upper = ci_df.iloc[:, 1].to_numpy(dtype=float)
                        ax_km.fill_between(ci_time, ci_lower, ci_upper, step="post", alpha=0.15, color=line_color)
                except Exception:
                    pass

            # --- Median survival ---
            median_time = kmf.median_survival_time_
            if np.isfinite(float(median_time)):
                median_survivals[group_name] = float(median_time)

            censor_times = durations.loc[events <= 0.0]
            if not censor_times.empty:
                censor_x = np.sort(censor_times.to_numpy(dtype=float))
                censor_y = np.asarray(kmf.predict(censor_x), dtype=float)
                ax_km.scatter(censor_x, censor_y, marker="|", s=90, linewidths=1.6, color=line_color, zorder=3)

            legend_handles.append(Line2D([0], [0], color=line_color, linewidth=2.2, label=legend_label))
            risk_row_labels.append(group_name)
            risk_row_counts.append([str(value) for value in _at_risk_counts(group_df[time_key], ticks)])
            risk_row_colors[group_name] = line_color

        excluded_curve_label = ""
        excluded_curve_count = 0
        if show_excluded_km_curve and not audit_df.empty:
            excluded_df = audit_df.loc[
                (~audit_df["used_in_test"].astype(bool))
                & np.isfinite(pd.to_numeric(audit_df.get("endpoint_time", pd.Series(np.nan, index=audit_df.index)), errors="coerce"))
                & np.isfinite(pd.to_numeric(audit_df.get("endpoint_censor", pd.Series(np.nan, index=audit_df.index)), errors="coerce")),
                :,
            ].copy()
            if not excluded_df.empty:
                excluded_df["event"] = 1 - pd.to_numeric(excluded_df["endpoint_censor"], errors="coerce") if use_censor_as_censored else pd.to_numeric(excluded_df["endpoint_censor"], errors="coerce")
                excluded_durations = pd.to_numeric(excluded_df["endpoint_time"], errors="coerce")
                excluded_events = pd.to_numeric(excluded_df["event"], errors="coerce").fillna(0.0)
                excluded_valid_mask = np.isfinite(excluded_durations) & np.isfinite(excluded_events)
                excluded_durations = excluded_durations.loc[excluded_valid_mask]
                excluded_events = excluded_events.loc[excluded_valid_mask].clip(lower=0.0, upper=1.0)
                if not excluded_durations.empty:
                    excluded_color = "#9ca3af"
                    excluded_curve_label = f"Excluded (n={len(excluded_durations)})"
                    excluded_curve_count = int(len(excluded_durations))
                    kmf_excluded = KaplanMeierFitter()
                    kmf_excluded.fit(excluded_durations, event_observed=excluded_events, label=excluded_curve_label)
                    excluded_survival_df = kmf_excluded.survival_function_.reset_index()
                    excluded_time_values = pd.to_numeric(excluded_survival_df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
                    excluded_survival_values = pd.to_numeric(excluded_survival_df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
                    ax_km.step(excluded_time_values, excluded_survival_values, where="post", color=excluded_color, linewidth=2.0, alpha=0.95)
                    excluded_censor_times = excluded_durations.loc[excluded_events <= 0.0]
                    if not excluded_censor_times.empty:
                        excluded_censor_x = np.sort(excluded_censor_times.to_numpy(dtype=float))
                        excluded_censor_y = np.asarray(kmf_excluded.predict(excluded_censor_x), dtype=float)
                        ax_km.scatter(excluded_censor_x, excluded_censor_y, marker="|", s=80, linewidths=1.3, color=excluded_color, zorder=3)
                    legend_handles.append(Line2D([0], [0], color=excluded_color, linewidth=2.0, label=excluded_curve_label))
                    median_time_excluded = kmf_excluded.median_survival_time_
                    if np.isfinite(float(median_time_excluded)):
                        median_survivals["Excluded"] = float(median_time_excluded)
                    risk_row_labels.append("Excluded")
                    risk_row_counts.append([str(value) for value in _at_risk_counts(excluded_df["endpoint_time"], ticks)])
                    risk_row_colors["Excluded"] = excluded_color

        ax_km.set_title(str(hit_row["feature_label"]))
        ax_km.set_xlabel("")
        ax_km.set_ylabel("Survival probability")
        ax_km.set_xticks(ticks)
        ax_km.set_xticklabels([_format_survival_tick(value, hide_zero=False) for value in ticks])
        ax_km.tick_params(axis="x", labelbottom=True, pad=2)
        ax_km.set_xlim(0.0, max(float(tick) for tick in ticks))
        ax_km.set_ylim(0.0, 1.02)
        ax_km.grid(False)
        ax_km.spines["top"].set_visible(False)
        ax_km.spines["right"].set_visible(False)

        ax_risk.axis("off")
        ax_risk.text(0.0, 0.56, "No. at Risk", transform=ax_risk.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
        risk_table = ax_risk.table(
            cellText=risk_row_counts,
            rowLabels=risk_row_labels,
            colLabels=[_format_survival_tick(value, hide_zero=True) for value in ticks],
            cellLoc="center",
            rowLoc="center",
            bbox=Bbox.from_bounds(0.0, -0.16, 1.0, 0.50),
        )
        risk_table.auto_set_font_size(False)
        risk_table.set_fontsize(10)
        risk_table.scale(1.0, 1.2)
        for (row_idx, col_idx), cell in risk_table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("white")
            if row_idx == 0:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color("#111827")
            if col_idx == -1 and row_idx > 0:
                group_name = risk_row_labels[row_idx - 1]
                cell.get_text().set_color(risk_row_colors.get(group_name, group_colors.get(group_name, NEUTRAL_COLOR)))
                cell.get_text().set_fontweight("bold")

        ax_text.axis("off")
        if legend_handles:
            ax_text.legend(
                handles=legend_handles,
                title=endpoint_label,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                borderaxespad=0.0,
                frameon=True,
                framealpha=0.95,
                fontsize=9,
                title_fontsize=10,
            )
        lines = [
            f"Endpoint: {endpoint_label}",
            f"Range: {str(hit_row.get('zone_label') or format_zone_label(hit_row.get('zone_width')))}",
            f"Stratum: {hit_row['stratum_label']}",
            f"p-value: {hit_row['p_value']:.3g}" if np.isfinite(hit_row["p_value"]) else "p-value: NA",
            f"FDR q: {hit_row['fdr_q']:.3g}" if np.isfinite(hit_row["fdr_q"]) else "FDR q: NA",
            f"Hazard ratio: {hit_row['hazard_ratio']:.3g}" if np.isfinite(hit_row["hazard_ratio"]) else "Hazard ratio: NA",
            f"95% CI: [{hit_row['ci_low']:.3g}, {hit_row['ci_high']:.3g}]" if np.isfinite(hit_row["ci_low"]) and np.isfinite(hit_row["ci_high"]) else "95% CI: NA",
            f"Cutpoint: {cutpoint:.3g}" if cutpoint is not None else "Cutpoint: NA",
        ]
        n_feature_patients = _safe_int(attrition_summary.get("n_feature_patients"), 0)
        n_missing_endpoint_patients = _safe_int(attrition_summary.get("n_missing_endpoint_patients"), 0)
        n_split_excluded_patients = _safe_int(attrition_summary.get("n_split_excluded_patients"), 0)
        n_final_patients = _safe_int(attrition_summary.get("n_final_patients"), 0)
        if n_feature_patients > 0:
            lines.append(f"Feature patients: {int(n_feature_patients)}")
        if n_missing_endpoint_patients > 0:
            lines.append(f"Dropped missing endpoint/value: {int(n_missing_endpoint_patients)}")
        if n_split_excluded_patients > 0:
            lines.append(f"Dropped by split: {int(n_split_excluded_patients)}")
        if n_final_patients > 0:
            lines.append(f"Used in test: {int(n_final_patients)}")
        if excluded_curve_count > 0:
            lines.append(f"Grey KM curve: {int(excluded_curve_count)} excluded patients with endpoint data")
        # Add concordance index if available
        c_idx = _safe_float(hit_row.get("concordance_index"), float("nan"))
        if np.isfinite(c_idx):
            lines.append(f"C-index: {c_idx:.3f}")
        # Add median survival per group
        for group_name in [*group_order, *(["Excluded"] if "Excluded" in median_survivals else [])]:
            if group_name in median_survivals:
                lines.append(f"Median ({group_name}): {median_survivals[group_name]:.1f}")
        ax_text.text(0.02, 0.73, "\n".join(lines), va="top", ha="left", fontsize=10)
        note_lines = []
        if show_ci:
            note_lines.append("Shaded bands show 95% confidence intervals.")
        note_lines.append("Vertical tick marks on the curves indicate censoring.")
        if excluded_curve_label:
            note_lines.append("Grey curve shows excluded patients with available survival data.")
        ax_text.text(
            0.02,
            0.18,
            "\n".join(note_lines),
            va="top",
            ha="left",
            fontsize=9,
            color="#475569",
        )
        fig.subplots_adjust(top=0.93, bottom=0.22, left=0.07, right=0.97, wspace=0.25, hspace=0.18)
        risk_bbox = ax_risk.get_position()
        fig.text((risk_bbox.x0 + risk_bbox.x1) / 2.0, risk_bbox.y0 - 0.06, time_unit, ha="center", va="top", fontsize=10.5, color="#111827")
        return fig

    if merged.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No endpoint groups available for this hit.", ha="center", va="center")
        ax.set_axis_off()
        return fig
    group_key = f"{endpoint_label}__group"
    merged = merged.dropna(subset=[group_key, "value"]).copy()
    if merged.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No endpoint groups available for this hit.", ha="center", va="center")
        ax.set_axis_off()
        return fig
    groups = _safe_unique(merged[group_key])
    data = [merged.loc[merged[group_key].astype(str) == group, "value"].astype(float).to_numpy(dtype=float) for group in groups]
    
    fig = plt.figure(figsize=(11.0, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.25)
    ax_dist = fig.add_subplot(grid[0])
    ax_roc = fig.add_subplot(grid[1])

    # --- Violin + Swarm Plot ---
    parts = ax_dist.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#81a1c1")
        pc.set_edgecolor("#4c566a")
        pc.set_alpha(0.4)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#2e3440")
        parts["cmedians"].set_linewidth(1.5)

    for idx, values in enumerate(data, start=1):
        jitter = np.random.default_rng(42).normal(0, 0.04, size=len(values))
        ax_dist.scatter(np.full(len(values), idx) + jitter, values, color="#1f5a91", alpha=0.6, s=18, edgecolors="white", linewidths=0.4)
        
    ax_dist.set_xticks(np.arange(1, len(groups) + 1))
    ax_dist.set_xticklabels(groups)
    ax_dist.set_title(f"Distribution | {endpoint_label}")
    ax_dist.set_ylabel("Patient feature value")
    ax_dist.grid(True, axis="y", alpha=0.2)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # --- ROC Curve ---
    if len(groups) == 2:
        try:
            from sklearn.metrics import roc_curve
            # We predict the first group (groups[0]) as the positive class
            y_true = (merged[group_key].astype(str) == groups[0]).astype(int)
            y_score = merged["value"].astype(float)
            # If the feature is lower in group 0, flip the score for ROC so AUC > 0.5
            if data[0].mean() < data[1].mean():
                y_score = -y_score
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = float(hit_row.get("auc", float("nan")))
            if not np.isfinite(roc_auc):
                roc_auc = float(roc_auc_score(y_true, y_score))
            
            ax_roc.plot(fpr, tpr, color="#d08770", lw=2, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], color="#4c566a", lw=1, linestyle="--", alpha=0.5)
            ax_roc.set_xlim([-0.02, 1.02])
            ax_roc.set_ylim([-0.02, 1.02])
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(f"ROC Curve ({groups[0]} vs {groups[1]})")
            ax_roc.legend(loc="lower right", frameon=True, fontsize=9)
            ax_roc.grid(True, alpha=0.15)
        except Exception as e:
            ax_roc.text(0.5, 0.5, f"ROC not available\n{e}", ha="center", va="center", fontsize=9, color="#4c566a")
            ax_roc.set_axis_off()
    else:
        ax_roc.text(0.5, 0.5, "ROC requires 2 groups", ha="center", va="center", fontsize=9, color="#4c566a")
        ax_roc.set_axis_off()

    fig.suptitle(str(hit_row['feature_label']), fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.95)
    return fig


def plot_top_hits_correlation(
    stats_df: pd.DataFrame,
    patient_features_by_width: dict[float, pd.DataFrame],
    selected_width: float,
    top_n: int = 20,
    endpoint_label: str | None = None,
    stratum_label: str = "All",
) -> Figure:
    """Spearman correlation heatmap of the top significant features at the selected width.

    This reveals whether top hits capture independent biology or are redundant
    (highly correlated).  Features are hierarchically clustered.
    """
    fig, ax = plt.subplots(figsize=(11.0, 9.0))
    if stats_df.empty or not patient_features_by_width:
        ax.text(0.5, 0.5, "No data available for correlation heatmap.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    patient_features = patient_features_by_width.get(float(selected_width), pd.DataFrame())
    if patient_features.empty:
        ax.text(0.5, 0.5, "No patient features for the selected width.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Filter stats to the selected width
    width_stats = stats_df[stats_df["zone_width"].astype(float) == float(selected_width)].copy()
    if endpoint_label:
        width_stats = width_stats[width_stats["endpoint_label"].astype(str) == str(endpoint_label)]
    if stratum_label:
        width_stats = width_stats[width_stats["stratum_label"].astype(str) == str(stratum_label)]
    width_stats = width_stats.sort_values(["fdr_q", "p_value"], na_position="last").head(max(1, int(top_n)))
    if len(width_stats) < 2:
        ax.text(0.5, 0.5, "Fewer than 2 features found; correlation heatmap requires at least 2.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Pivot patient features into a feature matrix
    feature_names = width_stats["feature_name"].unique().tolist()
    relevant = patient_features[patient_features["feature_name"].isin(feature_names)].copy()
    if relevant.empty:
        ax.text(0.5, 0.5, "No patient-level values for the selected features.", ha="center", va="center")
        ax.set_axis_off()
        return fig
    pivot = relevant.pivot_table(index="patient", columns="feature_name", values="value", aggfunc="first")
    # Drop features with insufficient variation
    pivot = pivot.dropna(axis=1, thresh=max(3, int(0.5 * len(pivot))))
    if pivot.shape[1] < 2:
        ax.text(0.5, 0.5, "Not enough non-missing features for correlation.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Compute Spearman correlation
    corr = pivot.corr(method="spearman")

    # Hierarchical clustering for ordering
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = squareform(1.0 - np.abs(corr.to_numpy(dtype=float)), checks=False)
        link = linkage(dist, method="average")
        order = leaves_list(link)
        corr = corr.iloc[order, order]
    except Exception:
        pass  # fall back to original order

    # Map feature names to labels
    name_to_label = dict(zip(width_stats["feature_name"].astype(str), width_stats["feature_label"].astype(str)))
    name_to_direction = dict(zip(width_stats["feature_name"].astype(str), width_stats.get("direction", pd.Series(dtype=str)).astype(str)))
    ordered_names = [str(name) for name in corr.columns]
    display_labels = [name_to_label.get(name, name) for name in ordered_names]
    display_directions = [name_to_direction.get(name, "") for name in ordered_names]
    n_features = len(display_labels)
    label_fontsize = 7 if n_features <= 12 else 6 if n_features <= 18 else 5
    score_fontsize = 6.0 if n_features <= 12 else 5.2 if n_features <= 18 else 4.5
    arrow_fontsize = score_fontsize + 1.1

    mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr_values = corr.to_numpy(dtype=float)
    masked_corr = np.ma.array(corr_values, mask=mask_upper)
    corr_cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.0, 1.0, 256)))
    corr_cmap.set_bad("#ffffff")
    im = ax.imshow(masked_corr, cmap=corr_cmap, vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=label_fontsize)
    ax.set_yticklabels(display_labels, fontsize=label_fontsize)
    ax.set_title(f"Feature Correlation | top {len(display_labels)} hits | {endpoint_label or 'All'} | {stratum_label}")
    ax.set_xticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate lower-triangle cells with the correlation score and feature-direction arrows.
    for i in range(len(display_labels)):
        row_arrow = _direction_arrow_text(display_directions[i])
        for j in range(i + 1):
            val = corr.iloc[i, j]
            if np.isfinite(val):
                text_color = "white" if abs(val) > 0.6 else "#111827"
                ax.text(j, i - 0.10, f"{val:.2f}", ha="center", va="center", fontsize=score_fontsize, color=text_color)
                col_arrow = _direction_arrow_text(display_directions[j])
                arrow_texts = [arrow for arrow in [row_arrow, col_arrow] if arrow]
                if i == j:
                    if row_arrow:
                        ax.text(j, i + 0.22, row_arrow, ha="center", va="center", fontsize=arrow_fontsize, color=text_color)
                elif len(arrow_texts) == 1:
                    ax.text(j, i + 0.22, arrow_texts[0], ha="center", va="center", fontsize=arrow_fontsize, color=text_color)
                else:
                    ax.text(j - 0.16, i + 0.22, row_arrow, ha="center", va="center", fontsize=arrow_fontsize, color=text_color)
                    ax.text(j + 0.16, i + 0.22, col_arrow, ha="center", va="center", fontsize=arrow_fontsize, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    fig.subplots_adjust(left=0.25, right=0.92, bottom=0.24, top=0.92)
    return fig


def parse_feature_components(feature_name: str) -> tuple[str, list[str]]:
    """Parse a feature name like ``triad::A::B::C`` into family and components."""
    parts = [str(part) for part in str(feature_name or "").split("::") if str(part)]
    if not parts:
        return "", []
    return str(parts[0]), [str(part) for part in parts[1:]]


def _safe_spearman_pair(x_values: pd.Series, y_values: pd.Series) -> tuple[float, float]:
    """Return ``(rho, p_value)`` from scipy.stats.spearmanr as plain floats."""
    try:
        result = stats.spearmanr(x_values, y_values, nan_policy="omit")
    except Exception:
        return float("nan"), float("nan")
    statistic = _to_float(getattr(result, "statistic", float("nan")), float("nan"))
    pvalue = _to_float(getattr(result, "pvalue", float("nan")), float("nan"))
    if np.isfinite(statistic) or np.isfinite(pvalue):
        return float(statistic), float(pvalue)
    if isinstance(result, tuple) and len(result) >= 2:
        return _to_float(result[0], float("nan")), _to_float(result[1], float("nan"))
    return float("nan"), float("nan")


def build_feature_pair_correlation_export_df(
    patient_features_df: pd.DataFrame,
    zone_width: float,
    stratum_label: str,
    target_feature_name: str,
    related_feature_name: str,
) -> pd.DataFrame:
    """Return patient-level paired values for a selected target/related feature pair."""
    if patient_features_df.empty:
        return pd.DataFrame()

    work = patient_features_df[
        (patient_features_df["zone_width"].astype(float) == float(zone_width))
        & (patient_features_df["stratum_label"].astype(str) == str(stratum_label))
        & (patient_features_df["feature_name"].astype(str).isin([str(target_feature_name), str(related_feature_name)]))
    ].copy()
    if work.empty:
        return pd.DataFrame()

    metadata_cols = [
        column
        for column in ["patient", "stratum_label", "zone_width", "zone_label", "n_regions"]
        if column in work.columns
    ]
    pivot = work.pivot_table(index=metadata_cols, columns="feature_name", values="value", aggfunc="first")
    if pivot.empty:
        return pd.DataFrame()
    pivot = pivot.reset_index().rename(
        columns={
            str(target_feature_name): "target_value",
            str(related_feature_name): "related_value",
        }
    )
    if "target_value" not in pivot.columns or "related_value" not in pivot.columns:
        return pd.DataFrame()
    return pivot.dropna(subset=["target_value", "related_value"]).copy()


def build_feature_influence_correlation_df(
    stats_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    zone_width: float,
    endpoint_label: str,
    stratum_label: str,
    target_feature_name: str,
    selected_components: list[str] | None = None,
    related_families: list[str] | None = None,
    require_all_components: bool = False,
    min_pairs: int = 5,
) -> pd.DataFrame:
    """Rank related features by patient-level Spearman correlation with a target feature.

    Relatedness is defined by phenotype-component overlap, so a target like
    ``triad::Fibroblast::IMC::Treg`` can be compared against density / proportion /
    ratio / dyad features that include one or more of those phenotypes.
    """
    if stats_df.empty or patient_features_df.empty:
        return pd.DataFrame()

    target_family, target_components = parse_feature_components(target_feature_name)
    component_pool = [str(value) for value in (selected_components or target_components) if str(value)]
    if not target_feature_name or not component_pool:
        return pd.DataFrame()

    allowed_families = {str(value) for value in (related_families or []) if str(value)}
    width_stats = stats_df[
        (stats_df["zone_width"].astype(float) == float(zone_width))
        & (stats_df["endpoint_label"].astype(str) == str(endpoint_label))
        & (stats_df["stratum_label"].astype(str) == str(stratum_label))
    ].copy()
    if width_stats.empty:
        return pd.DataFrame()
    width_stats = width_stats.sort_values(["fdr_q", "p_value"], na_position="last").reset_index(drop=True)
    width_stats["validation_rank"] = np.arange(1, len(width_stats) + 1)
    stats_lookup = width_stats.set_index("feature_name", drop=False)

    width_features = patient_features_df[
        (patient_features_df["zone_width"].astype(float) == float(zone_width))
        & (patient_features_df["stratum_label"].astype(str) == str(stratum_label))
    ].copy()
    if width_features.empty:
        return pd.DataFrame()

    target_df = width_features[width_features["feature_name"].astype(str) == str(target_feature_name)][["patient", "value"]].copy()
    if target_df.empty:
        return pd.DataFrame()
    target_df = target_df.rename(columns={"value": "target_value"})

    feature_meta = width_features[["feature_family", "feature_name", "feature_label"]].drop_duplicates(subset=["feature_name"]).copy()
    rows: list[dict[str, Any]] = []
    required_components = {str(value) for value in component_pool}
    min_pairs_int = max(3, int(min_pairs))

    for meta_row in feature_meta.itertuples(index=False):
        feature_family = str(meta_row.feature_family)
        feature_name = str(meta_row.feature_name)
        feature_label = str(meta_row.feature_label)
        if feature_name == str(target_feature_name):
            continue
        if allowed_families and feature_family not in allowed_families:
            continue

        _, candidate_components = parse_feature_components(feature_name)
        candidate_component_set = {str(value) for value in candidate_components}
        if not candidate_component_set:
            continue

        shared_components = sorted(required_components.intersection(candidate_component_set))
        if require_all_components:
            if not required_components.issubset(candidate_component_set):
                continue
        elif not shared_components:
            continue

        pair_df = build_feature_pair_correlation_export_df(
            width_features,
            zone_width=float(zone_width),
            stratum_label=str(stratum_label),
            target_feature_name=str(target_feature_name),
            related_feature_name=feature_name,
        )
        if len(pair_df) < min_pairs_int:
            continue

        target_values = pd.to_numeric(pair_df["target_value"], errors="coerce")
        related_values = pd.to_numeric(pair_df["related_value"], errors="coerce")
        valid_mask = np.isfinite(target_values) & np.isfinite(related_values)
        if int(valid_mask.sum()) < min_pairs_int:
            continue
        target_valid = target_values.loc[valid_mask]
        related_valid = related_values.loc[valid_mask]
        if target_valid.nunique(dropna=True) < 2 or related_valid.nunique(dropna=True) < 2:
            rho = float("nan")
            rho_p = float("nan")
        else:
            rho, rho_p = _safe_spearman_pair(target_valid, related_valid)

        stats_row = stats_lookup.loc[feature_name] if feature_name in stats_lookup.index else None
        if isinstance(stats_row, pd.DataFrame):
            stats_row = stats_row.iloc[0]
        rows.append(
            {
                "target_feature_family": target_family,
                "target_feature_name": str(target_feature_name),
                "related_feature_family": feature_family,
                "related_feature_name": feature_name,
                "related_feature_label": feature_label,
                "selected_components": ", ".join(component_pool),
                "shared_components": ", ".join(shared_components),
                "n_shared_components": int(len(shared_components)),
                "n_pairs": int(valid_mask.sum()),
                "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
                "spearman_p": float(rho_p) if np.isfinite(rho_p) else float("nan"),
                "abs_spearman_rho": abs(float(rho)) if np.isfinite(rho) else float("nan"),
                "validation_rank": int(_to_float(stats_row.get("validation_rank"), float("nan"))) if stats_row is not None else None,
                "p_value": float(_to_float(stats_row.get("p_value"), float("nan"))) if stats_row is not None else float("nan"),
                "fdr_q": float(_to_float(stats_row.get("fdr_q"), float("nan"))) if stats_row is not None else float("nan"),
                "significance_label": str(stats_row.get("significance_label") or "") if stats_row is not None else "",
                "direction": str(stats_row.get("direction") or "") if stats_row is not None else "",
            }
        )

    influence_df = pd.DataFrame(rows)
    if influence_df.empty:
        return influence_df
    influence_df = influence_df.sort_values(
        ["abs_spearman_rho", "spearman_p", "validation_rank", "related_feature_label"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return influence_df


def plot_feature_influence_barplot(
    influence_df: pd.DataFrame,
    target_feature_label: str,
    top_n: int = 15,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.42 * max(3, min(len(influence_df), int(top_n))) + 1.6)))
    if influence_df.empty:
        ax.text(0.5, 0.5, "No related features met the current influence-correlation filter.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    plot_df = influence_df.head(max(1, int(top_n))).copy().iloc[::-1].reset_index(drop=True)
    rho_values = pd.to_numeric(plot_df["spearman_rho"], errors="coerce")
    colors = ["#b45309" if np.isfinite(value) and value < 0 else "#1d4ed8" for value in rho_values]
    y_positions = np.arange(len(plot_df))
    ax.barh(y_positions, rho_values, color=colors, alpha=0.85)
    ax.axvline(0.0, color="#475569", linestyle=":", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["related_feature_label"].astype(str).tolist(), fontsize=8)
    ax.set_xlabel("Spearman ρ vs highlighted feature")
    ax.set_title(f"Potential Drivers | {target_feature_label}")
    ax.grid(True, axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.12, top=0.90)
    return fig


def plot_feature_influence_scatter(
    pair_df: pd.DataFrame,
    target_feature_label: str,
    related_feature_label: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    if pair_df.empty:
        ax.text(0.5, 0.5, "No paired patient values are available for this feature pair.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x_values = pd.to_numeric(pair_df.get("target_value", pd.Series(dtype=float)), errors="coerce")
    y_values = pd.to_numeric(pair_df.get("related_value", pd.Series(dtype=float)), errors="coerce")
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid_mask.sum()) < 3:
        ax.text(0.5, 0.5, "At least 3 paired patients are needed for the scatter view.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x_valid = x_values.loc[valid_mask]
    y_valid = y_values.loc[valid_mask]
    ax.scatter(x_valid, y_valid, s=34, color="#1f5a91", alpha=0.82, edgecolors="white", linewidths=0.4)
    if x_valid.nunique(dropna=True) >= 2:
        try:
            coeff = np.polyfit(x_valid.to_numpy(dtype=float), y_valid.to_numpy(dtype=float), 1)
            xs = np.linspace(float(x_valid.min()), float(x_valid.max()), 100)
            ys = coeff[0] * xs + coeff[1]
            ax.plot(xs, ys, color="#b45309", linewidth=1.6)
        except Exception:
            pass

    rho, rho_p = _safe_spearman_pair(x_valid, y_valid)
    rho_text = f"{rho:.2f}" if np.isfinite(rho) else "nan"
    p_text = f"{rho_p:.3g}" if np.isfinite(rho_p) else "nan"
    ax.set_xlabel(target_feature_label)
    ax.set_ylabel(related_feature_label)
    ax.set_title(f"Feature Pair Scatter\nSpearman ρ = {rho_text}, p = {p_text}")
    ax.grid(True, alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.90)
    return fig


def plot_feature_waterfall(
    hit_row: pd.Series,
    patient_features_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
) -> Figure:
    """Waterfall plot for binary endpoint hits.

    Patient-level feature values are sorted by magnitude and coloured by
    endpoint group (e.g. Responder / Non-responder).
    """
    payload = _prepare_feature_endpoint_payload(hit_row, patient_features_df, patient_outcomes_df, config)
    merged = payload.get("plot_df", pd.DataFrame()).copy()
    endpoint_label = str(hit_row["endpoint_label"])
    group_key = f"{endpoint_label}__group"

    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    if merged.empty or group_key not in merged.columns:
        ax.text(0.5, 0.5, "No data available for waterfall plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    merged = merged.dropna(subset=[group_key, "value"]).copy()
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged = merged.dropna(subset=["value"]).sort_values("value", ascending=False).reset_index(drop=True)
    if merged.empty:
        ax.text(0.5, 0.5, "No data available for waterfall plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    groups = _safe_unique(merged[group_key])
    color_palette = {"Positive": "#c43d3d", "Negative": "#2d6cdf"}
    if len(groups) == 2 and "Positive" not in groups:
        color_palette = {groups[0]: "#c43d3d", groups[1]: "#2d6cdf"}
    bar_colors = [color_palette.get(str(g), "#7f8c8d") for g in merged[group_key]]

    ax.bar(np.arange(len(merged)), merged["value"], color=bar_colors, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Patients (ranked)")
    ax.set_ylabel("Feature value")
    ax.set_title(f"Waterfall | {hit_row['feature_label']} | {endpoint_label}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color_palette.get(g, "#7f8c8d"),
               markersize=8, label=g)
        for g in groups
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Representative ROI — GeoJSON cell segmentation & zone-boundary helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_geojson_for_batch(
    root_dir: str | Path,
    batch_id: str,
    preferred_suffixes: list[str] | None = None,
) -> Path | None:
    """Return the best-matching GeoJSON segmentation file for *batch_id*.

    Searches *root_dir* recursively for files whose name starts with *batch_id*
    (case-insensitive) and ends with ``.geojson``.  When multiple candidates are
    found, files whose suffix matches one of *preferred_suffixes* (in order) are
    ranked first; otherwise alphabetical order is used.
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        return None

    lower_batch = str(batch_id).lower()
    matches = sorted(
        (f for f in root.rglob("*.geojson") if f.name.lower().startswith(lower_batch)),
        key=lambda f: str(f),
    )
    if not matches:
        return None

    _preferred = preferred_suffixes or [
        ".ome_cell_boundaries.geojson",
        "_phenotype_ml_consensus.geojson",
        "_phenotype_leiden.geojson",
    ]

    def _rank(f: Path) -> int:
        lower_name = f.name.lower()
        for i, suf in enumerate(_preferred):
            if lower_name.endswith(suf.lower()):
                return i
        return len(_preferred) + 1

    matches.sort(key=lambda f: (_rank(f), str(f)))
    return matches[0]


def _parse_geojson_polygon_features(
    geojson_path: Path,
    flip_y: bool = False,
    image_height: float = 0.0,
) -> list[dict[str, Any]]:
    """Parse a GeoJSON file and return a list of polygon descriptors.

    Each entry is a dict with:
    ``outer_ring`` (list[[x,y]]), ``centroid_x``, ``centroid_y``, ``label`` (str).
    Supports Polygon, MultiPolygon, and FeatureCollection root types.
    """
    import json as _json

    try:
        with open(geojson_path, "r", encoding="utf-8") as _fh:
            root = _json.load(_fh)
    except Exception:
        return []

    raw_features: list[dict[str, Any]] = []
    if isinstance(root, dict):
        gtype = str(root.get("type") or "")
        if gtype == "FeatureCollection":
            raw_features = [f for f in (root.get("features") or []) if isinstance(f, dict)]
        elif gtype == "Feature":
            raw_features = [root]
        elif gtype in ("Polygon", "MultiPolygon"):
            raw_features = [{"type": "Feature", "geometry": root, "properties": {}}]

    results: list[dict[str, Any]] = []
    for feature in raw_features:
        geom = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        gtype = str(geom.get("type") or "")
        coords = geom.get("coordinates")
        if not coords:
            continue

        polys: list[list[list[float]]] = []  # list of outer_rings
        if gtype == "Polygon":
            if coords and len(coords[0]) >= 3:
                polys.append(coords[0])
        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                if poly_coords and len(poly_coords[0]) >= 3:
                    polys.append(poly_coords[0])

        label = str(
            props.get("label") or props.get("classification") or props.get("name") or ""
        )

        for ring in polys:
            if len(ring) < 3:
                continue
            if flip_y and image_height > 0:
                ring = [[c[0], image_height - c[1]] for c in ring]
            arr = np.array(ring, dtype=float)
            cx = float(arr[:, 0].mean())
            cy = float(arr[:, 1].mean())
            results.append({
                "outer_ring": ring,
                "centroid_x": cx,
                "centroid_y": cy,
                "label": label,
            })

    return results


def load_geojson_cell_segmentation(
    geojson_path: str | Path,
    zone_df: pd.DataFrame,
    config: dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    flip_y: bool = False,
    max_match_distance: float = 50.0,
) -> list[dict[str, Any]]:
    """Load GeoJSON cell polygons and match them to cells in *zone_df* by proximity.

    Each polygon's centroid is matched to the nearest CSV cell within
    *max_match_distance* pixels (in CSV coordinate space).  The matched cell's
    phenotype label is attached to the polygon.

    Returns a list of ``{coords: np.ndarray(N,2), phenotype: str}`` dicts where
    ``coords`` are in display space (scaled by *scale_x* / *scale_y*).
    """
    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])
    phenotype_col = str(config["columns"]["phenotype"])

    gj_features = _parse_geojson_polygon_features(Path(geojson_path), flip_y=flip_y)
    if not gj_features:
        return []

    csv_xs = pd.to_numeric(zone_df[x_col], errors="coerce").values
    csv_ys = pd.to_numeric(zone_df[y_col], errors="coerce").values
    phenos = zone_df[phenotype_col].astype(str).values
    valid = np.isfinite(csv_xs) & np.isfinite(csv_ys)
    if not bool(valid.any()):
        return []

    csv_pts = np.column_stack([csv_xs[valid], csv_ys[valid]])
    csv_orig_idx = np.where(valid)[0]
    tree = cKDTree(csv_pts)

    gj_centroids = np.array([[f["centroid_x"], f["centroid_y"]] for f in gj_features])
    dists, nn_idxs = tree.query(gj_centroids, k=1, distance_upper_bound=float(max_match_distance))

    results: list[dict[str, Any]] = []
    for feat, dist, nn_i in zip(gj_features, dists, nn_idxs):
        if not np.isfinite(dist):
            continue
        phenotype = str(phenos[csv_orig_idx[nn_i]])
        ring = np.array(feat["outer_ring"], dtype=float)
        if ring.shape[1] < 2 or len(ring) < 3:
            continue
        coords = ring[:, :2].copy()
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
        results.append({"coords": coords, "phenotype": phenotype})

    return results


def build_voronoi_cell_segmentation(
    zone_df: pd.DataFrame,
    config: dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    cap_percentile: float = 50.0,
    fallback_radius: float = 15.0,
) -> list[dict[str, Any]]:
    """Build Voronoi-tessellation cell polygons from CSV centroids.

    Each Voronoi region is capped to a circle of radius determined by the
    *cap_percentile*-th percentile of nearest-neighbour distances (or
    *fallback_radius* when not enough points are available).

    Returns a list of ``{coords: np.ndarray(N,2), phenotype: str}`` in display space.
    """
    from scipy.spatial import Voronoi as _Voronoi

    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])
    phenotype_col = str(config["columns"]["phenotype"])

    xs_raw = pd.to_numeric(zone_df[x_col], errors="coerce").values
    ys_raw = pd.to_numeric(zone_df[y_col], errors="coerce").values
    phenos = zone_df[phenotype_col].astype(str).values
    valid = np.isfinite(xs_raw) & np.isfinite(ys_raw)
    if valid.sum() < 4:
        return build_circle_cell_segmentation(zone_df, config, scale_x, scale_y, fallback_radius)

    pts_x = xs_raw[valid] * scale_x
    pts_y = ys_raw[valid] * scale_y
    pts = np.column_stack([pts_x, pts_y])
    phens_v = phenos[valid]

    # Nearest-neighbour cap radius
    nn_tree = cKDTree(pts)
    nn_d, _ = nn_tree.query(pts, k=min(2, len(pts)))
    if nn_d.ndim == 2:
        nn_d = nn_d[:, 1]
    valid_nn = np.isfinite(nn_d) & (nn_d > 0)
    cap_r = (
        float(np.percentile(nn_d[valid_nn], float(cap_percentile)))
        if bool(valid_nn.any())
        else fallback_radius
    )
    cap_r = max(cap_r, fallback_radius)

    # Mirror-point boundary so all Voronoi regions are finite
    pad = cap_r * 2.0
    mn_x, mx_x = pts_x.min() - pad, pts_x.max() + pad
    mn_y, mx_y = pts_y.min() - pad, pts_y.max() + pad
    mid_x, mid_y = (mn_x + mx_x) / 2.0, (mn_y + mx_y) / 2.0
    mirror = np.array([
        [mn_x, mn_y], [mx_x, mn_y], [mn_x, mx_y], [mx_x, mx_y],
        [mid_x, mn_y], [mid_x, mx_y], [mn_x, mid_y], [mx_x, mid_y],
    ])
    all_pts = np.vstack([pts, mirror])
    n_real = len(pts)

    try:
        vor = _Voronoi(all_pts)
    except Exception:
        return build_circle_cell_segmentation(zone_df, config, scale_x, scale_y, cap_r)

    _theta = np.linspace(0.0, 2.0 * np.pi, 20)
    results: list[dict[str, Any]] = []

    for pt_i in range(n_real):
        region_i = vor.point_region[pt_i]
        region = vor.regions[region_i]
        cx, cy = pts[pt_i]

        if not region or -1 in region:
            # Infinite region — use cap circle
            coords = np.column_stack([cx + cap_r * np.cos(_theta), cy + cap_r * np.sin(_theta)])
        else:
            poly = vor.vertices[region]
            # Cap vertices that are too far from the cell centre
            dfc = np.sqrt((poly[:, 0] - cx) ** 2 + (poly[:, 1] - cy) ** 2)
            overshoot = dfc > cap_r
            if bool(overshoot.any()):
                angles = np.arctan2(poly[:, 1] - cy, poly[:, 0] - cx)
                poly[overshoot, 0] = cx + cap_r * np.cos(angles[overshoot])
                poly[overshoot, 1] = cy + cap_r * np.sin(angles[overshoot])
            coords = poly

        if len(coords) < 3:
            continue
        results.append({"coords": coords, "phenotype": str(phens_v[pt_i])})

    return results


def build_circle_cell_segmentation(
    zone_df: pd.DataFrame,
    config: dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    radius_px: float = 5.0,
) -> list[dict[str, Any]]:
    """Build circle-based cell polygon approximations from CSV centroids.

    Returns a list of ``{coords: np.ndarray(N,2), phenotype: str}`` in display space.
    """
    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])
    phenotype_col = str(config["columns"]["phenotype"])

    xs = pd.to_numeric(zone_df[x_col], errors="coerce").values
    ys = pd.to_numeric(zone_df[y_col], errors="coerce").values
    phenos = zone_df[phenotype_col].astype(str).values
    valid = np.isfinite(xs) & np.isfinite(ys)

    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    results: list[dict[str, Any]] = []
    for i in np.where(valid)[0]:
        cx = float(xs[i]) * scale_x
        cy = float(ys[i]) * scale_y
        results.append({
            "coords": np.column_stack([cx + radius_px * cos_t, cy + radius_px * sin_t]),
            "phenotype": str(phenos[i]),
        })
    return results


def _draw_cell_segmentation_patches(
    ax: "Axes",
    cell_features: list[dict[str, Any]],
    color_mapping: dict[str, str],
    fill_opacity: float = 1.0,
    outline_opacity: float | None = None,
    unclassified_color: str = "#9e9e9e",
    phenotype_filter: set[str] | None = None,
) -> set[str]:
    """Draw cell polygon patches on *ax*.

    Returns the set of phenotype labels that were actually drawn.  Pass
    *phenotype_filter* to restrict drawing to a subset of phenotypes.
    """
    import matplotlib.patches as _mpatches
    import matplotlib.colors as _mcolors

    if not cell_features:
        return set()

    _outline_opacity = outline_opacity if outline_opacity is not None else min(1.0, fill_opacity + 0.15)
    plotted: set[str] = set()

    for feat in cell_features:
        coords = feat.get("coords")
        if coords is None or len(coords) < 3:
            continue
        phenotype = str(feat.get("phenotype") or "")
        if phenotype_filter is not None and phenotype not in phenotype_filter:
            continue
        fill_hex = color_mapping.get(phenotype, unclassified_color)
        try:
            r, g, b, _ = _mcolors.to_rgba(fill_hex)
        except Exception:
            r, g, b = 0.55, 0.55, 0.55

        patch = _mpatches.Polygon(
            coords,
            closed=True,
            facecolor=(r, g, b, fill_opacity),
            edgecolor=(r * 0.65, g * 0.65, b * 0.65, _outline_opacity),
            linewidth=0.5,
            zorder=2,
        )
        ax.add_patch(patch)
        plotted.add(phenotype)

    return plotted


def _draw_zone_distance_contour(
    ax: "Axes",
    df: pd.DataFrame,
    config: dict[str, Any],
    distance_col: str,
    zone_start: float,
    zone_end: float,
    color: str = "#1f4e79",
    linewidth: float = 1.5,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    grid_resolution: int = 150,
) -> None:
    """Draw interpolated contour iso-lines at *zone_start* and *zone_end* distances.

    Uses ``scipy.interpolate.griddata`` (linear) to create a smooth distance
    field from discrete cell positions and then calls ``ax.contour`` to draw
    iso-lines — giving a publication-quality zone boundary even when cells are
    sparse.

    Does nothing if the distance column is missing or has too few valid values.
    """
    if distance_col not in df.columns:
        return

    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])

    xs = pd.to_numeric(df[x_col], errors="coerce").values * scale_x  # type: ignore[operator]
    ys = pd.to_numeric(df[y_col], errors="coerce").values * scale_y  # type: ignore[operator]
    dists = pd.to_numeric(df[distance_col], errors="coerce").values

    valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(dists)
    if valid.sum() < 10:
        return

    xs_v, ys_v, dists_v = xs[valid], ys[valid], dists[valid]

    pad_x = max((xs_v.max() - xs_v.min()) * 0.02, 1.0)
    pad_y = max((ys_v.max() - ys_v.min()) * 0.02, 1.0)
    xi = np.linspace(xs_v.min() - pad_x, xs_v.max() + pad_x, int(grid_resolution))
    yi = np.linspace(ys_v.min() - pad_y, ys_v.max() + pad_y, int(grid_resolution))
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        from scipy.interpolate import griddata as _griddata
        Zi = _griddata((xs_v, ys_v), dists_v, (Xi, Yi), method="linear")
    except Exception:
        return

    if Zi is None or bool(np.all(np.isnan(Zi))):
        return

    # Only draw levels that fall inside the data range
    d_min, d_max = float(dists_v.min()), float(dists_v.max())
    levels = sorted({
        v for v in (float(zone_start), float(zone_end))
        if np.isfinite(v) and (d_min - 5.0) <= v <= (d_max + 5.0)
    })
    if not levels:
        return

    try:
        ax.contour(
            Xi, Yi, Zi,
            levels=levels,
            colors=[color] * len(levels),
            linewidths=[linewidth] * len(levels),
            linestyles=["--"] * len(levels),
            zorder=6,
        )
    except Exception:
        pass


def plot_cell_segmentation_figure(
    region_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    cell_features: list[dict[str, Any]],
    color_mapping: dict[str, str],
    image: "np.ndarray | None",
    scale_x: float,
    scale_y: float,
    config: dict[str, Any],
    settings: dict[str, Any],
    title: str,
    zone_start: float,
    zone_end: float,
    *,
    overlay_on_image: bool = False,
    fill_opacity: float = 1.0,
    show_zone_contour: bool = False,
    distance_col: str = "neighborhood_distance_tumor",
    phenotype_filter: set[str] | None = None,
) -> Figure:
    """Render cell segmentation polygons coloured by phenotype.

    Parameters
    ----------
    overlay_on_image:
        When ``True``, draw translucent phenotype polygons on top of the marker
        image (equivalent to the Groovy ``exportPhenotypeOverlayOnMerged``).
        When ``False``, render polygons on a dark canvas (or the image if
        ``show_background`` is set in *settings*), reproducing
        ``exportPhenotypeSegmentation``.
    show_zone_contour:
        When ``True``, overlay iso-lines at *zone_start* / *zone_end* using the
        *distance_col* continuous values from *region_df* via
        ``_draw_zone_distance_contour``.  Falls back to the convex-hull overlay
        when the column is absent.
    phenotype_filter:
        If given, only polygons whose phenotype is in this set are drawn.
    """
    import matplotlib.patches as _mpatches  # noqa: F401 — needed for legend handles

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    show_bg = bool(settings.get("show_background", True))
    if image is not None and (show_bg or overlay_on_image):
        ax.imshow(image)
    elif not overlay_on_image:
        ax.set_facecolor("#111111")

    plotted = _draw_cell_segmentation_patches(
        ax,
        cell_features,
        color_mapping,
        fill_opacity=fill_opacity,
        phenotype_filter=phenotype_filter,
    )

    # ── Zone boundary ─────────────────────────────────────────────────────────
    zone_color = str(settings.get("zone_color", "#1f4e79"))
    zone_lw = float(settings.get("zone_linewidth", 1.5))
    if show_zone_contour and distance_col in region_df.columns:
        _draw_zone_distance_contour(
            ax,
            region_df,
            config,
            distance_col=distance_col,
            zone_start=zone_start,
            zone_end=zone_end,
            color=zone_color,
            linewidth=zone_lw,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    elif bool(settings.get("show_zone_overlay", True)):
        from scipy.spatial import ConvexHull as _CH
        x_col = str(config["columns"]["x"])
        y_col = str(config["columns"]["y"])
        zone_plot_x = pd.to_numeric(zone_df.get(x_col, pd.Series(dtype=float)), errors="coerce") * scale_x
        zone_plot_y = pd.to_numeric(zone_df.get(y_col, pd.Series(dtype=float)), errors="coerce") * scale_y
        zone_pts = np.column_stack([zone_plot_x, zone_plot_y])
        zone_pts = zone_pts[np.isfinite(zone_pts).all(axis=1)]
        if len(zone_pts) >= 3:
            try:
                hull = _CH(zone_pts)
                hull_pts = zone_pts[hull.vertices]
                ax.add_patch(
                    _mpatches.Polygon(
                        hull_pts,
                        closed=True,
                        facecolor=(0, 0, 0, 0),
                        edgecolor=zone_color,
                        linewidth=zone_lw,
                        linestyle="--",
                        zorder=7,
                    )
                )
            except Exception:
                pass

    # ── Set axis limits when there is no image ────────────────────────────────
    if image is None or not (show_bg or overlay_on_image):
        all_x = [c[0] for f in cell_features if f.get("coords") is not None for c in f["coords"]]
        all_y = [c[1] for f in cell_features if f.get("coords") is not None for c in f["coords"]]
        if all_x:
            pad = max(20.0, 0.05 * max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0))
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(max(all_y) + pad, min(all_y) - pad)
            ax.set_aspect("equal")

    # ── Legend ────────────────────────────────────────────────────────────────
    if bool(settings.get("show_legend", True)) and plotted:
        from ..tab7_phenotyping.cell_visualization import (
            _add_unified_legend as _add_legend,
        )
        _add_legend(fig, ax, plotted, color_mapping, title="Phenotype")

    ax.set_axis_off()
    if bool(settings.get("show_title", True)):
        ax.set_title(title, loc="left", fontsize=11)
    annotation_text = str(settings.get("annotation_text", "")).strip()
    if annotation_text:
        fig.text(0.985, 0.02, annotation_text, ha="right", va="bottom", fontsize=8)
    fig.subplots_adjust(
        left=0.03,
        right=0.76 if bool(settings.get("show_legend", True)) else 0.98,
        bottom=0.04,
        top=0.92,
    )
    return fig


def build_qc_summary(
    results: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured QC summary from pipeline results.

    Returns a dictionary with diagnostic keys suitable for display
    in the UI's QC tab.
    """
    analysis_df = results.get("analysis_df", pd.DataFrame())
    patient_outcomes = results.get("patient_outcomes", pd.DataFrame())
    candidate_summary = results.get("candidate_summary", pd.DataFrame())
    stats_all = results.get("stats_all", pd.DataFrame())
    feature_summary = results.get("feature_stability_summary", pd.DataFrame())

    qc: dict[str, Any] = {}

    # Dataset overview
    qc["total_cells"] = int(len(analysis_df)) if not analysis_df.empty else 0
    qc["total_patients"] = int(patient_outcomes["patient"].nunique()) if not patient_outcomes.empty else 0
    if not analysis_df.empty and "batch" in analysis_df.columns and "roi" in analysis_df.columns:
        qc["total_regions"] = int(analysis_df.groupby(["batch", "roi"]).ngroups)
    else:
        qc["total_regions"] = 0

    # Cells per patient distribution
    if not analysis_df.empty and "patient" in analysis_df.columns:
        cells_per_patient = analysis_df.groupby("patient").size()
        qc["cells_per_patient_median"] = float(cells_per_patient.median())
        qc["cells_per_patient_min"] = int(cells_per_patient.min())
        qc["cells_per_patient_max"] = int(cells_per_patient.max())
        qc["cells_per_patient_q25"] = float(cells_per_patient.quantile(0.25))
        qc["cells_per_patient_q75"] = float(cells_per_patient.quantile(0.75))
    else:
        qc["cells_per_patient_median"] = float("nan")
        qc["cells_per_patient_min"] = 0
        qc["cells_per_patient_max"] = 0

    # Screening summary
    qc["total_widths_scanned"] = int(len(candidate_summary)) if not candidate_summary.empty else 0
    qc["widths_with_fdr_hits"] = int((pd.to_numeric(candidate_summary.get("n_significant", pd.Series(dtype=int)), errors="coerce").fillna(0) > 0).sum()) if not candidate_summary.empty else 0
    qc["total_features_tested"] = int(len(stats_all)) if not stats_all.empty else 0

    if not stats_all.empty:
        qc["total_fdr_significant"] = int((stats_all["significance_label"] == "significant_fdr").sum())
        qc["total_fdr_discovery"] = int((stats_all["significance_label"].isin(["significant_fdr", "discovery_fdr"])).sum())
        qc["total_nominal_p"] = int((pd.to_numeric(stats_all["p_value"], errors="coerce") <= 0.05).sum())
    else:
        qc["total_fdr_significant"] = 0
        qc["total_fdr_discovery"] = 0
        qc["total_nominal_p"] = 0

    # Cox convergence rate
    if not stats_all.empty and "concordance_index" in stats_all.columns:
        survival_stats = stats_all[stats_all["endpoint_type"].astype(str) == "survival"]
        n_survival_tests = len(survival_stats)
        n_converged = int(pd.to_numeric(survival_stats["concordance_index"], errors="coerce").notna().sum())
        qc["cox_total_tests"] = n_survival_tests
        qc["cox_converged"] = n_converged
        qc["cox_convergence_rate"] = f"{100.0 * n_converged / max(n_survival_tests, 1):.1f}%"
    else:
        qc["cox_total_tests"] = 0
        qc["cox_converged"] = 0
        qc["cox_convergence_rate"] = "N/A"

    # Bootstrap diagnostics
    if not candidate_summary.empty and "bootstrap_valid_iterations" in candidate_summary.columns:
        valid_iters = pd.to_numeric(candidate_summary["bootstrap_valid_iterations"], errors="coerce").dropna()
        qc["bootstrap_widths_tested"] = int((valid_iters > 0).sum())
        qc["bootstrap_mean_iterations"] = float(valid_iters[valid_iters > 0].mean()) if (valid_iters > 0).any() else 0.0
    else:
        qc["bootstrap_widths_tested"] = 0
        qc["bootstrap_mean_iterations"] = 0.0

    # Feature stability
    if not feature_summary.empty:
        qc["total_stable_features"] = int(len(feature_summary))
        recovery = pd.to_numeric(feature_summary.get("feature_bootstrap_recovery", pd.Series(dtype=float)), errors="coerce")
        qc["features_with_high_recovery"] = int((recovery >= 0.8).sum())
        qc["mean_bootstrap_recovery"] = float(recovery.mean()) if recovery.notna().any() else float("nan")
    else:
        qc["total_stable_features"] = 0
        qc["features_with_high_recovery"] = 0
        qc["mean_bootstrap_recovery"] = float("nan")

    # Endpoint group balance
    if not patient_outcomes.empty:
        endpoint_balance: list[dict[str, Any]] = []
        for col in patient_outcomes.columns:
            if col.endswith("__group"):
                counts = patient_outcomes[col].dropna().value_counts()
                endpoint_balance.append({
                    "endpoint": col.replace("__group", ""),
                    "groups": counts.to_dict(),
                    "balance_ratio": float(counts.min()) / float(max(counts.max(), 1)),
                })
        qc["endpoint_balance"] = endpoint_balance
    else:
        qc["endpoint_balance"] = []

    return qc

def save_feature_screening_run(
    results: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | Path,
    log_lines: list[str] | None = None,
    log_fn: LogFn = None,
) -> Path:
    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = f"feature_screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_name
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    hit_tables_dir = tables_dir / "hit_patient_csvs"
    hit_figures_dir = figures_dir / "hit_plots"
    logs_dir = run_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    hit_tables_dir.mkdir(parents=True, exist_ok=True)
    hit_figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary = results["candidate_summary"]
    feature_stability_summary = results.get("feature_stability_summary", pd.DataFrame()).copy()
    stats_all = results["stats_all"]
    selected_width = float(results["selected_width"])
    patient_features_all = pd.concat(results["patient_features_by_width"].values(), ignore_index=True) if results["patient_features_by_width"] else pd.DataFrame()
    selected_combined = results["combined_by_width"].get(selected_width, pd.DataFrame()).copy()

    results["distance_summary"].to_csv(tables_dir / "distance_distribution.csv", index=False)
    candidate_summary.to_csv(tables_dir / "candidate_summary.csv", index=False)
    feature_export_df = feature_stability_summary.drop(columns=[column for column in feature_stability_summary.columns if str(column).endswith("_indices")], errors="ignore")
    if not feature_export_df.empty:
        feature_export_df.to_csv(tables_dir / "feature_stability_summary.csv", index=False)
    stats_all.to_csv(tables_dir / "screening_results_all.csv", index=False)
    results["patient_outcomes"].to_csv(tables_dir / "patient_outcomes.csv", index=False)
    patient_features_all.to_csv(tables_dir / "patient_features_all.csv", index=False)
    selected_combined.to_csv(tables_dir / "combined_features_selected_width.csv", index=False)

    candidate_fig = plot_candidate_scan(candidate_summary, selected_width)
    (figures_dir / "candidate_scan.png").write_bytes(_figure_bytes(candidate_fig, dpi=300))
    plt.close(candidate_fig)

    if not feature_stability_summary.empty and not stats_all.empty:
        heatmap_top_n = max(1, _safe_int(config["screening"].get("feature_stability_top_n"), 25))
        feature_heatmap = plot_feature_stability_heatmap(stats_all, feature_stability_summary, top_n=heatmap_top_n)
        (figures_dir / "feature_stability_heatmap.png").write_bytes(_figure_bytes(feature_heatmap, dpi=300))
        plt.close(feature_heatmap)

        ranked_feature_summary = feature_stability_summary.copy()
        if "feature_rank" in ranked_feature_summary.columns:
            ranked_feature_summary = ranked_feature_summary.sort_values("feature_rank", ascending=True, na_position="last")
        top_feature_row = ranked_feature_summary.head(1)
        if not top_feature_row.empty:
            feature_profile = plot_feature_range_profile(stats_all, top_feature_row.iloc[0])
            (figures_dir / "top_feature_range_profile.png").write_bytes(_figure_bytes(feature_profile, dpi=300))
            plt.close(feature_profile)

            feature_panel = plot_feature_stability_publication_panel(stats_all, feature_stability_summary, top_feature_row.iloc[0], top_n=heatmap_top_n)
            (figures_dir / "top_feature_stability_panel.png").write_bytes(_figure_bytes(feature_panel, dpi=300))
            plt.close(feature_panel)

    if not stats_all.empty:
        for (endpoint_label, stratum_label), _ in stats_all.groupby(["endpoint_label", "stratum_label"], dropna=False):
            volcano = plot_volcano(stats_all, str(endpoint_label), selected_width, str(stratum_label))
            (figures_dir / f"volcano_{_slugify(endpoint_label)}_{_slugify(stratum_label)}.png").write_bytes(_figure_bytes(volcano, dpi=300))
            plt.close(volcano)

            forest = plot_forest(stats_all, str(endpoint_label), selected_width, str(stratum_label))
            (figures_dir / f"forest_{_slugify(endpoint_label)}_{_slugify(stratum_label)}.png").write_bytes(_figure_bytes(forest, dpi=300))
            plt.close(forest)

            family = plot_family_summary(stats_all, str(endpoint_label), selected_width, str(stratum_label))
            (figures_dir / f"family_summary_{_slugify(endpoint_label)}_{_slugify(stratum_label)}.png").write_bytes(_figure_bytes(family, dpi=300))
            plt.close(family)

        top_hit_limit = max(1, _safe_int(config["screening"].get("top_n_km_plots"), 20))
        selected_hits = stats_all.loc[stats_all["zone_width"].astype(float) == float(selected_width)].sort_values(["fdr_q", "p_value"], na_position="last").head(top_hit_limit).copy()
        for _, hit_row in selected_hits.iterrows():
            width = float(hit_row["zone_width"])
            patient_features = results["patient_features_by_width"].get(width, pd.DataFrame())
            if patient_features.empty:
                continue
            hit_stub = "_".join(
                [
                    _slugify(hit_row["endpoint_label"]),
                    _slugify(hit_row["feature_name"]),
                    _slugify(hit_row["stratum_label"]),
                    _slugify(f"width_{width:.1f}"),
                ]
            )
            hit_fig = plot_feature_endpoint(hit_row, patient_features, results["patient_outcomes"], config)
            (hit_figures_dir / f"{hit_stub}.png").write_bytes(_figure_bytes(hit_fig, dpi=300))
            plt.close(hit_fig)

            hit_export_df = build_feature_endpoint_export_df(hit_row, patient_features, results["patient_outcomes"], config)
            if not hit_export_df.empty:
                hit_export_df.to_csv(hit_tables_dir / f"{hit_stub}.csv", index=False)

    manifest = {
        "source_csv": config.get("source_csv"),
        "selected_width": selected_width,
        "selected_zone_label": results.get("selected_zone_label"),
        "selected_feature_label": feature_stability_summary.iloc[0]["feature_label"] if not feature_stability_summary.empty else None,
        "outer_cap": results.get("outer_cap"),
        "available_widths": candidate_summary["zone_width"].astype(float).tolist(),
        "config": config,
        "saved_tables": [str(path.relative_to(run_dir)) for path in sorted(tables_dir.rglob("*.csv"))],
        "saved_figures": [str(path.relative_to(run_dir)) for path in sorted(figures_dir.rglob("*.png"))],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if log_lines:
        (logs_dir / "run.log").write_text("\n".join(log_lines), encoding="utf-8")
    _log(log_fn, f"Saved run outputs to {run_dir}")
    return run_dir


def load_saved_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser()
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables_dir = run_path / "tables"

    def _read_csv_or_empty(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    payload = {
        "run_dir": run_path,
        "manifest": manifest,
        "candidate_summary": _read_csv_or_empty(tables_dir / "candidate_summary.csv"),
        "feature_stability_summary": _read_csv_or_empty(tables_dir / "feature_stability_summary.csv"),
        "stats_all": _read_csv_or_empty(tables_dir / "screening_results_all.csv"),
        "patient_outcomes": _read_csv_or_empty(tables_dir / "patient_outcomes.csv"),
        "patient_features_all": _read_csv_or_empty(tables_dir / "patient_features_all.csv"),
        "combined_features_selected": _read_csv_or_empty(tables_dir / "combined_features_selected_width.csv"),
        "distance_summary": _read_csv_or_empty(tables_dir / "distance_distribution.csv"),
    }
    return payload