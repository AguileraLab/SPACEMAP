from __future__ import annotations

from spacemap_gui import terminal
import json
import os
import shutil
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from itertools import combinations
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point

from . import density_clouds as clouds
from .virtual_tissue import _build_region_geometry, _compute_buffer_distance, _count_holes

REQUIRED_COORDS = ["x_centroid", "y_centroid"]

_DENSITY_CORE_STATE_KEY = "_tab8_density_core_applied"
_DENSITY_PANEL1_RESULTS_KEY = "_tab8_density_panel1_results_cache"
_DENSITY_PANEL2_RESULTS_KEY = "_tab8_density_panel2_results_cache"
_DENSITY_PANEL3_RESULTS_KEY = "_tab8_density_panel3_results_cache"

METHOD_COLOR_DEFAULTS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]

# Stable per-method colors so a method keeps its color whatever order it is selected in.
# Values follow the published SPACEMAP figures (Leiden blue, SOM red, Vortex purple, RESOLVE green).
METHOD_COLOR_BY_NAME = {
    "phenotype_leiden": "#1f77b4",
    "phenotype_som": "#d62728",
    "phenotype_vortex": "#9467bd",
    "phenotype_threshold": "#2ca02c",
    "phenotype_resolve": "#2ca02c",
    "phenotype_scimap": "#ff7f0e",
    "phenotype_ml_mr": "#8c564b",
    "phenotype_ml_consensus": "#17becf",
}

PUBLICATION_STATS_VERSION = "matched_friedman_wilcoxon_holm_v1"


def default_method_color(method: str, index: int = 0) -> str:
    """Canonical color for a phenotype column; falls back to the palette by position."""

    return METHOD_COLOR_BY_NAME.get(str(method), METHOD_COLOR_DEFAULTS[index % len(METHOD_COLOR_DEFAULTS)])


def log_density(message: str):
    """Density Analysis progress line, printed only while the user is on Tab 8."""
    user_tab = os.environ.get("_spacemap_user_tab", "")
    if user_tab and user_tab != "tab8":
        return
    terminal.say("tab8", message, step="Density")


def _holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm step-down family-wise error correction, preserving NaNs."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return adjusted

    order = valid[np.argsort(p[valid], kind="stable")]
    running_max = 0.0
    family_size = int(order.size)
    for rank, original_idx in enumerate(order):
        candidate = min(1.0, float(p[original_idx]) * float(family_size - rank))
        running_max = max(running_max, candidate)
        adjusted[original_idx] = running_max
    return adjusted


def _matched_rank_biserial(first: np.ndarray, second: np.ndarray) -> float:
    """Matched-pairs rank-biserial effect size; positive means first > second."""

    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    denominator = positive + negative
    return float((positive - negative) / denominator) if denominator > 0 else 0.0


def _safe_paired_wilcoxon(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Two-sided paired Wilcoxon test with defined behavior for identical pairs."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid], second[valid]
    if first.size < 2:
        return np.nan, np.nan
    if np.allclose(first - second, 0.0, rtol=0.0, atol=0.0):
        return 0.0, 1.0
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros.*")
            warnings.filterwarnings("ignore", message="Sample size too small for normal approximation.*")
            result = stats.wilcoxon(
                first,
                second,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
    except (ValueError, FloatingPointError):
        return np.nan, np.nan
    return float(getattr(result, "statistic", np.nan)), float(getattr(result, "pvalue", np.nan))


def compute_matched_method_statistics(
    wide: pd.DataFrame,
    methods: list[str],
    analysis_unit: str = "connected_tissue",
) -> pd.DataFrame:
    """Publication-oriented inference for methods measured on the same units.

    A Friedman test is used as the omnibus test when at least three methods are
    available. Pairwise follow-up tests use two-sided Wilcoxon signed-rank tests,
    with Holm correction across every method pair in the family.
    """

    available = [str(method) for method in methods if str(method) in wide.columns]
    if len(available) < 2:
        return pd.DataFrame()

    paired = wide[available].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(paired) < 2:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    if len(available) >= 3:
        arrays = [paired[method].to_numpy(dtype=float) for method in available]
        try:
            omnibus = stats.friedmanchisquare(*arrays)
            statistic = float(getattr(omnibus, "statistic", np.nan))
            p_raw = float(getattr(omnibus, "pvalue", np.nan))
        except ValueError:
            # scipy raises when every method is identical in every block.
            statistic, p_raw = 0.0, 1.0
        kendall_w = statistic / (float(len(paired)) * float(len(available) - 1))
        rows.append(
            {
                "analysis_unit": str(analysis_unit),
                "test": "friedman",
                "comparison": "all methods",
                "method_1": "",
                "method_2": "",
                "n_paired_units": int(len(paired)),
                "statistic": statistic,
                "p_value_raw": p_raw,
                "p_value_adjusted": p_raw,
                "p_value": p_raw,
                "multiple_testing_correction": "not applicable (omnibus)",
                "correction_family_size": 1,
                "effect_size_name": "Kendall_W",
                "effect_size": float(kendall_w),
                "median_method_1": np.nan,
                "median_method_2": np.nan,
                "median_paired_difference": np.nan,
            }
        )

    pair_rows: list[dict[str, object]] = []
    for method_1, method_2 in combinations(available, 2):
        first = paired[method_1].to_numpy(dtype=float)
        second = paired[method_2].to_numpy(dtype=float)
        statistic, p_raw = _safe_paired_wilcoxon(first, second)
        pair_rows.append(
            {
                "analysis_unit": str(analysis_unit),
                "test": "paired_wilcoxon_signed_rank",
                "comparison": f"{method_1} vs {method_2}",
                "method_1": method_1,
                "method_2": method_2,
                "n_paired_units": int(len(paired)),
                "statistic": statistic,
                "p_value_raw": p_raw,
                "effect_size_name": "matched_rank_biserial",
                "effect_size": _matched_rank_biserial(first, second),
                "median_method_1": float(np.nanmedian(first)),
                "median_method_2": float(np.nanmedian(second)),
                "median_paired_difference": float(np.nanmedian(first - second)),
                "inference_warning": "low paired-unit count" if len(paired) < 6 else "",
            }
        )

    adjusted = _holm_adjust([float(row["p_value_raw"]) for row in pair_rows])
    for row, p_adjusted in zip(pair_rows, adjusted):
        row["p_value_adjusted"] = float(p_adjusted) if np.isfinite(p_adjusted) else np.nan
        # Existing plotting helpers consume p_value; it intentionally means adjusted P.
        row["p_value"] = row["p_value_adjusted"]
        row["multiple_testing_correction"] = "Holm"
        row["correction_family_size"] = int(len(pair_rows))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_zonal_method_statistics(
    zone_df: pd.DataFrame,
    group_cols: list[str],
    block_col: str,
    value_col: str,
    methods: list[str],
    method_col: str = "method",
) -> pd.DataFrame:
    """Matched method tests per distance zone, exported without altering curves.

    Pairwise P values are Holm-adjusted across every method-pair-by-zone test
    within the same biological outcome. Omnibus P values are Holm-adjusted across
    zones. This avoids visually overcrowding the distance-profile panels.
    """

    if zone_df.empty or block_col not in zone_df.columns:
        return pd.DataFrame()
    result_frames: list[pd.DataFrame] = []
    group_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, subset in zone_df.groupby(group_key, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        wide = subset.pivot_table(index=block_col, columns=method_col, values=value_col, aggfunc="first")
        stats_df = compute_matched_method_statistics(wide, methods, analysis_unit=block_col)
        if stats_df.empty:
            continue
        for column, value in zip(group_cols, key_values):
            stats_df[column] = value
        result_frames.append(stats_df)
    if not result_frames:
        return pd.DataFrame()

    result = pd.concat(result_frames, ignore_index=True)
    outcome_cols = [column for column in group_cols if column != "zone_center_um"]
    if not outcome_cols:
        result["__outcome__"] = "all"
        outcome_cols = ["__outcome__"]
    result["p_value_holm_within_zone"] = result["p_value_adjusted"]
    for _, indices in result.groupby(outcome_cols + ["test"], dropna=False).groups.items():
        index_list = list(indices)
        adjusted = _holm_adjust(result.loc[index_list, "p_value_raw"].to_numpy(dtype=float))
        result.loc[index_list, "p_value_adjusted"] = adjusted
    result["p_value"] = result["p_value_adjusted"]
    result["multiple_testing_correction"] = "Holm across distance-zone family"
    result["significant"] = result["p_value_adjusted"].lt(0.05)
    result["sig_label"] = result["p_value_adjusted"].map(_p_to_sig_label)
    return result.drop(columns=["__outcome__"], errors="ignore")


def create_publication_bundle(base_dir: Path, panel_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bundle = Path(base_dir) / "publication_exports" / f"{panel_name}_{timestamp}"
    bundle.mkdir(parents=True, exist_ok=False)
    return bundle


def copy_outputs_to_bundle(paths: list[Path], bundle_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in paths:
        source = Path(source)
        if not source.exists() or not source.is_file():
            continue
        destination = bundle_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination.name)
    return copied


def write_publication_metadata(path: Path, payload: dict[str, Any]) -> None:
    def jsonable(value: Any):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, np.ndarray):
            return [jsonable(item) for item in value.tolist()]
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [jsonable(item) for item in value]
        return value

    document = {
        "metadata_version": PUBLICATION_STATS_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(document), handle, indent=2, allow_nan=False)


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    if preferred in options:
        return options.index(preferred)
    return fallback


def _column_options(columns: list[str]) -> list[str]:
    return [""] + list(columns)


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _virtual_tissue_geometry(coords: np.ndarray, percentile: float | None, min_hole_size: float):
    if len(coords) < 2:
        return None, np.nan, np.array([])

    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]

    if percentile is None:
        buffer_distance = _compute_buffer_distance(coords, use_median=True, percentile=99.0)
    else:
        buffer_distance = _compute_buffer_distance(coords, use_median=False, percentile=float(percentile))

    polygon = _build_region_geometry(
        coords=coords,
        method="buffer_union",
        alpha=100.0,
        buffer_distance=float(buffer_distance),
        min_hole_size=float(min_hole_size),
    )

    return polygon, buffer_distance, nearest


def _virtual_area_mm2(coords: np.ndarray, pixel_size_um: float, percentile: float | None, min_hole_size: float) -> tuple[float, float]:
    if len(coords) < 2:
        return np.nan, np.nan

    polygon, buffer_distance, _ = _virtual_tissue_geometry(
        coords=coords,
        percentile=percentile,
        min_hole_size=min_hole_size,
    )
    if polygon is None:
        return np.nan, np.nan

    area_px2 = float(polygon.area)
    area_mm2 = area_px2 * ((pixel_size_um / 1000.0) ** 2)
    return area_mm2, buffer_distance


def _build_connected_group_columns(df: pd.DataFrame, batch_col: str, roi_col: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    group_cols: list[str] = []

    if batch_col:
        out = out[out[batch_col].notna()].copy()
        group_cols.append(batch_col)
    else:
        out["__batch_single__"] = "ALL"
        group_cols.append("__batch_single__")

    if roi_col:
        out[roi_col] = out[roi_col].astype(str).fillna("NA")
        group_cols.append(roi_col)
    else:
        out["__roi_single__"] = "ALL"
        group_cols.append("__roi_single__")

    return out, group_cols


def _draw_connected_tissue_preview(group_df: pd.DataFrame, title: str, percentile: float | None, min_hole_size: float):
    coords = group_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    polygon, buffer_distance, nearest = _virtual_tissue_geometry(
        coords=coords,
        percentile=percentile,
        min_hole_size=min_hole_size,
    )
    if polygon is None or len(nearest) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax0 = axes[0]
    ax0.hist(nearest, bins=50, color="steelblue", alpha=0.8, edgecolor="black")
    ax0.axvline(float(np.median(nearest)), linestyle="--", color="red", linewidth=2, label="median")
    ax0.axvline(float(np.percentile(nearest, 95)), linestyle="--", color="orange", linewidth=2, label="95th")
    ax0.axvline(float(np.percentile(nearest, 99)), linestyle="--", color="purple", linewidth=2, label="99th")
    ax0.set_xlabel("Nearest-neighbor distance (pixels)")
    ax0.set_ylabel("Cell count")
    ax0.set_title("Nearest-neighbor distance")
    ax0.grid(True, alpha=0.25)
    ax0.legend()

    ax1 = axes[1]
    ax1.scatter(coords[:, 0], coords[:, 1], s=3, c="gray", alpha=0.45, edgecolors="none", label="cells")
    if polygon.geom_type == "Polygon":
        exterior = getattr(polygon, "exterior", None)
        interiors = list(getattr(polygon, "interiors", []))
        if exterior is not None:
            x, y = exterior.xy
            ax1.fill(x, y, color="#2ca02c", alpha=0.25, label="virtual tissue")
            ax1.plot(x, y, "k-", linewidth=1.3)
            for hole in interiors:
                hx, hy = zip(*hole.coords)
                ax1.fill(hx, hy, color="white", alpha=1.0)
                ax1.plot(hx, hy, "r-", linewidth=1.0)
    elif polygon.geom_type == "MultiPolygon":
        first = True
        total_holes = 0
        for poly in list(getattr(polygon, "geoms", [])):
            x, y = poly.exterior.xy
            ax1.fill(x, y, color="#2ca02c", alpha=0.25, label="virtual tissue" if first else None)
            ax1.plot(x, y, "k-", linewidth=1.1)
            interiors = list(getattr(poly, "interiors", []))
            total_holes += len(interiors)
            for hole in interiors:
                hx, hy = zip(*hole.coords)
                ax1.fill(hx, hy, color="white", alpha=1.0)
                ax1.plot(hx, hy, "r-", linewidth=1.0)
            first = False
        if total_holes > 0:
            ax1.plot([], [], "r-", linewidth=1.0, label=f"holes (n={total_holes})")

    ax1.set_aspect("equal")
    ax1.invert_yaxis()
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_title(
        f"Connected tissue preview | buffer={float(buffer_distance):.2f}px | holes={int(_count_holes(polygon))}"
    )
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def _phenotype_pool(df: pd.DataFrame, method_cols: list[str]) -> list[str]:
    vals: set[str] = set()
    for col in method_cols:
        if col not in df.columns:
            continue
        vals.update(df[col].dropna().astype(str).unique().tolist())
    return sorted(vals)


def _apply_value_filters(df: pd.DataFrame, filter_cols: list[str], key_prefix: str) -> pd.DataFrame:
    out = df.copy()
    if not filter_cols:
        return out
    for col in filter_cols:
        options = sorted(out[col].dropna().astype(str).unique().tolist())
        vals = st.multiselect(
            f"Filter values for {col}",
            options=options,
            default=[],
            key=f"{key_prefix}_values_{col}",
            help="Leave empty to keep all values for this column.",
        )
        if vals:
            out = out[out[col].astype(str).isin(vals)]
    return out


def _render_filter_widgets(df: pd.DataFrame, filter_cols: list[str], key_prefix: str) -> dict[str, list[str]]:
    selections: dict[str, list[str]] = {}
    for col in filter_cols:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        selections[col] = st.multiselect(
            f"Filter values for {col}",
            options=options,
            key=f"{key_prefix}_values_{col}",
            help="Leave empty to keep all values for this column.",
        )
    return selections


def _sync_multiselect_state(widget_key: str, options: list[str], fallback_values: list[str]) -> list[str]:
    current = st.session_state.get(widget_key)
    if current is None:
        sanitized = [value for value in fallback_values if value in options]
    else:
        sanitized = [str(value) for value in current if str(value) in options]
        if not sanitized and options:
            sanitized = [value for value in fallback_values if value in options]
    st.session_state[widget_key] = sanitized
    return sanitized


def _sync_selectbox_state(widget_key: str, options: list[Any], fallback_value: Any = None) -> Any:
    if not options:
        st.session_state.pop(widget_key, None)
        return None

    safe_fallback = fallback_value if fallback_value in options else ("" if "" in options else options[0])
    current = st.session_state.get(widget_key)
    if current not in options:
        st.session_state[widget_key] = safe_fallback
        return safe_fallback
    return current


def _apply_filter_map(df: pd.DataFrame, filter_values: dict[str, list[str]] | tuple[tuple[str, tuple[str, ...]], ...]) -> pd.DataFrame:
    out = df.copy()
    if isinstance(filter_values, tuple):
        iterable = [(col, list(vals)) for col, vals in filter_values]
    else:
        iterable = list(filter_values.items())

    for col, values in iterable:
        if not values or col not in out.columns:
            continue
        out = out[out[col].astype(str).isin([str(v) for v in values])]
    return out


def _freeze_filter_map(filter_values: dict[str, list[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    frozen: list[tuple[str, tuple[str, ...]]] = []
    for col in sorted(filter_values):
        values = tuple(sorted(str(v) for v in filter_values.get(col, []) if str(v) != ""))
        if values:
            frozen.append((str(col), values))
    return tuple(frozen)


def _format_group_label(row: pd.Series, group_cols: list[str]) -> str:
    return " | ".join([f"{c}={row[c]}" for c in group_cols])


def _p_to_sig_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


_BRACKET_BASE_Y = 1.01
_BRACKET_STEP = 0.065
_BRACKET_HEIGHT = 0.02
_BRACKET_TEXT_GAP = 0.008


def _bracket_placements(methods: list[str], pairwise_stats: list[dict]) -> list[tuple[int, int, int, float]]:
    """Assign every significant pair to a bracket level that does not overlap."""
    sig = [r for r in pairwise_stats if np.isfinite(r.get("p_value", np.nan)) and float(r["p_value"]) < 0.05]
    if not sig:
        return []

    parsed = []
    for row in sig:
        m1 = str(row.get("method_1", ""))
        m2 = str(row.get("method_2", ""))
        if m1 not in methods or m2 not in methods:
            continue
        i1 = methods.index(m1) + 1
        i2 = methods.index(m2) + 1
        if i1 == i2:
            continue
        lo, hi = min(i1, i2), max(i1, i2)
        parsed.append((lo, hi, float(row["p_value"])))

    parsed = sorted(parsed, key=lambda t: ((t[1] - t[0]), t[0]))

    intervals_per_level: list[list[tuple[float, float]]] = []
    placements: list[tuple[int, int, int, float]] = []
    for lo, hi, p_val in parsed:
        level_idx = 0
        placed = False
        while not placed:
            if level_idx >= len(intervals_per_level):
                intervals_per_level.append([])
            overlaps = any(not (hi < a - 0.05 or lo > b + 0.05) for a, b in intervals_per_level[level_idx])
            if not overlaps:
                intervals_per_level[level_idx].append((lo, hi))
                placements.append((lo, hi, level_idx, p_val))
                placed = True
            else:
                level_idx += 1
    return placements


def _bracket_top_axes_fraction(methods: list[str], pairwise_stats: list[dict]) -> float:
    """Top of the significance-bracket stack in axes coordinates (1.0 when empty).

    Titles are positioned above this value so labels and brackets never overlap.
    """
    placements = _bracket_placements(methods, pairwise_stats)
    if not placements:
        return 1.0
    top_level = max(level for _, _, level, _ in placements)
    label_height = 0.042
    return _BRACKET_BASE_Y + (top_level * _BRACKET_STEP) + _BRACKET_HEIGHT + _BRACKET_TEXT_GAP + label_height


def _add_pairwise_significance_brackets(ax, methods: list[str], pairwise_stats: list[dict], y_data_max: float, y_data_min: float, sig_bracket_linewidth: float = 1.0, sig_fontsize: float = 10.0) -> float:
    # Draw brackets in axis-relative Y coordinates so manual/auto y-limits do not move or clip them.
    base_y_axes = _BRACKET_BASE_Y
    step_axes = _BRACKET_STEP
    bracket_h_axes = _BRACKET_HEIGHT
    text_gap_axes = _BRACKET_TEXT_GAP
    xaxis_transform = ax.get_xaxis_transform()

    placements = _bracket_placements(methods, pairwise_stats)
    if not placements:
        return y_data_max

    for lo, hi, lvl, p_val in placements:
        y = base_y_axes + (lvl * step_axes)
        ax.plot(
            [lo, lo, hi, hi],
            [y, y + bracket_h_axes, y + bracket_h_axes, y],
            color="black",
            linewidth=float(sig_bracket_linewidth),
            zorder=10,
            transform=xaxis_transform,
            clip_on=False,
        )
        ax.text(
            (lo + hi) / 2.0,
            y + bracket_h_axes + text_gap_axes,
            _p_to_sig_label(p_val),
            ha="center",
            va="bottom",
            fontsize=int(sig_fontsize),
            fontweight="bold",
            zorder=11,
            transform=xaxis_transform,
            clip_on=False,
        )

    return y_data_max


def _panel1_log_base(y_scale_mode: str) -> float | None:
    if y_scale_mode == "log10":
        return 10.0
    if y_scale_mode == "log2":
        return 2.0
    return None


def _panel1_prepare_display_arrays(
    data_arrays: list[np.ndarray],
    y_scale_mode: str,
) -> tuple[list[np.ndarray], float | None]:
    cleaned_arrays = [np.asarray(arr, dtype=float)[np.isfinite(arr)] for arr in data_arrays]
    log_base = _panel1_log_base(y_scale_mode)
    if log_base is None:
        return cleaned_arrays, None

    positive_arrays = [arr[arr > 0] for arr in cleaned_arrays if arr.size > 0]
    if not positive_arrays:
        return cleaned_arrays, None

    positive_values = np.concatenate(positive_arrays)
    if positive_values.size == 0:
        return cleaned_arrays, None

    floor_value = float(max(float(np.nanmin(positive_values)) / log_base, float(np.finfo(float).tiny)))
    display_arrays: list[np.ndarray] = []
    for arr in cleaned_arrays:
        display_arr = arr.copy()
        if display_arr.size > 0:
            display_arr[display_arr <= 0] = floor_value
        display_arrays.append(display_arr)
    return display_arrays, floor_value


def _panel1_add_significance_brackets(
    ax,
    methods: list[str],
    pairwise_stats: list[dict],
    y_data_max: float,
    y_data_min: float,
    y_scale_mode: str,
    sig_bracket_linewidth: float = 1.0,
    sig_fontsize: float = 10.0,
    use_axis_break: bool = False,
    break_threshold: float = np.nan,
    break_height: float = 0.15,
) -> float:
    return _add_pairwise_significance_brackets(
        ax=ax,
        methods=methods,
        pairwise_stats=pairwise_stats,
        y_data_max=y_data_max,
        y_data_min=y_data_min,
        sig_bracket_linewidth=sig_bracket_linewidth,
        sig_fontsize=sig_fontsize,
    )


def _panel1_apply_y_axis(
    ax,
    y_scale_mode: str,
    y_data_min: float,
    y_data_max: float,
    y_top: float,
    manual_y_min_enabled: bool,
    manual_y_min: float,
    manual_y_max_enabled: bool,
    manual_y_max: float,
):
    log_base = _panel1_log_base(y_scale_mode)
    auto_top = max(
        float(y_data_max) if np.isfinite(y_data_max) else 0.0,
        float(y_top) if np.isfinite(y_top) else 0.0,
    )

    if log_base is None:
        span = y_data_max - y_data_min
        if not np.isfinite(span) or span <= 0:
            span = max(abs(y_data_max), 1.0)
        auto_bottom = y_data_min - (0.08 * span)
        if np.isfinite(y_data_min) and y_data_min >= 0:
            auto_bottom = min(0.0, auto_bottom)

        if manual_y_min_enabled and np.isfinite(manual_y_min):
            lower = float(manual_y_min)
        else:
            lower = float(auto_bottom)

        if manual_y_max_enabled and np.isfinite(manual_y_max):
            upper = float(manual_y_max)
            upper = max(upper, float(auto_top)) if np.isfinite(auto_top) else upper
        else:
            upper = max(float(auto_top), float(y_data_max))

        if not np.isfinite(lower):
            lower = float(auto_bottom)
        if not np.isfinite(upper):
            upper = float(auto_top)
        if lower >= upper:
            lower = min(float(auto_bottom), upper - max(1.0, 0.1 * span))
        ax.set_ylim(lower, upper)
        return

    ax.set_yscale("log", base=log_base)
    positive_candidates = [
        float(v)
        for v in [y_data_min, y_data_max, y_top]
        if np.isfinite(v) and float(v) > 0
    ]
    if not positive_candidates:
        return

    positive_min = min(positive_candidates)
    auto_bottom = max(positive_min / (log_base ** 0.35), np.finfo(float).tiny)
    if manual_y_min_enabled and np.isfinite(manual_y_min) and float(manual_y_min) > 0:
        lower = float(manual_y_min)
    else:
        lower = float(auto_bottom)

    if manual_y_max_enabled and np.isfinite(manual_y_max) and float(manual_y_max) > 0:
        upper_candidate = float(manual_y_max)
    else:
        upper_candidate = float(auto_top)
    upper = max(float(auto_top), upper_candidate)

    if lower >= upper:
        lower = max(min(positive_min, upper / log_base), np.finfo(float).tiny)
    ax.set_ylim(lower, upper)


def _panel1_compute_axis_break(
    data_arrays: list[np.ndarray],
    enable_break: bool,
    break_percentile: float,
) -> tuple[bool, float, float]:
    if not enable_break:
        return False, np.nan, np.nan

    all_finite = np.concatenate([arr[np.isfinite(arr)] for arr in data_arrays if arr.size > 0])
    if all_finite.size == 0:
        return False, np.nan, np.nan

    threshold = float(np.nanpercentile(all_finite, break_percentile))
    outlier_vals = all_finite[all_finite > threshold]
    if outlier_vals.size == 0:
        return False, np.nan, np.nan

    return True, threshold, float(np.nanmax(outlier_vals))


def _panel1_compute_outlier_crop(
    data_arrays: list[np.ndarray],
    mode: str,
    percentile: float,
    iqr_multiplier: float,
) -> tuple[float, int, float] | None:
    """Upper display limit that crops extreme high outliers out of the view.

    Returns ``(limit, n_hidden, hidden_max)`` or ``None`` when nothing is cropped.
    Only the axis view is affected: boxes, whiskers and every statistic are still
    computed from the complete data.
    """
    if mode not in {"percentile", "iqr"}:
        return None

    finite = [np.asarray(arr, dtype=float)[np.isfinite(arr)] for arr in data_arrays]
    finite = [arr for arr in finite if arr.size > 0]
    if not finite:
        return None
    values = np.concatenate(finite)
    if values.size < 3:
        return None

    if mode == "percentile":
        limit = float(np.nanpercentile(values, float(percentile)))
    else:
        q1, q3 = [float(v) for v in np.nanpercentile(values, [25.0, 75.0])]
        limit = float(q3 + (float(iqr_multiplier) * (q3 - q1)))

    if not np.isfinite(limit):
        return None
    hidden = values[values > limit]
    if hidden.size == 0:
        return None
    return float(limit), int(hidden.size), float(np.nanmax(hidden))


def _panel1_transform_for_break(y_val: float, break_threshold: float, break_height: float = 0.15) -> float:
    if y_val <= break_threshold:
        return y_val
    return break_threshold + break_height + (y_val - break_threshold) * 0.5


def _panel1_draw_axis_break(ax, break_threshold: float, break_height: float = 0.15, x_range_frac: float = 0.03):
    ylim = ax.get_ylim()
    y_display = break_threshold + break_height * 0.5
    x_min, x_max = ax.get_xlim()
    x_width = x_max - x_min
    x_left = x_min + x_width * x_range_frac
    x_right = x_max - x_width * x_range_frac
    x_center = (x_left + x_right) / 2.0
    break_width = x_right - x_left
    half_width = break_width / 2.0

    n_zigzags = 4
    zigzag_x = x_left + np.linspace(0, break_width, n_zigzags * 2 + 1)
    zigzag_y = np.array([break_threshold + (i % 2) * (break_height * 0.4) for i in range(len(zigzag_x))])
    ax.plot(zigzag_x, zigzag_y, "k-", linewidth=1.2, clip_on=False, zorder=100)


def _panel1_transform_for_split(
    y_val: float,
    low_min: float,
    low_max: float,
    high_min: float,
    high_max: float,
    gap_height: float,
) -> float:
    if not np.isfinite(y_val):
        return np.nan
    y = float(y_val)
    if low_min <= y <= low_max:
        return y
    if high_min <= y <= high_max:
        return low_max + gap_height + (y - high_min)
    return np.nan


def _panel1_transform_array_for_split(
    y_vals: np.ndarray,
    low_min: float,
    low_max: float,
    high_min: float,
    high_max: float,
    gap_height: float,
) -> np.ndarray:
    if y_vals.size == 0:
        return y_vals
    transformed = np.array(
        [
            _panel1_transform_for_split(v, low_min, low_max, high_min, high_max, gap_height)
            for v in y_vals
        ],
        dtype=float,
    )
    return transformed[np.isfinite(transformed)]


def _panel1_inverse_split_transform(
    y_display: float,
    low_max: float,
    high_min: float,
    gap_height: float,
) -> float:
    if not np.isfinite(y_display):
        return np.nan
    y = float(y_display)
    if y <= low_max:
        return y
    if y >= (low_max + gap_height):
        return high_min + (y - (low_max + gap_height))
    return np.nan


def _panel1_apply_split_axis(
    ax,
    low_min: float,
    low_max: float,
    high_min: float,
    high_max: float,
    gap_height: float,
):
    upper_display = low_max + gap_height + (high_max - high_min)
    ax.set_ylim(low_min, upper_display)

    lower_ticks = np.linspace(low_min, low_max, num=4)
    upper_ticks = np.linspace(low_max + gap_height, upper_display, num=3)
    ticks = np.unique(np.round(np.concatenate([lower_ticks, upper_ticks]), 6))
    ax.set_yticks(ticks)

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda y, _: (
                ""
                if not np.isfinite(_panel1_inverse_split_transform(y, low_max, high_min, gap_height))
                else f"{_panel1_inverse_split_transform(y, low_max, high_min, gap_height):g}"
            )
        )
    )

    # Draw a centered break marker in data coordinates at the compressed gap.
    _panel1_draw_axis_break(ax, break_threshold=low_max, break_height=gap_height)


@st.cache_data(show_spinner=False)
def _prepare_density_context(
    df: pd.DataFrame,
    x_coord_col: str,
    y_coord_col: str,
    batch_col: str,
    roi_col: str,
    min_cells_group: int,
    use_median: bool,
    percentile: float,
    min_hole_size: float,
    core_filter_values: tuple[tuple[str, tuple[str, ...]], ...],
):
    if not x_coord_col or not y_coord_col:
        return {"error": "Please select X and Y coordinate columns and apply the core configuration."}
    missing_cols = [col for col in [x_coord_col, y_coord_col] if col not in df.columns]
    if missing_cols:
        return {"error": f"Selected coordinate column(s) not found in the loaded CSV: {', '.join(missing_cols)}"}

    work_df = df.copy()
    work_df["x_centroid"] = pd.to_numeric(work_df[x_coord_col], errors="coerce")
    work_df["y_centroid"] = pd.to_numeric(work_df[y_coord_col], errors="coerce")
    filtered_df = _apply_filter_map(work_df, core_filter_values)

    if x_coord_col == y_coord_col:
        return {"error": "X Coordinate Column and Y Coordinate Column must be different."}

    run_df = filtered_df.dropna(subset=REQUIRED_COORDS).copy()
    if run_df.empty:
        return {"warning": "No valid rows remain after current filters and coordinate cleanup."}

    non_coord_run_cols = [c for c in run_df.columns if c not in set(REQUIRED_COORDS + [x_coord_col, y_coord_col])]
    run_df, connected_cols = _build_connected_group_columns(run_df, batch_col=batch_col, roi_col=roi_col)
    percentile_value = None if use_median else float(percentile)

    grouped = run_df.groupby(connected_cols, dropna=False)
    group_rows = []
    for key, gdf in grouped:
        if len(gdf) < int(min_cells_group):
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(connected_cols, key)}
        row["n_cells"] = int(len(gdf))
        group_rows.append(row)

    if not group_rows:
        return {"warning": "No connected tissues remain after minimum cell threshold."}

    group_df = pd.DataFrame(group_rows)
    preview_options = group_df.apply(lambda r: _format_group_label(r, connected_cols), axis=1).tolist()

    return {
        "run_df": run_df,
        "connected_cols": connected_cols,
        "group_df": group_df,
        "preview_options": preview_options,
        "percentile_value": percentile_value,
        "non_coord_run_cols": non_coord_run_cols,
        "filtered_rows": int(len(filtered_df)),
        "run_rows": int(len(run_df)),
    }


def _render_panel1_results(panel_state: dict):
    st.info("Showing the most recent Panel 1 run.")
    st.dataframe(panel_state["results_df"], width="stretch")

    for pheno in panel_state.get("phenotypes", []):
        plot_path = panel_state.get("plot_paths", {}).get(pheno)
        if plot_path and Path(plot_path).exists():
            st.markdown(f"#### Panel 1 Plot: {pheno}")
            st.image(str(plot_path), width="stretch")
            footer_text = panel_state.get("footer_texts", {}).get(pheno, "")
            if footer_text:
                st.caption(footer_text)

        stats_df = panel_state.get("stats_by_pheno", {}).get(pheno)
        if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
            st.dataframe(stats_df, width="stretch")

    st.write(
        {
            "Panel1_results_csv": panel_state.get("results_csv"),
            "Connected_tissues": panel_state.get("connected_tissues"),
            "Publication_bundle": panel_state.get("publication_bundle"),
        }
    )


def _render_panel2_results(panel_state: dict):
    st.info("Showing the most recent Panel 2 run.")
    runtime_s = panel_state.get("runtime_seconds")
    if runtime_s is not None:
        st.caption(f"Last Panel 2 runtime: {float(runtime_s):.1f}s")
    st.dataframe(panel_state["summary_df"], width="stretch")

    stats_df = panel_state.get("stats_df")
    if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
        st.markdown("#### Panel 2 Statistics")
        st.dataframe(stats_df, width="stretch")

    method_stats_df = panel_state.get("method_stats_df")
    if isinstance(method_stats_df, pd.DataFrame) and not method_stats_df.empty:
        st.markdown("#### Matched method statistics by distance zone")
        st.dataframe(method_stats_df, width="stretch")

    for pheno in panel_state.get("phenotypes", []):
        plot_path = panel_state.get("plot_paths", {}).get(pheno)
        if plot_path and Path(plot_path).exists():
            st.markdown(f"#### Panel 2 Plot: {pheno}")
            st.image(str(plot_path), width="stretch")

    st.write(
        {
            "Panel2_detailed_csv": panel_state.get("detailed_csv"),
            "Panel2_summary_csv": panel_state.get("summary_csv"),
            "Rows_detailed": panel_state.get("rows_detailed"),
            "Rows_summary": panel_state.get("rows_summary"),
            "Publication_bundle": panel_state.get("publication_bundle"),
        }
    )


def _coords_cache_key(coords: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in np.asarray(coords, dtype=float))


@st.cache_data(show_spinner=False)
def _panel2_exact_geometry_summary_cached(
    coords_key: tuple[tuple[float, float], ...],
    pixel_size_um: float,
    percentile: float | None,
    min_hole_size: float,
) -> dict[str, float]:
    coords = np.asarray(coords_key, dtype=float)
    if len(coords) < 2:
        return {"area_mm2": np.nan, "buffer_distance_px": np.nan, "x_min": np.nan, "x_max": np.nan, "y_min": np.nan, "y_max": np.nan}

    polygon, buffer_distance, _ = _virtual_tissue_geometry(
        coords=coords,
        percentile=None if percentile is None else float(percentile),
        min_hole_size=float(min_hole_size),
    )
    if polygon is None:
        return {"area_mm2": np.nan, "buffer_distance_px": np.nan, "x_min": np.nan, "x_max": np.nan, "y_min": np.nan, "y_max": np.nan}

    area_px2 = float(polygon.area)
    area_mm2 = area_px2 * ((float(pixel_size_um) / 1000.0) ** 2)
    minx, miny, maxx, maxy = polygon.bounds
    return {
        "area_mm2": float(area_mm2),
        "buffer_distance_px": float(buffer_distance) if np.isfinite(buffer_distance) else np.nan,
        "x_min": float(minx),
        "x_max": float(maxx),
        "y_min": float(miny),
        "y_max": float(maxy),
    }


def _build_stratum_series(df: pd.DataFrame, strat_cols: list[str]) -> pd.Series:
    if not strat_cols:
        return pd.Series(["All"] * len(df), index=df.index, dtype="object")
    return df[strat_cols].astype(str).apply(lambda row: " | ".join(row), axis=1)


def _numeric_dict_value(mapping: dict[str, Any], key: str, default: float = np.nan) -> float:
    value = mapping.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def render_zonal_density_figure(
    *,
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    plot_methods: list[str],
    strat_cols: list[str],
    strata: list[str],
    overlay_active: bool,
    overlay_encoding: str,
    show_stats: bool,
    stratum_color_map: dict[str, str],
    method_color_map: dict[str, str],
    method_label_map: dict[str, str],
    method_color_defaults: list[str],
    min_distance_um: float,
    max_distance_um: float,
    title_for_stratum,
    y_label: str,
    x_label: str = "Distance from reference boundary (µm)",
    value_col: str = "density_mean",
    sem_col: str = "density_sem",
    key_prefix: str = "tab8_density_p2",
    stats_filter: dict[str, str] | None = None,
):
    """Draw the zonal density curves shared by the Density and Triad subtabs.

    Both subtabs call this so the curves, shading, legend and axis overrides are
    byte-for-byte the same figure style; only the measured quantity differs.
    """

    square_key = f"{key_prefix}_zonal_fig_square"
    if st.session_state.get(square_key, True):
        zonal_w = float(st.session_state.get(f"{key_prefix}_zonal_fig_size_linked", 5.0))
        zonal_h = zonal_w
    else:
        zonal_w = float(st.session_state.get(f"{key_prefix}_zonal_fig_width", 6.0))
        zonal_h = float(st.session_state.get(f"{key_prefix}_zonal_fig_height", 5.0))

    n_panels = 1 if overlay_active else len(strata)
    fig, axes = plt.subplots(1, n_panels, figsize=(zonal_w * n_panels, zonal_h), squeeze=False)
    axes = axes.flatten()
    stratum_line_styles = ["-", "--", ":", "-."]

    panel_specs = [(axes[0], None)] if overlay_active else list(zip(axes, strata))
    for ax, stratum in panel_specs:
        if st.session_state.get(square_key, True):
            ax.set_box_aspect(1)
        sdf = summary_df.copy() if overlay_active else summary_df[summary_df["__stratum__"] == stratum].copy()
        for method_idx, method in enumerate(plot_methods):
            stratum_iter = strata if overlay_active else [stratum]
            for stratum_idx, stratum_name in enumerate(stratum_iter):
                mdf = sdf[(sdf["method"] == method) & (sdf["__stratum__"] == stratum_name)].sort_values("zone_center_um")
                if mdf.empty:
                    continue
                x = mdf["zone_center_um"].to_numpy(dtype=float)
                y = mdf[value_col].to_numpy(dtype=float)
                yerr = mdf[sem_col].fillna(0).to_numpy(dtype=float)
                method_display = method_label_map.get(str(method), str(method))
                label = method_display if not overlay_active else f"{method_display} | {stratum_name}"
                if overlay_active and overlay_encoding == "color_by_stratum":
                    line_color = stratum_color_map.get(
                        str(stratum_name),
                        ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"][stratum_idx % 8],
                    )
                    line_style = "-"
                else:
                    line_color = method_color_map.get(
                        str(method),
                        method_color_defaults[method_idx % len(method_color_defaults)],
                    )
                    line_style = stratum_line_styles[stratum_idx % len(stratum_line_styles)] if overlay_active else "-"
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.8,
                    color=line_color,
                    linestyle=line_style,
                    label=label,
                )
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.10 if overlay_active else 0.15, color=line_color)

        if overlay_active and show_stats and strat_cols and len(strata) == 2 and not stats_df.empty:
            sig_df = stats_df[stats_df["significant"] == True].copy()
            for column, value in (stats_filter or {}).items():
                if column in sig_df.columns:
                    sig_df = sig_df[sig_df[column].astype(str) == str(value)]
            if not sig_df.empty:
                overall_span = sdf[value_col].max() - sdf[value_col].min()
                overall_span = float(overall_span) if np.isfinite(overall_span) and overall_span > 0 else 1.0
                for method_idx, method in enumerate(plot_methods):
                    method_sig = sig_df[sig_df["method"] == str(method)].copy()
                    if method_sig.empty:
                        continue
                    for _, sig_row in method_sig.iterrows():
                        higher_stratum = str(sig_row["higher_stratum"])
                        if overlay_encoding == "color_by_stratum":
                            sig_color = stratum_color_map.get(higher_stratum, "#111111")
                        else:
                            sig_color = method_color_map.get(
                                str(method),
                                method_color_defaults[method_idx % len(method_color_defaults)],
                            )
                        x_val = float(sig_row["zone_center_um"])
                        point_df = sdf[
                            (sdf["method"] == str(method))
                            & (sdf["__stratum__"] == higher_stratum)
                            & (np.isclose(sdf["zone_center_um"], x_val))
                        ]
                        if point_df.empty:
                            continue
                        point_row = point_df.iloc[0]
                        y_val = float(point_row[value_col]) + float(point_row.get(sem_col, 0.0) or 0.0)
                        y_val += overall_span * (0.07 + (0.04 * method_idx))
                        ax.text(
                            x_val,
                            y_val,
                            str(sig_row["sig_label"]),
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                            color=sig_color,
                        )

        if float(min_distance_um) <= 0 <= float(max_distance_um):
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        if float(min_distance_um) < 0:
            ax.axvspan(float(min_distance_um), min(0.0, float(max_distance_um)), color="red", alpha=0.08)
        if float(max_distance_um) > 0:
            ax.axvspan(max(0.0, float(min_distance_um)), float(max_distance_um), color="blue", alpha=0.08)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title_for_stratum(stratum))
        ax.grid(True, alpha=0.25)

        if st.session_state.get(f"{key_prefix}_zonal_override_x", False):
            ax.set_xlim(
                float(st.session_state.get(f"{key_prefix}_zonal_x_min", -600.0)),
                float(st.session_state.get(f"{key_prefix}_zonal_x_max", 600.0)),
            )
        if st.session_state.get(f"{key_prefix}_zonal_override_y", False):
            ax.set_ylim(
                float(st.session_state.get(f"{key_prefix}_zonal_y_min", 0.0)),
                float(st.session_state.get(f"{key_prefix}_zonal_y_max", 500.0)),
            )

        if st.session_state.get(f"{key_prefix}_zonal_show_legend", True):
            leg_loc_ui = st.session_state.get(f"{key_prefix}_zonal_legend_loc", "Outside Right")
            leg_font_size = int(st.session_state.get(f"{key_prefix}_legend_fontsize", 8))
            leg_loc_map = {
                "Best": "best",
                "Upper Right": "upper right",
                "Upper Left": "upper left",
                "Lower Right": "lower right",
                "Lower Left": "lower left",
            }
            if leg_loc_ui in leg_loc_map:
                ax.legend(fontsize=leg_font_size, loc=leg_loc_map[leg_loc_ui])
            elif leg_loc_ui == "Outside Right":
                ax.legend(fontsize=leg_font_size, loc="upper left", bbox_to_anchor=(1.02, 1.0))
            elif leg_loc_ui == "Outside Bottom":
                ax.legend(fontsize=leg_font_size, loc="upper center", bbox_to_anchor=(0.5, -0.25))

    plt.tight_layout()
    return fig


def render_zonal_plot_controls(key_prefix: str, note: str = ""):
    """The 'Plot style' controls for a zonal-density panel (shared by both subtabs)."""

    if note:
        st.caption(note)
    cz1, cz2, cz3, cz4 = st.columns(4)

    override_x = cz1.checkbox("Override X-axis range", value=False, key=f"{key_prefix}_zonal_override_x")
    cz1.number_input("Zonal X min (µm)", value=-600.0, step=50.0, key=f"{key_prefix}_zonal_x_min", disabled=not override_x)
    cz1.number_input("Zonal X max (µm)", value=600.0, step=50.0, key=f"{key_prefix}_zonal_x_max", disabled=not override_x)

    override_y = cz2.checkbox("Override Y-axis range", value=False, key=f"{key_prefix}_zonal_override_y")
    cz2.number_input("Zonal Y min", value=0.0, step=1.0, key=f"{key_prefix}_zonal_y_min", disabled=not override_y)
    cz2.number_input("Zonal Y max", value=500.0, step=50.0, key=f"{key_prefix}_zonal_y_max", disabled=not override_y)

    show_leg = cz3.checkbox("Show legend on zonal plot", value=True, key=f"{key_prefix}_zonal_show_legend")
    cz3.selectbox(
        "Legend location",
        options=["Best", "Upper Right", "Upper Left", "Lower Right", "Lower Left", "Outside Right", "Outside Bottom"],
        index=5,
        key=f"{key_prefix}_zonal_legend_loc",
        disabled=not show_leg,
    )
    cz3.slider(
        "Legend font size",
        min_value=4,
        max_value=20,
        value=8,
        step=1,
        key=f"{key_prefix}_legend_fontsize",
        disabled=not show_leg,
    )

    zonal_square = cz4.checkbox("Link panel width & height (Square)", value=True, key=f"{key_prefix}_zonal_fig_square")
    if zonal_square:
        cz4.number_input("Panel size (inches)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key=f"{key_prefix}_zonal_fig_size_linked")
    else:
        cz4.number_input("Panel width (inches)", min_value=1.0, max_value=20.0, value=6.0, step=0.5, key=f"{key_prefix}_zonal_fig_width")
        cz4.number_input("Panel height (inches)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key=f"{key_prefix}_zonal_fig_height")


def _compute_panel2_stats(
    zone_df: pd.DataFrame,
    strat_cols: list[str],
    stats_method: str,
    alpha: float,
) -> pd.DataFrame:
    if zone_df.empty or not strat_cols:
        return pd.DataFrame()

    stats_input = zone_df.copy()
    stats_input["__stratum__"] = _build_stratum_series(stats_input, strat_cols)
    stats_rows: list[dict[str, object]] = []

    for (phenotype, method, zone_center_um), sdf in stats_input.groupby(["phenotype", "method", "zone_center_um"], dropna=False):
        strata = sorted(sdf["__stratum__"].dropna().astype(str).unique().tolist())
        if len(strata) != 2:
            continue

        arr1 = sdf[sdf["__stratum__"] == strata[0]]["density_cells_per_mm2"].to_numpy(dtype=float)
        arr2 = sdf[sdf["__stratum__"] == strata[1]]["density_cells_per_mm2"].to_numpy(dtype=float)
        arr1 = arr1[np.isfinite(arr1)]
        arr2 = arr2[np.isfinite(arr2)]
        if len(arr1) < 2 or len(arr2) < 2:
            continue

        if stats_method == "mannwhitney_u":
            stat_res = stats.mannwhitneyu(arr1, arr2, alternative="two-sided")
        else:
            stat_res = stats.ttest_ind(arr1, arr2, equal_var=False)

        p_val = float(getattr(stat_res, "pvalue", np.nan))
        mean_1 = float(np.nanmean(arr1))
        mean_2 = float(np.nanmean(arr2))
        higher_stratum = strata[0] if mean_1 > mean_2 else strata[1]
        stats_rows.append(
            {
                "phenotype": str(phenotype),
                "method": str(method),
                "zone_center_um": float(zone_center_um),
                "stratum_1": strata[0],
                "stratum_2": strata[1],
                "mean_1": mean_1,
                "mean_2": mean_2,
                "n_1": int(len(arr1)),
                "n_2": int(len(arr2)),
                "test": str(stats_method),
                "statistic": float(getattr(stat_res, "statistic", np.nan)),
                "p_value_raw": p_val,
                "p_value_adjusted": np.nan,
                "p_value": np.nan,
                "higher_stratum": higher_stratum,
                "multiple_testing_correction": "Holm across distance zones",
                "significant": False,
                "sig_label": "",
            }
        )

    result = pd.DataFrame(stats_rows)
    if result.empty:
        return result
    family_cols = ["phenotype", "method", "stratum_1", "stratum_2", "test"]
    for _, indices in result.groupby(family_cols, dropna=False).groups.items():
        index_list = list(indices)
        adjusted = _holm_adjust(result.loc[index_list, "p_value_raw"].to_numpy(dtype=float))
        result.loc[index_list, "p_value_adjusted"] = adjusted
    result["p_value"] = result["p_value_adjusted"]
    result["significant"] = result["p_value_adjusted"].lt(float(alpha))
    result["sig_label"] = result["p_value_adjusted"].map(_p_to_sig_label)
    return result


def render_panel1_fragment(
    run_df: pd.DataFrame,
    phenotype_cols: list[str],
    shared: dict[str, Any],
    non_coord_run_cols: list[str],
    connected_cols: list[str],
    pixel_size_um: float,
    min_hole_size: float,
    min_cells_group: int,
    output_dir: str,
    percentile_value: float | None,
):
    st.caption(
        "Compares the density of each phenotype across phenotyping methods, one dot per connected tissue. "
        "Edits stay local until you click Run Panel 1."
    )

    p1_method_cols = [str(m) for m in shared.get("methods", []) if m in phenotype_cols]
    method_color_map: dict[str, str] = dict(shared.get("colors", {}))
    method_label_map: dict[str, str] = dict(shared.get("labels", {}))
    method_color_defaults = list(shared.get("color_defaults", []))

    tab_data, tab_style, tab_axes, tab_text = st.tabs(
        [
            "1 · What to compare",
            "2 · Plot style",
            "3 · Y-axis & outliers",
            "4 · Text, size & significance",
        ]
    )

    with tab_data:
        if p1_method_cols:
            st.caption(
                "Methods, display names and colors come from **Shared Figure Settings** at the top of the tab: "
                + ", ".join(f"{method_label_map.get(m, m)} ({m})" for m in p1_method_cols)
            )
        else:
            st.warning("Select at least one phenotyping method column in Shared Figure Settings.")

        p1a, p1b = st.columns(2)
        p1_pool = _phenotype_pool(run_df, p1_method_cols)
        p1_selected_phenotypes = p1a.multiselect(
            "Phenotypes to compare (one figure per phenotype)",
            options=p1_pool,
            default=p1_pool[: min(3, len(p1_pool))],
            key="tab8_density_p1_phenos",
            help="Each selected phenotype gets its own figure; the phenotype name is written above the plot.",
        )
        p1_strat_cols = p1b.multiselect(
            "Split figures by (optional stratification columns)",
            options=non_coord_run_cols,
            default=[],
            key="tab8_density_p1_strat_cols",
            help="If selected, each figure is split into one sub-panel per unique stratification combination.",
        )

        p1_filter_cols = st.multiselect(
            "Keep only rows matching (optional Panel 1 filters)",
            options=non_coord_run_cols,
            default=[],
            key="tab8_density_p1_filter_cols",
            help="Applied on top of the global filters from Shared Configuration.",
        )
        p1_filter_values = _render_filter_widgets(run_df, p1_filter_cols, key_prefix="tab8_density_p1")

    with tab_style:
        st.markdown("**Distribution geometry**")
        s1, s2, s3 = st.columns(3)
        p1_plot_type = s1.selectbox(
            "Plot type",
            options=["box", "violin", "box+violin"],
            index=0,
            key="tab8_density_p1_plot_type",
            help="Choose the main distribution geometry.",
        )
        p1_distribution_width = s2.slider(
            "Box/violin width",
            min_value=0.35,
            max_value=0.95,
            value=0.78,
            step=0.05,
            key="tab8_density_p1_distribution_width",
        )
        p1_box_alpha = s3.slider(
            "Box/violin fill opacity",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            key="tab8_density_p1_fill_alpha",
        )

        st.markdown("**Dots (one per connected tissue)**")
        s4, s5, s6 = st.columns(3)
        p1_dot_size = s4.slider(
            "Dot size",
            min_value=2,
            max_value=40,
            value=9,
            step=1,
            key="tab8_density_p1_dot_size",
        )
        p1_dot_alpha = s5.slider(
            "Dot opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="tab8_density_p1_dot_alpha",
        )
        p1_jitter_width = s6.slider(
            "Dot horizontal spread",
            min_value=0.02,
            max_value=0.35,
            value=0.08,
            step=0.01,
            key="tab8_density_p1_jitter_width",
        )

        st.markdown("**Line widths**")
        l1, l2, l3, l4 = st.columns(4)
        p1_box_linewidth = l1.slider(
            "Box/whisker (pt)",
            min_value=0.3,
            max_value=2.5,
            value=1.0,
            step=0.1,
            key="tab8_density_p1_box_linewidth",
        )
        p1_axis_linewidth = l2.slider(
            "Axis spine (pt)",
            min_value=0.3,
            max_value=2.0,
            value=0.8,
            step=0.1,
            key="tab8_density_p1_axis_linewidth",
        )
        p1_grid_linewidth = l3.slider(
            "Grid line (pt)",
            min_value=0.3,
            max_value=1.5,
            value=0.7,
            step=0.1,
            key="tab8_density_p1_grid_linewidth",
        )
        p1_sig_bracket_linewidth = l4.slider(
            "Significance bracket (pt)",
            min_value=0.3,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="tab8_density_p1_sig_bracket_linewidth",
            help="Line width for significance test bracket lines.",
        )

    with tab_axes:
        st.markdown("**Scale and range**")
        y1, y2 = st.columns([2, 3])
        with y1:
            p1_y_scale = st.selectbox(
                "Y-axis scale",
                options=["linear", "log10", "log2"],
                index=0,
                key="tab8_density_p1_y_scale",
                help="Linear, log₁₀, or log₂ scale. Log scales clip non-positive densities to a small visible floor.",
            )
        with y2:
            st.caption("Auto-adapts range to keep all data and significance labels visible.")

        y3, y4, y5 = st.columns(3)
        p1_manual_y_min = y3.checkbox(
            "Apply minimum value",
            value=False,
            key="tab8_density_p1_manual_y_min",
            help="Check to override auto y-min."
        )
        p1_y_min_value = y3.number_input(
            "Y min value",
            value=0.0,
            step=10.0,
            format="%.4f",
            key="tab8_density_p1_y_min_value",
        )
        p1_manual_y_max = y4.checkbox(
            "Apply maximum value",
            value=False,
            key="tab8_density_p1_manual_y_max",
            help="Check to override auto y-max. (Will auto-expand if needed for visibility.)"
        )
        p1_y_max_value = y4.number_input(
            "Y max value",
            value=2000.0,
            step=10.0,
            format="%.4f",
            key="tab8_density_p1_y_max_value",
        )
        p1_enable_split_range = y5.checkbox(
            "Use split y-axis",
            value=False,
            key="tab8_density_p1_enable_split_range",
            help="Show two y-ranges and skip the middle interval.",
        )

        st.divider()
        st.markdown("**Crop tall outliers (auto-zoom on the bulk of the data)**")
        st.caption(
            "Cuts the y-axis below the extreme values and hides those dots, so a handful of very high "
            "connected tissues stop flattening the boxes. Boxes, whiskers and all statistics keep using the full data."
        )
        oc1, oc2, oc3, oc4 = st.columns(4)
        p1_crop_mode = oc1.selectbox(
            "Outlier cropping",
            options=["off", "iqr", "percentile"],
            index=0,
            key="tab8_density_p1_crop_mode",
            format_func=lambda x: {
                "off": "Off (show everything)",
                "iqr": "Auto (Tukey IQR fence)",
                "percentile": "Percentile cut",
            }.get(x, x),
        )
        p1_crop_iqr_multiplier = oc2.number_input(
            "IQR multiplier",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.25,
            format="%.2f",
            key="tab8_density_p1_crop_iqr",
            help="Upper limit = Q3 + k × IQR across all compared methods. 1.5 is the standard Tukey fence.",
            disabled=p1_crop_mode != "iqr",
        )
        p1_crop_percentile = oc3.number_input(
            "Keep up to percentile",
            min_value=50.0,
            max_value=100.0,
            value=99.0,
            step=0.5,
            format="%.1f",
            key="tab8_density_p1_crop_percentile",
            help="Values above this percentile are cropped from the view.",
            disabled=p1_crop_mode != "percentile",
        )
        p1_crop_annotate = oc4.checkbox(
            "Note cropped points under figure",
            value=True,
            key="tab8_density_p1_crop_annotate",
            disabled=p1_crop_mode == "off",
            help="Adds a small line below the figure stating how many points were hidden and the true maximum.",
        )

        st.divider()
        st.markdown("**Split y-axis bounds** (only used when *Use split y-axis* is checked)")
        y6, y7, y8, y9 = st.columns(4)
        p1_split_low_min = y6.number_input(
            "Lower range min",
            value=0.0,
            step=10.0,
            format="%.4f",
            key="tab8_density_p1_split_low_min",
        )
        p1_split_low_max = y7.number_input(
            "Lower range max",
            value=500.0,
            step=10.0,
            format="%.4f",
            key="tab8_density_p1_split_low_max",
        )
        p1_split_high_min = y8.number_input(
            "Upper range min",
            value=800.0,
            step=10.0,
            format="%.4f",
            key="tab8_density_p1_split_high_min",
        )
        p1_split_high_max = y9.number_input(
            "Upper range max",
            value=1000.0,
            step=10.0,
            format="%.4f",
            key="tab8_density_p1_split_high_max",
        )
        p1_split_gap_height = st.number_input(
            "Split gap height",
            min_value=5.0,
            max_value=120.0,
            value=40.0,
            step=1.0,
            format="%.0f",
            key="tab8_density_p1_split_gap_height",
            help="Visual gap size between lower and upper ranges in plotted units.",
        )

        st.divider()
        st.markdown("**Axis break** (compress, instead of crop, the tall outliers)")
        ab1, ab2, ab3 = st.columns(3)
        p1_enable_axis_break = ab1.checkbox(
            "Enable axis break",
            value=False,
            key="tab8_density_p1_enable_axis_break",
        )
        p1_break_percentile = ab2.number_input(
            "Outlier percentile",
            min_value=50.0,
            max_value=99.9,
            value=90.0,
            step=1.0,
            format="%.1f",
            key="tab8_density_p1_break_percentile",
            help="Values above this percentile across all compared methods are visually compressed.",
        )
        p1_break_height = ab3.number_input(
            "Compressed height factor",
            min_value=0.05,
            max_value=0.4,
            value=0.15,
            step=0.05,
            format="%.2f",
            key="tab8_density_p1_break_height",
            help="Fraction of original plot height allocated for visually compressed outliers mapping.",
        )

    with tab_text:
        st.markdown("**Figure size**")
        d1, d2, d3 = st.columns(3)
        p1_fig_width = d1.slider(
            "Width per sub-panel (in)",
            min_value=3.0,
            max_value=12.0,
            value=5.0,
            step=0.5,
            key="tab8_density_p1_fig_width",
        )
        p1_fig_height = d2.slider(
            "Figure height (in)",
            min_value=3.5,
            max_value=9.0,
            value=5.5,
            step=0.5,
            key="tab8_density_p1_fig_height",
        )
        p1_panel_spacing = d3.slider(
            "Space between sub-panels",
            min_value=0.2,
            max_value=2.5,
            value=0.8,
            step=0.1,
            key="tab8_density_p1_panel_spacing",
        )

        st.markdown("**Fonts**")
        f1, f2, f3, f4 = st.columns(4)
        p1_title_fontsize = f1.slider(
            "Title font (pt)",
            min_value=8,
            max_value=18,
            value=11,
            step=1,
            key="tab8_density_p1_title_fontsize",
        )
        p1_axis_label_fontsize = f2.slider(
            "Axis label (pt)",
            min_value=8,
            max_value=16,
            value=10,
            step=1,
            key="tab8_density_p1_axis_label_fontsize",
        )
        p1_tick_fontsize = f3.slider(
            "Tick label (pt)",
            min_value=7,
            max_value=14,
            value=9,
            step=1,
            key="tab8_density_p1_tick_fontsize",
        )
        p1_sig_fontsize = f4.slider(
            "Significance (pt)",
            min_value=7,
            max_value=13,
            value=10,
            step=1,
            key="tab8_density_p1_sig_fontsize",
            help="Font size for p-value significance labels on brackets.",
        )

        st.markdown("**Title, labels and significance**")
        t1, t2, t3 = st.columns(3)
        p1_show_title = t1.checkbox(
            "Show phenotype title above plot",
            value=True,
            key="tab8_density_p1_show_title",
            help="The title is always placed above the significance brackets, so the two never overlap.",
        )
        p1_xlabel_rotation = t2.slider(
            "X-label tilt (°)",
            min_value=0,
            max_value=90,
            value=25,
            step=1,
            key="tab8_density_p1_xlabel_rotation",
        )
        p1_show_mean_sem = t3.checkbox(
            "Show mean ± SEM marker",
            value=True,
            key="tab8_density_p1_show_mean_sem",
            help="This is the black circle/error bar in each method column.",
        )
        t4, t5 = st.columns(2)
        p1_show_significance = t4.checkbox(
            "Show pairwise significance brackets",
            value=True,
            key="tab8_density_p1_show_significance",
        )
        p1_show_ns = t5.checkbox(
            "Include non-significant labels (ns)",
            value=False,
            key="tab8_density_p1_show_ns",
        )

    col_p1_btns = st.columns([1, 1])
    run_panel1 = col_p1_btns[0].button("Run Panel 1", type="primary", key="tab8_density_p1_run", width="stretch")
    save_panel1 = col_p1_btns[1].button("Save Parameters", type="secondary", key="tab8_density_p1_save", width="stretch", help="Save Panel 1 parameters to the project config.")

    if save_panel1:
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab8_analysis")
        st.success("Project settings saved to configuration file.")

    if run_panel1:
        if not p1_method_cols:
            st.error("Panel 1: Select at least one phenotyping method column.")
        elif not p1_selected_phenotypes:
            st.error("Panel 1: Select at least one phenotype to compare.")
        else:
            with st.status("Running Panel 1...", expanded=True) as status:
                log_density("Panel 1 started")
                status.write("Applying panel-specific filters.")
                p1_df = _apply_filter_map(run_df, p1_filter_values)
                log_density(f"Panel 1 rows after filters: {len(p1_df)}")

                grouped_p1 = p1_df.groupby(connected_cols, dropna=False)
                area_rows = []
                status.write("Computing connected tissue areas.")
                for key, gdf in grouped_p1:
                    gdf = gdf.dropna(subset=REQUIRED_COORDS)
                    if len(gdf) < int(min_cells_group):
                        continue
                    coords = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                    area_mm2, buffer_px = _virtual_area_mm2(
                        coords=coords,
                        pixel_size_um=pixel_size_um,
                        percentile=percentile_value,
                        min_hole_size=min_hole_size,
                    )
                    if not np.isfinite(area_mm2) or area_mm2 <= 0:
                        continue
                    if not isinstance(key, tuple):
                        key = (key,)
                    row: dict[str, object] = {col: val for col, val in zip(connected_cols, key)}
                    row.update(
                        {
                            "area_mm2": float(area_mm2),
                            "buffer_distance_px": float(buffer_px) if np.isfinite(buffer_px) else np.nan,
                            "group_n_cells": int(len(gdf)),
                        }
                    )
                    for s_col in p1_strat_cols:
                        row[s_col] = gdf[s_col].iloc[0] if s_col in gdf.columns and len(gdf) > 0 else None
                    area_rows.append(row)

                if not area_rows:
                    status.update(label="Panel 1 finished with no output", state="error")
                    log_density("Panel 1 ended with no valid connected tissues")
                    st.warning("Panel 1: No valid connected tissue groups after filters/thresholds.")
                else:
                    area_df = pd.DataFrame(area_rows)
                    status.write(f"Built area table for {len(area_df)} connected tissues.")
                    value_rows = []
                    for _, area_row in area_df.iterrows():
                        mask = np.ones(len(p1_df), dtype=bool)
                        for col in connected_cols:
                            mask &= p1_df[col].astype(str).to_numpy() == str(area_row[col])
                        gdf = p1_df[mask]

                        for pheno in p1_selected_phenotypes:
                            for method_col in p1_method_cols:
                                n_cells = int((gdf[method_col].astype(str) == str(pheno)).sum())
                                density = float(n_cells / area_row["area_mm2"]) if area_row["area_mm2"] > 0 else np.nan
                                out = {
                                    "phenotype": str(pheno),
                                    "method": str(method_col),
                                    "density_cells_per_mm2": density,
                                    "n_cells": n_cells,
                                    "area_mm2": float(area_row["area_mm2"]),
                                    "group_n_cells": int(area_row["group_n_cells"]),
                                }
                                for col in connected_cols:
                                    out[col] = area_row[col]
                                for s_col in p1_strat_cols:
                                    out[s_col] = area_row.get(s_col, None)
                                value_rows.append(out)

                    p1_results = pd.DataFrame(value_rows)
                    if p1_results.empty:
                        status.update(label="Panel 1 finished with no output", state="error")
                        log_density("Panel 1 produced an empty results dataframe")
                        st.warning("Panel 1: No output to display.")
                    else:
                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_csv = out_dir / "panel1_phenotype_method_density.csv"
                        p1_results.to_csv(out_csv, index=False)
                        log_density(f"Panel 1 wrote results CSV to {out_csv}")

                        plot_paths: dict[str, str] = {}
                        stats_by_pheno: dict[str, pd.DataFrame] = {}
                        footer_texts: dict[str, str] = {}
                        output_files: list[Path] = [out_csv]
                        panel1_scale_notes: set[str] = set()
                        panel1_crop_notes: set[str] = set()
                        for pheno in p1_selected_phenotypes:
                            pheno_df = p1_results[p1_results["phenotype"] == str(pheno)].copy()
                            if pheno_df.empty:
                                continue

                            if p1_strat_cols:
                                pheno_df["__stratum__"] = pheno_df[p1_strat_cols].astype(str).agg(" | ".join, axis=1)
                                strata = sorted(pheno_df["__stratum__"].dropna().unique().tolist())
                            else:
                                pheno_df["__stratum__"] = "All"
                                strata = ["All"]

                            fig, axes = plt.subplots(
                                1,
                                len(strata),
                                figsize=(float(p1_fig_width) * len(strata), float(p1_fig_height)),
                                squeeze=False,
                            )
                            axes = axes.flatten()
                            fig_crop_notes: list[str] = []

                            for ax, stratum in zip(axes, strata):
                                ax.set_box_aspect(1)
                                sdf = pheno_df[pheno_df["__stratum__"] == stratum].copy()
                                methods = [m for m in p1_method_cols if m in sdf["method"].unique().tolist()]
                                if not methods:
                                    ax.axis("off")
                                    continue

                                raw_data_arrays = [sdf[sdf["method"] == m]["density_cells_per_mm2"].to_numpy(dtype=float) for m in methods]
                                data_arrays, log_floor = _panel1_prepare_display_arrays(raw_data_arrays, p1_y_scale)
                                panel_y_scale = str(p1_y_scale)
                                if panel_y_scale != "linear" and log_floor is None:
                                    panel_y_scale = "linear"
                                    panel1_scale_notes.add(f"{pheno} | {stratum}: no positive densities available for {p1_y_scale}; used linear scale.")
                                
                                split_enabled = bool(p1_enable_split_range)
                                split_params: tuple[float, float, float, float, float] | None = None
                                if split_enabled and panel_y_scale != "linear":
                                    panel_y_scale = "linear"
                                    panel1_scale_notes.add(f"{pheno} | {stratum}: split y-axis requires linear scale; used linear scale.")

                                if split_enabled:
                                    low_min = float(p1_split_low_min)
                                    low_max = float(p1_split_low_max)
                                    high_min = float(p1_split_high_min)
                                    high_max = float(p1_split_high_max)
                                    gap_height = float(p1_split_gap_height)
                                    split_valid = (
                                        np.isfinite(low_min)
                                        and np.isfinite(low_max)
                                        and np.isfinite(high_min)
                                        and np.isfinite(high_max)
                                        and np.isfinite(gap_height)
                                        and low_min < low_max < high_min < high_max
                                        and gap_height > 0
                                    )
                                    if split_valid:
                                        split_params = (low_min, low_max, high_min, high_max, gap_height)
                                    else:
                                        split_enabled = False
                                        panel1_scale_notes.add(
                                            f"{pheno} | {stratum}: invalid split y-axis values; expected low_min < low_max < high_min < high_max."
                                        )

                                plot_data_arrays = data_arrays
                                if split_enabled and split_params is not None:
                                    plot_data_arrays = [
                                        _panel1_transform_array_for_split(arr, *split_params)
                                        for arr in data_arrays
                                    ]
                                    if not any(arr.size > 0 for arr in plot_data_arrays):
                                        split_enabled = False
                                        split_params = None
                                        plot_data_arrays = data_arrays
                                        panel1_scale_notes.add(
                                            f"{pheno} | {stratum}: no data points inside selected split ranges; split y-axis disabled for this stratum."
                                        )

                                use_axis_break, break_threshold, break_max = _panel1_compute_axis_break(
                                    plot_data_arrays, (p1_enable_axis_break and (not split_enabled)), float(p1_break_percentile)
                                )
                                break_height_factor = float(p1_break_height) if (p1_enable_axis_break and (not split_enabled)) else 0.0
                                x_positions = np.arange(1, len(methods) + 1)

                                # Crop tall outliers out of the view (data and statistics stay complete).
                                crop_info = None
                                if str(p1_crop_mode) != "off" and not split_enabled and not use_axis_break:
                                    crop_info = _panel1_compute_outlier_crop(
                                        plot_data_arrays,
                                        str(p1_crop_mode),
                                        float(p1_crop_percentile),
                                        float(p1_crop_iqr_multiplier),
                                    )
                                    if crop_info is not None:
                                        panel1_crop_notes.add(
                                            f"{pheno} | {stratum}: {crop_info[1]} point(s) above "
                                            f"{crop_info[0]:.4g} cropped from the view (max {crop_info[2]:.4g})."
                                        )

                                if p1_plot_type in {"violin", "box+violin"}:
                                    violin = ax.violinplot(
                                        plot_data_arrays,
                                        positions=x_positions,
                                        showmeans=False,
                                        showmedians=False,
                                        showextrema=False,
                                        widths=float(p1_distribution_width),
                                    )
                                    for i, body in enumerate(violin["bodies"]):
                                        method = methods[i]
                                        body.set_facecolor(method_color_map.get(method, method_color_defaults[i % len(method_color_defaults)]))
                                        body.set_alpha(max(0.12, float(p1_box_alpha) - 0.1))
                                        body.set_edgecolor("#444444")
                                        body.set_linewidth(0.8)

                                if p1_plot_type in {"box", "box+violin"}:
                                    box = ax.boxplot(
                                        plot_data_arrays,
                                        labels=[method_label_map.get(m, m) for m in methods],
                                        showfliers=False,
                                        patch_artist=True,
                                        positions=x_positions,
                                        widths=float(p1_distribution_width),
                                        medianprops={"color": "black", "linewidth": float(p1_box_linewidth)},
                                        whiskerprops={"color": "#333333", "linewidth": float(p1_box_linewidth)},
                                        capprops={"color": "#333333", "linewidth": float(p1_box_linewidth)},
                                        boxprops={"edgecolor": "#333333", "linewidth": float(p1_box_linewidth)},
                                    )
                                    for i, patch in enumerate(box["boxes"]):
                                        method = methods[i]
                                        patch.set_facecolor(method_color_map.get(method, method_color_defaults[i % len(method_color_defaults)]))
                                        patch.set_alpha(float(p1_box_alpha))

                                for i, method in enumerate(methods, start=1):
                                    y = plot_data_arrays[i - 1]
                                    if len(y) == 0:
                                        continue
                                    
                                    if use_axis_break:
                                        y_display = np.array([_panel1_transform_for_break(yv, break_threshold, break_height_factor) for yv in y])
                                    else:
                                        y_display = y

                                    if crop_info is not None:
                                        y_display = y_display[y_display <= float(crop_info[0])]

                                    method_color = method_color_map.get(method, method_color_defaults[(i - 1) % len(method_color_defaults)])
                                    if len(y_display):
                                        jitter = np.random.uniform(-float(p1_jitter_width), float(p1_jitter_width), size=len(y_display))
                                        ax.scatter(
                                            np.full(len(y_display), i) + jitter,
                                            y_display,
                                            s=float(p1_dot_size),
                                            alpha=float(p1_dot_alpha),
                                            c=method_color,
                                            edgecolors="white",
                                            linewidths=0.25,
                                            zorder=4,
                                        )

                                    if p1_show_mean_sem:
                                        mean_y = float(np.nanmean(y)) if len(y) else np.nan
                                        sem_y = float(np.nanstd(y, ddof=1) / np.sqrt(len(y))) if len(y) > 1 else 0.0
                                        if np.isfinite(mean_y):
                                            mean_display = _panel1_transform_for_break(mean_y, break_threshold, break_height_factor) if use_axis_break else mean_y
                                            sem_display = sem_y * 0.5 if use_axis_break else sem_y
                                            ax.errorbar(
                                                [i],
                                                [mean_display],
                                                yerr=[sem_display],
                                                fmt="o",
                                                color="#111111",
                                                markersize=4.2,
                                                elinewidth=float(p1_box_linewidth),
                                                capsize=2.5,
                                                zorder=5,
                                            )

                                pivot_sig = sdf.pivot_table(
                                    index=connected_cols,
                                    columns="method",
                                    values="density_cells_per_mm2",
                                    aggfunc="first",
                                )
                                matched_stats = compute_matched_method_statistics(
                                    pivot_sig,
                                    methods,
                                    analysis_unit="connected_tissue",
                                )
                                if matched_stats.empty:
                                    pairwise_for_plot = []
                                else:
                                    pairwise_for_plot = matched_stats[
                                        matched_stats["test"].eq("paired_wilcoxon_signed_rank")
                                    ][["method_1", "method_2", "p_value"]].to_dict("records")

                                non_empty_arrays = [arr for arr in plot_data_arrays if len(arr) > 0]
                                finite_vals = np.concatenate(non_empty_arrays) if non_empty_arrays else np.array([0.0])
                                y_max = float(np.nanmax(finite_vals)) if finite_vals.size > 0 else 1.0
                                y_min = float(np.nanmin(finite_vals)) if finite_vals.size > 0 else 0.0
                                pairwise_for_plot_show = pairwise_for_plot
                                if not p1_show_ns:
                                    pairwise_for_plot_show = [
                                        row
                                        for row in pairwise_for_plot
                                        if np.isfinite(row.get("p_value", np.nan)) and float(row["p_value"]) < 0.05
                                    ]

                                y_top = y_max
                                title_base = 1.0
                                if p1_show_significance:
                                    y_top = _panel1_add_significance_brackets(
                                        ax=ax,
                                        methods=methods,
                                        pairwise_stats=pairwise_for_plot_show,
                                        y_data_max=y_max,
                                        y_data_min=y_min,
                                        y_scale_mode=panel_y_scale,
                                        sig_bracket_linewidth=float(p1_sig_bracket_linewidth),
                                        sig_fontsize=int(p1_sig_fontsize),
                                        use_axis_break=use_axis_break,
                                        break_threshold=break_threshold if use_axis_break else np.nan,
                                        break_height=break_height_factor,
                                    )
                                    title_base = _bracket_top_axes_fraction(methods, pairwise_for_plot_show)

                                axis_y_max = float(crop_info[0]) if crop_info is not None else y_max
                                axis_y_top = float(crop_info[0]) if crop_info is not None else y_top
                                _panel1_apply_y_axis(
                                    ax=ax,
                                    y_scale_mode=panel_y_scale,
                                    y_data_min=y_min,
                                    y_data_max=axis_y_max,
                                    y_top=axis_y_top,
                                    manual_y_min_enabled=bool(p1_manual_y_min),
                                    manual_y_min=float(p1_y_min_value),
                                    manual_y_max_enabled=bool(p1_manual_y_max),
                                    manual_y_max=float(p1_y_max_value),
                                )
                                if crop_info is not None:
                                    crop_limit, crop_hidden, crop_hidden_max = crop_info
                                    lower_limit, upper_limit = ax.get_ylim()
                                    headroom = max(abs(upper_limit - lower_limit) * 0.05, 1e-9)
                                    ax.set_ylim(lower_limit, float(crop_limit) + headroom)
                                    note = f"{crop_hidden} point(s) > {crop_limit:.4g} not shown (max {crop_hidden_max:.4g})"
                                    fig_crop_notes.append(note if len(strata) < 2 else f"{stratum}: {note}")
                                if split_enabled and split_params is not None:
                                    _panel1_apply_split_axis(ax, *split_params)

                                if p1_show_title:
                                    # Sits above the significance brackets so the two never overlap.
                                    ax.set_title(
                                        f"{pheno} | {stratum}",
                                        fontsize=int(p1_title_fontsize),
                                        y=float(title_base) + 0.012,
                                    )
                                y_axis_label = "Density (cells/mm²)"
                                if panel_y_scale == "log10":
                                    y_axis_label += " | log10 scale"
                                elif panel_y_scale == "log2":
                                    y_axis_label += " | log2 scale"
                                ax.set_ylabel(y_axis_label, fontsize=int(p1_axis_label_fontsize))
                                ax.set_xlabel("Phenotyping Method", fontsize=int(p1_axis_label_fontsize))
                                ax.set_xticks(x_positions)
                                ax.set_xticklabels([method_label_map.get(m, m) for m in methods], fontsize=int(p1_tick_fontsize))
                                ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=float(p1_grid_linewidth))
                                ax.tick_params(axis="x", rotation=int(p1_xlabel_rotation), labelsize=int(p1_tick_fontsize))
                                ax.tick_params(axis="y", labelsize=int(p1_tick_fontsize))
                                ax.spines["top"].set_visible(False)
                                ax.spines["right"].set_visible(False)
                                ax.spines["left"].set_linewidth(float(p1_axis_linewidth))
                                ax.spines["bottom"].set_linewidth(float(p1_axis_linewidth))
                                
                                if use_axis_break:
                                    _panel1_draw_axis_break(ax, break_threshold, break_height_factor)

                            footer_text = (
                                "Dots: connected tissue units (batch/ROI groups). Black marker: mean ± SEM. "
                                "Brackets: paired two-sided Wilcoxon signed-rank tests with Holm-adjusted P values."
                            )
                            if p1_y_scale != "linear":
                                footer_text += " Non-positive densities are clipped to a small visible floor for log-scale display."
                            if panel1_scale_notes:
                                footer_text += " Some strata used fallback axis behavior based on data availability or axis settings."
                            pheno_crop_notes = sorted(note for note in panel1_crop_notes if note.startswith(f"{pheno} | "))
                            if pheno_crop_notes:
                                footer_text += " Outlier cropping affects the view only: " + " ".join(pheno_crop_notes)
                            footer_texts[str(pheno)] = footer_text

                            plt.tight_layout(w_pad=float(p1_panel_spacing))
                            if p1_crop_annotate and fig_crop_notes:
                                # Drawn under the figure, where it cannot collide with data or brackets.
                                fig.text(
                                    0.99,
                                    -0.015,
                                    "Cropped view: " + "; ".join(fig_crop_notes),
                                    ha="right",
                                    va="top",
                                    fontsize=max(6, int(p1_tick_fontsize) - 1),
                                    color="#444444",
                                )
                            fig_path = out_dir / f"panel1_{str(pheno).replace(' ', '_')}.png"
                            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                            svg_path = fig_path.with_suffix(".svg")
                            fig.savefig(svg_path, bbox_inches="tight")
                            plt.close(fig)
                            plot_paths[str(pheno)] = str(fig_path)
                            output_files.extend([fig_path, svg_path])

                            stats_rows = []
                            for stratum in strata:
                                sdf = pheno_df[pheno_df["__stratum__"] == stratum].copy()
                                pivot = sdf.pivot_table(
                                    index=connected_cols,
                                    columns="method",
                                    values="density_cells_per_mm2",
                                    aggfunc="first",
                                )
                                valid_methods = [m for m in p1_method_cols if m in pivot.columns]
                                matched = compute_matched_method_statistics(
                                    pivot,
                                    valid_methods,
                                    analysis_unit="batch_id x tissue_roi",
                                )
                                if matched.empty:
                                    continue
                                matched.insert(0, "stratum", str(stratum))
                                matched.insert(0, "phenotype", str(pheno))
                                stats_rows.extend(matched.to_dict("records"))

                            stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame()
                            if not stats_df.empty:
                                stats_csv = out_dir / f"panel1_{str(pheno).replace(' ', '_')}_stats.csv"
                                stats_df.to_csv(stats_csv, index=False)
                                output_files.append(stats_csv)
                            stats_by_pheno[str(pheno)] = stats_df

                        bundle_dir = create_publication_bundle(out_dir, "density_panel1")
                        copied_files = copy_outputs_to_bundle(output_files, bundle_dir)
                        metadata = {
                            "panel": "Density Panel 1 - phenotype density across methods",
                            "statistics": {
                                "omnibus": "Friedman test across matched connected tissues",
                                "post_hoc": "two-sided paired Wilcoxon signed-rank tests",
                                "multiplicity": "Holm correction across all method pairs within each phenotype and stratum",
                                "effect_sizes": ["Kendall_W", "matched_rank_biserial"],
                                "significance_brackets_use": "p_value_adjusted",
                            },
                            "analysis": {
                                "methods": p1_method_cols,
                                "phenotypes": [str(value) for value in p1_selected_phenotypes],
                                "connected_tissue_columns": connected_cols,
                                "n_connected_tissues": int(area_df.shape[0]),
                                "n_cells_across_connected_tissues": int(area_df["group_n_cells"].sum()),
                                "pixel_size_um": float(pixel_size_um),
                                "virtual_area_method": "buffer_union",
                                "nearest_neighbor_buffer_percentile": percentile_value,
                                "minimum_hole_size_px2": float(min_hole_size),
                                "minimum_cells_per_connected_tissue": int(min_cells_group),
                                "filters": p1_filter_values,
                                "stratification_columns": p1_strat_cols,
                            },
                            "display": {
                                "plot_type": p1_plot_type,
                                "y_scale": p1_y_scale,
                                "outlier_crop_mode": p1_crop_mode,
                                "outlier_crop_percentile": float(p1_crop_percentile),
                                "outlier_crop_affects_statistics": False,
                                "png_dpi": 300,
                            },
                            "files": copied_files,
                        }
                        metadata_path = bundle_dir / "analysis_metadata.json"
                        write_publication_metadata(metadata_path, metadata)

                        st.session_state[_DENSITY_PANEL1_RESULTS_KEY] = {
                            "results_df": p1_results,
                            "results_csv": str(out_csv),
                            "connected_tissues": int(area_df.shape[0]),
                            "phenotypes": [str(p) for p in p1_selected_phenotypes],
                            "plot_paths": plot_paths,
                            "stats_by_pheno": stats_by_pheno,
                            "footer_texts": footer_texts,
                            "publication_bundle": str(bundle_dir),
                        }
                        st.success("Panel 1 calculation complete.")

    panel1_state = st.session_state.get(_DENSITY_PANEL1_RESULTS_KEY)
    if panel1_state:
        _render_panel1_results(panel1_state)


@st.fragment
def render_panel2_fragment(
    run_df: pd.DataFrame,
    non_coord_run_cols: list[str],
    shared: dict[str, Any],
    pixel_size_um: float,
    min_hole_size: float,
    min_cells_group: int,
    output_dir: str,
    percentile_value: float | None,
    connected_cols: list[str],
    phenotype_cols: list[str],
    project_root: Path,
):
    st.caption(
        "Profiles phenotype density in distance zones around a reference boundary "
        "(negative = inside, positive = outside). Edits stay local until you click Run Panel 2."
    )

    method_color_map: dict[str, str] = dict(shared.get("colors", {}))
    method_label_map: dict[str, str] = dict(shared.get("labels", {}))
    method_color_defaults = list(shared.get("color_defaults", [])) or [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2",
    ]
    shared_methods = [str(m) for m in shared.get("methods", [])]

    tab_data, tab_zones, tab_style, tab_filters = st.tabs(
        [
            "1 · What to measure",
            "2 · Zones & statistics",
            "3 · Plot style",
            "4 · Filters & stratum colors",
        ]
    )

    with tab_data:
        st.markdown("**Phenotype to profile**")
        # Backward compatibility for old projects
        if "tab8_density_p2_pheno_col" in st.session_state and "tab8_density_p2_pheno_cols" not in st.session_state:
            old_val = st.session_state["tab8_density_p2_pheno_col"]
            st.session_state["tab8_density_p2_pheno_cols"] = [old_val] if old_val else []

        p2c4, p2c5 = st.columns(2)
        p2_pheno_cols = p2c4.multiselect(
            "Phenotype column(s) to measure",
            options=non_coord_run_cols,
            default=shared_methods if shared_methods else ([non_coord_run_cols[0]] if non_coord_run_cols else []),
            key="tab8_density_p2_pheno_cols",
            help="Columns holding the phenotype labels to profile. Defaults to the shared phenotyping methods.",
        )
        p2_target_value_options = []
        if p2_pheno_cols:
            vals: set[str] = set()
            for col in p2_pheno_cols:
                if col in run_df.columns:
                    vals.update(run_df[col].dropna().astype(str).unique().tolist())
            p2_target_value_options = sorted(list(vals))

        p2_selected_phenotypes = p2c5.multiselect(
            "Value(s) to profile (one figure per value)",
            options=p2_target_value_options,
            key="tab8_density_p2_target_values",
            help="Select one or more values in the chosen phenotype/value column.",
        )

        st.markdown("**Reference boundary (distance = 0)**")
        p2c6, p2c7 = st.columns(2)
        p2_annotation_col_options = [""] + non_coord_run_cols
        p2_annotation_col_value = _sync_selectbox_state(
            "tab8_density_p2_annotation_col",
            p2_annotation_col_options,
            "neighbourhood_consensus" if "neighbourhood_consensus" in p2_annotation_col_options else "",
        )
        p2_annotation_col = p2c6.selectbox(
            "Reference annotation column",
            options=p2_annotation_col_options,
            index=p2_annotation_col_options.index(p2_annotation_col_value) if p2_annotation_col_value in p2_annotation_col_options else _default_index(p2_annotation_col_options, "neighbourhood_consensus", fallback=0),
            key="tab8_density_p2_annotation_col",
            format_func=lambda x: x if x else "(not selected)",
            help="Boundary reference source. Example: neighborhood or annotation column containing 'tumor'.",
        )
        p2_ref_value_options = []
        if p2_annotation_col and p2_annotation_col in run_df.columns:
            p2_ref_value_options = sorted(run_df[p2_annotation_col].dropna().astype(str).unique().tolist())
        p2_reference_values = p2c7.multiselect(
            "Reference annotation value(s)",
            options=p2_ref_value_options,
            key="tab8_density_p2_ref_values",
            help="Cells matching these value(s) define the boundary when distance is computed from annotation.",
        )

        numeric_distance_candidates = [
            c
            for c in non_coord_run_cols
            if (
                ("distance" in c.lower())
                or pd.api.types.is_numeric_dtype(run_df[c])
            )
        ]
        p2_distance_col_options = ["None (compute from annotation)"] + numeric_distance_candidates
        p2_distance_col_value = _sync_selectbox_state(
            "tab8_density_p2_distance_col",
            p2_distance_col_options,
            "None (compute from annotation)",
        )
        p2_distance_col = st.selectbox(
            "Or use a pre-computed signed distance column",
            options=p2_distance_col_options,
            index=p2_distance_col_options.index(p2_distance_col_value) if p2_distance_col_value in p2_distance_col_options else 0,
            key="tab8_density_p2_distance_col",
            help="If selected, this column is used directly as signed distance (negative inside, positive outside) and the reference annotation settings above become optional.",
        )

        st.markdown("**Restrict cells before measuring (optional)**")
        p2c1, p2c2 = st.columns(2)
        p2_prefilter_col_options = ["None"] + non_coord_run_cols
        p2_prefilter_col_value = _sync_selectbox_state(
            "tab8_density_p2_prefilter_col",
            p2_prefilter_col_options,
            "tumor_bed" if "tumor_bed" in p2_prefilter_col_options else "None",
        )
        p2_prefilter_col = p2c1.selectbox(
            "Prefilter column",
            options=p2_prefilter_col_options,
            index=p2_prefilter_col_options.index(p2_prefilter_col_value) if p2_prefilter_col_value in p2_prefilter_col_options else _default_index(p2_prefilter_col_options, "tumor_bed", fallback=0),
            key="tab8_density_p2_prefilter_col",
            help="Optional first-pass filter. Example: choose tumor_bed then keep value 1.",
        )
        p2_prefilter_value_options = []
        if p2_prefilter_col != "None" and p2_prefilter_col in run_df.columns:
            p2_prefilter_value_options = sorted(run_df[p2_prefilter_col].dropna().astype(str).unique().tolist())
        p2_prefilter_values = p2c2.multiselect(
            "Prefilter value(s) to keep",
            options=p2_prefilter_value_options,
            key="tab8_density_p2_prefilter_values",
            help="Leave empty to keep all values from the prefilter column.",
        )

    with tab_zones:
        st.markdown("**Distance zones**")
        p2d1, p2d2, p2d3 = st.columns(3)
        min_distance_um = p2d1.number_input(
            "Min distance (µm)",
            min_value=-3000.0,
            max_value=3000.0,
            value=-600.0,
            step=50.0,
            key="tab8_density_p2_min_um",
            help="Minimum distance boundary around the reference boundary (negative inside, positive outside).",
        )
        max_distance_um = p2d2.number_input(
            "Max distance (µm)",
            min_value=-3000.0,
            max_value=3000.0,
            value=600.0,
            step=50.0,
            key="tab8_density_p2_max_um",
            help="Maximum distance boundary around the reference boundary (negative inside, positive outside).",
        )
        zone_width_um = p2d3.number_input(
            "Zone width (µm)",
            min_value=5.0,
            max_value=1000.0,
            value=40.0,
            step=5.0,
            key="tab8_density_p2_zone_um",
            help="Bin width for distance zones.",
        )

        st.markdown("**Inclusion thresholds**")
        p2e1, p2e2 = st.columns(2)
        min_cells_per_zone = p2e1.number_input(
            "Min cells per zone",
            min_value=0,
            max_value=100000,
            value=10,
            step=1,
            key="tab8_density_p2_min_cells_zone",
            help="Zones with fewer cells than this are excluded.",
        )
        min_rois_per_group = p2e2.number_input(
            "Min connected tissues per curve",
            min_value=1,
            max_value=1000,
            value=3,
            step=1,
            key="tab8_density_p2_min_rois",
            help="How many connected tissues must contribute to each zone point.",
        )
        st.caption("Panel 2 uses the exact connected virtual tissue area, so the density metric matches Panel 1.")

        st.markdown("**Stratification and statistics**")
        p2_strat_cols = st.multiselect(
            "Split/overlay curves by (optional stratification columns)",
            options=non_coord_run_cols,
            default=[],
            key="tab8_density_p2_strat_cols",
            help="Curves are split/overlaid by these metadata strata.",
        )
        p2g2, p2g3, p2g4 = st.columns(3)
        p2_overlay_strata = p2g2.checkbox(
            "Overlay stratified curves",
            value=True,
            key="tab8_density_p2_overlay_strata",
            help="When stratification columns are selected, overlay strata in the same panel instead of splitting them into separate panels.",
        )
        p2_show_stats = p2g3.checkbox(
            "Add zone-level stats",
            value=True,
            key="tab8_density_p2_show_stats",
            help="Compare stratified curves at each distance zone and annotate significant differences on the plot.",
        )
        p2_stats_alpha = p2g4.number_input(
            "Stats alpha",
            min_value=0.001,
            max_value=0.2,
            value=0.05,
            step=0.005,
            format="%.3f",
            key="tab8_density_p2_stats_alpha",
        )
        p2g5, p2g6 = st.columns(2)
        p2_overlay_encoding = p2g5.selectbox(
            "Overlay encoding",
            options=["color_by_stratum", "line_style_by_stratum"],
            index=0,
            key="tab8_density_p2_overlay_encoding",
            format_func=lambda x: "Distinct colors by stratum" if x == "color_by_stratum" else "Line style by stratum",
            help="Choose whether overlayed strata are distinguished by colors or by line styles.",
        )
        p2_stats_method = p2g6.selectbox(
            "Stats test for stratified comparison",
            options=["welch_t_test", "mannwhitney_u"],
            index=0,
            key="tab8_density_p2_stats_method",
            format_func=lambda x: "Welch t-test" if x == "welch_t_test" else "Mann-Whitney U",
            help="Used only when stratified overlay/statistics are enabled. Current implementation compares two strata per zone.",
        )

    with tab_style:
        render_zonal_plot_controls(
            "tab8_density_p2",
            note="Line colors and display names come from Shared Figure Settings at the top of the tab.",
        )

    with tab_filters:
        p2_filter_cols = st.multiselect(
            "Keep only rows matching (optional Panel 2 filters)",
            options=non_coord_run_cols,
            default=[],
            key="tab8_density_p2_filter_cols",
            help="Additional filters applied before distance and density calculation.",
        )
        p2_filter_values = _render_filter_widgets(run_df, p2_filter_cols, key_prefix="tab8_density_p2")
        p2_stratum_preview_df = run_df.copy()
        if p2_filter_cols:
            p2_stratum_preview_df = _apply_filter_map(p2_stratum_preview_df, p2_filter_values)
        if p2_prefilter_col != "None" and p2_prefilter_col in p2_stratum_preview_df.columns and p2_prefilter_values:
            p2_stratum_preview_df = p2_stratum_preview_df[p2_stratum_preview_df[p2_prefilter_col].astype(str).isin([str(v) for v in p2_prefilter_values])]
        preview_strata = []
        if p2_strat_cols and not p2_stratum_preview_df.empty:
            preview_strata = sorted(_build_stratum_series(p2_stratum_preview_df, p2_strat_cols).dropna().astype(str).unique().tolist())
        p2_stratum_color_map: dict[str, str] = {}
        if preview_strata and p2_overlay_strata and p2_overlay_encoding == "color_by_stratum":
            st.markdown("**Current stratum colors**")
            st.caption("These labels come from the currently selected stratification columns and active Panel 2 filters. They are not hardcoded.")
            stratum_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]
            color_cols = st.columns(min(3, len(preview_strata)))
            for idx, stratum_name in enumerate(preview_strata):
                with color_cols[idx % len(color_cols)]:
                    p2_stratum_color_map[stratum_name] = st.color_picker(
                        f"{stratum_name}",
                        value=stratum_palette[idx % len(stratum_palette)],
                        key=f"tab8_density_p2_stratum_color_{stratum_name}",
                    )

    col_p2_btns = st.columns([1, 1])
    run_panel2 = col_p2_btns[0].button("Run Panel 2", type="primary", key="tab8_density_p2_run", width="stretch")
    save_panel2 = col_p2_btns[1].button("Save Parameters", type="secondary", key="tab8_density_p2_save", width="stretch")

    if save_panel2:
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab8_analysis")
        st.success("Project settings saved to configuration file.")
        st.rerun()

    if run_panel2:
        if not p2_pheno_cols:
            st.error("Panel 2: Select at least one phenotype/value column to profile.")
        elif not p2_selected_phenotypes:
            st.error("Panel 2: Select at least one value to profile.")
        elif p2_distance_col == "None (compute from annotation)" and (not p2_annotation_col or not p2_reference_values):
            st.error("Panel 2: Select an annotation column and at least one reference value, or choose a signed distance column.")
        elif float(min_distance_um) >= float(max_distance_um):
            st.error("Panel 2: Min distance (µm) must be strictly less than Max distance (µm).")
        else:
            panel2_started = perf_counter()
            with st.status("Running Panel 2...", expanded=True) as status:
                log_density(
                    f"Panel 2 started with pheno_cols={p2_pheno_cols}, annotation_col={p2_annotation_col}, distance_col={p2_distance_col}"
                )
                status.write("Applying panel-specific filters.")
                p2_run = _apply_filter_map(run_df, p2_filter_values)
                if p2_prefilter_col != "None" and p2_prefilter_col in p2_run.columns and p2_prefilter_values:
                    p2_run = p2_run[p2_run[p2_prefilter_col].astype(str).isin([str(v) for v in p2_prefilter_values])]

                p2_run = p2_run.dropna(subset=REQUIRED_COORDS)
                if p2_run.empty:
                    status.update(label="Panel 2 finished with no rows", state="error")
                    log_density("Panel 2 ended with no rows after filters")
                    st.warning("Panel 2: No rows left after filters.")
                else:
                    zone_width_px = float(zone_width_um) / pixel_size_um
                    min_distance_px = float(min_distance_um) / pixel_size_um
                    max_distance_px = float(max_distance_um) / pixel_size_um
                    zone_edges = np.arange(min_distance_px, max_distance_px + zone_width_px, zone_width_px)
                    if len(zone_edges) < 2:
                        status.update(label="Panel 2 invalid zone configuration", state="error")
                        log_density("Panel 2 aborted due to invalid zone configuration")
                        st.error("Panel 2: Invalid zone configuration.")
                    else:
                        zone_rows: list[dict] = []
                        grouped_p2 = p2_run.groupby(connected_cols, dropna=False)
                        status.write("Building zonal densities per connected tissue.")
                        for key, gdf in grouped_p2:
                            if len(gdf) < int(min_cells_group):
                                continue

                            if p2_distance_col != "None (compute from annotation)" and p2_distance_col in gdf.columns:
                                signed_dist = pd.to_numeric(gdf[p2_distance_col], errors="coerce").to_numpy(dtype=float)
                            else:
                                ref_mask = gdf[p2_annotation_col].astype(str).isin([str(v) for v in p2_reference_values]).to_numpy()
                                ref_cells = gdf[ref_mask].dropna(subset=REQUIRED_COORDS)
                                if len(ref_cells) < 3:
                                    continue

                                ref_coords = ref_cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                                ref_hull = MultiPoint([Point(float(x), float(y)) for x, y in ref_coords]).convex_hull
                                ref_boundary = ref_hull.boundary

                                coords_all = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                                signed_dist = np.empty(len(coords_all), dtype=float)
                                for idx_pt, (x_coord, y_coord) in enumerate(coords_all):
                                    pt = Point(float(x_coord), float(y_coord))
                                    dist = float(pt.distance(ref_boundary))
                                    inside = bool(ref_mask[idx_pt])
                                    signed_dist[idx_pt] = -dist if inside else dist

                            zone_ids = np.digitize(signed_dist, zone_edges, right=False) - 1
                            valid_zone_mask = (zone_ids >= 0) & (zone_ids < (len(zone_edges) - 1))
                            if not np.any(valid_zone_mask):
                                continue

                            gdf = gdf.copy()
                            gdf["__distance_from_tumor_px__"] = signed_dist
                            gdf["__zone_idx__"] = zone_ids
                            zone_ready = gdf[valid_zone_mask].copy()
                            if zone_ready.empty:
                                continue

                            if not isinstance(key, tuple):
                                key = (key,)
                            key_map = {col: val for col, val in zip(connected_cols, key)}

                            for z_idx, zdf in zone_ready.groupby("__zone_idx__", sort=True):
                                z_idx_int = int(z_idx)
                                z_min = float(zone_edges[z_idx_int])
                                z_max = float(zone_edges[z_idx_int + 1])
                                z_center_px = (z_min + z_max) / 2.0
                                z_center_um = z_center_px * pixel_size_um
                                if len(zdf) < max(2, int(min_cells_per_zone)):
                                    continue

                                z_coords = zdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                                geom_summary = _panel2_exact_geometry_summary_cached(
                                    coords_key=_coords_cache_key(z_coords),
                                    pixel_size_um=pixel_size_um,
                                    percentile=percentile_value,
                                    min_hole_size=min_hole_size,
                                )
                                area_mm2 = _numeric_dict_value(geom_summary, "area_mm2")
                                if not np.isfinite(area_mm2) or area_mm2 <= 0:
                                    continue

                                for method_col in p2_pheno_cols:
                                    if method_col not in zdf.columns:
                                        continue
                                    counts = zdf[method_col].astype(str).value_counts()
                                    for pheno in p2_selected_phenotypes:
                                        n_pheno = int(counts.get(str(pheno), 0))
                                        density = float(n_pheno / area_mm2)
                                        row: dict[str, object] = {
                                            "phenotype": str(pheno),
                                            "method": str(method_col),
                                            "zone_center_um": float(z_center_um),
                                            "zone_min_um": float(z_min * pixel_size_um),
                                            "zone_max_um": float(z_max * pixel_size_um),
                                            "n_cells_zone": int(len(zdf)),
                                            "n_cells_phenotype": int(n_pheno),
                                            "area_mm2": float(area_mm2),
                                            "density_cells_per_mm2": float(density),
                                        }
                                        row.update(key_map)
                                        for s_col in p2_strat_cols:
                                            row[s_col] = gdf[s_col].iloc[0] if s_col in gdf.columns and len(gdf) > 0 else None
                                        zone_rows.append(row)

                        if not zone_rows:
                            status.update(label="Panel 2 finished with no valid zones", state="error")
                            log_density("Panel 2 produced no valid zone rows")
                            st.warning("Panel 2: No valid zone-level results. Consider relaxing filters or zone thresholds.")
                        else:
                            zone_df = pd.DataFrame(zone_rows)
                            zone_df["__group_id__"] = zone_df[connected_cols].astype(str).apply(lambda row: " | ".join(row), axis=1)

                            group_keys = ["phenotype", "method", "zone_center_um"] + p2_strat_cols
                            roi_counts = (
                                zone_df.groupby(group_keys, dropna=False)["__group_id__"]
                                .nunique()
                                .reset_index(name="n_connected_tissues")
                            )
                            valid_roi = roi_counts[roi_counts["n_connected_tissues"] >= int(min_rois_per_group)]
                            if valid_roi.empty:
                                status.update(label="Panel 2 finished with no valid curves", state="error")
                                log_density("Panel 2 had no curves meeting minimum connected tissue threshold")
                                st.warning("Panel 2: No curves satisfy min connected tissues threshold.")
                            else:
                                zone_df = zone_df.merge(valid_roi[group_keys], on=group_keys, how="inner")
                                if zone_df.empty:
                                    status.update(label="Panel 2 finished with no surviving rows", state="error")
                                    log_density("Panel 2 lost all rows after min connected tissue filtering")
                                    st.warning("Panel 2: No data left after min connected tissues filtering.")
                                else:
                                    summary_df = (
                                        zone_df.groupby(group_keys, dropna=False)
                                        .agg(
                                            density_mean=("density_cells_per_mm2", "mean"),
                                            density_std=("density_cells_per_mm2", "std"),
                                            n_connected_tissues=("__group_id__", "nunique"),
                                            total_cells_zone=("n_cells_zone", "sum"),
                                            total_area_mm2=("area_mm2", "sum"),
                                        )
                                        .reset_index()
                                    )
                                    summary_df["density_sem"] = summary_df["density_std"] / np.sqrt(
                                        np.maximum(summary_df["n_connected_tissues"], 1)
                                    )

                                    out_dir = Path(output_dir)
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    zone_csv = out_dir / "panel2_zonal_density_detailed.csv"
                                    summary_csv = out_dir / "panel2_zonal_density_summary.csv"
                                    zone_df.to_csv(zone_csv, index=False)
                                    summary_df.to_csv(summary_csv, index=False)
                                    log_density(f"Panel 2 wrote outputs to {zone_csv} and {summary_csv}")
                                    output_files: list[Path] = [zone_csv, summary_csv]

                                    method_stats_df = compute_zonal_method_statistics(
                                        zone_df=zone_df,
                                        group_cols=["phenotype", "zone_center_um", *p2_strat_cols],
                                        block_col="__group_id__",
                                        value_col="density_cells_per_mm2",
                                        methods=[str(method) for method in p2_pheno_cols],
                                    )
                                    if not method_stats_df.empty:
                                        method_stats_csv = out_dir / "panel2_zonal_density_method_stats.csv"
                                        method_stats_df.to_csv(method_stats_csv, index=False)
                                        output_files.append(method_stats_csv)

                                    stats_df = pd.DataFrame()
                                    if p2_show_stats and p2_strat_cols:
                                        status.write("Computing stratified zone-level statistics.")
                                        stats_df = _compute_panel2_stats(
                                            zone_df=zone_df,
                                            strat_cols=p2_strat_cols,
                                            stats_method=str(p2_stats_method),
                                            alpha=float(p2_stats_alpha),
                                        )
                                        if not stats_df.empty:
                                            stats_csv = out_dir / "panel2_zonal_density_stats.csv"
                                            stats_df.to_csv(stats_csv, index=False)
                                            output_files.append(stats_csv)
                                            log_density(f"Panel 2 wrote stats CSV to {stats_csv}")

                                    plot_paths: dict[str, str] = {}
                                    for pheno in p2_selected_phenotypes:
                                        ph_df = summary_df[summary_df["phenotype"] == str(pheno)].copy()
                                        if ph_df.empty:
                                            continue

                                        # Keep the shared method order so the legend matches the panels.
                                        present_methods = set(ph_df["method"].dropna().astype(str).unique().tolist())
                                        plot_methods = [str(m) for m in p2_pheno_cols if str(m) in present_methods]
                                        plot_methods += sorted(present_methods - set(plot_methods))
                                        if not plot_methods:
                                            continue

                                        if p2_strat_cols:
                                            ph_df["__stratum__"] = ph_df[p2_strat_cols].astype(str).agg(" | ".join, axis=1)
                                            strata = sorted(ph_df["__stratum__"].dropna().unique().tolist())
                                        else:
                                            ph_df["__stratum__"] = "All"
                                            strata = ["All"]

                                        overlay_active = bool(p2_overlay_strata and p2_strat_cols)
                                        fig = render_zonal_density_figure(
                                            summary_df=ph_df,
                                            stats_df=stats_df,
                                            plot_methods=plot_methods,
                                            strat_cols=[str(c) for c in p2_strat_cols],
                                            strata=strata,
                                            overlay_active=overlay_active,
                                            overlay_encoding=str(p2_overlay_encoding),
                                            show_stats=bool(p2_show_stats),
                                            stratum_color_map=p2_stratum_color_map,
                                            method_color_map=method_color_map,
                                            method_label_map=method_label_map,
                                            method_color_defaults=method_color_defaults,
                                            min_distance_um=float(min_distance_um),
                                            max_distance_um=float(max_distance_um),
                                            title_for_stratum=lambda stratum: (
                                                f"{pheno} | overlaid strata" if overlay_active else f"{pheno} | {stratum}"
                                            ),
                                            y_label="Density (cells/mm²)",
                                            key_prefix="tab8_density_p2",
                                            stats_filter={"phenotype": str(pheno)},
                                        )
                                        fig_path = out_dir / f"panel2_{str(pheno).replace(' ', '_')}_zonal_density.png"
                                        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                                        svg_path = fig_path.with_suffix(".svg")
                                        fig.savefig(svg_path, bbox_inches="tight")
                                        plt.close(fig)
                                        plot_paths[str(pheno)] = str(fig_path)
                                        output_files.extend([fig_path, svg_path])

                                    bundle_dir = create_publication_bundle(out_dir, "density_panel2_zonal")
                                    copied_files = copy_outputs_to_bundle(output_files, bundle_dir)
                                    write_publication_metadata(
                                        bundle_dir / "analysis_metadata.json",
                                        {
                                            "panel": "Density Panel 2 - phenotype density by boundary distance",
                                            "statistics": {
                                                "method_comparison_omnibus": "Friedman test within each distance zone",
                                                "method_comparison_post_hoc": "two-sided paired Wilcoxon signed-rank tests",
                                                "method_comparison_multiplicity": (
                                                    "Holm correction across the full method-pair-by-zone family within each phenotype/stratum"
                                                ),
                                                "stratified_comparison": str(p2_stats_method) if p2_strat_cols else "not run",
                                                "stratified_multiplicity": "Holm across zones within phenotype, method and stratum pair",
                                                "curve_summary": "mean +/- SEM across connected tissues",
                                                "curve_annotations": "method tests exported to CSV; not drawn to preserve figure readability",
                                            },
                                            "analysis": {
                                                "methods": [str(method) for method in p2_pheno_cols],
                                                "phenotypes": [str(value) for value in p2_selected_phenotypes],
                                                "connected_tissue_columns": connected_cols,
                                                "distance_source": (
                                                    str(p2_distance_col)
                                                    if p2_distance_col != "None (compute from annotation)"
                                                    else {
                                                        "annotation_column": str(p2_annotation_col),
                                                        "reference_values": [str(value) for value in p2_reference_values],
                                                    }
                                                ),
                                                "distance_min_um": float(min_distance_um),
                                                "distance_max_um": float(max_distance_um),
                                                "zone_width_um": float(zone_width_um),
                                                "minimum_cells_per_zone": int(min_cells_per_zone),
                                                "minimum_connected_tissues_per_curve": int(min_rois_per_group),
                                                "connected_tissues_per_point_min": int(summary_df["n_connected_tissues"].min()),
                                                "connected_tissues_per_point_max": int(summary_df["n_connected_tissues"].max()),
                                                "pixel_size_um": float(pixel_size_um),
                                                "virtual_area_method": "buffer_union",
                                                "nearest_neighbor_buffer_percentile": percentile_value,
                                                "minimum_hole_size_px2": float(min_hole_size),
                                                "prefilter": {
                                                    "column": str(p2_prefilter_col),
                                                    "values": [str(value) for value in p2_prefilter_values],
                                                },
                                                "filters": p2_filter_values,
                                                "stratification_columns": p2_strat_cols,
                                            },
                                            "display": {"png_dpi": 300, "figures_unchanged_by_statistical_export": True},
                                            "files": copied_files,
                                        },
                                    )

                                    st.session_state[_DENSITY_PANEL2_RESULTS_KEY] = {
                                        "summary_df": summary_df,
                                        "detailed_csv": str(zone_csv),
                                        "summary_csv": str(summary_csv),
                                        "rows_detailed": int(len(zone_df)),
                                        "rows_summary": int(len(summary_df)),
                                        "phenotypes": [str(p) for p in p2_selected_phenotypes],
                                        "detailed_df": zone_df,
                                        "plot_paths": plot_paths,
                                        "stats_df": stats_df,
                                        "method_stats_df": method_stats_df,
                                        "runtime_seconds": float(perf_counter() - panel2_started),
                                        "connected_cols": connected_cols,
                                        "method_cols": [str(col) for col in p2_pheno_cols],
                                        "strat_cols": [str(c) for c in p2_strat_cols],
                                        "tumor_bed_col": str(p2_prefilter_col),
                                        "tumor_bed_value": ",".join([str(v) for v in p2_prefilter_values]),
                                        "panel_filter_values": {k: [str(v) for v in vals] for k, vals in p2_filter_values.items() if vals},
                                        "overlay_encoding": str(p2_overlay_encoding),
                                        "publication_bundle": str(bundle_dir),
                                    }
                                    st.success("Panel 2 calculation complete.")

    panel2_state = st.session_state.get(_DENSITY_PANEL2_RESULTS_KEY)
    if panel2_state:
        _render_panel2_results(panel2_state)


def _render_panel3_results(panel_state: dict):
    st.info("Showing the most recent Panel 3 run.")
    st.caption(panel_state.get("context", ""))
    for entry in panel_state.get("figures", []):
        title = str(entry.get("title", ""))
        path = str(entry.get("path", ""))
        if path and Path(path).exists():
            if title:
                st.markdown(f"#### {title}")
            st.image(path, width="stretch")
    saved = panel_state.get("saved_files", [])
    if saved:
        st.write({"Panel3_output_dir": panel_state.get("output_dir"), "Panel3_files": saved})


@st.fragment
def render_panel3_fragment(
    run_df: pd.DataFrame,
    phenotype_cols: list[str],
    shared: dict[str, Any],
    batch_col: str,
    roi_col: str,
    pixel_size_um: float,
    output_dir: str,
    project_root: Path,
):
    st.caption(
        "Tissue maps in the published Figure 2E style: annotation layers from the merged GeoJSON files, "
        "a white outline around the selected tissue component and a Gaussian-smoothed density cloud "
        "for one phenotype, drawn on one shared color scale across phenotyping methods."
    )

    from spacemap_gui.ui_utils import directory_input_with_browse

    method_label_map: dict[str, str] = dict(shared.get("labels", {}))
    shared_methods = [str(m) for m in shared.get("methods", []) if str(m) in phenotype_cols]

    tab_data, tab_layers, tab_cloud, tab_layout = st.tabs(
        [
            "1 · Tissue & phenotype",
            "2 · Annotation layers & colors",
            "3 · Density cloud",
            "4 · Size, text & output",
        ]
    )

    annotations: dict[str, Any] = {}
    with tab_data:
        st.markdown("**Tissue annotations (GeoJSON)**")
        annot_dir = directory_input_with_browse(
            label="ROI annotation directory",
            text_key="tab8_density_p3_annot_dir",
            default_value=str(
                st.session_state.get(
                    "tab8_density_p3_annot_dir",
                    project_root / "outputs" / "processed_annotations",
                )
            ),
            browser_key_prefix="tab8_density_p3_annot_dir",
            help_text="Folder holding one merged GeoJSON per ROI image, e.g. P11_roi_1_merged.geojson.",
        )

        a1, a2 = st.columns(2)
        file_pattern = a1.text_input(
            "File name pattern",
            value=str(st.session_state.get("tab8_density_p3_file_pattern", clouds.DEFAULT_FILE_PATTERN)),
            key="tab8_density_p3_file_pattern",
            help="Use {batch} as the placeholder for the batch ID; used to preselect the file for the chosen batch.",
        )

        batch_values = ["ALL"]
        if batch_col and batch_col in run_df.columns:
            batch_values = sorted(run_df[batch_col].dropna().astype(str).unique().tolist()) or ["ALL"]
        batch_choice = _sync_selectbox_state("tab8_density_p3_batch", batch_values, batch_values[0])
        p3_batch = a2.selectbox(
            "Batch ID (ROI image)",
            options=batch_values,
            index=batch_values.index(batch_choice) if batch_choice in batch_values else 0,
            key="tab8_density_p3_batch",
            help="All cells of this batch are used; pick a single tissue ROI below to zoom into one component.",
        )

        geojson_files: list[str] = []
        annot_dir_path = Path(str(annot_dir)) if annot_dir else None
        if annot_dir_path is not None and annot_dir_path.is_dir():
            geojson_files = sorted(p.name for p in annot_dir_path.iterdir() if p.suffix.lower() == ".geojson")
        if not geojson_files:
            st.warning("No .geojson files found in the annotation directory.")
        expected_name = clouds.annotation_path("", str(p3_batch), str(file_pattern)).name
        file_options = geojson_files or [""]
        file_default = expected_name if expected_name in file_options else file_options[0]
        # Follow the batch selection: switching batch re-points the file to the matching GeoJSON.
        last_batch_key = "_tab8_density_p3_last_batch"
        if st.session_state.get(last_batch_key) != str(p3_batch):
            st.session_state[last_batch_key] = str(p3_batch)
            if expected_name in file_options:
                st.session_state["tab8_density_p3_geojson_file"] = expected_name
        file_choice = _sync_selectbox_state("tab8_density_p3_geojson_file", file_options, file_default)
        p3_geojson_file = st.selectbox(
            "GeoJSON file for this batch",
            options=file_options,
            index=file_options.index(file_choice) if file_choice in file_options else 0,
            key="tab8_density_p3_geojson_file",
            help=f"Auto-matched from the pattern above ({expected_name}); override here if the file is named differently.",
        )

        # Layers differ between ROI images, so a new file starts from that file's defaults
        # instead of silently keeping a selection the new file cannot offer.
        last_file_key = "_tab8_density_p3_last_file"
        if last_file_key in st.session_state and st.session_state[last_file_key] != str(p3_geojson_file):
            st.session_state.pop("tab8_density_p3_layers", None)
        st.session_state[last_file_key] = str(p3_geojson_file)

        if annot_dir_path is not None and p3_geojson_file:
            geojson_path = annot_dir_path / str(p3_geojson_file)
            if geojson_path.exists():
                try:
                    annotations = clouds.load_annotation_geometries(str(geojson_path), geojson_path.stat().st_mtime)
                except Exception as error:  # pragma: no cover - depends on user files
                    st.error(f"Could not read {geojson_path.name}: {error}")
        layer_names = sorted(annotations)
        if layer_names:
            st.caption(f"Layers in {p3_geojson_file}: {', '.join(layer_names)}")

        st.markdown("**Tissue component**")
        t1, t2 = st.columns(2)
        tissue_options = layer_names or [clouds.DEFAULT_TISSUE_LAYER]
        tissue_default = clouds.DEFAULT_TISSUE_LAYER if clouds.DEFAULT_TISSUE_LAYER in tissue_options else tissue_options[0]
        tissue_choice = _sync_selectbox_state("tab8_density_p3_tissue_layer", tissue_options, tissue_default)
        p3_tissue_layer = t1.selectbox(
            "Tissue outline layer",
            options=tissue_options,
            index=tissue_options.index(tissue_choice) if tissue_choice in tissue_options else 0,
            key="tab8_density_p3_tissue_layer",
            help="Layer whose polygons define the tissue outline and the crop of the map.",
        )

        roi_options = ["(whole tissue)"]
        if roi_col and roi_col in run_df.columns:
            batch_mask = (
                run_df[batch_col].astype(str) == str(p3_batch)
                if batch_col and batch_col in run_df.columns
                else np.ones(len(run_df), dtype=bool)
            )
            roi_options += sorted(run_df.loc[batch_mask, roi_col].dropna().astype(str).unique().tolist())
        roi_choice = _sync_selectbox_state("tab8_density_p3_roi", roi_options, roi_options[0])
        p3_roi = t2.selectbox(
            "Tissue ROI",
            options=roi_options,
            index=roi_options.index(roi_choice) if roi_choice in roi_options else 0,
            key="tab8_density_p3_roi",
            help="Pick one ROI to draw only the tissue component holding those cells, like the reference figure.",
        )

        st.markdown("**Phenotype to map**")
        p3_methods = list(shared_methods)
        if p3_methods:
            st.caption(
                "One map per phenotyping method, taken from Shared Figure Settings: "
                + ", ".join(f"{method_label_map.get(m, m)} ({m})" for m in p3_methods)
            )
        else:
            st.warning("Select at least one phenotyping method column in Shared Figure Settings.")
        p3_pool = _phenotype_pool(run_df, p3_methods)
        p3_phenotypes = st.multiselect(
            "Phenotype(s) to map (one figure per phenotype)",
            options=p3_pool,
            default=[p for p in ["CD8"] if p in p3_pool] or p3_pool[: min(1, len(p3_pool))],
            key="tab8_density_p3_phenotypes",
            help="For example CD8, CD4 or DC.",
        )

    with tab_layers:
        available_layers = [name for name in sorted(annotations) if name != str(st.session_state.get("tab8_density_p3_tissue_layer", ""))]
        default_layers = [name for name in clouds.DEFAULT_LAYERS if name in available_layers]
        p3_layers = st.multiselect(
            "Annotation layers to draw (in drawing/legend order)",
            options=available_layers,
            default=default_layers or available_layers[: min(3, len(available_layers))],
            key="tab8_density_p3_layers",
            help="The reference figure draws tumor, epithelia and stroma and omits endothelial.",
        )
        layer_color_map: dict[str, str] = {}
        if p3_layers:
            color_cols = st.columns(min(4, len(p3_layers)))
            for idx, name in enumerate(p3_layers):
                with color_cols[idx % len(color_cols)]:
                    layer_color_map[name] = st.color_picker(
                        f"{str(name).capitalize()} color",
                        value=clouds.default_layer_color(name, idx),
                        key=f"tab8_density_p3_layer_color_{name}",
                    )

        c1, c2, c3 = st.columns(3)
        p3_layer_alpha = c1.slider(
            "Annotation layer opacity",
            min_value=0.05,
            max_value=1.0,
            value=0.46,
            step=0.01,
            key="tab8_density_p3_layer_alpha",
        )
        p3_background = c2.color_picker(
            "Background color",
            value="#000000",
            key="tab8_density_p3_background",
        )
        p3_text_color = c3.color_picker(
            "Text color",
            value="#FFFFFF",
            key="tab8_density_p3_text_color",
        )

        b1, b2, b3 = st.columns(3)
        p3_show_boundary = b1.checkbox(
            "Outline the tissue component",
            value=True,
            key="tab8_density_p3_show_boundary",
        )
        p3_boundary_color = b2.color_picker(
            "Outline color",
            value="#FFFFFF",
            key="tab8_density_p3_boundary_color",
            disabled=not p3_show_boundary,
        )
        p3_boundary_width = b3.slider(
            "Outline width (pt)",
            min_value=0.1,
            max_value=3.0,
            value=0.7,
            step=0.1,
            key="tab8_density_p3_boundary_width",
            disabled=not p3_show_boundary,
        )

    with tab_cloud:
        d1, d2, d3 = st.columns(3)
        p3_show_cloud = d1.checkbox(
            "Show density cloud",
            value=True,
            key="tab8_density_p3_show_cloud",
        )
        p3_grid = d2.number_input(
            "Density grid (bins per axis)",
            min_value=60,
            max_value=1200,
            value=360,
            step=20,
            key="tab8_density_p3_grid",
            help="Reference value: 360.",
        )
        p3_sigma = d3.number_input(
            "Smoothing bandwidth (µm)",
            min_value=5.0,
            max_value=400.0,
            value=55.0,
            step=5.0,
            key="tab8_density_p3_sigma",
            help="Gaussian sigma in microns. Reference value: 55.",
        )

        st.markdown("**Cloud color ramp** (transparent → low → mid → high)")
        s1, s2, s3 = st.columns(3)
        p3_stop_low = s1.color_picker("Low", value=clouds.CLOUD_STOPS[0], key="tab8_density_p3_stop_low")
        p3_stop_mid = s2.color_picker("Mid", value=clouds.CLOUD_STOPS[1], key="tab8_density_p3_stop_mid")
        p3_stop_high = s3.color_picker("High", value=clouds.CLOUD_STOPS[2], key="tab8_density_p3_stop_high")

        st.markdown("**Cell centroids**")
        p1c, p2c, p3c = st.columns(3)
        p3_show_points = p1c.checkbox(
            "Show phenotype centroids",
            value=True,
            key="tab8_density_p3_show_points",
        )
        p3_point_color = p1c.color_picker(
            "Centroid color",
            value=clouds.CLOUD_HIGHLIGHT,
            key="tab8_density_p3_point_color",
            disabled=not p3_show_points,
        )
        p3_point_size = p2c.slider(
            "Centroid size",
            min_value=0.5,
            max_value=20.0,
            value=2.5,
            step=0.5,
            key="tab8_density_p3_point_size",
            disabled=not p3_show_points,
        )
        p3_point_alpha = p3c.slider(
            "Centroid opacity",
            min_value=0.02,
            max_value=1.0,
            value=0.22,
            step=0.02,
            key="tab8_density_p3_point_alpha",
            disabled=not p3_show_points,
        )

        cb1, cb2 = st.columns(2)
        p3_show_colorbar = cb1.checkbox(
            "Show color bar",
            value=True,
            key="tab8_density_p3_show_colorbar",
        )
        p3_colorbar_label = cb2.text_input(
            "Color bar label",
            value="Smoothed {phenotype} density (shared scale)",
            key="tab8_density_p3_colorbar_label",
            help="{phenotype} is replaced by the mapped phenotype.",
        )

    with tab_layout:
        l1, l2, l3 = st.columns(3)
        p3_panel_width = l1.number_input(
            "Width per map (in)",
            min_value=1.5,
            max_value=12.0,
            value=3.5,
            step=0.5,
            key="tab8_density_p3_panel_width",
        )
        p3_panel_height = l2.number_input(
            "Figure height (in)",
            min_value=2.0,
            max_value=16.0,
            value=5.25,
            step=0.25,
            key="tab8_density_p3_panel_height",
        )
        p3_dpi = l3.number_input(
            "DPI",
            min_value=72,
            max_value=900,
            value=300,
            step=50,
            key="tab8_density_p3_dpi",
        )

        f1, f2, f3, f4 = st.columns(4)
        p3_title_fontsize = f1.slider("Map title (pt)", min_value=6, max_value=24, value=11, step=1, key="tab8_density_p3_title_fontsize")
        p3_suptitle_fontsize = f2.slider("Figure title (pt)", min_value=8, max_value=28, value=13, step=1, key="tab8_density_p3_suptitle_fontsize")
        p3_legend_fontsize = f3.slider("Legend (pt)", min_value=5, max_value=20, value=9, step=1, key="tab8_density_p3_legend_fontsize")
        p3_colorbar_fontsize = f4.slider("Color bar (pt)", min_value=5, max_value=20, value=9, step=1, key="tab8_density_p3_colorbar_fontsize")

        p3_suptitle = st.text_input(
            "Figure title",
            value="Method-dependent {phenotype} landscape\\n{batch}, {roi}",
            key="tab8_density_p3_suptitle",
            help="Placeholders: {phenotype}, {batch}, {roi}. Use \\n for a line break; leave empty to hide the title.",
        )
        o1, o2, o3 = st.columns(3)
        p3_show_legend = o1.checkbox("Show legend", value=True, key="tab8_density_p3_show_legend")
        p3_single_figures = o2.checkbox(
            "Also save one figure per method",
            value=False,
            key="tab8_density_p3_single_figures",
            help="Matches the reference script, which writes a combined figure plus one map per method.",
        )
        p3_save_svg = o3.checkbox("Also save SVG", value=True, key="tab8_density_p3_save_svg")
        p3_output_dir = st.text_input(
            "Output directory",
            value=str(Path(output_dir) / "panel3_density_clouds"),
            key="tab8_density_p3_output_dir",
            help="Defaults to a panel3_density_clouds folder inside the shared output directory.",
        )

    col_p3_btns = st.columns([1, 1])
    run_panel3 = col_p3_btns[0].button("Render Panel 3", type="primary", key="tab8_density_p3_run", width="stretch")
    save_panel3 = col_p3_btns[1].button("Save Parameters", type="secondary", key="tab8_density_p3_save", width="stretch")

    if save_panel3:
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab8_analysis")
        st.success("Project settings saved to configuration file.")

    if run_panel3:
        if not annotations:
            st.error("Panel 3: Select an annotation directory containing the GeoJSON file for this batch.")
        elif not p3_methods:
            st.error("Panel 3: Select at least one phenotyping method column.")
        elif not p3_phenotypes:
            st.error("Panel 3: Select at least one phenotype to map.")
        elif p3_tissue_layer not in annotations:
            st.error(f"Panel 3: Layer '{p3_tissue_layer}' is not present in {p3_geojson_file}.")
        else:
            with st.status("Rendering Panel 3...", expanded=True) as status:
                log_density(f"Panel 3 started for batch={p3_batch}, roi={p3_roi}")
                cells = run_df
                if batch_col and batch_col in run_df.columns:
                    cells = cells[cells[batch_col].astype(str) == str(p3_batch)]
                roi_label = "all tissue ROIs"
                if p3_roi != "(whole tissue)" and roi_col and roi_col in cells.columns:
                    cells = cells[cells[roi_col].astype(str) == str(p3_roi)]
                    roi_label = f"{roi_col}={p3_roi}"
                cells = cells.dropna(subset=REQUIRED_COORDS)
                status.write(f"{len(cells):,} cells in {p3_batch} | {roi_label}.")

                if cells.empty:
                    status.update(label="Panel 3 finished with no cells", state="error")
                    st.warning("Panel 3: No cells for the selected batch/ROI.")
                else:
                    tissue_geometry = annotations[str(p3_tissue_layer)]
                    if p3_roi != "(whole tissue)":
                        component, scores = clouds.choose_tissue_component(
                            tissue_geometry, cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                        )
                        if component is None:
                            component = clouds.safe_geometry(tissue_geometry)
                            status.write("Could not match the ROI cells to a single polygon; using the whole tissue layer.")
                        else:
                            status.write(f"Matched tissue component with {max(scores):,} sampled cells inside.")
                    else:
                        component = clouds.safe_geometry(tissue_geometry)

                    layers = {
                        name: clouds.safe_geometry(component.intersection(annotations.get(name, clouds.GeometryCollection())))
                        for name in p3_layers
                    }

                    out_dir = Path(p3_output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    figures: list[dict[str, str]] = []
                    saved_files: list[str] = []

                    for phenotype in p3_phenotypes:
                        style = clouds.CloudStyle(
                            layer_colors={name: layer_color_map.get(name, clouds.default_layer_color(name, idx)) for idx, name in enumerate(p3_layers)},
                            layer_alpha=float(p3_layer_alpha),
                            background=str(p3_background),
                            text_color=str(p3_text_color),
                            show_boundary=bool(p3_show_boundary),
                            boundary_color=str(p3_boundary_color),
                            boundary_width=float(p3_boundary_width),
                            show_cloud=bool(p3_show_cloud),
                            cloud_stops=(str(p3_stop_low), str(p3_stop_mid), str(p3_stop_high)),
                            grid_size=int(p3_grid),
                            sigma_um=float(p3_sigma),
                            show_points=bool(p3_show_points),
                            point_color=str(p3_point_color),
                            point_size=float(p3_point_size),
                            point_alpha=float(p3_point_alpha),
                            panel_width=float(p3_panel_width),
                            panel_height=float(p3_panel_height),
                            title_fontsize=float(p3_title_fontsize),
                            suptitle_fontsize=float(p3_suptitle_fontsize),
                            legend_fontsize=float(p3_legend_fontsize),
                            colorbar_fontsize=float(p3_colorbar_fontsize),
                            colorbar_label=str(p3_colorbar_label).replace("{phenotype}", str(phenotype)),
                            show_colorbar=bool(p3_show_colorbar),
                            show_legend=bool(p3_show_legend),
                            cloud_label=f"{phenotype}-density cloud",
                            point_label=f"{phenotype} centroids",
                            count_noun=f"{phenotype} cells",
                            dpi=int(p3_dpi),
                        )

                        panels: list[dict[str, Any]] = []
                        for method in p3_methods:
                            if method not in cells.columns:
                                continue
                            selected = cells[cells[method].astype(str) == str(phenotype)]
                            panels.append(
                                {
                                    "label": method_label_map.get(str(method), str(method)),
                                    "points": selected[["x_centroid", "y_centroid"]].to_numpy(dtype=float),
                                }
                            )
                        if not panels:
                            continue

                        status.write(f"Computing {phenotype} density fields for {len(panels)} method(s).")
                        field_data = clouds.compute_cloud_fields(panels, component, style, float(pixel_size_um))
                        suptitle = (
                            str(p3_suptitle)
                            .replace("\\n", "\n")
                            .replace("{phenotype}", str(phenotype))
                            .replace("{batch}", str(p3_batch))
                            .replace("{roi}", roi_label)
                        )
                        fig = clouds.render_cloud_figure(panels, component, layers, style, field_data, suptitle=suptitle)
                        if fig is None:
                            continue
                        safe_name = str(phenotype).replace(" ", "_").replace("/", "-")
                        fig_path = out_dir / f"panel3_{safe_name}_density_clouds.png"
                        saved_files.extend(clouds.save_cloud_figure(fig, fig_path, style, also_svg=bool(p3_save_svg)))
                        st.markdown(f"#### {phenotype} density clouds")
                        st.pyplot(fig)
                        plt.close(fig)
                        figures.append({"title": f"{phenotype} density clouds", "path": str(fig_path)})
                        log_density(f"Panel 3 wrote {fig_path}")

                        if p3_single_figures:
                            for panel in panels:
                                single_style = replace(style, panel_width=float(p3_panel_width) * 1.7)
                                single_fig = clouds.render_cloud_figure(
                                    [panel], component, layers, single_style, field_data, suptitle=suptitle
                                )
                                if single_fig is None:
                                    continue
                                label_safe = str(panel["label"]).replace(" ", "_").replace("/", "-")
                                single_path = out_dir / f"panel3_{safe_name}_{label_safe}.png"
                                saved_files.extend(
                                    clouds.save_cloud_figure(single_fig, single_path, single_style, also_svg=bool(p3_save_svg))
                                )
                                plt.close(single_fig)

                    if not figures:
                        status.update(label="Panel 3 finished with no figures", state="error")
                        st.warning("Panel 3: Nothing to draw for the selected methods/phenotypes.")
                    else:
                        context = (
                            f"{p3_batch} | {roi_label} | layers: {', '.join(p3_layers) if p3_layers else 'none'} | "
                            f"grid={int(p3_grid)}, sigma={float(p3_sigma):g} µm, pixel={float(pixel_size_um):g} µm"
                        )
                        st.session_state[_DENSITY_PANEL3_RESULTS_KEY] = {
                            "figures": figures,
                            "saved_files": saved_files,
                            "output_dir": str(out_dir),
                            "context": context,
                        }
                        status.update(label="Panel 3 complete", state="complete")
                        st.success(f"Saved {len(saved_files)} file(s) to {out_dir}")

    elif st.session_state.get(_DENSITY_PANEL3_RESULTS_KEY):
        _render_panel3_results(st.session_state[_DENSITY_PANEL3_RESULTS_KEY])


def _pretty_method_name(column: str) -> str:
    """Short figure label for a phenotype column (same names Tab 9 uses)."""

    exact = {
        "phenotype_leiden": "Leiden",
        "phenotype_som": "SOM",
        "phenotype_vortex": "Vortex",
        "phenotype_threshold": "RESOLVE",
        "phenotype_resolve": "RESOLVE",
        "phenotype_scimap": "SCIMAP",
        "phenotype_ml_mr": "MR-ML",
        "phenotype_ml_consensus": "Consensus-ML",
    }
    value = str(column)
    if value in exact:
        return exact[value]
    for prefix in ("prediction_", "phenotype_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.replace("_", " ").strip()
    return {"som": "SOM", "ap": "AP", "bd": "BD"}.get(value.lower(), value.title())


def render_shared_figure_settings(
    phenotype_cols: list[str],
    default_methods: list[str],
    *,
    methods_key: str,
    label_key_fmt: str,
    color_key_fmt: str,
    legacy_methods_key: str | None = None,
    seed_label_key_fmt: str | None = None,
    seed_color_key_fmt: str | None = None,
    methods_help: str = "",
    caption: str = "",
) -> dict[str, Any]:
    """Render the methods / display names / colors block shared by a tab's panels.

    Key names are passed in so each subtab keeps its own saved settings while the
    controls and defaults stay identical. ``seed_*_key_fmt`` points at another
    tab's keys, so a subtab starts from the colors and names already chosen there.
    """

    legacy_methods = [
        str(m) for m in st.session_state.get(legacy_methods_key, []) if str(m) in phenotype_cols
    ] if legacy_methods_key else []
    fallback = legacy_methods or default_methods or phenotype_cols[: min(4, len(phenotype_cols))]
    methods = st.multiselect(
        "Phenotyping method columns (used by every panel)",
        options=phenotype_cols,
        default=fallback,
        key=methods_key,
        help=methods_help or "Every panel in this subtab uses these columns.",
    )

    labels: dict[str, str] = {}
    colors: dict[str, str] = {}
    if methods:
        st.caption(
            caption
            or "Display name and color for each method, used in every figure of this tab "
            "(figure axes, curve legends and cloud panel titles)."
        )
        for idx, method in enumerate(methods):
            default_label = _pretty_method_name(method)
            default_color = default_method_color(method, idx)
            if seed_label_key_fmt:
                default_label = str(st.session_state.get(seed_label_key_fmt.format(method=method), default_label))
            if seed_color_key_fmt:
                default_color = str(st.session_state.get(seed_color_key_fmt.format(method=method), default_color))
            name_col, color_col = st.columns([3, 1])
            labels[method] = name_col.text_input(
                f"Display name: {method}",
                value=default_label,
                key=label_key_fmt.format(method=method),
            )
            colors[method] = color_col.color_picker(
                "Color",
                value=default_color,
                key=color_key_fmt.format(method=method),
            )
    else:
        st.warning("Select at least one phenotyping method column to enable the panels.")

    return {
        "methods": methods,
        "labels": labels,
        "colors": colors,
        "color_defaults": METHOD_COLOR_DEFAULTS,
    }


def _render_shared_figure_settings(phenotype_cols: list[str], default_methods: list[str]) -> dict[str, Any]:
    """Density-subtab wrapper; keeps the historical widget keys."""

    return render_shared_figure_settings(
        phenotype_cols,
        default_methods,
        methods_key="tab8_density_shared_methods",
        label_key_fmt="tab8_density_p1_label_{method}",
        color_key_fmt="tab8_density_method_color_{method}",
        legacy_methods_key="tab8_density_p1_methods",
        methods_help=(
            "Panel 1 compares these columns, Panel 2 profiles them by default and "
            "Panel 3 draws one map per column."
        ),
    )


def render_density_analysis(project_root: Path, df: pd.DataFrame):
    st.subheader("Density Analysis")
    st.caption(
        "Panel 1: density per phenotyping method · Panel 2: density by distance from an annotation · "
        "Panel 3: Figure-2E style density clouds on the tissue. "
        "Shared settings at the top apply to all three panels."
    )

    if len(df.columns) < 2:
        st.error("Need at least two columns to select X/Y coordinates.")
        return

    all_columns = list(df.columns)
    coord_options = _column_options(all_columns)
    default_x_col = _preferred_or_blank(coord_options, "x_centroid")
    default_y_col = _preferred_or_blank(coord_options, "y_centroid")

    work_df = df.copy()
    if work_df.empty:
        st.warning("No rows left after global filtering.")
        return

    default_core_state = {
        "x_coord_col": default_x_col,
        "y_coord_col": default_y_col,
        "batch_col": "batch_id" if "batch_id" in work_df.columns else "",
        "roi_col": "tissue_roi" if "tissue_roi" in work_df.columns else "",
        "min_cells_group": 20,
        "pixel_size_um": 0.5083,
        "use_median": False,
        "percentile": 99.0,
        "min_hole_size": 10000.0,
        "output_dir": str(project_root / "outputs" / "spatial_analysis" / "density"),
        "selected_core_filters": [],
        "core_filter_values": {},
    }
    if _DENSITY_CORE_STATE_KEY not in st.session_state:
        st.session_state[_DENSITY_CORE_STATE_KEY] = default_core_state.copy()

    with st.expander("Shared Configuration · Connected Virtual Tissue Units", expanded=True):
        st.caption(
            "Applies to every panel. Settings take effect when you click Apply Core Configuration; "
            "each panel then recomputes only when you run that panel."
        )
        with st.form("tab8_density_core_form"):
            xy1, xy2 = st.columns(2)
            x_coord_value = _sync_selectbox_state("tab8_density_x_coord_col", coord_options, default_x_col)
            x_coord_col = xy1.selectbox(
                "X Coordinate Column",
                options=coord_options,
                index=coord_options.index(x_coord_value) if x_coord_value in coord_options else 0,
                key="tab8_density_x_coord_col",
                format_func=lambda x: x if x else "(not selected)",
            )
            y_coord_value = _sync_selectbox_state("tab8_density_y_coord_col", coord_options, default_y_col)
            y_coord_col = xy2.selectbox(
                "Y Coordinate Column",
                options=coord_options,
                index=coord_options.index(y_coord_value) if y_coord_value in coord_options else 0,
                key="tab8_density_y_coord_col",
                format_func=lambda x: x if x else "(not selected)",
            )

            cols_for_keys = [c for c in work_df.columns if c not in {x_coord_col, y_coord_col, ""}]
            batch_options = [""] + cols_for_keys
            roi_options = [""] + cols_for_keys

            c1, c2, c3 = st.columns(3)
            batch_value = _sync_selectbox_state("tab8_density_batch_col", batch_options, "batch_id" if "batch_id" in batch_options else "")
            batch_col = c1.selectbox(
                "Batch ID Column",
                options=batch_options,
                index=batch_options.index(batch_value) if batch_value in batch_options else _default_index(batch_options, "batch_id", fallback=0),
                key="tab8_density_batch_col",
                format_func=lambda x: x if x else "(not selected)",
                help="Select the batch/patient-level column. Default is batch_id if available.",
            )
            roi_value = _sync_selectbox_state("tab8_density_roi_col", roi_options, "tissue_roi" if "tissue_roi" in roi_options else "")
            roi_col = c2.selectbox(
                "ROI Column",
                options=roi_options,
                index=roi_options.index(roi_value) if roi_value in roi_options else _default_index(roi_options, "tissue_roi", fallback=0),
                key="tab8_density_roi_col",
                format_func=lambda x: x if x else "(not selected)",
                help="Select ROI column to separate disconnected ROIs inside each batch.",
            )
            min_cells_group = c3.number_input(
                "Min Cells / Connected Tissue",
                min_value=5,
                max_value=50000,
                value=20,
                step=5,
                key="tab8_density_min_cells_group",
            )

            d1, d2, d3 = st.columns(3)
            pixel_size_um = d1.number_input(
                "Pixel Size (µm)",
                min_value=0.01,
                value=0.5083,
                step=0.0001,
                format="%.4f",
                key="tab8_density_pixel_um",
            )
            use_median = d2.checkbox("Use median buffer distance", value=False, key="tab8_density_use_median")
            percentile = d3.slider(
                "Buffer Percentile",
                min_value=90.0,
                max_value=99.9,
                value=99.0,
                step=0.1,
                key="tab8_density_percentile",
            )

            e1, e2 = st.columns(2)
            min_hole_size = e1.number_input(
                "Min Hole Size (pixels²)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                key="tab8_density_min_hole",
            )
            output_dir = e2.text_input(
                "Output Directory",
                value=str(project_root / "outputs" / "spatial_analysis" / "density"),
                key="tab8_density_output_dir",
            )

            filter_candidates = [c for c in work_df.columns if c not in {x_coord_col, y_coord_col}]
            selected_core_filters = st.multiselect(
                "Optional Global Filters (Tab-level)",
                options=filter_candidates,
                default=[],
                key="tab8_density_core_filter_cols",
                help="Apply additional filters before running Panel 1/2. Leave empty to keep all rows.",
            )
            core_filter_values = _render_filter_widgets(work_df, selected_core_filters, key_prefix="tab8_density_core")
            col_btns = st.columns([1, 1])
            apply_core = col_btns[0].form_submit_button("Apply Core Configuration", type="primary", width="stretch")
            save_core = col_btns[1].form_submit_button("Save Parameters", type="secondary", width="stretch", help="Apply core settings and save project configurations.")

        if apply_core or save_core:
            if not x_coord_col or not y_coord_col:
                st.error("Please select both X and Y coordinate columns.")
                log_density("Core apply rejected because coordinate columns were not fully selected")
            elif x_coord_col == y_coord_col:
                st.error("X Coordinate Column and Y Coordinate Column must be different.")
                log_density("Core apply rejected because X and Y columns were identical")
            else:
                st.session_state[_DENSITY_CORE_STATE_KEY] = {
                    "x_coord_col": x_coord_col,
                    "y_coord_col": y_coord_col,
                    "batch_col": batch_col,
                    "roi_col": roi_col,
                    "min_cells_group": int(min_cells_group),
                    "pixel_size_um": float(pixel_size_um),
                    "use_median": bool(use_median),
                    "percentile": float(percentile),
                    "min_hole_size": float(min_hole_size),
                    "output_dir": str(output_dir),
                    "selected_core_filters": list(selected_core_filters),
                    "core_filter_values": {k: [str(v) for v in vals] for k, vals in core_filter_values.items() if vals},
                }
                st.session_state.pop(_DENSITY_PANEL1_RESULTS_KEY, None)
                st.session_state.pop(_DENSITY_PANEL2_RESULTS_KEY, None)
                log_density(
                    f"Applied core config with x={x_coord_col}, y={y_coord_col}, batch={batch_col or 'ALL'}, roi={roi_col or 'ALL'}"
                )
                if save_core:
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab8_analysis")
                    st.success("Core configuration applied and project settings saved.")
                    st.rerun()
                else:
                    st.success("Core configuration applied. Preview and panel runs now use the updated settings.")

    applied_core = dict(st.session_state.get(_DENSITY_CORE_STATE_KEY, default_core_state))
    sanitized_core = dict(applied_core)
    sanitized_core["x_coord_col"] = sanitized_core.get("x_coord_col") if sanitized_core.get("x_coord_col") in all_columns else default_x_col
    sanitized_core["y_coord_col"] = sanitized_core.get("y_coord_col") if sanitized_core.get("y_coord_col") in all_columns else default_y_col
    sanitized_core["batch_col"] = sanitized_core.get("batch_col") if sanitized_core.get("batch_col") in work_df.columns else ""
    sanitized_core["roi_col"] = sanitized_core.get("roi_col") if sanitized_core.get("roi_col") in work_df.columns else ""
    if sanitized_core != applied_core:
        st.session_state[_DENSITY_CORE_STATE_KEY] = sanitized_core
    applied_core = sanitized_core

    st.info(
        "Applied core config: "
        f"x={applied_core['x_coord_col'] or '(not selected)'}, y={applied_core['y_coord_col'] or '(not selected)'}, "
        f"batch={applied_core['batch_col'] or 'ALL'}, roi={applied_core['roi_col'] or 'ALL'}"
    )

    if not applied_core["x_coord_col"] or not applied_core["y_coord_col"]:
        st.warning("Select valid X and Y coordinate columns in Core Configuration, then click Apply Core Configuration.")
        return

    context = _prepare_density_context(
        df=df,
        x_coord_col=str(applied_core["x_coord_col"]),
        y_coord_col=str(applied_core["y_coord_col"]),
        batch_col=str(applied_core["batch_col"]),
        roi_col=str(applied_core["roi_col"]),
        min_cells_group=int(applied_core["min_cells_group"]),
        use_median=bool(applied_core["use_median"]),
        percentile=float(applied_core["percentile"]),
        min_hole_size=float(applied_core["min_hole_size"]),
        core_filter_values=_freeze_filter_map(applied_core.get("core_filter_values", {})),
    )
    if context.get("error"):
        st.error(str(context["error"]))
        return
    if context.get("warning"):
        st.warning(str(context["warning"]))
        return

    run_df = pd.DataFrame(context["run_df"])
    connected_cols = [str(col) for col in context["connected_cols"]]
    group_df = pd.DataFrame(context["group_df"])
    preview_options = [str(option) for option in context["preview_options"]]
    percentile_value = context["percentile_value"]
    if percentile_value is not None:
        percentile_value = float(percentile_value)
    non_coord_run_cols = [str(col) for col in context["non_coord_run_cols"]]
    pixel_size_um = float(applied_core["pixel_size_um"])
    min_hole_size = float(applied_core["min_hole_size"])
    min_cells_group = int(applied_core["min_cells_group"])
    output_dir = str(applied_core["output_dir"])

    st.caption(
        f"Prepared {context['run_rows']} rows across {len(group_df)} connected tissues "
        f"after filtering {context['filtered_rows']} candidate rows."
    )

    with st.expander("Connected Tissue Preview (quality check for the shared configuration)", expanded=False):
        st.caption(
            "Shows the virtual tissue geometry and the nearest-neighbour distance histogram for a single "
            "connected tissue, so you can confirm the buffer percentile and hole size before running a panel."
        )
        if not preview_options or group_df.empty:
            st.info("No connected tissue groups available for preview with the current settings.")
        else:
            show_preview = st.checkbox(
                "Draw preview",
                value=False,
                key="tab8_density_preview_enabled",
                help="Off by default because the geometry is rebuilt every time this tab redraws.",
            )
            preview_value = _sync_selectbox_state("tab8_density_preview_group", preview_options, preview_options[0])
            preview_choice = st.selectbox(
                "Connected tissue group",
                options=preview_options,
                index=preview_options.index(preview_value) if preview_value in preview_options else 0,
                key="tab8_density_preview_group",
            )
            if show_preview:
                selected_idx = preview_options.index(preview_choice) if preview_choice in preview_options else 0
                selected_preview_row = group_df.iloc[selected_idx]
                mask_preview = np.ones(len(run_df), dtype=bool)
                for col in connected_cols:
                    mask_preview &= run_df[col].astype(str).to_numpy() == str(selected_preview_row[col])
                preview_df = run_df[mask_preview].copy()
                fig_preview = _draw_connected_tissue_preview(
                    group_df=preview_df,
                    title=preview_options[selected_idx],
                    percentile=percentile_value,
                    min_hole_size=min_hole_size,
                )
                if fig_preview is not None:
                    st.pyplot(fig_preview)
                    plt.close(fig_preview)

    phenotype_cols = [c for c in run_df.columns if "phenotype" in c.lower()]
    default_methods = [c for c in ["phenotype_leiden", "phenotype_som", "phenotype_threshold", "phenotype_scimap"] if c in phenotype_cols]

    with st.expander("Shared Figure Settings · methods, display names and colors", expanded=True):
        shared = _render_shared_figure_settings(phenotype_cols, default_methods)

    st.markdown("### Panels")

    with st.expander("**Panel 1 · Phenotype Density by Phenotyping Method**", expanded=False):
        render_panel1_fragment(
            run_df=run_df,
            phenotype_cols=phenotype_cols,
            shared=shared,
            non_coord_run_cols=non_coord_run_cols,
            connected_cols=connected_cols,
            pixel_size_um=pixel_size_um,
            min_hole_size=min_hole_size,
            min_cells_group=min_cells_group,
            output_dir=output_dir,
            percentile_value=percentile_value,
        )

    with st.expander("**Panel 2 · Density by Distance from an Annotation**", expanded=False):
        render_panel2_fragment(
            run_df=run_df,
            non_coord_run_cols=non_coord_run_cols,
            shared=shared,
            pixel_size_um=pixel_size_um,
            min_hole_size=min_hole_size,
            min_cells_group=min_cells_group,
            output_dir=output_dir,
            percentile_value=percentile_value,
            connected_cols=connected_cols,
            phenotype_cols=phenotype_cols,
            project_root=project_root,
        )

    with st.expander("**Panel 3 · Tissue Density Clouds**", expanded=False):
        render_panel3_fragment(
            run_df=run_df,
            phenotype_cols=phenotype_cols,
            shared=shared,
            batch_col=str(applied_core["batch_col"]),
            roi_col=str(applied_core["roi_col"]),
            pixel_size_um=pixel_size_um,
            output_dir=output_dir,
            project_root=project_root,
        )
