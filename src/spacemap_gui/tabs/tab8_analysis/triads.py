from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from . import density_clouds as clouds
from .common import load_tab8_config_for_csv, save_tab8_config_for_csv, _is_transient_tab8_key, _to_jsonable


from spacemap_gui import terminal


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab8', message, step='Triads', level=level)


def _show_png_bytes(payload: Any, **kwargs) -> None:
    """Render cached figure bytes, ignoring stale non-binary values.

    Older project files stored the *repr* of these bytes, which Streamlit cannot
    open; such leftovers are skipped instead of raising MediaFileStorageError.
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        st.image(payload, **kwargs)
from .virtual_tissue import _build_region_geometry, _compute_buffer_distance
from .density_analysis import (
    _bracket_top_axes_fraction,
    _compute_panel2_stats,
    _panel1_log_base,
    _panel1_add_significance_brackets,
    _panel1_apply_y_axis,
    _panel1_compute_outlier_crop,
    _panel1_transform_for_break,
    _panel1_draw_axis_break,
    _panel1_compute_axis_break,
    _panel1_transform_array_for_split,
    _panel1_apply_split_axis,
    compute_matched_method_statistics,
    compute_zonal_method_statistics,
    copy_outputs_to_bundle,
    create_publication_bundle,
    render_shared_figure_settings,
    render_zonal_density_figure,
    render_zonal_plot_controls,
    write_publication_metadata,
)

try:
    import tifffile
except Exception:
    tifffile = None

try:
    from shapely.geometry import MultiPoint, Point, shape
    from shapely.ops import unary_union
except Exception:
    MultiPoint = None
    Point = None
    shape = None
    unary_union = None

REQUIRED_COORDS = ["x_centroid", "y_centroid"]


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    if preferred in options:
        return options.index(preferred)
    return fallback


def _format_opt(value: str) -> str:
    return value if value else "(not selected)"


def _column_options(columns: list[str]) -> list[str]:
    return [""] + list(columns)


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _init_selectbox_state(key: str, options: list[str], preferred: str | None = None, fallback: int = 0) -> None:
    if not options:
        return
    current = st.session_state.get(key)
    if current in options:
        return
    if preferred is not None and preferred in options:
        st.session_state[key] = preferred
        return
    safe_idx = min(max(int(fallback), 0), len(options) - 1)
    st.session_state[key] = options[safe_idx]


def _init_multiselect_state(key: str, options: list[str], default_value: list[str]) -> None:
    current = st.session_state.get(key)
    if isinstance(current, list):
        filtered = [v for v in current if v in options]
        st.session_state[key] = filtered
    else:
        st.session_state[key] = default_value


def _safe_unique(series: pd.Series) -> list[str]:
    vals = series.dropna().astype(str).unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def _resolve_triad_names_from_instances(
    triad_instances_df: pd.DataFrame,
    selected_style: str | None,
    fallback: tuple[str, str, str],
) -> tuple[str, str, str]:
    if triad_instances_df is None or triad_instances_df.empty:
        return fallback

    work = triad_instances_df.copy()
    if selected_style and "style" in work.columns:
        style_work = work[work["style"].astype(str) == str(selected_style)].copy()
        if not style_work.empty:
            work = style_work

    if all(col in work.columns for col in ["pheno_a", "pheno_b", "pheno_c"]):
        ph = work[["pheno_a", "pheno_b", "pheno_c"]].dropna()
        if not ph.empty:
            ph = ph.astype(str)
            try:
                top = ph.value_counts().index[0]
                return str(top[0]), str(top[1]), str(top[2])
            except Exception:
                row0 = ph.iloc[0]
                return str(row0["pheno_a"]), str(row0["pheno_b"]), str(row0["pheno_c"])

    if "triad_label" in work.columns:
        labels = work["triad_label"].dropna().astype(str)
        if not labels.empty:
            label = str(labels.iloc[0])
            parts = label.split("::")
            if len(parts) == 3:
                return str(parts[0]), str(parts[1]), str(parts[2])

    return fallback


def _um_to_px(distance_um: float, micron_per_pixel: float) -> float:
    return float(distance_um) / float(micron_per_pixel)


def _px_to_um(distance_px: float, micron_per_pixel: float) -> float:
    return float(distance_px) * float(micron_per_pixel)


def _log_terminal_event(message: str) -> None:
    _log_console(message)


def _build_adjacency(coords: np.ndarray, min_dist_px: float, max_dist_px: float) -> list[set[int]]:
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(max_dist_px), output_type="ndarray")
    adj: list[set[int]] = [set() for _ in range(len(coords))]

    if pairs is None or len(pairs) == 0:
        return adj

    if float(min_dist_px) <= 0:
        for i, j in pairs:
            ii, jj = int(i), int(j)
            adj[ii].add(jj)
            adj[jj].add(ii)
        return adj

    min_d2 = float(min_dist_px) ** 2
    for i, j in pairs:
        ii, jj = int(i), int(j)
        dx = float(coords[ii, 0] - coords[jj, 0])
        dy = float(coords[ii, 1] - coords[jj, 1])
        if (dx * dx + dy * dy) >= min_d2:
            adj[ii].add(jj)
            adj[jj].add(ii)
    return adj


def _count_and_collect_triads_anchor(
    phenos: np.ndarray,
    adj: list[set[int]],
    triad: tuple[str, str, str],
    max_collect: int,
) -> tuple[int, list[tuple[int, int, int]]]:
    p_anchor, p_b, p_c = triad
    anchor_idx = np.where(phenos == str(p_anchor))[0]
    triad_count = 0
    collected: list[tuple[int, int, int]] = []

    for a in anchor_idx:
        neigh = adj[int(a)]
        if not neigh:
            continue

        b_neighbors = [j for j in neigh if phenos[int(j)] == str(p_b)]
        c_neighbors = [k for k in neigh if phenos[int(k)] == str(p_c)]
        if not b_neighbors or not c_neighbors:
            continue

        for j in b_neighbors:
            adj_j = adj[int(j)]
            for k in c_neighbors:
                if int(k) in adj_j:
                    triad_count += 1
                    if len(collected) < int(max_collect):
                        collected.append((int(a), int(j), int(k)))

    return int(triad_count), collected


def _count_triads_anchor(phenos: np.ndarray, adj: list[set[int]], triad: tuple[str, str, str]) -> int:
    count, _ = _count_and_collect_triads_anchor(phenos, adj, triad, max_collect=0)
    return int(count)


def _detect_triad_enrichment(
    df: pd.DataFrame,
    triad_phenotypes: tuple[str, str, str],
    phenotype_col: str,
    batch_col: str,
    roi_col: str,
    tissue_col: str,
    tissue_keep_value: str,
    min_dist_um: float,
    max_dist_um: float,
    micron_per_pixel: float,
    min_cells_per_region: int,
    n_permutations: int,
    random_state: int,
    phenotype_subset_only: bool,
    max_triad_instances_to_save: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, pd.DataFrame]:
    work = df.copy()
    work = work.dropna(subset=REQUIRED_COORDS + [phenotype_col])

    if tissue_col != "None" and tissue_col in work.columns:
        work = work[work[tissue_col].astype(str) == str(tissue_keep_value)].copy()

    if batch_col:
        work = work[work[batch_col].notna()].copy()
    else:
        work["__triad_batch__"] = "ALL"
        batch_col = "__triad_batch__"

    if roi_col:
        work = work[work[roi_col].notna()].copy()
    else:
        work["__triad_roi__"] = "ALL"
        roi_col = "__triad_roi__"

    triad_set = {str(v) for v in triad_phenotypes}
    work[phenotype_col] = work[phenotype_col].astype(str)
    if phenotype_subset_only:
        work = work[work[phenotype_col].isin(triad_set)].copy()

    if work.empty or (not triad_set.issubset(set(work[phenotype_col].unique().tolist()))):
        summary = {
            "triad": "::".join(triad_phenotypes),
            "observed_total_count": 0,
            "expected_mean_count": 0.0,
            "expected_std_count": 0.0,
            "z_score": np.nan,
            "empirical_p_value_enrichment": 1.0,
            "n_valid_regions": 0,
            "n_permutations": int(n_permutations),
            "min_dist_um": float(min_dist_um),
            "max_dist_um": float(max_dist_um),
            "min_dist_px": float(_um_to_px(min_dist_um, micron_per_pixel)),
            "max_dist_px": float(_um_to_px(max_dist_um, micron_per_pixel)),
            "n_triad_instances_saved": 0,
        }
        return summary, pd.DataFrame(), np.zeros(int(n_permutations), dtype=int), pd.DataFrame()

    min_dist_px = _um_to_px(float(min_dist_um), float(micron_per_pixel))
    max_dist_px = _um_to_px(float(max_dist_um), float(micron_per_pixel))

    region_payloads: list[dict[str, Any]] = []
    per_region_rows: list[dict[str, Any]] = []
    triad_rows: list[dict[str, Any]] = []

    grouped = list(work.groupby([batch_col, roi_col], dropna=False))
    _log_terminal_event(f"Detect triads | regions={len(grouped)} | dist_um=({min_dist_um:.1f},{max_dist_um:.1f})")

    for region_idx, ((bval, rval), gdf) in enumerate(grouped, start=1):
        if len(gdf) < int(min_cells_per_region):
            continue

        phenos = gdf[phenotype_col].astype(str).to_numpy(copy=True)
        if not triad_set.issubset(set(phenos.tolist())):
            continue

        coords = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        adj = _build_adjacency(coords, min_dist_px=min_dist_px, max_dist_px=max_dist_px)
        if sum(len(s) for s in adj) == 0:
            continue

        obs, triads_local = _count_and_collect_triads_anchor(
            phenos=phenos,
            adj=adj,
            triad=triad_phenotypes,
            max_collect=max(0, int(max_triad_instances_to_save) - len(triad_rows)),
        )

        region_payloads.append(
            {
                "phenos": phenos,
                "adj": adj,
                "batch": str(bval),
                "roi": str(rval),
                "n_cells": int(len(gdf)),
                "obs": int(obs),
            }
        )
        per_region_rows.append(
            {
                "batch": str(bval),
                "roi": str(rval),
                "n_cells": int(len(gdf)),
                "observed_triad_count": int(obs),
            }
        )

        gdf_local = gdf.reset_index().rename(columns={"index": "global_row_index"})
        for a, b, c in triads_local:
            cell_a = gdf_local.iloc[int(a)]
            cell_b = gdf_local.iloc[int(b)]
            cell_c = gdf_local.iloc[int(c)]
            triad_rows.append(
                {
                    "batch": str(bval),
                    "roi": str(rval),
                    "phenotype_column": str(phenotype_col),
                    "triad_label": "::".join(triad_phenotypes),
                    "pheno_a": str(triad_phenotypes[0]),
                    "pheno_b": str(triad_phenotypes[1]),
                    "pheno_c": str(triad_phenotypes[2]),
                    "cell_a_global_index": int(cell_a["global_row_index"]),
                    "cell_b_global_index": int(cell_b["global_row_index"]),
                    "cell_c_global_index": int(cell_c["global_row_index"]),
                    "cell_a_x": float(cell_a["x_centroid"]),
                    "cell_a_y": float(cell_a["y_centroid"]),
                    "cell_b_x": float(cell_b["x_centroid"]),
                    "cell_b_y": float(cell_b["y_centroid"]),
                    "cell_c_x": float(cell_c["x_centroid"]),
                    "cell_c_y": float(cell_c["y_centroid"]),
                    "triad_centroid_x": float((cell_a["x_centroid"] + cell_b["x_centroid"] + cell_c["x_centroid"]) / 3.0),
                    "triad_centroid_y": float((cell_a["y_centroid"] + cell_b["y_centroid"] + cell_c["y_centroid"]) / 3.0),
                }
            )

        if region_idx % 20 == 0:
            _log_terminal_event(f"Processed {region_idx}/{len(grouped)} regions | saved_triads={len(triad_rows)}")

    observed_total = int(sum(r["obs"] for r in region_payloads))
    if not region_payloads:
        summary = {
            "triad": "::".join(triad_phenotypes),
            "observed_total_count": 0,
            "expected_mean_count": 0.0,
            "expected_std_count": 0.0,
            "z_score": np.nan,
            "empirical_p_value_enrichment": 1.0,
            "n_valid_regions": 0,
            "n_permutations": int(n_permutations),
            "min_dist_um": float(min_dist_um),
            "max_dist_um": float(max_dist_um),
            "min_dist_px": float(min_dist_px),
            "max_dist_px": float(max_dist_px),
            "n_triad_instances_saved": int(len(triad_rows)),
        }
        return summary, pd.DataFrame(per_region_rows), np.zeros(int(n_permutations), dtype=int), pd.DataFrame(triad_rows)

    rng = np.random.default_rng(int(random_state))
    null_counts = np.zeros(int(n_permutations), dtype=int)
    log_every = max(1, int(n_permutations // 10))
    _log_terminal_event(f"Starting permutations={int(n_permutations)}")

    for i in range(int(n_permutations)):
        total_perm = 0
        for rec in region_payloads:
            shuffled = rec["phenos"].copy()
            rng.shuffle(shuffled)
            total_perm += _count_triads_anchor(shuffled, rec["adj"], triad_phenotypes)
        null_counts[i] = int(total_perm)
        if (i + 1) % log_every == 0 or (i + 1) == int(n_permutations):
            _log_terminal_event(f"Permutation progress: {i + 1}/{int(n_permutations)}")

    null_mean = float(np.mean(null_counts))
    null_std = float(np.std(null_counts, ddof=0))
    z_score = (float(observed_total) - null_mean) / null_std if null_std > 0 else np.nan
    p_emp = float((np.sum(null_counts >= observed_total) + 1) / (int(n_permutations) + 1))

    summary = {
        "triad": "::".join(triad_phenotypes),
        "observed_total_count": int(observed_total),
        "expected_mean_count": float(null_mean),
        "expected_std_count": float(null_std),
        "z_score": float(z_score) if np.isfinite(z_score) else np.nan,
        "empirical_p_value_enrichment": float(p_emp),
        "n_valid_regions": int(len(region_payloads)),
        "n_permutations": int(n_permutations),
        "min_dist_um": float(min_dist_um),
        "max_dist_um": float(max_dist_um),
        "min_dist_px": float(min_dist_px),
        "max_dist_px": float(max_dist_px),
        "n_triad_instances_saved": int(len(triad_rows)),
    }

    per_region_df = pd.DataFrame(per_region_rows)
    if not per_region_df.empty:
        per_region_df = per_region_df.sort_values("observed_triad_count", ascending=False).reset_index(drop=True)

    return summary, per_region_df, null_counts, pd.DataFrame(triad_rows)


def _plot_null_distribution(
    null_counts: np.ndarray,
    observed: int,
    title: str,
    empirical_p_value: float | None = None,
    z_score: float | None = None,
):
    null_vals = np.asarray(null_counts, dtype=float)
    if null_vals.size == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8))
        ax.text(0.5, 0.5, "No permutation counts available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    obs = float(observed)
    null_mean = float(np.mean(null_vals))
    null_std = float(np.std(null_vals, ddof=0))
    perc_rank = 100.0 * float(np.mean(null_vals <= obs))

    q01 = float(np.percentile(null_vals, 1))
    q99 = float(np.percentile(null_vals, 99))
    pad = max((q99 - q01) * 0.12, 5.0)
    zoom_min = q01 - pad
    zoom_max = q99 + pad

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Full range panel
    ax_full.hist(null_vals, bins=40, color="#d9d9d9", alpha=0.95, edgecolor="white")
    ax_full.axvline(obs, color="#d62728", linestyle="--", linewidth=2.2, label=f"Observed={int(obs)}")
    ax_full.set_xlabel("Permuted triad count")
    ax_full.set_ylabel("Frequency")
    ax_full.set_title(title)
    ax_full.grid(True, alpha=0.2)
    ax_full.legend(loc="best", fontsize=8)

    stats_lines = [
        f"Null mean ± sd: {null_mean:.1f} ± {null_std:.1f}",
        f"Observed percentile in null: {perc_rank:.2f}%",
    ]
    if empirical_p_value is not None and np.isfinite(empirical_p_value):
        stats_lines.append(f"Empirical p (enrichment): {float(empirical_p_value):.4g}")
    if z_score is not None and np.isfinite(z_score):
        stats_lines.append(f"Z-score: {float(z_score):.2f}")
    ax_full.text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        transform=ax_full.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.3"},
    )

    # Zoomed null panel
    ax_zoom.hist(null_vals, bins=35, color="#8da0cb", alpha=0.85, edgecolor="white")
    ax_zoom.set_xlim(zoom_min, zoom_max)
    ax_zoom.set_xlabel("Permuted triad count (zoomed null)")
    ax_zoom.set_ylabel("Frequency")
    ax_zoom.set_title("Zoomed null range")
    ax_zoom.grid(True, alpha=0.2)

    if zoom_min <= obs <= zoom_max:
        ax_zoom.axvline(obs, color="#d62728", linestyle="--", linewidth=2.0)
    else:
        edge_x = zoom_max if obs > zoom_max else zoom_min
        ax_zoom.axvline(edge_x, color="#d62728", linestyle="--", linewidth=2.0)
        ax_zoom.annotate(
            f"Observed={int(obs)}\noutside zoom",
            xy=(edge_x, ax_zoom.get_ylim()[1] * 0.72),
            xytext=(-6 if obs > zoom_max else 6, 0),
            textcoords="offset points",
            ha="right" if obs > zoom_max else "left",
            va="center",
            fontsize=8,
            color="#d62728",
        )

    plt.tight_layout()
    return fig


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


def _add_pairwise_significance_brackets(
    ax,
    methods: list[str],
    pairwise_stats: list[dict],
    y_data_max: float,
    y_data_min: float,
) -> float:
    sig = [r for r in pairwise_stats if np.isfinite(r.get("p_value", np.nan)) and float(r["p_value"]) < 0.05]
    if not sig:
        return y_data_max

    span = y_data_max - y_data_min
    if not np.isfinite(span) or span <= 0:
        span = max(abs(y_data_max), 1.0)
    step = 0.08 * span
    bracket_h = 0.025 * span
    base_y = y_data_max + (0.12 * span)

    intervals_per_level: list[list[tuple[float, float]]] = []
    placements: list[tuple[int, int, int, float]] = []

    parsed: list[tuple[int, int, float]] = []
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

    for lo, hi, lvl, p_val in placements:
        y = base_y + (lvl * step)
        ax.plot([lo, lo, hi, hi], [y, y + bracket_h, y + bracket_h, y], color="black", linewidth=1.0, zorder=6)
        ax.text(
            (lo + hi) / 2.0,
            y + bracket_h + (0.01 * span),
            _p_to_sig_label(p_val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            zorder=7,
        )

    max_level = max((lvl for _, _, lvl, _ in placements), default=0)
    return base_y + ((max_level + 1) * step) + (2.5 * bracket_h)


def _virtual_area_mm2(
    coords: np.ndarray,
    pixel_size_um: float,
    method: str,
    alpha: float,
    use_median: bool,
    percentile: float,
    min_hole_size: float,
) -> tuple[float, float]:
    if len(coords) < 3:
        return np.nan, np.nan

    buffer_distance = _compute_buffer_distance(coords, use_median=bool(use_median), percentile=float(percentile))
    if not np.isfinite(buffer_distance) or float(buffer_distance) <= 0:
        return np.nan, np.nan

    polygon = _build_region_geometry(
        coords=coords,
        method=str(method),
        alpha=float(alpha),
        buffer_distance=float(buffer_distance),
        min_hole_size=float(min_hole_size),
    )
    if polygon is None:
        return np.nan, np.nan

    area_px2 = float(polygon.area)
    area_mm2 = area_px2 * ((float(pixel_size_um) / 1000.0) ** 2)
    return float(area_mm2), float(buffer_distance)


def _compute_group_triad_metrics(
    df: pd.DataFrame,
    methods: list[str],
    triad: tuple[str, str, str],
    style_name: str,
    min_dist_um: float,
    max_dist_um: float,
    pixel_size_um: float,
    min_cells_per_region: int,
    area_method: str,
    area_alpha: float,
    area_use_median: bool,
    area_percentile: float,
    min_hole_size: float,
    strat_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    min_px = _um_to_px(float(min_dist_um), float(pixel_size_um))
    max_px = _um_to_px(float(max_dist_um), float(pixel_size_um))

    grouped = df.groupby(["__batch__", "__roi__"], dropna=False)
    triad_set = {str(t) for t in triad}
    for (batch_val, roi_val), gdf in grouped:
        gdf = gdf.dropna(subset=REQUIRED_COORDS)
        if len(gdf) < int(min_cells_per_region):
            continue

        coords = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        area_mm2, buffer_px = _virtual_area_mm2(
            coords=coords,
            pixel_size_um=float(pixel_size_um),
            method=str(area_method),
            alpha=float(area_alpha),
            use_median=bool(area_use_median),
            percentile=float(area_percentile),
            min_hole_size=float(min_hole_size),
        )
        if not np.isfinite(area_mm2) or area_mm2 <= 0:
            continue

        adj = _build_adjacency(coords=coords, min_dist_px=float(min_px), max_dist_px=float(max_px))
        if sum(len(s) for s in adj) == 0:
            continue

        base_row = {
            "batch": str(batch_val),
            "roi": str(roi_val),
            "group_id": f"{batch_val} | {roi_val}",
            "style": str(style_name),
            "n_cells": int(len(gdf)),
            "area_mm2": float(area_mm2),
            "buffer_distance_px": float(buffer_px) if np.isfinite(buffer_px) else np.nan,
        }
        for s_col in strat_cols:
            base_row[s_col] = gdf[s_col].iloc[0] if s_col in gdf.columns and len(gdf) > 0 else None

        for method_col in methods:
            if method_col not in gdf.columns:
                continue
            phenos = gdf[method_col].astype(str).to_numpy(copy=True)
            tri_count = 0
            if triad_set.issubset(set(phenos.tolist())):
                tri_count = int(_count_triads_anchor(phenos=phenos, adj=adj, triad=triad))
            row = dict(base_row)
            row["method"] = str(method_col)
            row["triad_count"] = int(tri_count)
            row["triad_density_per_mm2"] = float(tri_count / area_mm2)
            rows.append(row)

    return pd.DataFrame(rows)


def _compute_zone_triad_metrics(
    df: pd.DataFrame,
    methods: list[str],
    triad: tuple[str, str, str],
    style_name: str,
    min_dist_um: float,
    max_dist_um: float,
    pixel_size_um: float,
    min_cells_per_region: int,
    tumor_col: str,
    tumor_positive_value: str,
    zone_width_um: float,
    max_zone_distance_um: float,
    min_cells_per_zone: int,
    min_rois_per_curve: int,
    area_method: str,
    area_alpha: float,
    area_use_median: bool,
    area_percentile: float,
    min_hole_size: float,
    strat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if MultiPoint is None or Point is None:
        return pd.DataFrame(), pd.DataFrame()

    min_px = _um_to_px(float(min_dist_um), float(pixel_size_um))
    max_px = _um_to_px(float(max_dist_um), float(pixel_size_um))
    zone_width_px = _um_to_px(float(zone_width_um), float(pixel_size_um))
    max_zone_px = _um_to_px(float(max_zone_distance_um), float(pixel_size_um))
    zone_edges = np.arange(-max_zone_px, max_zone_px + zone_width_px, zone_width_px)
    if len(zone_edges) < 2:
        return pd.DataFrame(), pd.DataFrame()

    triad_set = {str(t) for t in triad}
    zone_rows: list[dict[str, Any]] = []

    grouped = df.groupby(["__batch__", "__roi__"], dropna=False)
    for (batch_val, roi_val), gdf in grouped:
        gdf = gdf.dropna(subset=REQUIRED_COORDS)
        if len(gdf) < int(min_cells_per_region):
            continue

        tumor_mask = gdf[tumor_col].astype(str) == str(tumor_positive_value)
        tumor_cells = gdf[tumor_mask].dropna(subset=REQUIRED_COORDS)
        if len(tumor_cells) < 3:
            continue

        tumor_coords = tumor_cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        tumor_hull = MultiPoint([Point(float(x), float(y)) for x, y in tumor_coords]).convex_hull
        tumor_boundary = tumor_hull.boundary

        coords_all = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        points_all = [Point(float(x), float(y)) for x, y in coords_all]
        signed_dist: list[float] = []
        for idx_pt, pt in enumerate(points_all):
            dist = float(pt.distance(tumor_boundary))
            is_tumor = str(gdf.iloc[idx_pt][tumor_col]) == str(tumor_positive_value)
            signed_dist.append(-dist if is_tumor else dist)

        gdf = gdf.copy()
        gdf["__distance_from_tumor_px__"] = np.asarray(signed_dist, dtype=float)

        for z_idx in range(len(zone_edges) - 1):
            z_min = float(zone_edges[z_idx])
            z_max = float(zone_edges[z_idx + 1])
            z_center_um = _px_to_um((z_min + z_max) / 2.0, float(pixel_size_um))

            zdf = gdf[(gdf["__distance_from_tumor_px__"] >= z_min) & (gdf["__distance_from_tumor_px__"] < z_max)].copy()
            if len(zdf) < int(min_cells_per_zone):
                continue

            z_coords = zdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
            area_mm2, _ = _virtual_area_mm2(
                coords=z_coords,
                pixel_size_um=float(pixel_size_um),
                method=str(area_method),
                alpha=float(area_alpha),
                use_median=bool(area_use_median),
                percentile=float(area_percentile),
                min_hole_size=float(min_hole_size),
            )
            if not np.isfinite(area_mm2) or area_mm2 <= 0:
                continue

            adj = _build_adjacency(coords=z_coords, min_dist_px=float(min_px), max_dist_px=float(max_px))
            if sum(len(s) for s in adj) == 0:
                continue

            base_row = {
                "batch": str(batch_val),
                "roi": str(roi_val),
                "group_id": f"{batch_val} | {roi_val}",
                "style": str(style_name),
                "zone_center_um": float(z_center_um),
                "zone_min_um": _px_to_um(z_min, float(pixel_size_um)),
                "zone_max_um": _px_to_um(z_max, float(pixel_size_um)),
                "n_cells_zone": int(len(zdf)),
                "area_mm2": float(area_mm2),
            }
            for s_col in strat_cols:
                base_row[s_col] = gdf[s_col].iloc[0] if s_col in gdf.columns and len(gdf) > 0 else None

            for method_col in methods:
                if method_col not in zdf.columns:
                    continue
                phenos = zdf[method_col].astype(str).to_numpy(copy=True)
                tri_count = 0
                if triad_set.issubset(set(phenos.tolist())):
                    tri_count = int(_count_triads_anchor(phenos=phenos, adj=adj, triad=triad))
                row = dict(base_row)
                row["method"] = str(method_col)
                row["triad_count"] = int(tri_count)
                row["triad_density_per_mm2"] = float(tri_count / area_mm2)
                zone_rows.append(row)

    detailed_df = pd.DataFrame(zone_rows)
    if detailed_df.empty:
        return detailed_df, pd.DataFrame()

    group_keys = ["style", "method", "zone_center_um"] + strat_cols
    roi_counts = (
        detailed_df.groupby(group_keys, dropna=False)["group_id"]
        .nunique()
        .reset_index(name="n_connected_tissues")
    )
    valid_roi = roi_counts[roi_counts["n_connected_tissues"] >= int(min_rois_per_curve)]
    if valid_roi.empty:
        return detailed_df, pd.DataFrame()

    detailed_df = detailed_df.merge(valid_roi[group_keys], on=group_keys, how="inner")
    if detailed_df.empty:
        return detailed_df, pd.DataFrame()

    summary_df = (
        detailed_df.groupby(group_keys, dropna=False)
        .agg(
            triad_density_mean=("triad_density_per_mm2", "mean"),
            triad_density_std=("triad_density_per_mm2", "std"),
            triad_count_mean=("triad_count", "mean"),
            n_connected_tissues=("group_id", "nunique"),
        )
        .reset_index()
    )
    summary_df["triad_density_sem"] = summary_df["triad_density_std"] / np.sqrt(np.maximum(summary_df["n_connected_tissues"], 1))
    return detailed_df, summary_df


def _compute_zonal_triad_density(
    df: pd.DataFrame,
    methods: list[str],
    triad: tuple[str, str, str],
    min_dist_um: float,
    max_dist_um: float,
    pixel_size_um: float,
    annotation_col: str,
    reference_values: list[str],
    distance_col: str,
    zone_min_um: float,
    zone_max_um: float,
    zone_width_um: float,
    min_cells_group: int,
    min_cells_per_zone: int,
    min_rois_per_curve: int,
    area_method: str,
    area_alpha: float,
    area_use_median: bool,
    area_percentile: float,
    min_hole_size: float,
    strat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Triad density per distance zone, following the Density subtab's zonal recipe.

    Distances are signed exactly like Density Panel 2 (negative inside the reference
    annotation, positive outside); inside each zone the triads are counted on the
    proximity graph built from that zone's cells and divided by the zone's virtual
    tissue area. Column names match the Density panel so the statistics and the
    figure renderer are shared.
    """

    if MultiPoint is None or Point is None:
        return pd.DataFrame(), pd.DataFrame()

    triad_label = "::".join(str(t) for t in triad)
    triad_set = {str(t) for t in triad}
    min_px = _um_to_px(float(min_dist_um), float(pixel_size_um))
    max_px = _um_to_px(float(max_dist_um), float(pixel_size_um))

    zone_width_px = float(zone_width_um) / float(pixel_size_um)
    zone_min_px = float(zone_min_um) / float(pixel_size_um)
    zone_max_px = float(zone_max_um) / float(pixel_size_um)
    zone_edges = np.arange(zone_min_px, zone_max_px + zone_width_px, zone_width_px)
    if len(zone_edges) < 2:
        return pd.DataFrame(), pd.DataFrame()

    zone_rows: list[dict[str, Any]] = []
    for (batch_val, roi_val), gdf in df.groupby(["__batch__", "__roi__"], dropna=False):
        gdf = gdf.dropna(subset=REQUIRED_COORDS)
        if len(gdf) < int(min_cells_group):
            continue

        if distance_col and distance_col in gdf.columns:
            signed_dist = pd.to_numeric(gdf[distance_col], errors="coerce").to_numpy(dtype=float)
        else:
            if not annotation_col or annotation_col not in gdf.columns:
                continue
            ref_mask = gdf[annotation_col].astype(str).isin([str(v) for v in reference_values]).to_numpy()
            ref_cells = gdf[ref_mask].dropna(subset=REQUIRED_COORDS)
            if len(ref_cells) < 3:
                continue
            ref_coords = ref_cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
            ref_boundary = MultiPoint([Point(float(x), float(y)) for x, y in ref_coords]).convex_hull.boundary
            coords_all = gdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
            signed_dist = np.empty(len(coords_all), dtype=float)
            for idx_pt, (x_coord, y_coord) in enumerate(coords_all):
                dist = float(Point(float(x_coord), float(y_coord)).distance(ref_boundary))
                signed_dist[idx_pt] = -dist if bool(ref_mask[idx_pt]) else dist

        zone_ids = np.digitize(signed_dist, zone_edges, right=False) - 1
        valid_zone_mask = (zone_ids >= 0) & (zone_ids < (len(zone_edges) - 1))
        if not np.any(valid_zone_mask):
            continue

        gdf = gdf.copy()
        gdf["__zone_idx__"] = zone_ids
        zone_ready = gdf[valid_zone_mask].copy()
        if zone_ready.empty:
            continue

        for z_idx, zdf in zone_ready.groupby("__zone_idx__", sort=True):
            z_idx_int = int(z_idx)
            z_min = float(zone_edges[z_idx_int])
            z_max = float(zone_edges[z_idx_int + 1])
            z_center_um = ((z_min + z_max) / 2.0) * float(pixel_size_um)
            if len(zdf) < max(2, int(min_cells_per_zone)):
                continue

            z_coords = zdf[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
            area_mm2, _ = _virtual_area_mm2(
                coords=z_coords,
                pixel_size_um=float(pixel_size_um),
                method=str(area_method),
                alpha=float(area_alpha),
                use_median=bool(area_use_median),
                percentile=float(area_percentile),
                min_hole_size=float(min_hole_size),
            )
            if not np.isfinite(area_mm2) or area_mm2 <= 0:
                continue

            adj = _build_adjacency(coords=z_coords, min_dist_px=float(min_px), max_dist_px=float(max_px))
            for method_col in methods:
                if method_col not in zdf.columns:
                    continue
                phenos = zdf[method_col].astype(str).to_numpy(copy=True)
                tri_count = 0
                if triad_set.issubset(set(phenos.tolist())) and sum(len(s) for s in adj) > 0:
                    tri_count = int(_count_triads_anchor(phenos=phenos, adj=adj, triad=triad))
                row: dict[str, Any] = {
                    "phenotype": triad_label,
                    "method": str(method_col),
                    "batch": str(batch_val),
                    "roi": str(roi_val),
                    "__group_id__": f"{batch_val} | {roi_val}",
                    "zone_center_um": float(z_center_um),
                    "zone_min_um": float(z_min * float(pixel_size_um)),
                    "zone_max_um": float(z_max * float(pixel_size_um)),
                    "n_cells_zone": int(len(zdf)),
                    "n_triads_zone": int(tri_count),
                    "area_mm2": float(area_mm2),
                    "density_cells_per_mm2": float(tri_count / area_mm2),
                }
                for s_col in strat_cols:
                    row[s_col] = zdf[s_col].iloc[0] if s_col in zdf.columns and len(zdf) > 0 else None
                zone_rows.append(row)

    zone_df = pd.DataFrame(zone_rows)
    if zone_df.empty:
        return zone_df, pd.DataFrame()

    group_keys = ["phenotype", "method", "zone_center_um"] + list(strat_cols)
    roi_counts = (
        zone_df.groupby(group_keys, dropna=False)["__group_id__"]
        .nunique()
        .reset_index(name="n_connected_tissues")
    )
    valid_roi = roi_counts[roi_counts["n_connected_tissues"] >= int(min_rois_per_curve)]
    if valid_roi.empty:
        return zone_df, pd.DataFrame()

    zone_df = zone_df.merge(valid_roi[group_keys], on=group_keys, how="inner")
    if zone_df.empty:
        return zone_df, pd.DataFrame()

    summary_df = (
        zone_df.groupby(group_keys, dropna=False)
        .agg(
            density_mean=("density_cells_per_mm2", "mean"),
            density_std=("density_cells_per_mm2", "std"),
            n_connected_tissues=("__group_id__", "nunique"),
            total_triads_zone=("n_triads_zone", "sum"),
            total_cells_zone=("n_cells_zone", "sum"),
            total_area_mm2=("area_mm2", "sum"),
        )
        .reset_index()
    )
    summary_df["density_sem"] = summary_df["density_std"] / np.sqrt(np.maximum(summary_df["n_connected_tissues"], 1))
    return zone_df, summary_df


def _collect_region_triad_instances(
    region_df: pd.DataFrame,
    phenotype_col: str,
    triad: tuple[str, str, str],
    style_name: str,
    min_dist_um: float,
    max_dist_um: float,
    pixel_size_um: float,
    max_collect: int,
) -> pd.DataFrame:
    if phenotype_col not in region_df.columns:
        return pd.DataFrame()

    work = region_df.dropna(subset=REQUIRED_COORDS + [phenotype_col]).copy()
    if work.empty:
        return pd.DataFrame()

    phenos = work[phenotype_col].astype(str).to_numpy(copy=True)
    triad_set = {str(v) for v in triad}
    if not triad_set.issubset(set(phenos.tolist())):
        return pd.DataFrame()

    coords = work[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    min_dist_px = _um_to_px(float(min_dist_um), float(pixel_size_um))
    max_dist_px = _um_to_px(float(max_dist_um), float(pixel_size_um))
    adj = _build_adjacency(coords=coords, min_dist_px=float(min_dist_px), max_dist_px=float(max_dist_px))
    if sum(len(s) for s in adj) == 0:
        return pd.DataFrame()

    _, triads_local = _count_and_collect_triads_anchor(
        phenos=phenos,
        adj=adj,
        triad=triad,
        max_collect=int(max_collect),
    )
    if not triads_local:
        return pd.DataFrame()

    work_local = work.reset_index().rename(columns={"index": "global_row_index"})
    batch_value = str(work_local["__batch__"].iloc[0]) if "__batch__" in work_local.columns else "ALL"
    roi_value = str(work_local["__roi__"].iloc[0]) if "__roi__" in work_local.columns else "ALL"

    rows: list[dict[str, Any]] = []
    for a, b, c in triads_local:
        cell_a = work_local.iloc[int(a)]
        cell_b = work_local.iloc[int(b)]
        cell_c = work_local.iloc[int(c)]
        rows.append(
            {
                "batch": batch_value,
                "roi": roi_value,
                "style": str(style_name),
                "phenotype_column": str(phenotype_col),
                "triad_label": "::".join(triad),
                "pheno_a": str(triad[0]),
                "pheno_b": str(triad[1]),
                "pheno_c": str(triad[2]),
                "cell_a_global_index": int(cell_a["global_row_index"]),
                "cell_b_global_index": int(cell_b["global_row_index"]),
                "cell_c_global_index": int(cell_c["global_row_index"]),
                "cell_a_x": float(cell_a["x_centroid"]),
                "cell_a_y": float(cell_a["y_centroid"]),
                "cell_b_x": float(cell_b["x_centroid"]),
                "cell_b_y": float(cell_b["y_centroid"]),
                "cell_c_x": float(cell_c["x_centroid"]),
                "cell_c_y": float(cell_c["y_centroid"]),
                "triad_centroid_x": float((cell_a["x_centroid"] + cell_b["x_centroid"] + cell_c["x_centroid"]) / 3.0),
                "triad_centroid_y": float((cell_a["y_centroid"] + cell_b["y_centroid"] + cell_c["y_centroid"]) / 3.0),
            }
        )

    return pd.DataFrame(rows)


def _build_fuzzy_cloud(
    coords: np.ndarray,
    values: np.ndarray,
    buffer_distance_px: float,
    bins: int,
    sigma_scale: float,
    density_weight: bool,
    fixed_extent: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    if coords.size == 0:
        return np.zeros((10, 10), dtype=float), [0, 1, 0, 1]

    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    v = np.asarray(values, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if finite.sum() == 0:
        return np.zeros((10, 10), dtype=float), [0, 1, 0, 1]

    x = x[finite]
    y = y[finite]
    v = v[finite]

    v_lo = np.nanpercentile(v, 1)
    v_hi = np.nanpercentile(v, 99)
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        v_scaled = np.ones_like(v)
    else:
        v_scaled = np.clip((v - v_lo) / (v_hi - v_lo), 0.0, 1.0)

    if fixed_extent is not None:
        x_min, x_max, y_min, y_max = [float(v) for v in fixed_extent]
    else:
        margin = max(float(buffer_distance_px), 1.0)
        x_min, x_max = float(np.nanmin(x) - margin), float(np.nanmax(x) + margin)
        y_min, y_max = float(np.nanmin(y) - margin), float(np.nanmax(y) + margin)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    hist_count, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=[int(bins), int(bins)],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    hist_value, _, _ = np.histogram2d(
        x,
        y,
        bins=[int(bins), int(bins)],
        range=[[x_min, x_max], [y_min, y_max]],
        weights=v_scaled,
    )

    bin_w_x = (x_max - x_min) / max(int(bins), 1)
    bin_w_y = (y_max - y_min) / max(int(bins), 1)
    bin_w = max((bin_w_x + bin_w_y) / 2.0, 1e-6)
    sigma_bins = max((float(buffer_distance_px) / bin_w) * float(sigma_scale), 0.5)

    count_blur = gaussian_filter(hist_count, sigma=sigma_bins, mode="constant")
    value_blur = gaussian_filter(hist_value, sigma=sigma_bins, mode="constant")

    mean_expr = np.divide(
        value_blur,
        count_blur,
        out=np.zeros_like(value_blur, dtype=float),
        where=count_blur > 1e-12,
    )

    if density_weight:
        max_count = float(np.nanmax(count_blur)) if np.size(count_blur) else 0.0
        if max_count > 0:
            density = count_blur / max_count
            field = mean_expr * density
        else:
            field = mean_expr
    else:
        field = mean_expr

    # Keep all non-empty bins and rely on per-pixel alpha for soft fading edges.
    field[count_blur <= 1e-12] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return field, extent


def _hex_to_rgb255(hex_color: str) -> list[int]:
    text = str(hex_color).strip().lstrip("#")
    if len(text) != 6:
        return [0, 255, 128]
    try:
        return [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)]
    except Exception:
        return [0, 255, 128]


def _mask_cloud_to_geometry(field: np.ndarray, extent: list[float], geom) -> np.ndarray:
    if geom is None or field.size == 0 or Point is None:
        return field

    x0, x1, y0, y1 = [float(v) for v in extent]
    nx, ny = field.shape
    if nx <= 0 or ny <= 0:
        return field

    x_centers = np.linspace(x0, x1, nx, endpoint=False) + (x1 - x0) / (2 * nx)
    y_centers = np.linspace(y0, y1, ny, endpoint=False) + (y1 - y0) / (2 * ny)
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    inside = np.array(
        [geom.covers(Point(float(px), float(py))) for px, py in zip(xx.ravel(), yy.ravel())],
        dtype=bool,
    ).reshape(field.shape)

    out = field.copy()
    out[~inside] = np.nan
    return out


def _plot_triad_virtual_tissue(
    run_df: pd.DataFrame,
    triad_df: pd.DataFrame,
    batch_value: str,
    roi_value: str,
    pixel_size_um: float,
    method: str,
    alpha: float,
    use_median: bool,
    percentile: float,
    min_hole_size: float,
    triad_view_mode: str = "points",
    triad_color: str = "#d62728",
    triad_point_size: float = 24.0,
    triad_alpha: float = 0.85,
    cloud_bins: int = 400,
    cloud_sigma_scale: float = 1.0,
    cloud_density_weight: bool = True,
    cloud_cmap: str = "inferno",
    cloud_clip_to_tissue: bool = True,
    show_cell_context: bool = True,
    annotation_configs: list[dict[str, str]] = None,
    annotation_alpha: float = 0.22,
    
    # New visual parameters
    show_markers: bool = False,
    composite_image = None,
    full_hw: tuple[int, int] = (1, 1),
    markers_opacity: float = 1.0,
    
    show_geojson_annotations: bool = False,
    active_geojson_records: list[dict[str, Any]] = None,
    geojson_annotation_opacity: float = 0.5,
    geojson_annotation_line_width: float = 1.0,
    
    show_triad_points: bool = True,
    triad_point_color_mode: str = "Single Color", # "Single Color", "By Phenotype (A/B/C)", "By Style"
    color_a: str = "#1f77b4",
    color_b: str = "#d62728",
    color_c: str = "#2ca02c",
    ph_a: str = "A",
    ph_b: str = "B",
    ph_c: str = "C",
    
    show_triad_lines: bool = False,
    triad_line_color_mode: str = "Single Color",
    triad_line_color: str = "#d62728",
    triad_line_width: float = 1.0,
    triad_line_alpha: float = 0.7,
    
    show_triad_density_cloud: bool = False,
    cloud_color_mode: str = "Standard Colormap",
    cloud_color_from: str = "#0000ff",
    cloud_color_to: str = "#ff0000",
    
    show_contours: bool = False,
    contour_levels: int = 8,
    contour_line_width: float = 1.0,
    contour_color_mode: str = "Match colormap",
    contour_color: str = "#ffffff",
    
    bg_color_mode: str = "White",
    bg_custom_hex: str = "#ffffff",
    
    crop_to_cells: bool = True,
    crop_x_min: float = 0.0,
    crop_x_max: float = 4000.0,
    crop_y_min: float = 0.0,
    crop_y_max: float = 4000.0,
    crop_padding: float = 100.0,
    
    layer_order: list[str] = None,
):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from shapely.geometry import Polygon, MultiPolygon

    def plot_shapely_geometry(ax, geom, fill_color, border_color, alpha, linewidth, zorder=1, label=None):
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        if geom is None or geom.is_empty:
            return
        if isinstance(geom, Polygon):
            verts = []
            codes = []
            
            # Exterior ring
            ext_coords = list(geom.exterior.coords)
            verts.extend(ext_coords)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ext_coords) - 2) + [Path.CLOSEPOLY])
            
            # Interior rings (holes)
            for interior in geom.interiors:
                int_coords = list(interior.coords)
                verts.extend(int_coords)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(int_coords) - 2) + [Path.CLOSEPOLY])
                
            path = Path(verts, codes)
            patch = PathPatch(
                path,
                facecolor=to_rgba(fill_color, alpha),
                edgecolor=border_color if linewidth > 0 else "none",
                linewidth=linewidth,
                zorder=zorder,
                label=label,
            )
            ax.add_patch(patch)
        elif isinstance(geom, MultiPolygon):
            first = True
            for poly in geom.geoms:
                plot_shapely_geometry(
                    ax, poly, fill_color, border_color, alpha, linewidth, zorder=zorder,
                    label=label if first else None
                )
                first = False
        elif hasattr(geom, "geoms"):
            first = True
            for part in geom.geoms:
                plot_shapely_geometry(
                    ax, part, fill_color, border_color, alpha, linewidth, zorder=zorder,
                    label=label if first else None
                )
                first = False

    cells = run_df[(run_df["__batch__"].astype(str) == str(batch_value)) & (run_df["__roi__"].astype(str) == str(roi_value))].copy()
    triads = triad_df[(triad_df["batch"].astype(str) == str(batch_value)) & (triad_df["roi"].astype(str) == str(roi_value))].copy()
    if cells.empty:
        return None, None

    coords = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    roi_x_min = float(np.nanmin(coords[:, 0]))
    roi_x_max = float(np.nanmax(coords[:, 0]))
    roi_y_min = float(np.nanmin(coords[:, 1]))
    roi_y_max = float(np.nanmax(coords[:, 1]))

    buffer_distance = _compute_buffer_distance(coords, use_median=bool(use_median), percentile=float(percentile))
    if not np.isfinite(buffer_distance) or float(buffer_distance) <= 0:
        buffer_distance = 20.0

    polygon = _build_region_geometry(
        coords=coords,
        method=str(method),
        alpha=float(alpha),
        buffer_distance=float(buffer_distance),
        min_hole_size=float(min_hole_size),
    )

    if polygon is not None:
        area_mm2 = float(polygon.area) * ((float(pixel_size_um) / 1000.0) ** 2)
    else:
        area_mm2 = np.nan
    n_triads = int(len(triads))
    triad_density = (float(n_triads) / area_mm2) if area_mm2 > 0 else np.nan

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Set background color
    if bg_color_mode == "Black":
        bg_color_hex = "#000000"
    elif bg_color_mode == "White":
        bg_color_hex = "#ffffff"
    elif bg_color_mode == "Transparent":
        bg_color_hex = "Transparent"
    else:
        bg_color_hex = bg_custom_hex

    if bg_color_hex == "Transparent":
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        fig.patch.set_facecolor('none')
        ax.patch.set_facecolor('none')
    else:
        fig.patch.set_facecolor(bg_color_hex)
        ax.patch.set_facecolor(bg_color_hex)

    # Crop limits
    if crop_to_cells:
        x_min = roi_x_min - crop_padding
        x_max = roi_x_max + crop_padding
        y_min = roi_y_min - crop_padding
        y_max = roi_y_max + crop_padding
    else:
        x_min = crop_x_min
        x_max = crop_x_max
        y_min = crop_y_min
        y_max = crop_y_max

    # Virtual tissue boundary drawing helper
    def draw_virtual_tissue_layer(zorder=1):
        if polygon is None:
            return
        plot_shapely_geometry(
            ax,
            polygon,
            fill_color="#2ca02c",
            border_color="k",
            alpha=0.12,
            linewidth=1.2,
            zorder=zorder,
            label="virtual tissue",
        )

    # Old annotations drawing helper (kept for fallback compatibility)
    def draw_old_annotations_layer(zorder=1):
        if not annotation_configs:
            return
        for cfg in annotation_configs:
            col = str(cfg.get("column", ""))
            target_val = str(cfg.get("value", ""))
            color_hex = str(cfg.get("color_hex", "#1f77b4"))
            if not col or col not in cells.columns:
                continue
            ann_df = cells[cells[col].astype(str) == target_val].copy().dropna(subset=["x_centroid", "y_centroid"])
            if len(ann_df) < 1:
                continue
            ann_coords = ann_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
            ann_geom = None
            if len(ann_coords) >= 3:
                ann_geom = _build_region_geometry(
                    coords=ann_coords,
                    method=str(method),
                    alpha=float(alpha),
                    buffer_distance=float(buffer_distance) * 0.8,
                    min_hole_size=float(min_hole_size),
                )
            if ann_geom is not None and not ann_geom.is_empty:
                plot_shapely_geometry(ax, ann_geom, fill_color=color_hex, border_color=color_hex, alpha=annotation_alpha, linewidth=1.0, zorder=zorder)

    # Compute Triad Density cloud if needed
    tri_field, tri_extent = None, None
    vmin, vmax = None, None
    alpha_map = None
    density_cmap = None

    if (show_triad_density_cloud or show_contours) and not triads.empty:
        tri_coords = triads[["triad_centroid_x", "triad_centroid_y"]].to_numpy(dtype=float)
        tri_values = np.ones(len(tri_coords), dtype=float)
        tri_field, tri_extent = _build_fuzzy_cloud(
            coords=tri_coords,
            values=tri_values,
            buffer_distance_px=float(buffer_distance),
            bins=int(cloud_bins),
            sigma_scale=float(cloud_sigma_scale),
            density_weight=bool(cloud_density_weight),
            fixed_extent=(
                float(roi_x_min),
                float(roi_x_max),
                float(roi_y_min),
                float(roi_y_max),
            ),
        )
        if bool(cloud_clip_to_tissue) and polygon is not None:
            tri_field = _mask_cloud_to_geometry(tri_field, tri_extent, polygon)

        finite_vals = tri_field[np.isfinite(tri_field)]
        if finite_vals.size > 0:
            vmin = float(np.nanpercentile(finite_vals, 5))
            vmax = float(np.nanpercentile(finite_vals, 99.5))
            if vmin is not None and vmax is not None and vmax > vmin:
                norm_field = np.clip((tri_field - vmin) / (vmax - vmin), 0.0, 1.0)
                alpha_map = np.power(np.nan_to_num(norm_field, nan=0.0), 1.25) * float(triad_alpha)
                alpha_map = np.clip(alpha_map, 0.0, 1.0)

        if cloud_color_mode == "Custom gradient (From - To)":
            density_cmap = LinearSegmentedColormap.from_list("custom_cmap", [cloud_color_from, cloud_color_to])
        else:
            try:
                density_cmap = plt.get_cmap(cloud_cmap).copy()
                density_cmap.set_bad((0, 0, 0, 0))
                density_cmap.set_under((0, 0, 0, 0))
            except Exception:
                density_cmap = plt.get_cmap(cloud_cmap)

    # Default layer order if not specified
    if not layer_order:
        layer_order = [
            "Markers",
            "Cell Context",
            "Tissue Annotations",
            "Virtual Tissue",
            "Triad Density Cloud",
            "Triad Contours",
            "Triad Lines/Triangles",
            "Triad Points",
        ]

    # Render layers in exact sequence order (from bottom to top)
    for idx, layer in enumerate(layer_order):
        base_zorder = idx * 10
        if layer == "Markers":
            if show_markers and composite_image is not None:
                full_h, full_w = full_hw
                ax.imshow(composite_image, extent=(0, full_w, full_h, 0), alpha=markers_opacity, zorder=base_zorder)

        elif layer == "Cell Context":
            if show_cell_context:
                ax.scatter(cells["x_centroid"], cells["y_centroid"], s=2, c="#c7c7c7", alpha=0.35, edgecolors="none", label="cells", zorder=base_zorder)

        elif layer == "Tissue Annotations":
            if show_geojson_annotations and active_geojson_records:
                for record in active_geojson_records:
                    geom = record.get("geometry")
                    color = record.get("color_hex", "#1f77b4")
                    plot_shapely_geometry(
                        ax,
                        geom,
                        fill_color=color,
                        border_color=color,
                        alpha=geojson_annotation_opacity,
                        linewidth=geojson_annotation_line_width,
                        zorder=base_zorder,
                    )
            elif annotation_configs:
                draw_old_annotations_layer(zorder=base_zorder)

        elif layer == "Virtual Tissue":
            draw_virtual_tissue_layer(zorder=base_zorder)

        elif layer == "Triad Points":
            if show_triad_points and not triads.empty:
                if triad_point_color_mode == "Single Color":
                    ax.scatter(
                        triads["triad_centroid_x"],
                        triads["triad_centroid_y"],
                        s=float(triad_point_size),
                        c=str(triad_color),
                        alpha=float(triad_alpha),
                        edgecolors="white",
                        linewidths=0.4,
                        label="triad centroids",
                        zorder=base_zorder,
                    )
                elif triad_point_color_mode == "By Phenotype (A/B/C)":
                    ax.scatter(triads["cell_a_x"], triads["cell_a_y"], s=float(triad_point_size), c=color_a, alpha=float(triad_alpha), edgecolors="white", linewidths=0.4, label=ph_a, zorder=base_zorder)
                    ax.scatter(triads["cell_b_x"], triads["cell_b_y"], s=float(triad_point_size), c=color_b, alpha=float(triad_alpha), edgecolors="white", linewidths=0.4, label=ph_b, zorder=base_zorder)
                    ax.scatter(triads["cell_c_x"], triads["cell_c_y"], s=float(triad_point_size), c=color_c, alpha=float(triad_alpha), edgecolors="white", linewidths=0.4, label=ph_c, zorder=base_zorder)
                elif triad_point_color_mode == "By Style" and "style" in triads.columns:
                    local_mask = triads["style"] == "local"
                    neigh_mask = triads["style"] == "neighborhood"
                    if local_mask.any():
                        ax.scatter(
                            triads.loc[local_mask, "triad_centroid_x"],
                            triads.loc[local_mask, "triad_centroid_y"],
                            s=float(triad_point_size),
                            c="#1f77b4",
                            alpha=float(triad_alpha),
                            edgecolors="white",
                            linewidths=0.4,
                            label="Local triads",
                            zorder=base_zorder,
                        )
                    if neigh_mask.any():
                        ax.scatter(
                            triads.loc[neigh_mask, "triad_centroid_x"],
                            triads.loc[neigh_mask, "triad_centroid_y"],
                            s=float(triad_point_size),
                            c="#d62728",
                            alpha=float(triad_alpha),
                            edgecolors="white",
                            linewidths=0.4,
                            label="Annulus triads",
                            zorder=base_zorder,
                        )

        elif layer == "Triad Lines/Triangles":
            if show_triad_lines and not triads.empty:
                lines = []
                line_colors = []
                for _, row in triads.iterrows():
                    ax_x, ax_y = float(row["cell_a_x"]), float(row["cell_a_y"])
                    bx, by = float(row["cell_b_x"]), float(row["cell_b_y"])
                    cx, cy = float(row["cell_c_x"]), float(row["cell_c_y"])
                    lines.append([(ax_x, ax_y), (bx, by)])
                    lines.append([(bx, by), (cx, cy)])
                    lines.append([(cx, cy), (ax_x, ax_y)])
                    if triad_line_color_mode == "By Style" and "style" in row:
                        sc = "#1f77b4" if row["style"] == "local" else "#d62728"
                        line_colors.extend([sc, sc, sc])
                    else:
                        line_colors.extend([triad_line_color, triad_line_color, triad_line_color])
                lc = LineCollection(lines, colors=line_colors, linewidths=triad_line_width, alpha=triad_line_alpha, zorder=base_zorder)
                ax.add_collection(lc)

        elif layer == "Triad Density Cloud":
            if show_triad_density_cloud and not triads.empty and tri_field is not None:
                ax.imshow(
                    np.ma.masked_invalid(tri_field.T),
                    extent=(float(tri_extent[0]), float(tri_extent[1]), float(tri_extent[2]), float(tri_extent[3])),
                    origin="lower",
                    cmap=density_cmap,
                    alpha=alpha_map.T if alpha_map is not None else float(triad_alpha),
                    aspect="auto",
                    vmin=vmin,
                    vmax=vmax,
                    zorder=base_zorder,
                )

        elif layer == "Triad Contours":
            if show_contours and not triads.empty and tri_field is not None:
                c_colors = None
                if contour_color_mode == "Single Color":
                    c_colors = contour_color
                finite_vals = tri_field[np.isfinite(tri_field)]
                if finite_vals.size > 0:
                    c_vmin = float(np.nanmin(finite_vals))
                    c_vmax = float(np.nanmax(finite_vals))
                    c_levels = np.linspace(c_vmin + (c_vmax - c_vmin) * 0.05, c_vmax, max(2, contour_levels))
                    if len(c_levels) >= 2 and (c_levels[-1] > c_levels[0]):
                        try:
                            ax.contour(
                                tri_field.T,
                                extent=(float(tri_extent[0]), float(tri_extent[1]), float(tri_extent[2]), float(tri_extent[3])),
                                levels=c_levels,
                                cmap=density_cmap if contour_color_mode == "Match colormap" else None,
                                colors=c_colors,
                                linewidths=contour_line_width,
                                zorder=base_zorder,
                            )
                        except Exception as exc:
                            _log_console(f"Error drawing contours: {exc}", level="error")

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min) # Inverted Y axis
    ax.grid(False)
    ax.axis("off")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()

    metrics = {
        "batch": str(batch_value),
        "roi": str(roi_value),
        "n_cells": int(len(cells)),
        "n_triads": int(n_triads),
        "area_mm2": float(area_mm2),
        "triad_density_per_mm2": float(triad_density) if np.isfinite(triad_density) else np.nan,
        "buffer_distance_px": float(buffer_distance),
        "method": str(method),
        "buffer_mode": "median" if bool(use_median) else f"percentile_{float(percentile):.1f}",
        "triad_view_mode": triad_view_mode,
        "requested_annotations": "",
        "rendered_annotations": "",
        "n_annotations": len(active_geojson_records) if active_geojson_records else 0,
        "n_overlap_regions": 0,
    }
    return fig, metrics



def _extract_prefix(text: str, regex_pattern: str) -> str:
    try:
        m = re.search(regex_pattern, str(text))
        if m:
            return str(m.group(1))
    except Exception:
        pass
    return str(text)


def _index_files(directory: str, exts: tuple[str, ...], prefix_regex: str) -> list[dict[str, str]]:
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []
    out: list[dict[str, str]] = []
    for p in sorted(path.iterdir()):
        if not p.is_file():
            continue
        lname = p.name.lower()
        if not any(lname.endswith(ext.lower()) for ext in exts):
            continue
        stem = p.stem
        if stem.endswith(".ome"):
            stem = stem[:-4]
        out.append({"path": str(p), "name": p.name, "prefix": _extract_prefix(stem, prefix_regex)})
    return out


def _best_prefix_match(query_prefix: str, entries: list[dict[str, str]]) -> dict[str, str] | None:
    if not entries:
        return None
    query = str(query_prefix)
    exact = [e for e in entries if e.get("prefix", "") == query]
    if exact:
        return exact[0]
    contain = [e for e in entries if query in e.get("prefix", "") or e.get("prefix", "") in query]
    if contain:
        contain = sorted(contain, key=lambda e: abs(len(e.get("prefix", "")) - len(query)))
        return contain[0]
    return None


def _parse_channel_names_from_ome_xml(ome_xml: str) -> list[str]:
    names: list[str] = []
    if not ome_xml:
        return names
    try:
        root = ET.fromstring(ome_xml)
        for elem in root.iter():
            if "Channel" in elem.tag:
                name = elem.attrib.get("Name")
                names.append(str(name) if name else f"Channel_{len(names)}")
    except Exception:
        return []
    return names


def _normalize_name(text: str) -> str:
    cleaned = str(text).strip().lower()
    for suffix in ["_corr", "_norm", "_log"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def _resolve_marker_indices(channel_names: list[str], markers: list[str]) -> dict[str, int]:
    idx_by_name = {str(name): i for i, name in enumerate(channel_names)}
    idx_by_norm = {_normalize_name(str(name)): i for i, name in enumerate(channel_names)}
    out: dict[str, int] = {}
    for marker in markers:
        if marker in idx_by_name:
            out[str(marker)] = int(idx_by_name[marker])
            continue
        norm = _normalize_name(marker)
        if norm in idx_by_norm:
            out[str(marker)] = int(idx_by_norm[norm])
    return out


def _load_geojson_polygons(geojson_path: str):
    if shape is None:
        return []
    try:
        with open(geojson_path, "r") as f:
            gj = json.load(f)
    except Exception:
        return []
    feats = gj.get("features", [])
    polygons = []
    for feat in feats:
        geom = feat.get("geometry")
        if not geom:
            continue
        gtype = geom.get("type")
        if gtype not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            polygons.append(shape(geom))
        except Exception:
            continue
    return polygons


def _load_tiff_crops(image_path: str, channel_indices: list[int], x0: int, x1: int, y0: int, y1: int):
    if tifffile is None:
        raise RuntimeError("tifffile is required for triad image visualization.")

    crops: dict[int, np.ndarray] = {}
    channel_names: list[str] = []
    with tifffile.TiffFile(image_path) as tif:
        channel_names = _parse_channel_names_from_ome_xml(getattr(tif, "ome_metadata", "") or "")

        if len(tif.pages) >= max(channel_indices, default=-1) + 1:
            for idx in sorted(set(channel_indices)):
                if idx < 0 or idx >= len(tif.pages):
                    continue
                page = tif.pages[int(idx)]
                arr = page.asarray()
                if arr.ndim < 2:
                    continue
                yy1 = min(int(y1), int(arr.shape[-2]))
                xx1 = min(int(x1), int(arr.shape[-1]))
                yy0 = max(0, int(y0))
                xx0 = max(0, int(x0))
                crops[int(idx)] = arr[yy0:yy1, xx0:xx1]
        else:
            full = tif.asarray()
            if full.ndim == 3:
                n_channels = full.shape[0]
                for idx in sorted(set(channel_indices)):
                    if idx < 0 or idx >= n_channels:
                        continue
                    yy1 = min(int(y1), int(full.shape[-2]))
                    xx1 = min(int(x1), int(full.shape[-1]))
                    yy0 = max(0, int(y0))
                    xx0 = max(0, int(x0))
                    crops[int(idx)] = full[int(idx), yy0:yy1, xx0:xx1]

    return crops, channel_names


def _plot_triad_marker_figure(
    marker_to_crop: dict[str, np.ndarray],
    all_polygons,
    triad_points_global: dict[str, tuple[float, float]],
    crop_origin: tuple[int, int],
    triad_colors: dict[str, str],
    marker_colors: dict[str, str],
    show_surrounding_seg: bool,
    zoom_factor: float,
):
    if Point is None:
        raise RuntimeError("shapely is required for segmentation overlay in triad image visualization.")

    marker_names = list(marker_to_crop.keys())
    if not marker_names:
        return None

    first_img = marker_to_crop[marker_names[0]]
    if first_img.ndim != 2:
        return None
    h, w = first_img.shape
    x0, y0 = crop_origin

    # Identify target polygons for A/B/C by containment / nearest
    triad_polys: dict[str, Any] = {}
    for key, (gx, gy) in triad_points_global.items():
        pt = Point(float(gx), float(gy))
        target = None
        min_dist = np.inf
        for poly in all_polygons:
            try:
                if poly.contains(pt):
                    target = poly
                    break
                d = float(poly.distance(pt))
                if d < min_dist:
                    min_dist = d
                    target = poly
            except Exception:
                continue
        if target is not None:
            triad_polys[key] = target

    n_tiles = len(marker_names) + 1
    fig, axes = plt.subplots(1, n_tiles, figsize=(3.2 * n_tiles, 3.4), squeeze=False)
    axes = axes.flatten()

    cx = float(w) / 2.0
    cy = float(h) / 2.0
    zf = max(float(zoom_factor), 1.0)
    half_w = max(8.0, float(w) / (2.0 * zf))
    half_h = max(8.0, float(h) / (2.0 * zf))
    x_lim = (cx - half_w, cx + half_w)
    y_lim = (cy + half_h, cy - half_h)

    # Per-marker tiles (true colorized rendering per marker)
    for i, marker in enumerate(marker_names):
        ax = axes[i]
        img = marker_to_crop[marker].astype(float)
        if img.size == 0:
            ax.axis("off")
            continue
        lo, hi = np.percentile(img, [1, 99.5]) if np.size(img) > 20 else (float(np.min(img)), float(np.max(img)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(img)), float(np.max(img) + 1e-8)
        norm = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        norm = np.power(norm, 0.85)
        marker_color = marker_colors.get(marker, "#ffffff")
        try:
            rgb = np.asarray(to_rgb(marker_color), dtype=float)
        except Exception:
            rgb = np.asarray((1.0, 1.0, 1.0), dtype=float)
        tinted = np.zeros((h, w, 3), dtype=float)
        tinted[:, :, 0] = norm * rgb[0]
        tinted[:, :, 1] = norm * rgb[1]
        tinted[:, :, 2] = norm * rgb[2]
        ax.imshow(np.clip(tinted, 0.0, 1.0))

        if show_surrounding_seg:
            for poly in all_polygons:
                try:
                    minx, miny, maxx, maxy = poly.bounds
                    if maxx < x0 or minx > (x0 + w) or maxy < y0 or miny > (y0 + h):
                        continue
                    if poly.geom_type == "Polygon":
                        x, y = poly.exterior.xy
                        ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color="black", linewidth=0.5, alpha=0.35)
                    elif poly.geom_type == "MultiPolygon":
                        for p in list(getattr(poly, "geoms", [])):
                            x, y = p.exterior.xy
                            ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color="black", linewidth=0.5, alpha=0.35)
                except Exception:
                    continue

        for key, poly in triad_polys.items():
            col = triad_colors.get(key, "#ff0000")
            try:
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                    ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color=col, linewidth=2.0)
                elif poly.geom_type == "MultiPolygon":
                    for p in list(getattr(poly, "geoms", [])):
                        x, y = p.exterior.xy
                        ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color=col, linewidth=2.0)
            except Exception:
                continue

        ax.set_title(marker, fontsize=9, fontweight="bold")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.axis("off")

    # Overlay tile
    overlay_ax = axes[-1]
    overlay = np.zeros((h, w, 3), dtype=float)
    for marker in marker_names:
        img = marker_to_crop[marker].astype(float)
        if img.size == 0:
            continue
        lo, hi = np.percentile(img, [1, 99.5]) if np.size(img) > 20 else (float(np.min(img)), float(np.max(img)))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        norm = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        norm = np.power(norm, 0.9)
        marker_color = marker_colors.get(marker, "#ffffff")
        try:
            rgb = np.asarray(to_rgb(marker_color), dtype=float)
        except Exception:
            rgb = np.asarray((1.0, 1.0, 1.0), dtype=float)
        weighted = norm[:, :, None] * rgb[None, None, :]
        overlay = np.maximum(overlay, weighted)
    overlay = np.clip(overlay * 1.08, 0.0, 1.0)
    overlay_ax.imshow(overlay)

    if show_surrounding_seg:
        for poly in all_polygons:
            try:
                minx, miny, maxx, maxy = poly.bounds
                if maxx < x0 or minx > (x0 + w) or maxy < y0 or miny > (y0 + h):
                    continue
                if poly.geom_type == "Polygon":
                    x, y = poly.exterior.xy
                    overlay_ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color="black", linewidth=0.5, alpha=0.35)
                elif poly.geom_type == "MultiPolygon":
                    for p in list(getattr(poly, "geoms", [])):
                        x, y = p.exterior.xy
                        overlay_ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color="black", linewidth=0.5, alpha=0.35)
            except Exception:
                continue

    for key, poly in triad_polys.items():
        col = triad_colors.get(key, "#ff0000")
        try:
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                overlay_ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color=col, linewidth=2.2)
            elif poly.geom_type == "MultiPolygon":
                for p in list(getattr(poly, "geoms", [])):
                    x, y = p.exterior.xy
                    overlay_ax.plot(np.asarray(x) - x0, np.asarray(y) - y0, color=col, linewidth=2.2)
        except Exception:
            continue

    overlay_ax.set_title("Overlay", fontsize=9, fontweight="bold")
    overlay_ax.set_xlim(*x_lim)
    overlay_ax.set_ylim(*y_lim)
    overlay_ax.axis("off")
    plt.tight_layout()
    return fig


def _get_unique_values(df: pd.DataFrame, col: str) -> list[str]:
    cache_key = f"tab8_triad_unique_vals_{col}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    vals = _safe_unique(df[col])
    st.session_state[cache_key] = vals
    return vals


@st.cache_data(show_spinner=False)
def _compose_preview_image_cached(
    image_path_str: str,
    mtime: float,
    channel_names: list[str],
    color_map: dict[str, str],
    selected_markers: list[str],
    max_dim: int | None,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    from ..tab9_visualization.actual_tissue import _compose_preview_image
    img, hw = _compose_preview_image(
        image_path=Path(image_path_str),
        channel_names=channel_names,
        color_map=color_map,
        selected_markers=selected_markers,
        max_dim=max_dim,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    return np.array(img), hw


@st.cache_data(show_spinner=False)
def _load_geojson_groups_cached(path_str: str, mtime: float) -> list[dict[str, Any]]:
    from ..tab9_visualization.actual_tissue import _load_geojson_groups
    return _load_geojson_groups(Path(path_str))


@st.cache_data(show_spinner=False)
def _get_geojson_labels_cached(path_str: str, mtime: float) -> list[str]:
    from ..tab9_visualization.actual_tissue import _load_geojson_groups
    recs = _load_geojson_groups(Path(path_str))
    return sorted(list({rec["label"] for rec in recs}))


def _default_label_color(label: str) -> str:
    palette = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
    ]
    return palette[abs(hash(label)) % len(palette)]


def _render_panel_virtual_tissue(
    project_root: Path,
    run_df: pd.DataFrame,
    phenotype_candidates: list[str],
    triad_fallback_names: tuple[str, str, str],
    unified_csv_path: str | None = None,
):
    import os
    from pathlib import Path
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    
    # Import tab9_visualization helpers dynamically
    from ..tab9_visualization.actual_tissue import (
        _get_channel_info_cached,
        _list_files_cached,
        _best_match_file,
        _hex_from_rgb,
    )
    from spacemap_gui.persistence import load_directory_settings

    # Resolve visual settings directory defaults
    global_input_dir = st.session_state.get("global_spatial_dir")
    stored_viz = load_directory_settings(global_input_dir, section_key="tab9_viz") if global_input_dir else {}
    viz_roi_dir = stored_viz.get("viz_roi_dir", str(project_root / "outputs" / "registration" / "roi"))
    viz_seg_dir = stored_viz.get("viz_seg_dir", str(project_root / "outputs" / "segmentation" / "cellpose"))
    viz_ann_dir = stored_viz.get("viz_ann_dir", str(project_root / "outputs" / "segmentation" / "cellpose"))

    # Default Session States
    st.session_state.setdefault("tab8_triad_p2_show_markers", False)
    st.session_state.setdefault("tab8_triad_p2_roi_image_dir", viz_roi_dir)
    st.session_state.setdefault("tab8_triad_p2_marker_low_pct", 1.0)
    st.session_state.setdefault("tab8_triad_p2_marker_high_pct", 99.0)
    st.session_state.setdefault("tab8_triad_p2_markers_opacity", 1.0)

    st.session_state.setdefault("tab8_triad_p2_show_geojson_annotations", False)
    st.session_state.setdefault("tab8_triad_p2_annotation_dir", viz_ann_dir)
    st.session_state.setdefault("tab8_triad_p2_annotation_opacity", 0.35)
    st.session_state.setdefault("tab8_triad_p2_annotation_line_width", 1.0)

    st.session_state.setdefault("tab8_triad_p2_show_triad_points", True)
    st.session_state.setdefault("tab8_triad_p2_triad_point_color_mode", "Single Color")
    st.session_state.setdefault("tab8_triad_p2_triad_color", "#d62728")
    st.session_state.setdefault("tab8_triad_p2_color_a", "#1f77b4")
    st.session_state.setdefault("tab8_triad_p2_color_b", "#ff7f0e")
    st.session_state.setdefault("tab8_triad_p2_color_c", "#2ca02c")
    st.session_state.setdefault("tab8_triad_p2_triad_alpha", 0.85)
    st.session_state.setdefault("tab8_triad_p2_triad_size", 24)
    st.session_state.setdefault("tab8_triad_p2_show_cells", True)

    st.session_state.setdefault("tab8_triad_p2_show_triad_lines", False)
    st.session_state.setdefault("tab8_triad_p2_triad_line_color_mode", "Single Color")
    st.session_state.setdefault("tab8_triad_p2_triad_line_color", "#d62728")
    st.session_state.setdefault("tab8_triad_p2_triad_line_width", 1.0)
    st.session_state.setdefault("tab8_triad_p2_triad_line_alpha", 0.70)

    st.session_state.setdefault("tab8_triad_p2_show_triad_density_cloud", False)
    st.session_state.setdefault("tab8_triad_p2_cloud_color_mode", "Standard Colormap")
    st.session_state.setdefault("tab8_triad_p2_cloud_color_from", "#0000ff")
    st.session_state.setdefault("tab8_triad_p2_cloud_color_to", "#ff0000")
    st.session_state.setdefault("tab8_triad_p2_cloud_bins", 400)
    st.session_state.setdefault("tab8_triad_p2_cloud_sigma", 1.0)
    st.session_state.setdefault("tab8_triad_p2_cloud_density", True)
    st.session_state.setdefault("tab8_triad_p2_cloud_cmap", "inferno")
    st.session_state.setdefault("tab8_triad_p2_cloud_clip", True)

    st.session_state.setdefault("tab8_triad_p2_show_contours", False)
    st.session_state.setdefault("tab8_triad_p2_contour_levels", 8)
    st.session_state.setdefault("tab8_triad_p2_contour_line_width", 1.0)
    st.session_state.setdefault("tab8_triad_p2_contour_color_mode", "Match colormap")
    st.session_state.setdefault("tab8_triad_p2_contour_color", "#ffffff")

    st.session_state.setdefault("tab8_triad_p2_bg_color_mode", "White")
    st.session_state.setdefault("tab8_triad_p2_bg_custom_hex", "#ffffff")
    st.session_state.setdefault("tab8_triad_p2_crop_to_cells", True)
    st.session_state.setdefault("tab8_triad_p2_crop_padding", 100.0)

    st.session_state.setdefault("tab8_triad_p2_dpi", 300)
    st.session_state.setdefault("tab8_triad_p2_export_format", "PNG")
    st.session_state.setdefault("tab8_triad_p2_show_legend_triad", True)

    with st.expander("**Panel 3 · Triad Spatial Map on the Virtual Tissue**", expanded=False):
        st.info(
            "**How it works & What it does:** Panel 3 builds a virtual tissue map of the entire ROI or specimen. "
            "It maps the locations of phenotypes, plots the spatial network of detected triads, and overlays a continuous "
            "triad density cloud using Kernel Density Estimation (KDE) to visualize interaction contours.\n\n"
            "**Scientific Interpretation:** Helps identify macro-spatial patterns of triad clustering, such as enrichment "
            "along the tumor-stroma border (invasive margin) or clustering in functional domains like tertiary lymphoid structures (TLS)."
        )
        if run_df.empty:
            st.info("No data available for Panel 3.")
            return

        # Core calculations configuration
        st.markdown("#### 1. Phenotype & Distance Configurations")
        c_p1, c_p2 = st.columns(2)
        default_methods = [c for c in ["phenotype_leiden", "phenotype_som", "phenotype_threshold", "phenotype_scimap"] if c in phenotype_candidates] or phenotype_candidates[: min(4, len(phenotype_candidates))]
        _init_multiselect_state("tab8_triad_p2_methods", phenotype_candidates, default_methods)
        p3_methods = c_p1.multiselect(
            "Phenotyping method columns",
            options=phenotype_candidates,
            key="tab8_triad_p2_methods",
        )
        style_options = ["Direct-contact / local triads", "Neighborhood annulus triads"]
        _init_multiselect_state("tab8_triad_p2_styles", style_options, ["Direct-contact / local triads"])
        selected_styles = c_p2.multiselect(
            "Triad detection styles",
            options=style_options,
            key="tab8_triad_p2_styles",
        )

        p3_pool: set[str] = set()
        for mcol in p3_methods:
            if mcol in run_df.columns:
                p3_pool.update(run_df[mcol].dropna().astype(str).unique().tolist())
        p3_pheno_pool = sorted(list(p3_pool))
        if p3_methods and not p3_pheno_pool:
            st.warning("Selected methods have no phenotype values.")
            return

        t1, t2, t3 = st.columns(3)
        _init_selectbox_state("tab8_triad_p2_a", p3_pheno_pool, preferred=str(triad_fallback_names[0]), fallback=0)
        _init_selectbox_state("tab8_triad_p2_b", p3_pheno_pool, preferred=str(triad_fallback_names[1]), fallback=1)
        _init_selectbox_state("tab8_triad_p2_c", p3_pheno_pool, preferred=str(triad_fallback_names[2]), fallback=2)
        ph_a = t1.selectbox("Triad phenotype A", options=p3_pheno_pool, key="tab8_triad_p2_a")
        ph_b = t2.selectbox("Triad phenotype B", options=p3_pheno_pool, key="tab8_triad_p2_b")
        ph_c = t3.selectbox("Triad phenotype C", options=p3_pheno_pool, key="tab8_triad_p2_c")

        s1, s2, s3, s4 = st.columns(4)
        local_min_um = s1.number_input("Local min distance (µm)", min_value=0.0, max_value=5000.0, value=0.0, step=1.0, key="tab8_triad_p2_local_min_um")
        local_max_um = s2.number_input("Local max distance (µm)", min_value=0.1, max_value=5000.0, value=30.0, step=1.0, key="tab8_triad_p2_local_max_um")
        neigh_min_um = s3.number_input("Neighborhood min distance (µm)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0, key="tab8_triad_p2_neigh_min_um")
        neigh_max_um = s4.number_input("Neighborhood max distance (µm)", min_value=0.1, max_value=10000.0, value=400.0, step=1.0, key="tab8_triad_p2_neigh_max_um")

        style_configs: list[tuple[str, float, float]] = []
        if "Direct-contact / local triads" in selected_styles:
            style_configs.append(("local", float(local_min_um), float(local_max_um)))
        if "Neighborhood annulus triads" in selected_styles:
            style_configs.append(("neighborhood", float(neigh_min_um), float(neigh_max_um)))

        st.divider()

        # REORGANIZED 2-COLUMN LAYOUT
        col_controls, col_preview = st.columns([1, 1.2])

        with col_controls:
            run_render = st.button("Update Map Preview", key="tab8_triad_p2_update_btn", type="primary", width="stretch")
            
            control_tabs = st.tabs([
                "Data & Crop",
                "Virtual Tissue",
                "Markers",
                "Annotations",
                "Triad Style",
                "Output & Layers",
                "Themes & Presets"
            ])

            # 1. DATA & CROP TAB
            with control_tabs[0]:
                st.markdown("##### Region Selection")
                batch_vals = sorted(run_df["__batch__"].dropna().astype(str).unique().tolist())
                _init_selectbox_state("tab8_triad_p2_batch", batch_vals, fallback=0)
                selected_batch = st.selectbox("Batch", options=batch_vals, key="tab8_triad_p2_batch")
                roi_vals = sorted(
                    run_df[run_df["__batch__"].astype(str) == str(selected_batch)]["__roi__"].dropna().astype(str).unique().tolist()
                )
                _init_selectbox_state("tab8_triad_p2_roi", roi_vals, fallback=0)
                selected_roi = st.selectbox("ROI", options=roi_vals, key="tab8_triad_p2_roi")

                st.markdown("##### Crop Window & Resolution")
                crop_to_cells = st.checkbox("Crop to cells bounding box", value=True, key="tab8_triad_p2_crop_to_cells")
                crop_padding = st.number_input("Crop Padding (px)", min_value=0.0, value=150.0, step=10.0, key="tab8_triad_p2_crop_padding")

                # Get coordinate bounds of current batch/roi
                cells_in_roi = run_df[(run_df["__batch__"].astype(str) == str(selected_batch)) & (run_df["__roi__"].astype(str) == str(selected_roi))]
                if not cells_in_roi.empty:
                    c_x = cells_in_roi["x_centroid"].dropna().to_numpy()
                    c_y = cells_in_roi["y_centroid"].dropna().to_numpy()
                    default_xmin = float(c_x.min()) if c_x.size else 0.0
                    default_xmax = float(c_x.max()) if c_x.size else 4000.0
                    default_ymin = float(c_y.min()) if c_y.size else 0.0
                    default_ymax = float(c_y.max()) if c_y.size else 4000.0
                else:
                    default_xmin, default_xmax, default_ymin, default_ymax = 0.0, 4000.0, 0.0, 4000.0

                st.session_state.setdefault("tab8_triad_p2_crop_x_min", default_xmin)
                st.session_state.setdefault("tab8_triad_p2_crop_x_max", default_xmax)
                st.session_state.setdefault("tab8_triad_p2_crop_y_min", default_ymin)
                st.session_state.setdefault("tab8_triad_p2_crop_y_max", default_ymax)

                if not crop_to_cells:
                    st.caption("Manually adjust crop window:")
                    m_x1, m_x2 = st.columns(2)
                    crop_x_min = m_x1.number_input("X Min", value=st.session_state.tab8_triad_p2_crop_x_min, key="tab8_triad_p2_crop_x_min")
                    crop_x_max = m_x2.number_input("X Max", value=st.session_state.tab8_triad_p2_crop_x_max, key="tab8_triad_p2_crop_x_max")
                    m_y1, m_y2 = st.columns(2)
                    crop_y_min = m_y1.number_input("Y Min", value=st.session_state.tab8_triad_p2_crop_y_min, key="tab8_triad_p2_crop_y_min")
                    crop_y_max = m_y2.number_input("Y Max", value=st.session_state.tab8_triad_p2_crop_y_max, key="tab8_triad_p2_crop_y_max")
                else:
                    crop_x_min = default_xmin
                    crop_x_max = default_xmax
                    crop_y_min = default_ymin
                    crop_y_max = default_ymax

            # 2. VIRTUAL TISSUE TAB
            with control_tabs[1]:
                st.markdown("##### Virtual Tissue Bounds")
                d1, d2 = st.columns(2)
                pixel_size_um = d1.number_input("Pixel Size (µm)", min_value=0.01, value=0.5083, step=0.0001, format="%.4f", key="tab8_triad_p2_pixel_um")
                buffer_percentile = d2.number_input("Buffer Percentile", min_value=90.0, max_value=99.9, value=99.0, step=0.1, format="%.1f", key="tab8_triad_p2_percentile")
                
                use_median = st.checkbox("Use median distance", value=False, key="tab8_triad_p2_use_median")

                d4, d5, d6 = st.columns(3)
                p2_method_options = ["buffer_union", "alpha_shape"]
                _init_selectbox_state("tab8_triad_p2_method", p2_method_options, preferred="buffer_union", fallback=0)
                method = d4.selectbox("Method", options=p2_method_options, key="tab8_triad_p2_method")
                alpha = d5.number_input("Alpha (Concave)", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="tab8_triad_p2_alpha")
                min_hole_size = d6.number_input("Min Hole (px²)", min_value=0.0, value=10000.0, step=1000.0, key="tab8_triad_p2_min_hole")

            # 3. MARKERS TAB
            with control_tabs[2]:
                st.markdown("##### True Markers Overlay")
                show_markers = st.checkbox("Overlay true marker intensities", value=False, key="tab8_triad_p2_show_markers")
                roi_image_dir = st.text_input("ROI Image Directory (OME-TIFF)", value=st.session_state.tab8_triad_p2_roi_image_dir, key="tab8_triad_p2_roi_image_dir")
                
                composite_image = None
                full_hw = (1, 1)
                selected_markers = []
                marker_colors = {}
                marker_low_pct = 1.0
                marker_high_pct = 99.0
                markers_opacity = 1.0
                marker_res_mode = "Dynamic (Auto)"
                marker_target_res = 1600
                marker_manual_limit = 2000
                matched_image_path = None
                channel_names = []

                if show_markers:
                    if not roi_image_dir or not os.path.exists(roi_image_dir):
                        st.error("ROI image directory does not exist.")
                    else:
                        image_files = _list_files_cached(roi_image_dir, (".tif", ".tiff", ".ome.tif", ".ome.tiff"), recursive=True)
                        if not image_files:
                            st.warning("No OME-TIFF files found in the directory.")
                        else:
                            matched_image_path = _best_match_file(image_files, selected_batch, selected_roi)
                            if matched_image_path:
                                st.caption(f"Matched Image: `{Path(matched_image_path).name}`")
                                channel_names, channel_colors, _, _ = _get_channel_info_cached(
                                    matched_image_path, Path(matched_image_path).stat().st_mtime
                                )
                                channel_names = [str(n) for n in channel_names]
                                if channel_names:
                                    _init_multiselect_state(
                                        "tab8_triad_p2_markers",
                                        channel_names,
                                        channel_names[:min(3, len(channel_names))]
                                    )
                                    selected_markers = st.multiselect(
                                        "Select Markers to Display",
                                        options=channel_names,
                                        key="tab8_triad_p2_markers"
                                    )
                                    
                                    # Marker Color Picker Grid
                                    if selected_markers:
                                        st.markdown("###### Colors")
                                        mc_cols = st.columns(min(3, len(selected_markers)))
                                        for idx_m, marker_name in enumerate(selected_markers):
                                            c_idx = channel_names.index(marker_name)
                                            fallback_color = _hex_from_rgb(channel_colors[c_idx]) if c_idx < len(channel_colors) else "#ffffff"
                                            widget_key = f"tab8_triad_p2_marker_color_{marker_name}"
                                            default_color = st.session_state.get(widget_key, fallback_color)
                                            with mc_cols[idx_m % len(mc_cols)]:
                                                marker_colors[marker_name] = st.color_picker(marker_name, value=default_color, key=widget_key)
                                        
                                        st.markdown("###### Contrast Scaling")
                                        scale_c1, scale_c2 = st.columns(2)
                                        marker_low_pct = scale_c1.number_input("Lower Clip %", min_value=0.0, max_value=99.0, value=1.0, step=0.5, key="tab8_triad_p2_marker_low_pct")
                                        marker_high_pct = scale_c2.number_input("Upper Clip %", min_value=1.0, max_value=100.0, value=99.0, step=0.5, key="tab8_triad_p2_marker_high_pct")
                                        markers_opacity = st.slider("Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="tab8_triad_p2_markers_opacity")

                                        st.markdown("###### Resolution & Scaling")
                                        res_c1, res_c2 = st.columns(2)
                                        _init_selectbox_state("tab8_triad_p2_marker_res_mode", ["Dynamic (Auto)", "Manual Override"], preferred="Dynamic (Auto)", fallback=0)
                                        marker_res_mode = res_c1.selectbox(
                                            "Resolution Mode",
                                            options=["Dynamic (Auto)", "Manual Override"],
                                            key="tab8_triad_p2_marker_res_mode",
                                            help="Dynamic (Auto) adjusts the load resolution based on your crop window. Manual Override uses a fixed limit for the whole image."
                                        )
                                        if marker_res_mode == "Dynamic (Auto)":
                                            marker_target_res = res_c2.number_input(
                                                "Target Crop Resolution (px)",
                                                min_value=200,
                                                max_value=10000,
                                                value=1600,
                                                step=100,
                                                key="tab8_triad_p2_marker_target_res",
                                                help="Target maximum dimension of the cropped area in pixels."
                                            )
                                        else:
                                            marker_manual_limit = res_c2.number_input(
                                                "Load Limit (Max px)",
                                                min_value=200,
                                                max_value=25000,
                                                value=2000,
                                                step=500,
                                                key="tab8_triad_p2_marker_manual_limit",
                                                help="Fixed limit for the maximum dimension of the loaded OME-TIFF image."
                                            )

            # 4. ANNOTATIONS TAB
            with control_tabs[3]:
                st.markdown("##### GeoJSON Annotation Overlay")
                show_geojson_annotations = st.checkbox("Overlay GeoJSON annotations", value=False, key="tab8_triad_p2_show_geojson_annotations")
                annotation_dir = st.text_input("Annotations Directory", value=st.session_state.tab8_triad_p2_annotation_dir, key="tab8_triad_p2_annotation_dir")
                
                active_geojson_records = []
                geojson_annotation_opacity = 0.35
                geojson_annotation_line_width = 1.0
                selected_ann_files = []
                label_colors = {}
                visible_labels = []

                if show_geojson_annotations:
                    if not annotation_dir or not os.path.exists(annotation_dir):
                        st.error("Annotations directory does not exist.")
                    else:
                        ann_files = _list_files_cached(annotation_dir, (".geojson", ".json"), recursive=True)
                        if not ann_files:
                            st.warning("No GeoJSON/JSON files found in the directory.")
                        else:
                            matched_ann_file = _best_match_file(ann_files, selected_batch, selected_roi)
                            _init_multiselect_state(
                                "tab8_triad_p2_annotation_files",
                                ann_files,
                                [matched_ann_file] if matched_ann_file in ann_files else []
                            )
                            selected_ann_files = st.multiselect(
                                "GeoJSON Files to Load",
                                options=ann_files,
                                format_func=lambda path: Path(path).name,
                                key="tab8_triad_p2_annotation_files"
                            )
                            
                            # Parse labels and offer color pickers using fast cached label scanner
                            if selected_ann_files:
                                unique_labels = set()
                                for path_str in selected_ann_files:
                                    try:
                                        mtime = Path(path_str).stat().st_mtime
                                        lbls = _get_geojson_labels_cached(path_str, mtime)
                                        unique_labels.update(lbls)
                                    except Exception as exc:
                                        st.error(f"Error reading labels from {Path(path_str).name}: {exc}")
                                
                                unique_labels = sorted(list(unique_labels))
                                
                                if unique_labels:
                                    st.markdown("###### Annotation Styling")
                                    
                                    _init_multiselect_state(
                                        "tab8_triad_p2_visible_labels",
                                        unique_labels,
                                        unique_labels
                                    )
                                    visible_labels = st.multiselect(
                                        "Visible Annotation Labels",
                                        options=unique_labels,
                                        key="tab8_triad_p2_visible_labels",
                                        help="Annotations are rendered in the order they are selected (first selected = back, last selected = front)."
                                    )
                                    st.caption("Order of selection determines depth: first selected is at the back, last selected is in front.")
                                    
                                    # Color pickers per label
                                    st.markdown("###### Colors")
                                    lc_cols = st.columns(2)
                                    for idx_l, label in enumerate(unique_labels):
                                        fallback_color = _default_label_color(label)
                                        widget_key = f"tab8_triad_p2_annotation_color_{label}"
                                        default_color = st.session_state.get(widget_key, fallback_color)
                                        with lc_cols[idx_l % 2]:
                                            label_colors[label] = st.color_picker(label, value=default_color, key=widget_key)
                                            
                                    geojson_annotation_opacity = st.slider("Fill Opacity", min_value=0.0, max_value=1.0, value=0.35, step=0.05, key="tab8_triad_p2_annotation_opacity")
                                    geojson_annotation_line_width = st.slider("Border Line Width", min_value=0.0, max_value=5.0, value=1.0, step=0.5, key="tab8_triad_p2_annotation_line_width")

            # 5. TRIAD STYLE TAB
            with control_tabs[4]:
                st.markdown("##### Triad Representation")
                show_triad_points = st.checkbox("Show Triad Points/Centroids", value=True, key="tab8_triad_p2_show_triad_points")
                
                triad_point_color_mode = "Single Color"
                triad_color = "#d62728"
                triad_point_size = 24.0
                triad_alpha = 0.85
                show_cell_context = True
                
                color_a, color_b, color_c = "#1f77b4", "#ff7f0e", "#2ca02c"

                if show_triad_points:
                    _init_selectbox_state("tab8_triad_p2_triad_point_color_mode", ["Single Color", "By Phenotype (A/B/C)", "By Style"], fallback=0)
                    triad_point_color_mode = st.selectbox(
                        "Points Color Mode",
                        options=["Single Color", "By Phenotype (A/B/C)", "By Style"],
                        key="tab8_triad_p2_triad_point_color_mode"
                    )
                    
                    if triad_point_color_mode == "Single Color":
                        triad_color = st.color_picker("Point Color", value="#d62728", key="tab8_triad_p2_triad_color")
                    elif triad_point_color_mode == "By Phenotype (A/B/C)":
                        c_a, c_b, c_c = st.columns(3)
                        color_a = c_a.color_picker(f"Phenotype A ({ph_a})", value="#1f77b4", key="tab8_triad_p2_color_a")
                        color_b = c_b.color_picker(f"Phenotype B ({ph_b})", value="#ff7f0e", key="tab8_triad_p2_color_b")
                        color_c = c_c.color_picker(f"Phenotype C ({ph_c})", value="#2ca02c", key="tab8_triad_p2_color_c")
                        
                    triad_point_size = st.number_input("Point Size", min_value=1.0, max_value=200.0, value=24.0, step=1.0, key="tab8_triad_p2_triad_size")
                    triad_alpha = st.slider("Points Opacity", min_value=0.0, max_value=1.0, value=0.85, step=0.05, key="tab8_triad_p2_triad_alpha")
                    show_cell_context = st.checkbox("Show all cells background context", value=True, key="tab8_triad_p2_show_cells")

                st.divider()
                st.markdown("##### Triad Lines / Triangles")
                show_triad_lines = st.checkbox("Show Triad Lines/Triangles", value=False, key="tab8_triad_p2_show_triad_lines")
                triad_line_color_mode = "Single Color"
                triad_line_color = "#d62728"
                triad_line_width = 1.0
                triad_line_alpha = 0.7
                
                if show_triad_lines:
                    _init_selectbox_state("tab8_triad_p2_triad_line_color_mode", ["Single Color", "By Style"], fallback=0)
                    triad_line_color_mode = st.selectbox(
                        "Line Color Mode",
                        options=["Single Color", "By Style"],
                        key="tab8_triad_p2_triad_line_color_mode"
                    )
                    if triad_line_color_mode == "Single Color":
                        triad_line_color = st.color_picker("Line Color", value="#d62728", key="tab8_triad_p2_triad_line_color")
                    triad_line_width = st.number_input("Line Width", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="tab8_triad_p2_triad_line_width")
                    triad_line_alpha = st.slider("Line Opacity", min_value=0.0, max_value=1.0, value=0.70, step=0.05, key="tab8_triad_p2_triad_line_alpha")

                st.divider()
                st.markdown("##### Triad Density Cloud")
                show_triad_density_cloud = st.checkbox("Show Triad Density Cloud", value=False, key="tab8_triad_p2_show_triad_density_cloud")
                cloud_color_mode = "Standard Colormap"
                cloud_color_from = "#0000ff"
                cloud_color_to = "#ff0000"
                cloud_bins = 400
                cloud_sigma_scale = 1.0
                cloud_density_weight = True
                cloud_cmap = "inferno"
                cloud_clip_to_tissue = True

                if show_triad_density_cloud:
                    _init_selectbox_state("tab8_triad_p2_cloud_color_mode", ["Standard Colormap", "Custom gradient (From - To)"], fallback=0)
                    cloud_color_mode = st.selectbox(
                        "Cloud Color Mode",
                        options=["Standard Colormap", "Custom gradient (From - To)"],
                        key="tab8_triad_p2_cloud_color_mode"
                    )
                    if cloud_color_mode == "Standard Colormap":
                        p2_cmap_options = [
                            "inferno", "viridis", "magma", "plasma", "turbo",
                            "hot", "YlOrRd", "coolwarm", "Spectral", "cividis",
                            "RdYlBu_r", "OrRd", "PuRd"
                        ]
                        _init_selectbox_state("tab8_triad_p2_cloud_cmap", p2_cmap_options, preferred="inferno", fallback=0)
                        cloud_cmap = st.selectbox("Colormap", options=p2_cmap_options, key="tab8_triad_p2_cloud_cmap")
                    else:
                        g_from, g_to = st.columns(2)
                        cloud_color_from = g_from.color_picker("From Color", value="#0000ff", key="tab8_triad_p2_cloud_color_from")
                        cloud_color_to = g_to.color_picker("To Color", value="#ff0000", key="tab8_triad_p2_cloud_color_to")

                    st.markdown("###### Density Parameters")
                    cloud_bins = st.number_input("Grid resolution (bins)", min_value=80, max_value=1000, value=400, step=10, key="tab8_triad_p2_cloud_bins")
                    cloud_sigma_scale = st.number_input("Spread scale (sigma)", min_value=0.5, max_value=5.5, value=1.0, step=0.1, key="tab8_triad_p2_cloud_sigma")
                    cloud_density_weight = st.checkbox("Weight by local density", value=True, key="tab8_triad_p2_cloud_density")
                    cloud_clip_to_tissue = st.checkbox("Clip density cloud to tissue", value=True, key="tab8_triad_p2_cloud_clip")

                st.divider()
                st.markdown("##### Contour Lines")
                show_contours = st.checkbox("Show Contours", value=False, key="tab8_triad_p2_show_contours")
                contour_levels = 8
                contour_line_width = 1.0
                contour_color_mode = "Match colormap"
                contour_color = "#ffffff"

                if show_contours:
                    if not show_triad_density_cloud:
                        st.info("Contours will be generated using the density cloud configuration parameters.")
                        # Force cloud computation variables if only contours is enabled
                        cloud_bins = st.number_input("Grid resolution (bins)", min_value=80, max_value=1000, value=400, step=10, key="tab8_triad_p2_cloud_bins_ctr")
                        cloud_sigma_scale = st.number_input("Spread scale (sigma)", min_value=0.5, max_value=5.5, value=1.0, step=0.1, key="tab8_triad_p2_cloud_sigma_ctr")
                        cloud_density_weight = st.checkbox("Weight by local density", value=True, key="tab8_triad_p2_cloud_density_ctr")
                        cloud_clip_to_tissue = st.checkbox("Clip density cloud to tissue", value=True, key="tab8_triad_p2_cloud_clip_ctr")

                    contour_levels = st.slider("Contour Levels", min_value=2, max_value=30, value=8, step=1, key="tab8_triad_p2_contour_levels")
                    contour_line_width = st.number_input("Contour Line Width", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="tab8_triad_p2_contour_line_width")
                    _init_selectbox_state("tab8_triad_p2_contour_color_mode", ["Match colormap", "Single Color"], fallback=0)
                    contour_color_mode = st.selectbox(
                        "Contour Colors",
                        options=["Match colormap", "Single Color"],
                        key="tab8_triad_p2_contour_color_mode"
                    )
                    if contour_color_mode == "Single Color":
                        contour_color = st.color_picker("Contour Line Color", value="#ffffff", key="tab8_triad_p2_contour_color")

            # 6. OUTPUT & LAYERS TAB
            with control_tabs[5]:
                st.markdown("##### Background Customization")
                _init_selectbox_state("tab8_triad_p2_bg_color_mode", ["White", "Black", "Transparent", "Custom Hex"], fallback=0)
                bg_color_mode = st.selectbox(
                    "Background Color",
                    options=["White", "Black", "Transparent", "Custom Hex"],
                    key="tab8_triad_p2_bg_color_mode"
                )
                bg_custom_hex = "#ffffff"
                if bg_color_mode == "Custom Hex":
                    bg_custom_hex = st.color_picker("Choose Background", value="#ffffff", key="tab8_triad_p2_bg_custom_hex")

                st.divider()
                st.markdown("##### Layer Order (Bottom to Top)")
                layer_options = [
                    "None",
                    "Markers",
                    "Cell Context",
                    "Tissue Annotations",
                    "Virtual Tissue",
                    "Triad Density Cloud",
                    "Triad Contours",
                    "Triad Lines/Triangles",
                    "Triad Points"
                ]
                
                # Dynamic layer stacking slots
                _init_selectbox_state("tab8_triad_p2_layer_1", layer_options, fallback=1)
                _init_selectbox_state("tab8_triad_p2_layer_2", layer_options, fallback=2)
                _init_selectbox_state("tab8_triad_p2_layer_3", layer_options, fallback=3)
                _init_selectbox_state("tab8_triad_p2_layer_4", layer_options, fallback=4)
                _init_selectbox_state("tab8_triad_p2_layer_5", layer_options, fallback=5)
                _init_selectbox_state("tab8_triad_p2_layer_6", layer_options, fallback=6)
                _init_selectbox_state("tab8_triad_p2_layer_7", layer_options, fallback=7)
                _init_selectbox_state("tab8_triad_p2_layer_8", layer_options, fallback=8)
                l1 = st.selectbox("Layer 1 (Bottom)", options=layer_options, key="tab8_triad_p2_layer_1")
                l2 = st.selectbox("Layer 2", options=layer_options, key="tab8_triad_p2_layer_2")
                l3 = st.selectbox("Layer 3", options=layer_options, key="tab8_triad_p2_layer_3")
                l4 = st.selectbox("Layer 4", options=layer_options, key="tab8_triad_p2_layer_4")
                l5 = st.selectbox("Layer 5", options=layer_options, key="tab8_triad_p2_layer_5")
                l6 = st.selectbox("Layer 6", options=layer_options, key="tab8_triad_p2_layer_6")
                l7 = st.selectbox("Layer 7", options=layer_options, key="tab8_triad_p2_layer_7")
                l8 = st.selectbox("Layer 8 (Top)", options=layer_options, key="tab8_triad_p2_layer_8")
                layer_order = [l1, l2, l3, l4, l5, l6, l7, l8]

                st.divider()
                st.markdown("##### Export Settings")
                show_legend_triad = st.checkbox("Show triad legend", value=True, key="tab8_triad_p2_show_legend_triad")
                _init_selectbox_state("tab8_triad_p2_export_format", ["PNG", "SVG", "PDF"], fallback=0)
                export_format = st.selectbox("Export Format", options=["PNG", "SVG", "PDF"], key="tab8_triad_p2_export_format")
                dpi = st.slider("Export Resolution (DPI)", min_value=72, max_value=600, value=300, step=1, key="tab8_triad_p2_dpi")

            # 7. THEMES & PRESETS TAB
            with control_tabs[6]:
                st.markdown("##### Theme & Preset Manager")
                
                active_project = st.session_state.get("active_project", {})
                saved_themes = active_project.get("tab8_triad_themes", {})
                
                # Loading existing themes
                if saved_themes:
                    st.markdown("###### Load Theme")
                    theme_options = ["-- Select --"] + sorted(list(saved_themes.keys()))
                    selected_theme = st.selectbox("Select Theme", options=theme_options, key="tab8_triad_theme_select")
                    
                    if selected_theme != "-- Select --":
                        col_apply, col_delete = st.columns(2)
                        if col_apply.button("Apply Theme", key="tab8_triad_apply_theme_btn", type="primary", width="stretch"):
                            theme_dict = saved_themes[selected_theme]
                            for tk, tv in theme_dict.items():
                                st.session_state[tk] = tv
                            st.success(f"Applied theme '{selected_theme}'. Refreshing...")
                            st.rerun()
                            
                        if col_delete.button("Delete Theme", key="tab8_triad_delete_theme_btn", type="secondary", width="stretch"):
                            del saved_themes[selected_theme]
                            active_project["tab8_triad_themes"] = saved_themes
                            st.session_state["active_project"] = active_project
                            from spacemap_gui.persistence import save_active_project_state
                            save_active_project_state("tab8_analysis")
                            st.success(f"Deleted theme '{selected_theme}'.")
                            st.rerun()
                else:
                    st.info("No saved themes in this project yet.")
                    
                st.divider()
                st.markdown("###### Save Current Configuration as Theme")
                new_theme_name = st.text_input("Theme Name", placeholder="e.g. My Custom Theme", key="tab8_triad_new_theme_name")
                if st.button("Save Theme", key="tab8_triad_save_theme_btn", width="stretch"):
                    if not new_theme_name.strip():
                        st.error("Please enter a valid theme name.")
                    else:
                        theme_name_clean = new_theme_name.strip()
                        theme_dict = {}
                        for key, val in st.session_state.items():
                            if isinstance(key, str) and key.startswith("tab8_triad_") and not _is_transient_tab8_key(key):
                                theme_dict[key] = _to_jsonable(val)
                        
                        saved_themes[theme_name_clean] = theme_dict
                        active_project["tab8_triad_themes"] = saved_themes
                        st.session_state["active_project"] = active_project
                        from spacemap_gui.persistence import save_active_project_state
                        save_active_project_state("tab8_analysis")
                        st.success(f"Theme '{theme_name_clean}' saved and synced with project file.")
                        st.rerun()

        # Right Column preview area
        with col_preview:
            st.markdown("#### Preview & Export Map")

            # Collect active regional triad instances
            region_df = run_df[
                (run_df["__batch__"].astype(str) == str(selected_batch))
                & (run_df["__roi__"].astype(str) == str(selected_roi))
            ].copy()

            if region_df.empty:
                st.warning("No cells in selected Batch/ROI.")
                return

            max_collect = st.session_state.get("tab8_triad_p2_max_collect", 50000)
            metrics_rows = []
            
            p3_triad = (str(ph_a), str(ph_b), str(ph_c))
            
            if not p3_methods:
                st.warning("Please select at least one Phenotyping method column.")
                return
            if not style_configs:
                st.warning("Please select at least one Triad detection style.")
                return

            # Check if the cache matches the currently selected batch and ROI
            if "tab8_triad_p2_cached_runs" in st.session_state:
                cached_batch = st.session_state.get("tab8_triad_p2_cached_batch")
                cached_roi = st.session_state.get("tab8_triad_p2_cached_roi")
                if cached_batch != selected_batch or cached_roi != selected_roi:
                    st.session_state.pop("tab8_triad_p2_cached_runs", None)
                    st.session_state.pop("tab8_triad_p2_cached_summary_df", None)

            # Check if we should compute (Update Map Preview clicked)
            should_compute = run_render
            has_cache = "tab8_triad_p2_cached_runs" in st.session_state

            if should_compute:
                import io
                out_dir = Path(project_root) / "outputs" / "spatial_analysis" / "triad"
                out_dir.mkdir(parents=True, exist_ok=True)

                cached_runs = []
                metrics_rows = []

                # DEFERRED COMPOSITING OF MARKERS
                composite_image = None
                full_hw = (1, 1)
                if show_markers and matched_image_path and selected_markers:
                    with st.spinner("Compositing marker channels..."):
                        try:
                            mtime = Path(matched_image_path).stat().st_mtime
                            # Calculate max_dim based on resolution mode
                            res_mode = st.session_state.get("tab8_triad_p2_marker_res_mode", "Dynamic (Auto)")
                            if res_mode == "Dynamic (Auto)":
                                target_res = st.session_state.get("tab8_triad_p2_marker_target_res", 1600)
                                import tifffile
                                try:
                                    with tifffile.TiffFile(str(matched_image_path)) as tif:
                                        full_shape = tif.pages[0].shape
                                        full_h = int(full_shape[-2])
                                        full_w = int(full_shape[-1])
                                    full_max = max(full_h, full_w)
                                    crop_w = float(crop_x_max) - float(crop_x_min)
                                    crop_h = float(crop_y_max) - float(crop_y_min)
                                    crop_max_px = max(crop_w, crop_h)
                                    if crop_max_px > 0:
                                        scale = max(1.0, crop_max_px / target_res)
                                        max_dim = int(full_max / scale)
                                    else:
                                        max_dim = 1600
                                except Exception as exc:
                                    max_dim = 1600
                            else:
                                max_dim = int(st.session_state.get("tab8_triad_p2_marker_manual_limit", 2000))

                            comp_arr, full_hw = _compose_preview_image_cached(
                                image_path_str=str(matched_image_path),
                                mtime=mtime,
                                channel_names=channel_names,
                                color_map=marker_colors,
                                selected_markers=selected_markers,
                                max_dim=max_dim,
                                low_percentile=marker_low_pct,
                                high_percentile=marker_high_pct,
                            )
                            from PIL import Image
                            composite_image = Image.fromarray(comp_arr)
                        except Exception as exc:
                            st.error(f"Error compositing markers: {exc}")

                # DEFERRED PARSING OF GEOJSON GEOMETRIES
                active_geojson_records = []
                if show_geojson_annotations and selected_ann_files:
                    with st.spinner("Loading GeoJSON geometries..."):
                        for path_str in selected_ann_files:
                            try:
                                mtime = Path(path_str).stat().st_mtime
                                recs = _load_geojson_groups_cached(path_str, mtime)
                                for rec in recs:
                                    if rec["label"] in visible_labels:
                                        rec_copy = dict(rec)
                                        rec_copy["color_hex"] = label_colors[rec["label"]]
                                        active_geojson_records.append(rec_copy)
                            except Exception as exc:
                                st.error(f"Error loading geometries from {Path(path_str).name}: {exc}")
                        
                        # Sort active_geojson_records by order of selection in visible_labels (first selected = back, last selected = front)
                        label_order = {lbl: idx for idx, lbl in enumerate(visible_labels)}
                        active_geojson_records = sorted(active_geojson_records, key=lambda r: label_order.get(r["label"], 9999))

                for style_name, min_um, max_um in style_configs:
                    for method_col in p3_methods:
                        with st.spinner(f"Computing triads: {style_name} | {method_col}..."):
                            triad_df = _collect_region_triad_instances(
                                region_df=region_df,
                                phenotype_col=str(method_col),
                                triad=p3_triad,
                                style_name=str(style_name),
                                min_dist_um=float(min_um),
                                max_dist_um=float(max_um),
                                pixel_size_um=float(pixel_size_um),
                                max_collect=int(max_collect),
                            )

                        with st.spinner("Generating spatial map..."):
                            fig, metrics = _plot_triad_virtual_tissue(
                                run_df=run_df,
                                triad_df=triad_df,
                                batch_value=str(selected_batch),
                                roi_value=str(selected_roi),
                                pixel_size_um=float(pixel_size_um),
                                method=str(method),
                                alpha=float(alpha),
                                use_median=bool(use_median),
                                percentile=float(buffer_percentile),
                                min_hole_size=float(min_hole_size),
                                
                                # Render options
                                show_markers=show_markers,
                                composite_image=composite_image,
                                full_hw=full_hw,
                                markers_opacity=markers_opacity,
                                
                                show_geojson_annotations=show_geojson_annotations,
                                active_geojson_records=active_geojson_records,
                                geojson_annotation_opacity=geojson_annotation_opacity,
                                geojson_annotation_line_width=geojson_annotation_line_width,
                                
                                show_triad_points=show_triad_points,
                                triad_point_color_mode=triad_point_color_mode,
                                triad_color=triad_color,
                                triad_point_size=triad_point_size,
                                triad_alpha=triad_alpha,
                                
                                show_triad_lines=show_triad_lines,
                                triad_line_color_mode=triad_line_color_mode,
                                triad_line_color=triad_line_color,
                                triad_line_width=triad_line_width,
                                triad_line_alpha=triad_line_alpha,
                                
                                show_triad_density_cloud=show_triad_density_cloud,
                                cloud_color_mode=cloud_color_mode,
                                cloud_color_from=cloud_color_from,
                                cloud_color_to=cloud_color_to,
                                cloud_bins=cloud_bins,
                                cloud_sigma_scale=cloud_sigma_scale,
                                cloud_density_weight=cloud_density_weight,
                                cloud_cmap=cloud_cmap,
                                cloud_clip_to_tissue=cloud_clip_to_tissue,
                                
                                show_contours=show_contours,
                                contour_levels=contour_levels,
                                contour_line_width=contour_line_width,
                                contour_color_mode=contour_color_mode,
                                contour_color=contour_color,
                                
                                show_cell_context=show_cell_context,
                                
                                bg_color_mode=bg_color_mode,
                                bg_custom_hex=bg_custom_hex,
                                
                                crop_to_cells=crop_to_cells,
                                crop_x_min=crop_x_min,
                                crop_x_max=crop_x_max,
                                crop_y_min=crop_y_min,
                                crop_y_max=crop_y_max,
                                crop_padding=crop_padding,
                                
                                layer_order=layer_order,
                                
                                ph_a=ph_a,
                                ph_b=ph_b,
                                ph_c=ph_c,
                                color_a=color_a,
                                color_b=color_b,
                                color_c=color_c,
                            )

                        if fig is None:
                            st.warning(f"Could not render spatial map for method={method_col}, style={style_name}.")
                            continue

                        # Export map to bytes for instant preview display
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", transparent=(bg_color_mode == "Transparent"))
                        buf.seek(0)
                        preview_png_bytes = buf.getvalue()

                        # Save full resolution file to disk
                        output_stem = f"triad_spatial_{method_col}_{style_name}_{selected_batch}_{selected_roi}"
                        fig_path = out_dir / f"{output_stem}.{export_format.lower()}"
                        fig.savefig(fig_path, dpi=int(dpi), bbox_inches="tight", transparent=(bg_color_mode == "Transparent"))
                        plt.close(fig)

                        # Read full download file data
                        with open(fig_path, "rb") as f_data:
                            dl_data = f_data.read()

                        mrow = dict(metrics) if isinstance(metrics, dict) else {}
                        mrow["phenotype_method"] = str(method_col)
                        mrow["triad_style"] = str(style_name)
                        mrow["triad_definition"] = "::".join(p3_triad)
                        mrow["min_dist_um"] = float(min_um)
                        mrow["max_dist_um"] = float(max_um)
                        metrics_rows.append(mrow)

                        cached_runs.append({
                            "preview_png_bytes": preview_png_bytes,
                            "dl_data": dl_data,
                            "filename": fig_path.name,
                            "export_format": export_format,
                            "method_col": method_col,
                            "style_name": style_name,
                        })

                metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
                if not metrics_df.empty:
                    summary_csv = out_dir / f"triad_spatial_summary_{selected_batch}_{selected_roi}.csv"
                    metrics_df.to_csv(summary_csv, index=False)

                st.session_state["tab8_triad_p2_cached_runs"] = cached_runs
                st.session_state["tab8_triad_p2_cached_summary_df"] = metrics_df
                st.session_state["tab8_triad_p2_cached_batch"] = selected_batch
                st.session_state["tab8_triad_p2_cached_roi"] = selected_roi
                st.success("Calculations and preview generation complete!")
                st.rerun()

            # Render from cache
            if "tab8_triad_p2_cached_runs" in st.session_state:
                cached_runs = st.session_state["tab8_triad_p2_cached_runs"]
                for run in cached_runs:
                    st.markdown(f"##### Map: method={run['method_col']} | style={run['style_name']}")
                    _show_png_bytes(run["preview_png_bytes"], width="stretch")
                    
                    st.download_button(
                        label=f"Download {run['export_format']} Map",
                        data=run["dl_data"],
                        file_name=run["filename"],
                        mime=f"image/{run['export_format'].lower()}" if run['export_format'] != "PDF" else "application/pdf",
                        key=f"dl_{run['filename']}"
                    )
                
                metrics_df = st.session_state.get("tab8_triad_p2_cached_summary_df")
                if metrics_df is not None and not metrics_df.empty:
                    st.divider()
                    st.markdown("##### Summary Table")
                    st.dataframe(metrics_df, width="stretch")
            else:
                st.info("Map preview not loaded. Select your settings on the left and click 'Update Map Preview' above to visualize.")

def _render_panel_image_triads(
    project_root: Path,
    triad_instances_df: pd.DataFrame,
    marker_candidates: list[str],
    triad_fallback_names: tuple[str, str, str],
):
    with st.expander("**Panel 2 · Triad Image Visualization**", expanded=False):
        st.info(
            "**How it works & What it does:** This panel extracts regional fields of view (FOVs) centered on detected triads "
            "from multiplexed OME-TIFF images. It plots marker intensities, overlays cell centroids colored by phenotype, "
            "draws the triad interaction lines, and displays segmentation boundaries.\n\n"
            "**Scientific Interpretation:** Allows visual inspection to verify that the mathematically detected triads "
            "represent actual physical contact/spatial co-localization, rather than segmentation artifacts or z-plane overlap."
        )
        if triad_instances_df is None or triad_instances_df.empty:
            st.info("Run Panel 1 or load a triad instances CSV first.")
            return

        styles = sorted(triad_instances_df["style"].dropna().astype(str).unique().tolist()) if "style" in triad_instances_df.columns else ["unknown"]
        style_col, batch_col, roi_col = st.columns(3)
        selected_style = style_col.selectbox("Style", options=styles, key="tab8_triad_p3_style")
        ph_a, ph_b, ph_c = _resolve_triad_names_from_instances(
            triad_instances_df=triad_instances_df,
            selected_style=str(selected_style),
            fallback=triad_fallback_names,
        )
        work = triad_instances_df.copy()
        if "style" in work.columns:
            work = work[work["style"].astype(str) == str(selected_style)]

        batch_vals = sorted(work["batch"].dropna().astype(str).unique().tolist())
        if not batch_vals:
            st.warning("No valid batch values in triad instances.")
            return
        selected_batch = batch_col.selectbox("Batch", options=["All"] + batch_vals, key="tab8_triad_p3_batch")
        if selected_batch != "All":
            work = work[work["batch"].astype(str) == str(selected_batch)]

        roi_vals = sorted(work["roi"].dropna().astype(str).unique().tolist())
        selected_roi = roi_col.selectbox("ROI", options=["All"] + roi_vals, key="tab8_triad_p3_roi")
        if selected_roi != "All":
            work = work[work["roi"].astype(str) == str(selected_roi)]

        c1, c2 = st.columns(2)
        roi_dir = c1.text_input("ROI Image Directory", value=str(project_root / "data" / "images"), key="tab8_triad_p3_roi_dir")
        seg_dir = c2.text_input("Segmentation GeoJSON Directory", value=str(project_root / "outputs" / "segmentation" / "cellpose"), key="tab8_triad_p3_seg_dir")

        m1, m2, m3 = st.columns(3)
        batch_prefix_regex = m1.text_input("Batch prefix regex", value=r"^(.+_roi_\d+)", key="tab8_triad_p3_batch_regex")
        roi_prefix_regex = m2.text_input("ROI image prefix regex", value=r"^(.+_roi_\d+)", key="tab8_triad_p3_roi_regex")
        seg_prefix_regex = m3.text_input("GeoJSON prefix regex", value=r"^(.+_roi_\d+)", key="tab8_triad_p3_seg_regex")

        st.markdown("**Markers per triad phenotype (1-3 each)**")
        p1, p2, p3 = st.columns(3)
        marks_a = p1.multiselect(f"Markers for phenotype A ({ph_a})", options=marker_candidates, default=marker_candidates[:2], key="tab8_triad_p3_marks_a")
        marks_b = p2.multiselect(f"Markers for phenotype B ({ph_b})", options=marker_candidates, default=marker_candidates[:2], key="tab8_triad_p3_marks_b")
        marks_c = p3.multiselect(f"Markers for phenotype C ({ph_c})", options=marker_candidates, default=marker_candidates[:2], key="tab8_triad_p3_marks_c")

        cc1, cc2, cc3 = st.columns(3)
        color_a = cc1.color_picker(f"Cell A color ({ph_a})", value="#1f77b4", key="tab8_triad_p3_color_a")
        color_b = cc2.color_picker(f"Cell B color ({ph_b})", value="#d62728", key="tab8_triad_p3_color_b")
        color_c = cc3.color_picker(f"Cell C color ({ph_c})", value="#2ca02c", key="tab8_triad_p3_color_c")

        r1, r2 = st.columns(2)
        n_random = r1.number_input("Random triads to visualize", min_value=1, max_value=100, value=5, step=1, key="tab8_triad_p3_n_random")
        seed = r2.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1, key="tab8_triad_p3_seed")

        # Crop dimension controls (linked square by default)
        cd1, cd2 = st.columns([1, 2])
        crop_square = cd1.checkbox("Link width & height (Square)", value=True, key="tab8_triad_p3_crop_square")
        if crop_square:
            crop_size = cd2.number_input("Box size (µm)", min_value=40.0, max_value=4000.0, value=160.0, step=10.0, key="tab8_triad_p3_crop_size_linked")
            crop_width_um = crop_size
            crop_height_um = crop_size
        else:
            c_w, c_h = cd2.columns(2)
            crop_width_um = c_w.number_input("Box Width (µm)", min_value=40.0, max_value=4000.0, value=160.0, step=10.0, key="tab8_triad_p3_crop_width")
            crop_height_um = c_h.number_input("Box Height (µm)", min_value=40.0, max_value=4000.0, value=160.0, step=10.0, key="tab8_triad_p3_crop_height")

        pix = st.number_input("Pixel Size (µm)", min_value=0.01, value=0.5083, step=0.0001, format="%.4f", key="tab8_triad_p3_pixel_um")
        zoom_factor = st.number_input(
            "Zoom factor (higher = more zoom-in)",
            min_value=1.0,
            max_value=6.0,
            value=2.0,
            step=0.25,
            key="tab8_triad_p3_zoom_factor",
            help="Magnification level for the crop window around each triad.",
        )
        show_surrounding_seg = st.checkbox("Show surrounding cell segmentation (black outlines)", value=True, key="tab8_triad_p3_show_seg")

        out_dir = st.text_input(
            "Panel 2 Output Directory",
            value=str(project_root / "outputs" / "spatial_analysis" / "triad" / "panel3_visualization"),
            key="tab8_triad_p3_outdir",
        )

        run_panel2 = st.button("Render Panel 2", type="primary", key="tab8_triad_p3_run")
        if run_panel2:
            if tifffile is None or shape is None or Point is None:
                st.error("Panel 2 requires `tifffile` and `shapely` packages.")
                return

            if work.empty:
                st.warning("No triad rows left after filters.")
                return

            selected_markers = list(dict.fromkeys([*marks_a[:3], *marks_b[:3], *marks_c[:3]]))
            if not selected_markers:
                st.error("Select at least one marker.")
                return

            st.markdown("**Marker colors (used in single-marker tiles and overlay)**")
            marker_palette_defaults = ["#00ffff", "#ffcc00", "#ff00ff", "#00ff66", "#ff4444", "#6699ff", "#ffffff"]
            marker_colors: dict[str, str] = {}
            mcols = st.columns(min(4, max(1, len(selected_markers))))
            for idx_m, marker_name in enumerate(selected_markers):
                with mcols[idx_m % len(mcols)]:
                    marker_colors[marker_name] = st.color_picker(
                        f"{marker_name}",
                        value=marker_palette_defaults[idx_m % len(marker_palette_defaults)],
                        key=f"tab8_triad_p3_marker_color_{marker_name}",
                    )

            rng = np.random.default_rng(int(seed))
            n_take = min(int(n_random), len(work))
            sample_idx = rng.choice(len(work), size=n_take, replace=False)
            sample_df = work.iloc[sample_idx].reset_index(drop=True)

            roi_entries = _index_files(roi_dir, exts=(".tif", ".tiff", ".ome.tif", ".ome.tiff"), prefix_regex=roi_prefix_regex)
            seg_entries = _index_files(seg_dir, exts=(".geojson", ".json"), prefix_regex=seg_prefix_regex)
            if not roi_entries or not seg_entries:
                st.error("No ROI images or segmentation GeoJSON files found with current directories/regex.")
                return

            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            progress = st.progress(0)
            rendered = 0
            failed_rows = []

            cached_images = []

            for i, row in sample_df.iterrows():
                batch = str(row["batch"])
                batch_prefix = _extract_prefix(batch, batch_prefix_regex)
                roi_match = _best_prefix_match(batch_prefix, roi_entries)
                seg_match = _best_prefix_match(batch_prefix, seg_entries)

                if roi_match is None or seg_match is None:
                    failed_rows.append({"batch": batch, "roi": row.get("roi", ""), "reason": "No file match"})
                    progress.progress((i + 1) / max(len(sample_df), 1))
                    continue

                triad_centroid_x = float(row["triad_centroid_x"])
                triad_centroid_y = float(row["triad_centroid_y"])
                
                half_w_px = _um_to_px(float(crop_width_um) / 2.0, float(pix))
                half_h_px = _um_to_px(float(crop_height_um) / 2.0, float(pix))

                x0 = max(0, int(np.floor(triad_centroid_x - half_w_px)))
                x1 = int(np.ceil(triad_centroid_x + half_w_px))
                y0 = max(0, int(np.floor(triad_centroid_y - half_h_px)))
                y1 = int(np.ceil(triad_centroid_y + half_h_px))

                try:
                    with tifffile.TiffFile(roi_match["path"]) as tif:
                        channel_names = _parse_channel_names_from_ome_xml(getattr(tif, "ome_metadata", "") or "")
                    marker_to_idx = _resolve_marker_indices(channel_names, selected_markers)
                    if not marker_to_idx:
                        failed_rows.append({"batch": batch, "roi": row.get("roi", ""), "reason": "Markers not found in image channels"})
                        progress.progress((i + 1) / max(len(sample_df), 1))
                        continue

                    channel_indices = [marker_to_idx[m] for m in marker_to_idx.keys()]
                    crops_by_idx, _ = _load_tiff_crops(roi_match["path"], channel_indices, x0, x1, y0, y1)
                    marker_to_crop = {m: crops_by_idx[idx] for m, idx in marker_to_idx.items() if idx in crops_by_idx}
                    if not marker_to_crop:
                        failed_rows.append({"batch": batch, "roi": row.get("roi", ""), "reason": "Failed to read marker crops"})
                        progress.progress((i + 1) / max(len(sample_df), 1))
                        continue

                    polygons = _load_geojson_polygons(seg_match["path"])
                    triad_points = {
                        "A": (float(row["cell_a_x"]), float(row["cell_a_y"])),
                        "B": (float(row["cell_b_x"]), float(row["cell_b_y"])),
                        "C": (float(row["cell_c_x"]), float(row["cell_c_y"])),
                    }
                    triad_colors = {"A": str(color_a), "B": str(color_b), "C": str(color_c)}

                    fig = _plot_triad_marker_figure(
                        marker_to_crop=marker_to_crop,
                        all_polygons=polygons,
                        triad_points_global=triad_points,
                        crop_origin=(x0, y0),
                        triad_colors=triad_colors,
                        marker_colors=marker_colors,
                        show_surrounding_seg=bool(show_surrounding_seg),
                        zoom_factor=float(zoom_factor),
                    )
                    if fig is None:
                        failed_rows.append({"batch": batch, "roi": row.get("roi", ""), "reason": "Figure generation failed"})
                        progress.progress((i + 1) / max(len(sample_df), 1))
                        continue

                    # Save figure to bytes
                    import io
                    buf = io.BytesIO()
                    _global_dpi = int(st.session_state.get("tab8_density_global_png_dpi", 300))
                    fig.savefig(buf, format="png", dpi=_global_dpi, bbox_inches="tight")
                    buf.seek(0)
                    img_bytes = buf.getvalue()

                    save_name = out_path / f"triad_visual_{i + 1}_{batch}_{row.get('roi', '')}.png"
                    fig.savefig(save_name, dpi=_global_dpi, bbox_inches="tight")
                    plt.close(fig)

                    cached_images.append({
                        "title": f"Triad {i + 1}: batch={batch} | roi={row.get('roi', '')}",
                        "png_bytes": img_bytes,
                        "save_path": str(save_name),
                    })
                    rendered += 1
                except Exception as exc:
                    failed_rows.append({"batch": batch, "roi": row.get("roi", ""), "reason": str(exc)})

                progress.progress((i + 1) / max(len(sample_df), 1))

            progress.empty()
            st.session_state["tab8_triad_p2_visualizations"] = cached_images
            st.session_state["tab8_triad_p2_visual_rendered_count"] = rendered
            st.session_state["tab8_triad_p2_visual_failed_rows"] = failed_rows
            st.rerun()

        # Display cached visualizations
        if "tab8_triad_p2_visualizations" in st.session_state:
            cached_images = st.session_state["tab8_triad_p2_visualizations"]
            rendered_count = st.session_state.get("tab8_triad_p2_visual_rendered_count", 0)
            failed_rows = st.session_state.get("tab8_triad_p2_visual_failed_rows", [])

            st.success(f"Rendered {rendered_count} triad visualizations.")
            for img_info in cached_images:
                st.markdown(f"#### {img_info['title']}")
                _show_png_bytes(img_info["png_bytes"])
                st.caption(f"Saved: `{img_info['save_path']}`")

            if failed_rows:
                st.warning(f"{len(failed_rows)} triads could not be rendered.")
                st.dataframe(pd.DataFrame(failed_rows), width="stretch")


def _render_panel4_triad_stats(
    project_root: Path,
    run_df: pd.DataFrame,
    triad: tuple[str, str, str],
    phenotype_candidates: list[str],
    shared: dict[str, Any],
    selected_styles: list[str],
    local_min_um: float,
    local_max_um: float,
    neigh_min_um: float,
    neigh_max_um: float,
    default_pixel_size_um: float,
    default_min_cells_region: int,
    tissue_col: str,
    tissue_keep_value: str,
    default_out_dir: str,
):
    with st.expander("**Panel 4 · Triad Statistics Across Phenotyping Methods**", expanded=False):
        st.info(
            "**How it works & What it does:** Panel 4 compares triad metrics (counts, densities) across multiple phenotyping methods "
            "to assess the robustness of your findings. It performs paired statistical tests across connected tissues and maps "
            "zonal triad density gradients as a function of distance from the tumor-stroma boundary.\n\n"
            "**Scientific Interpretation:** Validates whether the discovered spatial associations are independent of the "
            "phenotyping algorithm and captures the exact spatial distribution profile (zonal enrichment) relative to the tumor boundary."
        )

        if not phenotype_candidates:
            st.info("No phenotype columns detected for Panel 4.")
            return

        p4_methods = [str(m) for m in shared.get("methods", []) if str(m) in phenotype_candidates]
        method_color_map: dict[str, str] = dict(shared.get("colors", {}))
        method_label_map: dict[str, str] = dict(shared.get("labels", {}))
        method_color_defaults = list(shared.get("color_defaults", [])) or [
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2",
        ]

        p4_pool_vals: set[str] = set()
        for mcol in p4_methods:
            if mcol in run_df.columns:
                p4_pool_vals.update(run_df[mcol].dropna().astype(str).unique().tolist())
        p4_pheno_pool = sorted(list(p4_pool_vals))
        if p4_methods and not p4_pheno_pool:
            st.warning("Panel 4: selected methods have no phenotype values after current filtering.")
            return

        tab_data, tab_style, tab_axes, tab_text, tab_zonal = st.tabs(
            [
                "1 · What to compare",
                "2 · Plot style",
                "3 · Y-axis & outliers",
                "4 · Text, size & significance",
                "5 · Tumor-bed subset & zonal curves",
            ]
        )

        with tab_data:
            if p4_methods:
                st.caption(
                    "Methods, display names and colors come from **Shared Figure Settings** at the top of the subtab: "
                    + ", ".join(f"{method_label_map.get(m, m)} ({m})" for m in p4_methods)
                )
            else:
                st.warning("Select at least one phenotyping method column in Shared Figure Settings.")

            p4t1, p4t2, p4t3 = st.columns(3)
            _init_selectbox_state("tab8_triad_p4_a", p4_pheno_pool, preferred=str(triad[0]), fallback=0)
            p4_triad_a = p4t1.selectbox(
                "Triad phenotype A",
                options=p4_pheno_pool,
                key="tab8_triad_p4_a",
            )
            _init_selectbox_state("tab8_triad_p4_b", p4_pheno_pool, preferred=str(triad[1]), fallback=1)
            p4_triad_b = p4t2.selectbox(
                "Triad phenotype B",
                options=p4_pheno_pool,
                key="tab8_triad_p4_b",
            )
            _init_selectbox_state("tab8_triad_p4_c", p4_pheno_pool, preferred=str(triad[2]), fallback=2)
            p4_triad_c = p4t3.selectbox(
                "Triad phenotype C",
                options=p4_pheno_pool,
                key="tab8_triad_p4_c",
            )

            c2, c3 = st.columns(2)
            p4_metrics = c2.multiselect(
                "Metrics to compare (one figure per metric)",
                options=["triad_count", "triad_density_per_mm2"],
                default=["triad_count", "triad_density_per_mm2"],
                key="tab8_triad_p4_metrics",
                format_func=lambda x: "Triad count" if x == "triad_count" else "Triad density (triads/mm²)",
            )
            p4_strat_cols = c3.multiselect(
                "Split figures by (optional stratification columns)",
                options=[c for c in run_df.columns if c not in set(REQUIRED_COORDS + ["__batch__", "__roi__"])],
                default=[],
                key="tab8_triad_p4_strat_cols",
            )

            st.markdown("**Triad distance styles**")
            selected_style_labels = st.multiselect(
                "Styles (one figure per style)",
                options=["Direct-contact / local triads", "Neighborhood annulus triads"],
                default=selected_styles[:1] if selected_styles else ["Direct-contact / local triads"],
                key="tab8_triad_p4_styles",
            )
            sd1, sd2, sd3, sd4 = st.columns(4)
            p4_local_min_um = sd1.number_input("Local min distance (µm)", min_value=0.0, max_value=5000.0, value=float(local_min_um), step=1.0, key="tab8_triad_p4_local_min_um")
            p4_local_max_um = sd2.number_input("Local max distance (µm)", min_value=0.1, max_value=5000.0, value=float(local_max_um), step=1.0, key="tab8_triad_p4_local_max_um")
            p4_neigh_min_um = sd3.number_input("Neighborhood min distance (µm)", min_value=0.0, max_value=5000.0, value=float(neigh_min_um), step=1.0, key="tab8_triad_p4_neigh_min_um")
            p4_neigh_max_um = sd4.number_input("Neighborhood max distance (µm)", min_value=0.1, max_value=10000.0, value=float(neigh_max_um), step=1.0, key="tab8_triad_p4_neigh_max_um")

        style_configs: list[tuple[str, float, float]] = []
        if "Direct-contact / local triads" in selected_style_labels:
            style_configs.append(("local", float(p4_local_min_um), float(p4_local_max_um)))
        if "Neighborhood annulus triads" in selected_style_labels:
            style_configs.append(("neighborhood", float(p4_neigh_min_um), float(p4_neigh_max_um)))

        # ── Computation ─────────────────────────────────────────────
        with tab_data:
            st.markdown("**Connected tissue area & detection**")
            d1, d2, d3 = st.columns(3)
            p4_pixel_size_um = d1.number_input("Pixel Size (µm)", min_value=0.01, value=float(default_pixel_size_um), step=0.0001, format="%.4f", key="tab8_triad_p4_pixel_um", help="Physical size of one pixel in micrometers.")
            p4_min_cells_region = d2.number_input("Min cells per connected tissue", min_value=3, max_value=50000, value=int(default_min_cells_region), step=1, key="tab8_triad_p4_min_cells_region", help="Regions with fewer cells are excluded from analysis.")
            p4_min_hole_size = d3.number_input("Min Hole Size (pixels²)", min_value=0.0, value=10000.0, step=1000.0, key="tab8_triad_p4_min_hole", help="Holes smaller than this are filled in the virtual tissue polygon.")

            e1, e2, e3, e4 = st.columns(4)
            p4_area_method_options = ["buffer_union", "alpha_shape"]
            _init_selectbox_state("tab8_triad_p4_area_method", p4_area_method_options, preferred="buffer_union", fallback=0)
            p4_area_method = e1.selectbox("Area method", options=p4_area_method_options, key="tab8_triad_p4_area_method", help="Geometry method for virtual tissue boundary.")
            p4_area_alpha = e2.number_input("Area alpha", min_value=1.0, max_value=500.0, value=100.0, step=1.0, key="tab8_triad_p4_area_alpha", help="Alpha radius for concave hull.")
            p4_area_use_median = e3.checkbox("Area: use median distance", value=False, key="tab8_triad_p4_area_use_median")
            p4_area_percentile = e4.number_input("Area: buffer percentile", min_value=90.0, max_value=99.9, value=99.0, step=0.1, format="%.1f", key="tab8_triad_p4_area_percentile", help="Percentile of nearest-neighbor distances for buffer.")

        with tab_data:
            filter_candidates = [
                c for c in run_df.columns
                if c not in set(REQUIRED_COORDS + ["__batch__", "__roi__"]) and c not in set(p4_methods)
            ]
            selected_filter_cols = st.multiselect(
                "Keep only rows matching (optional Panel 4 filters)",
                options=filter_candidates,
                default=[],
                key="tab8_triad_p4_filter_cols",
            )
            p4_df = _apply_value_filters(run_df, selected_filter_cols, key_prefix="tab8_triad_p4")

        with tab_style:
            st.markdown("**Distribution geometry**")
            s1, s2, s3 = st.columns(3)
            p4_plot_options = ["box", "violin", "box+violin"]
            _init_selectbox_state("tab8_triad_p4_plot_type", p4_plot_options, preferred="box", fallback=0)
            p4_plot_type = s1.selectbox(
                "Plot type",
                options=p4_plot_options,
                key="tab8_triad_p4_plot_type",
                help="Choose the main distribution geometry.",
            )
            p4_distribution_width = s2.slider(
                "Box/violin width",
                min_value=0.35,
                max_value=0.95,
                value=0.78,
                step=0.05,
                key="tab8_triad_p4_distribution_width",
            )
            p4_fill_alpha = s3.slider(
                "Box/violin fill opacity",
                min_value=0.1,
                max_value=0.9,
                value=0.35,
                step=0.05,
                key="tab8_triad_p4_fill_alpha",
            )

            st.markdown("**Dots (one per connected tissue)**")
            s4, s5, s6 = st.columns(3)
            p4_dot_size = s4.slider(
                "Dot size",
                min_value=2,
                max_value=40,
                value=9,
                step=1,
                key="tab8_triad_p4_dot_size",
            )
            p4_dot_alpha = s5.slider(
                "Dot opacity",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.05,
                key="tab8_triad_p4_dot_alpha",
            )
            p4_jitter_width = s6.slider(
                "Dot horizontal spread",
                min_value=0.02,
                max_value=0.35,
                value=0.08,
                step=0.01,
                key="tab8_triad_p4_jitter",
            )

            st.markdown("**Line widths**")
            l1, l2, l3, l4 = st.columns(4)
            p4_box_linewidth = l1.slider(
                "Box/whisker (pt)",
                min_value=0.3,
                max_value=2.5,
                value=1.0,
                step=0.1,
                key="tab8_triad_p4_box_lw",
            )
            p4_axis_linewidth = l2.slider(
                "Axis spine (pt)",
                min_value=0.3,
                max_value=2.0,
                value=0.8,
                step=0.1,
                key="tab8_triad_p4_axis_lw",
            )
            p4_grid_linewidth = l3.slider(
                "Grid line (pt)",
                min_value=0.3,
                max_value=1.5,
                value=0.7,
                step=0.1,
                key="tab8_triad_p4_grid_linewidth",
            )
            p4_sig_bracket_linewidth = l4.slider(
                "Significance bracket (pt)",
                min_value=0.3,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="tab8_triad_p4_sig_bracket_linewidth",
                help="Line width for significance test bracket lines.",
            )

        with tab_text:
            st.markdown("**Figure size**")
            d1, d2, d3 = st.columns(3)
            p4_fig_width = d1.slider(
                "Width per sub-panel (in)",
                min_value=3.0,
                max_value=12.0,
                value=5.0,
                step=0.5,
                key="tab8_triad_p4_fig_w",
            )
            p4_fig_height = d2.slider(
                "Figure height (in)",
                min_value=3.5,
                max_value=9.0,
                value=5.5,
                step=0.5,
                key="tab8_triad_p4_fig_h",
            )
            p4_panel_spacing = d3.slider(
                "Space between sub-panels",
                min_value=0.2,
                max_value=2.5,
                value=0.8,
                step=0.1,
                key="tab8_triad_p4_panel_spacing",
            )

            st.markdown("**Fonts**")
            f1, f2, f3, f4 = st.columns(4)
            p4_title_fontsize = f1.slider(
                "Title font (pt)",
                min_value=8,
                max_value=18,
                value=11,
                step=1,
                key="tab8_triad_p4_title_fontsize",
            )
            p4_axis_label_fontsize = f2.slider(
                "Axis label (pt)",
                min_value=8,
                max_value=16,
                value=10,
                step=1,
                key="tab8_triad_p4_axis_label_fontsize",
            )
            p4_tick_fontsize = f3.slider(
                "Tick label (pt)",
                min_value=7,
                max_value=14,
                value=9,
                step=1,
                key="tab8_triad_p4_tick_fontsize",
            )
            p4_sig_fontsize = f4.slider(
                "Significance (pt)",
                min_value=7,
                max_value=13,
                value=10,
                step=1,
                key="tab8_triad_p4_sig_fontsize",
                help="Font size for p-value significance labels on brackets.",
            )

            st.markdown("**Title, labels and significance**")
            t1, t2, t3 = st.columns(3)
            p4_show_title = t1.checkbox(
                "Show stratum title above plot",
                value=True,
                key="tab8_triad_p4_show_title",
                help="The title is always placed above the significance brackets, so the two never overlap.",
            )
            p4_xlabel_rotation = t2.slider(
                "X-label tilt (°)",
                min_value=0,
                max_value=90,
                value=25,
                step=1,
                key="tab8_triad_p4_x_rotation",
            )
            p4_show_mean_sem = t3.checkbox(
                "Show mean ± SEM marker",
                value=True,
                key="tab8_triad_p4_show_mean_sem",
                help="This is the black circle/error bar in each method column.",
            )
            t4, t5 = st.columns(2)
            p4_show_significance = t4.checkbox("Show pairwise significance brackets", value=True, key="tab8_triad_p4_show_sig")
            p4_show_ns = t5.checkbox("Include non-significant labels (ns)", value=False, key="tab8_triad_p4_show_ns")

        # ── Y-Axis Options ──────────────────────────────────────────
        with tab_axes:
            st.markdown("**Scale and range**")
            y1, y2 = st.columns([2, 3])
            with y1:
                p4_y_scale = st.selectbox(
                    "Y-axis scale",
                    options=["linear", "log10", "log2"],
                    index=0,
                    key="tab8_triad_p4_y_scale",
                    help="Linear, log₁₀, or log₂ scale. Log scales clip non-positive densities to a small visible floor.",
                )
            with y2:
                st.caption("Auto-adapts range to keep all data and significance labels visible.")

            y3, y4, y5 = st.columns(3)
            p4_manual_y_min = y3.checkbox(
                "Apply minimum value",
                value=False,
                key="tab8_triad_p4_manual_y_min",
                help="Check to override auto y-min."
            )
            p4_y_min_value = y3.number_input(
                "Y min value",
                value=0.0,
                step=10.0,
                format="%.4f",
                key="tab8_triad_p4_y_min_value",
            )
            p4_manual_y_max = y4.checkbox(
                "Apply maximum value",
                value=False,
                key="tab8_triad_p4_manual_y_max",
                help="Check to override auto y-max. (Will auto-expand if needed for visibility.)"
            )
            p4_y_max_value = y4.number_input(
                "Y max value",
                value=2000.0,
                step=10.0,
                format="%.4f",
                key="tab8_triad_p4_y_max_value",
            )
            p4_enable_split_range = y5.checkbox(
                "Use split y-axis",
                value=False,
                key="tab8_triad_p4_enable_split_range",
                help="Show two y-ranges and skip the middle interval.",
            )

            st.divider()
            st.markdown("**Crop tall outliers (auto-zoom on the bulk of the data)**")
            st.caption(
                "Cuts the y-axis below the extreme values and hides those dots, so a handful of very high "
                "connected tissues stop flattening the boxes. Boxes, whiskers and all statistics keep using the full data."
            )
            oc1, oc2, oc3, oc4 = st.columns(4)
            p4_crop_options = ["off", "iqr", "percentile"]
            _init_selectbox_state("tab8_triad_p4_crop_mode", p4_crop_options, preferred="off", fallback=0)
            p4_crop_mode = oc1.selectbox(
                "Outlier cropping",
                options=p4_crop_options,
                key="tab8_triad_p4_crop_mode",
                format_func=lambda x: {
                    "off": "Off (show everything)",
                    "iqr": "Auto (Tukey IQR fence)",
                    "percentile": "Percentile cut",
                }.get(x, x),
            )
            p4_crop_iqr_multiplier = oc2.number_input(
                "IQR multiplier",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.25,
                format="%.2f",
                key="tab8_triad_p4_crop_iqr",
                help="Upper limit = Q3 + k × IQR across all compared methods. 1.5 is the standard Tukey fence.",
                disabled=p4_crop_mode != "iqr",
            )
            p4_crop_percentile = oc3.number_input(
                "Keep up to percentile",
                min_value=50.0,
                max_value=100.0,
                value=99.0,
                step=0.5,
                format="%.1f",
                key="tab8_triad_p4_crop_percentile",
                help="Values above this percentile are cropped from the view.",
                disabled=p4_crop_mode != "percentile",
            )
            p4_crop_annotate = oc4.checkbox(
                "Note cropped points under figure",
                value=True,
                key="tab8_triad_p4_crop_annotate",
                disabled=p4_crop_mode == "off",
                help="Adds a small line below the figure stating how many points were hidden and the true maximum.",
            )

            st.divider()
            st.markdown("**Split y-axis bounds** (only used when *Use split y-axis* is checked)")
            y6, y7, y8, y9 = st.columns(4)
            p4_split_low_min = y6.number_input(
                "Lower range min",
                value=0.0,
                step=10.0,
                format="%.4f",
                key="tab8_triad_p4_split_low_min",
            )
            p4_split_low_max = y7.number_input(
                "Lower range max",
                value=500.0,
                step=10.0,
                format="%.4f",
                key="tab8_triad_p4_split_low_max",
            )
            p4_split_high_min = y8.number_input(
                "Upper range min",
                value=800.0,
                step=10.0,
                format="%.4f",
                key="tab8_triad_p4_split_high_min",
            )
            p4_split_high_max = y9.number_input(
                "Upper range max",
                value=1000.0,
                step=10.0,
                format="%.4f",
                key="tab8_triad_p4_split_high_max",
            )
            p4_split_gap_height = st.number_input(
                "Split gap height",
                min_value=5.0,
                max_value=120.0,
                value=40.0,
                step=1.0,
                format="%.0f",
                key="tab8_triad_p4_split_gap_height",
                help="Visual gap size between lower and upper ranges in plotted units.",
            )

            st.divider()
            st.markdown("**Axis break** (compress, instead of crop, the tall outliers)")
            ab1, ab2, ab3 = st.columns(3)
            p4_enable_axis_break = ab1.checkbox(
                "Enable axis break",
                value=False,
                key="tab8_triad_p4_enable_axis_break",
            )
            p4_break_percentile = ab2.number_input(
                "Outlier percentile",
                min_value=50.0,
                max_value=99.9,
                value=90.0,
                step=1.0,
                format="%.1f",
                key="tab8_triad_p4_break_percentile",
                help="Values above this percentile across all compared methods are visually compressed.",
            )
            p4_break_height = ab3.number_input(
                "Compressed height factor",
                min_value=0.05,
                max_value=0.4,
                value=0.15,
                step=0.05,
                format="%.2f",
                key="tab8_triad_p4_break_height_factor",
                help="Fraction of original plot height allocated for visually compressed outliers mapping.",
            )

        with tab_zonal:
            st.caption("Optional: subset data by a tumor-bed column and/or compute zonal triad-density curves around the tumor boundary.")
            bf1, bf2 = st.columns(2)
            bed_col_options = [""] + [c for c in run_df.columns if c not in set(REQUIRED_COORDS + ["__batch__", "__roi__"])]
            _init_selectbox_state("tab8_triad_p4_bed_col", bed_col_options, preferred="tumor_bed", fallback=0)
            p4_bed_col = bf1.selectbox(
                "Subset column (optional, e.g., tumor_bed)",
                options=bed_col_options,
                format_func=_format_opt,
                key="tab8_triad_p4_bed_col",
            )
            p4_bed_val = bf2.text_input("Subset keep value", value="1", key="tab8_triad_p4_bed_val")

            z1, z2, z3 = st.columns(3)
            tumor_col_options = [""] + [c for c in run_df.columns if c not in set(REQUIRED_COORDS + ["__batch__", "__roi__"])]
            _init_selectbox_state("tab8_triad_p4_tumor_col", tumor_col_options, preferred="tumor", fallback=0)
            p4_tumor_col = z1.selectbox(
                "Tumor annotation column",
                options=tumor_col_options,
                format_func=_format_opt,
                key="tab8_triad_p4_tumor_col",
            )
            p4_tumor_val = z2.text_input("Tumor positive value", value="1", key="tab8_triad_p4_tumor_val")
            p4_run_zonal = z3.checkbox("Run zonal triad-density curves", value=False, key="tab8_triad_p4_run_zonal")

            z4, z5, z6, z7 = st.columns(4)
            p4_zone_width_um = z4.number_input("Zone width (µm)", min_value=5.0, max_value=1000.0, value=40.0, step=5.0, key="tab8_triad_p4_zone_um")
            p4_max_zone_um = z5.number_input("Max distance (µm)", min_value=50.0, max_value=3000.0, value=600.0, step=50.0, key="tab8_triad_p4_max_zone_um")
            p4_min_cells_zone = z6.number_input("Min cells per zone", min_value=3, max_value=100000, value=10, step=1, key="tab8_triad_p4_min_cells_zone")
            p4_min_rois_curve = z7.number_input("Min connected tissues per curve", min_value=1, max_value=1000, value=3, step=1, key="tab8_triad_p4_min_rois")

            if p4_run_zonal:
                st.divider()
                st.markdown("##### Zonal Plot Visual Configuration (Dynamic)")
                cz1, cz2, cz3, cz4 = st.columns(4)
                
                override_x = cz1.checkbox("Override X-axis range", value=False, key="tab8_triad_p4_zonal_override_x")
                cz1.number_input("Zonal X min (µm)", value=-600.0, step=50.0, key="tab8_triad_p4_zonal_x_min", disabled=not override_x)
                cz1.number_input("Zonal X max (µm)", value=600.0, step=50.0, key="tab8_triad_p4_zonal_x_max", disabled=not override_x)
                
                override_y = cz2.checkbox("Override Y-axis range", value=False, key="tab8_triad_p4_zonal_override_y")
                cz2.number_input("Zonal Y min", value=0.0, step=1.0, key="tab8_triad_p4_zonal_y_min", disabled=not override_y)
                cz2.number_input("Zonal Y max", value=500.0, step=50.0, key="tab8_triad_p4_zonal_y_max", disabled=not override_y)
                
                show_leg = cz3.checkbox("Show legend on zonal plot", value=True, key="tab8_triad_p4_zonal_show_legend")
                cz3.selectbox(
                    "Legend location",
                    options=["Best", "Upper Right", "Upper Left", "Lower Right", "Lower Left", "Outside Right", "Outside Bottom"],
                    key="tab8_triad_p4_zonal_legend_loc",
                    disabled=not show_leg,
                )
                cz3.slider(
                    "Legend font size",
                    min_value=4,
                    max_value=20,
                    value=8,
                    step=1,
                    key="tab8_triad_p4_zonal_legend_fontsize",
                    disabled=not show_leg,
                )
                
                zonal_square = cz4.checkbox("Link panel width & height (Square)", value=True, key="tab8_triad_p4_zonal_fig_square")
                if zonal_square:
                    cz4.number_input("Panel size (inches)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key="tab8_triad_p4_zonal_fig_size_linked")
                else:
                    cz4.number_input("Panel width (inches)", min_value=1.0, max_value=20.0, value=6.0, step=0.5, key="tab8_triad_p4_zonal_fig_width")
                    cz4.number_input("Panel height (inches)", min_value=1.0, max_value=20.0, value=5.0, step=0.5, key="tab8_triad_p4_zonal_fig_height")

        out_dir = st.text_input(
            "Panel 4 Output Directory",
            value=str(Path(default_out_dir) / "panel4_stats"),
            key="tab8_triad_p4_outdir",
        )
        sh1, sh2 = st.columns(2)
        p4_show_detail_tables = sh1.checkbox(
            "Show detailed tables in app",
            value=False,
            key="tab8_triad_p4_show_detail_tables",
            help="CSV files are always saved even when hidden here.",
        )
        p4_show_stats_tables = sh2.checkbox(
            "Show stats tables in app",
            value=True,
            key="tab8_triad_p4_show_stats_tables",
        )

        cached_results = st.session_state.get("tab8_triad_p4_cached_results")

        run_panel4 = st.button("Run Panel 4", type="primary", key="tab8_triad_p4_run")
        if run_panel4:
            st.session_state.pop("tab8_triad_p4_publication_bundle", None)
            p4_triad = (str(p4_triad_a), str(p4_triad_b), str(p4_triad_c))
            if len(set(p4_triad)) != 3:
                st.error("Panel 4: Triad phenotype A/B/C must be three distinct values.")
                st.stop()
            if not p4_methods:
                st.error("Panel 4: Select at least one phenotyping method column.")
                st.stop()
            if not p4_metrics:
                st.error("Panel 4: Select at least one metric.")
                st.stop()
            if not style_configs:
                st.error("Panel 4: Select at least one triad style.")
                st.stop()
            for style_name, min_um, max_um in style_configs:
                if float(max_um) <= float(min_um):
                    st.error(f"Panel 4: {style_name} style requires max distance > min distance.")
                    st.stop()

            panel4_df = p4_df.copy()
            panel4_df = panel4_df.dropna(subset=REQUIRED_COORDS)
            if tissue_col != "None" and tissue_col in panel4_df.columns:
                panel4_df = panel4_df[panel4_df[tissue_col].astype(str) == str(tissue_keep_value)].copy()
            if p4_bed_col and p4_bed_col in panel4_df.columns:
                panel4_df = panel4_df[panel4_df[p4_bed_col].astype(str) == str(p4_bed_val)].copy()
            if panel4_df.empty:
                st.warning("Panel 4: No rows left after current filters.")
                st.session_state.pop("tab8_triad_p4_cached_results", None)
                st.stop()

            bed_context = "All data"
            if p4_bed_col:
                bed_context = f"Subset: {p4_bed_col}={p4_bed_val}"
            triad_context = "::".join(p4_triad)

            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            style_dfs = {}
            zone_results = {}

            with st.spinner("Computing Panel 4 metrics..."):
                for style_name, min_um, max_um in style_configs:
                    style_df = _compute_group_triad_metrics(
                        df=panel4_df,
                        methods=[m for m in p4_methods if m in panel4_df.columns],
                        triad=p4_triad,
                        style_name=str(style_name),
                        min_dist_um=float(min_um),
                        max_dist_um=float(max_um),
                        pixel_size_um=float(p4_pixel_size_um),
                        min_cells_per_region=int(p4_min_cells_region),
                        area_method=str(p4_area_method),
                        area_alpha=float(p4_area_alpha),
                        area_use_median=bool(p4_area_use_median),
                        area_percentile=float(p4_area_percentile),
                        min_hole_size=float(p4_min_hole_size),
                        strat_cols=p4_strat_cols,
                    )
                    style_dfs[style_name] = style_df

                    if p4_run_zonal:
                        if not p4_tumor_col:
                            zone_results[style_name] = ("no_tumor_col", None)
                        elif Point is None or MultiPoint is None:
                            zone_results[style_name] = ("no_shapely", None)
                        else:
                            z_detail_df, z_summary_df = _compute_zone_triad_metrics(
                                df=panel4_df,
                                methods=[m for m in p4_methods if m in panel4_df.columns],
                                triad=p4_triad,
                                style_name=str(style_name),
                                min_dist_um=float(min_um),
                                max_dist_um=float(max_um),
                                pixel_size_um=float(p4_pixel_size_um),
                                min_cells_per_region=int(p4_min_cells_region),
                                tumor_col=str(p4_tumor_col),
                                tumor_positive_value=str(p4_tumor_val),
                                zone_width_um=float(p4_zone_width_um),
                                max_zone_distance_um=float(p4_max_zone_um),
                                min_cells_per_zone=int(p4_min_cells_zone),
                                min_rois_per_curve=int(p4_min_rois_curve),
                                area_method=str(p4_area_method),
                                area_alpha=float(p4_area_alpha),
                                area_use_median=bool(p4_area_use_median),
                                area_percentile=float(p4_area_percentile),
                                min_hole_size=float(p4_min_hole_size),
                                strat_cols=p4_strat_cols,
                            )
                            zone_results[style_name] = (z_detail_df, z_summary_df)
                    else:
                        zone_results[style_name] = (None, None)

            cached_results = {
                "style_dfs": style_dfs,
                "zone_results": zone_results,
                "triad_context": triad_context,
                "bed_context": bed_context,
                "out_path_str": str(out_path),
            }
            st.session_state["tab8_triad_p4_cached_results"] = cached_results
            st.rerun()

        if cached_results is None:
            return

        style_dfs = cached_results["style_dfs"]
        zone_results = cached_results["zone_results"]
        triad_context = cached_results["triad_context"]
        bed_context = cached_results["bed_context"]
        out_path = Path(cached_results["out_path_str"])
        panel4_output_files: list[Path] = []

        for style_name, style_df in style_dfs.items():
            if style_df.empty:
                st.warning(f"Panel 4 ({style_name}): no valid connected tissues found.")
                continue

            detail_csv = out_path / f"panel4_{style_name}_group_metrics.csv"
            style_df.to_csv(detail_csv, index=False)
            panel4_output_files.append(detail_csv)
            st.markdown(f"#### Panel 4 ({style_name}) detailed metrics | Triad={triad_context} | {bed_context}")
            if p4_show_detail_tables:
                st.dataframe(style_df.head(500), width="stretch")

            for metric in p4_metrics:
                metric_label = "Triad count" if metric == "triad_count" else "Triad density (triads/mm²)"
                metric_df = style_df.copy()
                if p4_strat_cols:
                    metric_df["__stratum__"] = metric_df[p4_strat_cols].astype(str).agg(" | ".join, axis=1)
                    strata = sorted(metric_df["__stratum__"].dropna().unique().tolist())
                else:
                    metric_df["__stratum__"] = "All"
                    strata = ["All"]

                n_panels = len(strata)
                fig, axes = plt.subplots(
                    1,
                    n_panels,
                    figsize=(float(p4_fig_width) * n_panels, float(p4_fig_height)),
                    squeeze=False,
                )
                axes = axes.flatten()
                fig_crop_notes: list[str] = []
                title_bases: list[float] = []

                stats_rows: list[dict[str, Any]] = []
                for ax, stratum in zip(axes, strata):
                    ax.set_box_aspect(1)  # square panels, matching Density Panel 1
                    sdf = metric_df[metric_df["__stratum__"] == stratum].copy()
                    methods = [m for m in p4_methods if m in sdf["method"].unique().tolist()]
                    if not methods:
                        ax.axis("off")
                        continue

                    raw_data_arrays = [sdf[sdf["method"] == m][metric].to_numpy(dtype=float) for m in methods]
                    raw_data_arrays = [a[np.isfinite(a)] for a in raw_data_arrays]

                    # 1. Apply log scale if needed
                    data_arrays = []
                    log_floor = None
                    panel_y_scale = str(p4_y_scale)
                    if panel_y_scale != "linear":
                        b = _panel1_log_base(panel_y_scale)
                        if b is not None:
                            all_vals = np.concatenate(raw_data_arrays) if any(a.size > 0 for a in raw_data_arrays) else np.array([])
                            pos_vals = all_vals[all_vals > 0]
                            if pos_vals.size > 0:
                                log_floor = float(np.min(pos_vals)) * 0.5
                                for a in raw_data_arrays:
                                    a_fixed = a.copy()
                                    a_fixed[a_fixed <= 0] = log_floor
                                    data_arrays.append(a_fixed)
                            else:
                                panel_y_scale = "linear"
                                data_arrays = raw_data_arrays
                        else:
                            data_arrays = raw_data_arrays
                    else:
                        data_arrays = raw_data_arrays

                    # 2. Setup split ranges
                    split_enabled = bool(p4_enable_split_range)
                    split_params: tuple[float, float, float, float, float] | None = None
                    if split_enabled and panel_y_scale != "linear":
                        panel_y_scale = "linear"
                    if split_enabled:
                        low_min = float(p4_split_low_min)
                        low_max = float(p4_split_low_max)
                        high_min = float(p4_split_high_min)
                        high_max = float(p4_split_high_max)
                        gap_height = float(p4_split_gap_height)
                        if (
                            np.isfinite(low_min) and np.isfinite(low_max)
                            and np.isfinite(high_min) and np.isfinite(high_max)
                            and np.isfinite(gap_height)
                            and low_min < low_max < high_min < high_max
                            and gap_height > 0
                        ):
                            split_params = (low_min, low_max, high_min, high_max, gap_height)
                        else:
                            split_enabled = False

                    plot_data_arrays = data_arrays
                    if split_enabled and split_params is not None:
                        plot_data_arrays = [_panel1_transform_array_for_split(ar, *split_params) for ar in data_arrays]
                        if not any(a.size > 0 for a in plot_data_arrays):
                            split_enabled = False
                            split_params = None
                            plot_data_arrays = data_arrays

                    # 3. Setup axis break
                    use_axis_break, break_threshold, break_max = _panel1_compute_axis_break(
                        plot_data_arrays, (p4_enable_axis_break and not split_enabled), float(p4_break_percentile)
                    )
                    break_height_factor = float(p4_break_height) if (p4_enable_axis_break and not split_enabled) else 0.0

                    # 4. Crop tall outliers out of the view (data and statistics stay complete)
                    crop_info = None
                    if str(p4_crop_mode) != "off" and not split_enabled and not use_axis_break:
                        crop_info = _panel1_compute_outlier_crop(
                            plot_data_arrays,
                            str(p4_crop_mode),
                            float(p4_crop_percentile),
                            float(p4_crop_iqr_multiplier),
                        )

                    x_positions = np.arange(1, len(methods) + 1)

                    if p4_plot_type in {"violin", "box+violin"}:
                        violin = ax.violinplot(
                            plot_data_arrays,
                            positions=x_positions,
                            showmeans=False,
                            showmedians=False,
                            showextrema=False,
                            widths=float(p4_distribution_width),
                        )
                        for i, body in enumerate(violin["bodies"]):
                            method = methods[i]
                            body.set_facecolor(method_color_map.get(method, method_color_defaults[i % len(method_color_defaults)]))
                            body.set_alpha(max(0.12, float(p4_fill_alpha) - 0.1))
                            body.set_edgecolor("#444444")
                            body.set_linewidth(0.8)

                    if p4_plot_type in {"box", "box+violin"}:
                        box = ax.boxplot(
                            plot_data_arrays,
                            labels=[method_label_map.get(m, m) for m in methods],
                            showfliers=False,
                            patch_artist=True,
                            positions=x_positions,
                            widths=float(p4_distribution_width),
                            medianprops={"color": "black", "linewidth": float(p4_box_linewidth)},
                            whiskerprops={"color": "#333333", "linewidth": float(p4_box_linewidth)},
                            capprops={"color": "#333333", "linewidth": float(p4_box_linewidth)},
                            boxprops={"edgecolor": "#333333", "linewidth": float(p4_box_linewidth)},
                        )
                        for i, patch in enumerate(box["boxes"]):
                            method = methods[i]
                            patch.set_facecolor(method_color_map.get(method, method_color_defaults[i % len(method_color_defaults)]))
                            patch.set_alpha(float(p4_fill_alpha))

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
                            jitter = np.random.uniform(-float(p4_jitter_width), float(p4_jitter_width), size=len(y_display))
                            ax.scatter(
                                np.full(len(y_display), i) + jitter,
                                y_display,
                                s=float(p4_dot_size),
                                alpha=float(p4_dot_alpha),
                                c=method_color,
                                edgecolors="white",
                                linewidths=0.25,
                                zorder=4,
                            )

                        raw_y = raw_data_arrays[i - 1]
                        if p4_show_mean_sem and len(raw_y) > 0:
                            mean_y = float(np.nanmean(raw_y))
                            sem_y = float(np.nanstd(raw_y, ddof=1) / np.sqrt(len(raw_y))) if len(raw_y) > 1 else 0.0
                            if np.isfinite(mean_y):
                                mean_disp = mean_y
                                sem_disp = sem_y
                                if split_enabled and split_params is not None:
                                    mean_disp = _panel1_transform_array_for_split(np.array([mean_y]), *split_params)[0]
                                elif use_axis_break:
                                    mean_disp = _panel1_transform_for_break(mean_y, break_threshold, break_height_factor)
                                    sem_disp = sem_y * 0.5
                                ax.errorbar(
                                    [i],
                                    [mean_disp],
                                    yerr=[sem_disp],
                                    fmt="o",
                                    color="#111111",
                                    markersize=4.2,
                                    elinewidth=float(p4_box_linewidth),
                                    capsize=2.5,
                                    zorder=5,
                                )

                    pivot = sdf.pivot_table(index=["batch", "roi"], columns="method", values=metric, aggfunc="first")
                    valid_methods = [m for m in methods if m in pivot.columns]
                    matched_stats = compute_matched_method_statistics(
                        pivot,
                        valid_methods,
                        analysis_unit="batch x tissue_roi",
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

                    pairwise_to_show = pairwise_for_plot
                    if not p4_show_ns:
                        pairwise_to_show = [r for r in pairwise_for_plot if np.isfinite(r.get("p_value", np.nan)) and float(r["p_value"]) < 0.05]
                    
                    y_top = y_max
                    title_base = 1.0
                    if p4_show_significance:
                        y_top = _panel1_add_significance_brackets(
                            ax=ax,
                            methods=methods,
                            pairwise_stats=pairwise_to_show,
                            y_data_max=y_max,
                            y_data_min=y_min,
                            y_scale_mode=panel_y_scale,
                            sig_bracket_linewidth=float(p4_sig_bracket_linewidth),
                            sig_fontsize=int(p4_sig_fontsize),
                            use_axis_break=use_axis_break,
                            break_threshold=break_threshold if use_axis_break else np.nan,
                            break_height=break_height_factor,
                        )
                        title_base = _bracket_top_axes_fraction(methods, pairwise_to_show)

                    axis_y_max = float(crop_info[0]) if crop_info is not None else y_max
                    axis_y_top = float(crop_info[0]) if crop_info is not None else y_top
                    _panel1_apply_y_axis(
                        ax=ax,
                        y_scale_mode=panel_y_scale,
                        y_data_min=y_min,
                        y_data_max=axis_y_max,
                        y_top=axis_y_top,
                        manual_y_min_enabled=bool(p4_manual_y_min),
                        manual_y_min=float(p4_y_min_value),
                        manual_y_max_enabled=bool(p4_manual_y_max),
                        manual_y_max=float(p4_y_max_value),
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

                    title_bases.append(float(title_base))
                    if p4_show_title:
                        # Sits above the significance brackets so the two never overlap.
                        ax.set_title(
                            f"{stratum}",
                            fontsize=int(p4_title_fontsize),
                            y=float(title_base) + 0.012,
                        )
                    y_ax_lbl = metric_label
                    if panel_y_scale == "log10":
                        y_ax_lbl += " | log10 scale"
                    elif panel_y_scale == "log2":
                        y_ax_lbl += " | log2 scale"
                    ax.set_ylabel(y_ax_lbl, fontsize=int(p4_axis_label_fontsize))
                    ax.set_xlabel("Phenotyping Method", fontsize=int(p4_axis_label_fontsize))
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([method_label_map.get(m, m) for m in methods], fontsize=int(p4_tick_fontsize))
                    ax.grid(True, axis="y", alpha=0.18, linestyle="--", linewidth=float(p4_grid_linewidth))
                    ax.tick_params(axis="x", rotation=int(p4_xlabel_rotation), labelsize=int(p4_tick_fontsize))
                    ax.tick_params(axis="y", labelsize=int(p4_tick_fontsize))
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_linewidth(float(p4_axis_linewidth))
                    ax.spines["bottom"].set_linewidth(float(p4_axis_linewidth))

                    if use_axis_break:
                        _panel1_draw_axis_break(ax, break_threshold, break_height_factor)

                    if not matched_stats.empty:
                        matched_stats.insert(0, "stratum", str(stratum))
                        matched_stats.insert(0, "metric", str(metric))
                        matched_stats.insert(0, "style", str(style_name))
                        stats_rows.extend(matched_stats.to_dict("records"))

                plt.tight_layout(w_pad=float(p4_panel_spacing))
                # Place the figure title just above the tallest panel title / bracket stack.
                max_title_base = max(title_bases) if title_bases else 1.0
                panel_tops = [
                    ax.get_position().y0 + ax.get_position().height * (max_title_base + 0.012)
                    for ax in axes
                ]
                # One text line is fontsize/72 inches; leave room for the panel title and the figure title.
                line_fraction = float(p4_title_fontsize) / 72.0 / max(float(fig.get_figheight()), 0.1)
                suptitle_y = (max(panel_tops) if panel_tops else 1.0) + (line_fraction * 2.2) + 0.01
                fig.suptitle(
                    f"Triad={triad_context} | style={style_name} | metric={metric_label} | {bed_context}",
                    fontsize=int(p4_title_fontsize),
                    y=float(suptitle_y),
                )
                if p4_crop_annotate and fig_crop_notes:
                    # Drawn under the figure, where it cannot collide with data or brackets.
                    fig.text(
                        0.99,
                        -0.015,
                        "Cropped view: " + "; ".join(fig_crop_notes),
                        ha="right",
                        va="top",
                        fontsize=max(6, int(p4_tick_fontsize) - 1),
                        color="#444444",
                    )
                fig_path = out_path / f"panel4_{style_name}_{metric}.png"
                _global_dpi = int(st.session_state.get("tab8_density_global_png_dpi", 300))
                fig.savefig(fig_path, dpi=_global_dpi, bbox_inches="tight")
                svg_path = fig_path.with_suffix(".svg")
                fig.savefig(svg_path, bbox_inches="tight")
                panel4_output_files.extend([fig_path, svg_path])
                st.markdown(f"##### Panel 4 Plot: Triad={triad_context} | {style_name} | {metric_label} | {bed_context}")
                st.pyplot(fig)
                st.caption(
                    "Dots: connected tissue units (batch/ROI). Black marker: mean ± SEM. "
                    "Brackets use paired two-sided Wilcoxon signed-rank tests with Holm-adjusted P values."
                )
                plt.close(fig)

                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    stats_csv = out_path / f"panel4_{style_name}_{metric}_stats.csv"
                    stats_df.to_csv(stats_csv, index=False)
                    panel4_output_files.append(stats_csv)
                    if p4_show_stats_tables:
                        st.dataframe(stats_df, width="stretch")

            if p4_run_zonal:
                z_res = zone_results.get(style_name)
                if z_res is None or (isinstance(z_res, tuple) and len(z_res) >= 2 and z_res[0] is None and z_res[1] is None):
                    st.info(f"Panel 4 ({style_name}): Zonal curves not computed. Check 'Run zonal triad-density curves' and click 'Run Panel 4'.")
                elif isinstance(z_res, tuple) and len(z_res) >= 2 and isinstance(z_res[0], str) and z_res[0] == "no_tumor_col":
                    st.warning(f"Panel 4 ({style_name}): tumor column not selected, skipping zonal curves.")
                elif isinstance(z_res, tuple) and len(z_res) >= 2 and isinstance(z_res[0], str) and z_res[0] == "no_shapely":
                    st.warning("Panel 4 zonal analysis requires shapely geometry support.")
                else:
                    zone_detail_df, zone_summary_df = z_res
                    if zone_summary_df.empty:
                        st.warning(f"Panel 4 ({style_name}): no zonal summary results after thresholds.")
                    else:
                        zone_detail_csv = out_path / f"panel4_{style_name}_zonal_detailed.csv"
                        zone_summary_csv = out_path / f"panel4_{style_name}_zonal_summary.csv"
                        zone_detail_df.to_csv(zone_detail_csv, index=False)
                        zone_summary_df.to_csv(zone_summary_csv, index=False)
                        panel4_output_files.extend([zone_detail_csv, zone_summary_csv])

                        zonal_method_stats = compute_zonal_method_statistics(
                            zone_df=zone_detail_df.rename(
                                columns={"group_id": "__group_id__", "triad_density_per_mm2": "__triad_density__"}
                            ),
                            group_cols=["style", "zone_center_um", *p4_strat_cols],
                            block_col="__group_id__",
                            value_col="__triad_density__",
                            methods=[str(method) for method in p4_methods],
                        )
                        if not zonal_method_stats.empty:
                            zonal_stats_csv = out_path / f"panel4_{style_name}_zonal_method_stats.csv"
                            zonal_method_stats.to_csv(zonal_stats_csv, index=False)
                            panel4_output_files.append(zonal_stats_csv)

                        st.markdown(f"##### Panel 4 Zonal curves: Triad={triad_context} | {style_name} | {bed_context} | tumor={p4_tumor_col}={p4_tumor_val}")
                        if p4_show_detail_tables:
                            st.dataframe(zone_summary_df, width="stretch")

                        if p4_strat_cols:
                            zone_summary_df["__stratum__"] = zone_summary_df[p4_strat_cols].astype(str).agg(" | ".join, axis=1)
                            strata = sorted(zone_summary_df["__stratum__"].dropna().unique().tolist())
                        else:
                            zone_summary_df["__stratum__"] = "All"
                            strata = ["All"]

                        # Determine panel sizes
                        if st.session_state.get("tab8_triad_p4_zonal_fig_square", True):
                            zonal_w = float(st.session_state.get("tab8_triad_p4_zonal_fig_size_linked", 5.0))
                            zonal_h = zonal_w
                        else:
                            zonal_w = float(st.session_state.get("tab8_triad_p4_zonal_fig_width", 6.0))
                            zonal_h = float(st.session_state.get("tab8_triad_p4_zonal_fig_height", 5.0))

                        fig_z, axes_z = plt.subplots(1, len(strata), figsize=(zonal_w * len(strata), zonal_h), squeeze=False)
                        axes_z = axes_z.flatten()
                        for ax, stratum in zip(axes_z, strata):
                            if st.session_state.get("tab8_triad_p4_zonal_fig_square", True):
                                ax.set_box_aspect(1)
                            sdf = zone_summary_df[zone_summary_df["__stratum__"] == stratum].copy()
                            for method_idx, method in enumerate(p4_methods):
                                mdf = sdf[sdf["method"] == method].sort_values("zone_center_um")
                                if mdf.empty:
                                    continue
                                x = mdf["zone_center_um"].to_numpy(dtype=float)
                                y = mdf["triad_density_mean"].to_numpy(dtype=float)
                                yerr = mdf["triad_density_sem"].fillna(0).to_numpy(dtype=float)
                                method_color = method_color_map.get(method, method_color_defaults[method_idx % len(method_color_defaults)])
                                ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.8, color=method_color, label=method_label_map.get(method, method))
                                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=method_color)

                            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
                            ax.axvspan(-float(p4_max_zone_um), 0, color="red", alpha=0.08)
                            ax.axvspan(0, float(p4_max_zone_um), color="blue", alpha=0.08)
                            ax.set_xlabel("Distance from tumor boundary (µm)")
                            ax.set_ylabel("Triad density (triads/mm²)")
                            ax.set_title(f"Triad={triad_context} | {style_name} | {stratum} | {bed_context} | tumor={p4_tumor_col}={p4_tumor_val}")
                            ax.grid(True, alpha=0.25)
                            
                            # Apply X-axis overrides
                            if st.session_state.get("tab8_triad_p4_zonal_override_x", False):
                                xmin = float(st.session_state.get("tab8_triad_p4_zonal_x_min", -600.0))
                                xmax = float(st.session_state.get("tab8_triad_p4_zonal_x_max", 600.0))
                                ax.set_xlim(xmin, xmax)
                            
                            # Apply Y-axis overrides
                            if st.session_state.get("tab8_triad_p4_zonal_override_y", False):
                                ymin = float(st.session_state.get("tab8_triad_p4_zonal_y_min", 0.0))
                                ymax = float(st.session_state.get("tab8_triad_p4_zonal_y_max", 500.0))
                                ax.set_ylim(ymin, ymax)
                            
                            # Apply Legend overrides
                            if st.session_state.get("tab8_triad_p4_zonal_show_legend", True):
                                leg_loc_ui = st.session_state.get("tab8_triad_p4_zonal_legend_loc", "Best")
                                leg_font_size = int(st.session_state.get("tab8_triad_p4_zonal_legend_fontsize", 8))
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

                        fig_z.suptitle(
                            f"Zonal triad density | Triad={triad_context} | style={style_name} | {bed_context} | tumor={p4_tumor_col}={p4_tumor_val}",
                            fontsize=11,
                            y=1.02,
                        )
                        plt.tight_layout()
                        fig_z_path = out_path / f"panel4_{style_name}_zonal_density.png"
                        _global_dpi = int(st.session_state.get("tab8_density_global_png_dpi", 300))
                        fig_z.savefig(fig_z_path, dpi=_global_dpi, bbox_inches="tight")
                        fig_z_svg = fig_z_path.with_suffix(".svg")
                        fig_z.savefig(fig_z_svg, bbox_inches="tight")
                        panel4_output_files.extend([fig_z_path, fig_z_svg])
                        st.pyplot(fig_z)
                        plt.close(fig_z)

        existing_bundle = st.session_state.get("tab8_triad_p4_publication_bundle")
        if existing_bundle and Path(str(existing_bundle)).exists():
            bundle_dir = Path(str(existing_bundle))
        else:
            bundle_dir = create_publication_bundle(out_path, "triad_panel4_method_comparison")
            st.session_state["tab8_triad_p4_publication_bundle"] = str(bundle_dir)
        copied_files = copy_outputs_to_bundle(panel4_output_files, bundle_dir)
        nonempty_style_dfs = [frame for frame in style_dfs.values() if isinstance(frame, pd.DataFrame) and not frame.empty]
        n_connected = (
            int(pd.concat(nonempty_style_dfs, ignore_index=True)[["batch", "roi"]].drop_duplicates().shape[0])
            if nonempty_style_dfs
            else 0
        )
        n_cells = (
            int(
                pd.concat(nonempty_style_dfs, ignore_index=True)[["batch", "roi", "n_cells"]]
                .drop_duplicates()["n_cells"]
                .sum()
            )
            if nonempty_style_dfs
            else 0
        )
        write_publication_metadata(
            bundle_dir / "analysis_metadata.json",
            {
                "panel": "Triad Panel 4 - triad density across phenotyping methods",
                "statistics": {
                    "omnibus": "Friedman test across matched connected tissues",
                    "post_hoc": "two-sided paired Wilcoxon signed-rank tests",
                    "multiplicity": "Holm correction across all method pairs within each metric/style/stratum",
                    "effect_sizes": ["Kendall_W", "matched_rank_biserial"],
                    "significance_brackets_use": "p_value_adjusted",
                    "zonal_method_tests": "exported with Holm correction across method-pair-by-zone family",
                },
                "analysis": {
                    "methods": [str(method) for method in p4_methods],
                    "triad": [str(value) for value in triad_context.split("::")],
                    "triad_definition": "one cell of each phenotype; all three pairwise centroid distances within the selected range",
                    "cells_may_participate_in_multiple_triads": True,
                    "styles": [
                        {"name": str(name), "minimum_distance_um": float(minimum), "maximum_distance_um": float(maximum)}
                        for name, minimum, maximum in style_configs
                    ],
                    "n_connected_tissues": n_connected,
                    "n_cells_across_connected_tissues": n_cells,
                    "pixel_size_um": float(p4_pixel_size_um),
                    "minimum_cells_per_connected_tissue": int(p4_min_cells_region),
                    "tissue_filter": {"column": str(tissue_col), "value": str(tissue_keep_value)},
                    "tumor_bed_filter": {"column": str(p4_bed_col), "value": str(p4_bed_val)},
                    "additional_filters": selected_filter_cols,
                    "virtual_area_method": str(p4_area_method),
                    "nearest_neighbor_buffer_percentile": float(p4_area_percentile),
                    "minimum_hole_size_px2": float(p4_min_hole_size),
                    "stratification_columns": p4_strat_cols,
                },
                "display": {
                    "plot_type": p4_plot_type,
                    "y_scale": p4_y_scale,
                    "outlier_crop_mode": p4_crop_mode,
                    "outlier_crop_affects_statistics": False,
                    "png_dpi": int(st.session_state.get("tab8_density_global_png_dpi", 300)),
                },
                "files": copied_files,
            },
        )
        st.success("Panel 4 complete.")
        st.write(
            {
                "Panel4_output_dir": str(out_path),
                "Publication_bundle": str(bundle_dir),
                "Triad": triad_context,
                "Subset": bed_context,
            }
        )


def _render_panel6_results(panel_state: dict):
    st.info("Showing the most recent Panel 6 run.")
    runtime_s = panel_state.get("runtime_seconds")
    if runtime_s is not None:
        st.caption(f"Last Panel 6 runtime: {float(runtime_s):.1f}s")
    summary_df = panel_state.get("summary_df")
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        st.dataframe(summary_df, width="stretch")
    stats_df = panel_state.get("stats_df")
    if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
        st.markdown("#### Panel 6 Statistics")
        st.dataframe(stats_df, width="stretch")
    method_stats_df = panel_state.get("method_stats_df")
    if isinstance(method_stats_df, pd.DataFrame) and not method_stats_df.empty:
        st.markdown("#### Matched method statistics by distance zone")
        st.dataframe(method_stats_df, width="stretch")
    plot_path = panel_state.get("plot_path")
    if plot_path and Path(str(plot_path)).exists():
        st.markdown("#### Panel 6 Plot")
        st.image(str(plot_path), width="stretch")
    st.write(
        {
            "Panel6_detailed_csv": panel_state.get("detailed_csv"),
            "Panel6_summary_csv": panel_state.get("summary_csv"),
            "Rows_detailed": panel_state.get("rows_detailed"),
            "Rows_summary": panel_state.get("rows_summary"),
            "Publication_bundle": panel_state.get("publication_bundle"),
        }
    )


@st.fragment
def _render_panel6_zonal_triads(
    run_df: pd.DataFrame,
    phenotype_candidates: list[str],
    shared: dict[str, Any],
    triad: tuple[str, str, str],
    local_min_um: float,
    local_max_um: float,
    neigh_min_um: float,
    neigh_max_um: float,
    pixel_size_um: float,
    min_cells_region: int,
    default_out_dir: str,
):
    """Triad density by distance from an annotation (clone of Density Panel 2)."""

    with st.expander("**Panel 6 · Triad Density by Distance from an Annotation**", expanded=False):
        st.caption(
            "Profiles triad density in distance zones around a reference boundary "
            "(negative = inside, positive = outside), one curve per phenotyping method. "
            "Edits stay local until you click Run Panel 6."
        )

        method_color_map: dict[str, str] = dict(shared.get("colors", {}))
        method_label_map: dict[str, str] = dict(shared.get("labels", {}))
        method_color_defaults = list(shared.get("color_defaults", [])) or [
            "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2",
        ]
        p6_methods = [str(m) for m in shared.get("methods", []) if str(m) in phenotype_candidates]

        meta_cols = [c for c in run_df.columns if c not in set(REQUIRED_COORDS + ["__batch__", "__roi__"])]

        tab_data, tab_zones, tab_style, tab_filters = st.tabs(
            [
                "1 · What to measure",
                "2 · Zones & statistics",
                "3 · Plot style",
                "4 · Filters & stratum colors",
            ]
        )

        with tab_data:
            st.markdown("**Triad to profile**")
            if p6_methods:
                st.caption(
                    "One curve per phenotyping method, taken from Shared Figure Settings: "
                    + ", ".join(f"{method_label_map.get(m, m)} ({m})" for m in p6_methods)
                )
            else:
                st.warning("Select at least one phenotyping method column in Shared Figure Settings.")

            p6_pool: set[str] = set()
            for mcol in p6_methods:
                if mcol in run_df.columns:
                    p6_pool.update(run_df[mcol].dropna().astype(str).unique().tolist())
            p6_pheno_pool = sorted(p6_pool)
            tri1, tri2, tri3 = st.columns(3)
            _init_selectbox_state("tab8_triad_p6_a", p6_pheno_pool, preferred=str(triad[0]), fallback=0)
            _init_selectbox_state("tab8_triad_p6_b", p6_pheno_pool, preferred=str(triad[1]), fallback=1)
            _init_selectbox_state("tab8_triad_p6_c", p6_pheno_pool, preferred=str(triad[2]), fallback=2)
            p6_a = tri1.selectbox("Triad phenotype A (anchor)", options=p6_pheno_pool, key="tab8_triad_p6_a")
            p6_b = tri2.selectbox("Triad phenotype B", options=p6_pheno_pool, key="tab8_triad_p6_b")
            p6_c = tri3.selectbox("Triad phenotype C", options=p6_pheno_pool, key="tab8_triad_p6_c")

            st1, st2, st3 = st.columns(3)
            p6_style_options = ["local", "neighborhood"]
            _init_selectbox_state("tab8_triad_p6_style", p6_style_options, preferred="local", fallback=0)
            p6_style = st1.selectbox(
                "Triad distance style",
                options=p6_style_options,
                key="tab8_triad_p6_style",
                format_func=lambda x: "Direct-contact / local" if x == "local" else "Neighborhood annulus",
            )
            p6_min_um = st2.number_input(
                "Triad min distance (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=float(local_min_um if p6_style == "local" else neigh_min_um),
                step=1.0,
                key="tab8_triad_p6_triad_min_um",
            )
            p6_max_um = st3.number_input(
                "Triad max distance (µm)",
                min_value=0.1,
                max_value=10000.0,
                value=float(local_max_um if p6_style == "local" else neigh_max_um),
                step=1.0,
                key="tab8_triad_p6_triad_max_um",
                help="A triad needs all three pairwise distances inside this range.",
            )

            st.markdown("**Reference boundary (distance = 0)**")
            r1, r2 = st.columns(2)
            annotation_options = [""] + meta_cols
            _init_selectbox_state(
                "tab8_triad_p6_annotation_col",
                annotation_options,
                preferred="tumor_bed" if "tumor_bed" in annotation_options else ("tumor" if "tumor" in annotation_options else ""),
                fallback=0,
            )
            p6_annotation_col = r1.selectbox(
                "Reference annotation column",
                options=annotation_options,
                format_func=_format_opt,
                key="tab8_triad_p6_annotation_col",
                help="Boundary reference source, e.g. tumor_bed or a neighborhood/annotation column.",
            )
            p6_ref_value_options = []
            if p6_annotation_col and p6_annotation_col in run_df.columns:
                p6_ref_value_options = sorted(run_df[p6_annotation_col].dropna().astype(str).unique().tolist())
            p6_reference_values = r2.multiselect(
                "Reference annotation value(s)",
                options=p6_ref_value_options,
                key="tab8_triad_p6_ref_values",
                help="Cells matching these value(s) define the boundary when distance is computed from annotation.",
            )

            numeric_distance_candidates = [
                c for c in meta_cols
                if ("distance" in c.lower()) or pd.api.types.is_numeric_dtype(run_df[c])
            ]
            distance_options = ["None (compute from annotation)"] + numeric_distance_candidates
            _init_selectbox_state(
                "tab8_triad_p6_distance_col", distance_options, preferred="None (compute from annotation)", fallback=0
            )
            p6_distance_col = st.selectbox(
                "Or use a pre-computed signed distance column",
                options=distance_options,
                key="tab8_triad_p6_distance_col",
                help="If selected, this column is used directly as signed distance (negative inside, positive outside).",
            )

            st.markdown("**Restrict cells before measuring (optional)**")
            pf1, pf2 = st.columns(2)
            prefilter_options = ["None"] + meta_cols
            _init_selectbox_state(
                "tab8_triad_p6_prefilter_col",
                prefilter_options,
                preferred="tissue" if "tissue" in prefilter_options else "None",
                fallback=0,
            )
            p6_prefilter_col = pf1.selectbox(
                "Prefilter column",
                options=prefilter_options,
                key="tab8_triad_p6_prefilter_col",
                help="Optional first-pass filter. Example: choose tissue then keep value 1.",
            )
            p6_prefilter_value_options = []
            if p6_prefilter_col != "None" and p6_prefilter_col in run_df.columns:
                p6_prefilter_value_options = sorted(run_df[p6_prefilter_col].dropna().astype(str).unique().tolist())
            p6_prefilter_values = pf2.multiselect(
                "Prefilter value(s) to keep",
                options=p6_prefilter_value_options,
                key="tab8_triad_p6_prefilter_values",
                help="Leave empty to keep all values from the prefilter column.",
            )

        with tab_zones:
            st.markdown("**Distance zones**")
            z1, z2, z3 = st.columns(3)
            p6_zone_min_um = z1.number_input(
                "Min distance (µm)",
                min_value=-3000.0,
                max_value=3000.0,
                value=-600.0,
                step=50.0,
                key="tab8_triad_p6_min_um",
                help="Minimum distance around the reference boundary (negative inside, positive outside).",
            )
            p6_zone_max_um = z2.number_input(
                "Max distance (µm)",
                min_value=-3000.0,
                max_value=3000.0,
                value=600.0,
                step=50.0,
                key="tab8_triad_p6_max_um",
                help="Maximum distance around the reference boundary (negative inside, positive outside).",
            )
            p6_zone_width_um = z3.number_input(
                "Zone width (µm)",
                min_value=5.0,
                max_value=1000.0,
                value=40.0,
                step=5.0,
                key="tab8_triad_p6_zone_um",
                help="Bin width for distance zones.",
            )

            st.markdown("**Inclusion thresholds**")
            t1, t2, t3 = st.columns(3)
            p6_min_cells_zone = t1.number_input(
                "Min cells per zone",
                min_value=0,
                max_value=100000,
                value=10,
                step=1,
                key="tab8_triad_p6_min_cells_zone",
                help="Zones with fewer cells than this are excluded.",
            )
            p6_min_rois = t2.number_input(
                "Min connected tissues per curve",
                min_value=1,
                max_value=1000,
                value=3,
                step=1,
                key="tab8_triad_p6_min_rois",
                help="How many connected tissues must contribute to each zone point.",
            )
            p6_percentile = t3.slider(
                "Zone area buffer percentile",
                min_value=90.0,
                max_value=99.9,
                value=99.0,
                step=0.1,
                key="tab8_triad_p6_percentile",
            )
            st.caption("Zone areas use the same connected virtual tissue geometry as the Density subtab.")

            st.markdown("**Stratification and statistics**")
            p6_strat_cols = st.multiselect(
                "Split/overlay curves by (optional stratification columns)",
                options=meta_cols,
                default=[],
                key="tab8_triad_p6_strat_cols",
                help="Curves are split/overlaid by these metadata strata.",
            )
            g2, g3, g4 = st.columns(3)
            p6_overlay_strata = g2.checkbox(
                "Overlay stratified curves",
                value=True,
                key="tab8_triad_p6_overlay_strata",
                help="When stratification columns are selected, overlay strata in the same panel instead of splitting them.",
            )
            p6_show_stats = g3.checkbox(
                "Add zone-level stats",
                value=True,
                key="tab8_triad_p6_show_stats",
                help="Compare stratified curves at each distance zone and annotate significant differences on the plot.",
            )
            p6_stats_alpha = g4.number_input(
                "Stats alpha",
                min_value=0.001,
                max_value=0.2,
                value=0.05,
                step=0.005,
                format="%.3f",
                key="tab8_triad_p6_stats_alpha",
            )
            g5, g6 = st.columns(2)
            p6_overlay_encoding = g5.selectbox(
                "Overlay encoding",
                options=["color_by_stratum", "line_style_by_stratum"],
                index=0,
                key="tab8_triad_p6_overlay_encoding",
                format_func=lambda x: "Distinct colors by stratum" if x == "color_by_stratum" else "Line style by stratum",
            )
            p6_stats_method = g6.selectbox(
                "Stats test for stratified comparison",
                options=["welch_t_test", "mannwhitney_u"],
                index=0,
                key="tab8_triad_p6_stats_method",
                format_func=lambda x: "Welch t-test" if x == "welch_t_test" else "Mann-Whitney U",
                help="Used only when stratified overlay/statistics are enabled. Compares two strata per zone.",
            )

        with tab_style:
            render_zonal_plot_controls(
                "tab8_triad_p6",
                note="Line colors and display names come from Shared Figure Settings at the top of the subtab.",
            )

        with tab_filters:
            p6_filter_cols = st.multiselect(
                "Keep only rows matching (optional Panel 6 filters)",
                options=meta_cols,
                default=[],
                key="tab8_triad_p6_filter_cols",
                help="Additional filters applied before distance and density calculation.",
            )
            p6_df = _apply_value_filters(run_df, p6_filter_cols, key_prefix="tab8_triad_p6")
            preview_df = p6_df
            if p6_prefilter_col != "None" and p6_prefilter_col in preview_df.columns and p6_prefilter_values:
                preview_df = preview_df[preview_df[p6_prefilter_col].astype(str).isin([str(v) for v in p6_prefilter_values])]
            preview_strata: list[str] = []
            if p6_strat_cols and not preview_df.empty:
                preview_strata = sorted(
                    preview_df[p6_strat_cols].astype(str).apply(lambda row: " | ".join(row), axis=1).dropna().unique().tolist()
                )
            p6_stratum_color_map: dict[str, str] = {}
            if preview_strata and p6_overlay_strata and p6_overlay_encoding == "color_by_stratum":
                st.markdown("**Current stratum colors**")
                st.caption("These labels come from the currently selected stratification columns and active Panel 6 filters.")
                stratum_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#e377c2"]
                color_cols = st.columns(min(3, len(preview_strata)))
                for idx, stratum_name in enumerate(preview_strata):
                    with color_cols[idx % len(color_cols)]:
                        p6_stratum_color_map[stratum_name] = st.color_picker(
                            f"{stratum_name}",
                            value=stratum_palette[idx % len(stratum_palette)],
                            key=f"tab8_triad_p6_stratum_color_{stratum_name}",
                        )

        p6_out_dir = st.text_input(
            "Panel 6 Output Directory",
            value=str(Path(default_out_dir) / "panel6_zonal_triads"),
            key="tab8_triad_p6_outdir",
        )

        col_btns = st.columns([1, 1])
        run_panel6 = col_btns[0].button("Run Panel 6", type="primary", key="tab8_triad_p6_run", width="stretch")
        save_panel6 = col_btns[1].button("Save Parameters", type="secondary", key="tab8_triad_p6_save", width="stretch")

        if save_panel6:
            from spacemap_gui.persistence import save_active_project_state
            save_active_project_state("tab8_analysis")
            st.success("Project settings saved to configuration file.")

        p6_triad = (str(p6_a), str(p6_b), str(p6_c))
        if run_panel6:
            use_distance_col = p6_distance_col != "None (compute from annotation)"
            if not p6_methods:
                st.error("Panel 6: Select at least one phenotyping method column in Shared Figure Settings.")
            elif len(set(p6_triad)) != 3:
                st.error("Panel 6: Triad phenotypes A/B/C must be three distinct values.")
            elif float(p6_max_um) <= float(p6_min_um):
                st.error("Panel 6: Triad max distance must be greater than triad min distance.")
            elif (not use_distance_col) and (not p6_annotation_col or not p6_reference_values):
                st.error("Panel 6: Select an annotation column and at least one reference value, or choose a signed distance column.")
            elif float(p6_zone_min_um) >= float(p6_zone_max_um):
                st.error("Panel 6: Min distance (µm) must be strictly less than Max distance (µm).")
            else:
                panel6_started = perf_counter()
                with st.status("Running Panel 6...", expanded=True) as status:
                    _log_terminal_event(f"Panel 6 started with triad={p6_triad}, annotation={p6_annotation_col}")
                    status.write("Applying panel-specific filters.")
                    work_df = p6_df.dropna(subset=REQUIRED_COORDS)
                    if p6_prefilter_col != "None" and p6_prefilter_col in work_df.columns and p6_prefilter_values:
                        work_df = work_df[work_df[p6_prefilter_col].astype(str).isin([str(v) for v in p6_prefilter_values])]

                    if work_df.empty:
                        status.update(label="Panel 6 finished with no rows", state="error")
                        st.warning("Panel 6: No rows left after filters.")
                    else:
                        status.write("Building zonal triad densities per connected tissue.")
                        zone_df, summary_df = _compute_zonal_triad_density(
                            df=work_df,
                            methods=p6_methods,
                            triad=p6_triad,
                            min_dist_um=float(p6_min_um),
                            max_dist_um=float(p6_max_um),
                            pixel_size_um=float(pixel_size_um),
                            annotation_col=str(p6_annotation_col),
                            reference_values=[str(v) for v in p6_reference_values],
                            distance_col=str(p6_distance_col) if use_distance_col else "",
                            zone_min_um=float(p6_zone_min_um),
                            zone_max_um=float(p6_zone_max_um),
                            zone_width_um=float(p6_zone_width_um),
                            min_cells_group=int(min_cells_region),
                            min_cells_per_zone=int(p6_min_cells_zone),
                            min_rois_per_curve=int(p6_min_rois),
                            area_method="buffer_union",
                            area_alpha=100.0,
                            area_use_median=False,
                            area_percentile=float(p6_percentile),
                            min_hole_size=10000.0,
                            strat_cols=[str(c) for c in p6_strat_cols],
                        )

                        if zone_df.empty or summary_df.empty:
                            status.update(label="Panel 6 finished with no valid zones", state="error")
                            st.warning(
                                "Panel 6: No valid zone-level results. Consider relaxing the zone thresholds, "
                                "the minimum connected tissues, or the triad distance range."
                            )
                        else:
                            out_dir = Path(p6_out_dir)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            triad_label = "-".join(p6_triad)
                            zone_csv = out_dir / f"panel6_{triad_label}_zonal_triad_density_detailed.csv"
                            summary_csv = out_dir / f"panel6_{triad_label}_zonal_triad_density_summary.csv"
                            zone_df.to_csv(zone_csv, index=False)
                            summary_df.to_csv(summary_csv, index=False)
                            _log_terminal_event(f"Panel 6 wrote {zone_csv} and {summary_csv}")
                            output_files: list[Path] = [zone_csv, summary_csv]

                            method_stats_df = compute_zonal_method_statistics(
                                zone_df=zone_df,
                                group_cols=["phenotype", "zone_center_um", *p6_strat_cols],
                                block_col="__group_id__",
                                value_col="density_cells_per_mm2",
                                methods=[str(method) for method in p6_methods],
                            )
                            if not method_stats_df.empty:
                                method_stats_csv = out_dir / f"panel6_{triad_label}_zonal_method_stats.csv"
                                method_stats_df.to_csv(method_stats_csv, index=False)
                                output_files.append(method_stats_csv)

                            stats_df = pd.DataFrame()
                            if p6_show_stats and p6_strat_cols:
                                status.write("Computing stratified zone-level statistics.")
                                stats_df = _compute_panel2_stats(
                                    zone_df=zone_df,
                                    strat_cols=[str(c) for c in p6_strat_cols],
                                    stats_method=str(p6_stats_method),
                                    alpha=float(p6_stats_alpha),
                                )
                                if not stats_df.empty:
                                    stats_csv = out_dir / f"panel6_{triad_label}_zonal_triad_density_stats.csv"
                                    stats_df.to_csv(stats_csv, index=False)
                                    output_files.append(stats_csv)

                            plot_df = summary_df.copy()
                            if p6_strat_cols:
                                plot_df["__stratum__"] = plot_df[p6_strat_cols].astype(str).agg(" | ".join, axis=1)
                                strata = sorted(plot_df["__stratum__"].dropna().unique().tolist())
                            else:
                                plot_df["__stratum__"] = "All"
                                strata = ["All"]
                            # Keep the shared method order so the legend matches the panels.
                            present_methods = set(plot_df["method"].dropna().astype(str).unique().tolist())
                            plot_methods = [str(m) for m in p6_methods if str(m) in present_methods]
                            plot_methods += sorted(present_methods - set(plot_methods))
                            overlay_active = bool(p6_overlay_strata and p6_strat_cols)
                            triad_display = "::".join(p6_triad)

                            fig = render_zonal_density_figure(
                                summary_df=plot_df,
                                stats_df=stats_df,
                                plot_methods=plot_methods,
                                strat_cols=[str(c) for c in p6_strat_cols],
                                strata=strata,
                                overlay_active=overlay_active,
                                overlay_encoding=str(p6_overlay_encoding),
                                show_stats=bool(p6_show_stats),
                                stratum_color_map=p6_stratum_color_map,
                                method_color_map=method_color_map,
                                method_label_map=method_label_map,
                                method_color_defaults=method_color_defaults,
                                min_distance_um=float(p6_zone_min_um),
                                max_distance_um=float(p6_zone_max_um),
                                title_for_stratum=lambda stratum: (
                                    f"{triad_display} | overlaid strata" if overlay_active else f"{triad_display} | {stratum}"
                                ),
                                y_label="Triad density (triads/mm²)",
                                key_prefix="tab8_triad_p6",
                            )
                            fig_path = out_dir / f"panel6_{triad_label}_zonal_triad_density.png"
                            _global_dpi = int(st.session_state.get("tab8_density_global_png_dpi", 300))
                            fig.savefig(fig_path, dpi=_global_dpi, bbox_inches="tight")
                            svg_path = fig_path.with_suffix(".svg")
                            fig.savefig(svg_path, bbox_inches="tight")
                            output_files.extend([fig_path, svg_path])
                            st.markdown(f"#### {triad_display} triad density by distance")
                            st.pyplot(fig)
                            plt.close(fig)

                            bundle_dir = create_publication_bundle(out_dir, "triad_panel6_zonal")
                            copied_files = copy_outputs_to_bundle(output_files, bundle_dir)
                            write_publication_metadata(
                                bundle_dir / "analysis_metadata.json",
                                {
                                    "panel": "Triad Panel 6 - triad density by boundary distance",
                                    "statistics": {
                                        "method_comparison_omnibus": "Friedman test within each distance zone",
                                        "method_comparison_post_hoc": "two-sided paired Wilcoxon signed-rank tests",
                                        "method_comparison_multiplicity": (
                                            "Holm correction across the full method-pair-by-zone family"
                                        ),
                                        "stratified_comparison": str(p6_stats_method) if p6_strat_cols else "not run",
                                        "stratified_multiplicity": "Holm across zones within method and stratum pair",
                                        "curve_summary": "mean +/- SEM across connected tissues",
                                        "curve_annotations": "method tests exported to CSV; not drawn to preserve figure readability",
                                    },
                                    "analysis": {
                                        "methods": [str(method) for method in p6_methods],
                                        "triad": [str(value) for value in p6_triad],
                                        "triad_definition": (
                                            "one cell of each phenotype; all three pairwise centroid distances within the selected range"
                                        ),
                                        "cells_may_participate_in_multiple_triads": True,
                                        "triad_minimum_distance_um": float(p6_min_um),
                                        "triad_maximum_distance_um": float(p6_max_um),
                                        "distance_source": (
                                            str(p6_distance_col)
                                            if use_distance_col
                                            else {
                                                "annotation_column": str(p6_annotation_col),
                                                "reference_values": [str(value) for value in p6_reference_values],
                                                "geometry": "convex hull boundary of reference-positive cell centroids",
                                            }
                                        ),
                                        "distance_min_um": float(p6_zone_min_um),
                                        "distance_max_um": float(p6_zone_max_um),
                                        "zone_width_um": float(p6_zone_width_um),
                                        "minimum_cells_per_zone": int(p6_min_cells_zone),
                                        "minimum_connected_tissues_per_curve": int(p6_min_rois),
                                        "connected_tissues_per_point_min": int(summary_df["n_connected_tissues"].min()),
                                        "connected_tissues_per_point_max": int(summary_df["n_connected_tissues"].max()),
                                        "pixel_size_um": float(pixel_size_um),
                                        "virtual_area_method": "buffer_union",
                                        "nearest_neighbor_buffer_percentile": float(p6_percentile),
                                        "minimum_hole_size_px2": 10000.0,
                                        "prefilter": {
                                            "column": str(p6_prefilter_col),
                                            "values": [str(value) for value in p6_prefilter_values],
                                        },
                                        "additional_filter_columns": p6_filter_cols,
                                        "stratification_columns": p6_strat_cols,
                                    },
                                    "display": {
                                        "png_dpi": int(_global_dpi),
                                        "figures_unchanged_by_statistical_export": True,
                                    },
                                    "files": copied_files,
                                },
                            )

                            st.session_state["tab8_triad_p6_results"] = {
                                "summary_df": summary_df,
                                "stats_df": stats_df,
                                "method_stats_df": method_stats_df,
                                "detailed_csv": str(zone_csv),
                                "summary_csv": str(summary_csv),
                                "rows_detailed": int(len(zone_df)),
                                "rows_summary": int(len(summary_df)),
                                "plot_path": str(fig_path),
                                "runtime_seconds": float(perf_counter() - panel6_started),
                                "publication_bundle": str(bundle_dir),
                            }
                            status.update(label="Panel 6 complete", state="complete")
                            st.success(f"Panel 6 calculation complete. Saved to {out_dir}")
        elif st.session_state.get("tab8_triad_p6_results"):
            _render_panel6_results(st.session_state["tab8_triad_p6_results"])


def _render_panel5_results(panel_state: dict):
    st.info("Showing the most recent Panel 5 run.")
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
        st.write({"Panel5_output_dir": panel_state.get("output_dir"), "Panel5_files": saved})


@st.fragment
def _render_panel5_triad_density_cloud(
    project_root: Path,
    run_df: pd.DataFrame,
    phenotype_candidates: list[str],
    shared: dict[str, Any],
    triad: tuple[str, str, str],
    local_min_um: float,
    local_max_um: float,
    neigh_min_um: float,
    neigh_max_um: float,
    pixel_size_um: float,
    max_saved_triads: int,
    default_out_dir: str,
):
    """Figure 2E clone: triad-centroid density clouds on the annotated tissue."""

    with st.expander("**Panel 5 · Triad Density Cloud**", expanded=False):
        st.caption(
            "Triad maps in the published Figure 2E style: annotation layers from the merged GeoJSON files, "
            "a white outline around the selected tissue component and a Gaussian-smoothed cloud of triad "
            "centroids, drawn on one shared color scale across phenotyping methods."
        )

        from spacemap_gui.ui_utils import directory_input_with_browse

        method_label_map: dict[str, str] = dict(shared.get("labels", {}))
        p5_methods = [str(m) for m in shared.get("methods", []) if str(m) in phenotype_candidates]

        tab_data, tab_layers, tab_cloud, tab_layout = st.tabs(
            [
                "1 · Tissue & triad",
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
                text_key="tab8_triad_p5_annot_dir",
                default_value=str(
                    st.session_state.get(
                        "tab8_triad_p5_annot_dir",
                        st.session_state.get(
                            "tab8_density_p3_annot_dir",
                            project_root / "outputs" / "processed_annotations",
                        ),
                    )
                ),
                browser_key_prefix="tab8_triad_p5_annot_dir",
                help_text="Folder holding one merged GeoJSON per ROI image, e.g. P11_roi_1_merged.geojson.",
            )

            a1, a2 = st.columns(2)
            file_pattern = a1.text_input(
                "File name pattern",
                value=str(st.session_state.get("tab8_triad_p5_file_pattern", clouds.DEFAULT_FILE_PATTERN)),
                key="tab8_triad_p5_file_pattern",
                help="Use {batch} as the placeholder for the batch ID; used to preselect the file for the chosen batch.",
            )

            batch_values = sorted(run_df["__batch__"].dropna().astype(str).unique().tolist()) or ["ALL"]
            _init_selectbox_state("tab8_triad_p5_batch", batch_values, preferred=batch_values[0], fallback=0)
            p5_batch = a2.selectbox(
                "Batch ID (ROI image)",
                options=batch_values,
                key="tab8_triad_p5_batch",
                help="All cells of this batch are used; pick a single tissue ROI below to zoom into one component.",
            )

            geojson_files: list[str] = []
            annot_dir_path = Path(str(annot_dir)) if annot_dir else None
            if annot_dir_path is not None and annot_dir_path.is_dir():
                geojson_files = sorted(p.name for p in annot_dir_path.iterdir() if p.suffix.lower() == ".geojson")
            if not geojson_files:
                st.warning("No .geojson files found in the annotation directory.")
            expected_name = clouds.annotation_path("", str(p5_batch), str(file_pattern)).name
            file_options = geojson_files or [""]
            # Follow the batch selection: switching batch re-points the file to the matching GeoJSON.
            last_batch_key = "_tab8_triad_p5_last_batch"
            if st.session_state.get(last_batch_key) != str(p5_batch):
                st.session_state[last_batch_key] = str(p5_batch)
                if expected_name in file_options:
                    st.session_state["tab8_triad_p5_geojson_file"] = expected_name
            _init_selectbox_state(
                "tab8_triad_p5_geojson_file",
                file_options,
                preferred=expected_name if expected_name in file_options else file_options[0],
                fallback=0,
            )
            p5_geojson_file = st.selectbox(
                "GeoJSON file for this batch",
                options=file_options,
                key="tab8_triad_p5_geojson_file",
                help=f"Auto-matched from the pattern above ({expected_name}); override here if the file is named differently.",
            )

            # Layers differ between ROI images, so a new file starts from that file's defaults.
            last_file_key = "_tab8_triad_p5_last_file"
            if last_file_key in st.session_state and st.session_state[last_file_key] != str(p5_geojson_file):
                st.session_state.pop("tab8_triad_p5_layers", None)
            st.session_state[last_file_key] = str(p5_geojson_file)

            if annot_dir_path is not None and p5_geojson_file:
                geojson_path = annot_dir_path / str(p5_geojson_file)
                if geojson_path.exists():
                    try:
                        annotations = clouds.load_annotation_geometries(str(geojson_path), geojson_path.stat().st_mtime)
                    except Exception as error:  # pragma: no cover - depends on user files
                        st.error(f"Could not read {geojson_path.name}: {error}")
            layer_names = sorted(annotations)
            if layer_names:
                st.caption(f"Layers in {p5_geojson_file}: {', '.join(layer_names)}")

            st.markdown("**Tissue component**")
            t1, t2 = st.columns(2)
            tissue_options = layer_names or [clouds.DEFAULT_TISSUE_LAYER]
            _init_selectbox_state(
                "tab8_triad_p5_tissue_layer",
                tissue_options,
                preferred=clouds.DEFAULT_TISSUE_LAYER if clouds.DEFAULT_TISSUE_LAYER in tissue_options else tissue_options[0],
                fallback=0,
            )
            p5_tissue_layer = t1.selectbox(
                "Tissue outline layer",
                options=tissue_options,
                key="tab8_triad_p5_tissue_layer",
                help="Layer whose polygons define the tissue outline and the crop of the map.",
            )

            roi_options = ["(whole tissue)"] + sorted(
                run_df[run_df["__batch__"].astype(str) == str(p5_batch)]["__roi__"].dropna().astype(str).unique().tolist()
            )
            _init_selectbox_state("tab8_triad_p5_roi", roi_options, preferred=roi_options[0], fallback=0)
            p5_roi = t2.selectbox(
                "Tissue ROI",
                options=roi_options,
                key="tab8_triad_p5_roi",
                help="Pick one ROI to draw only the tissue component holding those cells, like the reference figure.",
            )

            st.markdown("**Triad to map**")
            if p5_methods:
                st.caption(
                    "One map per phenotyping method, taken from Shared Figure Settings: "
                    + ", ".join(f"{method_label_map.get(m, m)} ({m})" for m in p5_methods)
                )
            else:
                st.warning("Select at least one phenotyping method column in Shared Figure Settings.")

            p5_pool: set[str] = set()
            for mcol in p5_methods:
                if mcol in run_df.columns:
                    p5_pool.update(run_df[mcol].dropna().astype(str).unique().tolist())
            p5_pheno_pool = sorted(p5_pool)
            tri1, tri2, tri3 = st.columns(3)
            _init_selectbox_state("tab8_triad_p5_a", p5_pheno_pool, preferred=str(triad[0]), fallback=0)
            _init_selectbox_state("tab8_triad_p5_b", p5_pheno_pool, preferred=str(triad[1]), fallback=1)
            _init_selectbox_state("tab8_triad_p5_c", p5_pheno_pool, preferred=str(triad[2]), fallback=2)
            p5_a = tri1.selectbox("Triad phenotype A (anchor)", options=p5_pheno_pool, key="tab8_triad_p5_a")
            p5_b = tri2.selectbox("Triad phenotype B", options=p5_pheno_pool, key="tab8_triad_p5_b")
            p5_c = tri3.selectbox("Triad phenotype C", options=p5_pheno_pool, key="tab8_triad_p5_c")

            st1, st2, st3 = st.columns(3)
            p5_style_options = ["local", "neighborhood"]
            _init_selectbox_state("tab8_triad_p5_style", p5_style_options, preferred="local", fallback=0)
            p5_style = st1.selectbox(
                "Triad distance style",
                options=p5_style_options,
                key="tab8_triad_p5_style",
                format_func=lambda x: "Direct-contact / local" if x == "local" else "Neighborhood annulus",
            )
            p5_min_um = st2.number_input(
                "Min distance (µm)",
                min_value=0.0,
                max_value=5000.0,
                value=float(local_min_um if p5_style == "local" else neigh_min_um),
                step=1.0,
                key="tab8_triad_p5_min_um",
            )
            p5_max_um = st3.number_input(
                "Max distance (µm)",
                min_value=0.1,
                max_value=10000.0,
                value=float(local_max_um if p5_style == "local" else neigh_max_um),
                step=1.0,
                key="tab8_triad_p5_max_um",
                help="A triad needs all three pairwise distances inside this range, exactly like the reference script.",
            )

        with tab_layers:
            available_layers = [
                name for name in sorted(annotations)
                if name != str(st.session_state.get("tab8_triad_p5_tissue_layer", ""))
            ]
            default_layers = [name for name in clouds.DEFAULT_LAYERS if name in available_layers]
            p5_layers = st.multiselect(
                "Annotation layers to draw (in drawing/legend order)",
                options=available_layers,
                default=default_layers or available_layers[: min(3, len(available_layers))],
                key="tab8_triad_p5_layers",
                help="The reference figure draws tumor, epithelia and stroma and omits endothelial.",
            )
            layer_color_map: dict[str, str] = {}
            if p5_layers:
                color_cols = st.columns(min(4, len(p5_layers)))
                for idx, name in enumerate(p5_layers):
                    with color_cols[idx % len(color_cols)]:
                        layer_color_map[name] = st.color_picker(
                            f"{str(name).capitalize()} color",
                            value=str(
                                st.session_state.get(
                                    f"tab8_density_p3_layer_color_{name}",
                                    clouds.default_layer_color(name, idx),
                                )
                            ),
                            key=f"tab8_triad_p5_layer_color_{name}",
                        )

            c1, c2, c3 = st.columns(3)
            p5_layer_alpha = c1.slider(
                "Annotation layer opacity",
                min_value=0.05,
                max_value=1.0,
                value=0.46,
                step=0.01,
                key="tab8_triad_p5_layer_alpha",
            )
            p5_background = c2.color_picker("Background color", value="#000000", key="tab8_triad_p5_background")
            p5_text_color = c3.color_picker("Text color", value="#FFFFFF", key="tab8_triad_p5_text_color")

            b1, b2, b3 = st.columns(3)
            p5_show_boundary = b1.checkbox(
                "Outline the tissue component",
                value=True,
                key="tab8_triad_p5_show_boundary",
            )
            p5_boundary_color = b2.color_picker(
                "Outline color",
                value="#FFFFFF",
                key="tab8_triad_p5_boundary_color",
                disabled=not p5_show_boundary,
            )
            p5_boundary_width = b3.slider(
                "Outline width (pt)",
                min_value=0.1,
                max_value=3.0,
                value=0.7,
                step=0.1,
                key="tab8_triad_p5_boundary_width",
                disabled=not p5_show_boundary,
            )

        with tab_cloud:
            d1, d2, d3 = st.columns(3)
            p5_show_cloud = d1.checkbox("Show density cloud", value=True, key="tab8_triad_p5_show_cloud")
            p5_grid = d2.number_input(
                "Density grid (bins per axis)",
                min_value=60,
                max_value=1200,
                value=360,
                step=20,
                key="tab8_triad_p5_grid",
                help="Reference value: 360.",
            )
            p5_sigma = d3.number_input(
                "Smoothing bandwidth (µm)",
                min_value=5.0,
                max_value=400.0,
                value=55.0,
                step=5.0,
                key="tab8_triad_p5_sigma",
                help="Gaussian sigma in microns. Reference value: 55.",
            )

            st.markdown("**Cloud color ramp** (transparent → low → mid → high)")
            s1, s2, s3 = st.columns(3)
            p5_stop_low = s1.color_picker("Low", value=clouds.CLOUD_STOPS[0], key="tab8_triad_p5_stop_low")
            p5_stop_mid = s2.color_picker("Mid", value=clouds.CLOUD_STOPS[1], key="tab8_triad_p5_stop_mid")
            p5_stop_high = s3.color_picker("High", value=clouds.CLOUD_STOPS[2], key="tab8_triad_p5_stop_high")

            st.markdown("**Triad centroids**")
            pt1, pt2, pt3 = st.columns(3)
            p5_show_points = pt1.checkbox("Show triad centroids", value=True, key="tab8_triad_p5_show_points")
            p5_point_color = pt1.color_picker(
                "Centroid color",
                value=clouds.CLOUD_HIGHLIGHT,
                key="tab8_triad_p5_point_color",
                disabled=not p5_show_points,
            )
            p5_point_size = pt2.slider(
                "Centroid size",
                min_value=0.5,
                max_value=20.0,
                value=2.5,
                step=0.5,
                key="tab8_triad_p5_point_size",
                disabled=not p5_show_points,
            )
            p5_point_alpha = pt3.slider(
                "Centroid opacity",
                min_value=0.02,
                max_value=1.0,
                value=0.22,
                step=0.02,
                key="tab8_triad_p5_point_alpha",
                disabled=not p5_show_points,
            )

            cb1, cb2 = st.columns(2)
            p5_show_colorbar = cb1.checkbox("Show color bar", value=True, key="tab8_triad_p5_show_colorbar")
            p5_colorbar_label = cb2.text_input(
                "Color bar label",
                value="Smoothed triad density (shared scale)",
                key="tab8_triad_p5_colorbar_label",
            )

        with tab_layout:
            l1, l2, l3 = st.columns(3)
            p5_panel_width = l1.number_input(
                "Width per map (in)", min_value=1.5, max_value=12.0, value=3.5, step=0.5, key="tab8_triad_p5_panel_width"
            )
            p5_panel_height = l2.number_input(
                "Figure height (in)", min_value=2.0, max_value=16.0, value=5.25, step=0.25, key="tab8_triad_p5_panel_height"
            )
            p5_dpi = l3.number_input("DPI", min_value=72, max_value=900, value=300, step=50, key="tab8_triad_p5_dpi")

            f1, f2, f3, f4 = st.columns(4)
            p5_title_fontsize = f1.slider("Map title (pt)", min_value=6, max_value=24, value=11, step=1, key="tab8_triad_p5_title_fontsize")
            p5_suptitle_fontsize = f2.slider("Figure title (pt)", min_value=8, max_value=28, value=13, step=1, key="tab8_triad_p5_suptitle_fontsize")
            p5_legend_fontsize = f3.slider("Legend (pt)", min_value=5, max_value=20, value=9, step=1, key="tab8_triad_p5_legend_fontsize")
            p5_colorbar_fontsize = f4.slider("Color bar (pt)", min_value=5, max_value=20, value=9, step=1, key="tab8_triad_p5_colorbar_fontsize")

            p5_suptitle = st.text_input(
                "Figure title",
                value="Method-dependent {triad} triad landscape\\n{batch}, {roi}",
                key="tab8_triad_p5_suptitle",
                help="Placeholders: {triad}, {batch}, {roi}. Use \\n for a line break; leave empty to hide the title.",
            )
            o1, o2, o3 = st.columns(3)
            p5_show_legend = o1.checkbox("Show legend", value=True, key="tab8_triad_p5_show_legend")
            p5_single_figures = o2.checkbox(
                "Also save one figure per method",
                value=False,
                key="tab8_triad_p5_single_figures",
                help="Matches the reference script, which writes a combined figure plus one map per method.",
            )
            p5_save_svg = o3.checkbox("Also save SVG", value=True, key="tab8_triad_p5_save_svg")
            p5_output_dir = st.text_input(
                "Output directory",
                value=str(Path(default_out_dir) / "panel5_triad_density_clouds"),
                key="tab8_triad_p5_output_dir",
                help="Defaults to a panel5_triad_density_clouds folder inside the shared output directory.",
            )

        col_btns = st.columns([1, 1])
        run_panel5 = col_btns[0].button("Render Panel 5", type="primary", key="tab8_triad_p5_run", width="stretch")
        save_panel5 = col_btns[1].button("Save Parameters", type="secondary", key="tab8_triad_p5_save", width="stretch")

        if save_panel5:
            from spacemap_gui.persistence import save_active_project_state
            save_active_project_state("tab8_analysis")
            st.success("Project settings saved to configuration file.")

        p5_triad = (str(p5_a), str(p5_b), str(p5_c))
        if run_panel5:
            if not annotations:
                st.error("Panel 5: Select an annotation directory containing the GeoJSON file for this batch.")
            elif not p5_methods:
                st.error("Panel 5: Select at least one phenotyping method column in Shared Figure Settings.")
            elif len(set(p5_triad)) != 3:
                st.error("Panel 5: Triad phenotypes A/B/C must be three distinct values.")
            elif float(p5_max_um) <= float(p5_min_um):
                st.error("Panel 5: Max distance must be greater than min distance.")
            elif p5_tissue_layer not in annotations:
                st.error(f"Panel 5: Layer '{p5_tissue_layer}' is not present in {p5_geojson_file}.")
            else:
                with st.status("Rendering Panel 5...", expanded=True) as status:
                    _log_terminal_event(f"Panel 5 started for batch={p5_batch}, roi={p5_roi}")
                    cells = run_df[run_df["__batch__"].astype(str) == str(p5_batch)]
                    roi_label = "all tissue ROIs"
                    if p5_roi != "(whole tissue)":
                        cells = cells[cells["__roi__"].astype(str) == str(p5_roi)]
                        roi_label = f"tissue_roi={p5_roi}"
                    cells = cells.dropna(subset=REQUIRED_COORDS)
                    status.write(f"{len(cells):,} cells in {p5_batch} | {roi_label}.")

                    if cells.empty:
                        status.update(label="Panel 5 finished with no cells", state="error")
                        st.warning("Panel 5: No cells for the selected batch/ROI.")
                    else:
                        tissue_geometry = annotations[str(p5_tissue_layer)]
                        if p5_roi != "(whole tissue)":
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
                            for name in p5_layers
                        }

                        triad_label = "-".join(p5_triad)
                        style = clouds.CloudStyle(
                            layer_colors={
                                name: layer_color_map.get(name, clouds.default_layer_color(name, idx))
                                for idx, name in enumerate(p5_layers)
                            },
                            layer_alpha=float(p5_layer_alpha),
                            background=str(p5_background),
                            text_color=str(p5_text_color),
                            show_boundary=bool(p5_show_boundary),
                            boundary_color=str(p5_boundary_color),
                            boundary_width=float(p5_boundary_width),
                            show_cloud=bool(p5_show_cloud),
                            cloud_stops=(str(p5_stop_low), str(p5_stop_mid), str(p5_stop_high)),
                            grid_size=int(p5_grid),
                            sigma_um=float(p5_sigma),
                            show_points=bool(p5_show_points),
                            point_color=str(p5_point_color),
                            point_size=float(p5_point_size),
                            point_alpha=float(p5_point_alpha),
                            panel_width=float(p5_panel_width),
                            panel_height=float(p5_panel_height),
                            title_fontsize=float(p5_title_fontsize),
                            suptitle_fontsize=float(p5_suptitle_fontsize),
                            legend_fontsize=float(p5_legend_fontsize),
                            colorbar_fontsize=float(p5_colorbar_fontsize),
                            colorbar_label=str(p5_colorbar_label),
                            show_colorbar=bool(p5_show_colorbar),
                            show_legend=bool(p5_show_legend),
                            cloud_label="Triad-density cloud",
                            point_label="Triad centroids",
                            count_noun="triads",
                            dpi=int(p5_dpi),
                        )

                        panels: list[dict[str, Any]] = []
                        instance_frames: list[pd.DataFrame] = []
                        for method in p5_methods:
                            if method not in cells.columns:
                                continue
                            status.write(f"Detecting {triad_label} triads for {method_label_map.get(method, method)}...")
                            instances = _collect_region_triad_instances(
                                region_df=cells,
                                phenotype_col=str(method),
                                triad=p5_triad,
                                style_name=str(p5_style),
                                min_dist_um=float(p5_min_um),
                                max_dist_um=float(p5_max_um),
                                pixel_size_um=float(pixel_size_um),
                                max_collect=int(max_saved_triads),
                            )
                            if not instances.empty:
                                instances = instances.copy()
                                instances["method"] = str(method)
                                instance_frames.append(instances)
                                points = instances[["triad_centroid_x", "triad_centroid_y"]].to_numpy(dtype=float)
                            else:
                                points = np.empty((0, 2), dtype=float)
                            panels.append({"label": method_label_map.get(str(method), str(method)), "points": points})

                        if not panels:
                            status.update(label="Panel 5 finished with no maps", state="error")
                            st.warning("Panel 5: None of the selected method columns are present in the data.")
                        else:
                            out_dir = Path(p5_output_dir)
                            out_dir.mkdir(parents=True, exist_ok=True)
                            status.write("Computing shared-scale triad density fields.")
                            field_data = clouds.compute_cloud_fields(panels, component, style, float(pixel_size_um))
                            suptitle = (
                                str(p5_suptitle)
                                .replace("\\n", "\n")
                                .replace("{triad}", triad_label)
                                .replace("{batch}", str(p5_batch))
                                .replace("{roi}", roi_label)
                            )
                            fig = clouds.render_cloud_figure(panels, component, layers, style, field_data, suptitle=suptitle)
                            saved_files: list[str] = []
                            figures: list[dict[str, str]] = []
                            if fig is not None:
                                safe_name = triad_label.replace(" ", "_").replace("/", "-")
                                fig_path = out_dir / f"panel5_{safe_name}_triad_density_clouds.png"
                                saved_files.extend(clouds.save_cloud_figure(fig, fig_path, style, also_svg=bool(p5_save_svg)))
                                st.markdown(f"#### {triad_label} triad density clouds")
                                st.pyplot(fig)
                                plt.close(fig)
                                figures.append({"title": f"{triad_label} triad density clouds", "path": str(fig_path)})
                                _log_terminal_event(f"Panel 5 wrote {fig_path}")

                                if p5_single_figures:
                                    from dataclasses import replace as _dc_replace
                                    for panel in panels:
                                        single_style = _dc_replace(style, panel_width=float(p5_panel_width) * 1.7)
                                        single_fig = clouds.render_cloud_figure(
                                            [panel], component, layers, single_style, field_data, suptitle=suptitle
                                        )
                                        if single_fig is None:
                                            continue
                                        label_safe = str(panel["label"]).replace(" ", "_").replace("/", "-")
                                        single_path = out_dir / f"panel5_{safe_name}_{label_safe}.png"
                                        saved_files.extend(
                                            clouds.save_cloud_figure(single_fig, single_path, single_style, also_svg=bool(p5_save_svg))
                                        )
                                        plt.close(single_fig)

                            if instance_frames:
                                instances_all = pd.concat(instance_frames, ignore_index=True)
                                instances_csv = out_dir / f"panel5_{triad_label.replace(' ', '_')}_triad_instances.csv"
                                instances_all.to_csv(instances_csv, index=False)
                                saved_files.append(str(instances_csv))

                            counts = ", ".join(
                                f"{panel['label']}: {len(field_data['points'].get(panel['label'], [])):,}" for panel in panels
                            )
                            context = (
                                f"{p5_batch} | {roi_label} | triad {triad_label} ({p5_style}, "
                                f"{float(p5_min_um):g}–{float(p5_max_um):g} µm) | {counts}"
                            )
                            st.session_state["tab8_triad_p5_results"] = {
                                "figures": figures,
                                "saved_files": saved_files,
                                "output_dir": str(out_dir),
                                "context": context,
                            }
                            status.update(label="Panel 5 complete", state="complete")
                            st.success(f"Saved {len(saved_files)} file(s) to {out_dir}")
        elif st.session_state.get("tab8_triad_p5_results"):
            _render_panel5_results(st.session_state["tab8_triad_p5_results"])


def render_triad_analysis(project_root: Path, df: pd.DataFrame, unified_csv_path: str | None = None):
    if unified_csv_path:
        current_csv = str(unified_csv_path)
        last_loaded_csv = st.session_state.get("tab8_triad_loaded_for_csv")
        if (not isinstance(last_loaded_csv, str)) or (last_loaded_csv != current_csv):
            load_tab8_config_for_csv(unified_csv_path, key_prefixes=["tab8_triad_"])
            st.session_state["tab8_triad_loaded_for_csv"] = current_csv

    st.subheader("Triad")
    st.caption(
        "Panel 1: triad detection & enrichment · Panel 2: image-level triad validation · "
        "Panel 3: triad spatial map on the virtual tissue · Panel 4: cross-method triad statistics · "
        "Panel 5: triad density clouds on the annotated tissue. "
        "Shared settings at the top apply to all panels."
    )
    if unified_csv_path:
        st.caption("Settings are auto-saved per unified CSV (including panel directories).")

    if len(df.columns) < 2:
        st.error("Need at least two columns to select X/Y coordinates.")
        return

    all_columns = list(df.columns)
    coord_options = _column_options(all_columns)
    default_x_val = _preferred_or_blank(coord_options, "x_centroid")
    default_y_val = _preferred_or_blank(coord_options, "y_centroid")

    with st.expander("Shared Configuration · Coordinates, Regions & Triad Definition", expanded=True):
        st.caption("Applies to every panel below. Each panel runs only when you click its own run button.")
        xy1, xy2 = st.columns(2)
        current_x = st.session_state.get("tab8_triad_x_coord_col")
        if current_x not in coord_options:
            st.session_state["tab8_triad_x_coord_col"] = default_x_val
        x_coord_col = xy1.selectbox(
            "X Coordinate Column",
            options=coord_options,
            key="tab8_triad_x_coord_col",
            format_func=_format_opt,
        )

        current_y = st.session_state.get("tab8_triad_y_coord_col")
        if current_y not in coord_options:
            st.session_state["tab8_triad_y_coord_col"] = default_y_val
        y_coord_col = xy2.selectbox(
            "Y Coordinate Column",
            options=coord_options,
            key="tab8_triad_y_coord_col",
            format_func=_format_opt,
        )

        selectable_cols = [c for c in all_columns if c not in {x_coord_col, y_coord_col, ""}]
        phenotype_candidates = [c for c in selectable_cols if "phenotype" in c.lower()]
        phenotype_options = ["None"] + phenotype_candidates

        c1, c2, c3 = st.columns(3)
        default_pheno = "phenotype" if "phenotype" in phenotype_options else "None"
        current_pheno = st.session_state.get("tab8_triad_pheno_col")
        if current_pheno not in phenotype_options:
            st.session_state["tab8_triad_pheno_col"] = default_pheno
        phenotype_col = c1.selectbox("Phenotyping column", options=phenotype_options, key="tab8_triad_pheno_col")

        batch_col_options = [""] + selectable_cols
        _init_selectbox_state("tab8_triad_batch_col", batch_col_options, preferred="batch_id", fallback=0)
        batch_col = c2.selectbox("Batch ID Column", options=batch_col_options, format_func=_format_opt, key="tab8_triad_batch_col")

        roi_col_options = [""] + selectable_cols
        _init_selectbox_state("tab8_triad_roi_col", roi_col_options, preferred="tissue_roi", fallback=0)
        roi_col = c3.selectbox("Tissue ROI Column", options=roi_col_options, format_func=_format_opt, key="tab8_triad_roi_col")

        d1, d2, d3 = st.columns(3)
        tissue_col_options = ["None"] + selectable_cols
        _init_selectbox_state("tab8_triad_tissue_col", tissue_col_options, preferred="tissue", fallback=0)
        tissue_col = d1.selectbox("Tissue filter column", options=tissue_col_options, key="tab8_triad_tissue_col")
        tissue_keep_value = d2.text_input("Tissue keep value", value="1", key="tab8_triad_tissue_keep")
        min_cells_region = d3.number_input("Min cells per region", min_value=3, max_value=50000, value=20, step=1, key="tab8_triad_min_cells_region")

        e1, e2, e3 = st.columns(3)
        pixel_size_um = e1.number_input("Pixel Size (µm)", min_value=0.01, value=0.5083, step=0.0001, format="%.4f", key="tab8_triad_pixel_um")
        n_permutations = e2.number_input("Permutations", min_value=50, max_value=10000, value=1000, step=50, key="tab8_triad_perms")
        random_state = e3.number_input("random_state", min_value=0, max_value=100000, value=42, step=1, key="tab8_triad_random")

        p4, p5 = st.columns(2)
        max_saved_triads = p4.number_input("Max triad instances to save", min_value=1000, max_value=500000, value=50000, step=1000, key="tab8_triad_max_saved")
        phenotype_subset_only = p5.checkbox("Use only triad phenotypes during permutation", value=False, key="tab8_triad_subset_only")

        phenotype_values = _get_unique_values(df, phenotype_col) if phenotype_col != "None" and phenotype_col in df.columns else []
        st.markdown("#### Triad definition")
        t1, t2, t3 = st.columns(3)
        if phenotype_values:
            _init_selectbox_state("tab8_triad_a", phenotype_values, preferred=phenotype_values[0], fallback=0)
            _init_selectbox_state(
                "tab8_triad_b",
                phenotype_values,
                preferred=phenotype_values[1] if len(phenotype_values) > 1 else phenotype_values[0],
                fallback=1,
            )
            _init_selectbox_state(
                "tab8_triad_c",
                phenotype_values,
                preferred=phenotype_values[2] if len(phenotype_values) > 2 else phenotype_values[0],
                fallback=2,
            )
            triad_a = t1.selectbox("Triad phenotype A", options=phenotype_values, key="tab8_triad_a")
            triad_b = t2.selectbox("Triad phenotype B", options=phenotype_values, key="tab8_triad_b")
            triad_c = t3.selectbox("Triad phenotype C", options=phenotype_values, key="tab8_triad_c")
        else:
            triad_a = ""
            triad_b = ""
            triad_c = ""
            st.info("No phenotype values available for triad selection with the current phenotyping column.")

        style_options = ["Direct-contact / local triads", "Neighborhood annulus triads"]
        selected_styles = st.multiselect("Triad detection styles", options=style_options, default=style_options, key="tab8_triad_styles")

        local_min_um, local_max_um = 0.0, 30.0
        if "Direct-contact / local triads" in selected_styles:
            l1, l2 = st.columns(2)
            local_min_um = l1.number_input("Local min distance (µm)", min_value=0.0, max_value=5000.0, value=0.0, step=1.0, key="tab8_triad_local_min_um")
            local_max_um = l2.number_input("Local max distance (µm)", min_value=0.1, max_value=5000.0, value=30.0, step=1.0, key="tab8_triad_local_max_um")

        neigh_min_um, neigh_max_um = 100.0, 400.0
        if "Neighborhood annulus triads" in selected_styles:
            n1, n2 = st.columns(2)
            neigh_min_um = n1.number_input("Neighborhood min distance (µm)", min_value=0.0, max_value=5000.0, value=100.0, step=1.0, key="tab8_triad_neigh_min_um")
            neigh_max_um = n2.number_input("Neighborhood max distance (µm)", min_value=0.1, max_value=10000.0, value=400.0, step=1.0, key="tab8_triad_neigh_max_um")

        out_dir = st.text_input("Output Directory", value=str(project_root / "outputs" / "spatial_analysis" / "triad"), key="tab8_triad_outdir")

    if not x_coord_col or not y_coord_col:
        st.info("Select X and Y coordinate columns to enable triad analysis.")
        if unified_csv_path:
            save_tab8_config_for_csv(unified_csv_path)
        return

    if x_coord_col == y_coord_col:
        st.error("X Coordinate Column and Y Coordinate Column must be different.")
        if unified_csv_path:
            save_tab8_config_for_csv(unified_csv_path)
        return

    if phenotype_col == "None":
        st.info("Select a phenotyping column to run Panel 1.")
        if unified_csv_path:
            save_tab8_config_for_csv(unified_csv_path)
        return

    run_df_key = "tab8_triad_prepared_run_df"
    rebuild_run_df = True
    if run_df_key in st.session_state:
        cached_inputs = st.session_state.get("tab8_triad_run_df_inputs", {})
        current_inputs = {
            "csv_path": unified_csv_path,
            "x_coord_col": x_coord_col,
            "y_coord_col": y_coord_col,
            "batch_col": batch_col,
            "roi_col": roi_col,
        }
        if cached_inputs == current_inputs:
            rebuild_run_df = False
            run_df = st.session_state[run_df_key]
            
    if rebuild_run_df:
        run_df = df.copy()
        run_df["x_centroid"] = pd.to_numeric(run_df[x_coord_col], errors="coerce")
        run_df["y_centroid"] = pd.to_numeric(run_df[y_coord_col], errors="coerce")
        if batch_col:
            run_df["__batch__"] = run_df[batch_col].astype(str)
        else:
            run_df["__batch__"] = "ALL"
        if roi_col:
            run_df["__roi__"] = run_df[roi_col].astype(str)
        else:
            run_df["__roi__"] = "ALL"
        st.session_state[run_df_key] = run_df
        st.session_state["tab8_triad_run_df_inputs"] = {
            "csv_path": unified_csv_path,
            "x_coord_col": x_coord_col,
            "y_coord_col": y_coord_col,
            "batch_col": batch_col,
            "roi_col": roi_col,
        }

    triad = (str(triad_a), str(triad_b), str(triad_c))

    default_methods = [
        c for c in ["phenotype_leiden", "phenotype_som", "phenotype_threshold", "phenotype_scimap"]
        if c in phenotype_candidates
    ]
    with st.expander("Shared Figure Settings · methods, display names and colors", expanded=True):
        shared = render_shared_figure_settings(
            phenotype_candidates,
            default_methods,
            methods_key="tab8_triad_shared_methods",
            label_key_fmt="tab8_triad_method_label_{method}",
            color_key_fmt="tab8_triad_method_color_{method}",
            legacy_methods_key="tab8_triad_p4_methods",
            seed_label_key_fmt="tab8_density_p1_label_{method}",
            seed_color_key_fmt="tab8_density_method_color_{method}",
            methods_help=(
                "Panel 4 compares these columns and Panel 5 draws one triad map per column. "
                "Defaults follow the Density subtab so both tabs produce matching figures."
            ),
        )

    st.markdown("### Panels")

    with st.expander("**Panel 1 · Triad Discovery and Enrichment**", expanded=False):
        _render_panel1_triad_discovery(
            run_df=run_df,
            triad=triad,
            phenotype_col=phenotype_col,
            batch_col=batch_col,
            roi_col=roi_col,
            tissue_col=tissue_col,
            tissue_keep_value=str(tissue_keep_value),
            selected_styles=selected_styles,
            local_min_um=float(local_min_um),
            local_max_um=float(local_max_um),
            neigh_min_um=float(neigh_min_um),
            neigh_max_um=float(neigh_max_um),
            pixel_size_um=float(pixel_size_um),
            min_cells_region=int(min_cells_region),
            n_permutations=int(n_permutations),
            random_state=int(random_state),
            phenotype_subset_only=bool(phenotype_subset_only),
            max_saved_triads=int(max_saved_triads),
            out_dir=str(out_dir),
        )

    triad_instances_df = st.session_state.get("tab8_triad_instances_df")
    marker_candidates = [
        c for c in run_df.columns
        if c not in set(REQUIRED_COORDS + [x_coord_col, y_coord_col])
        and pd.api.types.is_numeric_dtype(run_df[c])
        and ("phenotype" not in c.lower())
    ]
    if isinstance(triad_instances_df, pd.DataFrame) and not triad_instances_df.empty:
        _render_panel_image_triads(
            project_root=project_root,
            triad_instances_df=triad_instances_df,
            marker_candidates=marker_candidates,
            triad_fallback_names=triad,
        )
    else:
        st.info("Panel 2 (image visualization) needs triad instances from Panel 1 or a loaded triad CSV.")

    _render_panel_virtual_tissue(
        project_root=project_root,
        run_df=run_df,
        phenotype_candidates=phenotype_candidates,
        triad_fallback_names=triad,
        unified_csv_path=unified_csv_path,
    )

    _render_panel4_triad_stats(
        project_root=project_root,
        run_df=run_df,
        triad=triad,
        phenotype_candidates=phenotype_candidates,
        shared=shared,
        selected_styles=selected_styles,
        local_min_um=float(local_min_um),
        local_max_um=float(local_max_um),
        neigh_min_um=float(neigh_min_um),
        neigh_max_um=float(neigh_max_um),
        default_pixel_size_um=float(pixel_size_um),
        default_min_cells_region=int(min_cells_region),
        tissue_col=str(tissue_col),
        tissue_keep_value=str(tissue_keep_value),
        default_out_dir=str(out_dir),
    )

    _render_panel5_triad_density_cloud(
        project_root=project_root,
        run_df=run_df,
        phenotype_candidates=phenotype_candidates,
        shared=shared,
        triad=triad,
        local_min_um=float(local_min_um),
        local_max_um=float(local_max_um),
        neigh_min_um=float(neigh_min_um),
        neigh_max_um=float(neigh_max_um),
        pixel_size_um=float(pixel_size_um),
        max_saved_triads=int(max_saved_triads),
        default_out_dir=str(out_dir),
    )

    _render_panel6_zonal_triads(
        run_df=run_df,
        phenotype_candidates=phenotype_candidates,
        shared=shared,
        triad=triad,
        local_min_um=float(local_min_um),
        local_max_um=float(local_max_um),
        neigh_min_um=float(neigh_min_um),
        neigh_max_um=float(neigh_max_um),
        pixel_size_um=float(pixel_size_um),
        min_cells_region=int(min_cells_region),
        default_out_dir=str(out_dir),
    )

    if unified_csv_path:
        save_tab8_config_for_csv(unified_csv_path)


def _render_panel1_triad_discovery(
    run_df: pd.DataFrame,
    triad: tuple[str, str, str],
    phenotype_col: str,
    batch_col: str,
    roi_col: str,
    tissue_col: str,
    tissue_keep_value: str,
    selected_styles: list[str],
    local_min_um: float,
    local_max_um: float,
    neigh_min_um: float,
    neigh_max_um: float,
    pixel_size_um: float,
    min_cells_region: int,
    n_permutations: int,
    random_state: int,
    phenotype_subset_only: bool,
    max_saved_triads: int,
    out_dir: str,
):
    st.info(
        "**How it works:** Panel 1 identifies three-cell spatial triangles (triads) among phenotypes A, B, and C "
        "within the configured distance thresholds (e.g., local direct contact <= 30µm or neighborhood annulus 100–400µm). "
        "It performs a permutation test by randomly shuffling phenotype labels across all cell coordinates to construct "
        "a random null distribution of expected triad counts.\n\n"
        "**Why is the Z-score/significance so high?** Globally shuffling phenotypes assumes a homogeneous cell mixture. "
        "However, real tissue is highly structured (e.g., compartmentalized into tumor vs. stroma beds). If cells are restricted "
        "to a compartment (like immune cells in the stroma), they are naturally clustered close to each other. "
        "Shuffling them globally scatters them into the tumor bed where they don't belong, drastically lowering the expected co-occurrence "
        "mean and standard deviation. This inflates the observed Z-score (often > 50).\n\n"
        "**How to improve scientifically:** To control for macro-structure, you can perform *compartment-restricted permutation* "
        "(shuffling phenotypes only within their specific tissue zone) or apply *spatial coordinate jittering/toroidal shifts* "
        "to preserve local densities while testing micro-scale relationships."
    )
    c_load1, c_load2 = st.columns([2, 1])
    with c_load1:
        load_instances_csv = st.text_input(
            "Optional: load triad instances CSV (skip detection)",
            value="",
            key="tab8_triad_load_instances_csv",
            help="Use a previously exported `triad_instances_<style>.csv`.",
        )
    with c_load2:
        load_btn = st.button("Load Triad CSV", key="tab8_triad_load_btn")

    if load_btn:
        try:
            loaded = pd.read_csv(load_instances_csv)
            st.session_state["tab8_triad_instances_df"] = loaded
            
            # Reset Panel 1 cache so it shows loading
            st.session_state.pop("tab8_triad_p1_summary_df", None)
            st.session_state.pop("tab8_triad_p1_run_details", None)
            st.session_state.pop("tab8_triad_p1_null_plot_bytes", None)
            
            st.success(f"Loaded triad instances: {len(loaded):,} rows")
        except Exception as exc:
            st.error(f"Failed to load triad CSV: {exc}")

    run_panel1 = st.button("Run Triad Analysis", type="primary", key="tab8_triad_run")
    if run_panel1:
        if len(set(triad)) != 3:
            st.error("Triad phenotype A/B/C must be three distinct values.")
        elif not selected_styles:
            st.error("Select at least one triad detection style.")
        else:
            style_configs = []
            if "Direct-contact / local triads" in selected_styles:
                if float(local_max_um) <= float(local_min_um):
                    st.error("Local style requires max distance > min distance.")
                else:
                    style_configs.append(("local", float(local_min_um), float(local_max_um)))
            if "Neighborhood annulus triads" in selected_styles:
                if float(neigh_max_um) <= float(neigh_min_um):
                    st.error("Neighborhood style requires max distance > min distance.")
                else:
                    style_configs.append(("neighborhood", float(neigh_min_um), float(neigh_max_um)))

            if style_configs:
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)

                summary_rows = []
                instances_all = []
                prog = st.progress(0)
                status_txt = st.empty()

                cached_p1_null_plot_bytes = {}
                cached_p1_run_details = []

                for idx, (style_name, min_um, max_um) in enumerate(style_configs):
                    status_txt.info(f"Running triad analysis: {style_name} ({min_um:.1f}–{max_um:.1f} µm)")
                    _log_terminal_event(f"Start style={style_name}")

                    summary, per_roi_df, null_counts, triad_instances_df = _detect_triad_enrichment(
                        df=run_df,
                        triad_phenotypes=triad,
                        phenotype_col=phenotype_col,
                        batch_col=batch_col,
                        roi_col=roi_col,
                        tissue_col=tissue_col,
                        tissue_keep_value=str(tissue_keep_value),
                        min_dist_um=float(min_um),
                        max_dist_um=float(max_um),
                        micron_per_pixel=float(pixel_size_um),
                        min_cells_per_region=int(min_cells_region),
                        n_permutations=int(n_permutations),
                        random_state=int(random_state),
                        phenotype_subset_only=bool(phenotype_subset_only),
                        max_triad_instances_to_save=int(max_saved_triads),
                    )

                    summary["style"] = style_name
                    summary["phenotype_column"] = str(phenotype_col)
                    summary["pixel_size_um"] = float(pixel_size_um)
                    summary_rows.append(summary)

                    run_detail = {
                        "style_name": style_name,
                        "observed_total": int(summary.get("observed_total_count", 0)),
                        "expected_mean": float(summary.get("expected_mean_count", 0.0)),
                        "z_score": float(summary.get("z_score", np.nan)) if np.isfinite(summary.get("z_score", np.nan)) else np.nan,
                        "empirical_p_enrichment": float(summary.get("empirical_p_value_enrichment", np.nan)),
                        "saved_instances": int(summary.get("n_triad_instances_saved", 0)),
                        "per_roi_df": per_roi_df,
                        "instances_csv": str(out_path / f"triad_instances_{style_name}.csv"),
                    }
                    cached_p1_run_details.append(run_detail)

                    fig = _plot_null_distribution(
                        null_counts=null_counts,
                        observed=int(summary.get("observed_total_count", 0)),
                        title=f"Triad null distribution ({style_name}) | {summary.get('triad', '')}",
                        empirical_p_value=float(summary.get("empirical_p_value_enrichment", np.nan)),
                        z_score=float(summary.get("z_score", np.nan)),
                    )
                    
                    import io
                    buf = io.BytesIO()
                    _global_dpi = int(st.session_state.get("tab8_density_global_png_dpi", 300))
                    fig.savefig(buf, format="png", dpi=_global_dpi, bbox_inches="tight")
                    buf.seek(0)
                    cached_p1_null_plot_bytes[style_name] = buf.getvalue()

                    fig.savefig(out_path / f"triad_null_distribution_{style_name}.png", dpi=_global_dpi, bbox_inches="tight")
                    plt.close(fig)

                    per_roi_csv = out_path / f"triad_per_roi_{style_name}.csv"
                    null_csv = out_path / f"triad_null_counts_{style_name}.csv"
                    instances_csv = out_path / f"triad_instances_{style_name}.csv"
                    (per_roi_df if not per_roi_df.empty else pd.DataFrame(columns=["batch", "roi", "n_cells", "observed_triad_count"]))\
                        .to_csv(per_roi_csv, index=False)
                    pd.DataFrame({"null_triad_count": null_counts}).to_csv(null_csv, index=False)

                    triad_instances_df = triad_instances_df.copy()
                    triad_instances_df["style"] = style_name
                    triad_instances_df.to_csv(instances_csv, index=False)
                    instances_all.append(triad_instances_df)

                    prog.progress((idx + 1) / max(len(style_configs), 1))

                prog.empty()
                status_txt.success("Panel 1 complete.")

                summary_df = pd.DataFrame(summary_rows)
                summary_csv = out_path / "triad_summary.csv"
                summary_df.to_csv(summary_csv, index=False)

                all_instances = pd.DataFrame()
                if instances_all:
                    all_instances = pd.concat(instances_all, ignore_index=True)
                    all_instances_csv = out_path / "triad_instances_all_styles.csv"
                    all_instances.to_csv(all_instances_csv, index=False)
                    st.session_state["tab8_triad_instances_df"] = all_instances

                # Cache in session state
                st.session_state["tab8_triad_p1_summary_df"] = summary_df
                st.session_state["tab8_triad_p1_run_details"] = cached_p1_run_details
                st.session_state["tab8_triad_p1_null_plot_bytes"] = cached_p1_null_plot_bytes
                st.session_state["tab8_triad_p1_summary_csv"] = str(summary_csv)
                if instances_all:
                    st.session_state["tab8_triad_p1_all_instances_csv"] = str(all_instances_csv)
                    st.session_state["tab8_triad_p1_out_dir"] = str(out_path)
                    st.session_state["tab8_triad_p1_styles"] = [s[0] for s in style_configs]
                    st.session_state["tab8_triad_p1_triad_label"] = "::".join(triad)
                else:
                    st.session_state.pop("tab8_triad_p1_all_instances_csv", None)
                    st.session_state.pop("tab8_triad_p1_out_dir", None)
                    st.session_state.pop("tab8_triad_p1_styles", None)
                    st.session_state.pop("tab8_triad_p1_triad_label", None)
                    
                st.rerun()

    # Display Panel 1 offline results from session state
    if "tab8_triad_p1_summary_df" in st.session_state:
        summary_df = st.session_state["tab8_triad_p1_summary_df"]
        run_details = st.session_state.get("tab8_triad_p1_run_details") or []
        null_plot_bytes = st.session_state.get("tab8_triad_p1_null_plot_bytes") or {}
        if not isinstance(run_details, list) or not isinstance(null_plot_bytes, dict):
            # Leftovers from an older project file that serialized these caches.
            run_details, null_plot_bytes = [], {}

        st.markdown("#### Discovery & Enrichment Results (Offline View)")
        for detail in run_details:
            if not isinstance(detail, dict):
                continue
            st.markdown(f"##### {str(detail.get('style_name', '')).title()} triads")
            st.write({
                "Observed_total": detail.get("observed_total"),
                "Expected_mean": detail.get("expected_mean"),
                "Z_score": detail.get("z_score"),
                "Empirical_p_enrichment": detail.get("empirical_p_enrichment"),
                "Saved_instances": detail.get("saved_instances"),
            })
            if detail.get("style_name") in null_plot_bytes:
                _show_png_bytes(null_plot_bytes[detail["style_name"]])

            per_roi_df = detail.get("per_roi_df")
            if isinstance(per_roi_df, pd.DataFrame) and not per_roi_df.empty:
                st.dataframe(per_roi_df.head(200), width="stretch")
            st.caption(f"Saved triad instances CSV: `{detail.get('instances_csv', '')}`")
            
        st.markdown("##### Triad Summary Table")
        st.dataframe(summary_df, width="stretch")
        
        summary_csv_path = st.session_state.get("tab8_triad_p1_summary_csv")
        all_instances_csv_path = st.session_state.get("tab8_triad_p1_all_instances_csv")
        out_path_str = st.session_state.get("tab8_triad_p1_out_dir")
        styles_run = st.session_state.get("tab8_triad_p1_styles")
        triad_label = st.session_state.get("tab8_triad_p1_triad_label")
        
        if summary_csv_path:
            st.caption(f"Saved triad summary CSV: `{summary_csv_path}`")
        if all_instances_csv_path:
            st.write({
                "Output_dir": out_path_str,
                "Summary_csv": summary_csv_path,
                "Instances_all_styles_csv": all_instances_csv_path,
                "Styles_run": styles_run,
                "Triad": triad_label,
            })
    else:
        st.info("Run Triad Analysis or load a triad instances CSV to discover triads.")
