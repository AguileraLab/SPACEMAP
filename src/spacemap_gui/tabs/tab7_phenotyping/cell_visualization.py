"""
Cell Visualization Subtab
Publication-ready cell figures using GeoJSON polygon or Voronoi segmentation.
Supports categorical, continuous, and validation (match/mismatch) coloring modes.
"""

from __future__ import annotations

import glob
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import Normalize, to_hex, to_rgb
from matplotlib.cm import ScalarMappable

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

try:
    from scipy.spatial import Voronoi
except ImportError:
    Voronoi = None

try:
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EXTRA_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", "#393b79",
]

_DEFAULT_COLOR_MAP: dict[str, str] = {
    "B":            "#00008B", "B cell":       "#00008B",
    "CD4":          "#FF69B4", "CD8":          "#DC143C",
    "Cytotoxic T":  "#FF0000", "DC":           "#008000",
    "Endothelial":  "#0000FF", "Endothelia":   "#0000FF",
    "Epithelial":   "#800080", "Epithelia":    "#800080",
    "Fibroblast":   "#808080", "CAF":          "#808080",
    "Granulocyte":  "#BDB76B", "Helper T":     "#FFC0CB",
    "IMC":          "#4682B4", "Mac":          "#ADD8E6",
    "Macrophage":   "#ADD8E6", "Plasma":       "#DA70D6",
    "Plasma B cell":"#DA70D6", "Treg":         "#FFA500",
    "Others":       "#A9A9A9", "unclassified": "#000000",
}

_CATEGORICAL_PALETTES = [
    "Default phenotype map",
    "tab10",
    "tab20",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "Dark2",
    "Accent",
    "husl",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_color(label: str, mapping: dict[str, str]) -> str:
    """Return a hex color for *label*, assigning a fallback if needed."""
    if label not in mapping:
        idx = len(mapping) % len(_EXTRA_COLORS)
        mapping[label] = _EXTRA_COLORS[idx]
    return mapping[label]


def _build_unified_color_mapping(
    df: pd.DataFrame,
    columns: list[str],
    base_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a single colour mapping covering all unique values across *columns*."""
    mapping = dict(base_mapping or _DEFAULT_COLOR_MAP)
    for col in columns:
        if col not in df.columns:
            continue
        for val in df[col].dropna().unique():
            _ensure_color(str(val), mapping)
    return mapping


def _sanitize_widget_suffix(value: str) -> str:
    """Return a Streamlit-safe suffix for dynamic widget keys."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "item"


def _collect_categorical_labels(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Collect sorted, unique labels across selected categorical columns."""
    labels: set[str] = set()
    for col in columns:
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            cleaned = value.strip()
            if cleaned:
                labels.add(cleaned)
    return sorted(labels, key=str.lower)


def _build_palette_mapping(labels: list[str], palette_name: str) -> dict[str, str]:
    """Build a label to colour mapping from a named palette."""
    if palette_name == "Default phenotype map":
        mapping = dict(_DEFAULT_COLOR_MAP)
        for label in labels:
            _ensure_color(label, mapping)
        return mapping

    try:
        palette = sns.color_palette(palette_name, n_colors=max(len(labels), 1)).as_hex()
    except Exception:
        palette = sns.color_palette("tab10", n_colors=max(len(labels), 1)).as_hex()
    return {label: str(palette[idx % len(palette)]) for idx, label in enumerate(labels)}


def _prepare_categorical_color_mapping(
    df: pd.DataFrame,
    columns: list[str],
    palette_name: str,
    widget_prefix: str = "cellviz_cat_color",
) -> dict[str, str]:
    """Create a categorical mapping and keep Streamlit color pickers synced to the selected preset."""
    labels = _collect_categorical_labels(df, columns)
    preset_mapping = _build_palette_mapping(labels, palette_name)

    last_palette_key = f"{widget_prefix}_last_palette"
    active_labels_key = f"{widget_prefix}_active_labels"
    preset_changed = st.session_state.get(last_palette_key) != palette_name
    previous_labels = set(st.session_state.get(active_labels_key, []))
    current_labels = set(labels)

    for label in labels:
        widget_key = f"{widget_prefix}_{_sanitize_widget_suffix(label)}"
        if preset_changed or label not in previous_labels or widget_key not in st.session_state:
            st.session_state[widget_key] = preset_mapping.get(label, "#000000")

    st.session_state[last_palette_key] = palette_name
    st.session_state[active_labels_key] = labels

    color_mapping = dict(preset_mapping)
    for label in labels:
        widget_key = f"{widget_prefix}_{_sanitize_widget_suffix(label)}"
        picked = st.session_state.get(widget_key, preset_mapping.get(label, "#000000"))
        try:
            color_mapping[label] = to_hex(picked)
        except ValueError:
            color_mapping[label] = preset_mapping.get(label, "#000000")
    return color_mapping


def _render_color_override_controls(
    labels: list[str],
    color_mapping: dict[str, str],
    widget_prefix: str = "cellviz_cat_color",
    n_columns: int = 3,
) -> None:
    """Render color pickers for each active categorical label."""
    if not labels:
        st.info("No category values found in the current filtered selection.")
        return

    columns = st.columns(max(1, n_columns))
    for idx, label in enumerate(labels):
        widget_key = f"{widget_prefix}_{_sanitize_widget_suffix(label)}"
        with columns[idx % len(columns)]:
            st.color_picker(
                label,
                value=color_mapping.get(label, "#000000"),
                key=widget_key,
            )


def _extract_batch_prefix(batch_id: str) -> str | None:
    """Extract ROI prefix from a batch_id, e.g. 'P11_roi_2_1' -> 'P11_roi_2'."""
    m = re.match(r'^(.+_roi_\d+)', str(batch_id))
    return m.group(1) if m else None


def _find_geojson_for_prefix(
    prefix: str,
    seg_dir: str,
) -> str | None:
    for folder in [seg_dir, os.path.join(seg_dir, "geojson")]:
        if os.path.exists(folder):
            candidates = glob.glob(os.path.join(folder, f"{prefix}*_cell_boundaries.geojson"))
            if candidates:
                return candidates[0]
            # Broader fallback
            candidates = glob.glob(os.path.join(folder, f"{prefix}*.geojson"))
            if candidates:
                return candidates[0]
    return None


def _build_geojson_match_map(
    roi_dir: str,
    seg_dir: str,
    batch_ids: list[str],
) -> dict[str, str]:
    """Build a batch_id to GeoJSON path lookup using CellVisualizer triad matching."""
    if CellVisualizer is None:
        return {}
    if not roi_dir or not seg_dir:
        return {}
    if not os.path.exists(roi_dir) or not os.path.exists(seg_dir):
        return {}

    cleaned_batch_ids = [str(batch_id).strip() for batch_id in batch_ids if str(batch_id).strip()]
    if not cleaned_batch_ids:
        return {}

    try:
        visualizer = CellVisualizer(roi_dir, seg_dir)
        matches = visualizer.match_file_triads(cleaned_batch_ids)
    except Exception:
        return {}

    geojson_map: dict[str, str] = {}
    for batch_id, paths in matches.items():
        if not isinstance(paths, dict):
            continue
        geojson_path = paths.get("geojson")
        if geojson_path:
            geojson_map[str(batch_id)] = str(geojson_path)
    return geojson_map


def _resolve_geojson_path(
    *,
    batch_label: str,
    df_batch: pd.DataFrame,
    batch_col: str,
    seg_dir: str,
    geojson_matches: dict[str, str] | None = None,
) -> str | None:
    """Resolve a GeoJSON path using linked matches first, then filename heuristics."""
    candidates: list[str] = []

    batch_label_str = str(batch_label).strip()
    if batch_label_str and batch_label_str != "all":
        candidates.append(batch_label_str)

    if batch_col in df_batch.columns:
        for value in df_batch[batch_col].dropna().astype(str).unique().tolist():
            value_str = str(value).strip()
            if value_str and value_str not in candidates:
                candidates.append(value_str)

    for candidate in candidates:
        matched_path = (geojson_matches or {}).get(candidate)
        if matched_path and os.path.exists(matched_path):
            return matched_path

    for candidate in candidates:
        prefix = _extract_batch_prefix(candidate)
        if prefix:
            geojson_path = _find_geojson_for_prefix(prefix, seg_dir)
            if geojson_path:
                return geojson_path

        direct_candidates = glob.glob(os.path.join(seg_dir, f"{candidate}*.geojson"))
        if direct_candidates:
            return direct_candidates[0]

    return None


# ---------------------------------------------------------------------------
# GeoJSON rendering helpers (adapted from Step14 / acc_utils)
# ---------------------------------------------------------------------------

def _render_geojson_polygons(
    ax: plt.Axes,
    geojson_path: str,
    df_batch: pd.DataFrame,
    color_col: str,
    color_mapping: dict[str, str],
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    mirror_y: bool = True,
) -> set[str]:
    """Draw GeoJSON polygons coloured by *color_col* onto *ax*.

    Returns the set of phenotype labels actually plotted.
    """
    if not _HAS_SHAPELY:
        st.error("shapely is required for GeoJSON rendering but is not installed.")
        return set()

    # Build spatial index from cell centroids
    cell_points = []
    for _, row in df_batch.iterrows():
        if pd.notna(row[x_col]) and pd.notna(row[y_col]):
            pt = Point(row[x_col], row[y_col])
            cell_points.append((pt, str(row[color_col]) if pd.notna(row[color_col]) else "unclassified"))
    if not cell_points:
        return set()

    points_list = [pt for pt, _ in cell_points]
    pt_to_label = {pt.wkb: label for pt, label in cell_points}
    tree = STRtree(points_list)

    try:
        with open(geojson_path, "r") as f:
            gj = json.load(f)
    except Exception as exc:
        st.warning(f"Could not read GeoJSON: {exc}")
        return set()

    features = gj.get("features", [])
    if not features:
        return set()

    plotted: set[str] = set()
    all_x: list[float] = []
    all_y: list[float] = []

    polys_to_draw: list[tuple[list, str]] = []

    for feat in features:
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        polygon = shape(geom)
        candidates = tree.query(polygon)
        cands = [points_list[i] for i in candidates]
        label = "unclassified"
        matched = False
        for cg in cands:
            if polygon.contains(cg):
                label = pt_to_label.get(cg.wkb, "unclassified")
                matched = True
                break
        if not matched:
            continue

        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]
        else:
            rings = [poly[0] for poly in geom["coordinates"]]

        for ring in rings:
            if len(ring) < 3:
                continue
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            all_x.extend(xs)
            all_y.extend(ys)
            polys_to_draw.append((ring, label))
            plotted.add(label)

    if not polys_to_draw:
        return set()

    min_y_val, max_y_val = min(all_y), max(all_y)

    for ring, label in polys_to_draw:
        hex_c = color_mapping.get(label, "#000000")
        try:
            rgb = to_rgb(hex_c)
        except ValueError:
            rgb = to_rgb("#000000")
        coords = [[c[0], (max_y_val - c[1] + min_y_val) if mirror_y else c[1]] for c in ring]
        patch = mpatches.Polygon(
            coords[:-1], closed=True,
            facecolor=rgb, edgecolor="white",
            linewidth=0.3, alpha=0.85,
        )
        ax.add_patch(patch)

    pad_x = (max(all_x) - min(all_x)) * 0.05
    pad_y = (max_y_val - min_y_val) * 0.05
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min_y_val - pad_y, max_y_val + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    return plotted


# ---------------------------------------------------------------------------
# Voronoi rendering helper (adapted from Step7)
# ---------------------------------------------------------------------------

def _render_voronoi(
    ax: plt.Axes,
    df_batch: pd.DataFrame,
    color_col: str,
    color_mapping: dict[str, str],
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    max_voronoi_cell_area: float = 400,
    auto_radius_buffer: float = 50,
    manual_radius: float | None = None,
    margin_factor: float = 0.05,
    edge_cell_reduction: bool = False,
    draw_cell_centers: bool = False,
) -> set[str]:
    """Render Voronoi diagram onto *ax*. Returns plotted labels."""
    if Voronoi is None:
        st.error("scipy is required for Voronoi rendering but is not installed.")
        return set()

    valid = df_batch[[x_col, y_col, color_col]].dropna()
    if len(valid) < 4:
        st.warning("Not enough cells for Voronoi diagram (need >= 4).")
        return set()

    points = valid[[x_col, y_col]].values
    labels = valid[color_col].astype(str).values

    # Calculate bounding circle
    x_center = (points[:, 0].min() + points[:, 0].max()) / 2
    y_center = (points[:, 1].min() + points[:, 1].max()) / 2
    
    if manual_radius is not None:
        core_radius = manual_radius
    else:
        dists = np.sqrt((points[:, 0] - x_center) ** 2 + (points[:, 1] - y_center) ** 2)
        core_radius = dists.max() + auto_radius_buffer

    # Nearest-neighbour average spacing
    from scipy.spatial import cKDTree

    kd = cKDTree(points)
    dd, _ = kd.query(points, k=2)
    avg_spacing = dd[:, 1].mean()

    # Boundary ring
    n_bnd = max(int(2 * np.pi * (core_radius + avg_spacing * 3) / (avg_spacing * 0.8)), 36)
    angles = np.linspace(0, 2 * np.pi, n_bnd, endpoint=False)
    bnd_r = core_radius + avg_spacing * 3
    bnd_pts = np.column_stack([x_center + bnd_r * np.cos(angles),
                                y_center + bnd_r * np.sin(angles)])

    all_pts = np.vstack([points, bnd_pts])

    try:
        vor = Voronoi(all_pts)
    except Exception as exc:
        st.warning(f"Voronoi computation failed: {exc}")
        return set()

    max_cell_r = np.sqrt(max_voronoi_cell_area / np.pi)
    plotted: set[str] = set()

    for idx in range(len(points)):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        verts = vor.vertices[region]
        cx, cy = points[idx]
        
        distance_from_center = np.sqrt((cx - x_center) ** 2 + (cy - y_center) ** 2)
        if edge_cell_reduction and distance_from_center > core_radius * 0.9:
            active_max_cell_r = max_cell_r * 0.8
        else:
            active_max_cell_r = max_cell_r
            
        # Clamp oversized cells
        clamped = []
        for v in verts:
            d = np.sqrt((v[0] - cx) ** 2 + (v[1] - cy) ** 2)
            if d <= active_max_cell_r:
                clamped.append(v)
            else:
                angle = np.arctan2(v[1] - cy, v[0] - cx)
                clamped.append([cx + active_max_cell_r * np.cos(angle),
                                cy + active_max_cell_r * np.sin(angle)])
        clamped = np.array(clamped)
        label = labels[idx]
        hex_c = color_mapping.get(label, "#000000")
        try:
            rgb = to_rgb(hex_c)
        except ValueError:
            rgb = to_rgb("#000000")
        patch = mpatches.Polygon(clamped, closed=True,
                                  facecolor=rgb, edgecolor="white",
                                  linewidth=0.2, alpha=0.85)
        ax.add_patch(patch)
        plotted.add(label)

    if draw_cell_centers:
        ax.scatter(points[:, 0], points[:, 1], c="black", s=1, alpha=0.9, edgecolors="none")

    margin = (core_radius * margin_factor) + (avg_spacing * 2)
    ax.set_xlim(x_center - core_radius - margin, x_center + core_radius + margin)
    ax.set_ylim(y_center + core_radius + margin, y_center - core_radius - margin)
    ax.set_aspect("equal")
    ax.axis("off")

    return plotted


# ---------------------------------------------------------------------------
# Continuous coloring renderer
# ---------------------------------------------------------------------------

def _render_continuous(
    ax: plt.Axes,
    df_batch: pd.DataFrame,
    value_col: str,
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    point_size: float = 4,
) -> None:
    """Render cells as scatter coloured by a continuous value."""
    valid = df_batch[[x_col, y_col, value_col]].dropna()
    if valid.empty:
        return
    sc = ax.scatter(
        valid[x_col], valid[y_col],
        c=valid[value_col],
        cmap=cmap_name,
        s=point_size,
        vmin=vmin, vmax=vmax,
        edgecolors="none", alpha=0.85,
    )
    ax.figure.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label=value_col)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Legend builder
# ---------------------------------------------------------------------------

def _add_unified_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    labels: set[str],
    color_mapping: dict[str, str],
    title: str = "Phenotype",
) -> None:
    """Add a legend with consistent colours to *ax*."""
    handles = []
    for lbl in sorted(labels):
        hex_c = color_mapping.get(lbl, "#000000")
        try:
            rgb = to_rgb(hex_c)
        except ValueError:
            rgb = to_rgb("#000000")
        handles.append(mpatches.Patch(facecolor=rgb, edgecolor="black",
                                       linewidth=0.5, label=lbl))
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, title=title, title_fontsize=8,
                  frameon=True, fancybox=True)


# ---------------------------------------------------------------------------
# Figure save / download helper
# ---------------------------------------------------------------------------

def _save_and_show(
    fig: plt.Figure,
    output_dir: str,
    filename: str,
    dpi: int,
    key_suffix: str,
) -> None:
    """Save figure to disk, show in Streamlit, and offer download."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.download_button(
        f"Download {filename}",
        data=buf, file_name=filename, mime="image/png",
        key=f"cellviz_dl_{key_suffix}",
    )
    plt.close(fig)
    st.success(f"Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_cell_visualization(
    project_root: Path,
    unified_csv_path: str,
    available_columns: list[str],
    cached_params: dict[str, Any],
    loaded_df: pd.DataFrame | None = None,
) -> None:
    """Render the Cell Visualization subtab inside Visualization."""
    st.markdown("### Cell Visualization")
    st.caption(
        "Generate publication-ready figures of segmented cells coloured by "
        "categorical phenotypes, continuous markers, or validation matching."
    )

    # ---- guard: CSV must exist ----
    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.info("Please set a valid unified CSV path in the Global Input Configuration above.")
        return

    # ---- use shared loaded dataframe ----
    if not isinstance(loaded_df, pd.DataFrame):
        st.info("Load the unified CSV once using 'Unified CSV Loader' above to activate Cell Visualization.")
        return
    df_full = loaded_df

    all_cols = list(df_full.columns)
    n_rows = len(df_full)
    st.caption(f"Loaded {n_rows:,} cells | {len(all_cols)} columns")

    # ==================================================================
    # A. FILTERING
    # ==================================================================
    st.markdown("---")
    st.markdown("#### A. Data Filtering")

    # ==================================================================
    # CONFIGURATION
    # ==================================================================
    st.markdown("---")
    st.markdown("#### Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    if "batch_id" in all_cols:
        def_batch = "batch_id"
    else:
        def_batch = all_cols[0] if all_cols else ""
        
    if "x_centroid" in all_cols:
        def_x = "x_centroid"
    elif "X:X" in all_cols:
        def_x = "X:X"
    else:
        def_x = all_cols[0] if all_cols else ""
        
    if "y_centroid" in all_cols:
        def_y = "y_centroid"
    elif "Y:Y" in all_cols:
        def_y = "Y:Y"
    else:
        def_y = all_cols[0] if all_cols else ""
        
    with col1:
        batch_col = st.selectbox(
            "Batch Column",
            options=all_cols,
            index=all_cols.index(def_batch) if def_batch in all_cols else 0,
            key="cellviz_config_batch"
        )
    with col2:
        x_col = st.selectbox(
            "X Coordinate Column",
            options=all_cols,
            index=all_cols.index(def_x) if def_x in all_cols else 0,
            key="cellviz_config_x"
        )
    with col3:
        y_col = st.selectbox(
            "Y Coordinate Column",
            options=all_cols,
            index=all_cols.index(def_y) if def_y in all_cols else 0,
            key="cellviz_config_y"
        )

    df_filtered = df_full.copy()

    if batch_col in all_cols:
        unique_batches = sorted(df_full[batch_col].dropna().astype(str).unique())
        selected_batches = st.multiselect(
            f"Filter by `{batch_col}`",
            options=unique_batches,
            default=unique_batches[:1] if unique_batches else [],
            key="cellviz_batch_filter",
        )
        if selected_batches:
            df_filtered = df_filtered[df_filtered[batch_col].astype(str).isin(selected_batches)]
    else:
        selected_batches = []
        st.info(f"Column `{batch_col}` not found; skipping batch filter.")

    # -- additional column filters --
    extra_filter_cols = st.multiselect(
        "Add column filters",
        options=[c for c in all_cols if c not in (batch_col, x_col, y_col)],
        key="cellviz_extra_filters",
    )

    for fc in extra_filter_cols:
        series = df_filtered[fc].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric:
            col1, col2, col3 = st.columns(3)
            with col1:
                op = st.selectbox(f"`{fc}` operator", [">=", "<=", "between"], key=f"cellviz_fop_{fc}")
            with col2:
                v1 = st.number_input(f"`{fc}` value", value=float(series.median()), key=f"cellviz_fv1_{fc}")
            with col3:
                v2 = st.number_input(f"`{fc}` upper", value=float(series.max()), key=f"cellviz_fv2_{fc}",
                                      disabled=(op != "between"))
            if op == ">=":
                df_filtered = df_filtered[df_filtered[fc] >= v1]
            elif op == "<=":
                df_filtered = df_filtered[df_filtered[fc] <= v1]
            else:
                df_filtered = df_filtered[(df_filtered[fc] >= v1) & (df_filtered[fc] <= v2)]
        else:
            uvals = sorted(series.astype(str).unique())
            chosen = st.multiselect(f"Filter `{fc}`", options=uvals, default=uvals,
                                     key=f"cellviz_fcat_{fc}")
            if chosen:
                df_filtered = df_filtered[df_filtered[fc].astype(str).isin(chosen)]

    st.caption(f"Cells after filtering: **{len(df_filtered):,}** / {n_rows:,}")

    if df_filtered.empty:
        st.warning("No cells remain after filtering. Adjust filters above.")
        return

    # ==================================================================
    # B. SEGMENTATION MODE
    # ==================================================================
    st.markdown("---")
    st.markdown("#### B. Segmentation Mode")

    seg_mode = st.radio(
        "Choose cell rendering",
        ["GeoJSON Polygons", "Voronoi Diagram"],
        horizontal=True,
        key="cellviz_seg_mode",
    )

    seg_dir = st.session_state.get("viz_seg_dir", "")
    voronoi_params: dict[str, float] = {}

    if seg_mode == "GeoJSON Polygons":
        if not seg_dir or not os.path.exists(seg_dir):
            st.warning(
                "Segmentation Directory is not set or does not exist. "
                "Configure it in 'Visual Validation Settings' above."
            )
        else:
            n_gj = len(glob.glob(os.path.join(seg_dir, "*.geojson")))
            st.caption(f"GeoJSON directory: `{seg_dir}` ({n_gj} files)")
    else:
        c1, c2 = st.columns(2)
        with c1:
            voronoi_params["max_area"] = st.number_input(
                "Max Voronoi cell area (px^2)", min_value=50, max_value=5000,
                value=int(cached_params.get("cellviz_max_voronoi_area", 400)),
                key="cellviz_max_voronoi_area",
            )
            use_manual_radius = st.checkbox("Use Manual TMA Radius", value=False, key="cellviz_use_man_rad")
            voronoi_params["manual_radius"] = st.number_input(
                "Manual Core Radius (px)", min_value=100, max_value=5000, value=900, step=50,
                key="cellviz_man_rad"
            ) if use_manual_radius else None
            voronoi_params["margin_factor"] = st.number_input(
                "Margin Factor", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                key="cellviz_margin_factor"
            )
        with c2:
            voronoi_params["radius_buf"] = st.number_input(
                "Auto-radius buffer (px)", min_value=0, max_value=500,
                value=int(cached_params.get("cellviz_radius_buf", 50)),
                key="cellviz_radius_buf",
            )
            voronoi_params["edge_cell_reduction"] = st.checkbox(
                "Apply 20% Reduction to Edge Cells", value=False, key="cellviz_edge_red"
            )
            voronoi_params["draw_cell_centers"] = st.checkbox(
                "Draw Cell Centers (Dots)", value=False, key="cellviz_draw_centers"
            )

    # ==================================================================
    # C. COLORING CONFIGURATION
    # ==================================================================
    st.markdown("---")
    st.markdown("#### C. Coloring Mode")

    color_mode = st.radio(
        "Coloring strategy",
        ["Categorical", "Continuous", "Validation (Match / Mismatch)"],
        horizontal=True,
        key="cellviz_color_mode",
    )

    # Detect categorical vs numeric columns
    cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(df_full[c])]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_full[c])
                and c not in (x_col, y_col)]
    pheno_cols = [c for c in all_cols if c.startswith("phenotype_")]

    selected_color_cols: list[str] = []
    continuous_col: str | None = None
    ref_col: str | None = None
    val_cols: list[str] = []
    cmap_name = "viridis"
    clip_min: float | None = None
    clip_max: float | None = None
    unclassified_handling = "treat_as_phenotype"
    categorical_color_mapping: dict[str, str] | None = None
    validation_color_mapping: dict[str, str] | None = None

    if color_mode == "Categorical":
        default_cats = [c for c in pheno_cols if c in cat_cols] or cat_cols[:1]
        selected_color_cols = st.multiselect(
            "Categorical columns to colour by",
            options=cat_cols + pheno_cols,
            default=default_cats[:1] if default_cats else [],
            key="cellviz_cat_cols",
        )
        if selected_color_cols:
            palette_name = st.selectbox(
                "Predefined color scheme",
                options=_CATEGORICAL_PALETTES,
                index=_CATEGORICAL_PALETTES.index(
                    st.session_state.get("cellviz_cat_palette", "Default phenotype map")
                ) if st.session_state.get("cellviz_cat_palette", "Default phenotype map") in _CATEGORICAL_PALETTES else 0,
                key="cellviz_cat_palette",
                help="Apply a preset palette, then fine-tune individual category colors below.",
            )
            categorical_color_mapping = _prepare_categorical_color_mapping(
                df_filtered,
                selected_color_cols,
                palette_name,
            )
            active_labels = _collect_categorical_labels(df_filtered, selected_color_cols)
            with st.expander("Edit category colors", expanded=False):
                st.caption("These overrides apply to the categories present in the selected variables after filtering.")
                _render_color_override_controls(active_labels, categorical_color_mapping)
    elif color_mode == "Continuous":
        continuous_col = st.selectbox(
            "Continuous column (marker intensity)",
            options=num_cols,
            index=0 if num_cols else 0,
            key="cellviz_cont_col",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            cmap_name = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma",
                                                    "cividis", "coolwarm", "RdYlGn"],
                                      key="cellviz_cmap")
        with c2:
            clip_min_val = st.number_input("Clip min (optional)", value=0.0, key="cellviz_clipmin")
            clip_min = clip_min_val if st.checkbox("Use clip min", key="cellviz_use_clipmin") else None
        with c3:
            clip_max_val = st.number_input("Clip max (optional)", value=1.0, key="cellviz_clipmax")
            clip_max = clip_max_val if st.checkbox("Use clip max", key="cellviz_use_clipmax") else None
    else:  # Validation
        default_ref = pheno_cols[:1] if pheno_cols else cat_cols[:1]
        ref_col = st.selectbox(
            "Reference column",
            options=pheno_cols + cat_cols,
            index=0,
            key="cellviz_ref_col",
        )
        remaining = [c for c in pheno_cols + cat_cols if c != ref_col]
        val_cols = st.multiselect(
            "Validation columns (compare against reference)",
            options=remaining,
            default=remaining[:1] if remaining else [],
            key="cellviz_val_cols",
        )
        unclassified_handling = st.selectbox(
            "Unclassified Handling",
            options=["treat_as_phenotype", "separate", "exclude"],
            index=0,
            key="cellviz_unclass_handling"
        )
        st.markdown("##### Validation Colors")
        validation_palette_defaults = {
            "Match": "#00CC00",
            "Mismatch": "#FF0000",
            "Unclassified": "#808080",
        }
        validation_cols = st.columns(3)
        validation_color_mapping = {}
        for idx, label in enumerate(["Match", "Mismatch", "Unclassified"]):
            widget_key = f"cellviz_val_color_{label.lower()}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = validation_palette_defaults[label]
            with validation_cols[idx]:
                validation_color_mapping[label] = st.color_picker(
                    f"{label} color",
                    value=st.session_state.get(widget_key, validation_palette_defaults[label]),
                    key=widget_key,
                )

    # ==================================================================
    # D. OUTPUT SETTINGS
    # ==================================================================
    st.markdown("---")
    st.markdown("#### D. Output Settings")

    c1, c2 = st.columns(2)
    with c1:
        dpi = st.number_input("DPI", min_value=100, max_value=1200, value=300,
                               step=50, key="cellviz_dpi")
    with c2:
        output_dir = st.text_input(
            "Output directory",
            value=cached_params.get(
                "cellviz_output_dir",
                str(project_root / "outputs" / "phenotyping" / "cell_viz"),
            ),
            key="cellviz_output_dir",
        )

    # ==================================================================
    # E. GENERATE FIGURES
    # ==================================================================
    st.markdown("---")

    if st.button("Generate Figures", type="primary", key="cellviz_generate"):
        _do_generate(
            df_filtered=df_filtered,
            seg_mode=seg_mode,
            roi_dir=st.session_state.get("viz_roi_dir", ""),
            seg_dir=seg_dir,
            color_mode=color_mode,
            selected_color_cols=selected_color_cols,
            categorical_color_mapping=categorical_color_mapping,
            continuous_col=continuous_col,
            ref_col=ref_col,
            val_cols=val_cols,
            validation_color_mapping=validation_color_mapping,
            unclassified_handling=unclassified_handling,
            cmap_name=cmap_name,
            clip_min=clip_min,
            clip_max=clip_max,
            voronoi_params=voronoi_params,
            batch_col=batch_col,
            x_col=x_col,
            y_col=y_col,
            dpi=dpi,
            output_dir=output_dir,
            selected_batches=selected_batches,
        )


# ---------------------------------------------------------------------------
# Core figure generation
# ---------------------------------------------------------------------------

def _do_generate(
    *,
    df_filtered: pd.DataFrame,
    seg_mode: str,
    roi_dir: str,
    seg_dir: str,
    color_mode: str,
    selected_color_cols: list[str],
    categorical_color_mapping: dict[str, str] | None,
    continuous_col: str | None,
    ref_col: str | None,
    val_cols: list[str],
    validation_color_mapping: dict[str, str] | None,
    unclassified_handling: str,
    cmap_name: str,
    clip_min: float | None,
    clip_max: float | None,
    voronoi_params: dict[str, float],
    batch_col: str,
    x_col: str,
    y_col: str,
    dpi: int,
    output_dir: str,
    selected_batches: list[str],
) -> None:
    """Execute figure generation based on selected options."""

    geojson_matches: dict[str, str] = {}
    if seg_mode == "GeoJSON Polygons" and batch_col in df_filtered.columns:
        unique_batches = df_filtered[batch_col].dropna().astype(str).unique().tolist()
        geojson_matches = _build_geojson_match_map(roi_dir, seg_dir, unique_batches)

    if color_mode == "Categorical":
        if not selected_color_cols:
            st.warning("Select at least one categorical column.")
            return
        _generate_categorical(
            df_filtered, seg_mode, seg_dir, selected_color_cols,
            categorical_color_mapping,
            voronoi_params, batch_col, x_col, y_col, dpi, output_dir,
            selected_batches, geojson_matches,
        )
    elif color_mode == "Continuous":
        if not continuous_col:
            st.warning("Select a continuous column.")
            return
        _generate_continuous(
            df_filtered, continuous_col, cmap_name, clip_min, clip_max,
            batch_col, x_col, y_col, dpi, output_dir, selected_batches,
        )
    else:
        if not ref_col or not val_cols:
            st.warning("Select a reference column and at least one validation column.")
            return
        _generate_validation(
            df_filtered, ref_col, val_cols, unclassified_handling,
            validation_color_mapping,
            seg_mode, seg_dir, voronoi_params,
            batch_col, x_col, y_col, dpi, output_dir, selected_batches, geojson_matches,
        )


# ---------------------------------------------------------------------------
# Categorical figure generation
# ---------------------------------------------------------------------------

def _generate_categorical(
    df: pd.DataFrame,
    seg_mode: str,
    seg_dir: str,
    color_cols: list[str],
    color_mapping: dict[str, str] | None,
    voronoi_params: dict[str, float],
    batch_col: str,
    x_col: str,
    y_col: str,
    dpi: int,
    output_dir: str,
    selected_batches: list[str],
    geojson_matches: dict[str, str] | None = None,
) -> None:
    # Build unified colour mapping across all selected columns
    color_mapping = color_mapping or _build_unified_color_mapping(df, color_cols)

    # Group by batch
    if batch_col in df.columns and selected_batches:
        groups = [(b, df[df[batch_col].astype(str) == b]) for b in selected_batches]
    else:
        groups = [("all", df)]

    progress = st.progress(0)
    total = len(groups) * len(color_cols)
    done = 0

    for batch_label, df_batch in groups:
        if df_batch.empty:
            continue
        for col in color_cols:
            if col not in df_batch.columns:
                st.warning(f"Column `{col}` not in data for batch `{batch_label}`, skipping.")
                done += 1
                continue

            fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
            fig.patch.set_facecolor("white")

            plotted_labels: set[str] = set()

            if seg_mode == "GeoJSON Polygons":
                geojson_path = _resolve_geojson_path(
                    batch_label=batch_label,
                    df_batch=df_batch,
                    batch_col=batch_col,
                    seg_dir=seg_dir,
                    geojson_matches=geojson_matches,
                )
                if geojson_path:
                    plotted_labels = _render_geojson_polygons(
                        ax, geojson_path, df_batch, col, color_mapping,
                        x_col=x_col, y_col=y_col,
                    )
                else:
                    st.info(f"No GeoJSON found for `{batch_label}`, falling back to Voronoi.")
                    plotted_labels = _render_voronoi(
                        ax, df_batch, col, color_mapping,
                        x_col=x_col, y_col=y_col,
                        max_voronoi_cell_area=voronoi_params.get("max_area", 400),
                        auto_radius_buffer=voronoi_params.get("radius_buf", 50),
                        manual_radius=voronoi_params.get("manual_radius"),
                        margin_factor=voronoi_params.get("margin_factor", 0.05),
                        edge_cell_reduction=voronoi_params.get("edge_cell_reduction", False),
                        draw_cell_centers=voronoi_params.get("draw_cell_centers", False),
                    )
            else:
                plotted_labels = _render_voronoi(
                    ax, df_batch, col, color_mapping,
                    x_col=x_col, y_col=y_col,
                    max_voronoi_cell_area=voronoi_params.get("max_area", 400),
                    auto_radius_buffer=voronoi_params.get("radius_buf", 50),
                    manual_radius=voronoi_params.get("manual_radius"),
                    margin_factor=voronoi_params.get("margin_factor", 0.05),
                    edge_cell_reduction=voronoi_params.get("edge_cell_reduction", False),
                    draw_cell_centers=voronoi_params.get("draw_cell_centers", False),
                )

            title = f"{col} | {batch_label}" if batch_label != "all" else col
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            _add_unified_legend(fig, ax, plotted_labels, color_mapping)

            fname = f"{batch_label}_{col}.png".replace(" ", "_")
            _save_and_show(fig, output_dir, fname, dpi, f"cat_{batch_label}_{col}")
            done += 1
            progress.progress(done / total)

    progress.empty()


# ---------------------------------------------------------------------------
# Continuous figure generation
# ---------------------------------------------------------------------------

def _generate_continuous(
    df: pd.DataFrame,
    value_col: str,
    cmap_name: str,
    clip_min: float | None,
    clip_max: float | None,
    batch_col: str,
    x_col: str,
    y_col: str,
    dpi: int,
    output_dir: str,
    selected_batches: list[str],
) -> None:
    if batch_col in df.columns and selected_batches:
        groups = [(b, df[df[batch_col].astype(str) == b]) for b in selected_batches]
    else:
        groups = [("all", df)]

    progress = st.progress(0)
    total = len(groups)
    done = 0

    for batch_label, df_batch in groups:
        if df_batch.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
        fig.patch.set_facecolor("white")
        _render_continuous(ax, df_batch, value_col, x_col=x_col, y_col=y_col,
                           cmap_name=cmap_name, vmin=clip_min, vmax=clip_max)
        title = f"{value_col} | {batch_label}" if batch_label != "all" else value_col
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        fname = f"{batch_label}_{value_col}_continuous.png".replace(" ", "_")
        _save_and_show(fig, output_dir, fname, dpi, f"cont_{batch_label}_{value_col}")
        done += 1
        progress.progress(done / total)

    progress.empty()


# ---------------------------------------------------------------------------
# Validation (match / mismatch) figure generation
# ---------------------------------------------------------------------------

def _generate_validation(
    df: pd.DataFrame,
    ref_col: str,
    val_cols: list[str],
    unclassified_handling: str,
    validation_color_mapping: dict[str, str] | None,
    seg_mode: str,
    seg_dir: str,
    voronoi_params: dict[str, float],
    batch_col: str,
    x_col: str,
    y_col: str,
    dpi: int,
    output_dir: str,
    selected_batches: list[str],
    geojson_matches: dict[str, str] | None = None,
) -> None:
    validation_color_mapping = validation_color_mapping or {}
    match_color = str(validation_color_mapping.get("Match", "#00CC00"))
    mismatch_color = str(validation_color_mapping.get("Mismatch", "#FF0000"))
    unclassified_color = str(validation_color_mapping.get("Unclassified", "#808080"))

    if batch_col in df.columns and selected_batches:
        groups = [(b, df[df[batch_col].astype(str) == b]) for b in selected_batches]
    else:
        groups = [("all", df)]

    progress = st.progress(0)
    total = len(groups) * len(val_cols)
    done = 0

    for batch_label, df_batch in groups:
        if df_batch.empty:
            continue
        for vc in val_cols:
            if ref_col not in df_batch.columns or vc not in df_batch.columns:
                st.warning(f"Missing column `{ref_col}` or `{vc}` in batch `{batch_label}`.")
                done += 1
                continue

            # Create match column
            tmp = df_batch.copy()
            is_na_ref = tmp[ref_col].isna()
            is_na_vc = tmp[vc].isna()
            is_unclass_ref = tmp[ref_col].astype(str).str.lower() == "unclassified"
            is_unclass_vc = tmp[vc].astype(str).str.lower() == "unclassified"
            
            if unclassified_handling == "treat_as_phenotype":
                mask_unclassified = is_na_ref | is_na_vc
            else:
                mask_unclassified = is_na_ref | is_na_vc | is_unclass_ref | is_unclass_vc
                
            mask_match = (~mask_unclassified) & (tmp[ref_col].astype(str) == tmp[vc].astype(str))
            
            if unclassified_handling == "exclude":
                tmp = tmp[~mask_unclassified].copy()
                mask_unclassified = pd.Series(False, index=tmp.index)
                mask_match = tmp[ref_col].astype(str) == tmp[vc].astype(str)
                
            total_comparable = (~mask_unclassified).sum() if unclassified_handling != "exclude" else len(tmp)
            matching_cells = mask_match.sum()
            accuracy_pct = (matching_cells / total_comparable * 100) if total_comparable > 0 else 0

            # Assign validation labels
            tmp["_val_color"] = mismatch_color
            tmp.loc[mask_match, "_val_color"] = match_color
            tmp.loc[mask_unclassified, "_val_color"] = unclassified_color

            tmp["_val_label"] = "Mismatch"
            tmp.loc[mask_match, "_val_label"] = "Match"
            tmp.loc[mask_unclassified, "_val_label"] = "Unclassified"

            val_mapping = {
                "Match": match_color,
                "Mismatch": mismatch_color,
                "Unclassified": unclassified_color,
            }

            fig, ax = plt.subplots(figsize=(12, 9), dpi=dpi)
            fig.patch.set_facecolor("white")

            if seg_mode == "GeoJSON Polygons":
                geojson_path = _resolve_geojson_path(
                    batch_label=batch_label,
                    df_batch=tmp,
                    batch_col=batch_col,
                    seg_dir=seg_dir,
                    geojson_matches=geojson_matches,
                )
                if geojson_path:
                    plotted = _render_geojson_polygons(
                        ax, geojson_path, tmp, "_val_label", val_mapping,
                        x_col=x_col, y_col=y_col,
                    )
                else:
                    plotted = _render_voronoi(
                        ax, tmp, "_val_label", val_mapping,
                        x_col=x_col, y_col=y_col,
                        max_voronoi_cell_area=voronoi_params.get("max_area", 400),
                        auto_radius_buffer=voronoi_params.get("radius_buf", 50),
                        manual_radius=voronoi_params.get("manual_radius"),
                        margin_factor=voronoi_params.get("margin_factor", 0.05),
                        edge_cell_reduction=voronoi_params.get("edge_cell_reduction", False),
                        draw_cell_centers=voronoi_params.get("draw_cell_centers", False),
                    )
            else:
                plotted = _render_voronoi(
                    ax, tmp, "_val_label", val_mapping,
                    x_col=x_col, y_col=y_col,
                    max_voronoi_cell_area=voronoi_params.get("max_area", 400),
                    auto_radius_buffer=voronoi_params.get("radius_buf", 50),
                    manual_radius=voronoi_params.get("manual_radius"),
                    margin_factor=voronoi_params.get("margin_factor", 0.05),
                    edge_cell_reduction=voronoi_params.get("edge_cell_reduction", False),
                    draw_cell_centers=voronoi_params.get("draw_cell_centers", False),
                )

            title = (
                f"{vc} vs {ref_col} | {batch_label}\n"
                f"Accuracy: {accuracy_pct:.1f}% ({matching_cells}/{total_comparable})"
            )
            ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
            _add_unified_legend(fig, ax, plotted, val_mapping, title="Validation")

            fname = f"{batch_label}_{vc}_vs_{ref_col}_validation.png".replace(" ", "_")
            _save_and_show(fig, output_dir, fname, dpi, f"val_{batch_label}_{vc}")
            done += 1
            progress.progress(done / total)

    progress.empty()
