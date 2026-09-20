from __future__ import annotations

import json
import uuid
from itertools import combinations
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box, mapping
from shapely.ops import unary_union


REQUIRED_COLS = ["x_centroid", "y_centroid"]


def _column_options(columns: list[str]) -> list[str]:
    return [""] + list(columns)


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _count_holes(geom) -> int:
    if geom is None:
        return 0
    if geom.geom_type == "Polygon":
        return len(list(getattr(geom, "interiors", [])))
    if geom.geom_type == "MultiPolygon":
        total = 0
        for poly in list(getattr(geom, "geoms", [])):
            total += len(list(getattr(poly, "interiors", [])))
        return total
    return 0


def _alpha_shape(points: np.ndarray, alpha: float):
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()

    for simplex in tri.simplices:
        p1, p2, p3 = points[simplex]
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2.0

        expr = s * (s - a) * (s - b) * (s - c)
        if expr <= 0:
            continue

        area = np.sqrt(expr)
        if area < 1e-10:
            continue

        circumradius = (a * b * c) / (4.0 * area)
        if circumradius < alpha:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[2], simplex[0]))))

    if not edges:
        return MultiPoint(points).convex_hull

    lines = [LineString([points[i], points[j]]) for i, j in edges]
    result = unary_union(lines)
    return result if result.geom_type in {"Polygon", "MultiPolygon"} else MultiPoint(points).convex_hull


def _process_polygon_holes(polygon, min_hole_size: float):
    if polygon.geom_type != "Polygon":
        return polygon
    exterior = polygon.exterior
    large_holes = [hole for hole in polygon.interiors if Polygon(hole).area >= min_hole_size]
    return Polygon(exterior, large_holes)


def _round_coords(obj):
    if isinstance(obj, dict):
        if obj.get("type") == "Polygon":
            obj["coordinates"] = [
                [[round(pt[0], 1), round(pt[1], 1)] for pt in ring] for ring in obj["coordinates"]
            ]
        elif obj.get("type") == "MultiPolygon":
            obj["coordinates"] = [
                [[[round(pt[0], 1), round(pt[1], 1)] for pt in ring] for ring in polygon]
                for polygon in obj["coordinates"]
            ]
        else:
            for value in obj.values():
                _round_coords(value)
    elif isinstance(obj, list):
        for item in obj:
            _round_coords(item)
    return obj


def _hex_to_rgb255(hex_color: str) -> list[int]:
    text = str(hex_color).strip().lstrip("#")
    if len(text) != 6:
        return [0, 255, 128]
    try:
        return [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)]
    except Exception:
        return [0, 255, 128]


def _normalize_marker(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if arr.size == 0:
        return arr

    if mode == "none":
        return arr
    if mode == "log1p":
        min_val = np.nanmin(arr)
        if np.isfinite(min_val) and min_val < 0:
            arr = arr - min_val
        return np.log1p(np.clip(arr, a_min=0, a_max=None))
    if mode == "clip_1_99":
        lo = np.nanpercentile(arr, 1)
        hi = np.nanpercentile(arr, 99)
        return np.clip(arr, lo, hi)
    if mode == "zscore":
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        if not np.isfinite(std) or std <= 0:
            return arr * np.nan
        return (arr - mean) / std
    if mode == "log1p_zscore":
        shifted = arr.copy()
        min_val = np.nanmin(shifted)
        if np.isfinite(min_val) and min_val < 0:
            shifted = shifted - min_val
        logged = np.log1p(np.clip(shifted, a_min=0, a_max=None))
        mean = np.nanmean(logged)
        std = np.nanstd(logged)
        if not np.isfinite(std) or std <= 0:
            return logged * np.nan
        return (logged - mean) / std
    return arr


def _build_fuzzy_cloud(
    coords: np.ndarray,
    values: np.ndarray,
    buffer_distance_px: float,
    bins: int,
    sigma_scale: float,
    density_weight: bool,
) -> tuple[np.ndarray, list[float]]:
    """Create a buffer-aware fuzzy cloud field for marker visualization."""
    if coords.size == 0:
        return np.zeros((10, 10), dtype=float), [0, 1, 0, 1]

    x = coords[:, 0].astype(float)
    y = coords[:, 1].astype(float)
    v = values.astype(float)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    if finite.sum() == 0:
        return np.zeros((10, 10), dtype=float), [0, 1, 0, 1]

    x = x[finite]
    y = y[finite]
    v = v[finite]

    # Robust scale to [0,1] for stable cloud intensities across markers/ranges.
    # Important: if values are constant (common for categorical clouds with all ones),
    # keep a non-zero signal so the cloud remains visible.
    v_lo = np.nanpercentile(v, 1)
    v_hi = np.nanpercentile(v, 99)
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or v_hi <= v_lo:
        v_scaled = np.ones_like(v)
    else:
        v_scaled = np.clip((v - v_lo) / (v_hi - v_lo), 0.0, 1.0)

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

    field[count_blur <= 1e-12] = np.nan
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return field, extent


def _mask_cloud_to_geometry(field: np.ndarray, extent: list[float], geom) -> np.ndarray:
    """Mask fuzzy cloud outside geometry so heatmap is cropped to virtual tissue."""
    if geom is None or field.size == 0:
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


def _transparent_to_color_cmap(hex_color: str, name: str) -> LinearSegmentedColormap:
    r, g, b = _hex_to_rgb255(hex_color)
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    return LinearSegmentedColormap.from_list(
        name,
        [
            (r_f, g_f, b_f, 0.0),
            (r_f, g_f, b_f, 1.0),
        ],
    )


def _is_category_like(series: pd.Series, max_unique: int = 30) -> bool:
    """Return True for categorical-like columns (including low-cardinality numeric encodings)."""
    if series is None:
        return False
    is_category_dtype = str(getattr(series, "dtype", "")) == "category"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or is_category_dtype:
        return True
    if pd.api.types.is_numeric_dtype(series):
        non_na = series.dropna()
        if non_na.empty:
            return False
        nunique = int(non_na.nunique())
        if nunique <= max_unique:
            return True
    return False


def _compute_buffer_distance(coords: np.ndarray, use_median: bool, percentile: float) -> float:
    if coords is None or len(coords) < 2:
        return np.nan
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    if use_median:
        return float(np.median(nearest))
    return float(np.percentile(nearest, percentile))


def _build_region_geometry(
    coords: np.ndarray,
    method: str,
    alpha: float,
    buffer_distance: float,
    min_hole_size: float,
):
    points = [Point(float(x), float(y)) for x, y in coords]
    if method == "buffer_union":
        geom = unary_union([point.buffer(buffer_distance) for point in points])
    else:
        try:
            alpha_poly = _alpha_shape(coords, float(alpha))
            geom = alpha_poly.buffer(buffer_distance)
        except Exception:
            geom = MultiPoint(coords).convex_hull.buffer(buffer_distance)

    if geom.geom_type == "Polygon":
        return _process_polygon_holes(geom, min_hole_size)
    if geom.geom_type == "MultiPolygon":
        geoms = list(getattr(geom, "geoms", []))
        return unary_union([_process_polygon_holes(poly, min_hole_size) for poly in geoms])
    return geom


def _split_overlap_by_nearest(
    overlap_geom,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    step_px: float,
):
    if overlap_geom is None or overlap_geom.is_empty:
        return overlap_geom, overlap_geom
    if coords_a is None or coords_b is None or len(coords_a) == 0 or len(coords_b) == 0:
        return overlap_geom, overlap_geom

    tree_a = cKDTree(coords_a)
    tree_b = cKDTree(coords_b)

    minx, miny, maxx, maxy = overlap_geom.bounds
    step = max(float(step_px), 1.0)
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)

    boxes_a = []
    boxes_b = []
    half = step / 2.0

    for x in xs:
        for y in ys:
            pt = Point(float(x), float(y))
            if not overlap_geom.covers(pt):
                continue
            d_a = float(tree_a.query([x, y], k=1)[0])
            d_b = float(tree_b.query([x, y], k=1)[0])
            pix = box(x - half, y - half, x + half, y + half)
            if d_a <= d_b:
                boxes_a.append(pix)
            else:
                boxes_b.append(pix)

    geom_a = unary_union(boxes_a).intersection(overlap_geom) if boxes_a else overlap_geom.buffer(0)
    geom_b = unary_union(boxes_b).intersection(overlap_geom) if boxes_b else overlap_geom.buffer(0)
    return geom_a, geom_b


def render_virtual_tissue(project_root: Path, df: pd.DataFrame):
    st.subheader("Virtual Tissue")
    st.caption("Generate virtual tissue regions (GeoJSON + figures) using buffer union or alpha-shape methods.")

    if len(df.columns) < 2:
        st.error("Need at least two columns to select X/Y coordinates.")
        return

    all_columns = list(df.columns)
    coord_options = _column_options(all_columns)
    default_x_col = _preferred_or_blank(coord_options, "x_centroid")
    default_y_col = _preferred_or_blank(coord_options, "y_centroid")

    with st.expander("Configuration", expanded=True):
        xy1, xy2 = st.columns(2)
        selected_x_coord_col = xy1.selectbox(
            "X Coordinate Column",
            options=coord_options,
            index=_column_options(all_columns).index(default_x_col) if default_x_col in coord_options else 0,
            key="tab9_vt_x_coord_col",
            format_func=lambda x: x if x else "(not selected)",
        )
        selected_y_coord_col = xy2.selectbox(
            "Y Coordinate Column",
            options=coord_options,
            index=_column_options(all_columns).index(default_y_col) if default_y_col in coord_options else 0,
            key="tab9_vt_y_coord_col",
            format_func=lambda x: x if x else "(not selected)",
        )

        col1, col2, col3 = st.columns(3)

        selectable_cols = [c for c in df.columns if c not in {selected_x_coord_col, selected_y_coord_col, ""}]

        batch_col_default_idx = 0
        batch_col_options = [""] + selectable_cols
        if "batch_id" in selectable_cols:
            batch_col_default_idx = batch_col_options.index("batch_id")

        selected_batch_col = col1.selectbox(
            "Batch ID Column",
            options=batch_col_options,
            index=batch_col_default_idx,
            key="tab9_vt_batch_col",
            help="Select the column that identifies image/ROI batches.",
            format_func=lambda x: x if x else "(not selected)",
        )

        selected_batch = None
        if selected_batch_col:
            batch_ids = sorted(df[selected_batch_col].dropna().astype(str).unique().tolist())
            if batch_ids:
                selected_batch = col2.selectbox(
                    "Batch Selection",
                    options=batch_ids,
                    key="tab9_vt_batch",
                    help="Select which batch value to build the virtual tissue boundary from.",
                )
            else:
                col2.caption("No non-null batch values found in selected Batch ID column.")
        else:
            col2.caption("Select a Batch ID Column first")

        tissue_col_options = ["None"] + selectable_cols
        tissue_col_default_idx = 0
        if "tissue_roi" in selectable_cols:
            tissue_col_default_idx = tissue_col_options.index("tissue_roi")
        selected_tissue_col = col3.selectbox(
            "Tissue ROI Column (optional)",
            options=tissue_col_options,
            index=tissue_col_default_idx,
            key="tab9_vt_tissue_col",
            help="Select the column that stores Tissue ROI labels. Leave as None to disable Tissue ROI filtering.",
        )

        col4, col5 = st.columns(2)
        selected_tissue = "All"
        if selected_tissue_col != "None" and selected_batch_col and selected_batch is not None:
            batch_mask_df = df[df[selected_batch_col].astype(str) == str(selected_batch)]
            tissue_values = ["All"] + sorted(batch_mask_df[selected_tissue_col].dropna().astype(str).unique().tolist())
            selected_tissue = col4.selectbox(
                "Tissue ROI Selection",
                options=tissue_values,
                key="tab9_vt_tissue",
                help="Choose 'All' or a specific Tissue ROI value from the selected Tissue ROI column.",
            )
        elif selected_tissue_col != "None":
            col4.caption("Select Batch ID Column and Batch Selection to list Tissue ROI values")
        else:
            col4.caption("Tissue ROI filtering disabled")

        phenotype_cols = [c for c in df.columns if "phenotype" in c.lower()]
        selected_pheno_col = col5.selectbox(
            "Phenotype Column",
            options=["None"] + phenotype_cols,
            key="tab9_vt_pheno_col",
            help="Choose a phenotype column if you want to generate virtual tissue for specific phenotypes only.",
        )

        selected_phenotypes: list[str] = []
        if selected_pheno_col != "None":
            vals = sorted(df[selected_pheno_col].dropna().astype(str).unique().tolist())
            selected_phenotypes = st.multiselect(
                "Phenotype Values (optional)",
                options=vals,
                default=[],
                key="tab9_vt_pheno_vals",
                help="Leave empty to include all phenotypes from the selected phenotype column.",
            )

        p1, p2, p3, p4 = st.columns(4)
        method = p1.selectbox(
            "Method",
            options=["buffer_union", "alpha_shape"],
            key="tab9_vt_method",
            help="buffer_union: robust default. alpha_shape: legacy concave-hull-style envelope then buffered.",
        )
        use_median = p2.checkbox(
            "Use median distance",
            value=False,
            key="tab9_vt_use_median",
            help="If checked, buffer radius uses median nearest-neighbor distance. If unchecked, percentile slider is used.",
        )
        percentile = p3.slider(
            "Buffer Percentile",
            min_value=90.0,
            max_value=99.9,
            value=99.9,
            step=0.1,
            key="tab9_vt_percentile",
            help="Percentile of nearest-neighbor distances used as buffer radius. Higher = broader tissue envelope.",
        )
        alpha = p4.number_input(
            "Alpha (if alpha_shape)",
            min_value=1.0,
            max_value=500.0,
            value=100.0,
            step=1.0,
            key="tab9_vt_alpha",
            help="Only used with alpha_shape method. Lower values create tighter hulls; higher values approach convex-like hulls.",
        )

        q1, q2, q3 = st.columns(3)
        min_hole_size = q1.number_input(
            "Min Hole Size (pixels²)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            key="tab9_vt_min_hole",
            help="Holes smaller than this area are filled (treated as noise). Larger holes are preserved.",
        )
        pixel_size_um = q2.number_input(
            "Pixel Size (µm)",
            min_value=0.01,
            value=0.5083,
            step=0.0001,
            format="%.4f",
            key="tab9_vt_pixel_um",
            help="Micrometers per pixel; used only for area conversion and density in cells/mm².",
        )
        output_dir = q3.text_input(
            "Output Directory",
            value=str(project_root / "outputs" / "spatial_analysis" / "virtual_tissue"),
            key="tab9_vt_outdir",
            help="Directory where GeoJSON, summary CSV, and figure PNG will be saved.",
        )

        if selected_batch_col and selected_batch is not None:
            preview_df = df[df[selected_batch_col].astype(str) == str(selected_batch)].copy()
        else:
            preview_df = df.copy()

        if selected_tissue != "All" and selected_tissue_col != "None" and selected_tissue_col in preview_df.columns:
            preview_df = preview_df[preview_df[selected_tissue_col].astype(str) == str(selected_tissue)]
        if selected_pheno_col != "None" and selected_phenotypes:
            preview_df = preview_df[preview_df[selected_pheno_col].astype(str).isin(selected_phenotypes)]

        st.markdown("#### Annotation Regions (optional)")
        st.caption("Define additional within-tissue regions from annotation columns (e.g., stroma=1, tumor=1).")

        excluded_cols = {"batch_id", "x_centroid", "y_centroid", selected_x_coord_col, selected_y_coord_col}
        annotation_candidates = [c for c in df.columns if c not in excluded_cols]
        selected_annot_cols = st.multiselect(
            "Annotation columns",
            options=annotation_candidates,
            default=[],
            key="tab9_vt_annot_cols",
            help="Pick one or more annotation columns. For each, choose the value representing positive cells and a display color.",
        )

        annotation_configs: list[dict] = []
        default_palette = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#ff7f0e"]
        for idx, col in enumerate(selected_annot_cols):
            vals = preview_df[col].dropna().astype(str).unique().tolist()
            try:
                vals = sorted(vals)
            except Exception:
                pass
            if not vals:
                st.warning(f"Annotation column '{col}' has no non-null values after current filters.")
                continue

            c_a, c_b, c_c = st.columns([2, 1.2, 1])
            selected_val = c_a.selectbox(
                f"{col}: positive value",
                options=vals,
                key=f"tab9_vt_annot_val_{idx}_{col}",
                help="Cells with this value in the selected annotation column are used to form this region.",
            )
            color_hex = c_b.color_picker(
                f"{col}: color",
                value=default_palette[idx % len(default_palette)],
                key=f"tab9_vt_annot_color_{idx}_{col}",
                help="Display and export color for this annotation region.",
            )
            c_c.caption(f"value={selected_val}")

            annotation_configs.append(
                {
                    "column": col,
                    "value": selected_val,
                    "color_hex": color_hex,
                    "color_rgb": _hex_to_rgb255(color_hex),
                }
            )

        st.markdown("#### Marker Heatmap Overlay (optional)")
        st.caption("Overlay continuous marker expression values as heatmaps within the virtual tissue.")

        excluded_numeric = {
            "x_centroid",
            "y_centroid",
            selected_x_coord_col,
            selected_y_coord_col,
            "tissue",
            "tissue_roi",
            "batch_id",
            "bed_roi",
        }
        numeric_cols = [
            c for c in preview_df.columns
            if pd.api.types.is_numeric_dtype(preview_df[c]) and c not in excluded_numeric
        ]
        selected_heatmap_markers = st.multiselect(
            "Marker columns (continuous)",
            options=numeric_cols,
            default=[],
            key="tab9_vt_heatmap_markers",
            help="Choose one or more continuous marker columns (e.g., intensity from 16-bit images).",
        )

        h1, h2, h3, h4 = st.columns(4)
        heatmap_norm = h1.selectbox(
            "Normalization",
            options=["none", "log1p", "clip_1_99", "zscore", "log1p_zscore"],
            index=1,
            key="tab9_vt_heatmap_norm",
            help="none: raw intensity; log1p: compresses dynamic range; clip_1_99: trims outliers; zscore/log1p_zscore: relative contrast.",
        )
        heatmap_cmap = h2.selectbox(
            "Colormap",
            options=["inferno", "viridis", "magma", "plasma", "turbo"],
            index=0,
            key="tab9_vt_heatmap_cmap",
            help="Color map for expression heatmaps.",
        )
        heatmap_alpha = h3.slider(
            "Heatmap alpha",
            min_value=0.1,
            max_value=1.0,
            value=0.85,
            step=0.05,
            key="tab9_vt_heatmap_alpha",
            help="Transparency of marker points in the heatmap overlay.",
        )
        heatmap_pt_size = h4.slider(
            "Heatmap point size",
            min_value=2,
            max_value=60,
            value=8,
            step=1,
            key="tab9_vt_heatmap_pt_size",
            help="Marker point size used in heatmap overlays.",
        )

        k1, k2, k3 = st.columns(3)
        heatmap_style = k1.selectbox(
            "Heatmap style",
            options=["point_cloud", "fuzzy_cloud"],
            index=0,
            key="tab9_vt_heatmap_style",
            help="point_cloud: color each cell point. fuzzy_cloud: buffer-aware smoothed cloud where dense regions appear darker.",
        )
        cloud_bins = k2.slider(
            "Cloud grid bins",
            min_value=80,
            max_value=500,
            value=220,
            step=10,
            key="tab9_vt_heatmap_bins",
            help="Resolution of fuzzy cloud grid. Higher values produce finer detail but are slower.",
        )
        cloud_sigma_scale = k3.slider(
            "Cloud spread scale",
            min_value=0.5,
            max_value=2.5,
            value=1.0,
            step=0.1,
            key="tab9_vt_heatmap_sigma_scale",
            help="Multiplier for cloud spread derived from selected buffer distance.",
        )

        cloud_density_weight = st.checkbox(
            "Fuzzy cloud: weight by local cell density",
            value=True,
            key="tab9_vt_heatmap_density_weight",
            help="When enabled, cloud intensity combines marker signal and local cell density (denser areas darker).",
        )

        copt1, copt2 = st.columns(2)
        heatmap_clip_to_tissue = copt1.checkbox(
            "Clip heatmap to virtual tissue boundary",
            value=True,
            key="tab9_vt_heatmap_clip_tissue",
            help="Removes outside margins by masking heatmap values outside the virtual tissue geometry.",
        )
        heatmap_show_context = copt2.checkbox(
            "Show tissue + annotation context under heatmap",
            value=True,
            key="tab9_vt_heatmap_show_context",
            help="Draws the selected virtual tissue and annotation overlays beneath the heatmap cloud.",
        )

        if len(selected_heatmap_markers) > 6:
            st.info("More than 6 markers can look busy; consider 1-6 markers for clearer small-multiples.")

        st.markdown("#### Categorical Density Overlay (optional)")
        st.caption("Create point/fuzzy density overlays for selected categories (e.g., phenotype_leiden = Epithelial).")

        cat_excluded = {"x_centroid", "y_centroid", selected_x_coord_col, selected_y_coord_col}
        categorical_candidates = [
            c for c in preview_df.columns
            if c not in cat_excluded and _is_category_like(preview_df[c], max_unique=30)
        ]
        selected_cat_col = st.selectbox(
            "Categorical column",
            options=["None"] + categorical_candidates,
            index=0,
            key="tab9_vt_cat_col",
            help="Pick a categorical column for category-specific density overlays.",
        )

        selected_cat_values: list[str] = []
        cat_value_colors: dict[str, str] = {}
        cat_style = "fuzzy_cloud"
        cat_alpha = 0.7
        cat_point_size = 8
        if selected_cat_col != "None":
            cat_vals = preview_df[selected_cat_col].dropna().astype(str).unique().tolist()
            try:
                cat_vals = sorted(cat_vals)
            except Exception:
                pass
            default_cat_values = cat_vals[:1] if cat_vals else []
            selected_cat_values = st.multiselect(
                "Category values",
                options=cat_vals,
                default=default_cat_values,
                key="tab9_vt_cat_values",
                help="Selected categories will be overlaid as density/point maps.",
            )
            cat_style = st.selectbox(
                "Categorical overlay style",
                options=["fuzzy_cloud", "point_cloud"],
                index=0,
                key="tab9_vt_cat_style",
                help="fuzzy_cloud: smoothed category density cloud; point_cloud: raw category points.",
            )
            ccat1, ccat2 = st.columns(2)
            cat_alpha = ccat1.slider(
                "Categorical alpha",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="tab9_vt_cat_alpha",
                help="Transparency for categorical overlays.",
            )
            cat_point_size = ccat2.slider(
                "Categorical point size",
                min_value=2,
                max_value=60,
                value=8,
                step=1,
                key="tab9_vt_cat_point_size",
                help="Point size when categorical style is point_cloud.",
            )

            cat_palette = ["#00bcd4", "#ff6f00", "#8e24aa", "#43a047", "#e53935", "#3949ab", "#6d4c41"]
            for idx, val in enumerate(selected_cat_values):
                cat_value_colors[val] = st.color_picker(
                    f"Category color: {val}",
                    value=cat_palette[idx % len(cat_palette)],
                    key=f"tab9_vt_cat_color_{idx}_{val}",
                    help="Display color for this category overlay.",
                )

        st.markdown("#### Overlay Layer Manager")
        st.caption("Enable/disable layers and set draw order (higher z-order is drawn on top).")
        with st.expander("How to use this (quick)", expanded=False):
            st.markdown(
                "- Toggle a layer checkbox to show/hide it.  \n"
                "- Set **z** values to control stacking: higher numbers are drawn on top.  \n"
                "- Simple default order: tissue=1, annotations=3, continuous=6, categorical=7.  \n"
                "- Pick one continuous marker in **Continuous marker for composite overlay**.  \n"
                "- Press **Run Virtual Tissue** to regenerate with current layer settings."
            )

        l1, l2, l3, l4 = st.columns(4)
        layer_show_tissue = l1.checkbox("Show virtual tissue", value=True, key="tab9_vt_layer_tissue")
        layer_show_annotations = l2.checkbox("Show annotations", value=True, key="tab9_vt_layer_annotations")
        layer_show_cont = l3.checkbox("Show continuous overlay", value=True, key="tab9_vt_layer_cont")
        layer_show_cat = l4.checkbox("Show categorical overlay", value=True, key="tab9_vt_layer_cat")

        z1, z2, z3, z4 = st.columns(4)
        z_tissue = z1.number_input("z: tissue", min_value=0, max_value=50, value=1, step=1, key="tab9_vt_z_tissue")
        z_annotations = z2.number_input("z: annotations", min_value=0, max_value=50, value=3, step=1, key="tab9_vt_z_annotations")
        z_cont = z3.number_input("z: continuous", min_value=0, max_value=50, value=6, step=1, key="tab9_vt_z_cont")
        z_cat = z4.number_input("z: categorical", min_value=0, max_value=50, value=7, step=1, key="tab9_vt_z_cat")

        cont_marker_for_composite = st.selectbox(
            "Continuous marker for composite overlay",
            options=["None"] + selected_heatmap_markers,
            index=0,
            key="tab9_vt_cont_marker_for_composite",
            help="Choose one continuous marker to include in the composite overlay figure.",
        )

    if not selected_x_coord_col or not selected_y_coord_col:
        st.info("Select X and Y coordinate columns to enable virtual tissue generation.")
        return

    if selected_x_coord_col == selected_y_coord_col:
        st.error("X Coordinate Column and Y Coordinate Column must be different.")
        return

    if not st.button("Run Virtual Tissue", type="primary", key="tab9_vt_run"):
        return

    if not selected_batch_col:
        st.error("Please select a Batch ID Column.")
        return
    if selected_batch is None:
        st.error("Please select a Batch Selection value.")
        return

    run_df = df[df[selected_batch_col].astype(str) == str(selected_batch)].copy()
    if selected_tissue != "All" and selected_tissue_col != "None" and selected_tissue_col in run_df.columns:
        run_df = run_df[run_df[selected_tissue_col].astype(str) == str(selected_tissue)]
    if selected_pheno_col != "None" and selected_phenotypes:
        run_df = run_df[run_df[selected_pheno_col].astype(str).isin(selected_phenotypes)]

    run_df["x_centroid"] = pd.to_numeric(run_df[selected_x_coord_col], errors="coerce")
    run_df["y_centroid"] = pd.to_numeric(run_df[selected_y_coord_col], errors="coerce")

    run_df = run_df.dropna(subset=["x_centroid", "y_centroid"])
    if len(run_df) < 10:
        st.error(f"Insufficient cells after filtering: n={len(run_df)} (need at least 10).")
        return

    coords = run_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]

    median_dist = float(np.median(nearest))
    p95 = float(np.percentile(nearest, 95))
    p99 = float(np.percentile(nearest, 99))

    buffer_distance = median_dist if use_median else float(np.percentile(nearest, percentile))
    buffer_label = "median" if use_median else f"{percentile:.1f}th percentile"

    cell_points = [Point(float(x), float(y)) for x, y in coords]

    if method == "buffer_union":
        virtual_tissue = unary_union([point.buffer(buffer_distance) for point in cell_points])
    else:
        try:
            alpha_poly = _alpha_shape(coords, float(alpha))
            virtual_tissue = alpha_poly.buffer(buffer_distance)
        except Exception:
            virtual_tissue = MultiPoint(coords).convex_hull.buffer(buffer_distance)

    if virtual_tissue.geom_type == "Polygon":
        holes_before = _count_holes(virtual_tissue)
        final_polygon = _process_polygon_holes(virtual_tissue, min_hole_size)
        holes_after = _count_holes(final_polygon)
    elif virtual_tissue.geom_type == "MultiPolygon":
        holes_before = _count_holes(virtual_tissue)
        geoms = list(getattr(virtual_tissue, "geoms", []))
        final_polygon = unary_union([_process_polygon_holes(poly, min_hole_size) for poly in geoms])
        holes_after = _count_holes(final_polygon)
    else:
        final_polygon = virtual_tissue
        holes_before = 0
        holes_after = 0

    pixel_size_mm = float(pixel_size_um) / 1000.0
    area_pixels = float(final_polygon.area)
    area_mm2 = area_pixels * (pixel_size_mm ** 2)
    density_cells_per_mm2 = len(run_df) / area_mm2 if area_mm2 > 0 else np.nan

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tissue_suffix = f"_roi{selected_tissue}" if selected_tissue != "All" else "_all"
    pheno_suffix = "_all" if not selected_phenotypes else "_" + "_".join([p.replace(" ", "-") for p in selected_phenotypes[:3]])
    stem = f"{selected_batch}{tissue_suffix}{pheno_suffix}_virtual_tissue"

    geojson_path = out_dir / f"{stem}.geojson"
    fig_path = out_dir / f"{stem}.png"
    summary_path = out_dir / f"{stem}_summary.csv"
    heatmap_path = out_dir / f"{stem}_marker_heatmaps.png"
    composite_path = out_dir / f"{stem}_composite_overlay.png"

    feature = {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "properties": {
            "classification": {"name": "virtual_tissue", "color": [0, 255, 128]},
            "isLocked": False,
            "measurements": [],
        },
        "geometry": mapping(final_polygon),
    }
    geojson_features: list[dict] = [feature]

    annotation_regions: list[dict] = []
    for cfg in annotation_configs:
        col = cfg["column"]
        target_val = str(cfg["value"])
        ann_df = run_df[run_df[col].astype(str) == target_val].copy()
        ann_df = ann_df.dropna(subset=["x_centroid", "y_centroid"])
        if len(ann_df) < 3:
            continue

        ann_coords = ann_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
        ann_buffer = _compute_buffer_distance(ann_coords, use_median=bool(use_median), percentile=float(percentile))
        if not np.isfinite(ann_buffer) or ann_buffer <= 0:
            ann_buffer = float(buffer_distance)

        ann_geom = _build_region_geometry(
            coords=ann_coords,
            method=method,
            alpha=float(alpha),
            buffer_distance=float(ann_buffer),
            min_hole_size=float(min_hole_size),
        )
        ann_geom = ann_geom.intersection(final_polygon)
        if ann_geom.is_empty:
            continue

        annotation_regions.append({
            "name": f"{col}={target_val}",
            "column": col,
            "value": target_val,
            "geom": ann_geom,
            "color_hex": cfg["color_hex"],
            "color_rgb": cfg["color_rgb"],
            "n_cells": int(len(ann_df)),
            "coords": ann_coords,
            "buffer_distance": float(ann_buffer),
        })

    # Resolve overlaps between adjacent annotations by nearest-cell split,
    # while preserving overlap where cells are truly multi-positive in both annotations.
    overlap_features: list[dict] = []
    if len(annotation_regions) >= 2:
        for idx_a, idx_b in combinations(range(len(annotation_regions)), 2):
            a = annotation_regions[idx_a]
            b = annotation_regions[idx_b]

            inter = a["geom"].intersection(b["geom"])
            if inter.is_empty or float(inter.area) <= 0:
                continue

            # Preserve overlap only where cells are positive for both annotations.
            multi_df = run_df[
                (run_df[a["column"]].astype(str) == str(a["value"]))
                & (run_df[b["column"]].astype(str) == str(b["value"]))
            ].dropna(subset=["x_centroid", "y_centroid"])

            preserve_overlap = inter.buffer(0)
            if len(multi_df) > 0:
                multi_coords = multi_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
                multi_buffer = _compute_buffer_distance(
                    multi_coords,
                    use_median=bool(use_median),
                    percentile=float(percentile),
                )
                if not np.isfinite(multi_buffer) or multi_buffer <= 0:
                    multi_buffer = float(min(a["buffer_distance"], b["buffer_distance"]))
                multi_geom = unary_union([Point(float(x), float(y)).buffer(float(multi_buffer)) for x, y in multi_coords])
                preserve_overlap = inter.intersection(multi_geom)
            else:
                preserve_overlap = inter.difference(inter)

            split_target = inter.difference(preserve_overlap)
            if (not split_target.is_empty) and float(split_target.area) > 0:
                step_px = max(2.0, min(12.0, float(min(a["buffer_distance"], b["buffer_distance"])) / 4.0))
                split_a, split_b = _split_overlap_by_nearest(
                    overlap_geom=split_target,
                    coords_a=a["coords"],
                    coords_b=b["coords"],
                    step_px=step_px,
                )

                a["geom"] = a["geom"].difference(split_target).union(split_a)
                b["geom"] = b["geom"].difference(split_target).union(split_b)

            if (not preserve_overlap.is_empty) and float(preserve_overlap.area) > 0:
                blend = [
                    int(round((a["color_rgb"][0] + b["color_rgb"][0]) / 2)),
                    int(round((a["color_rgb"][1] + b["color_rgb"][1]) / 2)),
                    int(round((a["color_rgb"][2] + b["color_rgb"][2]) / 2)),
                ]
                overlap_features.append(
                    {
                        "a_name": a["name"],
                        "b_name": b["name"],
                        "geom": preserve_overlap,
                        "color_rgb": blend,
                    }
                )

    # Export final annotation geometries after overlap resolution.
    for ann in annotation_regions:
        geojson_features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "properties": {
                    "classification": {
                        "name": f"annotation_{ann['column']}_{ann['value']}",
                        "color": ann["color_rgb"],
                    },
                    "isLocked": False,
                    "measurements": [
                        {"name": "annotation_column", "value": ann["column"]},
                        {"name": "annotation_value", "value": ann["value"]},
                        {"name": "n_cells", "value": int(ann["n_cells"])},
                        {"name": "buffer_distance_px", "value": float(ann["buffer_distance"])},
                    ],
                },
                "geometry": mapping(ann["geom"]),
            }
        )

    overlap_count = 0
    for ov in overlap_features:
        overlap_count += 1
        geojson_features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "properties": {
                    "classification": {"name": f"overlap_{ov['a_name']}__{ov['b_name']}", "color": ov["color_rgb"]},
                    "isLocked": False,
                    "measurements": [
                        {"name": "overlap_of", "value": f"{ov['a_name']} & {ov['b_name']}"},
                    ],
                },
                "geometry": mapping(ov["geom"]),
            }
        )

    geojson_data = {"type": "FeatureCollection", "features": geojson_features}
    _round_coords(geojson_data)

    with open(geojson_path, "w") as f:
        json.dump(geojson_data, f, separators=(",", ":"))

    summary_rows = [
        {
            "scope": "virtual_tissue",
            "name": "virtual_tissue",
            "batch_id": selected_batch,
            "tissue_roi": selected_tissue,
            "method": method,
            "buffer_mode": buffer_label,
            "buffer_distance_px": buffer_distance,
            "n_cells": len(run_df),
            "area_pixels2": area_pixels,
            "area_mm2": area_mm2,
            "density_cells_per_mm2": density_cells_per_mm2,
        }
    ]
    for ann in annotation_regions:
        ann_area_px = float(ann["geom"].area)
        ann_area_mm2 = ann_area_px * (pixel_size_mm ** 2)
        summary_rows.append(
            {
                "scope": "annotation",
                "name": ann["name"],
                "batch_id": selected_batch,
                "tissue_roi": selected_tissue,
                "method": method,
                "buffer_mode": buffer_label,
                "buffer_distance_px": buffer_distance,
                "n_cells": int(ann["n_cells"]),
                "area_pixels2": ann_area_px,
                "area_mm2": ann_area_mm2,
                "density_cells_per_mm2": (int(ann["n_cells"]) / ann_area_mm2) if ann_area_mm2 > 0 else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax_h = axes[0]
    ax_h.hist(nearest, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax_h.axvline(median_dist, color="red", linestyle="--", linewidth=2, label=f"Median: {median_dist:.1f} px")
    ax_h.axvline(p95, color="orange", linestyle="--", linewidth=2, label=f"95th: {p95:.1f} px")
    ax_h.axvline(p99, color="purple", linestyle="--", linewidth=2, label=f"99th: {p99:.1f} px")
    ax_h.set_xlabel("Nearest Neighbor Distance (pixels)")
    ax_h.set_ylabel("Frequency")
    ax_h.set_title("Nearest Neighbor Distance Distribution")
    ax_h.legend()
    ax_h.grid(True, alpha=0.3)

    ax_t = axes[1]
    ax_t.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.3, c="gray", label="Cells")
    if final_polygon.geom_type == "Polygon":
        exterior = getattr(final_polygon, "exterior", None)
        interiors = list(getattr(final_polygon, "interiors", []))
        if exterior is None:
            st.error("Generated polygon has no exterior boundary.")
            return

        x, y = exterior.xy
        ax_t.fill(x, y, color="green", alpha=0.3, label="Virtual Tissue")
        ax_t.plot(x, y, "g-", linewidth=2)
        for hole in interiors:
            hx, hy = zip(*hole.coords)
            ax_t.fill(hx, hy, color="white", alpha=1.0)
            ax_t.plot(hx, hy, "r-", linewidth=1)
    elif final_polygon.geom_type == "MultiPolygon":
        total_holes = 0
        for poly in list(getattr(final_polygon, "geoms", [])):
            x, y = poly.exterior.xy
            ax_t.fill(x, y, color="green", alpha=0.3)
            ax_t.plot(x, y, "g-", linewidth=2)
            interiors = list(getattr(poly, "interiors", []))
            total_holes += len(interiors)
            for hole in interiors:
                hx, hy = zip(*hole.coords)
                ax_t.fill(hx, hy, color="white", alpha=1.0)
                ax_t.plot(hx, hy, "r-", linewidth=1.5)
        if total_holes > 0:
            ax_t.plot([], [], "r-", linewidth=1.5, label=f"Holes (n={total_holes})")

    for ann in annotation_regions:
        geom = ann["geom"]
        color_hex = ann["color_hex"]
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax_t.fill(x, y, color=color_hex, alpha=0.35, label=ann["name"])
            ax_t.plot(x, y, color=color_hex, linewidth=1.2)
        elif geom.geom_type == "MultiPolygon":
            first = True
            for poly in list(getattr(geom, "geoms", [])):
                x, y = poly.exterior.xy
                ax_t.fill(x, y, color=color_hex, alpha=0.35, label=ann["name"] if first else None)
                ax_t.plot(x, y, color=color_hex, linewidth=1.2)
                first = False

    ax_t.set_aspect("equal")
    ax_t.invert_yaxis()
    ax_t.set_xlabel("X (pixels)")
    ax_t.set_ylabel("Y (pixels)")
    ax_t.set_title(f"Area: {area_mm2:.3f} mm² | Density: {density_cells_per_mm2:.1f} cells/mm²")
    ax_t.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    st.pyplot(fig)
    plt.close(fig)

    heatmap_saved = False
    if selected_heatmap_markers:
        valid_markers: list[str] = []
        for marker in selected_heatmap_markers:
            if marker in run_df.columns:
                ser = pd.to_numeric(run_df[marker], errors="coerce")
                if ser.notna().sum() >= 3:
                    valid_markers.append(marker)

        if valid_markers:
            n = len(valid_markers)
            ncols = 2 if n > 1 else 1
            nrows = int(math.ceil(n / ncols))
            fig_hm, axes_hm = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 6.5 * nrows))
            axes_arr = np.array(axes_hm).reshape(-1)

            for i, marker in enumerate(valid_markers):
                ax = axes_arr[i]
                vals_raw = pd.to_numeric(run_df[marker], errors="coerce").to_numpy(dtype=float)
                vals = _normalize_marker(vals_raw, heatmap_norm)
                mask = np.isfinite(vals)
                if mask.sum() == 0:
                    ax.set_title(f"{marker} (no finite values)")
                    ax.axis("off")
                    continue

                pts = coords[mask]
                v = vals[mask]

                if heatmap_show_context:
                    if final_polygon.geom_type == "Polygon":
                        exterior = getattr(final_polygon, "exterior", None)
                        if exterior is not None:
                            x, y = exterior.xy
                            ax.fill(x, y, color="#2ca02c", alpha=0.12, zorder=1)
                    elif final_polygon.geom_type == "MultiPolygon":
                        for poly in list(getattr(final_polygon, "geoms", [])):
                            x, y = poly.exterior.xy
                            ax.fill(x, y, color="#2ca02c", alpha=0.12, zorder=1)

                    for ann in annotation_regions:
                        geom = ann["geom"]
                        color_hex = ann["color_hex"]
                        if geom.geom_type == "Polygon":
                            x, y = geom.exterior.xy
                            ax.fill(x, y, color=color_hex, alpha=0.18, zorder=2)
                        elif geom.geom_type == "MultiPolygon":
                            for poly in list(getattr(geom, "geoms", [])):
                                x, y = poly.exterior.xy
                                ax.fill(x, y, color=color_hex, alpha=0.18, zorder=2)

                if heatmap_style == "fuzzy_cloud":
                    cloud_field, cloud_extent = _build_fuzzy_cloud(
                        coords=pts,
                        values=v,
                        buffer_distance_px=float(buffer_distance),
                        bins=int(cloud_bins),
                        sigma_scale=float(cloud_sigma_scale),
                        density_weight=bool(cloud_density_weight),
                    )
                    if heatmap_clip_to_tissue:
                        cloud_field = _mask_cloud_to_geometry(cloud_field, cloud_extent, final_polygon)
                    try:
                        cmap_obj = plt.get_cmap(heatmap_cmap).copy()
                    except Exception:
                        cmap_obj = plt.get_cmap(heatmap_cmap)
                    try:
                        cmap_obj.set_bad((0, 0, 0, 0))
                    except Exception:
                        pass
                    sc = ax.imshow(
                        cloud_field.T,
                        extent=cloud_extent,
                        origin="lower",
                        cmap=cmap_obj,
                        alpha=float(heatmap_alpha),
                        aspect="auto",
                        zorder=3,
                    )
                else:
                    sc = ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        c=v,
                        s=float(heatmap_pt_size),
                        cmap=heatmap_cmap,
                        alpha=float(heatmap_alpha),
                        edgecolors="none",
                        zorder=3,
                    )

                if final_polygon.geom_type == "Polygon":
                    exterior = getattr(final_polygon, "exterior", None)
                    if exterior is not None:
                        x, y = exterior.xy
                        ax.plot(x, y, "k-", linewidth=1.2, zorder=5)
                elif final_polygon.geom_type == "MultiPolygon":
                    for poly in list(getattr(final_polygon, "geoms", [])):
                        x, y = poly.exterior.xy
                        ax.plot(x, y, "k-", linewidth=1.0, zorder=5)

                if heatmap_show_context:
                    for ann in annotation_regions:
                        geom = ann["geom"]
                        color_hex = ann["color_hex"]
                        if geom.geom_type == "Polygon":
                            x, y = geom.exterior.xy
                            ax.plot(x, y, color=color_hex, linewidth=1.0, alpha=0.95, zorder=6)
                        elif geom.geom_type == "MultiPolygon":
                            for poly in list(getattr(geom, "geoms", [])):
                                x, y = poly.exterior.xy
                                ax.plot(x, y, color=color_hex, linewidth=1.0, alpha=0.95, zorder=6)

                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_xlabel("X (pixels)")
                ax.set_ylabel("Y (pixels)")
                ax.set_title(f"{marker} ({heatmap_norm})")
                ax.grid(True, alpha=0.2)
                cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("Expression")

            for j in range(len(valid_markers), len(axes_arr)):
                axes_arr[j].axis("off")

            plt.tight_layout()
            fig_hm.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            st.markdown("##### Marker Heatmap Overlay")
            st.pyplot(fig_hm)
            plt.close(fig_hm)
            heatmap_saved = True

    composite_saved = False
    if (layer_show_tissue or layer_show_annotations or layer_show_cont or layer_show_cat):
        fig_comp, ax_comp = plt.subplots(1, 1, figsize=(10, 8))

        if layer_show_tissue:
            if final_polygon.geom_type == "Polygon":
                exterior = getattr(final_polygon, "exterior", None)
                if exterior is not None:
                    x, y = exterior.xy
                    ax_comp.fill(x, y, color="#2ca02c", alpha=0.12, zorder=float(z_tissue))
                    ax_comp.plot(x, y, "k-", linewidth=1.4, zorder=float(z_tissue) + 0.01, label="virtual_tissue")
            elif final_polygon.geom_type == "MultiPolygon":
                first = True
                for poly in list(getattr(final_polygon, "geoms", [])):
                    x, y = poly.exterior.xy
                    ax_comp.fill(x, y, color="#2ca02c", alpha=0.12, zorder=float(z_tissue))
                    ax_comp.plot(
                        x,
                        y,
                        "k-",
                        linewidth=1.2,
                        zorder=float(z_tissue) + 0.01,
                        label="virtual_tissue" if first else None,
                    )
                    first = False

        if layer_show_annotations:
            for ann in annotation_regions:
                geom = ann["geom"]
                color_hex = ann["color_hex"]
                if geom.geom_type == "Polygon":
                    x, y = geom.exterior.xy
                    ax_comp.fill(x, y, color=color_hex, alpha=0.20, zorder=float(z_annotations), label=ann["name"])
                    ax_comp.plot(x, y, color=color_hex, linewidth=1.1, zorder=float(z_annotations) + 0.01)
                elif geom.geom_type == "MultiPolygon":
                    first = True
                    for poly in list(getattr(geom, "geoms", [])):
                        x, y = poly.exterior.xy
                        ax_comp.fill(
                            x,
                            y,
                            color=color_hex,
                            alpha=0.20,
                            zorder=float(z_annotations),
                            label=ann["name"] if first else None,
                        )
                        ax_comp.plot(x, y, color=color_hex, linewidth=1.1, zorder=float(z_annotations) + 0.01)
                        first = False

        if layer_show_cont and cont_marker_for_composite != "None" and cont_marker_for_composite in run_df.columns:
            cont_vals_raw = pd.to_numeric(run_df[cont_marker_for_composite], errors="coerce").to_numpy(dtype=float)
            cont_vals = _normalize_marker(cont_vals_raw, heatmap_norm)
            cont_mask = np.isfinite(cont_vals)
            if cont_mask.sum() > 0:
                cont_pts = coords[cont_mask]
                cont_v = cont_vals[cont_mask]
                if heatmap_style == "fuzzy_cloud":
                    cont_field, cont_extent = _build_fuzzy_cloud(
                        coords=cont_pts,
                        values=cont_v,
                        buffer_distance_px=float(buffer_distance),
                        bins=int(cloud_bins),
                        sigma_scale=float(cloud_sigma_scale),
                        density_weight=bool(cloud_density_weight),
                    )
                    if heatmap_clip_to_tissue:
                        cont_field = _mask_cloud_to_geometry(cont_field, cont_extent, final_polygon)
                    cmap_obj = plt.get_cmap(heatmap_cmap).copy()
                    cmap_obj.set_bad((0, 0, 0, 0))
                    comp_artist = ax_comp.imshow(
                        cont_field.T,
                        extent=(
                            float(cont_extent[0]),
                            float(cont_extent[1]),
                            float(cont_extent[2]),
                            float(cont_extent[3]),
                        ),
                        origin="lower",
                        cmap=cmap_obj,
                        alpha=float(heatmap_alpha),
                        aspect="auto",
                        zorder=float(z_cont),
                    )
                    cb_cont = plt.colorbar(comp_artist, ax=ax_comp, fraction=0.046, pad=0.04)
                    cb_cont.set_label(f"{cont_marker_for_composite} ({heatmap_norm})")
                else:
                    comp_artist = ax_comp.scatter(
                        cont_pts[:, 0],
                        cont_pts[:, 1],
                        c=cont_v,
                        s=float(heatmap_pt_size),
                        cmap=heatmap_cmap,
                        alpha=float(heatmap_alpha),
                        edgecolors="none",
                        zorder=float(z_cont),
                    )
                    cb_cont = plt.colorbar(comp_artist, ax=ax_comp, fraction=0.046, pad=0.04)
                    cb_cont.set_label(f"{cont_marker_for_composite} ({heatmap_norm})")

        if layer_show_cat and selected_cat_col != "None" and selected_cat_values:
            cat_rendered = 0
            for idx, cat_val in enumerate(selected_cat_values):
                cat_mask = run_df[selected_cat_col].astype(str).to_numpy() == str(cat_val)
                if cat_mask.sum() == 0:
                    continue
                cat_rendered += 1
                cat_pts = coords[cat_mask]
                color_hex = cat_value_colors.get(cat_val, "#00bcd4")
                if cat_style == "fuzzy_cloud":
                    cat_values = np.ones(cat_pts.shape[0], dtype=float)
                    cat_field, cat_extent = _build_fuzzy_cloud(
                        coords=cat_pts,
                        values=cat_values,
                        buffer_distance_px=float(buffer_distance),
                        bins=int(cloud_bins),
                        sigma_scale=float(cloud_sigma_scale),
                        density_weight=True,
                    )
                    if heatmap_clip_to_tissue:
                        cat_field = _mask_cloud_to_geometry(cat_field, cat_extent, final_polygon)
                    cat_cmap = _transparent_to_color_cmap(color_hex, name=f"cat_{selected_cat_col}_{idx}")
                    ax_comp.imshow(
                        cat_field.T,
                        extent=(
                            float(cat_extent[0]),
                            float(cat_extent[1]),
                            float(cat_extent[2]),
                            float(cat_extent[3]),
                        ),
                        origin="lower",
                        cmap=cat_cmap,
                        alpha=float(cat_alpha),
                        aspect="auto",
                        zorder=float(z_cat) + (idx * 0.01),
                    )
                else:
                    ax_comp.scatter(
                        cat_pts[:, 0],
                        cat_pts[:, 1],
                        s=float(cat_point_size),
                        c=color_hex,
                        alpha=float(cat_alpha),
                        edgecolors="none",
                        zorder=float(z_cat) + (idx * 0.01),
                        label=f"{selected_cat_col}={cat_val}",
                    )
            if cat_rendered == 0:
                st.warning(
                    "Categorical overlay selected, but none of the selected category values are present after current filters."
                )

        if final_polygon.geom_type == "Polygon":
            exterior = getattr(final_polygon, "exterior", None)
            if exterior is not None:
                x, y = exterior.xy
                ax_comp.plot(x, y, "k-", linewidth=1.5, zorder=max(float(z_tissue), float(z_annotations), float(z_cont), float(z_cat)) + 1)
        elif final_polygon.geom_type == "MultiPolygon":
            for poly in list(getattr(final_polygon, "geoms", [])):
                x, y = poly.exterior.xy
                ax_comp.plot(x, y, "k-", linewidth=1.2, zorder=max(float(z_tissue), float(z_annotations), float(z_cont), float(z_cat)) + 1)

        ax_comp.set_aspect("equal")
        ax_comp.invert_yaxis()
        ax_comp.set_xlabel("X (pixels)")
        ax_comp.set_ylabel("Y (pixels)")
        ax_comp.set_title("Composite Overlay: Tissue + Annotations + Continuous/Categorical Density")
        ax_comp.grid(True, alpha=0.2)

        handles, labels = ax_comp.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            ax_comp.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

        plt.tight_layout()
        fig_comp.savefig(composite_path, dpi=300, bbox_inches="tight")
        st.markdown("##### Composite Overlay Figure")
        st.pyplot(fig_comp)
        plt.close(fig_comp)
        composite_saved = True

    st.success("Virtual tissue generation complete.")
    st.write(
        {
            "GeoJSON": str(geojson_path),
            "Figure": str(fig_path),
            "Marker_heatmaps": str(heatmap_path) if heatmap_saved else None,
            "Composite_overlay": str(composite_path) if composite_saved else None,
            "Summary": str(summary_path),
            "Cells": int(len(run_df)),
            "Area_mm2": float(area_mm2),
            "Density_cells_mm2": float(density_cells_per_mm2) if np.isfinite(density_cells_per_mm2) else None,
            "Buffer_px": float(buffer_distance),
            "Buffer_mode": buffer_label,
            "Holes_before": int(holes_before),
            "Holes_after": int(holes_after),
            "Small_holes_removed": int(max(0, holes_before - holes_after)),
            "Annotation_regions": int(len(annotation_regions)),
            "Overlap_regions": int(overlap_count),
            "Heatmap_markers": selected_heatmap_markers,
            "Heatmap_norm": heatmap_norm if selected_heatmap_markers else None,
            "Heatmap_style": heatmap_style if selected_heatmap_markers else None,
            "Categorical_overlay_column": selected_cat_col if selected_cat_col != "None" else None,
            "Categorical_overlay_values": selected_cat_values if selected_cat_col != "None" else [],
        }
    )
