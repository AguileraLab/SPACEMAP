"""UMAP explorer subtab."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from spacemap_gui.ui_utils import clamp_numeric_widget_state
from .common import get_loaded_visualization_df, logger, sanitize_filename
from .phenotype_colors import render_phenotype_color_controls

_CATEGORICAL_PALETTES = [
    "tab10",
    "tab20",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "Dark2",
    "Pastel1",
    "Pastel2",
    "Accent",
    "Bold",
]

_CONTINUOUS_CMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Reds",
    "Blues",
    "Greens",
    "Purples",
    "Oranges",
    "YlOrRd",
    "YlGnBu",
    "RdYlBu",
    "coolwarm",
    "bwr",
    "Spectral",
    "RdBu_r",
    "PiYG",
]


def _classify_column(series: pd.Series, max_categories: int = 50) -> str:
    if series.dtype == object or series.dtype.name == "category":
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    n_unique = series.nunique(dropna=True)
    if n_unique <= max_categories and pd.api.types.is_integer_dtype(series):
        return "categorical"
    return "continuous"


@st.cache_data(show_spinner="Computing UMAP embedding...")
def _compute_umap_embedding(
    marker_values_bytes: bytes,
    n_cells: int,
    n_markers: int,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    n_pcs: int,
    metric: str,
    random_seed: int,
    subsample_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    import scanpy as sc

    logger.info("=" * 80)
    logger.info("UMAP EXPLORER - Compute Embedding")
    logger.info("=" * 80)
    logger.info(f"Input matrix: {n_cells:,} cells x {n_markers} markers")
    logger.info(
        f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}, n_pcs={n_pcs}, seed={random_seed}"
    )
    logger.info(f"Subsample: {subsample_size:,}" if subsample_size > 0 else "Subsample: OFF (all cells)")

    start = time.time()
    X = np.frombuffer(marker_values_bytes, dtype=np.float32).reshape(n_cells, n_markers)
    need_subsample = subsample_size > 0 and n_cells > subsample_size
    if need_subsample:
        np.random.seed(random_seed)
        sub_idx = np.random.choice(n_cells, subsample_size, replace=False)
        sub_idx.sort()
        X_sub = X[sub_idx]
        logger.info(f"Step 1/5: Subsampled {subsample_size:,} / {n_cells:,} cells ({subsample_size / n_cells * 100:.1f}%)")
    else:
        sub_idx = None
        X_sub = X
        logger.info(f"Step 1/5: No subsampling - using all {n_cells:,} cells")

    adata = sc.AnnData(X=X_sub.copy())
    adata.obs_names_make_unique()
    actual_pcs = min(n_pcs, min(adata.shape) - 1)
    logger.info(f"Step 2/5: Running PCA with {actual_pcs} components on {X_sub.shape[0]:,} cells...")
    pca_start = time.time()
    sc.tl.pca(adata, n_comps=actual_pcs, svd_solver="randomized", random_state=random_seed)
    logger.info(f"         PCA completed in {time.time() - pca_start:.2f}s")

    logger.info(f"Step 3/5: Building neighbor graph (n_neighbors={n_neighbors}, metric={metric})...")
    neigh_start = time.time()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_pcs, random_state=random_seed)
    logger.info(f"         Neighbors completed in {time.time() - neigh_start:.2f}s")

    logger.info(f"Step 4/5: Computing UMAP embedding (min_dist={min_dist})...")
    umap_start = time.time()
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_seed, n_components=n_components)
    logger.info(f"         UMAP completed in {time.time() - umap_start:.2f}s")
    umap_coords_sub = adata.obsm["X_umap"]

    if need_subsample:
        from sklearn.neighbors import KNeighborsRegressor

        logger.info(f"Step 5/5: Projecting remaining {n_cells - subsample_size:,} cells via k-NN in PCA space...")
        proj_start = time.time()
        X_centered = X - np.mean(X_sub, axis=0)
        pca_full = X_centered @ adata.varm["PCs"]
        pca_train = pca_full[sub_idx]
        knn = KNeighborsRegressor(n_neighbors=min(15, len(pca_train)), weights="distance")
        knn.fit(pca_train, umap_coords_sub)
        all_umap = knn.predict(pca_full)
        all_umap[sub_idx] = umap_coords_sub
        logger.info(f"         k-NN projection completed in {time.time() - proj_start:.2f}s")
        logger.info(f"COMPLETED: UMAP embedding for {n_cells:,} cells in {time.time() - start:.2f}s total")
        logger.info(
            f"  Coordinate range - X: [{all_umap[:, 0].min():.2f}, {all_umap[:, 0].max():.2f}], Y: [{all_umap[:, 1].min():.2f}, {all_umap[:, 1].max():.2f}]"
        )
        logger.info("=" * 80)
        return all_umap.astype(np.float32), sub_idx

    logger.info("Step 5/5: No projection needed (no subsampling)")
    umap_sub = np.asarray(umap_coords_sub)
    logger.info(f"COMPLETED: UMAP embedding for {n_cells:,} cells in {time.time() - start:.2f}s total")
    logger.info(
        f"  Coordinate range - X: [{umap_sub[:, 0].min():.2f}, {umap_sub[:, 0].max():.2f}], Y: [{umap_sub[:, 1].min():.2f}, {umap_sub[:, 1].max():.2f}]"
    )
    logger.info("=" * 80)
    return umap_sub.astype(np.float32), None


def _make_umap_figure_categorical(
    umap_x: np.ndarray,
    umap_y: np.ndarray,
    labels: np.ndarray,
    palette_name: str,
    custom_color_map: dict[str, Any] | None,
    point_size: float,
    alpha: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
    title: str,
    show_legend: bool,
    legend_fontsize: int,
    legend_ncol: int,
    legend_markerscale: float,
    label_fontsize: int,
    title_fontsize: int,
    axis_off: bool,
    rasterized: bool,
    background_color: str,
    grid_alpha: float,
    order: list[str] | None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    unique_labels = sorted(pd.Series(labels).dropna().unique(), key=str)
    if order:
        ordered = [label for label in order if label in unique_labels]
        remainder = [label for label in unique_labels if label not in ordered]
        unique_labels = ordered + remainder

    if custom_color_map:
        color_map = custom_color_map
    else:
        try:
            palette = sns.color_palette(palette_name, n_colors=max(len(unique_labels), 1))
        except Exception:
            palette = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))
        color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask = labels == label
        if mask.sum() == 0:
            continue
        color = color_map.get(label, "#999999")
        ax.scatter(
            umap_x[mask],
            umap_y[mask],
            c=[color],
            s=point_size,
            alpha=alpha,
            label=f"{label} ({mask.sum():,})",
            edgecolors="none",
            rasterized=rasterized,
        )

    ax.set_xlabel("UMAP 1", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)
    if axis_off:
        ax.axis("off")
    else:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=label_fontsize - 2)

    if show_legend and len(unique_labels) <= 60:
        legend = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=legend_fontsize,
            ncol=legend_ncol,
            markerscale=legend_markerscale,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )
        legend.get_frame().set_alpha(0.9)

    fig.tight_layout()
    return fig


def _make_umap_figure_continuous(
    umap_x: np.ndarray,
    umap_y: np.ndarray,
    values: np.ndarray,
    cmap_name: str,
    point_size: float,
    alpha: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
    title: str,
    colorbar_label: str,
    vmin: float | None,
    vmax: float | None,
    label_fontsize: int,
    title_fontsize: int,
    axis_off: bool,
    rasterized: bool,
    background_color: str,
    grid_alpha: float,
) -> Figure:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    scatter = ax.scatter(
        umap_x,
        umap_y,
        c=values,
        cmap=cmap_name,
        s=point_size,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        rasterized=rasterized,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=label_fontsize, fontweight="bold")
    cbar.ax.tick_params(labelsize=label_fontsize - 2)
    ax.set_xlabel("UMAP 1", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)
    if axis_off:
        ax.axis("off")
    else:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=label_fontsize - 2)
    fig.tight_layout()
    return fig


def render_umap_explorer(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
) -> None:
    st.markdown("### UMAP Explorer")
    st.caption(
        "Compute or load UMAP embeddings and create publication-ready scatter plots colored by any categorical or continuous column in the unified CSV."
    )

    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.error("Unified CSV file not found. Please set the global input path first.")
        return

    loaded_df = get_loaded_visualization_df(unified_csv_path)
    if not isinstance(loaded_df, pd.DataFrame):
        st.info("Load the unified CSV once using 'Unified CSV Loader' above to activate UMAP Explorer.")
        return

    logger.info("UMAP Explorer: Loading unified CSV: %s", os.path.basename(unified_csv_path))
    try:
        df_full = loaded_df
    except Exception as exc:
        st.error(f"Failed to load unified CSV: {exc}")
        logger.error("UMAP Explorer: Failed to load CSV - %s", exc)
        return

    n_total_cells = len(df_full)
    logger.info("UMAP Explorer: Loaded %s cells, %s columns", f"{n_total_cells:,}", len(df_full.columns))

    st.markdown("---")
    st.markdown("#### 0. Data Filtering")
    st.caption("Apply filters to select a subset of cells before UMAP computation or visualization. Only the filtered cells will be processed.")

    max_filter_cats = 50
    filterable_cols: list[str] = []
    for filter_col in df_full.columns:
        if df_full[filter_col].dtype == object or df_full[filter_col].dtype.name == "category":
            if df_full[filter_col].nunique(dropna=True) <= max_filter_cats:
                filterable_cols.append(filter_col)
        elif pd.api.types.is_bool_dtype(df_full[filter_col]):
            filterable_cols.append(filter_col)
        elif pd.api.types.is_numeric_dtype(df_full[filter_col]) and df_full[filter_col].nunique(dropna=True) <= max_filter_cats:
            filterable_cols.append(filter_col)

    if "umap_n_filters" not in st.session_state:
        st.session_state["umap_n_filters"] = 1
    n_filters = st.session_state["umap_n_filters"]
    filter_specs: list[tuple[str, list[str]]] = []

    for idx in range(n_filters):
        filter_label = f"Filter {idx + 1}" if n_filters > 1 else "Filter"
        col1, col2 = st.columns([1, 2])
        with col1:
            default_idx = 0
            cached_col = cached_params.get(f"umap_filter_col_{idx}")
            if cached_col and cached_col in filterable_cols:
                default_idx = filterable_cols.index(cached_col)
            elif idx == 0 and "tissue" in filterable_cols:
                default_idx = filterable_cols.index("tissue")
            selected_col = st.selectbox(
                f"{filter_label} - Column",
                options=filterable_cols if filterable_cols else ["(no filterable columns)"],
                index=min(default_idx, max(len(filterable_cols) - 1, 0)),
                key=f"umap_filt_col_{idx}",
            )
        with col2:
            if selected_col and selected_col in df_full.columns:
                unique_values = sorted(df_full[selected_col].dropna().unique(), key=str)
                unique_str = [str(value) for value in unique_values]
                cached_vals = cached_params.get(f"umap_filter_vals_{idx}", [])
                if not cached_vals and idx == 0 and selected_col == "tissue":
                    if "1" in unique_str:
                        cached_vals = ["1"]
                    elif 1 in unique_values:
                        cached_vals = ["1"]
                default_sel = [value for value in cached_vals if value in unique_str]
                selected_vals = st.multiselect(
                    f"{filter_label} - Values (keep)",
                    options=unique_str,
                    default=default_sel,
                    key=f"umap_filt_vals_{idx}",
                )
                if selected_vals:
                    filter_specs.append((selected_col, selected_vals))

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("+ Add Filter", key="umap_add_filter"):
            st.session_state["umap_n_filters"] = n_filters + 1
            st.rerun()
    with btn_col2:
        if n_filters > 1 and st.button("- Remove Filter", key="umap_rm_filter"):
            st.session_state["umap_n_filters"] = max(1, n_filters - 1)
            st.rerun()

    df = df_full.copy()
    filter_summary: list[str] = []
    for filter_col, filter_vals in filter_specs:
        before = len(df)
        df = df[df[filter_col].astype(str).isin(filter_vals)]
        after = len(df)
        filter_summary.append(f"`{filter_col}` in {filter_vals} ({before:,} -> {after:,})")

    original_indices = df.index.values.copy()
    n_filtered_cells = len(df)
    if filter_specs:
        st.success(f"**{n_filtered_cells:,}** / {n_total_cells:,} cells after filtering.")
        for summary in filter_summary:
            st.caption(f"  {summary}")
        logger.info("UMAP Explorer: Filters applied - %s / %s cells retained", f"{n_filtered_cells:,}", f"{n_total_cells:,}")
        for summary in filter_summary:
            logger.info("  Filter: %s", summary.replace("`", ""))
        for idx in range(n_filters):
            cached_params[f"umap_filter_col_{idx}"] = st.session_state.get(f"umap_filt_col_{idx}", "")
            cached_params[f"umap_filter_vals_{idx}"] = st.session_state.get(f"umap_filt_vals_{idx}", [])
    else:
        st.info(f"No filters applied. Using all **{n_total_cells:,}** cells.")
        logger.info("UMAP Explorer: No filters - using all %s cells", f"{n_total_cells:,}")

    if n_filtered_cells == 0:
        st.error("No cells remain after filtering. Adjust filter criteria.")
        return

    df = df.reset_index(drop=True)
    umap_x_col = None
    umap_y_col = None
    for x_col, y_col in [("umap_x", "umap_y"), ("UMAP_1", "UMAP_2"), ("umap_1", "umap_2"), ("UMAP1", "UMAP2")]:
        if x_col in df.columns and y_col in df.columns:
            umap_x_col, umap_y_col = x_col, y_col
            break
    has_existing_umap = umap_x_col is not None

    st.markdown("---")
    st.markdown("#### 1. UMAP Coordinates")
    coord_tab_load, coord_tab_compute = st.tabs(["Load Existing", "Compute New"])

    if "umap_explorer_x" not in st.session_state:
        st.session_state["umap_explorer_x"] = None
        st.session_state["umap_explorer_y"] = None
        st.session_state["umap_explorer_index"] = None
        st.session_state["umap_explorer_source"] = None
        st.session_state["umap_explorer_n_cells"] = None

    with coord_tab_load:
        if has_existing_umap:
            st.success(f"Found existing UMAP columns: **{umap_x_col}**, **{umap_y_col}**")
            valid_mask = df[umap_x_col].notna() & df[umap_y_col].notna()
            n_valid = valid_mask.sum()
            n_nan = (~valid_mask).sum()
            st.write(f"Valid coordinates: **{n_valid:,}** / {n_filtered_cells:,} cells (after filtering)")
            if n_nan > 0:
                st.warning(f"{n_nan:,} filtered cells have missing UMAP coordinates and will be excluded from plots.")
            if st.button("Load UMAP from CSV", key="umap_load_btn"):
                logger.info("UMAP Explorer: Loading existing UMAP coordinates from columns '%s', '%s'", umap_x_col, umap_y_col)
                logger.info("  Valid coordinates: %s / %s cells", f"{n_valid:,}", f"{n_filtered_cells:,}")
                st.session_state["umap_explorer_x"] = df.loc[valid_mask, umap_x_col].values.astype(np.float32)
                st.session_state["umap_explorer_y"] = df.loc[valid_mask, umap_y_col].values.astype(np.float32)
                st.session_state["umap_explorer_index"] = np.asarray(original_indices)[np.asarray(valid_mask.values)]
                st.session_state["umap_explorer_source"] = "loaded"
                st.session_state["umap_explorer_n_cells"] = n_filtered_cells
                st.success(f"Loaded {n_valid:,} UMAP coordinates.")
                st.rerun()
        else:
            st.info("No existing UMAP coordinates found in the filtered data. Use **Compute New** to generate them.")

    with coord_tab_compute:
        st.markdown("##### UMAP Computation Parameters")
        metadata_exclude = {
            "cell_id",
            "batch_id",
            "tissue",
            "x_centroid",
            "y_centroid",
            "area",
            "perimeter",
            "eccentricity",
            "solidity",
            "extent",
            "major_axis",
            "minor_axis",
            "orientation",
            "label",
            "index",
        }
        marker_candidates = [
            col
            for col in df.columns
            if col.lower() not in {meta.lower() for meta in metadata_exclude}
            and not col.startswith("umap_")
            and not col.startswith("UMAP")
            and not col.startswith("phenotype_")
            and not col.startswith("leiden_")
            and not col.startswith("som_")
            and not col.startswith("meta_cluster")
            and not col.endswith("_norm")
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        default_markers = [marker for marker in marker_candidates if marker.endswith("_corr")]
        if not default_markers:
            default_markers = marker_candidates[:20]

        with st.form("umap_compute_form", clear_on_submit=False):
            selected_markers = st.multiselect(
                "Markers for UMAP",
                options=marker_candidates,
                default=[marker for marker in cached_params.get("umap_markers", default_markers) if marker in marker_candidates],
                key="umap_compute_markers",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                n_neighbors = st.number_input("n_neighbors", min_value=2, max_value=200, value=int(cached_params.get("umap_n_neighbors", 15)), key="umap_n_neighbors")
                n_pcs = st.number_input("PCA components", min_value=2, max_value=100, value=int(cached_params.get("umap_n_pcs", 20)), key="umap_n_pcs")
            with col2:
                min_dist = st.number_input("min_dist", min_value=0.0, max_value=1.0, value=float(cached_params.get("umap_min_dist", 0.1)), step=0.05, format="%.2f", key="umap_min_dist")
                metric = st.selectbox(
                    "Distance metric",
                    ["euclidean", "cosine", "manhattan", "correlation"],
                    index=["euclidean", "cosine", "manhattan", "correlation"].index(cached_params.get("umap_metric", "euclidean")),
                    key="umap_metric",
                )
            with col3:
                # The bound follows the loaded CSV, so a remembered value from a
                # larger file has to be pulled back into range before the widget
                # is built, or Streamlit refuses to render it.
                subsample_default = clamp_numeric_widget_state(
                    "umap_subsample", 0, int(n_filtered_cells),
                    fallback=int(cached_params.get("umap_subsample", min(150000, n_filtered_cells))),
                )
                subsample = st.number_input(
                    "Subsample size (0=all)",
                    min_value=0,
                    max_value=n_filtered_cells,
                    value=int(subsample_default or 0),
                    step=10000,
                    key="umap_subsample",
                )
                random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=int(cached_params.get("umap_seed", 42)), key="umap_seed")

            apply_zscore = st.checkbox(
                "Z-score normalize markers before UMAP",
                value=cached_params.get("umap_zscore", True),
                key="umap_zscore_cb",
            )
            compute_btn = st.form_submit_button("Compute UMAP", type="primary")

        if compute_btn:
            if not selected_markers:
                st.error("Please select at least one marker for UMAP computation.")
            else:
                try:
                    logger.info("UMAP Explorer: Preparing data for computation")
                    logger.info("  Markers (%d): %s%s", len(selected_markers), ", ".join(selected_markers[:8]), f"... +{len(selected_markers) - 8} more" if len(selected_markers) > 8 else "")
                    X_raw = df[selected_markers].fillna(0).values.astype(np.float32)
                    logger.info("  Matrix shape: %s x %s  (%.1f MB)", f"{X_raw.shape[0]:,}", X_raw.shape[1], X_raw.nbytes / 1024 / 1024)
                    if apply_zscore:
                        from sklearn.preprocessing import StandardScaler

                        logger.info("  Applying Z-score normalization...")
                        X_raw = StandardScaler().fit_transform(X_raw).astype(np.float32)
                        logger.info("  Z-score normalization complete")
                    else:
                        logger.info("  Z-score normalization: OFF")

                    marker_bytes = X_raw.tobytes()
                    with st.spinner(f"Computing UMAP on {n_filtered_cells:,} filtered cells ({len(selected_markers)} markers)..."):
                        umap_coords, sub_idx = _compute_umap_embedding(
                            marker_values_bytes=marker_bytes,
                            n_cells=n_filtered_cells,
                            n_markers=len(selected_markers),
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=2,
                            n_pcs=n_pcs,
                            metric=metric,
                            random_seed=random_seed,
                            subsample_size=subsample,
                        )

                    st.session_state["umap_explorer_x"] = umap_coords[:, 0]
                    st.session_state["umap_explorer_y"] = umap_coords[:, 1]
                    st.session_state["umap_explorer_index"] = original_indices
                    st.session_state["umap_explorer_source"] = "computed"
                    st.session_state["umap_explorer_n_cells"] = n_filtered_cells
                    cached_params["umap_markers"] = selected_markers
                    cached_params["umap_n_neighbors"] = n_neighbors
                    cached_params["umap_n_pcs"] = n_pcs
                    cached_params["umap_min_dist"] = min_dist
                    cached_params["umap_metric"] = metric
                    cached_params["umap_subsample"] = subsample
                    cached_params["umap_seed"] = random_seed
                    cached_params["umap_zscore"] = apply_zscore
                    st.success(f"UMAP computed for **{n_filtered_cells:,}** filtered cells.")
                    if sub_idx is not None:
                        st.caption(f"Direct UMAP on {len(sub_idx):,} subsampled cells; remaining projected via k-NN.")
                    st.rerun()
                except ImportError:
                    st.error("scanpy is required for UMAP computation. Install with: `pip install scanpy`")
                except Exception as exc:
                    st.error(f"UMAP computation failed: {exc}")
                    logger.error(f"UMAP Explorer: Computation error - {exc}")

    ux = st.session_state.get("umap_explorer_x")
    uy = st.session_state.get("umap_explorer_y")
    u_idx = st.session_state.get("umap_explorer_index")
    u_source = st.session_state.get("umap_explorer_source")

    # Validate cached UMAP coordinates and indices to prevent out-of-bounds IndexError
    if ux is not None and u_idx is not None:
        is_invalid = False
        try:
            if len(ux) != len(u_idx):
                is_invalid = True
            elif np.any(u_idx >= len(df_full)) or np.any(u_idx < 0):
                is_invalid = True
        except Exception:
            is_invalid = True

        if is_invalid:
            logger.info("UMAP Explorer: Stale/out-of-bounds UMAP coordinates detected. Resetting cached coordinates.")
            st.session_state["umap_explorer_x"] = None
            st.session_state["umap_explorer_y"] = None
            st.session_state["umap_explorer_index"] = None
            st.session_state["umap_explorer_source"] = None
            st.session_state["umap_explorer_n_cells"] = None
            ux, uy, u_idx, u_source = None, None, None, None

    if ux is None or uy is None:
        st.markdown("---")
        st.info("Load or compute UMAP coordinates above to unlock plotting and export options.")
        return

    n_plot_cells = len(ux)
    st.markdown("---")
    st.markdown(f"#### 2. Visualization  &mdash;  {n_plot_cells:,} cells  ({u_source})")

    # Safe indexing guard
    if u_idx is not None:
        try:
            df_plot = df_full.iloc[u_idx].reset_index(drop=True).copy()
        except Exception as exc:
            logger.error(f"UMAP Explorer: Failed to index df_full with u_idx: {exc}")
            st.error(f"Failed to align UMAP coordinates: {exc}. Please reload or recompute UMAP.")
            df_plot = df.reset_index(drop=True).copy()
    else:
        df_plot = df.reset_index(drop=True).copy()

    all_cols = list(df_plot.columns)
    phenotype_cols = [col for col in all_cols if col.startswith("phenotype_")]
    cluster_cols = [col for col in all_cols if "cluster" in col.lower()]
    suggested_order = phenotype_cols + cluster_cols + [col for col in all_cols if col not in phenotype_cols + cluster_cols]
    color_col = st.selectbox("Color by column", options=suggested_order, index=0 if suggested_order else 0, key="umap_color_col")
    col_type = _classify_column(df_plot[color_col])

    excluded_categories: list[str] = []
    included_cats: list[str] = []
    categorical_color_map: dict[str, str] = {}
    if col_type == "categorical":
        all_cats = sorted(df_plot[color_col].dropna().astype(str).unique(), key=str)
        with st.expander("Include / Exclude categories", expanded=False):
            cached_inc = cached_params.get(f"umap_inc_cats_{color_col}")
            default_sel = cached_inc if (cached_inc is not None and set(cached_inc).issubset(set(all_cats))) else all_cats
            included_cats = st.multiselect("Categories to show", options=all_cats, default=default_sel, key=f"umap_inc_cats_{color_col}")
            excluded_categories = [category for category in all_cats if category not in included_cats]
            if excluded_categories:
                st.info(
                    f"Hiding {len(excluded_categories)} categor{'y' if len(excluded_categories) == 1 else 'ies'}: {', '.join(excluded_categories[:10])}{'...' if len(excluded_categories) > 10 else ''}"
                )
            cached_params[f"umap_inc_cats_{color_col}"] = included_cats
        categorical_color_map = render_phenotype_color_controls(
            all_cats,
            panel_key="tab9_umap",
            title="Category / phenotype colors",
        )

    st.markdown("##### Figure Settings")
    with st.form("umap_plot_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            display_default = clamp_numeric_widget_state(
                "umap_display_sub", 0, int(n_plot_cells),
                fallback=int(cached_params.get("umap_display_sub", 0)),
            )
            display_subsample = st.number_input("Cells to plot (0=all)", min_value=0, max_value=n_plot_cells, value=int(display_default or 0), step=50000, key="umap_display_sub")
            point_size = st.number_input("Point size", min_value=0.05, max_value=50.0, value=float(cached_params.get("umap_pt_size", 0.5)), step=0.1, format="%.2f", key="umap_pt_size")
            alpha_val = st.slider("Opacity", 0.01, 1.0, value=float(cached_params.get("umap_alpha", 0.7)), step=0.05, key="umap_alpha")
        with col2:
            fig_width = st.number_input("Figure width (in)", min_value=4.0, max_value=30.0, value=float(cached_params.get("umap_fig_w", 10.0)), step=0.5, key="umap_fig_w")
            fig_height = st.number_input("Figure height (in)", min_value=4.0, max_value=30.0, value=float(cached_params.get("umap_fig_h", 8.0)), step=0.5, key="umap_fig_h")
            fig_dpi = st.number_input("DPI", min_value=72, max_value=1200, value=int(cached_params.get("umap_dpi", 300)), step=50, key="umap_dpi")
        with col3:
            title_fontsize = st.number_input("Title font size", 8, 36, value=int(cached_params.get("umap_title_fs", 16)), key="umap_title_fs")
            label_fontsize = st.number_input("Label font size", 6, 28, value=int(cached_params.get("umap_label_fs", 12)), key="umap_label_fs")
            rasterize = st.checkbox("Rasterize points", value=cached_params.get("umap_raster", True), key="umap_raster")

        with st.expander("Advanced Styling", expanded=False):
            adv1, adv2 = st.columns(2)
            with adv1:
                bg_color = st.selectbox(
                    "Background",
                    ["white", "#f0f0f0", "black", "#1a1a2e", "none"],
                    index=["white", "#f0f0f0", "black", "#1a1a2e", "none"].index(cached_params.get("umap_bg", "white")),
                    key="umap_bg",
                )
                grid_alpha = st.slider("Grid opacity", 0.0, 1.0, value=float(cached_params.get("umap_grid_a", 0.3)), step=0.05, key="umap_grid_a")
                axis_off = st.checkbox("Hide axes", value=cached_params.get("umap_axis_off", False), key="umap_axis_off")
            with adv2:
                if col_type == "categorical":
                    show_legend = st.checkbox("Show legend", value=cached_params.get("umap_legend", True), key="umap_legend")
                    legend_fs = st.number_input("Legend font size", 4, 20, value=int(cached_params.get("umap_leg_fs", 8)), key="umap_leg_fs")
                    legend_ncol = st.number_input("Legend columns", 1, 5, value=int(cached_params.get("umap_leg_nc", 1)), key="umap_leg_nc")
                    legend_ms = st.number_input("Legend marker scale", 1.0, 15.0, value=float(cached_params.get("umap_leg_ms", 4.0)), step=0.5, key="umap_leg_ms")
                else:
                    show_legend = False
                    legend_fs = 8
                    legend_ncol = 1
                    legend_ms = 4.0

        palette_name = "tab10"
        cmap_name = "viridis"
        use_custom_vmin = False
        use_custom_vmax = False
        custom_vmin: float | None = None
        custom_vmax: float | None = None
        if col_type != "categorical":
            cmap_name = st.selectbox(
                "Colormap",
                _CONTINUOUS_CMAPS,
                index=_CONTINUOUS_CMAPS.index(cached_params.get("umap_cont_cmap", "viridis")) if cached_params.get("umap_cont_cmap", "viridis") in _CONTINUOUS_CMAPS else 0,
                key="umap_cont_cmap",
            )
            vmin_col1, vmin_col2 = st.columns(2)
            with vmin_col1:
                use_custom_vmin = st.checkbox("Custom vmin", value=False, key="umap_use_vmin")
                custom_vmin = st.number_input("vmin", value=0.0, key="umap_vmin") if use_custom_vmin else None
            with vmin_col2:
                use_custom_vmax = st.checkbox("Custom vmax", value=False, key="umap_use_vmax")
                custom_vmax = st.number_input("vmax", value=1.0, key="umap_vmax") if use_custom_vmax else None

        custom_title = st.text_input("Plot title (leave blank for auto)", value=cached_params.get("umap_title", ""), key="umap_title_input")
        plot_btn = st.form_submit_button("Generate UMAP Plot", type="primary")

    if plot_btn:
        if display_subsample > 0 and display_subsample < n_plot_cells:
            np.random.seed(42)
            vis_idx = np.random.choice(n_plot_cells, display_subsample, replace=False)
            vis_idx.sort()
        else:
            vis_idx = np.arange(n_plot_cells)

        ux_vis = ux[vis_idx]
        uy_vis = uy[vis_idx]
        color_data = df_plot[color_col].iloc[vis_idx]
        if col_type == "categorical":
            has_na = color_data.isna().any() or (color_data.astype(str).str.strip() == "").any()
            if excluded_categories or has_na:
                inc_set = set(included_cats) if included_cats else None
                if inc_set:
                    keep_mask = np.asarray(color_data.fillna("").astype(str).isin(inc_set))
                else:
                    keep_mask = np.asarray(~color_data.astype(str).isin(excluded_categories))
                vis_idx = vis_idx[keep_mask]
                ux_vis = ux_vis[keep_mask]
                uy_vis = uy_vis[keep_mask]
                color_data = color_data[keep_mask]

        auto_title = f"UMAP - {color_col}  (n={len(vis_idx):,})"
        title = str(custom_title).strip() if custom_title else auto_title
        if not title:
            title = auto_title

        try:
            if col_type == "categorical":
                labels = np.asarray(color_data.fillna("NA").astype(str).values)
                fig = _make_umap_figure_categorical(
                    umap_x=ux_vis,
                    umap_y=uy_vis,
                    labels=labels,
                    palette_name=palette_name,
                    custom_color_map=categorical_color_map,
                    point_size=point_size,
                    alpha=alpha_val,
                    fig_width=fig_width,
                    fig_height=fig_height,
                    dpi=fig_dpi,
                    title=title,
                    show_legend=show_legend,
                    legend_fontsize=legend_fs,
                    legend_ncol=legend_ncol,
                    legend_markerscale=legend_ms,
                    label_fontsize=label_fontsize,
                    title_fontsize=title_fontsize,
                    axis_off=axis_off,
                    rasterized=rasterize,
                    background_color=bg_color,
                    grid_alpha=grid_alpha,
                    order=None,
                )
            else:
                values = np.asarray(pd.to_numeric(color_data, errors="coerce").fillna(0).values)
                fig = _make_umap_figure_continuous(
                    umap_x=ux_vis,
                    umap_y=uy_vis,
                    values=values,
                    cmap_name=cmap_name,
                    point_size=point_size,
                    alpha=alpha_val,
                    fig_width=fig_width,
                    fig_height=fig_height,
                    dpi=fig_dpi,
                    title=title,
                    colorbar_label=color_col,
                    vmin=custom_vmin if use_custom_vmin else None,
                    vmax=custom_vmax if use_custom_vmax else None,
                    label_fontsize=label_fontsize,
                    title_fontsize=title_fontsize,
                    axis_off=axis_off,
                    rasterized=rasterize,
                    background_color=bg_color,
                    grid_alpha=grid_alpha,
                )

            st.pyplot(fig)
            st.markdown("##### Export Figure")
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=fig_dpi, bbox_inches="tight")
                buf_png.seek(0)
                st.download_button("Download PNG", buf_png, file_name="umap_plot.png", mime="image/png", key="umap_dl_png")
            with dl2:
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format="pdf", dpi=fig_dpi, bbox_inches="tight")
                buf_pdf.seek(0)
                st.download_button("Download PDF", buf_pdf, file_name="umap_plot.pdf", mime="application/pdf", key="umap_dl_pdf")
            with dl3:
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                buf_svg.seek(0)
                st.download_button("Download SVG", buf_svg, file_name="umap_plot.svg", mime="image/svg+xml", key="umap_dl_svg")

            output_dir = project_root / "outputs" / "phenotyping" / "umap_plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"umap_{sanitize_filename(color_col)}.png"
            fig.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")
            st.caption(f"Auto-saved: `{save_path}`")
            plt.close(fig)
        except Exception as exc:
            st.error(f"Plot generation failed: {exc}")
            logger.error(f"UMAP Explorer: Plot error - {exc}")

    st.markdown("---")
    st.markdown("#### 3. Save UMAP Coordinates to Unified CSV")
    st.caption("Write the current UMAP coordinates back into the unified CSV file so they are available for downstream analysis. Filtered cells get coordinates; unfiltered cells get NaN.")
    u_idx_len = len(u_idx) if u_idx is not None else 0
    is_partial = u_idx is not None and u_idx_len < n_total_cells
    if is_partial:
        st.info(f"Coordinates available for **{u_idx_len:,}** / {n_total_cells:,} cells. Unmatched rows will be written as NaN.")

    with st.form("umap_save_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            umap_x_save_col = st.text_input("UMAP X column name", value=cached_params.get("umap_save_x_name", "umap_x"), key="umap_save_x_name")
        with col2:
            umap_y_save_col = st.text_input("UMAP Y column name", value=cached_params.get("umap_save_y_name", "umap_y"), key="umap_save_y_name")
        overwrite_existing = st.checkbox("Overwrite existing columns if they already exist", value=True, key="umap_save_overwrite")
        save_btn = st.form_submit_button("Save Coordinates to CSV", type="primary")

    if save_btn:
        try:
            df_write = pd.read_csv(unified_csv_path, low_memory=False)
            if not overwrite_existing:
                conflict = [col for col in [umap_x_save_col, umap_y_save_col] if col in df_write.columns]
                if conflict:
                    st.error(f"Columns already exist and overwrite is disabled: {conflict}")
                    st.stop()
            df_write[umap_x_save_col] = np.nan
            df_write[umap_y_save_col] = np.nan
            if u_idx is not None:
                valid_idx = u_idx[u_idx < len(df_write)]
                df_write.loc[valid_idx, umap_x_save_col] = ux[: len(valid_idx)]
                df_write.loc[valid_idx, umap_y_save_col] = uy[: len(valid_idx)]
            else:
                df_write[umap_x_save_col] = ux
                df_write[umap_y_save_col] = uy
            df_write.to_csv(unified_csv_path, index=False)
            n_written = int((~df_write[umap_x_save_col].isna()).sum())
            st.success(
                f"Saved **{umap_x_save_col}** and **{umap_y_save_col}** to `{os.path.basename(unified_csv_path)}` ({n_written:,} coordinates in {len(df_write):,} rows)."
            )
            cached_params["umap_save_x_name"] = umap_x_save_col
            cached_params["umap_save_y_name"] = umap_y_save_col
            for key in list(st.session_state.keys()):
                if "tab9_viz_unified" in str(key):
                    del st.session_state[key]
        except Exception as exc:
            st.error(f"Failed to save coordinates: {exc}")
            logger.error(f"UMAP Explorer: Save error - {exc}")

    with st.expander("UMAP Coordinate Statistics", expanded=False):
        stat_df = pd.DataFrame({"UMAP_X": ux, "UMAP_Y": uy})
        st.write(stat_df.describe().round(4))
        if u_idx is not None:
            st.caption(f"Covers {u_idx_len:,} / {n_total_cells:,} total cells in the unified CSV.")
