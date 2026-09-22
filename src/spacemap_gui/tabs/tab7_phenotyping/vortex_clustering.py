"""
Vortex (X-shift) Clustering Subtab.

Professional Vortex clustering with notebook-synchronized parameters from
Step6E_phenotype_vortex.ipynb.
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.figure import Figure

from .common import (
    LOW_EXPRESSION_METRIC_OPTIONS,
    Timer,
    _extract_markers_from_notebook,
    build_cluster_marker_score_map_from_heatmap,
    render_assisted_assignment,
    build_marker_source_note,
    build_leiden_heatmap_figure,
    apply_heatmap_assignment_event,
    consume_heatmap_assignment_event,
    get_heatmap_assignment_scope,
    compute_low_expression_score,
    create_session_temp_parquet_path,
    ensure_cell_id_anchor,
    get_batch_id_column_help_text,
    get_cell_id_column_help_text,
    get_enable_row_filter_help_text,
    get_filter_column_help_text,
    get_filter_value_help_text,
    get_all_phenotype_options,
    initialize_optional_column_widget_state,
    get_low_expression_default_threshold,
    get_low_expression_filter_help_text,
    get_low_expression_metric_help_text,
    get_low_expression_metric_label,
    logger,
    get_marker_selection_help_text,
    get_marker_workspace_panel,
    render_method_marker_selection,
    get_normalization_flag_help_text,
    normalize_optional_column_name,
    get_output_column_help_text,
    get_random_seed_help_text,
    read_dataframe_parquet_safe,
    render_add_custom_phenotype_ui,
    render_export_split_data,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_settings_export_import,
    render_phenotyping_run_log_controls,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    normalize_phenotype_map_keys,
    resolve_notebook_marker_defaults,
    write_dataframe_parquet_atomic,
)
from .heatmap_click_selector import (
    heatmap_click_selector_available,
    render_heatmap_click_selector,
)

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

try:
    from spacemap import ph_vortex_utils
    sys.modules["ph_vortex_utils"] = ph_vortex_utils
except ImportError:
    try:
        # Check parents[3] / "spacemap" / "ph_vortex_utils.py" (current workspace package structure)
        _PH_VORTEX_UTILS_PATH = Path(__file__).resolve().parents[3] / "spacemap" / "ph_vortex_utils.py"
        if not _PH_VORTEX_UTILS_PATH.exists():
            # Check parents[4] / "src" / "spacemap" / "ph_vortex_utils.py"
            _PH_VORTEX_UTILS_PATH = Path(__file__).resolve().parents[4] / "src" / "spacemap" / "ph_vortex_utils.py"
        if not _PH_VORTEX_UTILS_PATH.exists():
            # Check legacy V13 path
            _PH_VORTEX_UTILS_PATH = Path(__file__).resolve().parents[3] / "src" / "ph_vortex_utils.py"

        if _PH_VORTEX_UTILS_PATH.exists():
            _ph_vortex_spec = importlib.util.spec_from_file_location("ph_vortex_utils", _PH_VORTEX_UTILS_PATH)
            if _ph_vortex_spec is not None and _ph_vortex_spec.loader is not None:
                ph_vortex_utils = importlib.util.module_from_spec(_ph_vortex_spec)
                sys.modules.setdefault("ph_vortex_utils", ph_vortex_utils)
                _ph_vortex_spec.loader.exec_module(ph_vortex_utils)
            else:
                ph_vortex_utils = None
        else:
            ph_vortex_utils = None
    except Exception:
        ph_vortex_utils = None


VORTEX_LOW_EXPR_CLUSTER_ID = -1
VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE = "Others"
VORTEX_LOW_EXPR_LABEL = "Low expression bucket"


def _dataio_load(path, columns=None):
    """Read a large table through the shared loader.

    Uses the Parquet fast copy when one is current, otherwise the CSV. Passing
    *columns* reads only those.
    """
    from spacemap import dataio

    return dataio.load(path, columns=columns)


def _read_csv_fast(path: str) -> pd.DataFrame:
    """Multithreaded CSV read shared with the other phenotyping tabs."""
    from spacemap.ph_accel_utils import read_csv_fast

    return read_csv_fast(path, logger=logger)


def _parse_int_list(text: str, fallback: list[int]) -> list[int]:
    values: list[int] = []
    for token in re.split(r"[\s,]+", str(text).strip()):
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            values.append(value)
    parsed = sorted(set(values))
    return parsed if parsed else fallback


def _to_python_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clean_marker_name(marker_name: str) -> str:
    return (
        str(marker_name)
        .replace("_corr_norm", "")
        .replace("_norm", "")
        .replace("_corr", "")
        .replace("_unmixed", "")
        .replace("_resolve", "")
    )


def _normalize_cluster_id(cluster_id):
    if pd.isna(cluster_id):
        return None
    if isinstance(cluster_id, np.generic):
        cluster_id = cluster_id.item()
    try:
        return int(cluster_id)
    except Exception:
        return cluster_id


def _cluster_sort_key(cluster_id):
    normalized = _normalize_cluster_id(cluster_id)
    if normalized is None:
        return (2, "")
    if isinstance(normalized, int):
        return (0, normalized)
    return (1, str(normalized))


def _cluster_matches(cluster_value, target_value) -> bool:
    left = _normalize_cluster_id(cluster_value)
    right = _normalize_cluster_id(target_value)
    if left is None or right is None:
        return False
    return left == right


def _is_low_expression_cluster(cluster_id, low_expr_cluster_id) -> bool:
    if low_expr_cluster_id is None:
        return False
    return _cluster_matches(cluster_id, low_expr_cluster_id)


def _cluster_display_label(cluster_id, low_expr_cluster_id=None) -> str:
    if _is_low_expression_cluster(cluster_id, low_expr_cluster_id):
        normalized = _normalize_cluster_id(cluster_id)
        return f"{VORTEX_LOW_EXPR_LABEL} ({normalized})"
    normalized = _normalize_cluster_id(cluster_id)
    return str(normalized) if normalized is not None else "Unknown"


def _format_low_expression_rule(metric: str | None, threshold: float | None) -> str:
    if threshold is None:
        return "screened outside clustering"
    metric_label = get_low_expression_metric_label(metric or "mean_zscore").lower()
    return f"{metric_label} <= {float(threshold):.4g}"


def _build_vortex_preview_df(
    df_results: pd.DataFrame,
    output_col: str,
    phenotype_map: dict,
) -> pd.DataFrame:
    preview_df = df_results.copy()
    if phenotype_map:
        preview_df[output_col] = preview_df["vortex_cluster"].map(phenotype_map).fillna("Unassigned")
    elif output_col not in preview_df.columns:
        preview_df[output_col] = "Unassigned"
    else:
        preview_df[output_col] = preview_df[output_col].fillna("Unassigned")
    return preview_df


def _build_cluster_marker_score_map(
    df_results: pd.DataFrame,
    norm_cols: list[str],
    cluster_col: str = "vortex_cluster",
) -> dict:
    valid_norm_cols = [
        column
        for column in norm_cols
        if column in df_results.columns and pd.api.types.is_numeric_dtype(df_results[column])
    ]
    if cluster_col not in df_results.columns or not valid_norm_cols:
        return {}

    grouped = df_results.groupby(cluster_col)[valid_norm_cols].mean()
    if grouped.empty:
        return {}

    grouped_t = grouped.T.copy()
    grouped_t.index = grouped_t.index.map(_clean_marker_name)
    collapsed = grouped_t.groupby(level=0).mean().T

    cluster_marker_scores = {}
    for cluster_id, row in collapsed.iterrows():
        normalized_cluster_id = _normalize_cluster_id(cluster_id)
        if normalized_cluster_id is None:
            continue
        marker_scores = {
            str(marker): float(score)
            for marker, score in row.items()
            if pd.notna(score)
        }
        cluster_marker_scores[normalized_cluster_id] = marker_scores
    return cluster_marker_scores


def _log_vortex_step(step_number: str, title: str, details: str | None = None) -> None:
    message = f"[Vortex] Step {step_number}: {title}"
    if details:
        message = f"{message} | {details}"
    logger.info(message)


def _build_umap_comparison_figure(
    plot_df: pd.DataFrame,
    output_col: str,
    n_clusters: int,
    point_size: float,
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cluster_scatter = axes[0].scatter(
        plot_df["umap_x"],
        plot_df["umap_y"],
        c=plot_df["vortex_cluster"].astype(int),
        cmap="tab20",
        s=point_size,
        alpha=0.6,
        rasterized=True,
    )
    fig.colorbar(cluster_scatter, ax=axes[0], label="Vortex Cluster", fraction=0.046, pad=0.04)
    axes[0].set_title(f"Before Assignment: Vortex Clusters (n={n_clusters})")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].grid(True, alpha=0.3)

    phenotype_series = plot_df[output_col].fillna("Unassigned").astype(str) if output_col in plot_df.columns else None
    if phenotype_series is not None:
        unique_phenotypes = phenotype_series.unique().tolist()
        if len(unique_phenotypes) == 1 and unique_phenotypes[0] == "Unassigned":
            axes[1].scatter(
                plot_df["umap_x"],
                plot_df["umap_y"],
                c="#9AA0A6",
                s=point_size,
                alpha=0.6,
                rasterized=True,
            )
            axes[1].set_title("After Assignment: Awaiting phenotype labels")
        else:
            colors = plt.get_cmap("Set3")(np.linspace(0, 1, max(1, len(unique_phenotypes))))
            phenotype_to_color = dict(zip(unique_phenotypes, colors))
            for phenotype in unique_phenotypes:
                mask = phenotype_series == phenotype
                axes[1].scatter(
                    plot_df.loc[mask, "umap_x"],
                    plot_df.loc[mask, "umap_y"],
                    c=[phenotype_to_color[phenotype]],
                    label=phenotype,
                    s=point_size,
                    alpha=0.6,
                    rasterized=True,
                )
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            axes[1].set_title(f"After Assignment: Phenotypes (n={len(unique_phenotypes)})")
    else:
        axes[1].scatter(
            plot_df["umap_x"],
            plot_df["umap_y"],
            c="#9AA0A6",
            s=point_size,
            alpha=0.6,
            rasterized=True,
        )
        axes[1].set_title("After Assignment: Awaiting phenotype labels")

    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _render_vortex_visual_validation(
    project_root: Path,
    df_results: pd.DataFrame,
    norm_cols: list[str],
    phenotype_map: dict,
    output_col: str,
    low_expr_cluster_id,
) -> None:
    with st.expander("Visual Validation of Vortex Clusters", expanded=False):
        st.caption(
            "Preview representative cells directly from Vortex cluster numbers before naming them, "
            "or switch to phenotype view after assignment."
        )

        if CellVisualizer is None:
            st.error("CellVisualizer is not available in this environment.")
            return

        viz_roi = st.session_state.get("viz_roi_dir")
        viz_seg = st.session_state.get("viz_seg_dir")
        if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
            st.warning("Please configure valid ROI and segmentation directories in Visual Validation Settings.")
            return

        batch_col = st.session_state.get("viz_col_batch", "batch_id")
        x_col = st.session_state.get("viz_col_x", "x_centroid")
        y_col = st.session_state.get("viz_col_y", "y_centroid")
        cell_id_col = st.session_state.get("vortex_cell_id_col", "cell_id")

        missing_cols = [column for column in [batch_col, x_col, y_col] if column not in df_results.columns]
        if missing_cols:
            st.error(f"Visual validation requires these columns in the Vortex result table: {missing_cols}")
            return

        preview_df = _build_vortex_preview_df(df_results, output_col, phenotype_map)

        viz = CellVisualizer(viz_roi, viz_seg)
        existing_matches = st.session_state.get("viz_file_matches") or {}
        if not existing_matches:
            unique_batches = preview_df[batch_col].dropna().astype(str).unique().tolist()
            matches = viz.match_file_triads(unique_batches)
            st.session_state["viz_file_matches"] = matches
            if not matches:
                st.error("No matching ROI/segmentation file pairs were found for the current Vortex batches.")
                return
        else:
            viz.file_map = existing_matches

        matched_batches = sorted(str(batch) for batch in viz.file_map.keys())
        if not matched_batches:
            st.error("No matched ROI/segmentation file pairs are cached for visual validation.")
            return

        preview_df = preview_df[preview_df[batch_col].astype(str).isin(matched_batches)].copy()
        if preview_df.empty:
            st.warning("No Vortex cells fall into batches with matched ROI/segmentation files.")
            return

        has_named_phenotypes = (
            output_col in preview_df.columns
            and preview_df[output_col].fillna("Unassigned").astype(str).ne("Unassigned").any()
        )
        preview_mode = st.radio(
            "Preview By",
            ["Cluster number", "Assigned phenotype"],
            horizontal=True,
            index=0,
            key="vortex_viz_mode",
        )
        if preview_mode == "Assigned phenotype" and not has_named_phenotypes:
            st.info("No phenotype labels are assigned yet. Cluster-number preview is available immediately after clustering.")
            preview_mode = "Cluster number"

        def _format_cluster_option(cluster_value) -> str:
            display_label = _cluster_display_label(cluster_value, low_expr_cluster_id)
            phenotype_label = phenotype_map.get(_normalize_cluster_id(cluster_value), "Unassigned")
            if phenotype_label and phenotype_label != "Unassigned":
                return f"{display_label} [{phenotype_label}]"
            return display_label

        with st.form("vortex_viz_form"):
            vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])

            with vv_col1:
                if preview_mode == "Cluster number":
                    cluster_options = sorted(
                        [_normalize_cluster_id(cluster_id) for cluster_id in preview_df["vortex_cluster"].dropna().unique()],
                        key=_cluster_sort_key,
                    )
                    selected_group = st.selectbox(
                        "Select Cluster",
                        cluster_options,
                        format_func=_format_cluster_option,
                        key="vortex_viz_cluster_sel",
                    )
                else:
                    phenotype_options = sorted(preview_df[output_col].fillna("Unassigned").astype(str).unique().tolist())
                    selected_group = st.selectbox(
                        "Select Phenotype",
                        phenotype_options,
                        key="vortex_viz_pheno_sel",
                    )

            with vv_col2:
                n_viz_cells = st.number_input(
                    "Cells to View",
                    min_value=1,
                    max_value=50,
                    value=int(st.session_state.get("viz_start_n_cells", 5)),
                    key="vortex_viz_n_cells",
                )

            with vv_col3:
                show_neighbors = st.checkbox("Show Neighbors", value=True, key="vortex_viz_neighbors")

            with vv_col4:
                batch_scope = st.radio(
                    "Batch Scope",
                    ["All matched batches", "Specific batch"],
                    horizontal=True,
                    key="vortex_viz_batch_scope",
                )
                selected_batch = None
                if batch_scope == "Specific batch":
                    selected_batch = st.selectbox("Batch", matched_batches, key="vortex_viz_batch_sel")
                st.write("")
                run_viz = st.form_submit_button("Visualize Random Cells", type="primary")

        if run_viz:
            if preview_mode == "Cluster number":
                subset = preview_df[preview_df["vortex_cluster"].apply(lambda value: _cluster_matches(value, selected_group))].copy()
                selection_label = _format_cluster_option(selected_group)
            else:
                subset = preview_df[preview_df[output_col].fillna("Unassigned").astype(str) == str(selected_group)].copy()
                selection_label = str(selected_group)

            if batch_scope == "Specific batch" and selected_batch is not None:
                subset = subset[subset[batch_col].astype(str) == str(selected_batch)].copy()

            subset = subset.dropna(subset=[x_col, y_col])
            if subset.empty:
                st.warning("No cells matched the selected preview criteria with valid coordinates.")
                return

            numeric_norm_cols = [
                column
                for column in norm_cols
                if column in subset.columns and pd.api.types.is_numeric_dtype(subset[column])
            ]
            top_marker_cols = []
            if numeric_norm_cols:
                top_marker_cols = subset[numeric_norm_cols].mean().nlargest(min(5, len(numeric_norm_cols))).index.tolist()

            marker_caption = ", ".join(_clean_marker_name(marker) for marker in top_marker_cols)
            if marker_caption:
                st.info(f"Top markers used for image overlays: {marker_caption}")
            else:
                st.info("No normalized marker columns were available. Using the matched TIFF channels directly.")

            n_actual = min(int(n_viz_cells), len(subset))
            sampled = subset.sample(n=n_actual, random_state=int(time.time())).sort_values(batch_col)
            sampled_batches = sampled[batch_col].astype(str).drop_duplicates().tolist()
            prewarm_fn = getattr(viz, "prewarm_batch", None)
            if callable(prewarm_fn):
                with st.spinner(f"Preloading {len(sampled_batches)} matched batch cache(s)..."):
                    for batch_id in sampled_batches:
                        prewarm_fn(batch_id, requested_channels=top_marker_cols or None)

            st.markdown("---")
            st.markdown(f"#### Representative Cell Preview: {selection_label}")
            cols = st.columns(3)
            progress_bar = st.progress(0.0)
            tile_size = int(st.session_state.get("viz_tile_size", 40))
            rendered = 0

            for idx, (_, row) in enumerate(sampled.iterrows()):
                try:
                    cell_data = viz.get_cell_context(
                        str(row[batch_col]),
                        float(row[x_col]),
                        float(row[y_col]),
                        tile_size=tile_size,
                        requested_channels=top_marker_cols or None,
                    )

                    returned_names = cell_data.get("channel_names") or []
                    if returned_names:
                        marker_names = [_clean_marker_name(name) for name in returned_names]
                        marker_indices = list(range(len(marker_names)))
                    else:
                        channel_count = int(cell_data["image"].shape[0]) if getattr(cell_data.get("image"), "ndim", 0) >= 1 else 0
                        marker_indices = list(range(min(5, channel_count)))
                        marker_names = [f"Ch {channel_idx}" for channel_idx in marker_indices]

                    if not marker_indices:
                        raise ValueError("No TIFF channels were available for this cell preview")

                    fig = viz.generate_visualization(
                        cell_data,
                        marker_indices,
                        marker_names,
                        display_size=tile_size,
                        show_neighbors=show_neighbors,
                    )

                    cluster_label = _cluster_display_label(row["vortex_cluster"], low_expr_cluster_id)
                    phenotype_label = str(row.get(output_col, "Unassigned"))
                    cell_identifier = row.get(cell_id_col, "n/a")

                    with cols[idx % 3]:
                        st.pyplot(fig)
                        st.caption(
                            f"Cell: {cell_identifier} | Batch: {row[batch_col]} | "
                            f"Cluster: {cluster_label} | Phenotype: {phenotype_label}"
                        )
                        st.caption(f"Location: ({float(row[x_col]):.1f}, {float(row[y_col]):.1f})")
                    plt.close(fig)
                    rendered += 1
                except Exception as exc:
                    with cols[idx % 3]:
                        st.warning(f"Could not render cell preview: {exc}")
                finally:
                    progress_bar.progress((idx + 1) / len(sampled))

            if rendered:
                st.success(f"Rendered {rendered} representative cell preview(s) for {selection_label}.")
            else:
                st.warning("No cell previews could be rendered for the selected group.")


def _reset_vortex_state(clear_run_state_only: bool = False) -> None:
    keys = [
        "vortex_phenotype_map",
        "vortex_available_phenotypes",
        "vortex_temp_file",
        "vortex_loaded_hash",
        "vortex_k_results_path",
        "vortex_k_plot_path",
        "vortex_qc_dashboard_path",
        "vortex_selected_k",
        "vortex_has_run",
        "vortex_run_input_path",
        "vortex_run_id",
        "vortex_loaded_source_path",
        "vortex_n_clusters",
        "vortex_markers",
        "vortex_norm_cols",
        "vortex_k_recommendation",
        "vortex_run_summary",
        "vortex_run_parameters",
        "vortex_qc_dashboard_pre_path",
        "vortex_qc_dashboard_post_path",
        "vortex_post_qc_hash",
        "vortex_run_log_lines",
        "vortex_low_expr_cluster_id",
        "vortex_low_expr_count",
        "vortex_low_expr_metric",
        "vortex_low_expr_threshold",
        "vortex_low_expr_default_pheno",
        "vortex_heatmap_component_event_id",
        "vortex_heatmap_component_reset_counter",
    ]
    if clear_run_state_only:
        keys = [key for key in keys if key not in {"vortex_available_phenotypes"}]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def _render_vortex_help_panel() -> None:
    with st.expander("Vortex Quick Guide: Method, K Selection, and Outputs", expanded=False):
        st.markdown(
            """
            **What Vortex/X-shift does**

            Vortex is a density-gradient clustering method. For each cell, it looks at a local
            `k`-nearest-neighbor neighborhood, estimates density, and follows the steepest density
            ascent to a local maximum. Cells converging to the same maximum become one cluster.

            **How to use the K diagnostics**

            - `Cluster Count vs K`: larger `k` usually merges clusters and reduces fragmentation.
            - `Silhouette vs K`: higher is better; use it comparatively, not as an absolute threshold.
            - `Davies-Bouldin vs K`: lower is better; it favors more compact, better-separated clusters.

            **Recommended workflow**

            1. Start with auto-K enabled.
            2. Inspect the K diagnostics and final cluster count.
            3. Check the heatmap and UMAP/QC tabs for biological plausibility.
            4. If satisfied, optionally lock `VORTEX_K` and disable auto-K for stable reruns.
            """
        )


def render_vortex(project_root: Path, global_input_path: str | None) -> None:
    st.subheader("Vortex Clustering Phenotyping")
    render_phenotyping_run_log_controls(Path(project_root), "vortex", "Vortex Clustering")
    st.markdown(
        "Professional Vortex/X-shift clustering with notebook-synchronized parameters, "
        "interactive phenotype assignment, and diagnostic figures."
    )
    _render_vortex_help_panel()

    cache_dir = project_root / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    vortex_cache = cache_dir / "vortex_params.json"

    if vortex_cache.exists():
        with open(vortex_cache, "r") as handle:
            cached_params = json.load(handle)
    else:
        cached_params = {}

    render_settings_export_import(
        "Vortex Clustering",
        ["vortex_"],
        {"vortex_params.json": vortex_cache},
    )

    def _apply_norm_for_heatmap(df_in: pd.DataFrame, markers: list[str]) -> tuple[pd.DataFrame, list[str]]:
        valid_markers = [marker for marker in markers if marker in df_in.columns]
        
        # Determine active cofactor
        cofactor = 5.0
        if 'bc_config' in st.session_state:
            cofactor = float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0))
            
        use_sinh = st.session_state.get('vortex_sinh', True)
        df_input = df_in
        if use_sinh:
            df_input = df_in.copy()
            for m in valid_markers:
                if m.endswith('_corr'):
                    df_input[m] = np.sinh(df_input[m]) * cofactor

        if ph_vortex_utils is not None:
            group_cols = [batch_id_col] if batch_id_col in df_input.columns else []
            norm_subset = ph_vortex_utils.apply_flexible_normalization(
                df=df_input,
                marker_cols=valid_markers,
                group_cols=group_cols,
                suffix="_norm",
                apply_z1=norm_z1,
                apply_z2=norm_z2,
                apply_cdf=norm_cdf,
                apply_nlog=norm_nlog,
            )
            norm_data = df_in.copy()
            norm_cols_local = list(norm_subset.columns)
            norm_data = pd.concat([norm_data, norm_subset], axis=1)
            norm_data[norm_cols_local] = norm_data[norm_cols_local].fillna(0)
            return norm_data, norm_cols_local

        norm_cols_local: list[str] = []
        norm_data = df_in.copy()
        for marker in valid_markers:
            if marker not in df_input.columns:
                continue
            col_data = df_input[marker].fillna(0).copy()
            if norm_nlog:
                col_data = np.log1p(col_data)
            if norm_cdf:
                col_data = col_data.rank(pct=True)
            if norm_z1:
                std_val = col_data.std()
                col_data = (col_data - col_data.mean()) / (std_val if std_val > 1e-9 else 1)
            if norm_z2:
                logger.warning("Z2 (per-cell normalization) is enabled - verify this is intended")
            norm_col = f"{marker}_norm"
            norm_data[norm_col] = col_data.fillna(0)
            norm_cols_local.append(norm_col)
        return norm_data, norm_cols_local

    st.markdown("##### Input File Selection")
    cached_input_dir = cached_params.get("input_dir", str(project_root / "outputs/batch_corrected"))
    cached_input_file = cached_params.get("input_file", "")

    use_global = st.checkbox(
        "Use Global Input File",
        value=cached_params.get("use_global", True),
        key="vortex_use_global",
    )

    input_dir = cached_input_dir
    selected_file = cached_input_file
    full_input_path = global_input_path

    if not use_global:
        input_dir = st.text_input(
            "Input Directory",
            value=cached_input_dir,
            help="Directory containing batch-corrected CSV files",
            key="vortex_input_dir",
        )
        available_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")]) if input_dir and os.path.exists(input_dir) else []
        if available_files:
            default_idx = available_files.index(cached_input_file) if cached_input_file in available_files else 0
            selected_file = st.selectbox(
                "Input File",
                available_files,
                index=default_idx,
                key="vortex_infile_select",
            )
            full_input_path = os.path.join(str(input_dir), selected_file)
        else:
            st.warning(f"No CSV files found in: {input_dir}")
            full_input_path = None

    if full_input_path:
        st.caption(f"Active File: {os.path.basename(full_input_path)}")

    available_columns: list[str] = []
    all_marker_candidates: list[str] = []
    if full_input_path and os.path.exists(full_input_path):
        try:
            sample_df = pd.read_csv(full_input_path, nrows=5)
            available_columns = list(sample_df.columns)
            metadata_cols = [
                "cell_id", "batch_id", "tissue", "x_centroid", "y_centroid",
                "area", "perimeter", "eccentricity", "solidity", "extent",
                "major_axis", "minor_axis", "orientation", "label", "index",
            ]
            metadata_cols_lower = [column.lower() for column in metadata_cols]
            all_marker_candidates = [
                column
                for column in available_columns
                if column.lower() not in metadata_cols_lower
                and not column.startswith("umap_")
                and not column.startswith("vortex_")
                and not column.startswith("phenotype_")
                and not column.endswith("_norm")
            ]
        except Exception as exc:
            st.warning(f"Could not read file: {exc}")

    available_marker_columns = list(all_marker_candidates)
    all_marker_candidates, marker_workspace_note = get_marker_workspace_panel(
        all_marker_candidates,
        full_input_path,
    )

    st.markdown("##### Marker Selection")
    st.caption(f"Select a Vortex subset from the shared panel. {marker_workspace_note}")

    if full_input_path and os.path.exists(full_input_path):
        notebook_corr_markers, notebook_non_corr_markers = _extract_markers_from_notebook("Step6E_phenotype_vortex.ipynb")
        marker_workspace = st.session_state.get("tab7_marker_workspace", {})
        expected_markers = resolve_notebook_marker_defaults(
            notebook_corr_markers,
            notebook_non_corr_markers,
            all_marker_candidates,
            str(marker_workspace.get("source_mode", "Raw markers")),
            str(marker_workspace.get("affix_position", "Suffix")),
            str(marker_workspace.get("correction_affix", "_corr")),
        )

    else:
        if "vortex_confirmed_markers" not in st.session_state:
            st.session_state["vortex_confirmed_markers"] = []

    _, markers_selected, marker_workspace_note = render_method_marker_selection(
        key_prefix="vortex",
        method_name="Vortex",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers if full_input_path else [],
        saved_markers=cached_params.get("markers", []),
    )

    with st.expander("Configuration & Parameters", expanded=True):
        with st.form("vortex_config_form", clear_on_submit=False):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Column Configuration")
                default_cell_id = "cell_id" if "cell_id" in available_columns else (available_columns[0] if available_columns else "cell_id")
                cell_id_options = [""] + available_columns if available_columns else [""]
                initialize_optional_column_widget_state(
                    "vortex_cell_id_col",
                    available_columns,
                    cached_params.get("cell_id_col", default_cell_id),
                )
                cell_id_col = st.selectbox(
                    "Cell ID Column",
                    cell_id_options,
                    index=cell_id_options.index(st.session_state.get("vortex_cell_id_col", default_cell_id)) if st.session_state.get("vortex_cell_id_col", default_cell_id) in cell_id_options else 0,
                    format_func=lambda value: value or "(None)",
                    help=get_cell_id_column_help_text(),
                    key="vortex_cell_id_col",
                )

                batch_id_options = [""] + available_columns if available_columns else [""]
                initialize_optional_column_widget_state(
                    "vortex_batch_id_col",
                    available_columns,
                    cached_params.get("batch_id_col", ""),
                )
                idx_batch_id = batch_id_options.index(st.session_state.get("vortex_batch_id_col", "")) if st.session_state.get("vortex_batch_id_col", "") in batch_id_options else 0
                batch_id_col = st.selectbox(
                    "Batch ID Column",
                    batch_id_options,
                    index=idx_batch_id,
                    format_func=lambda value: value or "(None)",
                    help=get_batch_id_column_help_text(),
                    key="vortex_batch_id_col",
                )

                st.markdown("**Filtering Criteria**")
                enable_filter = st.checkbox(
                    "Enable Row Filtering",
                    value=cached_params.get("enable_filter", False),
                    help=get_enable_row_filter_help_text(),
                    key="vortex_enable_filter",
                )
                if enable_filter:
                    filter_cols = [column for column in available_columns if column not in [cell_id_col]]
                    default_filter_col = "tissue" if "tissue" in filter_cols else (filter_cols[0] if filter_cols else "tissue")
                    cached_filter_col = cached_params.get("filter_col", default_filter_col)
                    default_filter_idx = filter_cols.index(cached_filter_col) if cached_filter_col in filter_cols else (filter_cols.index(default_filter_col) if default_filter_col in filter_cols else 0)
                    filter_col = st.selectbox(
                        "Filter Column",
                        filter_cols if filter_cols else [default_filter_col],
                        index=default_filter_idx,
                        help=get_filter_column_help_text(),
                        key="vortex_filter_col",
                    )
                    filter_val = st.text_input(
                        "Filter Value",
                        cached_params.get("filter_val", "1"),
                        help=get_filter_value_help_text(),
                        key="vortex_filter_val",
                    )
                else:
                    filter_col = None
                    filter_val = None

            with col2:
                st.markdown("##### Vortex / X-shift Parameters")
                st.caption("Notebook defaults synchronized with Step6E")

                auto_select_k = st.checkbox(
                    "Auto-select K",
                    value=cached_params.get("auto_select_k", True),
                    help=(
                        "Scan the candidate K values on a sketch of the data, score the results, and choose a recommendation before the final full run. The recommendation prefers reasonable cluster counts and penalizes candidates dominated by one very large cluster. Use this for exploratory runs; turn it off only when you already know the K you want."
                    ),
                    key="vortex_auto_select_k",
                )

                k_col1, k_col2 = st.columns(2)
                vortex_k = k_col1.number_input(
                    "Manual K",
                    1,
                    20000,
                    int(cached_params.get("vortex_k", 35)),
                    help=(
                        "Neighborhood size used by the density-gradient Vortex algorithm when auto-K is off or the scan cannot choose a candidate. Smaller K makes the density estimate more local and usually increases fragmentation; larger K smooths the density landscape and usually merges clusters. "
                        "K interacts with the subsample size: the cluster count tracks roughly K/subsample_size, so doubling the subsample needs roughly double the K for a comparable number of clusters."
                    ),
                    key="vortex_k_value",
                )
                n_clusters_target = k_col2.number_input(
                    "Target Clusters",
                    2,
                    200,
                    int(cached_params.get("n_clusters_target", 45)),
                    help=(
                        "Soft target used only during auto-K selection. It does not force the final run to produce exactly this number of clusters; it simply biases the recommendation toward K values whose scanned cluster count is near this range."
                    ),
                    key="vortex_n_clusters_target",
                )

                k_candidates_text = st.text_input(
                    "K Candidates",
                    cached_params.get("k_candidates_text", "10, 15, 20, 25, 30, 35, 45, 60"),
                    help=(
                        "Candidate K values tested during the auto-K scan. Include a sensible spread from more local to more global neighborhoods so the diagnostic pass can compare fragmented versus merged solutions."
                    ),
                    key="vortex_k_candidates_text",
                )

                perf_col1, perf_col2 = st.columns(2)
                k_scan_sketch_size = perf_col1.number_input(
                    "K-Scan Sketch Size",
                    1000,
                    1000000,
                    int(cached_params.get("k_scan_sketch_size", 25000)),
                    1000,
                    help=(
                        "Number of cells used only for the K diagnostic scan. Larger values give a more stable recommendation but make the scan slower. This does not change the size of the final clustering run."
                    ),
                    key="vortex_k_scan_sketch_size",
                )
                subsample_size = perf_col2.number_input(
                    "Clustering Subsample Size",
                    1000,
                    5000000,
                    int(cached_params.get("subsample_size", 150000)),
                    1000,
                    help=(
                        "Number of cells clustered directly in the final Vortex run before the remaining cells are assigned by KNN backfilling. Larger sketches usually capture structure better, especially for rare populations, but take longer."
                    ),
                    key="vortex_subsample_size",
                )

                gpu_col1, gpu_col2 = st.columns(2)
                _gpu_name = ph_vortex_utils.gpu_device_name() if ph_vortex_utils is not None else None
                use_gpu = gpu_col1.checkbox(
                    "Use GPU when available",
                    value=cached_params.get("use_gpu", True),
                    help=(
                        "Run the Vortex neighbour search and the KNN backfill on the GPU with an exact "
                        "brute-force search. Results match the CPU KD-tree path; only the speed differs. "
                        "Falls back to CPU automatically when no GPU is present."
                    ),
                    key="vortex_use_gpu",
                )
                max_geosketch_input = gpu_col2.number_input(
                    "Geosketch Input Cap",
                    0,
                    10000000,
                    int(cached_params.get("max_geosketch_input", 0)),
                    50000,
                    help=(
                        "Escape hatch for memory pressure only; 0 (the default) means no cap. "
                        "Geometric sketching earns its cost by flattening density, and that flattening "
                        "depends on how many cells it can choose from. Capping the input near the sketch "
                        "size forces it to keep almost every candidate, which reproduces the raw density "
                        "and collapses the run into one dominant cluster. Leave at 0 unless you hit RAM limits."
                    ),
                    key="vortex_max_geosketch_input",
                )
                if _gpu_name:
                    st.caption(f"GPU detected: {_gpu_name}")
                else:
                    st.caption("No GPU detected; the CPU path will be used automatically.")

                subsample_method_options = ["geosketch", "geosketch (fast)", "random"]
                cached_subsample_method = cached_params.get("subsample_method", "geosketch")
                if cached_subsample_method not in subsample_method_options:
                    cached_subsample_method = "geosketch"
                subsample_method = st.selectbox(
                    "Subsampling Method",
                    subsample_method_options,
                    index=subsample_method_options.index(cached_subsample_method),
                    help=(
                        "How the Vortex sketch is chosen. 'Geosketch' uses a grid-covering algorithm over "
                        "expression space (PCA-reduced) and is the recommended default: it flattens density, "
                        "which is what lets the density-gradient step resolve distinct populations instead of "
                        "collapsing into one dominant cluster. It now uses a vectorized implementation that "
                        "sketches millions of cells in seconds. 'Geosketch (fast)' is an older coarse binned "
                        "approximation. 'Random' is fastest but preserves the raw density, which typically "
                        "yields far fewer, more unbalanced clusters."
                    ),
                    key="vortex_subsample_method",
                )

                meta_clusters = st.number_input(
                    "Merge into N meta-clusters (0 = off)",
                    0,
                    2000,
                    int(cached_params.get("meta_clusters", 0)),
                    10,
                    help=(
                        "After clustering, merge clusters with similar marker profiles until "
                        "this many remain (hierarchical/Ward merge on cluster means, the same "
                        "step the SOM method applies to its nodes).\n\n"
                        "Prefer raising K first. Merging groups clusters by centroid distance "
                        "alone and ignores the density structure, so it can fuse populations "
                        "separated by a real density valley; raising K instead re-estimates the "
                        "density landscape and lets the algorithm decide what is genuinely one "
                        "mode. On a 2.8M-cell benchmark, reaching ~110 clusters by raising K "
                        "scored about 2.5 points higher than merging down to the same count, and "
                        "held up better on rare populations.\n\n"
                        "Use this when you need an exact cluster count -- for example to match "
                        "granularity across methods before a consensus vote -- or when tuning K "
                        "is impractical. Set 0 to keep every cluster."
                    ),
                    key="vortex_meta_clusters",
                )

                neighbor_col1, neighbor_col2 = st.columns(2)
                knn_neighbors = neighbor_col1.number_input(
                    "KNN Backfill Neighbors",
                    1,
                    100,
                    int(cached_params.get("knn_neighbors", 15)),
                    help=(
                        "Number of neighbors used when transferring Vortex cluster labels from the clustered sketch to cells that were not in the sketch. Smaller values keep assignments local; larger values smooth the boundary between clusters."
                    ),
                    key="vortex_knn_neighbors",
                )
                min_cluster_size = neighbor_col2.number_input(
                    "Min Cluster Size",
                    1,
                    5000,
                    int(cached_params.get("min_cluster_size", 1)),
                    help=(
                        "Minimum allowed cluster size after the core Vortex step. Clusters smaller than this are merged into their nearest larger cluster, which suppresses tiny unstable groups and outliers. Higher values merge more aggressively."
                    ),
                    key="vortex_min_cluster_size",
                )

                st.markdown("**UMAP & Plotting**")
                has_umap_in_csv = "umap_x" in available_columns and "umap_y" in available_columns
                # UMAP is visualisation/QC only and never changes the Vortex
                # assignments, so it is off by default rather than on whenever the
                # input happens to lack the columns.
                default_generate_umap = False
                generate_umap = st.checkbox(
                    "Generate UMAP",
                    value=cached_params.get("generate_umap", default_generate_umap),
                    help=(
                        "Build a UMAP on the clustered Vortex sketch and then place non-sketched cells at the centroid of their assigned cluster. This is for visualization and QC only; it does not change the Vortex cluster assignments."
                    ),
                    key="vortex_generate_umap",
                )
                if has_umap_in_csv:
                    st.caption("Pre-computed UMAP coordinates detected in the input; regenerating is unnecessary.")
                else:
                    st.caption(
                        "UMAP is off by default: it is used only for visualisation and QC, and on large "
                        "datasets it can take longer than the clustering itself."
                    )
                plot_col1, plot_col2, plot_col3 = st.columns(3)
                umap_neighbors = plot_col1.number_input(
                    "UMAP Neighbors",
                    2,
                    100,
                    int(cached_params.get("umap_neighbors", 15)),
                    help=(
                        "Neighborhood size for the visualization UMAP built from the Vortex sketch. Smaller values emphasize local separation; larger values give a smoother global layout. This setting does not affect clustering itself."
                    ),
                    key="vortex_umap_neighbors",
                )
                max_plot_cells = plot_col2.number_input(
                    "Max Plot Cells",
                    1000,
                    1000000,
                    int(cached_params.get("max_plot_cells", 250000)),
                    1000,
                    help=(
                        "Maximum number of cells drawn in live QC and UMAP figures. It only changes rendering speed and figure density, not saved cluster assignments or exports."
                    ),
                    key="vortex_max_plot_cells",
                )
                random_seed = plot_col3.number_input(
                    "Random Seed",
                    0,
                    10000,
                    int(cached_params.get("random_seed", 42)),
                    help=get_random_seed_help_text("K-scan sketches, final Vortex subsampling, and UMAP initialization"),
                    key="vortex_random_seed",
                )

                st.markdown("**Normalization Flags**")
                norm_col1, norm_col2, norm_col3, norm_col4 = st.columns(4)
                norm_z1 = norm_col1.checkbox("Z1", value=cached_params.get("norm_z1", True), help=get_normalization_flag_help_text("z1"), key="vortex_z1")
                norm_z2 = norm_col2.checkbox("Z2", value=cached_params.get("norm_z2", False), help=get_normalization_flag_help_text("z2"), key="vortex_z2")
                norm_cdf = norm_col3.checkbox("CDF", value=cached_params.get("norm_cdf", False), help=get_normalization_flag_help_text("cdf"), key="vortex_cdf")
                norm_nlog = norm_col4.checkbox("NLog", value=cached_params.get("norm_nlog", False), help=get_normalization_flag_help_text("nlog"), key="vortex_nlog")
                
                norm_sinh = st.checkbox(
                    "Inverse ArcSinh (Sinh) for Corrected Markers", 
                    value=cached_params.get("norm_sinh", True), 
                    help="Automatically convert batch-corrected (_corr) markers back to raw scale using sinh(val) * cofactor before normalization to maintain high-contrast 'ice heatmap' visualization.",
                    key="vortex_sinh"
                )

                st.markdown("**Output Configuration**")
                enable_low_expr_filter = st.checkbox(
                    "Flag Low-Expression Cells",
                    value=cached_params.get("enable_low_expr_filter", True),
                    help=get_low_expression_filter_help_text("Vortex clustering"),
                    key="vortex_low_expr_filter",
                )
                low_expr_metric = cached_params.get("low_expr_metric", "mean_zscore")
                if low_expr_metric not in LOW_EXPRESSION_METRIC_OPTIONS:
                    low_expr_metric = "mean_zscore"
                if enable_low_expr_filter:
                    low_expr_metric = st.selectbox(
                        "Low-Expression Metric",
                        options=list(LOW_EXPRESSION_METRIC_OPTIONS),
                        index=list(LOW_EXPRESSION_METRIC_OPTIONS).index(low_expr_metric),
                        format_func=get_low_expression_metric_label,
                        help=get_low_expression_metric_help_text(),
                        key="vortex_low_expr_metric",
                    )
                    vortex_default_thresh = cached_params.get("low_expr_threshold")
                    if vortex_default_thresh is None:
                        vortex_default_thresh = cached_params.get("min_marker_sum")
                    if vortex_default_thresh is None:
                        vortex_default_thresh = get_low_expression_default_threshold(low_expr_metric)
                    low_expr_threshold = st.number_input(
                        "Low-Expression Threshold",
                        -10.0,
                        100000.0,
                        float(vortex_default_thresh),
                        0.05 if low_expr_metric == "mean_zscore" else 0.1,
                        help=(
                            "Cells at or below this score are preserved in the output but excluded from the main Vortex run and placed in the low-expression bucket. "
                            "For mean marker z-score, start around -0.75."
                            if low_expr_metric == "mean_zscore"
                            else "Cells at or below this raw selected-marker sum are preserved in the output but excluded from the main Vortex run and placed in the low-expression bucket."
                        ),
                        key="vortex_low_expr_threshold",
                    )
                    if low_expr_metric == "mean_zscore":
                        st.caption(
                            "Recommended starting cutoff: -0.75. Lower values are more permissive; values closer to 0 move more panel-dim cells into the preserved bucket."
                        )
                    else:
                        st.caption(
                            "Raw-sum mode uses the pre-normalization selected markers. A cutoff of 0.1 is usually only near-zero signal."
                        )
                else:
                    low_expr_threshold = get_low_expression_default_threshold(low_expr_metric)

                output_col_name = st.text_input(
                    "Output Column Name",
                    cached_params.get("output_col", "phenotype_vortex"),
                    help=get_output_column_help_text(),
                    key="vortex_outcol",
                )

            save_col1, save_col2 = st.columns([1, 3])
            with save_col1:
                if st.form_submit_button("Save Parameters", help="Save current parameters to cache", key="vortex_save_params_btn"):
                    params_to_save = {
                        "input_dir": input_dir,
                        "input_file": selected_file,
                        "use_global": use_global,
                        "markers": markers_selected,
                        "cell_id_col": cell_id_col,
                        "batch_id_col": batch_id_col,
                        "auto_select_k": auto_select_k,
                        "vortex_k": int(vortex_k),
                        "k_candidates_text": k_candidates_text,
                        "n_clusters_target": int(n_clusters_target),
                        "k_scan_sketch_size": int(k_scan_sketch_size),
                        "subsample_size": int(subsample_size),
                        "subsample_method": subsample_method,
                        "meta_clusters": int(meta_clusters),
                        "use_gpu": bool(use_gpu),
                        "max_geosketch_input": int(max_geosketch_input),
                        "knn_neighbors": int(knn_neighbors),
                        "min_cluster_size": int(min_cluster_size),
                        "generate_umap": generate_umap,
                        "umap_neighbors": int(umap_neighbors),
                        "max_plot_cells": int(max_plot_cells),
                        "random_seed": int(random_seed),
                        "norm_z1": norm_z1,
                        "norm_z2": norm_z2,
                        "norm_cdf": norm_cdf,
                        "norm_nlog": norm_nlog,
                        "norm_sinh": norm_sinh,
                        "enable_low_expr_filter": enable_low_expr_filter,
                        "low_expr_metric": low_expr_metric,
                        "low_expr_threshold": float(low_expr_threshold) if low_expr_threshold is not None else 0.0,
                        "min_marker_sum": float(low_expr_threshold) if low_expr_threshold is not None else 0.0,
                        "output_col": output_col_name,
                        "enable_filter": enable_filter,
                        "filter_col": filter_col if enable_filter else "",
                        "filter_val": filter_val if enable_filter else "",
                    }
                    with open(vortex_cache, "w") as handle:
                        json.dump(params_to_save, handle, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                    st.success("Parameters saved to cache and project.")
                    st.rerun()

            with save_col2:
                st.caption(f"Parameters will be persisted in: `{vortex_cache.name}`")

    if full_input_path and os.path.exists(full_input_path):
        st.divider()
        st.markdown("##### Existing Vortex Results")

        can_load_existing = False
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            can_load_existing = "vortex_cluster" in df_head.columns and output_col_name in df_head.columns
        except Exception:
            can_load_existing = False

        load_col1, load_col2 = st.columns([1, 1])
        with load_col1:
            load_existing = st.button(
                "Load Existing Vortex Results",
                disabled=(not can_load_existing or not markers_selected),
                help="Load vortex_cluster plus phenotype_vortex from the unified CSV and rebuild the heatmap.",
                key="vortex_load_existing_btn",
            )
        with load_col2:
            start_fresh = st.button(
                "Start Fresh (Ignore Existing)",
                help="Clear loaded Vortex results and start a new clustering run.",
                key="vortex_start_fresh_btn",
            )

        if start_fresh:
            _reset_vortex_state()
            st.session_state['vortex_fresh_assignment_mode'] = True
            st.session_state['vortex_heatmap_component_reset_counter'] = (
                int(st.session_state.get('vortex_heatmap_component_reset_counter', 0)) + 1
            )
            st.success("Vortex state cleared. Ready for a fresh run.")

        if load_existing:
            try:
                st.session_state['vortex_fresh_assignment_mode'] = False
                df_full = _dataio_load(full_input_path)
                cluster_col = "vortex_cluster"
                pheno_col = output_col_name

                norm_df, norm_cols_loaded = _apply_norm_for_heatmap(df_full, markers_selected)
                pairs = norm_df[[cluster_col, pheno_col]].drop_duplicates().dropna()
                existing_map = {int(row[cluster_col]): row[pheno_col] for _, row in pairs.iterrows()}

                temp_file = create_session_temp_parquet_path(project_root, "vortex_existing")
                write_dataframe_parquet_atomic(norm_df, temp_file)

                st.session_state["vortex_temp_file"] = str(temp_file)
                st.session_state["vortex_markers"] = markers_selected
                st.session_state["vortex_norm_cols"] = norm_cols_loaded
                st.session_state["vortex_n_clusters"] = norm_df[cluster_col].nunique()
                st.session_state["vortex_phenotype_map"] = existing_map
                st.session_state["vortex_available_phenotypes"] = [
                    "Unassigned"
                ] + sorted({value for value in norm_df[pheno_col].dropna().unique().tolist() if str(value) != "Unassigned"})
                st.session_state["vortex_run_id"] = f"{time.time():.6f}"
                st.success("Loaded existing Vortex results from unified CSV.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load existing Vortex results: {exc}")

    st.divider()
    n_markers = len(markers_selected)
    if n_markers > 0:
        st.info(
            f"Ready to run with **{n_markers}** markers | Auto-K: **{'ON' if auto_select_k else 'OFF'}** | "
            f"Target clusters: ~**{n_clusters_target}**"
        )
    else:
        st.warning("Please select at least one marker to proceed")

    existing_run_log = logger.get_streamlit_buffer("vortex_run_log_lines")
    if existing_run_log:
        with st.expander("Last Vortex Tab Terminal", expanded=False):
            st.caption("Timestamped messages mirrored from the shell terminal during the last Vortex run.")
            st.code("\n".join(existing_run_log), language=None)

    if st.button("Run Vortex Clustering", type="primary", disabled=(n_markers == 0), key="vortex_run_btn"):
        if not full_input_path or not os.path.exists(full_input_path):
            st.error("Please select a valid input file.")
            return
        if not markers_selected:
            st.error("No markers selected. Please choose at least one marker.")
            return
        if ph_vortex_utils is None:
            st.error("ph_vortex_utils could not be imported. Please verify the Vortex utility module is available.")
            return

        run_audit = begin_phenotyping_run_audit(
            Path(project_root),
            "vortex",
            input_csv=full_input_path,
            markers=markers_selected,
        )
        audit_input_rows = None

        _reset_vortex_state(clear_run_state_only=True)

        with st.status("Running Vortex Pipeline (Notebook-Synchronized)...", expanded=True) as status:
            st.markdown("##### Vortex Tab Terminal")
            st.caption("Timestamped live log mirrored from the shell terminal.")
            tab_log_placeholder = st.empty()
            logger.start_streamlit_capture(tab_log_placeholder, "vortex_run_log_lines", clear=True)
            try:
                with Timer("Total Vortex Pipeline"):
                    logger.info("=" * 80)
                    logger.info("VORTEX CLUSTERING PIPELINE - Notebook Synchronized")
                    logger.info("=" * 80)

                    _log_vortex_step("1/5", "Loading input data", os.path.basename(full_input_path))
                    logger.info("Step 1: Loading data from %s", full_input_path)
                    df = _read_csv_fast(full_input_path)
                    audit_input_rows = len(df)
                    df, cell_id_anchor_col, using_internal_cell_anchor = ensure_cell_id_anchor(
                        df,
                        normalize_optional_column_name(cell_id_col),
                    )
                    logger.info(f"Loaded {len(df):,} rows from input file")

                    if enable_filter and filter_col and filter_val:
                        if filter_col in df.columns:
                            original_len = len(df)
                            try:
                                if pd.api.types.is_numeric_dtype(df[filter_col]):
                                    filter_num = float(filter_val)
                                    if pd.api.types.is_integer_dtype(df[filter_col]):
                                        filter_num = int(filter_num)
                                    df = df[df[filter_col] == filter_num]
                                else:
                                    df = df[df[filter_col].astype(str) == str(filter_val)]
                            except Exception:
                                df = df[df[filter_col].astype(str) == str(filter_val)]
                            logger.info(f"Filter applied: {filter_col}=={filter_val}: {original_len:,} -> {len(df):,} cells")
                        else:
                            logger.warning(f"Filter column '{filter_col}' not found. Using full dataset.")
                    else:
                        logger.info("No filtering applied. Using full dataset.")

                    if using_internal_cell_anchor:
                        logger.info("No valid Cell ID column selected/found. Using an internal row-order anchor for this Vortex run.")
                    else:
                        logger.info(f"Cell ID column '{cell_id_anchor_col}': {df[cell_id_anchor_col].nunique():,} unique cells")

                    missing_markers = [marker for marker in markers_selected if marker not in df.columns]
                    if missing_markers:
                        st.error(f"Missing markers in dataset: {missing_markers}")
                        logger.error(f"Missing markers: {missing_markers}")
                        return

                    logger.info(f"Processing {len(df):,} cells with {len(markers_selected)} markers")

                    # Isolate NaN/Inf cells in selected markers to prevent clustering artifacts
                    numeric_markers_temp = df[markers_selected].apply(pd.to_numeric, errors='coerce')
                    nan_mask = numeric_markers_temp.isna().any(axis=1) | np.isinf(numeric_markers_temp).any(axis=1)
                    nan_count = int(nan_mask.sum())
                    if nan_count > 0:
                        df_nan = df[nan_mask].copy()
                        df = df[~nan_mask].copy()
                        logger.info(f"Detected {nan_count:,} cells with NaN/Inf in selected markers. These cells will be excluded from clustering and UMAP, and will be assigned to cluster -1.")
                        if len(df) == 0:
                            st.error("All cells contain NaN values in the selected markers. Cannot perform clustering.")
                            logger.error("No valid cells remaining for clustering (all contain NaN).")
                            return
                    else:
                        df_nan = pd.DataFrame(columns=df.columns)

                    with Timer("Normalization"):
                        _log_vortex_step("2/5", "Applying normalization", f"markers={len(markers_selected)}")
                        logger.info("Step 2: Applying notebook-synchronized normalization")
                        logger.info(f"Normalization flags: Z1={norm_z1}, Z2={norm_z2}, CDF={norm_cdf}, NLog={norm_nlog}, Sinh={norm_sinh}")
                        
                        # Determine active cofactor
                        cofactor = 5.0
                        if 'bc_config' in st.session_state:
                            cofactor = float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0))
                            
                        df_input = df
                        if norm_sinh:
                            df_input = df.copy()
                            for m in markers_selected:
                                if m.endswith('_corr') and m in df_input.columns:
                                    df_input[m] = np.sinh(df_input[m]) * cofactor

                        batch_group_cols = [batch_id_col] if batch_id_col in df_input.columns else []
                        if batch_group_cols:
                            logger.info(f"Normalization grouping columns: {batch_group_cols}")
                        else:
                            logger.warning("Batch column not available. Falling back to global normalization.")

                        norm_subset = ph_vortex_utils.apply_flexible_normalization(
                            df=df_input,
                            marker_cols=markers_selected,
                            group_cols=batch_group_cols,
                            suffix="_norm",
                            apply_z1=norm_z1,
                            apply_z2=norm_z2,
                            apply_cdf=norm_cdf,
                            apply_nlog=norm_nlog,
                        )
                        norm_data = pd.concat([df.copy(), norm_subset], axis=1)
                        norm_cols = list(norm_subset.columns)
                        norm_data[norm_cols] = norm_data[norm_cols].fillna(0)

                        raw_sums = norm_data[markers_selected].sum(axis=1)
                        norm_data["vortex_marker_sum_raw"] = raw_sums
                        low_expr_scores = compute_low_expression_score(
                            df=norm_data,
                            raw_marker_cols=markers_selected,
                            norm_marker_cols=norm_cols,
                            metric=low_expr_metric,
                        )
                        norm_data["vortex_low_expr_score"] = low_expr_scores
                        low_expr_data = norm_data.iloc[0:0].copy()
                        low_expr_count = 0

                        if enable_low_expr_filter:
                            pre_filter = len(norm_data)
                            low_expr_thresh_val = float(low_expr_threshold) if low_expr_threshold is not None else float(get_low_expression_default_threshold(low_expr_metric))
                            low_expr_mask = low_expr_scores <= low_expr_thresh_val
                            low_expr_count = int(low_expr_mask.sum())
                            low_expr_data = norm_data[low_expr_mask].copy()
                            norm_data = norm_data[~low_expr_mask].copy()
                            logger.info(
                                "Low expression screen (%s <= %.4f): %s/%s cells preserved in the low-expression bucket; %s cells sent to Vortex clustering",
                                get_low_expression_metric_label(low_expr_metric),
                                low_expr_thresh_val,
                                f"{low_expr_count:,}",
                                f"{pre_filter:,}",
                                f"{len(norm_data):,}",
                            )
                        else:
                            logger.info("Low-expression screening is disabled.")

                        if norm_data.empty:
                            st.error(
                                "All cells fell into the low-expression bucket. Lower the expression threshold or disable low-expression screening."
                            )
                            logger.error("No cells remained for Vortex clustering after low-expression screening.")
                            return

                    selected_k = int(vortex_k)
                    k_results_path = None
                    k_plot_path = None
                    recommended_params = None
                    k_values = _parse_int_list(k_candidates_text or "", [10, 15, 20, 25, 30, 35, 45, 60])
                    plots_dir = project_root / "outputs/phenotyping/vortex/plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    if auto_select_k:
                        with Timer("Vortex K Diagnostic Scan"):
                            _log_vortex_step("3/5", "Scanning K candidates", f"candidates={k_values}")
                            logger.info("Step 3: Running Vortex K diagnostic scan")
                            logger.info(f"K candidates: {k_values}")
                            k_results = ph_vortex_utils.evaluate_vortex_k_grid(
                                df=norm_data,
                                norm_cols=norm_cols,
                                k_values=k_values,
                                sketch_size=int(k_scan_sketch_size),
                                subsample_method=subsample_method,
                                min_cluster_size=int(min_cluster_size),
                                random_seed=int(random_seed),
                                logger=logger,
                                use_gpu=use_gpu,
                                max_geosketch_input=int(max_geosketch_input),
                            )
                            k_results_path = project_root / "outputs/phenotyping/vortex_k_scan_results.csv"
                            k_results.to_csv(k_results_path, index=False)
                            k_plot_path = ph_vortex_utils.create_vortex_k_diagnostic_plot(
                                results_df=k_results,
                                plots_dir=plots_dir,
                                filename="vortex_k_diagnostics.png",
                            )
                            recommended = ph_vortex_utils.recommend_vortex_k(
                                results_df=k_results,
                                target_n_clusters=int(n_clusters_target),
                            )
                            recommended_params = {key: _to_python_scalar(value) for key, value in recommended.items()}
                            selected_k = int(recommended["k_neighbors"])
                            logger.info(f"Recommended Vortex K: {selected_k}")
                    else:
                        _log_vortex_step("3/5", "Skipping K scan", f"manual_k={selected_k}")
                        logger.info(f"Auto-K disabled. Using manual K={selected_k}")

                    with Timer("Vortex Clustering"):
                        _log_vortex_step("4/5", "Running final Vortex clustering", f"k={selected_k}")
                        logger.info("Step 4: Running Vortex clustering")
                        vortex_params = {
                            "k_neighbors": int(selected_k),
                            "subsample_size": int(subsample_size),
                            "subsample_method": subsample_method,
                            "random_seed": int(random_seed),
                            "knn_neighbors": int(knn_neighbors),
                            "min_cluster_size": int(min_cluster_size),
                            "generate_umap": bool(generate_umap),
                            "umap_neighbors": int(umap_neighbors),
                            "meta_clusters": int(meta_clusters),
                            "use_gpu": use_gpu,
                            "max_geosketch_input": int(max_geosketch_input),
                        }
                        cluster_assignments, umap_coords, vortex_diagnostics = ph_vortex_utils.intelligent_vortex_clustering(
                            df=norm_data,
                            norm_cols=norm_cols,
                            vortex_params=vortex_params,
                            logger=logger,
                        )

                        df_results = norm_data.copy()
                        df_results["vortex_cluster"] = cluster_assignments
                        df_results["vortex_low_expression"] = False
                        df_results["vortex_filter_status"] = "clustered"
                        if np.isfinite(umap_coords).any():
                            df_results["umap_x"] = umap_coords[:, 0]
                            df_results["umap_y"] = umap_coords[:, 1]

                        if low_expr_count:
                            low_expr_results = low_expr_data.copy()
                            low_expr_results["vortex_cluster"] = VORTEX_LOW_EXPR_CLUSTER_ID
                            low_expr_results["vortex_low_expression"] = True
                            low_expr_results["vortex_filter_status"] = "low_expression"
                            low_expr_results["umap_x"] = np.nan
                            low_expr_results["umap_y"] = np.nan
                            df_results = pd.concat([df_results, low_expr_results], axis=0)
                            logger.info(
                                "Preserved %s low-expression cells as %s with default phenotype '%s'.",
                                f"{low_expr_count:,}",
                                VORTEX_LOW_EXPR_LABEL,
                                VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE,
                            )

                        if len(df_nan) > 0:
                            nan_results = df_nan.copy()
                            nan_results["vortex_cluster"] = -1
                            nan_results["vortex_low_expression"] = False
                            nan_results["vortex_filter_status"] = "filtered_out"
                            if "umap_x" in df_results.columns:
                                nan_results["umap_x"] = np.nan
                            if "umap_y" in df_results.columns:
                                nan_results["umap_y"] = np.nan
                            for col in norm_cols:
                                nan_results[col] = np.nan
                            df_results = pd.concat([df_results, nan_results], axis=0)
                            logger.info(f"Appended {len(df_nan):,} NaN-filtered cells back with cluster -1.")

                        df_results = df_results.sort_index(kind="stable")

                    _log_vortex_step("5/5", "Generating pre-assignment QC dashboard")
                    qc_dashboard_path = ph_vortex_utils.create_vortex_qc_plots(
                        data=df_results,
                        plots_dir=plots_dir,
                        cluster_col="vortex_cluster",
                        phenotype_col=None,
                        max_plot_cells=int(max_plot_cells),
                        random_seed=int(random_seed),
                        filename="vortex_qc_dashboard.png",
                    )

                    clustered_df = df_results[~df_results["vortex_low_expression"]].copy()
                    cluster_sizes = clustered_df["vortex_cluster"].value_counts().sort_index()
                    n_clusters = int(clustered_df["vortex_cluster"].nunique())
                    logger.info(f"Clustering complete: {n_clusters} clusters identified")
                    logger.info("=" * 80)

                    if low_expr_count:
                        st.success(
                            f"Clustering completed. Generated **{n_clusters}** clusters and preserved **{low_expr_count:,}** low-expression cells as '{VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE}'."
                        )
                    else:
                        st.success(f"Clustering completed. Generated **{n_clusters}** clusters")
                    status.update(label="Clustering Complete!", state="complete", expanded=True)

                    temp_file = create_session_temp_parquet_path(project_root, "vortex_temp_res")
                    write_dataframe_parquet_atomic(df_results, temp_file)

                    st.session_state["vortex_temp_file"] = str(temp_file)
                    st.session_state["vortex_markers"] = markers_selected
                    st.session_state["vortex_norm_cols"] = norm_cols
                    st.session_state["vortex_n_clusters"] = n_clusters
                    initial_phenotype_map = {
                        _normalize_cluster_id(cluster_id): "Unassigned"
                        for cluster_id in sorted(cluster_sizes.index.tolist(), key=_cluster_sort_key)
                    }
                    if low_expr_count and not st.session_state.get('vortex_fresh_assignment_mode', False):
                        initial_phenotype_map[VORTEX_LOW_EXPR_CLUSTER_ID] = VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE
                    st.session_state["vortex_phenotype_map"] = initial_phenotype_map
                    st.session_state["vortex_has_run"] = True
                    st.session_state["vortex_run_input_path"] = full_input_path
                    st.session_state["vortex_run_id"] = f"{time.time():.6f}"
                    st.session_state["vortex_selected_k"] = int(selected_k)
                    st.session_state["vortex_k_results_path"] = str(k_results_path) if k_results_path is not None else None
                    st.session_state["vortex_k_plot_path"] = str(k_plot_path) if k_plot_path is not None else None
                    st.session_state["vortex_qc_dashboard_path"] = str(qc_dashboard_path)
                    st.session_state["vortex_qc_dashboard_pre_path"] = str(qc_dashboard_path)
                    st.session_state["vortex_qc_dashboard_post_path"] = None
                    st.session_state["vortex_loaded_source_path"] = full_input_path
                    st.session_state["vortex_k_recommendation"] = recommended_params
                    st.session_state["vortex_low_expr_cluster_id"] = VORTEX_LOW_EXPR_CLUSTER_ID if low_expr_count else None
                    st.session_state["vortex_low_expr_count"] = int(low_expr_count)
                    st.session_state["vortex_low_expr_metric"] = low_expr_metric if enable_low_expr_filter else None
                    st.session_state["vortex_low_expr_threshold"] = float(low_expr_threshold) if (enable_low_expr_filter and low_expr_threshold is not None) else None
                    st.session_state["vortex_low_expr_default_pheno"] = VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE
                    st.session_state["vortex_run_summary"] = {
                        "cells_clustered": int(len(clustered_df)),
                        "cells_total": int(len(df_results)),
                        "low_expression_cells": int(low_expr_count),
                        "n_clusters": int(n_clusters),
                        "runtime_seconds": float(vortex_diagnostics.get("runtime_seconds", 0.0)),
                        "min_cluster_size": int(cluster_sizes.min()),
                        "median_cluster_size": float(cluster_sizes.median()),
                        "max_cluster_size": int(cluster_sizes.max()),
                        "largest_cluster_fraction": float(cluster_sizes.max() / len(clustered_df)) if len(clustered_df) else 0.0,
                        "subsample_size": int(vortex_diagnostics.get("subsample_size", min(int(subsample_size), len(clustered_df)))),
                        "subsample_method": str(vortex_diagnostics.get("subsample_method", subsample_method)),
                        "umap_available": bool(np.isfinite(umap_coords).any()),
                    }
                    st.session_state["vortex_run_parameters"] = {
                        "auto_select_k": bool(auto_select_k),
                        "manual_k": int(vortex_k),
                        "selected_k": int(selected_k),
                        "target_clusters": int(n_clusters_target),
                        "k_candidates": k_values,
                        "k_scan_sketch_size": int(k_scan_sketch_size),
                        "subsample_size": int(subsample_size),
                        "subsample_method": str(subsample_method),
                        "knn_neighbors": int(knn_neighbors),
                        "min_cluster_size": int(min_cluster_size),
                        "generate_umap": bool(generate_umap),
                        "umap_neighbors": int(umap_neighbors),
                        "max_plot_cells": int(max_plot_cells),
                        "random_seed": int(random_seed),
                        "low_expression_metric": get_low_expression_metric_label(low_expr_metric),
                        "low_expression_threshold": float(low_expr_threshold) if (enable_low_expr_filter and low_expr_threshold is not None) else None,
                    }

                    finish_phenotyping_run_audit(
                        run_audit,
                        status="success",
                        output_path=temp_file,
                        input_rows=audit_input_rows,
                        output_rows=len(df_results),
                        output_column=output_col_name,
                        cluster_column="vortex_cluster",
                        phenotype_map=initial_phenotype_map,
                        extra=st.session_state["vortex_run_summary"],
                    )
                    run_audit = None

            except Exception as exc:
                import traceback

                finish_phenotyping_run_audit(
                    run_audit,
                    status="failed",
                    input_rows=audit_input_rows,
                    output_column=output_col_name,
                    cluster_column="vortex_cluster",
                    error=str(exc),
                )
                run_audit = None
                logger.error(f"Vortex pipeline error: {exc}")
                st.error(f"Error during execution: {exc}")
                st.code(traceback.format_exc())
                status.update(label="Vortex Pipeline Failed", state="error", expanded=True)
            finally:
                if run_audit is not None:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_col_name,
                        cluster_column="vortex_cluster",
                        error="Pipeline stopped before completion; see the Vortex terminal log.",
                    )
                logger.stop_streamlit_capture()

    vortex_temp_file = st.session_state.get("vortex_temp_file")
    has_vortex_results = bool(vortex_temp_file) and Path(vortex_temp_file).exists()
    df_results = pd.DataFrame()
    norm_cols: list[str] = []
    output_col = st.session_state.get("vortex_outcol", "phenotype_vortex")
    low_expr_cluster_id = st.session_state.get("vortex_low_expr_cluster_id")

    if has_vortex_results:
        st.divider()
        st.subheader("Interactive Visualization & Phenotype Assignment")
        st.caption("Professional phenotype assignment interface synchronized with the Vortex notebook workflow")

        temp_file = Path(str(vortex_temp_file))
        df_results = read_dataframe_parquet_safe(
            temp_file,
            session_state_key="vortex_temp_file",
            label="Vortex results",
        )
        if df_results is None:
            return
        norm_cols = st.session_state.get("vortex_norm_cols", [])
        if low_expr_cluster_id is None and "vortex_low_expression" in df_results.columns and df_results["vortex_low_expression"].fillna(False).any():
            low_expr_cluster_id = VORTEX_LOW_EXPR_CLUSTER_ID
        low_expr_mask = df_results.get("vortex_low_expression", pd.Series(False, index=df_results.index)).fillna(False).astype(bool)
        clustered_df = df_results[~low_expr_mask].copy()
        low_expr_count = int(low_expr_mask.sum())
        n_clusters = st.session_state.get("vortex_n_clusters", clustered_df["vortex_cluster"].nunique())

        fresh_assignment_reset = bool(st.session_state.pop("vortex_fresh_assignment_mode", False))
        if fresh_assignment_reset:
            st.session_state["vortex_phenotype_map"] = {
                _normalize_cluster_id(cluster_id): "Unassigned"
                for cluster_id in df_results["vortex_cluster"].unique()
                if pd.notna(cluster_id) and _normalize_cluster_id(cluster_id) is not None
            }
        elif "vortex_phenotype_map" not in st.session_state:
            cluster_ids = sorted(
                [_normalize_cluster_id(cluster_id) for cluster_id in df_results["vortex_cluster"].unique() if pd.notna(cluster_id)],
                key=_cluster_sort_key,
            )
            st.session_state["vortex_phenotype_map"] = {
                cluster_id: (
                    VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE
                    if _is_low_expression_cluster(cluster_id, low_expr_cluster_id)
                    else "Unassigned"
                )
                for cluster_id in cluster_ids
            }
        if "vortex_available_phenotypes" not in st.session_state:
            st.session_state["vortex_available_phenotypes"] = get_all_phenotype_options(include_unassigned=True)
        if low_expr_count and low_expr_cluster_id is not None and not fresh_assignment_reset:
            current_value = st.session_state["vortex_phenotype_map"].get(_normalize_cluster_id(low_expr_cluster_id))
            if current_value in {None, "Unassigned"}:
                st.session_state["vortex_phenotype_map"][_normalize_cluster_id(low_expr_cluster_id)] = (
                    st.session_state.get("vortex_low_expr_default_pheno", VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE)
                )
        st.session_state["vortex_phenotype_map"] = normalize_phenotype_map_keys(
            st.session_state.get("vortex_phenotype_map", {})
        )

        tab_heatmap, tab_umap, tab_kdiag, tab_assign, tab_export = st.tabs(
            [
                "Heatmap & Assignment",
                "UMAP & QC",
                "K Diagnostics",
                "Table Editor",
                "Export Results",
            ]
        )

        with tab_heatmap:
            st.markdown("### Interactive Cluster Expression Heatmap")
            st.caption("Phenotype assignment workflow aligned with the Leiden and SOM subtabs.")

            if "vortex_heatmap_config" not in st.session_state:
                st.session_state["vortex_heatmap_config"] = {
                    "cluster_x": True,
                    "cluster_y": True,
                    "show_counts": True,
                    "z_min": -4,
                    "z_max": 4,
                }

            hm_row1 = st.columns([2, 2, 2])
            with hm_row1[0]:
                if st.button("Toggle X-axis Clustering", help="Hierarchically cluster Vortex clusters (columns)", key="vortex_toggle_x"):
                    st.session_state["vortex_heatmap_config"]["cluster_x"] = not st.session_state["vortex_heatmap_config"]["cluster_x"]
                st.caption(f"X-axis clustering: {'ON' if st.session_state['vortex_heatmap_config']['cluster_x'] else 'OFF'}")
            with hm_row1[1]:
                if st.button("Toggle Y-axis Clustering", help="Hierarchically cluster markers (rows)", key="vortex_toggle_y"):
                    st.session_state["vortex_heatmap_config"]["cluster_y"] = not st.session_state["vortex_heatmap_config"]["cluster_y"]
                st.caption(f"Y-axis clustering: {'ON' if st.session_state['vortex_heatmap_config']['cluster_y'] else 'OFF'}")
            with hm_row1[2]:
                if st.button("Toggle Cell Counts", help="Show or hide cell counts in the heatmap labels", key="vortex_toggle_counts"):
                    st.session_state["vortex_heatmap_config"]["show_counts"] = not st.session_state["vortex_heatmap_config"]["show_counts"]
                st.caption(f"Cell counts: {'ON' if st.session_state['vortex_heatmap_config']['show_counts'] else 'OFF'}")

            with st.expander("Heatmap Configuration", expanded=False):
                current_min = st.session_state["vortex_heatmap_config"].get("z_min", -4)
                current_max = st.session_state["vortex_heatmap_config"].get("z_max", 4)
                z_range = st.slider(
                    "Color Scale Range (Z-Score)",
                    min_value=-10.0,
                    max_value=10.0,
                    value=(float(current_min), float(current_max)),
                    step=0.5,
                    key="vortex_z_range_slider",
                )
                if z_range[0] != current_min or z_range[1] != current_max:
                    st.session_state["vortex_heatmap_config"]["z_min"] = z_range[0]
                    st.session_state["vortex_heatmap_config"]["z_max"] = z_range[1]
                    st.rerun()

            cfg = st.session_state["vortex_heatmap_config"]
            if low_expr_count:
                low_expr_metric = st.session_state.get("vortex_low_expr_metric")
                threshold_text = st.session_state.get("vortex_low_expr_threshold")
                threshold_caption = f" using {_format_low_expression_rule(low_expr_metric, threshold_text)}" if threshold_text is not None else ""
                st.info(
                    f"{low_expr_count:,} low-expression cells were preserved outside the clustering heatmap{threshold_caption}. "
                    f"They default to phenotype '{st.session_state.get('vortex_low_expr_default_pheno', VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE)}' and remain available in the assignment and export tables."
                )

            if norm_cols and all(column in df_results.columns for column in norm_cols):
                data_hash = (
                    f"{st.session_state.get('vortex_run_id', 'no_run')}_{len(clustered_df)}_"
                    f"{clustered_df['vortex_cluster'].nunique()}_{str(sorted(norm_cols))}_{cfg['z_min']}_{cfg['z_max']}"
                )
                base_fig, cluster_order, marker_order, cell_counts_dict, heatmap_z_ordered = build_leiden_heatmap_figure(
                    data_hash,
                    clustered_df,
                    norm_cols,
                    cfg["cluster_x"],
                    cfg["cluster_y"],
                    cfg["show_counts"],
                    cluster_col="vortex_cluster",
                    cluster_label="Vortex Cluster",
                    z_min=cfg["z_min"],
                    z_max=cfg["z_max"],
                    marker_source_note=build_marker_source_note(norm_cols, method_key="vortex"),
                )

                if base_fig:
                    if "vortex_phenotype_map" not in st.session_state:
                        st.session_state["vortex_phenotype_map"] = {cluster_id: "Unassigned" for cluster_id in cluster_order}
                    if "vortex_available_phenotypes" not in st.session_state:
                        st.session_state["vortex_available_phenotypes"] = ["Unassigned"]
                    if "vortex_heatmap_component_reset_counter" not in st.session_state:
                        st.session_state["vortex_heatmap_component_reset_counter"] = 0

                    for cluster_id in cluster_order:
                        if cluster_id not in st.session_state["vortex_phenotype_map"]:
                            st.session_state["vortex_phenotype_map"][cluster_id] = "Unassigned"

                    phenotype_map = st.session_state["vortex_phenotype_map"]
                    available_phenotypes = get_all_phenotype_options(include_unassigned=True)
                    for phenotype in phenotype_map.values():
                        if phenotype != "Unassigned" and phenotype not in available_phenotypes:
                            available_phenotypes.append(phenotype)
                    st.session_state["vortex_available_phenotypes"] = available_phenotypes

                    fig = go.Figure(base_fig)
                    fig.update_layout(dragmode=False)
                    shapes = []
                    cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(cluster_order)}
                    for cluster_id in cluster_order:
                        idx = cluster_to_index.get(cluster_id)
                        if idx is None:
                            continue
                        if phenotype_map.get(cluster_id) != "Unassigned":
                            shapes.append(
                                dict(
                                    type="rect",
                                    x0=idx - 0.5,
                                    x1=idx + 0.5,
                                    y0=-0.5,
                                    y1=len(marker_order) - 0.5,
                                    line=dict(color="rgba(40, 167, 69, 0.9)", width=3),
                                    fillcolor="rgba(0,0,0,0)",
                                )
                            )
                    fig.update_layout(shapes=shapes)
                    fig.update_layout(
                        title_text=f"Vortex Clusters (Assigned: {sum(1 for value in phenotype_map.values() if value != 'Unassigned')})"
                    )
                    trace_customdata = np.tile(np.array(cluster_order, dtype=object), (len(marker_order), 1))
                    fig.update_traces(customdata=trace_customdata, selector={"type": "heatmap"})

                    st.markdown("#### Heatmap Click Selection")
                    st.caption(
                        "Click heatmap cells to buffer clusters locally, then apply one phenotype to all selected clusters at once. "
                        "The heatmap rendering and hover content stay unchanged."
                    )

                    cluster_display_labels = {
                        cluster_id: _cluster_display_label(cluster_id, low_expr_cluster_id)
                        for cluster_id in cluster_order
                    }
                    cluster_marker_scores = build_cluster_marker_score_map_from_heatmap(heatmap_z_ordered)

                    # Assisted assignment: score every cluster against user-written
                    # phenotype signatures and propose a name for each, so large cluster
                    # counts stay tractable without merging them away.
                    if render_assisted_assignment(
                        key_prefix="vortex",
                        phenotype_map_state_key="vortex_phenotype_map",
                        available_phenotypes_state_key="vortex_available_phenotypes",
                        cluster_marker_scores=cluster_marker_scores,
                        df_results=locals().get("df_results"),
                        marker_cols=locals().get("norm_cols") or [],
                        cluster_column="vortex_cluster",
                    ):
                        st.rerun()
                    if heatmap_click_selector_available():
                        component_event = render_heatmap_click_selector(
                            figure=fig,
                            available_phenotypes=available_phenotypes,
                            cluster_display_labels=cluster_display_labels,
                            cluster_marker_scores=cluster_marker_scores,
                            reset_counter=int(st.session_state.get("vortex_heatmap_component_reset_counter", 0)),
                            key="vortex_heatmap_click_selector",
                        )

                        event_is_new = consume_heatmap_assignment_event(
                            component_event,
                            scope=get_heatmap_assignment_scope(
                                "vortex", st.session_state.get("vortex_run_id", "no-run")
                            ),
                        )
                        assignment = (
                            apply_heatmap_assignment_event(
                                component_event,
                                st.session_state.get("vortex_phenotype_map", {}),
                                previous_event_id=st.session_state.get("vortex_heatmap_component_event_id"),
                            )
                            if event_is_new
                            else None
                        )
                        if assignment is not None:
                            (
                                updated_map,
                                selected_clusters,
                                selected_click_phenotype,
                                event_id,
                            ) = assignment
                            st.session_state["vortex_phenotype_map"] = updated_map
                            st.session_state["vortex_heatmap_component_event_id"] = event_id
                            st.session_state["vortex_heatmap_component_reset_counter"] = (
                                int(st.session_state.get("vortex_heatmap_component_reset_counter", 0)) + 1
                            )
                            st.success(
                                f"Assigned '{selected_click_phenotype}' to {len(selected_clusters)} cluster(s): "
                                + ", ".join(_cluster_display_label(cluster_id, low_expr_cluster_id) for cluster_id in selected_clusters)
                            )
                            st.rerun()
                    else:
                        st.plotly_chart(fig, key="vortex_heatmap_chart_fallback", width="stretch", config={"displayModeBar": False})
                        st.warning(
                            "Buffered heatmap click selection is unavailable because the local selector component has not been built yet. "
                            "The heatmap is shown in static mode until that frontend build exists."
                        )

                    st.divider()
                    st.markdown("#### Phenotype Summary Table & Assignment")
                    st.caption("Edit phenotypes below and click Sync to apply the changes to the heatmap and saved outputs.")
                    st.caption("Marker Z-score here means standardized marker expression across clusters. It is not a proportional score or cell-fraction score.")
                    st.caption("Need a phenotype not in the list? Add it here:")
                    render_add_custom_phenotype_ui(key_prefix="vortex")

                    summary_data = []
                    for cluster_id in cluster_order:
                        top5_str = ""
                        if not heatmap_z_ordered.empty and cluster_id in heatmap_z_ordered.columns:
                            top5_series = heatmap_z_ordered[cluster_id].nlargest(5)
                            top5_str = ", ".join([f"{marker} ({score:.2f})" for marker, score in top5_series.items()])
                        summary_data.append(
                            {
                                "Cluster": cluster_id,
                                "Group": "Vortex cluster",
                                "Cells": cell_counts_dict.get(cluster_id, 0),
                                "Top Markers (Marker Z-Score)": top5_str,
                                "Phenotype": phenotype_map.get(cluster_id, "Unassigned"),
                            }
                        )

                    if low_expr_count and low_expr_cluster_id is not None:
                        low_expr_metric = st.session_state.get("vortex_low_expr_metric")
                        low_expr_threshold = st.session_state.get("vortex_low_expr_threshold")
                        threshold_note = (
                            f"{_format_low_expression_rule(low_expr_metric, low_expr_threshold).capitalize()}; preserved outside clustering"
                            if low_expr_threshold is not None
                            else "Preserved outside clustering"
                        )
                        summary_data.append(
                            {
                                "Cluster": _normalize_cluster_id(low_expr_cluster_id),
                                "Group": VORTEX_LOW_EXPR_LABEL,
                                "Cells": low_expr_count,
                                "Top Markers (Marker Z-Score)": threshold_note,
                                "Phenotype": phenotype_map.get(
                                    _normalize_cluster_id(low_expr_cluster_id),
                                    st.session_state.get("vortex_low_expr_default_pheno", VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE),
                                ),
                            }
                        )

                    summary_df = pd.DataFrame(summary_data)
                    progress_col1, progress_col2 = st.columns([2, 1])
                    progress_col1.markdown("##### 1. Edit Cluster Phenotypes")
                    progress_col2.metric(
                        "Assigned",
                        f"{sum(1 for value in phenotype_map.values() if value != 'Unassigned')}/{len(phenotype_map)}",
                    )

                    with st.form("vortex_batch_assignment_form", clear_on_submit=False):
                        edited_df = st.data_editor(
                            summary_df,
                            column_config={
                                "Cluster": st.column_config.NumberColumn("Cluster", disabled=True),
                                "Group": st.column_config.TextColumn("Group", disabled=True, width="medium"),
                                "Cells": st.column_config.NumberColumn("Cells", disabled=True),
                                "Top Markers (Marker Z-Score)": st.column_config.TextColumn("Top Markers (Marker Z-Score)", disabled=True, width="large"),
                                "Phenotype": st.column_config.SelectboxColumn("Phenotype", options=available_phenotypes, required=True),
                            },
                            disabled=["Cluster", "Group", "Cells", "Top Markers (Marker Z-Score)"],
                            hide_index=True,
                            num_rows="fixed",
                            key="vortex_summary_editor_form",
                            width="stretch",
                        )
                        st.markdown("##### 2. Sync to Heatmap")
                        sync_col1, sync_col2 = st.columns([3, 1])
                        with sync_col1:
                            st.caption("All edits above are local until you click Sync & Update.")
                        with sync_col2:
                            submitted = st.form_submit_button("Sync & Update", type="primary")

                    if submitted:
                        st.session_state["vortex_phenotype_map"] = {
                            _normalize_cluster_id(row["Cluster"]): row["Phenotype"] for _, row in edited_df.iterrows()
                        }
                        st.success("Synced! Heatmap updated.")
                        st.rerun()

                    render_phenotype_statistics(
                        df_results=df_results,
                        cluster_col="vortex_cluster",
                        phenotype_map=st.session_state.get("vortex_phenotype_map", {}),
                        heatmap_z_ordered=heatmap_z_ordered,
                        cluster_label="Cluster",
                        score_label="Avg Marker Z-Score",
                    )

                    if low_expr_count:
                        st.caption(
                            "Low-expression bucket cells contribute to phenotype counts, but their marker ranking is excluded from the heatmap-based top-marker summary because they are not part of the Vortex clustering heatmap."
                        )

                    render_publication_heatmap(
                        heatmap_z_ordered=heatmap_z_ordered,
                        phenotype_map=st.session_state.get("vortex_phenotype_map", {}),
                        key_prefix="vortex",
                        df_results=df_results,
                        group_col="vortex_cluster",
                        marker_columns=norm_cols,
                    )

                    update_cols = ["vortex_cluster"]
                    for extra_col in ["vortex_low_expression", "vortex_filter_status", "vortex_low_expr_score", "vortex_marker_sum_raw"]:
                        if extra_col in df_results.columns:
                            update_cols.append(extra_col)
                    if "umap_x" in df_results.columns:
                        update_cols.append("umap_x")
                    if "umap_y" in df_results.columns:
                        update_cols.append("umap_y")

                    render_save_assignments(
                        df_results=df_results,
                        phenotype_map=st.session_state.get("vortex_phenotype_map", {}),
                        output_col=output_col,
                        project_root=project_root,
                        key_prefix="vortex",
                        cluster_col="vortex_cluster",
                        update_cluster_cols=update_cols,
                        cell_id_col=st.session_state.get("vortex_cell_id_col", "cell_id"),
                    )

                    render_export_split_data(
                        df_results=df_results,
                        output_col=output_col,
                        phenotype_map=st.session_state.get("vortex_phenotype_map", {}),
                        project_root=project_root,
                        key_prefix="vortex",
                        cluster_col="vortex_cluster",
                        cell_id_col=st.session_state.get("vortex_cell_id_col", "cell_id"),
                    )
                else:
                    st.warning("Could not generate the Vortex heatmap figure. Please rerun clustering.")
            else:
                st.warning("Normalized columns not found. Please rerun clustering.")

        with tab_umap:
            st.markdown("### UMAP Projection & QC")
            st.caption("Review Vortex cluster structure and quality-control figures before final phenotype export.")

            phenotype_map = st.session_state.get("vortex_phenotype_map", {})
            preview_df = _build_vortex_preview_df(df_results, output_col, phenotype_map)

            assigned_cells = int((preview_df[output_col] != "Unassigned").sum()) if output_col in preview_df.columns else 0
            assigned_clusters = int(sum(1 for value in phenotype_map.values() if value != "Unassigned"))

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("UMAP Available", "Yes" if {"umap_x", "umap_y"}.issubset(preview_df.columns) else "No")
            metric_col2.metric("Assigned Cells", f"{assigned_cells:,}")
            metric_col3.metric("Assigned Clusters", f"{assigned_clusters}/{len(phenotype_map)}")

            if low_expr_count:
                st.caption(
                    f"{low_expr_count:,} preserved low-expression cells stay in the results with the boolean flag 'vortex_low_expression'. "
                    "Cells without projected UMAP coordinates are skipped in the scatter view but remain available for export and cell-level preview."
                )

            control_col1, control_col2 = st.columns([2, 1])
            with control_col1:
                max_cells_plot_live = st.number_input(
                    "Max Cells to Plot",
                    1000,
                    1000000,
                    int(max_plot_cells),
                    1000,
                    help="Subsample for faster plotting in the live UMAP view.",
                    key="vortex_umap_max",
                )
            with control_col2:
                point_size = st.slider("Point Size", 0.1, 5.0, 0.5, 0.1, key="vortex_umap_size")

            if {"umap_x", "umap_y"}.issubset(preview_df.columns):
                if len(preview_df) > max_cells_plot_live:
                    plot_df = preview_df.sample(n=int(max_cells_plot_live), random_state=42)
                    st.caption(f"Showing {int(max_cells_plot_live):,} of {len(preview_df):,} cells")
                else:
                    plot_df = preview_df

                has_assigned_phenotypes = output_col in preview_df.columns and preview_df[output_col].astype(str).ne("Unassigned").any()
                phenotype_hash = None
                if phenotype_map:
                    phenotype_hash = hashlib.md5(
                        json.dumps(sorted((int(cluster_id), str(label)) for cluster_id, label in phenotype_map.items())).encode("utf-8")
                    ).hexdigest()

                if has_assigned_phenotypes and ph_vortex_utils is not None and st.session_state.get("vortex_post_qc_hash") != phenotype_hash:
                    logger.info("[Vortex] Generating post-assignment QC dashboard for the current phenotype map")
                    plots_dir = project_root / "outputs/phenotyping/vortex/plots"
                    post_qc_path = ph_vortex_utils.create_vortex_qc_plots(
                        data=preview_df,
                        plots_dir=plots_dir,
                        cluster_col="vortex_cluster",
                        phenotype_col=output_col,
                        max_plot_cells=int(max_cells_plot_live),
                        random_seed=int(random_seed),
                        filename="vortex_qc_dashboard_post_assignment.png",
                    )
                    st.session_state["vortex_qc_dashboard_post_path"] = str(post_qc_path)
                    st.session_state["vortex_post_qc_hash"] = phenotype_hash

                pre_qc_path = st.session_state.get("vortex_qc_dashboard_pre_path") or st.session_state.get("vortex_qc_dashboard_path")
                post_qc_path = st.session_state.get("vortex_qc_dashboard_post_path")

                if (pre_qc_path and Path(pre_qc_path).exists()) or (post_qc_path and Path(post_qc_path).exists()):
                    st.markdown("#### Saved QC Dashboards")
                    qc_col1, qc_col2 = st.columns(2)
                    with qc_col1:
                        if pre_qc_path and Path(pre_qc_path).exists():
                            st.image(str(pre_qc_path), caption="Before assignment: QC dashboard", width="stretch")
                    with qc_col2:
                        if post_qc_path and Path(post_qc_path).exists():
                            st.image(str(post_qc_path), caption="After assignment: QC dashboard", width="stretch")

                st.markdown("#### Before and After Phenotype Assignment")
                plot_df_umap = plot_df[np.isfinite(plot_df["umap_x"]) & np.isfinite(plot_df["umap_y"])].copy()
                omitted_no_umap = len(plot_df) - len(plot_df_umap)
                if omitted_no_umap:
                    st.caption(f"Omitted {omitted_no_umap:,} cells from the live UMAP scatter because they do not have finite UMAP coordinates.")
                fig = _build_umap_comparison_figure(
                    plot_df=plot_df_umap,
                    output_col=output_col,
                    n_clusters=n_clusters,
                    point_size=point_size,
                )
                st.pyplot(fig)
                plt.close(fig)

                qc_refresh_col1, qc_refresh_col2 = st.columns([1, 3])
                with qc_refresh_col1:
                    if st.button("Refresh QC Figure", key="vortex_refresh_qc_btn"):
                        if ph_vortex_utils is None:
                            st.error("ph_vortex_utils is not available, so the QC figure cannot be refreshed.")
                            st.stop()
                        plots_dir = project_root / "outputs/phenotyping/vortex/plots"
                        qc_filename = "vortex_qc_dashboard_post_assignment.png" if has_assigned_phenotypes else "vortex_qc_dashboard.png"
                        refreshed_path = ph_vortex_utils.create_vortex_qc_plots(
                            data=preview_df,
                            plots_dir=plots_dir,
                            cluster_col="vortex_cluster",
                            phenotype_col=output_col if has_assigned_phenotypes else None,
                            max_plot_cells=int(max_cells_plot_live),
                            random_seed=int(random_seed),
                            filename=qc_filename,
                        )
                        st.session_state["vortex_qc_dashboard_path"] = str(refreshed_path)
                        if has_assigned_phenotypes:
                            st.session_state["vortex_qc_dashboard_post_path"] = str(refreshed_path)
                            st.session_state["vortex_post_qc_hash"] = phenotype_hash
                        else:
                            st.session_state["vortex_qc_dashboard_pre_path"] = str(refreshed_path)
                        st.success("QC dashboard refreshed.")
                        st.rerun()
                with qc_refresh_col2:
                    st.caption("The left UMAP shows the raw Vortex clusters. The right UMAP updates to phenotype colors after assignment.")
            else:
                st.error("UMAP coordinates not found. Please rerun clustering.")

        with tab_kdiag:
            st.markdown("### K Selection Diagnostics")
            st.caption("These diagnostics are generated during auto-K scanning and used to choose the final Vortex K.")

            selected_k = st.session_state.get("vortex_selected_k")
            run_summary = st.session_state.get("vortex_run_summary", {})
            run_parameters = st.session_state.get("vortex_run_parameters", {})
            recommendation = st.session_state.get("vortex_k_recommendation")

            if run_summary or run_parameters:
                st.markdown("#### Notebook-Style Parameter Review")
                review_col1, review_col2, review_col3, review_col4 = st.columns(4)
                review_col1.metric("Selected K", int(run_parameters.get("selected_k", selected_k or 0)))
                review_col2.metric("Final Clusters", int(run_summary.get("n_clusters", 0)))
                review_col3.metric("Runtime (s)", f"{float(run_summary.get('runtime_seconds', 0.0)):.2f}")
                review_col4.metric(
                    "Largest Cluster",
                    f"{100 * float(run_summary.get('largest_cluster_fraction', 0.0)):.1f}%",
                )

                parameter_rows = [
                    {"Parameter": "Auto-select K", "Value": "Yes" if run_parameters.get("auto_select_k") else "No"},
                    {"Parameter": "Manual K", "Value": run_parameters.get("manual_k", "n/a")},
                    {"Parameter": "Target clusters", "Value": run_parameters.get("target_clusters", "n/a")},
                    {"Parameter": "K candidates", "Value": ", ".join([str(value) for value in run_parameters.get("k_candidates", [])])},
                    {"Parameter": "K-scan sketch size", "Value": run_parameters.get("k_scan_sketch_size", "n/a")},
                    {"Parameter": "Subsample size", "Value": run_summary.get("subsample_size", run_parameters.get("subsample_size", "n/a"))},
                    {"Parameter": "Subsample method", "Value": run_summary.get("subsample_method", run_parameters.get("subsample_method", "n/a"))},
                    {"Parameter": "KNN backfill neighbors", "Value": run_parameters.get("knn_neighbors", "n/a")},
                    {"Parameter": "Min cluster size", "Value": run_parameters.get("min_cluster_size", "n/a")},
                    {"Parameter": "Low-expression metric", "Value": run_parameters.get("low_expression_metric", "Disabled")},
                    {"Parameter": "Low-expression threshold", "Value": run_parameters.get("low_expression_threshold", "n/a")},
                    {"Parameter": "Low-expression cells", "Value": run_summary.get("low_expression_cells", 0)},
                    {"Parameter": "UMAP generated", "Value": "Yes" if run_summary.get("umap_available") else "No"},
                    {"Parameter": "Median cluster size", "Value": f"{float(run_summary.get('median_cluster_size', 0.0)):.1f}"},
                    {"Parameter": "Max cluster size", "Value": run_summary.get("max_cluster_size", "n/a")},
                ]
                parameter_df = pd.DataFrame(parameter_rows)
                parameter_df["Value"] = parameter_df["Value"].map(lambda value: str(value))
                st.dataframe(parameter_df, width="stretch", hide_index=True)

            if recommendation:
                st.markdown("#### Recommended Candidate From K Scan")
                rec_col1, rec_col2, rec_col3, rec_col4, rec_col5 = st.columns(5)
                rec_col1.metric("Recommended K", int(recommendation.get("k_neighbors", selected_k or 0)))
                rec_col2.metric("Predicted Clusters", recommendation.get("Num_Clusters") or "n/a")
                rec_col3.metric(
                    "Silhouette",
                    f"{float(recommendation['Silhouette_Score']):.4f}" if recommendation.get("Silhouette_Score") is not None else "n/a",
                )
                rec_col4.metric(
                    "DBI",
                    f"{float(recommendation['Davies_Bouldin_Index']):.4f}" if recommendation.get("Davies_Bouldin_Index") is not None else "n/a",
                )
                rec_col5.metric(
                    "Largest cluster %",
                    f"{100 * float(recommendation['Largest_Cluster_Fraction']):.1f}%" if recommendation.get("Largest_Cluster_Fraction") is not None else "n/a",
                )

                selection_stage = recommendation.get("Selection_Stage")
                if selection_stage:
                    st.caption(f"Selection stage: {str(selection_stage).replace('_', ' ')}")

                balance_cap = recommendation.get("Selection_Balance_Cap")
                if balance_cap is not None:
                    st.caption(
                        f"Recommendation excludes candidates with largest cluster > {100 * float(balance_cap):.0f}% when more balanced alternatives are available."
                    )

                target_band = recommendation.get("Selection_Target_Band")
                if target_band is not None and len(target_band) == 2:
                    st.caption(
                        f"Preferred target-cluster band during selection: {int(target_band[0])} to {int(target_band[1])} clusters."
                    )

                selection_warning = recommendation.get("Selection_Warning")
                if selection_warning:
                    st.warning(str(selection_warning))

                lock_col1, lock_col2 = st.columns([1, 3])
                with lock_col1:
                    if st.button("Use Recommended K", key="vortex_use_recommended_k_btn"):
                        st.session_state["vortex_k_value"] = int(recommendation["k_neighbors"])
                        st.session_state["vortex_auto_select_k"] = False
                        st.success("Recommended K copied to Manual K. Auto-select K has been turned off for the next rerun.")
                        st.rerun()
                with lock_col2:
                    st.caption("Copies the recommended K into the configuration panel so you can rerun with a fixed manual K.")

            k_results_path = st.session_state.get("vortex_k_results_path")
            k_plot_path = st.session_state.get("vortex_k_plot_path")

            if k_results_path and Path(k_results_path).exists():
                k_results_df = pd.read_csv(k_results_path)
                if recommendation and recommendation.get("k_neighbors") is not None:
                    recommended_k = int(recommendation["k_neighbors"])
                    k_results_df.insert(0, "Recommended", k_results_df["k_neighbors"].astype(int) == recommended_k)

                    inspect_options = [int(value) for value in k_results_df["k_neighbors"].tolist()]
                    default_idx = inspect_options.index(recommended_k) if recommended_k in inspect_options else 0
                    inspect_k = st.selectbox(
                        "Inspect candidate K",
                        inspect_options,
                        index=default_idx,
                        key="vortex_k_inspect_select",
                    )
                    inspect_row = k_results_df[k_results_df["k_neighbors"].astype(int) == int(inspect_k)].iloc[0]
                    inspect_col1, inspect_col2, inspect_col3, inspect_col4 = st.columns(4)
                    inspect_col1.metric("Candidate clusters", int(inspect_row["Num_Clusters"]))
                    inspect_col2.metric(
                        "Candidate silhouette",
                        f"{float(inspect_row['Silhouette_Score']):.4f}" if pd.notna(inspect_row["Silhouette_Score"]) else "n/a",
                    )
                    inspect_col3.metric(
                        "Candidate DBI",
                        f"{float(inspect_row['Davies_Bouldin_Index']):.4f}" if pd.notna(inspect_row["Davies_Bouldin_Index"]) else "n/a",
                    )
                    inspect_col4.metric(
                        "Largest cluster %",
                        f"{100 * float(inspect_row['Largest_Cluster_Fraction']):.1f}%" if pd.notna(inspect_row["Largest_Cluster_Fraction"]) else "n/a",
                    )
                st.dataframe(k_results_df, width="stretch", hide_index=True)
            else:
                st.info("No K-scan table is available for this run. Auto-K may have been disabled.")

            if k_plot_path and Path(k_plot_path).exists():
                st.image(str(k_plot_path), caption="Vortex K diagnostics", width="stretch")
            else:
                st.info("No K diagnostic plot is available for this run.")

        with tab_assign:
            st.markdown("### Phenotype Table Editor")
            st.caption("Full table view for editing Vortex assignments. Changes sync with the heatmap tab.")

            phenotype_map = st.session_state.get("vortex_phenotype_map", {})
            available_phenotypes = st.session_state.get("vortex_available_phenotypes", ["Unassigned"])
            cluster_stats = df_results.groupby("vortex_cluster").agg({"vortex_cluster": "count"}).rename(columns={"vortex_cluster": "n_cells"})
            cluster_stats["percentage"] = (cluster_stats["n_cells"] / len(df_results) * 100).round(2)
            clusters = sorted(
                [_normalize_cluster_id(cluster_id) for cluster_id in df_results["vortex_cluster"].unique() if pd.notna(cluster_id)],
                key=_cluster_sort_key,
            )

            sum_col1, sum_col2 = st.columns([1, 2])
            with sum_col1:
                st.metric("Progress", f"{sum(1 for value in phenotype_map.values() if value != 'Unassigned')}/{len(phenotype_map)} assigned")
            with sum_col2:
                st.caption("Defined phenotypes: " + ", ".join([phenotype for phenotype in available_phenotypes if phenotype != "Unassigned"]))

            st.divider()
            assign_rows = []
            for cluster_id in clusters:
                cluster_key = _normalize_cluster_id(cluster_id)
                assign_rows.append(
                    {
                        "Cluster": cluster_key,
                        "Group": VORTEX_LOW_EXPR_LABEL if _is_low_expression_cluster(cluster_key, low_expr_cluster_id) else "Vortex cluster",
                        "Cells": int(cluster_stats.loc[cluster_id, "n_cells"]) if cluster_id in cluster_stats.index else 0,
                        "%": float(cluster_stats.loc[cluster_id, "percentage"]) if cluster_id in cluster_stats.index else 0.0,
                        "Phenotype": phenotype_map.get(cluster_key, "Unassigned"),
                    }
                )
            assign_df = pd.DataFrame(assign_rows)

            if norm_cols:
                numeric_norm_cols = [column for column in norm_cols if column in df_results.columns and pd.api.types.is_numeric_dtype(df_results[column])]
                top_markers = []
                for cluster_id in clusters:
                    if _is_low_expression_cluster(cluster_id, low_expr_cluster_id):
                        low_expr_metric = st.session_state.get("vortex_low_expr_metric")
                        low_expr_threshold = st.session_state.get("vortex_low_expr_threshold")
                        threshold_label = _format_low_expression_rule(low_expr_metric, low_expr_threshold)
                        top_markers.append(threshold_label)
                        continue
                    if not numeric_norm_cols:
                        top_markers.append("")
                        continue
                    cluster_data = df_results[df_results["vortex_cluster"] == cluster_id][numeric_norm_cols].mean()
                    top3 = cluster_data.nlargest(3)
                    top_markers.append(
                        ", ".join([_clean_marker_name(marker) for marker in top3.index])
                    )
                assign_df["Top 3 Markers"] = top_markers

            edited_df = st.data_editor(
                assign_df,
                column_config={
                    "Cluster": st.column_config.NumberColumn(disabled=True),
                    "Group": st.column_config.TextColumn(disabled=True, width="medium"),
                    "Cells": st.column_config.NumberColumn(disabled=True, format="%d"),
                    "%": st.column_config.NumberColumn(disabled=True, format="%.1f"),
                    "Phenotype": st.column_config.SelectboxColumn("Phenotype", options=available_phenotypes, required=True),
                    "Top 3 Markers": st.column_config.TextColumn(disabled=True, width="medium"),
                },
                hide_index=True,
                num_rows="fixed",
                key="vortex_table_editor",
            )

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("Apply Changes", type="primary", key="vortex_apply_table"):
                    st.session_state["vortex_phenotype_map"] = {
                        _normalize_cluster_id(row["Cluster"]): row["Phenotype"] for _, row in edited_df.iterrows()
                    }
                    st.success("Assignments applied.")
                    st.rerun()
            with btn_col2:
                new_pheno = st.text_input("Add new phenotype", key="vortex_new_pheno_table", placeholder="Enter name")
                if new_pheno and new_pheno not in available_phenotypes:
                    if st.button("Add", key="vortex_add_pheno_btn"):
                        st.session_state["vortex_available_phenotypes"].append(new_pheno)
                        st.rerun()
            with btn_col3:
                if st.button("Reset All to Unassigned", key="vortex_reset_table"):
                    st.session_state["vortex_phenotype_map"] = {
                        _normalize_cluster_id(cluster_id): (
                            st.session_state.get("vortex_low_expr_default_pheno", VORTEX_LOW_EXPR_DEFAULT_PHENOTYPE)
                            if _is_low_expression_cluster(cluster_id, low_expr_cluster_id)
                            else "Unassigned"
                        )
                        for cluster_id in clusters
                    }
                    st.rerun()

        with tab_export:
            st.markdown("### Export Phenotyped Results")
            st.caption("Save final Vortex results with phenotype assignments.")

            phenotype_map = st.session_state.get("vortex_phenotype_map", {})
            df_export_preview = df_results.copy()
            df_export_preview[output_col] = df_export_preview["vortex_cluster"].map(phenotype_map).fillna("Unassigned")
            summary_stats = df_export_preview[output_col].value_counts()

            export_col1, export_col2 = st.columns([1, 1])
            with export_col1:
                st.markdown("#### Phenotype Distribution")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Phenotype": summary_stats.index,
                            "Cells": summary_stats.values,
                            "% of Total": (summary_stats.values / len(df_export_preview) * 100).round(2),
                        }
                    ).reset_index(drop=True),
                    hide_index=True,
                )
            with export_col2:
                st.markdown("#### Export Options")
                remove_norm = st.checkbox(
                    "Remove _norm columns",
                    value=True,
                    help="Remove normalized marker columns from the final CSV export.",
                    key="vortex_remove_norm",
                )
                output_filename = st.text_input(
                    "Output Filename",
                    "all_samples_combined_vortex_phenotyped.csv",
                    key="vortex_output_filename",
                )

            if low_expr_count:
                st.caption(
                    f"The exported table keeps {low_expr_count:,} preserved low-expression cells with 'vortex_low_expression', 'vortex_filter_status', 'vortex_low_expr_score', and 'vortex_marker_sum_raw' so you can screen them later."
                )

            if st.button("Export Final Results", type="primary", key="vortex_export_final"):
                with st.spinner("Saving results..."):
                    df_export = df_export_preview.copy()
                    if remove_norm:
                        cols_to_drop = [column for column in df_export.columns if column.endswith("_norm")]
                        df_export = df_export.drop(columns=cols_to_drop)
                        st.caption(f"Removed {len(cols_to_drop)} normalized columns")

                    final_out_dir = project_root / "outputs/phenotyping"
                    final_out_dir.mkdir(parents=True, exist_ok=True)
                    final_path = final_out_dir / output_filename
                    df_export.to_csv(final_path, index=False)
                    st.success("Results exported successfully.")
                    st.code(f"Output file: {final_path}", language=None)
                    st.metric("Total Cells", len(df_export))
                    st.metric("Unique Phenotypes", df_export[output_col].nunique())

    if st.session_state.get("viz_enabled"):
        st.divider()
        if has_vortex_results and not df_results.empty:
            _render_vortex_visual_validation(
                project_root=project_root,
                df_results=df_results,
                norm_cols=norm_cols,
                phenotype_map=st.session_state.get("vortex_phenotype_map", {}),
                output_col=output_col,
                low_expr_cluster_id=low_expr_cluster_id,
            )
        else:
            st.info("Run Vortex clustering or load existing Vortex results before opening direct cell preview.")
