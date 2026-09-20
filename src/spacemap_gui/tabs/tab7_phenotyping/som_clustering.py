"""
SOM Clustering Subtab
SOM clustering with notebook-synchronized parameters from Step6B.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import hashlib
import traceback
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Import shared utilities
from .common import (
    create_session_temp_parquet_path,
    logger,
    Timer,
    LOW_EXPRESSION_METRIC_OPTIONS,
    build_cluster_marker_score_map,
    build_cluster_marker_score_map_from_heatmap,
    render_assisted_assignment,
    build_marker_source_note,
    cluster_sort_key,
    compute_low_expression_score,
    ensure_cell_id_anchor,
    get_batch_id_column_help_text,
    get_cell_id_column_help_text,
    get_cluster_output_column_help_text,
    get_enable_row_filter_help_text,
    get_filter_column_help_text,
    get_filter_value_help_text,
    get_marker_list,
    get_low_expression_default_threshold,
    get_low_expression_filter_help_text,
    get_low_expression_metric_help_text,
    get_low_expression_metric_label,
    initialize_optional_column_widget_state,
    get_marker_selection_help_text,
    get_marker_workspace_panel,
    render_method_marker_selection,
    get_normalization_flag_help_text,
    _extract_markers_from_notebook,
    normalize_cluster_id,
    normalize_phenotype_map_keys,
    normalize_optional_column_name,
    get_output_column_help_text,
    plot_heatmap_figure,
    get_random_seed_help_text,
    save_plot,
    project_root_calc,
    build_leiden_heatmap_figure,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_export_split_data,
    render_data_filter_ui,
    apply_data_filter,
    apply_heatmap_assignment_event,
    consume_heatmap_assignment_event,
    get_heatmap_assignment_scope,
    read_dataframe_parquet_safe,
    render_settings_export_import,
    render_phenotyping_run_log_controls,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    resolve_notebook_marker_defaults,
    STANDARD_PHENOTYPES,
    get_all_phenotype_options,
    render_add_custom_phenotype_ui,
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
    from spacemap import ph_som_utils
    sys.modules["ph_som_utils"] = ph_som_utils
except ImportError:
    try:
        import ph_som_utils
    except Exception:
        ph_som_utils = None

try:
    from spacemap import ph_leiden_utils
    sys.modules["ph_leiden_utils"] = ph_leiden_utils
except ImportError:
    try:
        import ph_leiden_utils
    except Exception:
        ph_leiden_utils = None


SOM_LOW_EXPR_CLUSTER_ID = -1
SOM_LOW_EXPR_DEFAULT_PHENOTYPE = "Others"
SOM_LOW_EXPR_LABEL = "Low expression bucket"


def _is_som_low_expression_cluster(cluster_id, low_expr_cluster_id) -> bool:
    if low_expr_cluster_id is None:
        return False
    return normalize_cluster_id(cluster_id) == normalize_cluster_id(low_expr_cluster_id)


def _get_som_low_expression_mask(df_results: pd.DataFrame) -> pd.Series:
    if 'som_low_expression' in df_results.columns:
        return df_results['som_low_expression'].fillna(False).astype(bool)
    if 'meta_cluster' not in df_results.columns:
        return pd.Series(False, index=df_results.index)
    return df_results['meta_cluster'].apply(
        lambda value: _is_som_low_expression_cluster(value, SOM_LOW_EXPR_CLUSTER_ID)
    )


def _get_som_low_expression_note(metric: str | None, threshold: float | None) -> str:
    if threshold is None:
        return "Preserved outside clustering"
    return (
        f"{get_low_expression_metric_label(metric or 'mean_zscore').lower()} <= {float(threshold):.4g}; "
        "preserved outside clustering"
    )


def _coerce_som_low_expression_threshold(
    value,
    metric: str | None,
    fallback=None,
) -> float:
    """Return a numeric low-expression threshold, repairing stale imported None values."""
    metric_key = metric if metric in LOW_EXPRESSION_METRIC_OPTIONS else 'mean_zscore'

    for candidate in (value, fallback, get_low_expression_default_threshold(metric_key), 0.0):
        try:
            if candidate is None or pd.isna(candidate):
                continue
        except Exception:
            if candidate is None:
                continue

        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue

    return 0.0

def render_som(project_root, global_input_path):
    """
    SOM Clustering subtab - mirrors Leiden approach with SOM-specific parameters.
    Synchronized with Step6B_phenotype_SOM.ipynb notebook.
    """
    st.subheader("SOM Clustering Phenotyping")
    render_phenotyping_run_log_controls(Path(project_root), "som", "SOM Clustering")
    st.markdown("Self-Organizing Maps (SOM) for reproducible, notebook-aligned phenotyping and phenotype assignment.")
    st.caption(
        "Step6B defaults: batch-wise Z1 normalization, mean marker z-score filtering at -0.75, "
        "automatic 40–70 SOM grid, balanced training, and 50 final meta-clusters."
    )

    # Cache directory
    _som_cache_dir = project_root / 'outputs/phenotyping/cache'
    _som_cache_dir.mkdir(parents=True, exist_ok=True)
    som_cache = _som_cache_dir / 'som_params.json'
    
    if som_cache.exists():
        try:
            with open(som_cache, 'r') as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}
    else:
        cached_params = {}

    # --- Settings Export / Import ---
    render_settings_export_import(
        "SOM Clustering",
        ["som_"],
        {"som_params.json": som_cache},
        key_suffix="som",
    )

    def _apply_norm_for_heatmap(df_in: pd.DataFrame, markers: list[str]) -> tuple[pd.DataFrame, list[str]]:
        # Use batch-wise normalization if utils available
        valid_markers = [m for m in markers if m in df_in.columns]
        
        # Determine active cofactor
        cofactor = 5.0
        if 'bc_config' in st.session_state:
            cofactor = float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0))
            
        use_sinh = st.session_state.get('som_sinh', True)
        df_input = df_in
        if use_sinh:
            df_input = df_in.copy()
            for m in valid_markers:
                if m.endswith('_corr'):
                    df_input[m] = np.sinh(df_input[m]) * cofactor
        
        # Use ph_leiden_utils (contains normalization logic shared by SOM notebook)
        if ph_leiden_utils:
            group_cols = [batch_id_col] if batch_id_col in df_input.columns else []
            
            # Use shared utility for consistency
            norm_subset = ph_leiden_utils.apply_flexible_normalization(
                df=df_input,
                marker_cols=valid_markers,
                group_cols=group_cols,
                suffix='_norm',
                apply_z1=norm_z1,
                apply_z2=norm_z2,
                apply_cdf=norm_cdf,
                apply_nlog=norm_nlog
            )
            norm_data = df_in.copy()
            norm_cols_local = list(norm_subset.columns)
            norm_data = pd.concat([norm_data, norm_subset], axis=1)
            norm_data[norm_cols_local] = norm_data[norm_cols_local].fillna(0)
            return norm_data, norm_cols_local

        norm_cols_local: list[str] = []
        norm_data = df_in.copy()

        for m in valid_markers:
            if m not in df_input.columns:
                continue
            col_data = df_input[m].fillna(0).copy()

            if norm_nlog:
                col_data = np.log1p(col_data)
            if norm_cdf:
                col_data = col_data.rank(pct=True)
            if norm_z1:
                std_val = col_data.std()
                if std_val > 1e-9:
                    col_data = (col_data - col_data.mean()) / std_val
                else:
                    col_data = col_data - col_data.mean()
            if norm_z2:
                logger.warning("Z2 (per-cell normalization) is enabled - verify this is intended")

            col_data = col_data.fillna(0)
            norm_col = f"{m}_norm"
            norm_data[norm_col] = col_data
            norm_cols_local.append(norm_col)

        if norm_cols_local:
            norm_data[norm_cols_local] = norm_data[norm_cols_local].fillna(0)
        return norm_data, norm_cols_local
    
    # --- Input File Selection ---
    st.markdown("##### Input File Selection")
    use_global = st.checkbox("Use Global Input File", value=cached_params.get("use_global", True), key="som_use_global")
    full_input_path = global_input_path
    
    if not use_global:
        default_dir = cached_params.get("input_dir", str(project_root / 'outputs/batch_corrected'))
        input_dir = st.text_input("Input Directory", default_dir, key="som_input_dir")
        if input_dir and os.path.exists(input_dir):
            files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
        else:
            files = []
        if files:
            cached_file = cached_params.get("input_file", files[0])
            default_idx = files.index(cached_file) if cached_file in files else 0
            sel = st.selectbox("Input File", files, index=default_idx, key="som_infile_ov")
            full_input_path = os.path.join(input_dir, sel)
        else:
            st.warning(f"No CSV files found in: {input_dir}")
            full_input_path = None
    
    if full_input_path:
        st.caption(f"Active File: {os.path.basename(full_input_path)}")

    # --- Marker Selection (Identical to Leiden) ---
    st.markdown("##### Marker Selection")
    st.caption("Select markers for SOM clustering. Auto-selected from notebook configuration.")
    
    # Read available markers from input file
    all_marker_candidates = []
    if full_input_path and os.path.exists(full_input_path):
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            all_marker_candidates = get_marker_list(df_head)
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    available_marker_columns = list(all_marker_candidates)
    all_marker_candidates, marker_workspace_note = get_marker_workspace_panel(
        all_marker_candidates,
        full_input_path,
    )
    
    # AUTO-SELECT MARKERS FROM NOTEBOOK (Step6B_phenotype_SOM.ipynb)
    if full_input_path and os.path.exists(full_input_path):
        notebook_corr_markers, notebook_non_corr_markers = _extract_markers_from_notebook("Step6B_phenotype_SOM.ipynb")
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
        if 'som_confirmed_markers' not in st.session_state:
            st.session_state['som_confirmed_markers'] = []

    _, markers_selected, marker_workspace_note = render_method_marker_selection(
        key_prefix="som",
        method_name="SOM",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers if full_input_path else [],
        saved_markers=cached_params.get("markers", []),
    )
    
    if not markers_selected:
        st.warning("⚠️ No markers selected. Please select markers above before running.")

    # --- Configuration Panel ---
    if 'som_config' not in st.session_state:
        st.session_state['som_config'] = {
            'auto_grid': cached_params.get('auto_grid', False),
            'grid_dim': cached_params.get('grid_dim', 30),
            'target_cells_per_node': cached_params.get('target_cells_per_node', 1800),
            'min_grid': cached_params.get('min_grid', 20),
            'max_grid': cached_params.get('max_grid', 70),
            'mode': cached_params.get('mode', 'high_quality'),
            'subsample_size': cached_params.get('subsample_size', 150000),
            'subsample_method': cached_params.get('subsample_method', 'geosketch'),
            'early_stop': cached_params.get('early_stop', True),
            'early_stop_min_improvement': cached_params.get('early_stop_min_improvement', 0.01),
            'early_stop_min_epoch_fraction': cached_params.get('early_stop_min_epoch_fraction', 0.3),
            'random_seed': cached_params.get('random_seed', 42),
            'n_meta_target': cached_params.get('n_meta_target', 45),
            'meta_method': cached_params.get('meta_method', 'average'),
            'meta_metric': cached_params.get('meta_metric', 'correlation'),
            'norm_z1': cached_params.get('norm_z1', True),
            'norm_z2': cached_params.get('norm_z2', False),
            'norm_cdf': cached_params.get('norm_cdf', False),
            'norm_nlog': cached_params.get('norm_nlog', False),
            'norm_sinh': cached_params.get('norm_sinh', True),
            'output_col': cached_params.get('output_col', 'phenotype_som'),
            'cluster_col': cached_params.get('cluster_col', 'som_cluster'),
            # Step6B normalizes marker values within batch_id when that column
            # is available, and applies the mean-z-score low-expression filter.
            'batch_id_col': cached_params.get('batch_id_col') or 'batch_id',
            'enable_low_expr_filter': cached_params.get('enable_low_expr_filter', True),
            'low_expr_metric': cached_params.get('low_expr_metric', 'mean_zscore'),
            'low_expr_threshold': cached_params.get('low_expr_threshold', -0.75),
            'min_marker_sum': cached_params.get('min_marker_sum', -0.75),
            'cell_id_col': cached_params.get('cell_id_col', 'cell_id'),
            'enable_filter': cached_params.get('enable_filter'),
            'filter_col': cached_params.get('filter_col', 'tissue'),
            'filter_val': cached_params.get('filter_val', '1'),
            'markers': cached_params.get('markers', []),
        }

    with st.expander("Configuration & Parameters", expanded=True):
        with st.form("som_config_form", clear_on_submit=False):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Column Configuration")

                available_columns = []
                if full_input_path and os.path.exists(full_input_path):
                    try:
                        df_head = pd.read_csv(full_input_path, nrows=1)
                        available_columns = list(df_head.columns)
                    except: pass

                # Cell ID Selection
                default_cell_id = 'cell_id' if 'cell_id' in available_columns else (available_columns[0] if available_columns else 'cell_id')
                cell_id_options = [""] + available_columns if available_columns else [""]
                initialize_optional_column_widget_state(
                    "som_cell_id_col",
                    available_columns,
                    st.session_state['som_config'].get('cell_id_col', default_cell_id),
                )
                cell_id_col = st.selectbox(
                    "Cell ID Column",
                    cell_id_options,
                    index=cell_id_options.index(st.session_state.get("som_cell_id_col", default_cell_id)) if st.session_state.get("som_cell_id_col", default_cell_id) in cell_id_options else 0,
                    format_func=lambda value: value or "(None)",
                    help=get_cell_id_column_help_text(),
                    key="som_cell_id_col"
                )

                # Batch ID Column Selection
                batch_id_options = [""] + available_columns if available_columns else [""]
                initialize_optional_column_widget_state(
                    "som_batch_id_col",
                    available_columns,
                    st.session_state['som_config'].get('batch_id_col', ''),
                )
                idx_batch = batch_id_options.index(st.session_state.get('som_batch_id_col', '')) if st.session_state.get('som_batch_id_col', '') in batch_id_options else 0
                batch_id_col = st.selectbox(
                    "Batch ID Column (Normalization)",
                    batch_id_options,
                    index=idx_batch, 
                    format_func=lambda value: value or "(None)",
                    help=get_batch_id_column_help_text(),
                    key="som_batch_id_col"
                )

                output_col = st.text_input(
                    "Output Column Name",
                    st.session_state['som_config']['output_col'], 
                    key="som_outcol",
                    help=get_output_column_help_text()
                )
                cluster_col = st.text_input(
                    "Cluster Column Name",
                    st.session_state['som_config']['cluster_col'], 
                    key="som_cluster_col",
                    help=get_cluster_output_column_help_text("SOM node assignments")
                )

                st.markdown("**Filtering Criteria**")
                configured_enable_filter = st.session_state['som_config'].get('enable_filter')
                enable_filter = st.checkbox(
                    "Enable Row Filtering",
                    value=(
                        'corrected' in available_columns
                        if configured_enable_filter is None
                        else bool(configured_enable_filter)
                    ),
                    help=get_enable_row_filter_help_text(),
                    key="som_enable_filter",
                )
                filter_cols = [c for c in available_columns if c not in [cell_id_col]]
                default_filter_col = (
                    'corrected' if 'corrected' in filter_cols
                    else 'tissue' if 'tissue' in filter_cols
                    else (filter_cols[0] if filter_cols else 'tissue')
                )
                cached_filter_col = st.session_state['som_config'].get('filter_col') or default_filter_col
                default_filter_idx = filter_cols.index(cached_filter_col) if cached_filter_col in filter_cols else (filter_cols.index(default_filter_col) if default_filter_col in filter_cols else 0)

                filter_col = st.selectbox(
                    "Filter Column",
                    filter_cols if filter_cols else [default_filter_col],
                    index=default_filter_idx,
                    help=get_filter_column_help_text(),
                    key="som_filter_col",
                    disabled=not enable_filter
                )
                filter_val = st.text_input(
                    "Filter Value",
                    st.session_state['som_config'].get('filter_val', "1"),
                    help=get_filter_value_help_text(),
                    key="som_filter_val",
                    disabled=not enable_filter
                )

                st.markdown("##### SOM Clustering Parameters")
                auto_grid = st.checkbox(
                    "Auto Grid Size",
                    value=st.session_state['som_config']['auto_grid'],
                    key="som_auto_grid",
                    help=(
                        "If enabled, the code chooses a square SOM grid from the training subset size and the Target Cells/Node setting, then clamps the side length between Min Grid and Max Grid. Use this when dataset size changes between runs."
                    )
                )

                grid_dim_val = st.number_input(
                    "Manual Grid Size (N×N)", 30, 70, st.session_state['som_config']['grid_dim'],
                    help=(
                        "Fixed SOM grid side length used when Auto Grid Size is off. A larger grid creates more SOM nodes before meta-clustering."
                    ),
                    key="som_grid_manual",
                    disabled=auto_grid
                )
                target_cells_per_node_val = st.number_input(
                    "Target Cells/Node", 500, 5000, st.session_state['som_config']['target_cells_per_node'], 100,
                    help=(
                        "Used only when Auto Grid Size is on. The grid is chosen so each SOM node represents roughly this many training cells on average."
                    ),
                    key="som_target_cells",
                    disabled=not auto_grid
                )
                min_grid_val = st.number_input(
                    "Minimum Grid Side",
                    10,
                    50,
                    st.session_state['som_config']['min_grid'],
                    help=(
                        "Lower bound on the auto-selected grid side length."
                    ),
                    key="som_min_grid",
                    disabled=not auto_grid
                )
                max_grid_val = st.number_input(
                    "Maximum Grid Side",
                    40,
                    100,
                    st.session_state['som_config']['max_grid'],
                    help=(
                        "Upper bound on the auto-selected grid side length."
                    ),
                    key="som_max_grid",
                    disabled=not auto_grid
                )
                grid_dim = grid_dim_val if not auto_grid else st.session_state['som_config']['grid_dim']
                target_cells_per_node = target_cells_per_node_val if auto_grid else st.session_state['som_config']['target_cells_per_node']
                min_grid = min_grid_val if auto_grid else st.session_state['som_config']['min_grid']
                max_grid = max_grid_val if auto_grid else st.session_state['som_config']['max_grid']

                mode = st.selectbox(
                    "Training Mode",
                    ["fast", "balanced", "high_quality"],
                    index=["fast", "balanced", "high_quality"].index(st.session_state['som_config']['mode']),
                    help=(
                        "Controls the SOM training schedule."
                    ),
                    key="som_mode"
                )

            with col2:
                st.markdown("**Performance Options**")
                subsample_size = st.number_input(
                    "Subsample Size", 50000, 5000000, st.session_state['som_config']['subsample_size'], 10000,
                    help=(
                        "Maximum number of cells used to train the SOM itself."
                    ),
                    key="som_subsample"
                )

                subsample_method = st.selectbox(
                    "Subsampling Method", ["random", "geosketch"],
                    index=["random", "geosketch"].index(st.session_state['som_config'].get('subsample_method', 'random')),
                    help=(
                        "How the training subset is chosen when the dataset is larger than the subsample size. "
                        "'Geosketch' covers expression space evenly, so rare populations enter the training set "
                        "at well above their background rate; 'random' reproduces the true density. Geometric "
                        "sketching now uses a vectorized implementation and handles millions of cells in seconds, "
                        "so it is no longer the slow option."
                    ),
                    key="som_subsample_method"
                )

                early_stop = st.checkbox(
                    "Enable Early Stopping",
                    value=st.session_state['som_config'].get('early_stop', True),
                    help=(
                        "Stops the training loop when recent improvement in normalized quantization error plateaus."
                    ),
                    key="som_early_stop"
                )
                early_stop_min_improvement = st.number_input(
                    "Early Stop Min Improvement",
                    0.001, 0.05, float(st.session_state['som_config'].get('early_stop_min_improvement', 0.01)), 0.001,
                    help=(
                        "Plateau threshold for early stopping."
                    ),
                    key="som_early_stop_min_improvement",
                    disabled=not early_stop
                )
                early_stop_min_epoch_fraction = st.number_input(
                    "Early Stop Earliest Fraction",
                    0.1, 0.9, float(st.session_state['som_config'].get('early_stop_min_epoch_fraction', 0.3)), 0.05,
                    help=(
                        "Earliest point in the planned training schedule where early stopping is allowed."
                    ),
                    key="som_early_stop_min_epoch_fraction",
                    disabled=not early_stop
                )

                random_seed = st.number_input(
                    "Random Seed",
                    0,
                    9999,
                    st.session_state['som_config']['random_seed'],
                    help=get_random_seed_help_text("training-subset selection, PCA-based initialization, SOM weight initialization, and any randomized projections"),
                    key="som_seed"
                )

                st.markdown("**Meta-Clustering (Final Phenotype Groups)**")
                n_meta_target = st.number_input(
                    "Target Meta-Clusters", 2, 150, st.session_state['som_config']['n_meta_target'],
                    help=(
                        "Primary control for the final phenotype count. SOM nodes are first trained at higher resolution, then hierarchically merged toward this target. The realized number can be lower when the data contain fewer distinct node profiles."
                    ),
                    key="som_meta_target"
                )
                st.caption("Lower this value to merge redundant phenotypes; raise it only when biologically distinct populations are being combined.")
                meta_method = st.selectbox(
                    "Linkage Method", ["average", "ward", "complete"],
                    index=["average", "ward", "complete"].index(st.session_state['som_config'].get('meta_method', 'ward')),
                    help=(
                        "How SOM node summaries are merged into meta-clusters."
                    ), key="som_meta_method"
                )
                meta_metric = st.selectbox(
                    "Distance Metric", ["correlation", "euclidean", "cosine"],
                    index=["correlation", "euclidean", "cosine"].index(st.session_state['som_config'].get('meta_metric', 'euclidean' if st.session_state['som_config'].get('meta_method') == 'ward' else 'correlation')),
                    help=(
                        "Distance rule used during hierarchical meta-clustering when the chosen linkage method supports it."
                    ), key="som_meta_metric"
                )

                st.markdown("**Normalization Flags**")
                st.caption("Match notebook normalization pipeline")
                n1, n2, n3, n4 = st.columns(4)
                norm_z1 = n1.checkbox("Z1", value=st.session_state['som_config']['norm_z1'], key="som_norm_z1", help=get_normalization_flag_help_text("z1"))
                norm_z2 = n2.checkbox("Z2", value=st.session_state['som_config']['norm_z2'], key="som_norm_z2", help=get_normalization_flag_help_text("z2"))
                norm_cdf = n3.checkbox("CDF", value=st.session_state['som_config']['norm_cdf'], key="som_norm_cdf", help=get_normalization_flag_help_text("cdf"))
                norm_nlog = n4.checkbox("NLog", value=st.session_state['som_config']['norm_nlog'], key="som_norm_nlog", help=get_normalization_flag_help_text("nlog"))
                
                norm_sinh = st.checkbox(
                    "Inverse ArcSinh (Sinh) for Corrected Markers", 
                    value=st.session_state['som_config'].get('norm_sinh', True), 
                    help="Automatically convert batch-corrected (_corr) markers back to raw scale using sinh(val) * cofactor before normalization to maintain high-contrast 'ice heatmap' visualization.",
                    key="som_sinh"
                )

                st.markdown("**Output Configuration**")
                # Low Expression Filtering
                enable_low_expr_filter = st.checkbox(
                    "Filter Low-Expression Cells",
                    value=st.session_state['som_config'].get('enable_low_expr_filter', True),
                    help=get_low_expression_filter_help_text("SOM training"),
                    key="som_low_expr_filter"
                )

                low_expr_metric = st.session_state['som_config'].get('low_expr_metric', 'mean_zscore')
                if low_expr_metric not in LOW_EXPRESSION_METRIC_OPTIONS:
                    low_expr_metric = 'mean_zscore'

                low_expr_metric_selected = st.selectbox(
                    "Low-Expression Metric",
                    options=list(LOW_EXPRESSION_METRIC_OPTIONS),
                    index=list(LOW_EXPRESSION_METRIC_OPTIONS).index(low_expr_metric),
                    format_func=get_low_expression_metric_label,
                    help=get_low_expression_metric_help_text(),
                    key="som_low_expr_metric",
                    disabled=not enable_low_expr_filter,
                )
                low_expr_threshold_widget = st.number_input(
                    "Low-Expression Threshold",
                    -10.0,
                    100000.0,
                    _coerce_som_low_expression_threshold(
                        st.session_state['som_config'].get('low_expr_threshold'),
                        low_expr_metric,
                        fallback=st.session_state['som_config'].get(
                            'min_marker_sum',
                            get_low_expression_default_threshold(low_expr_metric)
                        ),
                    ),
                    0.05 if low_expr_metric == 'mean_zscore' else 0.1,
                    help=(
                        "Cells at or below this cutoff are kept in the output but excluded from SOM training and placed in the preserved low-expression bucket. For mean marker z-score, start around -0.75."
                        if low_expr_metric == 'mean_zscore'
                        else "Cells at or below this raw selected-marker sum are kept in the output but excluded from SOM training and placed in the preserved low-expression bucket."
                    ),
                    key="som_low_expr_threshold",
                    disabled=not enable_low_expr_filter,
                )
                if low_expr_metric_selected == 'mean_zscore':
                    st.caption("Recommended starting cutoff: -0.75. Lower values are more permissive; values closer to 0 remove more panel-dim cells.")
                else:
                    st.caption("Raw-sum mode uses the pre-normalization selected markers. A cutoff of 0.1 is usually only near-zero signal.")

                low_expr_threshold = low_expr_threshold_widget if enable_low_expr_filter else get_low_expression_default_threshold(low_expr_metric_selected)
                low_expr_metric = low_expr_metric_selected

            # Save Parameters Button
            save_col1, save_col2 = st.columns([1, 3])
            with save_col1:
                if st.form_submit_button("Save Parameters", type="primary", help="Save current parameters to cache", key="som_save_params_btn"):
                    st.session_state['som_config'] = {
                        'auto_grid': auto_grid,
                        'grid_dim': grid_dim,
                        'target_cells_per_node': target_cells_per_node,
                        'min_grid': min_grid,
                        'max_grid': max_grid,
                        'mode': mode,
                        'subsample_size': subsample_size,
                        'subsample_method': subsample_method,
                        'early_stop': early_stop,
                        'early_stop_min_improvement': early_stop_min_improvement,
                        'early_stop_min_epoch_fraction': early_stop_min_epoch_fraction,
                        'random_seed': random_seed,
                        'n_meta_target': n_meta_target,
                        'meta_method': meta_method,
                        'meta_metric': meta_metric,
                        'norm_z1': norm_z1,
                        'norm_z2': norm_z2,
                        'norm_cdf': norm_cdf,
                        'norm_nlog': norm_nlog,
                        'norm_sinh': norm_sinh,
                        'output_col': output_col,
                        'cluster_col': cluster_col,
                        'batch_id_col': batch_id_col,
                        'enable_low_expr_filter': enable_low_expr_filter,
                        'low_expr_metric': low_expr_metric,
                        'low_expr_threshold': _coerce_som_low_expression_threshold(low_expr_threshold, low_expr_metric),
                        'min_marker_sum': (
                            _coerce_som_low_expression_threshold(low_expr_threshold, low_expr_metric)
                            if enable_low_expr_filter
                            else get_low_expression_default_threshold(low_expr_metric)
                        ),
                        'cell_id_col': cell_id_col,
                        'enable_filter': enable_filter,
                        'filter_col': filter_col,
                        'filter_val': filter_val,
                        'markers': markers_selected,
                    }
                    with open(som_cache, 'w') as f:
                        json.dump(st.session_state['som_config'], f, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                    st.success("Parameters saved to cache.")
                    st.rerun()

            with save_col2:
                st.caption(f"Parameters will be persisted in: `{som_cache.name}`")

    # Use confirmed markers
    markers_selected = st.session_state.get('som_confirmed_markers', [])
    if 'som_config' in st.session_state:
        st.session_state['som_config']['markers'] = list(markers_selected)
    st.caption(f"Confirmed markers: {len(markers_selected)}")
    
    if not markers_selected:
        st.warning("⚠️ No markers selected. Please select markers above before running.")

    # --- LOAD EXISTING RESULTS / START FRESH ---
    if full_input_path and os.path.exists(full_input_path):
        st.divider()
        st.markdown("##### Existing SOM Results")

        can_load_existing = False
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            can_load_existing = (
                "meta_cluster" in df_head.columns
                and output_col in df_head.columns
            )
        except Exception:
            can_load_existing = False

        c1, c2 = st.columns([1, 1])
        with c1:
            load_existing = st.button(
                "Load Existing SOM Results",
                disabled=(not can_load_existing or not markers_selected),
                help="Load phenotype_som + meta_cluster from the unified CSV and rebuild the heatmap with selected markers.",
                key="som_load_existing_btn",
            )
        with c2:
            start_fresh = st.button(
                "Start Fresh (Ignore Existing)",
                help="Clear loaded SOM results and start a new clustering run.",
                key="som_start_fresh_btn",
            )

        if start_fresh:
            for key in [
                "som_phenotype_map",
                "som_available_phenotypes",
                "som_heatmap_cache",
                "som_temp_file",
                "som_heatmap_component_event_id",
                "som_heatmap_component_reset_counter",
                "som_run_id",
                "som_config_hash",
                "som_low_expr_cluster_id",
                "som_low_expr_count",
                "som_low_expr_metric",
                "som_low_expr_threshold",
                "som_low_expr_default_pheno",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['som_fresh_assignment_mode'] = True
            st.session_state['som_heatmap_component_reset_counter'] = (
                int(st.session_state.get('som_heatmap_component_reset_counter', 0)) + 1
            )
            st.success("SOM state cleared. Ready for a fresh run.")

        if load_existing:
            try:
                st.session_state['som_fresh_assignment_mode'] = False
                from spacemap.ph_accel_utils import read_csv_fast
                df_full = read_csv_fast(full_input_path)
                cluster_col = "meta_cluster"
                pheno_col = output_col

                norm_df, norm_cols_loaded = _apply_norm_for_heatmap(df_full, markers_selected)
                low_expr_mask = _get_som_low_expression_mask(norm_df)
                low_expr_count = int(low_expr_mask.sum())
                clustered_df = norm_df[~low_expr_mask].copy()

                pairs = norm_df[[cluster_col, pheno_col]].drop_duplicates().dropna()
                existing_map = {}
                for _, row in pairs.iterrows():
                    cluster_id = normalize_cluster_id(row[cluster_col])
                    if cluster_id is not None:
                        existing_map[cluster_id] = row[pheno_col]
                if low_expr_count and existing_map.get(SOM_LOW_EXPR_CLUSTER_ID) in {None, 'Unassigned'}:
                    existing_map[SOM_LOW_EXPR_CLUSTER_ID] = SOM_LOW_EXPR_DEFAULT_PHENOTYPE

                temp_file = create_session_temp_parquet_path(project_root, "som_existing")
                write_dataframe_parquet_atomic(norm_df, temp_file)

                st.session_state['som_temp_file'] = str(temp_file)
                st.session_state['som_markers'] = markers_selected
                st.session_state['som_norm_cols'] = norm_cols_loaded
                st.session_state['som_n_meta_clusters'] = clustered_df[cluster_col].nunique() if cluster_col in clustered_df.columns else 0
                st.session_state['som_cluster_col_name'] = st.session_state.get('som_cluster_col', 'som_cluster')
                st.session_state['som_phenotype_map'] = existing_map
                st.session_state['som_available_phenotypes'] = [
                    'Unassigned'
                ] + sorted({p for p in norm_df[pheno_col].dropna().unique().tolist() if str(p) != 'Unassigned'})
                st.session_state['som_run_id'] = f"{time.time():.6f}"
                st.session_state['som_low_expr_cluster_id'] = SOM_LOW_EXPR_CLUSTER_ID if low_expr_count else None
                st.session_state['som_low_expr_count'] = low_expr_count
                st.session_state['som_low_expr_metric'] = None
                st.session_state['som_low_expr_threshold'] = None
                st.session_state['som_low_expr_default_pheno'] = SOM_LOW_EXPR_DEFAULT_PHENOTYPE
                st.session_state['som_heatmap_component_event_id'] = None
                st.session_state['som_heatmap_component_reset_counter'] = (
                    int(st.session_state.get('som_heatmap_component_reset_counter', 0)) + 1
                )
                if 'som_heatmap_cache' in st.session_state:
                    del st.session_state['som_heatmap_cache']
                st.success("Loaded existing SOM results from unified CSV.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load existing SOM results: {e}")
        
    # --- Run Pipeline Button ---
    st.divider()
    
    # Compute config hash to detect parameter changes
    config_hash_items = [
        str(full_input_path),
        str(sorted(markers_selected)),
        str(auto_grid), str(grid_dim), str(target_cells_per_node),
        str(min_grid), str(max_grid), str(mode),
        str(subsample_size), str(subsample_method), str(early_stop),
        str(early_stop_min_improvement), str(early_stop_min_epoch_fraction), str(random_seed),
        str(n_meta_target), str(meta_method), str(meta_metric),
        str(norm_z1), str(norm_z2), str(norm_cdf), str(norm_nlog)
    ]
    current_config_hash = hashlib.md5('|'.join(config_hash_items).encode()).hexdigest()

    existing_run_log = logger.get_streamlit_buffer("som_run_log_lines")
    if existing_run_log:
        with st.expander("Last SOM Tab Terminal", expanded=False):
            st.caption("Timestamped messages mirrored from the shell terminal during the last SOM run.")
            st.code("\n".join(existing_run_log), language=None)
    
    if st.button(" Run SOM Clustering Pipeline", type="primary", width="stretch", key="som_run_btn"):
        def _warn_if_too_few_cells(remaining_cells, stage_label, filter_settings):
            if auto_grid:
                min_required_cells = int(min_grid) * int(min_grid)
                grid_message = f"auto-grid minimum of {min_grid}x{min_grid} ({min_required_cells} nodes)"
            else:
                min_required_cells = int(grid_dim) * int(grid_dim)
                grid_message = f"manual grid of {grid_dim}x{grid_dim} ({min_required_cells} nodes)"

            if remaining_cells >= min_required_cells:
                return False

            filter_message = "No metadata filter is currently enabled."
            if filter_settings.get('enable_filter', False):
                filter_message = (
                    f"Active filter: column '{filter_settings.get('filter_col', 'unknown')}' "
                    f"== '{filter_settings.get('filter_val', 'unknown')}'."
                )

            low_expr_message = "Low-expression filtering is disabled."
            if enable_low_expr_filter:
                som_thresh_val = _coerce_som_low_expression_threshold(low_expr_threshold, low_expr_metric)
                low_expr_message = (
                    f"Low-expression filter is enabled: {get_low_expression_metric_label(low_expr_metric)} "
                    f"> {som_thresh_val:.4g}."
                )

            warning_message = (
                f"Warning: only {remaining_cells:,} cells remain after {stage_label}. "
                f"The current SOM settings require at least {min_required_cells:,} cells because the pipeline is configured for a {grid_message}.\n\n"
                f"Please check whether your filtration step is correct before running again. A wrong filter column or filter value can easily reduce the dataset too far.\n\n"
                f"{filter_message}\n"
                f"{low_expr_message}\n\n"
                f"If this small dataset is intentional, reduce the grid size by lowering Min Grid or turn off Auto Grid and choose a smaller manual grid."
            )
            logger.warning(warning_message)
            st.warning(warning_message)
            return True

        if not full_input_path or not Path(full_input_path).exists():
            st.error(" Please select a valid input file.")
            return
        if not markers_selected:
            st.error(" No markers selected. Please save marker selection first.")
            return

        run_audit = begin_phenotyping_run_audit(
            Path(project_root),
            "som",
            input_csv=full_input_path,
            markers=markers_selected,
        )
        audit_input_rows = None
        
        # IMPORTANT: Clear previous results and phenotype assignments when running new clustering
        # This ensures a fresh start without stale green frames
        if 'som_phenotype_map' in st.session_state:
            del st.session_state['som_phenotype_map']
        if 'som_available_phenotypes' in st.session_state:
            del st.session_state['som_available_phenotypes']
        if 'som_heatmap_cache' in st.session_state:
            del st.session_state['som_heatmap_cache']
        if 'som_temp_file' in st.session_state:
            del st.session_state['som_temp_file']
        if 'som_heatmap_component_event_id' in st.session_state:
            del st.session_state['som_heatmap_component_event_id']
        if 'som_heatmap_component_reset_counter' in st.session_state:
            del st.session_state['som_heatmap_component_reset_counter']
        if 'som_run_id' in st.session_state:
            del st.session_state['som_run_id']
        for key in [
            'som_low_expr_cluster_id',
            'som_low_expr_count',
            'som_low_expr_metric',
            'som_low_expr_threshold',
            'som_low_expr_default_pheno',
        ]:
            if key in st.session_state:
                del st.session_state[key]
        
        with st.status("Running SOM Pipeline...", expanded=True) as status:
            st.markdown("##### SOM Tab Terminal")
            st.caption("Timestamped live log mirrored from the shell terminal.")
            tab_log_placeholder = st.empty()
            logger.start_streamlit_capture(tab_log_placeholder, "som_run_log_lines", clear=True)
            try:
                with Timer("Total SOM Pipeline"):
                    # Step 1: Data Loading
                    logger.info("="*80)
                    logger.info("STEP 1: DATA LOADING & PREPARATION")
                    logger.info("="*80)
                    logger.info(f"Loading data from: {full_input_path}")
                    
                    from spacemap.ph_accel_utils import read_csv_fast
                    df = read_csv_fast(full_input_path)
                    df, cell_id_col, using_internal_cell_anchor = ensure_cell_id_anchor(
                        df,
                        normalize_optional_column_name(st.session_state.get('som_config', {}).get('cell_id_col')),
                    )
                    original_count = len(df)
                    audit_input_rows = original_count
                    logger.info(f"Loaded: {original_count:,} rows")
                    
                    # Apply local filtering from config
                    som_cfg = st.session_state.get('som_config', {})
                    _filter_settings = {
                        'enable_filter': som_cfg.get('enable_filter', False),
                        'filter_col': som_cfg.get('filter_col', 'tissue'),
                        'filter_val': som_cfg.get('filter_val', '1'),
                    }
                    df = apply_data_filter(df, _filter_settings, logger_fn=logger.info)

                    if _warn_if_too_few_cells(len(df), "metadata filtering", _filter_settings):
                        status.update(label="Warning: too few cells remain after filtering", state="error", expanded=True)
                        return
                    
                    # Validate Cell ID Column
                    if using_internal_cell_anchor:
                        logger.info("No valid Cell ID column selected/found. Using an internal row-order anchor for this SOM run.")
                    else:
                        n_unique = df[cell_id_col].nunique()
                        logger.info(f"Cell ID column '{cell_id_col}': {n_unique:,} unique cells")
                    
                    # Check marker availability
                    missing_markers = [m for m in markers_selected if m not in df.columns]
                    if missing_markers:
                        st.error(f"❌ Missing markers in dataset: {missing_markers}")
                        logger.error(f"Missing markers: {missing_markers}")
                        return
                    
                    # Drop rows containing NaN/Inf in selected markers
                    if markers_selected:
                        original_len_nan = len(df)
                        valid_mask = ~df[markers_selected].apply(pd.to_numeric, errors='coerce').isna().any(axis=1)
                        df = df[valid_mask].copy()
                        logger.info(f"Filtered out {original_len_nan - len(df):,} cells containing NaN/Inf in selected markers. Remaining: {len(df):,} cells.")
                        if len(df) == 0:
                            st.error("No cells remain after filtering out NaN/Inf values in selected markers.")
                            logger.error("No cells remain after filtering out NaN/Inf values in selected markers.")
                            return
                    
                    logger.info(f"Processing {len(df):,} cells with {len(markers_selected)} markers")
                    logger.info(f"Markers: {', '.join(markers_selected[:5])}{'...' if len(markers_selected) > 5 else ''}")
                    
                    # Step 2: Normalization (Match Notebook Pipeline)
                    with Timer("Normalization"):
                        logger.info("="*80)
                        logger.info("STEP 2: NORMALIZATION (Notebook Synchronized)")
                        logger.info("="*80)
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

                        # Detect batch column from widget selection
                        batch_col = batch_id_col if batch_id_col in df_input.columns else None
                        
                        # Use ph_leiden_utils to ensure Batch-WISE normalization (matching notebook)
                        # NOTE: SOM notebook imports normalization FROM ph_leiden_utils
                        if ph_leiden_utils:
                            group_cols = [batch_col] if batch_col else []
                            if not group_cols:
                                logger.info("No batch column found - performing global normalization (Notebook uses batch-wise)")
                            else:
                                logger.info(f"Normalizing with grouping: {group_cols}")

                            norm_subset = ph_leiden_utils.apply_flexible_normalization(
                                df=df_input,
                                marker_cols=markers_selected,
                                group_cols=group_cols,
                                suffix='_norm',
                                apply_z1=norm_z1,
                                apply_z2=norm_z2,
                                apply_cdf=norm_cdf,
                                apply_nlog=norm_nlog
                            )
                            
                            # Merge back results (on original raw/un-sinh scaled df)
                            norm_data = df.copy()
                            norm_cols = list(norm_subset.columns)
                            norm_data = pd.concat([norm_data, norm_subset], axis=1)
                        else:
                            # Fallback manual normalization
                            logger.warning("ph_leiden_utils not found, falling back to global normalization")
                            norm_data = df.copy()
                            norm_cols = []
                            for m in markers_selected:
                                col_data = df_input[m].fillna(0).copy()
                                if norm_nlog: col_data = np.log1p(col_data)
                                if norm_cdf: col_data = col_data.rank(pct=True)
                                if norm_z1:
                                    std_val = col_data.std()
                                    col_data = (col_data - col_data.mean()) / (std_val if std_val > 1e-9 else 1)
                                if norm_z2: logger.warning("Z2 enabled")
                                col_data = col_data.fillna(0)
                                norm_col = f"{m}_norm"
                                norm_data[norm_col] = col_data
                                norm_cols.append(norm_col)
                        
                        # Ensure matrix is clean
                        norm_data[norm_cols] = norm_data[norm_cols].fillna(0)
                        logger.info(f"Normalization complete: {len(norm_cols)} markers normalized")

                        raw_marker_values = norm_data[markers_selected].apply(pd.to_numeric, errors='coerce').fillna(0)
                        norm_data['som_marker_sum_raw'] = raw_marker_values.sum(axis=1)
                        low_expr_scores = compute_low_expression_score(
                            df=norm_data,
                            raw_marker_cols=markers_selected,
                            norm_marker_cols=norm_cols,
                            metric=low_expr_metric,
                        )
                        norm_data['som_low_expr_score'] = low_expr_scores
                        low_expr_data = norm_data.iloc[0:0].copy()
                        low_expr_count = 0
                        
                        # Apply low-expression filtering using the shared cell-level score.
                        if enable_low_expr_filter:
                            pre_filter = len(norm_data)
                            som_low_expr_thresh_val = _coerce_som_low_expression_threshold(low_expr_threshold, low_expr_metric)
                            low_expr_mask = low_expr_scores <= som_low_expr_thresh_val
                            low_expr_count = int(low_expr_mask.sum())
                            low_expr_data = norm_data[low_expr_mask].copy()
                            norm_data = norm_data[~low_expr_mask].copy()
                            df = norm_data # Update main reference
                            logger.info(
                                "Low expression filter (%s <= %.4f): %s/%s cells preserved in the low-expression bucket; %s cells sent to SOM training",
                                get_low_expression_metric_label(low_expr_metric),
                                som_low_expr_thresh_val,
                                f"{low_expr_count:,}",
                                f"{pre_filter:,}",
                                f"{len(norm_data):,}",
                            )
                            if norm_data.empty:
                                st.error(
                                    "All cells fell into the low-expression bucket. Lower the expression threshold or disable low-expression screening."
                                )
                                logger.error("No cells remained for SOM training after low-expression screening.")
                                return
                            if _warn_if_too_few_cells(len(norm_data), "low-expression filtering", _filter_settings):
                                status.update(label="Warning: too few cells remain after filtering", state="error", expanded=True)
                                return
                        else:
                            logger.info("Low expression filtering is disabled.")
                            df = norm_data
                        
                        # Prepare scaled data for SOM
                        from sklearn.preprocessing import StandardScaler
                        clustering_data = norm_data[norm_cols].values
                        scaler = StandardScaler()
                        clustering_data_scaled = scaler.fit_transform(clustering_data)
                        logger.info(f"Scaled data shape: {clustering_data_scaled.shape}")
                    
                    # Step 3: SOM Training (Using ph_som_utils for notebook consistency)
                    with Timer("SOM Training"):
                        logger.info("="*80)
                        logger.info("STEP 3: SOM TRAINING (Advanced)")
                        logger.info("="*80)
                        
                        # Prepare adaptive config (matches notebook)
                        if auto_grid:
                            adaptive_config = {
                                'auto': True,
                                'target_cells_per_node': target_cells_per_node,
                                'min_grid': min_grid,
                                'max_grid': max_grid,
                                'aspect': 1.0,
                                'mode': mode,
                                'base_epochs': 80  # Notebook BASE_EPOCHS
                            }
                            grid_x = grid_y = 50  # Will be overridden by adaptive
                        else:
                            adaptive_config = {
                                'auto': False,
                                'mode': mode,
                                'base_epochs': 80
                            }
                            grid_x = grid_y = grid_dim
                        
                        # Mode parameters (from notebook MODE_PARAMS)
                        mode_params_map = {
                            'fast': {'iter_factor': 0.6, 'start_sigma': 1.3, 'lr': 0.08, 'batch_size': 4096},
                            'balanced': {'iter_factor': 1.0, 'start_sigma': 1.5, 'lr': 0.10, 'batch_size': 4096},
                            'high_quality': {'iter_factor': 1.5, 'start_sigma': 1.8, 'lr': 0.12, 'batch_size': 3072}
                        }
                        mode_params = mode_params_map[mode]
                        
                        logger.info(f"Training mode: {mode}")
                        logger.info(f"Adaptive config: {adaptive_config}")
                        logger.info(f"Subsample size: {subsample_size:,}")
                        
                        # Check if ph_som_utils is available
                        if 'ph_som_utils' in sys.modules and hasattr(ph_som_utils, 'train_som_advanced'):
                            logger.info("Using ph_som_utils.train_som_advanced (Optimized)")
                            
                            # Progress indicator
                            progress_bar = st.progress(0, text="Initializing SOM training...")
                            
                            try:
                                # Run advanced training (matches notebook exactly)
                                som_result = ph_som_utils.train_som_advanced(
                                    clustering_data_scaled=clustering_data_scaled,
                                    data_df=df,
                                    cell_id_column=cell_id_col,
                                    adaptive_config=adaptive_config,
                                    mode_params=mode_params,
                                    som_x_dim=grid_x,
                                    som_y_dim=grid_y,
                                    subsample_size=subsample_size,
                                    subsample_method=subsample_method,
                                    early_stop=early_stop,
                                    early_stop_min_improvement=early_stop_min_improvement,
                                    early_stop_min_epoch_fraction=early_stop_min_epoch_fraction,
                                    random_seed=random_seed,
                                    eval_size=min(4000, len(clustering_data_scaled)),
                                    second_stage_scaling=False,
                                    qe_scaling_strategy='none'
                                )
                                
                                progress_bar.progress(100, text="SOM training complete!")
                                
                                # Extract results
                                som = som_result['som']
                                som_clusters_df = som_result['som_clusters_df']
                                grid_size = som_result['grid_size']
                                training_summary = som_result['training_summary']
                                
                                # Prevent merge suffixes if som_cluster already exists
                                if 'som_cluster' in df.columns:
                                    df = df.drop(columns=['som_cluster'])

                                # Merge SOM clusters with original dataframe using cell_id anchoring
                                df = pd.merge(df, som_clusters_df, on=cell_id_col, how='left')
                                
                                # Display training summary
                                logger.info(f"Grid: {grid_size[0]}x{grid_size[1]} = {grid_size[0]*grid_size[1]} nodes")
                                logger.info(f"Training cells: {training_summary['n_training']:,}")
                                logger.info(f"Final QE: {training_summary['final_qe']:.3f}")
                                logger.info(f"Final nQE: {training_summary['final_nqe']:.3f}")
                                logger.info(f"Final TE: {training_summary['final_te']:.3f}")
                                logger.info(f"Epochs completed: {training_summary['epochs_completed']}")
                                
                            except Exception as e:
                                logger.error(f"Advanced SOM training failed: {e}")
                                st.error(f" SOM training error: {e}")
                                return
                        else:
                            logger.warning("ph_som_utils not available. Using fallback MiniSom.")
                            st.warning(" Advanced SOM utils not available. Using basic MiniSom (slower).")
                            
                            # Fallback to basic MiniSom
                            try:
                                from minisom import MiniSom
                                
                                # Use manual grid size
                                grid_x = grid_y = grid_dim if not auto_grid else 50
                                
                                st.write(f"Training SOM ({grid_x}×{grid_y})...")
                                som = MiniSom(grid_x, grid_y, clustering_data_scaled.shape[1], 
                                            sigma=1.5, learning_rate=0.10, random_seed=random_seed)
                                
                                # Subsample for training
                                if len(clustering_data_scaled) > subsample_size:
                                    np.random.seed(random_seed)
                                    train_indices = np.random.choice(len(clustering_data_scaled), 
                                                                    subsample_size, replace=False)
                                    train_data = clustering_data_scaled[train_indices]
                                else:
                                    train_data = clustering_data_scaled
                                
                                # Train
                                iterations = int(80 * mode_params['iter_factor'])
                                som.train_random(train_data, iterations)
                                
                                # Map all cells to nodes
                                logger.info("Mapping all cells to SOM nodes...")
                                from scipy.spatial.distance import cdist
                                weights_flat = som.get_weights().reshape(-1, clustering_data_scaled.shape[1])
                                
                                # Chunk processing for large datasets
                                chunk_size = 50000
                                n_samples = len(clustering_data_scaled)
                                som_nodes = np.zeros(n_samples, dtype=int)
                                
                                for i in range(0, n_samples, chunk_size):
                                    chunk = clustering_data_scaled[i:min(i + chunk_size, n_samples)]
                                    dists = cdist(chunk, weights_flat, metric='euclidean')
                                    som_nodes[i:i + len(chunk)] = np.argmin(dists, axis=1)
                                
                                df['som_cluster'] = som_nodes
                                logger.info(f"Assigned {len(np.unique(som_nodes))} unique SOM nodes")
                                
                            except Exception as e:
                                logger.error(f"Fallback MiniSom failed: {e}")
                                st.error(f" SOM training failed: {e}")
                                return
                    
                    # Rename cluster column to match output configuration
                    cluster_col_output = st.session_state.get('som_cluster_col', 'som_cluster')
                    if 'som_cluster' in df.columns and cluster_col_output != 'som_cluster':
                        df = df.rename(columns={'som_cluster': cluster_col_output})
                    
                    n_som_clusters = df[cluster_col_output].nunique()
                    logger.info(f"SOM clustering complete: {n_som_clusters} SOM clusters")
                    
                    # IMPORTANT: Add normalized columns to df BEFORE meta-clustering
                    # (characterize_som_clusters needs these columns)
                    for nc in norm_cols:
                        if nc in norm_data.columns:
                            df[nc] = norm_data[nc].values
                    logger.info(f"Added {len(norm_cols)} normalized columns to df for characterization")
                    
                    # Step 4: Meta-Clustering (Match Notebook)
                    with Timer("Meta-Clustering"):
                        logger.info("="*80)
                        logger.info("STEP 4: META-CLUSTERING")
                        logger.info("="*80)
                        logger.info(f"Target meta-clusters: {n_meta_target}")
                        logger.info(f"Method: {meta_method}, Metric: {meta_metric}")
                        
                        # Verify norm_cols exist in df
                        available_norm_cols = [c for c in norm_cols if c in df.columns]
                        if not available_norm_cols:
                            raise ValueError(f"No normalized columns found in df. Expected: {norm_cols[:3]}...")
                        logger.info(f"Using {len(available_norm_cols)} normalized columns for characterization")
                        
                        if 'ph_som_utils' in sys.modules and hasattr(ph_som_utils, 'characterize_som_clusters'):
                            # Use notebook utils for consistency
                            logger.info("Using ph_som_utils for characterization and meta-clustering")
                            
                            # Characterize SOM clusters (use available_norm_cols to ensure columns exist)
                            cluster_char_df = ph_som_utils.characterize_som_clusters(
                                df, df[cluster_col_output].values, available_norm_cols
                            )
                            
                            # Debug: log characterization results
                            logger.info(f"Characterization complete: {len(cluster_char_df)} clusters, columns: {list(cluster_char_df.columns)[:10]}...")
                            
                            # Meta-cluster
                            meta_char_df, meta_cluster_map = ph_som_utils.meta_cluster_clusters(
                                cluster_char_df.copy(),
                                target_n=n_meta_target,
                                method=meta_method,
                                metric=meta_metric
                            )
                            
                            # Apply meta-cluster assignments
                            df['meta_cluster'] = df[cluster_col_output].map(meta_cluster_map)
                            
                            logger.info(f"Meta-clustering complete: {len(meta_char_df)} meta-clusters")
                            
                        else:
                            logger.warning("Using fallback hierarchical clustering for meta-clusters")
                            
                            # Fallback: Simple hierarchical clustering
                            import scipy.cluster.hierarchy as sch
                            from scipy.cluster.hierarchy import fcluster
                            
                            # Get SOM weights
                            weights = som.get_weights().reshape(-1, clustering_data_scaled.shape[1])
                            
                            # Hierarchical clustering
                            Z = sch.linkage(weights, method=meta_method, metric=meta_metric)
                            meta_labels = fcluster(Z, t=n_meta_target, criterion='maxclust')
                            
                            # Map SOM nodes to meta-clusters
                            node_to_meta = dict(enumerate(meta_labels))
                            df['meta_cluster'] = df[cluster_col_output].map(node_to_meta)
                            
                            logger.info(f"Meta-clustering complete: {df['meta_cluster'].nunique()} meta-clusters")
                    
                    # (Normalized columns already added before meta-clustering)
                    
                    # Save results
                    logger.info("="*80)
                    logger.info("SAVING RESULTS")
                    logger.info("="*80)
                    
                    temp_file = create_session_temp_parquet_path(project_root, 'som_temp_res')
                    
                    df_results = df.copy()
                    df_results['som_low_expression'] = False
                    df_results['som_filter_status'] = 'clustered'
                    if low_expr_count:
                        low_expr_results = low_expr_data.copy()
                        low_expr_results[cluster_col_output] = SOM_LOW_EXPR_CLUSTER_ID
                        low_expr_results['meta_cluster'] = SOM_LOW_EXPR_CLUSTER_ID
                        low_expr_results['som_low_expression'] = True
                        low_expr_results['som_filter_status'] = 'low_expression'
                        df_results = pd.concat([df_results, low_expr_results], axis=0).sort_index(kind='stable')
                        logger.info(
                            "Preserved %s low-expression cells as %s with default phenotype '%s'.",
                            f"{low_expr_count:,}",
                            SOM_LOW_EXPR_LABEL,
                            SOM_LOW_EXPR_DEFAULT_PHENOTYPE,
                        )

                    write_dataframe_parquet_atomic(df_results, temp_file)
                    logger.info(f"Results saved: {temp_file}")
                    
                    # Store in session state (include config hash for cache validation)
                    st.session_state['som_temp_file'] = str(temp_file)
                    st.session_state['som_markers'] = markers_selected
                    st.session_state['som_norm_cols'] = norm_cols
                    st.session_state['som_n_meta_clusters'] = df['meta_cluster'].nunique()
                    st.session_state['som_cluster_col_name'] = cluster_col_output
                    st.session_state['som_config_hash'] = current_config_hash
                    st.session_state['som_run_id'] = f"{time.time():.6f}"
                    initial_phenotype_map = {
                        normalize_cluster_id(cluster_id): 'Unassigned'
                        for cluster_id in sorted(df['meta_cluster'].dropna().unique().tolist(), key=cluster_sort_key)
                        if normalize_cluster_id(cluster_id) is not None
                    }
                    if low_expr_count and not st.session_state.get('som_fresh_assignment_mode', False):
                        initial_phenotype_map[SOM_LOW_EXPR_CLUSTER_ID] = SOM_LOW_EXPR_DEFAULT_PHENOTYPE
                    st.session_state['som_phenotype_map'] = initial_phenotype_map
                    st.session_state['som_low_expr_cluster_id'] = SOM_LOW_EXPR_CLUSTER_ID if low_expr_count else None
                    st.session_state['som_low_expr_count'] = int(low_expr_count)
                    st.session_state['som_low_expr_metric'] = low_expr_metric if enable_low_expr_filter else None
                    st.session_state['som_low_expr_threshold'] = float(low_expr_threshold) if (enable_low_expr_filter and low_expr_threshold is not None) else None
                    st.session_state['som_low_expr_default_pheno'] = SOM_LOW_EXPR_DEFAULT_PHENOTYPE
                    
                    logger.info("="*80)
                    logger.info("SOM PIPELINE COMPLETE")
                    logger.info("="*80)
                    
                    status.update(label="SOM Pipeline Complete!", state="complete", expanded=False)
                    if low_expr_count:
                        st.success(
                            f"SOM clustering completed. Generated **{n_som_clusters}** SOM nodes → **{df['meta_cluster'].nunique()}** final meta-clusters and preserved **{low_expr_count:,}** low-expression cells as '{SOM_LOW_EXPR_DEFAULT_PHENOTYPE}'."
                        )
                    else:
                        st.success(f"SOM clustering completed. Generated **{n_som_clusters}** SOM nodes → **{df['meta_cluster'].nunique()}** final meta-clusters")

                    finish_phenotyping_run_audit(
                        run_audit,
                        status="success",
                        output_path=temp_file,
                        input_rows=audit_input_rows,
                        output_rows=len(df_results),
                        output_column=output_col,
                        cluster_column="meta_cluster",
                        phenotype_map=initial_phenotype_map,
                        extra={
                            "som_node_count": int(n_som_clusters),
                            "meta_cluster_count": int(df['meta_cluster'].nunique()),
                            "low_expression_cells": int(low_expr_count),
                        },
                    )
                    run_audit = None
                    
            except Exception as e:
                finish_phenotyping_run_audit(
                    run_audit,
                    status="failed",
                    input_rows=audit_input_rows,
                    output_column=output_col,
                    cluster_column="meta_cluster",
                    error=str(e),
                )
                run_audit = None
                logger.error(f"SOM pipeline error: {e}")
                import traceback
                st.error(f" Error during execution: {e}")
                st.code(traceback.format_exc())
                status.update(label=" Pipeline Failed", state="error")
            finally:
                if run_audit is not None:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_col,
                        cluster_column="meta_cluster",
                        error="Pipeline stopped before completion; see the SOM terminal log.",
                    )
                logger.stop_streamlit_capture()
    
    # --- INTERACTIVE VISUALIZATION & PHENOTYPE ASSIGNMENT ---
    # (This section is identical in structure to Leiden, adapted for SOM meta-clusters)
    if 'som_temp_file' in st.session_state and Path(st.session_state['som_temp_file']).exists():
        st.divider()
        st.subheader("Interactive Visualization & Phenotype Assignment")
        st.caption("Professional phenotype assignment interface synchronized with notebook workflow (Step6B)")
        
        # Load results
        temp_file = st.session_state['som_temp_file']
        df_results = read_dataframe_parquet_safe(
            temp_file,
            session_state_key='som_temp_file',
            label='SOM results',
        )
        if df_results is None:
            return
        norm_cols = st.session_state.get('som_norm_cols', [])
        low_expr_cluster_id = st.session_state.get('som_low_expr_cluster_id')
        if low_expr_cluster_id is None and 'som_low_expression' in df_results.columns and df_results['som_low_expression'].fillna(False).any():
            low_expr_cluster_id = SOM_LOW_EXPR_CLUSTER_ID
        low_expr_mask = _get_som_low_expression_mask(df_results)
        clustered_df = df_results[~low_expr_mask].copy()
        low_expr_count = int(low_expr_mask.sum())
        n_meta_clusters = st.session_state.get('som_n_meta_clusters', clustered_df['meta_cluster'].nunique() if 'meta_cluster' in clustered_df.columns else 0)
        output_col = st.session_state.get('som_outcol', 'phenotype_som')
        cluster_col_output = st.session_state.get('som_cluster_col_name', 'som_cluster')
        
        # Create tabs (mirror Leiden structure)
        tab_heatmap, tab_table, tab_export = st.tabs([
            "Heatmap & Assignment",
            "Table Editor",
            "Export Results"
        ])
        
        # TAB 1: HEATMAP & ASSIGNMENT
        with tab_heatmap:
            st.markdown("### Interactive Meta-Cluster Expression Heatmap")
            st.caption("Professional heatmap interface for phenotype assignment")
            
            # Initialize session state (filter NaN values from meta_cluster)
            valid_meta_clusters = sorted(
                [normalize_cluster_id(c) for c in df_results['meta_cluster'].unique() if normalize_cluster_id(c) is not None],
                key=cluster_sort_key,
            )
            fresh_assignment_reset = bool(st.session_state.pop('som_fresh_assignment_mode', False))
            if fresh_assignment_reset:
                st.session_state['som_phenotype_map'] = {
                    normalize_cluster_id(cluster_id): 'Unassigned'
                    for cluster_id in valid_meta_clusters
                    if normalize_cluster_id(cluster_id) is not None
                }
            elif 'som_phenotype_map' not in st.session_state:
                st.session_state['som_phenotype_map'] = {
                    cluster_id: (
                        st.session_state.get('som_low_expr_default_pheno', SOM_LOW_EXPR_DEFAULT_PHENOTYPE)
                        if _is_som_low_expression_cluster(cluster_id, low_expr_cluster_id)
                        else 'Unassigned'
                    )
                    for cluster_id in valid_meta_clusters
                }
            if low_expr_count and low_expr_cluster_id is not None and not fresh_assignment_reset:
                current_value = st.session_state['som_phenotype_map'].get(normalize_cluster_id(low_expr_cluster_id))
                if current_value in {None, 'Unassigned'}:
                    st.session_state['som_phenotype_map'][normalize_cluster_id(low_expr_cluster_id)] = (
                        st.session_state.get('som_low_expr_default_pheno', SOM_LOW_EXPR_DEFAULT_PHENOTYPE)
                    )
            st.session_state['som_phenotype_map'] = normalize_phenotype_map_keys(
                st.session_state.get('som_phenotype_map', {})
            )
            # Always rebuild available phenotypes from shared system
            st.session_state['som_available_phenotypes'] = get_all_phenotype_options(include_unassigned=True)
            
            # Heatmap controls (match Leiden behavior)
            if 'som_heatmap_config' not in st.session_state:
                st.session_state['som_heatmap_config'] = {
                    'cluster_x': True,
                    'cluster_y': True,
                    'show_counts': True,
                    'z_min': -4,
                    'z_max': 4
                }
            # Ensure keys exist (backward compatibility)
            if 'z_min' not in st.session_state['som_heatmap_config']: st.session_state['som_heatmap_config']['z_min'] = -4
            if 'z_max' not in st.session_state['som_heatmap_config']: st.session_state['som_heatmap_config']['z_max'] = 4

            hm_row1 = st.columns([2, 2, 2])
            with hm_row1[0]:
                if st.button("Toggle X-axis Clustering", help="Hierarchically cluster clusters (columns)", key="som_toggle_x"):
                    st.session_state['som_heatmap_config']['cluster_x'] = not st.session_state['som_heatmap_config']['cluster_x']
                    # Invalidate cache when toggle changes
                    if 'som_heatmap_cache' in st.session_state:
                        del st.session_state['som_heatmap_cache']
                st.caption(f"X-axis clustering: {'ON' if st.session_state['som_heatmap_config']['cluster_x'] else 'OFF'}")

            with hm_row1[1]:
                if st.button("Toggle Y-axis Clustering", help="Hierarchically cluster markers (rows)", key="som_toggle_y"):
                    st.session_state['som_heatmap_config']['cluster_y'] = not st.session_state['som_heatmap_config']['cluster_y']
                    # Invalidate cache when toggle changes
                    if 'som_heatmap_cache' in st.session_state:
                        del st.session_state['som_heatmap_cache']
                st.caption(f"Y-axis clustering: {'ON' if st.session_state['som_heatmap_config']['cluster_y'] else 'OFF'}")

            with hm_row1[2]:
                if st.button("Toggle Cell Counts", help="Show/hide cell counts", key="som_toggle_counts"):
                    st.session_state['som_heatmap_config']['show_counts'] = not st.session_state['som_heatmap_config']['show_counts']
                    # Invalidate cache when toggle changes
                    if 'som_heatmap_cache' in st.session_state:
                        del st.session_state['som_heatmap_cache']
                st.caption(f"Cell counts: {'ON' if st.session_state['som_heatmap_config']['show_counts'] else 'OFF'}")

            # Heatmap Range Config
            with st.expander("Heatmap Configuration", expanded=False):
                current_min = st.session_state['som_heatmap_config'].get('z_min', -4)
                current_max = st.session_state['som_heatmap_config'].get('z_max', 4)
                
                z_range = st.slider(
                    "Color Scale Range (Z-Score)",
                    min_value=-10.0, max_value=10.0,
                    value=(float(current_min), float(current_max)),
                    step=0.5,
                    key="som_z_range_slider"
                )
                
                if z_range[0] != current_min or z_range[1] != current_max:
                    st.session_state['som_heatmap_config']['z_min'] = z_range[0]
                    st.session_state['som_heatmap_config']['z_max'] = z_range[1]
                    if 'som_heatmap_cache' in st.session_state:
                        del st.session_state['som_heatmap_cache']
                    st.rerun()

            cluster_clusters = st.session_state['som_heatmap_config']['cluster_x']
            cluster_markers = st.session_state['som_heatmap_config']['cluster_y']
            show_counts = st.session_state['som_heatmap_config']['show_counts']
            z_min = st.session_state['som_heatmap_config']['z_min']
            z_max = st.session_state['som_heatmap_config']['z_max']

            if low_expr_count:
                st.info(
                    f"{low_expr_count:,} low-expression cells were preserved outside the clustering heatmap. "
                    f"They default to phenotype '{st.session_state.get('som_low_expr_default_pheno', SOM_LOW_EXPR_DEFAULT_PHENOTYPE)}' and remain available in the assignment and export tables."
                )

            phenotype_map = st.session_state['som_phenotype_map']
            available_phenotypes = st.session_state['som_available_phenotypes']
            if 'som_heatmap_component_reset_counter' not in st.session_state:
                st.session_state['som_heatmap_component_reset_counter'] = 0

            # Generate interactive Plotly heatmap (same engine as Leiden)
            heatmap_z_ordered = pd.DataFrame()
            cluster_order = []
            cell_counts_dict = {}
            marker_order = []
            numeric_norm_cols = [c for c in norm_cols if c in df_results.columns and pd.api.types.is_numeric_dtype(df_results[c])]
            if numeric_norm_cols:
                run_tag = st.session_state.get("som_run_id", "no_run")
                data_hash = f"{run_tag}_{len(clustered_df)}_{clustered_df['meta_cluster'].nunique() if 'meta_cluster' in clustered_df.columns else 0}_{str(sorted(numeric_norm_cols))}"
                # Include phenotype state hash to detect assignment changes (but not for base figure)
                pheno_state_hash = hashlib.md5(str(sorted(phenotype_map.items())).encode()).hexdigest()[:8]
                cache_key = f"som_heatmap::{data_hash}::{cluster_clusters}::{cluster_markers}::{show_counts}::{z_min}::{z_max}"

                cache = st.session_state.get('som_heatmap_cache', {})
                if cache.get('key') == cache_key:
                    base_fig = cache.get('base_fig')
                    cluster_order = cache.get('cluster_order', [])
                    marker_order = cache.get('marker_order', [])
                    cell_counts_dict = cache.get('cell_counts_dict', {})
                    heatmap_z_ordered = cache.get('heatmap_z_ordered', pd.DataFrame())
                else:
                    base_fig, cluster_order, marker_order, cell_counts_dict, heatmap_z_ordered = build_leiden_heatmap_figure(
                        data_hash,
                        clustered_df,
                        numeric_norm_cols,
                        cluster_clusters,
                        cluster_markers,
                        show_counts,
                        cluster_col='meta_cluster',
                        cluster_label='Meta-Cluster',
                        z_min=z_min,
                        z_max=z_max,
                        marker_source_note=build_marker_source_note(numeric_norm_cols, method_key="som"),
                    )
                    st.session_state['som_heatmap_cache'] = {
                        'key': cache_key,
                        'base_fig': base_fig,
                        'cluster_order': cluster_order,
                        'marker_order': marker_order,
                        'cell_counts_dict': cell_counts_dict,
                        'heatmap_z_ordered': heatmap_z_ordered
                    }

                if base_fig:
                    # Apply Leiden-style assigned-cluster frames
                    fig = go.Figure(base_fig)
                    fig.update_layout(dragmode=False)

                    shapes = []
                    c_to_idx = {cid: i for i, cid in enumerate(cluster_order)}
                    for cid in cluster_order:
                        i = c_to_idx.get(cid)
                        if i is None:
                            continue

                        col = None
                        width = 0
                        if phenotype_map.get(cid) != 'Unassigned':
                            col = 'rgba(40, 167, 69, 0.9)'
                            width = 3

                        if col:
                            shapes.append(dict(
                                type='rect', x0=i-0.5, x1=i+0.5, y0=-0.5, y1=len(marker_order)-0.5,
                                line=dict(color=col, width=width),
                                fillcolor='rgba(0,0,0,0)'
                            ))

                    fig.update_layout(shapes=shapes)
                    assigned_count = sum(1 for v in phenotype_map.values() if v != 'Unassigned')
                    fig.update_layout(
                        title_text=(
                            f"Meta-Clusters ({len(cluster_order)} total; "
                            f"{assigned_count} phenotypes assigned)"
                        )
                    )

                    available_phenotypes = get_all_phenotype_options(include_unassigned=True)
                    for phenotype in phenotype_map.values():
                        if phenotype != 'Unassigned' and phenotype not in available_phenotypes:
                            available_phenotypes.append(phenotype)
                    st.session_state['som_available_phenotypes'] = available_phenotypes

                    trace_customdata = np.tile(np.array(cluster_order, dtype=object), (len(marker_order), 1))
                    fig.update_traces(customdata=trace_customdata, selector={"type": "heatmap"})

                    st.markdown("#### Heatmap Click Selection")
                    st.caption(
                        "Click heatmap cells to buffer clusters locally, then apply one phenotype to all selected clusters at once. "
                        "The heatmap rendering and hover content stay unchanged."
                    )

                    cluster_display_labels = {cluster_id: str(cluster_id) for cluster_id in cluster_order}
                    cluster_marker_scores = build_cluster_marker_score_map_from_heatmap(heatmap_z_ordered)

                    # Assisted assignment: score every cluster against user-written
                    # phenotype signatures and propose a name for each, so large cluster
                    # counts stay tractable without merging them away.
                    if render_assisted_assignment(
                        key_prefix="som",
                        phenotype_map_state_key="som_phenotype_map",
                        available_phenotypes_state_key="som_available_phenotypes",
                        cluster_marker_scores=cluster_marker_scores,
                        df_results=locals().get("df_results"),
                        marker_cols=locals().get("norm_cols") or [],
                        cluster_column="meta_cluster",
                    ):
                        st.rerun()
                    if heatmap_click_selector_available():
                        component_event = render_heatmap_click_selector(
                            figure=fig,
                            available_phenotypes=available_phenotypes,
                            cluster_display_labels=cluster_display_labels,
                            cluster_marker_scores=cluster_marker_scores,
                            reset_counter=int(st.session_state.get('som_heatmap_component_reset_counter', 0)),
                            key="som_heatmap_click_selector",
                        )

                        event_is_new = consume_heatmap_assignment_event(
                            component_event,
                            scope=get_heatmap_assignment_scope(
                                "som", st.session_state.get("som_run_id", "no-run")
                            ),
                        )
                        assignment = (
                            apply_heatmap_assignment_event(
                                component_event,
                                st.session_state.get("som_phenotype_map", {}),
                                previous_event_id=st.session_state.get("som_heatmap_component_event_id"),
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
                            st.session_state["som_phenotype_map"] = updated_map
                            st.session_state["som_heatmap_component_event_id"] = event_id
                            st.session_state["som_heatmap_component_reset_counter"] = (
                                int(st.session_state.get("som_heatmap_component_reset_counter", 0)) + 1
                            )
                            logger.info(
                                "SOM phenotype assignment applied: phenotype=%s clusters=%s",
                                selected_click_phenotype,
                                selected_clusters,
                            )
                            st.success(
                                f"Assigned '{selected_click_phenotype}' to {len(selected_clusters)} cluster(s): "
                                + ", ".join(str(cluster_id) for cluster_id in selected_clusters)
                            )
                            st.rerun()
                    else:
                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config={'displayModeBar': False}
                        )
                        st.warning(
                            "Buffered heatmap click selection is unavailable because the local selector component has not been built yet. "
                            "The heatmap is shown in static mode until that frontend build exists."
                        )
                else:
                    st.warning("No valid data to render heatmap.")
            else:
                st.warning("No valid numeric normalized columns found for heatmap generation.")
            
            st.divider()
            
            # Phenotype Assignment Interface
            st.markdown("#### Phenotype Summary Table & Assignment")
            st.caption("Edit phenotypes in table below with no rerendering. Click Sync to apply all changes at once.")
            
            phenotype_map = st.session_state['som_phenotype_map']
            available_phenotypes = st.session_state['som_available_phenotypes']
            
            # Filter out NaN from meta_cluster before grouping
            df_valid_meta = df_results[df_results['meta_cluster'].notna()]
            cluster_stats = df_valid_meta.groupby('meta_cluster').size().to_dict()
            
            # Build summary table
            summary_data = []
            if cluster_order:
                ordered_clusters = cluster_order
            else:
                ordered_clusters = sorted(
                    [normalize_cluster_id(c) for c in df_results['meta_cluster'].unique() if normalize_cluster_id(c) is not None],
                    key=cluster_sort_key,
                )

            for c in ordered_clusters:
                # Top 5 markers from heatmap Z-scores WITH SCORES
                top_markers_str = "N/A"
                if not heatmap_z_ordered.empty and c in heatmap_z_ordered.columns:
                    top5_series = heatmap_z_ordered[c].nlargest(5)
                    top_markers_str = ", ".join([f"{m} ({v:.2f})" for m, v in top5_series.items()])

                summary_data.append({
                    'Cluster': int(c) if pd.notna(c) else 0,
                    'Cells': cell_counts_dict.get(c, cluster_stats.get(c, 0)),
                    'Top Markers (Score)': top_markers_str,
                    'Phenotype': phenotype_map.get(int(c) if pd.notna(c) else 0, 'Unassigned')
                })

            if low_expr_count and low_expr_cluster_id is not None:
                summary_data.append({
                    'Cluster': normalize_cluster_id(low_expr_cluster_id),
                    'Cells': low_expr_count,
                    'Top Markers (Score)': _get_som_low_expression_note(
                        st.session_state.get('som_low_expr_metric'),
                        st.session_state.get('som_low_expr_threshold'),
                    ),
                    'Phenotype': phenotype_map.get(
                        normalize_cluster_id(low_expr_cluster_id),
                        st.session_state.get('som_low_expr_default_pheno', SOM_LOW_EXPR_DEFAULT_PHENOTYPE),
                    )
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Progress metric
            col_p1, col_p2 = st.columns([2, 1])
            col_p1.markdown("##### 1. Select Clusters & Edit Phenotypes")
            
            n_assigned = sum(1 for v in phenotype_map.values() if v != 'Unassigned')
            n_total = len(phenotype_map)
            col_p2.metric("Assigned", f"{n_assigned}/{n_total}")
            
            # --- Add Custom Phenotype UI ---
            st.caption("Need a phenotype not in the list? Add it here:")
            render_add_custom_phenotype_ui(key_prefix="som")

            # Refresh available phenotypes after potential custom addition
            available_phenotypes = get_all_phenotype_options(include_unassigned=True)
            for phenotype in phenotype_map.values():
                if phenotype != 'Unassigned' and phenotype not in available_phenotypes:
                    available_phenotypes.append(phenotype)
            st.session_state['som_available_phenotypes'] = available_phenotypes

            # FORM prevents reruns during dropdown edits
            with st.form("som_batch_assignment_form", clear_on_submit=False):
                
                # Editable table inside form - dropdown changes buffer until submit
                edited_df = st.data_editor(
                    summary_df,
                    column_config={
                        "Cluster": st.column_config.NumberColumn("Cluster", disabled=True),
                        "Cells": st.column_config.NumberColumn("Cells", disabled=True),
                        "Top Markers (Score)": st.column_config.TextColumn("Top Markers (Score)", disabled=True, width="large"),
                        "Phenotype": st.column_config.SelectboxColumn(
                            "Phenotype",
                            options=available_phenotypes,
                            required=True
                        )
                    },
                    disabled=["Cluster", "Cells", "Top Markers (Score)"],
                    hide_index=True,
                    num_rows="fixed",
                    key="som_summary_editor_form",
                    width="stretch"
                )
                
                st.write("")
                st.markdown("##### 2. Sync to Heatmap")
                
                sync_col1, sync_col2 = st.columns([3, 1])
                with sync_col1:
                    st.caption("All edits above are local. Click Sync to apply changes and update heatmap.")
                
                with sync_col2:
                    submitted = st.form_submit_button("Sync & Update", type="primary")
            
            # Apply changes on form submit
            if submitted:
                # Apply edits from table to session state
                new_map = {int(row['Cluster']): row['Phenotype'] for _, row in edited_df.iterrows()}
                st.session_state['som_phenotype_map'] = new_map
                # Invalidate heatmap cache to trigger recompute with green frames
                if 'som_heatmap_cache' in st.session_state:
                    del st.session_state['som_heatmap_cache']
                st.success("Synced! Heatmap updated.")
                st.rerun()

            # --- SHARED POST-ASSIGNMENT SECTIONS (Same as Leiden) ---
            render_phenotype_statistics(
                df_results=df_results,
                cluster_col='meta_cluster',
                phenotype_map=st.session_state.get('som_phenotype_map', {}),
                heatmap_z_ordered=heatmap_z_ordered,
                cluster_label='Meta-Cluster'
            )

            if low_expr_count:
                st.caption(
                    "Low-expression bucket cells contribute to phenotype counts, but their marker ranking is excluded from the heatmap-based top-marker summary because they are preserved outside SOM clustering."
                )

            render_publication_heatmap(
                heatmap_z_ordered=heatmap_z_ordered,
                phenotype_map=st.session_state.get('som_phenotype_map', {}),
                key_prefix='som',
                df_results=df_results,
                group_col='meta_cluster',
                marker_columns=numeric_norm_cols,
            )

            update_cols = []
            for col_name in [cluster_col_output, 'meta_cluster', 'som_low_expression', 'som_filter_status', 'som_low_expr_score', 'som_marker_sum_raw']:
                if col_name in df_results.columns and col_name not in update_cols:
                    update_cols.append(col_name)

            render_save_assignments(
                df_results=df_results,
                phenotype_map=st.session_state.get('som_phenotype_map', {}),
                output_col=output_col,
                project_root=project_root,
                key_prefix='som',
                cluster_col='meta_cluster',
                update_cluster_cols=update_cols,
                cell_id_col=st.session_state.get('som_config', {}).get('cell_id_col', 'cell_id')
            )

            render_export_split_data(
                df_results=df_results,
                output_col=output_col,
                phenotype_map=st.session_state.get('som_phenotype_map', {}),
                project_root=project_root,
                key_prefix='som',
                cluster_col='meta_cluster',
                cell_id_col=st.session_state.get('som_config', {}).get('cell_id_col', 'cell_id')
            )
        
        # TAB 2: TABLE EDITOR
        with tab_table:
            st.markdown("### Phenotype Table Editor")
            st.caption("Alternative view for editing assignments. Changes apply when you click Sync.")
            
            phenotype_map = st.session_state.get('som_phenotype_map', {})
            available_phenotypes = st.session_state.get('som_available_phenotypes', ['Unassigned'])
            
            # Filter out NaN before groupby
            df_valid_table = df_results[df_results['meta_cluster'].notna()]
            cluster_stats = df_valid_table.groupby('meta_cluster').agg({
                'meta_cluster': 'count'
            }).rename(columns={'meta_cluster': 'n_cells'})
            cluster_stats['percentage'] = (cluster_stats['n_cells'] / len(df_results) * 100).round(2)
            
            # Filter out NaN values and convert to int for clean cluster list
            meta_clusters = sorted(
                [normalize_cluster_id(c) for c in df_results['meta_cluster'].unique() if normalize_cluster_id(c) is not None],
                key=cluster_sort_key,
            )
            
            assigned = sum(1 for v in phenotype_map.values() if v != 'Unassigned')
            total = len(phenotype_map)
            
            sum_col1, sum_col2 = st.columns([1, 2])
            with sum_col1:
                st.metric("Progress", f"{assigned}/{total}")
            with sum_col2:
                st.caption("Phenotypes: " + ", ".join([p for p in available_phenotypes if p != 'Unassigned'][:5]))
            
            st.divider()
            st.markdown("#### Edit Assignments")
            
            assign_df = pd.DataFrame([
                {
                    'Cluster': int(c) if pd.notna(c) else 0,
                    'Cells': int(cluster_stats.loc[c, 'n_cells']) if c in cluster_stats.index and pd.notna(cluster_stats.loc[c, 'n_cells']) else 0,
                    '%': float(cluster_stats.loc[c, 'percentage']) if c in cluster_stats.index and pd.notna(cluster_stats.loc[c, 'percentage']) else 0.0,
                    'Phenotype': phenotype_map.get(int(c) if pd.notna(c) else 0, 'Unassigned')
                }
                for c in meta_clusters
            ])
            
            # Form prevents reruns during editing
            with st.form("som_table_editor_form", clear_on_submit=False):
                
                edited_df = st.data_editor(
                    assign_df,
                    column_config={
                        "Cluster": st.column_config.NumberColumn(disabled=True),
                        "Cells": st.column_config.NumberColumn(disabled=True, format="%d"),
                        "%": st.column_config.NumberColumn(disabled=True, format="%.1f"),
                        "Phenotype": st.column_config.SelectboxColumn(
                            "Phenotype",
                            options=available_phenotypes,
                            required=True
                        )
                    },
                    hide_index=True,
                    num_rows="fixed",
                    key="som_table_editor",
                    width="stretch"
                )
                
                st.write("")
                st.markdown("##### Apply Changes")
                
                sync_col1, sync_col2 = st.columns([3, 1])
                with sync_col1:
                    st.caption("Edits above are local. Click Sync to apply to heatmap.")
                
                with sync_col2:
                    submitted = st.form_submit_button("Sync Changes", type="primary")
            
            if submitted:
                new_map = {int(row['Cluster']): row['Phenotype'] for _, row in edited_df.iterrows()}
                st.session_state['som_phenotype_map'] = new_map
                if 'som_heatmap_cache' in st.session_state:
                    del st.session_state['som_heatmap_cache']
                st.success("Changes synced!")
                st.rerun()
        
        # TAB 3: EXPORT RESULTS
        with tab_export:
            st.markdown("### Export Phenotyped Results")
            st.caption("Save final results with phenotype assignments (matches notebook Step 7)")
            
            phenotype_map = st.session_state.get('som_phenotype_map', {})
            # Map meta_cluster to phenotype, handling both int and float keys
            def map_phenotype(mc):
                if pd.isna(mc):
                    return 'Unassigned'
                try:
                    return phenotype_map.get(int(mc), 'Unassigned')
                except (ValueError, TypeError):
                    return 'Unassigned'
            
            df_results[output_col] = df_results['meta_cluster'].apply(map_phenotype)
            
            summary_stats = df_results[output_col].value_counts()
            
            export_col1, export_col2 = st.columns([1, 1])
            
            with export_col1:
                st.markdown("#### Phenotype Distribution")
                st.dataframe(
                    pd.DataFrame({
                        'Phenotype': summary_stats.index,
                        'Cells': summary_stats.values,
                        '% of Total': (summary_stats.values / len(df_results) * 100).round(2)
                    }).reset_index(drop=True),
                    hide_index=True,
                    width="stretch"
                )
            
            with export_col2:
                st.markdown("#### Export Options")
                
                remove_norm = st.checkbox(
                    "Remove _norm columns",
                    value=True,
                    help="Remove normalized marker columns from final CSV (keeps output clean)",
                    key="som_remove_norm"
                )
                
                output_filename = st.text_input(
                    "Output Filename",
                    "all_samples_combined_som_phenotyped.csv",
                    key="som_output_filename"
                )
            
            if st.button("Export Final Results", type="primary", key="som_export_final", width="stretch"):
                with st.spinner("Saving results..."):
                    df_export = df_results.copy()
                    
                    if remove_norm:
                        cols_to_drop = [c for c in df_export.columns if c.endswith('_norm')]
                        df_export = df_export.drop(columns=cols_to_drop)
                        st.caption(f"Removed {len(cols_to_drop)} normalized columns")
                    
                    final_out_dir = project_root / 'outputs/phenotyping'
                    final_out_dir.mkdir(parents=True, exist_ok=True)
                    final_path = final_out_dir / output_filename
                    
                    df_export.to_csv(final_path, index=False)
                    
                    st.success("Results exported successfully.")
                    st.code(f"Output file: {final_path}", language=None)
                    st.metric("Total Cells", len(df_export))
                    st.metric("Unique Phenotypes", df_export[output_col].nunique())


    # --- VISUAL VALIDATION COMPONENT (Ported from Leiden) ---
    if st.session_state.get('viz_enabled', True):
        st.divider()
        with st.expander("Visual Validation of Meta-Clusters", expanded=True):
            st.markdown("#### Visualize Representative Cells")
            st.caption("Inspect random cells from specific clusters to validate phenotype assignments.")
            
            # Ensure we have data
            if 'som_temp_file' in st.session_state and Path(st.session_state['som_temp_file']).exists():
                # Load data if not already
                if 'df_results' not in locals():
                    df_results = read_dataframe_parquet_safe(
                        st.session_state['som_temp_file'],
                        session_state_key='som_temp_file',
                        label='SOM results',
                    )
                    if df_results is None:
                        return
                
                # Check Global Viz Config
                viz_roi = st.session_state.get('viz_roi_dir')
                viz_seg = st.session_state.get('viz_seg_dir')
                
                if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
                    st.warning("Please configure valid ROI and Segmentation directories in Global Input Configuration (Top of Tab 7).")
                else:
                    viz = CellVisualizer(viz_roi, viz_seg)
                    
                    # File matching (cache in session state)
                    if not st.session_state.get('som_viz_file_matches'):
                         # Get unique batches
                         batch_col = st.session_state.get('som_batch_id_col', 'batch_id')
                         # Fallback search if not found
                         if batch_col not in df_results.columns:
                             batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                         
                         if batch_col in df_results.columns:
                             unique_batches = df_results[batch_col].unique().astype(str).tolist()
                             matches = viz.match_file_triads(unique_batches)
                             st.session_state.som_viz_file_matches = matches
                             if not matches:
                                 st.error("No matching image files found for these batches.")
                         else:
                             st.error(f"Batch column '{batch_col}' not found in results.")
                    else:
                        viz.file_map = st.session_state.som_viz_file_matches

                    # UI Controls
                    # 1. Cluster Selection (Meta-Cluster)
                    if 'meta_cluster' in df_results.columns:
                        avail_clusters = sorted([c for c in df_results['meta_cluster'].unique() if pd.notna(c)])
                        current_pheno_map = st.session_state.get('som_phenotype_map', {})
                        
                        # Options
                        cluster_options = {c: f"Meta-Cluster {int(c)} ({current_pheno_map.get(int(c), 'Unassigned')})" for c in avail_clusters}
                        
                        with st.form("som_viz_form"):
                             vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])
                             with vv_col1:
                                 selected_cluster_vv = st.selectbox(
                                     "Select Meta-Cluster", 
                                     avail_clusters,
                                     format_func=lambda c: cluster_options.get(c, str(c)),
                                     key="som_viz_cluster_sel"
                                 )
                             with vv_col2:
                                 n_viz_cells = st.number_input("Cells to View", 1, 50, 5, key="som_viz_n_cells")
                             with vv_col3:
                                 show_neighbors = st.checkbox("Show Neighbors", value=True, key="som_viz_neighbors")
                             with vv_col4:
                                 som_batch_scope = st.radio(
                                     "Batch Scope",
                                     ["All matched batches", "Specific batch"],
                                     horizontal=True,
                                     key="som_viz_batch_scope",
                                 )
                                 som_batch_options = sorted(str(batch) for batch in st.session_state.som_viz_file_matches.keys())
                                 selected_batch_vv = None
                                 if som_batch_scope == "Specific batch":
                                     selected_batch_vv = st.selectbox(
                                         "Batch",
                                         som_batch_options,
                                         key="som_viz_batch_sel",
                                     )
                                 st.write("") # Spacer
                                 run_viz = st.form_submit_button("Visualize Random Cells")
                        
                        # Logic
                        if run_viz and selected_cluster_vv is not None:
                            # 1. Determine Top 5 Markers
                            # SOM stores normalized data in _norm columns usually
                            # Try to find high expressing markers for this cluster
                            subset = df_results[df_results['meta_cluster'] == selected_cluster_vv]
                            
                            if len(subset) == 0:
                                st.warning("No cells in this cluster.")
                            else:
                                # Calculate means of _norm columns
                                norm_cols = [c for c in df_results.columns if c.endswith('_norm') and pd.api.types.is_numeric_dtype(df_results[c])]
                                if norm_cols:
                                    cluster_means = subset[norm_cols].mean()
                                    top_5_markers = cluster_means.nlargest(5).index.tolist()
                                    
                                    # Raw names
                                    top_5_names = [c.replace('_norm', '').replace('_corr', '').replace('_unmixed', '') for c in top_5_markers]
                                    st.info(f"Top markers for Meta-Cluster {int(selected_cluster_vv)}: {', '.join(top_5_names)}")
                                    
                                    # 2. Sample Cells (Group by batch)
                                    batch_col = st.session_state.get('som_batch_id_col', 'batch_id')
                                    if not batch_col or batch_col not in df_results.columns:
                                        batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                                    
                                    if batch_col not in df_results.columns:
                                        st.error(f"Batch column '{batch_col}' not found in results.")
                                        subset_valid = pd.DataFrame()
                                    elif not st.session_state.get('som_viz_file_matches'):
                                        st.error("No matching image files found for visualization batches.")
                                        subset_valid = pd.DataFrame()
                                    else:
                                        valid_batches = list(st.session_state.som_viz_file_matches.keys())
                                        subset_valid = subset[subset[batch_col].astype(str).isin(valid_batches)]
                                        if som_batch_scope == "Specific batch" and selected_batch_vv:
                                            subset_valid = subset_valid[subset_valid[batch_col].astype(str) == str(selected_batch_vv)]
                                    
                                    if len(subset_valid) == 0:
                                        st.warning("No cells found in batches with available images.")
                                    else:
                                        n_actual = min(n_viz_cells, len(subset_valid))
                                        sampled = subset_valid.sample(n=n_actual, random_state=int(time.time()))
                                        sampled = sampled.sort_values(batch_col)
                                        sampled_batches = sampled[batch_col].astype(str).drop_duplicates().tolist()
                                        prewarm_fn = getattr(viz, 'prewarm_batch', None)
                                        if callable(prewarm_fn):
                                            with st.spinner(f"Preloading {len(sampled_batches)} batch cache(s)..."):
                                                for prewarm_batch_id in sampled_batches:
                                                    prewarm_fn(prewarm_batch_id, requested_channels=top_5_markers)
                                        
                                        st.markdown("---")
                                        cols = st.columns(3)
                                        prog_bar = st.progress(0)
                                        
                                        x_col = st.session_state.get('som_cell_id_col', 'cell_id') # Usually used for ID, check config for X/Y
                                        # Wait, SOM config only tracks cell_id_col and batch_id_col.
                                        # X/Y columns might be standard 'x_centroid', 'y_centroid' or need auto-detection?
                                        # Leiden used st.session_state.get('viz_col_x', 'x_centroid').
                                        # I'll use defaults 'x_centroid', 'y_centroid' or try to detect 'AreaShape_Center_X' etc.
                                        
                                        x_col_name = next((c for c in ['x_centroid', 'AreaShape_Center_X', 'X_centroid', 'X'] if c in df_results.columns), 'x_centroid')
                                        y_col_name = next((c for c in ['y_centroid', 'AreaShape_Center_Y', 'Y_centroid', 'Y'] if c in df_results.columns), 'y_centroid')
                                        tile_sz = 40
                                        
                                        cid_col = st.session_state.get('som_cell_id_col', 'cell_id')

                                        for idx, (_, row) in enumerate(sampled.iterrows()):
                                            try:
                                                bid = str(row[batch_col])
                                                x, y = row[x_col_name], row[y_col_name]
                                                
                                                cell_data = viz.get_cell_context(
                                                    bid,
                                                    x,
                                                    y,
                                                    tile_size=tile_sz,
                                                    requested_channels=top_5_markers,
                                                )
                                                
                                                # Robust Name Matching (Ported Logic)
                                                ome_channels = cell_data.get('channel_names', [])
                                                marker_indices_for_plot = []
                                                names_for_plot = []
                                                
                                                if ome_channels:
                                                    ome_norm = [c.lower().strip() for c in ome_channels]
                                                    for m in top_5_markers:
                                                        raw_m = m.replace('_norm', '').replace('_corr', '').replace('_unmixed', '')
                                                        target = raw_m.lower().strip()
                                                        
                                                        found_idx = -1
                                                        if target in ome_norm:
                                                            found_idx = ome_norm.index(target)
                                                        elif any(target in c or c in target for c in ome_norm): # Fuzzy
                                                            # Find first match
                                                            for i, c in enumerate(ome_norm):
                                                                if target in c or c in target:
                                                                    found_idx = i
                                                                    break
                                                        
                                                        if found_idx != -1:
                                                            marker_indices_for_plot.append(found_idx)
                                                            names_for_plot.append(raw_m)
                                                
                                                if not marker_indices_for_plot:
                                                     # Fallback
                                                     marker_indices_for_plot = [0, 1, 2, 3, 4]
                                                     names_for_plot = [f"Ch{i}" for i in marker_indices_for_plot]
                                                
                                                fig = viz.generate_visualization(
                                                    cell_data, 
                                                    marker_indices_for_plot, 
                                                    names_for_plot,
                                                    display_size=tile_sz,
                                                    show_neighbors=show_neighbors
                                                )
                                                
                                                # Caption
                                                caption_txt = f"Cell: {row.get(cid_col, 'Unknown')}\nBatch: {bid}\nLoc: ({x:.1f}, {y:.1f})"
                                                
                                                with cols[idx % 3]:
                                                    st.pyplot(fig)
                                                    st.text(caption_txt)
                                                
                                                # Auto-Save
                                                try:
                                                    save_root = project_root / 'outputs/phenotyping/visual_validation'
                                                    save_root.mkdir(parents=True, exist_ok=True)
                                                    
                                                    clean_pheno = current_pheno_map.get(int(selected_cluster_vv), 'Unassigned').replace(' ', '_').replace('/', '-')
                                                    fname = f"{bid}_{row.get(cid_col, 'cell')}_MC{int(selected_cluster_vv)}_{clean_pheno}.png"
                                                    save_path = save_root / fname
                                                    fig.savefig(save_path, bbox_inches='tight', dpi=150)
                                                    
                                                    # Metadata CSV
                                                    csv_path = save_root / "visualized_cells_metadata.csv"
                                                    meta_rec = {
                                                        'timestamp': datetime.now().isoformat(),
                                                        'filename': fname,
                                                        'batch_id': bid,
                                                        'cell_id': row.get(cid_col, ''),
                                                        'cluster': selected_cluster_vv,
                                                        'phenotype': clean_pheno,
                                                        'x': x, 'y': y,
                                                        'top_markers': ", ".join(names_for_plot)
                                                    }
                                                    pd.DataFrame([meta_rec]).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                                                except Exception as e:
                                                    print(f"Auto-save error: {e}")
                                                
                                                plt.close(fig)
                                            except Exception as e:
                                                st.error(f"Error viewing cell: {e}")
                                            
                                            prog_bar.progress((idx + 1) / n_actual)
                                            
                                        st.success(f"Visualized {n_actual} cells. Output saved to `outputs/phenotyping/visual_validation/`")
                                else:
                                    st.warning("Could not identify top markers (no _norm columns found).")
                    else:
                        st.info("No 'meta_cluster' column found in results. Run SOM clustering first.")
