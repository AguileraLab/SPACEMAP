"""
Leiden Clustering Subtab
Professional Leiden clustering with notebook-synchronized parameters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Import shared utilities
from .common import (
    LOW_EXPRESSION_METRIC_OPTIONS,
    build_cluster_marker_score_map_from_heatmap,
    render_assisted_assignment,
    build_marker_source_note,
    create_session_temp_parquet_path,
    cluster_sort_key,
    get_batch_id_column_help_text,
    get_cell_id_display_value,
    get_cell_id_column_help_text,
    get_enable_row_filter_help_text,
    get_filter_column_help_text,
    get_filter_value_help_text,
    get_low_expression_default_threshold,
    get_low_expression_filter_help_text,
    get_low_expression_metric_help_text,
    get_low_expression_metric_label,
    get_marker_workspace_panel,
    render_method_marker_selection,
    get_normalization_flag_help_text,
    logger,
    _extract_markers_from_notebook,
    initialize_optional_column_widget_state,
    normalize_cluster_id,
    normalize_phenotype_map_keys,
    get_output_column_help_text,
    read_dataframe_parquet_safe,
    get_random_seed_help_text,
    build_leiden_heatmap_figure,
    apply_heatmap_assignment_event,
    consume_heatmap_assignment_event,
    get_heatmap_assignment_scope,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_export_split_data,
    get_all_phenotype_options,
    render_add_custom_phenotype_ui,
    render_settings_export_import,
    render_method_intro,
    begin_phenotyping_run_audit,
    resolve_notebook_marker_defaults,
    write_dataframe_parquet_atomic,
)
from .heatmap_click_selector import (
    heatmap_click_selector_available,
    render_heatmap_click_selector,
)
from spacemap_workflows.leiden_job import latest_job, read_job, read_log, start_job
try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

import plotly.graph_objects as go
from spacemap_gui import terminal

try:
    from spacemap import ph_leiden_utils
    sys.modules["ph_leiden_utils"] = ph_leiden_utils
except ImportError:
    try:
        import ph_leiden_utils
    except Exception:
        ph_leiden_utils = None


LEIDEN_LOW_EXPR_CLUSTER_ID = -1
LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE = "Others"
LEIDEN_LOW_EXPR_LABEL = "Low expression bucket"


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab7', message, step='Leiden', level=level)


def _dataio_load(path, columns=None):
    """Read the unified table through the shared loader.

    Uses the Parquet fast copy when one is current, so a large table loads in
    seconds; falls back to the CSV otherwise. Passing *columns* reads only
    those.
    """
    from spacemap import dataio

    return dataio.load(path, columns=columns)


def _is_leiden_low_expression_cluster(cluster_id, low_expr_cluster_id) -> bool:
    if low_expr_cluster_id is None:
        return False
    return normalize_cluster_id(cluster_id) == normalize_cluster_id(low_expr_cluster_id)


def _get_leiden_low_expression_mask(df_results: pd.DataFrame) -> pd.Series:
    if 'leiden_low_expression' in df_results.columns:
        return df_results['leiden_low_expression'].fillna(False).astype(bool)
    if 'leiden_cluster' not in df_results.columns:
        return pd.Series(False, index=df_results.index)
    return df_results['leiden_cluster'].apply(
        lambda value: _is_leiden_low_expression_cluster(value, LEIDEN_LOW_EXPR_CLUSTER_ID)
    )


def _get_leiden_low_expression_note(metric: str | None, threshold: float | None) -> str:
    if threshold is None:
        return "Preserved outside clustering"
    return (
        f"{get_low_expression_metric_label(metric or 'mean_zscore').lower()} <= {float(threshold):.4g}; "
        "preserved outside clustering"
    )


@st.fragment(run_every=2.0)
def _render_running_leiden_job(job_dir: str):
    job = read_job(job_dir)
    if job["status"] not in {"starting", "running"}:
        st.rerun()
    st.info("Leiden is running. You can refresh this page or return later; the run will continue.")
    st.code(read_log(job_dir) or "Starting Leiden…", language=None)
    st.caption(f"Run files: {job_dir}")


def _restore_leiden_job_result(job, full_input_path):
    """Attach a completed run once per session, including after reconnects."""
    if job["status"] != "success":
        return
    identity = job["job_dir"]  # Includes project identity, not just a session key.
    if st.session_state.get("_leiden_applied_job") == identity:
        return
    result_path = Path(identity) / "result.json"
    result = json.loads(result_path.read_text())
    if not full_input_path or Path(result["leiden_run_input_path"]).resolve() != Path(full_input_path).resolve():
        st.info("A completed Leiden run is saved. Select its input file to load it: "
                + result["leiden_run_input_path"])
        return
    if not Path(result["leiden_temp_file"]).is_file():
        st.error("The saved Leiden result file is missing. Run clustering again to recreate it.")
        return
    if (st.session_state.get("leiden_run_id") == result["leiden_run_id"]
            and st.session_state.get("leiden_temp_file") == result["leiden_temp_file"]):
        # Project persistence may already have restored the user's subsequent
        # phenotype edits; do not replace them with the initial Unassigned map.
        st.session_state["_leiden_applied_job"] = identity
        return
    result["leiden_phenotype_map"] = normalize_phenotype_map_keys(result["leiden_phenotype_map"])
    for key in ("leiden_available_phenotypes", "leiden_selected_clusters",
                "leiden_heatmap_component_event_id"):
        st.session_state.pop(key, None)
    st.session_state.update(result)
    st.session_state["_leiden_applied_job"] = identity


def _autoload_existing_leiden_from_csv(global_full_path: str | None, project_root: Path) -> None:
    """Auto-load Leiden results from unified CSV (leiden_cluster + phenotype_leiden).

    This mirrors the previous global auto-load, but only runs inside Leiden subtab.
    """
    if not global_full_path or not os.path.exists(global_full_path):
        return

    current_loaded = st.session_state.get('leiden_loaded_source_path', None)
    if current_loaded == global_full_path:
        return

    try:
        df_sample = pd.read_csv(global_full_path, nrows=5)
        lid_col = "leiden_cluster"
        found_leiden = lid_col in df_sample.columns
        found_pheno = "phenotype_leiden" in df_sample.columns

        if not (found_leiden and found_pheno):
            st.session_state['leiden_loaded_source_path'] = global_full_path
            return

        st.info("Found existing Leiden clustering results. Loading...")

        from spacemap.ph_accel_utils import read_csv_fast
        df_full = read_csv_fast(global_full_path)

        existing_map = {}
        pairs = df_full[[lid_col, 'phenotype_leiden']].drop_duplicates().dropna()
        for _, row in pairs.iterrows():
            existing_map[row[lid_col]] = row['phenotype_leiden']

        st.session_state['leiden_phenotype_map'] = existing_map

        metadata_cols = [
            'cell_id', 'batch_id', 'tissue', 'x_centroid', 'y_centroid',
            'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
            'major_axis', 'minor_axis', 'orientation', 'label', 'index'
        ]
        metadata_cols_lower = [c.lower() for c in metadata_cols]
        marker_candidates = [
            c for c in df_full.columns
            if c.lower() not in metadata_cols_lower
            and not str(c).startswith('umap_')
            and not str(c).startswith('leiden_')
            and not str(c).startswith('phenotype_')
            and not str(c).endswith('_norm')
        ]
        numeric_markers = [c for c in marker_candidates if pd.api.types.is_numeric_dtype(df_full[c])]
        st.session_state['leiden_markers'] = numeric_markers
        st.session_state['leiden_norm_cols'] = numeric_markers

        temp_file = create_session_temp_parquet_path(project_root, "leiden_autoload")
        write_dataframe_parquet_atomic(df_full, temp_file)

        st.session_state['leiden_temp_file'] = str(temp_file)
        st.session_state['leiden_n_clusters'] = df_full[lid_col].nunique()
        st.session_state['leiden_loaded_source_path'] = global_full_path

        st.success(f"Success! Loaded {len(existing_map)} phenotypes from file.")
    except Exception as e:
        logger.warning(f"Auto-load check failed: {e}")
        st.session_state['leiden_loaded_source_path'] = global_full_path

LEIDEN_GUIDE = """\
**What it does.** Groups cells with similar marker levels. Each cell is linked to the cells most like it, and Leiden finds tightly connected groups (clusters) in that network. It does not know cell types: you name the clusters afterwards.

**Steps**
1. Choose the input table and the markers to cluster on.
2. Set the parameters. **Resolution** matters most: higher gives more, smaller clusters; lower gives fewer, larger ones.
3. Run. Large tables run in the background, so you can leave the tab while it works.
4. Under **Results**, read the heatmap (average marker level per cluster) and give each cluster a phenotype name. Clusters given the same name are merged.
5. Save the names into the table, then check example cells under **Visual Validation**.

**Tip.** Too many clusters is easier to fix than too few: several clusters can share a name, but one cluster cannot be split here.
"""


def render_leiden(project_root, global_input_path):
    render_method_intro(
        project_root, "leiden", "Leiden Phenotyping",
        "Group similar cells into clusters, then name each cluster as a phenotype.",
        LEIDEN_GUIDE, notebook="Step6A_phenotype_leiden.ipynb", log_label="Leiden Clustering",
    )
    job = latest_job(project_root)
    job_running = bool(job and job["status"] in {"starting", "running"})
    
    # Initialize cache file for parameter persistence
    cache_dir = project_root / 'outputs/phenotyping/cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    leiden_cache = cache_dir / 'leiden_params.json'
    
    # Load parameters from cache if available
    import json
    if leiden_cache.exists():
        with open(leiden_cache, 'r') as f:
            cached_params = json.load(f)
    else:
        cached_params = {}

    # --- Settings Export / Import ---
    render_settings_export_import(
        "Leiden Clustering",
        ["leiden_"],
        {"leiden_params.json": leiden_cache},
    )

    def _apply_norm_for_heatmap(df_in: pd.DataFrame, markers: list[str]) -> tuple[pd.DataFrame, list[str]]:
        # Check specific markers availability
        valid_markers = [m for m in markers if m in df_in.columns]
        
        # Determine active cofactor
        cofactor = 5.0
        if 'bc_config' in st.session_state:
            cofactor = float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0))
            
        use_sinh = st.session_state.get('leiden_sinh', True)
        df_input = df_in
        if use_sinh:
            df_input = df_in.copy()
            for m in valid_markers:
                if m.endswith('_corr'):
                    df_input[m] = np.sinh(df_input[m]) * cofactor
        
        if ph_leiden_utils:
            # Use notebook-aligned normalization with batch support
            group_cols = [batch_id_col] if batch_id_col in df_input.columns else []
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

    # --- FILE SELECTION WITH PERSISTENCE ---
    st.markdown("### 1. Input File")
    
    # Load cached directory
    cached_input_dir = cached_params.get('input_dir', str(project_root / 'outputs/batch_corrected'))
    cached_input_file = cached_params.get('input_file', '')
    
    use_global = st.checkbox("Use Global Input File", value=cached_params.get('use_global', True), key="leiden_use_global")
    
    # Initialize variables
    input_dir = cached_input_dir
    selected_file = cached_input_file
    full_input_path = global_input_path
    
    if not use_global:
        # Directory input with cached default
        input_dir = st.text_input(
            "Input Directory",
            value=cached_input_dir,
            help="Directory containing batch-corrected CSV files",
            key="leiden_input_dir"
        )
        
        # List files in directory
        if input_dir and os.path.exists(input_dir):
            available_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        else:
            available_files = []
        
        if available_files:
            # Find cached file in list or default to first
            default_idx = 0
            if cached_input_file in available_files:
                default_idx = available_files.index(cached_input_file)
            
            selected_file = st.selectbox(
                "Input File",
                available_files,
                index=default_idx,
                key="leiden_infile_select"
            )
            full_input_path = os.path.join(input_dir, selected_file)
        else:
            st.warning(f"No CSV files found in: {input_dir}")
            full_input_path = None
    
    if full_input_path:
        st.caption(f"Active File: {os.path.basename(full_input_path)}")
                    
    # Discover ALL columns from CSV (for marker selection)
    available_columns = []
    all_marker_candidates = []
    
    if full_input_path and os.path.exists(full_input_path):
        try:
            sample_df = pd.read_csv(full_input_path, nrows=5)
            available_columns = list(sample_df.columns)
            
            # Get all potential marker columns (exclude metadata columns)
            metadata_cols = ['cell_id', 'batch_id', 'tissue', 'x_centroid', 'y_centroid', 
                           'area', 'perimeter', 'eccentricity', 'solidity', 'extent',
                           'major_axis', 'minor_axis', 'orientation', 'label', 'index']
            metadata_cols_lower = [c.lower() for c in metadata_cols]
            
            # Include ALL potential markers (both _corr and non-_corr versions)
            all_marker_candidates = [c for c in available_columns 
                                    if c.lower() not in metadata_cols_lower 
                                    and not c.startswith('umap_')
                                    and not c.startswith('leiden_')
                                    and not c.startswith('phenotype_')
                                    and not c.endswith('_norm')]
            
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    available_marker_columns = list(all_marker_candidates)
    all_marker_candidates, marker_workspace_note = get_marker_workspace_panel(
        all_marker_candidates,
        full_input_path,
    )
    notebook_corr_markers, notebook_non_corr_markers = _extract_markers_from_notebook(
        "Step6A_phenotype_leiden.ipynb"
    )
    marker_workspace = st.session_state.get("tab7_marker_workspace", {})
    expected_markers = resolve_notebook_marker_defaults(
        notebook_corr_markers,
        notebook_non_corr_markers,
        all_marker_candidates,
        str(marker_workspace.get("source_mode", "Raw markers")),
        str(marker_workspace.get("affix_position", "Suffix")),
        str(marker_workspace.get("correction_affix", "_corr")),
    )
    st.markdown("### 2. Marker Selection")
    _, markers_selected, marker_workspace_note = render_method_marker_selection(
        key_prefix="leiden",
        method_name="Leiden",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers,
        saved_markers=cached_params.get("markers", []),
    )
    
    # --- CONFIGURATION PANEL ---
    st.markdown("### 3. Configuration")
    with st.expander("Parameters", expanded=True):
        with st.form("leiden_config_form", clear_on_submit=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("##### Column Configuration")
                
                # Cell ID Column Selection
                default_cell_id = 'cell_id' if 'cell_id' in available_columns else (available_columns[0] if available_columns else 'cell_id')
                cell_id_options = [""] + available_columns if available_columns else [""]
                initialize_optional_column_widget_state(
                    "leiden_cell_id_col",
                    available_columns,
                    cached_params.get('cell_id_col', default_cell_id),
                )
                cell_id_col = st.selectbox(
                    "Cell ID Column", 
                    cell_id_options,
                    format_func=lambda value: value or "(None)",
                    help=get_cell_id_column_help_text(),
                    key="leiden_cell_id_col"
                )

                # Batch ID Column Selection (New Feature)
                batch_id_options = [""] + available_columns if available_columns else [""]
                initialize_optional_column_widget_state(
                    "leiden_batch_id_col",
                    available_columns,
                    cached_params.get('batch_id_col', ''),
                )
                idx_batch_id = batch_id_options.index(st.session_state.get("leiden_batch_id_col", "")) if st.session_state.get("leiden_batch_id_col", "") in batch_id_options else 0
                
                batch_id_col = st.selectbox(
                    "Batch ID Column",
                    batch_id_options,
                    format_func=lambda value: value or "(None)",
                    help=get_batch_id_column_help_text(),
                    key="leiden_batch_id_col"
                )
                
                # Filter Configuration
                st.markdown("**Filtering Criteria**")
                enable_filter = st.checkbox(
                    "Enable Row Filtering",
                    value=cached_params.get('enable_filter', False),
                    help=get_enable_row_filter_help_text(),
                    key="leiden_enable_filter",
                )
                
                if enable_filter:
                    # Auto-select 'tissue' if available
                    filter_cols = [c for c in available_columns if c not in [cell_id_col]]
                    default_filter_col = 'tissue' if 'tissue' in filter_cols else (filter_cols[0] if filter_cols else 'tissue')
                    cached_filter_col = cached_params.get('filter_col', default_filter_col)
                    if cached_filter_col in filter_cols:
                        default_filter_idx = filter_cols.index(cached_filter_col)
                    else:
                        default_filter_idx = filter_cols.index(default_filter_col) if default_filter_col in filter_cols else 0
                    
                    filter_col = st.selectbox(
                        "Filter Column",
                        filter_cols if filter_cols else [default_filter_col],
                        index=default_filter_idx,
                        help=get_filter_column_help_text(),
                        key="leiden_filter_col"
                    )
                    
                    filter_val = st.text_input(
                        "Filter Value", 
                        cached_params.get('filter_val', "1"),
                        help=get_filter_value_help_text(),
                        key="leiden_filter_val"
                    )
                else:
                    filter_col = None
                    filter_val = None
            
            with col2:
                st.markdown("##### Leiden Clustering Parameters")
                st.caption("All parameters synchronized with Step6A notebook")
                
                # Resolution
                resolution = st.slider(
                    "Resolution", 
                    0.1, 5.0, 
                    cached_params.get('resolution', 2.8), 
                    0.1,
                    help=(
                        "Main Leiden granularity control. Higher values split the graph into more, smaller clusters; lower values merge cells into fewer, broader groups. "
                        "Start lower when you want high-level phenotypes and raise it when biologically distinct populations are being merged together."
                    ),
                    key="leiden_resolution"
                )
                
                # Neighbors & PCs
                p1, p2 = st.columns(2)
                n_neighbors = p1.number_input(
                    "N Neighbors", 
                    5, 100, 
                    cached_params.get('n_neighbors', 15),
                    help=(
                        "Number of neighbors used to build the cell-cell graph before Leiden. Smaller values emphasize local structure and can separate rarer populations; larger values smooth the graph and favor broader, more stable clusters. "
                        "This setting affects both the direct Scanpy path and the sketch-based large-dataset path."
                    ),
                    key="leiden_n_neighbors"
                )
                n_pcs = p2.number_input(
                    "N PCs", 
                    5, 100, 
                    cached_params.get('n_pcs', 15),
                    help=(
                        "Number of principal components carried into neighbor-graph construction. Too few PCs can collapse real biology; too many can reintroduce noise. "
                        "With focused lineage-marker panels, choose enough PCs to capture marker structure without chasing minor variation."
                    ),
                    key="leiden_n_pcs"
                )
                
                # Target clusters (informational)
                n_clusters_target = st.number_input(
                    "Target Clusters",
                    5, 50,
                    cached_params.get('n_clusters_target', 45),
                    help=(
                        "Reference value only. This number is saved in the UI state and shown in the run summary, but the code does not use it to constrain Leiden. "
                        "Use it as a reminder of the cluster range you hope to see, not as a hard target."
                    ),
                    key="leiden_n_clusters_target"
                )
                
                # Performance Settings
                st.markdown("**Performance Options**")
                subsample_size = st.number_input(
                    "Subsample Size", 
                    10000, 1000000, 
                    cached_params.get('subsample_size', 200000), 
                    10000,
                    help=(
                        "Maximum sketch size used by the intelligent large-dataset Leiden path. That path clusters a representative subset first and then projects labels back to all cells. "
                        "Larger sketches are slower but usually stay closer to exact full-data clustering."
                    ),
                    key="leiden_subsample"
                )
                
                # Intelligent
                use_intelligent = st.checkbox(
                    "Use Intelligent Clustering",
                    value=cached_params.get('use_intelligent', True),
                    help=(
                        "Use the sketch-based large-dataset Leiden path when possible. In this code it only changes behavior when the dataset is larger than 50k cells and the shared Leiden utility is available; smaller runs still fall back to direct Scanpy Leiden automatically. "
                        "Turn it off only when you explicitly want exact full-data clustering despite the runtime cost."
                    ),
                    key="leiden_use_intelligent"
                )
                
                # UMAP is visualisation only and never affects cluster assignments,
                # but on large datasets it costs more than the clustering itself.
                compute_umap = st.checkbox(
                    "Compute UMAP embedding",
                    value=cached_params.get('compute_umap', False),
                    help=(
                        "Build a 2D UMAP for the UMAP Visualization tab. This is for visualisation and QC "
                        "only - it does not change the cluster assignments or any saved phenotype. On large "
                        "datasets it often takes longer than the clustering itself, so it is off by default. "
                        "Turn it on when you actually want to inspect the embedding."
                    ),
                    key="leiden_compute_umap"
                )

                meta_clusters = st.number_input(
                    "Merge into N meta-clusters (0 = off)",
                    0, 2000,
                    int(cached_params.get('meta_clusters', 0)),
                    10,
                    help=(
                        "After clustering, merge clusters with similar marker profiles until "
                        "this many remain (hierarchical/Ward merge on cluster means, the same "
                        "step the SOM method applies to its nodes).\n\n"
                        "Prefer tuning resolution first: it re-derives the community structure, whereas merging groups clusters by centroid distance alone. Leiden usually lands in a "
                        "workable range on its own, so this is mainly for matching granularity "
                        "across methods before a consensus vote, where a method that is far more "
                        "fine-grained than the others would otherwise fail to agree with them.\n\n"
                        "Set 0 to keep every cluster. Merging too far starts combining "
                        "genuinely different cell types."
                    ),
                    key="leiden_meta_clusters"
                )

                _gpu_name = None
                try:
                    from spacemap.ph_accel_utils import gpu_device_name as _gpu_device_name

                    _gpu_name = _gpu_device_name()
                except Exception:
                    pass
                use_gpu = st.checkbox(
                    "Use GPU when available",
                    value=cached_params.get('use_gpu', True),
                    help=(
                        "Leiden clusters a representative sample of cells, then gives every other cell the "
                        "cluster of its most similar sampled cells (nearest neighbours). This option runs "
                        "that matching step on the GPU with an exact search. Results match the CPU path; "
                        "only the speed differs. Falls back to CPU automatically when no GPU is present. "
                        "The Leiden graph partitioning itself still runs on the CPU."
                    ),
                    key="leiden_use_gpu"
                )
                st.caption(
                    f"GPU detected: {_gpu_name}" if _gpu_name
                    else "No GPU detected; the CPU path will be used automatically."
                )

                # Seed
                random_seed = st.number_input(
                    "Random Seed",
                    0, 10000,
                    cached_params.get('random_seed', 42),
                    help=get_random_seed_help_text("PCA initialization, graph construction, Leiden, UMAP, and sketch sampling"),
                    key="leiden_random_seed"
                )
                
                # Normalization
                st.markdown("**Normalization Flags**")
                n1, n2, n3, n4 = st.columns(4)
                norm_z1 = n1.checkbox("Z1", cached_params.get('norm_z1', True), help=get_normalization_flag_help_text("z1"), key="leiden_z1")
                norm_z2 = n2.checkbox("Z2", cached_params.get('norm_z2', False), help=get_normalization_flag_help_text("z2"), key="leiden_z2")
                norm_cdf = n3.checkbox("CDF", cached_params.get('norm_cdf', False), help=get_normalization_flag_help_text("cdf"), key="leiden_cdf")
                norm_nlog = n4.checkbox("NLog", cached_params.get('norm_nlog', False), help=get_normalization_flag_help_text("nlog"), key="leiden_nlog")
                
                norm_sinh = st.checkbox(
                    "Inverse ArcSinh (Sinh) for Corrected Markers", 
                    cached_params.get('norm_sinh', True), 
                    help="Automatically convert batch-corrected (_corr) markers back to raw scale using sinh(val) * cofactor before normalization to maintain high-contrast 'ice heatmap' visualization.",
                    key="leiden_sinh"
                )
                
                # Output
                st.markdown("**Output Configuration**")
                
                # Low Expression Filtering
                enable_low_expr_filter = st.checkbox(
                    "Filter Low-Expression Cells",
                    value=cached_params.get('enable_low_expr_filter', True),
                    help=get_low_expression_filter_help_text("Leiden clustering"),
                    key="leiden_low_expr_filter"
                )

                low_expr_metric = cached_params.get('low_expr_metric', 'mean_zscore')
                if low_expr_metric not in LOW_EXPRESSION_METRIC_OPTIONS:
                    low_expr_metric = 'mean_zscore'

                if enable_low_expr_filter:
                    low_expr_metric = st.selectbox(
                        "Low-Expression Metric",
                        options=list(LOW_EXPRESSION_METRIC_OPTIONS),
                        index=list(LOW_EXPRESSION_METRIC_OPTIONS).index(low_expr_metric),
                        format_func=get_low_expression_metric_label,
                        help=get_low_expression_metric_help_text(),
                        key="leiden_low_expr_metric",
                    )
                    default_thresh = cached_params.get('low_expr_threshold')
                    if default_thresh is None:
                        default_thresh = cached_params.get('min_marker_sum')
                    if default_thresh is None:
                        default_thresh = get_low_expression_default_threshold(low_expr_metric)
                    low_expr_threshold = st.number_input(
                        "Low-Expression Threshold",
                        -10.0,
                        100000.0,
                        float(default_thresh),
                        0.05 if low_expr_metric == 'mean_zscore' else 0.1,
                        help=(
                            "Cells at or below this cutoff are kept in the output but excluded from the main Leiden run and placed in the preserved low-expression bucket. "
                            "For mean marker z-score, start around -0.75."
                            if low_expr_metric == 'mean_zscore'
                            else "Cells at or below this raw selected-marker sum are kept in the output but excluded from the main Leiden run and placed in the preserved low-expression bucket."
                        ),
                        key="leiden_low_expr_threshold"
                    )
                    if low_expr_metric == 'mean_zscore':
                        st.caption("Recommended starting cutoff: -0.75. Lower values are more permissive; values closer to 0 remove more panel-dim cells.")
                    else:
                        st.caption("Raw-sum mode uses the pre-normalization selected markers. A cutoff of 0.1 is usually only near-zero signal.")
                else:
                    low_expr_threshold = get_low_expression_default_threshold(low_expr_metric)

                output_col_name = st.text_input(
                    "Output Column Name",
                    cached_params.get('output_col', "phenotype_leiden"),
                    help=get_output_column_help_text(),
                    key="leiden_outcol"
                )
        
            # Save Parameters Button
            save_col1, save_col2 = st.columns([1, 3])
            with save_col1:
                if st.form_submit_button("Save Parameters", type="primary"):
                    params_to_save = {
                        'input_dir': input_dir,
                        'input_file': selected_file,
                        'use_global': use_global,
                        'markers': markers_selected,
                        'cell_id_col': cell_id_col,
                        'batch_id_col': batch_id_col,
                        'resolution': resolution,
                        'n_neighbors': n_neighbors,
                        'n_pcs': n_pcs,
                        'n_clusters_target': n_clusters_target,
                        'subsample_size': subsample_size,
                        'use_intelligent': use_intelligent,
                        'compute_umap': compute_umap,
                        'meta_clusters': int(meta_clusters),
                        'use_gpu': use_gpu,
                        'random_seed': random_seed,
                        'norm_z1': norm_z1,
                        'norm_z2': norm_z2,
                        'norm_cdf': norm_cdf,
                        'norm_nlog': norm_nlog,
                        'norm_sinh': norm_sinh,
                        'low_expr_metric': low_expr_metric,
                        'low_expr_threshold': float(low_expr_threshold) if low_expr_threshold is not None else 0.0,
                        'min_marker_sum': float(low_expr_threshold) if low_expr_threshold is not None else 0.0,
                        'enable_low_expr_filter': enable_low_expr_filter,
                        'output_col': output_col_name,
                        'enable_filter': enable_filter,
                        'filter_col': filter_col if enable_filter else '',
                        'filter_val': filter_val if enable_filter else ''
                    }
                    with open(leiden_cache, 'w') as f:
                        json.dump(params_to_save, f, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                    st.success("Parameters saved to cache and project.")
                    st.rerun()
            
            with save_col2:
                st.caption(f"Parameters will be persisted in: `{leiden_cache.name}`")

    # --- LOAD EXISTING RESULTS / START FRESH ---
    if full_input_path and os.path.exists(full_input_path):
        st.divider()
        st.markdown("### 4. Load Previous Results (optional)")

        can_load_existing = False
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            can_load_existing = (
                "leiden_cluster" in df_head.columns
                and output_col_name in df_head.columns
            )
        except Exception:
            can_load_existing = False

        c1, c2 = st.columns([1, 1])
        with c1:
            load_existing = st.button(
                "Load Existing Leiden Results",
                disabled=(job_running or not can_load_existing or not markers_selected),
                help="Load phenotype_leiden + leiden_cluster from the unified CSV and rebuild the heatmap with selected markers.",
                key="leiden_load_existing_btn",
            )
        with c2:
            start_fresh = st.button(
                "Start Fresh (Ignore Existing)",
                disabled=job_running,
                help="Clear loaded Leiden results and start a new clustering run.",
                key="leiden_start_fresh_btn",
            )

        if start_fresh:
            if job:
                st.session_state["_leiden_applied_job"] = job["job_dir"]
            for key in [
                "leiden_phenotype_map",
                "leiden_available_phenotypes",
                "leiden_temp_file",
                "leiden_loaded_hash",
                "leiden_selected_clusters",
                "leiden_heatmap_component_event_id",
                "leiden_heatmap_component_reset_counter",
                "leiden_has_run",
                "leiden_run_input_path",
                "leiden_run_id",
                "leiden_low_expr_cluster_id",
                "leiden_low_expr_count",
                "leiden_low_expr_metric",
                "leiden_low_expr_threshold",
                "leiden_low_expr_default_pheno",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state['leiden_fresh_assignment_mode'] = True
            st.session_state['leiden_heatmap_component_reset_counter'] = (
                int(st.session_state.get('leiden_heatmap_component_reset_counter', 0)) + 1
            )
            st.success("Leiden state cleared. Ready for a fresh run.")

        if load_existing:
            if job:
                st.session_state["_leiden_applied_job"] = job["job_dir"]
            try:
                st.session_state['leiden_fresh_assignment_mode'] = False
                df_full = _dataio_load(full_input_path)
                cluster_col = "leiden_cluster"
                pheno_col = output_col_name

                norm_df, norm_cols_loaded = _apply_norm_for_heatmap(df_full, markers_selected)
                low_expr_mask = _get_leiden_low_expression_mask(norm_df)
                low_expr_count = int(low_expr_mask.sum())
                clustered_df = norm_df[~low_expr_mask].copy()

                pairs = norm_df[[cluster_col, pheno_col]].drop_duplicates().dropna()
                existing_map = {}
                for _, row in pairs.iterrows():
                    cluster_id = normalize_cluster_id(row[cluster_col])
                    if cluster_id is not None:
                        existing_map[cluster_id] = row[pheno_col]
                if low_expr_count and existing_map.get(LEIDEN_LOW_EXPR_CLUSTER_ID) in {None, 'Unassigned'}:
                    existing_map[LEIDEN_LOW_EXPR_CLUSTER_ID] = LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE

                temp_file = create_session_temp_parquet_path(project_root, "leiden_existing")
                write_dataframe_parquet_atomic(norm_df, temp_file)

                st.session_state['leiden_temp_file'] = str(temp_file)
                st.session_state['leiden_markers'] = markers_selected
                st.session_state['leiden_norm_cols'] = norm_cols_loaded
                st.session_state['leiden_n_clusters'] = clustered_df[cluster_col].nunique() if cluster_col in clustered_df.columns else 0
                st.session_state['leiden_phenotype_map'] = existing_map
                st.session_state['leiden_available_phenotypes'] = [
                    'Unassigned'
                ] + sorted({p for p in norm_df[pheno_col].dropna().unique().tolist() if str(p) != 'Unassigned'})
                st.session_state['leiden_run_id'] = f"{time.time():.6f}"
                st.session_state['leiden_low_expr_cluster_id'] = LEIDEN_LOW_EXPR_CLUSTER_ID if low_expr_count else None
                st.session_state['leiden_low_expr_count'] = low_expr_count
                st.session_state['leiden_low_expr_metric'] = None
                st.session_state['leiden_low_expr_threshold'] = None
                st.session_state['leiden_low_expr_default_pheno'] = LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE
                st.session_state['leiden_heatmap_component_event_id'] = None
                st.session_state['leiden_heatmap_component_reset_counter'] = (
                    int(st.session_state.get('leiden_heatmap_component_reset_counter', 0)) + 1
                )
                st.success("Loaded existing Leiden results from unified CSV.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load existing Leiden results: {e}")
    
    # --- ACTION: RUN CLUSTERING ---
    st.divider()
    
    # Show summary before run
    n_markers = len(markers_selected)
    if n_markers > 0:
        st.info(f"Ready to run with **{n_markers}** markers | Resolution: **{resolution}** | Target clusters: ~**{n_clusters_target}**")
    else:
        st.warning("Please select at least one marker to proceed")

    existing_run_log = logger.get_streamlit_buffer("leiden_run_log_lines")
    if existing_run_log:
        with st.expander("Last Leiden Tab Terminal", expanded=False):
            st.caption("Timestamped messages mirrored from the shell terminal during the last Leiden run.")
            st.code("\n".join(existing_run_log), language=None)
    
    st.markdown("### 5. Run Leiden Clustering")
    if st.button("Run Leiden Clustering", type="primary", disabled=(n_markers == 0 or job_running)):
        if not full_input_path or not os.path.exists(full_input_path):
            st.error("Please select a valid input file.")
            return
        settings = {
            'full_input_path': full_input_path,
            'markers_selected': markers_selected,
            'cell_id_col': cell_id_col,
            'batch_id_col': batch_id_col,
            'enable_filter': enable_filter,
            'filter_col': filter_col,
            'filter_val': filter_val,
            'norm_z1': norm_z1,
            'norm_z2': norm_z2,
            'norm_cdf': norm_cdf,
            'norm_nlog': norm_nlog,
            'norm_sinh': norm_sinh,
            'low_expr_metric': low_expr_metric,
            'enable_low_expr_filter': enable_low_expr_filter,
            'low_expr_threshold': low_expr_threshold,
            'resolution': resolution,
            'n_neighbors': n_neighbors,
            'n_pcs': n_pcs,
            'use_intelligent': use_intelligent,
            'subsample_size': subsample_size,
            'random_seed': random_seed,
            'compute_umap': compute_umap,
            'meta_clusters': meta_clusters,
            'use_gpu': use_gpu,
            'output_col_name': output_col_name,
            'cofactor': float(st.session_state.get('bc_config', {}).get('arcsinh_cofactor', 5.0)),
            'fresh_assignment_mode': bool(st.session_state.get('leiden_fresh_assignment_mode', False)),
        }
        run_audit = begin_phenotyping_run_audit(
            Path(project_root), "leiden", input_csv=full_input_path, markers=markers_selected,
        )
        try:
            job = start_job(project_root, settings, run_audit)
        except Exception as exc:
            st.error(f"Could not start Leiden: {exc}")
            return
        st.rerun()

    if job:
        if job_running:
            _render_running_leiden_job(job["job_dir"])
            # Avoid displaying an earlier run's assignments as current results.
            return
        log_text = read_log(job["job_dir"])
        log_lines = log_text.splitlines()[-400:]
        if st.session_state.get("leiden_run_log_lines") != log_lines:
            st.session_state["leiden_run_log_lines"] = log_lines
        if job["status"] == "success":
            _restore_leiden_job_result(job, full_input_path)
            st.success("Leiden clustering completed. Results are saved.")
        elif job["status"] == "interrupted":
            st.warning(job.get("error", "The Leiden process was interrupted."))
        else:
            st.error(job.get("error", "Leiden clustering failed. See the saved terminal log."))
        if job["status"] != "success" and st.session_state.get("leiden_temp_file"):
            st.caption("Results shown below belong to an earlier completed run.")
        with st.expander("Saved Leiden Run", expanded=job["status"] != "success"):
            st.code(log_text or "No worker output was recorded.", language=None)
            st.caption(f"Run files: {job['job_dir']}")

    # --- INTERACTIVE VISUALIZATION & PHENOTYPE ASSIGNMENT ---
    if 'leiden_temp_file' in st.session_state and Path(st.session_state['leiden_temp_file']).exists():
        st.divider()
        st.subheader("Results & Phenotype Assignment")
        st.caption("Professional phenotype assignment interface synchronized with notebook workflow")
        
        temp_file = st.session_state['leiden_temp_file']
        df_results = read_dataframe_parquet_safe(
            temp_file,
            session_state_key='leiden_temp_file',
            label='Leiden results',
        )
        if df_results is None:
            return
        norm_cols = st.session_state.get('leiden_norm_cols', [])
        low_expr_cluster_id = st.session_state.get('leiden_low_expr_cluster_id')
        if low_expr_cluster_id is None and 'leiden_low_expression' in df_results.columns and df_results['leiden_low_expression'].fillna(False).any():
            low_expr_cluster_id = LEIDEN_LOW_EXPR_CLUSTER_ID
        low_expr_mask = _get_leiden_low_expression_mask(df_results)
        clustered_df = df_results[~low_expr_mask].copy()
        low_expr_count = int(low_expr_mask.sum())
        n_clusters = st.session_state.get('leiden_n_clusters', clustered_df['leiden_cluster'].nunique() if 'leiden_cluster' in clustered_df.columns else 0)
        # Get output column from widget key (it's stored there automatically)
        output_col = st.session_state.get('leiden_outcol', 'phenotype_leiden')

        fresh_assignment_reset = bool(st.session_state.pop('leiden_fresh_assignment_mode', False))
        if fresh_assignment_reset:
            st.session_state['leiden_phenotype_map'] = {
                normalize_cluster_id(cluster_id): 'Unassigned'
                for cluster_id in df_results['leiden_cluster'].unique()
                if pd.notna(cluster_id) and normalize_cluster_id(cluster_id) is not None
            }
        elif 'leiden_phenotype_map' not in st.session_state:
            cluster_ids = sorted(
                [normalize_cluster_id(cluster_id) for cluster_id in df_results['leiden_cluster'].unique() if pd.notna(cluster_id)],
                key=cluster_sort_key,
            )
            st.session_state['leiden_phenotype_map'] = {
                cluster_id: (
                    st.session_state.get('leiden_low_expr_default_pheno', LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE)
                    if _is_leiden_low_expression_cluster(cluster_id, low_expr_cluster_id)
                    else 'Unassigned'
                )
                for cluster_id in cluster_ids
                if cluster_id is not None
            }
        if low_expr_count and low_expr_cluster_id is not None and not fresh_assignment_reset:
            current_value = st.session_state['leiden_phenotype_map'].get(normalize_cluster_id(low_expr_cluster_id))
            if current_value in {None, 'Unassigned'}:
                st.session_state['leiden_phenotype_map'][normalize_cluster_id(low_expr_cluster_id)] = (
                    st.session_state.get('leiden_low_expr_default_pheno', LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE)
                )
        st.session_state['leiden_phenotype_map'] = normalize_phenotype_map_keys(
            st.session_state.get('leiden_phenotype_map', {})
        )
        
        # Create tabs for different views
        tab_heatmap, tab_umap, tab_assign, tab_export = st.tabs([
            "Heatmap & Assignment", 
            "UMAP Visualization", 
            "Table Editor",
            "Export Results"
        ])
        
        # TAB 1: EXPRESSION HEATMAP + PHENOTYPE ASSIGNMENT (Combined view)
        with tab_heatmap:
            st.markdown("### Interactive Cluster Expression Heatmap")
            st.caption("Click on clusters to select/deselect | Assign phenotypes without constant rerendering")
            
            # Initialize session state for cluster selection
            if 'leiden_selected_clusters' not in st.session_state:
                st.session_state['leiden_selected_clusters'] = []
            if 'leiden_heatmap_config' not in st.session_state:
                st.session_state['leiden_heatmap_config'] = {
                    'cluster_x': True,
                    'cluster_y': True,
                    'show_counts': True,
                    'z_min': -4,
                    'z_max': 4
                }
            if 'z_min' not in st.session_state['leiden_heatmap_config']:
                st.session_state['leiden_heatmap_config']['z_min'] = -4
            if 'z_max' not in st.session_state['leiden_heatmap_config']:
                st.session_state['leiden_heatmap_config']['z_max'] = 4
            
            # Heatmap controls (use session_state to avoid reruns)
            hm_row1 = st.columns([2, 2, 2])
            with hm_row1[0]:
                if st.button("Toggle X-axis Clustering", help="Hierarchically cluster clusters (columns)", key="leiden_toggle_x"):
                    st.session_state['leiden_heatmap_config']['cluster_x'] = not st.session_state['leiden_heatmap_config']['cluster_x']
                st.caption(f"X-axis clustering: {'ON' if st.session_state['leiden_heatmap_config']['cluster_x'] else 'OFF'}")
            
            with hm_row1[1]:
                if st.button("Toggle Y-axis Clustering", help="Hierarchically cluster markers (rows)", key="leiden_toggle_y"):
                    st.session_state['leiden_heatmap_config']['cluster_y'] = not st.session_state['leiden_heatmap_config']['cluster_y']
                st.caption(f"Y-axis clustering: {'ON' if st.session_state['leiden_heatmap_config']['cluster_y'] else 'OFF'}")
            
            with hm_row1[2]:
                if st.button("Toggle Cell Counts", help="Show/hide cell counts", key="leiden_toggle_counts"):
                    st.session_state['leiden_heatmap_config']['show_counts'] = not st.session_state['leiden_heatmap_config']['show_counts']
                st.caption(f"Cell counts: {'ON' if st.session_state['leiden_heatmap_config']['show_counts'] else 'OFF'}")

            # Heatmap Range Config
            with st.expander("Heatmap Configuration", expanded=False):
                current_min = st.session_state['leiden_heatmap_config'].get('z_min', -4)
                current_max = st.session_state['leiden_heatmap_config'].get('z_max', 4)

                z_range = st.slider(
                    "Color Scale Range (Z-Score)",
                    min_value=-10.0,
                    max_value=10.0,
                    value=(float(current_min), float(current_max)),
                    step=0.5,
                    key="leiden_z_range_slider"
                )

                if z_range[0] != current_min or z_range[1] != current_max:
                    st.session_state['leiden_heatmap_config']['z_min'] = z_range[0]
                    st.session_state['leiden_heatmap_config']['z_max'] = z_range[1]
                    st.rerun()
            
            # Get config
            cluster_clusters = st.session_state['leiden_heatmap_config']['cluster_x']
            cluster_markers = st.session_state['leiden_heatmap_config']['cluster_y']
            show_counts = st.session_state['leiden_heatmap_config']['show_counts']
            z_min = st.session_state['leiden_heatmap_config']['z_min']
            z_max = st.session_state['leiden_heatmap_config']['z_max']

            if low_expr_count:
                st.info(
                    f"{low_expr_count:,} low-expression cells were preserved outside the clustering heatmap. "
                    f"They default to phenotype '{st.session_state.get('leiden_low_expr_default_pheno', LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE)}' and remain available in the assignment and export tables."
                )
            
            # Generate INTERACTIVE Plotly heatmap
            if norm_cols and all(c in df_results.columns for c in norm_cols):
                # Data Hash for Cache
                run_tag = st.session_state.get("leiden_run_id", "no_run")
                data_hash = (
                    f"{run_tag}_{len(clustered_df)}_{clustered_df['leiden_cluster'].nunique() if 'leiden_cluster' in clustered_df.columns else 0}_"
                    f"{str(sorted(norm_cols))}_{z_min}_{z_max}"
                )
                
                # --- CACHED FIGURE GENERATION ---
                base_fig, cluster_order, marker_order, cell_counts_dict, heatmap_z_ordered = build_leiden_heatmap_figure(
                    data_hash,
                    clustered_df,
                    norm_cols,
                    cluster_clusters,
                    cluster_markers,
                    show_counts,
                    z_min=z_min,
                    z_max=z_max,
                    marker_source_note=build_marker_source_note(norm_cols, method_key="leiden"),
                )
                
                if base_fig:
                    # Init State
                    if 'leiden_selected_clusters' not in st.session_state: st.session_state['leiden_selected_clusters'] = []
                    if 'leiden_phenotype_map' not in st.session_state: st.session_state['leiden_phenotype_map'] = {c: 'Unassigned' for c in cluster_order}
                    st.session_state['leiden_phenotype_map'] = normalize_phenotype_map_keys(
                        st.session_state.get('leiden_phenotype_map', {})
                    )
                    # Init phenotypes list
                    if 'leiden_available_phenotypes' not in st.session_state:
                         st.session_state['leiden_available_phenotypes'] = ['Unassigned']
                    if 'leiden_heatmap_component_reset_counter' not in st.session_state:
                        st.session_state['leiden_heatmap_component_reset_counter'] = 0

                    # Ensure map completeness
                    for c in cluster_order:
                        if c not in st.session_state['leiden_phenotype_map']: st.session_state['leiden_phenotype_map'][c] = 'Unassigned'

                    phenotype_map = st.session_state['leiden_phenotype_map']
                    available_phenotypes = get_all_phenotype_options(include_unassigned=True)
                    for pheno in phenotype_map.values():
                        if pheno != 'Unassigned' and pheno not in available_phenotypes:
                            available_phenotypes.append(pheno)
                    st.session_state['leiden_available_phenotypes'] = available_phenotypes
                    
                    fig = go.Figure(base_fig)
                    
                    # Force click interaction mode (disable drag-select for snappy clicking)
                    fig.update_layout(dragmode=False)
                    
                    # Shapes (Assigned Phenotypes Only - Green Frames)
                    shapes = []
                    c_to_idx = {cid: i for i, cid in enumerate(cluster_order)}
                    
                    for cid in cluster_order:
                        i = c_to_idx.get(cid)
                        if i is None: continue
                        
                        col = None
                        width = 0
                        
                        # Only show green frames for assigned clusters, as requested
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
                    fig.update_layout(title_text=f"Leiden Clusters (Assigned: {sum(1 for v in phenotype_map.values() if v!='Unassigned')})")

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
                        key_prefix="leiden",
                        phenotype_map_state_key="leiden_phenotype_map",
                        available_phenotypes_state_key="leiden_available_phenotypes",
                        cluster_marker_scores=cluster_marker_scores,
                        df_results=locals().get("df_results"),
                        marker_cols=locals().get("norm_cols") or [],
                        cluster_column="leiden_cluster",
                    ):
                        st.rerun()
                    if heatmap_click_selector_available():
                        component_event = render_heatmap_click_selector(
                            figure=fig,
                            available_phenotypes=available_phenotypes,
                            cluster_display_labels=cluster_display_labels,
                            cluster_marker_scores=cluster_marker_scores,
                            reset_counter=int(st.session_state.get('leiden_heatmap_component_reset_counter', 0)),
                            key="leiden_heatmap_click_selector",
                        )

                        event_is_new = consume_heatmap_assignment_event(
                            component_event,
                            scope=get_heatmap_assignment_scope(
                                "leiden", st.session_state.get("leiden_run_id", "no-run")
                            ),
                        )
                        assignment = (
                            apply_heatmap_assignment_event(
                                component_event,
                                st.session_state.get("leiden_phenotype_map", {}),
                                previous_event_id=st.session_state.get("leiden_heatmap_component_event_id"),
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
                            st.session_state["leiden_phenotype_map"] = updated_map
                            st.session_state["leiden_heatmap_component_event_id"] = event_id
                            st.session_state["leiden_heatmap_component_reset_counter"] = (
                                int(st.session_state.get("leiden_heatmap_component_reset_counter", 0)) + 1
                            )
                            st.success(
                                f"Assigned '{selected_click_phenotype}' to {len(selected_clusters)} cluster(s): "
                                + ", ".join(str(cluster_id) for cluster_id in selected_clusters)
                            )
                            st.rerun()
                    else:
                        st.plotly_chart(
                            fig,
                            key="leiden_heatmap_chart_fallback",
                            width="stretch",
                            config={'displayModeBar': False}
                        )
                        st.warning(
                            "Buffered heatmap click selection is unavailable because the local selector component has not been built yet. "
                            "The heatmap is shown in static mode until that frontend build exists."
                        )
                    
                    st.divider()
                    
                    # --- BATCH ASSIGNMENT INTERFACE (Inside FORM for No-Render Selection) ---
                    st.markdown("#### Phenotype Summary Table & Assignment")
                    st.caption("Edit phenotypes in table below with no rerendering. Click Sync to apply all changes at once.")
                    
                    # --- Add Custom Phenotype UI ---
                    st.caption("Need a phenotype not in the list? Add it here:")
                    render_add_custom_phenotype_ui(key_prefix="leiden")

                    # Build available phenotypes (standard + custom)
                    st.session_state['leiden_available_phenotypes'] = get_all_phenotype_options(include_unassigned=True)
                    
                    phenotype_map = st.session_state['leiden_phenotype_map']
                    available_phenotypes = st.session_state['leiden_available_phenotypes']

                    # Update available phenotypes if map contains new custom ones
                    for p in phenotype_map.values():
                        if p != 'Unassigned' and p not in available_phenotypes:
                            available_phenotypes.append(p)
                    st.session_state['leiden_available_phenotypes'] = available_phenotypes
                    
                    # Build summary table
                    summary_data = []
                    for c in cluster_order:
                        # Re-calculate Top 5 from cache WITH SCORES
                        top5_str = ""
                        if not heatmap_z_ordered.empty and c in heatmap_z_ordered.columns:
                            # Get top 5 markers and their values
                            top5_series = heatmap_z_ordered[c].nlargest(5)
                            # Format: Marker (1.2), Marker2 (0.9)
                            top5_parts = [f"{m} ({v:.2f})" for m, v in top5_series.items()]
                            top5_str = ", ".join(top5_parts)

                        summary_data.append({
                            'Cluster': c,
                            'Cells': cell_counts_dict.get(c, 0),
                            'Top Markers (Score)': top5_str,
                            'Phenotype': phenotype_map.get(c, 'Unassigned')
                        })

                    if low_expr_count and low_expr_cluster_id is not None:
                        summary_data.append({
                            'Cluster': normalize_cluster_id(low_expr_cluster_id),
                            'Cells': low_expr_count,
                            'Top Markers (Score)': _get_leiden_low_expression_note(
                                st.session_state.get('leiden_low_expr_metric'),
                                st.session_state.get('leiden_low_expr_threshold'),
                            ),
                            'Phenotype': phenotype_map.get(
                                normalize_cluster_id(low_expr_cluster_id),
                                st.session_state.get('leiden_low_expr_default_pheno', LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE),
                            ),
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Progress metric
                    col_p1, col_p2 = st.columns([2, 1])
                    col_p1.markdown("##### 1. Select Clusters & Edit Phenotypes")
                    
                    n_assigned = sum(1 for v in phenotype_map.values() if v != 'Unassigned')
                    n_total = len(phenotype_map)
                    col_p2.metric("Assigned", f"{n_assigned}/{n_total}")
                    
                    # FORM prevents reruns during dropdown edits
                    with st.form("leiden_batch_assignment_form", clear_on_submit=False):
                        
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
                            key="leiden_summary_editor_form",
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
                        st.session_state['leiden_phenotype_map'] = new_map
                        st.success("Synced! Heatmap updated.")
                        st.rerun()

                    # --- SHARED POST-ASSIGNMENT SECTIONS (Same as SOM) ---
                    render_phenotype_statistics(
                        df_results=df_results,
                        cluster_col='leiden_cluster',
                        phenotype_map=st.session_state.get('leiden_phenotype_map', {}),
                        heatmap_z_ordered=heatmap_z_ordered,
                        cluster_label='Cluster'
                    )

                    if low_expr_count:
                        st.caption(
                            "Low-expression bucket cells contribute to phenotype counts, but their marker ranking is excluded from the heatmap-based top-marker summary because they are preserved outside Leiden clustering."
                        )

                    render_publication_heatmap(
                        heatmap_z_ordered=heatmap_z_ordered,
                        phenotype_map=st.session_state.get('leiden_phenotype_map', {}),
                        key_prefix='leiden',
                        df_results=df_results,
                        group_col='leiden_cluster',
                        marker_columns=norm_cols,
                    )

                    update_cols = ['leiden_cluster']
                    for extra_col in ['leiden_low_expression', 'leiden_filter_status', 'leiden_low_expr_score', 'leiden_marker_sum_raw']:
                        if extra_col in df_results.columns:
                            update_cols.append(extra_col)
                    if 'umap_x' in df_results.columns:
                        update_cols.append('umap_x')
                    if 'umap_y' in df_results.columns:
                        update_cols.append('umap_y')

                    render_save_assignments(
                        df_results=df_results,
                        phenotype_map=st.session_state.get('leiden_phenotype_map', {}),
                        output_col=output_col,
                        project_root=project_root,
                        key_prefix='leiden',
                        cluster_col='leiden_cluster',
                        update_cluster_cols=update_cols,
                        cell_id_col=st.session_state.get('leiden_cell_id_col', 'cell_id')
                    )

                    render_export_split_data(
                        df_results=df_results,
                        output_col=output_col,
                        phenotype_map=st.session_state.get('leiden_phenotype_map', {}),
                        project_root=project_root,
                        key_prefix='leiden',
                        cluster_col='leiden_cluster',
                        cell_id_col=st.session_state.get('leiden_cell_id_col', 'cell_id')
                    )

            else:
                st.warning("Normalized columns not found. Please re-run clustering.")
        
        # TAB 2: UMAP VISUALIZATION
        with tab_umap:
            st.markdown("### UMAP Projection")
            st.caption("2D embedding of cells colored by cluster or phenotype (matches notebook Step 6)")
            
            # UMAP controls
            umap_col1, umap_col2, umap_col3 = st.columns([2, 2, 1])
            
            with umap_col1:
                color_by = st.radio(
                    "Color By",
                    ["Cluster", "Phenotype"],
                    horizontal=True,
                    help="Color cells by cluster ID or assigned phenotype",
                    key="leiden_umap_color"
                )
            
            with umap_col2:
                max_cells_plot = st.number_input(
                    "Max Cells to Plot",
                    1000, 500000, 50000, 1000,
                    help="Subsample for faster rendering (full dataset saved in output)",
                    key="leiden_umap_max"
                )
            
            with umap_col3:
                point_size = st.slider("Point Size", 0.1, 5.0, 0.5, 0.1, key="leiden_umap_size")
            
            # Check if phenotypes are assigned
            if output_col in df_results.columns:
                has_phenotypes = df_results[output_col].notna().any()
            else:
                has_phenotypes = False
            
            if color_by == "Phenotype" and not has_phenotypes:
                st.warning("No phenotypes assigned yet. Showing clusters instead.")
                color_by = "Cluster"
            
            # Generate UMAP plot
            if 'umap_x' in df_results.columns and 'umap_y' in df_results.columns:
                # Subsample for plotting
                if len(df_results) > max_cells_plot:
                    plot_df = df_results.sample(n=max_cells_plot, random_state=42)
                    st.caption(f"Showing {max_cells_plot:,} of {len(df_results):,} cells")
                else:
                    plot_df = df_results
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if color_by == "Cluster":
                    scatter = ax.scatter(
                        plot_df['umap_x'], 
                        plot_df['umap_y'],
                        c=plot_df['leiden_cluster'].astype(int),
                        cmap='tab20',
                        s=point_size,
                        alpha=0.6,
                        rasterized=True
                    )
                    plt.colorbar(scatter, ax=ax, label='Cluster ID')
                    ax.set_title(f'UMAP: Leiden Clusters (n={n_clusters})')
                else:
                    # Color by phenotype
                    unique_phenotypes = plot_df[output_col].unique()
                    n_pheno = len(unique_phenotypes)
                    colors = plt.cm.Set3(np.linspace(0, 1, n_pheno))
                    pheno_to_color = dict(zip(unique_phenotypes, colors))
                    
                    for phenotype in unique_phenotypes:
                        mask = plot_df[output_col] == phenotype
                        ax.scatter(
                            plot_df.loc[mask, 'umap_x'],
                            plot_df.loc[mask, 'umap_y'],
                            c=[pheno_to_color[phenotype]],
                            label=phenotype,
                            s=point_size,
                            alpha=0.6,
                            rasterized=True
                        )
                    
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax.set_title(f'UMAP: Phenotypes (n={n_pheno})')
                
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Save button
                if st.button("Save UMAP", key="save_leiden_umap"):
                    plots_dir = project_root / 'outputs/phenotyping/leiden/plots'
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    umap_file = plots_dir / 'leiden_umap_projection.png'
                    fig.savefig(umap_file, dpi=300, bbox_inches='tight')
                    st.success(f"UMAP saved: {umap_file.name}")
            else:
                st.info(
                    "No UMAP coordinates in this result. UMAP is off by default because it is used only "
                    "for visualisation and often takes longer than the clustering itself. To generate it, "
                    "tick **Compute UMAP embedding** in the clustering settings and re-run."
                )
        
        # TAB 3: TABLE EDITOR (Alternative assignment view)
        with tab_assign:
            st.markdown("### Phenotype Table Editor")
            st.caption("Full table view for editing assignments. Changes sync with Heatmap tab.")
            
            # Get phenotype data from session state
            phenotype_map = st.session_state.get('leiden_phenotype_map', {})
            available_phenotypes = st.session_state.get('leiden_available_phenotypes', ['Unassigned'])
            
            # Cluster statistics
            cluster_stats = df_results.groupby('leiden_cluster').agg({
                'leiden_cluster': 'count'
            }).rename(columns={'leiden_cluster': 'n_cells'})
            cluster_stats['percentage'] = (cluster_stats['n_cells'] / len(df_results) * 100).round(2)
            
            # Filter out NaN values and convert to int for clean cluster list
            clusters = sorted(
                [normalize_cluster_id(c) for c in df_results['leiden_cluster'].unique() if normalize_cluster_id(c) is not None],
                key=cluster_sort_key,
            )
            
            # Assignment summary
            assigned = sum(1 for v in phenotype_map.values() if v != 'Unassigned')
            total = len(phenotype_map)
            
            sum_col1, sum_col2 = st.columns([1, 2])
            with sum_col1:
                st.metric("Progress", f"{assigned}/{total} assigned")
            with sum_col2:
                st.caption("Defined phenotypes: " + ", ".join([p for p in available_phenotypes if p != 'Unassigned']))
            
            st.divider()
            
            # Full editable table
            st.markdown("#### Edit Assignments")
            
            # Build dataframe
            assign_df = pd.DataFrame([
                {
                    'Cluster': int(c) if pd.notna(c) else 0,
                    'Cells': int(cluster_stats.loc[c, 'n_cells']) if c in cluster_stats.index and pd.notna(cluster_stats.loc[c, 'n_cells']) else 0,
                    '%': float(cluster_stats.loc[c, 'percentage']) if c in cluster_stats.index and pd.notna(cluster_stats.loc[c, 'percentage']) else 0.0,
                    'Phenotype': phenotype_map.get(int(c) if pd.notna(c) else 0, 'Unassigned')
                }
                for c in clusters
            ])
            
            # Add top markers column
            if norm_cols:
                numeric_norm_cols = [c for c in norm_cols if c in df_results.columns and pd.api.types.is_numeric_dtype(df_results[c])]
                cluster_top_markers = []
                for c in clusters:
                    if _is_leiden_low_expression_cluster(c, low_expr_cluster_id):
                        cluster_top_markers.append(_get_leiden_low_expression_note(
                            st.session_state.get('leiden_low_expr_metric'),
                            st.session_state.get('leiden_low_expr_threshold'),
                        ))
                        continue
                    if not numeric_norm_cols:
                        cluster_top_markers.append("")
                        continue
                    cluster_data = df_results[df_results['leiden_cluster'] == c][numeric_norm_cols].mean()
                    top_3 = cluster_data.nlargest(3)
                    markers_str = ', '.join([f"{m.replace('_corr_norm', '').replace('_norm', '').replace('_corr', '')}" 
                                            for m in top_3.index])
                    cluster_top_markers.append(markers_str)
                assign_df['Top 3 Markers'] = cluster_top_markers
            
            # Editable table with dropdown for phenotypes
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
                    ),
                    "Top 3 Markers": st.column_config.TextColumn(disabled=True, width="medium")
                },
                hide_index=True,
                num_rows="fixed",
                key="leiden_table_editor"
            )
            
            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("Apply Changes", type="primary", key="leiden_apply_table"):
                    # Update phenotype map
                    new_map = {int(row['Cluster']): row['Phenotype'] for _, row in edited_df.iterrows()}
                    st.session_state['leiden_phenotype_map'] = new_map
                    st.success("Assignments applied.")
                    st.rerun()
            
            with btn_col2:
                # Add new phenotype
                new_pheno = st.text_input("Add new phenotype", key="leiden_new_pheno_table", placeholder="Enter name")
                if new_pheno and new_pheno not in available_phenotypes:
                    if st.button("Add", key="leiden_add_pheno_btn"):
                        st.session_state['leiden_available_phenotypes'].append(new_pheno)
                        st.rerun()
            
            with btn_col3:
                if st.button("Reset All to Unassigned", key="leiden_reset_table"):
                    st.session_state['leiden_phenotype_map'] = {
                        c: (
                            st.session_state.get('leiden_low_expr_default_pheno', LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE)
                            if _is_leiden_low_expression_cluster(c, low_expr_cluster_id)
                            else 'Unassigned'
                        )
                        for c in clusters
                    }
                    st.rerun()
        
        # TAB 4: EXPORT RESULTS
        with tab_export:
            st.markdown("### Export Phenotyped Results")
            st.caption("Save final results with phenotype assignments (matches notebook Step 7)")
            
            # Apply current phenotypes to dataframe
            phenotype_map = st.session_state.get('leiden_phenotype_map', {})
            df_results[output_col] = df_results['leiden_cluster'].map(phenotype_map).fillna('Unassigned')
            
            # Show final summary
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
                    hide_index=True
                )
            
            with export_col2:
                st.markdown("#### Export Options")
                
                # Remove normalized columns from final export
                remove_norm = st.checkbox(
                    "Remove _norm columns",
                    value=True,
                    help="Remove normalized marker columns from final CSV (keeps output clean)",
                    key="leiden_remove_norm"
                )
                
                output_filename = st.text_input(
                    "Output Filename",
                    "all_samples_combined_leiden_phenotyped.csv",
                    key="leiden_output_filename"
                )
            
            # Export button
            if st.button("Export Final Results", type="primary", key="leiden_export_final"):
                with st.spinner("Saving results..."):
                    # Prepare dataframe for export
                    df_export = df_results.copy()
                    
                    if remove_norm:
                        cols_to_drop = [c for c in df_export.columns if c.endswith('_norm')]
                        df_export = df_export.drop(columns=cols_to_drop)
                        st.caption(f"Removed {len(cols_to_drop)} normalized columns")
                    
                    # Save to outputs
                    final_out_dir = project_root / 'outputs/phenotyping'
                    final_out_dir.mkdir(parents=True, exist_ok=True)
                    final_path = final_out_dir / output_filename
                    
                    df_export.to_csv(final_path, index=False)
                    
                    st.success("Results exported successfully.")
                    st.code(f"Output file: {final_path}", language=None)
                    st.metric("Total Cells", len(df_export))
                    st.metric("Unique Phenotypes", df_export[output_col].nunique())
                    st.balloons()




    # --- VISUAL VALIDATION COMPONENT ---
    if st.session_state.get('viz_enabled'):
        st.divider()
        with st.expander("Visual Validation of Representative Cells", expanded=True):
            st.markdown("#### Visualize Representative Cells")
            st.caption("Inspect random cells from specific clusters to validate phenotype assignments.")
            
            # Ensure we have data
            if 'leiden_temp_file' in st.session_state and Path(st.session_state['leiden_temp_file']).exists():
                # Load data if not already (it might be loaded in the interactive block, but ensure scope)
                if 'df_results' not in locals():
                    df_results = read_dataframe_parquet_safe(
                        st.session_state['leiden_temp_file'],
                        session_state_key='leiden_temp_file',
                        label='Leiden results',
                    )
                    if df_results is None:
                        return
                
                # match batches if not done
                viz_roi = st.session_state.get('viz_roi_dir')
                viz_seg = st.session_state.get('viz_seg_dir')
                
                if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
                    st.warning("Please configure valid ROI and Segmentation directories in Global Input Configuration.")
                else:
                    viz = CellVisualizer(viz_roi, viz_seg)
                                       # File matching (cache in session state to avoid scanning every time)
                    if not st.session_state.get('viz_file_matches'):
                          # Get unique batches from DF
                          batch_col = st.session_state.get('viz_col_batch', 'batch_id')
                          if not batch_col or batch_col not in df_results.columns:
                              batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                          
                          if batch_col in df_results.columns:
                              unique_batches = df_results[batch_col].unique().astype(str).tolist()
                              matches = viz.match_file_triads(unique_batches)
                              st.session_state.viz_file_matches = matches
                              if not matches:
                                  st.error("No matching image files found for these batches.")
                          else:
                              st.error(f"Batch column '{batch_col}' not found in results.")
                    else:
                        viz.file_map = st.session_state.viz_file_matches

                    # UI Controls
                    # 1. Cluster Selection
                    avail_clusters = sorted(df_results['leiden_cluster'].unique())
                    current_pheno_map = st.session_state.get('leiden_phenotype_map', {})
                    
                    # Format options: "Cluster 0 (T cell)"
                    cluster_options = {c: f"Cluster {c} ({current_pheno_map.get(c, 'Unassigned')})" for c in avail_clusters}
                    
                    with st.form("leiden_viz_form"):
                        vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])
                        
                        with vv_col1:
                            selected_cluster_vv = st.selectbox(
                                "Select Cluster", 
                                avail_clusters,
                                format_func=lambda c: cluster_options[c],
                                key="leiden_viz_cluster_sel"
                            )
                        
                        with vv_col2:
                            n_viz_cells = st.number_input("Cells to View", 1, 50, 5, key="leiden_viz_n_cells")
                            
                        with vv_col3:
                            show_neighbors = st.checkbox("Show Neighbors", value=True, key="leiden_viz_neighbors")
                        with vv_col4:
                            leiden_batch_scope = st.radio(
                                "Batch Scope",
                                ["All matched batches", "Specific batch"],
                                horizontal=True,
                                key="leiden_viz_batch_scope",
                            )
                            leiden_batch_options = sorted(str(batch) for batch in st.session_state.viz_file_matches.keys())
                            selected_batch_vv = None
                            if leiden_batch_scope == "Specific batch":
                                selected_batch_vv = st.selectbox(
                                    "Batch",
                                    leiden_batch_options,
                                    key="leiden_viz_batch_sel",
                                )
                            st.write("") # Spacer
                            run_viz = st.form_submit_button("Visualize Random Cells")
                    
                    # Store generated figures for potential saving
                    if 'leiden_viz_figures' not in st.session_state:
                        st.session_state.leiden_viz_figures = []
                    
                    # Logic for Saving
                    if st.session_state.get('leiden_viz_figures'):
                        st.divider()
                        save_col1, save_col2 = st.columns([3, 1])
                        with save_col1:
                             st.info(f"Generated {len(st.session_state.leiden_viz_figures)} cell visualizations for {st.session_state.get('leiden_viz_last_cluster', 'selected clusters')}.")
                        with save_col2:
                             if st.button("Save Generated Cells", key="leiden_viz_save_btn"):
                                 try:
                                     # Create directory
                                     # Naming: outputs/phenotyping/visual_validation/[cluster]_[timestamp]/
                                     ts = int(time.time())
                                     last_cluster = st.session_state.get('leiden_viz_last_cluster', 'mixed').replace(' ', '_')
                                     save_dir = project_root / 'outputs' / 'phenotyping' / 'visual_validation' / f"{last_cluster}_{ts}"
                                     save_dir.mkdir(parents=True, exist_ok=True)
                                     
                                     meta_records = []
                                     
                                     for i, item in enumerate(st.session_state.leiden_viz_figures):
                                         # item: {'fig': fig, 'cell_id': cid, 'batch': bid, 'x': x, 'y': y, 'markers': top5}
                                         # We can't pickle figures reliably but we might have saved them in memory or just re-save here if we had the object.
                                         # Actually, matplotlib figures are tricky to persist in session_state across reruns if not careful.
                                         # Streamlit re-runs script. If we stored 'fig' object, it might be closed.
                                         # Better approach: Save IMMEDIATELY during generation or regen.
                                         # For now, let's assume the user wants to save what they JUST generated.
                                         # BUT, since we are inside the 'run_viz' block below, we should handle autosave OR offer save button that re-runs or uses cached data.
                                         
                                         # OPTION B: Auto-save or "Process & Save" mode. 
                                         # The user asked: "Can you also auto save these detected cells... and when you save them, add a corresponding csv"
                                         pass 
                                         
                                     # Re-implementing Save logic inside the generation loop is safer for Streamlit.
                                 except Exception as e:
                                     st.error(f"Save error: {e}")

                    if run_viz and selected_cluster_vv is not None:
                        # Clear previous
                        st.session_state.leiden_viz_figures = []
                        st.session_state.leiden_viz_last_cluster = f"Cluster_{selected_cluster_vv}"
                        # Logic to extract cells
                        subset = df_results[df_results['leiden_cluster'] == selected_cluster_vv]
                        
                        if len(subset) == 0:
                            st.warning("No cells in this cluster.")
                        else:
                            # 1. Determine Top 5 Markers for this cluster
                            # Calculate mean z-scores locally to be self-contained
                            norm_cols = st.session_state.get('leiden_norm_cols', [])
                            if norm_cols:
                                # Safe numeric check
                                valid_norm = [c for c in norm_cols if c in subset.columns and pd.api.types.is_numeric_dtype(subset[c])]
                                if valid_norm:
                                    cluster_means = subset[valid_norm].mean()
                                    top_5_markers = cluster_means.nlargest(5).index.tolist()
                                    
                                    # Get raw marker names (remove suffixes for display if needed)
                                    top_5_indices = [df_results.columns.get_loc(c) for c in top_5_markers]
                                    top_5_names = [c.replace('_norm', '').replace('_corr', '') for c in top_5_markers]
                                    
                                    st.info(f"Top markers: {', '.join(top_5_names)}")
                                    
                                    # 2. Sample Cells (Group by batch for speed)
                                    # Filter subset to only batches we have images for
                                    batch_col = st.session_state.get('viz_col_batch', 'batch_id')
                                    if not batch_col or batch_col not in df_results.columns:
                                        batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                                    
                                    if batch_col not in df_results.columns:
                                        st.error(f"Batch column '{batch_col}' not found in results.")
                                        subset_valid = pd.DataFrame()
                                    elif not st.session_state.get('viz_file_matches'):
                                        st.error("No matching image files found for visualization batches.")
                                        subset_valid = pd.DataFrame()
                                    else:
                                        valid_batches = list(st.session_state.viz_file_matches.keys())
                                        # Ensure string match
                                        subset_valid = subset[subset[batch_col].astype(str).isin(valid_batches)]
                                        if leiden_batch_scope == "Specific batch" and selected_batch_vv:
                                            subset_valid = subset_valid[subset_valid[batch_col].astype(str) == str(selected_batch_vv)]
                                    
                                    if len(subset_valid) == 0:
                                        st.warning("No cells found in batches with available images.")
                                    else:
                                        # Sample
                                        n_actual = min(n_viz_cells, len(subset_valid))
                                        sampled = subset_valid.sample(n=n_actual, random_state=int(time.time()))
                                        
                                        # Sort by batch to optimize IO
                                        sampled = sampled.sort_values(batch_col)
                                        sampled_batches = sampled[batch_col].astype(str).drop_duplicates().tolist()
                                        prewarm_fn = getattr(viz, 'prewarm_batch', None)
                                        if callable(prewarm_fn):
                                            with st.spinner(f"Preloading {len(sampled_batches)} batch cache(s)..."):
                                                for prewarm_batch_id in sampled_batches:
                                                    prewarm_fn(prewarm_batch_id, requested_channels=top_5_markers)
                                        
                                        # Grid Layout
                                        st.markdown("---")
                                        cols = st.columns(3) # 3 columns grid
                                        
                                        # Progress bar
                                        prog_bar = st.progress(0)
                                        
                                        x_col = st.session_state.get('viz_col_x', 'x_centroid')
                                        y_col = st.session_state.get('viz_col_y', 'y_centroid')
                                        tile_sz = st.session_state.get('viz_tile_size', 40)

                                        # Channel mapping: We need indices in the Image stack.
                                        # The image stack from OME-TIFF usually matches the `markers` list order 
                                        # IF the markers list matches the channel order.
                                        # This is tricky: The TIFF channels are usually unknown unless we read metadata.
                                        # But usually the 'markers_selected' list corresponds to the channels IF we assume matching order.
                                        # CRITICAL: We need to know which channel index corresponds to "CD45".
                                        # In absence of channel metadata mapping, we assume the user selected markers in order, 
                                        # OR we just try to fetch by index if standard.
                                        # actually, `extract_cell_image` gets raw stack. 
                                        # The notebook relied on `markers` list being correct order.
                                        # We will use the indices of `top_5_markers` within `markers_selected` 
                                        # assuming `markers_selected` matches the TIFF channel order.
                                        # This is a weak assumption but best we have without metadata reading.
                                        # fallback: Just plot provided indices assuming markers_selected represents channel 0..N
                                        
                                        all_markers = st.session_state.get('leiden_markers', []) 
                                        # Using 'leiden_markers' as reference for channel order is risky but standard.
                                        
                                        for idx, (_, row) in enumerate(sampled.iterrows()):
                                            try:
                                                bid = str(row[batch_col])
                                                x, y = row[x_col], row[y_col]
                                                
                                                cell_data = viz.get_cell_context(
                                                    bid,
                                                    x,
                                                    y,
                                                    tile_size=tile_sz,
                                                    requested_channels=top_5_markers,
                                                )
                                                
                                                # IMPROVED: Name-Based Fuzzy Matching
                                                # We have 'top_5_markers' (e.g., ['CD45_corr', 'CD3e'])
                                                # We have 'cell_data['channel_names']' from OME (e.g., ['DAPI', 'CD45', 'CD3e', ...])
                                                
                                                ome_channels = cell_data.get('channel_names', [])
                                                marker_indices_for_plot = []
                                                names_for_plot = []
                                                
                                                if ome_channels:
                                                    # Normalize OME channels for matching (lowercase, strip whitespace)
                                                    ome_norm = [c.lower().strip() for c in ome_channels]
                                                    
                                                    for m in top_5_markers:
                                                        # Normalize marker (strip suffix, lowercase)
                                                        # Suffixes to strip: _corr, _norm, _unmixed, etc.
                                                        raw_m = m.replace('_norm', '').replace('_corr', '').replace('_unmixed', '')
                                                        target = raw_m.lower().strip()
                                                        
                                                        # Find matches
                                                        # Priority 1: Exact Match (normalized)
                                                        found_idx = -1
                                                        if target in ome_norm:
                                                            found_idx = ome_norm.index(target)
                                                        
                                                        # Priority 2: Fuzzy / Containment
                                                        if found_idx == -1:
                                                            for i, ome_c in enumerate(ome_norm):
                                                                if target in ome_c or ome_c in target:
                                                                    found_idx = i
                                                                    break
                                                        
                                                        if found_idx != -1:
                                                            marker_indices_for_plot.append(found_idx)
                                                            names_for_plot.append(raw_m) # Use clean name for display
                                                        else:
                                                            # Warn or skip?
                                                            _log_console(f"Warning: Marker {m} (raw: {raw_m}) not found in OME channels: {ome_channels}", level="warning")
                                                            
                                                # FALLBACK: If OME channels missing or no matches found, fallback to index
                                                if not marker_indices_for_plot:
                                                     # Previous Logic: Match index in 'all_markers'
                                                     for m in top_5_markers:
                                                        raw_m = m.replace('_norm', '').replace('_corr', '')
                                                        matches_idx = [i for i, x in enumerate(all_markers) if x == raw_m or x == raw_m + "_corr"]
                                                        if matches_idx:
                                                            marker_indices_for_plot.append(matches_idx[0])
                                                            names_for_plot.append(raw_m)

                                                if not marker_indices_for_plot:
                                                     # Final Fallback
                                                     marker_indices_for_plot = [0, 1, 2, 3, 4]
                                                     names_for_plot = [f"Ch{i}" for i in marker_indices_for_plot]

                                                fig = viz.generate_visualization(
                                                    cell_data, 
                                                    marker_indices_for_plot, 
                                                    names_for_plot,
                                                    display_size=tile_sz,
                                                    show_neighbors=show_neighbors
                                                )
                                                cell_identifier = get_cell_id_display_value(
                                                    row,
                                                    st.session_state.get('leiden_cell_id_col'),
                                                )
                                                
                                                # Detailed Caption
                                                caption_txt = f"Cell: {cell_identifier}\n"
                                                caption_txt += f"Batch: {bid}\n"
                                                caption_txt += f"Loc: ({x:.1f}, {y:.1f})"
                                                
                                                with cols[idx % 3]:
                                                    st.pyplot(fig)
                                                    st.text(caption_txt) # Use text for multi-line support
                                                
                                                # AUTO-SAVE logic (as requested)
                                                # Save to outputs/phenotyping/visual_validation/
                                                # Filename: [batch]_[cell_id]_Cluster[c]_[pheno].png
                                                
                                                try:
                                                    save_root = project_root / 'outputs/phenotyping/visual_validation'
                                                    save_root.mkdir(parents=True, exist_ok=True)
                                                    
                                                    clean_pheno = current_pheno_map.get(selected_cluster_vv, 'Unassigned').replace(' ', '_').replace('/', '-')
                                                    fname = f"{bid}_{cell_identifier}_C{selected_cluster_vv}_{clean_pheno}.png"
                                                    save_path = save_root / fname
                                                    
                                                    fig.savefig(save_path, bbox_inches='tight', dpi=150)
                                                    _log_console(f"Saved: {save_path.name}")
                                                    
                                                    # Append to CSV log
                                                    csv_path = save_root / "visualized_cells_metadata.csv"
                                                    metadata = {
                                                        'timestamp': datetime.now().isoformat(),
                                                        'filename': fname,
                                                        'batch_id': bid,
                                                        'cell_id': cell_identifier,
                                                        'cluster': selected_cluster_vv,
                                                        'phenotype': clean_pheno,
                                                        'x': x,
                                                        'y': y,
                                                        'top_markers': ", ".join(names_for_plot)
                                                    }
                                                    # Add marker intensities if possible (complex with normalization etc, skipping valid_norm reading again)
                                                    
                                                    meta_df = pd.DataFrame([metadata])
                                                    meta_df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                                                    
                                                except Exception as save_err:
                                                    _log_console(f"Auto-save failed: {save_err}", level="error")
                                                
                                                plt.close(fig)
                                                
                                            except Exception as e:
                                                st.error(f"Error viewing cell: {e}")
                                            
                                            prog_bar.progress((idx + 1) / n_actual)
                                            
                                        st.success(f"Visualized {n_actual} cells. Images saved to `outputs/phenotyping/visual_validation/`")
                                            
                                else:
                                    st.warning(f"Could not determine top markers (columns missing?).")
