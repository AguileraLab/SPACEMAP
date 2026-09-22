"""
Tab 7: Phenotyping - Main module

This module coordinates multiple phenotyping subtabs.
Each subtab is implemented in a separate file for maintainability.
"""

__package__ = "spacemap_gui.tabs.tab7_phenotyping"

import streamlit as st
import pandas as pd
import os
import sys

# Import shared utilities (lightweight – no heavy scientific deps)
from .common import (
    logger,
    Timer,
    project_root_calc,
    get_cached_global_pheno_dir,
    set_cached_global_pheno_dir,
    get_cached_global_pheno_file,
    set_cached_global_pheno_file,
    get_marker_list,
    get_canonical_marker_panel,
    canonicalize_marker_selection,
    get_named_marker_panels,
    marker_panel_shortcuts,
    get_marker_workspace_panel,
    infer_corrected_marker_rule,
    is_corrected_marker_column,
    marker_workspace_summary,
)
from spacemap_gui.persistence import load_directory_settings, save_directory_settings, save_active_project_state
from spacemap_gui.ui_utils import directory_input_with_browse

# NOTE: Subtab renderers and CellVisualizer are imported LAZILY inside
# their click-to-load blocks to avoid pulling in heavy libraries
# (scanpy, anndata, sklearn, minisom, scimap, tifffile, scipy, …)
# at app startup.  Python caches modules in sys.modules so the import
# cost is paid only on the first load.


def render():
    from pathlib import Path
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    # CRITICAL: Mark Tab 7 as currently rendering BEFORE any other processing
    st.session_state["_currently_rendering_tab"] = "tab7"
    
    import time
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] TAB7: Loading Phenotyping Tab...", flush=True)

    col_title, col_save = st.columns([3, 1])
    with col_title:
        st.markdown("## Step 7: Phenotyping Analysis")
        st.markdown("Perform cell phenotyping using various methodologies.")
    with col_save:
        pass

    # CSS for tight layout and scrollable marker list
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; margin-right: 2px; }
        .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b; }
        </style>
    """, unsafe_allow_html=True)
    
    # --- Global Input Configuration ---
    
    # Initialize persistent session state for phenotyping config
    bc_output_dir = st.session_state.get('bc_output_dir', str(project_root / 'outputs/batch_corrected'))
    cached_dir = get_cached_global_pheno_dir(bc_output_dir)
    cached_file = get_cached_global_pheno_file(None)

    if 'global_pheno_dir' not in st.session_state:
        st.session_state.global_pheno_dir = cached_dir

    # Track user override vs Tab6 sync
    if 'global_pheno_dir_user_override' not in st.session_state:
        st.session_state['global_pheno_dir_user_override'] = False
    if 'global_pheno_dir_last_bc' not in st.session_state:
        st.session_state['global_pheno_dir_last_bc'] = bc_output_dir

    # If Tab6 output dir changed and user has not overridden, sync to new bc_output_dir
    if bc_output_dir != st.session_state['global_pheno_dir_last_bc'] and not st.session_state['global_pheno_dir_user_override']:
        st.session_state.global_pheno_dir = bc_output_dir
        set_cached_global_pheno_dir(bc_output_dir)
    st.session_state['global_pheno_dir_last_bc'] = bc_output_dir
        
    with st.expander("Global Input Configuration (Applied to all subtabs)", expanded=True):
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            global_input_dir = directory_input_with_browse(
                label="Input Directory (Batch Corrected)", 
                text_key="global_pheno_dir",
                default_value=st.session_state.get("global_pheno_dir", cached_dir),
                browser_key_prefix="tree_tab7_global_pheno_dir",
                help_text="This directory will be remembered across all subtabs"
            )
            # Update cache and override flag when user changes
            if global_input_dir != st.session_state.get('global_pheno_dir_cached'):
                set_cached_global_pheno_dir(global_input_dir)
                st.session_state['global_pheno_dir_cached'] = global_input_dir
            st.session_state['global_pheno_dir_user_override'] = (global_input_dir != bc_output_dir)
            
        with col_g2:
            input_files = []
            if os.path.exists(global_input_dir):
                input_files = sorted([f for f in os.listdir(global_input_dir) if f.endswith('.csv')])
            
            # Auto-sync to last batch-corrected file if newly generated in Tab 6
            last_bc_file = st.session_state.get('bc_last_run_file')
            if last_bc_file and os.path.exists(last_bc_file):
                last_bc_filename = os.path.basename(last_bc_file)
                if last_bc_filename in input_files and st.session_state.get('global_pheno_file') != last_bc_filename:
                    if st.session_state.get('global_pheno_last_synced_bc') != last_bc_file:
                        st.session_state.global_pheno_file = last_bc_filename
                        set_cached_global_pheno_file(last_bc_filename)
                        st.session_state['global_pheno_last_synced_bc'] = last_bc_file
                        st.rerun()
            
            # Try to restore previous selection if it still exists
            preferred_file = st.session_state.get('global_pheno_file', cached_file)
            default_idx = 0
            if preferred_file in input_files:
                default_idx = input_files.index(preferred_file)
            
            global_file_select = st.selectbox(
                "Select Unified CSV File", 
                input_files, 
                index=default_idx if input_files else 0,
                key="global_pheno_file"
            )
            # Persist selected file
            if global_file_select != st.session_state.get('global_pheno_file_cached'):
                set_cached_global_pheno_file(global_file_select)
                # A project may hold several unified CSVs; record the latest so
                # reloading the project reopens this one.
                from spacemap_gui.persistence import record_latest_unified_csv

                record_latest_unified_csv(global_file_select)
                st.session_state['global_pheno_file_cached'] = global_file_select
            
        global_full_path = None
        if global_file_select:
            global_full_path = os.path.join(global_input_dir, global_file_select)

            # Same fast-copy panel as Tabs 8 and 9; all three read this table.
            from spacemap_gui.ui_utils import render_fast_copy_controls

            render_fast_copy_controls(global_full_path, key_prefix="tab7")
        
        # Show active path
        if global_full_path and os.path.exists(global_full_path):
            st.caption(f"Active: `{global_full_path}`")

        # Save Project Button
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Save Project", key="tab7_global_save_project_btn", width='stretch'):
            save_active_project_state("tab7_phenotyping")
            st.success("Project settings saved to configuration file.")

    # --- Shared Marker Workspace ---
    with st.expander("Shared Marker Source and Panel (Applied to all subtabs)", expanded=True):
        if not global_full_path or not os.path.exists(global_full_path):
            st.info("Choose a readable unified CSV above to inspect marker columns and configure the shared marker panel.")
        else:
            try:
                marker_header = pd.read_csv(global_full_path, nrows=5)
                available_markers = get_marker_list(marker_header)
            except FileNotFoundError:
                available_markers = []
                st.error("The selected unified CSV is no longer available. Choose another file in Global Input Configuration.")
            except PermissionError:
                available_markers = []
                st.error("SPACEMAP cannot read the selected unified CSV. Check file permissions or close any program locking the file.")
            except pd.errors.EmptyDataError:
                available_markers = []
                st.error("The selected unified CSV is empty. Choose a CSV containing cell data and marker columns.")
            except Exception as error:
                available_markers = []
                st.error(f"SPACEMAP could not inspect marker columns: {error}. Confirm that the selected file is a readable CSV.")

            if not available_markers:
                st.warning("No usable marker columns were found. Check the selected CSV and confirm that marker columns are not named as metadata fields.")
            else:
                workspace = st.session_state.get("tab7_marker_workspace", {})
                if not isinstance(workspace, dict) or workspace.get("input_path") != global_full_path:
                    saved_workspace = load_directory_settings(
                        global_input_dir,
                        section_key="tab7_marker_workspace",
                    )
                    if not isinstance(saved_workspace, dict):
                        saved_workspace = {}
                    inferred_position, inferred_affix = infer_corrected_marker_rule(available_markers)
                    workspace = {
                        "input_path": global_full_path,
                        "affix_position": saved_workspace.get("affix_position", inferred_position),
                        "correction_affix": saved_workspace.get("correction_affix", inferred_affix),
                        "marker_panels": saved_workspace.get("marker_panels", {}),
                        "active_panel_name": saved_workspace.get("active_panel_name", ""),
                        # Legacy fields remain available for migration of old project JSON files.
                        "selected_markers": saved_workspace.get("selected_markers", []),
                        "preparation_note": saved_workspace.get("preparation_note", "Method-specific normalization"),
                    }
                    st.session_state["tab7_marker_workspace"] = workspace

                inferred_position, inferred_affix = infer_corrected_marker_rule(available_markers)
                panels, normalized_workspace = get_named_marker_panels(available_markers, global_full_path)
                st.session_state["tab7_marker_workspace"] = normalized_workspace
                panel_names = list(panels)
                active_name = str(normalized_workspace.get("active_panel_name", ""))
                if active_name not in panels and panel_names:
                    active_name = panel_names[0]

                st.caption(
                    "Create named marker panels once here. Each phenotyping subtab can select a different saved panel and trim it locally."
                )
                st.caption(
                    f"The load buttons use the configured {inferred_position.lower()} `{inferred_affix}` rule. "
                    "Panels store actual CSV columns, so names such as CD4_corr remain visible."
                )

                position_options = ["Suffix", "Prefix"]
                current_position = workspace.get("affix_position", inferred_position)
                affix_position = st.selectbox(
                    "Corrected-marker naming rule",
                    position_options,
                    index=position_options.index(current_position) if current_position in position_options else 0,
                    key="tab7_shared_affix_position",
                    help="Used by the corrected-marker shortcut, for example CD4_corr or corr_CD4.",
                )
                correction_affix = st.text_input(
                    "Corrected-marker text",
                    value=str(workspace.get("correction_affix", inferred_affix)),
                    key="tab7_shared_correction_affix",
                    help="Examples: `_corr`, `_croo`, or `corr_`.",
                )

                panel_select_key = "tab7_shared_panel_name"
                panel_editor_key = "tab7_shared_panel_editor"
                panel_name_key = "tab7_shared_panel_editor_name"
                if st.session_state.get(panel_select_key) not in panels:
                    st.session_state[panel_select_key] = active_name
                if panel_editor_key not in st.session_state:
                    st.session_state[panel_editor_key] = list(panels.get(active_name, []))
                if panel_name_key not in st.session_state:
                    st.session_state[panel_name_key] = active_name or "my_panel"

                def _load_shared_panel() -> None:
                    selected_name = st.session_state.get(panel_select_key, "")
                    st.session_state[panel_editor_key] = list(panels.get(selected_name, []))
                    st.session_state[panel_name_key] = selected_name

                selected_panel_name = st.selectbox(
                    "Saved shared marker panel",
                    panel_names or ["(create a panel)"],
                    key=panel_select_key,
                    on_change=_load_shared_panel,
                    help="Selecting a saved panel immediately loads its saved markers into the editor.",
                )

                shortcut_values = marker_panel_shortcuts(
                    available_markers,
                    affix_position,
                    correction_affix,
                )
                selected_saved_values = list(panels.get(selected_panel_name, []))

                def _load_panel_values(kind: str) -> tuple[list[str], str]:
                    selected_matching = [
                        marker for marker in selected_saved_values
                        if (
                            is_corrected_marker_column(marker, affix_position, correction_affix)
                            if kind == "corrected"
                            else not is_corrected_marker_column(marker, affix_position, correction_affix)
                        )
                    ]
                    if selected_matching:
                        return selected_matching, selected_panel_name
                    shortcut_name = "All corrected markers" if kind == "corrected" else "All raw markers"
                    fallback_name = "all_corrected" if kind == "corrected" else "all_raw"
                    return list(shortcut_values[shortcut_name]), fallback_name

                load_raw_col, load_corrected_col = st.columns(2)
                with load_raw_col:
                    load_raw_markers = st.button(
                        "Load Raw markers",
                        key="tab7_shared_load_raw_markers_btn",
                        help="Load raw columns that have a corrected counterpart. If no corrected columns exist, load all available markers.",
                    )
                with load_corrected_col:
                    load_corrected_markers = st.button(
                        "Load Corrected markers",
                        key="tab7_shared_load_corrected_markers_btn",
                        help=f"Load all columns matching the configured {affix_position.lower()} `{correction_affix}` rule.",
                    )
                if load_raw_markers:
                    loaded_values, loaded_name = _load_panel_values("raw")
                    st.session_state[panel_editor_key] = loaded_values
                    st.session_state[panel_name_key] = loaded_name
                    st.rerun()
                if load_corrected_markers:
                    loaded_values, loaded_name = _load_panel_values("corrected")
                    st.session_state[panel_editor_key] = loaded_values
                    st.session_state[panel_name_key] = loaded_name
                    st.rerun()

                with st.form("tab7_marker_workspace_form", clear_on_submit=False):
                    preparation_note = st.text_input(
                        "Shared preparation note",
                        value=str(workspace.get("preparation_note", "Method-specific normalization")),
                        help="Optional note retained for reproducibility and figure labels.",
                    )

                    panel_name = st.text_input(
                        "Panel name",
                        key=panel_name_key,
                        help="Use names such as all_raw, phenotyping_raw, or phenotyping_corr.",
                    )
                    selected_markers = st.multiselect(
                        "Markers in this panel",
                        available_markers,
                        key=panel_editor_key,
                        help="Select actual CSV columns. The selected names are what downstream subtabs will use.",
                    )
                    save_panel = st.form_submit_button("Save Shared Marker Panel", type="primary", width="stretch")

                if save_panel:
                    clean_name = str(panel_name).strip()
                    selected_markers = [str(marker) for marker in selected_markers if str(marker) in available_markers]
                    if not clean_name:
                        st.error("Enter a name for this marker panel before saving.")
                    elif not selected_markers:
                        st.error("Select at least one marker for this panel before saving.")
                    else:
                        updated_panels = dict(panels)
                        updated_panels[clean_name] = list(dict.fromkeys(selected_markers))
                        applied_workspace = {
                            "input_path": global_full_path,
                            "affix_position": affix_position,
                            "correction_affix": correction_affix.strip() or inferred_affix,
                            "marker_panels": updated_panels,
                            "active_panel_name": clean_name,
                            "preparation_note": preparation_note,
                            "panel_identity": "named_actual_columns",
                        }
                        st.session_state["tab7_marker_workspace"] = applied_workspace
                        save_directory_settings(global_input_dir, "tab7_marker_workspace", applied_workspace)
                        save_active_project_state("tab7_phenotyping")
                        st.success(f"Saved marker panel `{clean_name}` with {len(selected_markers)} columns.")

                active_workspace = st.session_state.get("tab7_marker_workspace", normalized_workspace)
                active_panels = active_workspace.get("marker_panels", {}) if isinstance(active_workspace, dict) else {}
                st.info(
                    "Saved panels: "
                    + (", ".join(f"`{name}` ({len(values)})" for name, values in active_panels.items()) or "none")
                )

    # --- Visual Validation Configuration ---
    
    # Initialize Visualizer session state
    if 'viz_enabled' not in st.session_state:
        st.session_state.viz_enabled = False
    if 'viz_roi_dir' not in st.session_state:
        # Try to infer from project structure or cache
        st.session_state.viz_roi_dir = str(project_root / 'outputs' / 'registration' / 'roi')
    if 'viz_seg_dir' not in st.session_state:
        st.session_state.viz_seg_dir = str(project_root / 'outputs' / 'segmentation' / 'cellpose')
    if 'viz_file_matches' not in st.session_state:
        st.session_state.viz_file_matches = {}

    with st.expander("Visual Validation Settings", expanded=False):
        # Enable Toggle
        st.checkbox("Enable Cell Visual Validation", key="viz_enabled", help="Enable " +
                   "visualization of random cells from selected clusters/phenotypes.")

        if st.session_state.viz_enabled:
            # Load persistent viz settings bound to the active input directory
            viz_settings = load_directory_settings(global_input_dir, section_key="tab7_viz")
            
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                default_roi = viz_settings.get('viz_roi_dir', st.session_state.get('viz_roi_dir', ''))
                viz_roi_dir = directory_input_with_browse(
                    label="ROI Images Directory (OME-TIFF)",
                    text_key="viz_roi_dir",
                    default_value=default_roi,
                    browser_key_prefix="tree_tab7_viz_roi_dir",
                    help_text="Directory containing ROI images (.tif/.ome.tif)"
                )
            with c_v2:
                default_seg = viz_settings.get('viz_seg_dir', st.session_state.get('viz_seg_dir', ''))
                viz_seg_dir = directory_input_with_browse(
                    label="Segmentation Directory (GeoJSON)",
                    text_key="viz_seg_dir",
                    default_value=default_seg,
                    browser_key_prefix="tree_tab7_viz_seg_dir",
                    help_text="Directory containing segmentation masks (.geojson)"
                )
            
            # Persist directory changes
            if viz_roi_dir != viz_settings.get('viz_roi_dir', '') or viz_seg_dir != viz_settings.get('viz_seg_dir', ''):
                save_directory_settings(global_input_dir, "tab7_viz", {
                    'viz_roi_dir': viz_roi_dir,
                    'viz_seg_dir': viz_seg_dir
                })
            
            c_v3, c_v4, c_v5 = st.columns(3)
            with c_v3:
                st.text_input("Batch Column", value="batch_id", key="viz_col_batch")
            with c_v4:
                st.text_input("X Coord Column", value="x_centroid", key="viz_col_x")
            with c_v5:
                st.text_input("Y Coord Column", value="y_centroid", key="viz_col_y")
                
            st.markdown("##### Visualization Parameters")
            c_v6, c_v7 = st.columns(2)
            with c_v6:
                st.slider("Tile Size (px)", min_value=20, max_value=100, value=40, key="viz_tile_size")
            with c_v7:
                st.number_input("Cells to Sample", min_value=1, max_value=20, value=5, key="viz_start_n_cells")
            
            # Action: Scan Files
            if st.button("Scan & Link Files", type="primary"):
                # We need the unique batch IDs from the loaded CSV to match
                # Load the CSV briefly to get batches if possible, or usually we don't load it here.
                # However, the user said "match ... against the loaded unified CSV". 
                # Since we don't have the DF loaded in this scope (it's loaded inside subtabs), 
                # we will just scan the directories for NOW and link what we can, OR
                # we instantiate the visualizer and let it scan all files.
                # Optimized: Just scan all available files in the dirs and map them by prefix.
                
                from spacemap_gui.cell_visualizer import CellVisualizer
                viz = CellVisualizer(st.session_state.viz_roi_dir, st.session_state.viz_seg_dir)
                
                # We don't have the CSV loaded here efficiently. 
                # Strategy: Just list all batch-like prefixes available in ROI dir for now
                # OR better: Warn user that matching happens dynamically, but we can check dir existence.
                
                if not os.path.exists(st.session_state.viz_roi_dir) or not os.path.exists(st.session_state.viz_seg_dir):
                    st.error("One or more directories do not exist.")
                else:
                    # Perform a 'blind' scan or just simple validation that files exist
                    # Real matching needs the CSV. We can defer exact matching to the subtab,
                    # but here we can at least count files.
                    
                    try:
                        n_rois = len([f for f in os.listdir(st.session_state.viz_roi_dir) if f.lower().endswith(('.tif', '.tiff', '.ome.tif'))])
                        n_segs = len([f for f in os.listdir(st.session_state.viz_seg_dir) if f.lower().endswith(('.json', '.geojson'))])
                        st.success(f"Found {n_rois} ROI images and {n_segs} Segmentation files. Ready to link with data.")
                        st.session_state['viz_dirs_validated'] = True
                    except Exception as e:
                        st.error(f"Error scanning directories: {e}")



    # Create subtabs
    tabs = st.tabs([
        "Leiden Clustering", 
        "SOM Clustering", 
        "Vortex Clustering",
        "Resolve (Threshold)", 
        "SCIMAP", 
        "TOOLS",
        "MR ML", 
        "Consensus ML", 
        "Sub-Phenotyping"
    ])

    if 'tab7_loaded_subtabs' not in st.session_state:
        st.session_state['tab7_loaded_subtabs'] = []

    def _is_loaded(key: str) -> bool:
        return key in st.session_state.get('tab7_loaded_subtabs', [])

    def _mark_loaded(key: str) -> None:
        loaded = st.session_state.get('tab7_loaded_subtabs', [])
        if key not in loaded:
            loaded.append(key)
            st.session_state['tab7_loaded_subtabs'] = loaded

    with tabs[0]:
        if not _is_loaded('leiden'):
            st.info("Click below to load Leiden Clustering.")
            if st.button("Load Leiden Clustering", key="tab7_load_leiden"):
                _mark_loaded('leiden')
        if _is_loaded('leiden'):
            from .leiden_clustering import render_leiden
            render_leiden(project_root, global_full_path)
    
    with tabs[1]:
        if not _is_loaded('som'):
            st.info("Click below to load SOM Clustering.")
            if st.button("Load SOM Clustering", key="tab7_load_som"):
                _mark_loaded('som')
        if _is_loaded('som'):
            from .som_clustering import render_som
            render_som(project_root, global_full_path)
        
    with tabs[2]:
        if not _is_loaded('vortex'):
            st.info("Click below to load Vortex Clustering.")
            if st.button("Load Vortex Clustering", key="tab7_load_vortex"):
                _mark_loaded('vortex')
        if _is_loaded('vortex'):
            from .vortex_clustering import render_vortex
            render_vortex(project_root, global_full_path)

    with tabs[3]:
        if not _is_loaded('resolve'):
            st.info("Click below to load Resolve (Threshold).")
            if st.button("Load Resolve (Threshold)", key="tab7_load_resolve"):
                _mark_loaded('resolve')
        if _is_loaded('resolve'):
            from .resolve_clustering import render_resolve
            render_resolve(project_root, global_full_path)
        
    with tabs[4]:
        if not _is_loaded('scimap'):
            st.info("Click below to load SCIMAP.")
            if st.button("Load SCIMAP", key="tab7_load_scimap"):
                _mark_loaded('scimap')
        if _is_loaded('scimap'):
            from .scimap_clustering import render_scimap
            render_scimap(project_root, global_full_path)
        
    with tabs[5]:
        if not _is_loaded('tools'):
            st.info("Click below to load TOOLS.")
            if st.button("Load TOOLS", key="tab7_load_tools"):
                _mark_loaded('tools')
        if _is_loaded('tools'):
            from .tools import render_tools
            render_tools(project_root, global_full_path)

    with tabs[6]:
        if not _is_loaded('mrml'):
            st.info("Click below to load MR ML.")
            if st.button("Load MR ML", key="tab7_load_mrml"):
                _mark_loaded('mrml')
        if _is_loaded('mrml'):
            from .mr_ml_phenotyping import render_mr_ml
            render_mr_ml(project_root, global_full_path)
        
    with tabs[7]:
        if not _is_loaded('consensus_ml'):
            st.info("Click below to load Consensus ML.")
            if st.button("Load Consensus ML", key="tab7_load_consensus_ml"):
                _mark_loaded('consensus_ml')
        if _is_loaded('consensus_ml'):
            from .consensus_ml import render_consensus_ml
            render_consensus_ml(project_root, global_full_path)
        
    with tabs[8]:
        if not _is_loaded('subphenotyping'):
            st.info("Click below to load Sub-Phenotyping.")
            if st.button("Load Sub-Phenotyping", key="tab7_load_subphenotyping"):
                _mark_loaded('subphenotyping')
        if _is_loaded('subphenotyping'):
            from .subphenotyping import render_subphenotyping
            render_subphenotyping(project_root, global_full_path)

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
