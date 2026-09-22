"""Tab 9: modular visualization package."""

from __future__ import annotations

__package__ = "spacemap_gui.tabs.tab9_visualization"

import streamlit as st

from .common import (
    get_cached_unified_df,
    get_csv_columns_fast,
    get_output_dir_default,
    load_cached_params,
    persist_cached_params,
    render_global_input_configuration,
    render_unified_csv_loader,
    render_visual_validation_settings,
)


from spacemap_gui.persistence import save_active_project_state
from spacemap_gui.process_log import record_process_event

def render() -> None:
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    st.session_state["_currently_rendering_tab"] = "tab9"

    st.markdown("## Step 9: Visualization")
    st.markdown("Publication-ready comparative visualizations and validation surfaces for phenotyping outputs.")

    unified_csv_path, global_input_dir = render_global_input_configuration(project_root)

    # Same fast copy Tab 8 offers; both tabs read the same unified table.
    from spacemap_gui.ui_utils import render_fast_copy_controls

    render_fast_copy_controls(unified_csv_path, key_prefix="tab9")

    cached_params = load_cached_params(project_root)
    loaded_df = render_unified_csv_loader(unified_csv_path)
    available_columns = list(loaded_df.columns) if loaded_df is not None else get_csv_columns_fast(unified_csv_path or "")
    
    viz_settings = render_visual_validation_settings(project_root, global_input_dir, available_columns)
    output_dir_default = get_output_dir_default(project_root, cached_params)



    tabs = st.tabs([
        "Proportion",
        "Pairwise",
        "F1 Score",
        "Heatmap",
        "Visual Validation",
        "UMAP Explorer",
        "Cell Visualization",
        "Virtual Tissue",
        "Actual Tissue",
        "Marker Analysis",
    ])

    if "tab9_loaded_subtabs" not in st.session_state:
        st.session_state["tab9_loaded_subtabs"] = []

    def _is_loaded(key: str) -> bool:
        return key in st.session_state.get("tab9_loaded_subtabs", [])

    def _mark_loaded(key: str) -> None:
        loaded = st.session_state.get("tab9_loaded_subtabs", [])
        if key not in loaded:
            loaded.append(key)
            st.session_state["tab9_loaded_subtabs"] = loaded
            record_process_event(
                f"tab9:{key}",
                "Visualization tool loaded",
                details={"tool": key, "input_csv": unified_csv_path},
            )

    def _get_tissue_df():
        if loaded_df is not None:
            return loaded_df
        if not unified_csv_path:
            st.error("Unified CSV file not found. Please set it in Global Input Configuration.")
            return None
        try:
            return get_cached_unified_df(unified_csv_path)
        except Exception as exc:
            st.error(f"Failed to load unified CSV: {exc}")
            return None

    with tabs[0]:
        if not _is_loaded("proportion"):
            st.info("Click below to load Proportion.")
            if st.button("Load Proportion", key="tab9_load_proportion"):
                _mark_loaded("proportion")
        if _is_loaded("proportion"):
            from .proportion import render_proportion

            render_proportion(project_root, unified_csv_path, available_columns, cached_params, output_dir_default)

    with tabs[1]:
        if not _is_loaded("pairwise"):
            st.info("Click below to load Pairwise.")
            if st.button("Load Pairwise", key="tab9_load_pairwise"):
                _mark_loaded("pairwise")
        if _is_loaded("pairwise"):
            from .pairwise import render_pairwise

            render_pairwise(project_root, unified_csv_path, available_columns, cached_params, output_dir_default)

    with tabs[2]:
        if not _is_loaded("f1"):
            st.info("Click below to load F1 Score.")
            if st.button("Load F1 Score", key="tab9_load_f1"):
                _mark_loaded("f1")
        if _is_loaded("f1"):
            from .f1_validation_ui import render_f1_score

            render_f1_score(project_root, unified_csv_path, available_columns, cached_params, output_dir_default)

    with tabs[3]:
        if not _is_loaded("heatmap"):
            st.info("Click below to load Heatmap.")
            if st.button("Load Heatmap", key="tab9_load_heatmap"):
                _mark_loaded("heatmap")
        if _is_loaded("heatmap"):
            from .heatmap import render_heatmap

            render_heatmap(project_root, unified_csv_path, available_columns, cached_params, output_dir_default)

    with tabs[4]:
        if not _is_loaded("visual_validation"):
            st.info("Click below to load Visual Validation.")
            if st.button("Load Visual Validation", key="tab9_load_visual_validation"):
                _mark_loaded("visual_validation")
        if _is_loaded("visual_validation"):
            from .visual_validation import render_visual_validation

            render_visual_validation(
                project_root,
                unified_csv_path,
                available_columns,
                cached_params,
                viz_settings,
            )

    with tabs[5]:
        if not _is_loaded("umap"):
            st.info("Click below to load UMAP Explorer.")
            if st.button("Load UMAP Explorer", key="tab9_load_umap"):
                _mark_loaded("umap")
        if _is_loaded("umap"):
            from .umap_explorer import render_umap_explorer

            render_umap_explorer(project_root, unified_csv_path, available_columns, cached_params)

    with tabs[6]:
        if not _is_loaded("cellviz"):
            st.info("Click below to load Cell Visualization.")
            if st.button("Load Cell Visualization", key="tab9_load_cellviz"):
                _mark_loaded("cellviz")
        if _is_loaded("cellviz"):
            from .cell_visualization import render_cell_visualization

            render_cell_visualization(
                project_root,
                unified_csv_path or "",
                available_columns,
                cached_params,
                viz_settings,
                loaded_df=loaded_df,
            )

    with tabs[7]:
        if not _is_loaded("virtual_tissue"):
            st.info("Click below to load Virtual Tissue.")
            if st.button("Load Virtual Tissue", key="tab9_load_virtual_tissue"):
                _mark_loaded("virtual_tissue")
        if _is_loaded("virtual_tissue"):
            from .virtual_tissue import render_virtual_tissue

            tissue_df = _get_tissue_df()
            if tissue_df is not None:
                render_virtual_tissue(project_root, tissue_df)

    with tabs[8]:
        if not _is_loaded("actual_tissue"):
            st.info("Click below to load Actual Tissue.")
            if st.button("Load Actual Tissue", key="tab9_load_actual_tissue"):
                _mark_loaded("actual_tissue")
        if _is_loaded("actual_tissue"):
            from .actual_tissue import render_actual_tissue

            tissue_df = _get_tissue_df()
            if tissue_df is not None:
                render_actual_tissue(project_root, tissue_df, viz_settings)

    with tabs[9]:
        if not _is_loaded("marker_analysis"):
            st.info("Click below to load Marker Analysis.")
            if st.button("Load Marker Analysis", key="tab9_load_marker_analysis"):
                _mark_loaded("marker_analysis")
        if _is_loaded("marker_analysis"):
            from .marker_analysis import render_marker_analysis

            render_marker_analysis(project_root)

    persist_cached_params(project_root, cached_params, unified_csv_path)

    # Automated JSON Sync: automatically save step 9 configurations on every render/rerun
    try:
        save_active_project_state("tab9_visualization")
    except Exception:
        pass

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
