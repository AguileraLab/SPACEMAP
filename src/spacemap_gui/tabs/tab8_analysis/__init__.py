"""
Tab 8: Spatial Analysis - modular package

This module mirrors the Tab 7 structure so each subtab can live in its own file.
Add future subtabs by creating a new file and appending to `SUBTAB_REGISTRY`.
"""

from __future__ import annotations

__package__ = "spacemap_gui.tabs.tab8_analysis"

from pathlib import Path

import streamlit as st

from .common import render_global_input_configuration, save_tab8_config_for_csv
from spacemap_gui.process_log import record_process_event

# NOTE: Subtab renderers are imported LAZILY inside their render blocks
# to avoid pulling in scipy / shapely at app startup.


def render():
    """Main render function for Tab 8: Spatial Analysis."""
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    st.session_state["_currently_rendering_tab"] = "tab8"

    csv_path, df = render_global_input_configuration(project_root)
    

        
    if csv_path is None or df is None:
        st.info("Select a valid unified CSV in the global section to enable subtabs.")
        return

    tabs = st.tabs([
        "Feature Screening",
        "Density Analysis",
        "Neighborhoods",
        "Cell-Cell Interaction",
        "Triad",
        "Clonality",
        "More Coming Soon",
    ])

    # ---- helpers for click-to-load gating ----
    if "tab8_loaded_subtabs" not in st.session_state:
        st.session_state["tab8_loaded_subtabs"] = []

    def _is_loaded(key: str) -> bool:
        return key in st.session_state.get("tab8_loaded_subtabs", [])

    def _mark_loaded(key: str) -> None:
        loaded = st.session_state.get("tab8_loaded_subtabs", [])
        if key not in loaded:
            loaded.append(key)
            st.session_state["tab8_loaded_subtabs"] = loaded
            record_process_event(
                f"tab8:{key}",
                "Spatial analysis tool loaded",
                details={"tool": key, "input_csv": csv_path},
            )

    with tabs[0]:
        if not _is_loaded("feature_screening"):
            st.info("Click below to load Feature Screening.")
            if st.button("Load Feature Screening", key="tab8_load_feature_screening"):
                _mark_loaded("feature_screening")
        if _is_loaded("feature_screening"):
            from .feature_screening import render_feature_screening
            render_feature_screening(project_root, df, csv_path)

    with tabs[1]:
        if not _is_loaded("density"):
            st.info("Click below to load Density Analysis.")
            if st.button("Load Density Analysis", key="tab8_load_density"):
                _mark_loaded("density")
        if _is_loaded("density"):
            from .density_analysis import render_density_analysis
            render_density_analysis(project_root, df)

    with tabs[2]:
        if not _is_loaded("neighborhoods"):
            st.info("Click below to load Neighborhoods.")
            if st.button("Load Neighborhoods", key="tab8_load_neighborhoods"):
                _mark_loaded("neighborhoods")
        if _is_loaded("neighborhoods"):
            from .neighborhoods import render_neighborhoods
            render_neighborhoods(project_root, df, csv_path)

    with tabs[3]:
        if not _is_loaded("cci"):
            st.info("Click below to load Cell-Cell Interaction.")
            if st.button("Load Cell-Cell Interaction", key="tab8_load_cci"):
                _mark_loaded("cci")
        if _is_loaded("cci"):
            from .cell_cell_interaction import render_cell_cell_interaction
            render_cell_cell_interaction(project_root, df)

    with tabs[4]:
        if not _is_loaded("triad"):
            st.info("Click below to load Triad analysis.")
            if st.button("Load Triad", key="tab8_load_triad"):
                _mark_loaded("triad")
        if _is_loaded("triad"):
            from .triads import render_triad_analysis
            render_triad_analysis(project_root, df, csv_path)

    with tabs[5]:
        if not _is_loaded("clonality"):
            st.info("Click below to load Clonality analysis.")
            if st.button("Load Clonality", key="tab8_load_clonality"):
                _mark_loaded("clonality")
        if _is_loaded("clonality"):
            from .clonality import render_clonality
            render_clonality(project_root, df, csv_path)

    with tabs[6]:
        st.info(
            "This modular Tab 8 package is ready for extension. "
            "Add new subtab files under `interface_gui/tabs/tab8_analysis/` and wire them in `__init__.py`."
        )

    # Persist all Tab 8 configuration keys per active CSV so they reload automatically
    # when this same unified CSV is selected again.
    save_tab8_config_for_csv(csv_path)

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
