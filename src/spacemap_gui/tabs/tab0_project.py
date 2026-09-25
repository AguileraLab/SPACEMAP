import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import streamlit as st
from spacemap_gui.ui_utils import directory_input_with_browse
from spacemap_gui.persistence import (
    load_projects_index,
    save_project_to_index,
    remove_project_from_index,
    load_project_config,
    save_project_config,
    save_active_project_state,
)
from spacemap_gui import terminal

# Global list of all directory keys used across all pipeline tabs.
# This is the SINGLE SOURCE OF TRUTH for directory-style session keys.
# `app.py` imports this as PERSISTENT_DIRECTORY_KEYS for auto-save, and
# `persistence.py` uses it for project config sync. Keep in sync with
# LIST_DIRECTORY_KEYS below for any keys that hold lists instead of strings.
DIRECTORY_KEYS = [
    # Registration
    "batch_codex_dir", "batch_he_dir", "tab1_batch_output_dir", "registration_output_dir",
    # Registration: Multi-stack (Tab 1 subtab 2)
    "multi_r1_dir", "multi_r2_dir", "multi_he_dir", "multi_out_dir",
    "reg_r1_dir", "reg_r2_dir", "reg_he_dir",
    # Background Subtraction
    "tab2_input_dir_reg", "tab2_input_dir_raw", "tab2_output_dir", "bg_subtraction_output_dir",
    # Background threshold tables (shared by Tab 2 and RESOLVE)
    "threshold_dir",
    # ROI Extraction
    "tab3_input_dir", "roi_output_dir", "tab3_output_dir", "tab3_geojson_dir",
    "tab3_multi_input_dir", "tab3_multi_geojson_dir", "tab3_multi_output_dir",
    # Segmentation
    "tab4_input_dir", "tab4_output_dir", "tab4_seg_mask_save_dir", "tab4_csv_dir", "tab4_geojson_dir",
    "tab4_annot_geojson_dir",
    "tab4_custom_model",
    "batch_input_dir", "batch_geojson_dir", "batch_output_dir", "batch_custom_model",
    # Annotation: tissue annotation (pixel classification)
    "annot_ta_input_dir", "annot_ta_output_dir", "annot_ta_pool_input_dir", "annot_ta_pool_output_dir",
    # Annotation: cell annotation & create zones
    "annot_input_dirs", "annot_image_dir", "annot_output_dir", "zone_input_dir", "zone_output_dir",
    # Cell Annotation & Visualization
    "cell_seg_dir", "cell_zone_dir", "cell_out_dir", "vis_csv_dir", "vis_img_dir",
    # Batch Correction
    "bc_input_dir", "bc_output_dir",
    # Phenotyping
    "scimap_bc_dir", "global_pheno_dir", "phenotyping_output_dir", "viz_roi_dir", "viz_seg_dir",
    # Spatial Analysis
    "spatial_input_dir", "spatial_output_dir", "global_spatial_dir",
    # Visualization Mappings (Tab 9)
    "tab9_actual_image_dir", "tab9_actual_roi_geojson_dir", "tab9_actual_overlay_geojson_dir", "tab9_actual_output_dir",
    "tab9_marker_single_image_dir", "tab9_marker_single_geojson_dir", "tab9_marker_single_output_dir",
    "tab9_marker_batch_image_dir", "tab9_marker_batch_geojson_dir", "tab9_marker_batch_output_dir",
    "global_visualization_dir",
]

# Directory keys that hold lists instead of single strings
LIST_DIRECTORY_KEYS = {"annot_input_dirs"}

PIPELINE_STATUS_STAGES = (
    ("1. Registration", "tab1_batch_output_dir"),
    ("2. Background subtraction", "tab2_output_dir"),
    ("3. ROI extraction", "tab3_output_dir"),
    ("4. Annotation", "cell_out_dir"),
    ("5. Segmentation", "tab4_output_dir"),
    ("6. Batch correction", "bc_output_dir"),
    ("7. Phenotyping", "phenotyping_output_dir"),
    ("8. Spatial analysis", "spatial_output_dir"),
    ("9. Visualization", "tab9_actual_output_dir"),
)


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab0', message, step=None, level=level)


def get_pipeline_status(directory_state: Mapping[str, object]) -> list[dict[str, object]]:
    """Summarize configured pipeline outputs without recursively scanning project data."""
    rows: list[dict[str, object]] = []
    for stage, directory_key in PIPELINE_STATUS_STAGES:
        directory_value = directory_state.get(directory_key, "")
        directory_path = Path(str(directory_value)).expanduser() if directory_value else None
        row: dict[str, object] = {
            "Stage": stage,
            "Status": "Not configured",
            "Files": 0,
            "Folders": 0,
            "Output directory": str(directory_path) if directory_path else "",
        }

        if directory_path is None:
            rows.append(row)
            continue
        if not directory_path.exists():
            row["Status"] = "Missing"
            rows.append(row)
            continue
        if not directory_path.is_dir():
            row["Status"] = "Not a directory"
            rows.append(row)
            continue

        try:
            entries = list(directory_path.iterdir())
        except OSError:
            row["Status"] = "Unreadable"
            rows.append(row)
            continue

        row["Files"] = sum(entry.is_file() for entry in entries)
        row["Folders"] = sum(entry.is_dir() for entry in entries)
        row["Status"] = "Ready" if entries else "Empty"
        rows.append(row)

    return rows

def discover_image_pairs(codex_dir: str, he_dir: str) -> list:
    """Discover raw CODEX and H&E images and match them by base name."""
    codex_path = Path(codex_dir) if codex_dir else None
    he_path = Path(he_dir) if he_dir else None

    # Supported extensions
    extensions = (".qptiff", ".tiff", ".tif")
    
    # List CODEX files
    codex_files = []
    if codex_path and codex_path.exists():
        try:
            codex_files = sorted([f for f in os.listdir(codex_path) if f.lower().endswith(extensions)])
        except Exception as e:
            _log_console(f"Error listing CODEX dir: {e}", level="error")

    # List H&E files
    he_files = []
    if he_path and he_path.exists():
        try:
            he_files = sorted([f for f in os.listdir(he_path) if f.lower().endswith(extensions)])
        except Exception as e:
            _log_console(f"Error listing H&E dir: {e}", level="error")

    # Helper to extract core stem
    def get_stem(filename):
        stem = Path(filename).name
        for ext in sorted(extensions, key=len, reverse=True):
            if stem.lower().endswith(ext):
                stem = stem[:-len(ext)]
                break
        return stem

    # Align pairs by base stem matching
    pairs = []
    for f in codex_files:
        stem = get_stem(f)
        matched_he = "-"
        matched_he_path = "-"
        if he_files:
            for he_f in he_files:
                he_stem = get_stem(he_f)
                if he_stem == stem or he_stem.startswith(stem) or stem.startswith(he_stem):
                    matched_he = he_f
                    matched_he_path = str(he_path / he_f)
                    break
        pairs.append({
            "identity": stem,
            "codex_filename": f,
            "codex_path": str(codex_path / f),
            "he_filename": matched_he,
            "he_path": matched_he_path,
        })

    return pairs

def paths_equal(p1, p2) -> bool:
    if p1 == p2:
        return True
    if p1 is None or p2 is None:
        return False
    if isinstance(p1, list) and isinstance(p2, list):
        if len(p1) != len(p2):
            return False
        return all(paths_equal(x, y) for x, y in zip(p1, p2))
    if isinstance(p1, list) or isinstance(p2, list):
        return False
    try:
        return os.path.realpath(str(p1)) == os.path.realpath(str(p2))
    except:
        return False

def sync_directory_states():
    """Bidirectionally synchronize global directory state and Project tab widget keys."""
    from spacemap_gui.persistence import sync_project_state
    sync_project_state()


def unload_project_callback():
    """Unload the active project and clear its session state gracefully."""
    # Wipe the persistent active session
    from spacemap_gui.persistence import save_active_session
    save_active_session("-- No Project Loaded --")
    
    # Save any active project config changes before unloading
    save_project_callback()
    
    # Set the sidebar sync flag
    st.session_state["_force_sidebar_proj_sync"] = True
    
    # Reset all directory and config settings in session state
    for key in list(st.session_state.keys()):
        if key in ["global_accent_color", "global_border_radius", "global_tab_font_size", 
                   "global_content_font_size", "global_tab_align", "global_cmap", 
                   "global_thumb_scale", "global_theme_mode", "_force_sidebar_proj_sync"]:
            continue
        del st.session_state[key]


def get_default_directory_mappings(project_root_str: str, codex_dir: str, he_dir: str) -> dict:
    """Retrieve default directory paths relative to the project root."""
    p_root = Path(project_root_str)
    outputs_dir = p_root / "outputs"
    subdirs = {
        "registered": outputs_dir / "registered",
        "background_subtracted": outputs_dir / "background_subtracted",
        "rois": outputs_dir / "rois",
        "segmentation": outputs_dir / "segmentation",
        "processed_annotations": outputs_dir / "processed_annotations",
        "annotated_csv": outputs_dir / "annotated_csv",
        "batch_corrected": outputs_dir / "batch_corrected",
        "phenotyping": outputs_dir / "phenotyping",
        "spatial_analysis": outputs_dir / "spatial_analysis",
        "project_boundaries": p_root / "data/boundaries",
        "project_seg_model": p_root / "data/seg_model",
        "project_thresholds": p_root / "data/thresholds",
    }
    
    return {
        # Tab 1: Registration
        "batch_codex_dir": codex_dir,
        "batch_he_dir": he_dir if he_dir else codex_dir,
        "tab1_batch_output_dir": str(subdirs["registered"]),
        "registration_output_dir": str(subdirs["registered"]),
        
        # Tab 2: Background Subtraction
        "tab2_input_dir_reg": str(subdirs["registered"]),
        "tab2_input_dir_raw": codex_dir,
        "tab2_output_dir": str(subdirs["background_subtracted"]),
        "bg_subtraction_output_dir": str(subdirs["background_subtracted"]),
        "threshold_dir": str(subdirs["project_thresholds"]),
        
        # Tab 3: ROI Extraction
        "tab3_input_dir": str(subdirs["background_subtracted"]),
        "roi_output_dir": str(subdirs["rois"]),
        "tab3_output_dir": str(subdirs["rois"]),
        "tab3_geojson_dir": str(subdirs["project_boundaries"]),
        "tab3_multi_input_dir": str(subdirs["background_subtracted"]),
        "tab3_multi_geojson_dir": str(subdirs["project_boundaries"]),
        "tab3_multi_output_dir": str(subdirs["rois"]),
        
        # Tab 5: Segmentation
        "tab4_input_dir": str(subdirs["rois"]),
        "tab4_annot_geojson_dir": str(subdirs["processed_annotations"] / "stage1b_compartments_zones"),
        "tab4_output_dir": str(subdirs["segmentation"]),
        "tab4_seg_mask_save_dir": str(subdirs["project_boundaries"]),
        "tab4_csv_dir": str(subdirs["segmentation"] / "cellpose/features"),
        "tab4_geojson_dir": str(subdirs["segmentation"] / "cellpose/boundaries"),
        "batch_input_dir": str(subdirs["rois"]),
        "batch_geojson_dir": str(subdirs["project_boundaries"]),
        "batch_output_dir": str(subdirs["segmentation"]),
        
        # Tab 4: Tissue annotation (pixel classification)
        "annot_ta_input_dir": str(subdirs["rois"]),
        "annot_ta_output_dir": str(subdirs["processed_annotations"] / "stage0_tissue_annotation"),
        "annot_ta_pool_input_dir": str(subdirs["processed_annotations"] / "stage0_tissue_annotation"),
        "annot_ta_pool_output_dir": str(subdirs["processed_annotations"] / "stage1b_compartments_zones"),

        # Tab 4: Annotation & Create Zones
        "annot_input_dirs": [str(p_root / 'data/annotations')],
        "annot_image_dir": codex_dir,
        "annot_output_dir": str(subdirs["processed_annotations"] / "stage1a_merged_tissue"),
        "zone_input_dir": str(subdirs["processed_annotations"] / "stage1a_merged_tissue"),
        "zone_output_dir": str(subdirs["processed_annotations"] / "stage1b_compartments_zones"),
        
        # Tab 5: Cell Annotation
        "cell_seg_dir": str(subdirs["segmentation"]),
        "cell_zone_dir": str(subdirs["processed_annotations"] / "stage1b_compartments_zones"),
        "cell_out_dir": str(subdirs["annotated_csv"]),
        
        # Tab 5: Visualization
        "vis_csv_dir": str(subdirs["batch_corrected"]),
        "vis_img_dir": str(subdirs["rois"]),
        
        # Tab 6: Batch Correction
        "bc_input_dir": str(subdirs["annotated_csv"]),
        "bc_output_dir": str(subdirs["batch_corrected"]),

        # Tab 7: Phenotyping
        "scimap_bc_dir": str(subdirs["batch_corrected"]),
        "global_pheno_dir": str(subdirs["batch_corrected"]),
        "phenotyping_output_dir": str(subdirs["phenotyping"]),
        "viz_roi_dir": str(subdirs["rois"]),
        "viz_seg_dir": str(subdirs["segmentation"]),

        # Tab 8: Spatial Analysis
        "spatial_input_dir": str(subdirs["batch_corrected"]),
        "spatial_output_dir": str(subdirs["spatial_analysis"]),
        "global_spatial_dir": str(subdirs["batch_corrected"]),

        # Tab 9: Visualization Mappings
        "tab9_actual_image_dir": str(subdirs["rois"]),
        "tab9_actual_roi_geojson_dir": str(subdirs["project_boundaries"]),
        "tab9_actual_overlay_geojson_dir": str(subdirs["segmentation"]),
        "tab9_actual_output_dir": str(subdirs["spatial_analysis"]),
        "tab9_marker_single_image_dir": str(subdirs["rois"]),
        "tab9_marker_single_geojson_dir": str(subdirs["segmentation"]),
        "tab9_marker_single_output_dir": str(subdirs["spatial_analysis"]),
        "tab9_marker_batch_image_dir": str(subdirs["rois"]),
        "tab9_marker_batch_geojson_dir": str(subdirs["segmentation"]),
        "tab9_marker_batch_output_dir": str(subdirs["spatial_analysis"]),
        "global_visualization_dir": str(subdirs["batch_corrected"]),
    }


def load_project_callback(project_root: str) -> bool:
    """Load project config and hydrate session state exclusively from it.

    Returns False when nothing could be loaded (missing folder or file), so
    callers can report it instead of retrying.
    """
    config = load_project_config(project_root)
    if not config:
        return False
        
    from spacemap_gui.persistence import hydrate_project_state
    hydrate_project_state(config, project_root)
    
    # Register project in index
    project_name = config.get("project_name", "imported_project")
    save_project_to_index(project_name, project_root)
    
    # Save the persistent active session so it survives restarts/refreshes
    from spacemap_gui.persistence import save_active_session
    save_active_session(project_name)
    
    # Force sidebar selectbox synchronization on rerun
    st.session_state["_force_sidebar_proj_sync"] = True
    return True


def create_project_callback():
    """Retrieve UI inputs and create project structure."""
    if "new_proj_error" in st.session_state:
        del st.session_state["new_proj_error"]
    if "new_proj_success" in st.session_state:
        del st.session_state["new_proj_success"]

    project_name = st.session_state.get("new_proj_name", "").strip()
    parent_dir = st.session_state.get("proj_parent_dir", "").strip()

    if not project_name:
        st.session_state["new_proj_error"] = "Project Name cannot be empty."
        return
    if not re.match(r"^[a-zA-Z0-9_-]+$", project_name):
        st.session_state["new_proj_error"] = "Project Name can only contain alphanumeric characters, underscores, and hyphens (no spaces)."
        return

    if not parent_dir:
        st.session_state["new_proj_error"] = "Parent Experiments Directory cannot be empty."
        return
    if not os.path.exists(parent_dir):
        st.session_state["new_proj_error"] = f"Parent Experiments Directory does not exist: {parent_dir}"
        return
    if not os.path.isdir(parent_dir):
        st.session_state["new_proj_error"] = f"Parent Experiments Directory is not a valid directory: {parent_dir}"
        return

    try:
        proj_root_path = create_project_structure(
            project_name=project_name,
            parent_dir=parent_dir
        )
        
        load_project_callback(proj_root_path)
        st.session_state["new_proj_success"] = f"Project '{project_name}' successfully created and initialized!"
    except Exception as e:
        st.session_state["new_proj_error"] = f"Failed to create project: {str(e)}"


def save_project_callback():
    """Explicitly save the active project state back to disk."""
    if "active_project" not in st.session_state or not st.session_state["active_project"]:
        return
    save_active_project_state()


def remove_project_callback(project_name: str):
    """Remove a project from the index."""
    remove_project_from_index(project_name)


def create_project_structure(project_name: str, parent_dir: str) -> str:
    """Create directory structure and save initialized project config."""
    if Path(parent_dir).name == "project":
        project_root = Path(parent_dir) / project_name
    else:
        project_root = Path(parent_dir) / "project" / project_name
        
    project_root.mkdir(parents=True, exist_ok=True)

    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    
    subdirs = {
        # Data folders
        "codex_raw": data_dir / "codex",
        "he_raw": data_dir / "he",
        "project_boundaries": data_dir / "boundaries",
        "project_seg_model": data_dir / "seg_model",
        "project_annotations": data_dir / "annotations",
        "project_thresholds": data_dir / "thresholds",
        # Outputs folders
        "registered": outputs_dir / "registered",
        "background_subtracted": outputs_dir / "background_subtracted",
        "rois": outputs_dir / "rois",
        "segmentation": outputs_dir / "segmentation",
        "processed_annotations": outputs_dir / "processed_annotations",
        "annotated_csv": outputs_dir / "annotated_csv",
        "batch_corrected": outputs_dir / "batch_corrected",
        "phenotyping": outputs_dir / "phenotyping",
        "spatial_analysis": outputs_dir / "spatial_analysis",
    }

    # Ensure all directories exist
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    codex_dir = str(subdirs["codex_raw"])
    he_dir = str(subdirs["he_raw"])

    # Discover image pairs
    pairs = discover_image_pairs(codex_dir, he_dir)

    # Load system defaults for structure mapping
    from spacemap_gui.persistence import get_all_system_defaults, get_tab_name_from_key
    defaults = get_all_system_defaults(str(project_root))
    
    directory_mappings = {}
    for key in DIRECTORY_KEYS:
        directory_mappings[key] = defaults.get(key, "")
        
    directory_mappings["batch_codex_dir"] = codex_dir
    directory_mappings["batch_he_dir"] = he_dir

    tab_configurations = {}
    for tab_name in ["tab1_registration", "tab2_background_subtraction", "tab3_roi_extraction", 
                     "tab4_annotation", "tab5_segmentation", "tab6_batch_correction", 
                     "tab7_phenotyping", "tab8_analysis", "tab9_visualization"]:
        tab_configurations[tab_name] = {}
        
    for k, v in defaults.items():
        if k in DIRECTORY_KEYS or k.startswith("proj_"):
            continue
        t_group = get_tab_name_from_key(k)
        if t_group:
            tab_configurations[t_group][k] = v

    config = {
        "project_name": project_name,
        "project_root": str(project_root),
        "codex_raw_dir": codex_dir,
        "he_raw_dir": he_dir,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "discovered_pairs": pairs,
        "directory_mappings": directory_mappings,
        "tab_configurations": tab_configurations,
        "mpp_codex": defaults["proj_mpp_codex"],
        "mpp_he": defaults["proj_mpp_he"],
        "use_gpu": defaults["proj_use_gpu"],
        "gpu_id": defaults["proj_gpu_id"],
        "num_workers": defaults["proj_num_workers"],
        "pi": "",
        "institution": "",
        "description": "",
        "unified_csv_filename": defaults["proj_unified_csv_filename"],
        "audit_log": [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tab_affected": "Project Configurations",
                "parameter_changed": "project_name",
                "old_value": "None",
                "new_value": project_name
            }
        ]
    }

    save_project_config(str(project_root), config)
    save_project_to_index(project_name, str(project_root))

    return str(project_root)


@st.cache_data(ttl=5)
def check_dir_exists(path_str: str) -> bool:
    if not path_str:
        return False
    try:
        p = Path(path_str)
        return p.exists() and p.is_dir()
    except Exception:
        return False


def generate_interactive_tree_html() -> str:
    """Generate a clean, collapsible, and hierarchical HTML directory tree map with status badges."""
    active_proj_name = st.session_state.get('active_project_name', 'Project')
    active_proj_root = st.session_state.get('active_project_root', '')
    
    import textwrap
    html = textwrap.dedent("""
    <style>
    .tree-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #E5E9F0;
        background: #111827;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1F2937;
        margin-bottom: 20px;
    }
    .tree-node {
        margin-left: 18px;
        border-left: 1px dashed #374151;
        padding-left: 12px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .tree-leaf {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 8px;
        border-radius: 6px;
        transition: background-color 0.2s;
        margin-bottom: 4px;
    }
    .tree-leaf:hover {
        background-color: #1F2937;
    }
    .badge-exists {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: auto;
    }
    .badge-missing {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: auto;
    }
    .badge-notset {
        background-color: rgba(156, 163, 175, 0.15);
        color: #9CA3AF;
        border: 1px solid rgba(156, 163, 175, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: auto;
    }
    .path-text {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.8rem;
        color: #9CA3AF;
        word-break: break-all;
        background-color: rgba(0, 0, 0, 0.25);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .folder-summary {
        cursor: pointer;
        font-weight: bold;
        list-style: none;
        display: flex;
        align-items: center;
        gap: 8px;
        outline: none;
        padding: 6px 10px;
        border-radius: 6px;
    }
    .folder-summary:hover {
        background-color: #1F2937;
    }
    .folder-summary::-webkit-details-marker {
        display: none;
    }
    </style>
    <div class="tree-container">
    """)
    
    html += textwrap.dedent(f"""
    <details open>
        <summary class="folder-summary">
            <span style="font-size: 1.05rem; color: #F3F4F6;">{active_proj_name}</span>
            <span style="font-size: 0.8rem; background: #374151; padding: 2px 8px; border-radius: 10px; color: #60A5FA; margin-left: auto;">Root Workspace</span>
        </summary>
        <div class="tree-node">
            <div style="font-size: 0.8rem; color: #9CA3AF; margin-bottom: 10px; padding-left: 8px;">
                Workspace Root Path: <code class="path-text">{active_proj_root}</code>
            </div>
    """)
    
    def gen_item_html(label: str, key: str) -> str:
        path = st.session_state.get(key, "")
        if not path:
            status_class = "badge-notset"
            status_text = "Not Set"
            path_display = "No path specified"
        else:
            if check_dir_exists(path):
                status_class = "badge-exists"
                status_text = "Exists"
                path_display = str(path)
            else:
                status_class = "badge-missing"
                status_text = "Missing"
                path_display = str(path)
                
        return textwrap.dedent(f"""
        <div class="tree-leaf">
            <span style="font-weight: 500; color: #E5E9F0;">{label}</span>
            <span class="path-text">{path_display}</span>
            <span class="{status_class}">{status_text}</span>
        </div>
        """)
        
    # Registration
    html += textwrap.dedent("""
    <details open style="margin-top: 8px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">Registration & Inputs</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("CODEX Raw Input", "batch_codex_dir")
    html += gen_item_html("H&E Raw Input", "batch_he_dir")
    html += gen_item_html("Registered Output", "tab1_batch_output_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    # Background Subtraction
    html += textwrap.dedent("""
    <details open style="margin-top: 6px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">Background Subtraction</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("Registered Input", "tab2_input_dir_reg")
    html += gen_item_html("Subtracted Output", "tab2_output_dir")
    html += gen_item_html("Threshold Tables (Tab 2 / RESOLVE)", "threshold_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    # ROI Extraction
    html += textwrap.dedent("""
    <details open style="margin-top: 6px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">ROI Extraction</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("Images Input", "tab3_input_dir")
    html += gen_item_html("Boundaries (GeoJSON)", "tab3_geojson_dir")
    html += gen_item_html("ROIs Directory", "tab3_output_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    # Segmentation
    html += textwrap.dedent("""
    <details open style="margin-top: 6px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">Cell Segmentation</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("ROI Images Input", "tab4_input_dir")
    html += gen_item_html("Segmentation Masks Save", "tab4_seg_mask_save_dir")
    html += gen_item_html("Features Output", "tab4_output_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    # Annotation & Zones
    html += textwrap.dedent("""
    <details open style="margin-top: 6px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">Annotation & Zones</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("Merged Tissue Annotations", "annot_output_dir")
    html += gen_item_html("Compartments/Zones Output", "zone_output_dir")
    html += gen_item_html("Cell Segmentation Mask Input", "cell_seg_dir")
    html += gen_item_html("Final Zone-Annotated Cell CSV", "cell_out_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    # Batch Correction
    html += textwrap.dedent("""
    <details open style="margin-top: 6px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">Batch Correction</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("Annotated CSV Input", "bc_input_dir")
    html += gen_item_html("Batch Corrected Output", "bc_output_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    # Phenotyping & Analysis
    html += textwrap.dedent("""
    <details open style="margin-top: 6px;">
        <summary class="folder-summary">
            <span style="color: #D1D5DB; font-size: 0.9rem;">Phenotyping & Spatial Analysis</span>
        </summary>
        <div class="tree-node">
    """)
    html += gen_item_html("SCIMAP Input (Batch Corrected)", "scimap_bc_dir")
    html += gen_item_html("Phenotyping Clusters Output", "phenotyping_output_dir")
    html += gen_item_html("Spatial Relationships Output", "spatial_output_dir")
    html += textwrap.dedent("""
    </div>
    </details>
    """)
    
    html += textwrap.dedent("""
    </div>
    </details>
    </div>
    """)
    return html

def render():
    from pathlib import Path
    import os
    
    current_dir = Path(os.path.abspath(__file__)).parent
    project_root_fallback = current_dir.parent.parent.parent
    project_root_default = Path(st.session_state.get('active_project_root') or project_root_fallback)
    
    new_project_parent_default = project_root_fallback / "project"
    if not new_project_parent_default.exists():
        new_project_parent_default = project_root_fallback
        
    new_project_codex_default = project_root_fallback / "data/codex"
    new_project_he_default = project_root_fallback / "data/he"

    # Execute bidirectional synchronization before rendering any widgets
    sync_directory_states()

    st.markdown(
        """
        <style>
        .section-header {
            font-size: 20px;
            font-weight: bold;
            color: #4A90E2;
            margin-top: 15px;
            margin-bottom: 10px;
        }
        .info-card {
            background-color: #1e2630;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #4A90E2;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Project Workspace and Configurations")
    st.markdown("Initialize a new sandboxed analysis project, organize directories across all pipeline tabs, and configure project-wide variables.")

    # Load project list
    projects = load_projects_index()
    active_project = st.session_state.get("active_project", None)

    # ACTIVE PROJECT CONTAINER
    if active_project:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Active Project: {st.session_state.get('active_project_name')}</h3>
                <p><b>Root Location:</b> <code>{st.session_state.get('active_project_root')}</code><br>
                <b>Created:</b> {datetime.fromisoformat(active_project.get("created_at", datetime.now().isoformat())).strftime("%Y-%m-%d %H:%M")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tabs for Active Project Management
        proj_tabs = st.tabs(["Workspace Tree Map", "Project Configurations", "Pipeline Directories",
                             "Configuration Audit Log", "Cache Management", "Portability & Restore Points"])

        with proj_tabs[0]:
            configured_directories = active_project.get("directory_mappings", {})
            pipeline_status = get_pipeline_status({
                key: st.session_state.get(key, configured_directories.get(key, ""))
                for _, key in PIPELINE_STATUS_STAGES
            })
            status_counts = {
                status: sum(row["Status"] == status for row in pipeline_status)
                for status in ("Ready", "Empty", "Missing", "Not configured")
            }

            st.markdown("### Pipeline Output Status")
            metric_columns = st.columns(4)
            for column, (label, count) in zip(metric_columns, status_counts.items()):
                column.metric(label, count)
            st.dataframe(pipeline_status, hide_index=True, width="stretch")
            st.caption("File and folder counts are limited to each output directory's first level.")

            st.markdown("### Project Workspace Tree Map")
            st.markdown("Visual diagram of the project folder structures. Status labels: [Exists] = exists, [Missing] = not yet created, [Not Set] = not set.")
            
            st.markdown(generate_interactive_tree_html(), unsafe_allow_html=True)

            # Display Discovered Sample list
            with st.expander("List Discovered Samples and Matching Pairs", expanded=True):
                pairs = active_project.get("discovered_pairs", [])
                if pairs:
                    st.table(pairs)
                else:
                    st.info("No sample pairs discovered. Ensure CODEX raw images are placed in the raw directory.")

            col_act1, col_act2 = st.columns(2)
            with col_act1:
                # Use callback for saving project config to prevent UI thread warnings
                st.button("Save Project", type="primary", width="stretch", on_click=save_project_callback)
            with col_act2:
                # Use callback for unloading project context
                st.button("Unload Project Context", type="secondary", width="stretch", on_click=unload_project_callback)

        with proj_tabs[1]:
            st.markdown("### Project-Wide Constants and Metadata")
            st.markdown("Configure global parameters used across different workflow steps. *Remember to click 'Save Project' on the main tab to write these permanently.*")
            
            col_meta, col_hardware = st.columns(2)
            
            with col_meta:
                st.subheader("Project Metadata")
                # Values come from the loaded project through each key; passing
                # value= as well made Streamlit warn that it was set twice.
                for meta_key, meta_default in (
                    ("proj_pi", ""), ("proj_institution", ""), ("proj_description", ""),
                    ("proj_mpp_codex", 0.377), ("proj_mpp_he", 0.220), ("proj_use_gpu", True),
                    ("proj_gpu_id", 0), ("proj_num_workers", 4),
                ):
                    if st.session_state.get(meta_key) is None:
                        st.session_state[meta_key] = meta_default
                for float_key in ("proj_mpp_codex", "proj_mpp_he"):
                    st.session_state[float_key] = float(st.session_state[float_key])
                for int_key, lowest in (("proj_gpu_id", 0), ("proj_num_workers", 1)):
                    st.session_state[int_key] = max(int(st.session_state[int_key]), lowest)
                st.text_input("Principal Investigator (PI)", key="proj_pi")
                st.text_input("Institution / Lab", key="proj_institution")
                st.text_area("Project Description / Notes", key="proj_description")

            with col_hardware:
                st.subheader("Computational & Image Parameters")
                st.number_input("CODEX Pixel Size (Microns per pixel - MPP)", format="%.4f", key="proj_mpp_codex")
                st.number_input("H&E Pixel Size (MPP)", format="%.4f", key="proj_mpp_he")
                st.checkbox("Enable GPU Acceleration (Cellpose/Deepcell)", key="proj_use_gpu")
                st.number_input("CUDA Device ID", min_value=0, key="proj_gpu_id")
                st.number_input("Parallel Workers (CPU Multiprocessing)", min_value=1, key="proj_num_workers")

        with proj_tabs[2]:
            st.markdown("### Customize Pipeline Directories")
            st.markdown(
                "Fine-tune directory mappings for specific pipeline stages. Edits made here "
                "sync to the corresponding tabs and are stored in the active project JSON."
            )

            # Group directory settings into logical sections
            
            # 1. Image Registration
            with st.expander("Step 1: Image Registration Mappings", expanded=False):
                st.markdown("Paths used to align CODEX tiles and match H&E histology stains.")
                directory_input_with_browse(
                    label="CODEX Raw Input Directory",
                    text_key="proj_dir_batch_codex_dir",
                    default_value=st.session_state.get("batch_codex_dir", ""),
                    browser_key_prefix="tree_codex_dir",
                )
                directory_input_with_browse(
                    label="H&E Raw Input Directory",
                    text_key="proj_dir_batch_he_dir",
                    default_value=st.session_state.get("batch_he_dir", ""),
                    browser_key_prefix="tree_he_dir",
                )
                reg_out = directory_input_with_browse(
                    label="Registered Images Output Directory",
                    text_key="proj_dir_tab1_batch_output_dir",
                    default_value=st.session_state.get("tab1_batch_output_dir", ""),
                    browser_key_prefix="tree_reg_out",
                )
                st.session_state.proj_dir_registration_output_dir = reg_out

            # 2. Background Subtraction
            with st.expander("Step 2: Background Subtraction Mappings", expanded=False):
                directory_input_with_browse(
                    label="Registered Input Directory",
                    text_key="proj_dir_tab2_input_dir_reg",
                    default_value=st.session_state.get("tab2_input_dir_reg", ""),
                    browser_key_prefix="tree_bg_reg",
                )
                directory_input_with_browse(
                    label="Raw CODEX Input Directory",
                    text_key="proj_dir_tab2_input_dir_raw",
                    default_value=st.session_state.get("tab2_input_dir_raw", ""),
                    browser_key_prefix="tree_bg_raw",
                )
                bg_out = directory_input_with_browse(
                    label="Background Subtracted Output Directory",
                    text_key="proj_dir_tab2_output_dir",
                    default_value=st.session_state.get("tab2_output_dir", ""),
                    browser_key_prefix="tree_bg_out",
                )
                st.session_state.proj_dir_bg_subtraction_output_dir = bg_out
                directory_input_with_browse(
                    label="Background Threshold Tables Directory (shared with RESOLVE)",
                    text_key="proj_dir_threshold_dir",
                    default_value=st.session_state.get(
                        "threshold_dir",
                        str(project_root_default / "data" / "thresholds"),
                    ),
                    browser_key_prefix="tree_threshold_dir",
                    help_text=(
                        "Tab 2 writes threshold CSV files here and RESOLVE reads them "
                        "from the same project-level directory. The default is "
                        "<project>/data/thresholds."
                    ),
                )

            # 3. ROI Extraction
            with st.expander("Step 3: ROI Extraction Mappings", expanded=False):
                directory_input_with_browse(
                    label="Input Directory (Background Subtracted / Registered Images)",
                    text_key="proj_dir_tab3_input_dir",
                    default_value=st.session_state.get("tab3_input_dir", ""),
                    browser_key_prefix="tree_roi_in",
                )
                directory_input_with_browse(
                    label="GeoJSON ROI Boundaries Directory",
                    text_key="proj_dir_tab3_geojson_dir",
                    default_value=st.session_state.get("tab3_geojson_dir", ""),
                    browser_key_prefix="tree_roi_geojson",
                )
                roi_out = directory_input_with_browse(
                    label="Cropped ROI Images Output Directory",
                    text_key="proj_dir_tab3_output_dir",
                    default_value=st.session_state.get("tab3_output_dir", ""),
                    browser_key_prefix="tree_roi_out",
                )
                st.session_state.proj_dir_roi_output_dir = roi_out

            # 4. Segmentation
            with st.expander("Step 4: Segmentation Mappings", expanded=False):
                directory_input_with_browse(
                    label="ROI Images Input Directory",
                    text_key="proj_dir_tab4_input_dir",
                    default_value=st.session_state.get("tab4_input_dir", ""),
                    browser_key_prefix="tree_seg_in",
                )
                directory_input_with_browse(
                    label="GeoJSON Boundaries Directory",
                    text_key="proj_dir_tab4_seg_mask_save_dir",
                    default_value=st.session_state.get("tab4_seg_mask_save_dir", ""),
                    browser_key_prefix="tree_seg_boundaries",
                )
                seg_out = directory_input_with_browse(
                    label="Segmentation Features & Masks Output Directory",
                    text_key="proj_dir_tab4_output_dir",
                    default_value=st.session_state.get("tab4_output_dir", ""),
                    browser_key_prefix="tree_seg_out",
                )
                directory_input_with_browse(
                    label="Extracted Feature CSV Directory",
                    text_key="proj_dir_tab4_csv_dir",
                    default_value=st.session_state.get("tab4_csv_dir", ""),
                    browser_key_prefix="tree_seg_csv",
                )
                directory_input_with_browse(
                    label="Segmentation GeoJSON Directory",
                    text_key="proj_dir_tab4_geojson_dir",
                    default_value=st.session_state.get("tab4_geojson_dir", ""),
                    browser_key_prefix="tree_seg_geojson",
                )
                st.session_state.proj_dir_batch_input_dir = st.session_state.get("tab4_input_dir", "")
                st.session_state.proj_dir_batch_geojson_dir = st.session_state.get("tab4_seg_mask_save_dir", "")
                st.session_state.proj_dir_batch_output_dir = seg_out

            # 5. Annotation
            with st.expander("Step 5: Tissue Annotation Mappings", expanded=False):
                directory_input_with_browse(
                    label="Raw Annotation GeoJSON Directory",
                    text_key="proj_dir_annot_input_dirs",
                    default_value=st.session_state.get("annot_input_dirs", [""])[0] if isinstance(st.session_state.get("annot_input_dirs"), list) and st.session_state.get("annot_input_dirs") else "",
                    browser_key_prefix="tree_annot_in",
                )
                directory_input_with_browse(
                    label="Stage 1A Merged Tissue Output Directory",
                    text_key="proj_dir_annot_output_dir",
                    default_value=st.session_state.get("annot_output_dir", ""),
                    browser_key_prefix="tree_annot_out",
                )
                st.session_state.proj_dir_zone_input_dir = st.session_state.get("annot_output_dir", "")
                
                zone_out = directory_input_with_browse(
                    label="Compartments & Microenvironment Zones Output",
                    text_key="proj_dir_zone_output_dir",
                    default_value=st.session_state.get("zone_output_dir", ""),
                    browser_key_prefix="tree_zone_out",
                )
                st.session_state.proj_dir_cell_zone_dir = zone_out
                
                directory_input_with_browse(
                    label="Segmentation Masks Input Directory (Cell Annotation)",
                    text_key="proj_dir_cell_seg_dir",
                    default_value=st.session_state.get("cell_seg_dir", ""),
                    browser_key_prefix="tree_cell_seg",
                )
                directory_input_with_browse(
                    label="Final Zone-Annotated Cell CSV Output Directory",
                    text_key="proj_dir_cell_out_dir",
                    default_value=st.session_state.get("cell_out_dir", ""),
                    browser_key_prefix="tree_cell_out",
                )
                st.markdown("**Visualization Directories**")
                directory_input_with_browse(
                    label="Annotated CSV Directory for Visualization",
                    text_key="proj_dir_vis_csv_dir",
                    default_value=st.session_state.get("vis_csv_dir", ""),
                    browser_key_prefix="tree_vis_csv",
                )
                directory_input_with_browse(
                    label="Image Directory for Visualization",
                    text_key="proj_dir_vis_img_dir",
                    default_value=st.session_state.get("vis_img_dir", ""),
                    browser_key_prefix="tree_vis_img",
                )

            # 6. Batch Correction
            with st.expander("Step 6: Batch Correction Mappings", expanded=False):
                directory_input_with_browse(
                    label="Annotated CSVs Input Directory",
                    text_key="proj_dir_bc_input_dir",
                    default_value=st.session_state.get("bc_input_dir", ""),
                    browser_key_prefix="tree_bc_in",
                )
                directory_input_with_browse(
                    label="Batch Corrected CSVs Output Directory",
                    text_key="proj_dir_bc_output_dir",
                    default_value=st.session_state.get("bc_output_dir", ""),
                    browser_key_prefix="tree_bc_out",
                )

            # 7 & 8. Phenotyping & Spatial Analysis
            with st.expander("Phenotyping and Spatial Analysis Mappings", expanded=False):
                unified_csv_dir = directory_input_with_browse(
                    label="Unified CSV Directory (Shared across Phenotyping, Spatial, Visualization)",
                    text_key="proj_dir_unified_csv_dir",
                    default_value=st.session_state.get("global_pheno_dir", ""),
                    browser_key_prefix="tree_unified_csv_dir",
                )
                if unified_csv_dir:
                    st.session_state["global_pheno_dir"] = unified_csv_dir
                    st.session_state["global_spatial_dir"] = unified_csv_dir
                    st.session_state["global_visualization_dir"] = unified_csv_dir
                
                # Shared Unified CSV File Name
                unified_csv = st.text_input(
                    "Unified CSV Filename (Shared across Phenotyping, Spatial, Visualization)", 
                    value=st.session_state.get("proj_unified_csv_filename", ""), 
                    key="proj_unified_csv_filename",
                    help="Filename of the unified CSV, e.g., 'INNATE.csv'. This will be automatically selected in Tabs 7, 8, and 9."
                )
                if unified_csv:
                    st.session_state["global_pheno_file"] = unified_csv
                    st.session_state["global_spatial_file"] = unified_csv
                    st.session_state["global_visualization_file"] = unified_csv
                
                directory_input_with_browse(
                    label="Phenotyping Clusters Output Directory",
                    text_key="proj_dir_phenotyping_output_dir",
                    default_value=st.session_state.get("phenotyping_output_dir", ""),
                    browser_key_prefix="tree_pheno_out",
                )
                directory_input_with_browse(
                    label="Spatial Relationships Output Directory",
                    text_key="proj_dir_spatial_output_dir",
                    default_value=st.session_state.get("spatial_output_dir", ""),
                    browser_key_prefix="tree_spatial_out",
                )
                directory_input_with_browse(
                    label="Visual Validation ROI Directory",
                    text_key="proj_dir_viz_roi_dir",
                    default_value=st.session_state.get("viz_roi_dir", ""),
                    browser_key_prefix="tree_viz_roi",
                )
                directory_input_with_browse(
                    label="Visual Validation Segmentation Directory",
                    text_key="proj_dir_viz_seg_dir",
                    default_value=st.session_state.get("viz_seg_dir", ""),
                    browser_key_prefix="tree_viz_seg",
                )

            # 9. Visualization Mappings
            with st.expander("Step 9: Visualization Mappings (Tab 9)", expanded=False):
                directory_input_with_browse(
                    label="Actual Tissue Images Input",
                    text_key="proj_dir_tab9_actual_image_dir",
                    default_value=st.session_state.get("tab9_actual_image_dir", ""),
                    browser_key_prefix="tree_v1",
                )
                directory_input_with_browse(
                    label="Actual Tissue GeoJSON ROI Input",
                    text_key="proj_dir_tab9_actual_roi_geojson_dir",
                    default_value=st.session_state.get("tab9_actual_roi_geojson_dir", ""),
                    browser_key_prefix="tree_v2",
                )
                directory_input_with_browse(
                    label="Actual Tissue Overlay GeoJSON Input",
                    text_key="proj_dir_tab9_actual_overlay_geojson_dir",
                    default_value=st.session_state.get("tab9_actual_overlay_geojson_dir", ""),
                    browser_key_prefix="tree_v3",
                )
                directory_input_with_browse(
                    label="Actual Tissue Visual Output",
                    text_key="proj_dir_tab9_actual_output_dir",
                    default_value=st.session_state.get("tab9_actual_output_dir", ""),
                    browser_key_prefix="tree_v4",
                )
                directory_input_with_browse(
                    label="Single Marker Image Input",
                    text_key="proj_dir_tab9_marker_single_image_dir",
                    default_value=st.session_state.get("tab9_marker_single_image_dir", ""),
                    browser_key_prefix="tree_v5",
                )
                directory_input_with_browse(
                    label="Single Marker GeoJSON Input",
                    text_key="proj_dir_tab9_marker_single_geojson_dir",
                    default_value=st.session_state.get("tab9_marker_single_geojson_dir", ""),
                    browser_key_prefix="tree_v6",
                )
                directory_input_with_browse(
                    label="Single Marker Visual Output",
                    text_key="proj_dir_tab9_marker_single_output_dir",
                    default_value=st.session_state.get("tab9_marker_single_output_dir", ""),
                    browser_key_prefix="tree_v7",
                )
                directory_input_with_browse(
                    label="Batch Marker Image Input",
                    text_key="proj_dir_tab9_marker_batch_image_dir",
                    default_value=st.session_state.get("tab9_marker_batch_image_dir", ""),
                    browser_key_prefix="tree_v8",
                )
                directory_input_with_browse(
                    label="Batch Marker GeoJSON Input",
                    text_key="proj_dir_tab9_marker_batch_geojson_dir",
                    default_value=st.session_state.get("tab9_marker_batch_geojson_dir", ""),
                    browser_key_prefix="tree_v9",
                )
                directory_input_with_browse(
                    label="Batch Marker Visual Output",
                    text_key="proj_dir_tab9_marker_batch_output_dir",
                    default_value=st.session_state.get("tab9_marker_batch_output_dir", ""),
                    browser_key_prefix="tree_v10",
                )
                
            st.info("Adjusting paths above immediately shifts session targets in other tabs. Save in Tab 0 Map to persist.")

        with proj_tabs[3]:
            st.markdown("### Configuration Audit Log")
            st.markdown("Historical record of configuration edits and state changes made to this project.")
            
            audit_log = active_project.get("audit_log", [])
            if audit_log:
                import pandas as pd
                df = pd.DataFrame(audit_log)
                if not df.empty:
                    df = df[["timestamp", "tab_affected", "parameter_changed", "old_value", "new_value"]]
                    st.dataframe(df, width="stretch")
                else:
                    st.info("No configuration tweaks recorded yet.")
            else:
                st.info("No configuration tweaks recorded yet.")

        with proj_tabs[4]:
            st.markdown("### Cache Management")
            st.markdown(
                "Everything the app writes outside your analysis outputs, grouped by what "
                "you actually lose by deleting it. Project data, the project config and the "
                "unified CSV are never touched, and only the **active** project is inspected."
            )

            from spacemap_gui.cache_manager import (
                CATEGORY_LABELS, clear_entry, format_size, scan_caches,
            )

            entries = scan_caches(st.session_state.get("active_project_root"))
            by_category: dict[str, list] = {}
            for entry in entries:
                by_category.setdefault(entry.category, []).append(entry)

            safe = [e for e in entries if e.category in ("recomputed", "results")]
            total_safe = sum(e.size_bytes for e in safe)

            head1, head2 = st.columns([1, 2])
            with head1:
                st.metric("Safe to reclaim", format_size(total_safe))
            with head2:
                if st.button(
                    "Clear All Safe Caches",
                    type="primary",
                    width="stretch",
                    disabled=total_safe == 0,
                    help="Clears everything below except the tab parameter files.",
                    key="btn_clear_all_safe",
                ):
                    freed = files = 0
                    for entry in safe:
                        count, size = clear_entry(entry)
                        files += count
                        freed += size
                    st.cache_data.clear()
                    st.success(f"Removed {files} file(s), freeing {format_size(freed)}.")
                    st.rerun()

            for category in ("recomputed", "results", "settings"):
                group = by_category.get(category, [])
                if not group:
                    continue
                st.markdown(f"#### {CATEGORY_LABELS[category]}")
                if category == "settings":
                    st.warning(
                        "These are not caches. Deleting them resets each tab to its defaults; "
                        "your data and results are untouched.",
                        icon=":material/warning:",
                    )
                for entry in group:
                    size_col, info_col, action_col = st.columns([1, 4, 1])
                    with size_col:
                        st.markdown(f"**{format_size(entry.size_bytes)}**")
                        st.caption(f"{entry.file_count} file(s)")
                    with info_col:
                        st.markdown(f"**{entry.label}**")
                        st.caption(entry.description)
                        st.caption("`" + "`, `".join(str(p) for p in entry.paths) + "`")
                    with action_col:
                        if st.button(
                            "Clear", key=f"btn_clear_{entry.key}",
                            disabled=entry.file_count == 0, width="stretch",
                        ):
                            count, size = clear_entry(entry)
                            st.cache_data.clear()
                            st.success(f"Removed {count} file(s), freeing {format_size(size)}.")
                            st.rerun()

            st.divider()
            st.markdown("#### In-memory caches")
            st.caption(
                "Streamlit keeps loaded tables in RAM for the session. Clearing forces the "
                "next read to come from disk - useful after editing a CSV outside the app."
            )
            if st.button("Clear In-Memory Caches", key="btn_clear_st_cache"):
                st.cache_data.clear()
                st.success("In-memory caches cleared.")

        with proj_tabs[5]:
            from spacemap_gui.persistence import (
                create_restore_point, list_restore_points, restore_point_dir,
            )
            from spacemap_gui.project_bundle import BUNDLE_KEY, external_paths

            active_root = st.session_state.get("active_project_root", "")

            st.markdown("### Sharing this project")
            st.markdown(
                "The project JSON carries every tab's settings, including the phenotyping "
                "parameter files and the Tab 8 / Tab 9 per-CSV configurations. Paths inside "
                "the project are stored as `${PROJECT_ROOT}/...`, so a collaborator can drop "
                "the file into their own workspace and load it as-is."
            )

            bundle = active_project.get(BUNDLE_KEY) or {}
            if bundle:
                params = bundle.get("tab_params", {})
                iface = bundle.get("interface_state", {})
                folded = sum(len(v) for v in params.values() if isinstance(v, dict))
                m1, m2, m3 = st.columns(3)
                m1.metric("Tab parameter files", len(params))
                m2.metric("Folded-in settings", folded)
                m3.metric("Tab 8/9 config sets", sum(len(v) for v in iface.values() if isinstance(v, dict)))
            else:
                st.info("Save the project once to fold the outlying settings into its JSON.")

            outside = external_paths(active_project.get("directory_mappings", {}), active_root)
            if outside:
                st.markdown("#### Paths a collaborator must remap")
                st.caption(
                    "These point outside the project folder, so they cannot travel with it. "
                    "Anything here that looks like *another project* is worth correcting now."
                )
                for path in outside:
                    inside_other_project = "/project/" in path
                    (st.error if inside_other_project else st.warning)(f"`{path}`", icon=":material/warning:")
            else:
                st.success("Every configured directory lives inside this project - fully portable.")

            st.divider()
            st.markdown("### Restore points")
            st.caption(
                "One snapshot is taken automatically on the first save after loading a project - "
                f"the state before this session's edits. The newest {5} are kept."
            )

            points = list_restore_points(active_root) if active_root else []
            rp1, rp2 = st.columns([1, 3])
            with rp1:
                if st.button("Create Restore Point", width="stretch", key="btn_make_restore_point"):
                    made = create_restore_point(active_root, label="manual")
                    if made:
                        st.success(f"Created {made.name}")
                        st.rerun()
                    else:
                        st.error("Could not create a restore point; the config may be invalid.")
            with rp2:
                if points:
                    st.caption(f"Stored in `{restore_point_dir(active_root)}`")
                else:
                    st.caption("No restore points yet.")

            if points:
                import pandas as pd

                rows = []
                for point in reversed(points):
                    stat = point.stat()
                    rows.append({
                        "Snapshot": point.name,
                        "Taken": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "Size": f"{stat.st_size / 1e6:.2f} MB",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(
                    "To roll back, copy a snapshot over the project JSON with the app closed, "
                    "then reload the project."
                )

        st.markdown("---")
    else:
        st.info("No project is currently loaded. Please load an existing project or configure a new one below.")
        
        st.subheader("Load Existing Project")
        if projects:
            project_names = list(projects.keys())
            col_sel, col_btn = st.columns([4, 1])
            with col_sel:
                selected_proj_name = st.selectbox("Select Project to Load", project_names, key="tab0_load_proj_select", label_visibility="collapsed")
            with col_btn:
                if st.button("Load Project", type="primary", width="stretch", key="tab0_load_proj_btn"):
                    load_project_callback(projects[selected_proj_name])
                    st.rerun()
        else:
            st.write("No registered projects found. You can browse and load a project directory below.")
            
        # Add "Browse to Load Project" option
        st.markdown("**or Browse to Load a Project Directory**")
        directory_input_with_browse(
            label="Browse Project Root Directory",
            text_key="browse_load_proj_dir",
            default_value="",
            browser_key_prefix="browse_load_proj",
            help_text="Select the root folder of an existing project (must contain the <project_name>.json or experiment.json file)."
        )
        if st.session_state.get("browse_load_proj_dir"):
            selected_dir = st.session_state["browse_load_proj_dir"].strip()
            if selected_dir and os.path.exists(selected_dir) and os.path.isdir(selected_dir):
                proj_name = Path(selected_dir).name
                json_path = Path(selected_dir) / f"{proj_name}.json"
                fallback_path = Path(selected_dir) / "experiment.json"
                if json_path.exists() or fallback_path.exists():
                    if st.button("Load Selected Directory", type="secondary", key="btn_load_browse_proj"):
                        load_project_callback(selected_dir)
                        st.session_state["browse_load_proj_dir"] = ""
                        st.rerun()
                else:
                    st.warning(f"Selected directory is not a valid project folder (no '{proj_name}.json' or 'experiment.json' found).")
            
        st.markdown("---")

    st.subheader("Configure New Project")
    
    # Show validation error or success messages if present
    if st.session_state.get("new_proj_error"):
        st.error(st.session_state["new_proj_error"])
    if st.session_state.get("new_proj_success"):
        st.success(st.session_state["new_proj_success"])
        # Clear success message so it only shows once
        del st.session_state["new_proj_success"]

    st.text_input("Project Name", value="RCC_cohort_study", help="e.g. RCC_cohort_study", key="new_proj_name")
    
    directory_input_with_browse(
        label="Parent Experiments Directory",
        text_key="proj_parent_dir",
        default_value=str(new_project_parent_default),
        browser_key_prefix="proj_parent_dir",
        help_text="Directory where the new project folder will be created."
    )

    # Trigger project creation in on_click callback
    st.button("Create and Initialize Project", type="primary", width="stretch", on_click=create_project_callback)

    st.markdown("---")

    # The optional AI assistant is configured here so that a user never has to
    # edit a file or set an environment variable to switch it on or off.  The
    # import is local to keep Tab 0 loadable when the assistant components are
    # missing or broken.
    try:
        from spacemap_gui.components.assistant_setup import render_assistant_setup

        render_assistant_setup()
    except Exception as exc:  # noqa: BLE001 - the project tab must always render
        st.subheader("AI Assistant (optional)")
        st.warning(f"The AI assistant settings panel could not be loaded: {exc}")


if __name__ == "__main__":
    render()
