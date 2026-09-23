# Force Streamlit module reload touch
import os
import errno
import sys
import importlib
import logging

# Streamlit can inherit an editable install from a different SPACEMAP
# checkout.  Put this app's source tree first so all spacemap_gui modules are
# imported from the same checkout as app.py.
_APP_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_SOURCE_ROOT in sys.path:
    sys.path.remove(_APP_SOURCE_ROOT)
sys.path.insert(0, _APP_SOURCE_ROOT)

_log = logging.getLogger("spacemap_gui")
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.setLevel(logging.INFO)

# Force reload of custom modules to ensure the running process picks up our bug fixes
for mod in [
    "spacemap_gui.persistence",
    "spacemap_gui.cell_visualizer",
    "spacemap_gui.ui_utils",
    "spacemap_gui.workflow_guidance",
    "spacemap_gui.process_log",
]:
    if mod in sys.modules:
        try:
            importlib.reload(sys.modules[mod])
        except Exception as e:
            _log.warning("Failed to reload %s: %s", mod, e)

# Fix for protobuf error: "Descriptors cannot be created directly"
# This must be set before importing streamlit or tensorflow
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Logging controls
# - SPACEMAP_TAB_LOGS: tab/pipeline progress logs (default ON)
# - SPACEMAP_APP_LOGS: app render/debug logs (default OFF)
# Backward compatible: if SPACEMAP_CONSOLE_LOGS is set, it overrides both.
os.environ.setdefault("SPACEMAP_TAB_LOGS", "1")
os.environ.setdefault("SPACEMAP_APP_LOGS", "1")
os.environ.setdefault("SPACEMAP_PERF_LOGS", "1")

# Active tab tracking for tab-specific logging
# We use session_state to track WHICH tab the user is actually viewing
# This persists across reruns and isn't overwritten by sequential tab rendering
_ACTIVE_TAB_KEY = "_spacemap_active_tab"
_USER_TAB_KEY = "_spacemap_user_tab"  # The tab user is actually in
_DRAW_MODE_KEY = "_spacemap_draw_mode"  # Skip heavy processing when drawing

from spacemap_gui import terminal


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('app', message, step=None, level=level)


def set_user_tab(tab_name: str):
    """Set the tab the user is actually viewing (from widget interactions)."""
    os.environ[_USER_TAB_KEY] = tab_name

def get_user_tab() -> str:
    """Get which tab the user is actually viewing."""
    return os.environ.get(_USER_TAB_KEY, "tab1")

def set_active_tab(tab_name: str):
    """Set the currently rendering tab (internal use)."""
    os.environ[_ACTIVE_TAB_KEY] = tab_name

def get_active_tab() -> str:
    """Get the currently rendering tab."""
    return os.environ.get(_ACTIVE_TAB_KEY, "tab1")

def is_user_in_tab(tab_name: str) -> bool:
    """Check if the user is actually viewing this tab."""
    return get_user_tab() == tab_name

def set_draw_mode(active: bool):
    """Set whether user is in draw mode (skip heavy processing)."""
    os.environ[_DRAW_MODE_KEY] = "1" if active else "0"

def is_draw_mode() -> bool:
    """Check if draw mode is active."""
    return os.environ.get(_DRAW_MODE_KEY, "0") == "1"

# Fix for CUDA Out of Memory error
# Enable expandable segments to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import logging
import json
import toml
import streamlit as st
from pathlib import PurePath, Path
from datetime import datetime

# Suppress Streamlit value assignment warnings
logging.getLogger("streamlit.elements.lib.policies").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.*")

def _has_paths(val):
    if isinstance(val, PurePath):
        return True
    if isinstance(val, dict):
        return any(_has_paths(v) for v in val.values())
    if isinstance(val, (list, tuple, set)):
        return any(_has_paths(x) for x in val)
    return False

def _clean_state_paths(val):
    if isinstance(val, PurePath):
        return str(val)
    if isinstance(val, dict):
        return {k: _clean_state_paths(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clean_state_paths(x) for x in val]
    if isinstance(val, tuple):
        return tuple(_clean_state_paths(x) for x in val)
    return val

for k in list(st.session_state.keys()):
    try:
        val = st.session_state[k]
        if _has_paths(val):
            st.session_state[k] = _clean_state_paths(val)
    except Exception as e:
        _log.warning("Failed to clean PurePath in session_state[%s]: %s", k, e)

if os.getenv("SPACEMAP_DEBUG_FILE") == "1":
    try:
        debug_file_path = Path(__file__).resolve().parent / "session_state_debug.txt"
        with open(debug_file_path, "a") as f:
            f.write(f"\n=== RUN AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} ===\n")
            f.write(f"active_project={st.session_state.get('active_project') is not None}\n")
            for k in list(st.session_state.keys()):
                if "dir" in k or "path" in k or "root" in k:
                    f.write(f"  {k} = {repr(st.session_state[k])}\n")
    except Exception as e:
        _log.warning("Failed to write session_state debug file: %s", e)

# Define paths FIRST before any code uses them
current_dir = Path(os.path.abspath(__file__)).parent
project_root = current_dir.parent.parent

_PAGE_CONFIG = {
    "page_title": "SPACEMAP AI Pipeline",
    "page_icon": ":material/biotech:",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}


def _is_stale_filesystem_error(exc: BaseException) -> bool:
    """Return True when the exception is a stale network filesystem handle."""
    return isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ESTALE


def _show_stale_filesystem_message(exc: OSError, phase: str) -> None:
    """Show a concise recovery hint instead of a full startup traceback."""
    _log_console(f"Stale filesystem handle during {phase}: {exc}")
    st.title("SPACEMAP Pipeline Interface")
    st.error("The project filesystem is temporarily unavailable.")
    st.markdown(
        "SPACEMAP hit a stale file handle while starting or loading a tab. "
        "This usually means the mounted workspace path needs a retry or remount."
    )
    st.info(
        "Retry the app first. If it keeps happening, the network share holding "
        "the project may need to be remounted."
    )
    st.code(f"Phase: {phase}\n{exc}")
    st.stop()


def _ensure_local_cache_dir(path: Path, env_var: str) -> None:
    """Create a local cache directory and surface stale mount issues cleanly."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if _is_stale_filesystem_error(exc):
            _show_stale_filesystem_message(exc, f"creating {path}")
        raise
    os.environ[env_var] = str(path)


st.set_page_config(**_PAGE_CONFIG)

# The AI assistant is optional.  When it is switched off -- which is the
# default -- its page is never registered, its modules are never imported, and
# SPACEMAP behaves as a pipeline-only application.  Tab 0 turns it on.
from spacemap.ai_settings import is_enabled as _assistant_enabled

ASSISTANT_ENABLED = _assistant_enabled()

# Define Pages
assistant_page = (
    st.Page(
        str(current_dir / "tabs/tab_ai_assistant.py"),
        title="AI Assistant",
        url_path="assistant",
    )
    if ASSISTANT_ENABLED
    else None
)
pages = {
    "Pipeline": [
        st.Page(str(current_dir / "tabs/tab0_project.py"), title="0. Project Workspace", url_path="project"),
        st.Page(str(current_dir / "tabs/tab1_registration.py"), title="1. Registration", url_path="registration"),
        st.Page(str(current_dir / "tabs/tab2_background_subtraction.py"), title="2. Background Subtraction", url_path="bg_subtraction"),
        st.Page(str(current_dir / "tabs/tab3_roi_extraction.py"), title="3. ROI Extraction", url_path="roi"),
        st.Page(str(current_dir / "tabs/tab4_annotation/__init__.py"), title="4. Annotation", url_path="annotation"),
        st.Page(str(current_dir / "tabs/tab5_segmentation.py"), title="5. Segmentation", url_path="segmentation"),
        st.Page(str(current_dir / "tabs/tab6_batch_correction.py"), title="6. Batch Correction", url_path="batch_correction"),
        st.Page(str(current_dir / "tabs/tab7_phenotyping/__init__.py"), title="7. Phenotyping", url_path="phenotyping"),
        st.Page(str(current_dir / "tabs/tab8_analysis/__init__.py"), title="8. Spatial Analysis", url_path="analysis"),
        st.Page(str(current_dir / "tabs/tab9_visualization/__init__.py"), title="9. Visualization", url_path="visualization"),
    ],
}
if assistant_page is not None:
    pages["Assistant"] = [assistant_page]

pg = st.navigation(pages, position="hidden")

# Regenerable caches live with the rest of the user-writable state so that an
# installed copy never writes into its own package directory.
from spacemap.userpaths import cache_dir as _spacemap_cache_dir

user_cache_root = _spacemap_cache_dir()
numba_cache_dir = user_cache_root / "numba_cache"
cellpose_cache_dir = user_cache_root / "cellpose_cache"
_ensure_local_cache_dir(numba_cache_dir, "NUMBA_CACHE_DIR")
_ensure_local_cache_dir(cellpose_cache_dir, "CELLPOSE_LOCAL_MODELS_PATH")

def log_app(msg: str):
    """App start-up messages (shown when SPACEMAP_APP_LOGS=1)."""
    terminal.say("app", msg, channel="app")


def log_perf(msg: str):
    """Page timing messages (hidden with SPACEMAP_PERF_LOGS=0)."""
    terminal.say("app", msg, step="Timing", channel="perf")


def _timed_call(label: str, fn):
    """Run a callable and log elapsed time in ms."""
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log_perf(f"{label}: {elapsed_ms:.1f} ms")
    return result

# ============================================================================
# PERSISTENT STATE MANAGEMENT
# ============================================================================

PERSISTENT_STATE_FILE = current_dir / "interface_state.json"

# Keys that represent directory selections (or directory lists) across the app.
# Imported from tab0_project.DIRECTORY_KEYS as the single source of truth so
# that app.py auto-save and persistence.py project-config sync stay in sync.
# tab0_project is lightweight (no heavy scientific deps) so this import is safe
# at app startup.
from spacemap_gui.tabs.tab0_project import DIRECTORY_KEYS as PERSISTENT_DIRECTORY_KEYS


def _collect_directory_state() -> dict:
    """Collect only directory selection values from session state."""
    state: dict[str, object] = {}
    for k in PERSISTENT_DIRECTORY_KEYS:
        state[k] = st.session_state.get(k, None)
    return state


def _update_persistent_state_file(updates: dict) -> None:
    """Update interface_state.json with partial keys, preserving other saved fields.

    Delegates to persistence.update_interface_state() for atomic write safety.
    """
    from spacemap_gui.persistence import update_interface_state
    update_interface_state(updates)


def auto_save_directories_if_changed(min_interval_s: float = 0.5) -> None:
    """Auto-save directory selections when they change.

    This avoids rewriting the full app state on every rerun, but still ensures
    directory changes are persisted immediately (no button required).
    """
    now = time.time()
    last_t = float(st.session_state.get('_spacemap_auto_save_dirs_last_t', 0.0))
    if (now - last_t) < float(min_interval_s):
        return

    current = _collect_directory_state()
    previous = st.session_state.get('_spacemap_auto_saved_dirs_snapshot', None)
    if previous is None:
        # Initialize snapshot from disk (if present) to avoid a write on first render.
        st.session_state['_spacemap_auto_saved_dirs_snapshot'] = current
        st.session_state['_spacemap_auto_save_dirs_last_t'] = now
        return

    if current != previous:
        _update_persistent_state_file(current)
        st.session_state['_spacemap_auto_saved_dirs_snapshot'] = current
        st.session_state['_spacemap_auto_save_dirs_last_t'] = now

def load_persistent_state():
    """Load directory selections from previous session."""
    from spacemap_gui.persistence import load_interface_state
    return load_interface_state()

def save_persistent_state():
    """Save directory selections for next session."""
    # Start with all directory keys dynamically
    state_to_save = _collect_directory_state()

    # Add complex objects that aren't in the directory list
    state_to_save['tab4_custom_model'] = st.session_state.get('tab4_custom_model', None)

    # Add file-specific marker selections
    cell_cfg_records = st.session_state.get('cell_annot_config_dict', {})
    for key in st.session_state.keys():
        if isinstance(key, str) and (key.startswith('seg_nuclear_') or key.startswith('seg_membrane_')):
            state_to_save[key] = st.session_state.get(key, {})
    
    state_to_save.update({
        # Tab 4: Annotation & Cell Annotation items not in directory list
        'cell_pair_flags': st.session_state.get('cell_pair_flags', {}),
        'cell_annot_config_dict': cell_cfg_records,
        'cell_annotations_found': list(st.session_state.get('cell_annotations_found', [])),
        
        # Tab 6: Batch Correction items
        'bc_config': st.session_state.get('bc_config', {}),
        'bc_vis_file': st.session_state.get('bc_vis_file', ''),
        'bc_files_df': st.session_state.bc_files_df.to_dict('records') if 'bc_files_df' in st.session_state and hasattr(st.session_state.bc_files_df, 'to_dict') else [],
        'bc_metadata_cols': st.session_state.get('bc_metadata_cols', []),
    })

    try:
        _update_persistent_state_file(state_to_save)
        log_app("Persistent state saved")
    except Exception as e:
        log_app(f"Error saving persistent state: {e}")

# Load persistent state on app start
persistent_state = load_persistent_state()

# NOTE: Tab modules are NOT imported here. With st.navigation() + st.Page(),
# Streamlit executes each page file directly as a script when the user navigates
# to it, so eager imports would only slow down startup (tabs 1-4 pull in tifffile,
# cv2, torch, shapely, etc.) without being used. Heavy tabs (5-9) are also
# imported lazily inside their own render() functions via click-to-load gating.

# Determine the active tab immediately so that state synchronization (which runs next)
# targets the correct tab's directories without delay.
pg_title = pg.title if hasattr(pg, "title") else ""
early_active_tab = "tab0"
if "Project Setup" in pg_title or "Tab 0" in pg_title:
    early_active_tab = "tab0"
elif "1. Registration" in pg_title:
    early_active_tab = "tab1"
elif "2. Background" in pg_title:
    early_active_tab = "tab2"
elif "3. ROI" in pg_title:
    early_active_tab = "tab3"
elif "4. Annotation" in pg_title:
    early_active_tab = "tab4"
elif "5. Segmentation" in pg_title:
    early_active_tab = "tab5"
elif "6. Batch" in pg_title:
    early_active_tab = "tab6"
elif "7. Phenotyping" in pg_title:
    early_active_tab = "tab7"
elif "8. Spatial" in pg_title:
    early_active_tab = "tab8"
elif "9. Visualization" in pg_title:
    early_active_tab = "tab9"
elif "AI Assistant" in pg_title:
    early_active_tab = "assistant"

st.session_state["_active_tab_session"] = early_active_tab
st.session_state["_currently_rendering_tab"] = early_active_tab
if early_active_tab.startswith("tab"):
    st.session_state["_spacemap_last_pipeline_tab"] = early_active_tab

# Initialize Session State
active_proj = st.session_state.get("active_project")
if active_proj:
    from spacemap_gui.persistence import sync_project_state
    sync_project_state()
else:
    # Fallback to persistent state values if present, without inventing paths in repo root
    for key in PERSISTENT_DIRECTORY_KEYS:
        if key not in st.session_state:
            if key in persistent_state and persistent_state[key] not in [None, "", []]:
                st.session_state[key] = persistent_state[key]
                
    if 'bc_config' not in st.session_state:
        from spacemap_gui.persistence import get_default_bc_config
        st.session_state['bc_config'] = persistent_state.get('bc_config', get_default_bc_config())
    if 'bc_vis_file' not in st.session_state:
        st.session_state['bc_vis_file'] = persistent_state.get('bc_vis_file', '')
    if 'bc_files_df' not in st.session_state:
        import pandas as pd
        records = persistent_state.get('bc_files_df', [])
        if records:
            st.session_state['bc_files_df'] = pd.DataFrame.from_records(records)
        else:
            st.session_state['bc_files_df'] = pd.DataFrame(columns=["Process", "Filename"])
    if 'bc_metadata_cols' not in st.session_state:
        st.session_state['bc_metadata_cols'] = persistent_state.get('bc_metadata_cols', [])
    if 'bc_config' not in st.session_state:
        # Use the canonical defaults from persistence.get_default_bc_config() so
        # that app.py and tab6_batch_correction.py stay in sync. Fall back to any
        # previously persisted value if present.
        from spacemap_gui.persistence import get_default_bc_config
        st.session_state['bc_config'] = persistent_state.get('bc_config', get_default_bc_config())
    if 'bc_vis_file' not in st.session_state:
        st.session_state['bc_vis_file'] = persistent_state.get('bc_vis_file', '')
    if 'bc_files_df' not in st.session_state:
        import pandas as pd
        records = persistent_state.get('bc_files_df', [])
        if records:
            st.session_state['bc_files_df'] = pd.DataFrame.from_records(records)
        else:
            st.session_state['bc_files_df'] = pd.DataFrame(columns=["Process", "Filename"])
    if 'bc_metadata_cols' not in st.session_state:
        st.session_state['bc_metadata_cols'] = persistent_state.get('bc_metadata_cols', [])

# Load Config
config_path = current_dir / "interface_config.json"
if config_path.exists():
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            for k, v in config.items():
                if k not in st.session_state:
                    st.session_state[k] = v
    except Exception as e:
        _log_console(f"Error loading config: {e}", level="error")

title_column, assistant_column = st.columns([4, 1], vertical_alignment="center")
with title_column:
    st.title("SPACEMAP AI Pipeline Interface")
with assistant_column:
    if not ASSISTANT_ENABLED:
        # Nothing is shown here when the assistant is off, so the header stays
        # clean for users who never enable it.
        pass
    elif early_active_tab == "assistant":
        st.page_link(
            pages["Pipeline"][0],
            label="Back to Workflow",
            icon=":material/undo:",
            width="stretch",
        )
    else:
        try:
            from spacemap_gui.components.assistant_panel import render_assistant_launcher

            render_assistant_launcher(early_active_tab, assistant_page)
        except Exception as exc:  # noqa: BLE001 - the pipeline must still load
            # A missing optional dependency must never take the whole app down.
            logging.getLogger(__name__).warning("AI assistant launcher unavailable: %s", exc)
            st.caption("AI assistant unavailable")

# Sidebar for Global Settings
with st.sidebar:
    st.header("Project Management")
    from spacemap_gui.persistence import load_projects_index
    from spacemap_gui.tabs.tab0_project import load_project_callback
    
    projects = load_projects_index()
    
    project_names = ["-- No Project Loaded --", "-- Create New Project --"]
    if projects:
        project_names += list(projects.keys())
        
    # Determine the currently active project
    active_project_name = st.session_state.get('active_project_name')
    if not active_project_name:
        from spacemap_gui.persistence import load_active_session
        persisted_active = load_active_session()
        if persisted_active and projects and persisted_active in projects:
            # Rehydrate silently on browser refresh or terminal restart
            from spacemap_gui.persistence import load_project_config, hydrate_project_state
            config = load_project_config(projects[persisted_active])
            if config:
                hydrate_project_state(config, projects[persisted_active])
                active_project_name = st.session_state.get('active_project_name', persisted_active)
                st.session_state["_force_sidebar_proj_sync"] = True
            else:
                active_project_name = "-- No Project Loaded --"
        else:
            active_project_name = "-- No Project Loaded --"
        
    idx = project_names.index(active_project_name) if active_project_name in project_names else 0
    
    # Sync the selectbox session state programmatically if requested
    if st.session_state.get("_force_sidebar_proj_sync"):
        st.session_state["_global_sidebar_proj"] = active_project_name
        st.session_state["_force_sidebar_proj_sync"] = False
        
    selected_proj_name = st.selectbox("Active Project", project_names, index=idx, key="_global_sidebar_proj")
    
    # If user changed the project, load it instantly
    if selected_proj_name != active_project_name:
        from spacemap_gui.tabs.tab0_project import (
            save_project_callback, unload_project_callback,
        )
        # Persist the outgoing project before anything replaces the session, or
        # every unsaved edit made since the last save is silently discarded.
        save_project_callback()

        if selected_proj_name == "-- No Project Loaded --":
            unload_project_callback()
            st.rerun()
        elif selected_proj_name == "-- Create New Project --":
            unload_project_callback()
            st.switch_page(str(current_dir / "tabs/tab0_project.py"))
        elif projects and selected_proj_name in projects:
            load_project_callback(projects[selected_proj_name])
            st.rerun()

    # What is actually loaded right now. The directory and unified CSV shown here
    # are the ones every tab will use, so a mismatch is visible immediately
    # instead of surfacing as a wrong file three tabs later.
    if st.session_state.get("active_project"):
        active_root = st.session_state.get("active_project_root", "")
        unified_dir = st.session_state.get("global_pheno_dir", "")
        unified_file = st.session_state.get("global_pheno_file", "") or "not set"
        st.caption(f"Root: `{active_root}`")
        st.caption(f"Unified CSV: `{unified_file}`")
        if unified_dir and active_root and not str(unified_dir).startswith(str(active_root)):
            st.warning(
                "The unified CSV directory sits outside this project. "
                "Check Tab 0 -> Pipeline Directories.",
                icon=":material/warning:",
            )
        if st.button("Save Project", width="stretch", key="_sidebar_save_project"):
            from spacemap_gui.tabs.tab0_project import save_project_callback
            save_project_callback()
            st.success("Project saved.")

    st.markdown("---")

    st.header("Global Settings")
    
    # Global Interface Parameters
    with st.expander("Interface Appearance", expanded=True):
        st.markdown("Customize the interface look and feel.")
        
        # Dynamic CSS Variables
        if "global_accent_color" in st.session_state:
            accent_color = st.color_picker("Accent Color", key="global_accent_color")
        else:
            accent_color = st.color_picker("Accent Color", "#4A90E2", key="global_accent_color")

        if "global_border_radius" in st.session_state:
            border_radius = st.slider("Corner Radius (px)", 0, 20, key="global_border_radius")
        else:
            border_radius = st.slider("Corner Radius (px)", 0, 20, 8, key="global_border_radius")

        theme_mode_options = ["System", "Light", "Dark"]
        if "global_theme_mode" in st.session_state:
            theme_mode = st.selectbox("Theme Mode", theme_mode_options, key="global_theme_mode")
        else:
            theme_mode = st.selectbox("Theme Mode", theme_mode_options, index=0, key="global_theme_mode")
        if theme_mode != "System":
            st.caption("Note: Refresh page to apply theme.")
        
        st.markdown("### Typography & Layout")
        if "global_tab_font_size" in st.session_state:
            tab_font_size = st.slider("Tab Title Font Size (px)", 12, 24, key="global_tab_font_size")
        else:
            tab_font_size = st.slider("Tab Title Font Size (px)", 12, 24, 16, key="global_tab_font_size")

        if "global_content_font_size" in st.session_state:
            content_font_size = st.slider("Content Font Size (px)", 12, 24, key="global_content_font_size")
        else:
            content_font_size = st.slider("Content Font Size (px)", 12, 24, 16, key="global_content_font_size")
        
        align_options = {"Top": "flex-start", "Center": "center", "Bottom": "flex-end"}
        tab_align_options = list(align_options.keys())
        if "global_tab_align" in st.session_state:
            tab_text_align = st.selectbox("Tab Title Vertical Align", tab_align_options, key="global_tab_align")
        else:
            tab_align_idx = 1 if len(tab_align_options) > 1 else 0
            tab_text_align = st.selectbox("Tab Title Vertical Align", tab_align_options, index=tab_align_idx, key="global_tab_align")
        align_css = align_options[tab_text_align]

        st.markdown("### View Parameters")
        st.caption("These settings apply to visualizations across all tabs.")
        cmap_options = ["viridis", "plasma", "inferno", "magma", "gray"]
        if "global_cmap" in st.session_state:
            colormap = st.selectbox("Default Colormap", cmap_options, key="global_cmap")
        else:
            colormap = st.selectbox("Default Colormap", cmap_options, index=0, key="global_cmap")

        if "global_thumb_scale" in st.session_state:
            thumbnail_size = st.slider("Thumbnail Scale", 0.5, 2.0, step=0.1, key="global_thumb_scale")
        else:
            thumbnail_size = st.slider("Thumbnail Scale", 0.5, 2.0, 1.0, 0.1, key="global_thumb_scale")

    st.info("Use this panel for settings that apply to the entire pipeline.")
    
    # Session State Management
    st.markdown("---")
    st.subheader("Session Settings")
    st.markdown("Save interface, directory, annotation, and batch-correction selections for the next session.")
    
    if st.button("Save Session Settings", type="primary", width='stretch'):
        save_persistent_state()
        st.success("Session settings saved. They will be restored on the next app start.")
    
    if st.button("Clear Saved Session Settings", width='stretch'):
        try:
            if PERSISTENT_STATE_FILE.exists():
                PERSISTENT_STATE_FILE.unlink()
            st.success("Saved session settings cleared.")
        except Exception as e:
            st.error(f"Error clearing saved state: {e}")
    
    st.markdown("---")
    
    if st.button("Save Configuration"):
        config = {
            "global_accent_color": st.session_state.get("global_accent_color"),
            "global_border_radius": st.session_state.get("global_border_radius"),
            "global_tab_font_size": st.session_state.get("global_tab_font_size"),
            "global_content_font_size": st.session_state.get("global_content_font_size"),
            "global_tab_align": st.session_state.get("global_tab_align"),
            "global_cmap": st.session_state.get("global_cmap"),
            "global_thumb_scale": st.session_state.get("global_thumb_scale"),
            "global_theme_mode": st.session_state.get("global_theme_mode"),
        }
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Save Streamlit Theme
            toml_dir = project_root / ".streamlit"
            toml_dir.mkdir(exist_ok=True)
            toml_path = toml_dir / "config.toml"
            
            current_toml = {}
            if toml_path.exists():
                current_toml = toml.load(toml_path)
            
            if "theme" not in current_toml:
                current_toml["theme"] = {}
                
            selected_theme = st.session_state.get("global_theme_mode")
            if selected_theme == "Light":
                current_toml["theme"]["base"] = "light"
            elif selected_theme == "Dark":
                current_toml["theme"]["base"] = "dark"
            else:
                if "base" in current_toml["theme"]:
                    del current_toml["theme"]["base"]
            
            with open(toml_path, "w") as f:
                toml.dump(current_toml, f)

            st.success("Configuration saved! Refresh to apply theme changes.")
        except Exception as e:
            st.error(f"Error saving config: {e}")

    # Custom CSS for Professional Tabs
    st.markdown(f"""
    <style>
        /* Global Content Font Size */
        .stMarkdown p, .stMarkdown li, .stText, .stHtml {{
            font-size: {content_font_size}px !important;
        }}

        /* Tab List Container */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            margin-bottom: 1rem;
        }}

        /* Individual Tab Styling */
        .stTabs [data-baseweb="tab"] {{
            height: 60px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: {border_radius}px {border_radius}px 0px 0px;
            padding: 0px 16px; /* Removed vertical padding for better flex control */
            color: #555555;
            border: 1px solid #E0E0E0;
            border-bottom: none;
            
            /* Flex Layout */
            display: flex !important;
            flex-direction: row !important; /* Force row to ensure align-items is vertical */
            justify-content: center !important; /* Horizontal Center */
            align-items: {align_css} !important; /* Vertical Align */
            
            flex-grow: 1;
            transition: all 0.2s ease-in-out;
        }}
        
        /* Tab Text Styling - Target inner elements */
        .stTabs [data-baseweb="tab"] div, 
        .stTabs [data-baseweb="tab"] p {{
            font-weight: 700 !important;
            font-size: {tab_font_size}px !important;
            line-height: 1.2 !important;
        }}

        /* Selected Tab Styling */

        /* Selected Tab Styling */
        .stTabs [aria-selected="true"] {{
            background-color: #FFFFFF;
            color: {accent_color};
            border-top: 4px solid {accent_color};
            border-left: 1px solid #E0E0E0;
            border-right: 1px solid #E0E0E0;
            border-bottom: none;
            box-shadow: 0 -4px 8px -2px rgba(0,0,0,0.05);
        }}

        /* Hover State */
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #FFFFFF;
            color: {accent_color};
            border-color: {accent_color};
        }}
        
        /* Global Button Styling to match */
        div.stButton > button {{
            border-radius: {border_radius}px;
            font-weight: 600;
        }}

        /* Pipeline page navigation */
        div[data-testid="stPageLink"] a {{
            text-decoration: none !important;
            justify-content: center !important;
            text-align: center !important;
            min-height: 48px !important;
            display: flex !important;
            align-items: {align_css} !important;
            background-color: #F0F2F6 !important;
            border-radius: {border_radius}px !important;
            padding: 6px 12px !important;
            color: #555555 !important;
            border: 1px solid #E0E0E0 !important;
            transition: all 0.2s ease-in-out !important;
            white-space: normal !important;
        }}
        div[data-testid="stPageLink"] a:hover {{
            background-color: #FFFFFF !important;
            color: {accent_color} !important;
            border-color: {accent_color} !important;
        }}
        /* Keep navigation labels compact even when content tabs use larger text. */
        div[data-testid="stPageLink"] a p,
        div[data-testid="stPageLink"] a div,
        div[data-testid="stPageLink"] a span {{
            font-weight: 700 !important;
            font-size: 15px !important;
            line-height: 1.2 !important;
            color: inherit !important;
        }}
    </style>
    """, unsafe_allow_html=True)

navigation_labels = (
    "Workspace",
    "Registration",
    "Background",
    "ROI",
    "Annotation",
    "Segmentation",
    "Batch Correction",
    "Phenotyping",
    "Spatial Analysis",
    "Visualization",
)
active_navigation_href = {
    "tab0": "",
    "tab1": "registration",
    "tab2": "bg_subtraction",
    "tab3": "roi",
    "tab4": "annotation",
    "tab5": "segmentation",
    "tab6": "batch_correction",
    "tab7": "phenotyping",
    "tab8": "analysis",
    "tab9": "visualization",
    "assistant": "assistant",
}[early_active_tab]
st.markdown(
    f"""
    <style>
        div[data-testid="stPageLink"] a[href="{active_navigation_href}"] {{
            background-color: #FFFFFF !important;
            color: {accent_color} !important;
            border-top: 4px solid {accent_color} !important;
            box-shadow: 0 -4px 8px -2px rgba(0,0,0,0.05) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
for row_start in range(0, len(pages["Pipeline"]), 5):
    nav_cols = st.columns(5, gap="small")
    for column, page, label in zip(
        nav_cols,
        pages["Pipeline"][row_start : row_start + 5],
        navigation_labels[row_start : row_start + 5],
    ):
        with column:
            st.page_link(page, label=label, width="stretch")

# Auto-save directory selections when they change (no button required)
auto_save_directories_if_changed()

# Set the active tab keys based on current page
active_tab = early_active_tab

set_active_tab(active_tab)
set_user_tab(active_tab)
st.session_state["_active_tab_session"] = active_tab
st.session_state["_currently_rendering_tab"] = active_tab

# Keep the purpose, input/output handoff, and next action consistent on every page.
from spacemap_gui.workflow_guidance import render_workflow_guidance
render_workflow_guidance(active_tab)

# Run the page
pg.run()

# Shared timestamped log for the current pipeline page and any of its subtabs.
from spacemap_gui.process_log import render_process_log_panel
render_process_log_panel(active_tab)

# Footer
st.markdown("---")
st.markdown("SPACEMAP AI | Aguilera Lab, UTSW, Dallas, TX")
