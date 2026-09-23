import json
import os
import shutil
import tempfile
import ast
import re
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import streamlit as st


from spacemap.userpaths import state_dir

# ``current_dir`` is the installed package directory.  It is read-only in a
# normal installation and is used here only to scan the source tree for widget
# keys; everything this module *writes* goes to the user state directory.
current_dir = Path(__file__).resolve().parent


def _user_state_file(name: str) -> Path:
    """Return the user-writable location for a state file.

    Earlier versions kept these files inside the package directory.  When such
    a file is found and no user copy exists yet, it is migrated once so that an
    in-place upgrade does not lose a configured workspace.
    """
    target = state_dir() / name
    if target.exists():
        return target
    legacy = current_dir / name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if legacy.is_file():
            shutil.copy2(legacy, target)
    except OSError:
        # A read-only or missing home directory must not stop the app from
        # starting; the callers below already tolerate an unreadable file.
        pass
    return target


PERSISTENT_SETTINGS_FILE = _user_state_file("interface_settings.json")

# ============================================================================
# INTERFACE STATE (interface_state.json) — shared atomic read-modify-write
# ============================================================================
# This file is the cross-session cache for directory selections and per-tab
# state (global_pheno_dir, tab8_csv_configs, tab9 params, etc.).
#
# Historically, app.py, tab7_phenotyping/common.py, tab8_analysis/common.py,
# and tab9_visualization/common.py each had their own _load_state/_update_state
# helpers that read-modify-wrote the whole file non-atomically. This caused
# lost-update races when two tabs wrote in the same render cycle.
#
# Persistence layer precedence (highest to lowest priority):
#   1. <project>.json  — active project config (authoritative when a project
#                        is loaded; written by save_project_config / sync_project_state)
#   2. interface_state.json — cross-session cache for directory selections and
#                        per-tab state when NO project is loaded, plus a few
#                        tab-specific caches (tab7/8/9) that live here even
#                        when a project IS loaded.
#   3. interface_settings.json — per-directory UI settings (e.g. tab7_viz
#                        settings bound to a specific input directory).
#
# When a project is loaded, project.json's directory_mappings overrides
# interface_state.json for directory keys (see sync_project_state). The
# tab-specific caches (tab8_csv_configs, tab9 params) still read/write
# interface_state.json regardless of project state.
INTERFACE_STATE_FILE = _user_state_file("interface_state.json")


@lru_cache(maxsize=1)
def _discover_action_widget_keys() -> frozenset[str]:
    """Return literal keys belonging to buttons/download actions.

    Streamlit stores button state in ``session_state`` during a rerun, but a
    button click is an event, not a reproducible pipeline parameter.  Using
    the source tree to identify literal action keys lets us persist similarly
    named *parameters* (for example ``tab9_actual_load_cells``) without
    relying on broad substring rules such as ``"load" in key``.
    """
    action_keys, _prefixes, _suffixes, _infixes = _scan_action_widget_keys()
    return action_keys


@lru_cache(maxsize=1)
def _discover_action_widget_prefixes() -> tuple[str, ...]:
    """Return literal prefixes of dynamically keyed buttons.

    Many action widgets are keyed with an f-string, for example
    ``key=f"tab4_fast_membrane_{file_key}"``.  Those keys cannot be matched
    exactly, but their literal prefix can be: any session-state key starting
    with one is a button event, not a pipeline parameter.  Without this, such a
    key is written into the project config and restored into ``session_state`` on
    a later run, and Streamlit then refuses to build the button at all
    (``StreamlitValueAssignmentNotAllowedError``).
    """
    _keys, prefixes, _suffixes, _infixes = _scan_action_widget_keys()
    return prefixes


ACTION_WIDGET_CALLS = {"button", "form_submit_button", "download_button"}
SETTING_WIDGET_CALLS = {
    "checkbox", "selectbox", "multiselect", "radio", "slider", "select_slider",
    "number_input", "text_input", "text_area", "color_picker", "date_input",
    "time_input", "data_editor", "file_uploader", "toggle",
}


@lru_cache(maxsize=1)
def _discover_action_widget_suffixes() -> tuple[str, ...]:
    """Literal suffixes of buttons keyed as ``f"{variable}_literal"``."""
    _keys, _prefixes, suffixes, _infixes = _scan_action_widget_keys()
    return suffixes


@lru_cache(maxsize=1)
def _discover_action_widget_infixes() -> tuple[str, ...]:
    """Literal middles of buttons keyed as ``f"{prefix}_literal_{index}"``.

    These keys have no literal prefix or suffix, so they are matched on the
    distinctive segment between the placeholders.
    """
    _keys, _prefixes, _suffixes, infixes = _scan_action_widget_keys()
    return infixes


@lru_cache(maxsize=1)
def _scan_action_widget_keys() -> tuple[frozenset[str], tuple[str, ...], tuple[str, ...]]:
    """Parse the GUI sources once for literal and f-string action-widget keys.

    Scans the whole ``spacemap_gui`` package, not just ``tabs``: shared modules
    such as ``process_log.py``, ``ui_utils.py`` and ``file_browser.py`` also
    create buttons, and their keys leak into the project config exactly the same
    way.

    Returns ``(exact_keys, prefixes, suffixes)``. Dynamic keys are matched on
    their literal prefix (``f"tab4_fast_membrane_{img}"``) or literal suffix
    (``f"{page}_process_log_clear"``). A suffix that also terminates a real
    setting's key is discarded, so recognising ``f"{p}_png"`` as an action can
    never stop ``tab4_save_png`` from being saved.
    """
    action_keys: set[str] = set()
    prefixes: set[str] = set()
    suffixes: set[str] = set()
    infixes: set[str] = set()
    setting_keys: set[str] = set()

    if not current_dir.exists():
        return frozenset(), (), ()

    for source_path in current_dir.rglob("*.py"):
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            is_action = node.func.attr in ACTION_WIDGET_CALLS
            is_setting = node.func.attr in SETTING_WIDGET_CALLS
            if not (is_action or is_setting):
                continue
            for keyword in node.keywords:
                if keyword.arg != "key":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    (action_keys if is_action else setting_keys).add(value.value)
                elif is_action and isinstance(value, ast.JoinedStr) and value.values:
                    head, tail = value.values[0], value.values[-1]
                    if isinstance(head, ast.Constant) and isinstance(head.value, str) and len(head.value) >= 3:
                        prefixes.add(head.value)
                    elif isinstance(tail, ast.Constant) and isinstance(tail.value, str) and len(tail.value) >= 3:
                        # Only usable when the key does not already start with a
                        # literal, otherwise the prefix is the tighter match.
                        suffixes.add(tail.value)
                    else:
                        # A key shaped f"{prefix}_literal_{index}" begins and ends
                        # with a placeholder, so it has neither a literal prefix
                        # nor suffix and was previously invisible here. Its middle
                        # segments still identify it, and missing one means the
                        # button's key is persisted as though it were a setting --
                        # which Streamlit then refuses to restore, raising
                        # StreamlitValueAssignmentNotAllowedError on the next load.
                        for part in value.values[1:-1]:
                            if (
                                isinstance(part, ast.Constant)
                                and isinstance(part.value, str)
                                and len(part.value.strip("_")) >= 4
                            ):
                                infixes.add(part.value)

    safe_suffixes = {
        suffix for suffix in suffixes
        if not any(setting.endswith(suffix) for setting in setting_keys)
    }
    safe_infixes = {
        infix for infix in infixes
        if not any(infix in setting for setting in setting_keys)
    }
    return (
        frozenset(action_keys),
        tuple(sorted(prefixes)),
        tuple(sorted(safe_suffixes)),
        tuple(sorted(safe_infixes)),
    )


#: Section holding one isolated cache per project root. Everything outside it is
#: the no-project cache.
PROJECT_STATE_SECTION = "__projects__"


def _read_interface_state_file() -> dict[str, Any]:
    """Raw contents of interface_state.json, including every project section."""
    if not INTERFACE_STATE_FILE.exists():
        return {}
    try:
        with open(INTERFACE_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        log_persistence(f"Error loading interface_state.json: {e}")
        return {}


def active_project_state_key() -> str | None:
    """Identity of the loaded project, or None when running without one."""
    try:
        root = st.session_state.get("active_project_root")
    except Exception:
        # Outside a Streamlit script run (tests, CLI) there is no session.
        return None
    return str(root) if root else None


def load_interface_state() -> dict[str, Any]:
    """Cached UI state for the *active* project only.

    Each project gets its own isolated section. Without this, directory and file
    selections written by one project were read back by the next one loaded --
    which is how a project could open showing another project's unified CSV.
    There is deliberately no fallback from a project's section to the shared
    one: an empty section means "this project has not chosen yet", and the
    correct source for that is the project config, not another project's cache.
    """
    raw = _read_interface_state_file()
    project_key = active_project_state_key()
    if project_key:
        section = raw.get(PROJECT_STATE_SECTION, {})
        return dict(section.get(project_key, {})) if isinstance(section, dict) else {}
    return {key: value for key, value in raw.items() if key != PROJECT_STATE_SECTION}


def update_interface_state(updates: dict[str, Any]) -> None:
    """Atomically merge ``updates`` into the active project's cache section.

    Reads the current state, merges ``updates`` on top, and writes back via
    a temp file + os.replace() so that concurrent writers don't corrupt the
    file or lose each other's keys.
    """
    state = _read_interface_state_file()
    project_key = active_project_state_key()
    if project_key:
        section = state.setdefault(PROJECT_STATE_SECTION, {})
        if not isinstance(section, dict):
            section = {}
            state[PROJECT_STATE_SECTION] = section
        section.setdefault(project_key, {}).update(updates)
    else:
        state.update(updates)
    try:
        INTERFACE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file in same dir, then rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(INTERFACE_STATE_FILE.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, str(INTERFACE_STATE_FILE))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        log_persistence(f"Error updating interface_state.json: {e}")


def get_interface_state_value(key: str, default: Any = None) -> Any:
    """Read a single key from interface_state.json."""
    return load_interface_state().get(key, default)


def set_interface_state_value(key: str, value: Any) -> None:
    """Write a single key to interface_state.json (atomic merge)."""
    update_interface_state({key: value})


def get_default_bc_config() -> dict:
    """Canonical default configuration for batch correction (Tab 6).

    Single source of truth shared by `app.py` (early initialization) and
    `tab6_batch_correction.render_setup()` (lazy initialization). Keep keys
    in sync with the UI widgets in `render_setup`.
    """
    return {
        'batch_id_strategy': 'Filename (strip suffixes)',
        'filename_pattern': 'P*_roi_*',
        'filename_suffixes_to_strip': '.ome_cell_features_annotated,_fully_annotated,_cell_features,_cell_features_annotated',
        'exclude_markers': ['DAPI', 'Blank', 'Empty', 'DNA'],
        'method': 'Combat',
        'cell_id_col': 'cell_id',
        'cell_id_strategy': 'Auto-generate global (cell_0, cell_1, ...)',
        'local_cell_id_col': 'label',
        'batch_id_col': 'batch_id',
        'tissue_col': 'tissue',
        'tissue_val': 1,
        'area_col': 'area',
        'area_min_pct': 0.1,
        'area_max_pct': 99.9,
        'outlier_high': 99.9,
        'outlier_low': 0.1,
        'add_corrected_indicator': False,
        'corrected_indicator_col': 'corrected',
        'arcsinh_cofactor': 5.0,
        'enable_post_qc': False,
        'filter_overcorrected': False,
        'qc_percentile_threshold': 95.0,
        'filter_distance': False,
        'max_correction_distance': 10.0,
    }

def log_persistence(msg: str):
    """Project-file messages (shown when SPACEMAP_APP_LOGS=1)."""
    from spacemap_gui import terminal

    terminal.say("project", msg, channel="app")

# ============================================================================
# DIRECTORY-SPECIFIC SETTINGS PERSISTENCE
# ============================================================================

def _get_settings_key(dir_path: str) -> str:
    """Normalize directory path to use as a storage key."""
    if not dir_path:
        return ""
    return str(Path(dir_path).resolve())

def load_directory_settings(dir_path: str, section_key: str) -> dict:
    """Load settings specific to a directory and section (e.g. 'tab6')."""
    if not dir_path or not PERSISTENT_SETTINGS_FILE.exists():
        return {}
    
    key = _get_settings_key(dir_path)
    if not key:
        return {}

    try:
        with open(PERSISTENT_SETTINGS_FILE, 'r') as f:
            all_settings = json.load(f)
            dir_settings = all_settings.get(key, {})
            return dir_settings.get(section_key, {})
    except Exception as e:
        log_persistence(f"Error loading directory settings: {e}")
        return {}

def save_directory_settings(dir_path: str, section_key: str, settings: dict) -> None:
    """Save settings for a specific directory and section."""
    if not dir_path:
        return
        
    key = _get_settings_key(dir_path)
    if not key:
        return

    all_settings = {}
    if PERSISTENT_SETTINGS_FILE.exists():
        try:
            with open(PERSISTENT_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                all_settings = json.load(f) or {}
        except Exception as e:
            log_persistence(f"Error loading directory settings for update: {e}")
            all_settings = {}
    
    if key not in all_settings:
        all_settings[key] = {}
    
    # Overwrite section settings
    all_settings[key][section_key] = settings
    
    try:
        # Keep the directory-scoped cache recoverable if Streamlit reruns or
        # two tabs save close together.  This cache is supplemental; project
        # JSON remains the source of truth for an active project.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(PERSISTENT_SETTINGS_FILE.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(sanitize_json_obj(all_settings), f, indent=2)
            os.replace(tmp_path, str(PERSISTENT_SETTINGS_FILE))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        log_persistence(f"Error saving directory settings: {e}")

# ============================================================================
# PROJECT INDEX & EXPERIMENT PERSISTENCE
# ============================================================================

def record_latest_unified_csv(filename: str | None) -> None:
    """Remember the most recently opened unified CSV on the active project.

    A project may legitimately hold several unified CSVs. Only the last one
    opened is stored, so reloading the project reopens what you were working on
    rather than snapping back to whatever was set when the project was created.

    Only the config is touched: ``proj_unified_csv_filename`` is a Tab 0 widget
    key, and writing a widget key from another page is exactly the kind of
    cross-page write Streamlit rejects. On the next load, sync_project_state
    derives both that widget and the three global_*_file keys from here.
    """
    name = str(filename or "").strip()
    if not name:
        return
    config = st.session_state.get("active_project")
    if not isinstance(config, dict) or config.get("unified_csv_filename") == name:
        return
    config["unified_csv_filename"] = name


def get_projects_index_path() -> Path:
    """Get the path to the global projects index file, creating parent directories."""
    path = Path.home() / ".spacemap_gui" / "projects.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_projects_index() -> dict:
    """Load the dictionary of saved projects mapping project names to project roots."""
    path = get_projects_index_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        log_persistence(f"Error loading projects index: {e}")
        return {}

def save_project_to_index(name: str, root_path: str) -> None:
    """Add or update a project in the global projects index file."""
    index = load_projects_index()
    index[name] = str(Path(root_path).resolve())
    path = get_projects_index_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        log_persistence(f"Error saving project to index: {e}")

def remove_project_from_index(name: str) -> None:
    """Remove a project from the global projects index file."""
    index = load_projects_index()
    if name in index:
        del index[name]
        path = get_projects_index_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            log_persistence(f"Error removing project from index: {e}")

def get_active_session_path() -> Path:
    path = Path.home() / ".spacemap_gui" / "active_session.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_active_session(project_name: str) -> None:
    path = get_active_session_path()
    try:
        if not project_name or project_name == "-- No Project Loaded --":
            if path.exists():
                path.unlink()
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"active_project_name": project_name}, f)
    except Exception as e:
        log_persistence(f"Error saving active session: {e}")

def load_active_session() -> str | None:
    path = get_active_session_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("active_project_name")
    except Exception as e:
        return None

def load_project_config(project_root: str) -> dict:
    """Load the project config from the project root directory.
    Checks for <project_name>.json first, then falls back to experiment.json.
    """
    project_root_path = Path(project_root)
    project_name = project_root_path.name
    path = project_root_path / f"{project_name}.json"
    if not path.exists():
        path = project_root_path / "experiment.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f) or {}
            
        # Dynamic path fix-up for portability:
        # If the project was moved or renamed, update all internal paths pointing to the old project_root
        saved_root = config.get("project_root")
        current_root = str(project_root_path.resolve())
        if saved_root and saved_root != current_root:
            log_persistence(f"Project moved: adapting paths from {saved_root} to {current_root}")
            
            def resolve_config_paths(val, old_root: str, new_root: str):
                if isinstance(val, str):
                    norm_old = os.path.normpath(old_root)
                    norm_new = os.path.normpath(new_root)
                    norm_val = os.path.normpath(val)
                    if norm_val.startswith(norm_old):
                        rel = os.path.relpath(norm_val, norm_old)
                        if rel == '.' or not rel:
                            return norm_new
                        return str(Path(norm_new) / rel)
                    return val
                elif isinstance(val, dict):
                    return {k: resolve_config_paths(v, old_root, new_root) for k, v in val.items()}
                elif isinstance(val, list):
                    return [resolve_config_paths(item, old_root, new_root) for item in val]
                return val
            
            config = resolve_config_paths(config, saved_root, current_root)
            config["project_root"] = current_root
            if "codex_raw_dir" in config:
                config["codex_raw_dir"] = resolve_config_paths(config["codex_raw_dir"], saved_root, current_root)
            if "he_raw_dir" in config:
                config["he_raw_dir"] = resolve_config_paths(config["he_raw_dir"], saved_root, current_root)
            
            # Save the updated config back to disk to store the corrected paths
            save_project_config(current_root, config)
            
        return config
    except Exception as e:
        # A project that fails to load is NOT an empty project. Returning {} here
        # silently hands every tab a blank configuration, which looks exactly
        # like "all my settings were deleted" while the file on disk is fine.
        # Make it loud, and say how big the file was: an unreadable config is
        # almost always an oversized one.
        try:
            size_mb = path.stat().st_size / 1e6
        except Exception:
            size_mb = float("nan")
        log_persistence(
            f"ERROR: could not load project config {path.name} ({size_mb:.1f} MB): {e}. "
            "Tabs will show blank settings until this is resolved -- the saved values are "
            "still in the file. A config far above a few hundred KB means data leaked into it."
        )
        try:
            import streamlit as _st

            _st.error(
                f"Project configuration could not be read ({path.name}, {size_mb:.1f} MB): {e}\n\n"
                "Settings below are defaults, not your saved values. Do not re-save the project "
                "until this is fixed, or the blank defaults will overwrite what is on disk."
            )
        except Exception:
            pass
        return {}

def sanitize_json_obj(obj):
    """Recursively convert numpy/pandas types to standard JSON-serializable types."""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, (np.integer, int)):
                new_key = int(k)
            elif isinstance(k, (np.floating, float)):
                new_key = float(k)
            elif isinstance(k, (np.bool_, bool)):
                new_key = bool(k)
            else:
                new_key = str(k)
            new_dict[new_key] = sanitize_json_obj(v)
        return new_dict
    elif isinstance(obj, list):
        return [sanitize_json_obj(x) for x in obj]
    elif isinstance(obj, tuple):
        return [sanitize_json_obj(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_json_obj(x) for x in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return sanitize_json_obj(obj.to_dict(orient="records"))
    elif isinstance(obj, pd.Series):
        return sanitize_json_obj(obj.to_dict())
    elif isinstance(obj, (set, frozenset)):
        return [sanitize_json_obj(x) for x in obj]
    
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return sanitize_json_obj(obj.to_dict())
            except Exception as e:
                log_persistence(f"sanitize_json_obj: to_dict() fallback failed: {e}")
        return str(obj)


#: Config sections whose values may contain paths inside the project. `audit_log`
#: is excluded on purpose: it records what happened, not where things live now.
PORTABLE_CONFIG_SECTIONS = (
    "directory_mappings",
    "tab_configurations",
    "discovered_pairs",
    "codex_raw_dir",
    "he_raw_dir",
)


def _project_directory_settings(project_root: str | Path) -> dict:
    """interface_settings.json sections whose directory sits inside this project."""
    from spacemap_gui.project_bundle import _real

    root = _real(project_root)
    try:
        with open(PERSISTENT_SETTINGS_FILE, "r", encoding="utf-8") as handle:
            everything = json.load(handle) or {}
    except Exception:
        return {}

    owned = {}
    for key, value in everything.items():
        # Keys are hashed directory paths plus a section name; the stored payload
        # carries the directory, so match on whatever path-like strings it holds.
        candidate = json.dumps(value)
        if root in candidate:
            owned[key] = value
    return owned


def attach_portable_state(project_root: str, config: dict) -> dict:
    """Return `config` with the out-of-file settings folded into it.

    Also rewrites directory mappings and tab configurations so paths inside the
    project become ``${PROJECT_ROOT}/...``; the on-load path expands them again.
    """
    from spacemap_gui.project_bundle import BUNDLE_KEY, collect_bundle, portablize

    # A copy: the live session config must keep real paths, or the running app
    # would start seeing ${PROJECT_ROOT} tokens in its directory fields.
    enriched = dict(config)
    enriched[BUNDLE_KEY] = collect_bundle(
        project_root,
        interface_state=load_interface_state(),
        directory_settings=_project_directory_settings(project_root),
    )
    for section in PORTABLE_CONFIG_SECTIONS:
        if section in enriched:
            enriched[section] = portablize(enriched[section], project_root)
    enriched["project_root"] = str(project_root)
    return enriched


def restore_portable_state(config: dict, project_root: str) -> dict:
    """Expand a loaded config back into real paths and restore the stray files.

    Returns the config with paths localized. The tab-params files are written to
    disk and the tab 8 / tab 9 interface state is installed for this project.
    """
    from spacemap_gui.project_bundle import BUNDLE_KEY, apply_bundle, localize

    # Mutated in place: callers and session state share this object, and
    # sync_project_state relies on mutations to it being visible.
    localized = config
    for section in PORTABLE_CONFIG_SECTIONS:
        if section in localized:
            localized[section] = localize(localized[section], project_root)
    localized["project_root"] = str(project_root)

    bundle = localized.get(BUNDLE_KEY)
    if not bundle:
        return localized

    name = Path(project_root).name
    config_file = Path(project_root) / f"{name}.json"
    config_mtime = None
    try:
        config_mtime = config_file.stat().st_mtime
    except OSError:
        pass

    try:
        interface_state = apply_bundle(project_root, bundle, config_mtime=config_mtime)
        if interface_state:
            update_interface_state(interface_state)
    except Exception as exc:
        log_persistence(f"Could not restore portable state: {exc}")
    return localized


#: Rolling restore points live here, newest last.
RESTORE_POINT_DIRNAME = ".restore_points"
RESTORE_POINT_LIMIT = 5

#: Project roots already snapshotted in this process, so a session takes one.
_RESTORE_POINTS_TAKEN: set[str] = set()


def restore_point_dir(project_root: str | Path) -> Path:
    return Path(project_root) / RESTORE_POINT_DIRNAME


def list_restore_points(project_root: str | Path) -> list[Path]:
    """Existing restore points, oldest first."""
    directory = restore_point_dir(project_root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), key=lambda p: p.name)


def prune_restore_points(project_root: str | Path, keep: int = RESTORE_POINT_LIMIT) -> int:
    """Delete all but the newest `keep` restore points. Returns how many went."""
    points = list_restore_points(project_root)
    removed = 0
    for stale in points[:max(len(points) - keep, 0)]:
        try:
            stale.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def create_restore_point(
    project_root: str | Path,
    label: str = "autosave",
    once_per_session: bool = False,
) -> Path | None:
    """Snapshot the current project config before it is overwritten.

    With `once_per_session`, only the first save after loading a project takes a
    snapshot -- that is the "before I started changing things" state, which is
    the one worth rolling back to. Explicit snapshots (the Tab 0 button) always
    write. A config that will not parse is never snapshotted, so a corrupt file
    cannot evict a good restore point.
    """
    root = str(project_root)
    if once_per_session and root in _RESTORE_POINTS_TAKEN:
        return None

    name = Path(root).name
    source = Path(root) / f"{name}.json"
    if not source.exists():
        source = Path(root) / "experiment.json"
    if not source.exists():
        return None

    try:
        with open(source, "r", encoding="utf-8") as handle:
            json.load(handle)
    except Exception as exc:
        log_persistence(f"Skipping restore point: config is not valid JSON ({exc})")
        return None

    try:
        directory = restore_point_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = directory / f"{stamp}_{label}.json"
        shutil.copy2(source, target)
        _RESTORE_POINTS_TAKEN.add(root)
        prune_restore_points(root)
        # Routine bookkeeping -- one of these is written on every save. Kept
        # silent so the terminal shows what is running, not what is filing.
        # Failures below are still logged.
        return target
    except Exception as exc:
        log_persistence(f"Could not create restore point: {exc}")
        return None


# A project file records configuration: directories, file choices and the values
# entered in each tab. A healthy one is a few hundred KB. Anything approaching
# this bound means data has leaked into it, so say so loudly at the moment of
# writing rather than letting it grow until the GUI takes minutes to start.
_PROJECT_CONFIG_WARN_BYTES = 8 * 1024 * 1024


def prune_oversized_config_entries(config: dict) -> list[tuple[str, str, int]]:
    """Drop data-shaped entries from a config before it is written.

    The per-value cap in ``update_stored_config_key`` only guards *new* writes.
    A project saved by an older build already carries the oversized entries, and
    because they arrive with the loaded config they are copied straight back out
    on every save. Removing them here lets an old project shrink itself the
    first time it is saved, instead of carrying the weight forever.

    Returns the ``(tab, key, size)`` triples that were removed.
    """
    removed: list[tuple[str, str, int]] = []
    for tab, tab_config in (config.get("tab_configurations") or {}).items():
        if not isinstance(tab_config, dict):
            continue
        for key in list(tab_config):
            if is_transient_key(key):
                tab_config.pop(key, None)
                removed.append((tab, key, 0))
                continue
            try:
                size = len(json.dumps(tab_config[key], default=str))
            except Exception:
                tab_config.pop(key, None)
                removed.append((tab, key, -1))
                continue
            if size > _MAX_CONFIG_VALUE_BYTES:
                tab_config.pop(key, None)
                removed.append((tab, key, size))
    return removed


def _warn_if_config_oversized(config: dict, path: Path) -> None:
    """Log the largest offenders when a project config grows data-shaped."""
    try:
        size = len(json.dumps(config, default=str))
    except Exception:
        return
    if size <= _PROJECT_CONFIG_WARN_BYTES:
        return
    offenders = []
    for tab, tab_cfg in (config.get("tab_configurations") or {}).items():
        if not isinstance(tab_cfg, dict):
            continue
        for key, value in tab_cfg.items():
            try:
                offenders.append((len(json.dumps(value, default=str)), f"{tab}:{key}"))
            except Exception:
                continue
    offenders.sort(reverse=True)
    log_persistence(
        f"WARNING: project config for {path.name} is {size / 1e6:.1f} MB, well beyond the "
        f"{_PROJECT_CONFIG_WARN_BYTES // (1024 * 1024)} MB expected for settings. "
        "Something is writing data, not configuration. Largest entries: "
        + ", ".join(f"{name} ({n / 1e6:.1f} MB)" for n, name in offenders[:5])
    )


def save_project_config(project_root: str, config: dict) -> None:
    """Save the project config as <project_name>.json to the project root directory."""
    project_root_path = Path(project_root)
    project_name = project_root_path.name
    path = project_root_path / f"{project_name}.json"
    pruned = prune_oversized_config_entries(config)
    if pruned:
        total = sum(size for _tab, _key, size in pruned if size > 0)
        log_persistence(
            f"Removed {len(pruned)} data-shaped entr(ies) ({total / 1e6:.1f} MB) from the project "
            "configuration: " + ", ".join(f"{tab}:{key}" for tab, key, _size in pruned[:5])
            + (" ..." if len(pruned) > 5 else "")
        )
    _warn_if_config_oversized(config, path)
    try:
        # Create directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fold in everything that lives outside this file so the saved config is a
        # complete, drop-in record of the project.
        try:
            config = attach_portable_state(project_root, config)
        except Exception as exc:  # never let bundling block a save
            log_persistence(f"Could not attach portable state: {exc}")

        # Sanitize config to prevent JSON serialization errors on numpy/pandas types
        sanitized_config = sanitize_json_obj(config)
        
        # Serialize to string FIRST to prevent file corruption/truncation on TypeErrors
        json_str = json.dumps(sanitized_config, indent=2)
        
        # One restore point per session rather than a full copy on every save.
        # These configs reach 8 MB+, and sync_project_state saves whenever any
        # setting changes, so copy-on-every-save cost megabytes per keystroke
        # while still keeping only a single level of history.
        create_restore_point(project_root, once_per_session=True)

        # Atomic write via a *uniquely named* temp file in the same directory.
        # A fixed name like "<project>.tmp" is shared by every process, so two
        # GUI sessions saving at once interleave their writes into one file and
        # os.replace() then publishes the mangled result as the project config.
        # mkstemp gives each writer its own file, matching interface_state.json.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
            raise
    except Exception as e:
        log_persistence(f"Error saving project config: {e}")

def save_active_project_state(tab_name: str = None) -> None:
    """Saves the active project's state including all tab configurations to the project JSON file."""
    if "active_project" not in st.session_state or not st.session_state["active_project"]:
        return
        
    project_root = st.session_state.get("active_project_root")
    if not project_root:
        return
        
    # Run sync to ensure everything is captured
    sync_project_state()
    
    config = st.session_state["active_project"]
    
    # Re-discover pairs
    from spacemap_gui.tabs.tab0_project import discover_image_pairs
    directory_mappings = config.get("directory_mappings", {})
    codex_dir = directory_mappings.get("batch_codex_dir", "")
    he_dir = directory_mappings.get("batch_he_dir", "")
    pairs = discover_image_pairs(codex_dir, he_dir)
    config["discovered_pairs"] = pairs
    config["updated_at"] = datetime.now().isoformat()
    
    previous = st.session_state.get("active_project")
    unchanged = isinstance(previous, dict) and values_equal(
        {k: v for k, v in previous.items() if k != "updated_at"},
        {k: v for k, v in config.items() if k != "updated_at"},
    )
    save_project_config(project_root, config)
    st.session_state["active_project"] = config
    # An unchanged re-save is the common case on a rerun; announcing it every
    # time buries the messages that describe actual work.
    if not unchanged:
        log_persistence(f"Saved project configuration ({tab_name})." if tab_name else "Saved project configuration.")


def get_tab_affected_from_key(key: str) -> str:
    """Map a session state key to the corresponding tab/component name."""
    from spacemap_gui.tabs.tab0_project import DIRECTORY_KEYS
    if key.startswith("proj_dir_"):
        key = key[9:]
        
    if key in DIRECTORY_KEYS:
        if key in ["batch_codex_dir", "batch_he_dir", "tab1_batch_output_dir", "registration_output_dir"]:
            return "Registration"
        elif key in ["tab2_input_dir_reg", "tab2_input_dir_raw", "tab2_output_dir", "bg_subtraction_output_dir"]:
            return "Background Subtraction"
        elif key in ["tab3_input_dir", "roi_output_dir", "tab3_output_dir", "tab3_geojson_dir", "tab3_multi_input_dir", "tab3_multi_geojson_dir", "tab3_multi_output_dir"]:
            return "ROI Extraction"
        elif key in ["tab4_input_dir", "tab4_output_dir", "tab4_seg_mask_save_dir", "tab4_csv_dir", "tab4_geojson_dir",
                     "tab4_annot_geojson_dir", "batch_input_dir", "batch_geojson_dir", "batch_output_dir"]:
            return "Segmentation"
        elif key in ["annot_input_dirs", "annot_image_dir", "annot_output_dir", "zone_input_dir", "zone_output_dir", "cell_seg_dir", "cell_zone_dir", "cell_out_dir", "vis_csv_dir", "vis_img_dir",
                     "annot_ta_input_dir", "annot_ta_output_dir", "annot_ta_pool_input_dir", "annot_ta_pool_output_dir"]:
            return "Annotation"
        elif key in ["bc_input_dir", "bc_output_dir"]:
            return "Batch Correction"
        elif key in ["scimap_bc_dir", "global_pheno_dir", "phenotyping_output_dir", "viz_roi_dir", "viz_seg_dir"]:
            return "Phenotyping"
        elif key in ["spatial_input_dir", "spatial_output_dir"]:
            return "Spatial Analysis"
        elif key.startswith("tab9_"):
            return "Visualization"
        return "Directories"

    if (key.startswith("tab1_") or key.startswith("batch_") or key.startswith("proc_")
            or key.startswith("over_") or key.startswith("pair_codex_mpp_")
            or key.startswith("pair_he_mpp_") or key.startswith("multi_")):
        return "Registration"
    elif key.startswith("tab2_") or key.startswith("bg_"):
        return "Background Subtraction"
    elif key.startswith("tab3_") or key.startswith("roi_"):
        return "ROI Extraction"
    elif key.startswith("tab4_") or key.startswith("seg_"):
        return "Segmentation"
    elif (key.startswith("tab5_") or key.startswith("annot_") or key.startswith("zone_")
          or key.startswith("cell_") or key.startswith("vis_") or key.startswith("edit_")):
        return "Annotation"
    elif key.startswith("tab6_") or key.startswith("bc_"):
        return "Batch Correction"
    elif (key.startswith("tab7_") or key.startswith("leiden_") or key.startswith("som_") or
          key.startswith("vortex_") or key.startswith("resolve_") or key.startswith("scimap_") or
          key.startswith("tools_") or key.startswith("mrml_") or key.startswith("mlc_") or
          key.startswith("subpheno_") or key.startswith("thresh_") or key.startswith("viz_") or
          key.startswith("global_pheno_")):
        return "Phenotyping"
    elif key.startswith("tab8_") or key.startswith("global_spatial_") or key.startswith("_tab8_"):
        return "Spatial Analysis"
    elif (key.startswith("tab9_") or key.startswith("viz_val_") or key.startswith("umap_") or
          key.startswith("cellviz_") or key.startswith("pairwise_") or key.startswith("proportion_") or
          key.startswith("heatmap_") or key.startswith("f1_") or key.startswith("global_visualization_")):
        return "Visualization"
    elif (key.startswith("proj_") or key in ["proj_mpp_codex", "proj_mpp_he", "proj_use_gpu",
                                               "proj_gpu_id", "proj_num_workers", "proj_pi",
                                               "proj_institution", "proj_description"]):
        return "Project Configurations"
    return "Unknown"


def get_tab_name_from_key(key: str) -> Optional[str]:
    """Get the JSON tab config group key from a session state key prefix."""
    affected = get_tab_affected_from_key(key)
    mapping = {
        "Registration": "tab1_registration",
        "Background Subtraction": "tab2_background_subtraction",
        "ROI Extraction": "tab3_roi_extraction",
        "Annotation": "tab4_annotation",
        "Segmentation": "tab5_segmentation",
        "Batch Correction": "tab6_batch_correction",
        "Phenotyping": "tab7_phenotyping",
        "Spatial Analysis": "tab8_analysis",
        "Visualization": "tab9_visualization",
    }
    return mapping.get(affected)


def values_equal(v1, v2) -> bool:
    """Compare custom types, lists, dictionaries, and DataFrames for equality."""
    import pandas as pd
    if isinstance(v1, pd.DataFrame) and isinstance(v2, pd.DataFrame):
        return v1.equals(v2)
    if isinstance(v1, pd.DataFrame) or isinstance(v2, pd.DataFrame):
        return False
    # A number is the same setting whether JSON gave it back as 40 or 40.0.
    # Comparing reprs marked those as changed on every startup. Booleans are
    # excluded so True never compares equal to 1.
    if (
        isinstance(v1, (int, float))
        and isinstance(v2, (int, float))
        and not isinstance(v1, bool)
        and not isinstance(v2, bool)
    ):
        try:
            return float(v1) == float(v2)
        except Exception:
            return False

    if isinstance(v1, dict) and isinstance(v2, dict):
        if len(v1) != len(v2):
            return False
        # JSON has no integer keys. A map saved as {1: "B"} comes back as
        # {"1": "B"}, so a literal key lookup reports every round-tripped map as
        # changed on each startup -- which is why som_phenotype_map logged
        # "dict with 45 keys -> dict with 45 keys" and forced a needless re-save.
        normalized = {str(key): value for key, value in v2.items()}
        for key, value in v1.items():
            key_text = str(key)
            if key_text not in normalized or not values_equal(value, normalized[key_text]):
                return False
        return True
    if isinstance(v1, list) and isinstance(v2, list):
        if len(v1) != len(v2):
            return False
        return all(values_equal(x, y) for x, y in zip(v1, v2))
    
    if v1 is None and v2 is None:
        return True
    if v1 is None or v2 is None:
        return False
        
    s1, s2 = str(v1), str(v2)
    if s1 == s2:
        return True
        
    # Check if both are absolute or relative directory/file paths
    if isinstance(v1, str) and isinstance(v2, str):
        v1_norm = v1.replace("\\", "/").rstrip("/")
        v2_norm = v2.replace("\\", "/").rstrip("/")
        if v1_norm == v2_norm:
            return True
        
    return False


def safe_set_session_state(key: str, value: any) -> None:
    """Safely set a key in st.session_state, avoiding StreamlitAPIException if the widget is already active."""
    if key in st.session_state:
        if values_equal(st.session_state[key], value):
            return
    try:
        st.session_state[key] = value
    except Exception as e:
        log_persistence(f"Skipped setting session_state['{key}'] = {value} due to widget constraint: {e}")


def get_all_system_defaults(project_root_str: str) -> dict:
    """Retrieve default parameters and directory configurations for all workflow steps."""
    p_root = Path(project_root_str)
    
    # Try to use active project's raw directories if loaded to prevent reverting to defaults
    codex_dir = None
    he_dir = None
    if "active_project" in st.session_state and st.session_state["active_project"]:
        config = st.session_state["active_project"]
        codex_dir = config.get("codex_raw_dir")
        he_dir = config.get("he_raw_dir")
        
    if not codex_dir:
        codex_dir = str(p_root / "data/codex")
    if not he_dir:
        he_dir = str(p_root / "data/he")
        
    # Import directory mappings
    from spacemap_gui.tabs.tab0_project import get_default_directory_mappings
    dirs = get_default_directory_mappings(str(p_root), codex_dir, he_dir)
    
    defaults = {}
    defaults.update(dirs)
    
    # General metadata
    defaults.update({
        "proj_mpp_codex": 0.377,
        "proj_mpp_he": 0.22,
        "proj_use_gpu": True,
        "proj_gpu_id": 0,
        "proj_num_workers": 4,
        "proj_pi": "",
        "proj_institution": "",
        "proj_description": "",
        "proj_unified_csv_filename": "",
    })
    
    # Tab specific configurations
    defaults.update({
        # Tab 1: Registration
        "batch_auto_apply_pair_settings": True,
        "multi_n_extra_codex_rounds": 1,
        "multi_include_he": True,
        "multi_intermediate_subdir": "intermediate_images",
        "multi_last_suggested_name": "",
        "multi_overwrite": False,
        "pair_configs": {},
        "active_pair_key": "",
        
        # Tab 2: Background Subtraction
        "bg_channels": [],
        "bg_colors": [],
        "bg_meta": [],
        "bg_mpp": 0.5083,
        "tab2_create_log": True,
        
        # Tab 3: ROI Extraction
        "roi_channels": [],
        "roi_visible": [],
        "tab3_show_overlay": True,
        "tab3_roi_labels": {},
        "tab3_canvas_objects": {},
        "tab3_deleted_rois": [],
        "canvas_reset_count": 0,
        "tab3_draw_active": False,
        
        # Tab 4: Segmentation
        "tab4_model_type": "cyto",
        "tab4_custom_model": str(p_root / "data/seg_model/CP_20241014_094617"),
        "tab4_diameter": 0,
        "tab4_flow": 0.8,
        "tab4_tile_size": 2048,
        "tab4_tile_overlap": 0.1,
        "tab4_cellpose_internal_tile_size": 224,
        "tab4_cellpose_batch_size": 1,
        "tab4_adaptive_gpu_memory": True,
        "tab4_min_cellpose_internal_tile_size": 128,
        "tab4_min_tissue_fraction": 0.002,
        "tab4_tissue_crop_padding": 64,
        "tab4_gpu": True,
        "tab4_use_mask": False,
        "tab4_save_geojson": True,
        "tab4_save_npy": False,
        "tab4_save_png": False,
        "batch_custom_model": str(p_root / "data/seg_model/CP_20241014_094617"),
        "tab4_create_log": True,
        
        # Tab 4: Annotation
        "annot_input_dirs": [str(p_root / "data/annotations")] * 3,
        "annot_ta_tissue_config": {},
        "annot_ta_artifact_config": {},
        "annot_ta_cluster_config": {},
        "annot_ta_pool_config": {},
        "annot_ta_cluster_markers": [],
        "annot_ta_artifact_markers": [],
        "annot_ta_domain_map": {},
        "annot_ta_domain_names": [],
        "annot_ta_domain_colors": {},
        "annot_ta_excluded_clusters": [],
        "annot_ta_pool_rules": [],
        "annot_ta_zone_rules": [],
        "annot_ta_save_figures": True,
        
        # Tab 6: Batch Correction
        "bc_config": get_default_bc_config(),
        "bc_input_mode": "Directory of CSV files",
        "bc_unified_csv_path": str(p_root / "outputs/batch_corrected/batch_corrected_data.csv"),
        "bc_unified_batch_col": "batch_id",
        "bc_selected_markers": [],
        "bc_vis_file": "",
        "bc_files_df": [],
        "bc_metadata_cols": [],
        
        # Tab 7, 8 & 9 Global Directories and Files
        "global_spatial_dir": str(p_root / "outputs/batch_corrected"),
        "global_visualization_dir": str(p_root / "outputs/batch_corrected"),
        "global_pheno_file": "batch_corrected_data.csv",
        "global_spatial_file": "batch_corrected_data.csv",
        "global_visualization_file": "batch_corrected_data.csv",
        
        # Tab 9: Actual Tissue Visualizations
        "tab9_actual_tissue_col": "None",
        "tab9_actual_batch_value": "",
        "tab9_actual_image_path": "",
        "tab9_actual_tissue_value": "All",
        "tab9_actual_cell_seg_file": "",
        "tab9_actual_overlay_files": [],
        "tab9_actual_enable_cell_filter": False,
        "tab9_actual_cell_filter_col": "None",
        "tab9_actual_cell_filter_op": "=",
        "tab9_actual_cell_filter_val": "",
        "tab9_actual_preview_max_dim": 1400,
        "tab9_actual_crop_to_cells": True,
        "tab9_actual_crop_padding": 150.0,
        "tab9_actual_pixel_size_um": 0.5083,
        "tab9_actual_crop_mode": "Rectangular context",
        "tab9_actual_roi_only_source": "Auto",
        "tab9_actual_load_cells": True,
        "tab9_actual_cell_seg_source": "GeoJSON boundaries file",
        "tab9_actual_cell_match_dist": 30,
        "tab9_actual_voronoi_max_dist": 40.0,
        "tab9_actual_voronoi_crop_annotations": True,
        "tab9_actual_cell_color_mode": "Single Color",
        "tab9_actual_cell_single_color": "#ffffff",
        "tab9_actual_cell_phenotype_col": "None",
        "tab9_actual_cell_phenotype_palette": "Set1",
        "tab9_actual_cell_continuous_col": "None",
        "tab9_actual_cell_continuous_cmap": "viridis",
        "tab9_actual_cell_lim_mode": "Min/Max",
        "tab9_actual_cell_lim_min": 0.0,
        "tab9_actual_cell_lim_max": 1.0,
        "tab9_actual_cell_compare_col1": "None",
        "tab9_actual_cell_compare_col2": "None",
        "tab9_actual_cell_compare_match_color": "#00ff00",
        "tab9_actual_cell_compare_mismatch_color": "#ff0000",
        "tab9_actual_cell_fill_alpha": 0.25,
        "tab9_actual_cell_line_width": 1,
        "tab9_actual_cell_outline_only": False,
        "tab9_actual_cell_outline_color_mode": "Match Cell Color",
        "tab9_actual_cell_outline_color": "#ffffff",
        "tab9_actual_visible_overlays": [],
        "tab9_actual_show_annotation_overlays": True,
        "tab9_actual_overlay_style": "transparent",
        "tab9_actual_overlay_alpha": 0.25,
        "tab9_actual_overlay_line_width": 2,
        "tab9_actual_mask_source": [],
        "tab9_actual_outside_mode": "dim",
        "tab9_actual_dim_strength": 0.8,
        "tab9_actual_outside_color": "#000000",
        "tab9_actual_output_dir": "",
        "tab9_actual_layer_1": "Markers",
        "tab9_actual_show_layer_1": True,
        "tab9_actual_layer_1_opacity": 1.0,
        "tab9_actual_layer_2": "Tissue Annotations",
        "tab9_actual_show_layer_2": True,
        "tab9_actual_layer_2_opacity": 1.0,
        "tab9_actual_layer_3": "Cell Segmentations",
        "tab9_actual_show_layer_3": True,
        "tab9_actual_layer_3_opacity": 1.0,
        "tab9_actual_show_markers": True,
        "tab9_actual_show_cells": True,
        "tab9_actual_markers_opacity": 1.0,
        "tab9_actual_background_color": "#ffffff",
        "tab9_actual_show_marker_legend": True,
        "tab9_actual_show_annotation_legend": True,
        "tab9_actual_show_phenotype_legend": True,
        "tab9_actual_marker_legend_position": "auto",
        "tab9_actual_annotation_legend_position": "auto",
        "tab9_actual_phenotype_legend_position": "below image",
        "tab9_actual_show_scale_bar": True,
        "tab9_actual_auto_scale_bar": True,
        "tab9_actual_scale_bar_um": 20.0,
        "tab9_actual_scale_bar_position": "auto",
        "tab9_actual_output_dpi": 300,
        "tab9_actual_adornment_background": True,
        "tab9_actual_save_pdf": True,
        "tab9_actual_save_svg": True,
        "tab9_actual_marker_legend_colored_labels": False,
    })
    
    return defaults


def format_val_concise(val) -> str:
    """Format a value to be concise, truncating long structures, lists, and dicts."""
    import pandas as pd
    if isinstance(val, pd.DataFrame):
        return f"DataFrame ({len(val)} rows)"
    if isinstance(val, dict):
        if val.get("__type__") == "DataFrame" and "data" in val:
            return f"DataFrame ({len(val['data'])} rows)"
        return f"dict with {len(val)} keys"
    if isinstance(val, list):
        return f"list with {len(val)} items"
    
    val_str = str(val)
    if len(val_str) > 150:
        return val_str[:147] + "..."
    return val_str


def _to_json_compatible(val):
    from pathlib import PurePath
    if isinstance(val, PurePath):
        return str(val)
    if isinstance(val, dict):
        return {k: _to_json_compatible(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_json_compatible(x) for x in val]
    if isinstance(val, tuple):
        return tuple(_to_json_compatible(x) for x in val)
    return val


def contains_binary_payload(value, _depth: int = 0) -> bool:
    """True when a value holds raw bytes anywhere (directly or nested)."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if _depth > 4:
        return False
    if isinstance(value, dict):
        return any(contains_binary_payload(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(contains_binary_payload(v, _depth + 1) for v in value)
    return False


def looks_like_serialized_bytes(value) -> bool:
    """True for values that are a stringified bytes repr from an older save."""

    if isinstance(value, str):
        return value.startswith(("b'", 'b"')) and "\\x" in value[:64]
    if isinstance(value, dict):
        return any(looks_like_serialized_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(looks_like_serialized_bytes(v) for v in value)
    return False


def is_transient_key(key: str) -> bool:
    """Check if a session state key is transient/cached/widget-owned and should not be serialized."""
    if not isinstance(key, str):
        return True
        
    # A leading underscore is the project-wide convention for private runtime
    # state: computed results, render caches, progress payloads, signatures.
    # These are rebuilt whenever the step is re-run and must never be written to
    # the project file. This is the structural guard: a key like
    # ``_tab8_fs_results`` holds whole per-cell DataFrames, and serializing one
    # expands it to a JSON object per row -- which is how a 292 KB config file
    # once grew past 5 GB.
    if key.startswith("_"):
        return True

    # Drawable-canvas scratch state. These hold a base64 PNG of what is on the
    # canvas plus the fabric.js scene graph, so a few ROIs can add megabytes of
    # image data to a project file that should only record settings. The drawing
    # itself is saved by the step that owns it, not by the widget's live state.
    _lowered = key.lower()
    if "canvas" in _lowered or "init_drawing" in _lowered:
        return True

    lk = key.lower()
    
    # 1. Exact matches for known massive/transient cache data frames and flags.
    # Keep this list focused: a user-facing control such as
    # ``tab9_actual_load_cells`` is a real reproducibility parameter, not a
    # load event, and must remain serializable.
    if key in [
        "tab9_viz_shared_df", "tab9_viz_unified_cache_df", "tab9_viz_shared_df_path", 
        "tab9_viz_shared_df_mtime", "tab9_viz_unified_cache_key", 
        "global_visualization_file_empty", "global_spatial_file_empty", "global_pheno_file_empty",
        "viz_file_matches", "_project_just_hydrated",
        "bc_aggrid_key_nonce", "bc_common_columns", "bc_common_markers",
        "bc_first_file_columns", "bc_union_columns", "bc_files_with_filter_col",
        "bc_filter_summary", "bc_last_calculated_config", "bc_restored_file_metadata",
        "bc_setup_complete", "bc_process_all_checkbox", "bc_vis_load_requested",
        "bc_vis_load_requested_path",
        "bc_marker_multiselect",
        "bc_marker_multiselect_form",
        "bc_unified_batch_col_select",
        # Lazy-loaded subtab state is deliberately session-only.  A project
        # may retain method parameters and result files, but it must not
        # decide which heavy UI modules are imported after a fresh app
        # session.  This keeps every phenotyping subtab behind its Load
        # button on restart while preserving the loaded state during normal
        # navigation in the same Streamlit session.
        "tab7_loaded_subtabs",
        "_tab7_heatmap_session_token",
        "resolve_normalized_plot_migration_v1",
        # One-shot "Start Fresh" actions must never be restored on a rerun;
        # restoring them would reset a newly assigned phenotype map back to
        # Unassigned immediately after the assignment is saved.
        "som_fresh_assignment_mode",
        "leiden_fresh_assignment_mode",
        "vortex_fresh_assignment_mode",
        "subpheno_fresh_assignment_mode",
        "resolve_add_criteria_col", "resolve_add_threshold_row", "resolve_add_threshold_row_auto",
        "tools_mr_merge",
        # Manual Reference card payloads are session-only. Persisting these restores
        # old labels/dropdown choices instead of initializing from the unified CSV.
        "tools_mr_vv_seen_keys",
        "tools_mr_vv_fetch_more",
        "tools_mr_vv_last_file",
        "tools_edit_csv_build_derived",
        "tools_edit_csv_add_filter",
        "umap_add_filter",
        "viz_heatmap_add_filter",
        "tools_edit_csv_save_unified",
        "tools_edit_csv_save_new",
        # Shared marker-panel editor state is staged UI state. The saved
        # ``tab7_marker_workspace`` dictionary is the authoritative record.
        "tab7_shared_panel_name",
        "tab7_shared_panel_editor",
        "tab7_shared_panel_editor_name",
        "tab7_shared_panel_shortcut",
    ]:
        return True
        
    if "container" in lk:
        return True

    # Annotation-tab buttons. A button press is a one-shot event, not a setting,
    # and Streamlit refuses to create a widget whose key already holds a value
    # assigned from code. A restored button key therefore does not merely add
    # noise to the project file: it stops the whole tab rendering. The worst case
    # is a button that has since been removed, because the key outlives the
    # widget that would explain it. The suffixes below are the action keys; the
    # settings keys beside them (annot_ta_s2_markers, annot_ta_s1_codex_markers,
    # the form and layer keys) deliberately do not match.
    if lk.startswith("annot_ta_"):
        if "_preset_" in lk:
            return True
        if lk.endswith((
            "_run", "_run_batch", "_less", "_more", "_export", "_btn",
            "_markers_all", "_markers_codex", "_markers_save", "_markers_dapi",
            "_save_channels", "_add_domain", "_add_presets", "_add_pool",
            "_load_legacy", "_load_s3",
        )):
            return True

    # Rendered figures and result payloads are session-only caches. They are not
    # reproducibility settings, and serializing them writes the repr of raw PNG
    # bytes into the project JSON, which later fails to render as an image.
    if lk.endswith(("_bytes", "_results", "_visualizations", "_run_details")):
        return True

    # Computed artifacts that dominate the project file. Each is written from a
    # single place immediately after being produced, and is regenerated by
    # re-running the step that made it -- so storing it buys nothing while
    # costing a full serialize plus a full backup copy on every save.
    #   tab2_bgdet_mask    ~25 MB, a background mask; re-run background detection
    #   f1_runtime_bundle  ~16 MB, validation results; re-run the validation
    # The settings that produce both are ordinary keys and remain persisted, so
    # a reloaded project still knows how the artifact was configured.
    if key in ("tab2_bgdet_mask", "f1_runtime_bundle"):
        return True

    # Assisted-assignment working data: suggestion tables, learned marker patterns
    # and scoring results are all recomputed from the settings, and a DataFrame
    # written to JSON comes back as a list of records. The settings that produce
    # them (chosen reference column, marker list, match mode, edited patterns)
    # are ordinary keys and stay persisted.
    if lk.endswith((
        "_reco_table",
        "_reco_learned",
        "_reco_seedstats",
        "_eval_result",
        "_pattern_table",
        "_reco_editor",
    )):
        return True

    # 2. Button/download keys are events, never configuration. Literal keys are
    # matched exactly; f-string keys are matched on their literal prefix.
    if key in _discover_action_widget_keys():
        return True
    if key.startswith(_discover_action_widget_prefixes()):
        return True
    if any(infix in key for infix in _discover_action_widget_infixes()):
        return True
    if key.endswith(_discover_action_widget_suffixes()):
        return True

    # Dynamic action keys (usually f-strings containing a file name) follow
    # these explicit prefixes/suffixes.  Do not use a broad substring match:
    # controls containing words like "load", "select", or "apply" can be
    # legitimate pipeline settings.
    if lk.startswith(("btn_", "tab4_btn_")):
        return True
    if lk == "viz_f1_save_outputs":
        return True
    if lk.startswith(("viz_f1_download_", "viz_heatmap_download_")):
        return True
    # Marker Analysis builds its PNG/PDF/SVG keys inside a helper as
    # ``f"{key_prefix}_{fmt}"``. The AST scanner cannot recover the runtime
    # ``key_prefix`` and deliberately rejects the generic ``_png`` suffix
    # because real settings also use it. Match this narrowly scoped family so
    # old click-state values are pruned from both project JSON and session state
    # before Streamlit recreates the download buttons.
    if lk.startswith("tab9_marker_") and "_download_" in lk:
        return True
    if lk.endswith(("_btn", "_button", "_download")):
        return True
    # Tab 8 ROI-explorer layer controls. Their keys are built as
    # ``_sk(prefix, f"add_{btn_type}")``, so the literal source scan above
    # cannot see them: it only recognises keys written as f-strings inline in
    # the ``st.button`` call. Restoring one of these puts a value in
    # session_state for a key that ``st.button`` later claims, which Streamlit
    # rejects with StreamlitValueAssignmentNotAllowedError.
    if lk.endswith(("_add_markers", "_add_annotations", "_add_cells", "_add_scatter")):
        return True
    # Layer move/delete buttons: ``{prefix}_{up|dn|del}_{layer_id}``.
    _tail = lk.rsplit("_", 1)
    if len(_tail) == 2 and _tail[1].isdigit() and _tail[0].endswith(("_up", "_dn", "_del")):
        return True
    if lk.endswith("_marker_value_source_widget"):
        return True
    if lk.endswith("_unified_export_checkbox") or "_unified_export_checkbox_" in lk:
        return True
    if lk.endswith("_unified_export_required"):
        return True
    if lk.endswith("_save_export_column_settings") or lk.endswith("_save_export_column_settings_btn"):
        return True
    if lk.endswith("_marker_panel_draft"):
        return True
    # Heatmap click payloads and counters are browser/session bookkeeping, not
    # reproducible method settings. Phenotype maps themselves remain persisted.
    if lk.endswith((
        "_heatmap_component_reset_counter",
        "_heatmap_component_event_id",
        "_heatmap_click_selector",
    )):
        return True

    # 3. Key ends with standard Streamlit widget suffixes or dynamic endings
    if lk.endswith("_in"):
        return True
        
    # Check for annot_in_{digit}
    if lk.startswith("annot_in_") and lk[9:].isdigit():
        return True
        
    # 3. Exclude only known dynamic coordinate/action bookkeeping.  Do not
    # exclude every key containing ``cell`` or ``roi``: Tab 9 has real
    # reproducibility settings such as ``tab9_actual_cell_phenotype_col`` and
    # ``tab9_actual_roi_only_source``.
    if lk.startswith("tools_mr_vv_assign_") or re.match(r"^cell_-?\d+_", lk):
        return True
    if "_assign_" in lk:
        return True

    # 4. Check for cache/editor payloads and dynamic UI bookkeeping.  Named
    # controls ending in ``_widget`` are intentionally *not* excluded: many
    # tabs use those keys for actual method parameters and copy them into the
    # canonical state key when a form is submitted.
    transient_tokens = [
        "data_editor", "editor", "_grouped_files", "_discovered",
        "_cache", "cached_", "prepared_", "unique_vals", "_temp", "uploader",
        "select_thumb", "width_table"
    ]
    if any(token in lk for token in transient_tokens):
        return True
        
    # Exclude any keys ending with a digit suffix if they belong to dynamic tables or groups
    # e.g., tab8_fs_live_checked_width_table_31
    if "width_table" in lk or "checked" in lk:
        return True
        
    return False


# Ceiling on a single serialized configuration value. Generous for any real
# setting (a long path list is a few KB) and far below anything data-shaped.
_MAX_CONFIG_VALUE_BYTES = 256 * 1024


def update_stored_config_key(config: dict, key: str, value) -> None:
    """Update a specific configuration parameter in the nested JSON configuration structure."""
    from spacemap_gui.tabs.tab0_project import DIRECTORY_KEYS
    import pandas as pd

    # Explicitly exclude transient/cached items and large data tables
    if is_transient_key(key):
        return

    # Never serialize binary payloads (rendered figures, downloads): JSON turns
    # them into a bytes repr string that breaks st.image when restored.
    if contains_binary_payload(value):
        return

    # Serialize DataFrame if needed, skipping massive ones
    if isinstance(value, pd.DataFrame):
        if len(value) > 100:
            return  # Discard large data tables from configuration serialization
        serialized_val = {"__type__": "DataFrame", "data": value.to_dict(orient="records")}
    else:
        serialized_val = _to_json_compatible(value)

    # The project file records input configuration -- directories, file choices
    # and the values the user typed. It is not a data store. The DataFrame check
    # above only sees a frame passed directly; a frame nested inside a dict or a
    # list slips past it and is expanded to one JSON object per row, which is how
    # a per-cell table once turned this file into gigabytes. Cap the serialized
    # size of every entry so no key, present or future, can do that again.
    try:
        encoded_size = len(json.dumps(serialized_val, default=str))
    except Exception:
        return  # not JSON-serializable: not a setting
    if encoded_size > _MAX_CONFIG_VALUE_BYTES:
        log_persistence(
            f"Skipped '{key}': {encoded_size / 1e6:.1f} MB exceeds the "
            f"{_MAX_CONFIG_VALUE_BYTES // 1024} KB per-setting limit. The project file "
            "stores configuration, not data."
        )
        return
        
    if key.startswith("proj_dir_"):
        key = key[9:]
        
    if key in DIRECTORY_KEYS:
        config.setdefault("directory_mappings", {})[key] = serialized_val
    elif key == "proj_mpp_codex": config["mpp_codex"] = serialized_val
    elif key == "proj_mpp_he": config["mpp_he"] = serialized_val
    elif key == "proj_use_gpu": config["use_gpu"] = serialized_val
    elif key == "proj_gpu_id": config["gpu_id"] = serialized_val
    elif key == "proj_num_workers": config["num_workers"] = serialized_val
    elif key == "proj_pi": config["pi"] = serialized_val
    elif key == "proj_institution": config["institution"] = serialized_val
    elif key == "proj_description": config["description"] = serialized_val
    elif key == "proj_unified_csv_filename": config["unified_csv_filename"] = serialized_val
    else:
        tab_name = get_tab_name_from_key(key)
        if tab_name:
            config.setdefault("tab_configurations", {}).setdefault(tab_name, {})[key] = serialized_val


LEGACY_THRESHOLD_DIRECTORY_KEYS = (
    # Prefer the batch destination because it is the producer RESOLVE consumes.
    "tab2_bgdet_batch_threshold_dir",
    "tab2_bgdet_threshold_dir",
    "resolve_threshold_dir",
)


def ensure_threshold_directory_mapping(config: dict, project_root: str) -> tuple[str, bool]:
    """Ensure the shared Tab 2/RESOLVE threshold directory is in project mappings.

    Older project files stored separate Tab 2 and RESOLVE widget keys in tab
    configurations.  Migrate the first non-empty legacy value into the single
    canonical ``directory_mappings.threshold_dir`` key.  The canonical value,
    once present, always wins on later loads.
    """
    directory_mappings = config.setdefault("directory_mappings", {})
    current = str(directory_mappings.get("threshold_dir") or "").strip()

    if not current:
        for legacy_key in LEGACY_THRESHOLD_DIRECTORY_KEYS:
            legacy_value = str(directory_mappings.get(legacy_key) or "").strip()
            if legacy_value:
                current = legacy_value
                break

    if not current:
        tab_configurations = config.get("tab_configurations", {})
        if isinstance(tab_configurations, dict):
            for legacy_key in LEGACY_THRESHOLD_DIRECTORY_KEYS:
                for settings in tab_configurations.values():
                    if not isinstance(settings, dict):
                        continue
                    legacy_value = str(settings.get(legacy_key) or "").strip()
                    if legacy_value:
                        current = legacy_value
                        break
                if current:
                    break

    if not current:
        data_root = Path(project_root) / "data"
        preferred = data_root / "thresholds"
        legacy_candidates = (data_root / "threshold", data_root / "boundries")

        def _contains_threshold_csv(path: Path) -> bool:
            try:
                return path.is_dir() and any(path.glob("*.csv"))
            except OSError:
                return False

        # New projects always use data/thresholds. For old projects with no
        # saved widget value, retain the populated legacy folder RESOLVE used
        # before this project-level setting existed.
        current = str(preferred)
        if not _contains_threshold_csv(preferred):
            for candidate in legacy_candidates:
                if _contains_threshold_csv(candidate):
                    current = str(candidate)
                    break

    changed = directory_mappings.get("threshold_dir") != current
    directory_mappings["threshold_dir"] = current

    # Finish the migration instead of retaining several stale sources of
    # truth in the project JSON.  Consumers now use ``threshold_dir`` only.
    for legacy_key in LEGACY_THRESHOLD_DIRECTORY_KEYS:
        if legacy_key in directory_mappings:
            del directory_mappings[legacy_key]
            changed = True
    tab_configurations = config.get("tab_configurations", {})
    if isinstance(tab_configurations, dict):
        for settings in tab_configurations.values():
            if not isinstance(settings, dict):
                continue
            for legacy_key in LEGACY_THRESHOLD_DIRECTORY_KEYS:
                if legacy_key in settings:
                    del settings[legacy_key]
                    changed = True

    # These aliases used to be widget/session keys. Leaving them alive during
    # a hot project reload lets the generic parameter-discovery pass serialize
    # them straight back into tab_configurations after the JSON migration.
    for legacy_key in LEGACY_THRESHOLD_DIRECTORY_KEYS:
        if legacy_key in st.session_state:
            try:
                del st.session_state[legacy_key]
            except Exception:
                pass

    return current, changed


def is_logging_switch(key) -> bool:
    """Run-log and process-log checkboxes, which always start switched on."""
    return isinstance(key, str) and key.endswith(("_create_log", "_process_log_enabled"))


def sync_project_state() -> None:
    """Centralized State Sync Manager to keep st.session_state aligned with the active project's config.
    Ensures that loaded project settings override defaults and are restored when Streamlit deletes them.
    Tracks changes and appends them to the audit log.
    """
    if "active_project" not in st.session_state or not st.session_state["active_project"]:
        return
        
    config = st.session_state["active_project"]
    project_root = st.session_state.get("active_project_root")
    if not project_root:
        return
        
    from spacemap_gui.tabs.tab0_project import DIRECTORY_KEYS
    
    just_hydrated = st.session_state.get("_project_just_hydrated", False)
    defaults = get_all_system_defaults(project_root)
    directory_mappings = config.setdefault("directory_mappings", {})
    tab_configs = config.setdefault("tab_configurations", {})
    _, threshold_mapping_migrated = ensure_threshold_directory_mapping(config, project_root)
    audit_log = config.setdefault("audit_log", [])
    pruned_transient_keys = []
    for tab_name, settings in tab_configs.items():
        if not isinstance(settings, dict):
            continue
        for key in list(settings.keys()):
            # Older saves stringified rendered figures into the config; those values
            # can no longer be displayed, so drop them wherever they appear.
            if is_transient_key(key) or looks_like_serialized_bytes(settings.get(key)):
                del settings[key]
                pruned_transient_keys.append(f"{tab_name}.{key}")
                # A previously saved action key may already have been restored
                # into session state; leaving it there keeps the widget broken
                # for the rest of the session.
                if key in st.session_state:
                    try:
                        del st.session_state[key]
                    except Exception:
                        pass
    
    # If project was just hydrated/loaded, force restore all state from directory_mappings
    # and ignore Streamlit's restoration of pre-hydration widget state values.
    if just_hydrated:
        for key in DIRECTORY_KEYS:
            stored_dir_val = directory_mappings.get(key, defaults.get(key))
            proj_key = f"proj_dir_{key}"
            safe_set_session_state(key, stored_dir_val)
            safe_set_session_state(proj_key, stored_dir_val)

        
        stored = {}
        stored["proj_mpp_codex"] = config.get("mpp_codex", defaults["proj_mpp_codex"])
        stored["proj_mpp_he"] = config.get("mpp_he", defaults["proj_mpp_he"])
        stored["proj_use_gpu"] = config.get("use_gpu", defaults["proj_use_gpu"])
        stored["proj_gpu_id"] = config.get("gpu_id", defaults["proj_gpu_id"])
        stored["proj_num_workers"] = config.get("num_workers", defaults["proj_num_workers"])
        stored["proj_pi"] = config.get("pi", defaults["proj_pi"])
        stored["proj_institution"] = config.get("institution", defaults["proj_institution"])
        stored["proj_description"] = config.get("description", defaults["proj_description"])
        unified_filename = config.get("unified_csv_filename", defaults["proj_unified_csv_filename"])
        stored["proj_unified_csv_filename"] = unified_filename
        # The unified CSV *filename* is not a directory key, so it is restored here
        # from the project config. It must go through `stored` rather than being set
        # directly: the defaults sweep below would otherwise overwrite it, which is
        # how a project could open on the previous project's file.
        for file_key in ("global_pheno_file", "global_spatial_file", "global_visualization_file"):
            stored[file_key] = unified_filename
        
        for tab_name, settings in tab_configs.items():
            if isinstance(settings, dict):
                for k, v in settings.items():
                    if is_transient_key(k):
                        continue
                    stored[k] = v
                    
        for k, default_val in defaults.items():
            if k not in stored:
                stored[k] = default_val
                
        for k, stored_val in stored.items():
            if k in DIRECTORY_KEYS or k.startswith("proj_dir_"):
                continue
            if is_transient_key(k):
                continue
            import pandas as pd
            if isinstance(stored_val, dict) and stored_val.get("__type__") == "DataFrame":
                st.session_state[k] = pd.DataFrame(stored_val["data"])
            elif k.endswith("_phenotype_map") and isinstance(stored_val, dict):
                # JSON converts dictionary keys to strings. Restore numeric
                # cluster IDs so heatmap lookups match the dataframe IDs.
                restored_map = {}
                for cluster_id, phenotype in stored_val.items():
                    normalized_id = cluster_id
                    if isinstance(cluster_id, str):
                        text_id = cluster_id.strip()
                        if text_id.lstrip("-").isdigit():
                            normalized_id = int(text_id)
                    restored_map[normalized_id] = phenotype
                st.session_state[k] = restored_map
            elif k == "bc_files_df" and isinstance(stored_val, list):
                st.session_state[k] = pd.DataFrame(stored_val) if stored_val else pd.DataFrame(columns=["Process", "Filename"])
            else:
                st.session_state[k] = stored_val

        # Logging switches start on in every session so every run is recorded.
        # A user can still switch one off, but only for the current session.
        for k in list(st.session_state.keys()):
            if is_logging_switch(k):
                st.session_state[k] = True

        st.session_state["_project_just_hydrated"] = False
        log_persistence("Forced initial configuration state sync (hydration) and bypassed change detection.")
        if pruned_transient_keys or threshold_mapping_migrated:
            config["updated_at"] = datetime.now().isoformat()
            save_project_config(project_root, config)
            if pruned_transient_keys:
                log_persistence(
                    f"Removed {len(pruned_transient_keys)} transient widget key(s) from project configuration."
                )
            if threshold_mapping_migrated:
                log_persistence(
                    "Migrated the shared threshold directory into project directory mappings."
                )
        return
    
    # Flatten stored config into a single dict for easy lookup
    stored = {}
    
    # General project metadata
    stored["proj_mpp_codex"] = config.get("mpp_codex", defaults["proj_mpp_codex"])
    stored["proj_mpp_he"] = config.get("mpp_he", defaults["proj_mpp_he"])
    stored["proj_use_gpu"] = config.get("use_gpu", defaults["proj_use_gpu"])
    stored["proj_gpu_id"] = config.get("gpu_id", defaults["proj_gpu_id"])
    stored["proj_num_workers"] = config.get("num_workers", defaults["proj_num_workers"])
    stored["proj_pi"] = config.get("pi", defaults["proj_pi"])
    stored["proj_institution"] = config.get("institution", defaults["proj_institution"])
    stored["proj_description"] = config.get("description", defaults["proj_description"])
    stored["proj_unified_csv_filename"] = config.get("unified_csv_filename", defaults["proj_unified_csv_filename"])
    
    for tab_name, settings in tab_configs.items():
        if isinstance(settings, dict):
            for k, v in settings.items():
                if is_transient_key(k):
                    continue
                stored[k] = v
                
    # Fill any other keys from defaults if not present anywhere
    for k, default_val in defaults.items():
        if k in DIRECTORY_KEYS or k.startswith("proj_dir_"):
            continue
        if k not in stored:
            stored[k] = default_val
            update_stored_config_key(config, k, default_val)
            
    # Discover and add any tab-specific parameters currently in session state
    for k in list(st.session_state.keys()):
        if k in DIRECTORY_KEYS or k.startswith("proj_dir_"):
            continue
        if is_transient_key(k):
            continue
        # Check if it belongs to a tab and is not transient/cached
        tab_name = get_tab_name_from_key(k)
        if tab_name and k not in stored:
            stored[k] = None
            
    # Group definitions of conceptually identical directory keys to sync across all tabs
    DIRECTORY_GROUPS = [
        {"batch_codex_dir", "annot_image_dir"},
        {"tab1_batch_output_dir", "registration_output_dir", "tab2_input_dir_reg"},
        {"tab2_output_dir", "bg_subtraction_output_dir", "tab3_input_dir", "tab3_multi_input_dir"},
        {
            "tab3_output_dir", "tab3_multi_output_dir", "roi_output_dir",
            "tab4_input_dir", "batch_input_dir", "vis_img_dir", "viz_roi_dir",
            "annot_ta_input_dir",
            "tab9_actual_image_dir", "tab9_marker_single_image_dir", "tab9_marker_batch_image_dir"
        },
        {"tab3_geojson_dir", "tab3_multi_geojson_dir", "tab4_seg_mask_save_dir", "batch_geojson_dir", "tab9_actual_roi_geojson_dir"},
        {
            "tab4_output_dir", "batch_output_dir", "cell_seg_dir", "viz_seg_dir",
            "tab9_actual_overlay_geojson_dir", "tab9_marker_single_geojson_dir", "tab9_marker_batch_geojson_dir"
        },
        {"annot_output_dir", "zone_input_dir"},
        {"annot_ta_output_dir", "annot_ta_pool_input_dir"},
        {"annot_ta_pool_output_dir", "tab4_annot_geojson_dir", "cell_zone_dir"},
        {"zone_output_dir", "cell_zone_dir"},
        {"cell_out_dir", "bc_input_dir"},
        {"bc_output_dir", "scimap_bc_dir", "global_pheno_dir", "spatial_input_dir", "vis_csv_dir", "global_spatial_dir", "global_visualization_dir"},
        {"spatial_output_dir", "tab9_actual_output_dir", "tab9_marker_single_output_dir", "tab9_marker_batch_output_dir"}
    ]

    # Detect any directory key that has been modified by user widget interaction
    from spacemap_gui.tabs.tab0_project import DIRECTORY_KEYS
    changed_group_val = None
    changed_group_keys = None
    
    # Restrict change detection to directories of the active rendering tab
    active_tab = st.session_state.get("_active_tab_session") or st.session_state.get("_currently_rendering_tab")
    tab_mapping = {
        "tab0": "Project Configurations",
        "tab1": "Registration",
        "tab2": "Background Subtraction",
        "tab3": "ROI Extraction",
        "tab4": "Annotation",
        "tab5": "Segmentation",
        "tab6": "Batch Correction",
        "tab7": "Phenotyping",
        "tab8": "Spatial Analysis",
        "tab9": "Visualization"
    }
    active_tab_affected = tab_mapping.get(active_tab) if active_tab else None
    
    for key in DIRECTORY_KEYS:
        # Skip checking keys that do not belong to the currently rendering tab
        affected = get_tab_affected_from_key(key)
        if active_tab_affected and affected != active_tab_affected and affected != "Directories":
            continue
            
        proj_key = f"proj_dir_{key}"
        stored_val = directory_mappings.get(key, defaults.get(key))
        
        # Check global tab key
        if key in st.session_state:
            val = st.session_state[key]
            if not values_equal(val, stored_val):
                changed_group_val = val
                for group in DIRECTORY_GROUPS:
                    if key in group:
                        changed_group_keys = group
                        break
                if changed_group_keys:
                    break
                    
        # Check project tab key
        if proj_key in st.session_state:
            val = st.session_state[proj_key]
            if not values_equal(val, stored_val):
                changed_group_val = val
                for group in DIRECTORY_GROUPS:
                    if key in group:
                        changed_group_keys = group
                        break
                if changed_group_keys:
                    break
                    
    changes_detected = []
    log_messages = []
    
    # Propagate the changed value to all keys in the same group
    if changed_group_keys and changed_group_val is not None:
        for k in changed_group_keys:
            old_val = directory_mappings.get(k)
            if not values_equal(old_val, changed_group_val):
                directory_mappings[k] = changed_group_val
                safe_set_session_state(k, changed_group_val)
                safe_set_session_state(f"proj_dir_{k}", changed_group_val)
                changes_detected.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tab_affected": get_tab_affected_from_key(k),
                    "parameter_changed": k,
                    "old_value": str(old_val),
                    "new_value": str(changed_group_val)
                })
                log_messages.append(f"Propagated directory change to {k}: {old_val} -> {changed_group_val}")
                
        # Special case: If tab4_output_dir was changed, automatically update derived subfolders tab4_csv_dir & tab4_geojson_dir
        if "tab4_output_dir" in changed_group_keys:
            for csv_key in ["tab4_csv_dir", "tab4_geojson_dir"]:
                sub = "cellpose/features" if csv_key == "tab4_csv_dir" else "cellpose/boundaries"
                new_sub_val = str(Path(changed_group_val) / sub)
                old_sub_val = directory_mappings.get(csv_key)
                if not values_equal(old_sub_val, new_sub_val):
                    directory_mappings[csv_key] = new_sub_val
                    safe_set_session_state(csv_key, new_sub_val)
                    safe_set_session_state(f"proj_dir_{csv_key}", new_sub_val)
                    changes_detected.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tab_affected": get_tab_affected_from_key(csv_key),
                        "parameter_changed": csv_key,
                        "old_value": str(old_sub_val),
                        "new_value": str(new_sub_val)
                    })
                    log_messages.append(f"Auto-updated derived subfolder {csv_key} to: {new_sub_val}")

    # Bidirectional Directory Sync Logic
    for key in DIRECTORY_KEYS:
        # Skip checking/restoring keys that do not belong to the currently rendering tab
        affected = get_tab_affected_from_key(key)
        if active_tab_affected and affected != active_tab_affected and affected != "Directories":
            continue
            
        stored_dir_val = directory_mappings.get(key, defaults.get(key))
        proj_key = f"proj_dir_{key}"
        
        global_in_state = (key in st.session_state)
        proj_in_state = (proj_key in st.session_state)
        
        global_changed = False
        if global_in_state:
            val_global = st.session_state[key]
            if not values_equal(val_global, stored_dir_val):
                global_changed = True
                
        proj_changed = False
        if proj_in_state:
            val_proj = st.session_state[proj_key]
            if not values_equal(val_proj, stored_dir_val):
                proj_changed = True
                
        if global_changed:
            new_val = st.session_state[key]
            directory_mappings[key] = new_val
            safe_set_session_state(proj_key, new_val)
            changes_detected.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tab_affected": get_tab_affected_from_key(key),
                "parameter_changed": key,
                "old_value": str(stored_dir_val),
                "new_value": str(new_val)
            })
            log_messages.append(f"Directory updated from global tab: {key} = {new_val}")
        elif proj_changed:
            new_val = st.session_state[proj_key]
            directory_mappings[key] = new_val
            safe_set_session_state(key, new_val)
            changes_detected.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tab_affected": get_tab_affected_from_key(key),
                "parameter_changed": key,
                "old_value": str(stored_dir_val),
                "new_value": str(new_val)
            })
            log_messages.append(f"Directory updated from project tab: {key} = {new_val}")
        else:
            # Restore state if missing
            if not global_in_state:
                safe_set_session_state(key, stored_dir_val)
            if not proj_in_state:
                safe_set_session_state(proj_key, stored_dir_val)
                
    # Sync regular parameters
    for k, stored_val in stored.items():
        if k in DIRECTORY_KEYS or k.startswith("proj_dir_"):
            continue
            
        if is_transient_key(k):
            continue
                
        if k not in st.session_state:
            import pandas as pd
            if isinstance(stored_val, dict) and stored_val.get("__type__") == "DataFrame":
                st.session_state[k] = pd.DataFrame(stored_val["data"])
            elif k == "bc_files_df" and isinstance(stored_val, list):
                st.session_state[k] = pd.DataFrame(stored_val) if stored_val else pd.DataFrame(columns=["Process", "Filename"])
            else:
                st.session_state[k] = stored_val
        else:
            current_val = st.session_state[k]
            import pandas as pd
            
            is_changed = False
            old_display_val = stored_val
            new_display_val = current_val
            
            if isinstance(current_val, pd.DataFrame):
                current_records = current_val.to_dict(orient="records")
                stored_records = stored_val.get("data", []) if isinstance(stored_val, dict) and stored_val.get("__type__") == "DataFrame" else (stored_val if isinstance(stored_val, list) else [])

                def _comparable(records):
                    """NaN != NaN, so a table holding a single blank cell compares
                    unequal to itself on every rerun and triggers an endless
                    save loop. Normalise missing values before comparing."""
                    normalised = []
                    for record in records:
                        normalised.append(
                            {
                                key: (None if isinstance(value, float) and value != value else value)
                                for key, value in record.items()
                            }
                        )
                    return normalised

                if _comparable(current_records) != _comparable(stored_records):
                    is_changed = True
                    old_display_val = f"DataFrame ({len(stored_records)} rows)"
                    new_display_val = f"DataFrame ({len(current_records)} rows)"
            else:
                stored_clean = stored_val
                if isinstance(stored_val, dict) and stored_val.get("__type__") == "DataFrame":
                    stored_clean = stored_val.get("data", [])
                # values_equal understands dicts, lists and numeric types;
                # str() comparison called a value changed whenever its repr
                # differed, which JSON round-tripping guarantees for any map
                # with non-string keys.
                if not values_equal(current_val, stored_clean):
                    is_changed = True
                    
            if is_changed:
                update_stored_config_key(config, k, current_val)
                old_lbl = format_val_concise(old_display_val) if old_display_val is not None else "None"
                new_lbl = format_val_concise(new_display_val)
                changes_detected.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tab_affected": get_tab_affected_from_key(k),
                    "parameter_changed": k,
                    "old_value": old_lbl,
                    "new_value": new_lbl
                })
                if old_display_val is not None:
                    log_messages.append(f"Parameter {k} changed: {old_lbl} -> {new_lbl}")
                else:
                    log_messages.append(f"Parameter {k} initialized to: {new_lbl}")

    # Process and print buffered log messages
    if log_messages:
        dir_propagations = [m for m in log_messages if "Propagated directory change" in m or "Auto-updated derived subfolder" in m]
        dir_updates = [m for m in log_messages if "Directory updated" in m]
        param_changes = [m for m in log_messages if "Parameter " in m]
        
        total_logs = len(log_messages)
        if total_logs > 3:
            summary_parts = []
            if dir_propagations:
                summary_parts.append(f"propagated {len(dir_propagations)} directories")
            if dir_updates:
                summary_parts.append(f"updated {len(dir_updates)} tab paths")
            if param_changes:
                summary_parts.append(f"synced {len(param_changes)} parameters")
            summary_str = ", ".join(summary_parts)
            log_persistence(f"Batch state sync completed: {summary_str}.")
        else:
            for msg in log_messages:
                log_persistence(msg)

    if changes_detected or pruned_transient_keys or threshold_mapping_migrated:
        if pruned_transient_keys:
            changes_detected.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tab_affected": "Project Configurations",
                "parameter_changed": "transient_widget_state",
                "old_value": f"{len(pruned_transient_keys)} saved transient key(s)",
                "new_value": "removed from project configuration",
            })
        audit_log.extend(changes_detected)
        if len(audit_log) > 500:
            audit_log = audit_log[-500:]
        config["audit_log"] = audit_log
        save_project_config(project_root, config)


def hydrate_project_state(config: dict, project_root: str) -> None:
    """Hydrates the session state from the loaded project config, clearing transient states."""
    # Clear existing keys to avoid state bleed
    for key in list(st.session_state.keys()):
        if key in ["global_accent_color", "global_border_radius", "global_tab_font_size", 
                   "global_content_font_size", "global_tab_align", "global_cmap", 
                   "global_thumb_scale", "global_theme_mode", "active_project", 
                   "active_project_root", "active_project_name", "_spacemap_loaded_top_tabs",
                   "_force_sidebar_proj_sync", "_global_sidebar_proj"]:
            # Note: the global_pheno/spatial/visualization dir+file keys used to be
            # preserved here. That is exactly what made a newly loaded project open
            # on the previous project's unified CSV, so they are now cleared and
            # restored from the incoming project config below.
            continue
        del st.session_state[key]
        
    # The root must be set before restoring, because interface-state routing and
    # ${PROJECT_ROOT} expansion both key off it.
    st.session_state["active_project_root"] = project_root
    config = restore_portable_state(config, project_root)

    st.session_state["active_project"] = config
    st.session_state["active_project_name"] = config.get("project_name")
    st.session_state["_project_just_hydrated"] = True
    
    sync_project_state()
