from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Persist Tab 8 global path choices in interface_state.json via the shared
# atomic accessor in spacemap_gui.persistence (same file used by Tab 7 / Tab 9
# / app.py, now safe against concurrent writes).
from spacemap_gui.persistence import (
    load_interface_state as _load_state,
    update_interface_state as _update_state,
)

_TAB8_SOURCE_DIR = Path(__file__).resolve().parent

# Session key used to track currently active Tab 8 CSV for restore logic
_TAB8_ACTIVE_CSV_KEY = "tab8_active_csv_path"

_TRANSIENT_DYNAMIC_KEY_PREFIXES: tuple[str, ...] = (
    "tab8_nb_write_csv_",
)


@lru_cache(maxsize=1)
def _discover_literal_action_keys() -> frozenset[str]:
    """Collect explicit button-like widget keys used across Tab 8."""
    button_keys: set[str] = set()
    button_attrs = {"button", "form_submit_button", "download_button"}

    for path in _TAB8_SOURCE_DIR.glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr not in button_attrs:
                continue
            for keyword in node.keywords:
                if keyword.arg != "key":
                    continue
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    button_keys.add(keyword.value.value)

    return frozenset(button_keys)


def _is_transient_tab8_key(key: str) -> bool:
    """Return True for Tab 8 action/widget keys that must never be persisted."""
    if key in _discover_literal_action_keys():
        return True
    if key in {"tab8_loaded_subtabs", _TAB8_ACTIVE_CSV_KEY}:
        return True
    if key in {"tab8_triad_loaded_for_csv"}:
        return True
    if key.startswith("tab8_load_"):
        return True
    if any(key.startswith(prefix) for prefix in _TRANSIENT_DYNAMIC_KEY_PREFIXES):
        return True
    if "_cached_" in key or "prepared_" in key or "unique_vals" in key:
        return True
    if "__" in key and not key.startswith("tab8_fs_filter_values__"):
        return True
    if "::" in key:
        return True
    if key.endswith(("_btn", "_addbtn", "_run")):
        return True
    if "_run_" in key:
        return True
    if key.endswith("_widget") or "_widget_" in key:
        return True
    # ROI Explorer add-layer buttons (st.button — cannot be pre-set via session_state)
    if key.endswith(("_add_markers", "_add_annotations", "_add_cells", "_add_scatter")):
        return True
    # Layer-card move / delete buttons: {prefix}_{up|dn|del}_{integer_lid}
    _parts = key.rsplit("_", 1)
    if len(_parts) == 2 and _parts[1].isdigit() and _parts[0].endswith(("_up", "_dn", "_del")):
        return True
    # Filter-rule remove buttons: {key}_r{integer}_del
    if key.endswith("_del") and "_r" in key:
        _before_del = key[:-4]  # strip trailing "_del"
        _r_parts = _before_del.rsplit("_r", 1)
        if len(_r_parts) == 2 and _r_parts[1].isdigit():
            return True
    # Rendered figures and result payloads are session-only caches, never settings.
    if key.endswith(("_bytes", "_results", "_visualizations", "_run_details")):
        return True
    return False


def _is_persistable_value(value: Any, _depth: int = 0) -> bool:
    """False for payloads that must never reach the project JSON.

    Raw bytes become a useless repr string once serialized (and then crash
    ``st.image`` when restored), and data frames belong in CSV outputs.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return False
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return False
    if _depth > 4:
        return True
    if isinstance(value, dict):
        return all(_is_persistable_value(v, _depth + 1) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return all(_is_persistable_value(v, _depth + 1) for v in value)
    if isinstance(value, str):
        # A stringified bytes payload from an older save.
        return not (value.startswith(("b'", 'b"')) and "\\x" in value[:64])
    return True


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable data."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _get_tab8_csv_configs() -> dict[str, Any]:
    state = _load_state()
    configs = state.get("tab8_csv_configs", {})
    return configs if isinstance(configs, dict) else {}


def _set_tab8_csv_configs(configs: dict[str, Any]) -> None:
    _update_state({"tab8_csv_configs": configs})


def _tab8_config_keys() -> list[str]:
    """Return all session keys that represent Tab 8 user configuration."""
    keys: list[str] = []
    for key in st.session_state.keys():
        if not isinstance(key, str):
            continue
        if not key.startswith("tab8_"):
            continue
        if _is_transient_tab8_key(key):
            continue
        # Skip non-UI payload/cache-like keys
        if key.endswith("_df") or key.endswith("_cache"):
            continue
        if key.endswith("_download") or "native_png" in key or "download_native" in key:
            continue
        if key in {"tab8_triad_instances_df"}:
            continue
        val = st.session_state.get(key)
        if isinstance(val, pd.DataFrame):
            continue
        if not _is_persistable_value(val):
            continue
        keys.append(key)
    return keys


def load_tab8_config_for_csv(csv_path: str | None, key_prefixes: list[str] | None = None) -> None:
    """Load stored Tab 8 config for a given CSV path into session_state.

    Parameters
    ----------
    csv_path : str | None
        Active CSV path used as config namespace.
    key_prefixes : list[str] | None
        Optional key-prefix allowlist (e.g., ["tab8_triad_"]).
        When provided, only matching keys are restored.
    """
    if not csv_path:
        return
    
    # 1. First, load from the CSV cache (interface_state.json) as a fallback
    configs = _get_tab8_csv_configs()
    cfg = configs.get(str(csv_path), {})
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            key_str = str(key)
            if _is_transient_tab8_key(key_str):
                continue
            # Feature-screening settings are project-scoped, not CSV-scoped: they
            # describe the analysis method, not the file. Restoring them from a
            # per-CSV store would reset the method every time the unified CSV is
            # switched. Their own project settings cache owns these keys.
            if key_str.startswith("tab8_fs_"):
                continue
            if not _is_persistable_value(value):
                continue
            if key_prefixes and not any(key_str.startswith(str(prefix)) for prefix in key_prefixes):
                continue
            try:
                st.session_state[key_str] = value
            except Exception:
                continue

    # 2. Next, load from the active project JSON configuration (tab8_analysis), which takes absolute priority!
    active_project = st.session_state.get("active_project")
    if active_project and isinstance(active_project, dict):
        project_settings = active_project.get("tab_configurations", {}).get("tab8_analysis", {})
        if isinstance(project_settings, dict):
            for key, value in project_settings.items():
                key_str = str(key)
                if _is_transient_tab8_key(key_str):
                    continue
                if not _is_persistable_value(value):
                    continue
                if key_prefixes and not any(key_str.startswith(str(prefix)) for prefix in key_prefixes):
                    continue
                try:
                    st.session_state[key_str] = value
                except Exception:
                    continue


def save_tab8_config_for_csv(csv_path: str | None) -> None:
    """Persist current Tab 8 config (all tab8_* keys) for the active CSV."""
    if not csv_path:
        return
    configs = _get_tab8_csv_configs()
    existing_cfg = configs.get(str(csv_path), {})
    cfg: dict[str, Any] = {}
    if isinstance(existing_cfg, dict):
        for key, value in existing_cfg.items():
            key_str = str(key)
            if _is_transient_tab8_key(key_str):
                continue
            # Drop payloads written by older versions instead of carrying them forward.
            if not _is_persistable_value(value):
                continue
            cfg[key_str] = value
    # Safeguard: do not overwrite the configuration if it has become blank/empty in session state.
    new_patient_col = st.session_state.get("tab8_fs_patient_col", "")
    old_patient_col = cfg.get("tab8_fs_patient_col", "")
    if old_patient_col and not new_patient_col:
        return
        
    for key in _tab8_config_keys():
        cfg[key] = _to_jsonable(st.session_state.get(key))
    configs[str(csv_path)] = cfg
    _set_tab8_csv_configs(configs)


def get_cached_global_spatial_dir(default_value: str) -> str:
    return _load_state().get("global_spatial_dir", default_value)


def set_cached_global_spatial_dir(value: str) -> None:
    _update_state({"global_spatial_dir": value})


def get_cached_global_spatial_file(default_value: str | None = None) -> str | None:
    return _load_state().get("global_spatial_file", default_value)


def set_cached_global_spatial_file(value: str | None) -> None:
    _update_state({"global_spatial_file": value})


def queue_tab8_csv_reload(csv_path: str | None) -> None:
    """Queue a reload of a specific Tab 8 CSV on the next rerun."""
    if not csv_path:
        return

    target_path = Path(csv_path)
    st.session_state["_tab8_pending_global_spatial_dir"] = str(target_path.parent)
    st.session_state["_tab8_pending_global_spatial_file"] = target_path.name
    try:
        load_unified_csv.clear()
    except Exception:
        pass


def load_unified_csv(csv_path, columns: list[str] | None = None):
    """Cache-safe wrapper: re-reads when the file on disk changes.

    Passing *columns* loads only those columns, which on the multi-gigabyte
    batch-corrected tables is the difference between minutes and seconds.
    Callers that genuinely need every column may omit it; the read still goes
    through the Parquet fast copy, which is far quicker than parsing the CSV.
    """
    from spacemap_gui.ui_utils import file_signature
    key = tuple(columns) if columns else None
    return _load_unified_csv__impl(csv_path, file_signature(csv_path), key)


@st.cache_data(show_spinner=False)
def _load_unified_columns__impl(
    csv_path: str, signature, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Cache a narrow column selection, which is cheap to store."""
    from spacemap import dataio

    return dataio.load(csv_path, columns=list(columns))


def _load_unified_csv__impl(
    csv_path: str, signature, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    # Streamlit's cache pickles what it stores and unpickles it on every hit.
    # That is fine for a narrow column selection and very expensive for the
    # whole table -- twenty to thirty seconds per hit on a 15 GB frame -- so a
    # full read bypasses the cache and is held by the caller instead.
    if columns:
        return _load_unified_columns__impl(csv_path, signature, columns)

    from spacemap import dataio

    return dataio.load(csv_path)


# The callers below invoke ``load_unified_csv.clear()`` to force a re-read.
# ``load_unified_csv`` is a plain function, so that call raised AttributeError
# and was swallowed by the surrounding ``except``: the cache was never actually
# cleared and a reload button did nothing. Forwarding to the cached
# implementation makes those calls work as they read.
load_unified_csv.clear = _load_unified_columns__impl.clear


def _safe_unique(series: pd.Series) -> list[Any]:
    vals = series.dropna().unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return vals


def render_optional_filter_controls(df: pd.DataFrame, key_prefix: str = "tab8") -> pd.DataFrame:
    """Backward-compatible no-op filter helper (filtering is now global)."""
    return df


def render_global_input_configuration(project_root: Path) -> tuple[str | None, pd.DataFrame | None]:
    """Render Tab 8 global input section like Tab 7 and return loaded dataframe."""
    from spacemap_gui.ui_utils import directory_input_with_browse

    col_title, col_save = st.columns([3, 1])
    with col_title:
        st.markdown("## Step 8: Spatial Analysis")
        st.markdown("Perform spatial analysis using a unified single-cell CSV file.")
    with col_save:
        pass

    pending_dir = st.session_state.pop("_tab8_pending_global_spatial_dir", None)
    pending_file = st.session_state.pop("_tab8_pending_global_spatial_file", None)
    if pending_dir:
        st.session_state["global_spatial_dir"] = str(pending_dir)
        st.session_state["global_spatial_dir_cached"] = str(pending_dir)
        set_cached_global_spatial_dir(str(pending_dir))
    if pending_file:
        st.session_state["global_spatial_file"] = str(pending_file)
        st.session_state["global_spatial_file_cached"] = str(pending_file)
        set_cached_global_spatial_file(str(pending_file))

    bc_output_dir = st.session_state.get("bc_output_dir", str(project_root / "outputs" / "batch_corrected"))
    cached_dir = get_cached_global_spatial_dir(bc_output_dir)
    cached_file = get_cached_global_spatial_file(None)

    if "global_spatial_dir" not in st.session_state:
        st.session_state.global_spatial_dir = cached_dir

    # Track user override vs Tab6 sync
    if "global_spatial_dir_user_override" not in st.session_state:
        st.session_state["global_spatial_dir_user_override"] = False
    if "global_spatial_dir_last_bc" not in st.session_state:
        st.session_state["global_spatial_dir_last_bc"] = bc_output_dir

    # If Tab6 output dir changed and user has not overridden, sync to new bc_output_dir
    if bc_output_dir != st.session_state["global_spatial_dir_last_bc"] and not st.session_state["global_spatial_dir_user_override"]:
        st.session_state.global_spatial_dir = bc_output_dir
        set_cached_global_spatial_dir(bc_output_dir)
    st.session_state["global_spatial_dir_last_bc"] = bc_output_dir

    with st.expander("Global Input Configuration (Applied to all subtabs)", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            global_input_dir = directory_input_with_browse(
                label="Input Directory",
                text_key="global_spatial_dir",
                default_value=st.session_state.get("global_spatial_dir", cached_dir),
                browser_key_prefix="tree_tab8_global_spatial_dir",
                help_text=(
                    "Directory holding the unified per-cell CSV. Remembered across all subtabs and "
                    "saved with the project (as directory_mappings.global_spatial_dir, stored "
                    "relative to the project root so the project stays portable). Defaults to the "
                    "Tab 6 output directory until you override it."
                ),
            )
            # Update cache and override flag when user changes
            if global_input_dir != st.session_state.get("global_spatial_dir_cached"):
                set_cached_global_spatial_dir(global_input_dir)
                st.session_state["global_spatial_dir_cached"] = global_input_dir
            st.session_state["global_spatial_dir_user_override"] = (global_input_dir != bc_output_dir)

        input_files: list[str] = []
        if global_input_dir and Path(global_input_dir).exists():
            csv_paths = sorted([f for f in Path(global_input_dir).iterdir() if f.suffix.lower() == ".csv"])
            input_files = [p.name for p in csv_paths]

        with col2:
            preferred_file = st.session_state.get("global_spatial_file", cached_file)
            default_idx = input_files.index(preferred_file) if preferred_file in input_files else 0

            if input_files:
                selected_file = st.selectbox(
                    "Select Unified CSV File",
                    options=input_files,
                    index=default_idx,
                    key="global_spatial_file",
                )
                if selected_file != st.session_state.get("global_spatial_file_cached"):
                    set_cached_global_spatial_file(selected_file)
                    # A project may hold several unified CSVs; record the latest so
                    # reloading the project reopens this one.
                    from spacemap_gui.persistence import record_latest_unified_csv

                    record_latest_unified_csv(selected_file)
                    st.session_state["global_spatial_file_cached"] = selected_file
            else:
                st.selectbox("Select Unified CSV File", options=[""], index=0, key="global_spatial_file_empty")
                st.warning("No CSV files found in the selected directory.")
                return None, None

        full_path = str(Path(global_input_dir) / selected_file)
        st.caption(f"Active: {full_path}")

        # Offer the Parquet fast copy for this table. Large CSVs must be parsed
        # from the beginning to reach any column, so a columnar copy is what
        # makes a few-column screen load in seconds rather than minutes.
        from spacemap_gui.ui_utils import render_fast_copy_controls

        render_fast_copy_controls(full_path, key_prefix="tab8")

        # Save Project Button
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Save Project", key="tab8_global_save_project_btn", width='stretch'):
            from spacemap_gui.persistence import save_active_project_state
            save_active_project_state("tab8_analysis")
            st.success("Project settings saved to configuration file.")

    # Restore per-CSV Tab 8 settings when switching to a different unified CSV.
    active_csv = st.session_state.get(_TAB8_ACTIVE_CSV_KEY)
    if full_path and active_csv != full_path:
        load_tab8_config_for_csv(full_path)
        st.session_state[_TAB8_ACTIVE_CSV_KEY] = full_path

    if not Path(full_path).exists():
        st.error("Selected CSV file does not exist.")
        return None, None

    try:
        df = load_unified_csv(full_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return full_path, None

    with st.expander("Global Row Filters (Applied to all subtabs)", expanded=True):
        st.caption(
            "Optional: add one or more filter columns and keep only selected values. "
            "If column 'tissue' exists and value 1 is present, it is auto-selected by default."
        )

        filter_candidates = list(df.columns)
        has_tissue = "tissue" in df.columns
        tissue_has_one = False
        if has_tissue:
            tissue_vals_str = [str(v) for v in df["tissue"].dropna().unique().tolist()]
            tissue_has_one = "1" in tissue_vals_str

        default_filter_cols = ["tissue"] if (has_tissue and tissue_has_one) else []
        filter_cols_key = "tab8_global_filter_cols"
        if filter_cols_key not in st.session_state:
            st.session_state[filter_cols_key] = default_filter_cols
        else:
            existing = st.session_state.get(filter_cols_key, [])
            if not isinstance(existing, list):
                existing = []
            st.session_state[filter_cols_key] = [c for c in existing if c in filter_candidates]

        selected_filter_cols = st.multiselect(
            "Filter columns",
            options=filter_candidates,
            key=filter_cols_key,
            help="Choose zero, one, or many columns. Leave empty for no filtering.",
        )

        global_filters: list[dict[str, Any]] = []
        for idx, col in enumerate(selected_filter_cols):
            col_vals = _safe_unique(df[col])
            col_vals_str = [str(v) for v in col_vals]
            default_vals: list[str] = ["1"] if (col == "tissue" and "1" in col_vals_str) else []
            filter_vals_key = f"tab8_global_filter_vals_{idx}_{col}"
            if filter_vals_key not in st.session_state:
                st.session_state[filter_vals_key] = default_vals
            else:
                existing_vals = st.session_state.get(filter_vals_key, [])
                if not isinstance(existing_vals, list):
                    existing_vals = []
                st.session_state[filter_vals_key] = [v for v in existing_vals if v in col_vals_str]

            selected_vals = st.multiselect(
                f"{col}: values to keep",
                options=col_vals_str,
                key=filter_vals_key,
                help="Selected values are kept. Leave empty to skip this column filter.",
            )
            global_filters.append({"column": col, "values": selected_vals})

        filtered_df = df.copy()
        for flt in global_filters:
            values = [str(v) for v in flt.get("values", []) if str(v) != ""]
            if not values:
                continue
            col = str(flt["column"])
            filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]

        st.caption(f"Global filtered rows: {len(filtered_df):,} / {len(df):,}")

    st.caption(f"Loaded rows: {len(df):,} | columns: {len(df.columns):,}")
    return full_path, filtered_df
