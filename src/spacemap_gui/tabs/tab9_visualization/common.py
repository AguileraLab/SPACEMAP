"""Shared helpers for Tab 9 visualization subtabs."""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except Exception:
    linkage = None
    leaves_list = None

from spacemap_gui.persistence import (
    is_transient_key as _is_transient_project_key,
    load_directory_settings,
    load_interface_state as _load_state,
    save_directory_settings,
    update_interface_state as _update_state,
)
from spacemap_gui.ui_utils import directory_input_with_browse

# Compatibility exports used by multiple Tab 9 subtabs.  Keep these names in
# this module even though their implementations live in the shared Tab 7
# phenotyping helpers.
from ..tab7_phenotyping.common import (  # noqa: F401
    _extract_markers_from_notebook,
    logger,
)

NOTEBOOK_NAME = "Step8_Comparison.ipynb"

# Persist Tab 9 state in interface_state.json via the shared atomic accessor
# in spacemap_gui.persistence (same file used by Tab 7 / Tab 8 / app.py).
_TAB9_SHARED_DF_KEY = "tab9_viz_shared_df"
_TAB9_SHARED_PATH_KEY = "tab9_viz_shared_df_path"
_TAB9_SHARED_MTIME_KEY = "tab9_viz_shared_df_mtime"
_TAB9_UNIFIED_CACHE_KEY = "tab9_viz_unified_cache_key"
_TAB9_UNIFIED_CACHE_DF = "tab9_viz_unified_cache_df"
_TAB9_ACTIVE_CSV_KEY = "tab9_active_csv_path"
_TAB9_CSV_CONFIG_PREFIXES = ("tab9_actual_", "tab9_vt_", "viz_f1_")
_TAB9_LEGACY_PREFIX_MAP = (
    ("tab8_actual_", "tab9_actual_"),
    ("tab8_vt_", "tab9_vt_"),
)


def timestamp() -> str:
    return time.strftime("%H:%M:%S")


def log_console(message: str) -> None:
    print(message, flush=True)


def log_terminal_event(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [Tab9 Visualization] {message}", flush=True)


def _is_transient_tab9_key(key: str) -> bool:
    if _is_transient_project_key(key):
        return True
    if key in {"tab9_loaded_subtabs", _TAB9_ACTIVE_CSV_KEY}:
        return True
    if key == "viz_f1_save_outputs":
        return True
    if key in {
        "tab9_viz_settings_source_dir",
        "tab9_viz_dirs_validated",
        _TAB9_SHARED_DF_KEY,
        _TAB9_SHARED_PATH_KEY,
        _TAB9_SHARED_MTIME_KEY,
        _TAB9_UNIFIED_CACHE_KEY,
        _TAB9_UNIFIED_CACHE_DF,
    }:
        return True
    if key.startswith("tab9_load_"):
        return True
    if "__" in key or "::" in key:
        return True
    if key.endswith("_btn") or key.endswith("_run"):
        return True
    if "_run_" in key or "_download_" in key:
        return True
    if key.endswith("_download"):
        return True
    return False


def _to_jsonable(value: Any) -> Any:
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


def _get_tab9_csv_configs() -> dict[str, Any]:
    state = _load_state()
    configs = state.get("tab9_csv_configs", {})
    return configs if isinstance(configs, dict) else {}


def _set_tab9_csv_configs(configs: dict[str, Any]) -> None:
    _update_state({"tab9_csv_configs": configs})


def _is_tab9_csv_config_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in _TAB9_CSV_CONFIG_PREFIXES)


def _tab9_config_keys() -> list[str]:
    keys: list[str] = []
    for key in st.session_state.keys():
        if not isinstance(key, str):
            continue
        if not _is_tab9_csv_config_key(key):
            continue
        if _is_transient_tab9_key(key):
            continue
        value = st.session_state.get(key)
        if isinstance(value, pd.DataFrame):
            continue
        keys.append(key)
    return keys


def _migrate_tab8_config_for_csv(csv_path: str | None) -> dict[str, Any]:
    if not csv_path:
        return {}
    state = _load_state()
    legacy_configs = state.get("tab8_csv_configs", {})
    if not isinstance(legacy_configs, dict):
        return {}

    legacy_cfg = legacy_configs.get(str(csv_path), {})
    if not isinstance(legacy_cfg, dict):
        return {}

    migrated: dict[str, Any] = {}
    for key, value in legacy_cfg.items():
        key_str = str(key)
        for old_prefix, new_prefix in _TAB9_LEGACY_PREFIX_MAP:
            if key_str.startswith(old_prefix):
                migrated[f"{new_prefix}{key_str[len(old_prefix):]}"] = value
                break

    if migrated:
        configs = _get_tab9_csv_configs()
        configs[str(csv_path)] = migrated
        _set_tab9_csv_configs(configs)
    return migrated


def load_tab9_config_for_csv(csv_path: str | None) -> None:
    if not csv_path:
        return
    configs = _get_tab9_csv_configs()
    cfg = configs.get(str(csv_path), {})
    if not isinstance(cfg, dict) or not cfg:
        cfg = _migrate_tab8_config_for_csv(csv_path)
    if not isinstance(cfg, dict):
        return

    for key, value in cfg.items():
        key_str = str(key)
        if _is_transient_tab9_key(key_str):
            continue
        if not _is_tab9_csv_config_key(key_str):
            continue
        try:
            st.session_state[key_str] = value
        except Exception:
            continue


def save_tab9_config_for_csv(csv_path: str | None) -> None:
    if not csv_path:
        return
    configs = _get_tab9_csv_configs()
    existing_cfg = configs.get(str(csv_path), {})
    cfg: dict[str, Any] = {}
    if isinstance(existing_cfg, dict):
        for key, value in existing_cfg.items():
            key_str = str(key)
            if _is_transient_tab9_key(key_str):
                continue
            if not _is_tab9_csv_config_key(key_str):
                continue
            cfg[key_str] = value

    for key in _tab9_config_keys():
        cfg[key] = _to_jsonable(st.session_state.get(key))

    configs[str(csv_path)] = cfg
    _set_tab9_csv_configs(configs)


def get_cached_global_visualization_dir(default_value: str) -> str:
    return _load_state().get("global_visualization_dir", default_value)


def set_cached_global_visualization_dir(value: str) -> None:
    _update_state({"global_visualization_dir": value})


def get_cached_global_visualization_file(default_value: str | None = None) -> str | None:
    return _load_state().get("global_visualization_file", default_value)


def set_cached_global_visualization_file(value: str | None) -> None:
    _update_state({"global_visualization_file": value})


def get_visualization_cache_path(project_root: Path) -> Path:
    cache_dir = project_root / "outputs" / "phenotyping" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "visualization_params.json"


def load_cached_params(project_root: Path) -> dict[str, Any]:
    cache_path = get_visualization_cache_path(project_root)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                return {
                    str(key): value
                    for key, value in data.items()
                    if not _is_transient_tab9_key(str(key))
                }
        except Exception:
            return {}
    return {}


def save_cached_params(project_root: Path, cached_params: dict[str, Any]) -> None:
    cache_path = get_visualization_cache_path(project_root)
    persistent_params = {
        str(key): value
        for key, value in cached_params.items()
        if not _is_transient_tab9_key(str(key))
    }
    cached_params.clear()
    cached_params.update(persistent_params)
    try:
        with open(cache_path, "w") as f:
            json.dump(persistent_params, f, indent=2)
    except Exception:
        pass


def sync_cached_params_from_session(cached_params: dict[str, Any]) -> None:
    for key in list(cached_params):
        if _is_transient_tab9_key(str(key)):
            cached_params.pop(key, None)

    for key, value in st.session_state.items():
        if not isinstance(key, str):
            continue
        if not (key.startswith("viz_") or key.startswith("cellviz_") or key.startswith("umap_")):
            continue
        if _is_transient_tab9_key(key):
            continue
        if isinstance(value, pd.DataFrame):
            continue
        cached_params[key] = value


def persist_cached_params(project_root: Path, cached_params: dict[str, Any], csv_path: str | None = None) -> None:
    sync_cached_params_from_session(cached_params)
    save_cached_params(project_root, cached_params)
    save_tab9_config_for_csv(csv_path)
    try:
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab9_visualization")
    except Exception:
        pass


def get_output_dir_default(project_root: Path, cached_params: dict[str, Any]) -> str:
    return cached_params.get(
        "viz_output_dir",
        str(project_root / "outputs" / "phenotyping" / "comparison"),
    )


def render_global_input_configuration(project_root: Path) -> tuple[str | None, str]:
    bc_output_dir = st.session_state.get("bc_output_dir", str(project_root / "outputs" / "batch_corrected"))
    cached_dir = get_cached_global_visualization_dir(bc_output_dir)
    cached_file = get_cached_global_visualization_file(None)

    if "global_visualization_dir" not in st.session_state:
        st.session_state["global_visualization_dir"] = cached_dir
    if "global_visualization_dir_user_override" not in st.session_state:
        st.session_state["global_visualization_dir_user_override"] = False
    if "global_visualization_dir_last_bc" not in st.session_state:
        st.session_state["global_visualization_dir_last_bc"] = bc_output_dir

    if (
        bc_output_dir != st.session_state["global_visualization_dir_last_bc"]
        and not st.session_state["global_visualization_dir_user_override"]
    ):
        st.session_state["global_visualization_dir"] = bc_output_dir
        set_cached_global_visualization_dir(bc_output_dir)
    st.session_state["global_visualization_dir_last_bc"] = bc_output_dir

    global_full_path: str | None = None
    with st.expander("Global Input Configuration (Applied to all subtabs)", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            global_input_dir = directory_input_with_browse(
                label="Input Directory (Batch Corrected)",
                text_key="global_visualization_dir",
                default_value=st.session_state.get("global_visualization_dir", cached_dir),
                browser_key_prefix="tree_tab9_global_visualization_dir",
                help_text="This directory is reused across all visualization subtabs.",
            )
            if global_input_dir != st.session_state.get("global_visualization_dir_cached"):
                set_cached_global_visualization_dir(global_input_dir)
                st.session_state["global_visualization_dir_cached"] = global_input_dir
            st.session_state["global_visualization_dir_user_override"] = (global_input_dir != bc_output_dir)

        with col2:
            input_files: list[str] = []
            if global_input_dir and os.path.exists(global_input_dir):
                input_files = sorted(
                    [f for f in os.listdir(global_input_dir) if f.lower().endswith(".csv")]
                )

            global_file_select: str | None = None
            if input_files:
                preferred_file = st.session_state.get("global_visualization_file", cached_file)
                default_idx = input_files.index(preferred_file) if preferred_file in input_files else 0
                global_file_select = st.selectbox(
                    "Select Unified CSV File",
                    input_files,
                    index=default_idx,
                    key="global_visualization_file",
                )
                if global_file_select != st.session_state.get("global_visualization_file_cached"):
                    set_cached_global_visualization_file(global_file_select)
                    st.session_state["global_visualization_file_cached"] = global_file_select
                    # A project may hold several unified CSVs; record the latest
                    # so reloading the project reopens this one.
                    from spacemap_gui.persistence import record_latest_unified_csv

                    record_latest_unified_csv(global_file_select)
            else:
                st.selectbox(
                    "Select Unified CSV File",
                    options=[""],
                    index=0,
                    key="global_visualization_file_empty",
                )
                st.warning("No CSV files found in the selected directory.")

        if global_file_select:
            global_full_path = os.path.join(global_input_dir, global_file_select)

        if global_full_path and os.path.exists(global_full_path):
            st.caption(f"Active: `{global_full_path}`")

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Save Project", key="tab9_global_save_project_btn", width='stretch'):
            from spacemap_gui.persistence import save_active_project_state
            save_active_project_state("tab9_visualization")
            st.success("Project settings saved to configuration file.")

    if global_full_path:
        active_csv = st.session_state.get(_TAB9_ACTIVE_CSV_KEY)
        if active_csv and active_csv != global_full_path:
            save_tab9_config_for_csv(active_csv)
        if active_csv != global_full_path:
            load_tab9_config_for_csv(global_full_path)
            st.session_state[_TAB9_ACTIVE_CSV_KEY] = global_full_path

    return global_full_path, st.session_state.get("global_visualization_dir", cached_dir)


def render_visual_validation_settings(project_root: Path, global_input_dir: str, available_columns: list[str] = None) -> dict[str, Any]:
    defaults = {
        "viz_roi_dir": str(project_root / "outputs" / "registration" / "roi"),
        "viz_seg_dir": str(project_root / "outputs" / "segmentation" / "cellpose"),
        "viz_ann_dir": str(project_root / "outputs" / "segmentation" / "cellpose"),
        "viz_col_batch": "batch_id",
        "viz_col_x": "x_centroid",
        "viz_col_y": "y_centroid",
        "viz_tile_size": 40,
        "viz_start_n_cells": 5,
    }

    stored = load_directory_settings(global_input_dir, section_key="tab9_viz") if global_input_dir else {}
    if st.session_state.get("tab9_viz_settings_source_dir") != global_input_dir:
        st.session_state["tab9_viz_roi_dir"] = stored.get("viz_roi_dir", defaults["viz_roi_dir"])
        st.session_state["tab9_viz_seg_dir"] = stored.get("viz_seg_dir", defaults["viz_seg_dir"])
        st.session_state["tab9_viz_ann_dir"] = stored.get("viz_ann_dir", defaults["viz_ann_dir"])
        st.session_state["tab9_viz_col_batch"] = stored.get("viz_col_batch", defaults["viz_col_batch"])
        st.session_state["tab9_viz_col_x"] = stored.get("viz_col_x", defaults["viz_col_x"])
        st.session_state["tab9_viz_col_y"] = stored.get("viz_col_y", defaults["viz_col_y"])
        st.session_state["tab9_viz_tile_size"] = int(stored.get("viz_tile_size", defaults["viz_tile_size"]))
        st.session_state["tab9_viz_start_n_cells"] = int(stored.get("viz_start_n_cells", defaults["viz_start_n_cells"]))
        st.session_state["tab9_viz_settings_source_dir"] = global_input_dir

    with st.expander("Visual Validation Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            roi_dir = directory_input_with_browse(
                label="ROI Images Directory (OME-TIFF)",
                text_key="tab9_viz_roi_dir",
                default_value=st.session_state.get("tab9_viz_roi_dir", defaults["viz_roi_dir"]),
                browser_key_prefix="tree_tab9_viz_roi_dir",
                start_dir_fallback=str(project_root),
                help_text="Directory containing ROI images such as .tif, .tiff, .ome.tif, or .ome.tiff.",
            )
        with col2:
            seg_dir = directory_input_with_browse(
                label="GeoJSON / Segmentation Directory",
                text_key="tab9_viz_seg_dir",
                default_value=st.session_state.get("tab9_viz_seg_dir", defaults["viz_seg_dir"]),
                browser_key_prefix="tree_tab9_viz_seg_dir",
                start_dir_fallback=str(project_root),
                help_text="Directory containing GeoJSON or JSON segmentation files used by validation and cell figures.",
            )
        with col3:
            ann_dir = directory_input_with_browse(
                label="Tissue Annotation Directory",
                text_key="tab9_viz_ann_dir",
                default_value=st.session_state.get("tab9_viz_ann_dir", defaults["viz_ann_dir"]),
                browser_key_prefix="tree_tab9_viz_ann_dir",
                start_dir_fallback=str(project_root),
                help_text="Directory containing tissue annotation GeoJSON files.",
            )

        col4, col5, col6 = st.columns(3)
        column_opts = [""] + available_columns if available_columns else [""]
        with col4:
            default_batch = st.session_state.get("tab9_viz_col_batch", defaults["viz_col_batch"])
            default_idx = column_opts.index(default_batch) if default_batch in column_opts else 0
            batch_col = st.selectbox("Batch Column", options=column_opts, index=default_idx, key="tab9_viz_col_batch")
        with col5:
            default_x = st.session_state.get("tab9_viz_col_x", defaults["viz_col_x"])
            default_idx = column_opts.index(default_x) if default_x in column_opts else 0
            x_col = st.selectbox("X Coord Column", options=column_opts, index=default_idx, key="tab9_viz_col_x")
        with col6:
            default_y = st.session_state.get("tab9_viz_col_y", defaults["viz_col_y"])
            default_idx = column_opts.index(default_y) if default_y in column_opts else 0
            y_col = st.selectbox("Y Coord Column", options=column_opts, index=default_idx, key="tab9_viz_col_y")

        st.markdown("##### Visualization Parameters")
        col7, col8 = st.columns(2)
        with col7:
            tile_size = st.slider(
                "Tile Size (px)",
                min_value=20,
                max_value=100,
                value=int(st.session_state.get("tab9_viz_tile_size", defaults["viz_tile_size"])),
                key="tab9_viz_tile_size",
            )
        with col8:
            start_n_cells = st.number_input(
                "Cells to Sample",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("tab9_viz_start_n_cells", defaults["viz_start_n_cells"])),
                key="tab9_viz_start_n_cells",
            )

        settings_to_save = {
            "viz_roi_dir": roi_dir,
            "viz_seg_dir": seg_dir,
            "viz_ann_dir": ann_dir,
            "viz_col_batch": batch_col,
            "viz_col_x": x_col,
            "viz_col_y": y_col,
            "viz_tile_size": int(tile_size),
            "viz_start_n_cells": int(start_n_cells),
        }
        if global_input_dir and settings_to_save != stored:
            save_directory_settings(global_input_dir, "tab9_viz", settings_to_save)

        if st.button("Scan & Link Files", type="primary", key="tab9_viz_scan_btn"):
            if not os.path.exists(roi_dir) or not os.path.exists(seg_dir):
                st.error("One or more directories do not exist.")
            else:
                try:
                    n_rois = len(
                        [
                            f for f in os.listdir(roi_dir)
                            if f.lower().endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff"))
                        ]
                    )
                    n_segs = len(
                        [f for f in os.listdir(seg_dir) if f.lower().endswith((".json", ".geojson"))]
                    )
                    st.success(
                        f"Found {n_rois} ROI images and {n_segs} segmentation files. Ready for visualization matching."
                    )
                    st.session_state["tab9_viz_dirs_validated"] = True
                except Exception as exc:
                    st.error(f"Error scanning directories: {exc}")

    return {
        "roi_dir": st.session_state.get("tab9_viz_roi_dir", defaults["viz_roi_dir"]),
        "seg_dir": st.session_state.get("tab9_viz_seg_dir", defaults["viz_seg_dir"]),
        "ann_dir": st.session_state.get("tab9_viz_ann_dir", defaults["viz_ann_dir"]),
        "batch_col": st.session_state.get("tab9_viz_col_batch", defaults["viz_col_batch"]),
        "x_col": st.session_state.get("tab9_viz_col_x", defaults["viz_col_x"]),
        "y_col": st.session_state.get("tab9_viz_col_y", defaults["viz_col_y"]),
        "tile_size": int(st.session_state.get("tab9_viz_tile_size", defaults["viz_tile_size"])),
        "start_n_cells": int(st.session_state.get("tab9_viz_start_n_cells", defaults["viz_start_n_cells"])),
    }


def get_csv_columns_fast(path: str) -> list[str]:
    """Column names only; reads no rows."""
    if not path or not os.path.exists(path):
        return []
    try:
        from spacemap import dataio

        return dataio.columns(path)
    except Exception:
        try:
            return list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            return []


def get_loaded_visualization_df(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    loaded_path = st.session_state.get(_TAB9_SHARED_PATH_KEY)
    loaded_mtime = st.session_state.get(_TAB9_SHARED_MTIME_KEY)
    df = st.session_state.get(_TAB9_SHARED_DF_KEY)
    if loaded_path != path or not isinstance(df, pd.DataFrame):
        return None
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return None
    if loaded_mtime != mtime:
        return None
    return df


def set_loaded_visualization_df(path: str, df: pd.DataFrame) -> None:
    st.session_state[_TAB9_SHARED_DF_KEY] = df
    st.session_state[_TAB9_SHARED_PATH_KEY] = path
    st.session_state[_TAB9_SHARED_MTIME_KEY] = os.path.getmtime(path)


def clear_loaded_visualization_df() -> None:
    st.session_state.pop(_TAB9_SHARED_DF_KEY, None)
    st.session_state.pop(_TAB9_SHARED_PATH_KEY, None)
    st.session_state.pop(_TAB9_SHARED_MTIME_KEY, None)
    st.session_state.pop(_TAB9_UNIFIED_CACHE_KEY, None)
    st.session_state.pop(_TAB9_UNIFIED_CACHE_DF, None)


@st.cache_data(show_spinner=False)
def _cached_read_columns(
    path: str, mtime: float, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Cache a *narrow* column selection.

    Streamlit's cache stores values pickled and unpickles them on every hit, so
    it only pays off for small results. Column subsets qualify; the whole table
    does not, which is why the full-frame path below bypasses it.
    """
    _ = mtime
    # Each column's type is decided from the whole column, never per chunk.
    # That matters for a sparsely populated label column -- a manual reference
    # phenotype present for ~1% of cells -- which chunked inference reads as
    # float (all-NaN) in most chunks and object in the few that hold labels,
    # producing a mixed-type warning and a dtype that depends on where the
    # populated rows happen to fall. Both paths below preserve that: Parquet
    # carries an explicit schema, and the CSV fallback reads whole columns.
    from spacemap import dataio

    return dataio.load(path, columns=list(columns) if columns else None)


def _get_unified_df(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the unified table, optionally only *columns*.

    Naming the columns a screen needs turns a multi-minute read of a 13 GB
    table into roughly a second, because the Parquet fast copy stores columns
    separately and skips the ones not asked for.

    A full-table read deliberately skips Streamlit's cache. That cache pickles
    what it stores and unpickles it on every hit, which on a 15 GB frame costs
    twenty to thirty seconds each time and gained nothing here: the caller
    already keeps the frame in ``session_state``, which stores the object
    itself.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Unified CSV file not found.")
    mtime = os.path.getmtime(path)
    if columns:
        return _cached_read_columns(path, mtime, tuple(columns))

    from spacemap import dataio

    return dataio.load(path)


def get_columns_df(path: str, columns: list[str]) -> pd.DataFrame:
    """Return only *columns* from the unified table.

    Reuses the shared in-memory frame when the user has loaded it and it
    already holds these columns, so nothing is read twice.  Otherwise reads
    just these columns, which on a 13 GB table is the difference between
    minutes and about a second.

    This is what lets a screen ask for what it needs at the moment it needs it,
    rather than every screen paying for all 133 columns up front.
    """
    wanted = [c for c in dict.fromkeys(columns) if c]
    if not wanted:
        return pd.DataFrame()
    shared = get_loaded_visualization_df(path)
    if isinstance(shared, pd.DataFrame) and set(wanted).issubset(shared.columns):
        return shared[wanted]
    available = set(get_csv_columns_fast(path))
    present = [c for c in wanted if c in available]
    if not present:
        return pd.DataFrame()
    return _get_unified_df(path, columns=present)


def get_cached_unified_df(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return the unified table, or only *columns* when they are named.

    Naming columns avoids reading the other 130-odd, which is what makes a
    cold start quick. The shared in-memory frame is still reused when it is
    already loaded and holds what is being asked for.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Unified CSV file not found.")

    if columns:
        return get_columns_df(path, list(columns))

    loaded_df = get_loaded_visualization_df(path)
    if isinstance(loaded_df, pd.DataFrame):
        return loaded_df

    mtime = os.path.getmtime(path)
    cache_key = f"tab9_viz_unified::{path}::{mtime}"
    if st.session_state.get(_TAB9_UNIFIED_CACHE_KEY) == cache_key:
        cached_df = st.session_state.get(_TAB9_UNIFIED_CACHE_DF)
        if isinstance(cached_df, pd.DataFrame):
            set_loaded_visualization_df(path, cached_df)
            return cached_df

    log_terminal_event(f"Loading unified CSV into memory: {os.path.basename(path)}")
    start = time.perf_counter()
    df = _get_unified_df(path)
    elapsed = time.perf_counter() - start
    log_terminal_event(f"Loaded {len(df):,} rows, {len(df.columns)} columns in {elapsed:.2f}s")

    st.session_state[_TAB9_UNIFIED_CACHE_KEY] = cache_key
    st.session_state[_TAB9_UNIFIED_CACHE_DF] = df
    set_loaded_visualization_df(path, df)
    return df


def render_unified_csv_loader(unified_csv_path: str | None) -> pd.DataFrame | None:
    st.markdown("#### Unified CSV Loader")
    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.warning("Unified CSV file not found. Please set it in Global Input Configuration.")
        return None

    loaded_df = get_loaded_visualization_df(unified_csv_path)
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Load Unified CSV", key="tab9_viz_shared_load_btn", type="primary"):
            try:
                with st.spinner("Loading unified CSV into memory..."):
                    loaded_df = get_cached_unified_df(unified_csv_path)
                st.success(f"Loaded {len(loaded_df):,} rows into memory.")
            except Exception as exc:
                st.error(f"Failed to load unified CSV: {exc}")
    with col2:
        if st.button("Unload CSV", key="tab9_viz_shared_unload_btn"):
            clear_loaded_visualization_df()
            log_terminal_event("Released unified CSV from Tab 9 visualization memory cache")
            st.success("Unified CSV cache cleared.")
            st.rerun()
    with col3:
        if isinstance(loaded_df, pd.DataFrame):
            st.caption(f"Loaded in memory: {len(loaded_df):,} rows | {len(loaded_df.columns)} columns")
        else:
            st.caption(
                "CSV not loaded yet. Tabs open normally; click 'Load Unified CSV' once to reuse it across all Visualization subtabs."
            )
    return loaded_df if isinstance(loaded_df, pd.DataFrame) else None


def extract_list_from_notebook(var_names: list[str]) -> list[str]:
    try:
        notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / NOTEBOOK_NAME
        if not notebook_path.exists():
            return []
        with open(notebook_path, "r") as f:
            nb = json.load(f)

        def _extract_list(var_name: str, cell_source: str) -> list[str]:
            match = re.search(rf"{re.escape(var_name)}\s*=\s*\[(.*?)\]", cell_source, flags=re.S)
            if not match:
                return []
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
            out: list[str] = []
            seen: set[str] = set()
            for token in tokens:
                cleaned = str(token).strip()
                if not cleaned or cleaned in seen:
                    continue
                out.append(cleaned)
                seen.add(cleaned)
            return out

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source:
                continue
            for var_name in var_names:
                if var_name in source:
                    values = _extract_list(var_name, source)
                    if values:
                        return values
        return []
    except Exception:
        return []


def extract_scalar_from_notebook(var_name: str) -> str | None:
    try:
        notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / NOTEBOOK_NAME
        if not notebook_path.exists():
            return None
        with open(notebook_path, "r") as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source or var_name not in source:
                continue
            match = re.search(rf"{re.escape(var_name)}\s*=\s*([^\n#]+)", source)
            if match:
                return match.group(1).strip().strip("'\"")
        return None
    except Exception:
        return None


def extract_list_defaults_from_notebook(var_name: str) -> list[str]:
    return extract_list_from_notebook([var_name]) or []


def cluster_order(values: pd.DataFrame, axis: int) -> list[str]:
    if linkage is None or leaves_list is None:
        return list(values.index if axis == 0 else values.columns)
    data = values.values if axis == 0 else values.values.T
    if data.shape[0] <= 1:
        return list(values.index if axis == 0 else values.columns)
    try:
        # Small expression matrices can coincidentally look like precomputed
        # distance matrices.  Here they are always observations, so SciPy's
        # heuristic ClusterWarning is not actionable.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            link = linkage(data, method="average", metric="euclidean")
        order = leaves_list(link)
        labels = list(values.index if axis == 0 else values.columns)
        return [labels[i] for i in order]
    except Exception:
        return list(values.index if axis == 0 else values.columns)


def apply_custom_order(items: list[str], custom_list: list[str]) -> list[str]:
    ordered = [item for item in custom_list if item in items]
    remainder = [item for item in items if item not in ordered]
    return ordered + remainder


def build_palette(colors: list[str]) -> list[str]:
    return colors if colors else sns.color_palette("tab10").as_hex()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "heatmap"


def strip_suffixes(value: str, suffixes: list[str]) -> str:
    cleaned = str(value)
    for suffix in suffixes:
        if suffix and cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned


def get_filterable_columns(columns: list[str], excluded: list[str] | None = None) -> list[str]:
    excluded_set = {str(col) for col in (excluded or [])}
    blocked_prefixes = ("umap_", "leiden_")
    blocked_suffixes = ("_norm",)
    filterable: list[str] = []
    for column in columns:
        column_str = str(column)
        column_lower = column_str.lower()
        if column_str in excluded_set:
            continue
        if column_lower.startswith(blocked_prefixes):
            continue
        if column_lower.endswith(blocked_suffixes):
            continue
        filterable.append(column_str)
    return filterable


def get_low_cardinality_filter_columns(
    df: pd.DataFrame,
    columns: list[str],
    excluded: list[str] | None = None,
    max_unique: int = 100,
) -> list[str]:
    candidate_columns = get_filterable_columns(columns, excluded=excluded)
    low_cardinality: list[str] = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        try:
            unique_count = int(non_null.nunique(dropna=True))
        except Exception:
            continue
        if unique_count == 0 or unique_count > max_unique:
            continue
        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            low_cardinality.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series) and unique_count <= min(20, max_unique):
            low_cardinality.append(column)
    return low_cardinality


def get_filter_value_options(df: pd.DataFrame, column: str, limit: int = 500) -> list[str]:
    if column not in df.columns:
        return []
    try:
        values = df[column].dropna().astype(str).map(str.strip)
        values = values[values != ""]
        return sorted(values.unique().tolist())[:limit]
    except Exception:
        return []


def read_heatmap_filter_specs(n_filters: int, key_prefix: str = "viz_heatmap") -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    for idx in range(n_filters):
        column = str(st.session_state.get(f"{key_prefix}_filter_col_{idx}", "") or "").strip()
        selected_values = st.session_state.get(f"{key_prefix}_filter_vals_{idx}", [])
        if not column or not isinstance(selected_values, list) or not selected_values:
            continue
        cleaned_values = [str(value) for value in selected_values if str(value).strip()]
        if cleaned_values:
            specs.append((column, cleaned_values))
    return specs


def apply_value_filters(df: pd.DataFrame, filter_specs: list[tuple[str, list[str]]]) -> tuple[pd.DataFrame, list[str]]:
    filtered_df = df.copy()
    summary_lines: list[str] = []
    for column, selected_values in filter_specs:
        if column not in filtered_df.columns:
            summary_lines.append(f"Skipped '{column}' (column not found)")
            continue
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected_values)].copy()
        after_count = len(filtered_df)
        preview = ", ".join(selected_values[:5])
        if len(selected_values) > 5:
            preview = f"{preview}, ..."
        summary_lines.append(f"{column} in [{preview}] ({before_count:,} -> {after_count:,})")
    return filtered_df, summary_lines


def prepare_filtered_df(df: pd.DataFrame, gt_col: str) -> pd.DataFrame:
    if gt_col not in df.columns:
        raise ValueError(f"Ground truth column '{gt_col}' not found.")
    work = df.copy()
    work = work[~work[gt_col].isin(["unclassified"])].copy()
    work = work.dropna(subset=[gt_col])
    return work


def compute_method_metrics(df: pd.DataFrame, gt_col: str, methods: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    available_methods = [method for method in methods if method in df.columns]
    for method in available_methods:
        method_df = df[df[method].notna() & (df[method] != "unknown") & df[gt_col].notna()].copy()
        if method_df.empty:
            continue
        y_true = method_df[gt_col]
        y_pred = method_df[method]
        labels = sorted(list(set(y_true) | set(y_pred)))
        predicted_value_counts = y_pred.astype(str).value_counts()
        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        results.append(
            {
                "method": method,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_weighted": precision_w,
                "recall_weighted": recall_w,
                "f1_weighted": f1_w,
                "cohen_kappa": cohen_kappa_score(y_true, y_pred),
                "matthews_corr": matthews_corrcoef(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
                "labels": labels,
                "per_class_precision": precision_per,
                "per_class_recall": recall_per,
                "per_class_f1": f1_per,
                "per_class_support": support_per,
                "predicted_class_support": [int(predicted_value_counts.get(str(label), 0)) for label in labels],
                "n_cells": len(method_df),
                "coverage": (len(method_df) / len(df)) * 100,
            }
        )
    return results


def collect_filtered_cell_types(results: list[dict[str, Any]], exclude_list: list[str]) -> list[str]:
    cell_types = {label for result in results for label in result.get("labels", [])}
    return sorted([cell_type for cell_type in cell_types if cell_type not in exclude_list])


def build_per_class_matrix(
    results: list[dict[str, Any]],
    metric_key: str,
    cell_types: list[str],
) -> pd.DataFrame:
    methods = [result["method"] for result in results]
    matrix = pd.DataFrame(index=cell_types, columns=methods, dtype=float)
    for result in results:
        method = result["method"]
        labels = result.get("labels", [])
        values = result.get(metric_key, [])
        for idx, label in enumerate(labels):
            if label in cell_types and idx < len(values):
                matrix.loc[label, method] = values[idx]
    return matrix


def plot_comprehensive_performance(
    results: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(22, 14),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    fig.suptitle("Comprehensive Phenotyping Validation", fontsize=16, fontweight="bold")

    methods = [result["method"] for result in results]

    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cohen_kappa"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa"]
    x = np.arange(len(methods))
    width = 0.8 / max(len(metrics), 1)
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [result[metric] for result in results]
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        ax1.bar(x + offset, values, width, label=name, alpha=0.88)
    ax1.set_title("Multiple Metrics Comparison", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_xlabel("Methods", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10, edgecolor="black")

    phenotype_totals: dict[str, int] = {}
    for result in results:
        labels = result.get("labels", [])
        predicted_support = result.get("predicted_class_support", result.get("per_class_support", []))
        for label, count in zip(labels, predicted_support, strict=False):
            label_text = str(label)
            phenotype_totals[label_text] = phenotype_totals.get(label_text, 0) + int(count)

    phenotype_order = [
        label for label, _ in sorted(phenotype_totals.items(), key=lambda item: (-item[1], item[0].lower())) if _ > 0
    ]
    if not phenotype_order:
        phenotype_order = sorted({str(label) for result in results for label in result.get("labels", [])})

    proportion_df = pd.DataFrame(0.0, index=methods, columns=phenotype_order, dtype=float)
    n_cells = [int(result.get("n_cells", 0)) for result in results]
    for result in results:
        method = result["method"]
        labels = [str(label) for label in result.get("labels", [])]
        predicted_support = [int(value) for value in result.get("predicted_class_support", result.get("per_class_support", []))]
        total = max(sum(predicted_support), 0)
        if total <= 0:
            continue
        for label, count in zip(labels, predicted_support, strict=False):
            if label in proportion_df.columns:
                proportion_df.loc[method, label] = count / total

    if len(phenotype_order) <= 20:
        phenotype_colors = sns.color_palette("tab20", n_colors=len(phenotype_order))
    else:
        phenotype_colors = sns.color_palette("husl", n_colors=len(phenotype_order))

    bottoms = np.zeros(len(methods), dtype=float)
    for phenotype, color in zip(phenotype_order, phenotype_colors, strict=False):
        values = proportion_df[phenotype].to_numpy(dtype=float)
        ax2.bar(
            x,
            values,
            bottom=bottoms,
            width=0.75,
            label=phenotype,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += values

    ax2.set_title("Phenotype Composition of Analyzed Cells", fontweight="bold")
    ax2.set_ylabel("Proportion of Analyzed Cells", fontweight="bold")
    ax2.set_xlabel("Phenotyping Methods", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylim(0, 1.12)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    for idx, total_cells in enumerate(n_cells):
        ax2.text(
            x[idx],
            1.02,
            f"n={total_cells:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    if phenotype_order:
        legend_cols = min(4, max(1, int(np.ceil(len(phenotype_order) / 8))))
        ax2.legend(
            title="Phenotypes",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=legend_cols,
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            edgecolor="black",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "comprehensive_performance_comparison.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def plot_spider_web_performance(
    results: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cohen_kappa"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    num_vars = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    for idx, result in enumerate(results):
        values = [result.get(metric, 0) for metric in metrics]
        if "cohen_kappa" in metrics:
            kappa_idx = metrics.index("cohen_kappa")
            values[kappa_idx] = (values[kappa_idx] + 1) / 2
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, "o-", color=color, linewidth=2.5, markersize=7, label=result["method"])
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=12)
    ax.tick_params(axis="x", pad=25)

    plt.suptitle("Multi-Dimensional Performance Comparison", fontsize=18, fontweight="bold", y=1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=11)

    fig.tight_layout(pad=2)
    fig_path = output_dir / "spider_web_performance_comparison.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def plot_confusion_matrices(
    results: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    n_methods = len(results)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows), squeeze=False)
    axes = axes.flatten()

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    for idx, result in enumerate(results):
        ax = axes[idx]
        cm = result["confusion_matrix"]
        labels = result["labels"]
        im = ax.imshow(cm, cmap="Blues")
        for i_cm in range(len(labels)):
            for j_cm in range(len(labels)):
                ax.text(j_cm, i_cm, cm[i_cm, j_cm], ha="center", va="center", color="black", fontweight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{result['method']}\n(Acc: {result['accuracy']:.3f})", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "comprehensive_confusion_matrices.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def plot_average_per_class_metrics(
    results: list[dict[str, Any]],
    cell_types: list[str],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    methods = [result["method"] for result in results]
    method_metrics: dict[str, dict[str, list[float]]] = {
        method: {"precision": [], "recall": [], "f1": []} for method in methods
    }

    for result in results:
        method = result["method"]
        labels = result.get("labels", [])
        for idx, label in enumerate(labels):
            if label in cell_types:
                method_metrics[method]["precision"].append(result["per_class_precision"][idx])
                method_metrics[method]["recall"].append(result["per_class_recall"][idx])
                method_metrics[method]["f1"].append(result["per_class_f1"][idx])

    metric_stats: dict[str, dict[str, float]] = {}
    for method in methods:
        metric_stats[method] = {
            "precision_mean": float(np.mean(method_metrics[method]["precision"])) if method_metrics[method]["precision"] else 0,
            "precision_std": float(np.std(method_metrics[method]["precision"])) if method_metrics[method]["precision"] else 0,
            "recall_mean": float(np.mean(method_metrics[method]["recall"])) if method_metrics[method]["recall"] else 0,
            "recall_std": float(np.std(method_metrics[method]["recall"])) if method_metrics[method]["recall"] else 0,
            "f1_mean": float(np.mean(method_metrics[method]["f1"])) if method_metrics[method]["f1"] else 0,
            "f1_std": float(np.std(method_metrics[method]["f1"])) if method_metrics[method]["f1"] else 0,
        }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Average Per-Class Metrics Across Methods (Filtered: {len(cell_types)} cell types)",
        fontsize=16,
        fontweight="bold",
    )

    x = np.arange(len(methods))
    width = 0.6

    precision_means = [metric_stats[m]["precision_mean"] for m in methods]
    precision_stds = [metric_stats[m]["precision_std"] for m in methods]
    ax1.bar(
        x,
        precision_means,
        width,
        yerr=precision_stds,
        capsize=8,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
        color="coral",
        error_kw={"linewidth": 2},
    )
    ax1.set_ylabel("Average Precision", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Methods", fontweight="bold", fontsize=12)
    ax1.set_title("Average Precision by Cell Type", fontweight="bold", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    recall_means = [metric_stats[m]["recall_mean"] for m in methods]
    recall_stds = [metric_stats[m]["recall_std"] for m in methods]
    ax2.bar(
        x,
        recall_means,
        width,
        yerr=recall_stds,
        capsize=8,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
        color="lightgreen",
        error_kw={"linewidth": 2},
    )
    ax2.set_ylabel("Average Recall", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Methods", fontweight="bold", fontsize=12)
    ax2.set_title("Average Recall by Cell Type", fontweight="bold", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    f1_means = [metric_stats[m]["f1_mean"] for m in methods]
    f1_stds = [metric_stats[m]["f1_std"] for m in methods]
    ax3.bar(
        x,
        f1_means,
        width,
        yerr=f1_stds,
        capsize=8,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
        color="skyblue",
        error_kw={"linewidth": 2},
    )
    ax3.set_ylabel("Average F1-Score", fontweight="bold", fontsize=12)
    ax3.set_xlabel("Methods", fontweight="bold", fontsize=12)
    ax3.set_title("Average F1-Score by Cell Type", fontweight="bold", fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha="right")
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "average_per_class_metrics_with_errorbars.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path
