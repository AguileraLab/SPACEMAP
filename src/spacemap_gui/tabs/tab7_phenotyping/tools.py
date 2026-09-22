"""
TOOLS Subtab
Accessories for manual reference and auxiliary columns (Step7 accessories).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spacemap_gui.ui_utils import directory_input_with_browse

from .common import (
    _extract_markers_from_notebook,
    get_all_phenotype_options,
    get_marker_workspace_panel,
    get_normalization_flag_help_text,
    logger,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    render_add_custom_phenotype_ui,
    render_method_marker_selection,
    render_settings_export_import,
    render_phenotyping_run_log_controls,
)

# acc_utils ships inside the `spacemap` package; `src` is what the launcher puts
# on PYTHONPATH, so the bare `acc_utils` name never resolves.
try:
    from spacemap.acc_utils import (
        extract_classification_data,
        batch_process_geojson_files,
        merge_ground_truth_into_unified,
        enrich_ground_truth_with_unified_data,
        relabel_ground_truth_files,
        parse_gt_prefix,
        annotate_geojson_with_phenotypes,
        generate_png_from_geojson,
        build_color_mapping,
        feature_label,
        missing_label_mask,
        build_agreement_labels,
        DEFAULT_UNASSIGNED_LABEL,
        DEFAULT_UNASSIGNED_COLOR,
        DEFAULT_NA_TOKENS,
        DEFAULT_MATCH_COLOR,
        DEFAULT_MISMATCH_COLOR,
        DEFAULT_IGNORED_COLOR,
    )
    ACC_UTILS_IMPORT_ERROR = None
except Exception as _acc_exc:  # noqa: BLE001 - surfaced in the UI instead
    ACC_UTILS_IMPORT_ERROR = _acc_exc
    extract_classification_data = None
    batch_process_geojson_files = None
    merge_ground_truth_into_unified = None
    enrich_ground_truth_with_unified_data = None
    relabel_ground_truth_files = None
    parse_gt_prefix = None
    annotate_geojson_with_phenotypes = None
    generate_png_from_geojson = None
    build_color_mapping = None
    feature_label = None
    missing_label_mask = None
    build_agreement_labels = None
    DEFAULT_UNASSIGNED_LABEL = "unassigned"
    DEFAULT_UNASSIGNED_COLOR = "#D9D9D9"
    DEFAULT_NA_TOKENS = ("", "nan", "none", "na", "n/a", "null", "<na>", "unassigned")
    DEFAULT_MATCH_COLOR = "#00693C"
    DEFAULT_MISMATCH_COLOR = "#D62728"
    DEFAULT_IGNORED_COLOR = "#D9D9D9"

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

try:
    from shapely.geometry import shape
except Exception:
    shape = None

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

if CellVisualizer is not None:
    @st.cache_resource
    def _get_cell_visualizer_cached(roi_dir: str, seg_dir: str):
        return CellVisualizer(roi_dir, seg_dir)
else:
    def _get_cell_visualizer_cached(roi_dir: str, seg_dir: str):
        return None

def _dataio_load(path, columns=None):
    """Read a large table through the shared loader.

    Uses the Parquet fast copy when one is current, otherwise the CSV. Passing
    *columns* reads only those.
    """
    from spacemap import dataio

    return dataio.load(path, columns=columns)


def _load_unified_csv_cached(path, columns: list[str] | None = None):
    """Cache-safe wrapper: re-reads when the file on disk changes.

    Naming *columns* reads only those, from the Parquet fast copy when one is
    available, which is what keeps a large table usable here.
    """
    from spacemap_gui.ui_utils import file_signature
    key = tuple(columns) if columns else None
    return __load_unified_csv_cached__impl(path, file_signature(path), key)


@st.cache_data(show_spinner=False)
def __load_unified_columns__impl(
    path: str, signature, columns: tuple[str, ...]
) -> pd.DataFrame | None:
    """Cache a narrow column selection, which is cheap to store."""
    from spacemap import dataio

    try:
        return dataio.load(path, columns=list(columns))
    except Exception:
        return None


def __load_unified_csv_cached__impl(
    path: str, signature, columns: tuple[str, ...] | None = None
) -> pd.DataFrame | None:
    # Streamlit's cache pickles what it stores and unpickles it on every hit.
    # That is fine for a narrow column selection and very expensive for the
    # whole table -- twenty to thirty seconds per hit on a 15 GB frame -- so a
    # full read bypasses the cache and is held by the caller instead.
    if columns:
        return __load_unified_columns__impl(path, signature, columns)

    from spacemap import dataio

    try:
        return dataio.load(path)
    except Exception:
        return None


def _get_unique_values_cached(path, column):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_unique_values_cached__impl(path, column, file_signature(path))


@st.cache_data(show_spinner=False)
def __get_unique_values_cached__impl(path: str, column: str, signature) -> list[str]:
    """Cache unique values from a specific column to avoid re-reading large CSVs."""
    try:
        series = _read_single_column(path, column)
        if series is None:
            return []
        return sorted(series.dropna().astype(str).unique().tolist())
    except Exception:
        return []


def _read_columns_fast(path: str, columns: list[str]) -> pd.DataFrame | None:
    """Read a subset of a CSV's columns as fast as the installed libraries allow.

    pandas scans the whole file to pull even a single column -- about 22s on a
    4.4 GB table -- which is long enough to make the filter controls feel frozen,
    since every column choice triggers one. The shared loader reads from the
    Parquet fast copy when one exists, which skips the unwanted columns on disk
    entirely, and otherwise uses pyarrow's threaded CSV reader. Falls back to
    pandas if neither is available, and returns None when the columns cannot be
    read at all.
    """
    try:
        from spacemap import dataio

        return dataio.load(path, columns=list(columns))
    except Exception:
        pass
    try:
        # low_memory=False avoids pandas' chunked type inference, which is what
        # emits "Columns (n) have mixed types" on heterogeneous columns.
        return pd.read_csv(path, usecols=list(columns), low_memory=False)
    except Exception:
        return None


def _read_single_column(path: str, column: str):
    """Read one column from a CSV. See `_read_columns_fast` for the mechanics."""
    frame = _read_columns_fast(path, [column])
    if frame is None or column not in frame.columns:
        return None
    return frame[column]


def _load_manual_reference_csv_fresh(
    path: str,
    columns: list[str],
) -> pd.DataFrame | None:
    """Read Manual Reference source rows directly from disk, without UI caching.

    Assignment cards edit real records rather than merely previewing them. Serving an
    older cached DataFrame here can show a pre-save phenotype and can put a cleared cell
    back into the review pool. Only the columns used by this workflow are read, keeping
    the fresh read practical even for a large unified CSV.
    """
    if not path or not os.path.exists(path):
        return None
    requested_columns = list(
        dict.fromkeys(str(column) for column in columns if str(column).strip())
    )
    if not requested_columns:
        return None
    return _read_columns_fast(path, requested_columns)


def _get_columns_cached(path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_columns_cached__impl(path, file_signature(path))


@st.cache_data(show_spinner=False)
def __get_columns_cached__impl(path: str, signature) -> list[str]:
    """Cache column names from a CSV file (fast, reads only header)."""
    try:
        if not path or not os.path.exists(path):
            return []
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _get_column_profile_cached(path, column):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_column_profile_cached__impl(path, column, file_signature(path))


def _get_unique_values_fresh(path: str, column: str) -> list[str]:
    """Read categorical choices directly from disk for Manual Reference controls."""
    try:
        series = _read_single_column(path, column)
        if series is None:
            return []
        return sorted(series.dropna().astype(str).unique().tolist())
    except Exception:
        return []


def _get_column_profile_fresh(path: str, column: str) -> dict[str, Any]:
    """Read numeric filter bounds directly from disk for Manual Reference controls."""
    if not path or not os.path.exists(path) or not column:
        return {}
    series = _read_single_column(path, column)
    if series is None:
        return {}
    non_null = series.dropna()
    numeric = (
        pd.to_numeric(non_null, errors="coerce")
        if not non_null.empty
        else pd.Series(dtype=float)
    )
    is_numeric = bool(not non_null.empty and numeric.notna().all())
    profile: dict[str, Any] = {
        "is_numeric": is_numeric,
        "unique_count": int(non_null.astype(str).nunique()) if not non_null.empty else 0,
    }
    if is_numeric and not numeric.empty:
        profile["min"] = float(numeric.min())
        profile["max"] = float(numeric.max())
    return profile


@st.cache_data(show_spinner=False)
def __get_column_profile_cached__impl(path: str, column: str, signature) -> dict[str, Any]:
    """Cache basic type and range information for a single CSV column."""
    if not path or not os.path.exists(path) or not column:
        return {}

    series = _read_single_column(path, column)
    if series is None:
        return {}

    non_null = series.dropna()
    numeric = pd.to_numeric(non_null, errors="coerce") if not non_null.empty else pd.Series(dtype=float)
    is_numeric = bool(not non_null.empty and numeric.notna().all())
    profile: dict[str, Any] = {
        "is_numeric": is_numeric,
        "unique_count": int(non_null.astype(str).nunique()) if not non_null.empty else 0,
    }
    if is_numeric and not numeric.empty:
        profile["min"] = float(numeric.min())
        profile["max"] = float(numeric.max())
    return profile


def _load_source_csv_cached(path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __load_source_csv_cached__impl(path, file_signature(path))


@st.cache_data(show_spinner=False)
def __load_source_csv_cached__impl(path: str, signature) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def _get_csv_row_count_cached(path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_csv_row_count_cached__impl(path, file_signature(path))


@st.cache_data(show_spinner=False)
def __get_csv_row_count_cached__impl(path: str, signature) -> int:
    try:
        if not path or not os.path.exists(path):
            return 0
        # Count lines using raw buffer reading (very fast, ~1-2 seconds for 6GB on typical fast disk)
        count = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        return max(0, count - 1)  # exclude header
    except Exception:
        return 0


def _get_csv_metadata_cached(path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_csv_metadata_cached__impl(path, file_signature(path))


@st.cache_data(show_spinner=False)
def __get_csv_metadata_cached__impl(path: str, signature) -> dict[str, Any]:
    """Get metadata and basic stats of the CSV file without loading it completely."""
    if not path or not os.path.exists(path):
        return {}
    try:
        file_size_gb = os.path.getsize(path) / (1024 ** 3)
        cols = _get_columns_cached(path)
        row_count = _get_csv_row_count_cached(path)
        return {
            "path": path,
            "file_size_gb": file_size_gb,
            "columns": cols,
            "row_count": row_count,
        }
    except Exception:
        return {}


def _get_csv_preview_df_cached(path, nrows: int = 200):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_csv_preview_df_cached__impl(path, nrows, file_signature(path))


@st.cache_data(show_spinner=False)
def __get_csv_preview_df_cached__impl(path: str, nrows: int = 200, signature=None) -> pd.DataFrame:
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def _get_csv_preview_grid_cached(path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return __get_csv_preview_grid_cached__impl(path, file_signature(path))


@st.cache_data(show_spinner=False)
def __get_csv_preview_grid_cached__impl(path: str, signature) -> pd.DataFrame:
    try:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        # Read first 10,000 rows to get types and missing counts
        df = pd.read_csv(path, nrows=10000)
        records = []
        for col in df.columns:
            series = df[col]
            col_type = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
            missing_sample = int(series.isna().sum())
            records.append({
                "column": col,
                "type": col_type,
                "missing (first 10k)": missing_sample,
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()



# Callers clear these caches after writing to the CSV so the next read sees the
# new file. The public name is a plain wrapper and carries no clear() of its own,
# so forward it to the cached implementation that actually holds the entries.
# The cached layer is the column-subset one; the full-table path is not
# cached, so this is where the clearable cache lives.
_load_unified_csv_cached.clear = __load_unified_columns__impl.clear  # type: ignore[attr-defined]
_get_unique_values_cached.clear = __get_unique_values_cached__impl.clear  # type: ignore[attr-defined]
_get_columns_cached.clear = __get_columns_cached__impl.clear  # type: ignore[attr-defined]
_get_column_profile_cached.clear = __get_column_profile_cached__impl.clear  # type: ignore[attr-defined]
_load_source_csv_cached.clear = __load_source_csv_cached__impl.clear  # type: ignore[attr-defined]
_get_csv_row_count_cached.clear = __get_csv_row_count_cached__impl.clear  # type: ignore[attr-defined]
_get_csv_metadata_cached.clear = __get_csv_metadata_cached__impl.clear  # type: ignore[attr-defined]
_get_csv_preview_df_cached.clear = __get_csv_preview_df_cached__impl.clear  # type: ignore[attr-defined]
_get_csv_preview_grid_cached.clear = __get_csv_preview_grid_cached__impl.clear  # type: ignore[attr-defined]


def _find_matching_indices(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    anchors: list[dict[str, Any]],
) -> pd.Series:
    """Find source DataFrame row indices matching target DataFrame rows."""
    # Ensure source has a clean unique range index to get direct integer coordinates/row positions
    source_df_clean = source_df.reset_index(drop=True)

    exact_pairs = [a for a in anchors if not a["is_tolerant"]]
    tolerant_pairs = [a for a in anchors if a["is_tolerant"]]

    matched_source_idx = pd.Series(index=target_df.index, dtype=object)

    if not tolerant_pairs:
        # 100% Vectorized exact match (runs in < 1 second for millions of rows)
        target_exact_cols = [a["target_col"] for a in exact_pairs]
        source_exact_cols = [a["source_col"] for a in exact_pairs]

        # Select key columns
        target_temp = target_df[target_exact_cols].copy()
        target_temp["_target_index"] = target_df.index

        source_temp = source_df_clean[source_exact_cols].copy()
        source_temp["_source_index"] = source_df_clean.index

        # Rename source columns to align with target
        rename_dict = {s: t for s, t in zip(source_exact_cols, target_exact_cols)}
        source_temp = source_temp.rename(columns=rename_dict)

        # Drop duplicates in source keys to ensure 1-to-1 or many-to-1 mapping
        source_temp = source_temp.drop_duplicates(subset=target_exact_cols)

        # Merge on key columns
        merged = pd.merge(target_temp, source_temp, on=target_exact_cols, how="left")

        # Map back to target index
        matched_source_idx = pd.Series(merged["_source_index"].values, index=merged["_target_index"].values)
        return matched_source_idx

    # If there are tolerant pairs
    if exact_pairs:
        target_exact_cols = [a["target_col"] for a in exact_pairs]
        source_exact_cols = [a["source_col"] for a in exact_pairs]

        # Group target by exact keys using fast native multi-column groupby
        target_groups = target_df.groupby(target_exact_cols)

        # Index source by exact keys for O(1) lookup
        source_indexed = source_df_clean.copy()
        source_indexed["_source_index"] = source_df_clean.index
        source_indexed = source_indexed.set_index(source_exact_cols).sort_index()

        for key, target_sub in target_groups:
            try:
                # Select corresponding source rows
                source_sub = source_indexed.loc[key]
            except KeyError:
                continue

            if source_sub.empty:
                continue

            # If source_sub is a Series (only 1 match and 1 column index), convert to DataFrame
            if isinstance(source_sub, pd.Series):
                source_sub = source_sub.to_frame().T

            # Run tolerant matching on subsets
            match_sub = _match_tolerant_subsets(target_sub, source_sub, tolerant_pairs)
            matched_source_idx.update(match_sub)
    else:
        # No exact keys: match whole datasets using tolerant keys
        match_sub = _match_tolerant_subsets(target_df, source_df_clean, tolerant_pairs)
        matched_source_idx.update(match_sub)

    return matched_source_idx


def _match_tolerant_subsets(
    target_sub: pd.DataFrame,
    source_sub: pd.DataFrame,
    tolerant_pairs: list[dict[str, Any]],
) -> pd.Series:
    t_cols_target = [a["target_col"] for a in tolerant_pairs]
    t_cols_source = [a["source_col"] for a in tolerant_pairs]

    tols = np.array([max(float(a["tolerance_val"]), 1e-9) for a in tolerant_pairs])

    # Convert coordinates to numpy arrays quickly
    target_vals = target_sub[t_cols_target].to_numpy(dtype=float)

    # Note: source_sub might have an index that is not unique if it was indexed by exact keys
    # But it has a column "_source_index" which holds the original row numbers!
    if "_source_index" in source_sub.columns:
        source_indices = source_sub["_source_index"].to_numpy()
    else:
        source_indices = source_sub.index.to_numpy()

    source_vals = source_sub[t_cols_source].to_numpy(dtype=float)

    # Scale values by tolerances for distance calculation
    target_scaled = target_vals / tols
    source_scaled = source_vals / tols

    result = pd.Series(index=target_sub.index, dtype=object)

    if cKDTree is not None and len(source_scaled) > 0:
        tree = cKDTree(source_scaled)
        distances, indices = tree.query(target_scaled, k=1)

        # Map selected indices to original source row positions
        matched_positions = source_indices[indices]

        # Verify individual box constraints
        diff = np.abs(target_vals - source_vals[indices])
        valid = np.all(diff <= tols, axis=1)

        result.loc[target_sub.index[valid]] = matched_positions[valid]
    else:
        # Faster numpy-vectorized fallback instead of row-by-row loop
        for t_idx, t_val in zip(target_sub.index, target_vals):
            diff = np.abs(source_vals - t_val)
            in_tol = np.all(diff <= tols, axis=1)
            if np.any(in_tol):
                valid_idx = np.where(in_tol)[0]
                dists = np.sum((diff[valid_idx] / tols) ** 2, axis=1)
                best_sub_idx = valid_idx[np.argmin(dists)]
                result.loc[t_idx] = source_indices[best_sub_idx]

    return result


def _init_edit_csv_state(unified_csv_path: str | None) -> None:
    path_key = unified_csv_path or ""
    if st.session_state.get("tools_edit_csv_path") != path_key:
        st.session_state["tools_edit_csv_path"] = path_key
        st.session_state["tools_edit_csv_df"] = None
        st.session_state["tools_edit_csv_undo"] = []
        st.session_state["tools_edit_csv_redo"] = []
        st.session_state["tools_edit_csv_last_preview"] = {}
        st.session_state["tools_edit_csv_filters"] = []
        st.session_state["tools_edit_csv_rules"] = None


def _push_edit_csv_state(
    new_df: pd.DataFrame,
    label: str,
    before_df: pd.DataFrame | None = None,
    after_df: pd.DataFrame | None = None,
) -> None:
    current_df = st.session_state.get("tools_edit_csv_df")
    if current_df is not None:
        undo_stack = st.session_state.setdefault("tools_edit_csv_undo", [])
        undo_stack.append({"label": label, "df": current_df.copy(deep=True)})
        st.session_state["tools_edit_csv_redo"] = []
    st.session_state["tools_edit_csv_df"] = new_df
    if before_df is not None and after_df is not None:
        st.session_state["tools_edit_csv_last_preview"] = {
            "label": label,
            "before": before_df.head(25).copy(),
            "after": after_df.head(25).copy(),
        }


def _preview_grid(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in df.columns:
        series = df[col]
        col_type = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        # Only compute missing count on head(10000) for instant performance
        missing_sample = int(series.head(10000).isna().sum())
        records.append(
            {
                "column": col,
                "type": col_type,
                "missing (first 10k)": missing_sample,
            }
        )
    return pd.DataFrame(records)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _apply_value_cleanup(
    series: pd.Series,
    *,
    trim_whitespace: bool,
    strip_prefix: str,
    strip_suffix: str,
    replace_from: str,
    replace_to: str,
    replace_regex: bool,
) -> pd.Series:
    result = series.astype(str)
    if trim_whitespace:
        result = result.str.strip()
    if strip_prefix:
        result = result.str.replace(f"^{re.escape(strip_prefix)}", "", regex=True)
    if strip_suffix:
        result = result.str.replace(f"{re.escape(strip_suffix)}$", "", regex=True)
    if replace_from:
        result = result.str.replace(replace_from, replace_to, regex=replace_regex)
    return result


def _build_rule_mask(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: str,
    value_2: str,
) -> pd.Series:
    series = df[column]
    if operator == "is missing":
        return series.isna()

    if pd.api.types.is_numeric_dtype(series):
        numeric = _coerce_numeric(series)
        if operator == "<":
            return numeric < float(value)
        if operator == "<=":
            return numeric <= float(value)
        if operator == ">":
            return numeric > float(value)
        if operator == ">=":
            return numeric >= float(value)
        if operator == "between":
            return numeric.between(float(value), float(value_2))
        return pd.Series([False] * len(df), index=df.index)

    text = series.astype(str)
    if operator == "equals":
        return text == str(value)
    if operator == "in list":
        values = [v.strip() for v in str(value).split(",") if v.strip()]
        return text.isin(values)
    if operator == "contains":
        return text.str.contains(str(value), case=False, na=False)
    if operator == "startswith":
        return text.str.startswith(str(value), na=False)
    if operator == "endswith":
        return text.str.endswith(str(value), na=False)
    return pd.Series([False] * len(df), index=df.index)


NOTEBOOK_NAME = "Step7_accessories.ipynb"


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _log_console(message: str) -> None:
    print(message, flush=True)


def _normalize_label(label: str) -> str:
    """Normalize phenotype labels for consistent relabeling."""
    value = str(label).strip().lower()
    value = value.replace("-", " ").replace("_", " ")
    value = " ".join(value.split())
    if value.endswith(" cell"):
        value = value[: -len(" cell")]
    if value.endswith("s") and len(value) > 3:
        value = value[:-1]
    return value


def _sanitize_filename_token(value: str) -> str:
    """Sanitize a value for safe use in filenames.

    Parameters
    ----------
    value : str
        Input value to sanitize.

    Returns
    -------
    str
        Safe filename token.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def _get_visual_marker_candidates(available_columns: list[str]) -> list[str]:
    metadata_cols = {
        "cell_id",
        "batch_id",
        "tissue",
        "x_centroid",
        "y_centroid",
        "area",
        "perimeter",
        "eccentricity",
        "solidity",
        "extent",
        "major_axis",
        "minor_axis",
        "orientation",
        "label",
        "index",
    }
    return [
        column
        for column in available_columns
        if column.lower() not in metadata_cols
        and not column.startswith("phenotype_")
        and not column.startswith("umap_")
        and not column.startswith("leiden_")
        and "_mrvv_" not in column
        and not column.endswith("_norm")
    ]


def _clean_visual_marker_name(
    marker_name: str,
    affix_position: str | None = None,
    correction_affix: str | None = None,
) -> str:
    cleaned = str(marker_name).strip()
    configured_affix = str(correction_affix or "").strip()
    configured_position = str(affix_position or "").strip().casefold()
    if configured_affix:
        if configured_position == "prefix" and cleaned.startswith(configured_affix):
            cleaned = cleaned[len(configured_affix):]
        elif configured_position != "prefix" and cleaned.endswith(configured_affix):
            cleaned = cleaned[:-len(configured_affix)]

    suffixes = (
        "_norm_resolve",
        "_resolve",
        "_corr_norm",
        "_mrvv_nlog",
        "_mrvv_cdf",
        "_mrvv_z2",
        "_mrvv_z1",
        "_unmixed",
        "_corr",
        "_norm",
        "_log",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                changed = True
                break
    return cleaned.strip("_ ")


def _get_visual_normalization_flags(method: str | list[str] | tuple[str, ...]) -> tuple[bool, bool, bool, bool]:
    if isinstance(method, (list, tuple, set)):
        selected = {str(value).strip().upper() for value in method}
        return "Z1" in selected, "Z2" in selected, "CDF" in selected, bool({"NLOG", "NLG"} & selected)

    normalized = str(method or "Z1").strip().upper()
    if normalized == "Z2":
        return True, True, False, False
    if normalized == "CDF":
        return True, False, True, False
    if normalized == "NLOG":
        return True, False, True, True
    return True, False, False, False


def _normal_cdf_array(values: np.ndarray) -> np.ndarray:
    erf_fn = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_fn(values / math.sqrt(2.0)))


def _apply_visual_normalization(
    df: pd.DataFrame,
    marker_cols: list[str],
    method: str | list[str] | tuple[str, ...],
    group_col: str | None,
    *,
    apply_arcsinh: bool = False,
    arcsinh_cofactor: float = 5.0,
) -> tuple[pd.DataFrame, list[str]]:
    valid_markers = [column for column in marker_cols if column in df.columns]
    if not valid_markers:
        raise ValueError("None of the selected markers were found in the unified CSV.")

    apply_z1, apply_z2, apply_cdf, apply_nlog = _get_visual_normalization_flags(method)
    work_df = df.copy()
    # A missing measurement must stay missing. Replacing it with zero before Z1
    # changes the mean/SD and creates a normalized score that can incorrectly enter
    # the cell's Top 5 markers.
    work_df[valid_markers] = work_df[valid_markers].apply(
        pd.to_numeric, errors="coerce"
    )
    if apply_arcsinh and float(arcsinh_cofactor) <= 0:
        raise ValueError("The arcsinh cofactor must be greater than zero.")

    selected_steps = [
        name
        for name, enabled in zip(
            ("z1", "z2", "cdf", "nlog"),
            (apply_z1, apply_z2, apply_cdf, apply_nlog),
            strict=True,
        )
        if enabled
    ]
    suffix_parts = (["asinh"] if apply_arcsinh else []) + selected_steps
    suffix = f"_mrvv_{'_'.join(suffix_parts) or 'identity'}"
    norm_cols = [f"{column}{suffix}" for column in valid_markers]
    epsilon = 1e-9

    def normalize_subset(subset_df: pd.DataFrame) -> pd.DataFrame:
        current_data = subset_df[valid_markers].astype(float).to_numpy(copy=True)
        if apply_arcsinh:
            current_data = np.arcsinh(current_data / float(arcsinh_cofactor))
        missing_measurements = ~np.isfinite(current_data)
        if subset_df.shape[0] < 2 and (apply_z1 or apply_z2):
            final_data = np.zeros((subset_df.shape[0], len(valid_markers)), dtype=float)
            final_data[missing_measurements] = np.nan
        else:
            processed_data = current_data.copy()
            if apply_z1:
                means_z1 = np.nanmean(processed_data, axis=0)
                stds_z1 = np.nanstd(processed_data, axis=0)
                stds_z1[~np.isfinite(stds_z1) | (stds_z1 == 0)] = epsilon
                processed_data = (processed_data - means_z1) / stds_z1
                processed_data = np.clip(processed_data, -10, 10)
            if apply_z2:
                means_z2 = np.nanmean(processed_data, axis=1, keepdims=True)
                stds_z2 = np.nanstd(processed_data, axis=1, keepdims=True)
                stds_z2[~np.isfinite(stds_z2) | (stds_z2 == 0)] = epsilon
                processed_data = (processed_data - means_z2) / stds_z2
            if apply_cdf:
                processed_data = _normal_cdf_array(processed_data)
            if apply_nlog:
                processed_data = -np.log(np.maximum(1.0 - processed_data, epsilon))
            final_data = np.asarray(processed_data, dtype=float)
            final_data[missing_measurements] = np.nan

        return pd.DataFrame(final_data, index=subset_df.index, columns=norm_cols)

    if group_col and group_col in work_df.columns:
        group_series = work_df[group_col].fillna("__missing__").astype(str)
        normalized_parts = [
            normalize_subset(group)
            for _, group in work_df.groupby(group_series, group_keys=False)
        ]
        normalized_df = pd.concat(normalized_parts).sort_index() if normalized_parts else pd.DataFrame(index=work_df.index)
    else:
        normalized_df = normalize_subset(work_df)

    work_df[norm_cols] = normalized_df.reindex(work_df.index)[norm_cols]
    return work_df, norm_cols


class _Step:
    """Print what a long operation is doing, and how long each part took.

    Manual Reference can spend minutes inside a single Streamlit rerun with no
    output, which makes it impossible to tell a slow step from a hung one.
    Every stage announces itself on the terminal as it starts and reports its
    own duration as it ends.  Set SPACEMAP_TRACE=0 to silence it.
    """

    _enabled = os.environ.get("SPACEMAP_TRACE", "1").strip().lower() not in {"0", "off", "false", "no"}

    def __init__(self, label: str, **details: Any) -> None:
        self.label = label
        self.details = details
        self.started = 0.0

    @staticmethod
    def note(message: str) -> None:
        if _Step._enabled:
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[MR {stamp}] {message}", flush=True)

    def __enter__(self) -> "_Step":
        self.started = time.perf_counter()
        if self._enabled:
            extra = "  ".join(f"{k}={v}" for k, v in self.details.items())
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[MR {stamp}] -> {self.label}{('   ' + extra) if extra else ''}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._enabled:
            elapsed = time.perf_counter() - self.started
            stamp = datetime.now().strftime("%H:%M:%S")
            state = "FAILED" if exc_type else "done"
            print(f"[MR {stamp}] <- {self.label}: {state} in {elapsed:.1f}s", flush=True)
        return False


def _visual_normalization_stats(
    df: pd.DataFrame,
    marker_cols: list[str],
    method: str | list[str] | tuple[str, ...],
    group_col: str | None,
    *,
    apply_arcsinh: bool = False,
    arcsinh_cofactor: float = 5.0,
) -> dict[str, Any]:
    """Compute only the population statistics that normalization needs.

    Manual Reference shows a few dozen cells but used to normalize the entire
    table first -- measured at about 1044 s for 8.6M rows, to produce values
    that take 16 ms for the cells actually displayed.

    Only the Z1 step looks across cells: it subtracts each marker's mean and
    divides by its standard deviation, computed over the population (or over
    each group when a grouping column is set). Z2, the CDF and the negative-log
    steps are per-cell or per-value. So the population pass can be done once,
    here, and the per-cell transform applied to just the rows on screen, which
    gives byte-identical results to normalizing everything.
    """
    valid_markers = [column for column in marker_cols if column in df.columns]
    if not valid_markers:
        raise ValueError("None of the selected markers were found in the unified CSV.")
    if apply_arcsinh and float(arcsinh_cofactor) <= 0:
        raise ValueError("The arcsinh cofactor must be greater than zero.")

    apply_z1, apply_z2, apply_cdf, apply_nlog = _get_visual_normalization_flags(method)
    selected_steps = [
        name
        for name, enabled in zip(
            ("z1", "z2", "cdf", "nlog"),
            (apply_z1, apply_z2, apply_cdf, apply_nlog),
            strict=True,
        )
        if enabled
    ]
    suffix_parts = (["asinh"] if apply_arcsinh else []) + selected_steps
    suffix = f"_mrvv_{'_'.join(suffix_parts) or 'identity'}"
    norm_cols = [f"{column}{suffix}" for column in valid_markers]

    def population_stats(frame: pd.DataFrame) -> dict[str, Any]:
        data = frame[valid_markers].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy(copy=True)
        if apply_arcsinh:
            data = np.arcsinh(data / float(arcsinh_cofactor))
        if not apply_z1:
            return {"rows": int(frame.shape[0]), "means": None, "stds": None}
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        stds[~np.isfinite(stds) | (stds == 0)] = 1e-9
        return {"rows": int(frame.shape[0]), "means": means, "stds": stds}

    groups: dict[Any, dict[str, Any]] = {}
    if group_col and group_col in df.columns:
        keys = df[group_col].fillna("__missing__").astype(str)
        for key, frame in df.groupby(keys, group_keys=False):
            groups[str(key)] = population_stats(frame)
    else:
        groups[None] = population_stats(df)

    return {
        "markers": valid_markers,
        "norm_cols": norm_cols,
        "suffix": suffix,
        "flags": (apply_z1, apply_z2, apply_cdf, apply_nlog),
        "group_col": group_col if (group_col and group_col in df.columns) else None,
        "groups": groups,
        "apply_arcsinh": bool(apply_arcsinh),
        "arcsinh_cofactor": float(arcsinh_cofactor),
    }


def _apply_visual_normalization_rows(
    rows: pd.DataFrame,
    stats: dict[str, Any],
) -> pd.DataFrame:
    """Add normalized columns to a handful of rows, using precomputed statistics.

    Produces the same values ``_apply_visual_normalization`` would for these
    rows, without touching the rest of the table.
    """
    valid_markers = stats["markers"]
    norm_cols = stats["norm_cols"]
    apply_z1, apply_z2, apply_cdf, apply_nlog = stats["flags"]
    epsilon = 1e-9

    work = rows.copy()
    data = work[valid_markers].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy(copy=True)
    if stats["apply_arcsinh"]:
        data = np.arcsinh(data / stats["arcsinh_cofactor"])
    missing = ~np.isfinite(data)

    group_col = stats["group_col"]
    if group_col:
        keys = work[group_col].fillna("__missing__").astype(str).to_numpy()
    else:
        keys = np.array([None] * len(work), dtype=object)

    processed = data.copy()
    if apply_z1:
        for index, key in enumerate(keys):
            entry = stats["groups"].get(key if group_col else None)
            if entry is None or entry["means"] is None:
                continue
            processed[index] = (processed[index] - entry["means"]) / entry["stds"]
        processed = np.clip(processed, -10, 10)
    if apply_z2:
        row_means = np.nanmean(processed, axis=1, keepdims=True)
        row_stds = np.nanstd(processed, axis=1, keepdims=True)
        row_stds[~np.isfinite(row_stds) | (row_stds == 0)] = epsilon
        processed = (processed - row_means) / row_stds
    if apply_cdf:
        processed = _normal_cdf_array(processed)
    if apply_nlog:
        processed = -np.log(np.maximum(1.0 - processed, epsilon))

    final = np.asarray(processed, dtype=float)
    final[missing] = np.nan

    # A group with fewer than two cells has no usable spread, which the
    # whole-table path represented as zeros.
    if apply_z1 or apply_z2:
        for index, key in enumerate(keys):
            entry = stats["groups"].get(key if group_col else None)
            if entry is not None and entry["rows"] < 2:
                final[index] = np.where(missing[index], np.nan, 0.0)

    work[norm_cols] = pd.DataFrame(final, index=work.index, columns=norm_cols)
    return work


def _sanitize_visual_filters(
    criteria: list[dict[str, Any]],
    available_columns: list[str],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for criterion in criteria:
        column = str(criterion.get("column", "")).strip()
        if not column or column not in available_columns:
            continue

        criterion_type = str(criterion.get("type", "categorical")).strip().lower()
        if criterion_type == "numeric":
            mode = str(criterion.get("mode", "between")).strip().lower()
            value = criterion.get("value")
            value_2 = criterion.get("value_2")
            if value is None:
                continue
            if mode == "between":
                if value_2 is None:
                    continue
                low = min(float(value), float(value_2))
                high = max(float(value), float(value_2))
                sanitized.append(
                    {
                        "column": column,
                        "type": "numeric",
                        "mode": "between",
                        "value": low,
                        "value_2": high,
                    }
                )
            elif mode in {"gte", "lte"}:
                sanitized.append(
                    {
                        "column": column,
                        "type": "numeric",
                        "mode": mode,
                        "value": float(value),
                    }
                )
            continue

        values = [str(value).strip() for value in criterion.get("values", []) if str(value).strip()]
        if not values:
            continue
        sanitized.append(
            {
                "column": column,
                "type": "categorical",
                "values": values,
            }
        )
    return sanitized


def _render_filter_criteria(
    key_prefix: str,
    available_columns: list[str],
    csv_path: str,
    *,
    empty_caption: str = "No filter criteria added. All rows are currently eligible.",
    add_label: str = "Add Criteria",
    clear_label: str = "Clear Criteria",
    fresh_from_disk: bool = False,
) -> list[dict[str, Any]]:
    """Render the shared add/remove filter builder and return sanitized criteria.

    Numeric columns get a Between / Above / Below control, everything else gets a
    value multiselect. All widget keys are namespaced by `key_prefix`, so each
    tab keeps its own independent stack of criteria.
    """
    rows_key = f"{key_prefix}_filter_rows"
    if rows_key not in st.session_state:
        st.session_state[rows_key] = []

    filter_rows: list[dict[str, Any]] = st.session_state[rows_key]
    if not filter_rows:
        st.caption(empty_caption)

    criteria: list[dict[str, Any]] = []
    for idx in range(len(filter_rows)):
        # One row per criterion: column, condition and values side by side,
        # so the controls stay together instead of the right-hand stack
        # running off the page.
        criteria_col, editor_col, remove_col = st.columns(
            [1.4, 3.2, 0.5], vertical_alignment="bottom"
        )
        with criteria_col:
            selected_filter_col = st.selectbox(
                f"Criteria {idx + 1} — column",
                available_columns,
                index=available_columns.index(filter_rows[idx].get("column")) if filter_rows[idx].get("column") in available_columns else 0,
                key=f"{key_prefix}_filter_col_{idx}",
            )
        with remove_col:
            if st.button("✕", key=f"{key_prefix}_filter_remove_{idx}", help="Remove this criterion"):
                rows = list(st.session_state.get(rows_key, []))
                if idx < len(rows):
                    rows.pop(idx)
                    st.session_state[rows_key] = rows
                    st.rerun()
        profile = (
            _get_column_profile_fresh(csv_path, selected_filter_col)
            if fresh_from_disk
            else _get_column_profile_cached(csv_path, selected_filter_col)
        )
        if profile.get("is_numeric"):
            with editor_col:
                min_value = float(profile.get("min", 0.0))
                max_value = float(profile.get("max", 0.0))
                mode_col, val_col1, val_col2 = st.columns([1.6, 1, 1], vertical_alignment="bottom")
                with mode_col:
                    numeric_mode = st.selectbox(
                        "Condition",
                        ["Between", "Above or Equal", "Below or Equal"],
                        index=0 if filter_rows[idx].get("mode") == "between" or not filter_rows[idx].get("mode") else 1 if filter_rows[idx].get("mode") == "gte" else 2,
                        key=f"{key_prefix}_filter_mode_{idx}",
                        help=f"Values in this column run from {min_value:.4g} to {max_value:.4g}.",
                    )
                if numeric_mode == "Between":
                    with val_col1:
                        filter_min = st.number_input(
                            "Min",
                            value=float(filter_rows[idx].get("value", min_value)),
                            key=f"{key_prefix}_filter_min_{idx}",
                            format="%.4f",
                        )
                    with val_col2:
                        filter_max = st.number_input(
                            "Max",
                            value=float(filter_rows[idx].get("value_2", max_value)),
                            key=f"{key_prefix}_filter_max_{idx}",
                            format="%.4f",
                        )
                    criteria.append(
                        {
                            "column": selected_filter_col,
                            "type": "numeric",
                            "mode": "between",
                            "value": filter_min,
                            "value_2": filter_max,
                        }
                    )
                else:
                    with val_col1:
                        threshold_value = st.number_input(
                            "Threshold",
                            value=float(filter_rows[idx].get("value", min_value if numeric_mode == "Above or Equal" else max_value)),
                            key=f"{key_prefix}_filter_threshold_{idx}",
                            format="%.4f",
                        )
                    criteria.append(
                        {
                            "column": selected_filter_col,
                            "type": "numeric",
                            "mode": "gte" if numeric_mode == "Above or Equal" else "lte",
                            "value": threshold_value,
                        }
                    )
        else:
            categorical_values = (
                _get_unique_values_fresh(csv_path, selected_filter_col)
                if fresh_from_disk
                else _get_unique_values_cached(csv_path, selected_filter_col)
            )
            with editor_col:
                selected_values = st.multiselect(
                    f"Values ({len(categorical_values)} available)",
                    categorical_values,
                    default=[
                        value
                        for value in filter_rows[idx].get("values", [])
                        if value in categorical_values
                    ],
                    key=f"{key_prefix}_filter_values_{idx}",
                )
            criteria.append(
                {
                    "column": selected_filter_col,
                    "type": "categorical",
                    "values": selected_values,
                }
            )

    filter_action_col1, filter_action_col2 = st.columns(2)
    if filter_action_col1.button(add_label, key=f"{key_prefix}_add_criteria"):
        st.session_state[rows_key] = st.session_state.get(rows_key, []) + [
            {
                "column": available_columns[0] if available_columns else "",
                "type": "categorical",
                "values": [],
            }
        ]
        st.rerun()
    if filter_action_col2.button(clear_label, key=f"{key_prefix}_clear_criteria"):
        st.session_state[rows_key] = []
        st.rerun()

    return _sanitize_visual_filters(criteria, available_columns)


def _apply_visual_filters(df: pd.DataFrame, criteria: list[dict[str, Any]]) -> pd.DataFrame:
    filtered_df = df.copy()
    for criterion in criteria:
        column = criterion["column"]
        if column not in filtered_df.columns:
            continue

        if criterion.get("type") == "numeric":
            numeric = pd.to_numeric(filtered_df[column], errors="coerce")
            mode = criterion.get("mode")
            if mode == "between":
                mask = numeric.between(float(criterion["value"]), float(criterion["value_2"]))
            elif mode == "gte":
                mask = numeric >= float(criterion["value"])
            else:
                mask = numeric <= float(criterion["value"])
            filtered_df = filtered_df.loc[mask.fillna(False)].copy()
            continue

        mask = filtered_df[column].astype(str).isin([str(value) for value in criterion.get("values", [])])
        filtered_df = filtered_df.loc[mask].copy()

    return filtered_df


def _build_manual_reference_pending_assignments(
    visual_results: list[dict[str, Any]],
    selections: dict[str, Any],
    default_output_col: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turn the on-screen phenotype choices into write instructions.

    ``selections`` maps a result key to whatever the phenotype dropdown holds. A cell that
    already carries a label and is set back to Unassigned is a request to remove that
    label; one that never had a label and is left Unassigned is simply undecided and is
    skipped.
    """
    assignments: list[dict[str, Any]] = []
    counts = {"clearing": 0, "unassigned": 0, "pending_new": 0}

    for result in visual_results:
        result_key = result.get("result_key")
        selected = _normalize_visual_phenotype_label(selections.get(result_key, "Unassigned"))

        if selected == "Add New Phenotype":
            counts["pending_new"] += 1
            continue

        current_stored_label = _normalize_visual_phenotype_label(
            result.get("current_phenotype")
        )
        clears_label = (
            not _has_visual_manual_reference_assignment(selected)
            and bool(current_stored_label)
        )
        if not clears_label and not _has_visual_manual_reference_assignment(selected):
            counts["unassigned"] += 1
            continue

        if clears_label:
            counts["clearing"] += 1

        assignments.append(
            {
                "batch_column": str(result.get("batch_col")),
                "batch_value": result.get("batch_value"),
                "x_column": str(result.get("default_x_col")).strip(),
                "x_value": result.get("x_value"),
                "y_column": str(result.get("default_y_col")).strip(),
                "y_value": result.get("y_value"),
                "output_column": str(result.get("default_output_col", default_output_col)).strip(),
                "phenotype_value": selected,
                "cell_id_column": str(result.get("cell_id_col") or "") or None,
                "cell_id_value": result.get("cell_id_value"),
                "clear_existing": clears_label,
                "result_key": result_key,
            }
        )

    return assignments, counts


def _build_manual_reference_save_receipt(
    assignments: list[dict[str, Any]],
    visual_results: list[dict[str, Any]],
    *,
    verified_count: int,
    target_path: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a session-only, user-visible receipt for a verified CSV save."""
    result_lookup = {
        str(result.get("result_key")): result for result in visual_results
    }
    details: list[dict[str, Any]] = []
    changed_count = 0
    cleared_count = 0
    unchanged_count = 0

    for assignment in assignments:
        result_key = str(assignment.get("result_key", ""))
        result = result_lookup.get(result_key, {})
        previous = _normalize_visual_phenotype_label(
            result.get("current_phenotype")
        )
        selected = _normalize_visual_phenotype_label(
            assignment.get("phenotype_value")
        )
        clearing = bool(assignment.get("clear_existing")) or not (
            _has_visual_manual_reference_assignment(selected)
        )

        if clearing:
            action = "Removed label"
            saved_value = "<blank>"
            cleared_count += 1
        elif previous == selected:
            action = "Verified unchanged"
            saved_value = selected
            unchanged_count += 1
        else:
            action = "Assigned" if not previous else "Changed label"
            saved_value = selected
            changed_count += 1

        details.append(
            {
                "Batch": assignment.get("batch_value"),
                "Cell ID": assignment.get("cell_id_value"),
                "Previous": previous or "<blank>",
                "Saved value": saved_value,
                "Action": action,
                "Output column": assignment.get("output_column"),
            }
        )

    # The old wording said "CSV row(s)", which was true when every save rewrote
    # the table. Assignments are now held beside it until they are written in,
    # so the message has to say where they actually are.
    try:
        from spacemap import labels

        total_pending = labels.pending_count(target_path)
    except Exception:
        total_pending = 0

    message = (
        f"Saved and verified: {int(verified_count)} cell(s). "
        f"{changed_count} assigned/changed, {cleared_count} label(s) removed, "
        f"and {unchanged_count} unchanged value(s) verified. "
        f"{total_pending} label(s) are now held alongside the CSV and shown "
        "everywhere in SPACEMAP; use \"Write labels into the CSV\" below to put "
        "them into the file itself."
    )
    return {
        "message": message,
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_path": str(target_path),
        "verified_count": int(verified_count),
        "changed_count": changed_count,
        "cleared_count": cleared_count,
        "unchanged_count": unchanged_count,
        "details": details,
    }


def _render_label_commit_controls(table_path: str | None, *, key_prefix: str) -> None:
    """Offer to fold saved labels into the CSV, and say what is outstanding.

    Assignments are held beside the table so a save takes under a second
    instead of rewriting a multi-gigabyte file. They are applied to everything
    SPACEMAP reads, but the CSV itself only changes when this is used.
    """
    if not table_path:
        return
    try:
        from spacemap import labels

        pending = labels.pending_count(table_path)
        columns = labels.pending_columns(table_path)
    except Exception:
        return
    if not pending:
        return

    st.info(
        f"**{pending} label(s)** are stored alongside the CSV in "
        f"{', '.join(f'`{name}`' for name in columns)}. Every SPACEMAP screen "
        "already shows them. The CSV file itself does not contain them yet."
    )
    left, right = st.columns([2, 1])
    if left.button(
        "Write labels into the CSV",
        key=f"{key_prefix}_commit_labels",
        type="primary",
        width="stretch",
        help="Rewrites the unified CSV and refreshes the fast copy. On a large "
             "table this takes several minutes; the labels are safe until then.",
    ):
        status = st.empty()
        with st.spinner("Writing the CSV and refreshing the fast copy..."):
            try:
                from spacemap import labels as label_store

                report = label_store.commit(table_path, progress=lambda m: status.caption(m))
            except Exception as exc:
                status.empty()
                st.error(f"Could not write the labels: {exc}")
            else:
                status.empty()
                st.success(
                    f"Wrote {report['applied']} label(s) into "
                    f"{', '.join(report['columns'])} in the CSV."
                )
                st.rerun()
    if right.button("Discard", key=f"{key_prefix}_discard_labels", width="stretch"):
        from spacemap import labels as label_store

        label_store.clear(table_path)
        st.rerun()


def _render_manual_reference_save_receipt(
    receipt: dict[str, Any],
    *,
    details_expanded: bool,
    show_dismiss: bool,
) -> None:
    """Render a verified save using the standard green Streamlit status box."""
    st.success(f"✅ {receipt['message']}")
    st.caption(
        f"Saved at {receipt.get('timestamp')} | "
        f"Unified CSV: {receipt.get('target_path')}"
    )

    # The control for writing labels into the CSV belongs here, next to the
    # save that produced them, rather than only in the input panel further up.
    _render_label_commit_controls(receipt.get("target_path"), key_prefix="mr_receipt")
    receipt_details = receipt.get("details", [])
    if receipt_details:
        with st.expander(
            f"Show {len(receipt_details)} saved/removed cell(s)",
            expanded=details_expanded,
        ):
            st.dataframe(
                pd.DataFrame(receipt_details),
                hide_index=True,
                width="stretch",
            )
    if show_dismiss and st.button(
        "Dismiss Save Confirmation",
        key="_tools_mr_vv_dismiss_save_receipt",
    ):
        st.session_state.pop("_tools_mr_vv_save_receipt", None)
        st.rerun()


def _accumulate_shown_cell_keys(previous_keys, newly_shown_keys) -> list[str]:
    """Every cell shown since the last fresh selection.

    "Get More Cells" must never hand back a cell already seen in this run, so the keys
    accumulate. "Select and Visualize Cells" clears the list instead, which is what makes
    it free to return the same cells again.
    """
    return sorted(set(previous_keys or []) | set(newly_shown_keys or []))


def _compute_top_markers_for_row(
    row: pd.Series,
    marker_cols: list[str],
    norm_cols: list[str],
    top_n: int = 5,
    *,
    affix_position: str | None = None,
    correction_affix: str | None = None,
) -> list[tuple[str, str, float]]:
    collapsed: dict[str, tuple[str, str, float]] = {}
    for raw_marker, norm_col in zip(marker_cols, norm_cols, strict=False):
        score = pd.to_numeric(pd.Series([row.get(norm_col)]), errors="coerce").iloc[0]
        if pd.isna(score):
            # A marker with no value must not compete for a place in the top N. Scoring it
            # as 0.0 let missing markers outrank real, genuinely negative ones — z-scored
            # and CDF-normalised values are routinely below zero.
            continue
        score_value = float(score)
        clean_name = _clean_visual_marker_name(raw_marker, affix_position, correction_affix)
        existing = collapsed.get(clean_name)
        if existing is None or score_value > existing[2]:
            collapsed[clean_name] = (raw_marker, clean_name, score_value)

    return sorted(collapsed.values(), key=lambda item: item[2], reverse=True)[:top_n]


def _resolve_overlay_channels(
    cell_ctx: dict[str, Any],
    marker_names: list[str],
    *,
    affix_position: str | None = None,
    correction_affix: str | None = None,
) -> tuple[list[int], list[str]]:
    available_channels = [str(name) for name in cell_ctx.get("channel_names", [])]
    if not available_channels:
        return [], []

    channel_lookup = {channel.lower().strip(): idx for idx, channel in enumerate(available_channels)}
    marker_indices: list[int] = []
    names_for_plot: list[str] = []
    used_indices: set[int] = set()

    for marker_name in marker_names:
        clean_name = _clean_visual_marker_name(marker_name, affix_position, correction_affix)
        target = clean_name.lower().strip()
        found_idx = channel_lookup.get(target)
        if found_idx is None:
            for idx, channel_name in enumerate(available_channels):
                normalized_channel = channel_name.lower().strip()
                if target and (target in normalized_channel or normalized_channel in target):
                    found_idx = idx
                    break

        if found_idx is None or found_idx in used_indices:
            continue

        used_indices.add(found_idx)
        marker_indices.append(found_idx)
        names_for_plot.append(clean_name)

    return marker_indices, names_for_plot


def _save_visual_manual_reference_figure(
    fig: Any,
    output_dir: str,
    batch_value: Any,
    cell_id_value: Any,
    x_value: Any,
    y_value: Any,
) -> str:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    batch_token = _sanitize_filename_token(str(batch_value))
    cell_token = _sanitize_filename_token(str(cell_id_value if pd.notna(cell_id_value) else "unknown"))
    x_token = _sanitize_filename_token(f"{float(x_value):.3f}")
    y_token = _sanitize_filename_token(f"{float(y_value):.3f}")
    file_name = f"mrvv_{batch_token}_cell_{cell_token}_x_{x_token}_y_{y_token}.png"
    save_path = target_dir / file_name
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return str(save_path)


_VISUAL_MISSING_PHENOTYPE_TOKENS = frozenset(
    {"", "nan", "none", "null", "<na>", "na", "n/a", "nat"}
)

# These choices all mean that a cell has no defensible manual-reference label.
# They clear the CSV field instead of becoming phenotype strings. "Others" is
# intentionally absent because it is a real broad phenotype in SPACEMAP.
_VISUAL_CLEAR_PHENOTYPE_TOKENS = frozenset(
    {
        "unassigned",
        "un assigned",
        "unclassified",
        "un classified",
        "unlabeled",
        "un labeled",
        "unlabelled",
        "un labelled",
        "unknown",
        "not assigned",
        "not classified",
        "no phenotype",
    }
)


def _normalize_visual_phenotype_label(value: Any) -> str:
    """Return a usable phenotype label, or an empty string for missing values."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in _VISUAL_MISSING_PHENOTYPE_TOKENS:
        return ""
    return text


def _has_visual_manual_reference_assignment(value: Any) -> bool:
    text = _normalize_visual_phenotype_label(value)
    normalized_token = re.sub(r"[\s_-]+", " ", text.casefold()).strip()
    return bool(text) and normalized_token not in _VISUAL_CLEAR_PHENOTYPE_TOKENS


_VISUAL_MANUAL_REFERENCE_RUNTIME_CACHE_KEYS = frozenset(
    {
        "tools_mr_vv_results",
        "tools_mr_vv_seen_keys",
        "tools_mr_vv_fetch_more",
        "tools_mr_vv_last_file",
    }
)
_VISUAL_MANUAL_REFERENCE_RUNTIME_STATE_VERSION = 2


def _is_visual_manual_reference_runtime_cache_key(key: Any) -> bool:
    """Return whether a TOOLS cache key is per-cell runtime state, not a setting."""
    key_text = str(key)
    return (
        key_text in _VISUAL_MANUAL_REFERENCE_RUNTIME_CACHE_KEYS
        or key_text.startswith("tools_mr_vv_assign_pheno_")
        or key_text.startswith("tools_mr_vv_assign_save_")
    )


def _prune_visual_manual_reference_runtime_cache(cache: dict[str, Any]) -> None:
    """Remove stale cards, seen-cell history, and dropdowns from persistent settings."""
    for key in list(cache):
        if _is_visual_manual_reference_runtime_cache_key(key):
            cache.pop(key, None)


def _clear_visual_manual_reference_assignment_widget_state() -> None:
    """Make newly fetched cards initialize from labels in the current CSV."""
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if key_text.startswith("tools_mr_vv_assign_pheno_") or key_text.startswith(
            "tools_mr_vv_assign_save_"
        ):
            st.session_state.pop(key, None)


def _clear_visual_manual_reference_csv_caches() -> None:
    _load_unified_csv_cached.clear()
    _get_unique_values_cached.clear()
    _get_columns_cached.clear()
    _get_column_profile_cached.clear()


def _write_verified_visual_manual_reference_updates(
    unified_df: pd.DataFrame,
    unified_csv_path: str,
    expected_updates: list[tuple[int, str, str]],
) -> int:
    """Record assignments in the label sidecar and verify they read back.

    This used to rewrite the whole unified table for every save: measured at
    about 13 minutes for a 13 GB CSV, plus another 2.7 minutes to rebuild the
    Parquet copy the rewrite invalidated. Assignments now append to a small
    sidecar beside the table, which takes well under a second, and every read
    merges them, so the interface shows current values immediately.

    Use "Write labels into the CSV" in the fast-loading panel to fold them into
    the CSV and Parquet when a labelling session is finished.
    """
    if not expected_updates:
        return 0

    from spacemap import labels

    stored = labels.append(
        unified_csv_path,
        [
            {"row": row_index, "column": output_column, "value": expected_phenotype}
            for row_index, output_column, expected_phenotype in expected_updates
        ],
    )

    # Read the sidecar back and confirm every cell holds what was asked for,
    # keeping the guarantee the previous whole-file verification provided.
    saved = labels.load(unified_csv_path)
    if saved.empty:
        raise IOError("Manual Reference labels could not be read back after saving.")
    lookup = {
        (int(row.row), str(row.column)): _normalize_visual_phenotype_label(row.value)
        for row in saved.itertuples(index=False)
    }
    verification_errors: list[str] = []
    for row_index, output_column, expected_phenotype in expected_updates:
        actual = lookup.get((int(row_index), str(output_column)))
        if actual != expected_phenotype:
            verification_errors.append(
                f"row {row_index}, {output_column}: expected '{expected_phenotype}', "
                f"found '{actual or '<blank>'}'"
            )
    if verification_errors:
        preview = "; ".join(verification_errors[:3])
        raise IOError(f"Manual Reference write verification failed: {preview}")

    _clear_visual_manual_reference_csv_caches()
    return stored


def _exclude_existing_visual_manual_reference_assignments(
    df: pd.DataFrame,
    output_column: str | None,
) -> pd.DataFrame:
    resolved_output_column = str(output_column or "").strip()
    if not resolved_output_column or resolved_output_column not in df.columns:
        return df.copy()

    assigned_mask = df[resolved_output_column].apply(_has_visual_manual_reference_assignment)
    return df.loc[~assigned_mask].copy()


def _match_visual_manual_reference_row(
    unified_df: pd.DataFrame,
    *,
    batch_column: str,
    batch_value: Any,
    x_column: str,
    x_value: Any,
    y_column: str,
    y_value: Any,
    cell_id_column: str | None = None,
    cell_id_value: Any | None = None,
) -> pd.Series:
    """Locate one cell, preferring its batch/Cell-ID identity over float coordinates."""
    batch_mask = unified_df[batch_column].astype(str) == str(batch_value)
    has_cell_id = (
        bool(cell_id_column)
        and cell_id_column in unified_df.columns
        and pd.notna(cell_id_value)
        and str(cell_id_value).strip() not in {"", "unknown"}
    )
    if has_cell_id:
        identity_mask = batch_mask & (
            unified_df[str(cell_id_column)].astype(str) == str(cell_id_value)
        )
        # Cell IDs are the most stable identifier and avoid float round-trip issues.
        if int(identity_mask.sum()) <= 1:
            return identity_mask
        candidate_mask = identity_mask
    else:
        candidate_mask = batch_mask

    x_numeric = pd.to_numeric(unified_df[x_column], errors="coerce")
    y_numeric = pd.to_numeric(unified_df[y_column], errors="coerce")
    coordinate_mask = (
        candidate_mask
        & np.isclose(x_numeric, float(x_value), rtol=0.0, atol=1e-6)
        & np.isclose(y_numeric, float(y_value), rtol=0.0, atol=1e-6)
    )
    return coordinate_mask


def _update_visual_manual_reference_assignment(
    *,
    unified_csv_path: str,
    batch_column: str,
    batch_value: Any,
    x_column: str,
    x_value: Any,
    y_column: str,
    y_value: Any,
    output_column: str,
    phenotype_value: str,
    cell_id_column: str | None = None,
    cell_id_value: Any | None = None,
    allow_clear: bool = False,
) -> int:
    if not unified_csv_path or not os.path.exists(unified_csv_path):
        raise FileNotFoundError("Unified CSV not found.")

    output_column = str(output_column).strip()
    if not output_column:
        raise ValueError("Output column name is required.")
    phenotype_label = _normalize_visual_phenotype_label(phenotype_value)
    if phenotype_label == "Add New Phenotype":
        raise ValueError("Create and select the new phenotype before saving.")
    # Unassigned means "this cell should carry no label". Reviewing existing labels is
    # pointless without a way to take a wrong one off, so with allow_clear the cell is
    # blanked rather than the write being refused.
    clearing_label = not _has_visual_manual_reference_assignment(phenotype_label)
    if clearing_label and not allow_clear:
        raise ValueError("Choose a concrete phenotype before saving; missing and Unassigned values are not written.")

    unified_df = _dataio_load(unified_csv_path)
    for column in (batch_column, x_column, y_column):
        if column not in unified_df.columns:
            raise KeyError(f"'{column}' column is missing from the unified CSV.")

    if output_column not in unified_df.columns:
        unified_df[output_column] = pd.Series(pd.NA, index=unified_df.index, dtype="object")
    else:
        unified_df[output_column] = unified_df[output_column].astype("object")

    mask = _match_visual_manual_reference_row(
        unified_df,
        batch_column=batch_column,
        batch_value=batch_value,
        x_column=x_column,
        x_value=x_value,
        y_column=y_column,
        y_value=y_value,
        cell_id_column=cell_id_column,
        cell_id_value=cell_id_value,
    )

    matched_rows = int(mask.sum())
    if matched_rows == 0:
        raise ValueError("No row matched the selected batch/X/Y coordinates in the unified CSV.")
    if matched_rows > 1:
        raise ValueError("Multiple rows matched the selected batch/X/Y coordinates. Refine the coordinate columns or Cell ID column.")

    row_index = int(unified_df.index[mask][0])
    unified_df.at[row_index, output_column] = "" if clearing_label else phenotype_label
    return _write_verified_visual_manual_reference_updates(
        unified_df,
        unified_csv_path,
        [(row_index, output_column, "" if clearing_label else phenotype_label)],
    )


def _update_visual_manual_reference_assignments(
    *,
    unified_csv_path: str,
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    if not unified_csv_path or not os.path.exists(unified_csv_path):
        raise FileNotFoundError("Unified CSV not found.")

    unified_df = _dataio_load(unified_csv_path)
    pending_updates: list[tuple[int, str, str]] = []
    cleared_count = 0
    skipped_unassigned = 0
    skipped_pending = 0
    errors: list[str] = []

    for idx, assignment in enumerate(assignments, start=1):
        phenotype_value = _normalize_visual_phenotype_label(
            assignment.get("phenotype_value", "")
        )
        if phenotype_value == "Add New Phenotype":
            skipped_pending += 1
            continue
        clearing_label = not _has_visual_manual_reference_assignment(phenotype_value)
        if clearing_label and not bool(assignment.get("clear_existing")):
            skipped_unassigned += 1
            continue

        output_column = str(assignment.get("output_column", "")).strip()
        if not output_column:
            errors.append(f"Cell {idx}: output column name is required.")
            continue

        for column in (assignment.get("batch_column"), assignment.get("x_column"), assignment.get("y_column")):
            if column not in unified_df.columns:
                errors.append(f"Cell {idx}: '{column}' column is missing from the unified CSV.")
                break
        else:
            if output_column not in unified_df.columns:
                unified_df[output_column] = pd.Series(pd.NA, index=unified_df.index, dtype="object")
            elif unified_df[output_column].dtype != "object":
                unified_df[output_column] = unified_df[output_column].astype("object")

            cell_id_column = str(assignment.get("cell_id_column") or "").strip()
            cell_id_value = assignment.get("cell_id_value")
            mask = _match_visual_manual_reference_row(
                unified_df,
                batch_column=str(assignment.get("batch_column")),
                batch_value=assignment.get("batch_value"),
                x_column=str(assignment.get("x_column")),
                x_value=assignment.get("x_value"),
                y_column=str(assignment.get("y_column")),
                y_value=assignment.get("y_value"),
                cell_id_column=cell_id_column or None,
                cell_id_value=cell_id_value,
            )

            matched_rows = int(mask.sum())
            if matched_rows == 0:
                errors.append(f"Cell {idx}: no row matched the selected batch/X/Y coordinates in the unified CSV.")
                continue
            if matched_rows > 1:
                errors.append(
                    f"Cell {idx}: multiple rows matched the selected batch/X/Y coordinates. Refine the coordinate columns or Cell ID column."
                )
                continue

            row_index = int(unified_df.index[mask][0])
            if clearing_label:
                cleared_count += 1
                pending_updates.append((row_index, output_column, ""))
            else:
                pending_updates.append((row_index, output_column, phenotype_value))

    if errors:
        preview = "; ".join(errors[:3])
        if len(errors) > 3:
            preview = f"{preview}; and {len(errors) - 3} more issue(s)"
        raise ValueError(preview)

    # Last assignment wins if the same cell/output pair somehow appears twice.
    deduplicated_updates: dict[tuple[int, str], str] = {}
    for row_index, output_column, phenotype_value in pending_updates:
        deduplicated_updates[(row_index, output_column)] = phenotype_value
    expected_updates = [
        (row_index, output_column, phenotype_value)
        for (row_index, output_column), phenotype_value in deduplicated_updates.items()
    ]
    for row_index, output_column, phenotype_value in expected_updates:
        unified_df.at[row_index, output_column] = phenotype_value

    verified_count = _write_verified_visual_manual_reference_updates(
        unified_df,
        unified_csv_path,
        expected_updates,
    )
    return {
        "saved_count": len(expected_updates),
        "verified_count": verified_count,
        "cleared_count": cleared_count,
        "skipped_unassigned": skipped_unassigned,
        "skipped_pending": skipped_pending,
        "output_columns": sorted({column for _, column, _ in expected_updates}),
        "target_path": str(Path(unified_csv_path).resolve()),
    }


def _scan_geojson_summary(
    geojson_dir: str,
    file_pattern: str,
    batch_ids: list[str],
    batch_id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan GeoJSON files, summarize classifications, and propose batch assignments."""
    if not geojson_dir or not os.path.isdir(geojson_dir):
        return pd.DataFrame(), pd.DataFrame()

    import glob

    _ = batch_id_col
    files = sorted(glob.glob(os.path.join(geojson_dir, file_pattern)))
    records: list[dict[str, Any]] = []
    assignment_records: list[dict[str, Any]] = []
    for path in files:
        annotations_df = _extract_geojson_annotations(path)
        prefix = _extract_geojson_prefix(path)
        assigned_batch, match_source, match_score = _best_batch_match(prefix, batch_ids)

        classified_cells = 0
        if not annotations_df.empty:
            classified_cells = int(
                annotations_df["classification"].astype(str).str.casefold().ne("unclassified").sum()
            )

        assignment_records.append(
            {
                "file": os.path.basename(path),
                "path": path,
                "derived_batch": prefix,
                "auto_batch": assigned_batch or "",
                "assigned_batch": assigned_batch or "",
                "match_source": match_source,
                "match_score": match_score,
                "annotated_cells": int(len(annotations_df)),
                "classified_cells": classified_cells,
            }
        )

        if annotations_df.empty:
            continue

        counts = annotations_df["classification"].astype(str).value_counts().to_dict()
        total = len(annotations_df)
        for cls_name, count in counts.items():
            records.append(
                {
                    "file": os.path.basename(path),
                    "prefix": prefix,
                    "assigned_batch": assigned_batch or "",
                    "match_status": match_source,
                    "classification": cls_name,
                    "count": int(count),
                    "% total": f"{(count / total) * 100:.1f}%",
                    "total_cells": total,
                }
            )

    if not records:
        return pd.DataFrame(), _ensure_mr_assignment_columns(pd.DataFrame(assignment_records))

    return pd.DataFrame(records), _ensure_mr_assignment_columns(pd.DataFrame(assignment_records))


def _extract_geojson_prefix(filename: str) -> str:
    base = os.path.basename(filename)
    base = re.sub(r"\.geojson$", "", base)
    base = re.sub(r"\.ome.*$", "", base)
    base = re.sub(r"_cell_boundaries.*$", "", base)
    return base



# The Phenotype GeoJSON tab draws one color picker per label; beyond this many the
# panel becomes unusable, so the remaining labels fall back to the default palette.
_MAX_COLOR_PICKERS = 40
_MAX_PNG_PREVIEWS = 8


# Label written into the GeoJSON when comparing two phenotype columns. It only
# ever exists on the in-memory frame, never in the unified CSV.
_AGREEMENT_LABEL_COL = "__pg_agreement__"


def _parse_na_tokens(text: str) -> list[str]:
    """Split the "treated as missing" input into comparison tokens."""
    tokens = [part.strip() for part in str(text or "").split(",")]
    return [token for token in tokens if token]


def _column_selectbox(
    label: str,
    options: list[str],
    preferred: str,
    key: str,
    help_text: str | None = None,
) -> str:
    """Selectbox over CSV columns that falls back to `preferred` when empty."""
    if not options:
        return preferred
    index = options.index(preferred) if preferred in options else 0
    return st.selectbox(label, options, index=index, key=key, help=help_text)


# One project-wide label -> color registry, so a phenotype keeps the same color
# across every column and every run rather than being re-drawn from the palette
# in whatever order the labels happened to be discovered.
_PG_COLOR_REGISTRY_KEY = "tools_pg_label_colors"


def _color_widget_key(label: str) -> str:
    """Stable session-state key for one label's color picker.

    Keyed on the label alone, so the same phenotype is one widget no matter which
    column is being tagged. Slugging alone is not enough -- labels such as
    "T cell" and "T-cell" collapse to the same slug and would then share a single
    picker -- so a digest of the raw label keeps every key distinct. The prefix
    deliberately avoids "tools_", which would make render_tools dump every picker
    into the params file alongside the registry that actually stores them.
    """
    digest = hashlib.md5(str(label).encode("utf-8")).hexdigest()[:8]
    return f"pg_color_{_slugify(label)}__{digest}"


def _agreement_summary(results: list[dict[str, Any]], meta: dict[str, Any]) -> pd.DataFrame:
    """Per-section match/mismatch rates for a comparison run, plus a TOTAL row.

    Ignored cells -- the reference values held back from scoring, such as
    "Others" -- are reported separately and left out of the percentages, so the
    rate answers "of the calls we could score, how many did the comparison get
    right".
    """
    match_label = str(meta.get("match_label", "Match"))
    mismatch_label = str(meta.get("mismatch_label", "Mismatch"))
    ignored_labels = {str(value) for value in meta.get("passthrough", [])}

    rows: list[dict[str, Any]] = []
    for result in results:
        counts = result.get("label_counts") or {}
        matched = int(counts.get(match_label, 0))
        mismatched = int(counts.get(mismatch_label, 0))
        ignored = sum(int(count) for label, count in counts.items() if label in ignored_labels)
        rows.append(
            {
                "batch_id": result.get("batch_id", ""),
                "cells_tagged": matched + mismatched + ignored,
                "cells_scored": matched + mismatched,
                match_label: matched,
                mismatch_label: mismatched,
                "ignored": ignored,
                "not_scored": int(result.get("non_assigned", 0)),
            }
        )

    if not rows:
        return pd.DataFrame()

    total = {key: sum(int(row[key]) for row in rows) for key in rows[0] if key != "batch_id"}
    total["batch_id"] = "TOTAL"
    rows.append(total)

    for row in rows:
        scored = row["cells_scored"]
        row[f"% {match_label}"] = round(100.0 * row[match_label] / scored, 2) if scored else 0.0
        row[f"% {mismatch_label}"] = round(100.0 * row[mismatch_label] / scored, 2) if scored else 0.0

    columns = [
        "batch_id",
        "cells_tagged",
        "cells_scored",
        match_label,
        mismatch_label,
        f"% {match_label}",
        f"% {mismatch_label}",
        "ignored",
        "not_scored",
    ]
    return pd.DataFrame(rows)[columns]


def _load_pg_color_registry(cached_params: dict[str, Any]) -> dict[str, str]:
    """Read the saved label -> color registry from the project params file.

    Colors saved by earlier versions lived in a per-column map; those are folded
    in so an existing project keeps the colors it already had.
    """
    registry = cached_params.get(_PG_COLOR_REGISTRY_KEY)
    registry = {str(k): str(v) for k, v in registry.items()} if isinstance(registry, dict) else {}

    legacy = cached_params.get("tools_pg_color_maps")
    if isinstance(legacy, dict):
        for per_column in legacy.values():
            if isinstance(per_column, dict):
                for label, color in per_column.items():
                    registry.setdefault(str(label), str(color))
    return registry


def _registry_colors_for(registry: dict[str, str], labels) -> dict[str, str]:
    """Colors already on record for `labels`, falling back to a case-insensitive hit."""
    lowered = {str(key).strip().lower(): value for key, value in registry.items()}
    resolved: dict[str, str] = {}
    for label in labels:
        text = str(label)
        color = registry.get(text) or lowered.get(text.strip().lower())
        if color:
            resolved[text] = color
    return resolved


def _remember_pg_colors(
    cached_params: dict[str, Any], registry: dict[str, str], mapping: dict[str, str]
) -> None:
    """Record the colors in use so the next run reuses them for the same labels."""
    registry.update({str(label): str(color) for label, color in mapping.items() if color})
    cached_params[_PG_COLOR_REGISTRY_KEY] = registry


def _slugify(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value))[:48]


def _reset_multiselect_when_options_change(widget_key: str, options: list) -> None:
    """Drop a stored multiselect value when its option list changes.

    Streamlit applies `default=` only the first time a widget is created. A
    multiselect drawn before its options were scanned therefore stores an empty
    selection and keeps it even once the options appear, so "select all by
    default" silently stops working. Clearing the stored value lets the new
    default take effect.
    """
    signature_key = f"__options__{widget_key}"
    signature = tuple(str(option) for option in options)
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state.pop(widget_key, None)


def _describe_filter_criterion(criterion: dict[str, Any]) -> str:
    """One-line human summary of a sanitized filter criterion."""
    column = criterion.get("column", "")
    if criterion.get("type") == "numeric":
        mode = criterion.get("mode")
        if mode == "between":
            return f"{column} in [{criterion['value']:.4g}, {criterion['value_2']:.4g}]"
        symbol = "≥" if mode == "gte" else "≤"
        return f"{column} {symbol} {criterion['value']:.4g}"
    values = [str(value) for value in criterion.get("values", [])]
    shown = ", ".join(values[:5]) + ("..." if len(values) > 5 else "")
    return f"{column} in ({shown})"


def _collect_geojson_entries(geojson_dir: str, pattern: str) -> list[dict[str, str]]:
    """List GeoJSON files in a directory and its `geojson/` subfolder.

    Segmentation output lands either directly in the chosen folder or in a
    `geojson/` subfolder depending on the segmentation backend, so both are searched.
    """
    entries: list[dict[str, str]] = []
    if not geojson_dir or not os.path.isdir(geojson_dir):
        return entries

    import glob

    seen: set[str] = set()
    search_dirs = [geojson_dir]
    nested = os.path.join(geojson_dir, "geojson")
    if os.path.isdir(nested):
        search_dirs.append(nested)
    for search_dir in search_dirs:
        for path in sorted(glob.glob(os.path.join(search_dir, pattern or "*.geojson"))):
            if path in seen:
                continue
            seen.add(path)
            entries.append(
                {
                    "path": path,
                    "file": os.path.basename(path),
                    "prefix": _extract_geojson_prefix(path),
                }
            )
    return entries


def _normalize_batch_token(value: str) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"\.geojson$", "", token)
    token = re.sub(r"\.ome.*$", "", token)
    token = re.sub(r"_cell_boundaries.*$", "", token)
    return re.sub(r"[^a-z0-9]+", "", token)


def _best_batch_match(geojson_prefix: str, batch_ids: list[str]) -> tuple[str | None, str, int]:
    prefix = str(geojson_prefix or "").strip()
    if not prefix:
        return None, "unmatched", -1

    normalized_prefix = _normalize_batch_token(prefix)
    best_batch = None
    best_source = "unmatched"
    best_score = -1

    for batch_id in batch_ids:
        batch_text = str(batch_id).strip()
        if not batch_text:
            continue

        normalized_batch = _normalize_batch_token(batch_text)
        score = -1
        source = "unmatched"

        if batch_text == prefix:
            score = 10_000 + len(prefix)
            source = "exact"
        elif normalized_batch and normalized_batch == normalized_prefix:
            score = 9_000 + len(normalized_batch)
            source = "normalized exact"
        elif prefix in batch_text or batch_text in prefix:
            score = 7_000 + len(os.path.commonprefix([prefix, batch_text]))
            source = "substring"
        elif normalized_prefix and (normalized_prefix in normalized_batch or normalized_batch in normalized_prefix):
            score = 6_000 + len(os.path.commonprefix([normalized_prefix, normalized_batch]))
            source = "normalized substring"
        else:
            common_len = len(os.path.commonprefix([normalized_prefix, normalized_batch]))
            max_len = max(len(normalized_prefix), len(normalized_batch), 1)
            similarity = common_len / max_len
            if common_len >= 4 and similarity >= 0.7:
                score = 1_000 + common_len
                source = "prefix"

        if score > best_score:
            best_batch = batch_text
            best_source = source
            best_score = score

    return best_batch, best_source, best_score


def _extract_geojson_annotations(geojson_path: str) -> pd.DataFrame:
    columns = ["feature_index", "x_centroid", "y_centroid", "classification", "area_pixels"]
    if shape is None or not geojson_path or not os.path.exists(geojson_path):
        return pd.DataFrame(columns=columns)

    try:
        with open(geojson_path, "r") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame(columns=columns)

    features = data.get("features", []) if isinstance(data, dict) else []
    records: list[dict[str, Any]] = []
    for idx, feature in enumerate(features):
        geom_data = feature.get("geometry") if isinstance(feature, dict) else None
        if not geom_data or geom_data.get("type") not in {"Polygon", "MultiPolygon"}:
            continue

        try:
            polygon = shape(geom_data)
            centroid = polygon.centroid
        except Exception:
            continue

        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        classification_data = properties.get("classification", {})
        if isinstance(classification_data, dict):
            classification = classification_data.get("name", "unclassified")
        elif isinstance(classification_data, str):
            classification = classification_data
        else:
            classification = "unclassified"

        records.append(
            {
                "feature_index": idx,
                "x_centroid": float(centroid.x),
                "y_centroid": float(centroid.y),
                "classification": str(classification),
                "area_pixels": float(polygon.area),
            }
        )

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)


def _ensure_mr_assignment_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    defaults: dict[str, Any] = {
        "file": "",
        "path": "",
        "derived_batch": "",
        "auto_batch": "",
        "assigned_batch": "",
        "match_source": "unmatched",
        "match_score": -1,
        "annotated_cells": 0,
        "classified_cells": 0,
    }
    result = df.copy() if df is not None else pd.DataFrame(columns=defaults.keys())
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    result = result[list(defaults.keys())]
    for column in ["file", "path", "derived_batch", "auto_batch", "assigned_batch", "match_source"]:
        result[column] = result[column].fillna("").astype(str)
    for column, default in [("match_score", -1), ("annotated_cells", 0), ("classified_cells", 0)]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(default).astype(int)
    return result


def _apply_saved_mr_assignments(
    scanned_assignments: pd.DataFrame,
    existing_assignments: pd.DataFrame | None,
) -> pd.DataFrame:
    scanned = _ensure_mr_assignment_columns(scanned_assignments)
    existing = _ensure_mr_assignment_columns(existing_assignments)
    if scanned.empty or existing.empty:
        return scanned

    existing_lookup = {row["file"]: row for row in existing.to_dict(orient="records")}
    for idx, row in scanned.iterrows():
        previous = existing_lookup.get(row["file"])
        if previous is None:
            continue

        previous_batch = str(previous.get("assigned_batch", "")).strip()
        if not previous_batch:
            continue

        scanned.at[idx, "assigned_batch"] = previous_batch
        auto_batch = str(scanned.at[idx, "auto_batch"]).strip()
        previous_source = str(previous.get("match_source", "")).strip()
        if auto_batch and previous_batch == auto_batch and previous_source not in {"", "manual", "unmatched"}:
            scanned.at[idx, "match_source"] = previous_source
        elif auto_batch and previous_batch == auto_batch:
            scanned.at[idx, "match_source"] = str(scanned.at[idx, "match_source"]) or "exact"
        else:
            scanned.at[idx, "match_source"] = "manual"

    return scanned


def _apply_assignments_to_summary(
    summary_df: pd.DataFrame | None,
    assignment_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if summary_df is None or summary_df.empty:
        return summary_df

    result = summary_df.copy()
    if "assigned_batch" not in result.columns:
        result["assigned_batch"] = ""
    if "match_status" not in result.columns:
        result["match_status"] = ""

    assignments = _ensure_mr_assignment_columns(assignment_df)
    if assignments.empty:
        return result

    batch_lookup = assignments.set_index("file")["assigned_batch"].to_dict()
    source_lookup = assignments.set_index("file")["match_source"].to_dict()
    result["assigned_batch"] = result["file"].map(batch_lookup).fillna(result["assigned_batch"])
    result["match_status"] = result["file"].map(source_lookup).fillna(result["match_status"])
    return result


def _store_mr_scan_cache(
    cached_params: dict[str, Any],
    *,
    cached_scan_key: str,
    geojson_dir_key: str,
    unified_csv_key: str,
    file_pattern: str,
    summary_df: pd.DataFrame | None,
    assignment_df: pd.DataFrame | None,
    mapping_df: pd.DataFrame | None,
) -> None:
    if not cached_scan_key or not geojson_dir_key or not unified_csv_key or not file_pattern:
        return

    assignments = _ensure_mr_assignment_columns(assignment_df)
    payload = {
        "summary": summary_df.to_dict(orient="records") if summary_df is not None and not summary_df.empty else [],
        "assignments": assignments.to_dict(orient="records") if not assignments.empty else [],
        "mapping": mapping_df.to_dict(orient="records") if mapping_df is not None and not mapping_df.empty else [],
    }
    cached_params[cached_scan_key] = payload
    cached_params["mr_scan_last"] = {
        "geojson_dir": geojson_dir_key,
        "unified_csv_path": unified_csv_key,
        "file_pattern": file_pattern,
        **payload,
    }


def _best_geojson_match(batch_id: str, geojson_entries: list[dict[str, str]]) -> tuple[str | None, str | None, int]:
    batch_id = str(batch_id)
    best = (None, None, -1)
    for entry in geojson_entries:
        filename = entry.get("file", "")
        # Exact substring match is best
        if batch_id in filename:
            # Score could be the length of batch_id + 1000 to guarantee it wins over prefix
            score = len(batch_id) + 1000 
            if score > best[2]:
                best = (entry.get("path"), entry.get("prefix"), score)
                continue

        # fallback to prefix matching if no substring match
        prefix = entry.get("prefix", "")
        if not prefix:
            continue
        if batch_id.startswith(prefix) or prefix.startswith(batch_id):
            common_len = len(os.path.commonprefix([batch_id, prefix]))
            if common_len > best[2]:
                best = (entry.get("path"), prefix, common_len)
    return best


def _scan_geojson_classes(paths: list[str]) -> dict[str, str]:
    class_map: dict[str, str] = {}
    for path in paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        features = data.get("features", []) if isinstance(data, dict) else []
        for feat in features:
            props = feat.get("properties", {}) if isinstance(feat, dict) else {}
            cls = props.get("classification", {})
            if isinstance(cls, dict):
                name = str(cls.get("name", "unclassified"))
                color_rgb = cls.get("colorRGB")
            elif isinstance(cls, str):
                name = str(cls)
                color_rgb = None
            else:
                name = "unclassified"
                color_rgb = None
            if name not in class_map:
                if isinstance(color_rgb, int):
                    class_map[name] = f"#{color_rgb:06x}"
                else:
                    class_map[name] = "#000000"
    return class_map


def _collect_gt_prefixes(
    gt_output_dir: str,
    file_pattern: str = "*_ground_truth.csv",
    custom_prefix_pattern: str | None = None,
) -> list[str]:
    """Collect unique ground-truth prefixes from CSV filenames."""
    if not gt_output_dir or not os.path.isdir(gt_output_dir):
        return []
    import glob

    prefixes: set[str] = set()
    for path in glob.glob(os.path.join(gt_output_dir, file_pattern)):
        base_name = os.path.basename(path)
        prefix = parse_gt_prefix(base_name, custom_prefix_pattern) if parse_gt_prefix else ""
        if prefix:
            prefixes.add(prefix)
    return sorted(prefixes)


def _prepare_unified_for_prefix_match(
    unified_csv_path: str,
    batch_id_col: str,
    prefixes: list[str],
    cache_dir: Path,
) -> tuple[str, str, bool]:
    """Create a temporary unified CSV with a prefix-matched batch_id column."""
    if not unified_csv_path or not os.path.exists(unified_csv_path) or not prefixes:
        return unified_csv_path, batch_id_col, False

    unified_df = _dataio_load(unified_csv_path)
    if batch_id_col not in unified_df.columns:
        return unified_csv_path, batch_id_col, False

    batch_series = unified_df[batch_id_col].astype(str)
    prefix_tuple = tuple(prefixes)
    startswith_matches = batch_series.str.startswith(prefix_tuple, na=False).any()
    if not startswith_matches:
        return unified_csv_path, batch_id_col, False

    temp_col = "__mr_batch_id_match"
    unified_df[temp_col] = batch_series

    prefixes_sorted = sorted(prefixes, key=len, reverse=True)
    for prefix in prefixes_sorted:
        mask = batch_series.str.startswith(prefix, na=False)
        unified_df.loc[mask, temp_col] = prefix

    temp_in_path = cache_dir / "tools_mr_unified_temp_in.csv"
    unified_df.to_csv(temp_in_path, index=False)
    return str(temp_in_path), temp_col, True


def _merge_geojson_into_unified(
    *,
    assignment_df: pd.DataFrame,
    unified_csv_path: str,
    phenotype_col_name: str,
    batch_id_column: str,
    x_centroid_column: str,
    y_centroid_column: str,
    tolerance_fraction: float,
    override_existing: bool,
    overwrite_entire_column: bool,
    exclude_classifications: list[str] | None,
    relabel_mapping: dict[str, str] | None,
    exclude_new_names: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = _ensure_mr_assignment_columns(assignment_df)
    if assignments.empty:
        raise ValueError("No GeoJSON files are available for merge.")
    if not os.path.exists(unified_csv_path):
        raise FileNotFoundError(f"Unified CSV not found: {unified_csv_path}")

    unified_df = _dataio_load(unified_csv_path)
    for col in (batch_id_column, x_centroid_column, y_centroid_column):
        if col not in unified_df.columns:
            raise KeyError(f"'{col}' column missing from unified CSV.")

    if phenotype_col_name not in unified_df.columns:
        unified_df[phenotype_col_name] = pd.NA
    if overwrite_entire_column:
        logger.warning(
            "Ignored legacy overwrite_entire_column=True for Manual Reference; "
            "existing phenotype values are always preserved."
        )

    exclude_classifications_normalized = {
        str(value).strip().casefold()
        for value in (exclude_classifications or [])
        if str(value).strip()
    }
    exclude_new_names_set = {
        str(value).strip()
        for value in (exclude_new_names or [])
        if str(value).strip()
    }
    normalized_mapping = {
        _normalize_label(k): v
        for k, v in (relabel_mapping or {}).items()
    }

    batch_series = unified_df[batch_id_column].astype(str)
    stats_records: list[dict[str, Any]] = []

    for row in assignments.to_dict(orient="records"):
        file_name = row["file"]
        assigned_batch = str(row.get("assigned_batch", "")).strip()
        geojson_path = str(row.get("path", "")).strip()

        stats_row: dict[str, Any] = {
            "file": file_name,
            "assigned_batch": assigned_batch,
            "match_source": str(row.get("match_source", "")).strip(),
            "raw_geojson_cells": 0,
            "filtered_cells": 0,
            "matched_cells": 0,
            "applied_cells": 0,
            "skipped_existing": 0,
            "tolerance_pixels": 0.0,
            "status": "pending",
        }

        if not geojson_path or not os.path.exists(geojson_path):
            stats_row["status"] = "missing GeoJSON"
            stats_records.append(stats_row)
            continue

        if not assigned_batch:
            stats_row["status"] = "needs batch assignment"
            stats_records.append(stats_row)
            continue

        annotations_df = _extract_geojson_annotations(geojson_path)
        stats_row["raw_geojson_cells"] = int(len(annotations_df))
        if annotations_df.empty:
            stats_row["status"] = "no valid polygons"
            stats_records.append(stats_row)
            continue

        annotations_df = annotations_df.dropna(subset=["x_centroid", "y_centroid"]).reset_index(drop=True)
        if exclude_classifications_normalized:
            annotations_df = annotations_df[
                ~annotations_df["classification"].astype(str).str.casefold().isin(exclude_classifications_normalized)
            ].reset_index(drop=True)
        if normalized_mapping and not annotations_df.empty:
            annotations_df["classification"] = annotations_df["classification"].apply(
                lambda value: normalized_mapping.get(_normalize_label(value), value)
            )
        if exclude_new_names_set and not annotations_df.empty:
            annotations_df = annotations_df[
                ~annotations_df["classification"].astype(str).isin(exclude_new_names_set)
            ].reset_index(drop=True)

        stats_row["filtered_cells"] = int(len(annotations_df))
        if annotations_df.empty:
            stats_row["status"] = "all annotations filtered"
            stats_records.append(stats_row)
            continue

        batch_df = unified_df.loc[batch_series == assigned_batch].dropna(
            subset=[x_centroid_column, y_centroid_column]
        )
        if batch_df.empty:
            stats_row["status"] = "assigned batch not found"
            stats_records.append(stats_row)
            continue

        x_range = batch_df[x_centroid_column].max() - batch_df[x_centroid_column].min()
        y_range = batch_df[y_centroid_column].max() - batch_df[y_centroid_column].min()
        max_range = max(x_range, y_range, 1.0)
        tolerance_pixels = tolerance_fraction * max_range
        stats_row["tolerance_pixels"] = round(float(tolerance_pixels), 3)

        batch_coords = batch_df[[x_centroid_column, y_centroid_column]].to_numpy(dtype=float, copy=True)
        geojson_coords = annotations_df[["x_centroid", "y_centroid"]].to_numpy(dtype=float, copy=True)
        chosen: dict[int, tuple[float, str]] = {}

        if cKDTree is not None:
            tree = cKDTree(batch_coords)
            dists, idxs = tree.query(geojson_coords, k=1, distance_upper_bound=tolerance_pixels)
            valid_mask = np.isfinite(dists) & (idxs < len(batch_df))
            valid_idxs = idxs[valid_mask]
            valid_dists = dists[valid_mask]
            valid_classes = annotations_df.loc[valid_mask, "classification"].astype(str).to_numpy()
            for idx, dist, cls_name in zip(valid_idxs, valid_dists, valid_classes, strict=False):
                unified_index = batch_df.index[int(idx)]
                if unified_index not in chosen or dist < chosen[unified_index][0]:
                    chosen[unified_index] = (float(dist), cls_name)
        else:
            for geojson_index, (gx, gy) in enumerate(geojson_coords):
                dists = np.sqrt(((batch_coords - np.array([gx, gy])) ** 2).sum(axis=1))
                min_idx = int(dists.argmin())
                dist = float(dists[min_idx])
                if dist > tolerance_pixels:
                    continue
                unified_index = batch_df.index[min_idx]
                cls_name = str(annotations_df.iloc[geojson_index]["classification"])
                if unified_index not in chosen or dist < chosen[unified_index][0]:
                    chosen[unified_index] = (dist, cls_name)

        stats_row["matched_cells"] = len(chosen)
        for unified_index, (_, cls_name) in chosen.items():
            if (not override_existing) and pd.notna(unified_df.at[unified_index, phenotype_col_name]):
                stats_row["skipped_existing"] += 1
                continue
            unified_df.at[unified_index, phenotype_col_name] = cls_name
            stats_row["applied_cells"] += 1

        stats_row["status"] = "merged" if stats_row["matched_cells"] else "no coordinate matches"
        stats_records.append(stats_row)
        _log_console(
            f"[{_timestamp()}] TOOLS-MR: {file_name} -> {assigned_batch} | "
            f"cells={stats_row['filtered_cells']} | matched={stats_row['matched_cells']} | "
            f"applied={stats_row['applied_cells']} | tolerance={tolerance_pixels:.3f} px"
        )

    unified_df.to_csv(unified_csv_path, index=False)
    return unified_df, pd.DataFrame(stats_records)


def _fast_merge_ground_truth(
    *,
    gt_output_dir: str,
    unified_csv_path: str,
    phenotype_col_name: str,
    batch_id_column: str,
    x_centroid_column: str,
    y_centroid_column: str,
    tolerance_fraction: float,
    override_existing: bool,
    overwrite_entire_column: bool,
    exclude_classifications: list[str] | None,
    relabel_mapping: dict[str, str] | None,
    exclude_new_names: list[str] | None,
    custom_prefix_pattern: str | None,
) -> pd.DataFrame:
    """Fast merge using KDTree nearest-neighbor matching per batch.

    Parameters
    ----------
    gt_output_dir : str
        Directory containing ground-truth CSV files.
    unified_csv_path : str
        Path to unified CSV.
    phenotype_col_name : str
        Output phenotype column name.
    batch_id_column : str
        Column name for batch IDs.
    x_centroid_column : str
        X centroid column in unified CSV.
    y_centroid_column : str
        Y centroid column in unified CSV.
    tolerance_fraction : float
        Tolerance fraction for matching.
    override_existing : bool
        Whether to overwrite existing values.
    overwrite_entire_column : bool
        Whether to clear the entire phenotype column before merging.
    exclude_classifications : list[str] | None
        Classifications to exclude before merging.
    custom_prefix_pattern : str | None
        Custom prefix regex for GeoJSON/GT filenames.
    """
    import glob

    gt_files = glob.glob(os.path.join(gt_output_dir, "*_ground_truth.csv"))
    if not gt_files:
        raise FileNotFoundError("No *_ground_truth.csv files found.")

    unified_df = _dataio_load(unified_csv_path)
    for col in (batch_id_column, x_centroid_column, y_centroid_column):
        if col not in unified_df.columns:
            raise KeyError(f"'{col}' column missing from unified CSV.")

    if phenotype_col_name not in unified_df.columns:
        unified_df[phenotype_col_name] = pd.NA
    if overwrite_entire_column:
        logger.warning(
            "Ignored legacy overwrite_entire_column=True for Manual Reference; "
            "existing phenotype values are always preserved."
        )

    exclude_classifications = exclude_classifications or []
    relabel_mapping = relabel_mapping or {}
    exclude_new_names = exclude_new_names or []
    batch_series = unified_df[batch_id_column].astype(str)

    for gt_file in gt_files:
        base_name = os.path.basename(gt_file)
        prefix = parse_gt_prefix(base_name, custom_prefix_pattern) if parse_gt_prefix else ""
        if not prefix:
            _log_console(f"  WARNING: Could not parse prefix from {base_name}; skipping.")
            continue

        gt_df = pd.read_csv(gt_file)
        required_cols = {"x_centroid", "y_centroid", "classification"}
        if not required_cols.issubset(gt_df.columns):
            _log_console(f"  WARNING: Missing required columns in {base_name}; skipping.")
            continue

        if exclude_classifications:
            gt_df = gt_df[~gt_df["classification"].isin(exclude_classifications)]
        if gt_df.empty:
            _log_console(f"  INFO: No classified cells remain in {base_name} after filtering; skipping.")
            continue

        batch_mask = batch_series.str.startswith(prefix, na=False)
        batch_df = unified_df.loc[batch_mask]
        if batch_df.empty:
            _log_console(f"  WARNING: No unified rows with {batch_id_column} starting '{prefix}'")
            continue

        x_range = batch_df[x_centroid_column].max() - batch_df[x_centroid_column].min()
        y_range = batch_df[y_centroid_column].max() - batch_df[y_centroid_column].min()
        max_range = max(x_range, y_range, 1.0)
        tolerance_pixels = tolerance_fraction * max_range

        batch_coords = batch_df[[x_centroid_column, y_centroid_column]].to_numpy()
        gt_coords = gt_df[["x_centroid", "y_centroid"]].to_numpy()

        tree = cKDTree(batch_coords)
        dists, idxs = tree.query(gt_coords, k=1, distance_upper_bound=tolerance_pixels)

        valid_mask = np.isfinite(dists)
        valid_idxs = idxs[valid_mask]
        valid_dists = dists[valid_mask]
        valid_classes = gt_df.loc[valid_mask, "classification"].astype(str).to_numpy()

        chosen: dict[int, tuple[float, str]] = {}
        for idx, dist, cls_name in zip(valid_idxs, valid_dists, valid_classes, strict=False):
            if idx >= len(batch_df):
                continue
            unified_index = batch_df.index[idx]
            if unified_index in chosen:
                if dist < chosen[unified_index][0]:
                    chosen[unified_index] = (dist, cls_name)
            else:
                chosen[unified_index] = (dist, cls_name)

        applied = 0
        for unified_index, (_, cls_name) in chosen.items():
            if (not override_existing) and pd.notna(unified_df.at[unified_index, phenotype_col_name]):
                continue
            unified_df.at[unified_index, phenotype_col_name] = cls_name
            applied += 1

        _log_console(
            f"  Processing {base_name} | prefix={prefix} | GT cells={len(gt_df)} | tolerance={tolerance_pixels:.3f} px"
        )
        _log_console(f"    Matched {applied}/{len(gt_df)} GT cells for prefix {prefix}")

    if relabel_mapping and phenotype_col_name in unified_df.columns:
        normalized_mapping = {
            _normalize_label(k): v for k, v in relabel_mapping.items()
        }
        unified_df[phenotype_col_name] = unified_df[phenotype_col_name].apply(
            lambda v: normalized_mapping.get(_normalize_label(v), v)
        )
    if exclude_new_names and phenotype_col_name in unified_df.columns:
        unified_df.loc[
            unified_df[phenotype_col_name].isin(list(exclude_new_names)),
            phenotype_col_name,
        ] = pd.NA

    unified_df.to_csv(unified_csv_path, index=False)
    return unified_df


def render_tools(project_root: Path, global_input_path: str | None) -> None:
    st.subheader("TOOLS")
    render_phenotyping_run_log_controls(Path(project_root), "tools", "Phenotyping Tools")
    st.markdown("Accessory utilities for manual reference and additional columns.")

    cache_dir = project_root / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tools_cache = cache_dir / "tools_params.json"

    cached_params: dict[str, Any] = {}
    if tools_cache.exists():
        try:
            with open(tools_cache, "r") as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}
    # Cell cards, dropdown values, and seen-cell history belong only to the active
    # browser session. Never restore them from tools_params.json.
    cached_key_count = len(cached_params)
    _prune_visual_manual_reference_runtime_cache(cached_params)
    if len(cached_params) != cached_key_count:
        try:
            with open(tools_cache, "w") as f:
                json.dump(cached_params, f, indent=2)
        except Exception:
            pass

    # Clear any runtime payload already present in a browser session that was open
    # before this no-cache behavior was introduced. The version key is deliberately
    # private/session-only, so every new app session also starts without old cards.
    if st.session_state.get("_tools_mr_vv_runtime_state_version") != (
        _VISUAL_MANUAL_REFERENCE_RUNTIME_STATE_VERSION
    ):
        st.session_state["tools_mr_vv_results"] = []
        st.session_state["tools_mr_vv_seen_keys"] = []
        st.session_state.pop("tools_mr_vv_fetch_more", None)
        st.session_state.pop("tools_mr_vv_last_file", None)
        _clear_visual_manual_reference_assignment_widget_state()
        st.session_state["_tools_mr_vv_runtime_state_version"] = (
            _VISUAL_MANUAL_REFERENCE_RUNTIME_STATE_VERSION
        )

    # --- Settings Export / Import ---
    render_settings_export_import(
        "TOOLS",
        ["tools_", "thresh_"],
        {"tools_params.json": tools_cache},
    )

    sub_tabs = st.tabs(["Manual Reference", "Phenotype GEOJSON", "Edit CSV", "Threshold"])

    with sub_tabs[0]:
        st.markdown("### Manual Reference")
        st.caption(
            "Build or revise manual cell references from CODEX images, or merge phenotype assignments made in QuPath."
        )
        unified_csv_path = global_input_path or ""
        available_columns = _get_columns_cached(unified_csv_path) if unified_csv_path else []
        create_reference_tab, qupath_merge_tab = st.tabs(
            ["Create Manual Reference", "Merge Phenotyped QuPath GeoJSON"]
        )

        with create_reference_tab:
            st.markdown("#### Create Manual Reference")
            st.caption(
                "Sample new cells or review existing manual labels, render each cell with its five highest transformed markers, "
                "and write only the reviewed rows back to the unified CSV."
            )

            # The receipt is independent of the rendered cards. Project-state cleanup or
            # a rerun may legitimately remove those runtime cards, but it must never hide
            # whether the CSV write succeeded or what it changed.
            save_receipt = st.session_state.get("_tools_mr_vv_save_receipt")
            if isinstance(save_receipt, dict) and save_receipt.get("message"):
                _render_manual_reference_save_receipt(
                    save_receipt,
                    details_expanded=False,
                    show_dismiss=True,
                )

            if not unified_csv_path or not os.path.exists(unified_csv_path):
                st.error("Unified CSV not found. Please select a valid unified CSV in the global settings.")
            elif CellVisualizer is None:
                st.error("CellVisualizer is not available in this environment.")
            else:
                if not st.session_state.get("viz_enabled", False):
                    st.info(
                        "This workflow uses the ROI, segmentation, batch, and coordinate values in Visual Validation Settings above. "
                        "Enable that panel to edit or scan those shared paths."
                    )
                viz_roi_dir = str(st.session_state.get("viz_roi_dir", "")).strip()
                viz_seg_dir = str(st.session_state.get("viz_seg_dir", "")).strip()
                viz_batch_col = str(st.session_state.get("viz_col_batch", "batch_id")).strip() or "batch_id"
                viz_default_x_col = str(st.session_state.get("viz_col_x", "x_centroid")).strip() or "x_centroid"
                viz_default_y_col = str(st.session_state.get("viz_col_y", "y_centroid")).strip() or "y_centroid"
                viz_tile_size = int(st.session_state.get("viz_tile_size", 40))
                viz_default_n_cells = int(st.session_state.get("viz_start_n_cells", 5))

                vv_available_columns = available_columns if available_columns else _get_columns_cached(unified_csv_path)
                marker_candidates = _get_visual_marker_candidates(vv_available_columns)
                notebook_corr_markers, notebook_non_corr_markers = _extract_markers_from_notebook("Step6A_phenotype_leiden.ipynb")
                expected_markers: list[str] = []
                for marker_name in notebook_corr_markers:
                    corr_col = f"{marker_name}_corr"
                    if corr_col in marker_candidates:
                        expected_markers.append(corr_col)
                for marker_name in notebook_non_corr_markers:
                    if marker_name in marker_candidates:
                        expected_markers.append(marker_name)

                shared_marker_panel, marker_workspace_note = get_marker_workspace_panel(
                    marker_candidates,
                    unified_csv_path,
                )
                marker_workspace = st.session_state.get("tab7_marker_workspace", {})
                if not isinstance(marker_workspace, dict):
                    marker_workspace = {}
                affix_position = str(marker_workspace.get("affix_position", "Suffix"))
                correction_affix = str(marker_workspace.get("correction_affix", "_corr"))

                if st.session_state.get("tools_mr_vv_last_file") != unified_csv_path:
                    st.session_state["tools_mr_vv_last_file"] = unified_csv_path
                    st.session_state["tools_mr_vv_filter_rows"] = []
                    st.session_state.pop("tools_mr_vv_saved_config", None)
                    st.session_state["tools_mr_vv_results"] = []
                    st.session_state["tools_mr_vv_seen_keys"] = []
                    _clear_visual_manual_reference_assignment_widget_state()

                if "tools_mr_vv_filter_rows" not in st.session_state:
                    st.session_state["tools_mr_vv_filter_rows"] = []

                st.caption(
                    f"Global visual validation settings: batch={viz_batch_col}, x={viz_default_x_col}, y={viz_default_y_col}, tile size={viz_tile_size}px"
                )
                if not viz_roi_dir or not os.path.exists(viz_roi_dir) or not viz_seg_dir or not os.path.exists(viz_seg_dir):
                    st.warning(
                        "ROI and segmentation directories from the universal Visual Validation settings must point to existing folders before cells can be rendered."
                    )

                st.markdown("##### Configuration")
                _, selected_vv_markers, marker_workspace_note = render_method_marker_selection(
                    key_prefix="tools_mr_vv",
                    method_name="Manual Visual Validation",
                    full_input_path=unified_csv_path,
                    available_columns=marker_candidates,
                    shared_panel=shared_marker_panel,
                    marker_workspace_note=marker_workspace_note,
                    fallback_markers=expected_markers,
                    saved_markers=st.session_state.get("tools_mr_vv_confirmed_markers", []),
                )

                config_col1, config_col2, config_col3 = st.columns([1, 2, 2])
                with config_col1:
                    apply_arcsinh = st.checkbox(
                        "Arcsinh first",
                        value=bool(cached_params.get("tools_mr_vv_arcsinh", False)),
                        key="tools_mr_vv_arcsinh",
                        help="Apply arcsinh(value / cofactor) before the selected normalization steps.",
                    )
                    arcsinh_cofactor = st.number_input(
                        "Arcsinh cofactor",
                        min_value=0.001,
                        value=float(cached_params.get("tools_mr_vv_arcsinh_cofactor", 5.0)),
                        key="tools_mr_vv_arcsinh_cofactor",
                        disabled=not apply_arcsinh,
                    )
                with config_col2:
                    saved_steps = cached_params.get("tools_mr_vv_norm_steps")
                    if not isinstance(saved_steps, list):
                        legacy_method = str(cached_params.get("tools_mr_vv_norm_method", "Z1"))
                        saved_steps = [step for step in ["Z1", "Z2", "CDF", "NLog"] if _get_visual_normalization_flags(legacy_method)[["Z1", "Z2", "CDF", "NLog"].index(step)]]
                    normalization_steps = st.multiselect(
                        "Normalization Steps (run in this order)",
                        ["Z1", "Z2", "CDF", "NLog"],
                        default=[step for step in ["Z1", "Z2", "CDF", "NLog"] if step in saved_steps],
                        key="tools_mr_vv_norm_steps",
                        help="Selected steps always run in sequence: Z1 → Z2 → CDF → NLog.",
                    )
                with config_col3:
                    group_by_options = [""] + vv_available_columns
                    default_group_by = cached_params.get(
                        "tools_mr_vv_group_by",
                        viz_batch_col if viz_batch_col in group_by_options else "",
                    )
                    group_by_col = st.selectbox(
                        "Normalize Within Groups",
                        group_by_options,
                        index=group_by_options.index(default_group_by) if default_group_by in group_by_options else 0,
                        format_func=lambda value: value or "(Global — no grouping)",
                        key="tools_mr_vv_group_by",
                        help="Choose Batch ID or any other CSV column. Use global normalization for already batch-corrected markers.",
                    )
                st.caption(
                    "Pipeline: "
                    + ("Arcsinh → " if apply_arcsinh else "")
                    + (" → ".join(normalization_steps) if normalization_steps else "no additional normalization")
                )

                st.markdown("##### Filter By")
                active_filters = _render_filter_criteria(
                    "tools_mr_vv",
                    vv_available_columns,
                    unified_csv_path,
                    empty_caption="No filter criteria added. All rows are currently eligible for visualization.",
                    fresh_from_disk=True,
                )
                saved_vv_config = {
                    "markers": list(selected_vv_markers),
                    "normalization_steps": list(normalization_steps),
                    "group_by_col": group_by_col or None,
                    "apply_arcsinh": bool(apply_arcsinh),
                    "arcsinh_cofactor": float(arcsinh_cofactor),
                    "filters": active_filters,
                    "batch_col": viz_batch_col,
                    "default_x_col": viz_default_x_col,
                    "default_y_col": viz_default_y_col,
                    "affix_position": affix_position,
                    "correction_affix": correction_affix,
                }
                st.session_state["tools_mr_vv_saved_config"] = saved_vv_config
                st.session_state["tools_mr_vv_confirmed_markers"] = list(selected_vv_markers)
                if not selected_vv_markers:
                    st.warning("Select at least one marker from the shared marker panel before creating references.")
                else:
                    st.caption(
                        f"Settings apply immediately: {len(selected_vv_markers)} marker(s) and "
                        f"{len(active_filters)} active filter(s)."
                    )

                if saved_vv_config:
                    st.divider()
                    st.markdown("##### Select and Visualize Cells")
                    clean_target_options = sorted({
                        _clean_visual_marker_name(marker_name, affix_position, correction_affix)
                        for marker_name in saved_vv_config["markers"]
                    })
                    # Keep this selector outside the form so changing modes
                    # immediately reruns the panel and reveals the relevant inputs.
                    # The search itself still starts only from the submit button.
                    mode = st.radio(
                        "Visualization Mode",
                        [
                            "Random Unlabeled Cells",
                            "Targeted Unlabeled Cells (Top 5 Markers)",
                            "Review / Edit Existing Labels",
                        ],
                        horizontal=True,
                        key="tools_mr_vv_mode",
                    )
                    with st.form("tools_mr_vv_fetch_form", clear_on_submit=False):
                        cells_to_visualize = st.number_input(
                            "Cells to Visualize",
                            min_value=1,
                            max_value=50,
                            value=viz_default_n_cells,
                            key="tools_mr_vv_n_cells",
                        )
                        target_markers = st.multiselect(
                            "Target Markers (up to 5) - Targeted Mode Only",
                            clean_target_options,
                            key="tools_mr_vv_target_markers",
                            help="Only cells whose transformed Top 5 markers include every selected target marker will be shown.",
                            disabled=mode != "Targeted Unlabeled Cells (Top 5 Markers)",
                        )

                        fetch_col1, fetch_col2, fetch_col3 = st.columns(3)
                        default_output_dir = cached_params.get(
                            "tools_mr_vv_output_dir",
                            str(project_root / "outputs" / "phenotyping" / "manual_reference_visual_validation"),
                        )
                        default_output_col = cached_params.get("tools_mr_vv_output_col", "phenotype_ref1")
                        with fetch_col1:
                            output_dir = st.text_input(
                                "Output Directory",
                                value=default_output_dir,
                                key="tools_mr_vv_output_dir",
                            )
                        with fetch_col2:
                            fetch_output_col = st.text_input(
                                "Output Column For New Assignments",
                                value=default_output_col,
                                key="tools_mr_vv_fetch_output_col",
                                help="Rows that already have a phenotype in this column are excluded from the random pool.",
                            )
                        cell_id_options = [""] + vv_available_columns
                        default_cell_id_col = cached_params.get(
                            "tools_mr_vv_cell_id_col",
                            "cell_id" if "cell_id" in cell_id_options else "",
                        )
                        with fetch_col3:
                            cell_id_col = st.selectbox(
                                "Cell ID Column",
                                cell_id_options,
                                index=cell_id_options.index(default_cell_id_col) if default_cell_id_col in cell_id_options else 0,
                                format_func=lambda value: value or "(None)",
                                key="tools_mr_vv_cell_id_col",
                            )

                        existing_label_options = []
                        if (
                            mode == "Review / Edit Existing Labels"
                            and str(fetch_output_col).strip() in vv_available_columns
                        ):
                            existing_label_options = _get_unique_values_fresh(
                                unified_csv_path,
                                str(fetch_output_col).strip(),
                            )
                            existing_label_options = [
                                value
                                for value in existing_label_options
                                if _has_visual_manual_reference_assignment(value)
                            ]
                        review_labels_key = "tools_mr_vv_review_labels"
                        if review_labels_key in st.session_state:
                            st.session_state[review_labels_key] = [
                                value
                                for value in st.session_state.get(review_labels_key, [])
                                if value in existing_label_options
                            ]
                        review_labels = st.multiselect(
                            "Existing Phenotypes to Review (optional)",
                            existing_label_options,
                            key=review_labels_key,
                            help="Leave empty to randomly review any assigned phenotype in the output column.",
                            disabled=mode != "Review / Edit Existing Labels",
                        )
                        fetch_clicked = st.form_submit_button("Select and Visualize Cells", type="primary")

                    fetch_more = bool(st.session_state.pop("tools_mr_vv_fetch_more", False))
                    if fetch_clicked:
                        # A fresh selection starts the "already shown" list over; Get more
                        # cells keeps it so the next batch is genuinely different.
                        st.session_state["tools_mr_vv_seen_keys"] = []

                    if fetch_clicked or fetch_more:
                        # A new batch must initialize from the current unified CSV, never
                        # from dropdown widget values belonging to old cards.
                        _clear_visual_manual_reference_assignment_widget_state()
                        st.session_state["tools_mr_vv_results"] = []
                        if mode == "Targeted Unlabeled Cells (Top 5 Markers)" and not target_markers:
                            st.error("Select at least one target marker for the targeted visualization mode.")
                        elif len(target_markers) > 5:
                            st.error("Select no more than five target markers.")
                        elif not str(fetch_output_col).strip():
                            st.error("Output column for new assignments is required before fetching cells.")
                        elif not viz_roi_dir or not os.path.exists(viz_roi_dir) or not viz_seg_dir or not os.path.exists(viz_seg_dir):
                            st.error("The ROI and segmentation directories in the universal Visual Validation settings must exist before cells can be rendered.")
                        else:
                            try:
                                output_dir_path = Path(output_dir)
                                output_dir_path.mkdir(parents=True, exist_ok=True)
                            except Exception as exc:
                                st.error(f"Could not create the output directory: {exc}")
                            else:
                                with st.spinner("Filtering data, computing normalized scores, and rendering cell overlays..."):
                                    fresh_source_columns = list(saved_vv_config["markers"])
                                    fresh_source_columns.extend(
                                        str(criterion.get("column", ""))
                                        for criterion in saved_vv_config.get("filters", [])
                                    )
                                    fresh_source_columns.extend(
                                        [
                                            str(saved_vv_config.get("batch_col", viz_batch_col)),
                                            str(saved_vv_config.get("default_x_col", viz_default_x_col)),
                                            str(saved_vv_config.get("default_y_col", viz_default_y_col)),
                                            str(saved_vv_config.get("group_by_col") or ""),
                                            str(cell_id_col or ""),
                                            str(fetch_output_col).strip(),
                                        ]
                                    )
                                    fresh_source_columns = [
                                        column
                                        for column in dict.fromkeys(fresh_source_columns)
                                        if column and column in vv_available_columns
                                    ]
                                    with _Step("load columns", n=len(fresh_source_columns)):
                                        df_full = _load_manual_reference_csv_fresh(
                                            unified_csv_path,
                                            fresh_source_columns,
                                        )
                                    _Step.note(f"   loaded {0 if df_full is None else len(df_full):,} rows")
                                    if df_full is None or df_full.empty:
                                        st.error("Could not load the unified CSV.")
                                    else:
                                        batch_col = saved_vv_config.get("batch_col", viz_batch_col)
                                        x_render_col = saved_vv_config.get("default_x_col", viz_default_x_col)
                                        y_render_col = saved_vv_config.get("default_y_col", viz_default_y_col)
                                        if batch_col not in df_full.columns:
                                            st.error(f"Batch column '{batch_col}' is missing from the unified CSV.")
                                        elif x_render_col not in df_full.columns or y_render_col not in df_full.columns:
                                            st.error("The global X/Y columns from the Visual Validation settings are missing from the unified CSV.")
                                        else:
                                            with _Step("apply filters", n=len(saved_vv_config.get("filters", []) or [])):
                                                filtered_df = _apply_visual_filters(df_full.copy(), saved_vv_config.get("filters", []))
                                            _Step.note(f"   {len(filtered_df):,} rows remain")
                                            if filtered_df.empty:
                                                st.warning("No cells remain after applying the saved filters.")
                                            else:
                                                try:
                                                    # Only the population statistics are computed here. The
                                                    # per-cell transform is applied further down, to the cells
                                                    # actually shown, which gives the same values without
                                                    # normalizing millions of rows nobody looks at.
                                                    _norm_step = _Step("normalization statistics", rows=len(filtered_df))
                                                    _norm_step.__enter__()
                                                    norm_stats = _visual_normalization_stats(
                                                        filtered_df,
                                                        saved_vv_config["markers"],
                                                        saved_vv_config["normalization_steps"],
                                                        saved_vv_config.get("group_by_col"),
                                                        apply_arcsinh=saved_vv_config.get("apply_arcsinh", False),
                                                        arcsinh_cofactor=saved_vv_config.get("arcsinh_cofactor", 5.0),
                                                    )
                                                    norm_cols = norm_stats["norm_cols"]
                                                    prepared_df = filtered_df
                                                    _norm_step.__exit__(None, None, None)
                                                except Exception as exc:
                                                    st.error(f"Normalization failed: {exc}")
                                                else:
                                                    viz = _get_cell_visualizer_cached(viz_roi_dir, viz_seg_dir)
                                                    with _Step("match ROI/segmentation files"):
                                                        matched_batches = viz.match_file_triads(
                                                            prepared_df[batch_col].dropna().astype(str).unique().tolist()
                                                        )
                                                    _Step.note(f"   {len(matched_batches)} batches matched")
                                                    valid_df = prepared_df[
                                                        prepared_df[batch_col].astype(str).isin(matched_batches.keys())
                                                    ].copy()
                                                    if valid_df.empty:
                                                        st.warning("No cells remain after restricting to batches with matched ROI and segmentation files.")
                                                    else:
                                                        fetch_output_col = str(fetch_output_col).strip()
                                                        if mode == "Review / Edit Existing Labels":
                                                            if fetch_output_col not in valid_df.columns:
                                                                eligible_df = valid_df.iloc[0:0].copy()
                                                            else:
                                                                assigned_mask = valid_df[fetch_output_col].apply(
                                                                    _has_visual_manual_reference_assignment
                                                                )
                                                                eligible_df = valid_df.loc[assigned_mask].copy()
                                                                if review_labels:
                                                                    eligible_df = eligible_df.loc[
                                                                        eligible_df[fetch_output_col].astype(str).isin(
                                                                            [str(value) for value in review_labels]
                                                                        )
                                                                    ].copy()
                                                            excluded_assigned_count = 0
                                                            empty_message = (
                                                                f"No already-labeled cells were found in '{fetch_output_col}' "
                                                                "for the selected phenotype filter."
                                                            )
                                                        else:
                                                            eligible_df = _exclude_existing_visual_manual_reference_assignments(
                                                                valid_df,
                                                                fetch_output_col,
                                                            )
                                                            excluded_assigned_count = int(len(valid_df) - len(eligible_df))
                                                            empty_message = (
                                                                f"No unlabeled cells remain in '{fetch_output_col}' after applying the filters."
                                                            )
                                                        if eligible_df.empty:
                                                            st.session_state["tools_mr_vv_results"] = []
                                                            st.warning(empty_message)
                                                        else:
                                                            with _Step("shuffle candidates", eligible=len(eligible_df)):
                                                                sampled_candidates = eligible_df.iloc[
                                                                    np.random.default_rng().permutation(len(eligible_df))
                                                                ]
                                                            _Step.note("   scanning candidates for cells to show")
                                                            _mr_loop_started = time.perf_counter()
                                                            _mr_examined = 0
                                                            results_to_store: list[dict[str, Any]] = []
                                                            target_marker_set = (
                                                                {str(marker_name) for marker_name in target_markers}
                                                                if mode == "Targeted Unlabeled Cells (Top 5 Markers)"
                                                                else set()
                                                            )
                                                            already_shown = set(
                                                                st.session_state.get("tools_mr_vv_seen_keys", []) or []
                                                            )
                                                            for _, candidate_row in sampled_candidates.iterrows():
                                                                candidate_key = _sanitize_filename_token(
                                                                    f"{candidate_row.get(batch_col)}_{candidate_row.get(cell_id_col, 'unknown')}_{candidate_row.get(x_render_col)}_{candidate_row.get(y_render_col)}"
                                                                )
                                                                if candidate_key in already_shown:
                                                                    continue
                                                                _mr_examined += 1
                                                                # Normalize this one cell now, using the statistics
                                                                # computed over the whole filtered population.
                                                                candidate_row = _apply_visual_normalization_rows(
                                                                    candidate_row.to_frame().T, norm_stats
                                                                ).iloc[0]
                                                                top_markers = _compute_top_markers_for_row(
                                                                    candidate_row,
                                                                    saved_vv_config["markers"],
                                                                    norm_cols,
                                                                    top_n=5,
                                                                    affix_position=affix_position,
                                                                    correction_affix=correction_affix,
                                                                )
                                                                if not top_markers:
                                                                    continue
                                                                clean_top_markers = [marker_info[1] for marker_info in top_markers]
                                                                if target_marker_set and not target_marker_set.issubset(set(clean_top_markers)):
                                                                    continue
                                                                _Step.note(
                                                                    f"   cell {len(results_to_store) + 1} ready "
                                                                    f"({_mr_examined} examined, "
                                                                    f"{time.perf_counter() - _mr_loop_started:.1f}s elapsed)"
                                                                )
                                                                results_to_store.append(
                                                                    {
                                                                        "result_key": candidate_key,
                                                                        "batch_value": candidate_row.get(batch_col),
                                                                        "cell_id_value": candidate_row.get(cell_id_col, "unknown") if cell_id_col else "unknown",
                                                                        "cell_id_col": cell_id_col,
                                                                        "x_value": candidate_row.get(x_render_col),
                                                                        "y_value": candidate_row.get(y_render_col),
                                                                        "batch_col": batch_col,
                                                                        "default_x_col": x_render_col,
                                                                        "default_y_col": y_render_col,
                                                                        "default_output_col": fetch_output_col,
                                                                        "current_phenotype": candidate_row.get(fetch_output_col) if fetch_output_col in candidate_row.index else None,
                                                                        "top_markers": top_markers,
                                                                    }
                                                                )
                                                                if len(results_to_store) >= int(cells_to_visualize):
                                                                    break

                                                            if not results_to_store:
                                                                st.session_state["tools_mr_vv_results"] = []
                                                                if already_shown:
                                                                    st.warning(
                                                                        f"No further cells are available. All {len(already_shown)} cell(s) "
                                                                        "matching these settings have already been shown. Click "
                                                                        "'Select and Visualize Cells' to start over."
                                                                    )
                                                                else:
                                                                    st.warning("No cells satisfied the current mode and filter settings.")
                                                            else:
                                                                prewarm_fn = getattr(viz, "prewarm_batch", None)
                                                                if callable(prewarm_fn):
                                                                    _batches = sorted({str(result["batch_value"]) for result in results_to_store})
                                                                    with _Step("open image files", batches=len(_batches)):
                                                                        for prewarm_batch_id in _batches:
                                                                            with _Step(f"   open {prewarm_batch_id}"):
                                                                                prewarm_fn(prewarm_batch_id)

                                                                rendered_results: list[dict[str, Any]] = []
                                                                _render_started = time.perf_counter()
                                                                _Step.note(f"rendering {len(results_to_store)} cell images")
                                                                for _cell_number, result in enumerate(results_to_store, start=1):
                                                                    try:
                                                                        top_marker_names = [marker_info[1] for marker_info in result["top_markers"]]
                                                                        _cell_started = time.perf_counter()
                                                                        cell_ctx = viz.get_cell_context(
                                                                            batch_id=str(result["batch_value"]),
                                                                            x=float(result["x_value"]),
                                                                            y=float(result["y_value"]),
                                                                            tile_size=viz_tile_size,
                                                                            requested_channels=top_marker_names,
                                                                        )
                                                                        marker_indices, names_for_plot = _resolve_overlay_channels(
                                                                            cell_ctx,
                                                                            top_marker_names,
                                                                            affix_position=affix_position,
                                                                            correction_affix=correction_affix,
                                                                        )
                                                                        if not marker_indices:
                                                                            available_channel_names = [
                                                                                str(channel_name)
                                                                                for channel_name in cell_ctx.get("channel_names", [])
                                                                            ]
                                                                            fallback_count = min(
                                                                                5,
                                                                                len(available_channel_names),
                                                                                int(cell_ctx.get("image", np.empty((0,))).shape[0]) if isinstance(cell_ctx.get("image"), np.ndarray) else 0,
                                                                            )
                                                                            marker_indices = list(range(fallback_count))
                                                                            names_for_plot = [
                                                                                available_channel_names[idx] if idx < len(available_channel_names) else f"Ch{idx}"
                                                                                for idx in marker_indices
                                                                            ]
                                                                        fig = viz.generate_visualization(
                                                                            cell_ctx,
                                                                            marker_indices,
                                                                            names_for_plot,
                                                                            display_size=viz_tile_size,
                                                                            show_neighbors=True,
                                                                        )
                                                                        figure_path = _save_visual_manual_reference_figure(
                                                                            fig,
                                                                            output_dir,
                                                                            result["batch_value"],
                                                                            result["cell_id_value"],
                                                                            result["x_value"],
                                                                            result["y_value"],
                                                                        )
                                                                        plt.close(fig)
                                                                        result["figure_path"] = figure_path
                                                                        _Step.note(
                                                                            f"   image {_cell_number}/{len(results_to_store)} "
                                                                            f"in {time.perf_counter() - _cell_started:.1f}s "
                                                                            f"({time.perf_counter() - _render_started:.1f}s total)"
                                                                        )
                                                                        rendered_results.append(result)
                                                                    except Exception as exc:
                                                                        logger.error(f"Tools MR visual validation render failed: {exc}")

                                                                if not rendered_results:
                                                                    st.session_state["tools_mr_vv_results"] = []
                                                                    st.warning("No figures could be rendered for the sampled cells.")
                                                                else:
                                                                    st.session_state["tools_mr_vv_results"] = rendered_results
                                                                    st.session_state["tools_mr_vv_seen_keys"] = _accumulate_shown_cell_keys(
                                                                        st.session_state.get("tools_mr_vv_seen_keys"),
                                                                        [r["result_key"] for r in rendered_results],
                                                                    )
                                                                    cached_params["tools_mr_vv_output_dir"] = output_dir
                                                                    cached_params["tools_mr_vv_output_col"] = fetch_output_col
                                                                    cached_params["tools_mr_vv_cell_id_col"] = cell_id_col
                                                                    try:
                                                                        with open(tools_cache, "w") as f:
                                                                            json.dump(cached_params, f, indent=2)
                                                                    except Exception:
                                                                        pass
                                                                    success_message = (
                                                                        f"Generated and saved {len(rendered_results)} figure(s) to {output_dir_path}."
                                                                    )
                                                                    if excluded_assigned_count and mode != "Review / Edit Existing Labels":
                                                                        success_message += (
                                                                            f" Skipped {excluded_assigned_count} cell(s) that already had values in '{fetch_output_col}'."
                                                                        )
                                                                    st.success(success_message)

                    visual_results = st.session_state.get("tools_mr_vv_results", [])
                    if visual_results:
                        st.divider()
                        st.markdown("##### Phenotype Assignment")
                        st.caption(
                            "Choose phenotypes below, then save them together. The dropdowns stay local until Save. "
                            "For a previously labeled cell, Unassigned or Unclassified removes the value from the unified CSV."
                        )
                        with st.expander("Add New Phenotype", expanded=False):
                            render_add_custom_phenotype_ui("tools_mr_vv")

                        phenotype_options: list[str] = []
                        for option in ["Unassigned", "Unclassified"] + get_all_phenotype_options(
                            include_unassigned=False
                        ) + ["Add New Phenotype"]:
                            normalized_option = _normalize_visual_phenotype_label(option)
                            if normalized_option and normalized_option not in phenotype_options:
                                phenotype_options.append(normalized_option)
                        if "Unassigned" not in phenotype_options:
                            phenotype_options.insert(0, "Unassigned")
                        default_output_col = str(
                            st.session_state.get(
                                "tools_mr_vv_fetch_output_col",
                                cached_params.get("tools_mr_vv_output_col", "phenotype_ref1"),
                            )
                        ).strip() or "phenotype_ref1"
                        with st.form("tools_mr_vv_assignments_form", clear_on_submit=False):
                            submitted_selections: dict[str, str] = {}
                            current_label_placeholders: dict[str, Any] = {}
                            for result in visual_results:
                                st.markdown("---")
                                display_col1, display_col2 = st.columns([3, 2])
                                with display_col1:
                                    figure_path = result.get("figure_path")
                                    if figure_path and os.path.exists(str(figure_path)):
                                        st.image(str(figure_path), width="stretch")
                                    else:
                                        st.warning("Saved figure file could not be found on disk.")
                                    marker_caption = ", ".join(
                                        f"{marker_name} ({score:.2f})"
                                        for _, marker_name, score in result.get("top_markers", [])
                                    )
                                    st.caption(
                                        f"Batch: {result.get('batch_value')} | Cell ID: {result.get('cell_id_value')} | "
                                        f"X: {result.get('x_value')} | Y: {result.get('y_value')}"
                                    )
                                    if marker_caption:
                                        st.caption(f"Top 5 normalized markers: {marker_caption}")
                                    if figure_path:
                                        st.caption(f"Saved figure: {figure_path}")

                                with display_col2:
                                    current_phenotype = _normalize_visual_phenotype_label(
                                        result.get("current_phenotype")
                                    )
                                    result_phenotype_options = list(phenotype_options)
                                    if current_phenotype and current_phenotype not in result_phenotype_options:
                                        result_phenotype_options.insert(1, current_phenotype)
                                    default_phenotype = current_phenotype if current_phenotype in result_phenotype_options else "Unassigned"
                                    assignment_widget_key = f"tools_mr_vv_assign_pheno_{result['result_key']}"
                                    persisted_selection = _normalize_visual_phenotype_label(
                                        st.session_state.get(assignment_widget_key)
                                    )
                                    if assignment_widget_key in st.session_state and not persisted_selection:
                                        st.session_state[assignment_widget_key] = "Unassigned"
                                    selected_phenotype = st.selectbox(
                                        "Phenotype",
                                        result_phenotype_options,
                                        index=result_phenotype_options.index(default_phenotype),
                                        key=assignment_widget_key,
                                    )
                                    selected_phenotype = _normalize_visual_phenotype_label(selected_phenotype)
                                    submitted_selections[str(result["result_key"])] = selected_phenotype
                                    if selected_phenotype == "Add New Phenotype":
                                        st.info("Use the shared 'Add New Phenotype' control above, then return here and select the new phenotype.")

                                    x_assign_col = str(result.get("default_x_col"))
                                    y_assign_col = str(result.get("default_y_col"))
                                    output_col_value = str(result.get("default_output_col", default_output_col)).strip() or default_output_col
                                    current_label_placeholder = st.empty()
                                    if current_phenotype:
                                        current_label_placeholder.caption(
                                            f"Current label: {current_phenotype}"
                                        )
                                    current_label_placeholders[
                                        str(result["result_key"])
                                    ] = current_label_placeholder
                                    st.caption(
                                        f"Shared matching columns — batch: {result.get('batch_col')}, "
                                        f"x: {x_assign_col}, y: {y_assign_col}"
                                    )
                                    st.caption(f"Assignments will be written to: {output_col_value}")

                                    save_this_cell = st.form_submit_button(
                                        "Save This Cell To Unified CSV",
                                        key=f"tools_mr_vv_assign_save_{result['result_key']}",
                                        type="primary",
                                    )
                                    if save_this_cell:
                                        clearing_this_cell = not _has_visual_manual_reference_assignment(selected_phenotype)
                                        if selected_phenotype == "Add New Phenotype":
                                            st.warning("Add the new phenotype above before saving this assignment.")
                                        elif clearing_this_cell and not current_phenotype:
                                            st.info("This cell has no label to remove.")
                                        else:
                                            try:
                                                matched_rows = _update_visual_manual_reference_assignment(
                                                    unified_csv_path=unified_csv_path,
                                                    batch_column=str(result.get("batch_col")),
                                                    batch_value=result.get("batch_value"),
                                                    x_column=x_assign_col,
                                                    x_value=result.get("x_value"),
                                                    y_column=y_assign_col,
                                                    y_value=result.get("y_value"),
                                                    output_column=output_col_value,
                                                    phenotype_value=selected_phenotype,
                                                    cell_id_column=str(result.get("cell_id_col") or "") or None,
                                                    cell_id_value=result.get("cell_id_value"),
                                                    allow_clear=True,
                                                )
                                            except Exception as exc:
                                                st.error(f"Could not save the phenotype assignment: {exc}")
                                            else:
                                                cached_params["tools_mr_vv_output_col"] = output_col_value
                                                try:
                                                    with open(tools_cache, "w") as f:
                                                        json.dump(cached_params, f, indent=2)
                                                except Exception:
                                                    pass

                                                single_receipt = _build_manual_reference_save_receipt(
                                                    [
                                                        {
                                                            "result_key": result["result_key"],
                                                            "batch_value": result.get("batch_value"),
                                                            "cell_id_value": result.get("cell_id_value"),
                                                            "output_column": output_col_value,
                                                            "phenotype_value": selected_phenotype,
                                                            "clear_existing": clearing_this_cell,
                                                        }
                                                    ],
                                                    visual_results,
                                                    verified_count=matched_rows,
                                                    target_path=str(Path(unified_csv_path).resolve()),
                                                )
                                                st.session_state[
                                                    "_tools_mr_vv_save_receipt"
                                                ] = single_receipt
                                                logger.info(
                                                    "Manual Reference save: %s",
                                                    single_receipt["message"],
                                                )

                                                # The rendered list is a snapshot taken when the
                                                # cells were fetched. Without this the card keeps
                                                # reporting the label the cell had before the save,
                                                # and a re-fetch appears to hand back the same cell.
                                                for stored_result in st.session_state.get("tools_mr_vv_results", []):
                                                    if stored_result.get("result_key") == result["result_key"]:
                                                        stored_result["current_phenotype"] = (
                                                            "" if clearing_this_cell else selected_phenotype
                                                        )
                                                        break

                                                if clearing_this_cell:
                                                    current_label_placeholder.empty()
                                                else:
                                                    current_label_placeholder.caption(
                                                        f"Current label: {selected_phenotype}"
                                                    )
                                                _render_manual_reference_save_receipt(
                                                    single_receipt,
                                                    details_expanded=True,
                                                    show_dismiss=False,
                                                )

                            # Use the values returned by the form widgets in this exact run.
                            # Reading session_state here loses offline form edits on submit.
                            pending_assignments, pending_counts = _build_manual_reference_pending_assignments(
                                visual_results,
                                submitted_selections,
                                default_output_col,
                            )

                            st.markdown("###### Save Assigned Cells")
                            st.caption(
                                "Save reads the staged dropdown values only when clicked. Get More "
                                "uses the same sampling criteria and skips cells already shown in this run."
                            )

                            c_batch_save, c_fetch_more = st.columns([2, 1])
                            batch_save = c_batch_save.form_submit_button(
                                "Save Assigned Cells To Unified CSV",
                                type="primary",
                            )
                            get_more_cells = c_fetch_more.form_submit_button(
                                "Get More Cells",
                                help="Fetch another set using the same criteria, skipping every cell already shown.",
                            )
                            if get_more_cells:
                                st.session_state["tools_mr_vv_fetch_more"] = True
                                st.rerun()
                            if batch_save:
                                if not pending_assignments:
                                    if pending_counts["pending_new"]:
                                        st.warning(
                                            "Create and select a concrete phenotype before saving the "
                                            "cells still set to Add New Phenotype."
                                        )
                                    else:
                                        st.warning(
                                            "No changes were submitted. Choose a phenotype, or choose "
                                            "Unassigned/Unclassified on a currently labeled cell to remove its label."
                                        )
                                else:
                                    try:
                                        batch_save_stats = _update_visual_manual_reference_assignments(
                                            unified_csv_path=unified_csv_path,
                                            assignments=pending_assignments,
                                        )
                                    except Exception as exc:
                                        st.error(f"Could not save the selected phenotype assignments: {exc}")
                                    else:
                                        output_columns = [
                                            str(assignment.get("output_column", "")).strip()
                                            for assignment in pending_assignments
                                            if str(assignment.get("output_column", "")).strip()
                                        ]
                                        if output_columns:
                                            cached_params["tools_mr_vv_output_col"] = output_columns[0]
                                        try:
                                            with open(tools_cache, "w") as f:
                                                json.dump(cached_params, f, indent=2)
                                        except Exception:
                                            pass
                                        save_receipt = _build_manual_reference_save_receipt(
                                            pending_assignments,
                                            visual_results,
                                            verified_count=batch_save_stats["verified_count"],
                                            target_path=batch_save_stats["target_path"],
                                        )
                                        st.session_state[
                                            "_tools_mr_vv_save_receipt"
                                        ] = save_receipt
                                        logger.info(
                                            "Manual Reference save: %s",
                                            save_receipt["message"],
                                        )

                                        # Same snapshot problem as the single-cell save: without
                                        # this the cards keep reporting pre-save labels. Every cell
                                        # carrying a concrete selection was just written, so its
                                        # snapshot can be brought up to date from that selection.
                                        for stored_result in st.session_state.get("tools_mr_vv_results", []):
                                            stored_result_key = str(
                                                stored_result.get("result_key")
                                            )
                                            written_label = _normalize_visual_phenotype_label(
                                                submitted_selections.get(
                                                    stored_result_key, "Unassigned"
                                                )
                                            )
                                            if written_label == "Add New Phenotype":
                                                continue
                                            if _has_visual_manual_reference_assignment(written_label):
                                                stored_result["current_phenotype"] = written_label
                                            elif _normalize_visual_phenotype_label(
                                                stored_result.get("current_phenotype")
                                            ):
                                                stored_result["current_phenotype"] = ""

                                            current_label_placeholder = (
                                                current_label_placeholders.get(
                                                    stored_result_key
                                                )
                                            )
                                            if current_label_placeholder is not None:
                                                saved_label = _normalize_visual_phenotype_label(
                                                    stored_result.get("current_phenotype")
                                                )
                                                if saved_label:
                                                    current_label_placeholder.caption(
                                                        f"Current label: {saved_label}"
                                                    )
                                                else:
                                                    current_label_placeholder.empty()

                                        _render_manual_reference_save_receipt(
                                            save_receipt,
                                            details_expanded=True,
                                            show_dismiss=False,
                                        )

        with qupath_merge_tab:
            st.markdown("#### Merge Phenotyped QuPath GeoJSON")
            st.caption(
                "Scan QuPath GeoJSON files, verify or override each detected batch, relabel classes, "
                "and merge matched cells into one manual-reference column."
            )
            optional_csv_utils_available = all(
                func is not None
                for func in (
                    batch_process_geojson_files,
                    relabel_ground_truth_files,
                    enrich_ground_truth_with_unified_data,
                )
            )
            if shape is None:
                st.warning(
                    "Shapely is required for the GeoJSON-based Manual Reference workflow. "
                    "The Create Manual Reference workflow remains available in the neighboring tab."
                )
            if not optional_csv_utils_available:
                st.info("Direct GeoJSON merge is available. Optional ground-truth CSV helpers from acc_utils could not be loaded.")

            st.markdown("#### 1. Inputs")
            c1, c2 = st.columns(2)
            project_root_str = st.session_state.get('active_project_root')
            project_root = Path(project_root_str) if project_root_str else Path('.')
            default_geojson_dir = cached_params.get(
                "tools_mr_geojson_dir",
                cached_params.get(
                    "mr_geojson_dir",
                    str(project_root / "data") if project_root_str else "",
                ),
            )
            default_file_pattern = cached_params.get("tools_mr_file_pattern", cached_params.get("mr_file_pattern", "*.geojson"))
            with c1:
                geojson_dir = st.text_input(
                    "GeoJSON Directory",
                    value=default_geojson_dir,
                    key="tools_mr_geojson_dir",
                    help="Directory containing phenotyped GeoJSON files (QuPath).",
                )
            with c2:
                file_pattern = st.text_input(
                    "File Pattern",
                    value=default_file_pattern,
                    key="tools_mr_file_pattern",
                    help="Pattern to match GeoJSON files.",
                )

            if cached_params.get("mr_geojson_dir") != geojson_dir or cached_params.get("mr_file_pattern") != file_pattern:
                try:
                    cached_params["mr_geojson_dir"] = geojson_dir
                    cached_params["mr_file_pattern"] = file_pattern
                    cached_params["tools_mr_geojson_dir"] = geojson_dir
                    cached_params["tools_mr_file_pattern"] = file_pattern
                    with open(tools_cache, "w") as f:
                        json.dump(cached_params, f, indent=2)
                except Exception:
                    pass

            st.markdown("#### 2. Unified CSV Linking")
            if unified_csv_path:
                st.caption(f"Using unified CSV: {unified_csv_path}")
            else:
                st.warning("No unified CSV selected in Global Input Configuration.")

            batch_id_col = str(st.session_state.get("viz_col_batch", "batch_id")).strip() or "batch_id"
            x_col = str(st.session_state.get("viz_col_x", "x_centroid")).strip() or "x_centroid"
            y_col = str(st.session_state.get("viz_col_y", "y_centroid")).strip() or "y_centroid"
            st.caption(
                "Using shared Visual Validation columns: "
                f"batch={batch_id_col}, x={x_col}, y={y_col}. "
                "Change them in Visual Validation Settings above."
            )

            phenotype_col = st.text_input(
                "Output Phenotype Column",
                value=cached_params.get("mr_phenotype_col", "phenotype_ref1"),
                key="tools_mr_phenotype_col",
                help="Column name to store manual reference phenotypes.",
            )

            # Manual references are additive by design. Never clear an existing
            # phenotype column; only rows matched to reviewed GeoJSON cells change.
            overwrite_entire_column = False
            st.info(
                "Safe merge: existing column values are preserved. Only matched cells are written; "
                "enable Override Existing to correct a previously assigned cell."
            )

            use_fast_merge = st.checkbox(
                "Use Fast Merge (KDTree)",
                value=bool(cached_params.get("mr_fast_merge", True)) if cKDTree else False,
                key="tools_mr_fast_merge",
                help="Uses KDTree nearest-neighbor matching for faster merging.",
            )
            if use_fast_merge and cKDTree is None:
                st.warning("Fast merge requires SciPy; falling back to standard merge.")

            tolerance_fraction = st.number_input(
                "Tolerance Fraction",
                min_value=0.001,
                max_value=0.1,
                value=float(cached_params.get("mr_tolerance_fraction", 0.01)),
                step=0.001,
                help="Matching tolerance as a fraction of max coordinate range.",
                key="tools_mr_tolerance",
            )

            override_existing = st.checkbox(
                "Override Existing",
                value=bool(cached_params.get("mr_override_existing", True)),
                key="tools_mr_override_existing",
                help="Overwrite existing values in the phenotype column.",
            )

            exclude_classes = st.text_input(
                "Exclude Classifications (comma-separated)",
                value=", ".join(cached_params.get("mr_exclude_classes", [])),
                key="tools_mr_exclude",
            )

            custom_prefix_pattern = st.text_input(
                "Custom Prefix Pattern (regex, optional)",
                value=cached_params.get("mr_custom_prefix", ""),
                key="tools_mr_prefix_pattern",
            )

            cached_params.update(
                {
                    "mr_batch_id_col": batch_id_col,
                    "mr_x_col": x_col,
                    "mr_y_col": y_col,
                    "mr_phenotype_col": phenotype_col,
                    "mr_overwrite_column": overwrite_entire_column,
                    "mr_fast_merge": use_fast_merge,
                    "mr_tolerance_fraction": tolerance_fraction,
                    "mr_override_existing": override_existing,
                    "mr_exclude_classes": [c.strip() for c in exclude_classes.split(",") if c.strip()],
                    "mr_custom_prefix": custom_prefix_pattern,
                }
            )
            try:
                with open(tools_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass

            batch_values: list[str] = []
            if unified_csv_path and os.path.exists(unified_csv_path) and batch_id_col:
                batch_values = _get_unique_values_cached(unified_csv_path, batch_id_col)

            st.divider()
            st.markdown("#### 3. Preview & Relabel")
            geojson_dir_key = os.path.abspath(geojson_dir) if geojson_dir else ""
            unified_csv_key = os.path.abspath(unified_csv_path) if unified_csv_path else ""
            cached_scan_key = f"mr_scan::{geojson_dir_key}::{file_pattern}::{unified_csv_key}"
            cached_scan = cached_params.get(cached_scan_key) if unified_csv_key and geojson_dir_key else None
            if cached_scan is None and unified_csv_key and geojson_dir_key:
                for key, value in cached_params.items():
                    if not isinstance(key, str) or not key.startswith("mr_scan::"):
                        continue
                    parts = key.split("::", maxsplit=3)
                    if len(parts) != 4:
                        continue
                    _, cached_geojson, cached_pattern, cached_unified = parts
                    if (
                        os.path.abspath(cached_geojson) == geojson_dir_key
                        and os.path.abspath(cached_unified) == unified_csv_key
                        and cached_pattern == file_pattern
                    ):
                        cached_scan = value
                        break
            if cached_scan is None and unified_csv_key and geojson_dir_key:
                last_scan = cached_params.get("mr_scan_last")
                if isinstance(last_scan, dict):
                    if (
                        os.path.abspath(last_scan.get("geojson_dir", "")) == geojson_dir_key
                        and os.path.abspath(last_scan.get("unified_csv_path", "")) == unified_csv_key
                        and last_scan.get("file_pattern") == file_pattern
                    ):
                        cached_scan = last_scan
            if cached_scan and "summary" in cached_scan:
                summary_state = st.session_state.get("tools_mr_summary")
                if summary_state is None or (hasattr(summary_state, "empty") and summary_state.empty):
                    st.session_state["tools_mr_summary"] = pd.DataFrame(cached_scan.get("summary", []))
                assignment_state = st.session_state.get("tools_mr_assignments")
                if assignment_state is None or (hasattr(assignment_state, "empty") and assignment_state.empty):
                    if cached_scan.get("assignments"):
                        st.session_state["tools_mr_assignments"] = _ensure_mr_assignment_columns(
                            pd.DataFrame(cached_scan.get("assignments", []))
                        )
                mapping_state = st.session_state.get("tools_mr_mapping")
                if mapping_state is None or (hasattr(mapping_state, "empty") and mapping_state.empty):
                    if cached_scan.get("mapping"):
                        st.session_state["tools_mr_mapping"] = pd.DataFrame(cached_scan.get("mapping", []))

            if st.button("Scan GeoJSON", key="tools_mr_scan"):
                summary_df, assignment_df = _scan_geojson_summary(geojson_dir, file_pattern, batch_values, batch_id_col)
                assignment_df = _apply_saved_mr_assignments(
                    assignment_df,
                    st.session_state.get("tools_mr_assignments"),
                )
                summary_df = _apply_assignments_to_summary(summary_df, assignment_df)
                st.session_state["tools_mr_summary"] = summary_df
                st.session_state["tools_mr_assignments"] = assignment_df
                if summary_df is not None and not summary_df.empty:
                    existing_mapping = st.session_state.get("tools_mr_mapping")
                    if existing_mapping is None or existing_mapping.empty:
                        cached_scan = cached_params.get(cached_scan_key)
                        cached_mapping = cached_scan.get("mapping", []) if isinstance(cached_scan, dict) else []
                        if cached_mapping:
                            existing_mapping = pd.DataFrame(cached_mapping)
                    existing_mapping = existing_mapping if existing_mapping is not None else pd.DataFrame(columns=["Original", "New"])
                    existing_mapping = existing_mapping.reset_index(drop=True)
                    existing_map_dict = {
                        str(row.get("Original")): row.get("New", "")
                        for _, row in existing_mapping.iterrows()
                        if row.get("Original")
                    }
                    new_classes = sorted(summary_df["classification"].dropna().unique().tolist())
                    for cls_name in new_classes:
                        if cls_name not in existing_map_dict:
                            existing_map_dict[cls_name] = ""
                    merged_mapping = pd.DataFrame(
                        {
                            "Original": list(existing_map_dict.keys()),
                            "New": list(existing_map_dict.values()),
                        }
                    )
                    st.session_state["tools_mr_mapping"] = merged_mapping
                _store_mr_scan_cache(
                    cached_params,
                    cached_scan_key=cached_scan_key,
                    geojson_dir_key=geojson_dir_key,
                    unified_csv_key=unified_csv_key,
                    file_pattern=file_pattern,
                    summary_df=summary_df,
                    assignment_df=assignment_df,
                    mapping_df=st.session_state.get("tools_mr_mapping"),
                )
                try:
                    with open(tools_cache, "w") as f:
                        json.dump(cached_params, f, indent=2)
                except Exception:
                    pass
                if assignment_df.empty:
                    st.warning("No GeoJSON files found.")
                elif summary_df is None or summary_df.empty:
                    st.warning("GeoJSON files were found, but no valid polygon annotations could be extracted.")
                else:
                    unmatched_files = int(assignment_df["assigned_batch"].astype(str).str.strip().eq("").sum())
                    if unmatched_files:
                        st.warning(
                            f"GeoJSON scan complete. Review batch assignment for {unmatched_files} file(s) before merging."
                        )
                    else:
                        st.success("GeoJSON scan complete.")
                    _log_console(f"[{_timestamp()}] TOOLS-MR: Scanned {len(assignment_df)} GeoJSON file(s)")

            assignment_df = _ensure_mr_assignment_columns(st.session_state.get("tools_mr_assignments"))
            if not assignment_df.empty:
                st.markdown("#### GeoJSON → Batch Assignment")
                st.caption(
                    "Auto-matches are inferred from the GeoJSON filename. Adjust Assigned Batch manually if needed; blank rows are skipped during merge."
                )
                batch_options = [""] + batch_values
                for value in assignment_df["assigned_batch"].tolist():
                    if value and value not in batch_options:
                        batch_options.append(value)

                assignment_editor_df = assignment_df[
                    ["file", "derived_batch", "assigned_batch", "match_source", "annotated_cells", "classified_cells"]
                ].copy()
                with st.form("tools_mr_assignment_form", clear_on_submit=False):
                    edited_assignments = st.data_editor(
                        assignment_editor_df,
                        hide_index=True,
                        num_rows="fixed",
                        key="tools_mr_assignment_editor",
                        column_order=[
                            "file",
                            "derived_batch",
                            "assigned_batch",
                            "match_source",
                            "annotated_cells",
                            "classified_cells",
                        ],
                        column_config={
                            "file": st.column_config.TextColumn("GeoJSON File", disabled=True),
                            "derived_batch": st.column_config.TextColumn("Detected Batch", disabled=True),
                            "assigned_batch": st.column_config.SelectboxColumn(
                                "Assigned Batch",
                                options=batch_options,
                                required=False,
                            ),
                            "match_source": st.column_config.TextColumn("Match Type", disabled=True),
                            "annotated_cells": st.column_config.NumberColumn("GeoJSON Cells", disabled=True, format="%d"),
                            "classified_cells": st.column_config.NumberColumn("Classified Cells", disabled=True, format="%d"),
                        },
                        disabled=["file", "derived_batch", "match_source", "annotated_cells", "classified_cells"],
                    )
                    save_assignments = st.form_submit_button("Save Batch Assignments", type="primary")
                if save_assignments:
                    updated_assignments = assignment_df.copy()
                    updated_assignments["assigned_batch"] = (
                        edited_assignments["assigned_batch"].fillna("").astype(str).str.strip()
                    )
                    match_sources: list[str] = []
                    for idx, row in updated_assignments.iterrows():
                        assigned_batch = str(row.get("assigned_batch", "")).strip()
                        auto_batch = str(row.get("auto_batch", "")).strip()
                        source = str(row.get("match_source", "")).strip()
                        if not assigned_batch:
                            match_sources.append("unmatched")
                        elif auto_batch and assigned_batch == auto_batch and source not in {"", "manual", "unmatched"}:
                            match_sources.append(source)
                        elif auto_batch and assigned_batch == auto_batch:
                            match_sources.append("exact")
                        else:
                            match_sources.append("manual")
                    updated_assignments["match_source"] = match_sources
                    st.session_state["tools_mr_assignments"] = updated_assignments
                    st.session_state["tools_mr_summary"] = _apply_assignments_to_summary(
                        st.session_state.get("tools_mr_summary"),
                        updated_assignments,
                    )
                    _store_mr_scan_cache(
                        cached_params,
                        cached_scan_key=cached_scan_key,
                        geojson_dir_key=geojson_dir_key,
                        unified_csv_key=unified_csv_key,
                        file_pattern=file_pattern,
                        summary_df=st.session_state.get("tools_mr_summary"),
                        assignment_df=updated_assignments,
                        mapping_df=st.session_state.get("tools_mr_mapping"),
                    )
                    try:
                        with open(tools_cache, "w") as f:
                            json.dump(cached_params, f, indent=2)
                        from spacemap_gui.persistence import save_active_project_state
                        save_active_project_state("tab7_phenotyping")
                    except Exception as ex:
                        logger.warning(f"Failed to auto-save tools cache or project state: {ex}")
                    st.success("Batch assignments saved.")

            summary_df = st.session_state.get("tools_mr_summary")
            if summary_df is not None and not summary_df.empty:
                summary_df = _apply_assignments_to_summary(summary_df, st.session_state.get("tools_mr_assignments"))
                st.dataframe(summary_df, width="stretch", hide_index=True)

            mapping_df = st.session_state.get("tools_mr_mapping")
            if mapping_df is None or mapping_df.empty:
                if summary_df is not None and not summary_df.empty:
                    unique_classes = sorted(summary_df["classification"].dropna().unique().tolist())
                    mapping_df = pd.DataFrame({"Original": unique_classes, "New": [""] * len(unique_classes)})
                else:
                    mapping_df = pd.DataFrame(columns=["Original", "New"])
            mapping_df = mapping_df.reset_index(drop=True)
            with st.form("tools_mr_mapping_form", clear_on_submit=False):
                st.caption("Relabel classifications before merging. Leave New empty to keep original.")
                edited_mapping = st.data_editor(
                    mapping_df,
                    num_rows="dynamic",
                    hide_index=True,
                    key="tools_mr_mapping_editor",
                )
                save_mapping = st.form_submit_button("Save Relabel Mapping", type="primary")
            if save_mapping:
                st.session_state["tools_mr_mapping"] = edited_mapping
                _store_mr_scan_cache(
                    cached_params,
                    cached_scan_key=cached_scan_key,
                    geojson_dir_key=geojson_dir_key,
                    unified_csv_key=unified_csv_key,
                    file_pattern=file_pattern,
                    summary_df=summary_df,
                    assignment_df=st.session_state.get("tools_mr_assignments"),
                    mapping_df=edited_mapping,
                )
                try:
                    with open(tools_cache, "w") as f:
                        json.dump(cached_params, f, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                except Exception as ex:
                    logger.warning(f"Failed to auto-save tools cache or project state: {ex}")
                st.success("Relabel mapping saved.")

            exclude_new_names_input = st.text_input(
                "Exclude Phenotype Names (post-relabel, comma-separated)",
                value=", ".join(cached_params.get("mr_exclude_new_names", [])),
                key="tools_mr_exclude_new_names",
                help="Names to exclude after relabeling. Applies to New name if provided, otherwise Original.",
            )
            cached_params["mr_exclude_new_names"] = [
                n.strip() for n in exclude_new_names_input.split(",") if n.strip()
            ]
            try:
                with open(tools_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass

            st.divider()
            st.markdown("#### 4. GeoJSON → Unified Merge")
            st.caption(
                "Direct merge uses GeoJSON polygon centroids together with the selected batch and coordinate columns. Exported *_ground_truth.csv files are optional and no longer required."
            )

            if st.button("Merge GeoJSON into Unified CSV", type="primary", key="tools_mr_merge"):
                _log_console(f"[{_timestamp()}] TOOLS-MR: Merge button clicked")
                status_box = st.empty()
                status_box.info("Starting GeoJSON merge...")
                if not unified_csv_path or not os.path.exists(unified_csv_path):
                    st.error("Unified CSV not found.")
                    return
                if not geojson_dir or not os.path.isdir(geojson_dir):
                    st.error("GeoJSON directory not found.")
                    return
                else:
                    assignment_df = _ensure_mr_assignment_columns(st.session_state.get("tools_mr_assignments"))
                    if assignment_df.empty:
                        scanned_summary_df, assignment_df = _scan_geojson_summary(
                            geojson_dir,
                            file_pattern,
                            batch_values,
                            batch_id_col,
                        )
                        assignment_df = _apply_saved_mr_assignments(
                            assignment_df,
                            st.session_state.get("tools_mr_assignments"),
                        )
                        st.session_state["tools_mr_assignments"] = assignment_df
                        st.session_state["tools_mr_summary"] = _apply_assignments_to_summary(
                            scanned_summary_df,
                            assignment_df,
                        )
                        _store_mr_scan_cache(
                            cached_params,
                            cached_scan_key=cached_scan_key,
                            geojson_dir_key=geojson_dir_key,
                            unified_csv_key=unified_csv_key,
                            file_pattern=file_pattern,
                            summary_df=st.session_state.get("tools_mr_summary"),
                            assignment_df=assignment_df,
                            mapping_df=st.session_state.get("tools_mr_mapping"),
                        )
                        try:
                            with open(tools_cache, "w") as f:
                                json.dump(cached_params, f, indent=2)
                        except Exception:
                            pass

                    if assignment_df.empty:
                        st.error("No GeoJSON files found in the selected directory.")
                        return
                    if assignment_df["assigned_batch"].astype(str).str.strip().eq("").all():
                        st.error("No batch assignments are set. Click 'Scan GeoJSON' and choose a batch for each GeoJSON file before merging.")
                        return

                    _log_console(f"[{_timestamp()}] TOOLS-MR: Loading unified CSV")
                    pre_df = None
                    if os.path.exists(unified_csv_path) and phenotype_col:
                        try:
                            pre_df = pd.read_csv(unified_csv_path, usecols=[phenotype_col])
                        except Exception:
                            pre_df = None

                    _log_console(f"[{_timestamp()}] TOOLS-MR: Merging GeoJSON into unified CSV")
                    mapping_df = st.session_state.get("tools_mr_mapping")
                    mapping: dict[str, str] = {}
                    if mapping_df is not None and not mapping_df.empty:
                        mapping = {
                            str(row.get("Original")): str(row.get("New"))
                            for _, row in mapping_df.iterrows()
                            if row.get("Original") and row.get("New")
                        }
                    if not mapping:
                        cached_scan = cached_params.get(cached_scan_key) if unified_csv_key and geojson_dir_key else None
                        cached_mapping = cached_scan.get("mapping", []) if isinstance(cached_scan, dict) else []
                        if cached_mapping:
                            mapping = {
                                str(row.get("Original")): str(row.get("New"))
                                for row in cached_mapping
                                if row.get("Original") and row.get("New")
                            }

                    exclude_new_list = cached_params.get("mr_exclude_new_names", [])
                    if isinstance(exclude_new_list, str):
                        exclude_new_list = [n.strip() for n in exclude_new_list.split(",") if n.strip()]

                    override_existing_effective = True if overwrite_entire_column else override_existing
                    run_audit = begin_phenotyping_run_audit(
                        Path(project_root),
                        "tools",
                        input_csv=unified_csv_path,
                        event_type="geojson_unified_merge",
                    )

                    try:
                        merged_df, merge_stats_df = _merge_geojson_into_unified(
                            assignment_df=assignment_df,
                            unified_csv_path=unified_csv_path,
                            phenotype_col_name=phenotype_col,
                            batch_id_column=batch_id_col,
                            x_centroid_column=x_col,
                            y_centroid_column=y_col,
                            tolerance_fraction=tolerance_fraction,
                            override_existing=override_existing_effective,
                            overwrite_entire_column=overwrite_entire_column,
                            exclude_classifications=cached_params.get("mr_exclude_classes", []),
                            relabel_mapping=mapping,
                            exclude_new_names=exclude_new_list,
                        )
                    except Exception as exc:  # noqa: BLE001
                        finish_phenotyping_run_audit(
                            run_audit,
                            status="failed",
                            output_path=unified_csv_path,
                            output_column=phenotype_col,
                            error=str(exc),
                        )
                        st.error(f"Merge failed: {exc}")
                        raise

                    if not merge_stats_df.empty:
                        st.markdown("#### GeoJSON Merge Details")
                        st.dataframe(
                            merge_stats_df,
                            width="stretch",
                            hide_index=True,
                        )

                    if phenotype_col in merged_df.columns:
                        pre_series = None
                        if pre_df is not None and phenotype_col in pre_df.columns:
                            pre_series = pre_df[phenotype_col]
                        else:
                            pre_series = pd.Series([pd.NA] * len(merged_df))
                        newly_labeled = pre_series.isna() & merged_df[phenotype_col].notna()
                        added = int(newly_labeled.sum())
                        total_labeled = int(merged_df[phenotype_col].notna().sum())
                    else:
                        added = 0
                        total_labeled = 0

                    if phenotype_col in merged_df.columns:
                        phenotype_counts = (
                            merged_df[phenotype_col]
                            .fillna("unclassified")
                            .astype(str)
                            .value_counts()
                        )
                        total_cells = int(phenotype_counts.sum()) if not phenotype_counts.empty else 0
                        summary_df = pd.DataFrame(
                            {
                                "phenotype": phenotype_counts.index,
                                "count": phenotype_counts.values,
                                "% total": [f"{(c / total_cells * 100):.2f}%" if total_cells else "0.00%" for c in phenotype_counts.values],
                            }
                        )
                        st.markdown("#### MR Merge Summary")
                        st.dataframe(summary_df, width="stretch", hide_index=True)

                    if added == 0:
                        status_box.info(
                            "Merge completed. No new labels were added. Check batch assignment, coordinate tolerance, or overwrite settings."
                        )
                        _log_console(
                            f"[{_timestamp()}] TOOLS-MR: Merge completed (0 new cells, total labeled={total_labeled})"
                        )
                    else:
                        status_box.success(f"Merged MR into unified file: {unified_csv_path} (+{added} cells)")
                        _log_console(
                            f"[{_timestamp()}] TOOLS-MR: Merge completed (+{added} cells, total labeled={total_labeled})"
                        )
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="success",
                        output_path=unified_csv_path,
                        output_rows=len(merged_df),
                        output_column=phenotype_col,
                        extra={
                            "newly_labeled_cells": added,
                            "total_labeled_cells": total_labeled,
                            "geojson_files": int(len(assignment_df)),
                        },
                    )

            st.divider()
            st.markdown("#### 5. Optional CSV Utilities")
            st.caption("These helpers export or enrich *_ground_truth.csv files for external inspection. They are not required for direct merge.")
            step1, step2, step3 = st.columns(3)
            with step1:
                if st.button("Export Ground Truth CSVs", key="tools_mr_extract", disabled=batch_process_geojson_files is None):
                    if not geojson_dir:
                        st.error("Please select a GeoJSON directory.")
                    else:
                        _log_console(f"[{_timestamp()}] TOOLS-MR: Extracting GeoJSON -> Manual Reference CSV")
                        batch_process_geojson_files(geojson_dir, geojson_dir, file_pattern=file_pattern)
                        st.success("Extraction complete.")
            with step2:
                if st.button("Relabel Exported CSVs", key="tools_mr_relabel", disabled=relabel_ground_truth_files is None):
                    mapping_df = st.session_state.get("tools_mr_mapping")
                    if mapping_df is None or mapping_df.empty:
                        st.warning("No relabel mapping provided.")
                    else:
                        mapping = {row["Original"]: row["New"] for _, row in mapping_df.iterrows() if row.get("Original") and row.get("New")}
                        if not mapping:
                            st.warning("No valid mapping rows.")
                        else:
                            _log_console(f"[{_timestamp()}] TOOLS-MR: Relabeling classifications")
                            relabel_ground_truth_files(geojson_dir, mapping, file_pattern=file_pattern)
                            st.success("Relabeling complete.")
            with step3:
                if st.button(
                    "Enrich Exported CSVs with Unified",
                    key="tools_mr_enrich",
                    disabled=enrich_ground_truth_with_unified_data is None,
                ):
                    if not unified_csv_path or not os.path.exists(unified_csv_path):
                        st.error("Unified CSV not found.")
                    else:
                        _log_console(f"[{_timestamp()}] TOOLS-MR: Enriching manual reference with unified data")
                        enrich_ground_truth_with_unified_data(
                            gt_output_dir=geojson_dir,
                            unified_csv_path=unified_csv_path,
                            file_pattern="*_ground_truth.csv",
                            tolerance_fraction=tolerance_fraction,
                            batch_id_column=batch_id_col,
                            x_centroid_column=x_col,
                            y_centroid_column=y_col,
                        )
                        st.success("Enrichment complete.")

    with sub_tabs[1]:
        st.markdown("### Phenotype GeoJSON")
        st.caption(
            "Tag the segmented cells in GeoJSON files with any per-cell property from the unified CSV "
            "-- a phenotype call, a cluster id, a region label -- or with how well two phenotype "
            "columns agree, then render a PNG of the tagged section."
        )

        if annotate_geojson_with_phenotypes is None or generate_png_from_geojson is None:
            st.error("The GeoJSON tagging helpers could not be imported from `spacemap.acc_utils`.")
            if ACC_UTILS_IMPORT_ERROR is not None:
                st.code(f"{type(ACC_UTILS_IMPORT_ERROR).__name__}: {ACC_UTILS_IMPORT_ERROR}")
            st.caption(
                "Check that `src/spacemap/acc_utils.py` is present and that shapely and matplotlib "
                "are installed in the environment running the GUI."
            )
        else:
            unified_csv_path = global_input_path or ""
            if unified_csv_path:
                st.caption(f"Unified CSV: `{unified_csv_path}`")
            else:
                st.warning("No unified CSV selected in Global Input Configuration.")

            available_columns = _get_columns_cached(unified_csv_path) if unified_csv_path else []

            # ---------------------------------------------------------------
            # 1. Inputs
            # ---------------------------------------------------------------
            st.markdown("#### 1. Inputs")

            viz_seg_dir = str(st.session_state.get("viz_seg_dir", "") or "")
            geojson_dir_default = (
                cached_params.get("tools_pg_geojson_dir")
                or viz_seg_dir
                or str(project_root / "outputs/segmentation/cellpose")
            )
            # Re-sync happens before the text input is created, otherwise Streamlit
            # refuses to overwrite a live widget value.
            if st.session_state.pop("pg_sync_seg_dir", False) and viz_seg_dir:
                st.session_state["tools_pg_geojson_dir"] = viz_seg_dir

            input_col1, input_col2 = st.columns([3, 1.6])
            with input_col1:
                geojson_dir = directory_input_with_browse(
                    label="Segmentation GeoJSON Directory",
                    text_key="tools_pg_geojson_dir",
                    default_value=geojson_dir_default,
                    browser_key_prefix="tree_tools_pg_geojson_dir",
                    help_text=(
                        "Folder holding the segmented-cell GeoJSON files. Defaults to the Segmentation "
                        "Directory set in Visual Validation Settings above."
                    ),
                )
            with input_col2:
                geojson_pattern = st.text_input(
                    "File Pattern",
                    value=cached_params.get("tools_pg_geojson_pattern", "*_cell_boundaries.geojson"),
                    key="tools_pg_geojson_pattern",
                    help="Glob pattern used to find the segmentation files.",
                )
                if viz_seg_dir and viz_seg_dir != geojson_dir:
                    if st.button(
                        "Use Visual Validation Folder",
                        key="tools_pg_use_viz_seg_dir",
                        help=f"Reset to {viz_seg_dir}",
                    ):
                        st.session_state["pg_sync_seg_dir"] = True
                        st.rerun()

            st.markdown("##### Key Columns")
            st.caption(
                "Seeded from Visual Validation Settings; override them here when this tab needs a "
                "different batch or coordinate column than the rest of the phenotyping tabs."
            )
            key_col1, key_col2, key_col3 = st.columns(3)
            with key_col1:
                batch_id_col = _column_selectbox(
                    "Batch Column",
                    available_columns,
                    str(cached_params.get("tools_pg_batch_col") or st.session_state.get("viz_col_batch") or "batch_id"),
                    "tools_pg_batch_col",
                    "Column that separates sections. 'Scan Batches' lists its distinct values.",
                )
            with key_col2:
                x_col = _column_selectbox(
                    "X Centroid Column",
                    available_columns,
                    str(cached_params.get("tools_pg_x_col") or st.session_state.get("viz_col_x") or "x_centroid"),
                    "tools_pg_x_col",
                )
            with key_col3:
                y_col = _column_selectbox(
                    "Y Centroid Column",
                    available_columns,
                    str(cached_params.get("tools_pg_y_col") or st.session_state.get("viz_col_y") or "y_centroid"),
                    "tools_pg_y_col",
                )
            missing_shared_cols = [
                col for col in (batch_id_col, x_col, y_col)
                if available_columns and col not in available_columns
            ]
            if missing_shared_cols:
                st.error(
                    "These columns are not in the unified CSV: "
                    + ", ".join(f"`{col}`" for col in missing_shared_cols)
                )

            # ---------------------------------------------------------------
            # 2. What to tag
            # ---------------------------------------------------------------
            st.markdown("#### 2. Property to Tag")
            phenotype_columns = [col for col in available_columns if col.startswith("phenotype_")]
            label_options = phenotype_columns + [
                col for col in available_columns
                if col not in phenotype_columns and col not in {x_col, y_col}
            ]

            tag_mode = st.radio(
                "Tagging Mode",
                ["Single column", "Compare two columns"],
                index=1 if cached_params.get("tools_pg_mode") == "Compare two columns" else 0,
                horizontal=True,
                key="tools_pg_mode",
                help=(
                    "Single column writes each cell's own value. Compare scores one column against a "
                    "reference and writes whether they agree."
                ),
            )
            compare_mode = tag_mode == "Compare two columns"

            na_text = st.text_input(
                "Values Treated as Missing",
                value=cached_params.get("tools_pg_na_tokens", ", ".join(t for t in DEFAULT_NA_TOKENS if t)),
                key="tools_pg_na_tokens",
                help=(
                    "Comma-separated, case-insensitive. Empty cells and real NaN always count as missing. "
                    "Cells with a missing value are never tagged, so they are removed (or labelled "
                    "unassigned) instead of being written out as a phenotype called \"None\"."
                ),
            )
            na_tokens = _parse_na_tokens(na_text)

            ref_col = comparison_col = ""
            match_label = mismatch_label = ""
            passthrough_values: list[str] = []
            if compare_mode:
                cmp_col1, cmp_col2 = st.columns(2)
                with cmp_col1:
                    ref_col = _column_selectbox(
                        "Reference Column",
                        label_options,
                        str(cached_params.get("tools_pg_ref_col") or (label_options[0] if label_options else "")),
                        "tools_pg_ref_col",
                        "The call to be trusted, e.g. a manual reference.",
                    )
                with cmp_col2:
                    default_cmp = str(cached_params.get("tools_pg_cmp_col") or "")
                    cmp_options = [col for col in label_options if col != ref_col]
                    comparison_col = _column_selectbox(
                        "Comparison Column",
                        cmp_options,
                        default_cmp if default_cmp in cmp_options else (cmp_options[0] if cmp_options else ""),
                        "tools_pg_cmp_col",
                        "The call being scored against the reference.",
                    )

                label_col = _AGREEMENT_LABEL_COL
                scan_source_col = ref_col

                lbl_col1, lbl_col2 = st.columns(2)
                with lbl_col1:
                    match_label = st.text_input(
                        "Label When They Agree",
                        value=cached_params.get("tools_pg_match_label", "Match"),
                        key="tools_pg_match_label",
                    ) or "Match"
                with lbl_col2:
                    mismatch_label = st.text_input(
                        "Label When They Disagree",
                        value=cached_params.get("tools_pg_mismatch_label", "Mismatch"),
                        key="tools_pg_mismatch_label",
                    ) or "Mismatch"
                st.caption(
                    f"Cells where `{comparison_col}` reproduces `{ref_col}` become **{match_label}**; "
                    f"the rest become **{mismatch_label}**. A cell missing from `{ref_col}` is never "
                    f"scored, and one missing from `{comparison_col}` counts as a disagreement."
                )
            else:
                saved_label_col = cached_params.get(
                    "tools_pg_label_col", cached_params.get("tools_pg_phenotype_col", "")
                )
                if saved_label_col not in label_options:
                    saved_label_col = label_options[0] if label_options else ""
                label_col = st.selectbox(
                    "Column to Tag onto Cells",
                    label_options if label_options else [saved_label_col or "phenotype_leiden"],
                    index=label_options.index(saved_label_col) if saved_label_col in label_options else 0,
                    key="tools_pg_label_col",
                    help=(
                        "Any column of the unified CSV. Phenotype columns are listed first, but a cluster id, "
                        "a region or any other categorical property works the same way."
                    ),
                )
                scan_source_col = label_col

            # ---------------------------------------------------------------
            # 3. Cell filters
            # ---------------------------------------------------------------
            st.markdown("#### 3. Filter Cells (optional)")
            st.caption("Only cells passing every criterion are tagged. Criteria are combined with AND.")
            active_filters = _render_filter_criteria(
                "tools_pg",
                available_columns,
                unified_csv_path,
                empty_caption="No filter criteria added. Every cell in the unified CSV is eligible.",
            )
            if active_filters:
                st.caption(
                    "Active: "
                    + "; ".join(_describe_filter_criterion(criterion) for criterion in active_filters)
                )

            # ---------------------------------------------------------------
            # 4. Colors
            # ---------------------------------------------------------------
            st.markdown("#### 4. Label Colors")
            st.caption(
                "Known cell types start from the house palette; everything else gets a palette color. "
                "Any change is saved per label in the project params file, so the same phenotype -- "
                "and the same Match / Mismatch labels -- keep their color across columns and runs."
            )
            color_registry = _load_pg_color_registry(cached_params)

            scan_cache_key = f"pg_label_values::{unified_csv_path}::{scan_source_col}"
            scanned_values: list[str] = st.session_state.get(scan_cache_key, [])
            # Anything the run would treat as missing must not be offered as a label.
            na_lookup = {token.strip().lower() for token in na_tokens}
            scanned_values = [
                value for value in scanned_values if str(value).strip().lower() not in na_lookup
            ]

            scan_col1, scan_col2 = st.columns([1, 3])
            with scan_col1:
                if st.button(
                    "Scan Labels",
                    key="tools_pg_scan_labels",
                    disabled=not unified_csv_path or scan_source_col not in available_columns,
                    help="Read the distinct values of the selected column so each can be given a color.",
                ):
                    with st.spinner(f"Reading distinct values of {scan_source_col}..."):
                        st.session_state[scan_cache_key] = _get_unique_values_cached(
                            unified_csv_path, scan_source_col
                        )
                        st.rerun()
            with scan_col2:
                if scanned_values:
                    st.caption(f"{len(scanned_values)} usable value(s) in `{scan_source_col}`.")
                else:
                    st.caption(f"Click **Scan Labels** to read the values of `{scan_source_col}`.")

            if compare_mode:
                passthrough_values = st.multiselect(
                    "Reference Values Kept As-Is (never scored)",
                    scanned_values,
                    default=[
                        value
                        for value in cached_params.get("tools_pg_passthrough", [])
                        if value in scanned_values
                    ],
                    key="tools_pg_passthrough",
                    help=(
                        "Reference calls that were never confident enough to score, such as 'Others'. "
                        "These cells keep their own name and color no matter what the comparison column says."
                    ),
                )
                known_labels = [match_label, mismatch_label] + [
                    value for value in passthrough_values if value not in {match_label, mismatch_label}
                ]
                default_palette_overrides = {
                    match_label: DEFAULT_MATCH_COLOR,
                    mismatch_label: DEFAULT_MISMATCH_COLOR,
                }
                for value in passthrough_values:
                    default_palette_overrides.setdefault(value, DEFAULT_IGNORED_COLOR)
            else:
                known_labels = list(scanned_values)
                default_palette_overrides = {}

            # A categorical filter on the tagged column itself narrows what can be written,
            # so there is no point offering colors for values that cannot appear.
            if not compare_mode:
                selected_label_values = {
                    str(value)
                    for criterion in active_filters
                    if criterion["column"] == label_col and criterion.get("type") == "categorical"
                    for value in criterion.get("values", [])
                }
                if selected_label_values:
                    known_labels = [label for label in known_labels if label in selected_label_values]

            if st.session_state.pop("pg_reset_colors", False):
                # Forget the saved color too, otherwise the registry would hand the
                # same color straight back on the next line.
                for label in known_labels:
                    st.session_state.pop(_color_widget_key(label), None)
                    color_registry.pop(str(label), None)
                cached_params[_PG_COLOR_REGISTRY_KEY] = color_registry

            color_mapping: dict[str, str] = {}
            if known_labels:
                if len(known_labels) > _MAX_COLOR_PICKERS:
                    st.warning(
                        f"`{scan_source_col}` has {len(known_labels)} distinct values; showing color pickers "
                        f"for the first {_MAX_COLOR_PICKERS}. The rest use the default palette."
                    )
                editable_labels = known_labels[:_MAX_COLOR_PICKERS]
                # A color already on record wins; then the mode's own defaults
                # (green/red for an agreement map); then the shared palette.
                seed_colors = dict(default_palette_overrides)
                seed_colors.update(_registry_colors_for(color_registry, known_labels))
                palette = build_color_mapping(known_labels, existing=seed_colors)
                picker_cols = st.columns(4)
                for idx, label in enumerate(editable_labels):
                    widget_key = _color_widget_key(label)
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = palette.get(label, "#000000")
                    with picker_cols[idx % 4]:
                        color_mapping[label] = st.color_picker(label, key=widget_key)
                for label in known_labels[_MAX_COLOR_PICKERS:]:
                    color_mapping[label] = palette.get(label, "#000000")
                # Newly drawn palette colors are recorded now, not only after an
                # edit, so a label keeps its first color for good.
                _remember_pg_colors(cached_params, color_registry, color_mapping)
                if st.button("Reset Colors to Default Palette", key="tools_pg_reset_colors_btn"):
                    st.session_state["pg_reset_colors"] = True
                    st.rerun()

            # ---------------------------------------------------------------
            # 5. Output
            # ---------------------------------------------------------------
            st.markdown("#### 5. Output")
            annotated_dir = directory_input_with_browse(
                label="Annotated GeoJSON Output Directory",
                text_key="tools_pg_annotated_dir",
                default_value=cached_params.get(
                    "tools_pg_annotated_dir",
                    str(project_root / "outputs/segmentation/cellpose/phenotyped"),
                ),
                browser_key_prefix="tree_tools_pg_annotated_dir",
                help_text="Tagged copies are written here; the source GeoJSON files are never modified.",
            )
            png_dir = directory_input_with_browse(
                label="PNG Output Directory",
                text_key="tools_pg_png_dir",
                default_value=cached_params.get(
                    "tools_pg_png_dir",
                    str(project_root / "outputs/segmentation/cellpose/phenotyped/png"),
                ),
                browser_key_prefix="tree_tools_pg_png_dir",
            )

            out_col1, out_col2 = st.columns(2)
            with out_col1:
                geojson_output_suffix = st.text_input(
                    "GeoJSON Output Suffix",
                    value=cached_params.get("tools_pg_geojson_suffix", "_phenotyped"),
                    key="tools_pg_geojson_suffix",
                    help="Appended to the source filename so the original segmentation is preserved.",
                )
                drop_unassigned = st.checkbox(
                    "Remove non-assigned cells from the output GeoJSON",
                    value=bool(cached_params.get("tools_pg_drop_unassigned", True)),
                    key="tools_pg_drop_unassigned",
                    help=(
                        "On: only cells that received a value are written. Cells whose value is missing "
                        "count as non-assigned. Off: every segmented cell is kept and the rest get the "
                        "label below."
                    ),
                )
                unassigned_label = DEFAULT_UNASSIGNED_LABEL
                unassigned_color = DEFAULT_UNASSIGNED_COLOR
                if not drop_unassigned:
                    unassigned_col1, unassigned_col2 = st.columns([2, 1])
                    with unassigned_col1:
                        unassigned_label = st.text_input(
                            "Label for Non-Assigned Cells",
                            value=cached_params.get("tools_pg_unassigned_label", DEFAULT_UNASSIGNED_LABEL),
                            key="tools_pg_unassigned_label",
                        ) or DEFAULT_UNASSIGNED_LABEL
                    with unassigned_col2:
                        unassigned_color = st.color_picker(
                            "Color",
                            value=cached_params.get("tools_pg_unassigned_color", DEFAULT_UNASSIGNED_COLOR),
                            key="tools_pg_unassigned_color",
                        )
            with out_col2:
                make_png = st.checkbox(
                    "Generate a representative PNG per file",
                    value=bool(cached_params.get("tools_pg_make_png", True)),
                    key="tools_pg_make_png",
                )
                mirror_y_axis = st.checkbox(
                    "Mirror Y Axis (PNG)",
                    value=bool(cached_params.get("tools_pg_mirror_y", True)),
                    key="tools_pg_mirror_y",
                    help="Enable when the GeoJSON coordinates use an image origin at top-left.",
                    disabled=not make_png,
                )
                png_dpi = int(
                    st.number_input(
                        "PNG DPI",
                        min_value=50,
                        max_value=600,
                        value=int(cached_params.get("tools_pg_png_dpi", 150)),
                        step=25,
                        key="tools_pg_png_dpi",
                        disabled=not make_png,
                    )
                )

            with st.expander("PNG Appearance", expanded=False):
                st.caption(
                    "On a dense section the cell outline can cover the fill entirely. Set the width to 0 "
                    "to drop the outline and show the phenotype colors alone."
                )
                style_col1, style_col2, style_col3, style_col4 = st.columns(4)
                with style_col1:
                    png_line_width = float(
                        st.number_input(
                            "Boundary Width",
                            min_value=0.0,
                            max_value=3.0,
                            value=float(cached_params.get("tools_pg_png_line_width", 0.0)),
                            step=0.05,
                            format="%.2f",
                            key="tools_pg_png_line_width",
                            disabled=not make_png,
                        )
                    )
                with style_col2:
                    png_edge_color = st.color_picker(
                        "Boundary Color",
                        value=cached_params.get("tools_pg_png_edge_color", "#FFFFFF"),
                        key="tools_pg_png_edge_color",
                        disabled=not make_png or png_line_width <= 0,
                    )
                with style_col3:
                    png_alpha = float(
                        st.slider(
                            "Fill Opacity",
                            min_value=0.1,
                            max_value=1.0,
                            value=float(cached_params.get("tools_pg_png_alpha", 1.0)),
                            step=0.05,
                            key="tools_pg_png_alpha",
                            disabled=not make_png,
                        )
                    )
                with style_col4:
                    png_background = st.color_picker(
                        "Background",
                        value=cached_params.get("tools_pg_png_background", "#FFFFFF"),
                        key="tools_pg_png_background",
                        disabled=not make_png,
                    )

            cached_params.update(
                {
                    "tools_pg_geojson_dir": geojson_dir,
                    "tools_pg_annotated_dir": annotated_dir,
                    "tools_pg_png_dir": png_dir,
                    "tools_pg_geojson_pattern": geojson_pattern,
                    "tools_pg_batch_col": batch_id_col,
                    "tools_pg_x_col": x_col,
                    "tools_pg_y_col": y_col,
                    "tools_pg_mode": tag_mode,
                    "tools_pg_na_tokens": na_text,
                    "tools_pg_ref_col": ref_col,
                    "tools_pg_cmp_col": comparison_col,
                    "tools_pg_match_label": match_label,
                    "tools_pg_mismatch_label": mismatch_label,
                    "tools_pg_mirror_y": mirror_y_axis,
                    "tools_pg_geojson_suffix": geojson_output_suffix,
                    "tools_pg_drop_unassigned": drop_unassigned,
                    "tools_pg_unassigned_label": unassigned_label,
                    "tools_pg_unassigned_color": unassigned_color,
                    _PG_COLOR_REGISTRY_KEY: color_registry,
                    "tools_pg_make_png": make_png,
                    "tools_pg_png_dpi": png_dpi,
                    "tools_pg_png_line_width": png_line_width,
                    "tools_pg_png_edge_color": png_edge_color,
                    "tools_pg_png_alpha": png_alpha,
                    "tools_pg_png_background": png_background,
                    "tools_pg_filters": active_filters,
                }
            )
            if not compare_mode:
                cached_params["tools_pg_label_col"] = label_col
            if not drop_unassigned:
                _remember_pg_colors(cached_params, color_registry, {unassigned_label: unassigned_color})
            try:
                with open(tools_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass

            # ---------------------------------------------------------------
            # 6. Match sections and run
            # ---------------------------------------------------------------
            st.markdown("#### 6. Match Sections and Run")

            batch_ids: list[str] = st.session_state.get("pg_batch_ids", [])
            batch_counts: dict[str, int] = st.session_state.get("pg_batch_counts", {})
            eligible_counts: dict[str, int] = st.session_state.get("pg_eligible_counts", {})

            batch_col1, batch_col2, batch_col3 = st.columns([1, 1.2, 2])
            with batch_col1:
                if st.button(
                    "Scan Batches",
                    key="tools_pg_scan_batches_btn",
                    disabled=not unified_csv_path or batch_id_col not in available_columns,
                    help=f"List the distinct values of `{batch_id_col}` and count the cells in each.",
                ):
                    with st.spinner(f"Scanning {batch_id_col}..."):
                        series = _read_single_column(unified_csv_path, batch_id_col)
                        if series is None:
                            st.session_state["pg_batch_ids"] = []
                            st.session_state["pg_batch_counts"] = {}
                        else:
                            counts = series.dropna().astype(str).value_counts()
                            st.session_state["pg_batch_ids"] = sorted(counts.index.tolist())
                            st.session_state["pg_batch_counts"] = {k: int(v) for k, v in counts.items()}
                        st.session_state.pop("pg_eligible_counts", None)
                        st.rerun()
            with batch_col2:
                if st.button(
                    "Count Taggable Cells",
                    key="tools_pg_count_eligible",
                    disabled=not batch_ids or bool(missing_shared_cols),
                    help="Apply the filters and drop missing values to show how many cells each section would tag.",
                ):
                    with st.spinner("Counting taggable cells..."):
                        count_cols = {batch_id_col}
                        count_cols.update(
                            [ref_col, comparison_col] if compare_mode else [label_col]
                        )
                        count_cols.update(criterion["column"] for criterion in active_filters)
                        count_cols = [col for col in available_columns if col in count_cols]
                        frame = _read_columns_fast(unified_csv_path, count_cols)
                        if frame is None:
                            st.session_state["pg_eligible_counts"] = {}
                        else:
                            frame = _apply_visual_filters(frame, active_filters)
                            if compare_mode:
                                usable = ~missing_label_mask(frame[ref_col], na_tokens)
                            else:
                                usable = ~missing_label_mask(frame[label_col], na_tokens)
                            counted = frame.loc[usable, batch_id_col].astype(str).value_counts()
                            st.session_state["pg_eligible_counts"] = {
                                k: int(v) for k, v in counted.items()
                            }
                        st.rerun()
            with batch_col3:
                if batch_ids:
                    st.caption(f"{len(batch_ids)} distinct value(s) in `{batch_id_col}`.")
                else:
                    st.caption(f"Click **Scan Batches** to list the values of `{batch_id_col}`.")

            _reset_multiselect_when_options_change("tools_pg_selected_batches", batch_ids)
            selected_batches = st.multiselect(
                "Batches to Process",
                batch_ids,
                default=batch_ids,
                help="Each batch is matched to the GeoJSON file whose name best fits its ID.",
                key="tools_pg_selected_batches",
            )

            geojson_entries = _collect_geojson_entries(geojson_dir, geojson_pattern)
            if geojson_dir and not os.path.isdir(geojson_dir):
                st.error(f"Segmentation GeoJSON directory does not exist: `{geojson_dir}`")
            elif not geojson_entries:
                st.warning(f"No files matching `{geojson_pattern}` were found in `{geojson_dir}`.")

            match_rows: list[dict[str, Any]] = []
            for batch_id in selected_batches:
                match_path, match_prefix, match_score = _best_geojson_match(batch_id, geojson_entries)
                row: dict[str, Any] = {
                    "batch_id": batch_id,
                    "cells_in_batch": batch_counts.get(str(batch_id), ""),
                }
                if eligible_counts:
                    row["cells_to_tag"] = eligible_counts.get(str(batch_id), 0)
                row.update(
                    {
                        "geojson_file": os.path.basename(match_path) if match_path else "",
                        "geojson_path": match_path or "",
                        "match_score": match_score if match_score >= 0 else "",
                    }
                )
                match_rows.append(row)

            matched_rows = [row for row in match_rows if row["geojson_path"]]
            contested: set[str] = set()
            if match_rows:
                st.dataframe(
                    pd.DataFrame(match_rows).drop(columns=["geojson_path"]),
                    width="stretch",
                    hide_index=True,
                )
                if eligible_counts:
                    st.caption(
                        f"`cells_to_tag` counts cells that pass the filters and carry a usable "
                        f"`{ref_col if compare_mode else label_col}` value."
                    )
                unmatched = [row["batch_id"] for row in match_rows if not row["geojson_path"]]
                if unmatched:
                    st.warning(
                        f"{len(unmatched)} batch(es) have no GeoJSON match and will be skipped: "
                        + ", ".join(str(b) for b in unmatched[:10])
                        + ("..." if len(unmatched) > 10 else "")
                    )
                # Two batches resolving to one file would write the same output name,
                # so the second run would silently overwrite the first.
                contested = {
                    row["geojson_file"]
                    for row in matched_rows
                    if sum(1 for other in matched_rows if other["geojson_file"] == row["geojson_file"]) > 1
                }
                if contested:
                    st.error(
                        "These GeoJSON files are matched by more than one batch and would overwrite "
                        "each other: " + ", ".join(sorted(contested))
                        + ". Narrow the batch selection or the file pattern before running."
                    )
            else:
                st.info("Scan and select batches to see how they pair with the GeoJSON files.")

            run_disabled = not matched_rows or (compare_mode and not comparison_col)
            if st.button(
                "Tag GeoJSON with Selected Property",
                type="primary",
                key="tools_pg_run",
                disabled=run_disabled,
            ):
                if not unified_csv_path or not os.path.exists(unified_csv_path):
                    st.error("Unified CSV file not found.")
                elif missing_shared_cols:
                    st.error("Fix the batch and coordinate column names before running.")
                elif contested:
                    st.error("Resolve the duplicate batch-to-GeoJSON matches before running.")
                else:
                    source_cols = [ref_col, comparison_col] if compare_mode else [label_col]
                    absent = [col for col in source_cols if col not in available_columns]
                    if absent:
                        st.error("Column(s) not in the unified CSV: " + ", ".join(f"`{c}`" for c in absent))
                    else:
                        needed_cols = {batch_id_col, x_col, y_col, *source_cols}
                        needed_cols.update(criterion["column"] for criterion in active_filters)
                        usecols = [col for col in available_columns if col in needed_cols]

                        run_audit = begin_phenotyping_run_audit(
                            Path(project_root),
                            "tools",
                            input_csv=unified_csv_path,
                            event_type="geojson_agreement_tagging" if compare_mode else "geojson_phenotype_tagging",
                        )
                        status = st.status("Tagging GeoJSON files...", expanded=True)
                        results: list[dict[str, Any]] = []
                        try:
                            status.write(f"Reading {len(usecols)} column(s) from the unified CSV...")
                            full_df = _read_columns_fast(unified_csv_path, usecols)
                            if full_df is None:
                                raise ValueError(
                                    f"Could not read {', '.join(usecols)} from {unified_csv_path}"
                                )
                            rows_before = len(full_df)
                            full_df = _apply_visual_filters(full_df, active_filters)
                            if active_filters:
                                status.write(f"{len(full_df):,} of {rows_before:,} cells pass the filters.")
                            else:
                                status.write(f"{rows_before:,} cells loaded.")

                            if compare_mode:
                                full_df = full_df.assign(
                                    **{
                                        _AGREEMENT_LABEL_COL: build_agreement_labels(
                                            full_df[ref_col],
                                            full_df[comparison_col],
                                            match_label=match_label,
                                            mismatch_label=mismatch_label,
                                            passthrough_values=passthrough_values,
                                            na_values=na_tokens,
                                        )
                                    }
                                )
                                scored = full_df[_AGREEMENT_LABEL_COL].notna()
                                agreement = full_df.loc[scored, _AGREEMENT_LABEL_COL]
                                status.write(
                                    f"{int(scored.sum()):,} cells scored against `{ref_col}`: "
                                    + ", ".join(
                                        f"{label} {count:,}"
                                        for label, count in agreement.value_counts().items()
                                    )
                                )

                            os.makedirs(annotated_dir, exist_ok=True)
                            if make_png:
                                os.makedirs(png_dir, exist_ok=True)

                            title_suffix = (
                                f"{comparison_col} vs {ref_col}" if compare_mode else label_col
                            )
                            batch_series = full_df[batch_id_col].astype(str)
                            for row in matched_rows:
                                batch_id = row["batch_id"]
                                geojson_path = row["geojson_path"]
                                batch_df = full_df.loc[batch_series == str(batch_id)]
                                if batch_df.empty and drop_unassigned:
                                    status.write(f"No cells left for batch {batch_id} - skipped.")
                                    continue

                                base_name = os.path.splitext(os.path.basename(geojson_path))[0]
                                output_path = os.path.join(
                                    annotated_dir, f"{base_name}{geojson_output_suffix}.geojson"
                                )
                                png_path = (
                                    os.path.join(png_dir, f"{base_name}{geojson_output_suffix}.png")
                                    if make_png
                                    else None
                                )
                                status.write(f"Tagging {os.path.basename(geojson_path)} with batch {batch_id}...")
                                file_stats: dict[str, Any] = {}
                                ok = annotate_geojson_with_phenotypes(
                                    csv_path=unified_csv_path,
                                    dataframe=batch_df,
                                    geojson_path=geojson_path,
                                    output_path=output_path,
                                    phenotype_col=label_col,
                                    x_col=x_col,
                                    y_col=y_col,
                                    color_mapping=color_mapping or None,
                                    keep_unassigned=not drop_unassigned,
                                    unassigned_label=unassigned_label,
                                    unassigned_color=unassigned_color,
                                    na_values=na_tokens,
                                    png_output_path=png_path,
                                    generate_png=bool(make_png),
                                    mirror_y_axis=mirror_y_axis,
                                    dpi=png_dpi,
                                    png_title=f"{base_name} | {title_suffix}",
                                    png_edge_color=png_edge_color,
                                    png_line_width=png_line_width,
                                    png_alpha=png_alpha,
                                    png_background=png_background,
                                    stats=file_stats,
                                )
                                if not ok:
                                    status.write(f"Failed to tag {os.path.basename(geojson_path)} - see console log.")
                                    continue

                                results.append(
                                    {
                                        "batch_id": batch_id,
                                        "geojson_file": os.path.basename(geojson_path),
                                        "cells_with_value": file_stats.get("csv_rows", 0),
                                        "cells_missing_value": file_stats.get("csv_rows_missing_label", 0),
                                        "polygons": file_stats.get("total_polygons", 0),
                                        "tagged": file_stats.get("matched_polygons", 0),
                                        "non_assigned": file_stats.get("unassigned_polygons", 0),
                                        "written": file_stats.get("written_polygons", 0),
                                        "output_path": output_path,
                                        "png_path": png_path if png_path and os.path.exists(png_path) else "",
                                        "label_counts": file_stats.get("label_counts", {}),
                                    }
                                )
                        except Exception as exc:  # noqa: BLE001 - reported in the UI
                            status.update(label="Tagging failed", state="error")
                            st.error(f"GeoJSON tagging failed: {exc}")
                            logger.exception("Phenotype GeoJSON tagging failed")
                            finish_phenotyping_run_audit(run_audit, status="error", extra={"error": str(exc)})
                        else:
                            st.session_state["pg_tag_results"] = results
                            st.session_state["pg_tag_meta"] = {
                                "compare_mode": compare_mode,
                                "match_label": match_label,
                                "mismatch_label": mismatch_label,
                                "passthrough": list(passthrough_values),
                                "label_col": comparison_col if compare_mode else label_col,
                            }
                            st.session_state["pg_last_annotated_dir"] = annotated_dir
                            status.update(
                                label=f"Tagging complete - {len(results)} file(s)",
                                state="complete" if results else "error",
                            )
                            total_written = sum(r["written"] for r in results)
                            _log_console(
                                f"[{_timestamp()}] TOOLS-PG: Tagged {len(results)} GeoJSON file(s) "
                                f"with '{title_suffix}' "
                                f"({total_written} polygons written)"
                            )
                            finish_phenotyping_run_audit(
                                run_audit,
                                status="success" if results else "no_output",
                                output_path=annotated_dir,
                                output_rows=total_written,
                                output_column=comparison_col if compare_mode else label_col,
                                extra={
                                    "mode": tag_mode,
                                    "reference_column": ref_col,
                                    "tagged_geojson_files": [r["output_path"] for r in results],
                                    "png_files": [r["png_path"] for r in results if r["png_path"]],
                                    "filters": active_filters,
                                    "drop_unassigned": drop_unassigned,
                                },
                            )

            results = st.session_state.get("pg_tag_results", [])
            if results:
                st.markdown("##### Results")
                summary_df = pd.DataFrame(
                    [
                        {key: value for key, value in row.items() if key != "label_counts"}
                        for row in results
                    ]
                )
                st.dataframe(summary_df, width="stretch", hide_index=True)

                png_results = [row for row in results if row.get("png_path")]
                if png_results:
                    with st.expander(f"PNG Previews ({len(png_results)})", expanded=True):
                        preview_cols = st.columns(2)
                        for idx, row in enumerate(png_results[:_MAX_PNG_PREVIEWS]):
                            with preview_cols[idx % 2]:
                                st.image(row["png_path"], caption=os.path.basename(row["png_path"]))
                        if len(png_results) > _MAX_PNG_PREVIEWS:
                            st.caption(
                                f"Showing {_MAX_PNG_PREVIEWS} of {len(png_results)} PNGs. "
                                f"All of them are in `{png_dir}`."
                            )

                run_meta = st.session_state.get("pg_tag_meta", {})
                if run_meta.get("compare_mode"):
                    st.markdown("##### Agreement Summary")
                    summary = _agreement_summary(results, run_meta)
                    if not summary.empty:
                        st.dataframe(summary, width="stretch", hide_index=True)
                        totals = summary.iloc[-1]
                        ignored_names = ", ".join(run_meta.get("passthrough", [])) or "none"
                        st.caption(
                            f"Percentages are over the {int(totals['cells_scored']):,} scored cells. "
                            f"`ignored` ({int(totals['ignored']):,} cells, reference value in: {ignored_names}) "
                            "is held out of them, and `not_scored` counts segmented cells with no "
                            "reference call at all."
                        )

                label_totals: dict[str, int] = {}
                for row in results:
                    for label, count in (row.get("label_counts") or {}).items():
                        label_totals[label] = label_totals.get(label, 0) + int(count)
                if label_totals:
                    st.markdown("##### Cells per Label")
                    total_tagged = sum(label_totals.values()) or 1
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "label": label,
                                    "cells": count,
                                    "percent": round(100.0 * count / total_tagged, 2),
                                }
                                for label, count in sorted(label_totals.items(), key=lambda item: -item[1])
                            ]
                            + [{"label": "TOTAL", "cells": sum(label_totals.values()), "percent": 100.0}]
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            # ---------------------------------------------------------------
            # 7. Re-render PNGs from already tagged files
            # ---------------------------------------------------------------
            with st.expander("Re-render PNGs from Tagged GeoJSON", expanded=False):
                st.caption(
                    "Render a different label subset or a new color scheme from files that are "
                    "already tagged, without repeating the spatial matching."
                )
                tagged_dir = directory_input_with_browse(
                    label="Tagged GeoJSON Directory",
                    text_key="tools_pg_tagged_dir",
                    default_value=st.session_state.get("pg_last_annotated_dir", annotated_dir),
                    browser_key_prefix="tree_tools_pg_tagged_dir",
                )

                tagged_files: list[str] = []
                if tagged_dir and os.path.isdir(tagged_dir):
                    import glob

                    tagged_files = sorted(glob.glob(os.path.join(tagged_dir, "*.geojson")))
                _reset_multiselect_when_options_change("tools_pg_tagged_select", tagged_files)
                selected_tagged = st.multiselect(
                    "Tagged GeoJSON Files",
                    tagged_files,
                    default=tagged_files,
                    format_func=os.path.basename,
                    key="tools_pg_tagged_select",
                )

                if st.button("Scan Classifications", key="tools_pg_scan_classes", disabled=not selected_tagged):
                    with st.spinner("Reading classifications..."):
                        st.session_state["pg_class_map"] = _scan_geojson_classes(selected_tagged)
                    st.rerun()

                class_map: dict[str, str] = st.session_state.get("pg_class_map", {})
                render_colors: dict[str, str] = {}
                if class_map:
                    # Registry first, then whatever color the tagged file already
                    # carries, then the palette.
                    seeded = {
                        label: color
                        for label, color in class_map.items()
                        if color and color != "#000000"
                    }
                    seeded.update(_registry_colors_for(color_registry, class_map.keys()))
                    palette = build_color_mapping(class_map.keys(), existing=seeded)

                    # Step 4 already owns a picker for every label it listed. Drawing a
                    # second one here would collide on the widget key, and letting this
                    # panel write the registry too would clobber an edit made up there.
                    shared = [label for label in sorted(class_map) if label in color_mapping]
                    extra = [label for label in sorted(class_map) if label not in color_mapping]
                    for label in shared:
                        render_colors[label] = color_mapping[label]
                    if shared:
                        st.caption(
                            "Using the step 4 colors for: " + ", ".join(shared)
                        )
                    if extra:
                        picker_cols = st.columns(4)
                        for idx, label in enumerate(extra):
                            widget_key = _color_widget_key(label)
                            if widget_key not in st.session_state:
                                st.session_state[widget_key] = palette.get(label, "#000000")
                            with picker_cols[idx % 4]:
                                render_colors[label] = st.color_picker(label, key=widget_key)
                    _remember_pg_colors(cached_params, color_registry, render_colors)
                else:
                    st.caption("Scan the selected files to list the classifications they carry.")

                _reset_multiselect_when_options_change("tools_pg_selected_phenotypes", sorted(class_map))
                selected_render_labels = st.multiselect(
                    "Labels to Draw",
                    sorted(class_map),
                    default=sorted(class_map),
                    key="tools_pg_selected_phenotypes",
                )

                if st.button("Generate PNGs", key="tools_pg_generate_pngs", disabled=not selected_tagged):
                    run_audit = begin_phenotyping_run_audit(
                        Path(project_root),
                        "tools",
                        input_csv=tagged_dir,
                        event_type="geojson_png_generation",
                    )
                    os.makedirs(png_dir, exist_ok=True)
                    generated_paths: list[str] = []
                    progress = st.progress(0.0)
                    for idx, path in enumerate(selected_tagged, start=1):
                        base_name = os.path.splitext(os.path.basename(path))[0]
                        out_png = os.path.join(png_dir, f"{base_name}.png")
                        if generate_png_from_geojson(
                            geojson_path=path,
                            png_output_path=out_png,
                            include_labels=selected_render_labels or None,
                            color_mapping=render_colors or None,
                            mirror_y_axis=mirror_y_axis,
                            dpi=png_dpi,
                            title=base_name,
                            background_color=png_background,
                            edge_color=png_edge_color,
                            line_width=png_line_width,
                            alpha=png_alpha,
                        ):
                            generated_paths.append(out_png)
                        progress.progress(idx / len(selected_tagged))
                    progress.empty()

                    try:
                        with open(tools_cache, "w") as f:
                            json.dump(cached_params, f, indent=2)
                    except Exception:
                        pass

                    if generated_paths:
                        st.success(f"Generated {len(generated_paths)} PNG(s) in `{png_dir}`.")
                        preview_cols = st.columns(2)
                        for idx, out_png in enumerate(generated_paths[:_MAX_PNG_PREVIEWS]):
                            with preview_cols[idx % 2]:
                                st.image(out_png, caption=os.path.basename(out_png))
                    else:
                        st.warning("No PNGs were generated. Check the file and label selections.")
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="success" if generated_paths else "no_output",
                        output_path=png_dir,
                        output_rows=len(generated_paths),
                        extra={
                            "source_geojson_count": len(selected_tagged),
                            "selected_labels": selected_render_labels,
                            "generated_png_files": generated_paths,
                        },
                    )

    with sub_tabs[2]:
        st.markdown("### Edit CSV")
        st.caption("Edit the active unified CSV in memory with previews and safe operations.")

        unified_csv_path = global_input_path or ""
        if unified_csv_path:
            st.caption(f"Using unified CSV: {unified_csv_path}")
        else:
            st.warning("No unified CSV selected in Global Input Configuration.")

        _init_edit_csv_state(unified_csv_path)
        df_working: pd.DataFrame | None = st.session_state.get("tools_edit_csv_df")

        load_col1, load_col2, load_col3 = st.columns([1, 1, 2])
        with load_col1:
            if st.button("Load Unified CSV", key="tools_edit_csv_load"):
                if unified_csv_path and os.path.exists(unified_csv_path):
                    with st.spinner("Reloading unified CSV from disk..."):
                        # Force reload from disk, bypassing cache
                        try:
                            df_working = _dataio_load(unified_csv_path)
                            st.session_state["tools_edit_csv_df"] = df_working
                            st.session_state["tools_edit_csv_undo"] = []
                            st.session_state["tools_edit_csv_redo"] = []
                            st.success(f"Reloaded {len(df_working):,} rows from disk.")
                        except Exception as e:
                            st.error(f"Error loading CSV: {e}")
                else:
                    st.error("Unified CSV path not found.")
        with load_col2:
            undo_stack = st.session_state.get("tools_edit_csv_undo", [])
            redo_stack = st.session_state.get("tools_edit_csv_redo", [])
            if st.button("Undo", key="tools_edit_csv_undo_btn"):
                if undo_stack:
                    last = undo_stack.pop()
                    current = st.session_state.get("tools_edit_csv_df")
                    if current is not None:
                        redo_stack.append({"label": last.get("label"), "df": current.copy(deep=True)})
                    st.session_state["tools_edit_csv_df"] = last.get("df")
                    st.session_state["tools_edit_csv_undo"] = undo_stack
                    st.session_state["tools_edit_csv_redo"] = redo_stack
                else:
                    st.info("Nothing to undo.")
        with load_col3:
            if st.button("Redo", key="tools_edit_csv_redo_btn"):
                if redo_stack:
                    last = redo_stack.pop()
                    current = st.session_state.get("tools_edit_csv_df")
                    if current is not None:
                        undo_stack.append({"label": last.get("label"), "df": current.copy(deep=True)})
                    st.session_state["tools_edit_csv_df"] = last.get("df")
                    st.session_state["tools_edit_csv_undo"] = undo_stack
                    st.session_state["tools_edit_csv_redo"] = redo_stack
                else:
                    st.info("Nothing to redo.")

        # Get target CSV metadata (offline, fast)
        target_metadata = None
        if unified_csv_path and os.path.exists(unified_csv_path):
            target_metadata = _get_csv_metadata_cached(unified_csv_path)

        df_working = st.session_state.get("tools_edit_csv_df")
        if df_working is None:
            if target_metadata:
                st.info(
                    f"📁 **Target CSV Metadata (Offline):**\n"
                    f"- Path: `{target_metadata['path']}`\n"
                    f"- Size: **{target_metadata['file_size_gb']:.2f} GB**\n"
                    f"- Rows: **{target_metadata['row_count']:,}**\n"
                    f"- Columns: **{len(target_metadata['columns'])}** columns"
                )
                st.info("💡 You can configure and run the **Import Columns from External CSV** panel (Panel F) below without loading the unified CSV into memory.")
            else:
                st.info("Load the unified CSV to begin editing.")

            with st.expander("A) Column Management", expanded=False):
                st.info("Load the unified CSV to manage columns in memory.")
            with st.expander("B) Value Inspection + Editing", expanded=False):
                st.info("Load the unified CSV to inspect/edit values in memory.")
            with st.expander("C) Derived / Conditional Column Builder", expanded=False):
                st.info("Load the unified CSV to build derived columns in memory.")
            with st.expander("D) Subset + Export Builder", expanded=False):
                st.info("Load the unified CSV to export subsets in memory.")
            with st.expander("E) Split + Group Export", expanded=False):
                st.info("Load the unified CSV to split columns in memory.")
        else:
            st.success(f"Working copy loaded: {len(df_working):,} rows, {len(df_working.columns):,} columns")

            with st.expander("A) Column Management", expanded=True):
                st.caption(
                    "Manage the structure of your CSV columns. "
                    "Pick an operation below to rename, delete, or duplicate columns."
                )
                available_columns = list(df_working.columns)

                col_op = st.radio(
                    "Operation",
                    ["Rename", "Delete Columns", "Duplicate Column"],
                    horizontal=True,
                    key="tools_edit_csv_col_op",
                    help=(
                        "**Rename** — Change the name of one or many columns.\n\n"
                        "**Delete Columns** — Remove one or more columns from the working data.\n\n"
                        "**Duplicate Column** — Copy an existing column into a new one."
                    ),
                )

                # ── Rename ──────────────────────────────────────────
                if col_op == "Rename":
                    rename_mode = st.radio(
                        "Rename Mode",
                        ["Single", "Batch"],
                        horizontal=True,
                        key="tools_edit_csv_rename_mode",
                        help=(
                            "**Single** — Pick one column and type its new name.\n\n"
                            "**Batch** — Find & replace a substring across multiple column names at once.\n\n"
                            "*Example (batch):*  Find `phenotype_` → Replace `` (empty) to strip the prefix from all matching columns."
                        ),
                    )
                    if rename_mode == "Single":
                        col_sel = st.selectbox(
                            "Column to Rename",
                            options=available_columns,
                            key="tools_edit_csv_single_rename_col",
                            help="Choose the column whose header you want to change.",
                        )
                        new_name = st.text_input(
                            "New Column Name",
                            key="tools_edit_csv_single_rename_name",
                            help="Type the desired new name. Must be non-empty and should be unique.",
                        )
                        confirm_overwrite = st.checkbox(
                            "Confirm overwrite if name already exists",
                            value=False,
                            key="tools_edit_csv_single_rename_confirm",
                        )
                        if st.button("Apply Rename", key="tools_edit_csv_single_rename_btn"):
                            if not new_name.strip():
                                st.error("New column name cannot be empty.")
                            elif new_name in available_columns and new_name != col_sel and not confirm_overwrite:
                                st.warning("Target name already exists. Tick the confirm checkbox to proceed.")
                            else:
                                before = df_working.copy()
                                df_new = df_working.rename(columns={col_sel: new_name})
                                _push_edit_csv_state(df_new, "Rename column", before, df_new)
                                st.success(f"Renamed **{col_sel}** → **{new_name}**")
                    else:
                        st.info(
                            "💡 **How Batch Rename works:** Every column whose name contains the *Find* "
                            "text will have that part replaced with the *Replace* text.\n\n"
                            "*Example:* Find `phenotype_` → Replace `pheno_` renames `phenotype_leiden` to `pheno_leiden`."
                        )
                        find_text = st.text_input(
                            "Find (in column names)",
                            key="tools_edit_csv_batch_find",
                            help="Substring (or regex) to match inside column names.",
                        )
                        replace_text = st.text_input(
                            "Replace with",
                            key="tools_edit_csv_batch_replace",
                            help="Text to substitute. Leave empty to remove the matched part.",
                        )
                        r1, r2 = st.columns(2)
                        with r1:
                            use_regex = st.checkbox("Use regex", key="tools_edit_csv_batch_regex")
                        with r2:
                            scope_cols = st.multiselect(
                                "Limit to these columns (optional)",
                                options=available_columns,
                                default=[],
                                key="tools_edit_csv_batch_scope",
                                help="Leave empty to apply to all columns.",
                            )
                        if st.button("Apply Batch Rename", key="tools_edit_csv_batch_btn"):
                            if not find_text:
                                st.error("Find text cannot be empty for batch rename.")
                            else:
                                targets = scope_cols if scope_cols else available_columns
                                rename_map = {}
                                for col in targets:
                                    if use_regex:
                                        new_col = re.sub(find_text, replace_text, col)
                                    else:
                                        new_col = col.replace(find_text, replace_text)
                                    rename_map[col] = new_col
                                new_names = list(rename_map.values())
                                if len(set(new_names)) != len(new_names):
                                    st.error("Batch rename would create duplicate column names. Adjust your find/replace pattern.")
                                else:
                                    changed = {k: v for k, v in rename_map.items() if k != v}
                                    before = df_working.copy()
                                    df_new = df_working.rename(columns=rename_map)
                                    _push_edit_csv_state(df_new, "Batch rename", before, df_new)
                                    st.success(f"Batch rename applied — {len(changed)} column(s) changed.")

                # ── Delete Columns ──────────────────────────────────
                elif col_op == "Delete Columns":
                    st.info(
                        "💡 Select one or more columns below, then click **Delete**. "
                        "This is reversible via the **Undo** button at the top."
                    )
                    del_cols = st.multiselect(
                        "Columns to Delete",
                        options=available_columns,
                        default=[],
                        key="tools_edit_csv_del_select",
                        help="Pick the column(s) you want to remove from the working data.",
                    )
                    delete_confirm = st.checkbox(
                        "I confirm I want to delete these columns",
                        key="tools_edit_csv_delete_confirm",
                    )
                    if st.button("Delete Selected Columns", key="tools_edit_csv_delete_btn"):
                        if not del_cols:
                            st.error("Select at least one column to delete.")
                        elif not delete_confirm:
                            st.warning("Tick the confirmation checkbox to proceed.")
                        else:
                            before = df_working.copy()
                            df_new = df_working.drop(columns=del_cols)
                            _push_edit_csv_state(df_new, "Delete columns", before, df_new)
                            st.success(f"Deleted {len(del_cols)} column(s).")

                # ── Duplicate Column ────────────────────────────────
                else:
                    st.info(
                        "💡 Creates an exact copy of a column under a new name. "
                        "Useful when you want to edit values without losing the original.\n\n"
                        "*Example:* Duplicate `phenotype_leiden` → `phenotype_leiden_backup`"
                    )
                    dup_source = st.selectbox(
                        "Source Column",
                        options=available_columns,
                        key="tools_edit_csv_dup_source",
                        help="The existing column to copy.",
                    )
                    dup_name = st.text_input(
                        "New Column Name",
                        key="tools_edit_csv_dup_name",
                        help="Name for the new duplicated column. Must be non-empty.",
                    )
                    dup_confirm = st.checkbox(
                        "Confirm overwrite if name already exists",
                        key="tools_edit_csv_dup_confirm",
                    )
                    if st.button("Duplicate Column", key="tools_edit_csv_dup_btn"):
                        if not dup_name.strip():
                            st.error("New column name cannot be empty.")
                        elif dup_name in available_columns and not dup_confirm:
                            st.warning("Target name already exists. Tick the confirm checkbox to proceed.")
                        else:
                            before = df_working.copy()
                            df_new = df_working.copy()
                            df_new[dup_name] = df_new[dup_source]
                            _push_edit_csv_state(df_new, "Duplicate column", before, df_new)
                            st.success(f"Duplicated **{dup_source}** → **{dup_name}**")

            with st.expander("B) Value Inspection + Editing", expanded=True):
                st.caption("Inspect values and apply mappings or cleanup rules.")
                col_target = st.selectbox(
                    "Target Column",
                    options=list(df_working.columns),
                    key="tools_edit_csv_value_col",
                )
                series = df_working[col_target]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                if is_numeric:
                    st.markdown("#### Numeric Summary")
                    summary = {
                        "min": _coerce_numeric(series).min(),
                        "max": _coerce_numeric(series).max(),
                        "median": _coerce_numeric(series).median(),
                    }
                    st.write(summary)
                    bins = pd.cut(_coerce_numeric(series), bins=10)
                    bucket = bins.value_counts().sort_index().reset_index()
                    bucket.columns = ["bucket", "count"]
                    st.dataframe(bucket, width="stretch", hide_index=True)
                else:
                    st.markdown("#### Unique Values")
                    search_val = st.text_input("Search values", key="tools_edit_csv_value_search")
                    value_counts = series.astype(str).fillna("NaN").value_counts()
                    if search_val:
                        value_counts = value_counts[value_counts.index.str.contains(search_val, case=False, na=False)]
                    max_show = 200
                    if len(value_counts) > max_show:
                        st.caption(f"Showing first {max_show} values. Refine search to see more.")
                        value_counts = value_counts.head(max_show)
                    value_df = value_counts.reset_index()
                    value_df.columns = ["value", "count"]
                    st.dataframe(value_df, width="stretch", hide_index=True)

                st.markdown("#### Rename Specific Values")
                mapping_key = f"tools_edit_csv_mapping_{col_target}"
                if mapping_key not in st.session_state:
                    unique_vals = series.astype(str).fillna("NaN").unique().tolist()
                    st.session_state[mapping_key] = pd.DataFrame(
                        {"From": unique_vals[:200], "To": [""] * min(len(unique_vals), 200)}
                    )
                mapping_df = st.session_state[mapping_key]

                # Wrap editor in a form to prevent reload on every keystroke (offline mode)
                with st.form(key=f"tools_edit_csv_mapping_form_{col_target}"):
                    edited_mapping = st.data_editor(
                        mapping_df,
                        num_rows="dynamic",
                        hide_index=True,
                        key=f"tools_edit_csv_mapping_editor_{col_target}",
                    )
                    
                    st.divider()
                    
                    apply_selected_only = st.checkbox(
                        "Apply to selected values only",
                        value=False,
                        key="tools_edit_csv_map_selected_only",
                    )
                    selected_values = st.multiselect(
                        "Selected values (for filtering)",
                        options=edited_mapping["From"].dropna().astype(str).unique().tolist(),
                        default=[],
                        key="tools_edit_csv_map_selected",
                    )
                    map_output_mode = st.radio(
                        "Mapping Output Mode",
                        ["Overwrite column", "Save into new column"],
                        horizontal=True,
                        key="tools_edit_csv_map_output_mode",
                    )
                    map_new_col = st.text_input("New column name (if saving as new)", key="tools_edit_csv_map_new_col")
                    map_confirm = st.checkbox(
                        "Confirm overwrite existing column",
                        value=False,
                        key="tools_edit_csv_map_confirm",
                    )
                    
                    # Submit button for the form
                    apply_mapping = st.form_submit_button("Apply Mapping")

                if apply_mapping:
                    st.session_state[mapping_key] = edited_mapping # Update state with final form values
                    mapping = {
                        str(row["From"]): row["To"]
                        for _, row in edited_mapping.iterrows()
                        if str(row.get("From", "")).strip() and str(row.get("To", "")).strip()
                    }
                    if not mapping:
                        st.warning("No mappings provided.")
                    else:
                        before = df_working.copy()
                        df_new = df_working.copy()
                        target_col = col_target
                        if map_output_mode == "Save into new column":
                            if not map_new_col.strip():
                                st.error("Provide a new column name.")
                                mapping = None
                            elif map_new_col in df_new.columns and not map_confirm:
                                st.warning("Target column exists. Confirm overwrite to continue.")
                                mapping = None
                            else:
                                target_col = map_new_col
                                df_new[target_col] = df_new[col_target]
                        else:
                            if not map_confirm:
                                st.warning("Confirm overwrite to apply mapping.")
                                mapping = None

                        if mapping:
                            series_text = df_new[target_col].astype(str)
                            if apply_selected_only and selected_values:
                                mask = series_text.isin([str(v) for v in selected_values])
                                df_new.loc[mask, target_col] = series_text[mask].map(mapping).fillna(series_text[mask])
                            else:
                                df_new[target_col] = series_text.map(mapping).fillna(series_text)
                            _push_edit_csv_state(df_new, "Value mapping", before, df_new)
                            st.success("Mapping applied.")

                st.markdown("#### String Cleanup")
                trim_whitespace = st.checkbox("Trim whitespace", value=True, key="tools_edit_csv_trim_ws")
                strip_prefix = st.text_input("Strip prefix", key="tools_edit_csv_strip_prefix")
                strip_suffix = st.text_input("Strip suffix", key="tools_edit_csv_strip_suffix")
                replace_from = st.text_input("Replace", key="tools_edit_csv_replace_from")
                replace_to = st.text_input("Replace with", key="tools_edit_csv_replace_to")
                replace_regex = st.checkbox("Replace using regex", key="tools_edit_csv_replace_regex")
                cleanup_output_mode = st.radio(
                    "Cleanup Output Mode",
                    ["Overwrite column", "Save into new column"],
                    horizontal=True,
                    key="tools_edit_csv_cleanup_output_mode",
                )
                cleanup_new_col = st.text_input("New column name", key="tools_edit_csv_cleanup_new_col")
                cleanup_confirm = st.checkbox("Confirm overwrite", value=False, key="tools_edit_csv_cleanup_confirm")
                if st.button("Apply Cleanup", key="tools_edit_csv_apply_cleanup"):
                    before = df_working.copy()
                    df_new = df_working.copy()
                    target_col = col_target
                    if cleanup_output_mode == "Save into new column":
                        if not cleanup_new_col.strip():
                            st.error("Provide a new column name.")
                            df_new = None
                        elif cleanup_new_col in df_new.columns and not cleanup_confirm:
                            st.warning("Target column exists. Confirm overwrite to continue.")
                            df_new = None
                        else:
                            target_col = cleanup_new_col
                            df_new[target_col] = df_new[col_target]
                    else:
                        if not cleanup_confirm:
                            st.warning("Confirm overwrite to apply cleanup.")
                            df_new = None
                    if df_new is not None:
                        df_new[target_col] = _apply_value_cleanup(
                            df_new[target_col],
                            trim_whitespace=trim_whitespace,
                            strip_prefix=strip_prefix,
                            strip_suffix=strip_suffix,
                            replace_from=replace_from,
                            replace_to=replace_to,
                            replace_regex=replace_regex,
                        )
                        _push_edit_csv_state(df_new, "String cleanup", before, df_new)
                        st.success("Cleanup applied.")

            with st.expander("C) Derived / Conditional Column Builder", expanded=True):
                st.caption(
                    "Create a new column using rule-based logic. "
                    "Start from a copy of an existing column or from scratch with a default value, "
                    "then apply conditional rules to override values row by row."
                )

                # -- Handle "Load Example Rule" BEFORE widgets to avoid session_state conflict --
                _load_example_key = "tools_edit_csv_load_example_pending"
                if st.session_state.get(_load_example_key):
                    st.session_state[_load_example_key] = False
                    # Values were set in session_state before rerun — widgets will pick them up

                if st.button(
                    "Load Example Rule",
                    key="tools_edit_csv_example_rule",
                    help=(
                        "Loads a ready-made example: create `phenotype_ml_consensus_refined` "
                        "by copying `phenotype_ml_consensus` then setting all rows where "
                        "`ml_consensus_score < 0.5` to **Others**."
                    ),
                ):
                    example_rows = []
                    if "ml_consensus_score" in df_working.columns:
                        example_rows.append(
                            {
                                "Rule": "1",
                                "Column": "ml_consensus_score",
                                "Operator": "<",
                                "Value": "0.5",
                                "Value2": "",
                                "Join": "AND",
                                "Set Value": "Others",
                            }
                        )
                    else:
                        st.warning(
                            "Column `ml_consensus_score` not found in the data. "
                            "A blank rule table was loaded — fill in your own columns."
                        )
                    st.session_state["tools_edit_csv_rules"] = pd.DataFrame(example_rows)
                    st.session_state["tools_edit_csv_derived_output"] = "phenotype_ml_consensus_refined"
                    if "phenotype_ml_consensus" in df_working.columns:
                        st.session_state["tools_edit_csv_derived_base_mode"] = "Copy from column"
                        st.session_state["tools_edit_csv_derived_base_col"] = "phenotype_ml_consensus"
                    st.session_state[_load_example_key] = True
                    st.rerun()

                output_col = st.text_input(
                    "New Column Name",
                    value=st.session_state.get("tools_edit_csv_derived_output", ""),
                    key="tools_edit_csv_derived_output",
                    help="Name for the derived column that will be created or overwritten.",
                )
                base_mode = st.radio(
                    "Base Behavior",
                    ["Copy from column", "Create from scratch"],
                    horizontal=True,
                    key="tools_edit_csv_derived_base_mode",
                    help=(
                        "**Copy from column** — Start with all values from an existing column, then override selected rows via rules.\n\n"
                        "**Create from scratch** — All rows start with a default value, then rules set specific rows to different values."
                    ),
                )
                if base_mode == "Copy from column":
                    base_col = st.selectbox(
                        "Base Column",
                        options=list(df_working.columns),
                        key="tools_edit_csv_derived_base_col",
                        help="The column to copy as the starting point.",
                    )
                    default_value = ""
                else:
                    base_col = None
                    default_value = st.text_input(
                        "Default Value",
                        value=st.session_state.get("tools_edit_csv_derived_default", ""),
                        key="tools_edit_csv_derived_default",
                        help="Every row starts with this value before rules are applied.",
                    )

                if st.session_state.get("tools_edit_csv_rules") is None:
                    st.session_state["tools_edit_csv_rules"] = pd.DataFrame(
                        columns=["Rule", "Column", "Operator", "Value", "Value2", "Join", "Set Value"]
                    )

                rules_df = st.session_state["tools_edit_csv_rules"]
                rules_df = st.data_editor(
                    rules_df,
                    num_rows="dynamic",
                    hide_index=True,
                    key="tools_edit_csv_rules_editor",
                    column_config={
                        "Operator": st.column_config.SelectboxColumn(
                            options=[
                                "<",
                                "<=",
                                ">",
                                ">=",
                                "between",
                                "equals",
                                "in list",
                                "contains",
                                "startswith",
                                "endswith",
                                "is missing",
                            ]
                        ),
                        "Join": st.column_config.SelectboxColumn(options=["AND", "OR"]),
                    },
                )
                st.session_state["tools_edit_csv_rules"] = rules_df
                derived_confirm = st.checkbox("Confirm overwrite if column exists", key="tools_edit_csv_derived_confirm")
                if st.button("Build Derived Column", key="tools_edit_csv_build_derived"):
                    if not output_col.strip():
                        st.error("Provide a new column name.")
                    elif output_col in df_working.columns and not derived_confirm:
                        st.warning("Target column exists. Confirm overwrite to continue.")
                    else:
                        before = df_working.copy()
                        df_new = df_working.copy()
                        if base_mode == "Copy from column" and base_col:
                            derived = df_new[base_col].copy()
                        else:
                            derived = pd.Series([default_value] * len(df_new), index=df_new.index)

                        rule_rows = rules_df.dropna(subset=["Column", "Operator"], how="any")
                        rule_ids = []
                        for rule_id in rule_rows["Rule"].astype(str).tolist():
                            if rule_id not in rule_ids:
                                rule_ids.append(rule_id)
                        applied_mask = pd.Series([False] * len(df_new), index=df_new.index)
                        for rule_id in rule_ids:
                            group = rule_rows[rule_rows["Rule"].astype(str) == rule_id]
                            if group.empty:
                                continue
                            rule_value = group["Set Value"].dropna().astype(str).tolist()
                            rule_value = rule_value[0] if rule_value else ""
                            combined = None
                            for idx, row in group.iterrows():
                                col_name = row.get("Column")
                                operator = row.get("Operator")
                                value = str(row.get("Value", ""))
                                value_2 = str(row.get("Value2", ""))
                                join = str(row.get("Join", "AND")).upper()
                                if col_name not in df_new.columns:
                                    continue
                                cond = _build_rule_mask(df_new, col_name, operator, value, value_2)
                                if combined is None:
                                    combined = cond
                                else:
                                    combined = combined | cond if join == "OR" else combined & cond
                            if combined is None:
                                continue
                            apply_mask = combined & (~applied_mask)
                            derived.loc[apply_mask] = rule_value
                            applied_mask = applied_mask | apply_mask

                        df_new[output_col] = derived
                        _push_edit_csv_state(df_new, "Derived column", before, df_new)
                        st.success("Derived column created.")

            with st.expander("D) Subset + Export Builder", expanded=True):
                st.caption("Export filtered rows/columns or apply filters to the working data.")
                col_search = st.text_input("Search columns", key="tools_edit_csv_export_search")
                export_cols = st.session_state.get("tools_edit_csv_export_cols")
                all_cols = list(df_working.columns)
                if export_cols is None:
                    export_cols = all_cols.copy()
                    st.session_state["tools_edit_csv_export_cols"] = export_cols
                filtered_cols = [c for c in all_cols if col_search.lower() in c.lower()]
                select_all, clear_all, _ = st.columns([1, 1, 3])
                with select_all:
                    if st.button("Select All", key="tools_edit_csv_export_select_all"):
                        st.session_state["tools_edit_csv_export_cols"] = all_cols.copy()
                with clear_all:
                    if st.button("Clear", key="tools_edit_csv_export_clear"):
                        st.session_state["tools_edit_csv_export_cols"] = []
                # Ensure options include currently selected items to avoid "value not in options" error during search
                final_opts = [c for c in all_cols if c in filtered_cols or c in export_cols]
                export_cols = st.multiselect(
                    "Columns to Export",
                    options=final_opts,
                    key="tools_edit_csv_export_cols",
                )

                st.markdown("#### Row Filters")
                filters = st.session_state.get("tools_edit_csv_filters", [])
                if st.button("Add Filter", key="tools_edit_csv_add_filter"):
                    filters.append({"column": "", "op": "", "value": "", "value2": "", "exclude": False, "keep_missing": True})
                updated_filters = []
                for idx, filt in enumerate(filters):
                    with st.container():
                        c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
                        with c1:
                            filt_col = st.selectbox(
                                "Column",
                                options=[""] + all_cols,
                                index=(all_cols.index(filt.get("column")) + 1) if filt.get("column") in all_cols else 0,
                                key=f"tools_edit_csv_filter_col_{idx}",
                            )
                        col_series = df_working[filt_col] if filt_col in df_working.columns else None
                        if col_series is not None and pd.api.types.is_numeric_dtype(col_series):
                            ops = ["between", ">=", "<=", ">", "<", "is missing"]
                        else:
                            ops = ["equals", "in list", "contains", "startswith", "endswith", "is missing"]
                        with c2:
                            filt_op = st.selectbox(
                                "Operator",
                                options=[""] + ops,
                                index=ops.index(filt.get("op")) + 1 if filt.get("op") in ops else 0,
                                key=f"tools_edit_csv_filter_op_{idx}",
                            )
                        with c3:
                            if filt_op == "between":
                                min_val = st.text_input("Min", value=filt.get("value", ""), key=f"tools_edit_csv_filter_min_{idx}")
                                max_val = st.text_input("Max", value=filt.get("value2", ""), key=f"tools_edit_csv_filter_max_{idx}")
                                filt_value = min_val
                                filt_value2 = max_val
                            else:
                                filt_value = st.text_input("Value", value=filt.get("value", ""), key=f"tools_edit_csv_filter_val_{idx}")
                                filt_value2 = ""
                        with c4:
                            remove = st.button("Remove", key=f"tools_edit_csv_filter_remove_{idx}")
                        if remove:
                            continue
                        exclude = st.checkbox("Exclude", value=bool(filt.get("exclude")), key=f"tools_edit_csv_filter_ex_{idx}")
                        keep_missing = st.checkbox(
                            "Keep missing",
                            value=bool(filt.get("keep_missing", True)),
                            key=f"tools_edit_csv_filter_keep_{idx}",
                        )
                        updated_filters.append(
                            {
                                "column": filt_col,
                                "op": filt_op,
                                "value": filt_value,
                                "value2": filt_value2,
                                "exclude": exclude,
                                "keep_missing": keep_missing,
                            }
                        )
                st.session_state["tools_edit_csv_filters"] = updated_filters

                def _apply_filters(df: pd.DataFrame, filters_list: list[dict[str, Any]]) -> pd.DataFrame:
                    if not filters_list:
                        return df
                    mask = pd.Series([True] * len(df), index=df.index)
                    for filt in filters_list:
                        col = filt.get("column")
                        op = filt.get("op")
                        if not col or col not in df.columns or not op:
                            continue
                        series = df[col]
                        cond = pd.Series([True] * len(df), index=df.index)
                        if op == "is missing":
                            cond = series.isna()
                        elif pd.api.types.is_numeric_dtype(series):
                            numeric = _coerce_numeric(series)
                            try:
                                if op == "between":
                                    cond = numeric.between(float(filt.get("value", 0)), float(filt.get("value2", 0)))
                                elif op == ">=":
                                    cond = numeric >= float(filt.get("value", 0))
                                elif op == "<=":
                                    cond = numeric <= float(filt.get("value", 0))
                                elif op == ">":
                                    cond = numeric > float(filt.get("value", 0))
                                elif op == "<":
                                    cond = numeric < float(filt.get("value", 0))
                            except Exception:
                                cond = pd.Series([False] * len(df), index=df.index)
                        else:
                            text = series.astype(str)
                            val = str(filt.get("value", ""))
                            if op == "equals":
                                cond = text == val
                            elif op == "in list":
                                values = [v.strip() for v in val.split(",") if v.strip()]
                                cond = text.isin(values)
                            elif op == "contains":
                                cond = text.str.contains(val, case=False, na=False)
                            elif op == "startswith":
                                cond = text.str.startswith(val, na=False)
                            elif op == "endswith":
                                cond = text.str.endswith(val, na=False)
                        if filt.get("exclude"):
                            cond = ~cond
                        if not filt.get("keep_missing", True):
                            cond = cond & (~series.isna())
                        mask = mask & cond
                    return df.loc[mask]

                filtered_df = _apply_filters(df_working, updated_filters)
                st.caption(f"Filtered rows: {len(filtered_df):,}")
                export_path = st.text_input(
                    "Export File Path",
                    value=str(project_root / "outputs/phenotyping/edit_csv/filtered.csv"),
                    key="tools_edit_csv_export_path",
                )
                apply_filters_confirm = st.checkbox(
                    "Apply filters to working data",
                    value=False,
                    key="tools_edit_csv_apply_filters_confirm",
                )
                if st.button("Export Filtered CSV", key="tools_edit_csv_export_btn"):
                    if not export_cols:
                        st.error("Select at least one column to export.")
                    elif not export_path:
                        st.error("Provide an export file path.")
                    else:
                        run_audit = begin_phenotyping_run_audit(
                            Path(project_root),
                            "tools",
                            input_csv=unified_csv_path,
                            event_type="filtered_csv_export",
                        )
                        os.makedirs(os.path.dirname(export_path), exist_ok=True)
                        export_df = filtered_df[export_cols]
                        export_df.to_csv(export_path, index=False)
                        st.success(f"Exported {len(export_df):,} rows to {export_path}")
                        if apply_filters_confirm:
                            before = df_working.copy()
                            df_new = filtered_df.copy()
                            _push_edit_csv_state(df_new, "Apply filters", before, df_new)
                        finish_phenotyping_run_audit(
                            run_audit,
                            status="success",
                            output_path=export_path,
                            input_rows=len(df_working),
                            output_rows=len(export_df),
                            extra={"export_columns": export_cols, "filters": updated_filters},
                        )

            with st.expander("E) Split + Group Export", expanded=False):
                st.caption("Split the working data into separate CSV files by a grouping column.")
                split_all_cols = list(df_working.columns)
                default_split = "batch_id" if "batch_id" in split_all_cols else (split_all_cols[0] if split_all_cols else "")
                split_col = st.selectbox(
                    "Grouping Column",
                    options=split_all_cols if split_all_cols else [""],
                    index=split_all_cols.index(default_split) if default_split in split_all_cols else 0,
                    help="Column used to split the data into per-group CSV files.",
                    key="tools_edit_csv_split_col",
                )
                default_split_dir = cached_params.get(
                    "tools_split_csv_output_dir",
                    str(project_root / "outputs/phenotyping/split_csv"),
                )
                split_output_dir = st.text_input(
                    "Output Directory",
                    value=default_split_dir,
                    key="tools_edit_csv_split_output_dir",
                    help="Folder where per-group CSV files will be written.",
                )
                if split_output_dir != cached_params.get("tools_split_csv_output_dir"):
                    cached_params["tools_split_csv_output_dir"] = split_output_dir
                    try:
                        with open(tools_cache, "w") as f:
                            json.dump(cached_params, f, indent=2)
                    except Exception:
                        pass

                st.markdown("##### Group Summary")
                if split_col and split_col in df_working.columns:
                    split_counts = (
                        df_working[split_col]
                        .astype(str)
                        .fillna("NaN")
                        .value_counts(dropna=False)
                        .rename_axis(split_col)
                        .reset_index(name="n_rows")
                    )
                    st.dataframe(split_counts, width="stretch", hide_index=True)
                    st.caption(f"Total rows: {len(df_working):,} | Unique groups: {len(split_counts):,}")
                else:
                    st.info("Select a valid grouping column to preview summary.")

                if st.button("Split into CSV Files", type="primary", key="tools_edit_csv_split_run"):
                    if split_col not in df_working.columns:
                        st.error("Selected grouping column is not available.")
                    elif not split_output_dir:
                        st.error("Please provide an output directory.")
                    else:
                        run_audit = begin_phenotyping_run_audit(
                            Path(project_root),
                            "tools",
                            input_csv=unified_csv_path,
                            event_type="split_csv_export",
                        )
                        os.makedirs(split_output_dir, exist_ok=True)
                        written = 0
                        for group_value, group_df in df_working.groupby(split_col, dropna=False):
                            label = "NaN" if pd.isna(group_value) else str(group_value)
                            safe_label = _sanitize_filename_token(label)
                            out_path = os.path.join(split_output_dir, f"{safe_label}.csv")
                            group_df.to_csv(out_path, index=False)
                            written += 1
                        st.success(f"Split complete. Wrote {written} CSV file(s) to `{split_output_dir}`.")
                        finish_phenotyping_run_audit(
                            run_audit,
                            status="success",
                            output_path=split_output_dir,
                            input_rows=len(df_working),
                            output_rows=len(df_working),
                            extra={"grouping_column": split_col, "files_written": written},
                        )

        # F) Import Columns from External CSV (always accessible, offline metadata supported)
        with st.expander("F) Import Columns from External CSV", expanded=True if df_working is None else False):
            st.caption(
                "Import columns from an external source CSV into the current working target CSV "
                "using anchoring column mapping (with optional shift tolerance for coordinates)."
            )

            # Form 1: Enter path and load header columns (reads only header, runs instantly)
            with st.form(key="tools_edit_csv_import_source_path_form"):
                source_path = st.text_input(
                    "Source CSV File Path",
                    value=st.session_state.get("tools_edit_csv_import_source_path_val", ""),
                    help="Path to the source CSV file."
                )
                load_btn = st.form_submit_button("Load Source CSV Header Columns", type="primary")

            if load_btn and source_path:
                st.session_state["tools_edit_csv_import_source_path_val"] = source_path
                _get_columns_cached.clear()

            source_df_cols = []
            source_path_val = st.session_state.get("tools_edit_csv_import_source_path_val", "")
            if source_path_val:
                if os.path.exists(source_path_val):
                    source_df_cols = _get_columns_cached(source_path_val)
                    if source_df_cols:
                        st.success(f"Source CSV headers loaded successfully: {len(source_df_cols)} columns found.")
                    else:
                        st.error("Failed to read columns from source CSV. Make sure it is a valid CSV.")
                else:
                    st.warning("Source CSV path does not exist.")

            if source_df_cols:
                if df_working is not None:
                    target_cols = list(df_working.columns)
                else:
                    target_cols = target_metadata["columns"] if target_metadata else []

                # Smart default anchors detection
                common_anchors = ["cell_id", "batch_id", "x_centroid", "y_centroid", "tissue", "label"]
                default_src_anchors = [col for col in common_anchors if col in source_df_cols and col in target_cols]
                default_tgt_anchors = [col for col in common_anchors if col in target_cols and col in source_df_cols]

                st.markdown("---")
                st.markdown("### 2. Configure Mapping & Merge Options (Offline)")
                st.caption("All configurations below are offline and will not reload the page until you click Scan or Run.")

                # Form 2: Main Configuration Form (handles everything offline, no page reruns!)
                with st.form(key="tools_edit_csv_import_merge_config_form"):
                    st.markdown("#### A. Anchor Column Selection")
                    st.caption("Select keys/coordinates columns from both CSV files in matching order.")
                    
                    col_anch1, col_anch2 = st.columns(2)
                    with col_anch1:
                        src_anchors = st.multiselect(
                            "Source Anchor Columns",
                            options=source_df_cols,
                            default=default_src_anchors,
                            help="Select the matching keys/coordinates in the source CSV."
                        )
                    with col_anch2:
                        tgt_anchors = st.multiselect(
                            "Target Anchor Columns (must match the order of Source Anchors)",
                            options=target_cols,
                            default=default_tgt_anchors,
                            help="Select the matching keys/coordinates in the target CSV."
                        )

                    st.markdown("#### B. Spatial / Coordinate Shift Tolerance (Optional)")
                    st.caption("Apply coordinate shift tolerances to coordinates (e.g. x_centroid, y_centroid).")
                    
                    tolerant_cols = st.multiselect(
                        "Select tolerant source anchor columns",
                        options=src_anchors if src_anchors else source_df_cols,
                        default=[col for col in ["x_centroid", "y_centroid"] if col in src_anchors],
                        help="Source anchors that require shift tolerance."
                    )

                    tolerance_input = st.text_input(
                        "Allowed Shift Buffer (e.g., '1.0%' for percentage of range, or '5.0' for absolute distance. Comma-separated in order of above columns)",
                        value="1.0%",
                        help="Examples: '1.0%' or '5.0'. If only one value is given, it applies to all tolerant columns."
                    )

                    st.markdown("#### C. Columns to Transfer")
                    transfer_cols = st.multiselect(
                        "Columns to Transfer from Source CSV",
                        options=[c for c in source_df_cols if c not in src_anchors],
                        help="Select precisely which columns you want to copy/transfer from the source CSV."
                    )

                    rename_mapping_raw = st.text_area(
                        "Rename Transferred Columns (Optional, format: source_col:new_name, one per line)",
                        value="",
                        placeholder="Example:\nphenotype_ref1:phenotype_ref1_new\nphenotype_ref2:phenotype_ref2_new",
                        help="Leave blank to keep original names."
                    )

                    collision_op = st.radio(
                        "Collision Strategy for existing target columns",
                        ["Replace (Overwrite entire column values for matched rows)", "Fill missing cells only (only update target cells if empty/NaN)"],
                        key="tools_edit_csv_import_collision_strategy_radio"
                    )

                    st.markdown("#### D. Output Action & Saving")
                    save_option = st.radio(
                        "Output Action",
                        [
                            "Apply to current working CSV in memory (default, allows preview/undo)",
                            "Overwrite target CSV file directly",
                            "Save as new CSV file"
                        ],
                        key="tools_edit_csv_import_save_option_radio",
                    )

                    custom_new_path = st.text_input(
                        "New CSV File Path (if saving as new)",
                        value=str(project_root / "outputs/phenotyping/edit_csv/imported_columns.csv"),
                        key="tools_edit_csv_import_new_path_str",
                    )

                    run_confirm = st.checkbox("Confirm and run import/merge", key="tools_edit_csv_import_run_confirm_box")

                    # Submit buttons inside the form
                    col_sub1, col_sub2 = st.columns(2)
                    with col_sub1:
                        scan_clicked = st.form_submit_button("Scan Overlap", type="secondary")
                    with col_sub2:
                        run_clicked = st.form_submit_button("Run Import / Merge", type="primary")

                # Handle form submissions (performs matching and computation only on click)
                if scan_clicked or run_clicked:
                    if len(src_anchors) != len(tgt_anchors):
                        st.error("Number of Source Anchors must match the number of Target Anchors. Please review selection.")
                    elif len(src_anchors) == 0:
                        st.error("Please configure at least one anchor column pair.")
                    elif len(transfer_cols) == 0:
                        st.error("Please select at least one column to transfer.")
                    else:
                        # 1. Parse rename mapping
                        rename_map = {}
                        for line in rename_mapping_raw.split("\n"):
                            line = line.strip()
                            if not line or ":" not in line:
                                continue
                            parts = line.split(":", 1)
                            src = parts[0].strip()
                            dst = parts[1].strip()
                            if src and dst:
                                rename_map[src] = dst

                        # 2. Parse tolerances
                        tols_raw = [t.strip() for t in tolerance_input.split(",") if t.strip()]
                        if len(tols_raw) == 1:
                            tols_raw = tols_raw * len(tolerant_cols)

                        tolerance_config = {}
                        for col, tol_str in zip(tolerant_cols, tols_raw, strict=False):
                            is_pct = "%" in tol_str
                            val_str = tol_str.replace("%", "").strip()
                            try:
                                val = float(val_str)
                                tolerance_config[col] = {"value": val, "is_pct": is_pct}
                            except Exception:
                                tolerance_config[col] = {"value": 1.0, "is_pct": True}

                        # 3. Construct anchor configurations (calculating ranges on single columns: fast)
                        anchors_config = []
                        for s_col, t_col in zip(src_anchors, tgt_anchors, strict=True):
                            is_tolerant = s_col in tolerant_cols
                            tol_info = tolerance_config.get(s_col, {"value": 0.0, "is_pct": False})

                            tolerance_val = 0.0
                            if is_tolerant:
                                if tol_info["is_pct"]:
                                    # Use in-memory target if loaded, else read single target column from disk
                                    if df_working is not None:
                                        tgt_series = df_working[t_col]
                                    else:
                                        tgt_series = pd.read_csv(unified_csv_path, usecols=[t_col])[t_col]
                                    tgt_series = pd.to_numeric(tgt_series, errors="coerce").dropna()
                                    if not tgt_series.empty:
                                        span = float(tgt_series.max() - tgt_series.min())
                                        tolerance_val = (tol_info["value"] / 100.0) * span
                                    else:
                                        is_tolerant = False
                                else:
                                    tolerance_val = tol_info["value"]

                            anchors_config.append({
                                "source_col": s_col,
                                "target_col": t_col,
                                "is_tolerant": is_tolerant,
                                "tolerance_val": tolerance_val,
                            })

                        if scan_clicked:
                            with st.spinner("Scanning rows & computing matches..."):
                                try:
                                    # Load ONLY the anchor columns to keep memory and loading time negligible
                                    if df_working is not None:
                                        target_anchor_df = df_working[tgt_anchors]
                                    else:
                                        target_anchor_df = pd.read_csv(unified_csv_path, usecols=tgt_anchors)
                                    source_anchor_df = pd.read_csv(source_path_val, usecols=src_anchors)

                                    matched_idx = _find_matching_indices(target_anchor_df, source_anchor_df, anchors_config)
                                    matched_count = int(matched_idx.notna().sum())
                                    total_target = len(target_anchor_df)
                                    match_pct = (matched_count / total_target) * 100.0 if total_target > 0 else 0.0

                                    st.markdown("#### Scan Overlap Summary")
                                    st.info(
                                        f"- Total Target Rows: **{total_target:,}**\n"
                                        f"- Total Source Rows: **{len(source_anchor_df):,}**\n"
                                        f"- Matched Rows: **{matched_count:,} ({match_pct:.2f}%)**"
                                    )

                                    # Fast check for collisions
                                    overlap_details = []
                                    for col in transfer_cols:
                                        dest = rename_map.get(col, col)
                                        if dest in target_cols:
                                            if df_working is not None:
                                                target_vals = df_working[dest]
                                            else:
                                                target_vals = pd.read_csv(unified_csv_path, usecols=[dest])[dest]
                                            target_matched_vals = target_vals.loc[matched_idx.notna()]
                                            not_null_mask = target_matched_vals.notna()
                                            if pd.api.types.is_string_dtype(target_matched_vals):
                                                not_null_mask = not_null_mask & (target_matched_vals != "") & (target_matched_vals != "nan")
                                            non_empty_count = int(not_null_mask.sum())
                                            
                                            strategy = "fill" if "Fill missing" in collision_op else "replace"
                                            overlap_details.append({
                                                "Column": dest,
                                                "Existing Values in Matched Rows": f"{non_empty_count:,}",
                                                "Strategy": "Overwrite" if strategy == "replace" else "Fill missing"
                                            })
                                    if overlap_details:
                                        st.markdown("**Colliding columns details:**")
                                        st.dataframe(pd.DataFrame(overlap_details), width="stretch", hide_index=True)
                                except Exception as e:
                                    st.error(f"Error during scan: {e}")

                        elif run_clicked:
                            if not run_confirm:
                                st.warning("Please check the 'Confirm and run import/merge' box to execute.")
                            elif save_option == "Save as new CSV file" and not custom_new_path.strip():
                                st.error("Please specify a valid path to save the new CSV file.")
                            else:
                                with st.spinner("Performing merge..."):
                                    try:
                                        # 1. Load ONLY anchor columns and match
                                        if df_working is not None:
                                            target_anchor_df = df_working[tgt_anchors]
                                        else:
                                            target_anchor_df = pd.read_csv(unified_csv_path, usecols=tgt_anchors)
                                        source_anchor_df = pd.read_csv(source_path_val, usecols=src_anchors)
                                        matched_idx = _find_matching_indices(target_anchor_df, source_anchor_df, anchors_config)
                                        matched_nonnull = matched_idx.dropna()
                                        matched_count = len(matched_nonnull)

                                        # 2. Load ONLY transfer columns
                                        source_transfer_df = pd.read_csv(source_path_val, usecols=transfer_cols)

                                        # 3. Apply changes to memory df_working or load full df for disk saving
                                        if save_option == "Apply to current working CSV in memory (default, allows preview/undo)":
                                            if df_working is None:
                                                df_new = _dataio_load(unified_csv_path)
                                            else:
                                                df_new = df_working.copy()
                                        else:
                                            df_new = _dataio_load(unified_csv_path)

                                        for col in transfer_cols:
                                            dest = rename_map.get(col, col)
                                            strategy = "fill" if "Fill missing" in collision_op else "replace"

                                            # Fast numpy array mapping
                                            source_vals = pd.Series(
                                                source_transfer_df[col].values[matched_nonnull.values.astype(int)],
                                                index=matched_nonnull.index
                                            )

                                            if dest in df_new.columns:
                                                if strategy == "replace":
                                                    df_new.loc[source_vals.index, dest] = source_vals
                                                else:
                                                    current_vals = df_new[dest]
                                                    is_null = current_vals.isna() | (current_vals.astype(str).str.strip() == "")
                                                    fill_idx = source_vals.index.intersection(df_new.index[is_null])
                                                    df_new.loc[fill_idx, dest] = source_vals.loc[fill_idx]
                                            else:
                                                df_new[dest] = pd.NA
                                                df_new.loc[source_vals.index, dest] = source_vals

                                        report_msg = (
                                            f"Successfully imported **{len(transfer_cols)}** column(s) "
                                            f"(`{', '.join(transfer_cols)}`) "
                                            f"into the target CSV.\n\n"
                                            f"- **{matched_count:,}** rows successfully matched and updated.\n"
                                            f"- Output location mode: **{save_option}**."
                                        )

                                        if save_option == "Apply to current working CSV in memory (default, allows preview/undo)":
                                            st.session_state["tools_edit_csv_df"] = df_new
                                            _push_edit_csv_state(df_new, "Import columns from external CSV", df_working if df_working is not None else pd.DataFrame(), df_new)
                                            st.success(report_msg)
                                        elif save_option == "Overwrite target CSV file directly":
                                            if unified_csv_path and os.path.exists(unified_csv_path):
                                                df_new.to_csv(unified_csv_path, index=False)
                                                _get_csv_metadata_cached.clear()
                                                _get_csv_preview_df_cached.clear()
                                                _get_csv_preview_grid_cached.clear()
                                                if df_working is not None:
                                                    st.session_state["tools_edit_csv_df"] = df_new
                                                    _push_edit_csv_state(df_new, "Import columns from external CSV", df_working, df_new)
                                                st.success(f"{report_msg}\n\nOverwrote target CSV at `{unified_csv_path}`.")
                                            else:
                                                st.error("Target CSV path not found or empty.")
                                        else:
                                            os.makedirs(os.path.dirname(custom_new_path), exist_ok=True)
                                            df_new.to_csv(custom_new_path, index=False)
                                            if df_working is not None:
                                                _push_edit_csv_state(df_new, "Import columns from external CSV", df_working, df_new)
                                            st.success(f"{report_msg}\n\nSaved to new file: `{custom_new_path}`.")

                                    except Exception as e:
                                        st.error(f"Error during import: {e}")

        # Previews & Save section (always visible, with memory vs offline disk fallbacks)
        st.divider()
        st.markdown("#### Preview Grid")
        if df_working is not None:
            preview_df = _preview_grid(df_working)
            st.caption(f"Rows: {len(df_working):,} | Columns: {len(df_working.columns):,} (In-Memory Working Copy)")
            st.dataframe(preview_df, width="stretch", hide_index=True)
            st.markdown("#### Data Preview (first 200 rows)")
            st.dataframe(df_working.head(200), width="stretch", hide_index=False)
        else:
            if target_metadata:
                preview_df = _get_csv_preview_grid_cached(unified_csv_path)
                st.caption(f"Rows: {target_metadata['row_count']:,} | Columns: {len(target_metadata['columns']):,} (Offline Disk Preview)")
                st.dataframe(preview_df, width="stretch", hide_index=True)
                st.markdown("#### Data Preview (first 200 rows)")
                disk_preview_df = _get_csv_preview_df_cached(unified_csv_path)
                st.dataframe(disk_preview_df, width="stretch", hide_index=False)
            else:
                st.info("No unified CSV metadata available for preview.")

        last_preview = st.session_state.get("tools_edit_csv_last_preview", {})
        if last_preview:
            st.divider()
            st.markdown("#### Before / After Preview")
            st.caption(last_preview.get("label", "Last operation"))
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("**Before**")
                st.dataframe(last_preview.get("before"), width="stretch", hide_index=True)
            with p2:
                st.markdown("**After**")
                st.dataframe(last_preview.get("after"), width="stretch", hide_index=True)

        st.divider()
        st.markdown("#### Save")
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            save_confirm = st.checkbox("Confirm overwrite unified CSV", key="tools_edit_csv_save_confirm")
            if st.button("Save to Unified CSV", type="primary", key="tools_edit_csv_save_unified"):
                if df_working is None:
                    st.error("No working copy loaded in memory to save. Use the import tool above to merge columns directly to files.")
                elif not unified_csv_path or not os.path.exists(unified_csv_path):
                    st.error("Unified CSV path not found.")
                elif not save_confirm:
                    st.warning("Confirm overwrite to save.")
                else:
                    df_working.to_csv(unified_csv_path, index=False)
                    st.success("Unified CSV overwritten successfully.")
        with save_col2:
            new_path = st.text_input(
                "Save As New CSV",
                value=str(project_root / "outputs/phenotyping/edit_csv/edited_unified.csv"),
                key="tools_edit_csv_save_as",
            )
            if st.button("Save As New CSV", key="tools_edit_csv_save_new"):
                if df_working is None:
                    st.error("No working copy loaded in memory to save. Use the import tool above to merge columns directly to files.")
                elif not new_path:
                    st.error("Provide a save path.")
                else:
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    df_working.to_csv(new_path, index=False)
                    st.success(f"Saved new CSV to {new_path}")


    with sub_tabs[3]:
        from .threshold_gating import render_threshold_tab
        render_threshold_tab(project_root, global_input_path)

    # Persist all current TOOLS session inputs automatically
    try:
        _prune_visual_manual_reference_runtime_cache(cached_params)
        for key, value in st.session_state.items():
            if not key.startswith("tools_"):
                continue
            if _is_visual_manual_reference_runtime_cache_key(key):
                continue
            if isinstance(value, pd.DataFrame):
                continue
            if isinstance(value, (set, Path)):
                cached_params[key] = list(value) if isinstance(value, set) else str(value)
                continue
            cached_params[key] = value
        with open(tools_cache, "w") as f:
            json.dump(cached_params, f, indent=2)
    except Exception:
        pass
