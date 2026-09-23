"""
Resolve (Threshold-Based) Phenotyping Subtab
Professional rule-based phenotyping with offline phenotype definition table.
Synchronized with Step6C_phenotype_threshold.ipynb notebook.
"""

import streamlit as st
import pandas as pd
from spacemap import dataio as _dataio
import numpy as np
import os
import sys
import json
import time
import re
import ast
import hashlib
from datetime import datetime
from pathlib import Path
from typing import cast
import matplotlib.pyplot as plt

from spacemap_gui.resolve_rule_parser import parse_resolve_rule_text

# Import shared utilities
from .common import (
    logger,
    get_batch_id_column_help_text,
    get_marker_list,
    get_marker_workspace_panel,
    render_method_marker_selection,
    normalize_optional_column_name,
    get_output_column_help_text,
    project_root_calc,
    _normalize_marker_name,
    _load_thresholds_table,
    _threshold_settings_fingerprint,
    _preferred_threshold_directory,
    _threshold_table_has_values,
    _threshold_table_coverage,
    load_universal_gating_csv,
    build_leiden_heatmap_figure,
    build_marker_source_note,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_export_split_data,
    render_data_filter_ui,
    apply_data_filter,
    render_settings_export_import,
    render_method_intro,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    resolve_notebook_marker_defaults,
    DEFAULT_PHENOTYPES,
    get_all_phenotype_options,
    render_add_custom_phenotype_ui,
    _THRESHOLD_MODE_TO_KEY,
    AUTO_THRESHOLD_METHODS,
    compute_auto_thresholds,
)
from spacemap_gui import terminal

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

try:
    from spacemap import ph_thres_utils
    sys.modules["ph_thres_utils"] = ph_thres_utils
    from spacemap.ph_thres_utils import (
        apply_thresholds_per_batch,
        assign_phenotypes_binary_tiebreak,
        apply_flexible_normalization,
        apply_threshold_phenotypes
    )
except ImportError:
    import importlib.util
    try:
        # Check parents[3] / "spacemap" / "ph_thres_utils.py" (current workspace package structure)
        _PH_THRES_PATH = Path(__file__).resolve().parents[3] / "spacemap" / "ph_thres_utils.py"
        if not _PH_THRES_PATH.exists():
            # Check parents[4] / "src" / "spacemap" / "ph_thres_utils.py"
            _PH_THRES_PATH = Path(__file__).resolve().parents[4] / "src" / "spacemap" / "ph_thres_utils.py"
        if not _PH_THRES_PATH.exists():
            # Check legacy V13 path
            _PH_THRES_PATH = Path(__file__).resolve().parents[3] / "src" / "ph_thres_utils.py"

        if _PH_THRES_PATH.exists():
            _ph_spec = importlib.util.spec_from_file_location("ph_thres_utils", _PH_THRES_PATH)
            if _ph_spec is not None and _ph_spec.loader is not None:
                ph_thres_utils = importlib.util.module_from_spec(_ph_spec)
                sys.modules.setdefault("ph_thres_utils", ph_thres_utils)
                _ph_spec.loader.exec_module(ph_thres_utils)
                
                apply_thresholds_per_batch = ph_thres_utils.apply_thresholds_per_batch
                assign_phenotypes_binary_tiebreak = ph_thres_utils.assign_phenotypes_binary_tiebreak
                apply_flexible_normalization = ph_thres_utils.apply_flexible_normalization
                apply_threshold_phenotypes = ph_thres_utils.apply_threshold_phenotypes
            else:
                ph_thres_utils = None
        else:
            ph_thres_utils = None
    except Exception:
        ph_thres_utils = None

if ph_thres_utils is None:
    apply_thresholds_per_batch = None
    assign_phenotypes_binary_tiebreak = None
    apply_flexible_normalization = None
    apply_threshold_phenotypes = None

# STANDARD_PHENOTYPES and DEFAULT_PHENOTYPES are imported from common.py


RESOLVE_TRANSFORM_OPTIONS = ("None", "ArcSinh", "Log")
RESOLVE_NORMALIZATION_OPTIONS = ("None", "Z1", "Z2", "CDF", "NLog")


def _dataio_load(path, columns=None):
    """Read a large table through the shared loader.

    Uses the Parquet fast copy when one is current, otherwise the CSV. Passing
    *columns* reads only those.
    """
    from spacemap import dataio

    return dataio.load(path, columns=columns)


def _canonical_resolve_transform(value: object) -> str:
    """Return the short UI name for a saved or runtime transformation value."""
    key = "" if value is None else str(value).strip().lower()
    aliases = {
        "": "None",
        "none": "None",
        "identity": "None",
        "identity_residual": "None",
        "raw": "None",
        "raw_residual": "None",
        "arcsinh": "ArcSinh",
        "asinh": "ArcSinh",
        "asinh_difference": "ArcSinh",
        "arcsinh_difference": "ArcSinh",
        "asinh_distance": "ArcSinh",
        "log": "Log",
        "log1p": "Log",
        "log_difference": "Log",
        "log1p_difference": "Log",
    }
    if key not in aliases:
        raise ValueError(f"Unknown RESOLVE transformation: {value!r}")
    return aliases[key]


def _canonical_resolve_normalization_step(value: object) -> str:
    """Normalize a normalization-slot value to one of the UI choices."""
    key = "" if value is None else str(value).strip().lower()
    aliases = {
        "": "None",
        "none": "None",
        "z1": "Z1",
        "z2": "Z2",
        "cdf": "CDF",
        "nlog": "NLog",
        "-log": "NLog",
    }
    if key not in aliases:
        raise ValueError(f"Unknown RESOLVE normalization step: {value!r}")
    return aliases[key]


def _normalize_resolve_pipeline(
    transformation: object,
    arcsinh_cofactor: object,
    normalization_slots: list[object] | tuple[object, ...] | None,
) -> dict[str, object]:
    """Build the runtime configuration from the simple transformation and four slots."""
    transform_name = _canonical_resolve_transform(transformation)
    raw_slots = list(normalization_slots or [])
    if len(raw_slots) > 4:
        raise ValueError("RESOLVE supports four normalization order slots.")
    slots = [_canonical_resolve_normalization_step(value) for value in raw_slots]
    slots.extend(["None"] * (4 - len(slots)))
    selected = [step for step in slots if step != "None"]
    if len(selected) != len(set(selected)):
        raise ValueError("A normalization step is listed more than once.")
    if arcsinh_cofactor in (None, ""):
        cofactor = 5.0
    else:
        try:
            cofactor = float(arcsinh_cofactor)
        except (TypeError, ValueError) as exc:
            raise ValueError("ArcSinh cofactor must be a positive number.") from exc
    if transform_name == "ArcSinh" and (
        not np.isfinite(cofactor) or cofactor <= 0
    ):
        raise ValueError("ArcSinh cofactor must be a positive finite number.")
    if not np.isfinite(cofactor) or cofactor <= 0:
        cofactor = 5.0

    runtime_names = {
        "None": "identity_residual",
        "ArcSinh": "asinh_difference",
        "Log": "log1p_difference",
    }
    return {
        "transformation": transform_name,
        "tie_score_transform": runtime_names[transform_name],
        "tie_score_cofactor": cofactor,
        "normalization_slots": slots,
        "norm_order": selected,
        "norm_z1": "Z1" in selected,
        "norm_z2": "Z2" in selected,
        "norm_cdf": "CDF" in selected,
        "norm_nlog": "NLog" in selected,
        # Kept only as a compatibility field for older saved projects. The
        # separate legacy inverse-sinh switch is intentionally no longer used.
        "norm_sinh": False,
    }


def _migrate_resolve_pipeline_config(config: dict | None) -> dict:
    """Migrate legacy checkbox/order settings to the simple four-slot model."""
    migrated = dict(config or {})
    transform_value = migrated.get(
        "transformation", migrated.get("tie_score_transform", "None")
    )
    if "normalization_slots" in migrated:
        slots = list(migrated.get("normalization_slots") or [])
    elif "norm_order" in migrated:
        slots = list(migrated.get("norm_order") or [])
    else:
        slots = [
            step
            for step, key in (
                ("Z1", "norm_z1"),
                ("Z2", "norm_z2"),
                ("CDF", "norm_cdf"),
                ("NLog", "norm_nlog"),
            )
            if bool(migrated.get(key, False))
        ]
    # A malformed legacy order should remain editable instead of preventing
    # the tab from loading. Keep the first occurrence and clear duplicates.
    clean_slots: list[str] = []
    seen_steps: set[str] = set()
    for value in slots[:4]:
        canonical = _canonical_resolve_normalization_step(value)
        if canonical != "None" and canonical in seen_steps:
            canonical = "None"
        if canonical != "None":
            seen_steps.add(canonical)
        clean_slots.append(canonical)
    pipeline = _normalize_resolve_pipeline(
        transform_value,
        migrated.get("tie_score_cofactor", 5.0),
        clean_slots,
    )
    migrated.update(pipeline)
    return migrated


def _resolve_pipeline_errors(pipeline: dict[str, object]) -> list[str]:
    """Return short scientific-order errors for a normalized pipeline."""
    order = list(pipeline.get("norm_order", []) or [])
    if "NLog" in order and (
        "CDF" not in order or order.index("CDF") > order.index("NLog")
    ):
        return ["NLog must come after CDF."]
    return []


def _resolve_pipeline_preview(
    transformation: object,
    normalization_slots: list[object] | tuple[object, ...] | None,
) -> str:
    """Return the compact sequence displayed beneath the controls."""
    transform_name = _canonical_resolve_transform(transformation)
    slots = [
        _canonical_resolve_normalization_step(value)
        for value in list(normalization_slots or [])
    ]
    selected = [step for step in slots if step != "None"]
    if len(selected) != len(set(selected)):
        raise ValueError("A normalization step is listed more than once.")
    sequence = ([] if transform_name == "None" else [transform_name]) + selected
    return " → ".join(sequence) if sequence else "None"


def _transform_resolve_threshold_distance(
    raw_values: pd.DataFrame,
    threshold_values: pd.DataFrame | pd.Series | float,
    transformation: object,
    arcsinh_cofactor: object = 5.0,
) -> pd.DataFrame:
    """Transform raw and threshold values before computing the positive distance."""
    transform_name = _canonical_resolve_transform(transformation)
    if transform_name == "ArcSinh":
        try:
            cofactor = float(arcsinh_cofactor)
        except (TypeError, ValueError) as exc:
            raise ValueError("ArcSinh cofactor must be a positive number.") from exc
        if not np.isfinite(cofactor) or cofactor <= 0:
            raise ValueError("ArcSinh cofactor must be a positive finite number.")
        distance = (
            np.arcsinh(raw_values / cofactor)
            - np.arcsinh(threshold_values / cofactor)
        )
    elif transform_name == "Log":
        # log1p is defined at zero and is safer for zero-valued intensity data.
        transformed_raw = np.log1p(raw_values.clip(lower=0))
        if hasattr(threshold_values, "clip"):
            transformed_threshold = np.log1p(threshold_values.clip(lower=0))
        else:
            transformed_threshold = np.log1p(max(float(threshold_values), 0.0))
        distance = transformed_raw - transformed_threshold
    else:
        distance = raw_values - threshold_values
    return distance.clip(lower=0)


def _extract_markers_base_from_step6c():
    """Extract MARKERS_BASE from Step6C_phenotype_threshold.ipynb cell 4."""
    try:
        notebook_path = project_root_calc / "notebooks" / "Step6C_phenotype_threshold.ipynb"
        if not notebook_path.exists():
            return []
        
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        for cell in nb.get('cells', []):
            source = "".join(cell.get('source', []))
            if not source or 'MARKERS_BASE' not in source:
                continue
            
            # Extract MARKERS_BASE list
            match = re.search(r'MARKERS_BASE\s*=\s*\[(.*?)\]', source, re.DOTALL)
            if match:
                try:
                    list_str = '[' + match.group(1) + ']'
                    markers = ast.literal_eval(list_str)
                    return markers if isinstance(markers, list) else []
                except Exception:
                    pass
        
        return []
    except Exception as e:
        _log_console(f"Warning: Could not extract MARKERS_BASE from Step6C: {e}", level="warning")
        return []


def _extract_config_from_step6c():
    """Extract configuration parameters from Step6C_phenotype_threshold.ipynb."""
    try:
        notebook_path = project_root_calc / "notebooks" / "Step6C_phenotype_threshold.ipynb"
        if not notebook_path.exists():
            return {}
        
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        config = {
            'phenotypes': {},
            'norm_z1': True,
            'norm_z2': True,
            'norm_cdf': True,
            'norm_nlog': True,
            'norm_sinh': False,
            'norm_order': ['Z1', 'Z2', 'CDF', 'NLog'],
            'threshold_statistic': 'median',
            'tie_score_transform': 'asinh_difference',
            'tie_score_cofactor': 5.0,
            'threshold_mode': 'Predefined (CSV)',
            'enable_filtering': False,
            'filter_column': '',
            'filter_value': ''
        }
        
        for cell in nb.get('cells', []):
            source = "".join(cell.get('source', []))
            if not source:
                continue
            
            # Extract PHENOTYPE_DEFINITIONS
            if 'PHENOTYPE_DEFINITIONS' in source and '=' in source:
                match = re.search(r'PHENOTYPE_DEFINITIONS\s*=\s*(\{.*?\n\})', source, re.DOTALL)
                if match:
                    try:
                        dict_str = match.group(1)
                        phenotypes = ast.literal_eval(dict_str)
                        config['phenotypes'] = phenotypes if isinstance(phenotypes, dict) else {}
                    except Exception:
                        pass
            
            # Extract each normalization flag from its own assignment. A cell
            # can contain a mixture of True and False values.
            for constant, config_key in (
                ('NORM_APPLY_Z1_MARKERWISE', 'norm_z1'),
                ('NORM_APPLY_Z2_CELLWISE', 'norm_z2'),
                ('NORM_APPLY_CDF', 'norm_cdf'),
                ('NORM_APPLY_NLOG', 'norm_nlog'),
            ):
                match = re.search(rf'{constant}\s*=\s*(True|False)', source)
                if match:
                    config[config_key] = match.group(1) == 'True'
            if 'NORMALIZATION_ORDER' in source:
                match = re.search(r'NORMALIZATION_ORDER\s*=\s*(\[[^\]]*\])', source)
                if match:
                    try:
                        parsed_order = ast.literal_eval(match.group(1))
                        if isinstance(parsed_order, list):
                            config['norm_order'] = [str(step) for step in parsed_order]
                    except Exception:
                        pass
            
            # Extract threshold statistic
            if 'THRESHOLD_STATISTIC' in source:
                match = re.search(r'THRESHOLD_STATISTIC\s*=\s*["\']([^"\']+)["\']', source)
                if match:
                    config['threshold_statistic'] = match.group(1)
            if 'TIE_SCORE_TRANSFORM' in source:
                match = re.search(r'TIE_SCORE_TRANSFORM\s*=\s*["\']([^"\']+)["\']', source)
                if match:
                    config['tie_score_transform'] = match.group(1)
            if 'TIE_SCORE_COFACTOR' in source:
                match = re.search(r'TIE_SCORE_COFACTOR\s*=\s*([0-9.eE+-]+)', source)
                if match:
                    try:
                        config['tie_score_cofactor'] = float(match.group(1))
                    except ValueError:
                        pass
        
        return config
    except Exception as e:
        _log_console(f"Warning: Could not extract config from Step6C: {e}", level="warning")
        return {}


def _resolve_enabled_normalization_steps(config: dict) -> list[str]:
    """Return Resolve's enabled normalization steps in their configured order."""
    enabled = {
        "Z1": bool(config.get("norm_z1", False)),
        "Z2": bool(config.get("norm_z2", False)),
        "CDF": bool(config.get("norm_cdf", False)),
        "NLog": bool(config.get("norm_nlog", False)),
    }
    configured_order = config.get("norm_order", ["Z1", "Z2", "CDF", "NLog"])
    ordered: list[str] = []
    for step in list(configured_order or []) + ["Z1", "Z2", "CDF", "NLog"]:
        canonical = str(step).strip()
        if canonical.lower() == "nlog":
            canonical = "NLog"
        else:
            canonical = canonical.upper()
        if canonical in enabled and enabled[canonical] and canonical not in ordered:
            ordered.append(canonical)
    return ordered


def _resolve_plot_normalization_label(
    config: dict,
    marker_columns: list[str] | None = None,
    provenance: dict | None = None,
) -> str:
    """Describe the exact cell-level values stored for Resolve visualization."""
    provenance = provenance if isinstance(provenance, dict) else {}
    steps = provenance.get("normalization_steps")
    if not isinstance(steps, list):
        steps = _resolve_enabled_normalization_steps(config)

    preparation: list[str] = []
    tie_transform = str(provenance.get(
        "tie_score_transform", config.get("tie_score_transform", "identity_residual")
    ))
    transform_name = _canonical_resolve_transform(tie_transform)
    if transform_name != "None":
        preparation.append(transform_name)

    preparation.extend(str(step) for step in steps)
    if not preparation:
        preparation.append("None")

    batch_column = provenance.get("batch_column", config.get("batch_col"))
    grouping = f"within {batch_column}" if batch_column else "globally"
    return f"{' → '.join(preparation)} ({grouping})"


def _apply_thresholds_from_table(
    df: pd.DataFrame,
    thresholds_table: pd.DataFrame,
    selected_markers: list[str],
    batch_col: str | None,
    norm_flags: tuple[bool, bool, bool, bool],
    norm_order: list[str] | None,
    method_tag: str,
    tie_score_transform: str = "identity_residual",
    tie_score_cofactor: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply table thresholds and prepare continuous RESOLVE tie scores."""
    bin_suffix = f"_bin_{method_tag}"
    norm_suffix = f"_norm_{method_tag}"

    def _normalize_batch_id(value: str) -> str:
        text = str(value).strip()
        text = re.sub(r"\.csv$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\.ome\.tiff_-_.*$", "", text, flags=re.IGNORECASE)  # strips _-_Merged_Image, _-_Fixed_Nolan_Image, etc.
        text = re.sub(r"\.ome\.tiff$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\.ome$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"_fully_annotated$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"_cell_features.*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"_annotated$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"_processed$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"_summary$", "", text, flags=re.IGNORECASE)
        return text.strip("._ ").lower()

    df_binary = pd.DataFrame(index=df.index)
    if thresholds_table is None or thresholds_table.empty:
        return df_binary, pd.DataFrame(index=df.index)

    table_cols = set(thresholds_table.columns)
    has_batch = batch_col in df.columns and "batch_id" in table_cols

    if has_batch:
        table_map = {}
        for _, row in thresholds_table.iterrows():
            if "batch_id" not in row:
                continue
            raw_key = row["batch_id"]
            if pd.isna(raw_key):
                continue
            norm_key = _normalize_batch_id(raw_key)
            if norm_key:
                table_map[norm_key] = row
        batch_series = df[batch_col]
        # Build fuzzy fallback map: normalized_data_batch -> nearest threshold key
        from difflib import get_close_matches as _gcm
        _table_keys = list(table_map.keys())
        _unique_data_norms = set(batch_series.astype(str).map(_normalize_batch_id))
        _fuzzy_map: dict[str, str] = {}
        for _dn in _unique_data_norms:
            if _dn not in table_map:
                _fm = _gcm(_dn, _table_keys, n=1, cutoff=0.6)
                if _fm:
                    _fuzzy_map[_dn] = _fm[0]
    else:
        table_map = {"__global__": thresholds_table.iloc[0]}
        batch_series = pd.Series("__global__", index=df.index)
        _fuzzy_map = {}

    markers_present = [m for m in selected_markers if m in df.columns]
    if not markers_present:
        return df_binary, pd.DataFrame(index=df.index)

    raw_values = df[markers_present].apply(pd.to_numeric, errors="coerce")
    threshold_values = pd.DataFrame(np.nan, index=df.index, columns=markers_present)
    for batch_val, row in table_map.items():
        if batch_val == "__global__":
            mask = batch_series == batch_val
        else:
            norm_data = batch_series.astype(str).map(_normalize_batch_id)
            mask = norm_data == batch_val
            if not mask.any():
                # fuzzy fallback: check if any data norm is fuzzy-mapped to this threshold key
                mask = norm_data.map(lambda n: _fuzzy_map.get(n) == batch_val)
        if not mask.any():
            continue
        for marker in markers_present:
            base = _normalize_marker_name(marker)
            thresh_val = row.get(marker)
            if pd.isna(thresh_val):
                thresh_val = row.get(base, np.nan)
            try:
                thr = float(thresh_val)
            except Exception:
                raise ValueError(
                    f"Threshold for marker '{marker}' and batch '{batch_val}' is missing or non-numeric."
                )
            if not np.isfinite(thr):
                raise ValueError(
                    f"Threshold for marker '{marker}' and batch '{batch_val}' is not finite."
                )
            threshold_values.loc[mask, marker] = thr

    missing_threshold_mask = threshold_values.isna()
    if missing_threshold_mask.any().any():
        missing_markers = missing_threshold_mask.columns[missing_threshold_mask.any()].tolist()
        raise ValueError(
            "Threshold table did not cover every cell for marker(s): "
            + ", ".join(missing_markers)
        )

    residual = raw_values - threshold_values
    for marker in markers_present:
        out_col = f"{marker}{bin_suffix}"
        df_binary[out_col] = (residual[marker] > 0).fillna(False).astype(int)

    temp = _transform_resolve_threshold_distance(
        raw_values,
        threshold_values,
        tie_score_transform,
        tie_score_cofactor,
    )

    apply_z1, apply_z2, apply_cdf, apply_nlog = norm_flags
    group_cols = [batch_col] if batch_col in df.columns else []
    norm_input = pd.concat([temp, df[group_cols]], axis=1) if group_cols else temp
    df_norm = _apply_flexible_normalization_safe(
        norm_input,
        marker_cols=markers_present,
        group_cols=group_cols if group_cols else None,
        suffix=norm_suffix,
        apply_z1=apply_z1,
        apply_z2=apply_z2,
        apply_cdf=apply_cdf,
        apply_nlog=apply_nlog,
        norm_order=norm_order,
    )

    return df_binary, df_norm


def _apply_resolve_without_thresholds(
    df: pd.DataFrame,
    selected_markers: list[str],
    batch_col: str | None,
    config: dict,
    norm_order: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the selected score pipeline with a zero threshold."""
    markers_present = [marker for marker in selected_markers if marker in df.columns]
    raw_values = df[markers_present].apply(pd.to_numeric, errors="coerce")
    binary_df = pd.DataFrame(index=df.index)
    for marker in markers_present:
        binary_df[f"{marker}_bin_resolve"] = (
            raw_values[marker] > 0
        ).fillna(False).astype(int)

    transformed = _transform_resolve_threshold_distance(
        raw_values,
        0.0,
        config.get("tie_score_transform", "identity_residual"),
        config.get("tie_score_cofactor", 5.0),
    )
    group_cols = [batch_col] if batch_col and batch_col in df.columns else []
    norm_input = (
        pd.concat([transformed, df[group_cols]], axis=1)
        if group_cols else transformed
    )
    norm_df = _apply_flexible_normalization_safe(
        norm_input,
        marker_cols=markers_present,
        group_cols=group_cols or None,
        suffix="_norm_resolve",
        apply_z1=bool(config.get("norm_z1", False)),
        apply_z2=bool(config.get("norm_z2", False)),
        apply_cdf=bool(config.get("norm_cdf", False)),
        apply_nlog=bool(config.get("norm_nlog", False)),
        norm_order=norm_order,
    )
    return binary_df, norm_df


RESOLVE_Z1_MODE = "Z1 score (no threshold files)"
DEFAULT_RESOLVE_Z1_CUTOFF = 1.5


def _apply_resolve_z1_positivity(
    df: pd.DataFrame,
    selected_markers: list[str],
    batch_col: str | None,
    cutoff: float = DEFAULT_RESOLVE_Z1_CUTOFF,
    cofactor: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positive where the per-batch Z1 score is above ``cutoff`` (see
    ``spacemap.ph_thres_utils.apply_z1_positivity``, shared with notebook Step6C)."""
    from spacemap.ph_thres_utils import apply_z1_positivity

    return apply_z1_positivity(
        df, selected_markers, batch_col, method_tag="resolve", cutoff=cutoff, cofactor=cofactor,
    )


def _resolve_threshold_fingerprint(
    threshold_dir: str | os.PathLike | None,
    config: dict,
    selected_markers: list[str],
) -> str:
    """Fingerprint the settings that produced a reviewed threshold table."""
    return _threshold_settings_fingerprint(
        threshold_dir,
        config.get("threshold_source", "Auto"),
        config.get("threshold_statistic", "median"),
        config.get("threshold_marker_col"),
        config.get("threshold_value_col"),
        config.get("polygon_prefix", "Polygon"),
        selected_markers,
    )




# Held apart from the criteria columns: these markers never define a phenotype,
# they only overturn one when they outweigh the markers that justified it.
_RESOLVE_EXCLUSION_COL = "Must not have"
# Supportive markers never gate a phenotype. They only strengthen it when two
# phenotypes both qualify and the tie-break has to choose, which is where the
# "conditional but not critical" evidence belongs in a binary rule system.
_RESOLVE_SUPPORT_COL = "Supportive"
_RESOLVE_EXTRA_COLS = (_RESOLVE_SUPPORT_COL, _RESOLVE_EXCLUSION_COL)


def _coerce_phenotype_table(df: pd.DataFrame, criteria_cols: list[str]) -> pd.DataFrame:
    """Ensure phenotype table has expected columns and list values."""
    if df is None or df.empty:
        return pd.DataFrame([{"Phenotype": "", **{c: [] for c in criteria_cols},
                              **{e: [] for e in _RESOLVE_EXTRA_COLS}}])

    if "Phenotype" not in df.columns:
        df["Phenotype"] = ""

    for col in criteria_cols:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]

    for col in criteria_cols:
        df[col] = df[col].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [str(v)]))

    for extra in _RESOLVE_EXTRA_COLS:
        if extra not in df.columns:
            df[extra] = [[] for _ in range(len(df))]
        df[extra] = df[extra].apply(
            lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [str(v)])
        )

    return df[["Phenotype", *criteria_cols, *_RESOLVE_EXTRA_COLS]].copy()


# Start with a single required set. Extra ones are alternatives the user adds
# only if the phenotype genuinely needs them.
_RESOLVE_DEFAULT_CRITERIA_COLS = [
    "Criteria 1",
]

_RESOLVE_PHENO_WIDGET_PREFIXES = (
    "resolve_pheno_",
    "resolve_criteria_",
    "resolve_remove_",
    "resolve_support_",
    "resolve_exclusion_",
)


def _build_default_resolve_pheno_payload() -> dict:
    """Return the default phenotype table payload for Resolve."""
    rows = []
    for pheno_name in DEFAULT_PHENOTYPES:
        row = {"Phenotype": pheno_name}
        for col_name in _RESOLVE_DEFAULT_CRITERIA_COLS:
            row[col_name] = []
        rows.append(row)
    if not rows:
        rows = [{"Phenotype": "", **{col_name: [] for col_name in _RESOLVE_DEFAULT_CRITERIA_COLS}}]
    return {
        "criteria_cols": list(_RESOLVE_DEFAULT_CRITERIA_COLS),
        "table": rows,
    }


def _normalize_resolve_pheno_payload(payload: dict | None) -> dict:
    """Normalize a Resolve phenotype payload read from disk or import."""
    default_payload = _build_default_resolve_pheno_payload()
    if not isinstance(payload, dict):
        payload = default_payload

    criteria_cols = payload.get("criteria_cols", default_payload["criteria_cols"])
    if not isinstance(criteria_cols, list):
        criteria_cols = list(default_payload["criteria_cols"])
    criteria_cols = [str(col).strip() for col in criteria_cols if str(col).strip()]
    if not criteria_cols:
        criteria_cols = list(default_payload["criteria_cols"])

    table_rows = payload.get("table", default_payload["table"])
    if isinstance(table_rows, pd.DataFrame):
        table_df = table_rows.copy()
    elif isinstance(table_rows, list):
        table_df = pd.DataFrame(table_rows)
    else:
        table_df = pd.DataFrame(default_payload["table"])
    table_df = _coerce_phenotype_table(table_df, criteria_cols)

    return {
        "criteria_cols": criteria_cols,
        "table": table_df.to_dict(orient="records"),
    }


def _get_resolve_criteria_cols_from_state(resolve_state: dict[str, object]) -> list[str]:
    """Return the current Resolve criteria columns from state or defaults."""
    criteria_cols = resolve_state.get('resolve_criteria_cols')
    if not isinstance(criteria_cols, list):
        return list(_RESOLVE_DEFAULT_CRITERIA_COLS)

    normalized_cols = [str(col).strip() for col in criteria_cols if str(col).strip()]
    if not normalized_cols:
        return list(_RESOLVE_DEFAULT_CRITERIA_COLS)
    return normalized_cols


def _get_resolve_pheno_table_from_state(
    resolve_state: dict[str, object],
    criteria_cols: list[str],
    active_cache_path: Path | None,
) -> pd.DataFrame:
    """Return the best available phenotype table from state, cache, or defaults."""
    state_table = resolve_state.get('resolve_pheno_table')
    if isinstance(state_table, pd.DataFrame):
        return _coerce_phenotype_table(state_table.copy(), criteria_cols)

    payload = _read_resolve_pheno_payload(active_cache_path)
    if payload is not None:
        return _coerce_phenotype_table(pd.DataFrame(payload['table']), criteria_cols)

    default_payload = _build_default_resolve_pheno_payload()
    return _coerce_phenotype_table(pd.DataFrame(default_payload['table']), criteria_cols)


def _snapshot_resolve_pheno_table_from_state(
    resolve_state: dict[str, object],
    criteria_cols: list[str],
    fallback_table: pd.DataFrame,
) -> pd.DataFrame:
    """Capture the currently displayed Resolve phenotype table from widget state."""
    table = _coerce_phenotype_table(fallback_table.copy(), criteria_cols)
    rows: list[dict[str, object]] = []

    for idx, row in table.iterrows():
        row_out: dict[str, object] = {
            'Phenotype': resolve_state.get(f'resolve_pheno_{idx}', row.get('Phenotype', '')),
        }
        for col_name in criteria_cols:
            widget_value = resolve_state.get(f'resolve_criteria_{idx}_{col_name}', row.get(col_name, []))
            if isinstance(widget_value, list):
                row_out[col_name] = [str(marker) for marker in widget_value if str(marker).strip()]
            elif pd.isna(widget_value):
                row_out[col_name] = []
            else:
                row_out[col_name] = [str(widget_value)]
        # Supportive and Must-not-have live outside criteria_cols, so they have to
        # be carried explicitly - rebuilding from criteria alone silently wiped
        # them on every rerun.
        for extra, widget_prefix in (
            (_RESOLVE_SUPPORT_COL, "resolve_support_"),
            (_RESOLVE_EXCLUSION_COL, "resolve_exclusion_"),
        ):
            widget_value = resolve_state.get(f"{widget_prefix}{idx}", row.get(extra, []))
            if isinstance(widget_value, list):
                row_out[extra] = [str(m) for m in widget_value if str(m).strip()]
            elif widget_value is None or (not isinstance(widget_value, (list, tuple)) and pd.isna(widget_value)):
                row_out[extra] = []
            else:
                row_out[extra] = [str(widget_value)]
        rows.append(row_out)

    return _coerce_phenotype_table(pd.DataFrame(rows), criteria_cols)


def _prune_resolve_pheno_table_markers(
    pheno_table: pd.DataFrame,
    criteria_cols: list[str],
    selected_markers: list[str],
) -> pd.DataFrame:
    """Keep phenotype rows intact while dropping criteria markers no longer selected."""
    pruned_table = _coerce_phenotype_table(pheno_table.copy(), criteria_cols)

    # The live panel selection and the confirmed selection are two different
    # lists, and the table is written against the confirmed one. Pruning against
    # the panel list alone silently emptied every criterion whenever the panel
    # had not repopulated - the phenotype names survived, so it looked like the
    # rules had been applied and then ignored.
    allowed_markers = set(selected_markers or [])
    confirmed = st.session_state.get("resolve_confirmed_markers", [])
    if isinstance(confirmed, list):
        allowed_markers |= {str(m) for m in confirmed}
    if not allowed_markers:
        # Nothing is known about the panel yet. Dropping every marker here is
        # never the right answer, so leave the table untouched.
        return pruned_table

    for col_name in [*criteria_cols, *_RESOLVE_EXTRA_COLS]:
        pruned_table[col_name] = pruned_table[col_name].apply(
            lambda values: [marker for marker in values if marker in allowed_markers]
        )
    return pruned_table


def _sync_resolve_pheno_table_for_marker_selection(
    resolve_state: dict[str, object],
    selected_markers: list[str],
    active_cache_path: Path | None,
    primary_cache_path: Path,
    file_cache_path: Path | None,
) -> pd.DataFrame:
    """Preserve the Resolve phenotype table when marker selections change."""
    # Applying text rules replaces the table wholesale. This function otherwise
    # rebuilds it from widget state and prunes it against the marker panel, both
    # of which discard a table the widgets have never rendered - so a freshly
    # applied set of rules is handed straight back as empty rows. One render of
    # deference lets the widgets draw from the new table first.
    if resolve_state.pop("_resolve_rules_just_applied", False):
        table = resolve_state.get("resolve_pheno_table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
    criteria_cols = _get_resolve_criteria_cols_from_state(resolve_state)
    fallback_table = _get_resolve_pheno_table_from_state(resolve_state, criteria_cols, active_cache_path)
    snapshot_table = _snapshot_resolve_pheno_table_from_state(resolve_state, criteria_cols, fallback_table)
    pruned_table = _prune_resolve_pheno_table_markers(snapshot_table, criteria_cols, selected_markers)
    payload = _write_resolve_pheno_payload(
        {
            'criteria_cols': criteria_cols,
            'table': pruned_table.to_dict(orient='records'),
        },
        [primary_cache_path, file_cache_path],
    )

    _clear_resolve_pheno_widget_state()
    resolve_state['resolve_criteria_cols'] = list(payload['criteria_cols'])
    resolve_state['resolve_pheno_table'] = pd.DataFrame(payload['table'])
    resolve_state['resolve_loaded_pheno_cache_sig'] = _resolve_pheno_cache_signature(primary_cache_path)
    return cast(pd.DataFrame, resolve_state['resolve_pheno_table'])


def _resolve_pheno_cache_signature(cache_path: Path | None) -> str | None:
    """Return a lightweight signature for a Resolve phenotype cache file."""
    if cache_path is None or not cache_path.exists():
        return None
    try:
        stat = cache_path.stat()
    except OSError:
        return None
    return f"{cache_path.name}:{stat.st_mtime_ns}:{stat.st_size}"


def _read_resolve_pheno_payload(cache_path: Path | None) -> dict | None:
    """Read and normalize Resolve phenotype payload from disk."""
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r') as f:
            return _normalize_resolve_pheno_payload(json.load(f))
    except Exception:
        return None


def _write_resolve_pheno_payload(payload: dict, cache_paths: list[Path | None]) -> dict:
    """Persist the Resolve phenotype payload to one or more cache files."""
    normalized_payload = _normalize_resolve_pheno_payload(payload)
    for cache_path in cache_paths:
        if cache_path is None:
            continue
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(normalized_payload, f, indent=2)
        except Exception:
            pass
    return normalized_payload


def _clear_resolve_pheno_widget_state() -> None:
    """Drop stale per-row Resolve phenotype widgets before rebuilding the panel."""
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if key_text.startswith(_RESOLVE_PHENO_WIDGET_PREFIXES):
            try:
                del st.session_state[key]
            except Exception:
                pass


def _apply_resolve_pheno_payload(payload: dict, cache_signature: str | None) -> None:
    """Replace the in-memory Resolve phenotype panel state from a payload."""
    normalized_payload = _normalize_resolve_pheno_payload(payload)
    resolve_state = cast(dict[str, object], st.session_state)
    _clear_resolve_pheno_widget_state()
    resolve_state['resolve_criteria_cols'] = list(normalized_payload['criteria_cols'])
    resolve_state['resolve_pheno_table'] = pd.DataFrame(normalized_payload['table'])
    resolve_state['resolve_loaded_pheno_cache_sig'] = cache_signature
    if 'resolve_loaded_pheno_hash' in resolve_state:
        del resolve_state['resolve_loaded_pheno_hash']


def _timestamp():
    """Return HH:MM:SS timestamp for logging."""
    return time.strftime("%H:%M:%S")


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab7', message, step='Resolve', level=level)


def _apply_flexible_normalization_safe(
    df: pd.DataFrame,
    marker_cols: list[str],
    group_cols: list[str] | None,
    suffix: str,
    apply_z1: bool,
    apply_z2: bool,
    apply_cdf: bool,
    apply_nlog: bool,
    norm_order: list[str] | None,
) -> pd.DataFrame:
    """Call flexible normalization while supporting older utility signatures."""
    try:
        return apply_flexible_normalization(
            df,
            marker_cols=marker_cols,
            group_cols=group_cols if group_cols else None,
            suffix=suffix,
            apply_z1=apply_z1,
            apply_z2=apply_z2,
            apply_cdf=apply_cdf,
            apply_nlog=apply_nlog,
            order=norm_order,
        )
    except TypeError:
        return apply_flexible_normalization(
            df,
            marker_cols=marker_cols,
            group_cols=group_cols if group_cols else None,
            suffix=suffix,
            apply_z1=apply_z1,
            apply_z2=apply_z2,
            apply_cdf=apply_cdf,
            apply_nlog=apply_nlog,
        )


def _otsu_threshold(values: np.ndarray) -> float:
    """Compute Otsu's threshold for a 1D array of positive values.

    Otsu's method finds the threshold that maximises the between-class
    variance of a bimodal distribution.  Here we operate on the subset of
    *non-zero* expression values so that the split separates "trace /
    background leakage" from "truly positive expression", rather than simply
    re-discovering the zero boundary.

    Returns 0.0 when the input is too small or degenerate.
    """
    values = values[~np.isnan(values)]
    positive = values[values > 0]
    if len(positive) < 20:
        return 0.0

    n_bins = min(256, max(10, len(positive) // 10))
    hist, bin_edges = np.histogram(positive, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = float(hist.sum())
    if total == 0:
        return 0.0

    best_thresh = 0.0
    best_var = -1.0
    cum_sum = 0.0
    cum_mean = 0.0
    total_mean = float((hist * bin_centers).sum()) / total

    for i in range(len(bin_centers)):
        cum_sum += hist[i]
        if cum_sum == 0:
            continue
        bg_w = cum_sum / total
        fg_w = 1.0 - bg_w
        if fg_w <= 0:
            break
        cum_mean += hist[i] * bin_centers[i]
        bg_mean = cum_mean / cum_sum
        fg_mean = (total_mean * total - cum_mean) / (total - cum_sum)
        var = bg_w * fg_w * (bg_mean - fg_mean) ** 2
        if var > best_var:
            best_var = var
            best_thresh = float(bin_centers[i])

    return best_thresh


def _adaptive_resolve_hierarchy(
    df: pd.DataFrame,
    binary_df: pd.DataFrame,
    phenotype_defs: dict,
    phenotype_hierarchy: dict,
    bin_suffix: str = "_bin_resolve",
    logger_fn=None,
) -> pd.DataFrame:
    """Adaptively threshold *distinguishing* markers between parent/child
    phenotypes when no external thresholds are available.

    Problem
    -------
    Without external thresholds every marker with expression > 0 is flagged
    as positive.  For daughter phenotypes (e.g. Treg) that differ from the
    parent (e.g. CD4) only by one extra marker (FOXP3), trace / background
    FOXP3 in ordinary CD4 cells causes false Treg assignments.

    Algorithm
    ---------
    For each parent -> child pair in the hierarchy:

    1. Identify the **distinguishing markers** = child_markers - parent_markers.
    2. Among all cells that satisfy *any* parent criterion (the candidate pool
       where the ambiguity exists), collect the non-zero expression values of
       each distinguishing marker.
    3. Apply **Otsu's method** to that distribution -- it finds the optimal
       split between "trace / leakage" and "truly high expression".
    4. Re-binarize the distinguishing marker globally using the adaptive
       threshold (instead of the naive > 0).

    This ensures that only cells with genuinely high distinguishing-marker
    expression are flagged as the daughter phenotype, while cells with mere
    trace expression stay classified as the parent.

    The method is fully automatic and applies to any parent/daughter pair
    in the hierarchy, not just CD4/Treg.
    """
    def _log(msg: str) -> None:
        if logger_fn:
            logger_fn(msg)

    updated = binary_df.copy()

    # Collect union of all markers per phenotype
    pheno_marker_sets: dict[str, set] = {}
    for pheno, criteria_sets in phenotype_defs.items():
        all_m: set[str] = set()
        for clause in criteria_sets:
            all_m.update(clause)
        pheno_marker_sets[pheno] = all_m

    processed_markers: set[str] = set()  # avoid re-thresholding a marker

    for child, parent in phenotype_hierarchy.items():
        if parent is None:
            continue

        child_markers = pheno_marker_sets.get(child, set())
        parent_markers = pheno_marker_sets.get(parent, set())

        # Distinguishing markers: present in child but NOT in parent
        distinguishing = child_markers - parent_markers
        if not distinguishing:
            continue

        # Build a mask for cells satisfying *any* parent criterion
        parent_candidate_mask = pd.Series(False, index=df.index)
        for and_clause in phenotype_defs.get(parent, []):
            clause_mask = pd.Series(True, index=df.index)
            for m in and_clause:
                bin_col = f"{m}{bin_suffix}"
                if bin_col in updated.columns:
                    clause_mask &= updated[bin_col].astype(bool)
                else:
                    clause_mask = pd.Series(False, index=df.index)
                    break
            parent_candidate_mask |= clause_mask

        n_candidates = int(parent_candidate_mask.sum())
        if n_candidates < 20:
            _log(f"  {parent}->{child}: only {n_candidates} parent candidates, "
                 f"skipping adaptive threshold")
            continue

        for marker in sorted(distinguishing):
            if marker in processed_markers:
                continue
            if marker not in df.columns:
                continue
            bin_col = f"{marker}{bin_suffix}"
            if bin_col not in updated.columns:
                continue

            # Expression of distinguishing marker among parent-candidate cells
            expr_values = df.loc[parent_candidate_mask, marker].dropna().values
            if len(expr_values) < 20:
                continue

            adaptive_thresh = _otsu_threshold(expr_values)
            if adaptive_thresh <= 0:
                _log(f"  {marker}: Otsu <= 0, keeping > 0 binarization")
                continue

            # Apply adaptive threshold globally
            original_pos = int(updated[bin_col].sum())
            updated[bin_col] = (df[marker] > adaptive_thresh).fillna(False).astype(int)
            new_pos = int(updated[bin_col].sum())

            _log(f"  {marker} ({parent}->{child}): adaptive threshold = "
                 f"{adaptive_thresh:.4f}  (positive: {original_pos:,} -> {new_pos:,})")
            processed_markers.add(marker)

    return updated


RESOLVE_GUIDE = """\
**What it does.** Assigns phenotypes from rules you write, without clustering. Each marker has a threshold, and a cell above it counts as positive for that marker. Each phenotype has a rule listing the markers a cell must be positive for, with optional alternatives, and a cell receives the phenotype whose rule it matches.

**Steps**
1. Choose the input table and the markers.
2. Set a threshold for each marker: import them from Tab 2's threshold tables, compute them automatically, or type them in.
3. Write the phenotype rules. **Must have** lists required markers; the **or** columns list alternatives.
4. Run, then review the heatmap and the number of cells per phenotype. Cells that match no rule are labelled **Unassigned**.
5. Save the labels into the table, then check example cells under **Visual Validation**.
"""


def render_resolve(project_root, global_input_path):
    """
    Resolve Clustering (Threshold-based Phenotyping) Subtab
    Professional interface matching Leiden/SOM patterns with notebook-synchronized parameters.
    """
    render_method_intro(
        project_root, "resolve", "Resolve Threshold Phenotyping",
        "Label cells with marker thresholds and rules you write. The same input always gives the same result.",
        RESOLVE_GUIDE, notebook="Step6C_phenotype_threshold.ipynb", log_label="Resolve Threshold Phenotyping",
    )
    
    # Initialize cache file for parameter persistence
    cache_dir = project_root / 'outputs/phenotyping/cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolve_cache = cache_dir / 'resolve_params.json'
    resolve_thresholds_cache = cache_dir / 'resolve_thresholds_table.csv'
    resolve_pheno_panel_cache = cache_dir / 'resolve_pheno_panel.json'
    
    # Load cached parameters
    cached_params = {}
    if resolve_cache.exists():
        try:
            with open(resolve_cache, 'r') as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}

    # --- Settings Export / Import ---
    render_settings_export_import(
        "Resolve (Threshold)",
        ["resolve_"],
        {
            "resolve_params.json": resolve_cache,
            "resolve_thresholds_table.csv": resolve_thresholds_cache,
            "resolve_pheno_panel.json": resolve_pheno_panel_cache,
        },
    )

    # Extract notebook config
    notebook_config = _extract_config_from_step6c()
    
    # --- FILE SELECTION ---
    st.markdown("### 1. Input File")
    
    use_global = st.checkbox("Use Global Input File", value=cached_params.get('use_global', True), key="resolve_use_global")
    full_input_path = global_input_path
    
    if not use_global:
        bc_dir = cached_params.get('bc_dir', str(project_root / 'outputs/batch_corrected'))
        bc_dir = st.text_input("Batch Corrected Directory", bc_dir, key="resolve_bc_dir")
        
        if os.path.exists(bc_dir):
            files = [f for f in os.listdir(bc_dir) if f.endswith('.csv')]
            if files:
                sel = st.selectbox("Input File", files, key="resolve_infile")
                full_input_path = os.path.join(bc_dir, sel)
    
    available_columns: list[str] = []
    resolve_pheno_cache = None
    if full_input_path:
        file_hash = hashlib.md5(str(full_input_path).encode()).hexdigest()
        resolve_pheno_cache = cache_dir / f"resolve_pheno_{file_hash}.json"

    def _get_active_resolve_pheno_cache() -> Path | None:
        if resolve_pheno_panel_cache.exists():
            return resolve_pheno_panel_cache
        if resolve_pheno_cache is not None and resolve_pheno_cache.exists():
            return resolve_pheno_cache
        return None

    if full_input_path and os.path.exists(full_input_path):
        st.caption(f"Active File: {os.path.basename(full_input_path)}")
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            available_columns = list(df_head.columns)
        except Exception:
            available_columns = []
    
    # --- SECTION 1: MARKER SELECTION ---
    st.divider()
    st.markdown("### 2. Marker Selection")
    st.caption("Select markers for threshold-based phenotyping. Auto-selected from Step6C notebook (MARKERS_BASE).")

    # Read available markers from input file
    all_marker_candidates = []
    if full_input_path and os.path.exists(full_input_path):
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            all_marker_candidates = get_marker_list(df_head)
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    available_marker_columns = list(all_marker_candidates)
    all_marker_candidates, marker_workspace_note = get_marker_workspace_panel(
        all_marker_candidates,
        full_input_path,
    )

    # AUTO-SELECT MARKERS FROM NOTEBOOK (Step6C_phenotype_threshold.ipynb)
    if full_input_path and os.path.exists(full_input_path):
        markers_base_from_notebook = _extract_markers_base_from_step6c()

        marker_workspace = st.session_state.get("tab7_marker_workspace", {})
        expected_markers = resolve_notebook_marker_defaults(
            markers_base_from_notebook,
            [],
            all_marker_candidates,
            str(marker_workspace.get("source_mode", "Raw markers")),
            str(marker_workspace.get("affix_position", "Suffix")),
            str(marker_workspace.get("correction_affix", "_corr")),
        )

    else:
        if 'resolve_confirmed_markers' not in st.session_state:
            st.session_state['resolve_confirmed_markers'] = []

    resolve_state = cast(dict[str, object], st.session_state)
    _, selected_markers, marker_workspace_note = render_method_marker_selection(
        key_prefix="resolve",
        method_name="Resolve",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers if full_input_path else [],
        saved_markers=cached_params.get("markers", []),
    )
    _sync_resolve_pheno_table_for_marker_selection(
        resolve_state,
        selected_markers,
        _get_active_resolve_pheno_cache(),
        resolve_pheno_panel_cache,
        resolve_pheno_cache,
    )

    if not selected_markers:
        st.warning("No markers selected. Please select markers above before running.")
    
    # --- SECTION 2: CONFIGURATION PANEL ---
    st.divider()
    st.markdown("### 3. Configuration")
    
    if 'resolve_config' not in st.session_state:
        # Migrate legacy 'apply_thresholds' boolean → threshold_mode string
        _legacy_apply = cached_params.get('apply_thresholds', notebook_config.get('apply_thresholds', True))
        _default_mode = cached_params.get(
            'threshold_mode',
            notebook_config.get('threshold_mode', 'Predefined (CSV)' if _legacy_apply else 'None (No Thresholds)'),
        )
        saved_transform = cached_params.get(
            'transformation', cached_params.get('tie_score_transform', 'ArcSinh')
        )
        saved_slots = cached_params.get(
            'normalization_slots', cached_params.get('norm_order', ['Z1'])
        )
        pipeline_config = _normalize_resolve_pipeline(
            saved_transform,
            cached_params.get('tie_score_cofactor', 5.0),
            saved_slots,
        )
        st.session_state['resolve_config'] = {
            'threshold_statistic': notebook_config.get('threshold_statistic', 'median'),
            'threshold_source': cached_params.get('threshold_source', 'Auto'),
            'threshold_mode': _default_mode,
            'z1_cutoff': float(cached_params.get('z1_cutoff', DEFAULT_RESOLVE_Z1_CUTOFF)),
            **pipeline_config,
            'enable_filtering': notebook_config.get('enable_filtering', False),
            'filter_column': notebook_config.get('filter_column', 'batch_id'),
            'filter_value': notebook_config.get('filter_value', ''),
            'batch_col': cached_params.get('batch_col', 'batch_id'),
            'cell_id_col': cached_params.get('cell_id_col', 'cell_id'),
            'threshold_marker_col': cached_params.get('threshold_marker_col', ''),
            'threshold_value_col': cached_params.get('threshold_value_col', ''),
            'polygon_prefix': cached_params.get('polygon_prefix', 'Polygon'),
            'output_col': cached_params.get('output_col', 'phenotype_threshold'),
            'unclassified_label': cached_params.get('unclassified_label', 'Others'),
            'cluster_col': 'resolve_cluster'
        }

    # Project hydration and older caches may supply the legacy checkbox model.
    # Rebuild the derived runtime flags on every render so the four dropdowns
    # remain the only source of truth.
    st.session_state['resolve_config'] = _migrate_resolve_pipeline_config(
        st.session_state['resolve_config']
    )
    
    with st.expander("Threshold & Normalization Configuration", expanded=True):
        with st.form("resolve_config_form", clear_on_submit=False):
            cfg_col1, cfg_col2 = st.columns([1, 1])
            
            with cfg_col1:
                st.markdown("**Threshold Settings**")
                # Resolve-specific: add "None (No Thresholds)" option
                # Otsu/Triangle/Li/Yen/... were computed on the per-cell means in the
                # unified CSV. A cell mean has already averaged noise over hundreds of
                # pixels, so its background mode is narrowed and shifted and the cut does
                # not mean what the same method means on pixels. Thresholds now come from
                # the images (Step 2). The Z1 score mode is a relative fallback for
                # projects that have no threshold tables yet.
                _resolve_mode_choices = ["Predefined (CSV)", RESOLVE_Z1_MODE, "None (No Thresholds)"]
                current_mode = st.session_state['resolve_config'].get('threshold_mode', 'Predefined (CSV)')
                mode_index = (
                    _resolve_mode_choices.index(current_mode)
                    if current_mode in _resolve_mode_choices
                    else _resolve_mode_choices.index("None (No Thresholds)")
                )
                threshold_mode = st.selectbox(
                    "Threshold Method",
                    _resolve_mode_choices,
                    index=mode_index,
                    help=(
                        "How Resolve decides whether a cell is positive for a marker.\n\n"
                        "**Predefined (CSV)** (recommended): uses the threshold tables made in "
                        "**Tab 2 > Detect Background**, one file per batch or ROI "
                        "(`<batch>.ome.tiff_-_Merged_Image.csv`). A cell is positive when its raw value is "
                        "above the marker's threshold.\n\n"
                        "**Z1 score (no threshold files)**: for when no threshold tables exist. Each marker is "
                        "ArcSinh-transformed and turned into a Z1 score within each batch (how many standard "
                        "deviations a cell sits above the batch average). A cell is positive when its Z1 score "
                        "is above the cutoff (default 1.5). Z2, CDF and NLog are switched off in this mode "
                        "because they would change what the cutoff means. This is a relative rule: markers "
                        "carried by a large share of cells, and markers with a few extremely bright cells, "
                        "tend to be under-called, so check the result with Visual Validation.\n\n"
                        "**None**: any value above 0 counts as positive. Only suitable when the background "
                        "was already removed; on raw data about 98% of cells are above zero for a typical "
                        "marker, so nearly every rule matches nearly every cell."
                    ),
                    key="resolve_threshold_mode"
                )

                _is_z1_mode = (threshold_mode == RESOLVE_Z1_MODE)
                z1_cutoff = st.number_input(
                    "Z1 cutoff (Z1 score mode only)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state['resolve_config'].get('z1_cutoff', DEFAULT_RESOLVE_Z1_CUTOFF)),
                    step=0.1,
                    key="resolve_z1_cutoff",
                    disabled=not _is_z1_mode,
                    help=(
                        "A marker counts as positive when the cell's Z1 score is above this value. "
                        "1.5 means 1.5 standard deviations above the batch average. Raise it to call "
                        "fewer cells positive, lower it to call more."
                    ),
                )

                # Threshold source/statistic are only relevant for Predefined CSV mode.
                _is_csv_mode = (threshold_mode == "Predefined (CSV)")
                threshold_source_options = ["Auto", "Polygon values", "Final scalar"]
                current_source = st.session_state['resolve_config'].get('threshold_source', 'Auto')
                source_index = (
                    threshold_source_options.index(current_source)
                    if current_source in threshold_source_options else 0
                )
                threshold_source = st.selectbox(
                    "Threshold Source (Predefined CSV only)",
                    threshold_source_options,
                    index=source_index,
                    format_func=lambda value: {
                        "Auto": "Auto — override > polygons > final scalar",
                        "Polygon values": "Polygon values — aggregate Polygon_*",
                        "Final scalar": "Final scalar — use threshold_final/threshold directly",
                    }[value],
                    help=(
                        "**Auto** first honors an explicitly configured value column, then a non-null manual override, "
                        "then aggregates `Polygon_*`, and finally falls back to `threshold_final`/`threshold`. "
                        "**Polygon values** deliberately ignores all scalar columns. **Final scalar** uses a direct "
                        "value and does not apply the statistic below."
                    ),
                    key="resolve_threshold_source",
                    disabled=not _is_csv_mode,
                )

                threshold_options = [
                    "median", "mean", "min", "percentile_05", "percentile_25", "percentile_75"
                ]
                current_stat = st.session_state['resolve_config']['threshold_statistic']
                threshold_index = threshold_options.index(current_stat) if current_stat in threshold_options else 0
                threshold_stat = st.selectbox(
                    "Threshold Statistic (Predefined CSV only)",
                    threshold_options,
                    index=threshold_index,
                    format_func=lambda value: {
                        "median": "Median",
                        "mean": "Mean",
                        "min": "Minimum",
                        "percentile_05": "5th percentile (p05)",
                        "percentile_25": "25th percentile (p25)",
                        "percentile_75": "75th percentile (p75)",
                    }[value],
                    help=(
                        "Only used for Predefined (CSV). When a threshold file contains multiple values for the same marker, they are reduced to one gate. "
                        "`median` uses the middle polygon, `mean` is more sensitive to outliers, and `min` uses the lowest polygon. "
                        "p05 is a near-minimum gate, p25 is a lower-quartile gate, and p75 is stricter. "
                        "Percentiles refer to background polygons, not to percentages of cells. With one polygon, "
                        "every statistic intentionally returns that one value."
                    ),
                    key="resolve_threshold_stat",
                    disabled=(not _is_csv_mode or threshold_source == "Final scalar"),
                )
            
            with cfg_col2:
                st.markdown("**Transformation**")
                current_transform = st.session_state['resolve_config'].get(
                    'transformation', 'ArcSinh'
                )
                transformation = st.selectbox(
                    "Transformation",
                    RESOLVE_TRANSFORM_OPTIONS,
                    index=(
                        RESOLVE_TRANSFORM_OPTIONS.index(current_transform)
                        if current_transform in RESOLVE_TRANSFORM_OPTIONS else 1
                    ),
                    help=(
                        "Changes the continuous score used only when multiple phenotypes match. "
                        "ArcSinh uses the selected cofactor. Log uses log1p so zero values are valid. "
                        "Marker-positive gates remain raw > threshold. In Z1 score mode this is fixed to ArcSinh."
                    ),
                    key="resolve_transform_choice",
                    disabled=_is_z1_mode,
                )
                if transformation == "ArcSinh" or _is_z1_mode:
                    arcsinh_cofactor = st.number_input(
                        "ArcSinh cofactor",
                        min_value=0.001,
                        value=float(st.session_state['resolve_config'].get(
                            'tie_score_cofactor', 5.0
                        )),
                        step=1.0,
                        key="resolve_arcsinh_cofactor",
                    )
                else:
                    arcsinh_cofactor = float(
                        st.session_state['resolve_config'].get('tie_score_cofactor', 5.0)
                    )

                st.markdown("**Normalization order**")
                current_slots = list(st.session_state['resolve_config'].get(
                    'normalization_slots', ['Z1', 'None', 'None', 'None']
                ))
                current_slots = (current_slots + ['None'] * 4)[:4]
                norm_order_help = (
                    "Choose the order. Use None for unused positions. "
                    "NLog must come after CDF."
                )
                order_columns = st.columns(4)
                order_raw: list[str] = []
                for slot_index, column in enumerate(order_columns):
                    slot_value = current_slots[slot_index]
                    if slot_value not in RESOLVE_NORMALIZATION_OPTIONS:
                        slot_value = 'None'
                    order_raw.append(column.selectbox(
                        str(slot_index + 1),
                        RESOLVE_NORMALIZATION_OPTIONS,
                        index=RESOLVE_NORMALIZATION_OPTIONS.index(slot_value),
                        key=f"resolve_normalization_{slot_index + 1}",
                        help=norm_order_help,
                        disabled=_is_z1_mode,
                    ))

                config_errors: list[str] = []
                pipeline_config = None
                try:
                    pipeline_config = _normalize_resolve_pipeline(
                        transformation,
                        arcsinh_cofactor,
                        order_raw,
                    )
                    config_errors.extend(_resolve_pipeline_errors(pipeline_config))
                except ValueError as exc:
                    config_errors.append(str(exc))

                try:
                    pipeline_preview = _resolve_pipeline_preview(
                        transformation,
                        order_raw,
                    )
                except ValueError:
                    preview_parts = (
                        ([] if transformation == 'None' else [transformation])
                        + [step for step in order_raw if step != 'None']
                    )
                    pipeline_preview = " → ".join(preview_parts) or "None"
                if _is_z1_mode:
                    # The saved slots are kept for when another mode is chosen again;
                    # the Z1 run ignores them.
                    pipeline_preview = f"ArcSinh -> Z1 per batch (positive if Z1 > {z1_cutoff:g})"
                    config_errors = []
                st.code(pipeline_preview, language="text")
                for config_error in config_errors:
                    st.warning(config_error)

            out_col = st.text_input(
                "Phenotype Output Column",
                value=st.session_state['resolve_config'].get('output_col', 'phenotype_threshold'),
                help=get_output_column_help_text("Resolve phenotype labels"),
                key="resolve_output_col"
            )

            unclassified_label = st.text_input(
                "Unclassified Label",
                value=st.session_state['resolve_config'].get('unclassified_label', 'Others'),
                help=(
                    "Label written when a cell matches no phenotype definition after thresholding and tie-break resolution. "
                    "This value is used in the output column, statistics, and exports."
                ),
                key="resolve_unclassified_label"
            )
            
            submitted_cfg = st.form_submit_button("Save Configuration", type="primary")
            if submitted_cfg and (config_errors or pipeline_config is None):
                st.error("Configuration was not saved. Fix the settings above.")
            elif submitted_cfg:
                st.session_state['resolve_config'].update({
                    'threshold_statistic': threshold_stat,
                    'threshold_source': threshold_source,
                    'threshold_mode': threshold_mode,
                    'z1_cutoff': float(z1_cutoff),
                    **pipeline_config,
                    'output_col': out_col,
                    'unclassified_label': unclassified_label
                })
                params_to_save = {
                    'batch_col': st.session_state['resolve_config'].get('batch_col', 'batch_id'),
                    'cell_id_col': st.session_state['resolve_config'].get('cell_id_col', 'cell_id'),
                    'threshold_marker_col': st.session_state['resolve_config'].get('threshold_marker_col', ''),
                    'threshold_value_col': st.session_state['resolve_config'].get('threshold_value_col', ''),
                    'threshold_source': st.session_state['resolve_config'].get('threshold_source', 'Auto'),
                    'polygon_prefix': st.session_state['resolve_config'].get('polygon_prefix', 'Polygon'),
                    'transformation': st.session_state['resolve_config'].get(
                        'transformation', 'ArcSinh'
                    ),
                    'tie_score_transform': st.session_state['resolve_config'].get(
                        'tie_score_transform', 'identity_residual'
                    ),
                    'tie_score_cofactor': st.session_state['resolve_config'].get(
                        'tie_score_cofactor', 5.0
                    ),
                    'output_col': st.session_state['resolve_config'].get('output_col', 'phenotype_threshold'),
                    'normalization_slots': st.session_state['resolve_config'].get(
                        'normalization_slots', ['Z1', 'None', 'None', 'None']
                    ),
                    'norm_order': st.session_state['resolve_config'].get('norm_order', ['Z1']),
                    'norm_z1': st.session_state['resolve_config'].get('norm_z1', True),
                    'norm_z2': st.session_state['resolve_config'].get('norm_z2', False),
                    'norm_cdf': st.session_state['resolve_config'].get('norm_cdf', False),
                    'norm_nlog': st.session_state['resolve_config'].get('norm_nlog', False),
                    'norm_sinh': False,
                    'unclassified_label': st.session_state['resolve_config'].get('unclassified_label', 'Others'),
                    'threshold_mode': st.session_state['resolve_config'].get('threshold_mode', 'Predefined (CSV)'),
                    'z1_cutoff': st.session_state['resolve_config'].get('z1_cutoff', DEFAULT_RESOLVE_Z1_CUTOFF),
                }
                try:
                    cached_params.update(params_to_save)
                    with open(resolve_cache, 'w') as f:
                        json.dump(cached_params, f, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                except Exception:
                    pass

    # --- Data Filtering ---
    resolve_filter_settings = render_data_filter_ui(
        available_columns=available_columns,
        key_prefix="resolve",
        cached_params=st.session_state.get('resolve_config', {}),
        full_input_path=full_input_path,
        show_cell_id=True,
    )
    # Store in config for pipeline use
    st.session_state['resolve_config']['enable_filter'] = resolve_filter_settings.get('enable_filter', False)
    st.session_state['resolve_config']['filter_col'] = resolve_filter_settings.get('filter_col')
    st.session_state['resolve_config']['filter_val'] = resolve_filter_settings.get('filter_val')
    st.session_state['resolve_config']['cell_id_col'] = normalize_optional_column_name(resolve_filter_settings.get('cell_id_col'))

    st.markdown("#### Threshold Configuration & Table")

    # Determine current threshold mode for conditional UI
    _resolve_mode_map = {**_THRESHOLD_MODE_TO_KEY, "None (No Thresholds)": "none", RESOLVE_Z1_MODE: "z1"}
    _current_mode = st.session_state['resolve_config'].get('threshold_mode', 'Predefined (CSV)')
    _mode_key = _resolve_mode_map.get(_current_mode, "predefined")
    _is_csv = (_mode_key == "predefined")
    _is_none = (_mode_key == "none")
    _is_z1 = (_mode_key == "z1")
    _is_auto = not (_is_csv or _is_none or _is_z1)

    # Batch column is always needed (for pipeline)
    if available_columns:
        default_batch = 'batch_id' if 'batch_id' in available_columns else available_columns[0]
        batch_options = [""] + available_columns
        batch_col = st.selectbox(
            "Batch Column",
            batch_options,
            index=batch_options.index(default_batch) if default_batch in batch_options else 0,
            format_func=lambda value: value or "(None)",
            key="resolve_batch_col",
            help=get_batch_id_column_help_text(),
        )
    else:
        batch_col = st.text_input(
            "Batch Column",
            value=st.session_state['resolve_config'].get('batch_col') or '',
            key="resolve_batch_col",
            help=get_batch_id_column_help_text(),
        )

    st.session_state['resolve_config']['batch_col'] = normalize_optional_column_name(batch_col)

    if _is_none:
        st.info(
            "**No thresholds** — cells are classified as positive when marker expression > 0. "
            "Use this for data that is already background-subtracted or pre-binarized."
        )
    if _is_z1:
        _z1_cutoff = float(st.session_state['resolve_config'].get('z1_cutoff', DEFAULT_RESOLVE_Z1_CUTOFF))
        st.info(
            f"**Z1 score mode** — a marker is positive when its Z1 score is above {_z1_cutoff:g} "
            "(ArcSinh, then a z-score within each batch). No threshold files are needed. "
            "Use raw marker columns: background-subtracted or batch-corrected values move the "
            "batch average and change which cells count as positive."
        )
        _affix = str(st.session_state.get("tab7_marker_workspace", {}).get("correction_affix", "_corr") or "")
        _corrected = [m for m in selected_markers if _affix and (m.endswith(_affix) or m.startswith(_affix))]
        if _corrected:
            st.warning(
                f"{len(_corrected)} selected marker(s) look batch-corrected ({', '.join(_corrected[:5])}). "
                "Z1 score mode is meant for raw markers; switch **Values for this method** to raw markers."
            )

    # --- CSV-specific inputs ---
    threshold_draft_refreshed = False
    if _is_csv:
        default_thresh_dir = _preferred_threshold_directory(project_root)
        if not str(st.session_state.get('threshold_dir') or '').strip():
            st.session_state['threshold_dir'] = str(default_thresh_dir)

        thresh_dir = st.text_input(
            "Project Thresholds Directory",
            key="threshold_dir",
            help=(
                "Project-level folder containing Predefined threshold CSVs for each batch or ROI. "
                "Tab 2 writes to this same directory, and Workspace stores it in the project JSON. "
                "Filenames are matched to batch IDs with a tolerant stem matcher, so QuPath-style "
                "names usually work without manual renaming."
            )
        )
        if cached_params.get('threshold_dir') != thresh_dir:
            try:
                cached_params['threshold_dir'] = thresh_dir
                with open(resolve_cache, 'w') as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass

        tcfg1, tcfg2, tcfg3 = st.columns([1, 1, 1])
        with tcfg1:
            marker_col_name = st.text_input(
                "Threshold Marker Column (optional)",
                value=st.session_state['resolve_config'].get('threshold_marker_col', ''),
                key="resolve_threshold_marker_col",
                help=(
                    "Optional override for the marker-name column in threshold CSVs. Leave blank to auto-detect names such as `marker` or `markers`."
                ),
            )
        with tcfg2:
            value_col_name = st.text_input(
                "Threshold Value Column (optional)",
                value=st.session_state['resolve_config'].get('threshold_value_col', ''),
                key="resolve_threshold_value_col",
                help=(
                    "Optional override for the numeric threshold column in threshold CSVs. Leave blank to auto-detect columns such as `threshold`, `value`, `cutoff`, or `gate`."
                ),
            )
        with tcfg3:
            polygon_prefix = st.text_input(
                "Polygon Column Prefix",
                value=st.session_state['resolve_config'].get('polygon_prefix', 'Polygon'),
                key="resolve_polygon_prefix",
                help=(
                    "Used when a threshold CSV stores one value per polygon column instead of a single threshold column. "
                    "Columns whose names start with this prefix are aggregated into one threshold per marker."
                ),
            )

        st.session_state['resolve_config']['threshold_marker_col'] = marker_col_name
        st.session_state['resolve_config']['threshold_value_col'] = value_col_name
        st.session_state['resolve_config']['polygon_prefix'] = polygon_prefix

        tcol1, tcol2, tcol3 = st.columns([1, 1, 2])
        load_thresh = tcol1.button("Load Thresholds", key="resolve_load_thresholds_btn")
        add_row = tcol2.button("Add Row", key="resolve_add_threshold_row_btn")
        tcol3.caption("Review and edit threshold values before running.")

        # Import from Threshold tab CSV
        with st.expander("Import from Threshold Tab CSV", expanded=False):
            st.caption("Import thresholds from a CSV file created in the Threshold gating tool.")
            import_file = st.file_uploader(
                "Upload Threshold CSV",
                type=["csv"],
                key="resolve_import_thresh_csv",
            )
            if import_file is not None:
                try:
                    import_df = pd.read_csv(import_file)
                    import_df.columns = [c.lower().strip() for c in import_df.columns]
                    if "marker" not in import_df.columns or "threshold" not in import_df.columns:
                        st.error("CSV must have 'marker' and 'threshold' columns.")
                    else:
                        has_batch_import = "batch_id" in import_df.columns
                        # Build threshold table from import
                        rows = []
                        for _, row in import_df.iterrows():
                            marker = str(row["marker"]).strip()
                            # Normalize marker name for matching
                            base_marker = _normalize_marker_name(marker)
                            # Find matching marker in selected markers
                            matched_marker = None
                            for sel_m in selected_markers:
                                if _normalize_marker_name(sel_m) == base_marker or sel_m == marker:
                                    matched_marker = sel_m
                                    break
                            if matched_marker is None:
                                continue
                            batch_id = str(row["batch_id"]).strip() if has_batch_import else "universal"
                            threshold_val = float(row["threshold"])
                            rows.append({
                                "batch_id": batch_id,
                                "Threshold_File": "imported",
                                matched_marker: threshold_val,
                            })
                        if rows:
                            # Merge rows by batch_id
                            merged = {}
                            for r in rows:
                                bid = r["batch_id"]
                                if bid not in merged:
                                    merged[bid] = {"batch_id": bid, "Threshold_File": "imported"}
                                for k, v in r.items():
                                    if k not in ["batch_id", "Threshold_File"]:
                                        merged[bid][k] = v
                            imported_df = pd.DataFrame(list(merged.values()))
                            st.session_state['resolve_threshold_table_draft'] = imported_df
                            threshold_draft_refreshed = True
                            st.success(f"Imported {len(merged)} batch(es) from Threshold CSV.")
                            # Warn about corrected markers
                            corr_markers = [m for m in selected_markers if m.endswith("_corr")]
                            import_corr = [str(row["marker"]) for _, row in import_df.iterrows() if str(row["marker"]).endswith("_corr")]
                            if import_corr and not corr_markers:
                                st.warning("Imported CSV contains corrected markers (_corr) but selected markers are raw. Check marker alignment.")
                            elif corr_markers and not import_corr:
                                st.warning("Selected markers are corrected (_corr) but imported CSV has raw markers. Check marker alignment.")
                        else:
                            st.warning("No matching markers found in import file.")
                except Exception as e:
                    st.error(f"Failed to import: {e}")

        if load_thresh:
            batch_values = []
            if full_input_path and os.path.exists(full_input_path) and batch_col:
                try:
                    batch_values = _dataio.load(full_input_path, columns=[batch_col])[batch_col].dropna().unique().tolist()
                except Exception:
                    batch_values = []

            st.session_state['resolve_threshold_table_draft'] = _load_thresholds_table(
                thresh_dir,
                selected_markers,
                batch_values,
                marker_col_name or None,
                value_col_name or None,
                polygon_prefix or None,
                st.session_state['resolve_config'].get('threshold_statistic', 'median'),
                st.session_state['resolve_config'].get('threshold_source', 'Auto'),
            )
            threshold_draft_refreshed = True
    else:
        # Provide defaults so variables exist for the save handler
        thresh_dir = st.session_state.get(
            'threshold_dir', str(_preferred_threshold_directory(project_root))
        )
        marker_col_name = st.session_state['resolve_config'].get('threshold_marker_col', '')
        value_col_name = st.session_state['resolve_config'].get('threshold_value_col', '')
        polygon_prefix = st.session_state['resolve_config'].get('polygon_prefix', 'Polygon')
        add_row = False

    # --- Auto-threshold Compute button ---
    if _is_auto:
        if st.button(f"Compute {_current_mode} Thresholds", type="primary", key="resolve_compute_auto_thresh"):
            if not full_input_path or not os.path.exists(full_input_path):
                st.error("Please select a valid input file first.")
            elif not selected_markers:
                st.error("Please select markers first.")
            else:
                with st.spinner(f"Computing {_current_mode} thresholds..."):
                    try:
                        _df_for_thresh = _dataio_load(full_input_path)
                        # Apply active filter
                        _filt = {
                            'enable_filter': st.session_state['resolve_config'].get('enable_filter', False),
                            'filter_col': st.session_state['resolve_config'].get('filter_col'),
                            'filter_val': st.session_state['resolve_config'].get('filter_val'),
                        }
                        _df_for_thresh = apply_data_filter(_df_for_thresh, _filt)
                        _auto_df = compute_auto_thresholds(
                            _df_for_thresh,
                            selected_markers,
                            _mode_key,
                            batch_col=batch_col if batch_col in _df_for_thresh.columns else None,
                        )
                        st.session_state['resolve_threshold_table_draft'] = _auto_df
                        threshold_draft_refreshed = True
                        st.success(f"Computed {_current_mode} thresholds for {len(_auto_df)} batch(es)")
                    except Exception as e:
                        st.error(f"Auto-threshold computation failed: {e}")
        add_row = st.button("Add Row", key="resolve_add_threshold_row_auto_btn")

    # --- Import Universal Gating Thresholds (from Threshold Gating tab) ---
    # This section is available in ALL threshold modes.
    if not _is_none:
        with st.expander("Import Universal Gating Thresholds (from Threshold Tab)", expanded=False):
            st.caption(
                "Import universal (no-batch) thresholds exported by the **Threshold Gating** tool. "
                "The same threshold value will be applied to **every batch** in your data."
            )
            _default_gating_csv = str(project_root / "outputs" / "phenotyping" / "thresholds_gating.csv")
            gating_csv_path = st.text_input(
                "Gating Thresholds CSV Path",
                value=_default_gating_csv,
                key="resolve_gating_csv_path",
                help=(
                    "Path to a Threshold Gating export with `marker` and `threshold` columns. Imported thresholds are treated as universal values "
                    "and broadcast to each batch in the current dataset, or used once globally when no batch column is selected."
                ),
            )
            if st.button("Import Universal Thresholds", key="resolve_import_gating_btn", type="primary"):
                if not gating_csv_path or not os.path.exists(gating_csv_path):
                    st.error(f"File not found: {gating_csv_path}")
                elif not selected_markers:
                    st.error("Please select markers first.")
                else:
                    # Read batch values from data
                    _batch_vals = []
                    if full_input_path and os.path.exists(full_input_path) and batch_col:
                        try:
                            _batch_vals = _dataio.load(
                                full_input_path, columns=[batch_col]
                            )[batch_col].dropna().unique().tolist()
                        except Exception:
                            pass
                    imported_df = load_universal_gating_csv(
                        gating_csv_path, selected_markers, _batch_vals or None
                    )
                    if imported_df.empty:
                        st.warning(
                            "No matching markers found. Ensure the CSV has 'marker' and "
                            "'threshold' columns and marker names match your selection."
                        )
                    else:
                        st.session_state['resolve_threshold_table_draft'] = imported_df
                        threshold_draft_refreshed = True
                        n_markers = sum(
                            1 for m in selected_markers
                            if m in imported_df.columns and imported_df[m].notna().any()
                        )
                        st.success(
                            f"Imported universal thresholds for **{n_markers}** markers → "
                            f"**{len(imported_df)}** batch row(s). Review in the table below."
                        )

    current_threshold_fingerprint = _resolve_threshold_fingerprint(
        thresh_dir, st.session_state['resolve_config'], selected_markers
    )
    if threshold_draft_refreshed:
        st.session_state['resolve_threshold_table_draft_fingerprint'] = (
            current_threshold_fingerprint
        )

    # --- Initialize draft table ---
    if 'resolve_threshold_table_draft' not in st.session_state:
        if resolve_thresholds_cache.exists():
            try:
                cached_threshold_table = pd.read_csv(resolve_thresholds_cache)
                st.session_state['resolve_threshold_table_draft'] = cached_threshold_table
                cached_table_fingerprint = str(
                    cached_params.get('threshold_table_fingerprint') or ''
                ).strip()
                if cached_table_fingerprint:
                    st.session_state['resolve_threshold_table_draft_fingerprint'] = (
                        cached_table_fingerprint
                    )
                    st.session_state['resolve_threshold_table'] = cached_threshold_table.copy()
                    st.session_state['resolve_threshold_table_fingerprint'] = (
                        cached_table_fingerprint
                    )
            except Exception:
                st.session_state['resolve_threshold_table_draft'] = pd.DataFrame()
        else:
            st.session_state['resolve_threshold_table_draft'] = pd.DataFrame()

    if add_row:
        new_row = {
            "batch_id": "",
            "Threshold_File": "manual",
            "Threshold_Source": "Manual table entry",
            "Threshold_Aggregation": "Direct scalar",
            "Polygon_Count": "0",
        }
        for m in selected_markers:
            new_row[m] = np.nan
        st.session_state['resolve_threshold_table_draft'] = pd.concat(
            [st.session_state['resolve_threshold_table_draft'], pd.DataFrame([new_row])],
            ignore_index=True
        )

    # --- Editable threshold table (shown for CSV and Auto modes, hidden for None) ---
    if not _is_none and selected_markers:
        thresh_table = st.session_state['resolve_threshold_table_draft']
        if not thresh_table.empty:
            # Older cached/imported tables pre-date source provenance.  Keep
            # them usable while making the unknown aggregation explicit.
            legacy_metadata_defaults = {
                "Threshold_Source": "Saved/imported scalar (source unknown)",
                "Threshold_Aggregation": "Already aggregated",
                "Polygon_Count": "Unknown",
            }
            missing_metadata = False
            for metadata_column, default_value in legacy_metadata_defaults.items():
                if metadata_column not in thresh_table.columns:
                    thresh_table[metadata_column] = default_value
                    missing_metadata = True
            if missing_metadata:
                st.session_state['resolve_threshold_table_draft'] = thresh_table
            column_config = {
                "batch_id": st.column_config.TextColumn("batch_id"),
                "Threshold_File": st.column_config.TextColumn("Threshold_File"),
                "Threshold_Source": st.column_config.TextColumn(
                    "Actual source",
                    help="The source that supplied the gate; Mixed means marker sources differed.",
                ),
                "Threshold_Aggregation": st.column_config.TextColumn(
                    "Aggregation",
                    help="Direct for scalar/override values; otherwise the statistic applied to Polygon_* values.",
                ),
                "Polygon_Count": st.column_config.TextColumn(
                    "Available polygon n",
                    help="Number, or range of numbers, of usable polygon values per marker in the source file.",
                ),
            }
            for m in selected_markers:
                column_config[m] = st.column_config.NumberColumn(m, format="%.4f")

            edited_thresh = st.data_editor(
                thresh_table,
                column_config=column_config,
                hide_index=True,
                num_rows="dynamic",
                key="resolve_threshold_editor",
                disabled=[
                    "Threshold_File", "Threshold_Source", "Threshold_Aggregation", "Polygon_Count"
                ],
            )
            st.session_state['resolve_threshold_table_draft'] = edited_thresh

            if "Polygon_Count" in edited_thresh.columns:
                polygon_count_text = edited_thresh["Polygon_Count"].astype(str)
                polygon_count_min = pd.to_numeric(
                    polygon_count_text.str.extract(r"^(\d+)")[0],
                    errors="coerce",
                ).dropna()
                polygon_count_max = pd.to_numeric(
                    polygon_count_text.str.extract(r"(?:–|-)(\d+)$")[0],
                    errors="coerce",
                ).combine_first(polygon_count_min)
                source_series = edited_thresh.get(
                    "Threshold_Source", pd.Series("", index=edited_thresh.index)
                ).astype(str)
                polygon_rows = source_series.str.contains("Polygon values", case=False, na=False)
                polygon_count_min_used = polygon_count_min.loc[
                    polygon_count_min.index.intersection(edited_thresh.index[polygon_rows])
                ]
                polygon_count_max_used = polygon_count_max.loc[
                    polygon_count_max.index.intersection(edited_thresh.index[polygon_rows])
                ]
                if not polygon_count_max_used.empty and (polygon_count_max_used <= 1).all():
                    st.info(
                        "Each polygon-sourced marker has one usable polygon value. The selected statistic is a "
                        "mathematical no-op; the value is still loaded directly."
                    )
                if (
                    st.session_state['resolve_config'].get('threshold_statistic') == 'percentile_05'
                    and not polygon_count_min_used.empty
                    and (polygon_count_min_used < 20).any()
                ):
                    st.warning(
                        "With fewer than 20 polygons, p05 is close to the minimum."
                    )
        else:
            if _is_csv:
                st.info("No threshold table loaded yet. Click 'Load Thresholds' to populate.")
            else:
                st.info(f"Click 'Compute {_current_mode} Thresholds' above to auto-generate the table.")
    elif not _is_none:
        st.info("Select markers first to enable threshold table.")

    if st.button("Save Threshold Table", type="primary", key="resolve_save_threshold_table"):
        draft_fingerprint = str(
            st.session_state.get('resolve_threshold_table_draft_fingerprint') or ''
        ).strip()
        if draft_fingerprint and draft_fingerprint != current_threshold_fingerprint:
            st.error(
                "Threshold source settings changed after this table was loaded. Reload thresholds "
                "with the current source/statistic settings before saving."
            )
        else:
            st.session_state['resolve_threshold_table'] = (
                st.session_state['resolve_threshold_table_draft'].copy()
            )
            st.session_state['resolve_threshold_table_fingerprint'] = (
                current_threshold_fingerprint
            )
            st.session_state['resolve_threshold_table_draft_fingerprint'] = (
                current_threshold_fingerprint
            )
            try:
                st.session_state['resolve_threshold_table'].to_csv(
                    resolve_thresholds_cache, index=False
                )
            except Exception:
                pass
            try:
                cached_params.update({
                    'batch_col': st.session_state['resolve_config'].get('batch_col'),
                    'cell_id_col': st.session_state['resolve_config'].get('cell_id_col'),
                    'threshold_marker_col': marker_col_name,
                    'threshold_value_col': value_col_name,
                    'threshold_source': st.session_state['resolve_config'].get('threshold_source', 'Auto'),
                    'polygon_prefix': polygon_prefix,
                    'threshold_dir': thresh_dir,
                    'threshold_table_fingerprint': current_threshold_fingerprint,
                })
                with open(resolve_cache, 'w') as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass
            st.success("Threshold table saved for the current source settings.")
    
    # --- SECTION 3: PHENOTYPE DEFINITIONS TABLE ---
    st.divider()
    st.markdown("### 4. Manage Phenotypes")
    st.caption(
        "Define phenotypes using selected markers. Click 'Save Phenotypes' to persist changes.\n\n"
        "**Must not have** does not define a phenotype - it overturns one. A cell keeps its call "
        "unless a forbidden marker is *stronger* than the markers that justified it, at which point "
        "the cell moves to the phenotype that forbidden marker belongs to. Faint signal from a "
        "touching neighbour therefore does not disqualify anything."
    )
    
    selected_markers = resolve_state.get('resolve_confirmed_markers', [])
    if not isinstance(selected_markers, list):
        selected_markers = []

    active_resolve_pheno_cache = _get_active_resolve_pheno_cache()
    active_pheno_cache_sig = _resolve_pheno_cache_signature(active_resolve_pheno_cache)
    if active_pheno_cache_sig != resolve_state.get('resolve_loaded_pheno_cache_sig'):
        pheno_payload = _read_resolve_pheno_payload(active_resolve_pheno_cache)
        if pheno_payload is not None:
            _apply_resolve_pheno_payload(pheno_payload, active_pheno_cache_sig)
        else:
            resolve_state['resolve_loaded_pheno_cache_sig'] = active_pheno_cache_sig

    if 'resolve_criteria_cols' not in resolve_state:
        resolve_state['resolve_criteria_cols'] = list(_RESOLVE_DEFAULT_CRITERIA_COLS)

    criteria_cols = resolve_state['resolve_criteria_cols']
    if not isinstance(criteria_cols, list):
        criteria_cols = list(_RESOLVE_DEFAULT_CRITERIA_COLS)
        resolve_state['resolve_criteria_cols'] = criteria_cols

    if 'resolve_pheno_table' not in resolve_state:
        default_payload = _build_default_resolve_pheno_payload()
        resolve_state['resolve_pheno_table'] = pd.DataFrame(default_payload['table'])

    resolve_state['resolve_pheno_table'] = _coerce_phenotype_table(
        cast(pd.DataFrame, resolve_state['resolve_pheno_table']),
        criteria_cols
    )

    st.markdown("#### Phenotype Definition Table")
    st.caption(
        "**Must have**: markers in one column are all required (AND). Add more columns for alternative "
        "ways to qualify (OR).  \n"
        "**Supportive**: any one of these decides between two phenotypes that both qualify. "
        "**Measured on this dataset it made results worse** (macro-F1 0.650 to 0.367), because the "
        "obvious supportive markers - CD45, CD3e, HLA-DR - are shared by most immune phenotypes and "
        "so cannot separate them. Leave it empty unless a marker is unique to one phenotype.  \n"
        "**Must NOT have**: any one of these overturns the call, but only when it is stronger than the "
        "markers that justified it - so faint spillover from a neighbour is ignored."
    )

    pheno_ctrl_col1, pheno_ctrl_col2 = st.columns([1, 3])
    if pheno_ctrl_col1.button("Add Criteria Column", key="resolve_add_criteria_col_btn"):
        next_idx = len(criteria_cols) + 1
        criteria_cols.append(f"Criteria {next_idx}")
        st.session_state['resolve_criteria_cols'] = criteria_cols
        st.session_state['resolve_pheno_table'] = _coerce_phenotype_table(
            st.session_state['resolve_pheno_table'],
            criteria_cols
        )
        _clear_resolve_pheno_widget_state()
        st.rerun()

    pheno_ctrl_col2.caption("Use the table below to define phenotype criteria. Save to apply changes.")

    remove_col = None
    if len(criteria_cols) > 1:
        remove_row = st.columns([1.2] + [1] * len(criteria_cols) + [1, 1])
        remove_row[0].caption("remove a column:")
        for i, col_name in enumerate(criteria_cols):
            with remove_row[i + 1]:
                if st.button(f"Remove {'Must have' if i == 0 else f'or {i}'}",
                             key=f"resolve_remove_{col_name}_btn", width="stretch"):
                    remove_col = col_name

    if remove_col:
        criteria_cols = [c for c in criteria_cols if c != remove_col]
        st.session_state['resolve_criteria_cols'] = criteria_cols
        st.session_state['resolve_pheno_table'] = _coerce_phenotype_table(
            st.session_state['resolve_pheno_table'],
            criteria_cols
        )
        _clear_resolve_pheno_widget_state()
        st.rerun()

    # --- Add Custom Phenotype UI ---
    st.caption("Need a phenotype not in the list? Add it here:")
    render_add_custom_phenotype_ui(key_prefix="resolve")

    with st.expander("Write the rules as text instead", expanded=False):
        st.markdown(
            "One phenotype per line. Markers inside one **[bracket]** must all be positive (**AND**); "
            "separate bracket groups are alternatives (**OR**). **SUPPORT** and **NOT** are optional "
            "lists. Quotes around marker names are optional. The older AND/OR text format still works."
        )
        st.code(
            "CD8 = [CD8, CD3e, CD45] OR [CD8, CD3e]; SUPPORT=[]; NOT=[]\n"
            "Treg = ['FoxP3', 'CD4', 'CD3e'] OR ['FoxP3', 'CD4', 'CD45']; SUPPORT=[]; NOT=[]\n"
            "Epithelial = [Pan-Cytokeratin] OR [E-cadherin]",
            language="text",
        )
        st.caption(
            "After you click Apply, the table automatically creates one row per phenotype and enough "
            "Criteria columns for the phenotype with the most OR alternatives."
        )
        rule_text = st.text_area(
            "Rules", value=st.session_state.get("resolve_rule_text", ""),
            key="resolve_rule_text", height=170, label_visibility="collapsed",
        )
        parsed, problems = parse_resolve_rule_text(rule_text, list(selected_markers or []))
        if rule_text.strip():
            if parsed:
                st.success(f"{len(parsed)} rule(s) understood.")
                st.dataframe(
                    pd.DataFrame([{
                        "Phenotype": r["Phenotype"],
                        "Must have": " OR ".join(" AND ".join(a) for a in r["alternatives"]),
                        "Supportive": ", ".join(r.get("support", [])),
                        "Must NOT have": ", ".join(r["excluded"]),
                    } for r in parsed]),
                    width="stretch", hide_index=True,
                )
            for problem in problems:
                st.warning(problem)
            replace = st.checkbox(
                "Replace the whole table (otherwise these are added to it)",
                value=True, key="resolve_rule_replace",
            )
            if st.button("Apply these rules to the table", type="primary",
                         key="resolve_rule_apply", disabled=not parsed):
                needed = max((len(r["alternatives"]) for r in parsed), default=1)
                if replace:
                    # Match the rules exactly; carrying over stale empty columns
                    # from the previous table only adds unusable widgets.
                    columns = [f"Criteria {i + 1}" for i in range(needed)]
                else:
                    columns = list(criteria_cols)
                    while len(columns) < needed:
                        columns.append(f"Criteria {len(columns) + 1}")
                built = []
                for rule in parsed:
                    row = {"Phenotype": rule["Phenotype"],
                           _RESOLVE_SUPPORT_COL: list(rule.get("support", [])),
                           _RESOLVE_EXCLUSION_COL: list(rule["excluded"])}
                    for position, column in enumerate(columns):
                        row[column] = list(rule["alternatives"][position]) if position < len(rule["alternatives"]) else []
                    built.append(row)
                table = pd.DataFrame(built)
                if not replace:
                    existing = _coerce_phenotype_table(
                        st.session_state["resolve_pheno_table"].copy(), columns)
                    keep = existing.loc[~existing["Phenotype"].isin(table["Phenotype"])]
                    table = pd.concat([keep, table], ignore_index=True)
                built_table = _coerce_phenotype_table(table, columns)
                # Writing straight to session state is not enough. A cache-signature
                # check further down re-reads the phenotype cache whenever its
                # signature differs from the one recorded in state, and that check
                # runs after this block on the next rerun - so an in-memory-only
                # change is silently replaced by whatever the cache still holds.
                # Persisting here, and recording the new signature, is what makes
                # the applied rules survive.
                payload = _write_resolve_pheno_payload(
                    {"criteria_cols": columns, "table": built_table.to_dict(orient="records")},
                    [resolve_pheno_panel_cache, resolve_pheno_cache],
                )
                _clear_resolve_pheno_widget_state()
                st.session_state["resolve_criteria_cols"] = list(payload["criteria_cols"])
                st.session_state["resolve_pheno_table"] = _coerce_phenotype_table(
                    pd.DataFrame(payload["table"]), list(payload["criteria_cols"])
                )
                st.session_state["resolve_loaded_pheno_cache_sig"] = _resolve_pheno_cache_signature(
                    resolve_pheno_panel_cache
                )
                st.session_state["_resolve_rules_just_applied"] = True

                # Seed the row widgets directly. st.multiselect ignores `default=`
                # whenever its key already exists in session state, so relying on
                # the default is what left every criterion blank: the widgets kept
                # whatever they held before. Writing the values under their keys
                # before the widgets are built is the only assignment Streamlit
                # honours, and it is legal because the widgets do not exist yet on
                # the coming rerun.
                allowed = set(selected_markers or [])
                seeded = st.session_state["resolve_pheno_table"].reset_index(drop=True)
                st.session_state["resolve_pheno_table"] = seeded
                for position, seed_row in seeded.iterrows():
                    # The name must be seeded alongside the markers. Seeding only
                    # the markers left each selectbox holding whatever phenotype
                    # occupied that row before, so the names and markers belonged
                    # to different rules.
                    st.session_state[f"resolve_pheno_{position}"] = str(seed_row.get("Phenotype", ""))
                    for column in list(payload["criteria_cols"]):
                        values = seed_row.get(column, [])
                        st.session_state[f"resolve_criteria_{position}_{column}"] = [
                            m for m in (values if isinstance(values, list) else []) if m in allowed
                        ]
                    for extra, prefix in ((_RESOLVE_SUPPORT_COL, "resolve_support_"),
                                          (_RESOLVE_EXCLUSION_COL, "resolve_exclusion_")):
                        values = seed_row.get(extra, [])
                        st.session_state[f"{prefix}{position}"] = [
                            m for m in (values if isinstance(values, list) else []) if m in allowed
                        ]
                st.success(f"Applied {len(parsed)} rule(s) and saved them.")
                st.rerun()

    with st.expander("Diagnostics - why is the table empty?", expanded=False):
        table_now = st.session_state.get("resolve_pheno_table")
        st.write(f"**Confirmed markers:** {len(selected_markers)}")
        if selected_markers:
            st.code(", ".join(map(str, selected_markers[:40])) + (" ..." if len(selected_markers) > 40 else ""))
        else:
            st.error(
                "No confirmed markers. The rows can only offer markers from this list, so every "
                "criterion will render empty. Confirm your marker selection above first."
            )
        if isinstance(table_now, pd.DataFrame) and not table_now.empty:
            st.write(f"**Stored table:** {table_now.shape[0]} row(s), columns {list(table_now.columns)}")
            preview = table_now.head(4).copy()
            for column in preview.columns:
                if column != "Phenotype":
                    preview[column] = preview[column].apply(
                        lambda v: ", ".join(map(str, v)) if isinstance(v, list) else str(v))
            st.dataframe(preview, width="stretch", hide_index=True)
            stored = {str(m) for _, row in table_now.iterrows()
                      for column in table_now.columns if column != "Phenotype"
                      for m in (row[column] if isinstance(row[column], list) else [])}
            unknown = sorted(stored - {str(m) for m in selected_markers})
            if stored and unknown:
                st.error(
                    "These markers are stored in the table but are **not** in the confirmed list, so "
                    "they are pruned from every row before display: " + ", ".join(unknown)
                )
            elif not stored:
                st.warning("The stored table has phenotype names but no markers in any column.")
            else:
                st.success("Every stored marker is present in the confirmed list.")
        else:
            st.write("**Stored table:** empty")

    with st.form("resolve_pheno_form", clear_on_submit=False):
        label_cols = st.columns([1.2] + [1] * len(criteria_cols) + [1, 1])
        label_cols[0].markdown("**Phenotype**")
        for position in range(len(criteria_cols)):
            label_cols[position + 1].markdown("**Must have**" if position == 0 else "**or**")
        label_cols[len(criteria_cols) + 1].markdown("**Supportive**")
        label_cols[len(criteria_cols) + 2].markdown("**Must NOT have**")
        pheno_data = []
        for idx, row in st.session_state['resolve_pheno_table'].iterrows():
            row_cols = st.columns([1.2] + [1] * len(criteria_cols) + [1, 1])

            with row_cols[0]:
                unclassified_label = st.session_state.get('resolve_config', {}).get('unclassified_label', 'Others')
                pheno_options = [""] + get_all_phenotype_options(include_unassigned=False)
                if unclassified_label and unclassified_label not in pheno_options:
                    pheno_options.append(unclassified_label)
                # A rule may introduce a phenotype the catalogue has not seen yet.
                # Streamlit raises if a selectbox holds a value outside its options,
                # so every name present in the table is offered.
                for known in st.session_state["resolve_pheno_table"]["Phenotype"]:
                    if str(known) and str(known) not in pheno_options:
                        pheno_options.append(str(known))
                current_name = str(st.session_state.get(f"resolve_pheno_{idx}", row['Phenotype']) or "")
                pheno_name = st.selectbox(
                    f"Phenotype row {idx}",
                    pheno_options,
                    index=pheno_options.index(current_name) if current_name in pheno_options else 0,
                    key=f"resolve_pheno_{idx}",
                    label_visibility="collapsed"
                )

            criteria_values: dict[str, list[str]] = {}
            for c_idx, c_name in enumerate(criteria_cols):
                with row_cols[c_idx + 1]:
                    if selected_markers:
                        # Filter saved defaults to only include markers still in the options list
                        raw_default = row[c_name] if isinstance(row[c_name], list) else []
                        safe_default = [m for m in raw_default if m in selected_markers]
                        criteria_values[c_name] = st.multiselect(
                            f"{c_name} row {idx}",
                            selected_markers,
                            default=safe_default,
                            key=f"resolve_criteria_{idx}_{c_name}",
                            label_visibility="collapsed"
                        )
                    else:
                        st.info("Select markers above first")
                        criteria_values[c_name] = []

            with row_cols[len(criteria_cols) + 1]:
                if selected_markers:
                    raw_sup = row.get(_RESOLVE_SUPPORT_COL, [])
                    safe_sup = [m for m in (raw_sup if isinstance(raw_sup, list) else []) if m in selected_markers]
                    support_values = st.multiselect(
                        f"Supportive row {idx}", selected_markers, default=safe_sup,
                        key=f"resolve_support_{idx}", label_visibility="collapsed",
                    )
                else:
                    support_values = []

            with row_cols[len(criteria_cols) + 2]:
                if selected_markers:
                    raw_excl = row.get(_RESOLVE_EXCLUSION_COL, [])
                    safe_excl = [m for m in (raw_excl if isinstance(raw_excl, list) else []) if m in selected_markers]
                    exclusion_values = st.multiselect(
                        f"Must not have row {idx}",
                        selected_markers,
                        default=safe_excl,
                        key=f"resolve_exclusion_{idx}",
                        label_visibility="collapsed",
                    )
                else:
                    exclusion_values = []

            row_out = {"Phenotype": pheno_name}
            row_out.update(criteria_values)
            row_out[_RESOLVE_SUPPORT_COL] = support_values
            row_out[_RESOLVE_EXCLUSION_COL] = exclusion_values
            pheno_data.append(row_out)

        add_row = st.form_submit_button("Add Phenotype Row")
        save_rows = st.form_submit_button("Save Phenotypes", type="primary")

        if add_row:
            new_row = {"Phenotype": "", **{e: [] for e in _RESOLVE_EXTRA_COLS}}
            for c in criteria_cols:
                new_row[c] = []
            st.session_state['resolve_pheno_table'] = pd.concat(
                [st.session_state['resolve_pheno_table'], pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.rerun()

        if save_rows:
            payload = _write_resolve_pheno_payload(
                {
                    "criteria_cols": st.session_state.get('resolve_criteria_cols', []),
                    "table": pheno_data,
                },
                [resolve_pheno_panel_cache, resolve_pheno_cache],
            )
            resolve_state['resolve_criteria_cols'] = list(payload['criteria_cols'])
            resolve_state['resolve_pheno_table'] = pd.DataFrame(payload['table'])
            resolve_state['resolve_loaded_pheno_cache_sig'] = _resolve_pheno_cache_signature(resolve_pheno_panel_cache)
            st.success("Phenotype definitions saved")
    
    # --- SECTION 4: RUN PIPELINE ---
    st.divider()
    
    selected_markers = st.session_state.get('resolve_confirmed_markers', [])
    
    st.markdown("### 5. Run Resolve")
    if not selected_markers:
        st.warning("Please select markers first")
    elif st.session_state['resolve_pheno_table'].empty or not any(
        st.session_state['resolve_pheno_table']['Phenotype']
    ):
        st.warning("Please define at least one phenotype")
    else:
        if st.button("Run Phenotype Pipeline", type="primary", width="stretch", key="resolve_run_btn"):
            if not full_input_path or not Path(full_input_path).exists():
                st.error("Please select a valid input file")
                return

            audit_cfg = st.session_state.get('resolve_config', {})
            audit_output_col = audit_cfg.get('output_col', 'phenotype_resolve')
            audit_cluster_col = audit_cfg.get('cluster_col', 'resolve_cluster')
            run_audit = begin_phenotyping_run_audit(
                Path(project_root),
                "resolve",
                input_csv=full_input_path,
                markers=selected_markers,
            )
            audit_input_rows = None
            
            # Clear previous results
            if 'resolve_temp_file' in st.session_state:
                del st.session_state['resolve_temp_file']
            if 'resolve_phenotype_map' in st.session_state:
                del st.session_state['resolve_phenotype_map']
            
            with st.status("Running Threshold Phenotyping Pipeline...", expanded=True) as status:
                try:
                    cfg = st.session_state['resolve_config']
                    
                    # Load data
                    status.write(f"[{_timestamp()}] Loading input data...")
                    _log_console(f"Loading input data...")
                    df = _dataio_load(full_input_path)
                    audit_input_rows = len(df)
                    original_df = df.copy()
                    status.write(f"[{_timestamp()}] Loaded {len(df):,} cells")
                    _log_console(f"Loaded {len(df):,} cells")

                    _resolve_filter = {
                        'enable_filter': cfg.get('enable_filter', False),
                        'filter_col': cfg.get('filter_col'),
                        'filter_val': cfg.get('filter_val'),
                    }
                    df = apply_data_filter(df, _resolve_filter, logger_fn=lambda msg: (
                        status.write(f"[{_timestamp()}] {msg}"),
                        _log_console(f"[{_timestamp()}] Resolve: {msg}")
                    ))
                    
                    # Drop rows containing NaN/Inf in selected markers
                    if selected_markers:
                        original_len_nan = len(df)
                        valid_mask = ~df[selected_markers].apply(pd.to_numeric, errors='coerce').isna().any(axis=1)
                        df = df[valid_mask].copy()
                        status.write(f"[{_timestamp()}] Filtered out {original_len_nan - len(df):,} cells containing NaN/Inf in selected markers. Remaining: {len(df):,} cells.")
                        _log_console(f"Filtered out {original_len_nan - len(df):,} cells containing NaN/Inf in selected markers. Remaining: {len(df):,} cells.")
                        if len(df) == 0:
                            status.error("No cells remain after filtering out NaN/Inf values in selected markers.")
                            return
                    
                    # Extract phenotype definitions from table (OR across criteria columns)
                    phenotype_defs = {}
                    phenotype_exclusions: dict[str, list[str]] = {}
                    phenotype_support: dict[str, list[str]] = {}
                    criteria_cols = st.session_state.get('resolve_criteria_cols', [])
                    for _, row in st.session_state['resolve_pheno_table'].iterrows():
                        pheno_name = row.get('Phenotype', '')
                        if not pheno_name:
                            continue
                        criteria_sets = []
                        for c in criteria_cols:
                            markers = row.get(c, [])
                            if isinstance(markers, list) and markers:
                                criteria_sets.append(markers)
                        if criteria_sets:
                            phenotype_defs[pheno_name] = criteria_sets
                        supportive = row.get(_RESOLVE_SUPPORT_COL, [])
                        if isinstance(supportive, list) and supportive:
                            phenotype_support[pheno_name] = list(supportive)
                        forbidden = row.get(_RESOLVE_EXCLUSION_COL, [])
                        if isinstance(forbidden, list) and forbidden:
                            phenotype_exclusions[pheno_name] = list(forbidden)
                    
                    if not phenotype_defs:
                        status.error("No valid phenotypes defined")
                        return
                    
                    status.write(f"[{_timestamp()}] Phenotypes to assign: {list(phenotype_defs.keys())}")
                    _log_console(f"Phenotypes to assign: {list(phenotype_defs.keys())}")
                    
                    # Apply thresholds (mode-aware)
                    _resolve_mode_map_run = {
                        **_THRESHOLD_MODE_TO_KEY, "None (No Thresholds)": "none", RESOLVE_Z1_MODE: "z1",
                    }
                    run_threshold_mode = cfg.get('threshold_mode', 'Predefined (CSV)')
                    run_mode_key = _resolve_mode_map_run.get(run_threshold_mode, "predefined")
                    apply_thresholds_flag = (run_mode_key != "none")

                    if run_mode_key == "predefined":
                        status.write(
                            f"[{_timestamp()}] Using saved threshold CSVs. To regenerate them from the "
                            "images, use Step 2 > Detect Background."
                        )
                    status.write(f"[{_timestamp()}] Threshold mode: {run_threshold_mode}")
                    _log_console(f"Threshold mode={run_threshold_mode} (key={run_mode_key})")

                    batch_col = normalize_optional_column_name(cfg.get('batch_col', 'batch_id'))
                    batch_group_col = batch_col if batch_col in df.columns else None
                    norm_flags = (cfg['norm_z1'], cfg['norm_z2'], cfg['norm_cdf'], cfg['norm_nlog'])
                    norm_order = cfg.get('norm_order', ["Z1", "Z2", "CDF", "NLog"])
                    thresholds_table = st.session_state.get('resolve_threshold_table')
                    runtime_threshold_dir = str(
                        st.session_state.get('threshold_dir')
                        or _preferred_threshold_directory(project_root)
                    )
                    runtime_threshold_fingerprint = _resolve_threshold_fingerprint(
                        runtime_threshold_dir, cfg, selected_markers
                    )
                    active_threshold_fingerprint = str(
                        st.session_state.get('resolve_threshold_table_fingerprint')
                        or cached_params.get('threshold_table_fingerprint')
                        or ''
                    ).strip()
                    if (
                        thresholds_table is not None
                        and not thresholds_table.empty
                        and active_threshold_fingerprint != runtime_threshold_fingerprint
                    ):
                        _log_console(f"Ignoring stale reviewed threshold table; "
                            "source settings fingerprint changed")
                        thresholds_table = None
                    used_external_thresholds = False

                    # Predefined mode is fail-closed: load through the same source-aware
                    # path used by the review table, then require every batch x marker
                    # gate before subtracting anything.  This removes the historical
                    # table-vs-disk semantic split and prevents a missing CSV from
                    # silently changing the method to raw marker > 0.
                    if apply_thresholds_flag and run_mode_key == "predefined":
                        expected_batches = (
                            df[batch_group_col].dropna().unique().tolist()
                            if batch_group_col else []
                        )
                        if batch_group_col and df[batch_group_col].isna().any():
                            missing_batch_cells = int(df[batch_group_col].isna().sum())
                            status.error(
                                f"Predefined thresholds cannot be matched for {missing_batch_cells:,} cell(s) "
                                f"whose '{batch_group_col}' value is missing. Fill or filter those rows first."
                            )
                            _log_console(f"Threshold coverage failed; "
                                f"{missing_batch_cells} cells have no batch ID", level="error")
                            return

                        table_has_values = _threshold_table_has_values(
                            thresholds_table, selected_markers
                        )
                        if (
                            thresholds_table is None
                            or thresholds_table.empty
                            or not table_has_values
                        ):
                            threshold_dir = runtime_threshold_dir
                            thresholds_table = _load_thresholds_table(
                                threshold_dir,
                                selected_markers,
                                expected_batches,
                                cfg.get('threshold_marker_col') or None,
                                cfg.get('threshold_value_col') or None,
                                cfg.get('polygon_prefix', 'Polygon') or None,
                                cfg.get('threshold_statistic', 'median'),
                                cfg.get('threshold_source', 'Auto'),
                            )
                            status.write(
                                f"[{_timestamp()}] Loaded {len(thresholds_table)} threshold row(s) "
                                f"from {threshold_dir}"
                            )
                            _log_console(f"Runtime threshold load rows="
                                f"{len(thresholds_table)}, source={cfg.get('threshold_source', 'Auto')}, "
                                f"statistic={cfg.get('threshold_statistic', 'median')}")

                        coverage = _threshold_table_coverage(
                            thresholds_table, selected_markers, expected_batches or None
                        )
                        if not coverage['complete']:
                            coverage_problems = []
                            if coverage['ambiguous_global']:
                                coverage_problems.append(
                                    "multiple threshold rows are available but no Batch Column is selected"
                                )
                            if coverage['missing_batches']:
                                values = coverage['missing_batches']
                                preview = ", ".join(values[:8])
                                coverage_problems.append(
                                    f"missing batch(es): {preview}"
                                    + (f" (+{len(values) - 8} more)" if len(values) > 8 else "")
                                )
                            if coverage['duplicate_batches']:
                                values = coverage['duplicate_batches']
                                coverage_problems.append(
                                    "duplicate threshold rows for batch(es): " + ", ".join(values[:8])
                                )
                            if coverage['missing_by_batch']:
                                missing_preview = []
                                for batch_name, markers_missing in list(
                                    coverage['missing_by_batch'].items()
                                )[:6]:
                                    marker_preview = ", ".join(markers_missing[:8])
                                    if len(markers_missing) > 8:
                                        marker_preview += f" (+{len(markers_missing) - 8} more)"
                                    missing_preview.append(f"{batch_name}: {marker_preview}")
                                coverage_problems.append(
                                    "missing/non-finite marker gates — " + "; ".join(missing_preview)
                                )
                            coverage_text = "; ".join(coverage_problems) or "no usable thresholds"
                            status.error(
                                "Predefined threshold coverage is incomplete, so RESOLVE stopped before "
                                f"phenotyping: {coverage_text}. Load/correct and save a complete threshold table."
                            )
                            _log_console(f"Threshold coverage failed: {coverage_text}", level="error")
                            return

                        try:
                            sample_ids = thresholds_table["batch_id"].astype(str).head(3).tolist()
                            _log_console(f"Complete threshold table rows="
                                f"{len(thresholds_table)}, sample batch_id={sample_ids}")
                        except Exception:
                            _log_console(f"Complete threshold table rows={len(thresholds_table)}")

                        try:
                            binary_df, norm_df = _apply_thresholds_from_table(
                                df=df,
                                thresholds_table=thresholds_table,
                                selected_markers=selected_markers,
                                batch_col=batch_group_col,
                                norm_flags=norm_flags,
                                norm_order=norm_order,
                                method_tag='resolve',
                                tie_score_transform=cfg.get(
                                    'tie_score_transform', 'identity_residual'
                                ),
                                tie_score_cofactor=cfg.get('tie_score_cofactor', 5.0),
                            )
                        except Exception as threshold_error:
                            status.error(
                                "Predefined threshold application failed; RESOLVE stopped without falling "
                                f"back to raw > 0: {threshold_error}"
                            )
                            _log_console(f"Threshold application failed closed: "
                                f"{threshold_error}", level="error")
                            return
                        used_external_thresholds = True
                        status.write(f"[{_timestamp()}] Applied complete predefined thresholds")
                        _log_console(f"Applied complete predefined thresholds")

                    # Branch 2b: Z1 score mode -> positive where Z1 > cutoff, no files.
                    elif run_mode_key == "z1":
                        z1_cutoff = float(cfg.get('z1_cutoff', DEFAULT_RESOLVE_Z1_CUTOFF))
                        z1_cofactor = float(cfg.get('tie_score_cofactor', 5.0))
                        binary_df, norm_df = _apply_resolve_z1_positivity(
                            df, selected_markers, batch_col, cutoff=z1_cutoff, cofactor=z1_cofactor,
                        )
                        status.write(
                            f"[{_timestamp()}] Z1 score mode: ArcSinh (cofactor {z1_cofactor:g}) -> Z1 per batch, "
                            f"positive if Z1 > {z1_cutoff:g}"
                        )
                        _log_console(f"Z1 positivity, cutoff={z1_cutoff:g}")

                    # Branch 3: Auto mode (Otsu, Triangle, etc.) → compute on the fly
                    elif apply_thresholds_flag:
                        auto_func = AUTO_THRESHOLD_METHODS.get(run_mode_key)
                        if auto_func is not None:
                            status.write(f"[{_timestamp()}] Computing {run_threshold_mode} thresholds on the fly...")
                            _log_console(f"On-the-fly auto-threshold ({run_threshold_mode})")
                            try:
                                auto_thres_df = compute_auto_thresholds(
                                    df, selected_markers, run_mode_key,
                                    batch_col=batch_col if batch_col in df.columns else None,
                                )
                                binary_df, norm_df = _apply_thresholds_from_table(
                                    df=df,
                                    thresholds_table=auto_thres_df,
                                    selected_markers=selected_markers,
                                    batch_col=batch_col,
                                    norm_flags=norm_flags,
                                    norm_order=norm_order,
                                    method_tag='resolve',
                                    tie_score_transform=cfg.get(
                                        'tie_score_transform', 'identity_residual'
                                    ),
                                    tie_score_cofactor=cfg.get('tie_score_cofactor', 5.0),
                                )
                                used_external_thresholds = True
                                status.write(f"[{_timestamp()}] Auto-thresholds applied ({len(auto_thres_df)} batches)")
                                _log_console(f"Auto-thresholds applied ({len(auto_thres_df)} batches)")
                            except Exception as e:
                                status.warning(f"[{_timestamp()}] Auto-threshold failed: {e}, falling back to > 0")
                                _log_console(f"Auto-threshold failed: {e}", level="error")
                                binary_df, norm_df = _apply_resolve_without_thresholds(
                                    df, selected_markers, batch_col, cfg, norm_order
                                )
                        else:
                            status.warning(f"[{_timestamp()}] Unknown auto-threshold method '{run_mode_key}', using > 0")
                            binary_df, norm_df = _apply_resolve_without_thresholds(
                                df, selected_markers, batch_col, cfg, norm_order
                            )

                    # Branch 4: None mode → raw > 0 binarization
                    else:
                        status.write(f"[{_timestamp()}] No thresholds applied (mode: None)")
                        _log_console(f"No thresholds (raw > 0)")
                        binary_df, norm_df = _apply_resolve_without_thresholds(
                            df, selected_markers, batch_col, cfg, norm_order
                        )

                    # Robust fallback: if threshold processing produced no binary
                    # columns (e.g. no threshold files found), fall back to simple
                    # > 0 binarization so the pipeline can still run.
                    if binary_df.empty or len(binary_df.columns) == 0:
                        if run_mode_key == "predefined":
                            status.error(
                                f"[{_timestamp()}] Predefined thresholding produced no binary marker columns. "
                                "RESOLVE stopped rather than changing the method to raw marker > 0."
                            )
                            _log_console(f"Predefined thresholding failed closed "
                                "because no binary columns were produced", level="error")
                            return
                        status.warning(f"[{_timestamp()}] No binary columns from thresholding -- falling back to raw marker > 0 binarization")
                        _log_console(f"Fallback binarization (marker > 0)")
                        binary_df = pd.DataFrame(index=df.index)
                        for m in selected_markers:
                            if m in df.columns:
                                binary_df[f"{m}_bin_resolve"] = (df[m] > 0).fillna(False).astype(int)
                        if norm_df.empty or len(norm_df.columns) == 0:
                            norm_df = _apply_flexible_normalization_safe(
                                df[[m for m in selected_markers if m in df.columns]],
                                marker_cols=[m for m in selected_markers if m in df.columns],
                                group_cols=[batch_col] if batch_col in df.columns else None,
                                suffix="_norm_resolve",
                                apply_z1=cfg['norm_z1'],
                                apply_z2=cfg['norm_z2'],
                                apply_cdf=cfg['norm_cdf'],
                                apply_nlog=cfg['norm_nlog'],
                                norm_order=norm_order,
                            )

                    # --- Auto-build phenotype hierarchy (parent/daughter) ---
                    # Detects when one phenotype's marker set is a strict subset
                    # of another's, establishing a parent -> child relationship.
                    # e.g. CD4 {CD45,CD3,CD4} is parent of Treg {CD45,CD3,CD4,FOXP3}
                    # The child (more specific) phenotype wins when both are positive.
                    phenotype_marker_sets: dict[str, set] = {}
                    for pheno, criteria_sets in phenotype_defs.items():
                        all_markers_for_pheno: set[str] = set()
                        for clause in criteria_sets:
                            all_markers_for_pheno.update(clause)
                        phenotype_marker_sets[pheno] = all_markers_for_pheno

                    phenotype_hierarchy: dict[str, str | None] = {}
                    for child, child_markers in phenotype_marker_sets.items():
                        best_parent: str | None = None
                        best_parent_size = 0
                        for parent, parent_markers in phenotype_marker_sets.items():
                            if parent == child:
                                continue
                            # parent markers must be a strict subset of child markers
                            if parent_markers < child_markers and len(parent_markers) > best_parent_size:
                                best_parent = parent
                                best_parent_size = len(parent_markers)
                        phenotype_hierarchy[child] = best_parent

                    has_hierarchy = any(v is not None for v in phenotype_hierarchy.values())
                    if has_hierarchy:
                        hierarchy_str = ", ".join(
                            f"{child} -> {parent}" for child, parent in phenotype_hierarchy.items() if parent
                        )
                        status.write(f"[{_timestamp()}] Auto-detected hierarchy: {hierarchy_str}")
                        _log_console(f"Auto hierarchy: {hierarchy_str}")

                    # --- Adaptive distinguishing-marker resolution ---
                    # When no external thresholds exist and the naive > 0
                    # binarization is in effect, trace expression of
                    # distinguishing markers (e.g. low FOXP3 in CD4 cells)
                    # causes false daughter assignments.  Use Otsu's method
                    # to find a data-driven split between "trace" and
                    # "truly positive" for each distinguishing marker.
                    if has_hierarchy and not used_external_thresholds:
                        status.write(f"[{_timestamp()}] Running adaptive threshold resolution for parent/child markers...")
                        _log_console(f"Adaptive threshold resolution (no external thresholds)")
                        binary_df = _adaptive_resolve_hierarchy(
                            df=df,
                            binary_df=binary_df,
                            phenotype_defs=phenotype_defs,
                            phenotype_hierarchy=phenotype_hierarchy,
                            bin_suffix="_bin_resolve",
                            logger_fn=lambda msg: (
                                status.write(f"[{_timestamp()}] {msg}"),
                                _log_console(f"[{_timestamp()}] Resolve: {msg}"),
                            ),
                        )

                    # Assign phenotypes
                    status.write(f"[{_timestamp()}] Assigning phenotypes...")
                    _log_console(f"Assigning phenotypes...")
                    phenotype_col = cfg['output_col']
                    unclassified_label = cfg.get('unclassified_label', 'Others')
                    try:
                        bin_cols = list(binary_df.columns)
                        norm_cols = list(norm_df.columns)
                        assign_df = assign_phenotypes_binary_tiebreak(
                            df_processed=pd.concat([binary_df, norm_df], axis=1),
                            bin_marker_cols=bin_cols,
                            norm_marker_cols=norm_cols,
                            phenotype_col_name=phenotype_col,
                            phenotype_definitions=phenotype_defs,
                            phenotype_hierarchy=phenotype_hierarchy,
                        )
                        assigned_series = assign_df[phenotype_col]
                        if phenotype_support:
                            from spacemap.ph_thres_utils import apply_supportive_markers

                            before_support = assigned_series.copy()
                            assigned_series = apply_supportive_markers(
                                assigned_series,
                                pd.concat([binary_df, norm_df], axis=1),
                                bin_cols,
                                norm_cols,
                                phenotype_defs,
                                phenotype_support,
                            )
                            moved = int((before_support != assigned_series).sum())
                            status.write(
                                f"[{_timestamp()}] Supportive markers settled {moved:,} contested cell(s)"
                            )
                        if phenotype_exclusions:
                            from spacemap.ph_thres_utils import apply_exclusion_markers

                            before = assigned_series.copy()
                            assigned_series = apply_exclusion_markers(
                                assigned_series,
                                pd.concat([binary_df, norm_df], axis=1),
                                norm_cols,
                                phenotype_defs,
                                phenotype_exclusions,
                                reassign=True,
                            )
                            changed = int((before != assigned_series).sum())
                            status.write(
                                f"[{_timestamp()}] Exclusion markers moved {changed:,} cell(s) "
                                f"whose forbidden marker outweighed the markers that defined the call"
                            )
                            _log_console(f"exclusion markers moved {changed:,} cells")
                        df[phenotype_col] = assigned_series.replace({"unclassified": unclassified_label})
                        status.write(f"[{_timestamp()}] Phenotypes assigned successfully")
                        _log_console(f"Phenotypes assigned successfully")
                    except Exception as e:
                        status.error(f"[{_timestamp()}] Phenotype assignment failed: {e}")
                        _log_console(f"Phenotype assignment failed: {e}", level="error")
                        return

                    try:
                        if phenotype_col in df.columns:
                            unclassified_count = (df[phenotype_col] == unclassified_label).sum()
                            if unclassified_count == len(df):
                                status.warning(f"[{_timestamp()}] All cells labeled '{unclassified_label}'. Check criteria and thresholds.")
                    except Exception:
                        pass

                    # Save/export controls are rendered below the result figures.
                    
                    # Create clusters from phenotypes (for visualization)
                    df_results = pd.concat([df, norm_df], axis=1)

                    unique_phenotypes = sorted([p for p in df_results[phenotype_col].unique() if p != 'Unassigned'])
                    pheno_to_cluster = {p: i for i, p in enumerate(unique_phenotypes)}
                    df_results[cfg['cluster_col']] = df_results[phenotype_col].map(pheno_to_cluster).fillna(-1)
                    
                    # Save to temp file for visualization
                    temp_file = project_root / 'outputs/phenotyping/cache' / f"resolve_results_{int(time.time())}.parquet"
                    df_results.to_parquet(temp_file)
                    st.session_state['resolve_temp_file'] = str(temp_file)
                    st.session_state['resolve_phenotype_map'] = {int(c): p for p, c in pheno_to_cluster.items()}
                    st.session_state['resolve_available_phenotypes'] = ['Unassigned'] + unique_phenotypes
                    st.session_state['resolve_norm_cols'] = list(norm_df.columns)
                    st.session_state['resolve_outcol'] = phenotype_col
                    st.session_state['resolve_cluster_col_name'] = cfg['cluster_col']
                    resolve_plot_provenance = {
                        'marker_columns': list(norm_df.columns),
                        'normalization_steps': _resolve_enabled_normalization_steps(cfg),
                        'threshold_mode': run_threshold_mode,
                        'thresholds_applied': bool(used_external_thresholds),
                        'tie_score_transform': cfg.get(
                            'tie_score_transform', 'identity_residual'
                        ),
                        'tie_score_cofactor': cfg.get('tie_score_cofactor', 5.0),
                        'inverse_sinh_applied': False,
                        'batch_column': batch_group_col,
                    }
                    st.session_state['resolve_plot_provenance'] = resolve_plot_provenance
                    st.session_state['resolve_config']['last_plot_preprocessing'] = resolve_plot_provenance
                    
                    status.write(f"[{_timestamp()}] Pipeline complete!")
                    _log_console(f"Pipeline complete!")
                    st.success("Threshold phenotyping complete")

                    finish_phenotyping_run_audit(
                        run_audit,
                        status="success",
                        output_path=temp_file,
                        input_rows=audit_input_rows,
                        output_rows=len(df_results),
                        output_column=phenotype_col,
                        cluster_column=cfg['cluster_col'],
                        phenotype_map=st.session_state['resolve_phenotype_map'],
                        extra={
                            "phenotype_definitions": phenotype_defs,
                            "phenotype_cell_counts": df_results[phenotype_col].value_counts(dropna=False).to_dict(),
                        },
                    )
                    run_audit = None
                    
                except Exception as e:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="failed",
                        input_rows=audit_input_rows,
                        output_column=audit_output_col,
                        cluster_column=audit_cluster_col,
                        error=str(e),
                    )
                    run_audit = None
                    status.error(f"[{_timestamp()}] Pipeline error: {str(e)}")
                    logger.error(f"Resolve pipeline error: {e}")
                finally:
                    if run_audit is not None:
                        finish_phenotyping_run_audit(
                            run_audit,
                            status="aborted",
                            input_rows=audit_input_rows,
                            output_column=audit_output_col,
                            cluster_column=audit_cluster_col,
                            error="Pipeline stopped before completion; see the Resolve status messages.",
                        )
    
    # --- RESULTS VISUALIZATION & EXPORT ---
    if 'resolve_temp_file' in st.session_state and Path(st.session_state['resolve_temp_file']).exists():
        st.divider()
        st.subheader("Results & Phenotype Assignment")
        st.caption("Interactive visualization and export of threshold phenotyping results")
        
        temp_file = st.session_state['resolve_temp_file']
        df_results = pd.read_parquet(temp_file)
        norm_cols = st.session_state.get('resolve_norm_cols', [])
        phenotype_map = st.session_state.get('resolve_phenotype_map', {})
        available_phenotypes = st.session_state.get('resolve_available_phenotypes', [])
        output_col = st.session_state.get('resolve_outcol', 'phenotype_resolve')
        cluster_col = st.session_state.get('resolve_cluster_col_name', 'resolve_cluster')
        
        # Keep figures and export controls in one continuous method view.
        heatmap_section = st.container()
        export_section = st.container()
        
        with heatmap_section:
            st.markdown("### Expression Heatmap & Phenotype Statistics")

            if output_col in df_results.columns:
                pheno_summary = (
                    df_results[output_col]
                    .fillna('Unassigned')
                    .value_counts()
                    .rename_axis('Phenotype')
                    .reset_index(name='Cell Count')
                )
                pheno_summary['% Total'] = (pheno_summary['Cell Count'] / len(df_results) * 100).map(lambda x: f"{x:.1f}%")
                st.dataframe(pheno_summary, width="stretch", hide_index=True)
            
            if 'resolve_heatmap_config' not in st.session_state:
                st.session_state['resolve_heatmap_config'] = {
                    'cluster_x': False,
                    'cluster_y': True,
                    'show_counts': True,
                    'z_min': -4.0,
                    'z_max': 4.0
                }

            # Migrate the old Resolve-only display defaults now that Resolve
            # uses the same normalized-value plotting path as the other methods.
            if not st.session_state.get('resolve_normalized_plot_migration_v1', False):
                heatmap_config = st.session_state['resolve_heatmap_config']
                if (
                    float(heatmap_config.get('z_min', -2.0)) == -2.0
                    and float(heatmap_config.get('z_max', 2.0)) == 2.0
                ):
                    heatmap_config['z_min'] = -4.0
                    heatmap_config['z_max'] = 4.0
                try:
                    if float(st.session_state.get('resolve_dot_positive_threshold', 1.5)) == 10.0:
                        st.session_state['resolve_dot_positive_threshold'] = 1.5
                except (TypeError, ValueError):
                    st.session_state['resolve_dot_positive_threshold'] = 1.5
                st.session_state['resolve_heatmap_score_mode'] = 'Selected Pipeline Normalization'
                st.session_state['resolve_normalized_plot_migration_v1'] = True
            
            hm_row1 = st.columns([2, 2, 2])
            with hm_row1[0]:
                st.session_state['resolve_heatmap_config']['cluster_x'] = st.checkbox(
                    "Cluster Phenotypes (X)",
                    st.session_state['resolve_heatmap_config']['cluster_x'],
                    key="resolve_cluster_x"
                )
            with hm_row1[1]:
                st.session_state['resolve_heatmap_config']['cluster_y'] = st.checkbox(
                    "Cluster Markers (Y)",
                    st.session_state['resolve_heatmap_config']['cluster_y'],
                    key="resolve_cluster_y"
                )
            with hm_row1[2]:
                st.session_state['resolve_heatmap_config']['show_counts'] = st.checkbox(
                    "Show Cell Counts",
                    st.session_state['resolve_heatmap_config']['show_counts'],
                    key="resolve_show_counts"
                )
            
            with st.expander("Heatmap Configuration", expanded=False):
                current_min = float(st.session_state['resolve_heatmap_config'].get('z_min', -4.0))
                current_max = float(st.session_state['resolve_heatmap_config'].get('z_max', 4.0))
                z_range = st.slider(
                    "Color Scale Range (Z-Score)",
                    min_value=-10.0,
                    max_value=10.0,
                    value=(current_min, current_max),
                    step=0.5,
                    key="resolve_z_range_slider",
                )
                st.session_state['resolve_heatmap_config']['z_min'] = float(z_range[0])
                st.session_state['resolve_heatmap_config']['z_max'] = float(z_range[1])
            
            # Resolve visualization is bound to the exact normalized columns
            # used by the phenotype-assignment tie-break.
            numeric_cols = [
                column for column in norm_cols
                if column in df_results.columns
                and pd.api.types.is_numeric_dtype(df_results[column])
            ]
            if numeric_cols:
                resolve_config = st.session_state.get('resolve_config', {})
                plot_provenance = st.session_state.get(
                    'resolve_plot_provenance',
                    resolve_config.get('last_plot_preprocessing', {}),
                )
                normalization_label = _resolve_plot_normalization_label(
                    resolve_config,
                    numeric_cols,
                    plot_provenance,
                )
                heatmap_preparation_label = (
                    f"{normalization_label}; phenotype means standardized per marker for the heatmap"
                )
                marker_source_note = build_marker_source_note(
                    numeric_cols,
                    method_key="resolve",
                    normalization_label=heatmap_preparation_label,
                )
                st.info(
                    "Heatmap and dot plot use the stored Resolve normalized columns that were used "
                    f"during phenotype assignment: {normalization_label}."
                )

                df_valid = df_results[df_results[cluster_col].notna()].copy()
                if not df_valid.empty:
                    z_min = float(st.session_state['resolve_heatmap_config'].get('z_min', -4.0))
                    z_max = float(st.session_state['resolve_heatmap_config'].get('z_max', 4.0))
                    data_hash = (
                        f"resolve_{temp_file}_{len(df_valid)}_{df_valid[cluster_col].nunique()}_"
                        f"{str(sorted(numeric_cols))}_{z_min}_{z_max}_{normalization_label}"
                    )
                    base_fig, _, _, _, heatmap_data = build_leiden_heatmap_figure(
                        data_hash,
                        df_valid,
                        numeric_cols,
                        st.session_state['resolve_heatmap_config']['cluster_x'],
                        st.session_state['resolve_heatmap_config']['cluster_y'],
                        st.session_state['resolve_heatmap_config']['show_counts'],
                        cluster_col=cluster_col,
                        cluster_label='Resolve Phenotype',
                        z_min=z_min,
                        z_max=z_max,
                        marker_source_note=marker_source_note,
                    )

                    if base_fig is not None and not heatmap_data.empty:
                        st.plotly_chart(
                            base_fig,
                            key="resolve_heatmap_chart",
                            width="stretch",
                            config={'displayModeBar': False},
                        )
                        render_phenotype_statistics(
                            df_results=df_results,
                            cluster_col=cluster_col,
                            phenotype_map=phenotype_map,
                            heatmap_z_ordered=heatmap_data,
                            cluster_label='Phenotype',
                            score_label='Selected-Normalization Heatmap Z-Score',
                        )

                        render_publication_heatmap(
                            heatmap_z_ordered=heatmap_data,
                            phenotype_map=phenotype_map,
                            key_prefix='resolve',
                            vmin=z_min,
                            vmax=z_max,
                            df_results=df_results,
                            group_col=cluster_col,
                            marker_columns=numeric_cols,
                            marker_source_note=marker_source_note,
                            normalization_label=normalization_label,
                            dot_positive_threshold_default=1.5,
                        )
                    else:
                        st.warning("No Resolve normalized values are available for visualization.")
            else:
                st.warning(
                    "No stored Resolve normalized marker columns were found. Re-run Resolve to "
                    "create figures from the selected normalization pipeline."
                )
        
        with export_section:
            st.divider()
            st.markdown("### Export Results")
            
            # Save assignments
            render_save_assignments(
                df_results=df_results,
                phenotype_map=phenotype_map,
                output_col=output_col,
                project_root=project_root,
                key_prefix='resolve',
                cluster_col=cluster_col,
                update_cluster_cols=[cluster_col, output_col],
                cell_id_col=st.session_state.get('resolve_config', {}).get('cell_id_col')
            )
            
            # Export split data
            render_export_split_data(
                df_results=df_results,
                output_col=output_col,
                phenotype_map=phenotype_map,
                project_root=project_root,
                key_prefix='resolve',
                cluster_col=cluster_col,
                cell_id_col=st.session_state.get('resolve_config', {}).get('cell_id_col')
            )

    # --- VISUAL VALIDATION COMPONENT (Ported from Leiden) ---
    if st.session_state.get('viz_enabled', True):
        st.divider()
        with st.expander("Visual Validation of Representative Cells", expanded=True):
            st.markdown("#### Visualize Representative Cells")
            st.caption("Inspect random cells from specific phenotypes to validate assignments.")
            
            # Ensure we have data
            if 'resolve_temp_file' in st.session_state and Path(st.session_state['resolve_temp_file']).exists():
                # Load data if not already
                if 'df_results' not in locals():
                    df_results = pd.read_parquet(st.session_state['resolve_temp_file'])
                
                # Check Global Viz Config
                viz_roi = st.session_state.get('viz_roi_dir')
                viz_seg = st.session_state.get('viz_seg_dir')
                
                if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
                    st.warning("Please configure valid ROI and Segmentation directories in Global Input Configuration (Top of Tab 7).")
                else:
                    viz = CellVisualizer(viz_roi, viz_seg)
                    
                    # File matching (cache in session state)
                    if not st.session_state.get('resolve_viz_file_matches'):
                         # Get unique batches
                         # Resolve config might not store batch col explicitly in session, try to find it
                         batch_col = st.session_state.get('resolve_batch_col', 'batch_id')
                         if batch_col not in df_results.columns:
                             batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                         
                         if batch_col in df_results.columns:
                             unique_batches = df_results[batch_col].unique().astype(str).tolist()
                             matches = viz.match_file_triads(unique_batches)
                             st.session_state.resolve_viz_file_matches = matches
                             if not matches:
                                 st.error("No matching image files found for these batches.")
                         else:
                             st.error(f"Batch column '{batch_col}' not found in results.")
                    else:
                        viz.file_map = st.session_state.resolve_viz_file_matches

                    # UI Controls
                    # Resolve uses 'phenotype_resolve' (output_col) directly, there are no clusters.
                    # We will allow selecting a Phenotype to view.
                    
                    available_phenos = sorted([p for p in df_results[output_col].unique() if pd.notna(p) and str(p) != 'Unassigned'])
                    
                    with st.form("resolve_viz_form"):
                         vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])
                         with vv_col1:
                             selected_pheno_vv = st.selectbox(
                                 "Select Phenotype", 
                                 available_phenos,
                                 key="resolve_viz_pheno_sel"
                             )
                         with vv_col2:
                             n_viz_cells = st.number_input("Cells to View", 1, 50, 5, key="resolve_viz_n_cells")
                         with vv_col3:
                             show_neighbors = st.checkbox("Show Neighbors", value=True, key="resolve_viz_neighbors")
                         with vv_col4:
                             resolve_batch_scope = st.radio(
                                 "Batch Scope",
                                 ["All matched batches", "Specific batch"],
                                 horizontal=True,
                                 key="resolve_viz_batch_scope",
                             )
                             resolve_batch_options = sorted(str(batch) for batch in st.session_state.resolve_viz_file_matches.keys())
                             selected_batch_vv = None
                             if resolve_batch_scope == "Specific batch":
                                 selected_batch_vv = st.selectbox(
                                     "Batch",
                                     resolve_batch_options,
                                     key="resolve_viz_batch_sel",
                                 )
                             st.write("") # Spacer
                             run_viz = st.form_submit_button("Visualize Random Cells")
                    
                    # Logic
                    if run_viz and selected_pheno_vv:
                        # 1. Determine Top 5 Markers (or use Definition?)
                        # In Resolve, phenotypes are defined by rules (e.g. CD3+ CD4+).
                        # Ideally we show the markers that DEFINED this phenotype.
                        # But for consistency, let's show the highest expressing markers in this group 
                        # OR if we have the rule table, show the positive markers.
                        
                        subset = df_results[df_results[output_col] == selected_pheno_vv]
                        
                        if len(subset) == 0:
                            st.warning("No cells in this phenotype group.")
                        else:
                            # Try to infer defining markers from expression
                            # Use _norm columns or raw if available
                            # Prioritize raw markers if they exist in the dataframe (User Request)
                            # "Resolve utilizes the raw markers, CD45, CD3e..."
                            raw_markers = st.session_state.get('resolve_markers', [])
                            valid_raw_markers = [c for c in raw_markers if c in df_results.columns]
                            
                            norm_cols = []
                            is_using_raw = False
                            
                            if valid_raw_markers:
                                norm_cols = valid_raw_markers
                                is_using_raw = True
                            else:
                                # Fallback to normalized columns used in heatmap
                                norm_cols = st.session_state.get('resolve_norm_cols', [])
                                if not norm_cols:
                                    # Fallback: Try to find them by common patterns
                                    numeric_cols = [c for c in df_results.columns if pd.api.types.is_numeric_dtype(df_results[c]) and c != 'cell_id' and not c.startswith('Area')]
                                    norm_cols = [c for c in numeric_cols if '_norm' in c]

                            top_5_markers = []
                            top_5_names = []
                            
                            if norm_cols:
                                cluster_means = subset[norm_cols].mean()
                                top_5_markers = cluster_means.nlargest(5).index.tolist()
                                # Clean names for display: remove _norm_resolve, _resolve, _norm, etc.
                                top_5_names = [c.replace('_norm_resolve', '').replace('_resolve', '').replace('_norm', '').replace('_corr', '').replace('_unmixed', '') for c in top_5_markers]
                                st.info(f"Top expressed markers for {selected_pheno_vv}: {', '.join(top_5_names)}")
                            else:
                                st.warning("Could not determine marker columns for expression context.")
                                
                            if top_5_names:
                                # 2. Sample Cells (Group by batch)
                                batch_col = st.session_state.get('resolve_batch_col', 'batch_id')
                                if not batch_col or batch_col not in df_results.columns: # Retry find
                                     batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                                
                                if batch_col not in df_results.columns:
                                    st.error(f"Batch column '{batch_col}' not found in results.")
                                    subset_valid = pd.DataFrame()
                                elif not st.session_state.get('resolve_viz_file_matches'):
                                    st.error("No matching image files found for visualization batches.")
                                    subset_valid = pd.DataFrame()
                                else:
                                    valid_batches = list(st.session_state.resolve_viz_file_matches.keys())
                                    subset_valid = subset[subset[batch_col].astype(str).isin(valid_batches)]
                                    if resolve_batch_scope == "Specific batch" and selected_batch_vv:
                                        subset_valid = subset_valid[subset_valid[batch_col].astype(str) == str(selected_batch_vv)]
                                
                                if len(subset_valid) == 0:
                                    st.warning("No cells found in batches with available images.")
                                else:
                                    n_actual = min(n_viz_cells, len(subset_valid))
                                    sampled = subset_valid.sample(n=n_actual, random_state=int(time.time()))
                                    sampled = sampled.sort_values(batch_col)
                                    sampled_batches = sampled[batch_col].astype(str).drop_duplicates().tolist()
                                    prewarm_fn = getattr(viz, 'prewarm_batch', None)
                                    if callable(prewarm_fn):
                                        with st.spinner(f"Preloading {len(sampled_batches)} batch cache(s)..."):
                                            for prewarm_batch_id in sampled_batches:
                                                prewarm_fn(prewarm_batch_id, requested_channels=top_5_markers)
                                    
                                    st.markdown("---")
                                    cols = st.columns(3)
                                    prog_bar = st.progress(0)
                                    
                                    # X/Y cols
                                    x_col_name = next((c for c in ['x_centroid', 'AreaShape_Center_X', 'X_centroid', 'X'] if c in df_results.columns), 'x_centroid')
                                    y_col_name = next((c for c in ['y_centroid', 'AreaShape_Center_Y', 'Y_centroid', 'Y'] if c in df_results.columns), 'y_centroid')
                                    tile_sz = 40
                                    
                                    cid_col = st.session_state.get('resolve_cell_id_col', 'cell_id')

                                    for idx, (_, row) in enumerate(sampled.iterrows()):
                                        try:
                                            bid = str(row[batch_col])
                                            x, y = row[x_col_name], row[y_col_name]
                                            
                                            cell_data = viz.get_cell_context(
                                                bid,
                                                x,
                                                y,
                                                tile_size=tile_sz,
                                                requested_channels=top_5_markers,
                                            )
                                            
                                            # Robust Name Matching
                                            ome_channels = cell_data.get('channel_names', [])
                                            marker_indices_for_plot = []
                                            names_for_plot = []
                                            
                                            if ome_channels:
                                                ome_norm = [c.lower().strip() for c in ome_channels]
                                                for m in top_5_markers:
                                                    raw_m = m.replace('_norm', '').replace('_corr', '').replace('_unmixed', '')
                                                    target = raw_m.lower().strip()
                                                    
                                                    found_idx = -1
                                                    if target in ome_norm:
                                                        found_idx = ome_norm.index(target)
                                                    elif any(target in c or c in target for c in ome_norm):
                                                        for i, c in enumerate(ome_norm):
                                                            if target in c or c in target:
                                                                found_idx = i
                                                                break
                                                    
                                                    if found_idx != -1:
                                                        marker_indices_for_plot.append(found_idx)
                                                        names_for_plot.append(raw_m)
                                            
                                            if not marker_indices_for_plot:
                                                 marker_indices_for_plot = [0, 1, 2, 3, 4]
                                                 names_for_plot = [f"Ch{i}" for i in marker_indices_for_plot]
                                            
                                            fig = viz.generate_visualization(
                                                cell_data, 
                                                marker_indices_for_plot, 
                                                names_for_plot,
                                                display_size=tile_sz,
                                                show_neighbors=show_neighbors
                                            )
                                            
                                            caption_txt = f"Cell: {row.get(cid_col, 'Unknown')}\nBatch: {bid}\nLoc: ({x:.1f}, {y:.1f})"
                                            
                                            with cols[idx % 3]:
                                                st.pyplot(fig)
                                                st.text(caption_txt)
                                            
                                            # Auto-Save
                                            try:
                                                save_root = project_root / 'outputs/phenotyping/visual_validation'
                                                save_root.mkdir(parents=True, exist_ok=True)
                                                
                                                clean_pheno = str(selected_pheno_vv).replace(' ', '_').replace('/', '-')
                                                fname = f"{bid}_{row.get(cid_col, 'cell')}_{clean_pheno}.png"
                                                save_path = save_root / fname
                                                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                                                
                                                csv_path = save_root / "visualized_cells_metadata.csv"
                                                meta_rec = {
                                                    'timestamp': datetime.now().isoformat(),
                                                    'filename': fname,
                                                    'batch_id': bid,
                                                    'cell_id': row.get(cid_col, ''),
                                                    'phenotype': selected_pheno_vv, # Direct phenotype
                                                    'method': 'resolve',
                                                    'x': x, 'y': y,
                                                    'top_markers': ", ".join(names_for_plot)
                                                }
                                                pd.DataFrame([meta_rec]).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                                            except Exception as e:
                                                _log_console(f"Auto-save error: {e}", level="error")
                                            
                                            plt.close(fig)
                                        except Exception as e:
                                            st.error(f"Error viewing cell: {e}")
                                        
                                        prog_bar.progress((idx + 1) / n_actual)
                                    st.success(f"Visualized {n_actual} cells. Output saved to `outputs/phenotyping/visual_validation/`")
