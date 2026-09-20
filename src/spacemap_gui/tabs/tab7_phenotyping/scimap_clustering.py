"""
SCIMAP Phenotyping Subtab
Notebook-aligned SCIMAP workflow (Log1p → Rescale → Gate) with offline tables.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from .common import (
    logger,
    Timer,
    get_batch_id_column_help_text,
    get_cell_id_column_help_text,
    get_marker_list,
    get_marker_selection_help_text,
    get_marker_workspace_panel,
    render_method_marker_selection,
    normalize_optional_column_name,
    get_output_column_help_text,
    project_root_calc,
    _normalize_marker_name,
    _extract_thresholds_from_csv,
    _load_thresholds_table,
    _threshold_table_has_values,
    load_universal_gating_csv,
    build_marker_source_note,
    get_analysis_marker_label,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_export_split_data,
    render_data_filter_ui,
    apply_data_filter,
    render_settings_export_import,
    render_phenotyping_run_log_controls,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    resolve_notebook_marker_defaults,
    THRESHOLD_MODE_CHOICES,
    _THRESHOLD_MODE_TO_KEY,
    AUTO_THRESHOLD_METHODS,
    compute_auto_thresholds,
)

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

try:
    from spacemap import ph_scimap_utils
    sys.modules["ph_scimap_utils"] = ph_scimap_utils
    from spacemap.ph_scimap_utils import manual_rescale_scimap
except ImportError:
    import importlib.util
    try:
        # Check parents[3] / "spacemap" / "ph_scimap_utils.py" (current workspace package structure)
        _PH_SCIMAP_PATH = Path(__file__).resolve().parents[3] / "spacemap" / "ph_scimap_utils.py"
        if not _PH_SCIMAP_PATH.exists():
            # Check parents[4] / "src" / "spacemap" / "ph_scimap_utils.py"
            _PH_SCIMAP_PATH = Path(__file__).resolve().parents[4] / "src" / "spacemap" / "ph_scimap_utils.py"
        if not _PH_SCIMAP_PATH.exists():
            # Check legacy V13 path
            _PH_SCIMAP_PATH = Path(__file__).resolve().parents[3] / "src" / "ph_scimap_utils.py"

        if _PH_SCIMAP_PATH.exists():
            _ph_spec = importlib.util.spec_from_file_location("ph_scimap_utils", _PH_SCIMAP_PATH)
            if _ph_spec is not None and _ph_spec.loader is not None:
                ph_scimap_utils = importlib.util.module_from_spec(_ph_spec)
                sys.modules.setdefault("ph_scimap_utils", ph_scimap_utils)
                _ph_spec.loader.exec_module(ph_scimap_utils)
                manual_rescale_scimap = ph_scimap_utils.manual_rescale_scimap
            else:
                ph_scimap_utils = None
                manual_rescale_scimap = None
        else:
            ph_scimap_utils = None
            manual_rescale_scimap = None
    except Exception:
        ph_scimap_utils = None
        manual_rescale_scimap = None
except Exception:
    ph_scimap_utils = None
    manual_rescale_scimap = None

try:
    import sys
    import io
    _orig_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        import scimap as sm
    finally:
        sys.stdout = _orig_stdout
    import anndata
    import scanpy as sc
    _HAS_SCIMAP = True
except Exception:
    sm = None
    anndata = None
    sc = None
    _HAS_SCIMAP = False


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _extract_markers_base_from_step6d() -> list[str]:
    """Extract MARKERS_BASE from Step6D_phenotype_SCIMAP.ipynb."""
    try:
        notebook_path = project_root_calc / "notebooks" / "Step6D_phenotype_SCIMAP.ipynb"
        if not notebook_path.exists():
            return []

        with open(notebook_path, "r") as f:
            nb = json.load(f)

        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            if not source or "MARKERS_BASE" not in source:
                continue
            match = re.search(r"MARKERS_BASE\s*=\s*\[(.*?)\]", source, re.DOTALL)
            if match:
                try:
                    list_str = "[" + match.group(1) + "]"
                    markers = ast.literal_eval(list_str)
                    return markers if isinstance(markers, list) else []
                except Exception:
                    pass
        return []
    except Exception as e:
        print(f"Warning: Could not extract MARKERS_BASE from Step6D: {e}")
        return []


def _extract_scimap_config() -> dict:
    """Extract SCIMAP configuration parameters from Step6D notebook."""
    try:
        notebook_path = project_root_calc / "notebooks" / "Step6D_phenotype_SCIMAP.ipynb"
        if not notebook_path.exists():
            return {}

        with open(notebook_path, "r") as f:
            nb = json.load(f)

        config = {
            "threshold_mode": "Predefined (CSV)",
            "threshold_statistic": "median",
            "gate_value": 0.5,
            "batch_col": "batch_id",
            "cell_id_col": "cell_id",
            "output_col": "phenotype_scimap",
            "log1p": True,
        }

        for cell in nb.get("cells", []):
            source = "".join(cell.get("source", []))
            if not source:
                continue

            # Legacy notebook variable → new threshold_mode
            if "APPLY_THRESHOLDS" in source:
                if "False" in source:
                    config["threshold_mode"] = "None (No Thresholds)"
                elif "True" in source:
                    config["threshold_mode"] = "Predefined (CSV)"

            if "THRESHOLD_STATISTIC" in source:
                match = re.search(r"THRESHOLD_STATISTIC\s*=\s*[\"\']([^\"\']+)[\"\']", source)
                if match:
                    config["threshold_statistic"] = match.group(1)

            if "GATE_VALUE" in source:
                match = re.search(r"GATE_VALUE\s*=\s*([0-9\.]+)", source)
                if match:
                    try:
                        config["gate_value"] = float(match.group(1))
                    except Exception:
                        pass

            if "BATCH_COL" in source:
                match = re.search(r"BATCH_COL\s*=\s*[\"\']([^\"\']+)[\"\']", source)
                if match:
                    config["batch_col"] = match.group(1)

            if "CELL_ID_COLUMN" in source:
                match = re.search(r"CELL_ID_COLUMN\s*=\s*[\"\']([^\"\']+)[\"\']", source)
                if match:
                    config["cell_id_col"] = match.group(1)

        return config
    except Exception as e:
        print(f"Warning: Could not extract config from Step6D: {e}")
        return {}


def _strategy_table_has_values(strategy_df: pd.DataFrame) -> bool:
    return strategy_df is not None and not strategy_df.empty


def _normalize_scimap_phenotype_name(value: object, fallback: str) -> str:
    """Return a non-empty phenotype label, defaulting to the original name."""
    if value is None or value is pd.NA:
        return fallback
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def _suggest_scimap_phenotype_name(original_name: str, saved_name: str) -> str:
    """Suggest a cleaner phenotype name without changing saved labels until apply."""
    if saved_name != original_name:
        return saved_name

    suggested_name = re.sub(r"(?i)^likely\s*-\s*", "", original_name).strip()
    return suggested_name or original_name


def _sync_scimap_phenotype_rename_map(
    df_results: pd.DataFrame,
    output_col: str,
) -> dict[str, str]:
    """Keep the SCIMAP phenotype rename map aligned with the current results."""
    if output_col not in df_results.columns:
        st.session_state["scimap_phenotype_rename_map"] = {}
        return {}

    existing_map = st.session_state.get("scimap_phenotype_rename_map", {})
    if not isinstance(existing_map, dict):
        existing_map = {}

    current_phenotypes = [
        str(name)
        for name in df_results[output_col].dropna().astype(str).unique().tolist()
        if str(name).strip()
    ]
    synced_map = {
        name: _normalize_scimap_phenotype_name(existing_map.get(name, name), name)
        for name in sorted(current_phenotypes)
    }
    st.session_state["scimap_phenotype_rename_map"] = synced_map
    return synced_map


def _build_scimap_phenotype_rename_editor_df(
    df_results: pd.DataFrame,
    output_col: str,
    rename_map: dict[str, str],
) -> pd.DataFrame:
    """Build the editable phenotype rename table shown in the SCIMAP results tab."""
    if output_col not in df_results.columns:
        return pd.DataFrame(
            columns=[
                "Phenotype Original",
                "Phenotype New Name/Suggestion",
                "Cell Count",
                "% Total",
            ]
        )

    counts = df_results[output_col].dropna().astype(str).value_counts()
    total_cells = int(counts.sum())
    rows: list[dict[str, object]] = []
    for phenotype_name, cell_count in counts.items():
        original_name = str(phenotype_name)
        rows.append(
            {
                "Phenotype Original": original_name,
                "Phenotype New Name/Suggestion": _suggest_scimap_phenotype_name(
                    original_name,
                    _normalize_scimap_phenotype_name(
                        rename_map.get(original_name, original_name),
                        original_name,
                    ),
                ),
                "Cell Count": int(cell_count),
                "% Total": f"{(float(cell_count) / total_cells * 100.0):.1f}%" if total_cells else "0.0%",
            }
        )
    return pd.DataFrame(rows)


def _render_scimap_phenotype_rename_table(
    df_results: pd.DataFrame,
    output_col: str,
) -> dict[str, str]:
    """Render the editable SCIMAP phenotype renaming table and return the saved mapping."""
    rename_map = _sync_scimap_phenotype_rename_map(df_results, output_col)
    editor_df = _build_scimap_phenotype_rename_editor_df(df_results, output_col, rename_map)

    st.markdown("#### Phenotype Statistics")
    st.caption(
        "Edit `Phenotype New Name/Suggestion` to rename or merge SCIMAP phenotypes. "
        "`Likely-` labels are prefilled as suggestions here, but the statistics table below keeps the original label until you click Apply."
    )

    if editor_df.empty:
        st.info("No phenotype assignments available yet.")
        return rename_map

    with st.form("scimap_phenotype_rename_form", clear_on_submit=False):
        edited_df = st.data_editor(
            editor_df,
            column_config={
                "Phenotype Original": st.column_config.TextColumn("Phenotype Original", disabled=True),
                "Phenotype New Name/Suggestion": st.column_config.TextColumn("Phenotype New Name/Suggestion"),
                "Cell Count": st.column_config.NumberColumn("Cell Count", disabled=True),
                "% Total": st.column_config.TextColumn("% Total", disabled=True),
            },
            disabled=["Phenotype Original", "Cell Count", "% Total"],
            hide_index=True,
            num_rows="fixed",
        )
        apply_names = st.form_submit_button("Apply Renamed Phenotypes", type="primary")

    if apply_names:
        updated_map: dict[str, str] = {}
        for row in edited_df.to_dict(orient="records"):
            original_name = str(row.get("Phenotype Original", "")).strip()
            if not original_name:
                continue
            updated_map[original_name] = _normalize_scimap_phenotype_name(
                row.get("Phenotype New Name/Suggestion"),
                original_name,
            )
        rename_map = {
            original_name: updated_map.get(original_name, original_name)
            for original_name in editor_df["Phenotype Original"].tolist()
        }
        st.session_state["scimap_phenotype_rename_map"] = rename_map
        st.success("Phenotype names updated.")

    return st.session_state.get("scimap_phenotype_rename_map", rename_map)


def _apply_scimap_phenotype_renames(
    df_results: pd.DataFrame,
    output_col: str,
    cluster_col: str,
    rename_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[int, str], list[str], str, dict[int, str]]:
    """Apply phenotype renames and prepare separate maps for export and heatmap display."""
    renamed_df = df_results.copy()
    if output_col not in renamed_df.columns:
        return renamed_df, {}, [], cluster_col, {}

    renamed_df[output_col] = renamed_df[output_col].apply(
        lambda value: _normalize_scimap_phenotype_name(
            rename_map.get(str(value), value),
            str(value),
        )
        if pd.notna(value)
        else value
    )

    export_phenotype_map: dict[int, str] = {}
    if cluster_col in renamed_df.columns:
        cluster_mapping_df = renamed_df[[cluster_col, output_col]].dropna(subset=[cluster_col]).drop_duplicates(subset=[cluster_col])
        for cluster_id, phenotype_name in zip(cluster_mapping_df[cluster_col], cluster_mapping_df[output_col]):
            if pd.notna(cluster_id):
                export_phenotype_map[int(cluster_id)] = str(phenotype_name)

    phenotype_names = sorted(
        [name for name in renamed_df[output_col].dropna().astype(str).unique().tolist() if name.strip()]
    )
    heatmap_cluster_col = f"{cluster_col}__renamed_display"
    pheno_to_display_cluster = {name: idx for idx, name in enumerate(phenotype_names)}
    renamed_df[heatmap_cluster_col] = renamed_df[output_col].map(pheno_to_display_cluster).fillna(-1)
    heatmap_phenotype_map = {int(idx): name for name, idx in pheno_to_display_cluster.items()}
    return renamed_df, export_phenotype_map, phenotype_names, heatmap_cluster_col, heatmap_phenotype_map


def _build_scimap_batch_norm_heatmap_data(
    df_results: pd.DataFrame,
    cluster_col: str,
    raw_marker_cols: list[str],
    batch_col: str,
) -> pd.DataFrame:
    """Build batch-normalized marker means per phenotype for SCIMAP summaries."""
    if cluster_col not in df_results.columns:
        return pd.DataFrame()

    valid_marker_cols = [
        marker_name
        for marker_name in raw_marker_cols
        if marker_name in df_results.columns and pd.api.types.is_numeric_dtype(df_results[marker_name])
    ]
    if not valid_marker_cols:
        return pd.DataFrame()

    df_valid = df_results[df_results[cluster_col].notna()].copy()
    if df_valid.empty:
        return pd.DataFrame()

    def _zscore_group_sci(series: pd.Series) -> pd.Series:
        std_dev = series.std()
        return (series - series.mean()) / (std_dev if std_dev > 0 else 1e-9)

    if batch_col in df_valid.columns:
        raw_z_sci = df_valid.groupby(batch_col, group_keys=False)[valid_marker_cols].transform(_zscore_group_sci)
    else:
        raw_z_sci = df_valid[valid_marker_cols].transform(_zscore_group_sci)

    raw_z_sci[cluster_col] = df_valid[cluster_col].values
    mean_expr_sci = raw_z_sci.groupby(cluster_col)[valid_marker_cols].mean()
    if isinstance(mean_expr_sci, pd.Series):
        mean_expr_sci = mean_expr_sci.to_frame()
    mean_expr_sci.columns = [get_analysis_marker_label(column_name) for column_name in mean_expr_sci.columns]
    return mean_expr_sci.T


def _find_threshold_file(batch_id: str, threshold_dir: str | Path) -> Path | None:
    if ph_scimap_utils and hasattr(ph_scimap_utils, "find_roi_threshold_file"):
        try:
            found = ph_scimap_utils.find_roi_threshold_file(batch_id, threshold_dir)
            return Path(found) if found else None
        except Exception:
            pass

    threshold_dir = Path(threshold_dir)
    base = str(batch_id)
    for delim in [".ome_cell_features", ".ome", "_cell_features", "_annotated", "_processed", "_summary"]:
        if delim in base:
            base = base.split(delim)[0]
            break
    base = base.rstrip("._ ")
    candidates = [
        threshold_dir / f"{base}.ome.tiff_-_Merged_Image.csv",
        threshold_dir / f"{base}_-_Merged_Image.csv",
        threshold_dir / f"{base}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _thresholds_from_table(
    thresholds_table: pd.DataFrame,
    batch_id: str,
    markers: list[str],
) -> dict:
    if thresholds_table is None or thresholds_table.empty:
        return {}

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

    if "batch_id" in thresholds_table.columns:
        row_match = thresholds_table[thresholds_table["batch_id"].astype(str) == str(batch_id)]
        if row_match.empty:
            row_match = thresholds_table[thresholds_table["batch_id"].astype(str).str.startswith(str(batch_id))]
        if row_match.empty:
            base = str(batch_id)
            for delim in [".ome_cell_features", ".ome", "_cell_features", "_annotated", "_processed", "_summary"]:
                if delim in base:
                    base = base.split(delim)[0]
                    break
            base = base.rstrip("._ ")
            if base:
                row_match = thresholds_table[thresholds_table["batch_id"].astype(str).str.contains(base, regex=False)]
        if row_match.empty and "Threshold_File" in thresholds_table.columns:
            base = str(batch_id)
            for delim in [".ome_cell_features", ".ome", "_cell_features", "_annotated", "_processed", "_summary"]:
                if delim in base:
                    base = base.split(delim)[0]
                    break
            base = base.rstrip("._ ")
            if base:
                row_match = thresholds_table[thresholds_table["Threshold_File"].astype(str).str.contains(base, regex=False)]
        if row_match.empty:
            norm_batch = _normalize_batch_id(batch_id)
            if norm_batch:
                norm_series = thresholds_table["batch_id"].astype(str).map(_normalize_batch_id)
                row_match = thresholds_table[norm_series == norm_batch]
        if row_match.empty and "Threshold_File" in thresholds_table.columns:
            norm_batch = _normalize_batch_id(batch_id)
            if norm_batch:
                    norm_files = thresholds_table["Threshold_File"].astype(str).map(_normalize_batch_id)
                    row_match = thresholds_table[norm_files == norm_batch]
        if row_match.empty:
            norm_batch = _normalize_batch_id(batch_id)
            if norm_batch:
                    norm_series = thresholds_table["batch_id"].astype(str).map(_normalize_batch_id)
                    row_match = thresholds_table[norm_series.str.contains(norm_batch, regex=False)]
        if row_match.empty and "Threshold_File" in thresholds_table.columns:
            norm_batch = _normalize_batch_id(batch_id)
            if norm_batch:
                    norm_files = thresholds_table["Threshold_File"].astype(str).map(_normalize_batch_id)
                    row_match = thresholds_table[norm_files.str.contains(norm_batch, regex=False)]
        if row_match.empty:
            # Fuzzy fallback using difflib
            from difflib import get_close_matches as _gcm
            norm_batch = _normalize_batch_id(batch_id)
            if norm_batch:
                _candidates = thresholds_table["batch_id"].astype(str).map(_normalize_batch_id).tolist()
                _fuzzy = _gcm(norm_batch, _candidates, n=1, cutoff=0.6)
                if _fuzzy:
                    _fn = _candidates.index(_fuzzy[0])
                    row_match = thresholds_table.iloc[[_fn]]
        if row_match.empty:
            return {}
        row = row_match.iloc[0]
    else:
        row = thresholds_table.iloc[0]

    thresholds = {}
    for m in markers:
        base = _normalize_marker_name(m)
        val = row.get(m)
        if pd.isna(val):
            val = row.get(base, np.nan)
        if pd.isna(val):
            continue
        try:
            thresholds[base] = float(val)
        except Exception:
            continue
    return thresholds


def _render_scimap_help_panel() -> None:
    with st.expander("SCIMAP Quick Guide: Workflow, Strategy Terms, and Outputs", expanded=False):
        st.markdown(
            "SCIMAP in this tab follows the same pattern as the notebook: `log1p -> rescale -> phenotype_cells`. "
            "After rescaling, values above the gate, usually `0.5`, are treated as positive and values below it are treated as negative."
        )
        st.markdown(
            "1. Choose the input file and confirm the marker list.\n"
            "2. Load, import, or compute thresholds so raw intensity becomes a meaningful positive/negative signal.\n"
            "3. Build the phenotype strategy table, where each row describes one phenotype with marker rules.\n"
            "4. Run SCIMAP to assign a phenotype label to each cell.\n"
            "5. Review the results, rename or merge phenotypes if needed, inspect cells, and export the output."
        )
        st.markdown(
            "| Strategy value | Meaning in the phenotype table | Typical use |\n"
            "| --- | --- | --- |\n"
            "| `pos` | Marker should be positive for this phenotype. With multiple `pos` markers, partial matches can still surface as likely or intermediate assignments for later review. | Softer defining markers or review-driven refinement. |\n"
            "| `neg` | Marker should be negative for this phenotype. | Excluding nearby phenotypes that share many positive markers. |\n"
            "| `allpos` | All markers marked `allpos` in that row must be positive. | Strict core identity markers. |\n"
            "| `allneg` | All markers marked `allneg` in that row must be negative. | Strict exclusion rules. |\n"
            "| `anypos` | At least one marker marked `anypos` in that row must be positive. | Lineage pools where one of several markers is enough. |\n"
            "| `anyneg` | At least one marker marked `anyneg` in that row must be negative. | Flexible exclusion when one negative marker is sufficient. |"
        )
        st.markdown(
            "Useful strategy-table rules:\n"
            "- Leave a marker cell blank when that marker should not participate in that phenotype row.\n"
            "- Prefer positive markers when possible; negatives work best as exclusion criteria.\n"
            "- Use `allpos` and `allneg` for strict identities, and `anypos` / `anyneg` for broader lineage logic.\n"
            "- If your workflow file uses a group column, rows under `all` apply globally while subgroup rows can refine broader classes."
        )
        st.markdown(
            "Outputs from this panel:\n"
            "- The phenotype label column, usually `phenotype_scimap`, is the per-cell result used downstream.\n"
            "- `scimap_cluster` keeps the numeric phenotype id for traceability and save-back operations.\n"
            "- The Heatmap & Statistics section uses the current phenotype names, including any renamed or merged labels from `Phenotype New Name`.\n"
            "- Cell visualization lets you inspect representative cells for each phenotype to verify thresholds and strategy rows.\n"
            "- Export and save-back use the phenotype label column for downstream analysis; keeping the numeric cluster column is useful as an audit trail."
        )


def render_scimap(project_root: Path, global_input_path: str | None) -> None:
    st.subheader("SCIMAP Phenotyping")
    render_phenotyping_run_log_controls(Path(project_root), "scimap", "SCIMAP Phenotyping")
    st.markdown("Notebook-aligned SCIMAP workflow (Log1p → Rescale → Gate).")
    _render_scimap_help_panel()

    cache_dir = project_root / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scimap_cache = cache_dir / "scimap_params.json"
    scimap_thresholds_cache = cache_dir / "scimap_thresholds_table.csv"
    scimap_strategy_cache = cache_dir / "scimap_strategy_table.csv"

    cached_params: dict = {}
    if scimap_cache.exists():
        try:
            with open(scimap_cache, "r") as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}

    # --- Settings Export / Import ---
    render_settings_export_import(
        "SCIMAP",
        ["scimap_"],
        {
            "scimap_params.json": scimap_cache,
            "scimap_thresholds_table.csv": scimap_thresholds_cache,
            "scimap_strategy_table.csv": scimap_strategy_cache,
        },
    )

    notebook_config = _extract_scimap_config()

    st.markdown("##### Input File Selection")
    use_global = st.checkbox("Use Global Input File", value=cached_params.get("use_global", True), key="scimap_use_global")
    full_input_path = global_input_path

    if not use_global:
        bc_dir = cached_params.get("bc_dir", str(project_root / "outputs/batch_corrected"))
        bc_dir = st.text_input("Batch Corrected Directory", bc_dir, key="scimap_bc_dir")
        if os.path.exists(bc_dir):
            files = [f for f in os.listdir(bc_dir) if f.endswith(".csv")]
            if files:
                sel = st.selectbox("Input File", files, key="scimap_infile")
                full_input_path = os.path.join(bc_dir, sel)

    available_columns: list[str] = []
    if full_input_path and os.path.exists(full_input_path):
        st.caption(f"Active File: {os.path.basename(full_input_path)}")
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            available_columns = list(df_head.columns)
        except Exception:
            available_columns = []

    st.divider()
    st.markdown("### 2. Marker Selection")
    st.caption("Select markers for SCIMAP phenotyping. Auto-selected from Step6D notebook (MARKERS_BASE).")

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

    # AUTO-SELECT MARKERS FROM NOTEBOOK (Step6D_phenotype_SCIMAP.ipynb)
    if full_input_path and os.path.exists(full_input_path):
        markers_base = _extract_markers_base_from_step6d()

        marker_workspace = st.session_state.get("tab7_marker_workspace", {})
        expected_markers = resolve_notebook_marker_defaults(
            markers_base,
            [],
            all_marker_candidates,
            str(marker_workspace.get("source_mode", "Raw markers")),
            str(marker_workspace.get("affix_position", "Suffix")),
            str(marker_workspace.get("correction_affix", "_corr")),
        )

    else:
        expected_markers = []

    _, selected_markers, marker_workspace_note = render_method_marker_selection(
        key_prefix="scimap",
        method_name="SCIMAP",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers,
        saved_markers=st.session_state.get("scimap_marker_bases", []),
    )

    if not selected_markers:
        st.warning("No markers selected. Please select markers above before running.")

    st.divider()
    st.markdown("### 3. Configuration (Notebook Synchronized)")

    if "scimap_config" not in st.session_state:
        # Migrate legacy 'apply_thresholds' boolean → threshold_mode string
        _legacy_apply = cached_params.get("apply_thresholds", notebook_config.get("apply_thresholds", True))
        _default_mode = cached_params.get(
            "threshold_mode",
            notebook_config.get("threshold_mode", "Predefined (CSV)" if _legacy_apply else "None (No Thresholds)"),
        )
        st.session_state["scimap_config"] = {
            "threshold_mode": _default_mode,
            "threshold_statistic": notebook_config.get("threshold_statistic", "median"),
            "gate_value": notebook_config.get("gate_value", 0.5),
            "batch_col": cached_params.get("batch_col", notebook_config.get("batch_col", "batch_id")),
            "cell_id_col": cached_params.get("cell_id_col", notebook_config.get("cell_id_col", "cell_id")),
            "output_col": cached_params.get("output_col", "phenotype_scimap"),
            "log1p": cached_params.get("log1p", True),
            "unclassified_label": cached_params.get("unclassified_label", "unclassified"),
        }

    with st.expander("SCIMAP Configuration", expanded=True):
        with st.form("scimap_config_form", clear_on_submit=False):
            cfg_col1, cfg_col2 = st.columns([1, 1])
            with cfg_col1:
                _scimap_mode_choices = ["None (No Thresholds)"] + THRESHOLD_MODE_CHOICES
                current_mode = st.session_state["scimap_config"].get("threshold_mode", "Predefined (CSV)")
                _mode_idx = _scimap_mode_choices.index(current_mode) if current_mode in _scimap_mode_choices else 1
                threshold_mode = st.selectbox(
                    "Threshold Method",
                    _scimap_mode_choices,
                    index=_mode_idx,
                    key="scimap_threshold_mode",
                    help=(
                        "Defines how SCIMAP chooses the marker-specific threshold anchor before "
                        "`manual_rescale_scimap` maps values into the gated range used by `phenotype_cells`.\n\n"
                        "**None**: skips threshold subtraction and uses 0 in log-space as the anchor. "
                        "With `log1p` enabled, any non-zero expression becomes functionally positive after rescaling, "
                        "so this is only safe for already background-subtracted data.\n\n"
                        "**Predefined (CSV)**: loads external per-marker thresholds and applies them batch-wise when a valid Batch Column is available.\n\n"
                        "**Otsu**: maximizes between-class variance.\n\n"
                        "**Triangle**: works well for skewed one-sided histograms.\n\n"
                        "**Li**: minimum cross-entropy iterative threshold.\n\n"
                        "**Yen**: maximum entropy threshold.\n\n"
                        "**Mean**: uses the mean of positive values.\n\n"
                        "**IsoData**: Ridler-Calvard iterative threshold.\n\n"
                        "**Minimum**: smooths toward a bimodal histogram and picks the valley."
                    ),
                )
                threshold_stat = st.selectbox(
                    "Threshold Statistic (Predefined CSV only)",
                    ["median", "mean", "min"],
                    index=["median", "mean", "min"].index(st.session_state["scimap_config"]["threshold_statistic"])
                    if st.session_state["scimap_config"]["threshold_statistic"] in ["median", "mean", "min"]
                    else 0,
                    key="scimap_thresh_stat",
                    help=(
                        "Only used for Predefined (CSV). When a threshold file contains multiple values for the same marker, "
                        "they are reduced to one gate before SCIMAP rescaling. `median` is the safest default, `mean` is more "
                        "sensitive to high outliers, and `min` is the most permissive."
                    ),
                )
                gate_value = st.number_input(
                    "Gate Value",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state["scimap_config"]["gate_value"]),
                    step=0.01,
                    key="scimap_gate_value",
                    help=(
                        "Final positivity cutoff passed to SCIMAP after rescaling. The notebook default is `0.5`, which matches "
                        "the rescaling scheme where values below the threshold sit on the negative side and values above it sit on the positive side."
                    ),
                )
            with cfg_col2:
                log1p = st.checkbox(
                    "Apply log1p Transform",
                    value=st.session_state["scimap_config"]["log1p"],
                    key="scimap_log1p",
                    help=(
                        "Applies `scanpy.pp.log1p` before thresholding and SCIMAP rescaling. Leave this on for raw positive intensities "
                        "and notebook-style threshold files; turn it off only if the selected marker columns are already log-transformed."
                    ),
                )
                output_col = st.text_input(
                    "Phenotype Output Column",
                    value=st.session_state["scimap_config"]["output_col"],
                    key="scimap_output_col",
                    help=get_output_column_help_text("SCIMAP phenotype labels"),
                )
                unclassified_label = st.text_input(
                    "Unclassified Label",
                    value=st.session_state["scimap_config"]["unclassified_label"],
                    key="scimap_unclassified_label",
                    help=(
                        "Label written when a cell does not satisfy any phenotype rule at the chosen gate. This value appears in the output column, "
                        "statistics, and exports."
                    ),
                )

            submitted_cfg = st.form_submit_button("Save Configuration", type="primary")
            if submitted_cfg:
                st.session_state["scimap_config"].update(
                    {
                        "threshold_mode": threshold_mode,
                        "threshold_statistic": threshold_stat,
                        "gate_value": gate_value,
                        "log1p": log1p,
                        "output_col": output_col,
                        "unclassified_label": unclassified_label,
                    }
                )
                params_to_save = {
                    "batch_col": st.session_state["scimap_config"]["batch_col"],
                    "cell_id_col": st.session_state["scimap_config"]["cell_id_col"],
                    "output_col": st.session_state["scimap_config"]["output_col"],
                    "threshold_statistic": st.session_state["scimap_config"]["threshold_statistic"],
                    "threshold_mode": st.session_state["scimap_config"]["threshold_mode"],
                    "log1p": st.session_state["scimap_config"]["log1p"],
                    "unclassified_label": st.session_state["scimap_config"]["unclassified_label"],
                }
                try:
                    cached_params.update(params_to_save)
                    with open(scimap_cache, "w") as f:
                        json.dump(cached_params, f, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                    st.success("Configuration saved.")
                except Exception as e:
                    st.error(f"Failed to save configuration: {e}")

    if available_columns:
        default_batch = "batch_id" if "batch_id" in available_columns else available_columns[0]
        batch_options = [""] + available_columns
        batch_col = st.selectbox(
            "Batch Column",
            batch_options,
            index=batch_options.index(default_batch) if default_batch in batch_options else 0,
            format_func=lambda value: value or "(None)",
            key="scimap_batch_col",
            help=get_batch_id_column_help_text(),
        )
        default_cell_id = "cell_id" if "cell_id" in available_columns else available_columns[0]
        cell_id_options = [""] + available_columns
        cell_id_col = st.selectbox(
            "Cell ID Column",
            cell_id_options,
            index=cell_id_options.index(default_cell_id) if default_cell_id in cell_id_options else 0,
            format_func=lambda value: value or "(None)",
            key="scimap_cell_id_col",
            help=get_cell_id_column_help_text(),
        )
    else:
        batch_col = st.text_input(
            "Batch Column",
            value=st.session_state["scimap_config"].get("batch_col") or "",
            key="scimap_batch_col",
            help=get_batch_id_column_help_text(),
        )
        cell_id_col = st.text_input(
            "Cell ID Column",
            value=st.session_state["scimap_config"].get("cell_id_col") or "",
            key="scimap_cell_id_col",
            help=get_cell_id_column_help_text(),
        )

    normalized_batch_col = normalize_optional_column_name(batch_col)
    normalized_cell_id_col = normalize_optional_column_name(cell_id_col)
    st.session_state["scimap_config"]["batch_col"] = normalized_batch_col
    st.session_state["scimap_config"]["cell_id_col"] = normalized_cell_id_col
    if cached_params.get("batch_col") != normalized_batch_col or cached_params.get("cell_id_col") != normalized_cell_id_col:
        try:
            cached_params["batch_col"] = normalized_batch_col
            cached_params["cell_id_col"] = normalized_cell_id_col
            with open(scimap_cache, "w") as f:
                json.dump(cached_params, f, indent=2)
        except Exception:
            pass

    # --- Data Filtering ---
    scimap_filter_settings = render_data_filter_ui(
        available_columns=available_columns,
        key_prefix="scimap_filt",
        cached_params=st.session_state.get("scimap_config", {}),
        full_input_path=full_input_path,
        show_cell_id=False,
    )
    st.session_state["scimap_config"]["enable_filter"] = scimap_filter_settings.get("enable_filter", False)
    st.session_state["scimap_config"]["filter_col"] = scimap_filter_settings.get("filter_col")
    st.session_state["scimap_config"]["filter_val"] = scimap_filter_settings.get("filter_val")

    st.divider()
    st.markdown("### 4. Thresholds")

    # --- Determine current threshold mode ---
    _scimap_mode_map = {**_THRESHOLD_MODE_TO_KEY, "None (No Thresholds)": "none"}
    _active_mode = st.session_state["scimap_config"].get("threshold_mode", "Predefined (CSV)")
    _mode_key = _scimap_mode_map.get(_active_mode, "predefined")
    _is_predefined = _mode_key == "predefined"
    _is_none = _mode_key == "none"
    _is_auto = not _is_predefined and not _is_none

    if _is_none:
        st.warning(
            "**No Thresholds mode:** Only use this mode for already background-subtracted data. "
            "Without thresholding, any non-zero value after log1p becomes functionally positive, "
            "causing widespread false positives in raw IMC/IF data due to autofluorescence. "
            "For raw data, use *Predefined (CSV)* or an Auto-threshold method."
        )
    elif _is_predefined:
        st.caption("Load predefined thresholds from CSV files (e.g. computed in QuPath).")
    else:
        st.caption(
            f"**{_active_mode}** auto-threshold will be computed per batch from cellular mean intensities. "
            "Click *Compute Thresholds* to preview, then edit if needed before running the pipeline."
        )

    # --- Predefined CSV controls (only when mode is CSV) ---
    default_thresh_dir = project_root / "data/thresholds"
    if "scimap_threshold_dir" not in st.session_state:
        st.session_state["scimap_threshold_dir"] = cached_params.get("threshold_dir", str(default_thresh_dir))

    if _is_predefined:
        thresh_dir = st.text_input(
            "Thresholds Directory",
            key="scimap_threshold_dir",
            help=(
                "Folder containing Predefined (CSV) threshold files. Filenames are matched to batch IDs with a tolerant stem matcher, "
                "so QuPath-style names usually work without manual renaming."
            ),
        )
        if cached_params.get("threshold_dir") != thresh_dir:
            try:
                cached_params["threshold_dir"] = thresh_dir
                with open(scimap_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass

        tcfg1, tcfg2, tcfg3 = st.columns([1, 1, 1])
        with tcfg1:
            marker_col_name = st.text_input(
                "Threshold Marker Column (optional)",
                value="",
                key="scimap_threshold_marker_col",
                help=(
                    "Optional override for the marker-name column in threshold CSVs. Leave blank to auto-detect names such as `marker` or `markers`."
                ),
            )
        with tcfg2:
            value_col_name = st.text_input(
                "Threshold Value Column (optional)",
                value="",
                key="scimap_threshold_value_col",
                help=(
                    "Optional override for the numeric threshold column in threshold CSVs. Leave blank to auto-detect columns such as `threshold`, `value`, `cutoff`, or `gate`."
                ),
            )
        with tcfg3:
            polygon_prefix = st.text_input(
                "Polygon Column Prefix",
                value="Polygon",
                key="scimap_polygon_prefix",
                help=(
                    "Used when a threshold CSV stores one value per polygon column instead of a single threshold column. "
                    "Columns whose names start with this prefix are aggregated into one threshold per marker."
                ),
            )
    else:
        # Preserve widget keys so they don't vanish on mode switch
        thresh_dir = st.session_state.get("scimap_threshold_dir", str(default_thresh_dir))
        marker_col_name = ""
        value_col_name = ""
        polygon_prefix = "Polygon"

    # --- Action buttons row ---
    if _is_none:
        load_thresh = False
        add_row = False
        compute_auto = False
    elif _is_predefined:
        tcol1, tcol2, tcol3 = st.columns([1, 1, 2])
        load_thresh = tcol1.button("Load Thresholds", key="scimap_load_thresholds")
        add_row = tcol2.button("Add Row", key="scimap_add_threshold_row")
        tcol3.caption("Review and edit threshold values before running.")
        compute_auto = False

        # Import from Threshold tab CSV
        with st.expander("Import from Threshold Tab CSV", expanded=False):
            st.caption("Import thresholds from a CSV file created in the Threshold gating tool.")
            import_file = st.file_uploader(
                "Upload Threshold CSV",
                type=["csv"],
                key="scimap_import_thresh_csv",
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
                            base_marker = _normalize_marker_name(marker)
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
                            merged = {}
                            for r in rows:
                                bid = r["batch_id"]
                                if bid not in merged:
                                    merged[bid] = {"batch_id": bid, "Threshold_File": "imported"}
                                for k, v in r.items():
                                    if k not in ["batch_id", "Threshold_File"]:
                                        merged[bid][k] = v
                            imported_df = pd.DataFrame(list(merged.values()))
                            st.session_state["scimap_threshold_table_draft"] = imported_df
                            st.success(f"Imported {len(merged)} batch(es) from Threshold CSV.")
                            corr_markers = [m for m in selected_markers if m.endswith("_corr")]
                            import_corr = [str(row["marker"]) for _, row in import_df.iterrows() if str(row["marker"]).endswith("_corr")]
                            if import_corr and not corr_markers:
                                st.warning("Imported CSV contains corrected markers (_corr) but selected markers are raw.")
                            elif corr_markers and not import_corr:
                                st.warning("Selected markers are corrected (_corr) but imported CSV has raw markers.")
                        else:
                            st.warning("No matching markers found in import file.")
                except Exception as e:
                    st.error(f"Failed to import: {e}")
    else:
        tcol1, tcol2, tcol3 = st.columns([1, 1, 2])
        compute_auto = tcol1.button(
            f"🔬 Compute {_active_mode} Thresholds",
            key="scimap_compute_auto_thresholds",
            type="primary",
        )
        add_row = tcol2.button("Add Row", key="scimap_add_threshold_row_auto")
        tcol3.caption("Thresholds computed per batch. Edit in the table below before running.")
        load_thresh = False

    # --- Load predefined thresholds from CSV ---
    if load_thresh:
        batch_values = []
        if full_input_path and os.path.exists(full_input_path) and batch_col:
            try:
                batch_values = pd.read_csv(full_input_path, usecols=[batch_col])[batch_col].dropna().unique().tolist()
            except Exception:
                batch_values = []
        st.session_state["scimap_threshold_table_draft"] = _load_thresholds_table(
            thresh_dir,
            selected_markers,
            batch_values,
            marker_col_name or None,
            value_col_name or None,
            polygon_prefix or None,
            st.session_state["scimap_config"]["threshold_statistic"],
        )

    # --- Compute auto-thresholds ---
    if compute_auto and selected_markers and full_input_path and os.path.exists(full_input_path):
        with st.spinner(f"Computing {_active_mode} thresholds per batch..."):
            try:
                df_tmp = pd.read_csv(full_input_path, low_memory=False)
                # Apply data filter if set
                _filt = {
                    'enable_filter': st.session_state["scimap_config"].get('enable_filter', False),
                    'filter_col': st.session_state["scimap_config"].get('filter_col'),
                    'filter_val': st.session_state["scimap_config"].get('filter_val'),
                }
                df_tmp = apply_data_filter(df_tmp, _filt)
                auto_table = compute_auto_thresholds(
                    df=df_tmp,
                    markers=selected_markers,
                    method_name=_mode_key,
                    batch_col=batch_col if batch_col in df_tmp.columns else None,
                )
                if not auto_table.empty:
                    st.session_state["scimap_threshold_table_draft"] = auto_table
                    st.success(
                        f"Computed {_active_mode} thresholds for {len(auto_table)} batch(es) × "
                        f"{len(selected_markers)} markers. Review / edit below."
                    )
                else:
                    st.warning("Auto-threshold computation returned no results.")
            except Exception as e:
                st.error(f"Auto-threshold computation failed: {e}")
    elif compute_auto:
        st.warning("Please select markers and an input file first.")

    # --- Import Universal Gating Thresholds (from Threshold Gating tab) ---
    # This section is available in ALL threshold modes.
    if not _is_none:
        with st.expander("📥 Import Universal Gating Thresholds (from Threshold Tab)", expanded=False):
            st.caption(
                "Import universal (no-batch) thresholds exported by the **Threshold Gating** tool. "
                "The same threshold value will be applied to **every batch** in your data."
            )
            _default_gating_csv = str(project_root / "outputs" / "phenotyping" / "thresholds_gating.csv")
            gating_csv_path = st.text_input(
                "Gating Thresholds CSV Path",
                value=_default_gating_csv,
                key="scimap_gating_csv_path",
                help=(
                    "Path to a Threshold Gating export with `marker` and `threshold` columns. Imported thresholds are treated as universal values "
                    "and broadcast to each batch in the current dataset, or used once globally when no batch column is selected."
                ),
            )
            if st.button("Import Universal Thresholds", key="scimap_import_gating_btn", type="primary"):
                if not gating_csv_path or not os.path.exists(gating_csv_path):
                    st.error(f"File not found: {gating_csv_path}")
                elif not selected_markers:
                    st.error("Please select markers first.")
                else:
                    # Read batch values from data
                    _batch_vals = []
                    if full_input_path and os.path.exists(full_input_path) and batch_col:
                        try:
                            _batch_vals = pd.read_csv(
                                full_input_path, usecols=[batch_col]
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
                        st.session_state["scimap_threshold_table_draft"] = imported_df
                        n_markers = sum(
                            1 for m in selected_markers
                            if m in imported_df.columns and imported_df[m].notna().any()
                        )
                        st.success(
                            f"Imported universal thresholds for **{n_markers}** markers → "
                            f"**{len(imported_df)}** batch row(s). Review in the table below."
                        )

    # --- Initialise threshold table draft from cache if needed ---
    if "scimap_threshold_table_draft" not in st.session_state:
        if scimap_thresholds_cache.exists():
            try:
                st.session_state["scimap_threshold_table_draft"] = pd.read_csv(scimap_thresholds_cache)
            except Exception:
                st.session_state["scimap_threshold_table_draft"] = pd.DataFrame()
        else:
            st.session_state["scimap_threshold_table_draft"] = pd.DataFrame()

    if add_row:
        new_row = {"batch_id": "", "Threshold_File": ""}
        for m in selected_markers:
            new_row[m] = np.nan
        st.session_state["scimap_threshold_table_draft"] = pd.concat(
            [st.session_state["scimap_threshold_table_draft"], pd.DataFrame([new_row])],
            ignore_index=True,
        )

    # --- Editable threshold table (shown for Predefined & Auto modes, hidden for None) ---
    if _is_none:
        pass  # No threshold table needed
    elif selected_markers:
        thresh_table = st.session_state["scimap_threshold_table_draft"]
        if not thresh_table.empty:
            column_config = {
                "batch_id": st.column_config.TextColumn("batch_id"),
                "Threshold_File": st.column_config.TextColumn("Source", disabled=True),
            }
            for m in selected_markers:
                column_config[m] = st.column_config.NumberColumn(m, format="%.4f")

            edited_thresh = st.data_editor(
                thresh_table,
                column_config=column_config,
                hide_index=True,
                num_rows="dynamic",
                key="scimap_threshold_editor",
            )
            st.session_state["scimap_threshold_table_draft"] = edited_thresh
        else:
            if _is_predefined:
                st.info("No threshold table loaded yet. Click 'Load Thresholds' to populate.")
            else:
                st.info(f"Click **Compute {_active_mode} Thresholds** above to auto-detect thresholds.")
    else:
        st.info("Select markers first to enable threshold table.")

    if st.button("Save Threshold Table", type="primary", key="scimap_save_threshold_table"):
        st.session_state["scimap_threshold_table"] = st.session_state["scimap_threshold_table_draft"]
        try:
            st.session_state["scimap_threshold_table"].to_csv(scimap_thresholds_cache, index=False)
        except Exception:
            pass
        try:
            cached_params.update({"threshold_dir": thresh_dir})
            with open(scimap_cache, "w") as f:
                json.dump(cached_params, f, indent=2)
            from spacemap_gui.persistence import save_active_project_state
            save_active_project_state("tab7_phenotyping")
        except Exception:
            pass
        st.success("Threshold table saved.")

    st.divider()
    st.markdown("### 5. Phenotype Strategy Table")
    st.caption("Load and edit SCIMAP phenotype strategy (supports allpos/anypos/allneg/anyneg columns).")

    default_strategy_path = cached_params.get("strategy_path", str(project_root / "data" / "phenotype" / "Phenotype_Strategy_SCIMAP.csv"))
    strategy_path = st.text_input("Strategy CSV Path", value=default_strategy_path, key="scimap_strategy_path")
    if cached_params.get("strategy_path") != strategy_path:
        try:
            cached_params["strategy_path"] = strategy_path
            with open(scimap_cache, "w") as f:
                json.dump(cached_params, f, indent=2)
        except Exception:
            pass
    load_strategy = st.button("Load Strategy Table", key="scimap_load_strategy")
    add_strategy_row = st.button("Add Strategy Row", key="scimap_add_strategy_row")

    if load_strategy:
        if strategy_path and os.path.exists(strategy_path):
            try:
                st.session_state["scimap_strategy_table_draft"] = pd.read_csv(strategy_path)
                st.session_state["scimap_strategy_loaded_path"] = strategy_path
            except Exception:
                st.session_state["scimap_strategy_table_draft"] = pd.DataFrame()
        else:
            st.session_state["scimap_strategy_table_draft"] = pd.DataFrame()

    if "scimap_strategy_table_draft" not in st.session_state:
        if scimap_strategy_cache.exists():
            try:
                st.session_state["scimap_strategy_table_draft"] = pd.read_csv(scimap_strategy_cache)
            except Exception:
                st.session_state["scimap_strategy_table_draft"] = pd.DataFrame()
        else:
            st.session_state["scimap_strategy_table_draft"] = pd.DataFrame()

    if add_strategy_row:
        strategy_df = st.session_state["scimap_strategy_table_draft"]
        if strategy_df.empty:
            st.session_state["scimap_strategy_table_draft"] = pd.DataFrame([{"phenotype": ""}])
        else:
            empty_row = {c: "" for c in strategy_df.columns}
            st.session_state["scimap_strategy_table_draft"] = pd.concat(
                [strategy_df, pd.DataFrame([empty_row])],
                ignore_index=True,
            )

    strategy_df = st.session_state["scimap_strategy_table_draft"]
    if not strategy_df.empty:
        st.caption("Edits are buffered. Click Save to apply changes.")
        column_config = {}
        for col in strategy_df.columns:
            if pd.api.types.is_numeric_dtype(strategy_df[col]):
                column_config[col] = st.column_config.NumberColumn(col)
            else:
                column_config[col] = st.column_config.TextColumn(col)

        with st.form("scimap_strategy_form", clear_on_submit=False):
            edited_strategy = st.data_editor(
                strategy_df,
                column_config=column_config,
                hide_index=True,
                num_rows="dynamic",
                key="scimap_strategy_editor",
            )
            save_strategy = st.form_submit_button("Save Strategy Table", type="primary")

        if save_strategy:
            st.session_state["scimap_strategy_table_draft"] = edited_strategy
            st.session_state["scimap_strategy_table"] = edited_strategy
            try:
                edited_strategy.to_csv(scimap_strategy_cache, index=False)
            except Exception:
                pass
            try:
                cached_params.update({"strategy_path": strategy_path})
                with open(scimap_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
                from spacemap_gui.persistence import save_active_project_state
                save_active_project_state("tab7_phenotyping")
            except Exception:
                pass
            st.success("Strategy table saved.")
    else:
        st.info("No strategy table loaded yet. Click 'Load Strategy Table' to populate.")

    st.divider()
    st.markdown("### 6. Run SCIMAP Pipeline")

    if not selected_markers:
        st.warning("⚠️ Please select markers first")
        return

    if st.button("Run SCIMAP Pipeline", type="primary", width="stretch", key="scimap_run_btn"):
        if not full_input_path or not Path(full_input_path).exists():
            st.error("❌ Please select a valid input file")
            return

        if not _HAS_SCIMAP or manual_rescale_scimap is None:
            st.error("SCIMAP, AnnData, and Scanpy are required for this pipeline.")
            return

        strategy_table = st.session_state.get("scimap_strategy_table")
        if not _strategy_table_has_values(strategy_table):
            st.error("Please load and save a valid SCIMAP strategy table.")
            return

        cfg = st.session_state["scimap_config"]
        requested_batch_col = normalize_optional_column_name(cfg.get("batch_col", "batch_id"))
        output_col = cfg.get("output_col", "phenotype_scimap")
        gate_value = float(cfg.get("gate_value", 0.5))
        unclassified_label = cfg.get("unclassified_label", "unclassified")
        run_audit = begin_phenotyping_run_audit(
            Path(project_root),
            "scimap",
            input_csv=full_input_path,
            markers=selected_markers,
        )
        audit_input_rows = None

        # --- Determine threshold mode ---
        _run_mode_map = {**_THRESHOLD_MODE_TO_KEY, "None (No Thresholds)": "none"}
        run_mode = cfg.get("threshold_mode", "Predefined (CSV)")
        run_mode_key = _run_mode_map.get(run_mode, "predefined")
        apply_thresholds = run_mode_key != "none"

        thresholds_table = st.session_state.get("scimap_threshold_table")
        if thresholds_table is None or thresholds_table.empty:
            thresholds_table = st.session_state.get("scimap_threshold_table_draft")
        table_has_values = _threshold_table_has_values(thresholds_table, selected_markers)

        with st.status("Running SCIMAP Pipeline...", expanded=True) as status:
            try:
                status.write(f"[{_timestamp()}] Loading input data...")
                print(f"[{_timestamp()}] Loading input data...")
                df = pd.read_csv(full_input_path, low_memory=False)
                audit_input_rows = len(df)
                status.write(f"[{_timestamp()}] Loaded {len(df):,} cells")
                print(f"[{_timestamp()}] Loaded {len(df):,} cells")

                _scimap_filter = {
                    'enable_filter': cfg.get('enable_filter', False),
                    'filter_col': cfg.get('filter_col'),
                    'filter_val': cfg.get('filter_val'),
                }
                df = apply_data_filter(df, _scimap_filter, logger_fn=lambda msg: (
                    status.write(f"[{_timestamp()}] {msg}"),
                    print(f"[{_timestamp()}] SCIMAP: {msg}")
                ))

                # Drop rows containing NaN/Inf in selected markers
                if selected_markers:
                    original_len_nan = len(df)
                    valid_mask = ~df[selected_markers].apply(pd.to_numeric, errors='coerce').isna().any(axis=1)
                    df = df[valid_mask].copy()
                    status.write(f"[{_timestamp()}] Filtered out {original_len_nan - len(df):,} cells containing NaN/Inf in selected markers. Remaining: {len(df):,} cells.")
                    print(f"[{_timestamp()}] SCIMAP: Filtered out {original_len_nan - len(df):,} cells containing NaN/Inf in selected markers. Remaining: {len(df):,} cells.")
                    if len(df) == 0:
                        status.error("No cells remain after filtering out NaN/Inf values in selected markers.")
                        return

                batch_col = requested_batch_col if requested_batch_col in df.columns else None
                if requested_batch_col and batch_col is None:
                    status.write(
                        f"[{_timestamp()}] Batch column '{requested_batch_col}' not found. Processing the dataset globally."
                    )
                    print(
                        f"[{_timestamp()}] SCIMAP: Batch column '{requested_batch_col}' not found. Processing globally."
                    )

                df_results_list = []
                if batch_col:
                    batch_frames = [
                        (str(batch_id), df[df[batch_col].astype(str) == str(batch_id)].copy())
                        for batch_id in df[batch_col].dropna().astype(str).unique().tolist()
                    ]
                    status.write(f"[{_timestamp()}] Processing {len(batch_frames)} batches")
                    print(f"[{_timestamp()}] Processing {len(batch_frames)} batches")
                else:
                    batch_frames = [("global", df.copy())]
                    status.write(f"[{_timestamp()}] Processing the full dataset without batch grouping")
                    print(f"[{_timestamp()}] SCIMAP: Processing the full dataset without batch grouping")

                for batch_id, batch_df in batch_frames:
                    if batch_df.empty:
                        continue

                    print(f"[{_timestamp()}] Batch {batch_id}: {len(batch_df):,} cells")

                    markers_present = [m for m in selected_markers if m in batch_df.columns]
                    if not markers_present:
                        batch_df[output_col] = unclassified_label
                        df_results_list.append(batch_df)
                        continue

                    print(
                        f"[{_timestamp()}] Batch {batch_id}: {len(markers_present)} markers present (e.g., {', '.join(markers_present[:5])})"
                    )

                    thresholds = {}
                    if apply_thresholds:
                        print(f"[{_timestamp()}] Batch {batch_id}: Threshold mode = {run_mode}")

                        # 1) Try thresholds from the editable table (covers both
                        #    predefined CSV and pre-computed auto thresholds)
                        if table_has_values:
                            table_for_lookup = thresholds_table
                            if batch_col is None and thresholds_table is not None:
                                table_for_lookup = thresholds_table.drop(columns=["batch_id", "Threshold_File"], errors="ignore")
                            thresholds = _thresholds_from_table(table_for_lookup, batch_id, markers_present)
                            print(
                                f"[{_timestamp()}] Batch {batch_id}: Threshold table hits = {len(thresholds)}"
                            )
                            if not thresholds and thresholds_table is not None and not thresholds_table.empty:
                                try:
                                    sample_batch_ids = thresholds_table["batch_id"].astype(str).head(3).tolist()
                                    print(
                                        f"[{_timestamp()}] Batch {batch_id}: No table match. Sample table batch_id: {', '.join(sample_batch_ids)}"
                                    )
                                except Exception:
                                    pass

                        # 2) Fallback: for predefined CSV mode, try per-batch
                        #    threshold files on disk
                        if not thresholds and run_mode_key == "predefined":
                            threshold_file = _find_threshold_file(batch_id, thresh_dir)
                            if threshold_file is None and batch_col is None:
                                threshold_files = sorted(Path(thresh_dir).glob("*.csv")) if Path(thresh_dir).exists() else []
                                if len(threshold_files) == 1:
                                    threshold_file = threshold_files[0]
                            if threshold_file and threshold_file.exists():
                                try:
                                    print(
                                        f"[{_timestamp()}] Batch {batch_id}: Loading threshold file {threshold_file.name}"
                                    )
                                    tdf = pd.read_csv(threshold_file)
                                    thresholds = _extract_thresholds_from_csv(
                                        tdf,
                                        markers_present,
                                        marker_col_name or None,
                                        value_col_name or None,
                                        polygon_prefix or None,
                                        cfg.get("threshold_statistic", "median"),
                                    )
                                    print(
                                        f"[{_timestamp()}] Batch {batch_id}: Threshold file hits = {len(thresholds)}"
                                    )
                                except Exception:
                                    thresholds = {}

                        # 3) Fallback: for auto modes, compute on-the-fly for
                        #    this batch if no pre-computed values exist
                        if not thresholds and run_mode_key != "predefined":
                            auto_func = AUTO_THRESHOLD_METHODS.get(run_mode_key)
                            if auto_func is not None:
                                print(
                                    f"[{_timestamp()}] Batch {batch_id}: Computing {run_mode} on-the-fly"
                                )
                                for m in markers_present:
                                    if m in batch_df.columns:
                                        val = auto_func(batch_df[m].values)
                                        if val > 0:
                                            thresholds[m] = val
                                print(
                                    f"[{_timestamp()}] Batch {batch_id}: Auto-computed {len(thresholds)} thresholds"
                                )

                        print(
                            f"[{_timestamp()}] Batch {batch_id}: Threshold keys (sample) = {', '.join(list(thresholds.keys())[:5]) if thresholds else 'none'}"
                        )

                        if not thresholds:
                            batch_df[output_col] = unclassified_label
                            df_results_list.append(batch_df)
                            continue

                        thresholds_for_markers = {}
                        for m in markers_present:
                            base = _normalize_marker_name(m)
                            val = thresholds.get(m)
                            if pd.isna(val) or val is None:
                                val = thresholds.get(base)
                            if pd.isna(val) or val is None:
                                continue
                            try:
                                thresholds_for_markers[m] = float(val)
                            except Exception:
                                continue
                        print(
                            f"[{_timestamp()}] Batch {batch_id}: Thresholds mapped to markers = {len(thresholds_for_markers)}"
                        )
                        if not thresholds_for_markers:
                            batch_df[output_col] = unclassified_label
                            df_results_list.append(batch_df)
                            continue

                    adata = anndata.AnnData(batch_df[markers_present].astype(np.float32))
                    adata.obs_names = batch_df.index.astype(str)
                    if batch_col:
                        adata.obs[batch_col] = batch_df[batch_col].astype(str).values

                    if cfg.get("log1p", True):
                        sc.pp.log1p(adata)

                    adata_scaled = adata.copy()
                    if not apply_thresholds:
                        # None mode: delegate to manual_rescale_scimap(apply_thresholds=False)
                        # Uses 0 as the log-space gate anchor:
                        #   cells with log1p(expr) == 0  -> 0.0  (below gate)
                        #   cells with log1p(expr) >  0  -> (0.5, 1.0]  (above gate)
                        status.write(
                            f"[{_timestamp()}] Batch {batch_id}: No-threshold mode — values anchored at 0."
                            " Ensure data is background-subtracted for this mode to be meaningful."
                        )
                        print(
                            f"[{_timestamp()}] Batch {batch_id}: None mode — using 0 as gate anchor"
                        )
                        adata_scaled.X = manual_rescale_scimap(adata, apply_thresholds=False)
                    elif apply_thresholds and thresholds_for_markers:
                        log_thresholds = {m: np.log1p(v) for m, v in thresholds_for_markers.items()}
                        gates = pd.Series(log_thresholds)
                        if gates.empty:
                            batch_df[output_col] = unclassified_label
                            df_results_list.append(batch_df)
                            continue
                        try:
                            gate_cover = {}
                            for m in markers_present:
                                if m in gates.index:
                                    col = adata[:, m].X
                                    if hasattr(col, "toarray"):
                                        col = col.toarray().ravel()
                                    gate_cover[m] = float(np.mean(col >= gates.loc[m]))
                            if gate_cover and all(v < 0.01 for v in gate_cover.values()):
                                print(
                                    f"[{_timestamp()}] Warning: <1% of values exceed gates for batch {batch_id}. "
                                    "Check threshold scale or log1p setting."
                                )
                            elif gate_cover:
                                top_low = sorted(gate_cover.items(), key=lambda kv: kv[1])[:5]
                                print(
                                    f"[{_timestamp()}] Batch {batch_id}: Gate coverage (lowest 5) = "
                                    + ", ".join([f"{m}:{v:.2%}" for m, v in top_low])
                                )
                        except Exception:
                            pass
                        adata_scaled.X = manual_rescale_scimap(adata, gates, apply_thresholds=True)
                    else:
                        adata_scaled.X = manual_rescale_scimap(adata, apply_thresholds=False)

                    # Rename strategy_table columns to exactly match markers in adata (e.g., handles E-cadherin_corr vs E-cadherin)
                    batch_strategy = strategy_table.copy()
                    rename_map = {}
                    str_cols = set(batch_strategy.columns)
                    for m in markers_present:
                        base = _normalize_marker_name(m)
                        if m in str_cols:
                            continue
                        if base in str_cols:
                            rename_map[base] = m
                    if rename_map:
                        batch_strategy = batch_strategy.rename(columns=rename_map)

                    # Drop strategy columns that still don't resolve to any
                    # variable in adata (e.g. strategy has 'CD3' but data only
                    # has 'CD31_corr').  Only drop columns whose values are
                    # exclusively SCIMAP gate expressions — this preserves
                    # required SCIMAP metadata columns (Phenotype, group, etc.)
                    # regardless of what the user named them.
                    _var_set = set(adata_scaled.var_names)
                    _GATE_VALS = frozenset(
                        {"pos", "neg", "allpos", "anypos", "allneg", "anyneg"}
                    )
                    _drop_cols = []
                    for _sc in batch_strategy.columns:
                        if _sc in _var_set:
                            continue
                        _unique = {
                            str(v).strip().lower()
                            for v in batch_strategy[_sc].dropna()
                        }
                        # Column is a stale marker column only if ALL its
                        # non-null values are gate keywords.  Metadata columns
                        # (Phenotype names, group labels, etc.) have other values.
                        if _unique and _unique.issubset(_GATE_VALS):
                            _drop_cols.append(_sc)
                    if _drop_cols:
                        print(
                            f"[{_timestamp()}] Batch {batch_id}: Dropping unresolvable "
                            f"marker strategy columns: {_drop_cols}"
                        )
                        batch_strategy = batch_strategy.drop(columns=_drop_cols)

                    sm.tl.phenotype_cells(
                        adata_scaled,
                        phenotype=batch_strategy,
                        gate=gate_value,
                        label=output_col,
                    )

                    phenotype_results = adata_scaled.obs[output_col].astype(str)
                    phenotype_results = phenotype_results.replace({"Unknown": unclassified_label, "unclassified": unclassified_label})
                    batch_df[output_col] = phenotype_results.values
                    df_results_list.append(batch_df)

                    try:
                        ph_counts = batch_df[output_col].value_counts(dropna=False).head(5)
                        print(
                            f"[{_timestamp()}] Batch {batch_id}: Top phenotypes = "
                            + ", ".join([f"{k} ({v:,})" for k, v in ph_counts.items()])
                        )
                    except Exception:
                        pass

                if not df_results_list:
                    status.error("No batches were processed successfully")
                    return

                df_results = pd.concat(df_results_list, ignore_index=True)
                status.write(f"[{_timestamp()}] SCIMAP phenotyping complete")
                print(f"[{_timestamp()}] SCIMAP phenotyping complete")

                unique_phenotypes = sorted([p for p in df_results[output_col].unique() if p != "Unassigned"])
                pheno_to_cluster = {p: i for i, p in enumerate(unique_phenotypes)}
                df_results["scimap_cluster"] = df_results[output_col].map(pheno_to_cluster).fillna(-1)

                temp_file = project_root / "outputs/phenotyping/cache" / f"scimap_results_{int(time.time())}.parquet"
                df_results.to_parquet(temp_file)

                st.session_state["scimap_temp_file"] = str(temp_file)
                st.session_state["scimap_phenotype_map"] = {int(c): p for p, c in pheno_to_cluster.items()}
                st.session_state["scimap_available_phenotypes"] = [unclassified_label] + unique_phenotypes
                st.session_state["scimap_marker_cols"] = [m for m in selected_markers if m in df_results.columns]
                st.session_state["scimap_outcol"] = output_col
                st.session_state["scimap_cluster_col_name"] = "scimap_cluster"

                status.update(label="SCIMAP Complete!", state="complete")
                st.success("SCIMAP phenotyping complete")

                finish_phenotyping_run_audit(
                    run_audit,
                    status="success",
                    output_path=temp_file,
                    input_rows=audit_input_rows,
                    output_rows=len(df_results),
                    output_column=output_col,
                    cluster_column="scimap_cluster",
                    phenotype_map=st.session_state["scimap_phenotype_map"],
                    extra={
                        "gate_value": gate_value,
                        "batch_column": batch_col,
                        "phenotype_cell_counts": df_results[output_col].value_counts(dropna=False).to_dict(),
                    },
                )
                run_audit = None

            except Exception as e:
                finish_phenotyping_run_audit(
                    run_audit,
                    status="failed",
                    input_rows=audit_input_rows,
                    output_column=output_col,
                    cluster_column="scimap_cluster",
                    error=str(e),
                )
                run_audit = None
                status.error(f"[{_timestamp()}] Pipeline error: {e}")
                logger.error(f"SCIMAP pipeline error: {e}")
            finally:
                if run_audit is not None:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_col,
                        cluster_column="scimap_cluster",
                        error="Pipeline stopped before completion; see the SCIMAP status messages.",
                    )

    if "scimap_temp_file" in st.session_state and Path(st.session_state["scimap_temp_file"]).exists():
        st.divider()
        st.subheader("Results & Phenotype Refinement")
        st.caption("Interactive visualization and export of SCIMAP phenotyping results")

        temp_file = st.session_state["scimap_temp_file"]
        df_results = pd.read_parquet(temp_file)
        marker_cols = st.session_state.get("scimap_marker_cols", [])
        output_col = st.session_state.get("scimap_outcol", "phenotype_scimap")
        cluster_col = st.session_state.get("scimap_cluster_col_name", "scimap_cluster")

        tab_heatmap, tab_export = st.tabs(["Heatmap & Statistics", "Export Results"])

        with tab_heatmap:
            st.markdown("### Expression Heatmap & Phenotype Statistics")

            rename_map = _render_scimap_phenotype_rename_table(df_results, output_col)
            df_results, phenotype_map, available_phenotypes, heatmap_cluster_col, heatmap_phenotype_map = _apply_scimap_phenotype_renames(
                df_results,
                output_col,
                cluster_col,
                rename_map,
            )
            st.session_state["scimap_phenotype_map"] = phenotype_map
            st.session_state["scimap_available_phenotypes"] = available_phenotypes

            _batch_col_sci = st.session_state.get("scimap_config", {}).get("batch_col", "batch_id")
            _confirmed_markers_sci = st.session_state.get("scimap_confirmed_markers", [])
            raw_marker_cols_sci = [
                marker_name
                for marker_name in _confirmed_markers_sci
                if marker_name in df_results.columns and pd.api.types.is_numeric_dtype(df_results[marker_name])
            ]
            if not raw_marker_cols_sci:
                raw_marker_cols_sci = [
                    marker_name
                    for marker_name in marker_cols
                    if marker_name in df_results.columns and pd.api.types.is_numeric_dtype(df_results[marker_name])
                ]

            batch_norm_heatmap_data = _build_scimap_batch_norm_heatmap_data(
                df_results=df_results,
                cluster_col=heatmap_cluster_col,
                raw_marker_cols=raw_marker_cols_sci,
                batch_col=_batch_col_sci,
            )
            if not batch_norm_heatmap_data.empty:
                st.caption(
                    "Top marker levels below use raw-marker Z-scores computed within each batch, "
                    "then averaged within each phenotype and ranked by the top 5 averages."
                )
                render_phenotype_statistics(
                    df_results=df_results,
                    cluster_col=heatmap_cluster_col,
                    phenotype_map=heatmap_phenotype_map,
                    heatmap_z_ordered=batch_norm_heatmap_data,
                    cluster_label="Phenotype",
                    score_label="Batch-Norm Expression",
                )

            if "scimap_heatmap_config" not in st.session_state:
                st.session_state["scimap_heatmap_config"] = {
                    "z_min": -2.0,
                    "z_max": 2.0,
                }

            with st.expander("Heatmap Configuration", expanded=False):
                z_col1, z_col2 = st.columns(2)
                with z_col1:
                    if "scimap_z_min" in st.session_state:
                        try:
                            st.session_state["scimap_z_min"] = float(st.session_state["scimap_z_min"])
                        except Exception:
                            st.session_state["scimap_z_min"] = -2.0
                    zmin_val_sci = float(st.session_state["scimap_heatmap_config"].get("z_min", -2.0))
                    st.session_state["scimap_heatmap_config"]["z_min"] = st.slider(
                        "Z-Score Min",
                        -5.0, 0.0,
                        zmin_val_sci,
                        step=0.1,
                        key="scimap_z_min",
                    )
                with z_col2:
                    if "scimap_z_max" in st.session_state:
                        try:
                            st.session_state["scimap_z_max"] = float(st.session_state["scimap_z_max"])
                        except Exception:
                            st.session_state["scimap_z_max"] = 2.0
                    zmax_val_sci = float(st.session_state["scimap_heatmap_config"].get("z_max", 2.0))
                    st.session_state["scimap_heatmap_config"]["z_max"] = st.slider(
                        "Z-Score Max",
                        0.0, 5.0,
                        zmax_val_sci,
                        step=0.1,
                        key="scimap_z_max",
                    )

            numeric_cols = [c for c in marker_cols if c in df_results.columns and pd.api.types.is_numeric_dtype(df_results[c])]
            if numeric_cols:
                # --- Score mode selector ---
                _scimap_score_mode = st.radio(
                    "Heatmap Score Mode",
                    options=[
                        "Batch-Norm Expression (Raw Markers)",
                        "Pipeline Norm Score (Cross-Phenotype Z)",
                    ],
                    index=0,
                    horizontal=True,
                    key="scimap_heatmap_score_mode",
                    help=(
                        "The phenotype statistics table above always uses batch-grouped raw-marker Z-scores. "
                        "This selector controls only the publication heatmap below.\n\n"
                        "**Batch-Norm Expression** \u2013 Z-scores each raw marker within each batch "
                        "(grouped by the Batch Column set in configuration), then averages per phenotype. "
                        "Directly reflects how each marker distributes within a phenotype relative to the "
                        "per-batch baseline. Removes batch intensity variation and avoids cross-phenotype "
                        "Z-score overinflation of rare phenotypes (e.g., Treg). Recommended.\n\n"
                        "**Pipeline Norm Score** \u2013 Averages raw marker values per phenotype then "
                        "re-Z-scores across all phenotypes. Susceptible to rarity-driven inflation: "
                        "small phenotype groups show artificially broad elevation because any deviation "
                        "from the cross-phenotype mean is amplified."
                    ),
                )

                df_valid = df_results[df_results[heatmap_cluster_col].notna()].copy()
                if not df_valid.empty:
                    if _scimap_score_mode == "Batch-Norm Expression (Raw Markers)":
                        heatmap_data = batch_norm_heatmap_data
                        score_label = "Batch-Norm Expression"
                    else:
                        # Pipeline Norm Score: existing behaviour
                        mean_expression = df_valid.groupby(heatmap_cluster_col)[numeric_cols].mean()
                        if isinstance(mean_expression, pd.Series):
                            mean_expression = mean_expression.to_frame()
                        if mean_expression.shape[0] > 1:
                            heatmap_z = mean_expression.apply(
                                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0,
                                axis=0,
                            )
                        else:
                            heatmap_z = mean_expression.copy()
                            heatmap_z.loc[:, :] = 0
                        heatmap_data = heatmap_z.T
                        heatmap_data.index = [get_analysis_marker_label(marker) for marker in heatmap_data.index]
                        score_label = "Pipeline Norm Score"

                    if not heatmap_data.empty:
                        render_publication_heatmap(
                            heatmap_z_ordered=heatmap_data,
                            phenotype_map=heatmap_phenotype_map,
                            key_prefix="scimap",
                            vmin=float(st.session_state["scimap_heatmap_config"].get("z_min", -2.0)),
                            vmax=float(st.session_state["scimap_heatmap_config"].get("z_max", 2.0)),
                            df_results=df_results,
                            group_col=heatmap_cluster_col,
                            marker_columns=(raw_marker_cols_sci if _scimap_score_mode == "Batch-Norm Expression (Raw Markers)" else numeric_cols),
                            marker_source_note=build_marker_source_note(
                                raw_marker_cols_sci if _scimap_score_mode == "Batch-Norm Expression (Raw Markers)" else numeric_cols,
                                method_key="scimap",
                                normalization_label=score_label,
                            ),
                        )
                    else:
                        st.warning("No marker columns available for the selected score mode.")
            else:
                st.warning("No numeric columns for heatmap")

        with tab_export:
            st.markdown("### Export Results")
            render_save_assignments(
                df_results=df_results,
                phenotype_map=phenotype_map,
                output_col=output_col,
                project_root=project_root,
                key_prefix="scimap",
                cluster_col=cluster_col,
                update_cluster_cols=[cluster_col, output_col],
                cell_id_col=st.session_state.get('scimap_config', {}).get('cell_id_col', 'cell_id'),
            )

            render_export_split_data(
                df_results=df_results,
                output_col=output_col,
                phenotype_map=phenotype_map,
                project_root=project_root,
                key_prefix="scimap",
                cluster_col=cluster_col,
                cell_id_col=st.session_state.get('scimap_config', {}).get('cell_id_col', 'cell_id'),
            )

    # --- VISUAL VALIDATION COMPONENT (Ported from Leiden) ---
    if st.session_state.get('viz_enabled', True):
        st.divider()
        with st.expander("Visual Validation of Phenotypes", expanded=True):
            st.markdown("#### Visualize Representative Cells")
            st.caption("Inspect random cells from specific phenotypes to validate assignments.")
            
            # Ensure we have data
            if 'scimap_temp_file' in st.session_state and Path(st.session_state['scimap_temp_file']).exists():
                # Load data if not already
                if 'df_results' not in locals():
                    df_results = pd.read_parquet(st.session_state['scimap_temp_file'])
                
                # Check Global Viz Config
                viz_roi = st.session_state.get('viz_roi_dir')
                viz_seg = st.session_state.get('viz_seg_dir')
                
                if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
                    st.warning("Please configure valid ROI and Segmentation directories in Global Input Configuration (Top of Tab 7).")
                else:
                    viz = CellVisualizer(viz_roi, viz_seg)
                    
                    # File matching (cache in session state)
                    if not st.session_state.get('scimap_viz_file_matches'):
                         # Get unique batches
                         # SCIMAP uses 'batch_id' default or configured
                         batch_col = st.session_state.get('scimap_batch_col', 'batch_id')
                         if batch_col not in df_results.columns:
                             batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                         
                         if batch_col in df_results.columns:
                             unique_batches = df_results[batch_col].unique().astype(str).tolist()
                             matches = viz.match_file_triads(unique_batches)
                             st.session_state.scimap_viz_file_matches = matches
                             if not matches:
                                 st.error("No matching image files found for these batches.")
                         else:
                             st.error(f"Batch column '{batch_col}' not found in results.")
                    else:
                        viz.file_map = st.session_state.scimap_viz_file_matches

                    # UI Controls
                    # SCIMAP also results in phenotypes directly (via manual gate or auto).
                    output_col = st.session_state.get("scimap_outcol", "phenotype_scimap")
                    
                    available_phenos = []
                    if output_col in df_results.columns:
                        available_phenos = sorted([p for p in df_results[output_col].unique() if pd.notna(p) and str(p).strip()])
                    
                    with st.form("scimap_viz_form"):
                         vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])
                         with vv_col1:
                             selected_pheno_vv = st.selectbox(
                                 "Select Phenotype", 
                                 available_phenos,
                                 key="scimap_viz_pheno_sel"
                             )
                         with vv_col2:
                             n_viz_cells = st.number_input("Cells to View", 1, 50, 5, key="scimap_viz_n_cells")
                         with vv_col3:
                             show_neighbors = st.checkbox("Show Neighbors", value=True, key="scimap_viz_neighbors")
                         with vv_col4:
                             scimap_batch_scope = st.radio(
                                 "Batch Scope",
                                 ["All matched batches", "Specific batch"],
                                 horizontal=True,
                                 key="scimap_viz_batch_scope",
                             )
                             scimap_batch_options = sorted(str(batch) for batch in st.session_state.scimap_viz_file_matches.keys())
                             selected_batch_vv = None
                             if scimap_batch_scope == "Specific batch":
                                 selected_batch_vv = st.selectbox(
                                     "Batch",
                                     scimap_batch_options,
                                     key="scimap_viz_batch_sel",
                                 )
                             st.write("") # Spacer
                             run_viz = st.form_submit_button("Visualize Random Cells")
                    
                    # Logic
                    if run_viz and selected_pheno_vv:
                        subset = df_results[df_results[output_col] == selected_pheno_vv]
                        
                        if len(subset) == 0:
                            st.warning("No cells in this phenotype group.")
                        else:
                            # 1. Determine Top 5 Markers
                            # Use configured marker cols
                            marker_cols = st.session_state.get("scimap_marker_cols", [])
                            # Filter for numeric
                            numeric_cols = [c for c in marker_cols if c in df_results.columns and pd.api.types.is_numeric_dtype(df_results[c])]
                            
                            if numeric_cols:
                                cluster_means = subset[numeric_cols].mean()
                                top_5_markers = cluster_means.nlargest(5).index.tolist()
                                top_5_names = [c.replace('_norm', '').replace('_corr', '').replace('_unmixed', '') for c in top_5_markers]
                                st.info(f"Top expressed markers for {selected_pheno_vv}: {', '.join(top_5_names)}")
                                
                                # 2. Sample Cells
                                batch_col = st.session_state.get('scimap_batch_col', 'batch_id')
                                if not batch_col or batch_col not in df_results.columns:
                                     batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                                
                                if batch_col not in df_results.columns:
                                    st.error(f"Batch column '{batch_col}' not found in results.")
                                    subset_valid = pd.DataFrame()
                                elif not st.session_state.get('scimap_viz_file_matches'):
                                    st.error("No matching image files found for visualization batches.")
                                    subset_valid = pd.DataFrame()
                                else:
                                    valid_batches = list(st.session_state.scimap_viz_file_matches.keys())
                                    subset_valid = subset[subset[batch_col].astype(str).isin(valid_batches)]
                                    if scimap_batch_scope == "Specific batch" and selected_batch_vv:
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
                                    
                                    x_col_name = next((c for c in ['x_centroid', 'AreaShape_Center_X', 'X_centroid', 'X'] if c in df_results.columns), 'x_centroid')
                                    y_col_name = next((c for c in ['y_centroid', 'AreaShape_Center_Y', 'Y_centroid', 'Y'] if c in df_results.columns), 'y_centroid')
                                    tile_sz = 40
                                    
                                    cid_col = st.session_state.get('scimap_cell_id_col', 'cell_id')

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
                                            
                                            # Robust Matching
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
                                                    'phenotype': selected_pheno_vv, 
                                                    'method': 'scimap',
                                                    'x': x, 'y': y,
                                                    'top_markers': ", ".join(names_for_plot)
                                                }
                                                pd.DataFrame([meta_rec]).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                                            except Exception as e:
                                                print(f"Auto-save error: {e}")
                                            
                                            plt.close(fig)
                                        except Exception as e:
                                            st.error(f"Error viewing cell: {e}")
                                        
                                        prog_bar.progress((idx + 1) / n_actual)
                                    st.success(f"Visualized {n_actual} cells. Output saved to `outputs/phenotyping/visual_validation/`")
                            else:
                                st.warning("Could not identify marker columns.")
