"""Tab 6: Batch Correction - Harmonize data across multiple slides."""

import streamlit as st

from spacemap_gui.ui_utils import clamp_numeric_widget_state
import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import io
import time
from difflib import SequenceMatcher
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Optional: AG Grid spreadsheet editor (supports Excel-like paste)
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    from st_aggrid.shared import JsCode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False
    AgGrid = None  # type: ignore[assignment]
    GridOptionsBuilder = None  # type: ignore[assignment]
    GridUpdateMode = None  # type: ignore[assignment]
    JsCode = None  # type: ignore[assignment]


checkbox_cell_renderer = None
if AGGRID_AVAILABLE and JsCode is not None:
    checkbox_cell_renderer = JsCode(
        """
        class CheckboxRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('input');
                this.eGui.type = 'checkbox';
                this.eGui.checked = !!params.value;
                this.eGui.style.cursor = 'pointer';
                this.eGui.addEventListener('change', (e) => {
                    params.setValue(e.target.checked);
                });
            }
            getGui() {
                return this.eGui;
            }
            refresh(params) {
                this.params = params;
                this.eGui.checked = !!params.value;
                return true;
            }
        }
        """
    )

# batch_correction_utils is loaded LAZILY inside render() to avoid importing
# sklearn, minisom, pycombat at app startup.
BATCH_UTILS_AVAILABLE = None  # Will be set on first render()
BatchCorrector = None  # type: ignore[assignment]

from spacemap_gui.file_browser import render_file_browser
from spacemap_gui.ui_utils import (
    directory_input_with_browse,
    show_preflight_errors,
    validate_run_directories,
    validate_selected_files,
)
from spacemap_gui.process_log import record_process_event
from spacemap_gui.persistence import (
    load_directory_settings,
    save_directory_settings,
    get_default_bc_config,
)

# Logging helper
def log_step(step_name: str):
    """Print timestamped log to stdout for debugging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] TAB6: {step_name}", flush=True)
    record_process_event("tab6", "Batch correction step", message=step_name)


def _transform_vector(x: pd.Series, transform: str, arcsinh_cofactor: float, log_pseudocount: float) -> pd.Series:
    x_num = pd.to_numeric(x, errors='coerce').astype(float)
    if transform == 'Linear':
        return x_num
    if transform == 'Log10':
        arr = np.log10(np.clip(x_num.to_numpy(), a_min=0, a_max=None) + float(log_pseudocount))
        return pd.Series(arr, index=x_num.index)
    if transform == 'Arcsinh':
        cof = float(arcsinh_cofactor) if float(arcsinh_cofactor) != 0 else 1.0
        arr = np.arcsinh(x_num.to_numpy() / cof)
        return pd.Series(arr, index=x_num.index)
    if transform == 'Z-score':
        mu = float(np.nanmean(x_num.to_numpy()))
        sigma = float(np.nanstd(x_num.to_numpy()))
        if sigma == 0 or np.isnan(sigma):
            return x_num * 0
        return (x_num - mu) / sigma
    return x_num


def _load_phenotyping_marker_defaults(config_path: Path | None = None) -> list[str]:
    """Load the marker panel used by the default Tab 7 phenotyping methods."""
    marker_config_path = config_path or Path(__file__).resolve().parents[3] / "marker_configs.json"
    try:
        with marker_config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return []

    profiles = config.get("profiles", {}) if isinstance(config, dict) else {}
    profile = profiles.get("default_phenotyping", {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile, dict):
        return []

    markers: list[str] = []
    for value in [*profile.get("corr_markers", []), *profile.get("non_corr_markers", [])]:
        marker = str(value).strip()
        if marker and marker not in markers:
            markers.append(marker)
    return markers


def _pick_batch_marker_defaults(
    common_columns: list[str],
    config: dict,
    metadata_columns: list[str] | None = None,
    phenotyping_markers: list[str] | None = None,
) -> list[str]:
    """Prefer Tab 7's marker panel, falling back to the existing heuristic."""
    shared_markers = phenotyping_markers
    if shared_markers is None:
        shared_markers = _load_phenotyping_marker_defaults()
    shared_defaults = [marker for marker in shared_markers if marker in common_columns]
    if shared_defaults:
        return shared_defaults

    ignore_exact = {
        "x", "y", "X", "Y", "id", "label", "area", "quality",
        "original_filename", "file_prefix", "batch",
        config.get("cell_id_col", "cell_id"),
        config.get("batch_id_col", "batch_id"),
        config.get("tissue_col", "tissue"),
        config.get("area_col", "area"),
    }
    ignore_exact.update(metadata_columns or [])
    excluded = [str(value).lower() for value in config.get("exclude_markers", [])]

    picked: list[str] = []
    for column in common_columns:
        if column in ignore_exact:
            continue
        lower_column = column.lower()
        if any(excluded_name in lower_column for excluded_name in excluded):
            continue
        if any(token in lower_column for token in ("umap", "pca", "cluster", "leiden", "phenotype")):
            continue
        picked.append(column)
    return picked


def _read_unified_batch_values(csv_path: str | Path, batch_column: str) -> list[str]:
    """Read unique non-null batch values from one unified CSV."""
    batch_values = pd.read_csv(csv_path, usecols=[batch_column])[batch_column]
    return sorted(batch_values.dropna().astype(str).unique().tolist())


def _assign_batch_correction_cell_ids(
    df: pd.DataFrame,
    config: dict,
    batch_id_col: str,
) -> pd.DataFrame:
    """Apply Tab 6's configured cell-ID strategy to a complete input table."""
    result = df.copy()
    cell_id_col = str(config.get("cell_id_col", "cell_id"))
    local_cell_id_col = str(config.get("local_cell_id_col", "label"))
    strategy = config.get("cell_id_strategy", "Auto-generate global (cell_0, cell_1, ...)")

    if strategy == "Combine Batch ID + Local Column" and local_cell_id_col in result.columns:
        result[cell_id_col] = (
            result[batch_id_col].fillna("Unknown").astype(str)
            + "_"
            + result[local_cell_id_col].fillna("").astype(str)
        )
    elif strategy == "Use original column directly (must be unique)" and local_cell_id_col in result.columns:
        result[cell_id_col] = result[local_cell_id_col].astype(str)
    else:
        result[cell_id_col] = [f"cell_{row_index}" for row_index in range(len(result))]

    if result[cell_id_col].isna().any() or result[cell_id_col].duplicated().any():
        raise ValueError(
            f"Cell ID strategy produced missing or duplicate values in '{cell_id_col}'. "
            "Choose a unique local ID column or use global auto-generation."
        )
    return result


def _replace_corrected_columns(
    base_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    *,
    cell_id_col: str,
    corrected_columns: list[str],
    active_cell_ids: pd.Series | list | set,
    indicator_col: str | None = None,
) -> pd.DataFrame:
    """Replace correction results for active cells while preserving other rows."""
    if cell_id_col not in base_df.columns or cell_id_col not in corrected_df.columns:
        raise ValueError(f"Missing cell ID column '{cell_id_col}' while updating corrected values.")
    if corrected_df[cell_id_col].duplicated().any():
        raise ValueError(f"Corrected results contain duplicate values in '{cell_id_col}'.")

    result = base_df.copy()
    active_mask = result[cell_id_col].isin(set(active_cell_ids))
    corrected_index = corrected_df.set_index(cell_id_col)
    corrected_mask = result[cell_id_col].isin(corrected_index.index)

    for column in corrected_columns:
        if column not in corrected_df.columns:
            raise ValueError(f"Corrected results are missing expected column '{column}'.")
        if column not in result.columns:
            result[column] = np.nan
        result.loc[active_mask, column] = np.nan
        result.loc[corrected_mask, column] = result.loc[corrected_mask, cell_id_col].map(
            corrected_index[column]
        )

    if indicator_col:
        if indicator_col not in result.columns:
            result[indicator_col] = 0
        result.loc[active_mask, indicator_col] = 0
        result.loc[corrected_mask, indicator_col] = 1
    return result


def _write_csv_atomic(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a CSV beside its destination and atomically replace the target."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(
        f".{target_path.name}.batch_correction_{os.getpid()}_{time.time_ns()}.tmp"
    )
    try:
        df.to_csv(temp_path, index=False)
        try:
            os.chmod(temp_path, target_path.stat().st_mode)
        except OSError:
            pass
        os.replace(temp_path, target_path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
    return target_path

def render():
    """Main render function for Tab 6."""
    import streamlit as st
    from pathlib import Path
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    # ---- Lazy-load heavy batch correction libraries on first render ----
    global BATCH_UTILS_AVAILABLE, BatchCorrector
    global discover_data_and_markers, load_and_combine_data, generate_diagnostic_visualizations
    global detect_overcorrected_cells, apply_post_correction_qc_filters
    if BATCH_UTILS_AVAILABLE is None:
        try:
            from spacemap.batch_correction_utils import (
                BatchCorrector as _BC,
                discover_data_and_markers as _DDM,
                load_and_combine_data as _LACD,
                generate_diagnostic_visualizations as _GDV,
                detect_overcorrected_cells as _DOC,
                apply_post_correction_qc_filters as _APQ,
            )
            BatchCorrector = _BC
            discover_data_and_markers = _DDM
            load_and_combine_data = _LACD
            generate_diagnostic_visualizations = _GDV
            detect_overcorrected_cells = _DOC
            apply_post_correction_qc_filters = _APQ
            BATCH_UTILS_AVAILABLE = True
        except ImportError as e:
            print(f"Error importing batch_correction_utils: {e}")
            BATCH_UTILS_AVAILABLE = False
            BatchCorrector = None  # type: ignore[assignment]

    # Mark Tab 6 as the currently rendering tab (for any cross-tab state logic).
    st.session_state["_currently_rendering_tab"] = "tab6"

    st.markdown("## Batch Correction Pipeline")
    
    if not BATCH_UTILS_AVAILABLE:
        st.error("Batch correction utilities could not be imported. Please check your installation.")
        return

    t1, t2, t3 = st.tabs(["1. Setup & Filter", "2. QC & Correction", "3. Visualization"])
    
    with t1:
        render_setup(project_root)
    
    with t2:
        render_qc_correction(project_root)
        
    with t3:
        render_visualization(project_root)

def render_setup(project_root: Path):
    """Subtab 1: Setup & Filter."""
    st.header("1. Setup & Filter")
    st.markdown("Configure input files, metadata, and apply quality filters.")
    
    # Initialize State
    updated = False
    if not st.session_state.get('bc_input_dir'):
        st.session_state.bc_input_dir = str(project_root / 'outputs/annotated_csv')
        updated = True
    if not st.session_state.get('bc_output_dir'):
        st.session_state.bc_output_dir = str(project_root / 'outputs/batch_corrected')
        updated = True
    if 'bc_input_mode' not in st.session_state:
        st.session_state.bc_input_mode = "Directory of CSV files"
        updated = True
    if 'bc_unified_csv_path' not in st.session_state:
        st.session_state.bc_unified_csv_path = str(
            project_root / 'outputs/batch_corrected/batch_corrected_data.csv'
        )
        updated = True
    if 'bc_unified_batch_col' not in st.session_state:
        st.session_state.bc_unified_batch_col = 'batch_id'
        updated = True
    if 'bc_config' not in st.session_state:
        st.session_state.bc_config = get_default_bc_config()
        updated = True
    if 'bc_files_df' not in st.session_state:
        st.session_state.bc_files_df = pd.DataFrame(columns=["Process", "Filename"])
        updated = True
    if 'bc_metadata_cols' not in st.session_state:
        st.session_state.bc_metadata_cols = []
        updated = True
    if 'bc_cleaned_data' not in st.session_state:
        st.session_state.bc_cleaned_data = None
        updated = True
    if 'bc_first_file_columns' not in st.session_state:
        st.session_state.bc_first_file_columns = []
        updated = True
    if 'bc_common_columns' not in st.session_state:
        st.session_state.bc_common_columns = []
        updated = True
    if 'bc_union_columns' not in st.session_state:
        st.session_state.bc_union_columns = []
        updated = True
    if 'bc_common_markers' not in st.session_state:
        st.session_state.bc_common_markers = []
        updated = True
    if 'bc_selected_markers' not in st.session_state:
        st.session_state.bc_selected_markers = []
        updated = True
    if 'bc_aggrid_key_nonce' not in st.session_state:
        st.session_state.bc_aggrid_key_nonce = 0
        updated = True
        
    if updated:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TAB6: Initialized default configuration values", flush=True)
        st.rerun()
    
    # 1. Directories
    st.subheader("1. Directories")
    
    def _persist_tab6_settings():
        """Save all Tab 6 settings for the current directory."""
        settings_root = (
            str(Path(st.session_state.bc_unified_csv_path).expanduser().parent)
            if st.session_state.bc_input_mode == "Unified CSV file"
            else st.session_state.bc_input_dir
        )
        if not settings_root:
            return
            
        settings = {
            'bc_config': st.session_state.bc_config,
            'bc_input_mode': st.session_state.bc_input_mode,
            'bc_unified_csv_path': st.session_state.bc_unified_csv_path,
            'bc_unified_batch_col': st.session_state.bc_unified_batch_col,
            'bc_metadata_cols': st.session_state.bc_metadata_cols,
            'bc_selected_markers': st.session_state.bc_selected_markers,
            # Save file metadata as records
            'files_metadata': st.session_state.bc_files_df.to_dict('records') if not st.session_state.bc_files_df.empty else []
        }
        save_directory_settings(settings_root, 'tab6', settings)

    def _save_tab6_project_settings():
        """Save Tab 6's current controls to the active project JSON.

        The marker picker used to live inside a form while the Save Project
        buttons lived outside it.  A user could therefore change the picker
        and click Save Project without Streamlit submitting the form value.
        The picker is now a normal widget, but this defensive copy also keeps
        the project save correct if an older session still has the former
        widget key.
        """
        marker_value = st.session_state.get("bc_marker_multiselect")
        if marker_value is None:
            marker_value = st.session_state.get("bc_marker_multiselect_form")
        if isinstance(marker_value, (list, tuple)):
            common = set(st.session_state.get("bc_common_columns", []))
            st.session_state.bc_selected_markers = [
                str(marker) for marker in marker_value if not common or marker in common
            ]
        _persist_tab6_settings()
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab6_batch_correction")

    def _apply_restored_file_metadata(df: pd.DataFrame, restored_records: list[dict]) -> pd.DataFrame:
        """Merge restored metadata values into the file dataframe."""
        if not restored_records or df.empty:
            return df
            
        restored_map = {r.get('Filename'): r for r in restored_records if 'Filename' in r}
        
        # We manipulate column values based on filename match
        # Using a loop for safety, though merge could be faster
        for i, row in df.iterrows():
            fname = str(row['Filename'])
            if fname in restored_map:
                rec = restored_map[fname]
                
                # 1. Process flag
                if 'Process' in rec:
                     df.at[i, 'Process'] = rec['Process']
                
                # 2. Metadata columns provided they exist in df
                for col_name, val in rec.items():
                    if col_name in df.columns and col_name not in ['Filename', 'Process']:
                        df.at[i, col_name] = val
        return df

    def _load_tab6_settings(dir_path: str):
        """Load settings for the given directory and update session state."""
        settings = load_directory_settings(dir_path, 'tab6')
        if not settings:
            return

        # Restore Configs
        if 'bc_config' in settings:
            st.session_state.bc_config.update(settings['bc_config'])
        # Restore Metadata Cols (this structure defines the columns for the table)
        if 'bc_metadata_cols' in settings:
            st.session_state.bc_metadata_cols = settings['bc_metadata_cols']
            
        # Restore Markers
        if 'bc_selected_markers' in settings:
            restored_markers = settings['bc_selected_markers']
            st.session_state.bc_selected_markers = list(restored_markers or [])
            # This key is a widget/cache key, so it is not serialized by the
            # project persistence layer.  Seed it from the saved parameter
            # whenever a directory-specific Tab 6 configuration is loaded.
            st.session_state["bc_marker_multiselect"] = list(st.session_state.bc_selected_markers)
            
        # Store file metadata in a temporary state to be applied during "Scan Files"
        if 'files_metadata' in settings:
            st.session_state.bc_restored_file_metadata = settings['files_metadata']
            if settings['files_metadata']:
                restored_df = pd.DataFrame(settings['files_metadata'])
                if "Filename" in restored_df.columns:
                    if "Process" not in restored_df.columns:
                        restored_df.insert(0, "Process", True)
                    restored_df["Process"] = restored_df["Process"].map(
                        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}
                    )
                    st.session_state.bc_files_df = restored_df
                    st.session_state.bc_files_df_draft = restored_df.copy()
            
        import os
        active_tab_env = os.getenv("_spacemap_active_tab", "")
        currently_rendering = st.session_state.get("_currently_rendering_tab", "")
        if currently_rendering == "tab6" or active_tab_env == "tab6":
            log_step(f"Restored settings for {dir_path}")

    input_mode = st.radio(
        "Input mode",
        ["Directory of CSV files", "Unified CSV file"],
        key="bc_input_mode",
        horizontal=True,
        help=(
            "Use one CSV per batch, or reuse an existing unified CSV and stratify cells "
            "by a selected batch column."
        ),
    )
    unified_mode = input_mode == "Unified CSV file"

    # Detect input source change. The identity includes the full file path in
    # unified mode so switching CSVs within one folder invalidates old headers.
    current_settings_root = (
        str(Path(st.session_state.bc_unified_csv_path).expanduser().parent)
        if unified_mode and st.session_state.bc_unified_csv_path
        else st.session_state.bc_input_dir
    )
    current_source_value = (
        str(Path(st.session_state.bc_unified_csv_path).expanduser())
        if unified_mode
        else st.session_state.bc_input_dir
    )
    current_source_identity = f"{input_mode}:{current_source_value}"
    last_loaded = st.session_state.get('bc_last_loaded_dir', None)
    
    if current_settings_root and current_source_identity != last_loaded:
        _load_tab6_settings(current_settings_root)
        # Common/union columns are derived from the selected files and are not
        # valid after changing the input directory.  Leaving them in session
        # state can make the marker picker display stale columns.
        for derived_key in (
            "bc_first_file_columns", "bc_common_columns", "bc_union_columns",
            "bc_common_markers",
        ):
            st.session_state[derived_key] = []
        st.session_state.bc_last_loaded_dir = current_source_identity
        # Derived headers are rebuilt below from the restored file table.  A
        # manual Scan Files action is still available to detect filesystem
        # additions/removals.
    
    if not unified_mode:
        directory_input_with_browse(
            label="Input Directory (CSVs)",
            text_key="bc_input_dir",
            default_value=st.session_state.bc_input_dir,
            browser_key_prefix="bc_in_browser",
            help_text="Directory containing one single-cell CSV per batch."
        )
    else:
        st.markdown("**Unified CSV File**")
        unified_path_col, unified_browse_col = st.columns([4, 1])
        with unified_path_col:
            st.text_input(
                "Unified CSV File",
                key="bc_unified_csv_path",
                label_visibility="collapsed",
                placeholder="/path/to/unified_cells.csv",
            )
        if unified_browse_col.button("Browse", key="bc_unified_csv_browse"):
            st.session_state.show_bc_unified_csv_browser = True
        if st.session_state.get("show_bc_unified_csv_browser"):
            picked_unified = render_file_browser(
                str(Path(st.session_state.bc_unified_csv_path).expanduser().parent),
                "bc_unified_csv_browser",
                "file",
            )
            if picked_unified == "CANCEL":
                st.session_state.show_bc_unified_csv_browser = False
                st.rerun()
            elif picked_unified:
                st.session_state.bc_unified_csv_path = picked_unified
                st.session_state.show_bc_unified_csv_browser = False
                st.rerun()

        unified_columns: list[str] = []
        unified_path = Path(st.session_state.bc_unified_csv_path).expanduser()
        if unified_path.is_file():
            try:
                unified_columns = list(pd.read_csv(unified_path, nrows=0).columns)
            except Exception as error:
                st.error(f"Could not read the unified CSV header: {error}")
        if unified_columns:
            saved_batch_column = st.session_state.bc_unified_batch_col
            batch_index = unified_columns.index(saved_batch_column) if saved_batch_column in unified_columns else 0
            batch_widget_key = "bc_unified_batch_col_select"
            if st.session_state.get(batch_widget_key) not in unified_columns:
                st.session_state[batch_widget_key] = unified_columns[batch_index]
            st.session_state.bc_unified_batch_col = st.selectbox(
                "Column used to split cells into batches",
                unified_columns,
                index=batch_index,
                key=batch_widget_key,
                help="Each unique non-empty value is treated like one input CSV in the existing workflow.",
            )
            st.session_state.bc_config['batch_id_col'] = st.session_state.bc_unified_batch_col
            st.session_state.bc_config['batch_id_is_auto'] = False
        st.warning(
            "Unified mode replaces this CSV atomically after correction. Existing corrected columns "
            "are replaced for the selected batches; all other rows and columns are preserved."
        )
    
    # Check again if input changed via widget this run
    current_settings_root = (
        str(Path(st.session_state.bc_unified_csv_path).expanduser().parent)
        if unified_mode and st.session_state.bc_unified_csv_path
        else st.session_state.bc_input_dir
    )
    current_source_value = (
        str(Path(st.session_state.bc_unified_csv_path).expanduser())
        if unified_mode
        else st.session_state.bc_input_dir
    )
    current_source_identity = f"{input_mode}:{current_source_value}"
    if current_source_identity != st.session_state.get('bc_last_loaded_dir', None):
         _load_tab6_settings(current_settings_root)
         for derived_key in (
             "bc_first_file_columns", "bc_common_columns", "bc_union_columns",
             "bc_common_markers",
         ):
             st.session_state[derived_key] = []
         st.session_state.bc_last_loaded_dir = current_source_identity

    if not unified_mode:
        directory_input_with_browse(
            label="Output Directory",
            text_key="bc_output_dir",
            default_value=st.session_state.bc_output_dir,
            browser_key_prefix="bc_out_browser",
            help_text="Directory where batch-corrected results will be saved."
        )



    # 2. File Selection & Metadata
    st.subheader("2. Batch Selection & Metadata" if unified_mode else "2. File Selection & Metadata")
    st.markdown(
        "Select batch values to process and add optional per-batch metadata."
        if unified_mode
        else "Select files to process and add optional metadata columns (e.g., Treatment, Response)."
    )

    def _coerce_bool_series(values: pd.Series) -> pd.Series:
        """Coerce mixed True/False representations to real booleans.

        Notes
        -----
        AgGrid can return boolean-looking values as strings (e.g., "True"/"False").
        """
        if values is None:
            return pd.Series([], dtype=bool)

        s = values.copy()
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)

        s_str = s.astype(str).str.strip().str.lower()
        truthy = {"true", "1", "yes", "y", "t"}
        falsy = {"false", "0", "no", "n", "f", "", "nan", "none"}

        out = pd.Series(False, index=s.index, dtype=bool)
        out.loc[s_str.isin(truthy)] = True
        out.loc[s_str.isin(falsy)] = False

        unknown = ~(s_str.isin(truthy) | s_str.isin(falsy))
        if unknown.any():
            coerced = pd.to_numeric(s_str.where(unknown), errors="coerce")
            out.loc[unknown] = coerced.fillna(0).astype(int).astype(bool)

        return out

    def _normalize_filename_for_match(value: str) -> str:
        """Normalize a file-like string for robust matching.

        Parameters
        ----------
        value
            Input string (e.g., a CSV filename or a metadata key).

        Returns
        -------
        str
            Normalized key for matching.
        """
        if value is None:
            return ""
        s = str(value).strip()
        s = s.replace("\\", "/")
        s = s.split("/")[-1]

        if st.session_state.get("bc_meta_strip_csv_ext", True):
            if s.lower().endswith(".csv"):
                s = s[:-4]

        if st.session_state.get("bc_meta_apply_suffix_stripping", True):
            suffixes_raw = st.session_state.bc_config.get("filename_suffixes_to_strip", "")
            suffixes = [x.strip() for x in str(suffixes_raw).split(",") if x.strip()]
            for suf in suffixes:
                suf_norm = suf
                if st.session_state.get("bc_meta_strip_csv_ext", True) and suf_norm.lower().endswith(".csv"):
                    suf_norm = suf_norm[:-4]
                if suf_norm and s.endswith(suf_norm):
                    s = s[: -len(suf_norm)]
                    break

        s = s.strip().lower()
        s = "".join(ch for ch in s if ch.isalnum() or ch in {"_", "-"})
        return s

    def _fuzzy_best_match(query: str, candidates: list[str]) -> tuple[int | None, float]:
        """Find the best fuzzy match index and score.

        Parameters
        ----------
        query
            Normalized query key.
        candidates
            Normalized candidate keys.

        Returns
        -------
        tuple
            (best_index, best_score) where score is in [0, 1].
        """
        if not query or not candidates:
            return None, 0.0

        best_idx: int | None = None
        best_score = 0.0
        for i, cand in enumerate(candidates):
            if not cand:
                continue
            if cand == query:
                return i, 1.0
            score = SequenceMatcher(None, query, cand).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, float(best_score)

    def _build_metadata_match_preview(
        files_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        anchor_col: str,
        min_score: float,
    ) -> pd.DataFrame:
        """Create a preview table mapping files to metadata rows.

        Parameters
        ----------
        files_df
            Current files table (must include 'Filename').
        metadata_df
            External metadata table.
        anchor_col
            Column in metadata_df used to match filenames.
        min_score
            Minimum score for accepting a match.

        Returns
        -------
        pandas.DataFrame
            Preview mapping with score and matched anchor value.
        """
        if files_df.empty or "Filename" not in files_df.columns:
            return pd.DataFrame(columns=["Filename", "metadata_match", "match_score"])
        if metadata_df.empty or anchor_col not in metadata_df.columns:
            return pd.DataFrame(columns=["Filename", "metadata_match", "match_score"])

        meta_anchor_values = metadata_df[anchor_col].astype(str).fillna("").tolist()
        meta_norm = [_normalize_filename_for_match(v) for v in meta_anchor_values]

        rows: list[dict[str, object]] = []
        for fname in files_df["Filename"].astype(str).fillna("").tolist():
            q = _normalize_filename_for_match(fname)
            idx, score = _fuzzy_best_match(q, meta_norm)
            match_val = meta_anchor_values[idx] if idx is not None else ""
            accepted = bool(idx is not None and score >= float(min_score))
            rows.append(
                {
                    "Filename": fname,
                    "metadata_match": match_val if accepted else "",
                    "match_score": score if accepted else 0.0,
                    "_meta_row_index": int(idx) if accepted and idx is not None else -1,
                }
            )

        preview = pd.DataFrame(rows)
        preview["matched"] = preview["_meta_row_index"].astype(int) >= 0
        return preview

    def _apply_metadata_merge(
        files_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        match_preview: pd.DataFrame,
        anchor_col: str,
    ) -> pd.DataFrame:
        """Merge external metadata columns into the files table.

        Notes
        -----
        - Matches are defined by `match_preview['_meta_row_index']`.
        - Columns with the same name are overwritten.

        Parameters
        ----------
        files_df
            Current files table.
        metadata_df
            External metadata table.
        match_preview
            Preview mapping from `_build_metadata_match_preview`.
        anchor_col
            Anchor column used in metadata_df.

        Returns
        -------
        pandas.DataFrame
            Updated files table.
        """
        if files_df.empty or metadata_df.empty or match_preview.empty:
            return files_df
        if "Filename" not in files_df.columns or "Filename" not in match_preview.columns:
            return files_df
        if "_meta_row_index" not in match_preview.columns:
            return files_df

        out_df = files_df.reset_index(drop=True).copy()
        metadata_cols = [c for c in metadata_df.columns if c != anchor_col and c not in {"Filename", "Process"}]
        for c in metadata_cols:
            if c not in out_df.columns:
                out_df[c] = ""

        col_indices = {c: int(out_df.columns.get_loc(c)) for c in metadata_cols}

        by_filename = match_preview.set_index("Filename")["_meta_row_index"].to_dict()
        for row_pos in range(len(out_df)):
            fname = str(out_df.at[row_pos, "Filename"])
            meta_idx = int(by_filename.get(fname, -1))
            if meta_idx < 0 or meta_idx >= len(metadata_df):
                continue
            meta_row = metadata_df.iloc[meta_idx]
            for c in metadata_cols:
                out_df.iloc[row_pos, col_indices[c]] = meta_row.get(c, "")

        return out_df

    def _read_csv_header_columns(csv_path):
        """Cache-safe wrapper: re-reads when the file on disk changes."""
        from spacemap_gui.ui_utils import file_signature
        return __read_csv_header_columns__impl(csv_path, file_signature(csv_path))


    @st.cache_data(ttl=300)
    def __read_csv_header_columns__impl(csv_path: str, signature) -> list[str]:
        """Read CSV header column order (no rows loaded)."""
        return list(pd.read_csv(csv_path, nrows=0).columns)

    def _compute_common_columns_for_selected_files(files_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """Return (first_file_columns, common_columns, union_columns) preserving order."""
        # Skip expensive computation if not currently in Tab 6 (avoid interference with other tabs)
        # Use both environment variable AND session_state for reliability
        import os
        active_tab_env = os.getenv("_spacemap_active_tab", "")
        active_tab_session = st.session_state.get("_active_tab_session", "")
        
        is_in_tab6 = True
        
        if not is_in_tab6:
            # Return cached result if available, otherwise empty
            if 'bc_cached_columns' in st.session_state:
                cached = st.session_state['bc_cached_columns']
                if len(cached) == 2:
                    return cached[0], cached[1], cached[0]
                return cached
            return [], [], []
        
        if files_df.empty or "Process" not in files_df.columns or "Filename" not in files_df.columns:
            return [], [], []

        # Fix: Force boolean conversion again to be absolutely sure
        df_clean = files_df.copy()
        if "Process" in df_clean.columns:
            df_clean["Process"] = _coerce_bool_series(df_clean["Process"])

        selected = df_clean[df_clean["Process"] == True]  # noqa: E712
        if selected.empty:
            log_step("Common cols: No files selected (Process=True). Returning empty.")
            result = ([], [], [])
            st.session_state['bc_cached_columns'] = result
            return result

        if unified_mode:
            try:
                columns = _read_csv_header_columns(st.session_state.bc_unified_csv_path)
            except Exception as error:
                log_step(f"Error reading unified CSV header: {error}")
                return [], [], []
            result = (columns, columns, columns)
            st.session_state['bc_cached_columns'] = result
            return result

        cols_list: list[list[str]] = []
        # Limit to checking first 5 files if many are selected to speed up UI, 
        # BUT for correctness we should arguably check all.
        # Let's check ALL but stop if we find 0 common quite fast.
        
        filenames = selected["Filename"].astype(str).tolist()
        log_step(f"Computing common and union columns for {len(filenames)} files...")
        
        for i, fname in enumerate(filenames):
            # Check direct path first
            fpath = os.path.join(st.session_state.bc_input_dir, fname)
            
            # If not found, try to strip potential extra whitespace or handle encoding
            if not os.path.exists(fpath):
                # Fallback: check if the file exists with slightly different spacing or in subdirectory if user navigated inside
                candidates = []
                if os.path.isdir(st.session_state.bc_input_dir):
                    candidates = os.listdir(st.session_state.bc_input_dir)
                
                # Try exact match in list
                if fname in candidates:
                     fpath = os.path.join(st.session_state.bc_input_dir, fname)
                else:
                    log_step(f"MISSING FILE during column scan: {fpath}")
                    # Try to log one candidate to see if paths are messed up
                    if i == 0:
                        log_step(f"Diagnostic: First 3 files in dir: {candidates[:3]}")
                        log_step(f"Diagnostic: Looking for: '{fname}'")
                    continue
            
            try:
                # Use cached reader
                cols = _read_csv_header_columns(fpath)
                cols_list.append(cols)
            except Exception as e:
                log_step(f"Error reading header for {fname}: {e}")
                continue

        if not cols_list:
            log_step("No headers could be read.")
            result = ([], [], [])
            st.session_state['bc_cached_columns'] = result
            return result

        first_cols = cols_list[0]
        common = set(first_cols)
        union = list(first_cols)
        union_set = set(first_cols)
        for cols in cols_list[1:]:
            common &= set(cols)
            for c in cols:
                if c not in union_set:
                    union.append(c)
                    union_set.add(c)
        
        common_in_order = [c for c in first_cols if c in common]
        log_step(f"Found {len(common_in_order)} common columns, {len(union)} union columns.")
        result = (first_cols, common_in_order, union)
        st.session_state['bc_cached_columns'] = result
        return result

    def _auto_pick_markers(common_columns: list[str]) -> list[str]:
        """Use Tab 7's default phenotyping panel, with heuristic fallback."""
        return _pick_batch_marker_defaults(
            common_columns,
            st.session_state.bc_config,
            st.session_state.bc_metadata_cols,
        )

    def _refresh_common_columns(files_df: pd.DataFrame) -> None:
        """Rebuild derived column lists and preserve saved marker choices."""
        first_cols, common_cols, union_cols = _compute_common_columns_for_selected_files(files_df)
        st.session_state.bc_first_file_columns = first_cols
        st.session_state.bc_common_columns = common_cols
        st.session_state.bc_union_columns = union_cols
        st.session_state.bc_common_markers = _auto_pick_markers(common_cols)

        selected = [
            marker for marker in st.session_state.get("bc_selected_markers", [])
            if marker in common_cols
        ]
        if not selected:
            selected = list(st.session_state.bc_common_markers)
        st.session_state.bc_selected_markers = selected
        st.session_state["bc_marker_multiselect"] = list(selected)

    st.markdown("**Add Metadata Column**")
    col_scan, col_add_name, col_add_btn = st.columns([1, 2, 1])
    
    scan_label = "Scan Batches" if unified_mode else "Scan Files"
    if col_scan.button(scan_label):
        source_label = st.session_state.bc_unified_csv_path if unified_mode else st.session_state.bc_input_dir
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BATCH_CORRECTION: Scanning '{source_label}'", flush=True)
        source_is_valid = False
        files: list[str] = []
        if unified_mode:
            unified_path = Path(st.session_state.bc_unified_csv_path).expanduser()
            batch_column = st.session_state.bc_unified_batch_col
            source_is_valid = unified_path.is_file() and bool(batch_column)
            if source_is_valid:
                try:
                    files = _read_unified_batch_values(unified_path, batch_column)
                except Exception as error:
                    st.error(f"Could not scan batch values from the unified CSV: {error}")
                    source_is_valid = False
        else:
            source_is_valid = os.path.isdir(st.session_state.bc_input_dir)
            if source_is_valid:
                files = sorted(
                    file_name for file_name in os.listdir(st.session_state.bc_input_dir)
                    if file_name.lower().endswith('.csv')
                )

        if source_is_valid:
            
            # Current state
            current_df = st.session_state.bc_files_df
            
            # 1. Filter existing rows: Keep only if filename exists in new scan
            if not current_df.empty and "Filename" in current_df.columns:
                current_df = current_df[current_df["Filename"].isin(files)]
            else:
                current_df = pd.DataFrame(columns=["Process", "Filename"] + st.session_state.bc_metadata_cols)
            
            # 2. Find new files not in filtered dataframe
            existing_files = set(current_df["Filename"].tolist()) if not current_df.empty else set()
            new_files = [f for f in files if f not in existing_files]
            
            # Initialize updated_df to filtered current state
            updated_df = current_df
            
            # 3. Create rows for new files
            if new_files:
                new_data = [{"Process": True, "Filename": f} for f in new_files]
                # Initialize metadata cols with empty strings
                for col in st.session_state.bc_metadata_cols:
                    for row in new_data:
                        row[col] = ""
                
                new_df = pd.DataFrame(new_data)
                updated_df = pd.concat([current_df, new_df], ignore_index=True)
                item_label = "batch values" if unified_mode else "files"
                st.success(f"List updated. Found {len(new_files)} new {item_label}.")
            else:
                if len(st.session_state.bc_files_df) > len(updated_df):
                    st.success("List updated. Removed missing files.")
                else:
                    st.info("No new files found.")
            
            # Apply restored metadata if available for this directory
            if 'bc_restored_file_metadata' in st.session_state:
                updated_df = _apply_restored_file_metadata(updated_df, st.session_state.bc_restored_file_metadata)
                
            st.session_state.bc_files_df = updated_df
            # CRITICAL FIX: Update the draft so AgGrid sees the new files!
            if "bc_files_df_draft" in st.session_state:
                st.session_state.bc_files_df_draft = updated_df.copy()
            
            st.session_state.bc_aggrid_key_nonce = int(st.session_state.bc_aggrid_key_nonce) + 1

            # Normalize Process column + keep Process All toggle aligned
            if "Process" in st.session_state.bc_files_df.columns:
                st.session_state.bc_files_df["Process"] = _coerce_bool_series(st.session_state.bc_files_df["Process"])
                all_selected = bool(st.session_state.bc_files_df["Process"].all()) if len(st.session_state.bc_files_df) else False
                st.session_state.prev_bc_process_all = all_selected

            # Compute common and union columns across selected CSVs (preserve header order).
            _refresh_common_columns(st.session_state.bc_files_df)
            
            # Auto-detect Tissue Column from the source data
            if files:
                try:
                    preview_path = (
                        st.session_state.bc_unified_csv_path
                        if unified_mode
                        else os.path.join(st.session_state.bc_input_dir, files[0])
                    )
                    first_df = pd.read_csv(preview_path, nrows=5)
                    potential_tissue = [c for c in first_df.columns if any(x in c.lower() for x in ['tissue', 'region', 'class'])]
                    if potential_tissue:
                        st.session_state.bc_config['tissue_col'] = potential_tissue[0]
                        st.toast(f"Detected tissue column: {potential_tissue[0]}")
                except Exception as e:
                    from spacemap_gui.ui_utils import log_exception
                    log_exception("auto-detecting tissue column", e)
            
            _persist_tab6_settings()
            st.rerun()
        else:
            st.error("Select a readable unified CSV and batch column." if unified_mode else "Invalid input directory.")

    # New column controls
    new_col_name = col_add_name.text_input("New Column Name", label_visibility="collapsed", placeholder="Enter column name (e.g. Treatment)")
    
    if col_add_btn.button("Add Column"):
        if new_col_name and new_col_name not in st.session_state.bc_files_df.columns:
            st.session_state.bc_metadata_cols.append(new_col_name)
            st.session_state.bc_files_df[new_col_name] = ""
            st.session_state.bc_aggrid_key_nonce = int(st.session_state.bc_aggrid_key_nonce) + 1
            _persist_tab6_settings()
            st.rerun()
        elif new_col_name in st.session_state.bc_files_df.columns:
            st.warning("Column already exists.")

    # ------------------------------------------------------------------
    # Group images into correction units
    # ------------------------------------------------------------------
    # Batch correction should run once per *processing* unit -- one staining or
    # acquisition run. That is often not one row per image: several images
    # frequently come from the same slide/run. Correcting each image on its own
    # forces them all to a common profile, which deletes any real difference
    # between them (different tissue regions, different samples on one slide).
    # This helper writes a grouping column that the correction step can use.
    with st.expander("Group images into correction units (slide / run)", expanded=False):
        st.markdown(
            "Batch correction removes differences **between** the units you define here, "
            "and preserves everything **inside** a unit. That single sentence decides the "
            "whole setting.\n\n"
            "**The rule:** one unit per *processing* run — the group of images that were "
            "stained and imaged together. Ask of any two images: *could these differ simply "
            "because they were stained separately?* If no, they belong in the **same** unit.\n\n"
            "Splitting more finely than the real processing run is the common and costly "
            "mistake. Every biological difference inside a unit is normalised away, so an "
            "over-split setting quietly deletes the signal you are studying."
        )

        st.markdown("**Common layouts**")
        st.table(
            pd.DataFrame(
                [
                    {
                        "Layout": "One sample per slide",
                        "Correction unit": "The slide (= the image)",
                        "What is preserved": "Nothing is nested inside, so all sample "
                        "differences are between units and get corrected. Protect study "
                        "variables with covariates.",
                    },
                    {
                        "Layout": "Several regions / ROIs imaged from one slide",
                        "Correction unit": "The slide (NOT each region)",
                        "What is preserved": "Region-to-region and any sample-to-sample "
                        "differences within that slide stay intact.",
                    },
                    {
                        "Layout": "TMA: many cores on one slide, one patient per core",
                        "Correction unit": "The TMA slide (NOT each core)",
                        "What is preserved": "All patient-to-patient differences, because "
                        "every patient shares one staining run.",
                    },
                    {
                        "Layout": "Several fields of view per core / region",
                        "Correction unit": "Still the slide",
                        "What is preserved": "Field-to-field and core-to-core variation.",
                    },
                    {
                        "Layout": "One slide restained or reimaged on another day",
                        "Correction unit": "Each staining session separately",
                        "What is preserved": "Differences within a session; the session "
                        "shift is removed.",
                    },
                ]
            ),
        )

        st.markdown(
            "**Worked example — a tissue microarray.** Two TMA slides, ~100 cores each, one "
            "patient per core. The correction unit is the **slide**, giving two units. Every "
            "core on a slide went through the same staining, so differences between cores are "
            "patient biology and are preserved. Setting the unit to the *core* instead would "
            "force all ~200 patients onto a common marker profile and erase precisely the "
            "between-patient variation the study is measuring.\n\n"
            "A TMA is the favourable case: because many patients share each slide, patient and "
            "slide are not confounded, so patient comparisons survive the correction. Contrast "
            "that with one-patient-per-slide designs, where the two cannot be separated at all "
            "and some genuine patient difference is unavoidably removed along with the "
            "staining difference.\n\n"
            "Two practical notes for TMAs. With only a handful of units the estimate leans on "
            "the fact that it is computed within matched cell populations — that is what stops "
            "a slide holding more of one tissue type from being mistaken for a slide that was "
            "stained brighter. And a damaged core (folded, torn, badly stained) is a quality "
            "problem: drop it, rather than splitting it into its own unit, since a unit of one "
            "is simply normalised to the group average and the artefact is hidden instead of "
            "removed.\n\n"
            "Fill the column automatically from the image names below, then correct any row by "
            "hand in the table above. Select this column as the **Correction unit column** in "
            "the Configuration section."
        )

        grp_c1, grp_c2 = st.columns([1, 1])
        group_col_name = grp_c1.text_input(
            "Grouping column name",
            value="slide_id",
            key="bc_group_col_name",
            help="Name of the metadata column to create or overwrite with the unit each image belongs to.",
        )
        group_pattern = grp_c2.text_input(
            "Remove this pattern from the name to get the unit",
            value=r"_roi_\d+$",
            key="bc_group_pattern",
            help=(
                "Regular expression stripped from each image name to produce its unit. "
                r"Examples: `_roi_\d+$` turns `slideA_roi_3` into `slideA`; "
                r"`\.csv$` strips a file extension; `_\d+$` removes any trailing number. "
                "Leave empty to create the column blank and assign every row yourself."
            ),
        )

        if st.button("Fill grouping column from names", key="bc_group_apply"):
            files_df = st.session_state.bc_files_df
            if files_df.empty or "Filename" not in files_df.columns:
                st.warning("Scan the input source first so there are images to group.")
            elif not group_col_name:
                st.warning("Enter a name for the grouping column.")
            else:
                names = files_df["Filename"].astype(str)
                if group_pattern:
                    try:
                        derived = names.str.replace(group_pattern, "", regex=True)
                    except re.error as exc:
                        derived = None
                        st.error(f"Invalid pattern: {exc}")
                else:
                    derived = pd.Series([""] * len(names), index=names.index)

                if derived is not None:
                    st.session_state.bc_files_df[group_col_name] = derived.values
                    if group_col_name not in st.session_state.bc_metadata_cols:
                        st.session_state.bc_metadata_cols.append(group_col_name)
                    if "bc_files_df_draft" in st.session_state:
                        st.session_state.bc_files_df_draft = st.session_state.bc_files_df.copy()
                    st.session_state.bc_aggrid_key_nonce = int(st.session_state.bc_aggrid_key_nonce) + 1
                    _persist_tab6_settings()
                    n_units = int(derived.replace("", pd.NA).nunique())
                    st.success(
                        f"'{group_col_name}' filled: {len(names)} images -> {n_units} unit(s). "
                        "Review the table above and edit any row that grouped incorrectly."
                    )
                    st.rerun()

        # Show what the current grouping looks like so mistakes are visible before running.
        files_df = st.session_state.bc_files_df
        if group_col_name and not files_df.empty and group_col_name in files_df.columns:
            preview = (
                files_df.assign(**{group_col_name: files_df[group_col_name].astype(str)})
                .groupby(group_col_name)["Filename"]
                .agg(images="size", examples=lambda s: ", ".join(sorted(s)[:4]))
                .reset_index()
                .sort_values(group_col_name)
            )
            singletons = int((preview["images"] == 1).sum())
            st.caption(
                f"Current grouping: {len(preview)} unit(s) across {len(files_df)} images."
                + (
                    f" {singletons} unit(s) contain a single image — check that this is intended, "
                    "since a unit of one gets normalised to the group average on its own."
                    if singletons
                    else ""
                )
            )
            st.dataframe(preview, width="stretch", hide_index=True)

    st.markdown("**Remove Metadata Column**")
    
    # Source options directly from the dataframe to ensure we catch ALL columns
    # (except system columns Process/Filename)
    current_cols = []
    if not st.session_state.bc_files_df.empty:
        current_cols = [
            c for c in st.session_state.bc_files_df.columns 
            if c not in ["Process", "Filename", "original_filename"]
        ]
    
    # Sync metadata_cols list with reality just in case
    # This fixes the issue where "remove list" is out of sync with "table"
    st.session_state.bc_metadata_cols = [c for c in st.session_state.bc_metadata_cols if c in current_cols]
    for c in current_cols:
        if c not in st.session_state.bc_metadata_cols:
             st.session_state.bc_metadata_cols.append(c)

    cols_to_remove = st.multiselect(
        "Select columns to remove",
        options=current_cols, # Use dynamic list from DF
        default=[],
        help="Select columns to delete from the file list."
    )

    if st.button("Remove Selected Columns"):
        if cols_to_remove:
            # Drop from dataframe
            if not st.session_state.bc_files_df.empty:
                st.session_state.bc_files_df = st.session_state.bc_files_df.drop(columns=cols_to_remove)
                
            # Drop from metadata list
            st.session_state.bc_metadata_cols = [
                c for c in st.session_state.bc_metadata_cols if c not in cols_to_remove
            ]
            
            # Update draft if exists
            if "bc_files_df_draft" in st.session_state and isinstance(st.session_state.bc_files_df_draft, pd.DataFrame):
                 try:
                     st.session_state.bc_files_df_draft = st.session_state.bc_files_df_draft.drop(columns=cols_to_remove, errors='ignore')
                 except Exception as e:
                     from spacemap_gui.ui_utils import log_exception
                     log_exception("dropping columns from bc_files_df_draft", e)

            st.success(f"Removed columns: {', '.join(cols_to_remove)}")
            _persist_tab6_settings()
            st.rerun()
        else:
            st.info("No columns selected.")


    st.markdown("### Import metadata CSV (optional)")
    st.caption(
        "Load a CSV containing per-file metadata (e.g., timepoint/arm/response), "
        "pick the anchor column, then fuzzy-match rows to the discovered files."
    )

    if "bc_metadata_csv_path" not in st.session_state:
        st.session_state.bc_metadata_csv_path = ""
    if "show_bc_metadata_csv_browser" not in st.session_state:
        st.session_state.show_bc_metadata_csv_browser = False
    if "bc_metadata_import_df" not in st.session_state:
        st.session_state.bc_metadata_import_df = None
    if "bc_metadata_anchor_col" not in st.session_state:
        st.session_state.bc_metadata_anchor_col = ""
    if "bc_metadata_match_preview" not in st.session_state:
        st.session_state.bc_metadata_match_preview = None
    if "bc_meta_min_score" not in st.session_state:
        st.session_state.bc_meta_min_score = 0.85
    if "bc_meta_strip_csv_ext" not in st.session_state:
        st.session_state.bc_meta_strip_csv_ext = True
    if "bc_meta_apply_suffix_stripping" not in st.session_state:
        st.session_state.bc_meta_apply_suffix_stripping = True

    st.markdown("**Metadata CSV**", help="Path to the metadata CSV file.")
    m1, m2, m3 = st.columns([4, 1, 1])
    with m1:
        meta_path = st.text_input(
            "Metadata CSV",
            value=st.session_state.bc_metadata_csv_path,
            placeholder="/path/to/metadata.csv",
            key="bc_metadata_csv_path_input",
            label_visibility="collapsed"
        )
    if meta_path != st.session_state.bc_metadata_csv_path:
        st.session_state.bc_metadata_csv_path = meta_path

    if m2.button("Browse", key="bc_browse_metadata_csv"):
        st.session_state.show_bc_metadata_csv_browser = True

    if st.session_state.get("show_bc_metadata_csv_browser"):
        picked = render_file_browser(
            str(Path(st.session_state.bc_input_dir).resolve()),
            "bc_metadata_csv_browser",
            "file",
        )
        if picked == "CANCEL":
            st.session_state.show_bc_metadata_csv_browser = False
            st.rerun()
        elif picked:
            st.session_state.bc_metadata_csv_path = picked
            st.session_state.show_bc_metadata_csv_browser = False
            st.rerun()

    load_clicked = m3.button("Load", key="bc_load_metadata_csv")
    if load_clicked:
        try:
            md = pd.read_csv(st.session_state.bc_metadata_csv_path)
            if md.empty:
                st.warning("Metadata CSV loaded, but it is empty.")
            st.session_state.bc_metadata_import_df = md
            preferred = "Filename" if "Filename" in md.columns else "file" if "file" in md.columns else ""
            st.session_state.bc_metadata_anchor_col = preferred or str(md.columns[0])
            st.session_state.bc_metadata_match_preview = None
            st.success(f"Loaded metadata CSV with {md.shape[0]} rows and {md.shape[1]} columns.")
            _persist_tab6_settings()
        except Exception as e:
            st.session_state.bc_metadata_import_df = None
            st.session_state.bc_metadata_match_preview = None
            st.error(f"Failed to load metadata CSV: {e}")

    md_df = st.session_state.bc_metadata_import_df
    if isinstance(md_df, pd.DataFrame) and not md_df.empty:
        a1, a2, a3, a4 = st.columns([2, 2, 2, 2])
        anchor_col = a1.selectbox(
            "Anchor column",
            options=list(md_df.columns),
            index=max(0, list(md_df.columns).index(st.session_state.bc_metadata_anchor_col))
            if st.session_state.bc_metadata_anchor_col in md_df.columns
            else 0,
            key="bc_metadata_anchor_select",
        )
        st.session_state.bc_metadata_anchor_col = anchor_col

        a2.slider(
            "Min match score",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.bc_meta_min_score),
            step=0.01,
            key="bc_meta_min_score",
        )
        a3.checkbox(
            "Strip .csv extension",
            value=bool(st.session_state.bc_meta_strip_csv_ext),
            key="bc_meta_strip_csv_ext",
        )
        a4.checkbox(
            "Apply suffix stripping",
            value=bool(st.session_state.bc_meta_apply_suffix_stripping),
            key="bc_meta_apply_suffix_stripping",
            help="Uses the suffix list from 'Filename suffixes to strip'.",
        )

        p1, p2 = st.columns([1, 1])
        if p1.button("Preview matching", key="bc_preview_metadata_match_btn"):
            preview = _build_metadata_match_preview(
                st.session_state.bc_files_df,
                md_df,
                anchor_col=anchor_col,
                min_score=float(st.session_state.bc_meta_min_score),
            )
            st.session_state.bc_metadata_match_preview = preview
            _persist_tab6_settings()

        preview_df = st.session_state.bc_metadata_match_preview
        if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
            matched_n = int(preview_df["matched"].sum())
            total_n = int(len(preview_df))
            st.write(f"Matched {matched_n}/{total_n} files.")
            st.dataframe(preview_df[["Filename", "metadata_match", "match_score", "matched"]], width='stretch')

        if p2.button("Apply metadata to grid", type="primary", key="bc_apply_metadata_to_grid"):
            if not isinstance(st.session_state.bc_metadata_match_preview, pd.DataFrame) or st.session_state.bc_metadata_match_preview.empty:
                st.error("Run 'Preview matching' first.")
            else:
                base = (
                    st.session_state.bc_files_df_draft.copy()
                    if "bc_files_df_draft" in st.session_state and isinstance(st.session_state.bc_files_df_draft, pd.DataFrame)
                    else st.session_state.bc_files_df.copy()
                )
                merged = _apply_metadata_merge(
                    base,
                    md_df,
                    st.session_state.bc_metadata_match_preview,
                    anchor_col=anchor_col,
                )

                # Update metadata column list and dataframes
                new_cols = [
                    c
                    for c in merged.columns
                    if c not in {"Process", "Filename"} and c not in st.session_state.bc_metadata_cols
                ]
                st.session_state.bc_metadata_cols.extend(new_cols)
                st.session_state.bc_files_df = merged
                st.session_state.bc_files_df["Process"] = _coerce_bool_series(st.session_state.bc_files_df["Process"])
                st.session_state.bc_files_df_draft = merged.copy()
                st.session_state.bc_aggrid_key_nonce = int(st.session_state.bc_aggrid_key_nonce) + 1
                st.success("Metadata applied to grid (columns overwritten when names matched).")
                _persist_tab6_settings()
                st.rerun()



    # Data Editor
    if not st.session_state.bc_files_df.empty:
        st.markdown("### File List & Metadata")

        # Normalize Process column (AgGrid may store strings)
        if "Process" in st.session_state.bc_files_df.columns:
            st.session_state.bc_files_df["Process"] = _coerce_bool_series(st.session_state.bc_files_df["Process"])
        if "bc_files_df_draft" in st.session_state and isinstance(st.session_state.bc_files_df_draft, pd.DataFrame):
            if "Process" in st.session_state.bc_files_df_draft.columns:
                st.session_state.bc_files_df_draft["Process"] = _coerce_bool_series(st.session_state.bc_files_df_draft["Process"])
        
        # Process All Toggle - OUTSIDE form for real-time update
        process_all = st.checkbox(
            "Process All",
            value=bool(st.session_state.get('prev_bc_process_all', True)),
            key="bc_process_all_checkbox",
        )
        
        def _save_and_recompute_state():
            """Helper to save table state and recompute columns."""
            # 1. Coerce Process column
            if "Process" in st.session_state.bc_files_df.columns:
                st.session_state.bc_files_df["Process"] = _coerce_bool_series(st.session_state.bc_files_df["Process"])
                all_selected = bool(st.session_state.bc_files_df["Process"].all()) if len(st.session_state.bc_files_df) else False
                st.session_state.prev_bc_process_all = all_selected

            # 2. Log Process state
            proc_counts = st.session_state.bc_files_df["Process"].value_counts()
            log_step(f"Save Changes: Process counts: {proc_counts.to_dict()}")

            # 3. Explicitly recompute here to populate state immediately
            first_cols, common_cols, union_cols = _compute_common_columns_for_selected_files(st.session_state.bc_files_df)
            st.session_state.bc_first_file_columns = first_cols
            st.session_state.bc_common_columns = common_cols
            st.session_state.bc_union_columns = union_cols
            st.session_state.bc_common_markers = _auto_pick_markers(common_cols)
            
            # 4. If we lost common cols, log why
            if not common_cols:
                    log_step("WARNING: No common columns found after Save Changes.")
                    
            st.session_state.bc_selected_markers = [
                m for m in st.session_state.bc_selected_markers if m in common_cols
            ]
            
            # 5. Synchronize bc_metadata_cols with the actual columns in bc_files_df
            st.session_state.bc_metadata_cols = [
                c for c in st.session_state.bc_files_df.columns 
                if c not in ["Process", "Filename", "original_filename"]
            ]

            st.success("Saved.")
            _persist_tab6_settings()
            st.rerun()

        if 'prev_bc_process_all' not in st.session_state:
            st.session_state.prev_bc_process_all = True
            
        # Apply Process All immediately
        if process_all != st.session_state.prev_bc_process_all:
            st.session_state.bc_files_df["Process"] = bool(process_all)
            if "bc_files_df_draft" in st.session_state and isinstance(st.session_state.bc_files_df_draft, pd.DataFrame):
                st.session_state.bc_files_df_draft["Process"] = bool(process_all)
            st.session_state.prev_bc_process_all = process_all
            # Force AgGrid to reload new data (otherwise it may keep stale internal state)
            st.session_state.bc_aggrid_key_nonce = int(st.session_state.bc_aggrid_key_nonce) + 1
            _persist_tab6_settings()
            st.rerun()

        st.caption("Edit metadata below. Click 'Save Changes' to apply updates.")

        # Prepare dataframe for editor
        df_editor = st.session_state.bc_files_df.copy()
        
        # Ensure Process is bool
        if "Process" in df_editor.columns:
            df_editor["Process"] = _coerce_bool_series(df_editor["Process"])
            # Move Process/Filename to front
            cols = ["Process", "Filename"] + [c for c in df_editor.columns if c not in ["Process", "Filename"]]
            df_editor = df_editor[cols]

        st.info("💡 You can copy-paste from Excel directly into this table.")
        
        with st.form("bc_file_list_form"):
            edited_df = st.data_editor(
                df_editor,
                key=f"bc_data_editor_{st.session_state.bc_aggrid_key_nonce}",
                width='stretch',
                num_rows="fixed",
                disabled=["Filename"], 
                hide_index=True,
                column_config={
                    "Process": st.column_config.CheckboxColumn("Process", help="Select to include in batch correction"),
                    "Filename": st.column_config.TextColumn(
                        "Batch Value" if unified_mode else "Filename",
                        disabled=True,
                    ),
                }
            )
            
            submit_changes = st.form_submit_button("Save Changes", type="primary")

        if submit_changes:
            st.session_state.bc_files_df = edited_df.copy()
            
            # Sync metadata cols
            new_meta = [c for c in edited_df.columns if c not in ["Process", "Filename", "original_filename"]]
            st.session_state.bc_metadata_cols = new_meta
            
            _save_and_recompute_state()

            with st.expander("Bulk paste (Excel)", expanded=False):
                df_current = st.session_state.bc_files_df.reset_index(drop=True).copy()
                editable_cols = ["Process", "Filename", *st.session_state.bc_metadata_cols]
                st.caption(
                    "Copy from Excel, then paste here. This fills down rows reliably."
                )

                paste_mode = st.radio(
                    "Paste mode",
                    ["Single column", "Block (tab-separated)"],
                    horizontal=True,
                    key="bc_paste_mode",
                )

                with st.form("bc_bulk_paste_form"):
                    target_col = None
                    target_cols = []
                    c_p1, c_p2, c_p3 = st.columns([2, 1, 1])
                    apply_only_selected = c_p1.checkbox(
                        "Apply only to rows where Process=True",
                        value=True,
                        key="bc_paste_only_selected",
                    )
                    clamp_numeric_widget_state(
                        "bc_paste_start_row", 0, max(len(df_current) - 1, 0), fallback=0
                    )
                    start_row = int(
                        c_p2.number_input(
                            "Start row (0-based)",
                            min_value=0,
                            max_value=max(len(df_current) - 1, 0),
                            value=0,
                            step=1,
                            key="bc_paste_start_row",
                        )
                    )
                    overwrite_blanks_only = c_p3.checkbox(
                        "Only fill blank cells",
                        value=False,
                        key="bc_paste_blanks_only",
                    )

                    if paste_mode == "Single column":
                        target_col = st.selectbox(
                            "Target column",
                            options=[c for c in editable_cols if c not in ["Filename"]],
                            index=0,
                            key="bc_paste_target_col",
                        )
                    else:
                        target_cols = st.multiselect(
                            "Target columns (for headerless blocks)",
                            options=[c for c in editable_cols if c not in ["Filename"]],
                            default=[c for c in st.session_state.bc_metadata_cols[:1] if c],
                            key="bc_paste_target_cols",
                            help="If your pasted block includes a header row, leave this empty and we'll match by header names.",
                        )

                    pasted_text = st.text_area(
                        "Paste from Excel",
                        value="",
                        height=160,
                        key="bc_paste_text",
                        help="Excel copy typically produces tab-separated text. One column copy is newline-separated.",
                    )

                    submitted = st.form_submit_button("Apply pasted values", type="primary")

                if submitted:
                    if not pasted_text.strip():
                        st.error("Nothing to paste.")
                    else:
                        if apply_only_selected and "Process" in df_current.columns:
                            row_indices = df_current.index[df_current["Process"] == True].tolist()  # noqa: E712
                        else:
                            row_indices = df_current.index.tolist()

                        if start_row >= len(row_indices):
                            st.error("Start row is beyond the available rows.")
                        else:
                            row_indices = row_indices[start_row:]

                            try:
                                if paste_mode == "Single column":
                                    if target_col is None:
                                        st.error("Please select a target column.")
                                        st.stop()

                                    values = [v for v in pasted_text.splitlines()]
                                    values = [v.rstrip("\r") for v in values]
                                    values = [v for v in values if v != ""]

                                    if not values:
                                        st.error("No values detected.")
                                    else:
                                        n = min(len(values), len(row_indices))
                                        for i in range(n):
                                            idx = int(row_indices[i])
                                            if overwrite_blanks_only:
                                                current_val = df_current.at[idx, target_col]
                                                if pd.isna(current_val) or str(current_val) == "":
                                                    df_current.at[idx, target_col] = values[i]
                                            else:
                                                df_current.at[idx, target_col] = values[i]

                                        st.session_state.bc_files_df = df_current
                                        st.success(f"Pasted {n} values into '{target_col}'.")
                                        _persist_tab6_settings()
                                        st.rerun()

                                else:
                                    block_df = pd.read_csv(
                                        io.StringIO(pasted_text.strip()),
                                        sep="\t",
                                        engine="python",
                                    )

                                    if block_df.shape[1] == 1:
                                        st.error("Detected a single column. Use 'Single column' mode.")
                                    else:
                                        header_matches = [
                                            c for c in block_df.columns if c in df_current.columns and c != "Filename"
                                        ]
                                        use_cols = header_matches
                                        if not use_cols:
                                            if not target_cols:
                                                st.error(
                                                    "No matching headers found. Include headers or select target columns."
                                                )
                                            elif len(target_cols) != block_df.shape[1]:
                                                st.error(
                                                    "Target columns count must match pasted block column count "
                                                    f"({len(target_cols)} vs {block_df.shape[1]})."
                                                )
                                            else:
                                                use_cols = list(target_cols)
                                                block_df.columns = use_cols

                                        if use_cols:
                                            n = min(len(block_df), len(row_indices))
                                            for i in range(n):
                                                idx = int(row_indices[i])
                                                for col in use_cols:
                                                    new_val = block_df.iloc[i][col]
                                                    if overwrite_blanks_only:
                                                        current_val = df_current.at[idx, col]
                                                        if pd.isna(current_val) or str(current_val) == "":
                                                            df_current.at[idx, col] = new_val
                                                    else:
                                                        df_current.at[idx, col] = new_val

                                            st.session_state.bc_files_df = df_current
                                            st.success(
                                                f"Pasted {n} rows into columns: {', '.join(use_cols)}."
                                            )
                                            _persist_tab6_settings()
                                            st.rerun()
                            except Exception as e:
                                st.error(f"Could not parse pasted content: {e}")
    else:
        st.info(
            "Click 'Scan Batches' to load unique values from the unified CSV."
            if unified_mode
            else "Click 'Scan Files' to load CSVs from the input directory."
        )

    # Common columns + marker selection (notebook-style requirement: markers must exist in all files)
    if not st.session_state.bc_files_df.empty:
        st.subheader("3. Common Columns & Marker Selection")

        # IMPORTANT: Do NOT auto-scan file headers during background reruns.
        # Scanning should only happen on explicit user actions (Scan Files / Save Changes / Force Scan Columns).
        # This prevents Tab 6 from spamming terminal logs when the user is working in other tabs.

        if not st.session_state.bc_common_columns:
            # Derived header lists are intentionally not stored in project JSON.
            # Rebuild them from the restored file table after F5 so the saved
            # marker selection is immediately usable without a manual rescan.
            _refresh_common_columns(st.session_state.bc_files_df)

        if not st.session_state.bc_common_columns:
            st.info("No common columns detected yet. Please ensure files are selected (Process=True) and readable.")
            if st.button("Force Scan Columns"):
                _refresh_common_columns(st.session_state.bc_files_df)
                _persist_tab6_settings()
                st.rerun()
        else:
            st.caption(
                "Marker defaults come from the same default phenotyping profile used by Tab 7. "
                "You can freely add or remove any numeric/common column before correction."
            )
            
            # This is deliberately outside a form.  Form values are not sent
            # to Streamlit until that form's submit button is clicked, which
            # made an adjacent Save Project button persist the previous list.
            marker_key = "bc_marker_multiselect"
            _bc_defaults = [
                marker for marker in st.session_state.get("bc_selected_markers", [])
                if marker in st.session_state.bc_common_columns
            ]
            if marker_key not in st.session_state:
                st.session_state[marker_key] = list(_bc_defaults)
            if st.button("Reset to Phenotyping Markers", key="bc_reset_phenotyping_markers"):
                defaults = _pick_batch_marker_defaults(
                    st.session_state.bc_common_columns,
                    st.session_state.bc_config,
                    st.session_state.bc_metadata_cols,
                )
                st.session_state.bc_selected_markers = list(defaults)
                st.session_state[marker_key] = list(defaults)
                _persist_tab6_settings()
                st.rerun()
            selected_markers = st.multiselect(
                "Markers to correct",
                options=st.session_state.bc_common_columns,
                key=marker_key,
                help="Start typing to search, or click to select/deselect. Changes are included by Save Project.",
            )
            st.session_state.bc_selected_markers = list(selected_markers)
            st.caption(f"Selected: {len(selected_markers)} markers")

    # 4. ID Columns & Batch Parsing
    st.subheader("4. ID Columns & Batch Parsing")

    with st.form("bc_id_config_form"):
        c_id1, c_id2, c_id3 = st.columns([1, 1, 2])

        # Cell ID strategy & Local Column
        strategy_options = [
            "Auto-generate global (cell_0, cell_1, ...)",
            "Combine Batch ID + Local Column",
            "Use original column directly (must be unique)"
        ]
        current_strategy = st.session_state.bc_config.get('cell_id_strategy', "Auto-generate global (cell_0, cell_1, ...)")
        if current_strategy not in strategy_options:
            current_strategy = "Auto-generate global (cell_0, cell_1, ...)"

        cell_id_strategy = c_id1.selectbox(
            "Cell ID strategy",
            options=strategy_options,
            index=strategy_options.index(current_strategy),
            help="Controls how the unique identifier for each cell is established.",
        )

        local_col_default = st.session_state.bc_config.get('local_cell_id_col', 'label')
        # Always bound, so saving works whichever picker below is rendered.
        local_cell_id_col = local_col_default

        if st.session_state.bc_common_columns:
            default_local_idx = (
                st.session_state.bc_common_columns.index(local_col_default)
                if local_col_default in st.session_state.bc_common_columns
                else 0
            )
            local_cell_id_col = c_id1.selectbox(
                "Local Cell ID column",
                options=st.session_state.bc_common_columns,
                index=default_local_idx,
                help="The column in the CSV files containing the original cell ID (e.g. label).",
                disabled=(cell_id_strategy == "Auto-generate global (cell_0, cell_1, ...)"),
            )
        else:
            local_cell_id_col = c_id1.text_input(
                "Local Cell ID column",
                value=local_col_default,
                help="The column in the CSV files containing the original cell ID (e.g. label).",
                disabled=(cell_id_strategy == "Auto-generate global (cell_0, cell_1, ...)"),
            )

        cell_id_col = c_id1.text_input(
            "Output Cell ID column name",
            value=st.session_state.bc_config.get('cell_id_col', 'cell_id'),
            help="The name of the column in the output batch-corrected file that will contain the unique Cell IDs.",
        )
        
        arcsinh_cofactor = c_id1.number_input(
            "Arcsinh cofactor",
            min_value=0.1,
            value=float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0)),
            step=0.5,
            help=(
                "Cofactor for the ArcSinh pre-transformation. Smaller values compress the "
                "bright end harder.\n\n"
                "A cofactor of 5 comes from mass cytometry, where the dynamic range is wide "
                "and the low end is noisy. Imaging intensities are usually better behaved, "
                "and phenotype identity depends on *how bright* a positive population sits "
                "above the negatives — exactly what strong compression flattens. If "
                "downstream clustering separates cell types poorly, try a larger cofactor "
                "(50–150, which keeps the transform close to linear) and compare.\n\n"
                "Whatever you pick is recorded with the run, and the corrected columns stay "
                "in this ArcSinh space."
            )
        )
        # In unified mode the batch column was already chosen in the file setup
        # section above. Show it read-only here rather than offering a second
        # picker, so the two can never disagree about which column splits batches.
        if unified_mode and st.session_state.bc_common_columns:
            batch_id_col = c_id2.selectbox(
                "Batch ID column",
                options=st.session_state.bc_common_columns,
                index=(
                    st.session_state.bc_common_columns.index(st.session_state.bc_unified_batch_col)
                    if st.session_state.bc_unified_batch_col in st.session_state.bc_common_columns
                    else 0
                ),
                disabled=True,
                help=(
                    "Set in the file setup section above (unified mode stratifies the single "
                    "CSV by that column). Change it there if you need a different column."
                ),
            )
        elif st.session_state.bc_common_columns:
            current_batch_id = st.session_state.bc_config.get('batch_id_col', 'batch_id')
            is_auto = st.session_state.bc_config.get('batch_id_is_auto', True)
            options = ["None (Auto-generate from files)"] + st.session_state.bc_common_columns

            # Map "batch_id" with is_auto=True to "None (Auto-generate from files)"
            if is_auto or current_batch_id not in options:
                default_batch_idx = 0
            else:
                default_batch_idx = options.index(current_batch_id)

            batch_id_col = c_id2.selectbox(
                "Batch ID column",
                options=options,
                index=default_batch_idx,
                help=(
                    "The column in your CSVs containing the batch identifiers. "
                    "Select 'None (Auto-generate from files)' if your CSV files do not have a batch column inside them "
                    "(e.g., if you loaded multiple files, and each file represents a single batch)."
                ),
            )
        else:
            is_auto = st.session_state.bc_config.get('batch_id_is_auto', True)
            batch_id_col = c_id2.text_input(
                "Batch ID column",
                value="None (Auto-generate from files)" if is_auto else st.session_state.bc_config.get('batch_id_col', 'batch_id'),
                help="Batch label assigned to every cell (assigned before filtering). Enter 'None' or 'None (Auto-generate from files)' to auto-detect.",
            )
        batch_strategy = c_id3.selectbox(
            "Batch ID source",
            options=["Filename (strip suffixes)", "Folder name"],
            index=0 if st.session_state.bc_config.get('batch_id_strategy') != 'Folder name' else 1,
            help=(
                "Controls how the batch ID is derived when 'None (Auto-generate from files)' is selected. "
                "You can extract the ID from the filename or the parent folder name."
            ),
            disabled=unified_mode,
        )
        suffixes = c_id3.text_input(
            "Filename suffixes to strip (comma-separated)",
            value=st.session_state.bc_config.get('filename_suffixes_to_strip', ''),
            help="Example: .ome_cell_features_annotated,_fully_annotated,_cell_features",
            disabled=unified_mode,
        )

        if st.form_submit_button("Save ID Settings"):
            st.session_state.bc_config['cell_id_strategy'] = cell_id_strategy
            st.session_state.bc_config['local_cell_id_col'] = str(local_cell_id_col or '').strip()
            st.session_state.bc_config['cell_id_col'] = str(cell_id_col or '').strip() or 'cell_id'
            if batch_id_col in ["None (Auto-generate from files)", "Auto-generate from files", "None", "none", ""]:
                st.session_state.bc_config['batch_id_col'] = 'batch_id'
                st.session_state.bc_config['batch_id_is_auto'] = True
            else:
                st.session_state.bc_config['batch_id_col'] = str(batch_id_col or '').strip()
                st.session_state.bc_config['batch_id_is_auto'] = False
            st.session_state.bc_config['batch_id_strategy'] = batch_strategy
            st.session_state.bc_config['filename_suffixes_to_strip'] = suffixes
            st.session_state.bc_config['arcsinh_cofactor'] = float(arcsinh_cofactor)
            st.success("ID & pre-transformation settings saved.")
            _persist_tab6_settings()
            st.rerun()

        # Check for conflicts between calculated Batch ID and Metadata columns
        current_batch_col = st.session_state.bc_config.get('batch_id_col', 'batch_id')
        if current_batch_col in st.session_state.bc_metadata_cols:
            st.warning(
                f"⚠️ Conflict detected: The Batch ID column '{current_batch_col}' is also defined in your "
                "Metadata columns (File List). The calculator above might overwrite your manual metadata values, "
                "or vice-versa. Please rename the Batch ID column or remove it from the metadata list if you intend "
                "to use the auto-calculation."
            )

    def _find_longest_common_suffix(strings: list[str]) -> str:
        """Find the longest common suffix among a list of strings."""
        if not strings:
            return ""
        # Reverse strings, find common prefix, then reverse back
        reversed_strings = [s[::-1] for s in strings]
        common_prefix = os.path.commonprefix(reversed_strings)
        return common_prefix[::-1]

    if not unified_mode and st.button("Auto-Detect Suffix from Files"):
        files = st.session_state.bc_files_df["Filename"].dropna().astype(str).unique().tolist()
        # Filter out empty strings
        files = [f for f in files if f and f.strip()]
        
        if not files:
            st.warning("No files loaded to detect suffix from.")
        else:
            common_suffix = _find_longest_common_suffix(files)
            if common_suffix:
                # User preference: exclude leading underscore if it leaves "roi_" or similar?
                # For now, literal common suffix is usually what they want.
                # Example: "P1_roi_1_fully_annotated.csv" -> "_fully_annotated.csv"
                st.session_state.bc_config['filename_suffixes_to_strip'] = common_suffix
                st.success(f"Detected common suffix: '{common_suffix}'")
                _persist_tab6_settings()
                st.rerun()
            else:
                st.info("No common suffix found.")

    def _get_filtration_summary(df, config, selected_markers, files_with_filter_col):
        import pandas as pd
        import numpy as np
        import numbers
        
        summary = {}
        total_loaded = len(df)
        summary['total_loaded'] = total_loaded
        
        # 1. Generic Filter Column
        t_col = config.get('tissue_col', 'None')
        t_val = config.get('tissue_val', 1)
        
        df_tissue = df
        if t_col and t_col != 'None' and t_col in df.columns:
            # Keep all rows from files that did not contain the filter column
            cond_no_col = ~df['original_filename'].isin(files_with_filter_col)
            try:
                cond_match = (df[t_col] == t_val)
                if isinstance(t_val, numbers.Number):
                    cond_match |= (pd.to_numeric(df[t_col], errors='coerce') == float(t_val))
            except Exception:
                cond_match = (df[t_col].astype(str) == str(t_val))
            df_tissue = df[cond_no_col | cond_match]
            
        summary['tissue_kept'] = len(df_tissue)
        summary['tissue_lost'] = total_loaded - len(df_tissue)
        
        # 2. Area Filter
        a_col = config.get('area_col', 'area')
        area_min = config.get('area_min_pct', 0.1)
        area_max = config.get('area_max_pct', 99.9)
        
        df_area = df_tissue
        if a_col in df_tissue.columns and len(df_tissue) > 0:
            low = np.percentile(df_tissue[a_col], area_min)
            high = np.percentile(df_tissue[a_col], area_max)
            df_area = df_tissue[(df_tissue[a_col] >= low) & (df_tissue[a_col] <= high)]
            summary['area_low_threshold'] = low
            summary['area_high_threshold'] = high
        else:
            summary['area_low_threshold'] = np.nan
            summary['area_high_threshold'] = np.nan
            
        summary['area_kept'] = len(df_area)
        summary['area_lost'] = len(df_tissue) - len(df_area)
        
        # 3. Outlier Filter
        out_high_pct = config.get('outlier_high', 99.9)
        out_low_pct = config.get('outlier_low', 0.1)
        
        df_outlier = df_area
        valid_markers = [m for m in selected_markers if m in df_area.columns] if selected_markers else []
        
        if valid_markers and len(df_area) > 0:
            qc_cols = []
            for m in valid_markers:
                col_data = df_area[m]
                if not pd.api.types.is_numeric_dtype(col_data):
                    col_data = pd.to_numeric(col_data, errors='coerce')
                qc_cols.append(col_data)
            qc_data = pd.concat(qc_cols, axis=1).fillna(0) if qc_cols else pd.DataFrame(index=df_area.index)
            high_thresholds = qc_data.quantile(out_high_pct / 100.0)
            low_thresholds = qc_data.quantile(out_low_pct / 100.0)
            
            high_outliers = (qc_data > high_thresholds).sum(axis=1)
            low_outliers = (qc_data < low_thresholds).sum(axis=1)
            
            n_markers = len(valid_markers)
            high_mask = (high_outliers / n_markers) > 0.5
            low_mask = (low_outliers / n_markers) > 0.5
            outlier_mask = high_mask | low_mask
            
            df_outlier = df_area[~outlier_mask]
            
        summary['outlier_kept'] = len(df_outlier)
        summary['outlier_lost'] = len(df_area) - len(df_outlier)
        summary['final_kept'] = len(df_outlier)
        summary['total_lost'] = total_loaded - len(df_outlier)
        
        return summary

    def render_filter_statistics():
        """Helper to calculate and render step-by-step filtration statistics."""
        loaded_df = st.session_state.get('bc_loaded_data')
        if loaded_df is None:
            st.info("💡 **Tip**: Click **'Load and Apply Filters'** below to import the dataset and view the step-by-step cell loss report.")
            return

        summary = st.session_state.get('bc_filter_summary')
        if summary is None:
            files_with_filter_col = st.session_state.get('bc_files_with_filter_col', set())
            markers = list(st.session_state.get('bc_selected_markers', []) or [])
            if not markers:
                markers = list(st.session_state.get('bc_common_markers', []) or [])
            with st.spinner("Calculating filtration impact summary..."):
                summary = _get_filtration_summary(loaded_df, st.session_state.bc_config, markers, files_with_filter_col)
                st.session_state.bc_filter_summary = summary
            
        st.markdown("### 📊 Active Filter Impact Analysis")
        st.markdown("This report shows how many cells will be retained for batch correction after applying each filter:")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("1. Total Loaded Cells", f"{summary['total_loaded']:,}")
        
        t_col = st.session_state.bc_config.get('tissue_col', 'None')
        t_val = st.session_state.bc_config.get('tissue_val', 1)
        if t_col and t_col != 'None':
            lbl_tissue = f"2. Filter Column ({t_col}=={t_val})"
            col_stat2.metric(lbl_tissue, f"{summary['tissue_kept']:,}", f"-{summary['tissue_lost']:,} lost", delta_color="inverse")
        else:
            col_stat2.metric("2. Filter Column", "Disabled", "No generic filter")
            
        a_col = st.session_state.bc_config.get('area_col', 'area')
        lbl_area = f"3. Area Filter ({a_col})"
        col_stat3.metric(lbl_area, f"{summary['area_kept']:,}", f"-{summary['area_lost']:,} lost", delta_color="inverse")
        
        col_stat4.metric("4. Markers Outlier Filter", f"{summary['outlier_kept']:,}", f"-{summary['outlier_lost']:,} lost", delta_color="inverse")
        
        st.markdown("#### Detailed Breakdown:")
        details = []
        details.append(f"- **Initial Dataset**: `{summary['total_loaded']:,}` total cells loaded from selected files.")
        
        if t_col and t_col != 'None':
            details.append(
                f"- **Metadata Filter (`{t_col} == {t_val}`)**: Kept `{summary['tissue_kept']:,}` cells. "
                f"Filtered out `{summary['tissue_lost']:,}` cells (retained all rows from files lacking the column `{t_col}`)."
            )
        else:
            details.append("- **Metadata Filter**: Disabled.")
            
        if not np.isnan(summary['area_low_threshold']):
            details.append(
                f"- **Area Filter (`{a_col}`)**: Kept `{summary['area_kept']:,}` cells. "
                f"Filtered out `{summary['area_lost']:,}` cells using percentile thresholds: "
                f"Min Area = `{summary['area_low_threshold']:.2f}` (Percentile: `{st.session_state.bc_config.get('area_min_pct', 0.1)}%`), "
                f"Max Area = `{summary['area_high_threshold']:.2f}` (Percentile: `{st.session_state.bc_config.get('area_max_pct', 99.9)}%`)."
            )
        else:
            details.append(f"- **Area Filter**: Skipped (column `{a_col}` not found).")
            
        details.append(
            f"- **Markers Outlier Filter**: Kept `{summary['outlier_kept']:,}` cells. "
            f"Filtered out `{summary['outlier_lost']:,}` cells. Cells were classified as outliers if more than 50% "
            f"of their marker values were below the `{st.session_state.bc_config.get('outlier_low', 0.1)}%` low percentile threshold "
            f"or above the `{st.session_state.bc_config.get('outlier_high', 99.9)}%` high percentile threshold."
        )
        
        retention_rate = (summary['final_kept'] / summary['total_loaded'] * 100) if summary['total_loaded'] > 0 else 0
        details.append(
            f"- **Final Retained Cells**: **`{summary['final_kept']:,}`** cells (**`{retention_rate:.2f}%`** retention rate) "
            f"will enter the batch correction pipeline."
        )
        
        st.markdown("\n".join(details))

    st.subheader("5. Filtration Settings")
    
    with st.container():
        c_filt1, c_filt2, c_filt3 = st.columns(3)
        
        # Filter Column (Generic Filter)
        c_filt1.markdown("**Filter Column**")
        union_cols = st.session_state.get('bc_union_columns', [])
        if not union_cols:
            union_cols = st.session_state.get('bc_common_columns', [])
            
        options = ["None"] + sorted(list(set(union_cols)))
        current_tissue = st.session_state.bc_config.get('tissue_col', 'None')
        if current_tissue not in options:
            options.append(current_tissue)
            
        default_tissue_idx = options.index(current_tissue) if current_tissue in options else 0
        tissue_col = c_filt1.selectbox(
            "Filter Column",
            options=options,
            index=default_tissue_idx,
            help="Select a column to filter cells. If a file does not contain this column, all cells from that file will be kept. Useful for subsetting analysis to a specific ROI (e.g. tissue=1)."
        )
        
        tissue_val_str = c_filt1.text_input(
            "Keep Value", 
            value=str(st.session_state.bc_config.get('tissue_val', 1)),
            help="Value in the Filter Column indicating cells to keep (e.g. 1, 'keep', 'yes'). Supports string or numeric matching."
        )
        try:
            if "." in tissue_val_str:
                tissue_val = float(tissue_val_str)
            else:
                tissue_val = int(tissue_val_str)
        except ValueError:
            tissue_val = tissue_val_str

        c_filt1.markdown("**Batch Correction Indicator**")
        add_indicator = c_filt1.checkbox(
            "Add indicator column for corrected cells",
            value=st.session_state.bc_config.get('add_corrected_indicator', False),
            help="Adds a column to the dataset with a value of 1 if the cell was successfully batch corrected, and 0 if it was not (e.g. excluded due to filter or NaN markers)."
        )
        indicator_col = c_filt1.text_input(
            "Indicator Column Name",
            value=st.session_state.bc_config.get('corrected_indicator_col', 'corrected'),
            help="Specify the name of the column that will hold the batch correction indicator value (e.g. 'corrected')."
        )
        
        # Area Filter
        c_filt2.markdown("**Area Filter**")
        if st.session_state.bc_common_columns:
            current_area = st.session_state.bc_config.get('area_col', 'area')
            default_area_idx = st.session_state.bc_common_columns.index(current_area) if current_area in st.session_state.bc_common_columns else 0
            area_col = c_filt2.selectbox(
                "Area Column",
                options=st.session_state.bc_common_columns,
                index=default_area_idx,
                help="Select the column representing cell/nuclei area (e.g. 'area'). Used for size-based filtration."
            )
        else:
            area_col = c_filt2.text_input(
                "Area Column",
                st.session_state.bc_config.get('area_col', 'area'),
                help="Select the column representing cell/nuclei area (e.g. 'area'). Used for size-based filtration."
            )
        area_min = c_filt2.number_input(
            "Min Percentile", 
            0.0, 100.0, 
            st.session_state.bc_config.get('area_min_pct', 0.1),
            help="Exclude cells with area below this percentile (e.g. 0.1% to remove segmentation artifacts or noise)."
        )
        area_max = c_filt2.number_input(
            "Max Percentile", 
            0.0, 100.0, 
            st.session_state.bc_config.get('area_max_pct', 99.9),
            help="Exclude cells with area above this percentile (e.g. 99.9% to remove cell doublets/clumps)."
        )
        
        # Outlier Filter
        c_filt3.markdown("**Markers Outlier Filter**")
        c_filt3.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True) # Aligns selectboxes or headers visually
        
        out_low = c_filt3.number_input(
            "Low Outlier %", 
            0.0, 100.0, 
            st.session_state.bc_config.get('outlier_low', 0.1),
            help="Percentile used to detect low outlier expression. Cells with more than 50% of markers below this threshold are filtered out."
        )
        out_high = c_filt3.number_input(
            "High Outlier %", 
            0.0, 100.0, 
            st.session_state.bc_config.get('outlier_high', 99.9),
            help="Percentile used to detect high outlier expression (staining artifacts). Cells with more than 50% of markers above this threshold are filtered out."
        )

    # Immediately sync current UI settings to config in session state
    current_filt_settings = {
        'tissue_col': tissue_col,
        'tissue_val': tissue_val,
        'add_corrected_indicator': add_indicator,
        'corrected_indicator_col': indicator_col,
        'area_col': area_col,
        'area_min_pct': area_min,
        'area_max_pct': area_max,
        'outlier_high': out_high,
        'outlier_low': out_low,
    }
    
    last_settings = st.session_state.get('bc_last_calculated_config')
    if last_settings != current_filt_settings:
        if 'bc_filter_summary' in st.session_state:
            del st.session_state.bc_filter_summary
        st.session_state.bc_last_calculated_config = current_filt_settings
        st.session_state.bc_config.update(current_filt_settings)

    # Render counts report
    render_filter_statistics()

    # 5. Load & Filter
    st.subheader("6. Load & Filter Data")
    
    c_run, c_save = st.columns(2)
    with c_run:
        run_btn = st.button("Load and Apply Filters", type="primary", width='stretch')
    with c_save:
        save_btn = st.button("Save Project", key="tab6_setup_save_project_btn", width='stretch')
        if save_btn:
            _save_tab6_project_settings()
            st.success("Project settings saved to configuration file.")
            
    if run_btn:
        if unified_mode:
            preflight_errors = validate_selected_files(
                [st.session_state.bc_unified_csv_path],
                item_name="unified CSV",
            )
            unified_parent = Path(st.session_state.bc_unified_csv_path).expanduser().parent
            if unified_parent.exists() and not os.access(unified_parent, os.W_OK | os.X_OK):
                preflight_errors.append(f"Cannot replace the unified CSV in: {unified_parent}")
        else:
            preflight_errors = validate_run_directories(
                {"CSV input directory": st.session_state.bc_input_dir},
                st.session_state.bc_output_dir,
            )
        files_df = st.session_state.bc_files_df
        if "Process" not in files_df.columns or "Filename" not in files_df.columns:
            preflight_errors.append(
                "Scan the unified CSV before loading batches."
                if unified_mode
                else "Scan the input directory before loading and filtering files."
            )
            files_to_process = pd.DataFrame()
        else:
            # Logic to load data, apply filters, and store in session state
            files_df["Process"] = _coerce_bool_series(files_df["Process"])
            files_to_process = files_df[files_df["Process"] == True]
            if unified_mode:
                if files_to_process.empty:
                    preflight_errors.append("Select at least one batch value before loading data.")
                if not str(st.session_state.bc_unified_batch_col).strip():
                    preflight_errors.append("Choose the unified CSV column used to split batches.")
            else:
                preflight_errors.extend(
                    validate_selected_files(
                        [Path(st.session_state.bc_input_dir) / str(name) for name in files_to_process["Filename"]],
                        item_name="CSV file",
                    )
                )

        for config_key, label in (("cell_id_col", "Cell ID column name"), ("batch_id_col", "Batch ID column name")):
            if not str(st.session_state.bc_config.get(config_key, "")).strip():
                preflight_errors.append(f"Provide a {label} before loading files.")

        if preflight_errors:
            record_process_event(
                "tab6",
                "Batch correction preflight failed",
                status="warning",
                details={"errors": preflight_errors},
            )
            show_preflight_errors(preflight_errors)
            return
            
        progress = st.progress(0)
        status_text = st.empty()
        status_container = st.status("Loading and filtering selected CSV files...", expanded=True)
        started_at = time.perf_counter()
        record_process_event(
            "tab6",
            "Batch correction data load started",
            details={
                "files_selected": len(files_to_process),
                "input_source": (
                    st.session_state.bc_unified_csv_path
                    if unified_mode
                    else st.session_state.bc_input_dir
                ),
                "output_directory": st.session_state.bc_output_dir,
            },
        )

        def update_status(message: str) -> None:
            status_text.text(message)
            status_container.write(message)
        
        try:
            total_files = len(files_to_process)
            log_step(f"Starting load of {total_files} files.")
            log_step(f"Metadata cols to add: {st.session_state.bc_metadata_cols}")

            def _derive_batch_id(csv_filename: str, full_path: str) -> str:
                strategy = st.session_state.bc_config.get('batch_id_strategy', 'Filename (strip suffixes)')
                if strategy == 'Folder name':
                    return os.path.basename(os.path.dirname(full_path))

                # Use full filename to allow stripping extensions like .csv
                name = csv_filename
                suffixes_raw = st.session_state.bc_config.get('filename_suffixes_to_strip', '')
                suffixes = [s.strip() for s in suffixes_raw.split(',') if s.strip()]
                
                # Sort suffixes by length (descending) to match longest first
                suffixes.sort(key=len, reverse=True)
                
                for sfx in suffixes:
                    if name.endswith(sfx):
                        name = name[: -len(sfx)]
                        break
                
                # If name still ends in .csv and the suffix didn't cover it (e.g. user entered "_annotated"), strip it
                if name.lower().endswith('.csv'):
                    name = name[:-4]
                    
                return name

            loaded_frames: list[pd.DataFrame] = []
            files_with_filter_col = set()
            update_status("Loading CSVs...")

            cell_id_col = st.session_state.bc_config.get('cell_id_col', 'cell_id')
            batch_id_col = st.session_state.bc_config.get('batch_id_col', 'batch_id')

            ids_preassigned = False
            if unified_mode:
                batch_id_col = st.session_state.bc_unified_batch_col
                st.session_state.bc_config['batch_id_col'] = batch_id_col
                st.session_state.bc_config['batch_id_is_auto'] = False
                update_status(f"Loading unified CSV: {Path(st.session_state.bc_unified_csv_path).name}...")
                try:
                    unified_df = pd.read_csv(st.session_state.bc_unified_csv_path)
                    if batch_id_col not in unified_df.columns:
                        raise ValueError(f"Batch column '{batch_id_col}' is missing from the unified CSV.")
                    unified_df = _assign_batch_correction_cell_ids(
                        unified_df,
                        st.session_state.bc_config,
                        batch_id_col,
                    )
                    ids_preassigned = True
                    selected_batch_values = set(files_to_process["Filename"].astype(str))
                    batch_values = unified_df[batch_id_col].astype(str)
                    selected_mask = batch_values.isin(selected_batch_values)
                    if not selected_mask.any():
                        raise ValueError("None of the selected batch values exist in the unified CSV.")

                    metadata_by_batch = files_to_process.set_index("Filename")
                    for metadata_column in st.session_state.bc_metadata_cols:
                        if metadata_column in metadata_by_batch.columns:
                            mapped_values = batch_values.map(metadata_by_batch[metadata_column])
                            unified_df.loc[selected_mask, metadata_column] = mapped_values.loc[selected_mask]

                    df_raw = unified_df.loc[selected_mask].copy()
                    df_raw['original_filename'] = batch_values.loc[selected_mask].to_numpy()
                    t_col = st.session_state.bc_config.get('tissue_col', 'None')
                    if t_col and t_col != 'None' and t_col in df_raw.columns:
                        files_with_filter_col.update(selected_batch_values)
                    loaded_frames.append(df_raw)
                    progress.progress(1.0)
                except Exception as e_load:
                    log_step(f"ERROR loading unified CSV: {e_load}")
                    st.error(f"Failed to load unified CSV: {e_load}")
            else:
                for i, (_, row) in enumerate(files_to_process.iterrows()):
                    fname = str(row["Filename"])
                    update_status(f"Loading {fname}...")
                    fpath = os.path.join(st.session_state.bc_input_dir, fname)
                    try:
                        df_raw = pd.read_csv(fpath)
                        t_col = st.session_state.bc_config.get('tissue_col', 'None')
                        if t_col and t_col != 'None' and t_col in df_raw.columns:
                            files_with_filter_col.add(fname)
                    except Exception as e_load:
                        log_step(f"ERROR loading {fname}: {e_load}")
                        st.error(f"Failed to load {fname}: {e_load}")
                        continue

                    for col in st.session_state.bc_metadata_cols:
                        df_raw[col] = row.get(col, "")

                    if i < 3 or "P1_P3_roi_2" in fname:
                         meta_debug = {c: row.get(c, "N/A") for c in st.session_state.bc_metadata_cols}
                         log_step(f"Loaded {fname}: {len(df_raw)} rows. Metadata applied: {meta_debug}")

                    is_auto = st.session_state.bc_config.get('batch_id_is_auto', True)
                    if is_auto or batch_id_col not in df_raw.columns:
                        df_raw[batch_id_col] = _derive_batch_id(fname, fpath)
                    else:
                        df_raw[batch_id_col] = df_raw[batch_id_col].fillna("Unknown").astype(str)
                    df_raw['original_filename'] = fname

                    loaded_frames.append(df_raw)
                    progress.progress((i + 1) / total_files)

            st.session_state.bc_files_with_filter_col = files_with_filter_col

            if not loaded_frames:
                st.error("No data loaded.")
                status_container.update(label="No usable input files were loaded", state="error", expanded=True)
                return

            combined_all = pd.concat(loaded_frames, ignore_index=True)
            log_step(f"Total loaded rows: {len(combined_all)}")
            log_step(f"Unique batches loaded: {combined_all[batch_id_col].nunique()}")
            if combined_all[batch_id_col].nunique() != total_files:
                log_step(f"WARNING: Loaded {total_files} files but only found {combined_all[batch_id_col].nunique()} unique batches. Check batch naming strategy.")

            # Assign universal cell IDs before filtering. Unified mode assigns
            # them before batch selection so IDs remain stable across reruns.
            if not ids_preassigned:
                combined_all = _assign_batch_correction_cell_ids(
                    combined_all,
                    st.session_state.bc_config,
                    batch_id_col,
                )

            # Store pre-filter data
            st.session_state.bc_loaded_data = combined_all

            # Apply filters AFTER IDs exist
            filtered = combined_all

            # Generic Filter Column
            t_col = st.session_state.bc_config.get('tissue_col', 'None')
            t_val = st.session_state.bc_config.get('tissue_val', 1)
            if t_col and t_col != 'None' and t_col in filtered.columns:
                before_n = len(filtered)
                
                # Keep all rows from files that did not contain the filter column
                cond_no_col = ~filtered['original_filename'].isin(files_with_filter_col)
                
                # Filter rows from files that did contain the filter column
                try:
                    cond_match = (filtered[t_col] == t_val)
                    import numbers
                    if isinstance(t_val, numbers.Number):
                        cond_match |= (pd.to_numeric(filtered[t_col], errors='coerce') == float(t_val))
                except Exception:
                    cond_match = (filtered[t_col].astype(str) == str(t_val))
                
                filtered = filtered[cond_no_col | cond_match]
                log_step(f"Filter Column ({t_col}={t_val}): {before_n} -> {len(filtered)} (retained all rows from files lacking '{t_col}')")
            else:
                log_step(f"Filter Column is '{t_col}', skipping generic filtration.")

            # Area filter
            a_col = st.session_state.bc_config.get('area_col', 'area')
            if a_col in filtered.columns and len(filtered) > 0:
                low = np.percentile(filtered[a_col], st.session_state.bc_config.get('area_min_pct', 0.1))
                high = np.percentile(filtered[a_col], st.session_state.bc_config.get('area_max_pct', 99.9))
                before_n = len(filtered)
                filtered = filtered[(filtered[a_col] >= low) & (filtered[a_col] <= high)]
                log_step(f"Area filter ({a_col} [{low:.2f}, {high:.2f}]): {before_n} -> {len(filtered)}")

            # Outlier filter
            out_high_pct = st.session_state.bc_config.get('outlier_high', 99.9)
            out_low_pct = st.session_state.bc_config.get('outlier_low', 0.1)
            selected_markers = st.session_state.get('bc_selected_markers', [])
            if not selected_markers:
                selected_markers = st.session_state.get('bc_common_markers', [])
            
            if selected_markers and len(filtered) > 0:
                before_n = len(filtered)
                valid_markers = [m for m in selected_markers if m in filtered.columns]
                if valid_markers:
                    qc_cols = []
                    for m in valid_markers:
                        col_data = filtered[m]
                        if not pd.api.types.is_numeric_dtype(col_data):
                            col_data = pd.to_numeric(col_data, errors='coerce')
                        qc_cols.append(col_data)
                    qc_data = pd.concat(qc_cols, axis=1).fillna(0) if qc_cols else pd.DataFrame(index=filtered.index)
                    high_thresholds = qc_data.quantile(out_high_pct / 100.0)
                    low_thresholds = qc_data.quantile(out_low_pct / 100.0)
                    
                    high_outliers = (qc_data > high_thresholds).sum(axis=1)
                    low_outliers = (qc_data < low_thresholds).sum(axis=1)
                    
                    n_markers = len(valid_markers)
                    high_mask = (high_outliers / n_markers) > 0.5
                    low_mask = (low_outliers / n_markers) > 0.5
                    outlier_mask = high_mask | low_mask
                    
                    filtered = filtered[~outlier_mask]
                    log_step(f"Outlier filter (>{out_low_pct}% low or >{out_high_pct}% high): {before_n} -> {len(filtered)}")

            st.session_state.bc_cleaned_data = filtered.reset_index(drop=True)
            st.session_state.bc_setup_complete = True
            
            final_batches = st.session_state.bc_cleaned_data[batch_id_col].nunique()
            log_step(f"Final cleaned data: {len(st.session_state.bc_cleaned_data)} cells. batches: {final_batches}")

            # Compute and cache filtration summary in session state
            st.session_state.bc_filter_summary = _get_filtration_summary(
                combined_all,
                st.session_state.bc_config,
                selected_markers,
                files_with_filter_col
            )

            summary = st.session_state.get('bc_filter_summary')
            if summary:
                st.success(
                    f"🎉 **Data Load & Filtration Complete!**\n\n"
                    f"Here is the step-by-step breakdown of cells retained:\n"
                    f"- **1. Total Loaded Cells**: `{summary['total_loaded']:,}` cells from `{total_files}` files.\n"
                    f"- **2. Filter Column (`{t_col}={t_val}`)**: `{summary['tissue_kept']:,}` cells retained (lost `{summary['tissue_lost']:,}`).\n"
                    f"- **3. Area Filter (`{a_col}`)**: `{summary['area_kept']:,}` cells retained (lost `{summary['area_lost']:,}`).\n"
                    f"- **4. Markers Outlier Filter**: `{summary['outlier_kept']:,}` cells retained (lost `{summary['outlier_lost']:,}`).\n\n"
                    f"**Final Retained for Correction**: **`{summary['final_kept']:,}`** cells (retained **`{final_batches}`** batches)."
                )
            else:
                st.success(
                    f"Loaded {len(combined_all)} cells from {total_files} files. "
                    f"After filtering: {len(st.session_state.bc_cleaned_data)} cells. "
                    f"Batches retained: {final_batches}/{total_files}"
                )
            st.info("Next: go to '2. QC & Correction' tab to continue.")
            record_process_event(
                "tab6",
                "Batch correction data load complete",
                status="complete",
                duration_s=time.perf_counter() - started_at,
                details={
                    "cells_retained": len(st.session_state.bc_cleaned_data),
                    "batches_retained": final_batches,
                    "files_selected": total_files,
                },
            )
            status_container.update(
                label=f"Load and filtering complete in {time.perf_counter() - started_at:.1f}s",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            st.error(f"Error loading data: {e}")
            log_step(f"EXCEPTION in Load and Apply Filters: {e}")
            record_process_event(
                "tab6",
                "Batch correction data load failed",
                message=str(e),
                status="error",
                duration_s=time.perf_counter() - started_at,
            )
            status_container.update(label="Load and filtering failed", state="error", expanded=True)

def render_qc_correction(project_root: Path):
    """Subtab 2: QC & Correction."""
    st.header("2. QC & Correction")
    unified_mode = st.session_state.get('bc_input_mode') == "Unified CSV file"
    
    if st.session_state.bc_cleaned_data is None:
        st.warning("Please load and filter data in the 'Setup & Filter' tab first.")
        return
        
    df = st.session_state.bc_cleaned_data
    
    # QC Visualization
    st.subheader("1. Pre-Correction QC")

    def _parse_csv_list(text: str) -> list[str]:
        return [t.strip() for t in (text or "").split(',') if t.strip()]

    def _shorten_labels(values: pd.Series, suffixes_to_strip: list[str]) -> pd.Series:
        if not suffixes_to_strip:
            return values.astype(str)
        shortened = values.astype(str)
        for sfx in suffixes_to_strip:
            if not sfx:
                continue
            shortened = shortened.str.replace(sfx + "$", "", regex=True)
        return shortened

    with st.expander("QC Plots", expanded=True):
        batch_col = st.session_state.bc_config['batch_id_col']

        with st.expander("Plot settings", expanded=False):
            c_s1, c_s2, c_s3 = st.columns([2, 2, 2])
            label_suffixes = c_s1.text_input(
                "Strip suffix from batch labels (comma-separated)",
                value=st.session_state.bc_config.get('qc_label_strip_suffixes', ''),
                help="Only affects displayed labels in plots (does not change the stored batch_id values).",
                key="bc_qc_label_strip_suffixes",
            )
            x_rotation = c_s2.slider(
                "X label rotation (degrees)",
                min_value=0,
                max_value=90,
                value=int(st.session_state.bc_config.get('qc_x_rotation', 45)),
                step=5,
                key="bc_qc_x_rotation",
            )
            show_legend = c_s3.checkbox(
                "Show legend (right)",
                value=bool(st.session_state.bc_config.get('qc_show_legend', True)),
                key="bc_qc_show_legend",
            )

            c_fs1, c_fs2, c_fs3 = st.columns([2, 2, 2])
            font_size = c_fs1.number_input(
                "Font size",
                min_value=6,
                max_value=24,
                value=int(st.session_state.bc_config.get('qc_font_size', 10)),
                step=1,
                key="bc_qc_font_size",
            )
            title_size = c_fs2.number_input(
                "Title size",
                min_value=6,
                max_value=28,
                value=int(st.session_state.bc_config.get('qc_title_size', 12)),
                step=1,
                key="bc_qc_title_size",
            )
            legend_size = c_fs3.number_input(
                "Legend size",
                min_value=6,
                max_value=24,
                value=int(st.session_state.bc_config.get('qc_legend_size', 9)),
                step=1,
                key="bc_qc_legend_size",
            )

            c_s4, c_s5, c_s6 = st.columns([2, 2, 2])
            palette_name = c_s4.selectbox(
                "Color palette",
                options=["tab20", "tab10", "Set3", "Dark2", "Paired", "husl", "viridis", "custom"],
                index=0,
                key="bc_qc_palette",
            )
            custom_colors = c_s5.text_input(
                "Custom colors (comma-separated hex)",
                value=st.session_state.bc_config.get('qc_custom_colors', ''),
                placeholder="#1f77b4,#ff7f0e,#2ca02c",
                key="bc_qc_custom_colors",
                disabled=(palette_name != "custom"),
            )
            count_sort = c_s6.selectbox(
                "Count sort",
                options=["Descending", "Ascending", "Alphabetical"],
                index=0,
                key="bc_qc_count_sort",
            )

            if st.button("Save plot settings", key="bc_qc_save_plot_settings_btn"):
                st.session_state.bc_config['qc_label_strip_suffixes'] = label_suffixes
                st.session_state.bc_config['qc_x_rotation'] = int(x_rotation)
                st.session_state.bc_config['qc_show_legend'] = bool(show_legend)
                st.session_state.bc_config['qc_custom_colors'] = custom_colors
                st.session_state.bc_config['qc_font_size'] = int(font_size)
                st.session_state.bc_config['qc_title_size'] = int(title_size)
                st.session_state.bc_config['qc_legend_size'] = int(legend_size)
                st.success("Plot settings saved.")

        suffixes_to_strip = _parse_csv_list(st.session_state.bc_config.get('qc_label_strip_suffixes', ''))
        display_batch = _shorten_labels(df[batch_col], suffixes_to_strip)

        batch_order = None
        counts = display_batch.value_counts()
        if st.session_state.get('bc_qc_count_sort') == 'Ascending':
            counts = counts.sort_values(ascending=True)
        elif st.session_state.get('bc_qc_count_sort') == 'Alphabetical':
            counts = counts.sort_index()
        else:
            counts = counts.sort_values(ascending=False)
        batch_order = counts.index.tolist()

        def _build_palette(n: int):
            if st.session_state.get('bc_qc_palette') == 'custom':
                colors = _parse_csv_list(st.session_state.get('bc_qc_custom_colors', ''))
                return colors if len(colors) >= n else None
            try:
                return sns.color_palette(st.session_state.get('bc_qc_palette', 'tab20'), n_colors=n)
            except Exception:
                return sns.color_palette('tab20', n_colors=n)

        palette = _build_palette(len(batch_order))
        color_map = None
        if palette is not None:
            color_map = {b: palette[i] for i, b in enumerate(batch_order)}
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cell Counts per Batch**")
            fig, ax = plt.subplots(figsize=(7, 4))
            fs = int(st.session_state.bc_config.get('qc_font_size', 10))
            ts = int(st.session_state.bc_config.get('qc_title_size', 12))
            ls = int(st.session_state.bc_config.get('qc_legend_size', 9))
            bar_colors = [color_map[b] for b in batch_order] if color_map else None
            ax.bar(batch_order, [counts[b] for b in batch_order], color=bar_colors)
            ax.set_ylabel("Cells", fontsize=fs)
            ax.set_xlabel("Batch", fontsize=fs)
            ax.set_title("Cell counts per batch", fontsize=ts)
            ax.tick_params(axis='both', labelsize=fs)
            ax.tick_params(axis='x', labelrotation=int(st.session_state.get('bc_qc_x_rotation', 45)))
            if show_legend and color_map:
                handles = [Line2D([0], [0], marker='s', linestyle='', color=color_map[b], label=b) for b in batch_order]
                ax.legend(handles=handles, title="Batches", fontsize=ls, title_fontsize=ls, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            st.pyplot(fig)
            
        with c2:
            st.markdown("**Marker Distributions (Sample)**")
            # Identify markers
            exclude = st.session_state.bc_config['exclude_markers']
            ignore = ['x', 'y', 'id', 'label', 'area', batch_col, st.session_state.bc_config['cell_id_col'], 'original_filename']
            ignore.extend(st.session_state.bc_metadata_cols)

            selected_markers = st.session_state.get('bc_selected_markers', [])
            if selected_markers:
                # Prefer user-selected shared markers
                markers = [m for m in selected_markers if m in df.columns]
            else:
                # Fallback: infer numeric markers
                markers = [
                    c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                    and not any(x.lower() in c.lower() for x in ignore)
                    and not any(ex.lower() in c.lower() for ex in exclude)
                ]
            
            if not markers:
                st.warning("No numeric marker columns detected for plotting.")
            else:
                c_m1, c_m2 = st.columns([2, 2])
                sel_marker = c_m1.selectbox("Marker", markers, key="bc_qc_marker")
                plot_kind = c_m2.selectbox(
                    "Plot type",
                    options=["Boxplot", "Violin", "Histogram (overlap)", "Histogram (facet by batch)"],
                    index=0,
                    key="bc_qc_plot_kind",
                )

                c_t1, c_t2, c_t3 = st.columns([2, 2, 2])
                transform = c_t1.selectbox(
                    "Transform",
                    options=["Linear", "Log10", "Arcsinh", "Z-score"],
                    index=0,
                    key="bc_qc_transform",
                )
                arcsinh_cofactor = c_t2.number_input(
                    "Arcsinh cofactor",
                    min_value=0.1,
                    value=float(st.session_state.bc_config.get('arcsinh_cofactor', st.session_state.bc_config.get('qc_arcsinh_cofactor', 5.0))),
                    step=0.5,
                    key="bc_qc_arcsinh",
                )
                log_pseudocount = c_t3.number_input(
                    "Log10 pseudocount",
                    min_value=0.0,
                    value=float(st.session_state.bc_config.get('qc_log_pseudocount', 1.0)),
                    step=0.5,
                    key="bc_qc_log_pseudocount",
                )

                hist_bins = 50
                hist_kde = True
                hist_common_norm = False
                show_percentiles = True
                percentile_list = "1,5,50,95,99"
                if "Histogram" in plot_kind:
                    c_h1, c_h2, c_h3 = st.columns([2, 2, 2])
                    hist_bins = int(
                        c_h1.number_input("Bins", min_value=5, max_value=400, value=50, step=5, key="bc_qc_bins")
                    )
                    hist_kde = c_h2.checkbox("Smooth (KDE)", value=True, key="bc_qc_kde")
                    hist_common_norm = c_h3.checkbox(
                        "Normalize to density", value=False, key="bc_qc_common_norm",
                        help="If enabled, plots densities instead of counts.",
                    )
                    c_p1, c_p2 = st.columns([2, 2])
                    show_percentiles = c_p1.checkbox("Show percentile lines", value=True, key="bc_qc_show_percentiles")
                    percentile_list = c_p2.text_input(
                        "Percentiles (comma-separated)", value=percentile_list, key="bc_qc_percentile_list"
                    )

                if sel_marker:
                    plot_df = df.copy()
                    plot_df["__batch_display__"] = display_batch
                    plot_df["__value__"] = _transform_vector(
                        plot_df[sel_marker],
                        transform=str(transform),
                        arcsinh_cofactor=float(arcsinh_cofactor),
                        log_pseudocount=float(log_pseudocount),
                    )

                    y_label = sel_marker
                    if transform != 'Linear':
                        y_label = f"{sel_marker} ({transform})"

                    fig, ax = plt.subplots(figsize=(7, 4))
                    if plot_kind == "Boxplot":
                        sns.boxplot(
                            data=plot_df,
                            x="__batch_display__",
                            y="__value__",
                            order=batch_order,
                            palette=color_map,
                            ax=ax,
                        )
                    elif plot_kind == "Violin":
                        sns.violinplot(
                            data=plot_df,
                            x="__batch_display__",
                            y="__value__",
                            order=batch_order,
                            palette=color_map,
                            cut=0,
                            inner="quartile",
                            ax=ax,
                        )
                    elif plot_kind == "Histogram (overlap)":
                        sns.histplot(
                            data=plot_df,
                            x="__value__",
                            hue="__batch_display__",
                            hue_order=batch_order,
                            bins=hist_bins,
                            element="step",
                            stat="density" if hist_common_norm else "count",
                            common_norm=False,
                            kde=hist_kde,
                            palette=color_map,
                            ax=ax,
                            legend=show_legend,
                        )
                    else:
                        sns.histplot(
                            data=plot_df,
                            x="__value__",
                            col="__batch_display__",
                            col_wrap=2,
                            bins=hist_bins,
                            element="step",
                            stat="density" if hist_common_norm else "count",
                            common_norm=False,
                            kde=hist_kde,
                            color="#4c78a8",
                        )
                        plt.close(fig)
                        st.pyplot(plt.gcf())
                        ax = None

                    if ax is not None:
                        fs = int(st.session_state.bc_config.get('qc_font_size', 10))
                        ts = int(st.session_state.bc_config.get('qc_title_size', 12))
                        ls = int(st.session_state.bc_config.get('qc_legend_size', 9))
                        ax.set_xlabel("Batch" if "Histogram" not in plot_kind else y_label, fontsize=fs)
                        ax.set_ylabel(y_label if "Histogram" not in plot_kind else ("Density" if hist_common_norm else "Count"), fontsize=fs)
                        if "Histogram" not in plot_kind:
                            ax.tick_params(axis='x', labelrotation=int(st.session_state.get('bc_qc_x_rotation', 45)))
                        ax.tick_params(axis='both', labelsize=fs)
                        ax.set_title(f"{sel_marker} distribution", fontsize=ts)

                        if show_percentiles and "Histogram" in plot_kind:
                            try:
                                ps = [float(p.strip()) for p in (percentile_list or "").split(',') if p.strip()]
                                ps = [p for p in ps if 0 <= p <= 100]
                                if ps:
                                    vals = plot_df["__value__"].dropna().values
                                    for p in ps:
                                        v = np.percentile(vals, p)
                                        ax.axvline(v, linestyle='--', linewidth=1, color='black', alpha=0.6)
                            except Exception:
                                pass

                        if show_legend and "Histogram" in plot_kind and ax.get_legend() is not None:
                            ax.legend(title="Batches", fontsize=ls, title_fontsize=ls, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

                        plt.tight_layout()
                        st.pyplot(fig)

    # Correction Config
    st.subheader("2. Run Correction")
    
    with st.form("bc_run_form"):
        c_conf1, c_conf2 = st.columns(2)
        
        norm_method = c_conf1.selectbox(
            "Normalization Method (for SOM clustering)",
            ["scale (Z-score)", "rank"],
            index=0 if st.session_state.bc_config.get('norm_method', 'scale') == 'scale' else 1,
            help="'scale' (Z-score) is recommended for datasets from the same experiment. 'rank' is recommended for datasets with heavy batch effects or from different technologies."
        )

        corrected_suffix = c_conf1.text_input(
            "Corrected marker suffix",
            value=str(st.session_state.bc_config.get('corrected_suffix', '_corr')),
            help="Corrected marker columns will be saved as: <marker><suffix> (e.g., CD45_corr).",
        )
        
        # Covariates selection: list all metadata and non-marker columns
        exclude_from_covar = {
            st.session_state.bc_config.get('cell_id_col', 'cell_id'),
            st.session_state.bc_config.get('batch_id_col', 'batch_id'),
            'original_filename', 'batch', 'som_cluster', 'file_prefix', 'quality', 'corrected',
            'Process', 'Filename', 'x', 'y', 'X', 'Y', 'UMAP1', 'UMAP2'
        }
        selected_markers = set(st.session_state.get('bc_selected_markers', []))

        # Build candidate list from metadata columns, file list table, common/union CSV columns, and loaded data
        raw_candidates = []
        if st.session_state.get('bc_metadata_cols'):
            for c in st.session_state.bc_metadata_cols:
                if c not in raw_candidates:
                    raw_candidates.append(c)
        if st.session_state.get('bc_files_df') is not None and not st.session_state.bc_files_df.empty:
            for c in st.session_state.bc_files_df.columns:
                if c not in raw_candidates:
                    raw_candidates.append(c)
        if st.session_state.get('bc_common_columns'):
            for c in st.session_state.bc_common_columns:
                if c not in raw_candidates:
                    raw_candidates.append(c)
        if st.session_state.get('bc_union_columns'):
            for c in st.session_state.bc_union_columns:
                if c not in raw_candidates:
                    raw_candidates.append(c)
        if st.session_state.bc_cleaned_data is not None:
            for c in st.session_state.bc_cleaned_data.columns:
                if c not in raw_candidates:
                    raw_candidates.append(c)

        covar_options = [
            c for c in raw_candidates
            if c not in exclude_from_covar and c not in selected_markers
        ]
        
        # Ensure default covariates exist in options list to prevent errors
        defaults = [c for c in st.session_state.bc_config.get('covariates', []) if c in covar_options]
        
        selected_covariates = c_conf1.multiselect(
            "Biological Covariates to protect (optional, e.g., tissue_roi)",
            options=covar_options,
            default=defaults,
            help="Columns containing biological variables (like tissue_roi or cell type annotations) whose biological signal you want ComBat to protect/preserve during correction."
        )
        
        parametric = c_conf1.checkbox(
            "Parametric empirical Bayes",
            value=st.session_state.bc_config.get('parametric', True),
            help="Use parametric priors (faster) vs non-parametric priors."
        )
        
        _correction_modes = ["monotonic", "per_cluster"]
        _saved_mode = st.session_state.bc_config.get('correction_mode', 'monotonic')
        correction_mode = c_conf1.selectbox(
            "Correction application",
            _correction_modes,
            index=_correction_modes.index(_saved_mode) if _saved_mode in _correction_modes else 0,
            format_func=lambda v: {
                "monotonic": "Monotonic (recommended)",
                "per_cluster": "Per-cluster ComBat (legacy)",
            }[v],
            help=(
                "How the estimated batch effect is applied to cells.\n\n"
                "**Monotonic** uses the SOM to estimate the batch effect within matched "
                "cell populations, then applies one shift/scale per batch and marker. "
                "Cell ranking inside a batch is preserved exactly, so the correction can "
                "never reorder cells or blur the negative/positive boundary that defines "
                "a phenotype.\n\n"
                "**Per-cluster ComBat** is the literal cyCombine translation: ComBat runs "
                "separately inside each SOM node. Two cells in the same batch with the "
                "same input value can then come out different, which reorders markers that "
                "have a large true-negative population. Choose this only to reproduce an "
                "earlier run."
            ),
        )

        _unit_options = ["(use batch column as-is)"] + [
            c for c in raw_candidates if c not in selected_markers
        ]
        _saved_unit = st.session_state.bc_config.get('batch_unit_col', "(use batch column as-is)")
        batch_unit_col = c_conf1.selectbox(
            "Correction unit column (optional)",
            _unit_options,
            index=_unit_options.index(_saved_unit) if _saved_unit in _unit_options else 0,
            help=(
                "Which column identifies one processing run (one staining/acquisition batch).\n\n"
                "Differences **between** these units are removed; everything **inside** a unit "
                "is preserved. Ask of any two images: *could these differ simply because they "
                "were stained separately?* If no, they belong to the same unit.\n\n"
                "- Several regions imaged from one slide -> the **slide**, not each region.\n"
                "- TMA with one patient per core -> the **TMA slide**, not each core, so "
                "patient-to-patient differences survive.\n"
                "- Each sample on its own slide -> leave as-is; every row really is its own run.\n\n"
                "Splitting finer than the real run is the costly mistake: it normalises away "
                "the biology you are trying to measure. See 'Group images into correction "
                "units' in the file setup section for the full guide."
            ),
        )
        batch_unit_pattern = c_conf1.text_input(
            "…or derive the unit by removing a pattern",
            value=st.session_state.bc_config.get('batch_unit_pattern', ''),
            help=(
                "Regular expression stripped from the batch value to obtain the run "
                "identifier, used only when no column is selected above. For batch IDs "
                r"like `slideA_region_3`, the pattern `_region_\d+$` groups all regions "
                "of a slide into one correction unit. Left empty, the batch column is "
                "used unchanged."
            ),
        )

        mode = 'som_clustered'
        grid_size = c_conf2.number_input("SOM Grid Dimension (NxN)", 2, 50, int(st.session_state.bc_config.get('som_grid_dim', 8)))
        iterations = c_conf2.number_input("SOM Iterations", 1000, 500000, int(st.session_state.bc_config.get('som_iterations', 100000)), step=10000)
        min_cells = c_conf2.number_input("Min cells per cluster", 1, 500, int(st.session_state.bc_config.get('min_cells_per_cluster', 10)))
        
        with c_conf2.expander("Post-Correction QC Options (Relative vs Absolute)", expanded=False):
            st.markdown(
                "**Post-Correction Quality Control (QC) Filtering**\n\n"
                "After batch correction (e.g., ComBat/cyCombine), some cells may experience extreme mathematical shifts ('overcorrection'). "
                "Post-correction QC detects and removes these distorted cells using Mahalanobis distance metrics.\n\n"
                "- **Option A: Relative Filter (Percentile)**\n"
                "  Flags cells whose correction vector distance falls into the top extreme percentile (e.g., top 5%) relative to all other cells.\n"
                "  *Best when:* All slides/batches have significant batch effects and you want to remove extreme statistical outliers.\n\n"
                "- **Option B: Absolute Distance Filter (Fixed Threshold)**\n"
                "  Flags cells whose correction distance exceeds a fixed numerical threshold (e.g., > 10.0).\n"
                "  *Best when:* Some slides have minimal or no batch effects, preventing unnecessary loss of valid cells that don't need shift trimming."
            )
            st.markdown("---")
            enable_post_qc = st.checkbox(
                "Enable Post-Correction QC Filtering",
                value=st.session_state.bc_config.get('enable_post_qc', False),
                key="bc_enable_post_qc",
                help="Filter out cells that were distorted or overcorrected (using Mahalanobis distance outlier detection)."
            )
            # Post-correction QC is intentionally opt-in.  If it is disabled,
            # neither filtering strategy should appear selected—even when an
            # older project JSON contains a stale Option A/Option B value.
            if not enable_post_qc:
                st.session_state["bc_filter_overcorrected"] = False
                st.session_state["bc_filter_distance"] = False
            filter_overcorrected = st.checkbox(
                "Use Option A (Relative Percentile Filter)",
                value=(
                    bool(st.session_state.bc_config.get('filter_overcorrected', False))
                    if enable_post_qc else False
                ),
                key="bc_filter_overcorrected",
                disabled=not enable_post_qc,
                help="Remove cells flagged as having correction vectors that are relative statistical outliers."
            )
            percentile_val = st.slider(
                "Option A Outlier Percentile Threshold (%)",
                min_value=80.0,
                max_value=99.9,
                value=float(st.session_state.bc_config.get('qc_percentile_threshold', 95.0)),
                step=0.5,
                key="bc_qc_percentile_threshold",
                disabled=not enable_post_qc,
                help="Discard cells in the top (100 - X)% most extreme corrections. Default is 95.0% (discard top 5%)."
            )
            filter_distance = st.checkbox(
                "Use Option B (Absolute Distance Filter)",
                value=(
                    bool(st.session_state.bc_config.get('filter_distance', False))
                    if enable_post_qc else False
                ),
                key="bc_filter_distance",
                disabled=not enable_post_qc,
                help="Remove cells with correction distance greater than a fixed absolute threshold."
            )
            max_dist = st.number_input(
                "Option B Max Correction Distance",
                min_value=0.1,
                max_value=100.0,
                value=float(st.session_state.bc_config.get('max_correction_distance', 10.0)),
                step=0.5,
                key="bc_max_correction_distance",
                disabled=not enable_post_qc,
                help="Cells with a Mahalanobis distance greater than this value will be discarded. Recommended range: 10.0 - 15.0."
            )
        # The single Tab 6 project-save control lives in Setup & Filter.
        # Keep the correction form focused on running the correction.
        run_btn = st.form_submit_button("Run Batch Correction", type="primary", width='stretch')

        # Keep the selected QC controls in the canonical Tab 6 configuration
        # so the single Setup & Filter save button captures them too.
        st.session_state.bc_config.update({
            "enable_post_qc": bool(enable_post_qc),
            "filter_overcorrected": bool(filter_overcorrected),
            "qc_percentile_threshold": float(percentile_val),
            "filter_distance": bool(filter_distance),
            "max_correction_distance": float(max_dist),
        })
                
        if run_btn:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BATCH_CORRECTION: Running Batch Correction cyCombine mode", flush=True)
            correction_started_at = time.perf_counter()
            record_process_event(
                "tab6",
                "Batch correction started",
                details={
                    "cells_loaded": len(df),
                    "markers_selected": len(st.session_state.get("bc_selected_markers", [])),
                    "output_directory": st.session_state.bc_output_dir,
                },
            )
            st.session_state.bc_config['norm_method'] = 'scale' if 'scale' in norm_method else 'rank'
            st.session_state.bc_config['mode'] = mode
            st.session_state.bc_config['som_grid_dim'] = int(grid_size)
            st.session_state.bc_config['som_iterations'] = int(iterations)
            st.session_state.bc_config['min_cells_per_cluster'] = int(min_cells)
            st.session_state.bc_config['covariates'] = selected_covariates
            st.session_state.bc_config['correction_mode'] = correction_mode
            st.session_state.bc_config['batch_unit_col'] = batch_unit_col
            st.session_state.bc_config['batch_unit_pattern'] = batch_unit_pattern
            st.session_state.bc_config['parametric'] = parametric
            st.session_state.bc_config['corrected_suffix'] = str(corrected_suffix or '_corr')
            st.session_state.bc_config['enable_post_qc'] = enable_post_qc
            st.session_state.bc_config['filter_overcorrected'] = filter_overcorrected
            st.session_state.bc_config['qc_percentile_threshold'] = float(percentile_val)
            st.session_state.bc_config['filter_distance'] = filter_distance
            st.session_state.bc_config['max_correction_distance'] = float(max_dist)
            
            # Run Logic
            try:
                status = st.empty()
                status.text("Running correction...")
                
                # Verify and sync metadata columns from File List to the loaded data
                file_meta_df = st.session_state.bc_files_df
                meta_cols = st.session_state.bc_metadata_cols
                
                def _update_df_metadata(target_df, source_df, cols):
                    """Sync missing metadata columns from source_df (file list) to target_df (loaded cells)"""
                    if target_df is None or source_df.empty or not cols:
                         return target_df
                    
                    if "Filename" not in source_df.columns:
                         return target_df
                    
                    missing = [c for c in cols if c not in target_df.columns]
                    if not missing:
                         return target_df
                         
                    log_step(f"Syncing missing metadata columns to dataframe: {missing}")
                    temp_source = source_df.set_index('Filename')
                    for col in missing:
                         if col in temp_source.columns:
                             target_df[col] = target_df['original_filename'].map(temp_source[col])
                    return target_df

                if st.session_state.bc_cleaned_data is not None:
                     st.session_state.bc_cleaned_data = _update_df_metadata(st.session_state.bc_cleaned_data, file_meta_df, meta_cols)
                     df = st.session_state.bc_cleaned_data # Refresh local ref

                if st.session_state.bc_loaded_data is not None:
                     st.session_state.bc_loaded_data = _update_df_metadata(st.session_state.bc_loaded_data, file_meta_df, meta_cols)

                # BatchCorrector expects a column named 'batch'
                assert BatchCorrector is not None
                df_for_correction = df.copy()
                if batch_col != "batch":
                    df_for_correction = df_for_correction.rename(columns={batch_col: "batch"})

                # The correction unit is whatever was processed together (one staining
                # or acquisition run), which is not always one row per image. When
                # several images come from the same run, correcting each image
                # separately forces them to a common profile and erases the real
                # differences between them. Collapsing to the true run identifier
                # first keeps those differences and still removes the run-to-run shift.
                if batch_unit_col and batch_unit_col != "(use batch column as-is)":
                    if batch_unit_col in df_for_correction.columns:
                        df_for_correction["batch"] = df_for_correction[batch_unit_col].astype(str)
                        log_step(
                            f"Correction unit set to '{batch_unit_col}': "
                            f"{df_for_correction['batch'].nunique()} units "
                            f"(was {df[batch_col].nunique()} by {batch_col})."
                        )
                elif batch_unit_pattern:
                    try:
                        collapsed = (
                            df_for_correction["batch"].astype(str)
                            .str.replace(batch_unit_pattern, "", regex=True)
                        )
                        n_units = collapsed.nunique()
                        if 0 < n_units < df_for_correction["batch"].nunique():
                            df_for_correction["batch"] = collapsed
                            log_step(
                                f"Correction unit derived by removing /{batch_unit_pattern}/: "
                                f"{n_units} units (was {df[batch_col].nunique()})."
                            )
                        else:
                            log_step(
                                f"Pattern /{batch_unit_pattern}/ produced {n_units} units; "
                                "leaving the batch column unchanged."
                            )
                    except re.error as exc:
                        log_step(f"Invalid correction-unit pattern ({exc}); using the batch column as-is.")

                # Markers must exist in all selected CSVs
                markers = list(st.session_state.get('bc_selected_markers', []) or [])
                if not markers:
                    raise ValueError(
                        "No markers selected for batch correction. "
                        "Go to '1. Setup & Filter' → 'Common Columns & Marker Selection' and select common markers."
                    )

                cell_id_col = st.session_state.bc_config.get('cell_id_col', 'cell_id')
                missing_required = [
                    c for c in [cell_id_col, 'batch', *markers]
                    if c not in df_for_correction.columns
                ]
                if missing_required:
                    raise ValueError(f"Missing required columns for correction: {missing_required}")

                df_for_correction.loc[:, markers] = (
                    df_for_correction[markers]
                    .apply(pd.to_numeric, errors='coerce')
                    .replace([np.inf, -np.inf], np.nan)
                )
                df_for_correction.loc[:, 'batch'] = df_for_correction['batch'].astype(str)

                valid_mask = (~df_for_correction[markers].isna().any(axis=1))
                dropped_n = int((~valid_mask).sum())
                if dropped_n > 0:
                    nan_by_marker = df_for_correction[markers].isna().sum().sort_values(ascending=False)
                    top_nan = nan_by_marker[nan_by_marker > 0].head(10)
                    st.warning(
                        f"Excluded {dropped_n:,} cells from ComBat due to NaN/Inf in marker values. "
                        "They will remain in the output with NaN corrected columns."
                    )
                    if not top_nan.empty:
                        st.caption("Markers with missing values (top 10):")
                        st.dataframe(top_nan.rename('missing_cells'))

                df_for_correction_valid = df_for_correction.loc[valid_mask].copy()
                
                valid_batches = df_for_correction_valid['batch'].unique()
                log_step(f"Batches entering ComBat: {len(valid_batches)}. (Loaded was {df[batch_col].nunique()})")

                corrector = BatchCorrector(
                    markers=markers,
                    normalization_method=st.session_state.bc_config.get('norm_method', 'scale'),
                    som_grid_size=(int(grid_size), int(grid_size)),
                    som_iterations=int(iterations),
                    random_seed=42,
                    verbose=True,
                    mode='som_clustered',
                    min_cells_per_cluster=int(min_cells),
                    covariates=selected_covariates,
                    parametric=parametric,
                    arcsinh_cofactor=float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0)),
                    correction_mode=correction_mode,
                )

                corrected_valid = corrector.correct_batch_effects(df_for_correction_valid)
                
                # Calculate metrics in native ComBat output format (using 'batch' and '_corr') before renaming
                from spacemap.batch_correction_utils import calculate_emd_reduction, calculate_mad_score
                
                # Subsample to speed up metric calculation dramatically (Option 1)
                subsample_size = 10000
                if len(df_for_correction_valid) > subsample_size:
                    df_sub = df_for_correction_valid.sample(n=subsample_size, random_state=42)
                    corr_sub = corrected_valid[corrected_valid[cell_id_col].isin(df_sub[cell_id_col])]
                else:
                    df_sub = df_for_correction_valid
                    corr_sub = corrected_valid
                    
                cofactor = float(st.session_state.bc_config.get('arcsinh_cofactor', 5.0))
                emd = calculate_emd_reduction(df_sub, corr_sub, markers, batch_col='batch', cofactor=cofactor)
                mad = calculate_mad_score(df_sub, corr_sub, markers, batch_col='batch', cofactor=cofactor)

                # Post-correction QC filtering (if enabled)
                corrected_valid_retained = corrected_valid.copy()
                if enable_post_qc:
                    status.text("Running post-correction QC...")
                    # 1. Detect overcorrected cells
                    corrected_valid_flagged = detect_overcorrected_cells(
                        data=corrected_valid,
                        markers=markers,
                        batch_col='batch',
                        percentile_threshold=float(percentile_val)
                    )
                    
                    # 2. Build thresholds dict
                    post_qc_thresholds = {
                        'enable_post_correction_qc': True,
                        'filter_overcorrected': filter_overcorrected,
                    }
                    if filter_distance:
                        post_qc_thresholds['max_correction_distance'] = max_dist
                        
                    # 3. Apply filters
                    corrected_valid_retained, qc_stats = apply_post_correction_qc_filters(
                        data=corrected_valid_flagged,
                        qc_thresholds=post_qc_thresholds
                    )
                    
                    # Display summary to the user
                    st.info(
                        f"**Post-Correction QC Summary:**\n"
                        f"- Initial cells: {qc_stats['initial_cells']:,}\n"
                        f"- Cells retained (passed QC): {qc_stats['final_cells']:,} ({qc_stats['retention_rate']:.1%})\n"
                        f"- Cells removed (failed QC): {qc_stats['total_removed']:,}"
                    )

                if batch_col != "batch":
                    corrected_valid_retained = corrected_valid_retained.rename(columns={"batch": batch_col})

                # Preserve all loaded cells (notebook behavior) and attach corrected values where available.
                base_df = st.session_state.get('bc_loaded_data')
                if base_df is None:
                    base_df = st.session_state.bc_cleaned_data

                suffix = str(st.session_state.bc_config.get('corrected_suffix', '_corr') or '_corr')
                corrected_cols_native = [f"{m}_corr" for m in markers]
                corrected_cols_final = [f"{m}{suffix}" for m in markers]
                rename_map = {
                    native: final
                    for native, final in zip(corrected_cols_native, corrected_cols_final)
                    if native != final
                }
                if rename_map:
                    corrected_valid_retained = corrected_valid_retained.rename(columns=rename_map)

                base_df = base_df.copy()
                for col in corrected_cols_final:
                    if col in base_df.columns:
                        base_df = base_df.drop(columns=[col])

                corr_subset = corrected_valid_retained[[cell_id_col, *corrected_cols_final]].copy()
                corrected_df = base_df.merge(corr_subset, on=cell_id_col, how='left')
                
                # Add indicator column if requested
                add_indicator = st.session_state.bc_config.get('add_corrected_indicator', False)
                indicator_col = st.session_state.bc_config.get('corrected_indicator_col', 'corrected')
                if add_indicator and indicator_col:
                    if indicator_col in corrected_df.columns:
                        corrected_df = corrected_df.drop(columns=[indicator_col])
                    corrected_df[indicator_col] = 0
                    corrected_df.loc[corrected_df[cell_id_col].isin(corrected_valid_retained[cell_id_col]), indicator_col] = 1
                    log_step(f"Added batch correction indicator column '{indicator_col}' (1 for corrected, 0 for excluded)")
                
                # Save either as the established directory outputs or replace
                # selected correction columns in the unified source CSV.
                if unified_mode:
                    unified_path = Path(st.session_state.bc_unified_csv_path).expanduser()
                    source_df = pd.read_csv(unified_path)
                    source_df = _assign_batch_correction_cell_ids(
                        source_df,
                        st.session_state.bc_config,
                        batch_col,
                    )
                    source_batch_values = source_df[batch_col].astype(str)
                    selected_rows = st.session_state.bc_files_df.copy()
                    selected_rows["Process"] = selected_rows["Process"].map(
                        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}
                    )
                    selected_batch_values = set(
                        selected_rows.loc[selected_rows["Process"], "Filename"].astype(str)
                    )
                    active_source_mask = source_batch_values.isin(selected_batch_values)
                    active_cell_ids = source_df.loc[active_source_mask, cell_id_col]

                    metadata_by_batch = selected_rows.set_index("Filename")
                    for metadata_column in st.session_state.bc_metadata_cols:
                        if metadata_column in metadata_by_batch.columns:
                            mapped_values = source_batch_values.map(metadata_by_batch[metadata_column])
                            source_df.loc[active_source_mask, metadata_column] = mapped_values.loc[active_source_mask]

                    retained_subset = corrected_valid_retained[
                        [cell_id_col, *corrected_cols_final]
                    ].copy()
                    source_df = _replace_corrected_columns(
                        source_df,
                        retained_subset,
                        cell_id_col=cell_id_col,
                        corrected_columns=corrected_cols_final,
                        active_cell_ids=active_cell_ids,
                        indicator_col=indicator_col if add_indicator and indicator_col else None,
                    )
                    out_path = str(_write_csv_atomic(source_df, unified_path))
                    corrected_df = source_df
                else:
                    os.makedirs(st.session_state.bc_output_dir, exist_ok=True)
                    out_name = f"batch_corrected_merged_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    out_path = os.path.join(st.session_state.bc_output_dir, out_name)
                    corrected_df.to_csv(out_path, index=False)

                    individual_dir = os.path.join(st.session_state.bc_output_dir, "individual_corrected")
                    os.makedirs(individual_dir, exist_ok=True)

                    if 'original_filename' in corrected_df.columns:
                        for fname in corrected_df['original_filename'].unique():
                            sub_df = corrected_df[corrected_df['original_filename'] == fname].copy()
                            sub_df = sub_df.drop(columns=['original_filename'])
                            sub_out_path = os.path.join(individual_dir, fname)
                            sub_df.to_csv(sub_out_path, index=False)
                
                st.session_state.bc_last_run_file = out_path
                st.session_state.bc_vis_file = out_path
                
                success_action = "Updated unified CSV" if unified_mode else "Saved"
                st.success(f"Correction complete! {success_action}: {out_path}")
                st.info(f"cyCombine Quality Metrics:\n- EMD Reduction: {emd:.4f} (higher is better, 1.0 is perfect)\n- MAD Score: {mad:.4f} (lower is better, 0.0 is perfect)")
                record_process_event(
                    "tab6",
                    "Batch correction complete",
                    status="complete",
                    duration_s=time.perf_counter() - correction_started_at,
                    details={
                        "output": out_path,
                        "cells_exported": len(corrected_df),
                        "markers_corrected": len(markers),
                        "emd_reduction": emd,
                        "mad_score": mad,
                    },
                )
                
            except Exception as e:
                st.exception(e)
                record_process_event(
                    "tab6",
                    "Batch correction failed",
                    message=str(e),
                    status="error",
                    duration_s=time.perf_counter() - correction_started_at,
                )

def _make_prebinned_histogram_trace(values, bin_size, val_min, val_max, density_norm, name, color, legend_group, show_legend):
    import numpy as np
    import plotly.graph_objects as go
    
    # Filter out NaNs and Infs
    vals = values[~np.isnan(values) & ~np.isinf(values)]
    if len(vals) == 0:
        return go.Bar(
            x=[], y=[],
            name=name,
            showlegend=show_legend,
            legendgroup=legend_group
        )
        
    counts, edges = np.histogram(vals, bins=bin_size, range=(val_min, val_max), density=density_norm)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    
    return go.Bar(
        x=bin_centers,
        y=counts,
        name=name,
        opacity=0.5,
        marker_color=color,
        legendgroup=legend_group,
        showlegend=show_legend,
    )

def _make_kde_trace(values, val_min, val_max, name, color, legend_group, show_legend):
    import numpy as np
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde
    
    # Filter out NaNs and Infs
    vals = values[~np.isnan(values) & ~np.isinf(values)]
    if len(vals) < 3: # Need at least a few points to estimate density
        return go.Scatter(
            x=[], y=[],
            mode='lines',
            name=name,
            showlegend=show_legend,
            legendgroup=legend_group
        )
        
    try:
        # Fit Gaussian KDE
        kde = gaussian_kde(vals)
        x_grid = np.linspace(val_min, val_max, 200)
        y_density = kde(x_grid)
    except Exception:
        # Fallback if covariance matrix singular
        return go.Scatter(
            x=[], y=[],
            mode='lines',
            name=name,
            showlegend=show_legend,
            legendgroup=legend_group
        )
        
    return go.Scatter(
        x=x_grid,
        y=y_density,
        mode='lines',
        line=dict(width=2, color=color, shape='spline'),
        fill='tozeroy',
        opacity=0.3,
        name=name,
        legendgroup=legend_group,
        showlegend=show_legend,
    )

def _make_distribution_trace(values, plot_style, bin_size, val_min, val_max, density_norm, name, color, legend_group, show_legend):
    if plot_style == "Smooth Density (KDE)":
        return _make_kde_trace(values, val_min, val_max, name, color, legend_group, show_legend)
    else:
        return _make_prebinned_histogram_trace(values, bin_size, val_min, val_max, density_norm, name, color, legend_group, show_legend)

def render_visualization(project_root: Path):
    """Subtab 2: Visualization."""
    st.header("2. Visualization & QC")
    st.markdown("Compare data distribution before and after batch correction.")
    
    # Initialize State
    if 'bc_vis_file' not in st.session_state or not st.session_state.bc_vis_file or not os.path.exists(st.session_state.bc_vis_file):
        latest_file = ""
        out_dir = st.session_state.get('bc_output_dir', '')
        if out_dir and os.path.isdir(out_dir):
            try:
                csv_files = [
                    os.path.join(out_dir, f)
                    for f in os.listdir(out_dir)
                    if f.lower().endswith('.csv') and os.path.isfile(os.path.join(out_dir, f))
                ]
                if csv_files:
                    csv_files.sort(key=os.path.getmtime, reverse=True)
                    latest_file = csv_files[0]
            except Exception:
                pass
        st.session_state.bc_vis_file = latest_file or st.session_state.get('bc_last_run_file', '')
        
    # 1. File Selection
    c1, c2 = st.columns([3, 1])
    
    default_val = st.session_state.bc_vis_file
    val_in_state = st.session_state.pop("bc_vis_file_in", default_val)
    vis_file = c1.text_input("Corrected Data CSV", value=val_in_state, key="bc_vis_file_in")
    if vis_file != st.session_state.bc_vis_file:
        st.session_state.bc_vis_file = vis_file
        
    if c2.button("Browse", key="bc_vis_browse"):
        st.session_state.show_bc_vis_browser = True
        
    if st.session_state.get('show_bc_vis_browser'):
        path = render_file_browser(st.session_state.bc_output_dir, "bc_vis_browser", 'file')
        if path == "CANCEL":
            st.session_state.show_bc_vis_browser = False
            st.rerun()
        elif path:
            st.session_state.bc_vis_file = path
            st.session_state["bc_vis_file_in"] = path
            st.session_state.show_bc_vis_browser = False
            st.rerun()
            
    if not st.session_state.bc_vis_file:
        st.info("Please select a batch corrected CSV file.")
        return
        
    if not os.path.exists(st.session_state.bc_vis_file):
        st.error(f"File not found: {st.session_state.bc_vis_file}. Please check the path and confirm case sensitivity on Linux (e.g. 'Innate' vs 'innate').")
        return

    # Tab 6 renders all three subtabs during a Streamlit rerun.  Do not make
    # a setup/scan action block while a very large corrected CSV is loaded by
    # the visualization subtab in the background.  Loading remains available
    # through an explicit button for users who do want to visualize it.
    vis_path = Path(st.session_state.bc_vis_file)
    vis_path_key = str(vis_path)
    if st.session_state.get("bc_vis_load_requested_path") != vis_path_key:
        st.session_state["bc_vis_load_requested_path"] = vis_path_key
        st.session_state["bc_vis_load_requested"] = False
    large_file_limit = 250 * 1024 * 1024
    try:
        vis_file_size = vis_path.stat().st_size
    except OSError:
        vis_file_size = 0
    if vis_file_size > large_file_limit and not st.session_state.get("bc_vis_load_requested", False):
        st.warning(
            f"The selected visualization file is {vis_file_size / (1024 ** 3):.1f} GB. "
            "It will not be loaded automatically during unrelated Tab 6 actions."
        )
        if st.button("Load Large Visualization File", key="bc_vis_load_large_file", type="secondary"):
            st.session_state["bc_vis_load_requested"] = True
            st.rerun()
        return
        
    # Load Data
    @st.cache_data
    def load_bc_data(path, modified_ns):
        return pd.read_csv(path)
        
    try:
        modified_ns = Path(st.session_state.bc_vis_file).stat().st_mtime_ns
        df = load_bc_data(st.session_state.bc_vis_file, modified_ns)
        st.caption(f"Loaded {len(df)} cells.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return
        
    batch_col = st.session_state.bc_config.get('batch_id_col', 'batch_id')
    if batch_col not in df.columns:
        for fallback in ['batch', 'batch_id', 'Filename', 'original_filename']:
            if fallback in df.columns:
                batch_col = fallback
                break
        
    # Identify Markers (Original vs Corrected)
    suffix = st.session_state.bc_config.get('corrected_suffix', '_corr')
    
    # Try to find pairs
    corrected_cols = [c for c in df.columns if c.endswith(suffix)]
    original_markers_found = []
    for c in corrected_cols:
        orig = c[:-len(suffix)]
        if orig in df.columns:
            original_markers_found.append(orig)
            
    original_markers_found = sorted(list(set(original_markers_found)))
    has_pairs = len(original_markers_found) > 0
    
    if not has_pairs:
        # Fallback: list all numeric columns
        original_markers_found = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != batch_col and not c.endswith('_corr')]
        st.warning("No batch-corrected paired columns (e.g. CD45 and CD45_corr) detected. Showing raw numeric columns.")
        
    # Plot controls
    with st.expander("Plot settings", expanded=False):
        c_vis1, c_vis2, c_vis3 = st.columns(3)
        transform = c_vis1.selectbox(
            "Transform",
            options=["Linear", "Log10", "Arcsinh", "Z-score"],
            index=0,
            key="bc_vis_transform",
        )
        arcsinh_cofactor = c_vis2.number_input(
            "Arcsinh cofactor",
            min_value=0.1,
            value=float(st.session_state.bc_config.get('arcsinh_cofactor', st.session_state.bc_config.get('qc_arcsinh_cofactor', 5.0))),
            step=0.5,
            key="bc_vis_arcsinh",
        )
        log_pseudocount = c_vis2.number_input(
            "Log10 pseudocount",
            min_value=0.0,
            value=float(st.session_state.bc_config.get('qc_log_pseudocount', 1.0)),
            step=0.5,
            key="bc_vis_log_pseudocount",
        )
        plot_style = c_vis3.selectbox("Plot Style", ["Binned Histogram", "Smooth Density (KDE)"], index=0, key="bc_vis_plot_style")
        bin_size = c_vis3.number_input("Bins", min_value=5, max_value=200, value=50, step=5, key="bc_vis_bins")
        density_norm = c_vis3.checkbox("Normalize to density", value=True, key="bc_vis_density_norm")

    # Layout using Tabs
    vis_tab1, vis_tab2, vis_tab3, vis_tab4 = st.tabs([
        "📊 Marker Distributions",
        "🌀 PCA & UMAP Batch Mixing",
        "📈 Correction Distance & Variance",
        "📊 Batch Integration Metrics (EMD & Silhouette)"
    ])
    
    unique_batches = sorted(df[batch_col].dropna().unique())
    n_batches = len(unique_batches)
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Cycle colors
    all_colors = (px.colors.qualitative.Plotly + 
                  px.colors.qualitative.D3 + 
                  px.colors.qualitative.Set2)
    colors = [all_colors[i % len(all_colors)] for i in range(n_batches)]
    
    with vis_tab1:
        st.subheader("Before/After Marker Overlay Histograms")
        selected_marker = st.selectbox("Select Marker to Visualize", original_markers_found)
        
        if selected_marker:
            if has_pairs and f"{selected_marker}{suffix}" in df.columns:
                # Calculate global min/max for the marker to align bins across plots
                all_before = _transform_vector(df[selected_marker], transform, arcsinh_cofactor, log_pseudocount)
                
                corr_m = f"{selected_marker}{suffix}"
                corr_linear = np.sinh(df[corr_m]) * arcsinh_cofactor
                all_after = _transform_vector(corr_linear, transform, arcsinh_cofactor, log_pseudocount)
                
                all_before_clean = all_before[~np.isnan(all_before) & ~np.isinf(all_before)]
                all_after_clean = all_after[~np.isnan(all_after) & ~np.isinf(all_after)]
                
                val_min = float(min(all_before_clean.min() if not all_before_clean.empty else 0.0,
                                    all_after_clean.min() if not all_after_clean.empty else 0.0))
                val_max = float(max(all_before_clean.max() if not all_before_clean.empty else 1.0,
                                    all_after_clean.max() if not all_after_clean.empty else 1.0))
                if np.isnan(val_min) or np.isinf(val_min): val_min = 0.0
                if np.isnan(val_max) or np.isinf(val_max): val_max = 1.0
                if val_min == val_max: val_max += 1.0

                fig = make_subplots(rows=1, cols=2, subplot_titles=("Before Batch Correction", "After Batch Correction"))
                
                # Plot Before
                for i, batch in enumerate(unique_batches):
                    batch_mask = df[batch_col] == batch
                    before_vals = _transform_vector(df.loc[batch_mask, selected_marker], transform, arcsinh_cofactor, log_pseudocount)
                    trace = _make_distribution_trace(
                        before_vals, plot_style, bin_size, val_min, val_max, density_norm,
                        name=str(batch), color=colors[i % len(colors)],
                        legend_group=str(batch), show_legend=True
                    )
                    fig.add_trace(trace, row=1, col=1)
                
                # Plot After
                for i, batch in enumerate(unique_batches):
                    batch_mask = df[batch_col] == batch
                    corrected_arcsinh = df.loc[batch_mask, corr_m]
                    corrected_linear = np.sinh(corrected_arcsinh) * arcsinh_cofactor
                    after_vals = _transform_vector(corrected_linear, transform, arcsinh_cofactor, log_pseudocount)
                    trace = _make_distribution_trace(
                        after_vals, plot_style, bin_size, val_min, val_max, density_norm,
                        name=str(batch), color=colors[i % len(colors)],
                        legend_group=str(batch), show_legend=False
                    )
                    fig.add_trace(trace, row=1, col=2)
                
                fig.update_layout(
                    barmode='overlay',
                    bargap=0.0,
                    height=450,
                    margin=dict(t=50, b=50, l=50, r=50),
                    legend=dict(title="Batches")
                )
                st.plotly_chart(fig, width='stretch')
            else:
                # Single plot
                all_vals = _transform_vector(df[selected_marker], transform, arcsinh_cofactor, log_pseudocount)
                all_vals_clean = all_vals[~np.isnan(all_vals) & ~np.isinf(all_vals)]
                val_min = float(all_vals_clean.min() if not all_vals_clean.empty else 0.0)
                val_max = float(all_vals_clean.max() if not all_vals_clean.empty else 1.0)
                if np.isnan(val_min) or np.isinf(val_min): val_min = 0.0
                if np.isnan(val_max) or np.isinf(val_max): val_max = 1.0
                if val_min == val_max: val_max += 1.0
                
                fig = go.Figure()
                for i, batch in enumerate(unique_batches):
                    batch_mask = df[batch_col] == batch
                    vals = _transform_vector(df.loc[batch_mask, selected_marker], transform, arcsinh_cofactor, log_pseudocount)
                    trace = _make_distribution_trace(
                        vals, plot_style, bin_size, val_min, val_max, density_norm,
                        name=str(batch), color=colors[i % len(colors)],
                        legend_group=str(batch), show_legend=True
                    )
                    fig.add_trace(trace)
                fig.update_layout(
                    title=f"Distribution of {selected_marker} by Batch",
                    barmode='overlay',
                    bargap=0.0,
                    height=450,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                st.plotly_chart(fig, width='stretch')
                
    with vis_tab2:
        st.subheader("Dimensionality Reduction comparisons")
        st.caption("Check how well the batches mix after correction using PCA and UMAP (subsampled to max 5,000 cells for responsiveness).")
        
        # Subsample for PCA/UMAP
        sample_size = min(len(df), 5000)
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
        
        # Original and Corrected columns
        pca_markers = [m for m in original_markers_found if m in df_sample.columns]
        pca_corr_markers = [f"{m}{suffix}" for m in pca_markers if f"{m}{suffix}" in df_sample.columns]
        
        if len(pca_markers) > 1 and len(pca_corr_markers) > 1:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            pca = PCA(n_components=2)
            
            # Before: apply ArcSinh transform to match corrected space (force numeric to avoid string division TypeError)
            orig_numeric = df_sample[pca_markers].apply(pd.to_numeric, errors='coerce').fillna(0)
            orig_arcsinh = np.arcsinh(orig_numeric / arcsinh_cofactor)
            scaled_before = scaler.fit_transform(orig_arcsinh)
            pca_before = pca.fit_transform(scaled_before)
            
            # After: already in ArcSinh space (force numeric to avoid string scaling TypeError)
            corr_numeric = df_sample[pca_corr_markers].apply(pd.to_numeric, errors='coerce').fillna(0)
            scaled_after = scaler.fit_transform(corr_numeric)
            pca_after = pca.fit_transform(scaled_after)
            
            # Plot PCA
            st.markdown("#### PCA (Principal Component Analysis)")
            fig_pca = make_subplots(rows=1, cols=2, subplot_titles=("Before Batch Correction", "After Batch Correction"))
            
            for i, batch in enumerate(unique_batches):
                batch_mask = (df_sample[batch_col] == batch).values
                fig_pca.add_trace(
                    go.Scattergl(
                        x=pca_before[batch_mask, 0],
                        y=pca_before[batch_mask, 1],
                        mode='markers',
                        name=str(batch),
                        marker=dict(size=3, opacity=0.6, color=colors[i % len(colors)]),
                        legendgroup=str(batch),
                        showlegend=True
                    ),
                    row=1, col=1
                )
                fig_pca.add_trace(
                    go.Scattergl(
                        x=pca_after[batch_mask, 0],
                        y=pca_after[batch_mask, 1],
                        mode='markers',
                        name=str(batch),
                        marker=dict(size=3, opacity=0.6, color=colors[i % len(colors)]),
                        legendgroup=str(batch),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
            fig_pca.update_layout(height=450, margin=dict(t=50, b=50, l=50, r=50))
            st.plotly_chart(fig_pca, width='stretch')
            
            # UMAP (Run on click)
            st.markdown("#### UMAP (Uniform Manifold Approximation)")
            
            if st.button("Run UMAP Comparison", key="bc_vis_run_umap"):
                with st.spinner("Computing UMAP embeddings..."):
                    try:
                        from umap import UMAP
                        reducer = UMAP(n_components=2, random_state=42)
                        
                        umap_before = reducer.fit_transform(scaled_before)
                        umap_after = reducer.fit_transform(scaled_after)
                        
                        fig_umap = make_subplots(rows=1, cols=2, subplot_titles=("Before Batch Correction", "After Batch Correction"))
                        
                        for i, batch in enumerate(unique_batches):
                            batch_mask = (df_sample[batch_col] == batch).values
                            fig_umap.add_trace(
                                go.Scattergl(
                                    x=umap_before[batch_mask, 0],
                                    y=umap_before[batch_mask, 1],
                                    mode='markers',
                                    name=str(batch),
                                    marker=dict(size=3, opacity=0.6, color=colors[i % len(colors)]),
                                    legendgroup=str(batch),
                                    showlegend=True
                                ),
                                row=1, col=1
                            )
                            fig_umap.add_trace(
                                go.Scattergl(
                                    x=umap_after[batch_mask, 0],
                                    y=umap_after[batch_mask, 1],
                                    mode='markers',
                                    name=str(batch),
                                    marker=dict(size=3, opacity=0.6, color=colors[i % len(colors)]),
                                    legendgroup=str(batch),
                                    showlegend=False
                                ),
                                row=1, col=2
                            )
                            
                        fig_umap.update_layout(height=450, margin=dict(t=50, b=50, l=50, r=50))
                        st.plotly_chart(fig_umap, width='stretch')
                    except ImportError:
                        st.error("umap-learn is not installed in the active environment.")
                    except Exception as e:
                        st.error(f"UMAP failed: {e}")
        else:
            st.info("PCA/UMAP requires at least 2 batch-corrected markers.")
            
    with vis_tab3:
        st.subheader("Correction Distance and Variance preservation")
        
        if has_pairs and len(pca_markers) > 0 and len(pca_corr_markers) > 0:
            # Subsample
            sample_size = min(len(df), 10000)
            df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
            
            # Compute in ArcSinh space (force numeric to avoid string division TypeError)
            orig_numeric = df_sample[pca_markers].apply(pd.to_numeric, errors='coerce').fillna(0)
            orig_vals_arcsinh = np.arcsinh(orig_numeric.values / arcsinh_cofactor)
            corr_numeric = df_sample[pca_corr_markers].apply(pd.to_numeric, errors='coerce').fillna(0)
            corr_vals = corr_numeric.values
            
            # Compute Euclidean correction distance per cell in ArcSinh space
            correction_deltas = corr_vals - orig_vals_arcsinh
            distances = np.linalg.norm(correction_deltas, axis=1)
            
            # Plot box plot of distances
            fig_dist = go.Figure()
            for i, batch in enumerate(unique_batches):
                batch_mask = (df_sample[batch_col] == batch).values
                fig_dist.add_trace(
                    go.Box(
                        y=distances[batch_mask],
                        name=str(batch),
                        marker_color=colors[i % len(colors)]
                    )
                )
            fig_dist.update_layout(
                title="Correction Magnitude by Batch (Euclidean distance in ArcSinh space)",
                yaxis_title="Euclidean Distance (ArcSinh)",
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            st.plotly_chart(fig_dist, width='stretch')
            
            # Table of metrics
            st.markdown("#### Marker Quality Control Metrics (Calculated in ArcSinh Space)")
            metrics = []
            for marker in pca_markers:
                corr_m = f"{marker}{suffix}"
                if corr_m in df.columns:
                    orig_numeric = df[marker].apply(pd.to_numeric, errors='coerce').fillna(0)
                    orig_arcsinh = np.arcsinh(orig_numeric / arcsinh_cofactor)
                    orig_var = orig_arcsinh.var()
                    corr_numeric = df[corr_m].apply(pd.to_numeric, errors='coerce').fillna(0)
                    corr_var = corr_numeric.var()
                    ratio = corr_var / orig_var if orig_var > 0 else 0
                    mean_diff = np.abs(corr_numeric - orig_arcsinh).mean()
                    
                    metrics.append({
                        "Marker": marker,
                        "Mean Shift (ArcSinh)": round(mean_diff, 4),
                        "Variance Before (ArcSinh)": round(orig_var, 4),
                        "Variance After (ArcSinh)": round(corr_var, 4),
                        "Variance Preservation Ratio": round(ratio, 4),
                        "Status": "Excellent (>=0.9)" if ratio >= 0.9 else "Good (0.7-0.9)" if ratio >= 0.7 else "Reduced (<0.7)"
                    })
                    
            st.dataframe(pd.DataFrame(metrics), width='stretch', hide_index=True)
        else:
            st.info("Metrics calculation requires at least one batch-corrected marker pair.")
            
    with vis_tab4:
        st.subheader("Batch Integration Metrics (EMD & Silhouette)")
        st.markdown(
            "These metrics quantify batch correction quality. "
            "**Earth Mover's Distance (EMD)** and **Silhouette score** assess how well the batch effects have been removed."
        )
        
        with st.spinner("Calculating batch integration metrics..."):
            try:
                from spacemap.batch_correction_utils import calculate_batch_emd, calculate_batch_silhouette
                
                # Compute EMD and Silhouette using correct scale
                emd_df = calculate_batch_emd(df, original_markers_found, batch_col, cofactor=arcsinh_cofactor)
                sil_dict = calculate_batch_silhouette(df, original_markers_found, batch_col, sample_size=2000)
                
                # Show Silhouette score comparison
                st.markdown("### 1. Silhouette Score Comparison")
                st.markdown(
                    "The Silhouette score measures batch clustering. "
                    "**Lower silhouette scores (closer to 0 or negative)** indicate that cells from different batches "
                    "are well-mixed and do not separate by batch."
                )
                
                c_sil1, c_sil2 = st.columns([1, 2])
                with c_sil1:
                    st.metric(
                        label="Silhouette Before",
                        value=f"{sil_dict['silhouette_before']:.4f}"
                    )
                    st.metric(
                        label="Silhouette After",
                        value=f"{sil_dict['silhouette_after']:.4f}",
                        delta=f"{sil_dict['silhouette_after'] - sil_dict['silhouette_before']:.4f}",
                        delta_color="inverse"
                    )
                
                with c_sil2:
                    fig_sil = go.Figure()
                    fig_sil.add_trace(go.Bar(
                        x=['Before Correction', 'After Correction'],
                        y=[sil_dict['silhouette_before'], sil_dict['silhouette_after']],
                        marker_color=['indianred', 'mediumseagreen'],
                        text=[f"{sil_dict['silhouette_before']:.3f}", f"{sil_dict['silhouette_after']:.3f}"],
                        textposition='auto',
                        width=0.4
                    ))
                    fig_sil.update_layout(
                        title="Batch Identity Silhouette Score (Lower is Better)",
                        yaxis_title="Silhouette Score",
                        height=350,
                        margin=dict(t=40, b=40, l=40, r=40)
                    )
                    st.plotly_chart(fig_sil, width="stretch")
                
                # Show EMD comparison
                st.markdown("### 2. Earth Mover's Distance (EMD) per Marker")
                st.markdown(
                    "EMD measures the distance between the distribution of each batch. "
                    "A reduction in EMD indicates that the batch distributions have been successfully aligned."
                )
                
                if not emd_df.empty:
                    fig_emd = go.Figure()
                    fig_emd.add_trace(go.Bar(
                        x=emd_df['Marker'],
                        y=emd_df['EMD_Before'],
                        name='Before Correction (ArcSinh)',
                        marker_color='indianred'
                    ))
                    fig_emd.add_trace(go.Bar(
                        x=emd_df['Marker'],
                        y=emd_df['EMD_After'],
                        name='After Correction (ArcSinh)',
                        marker_color='mediumseagreen'
                    ))
                    fig_emd.update_layout(
                        title="Earth Mover's Distance (EMD) per Marker: Before vs. After (Lower is Better)",
                        xaxis_title="Marker",
                        yaxis_title="Average Pairwise EMD",
                        barmode='group',
                        height=450,
                        margin=dict(t=50, b=50, l=50, r=50),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_emd, width="stretch")
                    
                    # EMD Table
                    st.markdown("#### Detailed EMD Metrics")
                    emd_df_disp = emd_df.copy()
                    emd_df_disp['Reduction'] = emd_df_disp['Reduction'].map(lambda x: f"{x:.1%}" if isinstance(x, float) else "0.0%")
                    emd_df_disp['EMD_Before'] = emd_df_disp['EMD_Before'].round(4)
                    emd_df_disp['EMD_After'] = emd_df_disp['EMD_After'].round(4)
                    st.dataframe(emd_df_disp, width='stretch', hide_index=True)
                else:
                    st.info("No EMD data could be calculated.")
                    
            except Exception as e:
                st.error(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
