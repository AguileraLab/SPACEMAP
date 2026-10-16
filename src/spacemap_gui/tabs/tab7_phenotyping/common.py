# CRITICAL: Protobuf fix MUST be the absolute first thing that runs
# Before ANY import that could transitively import protobuf (tensorflow, streamlit, scanpy, etc.)
import sys
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("SPACEMAP_TAB_LOGS", "1")

from spacemap_gui import terminal


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab7', message, step=None, level=level)


# Force flush for progress visibility in Streamlit
class _FlushStdout:
    def __init__(self, stream):
        self._stream = stream
    def write(self, data):
        try:
            self._stream.write(data)
            self._stream.flush()
        except (RuntimeError, ValueError):
            # Ignore errors during shutdown (e.g. reentrant calls)
            pass

    def flush(self):
        try:
            self._stream.flush()
        except (RuntimeError, ValueError):
            pass
    def __getattr__(self, attr):
        return getattr(self._stream, attr)

if not isinstance(sys.stdout, _FlushStdout):
    sys.stdout = _FlushStdout(sys.stdout)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import ast
import hashlib
import logging
import json
import re
import uuid
import csv
import platform
import socket
import subprocess
from datetime import datetime
from functools import lru_cache
from importlib import metadata as importlib_metadata

# ==============================================================================
# Setup Paths & Imports
# ==============================================================================
current_file = Path(__file__).resolve()
# In older version (V13): backup/V13/interface_gui/tabs/tab7_phenotyping/common.py -> parents[3] is project root
# In active version: src/spacemap_gui/tabs/tab7_phenotyping/common.py -> parents[4] is project root, parents[3] is src
if current_file.parents[3].name == "src":
    project_root_calc = current_file.parents[4]
    src_path = current_file.parents[3]
else:
    project_root_calc = current_file.parents[3]
    src_path = project_root_calc / "src"

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Tab 7 shared state cache (persisted across sessions via interface_state.json)
# Uses the shared atomic accessor in spacemap_gui.persistence so that
# concurrent writes from app.py / tab8 / tab9 don't lose Tab 7 keys.
from spacemap_gui.persistence import (
    load_interface_state as _load_tab7_state,
    update_interface_state as _update_tab7_state,
)
from spacemap_gui.process_log import record_process_event


def get_cached_global_pheno_dir(default_value: str) -> str:
    state = _load_tab7_state()
    return state.get('global_pheno_dir', default_value)


def set_cached_global_pheno_dir(value: str) -> None:
    _update_tab7_state({'global_pheno_dir': value})


def get_cached_global_pheno_file(default_value: str | None = None) -> str | None:
    state = _load_tab7_state()
    return state.get('global_pheno_file', default_value)


def set_cached_global_pheno_file(value: str | None) -> None:
    _update_tab7_state({'global_pheno_file': value})

# NOTE: ph_leiden_utils, ph_som_utils, ph_thres_utils, ph_scimap_utils,
# and config.default_config were previously imported here but were never
# referenced in common.py.  They are imported inside the individual subtab
# files that actually use them, so the top-level import was pure startup
# cost with no benefit.  Removed to speed up app loading.

try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    _HAS_STREAMLIT_PLOTLY_EVENTS = True
except Exception:
    plotly_events = None  # type: ignore
    _HAS_STREAMLIT_PLOTLY_EVENTS = False

# ==============================================================================
# Shared Utility Functions
# ==============================================================================

def normalize_cluster_id(cluster_id):
    if pd.isna(cluster_id):
        return None
    try:
        return int(cluster_id)
    except Exception:
        return str(cluster_id)


def normalize_phenotype_map_keys(phenotype_map: dict | None) -> dict:
    """Normalize persisted cluster-map keys after JSON round-trips.

    JSON object keys are always strings, while the heatmap cluster order is
    normally made from integer cluster IDs.  Keeping this conversion in one
    place prevents saved assignments such as ``{"12": "CD8"}`` from being
    missed by lookups using the integer key ``12``.
    """
    if not isinstance(phenotype_map, dict):
        return {}
    normalized = {}
    for cluster_id, phenotype in phenotype_map.items():
        normalized_id = normalize_cluster_id(cluster_id)
        if normalized_id is not None:
            normalized[normalized_id] = phenotype
    return normalized


def apply_heatmap_assignment_event(
    component_event: dict | None,
    phenotype_map: dict | None,
    *,
    previous_event_id=None,
) -> tuple[dict, list, str, object] | None:
    """Apply one buffered heatmap assignment safely.

    The local Plotly component can be remounted when Streamlit reruns or when
    the browser is refreshed. Its small ``eventId`` counter is therefore only
    a de-duplication hint, not proof that an assignment has already reached
    Python. We apply a valid payload whenever at least one requested cluster
    still has a different value. A payload that already matches the map is
    ignored regardless of its event ID, which prevents rerun loops when a
    remounted browser component emits a new counter for the same assignment.

    Returns ``(updated_map, selected_clusters, phenotype, event_id)`` when the
    session state should be updated, otherwise ``None``.
    """
    if not isinstance(component_event, dict) or component_event.get("action") != "apply":
        return None

    selected_clusters = sorted(
        {
            normalize_cluster_id(cluster_id)
            for cluster_id in component_event.get("selectedClusters", [])
            if normalize_cluster_id(cluster_id) is not None
        },
        key=cluster_sort_key,
    )
    if not selected_clusters:
        return None

    phenotype = str(component_event.get("phenotype", "Unassigned")).strip()
    if not phenotype:
        return None

    normalized_map = normalize_phenotype_map_keys(phenotype_map)
    requested_values_already_present = all(
        normalized_map.get(cluster_id) == phenotype
        for cluster_id in selected_clusters
    )
    event_id = component_event.get("eventId")
    event_token = (
        event_id
        if event_id is not None
        else f"{phenotype}|{','.join(str(cluster_id) for cluster_id in selected_clusters)}"
    )
    if requested_values_already_present:
        return None

    for cluster_id in selected_clusters:
        normalized_map[cluster_id] = phenotype
    return normalized_map, selected_clusters, phenotype, event_token


_HEATMAP_CONSUMED_EVENT_KEYS: set[str] = set()


def consume_heatmap_assignment_event(
    component_event: dict | None,
    *,
    scope: str,
) -> bool:
    """Return ``True`` only the first time an apply payload is seen.

    Streamlit custom components retain their last value while the Python
    script reruns. The browser therefore legitimately sends the same payload
    back on the rerun used to redraw the green assignment frames. This
    process-local guard consumes that payload once and prevents an accidental
    ``st.rerun()`` loop if session-state hydration changes the widget counter.
    The scope includes the method/project/run identifier at the call site, so
    a new clustering run can use the same browser event ID safely.
    """
    if not isinstance(component_event, dict) or component_event.get("action") != "apply":
        return False

    selected_clusters = sorted(
        {
            normalize_cluster_id(cluster_id)
            for cluster_id in component_event.get("selectedClusters", [])
            if normalize_cluster_id(cluster_id) is not None
        },
        key=cluster_sort_key,
    )
    phenotype = str(component_event.get("phenotype", "Unassigned")).strip()
    if not selected_clusters or not phenotype:
        return False

    event_id = component_event.get("eventId")
    event_token = event_id if event_id is not None else "no-id"
    key = json.dumps(
        [str(scope), str(event_token), phenotype, selected_clusters],
        separators=(",", ":"),
    )
    if key in _HEATMAP_CONSUMED_EVENT_KEYS:
        return False

    _HEATMAP_CONSUMED_EVENT_KEYS.add(key)
    # Bound this defensive process-local cache for long-running GUI sessions.
    if len(_HEATMAP_CONSUMED_EVENT_KEYS) > 2048:
        _HEATMAP_CONSUMED_EVENT_KEYS.clear()
        _HEATMAP_CONSUMED_EVENT_KEYS.add(key)
    return True


def get_heatmap_assignment_scope(method_key: str, run_key: str = "no-run") -> str:
    """Return an event scope unique to this Streamlit browser session.

    The frontend event counter starts at one whenever its iframe is recreated.
    Project/run-only scopes can therefore collide with an event consumed in a
    previous browser session and incorrectly discard a valid assignment.
    """
    session_token = st.session_state.get("_tab7_heatmap_session_token")
    if not session_token:
        session_token = uuid.uuid4().hex
        st.session_state["_tab7_heatmap_session_token"] = session_token
    project_root = st.session_state.get("active_project_root", "")
    return f"{method_key}::{project_root}::{run_key}::{session_token}"


TAB7_INTERNAL_CELL_ID_COL = "__tab7_internal_cell_id"


def normalize_optional_column_name(column_name):
    if column_name is None:
        return None
    text = str(column_name).strip()
    return text or None


def resolve_saved_column_default(
    saved_column: str | None,
    available_columns: list[str],
    fallback: str = "tissue",
) -> str:
    """Return a valid saved column before considering a conventional fallback."""
    columns = [str(column) for column in available_columns]
    normalized_saved = normalize_optional_column_name(saved_column)
    if normalized_saved in columns:
        return normalized_saved
    if fallback in columns:
        return fallback
    if columns:
        return columns[0]
    return fallback


def ensure_cell_id_anchor(
    df: pd.DataFrame,
    cell_id_col: str | None,
) -> tuple[pd.DataFrame, str, bool]:
    resolved_col = normalize_optional_column_name(cell_id_col)
    if resolved_col and resolved_col in df.columns:
        return df, resolved_col, False

    anchored_df = df.copy()
    if TAB7_INTERNAL_CELL_ID_COL not in anchored_df.columns:
        anchored_df[TAB7_INTERNAL_CELL_ID_COL] = np.arange(len(anchored_df), dtype=np.int64)
    return anchored_df, TAB7_INTERNAL_CELL_ID_COL, True


def get_cell_id_display_value(row: pd.Series, cell_id_col: str | None):
    resolved_col = normalize_optional_column_name(cell_id_col)
    if resolved_col and resolved_col in row.index:
        value = row.get(resolved_col)
        if pd.notna(value):
            return value

    if TAB7_INTERNAL_CELL_ID_COL in row.index:
        value = row.get(TAB7_INTERNAL_CELL_ID_COL)
        if pd.notna(value):
            return value

    for alt_col in ("cell_id", "CellID", "Cell_ID", "CellId", "id"):
        if alt_col in row.index:
            value = row.get(alt_col)
            if pd.notna(value):
                return value

    return "n/a"


def cluster_sort_key(cluster_id):
    normalized_cluster_id = normalize_cluster_id(cluster_id)
    if normalized_cluster_id is None:
        return (2, "")
    if isinstance(normalized_cluster_id, int):
        return (0, normalized_cluster_id)
    return (1, str(normalized_cluster_id))


_NON_PORTABLE_SETTINGS_KEY_SUFFIXES: tuple[str, ...] = (
    "_input_dir",
    "_input_file",
    "_infile_select",
    "_use_global",
    "_last_file",
    "_loaded_path",
    "_loaded_source_path",
    "_run_input_path",
)

_NON_PORTABLE_SETTINGS_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "input_dir",
        "input_file",
        "use_global",
        "last_file",
        "loaded_path",
        "loaded_source_path",
        "run_input_path",
        "source_file",
        "source_path",
        "input_path",
        "full_input_path",
        "csv_path",
        "unified_csv",
        "unified_csv_path",
    }
)


def _is_nonportable_settings_key(key_text: str) -> bool:
    return str(key_text).endswith(_NON_PORTABLE_SETTINGS_KEY_SUFFIXES)


def _sanitize_portable_settings_payload(value):
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in _NON_PORTABLE_SETTINGS_FIELD_NAMES or _is_nonportable_settings_key(key_text):
                continue
            sanitized[key_text] = _sanitize_portable_settings_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_portable_settings_payload(item) for item in value]
    return value


def _sanitize_cache_file_text(name: str, content: str) -> str:
    if not str(name).lower().endswith(".json"):
        return content

    try:
        parsed = json.loads(content)
    except Exception:
        return content

    sanitized = _sanitize_portable_settings_payload(parsed)
    try:
        return json.dumps(sanitized, indent=2)
    except Exception:
        return content


def create_session_temp_parquet_path(project_root: Path, prefix: str) -> Path:
    temp_dir = project_root / "outputs" / "phenotyping" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix)).strip("._") or "tab7_temp"
    return temp_dir / f"{safe_prefix}_{os.getpid()}_{time.time_ns()}.parquet"


def write_dataframe_parquet_atomic(df: pd.DataFrame, path: Path | str) -> Path:
    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = parquet_path.parent / f".{parquet_path.stem}.{os.getpid()}.{time.time_ns()}.tmp.parquet"
    try:
        df.to_parquet(temp_path)
        os.replace(temp_path, parquet_path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
    return parquet_path


def read_dataframe_parquet_safe(
    path: Path | str,
    *,
    session_state_key: str | None = None,
    label: str = "results",
) -> pd.DataFrame | None:
    parquet_path = Path(path)
    try:
        return pd.read_parquet(parquet_path)
    except Exception as exc:
        if session_state_key and session_state_key in st.session_state:
            del st.session_state[session_state_key]
        try:
            if parquet_path.exists():
                parquet_path.unlink()
        except Exception:
            pass
        logger.warning(f"Failed to read cached {label} parquet '{parquet_path}': {exc}")
        st.warning(get_data_access_guidance(parquet_path, exc, f"cached {label} parquet"))
        return None


def get_data_access_guidance(path: Path | str, error: BaseException, data_label: str) -> str:
    """Return a short, actionable recovery message for a data-access failure."""
    target = Path(path)
    if isinstance(error, FileNotFoundError):
        return f"The {data_label} is missing: {target.name}. Rerun the previous step or choose an available result file."
    if isinstance(error, PermissionError):
        return f"SPACEMAP cannot read the {data_label}: {target.name}. Check folder permissions and close programs that may be locking the file."
    return (
        f"Could not load the {data_label}: {target.name}. It may be incomplete after an interrupted run or use an unsupported format. "
        "The cached path was cleared; rerun or reload the result."
    )


def get_cell_id_column_help_text() -> str:
    return (
        "Column used to anchor cluster assignments, saved phenotypes, and exports back to the original rows. "
        "Choose a value that is unique per cell when possible. If left blank, the tab creates a temporary internal row-order anchor for the current session."
    )


def get_batch_id_column_help_text() -> str:
    return (
        "Optional grouping column for batch-wise normalization. "
        "When selected and present, the normalization pipeline runs separately within each batch; if left blank, normalization is global across the whole file. "
        "Use batch-wise normalization when batches still show technical shifts. If the markers are already well batch-corrected, global normalization is often the safer choice."
    )


def get_enable_row_filter_help_text(stage: str = "normalization and clustering") -> str:
    return (
        "Turn on metadata-based row filtering before "
        f"{stage}. "
        "Only matching rows continue downstream, so this changes the cells that are normalized, clustered, summarized, and exported."
    )


def get_filter_column_help_text(stage: str = "normalization and clustering") -> str:
    return (
        "Metadata column used when row filtering is enabled. "
        f"Only rows matching the chosen value continue to {stage}. "
        "Typical uses are analyzing one tissue, ROI, region, or sample subset at a time."
    )


def get_filter_value_help_text() -> str:
    return (
        "Only rows whose Filter Column matches this value are kept. "
        "For numeric columns the code compares numerically; for non-numeric columns it compares exact text. "
        "If you do not want to filter, turn the filter off instead of leaving this box empty."
    )


def get_marker_selection_help_text(method_name: str) -> str:
    return (
        f"Markers used to build the {method_name} feature space. "
        "The clustering run only sees the columns selected here. "
        "The UI allows mixing corrected and non-corrected markers, but avoid including both versions of the same marker unless you intentionally want both representations in the model."
    )


def get_output_column_help_text(result_label: str = "phenotype labels") -> str:
    return (
        f"Column name that will store the final {result_label} in saved and exported tables. "
        "Use a new name if you want to keep results from multiple runs side by side in the same CSV."
    )


def get_cluster_output_column_help_text(result_label: str) -> str:
    return (
        f"Column name used to store the {result_label} produced by this run. "
        "Change it only when you want to keep multiple clustering outputs side by side in the same table."
    )


def get_random_seed_help_text(purpose: str) -> str:
    return (
        "Reproducibility seed for stochastic steps in this tab. "
        f"It affects {purpose}. "
        "Keeping it fixed makes reruns easier to compare."
    )


def get_normalization_flag_help_text(flag: str) -> str:
    key = str(flag).strip().lower()
    if key == "z1":
        return (
            "Marker-wise z-score. Each selected marker column is centered and scaled across cells, or within each batch if a batch column is selected. "
            "This is usually the safest first transform because it puts markers on a comparable scale while preserving between-cell differences."
        )
    if key == "z2":
        return (
            "Cell-wise z-score applied after any enabled Z1 step. Each cell is re-centered and scaled across its own selected markers. "
            "This emphasizes within-cell marker pattern and removes much of the absolute intensity information, so use it only when that is the goal."
        )
    if key == "cdf":
        return (
            "CDF or percentile transform applied after earlier enabled steps in the shared normalization pipeline. "
            "It compresses outliers and makes the analysis less sensitive to a few very bright cells, but the values become rank-like rather than directly interpretable z-scores."
        )
    if key == "nlog":
        return (
            "Final NLog step in the shared notebook-synchronized pipeline. In the main path used by these tabs it is a negative-log stretch applied after the earlier enabled transforms, not just a raw log on the original intensities. "
            "It can increase separation near the high end, but it also changes the scale substantially."
        )
    return "Normalization step used in the shared tab pipeline."


def get_low_expression_filter_help_text(stage_label: str) -> str:
    return (
        f"Screen cells with globally low signal before the main {stage_label}. "
        "In these tabs the cells are not discarded: they are preserved in the results as a separate low-expression bucket so they can still be counted, named, and exported."
    )


def clean_heatmap_marker_name(marker_name):
    return (
        str(marker_name)
        .replace("_corr_norm", "")
        .replace("_norm", "")
        .replace("_corr", "")
    )


def build_cluster_marker_score_map(
    df_results: pd.DataFrame,
    norm_cols: list[str],
    cluster_col: str,
) -> dict:
    valid_norm_cols = [
        column
        for column in norm_cols
        if column in df_results.columns and pd.api.types.is_numeric_dtype(df_results[column])
    ]
    if cluster_col not in df_results.columns or not valid_norm_cols:
        return {}

    grouped = df_results.groupby(cluster_col)[valid_norm_cols].mean()
    if grouped.empty:
        return {}

    grouped_t = grouped.T.copy()
    grouped_t.index = grouped_t.index.map(clean_heatmap_marker_name)
    collapsed = grouped_t.groupby(level=0).mean().T

    cluster_marker_scores = {}
    for cluster_id, row in collapsed.iterrows():
        normalized_cluster_id = normalize_cluster_id(cluster_id)
        if normalized_cluster_id is None:
            continue
        marker_scores = {
            str(marker): float(score)
            for marker, score in row.items()
            if pd.notna(score)
        }
        cluster_marker_scores[normalized_cluster_id] = marker_scores
    return cluster_marker_scores


DEFAULT_SIGNATURE_TEMPLATE = """Fibroblast = [FN1] OR [Vimentin] OR [PDGFRa] OR [Podoplanin] OR [aSMA]; SUPPORT=[]; NOT=[]
IMC = [CD11b, S100A9, CD15] OR [CD11b, S100A9] OR [CD15, S100A9] OR [CD11b, CD15]; SUPPORT=[]; NOT=[]
DC = [CD11c, HLA-DR, CD45] OR [CD11c, HLA-DR] OR [CD11c, CD45]; SUPPORT=[]; NOT=[]
Macrophage = [CD68, CD163, CD45] OR [CD68, CD45] OR [CD68, HLA-DR] OR [CD163, HLA-DR] OR [CD163, CD45] OR [CD68, Vimentin] OR [CD163, Vimentin]; SUPPORT=[]; NOT=[]
Endothelial = [CD31]; SUPPORT=[]; NOT=[]
B = [CD20, HLA-DR, CD45] OR [CD20, HLA-DR] OR [CD20, CD45]; SUPPORT=[]; NOT=[]
Epithelial = [Pan-Cytokeratin] OR [E-cadherin]; SUPPORT=[]; NOT=[]
Plasma = [CD38]; SUPPORT=[]; NOT=[]
Treg = [FoxP3, CD4, CD3e, CD45] OR [FoxP3, CD4, CD3e] OR [FoxP3, CD4, CD45] OR [FoxP3, CD3e, CD45]; SUPPORT=[]; NOT=[]
CD8 = [CD8, CD45, CD3e] OR [CD8, CD45] OR [CD8, CD3e]; SUPPORT=[]; NOT=[]
CD4 = [CD4, CD45, CD3e] OR [CD4, CD45] OR [CD4, CD3e]; SUPPORT=[]; NOT=[]
"""


def render_phenotype_recommendation_panel(
    key_prefix: str,
    cluster_marker_scores: dict,
    seed_labels: dict | None = None,
    reference_matrix=None,
    reference_marker_names=None,
    reference_label_options: dict | None = None,
    cluster_labels=None,
    cluster_sizes: dict | None = None,
) -> dict | None:
    """Signature-driven phenotype suggestions for every cluster, with a review queue.

    Scores each cluster's mean marker profile against user-written signatures and
    proposes the best match. All clusters are scored in one pass, so the outcome
    does not depend on the order they are considered in. The table is then sorted
    by confidence ascending -- the margin between the best and second-best
    phenotype -- so the genuinely ambiguous clusters are reviewed first and the
    obvious ones can be accepted in bulk. That is what makes hundreds of clusters
    tractable without merging them away.

    Returns the accepted ``{cluster: phenotype}`` map when the user applies it,
    otherwise None. Nothing is written to the assignment until it is applied.
    """
    from spacemap.ph_accel_utils import (
        derive_signatures_from_reference,
        learn_phenotype_profiles,
        parse_phenotype_signatures,
        recommend_phenotypes,
        seed_labels_from_reference,
        signatures_to_text,
    )

    if not cluster_marker_scores:
        return None

    state_key = f"{key_prefix}_reco_table"
    sig_key = f"{key_prefix}_reco_signatures"
    open_key = f"{key_prefix}_reco_panel_open"

    # Streamlit expanders collapse on every rerun, which closes this panel each
    # time a control inside it is touched. Once the user has engaged with it --
    # chosen a reference, written signatures, or produced suggestions -- keep it
    # open so their place is not lost.
    st.session_state[open_key] = bool(
        st.session_state.get(open_key)
        or st.session_state.get(state_key) is not None
        or st.session_state.get(f"{key_prefix}_reco_refcol")
        or st.session_state.get(sig_key)
    )

    with st.expander("Assisted phenotype assignment", expanded=bool(st.session_state.get(open_key))):
        st.markdown(
            """
**The problem this solves.** Your run produced a lot of clusters. Naming every one by
eye takes days. This names them all in one go, so you only have to check the ones it
was unsure about.

**How it decides**

1. Each cluster gets averaged into a single expression fingerprint — one value per
   marker. A cluster of 400 cells and one of 400,000 take the same time to judge.
2. That fingerprint is compared against what each cell type looks like.
3. Each cluster gets whichever cell type it resembles most.
4. Clusters are then sorted by **how sure** the match was.

**How to use it.** The list below starts with the clusters it was *least* sure about.
Read down from the top and correct anything that looks wrong by typing over the cell
type. Once the calls start looking obviously right, you can stop — everything further
down was an easy call. Then press Apply.

**The `source` column says how each name was decided**

- `reference` — this cluster contains cells you labelled yourself, so your own labels
  were used. The most trustworthy.
- `profile` — its fingerprint matched a cell type learned from your annotated cells.
- `signature` — it matched a marker list you typed in below.
- `similarity` — nothing matched, so it copied the most similar cluster. Check these.

Nothing is written to your phenotype assignment until you press Apply.
"""
        )

        # Optional: draft the signatures from a column of manually annotated cells.
        if reference_label_options and reference_matrix is not None and reference_marker_names:
            ref_cols = [c for c in reference_label_options if c]
            if ref_cols:
                st.markdown("**Start from a manual reference (optional)**")
                st.caption(
                    "If you have a column where some cells were annotated by hand, the "
                    "signatures can be drafted from it: for each phenotype it picks the "
                    "markers that best separate it from the others. Treat the result as a "
                    "first draft — a marker shared by several phenotypes (CD45 across all "
                    "immune cells) separates them from everything else but not from each "
                    "other, so tightening those lines by hand usually helps. Annotated "
                    "cells are also used directly to name the clusters they fall in, which "
                    "is where most of the accuracy comes from."
                )
                try:
                    rc1, rc2 = st.columns([2, 1], vertical_alignment="bottom")
                except TypeError:  # older Streamlit without vertical_alignment
                    rc1, rc2 = st.columns([2, 1])
                    rc2.write("")
                prior = _selected_reference_columns(key_prefix, ref_cols)
                ref_selected = rc1.multiselect(
                    "Which column(s) hold your hand-labelled cells?", ref_cols,
                    default=prior or ([ref_cols[0]] if ref_cols else []),
                    key=f"{key_prefix}_reco_refcol",
                    help=(
                        "Columns where you assigned cell types yourself; unlabelled cells are "
                        "ignored. Pick several to pool separate annotation rounds — each can "
                        "contribute marker patterns the others never covered, and a cell type "
                        "missing from one can still be learned from another. Where a cell is "
                        "labelled in more than one, the column listed first wins."
                    ),
                )
                if len(ref_selected) > 1:
                    rc1.caption(f"Pooling {len(ref_selected)} columns, earlier ones take precedence.")
                if rc2.button("Fill this in from my labels", key=f"{key_prefix}_reco_derive"):
                    try:
                        merged = _combine_reference_columns(reference_label_options, ref_selected)
                        derived = derive_signatures_from_reference(
                            reference_matrix, reference_marker_names, merged,
                        ) if merged is not None else {}
                        if derived:
                            st.session_state[sig_key] = signatures_to_text(derived)
                            st.success(
                                f"Drafted {len(derived)} signature(s) from "
                                f"{', '.join(ref_selected)}. Review and edit below."
                            )
                            st.rerun()
                        else:
                            st.warning(
                                "Those column(s) did not yield signatures — at least two cell "
                                "types with enough annotated cells each are needed."
                            )
                    except Exception as exc:
                        st.error(f"Could not draft signatures: {exc}")

        if st.button("Use example marker rules", key=f"{key_prefix}_reco_example"):
            st.session_state[sig_key] = DEFAULT_SIGNATURE_TEMPLATE
        signature_text = st.text_area(
            "Marker list for each cell type",
            value=st.session_state.get(sig_key, DEFAULT_SIGNATURE_TEMPLATE),
            height=220,
            key=sig_key,
            help=(
                "One cell type per line: `Name = [A, B] OR [C]; SUPPORT=[D]; NOT=[E]`. "
                "Markers within each bracket belong together; OR allows alternative groups. "
                "SUPPORT is optional positive evidence; NOT penalizes high excluded markers.\n\n"
                "Groups use their weakest marker's expression; the strongest alternative wins. "
                "NOT subtracts the strongest positive excluded signal. SUPPORT adds 25% of "
                "its mean positive expression to an already positive score. These score cluster "
                "expression profiles, not per-cell gates.\n\n"
                "The older `Name: A, B, C-` format still works. A trailing '-' or leading '!' "
                "means a low marker. Lines starting with '#' are ignored."
            ),
        )
        signatures, signature_problems = parse_phenotype_signatures(signature_text, return_problems=True)
        if signature_problems:
            st.warning("Some marker rules were skipped:\n\n" + "\n\n".join(signature_problems))
        st.caption(f"{len(signatures)} signature(s) parsed: {', '.join(sorted(signatures))}")

        # How a cluster is matched to a phenotype. Profile matching needs a
        # reference to learn from; signatures are the only option without one.
        can_learn = bool(reference_label_options) and reference_matrix is not None and reference_marker_names
        mode_options = (["Learn from my annotated cells (recommended)", "Use my marker list"]
                        if can_learn else ["Use my marker list"])
        scoring_mode_label = st.radio(
            "How should clusters be named?",
            mode_options,
            horizontal=True,
            key=f"{key_prefix}_reco_mode",
            help=(
                "**Learn from my annotated cells** — looks at the cells you labelled by hand, "
                "works out what each cell type looks like across *all* your markers, then finds "
                "which one each cluster resembles. More accurate, because it uses the whole "
                "marker panel instead of the handful you list below.\n\n"
                "**Use my marker list** — goes only by the lines you type below. Use this when "
                "you have no annotated cells at all."
            ),
        )
        use_profile = scoring_mode_label.startswith("Learn from my annotated cells")
        if not signatures and not use_profile:
            st.info("Add at least one valid marker rule to get suggestions, or use the example rules.")
            return None
        if use_profile:
            # Spell out which marker-list lines still do anything in this mode, rather
            # than leaving the user to infer it.
            covered = set()
            chosen_cols = _selected_reference_columns(key_prefix, list(reference_label_options or {}))
            merged_labels = _combine_reference_columns(reference_label_options, chosen_cols)
            if merged_labels is not None:
                values = merged_labels.dropna().unique()
                covered = {
                    str(v) for v in values
                    if str(v).lower() not in {"others", "unassigned", "unclassified", "nan", "none"}
                }
            if covered:
                extra = sorted(set(signatures) - covered)
                overlap = sorted(set(signatures) & covered)
                message = (
                    f"Cell types taken from your labelled cells ({len(covered)}): "
                    f"{', '.join(sorted(covered))}."
                )
                if extra:
                    message += (
                        f" The marker list below adds {len(extra)} your labels do not cover — "
                        f"**{', '.join(extra)}** — so those lines are used."
                    )
                else:
                    message += " The marker list below adds nothing new, so none of it is used."
                if overlap:
                    message += (
                        f" Lines for {', '.join(overlap)} are ignored, because your labelled "
                        "cells already define those."
                    )
                st.caption(message)
            else:
                st.caption(
                    "Pick the column with your hand-labelled cells above. The marker list "
                    "below is then only used for cell types those labels do not cover."
                )

        if use_profile:
            n_sub = st.slider(
                "How many marker patterns to learn per cell type",
                1, 5, int(st.session_state.get(f"{key_prefix}_reco_nsub", 5)),
                key=f"{key_prefix}_reco_nsub",
                help=(
                    "A cell type is often not one uniform population — stromal cells may be "
                    "aSMA-high in one place and FN1-high in another, and a single average "
                    "describes neither. This learns several alternative patterns per cell "
                    "type, and a cluster matching any one of them counts.\n\n"
                    "Set 1 for a single average. A cell type only gets as many patterns as it "
                    "has cells to support, so small populations are never over-split. On a "
                    "2.8M-cell test, 5 scored about 1-2 points higher than 1."
                ),
            )
        else:
            n_sub = 1

        col_a, col_b, col_c = st.columns([1, 1, 1])
        use_seed = col_a.checkbox(
            "Keep the names I typed myself",
            value=True,
            key=f"{key_prefix}_reco_seed",
            help=(
                "Leave alone any cluster you have already named yourself, and only fill in "
                "the rest. Turn this off only if you want to start over."
            ),
        )
        use_reference_seed = col_b.checkbox(
            "Let my annotated cells name their own cluster",
            value=True,
            key=f"{key_prefix}_reco_refseed",
            help=(
                "If a cluster contains cells you labelled by hand, just use your own labels "
                "for that whole cluster. Example: a cluster holds 12 cells you annotated and "
                "11 of them say B cell, so the cluster is called B cell.\n\n"
                "This is the most reliable option because it uses your actual answers instead "
                "of working them out from markers. Turn it off if you want the naming to stay "
                "completely independent of your labels."
            ),
        )
        propagate = col_c.checkbox(
            "Name leftover clusters after their closest neighbour",
            value=True,
            key=f"{key_prefix}_reco_prop",
            help=(
                "If a cluster matches nothing at all, give it the same name as whichever "
                "named cluster it looks most like, so no cells are simply dropped. These are "
                "marked 'similarity' in the list so you can check them.\n\n"
                "Only applies when naming by marker list; when learning from annotated cells "
                "every cluster already gets a match."
            ),
        )

        st.session_state[open_key] = True
        if st.button("Name all clusters now", key=f"{key_prefix}_reco_run"):
            profiles = pd.DataFrame(cluster_marker_scores).T.sort_index()
            profiles = profiles.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            seeds = dict(seed_labels or {}) if use_seed else {}
            if use_reference_seed and reference_label_options and cluster_labels is not None:
                merged_seed = _combine_reference_columns(
                    reference_label_options,
                    _selected_reference_columns(key_prefix, list(reference_label_options)),
                )
                if merged_seed is not None:
                    try:
                        ref_seeds, stats = seed_labels_from_reference(cluster_labels, merged_seed)
                        # Hand assignments still win over the reference majority.
                        merged = {normalize_cluster_id(k): v for k, v in ref_seeds.items()}
                        merged.update(seeds)
                        seeds = merged
                        st.session_state[f"{key_prefix}_reco_seedstats"] = stats
                    except Exception as exc:
                        st.warning(f"Could not name clusters from your labels: {exc}")

            learned = None
            if use_profile:
                merged_learn = _combine_reference_columns(
                    reference_label_options or {},
                    _selected_reference_columns(key_prefix, list(reference_label_options or {})),
                )
                if merged_learn is not None:
                    learned = learn_phenotype_profiles(
                        reference_matrix, reference_marker_names, merged_learn,
                        n_subprofiles=int(n_sub),
                    )
                    st.session_state[f"{key_prefix}_reco_learned"] = {
                        g: v.get("details", []) for g, v in (learned or {}).items()
                    }
                    if not learned:
                        st.warning(
                            "Could not learn patterns from those column(s) — at least 15 "
                            "annotated cells per cell type are needed. Falling back to the "
                            "marker list."
                        )

            table = recommend_phenotypes(
                profiles, signatures, seed_labels=seeds or None, propagate=propagate,
                phenotype_profiles=learned,
                profile_marker_names=reference_marker_names if learned else None,
                prototype_markers=st.session_state.get(f"{key_prefix}_reco_patternmarkers"),
            )
            st.session_state[state_key] = table
            st.rerun()

        # Let the user inspect what was learned and switch off patterns that look
        # like artefacts rather than biology.
        details = st.session_state.get(f"{key_prefix}_reco_learned")
        if use_profile and details:
            patterns_open_key = f"{key_prefix}_patterns_panel_open"
            st.session_state[patterns_open_key] = bool(
                st.session_state.get(patterns_open_key)
                or st.session_state.get(f"{key_prefix}_reco_patternmarkers")
            )
            with st.expander("What each cell type looks like (edit to correct it)",
                             expanded=bool(st.session_state.get(patterns_open_key))):
                st.caption(
                    "Each row is one marker pattern found among your labelled cells, with the "
                    "markers that define it and how many cells follow it. **Edit the markers "
                    "column** to correct a pattern:\n\n"
                    "- **Delete a marker** that does not belong — signal bleeding in from a "
                    "touching cell, such as FoxP3 in cells you called epithelial. That pattern "
                    "then ignores that marker, so genuinely epithelial clusters without the "
                    "neighbour's signal are no longer penalised. Every other marker still "
                    "counts, including the ones not shown in the row.\n"
                    "- **Clear the whole row** to switch that pattern off completely.\n\n"
                    "Read the whole row before deleting. A pattern led by FoxP3 that still "
                    "carries Pan-Cytokeratin and E-cadherin is a real epithelial cell with "
                    "spill-over; one with foreign markers and none of its own is a mislabel. "
                    "Every cell type always keeps at least one pattern."
                )

                rows = []
                for name in sorted(details):
                    for entry in details[name]:
                        rows.append({
                            "cell type": name,
                            "pattern": int(entry["index"]) + 1,
                            "cells": int(entry["n_cells"]),
                            "markers": ", ".join(
                                f"{m} {v:+.2f}" for m, v in (entry.get("top_markers") or [])
                            ),
                        })
                # Rescanning re-reads the marker patterns from the annotated cells and
                # throws away every edit, which is the way back if a marker was deleted
                # by mistake or the reference columns have changed.
                rescan_col, note_col = st.columns([1, 3])
                if rescan_col.button(
                    "Rescan patterns (reset edits)", key=f"{key_prefix}_pattern_rescan",
                    help=(
                        "Relearn the patterns from your labelled cells and discard all edits, "
                        "restoring every marker. Use after changing the reference columns or "
                        "the number of patterns, or to undo a deletion."
                    ),
                ):
                    st.session_state.pop(f"{key_prefix}_reco_patternmarkers", None)
                    st.session_state.pop(f"{key_prefix}_pattern_table", None)
                    st.session_state.pop(f"{key_prefix}_pattern_table__stale_checked", None)
                    st.session_state[f"{key_prefix}_pattern_rescan_pending"] = True
                    st.rerun()
                edit_count = len(st.session_state.get(f"{key_prefix}_reco_patternmarkers") or {})
                note_col.caption(
                    f"{edit_count} pattern(s) edited." if edit_count
                    else "No edits — patterns are exactly as learned from your labelled cells."
                )

                table_key = f"{key_prefix}_pattern_table"
                _clear_stale_widget_state(table_key)
                original = pd.DataFrame(rows)
                if original.empty:
                    st.info("No patterns learned yet — press 'Name all clusters now' first.")
                else:
                    edited = st.data_editor(
                        original,
                        width="stretch",
                        hide_index=True,
                        disabled=["cell type", "pattern", "cells"],
                        column_config={
                            "markers": st.column_config.TextColumn(
                                "markers (edit me)",
                                help="Comma-separated. Delete any marker that does not belong; "
                                     "empty the row to switch the pattern off.",
                                width="large",
                            ),
                        },
                        key=table_key,
                    )

                    # Work out what survived each row: names only, numbers ignored.
                    def _names(text):
                        out = []
                        for token in str(text or "").split(","):
                            token = token.strip()
                            if not token:
                                continue
                            out.append(token.split()[0].strip())
                        return [t for t in out if t]

                    # The row shows only a pattern's strongest markers, but the pattern
                    # itself covers every marker. So record what the user DELETED, not
                    # what is left visible -- keeping only the visible ones would
                    # silently discard all the markers that were never displayed.
                    prototype_markers: dict = {}
                    n_dropped = n_off = 0
                    for (_, before), (_, after) in zip(original.iterrows(), edited.iterrows()):
                        kept = set(_names(after["markers"]))
                        was = set(_names(before["markers"]))
                        name = str(before["cell type"])
                        index = str(int(before["pattern"]) - 1)
                        if not kept:
                            prototype_markers.setdefault(name, {})[index] = None  # switched off
                            n_off += 1
                        elif kept != was:
                            removed = sorted(was - kept)
                            prototype_markers.setdefault(name, {})[index] = removed
                            n_dropped += len(removed)
                    st.session_state[f"{key_prefix}_reco_patternmarkers"] = prototype_markers

                    if n_dropped or n_off:
                        st.info(
                            f"{n_dropped} marker(s) removed, {n_off} pattern(s) switched off. "
                            "Press **Name all clusters now** again to apply."
                        )

        if st.session_state.pop(f"{key_prefix}_pattern_rescan_pending", False):
            st.info(
                "Patterns reset to how they were learned. Press **Name all clusters now** "
                "to rebuild the suggestions from them."
            )

        stats = st.session_state.get(f"{key_prefix}_reco_seedstats")
        if stats and use_reference_seed:
            st.caption(
                f"Annotated cells: {stats['annotated_cells_used']:,} of them fall in "
                f"{stats['clusters_with_annotations']} clusters — "
                f"{stats['clusters_named']} named directly, "
                f"{stats['clusters_too_mixed']} left to the signatures (labels disagreed)."
            )

        table = _as_dataframe(
            st.session_state.get(state_key),
            columns=["phenotype", "confidence", "certainty", "source", "runner_up", "evidence"],
        )
        if table.empty:
            return None
        if "cluster" in table.columns:
            table = table.set_index("cluster")
        # Guarantee the index carries the name, so reset_index() below always
        # produces a "cluster" column even for a cache that lost its index name.
        table.index.name = "cluster"

        # A signature naming markers that are not in the data scores nothing, which
        # would otherwise show up only as every cluster coming back unmatched.
        unresolved = list(table.attrs.get("unresolved_markers", []))
        scored = list(table.attrs.get("scored_phenotypes", []))
        named_by_profile = int((table["source"] == "profile").sum()) if "source" in table.columns else 0
        # In "learn from my annotated cells" mode the marker list is optional, so a
        # signature that scored nothing is not an error worth alarming the user about.
        signatures_matter = not (use_profile and named_by_profile)
        if unresolved and signatures_matter:
            st.warning(
                f"These markers are not in the clustered data: {', '.join(unresolved)}. "
                "Incomplete required groups are skipped; a missing NOT marker skips that cell type. "
                "Missing SUPPORT markers are ignored. Check the spelling against the "
                "heatmap rows — suffixes such as `_corr` or `_norm` are handled automatically, "
                "so plain marker names are fine."
            )
        if not scored and signatures_matter:
            st.error(
                "No cell type could be matched: none of the markers in the list above exist in "
                "this run. Check the names against the heatmap rows."
            )
        elif not scored and use_profile and named_by_profile:
            st.caption(
                f"All {named_by_profile} clusters were named from your labelled cells. The "
                "marker list was not needed."
            )

        view = table.reset_index()
        view["confidence"] = view["confidence"].replace([np.inf], 999.0)
        if "certainty" not in view.columns:
            view["certainty"] = 0.0

        cutoff = st.slider(
            "Leave the least confident clusters as 'Others'",
            0.0, 1.0,
            float(st.session_state.get(f"{key_prefix}_reco_cutoff", 0.0)),
            0.05,
            key=f"{key_prefix}_reco_cutoff",
            help=(
                "'How sure' runs from 0 to 1. A cluster named from your own labelled cells "
                "scores 1.00. Everything else is scored on how clearly it beat the runner-up: "
                "a clear winner scores high, a near-tie scores low.\n\n"
                "Leave at 0 to keep every suggestion. Raise it and the shakiest clusters are "
                "called 'Others' instead of being guessed at. Around 0.3-0.4 drops the "
                "genuinely doubtful ones; going much higher throws away cells you could "
                "have kept. The caption tells you how many cells each setting costs."
            ),
        )
        below = view["certainty"] < cutoff
        if cutoff > 0 and below.any():
            view.loc[below, "phenotype"] = "Others"
            view.loc[below, "source"] = "below cutoff"

        view = view.sort_values("certainty")          # least certain first
        counts = view["source"].value_counts().to_dict()
        n_cells_note = ""
        if cluster_sizes is not None and below.any():
            dropped = sum(int(cluster_sizes.get(c, 0)) for c in view.loc[below, "cluster"])
            total = sum(int(v) for v in cluster_sizes.values()) or 1
            n_cells_note = f" That is {dropped:,} cells ({dropped/total*100:.1f}%) sent to Others."
        st.caption(
            f"{len(view)} clusters — "
            + ", ".join(f"{v} {k}" for k, v in counts.items())
            + f".{n_cells_note} The least certain are listed first, so start at the top."
        )

        editor_key = f"{key_prefix}_reco_editor"
        _clear_stale_widget_state(editor_key)
        edited = st.data_editor(
            view[["cluster", "phenotype", "certainty", "source", "runner_up", "evidence"]],
            width="stretch",
            hide_index=True,
            disabled=["cluster", "certainty", "source", "runner_up", "evidence"],
            column_config={
                "certainty": st.column_config.ProgressColumn(
                    "how sure", min_value=0.0, max_value=1.0, format="%.2f",
                    help="0 to 1. Higher means the cluster clearly matched one cell type rather than being a close call.",
                )
            },
            key=editor_key,
        )

        apply_col, clear_col = st.columns([1, 3])
        if apply_col.button("Apply to phenotype assignment", type="primary", key=f"{key_prefix}_reco_apply"):
            accepted = {
                normalize_cluster_id(row["cluster"]): str(row["phenotype"])
                for _, row in edited.iterrows()
                if pd.notna(row["phenotype"]) and str(row["phenotype"]).strip()
            }
            st.session_state.pop(state_key, None)
            return accepted
        clear_col.caption("Edit any phenotype cell above before applying; blank rows are skipped.")
    return None


def _selected_reference_columns(key_prefix: str, available: list) -> list:
    """Reference columns chosen for learning, tolerating the older single-column value."""
    stored = st.session_state.get(f"{key_prefix}_reco_refcol")
    if isinstance(stored, str):
        stored = [stored]
    return [c for c in (stored or []) if c in available]


def _combine_reference_columns(options: dict, columns: list):
    """Merge several annotation columns into one label per cell.

    Cells annotated in more than one column keep the label from the column listed
    first, so the order chosen decides precedence. Combining is worth doing when
    separate annotation rounds cover different regions or cell types: each adds
    marker patterns the others never saw, and a cell type missing from one column
    can still be learned from another.
    """
    if not columns:
        return None
    combined = pd.Series(options[columns[0]]).astype("object").copy()
    for extra in columns[1:]:
        other = pd.Series(options[extra]).astype("object")
        combined = combined.where(combined.notna(), other)
    return combined


def _clear_stale_widget_state(key: str) -> None:
    """Drop a value restored into a widget key before that widget exists.

    Streamlit refuses to create a data_editor whose key already holds a value,
    and project restore can put one there from an earlier session. Clearing it
    once, before the widget is first built, keeps those projects loadable; the
    widget then owns the key for the rest of the session.
    """
    guard = f"{key}__stale_checked"
    if guard not in st.session_state:
        st.session_state.pop(key, None)
        st.session_state[guard] = True


def _as_dataframe(value, columns=None):
    """Coerce a cached value back into a DataFrame.

    Anything parked in ``st.session_state`` may be round-tripped through the
    project's JSON persistence, which turns a DataFrame into a list of records.
    Restoring it here keeps these panels working after a save/reload cycle
    instead of failing on the first column selection.
    """
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return pd.DataFrame(columns=list(columns or []))
    try:
        frame = pd.DataFrame(value)
    except Exception:
        return pd.DataFrame(columns=list(columns or []))
    for column in (columns or []):
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def render_reference_evaluation_panel(
    key_prefix: str,
    cluster_labels,
    phenotype_map: dict,
    reference_label_options: dict,
    cluster_marker_scores: dict | None = None,
    reference_matrix=None,
    reference_marker_names=None,
    seeded_from_reference: bool = False,
) -> None:
    """Report how well the current assignment recovers manually annotated cells.

    Offers two readings, because they answer different questions:

    * scoring the assignment as it stands, which is only meaningful when those
      annotations were not used to produce it; and
    * a held-out estimate, where signatures are rebuilt from part of the
      annotations and scored on the rest, which stays honest either way.
    """
    from spacemap.ph_accel_utils import (
        cross_validate_reference_assignment,
        score_against_reference,
    )

    if not reference_label_options:
        return

    eval_open_key = f"{key_prefix}_eval_panel_open"
    st.session_state[eval_open_key] = bool(
        st.session_state.get(eval_open_key)
        or st.session_state.get(f"{key_prefix}_eval_result") is not None
    )
    with st.expander("How accurate was the naming? (check against your own labels)",
                     expanded=bool(st.session_state.get(eval_open_key))):
        st.markdown(
            """
The cells you labelled by hand can tell you how well the automatic naming did — but
only if they were kept out of doing the naming. If a cluster was *named from* its
annotated cells, checking it against those same cells proves nothing.

**Fair test (recommended).** Works like an exam. Some of your labelled cells are hidden,
the naming is redone without them, and then those hidden cells are checked. Because
nothing saw the answers first, the score tells you how well it does on cells nobody has
labelled — which is the rest of your slide.

**Quick test.** Scores the naming exactly as it stands right now. Only meaningful if your
labelled cells played no part in producing it.

**The score.** An average across your cell types, from 0 (all wrong) to 1 (all correct).
Rare cell types count as much as common ones, so getting only the abundant types right
will not flatter the score. Cells you labelled *Others* are listed separately, since that
records you being unsure rather than a cell type to find.
annotator being unsure rather than a phenotype to be predicted.
"""
        )

        columns = [c for c in reference_label_options if c]
        if not columns:
            return
        eval_col = st.selectbox(
            "Which column holds your hand-labelled cells?", columns,
            key=f"{key_prefix}_eval_col",
            help="The column containing cell types you assigned yourself. Unlabelled cells are ignored.",
        )
        truth = reference_label_options[eval_col]

        can_cv = bool(
            cluster_marker_scores and reference_matrix is not None and reference_marker_names
        )
        c1, c2 = st.columns(2)
        run_cv = c1.button(
            "Fair test", key=f"{key_prefix}_eval_cv", disabled=not can_cv, type="primary",
            help=(
                "Hides some of your labelled cells, redoes the naming without them, then "
                "checks whether those hidden cells came out right. This is the number to "
                "trust and the one to report."
            ),
        )
        run_direct = c2.button(
            "Quick test", key=f"{key_prefix}_eval_direct",
            help=(
                "Scores the naming as it stands, using all your labelled cells. Faster, but "
                "if those cells helped name the clusters the score will look better than it is."
            ),
        )

        if seeded_from_reference:
            st.info(
                "Your labelled cells were used to name the clusters, so the Quick test would "
                "just be marking its own homework. Use the Fair test."
            )

        result, heading = None, ""
        if run_cv and can_cv:
            with st.spinner("Rebuilding signatures on held-out folds..."):
                profiles = pd.DataFrame(cluster_marker_scores).T.sort_index()
                profiles = profiles.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                use_seeding = st.session_state.get(f"{key_prefix}_reco_refseed", False)
                mode_label = st.session_state.get(f"{key_prefix}_reco_mode", "") or ""
                result = cross_validate_reference_assignment(
                    cluster_labels=cluster_labels, cluster_profiles=profiles,
                    matrix=reference_matrix, marker_names=reference_marker_names,
                    reference_labels=truth, use_seeding=bool(use_seeding),
                    scoring_mode=("profile" if mode_label.startswith("Learned profile") else "signature"),
                )
            heading = "Fair test (hidden cells)"
            st.session_state[f"{key_prefix}_eval_result"] = (heading, result)
        elif run_direct:
            result = score_against_reference(cluster_labels, truth, phenotype_map)
            heading = "Quick test (all cells used)"
            st.session_state[f"{key_prefix}_eval_result"] = (heading, result)
        else:
            cached = st.session_state.get(f"{key_prefix}_eval_result")
            if isinstance(cached, (list, tuple)) and len(cached) == 2:
                heading, result = cached
            elif cached:
                # Cache written in an older/persisted shape -- discard rather than crash.
                st.session_state.pop(f"{key_prefix}_eval_result", None)

        if result is None:
            if run_cv or run_direct:
                st.warning(
                    "Not enough annotated cells to score. At least ~30 cells across two or "
                    "more phenotypes are needed."
                )
            return

        st.markdown(f"**{heading}**")
        m1, m2, m3, m4 = st.columns(4)
        sd = result.get("macro_f1_sd")
        m1.metric("Overall score", f"{result['macro_f1']:.3f}",
                  delta=(f"±{sd:.3f} across folds" if sd is not None else None),
                  delta_color="off")
        m2.metric("Correct when it named one", f"{result['accuracy_of_assigned']:.3f}",
                  help="Of the cells it did give a cell type to, the share it got right.")
        m3.metric("Cells it named", f"{result['coverage']:.1%}",
                  help="Share of your labelled cells that got a cell type instead of being left as Others.")
        m4.metric("Cells checked", f"{result['n_scored']:,}")

        if result.get("used_seeding"):
            st.caption(
                "Both the cell-type fingerprints and the naming-from-your-labels step were "
                "redone with the hidden cells left out, matching how the naming actually runs."
            )
        elif heading.startswith("Fair test"):
            st.caption(
                "Only the cell-type fingerprints were relearned each time; naming clusters "
                "straight from your labels was switched off."
            )

        per = _as_dataframe(
            result.get("per_phenotype"),
            columns=["phenotype", "F1", "precision", "recall", "annotated_cells"],
        ).copy()
        if per.empty:
            st.info("No per-phenotype breakdown available for this result.")
            return
        for column in ("F1", "precision", "recall"):
            per[column] = pd.to_numeric(per[column], errors="coerce").round(3)
        st.dataframe(
            per, width="stretch", hide_index=True,
            column_config={
                "F1": st.column_config.ProgressColumn(
                    "F1", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )
        weak = per.loc[per["F1"].fillna(1.0) < 0.6, "phenotype"].astype(str).tolist()
        if weak:
            st.warning(
                "Hardest cell types here: " + ", ".join(weak)
                + ". These are being mixed up with something else. Look at which markers they "
                "share with the cell type they are confused with, and add that marker with a "
                "'-' after it to the marker list, so it counts against them."
            )
        if result.get("n_reference_others"):
            share = result.get("reference_others_given_a_phenotype")
            st.caption(
                f"{result['n_reference_others']:,} reference cells are labelled 'Others' and were "
                "not scored"
                + (f"; {share:.1%} of them were given a phenotype anyway." if share is not None else ".")
            )


def render_assisted_assignment(
    key_prefix: str,
    phenotype_map_state_key: str,
    available_phenotypes_state_key: str,
    cluster_marker_scores: dict,
    df_results: "pd.DataFrame | None" = None,
    marker_cols: "list[str] | None" = None,
    cluster_column: "str | None" = None,
) -> bool:
    """Render the assisted-assignment panel and apply whatever the user accepts.

    Wraps :func:`render_phenotype_recommendation_panel` with the wiring every
    method needs: existing assignments become seeds, any column of manual
    annotations in the results table can be used to draft signatures, and
    accepted suggestions are written back into the method's phenotype map.

    Returns True when an assignment was applied (the caller should rerun).
    """
    if not cluster_marker_scores:
        return False

    current_map = st.session_state.get(phenotype_map_state_key, {}) or {}
    seed_labels = {
        normalize_cluster_id(cluster): str(phenotype)
        for cluster, phenotype in current_map.items()
        if phenotype and str(phenotype) not in {"Unassigned", "nan", "None"}
    }

    # Any column holding manual phenotype labels can seed the signatures. Kept
    # broad on purpose: users name these columns in many different ways.
    reference_options: dict = {}
    reference_matrix = None
    reference_markers = None
    if df_results is not None and marker_cols:
        usable_markers = [m for m in marker_cols if m in df_results.columns]
        if usable_markers:
            for column in df_results.columns:
                name = str(column).lower()
                if not any(token in name for token in ("phenotype", "ref", "annot", "manual", "truth")):
                    continue
                if pd.api.types.is_numeric_dtype(df_results[column]):
                    continue
                series = df_results[column]
                if series.notna().sum() >= 30 and series.nunique(dropna=True) >= 2:
                    reference_options[str(column)] = series
            if reference_options:
                reference_matrix = df_results[usable_markers].to_numpy(dtype=float)
                reference_markers = [_clean_marker_name_for_signature(m) for m in usable_markers]

    cluster_labels = None
    cluster_sizes = None
    if df_results is not None and cluster_column and cluster_column in df_results.columns:
        cluster_labels = df_results[cluster_column].map(normalize_cluster_id)
        cluster_sizes = cluster_labels.value_counts().to_dict()

    accepted = render_phenotype_recommendation_panel(
        key_prefix=key_prefix,
        cluster_marker_scores=cluster_marker_scores,
        seed_labels=seed_labels,
        reference_matrix=reference_matrix,
        reference_marker_names=reference_markers,
        reference_label_options=reference_options or None,
        cluster_labels=cluster_labels,
        cluster_sizes=cluster_sizes,
    )

    # Accuracy reporting sits next to the assignment it grades, and knows whether
    # annotations were used to build that assignment (which would make a direct
    # score circular).
    if reference_options and cluster_labels is not None:
        render_reference_evaluation_panel(
            key_prefix=key_prefix,
            cluster_labels=cluster_labels,
            phenotype_map=st.session_state.get(phenotype_map_state_key, {}) or {},
            reference_label_options=reference_options,
            cluster_marker_scores=cluster_marker_scores,
            reference_matrix=reference_matrix,
            reference_marker_names=reference_markers,
            seeded_from_reference=bool(st.session_state.get(f"{key_prefix}_reco_refseed", False)),
        )

    if not accepted:
        return False

    updated = dict(current_map)
    updated.update(accepted)
    st.session_state[phenotype_map_state_key] = normalize_phenotype_map_keys(updated)

    known = list(st.session_state.get(available_phenotypes_state_key, []) or [])
    for phenotype in sorted(set(accepted.values())):
        if phenotype not in known:
            known.append(phenotype)
    st.session_state[available_phenotypes_state_key] = known

    st.success(f"Applied {len(accepted)} cluster assignment(s). Review them in the table below.")
    return True


def _clean_marker_name_for_signature(marker: str) -> str:
    """Strip pipeline suffixes so signatures can be written with plain marker names."""
    name = str(marker)
    for suffix in ("_corr_norm", "_norm", "_corr", "_unmixed", "_resolve"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def build_cluster_marker_score_map_from_heatmap(heatmap_z_ordered: pd.DataFrame) -> dict:
    """Build click-selector marker scores directly from the displayed heatmap matrix."""
    if heatmap_z_ordered is None:
        return {}

    if isinstance(heatmap_z_ordered, pd.Series):
        heatmap_z_ordered = heatmap_z_ordered.to_frame()

    if heatmap_z_ordered.empty:
        return {}

    heatmap_scores = heatmap_z_ordered.copy()
    if heatmap_scores.index.has_duplicates:
        heatmap_scores = heatmap_scores.groupby(level=0).mean()

    cluster_marker_scores = {}
    for cluster_id in heatmap_scores.columns:
        normalized_cluster_id = normalize_cluster_id(cluster_id)
        if normalized_cluster_id is None:
            continue
        marker_scores = {
            str(marker): float(score)
            for marker, score in heatmap_scores[cluster_id].items()
            if pd.notna(score)
        }
        cluster_marker_scores[normalized_cluster_id] = marker_scores
    return cluster_marker_scores


_PHENOTYPING_METHOD_LABELS = {
    "leiden": "Leiden",
    "vortex": "Vortex",
    "som": "SOM",
    "resolve": "Resolve",
    "scimap": "SCIMAP",
    "mrml": "MR-ML",
    "mlc": "MR Consensus ML",
    "subpheno": "Sub-Phenotyping",
}


def _normalization_flag(method_key: str, name: str) -> bool | None:
    """Read a method's current normalization flag without assuming its UI shape."""
    aliases = [
        f"{method_key}_{name}",
        f"{method_key}_norm_{name}",
    ]
    for key in aliases:
        if key in st.session_state:
            return bool(st.session_state[key])

    config = st.session_state.get(f"{method_key}_config", {})
    if isinstance(config, dict):
        for key in (name, f"norm_{name}"):
            if key in config:
                return bool(config[key])
    return None


def build_marker_source_note(
    marker_columns: list[str] | None,
    *,
    method_key: str | None = None,
    method_label: str | None = None,
    normalization_label: str | None = None,
) -> str:
    """Describe the actual marker source and transformation used by a method."""
    columns = [str(column) for column in (marker_columns or [])]
    source_flags = []
    for column in columns:
        stem = re.sub(r"(?:_norm|_unmixed|_resolve)+$", "", column, flags=re.IGNORECASE)
        source_flags.append(bool(re.search(r"_corr$", stem, flags=re.IGNORECASE)))

    if source_flags and all(source_flags):
        source = "Corrected markers (corrected columns)"
    elif source_flags and not any(source_flags):
        source = "Raw markers (raw columns)"
    elif source_flags:
        source = "Raw and corrected markers"
    else:
        source = "Selected marker columns"

    if normalization_label:
        preparation = str(normalization_label)
    elif method_key in {"mrml", "mlc"}:
        config = st.session_state.get(f"{method_key}_config", {})
        use_z = bool(config.get("use_z_normalization", False)) if isinstance(config, dict) else False
        preparation = "Batch-aware Z-score normalization" if use_z else "No additional marker normalization"
    else:
        flags = {
            name: _normalization_flag(method_key or "", name)
            for name in ("z1", "z2", "cdf", "nlog")
        }
        steps = [name.upper() for name, enabled in flags.items() if enabled]
        corrected = bool(source_flags and any(source_flags))
        sinh_enabled = _normalization_flag(method_key or "", "sinh")
        if corrected and sinh_enabled:
            cofactor = 5.0
            bc_config = st.session_state.get("bc_config", {})
            if isinstance(bc_config, dict):
                try:
                    cofactor = float(bc_config.get("arcsinh_cofactor", cofactor))
                except (TypeError, ValueError):
                    pass
            steps.insert(0, f"inverse arcsinh (cofactor={cofactor:g})")
        preparation = " → ".join(steps) if steps else (
            "Pipeline normalization" if any("_norm" in column.lower() for column in columns) else "No additional marker normalization"
        )

    method = method_label or _PHENOTYPING_METHOD_LABELS.get(method_key or "", method_key or "Phenotyping")
    return f"{method}: {source}; preparation: {preparation}."


def get_analysis_marker_label(marker: str) -> str:
    """Return a display label while preserving the user's raw/corrected choice."""
    return re.sub(
        r"(?:_norm|_unmixed|_resolve)+$",
        "",
        str(marker),
        flags=re.IGNORECASE,
    )

from spacemap.marker_config import marker_config_path

MARKER_CONFIG_PATH = marker_config_path()


def _normalize_marker_list(markers: object) -> list[str]:
    """Return unique, non-empty marker names while preserving their configured order."""
    if not isinstance(markers, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        marker_name = str(marker).strip()
        if marker_name and marker_name not in seen:
            normalized.append(marker_name)
            seen.add(marker_name)
    return normalized


def _load_markers_from_config(notebook_name: str) -> tuple[list[str], list[str]] | None:
    """Load marker defaults from the shared manifest, if it defines this notebook."""
    try:
        with open(MARKER_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        logging.warning("Could not load marker configuration %s: %s", MARKER_CONFIG_PATH, error)
        return None

    notebook_configs = config.get("notebooks", {}) if isinstance(config, dict) else {}
    marker_config = notebook_configs.get(Path(notebook_name).name)
    if not isinstance(marker_config, dict):
        return None

    profiles = config.get("profiles", {})
    profile_name = marker_config.get("profile")
    profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
    if not isinstance(profile, dict):
        profile = {}

    return (
        _normalize_marker_list(marker_config.get("corr_markers", profile.get("corr_markers", []))),
        _normalize_marker_list(marker_config.get("non_corr_markers", profile.get("non_corr_markers", []))),
    )


def _extract_markers_from_notebook(notebook_name="Step6A_phenotype_leiden.ipynb"):
    """Load marker defaults from the shared config, falling back to notebook parsing.
    
    Parameters
    ----------
    notebook_name : str
        Name of the notebook file (e.g., 'Step6A_phenotype_leiden.ipynb' or 'Step6B_phenotype_SOM.ipynb')
    
    Returns
    -------
    tuple[list[str], list[str]]
        (corr_markers, non_corr_markers)
    """
    configured_markers = _load_markers_from_config(notebook_name)
    if configured_markers is not None:
        return configured_markers

    try:
        notebook_path = project_root_calc / "notebooks" / notebook_name
        if not notebook_path.exists():
            return [], []
        
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        corr_markers: list[str] = []
        non_corr_markers: list[str] = []
        
        def _extract_list_literal_strings(var_name: str, cell_source: str) -> list[str]:
            """Extract quoted strings from a list literal like VAR = ["A", "B" ...].

            Notes
            -----
            - Uses regex to collect quoted tokens inside the brackets.
            - Robust to notebook typos like missing commas (adjacent strings), because we don't rely on Python eval.
            """
            m = re.search(rf"{re.escape(var_name)}\s*=\s*\[(.*?)\]", cell_source, flags=re.S)
            if not m:
                return []
            inside = m.group(1)
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", inside)
            # Deduplicate while preserving order
            out: list[str] = []
            seen: set[str] = set()
            for t in tokens:
                t = str(t).strip()
                if not t or t in seen:
                    continue
                out.append(t)
                seen.add(t)
            return out

        for cell in notebook.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            source = ''.join(cell.get('source', []))
            if not source:
                continue

            if not corr_markers and 'CORR_MARKERS' in source:
                corr_markers = _extract_list_literal_strings('CORR_MARKERS', source)
            if not non_corr_markers and 'NON_CORR_MARKERS' in source:
                non_corr_markers = _extract_list_literal_strings('NON_CORR_MARKERS', source)
        
        return corr_markers, non_corr_markers
    except Exception as e:
        _log_console(f"Warning: Could not extract markers from notebook: {e}", level="warning")
        return [], []


def _normalize_marker_name(marker: str) -> str:
    """Normalize marker names for matching threshold files."""
    marker = str(marker).strip()
    marker = re.sub(r"\s*\(.*\)$", "", marker)
    marker = re.sub(r"\s*\[.*\]$", "", marker)
    for suffix in ("_corr_norm", "_norm", "_corr"):
        if marker.endswith(suffix):
            marker = marker[: -len(suffix)]
            break
    return marker.strip()


def _numbered_polygon_columns(columns, polygon_prefix: str | None) -> list:
    """Return naturally ordered ``<prefix>_<integer>`` columns only.

    ``polygon_prefix`` may be supplied with or without the trailing underscore;
    regex metacharacters in custom prefixes are treated literally.
    """
    if not polygon_prefix:
        return []
    prefix = str(polygon_prefix).strip().rstrip("_")
    if not prefix:
        return []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", flags=re.IGNORECASE)
    numbered = []
    for position, column in enumerate(columns):
        match = pattern.fullmatch(str(column))
        if match:
            numbered.append((int(match.group(1)), position, column))
    return [column for _, _, column in sorted(numbered)]


def _threshold_settings_fingerprint(
    threshold_dir: str | os.PathLike | None,
    threshold_source: str | None,
    threshold_statistic: str | None,
    marker_col_name: str | None,
    value_col_name: str | None,
    polygon_prefix: str | None,
    selected_markers: list[str] | None = None,
) -> str:
    """Stable identity for source-aware threshold loading settings."""
    directory = ""
    if threshold_dir:
        try:
            directory = str(Path(threshold_dir).expanduser().resolve(strict=False))
        except OSError:
            directory = str(threshold_dir)
    payload = {
        "threshold_dir": directory,
        "threshold_source": str(threshold_source or "Auto").strip().lower(),
        "threshold_statistic": str(threshold_statistic or "median").strip().lower(),
        "marker_col_name": str(marker_col_name or "").strip().lower(),
        "value_col_name": str(value_col_name or "").strip().lower(),
        "polygon_prefix": str(polygon_prefix or "").strip().lower().rstrip("_"),
        "selected_markers": sorted(str(marker) for marker in (selected_markers or [])),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


LOW_EXPRESSION_METRIC_OPTIONS: tuple[str, ...] = (
    "mean_zscore",
    "raw_sum",
)

LOW_EXPRESSION_METRIC_LABELS: dict[str, str] = {
    "mean_zscore": "Mean marker z-score",
    "raw_sum": "Raw marker sum",
}

LOW_EXPRESSION_DEFAULT_THRESHOLDS: dict[str, float] = {
    "mean_zscore": -0.75,
    "raw_sum": 0.1,
}


def get_low_expression_metric_label(metric: str) -> str:
    return LOW_EXPRESSION_METRIC_LABELS.get(metric, LOW_EXPRESSION_METRIC_LABELS["mean_zscore"])


def get_low_expression_default_threshold(metric: str) -> float:
    return float(LOW_EXPRESSION_DEFAULT_THRESHOLDS.get(metric, LOW_EXPRESSION_DEFAULT_THRESHOLDS["mean_zscore"]))


def get_low_expression_metric_help_text() -> str:
    return (
        "Choose how low-expression cells are screened before clustering. "
        "Cells below the threshold are kept in the results as a separate low-expression bucket rather than being silently lost. "
        "These tabs apply normalization in the order Z1 -> Z2 -> CDF -> NLog.\n\n"
        "Rule-of-thumb table:\n\n"
        "```text\n"
        "Normalization flags        Recommended metric      Starting cutoff\n"
        "Z1 only                    Mean marker z-score     -0.75\n"
        "Z1 + CDF                   Mean marker z-score     tune near -0.75 to 0\n"
        "Z1 + NLog                  Mean marker z-score     tune empirically\n"
        "Z1 + CDF + NLog            Mean marker z-score     tune empirically\n"
        "Any setup with Z2          Raw marker sum          dataset-specific\n"
        "Z1 + Z2 + CDF + NLog       Raw marker sum          dataset-specific\n"
        "```\n\n"
        "Why: Mean marker z-score averages the final normalized marker columns. "
        "Once Z2 is enabled, that score is no longer a stable proxy for overall low expression, "
        "so Raw marker sum is the safer choice."
    )


def compute_low_expression_score(
    df: pd.DataFrame,
    raw_marker_cols: list[str],
    norm_marker_cols: list[str],
    metric: str = "mean_zscore",
) -> pd.Series:
    """Score each cell for low-expression screening.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing raw and/or normalized marker columns.
    raw_marker_cols : list[str]
        Raw marker columns selected for the run.
    norm_marker_cols : list[str]
        Normalized marker columns corresponding to the selected markers.
    metric : str
        ``"mean_zscore"`` uses the average value across normalized marker
        columns. ``"raw_sum"`` uses the sum of the raw selected markers.

    Returns
    -------
    pd.Series
        One score per cell. Lower scores indicate lower overall expression.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    if metric == "raw_sum":
        valid_raw = [column for column in raw_marker_cols if column in df.columns]
        if not valid_raw:
            raise ValueError("No raw marker columns are available for low-expression scoring.")
        raw_values = df[valid_raw].apply(pd.to_numeric, errors="coerce").fillna(0)
        return raw_values.sum(axis=1)

    valid_norm = [column for column in norm_marker_cols if column in df.columns]
    if valid_norm:
        norm_values = df[valid_norm].apply(pd.to_numeric, errors="coerce").fillna(0)
        return norm_values.mean(axis=1)

    valid_raw = [column for column in raw_marker_cols if column in df.columns]
    if not valid_raw:
        raise ValueError("No marker columns are available for low-expression scoring.")

    raw_values = df[valid_raw].apply(pd.to_numeric, errors="coerce").fillna(0)
    raw_z = raw_values.apply(
        lambda column: (column - column.mean()) / (column.std() if column.std() > 1e-9 else 1.0),
        axis=0,
    ).fillna(0)
    return raw_z.mean(axis=1)


def _extract_thresholds_from_csv(
    df: pd.DataFrame,
    selected_markers: list[str],
    marker_col_name: str | None,
    value_col_name: str | None,
    polygon_prefix: str | None,
    threshold_statistic: str,
    threshold_source: str = "Auto",
    *,
    return_metadata: bool = False,
) -> dict | tuple[dict, dict]:
    """Extract marker thresholds from one CSV with explicit source semantics.

    ``Auto`` is deliberately conservative and backwards compatible:

    1. an explicitly configured ``value_col_name``;
    2. a non-null manual override (or a true override flag paired with the
       final scalar);
    3. one or more ``Polygon_*`` values;
    4. a final/pre-computed scalar such as ``threshold_final`` or
       ``threshold``.

    ``Polygon values`` never consumes a scalar, while ``Final scalar`` uses a
    scalar directly and never applies the polygon statistic.  The optional
    metadata return is used by RESOLVE's review table; callers relying on the
    historical dictionary return remain unaffected.
    """
    thresholds: dict[str, float] = {}
    metadata: dict[str, dict] = {}
    if df.empty:
        return (thresholds, metadata) if return_metadata else thresholds

    source_key = str(threshold_source or "Auto").strip().lower().replace("_", " ")
    if source_key in {"polygon", "polygons", "polygon value", "polygon values"}:
        source_key = "polygon"
    elif source_key in {"final", "scalar", "final scalar", "precomputed", "direct"}:
        source_key = "scalar"
    else:
        source_key = "auto"

    statistic_key = str(threshold_statistic or "median").strip().lower()
    statistic_aliases = {
        "p05": "percentile_05",
        "p5": "percentile_05",
        "q05": "percentile_05",
        "percentile_5": "percentile_05",
        "p25": "percentile_25",
        "q25": "percentile_25",
        "percentile_25": "percentile_25",
        "p75": "percentile_75",
        "q75": "percentile_75",
        "percentile_75": "percentile_75",
    }
    statistic_key = statistic_aliases.get(statistic_key, statistic_key)
    statistic_labels = {
        "median": "Median",
        "mean": "Mean",
        "min": "Minimum",
        "percentile_05": "5th percentile (p05)",
        "percentile_25": "25th percentile (p25)",
        "percentile_75": "75th percentile (p75)",
    }

    def _stat(values: pd.Series) -> float | None:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return None
        if statistic_key == "mean":
            return float(values.mean())
        if statistic_key == "min":
            return float(values.min())
        if statistic_key == "percentile_05":
            return float(values.quantile(0.05))
        if statistic_key == "percentile_25":
            return float(values.quantile(0.25))
        if statistic_key == "percentile_75":
            return float(values.quantile(0.75))
        return float(values.median())

    def _numeric(value) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    def _first_numeric(row: pd.Series, columns: list[str]) -> tuple[float | None, str | None]:
        for column in columns:
            if column and column in row.index:
                value = _numeric(row[column])
                if value is not None:
                    return value, column
        return None, None

    def _flag_is_true(value) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "n", "none", "nan"}:
            return False
        if text in {"1", "true", "yes", "y", "manual", "override", "overridden"}:
            return True
        try:
            return float(text) != 0.0
        except ValueError:
            return False

    cols_lower = {c.lower(): c for c in df.columns}
    marker_col = cols_lower.get("marker") or cols_lower.get("markers")

    if marker_col_name:
        marker_col = cols_lower.get(str(marker_col_name).lower(), marker_col)

    explicit_value_col = None
    if value_col_name:
        explicit_value_col = cols_lower.get(str(value_col_name).lower())

    final_scalar_cols = [
        column for column in (
            cols_lower.get("threshold_final"),
            cols_lower.get("final_threshold"),
            cols_lower.get("threshold"),
            cols_lower.get("median"),
            cols_lower.get("value"),
            cols_lower.get("cutoff"),
            cols_lower.get("gate"),
        ) if column
    ]
    # Preserve order while removing aliases that resolve to the same column.
    final_scalar_cols = list(dict.fromkeys(final_scalar_cols))

    manual_value_cols = [
        column for column in (
            cols_lower.get("manual_override"),
            cols_lower.get("threshold_override"),
            cols_lower.get("override_value"),
        ) if column
    ]
    override_flag_cols = [
        column for column in (
            cols_lower.get("override"),
            cols_lower.get("override_applied"),
            cols_lower.get("manual_override_applied"),
            cols_lower.get("is_override"),
        ) if column
    ]

    polygon_cols = _numbered_polygon_columns(df.columns, polygon_prefix)

    def _manual_override(row: pd.Series) -> tuple[float | None, str | None]:
        # A numeric manual-override column carries the override itself.  A
        # boolean/string flag instead points at the exported final scalar.
        for column in manual_value_cols:
            raw = row.get(column)
            if raw is None or pd.isna(raw):
                continue
            if isinstance(raw, (bool, np.bool_)) or (
                isinstance(raw, str)
                and str(raw).strip().lower() in {
                    "true", "false", "yes", "no", "y", "n", "manual", "override", "overridden"
                }
            ):
                if _flag_is_true(raw):
                    return _first_numeric(row, final_scalar_cols)
                continue
            value = _numeric(raw)
            if value is not None:
                return value, column

        if any(_flag_is_true(row.get(column)) for column in override_flag_cols):
            return _first_numeric(row, final_scalar_cols)
        return None, None

    def _record(marker: str, value: float, source: str, aggregation: str, polygon_count: int) -> None:
        thresholds[marker] = float(value)
        metadata[marker] = {
            "source": source,
            "aggregation": aggregation,
            "polygon_count": int(polygon_count),
        }

    if marker_col and (explicit_value_col or final_scalar_cols or polygon_cols or manual_value_cols):
        for _, row in df.dropna(subset=[marker_col]).iterrows():
            m = _normalize_marker_name(row[marker_col])
            if not m:
                continue

            polygon_values = pd.to_numeric(row[polygon_cols], errors="coerce").dropna() if polygon_cols else pd.Series(dtype=float)
            polygon_count = int(len(polygon_values))
            value = None

            if source_key in {"auto", "scalar"} and explicit_value_col:
                value = _numeric(row.get(explicit_value_col))
                if value is not None:
                    _record(m, value, f"Explicit value ({explicit_value_col})", "Direct scalar", polygon_count)
                    continue

            if source_key == "auto":
                value, _ = _manual_override(row)
                if value is not None:
                    _record(m, value, "Manual override", "Direct scalar", polygon_count)
                    continue

            if source_key in {"auto", "polygon"} and polygon_count:
                value = _stat(polygon_values)
                if value is not None:
                    aggregation = (
                        "Direct (one polygon)"
                        if polygon_count == 1
                        else statistic_labels.get(statistic_key, "Median")
                    )
                    _record(m, value, "Polygon values", aggregation, polygon_count)
                    continue

            if source_key in {"auto", "scalar"}:
                value, scalar_col = _first_numeric(row, final_scalar_cols)
                if value is not None:
                    _record(
                        m,
                        value,
                        f"Final scalar ({scalar_col})" if scalar_col else "Final scalar",
                        "Direct scalar",
                        polygon_count,
                    )

        return (thresholds, metadata) if return_metadata else thresholds

    # Legacy wide layout: marker names are columns and rows contain scalar
    # threshold values.  It is a scalar source, so an explicit Polygon-values
    # request correctly leaves it unresolved for coverage validation.
    if source_key != "polygon":
        for marker in selected_markers:
            base = _normalize_marker_name(marker)
            for candidate in (marker, base):
                if candidate in df.columns:
                    val = _stat(df[candidate])
                    if val is not None:
                        _record(base, val, f"Wide scalar ({candidate})", "Direct scalar", 0)
                    break

    return (thresholds, metadata) if return_metadata else thresholds


def _normalize_batch_stem(value: str) -> str:
    """Strip common imaging-pipeline suffixes to get a canonical batch key.

    Handles QuPath / OME-TIFF / cell-features / annotation suffixes so that
    ``P11_roi_1.ome.tiff_-_Merged_Image`` normalises to ``p11_roi_1`` —
    matching the typical ``batch_id`` column value.
    """
    text = str(value).strip()
    text = re.sub(r"\.csv$", "", text, flags=re.IGNORECASE)
    # Wildcard: strips any _-_<ImageName> suffix (Merged_Image, Fixed_Nolan_Image, etc.)
    text = re.sub(r"\.ome\.tiff_-_.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_-_.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.ome\.tiff$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.ome$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_fully_annotated$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_cell_features.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_annotated$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_processed$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_summary$", "", text, flags=re.IGNORECASE)
    return text.strip("._ ").lower()


def _clean_batch_id_for_display(value: str) -> str:
    """Strip imaging-pipeline suffixes from a batch_id, PRESERVING original case.

    Used when storing batch_ids in the threshold table so that values like
    ``reg001_B.ome.tiff_-_Fixed_Nolan_Image`` are shown as ``reg001_B``.
    The matching logic in ``_thresholds_from_table`` uses normalised (lowercase)
    comparison, so the case-preserved value is safe to store.
    """
    text = str(value).strip()
    text = re.sub(r"\.csv$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.ome\.tiff_-_.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_-_.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.ome\.tiff$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.ome$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_fully_annotated$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_cell_features.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_annotated$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_processed$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_summary$", "", text, flags=re.IGNORECASE)
    return text.strip("._ ")


def _match_threshold_stem_to_batches(stem: str, batch_values: list) -> list[str]:
    """Match a threshold filename stem without alphanumeric prefix aliases.

    Normalized exact matches always win.  A boundary match is considered only
    when no exact match exists and requires the next character to be a real
    delimiter, so ``sample`` cannot alias ``sample2`` and ``P1`` cannot alias
    ``P11``.  Among boundary candidates, the most specific normalized key wins.
    """
    normalized_stem = _normalize_batch_stem(stem)
    candidates = [
        (str(batch), _normalize_batch_stem(str(batch)))
        for batch in batch_values
        if str(batch).strip()
    ]
    exact = [raw for raw, normalized in candidates if normalized == normalized_stem]
    if exact:
        return list(dict.fromkeys(exact))

    delimiters = {"_", "-", ".", " "}

    def _boundary_prefix(shorter: str, longer: str) -> bool:
        return (
            bool(shorter)
            and longer.startswith(shorter)
            and len(longer) > len(shorter)
            and longer[len(shorter)] in delimiters
        )

    boundary = [
        (raw, normalized)
        for raw, normalized in candidates
        if _boundary_prefix(normalized, normalized_stem)
        or _boundary_prefix(normalized_stem, normalized)
    ]
    if not boundary:
        return []
    most_specific = max(len(normalized) for _, normalized in boundary)
    return list(dict.fromkeys(
        raw for raw, normalized in boundary if len(normalized) == most_specific
    ))


def _preferred_threshold_directory(project_root: str | os.PathLike) -> Path:
    """Return the preferred RESOLVE threshold directory.

    New image-derived thresholds are written to ``data/thresholds``.  The
    Historical ``data/threshold`` and misspelled ``data/boundries`` folders
    remain read fallbacks only when they contain CSVs and the preferred
    directory does not.
    """
    root = Path(project_root)
    preferred = root / "data" / "thresholds"
    legacy_singular = root / "data" / "threshold"
    legacy = root / "data" / "boundries"

    def _contains_csv(path: Path) -> bool:
        try:
            return path.is_dir() and any(
                child.is_file() and child.suffix.lower() == ".csv"
                for child in path.iterdir()
            )
        except OSError:
            return False

    if _contains_csv(preferred):
        return preferred
    if _contains_csv(legacy_singular):
        return legacy_singular
    if _contains_csv(legacy):
        return legacy
    return preferred


def _load_thresholds_table(
    threshold_dir: str,
    selected_markers: list[str],
    batch_values: list[str],
    marker_col_name: str | None,
    value_col_name: str | None,
    polygon_prefix: str | None,
    threshold_statistic: str,
    threshold_source: str = "Auto",
) -> pd.DataFrame:
    """Load threshold CSV files into a review/edit table."""
    if not threshold_dir or not os.path.exists(threshold_dir) or not os.path.isdir(threshold_dir):
        return pd.DataFrame()

    files = sorted([f for f in os.listdir(threshold_dir) if f.lower().endswith('.csv')])
    rows = []
    for fname in files:
        fpath = os.path.join(threshold_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        thresholds, threshold_metadata = _extract_thresholds_from_csv(
            df,
            selected_markers,
            marker_col_name,
            value_col_name,
            polygon_prefix,
            threshold_statistic,
            threshold_source,
            return_metadata=True,
        )

        def _summarize_metadata(field: str) -> str:
            values = []
            for marker in selected_markers:
                marker_meta = threshold_metadata.get(_normalize_marker_name(marker), {})
                value = str(marker_meta.get(field, "")).strip()
                if value and value not in values:
                    values.append(value)
            if not values:
                return ""
            if len(values) == 1:
                return values[0]
            return "Mixed: " + " / ".join(values)

        polygon_counts = sorted({
            int(meta.get("polygon_count", 0))
            for meta in threshold_metadata.values()
            if int(meta.get("polygon_count", 0)) > 0
        })
        if not polygon_counts and polygon_prefix:
            detected_polygon_count = len(
                _numbered_polygon_columns(df.columns, polygon_prefix)
            )
            if detected_polygon_count:
                polygon_counts = [detected_polygon_count]
        if not polygon_counts:
            polygon_count_display = "0"
        elif len(polygon_counts) == 1:
            polygon_count_display = str(polygon_counts[0])
        else:
            polygon_count_display = f"{polygon_counts[0]}–{polygon_counts[-1]}"

        stem = Path(fname).stem
        matched_batches = _match_threshold_stem_to_batches(stem, batch_values)

        if not matched_batches:
            matched_batches = [stem]

        for batch_id in matched_batches:
            row = {
                # Normalise for display: strip .ome.tiff_-_* and similar suffixes
                # so that rows with data batch_ids like
                # 'reg001_B.ome.tiff_-_Fixed_Nolan_Image' display as 'reg001_B'.
                # _thresholds_from_table uses normalised matching, so the clean
                # value still maps back to the correct data rows.
                "batch_id": _clean_batch_id_for_display(batch_id),
                "Threshold_File": fname,
                "Threshold_Source": _summarize_metadata("source"),
                "Threshold_Aggregation": _summarize_metadata("aggregation"),
                "Polygon_Count": polygon_count_display,
            }
            for marker in selected_markers:
                base = _normalize_marker_name(marker)
                val = thresholds.get(marker)
                if val is None:
                    val = thresholds.get(base, np.nan)
                row[marker] = val
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _threshold_table_has_values(thresholds_table: pd.DataFrame, selected_markers: list[str]) -> bool:
    """Return True if the threshold table has any usable numeric values."""
    if thresholds_table is None or thresholds_table.empty:
        return False
    marker_cols = [m for m in selected_markers if m in thresholds_table.columns]
    if not marker_cols:
        return False
    return thresholds_table[marker_cols].notna().any().any()


def _threshold_table_coverage(
    thresholds_table: pd.DataFrame,
    selected_markers: list[str],
    batch_values: list | None = None,
) -> dict:
    """Validate complete, finite threshold coverage for every batch and marker.

    The returned dictionary is intentionally UI-agnostic so RESOLVE can fail
    closed with a useful message and focused unit tests can exercise the same
    contract.  Batch identifiers use the same canonicalisation as threshold
    file matching.
    """
    report = {
        "complete": False,
        "missing_batches": [],
        "duplicate_batches": [],
        "missing_by_batch": {},
        "ambiguous_global": False,
    }
    if thresholds_table is None or thresholds_table.empty:
        report["missing_batches"] = [str(value) for value in (batch_values or ["global"])]
        return report

    markers = list(dict.fromkeys(str(marker) for marker in selected_markers if str(marker)))
    expected = [value for value in (batch_values or []) if not pd.isna(value)]

    def _missing_markers(row: pd.Series) -> list[str]:
        missing = []
        for marker in markers:
            if marker not in row.index:
                missing.append(marker)
                continue
            try:
                value = float(row[marker])
            except (TypeError, ValueError):
                missing.append(marker)
                continue
            if not np.isfinite(value):
                missing.append(marker)
        return missing

    if not expected:
        # With no data batch column, applying the first of several batch rows
        # would be arbitrary.  A single universal/direct row is required.
        if len(thresholds_table) != 1:
            report["ambiguous_global"] = True
            return report
        missing = _missing_markers(thresholds_table.iloc[0])
        if missing:
            report["missing_by_batch"] = {"global": missing}
            return report
        report["complete"] = True
        return report

    if "batch_id" not in thresholds_table.columns:
        report["missing_batches"] = [str(value) for value in expected]
        return report

    rows_by_batch: dict[str, list[int]] = {}
    for index, raw_batch in thresholds_table["batch_id"].items():
        if pd.isna(raw_batch):
            continue
        normalised = _normalize_batch_stem(str(raw_batch))
        if normalised:
            rows_by_batch.setdefault(normalised, []).append(index)

    expected_by_normalised: dict[str, str] = {}
    for raw_batch in expected:
        normalised = _normalize_batch_stem(str(raw_batch))
        if normalised:
            expected_by_normalised[normalised] = str(raw_batch)

    for normalised, display_batch in expected_by_normalised.items():
        indices = rows_by_batch.get(normalised, [])
        if not indices:
            report["missing_batches"].append(display_batch)
            continue
        if len(indices) > 1:
            report["duplicate_batches"].append(display_batch)
            continue
        missing = _missing_markers(thresholds_table.loc[indices[0]])
        if missing:
            report["missing_by_batch"][display_batch] = missing

    report["complete"] = not any((
        report["missing_batches"],
        report["duplicate_batches"],
        report["missing_by_batch"],
        report["ambiguous_global"],
    ))
    return report


def load_universal_gating_csv(
    csv_path: str,
    selected_markers: list[str],
    batch_values: list[str] | None = None,
) -> pd.DataFrame:
    """Load universal (no-batch) thresholds from a Threshold Gating tab CSV.

    The CSV has columns ``marker, threshold`` (and optionally ``batch_id``).
    When the Gating tab exports without batch mode, every marker appears once
    per original batch, but only the first row carries the actual threshold
    and the rest are 0.0.

    This function:
        1. Reads the CSV.
        2. Extracts the **maximum non-zero** threshold per marker.
        3. Matches markers to *selected_markers* via :func:`_normalize_marker_name`.
        4. Broadcasts the single threshold to **every batch** in *batch_values*
           (or one ``"universal"`` row if no batches are given).

    Returns a DataFrame compatible with the threshold draft table
    (columns: ``batch_id``, ``Threshold_File``, + one column per matched marker).
    """
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    df.columns = [c.lower().strip() for c in df.columns]
    if "marker" not in df.columns or "threshold" not in df.columns:
        return pd.DataFrame()

    # Extract the best (max non-zero) threshold per marker
    marker_thresholds: dict[str, float] = {}
    for marker_raw, group in df.groupby("marker"):
        vals = group["threshold"].dropna()
        nonzero = vals[vals != 0.0]
        if len(nonzero) > 0:
            marker_thresholds[str(marker_raw).strip()] = float(nonzero.max())

    if not marker_thresholds:
        return pd.DataFrame()

    # Match to selected markers
    matched: dict[str, float] = {}
    for sel_m in selected_markers:
        base_sel = _normalize_marker_name(sel_m)
        # Try exact match first, then normalised
        for raw_m, val in marker_thresholds.items():
            if raw_m == sel_m or _normalize_marker_name(raw_m) == base_sel:
                matched[sel_m] = val
                break

    if not matched:
        return pd.DataFrame()

    # Broadcast to all batches (or one universal row)
    targets = batch_values if batch_values else ["universal"]
    rows = []
    for bid in targets:
        row: dict[str, object] = {
            "batch_id": str(bid),
            "Threshold_File": os.path.basename(csv_path),
        }
        for m in selected_markers:
            row[m] = matched.get(m, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


class StreamlitLogger:
    """Log both to terminal and Streamlit container."""
    def __init__(self):
        self.terminal_logger = logging.getLogger("SPACEMAP_UI")
        self.terminal_logger.setLevel(logging.INFO)
        self._streamlit_placeholder = None
        self._streamlit_session_key = None
        self._max_streamlit_lines = 400
        if not self.terminal_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
            self.terminal_logger.addHandler(handler)

    def start_streamlit_capture(self, placeholder, session_key: str, clear: bool = True):
        self._streamlit_placeholder = placeholder
        self._streamlit_session_key = session_key
        try:
            if clear or session_key not in st.session_state:
                st.session_state[session_key] = []
            self._render_streamlit_buffer()
        except Exception:
            pass

    def stop_streamlit_capture(self):
        self._streamlit_placeholder = None
        self._streamlit_session_key = None

    def get_streamlit_buffer(self, session_key: str) -> list[str]:
        try:
            value = st.session_state.get(session_key, [])
            return value if isinstance(value, list) else []
        except Exception:
            return []

    def _append_streamlit(self, message: str):
        if not self._streamlit_session_key:
            return
        try:
            timestamp = time.strftime('%H:%M:%S')
            lines = st.session_state.get(self._streamlit_session_key, [])
            if not isinstance(lines, list):
                lines = []
            lines.append(f"{timestamp} - {message}")
            st.session_state[self._streamlit_session_key] = lines[-self._max_streamlit_lines:]
            self._render_streamlit_buffer()
        except Exception:
            pass

    def _render_streamlit_buffer(self):
        if self._streamlit_placeholder is None or not self._streamlit_session_key:
            return
        try:
            lines = st.session_state.get(self._streamlit_session_key, [])
            if isinstance(lines, list):
                self._streamlit_placeholder.code("\n".join(lines), language=None)
        except Exception:
            pass

    def info(self, msg, *args):
        message = msg % args if args else str(msg)
        self.terminal_logger.info(message)
        self._append_streamlit(message)
        record_process_event("tab7", "Phenotyping step", message=message)

    def error(self, msg, *args):
        message = msg % args if args else str(msg)
        self.terminal_logger.error(message)
        self._append_streamlit(f"ERROR: {message}")
        record_process_event("tab7", "Phenotyping step", message=message, status="error")

    def warning(self, msg, *args):
        message = msg % args if args else str(msg)
        self.terminal_logger.warning(message)
        self._append_streamlit(f"WARNING: {message}")
        record_process_event("tab7", "Phenotyping step", message=message, status="warning")

    def exception(self, msg, *args):
        """Log an error plus the active traceback.

        Callers in except-blocks use logger.exception the way the stdlib logger
        works. Without it the AttributeError raised here replaces the real
        exception, so the underlying failure never reaches the user.
        """
        import traceback

        message = msg % args if args else str(msg)
        detail = traceback.format_exc()
        self.terminal_logger.error(f"{message}\n{detail}")
        self._append_streamlit(f"ERROR: {message}")
        record_process_event("tab7", "Phenotyping step", message=message, status="error")

    def debug(self, msg, *args):
        message = msg % args if args else str(msg)
        self.terminal_logger.debug(message)

logger = StreamlitLogger()

class Timer:
    """Context manager to time execution."""
    def __init__(self, name="Process"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"STARTED: {self.name}")
        st.toast(f"Started: {self.name}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        if exc_type:
            # Streamlit stop/rerun signals inherit BaseException. Logging them
            # through the UI can raise another control signal and hide the
            # original reason, so report it directly to the terminal.
            label = "FAILED" if issubclass(exc_type, Exception) else "INTERRUPTED"
            logger.terminal_logger.error(
                "%s: %s after %.2fs (%s: %s)",
                label, self.name, elapsed, exc_type.__name__, exc_val,
            )
        else:
            logger.info(f"COMPLETED: {self.name} in {elapsed:.2f}s")
            st.toast(f"Finished: {self.name} ({elapsed:.2f}s)")


# ==============================================================================
# Phenotyping reproducibility / run audit logs
# ==============================================================================

PHENOTYPING_AUDIT_COLUMNS = [
    "run_id",
    "event_type",
    "status",
    "method_key",
    "method_name",
    "project_name",
    "started_at",
    "finished_at",
    "duration_seconds",
    "input_csv",
    "input_file_size_bytes",
    "input_rows",
    "output_path",
    "output_rows",
    "output_column",
    "cluster_column",
    "marker_count",
    "markers_json",
    "assigned_cluster_count",
    "assigned_phenotypes_json",
    "phenotype_counts_json",
    "parameters_json",
    "extra_json",
    "hostname",
    "operating_system",
    "python_version",
    "software_versions_json",
    "cpu_model",
    "cpu_physical_cores",
    "cpu_logical_cores",
    "ram_total_gb",
    "ram_available_gb",
    "gpu_json",
    "git_commit",
    "error",
]


PHENOTYPING_METHOD_LABELS = {
    "leiden": "Leiden Clustering",
    "som": "SOM Clustering",
    "vortex": "Vortex Clustering",
    "resolve": "Resolve Threshold Phenotyping",
    "scimap": "SCIMAP Phenotyping",
    "tools": "Phenotyping Tools",
    "mrml": "MR ML Phenotyping",
    "mlc": "Consensus ML Phenotyping",
    "subpheno": "Sub-Phenotyping",
}


def get_phenotyping_audit_log_path(project_root: Path, method_key: str) -> Path:
    """Return the dedicated append-only CSV for one phenotyping method."""
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method_key)).strip("._") or "method"
    return Path(project_root) / "outputs" / "phenotyping" / "logs" / f"{safe_key}_runs.csv"


def _audit_json_value(value, *, depth: int = 0):
    """Convert common runtime values to compact, reviewable JSON values."""
    if depth > 6:
        return "<maximum nesting reached>"
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _audit_json_value(value.item(), depth=depth + 1)
    if isinstance(value, dict):
        return {
            str(key): _audit_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_audit_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, pd.DataFrame):
        if len(value) <= 200 and len(value.columns) <= 100:
            return {
                "type": "DataFrame",
                "shape": [int(value.shape[0]), int(value.shape[1])],
                "records": _audit_json_value(value.to_dict(orient="records"), depth=depth + 1),
            }
        return {"type": "DataFrame", "shape": [int(value.shape[0]), int(value.shape[1])]}
    if isinstance(value, pd.Series):
        if len(value) <= 500:
            return _audit_json_value(value.tolist(), depth=depth + 1)
        return {"type": "Series", "length": int(len(value))}
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            converted = to_list()
            if isinstance(converted, list) and len(converted) <= 1000:
                return _audit_json_value(converted, depth=depth + 1)
        except Exception:
            pass
    return str(value)


def _audit_json_dumps(value) -> str:
    return json.dumps(_audit_json_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def collect_phenotyping_parameter_snapshot(method_key: str) -> dict:
    """Collect persistent method settings without cache tables or button events."""
    try:
        from spacemap_gui.persistence import is_transient_key
    except Exception:
        is_transient_key = lambda _key: False  # type: ignore

    prefixes = {
        "leiden": ("leiden_",),
        "som": ("som_",),
        "vortex": ("vortex_",),
        "resolve": ("resolve_",),
        "scimap": ("scimap_",),
        "tools": ("tools_",),
        "mrml": ("mrml_",),
        "mlc": ("mlc_",),
        "subpheno": ("subpheno_",),
    }.get(str(method_key), (f"{method_key}_",))

    snapshot = {}
    for key in sorted(st.session_state.keys()):
        key_str = str(key)
        if not key_str.startswith(prefixes) or is_transient_key(key_str):
            continue
        if key_str.endswith("_phenotype_map"):
            continue
        try:
            value = st.session_state[key]
            if callable(value):
                continue
            snapshot[key_str] = _audit_json_value(value)
        except Exception:
            continue
    return snapshot


@lru_cache(maxsize=1)
def _phenotyping_machine_metadata() -> dict:
    cpu_model = platform.processor() or ""
    if not cpu_model:
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name") and ":" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    physical_cores = ""
    logical_cores = os.cpu_count() or ""
    ram_total_gb = ""
    ram_available_gb = ""
    try:
        import psutil  # type: ignore

        physical_cores = psutil.cpu_count(logical=False) or ""
        logical_cores = psutil.cpu_count(logical=True) or logical_cores
        vm = psutil.virtual_memory()
        ram_total_gb = round(float(vm.total) / (1024 ** 3), 3)
        ram_available_gb = round(float(vm.available) / (1024 ** 3), 3)
    except Exception:
        try:
            meminfo = {}
            with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    name, raw_value = line.split(":", 1)
                    meminfo[name] = float(raw_value.strip().split()[0]) * 1024
            ram_total_gb = round(meminfo.get("MemTotal", 0.0) / (1024 ** 3), 3)
            ram_available_gb = round(meminfo.get("MemAvailable", 0.0) / (1024 ** 3), 3)
        except Exception:
            pass

    gpu_records = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            for index, line in enumerate(result.stdout.splitlines()):
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 3:
                    gpu_records.append(
                        {
                            "index": index,
                            "name": parts[0],
                            "memory_total_mb": parts[1],
                            "driver_version": parts[2],
                        }
                    )
    except Exception:
        pass

    software_versions = {}
    for distribution in (
        "streamlit",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "scanpy",
        "anndata",
        "umap-learn",
        "MiniSom",
        "scimap",
    ):
        try:
            software_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue

    return {
        "hostname": socket.gethostname(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "software_versions_json": _audit_json_dumps(software_versions),
        "cpu_model": cpu_model,
        "cpu_physical_cores": physical_cores,
        "cpu_logical_cores": logical_cores,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "gpu_json": _audit_json_dumps(gpu_records),
    }


def _phenotyping_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _phenotype_audit_summary(phenotype_map: dict | None) -> tuple[int, dict, dict]:
    normalized = normalize_phenotype_map_keys(phenotype_map)
    assigned = {
        str(cluster_id): str(phenotype)
        for cluster_id, phenotype in normalized.items()
        if str(phenotype).strip() not in {"", "Unassigned", "None", "nan"}
    }
    counts: dict[str, int] = {}
    for phenotype in assigned.values():
        counts[phenotype] = counts.get(phenotype, 0) + 1
    return len(assigned), assigned, counts


def begin_phenotyping_run_audit(
    project_root: Path,
    method_key: str,
    *,
    input_csv: str | Path | None = None,
    markers: list[str] | None = None,
    parameters: dict | None = None,
    event_type: str = "analysis_run",
) -> dict | None:
    """Capture the immutable start of a method run when logging is enabled."""
    from spacemap_gui import run_log

    if not run_log.enabled("tab7"):
        return None
    now = datetime.now().astimezone()
    return {
        "run_id": f"{method_key}-{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "event_type": event_type,
        "started_at": now.isoformat(timespec="seconds"),
        "started_perf": time.perf_counter(),
        "input_csv": str(input_csv or ""),
        "markers": list(markers or []),
        "parameters": parameters if parameters is not None else collect_phenotyping_parameter_snapshot(method_key),
        "method_key": method_key,
        "project_root": str(project_root),
        "project_name": st.session_state.get("active_project_name", project_root.name),
    }


def finish_phenotyping_run_audit(
    audit: dict | None,
    *,
    status: str,
    output_path: str | Path | None = None,
    input_rows: int | None = None,
    output_rows: int | None = None,
    output_column: str | None = None,
    cluster_column: str | None = None,
    phenotype_map: dict | None = None,
    extra: dict | None = None,
    error: str | None = None,
    run_logger=None,
) -> Path | None:
    """Append one completed run/event row without allowing logging to fail analysis."""
    if not audit:
        return None
    audit_logger = run_logger if run_logger is not None else logger
    try:
        project_root = Path(audit["project_root"])
        method_key = str(audit["method_key"])
        finished = datetime.now().astimezone()
        input_csv = str(audit.get("input_csv", ""))
        input_size = ""
        if input_csv and Path(input_csv).is_file():
            input_size = Path(input_csv).stat().st_size

        assigned_count, assigned_map, phenotype_counts = _phenotype_audit_summary(phenotype_map)
        machine = _phenotyping_machine_metadata()
        row = {column: "" for column in PHENOTYPING_AUDIT_COLUMNS}
        row.update(machine)
        row.update(
            {
                "run_id": audit.get("run_id", ""),
                "event_type": audit.get("event_type", "analysis_run"),
                "status": status,
                "method_key": method_key,
                "method_name": PHENOTYPING_METHOD_LABELS.get(method_key, method_key),
                "project_name": audit["project_name"] if "project_name" in audit else st.session_state.get("active_project_name", project_root.name),
                "started_at": audit.get("started_at", ""),
                "finished_at": finished.isoformat(timespec="seconds"),
                "duration_seconds": round(time.perf_counter() - float(audit.get("started_perf", time.perf_counter())), 6),
                "input_csv": input_csv,
                "input_file_size_bytes": input_size,
                "input_rows": "" if input_rows is None else int(input_rows),
                "output_path": str(output_path or ""),
                "output_rows": "" if output_rows is None else int(output_rows),
                "output_column": str(output_column or ""),
                "cluster_column": str(cluster_column or ""),
                "marker_count": len(audit.get("markers", [])),
                "markers_json": _audit_json_dumps(audit.get("markers", [])),
                "assigned_cluster_count": assigned_count,
                "assigned_phenotypes_json": _audit_json_dumps(assigned_map),
                "phenotype_counts_json": _audit_json_dumps(phenotype_counts),
                "parameters_json": _audit_json_dumps(audit.get("parameters", {})),
                "extra_json": _audit_json_dumps(extra or {}),
                "git_commit": _phenotyping_git_commit(project_root),
                "error": str(error or ""),
            }
        )

        log_path = get_phenotyping_audit_log_path(project_root, method_key)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a+", newline="", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            handle.seek(0, os.SEEK_END)
            needs_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=PHENOTYPING_AUDIT_COLUMNS, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        audit_logger.info(f"Reproducibility log updated: {log_path}")
        return log_path
    except Exception as exc:
        audit_logger.warning(f"Could not write phenotyping reproducibility log: {exc}")
        return None


def record_phenotyping_audit_event(
    project_root: Path,
    method_key: str,
    *,
    event_type: str,
    status: str = "success",
    input_csv: str | Path | None = None,
    output_path: str | Path | None = None,
    markers: list[str] | None = None,
    phenotype_map: dict | None = None,
    input_rows: int | None = None,
    output_rows: int | None = None,
    output_column: str | None = None,
    cluster_column: str | None = None,
    parameters: dict | None = None,
    extra: dict | None = None,
    error: str | None = None,
) -> Path | None:
    audit = begin_phenotyping_run_audit(
        project_root,
        method_key,
        input_csv=input_csv,
        markers=markers,
        parameters=parameters,
        event_type=event_type,
    )
    return finish_phenotyping_run_audit(
        audit,
        status=status,
        output_path=output_path,
        input_rows=input_rows,
        output_rows=output_rows,
        output_column=output_column,
        cluster_column=cluster_column,
        phenotype_map=phenotype_map,
        extra=extra,
        error=error,
    )


def render_phenotyping_run_log_controls(
    project_root: Path,
    method_key: str,
    method_name: str | None = None,
) -> bool:
    """Render a persisted log checkbox and an explicit settings snapshot action."""
    label = method_name or PHENOTYPING_METHOD_LABELS.get(method_key, method_key)
    # The method's detail log follows Tab 7's single "Record run log" switch at
    # the top of the page; there is no second box to keep in step with it.
    from spacemap_gui import run_log

    enabled = run_log.enabled("tab7")
    with st.expander(f"{label} run details and settings snapshots", expanded=False):
        if not enabled:
            st.caption(f"'{run_log.LABEL}' is off for this tab, so {label} runs are not being recorded.")
        log_path = get_phenotyping_audit_log_path(Path(project_root), method_key)
        st.caption(f"{label} log: `{log_path}`")
        if log_path.exists():
            st.download_button(
                "Download Current Log CSV",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="text/csv",
                key=f"{method_key}_audit_log_download",
            )
        if st.button(
            "Write Current Settings Snapshot",
            key=f"{method_key}_audit_snapshot_btn",
            disabled=not enabled,
            help="Append the currently saved method settings without running the analysis.",
        ):
            global_dir = str(st.session_state.get("global_pheno_dir", "")).strip()
            global_file = str(st.session_state.get("global_pheno_file", "")).strip()
            input_csv = Path(global_dir) / global_file if global_dir and global_file else None
            written = record_phenotyping_audit_event(
                Path(project_root),
                method_key,
                event_type="configuration_snapshot",
                input_csv=input_csv,
            )
            if written:
                from spacemap_gui.persistence import save_active_project_state

                save_active_project_state("tab7_phenotyping")
                st.success(f"Settings snapshot appended to {written}")
        return enabled


def render_method_intro(
    project_root: Path,
    method_key: str,
    title: str,
    summary: str,
    guide=None,
    *,
    notebook: str | None = None,
    log_label: str | None = None,
) -> None:
    """Open every Phenotyping subtab the same way.

    Title, a one-line plain-language summary, a collapsed "How it works" guide
    that names the matching notebook, then the run-log controls. ``guide`` is
    Markdown text, or a callable that draws its own content inside the expander.
    """
    st.subheader(title)
    st.caption(summary)
    if guide is not None or notebook:
        with st.expander(f"How {title.replace(' Phenotyping', '')} works", expanded=False):
            if callable(guide):
                guide()
            elif guide:
                st.markdown(guide)
            if notebook:
                st.caption(f"The same steps can be run as a script in `notebooks/{notebook}`.")
    render_phenotyping_run_log_controls(Path(project_root), method_key, log_label or title)


def render_data_filter_ui(
    available_columns: list[str],
    key_prefix: str,
    cached_params: dict | None = None,
    full_input_path: str | None = None,
    show_cell_id: bool = True,
) -> dict:
    """Render a consistent per-subtab data filtering UI.

    Parameters
    ----------
    available_columns : list[str]
        Column names from the active CSV file header.
    key_prefix : str
        Unique prefix for Streamlit widget keys (e.g. 'leiden', 'som').
    cached_params : dict or None
        Previously saved parameters for defaults.
    full_input_path : str or None
        Path to the active CSV; used to scan for a default filter column.
    show_cell_id : bool
        Whether to render a Cell ID Column selector.

    Returns
    -------
    dict
        Keys: 'cell_id_col', 'enable_filter', 'filter_col', 'filter_val'.
    """
    if cached_params is None:
        cached_params = {}

    result: dict = {}

    st.markdown("**Data Filtering**")

    # --- Cell ID Column ---
    if show_cell_id:
        default_cell_id = normalize_optional_column_name(cached_params.get('cell_id_col'))
        if default_cell_id not in available_columns:
            default_cell_id = 'cell_id' if 'cell_id' in available_columns else None

        cell_id_options = [""] + available_columns if available_columns else [""]
        idx_cell = cell_id_options.index(default_cell_id) if default_cell_id in cell_id_options else 0
        cell_id_col = st.selectbox(
            "Cell ID Column",
            cell_id_options,
            index=idx_cell,
            format_func=lambda value: value or "(None)",
            help=get_cell_id_column_help_text(),
            key=f"{key_prefix}_cell_id_col",
        )
        result['cell_id_col'] = normalize_optional_column_name(cell_id_col)
    else:
        result['cell_id_col'] = normalize_optional_column_name(cached_params.get('cell_id_col'))

    # --- Enable filter toggle ---
    enable_filter = st.checkbox(
        "Enable Row Filtering",
        value=cached_params.get('enable_filter', False),
        help=get_enable_row_filter_help_text(),
        key=f"{key_prefix}_enable_filter",
    )
    result['enable_filter'] = enable_filter

    # Determine default filter column: prefer 'tissue' if present
    filter_cols = [c for c in available_columns if c not in [result.get('cell_id_col', '')]]
    default_filter_col = 'tissue' if 'tissue' in filter_cols else (
        filter_cols[0] if filter_cols else 'tissue')
    cached_filter_col = cached_params.get('filter_col', default_filter_col)
    if cached_filter_col not in filter_cols and filter_cols:
        cached_filter_col = default_filter_col
    idx_filter = filter_cols.index(cached_filter_col) if filter_cols and cached_filter_col in filter_cols else 0

    filter_col = st.selectbox(
        "Filter Column",
        filter_cols if filter_cols else [default_filter_col],
        index=idx_filter,
        help=get_filter_column_help_text(),
        key=f"{key_prefix}_filter_col",
        disabled=not enable_filter,
    )

    # Determine default filter value: '1' if column is 'tissue', else ''
    default_filter_val = '1' if filter_col == 'tissue' else cached_params.get('filter_val', '1')
    filter_val = st.text_input(
        "Filter Value",
        value=cached_params.get('filter_val', default_filter_val),
        help=get_filter_value_help_text(),
        key=f"{key_prefix}_filter_val",
        disabled=not enable_filter,
    )
    result['filter_col'] = filter_col
    result['filter_val'] = filter_val

    return result


def apply_data_filter(df: pd.DataFrame, filter_settings: dict, logger_fn=None) -> pd.DataFrame:
    """Apply the filtering settings returned by render_data_filter_ui.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    filter_settings : dict
        Output from render_data_filter_ui.
    logger_fn : callable or None
        Logging function (receives a single string message).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if not filter_settings.get('enable_filter', False):
        if logger_fn:
            logger_fn("No filtering applied. Using full dataset.")
        return df

    f_col = filter_settings.get('filter_col')
    f_val = filter_settings.get('filter_val')

    if not f_col or f_col not in df.columns:
        if logger_fn:
            logger_fn(f"Filter column '{f_col}' not found in data. Using full dataset.")
        return df

    # Guard against None / empty filter value — this would match the string
    # "None" or "" in the data, almost certainly yielding 0 rows unintentionally.
    if f_val is None or (isinstance(f_val, str) and f_val.strip() == ""):
        if logger_fn:
            logger_fn(f"Filter value is empty for column '{f_col}'. Skipping filter.")
        return df

    original_len = len(df)
    try:
        if pd.api.types.is_numeric_dtype(df[f_col]):
            val_num = float(f_val)
            if pd.api.types.is_integer_dtype(df[f_col]):
                val_num = int(val_num)
            df = df[df[f_col] == val_num].copy()
        else:
            df = df[df[f_col].astype(str) == str(f_val)].copy()
    except Exception:
        df = df[df[f_col].astype(str) == str(f_val)].copy()

    if logger_fn:
        logger_fn(f"Filter applied: {f_col}=={f_val}: {original_len:,} -> {len(df):,} cells")

    return df


def get_marker_list(df):
    """Filter for likely marker columns, excluding metadata."""
    exclude = ['CellID', 'cell_id', 'X_centroid', 'Y_centroid', 'batch_id', 'tissue_roi', 
               'Area', 'MajorAxisLength', 'MinorAxisLength', 'Solidity', 'Extent', 'Orientation', 'Eccentricity',
               'x_centroid', 'y_centroid', 'ROI_ID', 'cluster', 'phenotype']
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed') and not c.endswith('_norm')]


MARKER_SOURCE_OPTIONS = (
    "Raw markers",
    "Corrected markers",
    "Raw and corrected markers",
)
MARKER_AFFIX_POSITIONS = ("Suffix", "Prefix")
DEFAULT_CORRECTED_MARKER_AFFIX = "_corr"


def infer_corrected_marker_rule(marker_columns: list[str]) -> tuple[str, str]:
    """Infer a corrected-marker prefix or suffix from paired marker columns.

    The normal SPACEMAP convention is ``CD45_corr``. The inference also accepts
    custom conventions such as ``CD45_croo`` or ``corr_CD45`` when both the raw
    and corrected forms are present in the selected CSV.
    """
    columns = [str(column) for column in marker_columns]
    column_set = set(columns)
    candidates: dict[tuple[str, str], int] = {}

    for raw_marker in columns:
        for corrected_marker in columns:
            if corrected_marker == raw_marker:
                continue
            if corrected_marker.startswith(raw_marker) and len(corrected_marker) > len(raw_marker):
                key = ("Suffix", corrected_marker[len(raw_marker):])
                candidates[key] = candidates.get(key, 0) + 1
            if corrected_marker.endswith(raw_marker) and len(corrected_marker) > len(raw_marker):
                key = ("Prefix", corrected_marker[: -len(raw_marker)])
                candidates[key] = candidates.get(key, 0) + 1

    if candidates:
        position, affix = max(candidates, key=lambda key: (candidates[key], key == ("Suffix", "_corr")))
        return position, affix

    for affix in ("_corr", "_corrected", "_batch_corrected", "_croo"):
        if any(column.endswith(affix) for column in column_set):
            return "Suffix", affix
    return "Suffix", DEFAULT_CORRECTED_MARKER_AFFIX


def is_corrected_marker_column(marker: str, affix_position: str, correction_affix: str) -> bool:
    """Return whether a marker column follows the configured corrected naming rule."""
    marker_name = str(marker)
    affix = str(correction_affix).strip()
    if not affix:
        return False
    if affix_position == "Prefix":
        return marker_name.startswith(affix) and marker_name != affix
    return marker_name.endswith(affix) and marker_name != affix


def corrected_marker_column_name(raw_marker: str, affix_position: str, correction_affix: str) -> str:
    """Build the expected corrected column name for one raw marker."""
    return f"{correction_affix}{raw_marker}" if affix_position == "Prefix" else f"{raw_marker}{correction_affix}"


def canonical_marker_base_name(marker: str, affix_position: str, correction_affix: str) -> str:
    """Return the biological/base marker name for either raw or corrected columns."""
    marker_name = str(marker).strip()
    affix = str(correction_affix).strip()
    if affix and is_corrected_marker_column(marker_name, affix_position, affix):
        if affix_position == "Prefix":
            return marker_name[len(affix):]
        return marker_name[:-len(affix)]
    return marker_name


def get_canonical_marker_panel(
    marker_columns: list[str],
    affix_position: str,
    correction_affix: str,
) -> list[str]:
    """Build an ordered panel of biological marker bases from raw/corrected columns.

    Raw columns are preferred as the canonical identity. Corrected-only columns
    are included by stripping the configured affix, so corrected datasets remain
    usable while still exposing a stable biological marker name to the user.
    """
    columns = [str(column) for column in marker_columns]
    corrected = {
        column for column in columns
        if is_corrected_marker_column(column, affix_position, correction_affix)
    }
    panel: list[str] = []
    seen: set[str] = set()
    for column in columns:
        base = canonical_marker_base_name(column, affix_position, correction_affix)
        if not base or base in seen:
            continue
        if column in corrected and base in columns:
            continue
        panel.append(base)
        seen.add(base)
    return panel


def canonicalize_marker_selection(
    markers: list[str] | None,
    available_bases: list[str],
    affix_position: str,
    correction_affix: str,
) -> list[str]:
    """Normalize legacy/raw/corrected selections to canonical marker bases."""
    available = set(available_bases)
    result: list[str] = []
    for marker in markers or []:
        base = canonical_marker_base_name(str(marker), affix_position, correction_affix)
        if base in available and base not in result:
            result.append(base)
    return result


def resolve_marker_selection(
    marker_bases: list[str],
    available_columns: list[str],
    source_mode: str,
    affix_position: str,
    correction_affix: str,
) -> tuple[list[str], list[str]]:
    """Resolve canonical bases to actual CSV columns and report unavailable values."""
    available = {str(column) for column in available_columns}
    resolved: list[str] = []
    missing: list[str] = []
    for base in marker_bases:
        base_name = str(base)
        if source_mode == "Corrected markers":
            candidate = corrected_marker_column_name(base_name, affix_position, correction_affix)
        else:
            candidate = base_name
        if candidate in available:
            resolved.append(candidate)
        else:
            missing.append(candidate)
    return resolved, missing


def get_marker_source_columns(
    marker_columns: list[str],
    source_mode: str,
    affix_position: str,
    correction_affix: str,
) -> list[str]:
    """Return marker columns allowed by a raw, corrected, or combined source mode."""
    corrected = [
        marker for marker in marker_columns
        if is_corrected_marker_column(marker, affix_position, correction_affix)
    ]
    raw = [marker for marker in marker_columns if marker not in corrected]
    if source_mode == "Corrected markers":
        return corrected
    if source_mode == "Raw and corrected markers":
        return list(marker_columns)
    return raw


def resolve_notebook_marker_defaults(
    corr_markers: list[str],
    non_corr_markers: list[str],
    available_markers: list[str],
    source_mode: str,
    affix_position: str,
    correction_affix: str,
) -> list[str]:
    """Map notebook marker bases to the configured raw/corrected source columns."""
    available = set(available_markers)
    defaults: list[str] = []

    def add(marker: str) -> None:
        if marker in available and marker not in defaults:
            defaults.append(marker)

    for marker in corr_markers:
        raw_marker = str(marker)
        corrected_marker = corrected_marker_column_name(raw_marker, affix_position, correction_affix)
        if source_mode == "Raw markers":
            add(raw_marker)
        elif source_mode == "Corrected markers":
            # Canonical shared panels contain base names, not resolved column
            # names. Keep the base when notebook defaults are calculated for
            # such a panel; the per-method resolver selects the actual
            # corrected column later.
            add(corrected_marker)
            if corrected_marker not in available and raw_marker in available:
                add(raw_marker)
        else:
            add(corrected_marker)
            add(raw_marker)

    # Notebook non-corrected values are usually morphology features and remain
    # raw even when expression markers use corrected columns.
    for marker in non_corr_markers:
        add(str(marker))
    return defaults


def marker_workspace_summary(source_mode: str, affix_position: str, correction_affix: str, preparation_note: str) -> str:
    """Create a short source and preparation label for UI captions and figures."""
    source_label = source_mode.replace(" markers", "")
    rule = f"{affix_position.lower()} `{correction_affix}`" if source_mode != "Raw markers" else "raw columns"
    note = str(preparation_note).strip() or "method-specific normalization"
    return f"Marker source: {source_label} ({rule}); preparation: {note}."


def marker_panel_shortcuts(
    available_markers: list[str],
    affix_position: str,
    correction_affix: str,
) -> dict[str, list[str]]:
    """Return ready-to-load raw/corrected marker lists using the configured rule."""
    columns = [str(column) for column in available_markers]
    corrected = [
        column for column in columns
        if is_corrected_marker_column(column, affix_position, correction_affix)
    ]
    non_corrected = [column for column in columns if column not in corrected]
    if corrected:
        corrected_bases = {
            canonical_marker_base_name(column, affix_position, correction_affix)
            for column in corrected
        }
        raw = [column for column in non_corrected if column in corrected_bases]
    else:
        # If the CSV contains no corrected counterparts, every available
        # marker is treated as a raw marker, as requested by the UI workflow.
        raw = non_corrected
    return {
        "All raw markers": raw,
        "All corrected markers": corrected,
        "All raw and corrected markers": columns,
    }


def get_named_marker_panels(
    available_markers: list[str],
    input_path: str | None,
) -> tuple[dict[str, list[str]], dict]:
    """Load named shared panels and migrate the previous canonical-panel format."""
    workspace = st.session_state.get("tab7_marker_workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}
    elif workspace.get("input_path") != str(input_path or ""):
        # Switching CSV used to discard every saved panel, so a user who added a
        # column to their table lost the whole marker workspace. Panels are
        # filtered against available_markers below, so carrying them across is
        # safe: markers absent from the new file drop out on their own, and
        # anything still present is kept.
        workspace = dict(workspace)
        workspace["input_path"] = str(input_path or "")

    affix_position = str(workspace.get("affix_position", "Suffix"))
    correction_affix = str(workspace.get("correction_affix", DEFAULT_CORRECTED_MARKER_AFFIX))
    available = [str(column) for column in available_markers]
    available_set = set(available)
    raw_panels = workspace.get("marker_panels", {})
    panels: dict[str, list[str]] = {}
    if isinstance(raw_panels, dict):
        for name, values in raw_panels.items():
            panel_name = str(name).strip()
            if not panel_name or not isinstance(values, (list, tuple)):
                continue
            valid_values = [str(value) for value in values if str(value) in available_set]
            if valid_values:
                panels[panel_name] = list(dict.fromkeys(valid_values))

    # Migrate the old shared canonical panel into one named raw panel.
    if not panels:
        legacy_values = workspace.get("selected_markers", [])
        if isinstance(legacy_values, list):
            legacy_raw, _ = resolve_marker_selection(
                canonicalize_marker_selection(
                    legacy_values,
                    get_canonical_marker_panel(available, affix_position, correction_affix),
                    affix_position,
                    correction_affix,
                ),
                available,
                "Raw markers",
                affix_position,
                correction_affix,
            )
            if legacy_raw:
                panels["shared_raw"] = legacy_raw

    # A fresh workspace starts with useful shortcut panels. Users can rename,
    # edit, or delete them; they are stored as actual CSV columns.
    if not panels:
        shortcuts = marker_panel_shortcuts(available, affix_position, correction_affix)
        if shortcuts["All raw markers"]:
            panels["all_raw"] = shortcuts["All raw markers"]
        if shortcuts["All corrected markers"]:
            panels["all_corrected"] = shortcuts["All corrected markers"]

    normalized_workspace = dict(workspace)
    normalized_workspace.update({
        "input_path": str(input_path or ""),
        "affix_position": affix_position,
        "correction_affix": correction_affix,
        "marker_panels": panels,
        "active_panel_name": (
            str(workspace.get("active_panel_name"))
            if str(workspace.get("active_panel_name", "")) in panels
            else (next(iter(panels), ""))
        ),
    })
    return panels, normalized_workspace


def get_marker_workspace_panel(
    available_markers: list[str],
    input_path: str | None,
) -> tuple[list[str], str]:
    """Return the configured shared marker panel for a selected phenotyping CSV."""
    panels, workspace = get_named_marker_panels(available_markers, input_path)
    st.session_state["tab7_marker_workspace"] = workspace
    active_name = str(workspace.get("active_panel_name", ""))
    active_panel = list(panels.get(active_name, next(iter(panels.values()), [])))
    note = str(workspace.get("preparation_note", "")).strip()
    panel_note = f"Shared panel `{active_name}` ({len(active_panel)} markers)"
    if note:
        panel_note = f"{panel_note}; preparation note: {note}"
    return active_panel, panel_note


# ==============================================================================
# Shared threshold methods: the input population here is cellular mean intensity.
# Keep the old function names for saved workflows, without separate mathematics.
from spacemap.thresholding_engine import THRESHOLD_METHODS, METHOD_ALIASES, calculate as _calculate_threshold
from functools import partial


def _auto_threshold(values, method, **kwargs):
    return _calculate_threshold(values, method, params=kwargs.get("params"))["threshold"]


AUTO_THRESHOLD_METHODS = {method: partial(_auto_threshold, method=method) for method in THRESHOLD_METHODS}
auto_threshold_otsu = AUTO_THRESHOLD_METHODS["Otsu"]
auto_threshold_triangle = AUTO_THRESHOLD_METHODS["Triangle"]
auto_threshold_li = AUTO_THRESHOLD_METHODS["Li"]
auto_threshold_yen = AUTO_THRESHOLD_METHODS["Yen"]
auto_threshold_mean = AUTO_THRESHOLD_METHODS["Mean"]
auto_threshold_isodata = AUTO_THRESHOLD_METHODS["IsoData"]
auto_threshold_minimum = AUTO_THRESHOLD_METHODS["Minimum"]
THRESHOLD_MODE_CHOICES = ["Predefined (CSV)", *THRESHOLD_METHODS]
_THRESHOLD_MODE_TO_KEY = {"Predefined (CSV)": "predefined", **{m: m for m in THRESHOLD_METHODS}, **METHOD_ALIASES}


def compute_auto_thresholds(
    df: pd.DataFrame,
    markers: list[str],
    method_name: str,
    batch_col: str | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """Compute auto-thresholds for markers, optionally per batch.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with expression values.
    markers : list[str]
        Marker columns to threshold.
    method_name : str
        Key into ``AUTO_THRESHOLD_METHODS`` (e.g. ``'Otsu'``).
    batch_col : str | None
        If provided, compute per-batch thresholds.

    Returns
    -------
    pd.DataFrame
        Threshold table with columns ``['batch_id'] + markers``.  One row
        per batch (or a single row if ``batch_col`` is None).
    """
    func = AUTO_THRESHOLD_METHODS.get(method_name)
    if func is None:
        raise ValueError(f"Unknown threshold method: {method_name}")

    markers_present = [m for m in markers if m in df.columns]
    if not markers_present:
        return pd.DataFrame()

    rows: list[dict] = []
    if batch_col and batch_col in df.columns:
        for bid in sorted(df[batch_col].dropna().astype(str).unique()):
            batch_df = df[df[batch_col].astype(str) == bid]
            row: dict = {"batch_id": bid, "Threshold_File": f"auto_{method_name}"}
            for m in markers_present:
                row[m] = func(batch_df[m].values, params=params)
            rows.append(row)
    else:
        row = {"batch_id": "all", "Threshold_File": f"auto_{method_name}"}
        for m in markers_present:
            row[m] = func(df[m].values, params=params)
        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# Shared Phenotype Constants & Expandable Phenotype List
# ==============================================================================

STANDARD_PHENOTYPES: list[str] = [
    "Fibroblast", "IMC", "DC", "Macrophage", "Endothelial", "B",
    "Epithelial", "Plasma", "Others", "Treg", "CD8", "CD4",
    "Other LC", "Granulocyte", "Monocyte", "CAF", "Nerve",
    "NK", "Mast cell", "Neutrophil", "Lymphocyte", "T cell", 'CD4 proliferative', 'CD4 stressed', 'CD4 PD-L1', 'CD4 activated', 'CD4 naïve', 'CD4 memory',
    'CD8 proliferative', 'CD8 stressed', 'CD8 PD-L1', 'CD8 activated', 'CD8 naïve', 'CD8 memory',


]

DEFAULT_PHENOTYPES: list[str] = [
    "Fibroblast", "IMC", "DC", "Macrophage", "Endothelial", "B",
    "Epithelial", "Plasma", "Treg", "CD8", "CD4", "Others",
]


def _get_custom_phenotypes_state_key() -> str:
    """Session-state key that holds user-added custom phenotypes (shared)."""
    return "tab7_custom_phenotypes"


def get_all_phenotype_options(include_unassigned: bool = True) -> list[str]:
    """Return the combined standard + custom phenotype list.

    The returned list always starts with ``'Unassigned'`` (if requested) and
    preserves insertion order for custom entries.

    Parameters
    ----------
    include_unassigned : bool
        Whether to prepend ``'Unassigned'`` to the list.

    Returns
    -------
    list[str]
        Full phenotype option list.
    """
    key = _get_custom_phenotypes_state_key()
    custom: list[str] = st.session_state.get(key, [])
    merged: list[str] = list(STANDARD_PHENOTYPES)
    for p in custom:
        if p and p not in merged:
            merged.append(p)
    if include_unassigned:
        return ["Unassigned"] + merged
    return merged


def render_add_custom_phenotype_ui(key_prefix: str) -> None:
    """Render a compact UI to add user-defined phenotype names.

    Newly added names are stored in ``st.session_state`` under a shared key
    so they are immediately available to **all** subtabs (Leiden, SOM,
    Resolve, etc.) without requiring separate tracking per subtab.

    Parameters
    ----------
    key_prefix : str
        A short prefix used to create unique Streamlit widget keys (e.g.
        ``'leiden'``, ``'som'``, ``'resolve'``).
    """
    state_key = _get_custom_phenotypes_state_key()
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        new_pheno = st.text_input(
            "Add custom phenotype",
            key=f"{key_prefix}_add_custom_pheno_input",
            placeholder="Enter new phenotype name…",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Add", key=f"{key_prefix}_add_custom_pheno_btn"):
            name = new_pheno.strip()
            if name:
                existing = get_all_phenotype_options(include_unassigned=False)
                if name not in existing:
                    st.session_state[state_key].append(name)
                    # Persist custom list to disk so it survives restarts
                    _update_tab7_state({"custom_phenotypes": st.session_state[state_key]})
                    st.success(f"Added **{name}** to phenotype list")
                else:
                    st.info(f"**{name}** already in the phenotype list")
            else:
                st.warning("Please enter a phenotype name")

    with st.expander("Bulk add custom phenotypes", expanded=False):
        st.caption("Paste one or many names separated by commas, semicolons, or new lines.")
        with st.form(f"{key_prefix}_bulk_add_custom_phenos_form", clear_on_submit=False):
            bulk_text = st.text_area(
                "Bulk phenotype names",
                key=f"{key_prefix}_bulk_add_custom_pheno_input",
                placeholder="Example:\nTumor_Subset_A\nTumor_Subset_B, Tumor_Subset_C",
                height=120,
                label_visibility="collapsed",
            )
            add_bulk = st.form_submit_button("Add All", type="primary")

        if add_bulk:
            raw = bulk_text.replace(";", "\n").replace(",", "\n")
            requested = [x.strip() for x in raw.splitlines() if x.strip()]

            if not requested:
                st.warning("No phenotype names found in bulk input.")
            else:
                existing = set(get_all_phenotype_options(include_unassigned=False))
                to_add: list[str] = []
                for name in requested:
                    if name not in existing and name not in to_add:
                        to_add.append(name)

                if to_add:
                    st.session_state[state_key].extend(to_add)
                    _update_tab7_state({"custom_phenotypes": st.session_state[state_key]})
                    st.success(f"Added {len(to_add)} custom phenotypes.")
                else:
                    st.info("All provided phenotype names already exist.")

    # On first load, restore persisted custom phenotypes from disk
    if not st.session_state[state_key]:
        saved = _load_tab7_state().get("custom_phenotypes", [])
        if saved:
            st.session_state[state_key] = saved

    custom = st.session_state.get(state_key, [])
    if custom:
        st.caption(f"Custom phenotypes: {', '.join(custom)}")

def plot_heatmap_figure(df_agg, cmap='bwr', title='Expression Heatmap'):
    """Generate a clean seaborn heatmap figure."""
    if df_agg.empty:
        return None
    
    # Row-normalization (Z-score)
    df_norm = df_agg.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0, axis=1)
    
    height = max(6, 0.5 * len(df_agg))
    fig, ax = plt.subplots(figsize=(12, height))
    sns.heatmap(df_norm, cmap=cmap, center=0, annot=False, ax=ax, linewidths=0.5, linecolor='white')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Markers", fontsize=12)
    ax.set_ylabel("Clusters / Phenotypes", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def save_plot(fig, path):
    if fig:
        fig.savefig(path, dpi=300, bbox_inches='tight')


def initialize_optional_column_widget_state(
    widget_key: str,
    available_columns: list[str],
    saved_value: str | None = None,
) -> str:
    """Initialize an optional column widget without forcing a non-blank default.

    Blank selection is meaningful for several Tab 7 inputs, especially batch
    grouping where an empty value means global processing.
    """
    options = [""] + list(available_columns or [])
    default_value = normalize_optional_column_name(saved_value) or ""
    if default_value not in options:
        default_value = ""

    if widget_key not in st.session_state:
        current_value = default_value
    else:
        current_value = normalize_optional_column_name(st.session_state.get(widget_key)) or ""
        if current_value not in options:
            current_value = default_value

    if widget_key not in st.session_state or st.session_state.get(widget_key) != current_value:
        st.session_state[widget_key] = current_value

    return current_value


def initialize_marker_selection_state(
    *,
    key_prefix: str,
    full_input_path: str | None,
    available_markers: list[str],
    fallback_markers: list[str],
    saved_markers: list[str] | None = None,
) -> list[str]:
    """Restore marker selections without overwriting imported state on file mismatch.

    Older settings exports intentionally omit ``*_last_file`` because the active
    CSV path is non-portable. The clustering subtabs therefore cannot treat a
    missing ``*_last_file`` as a signal to reset markers back to notebook
    defaults; otherwise imported manual selections are lost on the next rerun.
    """

    def _filter_valid(markers: list[str] | None) -> list[str]:
        seen: set[str] = set()
        filtered: list[str] = []
        for marker in markers or []:
            if marker in available_markers and marker not in seen:
                filtered.append(marker)
                seen.add(marker)
        return filtered

    confirmed_key = f"{key_prefix}_confirmed_markers"
    multiselect_key = f"{key_prefix}_marker_multiselect"
    last_file_key = f"{key_prefix}_last_file"

    saved_valid = _filter_valid(saved_markers)
    fallback_valid = _filter_valid(fallback_markers)
    current_confirmed = _filter_valid(st.session_state.get(confirmed_key, []))
    file_changed = bool(full_input_path) and st.session_state.get(last_file_key) != full_input_path

    if confirmed_key not in st.session_state:
        restored_markers = saved_valid or fallback_valid
        st.session_state[confirmed_key] = restored_markers
        st.session_state[multiselect_key] = list(restored_markers)
        if full_input_path:
            st.session_state[last_file_key] = full_input_path
        return restored_markers

    restored_markers = current_confirmed or saved_valid or fallback_valid
    if st.session_state.get(confirmed_key) != restored_markers:
        st.session_state[confirmed_key] = restored_markers

    if multiselect_key not in st.session_state or file_changed:
        st.session_state[multiselect_key] = list(restored_markers)

    if full_input_path:
        st.session_state[last_file_key] = full_input_path

    return restored_markers


def _legacy_render_method_selection(
    *,
    key_prefix: str,
    method_name: str,
    full_input_path: str | None,
    available_columns: list[str],
    shared_panel: list[str],
    marker_workspace_note: str,
    fallback_markers: list[str] | None = None,
    saved_markers: list[str] | None = None,
    config_state_key: str | None = None,
) -> tuple[list[str], list[str], str]:
    """Render an offline per-method marker subset/source editor.

    The widget edits canonical marker bases (for example ``CD4``). The saved
    source mode independently resolves those bases to ``CD4`` or ``CD4_corr``
    at run time, so each phenotyping method can use a different representation.
    """
    workspace = st.session_state.get("tab7_marker_workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}
    affix_position = str(workspace.get("affix_position", "Suffix"))
    correction_affix = str(workspace.get("correction_affix", DEFAULT_CORRECTED_MARKER_AFFIX))
    panel = [str(marker) for marker in shared_panel]

    base_key = f"{key_prefix}_marker_bases"
    confirmed_key = f"{key_prefix}_confirmed_markers"
    multiselect_key = f"{key_prefix}_marker_multiselect"
    source_key = f"{key_prefix}_marker_value_source"
    source_widget_key = f"{key_prefix}_marker_value_source_widget"
    source_options = ("Raw markers", "Corrected markers")

    saved_bases = canonicalize_marker_selection(
        saved_markers,
        panel,
        affix_position,
        correction_affix,
    )
    if base_key not in st.session_state:
        legacy_selection = st.session_state.get(confirmed_key, [])
        st.session_state[base_key] = canonicalize_marker_selection(
            legacy_selection or saved_bases or panel or fallback_markers,
            panel,
            affix_position,
            correction_affix,
        )

    current_source = st.session_state.get(source_key)
    if current_source not in source_options:
        legacy_selection = st.session_state.get(confirmed_key, []) or saved_markers or []
        corrected_count = sum(
            1 for marker in legacy_selection
            if is_corrected_marker_column(str(marker), affix_position, correction_affix)
        )
        corrected_only_columns = any(
            is_corrected_marker_column(str(column), affix_position, correction_affix)
            and canonical_marker_base_name(str(column), affix_position, correction_affix)
            not in {str(candidate) for candidate in available_columns}
            for column in available_columns
        )
        inferred_source = (
            "Corrected markers"
            if (legacy_selection and corrected_count == len(legacy_selection)) or corrected_only_columns
            else "Raw markers"
        )
        st.session_state[source_key] = inferred_source
    if source_widget_key not in st.session_state:
        st.session_state[source_widget_key] = st.session_state[source_key]

    initialize_marker_selection_state(
        key_prefix=key_prefix,
        full_input_path=full_input_path,
        available_markers=panel,
        fallback_markers=canonicalize_marker_selection(
            fallback_markers,
            panel,
            affix_position,
            correction_affix,
        ),
        saved_markers=st.session_state.get(base_key, saved_bases),
    )
    st.session_state[base_key] = canonicalize_marker_selection(
        st.session_state.get(f"{key_prefix}_confirmed_markers", []),
        panel,
        affix_position,
        correction_affix,
    )
    st.markdown("##### Marker Selection")
    st.caption(
        f"Choose a subset from the shared panel for {method_name}. "
        "Changes are staged locally until you click Save Marker Settings."
    )
    source_mode = st.radio(
        "Values for this method",
        source_options,
        key=source_widget_key,
        horizontal=True,
        help="Raw shows original columns. Corrected shows the configured counterparts, such as CD4_corr, when they exist.",
    )

    # The visible selector contains the actual CSV columns for the selected
    # source.  Internally we keep canonical bases, so switching from Raw to
    # Corrected displays CD4_corr while retaining CD4 as the biological ID.
    display_columns, _ = resolve_marker_selection(
        panel,
        available_columns,
        source_mode,
        affix_position,
        correction_affix,
    )
    draft_values = st.session_state.get(multiselect_key)
    draft_bases = canonicalize_marker_selection(
        draft_values if isinstance(draft_values, list) else st.session_state[base_key],
        panel,
        affix_position,
        correction_affix,
    )
    st.session_state[multiselect_key] = resolve_marker_selection(
        draft_bases,
        available_columns,
        source_mode,
        affix_position,
        correction_affix,
    )[0]

    with st.form(f"{key_prefix}_marker_form", clear_on_submit=False):
        action_col1, action_col2, action_col3 = st.columns(3)
        use_shared_panel = action_col1.form_submit_button("Use Shared Panel")
        select_all = action_col2.form_submit_button("Select All Listed")
        clear_all = action_col3.form_submit_button("Clear All")
        if use_shared_panel or select_all:
            st.session_state[multiselect_key] = list(display_columns)
        if clear_all:
            st.session_state[multiselect_key] = []

        selected_columns = st.multiselect(
            "Marker subset (resolved CSV columns)",
            display_columns,
            key=multiselect_key,
            help=(
                f"Choose the {source_mode.lower()} columns for {method_name}. "
                "The saved JSON also retains their canonical biological marker bases."
            ),
        )
        save_settings = st.form_submit_button("Save Marker Settings", type="primary")

    if save_settings:
        selected_bases = canonicalize_marker_selection(
            selected_columns,
            panel,
            affix_position,
            correction_affix,
        )
        resolved, missing = resolve_marker_selection(
            selected_bases,
            available_columns,
            source_mode,
            affix_position,
            correction_affix,
        )
        st.session_state[base_key] = selected_bases
        st.session_state[source_key] = source_mode
        st.session_state[confirmed_key] = resolved
        st.session_state[f"{key_prefix}_marker_resolution"] = {
            "bases": list(selected_bases),
            "source_mode": source_mode,
            "resolved_columns": list(resolved),
            "missing_columns": list(missing),
        }
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab7_phenotyping")
        if missing:
            st.warning(
                f"Saved {len(selected_bases)} marker bases, but {len(missing)} requested "
                f"{source_mode.lower()} column(s) were not found."
            )
        else:
            st.success(f"Saved {len(resolved)} {source_mode.lower()} marker columns for {method_name}.")

    selected_bases = list(st.session_state.get(base_key, []))
    source_mode = st.session_state.get(source_key, "Raw markers")
    resolved, missing = resolve_marker_selection(
        selected_bases,
        available_columns,
        source_mode,
        affix_position,
        correction_affix,
    )
    st.session_state[confirmed_key] = resolved
    st.caption(
        f"Saved subset: {len(selected_bases)} bases → {len(resolved)} columns "
        f"({source_mode.lower()}). Displayed columns use the selected source."
    )
    if missing:
        st.warning(f"Unavailable for this source: {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}")
    return selected_bases, resolved, marker_workspace_note


def render_method_marker_selection(
    *,
    key_prefix: str,
    method_name: str,
    full_input_path: str | None,
    available_columns: list[str],
    shared_panel: list[str],
    marker_workspace_note: str,
    fallback_markers: list[str] | None = None,
    saved_markers: list[str] | None = None,
    config_state_key: str | None = None,
) -> tuple[list[str], list[str], str]:
    """Choose a named shared panel and stage a local marker subset."""
    panels, workspace = get_named_marker_panels(available_columns, full_input_path)
    st.session_state["tab7_marker_workspace"] = workspace
    panel_names = list(panels)
    if not panel_names:
        st.warning("No named shared marker panels are available. Create one in the common marker area first.")
        return [], [], marker_workspace_note

    panel_key = f"{key_prefix}_marker_panel_name"
    draft_key = f"{key_prefix}_marker_panel_draft"
    confirmed_key = f"{key_prefix}_confirmed_markers"
    multiselect_key = f"{key_prefix}_marker_multiselect"
    resolution_key = f"{key_prefix}_marker_resolution"

    selected_name = str(st.session_state.get(panel_key, ""))
    if selected_name not in panels:
        selected_name = str(workspace.get("active_panel_name", ""))
    if selected_name not in panels:
        selected_name = panel_names[0]
    st.session_state[panel_key] = selected_name

    available_set = set(str(column) for column in available_columns)
    saved_valid = [
        str(marker)
        for marker in (saved_markers or st.session_state.get(confirmed_key, []))
        if str(marker) in available_set
    ]
    if confirmed_key not in st.session_state:
        st.session_state[confirmed_key] = saved_valid or list(panels[selected_name])
    if draft_key not in st.session_state:
        st.session_state[draft_key] = list(st.session_state[confirmed_key]) or list(panels[selected_name])

    def _load_selected_panel() -> None:
        name = st.session_state.get(panel_key, panel_names[0])
        st.session_state[draft_key] = list(panels.get(name, []))
        st.session_state[multiselect_key] = list(st.session_state[draft_key])

    st.markdown("##### Marker Panel Selection")
    st.caption(
        f"Choose a saved shared marker panel for {method_name}, then trim it locally if needed. "
        "Changes are staged until Save Marker Settings."
    )
    selected_name = st.selectbox(
        "Shared marker panel",
        panel_names,
        key=panel_key,
        on_change=_load_selected_panel,
        help="Panels are created and named in the common marker workspace. Each subtab can use a different panel and subset.",
    )
    panel_columns = list(panels[selected_name])
    draft_columns = [
        str(marker)
        for marker in st.session_state.get(draft_key, panel_columns)
        if str(marker) in panel_columns
    ]
    st.session_state[draft_key] = draft_columns
    if multiselect_key not in st.session_state:
        st.session_state[multiselect_key] = list(draft_columns)
    else:
        st.session_state[multiselect_key] = [
            str(marker)
            for marker in st.session_state[multiselect_key]
            if str(marker) in panel_columns
        ]

    with st.form(f"{key_prefix}_marker_form", clear_on_submit=False):
        action_col1, action_col2, action_col3 = st.columns(3)
        load_panel = action_col1.form_submit_button("Load Full Panel")
        select_all = action_col2.form_submit_button("Select All Listed")
        clear_all = action_col3.form_submit_button("Clear All")
        if load_panel or select_all:
            st.session_state[multiselect_key] = list(panel_columns)
        if clear_all:
            st.session_state[multiselect_key] = []

        selected_columns = st.multiselect(
            "Marker subset",
            panel_columns,
            key=multiselect_key,
            help="These are the actual CSV columns stored in the selected named panel, including names such as CD4_corr.",
        )
        save_settings = st.form_submit_button("Save Marker Settings", type="primary")

    if save_settings:
        selected_columns = [str(marker) for marker in selected_columns if str(marker) in panel_columns]
        st.session_state[draft_key] = list(selected_columns)
        st.session_state[confirmed_key] = list(selected_columns)
        st.session_state[resolution_key] = {
            "panel_name": selected_name,
            "resolved_columns": list(selected_columns),
        }
        if config_state_key:
            canonical_config = st.session_state.get(config_state_key)
            if not isinstance(canonical_config, dict):
                canonical_config = {}
                st.session_state[config_state_key] = canonical_config
            canonical_config.update(
                {
                    "markers": list(selected_columns),
                    "marker_panel_name": selected_name,
                    "marker_resolution": dict(st.session_state[resolution_key]),
                }
            )
        workspace = dict(st.session_state.get("tab7_marker_workspace", workspace))
        workspace["active_panel_name"] = selected_name
        st.session_state["tab7_marker_workspace"] = workspace
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab7_phenotyping")
        st.success(f"Saved {len(selected_columns)} markers from panel `{selected_name}` for {method_name}.")

    confirmed = [
        str(marker)
        for marker in st.session_state.get(confirmed_key, [])
        if str(marker) in panel_columns
    ]
    st.session_state[confirmed_key] = confirmed
    st.caption(f"Saved subset: {len(confirmed)} of {len(panel_columns)} markers from `{selected_name}`.")
    return confirmed, confirmed, f"Shared marker panel `{selected_name}`."


def marker_selection_ui(full_input_path, key_prefix):
    """Render a marker-selection table without rerunning on each click.

    This uses a Streamlit form so checkbox toggles do not trigger reruns.
    Markers are only applied when the user clicks "Save markers".
    """
    if not full_input_path:
        st.warning("No input file selected.")
        return []

    try:
        df_head = pd.read_csv(full_input_path, nrows=1)
        avail_markers = get_marker_list(df_head)
    except Exception:
        st.error("Could not read input file header.")
        return []

    confirmed_key = f"{key_prefix}_confirmed_markers"
    draft_key = f"{key_prefix}_marker_df_draft"
    form_key = f"{key_prefix}_marker_form"

    default_markers = [m for m in avail_markers if 'corr' in m]

    # Initialize / refresh draft based on the current file header.
    if draft_key not in st.session_state:
        prev = set(st.session_state.get(confirmed_key, default_markers))
        st.session_state[draft_key] = pd.DataFrame({
            "Select": [m in prev for m in avail_markers],
            "Marker": avail_markers,
        })
    else:
        current_markers = set(st.session_state[draft_key]["Marker"].tolist())
        if current_markers != set(avail_markers):
            prev = set(st.session_state.get(confirmed_key, default_markers))
            st.session_state[draft_key] = pd.DataFrame({
                "Select": [m in prev for m in avail_markers],
                "Marker": avail_markers,
            })

    confirmed = st.session_state.get(confirmed_key, [])
    st.caption(f"Confirmed markers: {len(confirmed)}")

    with st.form(key=form_key, clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
        bulk_corr = c1.form_submit_button("Select all *_corr")
        bulk_all = c2.form_submit_button("Select all")
        bulk_none = c3.form_submit_button("Clear all")
        save_btn = c4.form_submit_button("Save markers", type="primary")

        edited_df = st.data_editor(
            st.session_state[draft_key],
            column_config={
                "Select": st.column_config.CheckboxColumn("Include", default=False),
                "Marker": st.column_config.TextColumn("Marker Name", disabled=True),
            },
            disabled=["Marker"],
            hide_index=True,
            height=300,
            key=f"{key_prefix}_editor_widget",
        )

        # Apply bulk actions on submit.
        if bulk_corr:
            edited_df["Select"] = edited_df["Marker"].apply(lambda x: 'corr' in str(x))
        elif bulk_all:
            edited_df["Select"] = True
        elif bulk_none:
            edited_df["Select"] = False

        # Persist the latest draft table (but do NOT change confirmed unless Save clicked).
        st.session_state[draft_key] = edited_df

        selected_now = edited_df.loc[edited_df["Select"], "Marker"].tolist()
        st.caption(f"Selected (not yet saved): {len(selected_now)}")

        if save_btn:
            st.session_state[confirmed_key] = selected_now

    # Only return the confirmed markers (i.e. what you saved).
    return st.session_state.get(confirmed_key, [])

@st.cache_resource(show_spinner=False)
def build_leiden_heatmap_figure(
    data_hash,
    df_input,
    norm_cols,
    cluster_x,
    cluster_y,
    show_counts,
    cluster_col='leiden_cluster',
    cluster_label='Leiden Cluster',
    z_min=-2,
    z_max=2,
    marker_source_note: str | None = None,
):
    """
    Cached function to build the base Leiden heatmap figure.
    Does strict data processing and base figure object creation.
    """
    try:
        import plotly.graph_objects as go
        from scipy.cluster.hierarchy import linkage, dendrogram

        # 1. Aggregation
        if cluster_col not in df_input.columns:
            return None, [], [], {}, pd.DataFrame()

        valid_cols = [c for c in norm_cols if c in df_input.columns]
        if not valid_cols:
            return None, [], [], {}, pd.DataFrame()

        # Filter to only numeric columns to avoid aggregation errors
        numeric_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df_input[c])]
        if not numeric_cols:
            return None, [], [], {}, pd.DataFrame()

        # Filter out NaN clusters for grouping
        df_use = df_input[df_input[cluster_col].notna()]
        if df_use.empty:
            return None, [], [], {}, pd.DataFrame()

        from spacemap.cluster_profiles import summarize
        heatmap_data, heatmap_z = summarize(df_use, cluster_col, numeric_cols, "mean")
        cell_counts = df_use.groupby(cluster_col).size()
        
        # Clean column names
        clean_names = {col: get_analysis_marker_label(col) for col in heatmap_data.columns}
        heatmap_data.rename(columns=clean_names, inplace=True)
        heatmap_z.rename(columns=clean_names, inplace=True)
        
        # Transpose
        heatmap_data = heatmap_data.T
        
        # Z-Score
        heatmap_z = heatmap_z.T
        
        # Clustering
        cluster_order = list(heatmap_z.columns)
        marker_order = list(heatmap_z.index)
        
        if cluster_x and len(cluster_order) > 1:
            try:
                col_linkage = linkage(heatmap_z.T, method='ward', metric='euclidean')
                col_dendro = dendrogram(col_linkage, no_plot=True)
                cluster_order = [heatmap_z.columns[i] for i in col_dendro['leaves']]
            except Exception:
                pass

        if cluster_y and len(marker_order) > 1:
            try:
                row_linkage = linkage(heatmap_z, method='ward', metric='euclidean')
                row_dendro = dendrogram(row_linkage, no_plot=True)
                marker_order = [heatmap_z.index[i] for i in row_dendro['leaves']]
            except Exception:
                 pass
        
        heatmap_z_ordered = heatmap_z.loc[marker_order, cluster_order]
        
        # Build Figure
        def _fmt_cluster_label(c):
            try:
                return str(int(c))
            except Exception:
                return str(c)

        if show_counts:
            x_labels = [f"{_fmt_cluster_label(c)}<br>({int(cell_counts.get(c, 0)):,})" for c in cluster_order]
        else:
            x_labels = [_fmt_cluster_label(c) for c in cluster_order]
            
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_z_ordered.values,
            x=x_labels,
            y=marker_order,
            colorscale='RdBu_r',
            zmid=0, zmin=z_min, zmax=z_max,
            colorbar=dict(title='Z-score', titleside='right', x=1.02),
            hovertemplate=f'{cluster_label}: %{{x}}<br>Marker: %{{y}}<br>Z-score: %{{z:.2f}}<extra></extra>'
        ))
        
        marker_source_note = marker_source_note or build_marker_source_note(
            valid_cols,
            method_label=cluster_label,
        )
        fig.update_layout(
            title=dict(
                text=f"{cluster_label} marker expression<br><sup>{marker_source_note}</sup>",
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(title=cluster_label, side='bottom', tickangle=0),
            yaxis=dict(title='Markers', side='left', autorange='reversed'),
            height=max(600, len(marker_order) * 20),
            margin=dict(l=150, r=100, t=90, b=100),
            plot_bgcolor='white',
            hovermode='closest',
            dragmode='select',
            clickmode='event+select'
        )
        
        def _norm_cluster_id(c):
            if pd.isna(c):
                return c
            try:
                return int(c)
            except Exception:
                return str(c)

        return fig, [_norm_cluster_id(c) for c in cluster_order], marker_order, cell_counts.to_dict(), heatmap_z_ordered

    except Exception as e:
        _log_console(f"Error in build_leiden_heatmap_figure: {e}", level="error")
        return None, [], [], {}, pd.DataFrame()


def render_phenotype_statistics(
    df_results: pd.DataFrame,
    cluster_col: str,
    phenotype_map: dict,
    heatmap_z_ordered: pd.DataFrame,
    cluster_label: str,
    score_label: str = "Avg Z-Score",
):
    """Render phenotype summary table using the same logic as Leiden."""
    st.divider()
    st.markdown("#### Phenotype Statistics")

    if not phenotype_map:
        st.info("No phenotypes assigned yet.")
        return

    pheno_stats = []
    df_summary = df_results.copy()
    df_summary['temp_pheno'] = df_summary[cluster_col].map(phenotype_map).fillna('Unassigned')

    if 'temp_pheno' not in df_summary.columns:
        st.error("Could not process phenotype summary.")
        return

    if isinstance(heatmap_z_ordered, pd.Series):
        heatmap_z_ordered = heatmap_z_ordered.to_frame()

    total_len = len(df_summary)
    for pheno, grp in df_summary.groupby('temp_pheno'):
        if pd.isna(pheno):
            continue

        c_count = len(grp)
        c_pct = (c_count / total_len) * 100
        clusters_in_pheno = sorted(grp[cluster_col].unique())
        clusters_str = ", ".join(map(str, clusters_in_pheno))

        # Top Markers (weighted average of Z-scores)
        top5_str = "N/A"
        if heatmap_z_ordered is not None and not heatmap_z_ordered.empty:
            avail_cols = heatmap_z_ordered.columns
            relevant_cols = [c for c in clusters_in_pheno if c in avail_cols]
            if not relevant_cols:
                relevant_cols = [c for c in clusters_in_pheno if pd.notna(c) and (str(c) in avail_cols or int(c) in avail_cols)]

            if relevant_cols:
                pheno_vec = heatmap_z_ordered[relevant_cols].mean(axis=1)
                top5 = pheno_vec.nlargest(5)
                
                # Strip suffixes for display (User Request)
                top5_parts = []
                for m, v in top5.items():
                    clean_m = str(m).replace('_corr', '').replace('_norm', '').replace('_unmixed', '').replace('_resolve', '')
                    top5_parts.append(f"{clean_m} ({v:.2f})")
                top5_str = ", ".join(top5_parts)

        pheno_stats.append({
            "Phenotype": pheno,
            "Cell Count": c_count,
            "% Total": f"{c_pct:.1f}%",
            f"{cluster_label}s": clusters_str,
            f"Top Markers ({score_label})": top5_str
        })

    if pheno_stats:
        df_pheno_stats = pd.DataFrame(pheno_stats)
        st.dataframe(df_pheno_stats, width="stretch", hide_index=True)
    else:
        st.info("No phenotypes assigned yet.")


class PhenotypeDotPlot:
    """Matplotlib dot plot with an aligned cell-count bar chart.

    ``mean_expression`` and ``fraction_percent`` use phenotype labels as the
    index and marker labels as columns.  The class is deliberately independent
    of Streamlit so the same figure can be used by Tab 7 and Tab 9 exports.
    """

    def __init__(
        self,
        mean_expression: pd.DataFrame,
        fraction_percent: pd.DataFrame,
        cell_counts: pd.Series,
        *,
        title: str = "Phenotype marker dot plot",
        color_vmin: float = 0.0,
        color_vmax: float = 2.0,
        colorbar_label: str = "Mean expression in group",
        positive_thresholds: pd.Series | None = None,
        color_map: str = "viridis",
    ) -> None:
        self.mean_expression = mean_expression.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.fraction_percent = fraction_percent.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.fraction_percent = self.fraction_percent.reindex(
            index=self.mean_expression.index,
            columns=self.mean_expression.columns,
        ).fillna(0.0).clip(0.0, 100.0)
        self.cell_counts = cell_counts.reindex(self.mean_expression.index).fillna(0).astype(float)
        self.title = title
        self.color_vmin = float(color_vmin)
        self.color_vmax = float(color_vmax)
        if not np.isfinite(self.color_vmin):
            self.color_vmin = 0.0
        if not np.isfinite(self.color_vmax) or self.color_vmax <= self.color_vmin:
            self.color_vmax = self.color_vmin + 1.0
        self.colorbar_label = colorbar_label
        self.color_map = str(color_map or "viridis")
        self.positive_thresholds = (
            positive_thresholds.reindex(self.mean_expression.columns).astype(float)
            if isinstance(positive_thresholds, pd.Series)
            else pd.Series(0.0, index=self.mean_expression.columns, dtype=float)
        )

    @staticmethod
    def _clean_marker_label(marker: str) -> str:
        return get_analysis_marker_label(marker)

    @staticmethod
    def _is_normalized_column(column: str) -> bool:
        return bool(re.search(r"_norm(?:_|$)|_resolve$", str(column), flags=re.IGNORECASE))

    @classmethod
    def _derive_color_scale(
        cls,
        mean_expression: pd.DataFrame,
        expression_columns: list[str],
        normalized_hint: bool | None = None,
    ) -> tuple[float, float, str]:
        """Derive a readable color scale from the values actually plotted.

        The previous implementation always used 0..2.5. That made ordinary
        raw intensities saturate at the brightest color and made every positive
        cell appear fully expressed. Normalized columns now use their observed
        signed/non-negative range; raw columns use a robust data-driven range.
        """
        values = mean_expression.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0, 1.0, "Mean expression in group"

        normalized = (
            any(cls._is_normalized_column(column) for column in expression_columns)
            if normalized_hint is None
            else bool(normalized_hint)
        )
        lower = float(np.nanpercentile(values, 1))
        upper = float(np.nanpercentile(values, 99))

        if lower < 0:
            extent = max(abs(lower), abs(upper), 0.5)
            bound = max(1.0, float(np.ceil(extent * 2.0) / 2.0))
            label = "Mean normalized expression in group" if normalized else "Mean expression in group"
            return -bound, bound, label

        upper = max(upper, 1e-6)
        bound = float(np.ceil(upper * 2.0) / 2.0)
        if normalized:
            # Keep the familiar 0, 1, 2+ reference scale when the selected
            # transformation produces a positive expression-like range, but
            # expand it if the actual normalized values require more room.
            bound = max(2.0, bound)
            return 0.0, bound, "Mean normalized expression in group"

        # Raw/input-scale markers can have arbitrary units, so never impose a
        # biological-looking 0..2 scale on them.
        return 0.0, max(bound, 1.0), "Mean input-scale expression in group"

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        group_col: str,
        *,
        phenotype_map: dict | None = None,
        marker_columns: list[str] | None = None,
        marker_labels: list[str] | None = None,
        positive_threshold: float | None = None,
        normalization_label: str | None = None,
        title: str = "Phenotype marker dot plot",
        color_map: str = "viridis",
    ) -> "PhenotypeDotPlot | None":
        """Build dot-plot statistics from cell-level expression data."""
        if group_col not in df.columns:
            return None

        available = [str(column) for column in df.columns]
        requested = [str(column) for column in (marker_columns or marker_labels or [])]
        if not requested:
            requested = [
                column for column in available
                if column != group_col and pd.api.types.is_numeric_dtype(df[column])
            ]
        explicit_columns = marker_columns is not None

        def normalized_name(value: str) -> str:
            return re.sub(r"(?:_corr|_norm|_unmixed|_resolve)+$", "", str(value), flags=re.IGNORECASE).lower()

        expression_columns: list[str] = []
        display_labels: list[str] = []
        for requested_index, requested_name in enumerate(requested):
            candidates: list[str] = []
            requested_base = normalized_name(requested_name)
            matching = [
                column for column in available
                if column != group_col
                and pd.api.types.is_numeric_dtype(df[column])
                and normalized_name(column) == requested_base
            ]
            # When the heatmap supplies clean marker labels (e.g. ``CD4``),
            # prefer the pipeline's transformed column (``CD4_norm`` or
            # ``CD4_norm_resolve``) over a same-named raw column. The old
            # exact-match-first logic silently plotted raw intensities instead.
            matching.sort(
                key=lambda column: (
                    0 if cls._is_normalized_column(column) else 1,
                    0 if str(column).lower().endswith("_corr") else 1,
                    str(column),
                )
            )
            if explicit_columns and requested_name in available:
                # An explicit marker column is authoritative. Clean labels
                # still use the transformed-column preference above.
                candidates.append(requested_name)
            candidates.extend(matching)
            selected = next((column for column in candidates if column in available and column not in expression_columns), None)
            if selected is not None:
                expression_columns.append(selected)
                display_label = (
                    str(marker_labels[requested_index])
                    if marker_labels and requested_index < len(marker_labels)
                    else cls._clean_marker_label(requested_name)
                )
                display_labels.append(
                    cls._clean_marker_label(display_label)
                )

        if not expression_columns:
            return None

        work = df[[group_col] + expression_columns].copy()

        def map_group(value: object) -> str:
            if pd.isna(value):
                return "Unassigned"
            if not phenotype_map:
                return str(value)
            if value in phenotype_map:
                return str(phenotype_map[value])
            value_string = str(value)
            if value_string in phenotype_map:
                return str(phenotype_map[value_string])
            try:
                return str(phenotype_map.get(int(value), "Unassigned"))
            except (TypeError, ValueError):
                return "Unassigned"

        work["__dotplot_group"] = work[group_col].map(map_group)
        work = work[work["__dotplot_group"].notna()]
        if work.empty:
            return None

        numeric = work[expression_columns].apply(pd.to_numeric, errors="coerce")
        numeric.columns = display_labels
        grouped = numeric.assign(__dotplot_group=work["__dotplot_group"].to_numpy())
        mean_expression = grouped.groupby("__dotplot_group", sort=False)[display_labels].mean()
        thresholds = pd.Series(
            0.0 if positive_threshold is None else float(positive_threshold),
            index=display_labels,
            dtype=float,
        )
        fraction_percent = (
            grouped[display_labels].gt(thresholds, axis="columns")
            .assign(__dotplot_group=work["__dotplot_group"].to_numpy())
            .groupby("__dotplot_group", sort=False)[display_labels]
            .mean()
            .mul(100.0)
        )
        cell_counts = work["__dotplot_group"].value_counts(sort=False)

        if phenotype_map:
            preferred_order = []
            for value in phenotype_map.values():
                label = str(value)
                if label in mean_expression.index and label not in preferred_order:
                    preferred_order.append(label)
            preferred_order.extend(label for label in mean_expression.index if label not in preferred_order)
            mean_expression = mean_expression.loc[preferred_order]
            fraction_percent = fraction_percent.loc[preferred_order]
            cell_counts = cell_counts.loc[preferred_order]

        normalized_hint = None
        if normalization_label is not None:
            label_text = normalization_label.lower()
            if any(token in label_text for token in ("batch-norm", "z-score", "pipeline norm", "normalized")):
                normalized_hint = True
            elif any(token in label_text for token in ("no additional", "input-scale")):
                normalized_hint = False

        color_vmin, color_vmax, colorbar_label = cls._derive_color_scale(
            mean_expression,
            expression_columns,
            normalized_hint=normalized_hint,
        )
        return cls(
            mean_expression,
            fraction_percent,
            cell_counts,
            title=title,
            color_vmin=color_vmin,
            color_vmax=color_vmax,
            colorbar_label=colorbar_label,
            positive_thresholds=thresholds,
            color_map=color_map,
        )

    @classmethod
    def mock(cls, seed: int = 7) -> "PhenotypeDotPlot":
        """Create deterministic mock data for documentation and development."""
        rng = np.random.default_rng(seed)
        groups = ["B", "CD4", "CD8", "DC", "Endothelial", "Epithelial"]
        markers = ["CD31", "E-cadherin", "CD20", "CD4", "CD68"]
        means = pd.DataFrame(rng.uniform(0.05, 2.4, (len(groups), len(markers))), index=groups, columns=markers)
        fractions = pd.DataFrame(rng.uniform(5, 100, means.shape), index=groups, columns=markers)
        counts = pd.Series(rng.integers(30_000, 1_000_000, len(groups)), index=groups)
        return cls(means, fractions, counts, title="Mock phenotype marker dot plot")

    @staticmethod
    def _format_count(value: float) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}m"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return f"{value:.0f}"

    @staticmethod
    def _fraction_dot_sizes(
        fractions: np.ndarray | list[float] | tuple[float, ...],
        minimum_area: float,
        maximum_area: float,
    ) -> np.ndarray:
        """Map percentages to scatter areas shared by the plot and legend."""
        clipped = np.clip(np.asarray(fractions, dtype=float), 0.0, 100.0)
        return minimum_area + (maximum_area - minimum_area) * clipped / 100.0

    def plot(
        self,
        *,
        show_grid: bool = True,
        show_count_bars: bool = True,
        show_count_labels: bool = True,
        show_colorbar: bool = True,
        show_size_legend: bool = True,
        figsize: tuple[float, float] | None = None,
        marker_tick_rotation: float = 90.0,
        dot_fill_fraction: float = 0.72,
        count_panel_gap: float = 0.06,
        lower_panel_gap: float = 0.06,
    ) -> plt.Figure:
        """Render a geometry-aware dot plot, count bars, and legends.

        The maximum dot diameter is derived from the rendered row and column
        spacing, so a 100% dot cannot overlap an adjacent grid position.  The
        exact same area mapping is reused by the vertical fraction legend.
        """
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        n_groups = len(self.mean_expression.index)
        n_markers = len(self.mean_expression.columns)
        marker_labels = [str(marker) for marker in self.mean_expression.columns]
        max_marker_label_length = max((len(label) for label in marker_labels), default=1)
        marker_rotation = float(np.clip(marker_tick_rotation, 0.0, 90.0))
        rotation_factor = abs(np.sin(np.deg2rad(marker_rotation)))
        # Reserve only the clearance that rotated marker labels actually need;
        # ``lower_panel_gap`` is the user's additional spacing preference.
        label_clearance = float(
            np.clip(0.18 + rotation_factor * max_marker_label_length * 0.018, 0.18, 0.58)
        )
        safe_dot_fill = float(np.clip(dot_fill_fraction, 0.25, 0.92))
        safe_count_gap = float(np.clip(count_panel_gap, 0.0, 0.8))
        # Lower-panel spacing is expressed in physical inches so it behaves
        # consistently at every DPI and figure size.
        safe_lower_gap = float(np.clip(lower_panel_gap, 0.0, 1.5))
        adaptive_figsize = (
            max(11.0, n_markers * 0.48 + 6.0),
            max(7.5, n_groups * 0.62 + 4.0 + label_clearance),
        )
        fig = plt.figure(figsize=figsize or adaptive_figsize)
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=(5.0, 1.35),
            height_ratios=(5.0, 2.0),
            hspace=label_clearance + 0.02,
            wspace=safe_count_gap,
        )
        ax = fig.add_subplot(grid[0, 0])
        bar_ax = fig.add_subplot(grid[0, 1], sharey=ax)
        colorbar_ax = fig.add_subplot(grid[1, 0])
        legend_ax = fig.add_subplot(grid[1, 1])

        y_positions = np.arange(n_groups)
        x_positions = np.arange(n_markers)
        xx, yy = np.meshgrid(x_positions, y_positions)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            marker_labels,
            rotation=marker_rotation,
            ha="right" if marker_rotation < 90 else "center",
            va="top",
        )
        ax.tick_params(axis="x", pad=7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(group) for group in self.mean_expression.index])
        ax.invert_yaxis()
        ax.set_xlim(-0.5, max(n_markers - 0.5, 0.5))
        ax.set_ylim(max(n_groups - 0.5, 0.5), -0.5)
        ax.set_xlabel("Marker", labelpad=6)
        ax.set_ylabel("Cell type")
        ax.set_title(self.title, pad=12)
        ax.spines[["top", "right"]].set_visible(False)

        fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.07)
        # Resolve the axes in physical units. Scatter sizes are points squared,
        # so using the real grid pitch makes the overlap cap independent of
        # DPI, figure size, and the number of markers or phenotypes.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        marker_label_artists = [ax.xaxis.label, *ax.get_xticklabels()]
        marker_label_boxes = [
            artist.get_window_extent(renderer=renderer)
            for artist in marker_label_artists
            if artist.get_visible() and artist.get_text()
        ]
        lowest_marker_label_y = min(
            (box.y0 for box in marker_label_boxes),
            default=ax.get_window_extent().y0,
        )
        axis_box = ax.get_window_extent()
        x_pitch_points = axis_box.width * 72.0 / fig.dpi / max(n_markers, 1)
        y_pitch_points = axis_box.height * 72.0 / fig.dpi / max(n_groups, 1)
        maximum_diameter = max(1.5, min(x_pitch_points, y_pitch_points) * safe_dot_fill)
        if show_size_legend:
            legend_box = legend_ax.get_window_extent()
            legend_pitch_points = legend_box.height * 72.0 / fig.dpi / 5.4
            maximum_diameter = min(maximum_diameter, max(1.5, legend_pitch_points * 0.82))
        minimum_diameter = min(maximum_diameter, max(1.5, maximum_diameter * 0.14))
        minimum_dot_area = minimum_diameter**2
        maximum_dot_area = maximum_diameter**2

        if show_grid:
            ax.scatter(
                xx.ravel(),
                yy.ravel(),
                s=max(1.5, minimum_dot_area * 0.75),
                color="#c9c9c9",
                edgecolors="none",
                zorder=1,
                label="Available marker/group",
            )

        mean_values = self.mean_expression.to_numpy(dtype=float)
        fraction_values = self.fraction_percent.to_numpy(dtype=float)
        dot_sizes = self._fraction_dot_sizes(
            fraction_values,
            minimum_dot_area,
            maximum_dot_area,
        )
        norm = Normalize(vmin=self.color_vmin, vmax=self.color_vmax)
        try:
            cmap = plt.get_cmap(self.color_map)
        except ValueError:
            try:
                cmap = sns.color_palette(self.color_map, as_cmap=True)
            except Exception:
                cmap = plt.get_cmap("viridis")
        ax.scatter(
            xx.ravel(),
            yy.ravel(),
            s=dot_sizes.ravel(),
            c=np.clip(mean_values, self.color_vmin, self.color_vmax).ravel(),
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            zorder=2,
        )

        if show_count_bars:
            group_colors = [plt.get_cmap("tab20")(index % 20) for index in range(n_groups)]
            counts = self.cell_counts.to_numpy(dtype=float)
            max_count = max(float(counts.max()) if len(counts) else 1.0, 1.0)
            bar_ax.barh(y_positions, counts, color=group_colors, edgecolor="#444444", linewidth=0.5, height=0.62)
            bar_ax.set_xlim(0, max_count * 1.30)
            bar_ax.set_xticks([])
            bar_ax.tick_params(axis="y", left=False, labelleft=False)
            bar_ax.set_xlabel("Cell count", labelpad=8)
            if show_count_labels:
                for y_value, count in zip(y_positions, counts):
                    bar_ax.text(
                        max_count * 1.03,
                        y_value,
                        self._format_count(count),
                        va="center",
                        ha="left",
                        fontsize=9,
                    )
            bar_ax.spines[["top", "right", "left"]].set_visible(False)
        else:
            bar_ax.axis("off")

        if show_colorbar:
            colorbar_position = colorbar_ax.get_position()
            compact_height = min(colorbar_position.height * 0.28, 0.035)
            desired_top = (
                lowest_marker_label_y - safe_lower_gap * fig.dpi
            ) / max(fig.bbox.height, 1.0)
            compact_bottom = max(0.09, desired_top - compact_height)
            colorbar_ax.set_position(
                [
                    colorbar_position.x0,
                    compact_bottom,
                    colorbar_position.width,
                    compact_height,
                ]
            )
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            scalar_map.set_array([])
            colorbar = fig.colorbar(scalar_map, cax=colorbar_ax, orientation="horizontal")
            if self.color_vmin < 0:
                bound = max(abs(self.color_vmin), abs(self.color_vmax))
                ticks = [-bound, 0.0, bound]
                ticklabels = [f"{value:g}" for value in ticks]
            elif self.color_vmax <= 1.0:
                ticks = [0.0, self.color_vmax / 2.0, self.color_vmax]
                ticklabels = [f"{value:g}" for value in ticks]
            elif self.colorbar_label.startswith("Mean normalized"):
                ticks = []
                for value in (0.0, 1.0, 2.0, self.color_vmax):
                    if self.color_vmin <= value <= self.color_vmax and not any(np.isclose(value, existing) for existing in ticks):
                        ticks.append(value)
                ticklabels = ["2+" if value == self.color_vmax and value >= 2.0 else f"{value:g}" for value in ticks]
            else:
                ticks = np.linspace(self.color_vmin, self.color_vmax, 4)
                ticklabels = [f"{value:g}" for value in ticks]
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(ticklabels)
            colorbar.set_label(self.colorbar_label, labelpad=8)
            colorbar.ax.xaxis.set_ticks_position("bottom")
        else:
            colorbar_ax.axis("off")

        legend_ax.axis("off")
        if show_size_legend:
            legend_fractions = np.asarray([0.0, 25.0, 50.0, 75.0, 100.0])
            legend_sizes = self._fraction_dot_sizes(
                legend_fractions,
                minimum_dot_area,
                maximum_dot_area,
            )
            legend_y = np.arange(len(legend_fractions), dtype=float)
            legend_ax.set_xlim(0.0, 1.0)
            legend_ax.set_ylim(4.65, -1.25)
            legend_ax.scatter(
                np.full_like(legend_y, 0.22),
                legend_y,
                s=legend_sizes,
                color="#777777",
                edgecolors="none",
                clip_on=False,
            )
            for y_value, fraction in zip(legend_y, legend_fractions):
                legend_ax.text(
                    0.48,
                    y_value,
                    f"{fraction:.0f}%",
                    va="center",
                    ha="left",
                    fontsize=9,
                )
            legend_ax.text(
                0.02,
                -1.12,
                "Fraction of cells\nin group (%)",
                va="bottom",
                ha="left",
                fontsize=10,
            )

        return fig


def render_publication_heatmap(
    heatmap_z_ordered: pd.DataFrame,
    phenotype_map: dict,
    key_prefix: str,
    vmin: float | None = None,
    vmax: float | None = None,
    df_results: pd.DataFrame | None = None,
    group_col: str | None = None,
    marker_columns: list[str] | None = None,
    marker_source_note: str | None = None,
    normalization_label: str | None = None,
    dot_positive_threshold_default: float = 0.0,
):
    """Render a selectable publication heatmap or phenotype dot plot."""
    st.divider()
    with st.expander("Generate Publication Heatmap or Dot Plot", expanded=False):
        st.markdown("Create a high-resolution heatmap or phenotype dot plot.")
        marker_source_note = marker_source_note or build_marker_source_note(
            marker_columns or [str(marker) for marker in (heatmap_z_ordered.index if heatmap_z_ordered is not None else [])],
            method_key=key_prefix,
        )
        st.caption(marker_source_note)
        method_label = _PHENOTYPING_METHOD_LABELS.get(key_prefix, key_prefix.replace("_", " ").title())
        dotplot_marker_labels = (
            [get_analysis_marker_label(column) for column in marker_columns]
            if marker_columns
            else [get_analysis_marker_label(marker) for marker in heatmap_z_ordered.index]
        )
        dotplot_normalization_label = normalization_label
        if dotplot_normalization_label is None:
            if key_prefix in {"mrml", "mlc"}:
                config = st.session_state.get(f"{key_prefix}_config", {})
                if isinstance(config, dict) and config.get("use_z_normalization", False):
                    dotplot_normalization_label = "Batch-aware Z-score normalization"
            elif marker_columns and any(PhenotypeDotPlot._is_normalized_column(column) for column in marker_columns):
                dotplot_normalization_label = "Pipeline normalization"

        figure_type = st.radio(
            "Figure type",
            ["Heatmap", "Dot Plot with Cell Count Bar Chart"],
            horizontal=True,
            key=f"{key_prefix}_pub_figure_type",
            help="Choose the publication figure generated by this section.",
        )

        if figure_type == "Dot Plot with Cell Count Bar Chart":
            dot_c1, dot_c2, dot_c3, dot_c4 = st.columns(4)
            dot_dpi = dot_c1.number_input("DPI", 100, 600, 300, key=f"{key_prefix}_dot_dpi")
            show_dot_grid = dot_c2.checkbox("Show grey grid dots", value=True, key=f"{key_prefix}_dot_show_grid")
            show_dot_bars = dot_c3.checkbox("Show cell-count bars", value=True, key=f"{key_prefix}_dot_show_bars")
            show_dot_count_labels = dot_c4.checkbox("Show count labels", value=True, key=f"{key_prefix}_dot_show_count_labels")
            dot_c5, dot_c6 = st.columns(2)
            show_dot_colorbar = dot_c5.checkbox("Show mean-expression colorbar", value=True, key=f"{key_prefix}_dot_show_colorbar")
            show_dot_size_legend = dot_c6.checkbox("Show fraction-size legend", value=True, key=f"{key_prefix}_dot_show_size_legend")
            dot_positive_threshold = st.number_input(
                "Marker positivity threshold",
                value=float(dot_positive_threshold_default),
                step=0.1,
                key=f"{key_prefix}_dot_positive_threshold",
                help="Dot size is the percentage of cells whose selected marker value is above this threshold. Color remains the mean expression after the selected normalization.",
            )
            st.caption("Dot size = percentage of cells above the positivity threshold; dot color = mean normalized marker expression.")

            if st.button("Generate Figure", type="primary", key=f"{key_prefix}_pub_generate_dot_btn"):
                if df_results is None or not group_col:
                    st.error("Dot plot data are unavailable for this method.")
                elif heatmap_z_ordered is None or heatmap_z_ordered.empty or not phenotype_map:
                    st.error("No expression data or phenotype assignments are available for the dot plot.")
                else:
                    dot_plot = PhenotypeDotPlot.from_dataframe(
                        df_results,
                        group_col,
                        phenotype_map=phenotype_map,
                        marker_columns=marker_columns,
                        marker_labels=dotplot_marker_labels,
                        positive_threshold=float(dot_positive_threshold),
                        normalization_label=dotplot_normalization_label,
                        title=f"{method_label} phenotype marker dot plot",
                    )
                    if dot_plot is None:
                        st.error("Could not match numeric marker columns for the dot plot.")
                    else:
                        figure = dot_plot.plot(
                            show_grid=show_dot_grid,
                            show_count_bars=show_dot_bars,
                            show_count_labels=show_dot_count_labels,
                            show_colorbar=show_dot_colorbar,
                            show_size_legend=show_dot_size_legend,
                        )
                        st.pyplot(figure)
                        import io
                        buffer = io.BytesIO()
                        figure.savefig(buffer, format="png", dpi=dot_dpi, bbox_inches="tight")
                        st.download_button(
                            "Download Dot Plot",
                            buffer.getvalue(),
                            f"{key_prefix}_phenotype_dot_plot.png",
                            "image/png",
                            key=f"{key_prefix}_pub_dot_download",
                        )
                        plt.close(figure)
            return

        gc1, gc2, gc3, gc4 = st.columns(4)
        dpi_val = gc1.number_input("DPI", 100, 600, 300, key=f"{key_prefix}_pub_dpi")
        cmap_val = gc2.selectbox("Color Map", ["bwr", "coolwarm", "viridis", "RdBu_r"], index=3, key=f"{key_prefix}_pub_cmap")
        orient_val = gc3.radio("Phenotypes on:", ["X Axis", "Y Axis"], index=1, key=f"{key_prefix}_pub_orient")
        cluster_x = gc4.checkbox("Cluster X axis", value=True, key=f"{key_prefix}_pub_cluster_x")
        cluster_y = gc4.checkbox("Cluster Y axis", value=True, key=f"{key_prefix}_pub_cluster_y")

        if st.button("Generate Figure", type="primary", key=f"{key_prefix}_pub_generate"):
            if heatmap_z_ordered is not None and not heatmap_z_ordered.empty and phenotype_map:
                try:
                    pmap = phenotype_map
                    pheno_z_df = heatmap_z_ordered.copy()

                    from spacemap.cluster_profiles import named_profiles
                    df_plot = named_profiles(pheno_z_df.T, pmap)

                    # Sanitize non-finite values before clustering (required by scipy linkage)
                    df_plot = df_plot.replace([np.inf, -np.inf], np.nan)

                    bad_phenotypes = []
                    bad_markers = []

                    # Drop phenotypes (rows) with no finite values at all
                    if not df_plot.empty:
                        row_finite = np.isfinite(df_plot.to_numpy(dtype=float)).any(axis=1)
                        bad_phenotypes = df_plot.index[~row_finite].tolist()
                        df_plot = df_plot.loc[row_finite]

                    # Drop markers (columns) with no finite values at all
                    if not df_plot.empty:
                        col_finite = np.isfinite(df_plot.to_numpy(dtype=float)).any(axis=0)
                        bad_markers = df_plot.columns[~col_finite].tolist()
                        df_plot = df_plot.loc[:, col_finite]

                    # Fill remaining sparse NaNs after aggregation
                    if not df_plot.empty and df_plot.isna().any().any():
                        df_plot = df_plot.fillna(0)

                    if bad_phenotypes or bad_markers:
                        msg_parts = []
                        if bad_phenotypes:
                            msg_parts.append(f"dropped phenotypes with all non-finite values: {', '.join(map(str, bad_phenotypes))}")
                        if bad_markers:
                            msg_parts.append(f"dropped markers with all non-finite values: {', '.join(map(str, bad_markers))}")
                        st.warning("Publication heatmap data cleanup: " + " | ".join(msg_parts))

                    if df_plot.empty:
                        st.error("No finite values available for publication heatmap after cleanup.")
                        return

                    if orient_val == "X Axis":
                        df_plot = df_plot.T

                    if df_plot.empty:
                        st.error("No finite values available for publication heatmap after orientation.")
                        return

                    do_row_cluster, do_col_cluster = cluster_y, cluster_x

                    from spacemap.heatmap import publication_heatmap
                    figure = publication_heatmap(df_plot,
                        f"{method_label} phenotype expression heatmap\n{marker_source_note}",
                        {"cmap": cmap_val, "cluster_y": do_row_cluster, "cluster_x": do_col_cluster,
                         "vmin": vmin, "vmax": vmax})

                    st.pyplot(figure)

                    import io
                    buf = io.BytesIO()
                    figure.savefig(buf, format="png", dpi=dpi_val, bbox_inches='tight')
                    st.download_button("Download Figure", buf.getvalue(), f"{key_prefix}_phenotype_heatmap.png", "image/png")
                except Exception as e:
                    st.error(f"Error plotting: {e}")
            else:
                st.error("No expression data or assignments available.")


def render_unified_csv_export_options(
    output_col: str,
    update_cluster_cols: list,
    key_prefix: str,
) -> list[str]:
    """Render and persist the optional columns written to the unified CSV.

    The phenotype output column is always written. Auxiliary result columns
    are opt-in so a normal save does not add QC, cluster, score, or UMAP
    bookkeeping columns unless the user explicitly requests them.
    """
    optional_columns = list(dict.fromkeys(
        str(column)
        for column in update_cluster_cols
        if str(column) and str(column) != str(output_col)
    ))
    selection_key = f"{key_prefix}_unified_export_columns"
    saved_selection = st.session_state.get(selection_key, [])
    if not isinstance(saved_selection, list):
        saved_selection = []
    saved_selection = [column for column in saved_selection if column in optional_columns]
    st.session_state[selection_key] = saved_selection

    st.markdown("##### Unified CSV Export Columns")
    st.caption(
        f"Always exported: `{output_col}`. Select optional columns to add to the unified CSV. "
        "These choices are saved in the project JSON."
    )
    st.checkbox(
        f"{output_col} (required phenotype output)",
        value=True,
        disabled=True,
        key=f"{key_prefix}_unified_export_required",
    )

    if not optional_columns:
        st.info("No optional result columns are available for this method.")
        return []

    checkbox_values: list[str] = []
    checkbox_columns = st.columns(2)
    for index, column in enumerate(optional_columns):
        checkbox_key = (
            f"{key_prefix}_unified_export_checkbox_"
            f"{re.sub(r'[^A-Za-z0-9]+', '_', column).strip('_').lower()}"
        )
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = column in saved_selection
        with checkbox_columns[index % 2]:
            if st.checkbox(column, key=checkbox_key):
                checkbox_values.append(column)

    if checkbox_values != saved_selection:
        st.session_state[selection_key] = checkbox_values

    save_export_settings = st.button(
        "Save Export Column Settings",
        key=f"{key_prefix}_save_export_column_settings_btn",
        help="Save the current optional-column checklist to the project JSON.",
    )
    if save_export_settings:
        st.session_state[selection_key] = checkbox_values
        from spacemap_gui.persistence import save_active_project_state
        save_active_project_state("tab7_phenotyping")
        st.success("Unified CSV export-column settings saved to the project JSON.")

    return checkbox_values


class PendingSaveBlocked(RuntimeError):
    """Saved-but-unwritten changes must be resolved before another save."""


def _save_assignments_as_pending(
    input_path: Path,
    df_results_linkable: pd.DataFrame,
    result_anchor_col: str,
    using_internal_anchor: bool,
    update_cols: list,
    cluster_col: str,
    output_col: str,
    get_pheno,
    *,
    source_note: str = "",
) -> list[str] | None:
    """Save assignments as pending columns instead of rewriting the whole CSV.

    Same result as the full rewrite: cells present in the results get the new
    value, every other cell keeps its current one, matched through the cell ID
    column. Returns the columns saved, or None when the fast path does not apply
    (no real cell ID column, or duplicate IDs), in which case the caller rewrites
    the CSV as before.
    """
    from spacemap import dataio, pending_columns

    if using_internal_anchor or result_anchor_col == TAB7_INTERNAL_CELL_ID_COL:
        return None
    anchor = result_anchor_col
    available = dataio.columns(input_path)
    if anchor not in available:
        return None
    if pending_columns.has_stale(input_path):
        raise PendingSaveBlocked(
            "Changes saved earlier no longer match the CSV (it was rewritten since). Open the "
            "file panel at the top of the tab and re-apply or discard them, then save again."
        )
    targets = [column for column in update_cols if column in df_results_linkable.columns]
    write_phenotype = cluster_col in df_results_linkable.columns
    wanted = [anchor] + [
        column for column in dict.fromkeys(targets + ([output_col] if write_phenotype else []))
        if column in available and column != anchor
    ]
    base = dataio.load(input_path, columns=wanted)
    if base[anchor].duplicated().any():
        return None
    result_ids = df_results_linkable[result_anchor_col]
    mask = base[anchor].isin(result_ids.values)
    ids_in_analysis = base.loc[mask, anchor]

    new_columns: dict[str, pd.Series] = {}
    for column in targets:
        id_to_value = dict(zip(result_ids, df_results_linkable[column]))
        series = base[column].copy() if column in base.columns else pd.Series(np.nan, index=base.index)
        series.loc[mask] = ids_in_analysis.map(id_to_value)
        new_columns[column] = series
    if write_phenotype:
        id_to_pheno = dict(zip(result_ids, df_results_linkable[cluster_col].apply(get_pheno)))
        series = (base[output_col].copy() if output_col in base.columns
                  else pd.Series(np.nan, index=base.index)).astype(object)
        series.loc[mask] = ids_in_analysis.map(id_to_pheno)
        new_columns[output_col] = series
    if not new_columns:
        return None
    return pending_columns.save(
        input_path,
        new_columns,
        anchor_column=anchor,
        anchor_values=base[anchor],
        source_note=source_note,
    )


def render_save_assignments(
    df_results: pd.DataFrame,
    phenotype_map: dict,
    output_col: str,
    project_root: Path,
    key_prefix: str,
    cluster_col: str,
    update_cluster_cols: list,
    cell_id_col: str = "cell_id",
    input_path_override: str | Path | None = None,
):
    """Render Save Assignments button and write outputs.

    update_cluster_cols: list of cluster columns to update in unified input file.
    """
    st.divider()

    selected_update_cols = render_unified_csv_export_options(
        output_col=output_col,
        update_cluster_cols=update_cluster_cols,
        key_prefix=key_prefix,
    )

    # Option for local copy
    save_local_copy = st.checkbox(
        "Save separate local copy", 
        value=False, 
        help="If checked, saves a full copy of the results to the outputs folder. Keep unchecked to only update the Unified Input File (faster).",
        key=f"{key_prefix}_save_local_copy"
    )

    if st.button("Save All Assignments to Disk", type="primary", width="stretch", key=f"{key_prefix}_save_all_btn"):
        input_path_for_log = (
            Path(input_path_override) if input_path_override else None
        )
        if input_path_for_log is None and st.session_state.get('global_pheno_file') and st.session_state.get('global_pheno_dir'):
            input_path_for_log = Path(st.session_state.global_pheno_dir) / st.session_state.global_pheno_file
        marker_candidates = (
            st.session_state.get(f"{key_prefix}_confirmed_markers")
            or st.session_state.get(f"{key_prefix}_marker_cols")
            or st.session_state.get(f"{key_prefix}_markers")
            or []
        )
        assignment_audit = begin_phenotyping_run_audit(
            Path(project_root),
            key_prefix,
            input_csv=input_path_for_log,
            markers=list(marker_candidates) if isinstance(marker_candidates, (list, tuple)) else [],
            event_type="assignment_save",
        )
        saved_output_path = None
        try:
            cur_map = phenotype_map
            df_results_linkable, result_anchor_col, using_internal_anchor = ensure_cell_id_anchor(df_results, cell_id_col)

            def get_pheno(clust):
                if pd.isna(clust):
                    return "Unassigned"

                if clust in cur_map:
                    return cur_map.get(clust, "Unassigned")

                clust_str = str(clust)
                if clust_str in cur_map:
                    return cur_map.get(clust_str, "Unassigned")

                try:
                    return cur_map.get(int(clust), "Unassigned")
                except Exception:
                    return "Unassigned"

            # Identify Input File
            input_path = Path(input_path_override) if input_path_override else None
            if input_path is None and st.session_state.get('global_pheno_file') and st.session_state.get('global_pheno_dir'):
                input_path = Path(st.session_state.global_pheno_dir) / st.session_state.global_pheno_file

            updated_input = False

            fast_saved = None
            if input_path and input_path.exists():
                try:
                    fast_saved = _save_assignments_as_pending(
                        input_path,
                        df_results_linkable,
                        result_anchor_col,
                        using_internal_anchor,
                        selected_update_cols,
                        cluster_col,
                        output_col,
                        get_pheno,
                        source_note=f"Tab 7 {PHENOTYPING_METHOD_LABELS.get(key_prefix, key_prefix)}",
                    )
                except PendingSaveBlocked as blocked:
                    st.error(str(blocked))
                    return
            if fast_saved:
                st.success(
                    f"Saved {', '.join(f'`{name}`' for name in fast_saved)} in seconds. Tabs 7, 8 and 9 "
                    "already show the new values. They are written into the CSV itself when you press "
                    "**Write pending changes into the CSV** in the file panel at the top of the tab."
                )
                saved_output_path = input_path
                updated_input = True
            elif input_path and input_path.exists():
                st.info(f"Reading input file: {input_path}")
                df_in = pd.read_csv(input_path)
                df_in_linkable, input_anchor_col, _ = ensure_cell_id_anchor(df_in, cell_id_col)

                if input_anchor_col in df_in_linkable.columns and result_anchor_col in df_results_linkable.columns:
                    mask_in_analysis = df_in_linkable[input_anchor_col].isin(df_results_linkable[result_anchor_col].values)

                    # Build maps
                    for col in selected_update_cols:
                        if col in df_results_linkable.columns:
                            id_to_col = dict(zip(df_results_linkable[result_anchor_col], df_results_linkable[col]))
                            df_in_linkable.loc[mask_in_analysis, col] = df_in_linkable.loc[mask_in_analysis, input_anchor_col].map(id_to_col)

                    if cluster_col in df_results_linkable.columns:
                        # The phenotype output must work even when the user
                        # did not opt in to exporting the cluster column.
                        # Resolve the label through the shared cell anchor
                        # instead of requiring that cluster_col already exist
                        # in the unified input file.
                        id_to_pheno = dict(
                            zip(
                                df_results_linkable[result_anchor_col],
                                df_results_linkable[cluster_col].apply(get_pheno),
                            )
                        )
                        df_in_linkable.loc[mask_in_analysis, output_col] = (
                            df_in_linkable.loc[mask_in_analysis, input_anchor_col].map(id_to_pheno)
                        )

                    if TAB7_INTERNAL_CELL_ID_COL in df_in_linkable.columns:
                        df_in_linkable = df_in_linkable.drop(columns=[TAB7_INTERNAL_CELL_ID_COL])

                    df_in_linkable.to_csv(input_path, index=False)
                    st.success(
                        f"Updated unified input file with '{output_col}'"
                        + (f" and {selected_update_cols}." if selected_update_cols else ".")
                    )
                    st.caption(f"Updated file: {input_path}")
                    saved_output_path = input_path
                    if using_internal_anchor:
                        st.caption("Used a temporary row-order anchor because no Cell ID column was selected.")
                    updated_input = True
                else:
                    st.warning("Could not link cells between the current results and the unified input file.")
            else:
                if save_local_copy:
                    st.warning("Input file path not found/valid. Saving local copy only.")
                else:
                    st.error("Input file path not found/valid and local save is OFF. Nothing will be saved.")

            # Standard output
            if save_local_copy:
                output_dir = project_root / f'outputs/phenotyping/{key_prefix}'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Inform user about the second save operation (since it can take time)
                st.caption(f"Writing local output copy to: {output_dir}...")
                
                df_local = df_results_linkable.copy()
                df_local[output_col] = df_local[cluster_col].apply(get_pheno)
                if TAB7_INTERNAL_CELL_ID_COL in df_local.columns:
                    df_local = df_local.drop(columns=[TAB7_INTERNAL_CELL_ID_COL])
                df_local.to_csv(output_dir / f'{key_prefix}_clustering_phenotyped.csv', index=False)
                saved_output_path = output_dir / f'{key_prefix}_clustering_phenotyped.csv'

                import json
                map_file = output_dir / 'phenotype_mapping.json'
                with open(map_file, 'w') as f:
                    json.dump(cur_map, f)

                st.success(f"Local results saved to {output_dir}")

            if updated_input or save_local_copy:
                try:
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                except Exception as ex:
                    logger.warning(f"Failed to auto-save active project state: {ex}")
                finish_phenotyping_run_audit(
                    assignment_audit,
                    status="success",
                    output_path=saved_output_path,
                    input_rows=len(df_results),
                    output_rows=len(df_results),
                    output_column=output_col,
                    cluster_column=cluster_col,
                    phenotype_map=cur_map,
                    extra={
                        "updated_unified_csv": bool(updated_input),
                        "saved_local_copy": bool(save_local_copy),
                        "optional_export_columns": list(selected_update_cols),
                    },
                )

            if not updated_input and not save_local_copy:
                st.warning("No files were saved. Please check 'Save separate local copy' or ensure global input file is configured.")
                finish_phenotyping_run_audit(
                    assignment_audit,
                    status="no_output",
                    input_rows=len(df_results),
                    output_column=output_col,
                    cluster_column=cluster_col,
                    phenotype_map=cur_map,
                )

        except Exception as e:
            st.error(f"Error saving: {e}")
            logger.error(f"Save error: {e}")
            finish_phenotyping_run_audit(
                assignment_audit,
                status="failed",
                input_rows=len(df_results),
                output_column=output_col,
                cluster_column=cluster_col,
                phenotype_map=phenotype_map,
                error=str(e),
            )


def render_export_split_data(
    df_results: pd.DataFrame,
    output_col: str,
    phenotype_map: dict,
    project_root: Path,
    key_prefix: str,
    cluster_col: str,
    cell_id_col: str = "cell_id",
):
    """Render Export Split Data section (shared)."""
    st.divider()
    with st.expander("Export Split Data (Batch Split)", expanded=False):
        st.markdown("Split the dataset into multiple CSV files based on a column (e.g., 'batch_id').")

        st.info("Tip: Ensure you have saved your assignments to the unified CSV above before splitting.")

        df_results_linkable, result_anchor_col, _ = ensure_cell_id_anchor(df_results, cell_id_col)
        split_cols = [column for column in df_results_linkable.columns.tolist() if column != TAB7_INTERNAL_CELL_ID_COL]
        default_split = 'batch_id' if 'batch_id' in split_cols else (split_cols[0] if split_cols else None)

        sp_c1, sp_c2 = st.columns([1, 1])
        split_col = sp_c1.selectbox("Split Column", split_cols, index=split_cols.index(default_split) if default_split in split_cols else 0, key=f"{key_prefix}_split_col")

        export_split_dir = sp_c2.text_input("Export Directory", str(project_root / 'outputs/split_batches'), key=f"{key_prefix}_split_dir")

        if st.button("Split & Export CSVs", type="primary", width="stretch", key=f"{key_prefix}_split_export_btn"):
            _log_console(f"{'='*40}\nSTARTING SPLIT EXPORT\nTarget Dir: {export_split_dir}\nSplit Column: {split_col}\n{'='*40}")
            try:
                input_path = None
                if st.session_state.get('global_pheno_file') and st.session_state.get('global_pheno_dir'):
                    input_path = Path(st.session_state.global_pheno_dir) / st.session_state.global_pheno_file
                    _log_console(f"Global source file configured: {input_path}")

                df_to_split = None
                if input_path and input_path.exists():
                    _log_console(f"Reading source file from disk...")
                    st.caption(f"Splitting source file: {input_path}")
                    df_to_split = pd.read_csv(input_path)
                    df_to_split, input_anchor_col, _ = ensure_cell_id_anchor(df_to_split, cell_id_col)
                    _log_console(f"Loaded {len(df_to_split)} rows.")
                    
                    if output_col not in df_to_split.columns or phenotype_map:
                        if input_anchor_col in df_to_split.columns and result_anchor_col in df_results_linkable.columns:
                            st.caption(f"Actively applying {len(phenotype_map)} phenotype assignments to split data...")
                            
                            # Create mapping: CellID -> Cluster -> Phenotype
                            # We use df_results because it links CellID to Cluster
                            id_to_pheno = {}
                            
                            # 1. Map Cluster ID -> Phenotype
                            cluster_to_pheno = {c: phenotype_map.get(c, 'Unassigned') for c in df_results_linkable[cluster_col].unique() if pd.notna(c)}
                            
                            # 2. Map Cell ID -> Phenotype (via Cluster)
                            # Create a temporary series for fast mapping
                            temp_mapping = df_results_linkable.set_index(result_anchor_col)[cluster_col].map(cluster_to_pheno)
                            id_to_pheno = temp_mapping.to_dict()
                            
                            # 3. Apply to df_to_split
                            # Only map cells that exist in our analysis
                            mask = df_to_split[input_anchor_col].isin(id_to_pheno.keys())
                            df_to_split.loc[mask, output_col] = df_to_split.loc[mask, input_anchor_col].map(id_to_pheno)
                            df_to_split[output_col] = df_to_split[output_col].fillna('Unassigned')
                        else:
                            st.warning("Could not apply local phenotype assignments because the current results could not be linked back to the source table.")
                else:
                    st.warning("Source file not found; using in-memory df_results.")
                    _log_console("Source file detection failed. Using in-memory DataFrame.", level="error")
                    df_to_split = df_results_linkable.copy()
                    if phenotype_map:
                        df_to_split[output_col] = df_to_split[cluster_col].apply(
                            lambda mc: phenotype_map.get(mc, phenotype_map.get(str(mc), phenotype_map.get(int(mc), 'Unassigned'))) if pd.notna(mc) else 'Unassigned'
                        )
                    
                    # Ensure input/result anchor names are set for checks below if using memory
                    input_anchor_col = result_anchor_col

                if split_col not in df_to_split.columns:
                    # Attempt to map from df_results if possible
                    if input_anchor_col in df_to_split.columns and result_anchor_col in df_results_linkable.columns and split_col in df_results_linkable.columns:
                        _log_console(f"Mapping missing split column '{split_col}' from memory using ID merge...")
                        st.info(f"Mapping missing split column '{split_col}' from memory...")
                        mapping = df_results_linkable.set_index(result_anchor_col)[split_col].to_dict()
                        df_to_split[split_col] = df_to_split[input_anchor_col].map(mapping)

                if split_col not in df_to_split.columns:
                    msg = f"Column '{split_col}' not found in dataset. Available: {list(df_to_split.columns)}"
                    _log_console(msg)
                    st.error(msg)
                else:
                    if TAB7_INTERNAL_CELL_ID_COL in df_to_split.columns:
                        df_to_split = df_to_split.drop(columns=[TAB7_INTERNAL_CELL_ID_COL])

                    # Sanitize input path to remove accidental whitespace (causes relative path bugs)
                    # User Incident: User pasted " /abs/path" -> became relative "CWD/ /abs/path"
                    clean_dir = export_split_dir.strip()
                    out_path = Path(clean_dir)
                    
                    if clean_dir != export_split_dir:
                         _log_console(f"Sanitized export directory (removed whitespace): \n'{export_split_dir}' -> '{clean_dir}'")

                    _log_console(f"Attempting to create directory: {out_path.resolve()}")
                    try:
                        out_path.mkdir(parents=True, exist_ok=True)
                        if not out_path.exists():
                            _log_console(f"FAILED: Directory does not exist after mkdir: {out_path}", level="error")
                            st.error(f"Failed to create directory: {out_path}")
                            return
                        if not os.access(out_path, os.W_OK):
                            _log_console(f"FAILED: Directory not writable: {out_path}", level="error")
                            st.error(f"Directory exists but is not writable: {out_path}")
                            return
                    except Exception as e:
                        _log_console(f"EXCEPTION creating directory: {e}")
                        st.error(f"Error creating output directory: {e}")
                        return

                    _log_console(f"Directory checks passed. Directory should exist: {out_path.exists()}")

                    groups = df_to_split.groupby(split_col)
                    progress = st.progress(0)
                    total = len(groups)
                    _log_console(f"Found {total} groups for splitting on '{split_col}'.")

                    for i, (val, grp) in enumerate(groups):
                        safe_val = str(val).replace('/', '_').replace('\\', '_')
                        fname = f"{safe_val}.csv"
                        fpath = out_path / fname
                        try:
                            grp.to_csv(fpath, index=False)
                            if i < 3:
                                _log_console(f"Wrote file: {fpath}")
                        except Exception as e:
                            _log_console(f"FAILED to write {fpath}: {e}", level="error")
                            st.error(f"Failed to write {fpath}: {e}")
                        progress.progress((i + 1) / total)

                    st.success(f"Exported {total} files to `{out_path.resolve()}`")
                    _log_console(f"Export loop finished. Total files intended: {total}")
                    
                    # Verify file existence to reassure user
                    if total > 0:
                         files_found = list(out_path.glob("*.csv"))
                         if files_found:
                             ex_str = ", ".join([f.name for f in files_found[:3]])
                             st.caption(f"Verified files on disk: {ex_str} ... (+{len(files_found)-3} more)")
                         else:
                             st.error(f"Export reported success but no CSV files found in {out_path}. Check permissions.")
            except Exception as e:
                st.error(f"Export failed: {e}")


# ==============================================================================
# Settings Export / Import Utilities (Tab7)
# ==============================================================================

# Session-state key substrings that hold large runtime data (DataFrames,
# numpy arrays, internal caches, etc.) and must never be exported.
_SKIP_STATE_SUBSTRINGS: tuple[str, ...] = (
    "_temp_file",
    "_heatmap_cache",
    "_loaded_source_path",
    "_run_id",
    "_n_clusters",
    "_compute_cache",
    "_pending_",
    "_payload",
    "_exp_",           # internal export payload keys
    "_imp_",           # internal import keys
    "_norm_cols",
    "_marker_multiselect",  # multiselect widget keys only (e.g. resolve_marker_multiselect)
                          # _confirmed_markers is intentionally NOT skipped so it restores from JSON
    "thresh_data",     # DataFrame
    "thresh_click_version",
    "thresh_active_marker",
    "thresh_active_batch",
    "_marker_df_draft",
    "_editor_widget",
    "_editor",         # st.data_editor keys (e.g. scimap_threshold_editor, leiden_table_editor)
    "som_heatmap_cache",
    # Widget types that Streamlit forbids setting via session_state:
    "_btn",            # st.button keys  (e.g. scimap_btn_corr, leiden_run_btn)
    "_run_btn",        # explicit run-button suffix
    "_save_btn",       # explicit save-button suffix
    "_select_btn",     # select-button suffix
    "FormSubmitter:",  # st.form_submit_button
    # DataFrame-typed state that is rebuilt from cache files on each render;
    # importing these as strings (json.dumps default=str) corrupts them:
    "_table_draft",      # e.g. scimap_threshold_table_draft, scimap_strategy_table_draft
    "_strategy_table",   # e.g. scimap_strategy_table (saved copy)
    "_threshold_table",  # e.g. scimap_threshold_table (saved copy)
    "_pheno_table",      # e.g. resolve_pheno_table (DataFrame of phenotype definitions)
    # st.file_uploader keys — their value is UploadedFile|None and cannot be set:
    "_import_thresh_csv",  # e.g. scimap_import_thresh_csv, resolve_import_thresh_csv
)

# Explicit action-button keys that do not follow the shared *_btn / *_run_btn
# conventions above. These must stay out of export/import bundles.
_TRANSIENT_ACTION_KEY_SUBSTRINGS: tuple[str, ...] = (
    "tab7_load_",
    "cellviz_generate",
    "_pub_generate",
    "_save_all_btn",
    "_split_export_btn",
    "mlc_run",
    "mrml_run",
    "resolve_compute_auto_thresh",
    "resolve_add_threshold_row_auto",
    "resolve_save_threshold_table",
    "scimap_save_threshold_table",
    "scimap_load_strategy",
    "scimap_add_strategy_row",
    "subpheno_save_params",
    "subpheno_run",
    "leiden_toggle_x",
    "leiden_toggle_y",
    "leiden_toggle_counts",
    "save_leiden_umap",
    "leiden_apply_table",
    "leiden_reset_table",
    "leiden_export_final",
    "som_toggle_x",
    "som_toggle_y",
    "som_toggle_counts",
    "som_export_final",
    "subpheno_export_final",
    "tools_mr_scan",
    "tools_mr_extract",
    "tools_mr_relabel",
    "tools_mr_enrich",
    "tools_mr_merge",
    "tools_pg_run",
    "tools_pg_scan_classes",
    "tools_pg_generate_pngs",
    "tools_edit_csv_load",
    "tools_edit_csv_apply_cleanup",
    "tools_edit_csv_build_derived",
    "tools_edit_csv_export_select_all",
    "tools_edit_csv_export_clear",
    "tools_edit_csv_add_filter",
    "tools_edit_csv_filter_remove_",
    "tools_edit_csv_split_run",
    "tools_edit_csv_save_unified",
    "tools_edit_csv_save_new",
    "umap_add_filter",
    "umap_rm_filter",
    "viz_prop_run",
    "viz_pair_run",
    "viz_f1_scan",
    "viz_f1_run",
    "viz_heatmap_add_filter",
    "viz_heatmap_remove_filter",
    "viz_heatmap_clear_filters",
    "viz_val_run",
)

_RUNTIME_ONLY_STATE_KEY_SUFFIXES: tuple[str, ...] = (
    "_phenotype_map",
    "_available_phenotypes",
    "_run_summary",
    "_run_parameters",
    "_loaded_hash",
    "_marker_cols",
    "_selected_clusters",
    "_selected_k",
    "_has_run",
    "_run_log_lines",
    "_k_results_path",
    "_k_plot_path",
    "_qc_dashboard_path",
    "_qc_dashboard_pre_path",
    "_qc_dashboard_post_path",
    "_k_recommendation",
    "_composition_results",
)


def _is_runtime_only_state_key(key_text: str) -> bool:
    """Return True for per-run result state that should not round-trip as settings."""
    normalized_key = str(key_text)
    if normalized_key in {
        "tools_mr_vv_results",
        "tools_mr_vv_seen_keys",
        "tools_mr_vv_fetch_more",
        "tools_mr_vv_last_file",
    }:
        return True
    if normalized_key.startswith(("tools_mr_vv_assign_pheno_", "tools_mr_vv_assign_save_")):
        return True
    return normalized_key.endswith(_RUNTIME_ONLY_STATE_KEY_SUFFIXES)


@lru_cache(maxsize=1)
def _discover_literal_button_keys() -> frozenset[str]:
    """Collect explicit key=... values used by button-like widgets in Tab7.

    This prevents imports from restoring stale button state when a new explicit
    button key is added in one subtab but the shared skip list is not updated.
    Dynamic f-string keys are still handled by the substring rules above.
    """
    button_keys: set[str] = set()
    button_attrs = {"button", "form_submit_button", "download_button"}

    for path in current_file.parent.glob("*.py"):
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


def _extract_dynamic_widget_key_prefix(node: ast.AST) -> str | None:
    """Return the static leading prefix from a dynamic Streamlit widget key."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                break
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            break
        prefix = "".join(parts).strip()
        return prefix or None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left_prefix = _extract_dynamic_widget_key_prefix(node.left)
        if not left_prefix:
            return None
        right_prefix = _extract_dynamic_widget_key_prefix(node.right)
        return f"{left_prefix}{right_prefix}" if right_prefix is not None else left_prefix
    return None


@lru_cache(maxsize=1)
def _discover_dynamic_button_key_prefixes() -> frozenset[str]:
    """Collect static prefixes from dynamic button-like widget keys in Tab7."""
    button_prefixes: set[str] = set()
    button_attrs = {"button", "form_submit_button", "download_button"}

    for path in current_file.parent.glob("*.py"):
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
                prefix = _extract_dynamic_widget_key_prefix(keyword.value)
                if prefix and not isinstance(keyword.value, (ast.Constant, ast.Str)):
                    button_prefixes.add(prefix)

    return frozenset(button_prefixes)


# Maximum JSON character length for a single session-state value.
# Values larger than this are skipped (e.g. phenotype maps with thousands of entries).
_MAX_EXPORT_VALUE_LEN: int = 500_000


def _should_skip_state_key(key: object) -> bool:
    """Return True for session-state keys that must never round-trip via JSON."""
    key_text = str(key)
    if key_text.startswith("FormSubmitter:") or key_text.startswith("_sett_"):
        return True
    if _is_runtime_only_state_key(key_text):
        return True
    if key_text in _discover_literal_button_keys():
        return True
    if any(key_text.startswith(prefix) for prefix in _discover_dynamic_button_key_prefixes()):
        return True
    if any(sub in key_text for sub in _SKIP_STATE_SUBSTRINGS):
        return True
    return any(sub in key_text for sub in _TRANSIENT_ACTION_KEY_SUBSTRINGS)


def _delete_state_key(key: object) -> None:
    """Best-effort session-state delete used during settings import."""
    session_key = key if isinstance(key, (str, int)) else str(key)
    if session_key in st.session_state:
        try:
            del st.session_state[session_key]
        except Exception:
            pass


def _infer_import_prefixes(imported_keys: "dict[str, object]") -> set[str]:
    """Infer top-level Tab7 key prefixes from an imported settings bundle."""
    prefixes: set[str] = set()
    for key in imported_keys:
        if key.startswith("FormSubmitter:") or "_" not in key:
            continue
        prefixes.add(f"{key.split('_', 1)[0]}_")
    return prefixes


def _clear_stale_import_state(imported_keys: "dict[str, object]") -> None:
    """Clear skipped/derived widget state so imported config can rebuild it cleanly."""
    imported_prefixes = _infer_import_prefixes(imported_keys)

    # Cache-backed tables, transient widgets, and action buttons are skipped from
    # JSON export. Clear any existing values for the imported prefixes so the next
    # rerun rebuilds them from the restored cache files / imported state.
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if any(key_text.startswith(prefix) for prefix in imported_prefixes) and _should_skip_state_key(key_text):
            _delete_state_key(key)

    for key in imported_keys:
        # Marker multiselect widgets are intentionally skipped from export; after
        # import they should re-initialize from *_confirmed_markers.
        if key.endswith("_confirmed_markers"):
            _delete_state_key(key.replace("_confirmed_markers", "_marker_multiselect"))

        # Older export files may contain the config dict but not the filter widget
        # keys. Clearing them forces the checkbox/select/text input to reflect the
        # imported config on the next rerun instead of stale current-session state.
        if key.endswith("_config"):
            base_prefix = key[: -len("config")]
            filter_prefixes = [base_prefix]
            if base_prefix == "scimap_":
                filter_prefixes.append("scimap_filt_")
            for filter_prefix in filter_prefixes:
                for suffix in ("enable_filter", "filter_col", "filter_val", "cell_id_col"):
                    _delete_state_key(f"{filter_prefix}{suffix}")


def _collect_settings_for_export(
    state_key_prefixes: list[str],
    cache_files: "dict[str, Path]",
    extra_state_keys: "list[str] | None" = None,
) -> dict:
    """Collect all settings into a serialisable dict for export.

    Parameters
    ----------
    state_key_prefixes : list[str]
        Only session-state keys whose names start with one of these prefixes
        are included (e.g. ``['leiden_']``).
    cache_files : dict[str, Path]
        Mapping of logical name → disk path for JSON/CSV cache files whose
        text content is embedded verbatim in the export bundle.
    extra_state_keys : list[str] or None
        Additional session-state keys included regardless of prefix (e.g.
        global config keys).

    Returns
    -------
    dict
        ``{"version": 1, "exported_at": ..., "session_state": {...}, "cache_files": {...}}``
    """
    result: dict = {
        "version": 1,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_state": {},
        "cache_files": {},
    }

    extra_keys: set = set(extra_state_keys or [])

    for key in list(st.session_state.keys()):
        # Skip transient runtime-only keys.
        key_text = str(key)
        if _should_skip_state_key(key_text):
            continue
        if _is_nonportable_settings_key(key_text):
            continue
        # Only include keys matching the requested prefixes or explicit extras
        if not any(key_text.startswith(p) for p in state_key_prefixes) and key_text not in extra_keys:
            continue
        value = st.session_state[key]
        try:
            sanitized_value = _sanitize_portable_settings_payload(value)
            serialized = json.dumps(sanitized_value, default=str)
            if len(serialized) > _MAX_EXPORT_VALUE_LEN:
                continue
            result["session_state"][key_text] = json.loads(serialized)
        except (TypeError, ValueError, OverflowError):
            continue

    for name, path in cache_files.items():
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    result["cache_files"][name] = _sanitize_cache_file_text(name, f.read())
            except Exception:
                pass

    return result


def _apply_imported_settings(
    settings: dict,
    cache_files: "dict[str, Path]",
) -> "list[str]":
    """Apply settings from an imported dict back to session state and disk.

    Parameters
    ----------
    settings : dict
        Dict produced by :func:`_collect_settings_for_export`.
    cache_files : dict[str, Path]
        Same name → path mapping used at export time.

    Returns
    -------
    list[str]
        Non-fatal warning messages; empty on full success.
    """
    warnings_out: list[str] = []

    # Restore cache files first so the next rerun reads fresh on-disk values
    for name, content in settings.get("cache_files", {}).items():
        if name not in cache_files:
            continue
        path = cache_files[name]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_sanitize_cache_file_text(name, str(content)))
        except Exception as exc:
            warnings_out.append(f"Could not restore file '{name}': {exc}")

    # Restore session-state values
    imported_keys: dict[str, object] = {
        str(key): _sanitize_portable_settings_payload(value)
        for key, value in settings.get("session_state", {}).items()
        if not _is_nonportable_settings_key(str(key))
    }

    # Clear skipped / derived state for the imported prefixes first. This avoids
    # stale widgets or cache-backed DataFrames from surviving a JSON import.
    _clear_stale_import_state(imported_keys)

    # Pre-clear pass: delete EVERY imported key from session state first.
    # Streamlit raises StreamlitValueAssignmentNotAllowedError not when we *set*
    # a widget key, but when the widget *renders* and finds its key already in
    # session state.  By deleting all imported keys (including button keys from
    # old export files that pre-date our skip filters) we guarantee that button
    # widget keys are absent from session state when the page rerenders — the
    # exact condition that prevents the error.
    for key in imported_keys:
        _delete_state_key(key)

    for key, value in imported_keys.items():
        # Skip button/widget keys that Streamlit forbids setting programmatically.
        if _should_skip_state_key(key):
            continue
        try:
            st.session_state[key] = value
        except Exception as exc:
            exc_type = type(exc).__name__
            if "ValueAssignmentNotAllowed" in exc_type or "WidgetState" in exc_type:
                # Extra safety net — silently skip any widget key we missed above
                pass
            else:
                warnings_out.append(f"Could not restore session key '{key}': {exc}")

    return warnings_out


def render_settings_export_import(
    subtab_name: str,
    state_key_prefixes: "list[str]",
    cache_files: "dict[str, Path]",
    extra_state_keys: "list[str] | None" = None,
    key_suffix: "str | None" = None,
) -> None:
    """Render a compact export / import settings panel inside a collapsible expander.

    Place this call near the top of any subtab render function, after the
    cache directory is initialised, to give the user a persistent settings
    backup/restore capability.

    Parameters
    ----------
    subtab_name : str
        Human-readable name shown in the expander header.
    state_key_prefixes : list[str]
        Session-state key prefixes to capture on export (e.g. ``['leiden_']``).
    cache_files : dict[str, Path]
        Logical name → disk path mapping for persistent JSON/CSV files to
        embed in the export bundle.
    extra_state_keys : list[str] or None
        Additional session-state keys included regardless of prefix.
    key_suffix : str or None
        Unique suffix for Streamlit widget keys.  Defaults to a slug derived
        from *subtab_name*.
    """
    uid = (key_suffix or subtab_name.lower()
           .replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))

    # Cleanup sweep — runs on EVERY render, BEFORE any widget in this subtab.
    # Removes stale widget keys left by old imports:
    #   • bool False  → button keys (st.button / form_submit_button)
    #   • None        → empty file-uploader keys (st.file_uploader returns None)
    # On the VERY FIRST render of this session only, also removes stuck
    # st.data_editor keys (_editor suffix) from prior imports.  After that
    # first-render sweep, editor keys are left alone so that in-progress
    # table edits survive subsequent reruns.
    _first_render_flag = f"_sett_initial_sweep_{uid}"
    _is_first_render = _first_render_flag not in st.session_state
    if _is_first_render:
        st.session_state[_first_render_flag] = True
    for _ck in list(st.session_state.keys()):
        _ck_text = str(_ck)
        if any(_ck_text.startswith(_p) for _p in state_key_prefixes):
            try:
                _v = st.session_state[_ck]
                _delete = (_should_skip_state_key(_ck_text) and ((isinstance(_v, bool) and _v is False) or _v is None))
                if not _delete and _is_first_render:
                    # First-render only: also evict stuck data-editor keys
                    _delete = any(sub in _ck_text for sub in ("_editor",))
                if _delete:
                    del st.session_state[_ck]
            except Exception:
                pass

    with st.expander(f"Export / Import {subtab_name} Settings", expanded=False):
        col_exp, col_imp = st.columns(2)

        # ---- Export side ----
        with col_exp:
            st.markdown("**Export Settings**")
            st.caption("Snapshot all current tab settings to a portable JSON file. The active CSV selection is excluded.")
            if st.button("Prepare Export", key=f"_sett_exp_{uid}_btn"):
                data = _collect_settings_for_export(
                    state_key_prefixes, cache_files, extra_state_keys
                )
                st.session_state[f"_sett_exp_{uid}_payload"] = json.dumps(
                    data, indent=2, default=str
                )

            payload = st.session_state.get(f"_sett_exp_{uid}_payload")
            if payload:
                st.download_button(
                    "Download Settings JSON",
                    payload,
                    file_name=f"tab7_{uid}_settings.json",
                    mime="application/json",
                    key=f"_sett_exp_{uid}_dl",
                )

        # ---- Import side ----
        with col_imp:
            st.markdown("**Import Settings**")
            st.caption("Upload a previously exported JSON to restore tab settings. The current CSV selection stays unchanged.")
            uploaded = st.file_uploader(
                "Settings JSON",
                type=["json"],
                key=f"_sett_imp_{uid}_upload",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                if st.button(
                    "Apply Imported Settings",
                    type="primary",
                    key=f"_sett_imp_{uid}_btn",
                ):
                    try:
                        content = uploaded.read()
                        imported = json.loads(content)
                        errs = _apply_imported_settings(imported, cache_files)
                        if errs:
                            for msg in errs:
                                st.warning(msg)
                        st.success("Settings restored successfully! Reloading...")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to apply settings: {exc}")
                else:
                    st.caption(f"File **{uploaded.name}** ready — click Apply to restore.")
