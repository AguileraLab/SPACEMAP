"""
Consensus ML Phenotyping Subtab
Machine learning phenotyping using consensus labels (Step9B).
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from spacemap import dataio as _dataio
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

from .common import (
    TAB7_INTERNAL_CELL_ID_COL,
    ensure_cell_id_anchor,
    logger,
    get_marker_list,
    get_marker_workspace_panel,
    render_method_marker_selection,
    resolve_saved_column_default,
    project_root_calc,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_export_split_data,
    _extract_markers_from_notebook,
    render_settings_export_import,
    render_method_intro,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    resolve_notebook_marker_defaults,
)
from spacemap_gui import terminal

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

try:
    from spacemap import ml_consensus
    sys.modules["ml_consensus"] = ml_consensus
    from spacemap.ml_consensus import ConsensusConfig, run_ml_consensus_analysis
except ImportError:
    import importlib.util
    try:
        # Check parents[3] / "spacemap" / "ml_consensus.py" (current workspace package structure)
        _ML_CONSENSUS_PATH = Path(__file__).resolve().parents[3] / "spacemap" / "ml_consensus.py"
        if not _ML_CONSENSUS_PATH.exists():
            # Check parents[4] / "src" / "spacemap" / "ml_consensus.py"
            _ML_CONSENSUS_PATH = Path(__file__).resolve().parents[4] / "src" / "spacemap" / "ml_consensus.py"
        if not _ML_CONSENSUS_PATH.exists():
            # Check legacy V13 path
            _ML_CONSENSUS_PATH = Path(__file__).resolve().parents[3] / "src" / "ml_consensus.py"

        if _ML_CONSENSUS_PATH.exists():
            _ph_spec = importlib.util.spec_from_file_location("ml_consensus", _ML_CONSENSUS_PATH)
            if _ph_spec is not None and _ph_spec.loader is not None:
                ml_consensus = importlib.util.module_from_spec(_ph_spec)
                sys.modules.setdefault("ml_consensus", ml_consensus)
                _ph_spec.loader.exec_module(ml_consensus)
                ConsensusConfig = ml_consensus.ConsensusConfig
                run_ml_consensus_analysis = ml_consensus.run_ml_consensus_analysis
            else:
                ConsensusConfig = None
                run_ml_consensus_analysis = None
        else:
            ConsensusConfig = None
            run_ml_consensus_analysis = None
    except Exception:
        ConsensusConfig = None
        run_ml_consensus_analysis = None
except Exception:
    ConsensusConfig = None
    run_ml_consensus_analysis = None


NOTEBOOK_NAME = "Step9B_ML_concensus.ipynb"
NO_MANUAL_VALIDATION_OPTION = "Don't score against manual labels (weak - see help)"


def _looks_like_manual_reference(column: str) -> bool:
    name = str(column).strip().lower()
    reviewer_reference = name in {
        "phenotype_bd",
        "phenotype_bd_x",
        "phenotype_bd_y",
        "phenotype_ap",
        "phenotype_ap_x",
        "phenotype_ap_y",
    }
    named_manual_reference = name in {
        "phenotype_manual",
        "manual_phenotype",
        "ground_truth",
        "phenotype_ground_truth",
    } or name.startswith("phenotype_manual_ref")
    return (
        name.startswith("phenotype_ref")
        or reviewer_reference
        or named_manual_reference
    )


def _looks_like_annotation_metadata(column: str) -> bool:
    name = str(column).strip().lower()
    return name == "phenotype_manual_status" or name.endswith(
        ("_match_distance", "_match_status")
    )


def _ordered_reference_columns(columns: list[str]) -> list[str]:
    return sorted(
        [
            column
            for column in columns
            if _looks_like_manual_reference(column)
            and not _looks_like_annotation_metadata(column)
        ],
        key=str.lower,
    )


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab7', message, step='Consensus ML', level=level)


def _extract_ml_consensus_config() -> dict[str, Any]:
    """Extract ML consensus config values from Step9B notebook.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values with defaults.
    """
    config: dict[str, Any] = {
        "unified_csv": None,
        "output_phenotype_col": "phenotype_ml_consensus",
        "confidence_col": "ml_consensus_score",
        "validation_ground_truth_col": None,
        "phenotyping_methods": [],
        "exclude_phenotypes": ["Others"],
        "morphology_cols": [],
        "compartment_cols": [],
        "feature_weights": {"markers": 1.0, "morphology": 0.0, "compartments": 0.0},
        "use_z_normalization": False,
        "batch_normalization_col": "batch_id",
        "enable_filter": True,
        "filter_col": "tissue",
        "filter_val": "1",
        "min_consensus_methods": 3,
        "test_size": 0.3,
        "n_estimators": 1500,
        "random_state": 42,
        "min_training_cells": 50,
        "figure_dpi": 300,
        "consensus_col_name": "consensus",
        "refined_confidence_threshold": 0.75,
        "refined_low_confidence_label": "Others",
        "write_backup_csv": False,
        "ml_method": "random_forest",
    }

    try:
        notebook_path = project_root_calc / "notebooks" / NOTEBOOK_NAME
        if not notebook_path.exists():
            return config

        with open(notebook_path, "r") as f:
            nb = json.load(f)

        def _extract_list(var_name: str, cell_source: str) -> list[str]:
            m = re.search(
                rf"^[ \t]*{re.escape(var_name)}\s*=\s*\[(.*?)\]",
                cell_source,
                flags=re.S | re.M,
            )
            if not m:
                return []
            inside = m.group(1)
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", inside)
            out: list[str] = []
            seen: set[str] = set()
            for t in tokens:
                t = str(t).strip()
                if not t or t in seen:
                    continue
                out.append(t)
                seen.add(t)
            return out

        def _extract_scalar(var_name: str, cell_source: str) -> str | None:
            m = re.search(
                rf"^[ \t]*{re.escape(var_name)}\s*=\s*([^\n#]+)",
                cell_source,
                flags=re.M,
            )
            if not m:
                return None
            return m.group(1).strip().strip("'\"")

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source:
                continue

            if "UNIFIED_CSV" in source and config["unified_csv"] is None:
                val = _extract_scalar("UNIFIED_CSV", source)
                if val:
                    config["unified_csv"] = val
            if "PHENOTYPING_METHODS" in source:
                config["phenotyping_methods"] = _extract_list("PHENOTYPING_METHODS", source)
            if "EXCLUDED_PHENOTYPES" in source:
                config["exclude_phenotypes"] = _extract_list("EXCLUDED_PHENOTYPES", source)

            if "VALIDATION_GROUND_TRUTH_COLUMN" in source:
                val = _extract_scalar("VALIDATION_GROUND_TRUTH_COLUMN", source)
                config["validation_ground_truth_col"] = (
                    None if val in {None, "None", "null", ""} else val
                )

            if "OUTPUT_PHENOTYPE_COL" in source:
                val = _extract_scalar("OUTPUT_PHENOTYPE_COL", source)
                if val:
                    config["output_phenotype_col"] = val
            if "CONFIDENCE_COL_NAME" in source:
                val = _extract_scalar("CONFIDENCE_COL_NAME", source)
                if val:
                    config["confidence_col"] = val
            if "CONSENSUS_COL_NAME" in source:
                val = _extract_scalar("CONSENSUS_COL_NAME", source)
                if val:
                    config["consensus_col_name"] = val
            if "REFINED_CONFIDENCE_THRESHOLD" in source:
                val = _extract_scalar("REFINED_CONFIDENCE_THRESHOLD", source)
                try:
                    config["refined_confidence_threshold"] = float(val) if val is not None else config["refined_confidence_threshold"]
                except Exception:
                    pass
            if "REFINED_LOW_CONFIDENCE_LABEL" in source:
                val = _extract_scalar("REFINED_LOW_CONFIDENCE_LABEL", source)
                if val:
                    config["refined_low_confidence_label"] = val
            if "WRITE_BACKUP_CSV" in source:
                val = _extract_scalar("WRITE_BACKUP_CSV", source)
                if val in {"True", "False"}:
                    config["write_backup_csv"] = val == "True"

            if "MORPHOLOGY_COLS" in source:
                config["morphology_cols"] = _extract_list("MORPHOLOGY_COLS", source)
            if "COMPARTMENT_COLS" in source:
                config["compartment_cols"] = _extract_list("COMPARTMENT_COLS", source)

            if "FEATURE_WEIGHTS" in source:
                match = re.search(r"FEATURE_WEIGHTS\s*=\s*(\{.*?\})", source, flags=re.S)
                if match:
                    try:
                        config["feature_weights"] = ast.literal_eval(match.group(1))
                    except Exception:
                        pass

            if "USE_Z_NORMALIZATION" in source:
                val = _extract_scalar("USE_Z_NORMALIZATION", source)
                if val in {"True", "False"}:
                    config["use_z_normalization"] = val == "True"

            if "BATCH_NORMALIZATION_COL" in source:
                val = _extract_scalar("BATCH_NORMALIZATION_COL", source)
                if val:
                    config["batch_normalization_col"] = val

            if "ANALYSIS_FILTER_COLUMN" in source:
                val = _extract_scalar("ANALYSIS_FILTER_COLUMN", source)
                config["enable_filter"] = val not in {None, "None", "null", ""}
                if config["enable_filter"]:
                    config["filter_col"] = val
            if "ANALYSIS_FILTER_VALUE" in source:
                val = _extract_scalar("ANALYSIS_FILTER_VALUE", source)
                if val is not None:
                    config["filter_val"] = val

            if "MIN_CONSENSUS_METHODS" in source:
                val = _extract_scalar("MIN_CONSENSUS_METHODS", source)
                try:
                    config["min_consensus_methods"] = int(float(val)) if val is not None else config["min_consensus_methods"]
                except Exception:
                    pass

            if "TEST_SIZE" in source:
                val = _extract_scalar("TEST_SIZE", source)
                try:
                    config["test_size"] = float(val) if val is not None else config["test_size"]
                except Exception:
                    pass

            if "N_ESTIMATORS" in source:
                val = _extract_scalar("N_ESTIMATORS", source)
                try:
                    config["n_estimators"] = int(float(val)) if val is not None else config["n_estimators"]
                except Exception:
                    pass

            if "RANDOM_STATE" in source:
                val = _extract_scalar("RANDOM_STATE", source)
                try:
                    config["random_state"] = int(float(val)) if val is not None else config["random_state"]
                except Exception:
                    pass

            if "MIN_TRAINING_CELLS" in source:
                val = _extract_scalar("MIN_TRAINING_CELLS", source)
                try:
                    config["min_training_cells"] = int(float(val)) if val is not None else config["min_training_cells"]
                except Exception:
                    pass

            if "FIGURE_DPI" in source:
                val = _extract_scalar("FIGURE_DPI", source)
                try:
                    config["figure_dpi"] = int(float(val)) if val is not None else config["figure_dpi"]
                except Exception:
                    pass

            if "ML_METHOD" in source:
                val = _extract_scalar("ML_METHOD", source)
                if val:
                    config["ml_method"] = val

        return config
    except Exception as e:
        logger.warning(f"Consensus ML config extraction failed: {e}")
        return config


MLC_GUIDE = """\
**What it does.** Combines the results of several phenotyping methods. Cells on which enough methods agree are treated as reliable examples; a machine-learning model learns from them and then gives a phenotype to every cell, including the cells where the methods disagreed.

**Steps**
1. Run at least two other methods first (for example Leiden, SOM and Resolve) and save their labels.
2. Choose the input table, the label columns to combine, and the markers.
3. Set **How many methods must agree**, then run.
4. Review the results, save the new label column, and check example cells.

**Note.** Agreement between methods shows the methods are consistent, not that they are right. Check the result against the images.
"""


def render_consensus_ml(project_root: Path, global_input_path: str | None) -> None:
    """Render the Consensus ML subtab UI.

    Parameters
    ----------
    project_root : Path
        Project root path.
    global_input_path : str | None
        Unified CSV path from global configuration.
    """
    render_method_intro(
        project_root, "mlc", "Consensus ML Phenotyping",
        "Combine several methods: learn from cells they agree on, then label every cell.",
        MLC_GUIDE, notebook="Step9B_ML_concensus.ipynb", log_label="Consensus ML Phenotyping",
    )

    cache_dir = project_root / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mlc_cache = cache_dir / "consensus_ml_params.json"

    cached_params: dict[str, Any] = {}
    if mlc_cache.exists():
        try:
            with open(mlc_cache, "r") as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}

    # Prefer the active project's canonical configuration over the legacy
    # method-local cache whenever both are available.
    project_params = st.session_state.get("mlc_config")
    if isinstance(project_params, dict):
        cached_params = {**cached_params, **project_params}

    # --- Settings Export / Import ---
    render_settings_export_import(
        "Consensus ML",
        ["mlc_"],
        {"consensus_ml_params.json": mlc_cache},
    )

    nb_config = _extract_ml_consensus_config()

    st.markdown("### 1. Input File")
    use_global = st.checkbox("Use the project-wide input file", value=cached_params.get("use_global", True), key="mlc_use_global")
    full_input_path = global_input_path

    if not use_global:
        default_dir = cached_params.get("input_dir", str(project_root / "outputs/batch_corrected"))
        input_dir = st.text_input("Folder containing the input file", default_dir, key="mlc_input_dir")
        if input_dir and os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        else:
            files = []
        if files:
            cached_file = cached_params.get("input_file", files[0])
            default_idx = files.index(cached_file) if cached_file in files else 0
            sel = st.selectbox("Input file", files, index=default_idx, key="mlc_input_file")
            full_input_path = os.path.join(input_dir, sel)

        if cached_params.get("input_dir") != input_dir or cached_params.get("input_file") != os.path.basename(full_input_path or ""):
            try:
                cached_params["input_dir"] = input_dir
                cached_params["input_file"] = os.path.basename(full_input_path) if full_input_path else ""
                with open(mlc_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
            except Exception:
                pass

    available_columns: list[str] = []
    if full_input_path and os.path.exists(full_input_path):
        st.caption(f"Active File: {os.path.basename(full_input_path)}")
        try:
            df_head = pd.read_csv(full_input_path, nrows=5)
            available_columns = list(df_head.columns)
        except Exception:
            available_columns = []

    st.divider()
    st.markdown("### 2. Marker Selection")

    corr_markers, non_corr_markers = _extract_markers_from_notebook(NOTEBOOK_NAME)
    expected_markers: list[str] = []
    all_marker_candidates = []
    if full_input_path and os.path.exists(full_input_path):
        try:
            df_head = pd.read_csv(full_input_path, nrows=1)
            all_marker_candidates = get_marker_list(df_head)
        except Exception:
            all_marker_candidates = []

    available_marker_columns = list(all_marker_candidates)
    all_marker_candidates, marker_workspace_note = get_marker_workspace_panel(
        all_marker_candidates,
        full_input_path,
    )
    marker_workspace = st.session_state.get("tab7_marker_workspace", {})
    expected_markers = resolve_notebook_marker_defaults(
        corr_markers,
        non_corr_markers,
        all_marker_candidates,
        str(marker_workspace.get("source_mode", "Raw markers")),
        str(marker_workspace.get("affix_position", "Suffix")),
        str(marker_workspace.get("correction_affix", "_corr")),
    )

    _, confirmed_markers, marker_workspace_note = render_method_marker_selection(
        key_prefix="mlc",
        method_name="Consensus ML",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers,
        saved_markers=cached_params.get("markers", []),
        config_state_key="mlc_config",
    )

    st.divider()
    st.markdown("### 3. Configuration")

    default_config = {
            "use_global": cached_params.get("use_global", True),
            "input_dir": cached_params.get(
                "input_dir", str(project_root / "outputs/batch_corrected")
            ),
            "input_file": cached_params.get("input_file", ""),
            "input_path": cached_params.get("input_path", ""),
            "markers": list(cached_params.get("markers", []) or []),
            "marker_panel_name": cached_params.get("marker_panel_name", ""),
            "marker_resolution": cached_params.get("marker_resolution", {}),
            "validation_ground_truth_col": cached_params.get(
                "validation_ground_truth_col", nb_config.get("validation_ground_truth_col")
            ),
            "output_phenotype_col": cached_params.get("output_phenotype_col", nb_config.get("output_phenotype_col", "phenotype_ml_consensus")),
            "confidence_col": cached_params.get("confidence_col", nb_config.get("confidence_col", "ml_consensus_score")),
            "use_z_normalization": cached_params.get("use_z_normalization", nb_config.get("use_z_normalization", False)),
            "batch_normalization_col": cached_params.get("batch_normalization_col", nb_config.get("batch_normalization_col", "batch_id")),
            "enable_filter": cached_params.get("enable_filter", nb_config.get("enable_filter", True)),
            "filter_col": cached_params.get("filter_col", nb_config.get("filter_col", "tissue")),
            "filter_val": cached_params.get("filter_val", nb_config.get("filter_val", "1")),
            "exclude_phenotypes": cached_params.get("exclude_phenotypes", nb_config.get("exclude_phenotypes", ["Others"])),
            "phenotyping_methods": cached_params.get("phenotyping_methods", nb_config.get("phenotyping_methods", [])),
            "morphology_cols": cached_params.get("morphology_cols", nb_config.get("morphology_cols", [])),
            "compartment_cols": cached_params.get("compartment_cols", nb_config.get("compartment_cols", [])),
            "feature_weights": cached_params.get("feature_weights", nb_config.get("feature_weights", {"markers": 1.0, "morphology": 0.0, "compartments": 0.0})),
            "min_consensus_methods": cached_params.get("min_consensus_methods", nb_config.get("min_consensus_methods", 3)),
            "test_size": cached_params.get("test_size", nb_config.get("test_size", 0.3)),
            "n_estimators": cached_params.get("n_estimators", nb_config.get("n_estimators", 1500)),
            "random_state": cached_params.get("random_state", nb_config.get("random_state", 42)),
            "min_training_cells": cached_params.get("min_training_cells", nb_config.get("min_training_cells", 50)),
            "figure_dpi": cached_params.get("figure_dpi", nb_config.get("figure_dpi", 300)),
            "consensus_col_name": cached_params.get("consensus_col_name", nb_config.get("consensus_col_name", "consensus")),
            "refined_confidence_threshold": cached_params.get("refined_confidence_threshold", nb_config.get("refined_confidence_threshold", 0.75)),
            "refined_low_confidence_label": cached_params.get("refined_low_confidence_label", nb_config.get("refined_low_confidence_label", "Others")),
            "write_backup_csv": cached_params.get(
                "write_backup_csv", nb_config.get("write_backup_csv", False)
            ),
            "ml_method": cached_params.get("ml_method", nb_config.get("ml_method", "random_forest")),
        }
    if not isinstance(st.session_state.get("mlc_config"), dict):
        st.session_state["mlc_config"] = default_config
    else:
        for config_key, default_value in default_config.items():
            st.session_state["mlc_config"].setdefault(config_key, default_value)
    st.session_state["mlc_config"].setdefault("write_backup_csv", False)

    phenotyping_candidates = [
        column
        for column in available_columns
        if column.startswith("phenotype_")
        and not _looks_like_annotation_metadata(column)
    ]
    reference_candidates = _ordered_reference_columns(available_columns)
    if phenotyping_candidates:
        st.caption(f"Detected phenotype columns: {', '.join(phenotyping_candidates)}")
    if available_columns and not reference_candidates:
        st.info(
            "No manual-reference label columns were detected. Consensus ML can still run, "
            "but only exploratory pseudo-label validation will be available."
        )

    with st.form("mlc_config_form", clear_on_submit=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            validation_options = [NO_MANUAL_VALIDATION_OPTION] + reference_candidates
            saved_validation = st.session_state["mlc_config"].get("validation_ground_truth_col")
            preferred_validation = (
                str(saved_validation)
                if saved_validation in validation_options
                else ("phenotype_ref1b" if "phenotype_ref1b" in validation_options else NO_MANUAL_VALIDATION_OPTION)
            )
            validation_selection = st.selectbox(
                "Score the result against these manual labels",
                options=validation_options,
                index=validation_options.index(preferred_validation),
                key="mlc_validation_gt_col",
                help=(
                    "A set of hand-assigned phenotypes used only to grade the finished result. "
                    "Any cell you label here is automatically kept out of training, so the score "
                    "reflects cells the model never saw.\n\n"
                    "If you pick the 'Don't score' option instead, the model is graded by holding "
                    "back some of the consensus cells it was built from. That only shows whether it "
                    "can reproduce the methods' own agreement - it cannot tell you whether that "
                    "agreement was correct. Treat those numbers as a sanity check, not evidence.\n\n"
                    "Either way this measures unseen *cells*, not an unseen *patient*. For that, "
                    "use the patient-out validation in the F1 Score tab."
                ),
            )
            validation_ground_truth_col = (
                None if validation_selection == NO_MANUAL_VALIDATION_OPTION else validation_selection
            )
            output_phenotype_col = st.text_input(
                "Name for the new phenotype column",
                value=st.session_state["mlc_config"]["output_phenotype_col"],
                key="mlc_output_pheno_col",
                help=(
                    "Column name used for review and later written to the unified CSV only when "
                    "'Save All Assignments to Disk' is clicked."
                ),
            )
            confidence_col = st.text_input(
                "Name for the new confidence column",
                value=st.session_state["mlc_config"].get("confidence_col", "ml_consensus_score"),
                key="mlc_confidence_col",
                help=(
                    "Direct-consensus cells receive their agreeing-method fraction; model-imputed cells receive "
                    "maximum class probability. Use this as an uncertainty score, not a calibrated probability."
                ),
            )
            consensus_col_name = st.text_input(
                "Column marking which cells reached consensus",
                value=st.session_state["mlc_config"].get("consensus_col_name", "consensus"),
                key="mlc_consensus_col_name",
                help="Column name for binary consensus flag (1=consensus, 0=non-consensus).",
            )

            st.markdown("**Refined Phenotype Column**")
            _cur_pheno_col = st.session_state["mlc_config"].get("output_phenotype_col", "phenotype_ml_consensus")
            st.caption(f"Auto-named: `{_cur_pheno_col}_refined`")
            refined_threshold = st.number_input(
                "Confidence below which a call is replaced",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["mlc_config"].get("refined_confidence_threshold", 0.75)),
                step=0.05,
                key="mlc_refined_threshold",
                help=(
                    "Cells below this threshold are relabeled in the refined column. For direct-consensus cells, "
                    "the score is the agreeing-method fraction; for imputed cells, it is maximum model probability."
                ),
            )
            refined_label = st.text_input(
                "Label to use for low-confidence cells",
                value=st.session_state["mlc_config"].get("refined_low_confidence_label", "Others"),
                key="mlc_refined_label",
                help="Label applied to low-confidence cells in the refined phenotype column (default: 'Others').",
            )
            st.markdown("**Run-time Storage**")
            write_backup_csv = st.checkbox(
                "Automatically save a separate Consensus-ML result CSV",
                value=bool(
                    st.session_state["mlc_config"].get("write_backup_csv", False)
                ),
                key="mlc_write_backup_csv",
                help=(
                    "Off by default. When off, running Consensus ML does not write any result CSV or "
                    "change the unified CSV. Review the results first, then use "
                    "'Save All Assignments to Disk'."
                ),
            )

            use_z_norm = st.checkbox(
                "Z-normalize features within each batch",
                value=bool(st.session_state["mlc_config"]["use_z_normalization"]),
                key="mlc_use_z",
                help=(
                    "Optional batch-wise Z-score preprocessing of markers and morphology. "
                    "If the selected markers are already batch-corrected, leave this off unless you are reproducing a predefined analysis."
                ),
            )

            batch_norm_col = st.text_input(
                "Column identifying the batch",
                value=st.session_state["mlc_config"]["batch_normalization_col"],
                key="mlc_batch_norm_col",
                help="Grouping column used only when Z-normalization is enabled.",
            )

            st.markdown("**Filtering**")
            enable_filter = st.checkbox(
                "Only use a subset of the cells",
                value=bool(st.session_state["mlc_config"].get("enable_filter", False)),
                key="mlc_enable_filter",
                help="Restrict consensus construction, model training, and prediction to rows matching the filter below.",
            )
            default_filter_col = resolve_saved_column_default(
                st.session_state["mlc_config"].get("filter_col"),
                available_columns,
            )
            filter_col = st.selectbox(
                "Filter on this column",
                available_columns if available_columns else [default_filter_col],
                index=available_columns.index(default_filter_col) if default_filter_col in available_columns else 0,
                key="mlc_filter_col",
                help="Column to filter for training/prediction. Default: tissue",
            )
            filter_val = st.text_input(
                "Keep rows equal to",
                value=str(st.session_state["mlc_config"].get("filter_val", "1")),
                key="mlc_filter_val",
                help="Only rows matching this value are used for ML.",
            )

        with col2:
            min_consensus_methods = st.number_input(
                "How many methods must agree",
                min_value=2,
                max_value=5,
                value=int(st.session_state["mlc_config"]["min_consensus_methods"][0] if isinstance(st.session_state["mlc_config"]["min_consensus_methods"], list) else st.session_state["mlc_config"]["min_consensus_methods"]),
                step=1,
                help="Minimum number of methods that must agree to define consensus.",
                key="mlc_min_consensus",
            )

            test_size = st.number_input(
                "Fraction of cells held back for the internal check",
                min_value=0.05,
                max_value=0.95,
                value=float(st.session_state["mlc_config"]["test_size"]),
                step=0.05,
                help=(
                    "Used only without a held-out manual validation column. This exploratory split measures how well the classifier reproduces consensus pseudo-labels."
                ),
                key="mlc_test_size",
                disabled=validation_ground_truth_col is not None,
            )

            n_estimators = st.number_input(
                "Number of trees",
                min_value=100,
                max_value=5000,
                value=int(st.session_state["mlc_config"]["n_estimators"]),
                step=100,
                help="Number of trees in the Random Forest / Gradient Boosting.",
                key="mlc_n_estimators",
            )

            _ml_methods = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "svm": "SVM (RBF Kernel)"}
            _ml_keys = list(_ml_methods.keys())
            _ml_labels = list(_ml_methods.values())
            _current_method = st.session_state["mlc_config"].get("ml_method", "random_forest")
            _ml_idx = _ml_keys.index(_current_method) if _current_method in _ml_keys else 0
            ml_method_label = st.selectbox(
                "Classifier",
                options=_ml_labels,
                index=_ml_idx,
                help="Random Forest (robust general-purpose), Gradient Boosting (stronger learner), SVM (good for marker patterns).",
                key="mlc_ml_method",
            )
            ml_method = _ml_keys[_ml_labels.index(ml_method_label)]

        st.markdown("**Phenotyping Methods (phenotype_*)**")
        method_defaults = st.session_state["mlc_config"]["phenotyping_methods"]
        _mlc_options = [
            column
            for column in phenotyping_candidates
            if not _looks_like_manual_reference(column)
            and not _looks_like_annotation_metadata(column)
            and not column.startswith("phenotype_ml_")
            and column not in {validation_ground_truth_col, output_phenotype_col}
        ]
        if not method_defaults and _mlc_options:
            method_defaults = _mlc_options
        _mlc_safe_defaults = [m for m in method_defaults if m in _mlc_options]
        phenotyping_methods = st.multiselect(
            "Phenotyping methods to combine",
            _mlc_options,
            default=_mlc_safe_defaults,
            key="mlc_methods",
            help=(
                "Automated phenotype columns used to construct consensus pseudo-labels. "
                "Manual references and prior ML outputs are excluded to prevent target leakage and circularity."
            ),
        )

        st.markdown("**Exclude Phenotypes from Consensus**")
        if st.form_submit_button("Scan Phenotypes", type="secondary"):
            if full_input_path and os.path.exists(full_input_path) and phenotyping_methods:
                try:
                    scan_cols = [c for c in phenotyping_methods if c in available_columns]
                    if scan_cols:
                        df_scan = _dataio.load(full_input_path, columns=scan_cols)
                        unique_vals = set()
                        for col in scan_cols:
                            unique_vals.update(df_scan[col].dropna().astype(str).unique().tolist())
                        st.session_state["mlc_exclude_candidates"] = sorted(unique_vals)
                except Exception as exc:
                    st.warning(f"Failed to scan phenotypes: {exc}")
            else:
                st.warning("Select phenotype methods and ensure the unified CSV is available.")

        default_excluded = st.session_state["mlc_config"].get("exclude_phenotypes", [])
        exclude_candidates = sorted(
            set(st.session_state.get("mlc_exclude_candidates", [])) | set(default_excluded)
        )
        valid_defaults = [x for x in default_excluded if x in exclude_candidates]
        
        exclude_phenotypes = st.multiselect(
            "Labels to treat as not-a-phenotype",
            options=exclude_candidates,
            default=valid_defaults,
            help=(
                "These labels are ignored as votes. A row may still form consensus if enough remaining automated "
                "methods uniquely agree on another phenotype."
            ),
            key="mlc_exclude_phenotypes",
        )

        st.markdown("**Feature Configuration**")
        morph_cols = st.text_input(
            "Morphology features to include (comma-separated)",
            value=", ".join(st.session_state["mlc_config"]["morphology_cols"]),
            key="mlc_morph_cols",
            help="Optional numeric morphology features added to the marker feature matrix.",
        )
        comp_cols = st.text_input(
            "Tissue-compartment features to include (comma-separated)",
            value=", ".join(st.session_state["mlc_config"]["compartment_cols"]),
            key="mlc_comp_cols",
            help="Optional numeric or binary tissue-compartment features added to the model.",
        )

        w1, w2, w3 = st.columns(3)
        feature_weights = st.session_state["mlc_config"]["feature_weights"]
        w_markers = w1.number_input("Marker weight", value=float(feature_weights.get("markers", 1.0)), step=0.1, key="mlc_w_markers", help="Set to zero to exclude markers. With the current standardized/tree classifiers, nonzero values should not be interpreted as learned feature importance.")
        w_morph = w2.number_input("Morphology weight", value=float(feature_weights.get("morphology", 0.0)), step=0.1, key="mlc_w_morph", help="Set to zero to exclude morphology. With the current standardized/tree classifiers, use this primarily as an include/exclude control.")
        w_comp = w3.number_input("Compartment weight", value=float(feature_weights.get("compartments", 0.0)), step=0.1, key="mlc_w_comp", help="Set to zero to exclude compartments. With the current standardized/tree classifiers, use this primarily as an include/exclude control.")

        t1, t2, t3 = st.columns(3)
        random_state = t1.number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=int(st.session_state["mlc_config"]["random_state"]),
            step=1,
            key="mlc_random_state",
            help="Random seed for the classifier and exploratory internal split.",
        )
        min_training_cells = t2.number_input(
            "Minimum consensus cells needed to train",
            min_value=10,
            max_value=5000,
            value=int(st.session_state["mlc_config"]["min_training_cells"]),
            step=10,
            key="mlc_min_training_cells",
            help="Minimum number of permitted consensus pseudo-labeled cells required for model fitting.",
        )
        figure_dpi = t3.number_input(
            "Figure resolution (DPI)",
            min_value=100,
            max_value=600,
            value=int(st.session_state["mlc_config"]["figure_dpi"]),
            step=50,
            key="mlc_figure_dpi",
            help="Resolution of saved validation and expression figures.",
        )

        st.caption(
            "Click the button below to persist every setting in this configuration "
            "section to the active project JSON."
        )
        save_cfg = st.form_submit_button(
            "Save All Configuration to Project", type="primary"
        )
        if save_cfg:
            st.session_state["mlc_config"].update(
                {
                    "use_global": bool(use_global),
                    "input_dir": str(
                        st.session_state.get(
                            "mlc_input_dir",
                            st.session_state["mlc_config"].get("input_dir", ""),
                        )
                    ),
                    "input_file": str(
                        st.session_state.get(
                            "mlc_input_file",
                            st.session_state["mlc_config"].get("input_file", ""),
                        )
                    ),
                    "input_path": str(full_input_path or ""),
                    "markers": list(confirmed_markers),
                    "marker_panel_name": str(
                        st.session_state.get("mlc_marker_panel_name", "")
                    ),
                    "marker_resolution": dict(
                        st.session_state.get("mlc_marker_resolution", {}) or {}
                    ),
                    "validation_ground_truth_col": validation_ground_truth_col,
                    "output_phenotype_col": output_phenotype_col,
                    "confidence_col": confidence_col,
                    "use_z_normalization": use_z_norm,
                    "batch_normalization_col": batch_norm_col,
                    "enable_filter": enable_filter,
                    "filter_col": filter_col,
                    "filter_val": filter_val,
                    "exclude_phenotypes": exclude_phenotypes,
                    "phenotyping_methods": phenotyping_methods,
                    "morphology_cols": [c.strip() for c in morph_cols.split(",") if c.strip()],
                    "compartment_cols": [c.strip() for c in comp_cols.split(",") if c.strip()],
                    "feature_weights": {"markers": w_markers, "morphology": w_morph, "compartments": w_comp},
                    "min_consensus_methods": int(min_consensus_methods),
                    "test_size": float(test_size),
                    "n_estimators": int(n_estimators),
                    "random_state": int(random_state),
                    "min_training_cells": int(min_training_cells),
                    "figure_dpi": int(figure_dpi),
                    "consensus_col_name": consensus_col_name,
                    "refined_confidence_threshold": float(refined_threshold),
                    "refined_low_confidence_label": refined_label,
                    "write_backup_csv": bool(write_backup_csv),
                    "ml_method": ml_method,
                }
            )
            try:
                cached_params.update(st.session_state["mlc_config"])
                with open(mlc_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
                from spacemap_gui.persistence import save_active_project_state
                save_active_project_state("tab7_phenotyping")
            except Exception as ex:
                logger.warning(f"Failed to auto-save configuration or project state: {ex}")
            st.success(
                "All Consensus-ML configuration settings were saved to the project JSON."
            )

    st.divider()
    st.markdown("### 4. Run Consensus ML Phenotyping")

    if st.button("Run Consensus ML Pipeline", type="primary", width="stretch", key="mlc_run"):
        _log_console(f"Run button clicked")
        _log_console(f"Input file: {full_input_path}")
        _log_console(f"Markers count: {len(confirmed_markers) if confirmed_markers else 0}")

        if not full_input_path or not os.path.exists(full_input_path):
            _log_console(f"ERROR - No valid unified CSV file selected", level="error")
            st.error("Please select a valid unified CSV file.")
            return
        if not confirmed_markers:
            _log_console(f"ERROR - No markers selected", level="error")
            st.error("No markers selected. Please save marker selection first.")
            return
        if run_ml_consensus_analysis is None or ConsensusConfig is None:
            _log_console(f"ERROR - ml_consensus module not loaded", level="error")
            st.error("Consensus ML utilities are not available. Please check ml_consensus.")
            return

        cfg = st.session_state["mlc_config"]
        if len(cfg.get("phenotyping_methods", [])) < 2:
            st.error("Consensus ML requires at least two automated phenotyping methods.")
            return
        if int(cfg.get("min_consensus_methods", 3)) > len(cfg.get("phenotyping_methods", [])):
            st.error(
                "MIN_CONSENSUS_METHODS cannot exceed the number of selected phenotyping methods."
            )
            return
        _log_console(f"ML method: {cfg.get('ml_method', 'random_forest')}")
        _log_console(f"Phenotyping methods: {cfg.get('phenotyping_methods', [])}")
        _log_console(f"Excluded phenotypes: {cfg.get('exclude_phenotypes', [])}")
        _log_console(f"Validation reference: {cfg.get('validation_ground_truth_col') or 'internal pseudo-label split'}")
        out_dir = project_root / "outputs/phenotyping/ML_consensus"
        output_phenotype_col = cfg.get("output_phenotype_col", "phenotype_ml_consensus")
        confidence_col = cfg.get("confidence_col", "ml_consensus_score")
        consensus_col_name = cfg.get("consensus_col_name", "consensus")
        refined_col = f"{output_phenotype_col}_refined"
        run_audit = begin_phenotyping_run_audit(
            Path(project_root),
            "mlc",
            input_csv=full_input_path,
            markers=confirmed_markers,
        )
        audit_input_rows = None
        filter_enabled = bool(cfg.get("enable_filter", False))
        filter_col = str(cfg.get("filter_col", "tissue"))
        filter_val = str(cfg.get("filter_val", "1"))
        exclude_phenotypes = [str(v) for v in cfg.get("exclude_phenotypes", [])]
        exclude_methods = [m for m in cfg.get("phenotyping_methods", []) if m in available_columns]

        # --- Load data once, filter in memory, pass directly to engine ---
        preloaded_df = None
        filtered_mode = False
        cell_id_col = "cell_id"

        _log_console(f"Loading dataset...")
        try:
            from spacemap.ph_accel_utils import read_csv_fast
            full_df = read_csv_fast(full_input_path)
            audit_input_rows = len(full_df)
            full_df, cell_id_col, _ = ensure_cell_id_anchor(full_df, cell_id_col)
            _log_console(f"Loaded {len(full_df):,} cells, {len(full_df.columns)} columns")
        except Exception as exc:
            finish_phenotyping_run_audit(
                run_audit,
                status="failed",
                output_column=output_phenotype_col,
                cluster_column="ml_consensus_cluster",
                error=str(exc),
            )
            _log_console(f"ERROR loading CSV: {exc}", level="error")
            st.error(f"Failed to load CSV: {exc}")
            return

        if filter_enabled:
            _log_console(f"Applying filter {filter_col} == {filter_val}...")
            if filter_col in full_df.columns:
                filtered_df = full_df[full_df[filter_col].astype(str) == filter_val].copy()
                filtered_df.reset_index(drop=True, inplace=True)
                filtered_mode = True
                preloaded_df = filtered_df

                _log_console(f"Filter applied: {len(filtered_df):,} cells match")
                st.info(f"Filtering enabled: {filter_col} == {filter_val} | {len(filtered_df):,} cells")
            else:
                _log_console(f"Filter column '{filter_col}' not found, running on full dataset")
                st.warning(f"Filter column '{filter_col}' not found. Running on full dataset.")
                preloaded_df = full_df
        else:
            preloaded_df = full_df

        # Drop rows containing NaN/Inf in selected markers
        if confirmed_markers:
            orig_len_nan = len(preloaded_df)
            numeric_markers = preloaded_df[confirmed_markers].apply(
                pd.to_numeric, errors="coerce"
            )
            valid_mask = pd.Series(
                np.isfinite(numeric_markers.to_numpy()).all(axis=1),
                index=preloaded_df.index,
            )
            if (orig_len_nan - valid_mask.sum()) > 0:
                filtered_mode = True
                preloaded_df = preloaded_df[valid_mask].copy()
                preloaded_df.reset_index(drop=True, inplace=True)

                st.info(f"Filtered out {orig_len_nan - len(preloaded_df):,} cells containing NaN/Inf in selected markers. Remaining: {len(preloaded_df):,} cells.")
                if len(preloaded_df) == 0:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_phenotype_col,
                        cluster_column="ml_consensus_cluster",
                        error="No cells remained after marker-value filtering.",
                    )
                    st.error("No cells remain after filtering out NaN/Inf values in selected markers.")
                    return

        if exclude_phenotypes:
            st.info(f"Consensus logic will enforce exclusion for: {exclude_phenotypes[:5]}...")


        with st.status("Running Consensus ML Pipeline...", expanded=True) as status:
            try:
                status.write(f"[{_timestamp()}] Starting ML consensus analysis...")
                _log_console(f"Starting analysis")
                _log_console(f"Unified CSV: {full_input_path}")
                _log_console(f"Output dir: {out_dir}")
                result_holder: dict[str, pd.DataFrame] = {}

                def _capture_results(result_frame: pd.DataFrame) -> None:
                    result_holder["dataframe"] = result_frame

                config = ConsensusConfig(
                    unified_csv=Path(full_input_path),
                    output_dir=out_dir,
                    phenotyping_methods=cfg["phenotyping_methods"],
                    clustering_markers=confirmed_markers,
                    morphology_cols=cfg["morphology_cols"],
                    compartment_cols=cfg["compartment_cols"],
                    feature_weights=cfg["feature_weights"],
                    use_z_normalization=cfg["use_z_normalization"],
                    batch_normalization_col=cfg["batch_normalization_col"],
                    min_consensus_methods=cfg["min_consensus_methods"],
                    test_size=cfg["test_size"],
                    n_estimators=cfg["n_estimators"],
                    random_state=cfg["random_state"],
                    min_training_cells=cfg["min_training_cells"],
                    figure_dpi=cfg["figure_dpi"],
                    write_backup_csv=bool(cfg.get("write_backup_csv", False)),
                    update_source_csv=False,
                    excluded_phenotypes=exclude_phenotypes,
                    output_phenotype_col=output_phenotype_col,
                    confidence_col_name=confidence_col,
                    consensus_col_name=consensus_col_name,
                    refined_confidence_threshold=float(cfg.get("refined_confidence_threshold", 0.75)),
                    refined_low_confidence_label=cfg.get("refined_low_confidence_label", "Others"),
                    ml_method=cfg.get("ml_method", "random_forest"),
                    validation_ground_truth_column=cfg.get("validation_ground_truth_col") or None,
                )

                ok = run_ml_consensus_analysis(
                    config,
                    progress_callback=status.write,
                    preloaded_df=preloaded_df,
                    results_callback=_capture_results,
                )

                if not ok:
                    status.error(f"[{_timestamp()}] Consensus ML failed.")
                    return

                status.write(f"[{_timestamp()}] Loading results...")
                df_results = result_holder.get("dataframe")
                if df_results is None:
                    status.warning(
                        f"[{_timestamp()}] Consensus ML did not return reviewable assignments."
                    )
                    return
                phenotype_col = output_phenotype_col
                confidence_col = confidence_col

                if output_phenotype_col not in df_results.columns:
                    status.warning(f"[{_timestamp()}] {output_phenotype_col} not found in output")
                    return

                unique_phenos = sorted(df_results[phenotype_col].dropna().unique().tolist())
                pheno_to_cluster = {p: i for i, p in enumerate(unique_phenos)}
                df_results["ml_consensus_cluster"] = df_results[phenotype_col].map(pheno_to_cluster).fillna(-1)

                temp_file = project_root / "outputs/phenotyping/cache" / f"ml_consensus_results_{int(time.time())}.parquet"
                df_save = df_results.copy()
                for _col in df_save.select_dtypes(include=["object"]).columns:
                    df_save[_col] = df_save[_col].astype(str)
                df_save.to_parquet(temp_file)
                status.write(f"[{_timestamp()}] Saving cached results for interactive review...")

                st.session_state["mlc_temp_file"] = str(temp_file)
                st.session_state["mlc_phenotype_map"] = {int(c): p for p, c in pheno_to_cluster.items()}
                st.session_state["mlc_marker_cols"] = confirmed_markers
                st.session_state["mlc_outcol"] = phenotype_col
                st.session_state["mlc_conf_col"] = confidence_col
                st.session_state["mlc_cluster_col"] = "ml_consensus_cluster"
                st.session_state["mlc_consensus_col"] = consensus_col_name
                st.session_state["mlc_refined_col"] = refined_col
                # Keep result-linking metadata separate from any present or
                # future widget-owned Cell ID key.
                st.session_state["mlc_result_cell_id_col"] = cell_id_col
                st.session_state["mlc_input_path"] = str(full_input_path)

                st.session_state["mlc_run_log"] = "Analysis completed successfully. Check terminal for full logs."

                status.update(label="Consensus ML Complete!", state="complete")
                st.success(
                    "Consensus ML phenotyping complete. Review the assignments below; "
                    "the unified CSV has not been changed."
                )
                finish_phenotyping_run_audit(
                    run_audit,
                    status="success",
                    output_path=temp_file,
                    input_rows=audit_input_rows,
                    output_rows=len(df_results),
                    output_column=phenotype_col,
                    cluster_column="ml_consensus_cluster",
                    phenotype_map=st.session_state["mlc_phenotype_map"],
                    extra={
                        "confidence_column": confidence_col,
                        "consensus_column": consensus_col_name,
                        "refined_column": refined_col,
                        "validation_reference_column": cfg.get("validation_ground_truth_col"),
                        "automatic_result_csv": bool(
                            cfg.get("write_backup_csv", False)
                        ),
                        "source_csv_updated": False,
                        "filtered_run": bool(filtered_mode),
                        "phenotype_cell_counts": df_results[phenotype_col].value_counts(dropna=False).to_dict(),
                    },
                )
                run_audit = None
            except Exception as e:
                finish_phenotyping_run_audit(
                    run_audit,
                    status="failed",
                    input_rows=audit_input_rows,
                    output_column=output_phenotype_col,
                    cluster_column="ml_consensus_cluster",
                    error=str(e),
                )
                run_audit = None
                status.error(f"[{_timestamp()}] Pipeline error: {e}")
                logger.error(f"Consensus ML pipeline error: {e}")
            finally:
                if run_audit is not None:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_phenotype_col,
                        cluster_column="ml_consensus_cluster",
                        error="Pipeline stopped before completion; see the Consensus ML status messages.",
                    )

    if "mlc_temp_file" in st.session_state and Path(st.session_state["mlc_temp_file"]).exists():
        st.divider()
        st.subheader("Results & Phenotype Assignment")
        st.caption("Interactive visualization and export of Consensus ML results")

        temp_file = st.session_state["mlc_temp_file"]
        df_results = pd.read_parquet(temp_file)

        # Recover metadata for review caches created by an older UI version.
        if "mlc_result_cell_id_col" not in st.session_state:
            if TAB7_INTERNAL_CELL_ID_COL in df_results.columns:
                result_cell_id_col = TAB7_INTERNAL_CELL_ID_COL
            else:
                result_cell_id_col = "cell_id"
            st.session_state["mlc_result_cell_id_col"] = result_cell_id_col
        if "mlc_input_path" not in st.session_state and full_input_path:
            st.session_state["mlc_input_path"] = str(full_input_path)
        marker_cols = st.session_state.get("mlc_marker_cols", [])
        phenotype_map = st.session_state.get("mlc_phenotype_map", {})
        output_col = st.session_state.get("mlc_outcol", "phenotype_ml_consensus")
        confidence_col = st.session_state.get("mlc_conf_col", "ml_consensus_score")
        consensus_col_name = st.session_state.get("mlc_consensus_col", "consensus")
        refined_col = st.session_state.get(
            "mlc_refined_col", f"{output_col}_refined"
        )
        cluster_col = st.session_state.get("mlc_cluster_col", "ml_consensus_cluster")

        st.markdown("### Phenotype Summary & Expression Heatmap")

        if output_col in df_results.columns:
            pheno_summary = (
                df_results[output_col]
                .fillna("Unassigned")
                .value_counts()
                .rename_axis("Phenotype")
                .reset_index(name="Cell Count")
            )
            pheno_summary["% Total"] = (pheno_summary["Cell Count"] / len(df_results) * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(pheno_summary, width="stretch", hide_index=True)

        numeric_cols = [c for c in marker_cols if c in df_results.columns and pd.api.types.is_numeric_dtype(df_results[c])]
        if numeric_cols:
            df_valid = df_results[df_results[cluster_col].notna()].copy()
            if not df_valid.empty:
                mean_expression = df_valid.groupby(cluster_col)[numeric_cols].mean()
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

                # Cache Z-scores for Visual Validation alignment
                st.session_state["mlc_heatmap_z"] = heatmap_data

                render_phenotype_statistics(
                    df_results=df_results,
                    cluster_col=cluster_col,
                    phenotype_map=phenotype_map,
                    heatmap_z_ordered=heatmap_data,
                    cluster_label="Phenotype",
                )

                render_publication_heatmap(
                    heatmap_z_ordered=heatmap_data,
                    phenotype_map=phenotype_map,
                    key_prefix="mlc",
                    df_results=df_results,
                    group_col=cluster_col,
                    marker_columns=marker_cols,
                )
        else:
            st.warning("No numeric marker columns available for heatmap")

        st.divider()
        st.markdown("### Save and Export Results")
        render_save_assignments(
            df_results=df_results,
            phenotype_map=phenotype_map,
            output_col=output_col,
            project_root=project_root,
            key_prefix="mlc",
            cluster_col=cluster_col,
            update_cluster_cols=[
                cluster_col,
                output_col,
                confidence_col,
                consensus_col_name,
                refined_col,
            ],
            cell_id_col=st.session_state.get("mlc_result_cell_id_col", "cell_id"),
            input_path_override=st.session_state.get("mlc_input_path"),
        )

        render_export_split_data(
            df_results=df_results,
            output_col=output_col,
            phenotype_map=phenotype_map,
            project_root=project_root,
            key_prefix="mlc",
            cluster_col=cluster_col,
            cell_id_col=st.session_state.get("mlc_result_cell_id_col", "cell_id"),
        )

    # --- VISUAL VALIDATION COMPONENT ---
    if st.session_state.get('viz_enabled', True):
        st.divider()
        with st.expander("Visual Validation of Representative Cells", expanded=True):
            st.markdown("#### Visualize Representative Cells")
            st.caption("Inspect random cells from specific phenotypes to validate assignments.")
            
            # Ensure we have data
            if 'mlc_temp_file' in st.session_state and Path(st.session_state['mlc_temp_file']).exists():
                # Load data if not already
                if 'df_results' not in locals():
                    df_results = pd.read_parquet(st.session_state['mlc_temp_file'])
                
                # Check Global Viz Config
                viz_roi = st.session_state.get('viz_roi_dir')
                viz_seg = st.session_state.get('viz_seg_dir')
                
                if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
                    st.warning("Please configure valid ROI and Segmentation directories in Global Input Configuration (Top of Tab 7).")
                else:
                    viz = CellVisualizer(viz_roi, viz_seg)
                    
                    # File matching (cache in session state)
                    if not st.session_state.get('mlc_viz_file_matches'):
                         # Get unique batches
                         batch_col = "batch_id"
                         if batch_col not in df_results.columns:
                             batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                         
                         if batch_col in df_results.columns:
                             unique_batches = df_results[batch_col].unique().astype(str).tolist()
                             matches = viz.match_file_triads(unique_batches)
                             st.session_state.mlc_viz_file_matches = matches
                             if not matches:
                                 st.error("No matching image files found for these batches.")
                         else:
                             st.error(f"Batch column '{batch_col}' not found in results.")
                    else:
                        viz.file_map = st.session_state.mlc_viz_file_matches

                    # UI Controls
                    output_col = st.session_state.get("mlc_outcol", "phenotype_ml_consensus")
                    
                    available_phenos = []
                    if output_col in df_results.columns:
                        available_phenos = sorted([p for p in df_results[output_col].unique() if pd.notna(p) and str(p) != 'Unassigned'])
                    
                    with st.form("mlc_viz_form"):
                         vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])
                         with vv_col1:
                             selected_pheno_vv = st.selectbox(
                                 "Phenotype to inspect", 
                                 available_phenos,
                                 key="mlc_viz_pheno_sel"
                             )
                         with vv_col2:
                             n_viz_cells = st.number_input("How many cells to show", 1, 50, 5, key="mlc_viz_n_cells")
                         with vv_col3:
                             show_neighbors = st.checkbox("Show Neighbors", value=True, key="mlc_viz_neighbors")
                         with vv_col4:
                             mlc_batch_scope = st.radio(
                                 "Which images to draw from",
                                 ["All matched batches", "Specific batch"],
                                 horizontal=True,
                                 key="mlc_viz_batch_scope",
                             )
                             mlc_batch_options = sorted(str(batch) for batch in st.session_state.mlc_viz_file_matches.keys())
                             selected_batch_vv = None
                             if mlc_batch_scope == "Specific batch":
                                 selected_batch_vv = st.selectbox(
                                     "Batch",
                                     mlc_batch_options,
                                     key="mlc_viz_batch_sel",
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
                            # 1. Determine Top 5 Markers (Using Cached Z-Scores for Alignment)
                            heatmap_z = st.session_state.get("mlc_heatmap_z")
                            phenotype_map = st.session_state.get("mlc_phenotype_map", {})
                            
                            found_z_scores = False
                            top_5_markers = []
                            top_5_values = []
                            
                            if heatmap_z is not None and not heatmap_z.empty:
                                # Find clusters associated with this phenotype
                                clusters_in_pheno = [c for c, p in phenotype_map.items() if p == selected_pheno_vv]
                                
                                # Filter columns that exist in heatmap_z
                                avail_cols = heatmap_z.columns
                                relevant_cols = []
                                for c in clusters_in_pheno:
                                    if c in avail_cols:
                                        relevant_cols.append(c)
                                    elif str(c) in avail_cols:
                                        relevant_cols.append(str(c))
                                    elif int(c) in avail_cols:
                                        relevant_cols.append(int(c))
                                
                                if relevant_cols:
                                    pheno_vec = heatmap_z[relevant_cols].mean(axis=1)
                                    top5 = pheno_vec.nlargest(5)
                                    top_5_markers = top5.index.tolist()
                                    top_5_values = top5.values.tolist()
                                    found_z_scores = True
                            
                            if not found_z_scores:
                                # Start of Fallback (Analysis Cols)
                                analysis_markers = st.session_state.get('mlc_marker_cols', [])
                                valid_analysis_markers = [c for c in analysis_markers if c in df_results.columns]
                                norm_cols = valid_analysis_markers if valid_analysis_markers else [c for c in df_results.columns if '_norm' in c]
                                
                                if norm_cols:
                                    cluster_means = subset[norm_cols].mean()
                                    top5 = cluster_means.nlargest(5)
                                    top_5_markers = top5.index.tolist()
                                    top_5_values = top5.values.tolist()
                            
                            if top_5_markers:
                                # Clean names for display and OME matching (User Request)
                                top_5_names = []
                                for m in top_5_markers:
                                    # Strip suffixes to get RAW name
                                    clean = str(m).replace('_norm_resolve', '').replace('_resolve', '').replace('_norm', '').replace('_corr', '').replace('_unmixed', '')
                                    top_5_names.append(clean)
                                    
                                # Format display string with values
                                display_strs = []
                                for m, val, clean_name in zip(top_5_markers, top_5_values, top_5_names):
                                    display_strs.append(f"{clean_name} ({val:.2f})")
                                    
                                st.info(f"Top expressed markers for {selected_pheno_vv}: {', '.join(display_strs)}")
                                
                                # 2. Sample Cells
                                batch_col = "batch_id"
                                if batch_col not in df_results.columns:
                                     batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                                
                                if batch_col not in df_results.columns:
                                    st.error(f"Batch column '{batch_col}' not found in results.")
                                    subset_valid = pd.DataFrame()
                                elif not st.session_state.get('mlc_viz_file_matches'):
                                    st.error("No matching image files found for visualization batches.")
                                    subset_valid = pd.DataFrame()
                                else:
                                    valid_batches = list(st.session_state.mlc_viz_file_matches.keys())
                                    subset_valid = subset[subset[batch_col].astype(str).isin(valid_batches)]
                                    if mlc_batch_scope == "Specific batch" and selected_batch_vv:
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
                                    
                                    cid_col = "cell_id" 
                                    if cid_col not in df_results.columns:
                                        cid_col = next((c for c in ['cell_id', 'ObjectNumber', 'CellID'] if c in df_results.columns), 'cell_id')

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
                                                    'method': 'consensus_ml',
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
                            else:
                                st.warning("Could not identify marker columns.")

        st.divider()
        st.markdown("### Run Log")
        run_log = st.session_state.get("mlc_run_log", "")
        if run_log:
            st.code(run_log)
        else:
            st.info("No run log captured yet. Run Consensus ML to view terminal-style output.")
