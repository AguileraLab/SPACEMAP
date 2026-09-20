"""
MR ML Phenotyping Subtab
Machine learning phenotyping based on manual reference (Step9A).
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

from .common import (
    TAB7_INTERNAL_CELL_ID_COL,
    ensure_cell_id_anchor,
    logger,
    get_batch_id_column_help_text,
    get_cell_id_column_help_text,
    get_enable_row_filter_help_text,
    get_filter_column_help_text,
    get_filter_value_help_text,
    get_marker_list,
    get_marker_workspace_panel,
    render_method_marker_selection,
    normalize_optional_column_name,
    resolve_saved_column_default,
    get_output_column_help_text,
    project_root_calc,
    get_random_seed_help_text,
    render_phenotype_statistics,
    render_publication_heatmap,
    render_save_assignments,
    render_export_split_data,
    _extract_markers_from_notebook,
    render_settings_export_import,
    render_phenotyping_run_log_controls,
    begin_phenotyping_run_audit,
    finish_phenotyping_run_audit,
    resolve_notebook_marker_defaults,
)

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None
try:
    import sys
    from spacemap import ml_mr_utils
    sys.modules["ml_mr_utils"] = ml_mr_utils
    from spacemap.ml_mr_utils import run_ml_mr_analysis
except ImportError:
    import importlib.util
    try:
        # Check parents[3] / "spacemap" / "ml_mr_utils.py" (current workspace package structure)
        _ML_MR_PATH = Path(__file__).resolve().parents[3] / "spacemap" / "ml_mr_utils.py"
        if not _ML_MR_PATH.exists():
            # Check parents[4] / "src" / "spacemap" / "ml_mr_utils.py"
            _ML_MR_PATH = Path(__file__).resolve().parents[4] / "src" / "spacemap" / "ml_mr_utils.py"
        if not _ML_MR_PATH.exists():
            # Check legacy V13 path
            _ML_MR_PATH = Path(__file__).resolve().parents[3] / "src" / "ml_mr_utils.py"

        if _ML_MR_PATH.exists():
            _ph_spec = importlib.util.spec_from_file_location("ml_mr_utils", _ML_MR_PATH)
            if _ph_spec is not None and _ph_spec.loader is not None:
                ml_mr_utils = importlib.util.module_from_spec(_ph_spec)
                sys.modules.setdefault("ml_mr_utils", ml_mr_utils)
                _ph_spec.loader.exec_module(ml_mr_utils)
                run_ml_mr_analysis = ml_mr_utils.run_ml_mr_analysis
            else:
                run_ml_mr_analysis = None
        else:
            run_ml_mr_analysis = None
    except Exception:
        run_ml_mr_analysis = None
except Exception:
    run_ml_mr_analysis = None


NOTEBOOK_NAME = "Step9A_ML_manual_reference.ipynb"
INTERNAL_VALIDATION_OPTION = "Internal stratified cell split (exploratory)"


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


def _select_default(options: list[str], preferred: list[str]) -> str:
    for value in preferred:
        if value and value in options:
            return value
    return options[0]


def _scan_selected_phenotypes(
    csv_path: str | Path,
    columns: list[str],
) -> list[str]:
    """Read only selected label columns and return cleaned unique phenotypes."""
    scan_columns = list(dict.fromkeys(str(column) for column in columns if column))
    if not scan_columns:
        return []

    labels_by_normalized_name: dict[str, str] = {}
    for chunk in pd.read_csv(csv_path, usecols=scan_columns, chunksize=100_000):
        for column in scan_columns:
            values = chunk[column].dropna().astype("string").str.strip()
            for value in values[values.ne("")].unique():
                display = str(value).strip()
                labels_by_normalized_name.setdefault(display.lower(), display)
    return sorted(labels_by_normalized_name.values(), key=str.lower)


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _log_console(message: str) -> None:
    print(message, flush=True)


def _extract_ml_mr_config() -> dict[str, Any]:
    """Extract MR-ML config values from Step9A notebook."""
    config: dict[str, Any] = {
        "unified_csv": None,
        "output_dir": None,
        "ground_truth_col": "phenotype_ref1",
        "validation_ground_truth_col": None,
        "output_phenotype_col": "phenotype_ml_mr",
        "confidence_col": "ml_mr_confidence",
        "refined_output_phenotype_col": "phenotype_ml_mr_refined",
        "refined_confidence_threshold": 0.75,
        "refined_low_confidence_label": "Others",
        "write_backup_csv": False,
        "use_z_normalization": False,
        "batch_normalization_col": "batch_id",
        "enable_filter": True,
        "filter_col": "tissue",
        "filter_val": "1",
        "phenotyping_methods": [],
        "exclude_phenotypes": ["Others"],
        "morphology_cols": [],
        "compartment_cols": [],
        "feature_weights": {"markers": 1.0, "morphology": 0.0, "compartments": 0.0},
        "test_size": 0.3,
        "n_estimators": 1500,
        "random_state": 42,
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
            if "OUTPUT_DIR" in source and config["output_dir"] is None:
                val = _extract_scalar("OUTPUT_DIR", source)
                if val:
                    config["output_dir"] = val

            if "GROUND_TRUTH_COLUMN" in source:
                val = _extract_scalar("GROUND_TRUTH_COLUMN", source)
                if val:
                    config["ground_truth_col"] = val

            if "VALIDATION_GROUND_TRUTH_COLUMN" in source:
                val = _extract_scalar("VALIDATION_GROUND_TRUTH_COLUMN", source)
                config["validation_ground_truth_col"] = (
                    None if val in {None, "None", "null", ""} else val
                )

            if "OUTPUT_PHENOTYPE_COL" in source or "OUTPUT_PHENOTYPE_COLUMN" in source:
                val = _extract_scalar("OUTPUT_PHENOTYPE_COL", source)
                if val is None:
                    val = _extract_scalar("OUTPUT_PHENOTYPE_COLUMN", source)
                if val:
                    config["output_phenotype_col"] = val

            if "CONFIDENCE_COL_NAME" in source:
                val = _extract_scalar("CONFIDENCE_COL_NAME", source)
                if val:
                    config["confidence_col"] = val

            if "REFINED_OUTPUT_PHENOTYPE_COL" in source:
                val = _extract_scalar("REFINED_OUTPUT_PHENOTYPE_COL", source)
                if val:
                    config["refined_output_phenotype_col"] = val
            if "REFINED_CONFIDENCE_THRESHOLD" in source:
                val = _extract_scalar("REFINED_CONFIDENCE_THRESHOLD", source)
                try:
                    if val is not None:
                        config["refined_confidence_threshold"] = float(val)
                except (TypeError, ValueError):
                    pass
            if "REFINED_LOW_CONFIDENCE_LABEL" in source:
                val = _extract_scalar("REFINED_LOW_CONFIDENCE_LABEL", source)
                if val:
                    config["refined_low_confidence_label"] = val
            if "WRITE_BACKUP_CSV" in source:
                val = _extract_scalar("WRITE_BACKUP_CSV", source)
                if val in {"True", "False"}:
                    config["write_backup_csv"] = val == "True"

            if "PHENOTYPING_METHODS" in source:
                config["phenotyping_methods"] = _extract_list("PHENOTYPING_METHODS", source)
            if "EXCLUDED_PHENOTYPES" in source:
                config["exclude_phenotypes"] = _extract_list(
                    "EXCLUDED_PHENOTYPES", source
                )

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

            if "ML_METHOD" in source:
                val = _extract_scalar("ML_METHOD", source)
                if val:
                    config["ml_method"] = val

        return config
    except Exception as e:
        logger.warning(f"MR ML config extraction failed: {e}")
        return config


def render_mr_ml(project_root: Path, global_input_path: str | None) -> None:
    st.subheader("MR ML Phenotyping")
    render_phenotyping_run_log_controls(Path(project_root), "mrml", "MR ML Phenotyping")
    st.markdown("Machine learning phenotyping using manual reference (Step9A).")

    cache_dir = project_root / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mr_cache = cache_dir / "mr_ml_params.json"

    cached_params: dict[str, Any] = {}
    if mr_cache.exists():
        try:
            with open(mr_cache, "r") as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}

    # The loaded project JSON is authoritative. The method-local JSON remains
    # a backwards-compatible fallback for projects created by older versions.
    project_params = st.session_state.get("mrml_config")
    if isinstance(project_params, dict):
        cached_params = {**cached_params, **project_params}

    # --- Settings Export / Import ---
    render_settings_export_import(
        "MR ML",
        ["mrml_"],
        {"mr_ml_params.json": mr_cache},
    )

    nb_config = _extract_ml_mr_config()

    st.markdown("##### Input File Selection")
    use_global = st.checkbox("Use Global Input File", value=cached_params.get("use_global", True), key="mrml_use_global")
    full_input_path = global_input_path

    if not use_global:
        default_dir = cached_params.get("input_dir", str(project_root / "outputs/batch_corrected"))
        input_dir = st.text_input("Input Directory", default_dir, key="mrml_input_dir")
        if input_dir and os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        else:
            files = []
        if files:
            cached_file = cached_params.get("input_file", files[0])
            default_idx = files.index(cached_file) if cached_file in files else 0
            sel = st.selectbox("Input File", files, index=default_idx, key="mrml_input_file")
            full_input_path = os.path.join(input_dir, sel)

        if cached_params.get("input_dir") != input_dir or cached_params.get("input_file") != os.path.basename(full_input_path or ""):
            try:
                cached_params["input_dir"] = input_dir
                cached_params["input_file"] = os.path.basename(full_input_path) if full_input_path else ""
                with open(mr_cache, "w") as f:
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
        key_prefix="mrml",
        method_name="MR-ML",
        full_input_path=full_input_path,
        available_columns=available_marker_columns,
        shared_panel=all_marker_candidates,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=expected_markers,
        saved_markers=cached_params.get("markers", []),
        config_state_key="mrml_config",
    )

    st.divider()
    st.markdown("### 3. Configuration (Notebook Synchronized)")

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
            "ground_truth_col": cached_params.get("ground_truth_col", nb_config.get("ground_truth_col", "phenotype_ref1")),
            "validation_ground_truth_col": cached_params.get(
                "validation_ground_truth_col", nb_config.get("validation_ground_truth_col")
            ),
            "output_dir": str(project_root / "outputs/phenotyping/ML_MR"),
            "output_phenotype_col": cached_params.get("output_phenotype_col", nb_config.get("output_phenotype_col", "phenotype_ml_mr")),
            "confidence_col": cached_params.get("confidence_col", nb_config.get("confidence_col", "ml_mr_confidence")),
            "refined_output_phenotype_col": cached_params.get(
                "refined_output_phenotype_col",
                nb_config.get("refined_output_phenotype_col", "phenotype_ml_mr_refined"),
            ),
            "refined_confidence_threshold": cached_params.get(
                "refined_confidence_threshold",
                nb_config.get("refined_confidence_threshold", 0.75),
            ),
            "refined_low_confidence_label": cached_params.get(
                "refined_low_confidence_label",
                nb_config.get("refined_low_confidence_label", "Others"),
            ),
            "write_backup_csv": cached_params.get(
                "write_backup_csv", nb_config.get("write_backup_csv", False)
            ),
            "use_z_normalization": cached_params.get("use_z_normalization", nb_config.get("use_z_normalization", False)),
            "batch_normalization_col": cached_params.get("batch_normalization_col", nb_config.get("batch_normalization_col", "batch_id")),
            "cell_id_col": cached_params.get("cell_id_col", nb_config.get("cell_id_col", "cell_id")),
            "enable_filter": cached_params.get("enable_filter", nb_config.get("enable_filter", True)),
            "filter_col": cached_params.get("filter_col", nb_config.get("filter_col", "tissue")),
            "filter_val": cached_params.get("filter_val", nb_config.get("filter_val", "1")),
            "phenotyping_methods": cached_params.get("phenotyping_methods", nb_config.get("phenotyping_methods", [])),
            "exclude_phenotypes": cached_params.get(
                "exclude_phenotypes", nb_config.get("exclude_phenotypes", ["Others"])
            ),
            "morphology_cols": cached_params.get("morphology_cols", nb_config.get("morphology_cols", [])),
            "compartment_cols": cached_params.get("compartment_cols", nb_config.get("compartment_cols", [])),
            "feature_weights": cached_params.get("feature_weights", nb_config.get("feature_weights", {"markers": 1.0, "morphology": 0.0, "compartments": 0.0})),
            "test_size": cached_params.get("test_size", nb_config.get("test_size", 0.3)),
            "n_estimators": cached_params.get("n_estimators", nb_config.get("n_estimators", 1500)),
            "random_state": cached_params.get("random_state", nb_config.get("random_state", 42)),
            "ml_method": cached_params.get("ml_method", nb_config.get("ml_method", "random_forest")),
        }
    if not isinstance(st.session_state.get("mrml_config"), dict):
        st.session_state["mrml_config"] = default_config
    else:
        for config_key, default_value in default_config.items():
            st.session_state["mrml_config"].setdefault(config_key, default_value)
    # Migrate already-open Streamlit sessions created before these options
    # existed, without requiring the user to restart the application.
    st.session_state["mrml_config"].setdefault(
        "exclude_phenotypes",
        cached_params.get(
            "exclude_phenotypes", nb_config.get("exclude_phenotypes", ["Others"])
        ),
    )
    st.session_state["mrml_config"].setdefault("write_backup_csv", False)
    st.session_state["mrml_config"]["output_dir"] = str(
        project_root / "outputs/phenotyping/ML_MR"
    )

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
        st.warning(
            "No manual-reference label columns were detected in the selected CSV. "
            "Select the enriched/merged table containing phenotype_ref1a and phenotype_ref1b."
        )

    output_dir = str(project_root / "outputs/phenotyping/ML_MR")
    with st.form("mrml_config_form", clear_on_submit=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            if not reference_candidates:
                reference_candidates = [
                    str(st.session_state["mrml_config"].get("ground_truth_col") or "phenotype_ref1")
                ]
            saved_training_reference = str(
                st.session_state["mrml_config"].get("ground_truth_col") or ""
            )
            default_training_reference = _select_default(
                reference_candidates,
                [
                    saved_training_reference
                    if saved_training_reference not in {"", "phenotype_ref1"}
                    else "",
                    "phenotype_ref1a",
                    saved_training_reference,
                    "phenotype_ref1",
                ],
            )
            ground_truth_col = st.selectbox(
                "Train on these manual labels",
                options=reference_candidates,
                index=reference_candidates.index(default_training_reference),
                key="mrml_gt_col",
                help=(
                    "The column of hand-assigned phenotypes the model learns from. It is also "
                    "used to score each component method, which sets how much weight that "
                    "method's vote carries. Pick your largest, most trusted annotation set."
                ),
            )
            validation_options = [INTERNAL_VALIDATION_OPTION] + [
                column for column in reference_candidates if column != ground_truth_col
            ]
            saved_validation = st.session_state["mrml_config"].get("validation_ground_truth_col")
            preferred_validation = [
                str(saved_validation or ""),
                "phenotype_ref1b" if ground_truth_col == "phenotype_ref1a" else "",
                INTERNAL_VALIDATION_OPTION,
            ]
            default_validation = _select_default(validation_options, preferred_validation)
            validation_selection = st.selectbox(
                "Score the result against these labels",
                options=validation_options,
                index=validation_options.index(default_validation),
                key="mrml_validation_gt_col",
                help=(
                    "A second annotation set, used only after training, to measure how well the "
                    "model did. It should cover different cells from the training column - a cell "
                    "labelled in both is one the model already saw.\n\n"
                    "This checks performance on unseen *cells*. It does not show performance on an "
                    "unseen *patient*; for that, use the patient-out validation in the F1 Score tab. "
                    "Choosing the internal split instead just holds back a slice of the training "
                    "labels, which is weaker still."
                ),
            )
            validation_ground_truth_col = (
                None if validation_selection == INTERNAL_VALIDATION_OPTION else validation_selection
            )
            drop_overlapping_validation = st.checkbox(
                "Drop cells labelled in both columns",
                value=bool(
                    st.session_state["mrml_config"].get("drop_overlapping_validation", True)
                ),
                key="mrml_drop_overlap",
                disabled=validation_selection == INTERNAL_VALIDATION_OPTION,
                help=(
                    "Cells carrying a label in both columns are removed from the scoring set, so "
                    "the model is never graded on a cell it trained on. Training is unaffected.\n\n"
                    "Leave this on unless you want the run to stop when the two sets touch."
                ),
            )
            st.session_state["mrml_config"]["drop_overlapping_validation"] = bool(
                drop_overlapping_validation
            )
            output_phenotype_col = st.text_input(
                "Output Phenotype Column",
                value=st.session_state["mrml_config"]["output_phenotype_col"],
                key="mrml_output_pheno_col",
                help=get_output_column_help_text("ML-MR phenotype labels"),
            )
            confidence_col = st.text_input(
                "Confidence Score Column",
                value=st.session_state["mrml_config"].get("confidence_col", "ml_mr_confidence"),
                key="mrml_confidence_col",
                help=(
                    "Column that stores the winning phenotype's fraction of the accumulated model-plus-method support. "
                    "It is useful for ranking uncertainty but is not a calibrated probability."
                ),
            )
            st.markdown("**Refined Phenotype Output**")
            refined_output_phenotype_col = st.text_input(
                "Refined Output Phenotype Column",
                value=st.session_state["mrml_config"].get(
                    "refined_output_phenotype_col", "phenotype_ml_mr_refined"
                ),
                key="mrml_refined_output_pheno_col",
                help=(
                    "Separate output column that preserves the primary MR-ML result while replacing "
                    "cells below the configured support threshold with the low-confidence label."
                ),
            )
            refined_threshold = st.number_input(
                "Confidence Threshold for Refined Column",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    st.session_state["mrml_config"].get(
                        "refined_confidence_threshold", 0.75
                    )
                ),
                step=0.05,
                key="mrml_refined_threshold",
                help=(
                    "Cells with MR-ML ensemble-support strictly below this value are relabeled only "
                    "in the refined column. The default is 0.75. This score is not a calibrated probability."
                ),
            )
            refined_label = st.text_input(
                "Low-Confidence Label",
                value=st.session_state["mrml_config"].get(
                    "refined_low_confidence_label", "Others"
                ),
                key="mrml_refined_label",
                help="Phenotype assigned below the threshold in the refined column (default: 'Others').",
            )
            st.markdown("**Run-time Storage**")
            write_backup_csv = st.checkbox(
                "Automatically save a separate ML-MR result CSV",
                value=bool(
                    st.session_state["mrml_config"].get("write_backup_csv", False)
                ),
                key="mrml_write_backup_csv",
                help=(
                    "Off by default. When off, running ML-MR does not write any result CSV or "
                    "change the unified CSV. Review the results first, then use "
                    "'Save All Assignments to Disk'."
                ),
            )
        with col2:
            use_z_norm = st.checkbox(
                "Use Z-normalization",
                value=bool(st.session_state["mrml_config"]["use_z_normalization"]),
                key="mrml_use_z",
                help=(
                    "Applies batch-wise z-normalization to marker features before training and prediction. If the chosen batch column is blank or missing, the backend logs a warning and skips this step instead of failing."
                ),
            )
            batch_norm_col = st.text_input(
                "Batch Normalization Column",
                value=st.session_state["mrml_config"]["batch_normalization_col"],
                key="mrml_batch_norm_col",
                help=(
                    get_batch_id_column_help_text()
                    + " In MR-ML this column is only used when `Use Z-normalization` is enabled; otherwise it is ignored."
                ),
            )
            cell_id_col_cfg = st.text_input(
                "Cell ID Column",
                value=st.session_state["mrml_config"].get("cell_id_col") or "",
                key="mrml_cell_id_col",
                help=get_cell_id_column_help_text(),
            )

            st.markdown("**Filtering**")
            enable_filter = st.checkbox(
                "Enable Row Filtering",
                value=bool(st.session_state["mrml_config"].get("enable_filter", False)),
                key="mrml_enable_filter",
                help=get_enable_row_filter_help_text("training and prediction"),
            )
            default_filter_col = resolve_saved_column_default(
                st.session_state["mrml_config"].get("filter_col"),
                available_columns,
            )
            filter_col = st.selectbox(
                "Filter Column",
                available_columns if available_columns else [default_filter_col],
                index=available_columns.index(default_filter_col) if default_filter_col in available_columns else 0,
                key="mrml_filter_col",
                help=get_filter_column_help_text("training and prediction"),
            )
            filter_val = st.text_input(
                "Filter Value",
                value=str(st.session_state["mrml_config"].get("filter_val", "1")),
                key="mrml_filter_val",
                help=get_filter_value_help_text(),
            )

        st.markdown("**Phenotyping Methods (phenotype_*)**")
        method_defaults = st.session_state["mrml_config"]["phenotyping_methods"]
        _mrml_options = [
            column
            for column in phenotyping_candidates
            if not _looks_like_manual_reference(column)
            and not _looks_like_annotation_metadata(column)
            and not column.startswith("phenotype_ml_")
            and column not in {ground_truth_col, validation_ground_truth_col, output_phenotype_col}
        ]
        if not method_defaults and _mrml_options:
            method_defaults = _mrml_options
        _mrml_safe_defaults = [m for m in method_defaults if m in _mrml_options]
        phenotyping_methods = st.multiselect(
            "Methods to integrate",
            _mrml_options,
            default=_mrml_safe_defaults,
            key="mrml_methods",
            help=(
                "Automated phenotype columns used as ensemble votes. Manual-reference columns and prior ML outputs are excluded to prevent target leakage and circular predictions. "
                "Per-method F1 weights are calculated from the training reference only."
            ),
        )

        st.markdown("**Exclude Phenotypes from ML MR**")
        if st.form_submit_button(
            "Scan Phenotypes", type="secondary", key="mrml_scan_phenotypes"
        ):
            if full_input_path and os.path.exists(full_input_path):
                try:
                    scan_columns = [
                        column
                        for column in [
                            ground_truth_col,
                            validation_ground_truth_col,
                            *phenotyping_methods,
                        ]
                        if column and column in available_columns
                    ]
                    if scan_columns:
                        detected_labels = _scan_selected_phenotypes(
                            full_input_path, scan_columns
                        )
                        st.session_state["mrml_exclude_candidates"] = detected_labels
                        st.session_state["mrml_exclude_scan_columns"] = scan_columns
                    else:
                        st.warning("No selected phenotype columns are available to scan.")
                except Exception as exc:
                    st.warning(f"Failed to scan phenotypes: {exc}")
            else:
                st.warning("Select an available unified CSV before scanning phenotypes.")

        default_excluded = list(
            st.session_state["mrml_config"].get("exclude_phenotypes", []) or []
        )
        prior_widget_exclusions = list(
            st.session_state.get("mrml_exclude_phenotypes", []) or []
        )
        scanned_candidates = list(
            st.session_state.get("mrml_exclude_candidates", []) or []
        )
        exclude_candidates = sorted(
            set(scanned_candidates)
            | set(default_excluded)
            | set(prior_widget_exclusions),
            key=str.lower,
        )
        valid_defaults = [
            phenotype
            for phenotype in default_excluded
            if phenotype in exclude_candidates
        ]
        if scanned_candidates:
            scanned_columns = st.session_state.get(
                "mrml_exclude_scan_columns", []
            )
            st.caption(
                f"Detected {len(scanned_candidates)} phenotype labels across "
                f"{len(scanned_columns)} selected columns: "
                + ", ".join(scanned_candidates)
            )
        else:
            st.caption(
                "Scan the selected training, validation, and automated-method columns "
                "to populate the available phenotype labels."
            )

        exclude_phenotypes = st.multiselect(
            "Exclude these phenotypes",
            options=exclude_candidates,
            default=valid_defaults,
            key="mrml_exclude_phenotypes",
            help=(
                "Selected labels are excluded case-insensitively from manual-reference "
                "training and validation targets and are ignored as automated-method votes. "
                "Eligible cells are still assigned one of the remaining trained classes. "
                "The separate refined output may still use the configured low-confidence label."
            ),
        )

        st.markdown("**Feature Configuration**")
        morph_cols = st.text_input(
            "Morphology Columns (comma-separated)",
            value=", ".join(st.session_state["mrml_config"]["morphology_cols"]),
            key="mrml_morph_cols",
            help=(
                "Optional numeric morphology features appended to the marker features. Missing column names are ignored. With z-normalization enabled, supported morphology features are normalized using the same batch column."
            ),
        )
        comp_cols = st.text_input(
            "Compartment Columns (comma-separated)",
            value=", ".join(st.session_state["mrml_config"]["compartment_cols"]),
            key="mrml_comp_cols",
            help=(
                "Optional numeric compartment features appended to the model input. Missing column names are ignored. These features are weighted but otherwise used as provided."
            ),
        )

        w1, w2, w3 = st.columns(3)
        feature_weights = st.session_state["mrml_config"]["feature_weights"]
        w_markers = w1.number_input(
            "Markers Weight",
            value=float(feature_weights.get("markers", 1.0)),
            step=0.1,
            key="mrml_w_markers",
            help="Feature-block multiplier. Set to `0` to exclude markers. Positive scaling does not alter tree splits and is removed by the SVM's internal standardization, so use this mainly as an include/exclude control.",
        )
        w_morph = w2.number_input(
            "Morphology Weight",
            value=float(feature_weights.get("morphology", 0.0)),
            step=0.1,
            key="mrml_w_morph",
            help="Feature-block multiplier. Set to `0` to exclude morphology; positive values should not be interpreted as learned importance.",
        )
        w_comp = w3.number_input(
            "Compartments Weight",
            value=float(feature_weights.get("compartments", 0.0)),
            step=0.1,
            key="mrml_w_comp",
            help="Feature-block multiplier. Set to `0` to exclude compartments; positive values should not be interpreted as learned importance.",
        )

        st.markdown("**Model Parameters**")
        t1, t2, t3 = st.columns(3)
        test_size = t1.number_input(
            "TEST_SIZE",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state["mrml_config"]["test_size"]),
            step=0.05,
            help=(
                "Used only when `Internal stratified cell split` is selected. It is ignored when a held-out validation-reference column is selected."
            ),
            key="mrml_test_size",
            disabled=validation_ground_truth_col is not None,
        )
        n_estimators = t2.number_input(
            "N_ESTIMATORS",
            min_value=100,
            max_value=5000,
            value=int(st.session_state["mrml_config"]["n_estimators"]),
            step=100,
            help=(
                "Number of trees for Random Forest. The same value is passed to Gradient Boosting, but that backend caps it at `500` internally. This setting is ignored by the SVM option."
            ),
            key="mrml_n_estimators",
        )
        random_state = t3.number_input(
            "RANDOM_STATE",
            min_value=0,
            max_value=9999,
            value=int(st.session_state["mrml_config"]["random_state"]),
            step=1,
            help=get_random_seed_help_text("classifier initialization and, when used, the internal stratified split"),
            key="mrml_random_state",
        )

        _ml_methods = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "svm": "SVM (RBF Kernel)"}
        _ml_keys = list(_ml_methods.keys())
        _ml_labels = list(_ml_methods.values())
        _current_method = st.session_state["mrml_config"].get("ml_method", "random_forest")
        _ml_idx = _ml_keys.index(_current_method) if _current_method in _ml_keys else 0
        ml_method_label = st.selectbox(
            "ML Classifier",
            options=_ml_labels,
            index=_ml_idx,
            help=(
                "Random Forest is the safest default and uses class-balanced trees. Gradient Boosting can be sharper but is slower and capped at `500` trees in this code. "
                "SVM uses internally standardized features, an RBF kernel, balanced classes, and probability estimates; it is typically slower on large datasets."
            ),
            key="mrml_ml_method",
        )
        ml_method = _ml_keys[_ml_labels.index(ml_method_label)]

        st.caption(
            "Click the button below to persist every setting in this configuration "
            "section to the active project JSON."
        )
        save_cfg = st.form_submit_button(
            "Save All Configuration to Project", type="primary"
        )
        if save_cfg:
            st.session_state["mrml_config"].update(
                {
                    "use_global": bool(use_global),
                    "input_dir": str(
                        st.session_state.get(
                            "mrml_input_dir",
                            st.session_state["mrml_config"].get("input_dir", ""),
                        )
                    ),
                    "input_file": str(
                        st.session_state.get(
                            "mrml_input_file",
                            st.session_state["mrml_config"].get("input_file", ""),
                        )
                    ),
                    "input_path": str(full_input_path or ""),
                    "markers": list(confirmed_markers),
                    "marker_panel_name": str(
                        st.session_state.get("mrml_marker_panel_name", "")
                    ),
                    "marker_resolution": dict(
                        st.session_state.get("mrml_marker_resolution", {}) or {}
                    ),
                    "ground_truth_col": ground_truth_col,
                    "validation_ground_truth_col": validation_ground_truth_col,
                    "output_phenotype_col": output_phenotype_col,
                    "confidence_col": confidence_col,
                    "refined_output_phenotype_col": refined_output_phenotype_col,
                    "refined_confidence_threshold": float(refined_threshold),
                    "refined_low_confidence_label": refined_label,
                    "write_backup_csv": bool(write_backup_csv),
                    "output_dir": output_dir,
                    "use_z_normalization": use_z_norm,
                    "batch_normalization_col": batch_norm_col,
                    "cell_id_col": normalize_optional_column_name(cell_id_col_cfg),
                    "enable_filter": enable_filter,
                    "filter_col": filter_col,
                    "filter_val": filter_val,
                    "phenotyping_methods": phenotyping_methods,
                    "exclude_phenotypes": exclude_phenotypes,
                    "morphology_cols": [c.strip() for c in morph_cols.split(",") if c.strip()],
                    "compartment_cols": [c.strip() for c in comp_cols.split(",") if c.strip()],
                    "feature_weights": {"markers": w_markers, "morphology": w_morph, "compartments": w_comp},
                    "test_size": test_size,
                    "n_estimators": int(n_estimators),
                    "random_state": int(random_state),
                    "ml_method": ml_method,
                }
            )
            try:
                cached_params.update(st.session_state["mrml_config"])
                with open(mr_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
                from spacemap_gui.persistence import save_active_project_state
                save_active_project_state("tab7_phenotyping")
            except Exception as ex:
                logger.warning(f"Failed to auto-save configuration or project state: {ex}")
            st.success(
                "All ML-MR configuration settings were saved to the project JSON."
            )

    st.divider()
    st.markdown("### 4. Run MR ML Phenotyping")

    if st.button("Run MR ML Pipeline", type="primary", width="stretch", key="mrml_run"):
        print(f"\n{'='*80}", flush=True)
        print(f"[{_timestamp()}] MR-ML: Run button clicked", flush=True)
        print(f"[{_timestamp()}] MR-ML: Input file: {full_input_path}", flush=True)
        print(f"[{_timestamp()}] MR-ML: Markers count: {len(confirmed_markers) if confirmed_markers else 0}", flush=True)

        if not full_input_path or not os.path.exists(full_input_path):
            print(f"[{_timestamp()}] MR-ML: ERROR - No valid unified CSV file selected", flush=True)
            st.error("Please select a valid unified CSV file.")
            return
        if not confirmed_markers:
            print(f"[{_timestamp()}] MR-ML: ERROR - No markers selected", flush=True)
            st.error("No markers selected. Please save marker selection first.")
            return
        if run_ml_mr_analysis is None:
            print(f"[{_timestamp()}] MR-ML: ERROR - ml_mr_utils module not loaded", flush=True)
            st.error("ML-MR utilities are not available. Please check ml_mr_utils.")
            return

        cfg = st.session_state["mrml_config"]
        if not cfg.get("phenotyping_methods"):
            st.error("Select at least one automated phenotyping method to integrate.")
            return
        print(f"[{_timestamp()}] MR-ML: ML method: {cfg.get('ml_method', 'random_forest')}", flush=True)
        print(f"[{_timestamp()}] MR-ML: Training reference: {cfg.get('ground_truth_col', 'N/A')}", flush=True)
        print(f"[{_timestamp()}] MR-ML: Validation reference: {cfg.get('validation_ground_truth_col') or 'internal split'}", flush=True)
        print(f"[{_timestamp()}] MR-ML: Phenotyping methods: {cfg.get('phenotyping_methods', [])}", flush=True)
        print(
            f"[{_timestamp()}] MR-ML: Excluded phenotypes: "
            f"{cfg.get('exclude_phenotypes', [])}",
            flush=True,
        )
        out_dir = project_root / "outputs/phenotyping/ML_MR"
        output_phenotype_col = cfg.get("output_phenotype_col", "phenotype_ml_mr")
        confidence_col = cfg.get("confidence_col", "ml_mr_confidence")
        refined_col = cfg.get(
            "refined_output_phenotype_col", "phenotype_ml_mr_refined"
        )
        refined_label = str(cfg.get("refined_low_confidence_label", "Others")).strip()
        configured_output_names = [output_phenotype_col, confidence_col, refined_col]
        if any(not str(column).strip() for column in configured_output_names):
            st.error("Primary phenotype, confidence, and refined output column names cannot be blank.")
            return
        if len(set(configured_output_names)) != len(configured_output_names):
            st.error("Primary phenotype, confidence, and refined output columns must be different.")
            return
        if not refined_label:
            st.error("The low-confidence phenotype label cannot be blank.")
            return
        run_audit = begin_phenotyping_run_audit(
            Path(project_root),
            "mrml",
            input_csv=full_input_path,
            markers=confirmed_markers,
        )
        audit_input_rows = None

        filter_enabled = bool(cfg.get("enable_filter", False))
        filter_col = str(cfg.get("filter_col", "tissue"))
        filter_val = str(cfg.get("filter_val", "1"))
        cell_id_col = normalize_optional_column_name(cfg.get("cell_id_col"))

        # --- Load data once, filter in memory, pass directly to engine ---
        preloaded_df = None
        filtered_mode = False

        print(f"[{_timestamp()}] MR-ML: Loading dataset...", flush=True)
        try:
            full_df = pd.read_csv(full_input_path)
            audit_input_rows = len(full_df)
            full_df, cell_id_col, _ = ensure_cell_id_anchor(
                full_df, cell_id_col
            )
            print(f"[{_timestamp()}] MR-ML: Loaded {len(full_df):,} cells, {len(full_df.columns)} columns", flush=True)
        except Exception as exc:
            finish_phenotyping_run_audit(
                run_audit,
                status="failed",
                output_column=output_phenotype_col,
                cluster_column="ml_mr_cluster",
                error=str(exc),
            )
            print(f"[{_timestamp()}] MR-ML: ERROR loading CSV: {exc}", flush=True)
            st.error(f"Failed to load CSV: {exc}")
            return

        training_reference = str(cfg.get("ground_truth_col") or "")
        validation_reference = cfg.get("validation_ground_truth_col") or None
        excluded_phenotypes = [
            str(value).strip()
            for value in cfg.get("exclude_phenotypes", [])
            if str(value).strip()
        ]
        excluded_normalized = {
            phenotype.lower() for phenotype in excluded_phenotypes
        }
        required_reference_columns = [training_reference] + ([validation_reference] if validation_reference else [])
        missing_reference_columns = [
            column for column in required_reference_columns if column not in full_df.columns
        ]
        if missing_reference_columns:
            finish_phenotyping_run_audit(
                run_audit,
                status="aborted",
                input_rows=audit_input_rows,
                output_column=output_phenotype_col,
                cluster_column="ml_mr_cluster",
                error=f"Missing reference columns: {missing_reference_columns}",
            )
            st.error(f"Missing reference columns: {missing_reference_columns}")
            return

        def _usable_reference_mask_frame(frame: pd.DataFrame, column: str) -> pd.Series:
            values = frame[column].astype("string").str.strip()
            return frame[column].notna() & ~values.str.lower().isin(
                {"", "unclassified", "unknown", "nan", "none"}
                | excluded_normalized
            )

        def _usable_reference_mask(column: str) -> pd.Series:
            return _usable_reference_mask_frame(full_df, column)

        train_reference_mask = _usable_reference_mask(training_reference)
        if not train_reference_mask.any():
            finish_phenotyping_run_audit(
                run_audit,
                status="aborted",
                input_rows=audit_input_rows,
                output_column=output_phenotype_col,
                cluster_column="ml_mr_cluster",
                error=f"No usable labels in {training_reference}",
            )
            st.error(f"No usable labels were found in `{training_reference}`.")
            return
        if validation_reference:
            validation_reference_mask = _usable_reference_mask(validation_reference)
            if not validation_reference_mask.any():
                finish_phenotyping_run_audit(
                    run_audit,
                    status="aborted",
                    input_rows=audit_input_rows,
                    output_column=output_phenotype_col,
                    cluster_column="ml_mr_cluster",
                    error=f"No usable labels in {validation_reference}",
                )
                st.error(f"No usable labels were found in `{validation_reference}`.")
                return
            overlap_count = int((train_reference_mask & validation_reference_mask).sum())
            if overlap_count and drop_overlapping_validation:
                # Removing the shared cells from scoring is enough to keep the
                # measurement honest, and it costs only those rows. Refusing to
                # run at all blocks otherwise valid pairs over a handful of
                # cells, which is a common result of annotating overlapping
                # regions at different times.
                validation_reference_mask = validation_reference_mask & ~train_reference_mask
                remaining = int(validation_reference_mask.sum())
                if remaining == 0:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_phenotype_col,
                        cluster_column="ml_mr_cluster",
                        error="every validation cell is also a training cell",
                    )
                    st.error(
                        f"Every cell in `{validation_reference}` is also labelled in "
                        f"`{training_reference}`, so nothing is left to score against. "
                        "Choose a validation column covering different cells."
                    )
                    return
                st.warning(
                    f"{overlap_count:,} cell(s) are labelled in both `{training_reference}` and "
                    f"`{validation_reference}` and were removed from scoring. "
                    f"{remaining:,} independent cells remain."
                )
            elif overlap_count:
                finish_phenotyping_run_audit(
                    run_audit,
                    status="aborted",
                    input_rows=audit_input_rows,
                    output_column=output_phenotype_col,
                    cluster_column="ml_mr_cluster",
                    error=f"{overlap_count} rows overlap between training and validation references",
                )
                st.error(
                    f"{overlap_count:,} cell(s) are labelled in both `{training_reference}` and "
                    f"`{validation_reference}`. Scoring on them would grade the model on cells it "
                    "trained on. Tick 'Drop cells labelled in both columns' to remove them, or "
                    "choose a validation column covering different cells."
                )
                return
            st.info(
                f"Scoring on unseen cells: {int(train_reference_mask.sum()):,} training labels in "
                f"`{training_reference}` and {int(validation_reference_mask.sum()):,} independent "
                f"labels in `{validation_reference}`."
            )
        if excluded_phenotypes:
            st.info(
                "Excluded from ML-MR training, validation, and method votes: "
                + ", ".join(excluded_phenotypes)
            )

        if filter_enabled:
            print(f"[{_timestamp()}] MR-ML: Applying filter {filter_col} == {filter_val}...", flush=True)
            if filter_col in full_df.columns:
                filtered_df = full_df[full_df[filter_col].astype(str) == filter_val].copy()
                filtered_df.reset_index(drop=True, inplace=True)
                preloaded_df = filtered_df
                filtered_mode = True
                print(f"[{_timestamp()}] MR-ML: Filter applied: {len(filtered_df):,} cells match", flush=True)
                st.info(f"Filtering enabled: {filter_col} == {filter_val} | {len(filtered_df):,} cells")
            else:
                print(f"[{_timestamp()}] MR-ML: Filter column '{filter_col}' not found, running on full dataset", flush=True)
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
                if cell_id_col not in preloaded_df.columns and cell_id_col in full_df.columns:
                    preloaded_df = preloaded_df.copy()
                    preloaded_df[cell_id_col] = full_df.loc[preloaded_df.index, cell_id_col].to_numpy()
                preloaded_df = preloaded_df[valid_mask].copy()

                st.info(f"Filtered out {orig_len_nan - len(preloaded_df):,} cells containing NaN/Inf in selected markers. Remaining: {len(preloaded_df):,} cells.")
                if len(preloaded_df) == 0:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_phenotype_col,
                        cluster_column="ml_mr_cluster",
                        error="No cells remained after marker-value filtering.",
                    )
                    st.error("No cells remain after filtering out NaN/Inf values in selected markers.")
                    return

        with st.status("Running MR ML Pipeline...", expanded=True) as status:
            try:
                status.write(f"[{_timestamp()}] Starting ML-MR analysis...")
                _log_console(f"[{_timestamp()}] MR-ML: Starting analysis")
                _log_console(f"[{_timestamp()}] MR-ML: Output dir: {out_dir}")
                result_holder: dict[str, pd.DataFrame] = {}

                def _capture_results(result_frame: pd.DataFrame) -> None:
                    result_holder["dataframe"] = result_frame

                # Blank the validation label on cells that also carry a training
                # label. The analysis receives only column names, so clearing the
                # value here is what actually keeps those cells out of scoring;
                # their training label and their features are untouched.
                if (
                    cfg.get("validation_ground_truth_col")
                    and cfg.get("drop_overlapping_validation", True)
                    and preloaded_df is not None
                ):
                    v_col = cfg["validation_ground_truth_col"]
                    t_col = cfg["ground_truth_col"]
                    if v_col in preloaded_df.columns and t_col in preloaded_df.columns:
                        shared = (
                            _usable_reference_mask_frame(preloaded_df, t_col)
                            & _usable_reference_mask_frame(preloaded_df, v_col)
                        )
                        if bool(shared.any()):
                            preloaded_df = preloaded_df.copy()
                            preloaded_df.loc[shared, v_col] = np.nan
                            status.write(
                                f"[{_timestamp()}] Removed {int(shared.sum()):,} overlapping "
                                f"cell(s) from the scoring set"
                            )

                ok = run_ml_mr_analysis(
                    unified_csv_path=Path(full_input_path),
                    performance_csv_path=None,
                    output_dir=out_dir if out_dir else Path(project_root / "outputs/phenotyping/ML_MR"),
                    ground_truth_column=cfg["ground_truth_col"],
                    phenotyping_methods=cfg["phenotyping_methods"],
                    clustering_markers=confirmed_markers,
                    morphology_cols=cfg["morphology_cols"],
                    compartment_cols=cfg["compartment_cols"],
                    feature_weights=cfg["feature_weights"],
                    use_z_normalization=cfg["use_z_normalization"],
                    batch_normalization_col=cfg["batch_normalization_col"],
                    test_size=cfg["test_size"],
                    n_estimators=cfg["n_estimators"],
                    random_state=cfg["random_state"],
                    ml_method=cfg.get("ml_method", "random_forest"),
                    output_phenotype_col=output_phenotype_col,
                    confidence_col_name=confidence_col,
                    refined_output_phenotype_col=refined_col,
                    refined_confidence_threshold=float(
                        cfg.get("refined_confidence_threshold", 0.75)
                    ),
                    refined_low_confidence_label=refined_label,
                    excluded_phenotypes=excluded_phenotypes,
                    progress_callback=status.write,
                    preloaded_df=preloaded_df,
                    validation_ground_truth_column=cfg.get("validation_ground_truth_col") or None,
                    write_backup_csv=bool(cfg.get("write_backup_csv", False)),
                    update_source_csv=False,
                    results_callback=_capture_results,
                )

                if not ok:
                    status.error(f"[{_timestamp()}] MR-ML failed; review the status messages above.")
                    return

                status.write(f"[{_timestamp()}] ML-MR analysis complete")
                _log_console(f"[{_timestamp()}] MR-ML: Analysis complete")
                status.write(f"[{_timestamp()}] Loading results and building phenotype mappings...")

                df_results = result_holder.get("dataframe")
                if df_results is None:
                    status.warning(
                        f"[{_timestamp()}] ML-MR did not return reviewable assignments."
                    )
                    return
                phenotype_col = output_phenotype_col

                required_result_columns = [phenotype_col, confidence_col, refined_col]
                missing_result_columns = [
                    column for column in required_result_columns if column not in df_results.columns
                ]
                if missing_result_columns:
                    status.warning(
                        f"[{_timestamp()}] Missing MR-ML result columns: {missing_result_columns}"
                    )
                    return

                unique_phenos = sorted(df_results[phenotype_col].dropna().unique().tolist())
                pheno_to_cluster = {p: i for i, p in enumerate(unique_phenos)}
                df_results["ml_mr_cluster"] = df_results[phenotype_col].map(pheno_to_cluster).fillna(-1)

                temp_file = project_root / "outputs/phenotyping/cache" / f"mr_ml_results_{int(time.time())}.parquet"
                df_results.to_parquet(temp_file)
                status.write(f"[{_timestamp()}] Saving cached results for interactive review...")

                st.session_state["mrml_temp_file"] = str(temp_file)
                st.session_state["mrml_phenotype_map"] = {int(c): p for p, c in pheno_to_cluster.items()}
                st.session_state["mrml_marker_cols"] = confirmed_markers
                st.session_state["mrml_outcol"] = phenotype_col
                st.session_state["mrml_conf_col"] = confidence_col
                st.session_state["mrml_refined_col"] = refined_col
                st.session_state["mrml_cluster_col"] = "ml_mr_cluster"
                # ``mrml_cell_id_col`` belongs to the Cell ID text widget.
                # Runtime result-linking metadata must use a separate key;
                # Streamlit rejects writes to a widget key after creation.
                st.session_state["mrml_result_cell_id_col"] = cell_id_col
                st.session_state["mrml_input_path"] = str(full_input_path)

                status.update(label="MR ML Complete!", state="complete")
                st.success(
                    "MR ML phenotyping complete. Review the assignments below; "
                    "the unified CSV has not been changed."
                )
                finish_phenotyping_run_audit(
                    run_audit,
                    status="success",
                    output_path=temp_file,
                    input_rows=audit_input_rows,
                    output_rows=len(df_results),
                    output_column=phenotype_col,
                    cluster_column="ml_mr_cluster",
                    phenotype_map=st.session_state["mrml_phenotype_map"],
                    extra={
                        "confidence_column": confidence_col,
                        "refined_column": refined_col,
                        "refined_confidence_threshold": float(
                            cfg.get("refined_confidence_threshold", 0.75)
                        ),
                        "refined_low_confidence_label": refined_label,
                        "training_reference_column": cfg.get("ground_truth_col"),
                        "validation_reference_column": cfg.get("validation_ground_truth_col"),
                        "excluded_phenotypes": excluded_phenotypes,
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
                    cluster_column="ml_mr_cluster",
                    error=str(e),
                )
                run_audit = None
                status.error(f"[{_timestamp()}] Pipeline error: {e}")
                logger.error(f"MR ML pipeline error: {e}")
            finally:
                if run_audit is not None:
                    finish_phenotyping_run_audit(
                        run_audit,
                        status="aborted",
                        input_rows=audit_input_rows,
                        output_column=output_phenotype_col,
                        cluster_column="ml_mr_cluster",
                        error="Pipeline stopped before completion; see the MR ML status messages.",
                    )

    if "mrml_temp_file" in st.session_state and Path(st.session_state["mrml_temp_file"]).exists():
        st.divider()
        st.subheader("Results & Phenotype Refinement")
        st.caption("Interactive visualization and export of MR ML results")

        temp_file = st.session_state["mrml_temp_file"]
        df_results = pd.read_parquet(temp_file)

        # Recover metadata for a review cache that was written before an older
        # UI version hit the widget/session-state collision fixed above.
        if "mrml_result_cell_id_col" not in st.session_state:
            configured_cell_id = normalize_optional_column_name(
                st.session_state.get("mrml_config", {}).get("cell_id_col")
            )
            if configured_cell_id and configured_cell_id in df_results.columns:
                result_cell_id_col = configured_cell_id
            elif TAB7_INTERNAL_CELL_ID_COL in df_results.columns:
                result_cell_id_col = TAB7_INTERNAL_CELL_ID_COL
            else:
                result_cell_id_col = configured_cell_id or "cell_id"
            st.session_state["mrml_result_cell_id_col"] = result_cell_id_col
        if "mrml_input_path" not in st.session_state and full_input_path:
            st.session_state["mrml_input_path"] = str(full_input_path)
        marker_cols = st.session_state.get("mrml_marker_cols", [])
        phenotype_map = st.session_state.get("mrml_phenotype_map", {})
        output_col = st.session_state.get("mrml_outcol", "phenotype_ml_mr")
        confidence_col = st.session_state.get("mrml_conf_col", "ml_mr_confidence")
        refined_col = st.session_state.get(
            "mrml_refined_col", "phenotype_ml_mr_refined"
        )
        cluster_col = st.session_state.get("mrml_cluster_col", "ml_mr_cluster")

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

        if refined_col in df_results.columns:
            threshold = float(
                st.session_state.get("mrml_config", {}).get(
                    "refined_confidence_threshold", 0.75
                )
            )
            low_confidence_count = (
                int((df_results[confidence_col] < threshold).sum())
                if confidence_col in df_results.columns
                else int((df_results[refined_col] != df_results[output_col]).sum())
            )
            st.markdown(f"#### Refined Phenotype Summary — `{refined_col}`")
            st.caption(
                f"{low_confidence_count:,} cells had ensemble support below {threshold:g} "
                "and were relabeled only in the refined output."
            )
            refined_summary = (
                df_results[refined_col]
                .fillna("Unassigned")
                .value_counts()
                .rename_axis("Refined Phenotype")
                .reset_index(name="Cell Count")
            )
            refined_summary["% Total"] = (
                refined_summary["Cell Count"] / len(df_results) * 100
            ).map(lambda value: f"{value:.1f}%")
            st.dataframe(refined_summary, width="stretch", hide_index=True)

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
                st.session_state["mrml_heatmap_z"] = heatmap_data

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
                    key_prefix="mrml",
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
            key_prefix="mrml",
            cluster_col=cluster_col,
            update_cluster_cols=[cluster_col, output_col, confidence_col, refined_col],
            cell_id_col=st.session_state.get("mrml_result_cell_id_col", "cell_id"),
            input_path_override=st.session_state.get("mrml_input_path"),
        )

        render_export_split_data(
            df_results=df_results,
            output_col=output_col,
            phenotype_map=phenotype_map,
            project_root=project_root,
            key_prefix="mrml",
            cluster_col=cluster_col,
            cell_id_col=st.session_state.get("mrml_result_cell_id_col", "cell_id"),
        )

    # --- VISUAL VALIDATION COMPONENT ---
    if st.session_state.get('viz_enabled', True):
        st.divider()
        with st.expander("Visual Validation of Phenotypes", expanded=True):
            st.markdown("#### Visualize Representative Cells")
            st.caption("Inspect random cells from specific phenotypes to validate assignments.")
            
            # Ensure we have data
            if 'mrml_temp_file' in st.session_state and Path(st.session_state['mrml_temp_file']).exists():
                # Load data if not already
                if 'df_results' not in locals():
                    df_results = pd.read_parquet(st.session_state['mrml_temp_file'])
                
                # Check Global Viz Config
                viz_roi = st.session_state.get('viz_roi_dir')
                viz_seg = st.session_state.get('viz_seg_dir')
                
                if not viz_roi or not viz_seg or not os.path.exists(viz_roi) or not os.path.exists(viz_seg):
                    st.warning("Please configure valid ROI and Segmentation directories in Global Input Configuration (Top of Tab 7).")
                else:
                    viz = CellVisualizer(viz_roi, viz_seg)
                    
                    # File matching (cache in session state)
                    if not st.session_state.get('mrml_viz_file_matches'):
                         # Get unique batches
                         batch_col = "batch_id"
                         if batch_col not in df_results.columns:
                             batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in df_results.columns), 'batch_id')
                         
                         if batch_col in df_results.columns:
                             unique_batches = df_results[batch_col].unique().astype(str).tolist()
                             matches = viz.match_file_triads(unique_batches)
                             st.session_state.mrml_viz_file_matches = matches
                             if not matches:
                                 st.error("No matching image files found for these batches.")
                         else:
                             st.error(f"Batch column '{batch_col}' not found in results.")
                    else:
                        viz.file_map = st.session_state.mrml_viz_file_matches

                    # UI Controls
                    output_col = st.session_state.get("mrml_outcol", "phenotype_ml_mr")
                    
                    available_phenos = []
                    if output_col in df_results.columns:
                        available_phenos = sorted([p for p in df_results[output_col].unique() if pd.notna(p) and str(p) != 'Unassigned'])
                    
                    with st.form("mrml_viz_form"):
                         vv_col1, vv_col2, vv_col3, vv_col4 = st.columns([2, 1, 1, 2])
                         with vv_col1:
                             selected_pheno_vv = st.selectbox(
                                 "Select Phenotype", 
                                 available_phenos,
                                 key="mrml_viz_pheno_sel"
                             )
                         with vv_col2:
                             n_viz_cells = st.number_input("Cells to View", 1, 50, 5, key="mrml_viz_n_cells")
                         with vv_col3:
                             show_neighbors = st.checkbox("Show Neighbors", value=True, key="mrml_viz_neighbors")
                         with vv_col4:
                             mrml_batch_scope = st.radio(
                                 "Batch Scope",
                                 ["All matched batches", "Specific batch"],
                                 horizontal=True,
                                 key="mrml_viz_batch_scope",
                             )
                             mrml_batch_options = sorted(str(batch) for batch in st.session_state.mrml_viz_file_matches.keys())
                             selected_batch_vv = None
                             if mrml_batch_scope == "Specific batch":
                                 selected_batch_vv = st.selectbox(
                                     "Batch",
                                     mrml_batch_options,
                                     key="mrml_viz_batch_sel",
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
                            heatmap_z = st.session_state.get("mrml_heatmap_z")
                            phenotype_map = st.session_state.get("mrml_phenotype_map", {})
                            
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
                                analysis_markers = st.session_state.get('mrml_marker_cols', [])
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
                                elif not st.session_state.get('mrml_viz_file_matches'):
                                    st.error("No matching image files found for visualization batches.")
                                    subset_valid = pd.DataFrame()
                                else:
                                    valid_batches = list(st.session_state.mrml_viz_file_matches.keys())
                                    subset_valid = subset[subset[batch_col].astype(str).isin(valid_batches)]
                                    if mrml_batch_scope == "Specific batch" and selected_batch_vv:
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
                                                    'method': 'mr_ml',
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
