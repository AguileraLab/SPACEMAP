"""Compact Streamlit controls for strict grouped phenotyping validation."""

from __future__ import annotations

import os
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from spacemap_gui.ui_utils import clamp_numeric_widget_state

from spacemap_gui.tabs.tab7_phenotyping.common import (
    get_marker_list,
    get_named_marker_panels,
)

from .f1_validation_utils import (
    EXCLUDED_GROUP_TOKEN,
    ROI_SUFFIX_PATTERN,
    clean_label_series,
    parse_mapping_text,
    prettify_column_name,
)


GROUP_MODES = ("As stored", "Remove ROI suffix", "Manual mapping")
MODEL_OPTIONS = {
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "SVM (RBF kernel)": "svm",
}
MORPHOLOGY_NAMES = {
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "orientation",
    "major_axis",
    "minor_axis",
    "major_axis_length",
    "minor_axis_length",
}
BLOCKED_FEATURE_EXACT = {
    "cell_id",
    "batch_id",
    "tissue",
    "tissue_roi",
    "roi",
    "roi_id",
    "patient",
    "patient_id",
    "slide",
    "slide_id",
    "sample",
    "sample_id",
    "timepoint",
    "x_centroid",
    "y_centroid",
    "label",
    "index",
    "corrected",
    "consensus",
}
BLOCKED_FEATURE_PREFIXES = (
    "phenotype_",
    "prediction_",
    "confidence_",
    "probability_",
    "ml_",
    "umap_",
    "cluster_",
    "leiden_",
    "source_",
)


def _cached(
    cached_params: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in st.session_state:
        return st.session_state[key]
    return cached_params.get(key, default)


@st.cache_data(show_spinner=False)
def _read_schema_sample(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path, nrows=64, low_memory=False)


def _sample_frame(path: str | None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return _read_schema_sample(str(path), os.path.getmtime(path))
    except Exception:
        return pd.DataFrame()


def _numeric_columns(
    columns: list[str],
    sample: pd.DataFrame,
) -> list[str]:
    if sample.empty:
        return list(columns)
    return [
        column
        for column in columns
        if column in sample.columns and pd.api.types.is_numeric_dtype(sample[column])
    ]


def _feature_candidates(
    columns: list[str],
    sample: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    numeric = _numeric_columns(columns, sample)
    morphology = [
        column for column in numeric if str(column).strip().lower() in MORPHOLOGY_NAMES
    ]
    compartments = [
        column
        for column in numeric
        if any(
            token in str(column).lower()
            for token in ("compartment", "nuclear", "cytoplasm", "membrane")
        )
        and column not in morphology
    ]

    markers: list[str] = []
    for column in numeric:
        lower = str(column).strip().lower()
        if lower in BLOCKED_FEATURE_EXACT:
            continue
        if lower.startswith(BLOCKED_FEATURE_PREFIXES):
            continue
        if column in morphology or column in compartments:
            continue
        if lower.endswith(("_match_distance", "_count", "_id")):
            continue
        markers.append(str(column))
    return markers, morphology, compartments


def _strict_component_candidates(method_options: list[str]) -> list[str]:
    result = []
    for column in method_options:
        lower = str(column).lower()
        if any(
            token in lower
            for token in (
                "ml_mr",
                "ml_consensus",
                "mr_ml_strict",
                "consensus_ml_strict",
                "consensus_phenotype",
            )
        ):
            continue
        result.append(str(column))
    return result


def _phenotyping_marker_candidates(
    columns: list[str],
    sample: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Use Tab 7's common marker list, then retain training-safe numerics."""
    safe_markers, morphology, compartments = _feature_candidates(columns, sample)
    phenotyping_frame = sample if not sample.empty else pd.DataFrame(columns=columns)
    common_markers = [
        str(column)
        for column in get_marker_list(phenotyping_frame)
        if str(column) in columns
    ]
    safe_marker_set = set(safe_markers)
    markers = [marker for marker in common_markers if marker in safe_marker_set]
    return markers or safe_markers, morphology, compartments


def _default_components(options: list[str]) -> list[str]:
    preferred = [
        "phenotype_leiden",
        "phenotype_som",
        "phenotype_vortex",
        "phenotype_threshold",
    ]
    selected = [column for column in preferred if column in options]
    return selected or options[:4]


def _default_group(group_options: list[str]) -> str:
    for preferred in ("patient_id", "slide_id", "validation_group", "batch_id"):
        if preferred in group_options:
            return preferred
    return group_options[0] if group_options else ""


def _default_reference(reference_options: list[str]) -> str:
    for preferred in ("phenotype_ref1", "phenotype_ref1a", "ground_truth"):
        if preferred in reference_options:
            return preferred
    return reference_options[0]


def _safe_choice(options: list[str], value: Any, fallback: str) -> str:
    if value in options:
        return str(value)
    if fallback in options:
        return fallback
    return options[0] if options else ""


def _safe_list(
    options: list[str],
    value: Any,
    fallback: list[str],
) -> list[str]:
    values = value if isinstance(value, list) else fallback
    selected = [str(item) for item in values if str(item) in options]
    return selected or [item for item in fallback if item in options]


def _marker_panel_controls(
    unified_csv_path: str | None,
    columns: list[str],
    cached_params: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    sample = _sample_frame(unified_csv_path)
    marker_options, morphology_options, compartment_options = (
        _phenotyping_marker_candidates(
            columns,
            sample,
        )
    )
    corrected_defaults = [
        marker for marker in marker_options if marker.lower().endswith("_corr")
    ]
    manual_defaults = corrected_defaults or marker_options

    try:
        panels, workspace = get_named_marker_panels(
            marker_options,
            unified_csv_path,
        )
        st.session_state["tab7_marker_workspace"] = workspace
    except Exception:
        panels = {}

    source_options = (
        ["Saved Step 7 marker panel", "Phenotyping common marker list"]
        if panels
        else ["Phenotyping common marker list"]
    )
    cached_source = _cached(
        cached_params,
        "viz_f1_strict_marker_source",
        source_options[0],
    )
    if cached_source == "Manual selection":
        cached_source = "Phenotyping common marker list"
    if cached_source not in source_options:
        cached_source = source_options[0]
    if st.session_state.get("viz_f1_strict_marker_source") != cached_source:
        st.session_state["viz_f1_strict_marker_source"] = cached_source
    marker_source = st.radio(
        "Marker source",
        options=source_options,
        index=source_options.index(
            _safe_choice(
                source_options,
                cached_source,
                source_options[0],
            )
        ),
        horizontal=True,
        key="viz_f1_strict_marker_source",
        help=(
            "The Phenotyping common list is generated by Tab 7's shared "
            "get_marker_list() definition. Saved Step 7 panels are loaded from "
            "that same marker workspace."
        ),
    )

    panel_name = "manual"
    selected_source_markers = manual_defaults
    if marker_source == "Saved Step 7 marker panel":
        panel_names = list(panels)
        preferred_panel = (
            "All corrected markers"
            if "All corrected markers" in panel_names
            else panel_names[0]
        )
        panel_name = st.selectbox(
            "Saved marker panel",
            options=panel_names,
            index=panel_names.index(
                _safe_choice(
                    panel_names,
                    _cached(
                        cached_params,
                        "viz_f1_strict_marker_panel",
                        preferred_panel,
                    ),
                    preferred_panel,
                )
            ),
            key="viz_f1_strict_marker_panel",
        )
        selected_source_markers = list(panels[panel_name])

    marker_key = "viz_f1_strict_markers"
    # The panel identity decides the selection; the CSV path does not. Changing
    # file must not wipe a marker list the user built, so the current selection
    # is carried over and simply filtered to markers the new file still has.
    source_token = f"{marker_source}|{panel_name}"
    previous_token = st.session_state.get("viz_f1_strict_marker_source_token")
    dropped: list[str] = []
    if previous_token != source_token:
        st.session_state[marker_key] = (
            list(selected_source_markers)
            if marker_source == "Saved Step 7 marker panel"
            else _safe_list(
                marker_options,
                cached_params.get(marker_key, []),
                selected_source_markers,
            )
        )
        st.session_state["viz_f1_strict_marker_source_token"] = source_token
    elif marker_key in st.session_state:
        current = [str(m) for m in st.session_state.get(marker_key) or []]
        kept = [m for m in current if m in marker_options]
        dropped = [m for m in current if m not in marker_options]
        if kept != current:
            st.session_state[marker_key] = kept or list(selected_source_markers)
    if marker_key not in st.session_state:
        st.session_state[marker_key] = _safe_list(
            marker_options,
            cached_params.get(marker_key, []),
            selected_source_markers,
        )

    markers = st.multiselect(
        "Training marker columns",
        options=marker_options,
        key=marker_key,
        help=(
            "Pick the expression markers the newly trained ML models may use as "
            "features. The choices come from Tab 7's common marker list and are "
            "limited to numeric, training-safe columns present in this CSV. Use "
            "the same raw/corrected representation used by your phenotyping workflow."
        ),
    )
    if dropped:
        st.warning(
            f"{len(dropped)} previously selected marker(s) are not in this CSV and were "
            f"dropped: {', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}. "
            "The rest of your selection was kept."
        )
    st.caption(
        "This selector imports Tab 7's shared marker list and named marker panels. "
        "Non-numeric entries and label/ID/coordinate columns are omitted because "
        "they cannot safely be ML training features. Your selection survives "
        "changing the input CSV when the same markers remain available."
    )
    return list(markers), morphology_options, compartment_options


@st.cache_data(show_spinner=False)
def _group_inventory(
    path: str,
    column: str,
    reference_columns: tuple[str, ...],
    modified_time: float,
) -> pd.DataFrame:
    """Cells and labelled reference cells for each value of the grouping column."""
    del modified_time
    wanted = [column] + [c for c in reference_columns if c and c != column]
    from spacemap import dataio

    present = set(dataio.columns(path))
    frame = dataio.load(path, columns=[c for c in wanted if c in present])
    if column not in frame.columns:
        return pd.DataFrame()
    key = frame[column].astype("string").str.strip()
    table = (
        key.value_counts(dropna=False).rename_axis("value").reset_index(name="cells")
    )
    for ref in reference_columns:
        if ref not in frame.columns:
            continue
        labelled = clean_label_series(frame[ref]).notna()
        counts = key.where(labelled).value_counts()
        table[prettify_column_name(ref)] = (
            table["value"].map(counts).fillna(0).astype(int)
        )
    return table


def _render_group_assignment_editor(
    *,
    unified_csv_path: str | None,
    group_column: str,
    group_mode: str,
    reference_options: list[str],
    cached_params: Mapping[str, Any],
) -> str:
    """Choose which images to use and which are held out together.

    Folds are the unit of validation: rows sharing a fold name are removed from
    training together. Two images from one patient must therefore share a name,
    or the model trains on that patient and is then tested on them again. The
    reference counts are shown because a fold with no manual labels cannot be
    scored, which is easy to create by accident and invisible otherwise.
    """
    st.markdown("**Which images to use, and which are held out together**")

    try:
        inventory = _group_inventory(
            str(unified_csv_path),
            group_column,
            tuple(reference_options),
            os.path.getmtime(str(unified_csv_path)),
        )
    except Exception:
        inventory = pd.DataFrame()

    if inventory.empty:
        st.caption(
            f"Could not read '{group_column}' from the selected CSV. "
            "Manual mapping still accepts `raw value = fold name` lines."
        )
        return str(_cached(cached_params, "viz_f1_strict_group_mapping", ""))

    stored = parse_mapping_text(
        str(_cached(cached_params, "viz_f1_strict_group_mapping", ""))
    )

    def auto_name(value: str) -> str:
        if group_mode == "Remove ROI suffix":
            return ROI_SUFFIX_PATTERN.sub("", str(value)).strip()
        return str(value)

    ref_cols = [c for c in inventory.columns if c not in ("value", "cells")]
    table = inventory.rename(columns={"value": group_column, "cells": "cells"})
    folds, use = [], []
    for raw in table[group_column]:
        assigned = stored.get(str(raw))
        if assigned is None:
            folds.append(auto_name(str(raw)))
            use.append(True)
        elif str(assigned).strip().lower() == EXCLUDED_GROUP_TOKEN:
            folds.append(auto_name(str(raw)))
            use.append(False)
        else:
            folds.append(str(assigned))
            use.append(True)
    table.insert(0, "use", use)
    table["fold"] = folds

    fill_cols = st.columns([1, 1, 2])
    if fill_cols[0].button(
        "Fill from ROI suffix",
        width="stretch",
        help="Set every fold to its image name with a trailing _roi_N removed.",
    ):
        table["fold"] = [
            ROI_SUFFIX_PATTERN.sub("", str(v)).strip() for v in table[group_column]
        ]
    if fill_cols[1].button(
        "One fold per image",
        width="stretch",
        help="Reset so every image is validated on its own.",
    ):
        table["fold"] = [str(v) for v in table[group_column]]

    edited = st.data_editor(
        table,
        key="viz_f1_strict_group_editor",
        width="stretch",
        hide_index=True,
        disabled=[group_column, "cells", *ref_cols],
        column_config={
            "use": st.column_config.CheckboxColumn(
                "use",
                help="Unticked images are left out of training, weighting and scoring.",
            ),
            group_column: st.column_config.TextColumn(group_column),
            "cells": st.column_config.NumberColumn("cells", format="%d"),
            "fold": st.column_config.TextColumn(
                "fold (held out together)",
                help="Type any name. Rows sharing a name form one validation round.",
            ),
            **{
                c: st.column_config.NumberColumn(f"{c} cells", format="%d")
                for c in ref_cols
            },
        },
    )

    active = edited[edited["use"].astype(bool)]
    if active.empty:
        st.error("Every image is unticked. Tick at least two folds' worth of images.")
        return "\n".join(
            f"{r[group_column]} = {EXCLUDED_GROUP_TOKEN}" for _, r in edited.iterrows()
        )

    agg = {"images": (group_column, "count"), "cells": ("cells", "sum")}
    agg.update({c: (c, "sum") for c in ref_cols})
    summary = active.groupby("fold").agg(**agg).sort_values("cells", ascending=False)
    st.caption("Resulting validation rounds")
    st.dataframe(summary, width="stretch")

    left, right = st.columns([1, 3])
    left.metric("Folds", len(summary))
    excluded = int((~edited["use"].astype(bool)).sum())
    if excluded:
        right.caption(f"{excluded} image(s) excluded entirely.")

    if len(summary) < 2:
        st.error("At least two folds are needed: one to hold out and one to train on.")
    if ref_cols:
        primary = ref_cols[0]
        empty = summary.index[summary[primary] == 0].tolist()
        if empty:
            st.warning(
                f"No {primary} cells in fold(s): {', '.join(map(str, empty))}. "
                "A fold with no reference cells cannot be scored when held out."
            )
        thin = summary.index[(summary[primary] > 0) & (summary[primary] < 50)].tolist()
        if thin:
            st.info(
                f"Very few {primary} cells in {', '.join(map(str, thin))} "
                "— those folds will give unstable scores."
            )
    if (summary["images"] == 1).all() and len(summary) == len(active):
        st.warning(
            "Every image is its own fold. If two images come from the same patient, "
            "the model will train on that patient and be tested on them again — give "
            "them the same fold name."
        )

    lines = []
    for _, row in edited.iterrows():
        target = str(row["fold"]).strip() if bool(row["use"]) else EXCLUDED_GROUP_TOKEN
        lines.append(f"{row[group_column]} = {target or EXCLUDED_GROUP_TOKEN}")
    return "\n".join(lines)


def render_strict_retraining_panel(
    *,
    unified_csv_path: str | None,
    available_columns: list[str],
    reference_options: list[str],
    method_options: list[str],
    group_options: list[str],
    cached_params: Mapping[str, Any],
    force_enabled: bool = False,
    title: str = "Re-train and validate per patient (recommended for publication)",
    expanded: bool | None = None,
) -> dict[str, Any]:
    """Render advanced controls and return a plain settings dictionary."""

    with st.expander(
        title,
        expanded=(
            bool(_cached(cached_params, "viz_f1_strict_enabled", False))
            if expanded is None
            else bool(expanded)
        ),
    ):
        if force_enabled:
            enabled = True
            st.info(
                "Publication mode always retrains the selected ML workflows. Existing "
                "saved ML columns are not used as held-out predictions."
            )
        else:
            enabled = st.checkbox(
                "Re-train the models, holding out one patient at a time",
                value=bool(
                    _cached(
                        cached_params,
                        "viz_f1_strict_enabled",
                        False,
                    )
                ),
                key="viz_f1_strict_enabled",
                help=(
                    "Strongest evidence, and what reviewers expect. Each patient is removed "
                    "from training entirely, then predicted by a model that never saw them. "
                    "Leave off to simply score prediction columns already in your CSV."
                ),
            )
        if not enabled:
            st.caption(
                "Leave this off for quick comparison of phenotype columns already "
                "saved in the selected CSV."
            )
            return {"enabled": False}

        if not group_options:
            error_message = (
                "Strict retraining needs a patient, slide, batch, or validation-group "
                "column in the selected CSV."
            )
            st.error(error_message)
            return {"enabled": True, "configuration_error": error_message}

        st.success(
            "Strict mode verifies group separation itself. The main analysis will "
            "use the generated out-of-fold predictions and will not modify the CSV."
        )
        tabs = st.tabs(
            [
                "1. Patients & references",
                "2. Features",
                "3. Model settings",
            ]
        )

        component_options = _strict_component_candidates(method_options)
        with tabs[0]:
            primary_default = _default_reference(reference_options)
            primary_reference = st.selectbox(
                "Train on these manual labels",
                options=reference_options,
                index=reference_options.index(
                    _safe_choice(
                        reference_options,
                        _cached(
                            cached_params,
                            "viz_f1_strict_primary_reference",
                            primary_default,
                        ),
                        primary_default,
                    )
                ),
                key="viz_f1_strict_primary_reference",
                help=(
                    "The column of hand-assigned phenotypes MR-ML learns from, and that sets "
                    "how much weight each method's vote carries. Within every fold only cells "
                    "outside the held-out patient are used, so the held-out patient never "
                    "contributes to fitting."
                ),
            )
            component_methods = st.multiselect(
                "Component phenotyping methods",
                options=component_options,
                default=_safe_list(
                    component_options,
                    _cached(
                        cached_params,
                        "viz_f1_strict_components",
                        _default_components(component_options),
                    ),
                    _default_components(component_options),
                ),
                key="viz_f1_strict_components",
                format_func=prettify_column_name,
                help=(
                    "These methods form the consensus vote and are compared with "
                    "MR-ML and Consensus-ML on identical held-out cells."
                ),
            )

            group_default = _default_group(group_options)
            group_cols = st.columns(2)
            with group_cols[0]:
                validation_group_column = st.selectbox(
                    "Column identifying the patient or slide",
                    options=group_options,
                    index=group_options.index(
                        _safe_choice(
                            group_options,
                            _cached(
                                cached_params,
                                "viz_f1_strict_group_column",
                                group_default,
                            ),
                            group_default,
                        )
                    ),
                    key="viz_f1_strict_group_column",
                    help="Choose the column defining independent biological units. Every row assigned to one fold is excluded from all training steps when that fold is tested.",
                )
            with group_cols[1]:
                group_mode = st.selectbox(
                    "How to turn that column into folds",
                    options=list(GROUP_MODES),
                    index=list(GROUP_MODES).index(
                        _safe_choice(
                            list(GROUP_MODES),
                            _cached(
                                cached_params,
                                "viz_f1_strict_group_mode",
                                "Remove ROI suffix",
                            ),
                            "As stored",
                        )
                    ),
                    key="viz_f1_strict_group_mode",
                    help="As stored uses values unchanged; Remove ROI suffix groups multiple ROIs from the same image/patient; Manual mapping lets you explicitly combine or exclude values.",
                )

            group_mapping_text = _render_group_assignment_editor(
                unified_csv_path=unified_csv_path,
                group_column=validation_group_column,
                group_mode=group_mode,
                reference_options=list(reference_options),
                cached_params=cached_params,
            )

            run_cols = st.columns(3)
            with run_cols[0]:
                run_grouped_mr = st.checkbox(
                    "Grouped MR-ML",
                    value=bool(
                        _cached(
                            cached_params,
                            "viz_f1_strict_run_mr",
                            True,
                        )
                    ),
                    key="viz_f1_strict_run_mr",
                    help="Retrain ML-MR from manual-reference cells in non-held-out folds and predict the complete hidden fold.",
                )
            with run_cols[1]:
                run_grouped_consensus = st.checkbox(
                    "Grouped Consensus-ML",
                    value=bool(
                        _cached(
                            cached_params,
                            "viz_f1_strict_run_consensus",
                            True,
                        )
                    ),
                    key="viz_f1_strict_run_consensus",
                    help="Recompute consensus outside the hidden fold, train Consensus-ML on those pseudo-labels, and predict non-consensus cells in the hidden fold.",
                )
            with run_cols[2]:
                run_internal = st.checkbox(
                    "Extra check: split the reference in two",
                    value=bool(
                        _cached(
                            cached_params,
                            "viz_f1_strict_run_internal",
                            False,
                        )
                    ),
                    key="viz_f1_strict_run_internal",
                    help=(
                        "Supporting row-disjoint analysis; it is not a replacement "
                        "for patient- or slide-grouped validation."
                    ),
                )

            development_reference = None
            internal_test_reference = None
            if run_internal:
                internal_cols = st.columns(2)
                default_development = (
                    "phenotype_ref1a"
                    if "phenotype_ref1a" in reference_options
                    else primary_reference
                )
                default_test = (
                    "phenotype_ref1b"
                    if "phenotype_ref1b" in reference_options
                    else reference_options[-1]
                )
                with internal_cols[0]:
                    development_reference = st.selectbox(
                        "Development reference",
                        options=reference_options,
                        index=reference_options.index(
                            _safe_choice(
                                reference_options,
                                _cached(
                                    cached_params,
                                    "viz_f1_strict_development_reference",
                                    default_development,
                                ),
                                default_development,
                            )
                        ),
                        key="viz_f1_strict_development_reference",
                        help="Manual-reference subset allowed to enter the optional row-disjoint development fit.",
                    )
                with internal_cols[1]:
                    internal_test_reference = st.selectbox(
                        "Held-out internal reference",
                        options=reference_options,
                        index=reference_options.index(
                            _safe_choice(
                                reference_options,
                                _cached(
                                    cached_params,
                                    "viz_f1_strict_internal_test_reference",
                                    default_test,
                                ),
                                default_test,
                            )
                        ),
                        key="viz_f1_strict_internal_test_reference",
                        help="Separate manual-reference subset used only to score the optional internal check.",
                    )

        with tabs[1]:
            markers, morphology_options, compartment_options = _marker_panel_controls(
                unified_csv_path,
                available_columns,
                cached_params,
            )
            with st.expander("Optional morphology and compartment features"):
                morphology_columns = st.multiselect(
                    "Morphology columns",
                    options=morphology_options,
                    default=_safe_list(
                        morphology_options,
                        _cached(
                            cached_params,
                            "viz_f1_strict_morphology",
                            [],
                        ),
                        [],
                    ),
                    key="viz_f1_strict_morphology",
                    help="Optional numeric cell-shape features such as area or eccentricity. Leave empty to train only on selected expression markers.",
                )
                compartment_columns = st.multiselect(
                    "Compartment columns",
                    options=compartment_options,
                    default=_safe_list(
                        compartment_options,
                        _cached(
                            cached_params,
                            "viz_f1_strict_compartments",
                            [],
                        ),
                        [],
                    ),
                    key="viz_f1_strict_compartments",
                    help="Optional numeric nuclear, cytoplasmic, or membrane features. Leave empty unless these were part of the intended ML workflow.",
                )

            weight_cols = st.columns(3)
            with weight_cols[0]:
                marker_weight = st.number_input(
                    "Marker weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(
                        _cached(
                            cached_params,
                            "viz_f1_strict_marker_weight",
                            1.0,
                        )
                    ),
                    step=0.1,
                    key="viz_f1_strict_marker_weight",
                    help="Multiplicative scaling applied to expression-marker features before fitting. Use 1 for their original relative contribution; 0 excludes the feature block.",
                )
            with weight_cols[1]:
                morphology_weight = st.number_input(
                    "Morphology weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(
                        _cached(
                            cached_params,
                            "viz_f1_strict_morphology_weight",
                            1.0,
                        )
                    ),
                    step=0.1,
                    key="viz_f1_strict_morphology_weight",
                    disabled=not bool(morphology_columns),
                    help="Multiplicative scaling for the optional morphology feature block. It is not a learned feature-importance score.",
                )
            with weight_cols[2]:
                compartment_weight = st.number_input(
                    "Compartment weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(
                        _cached(
                            cached_params,
                            "viz_f1_strict_compartment_weight",
                            0.5,
                        )
                    ),
                    step=0.1,
                    key="viz_f1_strict_compartment_weight",
                    disabled=not bool(compartment_columns),
                    help="Multiplicative scaling for the optional compartment feature block. It is not a learned feature-importance score.",
                )

            preprocessing_cols = st.columns(2)
            with preprocessing_cols[0]:
                use_z_normalization = st.checkbox(
                    "Batch-wise z-normalize selected features",
                    value=bool(
                        _cached(
                            cached_params,
                            "viz_f1_strict_z_normalization",
                            False,
                        )
                    ),
                    key="viz_f1_strict_z_normalization",
                    help=(
                        "Leave off for the current batch-corrected manuscript "
                        "markers unless this matches the deployed ML configuration."
                    ),
                )
            batch_default = (
                "batch_id"
                if "batch_id" in available_columns
                else validation_group_column
            )
            with preprocessing_cols[1]:
                batch_normalization_column = st.selectbox(
                    "Normalization batch column",
                    options=available_columns,
                    index=available_columns.index(
                        _safe_choice(
                            available_columns,
                            _cached(
                                cached_params,
                                "viz_f1_strict_batch_normalization_column",
                                batch_default,
                            ),
                            batch_default,
                        )
                    ),
                    key="viz_f1_strict_batch_normalization_column",
                    disabled=not use_z_normalization,
                    help="Column defining the batches within which selected features are standardized when batch-wise z-normalization is enabled.",
                )

            drop_invalid = st.checkbox(
                "Exclude rows with missing or infinite selected marker values",
                value=bool(
                    _cached(
                        cached_params,
                        "viz_f1_strict_drop_invalid",
                        True,
                    )
                ),
                key="viz_f1_strict_drop_invalid",
                help="Recommended. Removes cells that contain NaN or infinite values in any selected marker rather than passing invalid inputs to a model.",
            )

            with st.expander("Optional cell eligibility filter"):
                filter_enabled = st.checkbox(
                    "Filter cells before model fitting and validation",
                    value=bool(
                        _cached(
                            cached_params,
                            "viz_f1_strict_filter_enabled",
                            False,
                        )
                    ),
                    key="viz_f1_strict_filter_enabled",
                    help="Restrict both training and validation to cells matching one exact column value, for example tissue = 1.",
                )
                filter_cols = st.columns(2)
                default_filter = (
                    "tissue" if "tissue" in available_columns else available_columns[0]
                )
                with filter_cols[0]:
                    filter_column = st.selectbox(
                        "Filter column",
                        options=available_columns,
                        index=available_columns.index(
                            _safe_choice(
                                available_columns,
                                _cached(
                                    cached_params,
                                    "viz_f1_strict_filter_column",
                                    default_filter,
                                ),
                                default_filter,
                            )
                        ),
                        key="viz_f1_strict_filter_column",
                        disabled=not filter_enabled,
                        help="Column evaluated by the optional exact-value eligibility filter.",
                    )
                with filter_cols[1]:
                    filter_value = st.text_input(
                        "Required value",
                        value=str(
                            _cached(
                                cached_params,
                                "viz_f1_strict_filter_value",
                                "1",
                            )
                        ),
                        key="viz_f1_strict_filter_value",
                        disabled=not filter_enabled,
                        help="Only rows equal to this value are retained. Numeric columns are compared numerically when possible.",
                    )

        with tabs[2]:
            quick_mode = st.checkbox(
                "Fast preview (fewer trees, not for final results)",
                value=bool(
                    _cached(
                        cached_params,
                        "viz_f1_strict_quick_mode",
                        True,
                    )
                ),
                key="viz_f1_strict_quick_mode",
                help=(
                    "Caps both models at the preview tree count. Use the full "
                    "configuration only for the final manuscript run."
                ),
            )
            model_display = st.selectbox(
                "Classifier",
                options=list(MODEL_OPTIONS),
                index=list(MODEL_OPTIONS).index(
                    _safe_choice(
                        list(MODEL_OPTIONS),
                        _cached(
                            cached_params,
                            "viz_f1_strict_model",
                            "Random Forest",
                        ),
                        "Random Forest",
                    )
                ),
                key="viz_f1_strict_model",
                help="Classifier fitted anew inside every held-out fold. Random Forest reproduces the reference workflow.",
            )

            tree_cols = st.columns(3)
            with tree_cols[0]:
                mr_estimators = st.number_input(
                    "MR-ML trees",
                    min_value=10,
                    max_value=10000,
                    value=int(
                        _cached(
                            cached_params,
                            "viz_f1_strict_mr_estimators",
                            5000,
                        )
                    ),
                    step=50,
                    key="viz_f1_strict_mr_estimators",
                    help="Number of trees fitted for ML-MR inside each held-out fold. More trees improve stability but increase runtime and memory use.",
                )
            with tree_cols[1]:
                consensus_estimators = st.number_input(
                    "Consensus-ML trees",
                    min_value=10,
                    max_value=10000,
                    value=int(
                        _cached(
                            cached_params,
                            "viz_f1_strict_consensus_estimators",
                            1500,
                        )
                    ),
                    step=50,
                    key="viz_f1_strict_consensus_estimators",
                    help="Number of trees fitted for Consensus-ML inside each held-out fold. The reference analysis used a separate value from ML-MR.",
                )
            with tree_cols[2]:
                quick_estimators = st.number_input(
                    "Trees to use in preview",
                    min_value=5,
                    max_value=500,
                    value=int(
                        _cached(
                            cached_params,
                            "viz_f1_strict_quick_estimators",
                            100,
                        )
                    ),
                    step=5,
                    key="viz_f1_strict_quick_estimators",
                    disabled=not quick_mode,
                    help="Temporary per-model tree cap used only while Fast preview is enabled.",
                )

            parameter_cols = st.columns(3)
            with parameter_cols[0]:
                random_state = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=int(
                        _cached(
                            cached_params,
                            "viz_f1_strict_random_state",
                            42,
                        )
                    ),
                    step=1,
                    key="viz_f1_strict_random_state",
                    help="Makes sampling and model fitting reproducible. Keep the same seed when comparing datasets or regenerating publication figures.",
                )
            max_consensus = max(2, len(component_methods))
            default_consensus = min(3, max_consensus)
            clamp_numeric_widget_state(
                "viz_f1_strict_consensus_min_methods", 2, int(max_consensus),
                fallback=int(default_consensus),
            )
            with parameter_cols[1]:
                consensus_min_methods = st.number_input(
                    "How many methods must agree to call it consensus",
                    min_value=2,
                    max_value=max_consensus,
                    value=int(
                        min(
                            max_consensus,
                            max(
                                2,
                                int(
                                    _cached(
                                        cached_params,
                                        "viz_f1_strict_consensus_min_methods",
                                        default_consensus,
                                    )
                                ),
                            ),
                        )
                    ),
                    step=1,
                    key="viz_f1_strict_consensus_min_methods",
                    help=(
                        "A cell is direct consensus only when at least this many "
                        "selected component methods assign the same phenotype. The "
                        "reference setup uses 3 of 4 methods. With four methods, 2 is "
                        "looser (more direct calls, potentially noisier), 3 is the "
                        "reference default, and 4 requires unanimity (fewer direct "
                        "calls, more cells routed through Consensus-ML). With only two "
                        "methods, use 2."
                    ),
                )
            with parameter_cols[2]:
                confidence_threshold = st.slider(
                    "Confidence below which a call is replaced",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(
                        _cached(
                            cached_params,
                            "viz_f1_strict_confidence_threshold",
                            0.75,
                        )
                    ),
                    step=0.01,
                    key="viz_f1_strict_confidence_threshold",
                    help="Used only for the optional refined prediction columns: model calls below this probability are relabelled with the low-confidence label. Raw predictions remain available.",
                )

            final_cols = st.columns(3)
            with final_cols[0]:
                low_confidence_label = st.text_input(
                    "Low-confidence label",
                    value=str(
                        _cached(
                            cached_params,
                            "viz_f1_strict_low_confidence_label",
                            "Others",
                        )
                    ),
                    key="viz_f1_strict_low_confidence_label",
                    help="Category assigned to calls below the refinement confidence threshold, normally Others or Unclassified.",
                )
            with final_cols[1]:
                minimum_consensus_training = st.number_input(
                    "Minimum consensus cells needed to train a fold",
                    min_value=2,
                    max_value=10000,
                    value=int(
                        _cached(
                            cached_params,
                            "viz_f1_strict_min_consensus_training",
                            50,
                        )
                    ),
                    step=10,
                    key="viz_f1_strict_min_consensus_training",
                    help="Safety minimum for direct-consensus pseudo-labelled training cells remaining outside a held-out group. A fold stops instead of fitting an unreliable model below this count.",
                )
            with final_cols[2]:
                include_refined = st.checkbox(
                    "Also score the confidence-filtered versions separately",
                    value=bool(
                        _cached(
                            cached_params,
                            "viz_f1_strict_include_refined",
                            False,
                        )
                    ),
                    key="viz_f1_strict_include_refined",
                    help="Adds separate score rows where predictions below the confidence threshold are replaced by the low-confidence label; raw model scores are still retained.",
                )

            if quick_mode:
                st.info(
                    f"Preview will use at most {int(quick_estimators):,} trees "
                    "per model in each fold."
                )
            else:
                st.warning(
                    "Full grouped retraining may take substantial time and memory. "
                    "Use it for the final fixed manuscript configuration."
                )

        return {
            "enabled": True,
            "primary_reference": primary_reference,
            "component_methods": list(component_methods),
            "validation_group_column": validation_group_column,
            "group_mode": group_mode,
            "group_mapping": parse_mapping_text(group_mapping_text),
            "run_grouped_mr": bool(run_grouped_mr),
            "run_grouped_consensus": bool(run_grouped_consensus),
            "run_internal": bool(run_internal),
            "development_reference": development_reference,
            "internal_test_reference": internal_test_reference,
            "markers": list(markers),
            "morphology_columns": list(morphology_columns),
            "compartment_columns": list(compartment_columns),
            "feature_weights": {
                "markers": float(marker_weight),
                "morphology": float(morphology_weight),
                "compartments": float(compartment_weight),
            },
            "use_z_normalization": bool(use_z_normalization),
            "batch_normalization_column": batch_normalization_column,
            "drop_invalid_marker_rows": bool(drop_invalid),
            "filter_column": filter_column if filter_enabled else None,
            "filter_value": filter_value if filter_enabled else None,
            "ml_method": MODEL_OPTIONS[model_display],
            "mr_n_estimators": int(mr_estimators),
            "consensus_n_estimators": int(consensus_estimators),
            "quick_mode": bool(quick_mode),
            "quick_n_estimators": int(quick_estimators),
            "random_state": int(random_state),
            "consensus_min_methods": int(consensus_min_methods),
            "minimum_consensus_training": int(minimum_consensus_training),
            "confidence_threshold": float(confidence_threshold),
            "low_confidence_label": str(low_confidence_label).strip() or "Others",
            "include_refined": bool(include_refined),
        }
