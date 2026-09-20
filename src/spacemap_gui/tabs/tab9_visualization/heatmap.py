"""Publication heatmap and phenotype dot-plot visualization subtab."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from spacemap_gui.tabs.tab7_phenotyping.common import PhenotypeDotPlot, get_named_marker_panels

from .common import (
    apply_custom_order,
    apply_value_filters,
    cluster_order,
    extract_list_from_notebook,
    extract_scalar_from_notebook,
    get_cached_unified_df,
    get_filter_value_options,
    get_low_cardinality_filter_columns,
    read_heatmap_filter_specs,
    save_cached_params,
)
from .heatmap_utils import (
    INPUT_AS_IS,
    INPUT_TRANSFORM_OPTIONS,
    apply_marker_transformations,
    build_group_statistics,
    normalization_label,
    publication_file_stem,
    settings_as_json,
    unique_display_labels,
)
from .phenotype_colors import (
    add_category_color_strip,
    apply_category_tick_colors,
    category_labels_for_columns,
    render_phenotype_color_controls,
    shared_phenotype_colors,
)


FIGURE_TYPES = ("Heatmap", "Dot Plot with Cell Count Bar Chart")
MARKER_ORDER_OPTIONS = ("Selected order", "Alphabetical", "Clustered", "Custom")
PHENOTYPE_ORDER_OPTIONS = (
    "Cell count (descending)",
    "Cell count (ascending)",
    "Alphabetical",
    "Observed order",
    "Clustered",
    "Custom",
)
COLOR_MAPS = ("vlag", "coolwarm", "RdBu_r", "icefire", "viridis", "magma", "cividis", "rocket")


def _safe_choice(value: Any, options: tuple[str, ...], default: str) -> str:
    candidate = str(value or "")
    return candidate if candidate in options else default


def _marker_candidates(columns: list[str], df: pd.DataFrame | None) -> list[str]:
    """Return plausible expression columns while excluding known metadata."""
    blocked_exact = {
        "cell_id",
        "batch_id",
        "tissue",
        "tissue_roi",
        "roi",
        "slide",
        "slide_id",
        "patient",
        "patient_id",
        "sample",
        "sample_id",
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
        "corrected",
    }
    blocked_prefixes = (
        "phenotype_",
        "umap_",
        "leiden_",
        "cluster_",
        "confidence_",
        "probability_",
    )
    candidates: list[str] = []
    for column in columns:
        column_text = str(column)
        column_lower = column_text.lower()
        if column_lower in blocked_exact or column_lower.startswith(blocked_prefixes):
            continue
        if df is not None and column_text in df.columns and not pd.api.types.is_numeric_dtype(df[column_text]):
            continue
        candidates.append(column_text)
    return candidates


def _automatic_color_range(mean_values: pd.DataFrame, center_zero: bool) -> tuple[float, float]:
    finite = mean_values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lower = float(np.nanpercentile(finite, 1.0))
    upper = float(np.nanpercentile(finite, 99.0))
    if center_zero:
        bound = max(abs(lower), abs(upper), 1e-6)
        return -bound, bound
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.5)
        return lower - padding, upper + padding
    return lower, upper


def _ordered_statistics(
    means: pd.DataFrame,
    fractions: pd.DataFrame,
    counts: pd.Series,
    *,
    marker_mode: str,
    phenotype_mode: str,
    marker_custom: list[str],
    phenotype_custom: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    markers = list(means.columns)
    phenotypes = list(means.index)

    if marker_mode == "Alphabetical":
        markers = sorted(markers, key=lambda value: str(value).casefold())
    elif marker_mode == "Custom":
        markers = apply_custom_order(markers, marker_custom)
    elif marker_mode == "Clustered":
        markers = cluster_order(means, axis=1)

    if phenotype_mode == "Cell count (descending)":
        phenotypes = list(counts.sort_values(ascending=False, kind="stable").index)
    elif phenotype_mode == "Cell count (ascending)":
        phenotypes = list(counts.sort_values(ascending=True, kind="stable").index)
    elif phenotype_mode == "Alphabetical":
        phenotypes = sorted(phenotypes, key=lambda value: str(value).casefold())
    elif phenotype_mode == "Custom":
        phenotypes = apply_custom_order(phenotypes, phenotype_custom)
    elif phenotype_mode == "Clustered":
        phenotypes = cluster_order(means, axis=0)

    return (
        means.loc[phenotypes, markers],
        fractions.loc[phenotypes, markers],
        counts.loc[phenotypes],
    )


def _initial_publication_output_dir(output_dir_default: str, cached_params: dict[str, Any]) -> str:
    cached = str(cached_params.get("viz_heatmap_output_dir", "") or "").strip()
    if cached:
        return cached
    return str(Path(output_dir_default) / "publication_heatmaps")


def render_heatmap(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
    output_dir_default: str,
) -> None:
    st.markdown("### Heatmaps and Dot Plots")
    st.caption(
        "Generate publication-ready marker summaries for one or more phenotyping columns. "
        "Each selected phenotype column is exported as a separate figure."
    )

    loaded_df: pd.DataFrame | None = None
    if unified_csv_path and os.path.exists(unified_csv_path):
        try:
            loaded_df = get_cached_unified_df(unified_csv_path)
        except Exception as exc:
            st.warning(f"The selected CSV could not be loaded: {exc}")

    phenotype_candidates = [
        column
        for column in available_columns
        if column.lower().startswith("phenotype_") or column.lower().endswith("_phenotype")
    ]
    remaining_group_candidates = [column for column in available_columns if column not in phenotype_candidates]
    group_candidates = phenotype_candidates + remaining_group_candidates
    notebook_default_pheno = extract_scalar_from_notebook("PHENOTYPE_COLUMN")
    default_pheno = (
        notebook_default_pheno
        if notebook_default_pheno in group_candidates
        else (phenotype_candidates[0] if phenotype_candidates else (group_candidates[0] if group_candidates else ""))
    )

    notebook_markers = extract_list_from_notebook(["MARKERS"])
    all_marker_options = _marker_candidates(available_columns, loaded_df)
    notebook_marker_defaults = [marker for marker in notebook_markers if marker in all_marker_options]

    try:
        shared_marker_panels, marker_workspace = get_named_marker_panels(all_marker_options, unified_csv_path)
        st.session_state["tab7_marker_workspace"] = marker_workspace
    except Exception as exc:
        shared_marker_panels = {}
        st.warning(f"Saved Tab 7 marker panels could not be loaded: {exc}")

    st.markdown("#### Marker Configuration")
    st.info(
        "Reusable marker lists are created in **Step 7: Phenotyping**, in the common marker workspace "
        "shown above the individual phenotyping methods. Name and save a panel there, then return here "
        "and select **Saved Tab 7 panel**. Choose **Manual selection** to use any numeric marker column directly."
    )

    marker_source_options = ["Saved Tab 7 panel", "Manual selection"] if shared_marker_panels else ["Manual selection"]
    cached_source = str(cached_params.get("viz_heatmap_marker_source", marker_source_options[0]))
    if cached_source not in marker_source_options:
        cached_source = marker_source_options[0]
    marker_source = st.radio(
        "Marker source",
        marker_source_options,
        index=marker_source_options.index(cached_source),
        horizontal=True,
        key="viz_heatmap_marker_source",
        help="A saved panel gives consistent marker definitions across phenotyping and visualization; manual mode is unrestricted.",
    )

    selected_panel_name = "manual"
    if marker_source == "Saved Tab 7 panel":
        panel_names = list(shared_marker_panels)
        cached_panel = str(cached_params.get("viz_heatmap_shared_panel_name", panel_names[0]))
        if st.session_state.get("viz_heatmap_shared_panel_name") not in shared_marker_panels:
            st.session_state["viz_heatmap_shared_panel_name"] = cached_panel if cached_panel in shared_marker_panels else panel_names[0]
        selected_panel_name = st.selectbox(
            "Saved marker panel",
            panel_names,
            key="viz_heatmap_shared_panel_name",
            help="Panels are stored by the common marker workspace in Step 7.",
        )
        active_marker_options = list(shared_marker_panels[selected_panel_name])
        source_defaults = list(active_marker_options)
    else:
        active_marker_options = list(all_marker_options)
        source_defaults = notebook_marker_defaults or list(active_marker_options)

    source_token = f"{unified_csv_path}|{marker_source}|{selected_panel_name}"
    if st.session_state.get("viz_heatmap_marker_source_token") != source_token:
        cached_markers = [
            marker
            for marker in cached_params.get("viz_heatmap_markers", [])
            if marker in active_marker_options
        ]
        cached_marker_source = str(cached_params.get("viz_heatmap_marker_source", marker_source))
        cached_panel_name = str(cached_params.get("viz_heatmap_shared_panel_name", selected_panel_name))
        can_restore_cached = (
            cached_markers
            and cached_marker_source == marker_source
            and (marker_source == "Manual selection" or cached_panel_name == selected_panel_name)
        )
        st.session_state["viz_heatmap_marker_multiselect"] = (
            list(cached_markers) if can_restore_cached else list(source_defaults)
        )
        st.session_state["viz_heatmap_marker_source_token"] = source_token
    else:
        st.session_state["viz_heatmap_marker_multiselect"] = [
            marker
            for marker in st.session_state.get("viz_heatmap_marker_multiselect", source_defaults)
            if marker in active_marker_options
        ]

    def _select_all_markers() -> None:
        st.session_state["viz_heatmap_marker_multiselect"] = list(active_marker_options)

    def _clear_markers() -> None:
        st.session_state["viz_heatmap_marker_multiselect"] = []

    marker_button_1, marker_button_2 = st.columns(2)
    with marker_button_1:
        st.button("Select all available markers", key="viz_heatmap_select_all", on_click=_select_all_markers)
    with marker_button_2:
        st.button("Clear marker selection", key="viz_heatmap_clear_all", on_click=_clear_markers)

    selected_markers = st.multiselect(
        "Markers",
        options=active_marker_options,
        key="viz_heatmap_marker_multiselect",
        help="The displayed order is retained when Marker order is set to Selected order.",
    )
    st.session_state["viz_heatmap_confirmed_markers"] = list(selected_markers)
    st.session_state["viz_heatmap_markers"] = list(selected_markers)
    st.caption(f"Selected {len(selected_markers)} of {len(active_marker_options)} available marker columns.")

    # Filters remain outside the generation form so row counts can be previewed
    # immediately without reading the CSV again.
    cached_filters = cached_params.get("viz_heatmap_filters", [])
    if st.session_state.get("viz_heatmap_filter_file") != unified_csv_path:
        previous_filter_count = int(st.session_state.get("viz_heatmap_n_filters", 0) or 0)
        initial_filter_count = max(1, len(cached_filters))
        st.session_state["viz_heatmap_n_filters"] = initial_filter_count
        for index in range(initial_filter_count):
            spec = cached_filters[index] if index < len(cached_filters) else {}
            st.session_state[f"viz_heatmap_filter_col_{index}"] = str(spec.get("column", "") or "")
            st.session_state[f"viz_heatmap_filter_vals_{index}"] = [
                str(value) for value in spec.get("values", []) if str(value).strip()
            ]
        for index in range(initial_filter_count, previous_filter_count):
            st.session_state.pop(f"viz_heatmap_filter_col_{index}", None)
            st.session_state.pop(f"viz_heatmap_filter_vals_{index}", None)
        st.session_state["viz_heatmap_filter_file"] = unified_csv_path

    filter_columns = (
        get_low_cardinality_filter_columns(loaded_df, available_columns, excluded=all_marker_options)
        if loaded_df is not None
        else []
    )
    st.markdown("#### Optional Data Filters")
    st.caption("Filters are combined with AND logic. Empty filters are ignored.")
    if loaded_df is None:
        st.info("Load a valid unified CSV to enable filters.")
    elif not filter_columns:
        st.info("No categorical or low-cardinality filter columns were detected.")
    else:
        filter_count = max(1, int(st.session_state.get("viz_heatmap_n_filters", 1) or 1))
        preview_specs: list[tuple[str, list[str]]] = []
        for filter_index in range(filter_count):
            column_key = f"viz_heatmap_filter_col_{filter_index}"
            values_key = f"viz_heatmap_filter_vals_{filter_index}"
            if st.session_state.get(column_key, "") not in ([""] + filter_columns):
                st.session_state[column_key] = ""

            filter_col_1, filter_col_2 = st.columns([2, 3])
            with filter_col_1:
                selected_filter_column = st.selectbox(
                    f"Filter {filter_index + 1} column",
                    options=[""] + filter_columns,
                    key=column_key,
                    format_func=lambda value: "Select a column" if value == "" else value,
                )
            value_options = (
                get_filter_value_options(loaded_df, selected_filter_column) if selected_filter_column else []
            )
            st.session_state[values_key] = [
                value for value in st.session_state.get(values_key, []) if value in value_options
            ]
            with filter_col_2:
                selected_filter_values = st.multiselect(
                    f"Filter {filter_index + 1} values",
                    options=value_options,
                    key=values_key,
                    help="Select one or more values to retain. Leave empty to disable this row.",
                )
            if selected_filter_column and selected_filter_values:
                preview_specs.append((selected_filter_column, selected_filter_values))

        def _add_filter() -> None:
            st.session_state["viz_heatmap_n_filters"] = filter_count + 1

        def _remove_filter() -> None:
            last_index = filter_count - 1
            st.session_state.pop(f"viz_heatmap_filter_col_{last_index}", None)
            st.session_state.pop(f"viz_heatmap_filter_vals_{last_index}", None)
            st.session_state["viz_heatmap_n_filters"] = max(1, last_index)

        def _clear_filters() -> None:
            for index in range(filter_count):
                st.session_state[f"viz_heatmap_filter_col_{index}"] = ""
                st.session_state[f"viz_heatmap_filter_vals_{index}"] = []

        filter_button_1, filter_button_2, filter_button_3 = st.columns(3)
        with filter_button_1:
            st.button("+ Add filter", key="viz_heatmap_add_filter", on_click=_add_filter)
        with filter_button_2:
            st.button(
                "− Remove filter",
                key="viz_heatmap_remove_filter",
                on_click=_remove_filter,
                disabled=filter_count <= 1,
            )
        with filter_button_3:
            st.button("Clear filters", key="viz_heatmap_clear_filters", on_click=_clear_filters)

        if preview_specs:
            preview_df, preview_summary = apply_value_filters(loaded_df, preview_specs)
            st.success(f"{len(preview_df):,} / {len(loaded_df):,} cells match the active filters.")
            for summary_line in preview_summary:
                st.caption(summary_line)
        else:
            st.info(f"No filters are active; all {len(loaded_df):,} cells will be used.")

    if (
        "viz_heatmap_pheno_cols_select" not in st.session_state
        or st.session_state.get("viz_heatmap_pheno_column_file") != unified_csv_path
    ):
        current_pheno_defaults = cached_params.get(
            "viz_heatmap_pheno_cols",
            [default_pheno] if default_pheno else [],
        )
        st.session_state["viz_heatmap_pheno_cols_select"] = [
            column for column in current_pheno_defaults if column in group_candidates
        ]
        st.session_state["viz_heatmap_pheno_column_file"] = unified_csv_path
    else:
        st.session_state["viz_heatmap_pheno_cols_select"] = [
            column
            for column in st.session_state.get("viz_heatmap_pheno_cols_select", [])
            if column in group_candidates
        ]

    st.markdown("#### Figure Configuration")
    with st.container(border=True):
        phenotype_columns = st.multiselect(
            "Phenotyping / grouping columns",
            options=group_candidates,
            key="viz_heatmap_pheno_cols_select",
            help="Select more than one column to export one independent figure and summary for each method.",
        )
        st.caption("Phenotype columns are listed first; any categorical grouping column may also be used.")
        phenotype_color_map = render_phenotype_color_controls(
            category_labels_for_columns(loaded_df, phenotype_columns),
            panel_key="tab9_heatmap",
            title="Phenotype annotation colors",
        )

        cached_figure_type = _safe_choice(
            cached_params.get("viz_heatmap_figure_type", "Heatmap"), FIGURE_TYPES, "Heatmap"
        )
        figure_type = st.radio(
            "Figure type",
            FIGURE_TYPES,
            index=FIGURE_TYPES.index(cached_figure_type),
            horizontal=True,
            key="viz_heatmap_figure_type",
        )

        st.markdown("##### Transformation and Normalization")
        transform_col_1, transform_col_2 = st.columns([2, 1])
        with transform_col_1:
            cached_input_transform = _safe_choice(
                cached_params.get("viz_heatmap_input_transform", INPUT_AS_IS),
                INPUT_TRANSFORM_OPTIONS,
                INPUT_AS_IS,
            )
            input_transform = st.selectbox(
                "Input-scale transformation",
                INPUT_TRANSFORM_OPTIONS,
                index=INPUT_TRANSFORM_OPTIONS.index(cached_input_transform),
                key="viz_heatmap_input_transform",
                help=(
                    "ArcSinh applies arcsinh(value/cofactor). Inverse ArcSinh applies sinh(value)*cofactor only "
                    "to columns ending in _corr, matching the optional phenotyping visualization behavior."
                ),
            )
        with transform_col_2:
            arcsinh_cofactor = st.number_input(
                "ArcSinh cofactor",
                min_value=0.001,
                max_value=1000.0,
                value=float(cached_params.get("viz_heatmap_arcsinh_cofactor", 5.0)),
                step=0.5,
                key="viz_heatmap_arcsinh_cofactor",
                disabled=input_transform == INPUT_AS_IS,
            )

        norm_col_1, norm_col_2, norm_col_3, norm_col_4 = st.columns(4)
        with norm_col_1:
            apply_z1 = st.checkbox(
                "Z1",
                value=bool(cached_params.get("viz_heatmap_norm_z1", True)),
                key="viz_heatmap_norm_z1",
                help="Marker-wise z-score. With grouping columns, each marker is standardized independently within each group.",
            )
        with norm_col_2:
            apply_z2 = st.checkbox(
                "Z2",
                value=bool(cached_params.get("viz_heatmap_norm_z2", False)),
                key="viz_heatmap_norm_z2",
                help="Cell-wise z-score across the selected markers.",
            )
        with norm_col_3:
            apply_cdf = st.checkbox(
                "CDF",
                value=bool(cached_params.get("viz_heatmap_norm_cdf", False)),
                key="viz_heatmap_norm_cdf",
                help="Apply the standard normal cumulative distribution function to the current values.",
            )
        with norm_col_4:
            apply_nlog = st.checkbox(
                "NLog",
                value=bool(cached_params.get("viz_heatmap_norm_nlog", False)),
                key="viz_heatmap_norm_nlog",
                help="Apply -log(1-value). This is normally used after CDF.",
            )
        st.caption("Enabled normalization steps run in the same fixed order used by phenotyping: Z1 → Z2 → CDF → NLog.")
        if apply_nlog and not apply_cdf:
            st.warning("NLog is usually paired with CDF; without CDF, values at or above 1 are clipped before the log step.")

        default_group_columns = ["batch_id"] if "batch_id" in available_columns else []
        cached_group_columns = cached_params.get("viz_heatmap_groupby_cols", default_group_columns)
        normalization_groups = st.multiselect(
            "Normalization grouping columns",
            options=available_columns,
            default=[column for column in cached_group_columns if column in available_columns],
            key="viz_heatmap_groupby_cols",
            help=(
                "The complete selected normalization sequence is calculated independently within each group. "
                "Use batch_id for batch-wise normalization, or leave empty for global normalization."
            ),
        )

        st.markdown("##### Ordering")
        order_col_1, order_col_2 = st.columns(2)
        legacy_marker_order = str(cached_params.get("viz_heatmap_marker_order", "Selected order"))
        if legacy_marker_order == "Notebook":
            legacy_marker_order = "Selected order"
        if bool(cached_params.get("viz_heatmap_cluster_y", False)):
            legacy_marker_order = "Clustered"
        legacy_phenotype_order = str(cached_params.get("viz_heatmap_pheno_order", "Cell count (descending)"))
        if bool(cached_params.get("viz_heatmap_cluster_x", False)):
            legacy_phenotype_order = "Clustered"
        with order_col_1:
            marker_order = st.selectbox(
                "Marker order",
                MARKER_ORDER_OPTIONS,
                index=MARKER_ORDER_OPTIONS.index(
                    _safe_choice(legacy_marker_order, MARKER_ORDER_OPTIONS, "Selected order")
                ),
                key="viz_heatmap_marker_order",
            )
        with order_col_2:
            phenotype_order = st.selectbox(
                "Phenotype order",
                PHENOTYPE_ORDER_OPTIONS,
                index=PHENOTYPE_ORDER_OPTIONS.index(
                    _safe_choice(legacy_phenotype_order, PHENOTYPE_ORDER_OPTIONS, "Cell count (descending)")
                ),
                key="viz_heatmap_pheno_order",
            )
        st.caption(
            "Choose Clustered for both controls to cluster both biological axes. Ordering follows markers and "
            "phenotypes even when the heatmap axes are flipped."
        )
        marker_custom_text = st.text_area(
            "Custom marker order (comma-separated)",
            value=str(cached_params.get("viz_heatmap_marker_custom", "")),
            key="viz_heatmap_marker_custom",
            disabled=marker_order != "Custom",
        )
        phenotype_custom_text = st.text_area(
            "Custom phenotype order (comma-separated)",
            value=str(cached_params.get("viz_heatmap_pheno_custom", "")),
            key="viz_heatmap_pheno_custom",
            disabled=phenotype_order != "Custom",
        )

        st.markdown("##### Appearance and Export")
        appearance_col_1, appearance_col_2, appearance_col_3 = st.columns(3)
        with appearance_col_1:
            dpi_value = st.number_input(
                "PNG resolution (DPI)",
                min_value=300,
                max_value=1200,
                value=max(300, int(cached_params.get("viz_heatmap_dpi", 300))),
                step=50,
                key="viz_heatmap_dpi",
                help="Publication exports are restricted to at least 300 DPI.",
            )
        with appearance_col_2:
            figure_width = st.number_input(
                "Figure width (inches)",
                min_value=4.0,
                max_value=40.0,
                value=float(cached_params.get("viz_heatmap_width", 12.0)),
                step=0.5,
                key="viz_heatmap_width",
            )
        with appearance_col_3:
            figure_height = st.number_input(
                "Figure height (inches)",
                min_value=4.0,
                max_value=40.0,
                value=float(cached_params.get("viz_heatmap_height", 10.0)),
                step=0.5,
                key="viz_heatmap_height",
            )

        color_col_1, color_col_2, color_col_3 = st.columns(3)
        with color_col_1:
            cached_cmap = _safe_choice(cached_params.get("viz_heatmap_cmap", "vlag"), COLOR_MAPS, "vlag")
            color_map = st.selectbox(
                "Color map",
                COLOR_MAPS,
                index=COLOR_MAPS.index(cached_cmap),
                key="viz_heatmap_cmap",
            )
        with color_col_2:
            range_mode = st.selectbox(
                "Color range",
                ("Automatic (1st–99th percentile)", "Manual"),
                index=0
                if cached_params.get("viz_heatmap_range_mode", "Automatic (1st–99th percentile)")
                == "Automatic (1st–99th percentile)"
                else 1,
                key="viz_heatmap_range_mode",
            )
        with color_col_3:
            center_zero = st.checkbox(
                "Center color scale at zero",
                value=bool(cached_params.get("viz_heatmap_center_zero", True)),
                key="viz_heatmap_center_zero",
                help="When automatic, use symmetric limits around zero. When manual, place the color-map center at zero if zero is within the selected limits.",
            )

        range_col_1, range_col_2 = st.columns(2)
        with range_col_1:
            color_minimum = st.number_input(
                "Color scale minimum",
                value=float(cached_params.get("viz_heatmap_zmin", -4.0)),
                step=0.25,
                key="viz_heatmap_zmin",
                disabled=range_mode != "Manual",
            )
        with range_col_2:
            color_maximum = st.number_input(
                "Color scale maximum",
                value=float(cached_params.get("viz_heatmap_zmax", 4.0)),
                step=0.25,
                key="viz_heatmap_zmax",
                disabled=range_mode != "Manual",
            )

        layout_col_1, layout_col_2, layout_col_3 = st.columns(3)
        with layout_col_1:
            flip_axes = st.checkbox(
                "Flip heatmap axes",
                value=bool(cached_params.get("viz_heatmap_flip", False)),
                key="viz_heatmap_flip",
                disabled=figure_type != "Heatmap",
            )
        with layout_col_2:
            square_cells = st.checkbox(
                "Square heatmap cells",
                value=bool(cached_params.get("viz_heatmap_square", False)),
                key="viz_heatmap_square",
                disabled=figure_type != "Heatmap",
            )
        with layout_col_3:
            x_tick_rotation = st.number_input(
                "Marker-label rotation",
                min_value=0,
                max_value=90,
                value=int(cached_params.get("viz_heatmap_xtick_rotation", 45)),
                step=5,
                key="viz_heatmap_xtick_rotation",
            )

        suffix_text = st.text_input(
            "Marker suffixes to hide in the figure (comma-separated)",
            value=str(cached_params.get("viz_heatmap_marker_suffixes", "_corr,_norm,_unmixed,_resolve")),
            key="viz_heatmap_marker_suffixes",
            help="If stripping creates duplicate labels (for example CD4 and CD4_corr), the original names are retained to avoid ambiguity.",
        )

        corrected_only = st.checkbox(
            "Use corrected == 1 cells if the column exists",
            value=bool(cached_params.get("viz_heatmap_corrected_only", True)),
            key="viz_heatmap_corrected_only",
        )
        top_n_markers = st.number_input(
            "Top markers per phenotype in summary",
            min_value=3,
            max_value=20,
            value=int(cached_params.get("viz_heatmap_summary_top_n", 5)),
            step=1,
            key="viz_heatmap_summary_top_n",
        )

        dot_show_grid = bool(cached_params.get("viz_heatmap_dot_show_grid", True))
        dot_show_bars = bool(cached_params.get("viz_heatmap_dot_show_bars", True))
        dot_show_count_labels = bool(cached_params.get("viz_heatmap_dot_show_count_labels", True))
        dot_show_colorbar = bool(cached_params.get("viz_heatmap_dot_show_colorbar", True))
        dot_show_size_legend = bool(cached_params.get("viz_heatmap_dot_show_size_legend", True))
        dot_positive_threshold = float(cached_params.get("viz_heatmap_dot_positive_threshold", 0.0))
        dot_fill_percent = int(cached_params.get("viz_heatmap_dot_fill_percent", 72))
        dot_count_panel_gap = float(cached_params.get("viz_heatmap_dot_count_panel_gap", 0.06))
        dot_lower_panel_gap = float(cached_params.get("viz_heatmap_dot_lower_panel_gap", 0.06))
        if figure_type == "Dot Plot with Cell Count Bar Chart":
            st.markdown("##### Dot Plot Options")
            dot_col_1, dot_col_2, dot_col_3 = st.columns(3)
            with dot_col_1:
                dot_show_grid = st.checkbox(
                    "Show background grid dots", value=dot_show_grid, key="viz_heatmap_dot_show_grid"
                )
            with dot_col_2:
                dot_show_bars = st.checkbox(
                    "Show cell-count bars", value=dot_show_bars, key="viz_heatmap_dot_show_bars"
                )
            with dot_col_3:
                dot_show_count_labels = st.checkbox(
                    "Show count labels", value=dot_show_count_labels, key="viz_heatmap_dot_show_count_labels"
                )
            dot_col_4, dot_col_5 = st.columns(2)
            with dot_col_4:
                dot_show_colorbar = st.checkbox(
                    "Show expression colorbar", value=dot_show_colorbar, key="viz_heatmap_dot_show_colorbar"
                )
            with dot_col_5:
                dot_show_size_legend = st.checkbox(
                    "Show fraction-size legend", value=dot_show_size_legend, key="viz_heatmap_dot_show_size_legend"
                )
            dot_positive_threshold = st.number_input(
                "Positive-cell threshold on transformed values",
                value=dot_positive_threshold,
                step=0.1,
                key="viz_heatmap_dot_positive_threshold",
                help="Dot size is the percentage of cells above this threshold; dot color is the group mean.",
            )
            st.markdown("**Smart Dot and Panel Spacing**")
            spacing_col_1, spacing_col_2, spacing_col_3 = st.columns(3)
            with spacing_col_1:
                dot_fill_percent = st.slider(
                    "Maximum dot diameter (% of grid spacing)",
                    min_value=25,
                    max_value=90,
                    value=max(25, min(90, int(dot_fill_percent))),
                    step=1,
                    key="viz_heatmap_dot_fill_percent",
                    help=(
                        "The largest dot is dynamically capped to this fraction of the smaller row/column spacing. "
                        "Values below 100% prevent neighboring circles from overlapping."
                    ),
                )
            with spacing_col_2:
                dot_count_panel_gap = st.slider(
                    "Gap: dot plot to cell counts",
                    min_value=0.0,
                    max_value=0.50,
                    value=max(0.0, min(0.50, float(dot_count_panel_gap))),
                    step=0.01,
                    key="viz_heatmap_dot_count_panel_gap",
                    help="Horizontal spacing between the marker-dot matrix and the cell-count panel.",
                )
            with spacing_col_3:
                dot_lower_panel_gap = st.slider(
                    "Gap: marker labels to lower scale (inches)",
                    min_value=0.0,
                    max_value=0.60,
                    value=max(0.0, min(0.60, float(dot_lower_panel_gap))),
                    step=0.01,
                    key="viz_heatmap_dot_lower_panel_gap",
                    help=(
                        "Exact physical spacing between the lowest marker-axis label and the top of the color scale. "
                        "Use 0 for the tightest layout."
                    ),
                )
            st.caption(
                "Dot diameters adapt to the rendered grid. The vertical 0%, 25%, 50%, 75%, and 100% legend "
                "uses exactly the same circle areas as the dot matrix."
            )

        save_pdf = st.checkbox(
            "Also save a vector PDF",
            value=bool(cached_params.get("viz_heatmap_save_pdf", False)),
            key="viz_heatmap_save_pdf",
            help="PNG is always saved at the selected DPI; PDF is optional and useful for manuscript assembly.",
        )
        output_directory = st.text_input(
            "Publication output directory",
            value=_initial_publication_output_dir(output_dir_default, cached_params),
            key="viz_heatmap_output_dir",
        )

        action_col_1, action_col_2 = st.columns(2)
        with action_col_1:
            save_configuration = st.button(
                "Save Configuration",
                key="viz_heatmap_save_configuration",
                width="stretch",
            )
        with action_col_2:
            generate_now = st.button(
                "Generate Dot Plots"
                if figure_type == "Dot Plot with Cell Count Bar Chart"
                else "Generate Heatmaps",
                type="primary",
                key="viz_heatmap_generate",
                width="stretch",
            )

    active_filters = read_heatmap_filter_specs(
        int(st.session_state.get("viz_heatmap_n_filters", cached_params.get("viz_heatmap_n_filters", 1)) or 1)
    )
    serializable_filters = [
        {"column": column, "values": values} for column, values in active_filters
    ]
    configuration = {
        "viz_heatmap_pheno_cols": list(phenotype_columns),
        "viz_heatmap_marker_source": marker_source,
        "viz_heatmap_shared_panel_name": selected_panel_name,
        "viz_heatmap_markers": list(selected_markers),
        "viz_heatmap_figure_type": figure_type,
        "viz_heatmap_input_transform": input_transform,
        "viz_heatmap_arcsinh_cofactor": float(arcsinh_cofactor),
        "viz_heatmap_norm_z1": bool(apply_z1),
        "viz_heatmap_norm_z2": bool(apply_z2),
        "viz_heatmap_norm_cdf": bool(apply_cdf),
        "viz_heatmap_norm_nlog": bool(apply_nlog),
        "viz_heatmap_groupby_cols": list(normalization_groups),
        "viz_heatmap_marker_order": marker_order,
        "viz_heatmap_pheno_order": phenotype_order,
        "viz_heatmap_marker_custom": marker_custom_text,
        "viz_heatmap_pheno_custom": phenotype_custom_text,
        "viz_heatmap_dpi": int(dpi_value),
        "viz_heatmap_width": float(figure_width),
        "viz_heatmap_height": float(figure_height),
        "viz_heatmap_cmap": color_map,
        "viz_heatmap_range_mode": range_mode,
        "viz_heatmap_center_zero": bool(center_zero),
        "viz_heatmap_zmin": float(color_minimum),
        "viz_heatmap_zmax": float(color_maximum),
        "viz_heatmap_flip": bool(flip_axes),
        "viz_heatmap_square": bool(square_cells),
        "viz_heatmap_xtick_rotation": int(x_tick_rotation),
        "viz_heatmap_marker_suffixes": suffix_text,
        "viz_heatmap_corrected_only": bool(corrected_only),
        "viz_heatmap_summary_top_n": int(top_n_markers),
        "viz_heatmap_dot_show_grid": bool(dot_show_grid),
        "viz_heatmap_dot_show_bars": bool(dot_show_bars),
        "viz_heatmap_dot_show_count_labels": bool(dot_show_count_labels),
        "viz_heatmap_dot_show_colorbar": bool(dot_show_colorbar),
        "viz_heatmap_dot_show_size_legend": bool(dot_show_size_legend),
        "viz_heatmap_dot_positive_threshold": float(dot_positive_threshold),
        "viz_heatmap_dot_fill_percent": int(dot_fill_percent),
        "viz_heatmap_dot_count_panel_gap": float(dot_count_panel_gap),
        "viz_heatmap_dot_lower_panel_gap": float(dot_lower_panel_gap),
        "viz_heatmap_save_pdf": bool(save_pdf),
        "viz_heatmap_output_dir": output_directory,
        "viz_heatmap_filters": serializable_filters,
        "viz_heatmap_n_filters": int(st.session_state.get("viz_heatmap_n_filters", 1) or 1),
    }

    if save_configuration:
        st.session_state["viz_heatmap_pheno_cols"] = list(phenotype_columns)
        cached_params.update(configuration)
        save_cached_params(project_root, cached_params)
        st.success("Heatmap configuration saved.")

    if not generate_now:
        return

    st.session_state["viz_heatmap_pheno_cols"] = list(phenotype_columns)
    cached_params.update(configuration)
    save_cached_params(project_root, cached_params)

    if loaded_df is None:
        st.error("A valid unified CSV is required before figures can be generated.")
        return
    if not phenotype_columns:
        st.error("Select at least one phenotyping or grouping column.")
        return
    if not selected_markers:
        st.error("Select at least one marker.")
        return
    if range_mode == "Manual" and float(color_minimum) >= float(color_maximum):
        st.error("The manual color-scale minimum must be smaller than the maximum.")
        return

    working_df = loaded_df.copy()
    initial_cell_count = len(working_df)
    if corrected_only and "corrected" in working_df.columns:
        working_df = working_df.loc[working_df["corrected"] == 1].copy()
        st.caption(f"corrected == 1 retained {len(working_df):,} / {initial_cell_count:,} cells.")

    if active_filters:
        working_df, filter_summary = apply_value_filters(working_df, active_filters)
        for summary_line in filter_summary:
            st.caption(summary_line)
    if working_df.empty:
        st.error("No cells remain after applying the selected filters.")
        return

    markers = [marker for marker in selected_markers if marker in working_df.columns]
    missing_markers = [marker for marker in selected_markers if marker not in working_df.columns]
    if missing_markers:
        st.warning(f"Unavailable markers were skipped: {', '.join(missing_markers)}")
    if not markers:
        st.error("None of the selected markers are present in the unified CSV.")
        return

    output_path = Path(output_directory).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        st.error(f"The output directory could not be created: {exc}")
        return

    marker_custom = [value.strip() for value in marker_custom_text.split(",") if value.strip()]
    phenotype_custom = [value.strip() for value in phenotype_custom_text.split(",") if value.strip()]
    suffixes = [value.strip() for value in suffix_text.split(",") if value.strip()]
    normalization_groups = [column for column in normalization_groups if column in working_df.columns]
    transform_description = normalization_label(
        input_transform=input_transform,
        apply_z1=apply_z1,
        apply_z2=apply_z2,
        apply_cdf=apply_cdf,
        apply_nlog=apply_nlog,
    )

    for phenotype_column in phenotype_columns:
        if phenotype_column not in working_df.columns:
            st.warning(f"Grouping column '{phenotype_column}' is unavailable and was skipped.")
            continue

        required_columns = list(dict.fromkeys([phenotype_column, *markers, *normalization_groups]))
        subset = working_df[required_columns].copy()
        valid_group = subset[phenotype_column].notna() & subset[phenotype_column].astype(str).str.strip().ne("")
        subset = subset.loc[valid_group].copy()
        if subset.empty:
            st.warning(f"'{phenotype_column}' has no labeled cells after filtering and was skipped.")
            continue

        numeric_markers = subset[markers].apply(pd.to_numeric, errors="coerce")
        invalid_rows = numeric_markers.notna().sum(axis=1).eq(0)
        if invalid_rows.any():
            dropped = int(invalid_rows.sum())
            subset = subset.loc[~invalid_rows].copy()
            st.warning(f"{phenotype_column}: removed {dropped:,} cells with no numeric values in the selected markers.")
        if subset.empty:
            st.warning(f"'{phenotype_column}' has no usable marker rows and was skipped.")
            continue

        try:
            transformed = apply_marker_transformations(
                subset,
                markers,
                group_columns=normalization_groups,
                input_transform=input_transform,
                arcsinh_cofactor=float(arcsinh_cofactor),
                apply_z1=bool(apply_z1),
                apply_z2=bool(apply_z2),
                apply_cdf=bool(apply_cdf),
                apply_nlog=bool(apply_nlog),
            )
            means, fractions, counts = build_group_statistics(
                transformed,
                subset[phenotype_column],
                positive_threshold=float(dot_positive_threshold),
            )
            means, fractions, counts = _ordered_statistics(
                means,
                fractions,
                counts,
                marker_mode=marker_order,
                phenotype_mode=phenotype_order,
                marker_custom=marker_custom,
                phenotype_custom=phenotype_custom,
            )
        except Exception as exc:
            st.error(f"Could not prepare '{phenotype_column}': {exc}")
            continue

        display_labels = unique_display_labels(list(means.columns), suffixes)
        means_display = means.copy()
        fractions_display = fractions.copy()
        means_display.columns = display_labels
        fractions_display.columns = display_labels

        if range_mode == "Automatic (1st–99th percentile)":
            effective_minimum, effective_maximum = _automatic_color_range(means_display, bool(center_zero))
        else:
            effective_minimum, effective_maximum = float(color_minimum), float(color_maximum)
        color_center = 0.0 if center_zero and effective_minimum < 0 < effective_maximum else None
        colorbar_label = f"Mean marker value ({transform_description})"
        active_phenotype_colors = shared_phenotype_colors([str(value) for value in means_display.index])
        active_phenotype_colors.update(
            {
                label: color
                for label, color in phenotype_color_map.items()
                if label in {str(value) for value in means_display.index}
            }
        )

        figure_kind = "dot-plot" if figure_type == "Dot Plot with Cell Count Bar Chart" else "heatmap"
        figure_settings: dict[str, Any] = {
            "input_csv": str(unified_csv_path or ""),
            "figure_kind": figure_kind,
            "phenotype_column": phenotype_column,
            "marker_source": marker_source,
            "marker_panel": selected_panel_name,
            "markers": list(means.columns),
            "marker_count": len(means.columns),
            "filters": serializable_filters,
            "corrected_only": bool(corrected_only),
            "normalization": transform_description,
            "input_transform": input_transform,
            "arcsinh_cofactor": float(arcsinh_cofactor),
            "normalization_groups": normalization_groups,
            "marker_order": marker_order,
            "phenotype_order": phenotype_order,
            "color_map": color_map,
            "color_range_mode": range_mode,
            "color_minimum": float(effective_minimum),
            "color_maximum": float(effective_maximum),
            "center_zero": bool(center_zero),
            "phenotype_colors": active_phenotype_colors,
            "figure_width_inches": float(figure_width),
            "figure_height_inches": float(figure_height),
            "png_dpi": int(dpi_value),
            "cell_count": int(counts.sum()),
            "group_count": int(len(counts)),
            "positive_threshold": float(dot_positive_threshold) if figure_kind == "dot-plot" else None,
            "dot_fill_percent": int(dot_fill_percent) if figure_kind == "dot-plot" else None,
            "dot_count_panel_gap": float(dot_count_panel_gap) if figure_kind == "dot-plot" else None,
            "dot_lower_panel_gap": float(dot_lower_panel_gap) if figure_kind == "dot-plot" else None,
        }
        file_stem = publication_file_stem(figure_kind, phenotype_column, figure_settings)

        title = f"{phenotype_column}: mean marker profile"
        with plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.titlesize": 14,
                "axes.labelsize": 11,
                "savefig.facecolor": "white",
                "figure.facecolor": "white",
            }
        ):
            if figure_type == "Dot Plot with Cell Count Bar Chart":
                thresholds = pd.Series(float(dot_positive_threshold), index=means_display.columns, dtype=float)
                dot_plot = PhenotypeDotPlot(
                    means_display,
                    fractions_display,
                    counts,
                    title=title,
                    color_vmin=effective_minimum,
                    color_vmax=effective_maximum,
                    colorbar_label=colorbar_label,
                    positive_thresholds=thresholds,
                    color_map=color_map,
                )
                figure = dot_plot.plot(
                    show_grid=dot_show_grid,
                    show_count_bars=dot_show_bars,
                    show_count_labels=dot_show_count_labels,
                    show_colorbar=dot_show_colorbar,
                    show_size_legend=dot_show_size_legend,
                    figsize=(float(figure_width), float(figure_height)),
                    marker_tick_rotation=float(x_tick_rotation),
                    dot_fill_fraction=float(dot_fill_percent) / 100.0,
                    count_panel_gap=float(dot_count_panel_gap),
                    lower_panel_gap=float(dot_lower_panel_gap),
                )
                apply_category_tick_colors(figure.axes[0], active_phenotype_colors, which="y")
                if dot_show_bars and len(figure.axes) > 1:
                    for bar, phenotype_name in zip(figure.axes[1].patches, means_display.index):
                        bar.set_facecolor(active_phenotype_colors.get(str(phenotype_name), "#777777"))
                download_label = f"Download {phenotype_column} dot plot"
            else:
                heatmap_data = means_display.T
                if flip_axes:
                    heatmap_data = heatmap_data.T
                figure, axis = plt.subplots(figsize=(float(figure_width), float(figure_height)))
                line_width = 0.4 if heatmap_data.size <= 1000 else 0.1
                sns.heatmap(
                    heatmap_data,
                    cmap=color_map,
                    center=color_center,
                    vmin=effective_minimum,
                    vmax=effective_maximum,
                    linewidths=line_width,
                    linecolor="#e6e6e6",
                    cbar_kws={"label": colorbar_label, "shrink": 0.85},
                    square=bool(square_cells),
                    ax=axis,
                )
                axis.set_title(title, pad=14, weight="semibold")
                axis.set_xlabel("Marker" if flip_axes else "Phenotype")
                axis.set_ylabel("Phenotype" if flip_axes else "Marker")
                axis.tick_params(axis="x", rotation=int(x_tick_rotation))
                for label in axis.get_xticklabels():
                    label.set_horizontalalignment("right" if int(x_tick_rotation) else "center")
                axis.tick_params(axis="y", rotation=0)
                apply_category_tick_colors(
                    axis,
                    active_phenotype_colors,
                    which="y" if flip_axes else "x",
                )
                add_category_color_strip(
                    axis,
                    heatmap_data.index if flip_axes else heatmap_data.columns,
                    active_phenotype_colors,
                    which="y" if flip_axes else "x",
                )
                figure.tight_layout()
                download_label = f"Download {phenotype_column} heatmap"

        png_path = output_path / f"{file_stem}.png"
        summary_path = output_path / f"{file_stem}__summary.csv"
        settings_path = output_path / f"{file_stem}__settings.json"
        figure.savefig(png_path, dpi=int(dpi_value), bbox_inches="tight", facecolor="white")
        if save_pdf:
            figure.savefig(output_path / f"{file_stem}.pdf", bbox_inches="tight", facecolor="white")

        summary_rows: list[dict[str, Any]] = []
        top_count = min(int(top_n_markers), len(means.columns))
        for phenotype_name, marker_scores in means.iterrows():
            top_markers = marker_scores.nlargest(top_count)
            summary_row: dict[str, Any] = {
                "Phenotype": str(phenotype_name),
                "Cell Count": int(counts.loc[phenotype_name]),
                "Percent of Included Cells": float(100.0 * counts.loc[phenotype_name] / max(counts.sum(), 1)),
            }
            for rank, (marker_name, marker_score) in enumerate(top_markers.items(), start=1):
                display_name = display_labels[list(means.columns).index(marker_name)]
                summary_row[f"Top {rank} Marker"] = display_name
                summary_row[f"Top {rank} Mean Value"] = float(marker_score)
            summary_rows.append(summary_row)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        settings_path.write_text(settings_as_json(figure_settings) + "\n", encoding="utf-8")

        st.markdown(f"##### {phenotype_column}")
        st.success(f"Saved 300+ DPI PNG: `{png_path}`")
        if save_pdf:
            st.caption(f"Vector PDF: `{output_path / f'{file_stem}.pdf'}`")
        st.caption(f"Reproducibility settings: `{settings_path}`")
        st.pyplot(figure)
        st.dataframe(summary_df, width="stretch", hide_index=True)

        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", dpi=int(dpi_value), bbox_inches="tight", facecolor="white")
        image_buffer.seek(0)
        st.download_button(
            label=download_label,
            data=image_buffer,
            file_name=png_path.name,
            mime="image/png",
        )
        plt.close(figure)
