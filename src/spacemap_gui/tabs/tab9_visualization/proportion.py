"""Proportion visualization subtab."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import streamlit as st

from .common import (
    extract_list_from_notebook,
    get_columns_df,
    log_console,
    logger,
    save_cached_params,
    timestamp,
)
from .phenotype_colors import (
    apply_category_tick_colors,
    category_labels_for_columns,
    render_phenotype_color_controls,
    shared_phenotype_colors,
)


_METHOD_HATCHES = ("", "//", "\\\\", "xx", "..", "++", "oo", "--")


def _make_proportion_figure(
    summary_long: pd.DataFrame,
    *,
    methods: list[str],
    phenotype_order: list[str],
    phenotype_colors: dict[str, str],
    figsize: tuple[float, float],
) -> plt.Figure:
    """Plot method comparisons while keeping color tied to cell phenotype."""
    figure, axis = plt.subplots(figsize=figsize)
    x_positions = np.arange(len(phenotype_order), dtype=float)
    n_methods = max(len(methods), 1)
    bar_width = 0.82 / n_methods

    for method_index, method in enumerate(methods):
        method_rows = summary_long.loc[summary_long["method"] == method]
        values = (
            method_rows.set_index("phenotype")["proportion"]
            .reindex(phenotype_order)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        offset = (method_index - (n_methods - 1) / 2.0) * bar_width
        axis.bar(
            x_positions + offset,
            values,
            width=bar_width,
            color=[phenotype_colors.get(label, "#777777") for label in phenotype_order],
            edgecolor="#3f3f3f",
            linewidth=0.55,
            hatch=_METHOD_HATCHES[method_index % len(_METHOD_HATCHES)],
            label=method,
        )

    axis.set_title("Phenotype Proportions Across Methods")
    axis.set_ylabel("Proportion of Cells")
    axis.set_xlabel("Phenotype")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(phenotype_order, rotation=45, ha="right")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    apply_category_tick_colors(axis, phenotype_colors, which="x")

    if methods:
        method_handles = [
            Patch(
                facecolor="white",
                edgecolor="#3f3f3f",
                hatch=_METHOD_HATCHES[index % len(_METHOD_HATCHES)],
                label=method,
            )
            for index, method in enumerate(methods)
        ]
        axis.legend(
            handles=method_handles,
            title="Phenotyping column",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    figure.tight_layout()
    return figure


def apply_cell_filters(df: pd.DataFrame, filter_rules: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Apply cell filter rules to df.
    Each rule specifies column, filter type (Categorical/Continuous), operator, and values/thresholds.
    """
    if not filter_rules or df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    for rule in filter_rules:
        col = str(rule.get("column", ""))
        if not col or col not in df.columns:
            continue

        series = df[col]
        is_missing = series.isna() | series.astype(str).str.strip().isin(["", "nan", "None", "NaN"])
        rule_type = rule.get("type", "Categorical")
        keep_missing = bool(rule.get("include_missing", False))

        if rule_type == "Categorical":
            cat_op = rule.get("cat_operator", "Is in (Include)")
            allowed_vals = [str(v) for v in (rule.get("cat_values") or [])]
            if allowed_vals:
                if cat_op == "Is in (Include)":
                    row_match = series.astype(str).isin(allowed_vals)
                else:
                    row_match = ~series.astype(str).isin(allowed_vals)
            else:
                row_match = pd.Series(True, index=df.index)
        else:  # Continuous
            numeric = pd.to_numeric(series, errors="coerce")
            num_op = rule.get("num_operator", ">=")
            if num_op == ">":
                row_match = numeric > float(rule.get("num_val", 0.0))
            elif num_op == ">=":
                row_match = numeric >= float(rule.get("num_val", 0.0))
            elif num_op == "<":
                row_match = numeric < float(rule.get("num_val", 0.0))
            elif num_op == "<=":
                row_match = numeric <= float(rule.get("num_val", 0.0))
            elif num_op == "==":
                row_match = numeric == float(rule.get("num_val", 0.0))
            elif num_op == "!=":
                row_match = numeric != float(rule.get("num_val", 0.0))
            elif num_op == "Between":
                min_v = float(rule.get("num_val_min", 0.0))
                max_v = float(rule.get("num_val_max", 0.0))
                row_match = (numeric >= min_v) & (numeric <= max_v)
            else:
                row_match = pd.Series(True, index=df.index)

        mask &= (row_match | is_missing) if keep_missing else (row_match & ~is_missing)

    return df.loc[mask].copy()



def render_proportion(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
    output_dir_default: str,
) -> None:
    st.markdown("### Proportion")
    st.caption("Phenotype proportion comparison across methods (Step8 cell 5).")

    # Nothing is read here. Each control below asks for the one or two columns
    # it needs, at the moment it needs them. Loading the whole table up front
    # cost minutes on a large file before the user had chosen anything.
    def _columns(names: list[str]) -> pd.DataFrame | None:
        if not unified_csv_path or not os.path.exists(unified_csv_path):
            return None
        try:
            frame = get_columns_df(unified_csv_path, names)
        except Exception:
            return None
        return frame if not frame.empty else None

    notebook_methods = extract_list_from_notebook(["columns_to_process", "PHENOTYPING_METHODS"])
    if "viz_prop_methods" not in st.session_state or st.session_state.get("viz_prop_last_file") != unified_csv_path:
        defaults = [method for method in notebook_methods if method in available_columns]
        if not defaults:
            defaults = [col for col in available_columns if col.startswith("phenotype_")]
        st.session_state["viz_prop_methods"] = defaults
        st.session_state["viz_prop_methods_select"] = defaults
        st.session_state["viz_prop_last_file"] = unified_csv_path

    viz_prop_options = [col for col in available_columns if col.startswith("phenotype_")] or available_columns
    st.session_state["viz_prop_methods_select"] = [
        method
        for method in st.session_state.get("viz_prop_methods_select", st.session_state.get("viz_prop_methods", []))
        if method in viz_prop_options
    ]
    methods = st.multiselect(
        "Phenotype Columns",
        options=viz_prop_options,
        key="viz_prop_methods_select",
    )
    st.session_state["viz_prop_methods"] = list(methods)
    phenotype_color_map = render_phenotype_color_controls(
        category_labels_for_columns(_columns(list(methods)), methods),
        panel_key="tab9_proportion",
        title="Phenotype colors",
    )
    if methods:
        st.caption(
            "Bar fill identifies the phenotype; position and hatch distinguish the selected phenotyping columns."
        )

    with st.form("viz_prop_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            dpi_val = st.number_input(
                "Figure DPI",
                min_value=100,
                max_value=600,
                value=int(cached_params.get("viz_prop_dpi", 300)),
                step=50,
                key="viz_prop_dpi",
            )
        with col2:
            fig_width = st.number_input(
                "Figure Width",
                min_value=6.0,
                max_value=30.0,
                value=float(cached_params.get("viz_prop_width", 22.0)),
                step=1.0,
                key="viz_prop_width",
            )
        with col3:
            fig_height = st.number_input(
                "Figure Height",
                min_value=4.0,
                max_value=15.0,
                value=float(cached_params.get("viz_prop_height", 8.0)),
                step=0.5,
                key="viz_prop_height",
            )

        output_dir = st.text_input(
            "Output Directory",
            value=output_dir_default,
            key="viz_prop_output_dir",
        )

        # ── Cell Filtration Box ──────────────────────────────────────
        saved_filters = cached_params.get("viz_prop_filters", [])
        with st.expander("🔍 Cell Filtration Options", expanded=bool(saved_filters)):
            st.caption("Filter cells used for proportion calculations by categorical or continuous column criteria.")
            num_filters = st.number_input(
                "Number of Cell Filters",
                min_value=0,
                max_value=10,
                value=int(cached_params.get("viz_prop_num_filters", len(saved_filters))),
                step=1,
                key="viz_prop_num_filters_input",
            )

            filter_rules: list[dict[str, Any]] = []
            for i in range(int(num_filters)):
                st.markdown(f"**Filter {i + 1}**")
                prev_rule = saved_filters[i] if i < len(saved_filters) else {}

                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    prev_col = prev_rule.get("column", "")
                    default_col_idx = available_columns.index(prev_col) if prev_col in available_columns else 0
                    sel_col = st.selectbox(
                        "Column",
                        options=available_columns,
                        index=default_col_idx,
                        key=f"viz_prop_f{i}_col",
                    )

                is_numeric_col = False
                _sel_frame = _columns([sel_col]) if sel_col else None
                if _sel_frame is not None and sel_col in _sel_frame.columns:
                    is_numeric_col = pd.api.types.is_numeric_dtype(_sel_frame[sel_col])

                default_type_idx = 1 if is_numeric_col else 0
                prev_type = prev_rule.get("type", "Continuous" if is_numeric_col else "Categorical")
                if prev_type in ["Categorical", "Continuous"]:
                    default_type_idx = ["Categorical", "Continuous"].index(prev_type)

                with f_col2:
                    sel_type = st.selectbox(
                        "Filter Type",
                        options=["Categorical", "Continuous"],
                        index=default_type_idx,
                        key=f"viz_prop_f{i}_type",
                    )

                rule_dict: dict[str, Any] = {
                    "column": sel_col,
                    "type": sel_type,
                }

                if sel_type == "Categorical":
                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        cat_op = st.selectbox(
                            "Criteria",
                            options=["Is in (Include)", "Is not in (Exclude)"],
                            index=0 if prev_rule.get("cat_operator") != "Is not in (Exclude)" else 1,
                            key=f"viz_prop_f{i}_cat_op",
                        )
                        rule_dict["cat_operator"] = cat_op

                    unique_options: list[str] = []
                    _opt_frame = _columns([sel_col]) if sel_col else None
                    if _opt_frame is not None and sel_col in _opt_frame.columns:
                        unique_options = sorted([str(v) for v in _opt_frame[sel_col].dropna().unique()])

                    prev_vals = [str(v) for v in prev_rule.get("cat_values", [])]
                    valid_defaults = [v for v in prev_vals if v in unique_options] if unique_options else prev_vals

                    with c_col2:
                        sel_vals = st.multiselect(
                            "Select Variable(s)",
                            options=unique_options if unique_options else valid_defaults,
                            default=valid_defaults,
                            key=f"viz_prop_f{i}_cat_vals",
                        )
                        rule_dict["cat_values"] = sel_vals
                else:  # Continuous
                    n_col1, n_col2 = st.columns(2)
                    op_options = [">=", ">", "<=", "<", "==", "!=", "Between"]
                    prev_num_op = prev_rule.get("num_operator", ">=")
                    default_op_idx = op_options.index(prev_num_op) if prev_num_op in op_options else 0

                    with n_col1:
                        num_op = st.selectbox(
                            "Criteria",
                            options=op_options,
                            index=default_op_idx,
                            key=f"viz_prop_f{i}_num_op",
                        )
                        rule_dict["num_operator"] = num_op

                    with n_col2:
                        if num_op == "Between":
                            b_col1, b_col2 = st.columns(2)
                            with b_col1:
                                min_val = st.number_input(
                                    "Min Value",
                                    value=float(prev_rule.get("num_val_min", 0.0)),
                                    key=f"viz_prop_f{i}_num_min",
                                )
                                rule_dict["num_val_min"] = min_val
                            with b_col2:
                                max_val = st.number_input(
                                    "Max Value",
                                    value=float(prev_rule.get("num_val_max", 1.0)),
                                    key=f"viz_prop_f{i}_num_max",
                                )
                                rule_dict["num_val_max"] = max_val
                        else:
                            val = st.number_input(
                                "Threshold Value",
                                value=float(prev_rule.get("num_val", 0.0)),
                                key=f"viz_prop_f{i}_num_val",
                            )
                            rule_dict["num_val"] = val

                inc_miss = st.checkbox(
                    "Include missing / NaN values",
                    value=bool(prev_rule.get("include_missing", False)),
                    key=f"viz_prop_f{i}_inc_miss",
                )
                rule_dict["include_missing"] = inc_miss

                filter_rules.append(rule_dict)

        save_cfg = st.form_submit_button("Save Configuration", type="primary")
        if save_cfg:
            st.session_state["viz_prop_methods"] = methods
            st.session_state["viz_prop_filters"] = filter_rules
            cached_params.update(
                {
                    "viz_prop_methods": methods,
                    "viz_prop_dpi": dpi_val,
                    "viz_prop_width": fig_width,
                    "viz_prop_height": fig_height,
                    "viz_output_dir": output_dir,
                    "viz_prop_num_filters": num_filters,
                    "viz_prop_filters": filter_rules,
                }
            )
            save_cached_params(project_root, cached_params)
            st.success("Configuration saved.")

    if st.button("Generate Proportion Figure", type="primary", key="viz_prop_run"):
        log_lines: list[str] = []

        def _log(message: str) -> None:
            line = f"[{timestamp()}] {message}"
            log_lines.append(line)
            log_console(line)
            logger.info(line)

        if not unified_csv_path or not os.path.exists(unified_csv_path):
            st.error("Unified CSV file not found.")
        elif not st.session_state.get("viz_prop_methods"):
            st.error("Select at least one phenotype column.")
        else:
            _log("Starting proportion analysis")
            methods = st.session_state.get("viz_prop_methods", [])
            filters = st.session_state.get("viz_prop_filters", cached_params.get("viz_prop_filters", []))
            # Read the phenotype columns plus any column a filter refers to,
            # and nothing else.
            needed = list(methods) + [
                str(rule.get("column"))
                for rule in (filters or [])
                if isinstance(rule, dict) and rule.get("column")
            ]
            df = get_columns_df(unified_csv_path, needed).copy()
            initial_count = len(df)
            _log(f"Loaded {len(df.columns)} column(s), {initial_count:,} rows")

            if filters:
                df = apply_cell_filters(df, filters)
                _log(f"Applied {len(filters)} cell filter(s): {initial_count:,} -> {len(df):,} cells remaining")

            records: list[dict[str, Any]] = []

            for method in methods:
                if method not in df.columns:
                    continue
                value_counts = df[method].value_counts(dropna=False)
                total = int(value_counts.sum())
                _log(f"Method '{method}' breakdown (Total: {total:,} cells):")
                for phenotype, count in value_counts.items():
                    if pd.isna(phenotype):
                        continue
                    proportion = count / total if total else 0
                    _log(f"  - {phenotype}: {count:,} cells ({proportion:.2%})")
                    records.append(
                        {
                            "method": method,
                            "phenotype": str(phenotype),
                            "count": int(count),
                            "total_cells": total,
                            "proportion": proportion,
                        }
                    )
            summary_long = pd.DataFrame(records)
            if summary_long.empty:
                st.warning("No phenotypes found for the selected columns.")
                _log("No phenotypes found for selected columns")
            else:
                phenotype_order = (
                    summary_long.groupby("phenotype")["proportion"]
                    .mean()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                output_dir = cached_params.get("viz_output_dir", output_dir_default)
                os.makedirs(output_dir, exist_ok=True)
                long_path = Path(output_dir) / "phenotype_proportions_long.csv"
                pivot_path = Path(output_dir) / "phenotype_proportions_pivot.csv"
                summary_long.to_csv(long_path, index=False)
                pivot_prop = summary_long.pivot(index="phenotype", columns="method", values="proportion").fillna(0)
                pivot_prop.to_csv(pivot_path)
                _log(f"Saved tables to {output_dir}")

                active_phenotype_colors = shared_phenotype_colors(phenotype_order)
                active_phenotype_colors.update(
                    {
                        label: color
                        for label, color in phenotype_color_map.items()
                        if label in phenotype_order
                    }
                )
                fig = _make_proportion_figure(
                    summary_long,
                    methods=list(methods),
                    phenotype_order=phenotype_order,
                    phenotype_colors=active_phenotype_colors,
                    figsize=(
                        float(cached_params.get("viz_prop_width", 22.0)),
                        float(cached_params.get("viz_prop_height", 8.0)),
                    ),
                )
                fig_path = Path(output_dir) / "phenotype_proportions_grouped.png"
                fig.savefig(fig_path, dpi=int(cached_params.get("viz_prop_dpi", 300)))
                st.pyplot(fig)
                st.success(f"Saved figures and tables to {output_dir}")
                _log(f"Saved figure: {fig_path}")
                plt.close(fig)

        if log_lines:
            st.markdown("#### Proportion Log")
            st.code("\n".join(log_lines))
