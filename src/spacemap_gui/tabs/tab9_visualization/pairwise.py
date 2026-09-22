"""Pairwise visualization subtab."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from .common import (
    extract_list_from_notebook,
    get_cached_unified_df,
    log_console,
    logger,
    save_cached_params,
    timestamp,
)
from .f1_validation_utils import parse_mapping_text, prettify_column_name

DEFAULT_TITLE = "Pairwise Method Similarity (% of matching phenotypes)"
DEFAULT_CBAR_LABEL = "Similarity Percentage (%)"


def _default_label_text(columns: Sequence[str]) -> str:
    """Prefill the rename box with one ``column = Label`` line per column."""

    return "\n".join(f"{column} = {prettify_column_name(column)}" for column in columns)


def _resolve_labels(columns: Sequence[str], text: str) -> dict[str, str]:
    """Map each column to its figure label.

    Accepts both orientations (``phenotype_leiden = Leiden`` and
    ``Leiden = phenotype_leiden``); columns without an entry fall back to the
    standard SPACEMAP short name. Duplicate labels get a numeric suffix so the
    heatmap axes stay unambiguous.
    """

    known = set(columns)
    resolved: dict[str, str] = {}
    for left, right in parse_mapping_text(text).items():
        if left in known:
            resolved[left] = right
        elif right in known:
            resolved[right] = left

    labels: dict[str, str] = {}
    used: dict[str, int] = {}
    for column in columns:
        label = resolved.get(column) or prettify_column_name(column)
        if label in used:
            used[label] += 1
            label = f"{label} ({used[label]})"
        else:
            used[label] = 1
        labels[column] = label
    return labels


def _reset_pair_labels() -> None:
    selected = st.session_state.get("viz_pair_methods_select") or st.session_state.get("viz_pair_methods", [])
    st.session_state["viz_pair_label_text"] = _default_label_text(selected)


def _param(cached_params: dict[str, Any], key: str, default: Any) -> Any:
    """Prefer the last submitted widget value, falling back to saved config."""

    if key in st.session_state:
        return st.session_state[key]
    return cached_params.get(key, default)


def render_pairwise(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
    output_dir_default: str,
) -> None:
    st.markdown("### Pairwise")
    st.caption("Pairwise phenotype agreement heatmap (Step8 cell 7).")

    notebook_methods = extract_list_from_notebook(["METHODS_TO_COMPARE", "PHENOTYPING_METHODS"])
    if "viz_pair_methods" not in st.session_state or st.session_state.get("viz_pair_last_file") != unified_csv_path:
        defaults = [method for method in notebook_methods if method in available_columns]
        if not defaults:
            defaults = [col for col in available_columns if col.startswith("phenotype_")]
        st.session_state["viz_pair_methods"] = defaults
        st.session_state["viz_pair_last_file"] = unified_csv_path

    with st.form("viz_pair_form", clear_on_submit=False):
        viz_pair_options = [col for col in available_columns if col.startswith("phenotype_")] or available_columns
        methods = st.multiselect(
            "Phenotype Columns",
            options=viz_pair_options,
            default=[method for method in st.session_state.get("viz_pair_methods", []) if method in viz_pair_options],
            key="viz_pair_methods_select",
        )
        label_text = st.text_area(
            "Figure Labels (one `column = Label` per line)",
            value=str(cached_params.get("viz_pair_label_text", "") or _default_label_text(methods)),
            key="viz_pair_label_text",
            height=130,
            help=(
                "Rename columns for the figure axes, e.g. `phenotype_leiden = Leiden`. "
                "The reversed form (`Leiden = phenotype_leiden`) is also accepted. "
                "Columns without a line use the standard short name. "
                "Use 'Reset Labels' after changing the column selection."
            ),
        )
        dpi_val = st.number_input(
            "Figure DPI",
            min_value=100,
            max_value=600,
            value=int(cached_params.get("viz_pair_dpi", 300)),
            step=50,
            key="viz_pair_dpi",
        )

        with st.expander("Figure Layout, Fonts & Tick Labels", expanded=True):
            size_cols = st.columns(3)
            with size_cols[0]:
                fig_width = st.number_input(
                    "Figure Width (in)",
                    min_value=2.0,
                    max_value=40.0,
                    value=float(cached_params.get("viz_pair_fig_width", 10.0)),
                    step=0.5,
                    key="viz_pair_fig_width",
                )
            with size_cols[1]:
                fig_height = st.number_input(
                    "Figure Height (in)",
                    min_value=2.0,
                    max_value=40.0,
                    value=float(cached_params.get("viz_pair_fig_height", 8.0)),
                    step=0.5,
                    key="viz_pair_fig_height",
                )
            with size_cols[2]:
                square_cells = st.checkbox(
                    "Square Cells",
                    value=bool(cached_params.get("viz_pair_square_cells", False)),
                    key="viz_pair_square_cells",
                    help="Force each heatmap cell to be a square regardless of figure size.",
                )

            title_text = st.text_input(
                "Figure Title (leave empty to hide)",
                value=str(cached_params.get("viz_pair_title", DEFAULT_TITLE)),
                key="viz_pair_title",
            )
            cbar_label = st.text_input(
                "Colorbar Label",
                value=str(cached_params.get("viz_pair_cbar_label", DEFAULT_CBAR_LABEL)),
                key="viz_pair_cbar_label",
            )

            font_cols = st.columns(4)
            with font_cols[0]:
                title_size = st.number_input(
                    "Title Size",
                    min_value=4.0,
                    max_value=48.0,
                    value=float(cached_params.get("viz_pair_title_size", 12.0)),
                    step=1.0,
                    key="viz_pair_title_size",
                )
            with font_cols[1]:
                tick_size = st.number_input(
                    "Tick Label Size",
                    min_value=4.0,
                    max_value=36.0,
                    value=float(cached_params.get("viz_pair_tick_size", 10.0)),
                    step=1.0,
                    key="viz_pair_tick_size",
                )
            with font_cols[2]:
                annot_size = st.number_input(
                    "Cell Value Size",
                    min_value=4.0,
                    max_value=36.0,
                    value=float(cached_params.get("viz_pair_annot_size", 12.0)),
                    step=1.0,
                    key="viz_pair_annot_size",
                )
            with font_cols[3]:
                cbar_size = st.number_input(
                    "Colorbar Text Size",
                    min_value=4.0,
                    max_value=36.0,
                    value=float(cached_params.get("viz_pair_cbar_size", 10.0)),
                    step=1.0,
                    key="viz_pair_cbar_size",
                )

            rot_cols = st.columns(3)
            with rot_cols[0]:
                x_rotation = st.number_input(
                    "X Label Rotation (deg)",
                    min_value=-180,
                    max_value=180,
                    value=int(cached_params.get("viz_pair_x_rotation", 0)),
                    step=15,
                    key="viz_pair_x_rotation",
                )
            with rot_cols[1]:
                y_rotation = st.number_input(
                    "Y Label Rotation (deg)",
                    min_value=-180,
                    max_value=180,
                    value=int(cached_params.get("viz_pair_y_rotation", 0)),
                    step=15,
                    key="viz_pair_y_rotation",
                )
            with rot_cols[2]:
                decimals = st.number_input(
                    "Value Decimals",
                    min_value=0,
                    max_value=3,
                    value=int(cached_params.get("viz_pair_decimals", 1)),
                    step=1,
                    key="viz_pair_decimals",
                )

            cell_cols = st.columns(4)
            with cell_cols[0]:
                show_annot = st.checkbox(
                    "Show Values",
                    value=bool(cached_params.get("viz_pair_show_annot", True)),
                    key="viz_pair_show_annot",
                )
            with cell_cols[1]:
                show_cbar = st.checkbox(
                    "Show Colorbar",
                    value=bool(cached_params.get("viz_pair_show_cbar", True)),
                    key="viz_pair_show_cbar",
                )
            with cell_cols[2]:
                lower_only = st.checkbox(
                    "Lower Triangle Only",
                    value=bool(cached_params.get("viz_pair_lower_only", True)),
                    key="viz_pair_lower_only",
                )
            with cell_cols[3]:
                line_width = st.number_input(
                    "Cell Line Width",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(cached_params.get("viz_pair_linewidth", 0.5)),
                    step=0.1,
                    key="viz_pair_linewidth",
                )

        st.markdown("**Heatmap Color Scale**")
        color_range = st.slider(
            "Color Scale Range (%)",
            min_value=0.0,
            max_value=100.0,
            value=(
                float(cached_params.get("viz_pair_vmin", 0.0)),
                float(cached_params.get("viz_pair_vmax", 100.0))
            ),
            step=1.0,
            help="Adjust the minimum and maximum percentages of the color scale to enhance contrast.",
            key="viz_pair_color_range",
        )

        output_dir = st.text_input(
            "Output Directory",
            value=output_dir_default,
            key="viz_pair_output_dir",
        )

        button_cols = st.columns([1, 1])
        with button_cols[0]:
            save_cfg = st.form_submit_button("Save Configuration", type="primary")
        with button_cols[1]:
            st.form_submit_button("Reset Labels", on_click=_reset_pair_labels)
        if save_cfg:
            st.session_state["viz_pair_methods"] = methods
            cached_params.update(
                {
                    "viz_pair_methods": methods,
                    "viz_pair_dpi": dpi_val,
                    "viz_output_dir": output_dir,
                    "viz_pair_excluded_phenotypes": st.session_state.get("viz_pair_excluded_phenotypes", []),
                    "viz_pair_vmin": float(color_range[0]),
                    "viz_pair_vmax": float(color_range[1]),
                    "viz_pair_label_text": label_text,
                    "viz_pair_fig_width": float(fig_width),
                    "viz_pair_fig_height": float(fig_height),
                    "viz_pair_square_cells": bool(square_cells),
                    "viz_pair_title": title_text,
                    "viz_pair_cbar_label": cbar_label,
                    "viz_pair_title_size": float(title_size),
                    "viz_pair_tick_size": float(tick_size),
                    "viz_pair_annot_size": float(annot_size),
                    "viz_pair_cbar_size": float(cbar_size),
                    "viz_pair_x_rotation": int(x_rotation),
                    "viz_pair_y_rotation": int(y_rotation),
                    "viz_pair_decimals": int(decimals),
                    "viz_pair_show_annot": bool(show_annot),
                    "viz_pair_show_cbar": bool(show_cbar),
                    "viz_pair_lower_only": bool(lower_only),
                    "viz_pair_linewidth": float(line_width),
                }
            )
            save_cached_params(project_root, cached_params)
            st.success("Configuration saved.")

    # --- Phenotype exclusion (outside form so we can read data for options) ---
    _excl_options: list[str] = []
    if unified_csv_path and os.path.exists(unified_csv_path):
        try:
            # Only the chosen phenotype columns are needed to list their values.
            _sel_cols = list(st.session_state.get("viz_pair_methods", []))
            _df_preview = (
                get_cached_unified_df(unified_csv_path, columns=_sel_cols)
                if _sel_cols
                else None
            )
            _sel_cols = [c for c in _sel_cols if _df_preview is not None and c in _df_preview.columns]
            if _sel_cols:
                _excl_options = sorted(
                    set().union(*[
                        set(_df_preview[c].dropna().astype(str).unique())
                        for c in _sel_cols
                    ])
                )
        except Exception:
            pass

    _excl_default = [v for v in cached_params.get("viz_pair_excluded_phenotypes", []) if v in _excl_options]
    if "viz_pair_excluded_phenotypes" not in st.session_state:
        st.session_state["viz_pair_excluded_phenotypes"] = _excl_default

    st.multiselect(
        "Exclude Phenotypes from Comparison",
        options=_excl_options,
        key="viz_pair_excluded_phenotypes",
        help=(
            "Cells assigned any of these phenotype labels will be dropped from all columns "
            "before computing pairwise similarity. Useful for removing catch-all labels "
            "such as 'Others' or 'unclassified' that would inflate agreement scores."
        ),
    )
    if st.session_state.get("viz_pair_excluded_phenotypes"):
        st.caption(f"Excluding: {', '.join(st.session_state['viz_pair_excluded_phenotypes'])}")

    if st.button("Generate Pairwise Heatmap", type="primary", key="viz_pair_run"):
        log_lines: list[str] = []

        def _log(message: str) -> None:
            line = f"[{timestamp()}] {message}"
            log_lines.append(line)
            log_console(line)
            logger.info(line)

        if not unified_csv_path or not os.path.exists(unified_csv_path):
            st.error("Unified CSV file not found.")
        elif not st.session_state.get("viz_pair_methods"):
            st.error("Select at least one phenotype column.")
        else:
            _log("Starting pairwise similarity analysis")
            methods = st.session_state.get("viz_pair_methods", [])
            # The next line used to read all 133 columns and then keep these.
            df = get_cached_unified_df(unified_csv_path, columns=list(methods)).copy()
            _log(f"Loaded {len(df.columns)} column(s), {len(df):,} rows")
            df = df[[c for c in methods if c in df.columns]]

            # Apply phenotype exclusion filter
            _excluded = st.session_state.get("viz_pair_excluded_phenotypes", [])
            if _excluded:
                _before = len(df)
                _excl_mask = ~df.isin(_excluded).any(axis=1)
                df = df[_excl_mask]
                _log(f"Excluded {_before - len(df):,} rows containing phenotypes: {_excluded} ({len(df):,} rows remain)")
            similarity_matrix = pd.DataFrame(
                np.zeros((len(methods), len(methods))),
                index=methods,
                columns=methods,
            )
            for idx, method_one in enumerate(methods):
                for method_two in methods[idx:]:
                    filtered = df[[method_one, method_two]].dropna()
                    series_one = filtered[method_one]
                    if isinstance(series_one, pd.DataFrame):
                        series_one = series_one.iloc[:, 0]
                    series_two = filtered[method_two]
                    if isinstance(series_two, pd.DataFrame):
                        series_two = series_two.iloc[:, 0]
                    similarity = float(((series_one == series_two).sum() / len(series_one)) * 100) if len(series_one) else 0
                    similarity_matrix.at[method_one, method_two] = similarity
                    similarity_matrix.at[method_two, method_one] = similarity
            _log("Computed similarity matrix")

            output_dir = cached_params.get("viz_output_dir", output_dir_default)
            os.makedirs(output_dir, exist_ok=True)
            matrix_csv_path = Path(output_dir) / "pairwise_similarity_matrix.csv"
            similarity_matrix.to_csv(matrix_csv_path)
            _log(f"Saved matrix: {matrix_csv_path}")

            pair_vmin = float(_param(cached_params, "viz_pair_vmin", 0.0))
            pair_vmax = float(_param(cached_params, "viz_pair_vmax", 100.0))
            _range = _param(cached_params, "viz_pair_color_range", None)
            if isinstance(_range, (list, tuple)) and len(_range) == 2:
                pair_vmin, pair_vmax = float(_range[0]), float(_range[1])

            # Figure labels (CSV keeps the raw column names)
            labels = _resolve_labels(methods, str(_param(cached_params, "viz_pair_label_text", "")))
            plot_matrix = similarity_matrix.copy()
            plot_matrix.index = [labels[method] for method in methods]
            plot_matrix.columns = [labels[method] for method in methods]
            _renamed = [f"{method} -> {labels[method]}" for method in methods if labels[method] != method]
            if _renamed:
                _log(f"Figure labels: {', '.join(_renamed)}")

            fig_width = float(_param(cached_params, "viz_pair_fig_width", 10.0))
            fig_height = float(_param(cached_params, "viz_pair_fig_height", 8.0))
            title_text = str(_param(cached_params, "viz_pair_title", DEFAULT_TITLE))
            cbar_label = str(_param(cached_params, "viz_pair_cbar_label", DEFAULT_CBAR_LABEL))
            title_size = float(_param(cached_params, "viz_pair_title_size", 12.0))
            tick_size = float(_param(cached_params, "viz_pair_tick_size", 10.0))
            annot_size = float(_param(cached_params, "viz_pair_annot_size", 12.0))
            cbar_size = float(_param(cached_params, "viz_pair_cbar_size", 10.0))
            x_rotation = int(_param(cached_params, "viz_pair_x_rotation", 0))
            y_rotation = int(_param(cached_params, "viz_pair_y_rotation", 0))
            decimals = int(_param(cached_params, "viz_pair_decimals", 1))
            show_annot = bool(_param(cached_params, "viz_pair_show_annot", True))
            show_cbar = bool(_param(cached_params, "viz_pair_show_cbar", True))
            lower_only = bool(_param(cached_params, "viz_pair_lower_only", True))
            line_width = float(_param(cached_params, "viz_pair_linewidth", 0.5))
            square_cells = bool(_param(cached_params, "viz_pair_square_cells", False))

            mask = np.triu(np.ones_like(plot_matrix, dtype=bool), k=1) if lower_only else None
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                plot_matrix,
                mask=mask,
                annot=show_annot,
                fmt=f".{decimals}f",
                annot_kws={"size": annot_size},
                cmap="YlGnBu",
                linewidths=line_width,
                vmin=pair_vmin,
                vmax=pair_vmax,
                square=square_cells,
                cbar=show_cbar,
                cbar_kws={"label": cbar_label},
                ax=ax,
            )
            ax.set_title(title_text, fontsize=title_size)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelsize=tick_size, rotation=x_rotation)
            ax.tick_params(axis="y", labelsize=tick_size, rotation=y_rotation)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("center" if x_rotation % 90 == 0 else "right")
            for label in ax.get_yticklabels():
                label.set_verticalalignment("center")
            if show_cbar:
                colorbar = ax.collections[0].colorbar
                if colorbar is not None:
                    colorbar.ax.tick_params(labelsize=cbar_size)
                    colorbar.set_label(cbar_label, fontsize=cbar_size)
            fig.tight_layout()
            heatmap_path = Path(output_dir) / "pairwise_similarity_heatmap.png"
            fig.savefig(heatmap_path, dpi=int(_param(cached_params, "viz_pair_dpi", 300)))
            st.pyplot(fig)
            st.success(f"Saved heatmap and CSV to {output_dir}")
            _log(f"Saved heatmap: {heatmap_path}")
            plt.close(fig)

        if log_lines:
            st.markdown("#### Pairwise Log")
            st.code("\n".join(log_lines))
