"""Visual validation subtab."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from cell_visualizer import CellVisualizer
except ImportError:
    try:
        from ...cell_visualizer import CellVisualizer
    except Exception:
        CellVisualizer = None

from .common import _extract_markers_from_notebook, get_cached_unified_df, get_loaded_visualization_df, logger


def render_visual_validation(
    project_root: Path,
    unified_csv_path: str | None,
    available_columns: list[str],
    cached_params: dict[str, Any],
    viz_settings: dict[str, Any],
) -> None:
    st.markdown("### Visual Validation")
    st.caption("Inspect random cells from specific phenotypes to validate assignments.")

    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.error("Unified CSV file not found. Please ensure global input path is set.")
        return

    st.markdown("##### Marker Configuration")
    notebook_corr_markers, notebook_non_corr_markers = _extract_markers_from_notebook("Step6A_phenotype_leiden.ipynb")

    df_viz_val = get_loaded_visualization_df(unified_csv_path)
    if isinstance(df_viz_val, pd.DataFrame):
        available_cols = list(df_viz_val.columns)
    else:
        available_cols = available_columns
        df_viz_val = pd.DataFrame()
        st.info("Load the unified CSV once using 'Unified CSV Loader' above to enable Visual Validation preprocessing.")

    metadata_cols = [
        "cell_id",
        "batch_id",
        "tissue",
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
    ]
    metadata_lower = [meta.lower() for meta in metadata_cols]
    all_marker_candidates = [
        col
        for col in available_cols
        if col.lower() not in metadata_lower
        and not col.startswith("phenotype_")
        and not col.startswith("umap_")
        and not col.startswith("leiden_")
        and not col.endswith("_norm")
    ]

    expected_markers: list[str] = []
    for marker in notebook_corr_markers:
        corr_col = f"{marker}_corr"
        if corr_col in all_marker_candidates:
            expected_markers.append(corr_col)
    for marker in notebook_non_corr_markers:
        if marker in all_marker_candidates:
            expected_markers.append(marker)

    if "viz_val_confirmed_markers" not in st.session_state or st.session_state.get("viz_val_last_file") != unified_csv_path:
        st.session_state["viz_val_confirmed_markers"] = expected_markers
        st.session_state["viz_val_last_file"] = unified_csv_path
        if expected_markers:
            st.info(f"Auto-selected {len(expected_markers)} markers from notebook configuration")

    pending_marker_selection = st.session_state.pop("viz_val_marker_pending_selection", None)
    if isinstance(pending_marker_selection, list):
        st.session_state["viz_val_confirmed_markers"] = pending_marker_selection
        st.session_state["viz_val_marker_multiselect"] = pending_marker_selection

    with st.form("viz_val_marker_form", clear_on_submit=False):
        st.markdown("**Available Markers** (auto-selected from notebook)")
        col1, col2, col3 = st.columns(3)
        with col1:
            select_all_corr = st.form_submit_button("Select all _corr markers")
        with col2:
            select_all_all = st.form_submit_button("Select all markers")
        with col3:
            clear_all = st.form_submit_button("Clear all")

        current_selection = [marker for marker in st.session_state.get("viz_val_confirmed_markers", []) if marker in all_marker_candidates]
        if "viz_val_marker_multiselect" not in st.session_state:
            st.session_state["viz_val_marker_multiselect"] = current_selection
        else:
            st.session_state["viz_val_marker_multiselect"] = [
                marker for marker in st.session_state.get("viz_val_marker_multiselect", []) if marker in all_marker_candidates
            ]

        selected_markers = st.multiselect(
            "Select Markers",
            all_marker_candidates,
            help="Choose markers for validation. Can mix _corr and non-_corr versions. Auto-selected from notebook.",
            key="viz_val_marker_multiselect",
        )

        save_markers_btn = st.form_submit_button("Save Marker Selection", type="primary")
        if save_markers_btn:
            st.session_state["viz_val_confirmed_markers"] = selected_markers
            st.success(f"Saved {len(selected_markers)} markers")

        if select_all_corr:
            corr_markers = [marker for marker in all_marker_candidates if marker.endswith("_corr")]
            current_selected = st.session_state.get("viz_val_marker_multiselect", current_selection)
            current_selected_set = set(current_selected if isinstance(current_selected, list) else [])
            merged_selection = [marker for marker in all_marker_candidates if marker in current_selected_set or marker in corr_markers]
            st.session_state["viz_val_marker_pending_selection"] = merged_selection
            st.session_state["viz_val_confirmed_markers"] = merged_selection
            st.rerun()

        if select_all_all:
            marker_set = set(all_marker_candidates)
            paired_base_markers: list[str] = []
            for marker in all_marker_candidates:
                if marker.endswith("_corr"):
                    base_marker = marker[:-5]
                    if base_marker in marker_set:
                        paired_base_markers.append(base_marker)
            paired_base_markers = list(dict.fromkeys(paired_base_markers))
            st.session_state["viz_val_marker_pending_selection"] = paired_base_markers
            st.session_state["viz_val_confirmed_markers"] = paired_base_markers
            st.rerun()

        if clear_all:
            st.session_state["viz_val_marker_pending_selection"] = []
            st.session_state["viz_val_confirmed_markers"] = []
            st.rerun()

    markers_final = st.session_state.get("viz_val_confirmed_markers", [])
    st.caption(f"Confirmed markers: {len(markers_final)}")

    st.divider()
    st.markdown("##### Data Loading & Normalization")
    norm_col1, norm_col2 = st.columns(2)
    with norm_col1:
        enable_znorm = st.checkbox(
            "Enable Z-Score Normalization",
            value=True,
            help="Normalize marker expression per batch.",
            key="viz_val_enable_znorm",
        )
    with norm_col2:
        batch_candidates = [col for col in available_cols if "batch" in col.lower()] or [viz_settings.get("batch_col", "batch_id")]
        default_batch = viz_settings.get("batch_col", "batch_id")
        default_idx = batch_candidates.index(default_batch) if default_batch in batch_candidates else 0
        batch_col_vv = st.selectbox(
            "Batch Column",
            batch_candidates,
            index=default_idx,
            key="viz_val_batch_col",
        )

    if st.button("Load & Normalize Data", type="primary", key="viz_val_load_btn"):
        if not markers_final:
            st.error("Please save marker selection first.")
        else:
            try:
                with st.spinner("Loading and normalizing data..."):
                    # Work out the wanted columns from the header, then read
                    # only those, instead of reading all of them and slicing.
                    from .common import get_csv_columns_fast

                    all_columns = get_csv_columns_fast(unified_csv_path)
                    meta_needed = ["cell_id", "x_centroid", "y_centroid", batch_col_vv]
                    cols_to_keep = markers_final + meta_needed
                    cols_to_keep += [col for col in all_columns if col.startswith("phenotype_")]
                    cols_to_keep = list(dict.fromkeys([col for col in cols_to_keep if col in all_columns]))

                    viz_df = get_cached_unified_df(unified_csv_path, columns=cols_to_keep).copy()
                    for marker in markers_final:
                        if marker in viz_df.columns:
                            viz_df[marker] = pd.to_numeric(viz_df[marker], errors="coerce").fillna(0)

                    batch_stats: dict[str, Any] = {}
                    if enable_znorm:
                        if batch_col_vv not in viz_df.columns:
                            st.warning(f"Batch column {batch_col_vv} not found. Skipping normalization.")
                        else:
                            grouped = viz_df.groupby(batch_col_vv)[markers_final]
                            means = grouped.mean()
                            stds = grouped.std().replace(0, 1)
                            batch_stats["means"] = means.to_dict(orient="index")
                            batch_stats["stds"] = stds.to_dict(orient="index")

                    st.session_state["viz_val_data_ready"] = True
                    st.session_state["viz_val_cached_df"] = viz_df
                    st.session_state["viz_val_batch_stats"] = batch_stats
                    st.session_state["viz_val_loaded_markers"] = markers_final
                    st.session_state["viz_val_loaded_batch_col"] = batch_col_vv

                st.success("Data loaded and normalized successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Error loading data: {exc}")
                logger.error(f"VizVal Load Error: {exc}")

    if not st.session_state.get("viz_val_data_ready", False):
        st.info("Please load and normalize data to proceed.")
        return

    st.divider()
    st.markdown("##### Visualization Target")

    viz_df = st.session_state["viz_val_cached_df"]
    # Only the phenotype columns are read from here on.
    from .common import get_csv_columns_fast

    _phenotype_cols = [c for c in get_csv_columns_fast(unified_csv_path) if c.startswith("phenotype_")]
    full_df = get_cached_unified_df(unified_csv_path, columns=_phenotype_cols)
    loaded_markers = st.session_state.get("viz_val_loaded_markers", [])
    if loaded_markers != markers_final:
        st.warning("Marker selection has changed. Please reload data above.")
        return

    phenotype_first_cols = [
        col for col in full_df.columns if col.lower().startswith("phenotype_") or col.lower().endswith("_phenotype")
    ]
    remaining_cols = [col for col in full_df.columns if col not in phenotype_first_cols]
    pheno_cols = phenotype_first_cols + remaining_cols

    col1, col2 = st.columns(2)
    with col1:
        if pheno_cols:
            selected_pheno_col = st.selectbox("Phenotype Column", pheno_cols, index=0, key="viz_val_pheno_col")
            st.caption("Columns prefixed with 'phenotype_' or suffixed with '_phenotype' are shown first.")
        else:
            selected_pheno_col = None
            st.warning("No columns available for phenotype selection.")
    with col2:
        if selected_pheno_col in full_df.columns:
            unique_phenos = sorted(full_df[selected_pheno_col].dropna().astype(str).unique().tolist())
            selected_phenotype = st.selectbox("Select Phenotype Class", unique_phenos, key="viz_val_target_pheno")
        else:
            selected_phenotype = None

    default_n_cells = int(viz_settings.get("start_n_cells", 5))
    n_view = st.number_input("Number of Random Cells", 1, 50, default_n_cells, key="viz_val_n_cells")
    batch_col_loaded = st.session_state.get("viz_val_loaded_batch_col", batch_col_vv)
    available_viz_batches: list[str] = []
    if batch_col_loaded in viz_df.columns:
        available_viz_batches = sorted(viz_df[batch_col_loaded].dropna().astype(str).unique().tolist())
    viz_batch_scope = st.radio(
        "Batch Scope",
        ["All matched batches", "Specific batch"],
        horizontal=True,
        key="viz_val_batch_scope",
    )
    selected_viz_batch = None
    if viz_batch_scope == "Specific batch" and available_viz_batches:
        selected_viz_batch = st.selectbox("Batch", available_viz_batches, key="viz_val_batch_sel")

    st.markdown("##### Export Settings")
    default_out = project_root / "outputs" / "visual_validation"
    export_dir_str = st.text_input("Output Directory", str(default_out), key="viz_val_out_dir")

    if st.button("Visualize & Save", type="primary", key="viz_val_run"):
        if CellVisualizer is None:
            st.error("CellVisualizer is not available in this environment.")
            return
        if selected_pheno_col not in full_df.columns:
            st.error("Selected phenotype column is not available in the unified CSV.")
            subset_df = pd.DataFrame()
        else:
            filter_mask = full_df[selected_pheno_col].astype(str) == str(selected_phenotype)
            subset_df = viz_df.loc[filter_mask]

        if len(subset_df) == 0:
            st.warning("No cells found for this phenotype.")
            return

        timestamp_int = int(time.time())
        export_dir = Path(export_dir_str)
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            st.error(f"Could not create output directory: {exc}")
            export_dir = None

        viz_roi = viz_settings.get("roi_dir", "")
        viz_seg = viz_settings.get("seg_dir", "")
        tile_size = int(viz_settings.get("tile_size", 40))
        if not viz_roi or not viz_seg or not os.path.exists(viz_roi):
            st.error("ROI or GeoJSON directories are not configured correctly in Visual Validation Settings.")
            return

        try:
            viz = CellVisualizer(viz_roi, viz_seg)
            batch_col = st.session_state.get("viz_val_loaded_batch_col", "batch_id")
            if not batch_col or batch_col not in subset_df.columns:
                batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in subset_df.columns), 'batch_id')
            
            if batch_col not in subset_df.columns:
                st.error(f"Batch column '{batch_col}' not found in results.")
                return
            
            unique_batches = subset_df[batch_col].unique().astype(str).tolist()
            matches = viz.match_file_triads(unique_batches)

            valid_subset = subset_df[subset_df[batch_col].astype(str).isin(matches.keys())]
            if viz_batch_scope == "Specific batch" and selected_viz_batch:
                valid_subset = valid_subset[valid_subset[batch_col].astype(str) == str(selected_viz_batch)]

            if len(valid_subset) == 0:
                st.warning("No matching images for these batches.")
                return

            n_actual = min(n_view, len(valid_subset))
            random_sample = valid_subset.sample(n=n_actual, random_state=timestamp_int).sort_values(batch_col)
            sampled_batches = random_sample[batch_col].astype(str).drop_duplicates().tolist()
            prewarm_fn = getattr(viz, "prewarm_batch", None)
            if callable(prewarm_fn):
                with st.spinner(f"Preloading {len(sampled_batches)} batch cache(s)..."):
                    for prewarm_batch_id in sampled_batches:
                        prewarm_fn(prewarm_batch_id)

            st.info(f"Visualizing {n_actual} cells... (Results saved to {export_dir.name if export_dir else export_dir_str})")
            batch_stats = st.session_state.get("viz_val_batch_stats", {})
            has_stats = "means" in batch_stats
            cols = st.columns(min(n_actual, 3))
            idx_plot = 0
            summary_rows: list[dict[str, Any]] = []
            mic_scale = 0.5083

            for _, cell_row in random_sample.iterrows():
                try:
                    batch_val = cell_row[batch_col]
                    row_vals_raw = cell_row[loaded_markers]
                    row_vals = pd.to_numeric(row_vals_raw, errors="coerce").fillna(0)
                    if row_vals.dtype == "object":
                        st.error("Failed to convert row values to numeric.")
                        continue

                    if has_stats and batch_val in batch_stats["means"]:
                        batch_mean = pd.Series(batch_stats["means"][batch_val])
                        batch_std = pd.Series(batch_stats["stds"][batch_val])
                        batch_mean = batch_mean.reindex(row_vals.index).fillna(0)
                        batch_std = batch_std.reindex(row_vals.index).replace(0, 1).fillna(1)
                        z_scores = (row_vals - batch_mean) / batch_std
                        z_scores = pd.to_numeric(z_scores, errors="coerce").fillna(0)
                        top_5 = z_scores.nlargest(5)
                    else:
                        top_5 = row_vals.nlargest(5)

                    top_5_list: list[str] = []
                    top_5_clean: list[str] = []
                    row_data: dict[str, Any] = {
                        "cell_id": cell_row.get("cell_id"),
                        "batch_id": batch_val,
                        "x_centroid_px": cell_row.get("x_centroid"),
                        "y_centroid_px": cell_row.get("y_centroid"),
                        "x_centroid_mic_m": cell_row.get("x_centroid", 0) * mic_scale,
                        "y_centroid_mic_m": cell_row.get("y_centroid", 0) * mic_scale,
                        "phenotype_column": selected_pheno_col,
                        "phenotype_class": selected_phenotype,
                    }

                    score_idx = 1
                    for marker, score in top_5.items():
                        display_name = str(marker)
                        for suffix in ("_corr", "_norm", "_log"):
                            display_name = display_name.replace(suffix, "")
                        top_5_list.append(f"{display_name} ({float(score):.2f})")
                        top_5_clean.append(display_name)
                        row_data[f"Top_{score_idx}_Marker"] = display_name
                        row_data[f"Top_{score_idx}_Score"] = float(score)
                        score_idx += 1
                    summary_rows.append(row_data)

                    cell_ctx = viz.get_cell_context(
                        batch_id=str(batch_val),
                        x=cell_row[st.session_state.get("tab9_viz_col_x", "x_centroid")],
                        y=cell_row[st.session_state.get("tab9_viz_col_y", "y_centroid")],
                        requested_channels=top_5_clean,
                    )
                    available_channels = cell_ctx.get("channel_names", [])
                    indices_to_plot: list[int] = []
                    names_to_plot: list[str] = []
                    if available_channels:
                        chan_map = {channel.lower(): idx for idx, channel in enumerate(available_channels)}
                        for marker_name in top_5_clean:
                            channel_idx = -1
                            if marker_name in available_channels:
                                channel_idx = available_channels.index(marker_name)
                            elif marker_name.lower() in chan_map:
                                channel_idx = chan_map[marker_name.lower()]
                            if channel_idx != -1:
                                indices_to_plot.append(channel_idx)
                                names_to_plot.append(marker_name)

                    if not indices_to_plot:
                        st.warning(f"Cell {cell_row.get('cell_id')}: markers not found in image channels.")
                        continue

                    fig = viz.generate_visualization(
                        cell_data=cell_ctx,
                        marker_indices=indices_to_plot,
                        marker_names=names_to_plot,
                        display_size=tile_size,
                        show_neighbors=True,
                    )

                    if export_dir:
                        safe_pheno = "".join(char for char in str(selected_phenotype) if char.isalnum() or char in "_-")
                        fname = f"vv_{selected_pheno_col}_{safe_pheno}_cell_{cell_row['cell_id']}_{timestamp_int}.png"
                        fig.savefig(export_dir / fname, bbox_inches="tight")

                    with cols[idx_plot % 3]:
                        st.pyplot(fig)
                        st.caption(f"**Cell {cell_row['cell_id']}**\n{', '.join(top_5_list)}")
                    idx_plot += 1
                except Exception as inner_exc:
                    st.error(f"Error processing cell {cell_row.get('cell_id', 'unknown')}: {inner_exc}")
                    logger.error(f"Visual validation cell error: {inner_exc}")

            if summary_rows:
                st.markdown("##### Summary Table")
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df)
                if export_dir:
                    safe_pheno_col = "".join(char for char in str(selected_pheno_col) if char.isalnum() or char in "_-")
                    safe_pheno = "".join(char for char in str(selected_phenotype) if char.isalnum() or char in "_-")
                    csv_name = f"vv_summary_{safe_pheno_col}_{safe_pheno}_{timestamp_int}.csv"
                    csv_path = export_dir / csv_name
                    summary_df.to_csv(csv_path, index=False)
                    st.success(f"Saved summary to: `{csv_path}`")
                    st.success(f"Saved {idx_plot} images to: `{export_dir}`")
        except Exception as exc:
            st.error(f"Visualization Error: {exc}")
            logger.error(f"Outer Viz Error: {exc}")