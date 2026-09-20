from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import streamlit as st

from spacemap_gui.ui_utils import clamp_numeric_widget_state

from spacemap_gui.ui_utils import directory_input_with_browse

from spacemap.background_subtraction_utils import (
    get_channel_info,
    is_supported_image_file,
    load_preview_channels_compatible,
    normalize_image_stem,
)

from ..tab7_phenotyping.cell_visualization import (
    _DEFAULT_COLOR_MAP as _CELLVIZ_DEFAULT_COLOR_MAP,
    _add_unified_legend as _cellviz_add_unified_legend,
    _render_voronoi as _cellviz_render_voronoi,
)

from .feature_screening_core import (
    GPU_AVAILABLE,
    gpu_device_name,
    build_feature_endpoint_audit_df,
    build_feature_endpoint_attrition_summary,
    build_motif_threshold_sensitivity,
    build_triad_secondary_analysis,
    build_zone_patient_coverage_row,
    build_feature_influence_correlation_df,
    build_feature_endpoint_export_df,
    build_feature_pair_correlation_export_df,
    build_qc_summary,
    build_circle_cell_segmentation,
    build_voronoi_cell_segmentation,
    compute_roi_motif_visualization_data,
    find_geojson_for_batch,
    format_zone_interval_label,
    format_zone_label,
    load_geojson_cell_segmentation,
    load_saved_run,
    plot_candidate_scan,
    plot_cell_segmentation_figure,
    plot_family_summary,
    plot_feature_range_profile,
    plot_feature_influence_barplot,
    plot_feature_influence_scatter,
    plot_feature_stability_publication_panel,
    plot_feature_stability_heatmap,
    plot_feature_endpoint,
    plot_secondary_metric_survival_km,
    build_motif_enrichment_summary,
    plot_triad_patient_dotplot,
    plot_motif_threshold_sensitivity,
    plot_zone_patient_coverage,
    plot_feature_waterfall,
    plot_forest,
    plot_motif_density_cloud,
    plot_motif_null_histogram,
    plot_motif_zoom,
    plot_top_hits_correlation,
    plot_volcano,
    parse_feature_components,
    rank_candidates,
    run_feature_screening_pipeline,
    run_feature_validation_pipeline,
    save_feature_screening_run,
    zone_distance_interval_mask,
    zone_distance_mask,
    _draw_cell_segmentation_patches,
    _draw_zone_distance_contour,
)



def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        f = float(value)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default

def _log_terminal_event(message: str) -> None:
    print(f"[Tab8 Feature Screening] {message}", flush=True)
    logs = st.session_state.setdefault("_tab8_fs_logs", [])
    logs.append(message)
    st.session_state["_tab8_fs_logs"] = logs[-500:]


def _volcano_label_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    screening = config.get("screening", {}) if isinstance(config, dict) else {}
    top_n = int(
        max(
            1,
            round(
                _to_float(
                    screening.get(
                        "volcano_label_other_top_n",
                        st.session_state.get("tab8_fs_volcano_label_other_top_n", 3),
                    ),
                    3.0,
                )
            ),
        )
    )
    return {
        "volcano_label_discovery": bool(
            screening.get(
                "volcano_label_discovery",
                st.session_state.get("tab8_fs_volcano_label_discovery", False),
            )
        ),
        "volcano_label_nominal": bool(
            screening.get(
                "volcano_label_nominal",
                st.session_state.get("tab8_fs_volcano_label_nominal", False),
            )
        ),
        "volcano_label_other_top_n": top_n,
    }


def _default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    if preferred in options:
        return options.index(preferred)
    if not options:
        return 0
    return min(max(int(fallback), 0), len(options) - 1)


def _guess_column(columns: list[str], candidates: list[str], fallback: str = "") -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        exact = lowered.get(candidate.lower())
        if exact:
            return exact
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for column in columns:
            if candidate_lower in column.lower():
                return column
    return fallback if fallback in columns else ""


def _safe_unique(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique().tolist()
    try:
        return sorted(values)
    except Exception:
        return values


def _ensure_state_value(key: str, default: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = default


def _ensure_select_state(key: str, options: list[Any], default: Any | None = None) -> None:
    non_empty_options = [opt for opt in options if opt is not None and str(opt).strip() != ""]
    if not non_empty_options:
        return
    current = st.session_state.get(key, default)
    if current not in options:
        matched_option = None
        if isinstance(current, str):
            for opt in options:
                if isinstance(opt, str) and opt.lower() == current.lower():
                    matched_option = opt
                    break
        if matched_option is not None:
            st.session_state[key] = matched_option
        else:
            st.session_state[key] = default if default in options else options[0]


def _ensure_multiselect_state(key: str, options: list[Any], default: list[Any] | None = None) -> None:
    allowed = list(options)
    non_empty_allowed = [opt for opt in allowed if opt is not None and str(opt).strip() != ""]
    if not non_empty_allowed:
        return
    current = st.session_state.get(key, default or [])
    if not isinstance(current, (list, tuple, set)):
        current = [current] if current in allowed else []
    
    valid = []
    allowed_lower = {str(val).lower(): val for val in allowed}
    for val in current:
        if val in allowed:
            valid.append(val)
        else:
            val_str = str(val)
            if val_str.lower() in allowed_lower:
                valid.append(allowed_lower[val_str.lower()])
                
    if key not in st.session_state:
        seeded = []
        for val in (default or []):
            if val in allowed:
                seeded.append(val)
            else:
                val_str = str(val)
                if val_str.lower() in allowed_lower:
                    seeded.append(allowed_lower[val_str.lower()])
        st.session_state[key] = seeded or valid
    elif valid != list(current):
        st.session_state[key] = valid


def _stateful_selectbox(label: str, options: list[Any], key: str, default: Any | None = None, help: str | None = None) -> None:
    _ensure_select_state(key, options, default=default)
    st.selectbox(label, options=options, key=key, help=help)


def _stateful_multiselect(label: str, options: list[Any], key: str, default: list[Any] | None = None, help: str | None = None) -> None:
    _ensure_multiselect_state(key, options, default=default)
    st.multiselect(label, options=options, key=key, help=help)


def _stateful_checkbox(label: str, key: str, default: bool, help: str | None = None, disabled: bool = False) -> None:
    _ensure_state_value(key, bool(default))
    st.checkbox(label, key=key, help=help, disabled=disabled)


def _stateful_text_input(label: str, key: str, default: str, help: str | None = None) -> None:
    _ensure_state_value(key, str(default))
    st.text_input(label, key=key, help=help)


def _stateful_text_area(label: str, key: str, default: str, height: int = 90, help: str | None = None) -> None:
    _ensure_state_value(key, str(default))
    st.text_area(label, key=key, height=height, help=help)


def _stateful_number_input(label: str, key: str, default: Any, help: str | None = None, **kwargs: Any) -> None:
    _ensure_state_value(key, default)
    # Several callers derive min/max from the loaded table. A value remembered
    # from a larger table would put the widget out of range and raise, so clamp
    # it here rather than at each call site.
    if "min_value" in kwargs and "max_value" in kwargs:
        clamp_numeric_widget_state(key, kwargs["min_value"], kwargs["max_value"], fallback=default)
    st.number_input(label, key=key, help=help, **kwargs)


def _direction_text(direction: Any, endpoint_type: Any, hazard_ratio: Any = None) -> str:
    endpoint_text = str(endpoint_type or "").strip().lower()
    if endpoint_text != "survival":
        return "N/A"
    hazard_ratio_value = _to_float(hazard_ratio)
    direction_text = str(direction or "").strip().lower()
    if direction_text == "risk":
        return "Risk"
    if direction_text == "protective":
        return "Protective"
    if direction_text == "neutral":
        if np.isfinite(hazard_ratio_value) and abs(float(hazard_ratio_value) - 1.0) < 1e-9:
            return "Neutral"
        return "Unclear"
    return "-"


def _significance_display_label(label: Any) -> str:
    label_str = str(label or "").strip()
    return {
        "significant_fdr": "FDR significant",
        "discovery_fdr": "FDR discovery",
        "not_significant": "Not significant",
    }.get(label_str, "-")


_FIGURE_MIME = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "tiff": "image/tiff",
}


def _publication_output_settings() -> dict[str, Any]:
    """Export settings shared by every figure in the tab.

    Read from session state at render time rather than passed down, so the one
    download helper governs all figures without threading a settings object
    through every plotting call site.
    """
    formats = [str(v).lower() for v in st.session_state.get("tab8_fs_figure_formats", ["png"]) if str(v)]
    return {
        "dpi": int(_to_float(st.session_state.get("tab8_fs_figure_dpi", 300), 300.0)),
        "formats": formats or ["png"],
        "transparent": bool(st.session_state.get("tab8_fs_figure_transparent", False)),
        "eager": bool(st.session_state.get("tab8_fs_figure_eager_downloads", False)),
    }


def _figure_download_bytes(
    fig: Figure,
    dpi: int = 300,
    fmt: str = "png",
    transparent: bool = False,
) -> bytes:
    buf = io.BytesIO()
    # Vector formats ignore dpi for the geometry but still use it to rasterise
    # any embedded raster artists, so it is passed through in every format.
    fig.savefig(
        buf,
        format=fmt,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="none" if transparent else "white",
        transparent=bool(transparent),
    )
    buf.seek(0)
    return buf.getvalue()


def _render_figure_with_download(
    fig: Figure,
    title: str,
    key_prefix: str,
    download_dpi: int | None = None,
    download_fig: Figure | None = None,
) -> None:
    st.markdown(f"#### {title}")
    st.pyplot(fig, clear_figure=False)
    export_fig = download_fig if download_fig is not None else fig

    settings = _publication_output_settings()
    dpi = int(download_dpi) if download_dpi is not None else int(settings["dpi"])
    formats = [f for f in (str(v).lower() for v in settings["formats"]) if f in _FIGURE_MIME]
    if not formats:
        return

    # Streamlit needs a download button's bytes up front, so rendering every
    # figure to every format inline means a full savefig sweep on EVERY rerun --
    # even one caused by ticking an unrelated checkbox. At ~0.3-1.0 s per figure
    # per format (a single 300 dpi TIFF is ~10 MB) that dominates page load and
    # makes the tab feel frozen. So exports are prepared on request and cached
    # for the rest of the session; only the on-screen figure is drawn by default.
    cache_key = f"_tab8_fig_export__{key_prefix}"
    signature = (tuple(formats), int(dpi), bool(settings["transparent"]))
    cached = st.session_state.get(cache_key)
    payloads = cached["payloads"] if isinstance(cached, dict) and cached.get("signature") == signature else None

    if payloads is None and not settings["eager"]:
        if st.button(
            f"Prepare {title} download ({', '.join(f.upper() for f in formats)})",
            key=f"{key_prefix}_prepare_export_btn",
            help="Renders this figure for download. Done on request so that ordinary page "
                 "interactions do not re-export every figure on the page.",
        ):
            st.session_state[cache_key] = {"signature": signature, "payloads": None}
            st.rerun()
        return

    if payloads is None:
        payloads = {}
        for fmt in formats:
            try:
                payloads[fmt] = _figure_download_bytes(
                    export_fig, dpi=dpi, fmt=fmt, transparent=settings["transparent"]
                )
            except Exception as exc:
                payloads[fmt] = exc
        st.session_state[cache_key] = {"signature": signature, "payloads": payloads}

    columns = st.columns(len(payloads)) if len(payloads) > 1 else [st]
    for column, (fmt, payload) in zip(columns, payloads.items()):
        if isinstance(payload, Exception):
            column.caption(f"{fmt.upper()} export unavailable: {payload}")
            continue
        label = (
            f"Download {title} ({fmt.upper()})"
            if fmt in {"svg", "pdf"}
            else f"Download {title} ({fmt.upper()}, {dpi} dpi)"
        )
        column.download_button(
            label,
            data=payload,
            file_name=f"{key_prefix}.{fmt}",
            mime=_FIGURE_MIME[fmt],
            key=f"{key_prefix}_download_{fmt}",
        )


_RENDER_TIMING_KEY = "_tab8_fs_render_timings"


class _timed_section:
    """Context manager that records how long a render section took.

    Streamlit re-executes the whole script on every widget interaction, so a
    single slow section makes an unrelated dropdown feel frozen. Timings are
    logged to the terminal and kept in session state so the Process Log can show
    where a slow rerun actually went, instead of leaving it to guesswork.
    """

    def __init__(self, label: str, threshold_s: float = 1.0):
        self.label = str(label)
        self.threshold_s = float(threshold_s)
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        timings = st.session_state.setdefault(_RENDER_TIMING_KEY, {})
        timings[self.label] = round(float(elapsed), 3)
        if elapsed >= self.threshold_s:
            _log_terminal_event(f"RENDER {self.label}: {elapsed:.1f}s")
        return False


def _render_log_panel() -> None:
    with st.expander("Process Log", expanded=False):
        progress_state = st.session_state.get("_tab8_fs_progress", {})
        latest_message = str(progress_state.get("message") or "").strip()
        if latest_message:
            st.caption(f"Latest progress: {latest_message}")
        logs = st.session_state.get("_tab8_fs_logs", [])
        if not logs:
            st.caption("No log events yet.")
            return
        joined = "\n".join(str(line) for line in logs)
        st.code(joined, language="text")
        st.download_button(
            "Download log",
            data=joined.encode("utf-8"),
            file_name="feature_screening.log",
            mime="text/plain",
            key="tab8_fs_download_log",
        )


def _render_reference_defaults_action(project_root: Path, unified_csv_path: str | None) -> None:
    col_reset, col_export, col_import = st.columns([1, 1, 2])
    with col_reset:
        if st.button(
            "↺ Reset to Defaults",
            key="tab8_fs_reset_defaults_top_btn",
            help="Restore the original reference starting values for the current CSV. Does not delete saved run folders on disk.",
            width="stretch",
        ):
            _queue_reference_defaults_reset()
            st.rerun()
    with col_export:
        snap = _collect_settings_snapshot()
        snap["_exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        snap_json = json.dumps(snap, indent=2, default=_json_settings_default)
        st.download_button(
            "⬇ Export Settings",
            data=snap_json.encode("utf-8"),
            file_name="tab8_settings.json",
            mime="application/json",
            key="tab8_fs_export_settings_btn",
            help="Download all current Tab 8 settings as a JSON file you can share or re-import later.",
            width="stretch",
        )
    with col_import:
        st.caption("⬆ Import settings from a previously exported JSON file")
        import_counter = int(st.session_state.get("_tab8_fs_import_counter", 0))
        uploaded = st.file_uploader(
            "Import settings JSON",
            type=["json"],
            key=f"tab8_fs_import_settings_uploader_{import_counter}",
            label_visibility="collapsed",
            help="Upload a Tab 8 settings JSON to restore its configuration.",
        )
        if uploaded is not None:
            try:
                raw = json.loads(uploaded.read().decode("utf-8"))
                if isinstance(raw, dict):
                    valid = {
                        k: v
                        for k, v in raw.items()
                        if isinstance(k, str) and k.startswith("tab8_fs_") and not _is_widget_key(k)
                    }
                    _apply_settings_snapshot(valid)
                    _save_settings_to_cache(project_root, unified_csv_path)
                    st.session_state["_tab8_fs_import_counter"] = import_counter + 1
                    st.toast(f"Imported {len(valid)} settings. Apply configuration to use them.", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid settings file: expected a JSON object.")
            except Exception as exc:
                st.error(f"Failed to import settings: {exc}")


def _width_state_key(width_value: Any) -> str:
    numeric = _to_float(width_value)
    if not np.isfinite(numeric):
        return ""
    return f"{float(numeric):.6f}"


def _reference_defaults(df: pd.DataFrame, project_root: Path, unified_csv_path: str | None) -> dict[str, Any]:
    columns = df.columns.tolist()
    defaults: dict[str, Any] = {
        "tab8_fs_patient_col": _guess_column(columns, ["Patient", "patient_id", "patient"]),
        "tab8_fs_batch_col": _guess_column(columns, ["File Name", "file_name", "batch", "sample", "image"]),
        "tab8_fs_roi_col": _guess_column(columns, ["ROI", "roi", "region", "batch_id"]),
        "tab8_fs_x_col": _guess_column(columns, ["x_centroid", "X:X", "x"]),
        "tab8_fs_y_col": _guess_column(columns, ["y_centroid", "Y:Y", "y"]),
        "tab8_fs_phenotype_col": _guess_column(columns, ["phenotype_ml_consensus", "phenotype_ml_consensus_refined", "phenotype", "cell_type"]),
        "tab8_fs_distance_col": _guess_column(columns, ["neighborhood_distance_tumor", "distance_to_tumor", "distance"]),
        "tab8_fs_context_col": _guess_column(columns, ["neighborhood", "tumor_compartment", "region_class"]),
        "tab8_fs_output_dir": str(project_root / "outputs" / "spatial_feature_screening"),
        "tab8_fs_raw_image_dir": str(project_root / "outputs" / "registration" / "merged"),
        "tab8_fs_os_time_col": _guess_column(columns, ["OS", "overall_survival", "os_time"]),
        "tab8_fs_os_censor_col": _guess_column(columns, ["OS_Censor", "os_censor", "OS_event"]),
        "tab8_fs_dfs_time_col": _guess_column(columns, ["DFS", "disease_free_survival", "dfs_time"]),
        "tab8_fs_dfs_censor_col": _guess_column(columns, ["DFS_Censor", "dfs_censor", "DFS_event"]),
        "tab8_fs_response_col": _guess_column(columns, ["response", "Response", "best_response"]),
        "tab8_fs_response_multi_mode": False,
        "tab8_fs_response_multi_cols": [],
        "tab8_fs_apply_note": "",
        "tab8_fs_feature_families": ["density", "proportion", "ratio", "dyad", "triad", "diversity", "nn_distance"],
        "tab8_fs_stratify_cols": [],
        "tab8_fs_row_filter_cols": [],
        "tab8_fs_context_values": [],
        "tab8_fs_triads_custom_text": "",
        "tab8_fs_loaded_run_dir": "",
        "tab8_fs_unified_csv_path": unified_csv_path or "",
        "tab8_fs_enable_os": True,
        "tab8_fs_os_label": "OS",
        "tab8_fs_os_censor_means_censored": False,
        "tab8_fs_enable_dfs": True,
        "tab8_fs_dfs_label": "DFS",
        "tab8_fs_dfs_censor_means_censored": False,
        "tab8_fs_enable_response": False,
        "tab8_fs_response_label": "Response",
        "tab8_fs_response_multi_mode": False,
        "tab8_fs_response_multi_cols": [],
        "tab8_fs_response_positive_value": "",
        "tab8_fs_outer_cap_mode": "percentile",
        "tab8_fs_zone_geometry": "cumulative",
        "tab8_fs_workflow_panel": "Discovery Scan",
        "tab8_fs_figure_dpi": 300,
        "tab8_fs_figure_formats": ["png"],
        "tab8_fs_figure_transparent": False,
        "tab8_fs_figure_eager_downloads": False,
        "tab8_fs_outer_cap_percentile": 99.0,
        "tab8_fs_outer_cap_value": 300.0,
        "tab8_fs_inner_cap_percentile": 5.0,
        "tab8_fs_n_candidate_widths": 15,
        "tab8_fs_round_width_to": 5.0,
        "tab8_fs_max_width": 0.0,
        "tab8_fs_exclude_negative": True,
        "tab8_fs_excluded_phenotypes": [],
        "tab8_fs_use_micron_per_pixel": True,
        "tab8_fs_micron_per_pixel": 0.5083,
        # NOTE: the block below is seeded from a validated working configuration.
        # Two values were deliberately NOT copied from that run:
        #   * micron_per_pixel stays 0.5083, not the 0.0001 placeholder. Enabling
        #     the conversion with 0.0001 multiplies every distance by 10,000.
        #   * min_obs_motif_count stays 3, not 0. At 0 an ROI holding a single
        #     motif instance still produces a Z-score.
        "tab8_fs_min_cells_per_zone": 40,
        "tab8_fs_min_cells_for_feature": 10,
        "tab8_fs_ratio_pseudocount": 1.0,
        "tab8_fs_log_transform_ratios": True,
        "tab8_fs_auto_interaction_distance": True,
        "tab8_fs_auto_interaction_percentile": 95.0,
        "tab8_fs_allow_self_interaction_features": False,
        "tab8_fs_use_adaptive_interaction_radius": True,
        "tab8_fs_adaptive_radius_direct_neighbours_only": True,
        "tab8_fs_adaptive_radius_min_pct": 0.0,
        "tab8_fs_adaptive_radius_max_pct": 99.0,
        "tab8_fs_adaptive_radius_max_sample": 500,
        "tab8_fs_dyad_triad_permutation_n_iter": 1000,
        "tab8_fs_min_obs_motif_count": 3,
        "tab8_fs_permutation_chunk_size": 64,
        "tab8_fs_permutation_use_gpu": True,
        "tab8_fs_n_jobs": 6,
        "tab8_fs_dyad_min_um": 0.0,
        "tab8_fs_dyad_max_um": 30.0,
        "tab8_fs_triad_min_um": 0.0,
        "tab8_fs_triad_max_um": 30.0,
        "tab8_fs_nn_k": 4,
        "tab8_fs_area_use_median": True,
        "tab8_fs_area_percentile": 95.0,
        "tab8_fs_min_hole_size_px2": 0.0,
        "tab8_fs_patient_aggregation": "median",
        "tab8_fs_split_method": "median",
        "tab8_fs_quantile_low": 0.25,
        "tab8_fs_quantile_high": 0.75,
        "tab8_fs_custom_cutpoint": 0.0,
        "tab8_fs_min_patients_per_group": 8,
        "tab8_fs_show_ci": False,
        "tab8_fs_show_excluded_km_curve": False,
        "tab8_fs_volcano_label_discovery": False,
        "tab8_fs_volcano_label_nominal": False,
        "tab8_fs_volcano_label_other_top_n": 4,
        "tab8_fs_time_unit": "Months",
        "tab8_fs_fdr_mode": "per_family",
        "tab8_fs_fdr_alpha": 0.05,
        "tab8_fs_fdr_alpha_discovery": 0.10,
        "tab8_fs_top_k_for_bootstrap": 5,
        "tab8_fs_bootstrap_iterations": 30,
        "tab8_fs_bootstrap_sampling_unit": "patient",
        "tab8_fs_bootstrap_patient_fraction": 0.80,
        "tab8_fs_bootstrap_random_seed": 20260407,
        "tab8_fs_feature_bootstrap_neighbor_steps": 1,
        "tab8_fs_feature_stability_top_n": 20,
        "tab8_fs_min_patients_with_zone": 12,
        "tab8_fs_repr_per_group": 1,
        "tab8_fs_thumbnail_padding_px": 0,
        "tab8_fs_thumbnail_max_dim": 200,
        "tab8_fs_repr_export_dpi": 300,
        "tab8_fs_repr_export_max_dim": 2200,
        "tab8_fs_repr_selected_markers": [],
        "tab8_fs_repr_feature_overlay_mode": "Feature dots",
        "tab8_fs_repr_feature_overlay_color": "#ff6b00",
        "tab8_fs_repr_feature_overlay_alpha": 0.70,
        "tab8_fs_repr_feature_overlay_size": 36.0,
        "tab8_fs_val_zone_width": 50.0,
        "tab8_fs_val_zone_start": 0.0,
        "tab8_fs_val_zone_end": 50.0,
        "tab8_fs_val_selected_phenotypes": [],
        "tab8_fs_val_top_n": 25,
        "tab8_fs_val_endpoint": "",
        "tab8_fs_val_stratum": "",
        "tab8_fs_val_family_filter": "All",
        "tab8_fs_val_focus_elements": [],
        "tab8_fs_val_focus_require_all": True,
        "tab8_fs_val_focus_features": [],
        "tab8_fs_val_highlight_feature": "",
        "tab8_fs_val_influence_components": [],
        "tab8_fs_val_influence_families": ["density", "proportion", "ratio", "dyad", "triad", "nn_distance"],
        "tab8_fs_val_influence_require_all": False,
        "tab8_fs_val_influence_min_pairs": 5,
        "tab8_fs_val_influence_top_n": 15,
        "tab8_fs_val_influence_related_feature": "",
    }
    phenotype_col = str(defaults.get("tab8_fs_phenotype_col") or "")
    # Screen every real phenotype the column contains, minus the catch-all and
    # missing labels. The old default silently truncated to the first 10, which
    # on a 13-type panel dropped three cell types from every run without saying so.
    if phenotype_col and phenotype_col in df.columns:
        _pheno_values = [
            value for value in _safe_unique(df[phenotype_col])
            if str(value).strip().lower() not in {"", "nan", "none", "others", "other", "unassigned", "unknown"}
        ]
        defaults["tab8_fs_selected_phenotypes"] = _pheno_values
        defaults["tab8_fs_excluded_phenotypes"] = [
            value for value in _safe_unique(df[phenotype_col])
            if str(value).strip().lower() in {"others", "other", "unassigned", "unknown"}
        ]
    else:
        defaults["tab8_fs_selected_phenotypes"] = []
    defaults["tab8_fs_val_phenotype_col"] = phenotype_col
    defaults["tab8_fs_val_feature_families"] = list(defaults.get("tab8_fs_feature_families", []))
    return defaults


def _apply_reference_defaults(df: pd.DataFrame, project_root: Path, unified_csv_path: str | None) -> None:
    defaults = _reference_defaults(df, project_root, unified_csv_path)
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("tab8_fs_filter_values__"):
            st.session_state.pop(key, None)
    for key, value in defaults.items():
        st.session_state[key] = value
    for key in [
        "_tab8_fs_results",
        "_tab8_fs_loaded_run",
        "_tab8_fs_applied_config",
        "_tab8_fs_applied_signature",
        "_tab8_fs_validation_results",
        "_tab8_fs_validation_influence_results",
        "_tab8_fs_validation_influence_signature",
        "_tab8_fs_val_influence_last_feature_key",
    ]:
        st.session_state.pop(key, None)
    st.session_state["tab8_fs_apply_note"] = "Reference defaults restored. Apply the configuration or run the screen when ready."
    from spacemap_gui.persistence import save_active_project_state
    save_active_project_state("tab8_analysis")


def _queue_reference_defaults_reset() -> None:
    st.session_state["_tab8_fs_pending_reference_reset"] = True


def _consume_pending_reference_reset(df: pd.DataFrame, project_root: Path, unified_csv_path: str | None) -> bool:
    if not bool(st.session_state.pop("_tab8_fs_pending_reference_reset", False)):
        return False
    _apply_reference_defaults(df, project_root, unified_csv_path)
    _log_terminal_event("Reference defaults restored")
    return True


# ── Settings persistence (auto-save / export / import) ────────────────────────

_SETTINGS_KEY_PREFIX = "tab8_fs_"
_SETTINGS_CACHE_DIR = ".tab8_cache"


_SETTINGS_WIDGET_SUFFIXES = (
    "_btn",
    "_button",
    "_uploader",
    "_download",
    "_form_submit",
    "_patient_csv",
    "_motif_cell_csv",
    "_motif_instance_csv",
    "_prepare_all_roi_motif_export",
    # ROI Explorer layer management buttons
    "_add_markers",
    "_add_annotations",
    "_add_cells",
    "_add_scatter",
    "_addbtn",  # filter-rule "Add filter" button (*_filter_rules_addbtn)
    "_del",     # layer delete (*_del_{int}) and filter-rule remove (*_r{int}_del)
    "_up",      # layer move-up:   *_up_{int}
    "_dn",      # layer move-down: *_dn_{int}
)
_SETTINGS_WIDGET_TOKENS = ("_download_",)
_LEGACY_WIDGET_STATE_KEYS = (
    "tab8_fs_selected_hit_prepare_all_roi_motif_export",
    "tab8_fs_val_selected_hit_prepare_all_roi_motif_export",
)


def _is_widget_key(key: str) -> bool:
    """Return True for keys owned by Streamlit widgets (buttons, uploaders, download buttons).
    These cannot be written to session_state directly."""
    lk = key.lower()
    for token in _SETTINGS_WIDGET_TOKENS:
        if token in lk:
            return True
    for suffix in _SETTINGS_WIDGET_SUFFIXES:
        # exact suffix or suffix followed by underscore+digits (dynamic counters)
        if lk.endswith(suffix):
            return True
        idx = lk.rfind(suffix)
        if idx != -1 and lk[idx + len(suffix) :].lstrip("_").isdigit():
            return True
    return False


def _drop_legacy_widget_state_keys() -> None:
    for key in _LEGACY_WIDGET_STATE_KEYS:
        st.session_state.pop(key, None)


def _json_settings_default(obj: Any) -> Any:
    """JSON fallback serializer: converts numpy scalars to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


_SETTINGS_PROJECT_FILENAME = "settings_tab8_project.json"


def _settings_cache_path(project_root: Path, unified_csv_path: str | None = None) -> Path:
    """Path of the Tab 8 settings file.

    Deliberately **project-scoped, not CSV-scoped**. These are method settings --
    zone geometry, FDR thresholds, permutation counts, feature families -- and
    they describe how this project analyses data, not which file it was pointed
    at. Keying them per CSV meant swapping the unified CSV silently reset every
    control to defaults. ``unified_csv_path`` is accepted only so existing call
    sites keep working.
    """
    return project_root / _SETTINGS_CACHE_DIR / _SETTINGS_PROJECT_FILENAME


def _legacy_settings_cache_paths(project_root: Path) -> list[Path]:
    """Per-CSV settings files written before settings became project-scoped."""
    cache_dir = project_root / _SETTINGS_CACHE_DIR
    if not cache_dir.is_dir():
        return []
    return sorted(
        (p for p in cache_dir.glob("settings_*.json") if p.name != _SETTINGS_PROJECT_FILENAME),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


# A settings entry is a control value the user typed or picked. Anything that is
# actually data -- a table of cells, a rendered figure, a cached result -- must
# never reach the project file. Values are capped individually and in total, so
# no future key can quietly turn the project JSON into a data dump.
_SETTINGS_MAX_VALUE_BYTES = 64 * 1024        # 64 KB per setting
_SETTINGS_MAX_TOTAL_BYTES = 2 * 1024 * 1024  # 2 MB for the whole Tab 8 block


def _settings_safe_value(value: Any) -> Any | None:
    """Return a JSON-safe copy of ``value``, or None if it is not a setting.

    Rejects data payloads (frames, arrays, bytes) and anything that serializes
    larger than ``_SETTINGS_MAX_VALUE_BYTES``. A dict or list is inspected
    recursively, because the container itself is small while its contents may
    not be -- which is exactly how a per-cell table reached the project file.
    """
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, bytes, bytearray, memoryview)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > _SETTINGS_MAX_VALUE_BYTES:
            return None
        return value
    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for item in value:
            safe_item = _settings_safe_value(item)
            if safe_item is None:
                # Reject the whole list rather than quietly dropping an element:
                # a list holding data is not a setting, and silently returning a
                # shortened list would corrupt the setting instead of skipping it.
                return None
            cleaned.append(safe_item)
        value = cleaned
    elif isinstance(value, dict):
        cleaned_map = {}
        for k, v in value.items():
            safe = _settings_safe_value(v)
            if safe is None:
                return None  # a dict holding data is not a setting
            cleaned_map[str(k)] = safe
        value = cleaned_map
    else:
        try:
            value = str(value)
        except Exception:
            return None
        if len(value) > _SETTINGS_MAX_VALUE_BYTES:
            return None
        return value

    try:
        if len(json.dumps(value, default=_json_settings_default)) > _SETTINGS_MAX_VALUE_BYTES:
            return None
    except Exception:
        return None
    return value


def _collect_settings_snapshot() -> dict[str, Any]:
    """Return all public tab8_fs_* session-state entries as a JSON-safe dict.
    Widget-owned keys (buttons, uploaders) are excluded since they cannot be
    restored via session_state assignment."""
    snap: dict[str, Any] = {}
    for key, value in st.session_state.items():
        if not isinstance(key, str):
            continue
        if not key.startswith(_SETTINGS_KEY_PREFIX) or key.startswith("_"):
            continue
        if _is_widget_key(key):
            continue
        safe = _settings_safe_value(value)
        if safe is None:
            continue
        snap[key] = safe

    # Final backstop on the whole block.
    try:
        if len(json.dumps(snap, default=_json_settings_default)) > _SETTINGS_MAX_TOTAL_BYTES:
            _log_terminal_event(
                f"Tab 8 settings snapshot exceeded {_SETTINGS_MAX_TOTAL_BYTES // 1024} KB; "
                "dropping the largest entries rather than writing a bloated project file."
            )
            ordered = sorted(
                snap.items(),
                key=lambda kv: len(json.dumps(kv[1], default=_json_settings_default)),
                reverse=True,
            )
            for key, _value in ordered:
                snap.pop(key, None)
                if len(json.dumps(snap, default=_json_settings_default)) <= _SETTINGS_MAX_TOTAL_BYTES:
                    break
    except Exception:
        pass
    return snap


def _apply_settings_snapshot(snapshot: dict[str, Any]) -> None:
    """Write a settings snapshot dict into Streamlit session state, skipping widget keys."""
    for key, value in snapshot.items():
        if not isinstance(key, str):
            continue
        if not key.startswith(_SETTINGS_KEY_PREFIX):
            continue
        if _is_widget_key(key):
            continue
        try:
            st.session_state[key] = value
        except Exception:
            pass


def _save_settings_to_cache(project_root: Path, unified_csv_path: str | None) -> None:
    """Silently persist current settings to a per-CSV JSON cache file."""
    try:
        cache_path = _settings_cache_path(project_root, unified_csv_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        snap = _collect_settings_snapshot()
        
        # Safeguard: do not overwrite a valid cache file if the current settings in st.session_state are completely empty/blank.
        if cache_path.exists():
            try:
                old_data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(old_data, dict):
                    old_patient_col = old_data.get("tab8_fs_patient_col", "")
                    new_patient_col = snap.get("tab8_fs_patient_col", "")
                    if old_patient_col and not new_patient_col:
                        return
            except Exception:
                pass
                
        snap["_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        cache_path.write_text(json.dumps(snap, indent=2, default=_json_settings_default), encoding="utf-8")
    except Exception:
        pass


def _sync_settings_into_project_json(project_root: Path, unified_csv_path: str | None) -> None:
    """Mirror the current Tab 8 settings into the project JSON when they change.

    The settings cache lives inside the project folder and is what this tab
    reads back, but the project JSON is what travels with the project and what
    other sessions and machines load. Writing only on an explicit "Save Project"
    click meant a second session could open the project and receive settings
    from whenever that button was last pressed, silently discarding newer work.

    Guarded by a hash so the project file is rewritten only on a real change,
    not on every Streamlit rerun.
    """
    try:
        snapshot = _collect_settings_snapshot()
        if not snapshot:
            return
        digest = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, default=_json_settings_default).encode()
        ).hexdigest()
        if st.session_state.get("_tab8_fs_project_sync_digest") == digest:
            return

        active_project = st.session_state.get("active_project")
        if not isinstance(active_project, dict):
            return
        tab_configs = active_project.setdefault("tab_configurations", {})
        tab8_config = tab_configs.setdefault("tab8_analysis", {})
        if not isinstance(tab8_config, dict):
            tab8_config = {}
            tab_configs["tab8_analysis"] = tab8_config
        tab8_config.update(snapshot)

        from spacemap_gui.persistence import save_active_project_state

        save_active_project_state("tab8_analysis")
        st.session_state["_tab8_fs_project_sync_digest"] = digest
    except Exception:
        # Persistence must never take the tab down; the settings cache above is
        # still written, so nothing is lost even when this path fails.
        pass


def _load_settings_from_cache(project_root: Path, unified_csv_path: str | None) -> dict[str, Any] | None:
    """Load settings from the per-CSV cache file. Returns None if absent or invalid."""
    try:
        cache_path = _settings_cache_path(project_root, unified_csv_path)
        if not cache_path.exists():
            # One-time migration: settings used to be stored per CSV. Adopt the
            # most recently written per-CSV file so upgrading does not throw the
            # user's existing configuration away.
            legacy = _legacy_settings_cache_paths(project_root)
            if not legacy:
                return None
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(legacy[0].read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                cache_path = legacy[0]
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        saved_settings = {k: v for k, v in data.items() if isinstance(k, str) and k.startswith(_SETTINGS_KEY_PREFIX)}
        sanitized = {k: v for k, v in saved_settings.items() if not _is_widget_key(k)}
        if len(sanitized) != len(saved_settings):
            clean_payload = {
                **sanitized,
                "_saved_at": data.get("_saved_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
            }
            cache_path.write_text(json.dumps(clean_payload, indent=2, default=_json_settings_default), encoding="utf-8")
        return sanitized
    except Exception:
        return None


def _init_defaults(df: pd.DataFrame, project_root: Path, unified_csv_path: str | None) -> None:
    defaults = _reference_defaults(df, project_root, unified_csv_path)
    
    # 1. Get settings from active project JSON configuration (tab8_analysis)
    project_settings = {}
    active_project = st.session_state.get("active_project")
    if active_project and isinstance(active_project, dict):
        project_settings = active_project.get("tab_configurations", {}).get("tab8_analysis", {})
        if not isinstance(project_settings, dict):
            project_settings = {}

    # 2. Get settings from local cache file
    cached = _load_settings_from_cache(project_root, unified_csv_path)
    
    # Merge them: project JSON settings take highest priority, then cached, then reference defaults
    # Priority, lowest to highest: reference defaults -> local settings cache ->
    # project JSON. The project JSON is the authoritative, portable record: it is
    # what travels with the project, what a second machine reads, and what you
    # copy to clone a configuration onto a different unified CSV. The cache is
    # only a same-machine fallback for a project whose JSON has no Tab 8 block yet.
    #
    # This ordering is safe only because settings are now mirrored into the
    # project JSON automatically whenever they change (see
    # ``_sync_settings_into_project_json``). Before that, the JSON was written
    # solely by the Save Project button, so putting it last let a stale file
    # silently revert newer settings.
    merged = {**defaults, **(cached or {}), **project_settings}
    
    # On the first render of a session the merged configuration is authoritative:
    # write it even over keys that already carry a value. Widgets here seed their
    # own default from session state (``default=st.session_state.get(key, ...)``),
    # so any key left unset -- or set empty by a partial/failed earlier load --
    # makes the widget seed itself empty and then persist that emptiness. Seeding
    # once, up front, removes that whole failure mode.
    #
    # After the first render the flag stops this, so edits made in the UI are
    # never overwritten by the saved configuration.
    first_render = not bool(st.session_state.get("_tab8_fs_defaults_seeded"))
    applied = 0
    preexisting = 0
    for key, value in merged.items():
        if not isinstance(key, str):
            continue
        if _is_widget_key(key):
            continue
        if first_render or key not in st.session_state:
            try:
                st.session_state[key] = value
                applied += 1
            except Exception:
                # A widget already instantiated this key on this run; leave it.
                preexisting += 1
        else:
            preexisting += 1
    st.session_state["_tab8_fs_defaults_seeded"] = True

    # One line, once per session, saying where the settings actually came from.
    # Without it, "all my boxes are empty" is indistinguishable between a config
    # that failed to load, a CSV that produced no columns, and values already
    # sitting in session state from an earlier partial load.
    if not st.session_state.get("_tab8_fs_init_reported"):
        st.session_state["_tab8_fs_init_reported"] = True
        try:
            df_shape = tuple(df.shape) if isinstance(df, pd.DataFrame) else None
        except Exception:
            df_shape = None
        _log_terminal_event(
            f"init: df={df_shape} | active_project={'yes' if active_project else 'NO'}"
            f" | project_settings={len(project_settings)} | cache={len(cached or {})}"
            f" | defaults={len(defaults)} | applied={applied} | already_in_session={preexisting}"
            f" | patient_col={st.session_state.get('tab8_fs_patient_col')!r}"
            f" | min_cells={st.session_state.get('tab8_fs_min_cells_per_zone')!r}"
        )


def _parse_triads_text(raw_text: str) -> list[tuple[str, str, str]]:
    triads: list[tuple[str, str, str]] = []
    for line in str(raw_text or "").splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) == 3:
            triads.append((parts[0], parts[1], parts[2]))
    return triads


def _build_config(df: pd.DataFrame, unified_csv_path: str | None) -> dict[str, Any]:
    columns = {
        "patient": st.session_state.get("tab8_fs_patient_col", ""),
        "batch": st.session_state.get("tab8_fs_batch_col", ""),
        "roi": st.session_state.get("tab8_fs_roi_col", ""),
        "x": st.session_state.get("tab8_fs_x_col", ""),
        "y": st.session_state.get("tab8_fs_y_col", ""),
        "phenotype": st.session_state.get("tab8_fs_phenotype_col", ""),
        "distance": st.session_state.get("tab8_fs_distance_col", ""),
        "context_column": st.session_state.get("tab8_fs_context_col", ""),
        "context_values": st.session_state.get("tab8_fs_context_values", []),
        "stratify": st.session_state.get("tab8_fs_stratify_cols", []),
    }

    row_filters: dict[str, list[str]] = {}
    for column in st.session_state.get("tab8_fs_row_filter_cols", []):
        values = st.session_state.get(f"tab8_fs_filter_values__{column}", [])
        if values:
            row_filters[str(column)] = [str(value) for value in values]

    endpoints = [
        {
            "enabled": bool(st.session_state.get("tab8_fs_enable_os", True)),
            "type": "survival",
            "label": st.session_state.get("tab8_fs_os_label", "Overall Survival") or "Overall Survival",
            "time_col": st.session_state.get("tab8_fs_os_time_col", ""),
            "censor_col": st.session_state.get("tab8_fs_os_censor_col", ""),
            "censor_flag_means_censored": bool(st.session_state.get("tab8_fs_os_censor_means_censored", False)),
        },
        {
            "enabled": bool(st.session_state.get("tab8_fs_enable_dfs", True)),
            "type": "survival",
            "label": st.session_state.get("tab8_fs_dfs_label", "Disease-Free Survival") or "Disease-Free Survival",
            "time_col": st.session_state.get("tab8_fs_dfs_time_col", ""),
            "censor_col": st.session_state.get("tab8_fs_dfs_censor_col", ""),
            "censor_flag_means_censored": bool(st.session_state.get("tab8_fs_dfs_censor_means_censored", False)),
        },
        {
            "enabled": bool(st.session_state.get("tab8_fs_enable_response", False)),
            "type": "binary",
            "multi_mode": bool(st.session_state.get("tab8_fs_response_multi_mode", False)),
            "multi_cols": st.session_state.get("tab8_fs_response_multi_cols", []),
            "label": st.session_state.get("tab8_fs_response_label", "Response") or "Response",
            "group_col": st.session_state.get("tab8_fs_response_col", ""),
            "positive_value": st.session_state.get("tab8_fs_response_positive_value", ""),
        },
    ]

    config = {
        "source_csv": unified_csv_path,
        "columns": columns,
        "selected_phenotypes": st.session_state.get("tab8_fs_selected_phenotypes", []),
        "excluded_phenotypes": st.session_state.get("tab8_fs_excluded_phenotypes", []),
        "feature_families": st.session_state.get("tab8_fs_feature_families", []),
        "row_filters": row_filters,
        "endpoints": endpoints,
        "screening": {
            "outer_cap_mode": st.session_state.get("tab8_fs_outer_cap_mode", "percentile"),
            "zone_geometry": st.session_state.get("tab8_fs_zone_geometry", "cumulative"),
            "outer_cap_percentile": st.session_state.get("tab8_fs_outer_cap_percentile", 95.0),
            "outer_cap_value": st.session_state.get("tab8_fs_outer_cap_value", 300.0),
            "inner_cap_percentile": st.session_state.get("tab8_fs_inner_cap_percentile", 5.0),
            "n_candidate_widths": st.session_state.get("tab8_fs_n_candidate_widths", 15),
            "round_width_to": st.session_state.get("tab8_fs_round_width_to", 5.0),
            "max_width": st.session_state.get("tab8_fs_max_width", 0.0),
            "exclude_negative_distances": bool(st.session_state.get("tab8_fs_exclude_negative", True)),
            "use_micron_per_pixel": bool(st.session_state.get("tab8_fs_use_micron_per_pixel", True)),
            "micron_per_pixel": st.session_state.get("tab8_fs_micron_per_pixel", 0.5083),
            "min_cells_per_zone": st.session_state.get("tab8_fs_min_cells_per_zone", 20),
            "min_cells_for_feature": st.session_state.get("tab8_fs_min_cells_for_feature", 10),
            "ratio_pseudocount": st.session_state.get("tab8_fs_ratio_pseudocount", 1.0),
            "log_transform_ratios": bool(st.session_state.get("tab8_fs_log_transform_ratios", True)),
            "auto_interaction_distance": bool(st.session_state.get("tab8_fs_auto_interaction_distance", True)),
            "auto_interaction_percentile": st.session_state.get("tab8_fs_auto_interaction_percentile", 95.0),
            "allow_self_interaction_features": bool(st.session_state.get("tab8_fs_allow_self_interaction_features", False)),
            "use_adaptive_interaction_radius": bool(st.session_state.get("tab8_fs_use_adaptive_interaction_radius", True)),
            "adaptive_radius_direct_neighbours_only": bool(st.session_state.get("tab8_fs_adaptive_radius_direct_neighbours_only", True)),
            "adaptive_radius_min_pct": st.session_state.get("tab8_fs_adaptive_radius_min_pct", 0.1),
            "adaptive_radius_max_pct": st.session_state.get("tab8_fs_adaptive_radius_max_pct", 99.0),
            "adaptive_radius_max_sample": st.session_state.get("tab8_fs_adaptive_radius_max_sample", 500),
            "dyad_triad_permutation_n_iter": st.session_state.get("tab8_fs_dyad_triad_permutation_n_iter", 1000),
            "min_obs_motif_count": st.session_state.get("tab8_fs_min_obs_motif_count", 3),
            "permutation_chunk_size": st.session_state.get("tab8_fs_permutation_chunk_size", 64),
            "permutation_use_gpu": bool(st.session_state.get("tab8_fs_permutation_use_gpu", True)),
            "n_jobs": st.session_state.get("tab8_fs_n_jobs", 1),
            "dyad_min_dist_um": st.session_state.get("tab8_fs_dyad_min_um", 0.0),
            "dyad_max_dist_um": st.session_state.get("tab8_fs_dyad_max_um", 30.0),
            "triad_min_dist_um": st.session_state.get("tab8_fs_triad_min_um", 0.0),
            "triad_max_dist_um": st.session_state.get("tab8_fs_triad_max_um", 30.0),
            "triad_mode": st.session_state.get("tab8_fs_triad_mode", "all"),
            "triad_definitions": _parse_triads_text(st.session_state.get("tab8_fs_triads_custom_text", "")),
            "nn_k": st.session_state.get("tab8_fs_nn_k", 3),
            "area_use_median": bool(st.session_state.get("tab8_fs_area_use_median", False)),
            "area_percentile": st.session_state.get("tab8_fs_area_percentile", 95.0),
            "min_hole_size_px2": st.session_state.get("tab8_fs_min_hole_size_px2", 0.0),
            "patient_aggregation": st.session_state.get("tab8_fs_patient_aggregation", "median"),
            "split_method": st.session_state.get("tab8_fs_split_method", "median"),
            "quantile_low": st.session_state.get("tab8_fs_quantile_low", 0.25),
            "quantile_high": st.session_state.get("tab8_fs_quantile_high", 0.75),
            "custom_cutpoint": st.session_state.get("tab8_fs_custom_cutpoint", 0.0),
            "min_patients_per_group": st.session_state.get("tab8_fs_min_patients_per_group", 5),
            "show_ci": bool(st.session_state.get("tab8_fs_show_ci", False)),
            "show_excluded_km_curve": bool(st.session_state.get("tab8_fs_show_excluded_km_curve", False)),
            "volcano_label_discovery": bool(st.session_state.get("tab8_fs_volcano_label_discovery", False)),
            "volcano_label_nominal": bool(st.session_state.get("tab8_fs_volcano_label_nominal", False)),
            "volcano_label_other_top_n": int(st.session_state.get("tab8_fs_volcano_label_other_top_n", 3)),
            "time_unit": st.session_state.get("tab8_fs_time_unit", "Months"),
            "fdr_mode": st.session_state.get("tab8_fs_fdr_mode", "per_family"),
            "fdr_alpha": st.session_state.get("tab8_fs_fdr_alpha", 0.05),
            "fdr_alpha_discovery": st.session_state.get("tab8_fs_fdr_alpha_discovery", 0.10),
            "top_k_for_bootstrap": st.session_state.get("tab8_fs_top_k_for_bootstrap", 5),
            "bootstrap_iterations": st.session_state.get("tab8_fs_bootstrap_iterations", 30),
            "bootstrap_sampling_unit": st.session_state.get("tab8_fs_bootstrap_sampling_unit", "patient"),
            "bootstrap_patient_fraction": st.session_state.get("tab8_fs_bootstrap_patient_fraction", 0.80),
            "bootstrap_random_seed": st.session_state.get("tab8_fs_bootstrap_random_seed", 20260407),
            "feature_bootstrap_neighbor_steps": st.session_state.get("tab8_fs_feature_bootstrap_neighbor_steps", 1),
            "feature_stability_top_n": st.session_state.get("tab8_fs_feature_stability_top_n", 25),
            "min_patients_with_zone": st.session_state.get("tab8_fs_min_patients_with_zone", 12),
        },
        "representatives": {
            "raw_image_dir": st.session_state.get("tab8_fs_raw_image_dir", ""),
            "max_examples_per_group": st.session_state.get("tab8_fs_repr_per_group", 3),
            "thumbnail_padding_px": st.session_state.get("tab8_fs_thumbnail_padding_px", 120),
            "thumbnail_max_dim": st.session_state.get("tab8_fs_thumbnail_max_dim", 700),
        },
    }
    return config


def _validate_config(config: dict[str, Any], df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    columns = config["columns"]
    required_columns = [columns["patient"], columns["x"], columns["y"], columns["phenotype"], columns["distance"]]
    if not all(required_columns):
        errors.append("Patient, X, Y, phenotype, and distance columns are required.")
    for column in required_columns:
        if column and column not in df.columns:
            errors.append(f"Column not found: {column}")
    selected_phenotypes = [str(value) for value in config.get("selected_phenotypes", []) if str(value)]
    excluded_phenotypes = {str(value) for value in config.get("excluded_phenotypes", []) if str(value)}
    effective_selected_phenotypes = [value for value in selected_phenotypes if value not in excluded_phenotypes]
    if not selected_phenotypes:
        errors.append("Select at least one phenotype to screen.")
    elif not effective_selected_phenotypes:
        errors.append("At least one selected phenotype must remain after applying the excluded-phenotype list.")
    if not config.get("feature_families"):
        errors.append("Enable at least one feature family.")

    enabled_endpoint = False
    for endpoint in config.get("endpoints", []):
        if not endpoint.get("enabled", False):
            continue
        enabled_endpoint = True
        if endpoint.get("type") == "survival":
            if not endpoint.get("time_col") or not endpoint.get("censor_col"):
                errors.append(f"Endpoint '{endpoint.get('label', 'survival')}' needs both time and censor columns.")
        elif endpoint.get("type") == "binary":
            if endpoint.get("multi_mode", False):
                if not endpoint.get("multi_cols"):
                    errors.append(f"Endpoint '{endpoint.get('label', 'binary')}' needs at least one clinical column selected in multi-column mode.")
            else:
                if not endpoint.get("group_col"):
                    errors.append(f"Endpoint '{endpoint.get('label', 'binary')}' needs a grouping column.")
    if not enabled_endpoint:
        errors.append("Enable at least one endpoint (OS, DFS, or response).")
    return errors


def _render_column_mapping(df: pd.DataFrame) -> None:
    columns = df.columns.tolist()
    all_opts = [""] + columns
    st.markdown("### 1. Data Mapping")
    st.caption("**Which column holds what.** Point the screen at your cell coordinates, patient ID, phenotype labels and the distance-to-annotation column. Nothing is measured yet.")
    
    mapping_tabs = st.tabs(["Core Identifiers", "Phenotypes", "Spatial Context & Subgrouping"])
    
    with mapping_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            _stateful_selectbox("Patient column", all_opts, "tab8_fs_patient_col", default=st.session_state.get("tab8_fs_patient_col", ""), help="Patient identifier used for aggregation across multiple ROIs or batches.")
            _stateful_selectbox("Batch / image column", all_opts, "tab8_fs_batch_col", default=st.session_state.get("tab8_fs_batch_col", ""), help="Image or batch identifier used to resolve representative ROI thumbnails and per-region grouping.")
            _stateful_selectbox("ROI column", all_opts, "tab8_fs_roi_col", default=st.session_state.get("tab8_fs_roi_col", ""), help="Optional ROI identifier. Leave blank only if every batch corresponds to a single spatial unit.")
        with c2:
            _stateful_selectbox("X coordinate", all_opts, "tab8_fs_x_col", default=st.session_state.get("tab8_fs_x_col", ""), help="Cell X coordinate in pixels.")
            _stateful_selectbox("Y coordinate", all_opts, "tab8_fs_y_col", default=st.session_state.get("tab8_fs_y_col", ""), help="Cell Y coordinate in pixels.")
            
    with mapping_tabs[1]:
        _stateful_selectbox("Phenotype column", all_opts, "tab8_fs_phenotype_col", default=st.session_state.get("tab8_fs_phenotype_col", ""), help="Phenotype labels used to compute density, dyads, triads, ratios, and related spatial features.")
        phenotype_col = st.session_state.get("tab8_fs_phenotype_col", "")
        if phenotype_col and phenotype_col in df.columns:
            options = _safe_unique(df[phenotype_col])
            _stateful_multiselect("Phenotypes to screen", options=options, key="tab8_fs_selected_phenotypes", default=st.session_state.get("tab8_fs_selected_phenotypes", []), help="Only the selected phenotypes participate in feature calculation.")
            _stateful_multiselect(
                "Phenotypes to exclude",
                options=options,
                key="tab8_fs_excluded_phenotypes",
                default=st.session_state.get("tab8_fs_excluded_phenotypes", []),
                help="Optional exclusion applied after the inclusion list. Excluded phenotypes are removed from both discovery and validation, even if they are part of the selected phenotype list.",
            )

    with mapping_tabs[2]:
        mc1, mc2 = st.columns(2)
        with mc1:
            _stateful_selectbox("Distance to Annotation", all_opts, "tab8_fs_distance_col", default=st.session_state.get("tab8_fs_distance_col", ""), help="Continuous column storing the distance of each cell to an annotation or ROI boundary (e.g. distance to tumor border). This is used to build candidate proximity windows anchored at 0 (e.g. 0 to 50 microns).")
            _stateful_selectbox("Context / compartment column", all_opts, "tab8_fs_context_col", default=st.session_state.get("tab8_fs_context_col", ""), help="Optional categorical/compartment column (e.g. 'tumor_bed') to restrict cell screening to specific areas. If set, you can select which values to keep below.")
            
            context_col = st.session_state.get("tab8_fs_context_col", "")
            if context_col and context_col in df.columns:
                _stateful_multiselect(
                    "Context values to keep",
                    options=_safe_unique(df[context_col]),
                    key="tab8_fs_context_values",
                    default=st.session_state.get("tab8_fs_context_values", []),
                    help="Filter rows by keeping only these compartment values (e.g., keeping only '1' or 'tumor_bed'). Leave empty to keep all rows in the context column.",
                )
        with mc2:
            _stateful_multiselect("Stratify by columns", options=columns, key="tab8_fs_stratify_cols", default=st.session_state.get("tab8_fs_stratify_cols", []), help="Optional subgrouping columns (e.g. timepoint, arm, cohort). Each stratum is screened independently. If you are doing multi-column Response screening, do NOT fill this box; stratification will be handled automatically.")
            
            _stateful_multiselect(
                "Additional dataset filters",
                options=columns,
                key="tab8_fs_row_filter_cols",
                default=st.session_state.get("tab8_fs_row_filter_cols", []),
                help="Optional high-level filters applied before screening, for example keeping one cohort, staining panel, or sample subset.",
            )
            filter_cols = st.session_state.get("tab8_fs_row_filter_cols", [])
            for filter_col in filter_cols:
                _stateful_multiselect(
                    f"Values for {filter_col}",
                    options=_safe_unique(df[filter_col]),
                    key=f"tab8_fs_filter_values__{filter_col}",
                    default=st.session_state.get(f"tab8_fs_filter_values__{filter_col}", []),
                    help=f"Keep only the selected values from {filter_col}.",
                )


def _render_feature_controls() -> None:
    st.markdown("### 2. Feature Families — *what* to measure")
    st.caption("Which kinds of spatial quantity to compute per region: how much of a cell type (density, proportion), how mixed the tissue is (diversity), which cell types sit together (dyad, triad), and how close they are (nn_distance).")
    st.caption("These controls start from the reference script defaults. If this CSV already had saved Tab 8 settings, those saved values are restored until you use the reset button at the top of the tab.")
    
    feature_tabs = st.tabs(["Feature Selection & Density", "Abundance Ratios", "Motif Topology (Dyads & Triads)", "Nearest Neighbors & Resolution"])
    
    with feature_tabs[0]:
        _stateful_multiselect(
            "Feature families to screen",
            options=["density", "proportion", "ratio", "dyad", "triad", "diversity", "nn_distance"],
            key="tab8_fs_feature_families",
            default=st.session_state.get("tab8_fs_feature_families", []),
            help="These are the screening methodologies. Density and proportion measure abundance, ratio compares phenotype balance, dyad and triad capture short-range topology, diversity measures composition complexity, and nearest-neighbor distance measures spatial separation.",
        )
        c1, c2 = st.columns(2)
        with c1:
            _stateful_number_input("Minimum cells per zone", key="tab8_fs_min_cells_per_zone", default=20, min_value=1, help="If a zone has fewer than this many cells total, the zone is skipped. This avoids running statistics on tiny regions that are mostly noise.")
        with c2:
            _stateful_number_input("Minimum cells per feature phenotype", key="tab8_fs_min_cells_for_feature", default=10, min_value=1, help="A feature is skipped unless at least this many cells of the needed phenotype are present in that zone. Example: if this is 10 and a zone has only 3 CD8 cells, CD8-based features are not calculated there.")
            
    with feature_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            _stateful_number_input("Ratio pseudocount", key="tab8_fs_ratio_pseudocount", default=1.0, min_value=0.0, step=0.5, format="%.1f", help="Added to both numerator and denominator before computing the ratio, providing additive smoothing for sparse populations. Default of 1 (Laplace smoothing) prevents extreme log2 values when one phenotype is at the minimum cell threshold.")
        with c2:
            _stateful_checkbox("Log2-transform ratios", key="tab8_fs_log_transform_ratios", default=True, help="Ratios can become very large when one phenotype is rare. Turning this on makes opposite changes easier to compare. Example: a 2x increase and a 1/2x decrease become mirror images instead of one side dominating the scale.")
            
    with feature_tabs[2]:
        use_micron_per_pixel = bool(st.session_state.get("tab8_fs_use_micron_per_pixel", True))
        auto_interaction_distance = bool(st.session_state.get("tab8_fs_auto_interaction_distance", True))
        distance_unit = "um" if use_micron_per_pixel else "px"
        
        c1, c2 = st.columns(2)
        with c1:
            _stateful_checkbox("Auto-calculate dyad and triad distances", key="tab8_fs_auto_interaction_distance", default=True, help="This follows the reference workflow. The app keeps the minimum distance at 0 and estimates the maximum interaction distance from the cell spacing in your dataset.")
            _stateful_number_input("Auto distance percentile", key="tab8_fs_auto_interaction_percentile", default=95.0, min_value=50.0, max_value=100.0, step=1.0, disabled=not auto_interaction_distance, help="95 matches the reference workflow. Higher values allow a slightly wider automatic interaction distance.")
            _stateful_checkbox("Allow same-phenotype pairings (dyad, triad, NN A→A)", key="tab8_fs_allow_self_interaction_features", default=False, help="By default, same-phenotype features are excluded: Dyad A-A, Triad A-A-A, and NN distance A→A. Note that when A→A NN distance is computed, the source cell is automatically excluded from its own neighbourhood (self is never counted as a neighbour). This flag only controls whether A→A features appear in the screening results at all.")
        with c2:
            _stateful_number_input(f"Dyad min distance ({distance_unit})", key="tab8_fs_dyad_min_um", default=0.0, min_value=0.0, step=1.0, disabled=auto_interaction_distance, help="Usually leave this at 0. In the reference workflow, automatic mode fixes the minimum distance at 0.")
            _stateful_number_input(f"Dyad max distance ({distance_unit})", key="tab8_fs_dyad_max_um", default=30.0, min_value=0.1, step=1.0, disabled=auto_interaction_distance, help="Used only when automatic interaction distance is turned off. Larger values make it easier for two cells to count as an interaction.")
            _stateful_number_input(f"Triad min distance ({distance_unit})", key="tab8_fs_triad_min_um", default=0.0, min_value=0.0, step=1.0, disabled=auto_interaction_distance, help="Usually leave this at 0. In the reference workflow, automatic mode fixes the minimum distance at 0.")
            _stateful_number_input(f"Triad max distance ({distance_unit})", key="tab8_fs_triad_max_um", default=30.0, min_value=0.1, step=1.0, disabled=auto_interaction_distance, help="Used only when automatic interaction distance is turned off. Larger values make it easier for three nearby cells to count as a triad.")

        if auto_interaction_distance:
            st.info(
                f"Automatic interaction distance is ON. The app will estimate the dyad and triad cutoff from the data at run time, using the {int(st.session_state.get('tab8_fs_auto_interaction_percentile', 95.0))}th percentile of within-region nearest-neighbor spacing. Distances will be reported in {distance_unit}."
            )
        else:
            st.caption(f"Automatic interaction distance is OFF. The dyad and triad cutoffs above are treated as manual {distance_unit} values.")
            
        st.markdown("**Permutation-based significance (dyad & triad)**")
        perm_col1, perm_col2, perm_col3, perm_col4 = st.columns(4)
        with perm_col1:
            _stateful_checkbox(
                "Per-ROI adaptive radius",
                key="tab8_fs_use_adaptive_interaction_radius",
                default=True,
                help="When ON, the interaction window for each ROI is determined from the local cell-cell distance distribution rather than a fixed cutoff. This adjusts dynamically for dense vs. sparse regions. Recommended: Checked (ON).",
            )
            _stateful_checkbox(
                "Direct neighbours only (1×p95 NN)",
                key="tab8_fs_adaptive_radius_direct_neighbours_only",
                default=True,
                disabled=not bool(st.session_state.get("tab8_fs_use_adaptive_interaction_radius", True)),
                help="When ON, limits interactions to immediate spatial neighbors. When OFF, extends up to ~3× that distance. Recommended: Checked (ON) for direct contact signaling; Unchecked (OFF) for short-range paracrine signaling.",
            )
        with perm_col2:
            _stateful_number_input(
                "Permutation iterations",
                key="tab8_fs_dyad_triad_permutation_n_iter",
                default=1000,
                min_value=0,
                step=100,
                help="Number of label shuffles to build the null distribution. Recommended: 1000 for final results, 100-200 for rapid testing, or 0 to disable permutation completely (reverts to raw counts).",
            )
            _gpu_name = gpu_device_name()
            _stateful_checkbox(
                "Use GPU for permutation testing when available",
                key="tab8_fs_permutation_use_gpu",
                default=True,
                disabled=not GPU_AVAILABLE,
                help=(
                    "Run the dyad/triad label-shuffling null on the GPU. The motifs found and the "
                    "statistic are identical to the CPU path -- only the random stream differs, exactly "
                    "as between two CPU runs with different seeds -- but it is roughly 60x faster on an "
                    "A100. That speed is what makes a defensible permutation count affordable: at 20,000 "
                    "shuffles the largest triad Z-scores still move by about +/-0.5 between runs, so low "
                    "counts leave real Monte-Carlo noise in the rankings. Falls back to CPU automatically."
                    + (f"\n\nDetected GPU: {_gpu_name}." if _gpu_name else "\n\nNo GPU detected on this node.")
                ),
            )
            _stateful_number_input(
                "Min motif instances per ROI",
                key="tab8_fs_min_obs_motif_count",
                default=3,
                min_value=0,
                step=1,
                help="Minimum observed motif instances in an ROI to compute statistics. Recommended: 3 (avoids noisy Z-scores from sparse regions).",
            )
        with perm_col3:
            _stateful_number_input(
                "Adaptive min percentile",
                key="tab8_fs_adaptive_radius_min_pct",
                default=0.1,
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                format="%.1f",
                disabled=not bool(st.session_state.get("tab8_fs_use_adaptive_interaction_radius", True)),
                help="Lower percentile threshold of cell distances. Recommended: 0.1 (filters out overlapping segmentation artifacts).",
            )
        with perm_col4:
            _stateful_number_input(
                "Adaptive max percentile",
                key="tab8_fs_adaptive_radius_max_pct",
                default=99.0,
                min_value=50.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
                disabled=not bool(st.session_state.get("tab8_fs_use_adaptive_interaction_radius", True)),
                help="Upper percentile threshold for adaptive cell neighborhood. Recommended: 95.0 or 99.0.",
            )
        n_perm_val = int(st.session_state.get("tab8_fs_dyad_triad_permutation_n_iter", 1000))
        use_adap = bool(st.session_state.get("tab8_fs_use_adaptive_interaction_radius", True))
        direct_nn = bool(st.session_state.get("tab8_fs_adaptive_radius_direct_neighbours_only", True))
        if n_perm_val > 0:
            if use_adap:
                if direct_nn:
                    _adap_desc = "Adaptive radius: direct neighbours only (1×p95 NN — cells must be immediately adjacent)."
                else:
                    _adap_desc = (
                        f"Adaptive radius: {st.session_state.get('tab8_fs_adaptive_radius_min_pct', 0.1):.1f}–"
                        f"{st.session_state.get('tab8_fs_adaptive_radius_max_pct', 99.0):.0f}th percentile of within-ROI pairwise distances (~3×p95 NN window)."
                    )
            else:
                _adap_desc = "Fixed distance window used."
            st.caption(
                f"Permutation ON ({n_perm_val} iterations/ROI). Feature value = Z-score vs. null. "
                f"Z > 1.96 indicates significant enrichment. {_adap_desc}"
            )
        else:
            st.caption("Permutation OFF. Feature value = raw dyad/triad proportion (original behaviour).")

        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            _stateful_number_input(
                "Parallel threads (n_jobs)",
                key="tab8_fs_n_jobs",
                default=1,
                min_value=0,
                step=1,
                help="Number of CPU cores to use. Recommended: 0 (uses all available cores) or 4-8 on shared compute nodes. Set to 1 to run sequentially.",
            )
        with perf_col2:
            _stateful_number_input(
                "Permutation chunk size",
                key="tab8_fs_permutation_chunk_size",
                default=64,
                min_value=8,
                step=8,
                help="Label shuffles processed per vector batch. Recommended: 64 (higher speeds up permutations on systems with high RAM).",
            )

        triad_col1, triad_col2 = st.columns([1, 2])
        with triad_col1:
            _stateful_selectbox("Triad mode", options=["all", "custom"], key="tab8_fs_triad_mode", default=st.session_state.get("tab8_fs_triad_mode", "all"), help="'all' screens every possible 3-phenotype motif. 'custom' limits triads to specific user-provided combinations. Recommended: 'all'.")
        with triad_col2:
            if st.session_state.get("tab8_fs_triad_mode", "all") == "custom":
                _stateful_text_area(
                    "Custom triads (one triad per line, comma-separated)",
                    key="tab8_fs_triads_custom_text",
                    default=st.session_state.get("tab8_fs_triads_custom_text", ""),
                    height=90,
                    help="Example:\nCD8,Treg,Macrophage\nTumor,Fibroblast,Endothelial",
                )

    with feature_tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            _stateful_number_input("Nearest-neighbor k", key="tab8_fs_nn_k", default=3, min_value=1, help="For the nearest-neighbor distance feature, this is how many neighbours to average per source cell. When source and target are the same phenotype (A→A), the cell itself is automatically excluded from the query, so you always get k true neighbours.")
        with c2:
            _stateful_checkbox("Use micron-per-pixel conversion", key="tab8_fs_use_micron_per_pixel", default=True, help="Turn this on only if you know the image scale. When it is on, distances are shown in microns. When it is off, the app keeps distances in raw pixels, which is useful for public datasets that do not report micron-per-pixel.")
            _stateful_number_input("Microns per pixel", key="tab8_fs_micron_per_pixel", default=0.5083, min_value=0.0001, step=0.0001, format="%.4f", disabled=not bool(st.session_state.get("tab8_fs_use_micron_per_pixel", True)), help="Reference default is 0.5083. This number is only used when micron-per-pixel conversion is turned on.")


def _render_endpoint_controls(df: pd.DataFrame) -> None:
    columns = [""] + df.columns.tolist()
    st.markdown("### 3. Endpoints — what to measure it *against*")
    st.caption("The clinical outcome each feature is tested against. Every enabled endpoint is screened independently and gets its own multiple-testing correction.")
    os_exp, dfs_exp, resp_exp = st.tabs(["Overall Survival", "Disease-Free Survival", "Response / Binary Outcome"])
    with os_exp:
        _stateful_checkbox("Enable OS screening", key="tab8_fs_enable_os", default=True, help="Turns on survival screening using a time column and censor column.")
        _stateful_text_input("OS label", key="tab8_fs_os_label", default="Overall Survival", help="Label shown in tables and plots.")
        _stateful_selectbox("OS time column", columns, key="tab8_fs_os_time_col", default=st.session_state.get("tab8_fs_os_time_col", ""), help="Continuous survival time column.")
        _stateful_selectbox("OS censor column", columns, key="tab8_fs_os_censor_col", default=st.session_state.get("tab8_fs_os_censor_col", ""), help="Censor or event flag. For this project, the default assumes 1 means event and 0 means censored.")
        _stateful_checkbox("OS censor flag means censored", key="tab8_fs_os_censor_means_censored", default=False, help="If checked, event is computed as 1 - censor flag. Leave it unchecked when your dataset already stores event=1.")
    with dfs_exp:
        _stateful_checkbox("Enable DFS screening", key="tab8_fs_enable_dfs", default=True, help="Turns on disease-free survival screening with its own time and censor columns.")
        _stateful_text_input("DFS label", key="tab8_fs_dfs_label", default="Disease-Free Survival", help="Label shown in tables and plots.")
        _stateful_selectbox("DFS time column", columns, key="tab8_fs_dfs_time_col", default=st.session_state.get("tab8_fs_dfs_time_col", ""), help="Continuous DFS time column.")
        _stateful_selectbox("DFS censor column", columns, key="tab8_fs_dfs_censor_col", default=st.session_state.get("tab8_fs_dfs_censor_col", ""), help="DFS censor or event flag.")
        _stateful_checkbox("DFS censor flag means censored", key="tab8_fs_dfs_censor_means_censored", default=False, help="If checked, event is computed as 1 - censor flag. Leave it unchecked when your dataset already stores event=1.")
    with resp_exp:
        _stateful_checkbox("Enable binary endpoint screening", key="tab8_fs_enable_response", default=False, help="Use this if the dataset lacks OS / DFS but has response, treatment benefit, recurrence yes/no, or another binary outcome.")
        if st.session_state.get("tab8_fs_enable_response", False):
            _stateful_checkbox("Enable multi-column comprehensive screening", key="tab8_fs_response_multi_mode", default=False, help="Analyze multiple clinical/categorical columns. It will automatically test all pairwise comparisons and all combinations of stratifications from the chosen columns.")
            if st.session_state.get("tab8_fs_response_multi_mode", False):
                _stateful_multiselect(
                    "Select columns for comparison & stratification",
                    options=[col for col in columns if col],
                    key="tab8_fs_response_multi_cols",
                    default=st.session_state.get("tab8_fs_response_multi_cols", []),
                    help="E.g., Response, Treatment, Timepoint. The pipeline will compare categories within each column (like CR vs PR) both overall and stratified by other selected columns."
                )
            else:
                _stateful_text_input("Binary endpoint label", key="tab8_fs_response_label", default="Response", help="Label shown in tables and plots.")
                _stateful_selectbox("Binary grouping column", columns, key="tab8_fs_response_col", default=st.session_state.get("tab8_fs_response_col", ""), help="Categorical column used to split patients into two groups.")
                response_col = st.session_state.get("tab8_fs_response_col", "")
                response_options = _safe_unique(df[response_col]) if response_col and response_col in df.columns else []
                _stateful_selectbox("Positive / responder value", options=[""] + response_options, key="tab8_fs_response_positive_value", default=st.session_state.get("tab8_fs_response_positive_value", ""), help="If set, the chosen value is compared against all other values. Leave empty only when the grouping column already has exactly two values.")


def _render_screening_controls() -> None:
    st.markdown("### 4. Windowing, Aggregation, and Statistics — *where*, and how hard to test")
    st.caption("Three decisions: which distance zones around the annotation to scan, how several regions from one patient become a single patient value, and how strictly a result must hold up "
               "before it counts. The statistics block at the bottom applies to everything above it.")
    
    stat_tabs = st.tabs(["Distance Windowing", "Patient Aggregation & Splitting", "Feature Stability & Bootstrap"])
    
    with stat_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            _stateful_selectbox(
                "Zone geometry",
                options=["cumulative", "ring"],
                key="tab8_fs_zone_geometry",
                default=st.session_state.get("tab8_fs_zone_geometry", "cumulative"),
                help=(
                    "What each scanned width selects.\n\n"
                    "**cumulative** (default) - every zone runs from the annotation boundary outward "
                    "(0 to width). Zones are nested, so a signal living in the innermost band keeps "
                    "reappearing, progressively diluted, at every larger width. This is the right mode "
                    "for *how wide does the band have to be before the effect appears, and at what "
                    "width does it wash out* - it is what the 'Feature stability across ranges' "
                    "heatmap reads.\n\n"
                    "**ring** - each zone is the annulus between the previous width and this one, so "
                    "the columns are disjoint bands of tissue. This is the right mode for *at which "
                    "distance does the effect actually live*, because a hit at one ring is not "
                    "inherited from the rings inside it. Rings hold fewer cells per zone, so raise "
                    "'Minimum cells per zone' accordingly."
                ),
            )
            _stateful_selectbox("Outer cap mode", options=["percentile", "value"], key="tab8_fs_outer_cap_mode", default=st.session_state.get("tab8_fs_outer_cap_mode", "percentile"), help="Choose whether the scan's maximum width is derived from the pooled distance distribution or specified directly.")
            _stateful_number_input("Outer cap percentile", key="tab8_fs_outer_cap_percentile", default=95.0, min_value=50.0, max_value=100.0, step=1.0, help="Used when outer cap mode is percentile.")
            _stateful_number_input("Outer cap value", key="tab8_fs_outer_cap_value", default=300.0, min_value=1.0, step=5.0, help="Used when outer cap mode is a fixed distance value.")
        with c2:
            _stateful_number_input("Inner cap percentile", key="tab8_fs_inner_cap_percentile", default=5.0, min_value=0.0, max_value=50.0, step=1.0, help="Lower percentile used to avoid extremely narrow zones with unstable counts.")
            _stateful_number_input("Candidate widths", key="tab8_fs_n_candidate_widths", default=12, min_value=1, help="Number of candidate proximal windows to evaluate between the inner and outer caps.")
            _stateful_number_input("Round widths to", key="tab8_fs_round_width_to", default=5.0, min_value=0.1, step=0.5, help="Rounds candidate window sizes to cleaner values for easier interpretation.")
            _stateful_number_input("Maximum width override", key="tab8_fs_max_width", default=0.0, min_value=0.0, step=5.0, help="Optional hard upper bound. Keep zero to disable.")
            _stateful_checkbox("Exclude negative distances", key="tab8_fs_exclude_negative", default=True, help="Recommended when the distance column encodes outside-tumor / inside-tumor with negative values and the current screen should focus on proximal distances from tumor outward.")
            
        if not bool(st.session_state.get("tab8_fs_exclude_negative", True)):
            st.info(
                "Signed scanning is enabled. Candidate cutpoints are now drawn from the pooled signed distance distribution, zero stays the center line, and the scan evaluates negative and positive windows separately as ranges such as 0 to -100 and 0 to 75."
            )
            
    with stat_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            _stateful_selectbox("Patient aggregation", options=["median", "mean", "max", "p75"], key="tab8_fs_patient_aggregation", default=st.session_state.get("tab8_fs_patient_aggregation", "median"), help="How multiple ROIs for the same patient are collapsed before endpoint testing.")
            _stateful_selectbox("High / low split method", options=["median", "quartiles", "tertiles", "optimal", "custom_cutpoint"], key="tab8_fs_split_method", default=st.session_state.get("tab8_fs_split_method", "median"), help="How patients are grouped for survival plots and screening statistics. 'optimal' tests multiple percentiles and selects the cutpoint that maximizes log-rank significance.")
        with c2:
            _stateful_number_input("Low quantile", key="tab8_fs_quantile_low", default=0.25, min_value=0.0, max_value=0.5, step=0.05, help="Only used for quartile-based splitting.")
            _stateful_number_input("High quantile", key="tab8_fs_quantile_high", default=0.75, min_value=0.5, max_value=1.0, step=0.05, help="Only used for quartile-based splitting.")
            _stateful_number_input("Custom cutpoint", key="tab8_fs_custom_cutpoint", default=0.0, step=0.1, help="Only used for a custom cutpoint split.")
            
    with stat_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            _stateful_number_input(
                "Neighboring-range tolerance (steps)",
                key="tab8_fs_feature_bootstrap_neighbor_steps",
                default=1,
                min_value=0,
                max_value=6,
                step=1,
                help="Feature bootstrap recovery counts a feature as rediscovered if it reappears in the same range or within this many adjacent candidate ranges on either side.",
            )
            _stateful_number_input(
                "Default feature heatmap top N",
                key="tab8_fs_feature_stability_top_n",
                default=25,
                min_value=5,
                max_value=100,
                step=1,
                help="Default number of top-ranked stable features shown in the feature heatmap and combined export panel.",
            )
        with c2:
            _stateful_selectbox(
                "Bootstrap resampling unit",
                options=["patient", "roi"],
                key="tab8_fs_bootstrap_sampling_unit",
                default=st.session_state.get("tab8_fs_bootstrap_sampling_unit", "patient"),
                help="Patient resampling re-samples patients after ROI features are aggregated. ROI resampling re-samples ROI-level feature rows first, then re-aggregates to patient values inside each bootstrap iteration.",
            )
        st.caption(
            "These feature-stability controls now feed both the width-ranking logic and the saved/exported feature-level summary figures. "
            "Bootstrap resampling can now be done either at the patient level or at the ROI level before patient aggregation."
        )

    # These controls govern every feature family and every zone width, so they
    # deliberately sit outside the three sub-tabs above. Without this heading
    # they render directly beneath whichever sub-tab happens to be open and read
    # as if they belonged to it -- which is why they look duplicated.
    st.markdown("---")
    st.markdown("#### Statistics and significance")
    st.caption(
        "Shared by **all** feature families and **all** zone widths above, which is why this block "
        "sits outside the three tabs rather than inside any one of them. Changing anything here "
        "changes every result in the run."
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Group sizes and survival display**")
        _stateful_number_input("Minimum patients per group", key="tab8_fs_min_patients_per_group", default=5, min_value=2, help="Endpoint tests are skipped unless both groups meet this threshold.")
        _stateful_checkbox("Show survival confidence intervals", key="tab8_fs_show_ci", default=False, help="Currently kept off for the publication-style Kaplan-Meier panel so the survival curves render without shaded confidence bands.")
        _stateful_checkbox("Show excluded-patient grey KM curve", key="tab8_fs_show_excluded_km_curve", default=False, help="When enabled, excluded patients with endpoint data are drawn as a separate grey survival curve. When disabled, they remain only in the audit/output table and are removed from the figures.")
        _stateful_text_input("Time unit label", key="tab8_fs_time_unit", default="Months", help="Axis label for survival plots.")
    with d2:
        st.markdown("**Multiple-testing correction**")
        _stateful_selectbox("FDR mode", options=["per_family", "global"], key="tab8_fs_fdr_mode", default=st.session_state.get("tab8_fs_fdr_mode", "per_family"), help="Per-family FDR is more permissive when families differ greatly in feature count. Global FDR is stricter.")
        _stateful_number_input("FDR alpha", key="tab8_fs_fdr_alpha", default=0.05, min_value=0.001, max_value=0.5, step=0.01, help="Strict significance threshold after multiple-testing correction.")
        _stateful_number_input("Discovery FDR alpha", key="tab8_fs_fdr_alpha_discovery", default=0.10, min_value=0.001, max_value=0.5, step=0.01, help="Optional looser discovery threshold for exploratory analyses.")
    with d3:
        st.markdown("**Virtual tissue area (density denominator)**")
        _stateful_checkbox("Use median area buffer", key="tab8_fs_area_use_median", default=False, help="If enabled, virtual tissue area uses a median nearest-neighbor buffer instead of a percentile-based buffer.")
        _stateful_number_input("Area percentile", key="tab8_fs_area_percentile", default=95.0, min_value=50.0, max_value=100.0, step=1.0, help="Percentile used to derive the virtual tissue buffer when median mode is disabled.")
        _stateful_number_input("Minimum hole size (px^2)", key="tab8_fs_min_hole_size_px2", default=0.0, min_value=0.0, step=50.0, help="Small internal holes below this size are removed from the virtual tissue polygon.")

    with st.expander("Volcano plot labelling (display only -- does not affect any statistic)", expanded=False):
        volcano_col1, volcano_col2, volcano_col3 = st.columns(3)
        with volcano_col1:
            _stateful_checkbox(
                "Volcano labels: include discovery FDR",
                key="tab8_fs_volcano_label_discovery",
                default=False,
                help="FDR-significant labels are always shown. Turn this on to also label top-ranked discovery-FDR features.",
            )
        with volcano_col2:
            _stateful_checkbox(
                "Volcano labels: include nominal p",
                key="tab8_fs_volcano_label_nominal",
                default=False,
                help="Adds top-ranked nominal p < 0.05 features to the volcano labels.",
            )
        with volcano_col3:
            _stateful_number_input(
                "Volcano top N extra labels",
                key="tab8_fs_volcano_label_other_top_n",
                default=3,
                min_value=1,
                max_value=20,
                step=1,
                help="Maximum combined number of extra discovery-FDR and nominal-p labels to add when those options are enabled.",
            )
        st.caption("Volcano plots always label FDR-significant hits. Discovery-FDR and nominal-p labels are optional and share the same top-N cap above.")


def _render_output_controls(project_root: Path) -> None:
    st.markdown("### 5. Outputs")
    st.caption("Where results are written, and how figures are exported for publication.")
    c1, c2, c3 = st.columns(3)
    with c1:
        directory_input_with_browse(
            label="Output directory",
            text_key="tab8_fs_output_dir",
            default_value=st.session_state.get("tab8_fs_output_dir", str(project_root / "outputs" / "spatial_feature_screening")),
            browser_key_prefix="tab8_fs_output_dir",
            start_dir_fallback=str(project_root),
            help_text="Each screening run writes a timestamped folder with CSVs, summary figures, logs, and a manifest.",
        )
        directory_input_with_browse(
            label="Raw image directory",
            text_key="tab8_fs_raw_image_dir",
            default_value=st.session_state.get("tab8_fs_raw_image_dir", str(project_root / "outputs" / "registration" / "merged")),
            browser_key_prefix="tab8_fs_raw_image_dir",
            start_dir_fallback=str(project_root),
            help_text="Used to resolve representative ROI thumbnails from the batch / image column.",
        )
    with c1:
        st.markdown("**Publication export**")
        _stateful_multiselect(
            "Figure download formats",
            options=["png", "svg", "pdf", "tiff"],
            key="tab8_fs_figure_formats",
            default=["png"],
            help=(
                "Formats offered under every figure in this tab. **SVG** or **PDF** keep text and "
                "lines as vectors, so a journal can scale them without pixelation and you can "
                "restyle fonts in Illustrator or Inkscape -- pick one of those for final figures. "
                "**PNG** at 300+ dpi is fine for drafts and for figures containing image crops. "
                "**TIFF** is what some journals demand for raster submissions."
            ),
        )
        _stateful_number_input(
            "Figure DPI",
            key="tab8_fs_figure_dpi",
            default=300,
            min_value=72,
            max_value=1200,
            step=50,
            help=(
                "Resolution for raster downloads (PNG/TIFF). 300 dpi is the usual journal minimum "
                "for figures; 600 is often requested for line art. Ignored for the geometry of "
                "SVG/PDF, which are resolution-independent."
            ),
        )
        _stateful_checkbox(
            "Transparent figure background",
            key="tab8_fs_figure_transparent",
            default=False,
            help=(
                "Export with no background fill instead of white, so figures drop cleanly onto a "
                "coloured slide or poster. Leave off for journal submission, where a white "
                "background is normally expected."
            ),
        )
        _stateful_checkbox(
            "Prepare figure downloads automatically",
            key="tab8_fs_figure_eager_downloads",
            default=False,
            help=(
                "Off (recommended): each figure gets a 'Prepare download' button, and the export is "
                "rendered only when you ask for it, then cached for the session.\n\n"
                "On: every figure on the page is exported to every selected format on every rerun. "
                "That is roughly 0.3-1.0 s per figure per format -- and a single 300 dpi TIFF is about "
                "10 MB -- so with several figures the page can take minutes to redraw after any "
                "interaction. Turn this on only for a final export sweep."
            ),
        )

    with c2:
        _stateful_number_input("Representative ROIs per group", key="tab8_fs_repr_per_group", default=3, min_value=1, help="Number of high-value and low-value representative ROIs to preview for a selected hit.")
        _stateful_number_input("Thumbnail padding (px)", key="tab8_fs_thumbnail_padding_px", default=120, min_value=0, step=20, help="Padding around the selected ROI or hotspot when cropping preview thumbnails.")
    with c3:
        _stateful_number_input("Thumbnail max dimension", key="tab8_fs_thumbnail_max_dim", default=700, min_value=200, step=50, help="Maximum width or height for in-interface thumbnails.")


def _render_apply_controls(config: dict[str, Any], df: pd.DataFrame, project_root: Path, unified_csv_path: str | None) -> bool:
    config_text = json.dumps(config, sort_keys=True, default=str)
    applied_signature = st.session_state.get("_tab8_fs_applied_signature")
    is_dirty = applied_signature != config_text
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Apply configuration", key="tab8_fs_apply_btn", help="Validates and snapshots the current screening configuration without running the analysis."):
            errors = _validate_config(config, df)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.session_state["_tab8_fs_applied_signature"] = config_text
                st.session_state["_tab8_fs_applied_config"] = config
                st.session_state["tab8_fs_apply_note"] = "Configuration applied. You can now run the scan or load a previous run." 
                _log_terminal_event("Configuration applied")
                from spacemap_gui.persistence import save_active_project_state
                save_active_project_state("tab8_analysis")
                st.rerun()
    with c2:
        if st.button("Clear transient results", key="tab8_fs_clear_btn", help="Clears in-memory screening outputs and the live log panel without touching the saved files on disk."):
            for key in [
                "_tab8_fs_results",
                "_tab8_fs_loaded_run",
                "_tab8_fs_validation_results",
                "_tab8_fs_validation_influence_results",
                "_tab8_fs_validation_influence_signature",
                "_tab8_fs_logs",
                "_tab8_fs_progress",
                "_tab8_fs_progress_meta",
                "_tab8_fs_live_width_rows",
                "_tab8_fs_live_hit_rows",
                "_tab8_fs_live_hits_by_width",
                "_tab8_fs_recent_width_events",
                "_tab8_fs_live_preview_width",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
    with c3:
        st.caption("Unsaved changes pending." if is_dirty else "Current settings match the applied configuration.")
        note = st.session_state.get("tab8_fs_apply_note", "")
        if note:
            st.caption(note)
    return bool(st.session_state.get("_tab8_fs_applied_config"))


def _run_pipeline(raw_df: pd.DataFrame) -> None:
    config = st.session_state.get("_tab8_fs_applied_config")
    if not config:
        st.warning("Apply the configuration before running the feature screen.")
        return

    st.session_state["_tab8_fs_logs"] = []
    st.session_state["_tab8_fs_progress"] = {
        "stage": "initializing",
        "message": "Preparing feature screening run",
        "current": 0,
        "total": 1,
    }
    run_started_at = time.monotonic()
    st.session_state["_tab8_fs_progress_meta"] = {
        "run_started_at": run_started_at,
        "phase_group": "initializing",
        "phase_started_at": run_started_at,
    }
    st.session_state["_tab8_fs_live_width_rows"] = []
    st.session_state["_tab8_fs_live_hit_rows"] = []
    st.session_state["_tab8_fs_live_hits_by_width"] = {}
    st.session_state["_tab8_fs_recent_width_events"] = []
    st.session_state["_tab8_fs_live_candidate_widths"] = []
    st.session_state["_tab8_fs_live_distance_summary"] = pd.DataFrame()
    st.session_state["_tab8_fs_live_distance_histogram"] = pd.DataFrame()
    st.session_state["_tab8_fs_live_signed_width_scan"] = False
    st.session_state["_tab8_fs_live_render_nonce"] = 0
    st.session_state.pop("_tab8_fs_live_preview_width", None)
    _log_terminal_event("Feature screening started")
    with st.status("Running feature screening...", expanded=True) as status:
        progress_bar = st.progress(0.0, text="Preparing feature scan...")
        dashboard_placeholder = st.empty()
        consolidated_placeholder = st.empty()

        def _format_progress_value(value: Any, decimals: int = 2, as_percent: bool = False) -> str:
            numeric = _to_float(value)
            if not np.isfinite(numeric):
                return "-"
            if as_percent:
                return f"{float(numeric) * 100:.1f}%"
            return f"{float(numeric):.{int(decimals)}f}"

        def _format_duration(seconds: float | None) -> str:
            if seconds is None or not np.isfinite(seconds):
                return "-"
            whole_seconds = max(0, int(round(float(seconds))))
            minutes, secs = divmod(whole_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                return f"{hours}h {minutes}m"
            if minutes > 0:
                return f"{minutes}m {secs}s"
            return f"{secs}s"

        def _status_text(label: Any) -> str:
            mapping = {
                "significant_fdr": "FDR significant",
                "discovery_fdr": "FDR discovery",
                "nominal_p": "Nominal p<0.05",
                "not_significant": "Not significant",
            }
            return mapping.get(str(label or ""), str(label or "-") or "-")

        def _phase_group(stage: str) -> str:
            if stage.startswith("bootstrap"):
                return "bootstrap"
            if stage.startswith("width"):
                return "width"
            return stage

        def _estimate_eta(progress_state: dict[str, Any], meta: dict[str, Any], now: float) -> float | None:
            current = max(0, int(progress_state.get("current") or 0))
            total = max(0, int(progress_state.get("total") or 0))
            if total <= 0 or current <= 0 or current >= total:
                return None
            phase_started_at = float(meta.get("phase_started_at") or now)
            elapsed = max(0.0, now - phase_started_at)
            if elapsed <= 0:
                return None
            return (elapsed / float(current)) * float(total - current)

        def _upsert_live_width_row(progress_state: dict[str, Any]) -> None:
            width_value = _to_float(progress_state.get("width"))
            if not np.isfinite(width_value):
                return
            rows: list[dict[str, Any]] = [dict(row) for row in st.session_state.get("_tab8_fs_live_width_rows", [])]
            existing_index = next((idx for idx, row in enumerate(rows) if abs(float(row.get("zone_width", -1)) - float(width_value)) < 1e-9), None)
            target: dict[str, Any] = rows[existing_index] if existing_index is not None else {"zone_width": float(width_value)}
            for key in [
                "n_significant",
                "n_discovery",
                "best_fdr_q",
                "best_p_value",
                "patient_coverage_fraction",
                "n_patients_with_zone",
                "strongest_feature_family",
                "strongest_feature_label",
                "strongest_endpoint_label",
                "strongest_significance_label",
                "bootstrap_reference_recovery",
                "bootstrap_valid_iterations",
                "bootstrap_n_significant_mean",
            ]:
                if key in progress_state and progress_state.get(key) is not None:
                    target[key] = progress_state.get(key)
            if existing_index is None:
                rows.append(target)
            else:
                rows[existing_index] = target
            st.session_state["_tab8_fs_live_width_rows"] = rows

        def _best_widths_dataframe() -> pd.DataFrame:
            rows = st.session_state.get("_tab8_fs_live_width_rows", [])
            if not rows:
                return pd.DataFrame()
            live_df = pd.DataFrame(rows).copy()
            if live_df.empty:
                return live_df
            for column in ["zone_width", "n_significant", "n_discovery", "best_fdr_q", "best_p_value", "patient_coverage_fraction", "bootstrap_reference_recovery", "bootstrap_valid_iterations", "bootstrap_n_significant_mean"]:
                if column in live_df.columns:
                    live_df[column] = pd.to_numeric(live_df[column], errors="coerce")
            live_df = rank_candidates(live_df, use_bootstrap=True)
            live_df["Range"] = live_df["zone_width"].map(format_zone_label)
            live_df["FDR hits"] = live_df["n_significant"].fillna(0).astype(int)
            live_df["Discovery hits"] = live_df["n_discovery"].fillna(0).astype(int)
            live_df["Best FDR"] = live_df["best_fdr_q"].map(lambda value: _format_progress_value(value, decimals=3))
            live_df["Best p"] = live_df["best_p_value"].map(lambda value: _format_progress_value(value, decimals=3))
            live_df["Coverage"] = live_df["patient_coverage_fraction"].map(lambda value: _format_progress_value(value, as_percent=True))
            live_df["Bootstrap recovery"] = live_df.get("bootstrap_reference_recovery", pd.Series(np.nan, index=live_df.index)).map(lambda value: _format_progress_value(value))
            live_df["Strongest family"] = live_df.get("strongest_feature_family", pd.Series("", index=live_df.index)).fillna("").astype(str).str.replace("_", " ").str.title()
            live_df["Strongest hit"] = live_df.get("strongest_feature_label", pd.Series("", index=live_df.index)).fillna("").astype(str)
            live_df["Endpoint"] = live_df.get("strongest_endpoint_label", pd.Series("", index=live_df.index)).fillna("").astype(str)
            live_df["Status"] = live_df.get("strongest_significance_label", pd.Series("", index=live_df.index)).map(_status_text)
            return live_df[["Range", "FDR hits", "Discovery hits", "Best FDR", "Best p", "Coverage", "Bootstrap recovery", "Strongest family", "Strongest hit", "Endpoint", "Status"]].head(6)

        def _hit_effect_text(row: pd.Series) -> str:
            endpoint_type = str(row.get("endpoint_type") or "")
            hazard_ratio = _to_float(row.get("hazard_ratio"))
            effect_size = _to_float(row.get("effect_size"))
            if endpoint_type == "survival":
                return _format_progress_value(hazard_ratio, decimals=3)
            if np.isfinite(effect_size):
                return _format_progress_value(effect_size, decimals=3)
            return "-"

        def _next_live_widget_key(base_key: str) -> str:
            next_nonce = int(st.session_state.get("_tab8_fs_live_render_nonce", 0)) + 1
            st.session_state["_tab8_fs_live_render_nonce"] = next_nonce
            return f"{base_key}_{next_nonce}"

        def _live_hits_dataframe(live_hit_rows: list[dict[str, Any]], detailed: bool = False) -> pd.DataFrame:
            if not live_hit_rows:
                return pd.DataFrame()
            hits_df = pd.DataFrame(live_hit_rows).copy()
            if hits_df.empty:
                return hits_df
            zone_width = pd.to_numeric(hits_df.get("zone_width", pd.Series(np.nan, index=hits_df.index)), errors="coerce")
            hits_df["Endpoint"] = hits_df.get("endpoint_label", pd.Series("", index=hits_df.index)).fillna("").astype(str)
            hits_df["Stratum"] = hits_df.get("stratum_label", pd.Series("", index=hits_df.index)).fillna("").astype(str)
            hits_df["Family"] = hits_df.get("feature_family", pd.Series("", index=hits_df.index)).fillna("").astype(str).str.replace("_", " ").str.title()
            hits_df["Feature"] = hits_df.get("feature_label", pd.Series("", index=hits_df.index)).fillna("").astype(str)
            hits_df["Range"] = zone_width.map(format_zone_label)
            hits_df["Status"] = hits_df.get("significance_label", pd.Series("", index=hits_df.index)).map(_status_text)
            hits_df["FDR q"] = hits_df.get("fdr_q", pd.Series(np.nan, index=hits_df.index)).map(lambda value: _format_progress_value(value, decimals=3))
            hits_df["p value"] = hits_df.get("p_value", pd.Series(np.nan, index=hits_df.index)).map(lambda value: _format_progress_value(value, decimals=3))
            direction_series = hits_df.get("direction", pd.Series("", index=hits_df.index)).fillna("").astype(str)
            hits_df["Direction"] = [
                _direction_text(direction, endpoint_type, hazard_ratio)
                for direction, endpoint_type, hazard_ratio in zip(
                    direction_series,
                    hits_df.get("endpoint_type", pd.Series("", index=hits_df.index)),
                    hits_df.get("hazard_ratio", pd.Series(np.nan, index=hits_df.index)),
                )
            ]
            hits_df["HR / effect"] = hits_df.apply(_hit_effect_text, axis=1)
            if detailed:
                return hits_df[["Range", "Endpoint", "Stratum", "Family", "Feature", "Direction", "HR / effect", "FDR q", "p value", "Status"]]
            return hits_df[["Endpoint", "Stratum", "Family", "Feature", "Status", "FDR q", "p value", "Direction"]]

        def _selected_dataframe_rows(selection_event: Any) -> list[int]:
            if selection_event is None:
                return []
            selection = getattr(selection_event, "selection", None)
            if selection is not None:
                rows = getattr(selection, "rows", None)
                if rows is not None:
                    return list(rows)
            if isinstance(selection_event, dict):
                return list(((selection_event.get("selection") or {}).get("rows") or []))
            return []

        def _checked_widths_dataframe() -> pd.DataFrame:
            rows = st.session_state.get("_tab8_fs_live_width_rows", [])
            if not rows:
                return pd.DataFrame()
            checked_df = pd.DataFrame(rows).copy()
            if checked_df.empty:
                return checked_df
            for column in ["zone_width", "n_significant", "n_discovery", "best_fdr_q", "best_p_value", "patient_coverage_fraction", "bootstrap_reference_recovery"]:
                if column in checked_df.columns:
                    checked_df[column] = pd.to_numeric(checked_df[column], errors="coerce")
            checked_df = checked_df.sort_values("zone_width", ascending=True, na_position="last").reset_index(drop=True)
            checked_df["Range"] = checked_df["zone_width"].map(format_zone_label)
            checked_df["FDR hits"] = checked_df.get("n_significant", pd.Series(0, index=checked_df.index)).fillna(0).astype(int)
            checked_df["Discovery hits"] = checked_df.get("n_discovery", pd.Series(0, index=checked_df.index)).fillna(0).astype(int)
            checked_df["Best FDR"] = checked_df.get("best_fdr_q", pd.Series(np.nan, index=checked_df.index)).map(lambda value: _format_progress_value(value, decimals=3))
            checked_df["Best p"] = checked_df.get("best_p_value", pd.Series(np.nan, index=checked_df.index)).map(lambda value: _format_progress_value(value, decimals=3))
            checked_df["Coverage"] = checked_df.get("patient_coverage_fraction", pd.Series(np.nan, index=checked_df.index)).map(lambda value: _format_progress_value(value, as_percent=True))
            checked_df["Bootstrap recovery"] = checked_df.get("bootstrap_reference_recovery", pd.Series(np.nan, index=checked_df.index)).map(lambda value: _format_progress_value(value))
            checked_df["Top hit"] = checked_df.get("strongest_feature_label", pd.Series("", index=checked_df.index)).fillna("").astype(str)
            checked_df["Status"] = checked_df.get("strongest_significance_label", pd.Series("", index=checked_df.index)).map(_status_text)
            return checked_df[["zone_width", "Range", "FDR hits", "Discovery hits", "Best FDR", "Best p", "Coverage", "Bootstrap recovery", "Top hit", "Status"]]

        def _recent_width_events_dataframe() -> pd.DataFrame:
            events = st.session_state.get("_tab8_fs_recent_width_events", [])
            if not events:
                return pd.DataFrame()
            events_df = pd.DataFrame(events[::-1]).copy()
            if events_df.empty:
                return events_df
            events_df["Completed"] = events_df.get("elapsed_seconds", pd.Series(np.nan, index=events_df.index)).map(_format_duration)
            events_df["Range"] = events_df.get("width", pd.Series(np.nan, index=events_df.index)).map(format_zone_label)
            events_df["FDR hits"] = pd.to_numeric(events_df.get("n_significant", pd.Series(0, index=events_df.index)), errors="coerce").fillna(0).astype(int)
            events_df["Discovery hits"] = pd.to_numeric(events_df.get("n_discovery", pd.Series(0, index=events_df.index)), errors="coerce").fillna(0).astype(int)
            events_df["Best FDR"] = events_df.get("best_fdr_q", pd.Series(np.nan, index=events_df.index)).map(lambda value: _format_progress_value(value, decimals=3))
            events_df["Strongest family"] = events_df.get("strongest_feature_family", pd.Series("", index=events_df.index)).fillna("").astype(str).str.replace("_", " ").str.title()
            events_df["Top hit"] = events_df.get("strongest_feature_label", pd.Series("", index=events_df.index)).fillna("").astype(str)
            events_df["Status"] = events_df.get("strongest_significance_label", pd.Series("", index=events_df.index)).map(_status_text)
            return events_df[["Completed", "Range", "FDR hits", "Discovery hits", "Best FDR", "Strongest family", "Top hit", "Status"]].head(5)

        def _distance_summary_lookup() -> dict[str, float]:
            summary_df = st.session_state.get("_tab8_fs_live_distance_summary", pd.DataFrame())
            if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
                return {}
            if "statistic" not in summary_df.columns or "value" not in summary_df.columns:
                return {}
            numeric = pd.to_numeric(summary_df["value"], errors="coerce")
            return {
                str(stat): float(value)
                for stat, value in zip(summary_df["statistic"].astype(str), numeric)
                if np.isfinite(value)
            }

        def _live_distance_histogram_figure(progress_state: dict[str, Any]) -> Figure | None:
            histogram_df = st.session_state.get("_tab8_fs_live_distance_histogram", pd.DataFrame())
            if not isinstance(histogram_df, pd.DataFrame) or histogram_df.empty:
                return None
            plot_df = histogram_df.copy()
            for column in ["bin_left", "bin_right", "bin_center", "count"]:
                if column not in plot_df.columns:
                    return None
                plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
            plot_df = plot_df.dropna(subset=["bin_left", "bin_right", "bin_center", "count"]).copy()
            if plot_df.empty:
                return None
            fig, ax = plt.subplots(figsize=(5.4, 2.7))
            bar_width = (plot_df["bin_right"] - plot_df["bin_left"]).clip(lower=1e-6)
            bar_colors = ["#9bb7d6" if float(center) < 0 else "#e2b48f" for center in plot_df["bin_center"]]
            ax.bar(
                plot_df["bin_center"],
                plot_df["count"],
                width=bar_width * 0.95,
                color=bar_colors,
                edgecolor="white",
                linewidth=0.45,
                align="center",
            )
            ax.axvline(0.0, color="#1f2937", linewidth=1.1, linestyle="-")

            candidate_widths = [
                float(value)
                for value in st.session_state.get("_tab8_fs_live_candidate_widths", [])
                if pd.notna(value) and np.isfinite(float(value))
            ]
            for width_value in candidate_widths:
                ax.axvline(width_value, color="#64748b", linewidth=0.7, linestyle=":", alpha=0.18)

            current_width = _to_float(progress_state.get("width"))
            if np.isfinite(current_width):
                ax.axvspan(min(0.0, float(current_width)), max(0.0, float(current_width)), color="#9f1d35", alpha=0.12)
                ax.axvline(float(current_width), color="#9f1d35", linewidth=2.0, linestyle="--")

            ax.set_title("Distance distribution", loc="left", fontsize=10)
            ax.set_xlabel("Signed distance")
            ax.set_ylabel("Cells")
            ax.grid(True, axis="y", alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.88)
            return fig

        def _render_live_progress(progress_state: dict[str, Any]) -> None:
            stage = str(progress_state.get("stage") or "working")
            message = str(progress_state.get("message") or "Running feature screening...")
            current = max(0, int(progress_state.get("current") or 0))
            total = max(0, int(progress_state.get("total") or 0))
            fraction = min(1.0, max(float(current) / float(total), 0.0)) if total > 0 else 0.0
            now = time.monotonic()
            progress_meta = st.session_state.get("_tab8_fs_progress_meta", {"run_started_at": now, "phase_started_at": now})
            elapsed = max(0.0, now - float(progress_meta.get("run_started_at") or now))
            eta_seconds = _estimate_eta(progress_state, progress_meta, now)
            progress_label = message
            if stage.startswith("width") and total > 0:
                progress_label = f"Range scan {min(current, total)}/{total}: {message}"
            elif stage.startswith("bootstrap") and total > 0:
                progress_label = f"Bootstrap {min(current, total)}/{total}: {message}"
            progress_bar.progress(fraction, text=progress_label)

            summary_parts = [message]
            width_value = progress_state.get("width")
            if width_value is not None:
                summary_parts.append(f"Range: {format_zone_label(width_value)}")
            summary_parts.append(f"Elapsed: {_format_duration(elapsed)}")
            if eta_seconds is not None:
                summary_parts.append(f"ETA: {_format_duration(eta_seconds)}")
            if stage == "setup":
                summary_parts.append(f"Candidates: {int(progress_state.get('total_widths') or 0)}")
                summary_parts.append(f"Patients: {int(progress_state.get('total_patients') or 0)}")
                auto_distance = progress_state.get("auto_interaction_distance_value")
                auto_unit = str(progress_state.get("interaction_distance_unit") or "").strip()
                if pd.notna(auto_distance):
                    summary_parts.append(f"Auto distance: {_format_progress_value(auto_distance, decimals=1)} {auto_unit}".strip())
            elif stage in {"width_complete", "complete"}:
                summary_parts.append(f"FDR hits: {int(progress_state.get('n_significant') or 0)}")
                summary_parts.append(f"Discovery hits: {int(progress_state.get('n_discovery') or 0)}")
                coverage = progress_state.get("patient_coverage_fraction")
                if pd.notna(coverage):
                    summary_parts.append(f"Coverage: {_format_progress_value(coverage, as_percent=True)}")
            elif stage == "bootstrap_complete":
                summary_parts.append(
                    f"Recovery: {_format_progress_value(progress_state.get('bootstrap_reference_recovery'))}"
                )
                iterations = progress_state.get("bootstrap_valid_iterations")
                if pd.notna(iterations):
                    summary_parts.append(f"Valid bootstraps: {int(iterations)}")

            strongest_family = str(progress_state.get("strongest_feature_family") or "").strip()
            strongest_feature = str(progress_state.get("strongest_feature_label") or "").strip()
            strongest_endpoint = str(progress_state.get("strongest_endpoint_label") or "").strip()
            strongest_status = _status_text(progress_state.get("strongest_significance_label"))
            detail_parts: list[str] = []
            if strongest_family:
                detail_parts.append(f"Strongest family: {strongest_family.replace('_', ' ').title()}")
            if strongest_endpoint:
                detail_parts.append(f"Endpoint: {strongest_endpoint}")
            if strongest_feature:
                detail_parts.append(f"Top feature: {strongest_feature}")
            if strongest_status != "-":
                detail_parts.append(f"Status: {strongest_status}")
            strongest_fdr_q = progress_state.get("strongest_fdr_q")
            strongest_p_value = progress_state.get("strongest_p_value")
            if strongest_fdr_q is not None:
                detail_parts.append(f"Top FDR q: {_format_progress_value(strongest_fdr_q, decimals=3)}")
            if strongest_p_value is not None:
                detail_parts.append(f"Top p: {_format_progress_value(strongest_p_value, decimals=3)}")
            with dashboard_placeholder.container():
                left_col, right_col = st.columns([1.55, 1.05], gap="large")
                with left_col:
                    st.caption(" | ".join(part for part in summary_parts if part))
                    if detail_parts:
                        st.info(" | ".join(detail_parts))
                    elif stage in {"width_complete", "complete"}:
                        st.caption("No FDR or discovery hits have been found yet for the latest completed width.")
                    else:
                        st.caption("Live hit details will appear here as each width finishes.")

                    metric_cols = st.columns(5)
                    metric_cols[0].metric(
                        "Current width",
                        _format_progress_value(progress_state.get("width"), decimals=1),
                        delta=f"{int(progress_state.get('width_index') or progress_state.get('bootstrap_index') or 0)}/{int(progress_state.get('total_widths') or progress_state.get('total_bootstrap_widths') or total or 0)}",
                    )
                    metric_cols[1].metric(
                        "FDR hits",
                        str(int(progress_state.get("n_significant") or 0)),
                        delta=f"Discovery {int(progress_state.get('n_discovery') or 0)}",
                    )
                    metric_cols[2].metric(
                        "Best FDR",
                        _format_progress_value(progress_state.get("best_fdr_q"), decimals=3),
                        delta=f"Best p {_format_progress_value(progress_state.get('best_p_value'), decimals=3)}",
                    )
                    coverage_text = _format_progress_value(progress_state.get("patient_coverage_fraction"), as_percent=True)
                    if stage == "bootstrap_complete":
                        coverage_text = _format_progress_value(progress_state.get("bootstrap_reference_recovery"))
                        coverage_delta = f"Bootstraps {int(progress_state.get('bootstrap_valid_iterations') or 0)}"
                    else:
                        coverage_delta = (
                            f"Patients {int(progress_state.get('n_patients_with_zone') or 0)}"
                            if progress_state.get("n_patients_with_zone") is not None
                            else None
                        )
                    metric_cols[3].metric(
                        "Coverage / stability",
                        coverage_text,
                        delta=coverage_delta,
                    )
                    metric_cols[4].metric(
                        "Strongest family",
                        strongest_family.replace("_", " ").title() if strongest_family else "-",
                        delta=f"{strongest_endpoint or '-'} | {strongest_status}",
                    )

                with right_col:
                    summary_lookup = _distance_summary_lookup()
                    live_hist_fig = _live_distance_histogram_figure(progress_state)
                    st.markdown("#### Distance Distribution")
                    if live_hist_fig is None:
                        st.caption("The signed distance histogram will appear once the scan setup finishes.")
                    else:
                        signed_scan = bool(st.session_state.get("_tab8_fs_live_signed_width_scan", False))
                        range_caption = [
                            f"Scan range: {format_zone_label(progress_state.get('scan_min_width', min(st.session_state.get('_tab8_fs_live_candidate_widths', [0.0])) if st.session_state.get('_tab8_fs_live_candidate_widths') else float('nan')))}",
                            f"to {format_zone_label(progress_state.get('scan_max_width', max(st.session_state.get('_tab8_fs_live_candidate_widths', [0.0])) if st.session_state.get('_tab8_fs_live_candidate_widths') else float('nan')))}",
                            f"Cutpoints: {len(st.session_state.get('_tab8_fs_live_candidate_widths', []))}",
                        ]
                        if signed_scan:
                            range_caption.append("Signed pooled quantiles around 0")
                        else:
                            range_caption.append("Positive-only quantiles")
                        percentile_bits = []
                        for stat_name in ["p5", "median", "p95"]:
                            if stat_name in summary_lookup:
                                percentile_bits.append(f"{stat_name}: {_format_progress_value(summary_lookup[stat_name], decimals=1)}")
                        st.caption(" | ".join(bit for bit in range_caption if bit))
                        if percentile_bits:
                            st.caption(" | ".join(percentile_bits))
                        st.pyplot(live_hist_fig, width="stretch")
                        plt.close(live_hist_fig)

            with consolidated_placeholder.container():
                st.markdown("#### Consolidated Discoveries")
                st.caption("Auto-updating table of significant FDR hits (or the best nominal feature) for every completed range.")
                hits_by_width = st.session_state.get("_tab8_fs_live_hits_by_width", {})
                all_hits = []
                for h_list in hits_by_width.values():
                    all_hits.extend(h_list)
                consolidated_df = _live_hits_dataframe(all_hits, detailed=True)
                if consolidated_df.empty:
                    st.info("No discoveries have been found yet.")
                else:
                    st.dataframe(
                        consolidated_df,
                        width="stretch",
                        hide_index=True,
                        height=min(400, 40 + 35 * (len(consolidated_df) + 1)),
                    )

        def _ui_log(message: str) -> None:
            _log_terminal_event(message)
            status.write(message)

        def _ui_progress(progress_state: dict[str, Any]) -> None:
            now = time.monotonic()
            meta = dict(st.session_state.get("_tab8_fs_progress_meta", {"run_started_at": now, "phase_group": "initializing", "phase_started_at": now}))
            progress_group = _phase_group(str(progress_state.get("stage") or "working"))
            if meta.get("phase_group") != progress_group:
                meta["phase_group"] = progress_group
                meta["phase_started_at"] = now
            meta["last_update_at"] = now
            st.session_state["_tab8_fs_progress_meta"] = meta
            if progress_state.get("candidate_widths") is not None:
                st.session_state["_tab8_fs_live_candidate_widths"] = list(progress_state.get("candidate_widths") or [])
            if progress_state.get("distance_summary") is not None:
                st.session_state["_tab8_fs_live_distance_summary"] = progress_state.get("distance_summary")
            if progress_state.get("distance_histogram") is not None:
                st.session_state["_tab8_fs_live_distance_histogram"] = progress_state.get("distance_histogram")
            if progress_state.get("signed_width_scan") is not None:
                st.session_state["_tab8_fs_live_signed_width_scan"] = bool(progress_state.get("signed_width_scan"))
            if progress_state.get("stage") == "width_complete":
                _upsert_live_width_row(progress_state)
                live_hit_rows = list(progress_state.get("live_hit_rows") or [])
                st.session_state["_tab8_fs_live_hit_rows"] = live_hit_rows
                width_key = _width_state_key(progress_state.get("width"))
                if width_key:
                    hits_by_width = dict(st.session_state.get("_tab8_fs_live_hits_by_width", {}))
                    hits_by_width[width_key] = live_hit_rows
                    st.session_state["_tab8_fs_live_hits_by_width"] = hits_by_width
                    
                    output_dir = st.session_state.get("tab8_fs_output_dir", "")
                    if output_dir:
                        out_path = Path(output_dir)
                        out_path.mkdir(parents=True, exist_ok=True)
                        all_hits = []
                        for h_list in hits_by_width.values():
                            all_hits.extend(h_list)
                        consolidated_df = _live_hits_dataframe(all_hits, detailed=True)
                        if not consolidated_df.empty:
                            try:
                                consolidated_df.to_csv(out_path / "live_discoveries_tracking.csv", index=False)
                            except Exception:
                                pass
                run_started_at = float(meta.get("run_started_at") or now)
                recent_events = list(st.session_state.get("_tab8_fs_recent_width_events", []))
                recent_events.append(
                    {
                        "elapsed_seconds": max(0.0, now - run_started_at),
                        "width": progress_state.get("width"),
                        "n_significant": progress_state.get("n_significant"),
                        "n_discovery": progress_state.get("n_discovery"),
                        "best_fdr_q": progress_state.get("best_fdr_q"),
                        "strongest_feature_family": progress_state.get("strongest_feature_family"),
                        "strongest_feature_label": progress_state.get("strongest_feature_label"),
                        "strongest_significance_label": progress_state.get("strongest_significance_label"),
                    }
                )
                st.session_state["_tab8_fs_recent_width_events"] = recent_events[-5:]
            elif progress_state.get("stage") in {"bootstrap_complete", "bootstrap_skipped"}:
                _upsert_live_width_row(progress_state)
            elif progress_state.get("stage") == "complete":
                st.session_state["_tab8_fs_live_hit_rows"] = list(progress_state.get("live_hit_rows") or st.session_state.get("_tab8_fs_live_hit_rows", []))
            st.session_state["_tab8_fs_progress"] = dict(progress_state)
            _render_live_progress(progress_state)
            if progress_state.get("stage") in {"setup", "width_complete", "bootstrap_complete", "bootstrap_skipped", "complete"}:
                status.write(str(progress_state.get("message") or ""))

        _render_live_progress(st.session_state.get("_tab8_fs_progress", {}))
        try:
            results = run_feature_screening_pipeline(raw_df, config, log_fn=_ui_log, progress_fn=_ui_progress)
            run_dir = save_feature_screening_run(
                results,
                config,
                output_dir=st.session_state.get("tab8_fs_output_dir", ""),
                log_lines=st.session_state.get("_tab8_fs_logs", []),
                log_fn=_ui_log,
            )
            results["run_dir"] = run_dir
            st.session_state["_tab8_fs_results"] = results
            st.session_state["_tab8_fs_progress"] = {
                **st.session_state.get("_tab8_fs_progress", {}),
                "stage": "complete",
                "message": f"Feature screening complete. Outputs saved to {run_dir}",
            }
            _render_live_progress(st.session_state["_tab8_fs_progress"])
            status.update(label="Feature screening complete", state="complete")
            st.success(f"Screening completed. Outputs saved to {run_dir}")
        except Exception as exc:
            _log_terminal_event(f"Run failed: {exc}")
            st.session_state["_tab8_fs_progress"] = {
                **st.session_state.get("_tab8_fs_progress", {}),
                "stage": "error",
                "message": f"Feature screening failed: {exc}",
            }
            _render_live_progress(st.session_state["_tab8_fs_progress"])
            status.update(label="Feature screening failed", state="error")
            st.error(f"Feature screening failed: {exc}")


def _render_dataframe_download(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, data=df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv", key=key)


def _render_selected_candidate_summary(candidate_summary: pd.DataFrame, selected_width: float, key_prefix: str) -> None:
    if candidate_summary.empty:
        return
    selected_rows = candidate_summary.loc[candidate_summary["zone_width"].astype(float) == float(selected_width)].copy()
    if selected_rows.empty:
        return
    selected = selected_rows.iloc[0]
    summary_parts = [
        f"Selected range {str(selected.get('zone_label') or format_zone_label(selected_width))}",
        f"FDR hits: {int(_to_float(selected.get('n_significant'), 0.0))}",
        f"Discovery hits: {int(_to_float(selected.get('n_discovery'), 0.0))}",
    ]
    bootstrap_recovery = _to_float(selected.get("bootstrap_reference_recovery"))
    bootstrap_iterations = _to_float(selected.get("bootstrap_valid_iterations"))
    coverage = _to_float(selected.get("patient_coverage_fraction"))
    roi_coverage = _to_float(selected.get("roi_coverage_fraction"))
    bootstrap_unit = str(selected.get("bootstrap_sampling_unit") or "patient").strip().lower()
    if np.isfinite(bootstrap_recovery):
        summary_parts.append(f"Bootstrap recovery: {float(bootstrap_recovery):.2f}")
    if np.isfinite(bootstrap_iterations):
        summary_parts.append(f"Bootstrap iterations: {int(bootstrap_iterations)}")
    summary_parts.append(f"Bootstrap unit: {'ROI' if bootstrap_unit == 'roi' else 'Patient'}")
    if np.isfinite(coverage):
        summary_parts.append(f"Patient coverage: {float(coverage):.2f}")
    if bootstrap_unit == "roi" and np.isfinite(roi_coverage):
        summary_parts.append(f"ROI coverage: {float(roi_coverage):.2f}")
    st.caption(" | ".join(summary_parts))
    detail_columns = [
        column
        for column in [
            "zone_width",
            "candidate_rank",
            "n_significant",
            "n_discovery",
            "best_fdr_q",
            "best_p_value",
            "bootstrap_reference_recovery",
            "bootstrap_n_significant_mean",
            "bootstrap_valid_iterations",
            "patient_coverage_fraction",
            "candidate_priority_score",
            "candidate_reproducibility_score",
            "candidate_significance_score",
            "feature_stability_score",
            "feature_stability_exact_score",
            "feature_stability_band_hits",
            "feature_stability_exact_hits",
            "feature_stability_top_feature_label",
        ]
        if column in selected_rows.columns
    ]
    if detail_columns:
        st.dataframe(selected_rows[detail_columns], width="stretch", hide_index=True, key=f"{key_prefix}_selected_candidate")


def _render_feature_stability_panel(
    feature_summary: pd.DataFrame,
    stats_all: pd.DataFrame,
    key_prefix: str,
    default_top_n: int = 25,
    neighbor_steps: int | None = None,
) -> None:
    if feature_summary.empty or stats_all.empty:
        st.info("Feature stability results are not available for this run.")
        return

    ranked_df = feature_summary.copy()
    if "feature_rank" in ranked_df.columns:
        ranked_df = ranked_df.sort_values("feature_rank", ascending=True, na_position="last")
    else:
        ranked_df = ranked_df.sort_values(["best_fdr_q", "best_p_value"], ascending=[True, True], na_position="last")
    ranked_df = ranked_df.reset_index(drop=True)
    export_df = ranked_df.drop(columns=[column for column in ranked_df.columns if str(column).endswith("_indices")], errors="ignore").copy()

    caption_parts = [
        "This view ranks features by how persistently they appear across ranges and how often they are rediscovered in nearby ranges during bootstrap reruns.",
        f"Default heatmap top N: {int(default_top_n)}.",
    ]
    if neighbor_steps is not None:
        caption_parts.append(f"Bootstrap neighboring-range tolerance: +/- {int(neighbor_steps)} range step(s).")
    st.caption(" ".join(caption_parts))

    preview_df = export_df.copy()
    preview_df["Direction"] = [
        _direction_text(direction, endpoint_type, hazard_ratio)
        for direction, endpoint_type, hazard_ratio in zip(
            preview_df.get("best_direction", pd.Series("", index=preview_df.index)),
            preview_df.get("endpoint_type", pd.Series("", index=preview_df.index)),
            preview_df.get("best_hazard_ratio", pd.Series(np.nan, index=preview_df.index)),
        )
    ]
    preview_df["Reference status"] = preview_df.get("reference_significance_label", pd.Series("", index=preview_df.index)).replace(
        {
            "significant_fdr": "FDR significant",
            "discovery_fdr": "FDR discovery",
            "not_significant": "Not significant",
        }
    )
    preview_df["Bootstrap recovery"] = pd.to_numeric(preview_df.get("feature_bootstrap_recovery", pd.Series(np.nan, index=preview_df.index)), errors="coerce")
    preview_df["Exact recovery"] = pd.to_numeric(preview_df.get("feature_bootstrap_exact_recovery", pd.Series(np.nan, index=preview_df.index)), errors="coerce")
    preview_df["Best FDR"] = pd.to_numeric(preview_df.get("best_fdr_q", pd.Series(np.nan, index=preview_df.index)), errors="coerce")
    preview_df["FDR ranges"] = pd.to_numeric(preview_df.get("n_significant_ranges", pd.Series(0, index=preview_df.index)), errors="coerce").fillna(0).astype(int)
    preview_df["Discovery ranges"] = pd.to_numeric(preview_df.get("n_discovery_ranges", pd.Series(0, index=preview_df.index)), errors="coerce").fillna(0).astype(int)

    table_columns = [
        column
        for column in [
            "feature_rank",
            "feature_label",
            "endpoint_label",
            "stratum_label",
            "best_zone_label",
            "Reference status",
            "FDR ranges",
            "Discovery ranges",
            "significant_band_label",
            "discovery_band_label",
            "Bootstrap recovery",
            "Exact recovery",
            "Direction",
        ]
        if column in preview_df.columns
    ]
    st.dataframe(preview_df[table_columns], width="stretch", hide_index=True, height=min(420, 40 + 35 * (len(preview_df.head(12)) + 1)))
    _render_dataframe_download(export_df, "Download feature stability summary", "feature_stability_summary.csv", f"{key_prefix}_feature_summary_download")

    heatmap_top_n_key = f"{key_prefix}_feature_heatmap_top_n"
    max_heatmap_n = max(5, min(100, len(ranked_df)))
    default_heatmap_n = min(max(5, int(default_top_n)), max_heatmap_n)
    _stateful_number_input(
        "Heatmap top N features",
        key=heatmap_top_n_key,
        default=default_heatmap_n,
        min_value=5,
        max_value=max_heatmap_n,
        step=1,
        help="Limits the heatmap to the highest-ranked stable features for readability.",
    )
    heatmap_top_n = int(st.session_state.get(heatmap_top_n_key, default_heatmap_n))
    heatmap_fig = plot_feature_stability_heatmap(
        stats_all,
        ranked_df,
        top_n=heatmap_top_n,
    )
    _render_figure_with_download(heatmap_fig, "Feature stability heatmap", f"{key_prefix}_feature_heatmap")
    plt.close(heatmap_fig)

    option_labels = ranked_df.apply(
        lambda row: f"#{int(_to_float(row.get('feature_rank'), 0.0))} | {str(row.get('feature_label') or row.get('feature_name') or 'Feature')} | {str(row.get('endpoint_label') or '')} | {str(row.get('stratum_label') or '')}",
        axis=1,
    ).tolist()
    feature_lookup = dict(zip(option_labels, ranked_df.index.tolist()))
    _stateful_selectbox(
        "Feature profile",
        options=option_labels,
        key=f"{key_prefix}_feature_profile_label",
        default=option_labels[0] if option_labels else None,
    )
    selected_label = st.session_state.get(f"{key_prefix}_feature_profile_label", option_labels[0] if option_labels else "")
    selected_row = ranked_df.iloc[int(feature_lookup.get(selected_label, 0))] if option_labels else pd.Series(dtype=object)

    if not selected_row.empty:
        summary_bits = [
            f"Best range: {str(selected_row.get('best_zone_label') or '-')}",
            f"Reference band: {str(selected_row.get('reference_band_label') or '-')}",
            f"FDR band: {str(selected_row.get('significant_band_label') or '-')}",
            f"Discovery band: {str(selected_row.get('discovery_band_label') or '-')}",
            f"Bootstrap recovery: {_to_float(selected_row.get('feature_bootstrap_recovery')):.2f}" if np.isfinite(_to_float(selected_row.get('feature_bootstrap_recovery'))) else "Bootstrap recovery: -",
            f"Exact recovery: {_to_float(selected_row.get('feature_bootstrap_exact_recovery')):.2f}" if np.isfinite(_to_float(selected_row.get('feature_bootstrap_exact_recovery'))) else "Exact recovery: -",
        ]
        st.caption(" | ".join(summary_bits))

        profile_fig = plot_feature_range_profile(stats_all, selected_row)
        _render_figure_with_download(profile_fig, "Selected feature range profile", f"{key_prefix}_feature_profile")
        plt.close(profile_fig)

        selected_feature_rank = int(pd.to_numeric(pd.Series([selected_row.get("feature_rank")]), errors="coerce").fillna(1).iloc[0])
        publication_top_n = max(heatmap_top_n, selected_feature_rank)
        publication_fig = plot_feature_stability_publication_panel(stats_all, ranked_df, selected_row, top_n=publication_top_n)
        _render_figure_with_download(publication_fig, "Publication feature stability panel", f"{key_prefix}_feature_publication_panel")
        plt.close(publication_fig)

        feature_ranges = stats_all[
            (stats_all["endpoint_label"].astype(str) == str(selected_row.get("endpoint_label") or ""))
            & (stats_all["stratum_label"].astype(str) == str(selected_row.get("stratum_label") or ""))
            & (stats_all["feature_name"].astype(str) == str(selected_row.get("feature_name") or ""))
        ].copy()
        if not feature_ranges.empty:
            feature_ranges["zone_width"] = pd.to_numeric(feature_ranges["zone_width"], errors="coerce")
            feature_ranges = feature_ranges.sort_values("zone_width", ascending=True, na_position="last").reset_index(drop=True)
            feature_ranges["Range"] = feature_ranges["zone_width"].map(format_zone_label)
            feature_ranges["Status"] = feature_ranges.get("significance_label", pd.Series("", index=feature_ranges.index)).replace(
                {
                    "significant_fdr": "FDR significant",
                    "discovery_fdr": "FDR discovery",
                    "nominal_p": "Nominal p<0.05",
                    "not_significant": "Not significant",
                }
            )
            feature_ranges["Direction"] = [
                _direction_text(direction, endpoint_type, hazard_ratio)
                for direction, endpoint_type, hazard_ratio in zip(
                    feature_ranges.get("direction", pd.Series("", index=feature_ranges.index)),
                    feature_ranges.get("endpoint_type", pd.Series("", index=feature_ranges.index)),
                    feature_ranges.get("hazard_ratio", pd.Series(np.nan, index=feature_ranges.index)),
                )
            ]
            feature_ranges["HR / effect"] = feature_ranges.apply(
                lambda row: (
                    f"{float(_to_float(row.get('hazard_ratio'))):.3f}"
                    if str(row.get("endpoint_type") or "") == "survival" and np.isfinite(_to_float(row.get('hazard_ratio')))
                    else (
                        f"{float(_to_float(row.get('effect_size'))):.3f}"
                        if np.isfinite(_to_float(row.get('effect_size')))
                        else "-"
                    )
                ),
                axis=1,
            )
            st.markdown("#### Per-Range Results For Selected Feature")
            range_table = feature_ranges[["Range", "Status", "fdr_q", "p_value", "Direction", "HR / effect"]].copy()
            st.dataframe(range_table, width="stretch", hide_index=True, height=min(340, 40 + 35 * (len(range_table) + 1)))
            _render_dataframe_download(
                feature_ranges.drop(columns=[column for column in feature_ranges.columns if str(column).startswith("_")], errors="ignore"),
                "Download selected feature per-range results",
                "selected_feature_ranges.csv",
                f"{key_prefix}_feature_ranges_download",
            )


def _render_feature_linkage_panel(feature_summary: pd.DataFrame, key_prefix: str) -> None:
    if feature_summary.empty:
        st.info("Feature linkage results are not available for this run.")
        return

    st.markdown("### Feature Linkage Summary")
    st.markdown("Find how specific biological components (e.g., CD4) are linked to survival across multiple feature families and ranges.")
    
    component_query = st.text_input("Search Component (e.g., CD4, Treg, Tumor)", key=f"{key_prefix}_linkage_query")
    if component_query:
        query_lower = str(component_query).strip().lower()
        
        linkage_df = feature_summary[
            feature_summary["feature_name"].astype(str).str.lower().str.contains(query_lower, na=False) |
            feature_summary["feature_label"].astype(str).str.lower().str.contains(query_lower, na=False)
        ].copy()
        
        if not linkage_df.empty:
            # Sort by best FDR
            linkage_df = linkage_df.sort_values(["best_fdr_q", "best_p_value"], ascending=[True, True], na_position="last").reset_index(drop=True)
            
            display_cols = []
            if "endpoint_label" in linkage_df.columns: display_cols.append("endpoint_label")
            if "stratum_label" in linkage_df.columns: display_cols.append("stratum_label")
            if "feature_family" in linkage_df.columns: display_cols.append("feature_family")
            if "feature_label" in linkage_df.columns: display_cols.append("feature_label")
            if "significant_band_label" in linkage_df.columns: display_cols.append("significant_band_label")
            if "reference_significance_label" in linkage_df.columns: display_cols.append("reference_significance_label")
            
            st.dataframe(linkage_df[display_cols], width="stretch", hide_index=True)
            
            _render_dataframe_download(
                linkage_df.drop(columns=[col for col in linkage_df.columns if str(col).startswith("_")], errors="ignore"),
                f"Download linkage results for '{component_query}'",
                f"linkage_summary_{query_lower}.csv",
                f"{key_prefix}_linkage_download"
            )
        else:
            st.info(f"No hits found containing '{component_query}'.")
    else:
        st.info("Enter a component name in the search box to view linked features.")


def _render_existing_run_loader() -> None:
    st.markdown("### Load Existing Run")
    existing_run_dir = directory_input_with_browse(
        label="Existing run directory",
        text_key="tab8_fs_loaded_run_dir",
        default_value=st.session_state.get("tab8_fs_loaded_run_dir", str(Path.cwd())),
        browser_key_prefix="tab8_fs_loaded_run_dir",
        start_dir_fallback=st.session_state.get("tab8_fs_output_dir", str(Path.cwd())),
        help_text="Point this to a previously saved screening run folder containing manifest.json and the exported tables.",
    )
    if st.button("Load saved run", key="tab8_fs_load_run_btn"):
        run_dir = existing_run_dir or st.session_state.get("tab8_fs_loaded_run_dir", "")
        if not run_dir:
            st.warning("Choose a run directory first.")
        else:
            try:
                st.session_state["_tab8_fs_loaded_run"] = load_saved_run(run_dir)
                _log_terminal_event(f"Loaded saved run: {run_dir}")
                st.success(f"Loaded saved run from {run_dir}")
            except Exception as exc:
                st.error(f"Could not load saved run: {exc}")


def _resolve_image_lookup(raw_image_dir: str) -> dict[str, Path]:
    root = Path(raw_image_dir).expanduser()
    if not root.exists():
        return {}
    lookup: dict[str, Path] = {}
    for path in root.iterdir():
        if not path.is_file() or not is_supported_image_file(path.name):
            continue
        stem = normalize_image_stem(path.name).lower()
        lookup[stem] = path
        if stem.endswith("_merged"):
            lookup.setdefault(stem[:-7], path)
    return lookup


def _normalize_marker_name(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _color_to_hex(color: Any) -> str:
    try:
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
            arr = np.asarray(color[:3], dtype=float)
            if np.nanmax(arr) > 1.0:
                arr = arr / 255.0
            return str(to_hex(np.clip(arr, 0.0, 1.0)))
        return str(to_hex(color))
    except Exception:
        return "#ffffff"


def _resolve_representative_marker_options(image_paths: list[Path]) -> tuple[list[str], dict[str, str]]:
    for image_path in image_paths:
        try:
            channel_names, channel_colors, _, _ = get_channel_info(str(image_path))
        except Exception:
            continue
        if not channel_names:
            continue
        marker_names = [str(name) for name in channel_names if str(name)]
        color_map = {
            str(name): _color_to_hex(channel_colors[idx] if idx < len(channel_colors) else (255, 255, 255))
            for idx, name in enumerate(marker_names)
        }
        return marker_names, color_map
    return [], {}


def _resolve_marker_indices(channel_names: list[str], selected_markers: list[str]) -> list[tuple[str, int]]:
    normalized_lookup = {
        _normalize_marker_name(name): idx
        for idx, name in enumerate(channel_names)
        if str(name)
    }
    resolved: list[tuple[str, int]] = []
    for marker in selected_markers:
        marker_name = str(marker)
        idx = normalized_lookup.get(_normalize_marker_name(marker_name))
        if idx is None:
            continue
        resolved.append((marker_name, int(idx)))
    return resolved


def _compose_marker_preview(
    preview_data: dict[int, np.ndarray],
    ordered_markers: list[tuple[str, int]],
    marker_color_mapping: dict[str, str],
) -> np.ndarray | None:
    if not ordered_markers:
        return None
    first_array: np.ndarray | None = None
    composite: np.ndarray | None = None
    for marker_name, channel_index in ordered_markers:
        plane = preview_data.get(int(channel_index))
        if plane is None:
            continue
        arr = np.asarray(plane, dtype=float)
        if arr.ndim != 2:
            continue
        if first_array is None:
            first_array = arr
            composite = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=float)
        if composite is None:
            continue
        if arr.size > 20:
            lo, hi = np.nanpercentile(arr, [1.0, 99.5])
        else:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        try:
            from matplotlib.colors import to_rgb as _to_rgb

            rgb = np.asarray(_to_rgb(marker_color_mapping.get(marker_name, "#ffffff")), dtype=float)
        except Exception:
            rgb = np.asarray((1.0, 1.0, 1.0), dtype=float)
        composite = np.maximum(composite, norm[:, :, None] * rgb[None, None, :])
    if composite is None:
        return None
    return np.clip(composite, 0.0, 1.0)


def _scale_motif_visualization_data(motif_data: dict[str, Any], scale_x: float, scale_y: float) -> dict[str, Any]:
    scaled = dict(motif_data)
    scaled["plot_coords"] = np.asarray(motif_data.get("plot_coords", np.empty((0, 2), dtype=float)), dtype=float).copy()
    scaled["centroids"] = np.asarray(motif_data.get("centroids", np.empty((0, 2), dtype=float)), dtype=float).copy()
    if scaled["plot_coords"].size:
        scaled["plot_coords"][:, 0] = scaled["plot_coords"][:, 0] * float(scale_x)
        scaled["plot_coords"][:, 1] = scaled["plot_coords"][:, 1] * float(scale_y)
    if scaled["centroids"].size:
        scaled["centroids"][:, 0] = scaled["centroids"][:, 0] * float(scale_x)
        scaled["centroids"][:, 1] = scaled["centroids"][:, 1] * float(scale_y)
    scaled["max_px_scaled"] = float(_to_float(motif_data.get("max_px_scaled"), 0.0)) * max(float(scale_x), float(scale_y))
    return scaled


def _build_representative_feature_overlay(
    hit_row: pd.Series,
    zone_plot_df: pd.DataFrame,
    target_phenotypes: list[str],
    motif_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_family = str(hit_row.get("feature_family", "") or "")
    target_set = {str(value) for value in target_phenotypes if str(value)}
    coords = np.empty((0, 2), dtype=float)
    overlay_kind = "none"
    if feature_family in {"dyad", "triad"} and isinstance(motif_data, dict):
        coords = np.asarray(motif_data.get("centroids", np.empty((0, 2), dtype=float)), dtype=float)
        overlay_kind = "motif_centroids"
    elif feature_family in {"density", "proportion", "ratio", "nn_distance"} and target_set and not zone_plot_df.empty:
        coords = zone_plot_df.loc[
            zone_plot_df["__phenotype__"].astype(str).isin(target_set),
            ["__plot_x__", "__plot_y__"],
        ].to_numpy(dtype=float)
        overlay_kind = "target_cells"
    return {
        "feature_family": feature_family,
        "coords": coords,
        "kind": overlay_kind,
        "count": int(len(coords)),
        "label": str(hit_row.get("feature_label") or hit_row.get("feature_name") or "Selected feature"),
    }


def _default_representative_feature_overlay_mode(feature_family: str) -> str:
    family = str(feature_family or "").strip().lower()
    if family in {"dyad", "triad"}:
        return "Feature fuzzy cloud"
    return "Feature dots"


def _draw_representative_feature_overlay(
    ax: Axes,
    region_plot_df: pd.DataFrame,
    overlay_data: dict[str, Any] | None,
    settings: dict[str, Any],
) -> None:
    if not isinstance(overlay_data, dict):
        return
    overlay_mode = str(settings.get("feature_overlay_mode", "None") or "None")
    if overlay_mode == "None":
        return
    coords = np.asarray(overlay_data.get("coords", np.empty((0, 2), dtype=float)), dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return

    overlay_color = str(settings.get("feature_overlay_color", "#ff6b00"))
    overlay_alpha = float(settings.get("feature_overlay_alpha", 0.65))
    overlay_size = float(settings.get("feature_overlay_size", 36.0))

    if overlay_mode in {"Feature fuzzy cloud", "Dots + fuzzy cloud"} and coords.shape[0] >= 3 and not region_plot_df.empty:
        try:
            from matplotlib.colors import LinearSegmentedColormap, to_rgb as _to_rgb
            from scipy.stats import gaussian_kde

            x_min = float(region_plot_df["__plot_x__"].min())
            x_max = float(region_plot_df["__plot_x__"].max())
            y_min = float(region_plot_df["__plot_y__"].min())
            y_max = float(region_plot_df["__plot_y__"].max())
            pad_x = max((x_max - x_min) * 0.04, 5.0)
            pad_y = max((y_max - y_min) * 0.04, 5.0)
            grid_x, grid_y = np.mgrid[x_min - pad_x : x_max + pad_x : 80j, y_min - pad_y : y_max + pad_y : 80j]
            kde = gaussian_kde(coords.T)
            kde_vals = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
            kde_norm = (kde_vals - kde_vals.min()) / max(kde_vals.max() - kde_vals.min(), 1e-12)
            rgb = _to_rgb(overlay_color)
            cmap = LinearSegmentedColormap.from_list(
                "representative_feature_cloud",
                [(rgb[0], rgb[1], rgb[2], 0.0), (rgb[0], rgb[1], rgb[2], min(1.0, overlay_alpha + 0.15))],
                N=256,
            )
            ax.contourf(
                grid_x,
                grid_y,
                kde_norm,
                levels=np.linspace(0.18, 1.0, 7),
                cmap=cmap,
                antialiased=True,
                zorder=4,
            )
        except Exception:
            pass

    if overlay_mode in {"Feature dots", "Dots + fuzzy cloud", "Feature fuzzy cloud"}:
        dot_size = overlay_size if overlay_mode != "Feature fuzzy cloud" or coords.shape[0] < 3 else max(overlay_size * 0.55, 12.0)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=dot_size,
            alpha=overlay_alpha if overlay_mode != "Feature fuzzy cloud" else min(overlay_alpha + 0.1, 0.95),
            c=overlay_color,
            edgecolors="white",
            linewidths=0.35,
            zorder=6,
        )


def _build_representative_motif_export_df(
    zone_df: pd.DataFrame,
    motif_data: dict[str, Any],
    hit_row: pd.Series,
    representative_meta: dict[str, Any],
    source_section: str,
    zone_start: float,
    zone_end: float,
) -> pd.DataFrame:
    instances = list(motif_data.get("instances", [])) if isinstance(motif_data, dict) else []
    if zone_df.empty or not instances:
        return pd.DataFrame()

    zone_export = zone_df.reset_index().rename(columns={"index": "unified_csv_row_index"})
    centroids = np.asarray(motif_data.get("centroids", np.empty((0, 2), dtype=float)), dtype=float)
    target_sorted = [str(value) for value in motif_data.get("target_sorted", []) if str(value)]
    hit_meta = {f"hit_{column}": value for column, value in hit_row.to_dict().items()}
    representative_prefixed = {f"representative_{column}": value for column, value in representative_meta.items()}
    rows: list[dict[str, Any]] = []

    for instance_rank, instance in enumerate(instances, start=1):
        motif_id = "::".join(
            [
                str(source_section).lower(),
                normalize_image_stem(str(representative_meta.get("batch", "batch"))),
                normalize_image_stem(str(representative_meta.get("roi", "roi"))),
                normalize_image_stem(str(hit_row.get("feature_name") or hit_row.get("feature_label") or "motif")),
                f"{int(instance_rank):05d}",
            ]
        )
        centroid_x = float("nan")
        centroid_y = float("nan")
        if centroids.ndim == 2 and instance_rank - 1 < len(centroids):
            centroid_x = float(centroids[instance_rank - 1, 0])
            centroid_y = float(centroids[instance_rank - 1, 1])
        for cell_ordinal, zone_position in enumerate(instance, start=1):
            if int(zone_position) >= len(zone_export):
                continue
            cell_row = zone_export.iloc[int(zone_position)].to_dict()
            export_row = {
                "source_section": str(source_section),
                "motif_instance_id": motif_id,
                "motif_instance_rank": int(instance_rank),
                "motif_instance_size": int(len(instance)),
                "motif_cell_ordinal": int(cell_ordinal),
                "motif_cell_role": f"cell_{int(cell_ordinal)}",
                "motif_target_phenotypes": "::".join(target_sorted),
                "motif_centroid_x": centroid_x,
                "motif_centroid_y": centroid_y,
                "representative_zone_start": float(zone_start),
                "representative_zone_end": float(zone_end),
                "representative_zone_label": format_zone_interval_label(zone_start, zone_end),
            }
            export_row.update(hit_meta)
            export_row.update(representative_prefixed)
            export_row.update(cell_row)
            rows.append(export_row)

    return pd.DataFrame(rows)


def _build_representative_motif_instance_summary_df(
    motif_cell_df: pd.DataFrame,
    phenotype_col: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    if motif_cell_df.empty:
        return pd.DataFrame()

    constant_cols = [
        column
        for column in motif_cell_df.columns
        if column in {
            "source_section",
            "representative_zone_start",
            "representative_zone_end",
            "representative_zone_label",
        }
        or (str(column).startswith("motif_") and not str(column).startswith("motif_cell_"))
        or str(column).startswith("hit_")
        or str(column).startswith("representative_")
        or str(column).startswith("selected_hit_")
        or str(column).startswith("export_")
    ]
    rows: list[dict[str, Any]] = []

    grouped = motif_cell_df.sort_values(["motif_instance_rank", "motif_cell_ordinal"], na_position="last").groupby("motif_instance_id", sort=False)
    for _motif_id, group in grouped:
        first_row = group.iloc[0]
        summary_row = {column: first_row[column] for column in constant_cols if column in group.columns}
        summary_row["motif_member_count"] = int(len(group))
        if "unified_csv_row_index" in group.columns:
            summary_row["motif_member_row_indices"] = "::".join(str(int(value)) for value in pd.to_numeric(group["unified_csv_row_index"], errors="coerce").dropna().tolist())
        if phenotype_col and phenotype_col in group.columns:
            summary_row["motif_member_phenotypes"] = "::".join(group[phenotype_col].astype(str).tolist())
        if x_col and x_col in group.columns:
            summary_row["motif_member_x_coords"] = "::".join(f"{float(value):.6g}" for value in pd.to_numeric(group[x_col], errors="coerce").dropna().tolist())
        if y_col and y_col in group.columns:
            summary_row["motif_member_y_coords"] = "::".join(f"{float(value):.6g}" for value in pd.to_numeric(group[y_col], errors="coerce").dropna().tolist())

        for cell_row in group.to_dict("records"):
            cell_ordinal = int(_to_float(cell_row.get("motif_cell_ordinal"), 0.0))
            if cell_ordinal <= 0:
                continue
            for column, value in cell_row.items():
                if column == "motif_instance_id" or column in constant_cols:
                    continue
                summary_row[f"cell_{cell_ordinal}_{column}"] = value
        rows.append(summary_row)

    return pd.DataFrame(rows)


def _feature_significant_zone_labels(stats_all: pd.DataFrame, hit_row: pd.Series) -> list[str]:
    if not isinstance(stats_all, pd.DataFrame) or stats_all.empty:
        return []
    mask = (
        (stats_all["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
        & (stats_all["endpoint_label"].astype(str) == str(hit_row.get("endpoint_label") or ""))
        & (stats_all["stratum_label"].astype(str) == str(hit_row.get("stratum_label") or ""))
        & (stats_all["significance_label"].astype(str).isin(["significant_fdr", "discovery_fdr"]))
    )
    matched = stats_all.loc[mask].copy()
    if matched.empty:
        return []
    matched = matched.sort_values(["zone_width", "p_value"], na_position="last")
    labels: list[str] = []
    for row in matched.itertuples(index=False):
        row_label = str(getattr(row, "zone_label", "") or format_zone_label(getattr(row, "zone_width", 0.0)))
        if row_label and row_label not in labels:
            labels.append(row_label)
    return labels


def _selected_hit_motif_export_signature(
    hit_row: pd.Series,
    section_label: str,
    zone_start: float,
    zone_end: float,
) -> str:
    payload = {
        "section": str(section_label),
        "feature_name": str(hit_row.get("feature_name") or ""),
        "feature_family": str(hit_row.get("feature_family") or ""),
        "endpoint_label": str(hit_row.get("endpoint_label") or ""),
        "stratum_label": str(hit_row.get("stratum_label") or ""),
        "zone_start": float(zone_start),
        "zone_end": float(zone_end),
        "cutpoint": float(_to_float(hit_row.get("cutpoint"), float("nan"))) if np.isfinite(_to_float(hit_row.get("cutpoint"), float("nan"))) else None,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _build_selected_hit_all_roi_motif_exports(
    analysis_df: pd.DataFrame,
    stats_all: pd.DataFrame,
    config: dict[str, Any],
    hit_row: pd.Series,
    section_label: str,
    zone_start: float,
    zone_end: float,
    selected_roi_features_df: pd.DataFrame | None = None,
    selected_patient_export_df: pd.DataFrame | None = None,
    selected_patient_features_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_family = str(hit_row.get("feature_family") or "")
    if analysis_df.empty or feature_family not in {"dyad", "triad"}:
        return pd.DataFrame(), pd.DataFrame()

    target_phenotypes = _representative_target_phenotypes(hit_row)
    if len(target_phenotypes) not in {2, 3}:
        return pd.DataFrame(), pd.DataFrame()

    batch_col = str(config["columns"].get("batch") or "")
    roi_col = str(config["columns"].get("roi") or "")
    patient_col = str(config["columns"].get("patient") or "")
    phenotype_col = str(config["columns"].get("phenotype") or "")
    x_col = str(config["columns"].get("x") or "")
    y_col = str(config["columns"].get("y") or "")
    required_cols = [column for column in [batch_col, roi_col, phenotype_col, x_col, y_col] if column]
    if any(column not in analysis_df.columns for column in required_cols):
        return pd.DataFrame(), pd.DataFrame()

    significant_zone_labels = _feature_significant_zone_labels(stats_all, hit_row)
    selected_zone_label = format_zone_interval_label(zone_start, zone_end)
    significant_zone_text = "::".join(significant_zone_labels) if significant_zone_labels else selected_zone_label
    grouped = analysis_df.groupby([batch_col, roi_col], dropna=False, sort=False)
    hit_meta = {f"hit_{column}": value for column, value in hit_row.to_dict().items()}
    score_type = "permutation_zscore" if int(max(_to_float(config.get("screening", {}).get("dyad_triad_permutation_n_iter"), 0.0), 0.0)) > 0 else "relative_motif_fraction"
    selected_roi_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(selected_roi_features_df, pd.DataFrame) and not selected_roi_features_df.empty:
        roi_feature_filter = (
            (selected_roi_features_df["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
            & (selected_roi_features_df["zone_width"].astype(float) == float(hit_row.get("zone_width") or 0.0))
            & (selected_roi_features_df["stratum_label"].astype(str) == str(hit_row.get("stratum_label") or ""))
        )
        for row in selected_roi_features_df.loc[roi_feature_filter].itertuples(index=False):
            row_dict = row._asdict()
            selected_roi_lookup[(str(row_dict.get("batch", "")), str(row_dict.get("roi", "")))] = {
                "selected_hit_roi_feature_value": _to_float(row_dict.get("value"), float("nan")),
                "selected_hit_roi_feature_score_type": score_type,
                "selected_hit_roi_zone_label": str(row_dict.get("zone_label") or selected_zone_label),
                "selected_hit_roi_zone_start": _to_float(row_dict.get("zone_start"), float(zone_start)),
                "selected_hit_roi_zone_end": _to_float(row_dict.get("zone_end"), float(zone_end)),
                "selected_hit_roi_zone_cell_count": int(_to_float(row_dict.get("n_zone_cells"), 0.0)),
                "selected_hit_roi_area_mm2": _to_float(row_dict.get("area_mm2"), float("nan")),
                "selected_hit_roi_buffer_distance_px": _to_float(row_dict.get("buffer_distance_px"), float("nan")),
                "selected_hit_roi_feature_available": np.isfinite(_to_float(row_dict.get("value"), float("nan"))),
            }
    selected_patient_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(selected_patient_export_df, pd.DataFrame) and not selected_patient_export_df.empty:
        patient_export_filter = (
            (selected_patient_export_df["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
            & (selected_patient_export_df["zone_width"].astype(float) == float(hit_row.get("zone_width") or 0.0))
        )
        for row in selected_patient_export_df.loc[patient_export_filter].itertuples(index=False):
            row_dict = row._asdict()
            patient_id = str(row_dict.get("patient", ""))
            if not patient_id:
                continue
            selected_patient_lookup[patient_id] = {
                "selected_hit_patient_feature_value": _to_float(row_dict.get("feature_value"), float("nan")),
                "selected_hit_patient_screen_group": str(row_dict.get("screen_group") or row_dict.get("endpoint_group") or ""),
                "selected_hit_patient_cutpoint": _to_float(row_dict.get("cutpoint"), float("nan")),
                "selected_hit_patient_endpoint_time": _to_float(row_dict.get("endpoint_time"), float("nan")),
                "selected_hit_patient_endpoint_censor": _to_float(row_dict.get("endpoint_censor"), float("nan")),
                "selected_hit_patient_event_flag": _to_float(row_dict.get("event_flag"), float("nan")),
                "selected_hit_patient_risk_direction": str(row_dict.get("risk_direction") or ""),
                "selected_hit_patient_screen_group_color": str(row_dict.get("screen_group_color") or ""),
            }
    if isinstance(selected_patient_features_df, pd.DataFrame) and not selected_patient_features_df.empty:
        patient_feature_filter = (
            (selected_patient_features_df["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
            & (selected_patient_features_df["zone_width"].astype(float) == float(hit_row.get("zone_width") or 0.0))
            & (selected_patient_features_df["stratum_label"].astype(str) == str(hit_row.get("stratum_label") or ""))
        )
        for row in selected_patient_features_df.loc[patient_feature_filter].itertuples(index=False):
            row_dict = row._asdict()
            patient_id = str(row_dict.get("patient", ""))
            if not patient_id:
                continue
            patient_meta = selected_patient_lookup.setdefault(patient_id, {})
            if "selected_hit_patient_feature_value" not in patient_meta:
                patient_meta["selected_hit_patient_feature_value"] = _to_float(row_dict.get("value"), float("nan"))
            patient_meta["selected_hit_patient_n_regions"] = int(_to_float(row_dict.get("n_regions"), 0.0))

    cell_frames: list[pd.DataFrame] = []

    for (batch_value, roi_value), group_df in grouped:
        region_df = group_df.copy()
        if region_df.empty:
            continue
        motif_data = compute_roi_motif_visualization_data(
            zone_df=region_df,
            config=config,
            target_phenotypes=target_phenotypes,
            scale_x=1.0,
            scale_y=1.0,
        )
        instances = list(motif_data.get("instances", [])) if isinstance(motif_data, dict) else []
        if not instances:
            continue

        region_export = region_df.reset_index().rename(columns={"index": "unified_csv_row_index"})
        centroids = np.asarray(motif_data.get("centroids", np.empty((0, 2), dtype=float)), dtype=float)
        target_sorted = [str(value) for value in motif_data.get("target_sorted", []) if str(value)]
        selected_zone_index = set(_representative_zone_subset(region_df, config, zone_end, zone_start).index.tolist())
        export_patient = str(region_df[patient_col].iloc[0]) if patient_col and patient_col in region_df.columns and not region_df.empty else ""
        roi_meta = selected_roi_lookup.get((str(batch_value), str(roi_value)), {})
        patient_meta = selected_patient_lookup.get(export_patient, {})
        export_meta = {
            "export_batch": str(batch_value),
            "export_roi": str(roi_value),
            "export_patient": export_patient,
            "export_roi_cell_count": int(len(region_df)),
            "export_selected_hit_zone_cell_count": int(len(selected_zone_index)),
            "export_all_roi_target_motif_instance_count": int(len(instances)),
            "export_selected_hit_score_type": score_type,
        }
        rows: list[dict[str, Any]] = []

        for instance_rank, instance in enumerate(instances, start=1):
            member_rows: list[dict[str, Any]] = []
            member_in_range_flags: list[bool] = []
            for region_position in instance:
                if int(region_position) >= len(region_export):
                    continue
                cell_row = region_export.iloc[int(region_position)].to_dict()
                member_rows.append(cell_row)
                member_in_range_flags.append(int(cell_row.get("unified_csv_row_index", -1)) in selected_zone_index)
            if not member_rows:
                continue

            centroid_x = float("nan")
            centroid_y = float("nan")
            if centroids.ndim == 2 and instance_rank - 1 < len(centroids):
                centroid_x = float(centroids[instance_rank - 1, 0])
                centroid_y = float(centroids[instance_rank - 1, 1])
            if all(member_in_range_flags):
                range_status = "within_selected_significant_range"
            elif any(member_in_range_flags):
                range_status = "overlaps_selected_significant_range"
            else:
                range_status = "outside_selected_significant_range"
            motif_id = "::".join(
                [
                    str(section_label).lower(),
                    normalize_image_stem(str(batch_value)),
                    normalize_image_stem(str(roi_value)),
                    normalize_image_stem(str(hit_row.get("feature_name") or hit_row.get("feature_label") or "motif")),
                    f"{int(instance_rank):05d}",
                ]
            )

            for cell_ordinal, (cell_row, cell_in_range) in enumerate(zip(member_rows, member_in_range_flags), start=1):
                export_row = {
                    "source_section": str(section_label),
                    "motif_export_scope": "all_roi_instances",
                    "motif_instance_id": motif_id,
                    "motif_instance_rank": int(instance_rank),
                    "motif_instance_size": int(len(member_rows)),
                    "motif_target_phenotypes": "::".join(target_sorted),
                    "motif_centroid_x": centroid_x,
                    "motif_centroid_y": centroid_y,
                    "motif_cells_in_selected_hit_range": int(sum(member_in_range_flags)),
                    "motif_any_cell_in_selected_hit_range": bool(any(member_in_range_flags)),
                    "motif_all_cells_in_selected_hit_range": bool(all(member_in_range_flags)),
                    "motif_selected_hit_range_status": range_status,
                    "motif_selected_hit_zone_membership": "::".join(
                        "in_selected_hit_range" if flag else "outside_selected_hit_range"
                        for flag in member_in_range_flags
                    ),
                    "motif_cell_ordinal": int(cell_ordinal),
                    "motif_cell_role": f"cell_{int(cell_ordinal)}",
                    "motif_cell_in_selected_hit_range": bool(cell_in_range),
                    "selected_hit_zone_start": float(zone_start),
                    "selected_hit_zone_end": float(zone_end),
                    "selected_hit_zone_label": selected_zone_label,
                    "selected_hit_significant_zone_labels": significant_zone_text,
                }
                export_row.update(hit_meta)
                export_row.update(export_meta)
                export_row.update(roi_meta)
                export_row.update(patient_meta)
                export_row.update(cell_row)
                rows.append(export_row)

        if rows:
            cell_frames.append(pd.DataFrame(rows))

    if not cell_frames:
        return pd.DataFrame(), pd.DataFrame()

    cell_df = pd.concat(cell_frames, ignore_index=True)
    instance_df = _build_representative_motif_instance_summary_df(
        cell_df,
        phenotype_col=phenotype_col,
        x_col=x_col,
        y_col=y_col,
    )
    return cell_df, instance_df


def _render_selected_hit_motif_export_section(
    analysis_df: pd.DataFrame,
    stats_all: pd.DataFrame,
    config: dict[str, Any],
    hit_row: pd.Series,
    section_label: str,
    zone_start: float,
    zone_end: float,
    key_prefix: str,
    selected_roi_features_df: pd.DataFrame | None = None,
    selected_patient_export_df: pd.DataFrame | None = None,
    selected_patient_features_df: pd.DataFrame | None = None,
) -> None:
    feature_family = str(hit_row.get("feature_family") or "")
    if feature_family not in {"dyad", "triad"}:
        return

    target_phenotypes = _representative_target_phenotypes(hit_row)
    if len(target_phenotypes) not in {2, 3}:
        return

    st.markdown("#### All ROI Motif Extraction CSV")
    st.caption(
        "Build one export for the currently selected dyad or triad across every ROI in the current in-memory dataset. "
        "The cell-level CSV keeps all original unified-CSV columns and adds `motif_selected_hit_range_status`, `motif_cell_in_selected_hit_range`, `selected_hit_significant_zone_labels`, `selected_hit_roi_feature_value`, `selected_hit_patient_feature_value`, and `selected_hit_patient_screen_group` so you can replot all motif instances and filter by ROI score or patient High/Low group."
    )

    signature = _selected_hit_motif_export_signature(hit_row, section_label, zone_start, zone_end)
    cache_key = f"_{key_prefix}_all_roi_motif_export"
    cached = st.session_state.get(cache_key, {})
    if st.button(
        f"Prepare all ROI {feature_family} extraction CSV",
        key=f"{key_prefix}_prepare_all_roi_motif_export",
        width="stretch",
        help="Enumerates the selected dyad or triad across every ROI in the current dataset, not only the representative ones.",
    ):
        with st.spinner(f"Building all ROI {feature_family} extraction export..."):
            cell_df, instance_df = _build_selected_hit_all_roi_motif_exports(
                analysis_df=analysis_df,
                stats_all=stats_all,
                config=config,
                hit_row=hit_row,
                section_label=section_label,
                zone_start=zone_start,
                zone_end=zone_end,
                selected_roi_features_df=selected_roi_features_df,
                selected_patient_export_df=selected_patient_export_df,
                selected_patient_features_df=selected_patient_features_df,
            )
        st.session_state[cache_key] = {
            "signature": signature,
            "cell_df": cell_df,
            "instance_df": instance_df,
        }
        cached = st.session_state.get(cache_key, {})

    if not cached or cached.get("signature") != signature:
        return

    cell_df = cached.get("cell_df", pd.DataFrame())
    instance_df = cached.get("instance_df", pd.DataFrame())
    if not isinstance(cell_df, pd.DataFrame) or cell_df.empty:
        st.info("No motif instances matching the selected dyad or triad were found across the current ROIs.")
        return

    instance_count = int(cell_df["motif_instance_id"].astype(str).nunique()) if "motif_instance_id" in cell_df.columns else 0
    roi_count = int(cell_df[["export_batch", "export_roi"]].drop_duplicates().shape[0]) if {"export_batch", "export_roi"}.issubset(cell_df.columns) else 0
    within_count = int((cell_df.groupby("motif_instance_id")["motif_all_cells_in_selected_hit_range"].first()).sum()) if {"motif_instance_id", "motif_all_cells_in_selected_hit_range"}.issubset(cell_df.columns) else 0
    st.caption(
        f"Prepared {instance_count} {feature_family} instance(s) across {roi_count} ROI(s). "
        f"{within_count} instance(s) are fully inside the selected hit range {format_zone_interval_label(zone_start, zone_end)}."
    )

    feature_stub = str(hit_row.get("feature_name") or hit_row.get("feature_label") or feature_family).replace("::", "_").replace("/", "_")
    _render_dataframe_download(
        cell_df,
        f"Download all ROI {feature_family} cell-level extraction CSV",
        f"{str(section_label).lower()}_{feature_stub}_all_roi_cells.csv",
            f"{key_prefix}_all_roi_motif_cell_csv_download",
    )
    if isinstance(instance_df, pd.DataFrame) and not instance_df.empty:
        _render_dataframe_download(
            instance_df,
            f"Download all ROI {feature_family} instance-level extraction CSV",
            f"{str(section_label).lower()}_{feature_stub}_all_roi_instances.csv",
            f"{key_prefix}_all_roi_motif_instance_csv_download",
        )


def _representative_target_phenotypes(hit_row: pd.Series) -> list[str]:
    feature_family = str(hit_row.get("feature_family", ""))
    parts = [str(part) for part in str(hit_row.get("feature_name", "")).split("::")[1:] if str(part)]
    if feature_family in {"density", "proportion"}:
        return parts[:1]
    if feature_family in {"ratio", "dyad", "triad", "nn_distance"}:
        return parts
    return []


def _publication_group_order(groups: list[str]) -> list[str]:
    unique_groups = [str(group) for group in groups if str(group)]
    ordered: list[str] = []
    for preferred in ["High", "Low", "Case", "Control", "Positive", "Negative", "Responder", "Non-responder"]:
        for group in unique_groups:
            if group not in ordered and group.lower() == preferred.lower():
                ordered.append(group)
    for group in unique_groups:
        if group not in ordered:
            ordered.append(group)
    return ordered


def _publication_feature_value_label(hit_row: pd.Series) -> str:
    family = str(hit_row.get("feature_family") or "").strip().lower()
    if family == "triad":
        return "ROI triad enrichment score"
    if family == "dyad":
        return "ROI dyad enrichment score"
    if family == "nn_distance":
        return "ROI nearest-neighbour feature value"
    if family == "ratio":
        feature_label = str(hit_row.get("feature_label") or "")
        if feature_label.startswith("Log2"):
            return "ROI log\u2082 ratio feature value"
        return "ROI ratio feature value"
    if family in {"density", "proportion"}:
        return f"ROI {family} feature value"
    return "ROI feature value"


def _publication_group_metadata(
    hit_row: pd.Series,
    patient_features: pd.DataFrame,
    patient_export_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, str], dict[str, float], dict[str, str], str]:
    patient_group_lookup: dict[str, str] = {}
    patient_value_lookup: dict[str, float] = {}
    patient_color_lookup: dict[str, str] = {}
    group_label = "Patient group"

    if isinstance(patient_export_df, pd.DataFrame) and not patient_export_df.empty:
        group_col = "screen_group" if "screen_group" in patient_export_df.columns else "endpoint_group" if "endpoint_group" in patient_export_df.columns else ""
        if group_col:
            group_label = "Patient enrichment group" if group_col == "screen_group" else "Endpoint group"
            for row in patient_export_df.itertuples(index=False):
                row_dict = row._asdict()
                patient_id = str(row_dict.get("patient") or "")
                if not patient_id:
                    continue
                patient_group_lookup[patient_id] = str(row_dict.get(group_col) or "")
                patient_value_lookup[patient_id] = _to_float(row_dict.get("feature_value"), float("nan"))
                patient_color = str(row_dict.get("screen_group_color") or "")
                if patient_color:
                    patient_color_lookup[str(row_dict.get(group_col) or "")] = patient_color

    if patient_group_lookup:
        return patient_group_lookup, patient_value_lookup, patient_color_lookup, group_label

    if isinstance(patient_features, pd.DataFrame) and not patient_features.empty:
        feature_filter = (
            (patient_features["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
            & (patient_features["zone_width"].astype(float) == float(hit_row.get("zone_width") or 0.0))
            & (patient_features["stratum_label"].astype(str) == str(hit_row.get("stratum_label") or ""))
        )
        filtered = patient_features.loc[feature_filter].copy()
        if not filtered.empty:
            cutpoint = _to_float(hit_row.get("cutpoint"), float("nan"))
            if not np.isfinite(cutpoint):
                cutpoint = float(pd.to_numeric(filtered["value"], errors="coerce").median())
            values = pd.to_numeric(filtered["value"], errors="coerce")
            filtered["__publication_group__"] = np.where(values >= float(cutpoint), "High", "Low")
            for row in filtered.itertuples(index=False):
                patient_id = str(getattr(row, "patient", "") or "")
                if not patient_id:
                    continue
                patient_group_lookup[patient_id] = str(getattr(row, "__publication_group__", "") or "")
                patient_value_lookup[patient_id] = _to_float(getattr(row, "value", float("nan")), float("nan"))
            patient_color_lookup = {"High": "#c43d3d", "Low": "#2d6cdf"}
            group_label = "Patient enrichment group"

    return patient_group_lookup, patient_value_lookup, patient_color_lookup, group_label


def _build_publication_roi_support_df(
    analysis_df: pd.DataFrame,
    combined_hit: pd.DataFrame,
    config: dict[str, Any],
    hit_row: pd.Series,
    patient_features: pd.DataFrame,
    patient_export_df: pd.DataFrame,
    zone_start: float,
    zone_end: float,
) -> tuple[pd.DataFrame, str]:
    if analysis_df.empty or combined_hit.empty:
        return pd.DataFrame(), "Patient group"

    batch_col = str(config["columns"]["batch"])
    roi_col = str(config["columns"]["roi"])
    patient_col = str(config["columns"]["patient"])
    phenotype_col = str(config["columns"]["phenotype"])
    target_phenotypes = _representative_target_phenotypes(hit_row)
    target_stubs = {phenotype: normalize_image_stem(str(phenotype)).replace("-", "_") for phenotype in target_phenotypes}

    patient_group_lookup, patient_value_lookup, patient_color_lookup, group_label = _publication_group_metadata(
        hit_row=hit_row,
        patient_features=patient_features,
        patient_export_df=patient_export_df,
        config=config,
    )

    requested_keys = {
        (str(row.batch), str(row.roi))
        for row in combined_hit[["batch", "roi"]].drop_duplicates().itertuples(index=False)
    }
    region_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for (batch_value, roi_value), group_df in analysis_df.groupby([batch_col, roi_col], dropna=False, sort=False):
        key = (str(batch_value), str(roi_value))
        if key in requested_keys:
            region_lookup[key] = group_df.copy()

    rows: list[dict[str, Any]] = []
    deduped = combined_hit.sort_values("value", ascending=False, na_position="last").drop_duplicates(subset=["batch", "roi"]).copy()
    for row in deduped.itertuples(index=False):
        region_key = (str(row.batch), str(row.roi))
        region_df = region_lookup.get(region_key, pd.DataFrame())
        if region_df.empty:
            continue
        zone_df = _representative_zone_subset(region_df, config, zone_end, zone_start)
        region_patient_values = region_df[patient_col].dropna().astype(str).tolist() if patient_col in region_df.columns else []
        patient_id = str(getattr(row, "patient", "") or (region_patient_values[0] if region_patient_values else "") or "")
        summary_row: dict[str, Any] = {
            "batch": str(row.batch),
            "roi": str(row.roi),
            "patient": patient_id,
            "roi_feature_value": _to_float(getattr(row, "value", float("nan")), float("nan")),
            "patient_group": patient_group_lookup.get(patient_id, "Ungrouped"),
            "patient_feature_value": patient_value_lookup.get(patient_id, float("nan")),
            "patient_group_color": patient_color_lookup.get(patient_group_lookup.get(patient_id, ""), ""),
            "roi_total_cells": int(len(region_df)),
            "zone_total_cells": int(len(zone_df)),
            "zone_label": format_zone_interval_label(zone_start, zone_end),
        }
        for phenotype in target_phenotypes:
            stub = target_stubs[phenotype]
            roi_count = int((region_df[phenotype_col].astype(str) == str(phenotype)).sum()) if phenotype_col in region_df.columns else 0
            zone_count = int((zone_df[phenotype_col].astype(str) == str(phenotype)).sum()) if phenotype_col in zone_df.columns else 0
            summary_row[f"{stub}_roi_count"] = roi_count
            summary_row[f"{stub}_roi_fraction"] = float(roi_count) / float(len(region_df)) if len(region_df) else float("nan")
            summary_row[f"{stub}_zone_count"] = zone_count
            summary_row[f"{stub}_zone_fraction"] = float(zone_count) / float(len(zone_df)) if len(zone_df) else float("nan")
        rows.append(summary_row)

    support_df = pd.DataFrame(rows)
    if support_df.empty:
        return support_df, group_label

    ordered_groups = _publication_group_order(support_df["patient_group"].astype(str).tolist())
    group_rank = {group: idx for idx, group in enumerate(ordered_groups)}
    support_df["_group_rank"] = support_df["patient_group"].map(group_rank).fillna(len(group_rank)).astype(int)
    support_df = support_df.sort_values(["_group_rank", "roi_feature_value", "patient", "roi"], ascending=[True, False, True, True], na_position="last").reset_index(drop=True)
    return support_df.drop(columns=["_group_rank"], errors="ignore"), group_label


def _select_publication_representative_roi_rows(
    roi_support_df: pd.DataFrame,
    hit_row: pd.Series,
    target_phenotypes: list[str],
) -> tuple[pd.DataFrame, str]:
    if roi_support_df.empty:
        return pd.DataFrame(), ""

    ordered_groups = _publication_group_order(roi_support_df["patient_group"].astype(str).tolist())
    high_group = next((group for group in ordered_groups if group.lower() == "high"), ordered_groups[0] if ordered_groups else "")
    low_group = next((group for group in ordered_groups if group.lower() == "low"), ordered_groups[1] if len(ordered_groups) > 1 else ordered_groups[0] if ordered_groups else "")

    high_pool = roi_support_df.loc[roi_support_df["patient_group"].astype(str) == str(high_group)].copy()
    low_pool = roi_support_df.loc[roi_support_df["patient_group"].astype(str) == str(low_group)].copy()
    if high_pool.empty or low_pool.empty or str(high_group) == str(low_group):
        fallback = roi_support_df.sort_values("roi_feature_value", ascending=False, na_position="last")
        if fallback.empty:
            return pd.DataFrame(), ""
        selected_rows = pd.concat([fallback.head(1), fallback.tail(1)], ignore_index=True).drop_duplicates(subset=["batch", "roi"]).head(2).copy()
        if len(selected_rows) < 2:
            selected_rows = fallback.head(min(2, len(fallback))).copy()
        selected_rows["publication_role"] = ["High-score example", "Low-score example"][: len(selected_rows)]
        return selected_rows, "Representative ROIs were chosen as the highest and lowest ROI-level examples because a matched High-vs-Low pair was not available."

    high_pool = high_pool.sort_values("roi_feature_value", ascending=False, na_position="last").head(12).copy()
    low_pool = low_pool.sort_values("roi_feature_value", ascending=True, na_position="last").head(12).copy()

    abundance_cols = []
    for phenotype in target_phenotypes:
        stub = normalize_image_stem(str(phenotype)).replace("-", "_")
        for suffix in ["roi_fraction", "zone_fraction"]:
            column = f"{stub}_{suffix}"
            if column in roi_support_df.columns:
                abundance_cols.append(column)

    pair_rows: list[dict[str, Any]] = []
    for high_row in high_pool.to_dict("records"):
        for low_row in low_pool.to_dict("records"):
            if not high_row or not low_row:
                continue
            same_patient = str(high_row.get("patient") or "") == str(low_row.get("patient") or "")
            score_gap = _to_float(high_row.get("roi_feature_value"), float("nan")) - _to_float(low_row.get("roi_feature_value"), float("nan"))
            abundance_distance = float("nan")
            if abundance_cols:
                diffs: list[float] = []
                for column in abundance_cols:
                    high_value = _to_float(high_row.get(column), float("nan"))
                    low_value = _to_float(low_row.get(column), float("nan"))
                    if np.isfinite(high_value) and np.isfinite(low_value):
                        diffs.append(abs(high_value - low_value))
                if diffs:
                    abundance_distance = float(np.mean(diffs))
            pair_rows.append(
                {
                    "high_batch": str(high_row.get("batch") or ""),
                    "high_roi": str(high_row.get("roi") or ""),
                    "low_batch": str(low_row.get("batch") or ""),
                    "low_roi": str(low_row.get("roi") or ""),
                    "same_patient": bool(same_patient),
                    "score_gap": score_gap,
                    "abundance_distance": abundance_distance,
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        return pd.DataFrame(), ""

    if abundance_cols and pair_df["abundance_distance"].notna().any():
        score_threshold = float(pair_df["score_gap"].quantile(0.50)) if len(pair_df) > 1 else float(pair_df["score_gap"].iloc[0])
        filtered_pairs = pair_df.loc[pair_df["score_gap"] >= score_threshold].copy()
        if filtered_pairs.empty:
            filtered_pairs = pair_df.copy()
        filtered_pairs = filtered_pairs.sort_values(["same_patient", "abundance_distance", "score_gap"], ascending=[True, True, False], na_position="last")
        best_pair = filtered_pairs.iloc[0]
        selection_note = (
            "Publication representatives were chosen as a High-vs-Low pair with similar target-phenotype abundance "
            "but separated ROI-level feature scores, so the image comparison emphasizes spatial organization rather than simple frequency."
        )
        if np.isfinite(_to_float(best_pair.get("abundance_distance"), float("nan"))):
            selection_note += f" Mean abundance distance across the matched phenotype fractions: {float(best_pair['abundance_distance']) * 100:.1f} percentage points."
    else:
        filtered_pairs = pair_df.sort_values(["same_patient", "score_gap"], ascending=[True, False], na_position="last")
        best_pair = filtered_pairs.iloc[0]
        selection_note = "Publication representatives were chosen as the most separated High-vs-Low ROI pair because no reliable abundance-matching variables were available."

    selected = roi_support_df.loc[
        ((roi_support_df["batch"].astype(str) == str(best_pair["high_batch"])) & (roi_support_df["roi"].astype(str) == str(best_pair["high_roi"])))
        | ((roi_support_df["batch"].astype(str) == str(best_pair["low_batch"])) & (roi_support_df["roi"].astype(str) == str(best_pair["low_roi"])))
    ].copy()
    if selected.empty:
        return pd.DataFrame(), selection_note

    selected["publication_role"] = np.where(
        (selected["batch"].astype(str) == str(best_pair["high_batch"])) & (selected["roi"].astype(str) == str(best_pair["high_roi"])),
        f"{high_group} representative",
        f"{low_group} representative",
    )
    role_order = {f"{high_group} representative": 0, f"{low_group} representative": 1}
    selected["_role_rank"] = selected["publication_role"].map(role_order).fillna(99).astype(int)
    selected = selected.sort_values(["_role_rank", "roi_feature_value"], ascending=[True, False], na_position="last").drop(columns=["_role_rank"], errors="ignore")
    return selected.reset_index(drop=True), selection_note


def _plot_publication_roi_score_support(
    roi_support_df: pd.DataFrame,
    hit_row: pd.Series,
    target_phenotypes: list[str],
    group_label: str,
) -> Figure:
    plot_df = roi_support_df.copy()
    fig_width = max(10.0, min(4.0 + 0.65 * max(len(plot_df), 1), 28.0))
    fig, (ax, ax_table) = plt.subplots(
        2,
        1,
        figsize=(fig_width, 8.8),
        gridspec_kw={"height_ratios": [3.4, 2.6]},
        constrained_layout=True,
    )

    if plot_df.empty:
        ax.text(0.5, 0.5, "No ROI-level values are available for this feature.", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        ax_table.axis("off")
        return fig

    ordered_groups = _publication_group_order(plot_df["patient_group"].astype(str).tolist())
    group_rank = {group: idx for idx, group in enumerate(ordered_groups)}
    plot_df["_group_rank"] = plot_df["patient_group"].map(group_rank).fillna(len(group_rank)).astype(int)
    plot_df = plot_df.sort_values(["_group_rank", "roi_feature_value", "patient", "roi"], ascending=[True, False, True, True], na_position="last").reset_index(drop=True)
    x_positions = np.arange(len(plot_df), dtype=float)
    fallback_palette = ["#c43d3d", "#2d6cdf", "#2ca25f", "#9467bd", "#7f8c8d"]
    color_lookup = {
        group: next(
            (
                color
                for color in plot_df.loc[plot_df["patient_group"].astype(str) == str(group), "patient_group_color"].astype(str).tolist()
                if color
            ),
            fallback_palette[idx % len(fallback_palette)],
        )
        for idx, group in enumerate(ordered_groups)
    }

    if ordered_groups:
        for group in ordered_groups:
            mask = plot_df["patient_group"].astype(str) == str(group)
            if not bool(mask.any()):
                continue
            idxs = np.flatnonzero(mask.to_numpy())
            color = color_lookup.get(str(group), "#d9d9d9")
            ax.axvspan(float(idxs.min()) - 0.5, float(idxs.max()) + 0.5, color=color, alpha=0.06, zorder=0)
            median_value = pd.to_numeric(plot_df.loc[mask, "roi_feature_value"], errors="coerce").median()
            if np.isfinite(median_value):
                ax.hlines(
                    float(median_value),
                    float(idxs.min()) - 0.35,
                    float(idxs.max()) + 0.35,
                    colors=color,
                    linewidth=2.0,
                    linestyles="-",
                    zorder=2,
                )
            ax.text(
                float(idxs.mean()),
                1.01,
                str(group),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.scatter(
        x_positions,
        pd.to_numeric(plot_df["roi_feature_value"], errors="coerce"),
        s=52,
        c=[color_lookup.get(str(group), "#7f8c8d") for group in plot_df["patient_group"].astype(str).tolist()],
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.plot(
        x_positions,
        pd.to_numeric(plot_df["roi_feature_value"], errors="coerce"),
        color="#7f8c8d",
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )
    ax.set_xlim(-0.6, max(float(len(plot_df)) - 0.4, 0.6))
    ax.set_ylabel(_publication_feature_value_label(hit_row), fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["roi"].astype(str).tolist(), rotation=90, fontsize=7)
    ax.set_xlabel("ROI", fontsize=9)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"ROI-level support for {str(hit_row.get('feature_label') or hit_row.get('feature_name') or 'selected feature')}",
        loc="left",
        fontsize=11,
    )

    table_rows: list[list[str]] = []
    row_labels: list[str] = []
    row_labels.append("Patient")
    table_rows.append(plot_df["patient"].astype(str).tolist())
    row_labels.append(group_label)
    table_rows.append(plot_df["patient_group"].astype(str).tolist())
    row_labels.append("ROI score")
    table_rows.append([
        f"{float(value):.3f}" if np.isfinite(_to_float(value, float('nan'))) else "-"
        for value in plot_df["roi_feature_value"].tolist()
    ])
    if pd.to_numeric(plot_df["patient_feature_value"], errors="coerce").notna().any():
        row_labels.append("Patient score")
        table_rows.append([
            f"{float(value):.3f}" if np.isfinite(_to_float(value, float('nan'))) else "-"
            for value in plot_df["patient_feature_value"].tolist()
        ])
    row_labels.append("ROI cells")
    table_rows.append([str(int(_to_float(value, 0.0))) for value in plot_df["roi_total_cells"].tolist()])
    row_labels.append("Boundary cells")
    table_rows.append([str(int(_to_float(value, 0.0))) for value in plot_df["zone_total_cells"].tolist()])
    for phenotype in target_phenotypes:
        stub = normalize_image_stem(str(phenotype)).replace("-", "_")
        roi_count_col = f"{stub}_roi_count"
        roi_frac_col = f"{stub}_roi_fraction"
        zone_count_col = f"{stub}_zone_count"
        zone_frac_col = f"{stub}_zone_fraction"
        if roi_count_col in plot_df.columns and roi_frac_col in plot_df.columns:
            row_labels.append(f"{phenotype} in ROI")
            table_rows.append([
                f"{int(_to_float(count, 0.0))} ({_to_float(frac, 0.0) * 100:.1f}%)"
                for count, frac in zip(plot_df[roi_count_col].tolist(), plot_df[roi_frac_col].tolist())
            ])
        if zone_count_col in plot_df.columns and zone_frac_col in plot_df.columns:
            row_labels.append(f"{phenotype} in boundary")
            table_rows.append([
                f"{int(_to_float(count, 0.0))} ({_to_float(frac, 0.0) * 100:.1f}%)"
                for count, frac in zip(plot_df[zone_count_col].tolist(), plot_df[zone_frac_col].tolist())
            ])

    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_rows,
        rowLabels=row_labels,
        colLabels=plot_df["roi"].astype(str).tolist(),
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.28)
    return fig


def _plot_publication_patient_roi_support(
    roi_support_df: pd.DataFrame,
    hit_row: pd.Series,
    group_label: str,
) -> Figure:
    plot_df = roi_support_df.copy()
    n_patients = int(plot_df["patient"].astype(str).nunique()) if not plot_df.empty and "patient" in plot_df.columns else 0
    fig_width = max(10.0, min(5.0 + 0.55 * max(n_patients, 1), 26.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), constrained_layout=True)

    if plot_df.empty:
        ax.text(0.5, 0.5, "No patient-level ROI values are available for this feature.", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    ordered_groups = _publication_group_order(plot_df["patient_group"].astype(str).tolist())
    fallback_palette = ["#c43d3d", "#2d6cdf", "#2ca25f", "#9467bd", "#7f8c8d"]
    color_lookup = {
        group: next(
            (
                color
                for color in plot_df.loc[plot_df["patient_group"].astype(str) == str(group), "patient_group_color"].astype(str).tolist()
                if color
            ),
            fallback_palette[idx % len(fallback_palette)],
        )
        for idx, group in enumerate(ordered_groups)
    }

    patient_df = plot_df[["patient", "patient_group", "patient_feature_value", "patient_group_color"]].drop_duplicates().copy()
    group_rank = {group: idx for idx, group in enumerate(ordered_groups)}
    patient_df["_group_rank"] = patient_df["patient_group"].astype(str).map(group_rank).fillna(len(group_rank)).astype(int)
    patient_df["_patient_sort_value"] = pd.to_numeric(patient_df["patient_feature_value"], errors="coerce")
    patient_df = patient_df.sort_values(
        ["_group_rank", "_patient_sort_value", "patient"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    patient_positions = {str(patient): float(idx) for idx, patient in enumerate(patient_df["patient"].astype(str).tolist())}
    rng = np.random.default_rng(20260429)

    if ordered_groups:
        for group in ordered_groups:
            patient_mask = patient_df["patient_group"].astype(str) == str(group)
            if not bool(patient_mask.any()):
                continue
            idxs = np.flatnonzero(patient_mask.to_numpy())
            color = color_lookup.get(str(group), "#d9d9d9")
            ax.axvspan(float(idxs.min()) - 0.5, float(idxs.max()) + 0.5, color=color, alpha=0.06, zorder=0)
            group_values = pd.to_numeric(
                plot_df.loc[plot_df["patient_group"].astype(str) == str(group), "roi_feature_value"],
                errors="coerce",
            )
            group_values = group_values[np.isfinite(group_values)]
            if not group_values.empty:
                ax.hlines(
                    float(group_values.median()),
                    float(idxs.min()) - 0.35,
                    float(idxs.max()) + 0.35,
                    colors=color,
                    linewidth=2.0,
                    linestyles="-",
                    zorder=2,
                )
            ax.text(
                float(idxs.mean()),
                1.01,
                str(group),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    for patient_row in patient_df.itertuples(index=False):
        patient_id = str(patient_row.patient)
        patient_group = str(patient_row.patient_group)
        x_pos = float(patient_positions.get(patient_id, 0.0))
        patient_points = plot_df.loc[plot_df["patient"].astype(str) == patient_id].copy()
        values = pd.to_numeric(patient_points["roi_feature_value"], errors="coerce")
        valid_mask = np.isfinite(values.to_numpy(dtype=float))
        patient_points = patient_points.loc[valid_mask].copy()
        values = pd.to_numeric(patient_points["roi_feature_value"], errors="coerce")
        if patient_points.empty:
            continue
        jitter = rng.normal(0.0, 0.08, size=len(patient_points))
        x_values = np.full(len(patient_points), x_pos, dtype=float) + jitter
        color = color_lookup.get(patient_group, "#7f8c8d")
        if len(values) > 1:
            ax.vlines(x_pos, float(values.min()), float(values.max()), color=color, alpha=0.25, linewidth=1.0, zorder=1)
        ax.scatter(
            x_values,
            values.to_numpy(dtype=float),
            s=52,
            c=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        patient_median = float(values.median())
        if np.isfinite(patient_median):
            ax.hlines(patient_median, x_pos - 0.22, x_pos + 0.22, colors=color, linewidth=2.0, zorder=4)

    ax.set_xlim(-0.6, max(float(len(patient_df)) - 0.4, 0.6))
    ax.set_ylabel(_publication_feature_value_label(hit_row), fontsize=10)
    ax.set_xticks(np.arange(len(patient_df), dtype=float))
    ax.set_xticklabels(patient_df["patient"].astype(str).tolist(), rotation=90, fontsize=7)
    ax.set_xlabel("Patient", fontsize=9)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Patient-level ROI support for {str(hit_row.get('feature_label') or hit_row.get('feature_name') or 'selected feature')}",
        loc="left",
        fontsize=11,
    )
    ax.text(
        0.01,
        -0.20,
        "Each dot is one ROI. Patients with multiple ROIs show multiple dots at the same patient position.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    return fig


def _build_publication_range_hit_df(
    candidate_summary: pd.DataFrame | None = None,
    stats_all: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if isinstance(candidate_summary, pd.DataFrame) and not candidate_summary.empty and "zone_width" in candidate_summary.columns:
        work = candidate_summary.copy()
        plot_df = pd.DataFrame(
            {
                "zone_width": pd.to_numeric(work.get("zone_width", pd.Series(dtype=float)), errors="coerce"),
                "zone_label": work.get("zone_label", pd.Series(dtype=str)).fillna("").astype(str),
                "n_p_value_hits": pd.to_numeric(work.get("n_p_value_hits", pd.Series(dtype=float)), errors="coerce").fillna(0),
                "n_discovery": pd.to_numeric(work.get("n_discovery", pd.Series(dtype=float)), errors="coerce").fillna(0),
                "n_significant": pd.to_numeric(work.get("n_significant", pd.Series(dtype=float)), errors="coerce").fillna(0),
                "bootstrap_reference_recovery": pd.to_numeric(work.get("bootstrap_reference_recovery", pd.Series(dtype=float)), errors="coerce"),
                "patient_coverage_fraction": pd.to_numeric(work.get("patient_coverage_fraction", pd.Series(dtype=float)), errors="coerce"),
                "roi_coverage_fraction": pd.to_numeric(work.get("roi_coverage_fraction", pd.Series(dtype=float)), errors="coerce"),
                "bootstrap_sampling_unit": work.get("bootstrap_sampling_unit", pd.Series("patient", index=work.index)).fillna("patient").astype(str),
            }
        )
    elif isinstance(stats_all, pd.DataFrame) and not stats_all.empty and "zone_width" in stats_all.columns:
        rows: list[dict[str, Any]] = []
        for width_value, width_df in stats_all.groupby("zone_width", dropna=False, sort=True):
            numeric_width = _to_float(width_value, float("nan"))
            if not np.isfinite(numeric_width):
                continue
            p_values = pd.to_numeric(width_df.get("p_value", pd.Series(dtype=float)), errors="coerce")
            sig_labels = width_df.get("significance_label", pd.Series(dtype=str)).astype(str)
            bootstrap_value = pd.to_numeric(width_df.get("bootstrap_reference_recovery", pd.Series(dtype=float)), errors="coerce")
            coverage_value = pd.to_numeric(width_df.get("patient_coverage_fraction", pd.Series(dtype=float)), errors="coerce")
            rows.append(
                {
                    "zone_width": float(numeric_width),
                    "zone_label": str(width_df.get("zone_label", pd.Series(dtype=str)).dropna().astype(str).iloc[0]) if "zone_label" in width_df.columns and not width_df.get("zone_label", pd.Series(dtype=str)).dropna().empty else format_zone_label(numeric_width),
                    "n_p_value_hits": int((p_values <= 0.05).sum()),
                    "n_discovery": int(sig_labels.isin(["significant_fdr", "discovery_fdr"]).sum()),
                    "n_significant": int((sig_labels == "significant_fdr").sum()),
                    "bootstrap_reference_recovery": float(bootstrap_value.dropna().iloc[0]) if not bootstrap_value.dropna().empty else float("nan"),
                    "patient_coverage_fraction": float(coverage_value.dropna().iloc[0]) if not coverage_value.dropna().empty else float("nan"),
                    "roi_coverage_fraction": float("nan"),
                    "bootstrap_sampling_unit": "patient",
                }
            )
        plot_df = pd.DataFrame(rows)
    else:
        return pd.DataFrame()

    if plot_df.empty:
        return plot_df

    plot_df["zone_label"] = plot_df.apply(
        lambda row: str(row.get("zone_label") or format_zone_label(row.get("zone_width"))),
        axis=1,
    )
    plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["zone_width"], errors="coerce"))].copy()
    plot_df = plot_df.sort_values("zone_width", ascending=True, na_position="last").reset_index(drop=True)
    return plot_df


def _build_publication_zone_coverage_df(
    analysis_df: pd.DataFrame,
    combined: pd.DataFrame,
    patient_outcomes: pd.DataFrame,
    candidate_summary: pd.DataFrame | None,
    config: dict[str, Any],
    zone_start: float,
    zone_end: float,
) -> pd.DataFrame:
    if isinstance(candidate_summary, pd.DataFrame) and not candidate_summary.empty and "zone_width" in candidate_summary.columns:
        coverage_columns = [
            column
            for column in [
                "zone_width",
                "zone_label",
                "total_patients",
                "n_patients_with_zone",
                "patient_coverage_fraction",
                "n_patients_with_zone_cells",
                "n_patients_with_valid_zone_roi",
                "n_patients_without_zone_cells",
                "n_patients_below_min_cells_per_zone",
                "n_patients_without_computable_features",
                "n_patients_dropped_total",
            ]
            if column in candidate_summary.columns
        ]
        if coverage_columns:
            work = candidate_summary.loc[:, coverage_columns].copy().sort_values("zone_width", ascending=True, na_position="last").reset_index(drop=True)
            total_patients = int(patient_outcomes["patient"].dropna().astype(str).nunique()) if "patient" in patient_outcomes.columns else 0
            if "total_patients" not in work.columns and total_patients > 0:
                work["total_patients"] = total_patients
            if "n_patients_dropped_total" not in work.columns and {"total_patients", "n_patients_with_zone"}.issubset(work.columns):
                dropped = pd.to_numeric(work["total_patients"], errors="coerce").fillna(0) - pd.to_numeric(work["n_patients_with_zone"], errors="coerce").fillna(0)
                work["n_patients_dropped_total"] = dropped.clip(lower=0)
            return work

    fallback_row = build_zone_patient_coverage_row(
        analysis_df=analysis_df,
        config=config,
        width=float(zone_end),
        patient_outcomes=patient_outcomes,
        combined_features=combined,
        zone_start=float(zone_start) if np.isfinite(float(zone_start)) else None,
        zone_end=float(zone_end) if np.isfinite(float(zone_end)) else None,
    )
    return pd.DataFrame([fallback_row]) if fallback_row else pd.DataFrame()


def _selected_zone_coverage_row(zone_coverage_df: pd.DataFrame, selected_width: float) -> pd.Series | None:
    if zone_coverage_df.empty or "zone_width" not in zone_coverage_df.columns:
        return None
    widths = pd.to_numeric(zone_coverage_df["zone_width"], errors="coerce")
    selected_rows = zone_coverage_df.loc[widths.map(lambda value: bool(np.isfinite(value) and np.isclose(float(value), float(selected_width), atol=1e-6)))]
    if not selected_rows.empty:
        return selected_rows.iloc[0]
    return None


def _publication_zone_dropout_text(
    zone_row: pd.Series,
    config: dict[str, Any],
    patient_export_df: pd.DataFrame,
    hit_row: pd.Series,
    attrition_summary: dict[str, Any] | None = None,
    selected_candidate_width: float | None = None,
) -> str:
    zone_label = str(zone_row.get("zone_label") or format_zone_label(_to_float(zone_row.get("zone_width"), 0.0)))
    total_patients = int(pd.to_numeric(pd.Series([zone_row.get("total_patients")]), errors="coerce").fillna(0).iloc[0])
    n_patients_with_zone = int(pd.to_numeric(pd.Series([zone_row.get("n_patients_with_zone")]), errors="coerce").fillna(0).iloc[0])
    coverage_fraction = _to_float(zone_row.get("patient_coverage_fraction"), float("nan"))
    if not np.isfinite(coverage_fraction) and total_patients > 0:
        coverage_fraction = float(n_patients_with_zone) / float(total_patients)
    n_dropped_total = int(pd.to_numeric(pd.Series([zone_row.get("n_patients_dropped_total")]), errors="coerce").fillna(max(0, total_patients - n_patients_with_zone)).iloc[0])
    min_cells_per_zone = max(1, int(round(_to_float(config.get("screening", {}).get("min_cells_per_zone"), 20.0))))

    reason_parts: list[str] = []
    no_zone_cells = int(pd.to_numeric(pd.Series([zone_row.get("n_patients_without_zone_cells")]), errors="coerce").fillna(0).iloc[0])
    below_min_cells = int(pd.to_numeric(pd.Series([zone_row.get("n_patients_below_min_cells_per_zone")]), errors="coerce").fillna(0).iloc[0])
    feature_filtered = int(pd.to_numeric(pd.Series([zone_row.get("n_patients_without_computable_features")]), errors="coerce").fillna(0).iloc[0])
    if no_zone_cells > 0:
        reason_parts.append(f"{no_zone_cells} had no cells inside this zone")
    if below_min_cells > 0:
        reason_parts.append(f"{below_min_cells} had cells in the zone but no ROI reached the minimum {min_cells_per_zone} in-zone cells")
    if feature_filtered > 0:
        reason_parts.append(f"{feature_filtered} reached an eligible ROI but none of the enabled features were computable there")

    coverage_text = f"{n_patients_with_zone}/{total_patients} patients ({coverage_fraction * 100.0:.1f}%)" if total_patients > 0 and np.isfinite(coverage_fraction) else f"{n_patients_with_zone} patients"
    if n_dropped_total <= 0:
        summary = f"At {zone_label}, {coverage_text} contributed at least one analyzable feature. No patients were dropped before feature computation at this range."
    elif reason_parts:
        summary = f"At {zone_label}, {coverage_text} contributed at least one analyzable feature. Of the {n_dropped_total} patients excluded before feature scoring, " + "; ".join(reason_parts) + "."
    else:
        summary = f"At {zone_label}, {coverage_text} contributed at least one analyzable feature. {n_dropped_total} patients were excluded before feature scoring at this range."

    if isinstance(patient_export_df, pd.DataFrame) and not patient_export_df.empty and "patient" in patient_export_df.columns:
        endpoint_patients = int(patient_export_df["patient"].dropna().astype(str).nunique())
        if endpoint_patients < n_patients_with_zone:
            extra_dropped = n_patients_with_zone - endpoint_patients
            endpoint_type = str(hit_row.get("endpoint_type") or "")
            split_method = str(config.get("screening", {}).get("split_method") or "median")
            extra_reason = "missing endpoint values"
            if endpoint_type == "survival" and split_method in {"quartiles", "tertiles"}:
                extra_reason = f"missing endpoint values or middle-group exclusion from the {split_method} split"
            summary += f" The selected endpoint figure uses {endpoint_patients} patients, so {extra_dropped} additional patients from this zone drop only at the endpoint-testing step because of {extra_reason}."

    feature_summary = attrition_summary if isinstance(attrition_summary, dict) else {}
    hit_width = _to_float(hit_row.get("zone_width"), float("nan"))
    hit_zone_label = str(hit_row.get("zone_label") or format_zone_label(hit_width)) if np.isfinite(hit_width) else str(hit_row.get("zone_label") or "selected hit")
    n_feature_patients = int(_to_float(feature_summary.get("n_feature_patients"), 0.0))
    n_missing_endpoint_patients = int(_to_float(feature_summary.get("n_missing_endpoint_patients"), 0.0))
    n_split_excluded_patients = int(_to_float(feature_summary.get("n_split_excluded_patients"), 0.0))
    n_final_patients = int(_to_float(feature_summary.get("n_final_patients"), 0.0))
    if np.isfinite(_to_float(selected_candidate_width, float("nan"))) and np.isfinite(hit_width) and not np.isclose(float(selected_candidate_width), float(hit_width), atol=1e-6):
        summary += f" The highlighted hit itself is from {hit_zone_label}, while the discovery pipeline selected {format_zone_label(float(selected_candidate_width))} as the best candidate range."
    if n_feature_patients > 0:
        hit_parts = [f"For the highlighted feature at {hit_zone_label}, {n_feature_patients} patients had a computable feature value"]
        if n_missing_endpoint_patients > 0:
            hit_parts.append(f"{n_missing_endpoint_patients} then dropped for missing endpoint/value data")
        if n_split_excluded_patients > 0:
            hit_parts.append(f"{n_split_excluded_patients} then dropped at the High/Low split stage")
        if n_final_patients > 0:
            hit_parts.append(f"{n_final_patients} were used in the final test")
        summary += " " + "; ".join(hit_parts) + "."

    return summary


def _render_patient_inclusion_audit(
    audit_df: pd.DataFrame,
    download_label: str,
    download_filename: str,
    download_key: str,
) -> None:
    if audit_df.empty:
        return
    st.markdown("#### Patient Inclusion Audit")
    st.caption("This table includes both patients used in the figure and patients excluded from it, with the pipeline stage, supporting evidence, and exclusion reason.")
    preferred_columns = [
        "patient",
        "stratum_label",
        "km_status",
        "audit_stage",
        "audit_detail",
        "zone_status",
        "n_zone_rois",
        "n_valid_zone_rois",
        "feature_status",
        "feature_compute_stage",
        "feature_compute_detail",
        "triad_required_components",
        "triad_components_present_in_valid_roi",
        "triad_component_roi_count",
        "triad_qualifying_triangle_found",
        "triad_triangle_roi_count",
        "triad_max_observed_motif_count",
        "triad_min_obs_required",
        "feature_value",
        "has_zone_cells",
        "has_valid_zone_roi",
        "endpoint_status",
        "endpoint_time",
        "endpoint_censor",
        "endpoint_group",
        "screen_group",
        "split_status",
        "exclusion_reason",
    ]
    display_columns = [column for column in preferred_columns if column in audit_df.columns]
    display_df = audit_df.loc[:, display_columns].copy() if display_columns else audit_df.copy()
    st.dataframe(display_df, width="stretch", height=min(360, 80 + 28 * len(display_df)))
    _render_dataframe_download(
        audit_df,
        download_label,
        download_filename,
        download_key,
    )


def _plot_publication_component_abundance(
    roi_support_df: pd.DataFrame,
    hit_row: pd.Series,
    target_phenotypes: list[str],
    group_label: str,
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 4.8 + 2.2 * max(len(target_phenotypes), 1)), 4.8), constrained_layout=True)
    plot_df = roi_support_df.copy()
    if plot_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No component-abundance data are available.", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        return fig

    ordered_groups = _publication_group_order(plot_df["patient_group"].astype(str).tolist())
    fallback_palette = ["#c43d3d", "#2d6cdf", "#2ca25f", "#9467bd", "#7f8c8d"]
    color_lookup = {
        group: next(
            (
                color
                for color in plot_df.loc[plot_df["patient_group"].astype(str) == str(group), "patient_group_color"].astype(str).tolist()
                if color
            ),
            fallback_palette[idx % len(fallback_palette)],
        )
        for idx, group in enumerate(ordered_groups)
    }
    rng = np.random.default_rng(20260429)

    if not target_phenotypes:
        target_phenotypes = [str(hit_row.get("feature_family") or "feature")]
        plot_df["fallback_roi_fraction"] = 1.0
        plot_df["fallback_zone_fraction"] = 1.0

    for ax, suffix, title in zip(axes, ["roi", "zone"], ["Whole ROI abundance", "Boundary abundance"]):
        x_tick_positions: list[float] = []
        x_tick_labels: list[str] = []
        legend_seen: set[str] = set()
        for phenotype_idx, phenotype in enumerate(target_phenotypes):
            stub = normalize_image_stem(str(phenotype)).replace("-", "_")
            value_col = f"{stub}_{suffix}_fraction"
            if value_col not in plot_df.columns:
                value_col = f"fallback_{suffix}_fraction"
            group_positions = []
            for group_idx, group in enumerate(ordered_groups):
                pos = phenotype_idx * (len(ordered_groups) + 1) + group_idx
                mask = plot_df["patient_group"].astype(str) == str(group)
                values = pd.to_numeric(plot_df.loc[mask, value_col], errors="coerce") * 100.0
                values = values[np.isfinite(values)]
                if values.empty:
                    continue
                jitter = rng.normal(0.0, 0.06, size=len(values))
                label = str(group) if str(group) not in legend_seen else None
                ax.scatter(
                    np.full(len(values), pos, dtype=float) + jitter,
                    values.to_numpy(dtype=float),
                    s=34,
                    alpha=0.78,
                    c=color_lookup.get(str(group), "#7f8c8d"),
                    edgecolors="white",
                    linewidths=0.4,
                    label=label,
                    zorder=3,
                )
                legend_seen.add(str(group))
                median_value = float(np.nanmedian(values.to_numpy(dtype=float))) if len(values) else float("nan")
                if np.isfinite(median_value):
                    ax.hlines(median_value, pos - 0.22, pos + 0.22, colors=color_lookup.get(str(group), "#7f8c8d"), linewidth=2.0, zorder=2)
                group_positions.append(pos)
            if group_positions:
                x_tick_positions.append(float(np.mean(group_positions)))
                x_tick_labels.append(str(phenotype))

        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=9)
        ax.set_ylabel("Percent of cells", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ordered_groups:
            ax.legend(title=group_label, fontsize=8, title_fontsize=8, frameon=False, loc="upper right")

    fig.suptitle(
        f"Underlying component abundance for {str(hit_row.get('feature_label') or hit_row.get('feature_name') or 'selected feature')}",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    return fig


def _render_publication_representative_rois(
    analysis_df: pd.DataFrame,
    config: dict[str, Any],
    hit_row: pd.Series,
    roi_support_df: pd.DataFrame,
    section_label: str,
    zone_start: float,
    zone_end: float,
    key_prefix: str,
) -> None:
    target_phenotypes = _representative_target_phenotypes(hit_row)
    selected_rows, selection_note = _select_publication_representative_roi_rows(roi_support_df, hit_row, target_phenotypes)
    if selected_rows.empty:
        st.info("No representative ROI candidates are available for this feature yet.")
        return

    st.markdown("#### Publication representative ROIs")
    if selection_note:
        st.caption(selection_note)

    batch_col = str(config["columns"]["batch"])
    roi_col = str(config["columns"]["roi"])
    phenotype_col = str(config["columns"]["phenotype"])
    feature_family = str(hit_row.get("feature_family") or "")
    overlay_supported = feature_family in {"density", "proportion", "ratio", "dyad", "triad", "nn_distance"}
    image_lookup = _resolve_image_lookup(config["representatives"].get("raw_image_dir", ""))

    selected_image_paths: list[Path] = []
    active_phenotypes: list[str] = []
    region_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for row in selected_rows.itertuples(index=False):
        region_key = (str(row.batch), str(row.roi))
        region_df = analysis_df[
            (analysis_df[batch_col].astype(str) == region_key[0])
            & (analysis_df[roi_col].astype(str) == region_key[1])
        ].copy()
        if region_df.empty:
            continue
        region_lookup[region_key] = region_df
        active_phenotypes.extend(_safe_unique(region_df[phenotype_col]) if phenotype_col in region_df.columns else [])
        image_path = image_lookup.get(normalize_image_stem(region_key[0]).lower())
        if isinstance(image_path, Path) and image_path.exists():
            selected_image_paths.append(image_path)
    active_phenotypes = sorted({str(value) for value in active_phenotypes})

    marker_options, marker_default_colors = _resolve_representative_marker_options(selected_image_paths)
    selected_markers = [str(value) for value in st.session_state.get("tab8_fs_repr_selected_markers", []) if str(value) in marker_options]
    if not selected_markers:
        selected_markers = marker_options[: min(3, len(marker_options))]
    marker_color_mapping = {
        marker_name: st.session_state.get(
            f"tab8_fs_repr_marker_color__{normalize_image_stem(str(marker_name))}",
            marker_default_colors.get(marker_name, "#ffffff"),
        )
        for marker_name in selected_markers
    }
    color_mapping = _representative_default_colors(active_phenotypes or target_phenotypes)
    settings = {
        "render_mode": st.session_state.get("tab8_fs_repr_render_mode", "Scatter points"),
        "show_background": bool(st.session_state.get("tab8_fs_repr_show_background", True)),
        "show_legend": bool(st.session_state.get("tab8_fs_repr_show_legend", True)),
        "show_zone_overlay": True,
        "zone_hatched": bool(st.session_state.get("tab8_fs_repr_zone_hatched", True)),
        "dim_outside_zone": bool(st.session_state.get("tab8_fs_repr_dim_outside_zone", True)),
        "show_title": True,
        "point_alpha": float(st.session_state.get("tab8_fs_repr_point_alpha", 0.80)),
        "point_size": float(st.session_state.get("tab8_fs_repr_marker_size", 18.0)),
        "zone_color": st.session_state.get("tab8_fs_repr_zone_color", "#1f4e79"),
        "zone_linewidth": float(st.session_state.get("tab8_fs_repr_zone_linewidth", 1.5)),
        "voronoi_max_area": float(st.session_state.get("tab8_fs_repr_voronoi_max_area", 400.0)),
        "draw_centers": bool(st.session_state.get("tab8_fs_repr_draw_centers", False)),
        "draw_edges": bool(st.session_state.get("tab8_fs_repr_draw_edges", False)),
        "annotation_text": "",
        "feature_overlay_mode": _default_representative_feature_overlay_mode(feature_family) if overlay_supported else "None",
        "feature_overlay_color": st.session_state.get("tab8_fs_repr_feature_overlay_color", "#ff6b00"),
        "feature_overlay_alpha": float(st.session_state.get("tab8_fs_repr_feature_overlay_alpha", 0.70)),
        "feature_overlay_size": float(st.session_state.get("tab8_fs_repr_feature_overlay_size", 36.0)),
    }
    preview_max_dim = int(max(_to_float(st.session_state.get("tab8_fs_thumbnail_max_dim", 700), 700.0), 300.0))
    export_max_dim = int(max(_to_float(st.session_state.get("tab8_fs_repr_export_max_dim", 2200), 2200.0), float(preview_max_dim)))
    export_dpi = int(max(_to_float(st.session_state.get("tab8_fs_repr_export_dpi", 300), 300.0), 72.0))
    publication_columns = st.columns(len(selected_rows))

    for container, row in zip(publication_columns, selected_rows.itertuples(index=False)):
        with container:
            region_key = (str(row.batch), str(row.roi))
            region_df = region_lookup.get(region_key, pd.DataFrame()).copy()
            if region_df.empty:
                st.caption(f"No cells were found for {row.batch} | {row.roi}.")
                continue
            zone_df = _representative_zone_subset(region_df, config, zone_end, zone_start)
            image_path = image_lookup.get(normalize_image_stem(str(row.batch)).lower())
            if settings["show_background"]:
                image, scale_x, scale_y = _load_representative_preview(
                    image_path,
                    preview_max_dim,
                    selected_markers=selected_markers,
                    marker_color_mapping=marker_color_mapping,
                )
                if export_max_dim > preview_max_dim:
                    export_image, export_scale_x, export_scale_y = _load_representative_preview(
                        image_path,
                        export_max_dim,
                        selected_markers=selected_markers,
                        marker_color_mapping=marker_color_mapping,
                    )
                else:
                    export_image, export_scale_x, export_scale_y = image, scale_x, scale_y
            else:
                image, scale_x, scale_y = None, 1.0, 1.0
                export_image, export_scale_x, export_scale_y = None, 1.0, 1.0

            region_plot_df = _representative_plot_df(region_df, config, scale_x, scale_y)
            zone_plot_df = region_plot_df.loc[region_plot_df.index.intersection(zone_df.index)].copy()
            export_region_plot_df = _representative_plot_df(region_df, config, export_scale_x, export_scale_y)
            export_zone_plot_df = export_region_plot_df.loc[export_region_plot_df.index.intersection(zone_df.index)].copy()
            base_motif_data: dict[str, Any] | None = None
            preview_motif_data: dict[str, Any] | None = None
            export_motif_data: dict[str, Any] | None = None
            if feature_family in {"dyad", "triad"} and target_phenotypes:
                base_motif_data = compute_roi_motif_visualization_data(
                    zone_df=zone_df,
                    config=config,
                    target_phenotypes=target_phenotypes,
                    scale_x=1.0,
                    scale_y=1.0,
                )
                preview_motif_data = _scale_motif_visualization_data(base_motif_data, scale_x, scale_y)
                export_motif_data = _scale_motif_visualization_data(base_motif_data, export_scale_x, export_scale_y)
            preview_feature_overlay = _build_representative_feature_overlay(
                hit_row,
                zone_plot_df,
                target_phenotypes,
                motif_data=preview_motif_data,
            )
            export_feature_overlay = _build_representative_feature_overlay(
                hit_row,
                export_zone_plot_df,
                target_phenotypes,
                motif_data=export_motif_data,
            )

            fig = _representative_figure(
                image=image,
                region_plot_df=region_plot_df,
                display_plot_df=region_plot_df,
                zone_plot_df=zone_plot_df,
                color_mapping=color_mapping,
                title=f"{row.publication_role} | {row.batch} | {row.roi}",
                legend_title=f"Patient: {row.patient}\nPhenotype",
                settings=settings,
                show_context_background=True,
                target_phenotypes=target_phenotypes,
                interaction_distance=float(config.get("screening", {}).get("auto_interaction_distance_value", 30.0)) * scale_x,
                feature_overlay=preview_feature_overlay,
            )
            export_fig = fig
            if export_image is not image or export_scale_x != scale_x or export_scale_y != scale_y:
                export_fig = _representative_figure(
                    image=export_image,
                    region_plot_df=export_region_plot_df,
                    display_plot_df=export_region_plot_df,
                    zone_plot_df=export_zone_plot_df,
                    color_mapping=color_mapping,
                    title=f"{row.publication_role} | {row.batch} | {row.roi}",
                    legend_title=f"Patient: {row.patient}\nPhenotype",
                    settings=settings,
                    show_context_background=True,
                    target_phenotypes=target_phenotypes,
                    interaction_distance=float(config.get("screening", {}).get("auto_interaction_distance_value", 30.0)) * export_scale_x,
                    feature_overlay=export_feature_overlay,
                )
            figure_stub = f"{key_prefix}_{normalize_image_stem(str(row.publication_role))}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_publication_roi"
            _render_figure_with_download(fig, str(row.publication_role), figure_stub, download_dpi=export_dpi, download_fig=export_fig)
            plt.close(fig)
            if export_fig is not fig:
                plt.close(export_fig)

            metric_parts = [
                f"patient={row.patient}",
                f"ROI score={_to_float(row.roi_feature_value, float('nan')):.3f}" if np.isfinite(_to_float(row.roi_feature_value, float('nan'))) else "ROI score=-",
                f"patient score={_to_float(row.patient_feature_value, float('nan')):.3f}" if np.isfinite(_to_float(row.patient_feature_value, float('nan'))) else "patient score=-",
            ]
            for phenotype in target_phenotypes:
                stub = normalize_image_stem(str(phenotype)).replace("-", "_")
                zone_count = int(_to_float(getattr(row, f"{stub}_zone_count", 0), 0.0))
                zone_fraction = _to_float(getattr(row, f"{stub}_zone_fraction", float("nan")), float("nan"))
                if np.isfinite(zone_fraction):
                    metric_parts.append(f"{phenotype} boundary={zone_count} ({zone_fraction * 100:.1f}%)")
            st.caption(" | ".join(metric_parts))


def _render_publication_figures_section(
    analysis_df: pd.DataFrame,
    combined: pd.DataFrame,
    patient_features: pd.DataFrame,
    patient_outcomes: pd.DataFrame,
    patient_export_df: pd.DataFrame,
    stats_all: pd.DataFrame,
    candidate_summary: pd.DataFrame | None,
    config: dict[str, Any],
    hit_row: pd.Series,
    section_label: str,
    zone_start: float,
    zone_end: float,
    selected_candidate_width: float | None,
    key_prefix: str,
) -> None:
    st.caption(
        "This tab collects the main selected-hit figure, the ROI-level support plot, publication-matched representative ROIs, and a component-abundance panel in one place. "
        "For dyads and triads, the representative ROI pair is chosen to keep target-phenotype abundance as similar as possible while separating motif score."
    )

    main_fig = plot_feature_endpoint(hit_row, patient_features, patient_outcomes, config)
    _render_figure_with_download(main_fig, "Main endpoint figure", f"{key_prefix}_main_panel")
    plt.close(main_fig)

    range_hit_df = _build_publication_range_hit_df(candidate_summary=candidate_summary, stats_all=stats_all)
    if not range_hit_df.empty:
        selected_scan_width = _to_float(selected_candidate_width, _to_float(hit_row.get("zone_width"), zone_end))
        if not np.isfinite(selected_scan_width):
            selected_scan_width = zone_end
        range_scan_fig = plot_candidate_scan(
            range_hit_df,
            float(selected_scan_width),
            top_panel_order=("significant", "discovery", "nominal"),
            title="Hit counts across scanned ranges",
        )
        _render_figure_with_download(range_scan_fig, "Hit counts across scanned ranges", f"{key_prefix}_range_hits_scan")
        plt.close(range_scan_fig)
        if not isinstance(candidate_summary, pd.DataFrame) or candidate_summary.empty or len(range_hit_df) <= 1:
            st.caption("Only the current tested range is available in this context. To see all scanned bins here, run or keep the paired discovery scan in memory.")

    zone_coverage_df = _build_publication_zone_coverage_df(
        analysis_df=analysis_df,
        combined=combined,
        patient_outcomes=patient_outcomes,
        candidate_summary=candidate_summary,
        config=config,
        zone_start=zone_start,
        zone_end=zone_end,
    )
    if not zone_coverage_df.empty:
        selected_coverage_width = _to_float(selected_candidate_width, _to_float(hit_row.get("zone_width"), zone_end))
        if not np.isfinite(selected_coverage_width):
            selected_coverage_width = zone_end
        zone_coverage_fig = plot_zone_patient_coverage(
            zone_coverage_df,
            float(selected_coverage_width),
            title="Patients contributing per scanned zone",
        )
        _render_figure_with_download(zone_coverage_fig, "Patients contributing per scanned zone", f"{key_prefix}_zone_patient_coverage")
        plt.close(zone_coverage_fig)

        selected_zone_row = _selected_zone_coverage_row(zone_coverage_df, float(selected_coverage_width))
        required_reason_columns = {
            "total_patients",
            "n_patients_without_zone_cells",
            "n_patients_below_min_cells_per_zone",
            "n_patients_without_computable_features",
        }
        if selected_zone_row is None and len(zone_coverage_df) == 1:
            selected_zone_row = zone_coverage_df.iloc[0]
        if selected_zone_row is None or not required_reason_columns.issubset(set(selected_zone_row.index)):
            fallback_row = build_zone_patient_coverage_row(
                analysis_df=analysis_df,
                config=config,
                width=float(selected_coverage_width),
                patient_outcomes=patient_outcomes,
                combined_features=combined,
                zone_start=float(zone_start) if np.isfinite(float(zone_start)) else None,
                zone_end=float(zone_end) if np.isfinite(float(zone_end)) else None,
            )
            if fallback_row:
                selected_zone_row = pd.Series(fallback_row)
        if selected_zone_row is not None:
            hit_attrition_summary = build_feature_endpoint_attrition_summary(hit_row, patient_features, patient_outcomes, config)
            st.caption(_publication_zone_dropout_text(selected_zone_row, config, patient_export_df, hit_row, attrition_summary=hit_attrition_summary, selected_candidate_width=selected_candidate_width))

    if analysis_df.empty or combined.empty:
        st.info("Publication support figures require the current in-memory ROI-level data.")
        return

    combined_hit = combined[
        (combined["feature_name"].astype(str) == str(hit_row.get("feature_name") or ""))
        & (combined["zone_width"].astype(float) == float(hit_row.get("zone_width") or 0.0))
        & (combined["stratum_label"].astype(str) == str(hit_row.get("stratum_label") or ""))
    ].copy()
    if combined_hit.empty:
        st.info("No ROI-level rows are available for the selected hit.")
        return

    target_phenotypes = _representative_target_phenotypes(hit_row)
    roi_support_df, group_label = _build_publication_roi_support_df(
        analysis_df=analysis_df,
        combined_hit=combined_hit,
        config=config,
        hit_row=hit_row,
        patient_features=patient_features,
        patient_export_df=patient_export_df,
        zone_start=zone_start,
        zone_end=zone_end,
    )
    if roi_support_df.empty:
        st.info("The selected hit does not currently have enough ROI-level support data to build publication figures.")
        return

    score_fig = _plot_publication_roi_score_support(
        roi_support_df=roi_support_df,
        hit_row=hit_row,
        target_phenotypes=target_phenotypes,
        group_label=group_label,
    )
    _render_figure_with_download(score_fig, "ROI-level support plot", f"{key_prefix}_roi_support")
    plt.close(score_fig)

    patient_score_fig = _plot_publication_patient_roi_support(
        roi_support_df=roi_support_df,
        hit_row=hit_row,
        group_label=group_label,
    )
    _render_figure_with_download(patient_score_fig, "Patient-level ROI support plot", f"{key_prefix}_patient_roi_support")
    plt.close(patient_score_fig)

    summary_columns = [
        column
        for column in [
            "patient_group",
            "patient",
            "batch",
            "roi",
            "roi_feature_value",
            "patient_feature_value",
            "roi_total_cells",
            "zone_total_cells",
        ]
        if column in roi_support_df.columns
    ]
    _seen_ph_stubs: set[str] = set()
    for phenotype in target_phenotypes:
        stub = normalize_image_stem(str(phenotype)).replace("-", "_")
        if stub in _seen_ph_stubs:
            continue
        _seen_ph_stubs.add(stub)
        for suffix in ["roi_count", "roi_fraction", "zone_count", "zone_fraction"]:
            column = f"{stub}_{suffix}"
            if column in roi_support_df.columns:
                summary_columns.append(column)
    summary_df = roi_support_df.loc[:, summary_columns].copy() if summary_columns else roi_support_df.copy()
    st.dataframe(summary_df, width="stretch", height=min(420, 80 + 28 * len(summary_df)))
    _render_dataframe_download(
        summary_df,
        "Download publication ROI support CSV",
        f"{str(section_label).lower()}_{normalize_image_stem(str(hit_row.get('feature_name') or 'feature'))}_publication_roi_support.csv",
        f"{key_prefix}_roi_support_csv_download",
    )

    _render_publication_representative_rois(
        analysis_df=analysis_df,
        config=config,
        hit_row=hit_row,
        roi_support_df=roi_support_df,
        section_label=section_label,
        zone_start=zone_start,
        zone_end=zone_end,
        key_prefix=f"{key_prefix}_representatives",
    )

    abundance_fig = _plot_publication_component_abundance(
        roi_support_df=roi_support_df,
        hit_row=hit_row,
        target_phenotypes=target_phenotypes,
        group_label=group_label,
    )
    _render_figure_with_download(abundance_fig, "Underlying component abundance", f"{key_prefix}_component_abundance")
    plt.close(abundance_fig)


def _representative_zone_subset(
    region_df: pd.DataFrame,
    config: dict[str, Any],
    zone_end: float,
    zone_start: float = 0.0,
) -> pd.DataFrame:
    distance_col = str(config["columns"]["distance"])
    if distance_col not in region_df.columns:
        return region_df.iloc[0:0].copy()
    if np.isfinite(float(zone_start)) and not np.isclose(float(zone_start), 0.0):
        mask = zone_distance_interval_mask(region_df[distance_col], zone_start, zone_end)
    else:
        mask = zone_distance_mask(region_df[distance_col], zone_end)
    return region_df.loc[mask].copy()


def _load_representative_preview(
    image_path: Path | None,
    max_dim: int,
    selected_markers: list[str] | None = None,
    marker_color_mapping: dict[str, str] | None = None,
) -> tuple[np.ndarray | None, float, float]:
    if image_path is None or not image_path.exists():
        return None, 1.0, 1.0
    marker_color_mapping = marker_color_mapping or {}
    ordered_markers: list[tuple[str, int]] = []
    indices_to_load = [0, 1, 2]
    try:
        channel_names, _channel_colors, _channel_metadata, _mpp = get_channel_info(str(image_path))
        ordered_markers = _resolve_marker_indices([str(name) for name in channel_names], [str(marker) for marker in (selected_markers or []) if str(marker)])
        if ordered_markers:
            indices_to_load = [idx for _, idx in ordered_markers]
    except Exception:
        ordered_markers = []
    try:
        preview_data, full_shape, is_rgb = load_preview_channels_compatible(str(image_path), indices_to_load, max_dim=max_dim)
    except Exception:
        return None, 1.0, 1.0

    if is_rgb and "rgb" in preview_data:
        image = np.asarray(preview_data["rgb"])
    else:
        image = _compose_marker_preview(preview_data, ordered_markers, marker_color_mapping)
        if image is None:
            channels = []
            for key in sorted(preview_data.keys())[:3]:
                arr = np.asarray(preview_data[key], dtype=float)
                arr = (arr - np.nanmin(arr)) / max(np.nanmax(arr) - np.nanmin(arr), 1e-6)
                channels.append(arr)
            if not channels:
                image = np.zeros((max_dim, max_dim, 3), dtype=float)
            elif len(channels) == 1:
                image = np.dstack([channels[0], channels[0], channels[0]])
            else:
                while len(channels) < 3:
                    channels.append(np.zeros_like(channels[0]))
                image = np.dstack(channels[:3])
        if image is None:
            image = np.zeros((max_dim, max_dim, 3), dtype=float)

    if full_shape and len(full_shape) >= 2:
        scale_x = image.shape[1] / float(full_shape[-1])
        scale_y = image.shape[0] / float(full_shape[-2])
    else:
        scale_x = 1.0
        scale_y = 1.0
    return image, scale_x, scale_y


def _representative_plot_df(
    region_df: pd.DataFrame,
    config: dict[str, Any],
    scale_x: float,
    scale_y: float,
) -> pd.DataFrame:
    x_col = str(config["columns"]["x"])
    y_col = str(config["columns"]["y"])
    phenotype_col = str(config["columns"]["phenotype"])
    plot_df = region_df.copy()
    plot_df["__plot_x__"] = pd.to_numeric(plot_df[x_col], errors="coerce") * scale_x
    plot_df["__plot_y__"] = pd.to_numeric(plot_df[y_col], errors="coerce") * scale_y
    plot_df["__phenotype__"] = plot_df[phenotype_col].astype(str)
    mask = np.isfinite(plot_df["__plot_x__"].to_numpy(dtype=float)) & np.isfinite(plot_df["__plot_y__"].to_numpy(dtype=float))
    return plot_df.loc[mask].copy()


def _representative_default_colors(phenotypes: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    palette = plt.cm.get_cmap("tab20", max(len(phenotypes), 1))
    for idx, phenotype in enumerate(phenotypes):
        colors[phenotype] = _CELLVIZ_DEFAULT_COLOR_MAP.get(phenotype, to_hex(palette(idx % palette.N)))
    return colors


def _draw_representative_zone_overlay(
    ax: Axes,
    zone_plot_df: pd.DataFrame,
    color: str,
    linewidth: float,
    use_hatch: bool,
) -> None:
    if zone_plot_df.empty:
        return
    points = zone_plot_df[["__plot_x__", "__plot_y__"]].to_numpy(dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return

    polygon_points: np.ndarray | None = None
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            polygon_points = points[hull.vertices]
        except Exception:
            polygon_points = None

    if polygon_points is None:
        min_x = float(np.min(points[:, 0]))
        max_x = float(np.max(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))
        padding = max(12.0, 0.05 * max(max_x - min_x, max_y - min_y, 1.0))
        polygon_points = np.array(
            [
                [min_x - padding, min_y - padding],
                [max_x + padding, min_y - padding],
                [max_x + padding, max_y + padding],
                [min_x - padding, max_y + padding],
            ]
        )

    patch = mpatches.Polygon(
        polygon_points,
        closed=True,
        facecolor=(0, 0, 0, 0),
        edgecolor=color,
        linewidth=linewidth,
        linestyle="--",
    )
    if use_hatch:
        patch.set_hatch("////")
    ax.add_patch(patch)


def _render_representative_scatter(
    ax: Axes,
    plot_df: pd.DataFrame,
    color_mapping: dict[str, str],
    point_alpha: float,
    point_size: float,
) -> set[str]:
    plotted: set[str] = set()
    if plot_df.empty:
        return plotted
    for phenotype, sub_df in plot_df.groupby("__phenotype__"):
        ax.scatter(
            sub_df["__plot_x__"],
            sub_df["__plot_y__"],
            s=point_size,
            alpha=point_alpha,
            c=color_mapping.get(str(phenotype), "#8a8a8a"),
            edgecolors="white",
            linewidths=0.25,
        )
        plotted.add(str(phenotype))
    return plotted


def _representative_view_specs(
    region_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    config: dict[str, Any],
    target_phenotypes: list[str],
    visible_phenotypes: list[str],
) -> list[dict[str, Any]]:
    phenotype_col = str(config["columns"]["phenotype"])
    visible_set = {str(item) for item in visible_phenotypes}
    target_set = {str(item) for item in target_phenotypes if str(item) in visible_set}
    region_visible = region_df[region_df[phenotype_col].astype(str).isin(visible_set)].copy()
    zone_visible = zone_df[zone_df[phenotype_col].astype(str).isin(visible_set)].copy()
    target_visible = zone_visible[zone_visible[phenotype_col].astype(str).isin(target_set)].copy()
    other_visible = zone_visible[~zone_visible[phenotype_col].astype(str).isin(target_set)].copy()

    specs: list[dict[str, Any]] = []
    if not region_visible.empty:
        specs.append(
            {
                "title": "ROI context",
                "caption": "All selected phenotypes in the ROI. The discovered zone is shown as an overlay so the surrounding architecture stays visible.",
                "display_df": region_visible,
                "show_context_background": True,
            }
        )
    if not target_visible.empty:
        specs.append(
            {
                "title": "Targeted phenotype(s) in discovered zone",
                "caption": "Only the phenotype(s) that define the hit are shown inside the discovered zone.",
                "display_df": target_visible,
                "show_context_background": False,
            }
        )
    if not zone_visible.empty:
        specs.append(
            {
                "title": "All phenotypes in discovered zone",
                "caption": "Every currently selected phenotype inside the discovered zone.",
                "display_df": zone_visible,
                "show_context_background": False,
            }
        )
    if not other_visible.empty and bool(target_set):
        specs.append(
            {
                "title": "Other phenotypes in discovered zone",
                "caption": "The non-target phenotypes inside the discovered zone, shown separately for comparison.",
                "display_df": other_visible,
                "show_context_background": False,
            }
        )
    return specs


def _representative_figure(
    image: np.ndarray | None,
    region_plot_df: pd.DataFrame,
    display_plot_df: pd.DataFrame,
    zone_plot_df: pd.DataFrame,
    color_mapping: dict[str, str],
    title: str,
    legend_title: str,
    settings: dict[str, Any],
    show_context_background: bool,
    target_phenotypes: list[str] | None = None,
    interaction_distance: float = 30.0,
    feature_overlay: dict[str, Any] | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    if image is not None and bool(settings.get("show_background", True)):
        ax.imshow(image)

    if show_context_background and bool(settings.get("dim_outside_zone", True)):
        outside_plot_df = region_plot_df.drop(index=zone_plot_df.index, errors="ignore")
        if not outside_plot_df.empty:
            ax.scatter(
                outside_plot_df["__plot_x__"],
                outside_plot_df["__plot_y__"],
                s=max(float(settings.get("point_size", 18.0)) * 0.7, 4.0),
                alpha=0.18,
                c="#bdbdbd",
                edgecolors="none",
            )

    plotted_labels: set[str]
    render_mode = str(settings.get("render_mode", "Scatter points"))
    if render_mode == "Voronoi cells" and len(display_plot_df) >= 4:
        plotted_labels = _cellviz_render_voronoi(
            ax,
            display_plot_df,
            color_col="__phenotype__",
            color_mapping=color_mapping,
            x_col="__plot_x__",
            y_col="__plot_y__",
            max_voronoi_cell_area=float(settings.get("voronoi_max_area", 400.0)),
            draw_cell_centers=bool(settings.get("draw_centers", False)),
        )
    else:
        plotted_labels = _render_representative_scatter(
            ax,
            display_plot_df,
            color_mapping,
            point_alpha=float(settings.get("point_alpha", 0.80)),
            point_size=float(settings.get("point_size", 18.0)),
        )
        if not display_plot_df.empty:
            min_x = float(display_plot_df["__plot_x__"].min())
            max_x = float(display_plot_df["__plot_x__"].max())
            min_y = float(display_plot_df["__plot_y__"].min())
            max_y = float(display_plot_df["__plot_y__"].max())
            padding = max(20.0, 0.07 * max(max_x - min_x, max_y - min_y, 1.0))
            if image is None or not bool(settings.get("show_background", True)):
                ax.set_xlim(min_x - padding, max_x + padding)
                ax.set_ylim(max_y + padding, min_y - padding)
                ax.set_aspect("equal")

    if bool(settings.get("show_zone_overlay", True)):
        _draw_representative_zone_overlay(
            ax,
            zone_plot_df,
            color=str(settings.get("zone_color", "#1f4e79")),
            linewidth=float(settings.get("zone_linewidth", 1.5)),
            use_hatch=bool(settings.get("zone_hatched", True)),
        )

    if bool(settings.get("draw_edges", False)) and target_phenotypes and len(target_phenotypes) >= 2:
        try:
            from scipy.spatial import cKDTree
            p1 = str(target_phenotypes[0])
            p2 = str(target_phenotypes[1])
            df1 = display_plot_df[display_plot_df["__phenotype__"].astype(str) == p1]
            df2 = display_plot_df[display_plot_df["__phenotype__"].astype(str) == p2]
            if not df1.empty and not df2.empty:
                pts1 = df1[["__plot_x__", "__plot_y__"]].to_numpy(dtype=float)
                pts2 = df2[["__plot_x__", "__plot_y__"]].to_numpy(dtype=float)
                tree2 = cKDTree(pts2)
                # Query all points within the interaction distance
                res = tree2.query_ball_point(pts1, r=interaction_distance)
                edge_x = []
                edge_y = []
                for i, neighbors in enumerate(res):
                    for j in neighbors:
                        edge_x.extend([pts1[i, 0], pts2[j, 0], None])
                        edge_y.extend([pts1[i, 1], pts2[j, 1], None])
                if edge_x:
                    ax.plot(edge_x, edge_y, color="#7f8c8d", lw=1.2, alpha=0.5, zorder=1)
        except Exception:
            pass

    _draw_representative_feature_overlay(ax, region_plot_df, feature_overlay, settings)

    if bool(settings.get("show_legend", True)) and plotted_labels:
        _cellviz_add_unified_legend(fig, ax, plotted_labels, color_mapping, title=legend_title)

    ax.set_axis_off()
    if bool(settings.get("show_title", True)):
        ax.set_title(title, loc="left", fontsize=11)
    annotation_text = str(settings.get("annotation_text", "")).strip()
    if annotation_text:
        fig.text(0.985, 0.02, annotation_text, ha="right", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.03, right=0.76 if bool(settings.get("show_legend", True)) else 0.98, bottom=0.04, top=0.92)
    return fig


def _render_representative_rois(
    results: dict[str, Any],
    config: dict[str, Any],
    section_label: str = "Discovery",
    state_prefix: str = "tab8_fs_repr",
) -> None:
    def _repr_key(name: str) -> str:
        return f"{state_prefix}_{name}"

    def _repr_state(name: str, default: Any) -> Any:
        key = _repr_key(name)
        if key in st.session_state:
            return st.session_state.get(key)
        if state_prefix != "tab8_fs_repr":
            legacy_key = f"tab8_fs_repr_{name}"
            if legacy_key in st.session_state:
                return st.session_state.get(legacy_key)
        return default

    stats_all = results.get("stats_all", pd.DataFrame())
    if stats_all.empty:
        st.info("No screening hits are available yet.")
        return

    analysis_df = results.get("analysis_df", pd.DataFrame())
    if analysis_df.empty:
        st.info("Representative ROI previews require the current in-memory analysis dataframe.")
        return

    patient_features_by_width = results.get("patient_features_by_width", {})
    combined_by_width = results.get("combined_by_width", {})
    is_discovery_run = isinstance(patient_features_by_width, dict) and isinstance(combined_by_width, dict) and bool(patient_features_by_width) and bool(combined_by_width)

    top_hits = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").head(50).copy()
    top_hits["_label"] = top_hits.apply(lambda row: f"{row['endpoint_label']} | {row['feature_label']} | {row['stratum_label']} | range={str(row.get('zone_label') or format_zone_label(row['zone_width']))}", axis=1)
    hit_key = f"{state_prefix}_hit"
    _stateful_selectbox("Hit for representative ROIs", options=top_hits["_label"].tolist(), key=hit_key, default=top_hits["_label"].iloc[0] if not top_hits.empty else None)
    selected_label = st.session_state.get(hit_key, "")
    if not selected_label:
        return
    hit_row = top_hits.loc[top_hits["_label"] == selected_label].iloc[0]

    width = float(hit_row["zone_width"])
    if is_discovery_run:
        patient_features = patient_features_by_width.get(width, pd.DataFrame())
        combined = combined_by_width.get(width, pd.DataFrame())
    else:
        patient_features = results.get("patient_features", pd.DataFrame())
        combined = results.get("combined", pd.DataFrame())
    if patient_features.empty or combined.empty:
        st.info("Representative ROI previews are only available for hits generated in the current session.")
        return

    validation_zone_start = _to_float(results.get("validation_zone_start"), 0.0)
    validation_zone_end = _to_float(results.get("validation_zone_end"), width)
    explicit_interval_mode = (not is_discovery_run) and np.isfinite(validation_zone_end) and not np.isclose(validation_zone_start, 0.0)
    base_zone_start = float(validation_zone_start if explicit_interval_mode else 0.0)
    base_zone_end = float(validation_zone_end if np.isfinite(validation_zone_end) else width)

    patient_hit = patient_features[
        (patient_features["feature_name"].astype(str) == str(hit_row["feature_name"]))
        & (patient_features["zone_width"].astype(float) == width)
        & (patient_features["stratum_label"].astype(str) == str(hit_row["stratum_label"]))
    ].copy()
    if patient_hit.empty:
        st.info("No patient-level values were found for the selected hit.")
        return

    median_value = float(pd.to_numeric(patient_hit["value"], errors="coerce").median())
    patient_hit["screen_group"] = np.where(pd.to_numeric(patient_hit["value"], errors="coerce") >= median_value, "High", "Low")
    combined_hit = combined[
        (combined["feature_name"].astype(str) == str(hit_row["feature_name"]))
        & (combined["zone_width"].astype(float) == width)
        & (combined["stratum_label"].astype(str) == str(hit_row["stratum_label"]))
    ].copy()
    combined_hit = combined_hit.merge(patient_hit[["patient", "screen_group"]], on="patient", how="left")
    if combined_hit.empty:
        st.info("No region-level rows were found for the selected hit.")
        return

    image_lookup = _resolve_image_lookup(config["representatives"].get("raw_image_dir", ""))
    if not image_lookup:
        st.warning("No representative images could be resolved because the raw image directory is empty or does not contain supported image files. The ROI figures can still render on a blank spatial canvas if you keep working from the unified CSV only.")

    batch_col = str(config["columns"]["batch"])
    roi_col = str(config["columns"]["roi"])
    phenotype_col = str(config["columns"]["phenotype"])
    x_col = str(config["columns"].get("x") or "")
    y_col = str(config["columns"].get("y") or "")
    max_examples = int(config["representatives"].get("max_examples_per_group", 3))
    high_rows = combined_hit[combined_hit["screen_group"] == "High"].sort_values("value", ascending=False).head(max_examples)
    low_rows = combined_hit[combined_hit["screen_group"] == "Low"].sort_values("value", ascending=True).head(max_examples)
    if high_rows.empty and low_rows.empty:
        st.info("No representative regions were found for the selected hit.")
        return

    representative_regions: dict[tuple[str, str], pd.DataFrame] = {}
    active_phenotypes: list[str] = []
    for row in pd.concat([high_rows, low_rows], axis=0).drop_duplicates(subset=["batch", "roi"]).itertuples(index=False):
        region_key = (str(row.batch), str(row.roi))
        region_df = analysis_df[
            (analysis_df[batch_col].astype(str) == region_key[0])
            & (analysis_df[roi_col].astype(str) == region_key[1])
        ].copy()
        representative_regions[region_key] = region_df
        active_phenotypes.extend(_safe_unique(region_df[phenotype_col]) if phenotype_col in region_df.columns else [])
    active_phenotypes = sorted({str(item) for item in active_phenotypes})
    target_phenotypes = _representative_target_phenotypes(hit_row)
    if not active_phenotypes:
        st.info("Representative regions were found, but no phenotype labels are available to render them.")
        return

    feature_family = str(hit_row.get("feature_family", "") or "")
    overlay_supported = feature_family in {"density", "proportion", "ratio", "dyad", "triad", "nn_distance"}
    screen_zone_label = format_zone_interval_label(base_zone_start, base_zone_end)

    st.caption(
        f"Representative ROIs are the highest and lowest region-level examples for the selected {section_label.lower()} hit. "
        f"The discovered zone is reconstructed directly from the unified CSV using {screen_zone_label}. "
        "When the raw image is available, the figures are overlaid on the image; otherwise they render on the spatial coordinates alone."
    )

    visible_phenotypes_key = _repr_key("visible_phenotypes")
    selected_markers_key = _repr_key("selected_markers")
    render_mode_key = _repr_key("render_mode")
    show_background_key = _repr_key("show_background")
    show_legend_key = _repr_key("show_legend")
    export_dpi_key = _repr_key("export_dpi")
    show_zone_overlay_key = _repr_key("show_zone_overlay")
    zone_hatched_key = _repr_key("zone_hatched")
    dim_outside_zone_key = _repr_key("dim_outside_zone")
    show_title_key = _repr_key("show_title")
    export_max_dim_key = _repr_key("export_max_dim")
    point_alpha_key = _repr_key("point_alpha")
    marker_size_key = _repr_key("marker_size")
    zone_linewidth_key = _repr_key("zone_linewidth")
    voronoi_max_area_key = _repr_key("voronoi_max_area")
    draw_centers_key = _repr_key("draw_centers")
    draw_edges_key = _repr_key("draw_edges")
    annotation_text_key = _repr_key("annotation_text")
    zone_color_key = _repr_key("zone_color")
    overlay_mode_key = _repr_key("feature_overlay_mode")
    overlay_default_key = f"_{state_prefix}_last_overlay_default"
    feature_overlay_color_key = _repr_key("feature_overlay_color")
    feature_overlay_alpha_key = _repr_key("feature_overlay_alpha")
    feature_overlay_size_key = _repr_key("feature_overlay_size")

    _ensure_multiselect_state(visible_phenotypes_key, active_phenotypes, default=active_phenotypes)

    zone_end_key = f"{state_prefix}_zone_end_override" if explicit_interval_mode else f"{state_prefix}_zone_width_override"
    zone_start_key = f"{state_prefix}_zone_start_override"
    last_hit_key = f"_{state_prefix}_last_hit_label"
    overlay_default_mode = _default_representative_feature_overlay_mode(feature_family) if overlay_supported else "None"
    if st.session_state.get(last_hit_key, "") != selected_label:
        current_overlay_mode = str(st.session_state.get(overlay_mode_key, "") or "")
        previous_overlay_default = str(st.session_state.get(overlay_default_key, "") or "")
        if overlay_mode_key not in st.session_state or not current_overlay_mode or current_overlay_mode == previous_overlay_default:
            st.session_state[overlay_mode_key] = overlay_default_mode
        st.session_state[overlay_default_key] = overlay_default_mode
        st.session_state[zone_start_key] = float(base_zone_start)
        st.session_state[zone_end_key] = float(base_zone_end)
        st.session_state[last_hit_key] = selected_label

    representative_image_paths = []
    image_lookup = _resolve_image_lookup(config["representatives"].get("raw_image_dir", ""))
    for row in pd.concat([high_rows, low_rows], axis=0).drop_duplicates(subset=["batch", "roi"]).itertuples(index=False):
        image_path = image_lookup.get(normalize_image_stem(str(row.batch)).lower())
        if isinstance(image_path, Path) and image_path.exists():
            representative_image_paths.append(image_path)
    marker_options, marker_default_colors = _resolve_representative_marker_options(representative_image_paths)
    default_marker_selection = [str(value) for value in _repr_state("selected_markers", []) if str(value) in marker_options]
    if not default_marker_selection:
        default_marker_selection = marker_options[: min(3, len(marker_options))]
    _ensure_multiselect_state(selected_markers_key, marker_options, default=default_marker_selection)

    with st.expander("Representative figure controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            _stateful_selectbox(
                "Render mode",
                options=["Scatter points", "Voronoi cells"],
                key=render_mode_key,
                default=str(_repr_state("render_mode", "Scatter points") or "Scatter points"),
                help="Use scatter points for a lightweight view, or Voronoi cells for a publication-style cell tessellation similar to the visualization tab.",
            )
            if explicit_interval_mode:
                _stateful_number_input(
                    "Representative zone start",
                    key=zone_start_key,
                    default=float(base_zone_start),
                    step=max(abs(base_zone_end - base_zone_start) / 20.0, 1.0),
                    help="Adjust the validation interval used to redraw the representative ROIs without rerunning validation.",
                )
                _stateful_number_input(
                    "Representative zone end",
                    key=zone_end_key,
                    default=float(base_zone_end),
                    step=max(abs(base_zone_end - base_zone_start) / 20.0, 1.0),
                    help="Adjust the validation interval end used for representative redraws.",
                )
            else:
                _stateful_number_input(
                    "Representative zone boundary",
                    key=zone_end_key,
                    default=float(base_zone_end),
                    step=max(abs(base_zone_end) / 20.0, 1.0),
                    help="This lets you regenerate the representative figures with a slightly different signed zone boundary while keeping the selected hit fixed.",
                )
            _stateful_checkbox("Show raw image background", key=show_background_key, default=bool(_repr_state("show_background", True)), help="Turn this off to render only the cell coordinates from the unified CSV.")
            _stateful_checkbox("Show legend outside figure", key=show_legend_key, default=bool(_repr_state("show_legend", True)), help="Places the batch and phenotype legend on the right side so it does not cover the ROI.")
            _stateful_number_input(
                "Export DPI",
                key=export_dpi_key,
                default=int(_to_float(_repr_state("export_dpi", 300), 300.0)),
                min_value=72,
                step=50,
                help="Representative figure downloads use this DPI. The on-screen preview stays lightweight.",
            )
        with c2:
            _stateful_checkbox("Show discovered zone overlay", key=show_zone_overlay_key, default=bool(_repr_state("show_zone_overlay", True)), help="The overlay is drawn as a convex hull around the in-zone cells because the unified CSV contains cell coordinates and distances, not an explicit tumor-boundary polygon.")
            _stateful_checkbox("Use hatched zone outline", key=zone_hatched_key, default=bool(_repr_state("zone_hatched", True)), help="Use a striped outline for the discovered zone. Turn this off for a simple dashed line.")
            _stateful_checkbox("Dim cells outside zone in ROI context", key=dim_outside_zone_key, default=bool(_repr_state("dim_outside_zone", True)), help="Keeps the architecture visible while making the discovered zone easier to read.")
            _stateful_checkbox("Show figure title", key=show_title_key, default=bool(_repr_state("show_title", True)), help="Turn this off if you want a cleaner export and plan to add titles later in the manuscript layout.")
            _stateful_number_input(
                "Export max image dimension",
                key=export_max_dim_key,
                default=int(_to_float(_repr_state("export_max_dim", 2200), 2200.0)),
                min_value=500,
                step=100,
                help="Downloads reload the representative background with a larger preview size up to this dimension while the interactive thumbnail stays fast.",
            )
            if overlay_supported:
                _stateful_selectbox(
                    "Discovered feature overlay",
                    options=["None", "Feature dots", "Feature fuzzy cloud", "Dots + fuzzy cloud"],
                    key=overlay_mode_key,
                    default=overlay_default_mode,
                    help="Overlay the discovered feature directly on the representative ROI. Dyads and triads use motif centroids; density-like features use the matching target cells.",
                )
        with c3:
            _stateful_number_input("Cell opacity", key=point_alpha_key, default=float(_to_float(_repr_state("point_alpha", 0.80), 0.80)), min_value=0.05, max_value=1.0, step=0.05, format="%.2f", help="Opacity for scatter points when scatter mode is used.")
            _stateful_number_input("Marker size", key=marker_size_key, default=float(_to_float(_repr_state("marker_size", 18.0), 18.0)), min_value=1.0, step=1.0, help="Point size for scatter mode.")
            _stateful_number_input("Zone outline width", key=zone_linewidth_key, default=float(_to_float(_repr_state("zone_linewidth", 1.5), 1.5)), min_value=0.5, step=0.5, format="%.1f", help="Line width for the discovered-zone outline.")
            _stateful_number_input("Voronoi max cell area", key=voronoi_max_area_key, default=float(_to_float(_repr_state("voronoi_max_area", 400.0), 400.0)), min_value=10.0, step=10.0, disabled=st.session_state.get(render_mode_key, "Scatter points") != "Voronoi cells", help="Larger values allow broader Voronoi tiles before clipping.")
            _stateful_checkbox("Draw Voronoi cell centers", key=draw_centers_key, default=bool(_repr_state("draw_centers", False)), help="Optionally add tiny cell-center dots on top of the Voronoi rendering.")
            _stateful_checkbox("Draw interaction edges", key=draw_edges_key, default=bool(_repr_state("draw_edges", False)), help="Draw lines connecting cells of the target phenotypes if they are within the interaction distance threshold. Useful for visualizing dyad/triad and nearest-neighbor features.")
            if overlay_supported:
                _stateful_number_input("Feature overlay opacity", key=feature_overlay_alpha_key, default=float(_to_float(_repr_state("feature_overlay_alpha", 0.70), 0.70)), min_value=0.05, max_value=1.0, step=0.05, format="%.2f")
                _stateful_number_input("Feature overlay dot size", key=feature_overlay_size_key, default=float(_to_float(_repr_state("feature_overlay_size", 36.0), 36.0)), min_value=4.0, step=2.0)

        _stateful_multiselect(
            "Visible phenotypes",
            options=active_phenotypes,
            key=visible_phenotypes_key,
            default=_repr_state("visible_phenotypes", active_phenotypes),
            help="Use this to add or remove phenotypes from the representative figures before exporting them.",
        )
        _stateful_text_input(
            "Figure annotation",
            key=annotation_text_key,
            default=str(_repr_state("annotation_text", "") or ""),
            help="Optional note shown on the figure, useful for manuscript labels or review comments.",
        )
        if st.session_state.get(show_zone_overlay_key, True):
            if zone_color_key not in st.session_state:
                st.session_state[zone_color_key] = str(_repr_state("zone_color", "#1f4e79") or "#1f4e79")
            st.color_picker("Zone outline color", key=zone_color_key)
        if overlay_supported:
            if feature_overlay_color_key not in st.session_state:
                st.session_state[feature_overlay_color_key] = str(_repr_state("feature_overlay_color", "#ff6b00") or "#ff6b00")
            st.color_picker("Feature overlay color", key=feature_overlay_color_key)

        if marker_options:
            _stateful_multiselect(
                "Raw-image markers",
                options=marker_options,
                key=selected_markers_key,
                default=default_marker_selection,
                help="Choose the marker channels to composite into the representative background image. The screen preview stays lightweight; downloads use the higher export setting above.",
            )
            selected_markers = [str(value) for value in st.session_state.get(selected_markers_key, []) if str(value) in marker_options]
            if selected_markers:
                st.markdown("Marker colors")
                marker_columns = st.columns(min(4, max(len(selected_markers), 1)))
                for idx, marker_name in enumerate(selected_markers):
                    marker_color_key = _repr_key(f"marker_color__{normalize_image_stem(str(marker_name))}")
                    if marker_color_key not in st.session_state:
                        st.session_state[marker_color_key] = str(_repr_state(f"marker_color__{normalize_image_stem(str(marker_name))}", marker_default_colors.get(marker_name, "#ffffff")) or marker_default_colors.get(marker_name, "#ffffff"))
                    with marker_columns[idx % len(marker_columns)]:
                        st.color_picker(f"{marker_name}", key=marker_color_key)
        elif image_lookup and st.session_state.get(show_background_key, True):
            st.caption("Representative raw images were found, but no channel metadata could be resolved for marker selection. The background will fall back to the default channel composite.")

        visible_phenotypes = st.session_state.get(visible_phenotypes_key, active_phenotypes)
        color_defaults = _representative_default_colors(active_phenotypes)
        st.markdown("Color overrides")
        color_columns = st.columns(min(4, max(len(visible_phenotypes), 1)))
        for idx, phenotype in enumerate(visible_phenotypes):
            color_key = _repr_key(f"color__{normalize_image_stem(str(phenotype))}")
            if color_key not in st.session_state:
                st.session_state[color_key] = str(_repr_state(f"color__{normalize_image_stem(str(phenotype))}", color_defaults.get(str(phenotype), "#808080")) or color_defaults.get(str(phenotype), "#808080"))
            with color_columns[idx % len(color_columns)]:
                st.color_picker(f"{phenotype}", key=color_key)

    # ── Cell Segmentation & Zone Boundary ────────────────────────────────────
    seg_mode_key          = _repr_key("seg_mode")
    geojson_root_key      = _repr_key("geojson_root")
    geojson_suffixes_key  = _repr_key("geojson_suffixes")
    geojson_flip_y_key    = _repr_key("geojson_flip_y")
    geojson_match_dist_key= _repr_key("geojson_match_dist")
    voronoi_pct_key       = _repr_key("voronoi_percentile")
    voronoi_fallback_key  = _repr_key("voronoi_fallback")
    circle_radius_key     = _repr_key("circle_radius")
    show_contour_key      = _repr_key("show_distance_contour")
    distance_col_key      = _repr_key("distance_col")
    contour_color_key     = _repr_key("contour_color")
    contour_lw_key        = _repr_key("contour_linewidth")
    seg_fill_op_key       = _repr_key("seg_fill_opacity")
    seg_overlay_op_key    = _repr_key("seg_overlay_opacity")
    export_ind_markers_key= _repr_key("export_individual_markers")

    # Auto-detect distance column from dataframe
    _dist_col_default = "neighborhood_distance_tumor"
    for _col_candidate in analysis_df.columns:
        if "distance" in str(_col_candidate).lower() and "tumor" in str(_col_candidate).lower():
            _dist_col_default = str(_col_candidate)
            break

    with st.expander("Cell Segmentation & Zone Boundary", expanded=False):
        st.caption(
            "Load cell segmentation polygons (from a GeoJSON file, Voronoi tessellation, or circles) "
            "to produce publication-quality phenotype figures.  "
            "The zone boundary contour draws iso-lines at the zone distances using the continuous "
            "distance column (``neighborhood_distance_tumor``) via scattered-point interpolation — "
            "replacing the convex-hull approximation."
        )
        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            _stateful_selectbox(
                "Cell shape mode",
                options=["None", "GeoJSON", "Voronoi", "Circle"],
                key=seg_mode_key,
                default=str(_repr_state("seg_mode", "None")),
                help=(
                    "**None** — use only scatter/Voronoi rendering from the figure controls above.  \n"
                    "**GeoJSON** — load actual cell boundary polygons from a `.geojson` file.  \n"
                    "**Voronoi** — tessellate cell centroids from the CSV (no external file needed).  \n"
                    "**Circle** — draw a fixed-radius circle around each cell centroid."
                ),
            )
            _seg_mode_now = str(st.session_state.get(seg_mode_key, "None"))
            if _seg_mode_now == "GeoJSON":
                _stateful_text_input(
                    "GeoJSON root directory",
                    key=geojson_root_key,
                    default=str(_repr_state("geojson_root", "")),
                    help="Root folder to search recursively for `*.geojson` files whose name starts with the batch ID.",
                )
                _stateful_text_area(
                    "Preferred GeoJSON suffixes (one per line)",
                    key=geojson_suffixes_key,
                    default=str(_repr_state("geojson_suffixes", ".ome_cell_boundaries.geojson\n_phenotype_ml_consensus.geojson\n_phenotype_leiden.geojson")),
                    height=80,
                    help="Files matching earlier suffixes are preferred when multiple GeoJSON files exist for the same batch.",
                )
                _stateful_number_input(
                    "Max match distance (px)",
                    key=geojson_match_dist_key,
                    default=float(_to_float(_repr_state("geojson_match_dist", 50.0), 50.0)),
                    min_value=1.0,
                    step=5.0,
                    help="GeoJSON polygon centroid must be within this many pixels of a CSV cell centroid to be matched.",
                )
                _stateful_checkbox(
                    "Flip Y coordinate in GeoJSON",
                    key=geojson_flip_y_key,
                    default=bool(_repr_state("geojson_flip_y", False)),
                    help="Enable if the GeoJSON Y coordinates are flipped relative to the image.",
                )
            elif _seg_mode_now == "Voronoi":
                _stateful_number_input(
                    "Voronoi cap percentile",
                    key=voronoi_pct_key,
                    default=float(_to_float(_repr_state("voronoi_percentile", 50.0), 50.0)),
                    min_value=10.0, max_value=99.0, step=5.0, format="%.0f",
                    help="Voronoi cells are capped at this percentile of nearest-neighbour distances.",
                )
                _stateful_number_input(
                    "Voronoi fallback radius (px)",
                    key=voronoi_fallback_key,
                    default=float(_to_float(_repr_state("voronoi_fallback", 15.0), 15.0)),
                    min_value=1.0, step=1.0,
                    help="Minimum cap radius applied when the NN distance percentile is too small.",
                )
            elif _seg_mode_now == "Circle":
                _stateful_number_input(
                    "Circle radius (px, display space)",
                    key=circle_radius_key,
                    default=float(_to_float(_repr_state("circle_radius", 5.0), 5.0)),
                    min_value=1.0, step=0.5, format="%.1f",
                )

        with _sc2:
            _stateful_number_input(
                "Polygon fill opacity",
                key=seg_fill_op_key,
                default=float(_to_float(_repr_state("seg_fill_opacity", 1.0), 1.0)),
                min_value=0.05, max_value=1.0, step=0.05, format="%.2f",
                help="Opacity for the 'Phenotype polygons' figure (solid on dark canvas).",
            )
            _stateful_number_input(
                "Overlay opacity",
                key=seg_overlay_op_key,
                default=float(_to_float(_repr_state("seg_overlay_opacity", 0.45), 0.45)),
                min_value=0.05, max_value=1.0, step=0.05, format="%.2f",
                help="Opacity for the 'Phenotype on markers' overlay figure.",
            )
            _stateful_checkbox(
                "Export individual marker channels",
                key=export_ind_markers_key,
                default=bool(_repr_state("export_individual_markers", False)),
                help="When enabled, each selected marker is rendered as a separate download in addition to the composite figure.",
            )

        with _sc3:
            _stateful_checkbox(
                "Draw zone distance contour",
                key=show_contour_key,
                default=bool(_repr_state("show_distance_contour", False)),
                help=(
                    "Replace the convex-hull zone outline with interpolated iso-lines at the "
                    "zone-start and zone-end distances.  Requires a continuous distance column "
                    "(e.g. ``neighborhood_distance_tumor``) in the unified CSV."
                ),
            )
            _stateful_text_input(
                "Distance column for contour",
                key=distance_col_key,
                default=str(_repr_state("distance_col", _dist_col_default)),
                help="Column in the unified CSV that contains the continuous signed distance to the tumor boundary.",
            )
            if st.session_state.get(show_contour_key, False):
                if contour_color_key not in st.session_state:
                    st.session_state[contour_color_key] = str(_repr_state("contour_color", "#e74c3c") or "#e74c3c")
                st.color_picker("Contour line color", key=contour_color_key)
                _stateful_number_input(
                    "Contour line width",
                    key=contour_lw_key,
                    default=float(_to_float(_repr_state("contour_linewidth", 2.0), 2.0)),
                    min_value=0.5, step=0.5, format="%.1f",
                )
    # ── Collect segmentation settings from new expander ─────────────────────
    _seg_mode_val       = str(st.session_state.get(seg_mode_key, "None"))
    _geojson_root_val   = str(st.session_state.get(geojson_root_key, ""))
    _geojson_suf_val    = [
        s.strip()
        for s in str(st.session_state.get(geojson_suffixes_key, "")).splitlines()
        if s.strip()
    ]
    _geojson_flip_y     = bool(st.session_state.get(geojson_flip_y_key, False))
    _geojson_match_dist = float(_to_float(st.session_state.get(geojson_match_dist_key, 50.0), 50.0))
    _voronoi_pct        = float(_to_float(st.session_state.get(voronoi_pct_key, 50.0), 50.0))
    _voronoi_fallback   = float(_to_float(st.session_state.get(voronoi_fallback_key, 15.0), 15.0))
    _circle_radius      = float(_to_float(st.session_state.get(circle_radius_key, 5.0), 5.0))
    _show_contour       = bool(st.session_state.get(show_contour_key, False))
    _distance_col       = str(st.session_state.get(distance_col_key, _dist_col_default))
    _contour_color      = str(st.session_state.get(contour_color_key, "#e74c3c") or "#e74c3c")
    _contour_lw         = float(_to_float(st.session_state.get(contour_lw_key, 2.0), 2.0))
    _seg_fill_op        = float(_to_float(st.session_state.get(seg_fill_op_key, 1.0), 1.0))
    _seg_overlay_op     = float(_to_float(st.session_state.get(seg_overlay_op_key, 0.45), 0.45))
    _export_ind_markers = bool(st.session_state.get(export_ind_markers_key, False))

    settings = {
        "render_mode": st.session_state.get(render_mode_key, _repr_state("render_mode", "Scatter points")),
        "show_background": bool(st.session_state.get(show_background_key, _repr_state("show_background", True))),
        "show_legend": bool(st.session_state.get(show_legend_key, _repr_state("show_legend", True))),
        "show_zone_overlay": bool(st.session_state.get(show_zone_overlay_key, _repr_state("show_zone_overlay", True))),
        "zone_hatched": bool(st.session_state.get(zone_hatched_key, _repr_state("zone_hatched", True))),
        "dim_outside_zone": bool(st.session_state.get(dim_outside_zone_key, _repr_state("dim_outside_zone", True))),
        "show_title": bool(st.session_state.get(show_title_key, _repr_state("show_title", True))),
        "point_alpha": float(_to_float(st.session_state.get(point_alpha_key, _repr_state("point_alpha", 0.80)), 0.80)),
        "point_size": float(_to_float(st.session_state.get(marker_size_key, _repr_state("marker_size", 18.0)), 18.0)),
        "zone_color": st.session_state.get(zone_color_key, _repr_state("zone_color", "#1f4e79")),
        "zone_linewidth": float(_to_float(st.session_state.get(zone_linewidth_key, _repr_state("zone_linewidth", 1.5)), 1.5)),
        "voronoi_max_area": float(_to_float(st.session_state.get(voronoi_max_area_key, _repr_state("voronoi_max_area", 400.0)), 400.0)),
        "draw_centers": bool(st.session_state.get(draw_centers_key, _repr_state("draw_centers", False))),
        "draw_edges": bool(st.session_state.get(draw_edges_key, _repr_state("draw_edges", False))),
        "annotation_text": st.session_state.get(annotation_text_key, _repr_state("annotation_text", "")),
        "feature_overlay_mode": st.session_state.get(overlay_mode_key, overlay_default_mode if overlay_supported else "None"),
        "feature_overlay_color": st.session_state.get(feature_overlay_color_key, _repr_state("feature_overlay_color", "#ff6b00")),
        "feature_overlay_alpha": float(_to_float(st.session_state.get(feature_overlay_alpha_key, _repr_state("feature_overlay_alpha", 0.70)), 0.70)),
        "feature_overlay_size": float(_to_float(st.session_state.get(feature_overlay_size_key, _repr_state("feature_overlay_size", 36.0)), 36.0)),
    }

    zone_start_override = float(st.session_state.get(zone_start_key, base_zone_start)) if explicit_interval_mode else 0.0
    zone_end_override   = float(st.session_state.get(zone_end_key, base_zone_end))
    if zone_end_override < zone_start_override:
        zone_start_override, zone_end_override = zone_end_override, zone_start_override

    export_dpi = int(max(_to_float(st.session_state.get(export_dpi_key, _repr_state("export_dpi", 300)), 300.0), 72.0))
    preview_max_dim = int(max(_to_float(st.session_state.get("tab8_fs_thumbnail_max_dim", 700), 700.0), 300.0))
    export_max_dim = int(max(_to_float(st.session_state.get(export_max_dim_key, _repr_state("export_max_dim", 2200)), 2200.0), float(preview_max_dim)))
    selected_markers = [str(value) for value in st.session_state.get(selected_markers_key, []) if str(value) in marker_options]
    marker_color_mapping = {
        marker_name: st.session_state.get(
            _repr_key(f"marker_color__{normalize_image_stem(str(marker_name))}"),
            _repr_state(f"marker_color__{normalize_image_stem(str(marker_name))}", marker_default_colors.get(marker_name, "#ffffff")),
        )
        for marker_name in selected_markers
    }
    color_mapping = {
        str(phenotype): st.session_state.get(_repr_key(f"color__{normalize_image_stem(str(phenotype))}"), _repr_state(f"color__{normalize_image_stem(str(phenotype))}", color))
        for phenotype, color in _representative_default_colors(visible_phenotypes).items()
    }

    patient_feature_lookup = (
        patient_hit.groupby("patient", dropna=False)["value"].first().to_dict()
        if not patient_hit.empty else {}
    )
    motif_export_frames: list[pd.DataFrame] = []
    motif_instance_summary_frames: list[pd.DataFrame] = []

    group_columns = st.columns(2)
    for group_name, group_rows, container in [("High", high_rows, group_columns[0]), ("Low", low_rows, group_columns[1])]:
        with container:
            st.markdown(f"#### {group_name}-feature representatives")
            if group_rows.empty:
                st.caption(f"No {group_name.lower()} representatives were found for this hit.")
                continue
            for index, row in enumerate(group_rows.itertuples(index=False), start=1):
                region_key = (str(row.batch), str(row.roi))
                region_df = representative_regions.get(region_key, pd.DataFrame()).copy()
                if region_df.empty:
                    st.caption(f"No unified-CSV cells were found for {row.batch} | {row.roi}.")
                    continue
                zone_df = _representative_zone_subset(region_df, config, zone_end_override, zone_start_override)
                view_specs = _representative_view_specs(region_df, zone_df, config, target_phenotypes, visible_phenotypes)
                if not view_specs:
                    st.caption(f"No representative views remain after the current phenotype filter for {row.batch} | {row.roi}.")
                    continue

                image_key = normalize_image_stem(str(row.batch)).lower()
                image_path = image_lookup.get(image_key)
                if settings["show_background"]:
                    image, scale_x, scale_y = _load_representative_preview(
                        image_path,
                        preview_max_dim,
                        selected_markers=selected_markers,
                        marker_color_mapping=marker_color_mapping,
                    )
                    if export_max_dim > preview_max_dim:
                        export_image, export_scale_x, export_scale_y = _load_representative_preview(
                            image_path,
                            export_max_dim,
                            selected_markers=selected_markers,
                            marker_color_mapping=marker_color_mapping,
                        )
                    else:
                        export_image, export_scale_x, export_scale_y = image, scale_x, scale_y
                else:
                    image, scale_x, scale_y = None, 1.0, 1.0
                    export_image, export_scale_x, export_scale_y = None, 1.0, 1.0
                region_plot_df = _representative_plot_df(region_df, config, scale_x, scale_y)
                if region_plot_df.empty:
                    st.caption(f"No plottable coordinates were found for {row.batch} | {row.roi}.")
                    continue
                zone_plot_df = region_plot_df.loc[region_plot_df.index.intersection(zone_df.index)].copy()
                export_region_plot_df = _representative_plot_df(region_df, config, export_scale_x, export_scale_y)
                export_zone_plot_df = export_region_plot_df.loc[export_region_plot_df.index.intersection(zone_df.index)].copy()

                base_motif_data: dict[str, Any] | None = None
                preview_motif_data: dict[str, Any] | None = None
                export_motif_data: dict[str, Any] | None = None
                motif_export_df = pd.DataFrame()
                motif_instance_summary_df = pd.DataFrame()
                if feature_family in {"dyad", "triad"} and target_phenotypes:
                    base_motif_data = compute_roi_motif_visualization_data(
                        zone_df=zone_df,
                        config=config,
                        target_phenotypes=target_phenotypes,
                        scale_x=1.0,
                        scale_y=1.0,
                    )
                    preview_motif_data = _scale_motif_visualization_data(base_motif_data, scale_x, scale_y)
                    export_motif_data = _scale_motif_visualization_data(base_motif_data, export_scale_x, export_scale_y)
                    representative_meta = {
                        **row._asdict(),
                        "source_group": group_name,
                        "roi_feature_value": float(_to_float(row.value, float("nan"))),
                        "patient_feature_value": _to_float(patient_feature_lookup.get(str(row.patient)), float("nan")),
                        "section_label": section_label,
                    }
                    motif_export_df = _build_representative_motif_export_df(
                        zone_df=zone_df,
                        motif_data=base_motif_data,
                        hit_row=hit_row,
                        representative_meta=representative_meta,
                        source_section=section_label,
                        zone_start=zone_start_override,
                        zone_end=zone_end_override,
                    )
                    if not motif_export_df.empty:
                        motif_export_frames.append(motif_export_df)
                        motif_instance_summary_df = _build_representative_motif_instance_summary_df(
                            motif_export_df,
                            phenotype_col=phenotype_col,
                            x_col=x_col,
                            y_col=y_col,
                        )
                        if not motif_instance_summary_df.empty:
                            motif_instance_summary_frames.append(motif_instance_summary_df)

                preview_feature_overlay = _build_representative_feature_overlay(
                    hit_row,
                    zone_plot_df,
                    target_phenotypes,
                    motif_data=preview_motif_data,
                )
                export_feature_overlay = _build_representative_feature_overlay(
                    hit_row,
                    export_zone_plot_df,
                    target_phenotypes,
                    motif_data=export_motif_data,
                )

                # ── Build cell segmentation features for this ROI ──────────────────────────
                _cell_feat_preview: list[dict] = []
                _cell_feat_export: list[dict] = []
                if _seg_mode_val == "GeoJSON" and _geojson_root_val:
                    _gj_path = find_geojson_for_batch(
                        _geojson_root_val, str(row.batch),
                        _geojson_suf_val or None,
                    )
                    if _gj_path is not None:
                        try:
                            _cell_feat_preview = load_geojson_cell_segmentation(
                                _gj_path, zone_df, config, scale_x, scale_y,
                                flip_y=_geojson_flip_y, max_match_distance=_geojson_match_dist,
                            )
                            _cell_feat_export = (
                                load_geojson_cell_segmentation(
                                    _gj_path, zone_df, config, export_scale_x, export_scale_y,
                                    flip_y=_geojson_flip_y, max_match_distance=_geojson_match_dist,
                                )
                                if (export_scale_x != scale_x or export_scale_y != scale_y)
                                else _cell_feat_preview
                            )
                        except Exception:
                            _cell_feat_preview = []
                    else:
                        st.caption(f"No GeoJSON file found for batch '{row.batch}' in `{_geojson_root_val}`.")
                elif _seg_mode_val == "Voronoi":
                    _cell_feat_preview = build_voronoi_cell_segmentation(
                        zone_df, config, scale_x, scale_y, _voronoi_pct, _voronoi_fallback,
                    )
                    _cell_feat_export = (
                        build_voronoi_cell_segmentation(
                            zone_df, config, export_scale_x, export_scale_y,
                            _voronoi_pct, _voronoi_fallback,
                        )
                        if (export_scale_x != scale_x or export_scale_y != scale_y)
                        else _cell_feat_preview
                    )
                elif _seg_mode_val == "Circle":
                    _cell_feat_preview = build_circle_cell_segmentation(
                        zone_df, config, scale_x, scale_y, _circle_radius,
                    )
                    _cell_feat_export = (
                        build_circle_cell_segmentation(
                            zone_df, config, export_scale_x, export_scale_y, _circle_radius,
                        )
                        if (export_scale_x != scale_x or export_scale_y != scale_y)
                        else _cell_feat_preview
                    )

                # ── Per-marker individual exports (load per-channel images) ───────────────────
                _marker_channel_images: dict[str, "np.ndarray | None"] = {}
                _marker_channel_images_export: dict[str, "np.ndarray | None"] = {}
                if _export_ind_markers and selected_markers and image_path is not None:
                    for _mk in selected_markers:
                        try:
                            _mk_arr_p, _, _ = _load_representative_preview(
                                image_path, preview_max_dim,
                                selected_markers=[_mk],
                                marker_color_mapping={_mk: marker_color_mapping.get(_mk, "#ffffff")},
                            )
                            _marker_channel_images[_mk] = _mk_arr_p
                            if export_max_dim > preview_max_dim:
                                _mk_arr_e, _, _ = _load_representative_preview(
                                    image_path, export_max_dim,
                                    selected_markers=[_mk],
                                    marker_color_mapping={_mk: marker_color_mapping.get(_mk, "#ffffff")},
                                )
                                _marker_channel_images_export[_mk] = _mk_arr_e
                            else:
                                _marker_channel_images_export[_mk] = _mk_arr_p
                        except Exception:
                            _marker_channel_images[_mk] = None

                # ── Assemble all tab names ────────────────────────────────────────────────────
                _extra_tab_names: list[str] = []
                if _cell_feat_preview:
                    _extra_tab_names.append("Phenotype polygons")
                    if image is not None:
                        _extra_tab_names.append("Phenotype on markers")
                if _show_contour and _distance_col in region_df.columns:
                    _extra_tab_names.append("Zone boundary contour")
                if _export_ind_markers and _marker_channel_images:
                    for _mk in selected_markers:
                        if _marker_channel_images.get(_mk) is not None:
                            _extra_tab_names.append(f"Marker: {_mk}")

                with st.expander(
                    f"{index}. {row.batch} | {row.roi} | patient={row.patient} | value={float(row.value):.3f}",
                    expanded=(index == 1),
                ):
                    if image_path is None:
                        st.caption("No raw image match was found for this batch. The figures below use only the spatial coordinates from the unified CSV.")
                    _all_tab_names = [spec["title"] for spec in view_specs] + _extra_tab_names
                    tabs = st.tabs(_all_tab_names)
                    for tab, spec in zip(tabs[:len(view_specs)], view_specs):
                        with tab:
                            st.caption(spec["caption"])
                            display_plot_df = region_plot_df.loc[region_plot_df.index.intersection(spec["display_df"].index)].copy()
                            export_display_plot_df = export_region_plot_df.loc[export_region_plot_df.index.intersection(spec["display_df"].index)].copy()
                            fig = _representative_figure(
                                image=image,
                                region_plot_df=region_plot_df,
                                display_plot_df=display_plot_df,
                                zone_plot_df=zone_plot_df,
                                color_mapping=color_mapping,
                                title=f"{group_name} | {row.batch} | {row.roi} | {spec['title']}",
                                legend_title=f"Batch: {row.batch}\nPhenotype",
                                settings=settings,
                                show_context_background=bool(spec.get("show_context_background", False)),
                                target_phenotypes=target_phenotypes,
                                interaction_distance=float(config.get("screening", {}).get("auto_interaction_distance_value", 30.0)) * scale_x,
                                feature_overlay=preview_feature_overlay,
                            )
                            export_fig = fig
                            if export_image is not image or export_scale_x != scale_x or export_scale_y != scale_y:
                                export_fig = _representative_figure(
                                    image=export_image,
                                    region_plot_df=export_region_plot_df,
                                    display_plot_df=export_display_plot_df,
                                    zone_plot_df=export_zone_plot_df,
                                    color_mapping=color_mapping,
                                    title=f"{group_name} | {row.batch} | {row.roi} | {spec['title']}",
                                    legend_title=f"Batch: {row.batch}\nPhenotype",
                                    settings=settings,
                                    show_context_background=bool(spec.get("show_context_background", False)),
                                    target_phenotypes=target_phenotypes,
                                    interaction_distance=float(config.get("screening", {}).get("auto_interaction_distance_value", 30.0)) * export_scale_x,
                                    feature_overlay=export_feature_overlay,
                                )
                            key_suffix = f"{state_prefix}_{group_name.lower()}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_{normalize_image_stem(spec['title'])}"
                            _render_figure_with_download(fig, spec["title"], key_suffix, download_dpi=export_dpi, download_fig=export_fig)
                            plt.close(fig)
                            if export_fig is not fig:
                                plt.close(export_fig)

                    # ── Extra figure tabs: segmentation / contour / per-marker ──
                    _extra_tabs = tabs[len(view_specs):]
                    _extra_i = 0
                    _roi_slug = f"{state_prefix}_{group_name.lower()}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}"
                    _seg_settings_override = {
                        **settings,
                        "zone_color": _contour_color,
                        "zone_linewidth": _contour_lw,
                        "show_zone_overlay": not _show_contour,  # convex hull only if no contour
                    }

                    if _cell_feat_preview:
                        # ── Phenotype polygons (solid, dark canvas) ──────────────
                        with _extra_tabs[_extra_i]:
                            _seg_fig = plot_cell_segmentation_figure(
                                region_df=region_df,
                                zone_df=zone_df,
                                cell_features=_cell_feat_preview,
                                color_mapping=color_mapping,
                                image=None,
                                scale_x=scale_x,
                                scale_y=scale_y,
                                config=config,
                                settings=_seg_settings_override,
                                title=f"{group_name} | {row.batch} | {row.roi} | Phenotype polygons",
                                zone_start=zone_start_override,
                                zone_end=zone_end_override,
                                overlay_on_image=False,
                                fill_opacity=_seg_fill_op,
                                show_zone_contour=_show_contour,
                                distance_col=_distance_col,
                                phenotype_filter=set(visible_phenotypes),
                            )
                            _seg_export_fig = _seg_fig
                            if _cell_feat_export is not _cell_feat_preview:
                                _seg_export_fig = plot_cell_segmentation_figure(
                                    region_df=region_df,
                                    zone_df=zone_df,
                                    cell_features=_cell_feat_export,
                                    color_mapping=color_mapping,
                                    image=None,
                                    scale_x=export_scale_x,
                                    scale_y=export_scale_y,
                                    config=config,
                                    settings=_seg_settings_override,
                                    title=f"{group_name} | {row.batch} | {row.roi} | Phenotype polygons",
                                    zone_start=zone_start_override,
                                    zone_end=zone_end_override,
                                    overlay_on_image=False,
                                    fill_opacity=_seg_fill_op,
                                    show_zone_contour=_show_contour,
                                    distance_col=_distance_col,
                                    phenotype_filter=set(visible_phenotypes),
                                )
                            _render_figure_with_download(
                                _seg_fig, "Phenotype polygons",
                                f"{_roi_slug}_segpoly",
                                download_dpi=export_dpi,
                                download_fig=_seg_export_fig,
                            )
                            plt.close(_seg_fig)
                            if _seg_export_fig is not _seg_fig:
                                plt.close(_seg_export_fig)
                        _extra_i += 1

                        # ── Phenotype on markers (overlay) ────────────────────────
                        if image is not None:
                            with _extra_tabs[_extra_i]:
                                _ov_fig = plot_cell_segmentation_figure(
                                    region_df=region_df,
                                    zone_df=zone_df,
                                    cell_features=_cell_feat_preview,
                                    color_mapping=color_mapping,
                                    image=image,
                                    scale_x=scale_x,
                                    scale_y=scale_y,
                                    config=config,
                                    settings=_seg_settings_override,
                                    title=f"{group_name} | {row.batch} | {row.roi} | Phenotype on markers",
                                    zone_start=zone_start_override,
                                    zone_end=zone_end_override,
                                    overlay_on_image=True,
                                    fill_opacity=_seg_overlay_op,
                                    show_zone_contour=_show_contour,
                                    distance_col=_distance_col,
                                    phenotype_filter=set(visible_phenotypes),
                                )
                                _ov_export_fig = _ov_fig
                                if _cell_feat_export is not _cell_feat_preview:
                                    _ov_export_fig = plot_cell_segmentation_figure(
                                        region_df=region_df,
                                        zone_df=zone_df,
                                        cell_features=_cell_feat_export,
                                        color_mapping=color_mapping,
                                        image=export_image,
                                        scale_x=export_scale_x,
                                        scale_y=export_scale_y,
                                        config=config,
                                        settings=_seg_settings_override,
                                        title=f"{group_name} | {row.batch} | {row.roi} | Phenotype on markers",
                                        zone_start=zone_start_override,
                                        zone_end=zone_end_override,
                                        overlay_on_image=True,
                                        fill_opacity=_seg_overlay_op,
                                        show_zone_contour=_show_contour,
                                        distance_col=_distance_col,
                                        phenotype_filter=set(visible_phenotypes),
                                    )
                                _render_figure_with_download(
                                    _ov_fig, "Phenotype on markers",
                                    f"{_roi_slug}_segoverlay",
                                    download_dpi=export_dpi,
                                    download_fig=_ov_export_fig,
                                )
                                plt.close(_ov_fig)
                                if _ov_export_fig is not _ov_fig:
                                    plt.close(_ov_export_fig)
                            _extra_i += 1

                    # ── Zone boundary contour (scatter + iso-lines only) ──────────
                    if _show_contour and _distance_col in region_df.columns:
                        with _extra_tabs[_extra_i]:
                            _dc_settings = {**_seg_settings_override, "show_background": True, "show_zone_overlay": False}
                            _dc_fig = plot_cell_segmentation_figure(
                                region_df=region_df,
                                zone_df=zone_df,
                                cell_features=[],        # no polygons — scatter drawn manually below
                                color_mapping=color_mapping,
                                image=image,
                                scale_x=scale_x,
                                scale_y=scale_y,
                                config=config,
                                settings=_dc_settings,
                                title=f"{group_name} | {row.batch} | {row.roi} | Zone boundary contour ({_distance_col})",
                                zone_start=zone_start_override,
                                zone_end=zone_end_override,
                                overlay_on_image=False,
                                fill_opacity=0.0,
                                show_zone_contour=True,
                                distance_col=_distance_col,
                            )
                            # Add scatter of zone cells on top for tissue context
                            if zone_plot_df is not None and not zone_plot_df.empty:
                                _dc_visible = zone_plot_df[zone_plot_df["__phenotype__"].isin(set(visible_phenotypes))]
                                _render_representative_scatter(
                                    _dc_fig.axes[0],
                                    _dc_visible,
                                    color_mapping,
                                    point_alpha=float(settings.get("point_alpha", 0.5)) * 0.65,
                                    point_size=float(settings.get("point_size", 12.0)) * 0.65,
                                )
                            _render_figure_with_download(
                                _dc_fig, "Zone boundary contour",
                                f"{_roi_slug}_zonecontour",
                                download_dpi=export_dpi,
                            )
                            plt.close(_dc_fig)
                        _extra_i += 1

                    # ── Individual marker channel figures ─────────────────────────
                    if _export_ind_markers and _marker_channel_images:
                        for _mk_name in selected_markers:
                            _mk_img = _marker_channel_images.get(_mk_name)
                            if _mk_img is None:
                                continue
                            with _extra_tabs[_extra_i]:
                                _mk_fig, _mk_ax = plt.subplots(figsize=(7.2, 6.4))
                                _mk_ax.imshow(_mk_img)
                                _mk_ax.set_axis_off()
                                if bool(settings.get("show_title", True)):
                                    _mk_ax.set_title(
                                        f"{group_name} | {row.batch} | {row.roi} | {_mk_name}",
                                        loc="left", fontsize=11,
                                    )
                                _annotation = str(settings.get("annotation_text", "")).strip()
                                if _annotation:
                                    _mk_fig.text(0.985, 0.02, _annotation, ha="right", va="bottom", fontsize=8)
                                _mk_fig.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.92)
                                _mk_export_fig = _mk_fig
                                _mk_export_img = _marker_channel_images_export.get(_mk_name)
                                if _mk_export_img is not None and _mk_export_img is not _mk_img:
                                    _mk_export_fig, _mk_export_ax = plt.subplots(figsize=(7.2, 6.4))
                                    _mk_export_ax.imshow(_mk_export_img)
                                    _mk_export_ax.set_axis_off()
                                    if bool(settings.get("show_title", True)):
                                        _mk_export_ax.set_title(
                                            f"{group_name} | {row.batch} | {row.roi} | {_mk_name}",
                                            loc="left", fontsize=11,
                                        )
                                    if _annotation:
                                        _mk_export_fig.text(0.985, 0.02, _annotation, ha="right", va="bottom", fontsize=8)
                                    _mk_export_fig.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.92)
                                _render_figure_with_download(
                                    _mk_fig, f"Marker: {_mk_name}",
                                    f"{_roi_slug}_marker_{normalize_image_stem(str(_mk_name))}",
                                    download_dpi=export_dpi,
                                    download_fig=_mk_export_fig,
                                )
                                plt.close(_mk_fig)
                                if _mk_export_fig is not _mk_fig:
                                    plt.close(_mk_export_fig)
                            _extra_i += 1

                    # ── Motif-specific figures (dyad / triad only) ────────────────
                    if feature_family in {"dyad", "triad"} and target_phenotypes:
                        _motif_zone_df = zone_df
                        motif_data = preview_motif_data or compute_roi_motif_visualization_data(
                            zone_df=_motif_zone_df,
                            config=config,
                            target_phenotypes=target_phenotypes,
                            scale_x=scale_x,
                            scale_y=scale_y,
                        )
                        export_motif_display = export_motif_data or compute_roi_motif_visualization_data(
                            zone_df=_motif_zone_df,
                            config=config,
                            target_phenotypes=target_phenotypes,
                            scale_x=export_scale_x,
                            scale_y=export_scale_y,
                        )
                        perm_n_iter_display = max(200, min(500, int(st.session_state.get("tab8_fs_dyad_triad_permutation_n_iter", 1000) // 2)))
                        perm_seed_display = int(st.session_state.get("tab8_fs_bootstrap_random_seed", 20260407))
                        motif_label_str = (
                            " ↔ ".join(target_phenotypes) if feature_family == "dyad"
                            else " + ".join(target_phenotypes)
                        )
                        st.markdown(f"##### Motif analysis: {motif_label_str}")
                        # ── Diagnostic info to help debug empty-motif cases ─────
                        _pheno_col = str(config["columns"]["phenotype"])
                        _zone_n_cells = len(_motif_zone_df)
                        _target_ph_counts = {
                            ph: int((_motif_zone_df[_pheno_col].astype(str) == ph).sum())
                            for ph in target_phenotypes
                            if _pheno_col in _motif_zone_df.columns
                        }
                        _direct_nn_mode = bool(config.get("screening", {}).get("adaptive_radius_direct_neighbours_only", True))
                        st.caption(
                            f"Zone: {_zone_n_cells} cells | "
                            f"Target cell counts: { {k: v for k, v in _target_ph_counts.items()} } | "
                            f"Radius mode: {'direct neighbours (1×p95 NN)' if _direct_nn_mode else 'wide (3×p95 NN, p99 trim)'} | "
                            f"Motif instances found: {motif_data['n_found']} | "
                            f"Interaction radius: {motif_data['max_px_scaled']:.1f} px (display space)"
                        )
                        if not motif_export_df.empty:
                            st.caption(
                                f"This ROI contains {motif_export_df['motif_instance_id'].nunique()} motif instance(s). The cell-level CSV exports one row per cell in each {feature_family} instance, while the instance-level CSV collapses each motif to one row with flattened per-cell fields plus the hit, patient, and ROI metadata."
                            )
                            _render_dataframe_download(
                                motif_export_df,
                                "Download this ROI motif cell-level CSV",
                                f"representative_{section_label.lower()}_{feature_family}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_cells.csv",
                                key=f"{state_prefix}_{group_name.lower()}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_motif_cell_csv_download",
                            )
                            if not motif_instance_summary_df.empty:
                                _render_dataframe_download(
                                    motif_instance_summary_df,
                                    "Download this ROI motif instance CSV",
                                    f"representative_{section_label.lower()}_{feature_family}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_instances.csv",
                                    key=f"{state_prefix}_{group_name.lower()}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_motif_instance_csv_download",
                                )
                        _stateful_checkbox(
                            "Show motif figures",
                            key=f"{state_prefix}_show_motif_{group_name}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}",
                            default=True,
                            help="These three figures visualize the spatial motif (dyad or triad) found in this ROI: a density cloud, a zoomed cell-level view, and the permutation significance histogram.",
                        )
                        show_motif_key = f"{state_prefix}_show_motif_{group_name}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}"
                        if bool(st.session_state.get(show_motif_key, True)):
                            if motif_data["n_found"] == 0:
                                st.warning(
                                    f"No {feature_family} instances of **{motif_label_str}** were found in this ROI zone "
                                    f"({_zone_n_cells} cells, radius = {motif_data['max_px_scaled']:.1f} px display space). "
                                    f"This ROI was selected because its **feature Z-score was high during screening** — the motif exists in the data. "
                                    f"Likely causes: (1) the representative interval differs from the screening interval "
                                    f"(screening used **{screen_zone_label}**, current override is **{format_zone_interval_label(zone_start_override, zone_end_override)}**), "
                                    f"or (2) the 'Direct neighbours only' radius is too strict for this tissue density. "
                                    f"Try: disable 'Direct neighbours only' in feature controls, or set the representative interval back to **{screen_zone_label}**."
                                )
                            else:
                                motif_col1, motif_col2 = st.columns(2)

                                key_base = f"{state_prefix}_{group_name.lower()}_{normalize_image_stem(str(row.batch))}_{normalize_image_stem(str(row.roi))}_motif"

                                with motif_col1:
                                    density_fig = plot_motif_density_cloud(
                                        zone_df=_motif_zone_df,
                                        motif_data=motif_data,
                                        target_phenotypes=target_phenotypes,
                                        color_mapping=color_mapping,
                                        image=image,
                                        scale_x=scale_x,
                                        scale_y=scale_y,
                                        title=f"{motif_label_str} | Density cloud",
                                    )
                                    density_export_fig = density_fig
                                    if export_image is not image or export_scale_x != scale_x or export_scale_y != scale_y:
                                        density_export_fig = plot_motif_density_cloud(
                                            zone_df=_motif_zone_df,
                                            motif_data=export_motif_display,
                                            target_phenotypes=target_phenotypes,
                                            color_mapping=color_mapping,
                                            image=export_image,
                                            scale_x=export_scale_x,
                                            scale_y=export_scale_y,
                                            title=f"{motif_label_str} | Density cloud",
                                        )
                                    _render_figure_with_download(density_fig, "Motif density cloud", f"{key_base}_density", download_dpi=export_dpi, download_fig=density_export_fig)
                                    plt.close(density_fig)
                                    if density_export_fig is not density_fig:
                                        plt.close(density_export_fig)

                                with motif_col2:
                                    zoom_fig = plot_motif_zoom(
                                        zone_df=_motif_zone_df,
                                        motif_data=motif_data,
                                        target_phenotypes=target_phenotypes,
                                        color_mapping=color_mapping,
                                        config=config,
                                        image=image,
                                        scale_x=scale_x,
                                        scale_y=scale_y,
                                        title=f"{motif_label_str} | Closest-to-median instance",
                                    )
                                    zoom_export_fig = zoom_fig
                                    if export_image is not image or export_scale_x != scale_x or export_scale_y != scale_y:
                                        zoom_export_fig = plot_motif_zoom(
                                            zone_df=_motif_zone_df,
                                            motif_data=export_motif_display,
                                            target_phenotypes=target_phenotypes,
                                            color_mapping=color_mapping,
                                            config=config,
                                            image=export_image,
                                            scale_x=export_scale_x,
                                            scale_y=export_scale_y,
                                            title=f"{motif_label_str} | Closest-to-median instance",
                                        )
                                    _render_figure_with_download(zoom_fig, "Motif zoom (cellular level)", f"{key_base}_zoom", download_dpi=export_dpi, download_fig=zoom_export_fig)
                                    plt.close(zoom_fig)
                                    if zoom_export_fig is not zoom_fig:
                                        plt.close(zoom_export_fig)

                                hist_fig = plot_motif_null_histogram(
                                    zone_df=_motif_zone_df,
                                    config=config,
                                    target_phenotypes=target_phenotypes,
                                    n_iter=perm_n_iter_display,
                                    seed=perm_seed_display,
                                    title=f"{motif_label_str} | Permutation significance ({perm_n_iter_display} shuffles)",
                                )
                                _render_figure_with_download(hist_fig, "Permutation null histogram", f"{key_base}_histogram", download_dpi=export_dpi)
                                plt.close(hist_fig)

    if motif_export_frames:
        motif_export_all = pd.concat(motif_export_frames, ignore_index=True)
        st.markdown(f"#### {section_label} motif CSV exports")
        st.caption(
            f"The cell-level export keeps one row per cell inside each representative dyad or triad instance for the selected hit. It preserves the original unified-CSV columns and adds motif IDs, centroid coordinates, hit statistics, patient-level values, and ROI-level representative metadata."
        )
        _render_dataframe_download(
            motif_export_all,
            f"Download {section_label.lower()} representative motif cell-level CSV",
            f"representative_{section_label.lower()}_{feature_family}_cells.csv",
            key=f"{state_prefix}_motif_cell_csv_download",
        )
        if motif_instance_summary_frames:
            motif_instance_summary_all = pd.concat(motif_instance_summary_frames, ignore_index=True)
            st.caption(
                f"The instance-level export collapses each representative dyad or triad to one row and expands the participating cells into flattened per-cell columns for easier manuscript review and spreadsheet filtering."
            )
            _render_dataframe_download(
                motif_instance_summary_all,
                f"Download {section_label.lower()} representative motif instance CSV",
                f"representative_{section_label.lower()}_{feature_family}_instances.csv",
                key=f"{state_prefix}_motif_instance_csv_download",
            )


def _suggest_validation_width(base_config: dict[str, Any]) -> float:
    results = st.session_state.get("_tab8_fs_results")
    if isinstance(results, dict):
        selected_width = _to_float(results.get("selected_width"), float("nan"))
        if np.isfinite(selected_width) and abs(float(selected_width)) > 0.0:
            return float(selected_width)
    max_width = _to_float(base_config.get("screening", {}).get("max_width"), float("nan"))
    if np.isfinite(max_width) and abs(float(max_width)) > 0.0:
        return float(max_width)
    round_width = max(_to_float(base_config.get("screening", {}).get("round_width_to"), 5.0), 1.0)
    return float(round_width * 10.0)


def _phenotype_column_options(raw_df: pd.DataFrame) -> list[str]:
    """Columns that plausibly hold a per-cell phenotype label.

    Used by the multi-method validation control. Restricted to low-cardinality
    object/categorical columns so a continuous marker column can never be
    offered as a phenotyping method.
    """
    options: list[str] = []
    for column in raw_df.columns:
        name = str(column)
        if "phenotype" not in name.lower() and "celltype" not in name.lower().replace("_", ""):
            continue
        try:
            series = raw_df[column]
            if pd.api.types.is_numeric_dtype(series):
                continue
            if int(series.nunique(dropna=True)) > 100:
                continue
        except Exception:
            continue
        options.append(name)
    return options


def _config_for_phenotype_column(
    base_validation_config: dict[str, Any],
    raw_df: pd.DataFrame,
    phenotype_column: str,
) -> dict[str, Any]:
    """Retarget a validation config at a different phenotype column.

    Each phenotyping method is assessed entirely on its own terms: the config is
    repointed at that column and the phenotype selection is re-derived from the
    labels that column actually contains. Carrying the discovery column's
    selection over unchanged would silently screen for labels a different method
    never produces.
    """
    config = copy.deepcopy(base_validation_config)
    config.setdefault("columns", {})["phenotype"] = str(phenotype_column)

    present = {str(value) for value in _safe_unique(raw_df[phenotype_column])}
    excluded = {str(value) for value in config.get("excluded_phenotypes", []) if str(value)}
    selected = [str(value) for value in config.get("selected_phenotypes", []) if str(value)]

    kept = [value for value in selected if value in present and value not in excluded]
    if not kept:
        kept = sorted(value for value in present if value not in excluded and value.lower() != "nan")
    config["selected_phenotypes"] = kept
    config["excluded_phenotypes"] = sorted(excluded & present)
    return config


def _build_multi_phenotype_comparison_df(results_by_column: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per (phenotype column, feature, endpoint) for side-by-side reading.

    No multiple-testing adjustment is applied across columns. Each method is its
    own independent assessment, so the per-method FDR already carried on each
    row is the correct one; pooling them would be a different question than the
    one this panel asks.
    """
    frames: list[pd.DataFrame] = []
    for column, results in results_by_column.items():
        stats_all = results.get("stats_all", pd.DataFrame()) if isinstance(results, dict) else pd.DataFrame()
        if not isinstance(stats_all, pd.DataFrame) or stats_all.empty:
            continue
        keep = [
            c for c in [
                "feature_family", "feature_label", "feature_name", "endpoint_label",
                "stratum_label", "p_value", "fdr_q", "significance_label",
                "hazard_ratio", "direction", "n_patients",
            ] if c in stats_all.columns
        ]
        frame = stats_all[keep].copy()
        frame.insert(0, "phenotype_column", str(column))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_validation_config(
    raw_df: pd.DataFrame,
    base_config: dict[str, Any],
    unified_csv_path: str | None,
) -> dict[str, Any]:
    validation_config = copy.deepcopy(base_config)
    validation_config["source_csv"] = unified_csv_path
    return validation_config


def _effective_screening_phenotypes(raw_df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    phenotype_col = str(config.get("columns", {}).get("phenotype") or "")
    if not phenotype_col or phenotype_col not in raw_df.columns:
        return []
    selected = [str(value) for value in config.get("selected_phenotypes", []) if str(value)]
    excluded = {str(value) for value in config.get("excluded_phenotypes", []) if str(value)}
    if selected:
        return [value for value in selected if value not in excluded]
    return [value for value in _safe_unique(raw_df[phenotype_col]) if value not in excluded]


def _feature_matches_focus_elements(feature_name: str, selected_elements: list[str], require_all: bool) -> bool:
    selected = {str(value) for value in selected_elements if str(value)}
    if not selected:
        return True
    _, components = parse_feature_components(feature_name)
    component_set = {str(value) for value in components if str(value)}
    if not component_set:
        return False
    if require_all:
        return selected.issubset(component_set)
    return bool(selected & component_set)


def _selected_discovery_hit_row(results: dict[str, Any]) -> pd.Series | None:
    stats_all = results.get("stats_all", pd.DataFrame()) if isinstance(results, dict) else pd.DataFrame()
    if not isinstance(stats_all, pd.DataFrame) or stats_all.empty:
        return None
    stats_view = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").copy()
    if stats_view.empty:
        return None
    stats_view["_label"] = stats_view.apply(
        lambda row: f"{row['endpoint_label']} | {row['feature_label']} | {row['stratum_label']} | range={str(row.get('zone_label') or format_zone_label(row['zone_width']))}",
        axis=1,
    )
    selected_hit_label = str(st.session_state.get("tab8_fs_hit_label", "") or "")
    if selected_hit_label and (stats_view["_label"].astype(str) == selected_hit_label).any():
        return stats_view.loc[stats_view["_label"].astype(str) == selected_hit_label].iloc[0]
    return stats_view.iloc[0]


def _render_validation_influence_section(
    raw_df: pd.DataFrame,
    validation_results: dict[str, Any],
    hit_row: pd.Series,
    selected_endpoint: str,
    selected_stratum: str,
) -> None:
    validation_config = validation_results.get("config", {})
    validation_width = float(validation_results.get("validation_width", 0.0))
    target_feature_name = str(hit_row.get("feature_name") or "")
    target_feature_label = str(hit_row.get("feature_label") or target_feature_name or "Selected feature")
    target_feature_family, target_components = parse_feature_components(target_feature_name)
    if not target_components:
        st.info("The highlighted feature does not expose phenotype components, so targeted driver analysis is not available for this feature.")
        return

    feature_signature = "|".join(
        [
            target_feature_name,
            str(selected_endpoint),
            str(selected_stratum),
            f"{validation_width:.6f}",
            str(validation_config.get("columns", {}).get("phenotype") or ""),
        ]
    )
    if st.session_state.get("_tab8_fs_val_influence_last_feature_key", "") != feature_signature:
        st.session_state["tab8_fs_val_influence_components"] = list(target_components)
        st.session_state["tab8_fs_val_influence_related_feature"] = ""
        st.session_state["_tab8_fs_val_influence_last_feature_key"] = feature_signature
        st.session_state.pop("_tab8_fs_validation_influence_results", None)
        st.session_state.pop("_tab8_fs_validation_influence_signature", None)

    st.caption(
        "This panel asks whether the highlighted validated feature is tracking simpler component-level patterns, "
        "such as density, proportion, dyads, or nearest-neighbour features built from the same phenotype elements."
    )

    influence_family_options = ["density", "proportion", "ratio", "dyad", "triad", "nn_distance"]
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        _stateful_multiselect(
            "Phenotype elements to probe",
            options=target_components,
            key="tab8_fs_val_influence_components",
            default=target_components,
            help="Pick one or more components from the highlighted feature. Related features are filtered by overlap with these phenotype elements.",
        )
    with ctrl_col2:
        _stateful_multiselect(
            "Potential driver families",
            options=influence_family_options,
            key="tab8_fs_val_influence_families",
            default=st.session_state.get("tab8_fs_val_influence_families", influence_family_options),
            help="Families to test as possible drivers of the highlighted feature.",
        )
    with ctrl_col3:
        _stateful_checkbox(
            "Require all selected elements",
            key="tab8_fs_val_influence_require_all",
            default=False,
            help="When ON, related features must include every selected phenotype element instead of just one overlapping element.",
        )
        _stateful_number_input(
            "Minimum paired patients",
            key="tab8_fs_val_influence_min_pairs",
            default=5,
            min_value=3,
            step=1,
            help="Correlation rows with fewer paired patients than this are dropped.",
        )
    with ctrl_col4:
        _stateful_number_input(
            "Top related features to plot",
            key="tab8_fs_val_influence_top_n",
            default=15,
            min_value=5,
            max_value=50,
            step=1,
            help="Maximum number of related features shown in the driver bar plot.",
        )

    selected_components = [str(value) for value in st.session_state.get("tab8_fs_val_influence_components", []) if str(value)]
    selected_families = [str(value) for value in st.session_state.get("tab8_fs_val_influence_families", []) if str(value)]
    require_all_components = bool(st.session_state.get("tab8_fs_val_influence_require_all", False))
    min_pairs = max(3, int(st.session_state.get("tab8_fs_val_influence_min_pairs", 5)))
    top_n = max(5, int(st.session_state.get("tab8_fs_val_influence_top_n", 15)))

    if not selected_components:
        st.info("Select at least one phenotype element to evaluate possible drivers.")
        return
    if not selected_families:
        st.info("Select at least one potential driver family.")
        return

    required_families = sorted({str(target_feature_family)} | {str(value) for value in selected_families})
    current_validation_families = {str(value) for value in validation_config.get("feature_families", []) if str(value)}
    can_reuse_validation = set(required_families).issubset(current_validation_families)

    influence_signature = json.dumps(
        {
            "width": float(validation_width),
            "phenotype_col": str(validation_config.get("columns", {}).get("phenotype") or ""),
            "selected_phenotypes": list(validation_config.get("selected_phenotypes", [])),
            "required_families": required_families,
        },
        sort_keys=True,
        default=str,
    )

    influence_results = validation_results if can_reuse_validation else None
    if can_reuse_validation:
        st.caption("Using the current validation run because it already contains all requested potential-driver families.")
    else:
        st.caption(
            "The current validation run does not include all requested driver families. "
            "Use the button below to run one auxiliary fixed-width validation for those families only."
        )
        if st.button(
            "Run influence correlation",
            key="tab8_fs_val_influence_run_btn",
            width="stretch",
            help="Runs a fixed-width auxiliary validation including the highlighted feature family plus the requested potential-driver families.",
        ):
            influence_config = copy.deepcopy(validation_config)
            influence_config["feature_families"] = required_families
            try:
                aux_results = run_feature_validation_pipeline(
                    raw_df,
                    influence_config,
                    float(validation_width),
                    log_fn=_log_terminal_event,
                )
                st.session_state["_tab8_fs_validation_influence_results"] = aux_results
                st.session_state["_tab8_fs_validation_influence_signature"] = influence_signature
                st.success("Influence correlation feature set prepared.")
            except Exception as exc:
                st.error(f"Influence correlation run failed: {exc}")

        if st.session_state.get("_tab8_fs_validation_influence_signature") == influence_signature:
            influence_results = st.session_state.get("_tab8_fs_validation_influence_results")
        else:
            st.info("Run the influence correlation step to generate the requested driver families at this fixed width.")
            return

    if not isinstance(influence_results, dict):
        st.info("No influence-correlation result is available yet.")
        return

    influence_df = build_feature_influence_correlation_df(
        stats_df=influence_results.get("stats_all", pd.DataFrame()),
        patient_features_df=influence_results.get("patient_features", pd.DataFrame()),
        zone_width=float(influence_results.get("validation_width", validation_width)),
        endpoint_label=str(selected_endpoint),
        stratum_label=str(selected_stratum),
        target_feature_name=target_feature_name,
        selected_components=selected_components,
        related_families=selected_families,
        require_all_components=require_all_components,
        min_pairs=min_pairs,
    )
    if influence_df.empty:
        st.warning(
            "No related features met the current overlap and paired-patient criteria. "
            "Try broadening the component selection, lowering the minimum paired-patient threshold, or turning off 'Require all selected elements'."
        )
        return

    influence_display = influence_df.copy()
    influence_display["Status"] = influence_display["significance_label"].map(_significance_display_label)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Related features", str(int(len(influence_display))))
    metric_cols[1].metric("Top |ρ|", f"{float(pd.to_numeric(influence_display['abs_spearman_rho'], errors='coerce').iloc[0]):.2f}")
    metric_cols[2].metric("Top family", str(influence_display.iloc[0].get("related_feature_family") or "-"))
    metric_cols[3].metric("Shared elements", str(influence_display.iloc[0].get("shared_components") or "-"))

    table_columns = [
        column
        for column in [
            "validation_rank",
            "related_feature_family",
            "related_feature_label",
            "shared_components",
            "n_pairs",
            "spearman_rho",
            "spearman_p",
            "fdr_q",
            "Status",
        ]
        if column in influence_display.columns
    ]
    st.dataframe(influence_display[table_columns], width="stretch", height=min(420, 40 + 35 * (len(influence_display.head(12)) + 1)))
    _render_dataframe_download(
        influence_display,
        "Download influence-correlation table",
        "validation_influence_correlation.csv",
        "tab8_fs_val_influence_table_download",
    )

    plot_col1, plot_col2 = st.columns([1.05, 0.95])
    with plot_col1:
        bar_fig = plot_feature_influence_barplot(influence_display, target_feature_label=target_feature_label, top_n=top_n)
        _render_figure_with_download(bar_fig, "Potential driver summary", "tab8_fs_val_influence_barplot")
        plt.close(bar_fig)
    with plot_col2:
        related_feature_options = influence_display.apply(
            lambda row: (
                f"#{int(_to_float(row.get('validation_rank'), 0.0)) if np.isfinite(_to_float(row.get('validation_rank'), float('nan'))) else 0} | "
                f"{str(row.get('related_feature_label') or row.get('related_feature_name') or '')} | "
                f"ρ={_to_float(row.get('spearman_rho'), float('nan')):.2f}"
            ),
            axis=1,
        ).tolist()
        related_lookup = dict(zip(related_feature_options, influence_display.index.tolist()))
        _stateful_selectbox(
            "Detailed related feature",
            options=related_feature_options,
            key="tab8_fs_val_influence_related_feature",
            default=related_feature_options[0] if related_feature_options else None,
        )
        selected_related = st.session_state.get(
            "tab8_fs_val_influence_related_feature",
            related_feature_options[0] if related_feature_options else "",
        )
        related_row = influence_display.loc[int(related_lookup.get(selected_related, influence_display.index[0]))]
        pair_df = build_feature_pair_correlation_export_df(
            influence_results.get("patient_features", pd.DataFrame()),
            zone_width=float(influence_results.get("validation_width", validation_width)),
            stratum_label=str(selected_stratum),
            target_feature_name=target_feature_name,
            related_feature_name=str(related_row.get("related_feature_name") or ""),
        )
        scatter_fig = plot_feature_influence_scatter(
            pair_df,
            target_feature_label=target_feature_label,
            related_feature_label=str(related_row.get("related_feature_label") or related_row.get("related_feature_name") or "Related feature"),
        )
        _render_figure_with_download(scatter_fig, "Target vs related feature", "tab8_fs_val_influence_scatter")
        plt.close(scatter_fig)
        if not pair_df.empty:
            _render_dataframe_download(
                pair_df,
                "Download paired patient values",
                "validation_influence_pairs.csv",
                "tab8_fs_val_influence_pairs_download",
            )


def _format_secondary_stat(value: Any, decimals: int = 3) -> str:
    numeric = _to_float(value, float("nan"))
    return f"{numeric:.{int(decimals)}f}" if np.isfinite(numeric) else "NA"


def _secondary_metric_short_label(metric_col: str, fallback: str = "") -> str:
    mapping = {
        "raw_triad_proportion": "Raw triad proportion",
        "raw_triad_density": "Raw triad density",
    }
    return mapping.get(metric_col, fallback or metric_col.replace("_", " ").title())


def _secondary_metric_definition(metric_col: str) -> str:
    mapping = {
        "raw_triad_proportion": "Pooled selected triad count across valid ROIs / pooled all observed triads across valid ROIs",
        "raw_triad_density": "Pooled selected triad count across valid ROIs / pooled valid-zone area (mm^2)",
    }
    return mapping.get(metric_col, "")


def _build_secondary_summary_tables(
    secondary_results: dict[str, Any],
    hit_row: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_triad = secondary_results.get("target_triad") or ()
    triad_label = " + ".join(str(value) for value in target_triad) if target_triad else str(hit_row.get("feature_label") or hit_row.get("feature_name") or "Selected triad")
    zone_label = str(hit_row.get("zone_label") or "")
    endpoint_label = str(hit_row.get("endpoint_label") or "")
    stratum_label = str(hit_row.get("stratum_label") or "")

    display_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    metric_sections = [
        ("proportion_summary", "raw_triad_proportion"),
        ("density_summary", "raw_triad_density"),
    ]
    for summary_key, default_metric_col in metric_sections:
        summary = secondary_results.get(summary_key, {})
        if not isinstance(summary, dict) or not summary:
            continue
        metric_col = str(summary.get("metric_col") or default_metric_col)
        metric_label = _secondary_metric_short_label(metric_col, str(summary.get("metric_label") or ""))
        definition = _secondary_metric_definition(metric_col)
        hr = _to_float(summary.get("hazard_ratio_per_sd"), float("nan"))
        ci_low = _to_float(summary.get("ci_low_per_sd"), float("nan"))
        ci_high = _to_float(summary.get("ci_high_per_sd"), float("nan"))
        export_row = {
            "Selected triad": triad_label,
            "Endpoint": endpoint_label,
            "Range": zone_label,
            "Stratum": stratum_label,
            "Metric": metric_label,
            "Definition": definition,
            "Primary test": "Continuous Cox PH (per 1 SD increase)",
            "Patients in continuous Cox": int(_to_float(summary.get("n_included"), 0.0)),
            "Events": int(_to_float(summary.get("n_events"), 0.0)),
            "Zero-burden patients": int(_to_float(summary.get("n_zero_patients"), 0.0)),
            "Patients meeting primary Z-score threshold": int(_to_float(summary.get("n_primary_scoreable"), 0.0)),
            "HR per 1 SD increase": _format_secondary_stat(hr),
            "95% CI": (
                f"[{ci_low:.3f}, {ci_high:.3f}]"
                if np.isfinite(ci_low) and np.isfinite(ci_high)
                else "NA"
            ),
            "Cox p-value": _format_secondary_stat(summary.get("p_value")),
            "C-index": _format_secondary_stat(summary.get("concordance_index")),
            "PH test p-value": _format_secondary_stat(summary.get("ph_test_p")),
            "Median burden": _format_secondary_stat(summary.get("median_value")),
            "Minimum burden": _format_secondary_stat(summary.get("min_value")),
            "Maximum burden": _format_secondary_stat(summary.get("max_value")),
        }
        export_rows.append(export_row)
        display_rows.append(
            {
                "Metric": metric_label,
                "Primary test": export_row["Primary test"],
                "Definition": definition,
                "Patients in continuous Cox": export_row["Patients in continuous Cox"],
                "Events": export_row["Events"],
                "Zero-burden patients": export_row["Zero-burden patients"],
                "HR per 1 SD increase": export_row["HR per 1 SD increase"],
                "95% CI": export_row["95% CI"],
                "Cox p-value": export_row["Cox p-value"],
                "C-index": export_row["C-index"],
                "PH test p-value": export_row["PH test p-value"],
                "Median burden": export_row["Median burden"],
            }
        )

    return pd.DataFrame(display_rows), pd.DataFrame(export_rows)


def _build_secondary_patient_export_df(patient_df: pd.DataFrame) -> pd.DataFrame:
    if patient_df.empty:
        return pd.DataFrame()

    export_df = patient_df.copy()
    yes_no_columns = [
        "secondary_base_eligible",
        "has_zone_cells",
        "has_valid_zone_roi",
        "triad_present",
        "supports_primary_permutation_score",
    ]
    for column in yes_no_columns:
        if column in export_df.columns:
            export_df[column] = np.where(export_df[column].fillna(False).astype(bool), "Yes", "No")

    preferred_columns = [
        "patient",
        "secondary_metric_label",
        "secondary_zone_label",
        "stratum_label",
        "secondary_base_eligible",
        "secondary_exclusion_reason",
        "has_zone_cells",
        "has_valid_zone_roi",
        "n_zone_rois",
        "n_valid_zone_rois",
        "total_target_triads",
        "total_observed_triads",
        "total_zone_area_mm2",
        "triad_present",
        "supports_primary_permutation_score",
        "raw_triad_proportion",
        "raw_triad_density",
        "endpoint_time",
        "endpoint_censor",
        "event",
    ]
    rename_map = {
        "patient": "Patient",
        "secondary_metric_label": "Selected triad label",
        "secondary_zone_label": "Range",
        "stratum_label": "Stratum",
        "secondary_base_eligible": "Included in continuous secondary Cox",
        "secondary_exclusion_reason": "If excluded, why",
        "has_zone_cells": "Cells found in selected range",
        "has_valid_zone_roi": "Has valid ROI in selected range",
        "n_zone_rois": "ROIs intersecting selected range",
        "n_valid_zone_rois": "Valid ROIs contributing to pooled patient metric",
        "total_target_triads": "Pooled selected triad count across valid ROIs",
        "total_observed_triads": "Pooled all observed triads across valid ROIs",
        "total_zone_area_mm2": "Pooled valid-zone area (mm^2)",
        "triad_present": "Selected triad observed",
        "supports_primary_permutation_score": "Meets primary Z-score threshold",
        "raw_triad_proportion": "Patient-level pooled raw triad proportion",
        "raw_triad_density": "Patient-level pooled raw triad density (per mm^2)",
        "endpoint_time": "Survival time",
        "endpoint_censor": "Endpoint censor flag",
        "event": "Event observed (1=yes)",
    }
    selected_columns = [column for column in preferred_columns if column in export_df.columns]
    return export_df.loc[:, selected_columns].rename(columns=rename_map)


def _render_secondary_analysis_section(
    analysis_df: pd.DataFrame,
    patient_outcomes_df: pd.DataFrame,
    config: dict[str, Any],
    hit_row: pd.Series,
    zone_start: float,
    zone_end: float,
    state_prefix: str,
    download_prefix: str,
) -> None:
    st.caption(
        "Secondary analysis aggregates to the patient level, keeps zero-valued raw triad burden, and fits a continuous Cox model per 1 SD increase. "
        "No median split is used for the primary inference; any Kaplan-Meier plots below are optional grouped visualizations only."
    )
    st.caption(
        "Patient-level aggregation pools counts and area across each patient's valid ROIs before computing burden metrics. "
        "It does not take the median of ROI-level proportions or densities."
    )

    if str(hit_row.get("feature_family") or "") != "triad":
        st.info("Secondary analysis is currently available only for highlighted triad features.")
        return
    if str(hit_row.get("endpoint_type") or "") != "survival":
        st.info("Secondary analysis is currently available only for survival endpoints.")
        return

    secondary_results = build_triad_secondary_analysis(
        hit_row,
        analysis_df,
        patient_outcomes_df,
        config,
        zone_start=zone_start,
        zone_end=zone_end,
    )
    if str(secondary_results.get("status") or "") != "ok":
        st.info(str(secondary_results.get("message") or "Secondary analysis could not be prepared for this hit."))
        return

    patient_df = secondary_results.get("patient_df", pd.DataFrame())
    if patient_df.empty:
        st.info("No patient-level data were available for the requested secondary analysis.")
        return

    summary_display_df, summary_export_df = _build_secondary_summary_tables(secondary_results, hit_row)
    patient_export_df = _build_secondary_patient_export_df(patient_df)
    proportion_summary = secondary_results.get("proportion_summary", {}) if isinstance(secondary_results.get("proportion_summary", {}), dict) else {}
    density_summary = secondary_results.get("density_summary", {}) if isinstance(secondary_results.get("density_summary", {}), dict) else {}
    eligible_mask = patient_df.get("secondary_base_eligible", pd.Series(False, index=patient_df.index)).astype(bool)
    valid_zone_mask = patient_df.get("has_valid_zone_roi", pd.Series(False, index=patient_df.index)).astype(bool)
    present_mask = patient_df.get("triad_present", pd.Series(False, index=patient_df.index)).astype(bool)
    primary_scoreable_mask = patient_df.get("supports_primary_permutation_score", pd.Series(False, index=patient_df.index)).astype(bool)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Patients with valid ROI in range", str(int(valid_zone_mask.sum())))
    metric_cols[1].metric("Patients in continuous secondary Cox", str(int(eligible_mask.sum())))
    metric_cols[2].metric("Zero-burden patients", str(int((eligible_mask & ~present_mask).sum())))
    metric_cols[3].metric("Meet primary Z-score threshold", str(int((eligible_mask & primary_scoreable_mask).sum())))

    st.markdown("#### Continuous Secondary Tests")
    st.dataframe(summary_display_df, width="stretch", hide_index=True)

    explanation_parts = []
    target_triad = secondary_results.get("target_triad") or ()
    if target_triad:
        explanation_parts.append(f"Selected triad: {' + '.join(str(value) for value in target_triad)}")
    explanation_parts.append(f"Raw triad proportion = {_secondary_metric_definition('raw_triad_proportion')}")
    explanation_parts.append(f"Raw triad density = {_secondary_metric_definition('raw_triad_density')}")
    if bool(secondary_results.get("uses_permutation_threshold", False)):
        explanation_parts.append(
            f"Primary scoreability reference = at least one valid ROI with >= {int(_to_float(secondary_results.get('min_obs_motif_count'), 0.0))} observed motifs"
        )
    st.caption(" | ".join(explanation_parts))
    st.caption(
        "Workflow used here: pool valid ROI counts/area to a patient-level burden metric -> retain that metric as continuous -> fit continuous Cox PH. "
        "If you turn on the KM plots below, the High/Low groups are created only for display after the continuous model has already been defined."
    )

    _stateful_checkbox(
        "Show optional grouped KM visualizations",
        key=f"{state_prefix}_secondary_show_km",
        default=False,
        help="Draws grouped Kaplan-Meier curves for the zero-retained raw triad metrics. These curves are for visualization only; the continuous Cox models above are the primary secondary-analysis tests, and the grouping is created only for display.",
    )
    _stateful_checkbox(
        "Show patient distribution dot plots",
        key=f"{state_prefix}_secondary_show_dots",
        default=True,
        help=(
            "One dot per patient behind the high/low split, so you can see the measurement the "
            "grouping is built on rather than only its consequence. Look for: groups that overlap "
            "heavily (a fragile split), a group driven by two or three extreme patients, or a large "
            "block of patients at zero burden -- each of which weakens a KM curve that otherwise "
            "looks convincing."
        ),
    )
    if bool(st.session_state.get(f"{state_prefix}_secondary_show_km", False)):
        proportion_fig = plot_secondary_metric_survival_km(
            patient_df,
            metric_col="raw_triad_proportion",
            metric_label="Raw triad proportion",
            endpoint_label=str(hit_row.get("endpoint_label") or ""),
            config=config,
            summary=proportion_summary,
        )
        _render_figure_with_download(proportion_fig, "Optional grouped KM visualization: raw triad proportion", f"{state_prefix}_secondary_prop_km")
        plt.close(proportion_fig)

        density_fig = plot_secondary_metric_survival_km(
            patient_df,
            metric_col="raw_triad_density",
            metric_label="Raw triad density",
            endpoint_label=str(hit_row.get("endpoint_label") or ""),
            config=config,
            summary=density_summary,
        )
        _render_figure_with_download(density_fig, "Optional grouped KM visualization: raw triad density", f"{state_prefix}_secondary_density_km")
        plt.close(density_fig)

    # Per-patient dot plots. The KM curves above show what happened to the two
    # groups; these show what the two groups *are* -- the spread of the actual
    # measurement behind the split, whether the groups separate or overlap, and
    # how many patients sit at zero burden. A split that looks decisive on a KM
    # curve but overlaps completely here is not a finding to lean on.
    if bool(st.session_state.get(f"{state_prefix}_secondary_show_dots", True)):
        st.markdown("#### Patient distribution behind the split")
        st.caption(
            "One dot per patient, grouped by the same cutpoint used for the survival split. "
            "Filled dots had an event, hollow dots were censored; the bar is the group median "
            "and the dashed line is the cutpoint."
        )
        triad_label = str(hit_row.get("feature_label") or hit_row.get("feature_name") or "")
        zone_text = str(hit_row.get("zone_label") or "")
        endpoint_text = str(hit_row.get("endpoint_label") or "")
        split_method = str((config.get("screening", {}) or {}).get("split_method", "median"))

        dot_columns = st.columns(2)
        for column, (metric_col, metric_label, stub) in zip(
            dot_columns,
            [
                ("raw_triad_proportion", "Raw triad proportion", "prop"),
                ("raw_triad_density", "Raw triad density (per mm2)", "density"),
            ],
        ):
            with column:
                dot_fig, dot_summary = plot_triad_patient_dotplot(
                    patient_df,
                    metric_col=metric_col,
                    metric_label=metric_label,
                    triad_label=triad_label,
                    zone_label=zone_text,
                    endpoint_label=endpoint_text,
                    split_method=split_method,
                )
                _render_figure_with_download(
                    dot_fig,
                    f"Patient distribution: {metric_label}",
                    f"{state_prefix}_secondary_{stub}_dots",
                )
                plt.close(dot_fig)
                if dot_summary.get("status") == "ok":
                    st.caption(
                        f"Low n={dot_summary['n_low']} (median {_format_secondary_stat(dot_summary.get('median_low'))}) | "
                        f"High n={dot_summary['n_high']} (median {_format_secondary_stat(dot_summary.get('median_high'))}) | "
                        f"cutpoint {_format_secondary_stat(dot_summary.get('cutpoint'))} | "
                        f"{int(dot_summary.get('n_zero_value_patients', 0))} patient(s) at zero burden"
                    )

    st.markdown("#### Patient-Level Secondary Analysis Table")
    st.caption("Downloads below use manuscript-facing column titles while preserving the exact patient-level values used in the analysis.")
    st.dataframe(patient_export_df, width="stretch", height=min(380, 80 + 28 * len(patient_export_df)))

    feature_stub = str(hit_row.get("feature_name") or "selected_feature").replace("::", "_").replace("/", "_")
    _render_dataframe_download(
        summary_export_df,
        "Download secondary-analysis summary",
        f"{download_prefix}_{feature_stub}_secondary_summary.csv",
        f"{state_prefix}_secondary_summary_download",
    )
    _render_dataframe_download(
        patient_export_df,
        "Download secondary-analysis patient table",
        f"{download_prefix}_{feature_stub}_secondary_patients.csv",
        f"{state_prefix}_secondary_patient_download",
    )


def _render_validation_secondary_analysis_section(
    validation_results: dict[str, Any],
    hit_row: pd.Series,
) -> None:
    validation_width = float(_to_float(validation_results.get("validation_width"), 0.0))
    zone_start = float(_to_float(validation_results.get("validation_zone_start"), 0.0))
    zone_end = float(_to_float(validation_results.get("validation_zone_end"), validation_width))
    _render_secondary_analysis_section(
        analysis_df=validation_results.get("analysis_df", pd.DataFrame()),
        patient_outcomes_df=validation_results.get("patient_outcomes", pd.DataFrame()),
        config=validation_results.get("config", {}),
        hit_row=hit_row,
        zone_start=zone_start,
        zone_end=zone_end,
        state_prefix="tab8_fs_val",
        download_prefix="validation",
    )


def _render_validation_support_sensitivity_section(
    validation_results: dict[str, Any],
    hit_row: pd.Series,
) -> None:
    st.caption(
        "Sensitivity analysis reruns the selected primary dyad/triad validation while varying min_obs_motif_count. "
        "Use it to see when patient inclusion, hazard ratio direction, and nominal significance stop changing materially."
    )
    st.caption(
        "Recommended interpretation: do not pick the threshold that gives the smallest p-value. "
        "Pick the smallest threshold after which the direction, confidence interval, and patient-retention curves stabilize."
    )

    feature_family = str(hit_row.get("feature_family") or "")
    if feature_family not in {"dyad", "triad"}:
        st.info("Support-threshold sensitivity is currently available only for highlighted dyad or triad features.")
        return
    if str(hit_row.get("endpoint_type") or "") != "survival":
        st.info("Support-threshold sensitivity is currently available only for survival endpoints.")
        return

    config = validation_results.get("config", {})
    screening = config.get("screening", {}) if isinstance(config.get("screening", {}), dict) else {}
    if int(_to_float(screening.get("dyad_triad_permutation_n_iter"), 0.0)) <= 0:
        st.info("Permutation testing is disabled for the current validation settings, so min_obs_motif_count sensitivity is not applicable.")
        return

    current_threshold = int(max(0, _to_float(screening.get("min_obs_motif_count"), 3.0)))
    _stateful_number_input(
        "Maximum min_obs_motif_count to test",
        key="tab8_fs_val_support_sensitivity_max_threshold",
        default=max(4, current_threshold),
        min_value=0,
        max_value=12,
        step=1,
        help="The sensitivity run tests every integer threshold from 0 up to this maximum.",
    )
    max_threshold = int(max(current_threshold, _to_float(st.session_state.get("tab8_fs_val_support_sensitivity_max_threshold"), max(4, current_threshold))))
    threshold_grid = list(range(0, max_threshold + 1))
    zone_start = float(_to_float(validation_results.get("validation_zone_start"), 0.0))
    zone_end = float(_to_float(validation_results.get("validation_zone_end"), validation_results.get("validation_width", 0.0)))
    signature = "|".join(
        [
            str(hit_row.get("feature_name") or ""),
            str(hit_row.get("endpoint_label") or ""),
            str(hit_row.get("stratum_label") or ""),
            f"{zone_start:.6f}",
            f"{zone_end:.6f}",
            str(max_threshold),
            str(current_threshold),
        ]
    )

    if st.button(
        "Run support-threshold sensitivity",
        key="tab8_fs_val_support_sensitivity_run_btn",
        type="primary",
        width="stretch",
        help="Reruns the selected primary dyad/triad validation across the requested min_obs_motif_count thresholds.",
    ):
        with st.spinner("Running support-threshold sensitivity across motif-support thresholds..."):
            sensitivity_results = build_motif_threshold_sensitivity(
                hit_row,
                validation_results.get("analysis_df", pd.DataFrame()),
                validation_results.get("patient_outcomes", pd.DataFrame()),
                config,
                zone_start=zone_start,
                zone_end=zone_end,
                thresholds=threshold_grid,
            )
        sensitivity_results["signature"] = signature
        sensitivity_results["max_threshold"] = int(max_threshold)
        st.session_state["_tab8_fs_val_support_sensitivity"] = sensitivity_results

    stored_results = st.session_state.get("_tab8_fs_val_support_sensitivity")
    if not isinstance(stored_results, dict) or str(stored_results.get("signature") or "") != signature:
        st.info(
            f"Run the support-threshold sensitivity to test thresholds 0 to {max_threshold} for the currently highlighted {feature_family}. "
            f"The configured validation threshold is currently {current_threshold}."
        )
        return

    if str(stored_results.get("status") or "") != "ok":
        st.info(str(stored_results.get("message") or "Support-threshold sensitivity could not be prepared for this hit."))
        return

    sensitivity_df = stored_results.get("sensitivity_df", pd.DataFrame())
    if sensitivity_df.empty:
        st.info("No sensitivity rows were generated for the selected feature.")
        return

    result_available = sensitivity_df.get("result_available", pd.Series(False, index=sensitivity_df.index)).fillna(False).astype(bool)
    finite_p_mask = np.isfinite(pd.to_numeric(sensitivity_df.get("p_value", pd.Series(dtype=float)), errors="coerce"))
    first_valid_threshold = sensitivity_df.loc[result_available, "min_obs_motif_count"].min() if bool(result_available.any()) else float("nan")
    best_p_threshold = sensitivity_df.loc[finite_p_mask, "min_obs_motif_count"].iloc[
        pd.to_numeric(sensitivity_df.loc[finite_p_mask, "p_value"], errors="coerce").astype(float).argmin()
    ] if bool(finite_p_mask.any()) else float("nan")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Configured threshold", str(current_threshold))
    metric_cols[1].metric("Thresholds tested", str(int(len(sensitivity_df))))
    metric_cols[2].metric("First valid test", str(int(first_valid_threshold)) if np.isfinite(first_valid_threshold) else "NA")
    metric_cols[3].metric("Lowest nominal p", str(int(best_p_threshold)) if np.isfinite(best_p_threshold) else "NA")

    sensitivity_fig = plot_motif_threshold_sensitivity(sensitivity_df, hit_row, current_threshold)
    _render_figure_with_download(sensitivity_fig, "Validation support-threshold sensitivity", "tab8_fs_val_support_sensitivity_fig")
    plt.close(sensitivity_fig)

    display_df = sensitivity_df.copy()
    display_df["Feature emitted"] = np.where(display_df.get("feature_emitted", False), "Yes", "No")
    display_df["Valid survival test"] = np.where(display_df.get("result_available", False), "Yes", "No")
    display_df["Direction"] = display_df.get("direction", pd.Series("", index=display_df.index)).astype(str).str.replace("protective", "protective").str.replace("risk", "risk")
    display_df = display_df.rename(
        columns={
            "min_obs_motif_count": "Threshold",
            "n_feature_patients": "Patients with emitted primary motif score",
            "n_missing_endpoint_patients": "Dropped missing endpoint/value",
            "n_split_excluded_patients": "Dropped by High/Low split",
            "n_final_patients": "Patients used in survival test",
            "group_high_n": "High group n",
            "group_low_n": "Low group n",
            "p_value": "Log-rank p-value",
            "hazard_ratio": "Hazard ratio",
            "ci_low": "CI low",
            "ci_high": "CI high",
            "concordance_index": "C-index",
            "ph_test_p": "PH test p-value",
            "cutpoint": "Display cutpoint",
            "significance_label": "Significance label",
        }
    )
    display_columns = [
        column
        for column in [
            "Threshold",
            "Feature emitted",
            "Valid survival test",
            "Patients with emitted primary motif score",
            "Dropped missing endpoint/value",
            "Dropped by High/Low split",
            "Patients used in survival test",
            "High group n",
            "Low group n",
            "Hazard ratio",
            "CI low",
            "CI high",
            "Log-rank p-value",
            "C-index",
            "PH test p-value",
            "Display cutpoint",
            "Direction",
            "Significance label",
            "error",
        ]
        if column in display_df.columns
    ]
    st.markdown("#### Threshold Sensitivity Table")
    st.dataframe(display_df[display_columns], width="stretch", hide_index=True)
    _render_dataframe_download(
        display_df[display_columns],
        "Download threshold sensitivity table",
        f"validation_{str(hit_row.get('feature_name') or 'selected_feature').replace('::', '_').replace('/', '_')}_min_obs_sensitivity.csv",
        "tab8_fs_val_support_sensitivity_download",
    )


def _render_validation_results(
    raw_df: pd.DataFrame,
    validation_results: dict[str, Any],
) -> None:
    config = validation_results.get("config", {})
    analysis_df = validation_results.get("analysis_df", pd.DataFrame())
    combined = validation_results.get("combined", pd.DataFrame())
    stats_all = validation_results.get("stats_all", pd.DataFrame())
    patient_features = validation_results.get("patient_features", pd.DataFrame())
    patient_outcomes = validation_results.get("patient_outcomes", pd.DataFrame())
    validation_width = float(validation_results.get("validation_width", 0.0))
    validation_zone_label = str(validation_results.get("validation_zone_label") or format_zone_label(validation_width))
    effective_phenotypes = _effective_screening_phenotypes(raw_df, config)
    excluded_phenotypes = [str(value) for value in config.get("excluded_phenotypes", []) if str(value)]

    st.markdown("#### Validation Results")
    st.caption(
        f"Fixed range: {validation_zone_label} | "
        f"Phenotyping column: {str(config.get('columns', {}).get('phenotype') or '')} | "
        f"Families: {', '.join(str(value) for value in config.get('feature_families', [])) or '-'}"
    )
    st.caption(
        f"Using the applied discovery phenotype setup: {len(effective_phenotypes)} included"
        + (f", {len(excluded_phenotypes)} excluded" if excluded_phenotypes else "")
        + "."
    )

    if stats_all.empty:
        st.warning(
            "Validation finished, but no endpoint-level statistics could be computed at this width. "
            "That usually means the selected subset did not leave enough patients per group for the enabled endpoints."
        )
        if not patient_features.empty:
            preview_df = patient_features.sort_values(["feature_family", "feature_label"], na_position="last").copy()
            st.dataframe(preview_df, width="stretch", height=min(420, 40 + 35 * (len(preview_df.head(12)) + 1)))
            _render_dataframe_download(
                preview_df,
                "Download patient-level validation features",
                "validation_patient_features.csv",
                "tab8_fs_val_patient_features_download",
            )
        return

    ranked = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").reset_index(drop=True).copy()
    ranked["overall_rank"] = np.arange(1, len(ranked) + 1)
    ranked["family_rank"] = ranked.groupby(["endpoint_label", "stratum_label", "feature_family"], dropna=False).cumcount() + 1
    ranked["nominal_hit"] = pd.to_numeric(ranked.get("p_value", pd.Series(np.nan, index=ranked.index)), errors="coerce") <= 0.05
    ranked["Status"] = ranked["significance_label"].map(_significance_display_label)

    endpoint_options = sorted(ranked["endpoint_label"].astype(str).unique().tolist())
    _stateful_selectbox(
        "Validation endpoint",
        options=endpoint_options,
        key="tab8_fs_val_endpoint",
        default=endpoint_options[0] if endpoint_options else None,
    )
    selected_endpoint = st.session_state.get("tab8_fs_val_endpoint", endpoint_options[0] if endpoint_options else "")

    stratum_options = sorted(
        ranked.loc[ranked["endpoint_label"].astype(str) == str(selected_endpoint), "stratum_label"].astype(str).unique().tolist()
    )
    _stateful_selectbox(
        "Validation stratum",
        options=stratum_options,
        key="tab8_fs_val_stratum",
        default=stratum_options[0] if stratum_options else None,
    )
    selected_stratum = st.session_state.get("tab8_fs_val_stratum", stratum_options[0] if stratum_options else "")

    endpoint_df = ranked[
        (ranked["endpoint_label"].astype(str) == str(selected_endpoint))
        & (ranked["stratum_label"].astype(str) == str(selected_stratum))
    ].copy()
    if endpoint_df.empty:
        st.info("No validation rows are available for the selected endpoint and stratum.")
        return

    family_options = ["All"] + sorted(endpoint_df["feature_family"].astype(str).unique().tolist())
    _stateful_selectbox(
        "Family filter",
        options=family_options,
        key="tab8_fs_val_family_filter",
        default=family_options[0] if family_options else None,
    )
    selected_family = st.session_state.get("tab8_fs_val_family_filter", "All")
    family_df = endpoint_df.copy() if selected_family == "All" else endpoint_df[endpoint_df["feature_family"].astype(str) == str(selected_family)].copy()
    if family_df.empty:
        st.info("No rows remain after the selected family filter.")
        return

    focus_elements = [str(value) for value in st.session_state.get("tab8_fs_val_focus_elements", []) if str(value)]
    focus_require_all = bool(st.session_state.get("tab8_fs_val_focus_require_all", True))
    focus_mask = family_df["feature_name"].astype(str).map(
        lambda value: _feature_matches_focus_elements(value, focus_elements, focus_require_all)
    )
    focused_df = family_df.loc[focus_mask].copy() if focus_elements else family_df.copy()
    matched_feature_names = focused_df["feature_name"].astype(str).tolist() if focus_elements else []
    if focus_elements and focused_df.empty:
        st.info(
            "No validated rows matched the current phenotype-element highlight selection. "
            "The validation run still used all applied discovery settings; broaden the element selection or turn off 'Require all selected elements' to highlight more rows."
        )
    highlight_options = focused_df.apply(
        lambda row: f"#{int(row['overall_rank'])} | {str(row['feature_label'])} | {str(row['Status'])}",
        axis=1,
    ).tolist()
    highlight_lookup = dict(zip(highlight_options, focused_df.index.tolist()))
    _stateful_selectbox(
        "Highlighted feature",
        options=highlight_options,
        key="tab8_fs_val_highlight_feature",
        default=highlight_options[0] if highlight_options else None,
    )
    selected_highlight = st.session_state.get("tab8_fs_val_highlight_feature", highlight_options[0] if highlight_options else "")
    hit_row = focused_df.loc[int(highlight_lookup.get(selected_highlight, focused_df.index[0]))] if highlight_options else family_df.iloc[0]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Tested features", str(int(len(family_df))))
    metric_cols[1].metric("Nominal p < 0.05", str(int(family_df["nominal_hit"].sum())))
    metric_cols[2].metric("FDR discovery", str(int(family_df["significance_label"].isin(["significant_fdr", "discovery_fdr"]).sum())))
    metric_cols[3].metric("FDR significant", str(int((family_df["significance_label"] == "significant_fdr").sum())))
    metric_cols[4].metric("Highlighted matches", str(int(len(focused_df))) if focus_elements else str(int(len(family_df))))

    results_tabs = st.tabs(["Ranked Results", "Summary Figures", "Highlighted Feature", "Secondary Analysis", "Support Sensitivity", "Publication Figures", "Influence Correlation", "Representative ROIs"])
    with results_tabs[0]:
        top_n = max(5, int(st.session_state.get("tab8_fs_val_top_n", 25)))
        display_df = family_df.copy()
        display_df["Nominal"] = np.where(display_df["nominal_hit"], "p < 0.05", "p >= 0.05")
        display_df["Direction"] = [
            _direction_text(direction, endpoint_type, hazard_ratio)
            for direction, endpoint_type, hazard_ratio in zip(
                display_df.get("direction", pd.Series("", index=display_df.index)),
                display_df.get("endpoint_type", pd.Series("", index=display_df.index)),
                display_df.get("hazard_ratio", pd.Series(np.nan, index=display_df.index)),
            )
        ]
        display_columns = [
            column
            for column in [
                "overall_rank",
                "family_rank",
                "feature_family",
                "feature_label",
                "Nominal",
                "Status",
                "p_value",
                "fdr_q",
                "Direction",
                "effect_size",
                "hazard_ratio",
            ]
            if column in display_df.columns
        ]
        st.dataframe(display_df[display_columns].head(top_n), width="stretch", height=min(420, 40 + 35 * (min(len(display_df), top_n) + 1)))
        _render_dataframe_download(
            display_df,
            "Download validation ranking",
            "validation_ranked_results.csv",
            "tab8_fs_val_results_download",
        )
        if focus_elements and not focused_df.empty:
            st.markdown("#### Highlighted Matches")
            focused_display = focused_df[[column for column in display_columns if column in focused_df.columns]].copy()
            st.dataframe(focused_display, width="stretch", hide_index=True)

    with results_tabs[1]:
        volcano = plot_volcano(
            ranked,
            selected_endpoint,
            validation_width,
            selected_stratum,
            highlight_feature_names=matched_feature_names,
            highlight_feature_name=str(hit_row.get("feature_name") or ""),
            highlight_feature_label=str(hit_row.get("feature_label") or ""),
            label_settings=_volcano_label_settings(config),
        )
        _render_figure_with_download(volcano, "Validation volcano plot", "tab8_fs_val_volcano")
        plt.close(volcano)

        forest = plot_forest(ranked, selected_endpoint, validation_width, selected_stratum)
        _render_figure_with_download(forest, "Validation forest plot", "tab8_fs_val_forest")
        plt.close(forest)

        family_fig = plot_family_summary(ranked, selected_endpoint, validation_width, selected_stratum)
        _render_figure_with_download(family_fig, "Validation family summary", "tab8_fs_val_family_summary")
        plt.close(family_fig)

    with results_tabs[2]:
        hit_overall_rank = int(pd.to_numeric(pd.Series([hit_row.get("overall_rank")]), errors="coerce").fillna(1).iloc[0])
        hit_family_rank = int(pd.to_numeric(pd.Series([hit_row.get("family_rank")]), errors="coerce").fillna(1).iloc[0])
        p_value = _to_float(hit_row.get("p_value"), float("nan"))
        fdr_q = _to_float(hit_row.get("fdr_q"), float("nan"))
        p_value_text = f"{p_value:.4g}" if np.isfinite(p_value) else "nan"
        fdr_q_text = f"{fdr_q:.4g}" if np.isfinite(fdr_q) else "nan"
        st.caption(
            f"Highlighted feature: {str(hit_row.get('feature_label') or hit_row.get('feature_name') or '-')} | "
            f"Overall rank: #{hit_overall_rank} | Family rank: #{hit_family_rank} | "
            f"Nominal p: {p_value_text} | "
            f"Status: {_significance_display_label(hit_row.get('significance_label'))} | "
            f"FDR q: {fdr_q_text}"
        )
        hit_fig = plot_feature_endpoint(hit_row, patient_features, patient_outcomes, config)
        _render_figure_with_download(hit_fig, "Validation highlighted feature", "tab8_fs_val_selected_hit")
        plt.close(hit_fig)
        if str(hit_row.get("endpoint_type", "")) == "binary":
            waterfall_fig = plot_feature_waterfall(hit_row, patient_features, patient_outcomes, config)
            _render_figure_with_download(waterfall_fig, "Validation waterfall plot", "tab8_fs_val_waterfall")
            plt.close(waterfall_fig)

        patient_export_df = build_feature_endpoint_export_df(hit_row, patient_features, patient_outcomes, config)
        patient_audit_df = build_feature_endpoint_audit_df(
            hit_row,
            analysis_df,
            patient_features,
            patient_outcomes,
            config,
            zone_start=float(_to_float(validation_results.get("validation_zone_start"), 0.0)),
            zone_end=float(_to_float(validation_results.get("validation_zone_end"), validation_width)),
        )
        if not patient_export_df.empty:
            st.dataframe(patient_export_df, width="stretch", height=260)
            _render_dataframe_download(
                patient_export_df,
                "Download patient CSV for highlighted feature",
                "validation_highlighted_feature_patients.csv",
                "tab8_fs_val_selected_hit_patient_csv_download",
            )
        _render_patient_inclusion_audit(
            patient_audit_df,
            "Download patient inclusion audit",
            "validation_highlighted_feature_patient_audit.csv",
            "tab8_fs_val_selected_hit_patient_audit_download",
        )

        _render_selected_hit_motif_export_section(
            analysis_df=analysis_df,
            stats_all=stats_all,
            config=config,
            hit_row=hit_row,
            section_label="Validation",
            zone_start=float(_to_float(validation_results.get("validation_zone_start"), 0.0)),
            zone_end=float(_to_float(validation_results.get("validation_zone_end"), validation_width)),
            key_prefix="tab8_fs_val_selected_hit",
            selected_roi_features_df=combined,
            selected_patient_export_df=patient_export_df,
            selected_patient_features_df=patient_features,
        )

    with results_tabs[3]:
        with _timed_section("validation:secondary_analysis"):
            _render_validation_secondary_analysis_section(validation_results, hit_row)

    with results_tabs[4]:
        with _timed_section("validation:support_sensitivity"):
            _render_validation_support_sensitivity_section(validation_results, hit_row)

    with results_tabs[5]:
        live_discovery_results = st.session_state.get("_tab8_fs_results")
        validation_candidate_summary = live_discovery_results.get("candidate_summary", pd.DataFrame()) if isinstance(live_discovery_results, dict) else pd.DataFrame()
        _render_publication_figures_section(
            analysis_df=analysis_df,
            combined=combined,
            patient_features=patient_features,
            patient_outcomes=patient_outcomes,
            patient_export_df=patient_export_df,
            stats_all=stats_all,
            candidate_summary=validation_candidate_summary,
            config=config,
            hit_row=hit_row,
            section_label="Validation",
            zone_start=float(_to_float(validation_results.get("validation_zone_start"), 0.0)),
            zone_end=float(_to_float(validation_results.get("validation_zone_end"), validation_width)),
            selected_candidate_width=_to_float(live_discovery_results.get("selected_width"), float("nan")) if isinstance(live_discovery_results, dict) else float("nan"),
            key_prefix="tab8_fs_val_publication",
        )

    with results_tabs[6]:
        _render_validation_influence_section(
            raw_df=raw_df,
            validation_results=validation_results,
            hit_row=hit_row,
            selected_endpoint=selected_endpoint,
            selected_stratum=selected_stratum,
        )

    with results_tabs[7]:
        from .roi_explorer import _render_roi_explorer
        with _timed_section("validation:roi_explorer"):
            _render_roi_explorer(raw_df, config, state_prefix="tab8_fs_val_roi")


def _render_validation_panel(raw_df: pd.DataFrame, base_config: dict[str, Any], unified_csv_path: str | None) -> None:
    st.markdown("### Targeted Validation")
    st.caption(
        "Validate one fixed distance without running the full candidate scan. This panel reuses the applied discovery phenotype column, phenotype selection, exclusions, feature families, endpoints, and other shared settings."
    )
    st.caption(
        "To change phenotype column, phenotype inclusion/exclusion, or feature families, edit the shared controls above and Apply. Validation here is only for the range and optional highlight focus."
    )

    suggested_width = _suggest_validation_width(base_config)
    current_phenotype_col = str(base_config.get("columns", {}).get("phenotype") or "")
    current_feature_families = [str(value) for value in base_config.get("feature_families", []) if str(value)]
    effective_phenotypes = _effective_screening_phenotypes(raw_df, base_config)
    excluded_phenotypes = [str(value) for value in base_config.get("excluded_phenotypes", []) if str(value)]
    discovery_results = st.session_state.get("_tab8_fs_results")
    selected_hit_row = _selected_discovery_hit_row(discovery_results) if isinstance(discovery_results, dict) else None

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        if st.button(
            "Use latest discovery range",
            key="tab8_fs_val_copy_hit_width_btn",
            help="Copies the currently selected discovery range into the validation distance box when a discovery result is available.",
            disabled=not np.isfinite(suggested_width),
            width="stretch",
        ):
            st.session_state["tab8_fs_val_zone_start"] = 0.0
            st.session_state["tab8_fs_val_zone_end"] = float(suggested_width)
            st.session_state["tab8_fs_val_zone_width"] = float(suggested_width)
            st.rerun()
        if st.button(
            "Import selected discovery hit",
            key="tab8_fs_val_import_hit_btn",
            help="Copies the currently selected discovery hit into validation so endpoint, stratum, family, and distance match the discovery result.",
            disabled=selected_hit_row is None,
            width="stretch",
        ):
            if selected_hit_row is not None:
                imported_width = _to_float(selected_hit_row.get("zone_width"), suggested_width)
                imported_family = str(selected_hit_row.get("feature_family") or "")
                imported_name = str(selected_hit_row.get("feature_name") or "")
                _, imported_components = parse_feature_components(imported_name)
                st.session_state["tab8_fs_val_zone_start"] = 0.0
                st.session_state["tab8_fs_val_zone_end"] = float(imported_width)
                st.session_state["tab8_fs_val_zone_width"] = float(imported_width)
                st.session_state["tab8_fs_val_endpoint"] = str(selected_hit_row.get("endpoint_label") or "")
                st.session_state["tab8_fs_val_stratum"] = str(selected_hit_row.get("stratum_label") or "")
                st.session_state["tab8_fs_val_family_filter"] = imported_family or "All"
                st.session_state["tab8_fs_val_focus_elements"] = list(imported_components)
                st.session_state["tab8_fs_val_focus_require_all"] = True
                st.session_state["tab8_fs_val_highlight_feature"] = ""
                st.rerun()
        _stateful_number_input(
            "Validation range start",
            key="tab8_fs_val_zone_start",
            default=0.0,
            step=max(abs(float(suggested_width)) / 20.0, 1.0),
            help="Inclusive start of the validation interval. Discovery-style validation uses 0 here.",
        )
        _stateful_number_input(
            "Validation range end",
            key="tab8_fs_val_zone_end",
            default=float(st.session_state.get("tab8_fs_val_zone_width", suggested_width)),
            step=max(abs(float(suggested_width)) / 20.0, 1.0),
            help="Exclusive end of the validation interval. To reproduce a discovery width like 0 to 305, use start = 0 and end = 305.",
        )
    with control_col2:
        _stateful_number_input(
            "Top rows to display",
            key="tab8_fs_val_top_n",
            default=25,
            min_value=5,
            max_value=200,
            step=5,
            help="How many top-ranked rows to show in the validation results table.",
        )
        _stateful_multiselect(
            "Phenotype elements to highlight (optional)",
            options=effective_phenotypes,
            key="tab8_fs_val_focus_elements",
            default=st.session_state.get("tab8_fs_val_focus_elements", []),
            help="Does not change the validation computation. It only highlights validated features that contain these phenotype elements.",
        )
        _stateful_checkbox(
            "Require all selected elements",
            key="tab8_fs_val_focus_require_all",
            default=True,
            help="When ON, a validated feature must contain every selected phenotype element to be highlighted. When OFF, any overlap is enough.",
        )
    with control_col3:
        st.info(
            f"Applied phenotype column: {current_phenotype_col or '-'}\n\n"
            f"Included phenotypes: {len(effective_phenotypes)}\n"
            f"Excluded phenotypes: {len(excluded_phenotypes)}\n"
            f"Feature families: {', '.join(current_feature_families) or '-'}\n\n"
            "Validation inherits the applied discovery settings above. Change them there, click Apply, then rerun validation here."
        )

    validation_range_start = _to_float(st.session_state.get("tab8_fs_val_zone_start"), 0.0)
    validation_range_end = _to_float(st.session_state.get("tab8_fs_val_zone_end"), float(suggested_width))
    st.caption(
        f"Validation interval currently set to {validation_range_start:.1f} to {validation_range_end:.1f}. "
        "Discovery widths are zero-anchored, so a discovery hit reported at 0 to 305 should be validated here with start = 0 and end = 305."
    )
    if st.session_state.get("tab8_fs_val_focus_elements", []):
        st.caption(
            "Highlight focus does not restrict the validation run. It only marks matching validated features in the results using the same applied discovery settings."
        )

    _pheno_col_options = [c for c in _phenotype_column_options(raw_df) if c != current_phenotype_col]
    if _pheno_col_options:
        _stateful_multiselect(
            "Also validate with these phenotype columns",
            options=_pheno_col_options,
            key="tab8_fs_val_extra_phenotype_cols",
            default=[],
            help=(
                "Re-run this same validation independently for each additional phenotyping method, "
                "so you can see whether a finding survives a change of phenotyping.\n\n"
                "Each method is assessed entirely on its own: the phenotype selection is re-derived "
                "from the labels that column actually contains, and each method keeps its own FDR. "
                "No correction is applied across methods -- these are separate assessments of the "
                "same question, not one combined test."
            ),
        )
        st.caption(
            f"Validation will run for **{current_phenotype_col}** (from discovery) plus "
            f"**{len(st.session_state.get('tab8_fs_val_extra_phenotype_cols', []) or [])}** additional method(s)."
        )

    if st.button(
        "Run targeted validation",
        key="tab8_fs_val_run_btn",
        type="primary",
        help="Runs the fixed-width validation immediately using the current settings in this panel.",
    ):
        validation_config = _build_validation_config(raw_df, base_config, unified_csv_path)
        validation_start = _to_float(st.session_state.get("tab8_fs_val_zone_start"), float("nan"))
        validation_end = _to_float(st.session_state.get("tab8_fs_val_zone_end"), float("nan"))
        st.session_state["tab8_fs_val_zone_width"] = float(validation_end) if np.isfinite(validation_end) else st.session_state.get("tab8_fs_val_zone_width", suggested_width)
        errors = _validate_config(validation_config, raw_df)
        if not np.isfinite(validation_start) or not np.isfinite(validation_end) or float(validation_start) == float(validation_end):
            errors.append("Validation range start and end must both be finite and cannot be equal.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            extra_cols = [
                str(c) for c in (st.session_state.get("tab8_fs_val_extra_phenotype_cols", []) or [])
                if str(c) and str(c) != current_phenotype_col and str(c) in raw_df.columns
            ]
            columns_to_run = [current_phenotype_col] + extra_cols
            results_by_column: dict[str, Any] = {}
            failures: dict[str, str] = {}
            progress = st.progress(0.0, text="Validating...")
            for index, phenotype_column in enumerate(columns_to_run, start=1):
                progress.progress(
                    (index - 1) / max(len(columns_to_run), 1),
                    text=f"Validating with {phenotype_column} ({index}/{len(columns_to_run)})",
                )
                try:
                    column_config = (
                        validation_config if phenotype_column == current_phenotype_col
                        else _config_for_phenotype_column(validation_config, raw_df, phenotype_column)
                    )
                    results_by_column[phenotype_column] = run_feature_validation_pipeline(
                        raw_df,
                        column_config,
                        float(validation_end),
                        zone_start=float(validation_start),
                        zone_end=float(validation_end),
                        log_fn=_log_terminal_event,
                    )
                except Exception as exc:
                    failures[phenotype_column] = str(exc)
            progress.empty()

            range_label = format_zone_interval_label(validation_start, validation_end)
            if current_phenotype_col in results_by_column:
                # Keep the primary column under the original key so every existing
                # downstream panel keeps working exactly as before.
                st.session_state["_tab8_fs_validation_results"] = results_by_column[current_phenotype_col]
            st.session_state["_tab8_fs_validation_results_by_column"] = results_by_column

            if results_by_column:
                st.success(
                    f"Validation completed for range {range_label} across "
                    f"{len(results_by_column)} phenotyping method(s): {', '.join(results_by_column)}."
                )
            for phenotype_column, message in failures.items():
                st.error(f"Validation failed for phenotype column '{phenotype_column}': {message}")

    results_by_column = st.session_state.get("_tab8_fs_validation_results_by_column") or {}
    validation_results = st.session_state.get("_tab8_fs_validation_results")

    if len(results_by_column) > 1:
        st.markdown("#### Cross-method comparison")
        st.caption(
            "The same feature, range and endpoint assessed independently under each phenotyping "
            "method. Each row carries that method's own FDR; nothing is corrected across methods. "
            "A finding that holds its direction and significance across methods is robust to the "
            "phenotyping choice -- one that only appears under a single method is not."
        )
        comparison_df = _build_multi_phenotype_comparison_df(results_by_column)
        if not comparison_df.empty:
            sig_only = st.checkbox(
                "Show only rows significant under at least one method",
                value=True,
                key="tab8_fs_val_multi_sig_only",
            )
            table = comparison_df
            if sig_only and "significance_label" in comparison_df.columns:
                hit_keys = set(
                    comparison_df.loc[
                        comparison_df["significance_label"].isin(["significant_fdr", "discovery_fdr"]),
                        "feature_name",
                    ].astype(str)
                )
                if hit_keys:
                    table = comparison_df[comparison_df["feature_name"].astype(str).isin(hit_keys)]
            st.dataframe(table, width="stretch", hide_index=True)
            _render_dataframe_download(
                comparison_df,
                "Download cross-method comparison (CSV)",
                "validation_cross_method_comparison.csv",
                key="tab8_fs_val_multi_cmp_download",
            )

            # Concordance: for each feature/endpoint, how many methods agree it is a hit.
            if {"feature_name", "endpoint_label", "significance_label"}.issubset(comparison_df.columns):
                hits = comparison_df.copy()
                hits["is_hit"] = hits["significance_label"].isin(["significant_fdr", "discovery_fdr"])
                concordance = (
                    hits.groupby(["feature_label", "endpoint_label"], dropna=False)
                    .agg(
                        methods_tested=("phenotype_column", "nunique"),
                        methods_significant=("is_hit", "sum"),
                        directions=("direction", lambda v: ", ".join(sorted({str(x) for x in v if str(x) and str(x) != "nan"}))),
                        best_q=("fdr_q", "min"),
                    )
                    .reset_index()
                    .sort_values(["methods_significant", "best_q"], ascending=[False, True])
                )
                concordance["direction_consistent"] = concordance["directions"].astype(str).str.count(",").eq(0)
                st.markdown("**Concordance across methods**")
                st.dataframe(concordance, width="stretch", hide_index=True)
        else:
            st.info("No comparable statistics were produced across the selected phenotype columns.")

        st.markdown("#### Full results for one method")
        chosen = st.selectbox(
            "Phenotyping method to inspect",
            options=list(results_by_column),
            key="tab8_fs_val_multi_selected_column",
            help=(
                "Figures and statistics below are rendered for this method only. Switch method to "
                "inspect another; the comparison table above always covers every method that ran."
            ),
        )
        validation_results = results_by_column.get(chosen, validation_results)

    if validation_results:
        with _timed_section("validation:results"):
            _render_validation_results(raw_df, validation_results)


def _build_zone_selection_rationale_text(
    candidate_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    selected_width: float,
) -> str:
    """Return a markdown narrative explaining why the selected zone won."""
    if candidate_summary.empty:
        return "_No candidate data available._"

    ranked = candidate_summary.sort_values("candidate_rank", na_position="last").reset_index(drop=True)
    sel_mask = ranked["zone_width"].astype(float).round(2) == round(float(selected_width), 2)
    if not bool(sel_mask.any()):
        return "_Selected zone not found in candidate summary._"

    sel = ranked[sel_mask].iloc[0]
    others = ranked[~sel_mask].reset_index(drop=True)

    sel_label     = str(sel.get("zone_label") or format_zone_label(selected_width))
    n_sig         = int(_to_float(sel.get("n_significant"), 0.0))
    n_disc        = int(_to_float(sel.get("n_discovery"), 0.0))
    boot          = _to_float(sel.get("bootstrap_reference_recovery"))
    boot_iters    = _to_float(sel.get("bootstrap_valid_iterations"))
    coverage      = _to_float(sel.get("patient_coverage_fraction"))
    feat_stab     = _to_float(sel.get("feature_stability_score"))
    feat_stab_hits= int(_to_float(sel.get("feature_stability_band_hits"), 0.0))
    top_feat      = str(sel.get("feature_stability_top_feature_label") or "")
    prior_score   = _to_float(sel.get("candidate_priority_score"))
    repro_score   = _to_float(sel.get("candidate_reproducibility_score"))
    sig_score     = _to_float(sel.get("candidate_significance_score"))

    lines: list[str] = []
    lines.append(f"**Selected zone: {sel_label}**\n")
    lines.append("**Metrics for the selected zone:**")
    lines.append(f"- FDR significant hits: **{n_sig}**  |  FDR discovery hits: **{n_disc}**")
    if np.isfinite(boot):
        iters_str = f"{int(boot_iters)} iterations" if np.isfinite(boot_iters) else "N iterations"
        lines.append(
            f"- Bootstrap recovery: **{float(boot):.2f}** — the FDR hits were independently "
            f"rediscovered in {float(boot)*100:.0f}% of sub-cohort resamples ({iters_str}, 80% of patients per resample)."
        )
    else:
        lines.append("- Bootstrap recovery: **not computed** (zone had no FDR hits at scan time, so it was not bootstrapped).")
    if np.isfinite(coverage):
        lines.append(f"- Patient coverage: **{float(coverage)*100:.1f}%** of the cohort had cells in this zone.")
    if np.isfinite(feat_stab) and float(feat_stab) > 0:
        lines.append(
            f"- Feature stability score: **{float(feat_stab):.2f}** — "
            f"{feat_stab_hits} feature(s) showed stable significance across neighbouring zone widths, "
            f"anchoring this zone as their optimal range."
        )
        if top_feat:
            lines.append(f"- Strongest stable feature anchoring this zone: **{top_feat}**")
    lines.append(
        f"- Combined priority score: **{float(prior_score):.3f}** "
        f"(reproducibility: {float(repro_score):.3f}, significance: {float(sig_score):.3f})"
        if np.isfinite(prior_score) else "- Combined priority score: N/A"
    )
    lines.append("")

    # ── Runner-up comparison ──────────────────────────────────────────────────
    if not others.empty:
        ru = others.iloc[0]
        ru_label   = str(ru.get("zone_label") or format_zone_label(_to_float(ru.get("zone_width"))))
        ru_n_sig   = int(_to_float(ru.get("n_significant"), 0.0))
        ru_n_disc  = int(_to_float(ru.get("n_discovery"), 0.0))
        ru_boot    = _to_float(ru.get("bootstrap_reference_recovery"))
        ru_feat    = _to_float(ru.get("feature_stability_score"))
        ru_score   = _to_float(ru.get("candidate_priority_score"))

        lines.append(f"**Why not zone {ru_label}?** (2nd-ranked candidate)")
        lines.append(f"- FDR significant hits: {ru_n_sig}  |  FDR discovery hits: {ru_n_disc}")
        if np.isfinite(ru_boot):
            lines.append(f"- Bootstrap recovery: {float(ru_boot):.2f}")
        if np.isfinite(ru_feat):
            lines.append(f"- Feature stability score: {float(ru_feat):.2f}")
        if np.isfinite(ru_score):
            lines.append(f"- Combined priority score: {float(ru_score):.3f}")

        # ── Decisive-factor sentence ──────────────────────────────────────────
        decisive: list[str] = []
        boot_higher_ru = np.isfinite(ru_boot) and np.isfinite(boot) and float(ru_boot) > float(boot)
        boot_diff_abs  = abs(float(ru_boot) - float(boot)) if (np.isfinite(ru_boot) and np.isfinite(boot)) else 0.0

        if boot_higher_ru and boot_diff_abs > 0.02:
            # Runner-up had higher bootstrap — explain what outweighed it
            advantages: list[str] = []
            if np.isfinite(feat_stab) and np.isfinite(ru_feat) and float(feat_stab) > float(ru_feat) + 0.1:
                advantages.append(
                    f"higher feature stability ({float(feat_stab):.2f} vs {float(ru_feat):.2f}, "
                    f"weight ×5.0 in scoring)"
                )
            if n_sig > ru_n_sig:
                advantages.append(f"more FDR hits ({n_sig} vs {ru_n_sig}, weight ×2.0)")
            if n_disc > ru_n_disc:
                advantages.append(f"more discovery hits ({n_disc} vs {ru_n_disc}, weight ×1.5)")
            if advantages:
                decisive.append(
                    f"Although zone {ru_label} had a slightly higher bootstrap recovery "
                    f"({float(ru_boot):.2f} vs {float(boot):.2f}), the selected zone prevailed "
                    f"due to its {', '.join(advantages)}. "
                    f"All metrics are converted to 0–1 rank scores before weighting, so a small "
                    f"bootstrap gap (~{boot_diff_abs:.2f}) can be outweighed by larger advantages "
                    f"elsewhere in the combined score."
                )
            else:
                decisive.append(
                    f"Zone {ru_label} had a marginally higher bootstrap recovery "
                    f"({float(ru_boot):.2f} vs {float(boot):.2f}) but lower combined priority "
                    f"score ({float(ru_score):.3f} vs {float(prior_score):.3f})."
                )
        elif np.isfinite(boot) and not np.isfinite(ru_boot):
            decisive.append(
                f"Zone {ru_label} was not bootstrapped (no FDR hits at scan time). "
                f"The selected zone's bootstrap recovery of {float(boot):.2f} gave it a strong "
                f"reproducibility advantage."
            )
        elif np.isfinite(feat_stab) and np.isfinite(ru_feat) and float(feat_stab) > float(ru_feat) + 0.1:
            decisive.append(
                f"The selected zone's higher feature stability score ({float(feat_stab):.2f} vs "
                f"{float(ru_feat):.2f}) was the decisive factor."
            )
        elif n_sig > ru_n_sig:
            decisive.append(
                f"The selected zone's higher FDR hit count ({n_sig} vs {ru_n_sig}) was the decisive factor."
            )

        if decisive:
            lines.append(f"- **Why the selected zone won**: {decisive[0]}")
        lines.append("")

    # ── Scoring methodology note ──────────────────────────────────────────────
    lines.append("**How the scoring works (weights):**")
    lines.append(
        "Every metric is first converted to a 0–1 rank score across all scanned zones, "
        "then multiplied by a weight and summed into two sub-scores:"
    )
    lines.append(
        "- *Reproducibility* (max ~26.5 pts): bootstrap hit recovery ×**8.0** · "
        "feature stability ×**5.0** + ×2.5 · coverage ×1.5 · bootstrap validity ×2.5 · "
        "stability hits ×1.5+1.0 · bootstrap mean FDR ×1.5 · bootstrap mean discovery ×1.0 · ROI coverage ×0.5"
    )
    lines.append(
        "- *Significance* (max ~4.6 pts): FDR hits ×**2.0** · discovery hits ×**1.5** · "
        "best FDR q ×0.75 · best p ×0.35"
    )
    lines.append(
        "Reproducibility outweighs significance ~6:1. On exact ties the narrower zone is preferred."
    )

    return "\n".join(lines)


def _build_fdr_hits_summary_df(
    stats_all: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    combined_by_width: dict[float, pd.DataFrame],
    selected_width: float,
) -> pd.DataFrame:
    """Build an enriched summary table of every FDR-significant hit."""
    if stats_all.empty:
        return pd.DataFrame()

    hits = stats_all[stats_all["significance_label"] == "significant_fdr"].copy()
    if hits.empty:
        return pd.DataFrame()

    # ── Bootstrap recovery lookup keyed by zone_width ────────────────────────
    boot_lookup: dict[float, float] = {}
    if not candidate_summary.empty and "bootstrap_reference_recovery" in candidate_summary.columns:
        for _, crow in candidate_summary.iterrows():
            w = _to_float(crow.get("zone_width"))
            r = _to_float(crow.get("bootstrap_reference_recovery"))
            if np.isfinite(w):
                boot_lookup[round(float(w), 2)] = float(r) if np.isfinite(r) else float("nan")

    # ── Feature-level stability lookup keyed by (endpoint, feature_name, stratum) ─
    feat_info_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not feature_summary.empty:
        for _, frow in feature_summary.iterrows():
            key = (
                str(frow.get("endpoint_label") or ""),
                str(frow.get("feature_name") or ""),
                str(frow.get("stratum_label") or ""),
            )
            feat_info_lookup[key] = {
                "feature_rank": int(_to_float(frow.get("feature_rank"), 9999.0)),
                "feature_bootstrap_recovery": _to_float(frow.get("feature_bootstrap_recovery")),
                "n_significant_ranges": int(_to_float(frow.get("n_significant_ranges"), 0.0)),
                "significant_band_size": int(_to_float(frow.get("significant_band_size"), 0.0)),
                "significant_band_label": str(frow.get("significant_band_label") or ""),
                "direction_consistency": str(frow.get("direction_consistency") or ""),
                "best_zone_label": str(frow.get("best_zone_label") or ""),
            }

    # ── Representative ROI lookup (top/bottom feature value per ROI at the hit zone) ─
    def _rep_roi(width: float, feat_name: str) -> tuple[str, str]:
        """Return (high_roi_label, low_roi_label) for a feature at a given zone width."""
        try:
            comb = combined_by_width.get(round(float(width), 2)) or combined_by_width.get(float(width))
            if comb is None or comb.empty:
                return "-", "-"
            sub = comb[comb["feature_name"].astype(str) == str(feat_name)].copy()
            if sub.empty:
                return "-", "-"
            sub["_v"] = pd.to_numeric(sub["value"], errors="coerce")
            sub = sub.dropna(subset=["_v"])
            if sub.empty:
                return "-", "-"
            top = sub.loc[sub["_v"].idxmax()]
            bot = sub.loc[sub["_v"].idxmin()]
            def _roi_label(r: pd.Series) -> str:
                parts = [str(r.get("patient", "")), str(r.get("roi", ""))]
                return " / ".join(p for p in parts if p and p != "nan")
            return _roi_label(top), _roi_label(bot)
        except Exception:
            return "-", "-"

    rows: list[dict[str, Any]] = []
    for _, row in hits.iterrows():
        width       = _to_float(row.get("zone_width"))
        zone_label  = str(row.get("zone_label") or format_zone_label(width))
        is_selected = np.isfinite(width) and round(float(width), 2) == round(float(selected_width), 2)
        endpoint    = str(row.get("endpoint_label") or "")
        feat_name   = str(row.get("feature_name") or "")
        stratum     = str(row.get("stratum_label") or "All")
        fi          = feat_info_lookup.get((endpoint, feat_name, stratum), {})

        hr   = _to_float(row.get("hazard_ratio"))
        ci_lo= _to_float(row.get("ci_low"))
        ci_hi= _to_float(row.get("ci_high"))
        hr_str = (f"{float(hr):.2f} [{float(ci_lo):.2f}–{float(ci_hi):.2f}]"
                  if np.isfinite(hr) and np.isfinite(ci_lo) and np.isfinite(ci_hi) else "-")

        fdr_q = _to_float(row.get("fdr_q"))
        p_val = _to_float(row.get("p_value"))
        boot_r = boot_lookup.get(round(float(width), 2) if np.isfinite(width) else -1, float("nan"))

        feat_rank  = fi.get("feature_rank", 9999)
        feat_boot  = fi.get("feature_bootstrap_recovery", float("nan"))
        n_sig_rng  = fi.get("n_significant_ranges", 0)
        sig_band   = fi.get("significant_band_size", 0)
        sig_band_l = fi.get("significant_band_label", "")
        dir_cons   = fi.get("direction_consistency", "")

        rep_high, rep_low = _rep_roi(width, feat_name)
        is_top = isinstance(feat_rank, int) and int(feat_rank) == 1

        rows.append({
            "Zone":                       zone_label,
            "★ Selected":                 "★" if is_selected else "",
            "Endpoint":                   endpoint,
            "Stratum":                    stratum,
            "Family":                     str(row.get("feature_family") or "").replace("_", " ").title(),
            "Feature":                    str(row.get("feature_label") or feat_name),
            "FDR q":                      f"{float(fdr_q):.4f}" if np.isfinite(fdr_q) else "-",
            "p value":                    f"{float(p_val):.4f}" if np.isfinite(p_val) else "-",
            "Direction":                  _direction_text(row.get("direction"), row.get("endpoint_type"), hr),
            "HR [95% CI]":                hr_str,
            "n (High / Low)":             f"{int(_to_float(row.get('group_high_n'), 0))}/{int(_to_float(row.get('group_low_n'), 0))}",
            "Zone bootstrap recovery":    f"{float(boot_r):.2f}" if np.isfinite(boot_r) else "-",
            "Feature rank":               int(feat_rank) if feat_rank < 9999 else "-",
            "Feature bootstrap recovery": f"{float(feat_boot):.2f}" if np.isfinite(feat_boot) else "-",
            "FDR significant in N zones": int(n_sig_rng),
            "Consecutive FDR band":       sig_band_l if sig_band_l else (str(sig_band) + " zone(s)" if sig_band > 0 else "-"),
            "Direction consistent":       dir_cons,
            "Rep. ROI – High group":      rep_high,
            "Rep. ROI – Low group":       rep_low,
            "Comment":                    (
                "⭐ Top-ranked feature (most reproducible cross-zone)"
                if is_top else ""
            ),
            # hidden sort keys
            "_feat_rank_sort": int(feat_rank),
            "_fdr_sort": float(fdr_q) if np.isfinite(fdr_q) else 1.0,
        })

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows).sort_values(
        ["_feat_rank_sort", "_fdr_sort"], ascending=[True, True], na_position="last"
    ).drop(columns=["_feat_rank_sort", "_fdr_sort"]).reset_index(drop=True)
    return result_df


def _build_spatial_range_narrative(
    stats_all: pd.DataFrame,
    feature_summary: pd.DataFrame,
    ordered_widths: list[float],
    zone_geometry: str = "cumulative",
    max_features: int = 6,
) -> str:
    """Narrate, per feature, where the effect appears and where it is lost.

    In cumulative geometry the widths are nested, so a feature turning
    significant at some width and dropping out at a larger one is read as
    *onset* (the band has to reach this far before the effect is measurable) and
    *washout* (beyond here the added tissue dilutes it). In ring geometry the
    zones are disjoint, so the same two numbers instead bracket the distance
    band the effect actually occupies. The wording changes with the mode because
    the same numbers mean different things.
    """
    if stats_all.empty or not ordered_widths:
        return ""

    is_ring = str(zone_geometry).strip().lower() == "ring"
    width_index = {round(float(w), 6): i for i, w in enumerate(ordered_widths)}

    required = {"feature_label", "endpoint_label", "zone_width", "significance_label"}
    if not required.issubset(stats_all.columns):
        return ""

    work = stats_all.copy()
    work["_is_hit"] = work["significance_label"].astype(str).isin(["significant_fdr", "discovery_fdr"])
    hits = work[work["_is_hit"]]
    if hits.empty:
        return ""

    boot_lookup: dict[tuple[str, str], float] = {}
    if not feature_summary.empty and {"feature_label", "endpoint_label"}.issubset(feature_summary.columns):
        for _, row in feature_summary.iterrows():
            boot_lookup[(str(row.get("feature_label")), str(row.get("endpoint_label")))] = _to_float(
                row.get("feature_bootstrap_recovery")
            )

    lines: list[str] = []
    grouped = hits.groupby(["feature_label", "endpoint_label"], dropna=False)
    ranked = sorted(
        grouped,
        key=lambda kv: (
            -int(kv[1]["significance_label"].astype(str).eq("significant_fdr").sum()),
            _to_float(kv[1]["fdr_q"].min()) if "fdr_q" in kv[1].columns else 1.0,
        ),
    )

    for (feature_label, endpoint_label), group in ranked[: int(max_features)]:
        widths_hit = sorted({round(float(w), 6) for w in group["zone_width"].tolist()})
        if not widths_hit:
            continue
        indices = sorted({width_index[w] for w in widths_hit if w in width_index})
        if not indices:
            continue
        onset_w = ordered_widths[indices[0]]
        last_w = ordered_widths[indices[-1]]
        n_scanned = len(ordered_widths)
        held_to_end = indices[-1] == n_scanned - 1

        directions = {str(v) for v in group.get("direction", pd.Series(dtype=str)).tolist() if str(v) and str(v) != "nan"}
        direction_txt = directions.pop() if len(directions) == 1 else "mixed"
        best_q = _to_float(group["fdr_q"].min()) if "fdr_q" in group.columns else float("nan")
        boot = boot_lookup.get((str(feature_label), str(endpoint_label)), float("nan"))

        if is_ring:
            # A ring's label is the annulus between the previous scanned width
            # and this one, not the zero-anchored band the cumulative label
            # prints, so build it from the neighbouring widths.
            def _ring_label(idx: int) -> str:
                inner = float(ordered_widths[idx - 1]) if idx > 0 else 0.0
                return format_zone_interval_label(inner, float(ordered_widths[idx]))

            if indices[0] == indices[-1]:
                where = f"is confined to the **{_ring_label(indices[0])}** band"
            else:
                where = f"spans **{_ring_label(indices[0])}** out to **{_ring_label(indices[-1])}**"
                if len(indices) < (indices[-1] - indices[0] + 1):
                    where += " (with gaps -- not every band in between is significant)"
            sentence = f"- **{feature_label}** ({endpoint_label}): {where}"
        elif indices[0] == indices[-1] and not held_to_end:
            sentence = (
                f"- **{feature_label}** ({endpoint_label}): is significant only at "
                f"**{format_zone_label(onset_w)}** and does not survive at any wider zone"
            )
        else:
            emerges = f"first reaches significance once the zone extends to **{format_zone_label(onset_w)}**"
            if held_to_end:
                fades = f"and holds to the widest zone scanned (**{format_zone_label(last_w)}**), so its outer limit was not reached by this scan"
            else:
                fades = f"and is lost beyond **{format_zone_label(last_w)}**, where the wider zone dilutes it"
            sentence = f"- **{feature_label}** ({endpoint_label}): {emerges}, {fades}"

        extras: list[str] = []
        if direction_txt != "mixed":
            extras.append(f"{direction_txt} throughout")
        else:
            extras.append("**direction flips between zones — treat with caution**")
        if np.isfinite(best_q):
            extras.append(f"best q={best_q:.3g}")
        if np.isfinite(boot):
            extras.append(f"bootstrap recovery {boot:.2f}")
        if extras:
            sentence += " (" + "; ".join(extras) + ")"
        lines.append(sentence)

    if not lines:
        return ""

    header = (
        "**Where each effect lives.** Each zone is an independent band of tissue, so these ranges "
        "localise the effect directly."
        if is_ring else
        "**Where each effect appears and fades.** Zones are cumulative (0 to width) and therefore "
        "nested, so a feature reappearing at larger widths is the *same* inner signal being diluted, "
        "not a separate finding further out. Re-run in ring geometry to localise the distance band."
    )
    return header + "\n\n" + "\n".join(lines)


def _build_top_feature_commentary(
    feature_summary: pd.DataFrame,
    stats_all: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    selected_width: float,
) -> str:
    """Return a markdown paragraph explaining the top-ranked feature."""
    if feature_summary.empty or stats_all.empty:
        return ""

    ranked = feature_summary[feature_summary["feature_rank"].astype(float) == 1.0] if "feature_rank" in feature_summary.columns else pd.DataFrame()
    if ranked.empty:
        ranked = feature_summary.sort_values("feature_rank", na_position="last").head(1)
    if ranked.empty:
        return ""

    top = ranked.iloc[0]
    feat_label  = str(top.get("feature_label") or top.get("feature_name") or "?")
    endpoint    = str(top.get("endpoint_label") or "?")
    stratum     = str(top.get("stratum_label") or "All")
    best_q      = _to_float(top.get("best_fdr_q"))
    best_p      = _to_float(top.get("best_p_value"))
    feat_boot   = _to_float(top.get("feature_bootstrap_recovery"))
    n_sig_rng   = int(_to_float(top.get("n_significant_ranges"), 0.0))
    sig_band    = int(_to_float(top.get("significant_band_size"), 0.0))
    sig_band_l  = str(top.get("significant_band_label") or "")
    best_zone_l = str(top.get("best_zone_label") or format_zone_label(_to_float(top.get("best_zone_width"))))
    dir_cons    = str(top.get("direction_consistency") or "")
    best_hr     = _to_float(top.get("best_hazard_ratio"))

    sel_label   = format_zone_label(selected_width)
    is_in_sel   = best_zone_l == sel_label or (
        not feature_summary.empty and
        "reference_band_indices" in feature_summary.columns
    )

    lines: list[str] = [f"**Top-ranked feature: {feat_label}** (endpoint: {endpoint}"
                        + (f", stratum: {stratum}" if stratum not in {"All", ""} else "") + ")"]

    reasons: list[str] = []
    if np.isfinite(feat_boot):
        reasons.append(f"a feature-level bootstrap recovery of **{float(feat_boot):.2f}**")
    if n_sig_rng > 0:
        band_str = f"including a consecutive FDR band spanning {sig_band_l}" if sig_band_l else f"across {n_sig_rng} zone(s)"
        reasons.append(f"FDR significance in **{n_sig_rng} scanned zone(s)** ({band_str})")
    if dir_cons:
        reasons.append(f"**{dir_cons.lower()}** effect direction across all significant zones")
    if np.isfinite(best_hr):
        reasons.append(f"best HR = **{float(best_hr):.2f}** at zone {best_zone_l}")
    elif np.isfinite(best_q):
        reasons.append(f"best FDR q = **{float(best_q):.4f}**, p = {float(best_p):.4f}" if np.isfinite(best_p) else f"best FDR q = **{float(best_q):.4f}**")

    if reasons:
        lines.append("This feature was ranked #1 based on: " + "; ".join(reasons) + ".")

    # Is its best zone the selected zone?
    sel_rows = candidate_summary[candidate_summary["zone_width"].astype(float).round(2) == round(float(selected_width), 2)] if not candidate_summary.empty else pd.DataFrame()
    if not sel_rows.empty:
        top_feat_col = str(sel_rows.iloc[0].get("feature_stability_top_feature_label") or "")
        if top_feat_col and top_feat_col == feat_label:
            lines.append(
                f"Its optimal zone coincides with the **selected zone ({sel_label})**, "
                f"reinforcing the zone selection: this feature's stable significance band anchored "
                f"the {sel_label} zone's feature stability score."
            )

    return "\n".join(lines)


def _render_live_results(results: dict[str, Any], config: dict[str, Any]) -> None:
    stats_all = results.get("stats_all", pd.DataFrame())
    candidate_summary = results.get("candidate_summary", pd.DataFrame())
    feature_summary = results.get("feature_stability_summary", pd.DataFrame())
    selected_width = float(results.get("selected_width", 0.0))
    combined_by_width = results.get("combined_by_width", {})
    patient_features_by_width = results.get("patient_features_by_width", {})
    patient_outcomes = results.get("patient_outcomes", pd.DataFrame())

    result_tabs = st.tabs(["Tables", "Summary Figures", "Hit Browser", "Secondary Analysis", "Publication Figures", "Feature Stability", "Feature Linkage", "Correlation", "QC Summary", "Representative ROIs"])
    with result_tabs[0]:
        def _format_preview_value(value: Any, decimals: int = 2, as_percent: bool = False) -> str:
            numeric = _to_float(value)
            if not np.isfinite(numeric):
                return "-"
            if as_percent:
                return f"{float(numeric) * 100:.1f}%"
            return f"{float(numeric):.{int(decimals)}f}"

        def _preview_status_text(label: Any) -> str:
            mapping = {
                "significant_fdr": "FDR significant",
                "discovery_fdr": "FDR discovery",
                "nominal_p": "Nominal p<0.05",
                "not_significant": "Not significant",
            }
            return mapping.get(str(label or ""), str(label or "-") or "-")

        def _interactive_width_preview_df(candidate_df: pd.DataFrame) -> pd.DataFrame:
            preview_df = candidate_df.copy()
            preview_df = preview_df.sort_values(["candidate_rank", "zone_width"], ascending=[True, True], na_position="last").head(8).reset_index(drop=True)
            preview_df["Range"] = preview_df["zone_width"].map(format_zone_label)
            preview_df["FDR hits"] = pd.to_numeric(preview_df.get("n_significant", pd.Series(0, index=preview_df.index)), errors="coerce").fillna(0).astype(int)
            preview_df["Discovery hits"] = pd.to_numeric(preview_df.get("n_discovery", pd.Series(0, index=preview_df.index)), errors="coerce").fillna(0).astype(int)
            preview_df["Best FDR"] = preview_df.get("best_fdr_q", pd.Series(np.nan, index=preview_df.index)).map(lambda value: _format_preview_value(value, decimals=3))
            preview_df["Best p"] = preview_df.get("best_p_value", pd.Series(np.nan, index=preview_df.index)).map(lambda value: _format_preview_value(value, decimals=3))
            preview_df["Coverage"] = preview_df.get("patient_coverage_fraction", pd.Series(np.nan, index=preview_df.index)).map(lambda value: _format_preview_value(value, as_percent=True))
            preview_df["Strongest family"] = preview_df.get("strongest_feature_family", pd.Series("", index=preview_df.index)).fillna("").astype(str).str.replace("_", " ").str.title()
            preview_df["Strongest hit"] = preview_df.get("strongest_feature_label", pd.Series("", index=preview_df.index)).fillna("").astype(str)
            preview_df["Endpoint"] = preview_df.get("strongest_endpoint_label", pd.Series("", index=preview_df.index)).fillna("").astype(str)
            preview_df["Status"] = preview_df.get("strongest_significance_label", pd.Series("", index=preview_df.index)).map(_preview_status_text)
            return preview_df

        def _selected_dataframe_rows(selection_event: Any) -> list[int]:
            if selection_event is None:
                return []
            selection = getattr(selection_event, "selection", None)
            if selection is not None:
                rows = getattr(selection, "rows", None)
                if rows is not None:
                    return list(rows)
            if isinstance(selection_event, dict):
                return list(((selection_event.get("selection") or {}).get("rows") or []))
            return []

        def _width_hit_preview_df(stats_df: pd.DataFrame, width_value: float) -> tuple[pd.DataFrame, str]:
            width_hits = stats_df.loc[stats_df["zone_width"].astype(float) == float(width_value)].copy()
            if width_hits.empty:
                return pd.DataFrame(), "No hits are available for this range yet."
            significant_hits = width_hits[width_hits["significance_label"].isin(["significant_fdr", "discovery_fdr"])].copy()
            if significant_hits.empty:
                preview_hits = width_hits.sort_values(["fdr_q", "p_value"], na_position="last").head(8).copy()
                note = "This range has no FDR or discovery hits, so the preview shows the lowest p-value rows instead."
            else:
                preview_hits = significant_hits.sort_values(["fdr_q", "p_value"], na_position="last").head(8).copy()
                note = "Previewing the current significant or discovery hits for the selected range."
            preview_hits["Endpoint"] = preview_hits.get("endpoint_label", pd.Series("", index=preview_hits.index)).fillna("").astype(str)
            preview_hits["Stratum"] = preview_hits.get("stratum_label", pd.Series("", index=preview_hits.index)).fillna("").astype(str)
            preview_hits["Family"] = preview_hits.get("feature_family", pd.Series("", index=preview_hits.index)).fillna("").astype(str).str.replace("_", " ").str.title()
            preview_hits["Feature"] = preview_hits.get("feature_label", pd.Series("", index=preview_hits.index)).fillna("").astype(str)
            preview_hits["Status"] = preview_hits.get("significance_label", pd.Series("", index=preview_hits.index)).map(_preview_status_text)
            preview_hits["FDR q"] = preview_hits.get("fdr_q", pd.Series(np.nan, index=preview_hits.index)).map(lambda value: _format_preview_value(value, decimals=3))
            preview_hits["p value"] = preview_hits.get("p_value", pd.Series(np.nan, index=preview_hits.index)).map(lambda value: _format_preview_value(value, decimals=3))
            preview_hits["Direction"] = [
                _direction_text(direction, endpoint_type, hazard_ratio)
                for direction, endpoint_type, hazard_ratio in zip(
                    preview_hits.get("direction", pd.Series("", index=preview_hits.index)),
                    preview_hits.get("endpoint_type", pd.Series("", index=preview_hits.index)),
                    preview_hits.get("hazard_ratio", pd.Series(np.nan, index=preview_hits.index)),
                )
            ]
            return preview_hits[["Endpoint", "Stratum", "Family", "Feature", "Status", "FDR q", "p value", "Direction"]], note

        st.markdown("### Candidate Summary")
        _render_selected_candidate_summary(candidate_summary, selected_width, "tab8_fs_live")

        # ── Zone Selection Rationale ──────────────────────────────────────────
        if not candidate_summary.empty:
            with st.expander("Zone Selection Rationale", expanded=True):
                rationale_md = _build_zone_selection_rationale_text(
                    candidate_summary, feature_summary, selected_width
                )
                st.markdown(rationale_md)

                _ordered_widths = sorted(
                    {float(w) for w in candidate_summary.get("zone_width", pd.Series(dtype=float)).tolist()
                     if np.isfinite(_to_float(w))}
                )
                _geometry = str(
                    (config.get("screening", {}) or {}).get("zone_geometry", "cumulative")
                )
                _range_md = _build_spatial_range_narrative(
                    stats_all, feature_summary, _ordered_widths, zone_geometry=_geometry
                )
                if _range_md:
                    st.markdown("---")
                    st.markdown(_range_md)

        recent_events = st.session_state.get("_tab8_fs_recent_width_events", [])
        if recent_events:
            recent_df = pd.DataFrame(recent_events[::-1]).copy()
            recent_df["Completed"] = recent_df.get("elapsed_seconds", pd.Series(np.nan, index=recent_df.index)).map(
                lambda value: f"{int(round(float(value)))}s" if pd.notna(value) and np.isfinite(float(value)) else "-"
            )
            recent_df["Width"] = recent_df.get("width", pd.Series(np.nan, index=recent_df.index)).map(lambda value: _format_preview_value(value, decimals=1))
            recent_df["FDR hits"] = pd.to_numeric(recent_df.get("n_significant", pd.Series(0, index=recent_df.index)), errors="coerce").fillna(0).astype(int)
            recent_df["Discovery hits"] = pd.to_numeric(recent_df.get("n_discovery", pd.Series(0, index=recent_df.index)), errors="coerce").fillna(0).astype(int)
            recent_df["Best FDR"] = recent_df.get("best_fdr_q", pd.Series(np.nan, index=recent_df.index)).map(lambda value: _format_preview_value(value, decimals=3))
            recent_df["Strongest family"] = recent_df.get("strongest_feature_family", pd.Series("", index=recent_df.index)).fillna("").astype(str).str.replace("_", " ").str.title()
            recent_df["Top hit"] = recent_df.get("strongest_feature_label", pd.Series("", index=recent_df.index)).fillna("").astype(str)
            recent_df["Status"] = recent_df.get("strongest_significance_label", pd.Series("", index=recent_df.index)).map(_preview_status_text)
            st.markdown("#### Recent Completed Ranges")
            recent_df["Range"] = recent_df.get("width", pd.Series(np.nan, index=recent_df.index)).map(format_zone_label)
            st.dataframe(recent_df[["Completed", "Range", "FDR hits", "Discovery hits", "Best FDR", "Strongest family", "Top hit", "Status"]], width="stretch", hide_index=True, height=min(220, 40 + 35 * (len(recent_df) + 1)))
        if not candidate_summary.empty and not stats_all.empty:
            st.markdown("#### Range Hit Preview")
            st.caption("Click a row to preview the hit list for that range.")
            preview_source = _interactive_width_preview_df(candidate_summary)
            selection_event = st.dataframe(
                preview_source[["Range", "FDR hits", "Discovery hits", "Best FDR", "Best p", "Coverage", "Strongest family", "Strongest hit", "Endpoint", "Status"]],
                width="stretch",
                hide_index=True,
                key="tab8_fs_live_width_preview_table",
                on_select="rerun",
                selection_mode="single-row",
                height=min(280, 40 + 35 * (len(preview_source) + 1)),
            )
            selected_rows = _selected_dataframe_rows(selection_event)
            selected_preview_width = _to_float(st.session_state.get("_tab8_fs_live_preview_width", selected_width))
            if selected_rows:
                selected_preview_width = float(_to_float(preview_source.iloc[selected_rows[0]].get("zone_width")))
                st.session_state["_tab8_fs_live_preview_width"] = selected_preview_width
            elif not np.isfinite(selected_preview_width) or _width_state_key(selected_preview_width) not in {_width_state_key(value) for value in preview_source["zone_width"].tolist()}:
                selected_preview_width = float(_to_float(preview_source.iloc[0].get("zone_width")))
                st.session_state["_tab8_fs_live_preview_width"] = selected_preview_width

            _stateful_checkbox("Hide redundant features", key="tab8_fs_hide_redundant_live", default=False, help="Hide features that are highly correlated with a more significant feature in the same range/endpoint/stratum group.")
            hide_redundant = st.session_state.get("tab8_fs_hide_redundant_live", False)
            stats_view = stats_all.copy()
            if hide_redundant and "is_redundant" in stats_view.columns:
                stats_view = stats_view[~stats_view["is_redundant"]].copy()

            preview_hits_df, preview_note = _width_hit_preview_df(stats_view, float(selected_preview_width))
            st.caption(f"Selected range {format_zone_label(selected_preview_width)}. {preview_note}")
            if preview_hits_df.empty:
                st.info("No hit rows are available for the selected range.")
            else:
                st.dataframe(preview_hits_df, width="stretch", hide_index=True, height=min(260, 40 + 35 * (len(preview_hits_df) + 1)))
        st.dataframe(candidate_summary, width="stretch")
        _render_dataframe_download(candidate_summary, "Download candidate summary", "candidate_summary.csv", "tab8_fs_candidate_download")

        # ── FDR Significant Hits Summary ──────────────────────────────────────
        if not stats_all.empty:
            st.markdown("### FDR Significant Hits Summary")
            fdr_hits_df = _build_fdr_hits_summary_df(
                stats_all, candidate_summary, feature_summary, combined_by_width, selected_width
            )
            if fdr_hits_df.empty:
                st.info("No FDR significant hits (q ≤ 0.05) were found in this run.")
            else:
                # ── Top Feature Commentary ────────────────────────────────────
                top_feat_comment = _build_top_feature_commentary(
                    feature_summary, stats_all, candidate_summary, selected_width
                )
                if top_feat_comment:
                    st.info(top_feat_comment)

                # ── Motif discovery: spatial organisation x outcome ───────────
                _motif_summary = build_motif_enrichment_summary(combined_by_width, stats_all)
                if not _motif_summary.empty:
                    with st.expander("Motif discovery (dyads & triads): organised vs prognostic", expanded=False):
                        st.caption(
                            "Two independent questions per motif and zone. **Enrichment Z** comes from the "
                            "permutation null and asks whether the motif is arranged more often than random "
                            "labelling of the same cells would produce. **FDR q** comes from the survival "
                            "screen and asks whether a patient's amount of it tracks outcome. A motif can "
                            "do either without the other -- the rows worth pursuing are those flagged in "
                            "**both**. Requires permutation testing to be enabled."
                        )
                        _both = _motif_summary[_motif_summary["both"]] if "both" in _motif_summary.columns else pd.DataFrame()
                        if not _both.empty:
                            for _, _row in _both.head(8).iterrows():
                                _frac = _to_float(_row.get("frac_regions_enriched"))
                                st.markdown(
                                    f"- **{_row.get('feature_label')}** at **{_row.get('zone_label')}**: "
                                    f"arranged non-randomly (mean Z = {_to_float(_row.get('mean_enrichment_z')):.2f}"
                                    + (f", enriched in {_frac*100:.0f}% of regions" if np.isfinite(_frac) else "")
                                    + f") **and** linked to {_row.get('endpoint_label')} "
                                    f"(q = {_to_float(_row.get('fdr_q')):.3g}, {_row.get('direction')})."
                                )
                        else:
                            st.info(
                                "No motif is both spatially enriched and outcome-linked in this run. "
                                "Rows below still show each motif's organisation and outcome result separately."
                            )
                        _motif_cols = [c for c in [
                            "feature_family", "feature_label", "zone_label", "mean_enrichment_z",
                            "frac_regions_enriched", "n_regions", "endpoint_label", "fdr_q",
                            "significance_label", "hazard_ratio", "direction",
                            "is_spatially_enriched", "is_outcome_linked", "both",
                        ] if c in _motif_summary.columns]
                        st.dataframe(_motif_summary[_motif_cols], width="stretch", hide_index=True)
                        _render_dataframe_download(
                            _motif_summary,
                            "Download motif enrichment summary (CSV)",
                            "motif_enrichment_summary.csv",
                            "tab8_fs_motif_enrichment_download",
                        )

                st.caption(
                    f"{len(fdr_hits_df)} FDR-significant hit(s) across all scanned zones. "
                    "★ marks rows belonging to the selected zone. ⭐ marks the top-ranked feature. "
                    "'Rep. ROI' columns show the patient/ROI with the highest and lowest feature value "
                    "in the zone, as a starting point for visual inspection."
                )
                display_cols = [
                    c for c in [
                        "Zone", "★ Selected", "Endpoint", "Stratum", "Family", "Feature",
                        "FDR q", "p value", "Direction", "HR [95% CI]", "n (High / Low)",
                        "Zone bootstrap recovery", "Feature rank", "Feature bootstrap recovery",
                        "FDR significant in N zones", "Consecutive FDR band",
                        "Direction consistent", "Rep. ROI – High group", "Rep. ROI – Low group",
                        "Comment",
                    ] if c in fdr_hits_df.columns
                ]
                st.dataframe(fdr_hits_df[display_cols], width="stretch", hide_index=True,
                             height=min(600, 40 + 38 * (len(fdr_hits_df) + 1)))
                _render_dataframe_download(
                    fdr_hits_df, "Download FDR hits summary",
                    "fdr_hits_summary.csv", "tab8_fs_fdr_hits_download"
                )

        if not stats_all.empty:
            st.markdown("### Full Screening Results")
            display_stats = stats_view.sort_values(["fdr_q", "p_value"], na_position="last") if 'stats_view' in locals() else stats_all.sort_values(["fdr_q", "p_value"], na_position="last")
            st.dataframe(display_stats, width="stretch", height=420)
            _render_dataframe_download(display_stats, "Download screening results", "screening_results_all.csv", "tab8_fs_stats_download")
    with result_tabs[1]:
        if candidate_summary.empty:
            st.info("No figures are available yet.")
        else:
            candidate_fig = plot_candidate_scan(candidate_summary, selected_width)
            _render_figure_with_download(candidate_fig, "Candidate scan", "tab8_fs_candidate_scan")
            plt.close(candidate_fig)
            if not stats_all.empty:
                endpoint_options = sorted(stats_all["endpoint_label"].astype(str).unique().tolist())
                _stateful_selectbox("Endpoint for summary figures", options=endpoint_options, key="tab8_fs_summary_endpoint", default=endpoint_options[0] if endpoint_options else None)
                selected_endpoint = st.session_state.get("tab8_fs_summary_endpoint", "")
                stratum_options = sorted(stats_all.loc[stats_all["endpoint_label"].astype(str) == str(selected_endpoint), "stratum_label"].astype(str).unique().tolist())
                _stateful_selectbox("Stratum", options=stratum_options, key="tab8_fs_summary_stratum", default=stratum_options[0] if stratum_options else None)
                selected_stratum = st.session_state.get("tab8_fs_summary_stratum", "")
                volcano = plot_volcano(
                    stats_all,
                    selected_endpoint,
                    selected_width,
                    selected_stratum,
                    label_settings=_volcano_label_settings(config),
                )
                _render_figure_with_download(volcano, "Volcano plot", "tab8_fs_volcano")
                plt.close(volcano)
                forest = plot_forest(stats_all, selected_endpoint, selected_width, selected_stratum)
                _render_figure_with_download(forest, "Forest plot", "tab8_fs_forest")
                plt.close(forest)
                family = plot_family_summary(stats_all, selected_endpoint, selected_width, selected_stratum)
                _render_figure_with_download(family, "Family summary", "tab8_fs_family_summary")
                plt.close(family)
    with result_tabs[2]:
        if stats_all.empty:
            st.info("Run the screen first to inspect individual hits.")
        else:
            _stateful_checkbox("Hide redundant features", key="tab8_fs_hide_redundant_browser", default=False, help="Hide features that are highly correlated with a more significant feature in the same range/endpoint/stratum group.")
            hide_redundant = st.session_state.get("tab8_fs_hide_redundant_browser", False)
            stats_view = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").copy()
            if hide_redundant and "is_redundant" in stats_view.columns:
                stats_view = stats_view[~stats_view["is_redundant"]].copy()

            stats_view["_label"] = stats_view.apply(lambda row: f"{row['endpoint_label']} | {row['feature_label']} | {row['stratum_label']} | range={str(row.get('zone_label') or format_zone_label(row['zone_width']))}", axis=1)
            _stateful_selectbox("Hit", options=stats_view["_label"].tolist(), key="tab8_fs_hit_label", default=stats_view["_label"].iloc[0] if not stats_view.empty else None)
            selected_hit_label = st.session_state.get("tab8_fs_hit_label", "")
            hit_row = stats_view.loc[stats_view["_label"] == selected_hit_label].iloc[0]
            width = float(hit_row["zone_width"])
            combined = combined_by_width.get(width, pd.DataFrame())
            patient_features = patient_features_by_width.get(width, pd.DataFrame())
            hit_fig = plot_feature_endpoint(hit_row, patient_features, patient_outcomes, config)
            _render_figure_with_download(hit_fig, "Selected hit plot", "tab8_fs_selected_hit")
            plt.close(hit_fig)

            # Waterfall plot for binary endpoints
            if str(hit_row.get("endpoint_type", "")) == "binary":
                waterfall_fig = plot_feature_waterfall(hit_row, patient_features, patient_outcomes, config)
                _render_figure_with_download(waterfall_fig, "Waterfall plot", "tab8_fs_waterfall")
                plt.close(waterfall_fig)

            patient_export_df = build_feature_endpoint_export_df(hit_row, patient_features, patient_outcomes, config)
            patient_audit_df = build_feature_endpoint_audit_df(
                hit_row,
                results.get("analysis_df", pd.DataFrame()),
                patient_features,
                patient_outcomes,
                config,
                zone_start=0.0,
                zone_end=float(width),
            )
            if not patient_export_df.empty:
                st.markdown("#### Patient Data Behind This Figure")
                st.dataframe(patient_export_df, width="stretch", height=260)
                endpoint_stub = str(hit_row["endpoint_label"]).lower().replace(" ", "_")
                feature_stub = str(hit_row["feature_name"]).replace("::", "_").replace("/", "_")
                _render_dataframe_download(
                    patient_export_df,
                    "Download patient CSV for selected hit",
                    f"{endpoint_stub}_{feature_stub}_patients.csv",
                    "tab8_fs_selected_hit_patient_csv_download",
                )
                _render_patient_inclusion_audit(
                    patient_audit_df,
                    "Download patient inclusion audit",
                    f"{endpoint_stub}_{feature_stub}_patient_audit.csv",
                    "tab8_fs_selected_hit_patient_audit_download",
                )
            elif not patient_audit_df.empty:
                endpoint_stub = str(hit_row["endpoint_label"]).lower().replace(" ", "_")
                feature_stub = str(hit_row["feature_name"]).replace("::", "_").replace("/", "_")
                _render_patient_inclusion_audit(
                    patient_audit_df,
                    "Download patient inclusion audit",
                    f"{endpoint_stub}_{feature_stub}_patient_audit.csv",
                    "tab8_fs_selected_hit_patient_audit_download",
                )

            _render_selected_hit_motif_export_section(
                analysis_df=results.get("analysis_df", pd.DataFrame()),
                stats_all=stats_all,
                config=config,
                hit_row=hit_row,
                section_label="Discovery",
                zone_start=0.0,
                zone_end=float(width),
                key_prefix="tab8_fs_selected_hit",
                selected_roi_features_df=combined,
                selected_patient_export_df=patient_export_df,
                selected_patient_features_df=patient_features,
            )
    with result_tabs[3]:
        if stats_all.empty:
            st.info("Run the screen first to prepare the secondary analysis.")
        else:
            stats_view = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").copy()
            stats_view["_label"] = stats_view.apply(lambda row: f"{row['endpoint_label']} | {row['feature_label']} | {row['stratum_label']} | range={str(row.get('zone_label') or format_zone_label(row['zone_width']))}", axis=1)
            selected_hit_label = st.session_state.get("tab8_fs_hit_label", stats_view["_label"].iloc[0] if not stats_view.empty else "")
            if selected_hit_label and bool((stats_view["_label"] == selected_hit_label).any()):
                secondary_hit_row = stats_view.loc[stats_view["_label"] == selected_hit_label].iloc[0]
                secondary_width = float(secondary_hit_row["zone_width"])
                _render_secondary_analysis_section(
                    analysis_df=results.get("analysis_df", pd.DataFrame()),
                    patient_outcomes_df=patient_outcomes,
                    config=config,
                    hit_row=secondary_hit_row,
                    zone_start=0.0,
                    zone_end=float(secondary_width),
                    state_prefix="tab8_fs_live",
                    download_prefix="discovery",
                )
            else:
                st.info("Select a hit in the Hit Browser first to populate the secondary analysis.")

    with result_tabs[4]:
        if stats_all.empty:
            st.info("Run the screen first to prepare publication-style selected-hit figures.")
        else:
            stats_view = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").copy()
            stats_view["_label"] = stats_view.apply(lambda row: f"{row['endpoint_label']} | {row['feature_label']} | {row['stratum_label']} | range={str(row.get('zone_label') or format_zone_label(row['zone_width']))}", axis=1)
            selected_hit_label = st.session_state.get("tab8_fs_hit_label", stats_view["_label"].iloc[0] if not stats_view.empty else "")
            if selected_hit_label and bool((stats_view["_label"] == selected_hit_label).any()):
                publication_hit_row = stats_view.loc[stats_view["_label"] == selected_hit_label].iloc[0]
                publication_width = float(publication_hit_row["zone_width"])
                publication_combined = combined_by_width.get(publication_width, pd.DataFrame())
                publication_patient_features = patient_features_by_width.get(publication_width, pd.DataFrame())
                publication_patient_export_df = build_feature_endpoint_export_df(publication_hit_row, publication_patient_features, patient_outcomes, config)
                _render_publication_figures_section(
                    analysis_df=results.get("analysis_df", pd.DataFrame()),
                    combined=publication_combined,
                    patient_features=publication_patient_features,
                    patient_outcomes=patient_outcomes,
                    patient_export_df=publication_patient_export_df,
                    stats_all=stats_all,
                    candidate_summary=results.get("candidate_summary", pd.DataFrame()),
                    config=config,
                    hit_row=publication_hit_row,
                    section_label="Discovery",
                    zone_start=0.0,
                    zone_end=float(publication_width),
                    selected_candidate_width=_to_float(results.get("selected_width"), float("nan")),
                    key_prefix="tab8_fs_publication",
                )
            else:
                st.info("Select a hit in the Hit Browser first to populate the publication figures.")

    with result_tabs[5]:
        _render_feature_stability_panel(
            feature_summary=feature_summary,
            stats_all=stats_all,
            key_prefix="tab8_fs_live",
            default_top_n=int(config.get("screening", {}).get("feature_stability_top_n", 25)),
            neighbor_steps=int(config.get("screening", {}).get("feature_bootstrap_neighbor_steps", 1)),
        )
    with result_tabs[6]:
        _render_feature_linkage_panel(
            feature_summary=feature_summary,
            key_prefix="tab8_fs_live",
        )
    with result_tabs[7]:
        st.markdown("### Feature Correlation")
        st.caption(
            "Spearman rank correlation heatmap of top significant features at the selected width. "
            "Hierarchically clustered to reveal groups of redundant (highly correlated) features "
            "versus independent discovery signals."
        )
        if stats_all.empty or not patient_features_by_width:
            st.info("Run the screen first to generate the correlation heatmap.")
        else:
            endpoint_options = sorted(stats_all["endpoint_label"].astype(str).unique().tolist())
            _stateful_selectbox("Endpoint for correlation", options=endpoint_options, key="tab8_fs_corr_endpoint", default=endpoint_options[0] if endpoint_options else None)
            selected_corr_endpoint = st.session_state.get("tab8_fs_corr_endpoint", "")
            stratum_options = sorted(stats_all.loc[stats_all["endpoint_label"].astype(str) == str(selected_corr_endpoint), "stratum_label"].astype(str).unique().tolist())
            _stateful_selectbox("Stratum for correlation", options=stratum_options, key="tab8_fs_corr_stratum", default=stratum_options[0] if stratum_options else None)
            selected_corr_stratum = st.session_state.get("tab8_fs_corr_stratum", "")
            corr_top_n = st.number_input("Top N features", min_value=2, max_value=50, value=20, key="tab8_fs_corr_top_n")
            corr_fig = plot_top_hits_correlation(
                stats_all,
                patient_features_by_width,
                selected_width,
                top_n=int(corr_top_n),
                endpoint_label=selected_corr_endpoint,
                stratum_label=selected_corr_stratum,
            )
            _render_figure_with_download(corr_fig, "Feature correlation heatmap", "tab8_fs_correlation")
            plt.close(corr_fig)
    with result_tabs[8]:
        st.markdown("### QC Summary")
        st.caption(
            "Structured diagnostic metrics for the current screening run. "
            "Use this to verify data quality, model convergence, and endpoint balance."
        )
        qc = build_qc_summary(results, config)
        qc_col1, qc_col2, qc_col3 = st.columns(3)
        with qc_col1:
            st.metric("Total Cells", f"{qc.get('total_cells', 0):,}")
            st.metric("Total Patients", f"{qc.get('total_patients', 0):,}")
            st.metric("Total Regions", f"{qc.get('total_regions', 0):,}")
            median_cells = qc.get("cells_per_patient_median", 0)
            st.metric("Cells/Patient (median)", f"{median_cells:,.0f}" if np.isfinite(float(median_cells)) else "N/A")
        with qc_col2:
            st.metric("Widths Scanned", f"{qc.get('total_widths_scanned', 0):,}")
            st.metric("Widths with FDR Hits", f"{qc.get('widths_with_fdr_hits', 0):,}")
            st.metric("Features Tested", f"{qc.get('total_features_tested', 0):,}")
            st.metric("FDR Significant", f"{qc.get('total_fdr_significant', 0):,}")
        with qc_col3:
            st.metric("FDR Discovery", f"{qc.get('total_fdr_discovery', 0):,}")
            st.metric("Nominal p < 0.05", f"{qc.get('total_nominal_p', 0):,}")
            st.metric("Cox Convergence Rate", str(qc.get("cox_convergence_rate", "N/A")))
            mean_recovery = qc.get("mean_bootstrap_recovery", float("nan"))
            st.metric("Mean Bootstrap Recovery", f"{mean_recovery:.2f}" if np.isfinite(float(mean_recovery)) else "N/A")

        # Bootstrap diagnostics
        st.markdown("#### Bootstrap Diagnostics")
        boot_col1, boot_col2 = st.columns(2)
        with boot_col1:
            st.metric("Bootstrap Widths Tested", f"{qc.get('bootstrap_widths_tested', 0):,}")
        with boot_col2:
            st.metric("Mean Iterations per Width", f"{qc.get('bootstrap_mean_iterations', 0):.1f}")
        st.metric("Features with Recovery ≥ 80%", f"{qc.get('features_with_high_recovery', 0):,}")

        # Endpoint balance
        endpoint_balance = qc.get("endpoint_balance", [])
        if endpoint_balance:
            st.markdown("#### Endpoint Group Balance")
            for eb in endpoint_balance:
                groups_str = ", ".join(f"{k}: {v}" for k, v in eb.get("groups", {}).items())
                balance_ratio = eb.get("balance_ratio", 0.0)
                warning = " ⚠️ Unbalanced" if balance_ratio < 0.33 else ""
                st.text(f"{eb.get('endpoint', '?')}: {groups_str}  (ratio: {balance_ratio:.2f}){warning}")
    with result_tabs[9]:
        from .roi_explorer import _render_roi_explorer
        _render_roi_explorer(results.get("analysis_df", pd.DataFrame()), config, state_prefix="tab8_fs_disc_roi")


def _render_loaded_run(loaded_run: dict[str, Any]) -> None:
    st.markdown("## Loaded Run")
    manifest = loaded_run.get("manifest", {})
    st.caption(f"Run directory: {loaded_run.get('run_dir')}")
    st.json({
        "source_csv": manifest.get("source_csv"),
        "selected_width": manifest.get("selected_width"),
        "available_widths": manifest.get("available_widths", []),
    })

    tabs = st.tabs(["Tables", "Summary Figures", "Hit Browser", "Feature Stability", "Feature Linkage"])
    with tabs[0]:
        if not loaded_run["candidate_summary"].empty:
            _render_selected_candidate_summary(loaded_run["candidate_summary"], float(manifest.get("selected_width", loaded_run["candidate_summary"]["zone_width"].iloc[0])), "tab8_fs_loaded")
            st.dataframe(loaded_run["candidate_summary"], width="stretch")
        if not loaded_run["stats_all"].empty:
            st.dataframe(loaded_run["stats_all"].sort_values(["fdr_q", "p_value"], na_position="last"), width="stretch", height=420)
    with tabs[1]:
        candidate = loaded_run["candidate_summary"]
        stats_all = loaded_run["stats_all"]
        selected_width = float(manifest.get("selected_width", candidate["zone_width"].iloc[0] if not candidate.empty else 0.0))
        if not candidate.empty:
            candidate_fig = plot_candidate_scan(candidate, selected_width)
            _render_figure_with_download(candidate_fig, "Candidate scan", "tab8_fs_loaded_candidate")
            plt.close(candidate_fig)
        if not stats_all.empty:
            endpoint_options = sorted(stats_all["endpoint_label"].astype(str).unique().tolist())
            _stateful_selectbox("Loaded endpoint", options=endpoint_options, key="tab8_fs_loaded_endpoint", default=endpoint_options[0] if endpoint_options else None)
            selected_endpoint = st.session_state.get("tab8_fs_loaded_endpoint", "")
            stratum_options = sorted(stats_all.loc[stats_all["endpoint_label"].astype(str) == str(selected_endpoint), "stratum_label"].astype(str).unique().tolist())
            _stateful_selectbox("Loaded stratum", options=stratum_options, key="tab8_fs_loaded_stratum", default=stratum_options[0] if stratum_options else None)
            selected_stratum = st.session_state.get("tab8_fs_loaded_stratum", "")
            volcano = plot_volcano(
                stats_all,
                selected_endpoint,
                selected_width,
                selected_stratum,
                label_settings=_volcano_label_settings(),
            )
            _render_figure_with_download(volcano, "Volcano plot", "tab8_fs_loaded_volcano")
            plt.close(volcano)
            forest = plot_forest(stats_all, selected_endpoint, selected_width, selected_stratum)
            _render_figure_with_download(forest, "Forest plot", "tab8_fs_loaded_forest")
            plt.close(forest)
            family = plot_family_summary(stats_all, selected_endpoint, selected_width, selected_stratum)
            _render_figure_with_download(family, "Family summary", "tab8_fs_loaded_family")
            plt.close(family)
    with tabs[2]:
        stats_all = loaded_run["stats_all"]
        patient_features = loaded_run["patient_features_all"]
        patient_outcomes = loaded_run["patient_outcomes"]
        config = manifest.get("config", {})
        if stats_all.empty or patient_features.empty or patient_outcomes.empty or not config:
            st.info("Loaded run is missing the tables required to regenerate selected-hit plots.")
        else:
            _stateful_checkbox("Hide redundant features", key="tab8_fs_hide_redundant_loaded", default=False, help="Hide features that are highly correlated with a more significant feature in the same range/endpoint/stratum group.")
            hide_redundant = st.session_state.get("tab8_fs_hide_redundant_loaded", False)
            stats_view = stats_all.sort_values(["fdr_q", "p_value"], na_position="last").copy()
            if hide_redundant and "is_redundant" in stats_view.columns:
                stats_view = stats_view[~stats_view["is_redundant"]].copy()

            stats_view["_label"] = stats_view.apply(lambda row: f"{row['endpoint_label']} | {row['feature_label']} | {row['stratum_label']} | range={str(row.get('zone_label') or format_zone_label(row['zone_width']))}", axis=1)
            _labels = stats_view["_label"].tolist()
            # Guard: stale session key from a previous run may not exist in this view
            _current_hit = st.session_state.get("tab8_fs_loaded_hit", "")
            _default_hit = _current_hit if _current_hit in _labels else (_labels[0] if _labels else None)
            _stateful_selectbox("Loaded hit", options=_labels, key="tab8_fs_loaded_hit", default=_default_hit)
            selected_label = st.session_state.get("tab8_fs_loaded_hit", "")
            _hit_matches = stats_view.loc[stats_view["_label"] == selected_label]
            if _hit_matches.empty:
                st.info("No matching hit found for the current selection. Please choose a hit from the list above.")
                return
            hit_row = _hit_matches.iloc[0]
            hit_fig = plot_feature_endpoint(hit_row, patient_features, patient_outcomes, config)
            _render_figure_with_download(hit_fig, "Selected hit plot", "tab8_fs_loaded_hit_plot")
            plt.close(hit_fig)
            patient_export_df = build_feature_endpoint_export_df(hit_row, patient_features, patient_outcomes, config)
            if not patient_export_df.empty:
                st.markdown("#### Patient Data Behind This Figure")
                st.dataframe(patient_export_df, width="stretch", height=260)
                endpoint_stub = str(hit_row["endpoint_label"]).lower().replace(" ", "_")
                feature_stub = str(hit_row["feature_name"]).replace("::", "_").replace("/", "_")
                _render_dataframe_download(
                    patient_export_df,
                    "Download patient CSV for loaded hit",
                    f"{endpoint_stub}_{feature_stub}_patients.csv",
                    "tab8_fs_loaded_hit_patient_csv_download",
                )
    with tabs[3]:
        _render_feature_stability_panel(
            feature_summary=loaded_run.get("feature_stability_summary", pd.DataFrame()),
            stats_all=loaded_run.get("stats_all", pd.DataFrame()),
            key_prefix="tab8_fs_loaded",
            default_top_n=int((manifest.get("config", {}) or {}).get("screening", {}).get("feature_stability_top_n", 25)),
            neighbor_steps=int((manifest.get("config", {}) or {}).get("screening", {}).get("feature_bootstrap_neighbor_steps", 1)),
        )
    with tabs[4]:
        _render_feature_linkage_panel(
            feature_summary=loaded_run.get("feature_stability_summary", pd.DataFrame()),
            key_prefix="tab8_fs_loaded",
        )


def render_feature_screening(project_root: Path, df: pd.DataFrame, unified_csv_path: str | None = None) -> None:
    _drop_legacy_widget_state_keys()
    _init_defaults(df, project_root, unified_csv_path)
    _consume_pending_reference_reset(df, project_root, unified_csv_path)
    # Auto-save on every render so settings survive session restarts, and mirror
    # them into the project JSON so another session, or another node, opening the
    # same project receives them too.
    _save_settings_to_cache(project_root, unified_csv_path)
    _sync_settings_into_project_json(project_root, unified_csv_path)

    st.markdown("### Feature Screening")
    st.caption(
        "Scan tumor-adjacent spatial feature families across candidate distance windows, "
        "screen them against survival or binary outcomes, and review figures and representative ROIs without leaving Tab 8."
    )
    st.caption(
        "Settings are **auto-saved** for each CSV so they persist across sessions. "
        "Use **Export Settings** to save a portable JSON or **Import Settings** to restore a previous configuration. "
        "Use **↺ Reset to Defaults** to go back to the reference starting values."
    )
    _render_reference_defaults_action(project_root, unified_csv_path)

    with st.expander("Method notes", expanded=False):
        st.markdown(
            "This tab adapts the dynamic feature-screening workflow into a GUI-first interface. "
            "You can remap every key column, choose which feature families to screen, "
            "configure OS / DFS / binary endpoints, stratify by treatment or timepoint, "
            "and save a run manifest that lets you reload tables and regenerate figures later."
        )

    _render_column_mapping(df)
    _render_feature_controls()
    _render_endpoint_controls(df)
    _render_screening_controls()
    _render_output_controls(project_root)

    config = _build_config(df, unified_csv_path)
    has_applied_config = _render_apply_controls(config, df, project_root, unified_csv_path)

    # Deliberately a radio, not st.tabs. Streamlit executes the body of EVERY
    # st.tabs branch on every rerun and merely hides the inactive one with CSS.
    # With Discovery and Validation as tabs, changing any widget in Validation
    # also rebuilt the entire Discovery output -- every figure across all scanned
    # widths -- and vice versa, so a dropdown change cost minutes. A radio runs
    # only the selected branch, so switching a display option is instant.
    _stateful_selectbox(
        "Workflow panel",
        options=["Discovery Scan", "Validation"],
        key="tab8_fs_workflow_panel",
        default=st.session_state.get("tab8_fs_workflow_panel", "Discovery Scan"),
        help=(
            "Only the selected panel is rendered. The other panel's results stay in memory "
            "and reappear unchanged when you switch back -- nothing is recomputed by switching."
        ),
    )
    _active_panel = str(st.session_state.get("tab8_fs_workflow_panel", "Discovery Scan"))

    if _active_panel == "Discovery Scan":
        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            if st.button("Run feature screening", key="tab8_fs_run_btn", type="primary", disabled=not has_applied_config, help="Runs the scan using the last applied configuration and saves a timestamped output folder."):
                _run_pipeline(df)
        with action_col2:
            _render_existing_run_loader()

        _render_log_panel()

        results = st.session_state.get("_tab8_fs_results")
        if results:
            with _timed_section("discovery:live_results"):
                _render_live_results(results, st.session_state.get("_tab8_fs_applied_config", config))

        loaded_run = st.session_state.get("_tab8_fs_loaded_run")
        if loaded_run:
            _render_loaded_run(loaded_run)
    else:
        with _timed_section("validation:panel"):
            _render_validation_panel(df, st.session_state.get("_tab8_fs_applied_config", config), unified_csv_path)