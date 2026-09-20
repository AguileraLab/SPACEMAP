"""Publication-ready cell / tissue overlay figures for Tab 9.

Builds one composited figure per ROI from up to three layers -- the ROI image,
tissue annotation polygons, and cells -- any of which may be absent. All the
rendering lives in :mod:`overlay_core` (no Streamlit), so this module is only
the control panel; the same engine is reusable from the actual-tissue panel.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from . import overlay_core as oc
from ..tab7_phenotyping.cell_visualization import (
    render_cell_visualization as _legacy_render_cell_visualization,
)

_K = "tab9_ov_"   # settings keys: plain names, persisted with the project
# Render outputs (image bytes, the info dict, the download filename) are results,
# not settings. They are given a leading underscore so the persistence layer
# never even considers them -- a rendered PNG has no business in a project file,
# and logging its repr on every render floods the terminal.
_T = "_tab9_ov_"

# Same palette actual_tissue seeds its phenotype pickers from, so a fresh
# category gets the same colour in both panels.
# The house scheme used by the published figures and the GeoJSON overlays.
# Importing it (rather than repeating a palette here) means a cell type keeps
# the same colour in this panel, in the exported figures and in Tab 7.
try:
    from spacemap.acc_utils import (
        DEFAULT_IGNORED_COLOR,
        DEFAULT_MATCH_COLOR,
        DEFAULT_MISMATCH_COLOR,
        build_color_mapping,
    )
except Exception:  # spacemap unavailable (bare unit-test context)
    DEFAULT_MATCH_COLOR, DEFAULT_MISMATCH_COLOR, DEFAULT_IGNORED_COLOR = "#00693C", "#D62728", "#D9D9D9"

    def build_color_mapping(labels, existing=None, **_kwargs):  # type: ignore[misc]
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        mapping = dict(existing or {})
        for index, label in enumerate(sorted({str(v) for v in labels})):
            mapping.setdefault(label, palette[index % len(palette)])
        return mapping


def _sanitize_key(text: str) -> str:
    """Matches actual_tissue._sanitize_key so colour keys line up exactly."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_") or "item"


def _numeric_columns(frame: pd.DataFrame | None, fallback: list[str]) -> list[str]:
    if frame is None or frame.empty:
        return list(fallback)
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    return numeric or list(fallback)


def apply_cell_filters(
    frame: pd.DataFrame | None,
    filter_rules: list[dict[str, Any]],
) -> pd.DataFrame | None:
    """Apply AND-combined cell filters, always excluding missing rule values."""
    if frame is None or frame.empty or not filter_rules:
        return frame

    mask = pd.Series(True, index=frame.index, dtype=bool)
    missing_tokens = {"", "nan", "none", "null", "<na>"}
    for rule in filter_rules:
        column = str(rule.get("column") or "")
        if not column or column not in frame.columns:
            continue

        series = frame[column]
        text = series.astype(str).str.strip()
        is_missing = series.isna() | text.str.casefold().isin(missing_tokens)
        filter_type = str(rule.get("type") or "Categorical")

        if filter_type == "Categorical":
            allowed = {str(value) for value in (rule.get("values") or [])}
            row_match = text.isin(allowed) if allowed else pd.Series(False, index=frame.index)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            is_missing |= numeric.isna()
            operator = str(rule.get("operator") or "Between")
            if operator == "Above (>=)":
                row_match = numeric >= float(rule.get("threshold", 0.0))
            elif operator == "Below (<=)":
                row_match = numeric <= float(rule.get("threshold", 0.0))
            elif operator == "Equal (==)":
                threshold = float(rule.get("threshold", 0.0))
                tolerance = max(abs(threshold) * 1e-9, 1e-12)
                row_match = (numeric - threshold).abs() <= tolerance
            else:
                minimum = float(rule.get("minimum", float("-inf")))
                maximum = float(rule.get("maximum", float("inf")))
                row_match = numeric.between(minimum, maximum, inclusive="both")

        mask &= row_match.fillna(False) & ~is_missing
    return frame.loc[mask].copy()


def _cell_label_token(value: Any) -> str | None:
    """Normalize integer-like CSV/GeoJSON cell labels to the same token."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "<na>"}:
        return None
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError, OverflowError):
        pass
    return text


def _frame_cell_label_column(frame: pd.DataFrame | None) -> str | None:
    if frame is None:
        return None
    by_casefold = {str(column).casefold(): str(column) for column in frame.columns}
    for candidate in ("label", "cell_id", "cellid", "cell_label", "object_id", "objectid"):
        if candidate in by_casefold:
            return by_casefold[candidate]
    return None


def _filter_features_to_frame_labels(
    features: list[dict[str, Any]],
    frame: pd.DataFrame | None,
    *,
    frame_label_column: str,
    feature_label_key: str = "label",
) -> tuple[list[dict[str, Any]], int]:
    """Keep segmentation polygons whose cell labels survived dataframe filters."""
    if frame is None or frame.empty or frame_label_column not in frame.columns:
        return [], 0
    allowed = {
        token
        for token in frame[frame_label_column].map(_cell_label_token)
        if token is not None
    }
    retained = [
        feature
        for feature in features
        if _cell_label_token((feature.get("properties", {}) or {}).get(feature_label_key)) in allowed
    ]
    return retained, len(retained)


def _padded_feature_bounds(
    features: list[dict[str, Any]],
    padding_percent: float,
    full_shape: tuple[int, int] | None = None,
) -> tuple[float, float, float, float] | None:
    """Return a padded box containing every surviving cell geometry."""
    bounds = [oc.geometry_bounds(feature.get("geometry")) for feature in features]
    bounds = [bound for bound in bounds if bound is not None]
    if not bounds:
        return None
    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)
    padding = max(float(padding_percent), 0.0) / 100.0
    pad_x = max((max_x - min_x) * padding, 1.0 if padding else 0.0)
    pad_y = max((max_y - min_y) * padding, 1.0 if padding else 0.0)
    padded = (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)
    if full_shape is None:
        return padded
    full_height, full_width = full_shape
    return (
        max(0.0, padded[0]),
        max(0.0, padded[1]),
        min(float(full_width), padded[2]),
        min(float(full_height), padded[3]),
    )


_FIGURE_MIME = {"png": "image/png", "tiff": "image/tiff", "pdf": "application/pdf"}
_CATEGORY_PALETTE_STATE_VERSION = 1


def _initial_category_colors(
    labels: list[str],
    stored_colors: dict[str, Any],
    *,
    migrate_legacy_all_black: bool = False,
) -> dict[str, str]:
    """Return durable category colours, repairing the legacy all-black state.

    Older versions created color pickers without an explicit value. Streamlit
    could consequently recreate every picker at its ``#000000`` default after
    an unrelated widget changed. Treat that specific all-black palette as
    uninitialised once; black choices in a mixed/custom palette are preserved.
    """
    valid: dict[str, str] = {}
    for label in labels:
        value = stored_colors.get(label)
        try:
            valid[label] = "#%02x%02x%02x" % oc.parse_hex_color(str(value))
        except (TypeError, ValueError):
            continue

    if (
        migrate_legacy_all_black
        and len(valid) == len(labels)
        and valid
        and all(color == "#000000" for color in valid.values())
    ):
        valid = {}
    return build_color_mapping(labels, existing=valid)


# ── cached helpers ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=8)
def _discover(roi_dir: str, seg_dir: str, ann_dir: str, token: str) -> list[dict[str, Any]]:
    """``token`` participates in the cache key so a rescan can force a refresh."""
    return oc.discover_overlay_sets(roi_dir, seg_dir, ann_dir)


@st.cache_data(show_spinner=False, max_entries=64)
def _channels(image_path: str, mtime: float) -> list[str]:
    return oc.read_ome_channel_names(image_path)


@st.cache_data(show_spinner=False, max_entries=64)
def _levels(image_path: str, mtime: float) -> list[tuple[int, int]]:
    return oc.pyramid_level_shapes(image_path)


@st.cache_data(show_spinner=False, max_entries=64)
def _ann_classes(path: str, mtime: float) -> list[dict[str, Any]]:
    return oc.annotation_classes(path)


def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0


def _state(name: str, default: Any) -> Any:
    key = _K + name
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _comparison_columns_to_render(
    cell_colour_mode: str,
    compare_columns: list[str],
) -> list[str]:
    """Return ordered comparison jobs; non-comparison modes need one job."""
    if cell_colour_mode != "Match vs reference":
        return [""]
    return list(dict.fromkeys(str(column) for column in compare_columns if str(column)))


def _comparison_filename_suffix(reference_column: str, compare_column: str) -> str:
    """Unambiguous filename suffix for one compared-vs-reference figure."""
    if not compare_column:
        return ""
    reference = _sanitize_key(reference_column) if reference_column else "reference"
    compared = _sanitize_key(compare_column)
    return f"_{compared}_vs_{reference}"


def _render_cell_filter_controls(
    roi_rows: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], bool, float]:
    """Render Actual-Tissue-style filter rows for the current ROI."""
    st.markdown("##### Cell filtration")
    has_rows = roi_rows is not None and not roi_rows.empty
    enable_filters = st.checkbox(
        "Filter cells by CSV column values",
        value=bool(_state("filter_enable", False)),
        key=_K + "filter_enable",
        disabled=not has_rows,
        help="Multiple filters are combined with AND logic before cells are rendered.",
    )
    if not has_rows:
        st.caption("Unified CSV rows are required to filter cells.")
        return [], False, 8.0
    if not enable_filters:
        return [], False, float(_state("filter_zoom_padding", 8.0))

    st.caption(
        "Missing/NaN values in each selected filter column are always excluded. "
        "Multiple filters are combined with AND logic."
    )
    filter_count = int(
        st.number_input(
            "Number of Filters",
            min_value=1,
            max_value=10,
            value=int(_state("filter_count", 1)),
            step=1,
            key=_K + "filter_count",
        )
    )
    columns = [str(column) for column in roi_rows.columns]
    rules: list[dict[str, Any]] = []

    for index in range(filter_count):
        st.markdown(f"**Filter {index + 1}**")
        first, second = st.columns(2)
        column_key = _K + f"filter_col_{index}"
        if st.session_state.get(column_key, "") not in ([""] + columns):
            st.session_state[column_key] = ""
        filter_column = first.selectbox("Column", options=[""] + columns, key=column_key)
        if not filter_column:
            second.caption("Select a column to configure this filter.")
            continue

        series = roi_rows[filter_column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_values = numeric.dropna()
        inferred_type = (
            "Continuous"
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 15
            else "Categorical"
        )
        source_key = _K + f"filter_source_{index}"
        type_key = _K + f"filter_type_{index}"
        values_key = _K + f"filter_vals_{index}"
        operator_key = _K + f"filter_op_{index}"
        minimum_key = _K + f"filter_min_{index}"
        maximum_key = _K + f"filter_max_{index}"
        threshold_key = _K + f"filter_threshold_{index}"

        unique_values = sorted(
            {
                str(value).strip()
                for value in series.dropna().unique()
                if str(value).strip().casefold() not in {"", "nan", "none", "null", "<na>"}
            },
            key=str.casefold,
        )
        data_min = float(numeric_values.min()) if not numeric_values.empty else 0.0
        data_max = float(numeric_values.max()) if not numeric_values.empty else 1.0
        if st.session_state.get(source_key) != filter_column:
            st.session_state[source_key] = filter_column
            st.session_state[type_key] = inferred_type
            st.session_state[values_key] = list(unique_values)
            st.session_state[operator_key] = "Between"
            st.session_state[minimum_key] = data_min
            st.session_state[maximum_key] = data_max
            st.session_state[threshold_key] = (data_min + data_max) / 2.0

        if st.session_state.get(type_key) not in {"Categorical", "Continuous"}:
            st.session_state[type_key] = inferred_type
        filter_type = second.selectbox(
            "Type",
            options=["Categorical", "Continuous"],
            key=type_key,
        )

        if filter_type == "Categorical":
            st.session_state[values_key] = [
                value for value in st.session_state.get(values_key, []) if value in unique_values
            ]
            selected_values = st.multiselect(
                "Allowed Values",
                options=unique_values,
                key=values_key,
                help="Only cells matching one of these values are retained.",
            )
            rules.append(
                {"column": filter_column, "type": "Categorical", "values": list(selected_values)}
            )
        else:
            if numeric_values.empty:
                st.warning(f"`{filter_column}` has no numeric values; this filter will retain no cells.")
            if st.session_state.get(operator_key) not in {
                "Between", "Above (>=)", "Below (<=)", "Equal (==)"
            }:
                st.session_state[operator_key] = "Between"
            operator = st.selectbox(
                "Condition",
                options=["Between", "Above (>=)", "Below (<=)", "Equal (==)"],
                key=operator_key,
            )
            step = max((data_max - data_min) / 100.0, 1e-6)
            if operator == "Between":
                range_col_1, range_col_2 = st.columns(2)
                minimum = range_col_1.number_input(
                    "Minimum",
                    value=float(st.session_state.get(minimum_key, data_min)),
                    step=step,
                    key=minimum_key,
                )
                maximum = range_col_2.number_input(
                    "Maximum",
                    value=float(st.session_state.get(maximum_key, data_max)),
                    step=step,
                    key=maximum_key,
                )
                rules.append(
                    {
                        "column": filter_column,
                        "type": "Continuous",
                        "operator": operator,
                        "minimum": float(minimum),
                        "maximum": float(maximum),
                    }
                )
            else:
                threshold = st.number_input(
                    "Threshold",
                    value=float(st.session_state.get(threshold_key, (data_min + data_max) / 2.0)),
                    step=step,
                    key=threshold_key,
                )
                rules.append(
                    {
                        "column": filter_column,
                        "type": "Continuous",
                        "operator": operator,
                        "threshold": float(threshold),
                    }
                )

    zoom_col_1, zoom_col_2 = st.columns(2)
    auto_zoom = zoom_col_1.checkbox(
        "Auto-zoom to filtered cells",
        value=bool(_state("filter_auto_zoom", True)),
        key=_K + "filter_auto_zoom",
        help="Fits the viewport to all retained cell shapes; excluded cells do not affect the view.",
    )
    zoom_padding = float(
        zoom_col_2.slider(
            "Zoom padding (%)",
            min_value=0,
            max_value=50,
            value=int(_state("filter_zoom_padding", 8)),
            step=1,
            key=_K + "filter_zoom_padding",
            disabled=not auto_zoom,
        )
    )
    return rules, bool(auto_zoom), zoom_padding


# ── panel ────────────────────────────────────────────────────────────────────

def render_cell_visualization(
    project_root: Path,
    unified_csv_path: str,
    available_columns: list[str],
    cached_params: dict[str, Any],
    viz_settings: dict[str, Any],
    loaded_df: pd.DataFrame | None = None,
) -> None:
    st.markdown("### Cell & Tissue Overlay Figures")
    st.caption(
        "Compose a publication figure from three independent layers: the ROI image, "
        "tissue annotation polygons, and cells. Any layer may be missing -- a figure "
        "still renders from whatever is available."
    )

    mode = st.radio(
        "Panel",
        options=["Overlay figures", "Legacy cell tiles"],
        horizontal=True,
        key=_K + "panel_mode",
        help=(
            "**Overlay figures** composes whole-ROI publication figures.\n\n"
            "**Legacy cell tiles** is the previous per-cell thumbnail grid, kept unchanged."
        ),
    )
    if mode == "Legacy cell tiles":
        previous = {
            "viz_roi_dir": st.session_state.get("viz_roi_dir"),
            "viz_seg_dir": st.session_state.get("viz_seg_dir"),
        }
        st.session_state["viz_roi_dir"] = viz_settings.get("roi_dir", "")
        st.session_state["viz_seg_dir"] = viz_settings.get("seg_dir", "")
        try:
            _legacy_render_cell_visualization(
                project_root, unified_csv_path, available_columns, cached_params, loaded_df=loaded_df
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    st.session_state.pop(key, None)
                else:
                    st.session_state[key] = value
        return

    _render_overlay_panel(project_root, available_columns, viz_settings, loaded_df)


def _render_overlay_panel(
    project_root: Path,
    available_columns: list[str],
    viz_settings: dict[str, Any],
    loaded_df: pd.DataFrame | None,
) -> None:
    roi_dir = str(viz_settings.get("roi_dir") or "")
    seg_dir = str(viz_settings.get("seg_dir") or "")
    ann_dir = str(viz_settings.get("ann_dir") or "")

    # ── 1. file matching ────────────────────────────────────────────────────
    st.markdown("#### 1. Matched files")
    st.caption(
        "ROI images, cell segmentations and tissue annotations are paired by ROI key "
        "(the file name with producer suffixes such as `.ome`, `_cell_boundaries` and "
        "`_merged` removed). **complete** has all three, **partial** two, **single** one."
    )
    cols = st.columns([3, 3, 3, 1])
    cols[0].caption(f"ROI images: `{roi_dir or '(not set)'}`")
    cols[1].caption(f"Segmentation: `{seg_dir or '(not set)'}`")
    cols[2].caption(f"Annotations: `{ann_dir or '(not set)'}`")
    if cols[3].button("Rescan", key=_K + "rescan_btn", help="Re-read the three directories."):
        st.session_state[_T + "scan_token"] = str(pd.Timestamp.now())

    if not any([roi_dir, seg_dir, ann_dir]):
        st.info("Set at least one directory in **Visual Validation Settings** above.")
        return

    records = _discover(roi_dir, seg_dir, ann_dir, str(_state("scan_token", "initial")))
    if not records:
        st.warning("No files matched. Check the directories in Visual Validation Settings.")
        return

    frame = pd.DataFrame(records)
    counts = frame["status"].value_counts().to_dict()
    metric = st.columns(4)
    metric[0].metric("ROI keys", f"{len(frame):,}")
    metric[1].metric("Complete (3 layers)", f"{counts.get('complete', 0):,}")
    metric[2].metric("Partial (2)", f"{counts.get('partial', 0):,}")
    metric[3].metric("Single (1)", f"{counts.get('single', 0):,}")

    show = st.selectbox(
        "Show",
        ["All", "Complete only", "Missing cells", "Missing tissue", "Missing image"],
        key=_K + "match_filter",
    )
    view = frame
    if show == "Complete only":
        view = frame[frame["status"] == "complete"]
    elif show == "Missing cells":
        view = frame[~frame["has_cells"]]
    elif show == "Missing tissue":
        view = frame[~frame["has_tissue"]]
    elif show == "Missing image":
        view = frame[~frame["has_image"]]
    st.dataframe(
        view[["roi", "has_image", "has_cells", "has_tissue", "status"]],
        width="stretch", hide_index=True, height=240,
    )

    # ── 2. ROI selection ────────────────────────────────────────────────────
    st.markdown("#### 2. ROI")
    options = view["roi"].tolist() or frame["roi"].tolist()
    saved = st.session_state.get(_K + "roi")
    roi = st.selectbox(
        "ROI to render",
        options=options,
        index=options.index(saved) if saved in options else 0,
        key=_K + "roi",
    )
    record = frame[frame["roi"] == roi].iloc[0].to_dict()
    st.caption(
        f"image: {'yes' if record['has_image'] else 'no'} | "
        f"cells: {'yes' if record['has_cells'] else 'no'} | "
        f"tissue: {'yes' if record['has_tissue'] else 'no'}"
    )

    tabs = st.tabs(["Base image", "Tissue polygons", "Cells", "Title & export"])

    # ── base layer ──────────────────────────────────────────────────────────
    with tabs[0]:
        if record["has_image"]:
            names = _channels(record["image"], _mtime(record["image"]))
            shapes = _levels(record["image"], _mtime(record["image"]))
            st.caption(
                f"{len(names)} channels; pyramid levels: "
                + ", ".join(f"L{i} {w}x{h}" for i, (h, w) in enumerate(shapes))
            )
            base_source = st.selectbox(
                "Base layer", ["ROI image", "Blank canvas"], key=_K + "base_source",
                help="A blank canvas renders cells and annotations alone, which is the only "
                     "option when an ROI image is unavailable.",
            )
            default_markers = [m for m in oc.default_marker_selection(names) if m in names]
            if _K + "markers" not in st.session_state:
                st.session_state[_K + "markers"] = default_markers
            markers = st.multiselect(
                "Markers to composite", options=names, key=_K + "markers",
                help="Each marker is tinted and screen-blended, so overlapping signal stays "
                     "visible instead of the last one drawn hiding the rest. Defaults to DAPI, "
                     "Vimentin and any keratin found in the panel.",
            )
            colour_cols = st.columns(min(max(len(markers), 1), 6))
            marker_colors: dict[str, str] = {}
            for index, marker in enumerate(markers[:6]):
                with colour_cols[index % len(colour_cols)]:
                    marker_colors[marker] = st.color_picker(
                        marker, value=oc.DEFAULT_MARKER_COLORS[index % len(oc.DEFAULT_MARKER_COLORS)],
                        key=_K + f"mcol_{oc.roi_key(marker)}_{index}",
                    )
            c1, c2, c3 = st.columns(3)
            level_choice = c1.selectbox(
                "Pyramid level", ["auto"] + [f"L{i}" for i in range(len(shapes))],
                key=_K + "level",
                help="Level 0 is full resolution; each level halves it. 'auto' picks the "
                     "highest-resolution level that fits the output size below, which is "
                     "usually what you want -- lower levels load much faster.",
            )
            max_size = c2.number_input(
                "Max output size (px)", 400, 8000, int(_state("max_size", 2000)), 200,
                key=_K + "max_size",
            )
            contrast = c3.slider(
                "Contrast percentiles", 0.0, 100.0, (1.0, 99.5), 0.5, key=_K + "contrast",
                help="Low/high percentiles used to stretch each channel. Widen the range if "
                     "the image looks washed out; narrow it to bring up dim signal.",
            )
        else:
            st.info("No ROI image for this key -- rendering on a blank canvas.")
            base_source, markers, marker_colors = "Blank canvas", [], {}
            level_choice, max_size, contrast = "auto", int(_state("max_size", 2000)), (1.0, 99.5)
        background = st.color_picker("Blank-canvas background", value=_state("bg", "#000000"), key=_K + "bg")

    # ── tissue layer ────────────────────────────────────────────────────────
    with tabs[1]:
        tissue_classes: list[str] = []
        tissue_class_colors: dict[str, str] = {}
        tissue_use_embedded = True
        tissue_color, tissue_alpha, tissue_outline_w = "#00D9FF", 55, 2
        crop_to_tissue = mask_outside = clip_cells_to_tissue = False
        if record["has_tissue"]:
            classes = _ann_classes(record["tissue"], _mtime(record["tissue"]))
            st.caption(
                "One row per annotation class. A class is often a single MultiPolygon "
                "holding many separate pieces, so 'polygons' can far exceed 'features'."
            )
            names_all = [c["name"] for c in classes]
            preferred = [n for n in names_all if n.casefold() not in {"bkg", "background"}]
            if _K + "tclasses" not in st.session_state:
                st.session_state[_K + "tclasses"] = preferred
            tissue_classes = st.multiselect(
                "Polygons to display", options=names_all, key=_K + "tclasses",
                help="Background-like classes are deselected by default because they usually "
                     "cover the whole frame and hide everything beneath.",
            )
            tissue_use_embedded = st.checkbox(
                "Use the colours saved in the annotation file", value=_state("t_embed", True),
                key=_K + "t_embed",
                help="Annotation files carry the colour the annotator chose "
                     "(classification.color). Untick to set each class yourself below.",
            )

            # Apply a pending reset before the pickers exist: Streamlit refuses a
            # write to a widget key once that widget has been created this run.
            if st.session_state.pop(_T + "reset_tissue_colors", False):
                for entry in classes:
                    st.session_state[_K + f"tcol_{_sanitize_key(entry['name'])}"] = entry["color"]

            st.caption("Click a swatch to change that class's colour.")
            picker_cols = st.columns(min(max(len(classes), 1), 4))
            for index, entry in enumerate(classes):
                widget_key = _K + f"tcol_{_sanitize_key(entry['name'])}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = entry["color"]
                with picker_cols[index % len(picker_cols)]:
                    chosen = st.color_picker(
                        f"{entry['name']}  ({entry['n_polygons']} polys)",
                        key=widget_key, disabled=tissue_use_embedded,
                    )
                    tissue_class_colors[entry["name"]] = chosen
            if st.button("Reset to the file's colours", key=_K + "t_reset_btn",
                         disabled=tissue_use_embedded):
                st.session_state[_T + "reset_tissue_colors"] = True
                st.rerun()

            t2, t3 = st.columns(2)
            tissue_alpha = t2.slider("Fill opacity", 0, 255, int(_state("t_alpha", 55)), key=_K + "t_alpha")
            tissue_outline_w = t3.slider("Outline width", 0, 8, int(_state("t_width", 2)), key=_K + "t_width")
            crop_to_tissue = st.checkbox(
                "Crop the figure to the selected polygons", value=False, key=_K + "t_crop",
                help="Zooms the figure to the bounding box of the selected classes.",
            )
            clip_cells_to_tissue = crop_to_tissue and st.checkbox(
                "Clip each cell to its containing selected polygon",
                value=False,
                key=_K + "t_clip_cells",
                help=(
                    "Assigns each cell by its centroid, then intersects its shape with that "
                    "specific tissue annotation. Cells outside all selected annotations are hidden."
                ),
            )
            mask_outside = st.checkbox(
                "Hide everything outside the selected polygons", value=False, key=_K + "t_mask",
                help="Blanks the area outside the annotation, so a single tissue piece can be "
                     "shown in isolation even when its outline itself is not drawn.",
            )
        else:
            st.info("No tissue annotation for this ROI. Cells and image still render.")

    # ── cell layer ──────────────────────────────────────────────────────────
    with tabs[2]:
        _roi_rows = _cells_for_roi(loaded_df, viz_settings, record["roi"])
        _n_rows = 0 if _roi_rows is None else len(_roi_rows)
        if record["has_cells"]:
            st.caption(f"Segmentation file present. Unified CSV rows for this ROI: {_n_rows:,}.")
        elif _n_rows:
            st.caption(f"No segmentation file, but {_n_rows:,} CSV rows -- Voronoi and Circles are available.")
        else:
            st.warning(
                f"ROI `{record['roi']}` has no segmentation file and no rows in the unified CSV, "
                "so no cell layer can be drawn. The image and any annotations still render."
            )
        cell_filter_rules, auto_zoom_filtered_cells, zoom_padding = _render_cell_filter_controls(_roi_rows)
        filtered_roi_rows = apply_cell_filters(_roi_rows, cell_filter_rules)
        if cell_filter_rules:
            filtered_count = 0 if filtered_roi_rows is None else len(filtered_roi_rows)
            st.caption(f"Filter preview: {filtered_count:,} / {_n_rows:,} cells retained for this ROI.")
            if filtered_count == 0:
                st.warning("No cells match the active filters. The image and tissue layers can still render.")

        sources = (["Segmentation polygons"] if record["has_cells"] else []) + ["Voronoi", "Circles", "None"]
        s1, s2 = st.columns(2)
        cell_source = s1.selectbox(
            "Cell representation", sources, key=_K + "cell_source",
            help=(
                "**Segmentation polygons** draw each cell's real measured outline -- the most "
                "faithful option, available only where a segmentation file exists.\n\n"
                "**Voronoi** divides space so every point belongs to its nearest cell centre, "
                "then trims each tile to a maximum radius. Useful when no segmentation exists, "
                "but the shapes are constructed, not measured.\n\n"
                "**Circles** draw a fixed-radius disc at each centroid: simplest and fastest, "
                "and honest about being a schematic."
            ),
        )
        cell_alpha = s2.slider("Cell fill opacity", 0, 255, int(_state("c_alpha", 170)), key=_K + "c_alpha")
        voronoi_radius = (
            st.slider(
                "Voronoi / circle radius (px)", 5, 200, int(_state("c_radius", 30)), key=_K + "c_radius",
                help="For Voronoi this caps how far a tile may extend from its cell centre, so an "
                     "isolated cell cannot claim a huge empty region. For circles it is the disc radius.",
            )
            if cell_source in {"Voronoi", "Circles"} else 30
        )

        st.markdown("##### Cell colouring")
        cell_colour_mode = st.radio(
            "Mode",
            ["Unique per cell", "Categorical column", "Continuous marker", "Match vs reference", "Fixed"],
            horizontal=True, key=_K + "c_mode",
            help=(
                "**Unique per cell** - a stable distinct hue per cell; best for showing boundaries.\n\n"
                "**Categorical column** - colour by a label column such as a phenotype.\n\n"
                "**Continuous marker** - colour by a numeric column on a low-to-high gradient.\n\n"
                "**Match vs reference** - compare a method column against a reference column and "
                "colour agreement and disagreement.\n\n"
                "**Fixed** - one colour for every cell."
            ),
        )
        column_options = [""] + list(available_columns)
        category_column = continuous_column = reference_column = ""
        compare_columns: list[str] = []
        category_colors: dict[str, str] = {}
        low_color, mid_color, high_color = "#2C7BB6", None, "#D7191C"
        clip_pcts = (2.0, 98.0)
        match_color, mismatch_color, ignored_color = "#00B050", "#E03131", oc.NEUTRAL_GREY
        ignore_values: list[str] = []
        cell_fixed = _state("c_fixed", "#FF5B40")

        if cell_colour_mode == "Fixed":
            cell_fixed = st.color_picker("Cell colour", value=cell_fixed, key=_K + "c_fixed")

        elif cell_colour_mode == "Categorical column":
            category_column = st.selectbox("Label column", column_options, key=_K + "c_cat_col")
            if (
                category_column
                and filtered_roi_rows is not None
                and category_column in filtered_roi_rows.columns
                and not filtered_roi_rows.empty
            ):
                labels = sorted({str(v) for v in filtered_roi_rows[category_column].dropna().unique()})
                st.caption(
                    f"{len(labels)} categories. Colours are shared with the Actual Tissue panel "
                    "(same saved keys), so setting them in either place keeps one scheme."
                )
                house = build_color_mapping(labels)
                widget_keys = {
                    label: f"tab9_actual_cell_color_val_{_sanitize_key(label)}"
                    for label in labels
                }
                palette_version_key = _K + "c_palette_state_version"
                category_colors = _initial_category_colors(
                    labels,
                    {
                        label: st.session_state.get(widget_key)
                        for label, widget_key in widget_keys.items()
                    },
                    migrate_legacy_all_black=(
                        st.session_state.get(palette_version_key, 0)
                        < _CATEGORY_PALETTE_STATE_VERSION
                    ),
                )
                # Applying a reset must happen BEFORE the pickers are created:
                # Streamlit forbids writing a widget's key once that widget has
                # been instantiated in the same run.
                if st.session_state.pop(_T + "reset_colors", False):
                    for label in labels:
                        category_colors[label] = house.get(label, "#777777")
                        st.session_state[widget_keys[label]] = category_colors[label]
                else:
                    for label in labels:
                        st.session_state[widget_keys[label]] = category_colors[label]
                st.session_state[palette_version_key] = _CATEGORY_PALETTE_STATE_VERSION
                grid = st.columns(4)
                for index, label in enumerate(labels):
                    # Reuse the actual-tissue key so both panels read and write a
                    # single saved palette, seeded from the house scheme so known
                    # cell types match the published figures out of the box.
                    widget_key = widget_keys[label]
                    with grid[index % 4]:
                        category_colors[label] = st.color_picker(
                            label,
                            value=category_colors[label],
                            key=widget_key,
                        )
                if st.button("Reset to house colour scheme", key=_K + "c_reset_colors_btn"):
                    # Only flag it here; the values are written at the top of the
                    # next run, before the pickers exist.
                    st.session_state[_T + "reset_colors"] = True
                    st.rerun()
            elif category_column:
                st.info("No cells remain for this ROI after the active filters, so categories cannot be listed.")

        elif cell_colour_mode == "Continuous marker":
            numeric_options = [""] + _numeric_columns(filtered_roi_rows, available_columns)
            continuous_column = st.selectbox("Numeric column", numeric_options, key=_K + "c_cont_col")
            g1, g2, g3, g4 = st.columns(4)
            use_three = g1.checkbox(
                "Three stops", value=_state("c_three", False), key=_K + "c_three",
                help="Two stops run straight from low to high. Three stops add a middle colour, "
                     "which is the right choice when the middle of the range is meaningful "
                     "(a neutral level with depletion below and enrichment above).",
            )
            low_color = g2.color_picker("Low", value=_state("c_low", "#2C7BB6"), key=_K + "c_low")
            mid_color = g3.color_picker("Mid", value=_state("c_mid", "#FFFFBF"), key=_K + "c_mid") if use_three else None
            high_color = g4.color_picker("High", value=_state("c_high", "#D7191C"), key=_K + "c_high")
            clip_pcts = st.slider(
                "Clip percentiles", 0.0, 100.0, (2.0, 98.0), 0.5, key=_K + "c_clip",
                help="Values outside this percentile range are pinned to the end colours, so a "
                     "few extreme cells cannot flatten the gradient for everything else.",
            )

        elif cell_colour_mode == "Match vs reference":
            m1, m2 = st.columns(2)
            reference_column = m1.selectbox("Reference column", column_options, key=_K + "c_ref_col")
            comparison_options = [column for column in available_columns if column != reference_column]
            compare_key = _K + "c_cmp_cols"
            if compare_key not in st.session_state:
                previous_compare = st.session_state.get(_K + "c_cmp_col", "")
                st.session_state[compare_key] = (
                    [previous_compare] if previous_compare in comparison_options else []
                )
            else:
                current_comparisons = st.session_state.get(compare_key, [])
                if not isinstance(current_comparisons, list):
                    current_comparisons = [current_comparisons]
                st.session_state[compare_key] = [
                    column for column in current_comparisons if column in comparison_options
                ]
            compare_columns = m2.multiselect(
                "Columns to compare",
                options=comparison_options,
                key=compare_key,
                help=(
                    "One match/mismatch figure is generated for every selected column. "
                    "The figures are displayed vertically and exported separately."
                ),
            )
            c1, c2, c3 = st.columns(3)
            match_color = c1.color_picker("Match", value=_state("c_match", DEFAULT_MATCH_COLOR), key=_K + "c_match")
            mismatch_color = c2.color_picker("Mismatch", value=_state("c_mismatch", DEFAULT_MISMATCH_COLOR), key=_K + "c_mismatch")
            ignored_color = c3.color_picker("Ignored", value=_state("c_ignored", DEFAULT_IGNORED_COLOR), key=_K + "c_ignored")
            candidates: list[str] = []
            if filtered_roi_rows is not None:
                for column in [reference_column, *compare_columns]:
                    if column and column in filtered_roi_rows.columns:
                        candidates.extend(str(v) for v in filtered_roi_rows[column].dropna().unique())
            ignore_values = st.multiselect(
                "Labels to ignore", sorted(set(candidates)),
                default=[v for v in ("Others", "Unassigned", "Unknown") if v in set(candidates)],
                key=_K + "c_ignore",
                help="A cell carrying one of these labels is neither agreement nor disagreement, "
                     "so it is drawn in the ignored colour and excluded from the agreement rate.",
            )

    # ── title & export ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("##### Layer order")
        st.caption(
            "The base image is always at the back. These two decide what sits on top: "
            "the default draws annotations over the image and cells in front of both, which "
            "is what a reader expects. Reversing it lets a translucent annotation tint the cells."
        )
        order_choice = st.radio(
            "Draw order (back to front)",
            ["Image -> Tissue -> Cells", "Image -> Cells -> Tissue"],
            key=_K + "layer_order", horizontal=True,
        )
        layer_order = ("tissue", "cells") if order_choice.endswith("Cells") else ("cells", "tissue")

        st.markdown("##### Legend")
        l1, l2, l3 = st.columns(3)
        show_legend = l1.checkbox("Draw legend on the figure", value=_state("legend_on", True), key=_K + "legend_on")
        legend_place = l2.selectbox(
            "Placement",
            ["right", "left", "top", "bottom",
             "overlay top-left", "overlay top-right", "overlay bottom-left", "overlay bottom-right"],
            key=_K + "legend_place",
            help="**right / left / top / bottom** add a margin beside the figure so no tissue is "
                 "covered -- the right choice for a publication panel. The **overlay** options "
                 "draw the legend on the image itself to save space.",
        )
        legend_bg = l3.color_picker("Legend background", value=_state("legend_bg", "#FFFFFF"), key=_K + "legend_bg")

        st.markdown("##### Title")
        t1, t2 = st.columns(2)
        show_title = t1.checkbox("Draw a title on the figure", value=_state("title_on", True), key=_K + "title_on")
        title_parts = t2.multiselect(
            "Include in the title", ["ROI name", "Cell count", "Layers present", "Markers"],
            default=_state("title_parts", ["ROI name", "Cell count"]), key=_K + "title_parts",
        )
        custom_title = st.text_input("Extra title text", value=_state("title_text", ""), key=_K + "title_text")
        e1, e2 = st.columns(2)
        export_dpi = e1.number_input("Export DPI", 72, 1200, int(_state("dpi", 300)), 50, key=_K + "dpi")
        export_fmt = e2.selectbox("Format", list(_FIGURE_MIME), key=_K + "fmt")

    # One settings bundle drives both the single render and the batch, so a
    # batch can never silently use different options from the preview above.
    layer_settings: dict[str, Any] = dict(
        loaded_df=loaded_df, viz_settings=viz_settings,
        base_source=base_source, markers=markers, marker_colors=marker_colors,
        level_choice=level_choice, max_size=int(max_size), contrast=contrast,
        background=background,
        tissue_classes=tissue_classes, tissue_use_embedded=tissue_use_embedded,
        tissue_color=tissue_color, tissue_class_colors=tissue_class_colors,
        tissue_alpha=int(tissue_alpha), tissue_outline_w=int(tissue_outline_w),
        crop_to_tissue=crop_to_tissue, mask_outside=mask_outside,
        clip_cells_to_tissue=clip_cells_to_tissue,
        cell_filter_rules=cell_filter_rules,
        auto_zoom_filtered_cells=auto_zoom_filtered_cells,
        zoom_padding=zoom_padding,
        cell_source=cell_source, cell_alpha=int(cell_alpha),
        cell_colour_mode=cell_colour_mode, cell_fixed=cell_fixed,
        radius=float(voronoi_radius),
        category_column=category_column, category_colors=category_colors,
        continuous_column=continuous_column, low_color=low_color,
        mid_color=mid_color, high_color=high_color, clip_pcts=clip_pcts,
        reference_column=reference_column, compare_column="",
        match_color=match_color, mismatch_color=mismatch_color,
        ignored_color=ignored_color, ignore_values=ignore_values,
        layer_order=layer_order, show_legend=show_legend,
        legend_place=legend_place, legend_bg=legend_bg,
        show_title=show_title, title_parts=title_parts, custom_title=custom_title,
    )

    comparison_jobs = _comparison_columns_to_render(cell_colour_mode, compare_columns)
    missing_comparison = cell_colour_mode == "Match vs reference" and (
        not reference_column or not comparison_jobs
    )
    if missing_comparison:
        st.info("Select a reference column and at least one column to compare.")

    render = st.button(
        "Render figures" if len(comparison_jobs) > 1 else "Render figure",
        type="primary",
        key=_K + "render_btn",
        disabled=missing_comparison,
    )
    if render:
        try:
            results: list[dict[str, Any]] = []
            with st.spinner(f"Rendering {len(comparison_jobs)} figure(s) for {roi} ..."):
                for compare_column in comparison_jobs:
                    current_settings = dict(layer_settings, compare_column=compare_column)
                    image, info = _build_figure(record=record, **current_settings)
                    buffer = io.BytesIO()
                    save_kwargs = (
                        {"dpi": (int(export_dpi), int(export_dpi))}
                        if export_fmt in {"png", "tiff"}
                        else {}
                    )
                    image.convert("RGB").save(buffer, format=export_fmt.upper(), **save_kwargs)
                    suffix = _comparison_filename_suffix(reference_column, compare_column)
                    results.append(
                        {
                            "payload": buffer.getvalue(),
                            "info": info,
                            "name": f"{roi}{suffix}_overlay.{export_fmt}",
                            "comparison": compare_column,
                            "mime": _FIGURE_MIME.get(export_fmt, "image/png"),
                        }
                    )
            st.session_state[_T + "last_results"] = results
        except Exception as exc:
            st.error(f"Rendering failed: {exc}")
            return

    results = st.session_state.get(_T + "last_results")
    if results is None:
        st.caption("Set the layers above, then **Render figure**. Rendering runs only on request.")
    for result_index, result in enumerate(results or []):
        if result.get("comparison"):
            result_reference = result.get("info", {}).get("reference_column", "")
            st.markdown(
                f"**Compared: `{result['comparison']}` · Reference: `{result_reference}`**"
            )
        st.image(result["payload"], width="stretch")
        info = result.get("info", {})
        if info:
            st.caption(" | ".join(f"{k}: {v}" for k, v in info.items()))
        st.download_button(
            f"Download figure ({result['name']})",
            data=result["payload"],
            file_name=result["name"],
            mime=result.get("mime", "image/png"),
            key=_K + f"download_figure_{result_index}_{_sanitize_key(result['name'])}",
        )

    # ── batch rendering ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Batch render many ROIs", expanded=False):
        st.caption(
            "Renders every selected ROI with **exactly the settings above** and writes one file "
            "per ROI/comparison pair. A layer that is missing for a given ROI is simply skipped, "
            "so a batch over mixed ROIs still produces each possible figure. One bad ROI never "
            "stops the run."
        )
        b1, b2 = st.columns([3, 2])
        scope = b1.selectbox(
            "Which ROIs",
            ["Currently filtered list", "Complete only (all 3 layers)", "All ROIs", "Choose explicitly"],
            key=_K + "batch_scope",
            help="'Currently filtered list' uses the same filter as the match table above.",
        )
        limit = b2.number_input(
            "Limit (0 = no limit)", 0, 10000, int(_state("batch_limit", 0)), 1, key=_K + "batch_limit",
            help="Render only the first N after sorting -- useful for a quick check before "
                 "committing to the whole set.",
        )

        if scope == "Currently filtered list":
            batch_rois = view["roi"].tolist()
        elif scope == "Complete only (all 3 layers)":
            batch_rois = frame.loc[frame["status"] == "complete", "roi"].tolist()
        elif scope == "All ROIs":
            batch_rois = frame["roi"].tolist()
        else:
            batch_rois = st.multiselect(
                "ROIs", options=frame["roi"].tolist(), default=[roi], key=_K + "batch_pick"
            )
        if limit:
            batch_rois = batch_rois[: int(limit)]

        default_out = str(Path(project_root) / "outputs" / "overlay_figures")
        out_dir = st.text_input("Output directory", value=_state("batch_out", default_out), key=_K + "batch_out")
        o1, o2 = st.columns(2)
        overwrite = o1.checkbox(
            "Overwrite existing files", value=_state("batch_overwrite", False), key=_K + "batch_overwrite",
            help="Off by default: an ROI whose file already exists is skipped, so an interrupted "
                 "batch can be resumed without redoing work.",
        )
        suffix = o2.text_input("Filename suffix", value=_state("batch_suffix", "_overlay"), key=_K + "batch_suffix")

        n_batch_figures = len(batch_rois) * len(comparison_jobs)
        st.caption(
            f"**{len(batch_rois)}** ROI(s), **{n_batch_figures}** figure(s) queued -> `{out_dir}`"
        )
        if st.button(
            "Run batch render",
            type="primary",
            key=_K + "batch_run_btn",
            disabled=not batch_rois or missing_comparison,
        ):
            _run_batch(
                rois=batch_rois, frame=frame, out_dir=out_dir, suffix=suffix,
                overwrite=bool(overwrite), export_fmt=export_fmt, export_dpi=int(export_dpi),
                layer_settings=layer_settings, compare_columns=comparison_jobs,
            )

        summary = st.session_state.get(_T + "batch_summary")
        if summary is not None and not summary.empty:
            done = int((summary["status"] == "rendered").sum())
            skipped = int((summary["status"] == "skipped").sum())
            failed = int((summary["status"] == "failed").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Rendered", f"{done:,}")
            m2.metric("Skipped (existing)", f"{skipped:,}")
            m3.metric("Failed", f"{failed:,}")
            st.dataframe(summary, width="stretch", hide_index=True, height=260)
            if failed:
                st.warning(
                    "Failed ROIs are listed above with the reason. The rest of the batch completed."
                )
            st.download_button(
                "Download batch manifest (CSV)",
                data=summary.to_csv(index=False).encode(),
                file_name="overlay_batch_manifest.csv",
                mime="text/csv",
                key=_K + "batch_manifest_download",
            )


def _run_batch(
    *,
    rois: list[str],
    frame: pd.DataFrame,
    out_dir: str,
    suffix: str,
    overwrite: bool,
    export_fmt: str,
    export_dpi: int,
    layer_settings: dict[str, Any],
    compare_columns: list[str],
) -> None:
    """Render every ROI/comparison pair, continuing past individual failures."""
    target = Path(out_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        st.error(f"Cannot create output directory: {exc}")
        return

    lookup = {row["roi"]: row for row in frame.to_dict("records")}
    rows: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Starting batch ...")
    status = st.empty()

    jobs = [(roi_name, compare_column) for roi_name in rois for compare_column in compare_columns]
    reference_column = str(layer_settings.get("reference_column") or "")

    for index, (roi_name, compare_column) in enumerate(jobs, start=1):
        job_label = f"{roi_name} · {compare_column}" if compare_column else roi_name
        progress.progress(
            (index - 1) / max(len(jobs), 1),
            text=f"{job_label} ({index}/{len(jobs)})",
        )
        record = lookup.get(roi_name)
        comparison_suffix = _comparison_filename_suffix(reference_column, compare_column)
        path = target / f"{roi_name}{comparison_suffix}{suffix}.{export_fmt}"
        if record is None:
            rows.append(
                {
                    "roi": roi_name,
                    "comparison": compare_column,
                    "status": "failed",
                    "detail": "not in match table",
                    "file": "",
                }
            )
            continue
        if path.exists() and not overwrite:
            rows.append(
                {
                    "roi": roi_name,
                    "comparison": compare_column,
                    "status": "skipped",
                    "detail": "file exists",
                    "file": str(path),
                }
            )
            continue
        try:
            current_settings = dict(layer_settings, compare_column=compare_column)
            image, info = _build_figure(record=record, **current_settings)
            save_kwargs = {"dpi": (export_dpi, export_dpi)} if export_fmt in {"png", "tiff"} else {}
            image.convert("RGB").save(path, format=export_fmt.upper(), **save_kwargs)
            rows.append(
                {
                    "roi": roi_name,
                    "comparison": compare_column,
                    "status": "rendered",
                    "detail": " | ".join(f"{k}={v}" for k, v in info.items() if k != "roi"),
                    "file": str(path),
                }
            )
        except Exception as exc:
            # Keep going: one unreadable image or malformed GeoJSON should not
            # cost the whole batch.
            rows.append(
                {
                    "roi": roi_name,
                    "comparison": compare_column,
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "file": "",
                }
            )
        status.caption(
            f"rendered {sum(r['status'] == 'rendered' for r in rows)} | "
            f"skipped {sum(r['status'] == 'skipped' for r in rows)} | "
            f"failed {sum(r['status'] == 'failed' for r in rows)}"
        )

    progress.progress(1.0, text="Batch complete")
    summary = pd.DataFrame(rows)
    st.session_state[_T + "batch_summary"] = summary
    done = int((summary["status"] == "rendered").sum())
    failed = int((summary["status"] == "failed").sum())
    if failed:
        st.warning(f"Batch finished: {done} rendered, {failed} failed. See the table below.")
    else:
        st.success(f"Batch finished: {done} rendered to {target}")

# ── figure builder ───────────────────────────────────────────────────────────

def _build_figure(
    *,
    record: dict[str, Any],
    loaded_df: pd.DataFrame | None,
    viz_settings: dict[str, Any],
    base_source: str,
    markers: list[str],
    marker_colors: dict[str, str],
    level_choice: str,
    max_size: int,
    contrast: tuple[float, float],
    background: str,
    tissue_classes: list[str],
    tissue_use_embedded: bool,
    tissue_color: str,
    tissue_class_colors: dict[str, str] | None = None,
    tissue_alpha: int = 55,
    tissue_outline_w: int,
    crop_to_tissue: bool,
    mask_outside: bool,
    clip_cells_to_tissue: bool = False,
    cell_filter_rules: list[dict[str, Any]] | None = None,
    auto_zoom_filtered_cells: bool = True,
    zoom_padding: float = 8.0,
    cell_source: str,
    cell_alpha: int,
    cell_colour_mode: str,
    cell_fixed: str,
    radius: float,
    category_column: str = "",
    category_colors: dict[str, str] | None = None,
    continuous_column: str = "",
    low_color: str = "#2C7BB6",
    mid_color: str | None = None,
    high_color: str = "#D7191C",
    clip_pcts: tuple[float, float] = (2.0, 98.0),
    reference_column: str = "",
    compare_column: str = "",
    match_color: str = DEFAULT_MATCH_COLOR,
    mismatch_color: str = DEFAULT_MISMATCH_COLOR,
    ignored_color: str = "#777777",
    ignore_values: list[str] | None = None,
    layer_order: tuple[str, ...] = ("tissue", "cells"),
    show_legend: bool = True,
    legend_place: str = "top-left",
    legend_bg: str = "#FFFFFF",
    show_title: bool = True,
    title_parts: list[str],
    custom_title: str,
) -> tuple[Image.Image, dict[str, Any]]:
    """Compose one figure. Pure enough to be callable outside Streamlit."""
    info: dict[str, Any] = {"roi": record["roi"]}

    # Base ------------------------------------------------------------------
    base: Image.Image | None = None
    full_shape: tuple[int, int] | None = None
    if base_source == "ROI image" and record["has_image"]:
        level = None if level_choice == "auto" else int(str(level_choice).lstrip("L"))
        base, full_shape, used = oc.load_base_image(
            record["image"], channels=markers, channel_colors=marker_colors,
            level=level, max_size=max_size, low_pct=contrast[0], high_pct=contrast[1],
        )
        info["level"] = f"L{used}"
        info["size"] = f"{base.width}x{base.height}"
    else:
        if record["has_image"]:
            full_shape = oc.pyramid_level_shapes(record["image"])[0]
        info["base"] = "blank canvas"

    # Tissue ----------------------------------------------------------------
    tissue_features: list[dict[str, Any]] = []
    if record["has_tissue"] and tissue_classes:
        tissue_features = oc.features_for_classes(record["tissue"], tissue_classes)
        info["tissue_features"] = len(tissue_features)

    crop_box = None
    if crop_to_tissue and tissue_features:
        bounds = [oc.geometry_bounds(f.get("geometry")) for f in tissue_features]
        bounds = [b for b in bounds if b]
        if bounds:
            crop_box = (
                min(b[0] for b in bounds), min(b[1] for b in bounds),
                max(b[2] for b in bounds), max(b[3] for b in bounds),
            )
            info["crop"] = tuple(round(v) for v in crop_box)

    # Cells -----------------------------------------------------------------
    cell_features: list[dict[str, Any]] = []
    label_key = "label"
    unfiltered_cells_df = _cells_for_roi(loaded_df, viz_settings, record["roi"])
    cells_df = apply_cell_filters(unfiltered_cells_df, cell_filter_rules or [])
    if cell_filter_rules:
        before_count = 0 if unfiltered_cells_df is None else len(unfiltered_cells_df)
        after_count = 0 if cells_df is None else len(cells_df)
        info["cell_filter"] = f"{after_count}/{before_count} retained"

    needed = [c for c in (category_column, continuous_column, reference_column, compare_column) if c]
    if cell_source == "Segmentation polygons" and record["has_cells"]:
        cell_features = oc.load_geojson(record["cells"]).get("features", [])
        label_source_df = cells_df if cells_df is not None else unfiltered_cells_df
        frame_label_column = _frame_cell_label_column(label_source_df)
        if cell_filter_rules:
            if frame_label_column is None:
                raise ValueError(
                    "Cell filters cannot be applied to segmentation polygons because the unified CSV "
                    "has no cell label column (expected label, cell_id, cell_label, or object_id)."
                )
            cell_features, retained = _filter_features_to_frame_labels(
                cell_features,
                cells_df,
                frame_label_column=frame_label_column,
            )
            info["segmentation_filter"] = f"{retained} polygons retained"
        if needed and cells_df is not None and not cells_df.empty:
            # Segmentation polygons carry only label/area; every column used for
            # colouring has to be joined on from the unified CSV by cell label.
            matched = oc.attach_csv_properties(
                cell_features,
                cells_df,
                needed,
                frame_label_column=frame_label_column or "label",
            )
            info["csv_join"] = f"{matched}/{len(cell_features)}"
    elif cell_source in {"Voronoi", "Circles"} and cells_df is not None and not cells_df.empty:
        x_col = str(viz_settings.get("x_col") or "x_centroid")
        y_col = str(viz_settings.get("y_col") or "y_centroid")
        extra = needed
        if cell_source == "Voronoi":
            from .actual_tissue import _generate_voronoi_cells

            voronoi = _generate_voronoi_cells(cells_df, x_col, y_col, float(radius), None)
            cell_features = oc.features_from_voronoi(voronoi, property_columns=extra)
        else:
            cell_features = oc.features_from_circles(
                cells_df, x_col, y_col, float(radius),
                label_col=category_column or None, property_columns=extra,
            )
    if clip_cells_to_tissue and cell_features and tissue_features:
        cell_features, clip_stats = oc.clip_cells_to_containing_tissue(
            cell_features,
            tissue_features,
        )
        info["cells_tissue_clipped"] = clip_stats["geometry_clipped"]
        info["cells_outside_tissue"] = clip_stats["outside"]
    info["cells"] = len(cell_features)

    if cell_filter_rules and auto_zoom_filtered_cells and cell_features:
        filtered_cell_box = _padded_feature_bounds(cell_features, zoom_padding, full_shape)
        if filtered_cell_box is not None:
            crop_box = filtered_cell_box
            info["crop"] = tuple(round(value) for value in crop_box)
            info["crop_source"] = "filtered cells"

    # ``crop_box`` is expressed in the level-0 coordinate system shared by the
    # tissue and cell GeoJSONs. Crop the displayed pyramid image only after cell
    # filtering/auto-fit has selected the final box, then map every overlay into
    # that same effective box.
    if crop_box is not None and base is not None and full_shape is not None:
        base, crop_box = oc.crop_base_to_fullres_box(base, full_shape, crop_box)
        info["size"] = f"{base.width}x{base.height}"

    mode = {
        "Unique per cell": "unique", "Categorical column": "categorical",
        "Continuous marker": "continuous", "Match vs reference": "comparison",
        "Fixed": "fixed",
    }[cell_colour_mode]
    if mode == "comparison":
        # Keep the chosen columns with the rendered result so the caption states
        # exactly which comparison the match/mismatch colours represent.
        info["reference_column"] = reference_column
        info["compared_column"] = compare_column
    explicit_colors, colour_legend = oc.resolve_cell_colors(
        cell_features, mode,
        fixed_color=cell_fixed,
        category_column=category_column, category_colors=category_colors or {},
        continuous_column=continuous_column,
        low_color=low_color, mid_color=mid_color, high_color=high_color,
        clip_percentiles=clip_pcts,
        reference_column=reference_column, compare_column=compare_column,
        match_color=match_color, mismatch_color=mismatch_color,
        ignore_values=ignore_values or [], ignored_color=ignored_color,
    )
    if mode == "comparison" and colour_legend.get("agreement") is not None:
        info["agreement"] = f"{colour_legend['agreement'] * 100:.1f}%"
        info["match/mismatch"] = f"{colour_legend['match']}/{colour_legend['mismatch']}"
    elif mode == "continuous" and colour_legend.get("vmin") is not None:
        info["range"] = f"{colour_legend['vmin']} - {colour_legend['vmax']}"

    # Canvas size when there is no base image -------------------------------
    canvas_size = None
    if base is None:
        if crop_box:
            span_x, span_y = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
        elif full_shape:
            span_y, span_x = full_shape
        else:
            all_bounds = [oc.geometry_bounds(f.get("geometry")) for f in (cell_features or tissue_features)]
            all_bounds = [b for b in all_bounds if b]
            if not all_bounds:
                raise ValueError("Nothing to draw: no image, no cells and no annotation polygons.")
            span_x = max(b[2] for b in all_bounds) - min(b[0] for b in all_bounds)
            span_y = max(b[3] for b in all_bounds) - min(b[1] for b in all_bounds)
            if not crop_box:
                crop_box = (
                    min(b[0] for b in all_bounds), min(b[1] for b in all_bounds),
                    max(b[2] for b in all_bounds), max(b[3] for b in all_bounds),
                )
        scale = min(1.0, max_size / max(span_x, span_y, 1.0))
        canvas_size = (max(1, round(span_x * scale)), max(1, round(span_y * scale)))
        info["size"] = f"{canvas_size[0]}x{canvas_size[1]}"

    image, stats = oc.render_overlay(
        base=base, canvas_size=canvas_size, background=background,
        full_shape=full_shape, crop_box=crop_box,
        tissue_features=tissue_features,
        tissue_color=tissue_color,
        tissue_color_lookup=(
            None if tissue_use_embedded else (tissue_class_colors or None)
        ) or (
            {
                oc.feature_name(f.get("properties", {}) or {}): oc.feature_color(
                    f.get("properties", {}) or {}, tissue_color
                )
                for f in tissue_features
            }
            if tissue_use_embedded else None
        ),
        tissue_alpha=tissue_alpha, tissue_outline_width=tissue_outline_w,
        cell_features=cell_features, cell_alpha=cell_alpha,
        cell_color_mode=mode, cell_fixed_color=cell_fixed,
        cell_explicit_colors=explicit_colors, cell_label_key=label_key,
        layer_order=layer_order,
    )
    info["polygons"] = stats["cell_polygons"] + stats["tissue_polygons"]
    if stats["cells_skipped_offscreen"]:
        info["cells_offscreen"] = stats["cells_skipped_offscreen"]

    if mask_outside and tissue_features and full_shape:
        image = oc.mask_outside_geometry(
            image, [f.get("geometry") for f in tissue_features], full_shape, crop_box,
            dim_color=background,
        )

    if show_legend:
        # Three separate blocks, because the three layers answer different
        # questions: which markers the base shows, which annotation classes are
        # drawn, and what the cell colours mean.
        marker_entries: list[tuple[str, str]] = []
        if base is not None and markers:
            for index, marker in enumerate(markers):
                marker_entries.append(
                    (marker, marker_colors.get(marker,
                     oc.DEFAULT_MARKER_COLORS[index % len(oc.DEFAULT_MARKER_COLORS)]))
                )

        tissue_entries = [
            (
                name,
                oc.feature_color(feature.get("properties", {}) or {}, tissue_color)
                if tissue_use_embedded
                else (tissue_class_colors or {}).get(name, tissue_color),
            )
            for name, feature in {
                oc.feature_name(f.get("properties", {}) or {}): f for f in tissue_features
            }.items()
        ]

        raw = oc.legend_entries_from(colour_legend, fixed_color=cell_fixed)
        placeholder = {
            "__RAMP_LOW__": low_color, "__RAMP_HIGH__": high_color,
            "__MATCH__": match_color, "__MISMATCH__": mismatch_color,
            "__IGNORED__": ignored_color,
        }
        cell_entries = [(label, placeholder.get(color, color)) for label, color in raw]

        panels = [
            ("Markers", marker_entries),
            ("Tissue annotation", tissue_entries),
            ("Cells", cell_entries),
        ]
        if legend_place.startswith("overlay"):
            flat = [row for _title, rows in panels for row in rows]
            image = oc.draw_legend(
                image, flat, placement=legend_place.replace("overlay ", ""), background=legend_bg
            )
        else:
            image = oc.attach_legend_panels(
                image, panels, placement=legend_place, background=legend_bg
            )

    if show_title:
        image = _draw_title(image, record, title_parts, custom_title, markers, info)
    return image, info


def _first_class_color(features: list[dict[str, Any]], fallback: str) -> str:
    for feature in features:
        colour = oc.feature_color(feature.get("properties", {}) or {}, "")
        if colour:
            return colour
    return fallback


def _cells_for_roi(
    loaded_df: pd.DataFrame | None, viz_settings: dict[str, Any], roi: str
) -> pd.DataFrame | None:
    """Rows of the unified CSV belonging to one ROI key."""
    if loaded_df is None or loaded_df.empty:
        return None
    batch_col = str(viz_settings.get("batch_col") or "")
    if not batch_col or batch_col not in loaded_df.columns:
        return None
    keys = loaded_df[batch_col].astype(str).map(oc.roi_key)
    subset = loaded_df.loc[keys == roi]
    return subset if not subset.empty else None


def _title_text(
    record: dict[str, Any],
    parts: list[str],
    custom: str,
    markers: list[str],
    info: dict[str, Any],
) -> str:
    """Build the bottom caption, including active comparison column names."""
    bits: list[str] = []
    if "ROI name" in parts:
        bits.append(str(record["roi"]))
    if "Cell count" in parts and "cells" in info:
        bits.append(f"{int(info['cells']):,} cells")
    if "Layers present" in parts:
        layers = [n for n, k in (("image", "has_image"), ("cells", "has_cells"), ("tissue", "has_tissue")) if record[k]]
        bits.append("+".join(layers) if layers else "no layers")
    if "Markers" in parts and markers:
        bits.append(", ".join(markers))
    if info.get("reference_column"):
        bits.append(f"Reference: {info['reference_column']}")
    if info.get("compared_column"):
        bits.append(f"Compared: {info['compared_column']}")
    if custom.strip():
        bits.append(custom.strip())
    return "   |   ".join(bits)


def _draw_title(
    image: Image.Image,
    record: dict[str, Any],
    parts: list[str],
    custom: str,
    markers: list[str],
    info: dict[str, Any],
) -> Image.Image:
    """Draw a caption band under the figure rather than over the tissue."""
    from PIL import ImageDraw, ImageFont

    text = _title_text(record, parts, custom, markers, info)
    if not text:
        return image

    scale = max(1, round(max(image.size) / 1200))
    band = 26 * scale
    canvas = Image.new("RGB", (image.width, image.height + band), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=13 * scale)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    draw.text((8 * scale, image.height + 5 * scale), text, fill=(15, 15, 15), font=font)
    return canvas
