"""
ROI Explorer — Streamlit UI.
Provides _render_roi_explorer(), which replaces _render_representative_rois()
at both call sites in feature_screening.py.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spacemap_gui.ui_utils import directory_input_with_browse

from .roi_explorer_core import (
    apply_cell_filters,
    build_circle_shapes,
    build_composite_image,
    build_continuous_color_map,
    build_roi_figure,
    build_voronoi_shapes,
    _col_is_microns,
    default_annotation_colors,
    default_cell_colors,
    extract_image_channels,
    find_image_for_batch,
    fuzzy_match_geojson_files,
    load_annotation_geojson,
    load_cell_geojson,
    match_cell_geojson_to_csv,
)

# ── Key helpers ────────────────────────────────────────────────────────────────

def _sk(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _sget(prefix: str, name: str, default: Any = None) -> Any:
    return st.session_state.get(_sk(prefix, name), default)


def _lk(prefix: str, lid: int, field: str) -> str:
    return f"{prefix}_l{lid}_{field}"


def _lget(prefix: str, lid: int, field: str, default: Any = None) -> Any:
    return st.session_state.get(_lk(prefix, lid, field), default)


def _lset(prefix: str, lid: int, field: str, value: Any) -> None:
    st.session_state[_lk(prefix, lid, field)] = value


def _slug(text: str) -> str:
    """Convert arbitrary text to a valid identifier fragment for state keys."""
    return re.sub(r"[^a-zA-Z0-9]", "_", str(text)).strip("_")[:48]


# ── Column helpers ─────────────────────────────────────────────────────────────

def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == object or df[c].nunique() <= 50]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ── Layer skeleton ─────────────────────────────────────────────────────────────

def _new_layer(layer_id: int, layer_type: str) -> dict[str, Any]:
    titles = {
        "markers": "Markers",
        "annotations": "Tissue Annotations",
        "cells": "Cell Segmentation",
        "scatter": "Cell Scatter",
    }
    return {"id": layer_id, "type": layer_type, "title": titles.get(layer_type, layer_type)}


# ── Filter rule builder ────────────────────────────────────────────────────────

def _render_filter_rules(
    prefix: str,
    lid: int,
    analysis_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Render the filter-rule editor for a layer and return current rules."""
    fk = _lk(prefix, lid, "filter_rules")
    if fk not in st.session_state:
        st.session_state[fk] = []
    rules: list[dict[str, Any]] = st.session_state[fk]

    all_cols = list(analysis_df.columns)
    if not all_cols:
        return rules

    ca, _ = st.columns([3, 1])
    with ca:
        add_col = st.selectbox(
            "Add filter on column",
            ["— pick column —"] + all_cols,
            key=f"{fk}_addcol",
        )
    if st.button("Add filter", key=f"{fk}_addbtn") and add_col != "— pick column —":
        is_num = pd.api.types.is_numeric_dtype(analysis_df[add_col]) if add_col in analysis_df.columns else False
        rules.append({
            "column": add_col,
            "mode": "range" if is_num else "categorical",
            "values": [],
            "min_val": None,
            "max_val": None,
            "include_missing": True,
        })
        st.session_state[fk] = rules
        st.rerun()

    to_delete: list[int] = []
    for ri, rule in enumerate(rules):
        col = str(rule["column"])
        with st.expander(f"Filter: {col}", expanded=True):
            mode = rule.get("mode", "categorical")
            if mode == "categorical":
                unique_vals = sorted(
                    analysis_df[col].dropna().astype(str).unique().tolist()
                ) if col in analysis_df.columns else []
                cur_vals = rule.get("values") or unique_vals
                new_vals = st.multiselect(
                    f"Allowed values",
                    unique_vals,
                    default=[v for v in cur_vals if v in unique_vals],
                    key=f"{fk}_r{ri}_vals",
                )
                rule["values"] = new_vals
            else:
                if col in analysis_df.columns:
                    n_series = pd.to_numeric(analysis_df[col], errors="coerce").dropna()
                    dmin = float(n_series.min()) if len(n_series) else 0.0
                    dmax = float(n_series.max()) if len(n_series) else 1.0
                else:
                    dmin, dmax = 0.0, 1.0
                step = (dmax - dmin) / 100.0 if dmax > dmin else 1.0
                r1, r2 = st.columns(2)
                with r1:
                    rule["min_val"] = st.number_input(
                        "Min", value=float(rule.get("min_val") or dmin), step=step,
                        key=f"{fk}_r{ri}_min",
                    )
                with r2:
                    rule["max_val"] = st.number_input(
                        "Max", value=float(rule.get("max_val") or dmax), step=step,
                        key=f"{fk}_r{ri}_max",
                    )
            rule["include_missing"] = st.checkbox(
                "Include missing values",
                value=bool(rule.get("include_missing", True)),
                key=f"{fk}_r{ri}_miss",
            )
            if st.button("Remove filter", key=f"{fk}_r{ri}_del"):
                to_delete.append(ri)

    if to_delete:
        for ri in sorted(to_delete, reverse=True):
            rules.pop(ri)
        st.session_state[fk] = rules
        st.rerun()

    st.session_state[fk] = rules
    return rules


# ── Layer card ─────────────────────────────────────────────────────────────────

def _render_layer_card(
    prefix: str,
    layer: dict[str, Any],
    idx: int,
    total: int,
    available_channels: list[dict[str, Any]],
    batch_id: str,
    annot_geojson_dir: str,
    cell_geojson_dir: str,
    analysis_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Render the config UI for one layer.
    Returns an action dict {_action, idx} if a structural change was requested,
    or None otherwise.
    """
    lid = layer["id"]
    ltype = str(layer.get("type", ""))

    # ── Header row ──────────────────────────────────────────────────────────
    hc, uc, dc, xc = st.columns([6, 1, 1, 1])
    with hc:
        cur_title = _lget(prefix, lid, "title", layer.get("title", ltype))
        new_title = st.text_input(
            "Layer title",
            value=str(cur_title),
            key=_lk(prefix, lid, "title_input"),
            label_visibility="collapsed",
        )
        _lset(prefix, lid, "title", new_title)
    with uc:
        if idx > 0 and st.button("↑", key=f"{prefix}_up_{lid}", help="Move up"):
            return {"_action": "move_up", "idx": idx}
    with dc:
        if idx < total - 1 and st.button("↓", key=f"{prefix}_dn_{lid}", help="Move down"):
            return {"_action": "move_dn", "idx": idx}
    with xc:
        if st.button(":material/delete:", key=f"{prefix}_del_{lid}", help="Remove layer"):
            return {"_action": "delete", "idx": idx}

    st.caption(f"**{ltype}**")

    # ── Common controls ─────────────────────────────────────────────────────
    cc1, cc2 = st.columns(2)
    with cc1:
        enabled = st.checkbox("Enabled", value=bool(_lget(prefix, lid, "enabled", True)),
                              key=_lk(prefix, lid, "enabled"))
    with cc2:
        opacity = st.slider("Opacity", 0.0, 1.0,
                            float(_lget(prefix, lid, "opacity", 1.0)), 0.05,
                            key=_lk(prefix, lid, "opacity"))

    # ── Type-specific controls ───────────────────────────────────────────────
    cols_cfg = config.get("columns", {})
    x_col = str(cols_cfg.get("x") or "")
    y_col = str(cols_cfg.get("y") or "")
    ph_col = str(cols_cfg.get("phenotype") or "")

    if ltype == "markers":
        _render_layer_markers(prefix, lid, available_channels)

    elif ltype == "annotations":
        _render_layer_annotations(prefix, lid, batch_id, annot_geojson_dir)

    elif ltype in ("cells", "scatter"):
        _render_layer_cells(prefix, lid, ltype, batch_id, cell_geojson_dir,
                            analysis_df, roi_df, ph_col, x_col=x_col, y_col=y_col)

    return None  # no structural action


# ── Per-type sub-renderers ─────────────────────────────────────────────────────

def _render_layer_markers(
    prefix: str,
    lid: int,
    available_channels: list[dict[str, Any]],
) -> None:
    ch_names = [ch["name"] for ch in available_channels]
    default_sel = _lget(prefix, lid, "selected_names", ch_names[:min(len(ch_names), 6)])
    sel_names = st.multiselect(
        "Channels to show",
        ch_names,
        default=[n for n in default_sel if n in ch_names],
        key=_lk(prefix, lid, "selected_names"),
    )
    sel_channels: list[dict[str, Any]] = []
    if sel_names:
        st.markdown("**Channel colors:**")
        ncols = min(len(sel_names), 4)
        cols = st.columns(ncols) if ncols > 1 else [st.container()]
        ci = 0
        for ch in available_channels:
            if ch["name"] not in sel_names:
                continue
            ck = _lk(prefix, lid, f"chcol_{_slug(ch['name'])}")
            cur_col = _lget(prefix, lid, f"chcol_{_slug(ch['name'])}", ch["color"])
            with cols[ci % ncols]:
                new_col = st.color_picker(ch["name"], value=str(cur_col), key=ck)
            sel_channels.append({**ch, "color": new_col})
            ci += 1
    _lset(prefix, lid, "selected_channels", sel_channels)


def _render_layer_annotations(
    prefix: str,
    lid: int,
    batch_id: str,
    annot_geojson_dir: str,
) -> None:
    matches = fuzzy_match_geojson_files(batch_id, annot_geojson_dir, role="annotations")
    match_options = ["— none —"] + [str(p) for _, p in matches]
    default_path = _lget(prefix, lid, "geojson_path_sel",
                         match_options[1] if len(match_options) > 1 else "— none —")
    if default_path not in match_options:
        default_path = match_options[0]
    sel_path = st.selectbox(
        "Annotation GeoJSON file",
        match_options,
        index=match_options.index(default_path),
        key=_lk(prefix, lid, "geojson_path_sel"),
    )
    override = st.text_input(
        "Or enter path manually",
        value=str(_lget(prefix, lid, "geojson_manual", "")),
        key=_lk(prefix, lid, "geojson_manual"),
    )
    eff_path = override.strip() if override.strip() else (sel_path if sel_path != "— none —" else "")
    _lset(prefix, lid, "geojson_path", eff_path)

    c1, c2 = st.columns(2)
    with c1:
        fill = st.checkbox("Fill polygons", value=bool(_lget(prefix, lid, "fill", True)),
                           key=_lk(prefix, lid, "fill"))
        flip_y = st.checkbox("Flip Y", value=bool(_lget(prefix, lid, "flip_y", False)),
                             key=_lk(prefix, lid, "flip_y"))
    with c2:
        ot = st.slider("Outline", 0.5, 5.0, float(_lget(prefix, lid, "outline_thickness", 1.5)), 0.5,
                       key=_lk(prefix, lid, "outline_thickness"))

    # Discover annotation classes from file and show color pickers
    if eff_path and Path(eff_path).exists():
        try:
            feats = load_annotation_geojson(eff_path)
        except Exception:
            feats = []
        class_names = sorted({f["class_name"] for f in feats})
        if class_names:
            st.markdown("**Class colors:**")
            defaults = default_annotation_colors(class_names)
            ncols = min(len(class_names), 4)
            cols = st.columns(ncols) if ncols > 1 else [st.container()]
            for ci, cls in enumerate(class_names):
                hint = next((f["color_hint"] for f in feats
                             if f["class_name"] == cls and f.get("color_hint")), None)
                ck = _lk(prefix, lid, f"acol_{_slug(cls)}")
                cur = _lget(prefix, lid, f"acol_{_slug(cls)}",
                            hint or defaults.get(cls, "#888888"))
                with cols[ci % ncols]:
                    new_c = st.color_picker(cls, value=str(cur), key=ck)
    elif eff_path:
        st.warning(f"File not found: `{eff_path}`")


def _render_layer_cells(
    prefix: str,
    lid: int,
    ltype: str,
    batch_id: str,
    cell_geojson_dir: str,
    analysis_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    ph_col: str,
    x_col: str = "",
    y_col: str = "",
) -> None:
    if ltype == "cells":
        shape_opts = ["Circle", "Voronoi", "GeoJSON"]
        # Default to GeoJSON on first render if a cell segmentation folder is configured
        _shape_default = (
            "GeoJSON"
            if cell_geojson_dir and _lk(prefix, lid, "shape_mode") not in st.session_state
            else "Circle"
        )
        shape_mode = st.radio(
            "Cell shape mode", shape_opts,
            index=shape_opts.index(_lget(prefix, lid, "shape_mode", _shape_default)),
            horizontal=True, key=_lk(prefix, lid, "shape_mode"),
        )

        if shape_mode == "GeoJSON":
            matches = fuzzy_match_geojson_files(batch_id, cell_geojson_dir, role="cells")
            opts = ["— none —"] + [str(p) for _, p in matches]
            dp = _lget(prefix, lid, "cell_gj_sel", opts[1] if len(opts) > 1 else "— none —")
            if dp not in opts:
                dp = opts[0]
            sel = st.selectbox("Cell segmentation GeoJSON", opts,
                               index=opts.index(dp), key=_lk(prefix, lid, "cell_gj_sel"))
            ov = st.text_input("Or enter path manually",
                               value=str(_lget(prefix, lid, "cell_gj_manual", "")),
                               key=_lk(prefix, lid, "cell_gj_manual"))
            eff = ov.strip() if ov.strip() else (sel if sel != "— none —" else "")
            _lset(prefix, lid, "cell_geojson_path", eff)
            c1, c2 = st.columns(2)
            with c1:
                flip_y = st.checkbox("Flip Y", value=bool(_lget(prefix, lid, "cell_flip_y", False)),
                                     key=_lk(prefix, lid, "cell_flip_y"))
            with c2:
                md = st.number_input("Max match dist (px)",
                                     value=float(_lget(prefix, lid, "match_dist", 50.0)),
                                     min_value=1.0, max_value=500.0, step=5.0,
                                     key=_lk(prefix, lid, "match_dist"))
            # MPP widget — shown whenever the selected coordinate columns look
            # like micron units (e.g. "Centroid X µm" from Tab 4).
            if _col_is_microns(x_col) or _col_is_microns(y_col):
                mpp_val = st.number_input(
                    "Image MPP (µm / px) for coordinate matching",
                    value=float(_lget(prefix, lid, "csv_mpp", 0.0)),
                    min_value=0.0, max_value=5.0, step=0.001, format="%.4f",
                    help=(
                        "The selected X/Y columns appear to be in microns. "
                        "Enter the image resolution so coordinates can be "
                        "converted to pixels before matching against GeoJSON shapes. "
                        "Set to 0 to skip conversion."
                    ),
                    key=_lk(prefix, lid, "csv_mpp"),
                )

        elif shape_mode == "Voronoi":
            c1, c2 = st.columns(2)
            with c1:
                pct = st.slider("Cap percentile", 10, 90,
                                int(_lget(prefix, lid, "voronoi_pct", 50)),
                                key=_lk(prefix, lid, "voronoi_pct"))
            with c2:
                fbr = st.number_input("Fallback radius (px)",
                                      value=float(_lget(prefix, lid, "fallback_r", 15.0)),
                                      min_value=1.0, max_value=100.0, step=1.0,
                                      key=_lk(prefix, lid, "fallback_r"))
        else:
            radius = st.number_input("Circle radius (px)",
                                     value=float(_lget(prefix, lid, "circle_r", 5.0)),
                                     min_value=0.5, max_value=100.0, step=0.5,
                                     key=_lk(prefix, lid, "circle_r"))

        c1, c2 = st.columns(2)
        with c1:
            fill = st.checkbox("Fill cells", value=bool(_lget(prefix, lid, "fill", True)),
                               key=_lk(prefix, lid, "fill"))
        with c2:
            ot = st.slider("Outline", 0.0, 3.0, float(_lget(prefix, lid, "outline_thickness", 0.5)), 0.25,
                           key=_lk(prefix, lid, "outline_thickness"))
    else:
        # scatter
        size = st.number_input("Point size", value=float(_lget(prefix, lid, "size", 20.0)),
                               min_value=1.0, max_value=300.0, step=1.0,
                               key=_lk(prefix, lid, "size"))

    # Color-by
    color_by_opts = ["Phenotype column", "Categorical column", "Continuous column"]
    color_by = st.radio(
        "Color cells by", color_by_opts,
        index=color_by_opts.index(_lget(prefix, lid, "color_by", "Phenotype column")),
        horizontal=True, key=_lk(prefix, lid, "color_by"),
    )

    if color_by == "Categorical column":
        cat_cols = _categorical_columns(analysis_df)
        if cat_cols:
            default_col = _lget(prefix, lid, "color_col", ph_col)
            idx = cat_cols.index(default_col) if default_col in cat_cols else 0
            sel_col = st.selectbox("Column", cat_cols, index=idx, key=_lk(prefix, lid, "color_col"))
            eff_ph = str(sel_col)
        else:
            st.caption("No categorical columns found.")
            eff_ph = ph_col
    elif color_by == "Continuous column":
        num_cols = _numeric_columns(analysis_df)
        if num_cols:
            sel_col = st.selectbox("Column", num_cols, key=_lk(prefix, lid, "cont_col"))
            cmap_opts = ["viridis", "plasma", "magma", "inferno", "coolwarm",
                         "RdYlBu", "RdBu", "YlOrRd", "Blues", "Greens"]
            cmap = st.selectbox("Color map", cmap_opts, key=_lk(prefix, lid, "cmap"))
        else:
            st.caption("No numeric columns found.")
        eff_ph = ph_col
    else:
        eff_ph = ph_col

    # Per-phenotype color pickers (phenotype or categorical only)
    if color_by in ("Phenotype column", "Categorical column") and not roi_df.empty:
        col_eff = _lget(prefix, lid, "color_col", ph_col) if color_by == "Categorical column" else ph_col
        if col_eff and col_eff in roi_df.columns:
            unique_vals = sorted(roi_df[col_eff].dropna().astype(str).unique().tolist())
            if unique_vals:
                defaults = default_cell_colors(unique_vals)
                with st.expander(f"Per-category colors ({col_eff})", expanded=False):
                    ncols = min(len(unique_vals), 4)
                    cols = st.columns(ncols) if ncols > 1 else [st.container()]
                    for ci, val in enumerate(unique_vals):
                        ck = _lk(prefix, lid, f"ccol_{_slug(val)}")
                        cur = _lget(prefix, lid, f"ccol_{_slug(val)}", defaults.get(val, "#888888"))
                        with cols[ci % ncols]:
                            new_c = st.color_picker(val, value=str(cur), key=ck)

    # Visible-phenotype quick-filter — deselect to hide individual phenotypes
    if ph_col and ph_col in roi_df.columns and not roi_df.empty:
        _ph_all = sorted(roi_df[ph_col].dropna().astype(str).unique().tolist())
        if _ph_all:
            _cur_vis = _lget(prefix, lid, "visible_phs", None)
            _cur_vis = [v for v in _cur_vis if v in _ph_all] if _cur_vis is not None else _ph_all
            st.multiselect(
                "Visible phenotypes",
                _ph_all,
                default=_cur_vis,
                key=_lk(prefix, lid, "visible_phs"),
                help="Deselect phenotypes to hide them from the rendered figure.",
            )

    # Cell filter rules
    with st.expander("Cell filters", expanded=False):
        _render_filter_rules(prefix, lid, analysis_df)


# ── Build layer data (called at render time) ───────────────────────────────────

def _build_layer_data(
    layer: dict[str, Any],
    prefix: str,
    roi_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    scale_x: float,
    scale_y: float,
    annot_dir: str,
    cell_dir: str,
    batch_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Populate a layer dict with computed rendering data."""
    lid = layer["id"]
    ltype = str(layer.get("type", ""))
    cols_cfg = config.get("columns", {})
    x_col = str(cols_cfg.get("x") or "")
    y_col = str(cols_cfg.get("y") or "")
    ph_col = str(cols_cfg.get("phenotype") or "")

    result = dict(layer)
    result["enabled"] = bool(_lget(prefix, lid, "enabled", True))
    result["opacity"] = float(_lget(prefix, lid, "opacity", 1.0))
    result["title"] = str(_lget(prefix, lid, "title", layer.get("title", ltype)))

    if ltype == "markers":
        result["selected_channels"] = _lget(prefix, lid, "selected_channels", [])

    elif ltype == "annotations":
        eff_path = str(_lget(prefix, lid, "geojson_path", "") or "")
        fill = bool(_lget(prefix, lid, "fill", True))
        flip_y = bool(_lget(prefix, lid, "flip_y", False))
        ot = float(_lget(prefix, lid, "outline_thickness", 1.5))
        features: list[dict[str, Any]] = []
        color_map: dict[str, str] = {}
        if eff_path and Path(eff_path).exists():
            features = load_annotation_geojson(eff_path, scale_x=scale_x, scale_y=scale_y, flip_y=flip_y)
            class_names = sorted({f["class_name"] for f in features})
            defaults = default_annotation_colors(class_names)
            for cls in class_names:
                color_map[cls] = str(_lget(prefix, lid, f"acol_{_slug(cls)}", defaults.get(cls, "#888888")))
        result["_features"] = features
        result["color_map"] = color_map
        result["fill"] = fill
        result["outline_thickness"] = ot

    elif ltype in ("cells", "scatter"):
        color_by = str(_lget(prefix, lid, "color_by", "Phenotype column"))
        if color_by == "Categorical column":
            eff_ph = str(_lget(prefix, lid, "color_col", ph_col) or ph_col)
        else:
            eff_ph = ph_col

        filter_rules = st.session_state.get(_lk(prefix, lid, "filter_rules"), [])
        filtered_df = apply_cell_filters(roi_df, filter_rules)
        # Apply visible-phenotype filter (from the multiselect in the layer config UI)
        visible_phs = _lget(prefix, lid, "visible_phs", None)
        if (visible_phs is not None and isinstance(visible_phs, list)
                and eff_ph and eff_ph in filtered_df.columns):
            filtered_df = filtered_df[filtered_df[eff_ph].astype(str).isin(set(visible_phs))].copy()

        if ltype == "cells":
            shape_mode = str(_lget(prefix, lid, "shape_mode", "Circle"))
            fill = bool(_lget(prefix, lid, "fill", True))
            ot = float(_lget(prefix, lid, "outline_thickness", 0.5))
            features = []
            if shape_mode == "GeoJSON":
                gj_path = str(_lget(prefix, lid, "cell_geojson_path", "") or "")
                flip_y = bool(_lget(prefix, lid, "cell_flip_y", False))
                max_dist = float(_lget(prefix, lid, "match_dist", 50.0))
                if gj_path and Path(gj_path).exists():
                    features = load_cell_geojson(gj_path, scale_x=scale_x, scale_y=scale_y,
                                                  flip_y=flip_y, phenotype_col=eff_ph)
                    csv_mpp = float(_lget(prefix, lid, "csv_mpp", 0.0) or 0.0)
                    features = match_cell_geojson_to_csv(
                        features, filtered_df, x_col, y_col,
                        eff_ph, scale_x, scale_y, max_dist,
                        mpp=csv_mpp if csv_mpp > 0 else None,
                    )
            elif shape_mode == "Voronoi":
                pct = float(_lget(prefix, lid, "voronoi_pct", 50))
                fbr = float(_lget(prefix, lid, "fallback_r", 15.0))
                features = build_voronoi_shapes(filtered_df, x_col, y_col, eff_ph, scale_x, scale_y, pct, fbr)
            else:
                r = float(_lget(prefix, lid, "circle_r", 5.0))
                features = build_circle_shapes(filtered_df, x_col, y_col, eff_ph, scale_x, scale_y, r)

            all_ph = sorted({f.get("phenotype", "Unknown") for f in features}) if features else []
            defaults = default_cell_colors(all_ph)
            color_map = {ph: str(_lget(prefix, lid, f"ccol_{_slug(ph)}", defaults.get(ph, "#888888")))
                         for ph in all_ph}
            result["_features"] = features
            result["color_map"] = color_map
            result["fill"] = fill
            result["outline_thickness"] = ot

        else:  # scatter
            size = float(_lget(prefix, lid, "size", 20.0))
            index_color_map: dict[Any, str] | None = None

            if color_by == "Continuous column":
                cont_col = str(_lget(prefix, lid, "cont_col", "") or "")
                cmap = str(_lget(prefix, lid, "cmap", "viridis") or "viridis")
                if cont_col and cont_col in filtered_df.columns:
                    index_color_map, _, _ = build_continuous_color_map(filtered_df[cont_col], cmap)
                color_map = {}
            else:
                unique_vals = sorted(filtered_df[eff_ph].dropna().astype(str).unique().tolist()) \
                    if eff_ph in filtered_df.columns else []
                defaults = default_cell_colors(unique_vals)
                color_map = {v: str(_lget(prefix, lid, f"ccol_{_slug(v)}", defaults.get(v, "#888888")))
                             for v in unique_vals}

            result["_filtered_df"] = filtered_df
            result["phenotype_col"] = eff_ph
            result["color_map"] = color_map
            result["size"] = size
            if index_color_map is not None:
                result["_index_color_map"] = index_color_map

    return result


# ── Render + export ────────────────────────────────────────────────────────────

def _do_render_and_show(
    prefix: str,
    roi_df: pd.DataFrame,
    layers: list[dict[str, Any]],
    image_path: Path | None,
    available_channels: list[dict[str, Any]],
    analysis_df: pd.DataFrame,
    config: dict[str, Any],
    batch_id: str,
    roi_id: str,
    annot_dir: str,
    cell_dir: str,
    show_legend: bool,
    show_title: bool,
    bg_color: str,
    title_override: str,
    legend_frac: float,
    preview_max_dim: int,
    preview_dpi: int,
    export_dpi: int,
    is_export: bool,
) -> None:
    """Build the ROI figure and either show a preview or offer a PNG download."""
    cols_cfg = config.get("columns", {})
    x_col = str(cols_cfg.get("x") or "")
    y_col = str(cols_cfg.get("y") or "")
    dpi = export_dpi if is_export else preview_dpi
    load_max = max(export_dpi * 8, 2000) if is_export else preview_max_dim

    # Load composite image from the first enabled markers layer
    image: np.ndarray | None = None
    scale_x, scale_y = 1.0, 1.0
    if image_path:
        markers_channels: list[dict[str, Any]] = []
        for layer in layers:
            if layer.get("type") == "markers":
                lid = layer["id"]
                if bool(_lget(prefix, lid, "enabled", True)):
                    markers_channels = _lget(prefix, lid, "selected_channels", [])
                    if not markers_channels:
                        markers_channels = available_channels[:min(len(available_channels), 6)]
                    break
        channels_to_use = markers_channels or available_channels[:min(len(available_channels), 6)]
        if channels_to_use:
            image, scale_x, scale_y = build_composite_image(str(image_path), load_max, channels_to_use)

    # Populate all layers with computed data
    computed_layers: list[dict[str, Any]] = []
    for layer in layers:
        try:
            cl = _build_layer_data(
                layer=layer, prefix=prefix,
                roi_df=roi_df, analysis_df=analysis_df,
                scale_x=scale_x, scale_y=scale_y,
                annot_dir=annot_dir, cell_dir=cell_dir,
                batch_id=batch_id, config=config,
            )
            computed_layers.append(cl)
        except Exception as exc:
            st.warning(f"Layer {layer.get('id', '?')} ({layer.get('type', '?')}): {exc}")

    title = title_override.strip() if title_override.strip() else (
        f"{batch_id}" + (f" — ROI {roi_id}" if roi_id else "")
    )

    try:
        fig = build_roi_figure(
            roi_df=roi_df, image=image, layers=computed_layers,
            scale_x=scale_x, scale_y=scale_y,
            x_col=x_col, y_col=y_col, title=title,
            dpi=dpi, fig_inches=8.0, legend_fraction=legend_frac,
            bg_color=bg_color, show_legend=show_legend, show_title=show_title,
        )
    except Exception as exc:
        st.error(f"Figure rendering failed: {exc}")
        return

    if is_export:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=export_dpi,
                    bbox_inches="tight", facecolor=bg_color)
        buf.seek(0)
        plt.close(fig)
        safe = re.sub(r"[^\w\-.]", "_", f"{batch_id}_{roi_id}_roi_explorer.png")
        st.download_button(
            f"Download {safe}",
            data=buf, file_name=safe, mime="image/png",
            key=f"{prefix}_dl_{abs(hash(safe)) % 99999}",
        )
    else:
        st.session_state[_sk(prefix, "preview_fig")] = fig
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# ── Public entry point ─────────────────────────────────────────────────────────

def _render_roi_explorer(
    analysis_df: pd.DataFrame,
    config: dict[str, Any],
    state_prefix: str = "tab8_roi",
) -> None:
    """
    Full ROI Explorer UI — replaces _render_representative_rois at both call sites.

    Parameters
    ----------
    analysis_df:
        The unified cell dataframe (all batches / ROIs).
    config:
        Feature-screening config dict with ``config["columns"]`` keys:
        batch, roi, x, y, phenotype.
    state_prefix:
        Unique session-state namespace (different per call site).
    """
    p = state_prefix

    if analysis_df is None or (isinstance(analysis_df, pd.DataFrame) and analysis_df.empty):
        st.info("No data loaded. Provide a unified CSV to use the ROI Explorer.")
        return

    cols_cfg = config.get("columns", {})
    batch_col = str(cols_cfg.get("batch") or "")
    roi_col = str(cols_cfg.get("roi") or "")
    x_col = str(cols_cfg.get("x") or "")
    y_col = str(cols_cfg.get("y") or "")
    ph_col = str(cols_cfg.get("phenotype") or "")

    if not batch_col or batch_col not in analysis_df.columns:
        st.warning("Batch column is not configured or not found in the data.")
        return

    # ── Data Sources ──────────────────────────────────────────────────────────
    with st.expander("Data Sources", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Raw image directory**")
            img_dir_default = str(config.get("representatives", {}).get("raw_image_dir", "") or "")
            img_dir = directory_input_with_browse(
                label="Image directory",
                text_key=_sk(p, "image_dir"),
                default_value=img_dir_default,
                browser_key_prefix=_sk(p, "image_dir_br"),
            )
        with c2:
            st.markdown("**Annotation GeoJSON folder**")
            annot_dir = directory_input_with_browse(
                label="Annotation GeoJSON folder",
                text_key=_sk(p, "annot_dir"),
                default_value=str(st.session_state.get(_sk(p, "annot_dir"), "") or ""),
                browser_key_prefix=_sk(p, "annot_dir_br"),
            )
        with c3:
            st.markdown("**Cell segmentation GeoJSON folder**")
            cell_dir = directory_input_with_browse(
                label="Cell GeoJSON folder",
                text_key=_sk(p, "cell_dir"),
                default_value=str(st.session_state.get(_sk(p, "cell_dir"), "") or ""),
                browser_key_prefix=_sk(p, "cell_dir_br"),
            )

    img_dir = str(st.session_state.get(_sk(p, "image_dir"), img_dir_default) or "")
    annot_dir = str(st.session_state.get(_sk(p, "annot_dir"), "") or "")
    cell_dir = str(st.session_state.get(_sk(p, "cell_dir"), "") or "")

    # ── ROI Selector ──────────────────────────────────────────────────────────
    st.markdown("#### Select ROI")
    unique_batches = sorted(analysis_df[batch_col].dropna().astype(str).unique().tolist())
    if not unique_batches:
        st.info("No batch IDs found in the data.")
        return

    sel_c1, sel_c2 = st.columns(2)
    with sel_c1:
        cur_batch = st.session_state.get(_sk(p, "batch"), unique_batches[0])
        batch_id = st.selectbox(
            "Batch / Image",
            unique_batches,
            index=unique_batches.index(cur_batch) if cur_batch in unique_batches else 0,
            key=_sk(p, "batch"),
        )
    batch_df = analysis_df[analysis_df[batch_col].astype(str) == str(batch_id)].copy()

    unique_rois: list[str] = []
    if roi_col and roi_col in batch_df.columns:
        unique_rois = sorted(batch_df[roi_col].dropna().astype(str).unique().tolist())

    with sel_c2:
        if unique_rois:
            cur_roi = st.session_state.get(_sk(p, "roi"), unique_rois[0])
            roi_id = st.selectbox(
                "ROI",
                unique_rois,
                index=unique_rois.index(cur_roi) if cur_roi in unique_rois else 0,
                key=_sk(p, "roi"),
            )
            roi_df = batch_df[batch_df[roi_col].astype(str) == str(roi_id)].copy()
        else:
            st.caption("No ROI column configured — showing entire batch.")
            roi_id = ""
            roi_df = batch_df.copy()

    # Quick summary metrics
    if not roi_df.empty:
        m_cols = [c for c in (ph_col, str(cols_cfg.get("neighborhood_distance_tumor", "")),
                               str(cols_cfg.get("context", ""))) if c and c in roi_df.columns]
        stat_cols = st.columns(min(len(m_cols) + 1, 4))
        with stat_cols[0]:
            st.metric("Cells in ROI", f"{len(roi_df):,}")
        for si, sc in enumerate(m_cols[:3], 1):
            with stat_cols[si]:
                if pd.api.types.is_numeric_dtype(roi_df[sc]):
                    med = pd.to_numeric(roi_df[sc], errors="coerce").median()
                    st.metric(f"Median {sc}", f"{med:.1f}" if pd.notna(med) else "—")
                else:
                    st.metric(f"Unique {sc}", f"{roi_df[sc].nunique()}")

    # ── Auto-detect image channels ─────────────────────────────────────────────
    image_path: Path | None = find_image_for_batch(str(batch_id), img_dir) if img_dir else None
    available_channels: list[dict[str, Any]] = []
    if image_path:
        cache_key = _sk(p, f"ch_{_slug(str(batch_id))}")
        if cache_key not in st.session_state:
            st.session_state[cache_key] = extract_image_channels(image_path)
        available_channels = st.session_state[cache_key]
        st.caption(f"Image: `{image_path.name}` — **{len(available_channels)}** channels detected")
    elif img_dir:
        st.caption(f"No image found for batch `{batch_id}` in `{img_dir}`")

    # ── Layer Manager ─────────────────────────────────────────────────────────
    st.markdown("#### Layers")
    layers_key = _sk(p, "layers")
    nid_key = _sk(p, "next_lid")
    if layers_key not in st.session_state:
        st.session_state[layers_key] = []
        st.session_state[nid_key] = 0

    layers: list[dict[str, Any]] = st.session_state[layers_key]

    # Add-layer buttons
    ba1, ba2, ba3, ba4, _ = st.columns([2, 2, 2, 2, 4])
    for btn_label, btn_type, btn_col in [
        ("+ Markers", "markers", ba1),
        ("+ Annotations", "annotations", ba2),
        ("+ Cells (shapes)", "cells", ba3),
        ("+ Cells (scatter)", "scatter", ba4),
    ]:
        with btn_col:
            if st.button(btn_label, key=_sk(p, f"add_{btn_type}")):
                nid = st.session_state[nid_key]
                layers.append(_new_layer(nid, btn_type))
                st.session_state[nid_key] = nid + 1
                st.session_state[layers_key] = layers
                st.rerun()

    if not layers:
        st.caption("No layers yet — add one above.")

    action: dict[str, Any] | None = None
    for i, layer in enumerate(layers):
        with st.container(border=True):
            act = _render_layer_card(
                prefix=p, layer=layer, idx=i, total=len(layers),
                available_channels=available_channels,
                batch_id=str(batch_id),
                annot_geojson_dir=annot_dir,
                cell_geojson_dir=cell_dir,
                analysis_df=analysis_df,
                roi_df=roi_df,
                config=config,
            )
            if act is not None:
                action = act
                break

    if action:
        idx_a = action["idx"]
        if action["_action"] == "delete":
            layers.pop(idx_a)
        elif action["_action"] == "move_up" and idx_a > 0:
            layers[idx_a], layers[idx_a - 1] = layers[idx_a - 1], layers[idx_a]
        elif action["_action"] == "move_dn" and idx_a < len(layers) - 1:
            layers[idx_a], layers[idx_a + 1] = layers[idx_a + 1], layers[idx_a]
        st.session_state[layers_key] = layers
        st.rerun()

    # ── Figure Settings ───────────────────────────────────────────────────────
    with st.expander("Figure Settings", expanded=False):
        fs1, fs2, fs3 = st.columns(3)
        with fs1:
            show_legend = st.checkbox("Show legend", value=bool(st.session_state.get(_sk(p, "show_legend"), True)),
                                       key=_sk(p, "show_legend"))
            show_title = st.checkbox("Show title", value=bool(st.session_state.get(_sk(p, "show_title"), True)),
                                      key=_sk(p, "show_title"))
            bg_color = st.color_picker("Background", value=str(st.session_state.get(_sk(p, "bg_color"), "#111111")),
                                        key=_sk(p, "bg_color"))
        with fs2:
            title_override = st.text_input(
                "Title (blank = auto)",
                value=str(st.session_state.get(_sk(p, "title_override"), "")),
                key=_sk(p, "title_override"),
            )
            preview_max_dim = int(st.number_input(
                "Preview image max dim (px)",
                value=int(st.session_state.get(_sk(p, "prev_dim"), 500)),
                min_value=200, max_value=2000, step=100,
                key=_sk(p, "prev_dim"),
                help="Pixel cap for image loading during preview. Lower = faster.",
            ))
            preview_dpi = int(st.number_input(
                "Preview DPI",
                value=int(st.session_state.get(_sk(p, "prev_dpi"), 72)),
                min_value=36, max_value=150, step=12,
                key=_sk(p, "prev_dpi"),
                help="Figure resolution for interactive preview. Lower = faster rendering.",
            ))
        with fs3:
            export_dpi = int(st.number_input(
                "Export DPI",
                value=int(st.session_state.get(_sk(p, "export_dpi"), 300)),
                min_value=72, max_value=600, step=50,
                key=_sk(p, "export_dpi"),
            ))
            legend_frac = float(st.slider(
                "Legend width",
                0.10, 0.40,
                float(st.session_state.get(_sk(p, "legend_frac"), 0.22)), 0.02,
                key=_sk(p, "legend_frac"),
            ))

    # ── Preview / Export ──────────────────────────────────────────────────────
    st.markdown("#### Preview")
    pb1, pb2 = st.columns(2)
    with pb1:
        render_btn = st.button("Render / Update", key=_sk(p, "render_btn"), type="primary")
    with pb2:
        export_btn = st.button("Export high-DPI PNG", key=_sk(p, "export_btn"))

    if render_btn or export_btn:
        if roi_df.empty:
            st.warning("No cells found for the selected ROI.")
        else:
            _do_render_and_show(
                prefix=p,
                roi_df=roi_df,
                layers=layers,
                image_path=image_path,
                available_channels=available_channels,
                analysis_df=analysis_df,
                config=config,
                batch_id=str(batch_id),
                roi_id=str(roi_id),
                annot_dir=annot_dir,
                cell_dir=cell_dir,
                show_legend=show_legend,
                show_title=show_title,
                bg_color=str(bg_color),
                title_override=str(title_override),
                legend_frac=legend_frac,
                preview_max_dim=preview_max_dim,
                preview_dpi=preview_dpi,
                export_dpi=export_dpi,
                is_export=bool(export_btn),
            )

    # Persist last preview between reruns
    elif _sk(p, "preview_fig") in st.session_state:
        fig_cached = st.session_state[_sk(p, "preview_fig")]
        if fig_cached is not None:
            try:
                st.pyplot(fig_cached, width="stretch")
            except Exception:
                pass
