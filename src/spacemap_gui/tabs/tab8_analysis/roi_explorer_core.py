"""
ROI Explorer — backend: image loading, GeoJSON parsing, cell shapes, and figure assembly.
No Streamlit imports.
"""
from __future__ import annotations

import difflib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba, Normalize
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, Voronoi

from spacemap.background_subtraction_utils import (
    get_channel_info,
    is_supported_image_file,
    load_preview_channels_compatible,
    normalize_image_stem,
)

# ── Palettes ───────────────────────────────────────────────────────────────────

_ANNOTATION_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#808080", "#ffe119", "#a9a9a9",
]

_CELL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

# ── Image channel utilities ────────────────────────────────────────────────────

def extract_image_channels(image_path: str | Path) -> list[dict[str, Any]]:
    """Return [{name, color, index}] for each channel in the image file."""
    try:
        channel_names, channel_colors, _meta, _mpp = get_channel_info(str(image_path))
    except Exception:
        return []
    results: list[dict[str, Any]] = []
    for idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
        hex_color = "#ffffff"
        try:
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                r, g, b = [int(min(255, max(0, float(c)))) for c in color[:3]]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
            elif isinstance(color, str) and color.startswith("#"):
                hex_color = color
        except Exception:
            pass
        results.append({"name": str(name), "color": hex_color, "index": idx})
    return results


def build_composite_image(
    image_path: str | Path,
    max_dim: int,
    selected_channels: list[dict[str, Any]],
) -> tuple[np.ndarray | None, float, float]:
    """
    Load and composite selected channels into an RGB float32 image.
    Returns (image_hwc, scale_x, scale_y) where scale = preview_size / full_size.
    """
    if not image_path or not Path(image_path).exists():
        return None, 1.0, 1.0
    indices = [ch["index"] for ch in selected_channels] if selected_channels else list(range(3))
    try:
        preview_data, full_shape, is_rgb = load_preview_channels_compatible(
            str(image_path), indices, max_dim=max_dim
        )
    except Exception:
        return None, 1.0, 1.0

    image: np.ndarray | None = None
    if is_rgb and "rgb" in preview_data:
        arr = np.asarray(preview_data["rgb"], dtype=float)
        image = arr / 255.0 if arr.max() > 1.5 else arr
    elif not selected_channels:
        channels = []
        for key in sorted(preview_data.keys())[:3]:
            arr = np.asarray(preview_data[key], dtype=float)
            lo, hi = np.nanpercentile(arr, [1, 99.5])
            channels.append(np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1))
        while len(channels) < 3:
            channels.append(np.zeros_like(channels[0]) if channels else np.zeros((64, 64)))
        image = np.dstack(channels[:3])
    else:
        composite: np.ndarray | None = None
        for ch in selected_channels:
            plane = preview_data.get(ch["index"])
            if plane is None:
                continue
            arr = np.asarray(plane, dtype=float)
            if arr.ndim != 2:
                continue
            if composite is None:
                composite = np.zeros((*arr.shape, 3), dtype=float)
            lo, hi = np.nanpercentile(arr, [1, 99.5])
            norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
            try:
                from matplotlib.colors import to_rgb as _to_rgb
                rgb = np.array(_to_rgb(ch.get("color", "#ffffff")), dtype=float)
            except Exception:
                rgb = np.ones(3)
            composite = np.maximum(composite, norm[:, :, None] * rgb[None, None, :])
        image = composite

    if image is None:
        return None, 1.0, 1.0
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    if full_shape and len(full_shape) >= 2:
        sx = image.shape[1] / float(full_shape[-1])
        sy = image.shape[0] / float(full_shape[-2])
    else:
        sx, sy = 1.0, 1.0
    return image, sx, sy


def find_image_for_batch(batch_id: str, image_dir: str | Path) -> Path | None:
    """Fuzzy-match batch_id to an image file stem in image_dir. Exact match wins."""
    if not image_dir or not Path(image_dir).exists():
        return None
    folder = Path(image_dir)
    b_norm = normalize_image_stem(batch_id).lower()
    best: tuple[float, Path] | None = None
    for f in folder.rglob("*"):
        if not f.is_file() or not is_supported_image_file(f.name):
            continue
        s_norm = normalize_image_stem(f.stem).lower()
        if s_norm == b_norm:
            return f
        score = difflib.SequenceMatcher(None, b_norm, s_norm).ratio()
        if best is None or score > best[0]:
            best = (score, f)
    if best and best[0] >= 0.5:
        return best[1]
    return None


# ── GeoJSON file-naming conventions ──────────────────────────────────────────
# These suffix patterns reflect how the Tab 4 / Tab 5 pipelines name their
# outputs.  They are used to give a score bonus when the caller specifies a
# ``role`` argument so that the right file type floats to the top.

# Tab 4: {stem}_cell_boundaries.geojson, {stem}_roi_bounding_boxes_coords_seg.geojson
_CELL_GJ_HINTS: list[str] = [
    "_cell_boundaries",
    "_cell_boundary",
    "_boundaries",
    "_seg",
    "_cell_seg",
    "_segmentation",
    "_cells",
]

# Tab 5: merged_tissue, zones, roi_bounding_boxes_coords (drawn tissue ROIs)
_ANNOT_GJ_HINTS: list[str] = [
    "_roi_bounding_boxes_coords",
    "_merged_tissue",
    "_zones",
    "_zone",
    "_annotations",
    "_annotation",
    "_tissue",
    "_merged",
    "stage1",
    "stage2",
]


def _role_bonus(stem_lower: str, role: str) -> float:
    """Return a small score bonus if the filename stem matches the expected role."""
    hints = _CELL_GJ_HINTS if role == "cells" else _ANNOT_GJ_HINTS if role == "annotations" else []
    for h in hints:
        if h in stem_lower:
            return 0.15
    return 0.0


# ── GeoJSON utilities ──────────────────────────────────────────────────────────

def fuzzy_match_geojson_files(
    batch_id: str,
    folder: str | Path,
    n: int = 8,
    role: str = "",
) -> list[tuple[float, Path]]:
    """
    Return top-n (score, path) GeoJSON file matches for batch_id in folder.

    Parameters
    ----------
    batch_id:
        The image/batch stem to match (e.g. ``"sample_01_merged"``).
    folder:
        Directory to search recursively for ``.geojson`` files.
    n:
        Maximum number of results.
    role:
        Optional context — ``"cells"`` or ``"annotations"`` — used to give a
        score bonus to files whose name follows the expected Tab 4/5 conventions
        (e.g. ``*_cell_boundaries.geojson`` for ``"cells"`` role).
    """
    if not folder or not Path(folder).exists():
        return []
    folder_path = Path(folder)
    b_norm = normalize_image_stem(batch_id).lower()
    candidates: list[tuple[float, Path]] = []
    for f in folder_path.rglob("*.geojson"):
        if not f.is_file():
            continue
        s_norm = normalize_image_stem(f.stem).lower()
        if s_norm == b_norm or s_norm.startswith(b_norm) or b_norm in s_norm:
            base_score = 1.0
        else:
            base_score = difflib.SequenceMatcher(None, b_norm, s_norm).ratio()
        score = min(1.0, base_score + _role_bonus(s_norm, role))
        candidates.append((score, f))
    candidates.sort(key=lambda t: (-t[0], t[1].name))
    return candidates[:n]


def _qupath_color_to_hex(color_int: int) -> str | None:
    """Convert a QuPath signed-ARGB integer to #rrggbb hex."""
    try:
        c = int(color_int) & 0xFFFFFF
        return f"#{c:06x}"
    except Exception:
        return None


def _col_is_microns(col_name: str) -> bool:
    """
    Heuristic: return True when a column's name suggests its values are in
    microns (e.g. ``"Centroid X µm"``, ``"centroid_x_um"``).
    This is used to decide whether to apply an MPP conversion before matching
    GeoJSON (pixel-space) centroids to CSV rows.
    """
    lowered = col_name.lower()
    return "µm" in col_name or " um" in lowered or "_um" in lowered or "micron" in lowered


def _guess_annotation_class(props: dict[str, Any]) -> str:
    """Extract annotation class name from GeoJSON feature properties (best-effort)."""
    # QuPath: properties.classification.name
    cls = props.get("classification")
    if isinstance(cls, dict):
        n = cls.get("name")
        if n and str(n).strip():
            return str(n).strip()
    # Generic keys in priority order
    for key in ("name", "label", "class", "tissue_type", "annotation_class",
                "region", "structure", "category", "objectType"):
        val = props.get(key)
        if val and isinstance(val, str) and val.strip() \
                and val.lower() not in ("feature", "polygon", "annotation", "unknown"):
            return val.strip()
    return "Unknown"


def _guess_annotation_color(props: dict[str, Any]) -> str | None:
    """Extract color hint from QuPath classification.colorRGB."""
    cls = props.get("classification")
    if isinstance(cls, dict):
        c = cls.get("colorRGB")
        if c is not None:
            return _qupath_color_to_hex(int(c))
    return None


def _guess_cell_phenotype(props: dict[str, Any], phenotype_col: str = "") -> str:
    """Extract cell phenotype from GeoJSON feature properties (best-effort)."""
    if phenotype_col:
        val = props.get(phenotype_col)
        if val is not None and str(val).strip():
            return str(val).strip()
    cls = props.get("classification")
    if isinstance(cls, dict):
        n = cls.get("name")
        if n and str(n).strip():
            return str(n).strip()
    for key in ("phenotype", "cell_type", "celltype", "Phenotype",
                "Cell_type", "type", "label", "class"):
        val = props.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return "Unknown"


def _parse_polygon_features(
    geojson_path: str | Path,
    flip_y: bool = False,
    image_height: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Parse a GeoJSON file into a list of polygon feature dicts.
    Each dict: {outer_ring: np.ndarray(N,2), centroid_x, centroid_y, properties}.
    """
    try:
        with open(geojson_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    root_type = data.get("type", "")
    if root_type == "FeatureCollection":
        raw_features = data.get("features", [])
    elif root_type == "Feature":
        raw_features = [data]
    elif root_type in ("Polygon", "MultiPolygon"):
        raw_features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raw_features = data if isinstance(data, list) else []

    results: list[dict[str, Any]] = []
    for feat in raw_features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        geom_type = geom.get("type", "")
        coords_list = geom.get("coordinates", [])

        rings: list[list] = []
        if geom_type == "Polygon" and coords_list:
            rings = [coords_list[0]]
        elif geom_type == "MultiPolygon":
            for poly in coords_list:
                if poly:
                    rings.append(poly[0])

        for ring in rings:
            if not ring or len(ring) < 3:
                continue
            try:
                pts: list[list[float]] = []
                for pt in ring:
                    if len(pt) >= 2:
                        x, y = float(pt[0]), float(pt[1])
                        if flip_y:
                            y = float(image_height) - y
                        pts.append([x, y])
                if len(pts) < 3:
                    continue
                arr = np.asarray(pts, dtype=float)
                results.append({
                    "outer_ring": arr,
                    "centroid_x": float(np.mean(arr[:, 0])),
                    "centroid_y": float(np.mean(arr[:, 1])),
                    "properties": props,
                })
            except Exception:
                continue
    return results


def load_annotation_geojson(
    geojson_path: str | Path,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    flip_y: bool = False,
    image_height: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Load a tissue-annotation GeoJSON file.
    Returns [{class_name, color_hint, coords (N×2 display-space)}].
    """
    parsed = _parse_polygon_features(geojson_path, flip_y=flip_y, image_height=image_height)
    results: list[dict[str, Any]] = []
    for feat in parsed:
        props = feat["properties"]
        coords = feat["outer_ring"].copy()
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
        results.append({
            "class_name": _guess_annotation_class(props),
            "color_hint": _guess_annotation_color(props),
            "coords": coords,
        })
    return results


def load_cell_geojson(
    geojson_path: str | Path,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    flip_y: bool = False,
    image_height: float = 0.0,
    phenotype_col: str = "",
) -> list[dict[str, Any]]:
    """
    Load a cell-segmentation GeoJSON file.
    Returns [{phenotype, coords (N×2 display-space), centroid_x, centroid_y}].
    """
    parsed = _parse_polygon_features(geojson_path, flip_y=flip_y, image_height=image_height)
    results: list[dict[str, Any]] = []
    for feat in parsed:
        props = feat["properties"]
        coords = feat["outer_ring"].copy()
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y
        results.append({
            "phenotype": _guess_cell_phenotype(props, phenotype_col),
            "coords": coords,
            "centroid_x": feat["centroid_x"] * scale_x,
            "centroid_y": feat["centroid_y"] * scale_y,
        })
    return results


def match_cell_geojson_to_csv(
    cell_features: list[dict[str, Any]],
    roi_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phenotype_col: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    max_dist: float = 50.0,
    mpp: float | None = None,
) -> list[dict[str, Any]]:
    """
    Enrich 'Unknown' phenotype cell features with the phenotype of the nearest CSV row.
    Only replaces features whose phenotype is 'Unknown' or empty.

    The ``mpp`` (microns-per-pixel) parameter is used when the CSV X/Y columns
    store coordinates in microns (e.g. columns containing µm / um).  In that
    case the CSV values are divided by ``mpp`` to convert to pixels before the
    nearest-neighbour search.  If ``mpp`` is ``None`` the unit is auto-detected
    from the column name; if it cannot be determined the CSV values are used as-
    is (assumed to be in the same pixel space as the GeoJSON).
    """
    if roi_df.empty or not all(c in roi_df.columns for c in (x_col, y_col, phenotype_col)):
        return cell_features
    csv_x = pd.to_numeric(roi_df[x_col], errors="coerce").to_numpy(float)
    csv_y = pd.to_numeric(roi_df[y_col], errors="coerce").to_numpy(float)
    # Auto-detect micron columns and convert to pixels
    if mpp is not None and mpp > 0:
        if _col_is_microns(x_col):
            csv_x = csv_x / mpp
        if _col_is_microns(y_col):
            csv_y = csv_y / mpp
    # Scale to preview space (same space as GeoJSON centroids after load)
    csv_x = csv_x * scale_x
    csv_y = csv_y * scale_y
    csv_ph = roi_df[phenotype_col].astype(str).to_numpy()
    valid = np.isfinite(csv_x) & np.isfinite(csv_y)
    if not valid.any():
        return cell_features
    tree = cKDTree(np.column_stack([csv_x[valid], csv_y[valid]]))
    ph_valid = csv_ph[valid]
    enriched: list[dict[str, Any]] = []
    for feat in cell_features:
        if feat.get("phenotype", "Unknown") in ("Unknown", ""):
            dist, idx = tree.query([feat["centroid_x"], feat["centroid_y"]], k=1)
            if dist <= max_dist:
                feat = {**feat, "phenotype": str(ph_valid[idx])}
        enriched.append(feat)
    return enriched


# ── Cell shape builders ────────────────────────────────────────────────────────

def _circle_coords(cx: float, cy: float, r: float, n: int = 16) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + np.cos(a) * r, cy + np.sin(a) * r])


def build_circle_shapes(
    roi_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phenotype_col: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    radius: float = 5.0,
) -> list[dict[str, Any]]:
    """Build circle-polygon shapes centred on each cell in roi_df."""
    if roi_df.empty or x_col not in roi_df.columns or y_col not in roi_df.columns:
        return []
    ph_arr = roi_df[phenotype_col].astype(str).to_numpy() if phenotype_col in roi_df.columns else np.full(len(roi_df), "Unknown")
    xs = pd.to_numeric(roi_df[x_col], errors="coerce").to_numpy(float) * scale_x
    ys = pd.to_numeric(roi_df[y_col], errors="coerce").to_numpy(float) * scale_y
    results: list[dict[str, Any]] = []
    for cx, cy, ph in zip(xs, ys, ph_arr):
        if np.isfinite(cx) and np.isfinite(cy):
            results.append({"phenotype": str(ph), "coords": _circle_coords(cx, cy, radius), "centroid_x": cx, "centroid_y": cy})
    return results


def build_voronoi_shapes(
    roi_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phenotype_col: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    cap_percentile: float = 50.0,
    fallback_radius: float = 15.0,
) -> list[dict[str, Any]]:
    """Build Voronoi cell shapes, capped to cap_percentile of NN distances."""
    if roi_df.empty or x_col not in roi_df.columns or y_col not in roi_df.columns:
        return []
    xs = pd.to_numeric(roi_df[x_col], errors="coerce").to_numpy(float) * scale_x
    ys = pd.to_numeric(roi_df[y_col], errors="coerce").to_numpy(float) * scale_y
    ph_arr = roi_df[phenotype_col].astype(str).to_numpy() if phenotype_col in roi_df.columns else np.full(len(xs), "Unknown")
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, ph_arr = xs[valid], ys[valid], ph_arr[valid]
    if len(xs) < 4:
        return build_circle_shapes(roi_df, x_col, y_col, phenotype_col, scale_x, scale_y, fallback_radius)
    tree = cKDTree(np.column_stack([xs, ys]))
    nn_dists, _ = tree.query(np.column_stack([xs, ys]), k=2)
    cap = float(np.nanpercentile(nn_dists[:, 1], cap_percentile))
    if not np.isfinite(cap) or cap <= 0:
        cap = fallback_radius
    span = max(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)), 1.0) * 2.0
    cx0, cy0 = float(np.mean(xs)), float(np.mean(ys))
    mirrors = np.array([
        [cx0 - span, cy0 - span], [cx0 + span, cy0 - span],
        [cx0 + span, cy0 + span], [cx0 - span, cy0 + span],
        [cx0, cy0 - span], [cx0, cy0 + span],
        [cx0 - span, cy0], [cx0 + span, cy0],
    ])
    all_pts = np.vstack([np.column_stack([xs, ys]), mirrors])
    try:
        vor = Voronoi(all_pts)
    except Exception:
        return build_circle_shapes(roi_df, x_col, y_col, phenotype_col, scale_x, scale_y, fallback_radius)
    results: list[dict[str, Any]] = []
    for i, (x, y, ph) in enumerate(zip(xs, ys, ph_arr)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            results.append({"phenotype": str(ph), "coords": _circle_coords(x, y, fallback_radius), "centroid_x": x, "centroid_y": y})
            continue
        verts = vor.vertices[region]
        if np.linalg.norm(verts - np.array([x, y]), axis=1).max() > cap * 3:
            results.append({"phenotype": str(ph), "coords": _circle_coords(x, y, min(cap, fallback_radius)), "centroid_x": x, "centroid_y": y})
        else:
            results.append({"phenotype": str(ph), "coords": verts, "centroid_x": x, "centroid_y": y})
    return results


# ── Cell filtering ─────────────────────────────────────────────────────────────

def apply_cell_filters(roi_df: pd.DataFrame, filter_rules: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Apply filter rules to roi_df.
    Each rule: {column, mode ('categorical'|'range'), values|min_val/max_val, include_missing}.
    """
    if not filter_rules or roi_df.empty:
        return roi_df
    mask = pd.Series(True, index=roi_df.index)
    for rule in filter_rules:
        col = str(rule.get("column", ""))
        mode = str(rule.get("mode", "categorical"))
        keep_missing = bool(rule.get("include_missing", True))
        if not col or col not in roi_df.columns:
            continue
        series = roi_df[col]
        is_missing = series.isna() | series.astype(str).str.strip().isin(["", "nan", "None", "NaN"])
        if mode == "categorical":
            allowed = {str(v) for v in (rule.get("values") or [])}
            row_match = series.astype(str).isin(allowed) if allowed else pd.Series(True, index=roi_df.index)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            row_match = pd.Series(True, index=roi_df.index)
            if rule.get("min_val") is not None:
                row_match &= numeric >= float(rule["min_val"])
            if rule.get("max_val") is not None:
                row_match &= numeric <= float(rule["max_val"])
        mask &= (row_match | is_missing) if keep_missing else (row_match & ~is_missing)
    return roi_df.loc[mask].copy()


# ── Color utilities ────────────────────────────────────────────────────────────

def default_annotation_colors(class_names: list[str]) -> dict[str, str]:
    """Assign default colors to a list of annotation class names."""
    return {name: _ANNOTATION_PALETTE[i % len(_ANNOTATION_PALETTE)]
            for i, name in enumerate(sorted(set(class_names)))}


def default_cell_colors(phenotypes: list[str]) -> dict[str, str]:
    """Assign default colors to phenotype names, preferring the tab7 color map."""
    try:
        from spacemap_gui.tabs.tab7_phenotyping.cell_visualization import _DEFAULT_COLOR_MAP as _CMAP
        _cmap: dict[str, str] = dict(_CMAP)
    except Exception:
        _cmap = {}
    return {name: _cmap.get(name, _CELL_PALETTE[i % len(_CELL_PALETTE)])
            for i, name in enumerate(sorted(set(phenotypes)))}


def build_continuous_color_map(
    series: pd.Series,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[dict[Any, str], float, float]:
    """
    Map a continuous Series to hex colors using a matplotlib colormap.
    Returns ({index → hex_color}, actual_vmin, actual_vmax).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    lo = float(vmin) if vmin is not None else (float(valid.min()) if len(valid) else 0.0)
    hi = float(vmax) if vmax is not None else (float(valid.max()) if len(valid) else 1.0)
    if hi <= lo:
        hi = lo + 1.0
    try:
        cmap = get_cmap(cmap_name)
    except Exception:
        cmap = get_cmap("viridis")
    norm = Normalize(vmin=lo, vmax=hi)
    mapping: dict[Any, str] = {}
    for idx, val in numeric.items():
        mapping[idx] = "#888888" if pd.isna(val) else to_hex(cmap(norm(float(val))))
    return mapping, lo, hi


# ── Layer renderers ────────────────────────────────────────────────────────────

def render_markers_layer(ax: Any, image: np.ndarray, opacity: float = 1.0) -> None:
    """Draw the composited multi-channel image on ax."""
    if image is None:
        return
    ax.imshow(np.clip(image, 0, 1), origin="upper", aspect="auto",
              alpha=float(np.clip(opacity, 0, 1)), zorder=1)


def render_annotation_layer(
    ax: Any,
    features: list[dict[str, Any]],
    color_map: dict[str, str],
    opacity: float = 0.35,
    fill: bool = True,
    outline_thickness: float = 1.5,
) -> list[dict[str, str]]:
    """Draw tissue-annotation polygons. Returns [{label, color}] for the legend."""
    legend_seen: dict[str, str] = {}
    for feat in features:
        cls = feat.get("class_name", "Unknown")
        coords = feat.get("coords")
        if coords is None or len(coords) < 3:
            continue
        color = color_map.get(cls, "#888888")
        try:
            r, g, b, _ = to_rgba(color)
        except Exception:
            r, g, b = 0.5, 0.5, 0.5
        a = float(np.clip(opacity, 0, 1))
        fc = (r, g, b, a) if fill else (0, 0, 0, 0)
        ec = (r, g, b, min(1.0, a + 0.25))
        ax.add_patch(mpatches.Polygon(coords, closed=True, facecolor=fc, edgecolor=ec,
                                      linewidth=outline_thickness, zorder=3))
        legend_seen[cls] = color
    return [{"label": k, "color": v} for k, v in sorted(legend_seen.items())]


def render_cell_layer(
    ax: Any,
    features: list[dict[str, Any]],
    color_map: dict[str, str],
    opacity: float = 0.85,
    fill: bool = True,
    outline_thickness: float = 0.5,
) -> list[dict[str, str]]:
    """Draw cell-segmentation polygons. Returns [{label, color}] for the legend."""
    legend_seen: dict[str, str] = {}
    for feat in features:
        ph = feat.get("phenotype", "Unknown")
        coords = feat.get("coords")
        if coords is None or len(coords) < 3:
            continue
        color = color_map.get(ph, "#888888")
        try:
            r, g, b, _ = to_rgba(color)
        except Exception:
            r, g, b = 0.5, 0.5, 0.5
        a = float(np.clip(opacity, 0, 1))
        fc = (r, g, b, a) if fill else (0, 0, 0, 0)
        ec = (r, g, b, min(1.0, a + 0.15))
        ax.add_patch(mpatches.Polygon(coords, closed=True, facecolor=fc, edgecolor=ec,
                                      linewidth=outline_thickness, zorder=5))
        legend_seen[ph] = color
    return [{"label": k, "color": v} for k, v in sorted(legend_seen.items())]


def render_scatter_layer(
    ax: Any,
    roi_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    scale_x: float,
    scale_y: float,
    color_map: dict[str, str],
    phenotype_col: str = "",
    size: float = 20.0,
    opacity: float = 0.8,
    index_color_map: dict[Any, str] | None = None,
) -> list[dict[str, str]]:
    """Draw cell scatter points. Returns [{label, color}] for the legend."""
    if roi_df.empty or x_col not in roi_df.columns or y_col not in roi_df.columns:
        return []
    xs = pd.to_numeric(roi_df[x_col], errors="coerce").to_numpy(float) * scale_x
    ys = pd.to_numeric(roi_df[y_col], errors="coerce").to_numpy(float) * scale_y
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    a = float(np.clip(opacity, 0, 1))

    # Continuous coloring (per-row index)
    if index_color_map is not None:
        idx_arr = roi_df.index[valid]
        colors = [index_color_map.get(i, "#888888") for i in idx_arr]
        ax.scatter(xs, ys, c=colors, s=size, alpha=a, linewidths=0, zorder=5)
        return []

    phs = roi_df[phenotype_col].astype(str).to_numpy()[valid] if phenotype_col in roi_df.columns else np.full(len(xs), "Unknown")
    legend_seen: dict[str, str] = {}
    for ph in sorted(set(phs)):
        m = phs == ph
        c = color_map.get(ph, "#888888")
        ax.scatter(xs[m], ys[m], c=c, s=size, alpha=a, linewidths=0, zorder=5)
        legend_seen[ph] = c
    return [{"label": k, "color": v} for k, v in sorted(legend_seen.items())]


# ── Legend ─────────────────────────────────────────────────────────────────────

def draw_roi_legend(
    ax_leg: Any,
    legend_sections: list[dict[str, Any]],
    bg: str = "#111111",
    fg: str = "#ffffff",
) -> None:
    """
    Draw a multi-section legend on ax_leg.
    legend_sections: [{title, items: [{label, color}]}]
    """
    ax_leg.set_facecolor(bg)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    y = 0.97
    dy_h = 0.038
    dy_i = 0.028
    sw, sh = 0.11, 0.018
    tx = 0.22
    for sec in legend_sections:
        items = sec.get("items") or []
        if not items:
            continue
        title = str(sec.get("title", ""))
        if title:
            ax_leg.text(0.04, y, title, fontsize=7.5, fontweight="bold", color=fg,
                        transform=ax_leg.transAxes, va="top")
            y -= dy_h
        for item in items:
            if y < 0.02:
                break
            color = str(item.get("color", "#888888"))
            try:
                r, g, b, _ = to_rgba(color)
                patch_color: Any = (r, g, b, 1.0)
            except Exception:
                patch_color = color
            ax_leg.add_patch(mpatches.FancyBboxPatch(
                (0.04, y - sh), sw, sh,
                boxstyle="round,pad=0.003",
                facecolor=patch_color,
                edgecolor="#ffffff66",
                linewidth=0.4,
                transform=ax_leg.transAxes,
                clip_on=False,
            ))
            ax_leg.text(tx, y - sh / 2, str(item.get("label", "")),
                        fontsize=6.8, color=fg, transform=ax_leg.transAxes, va="center")
            y -= dy_i
        y -= dy_h * 0.4


# ── Main figure builder ────────────────────────────────────────────────────────

def build_roi_figure(
    roi_df: pd.DataFrame,
    image: np.ndarray | None,
    layers: list[dict[str, Any]],
    scale_x: float,
    scale_y: float,
    x_col: str,
    y_col: str,
    title: str = "",
    dpi: int = 150,
    fig_inches: float = 8.0,
    legend_fraction: float = 0.22,
    bg_color: str = "#111111",
    show_legend: bool = True,
    show_title: bool = True,
) -> Figure:
    """
    Assemble the full ROI figure with all enabled layers and an optional legend.

    Each layer dict (from _build_layer_data) must have:
      enabled, type, opacity
    Plus type-specific keys:
      markers:     selected_channels (list[dict])
      annotations: _features, color_map, fill, outline_thickness
      cells:       _features, color_map, fill, outline_thickness
      scatter:     _filtered_df, phenotype_col, color_map, size
                   optional: _index_color_map
    """
    # Determine canvas size from image or data extent
    if image is not None:
        h_px, w_px = image.shape[0], image.shape[1]
    elif not roi_df.empty and x_col in roi_df.columns and y_col in roi_df.columns:
        xs = pd.to_numeric(roi_df[x_col], errors="coerce") * scale_x
        ys = pd.to_numeric(roi_df[y_col], errors="coerce") * scale_y
        w_px = max(int(xs.max()) + 60, 400) if xs.notna().any() else 400
        h_px = max(int(ys.max()) + 60, 400) if ys.notna().any() else 400
    else:
        h_px, w_px = 500, 500

    aspect = h_px / max(w_px, 1)
    lf = legend_fraction if show_legend else 0.0

    if lf > 0:
        # Preserve the image's natural W×H ratio: the image panel is always
        # fig_inches wide and fig_inches*aspect tall.  The legend is appended
        # to the right by widening the figure, NOT by squishing the image.
        # `lf` is treated as (legend width / image width).
        total_w_in = fig_inches * (1.0 + lf)
        img_frac = fig_inches / total_w_in  # fraction of total figure width used by image
        gap = 0.008  # small separator in normalised figure-width units
        fig = plt.figure(figsize=(total_w_in, fig_inches * aspect), dpi=dpi, facecolor=bg_color)
        ax = fig.add_axes([0.0, 0.0, img_frac - gap, 1.0])
        ax_leg = fig.add_axes([img_frac + gap, 0.0, 1.0 - img_frac - 2 * gap, 1.0])
        ax_leg.set_facecolor(bg_color)
    else:
        fig = plt.figure(figsize=(fig_inches, fig_inches * aspect), dpi=dpi, facecolor=bg_color)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax_leg = None

    ax.set_facecolor(bg_color)
    ax.axis("off")
    legend_sections: list[dict[str, Any]] = []

    for layer in layers:
        if not layer.get("enabled", True):
            continue
        ltype = str(layer.get("type", ""))
        op = float(layer.get("opacity", 1.0))

        if ltype == "markers" and image is not None:
            render_markers_layer(ax, image, opacity=op)
            sel = layer.get("selected_channels") or []
            items = [{"label": ch["name"], "color": ch["color"]} for ch in sel if ch.get("name")]
            if items:
                legend_sections.append({"title": "Markers", "items": items})

        elif ltype == "annotations":
            items = render_annotation_layer(
                ax, layer.get("_features", []), layer.get("color_map", {}),
                opacity=op, fill=bool(layer.get("fill", True)),
                outline_thickness=float(layer.get("outline_thickness", 1.5)),
            )
            if items:
                legend_sections.append({"title": str(layer.get("title", "Annotations")), "items": items})

        elif ltype == "cells":
            items = render_cell_layer(
                ax, layer.get("_features", []), layer.get("color_map", {}),
                opacity=op, fill=bool(layer.get("fill", True)),
                outline_thickness=float(layer.get("outline_thickness", 0.5)),
            )
            if items:
                legend_sections.append({"title": str(layer.get("title", "Cells")), "items": items})

        elif ltype == "scatter":
            items = render_scatter_layer(
                ax, layer.get("_filtered_df", roi_df), x_col, y_col, scale_x, scale_y,
                color_map=layer.get("color_map", {}),
                phenotype_col=str(layer.get("phenotype_col", "")),
                size=float(layer.get("size", 20.0)),
                opacity=op,
                index_color_map=layer.get("_index_color_map"),
            )
            if items:
                legend_sections.append({"title": str(layer.get("title", "Cells")), "items": items})

    # Set axis limits
    if image is not None:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
    elif not roi_df.empty and x_col in roi_df.columns and y_col in roi_df.columns:
        xs = pd.to_numeric(roi_df[x_col], errors="coerce") * scale_x
        ys = pd.to_numeric(roi_df[y_col], errors="coerce") * scale_y
        xr = float(xs.max()) - float(xs.min())
        yr = float(ys.max()) - float(ys.min())
        ax.set_xlim(float(xs.min()) - xr * 0.04, float(xs.max()) + xr * 0.04)
        ax.set_ylim(float(ys.max()) + yr * 0.04, float(ys.min()) - yr * 0.04)

    if show_title and title:
        ax.set_title(title, color="#ffffff", fontsize=9, pad=3)

    if ax_leg is not None and legend_sections:
        draw_roi_legend(ax_leg, legend_sections, bg=bg_color)
    elif ax_leg is not None:
        ax_leg.set_facecolor(bg_color)
        ax_leg.axis("off")

    return fig
