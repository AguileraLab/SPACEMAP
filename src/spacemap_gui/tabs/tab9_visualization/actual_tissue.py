from __future__ import annotations

import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from PIL import Image, ImageColor, ImageDraw, ImageFont
from shapely.affinity import scale as scale_geometry
from shapely.affinity import translate as translate_geometry
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon, box, shape
from shapely.ops import unary_union
import scipy.spatial as spatial

from spacemap_gui.ui_utils import directory_input_with_browse
from spacemap.background_subtraction_utils import get_channel_info
from spacemap.utils import save_roi_pyramidal_ome_tiff
from spacemap_gui import terminal


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab9', message, step='Actual Tissue', level=level)


def _get_widget_index(key: str, options: list[Any], default_index: int = 0) -> int:
    """Helper to return the index of the option currently in session state, or a fallback."""
    if key in st.session_state:
        val = st.session_state[key]
        if val in options:
            return options.index(val)
    return default_index



@st.cache_data(show_spinner=False)
def _load_cell_segmentations_cached(file_path: str, mtime: float) -> list[dict[str, Any]]:
    """Load and parse cell polygons from GeoJSON, returning list of cell dicts with geometry and centroid."""
    try:
        with open(file_path, "r") as handle:
            data = json.load(handle)
        
        features = []
        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection":
                features = data.get("features", [])
            elif data.get("type") == "Feature":
                features = [data]
        
        cells = []
        for idx, feature in enumerate(features):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            try:
                geom = shape(geometry)
            except Exception:
                continue
            if geom.is_empty:
                continue
            
            props = feature.get("properties", {}) or {}
            # Get cell label or ID
            cell_id = props.get("id") or props.get("cell_id") or props.get("label") or props.get("name") or idx
            
            centroid = geom.centroid
            cells.append({
                "cell_id": cell_id,
                "geometry": geom,
                "x": centroid.x,
                "y": centroid.y,
                "properties": props
            })
        return cells
    except Exception as e:
        _log_console(f"Error loading cell segmentation GeoJSON: {e}", level="error")
        return []


def _match_cells_to_csv(
    cells: list[dict[str, Any]],
    csv_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_dist: float = 30.0
) -> list[dict[str, Any]]:
    """Map cell polygons to CSV rows using KD-Tree spatial proximity matching."""
    if not cells or csv_df.empty:
        return cells
    
    try:
        # Convert columns to numeric, dropping NaNs
        df_clean = csv_df.dropna(subset=[x_col, y_col]).copy()
        df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[x_col, y_col])
        
        if df_clean.empty:
            return cells
            
        csv_coords = df_clean[[x_col, y_col]].to_numpy()
        tree = spatial.cKDTree(csv_coords)
        
        matched_cells = []
        for cell in cells:
            px, py = cell["x"], cell["y"]
            dist, idx = tree.query([px, py])
            new_cell = cell.copy()
            if dist <= max_dist:
                csv_row = df_clean.iloc[idx]
                new_cell["csv_data"] = csv_row.to_dict()
                new_cell["matched"] = True
                new_cell["match_dist"] = dist
            else:
                new_cell["csv_data"] = {}
                new_cell["matched"] = False
                new_cell["match_dist"] = dist
            matched_cells.append(new_cell)
        return matched_cells
    except Exception as e:
        _log_console(f"Error in cell matching: {e}", level="error")
        return cells


def _generate_voronoi_cells(
    csv_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_dist: float,
    active_annotations_union: Any = None
) -> list[dict[str, Any]]:
    """
    Generate Voronoi cell polygons centered on cell centroids in csv_df.
    Each cell's region is intersected with a circle of radius max_dist.
    If active_annotations_union is provided, the region is also intersected with it.
    """
    import numpy as np
    from scipy import spatial
    from shapely.geometry import Point, Polygon
    
    cells = []
    if csv_df.empty:
        return cells
        
    try:
        df_clean = csv_df.dropna(subset=[x_col, y_col]).copy()
        df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
        df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
        df_clean = df_clean.dropna(subset=[x_col, y_col])
        
        if df_clean.empty:
            return cells
            
        coords = df_clean[[x_col, y_col]].to_numpy()
        n_points = len(coords)
        
        # If we have too few points to run Voronoi, fallback to circular buffers
        if n_points < 4:
            for idx, row in df_clean.iterrows():
                cx = float(row[x_col])
                cy = float(row[y_col])
                geom = Point(cx, cy).buffer(max_dist)
                if active_annotations_union is not None:
                    try:
                        geom = geom.intersection(active_annotations_union)
                    except Exception:
                        pass
                if not geom.is_empty:
                    cells.append({
                        "cell_id": idx,
                        "geometry": geom,
                        "x": cx,
                        "y": cy,
                        "csv_data": row.to_dict(),
                        "matched": True
                    })
            return cells
            
        # To handle infinite regions, add dummy boundary points far outside
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        margin = max(max_dist * 10, 5000.0)
        
        dummy_points = np.array([
            [min_x - margin, min_y - margin],
            [min_x - margin, max_y + margin],
            [max_x + margin, min_y - margin],
            [max_x + margin, max_y + margin],
            [min_x - margin, (min_y + max_y)/2],
            [max_x + margin, (min_y + max_y)/2],
            [(min_x + max_x)/2, min_y - margin],
            [(min_x + max_x)/2, max_y + margin]
        ])
        
        all_points = np.vstack([coords, dummy_points])
        vor = spatial.Voronoi(all_points)
        
        for i in range(n_points):
            row = df_clean.iloc[i]
            cx = coords[i, 0]
            cy = coords[i, 1]
            
            reg_idx = vor.point_region[i]
            if reg_idx == -1:
                geom = Point(cx, cy).buffer(max_dist)
            else:
                vert_indices = vor.regions[reg_idx]
                if not vert_indices or -1 in vert_indices:
                    geom = Point(cx, cy).buffer(max_dist)
                else:
                    try:
                        vertices = vor.vertices[vert_indices]
                        poly = Polygon(vertices)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        circle = Point(cx, cy).buffer(max_dist)
                        geom = poly.intersection(circle)
                    except Exception:
                        geom = Point(cx, cy).buffer(max_dist)
            
            if active_annotations_union is not None:
                try:
                    geom = geom.intersection(active_annotations_union)
                except Exception:
                    pass
                    
            if not geom.is_empty:
                cells.append({
                    "cell_id": df_clean.index[i],
                    "geometry": geom,
                    "x": cx,
                    "y": cy,
                    "csv_data": row.to_dict(),
                    "matched": True
                })
                
        return cells
    except Exception as e:
        _log_console(f"Error generating Voronoi cells: {e}", level="error")
        return cells


def _draw_cell_polygons(
    image: Image.Image,
    cells: list[dict[str, Any]],
    crop_box_native: tuple[int, int, int, int],
    crop_box_px: tuple[int, int, int, int] | None,
    scale_x: float,
    scale_y: float,
    color_mode: str,
    single_color: str,
    phenotype_col: str | None,
    phenotype_colors: dict[str, str],
    continuous_col: str | None,
    continuous_cmap_name: str,
    continuous_limits: tuple[float, float],
    fill_alpha: float,
    line_width: int,
    outline_only: bool,
    compare_col1: str | None = None,
    compare_col2: str | None = None,
    compare_match_color: str = "#00ff00",
    compare_mismatch_color: str = "#ff0000",
    outline_color_mode: str = "Match Cell Color",
    outline_color: str = "#ffffff",
) -> Image.Image:
    if not cells:
        return image
    
    left, top, right, bottom = crop_box_native
    crop_geom = box(left, top, right, bottom)
    
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    
    # Prepare continuous colormap if needed
    cmap = None
    if color_mode == "Continuous Column" and continuous_col:
        try:
            cmap = plt.get_cmap(continuous_cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")
            
    c_min, c_max = continuous_limits
    if c_max <= c_min:
        c_max = c_min + 1e-6
        
    for cell in cells:
        geom = cell["geometry"]
        if not crop_geom.intersects(geom):
            continue
            
        # Transform using scale then translate to local crop coordinates to ensure perfect alignment with overlays
        local_scaled = scale_geometry(geom, xfact=scale_x, yfact=scale_y, origin=(0, 0))
        if crop_box_px is not None:
            local_scaled = translate_geometry(local_scaled, xoff=-float(crop_box_px[0]), yoff=-float(crop_box_px[1]))
        
        # Determine color
        color_hex = single_color
        if color_mode == "Categorical / Phenotype Column" and phenotype_col:
            val = cell.get("csv_data", {}).get(phenotype_col)
            if val is not None:
                color_hex = phenotype_colors.get(str(val), "#777777")
            else:
                color_hex = "#777777"
        elif color_mode == "Continuous Column" and continuous_col and cmap:
            val = cell.get("csv_data", {}).get(continuous_col)
            if val is not None:
                try:
                    val_f = float(val)
                    norm_val = np.clip((val_f - c_min) / (c_max - c_min), 0.0, 1.0)
                    rgba = cmap(norm_val)
                    color_hex = "#%02x%02x%02x" % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                except Exception:
                    color_hex = "#777777"
            else:
                color_hex = "#777777"
        elif color_mode == "Column Comparison" and compare_col1 and compare_col2:
            val1 = cell.get("csv_data", {}).get(compare_col1)
            val2 = cell.get("csv_data", {}).get(compare_col2)
            if val1 is None or val2 is None or pd.isna(val1) or pd.isna(val2):
                color_hex = "#777777"
            else:
                match = False
                try:
                    if np.isclose(float(val1), float(val2)):
                        match = True
                except Exception:
                    pass
                if not match and str(val1).strip() == str(val2).strip():
                    match = True
                color_hex = compare_match_color if match else compare_mismatch_color
                
        rgb = _rgb_tuple(color_hex)
        if outline_only or fill_alpha <= 0:
            fill = None
        else:
            fill = rgb + (max(0, min(255, int(fill_alpha * 255))),)
            
        if outline_color_mode == "Custom Color":
            outline = _rgb_tuple(outline_color) + (255,)
        else:
            outline = rgb + (255,)
        
        for polygon in _geometry_to_polygons(local_scaled):
            exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
            if len(exterior) >= 3:
                draw.polygon(exterior, fill=fill, outline=outline, width=int(line_width))
            for hole in polygon.interiors:
                interior = [(float(x), float(y)) for x, y in hole.coords]
                if len(interior) >= 3:
                    draw.polygon(interior, fill=(0, 0, 0, 0), outline=outline if outline_only else None, width=int(line_width))
                    
    return Image.alpha_composite(base, overlay)


IMAGE_EXTENSIONS = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".qptiff")
GEOJSON_EXTENSIONS = (".geojson", ".json")
DEFAULT_MARKER_FALLBACKS = ["DAPI", "Pan-Cytokeratin", "CD45", "CD3e", "CD20", "CD68"]
SCALE_BAR_UM_CHOICES = [5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0]
ADORNMENT_POSITIONS = ["auto", "top-left", "top-right", "bottom-left", "bottom-right", "below image"]


def _preferred_existing_dir(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else ""


def _sanitize_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_") or "item"


def _sanitize_filename_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")
    return cleaned or "item"


def _column_options(columns: list[str]) -> list[str]:
    return [""] + list(columns)


def _preferred_or_blank(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else ""


def _format_optional_column(value: str) -> str:
    return value if value else "(not selected)"


def _safe_sorted(values: list[Any]) -> list[Any]:
    try:
        return sorted(values)
    except Exception:
        return values


def _normalize_stem(name: str) -> str:
    stem = Path(str(name)).stem
    if stem.endswith(".ome"):
        stem = stem[:-4]
    stem = stem.replace("_cell_boundaries", "")
    stem = stem.replace("_roi_bounding_boxes_coords", "")
    return stem.strip().lower()


def _sample_prefix(text: str) -> str:
    sample = _normalize_stem(text)
    sample = re.sub(r"_roi_\d+$", "", sample)
    sample = re.sub(r"_roi$", "", sample)
    return sample


def _extract_roi_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "all":
        return None
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text


def _candidate_match_tokens(batch_value: Any, tissue_value: Any) -> list[str]:
    batch = str(batch_value or "").strip()
    roi = _extract_roi_token(tissue_value)
    tokens: list[str] = []
    if batch:
        tokens.append(batch)
    batch_base = _sample_prefix(batch)
    if batch_base and batch_base not in tokens:
        tokens.append(batch_base)
    if roi:
        if batch and f"_roi_{roi}" not in batch.lower():
            tokens.append(f"{batch}_roi_{roi}")
        if batch_base:
            tokens.append(f"{batch_base}_roi_{roi}")
    deduped: list[str] = []
    for token in tokens:
        normalized = _normalize_stem(token)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _score_name_match(candidate_name: str, tokens: list[str]) -> float:
    stem = _normalize_stem(candidate_name)
    if not stem:
        return -1.0
    best = -1.0
    stem_prefix = _sample_prefix(stem)
    for token in tokens:
        if stem == token:
            best = max(best, 100.0)
        elif stem.startswith(token):
            best = max(best, 85.0 - abs(len(stem) - len(token)) * 0.1)
        elif token in stem:
            best = max(best, 70.0 - abs(len(stem) - len(token)) * 0.1)
        elif stem_prefix and stem_prefix == _sample_prefix(token):
            best = max(best, 35.0)
    return best


def _list_files_cached(directory: str, extensions: tuple[str, ...], recursive: bool) -> list[str]:
    root = Path(directory)
    if not directory or not root.exists() or not root.is_dir():
        return []
    pattern_iter = root.rglob("*") if recursive else root.iterdir()
    out: list[str] = []
    for path in pattern_iter:
        if path.is_file() and path.name.lower().endswith(tuple(ext.lower() for ext in extensions)):
            out.append(str(path))
    return sorted(out)


def _best_match_file(paths: list[str], batch_value: Any, tissue_value: Any) -> str | None:
    if not paths:
        return None
    tokens = _candidate_match_tokens(batch_value, tissue_value)
    if not tokens:
        return paths[0]
    ranked = sorted(
        (( _score_name_match(Path(path).name, tokens), path) for path in paths),
        key=lambda item: item[0],
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 0 else paths[0]


def _best_match_value(options: list[Any], source_value: Any, tissue_value: Any = None) -> Any | None:
    if not options:
        return None
    tokens = _candidate_match_tokens(source_value, tissue_value)
    if not tokens:
        return options[0]
    ranked = sorted(
        ((_score_name_match(str(option), tokens), option) for option in options),
        key=lambda item: item[0],
        reverse=True,
    )
    return ranked[0][1] if ranked and ranked[0][0] >= 0 else options[0]


@st.cache_data(show_spinner=False)
def _get_channel_info_cached(image_path: str, mtime: float):
    return get_channel_info(image_path)


@st.cache_data(show_spinner=False)
def _load_preview_planes(
    image_path: str,
    mtime: float,
    channel_indices: tuple[int, ...],
    max_dim: int | None,
) -> tuple[dict[int, np.ndarray], tuple[int, int]]:
    loaded: dict[int, np.ndarray] = {}
    with tifffile.TiffFile(image_path) as tif:
        page0 = tif.pages[0]
        full_shape = page0.shape
        full_h = int(full_shape[-2])
        full_w = int(full_shape[-1])
        if max_dim is None or int(max_dim) <= 0:
            scale = 1
        else:
            scale = max(int(max(full_h, full_w) // max(int(max_dim), 1)), 1)

        if len(tif.pages) > 1 and page0.ndim == 2:
            for idx in channel_indices:
                if 0 <= idx < len(tif.pages):
                    arr = tif.pages[idx].asarray()
                    if scale > 1:
                        arr = arr[::scale, ::scale]
                    loaded[int(idx)] = np.asarray(arr)
        else:
            arr = np.asarray(tif.series[0].asarray())
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
                    arr = np.moveaxis(arr, -1, 0)
                elif arr.shape[1] < arr.shape[0] and arr.shape[1] < arr.shape[2]:
                    arr = np.moveaxis(arr, 1, 0)
            if scale > 1:
                arr = arr[:, ::scale, ::scale]
            for idx in channel_indices:
                if 0 <= idx < arr.shape[0]:
                    loaded[int(idx)] = np.asarray(arr[int(idx)])

    return loaded, (full_h, full_w)


def _normalize_plane(plane: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    arr = np.asarray(plane, dtype=float)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=float)
    low = max(0.0, min(float(low_percentile), 100.0))
    high = max(low + 1e-6, min(float(high_percentile), 100.0))
    lo = float(np.nanpercentile(arr, low))
    hi = float(np.nanpercentile(arr, high))
    if not np.isfinite(lo):
        lo = float(np.nanmin(arr)) if np.size(arr) else 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out


def _hex_from_rgb(color_value: Any) -> str:
    if isinstance(color_value, str):
        text = color_value.strip()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
            return text
        if re.fullmatch(r"[0-9A-Fa-f]{6}", text):
            return f"#{text}"
    if isinstance(color_value, (list, tuple)) and len(color_value) >= 3:
        try:
            return "#%02x%02x%02x" % tuple(max(0, min(255, int(v))) for v in color_value[:3])
        except Exception:
            pass
    return "#ffffff"


def _rgb_tuple(color_value: Any) -> tuple[int, int, int]:
    try:
        rgb = ImageColor.getrgb(_hex_from_rgb(color_value))
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception:
        return (255, 255, 255)


def _default_marker_selection(channel_names: list[str]) -> list[str]:
    defaults = [marker for marker in DEFAULT_MARKER_FALLBACKS if marker in channel_names]
    if defaults:
        return defaults[:4]
    return channel_names[: min(4, len(channel_names))]


def _compose_preview_image(
    image_path: Path,
    channel_names: list[str],
    color_map: dict[str, str],
    selected_markers: list[str],
    max_dim: int | None,
    low_percentile: float,
    high_percentile: float,
) -> tuple[Image.Image, tuple[int, int]]:
    valid_markers = [marker for marker in selected_markers if marker in channel_names]
    if not valid_markers and channel_names:
        valid_markers = [channel_names[0]]
    channel_indices = tuple(channel_names.index(marker) for marker in valid_markers)
    loaded, (full_h, full_w) = _load_preview_planes(
        str(image_path),
        image_path.stat().st_mtime,
        channel_indices,
        max_dim,
    )
    if not loaded:
        fallback_dim = int(max_dim) if isinstance(max_dim, int) and max_dim > 0 else 512
        return Image.new("RGB", (fallback_dim, fallback_dim), (0, 0, 0)), (full_h, full_w)

    first = next(iter(loaded.values()))
    comp = np.zeros((first.shape[0], first.shape[1], 3), dtype=float)
    for marker in valid_markers:
        idx = channel_names.index(marker)
        plane = loaded.get(idx)
        if plane is None:
            continue
        plane_norm = _normalize_plane(plane, low_percentile=low_percentile, high_percentile=high_percentile)
        rgb = np.asarray(_rgb_tuple(color_map.get(marker, "#ffffff")), dtype=float) / 255.0
        comp += plane_norm[..., None] * rgb

    comp = np.clip(comp, 0.0, 1.0)
    image = Image.fromarray((comp * 255.0).astype(np.uint8), mode="RGB")
    return image, (full_h, full_w)


def _render_actual_image(
    composite_image: Image.Image,
    full_hw: tuple[int, int],
    crop_box_full: tuple[float, float, float, float] | None,
    crop_mode: str,
    transformed_roi_geom,
    selected_overlay_records: list[dict[str, Any]],
    overlay_geometries: dict[str, Any],
    mask_source: str,
    outside_mode: str,
    outside_custom_color: str,
    outside_dim_strength: float,
    show_roi_outline: bool,
    roi_outline_color: str,
    overlay_color_map: dict[str, str],
    overlay_style: str,
    overlay_fill_alpha: float,
    overlay_line_width: int,
    show_annotation_overlays: bool,
    show_cells: bool = False,
    cells_list: list[dict[str, Any]] = None,
    cell_color_mode: str = "Single Color",
    cell_single_color: str = "#ffffff",
    cell_phenotype_col: str | None = None,
    cell_phenotype_colors: dict[str, str] = None,
    cell_continuous_col: str | None = None,
    cell_continuous_cmap: str = "viridis",
    cell_continuous_limits: tuple[float, float] = (0.0, 1.0),
    cell_fill_alpha: float = 0.25,
    cell_line_width: int = 1,
    cell_outline_only: bool = False,
    layer_order: list[str] = ("Markers", "Tissue Annotations", "Cell Segmentations"),
    markers_opacity: float = 1.0,
    show_markers: bool = True,
    cell_compare_col1: str | None = None,
    cell_compare_col2: str | None = None,
    cell_compare_match_color: str = "#00ff00",
    cell_compare_mismatch_color: str = "#ff0000",
    cell_outline_color_mode: str = "Match Cell Color",
    cell_outline_color: str = "#ffffff",
    background_color: str = "#ffffff",
) -> tuple[Image.Image, tuple[int, int, int, int] | None, Any]:
    scale_x = composite_image.width / float(max(full_hw[1], 1))
    scale_y = composite_image.height / float(max(full_hw[0], 1))
    crop_box_px = _px_crop_box_from_full(crop_box_full, scale_x, scale_y, composite_image.width, composite_image.height)
    crop_box_native = _crop_box_native_from_full(crop_box_full, full_hw)

    if crop_box_px is not None:
        preview_image = composite_image.crop(crop_box_px)
    else:
        preview_image = composite_image.copy()

    transformed_roi_geom_local = None
    if transformed_roi_geom is not None:
        transformed_roi_geom_local = _transform_geometry_to_crop(transformed_roi_geom, scale_x, scale_y, crop_box_px)

    overlay_geometries_local: dict[str, Any] = {}
    for record in selected_overlay_records:
        overlay_geometries_local[record["key"]] = _transform_geometry_to_crop(
            overlay_geometries.get(record["key"]),
            scale_x,
            scale_y,
            crop_box_px,
        )

    mask_geom = None
    if crop_mode == "Selected ROI only" and transformed_roi_geom_local is not None:
        mask_geom = transformed_roi_geom_local

    selected_mask_geom = None
    mask_sources_list = []
    if isinstance(mask_source, str):
        if mask_source != "None" and mask_source:
            mask_sources_list = [mask_source]
    elif mask_source:
        mask_sources_list = [ms for ms in mask_source if ms != "None"]
        
    geoms_to_union = []
    for ms in mask_sources_list:
        if ms == "Matched tissue ROI" and transformed_roi_geom_local is not None:
            geoms_to_union.append(transformed_roi_geom_local)
        else:
            g = overlay_geometries_local.get(ms)
            if g is not None:
                geoms_to_union.append(g)
                
    if geoms_to_union:
        try:
            selected_mask_geom = unary_union(geoms_to_union)
        except Exception as e:
            _log_console(f"[ERROR] Failed to union mask geometries: {e}", level="error")

    if mask_geom is None:
        mask_geom = selected_mask_geom
    elif selected_mask_geom is not None:
        try:
            mask_geom = mask_geom.intersection(selected_mask_geom)
        except Exception:
            mask_geom = selected_mask_geom

    # Filter layer order to unique enabled layers
    active_layers = []
    for layer in layer_order:
        if layer == "Markers" and not show_markers:
            continue
        if layer == "Tissue Annotations" and not show_annotation_overlays:
            continue
        if layer == "Cell Segmentations" and not show_cells:
            continue
        if layer != "None" and layer not in active_layers:
            active_layers.append(layer)

    # Initialize canvas dimensions
    w = preview_image.width
    h = preview_image.height
    bg_rgb = _rgb_tuple(background_color)
    rendered = Image.new("RGBA", (w, h), bg_rgb + (255,))

    # Helper to check if Markers should be drawn first
    draw_markers_first = len(active_layers) > 0 and active_layers[0] == "Markers"

    if draw_markers_first:
        # Markers is the bottom layer, initialize with masked markers
        if mask_geom is not None:
            markers_layer = _apply_outside_mask(
                preview_image,
                mask_geom,
                outside_mode=outside_mode,
                custom_color=outside_custom_color,
                dim_strength=outside_dim_strength,
            )
        else:
            markers_layer = preview_image.convert("RGBA")
        
        # Apply markers opacity if needed
        if markers_opacity < 1.0:
            alpha = markers_layer.getchannel("A")
            new_alpha = alpha.point(lambda p: int(p * markers_opacity))
            markers_layer.putalpha(new_alpha)
            
        rendered.alpha_composite(markers_layer)
        remaining_layers = active_layers[1:]
    else:
        remaining_layers = active_layers

    # Draw remaining layers in sequence (bottom to top)
    for layer in remaining_layers:
        if layer == "Markers":
            if mask_geom is not None:
                markers_layer = _apply_outside_mask(
                    preview_image,
                    mask_geom,
                    outside_mode=outside_mode,
                    custom_color=outside_custom_color,
                    dim_strength=outside_dim_strength,
                )
            else:
                markers_layer = preview_image.convert("RGBA")
            
            if markers_opacity < 1.0:
                alpha = markers_layer.getchannel("A")
                new_alpha = alpha.point(lambda p: int(p * markers_opacity))
                markers_layer.putalpha(new_alpha)
                
            rendered.alpha_composite(markers_layer)

        elif layer == "Tissue Annotations":
            if show_roi_outline and transformed_roi_geom_local is not None:
                rendered = _draw_overlay(
                    rendered,
                    transformed_roi_geom_local,
                    roi_outline_color,
                    "outline",
                    0.0,
                    max(overlay_line_width, 2),
                )

            if show_annotation_overlays:
                for record in selected_overlay_records:
                    geom = overlay_geometries_local.get(record["key"])
                    if geom is None or geom.is_empty:
                        continue
                    rendered = _draw_overlay(
                        rendered,
                        geom,
                        str(overlay_color_map.get(record["key"], record["color"]) or "#ffffff"),
                        overlay_style,
                        overlay_fill_alpha,
                        overlay_line_width,
                    )

        elif layer == "Cell Segmentations":
            if show_cells and cells_list:
                if st.session_state.get("tab9_actual_enable_cell_filter", False):
                    draw_cells = [c for c in cells_list if c.get("matched", False)]
                else:
                    draw_cells = cells_list
                rendered = _draw_cell_polygons(
                    image=rendered,
                    cells=draw_cells,
                    crop_box_native=crop_box_native,
                    crop_box_px=crop_box_px,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    color_mode=cell_color_mode,
                    single_color=cell_single_color,
                    phenotype_col=cell_phenotype_col,
                    phenotype_colors=cell_phenotype_colors,
                    continuous_col=cell_continuous_col,
                    continuous_cmap_name=cell_continuous_cmap,
                    continuous_limits=cell_continuous_limits,
                    fill_alpha=cell_fill_alpha,
                    line_width=cell_line_width,
                    outline_only=cell_outline_only,
                    compare_col1=cell_compare_col1,
                    compare_col2=cell_compare_col2,
                    compare_match_color=cell_compare_match_color,
                    compare_mismatch_color=cell_compare_mismatch_color,
                    outline_color_mode=cell_outline_color_mode,
                    outline_color=cell_outline_color,
                )

    return rendered, crop_box_px, mask_geom


def _feature_label(feature: dict[str, Any], default_index: int) -> str:
    props = feature.get("properties", {}) or {}
    classification = props.get("classification", {}) or {}
    for candidate in [props.get("label"), classification.get("name"), props.get("name"), props.get("type")]:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return f"feature_{default_index + 1}"


def _feature_color(feature: dict[str, Any], label: str) -> str:
    props = feature.get("properties", {}) or {}
    classification = props.get("classification", {}) or {}
    color = classification.get("color", props.get("color"))
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return _hex_from_rgb(color)
    if isinstance(color, str):
        return _hex_from_rgb(color)
    palette = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
    ]
    return palette[abs(hash(label)) % len(palette)]


def _load_geojson_groups(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as handle:
        data = json.load(handle)
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = []

    grouped: dict[str, dict[str, Any]] = {}
    for idx, feature in enumerate(features):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if geom.is_empty:
            continue
        label = _feature_label(feature, idx)
        key = f"{path.name}::{label}"
        entry = grouped.setdefault(
            key,
            {
                "key": key,
                "file": str(path),
                "file_name": path.name,
                "label": label,
                "color": _feature_color(feature, label),
                "geometries": [],
            },
        )
        entry["geometries"].append(geom)

    out: list[dict[str, Any]] = []
    for key, entry in grouped.items():
        geom = unary_union(entry["geometries"])
        if geom.is_empty:
            continue
        out.append(
            {
                "key": key,
                "file": entry["file"],
                "file_name": entry["file_name"],
                "label": entry["label"],
                "color": entry["color"],
                "geometry": geom,
            }
        )
    return sorted(out, key=lambda item: (item["file_name"], item["label"]))


def _roi_record_for_selection(records: list[dict[str, Any]], tissue_value: Any) -> dict[str, Any] | None:
    if not records:
        return None
    roi_token = _extract_roi_token(tissue_value)
    if roi_token is None:
        return records[0] if len(records) == 1 else None
    for record in records:
        label_digits = re.findall(r"\d+", str(record.get("label", "")))
        if roi_token in label_digits:
            return record
    if roi_token.isdigit():
        idx = int(roi_token) - 1
        if 0 <= idx < len(records):
            return records[idx]
    return records[0] if len(records) == 1 else None


def _geometry_to_polygons(geom) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [poly for poly in geom.geoms if isinstance(poly, Polygon) and not poly.is_empty]
    if hasattr(geom, "geoms"):
        polys: list[Polygon] = []
        for part in geom.geoms:
            polys.extend(_geometry_to_polygons(part))
        return polys
    return []


def _transform_geometry_to_crop(
    geom,
    scale_x: float,
    scale_y: float,
    crop_px: tuple[int, int, int, int] | None,
):
    if geom is None or geom.is_empty:
        return geom
    transformed = scale_geometry(geom, xfact=float(scale_x), yfact=float(scale_y), origin=(0, 0))
    if crop_px is not None:
        transformed = translate_geometry(transformed, xoff=-float(crop_px[0]), yoff=-float(crop_px[1]))
    return transformed


def _rasterize_geometry(size: tuple[int, int], geom) -> Image.Image:
    width, height = size
    mask = Image.new("L", (max(int(width), 1), max(int(height), 1)), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in _geometry_to_polygons(geom):
        exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for hole in polygon.interiors:
            interior = [(float(x), float(y)) for x, y in hole.coords]
            if len(interior) >= 3:
                draw.polygon(interior, fill=0)
    return mask


def _apply_outside_mask(
    image: Image.Image,
    geom,
    outside_mode: str,
    custom_color: str,
    dim_strength: float,
) -> Image.Image:
    if geom is None or geom.is_empty:
        return image
    base = image.convert("RGBA")
    mask = np.asarray(_rasterize_geometry(base.size, geom), dtype=float) / 255.0
    base_arr = np.asarray(base, dtype=float)
    if outside_mode == "transparent":
        result_rgb = base_arr[..., :3]
        result_alpha = base_arr[..., 3] * mask
        result = np.dstack([result_rgb, result_alpha])
    elif outside_mode == "black":
        outside_rgb = np.zeros(3, dtype=float)
        result_rgb = base_arr[..., :3] * mask[..., None]
        result = np.dstack([result_rgb, np.full(mask.shape, 255.0)])
    elif outside_mode == "custom":
        outside_rgb = np.asarray(_rgb_tuple(custom_color), dtype=float)
        result_rgb = base_arr[..., :3] * mask[..., None] + outside_rgb[None, None, :] * (1.0 - mask[..., None])
        result = np.dstack([result_rgb, np.full(mask.shape, 255.0)])
    else:
        dim = max(0.0, min(1.0, float(dim_strength)))
        result_rgb = base_arr[..., :3] * (mask[..., None] + (1.0 - mask[..., None]) * (1.0 - dim))
        result = np.dstack([result_rgb, np.full(mask.shape, 255.0)])
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGBA")


def _draw_overlay(
    image: Image.Image,
    geom,
    color_hex: str,
    style: str,
    fill_alpha: float,
    line_width: int,
) -> Image.Image:
    if geom is None or geom.is_empty:
        return image
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rgb = _rgb_tuple(color_hex)
    fill = None
    if style == "solid":
        fill = rgb + (255,)
    elif style == "transparent":
        fill = rgb + (max(0, min(255, int(fill_alpha * 255))),)
    outline = rgb + (255,)
    for polygon in _geometry_to_polygons(geom):
        exterior = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=fill, outline=outline if style != "solid" else outline, width=int(line_width))
        elif len(exterior) >= 2:
            draw.line(exterior, fill=outline, width=int(line_width))
        for hole in polygon.interiors:
            interior = [(float(x), float(y)) for x, y in hole.coords]
            if len(interior) >= 3:
                draw.polygon(interior, fill=(0, 0, 0, 0), outline=outline if style == "outline" else None, width=int(line_width))
    return Image.alpha_composite(base, overlay)


def _crop_box_from_cells(
    cells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    padding_px: float,
    full_hw: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    if cells_df.empty:
        return None
    x = pd.to_numeric(cells_df[x_col], errors="coerce")
    y = pd.to_numeric(cells_df[y_col], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return None
    min_x = max(float(np.nanmin(x[valid]) - padding_px), 0.0)
    max_x = min(float(np.nanmax(x[valid]) + padding_px), float(full_hw[1]))
    min_y = max(float(np.nanmin(y[valid]) - padding_px), 0.0)
    max_y = min(float(np.nanmax(y[valid]) + padding_px), float(full_hw[0]))
    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x, max_y


def _sample_cell_coords(cells_df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 1500) -> np.ndarray:
    if cells_df.empty:
        return np.empty((0, 2), dtype=float)
    coords = cells_df[[x_col, y_col]].to_numpy(dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) <= max_points:
        return coords
    indices = np.linspace(0, len(coords) - 1, max_points, dtype=int)
    return coords[indices]


def _estimate_cell_buffer_distance(coords: np.ndarray) -> float:
    if coords is None or len(coords) < 2:
        return 20.0
    sample = coords
    if len(sample) > 1200:
        indices = np.linspace(0, len(sample) - 1, 1200, dtype=int)
        sample = sample[indices]
    diff = sample[:, None, :] - sample[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 20.0
    estimate = float(np.percentile(nearest, 95))
    return float(np.clip(estimate * 1.2, 8.0, 120.0))


def _build_cell_buffer_geometry(cells_df: pd.DataFrame, x_col: str, y_col: str, padding_px: float):
    coords = _sample_cell_coords(cells_df, x_col, y_col, max_points=3000)
    if len(coords) == 0:
        return None
    buffer_distance = max(_estimate_cell_buffer_distance(coords), min(float(padding_px) * 0.25, 60.0))
    geom = MultiPoint([(float(x), float(y)) for x, y in coords]).buffer(buffer_distance)
    smooth = max(buffer_distance * 0.35, 2.0)
    try:
        geom = geom.buffer(smooth).buffer(-smooth * 0.5)
    except Exception:
        pass
    return geom if geom is not None and not geom.is_empty else None


def _select_overlap_annotation_record(
    cells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    candidate_records: list[dict[str, Any]],
    preferred_term: str = "tissue",
) -> dict[str, Any] | None:
    if not candidate_records or cells_df.empty:
        return None
    coords = _sample_cell_coords(cells_df, x_col, y_col, max_points=1000)
    if len(coords) == 0:
        return None
    sampled_points = [Point(float(x), float(y)) for x, y in coords]
    best_record = None
    best_score = -1.0
    for record in candidate_records:
        geom = record.get("geometry")
        if geom is None or geom.is_empty:
            continue
        count = 0
        try:
            for point in sampled_points:
                if geom.covers(point):
                    count += 1
        except Exception:
            continue
        label = str(record.get("label", "")).lower()
        bonus = 0.25 if preferred_term and preferred_term.lower() in label else 0.0
        score = float(count) + bonus
        if score > best_score:
            best_score = score
            best_record = record
    return best_record if best_score > 0 else None


def _resolve_roi_isolation_geometry(
    crop_mode: str,
    roi_only_source: str,
    roi_only_annotation_key: str,
    cells_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    padding_px: float,
    matched_roi_geom,
    selected_overlay_records: list[dict[str, Any]],
):
    if crop_mode != "Selected ROI only":
        return None, None
    overlay_by_key = {record["key"]: record for record in selected_overlay_records}
    explicit_annotation = None
    if roi_only_annotation_key and roi_only_annotation_key != "Auto from visible annotations":
        explicit_annotation = overlay_by_key.get(roi_only_annotation_key)

    if roi_only_source == "Matched ROI boundary":
        return matched_roi_geom, "Matched ROI boundary"
    if roi_only_source == "Cell buffer region":
        return _build_cell_buffer_geometry(cells_df, x_col, y_col, padding_px), "Cell buffer region"
    if roi_only_source == "Selected annotation region":
        if explicit_annotation is not None:
            return explicit_annotation.get("geometry"), explicit_annotation["key"]
        if len(selected_overlay_records) == 1:
            record = selected_overlay_records[0]
            return record.get("geometry"), record["key"]
        overlap_record = _select_overlap_annotation_record(
            cells_df=cells_df,
            x_col=x_col,
            y_col=y_col,
            candidate_records=selected_overlay_records,
            preferred_term="tissue",
        )
        return (overlap_record.get("geometry") if overlap_record else None), (overlap_record["key"] if overlap_record else None)

    if explicit_annotation is not None:
        return explicit_annotation.get("geometry"), explicit_annotation["key"]

    if len(selected_overlay_records) == 1:
        record = selected_overlay_records[0]
        return record.get("geometry"), record["key"]

    overlap_record = _select_overlap_annotation_record(
        cells_df=cells_df,
        x_col=x_col,
        y_col=y_col,
        candidate_records=selected_overlay_records,
        preferred_term="tissue",
    )
    if overlap_record is not None:
        return overlap_record.get("geometry"), overlap_record["key"]

    cell_geom = _build_cell_buffer_geometry(cells_df, x_col, y_col, padding_px)
    if cell_geom is not None and not cell_geom.is_empty:
        return cell_geom, "Cell buffer region"
    return matched_roi_geom, "Matched ROI boundary"


def _px_crop_box_from_full(
    crop_box: tuple[float, float, float, float] | None,
    scale_x: float,
    scale_y: float,
    max_w: int,
    max_h: int,
) -> tuple[int, int, int, int] | None:
    if crop_box is None:
        return None
    x0, y0, x1, y1 = crop_box
    left = min(max(int(math.floor(x0 * scale_x)), 0), max(max_w - 1, 0))
    top = min(max(int(math.floor(y0 * scale_y)), 0), max(max_h - 1, 0))
    right = min(max(int(math.ceil(x1 * scale_x)), left + 1), max_w)
    bottom = min(max(int(math.ceil(y1 * scale_y)), top + 1), max_h)
    return left, top, right, bottom


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_png_with_dpi(image: Image.Image, path: Path, dpi: int) -> None:
    image.save(path, format="PNG", dpi=(int(dpi), int(dpi)))


def _image_bytes_with_dpi(image: Image.Image, fmt: str, dpi: int) -> bytes:
    buffer = io.BytesIO()
    save_fmt = fmt.upper()
    if save_fmt == "PNG":
        image.save(buffer, format="PNG", dpi=(int(dpi), int(dpi)))
    else:
        image.save(buffer, format=save_fmt)
    return buffer.getvalue()


def _figure_bytes(image: Image.Image, fmt: str, dpi: int) -> bytes:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    figure = plt.figure(figsize=(max(width, 1) / float(dpi), max(height, 1) / float(dpi)), dpi=int(dpi))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(rgba)
    axis.axis("off")
    buffer = io.BytesIO()
    figure.savefig(buffer, format=fmt, dpi=int(dpi), bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return buffer.getvalue()


def _save_publication_figure(image: Image.Image, path: Path, fmt: str, dpi: int) -> None:
    if fmt.lower() == "png":
        _save_png_with_dpi(image, path, dpi)
        return
    data = _figure_bytes(image, fmt=fmt, dpi=dpi)
    path.write_bytes(data)


def _load_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", max(int(size), 10))
    except Exception:
        return ImageFont.load_default()


def _text_color(use_background: bool) -> tuple[int, int, int, int]:
    return (255, 255, 255, 255) if use_background else (0, 0, 0, 255)


def _nice_scale_bar_um(image_width_px: int, pixel_size_um: float) -> float:
    if pixel_size_um <= 0 or image_width_px <= 0:
        return 20.0
    max_um = image_width_px * 0.22 * pixel_size_um
    candidates = [value for value in SCALE_BAR_UM_CHOICES if value <= max_um]
    return candidates[-1] if candidates else min(SCALE_BAR_UM_CHOICES)


def _pick_best_anchor(image: Image.Image, box_size: tuple[int, int], margin: int = 16) -> tuple[int, int]:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba, dtype=float)
    height, width = arr.shape[:2]
    box_w = min(max(int(box_size[0]), 1), width)
    box_h = min(max(int(box_size[1]), 1), height)
    candidates = [
        (margin, margin),
        (max(width - box_w - margin, 0), margin),
        (margin, max(height - box_h - margin, 0)),
        (max(width - box_w - margin, 0), max(height - box_h - margin, 0)),
    ]
    best = candidates[0]
    best_score = -1e9
    for x0, y0 in candidates:
        x1 = min(x0 + box_w, width)
        y1 = min(y0 + box_h, height)
        region = arr[y0:y1, x0:x1, :3]
        if region.size == 0:
            continue
        brightness = float(np.mean(region)) / 255.0
        flatness = 1.0 - min(float(np.std(region)) / 128.0, 1.0)
        score = brightness + flatness
        if score > best_score:
            best_score = score
            best = (x0, y0)
    return best


def _make_legend_block(
    title: str,
    items: list[tuple[str, str]],
    image_size: tuple[int, int],
    use_background: bool,
    color_labels_with_item_color: bool = False,
) -> Image.Image | None:
    if not items:
        return None
    width, height = image_size
    padding = max(12, int(round(width * 0.012)))
    row_h = max(18, int(round(height * 0.035)))
    title_font = _load_font(max(int(round(width * 0.02)), 13))
    item_font = _load_font(max(int(round(width * 0.018)), 12))
    temp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp, "RGBA")
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    text_width = title_bbox[2] - title_bbox[0]
    for name, _color in items:
        bbox = draw.textbbox((0, 0), str(name), font=item_font)
        text_width = max(text_width, bbox[2] - bbox[0])
    block_w = int(round(padding * 3 + row_h + text_width))
    block_h = int(round(padding * 3 + row_h * len(items) + (title_bbox[3] - title_bbox[1])))
    block = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    block_draw = ImageDraw.Draw(block, "RGBA")
    if use_background or color_labels_with_item_color:
        bg_fill = (0, 0, 0, 255) if color_labels_with_item_color else (0, 0, 0, 120)
        block_draw.rounded_rectangle([0, 0, block_w, block_h], radius=max(8, padding // 2), fill=bg_fill)
    text_fill = _text_color(use_background or color_labels_with_item_color)
    block_draw.text((padding, padding * 0.8), title, fill=text_fill, font=title_font)
    y0 = padding * 1.8 + (title_bbox[3] - title_bbox[1])
    for idx, (name, color_hex) in enumerate(items):
        y = y0 + idx * row_h
        square = [padding, y + row_h * 0.2, padding + row_h * 0.6, y + row_h * 0.8]
        block_draw.rounded_rectangle(square, radius=4, fill=_rgb_tuple(color_hex) + (255,))
        item_text_fill = _rgb_tuple(color_hex) + (255,) if color_labels_with_item_color else text_fill
        block_draw.text((padding * 2 + row_h * 0.6, y + row_h * 0.05), str(name), fill=item_text_fill, font=item_font)
    return block


def _make_scale_bar_block(
    scale_bar_um: float,
    pixel_size_um: float,
    image_size: tuple[int, int],
    use_background: bool,
) -> Image.Image | None:
    if pixel_size_um <= 0:
        return None
    width, height = image_size
    padding = max(12, int(round(width * 0.012)))
    scale_font = _load_font(max(int(round(width * 0.016)), 11))
    temp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp, "RGBA")
    text = f"{scale_bar_um:g} µm"
    bbox = draw.textbbox((0, 0), text, font=scale_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    bar_px = max(int(round(scale_bar_um / pixel_size_um)), 1)
    bar_h = max(4, int(round(height * 0.006)))
    block_w = int(round(padding * 2 + max(bar_px, text_w)))
    block_h = int(round(padding * 3 + bar_h + text_h))
    block = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    block_draw = ImageDraw.Draw(block, "RGBA")
    if use_background:
        block_draw.rounded_rectangle([0, 0, block_w, block_h], radius=max(8, padding // 2), fill=(0, 0, 0, 120))
    text_fill = _text_color(use_background)
    bar_left = padding
    bar_top = padding
    block_draw.rectangle([bar_left, bar_top, bar_left + bar_px, bar_top + bar_h], fill=text_fill)
    block_draw.text((padding, bar_top + bar_h + padding), text, fill=text_fill, font=scale_font)
    return block


def _corner_anchor(placement: str, canvas_size: tuple[int, int], block_size: tuple[int, int], margin: int, offsets: dict[str, int]) -> tuple[int, int]:
    canvas_w, canvas_h = canvas_size
    block_w, block_h = block_size
    if placement == "top-left":
        x = margin
        y = margin + offsets.get(placement, 0)
        offsets[placement] = offsets.get(placement, 0) + block_h + margin
        return x, y
    if placement == "top-right":
        x = max(canvas_w - block_w - margin, 0)
        y = margin + offsets.get(placement, 0)
        offsets[placement] = offsets.get(placement, 0) + block_h + margin
        return x, y
    if placement == "bottom-left":
        offset = offsets.get(placement, 0)
        x = margin
        y = max(canvas_h - block_h - margin - offset, 0)
        offsets[placement] = offset + block_h + margin
        return x, y
    x = max(canvas_w - block_w - margin, 0)
    offset = offsets.get("bottom-right", 0)
    y = max(canvas_h - block_h - margin - offset, 0)
    offsets["bottom-right"] = offset + block_h + margin
    return x, y


def _draw_publication_adornments(
    image: Image.Image,
    marker_items: list[tuple[str, str]],
    annotation_items: list[tuple[str, str]],
    phenotype_items: list[tuple[str, str]],
    show_marker_legend: bool,
    show_annotation_legend: bool,
    show_phenotype_legend: bool,
    show_scale_bar: bool,
    marker_legend_position: str,
    annotation_legend_position: str,
    phenotype_legend_position: str,
    scale_bar_position: str,
    scale_bar_um: float,
    pixel_size_um: float,
    use_background: bool,
    marker_legend_colored_labels: bool = False,
) -> Image.Image:
    base = image.convert("RGBA")
    blocks: list[tuple[str, Image.Image]] = []
    if show_marker_legend:
        marker_block = _make_legend_block("Markers", marker_items, base.size, use_background, color_labels_with_item_color=marker_legend_colored_labels)
        if marker_block is not None:
            blocks.append((marker_legend_position, marker_block))
    if show_annotation_legend:
        annotation_block = _make_legend_block("Annotations", annotation_items, base.size, use_background)
        if annotation_block is not None:
            blocks.append((annotation_legend_position, annotation_block))
    if show_phenotype_legend:
        phenotype_block = _make_legend_block("Phenotypes", phenotype_items, base.size, use_background)
        if phenotype_block is not None:
            blocks.append((phenotype_legend_position, phenotype_block))
    if show_scale_bar:
        scale_block = _make_scale_bar_block(scale_bar_um, pixel_size_um, base.size, use_background)
        if scale_block is not None:
            blocks.append((scale_bar_position, scale_block))

    padding = max(12, int(round(base.width * 0.012)))
    below_blocks = [block for placement, block in blocks if placement == "below image"]
    corner_blocks = [(placement, block) for placement, block in blocks if placement != "below image"]

    if below_blocks:
        below_height = padding
        below_width = base.width
        for block in below_blocks:
            below_height += block.height + padding
        extended = Image.new("RGBA", (below_width, base.height + below_height), (255, 255, 255, 0))
        extended.alpha_composite(base, (0, 0))
        y_cursor = base.height + padding
        for block in below_blocks:
            x = max((below_width - block.width) // 2, 0)
            extended.alpha_composite(block, (x, y_cursor))
            y_cursor += block.height + padding
        base = extended

    offsets = {"top-left": 0, "top-right": 0, "bottom-left": 0, "bottom-right": 0}
    for placement, block in corner_blocks:
        if placement == "auto":
            x, y = _pick_best_anchor(base, block.size, margin=padding)
        else:
            x, y = _corner_anchor(placement, base.size, block.size, padding, offsets)
        base.alpha_composite(block, (int(x), int(y)))
    return base


def _crop_box_native_from_full(crop_box: tuple[float, float, float, float] | None, full_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    if crop_box is None:
        return (0, 0, int(full_hw[1]), int(full_hw[0]))
    x0, y0, x1, y1 = crop_box
    return (
        max(int(math.floor(x0)), 0),
        max(int(math.floor(y0)), 0),
        max(int(math.ceil(x1)), max(int(math.floor(x0)), 0) + 1),
        max(int(math.ceil(y1)), max(int(math.floor(y0)), 0) + 1),
    )


def _local_mask_geom_fullres(
    crop_mode: str,
    roi_isolation_geom,
    selected_overlay_records: list[dict[str, Any]],
    overlay_geometries: dict[str, Any],
    mask_source: str,
    crop_box_native: tuple[int, int, int, int],
):
    left, top, _right, _bottom = crop_box_native
    roi_local = None
    if roi_isolation_geom is not None:
        roi_local = translate_geometry(roi_isolation_geom, xoff=-float(left), yoff=-float(top))
    overlay_local: dict[str, Any] = {}
    for record in selected_overlay_records:
        geom = overlay_geometries.get(record["key"])
        overlay_local[record["key"]] = translate_geometry(geom, xoff=-float(left), yoff=-float(top)) if geom is not None else None

    mask_geom = None
    if crop_mode == "Selected ROI only" and roi_local is not None:
        mask_geom = roi_local
    selected_mask_geom = None
    mask_sources_list = []
    if isinstance(mask_source, str):
        if mask_source != "None" and mask_source:
            mask_sources_list = [mask_source]
    elif mask_source:
        mask_sources_list = [ms for ms in mask_source if ms != "None"]
        
    geoms_to_union = []
    for ms in mask_sources_list:
        if ms == "Matched tissue ROI" and roi_local is not None:
            geoms_to_union.append(roi_local)
        else:
            g = overlay_local.get(ms)
            if g is not None:
                geoms_to_union.append(g)
                
    if geoms_to_union:
        try:
            selected_mask_geom = unary_union(geoms_to_union)
        except Exception as e:
            _log_console(f"[ERROR] Failed to union mask geometries in fullres: {e}", level="error")

    if mask_geom is None:
        mask_geom = selected_mask_geom
    elif selected_mask_geom is not None:
        try:
            mask_geom = mask_geom.intersection(selected_mask_geom)
        except Exception:
            mask_geom = selected_mask_geom
    return mask_geom


def _prepare_native_layers(
    image_path: Path,
    channel_names: list[str],
    selected_markers: list[str],
    crop_box_full: tuple[float, float, float, float] | None,
    full_hw: tuple[int, int],
    crop_mode: str,
    roi_isolation_geom,
    selected_overlay_records: list[dict[str, Any]],
    overlay_geometries: dict[str, Any],
    mask_source: str,
) -> tuple[list[np.ndarray], tuple[int, int, int, int], Any]:
    valid_markers = [marker for marker in selected_markers if marker in channel_names]
    if not valid_markers and channel_names:
        valid_markers = [channel_names[0]]
    channel_indices = tuple(channel_names.index(marker) for marker in valid_markers)
    loaded, _ = _load_preview_planes(
        str(image_path),
        image_path.stat().st_mtime,
        channel_indices,
        None,
    )
    crop_box_native = _crop_box_native_from_full(crop_box_full, full_hw)
    left, top, right, bottom = crop_box_native
    mask_geom = _local_mask_geom_fullres(
        crop_mode=crop_mode,
        roi_isolation_geom=roi_isolation_geom,
        selected_overlay_records=selected_overlay_records,
        overlay_geometries=overlay_geometries,
        mask_source=mask_source,
        crop_box_native=crop_box_native,
    )
    mask_array = None
    if mask_geom is not None and not mask_geom.is_empty:
        mask_array = np.asarray(_rasterize_geometry((right - left, bottom - top), mask_geom), dtype=np.uint8) > 0

    layers: list[np.ndarray] = []
    for marker in valid_markers:
        idx = channel_names.index(marker)
        plane = np.asarray(loaded[idx])
        cropped = plane[top:bottom, left:right].copy()
        if mask_array is not None:
            cropped = np.where(mask_array, cropped, 0)
        if cropped.dtype != np.uint16:
            if np.issubdtype(cropped.dtype, np.floating):
                cropped = np.nan_to_num(cropped, nan=0.0)
            cropped = np.clip(cropped, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        layers.append(cropped)
    return layers, crop_box_native, mask_geom


def _save_plain_tiff_stack(layers: list[np.ndarray], output_path: Path) -> None:
    if not layers:
        raise ValueError("No layers provided for TIFF export.")
    stack = np.stack(layers, axis=0)
    tifffile.imwrite(str(output_path), stack, photometric="minisblack")


def _build_rgb16_composite(
    layers: list[np.ndarray],
    marker_colors: list[tuple[int, int, int]],
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    if not layers:
        raise ValueError("No layers provided for RGB composite export.")
    first = np.asarray(layers[0])
    composite = np.zeros((first.shape[0], first.shape[1], 3), dtype=float)
    for layer, color in zip(layers, marker_colors):
        plane = _normalize_plane(np.asarray(layer), low_percentile=low_percentile, high_percentile=high_percentile)
        rgb = np.asarray(color, dtype=float) / 255.0
        composite += plane[..., None] * rgb
    composite = np.clip(composite, 0.0, 1.0)
    return np.round(composite * float(np.iinfo(np.uint16).max)).astype(np.uint16)


def _save_rgb16_tiff(rgb_image: np.ndarray, output_path: Path) -> None:
    tifffile.imwrite(str(output_path), rgb_image, photometric="rgb")


def _build_output_stem(batch_value: Any, tissue_value: Any, selected_markers: list[str]) -> str:
    stem_parts = [_sanitize_filename_part(str(batch_value))]
    if str(tissue_value) != "All":
        stem_parts.append(f"roi-{_sanitize_filename_part(str(tissue_value))}")
    if selected_markers:
        marker_text = "-".join(_sanitize_filename_part(marker) for marker in selected_markers)
        stem_parts.append(f"markers-{marker_text}")
    stem_parts.append("actual-tissue")
    stem = "_".join(part for part in stem_parts if part)
    return stem[:220]


def _seed_text_default(session_key: str, default_value: str) -> None:
    current = st.session_state.get(session_key)
    if current is None or not str(current).strip():
        st.session_state[session_key] = str(default_value)


def _repair_directory_value(current_value: Any, default_value: str, project_root: Path) -> str:
    text = str(current_value or "").strip()
    default_path = Path(default_value)
    if not text:
        return str(default_path)
    current_path = Path(text)
    if current_path.exists() and current_path.is_dir():
        return str(current_path)
    root_name = project_root.name
    parts = list(current_path.parts)
    if root_name in parts:
        idx = parts.index(root_name)
        tail = parts[idx + 1 :]
        candidate = project_root.joinpath(*tail)
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
    if default_path.exists() and default_path.is_dir():
        return str(default_path)
    return text


def _seed_directory_default(session_key: str, default_value: str, project_root: Path) -> None:
    repaired = _repair_directory_value(st.session_state.get(session_key), default_value, project_root)
    st.session_state[session_key] = repaired


def _apply_custom_filters(df_to_filter: pd.DataFrame) -> pd.DataFrame:
    """Apply custom column filtration configurations to the given DataFrame."""
    if df_to_filter.empty:
        return df_to_filter
        
    if not st.session_state.get("tab9_actual_enable_cell_filter", False):
        return df_to_filter
        
    num_filters = st.session_state.get("tab9_actual_num_filters", 1)
    res_df = df_to_filter.copy()
    
    for i in range(int(num_filters)):
        col = st.session_state.get(f"tab9_actual_filter_col_{i}")
        if not col:
            continue
            
        if col not in res_df.columns:
            continue
            
        res_df = res_df.dropna(subset=[col])
        if res_df.empty:
            break
            
        filter_type = st.session_state.get(f"tab9_actual_filter_type_{i}", "Categorical")
        if filter_type == "Categorical":
            selected_vals = st.session_state.get(f"tab9_actual_filter_vals_{i}", [])
            if selected_vals:
                res_df = res_df[res_df[col].astype(str).isin(selected_vals)]
            else:
                res_df = res_df[res_df[col].astype(str).isin([])]
        elif filter_type == "Continuous":
            op = st.session_state.get(f"tab9_actual_filter_op_{i}", "Between")
            numeric_col = pd.to_numeric(res_df[col], errors="coerce")
            
            valid_mask = numeric_col.notna()
            res_df = res_df[valid_mask]
            numeric_col = numeric_col[valid_mask]
            if res_df.empty:
                break
                
            if op == "Between":
                val_range = st.session_state.get(f"tab9_actual_filter_range_{i}")
                if val_range:
                    res_df = res_df[(numeric_col >= val_range[0]) & (numeric_col <= val_range[1])]
            elif op == "Above (>=)":
                thresh = st.session_state.get(f"tab9_actual_filter_thresh_{i}")
                if thresh is not None:
                    res_df = res_df[numeric_col >= thresh]
            elif op == "Below (<=)":
                thresh = st.session_state.get(f"tab9_actual_filter_thresh_{i}")
                if thresh is not None:
                    res_df = res_df[numeric_col <= thresh]
            elif op == "Equal (==)":
                thresh = st.session_state.get(f"tab9_actual_filter_thresh_{i}")
                if thresh is not None:
                    res_df = res_df[np.isclose(numeric_col, thresh)]
                    
    return res_df


def render_actual_tissue(project_root: Path, df: pd.DataFrame, viz_settings: dict[str, Any]) -> None:
    # Streamlit Widget State Preserver
    # Copy active widget values to backup keys so switching tabs doesn't delete them,
    # and restore them if Streamlit's rendering cleanup has deleted the original key.
    for k in list(st.session_state.keys()):
        if k.startswith("tab9_actual_"):
            st.session_state[f"_backup_{k}"] = st.session_state[k]
            
    for k in list(st.session_state.keys()):
        if k.startswith("_backup_tab9_actual_"):
            orig_key = k[8:]
            if orig_key not in st.session_state:
                st.session_state[orig_key] = st.session_state[k]

    st.subheader("Actual Tissue")
    st.caption(
        "Create a real tissue image preview from the matched ROI TIFF, then crop, mask, and overlay GeoJSON annotations and cell segmentations."
    )

    if df.empty:
        st.warning("The active unified CSV is empty.")
        return

    # Extract global visualization settings
    image_dir = viz_settings.get("roi_dir", "")
    roi_geojson_dir = viz_settings.get("seg_dir", "")
    global_ann_dir = viz_settings.get("ann_dir", "")
    overlay_geojson_dir = global_ann_dir if global_ann_dir else st.session_state.get("tab9_actual_annotation_dir", "")
    
    # Check if directories exist
    if not image_dir or not os.path.exists(image_dir):
        st.error("ROI Images Directory is not configured or does not exist. Please set it in Step 9 Visual Validation Settings.")
        return
        
    cols = list(df.columns)
    column_options = _column_options(cols)
    x_col = _preferred_or_blank(column_options, "x_centroid")
    y_col = _preferred_or_blank(column_options, "y_centroid")
    batch_col_default = _preferred_or_blank(column_options, "batch_id")
    tissue_col_default = "tissue_roi" if "tissue_roi" in cols else "None"

    selected_batch_col = st.session_state.get("tab9_viz_col_batch", viz_settings.get("batch_col", batch_col_default))
    selected_x_col = st.session_state.get("tab9_viz_col_x", viz_settings.get("x_col", x_col))
    selected_y_col = st.session_state.get("tab9_viz_col_y", viz_settings.get("y_col", y_col))

    # Default output directory
    output_dir_default = str(project_root / "outputs" / "spatial_analysis" / "actual_tissue")
    _seed_directory_default("tab9_actual_output_dir", output_dir_default, project_root)
    
    st.session_state.setdefault("tab9_actual_show_marker_legend", True)
    st.session_state.setdefault("tab9_actual_show_annotation_legend", True)
    st.session_state.setdefault("tab9_actual_overlay_files", [])
    st.session_state.setdefault("tab9_actual_marker_legend_position", "below image")
    st.session_state.setdefault("tab9_actual_annotation_legend_position", "below image")
    st.session_state.setdefault("tab9_actual_scale_bar_position", "bottom-right")
    st.session_state.setdefault("tab9_actual_layer_1", "Markers")
    st.session_state.setdefault("tab9_actual_layer_2", "Tissue Annotations")
    st.session_state.setdefault("tab9_actual_layer_3", "Cell Segmentations")
    st.session_state.setdefault("tab9_actual_show_markers", True)
    st.session_state.setdefault("tab9_actual_background_color", "#ffffff")
    
    # Force opacity reset if 0.0 or not set
    if st.session_state.get("tab9_actual_markers_opacity", 1.0) == 0.0:
        st.session_state["tab9_actual_markers_opacity"] = 1.0
    elif "tab9_actual_markers_opacity" not in st.session_state:
        st.session_state["tab9_actual_markers_opacity"] = 1.0

    # SCAN IMAGES & SEGMENTATIONS FROM GLOBAL DIRS
    image_files = _list_files_cached(image_dir, IMAGE_EXTENSIONS, recursive=True)
    cell_seg_files = []
    if roi_geojson_dir and os.path.exists(roi_geojson_dir):
        cell_seg_files = _list_files_cached(roi_geojson_dir, GEOJSON_EXTENSIONS, recursive=True)
    image_labels = [
        str(Path(path).relative_to(image_dir)) if Path(path).is_relative_to(Path(image_dir)) else Path(path).name
        for path in image_files
    ]
    
    if not image_files:
        st.warning(f"No ROI images found in the global ROI Images Directory: `{image_dir}`")
        return

    # Get available batches from CSV
    batch_col_name = viz_settings.get("batch_col", "batch_id")
    if batch_col_name not in cols:
        batch_col_name = batch_col_default if batch_col_default else cols[0]
        
    # Resolve batch, image and tissue selection first so they are available for data loading
    selected_tissue_col = st.session_state.get("tab9_actual_tissue_col", tissue_col_default)
    
    available_batches = _safe_sorted(df[selected_batch_col].dropna().astype(str).unique().tolist()) if selected_batch_col else []
    
    # 1. Resolve batch_value
    current_batch_value = st.session_state.get("tab9_actual_batch_value")
    if (not current_batch_value or current_batch_value not in available_batches) and available_batches:
        current_batch_value = available_batches[0]
        st.session_state["tab9_actual_batch_value"] = current_batch_value
        
    # Check if batch changed compared to last run
    last_batch_key = "_tab9_actual_last_batch_value"
    last_image_key = "_tab9_actual_last_image_path"
    last_batch_value = st.session_state.get(last_batch_key, "")
    batch_changed = (last_batch_value != "" and last_batch_value != str(current_batch_value))
    
    # 2. Resolve tissue ROI values
    if selected_tissue_col != "None" and selected_batch_col and current_batch_value:
        tissue_series = df.loc[df[selected_batch_col].astype(str) == str(current_batch_value), selected_tissue_col]
        tissue_values = _safe_sorted(tissue_series.dropna().astype(str).unique().tolist())
    else:
        tissue_values = []
        
    # If batch changed, reset tissue to "All"
    if batch_changed:
        st.session_state["tab9_actual_tissue_value"] = "All"
        selected_tissue_value = "All"
    else:
        selected_tissue_value = st.session_state.get("tab9_actual_tissue_value", "All")
        if selected_tissue_value not in ["All"] + tissue_values:
            selected_tissue_value = "All"
            st.session_state["tab9_actual_tissue_value"] = "All"
            
    # 3. Resolve matched ROI image
    auto_image = _best_match_file(image_files, current_batch_value, selected_tissue_value) if image_files else None
    
    # 4. Resolve cell segmentation file
    auto_cell_seg = _best_match_file(cell_seg_files, _sample_prefix(str(current_batch_value)), None) if cell_seg_files else None
    
    # 5. Resolve annotations GeoJSON files
    overlay_geojson_files = []
    if overlay_geojson_dir and os.path.exists(overlay_geojson_dir):
        overlay_geojson_files = _list_files_cached(overlay_geojson_dir, GEOJSON_EXTENSIONS, recursive=True)
    batch_prefix = _sample_prefix(str(current_batch_value))
    auto_annotation_files = []
    if batch_prefix and overlay_geojson_files:
        auto_annotation_files = [
            f for f in overlay_geojson_files
            if batch_prefix.lower() in Path(f).name.lower()
        ]
        
    # Auto-matching updates if batch changed
    if batch_changed:
        if auto_image:
            st.session_state["tab9_actual_image_path"] = auto_image
        if auto_cell_seg:
            st.session_state["tab9_actual_cell_seg_file"] = auto_cell_seg
        st.session_state["tab9_actual_overlay_files"] = auto_annotation_files
    else:
        # Validate/ensure current values are correct
        current_image = st.session_state.get("tab9_actual_image_path")
        if (not current_image or current_image not in image_files) and image_files:
            st.session_state["tab9_actual_image_path"] = auto_image or image_files[0]
            
        current_seg = st.session_state.get("tab9_actual_cell_seg_file")
        if (not current_seg or current_seg not in cell_seg_files) and cell_seg_files:
            st.session_state["tab9_actual_cell_seg_file"] = auto_cell_seg or cell_seg_files[0]
            
        current_overlays = st.session_state.get("tab9_actual_overlay_files")
        if current_overlays is None:
            st.session_state["tab9_actual_overlay_files"] = auto_annotation_files
            
    # Set active variables
    batch_value = current_batch_value
    selected_image = st.session_state.get("tab9_actual_image_path", "")
    selected_seg_file = st.session_state.get("tab9_actual_cell_seg_file", "")
    selected_overlay_files = st.session_state.get("tab9_actual_overlay_files", [])
    
    # Store history for changes tracking
    st.session_state[last_batch_key] = str(batch_value)
    st.session_state[last_image_key] = str(selected_image) if selected_image else ""

    # Filter CSV for current batch/tissue
    csv_batch_df = df[df[selected_batch_col].astype(str) == str(batch_value)].copy() if selected_batch_col else df.copy()
    if selected_tissue_col != "None" and selected_tissue_value != "All":
        csv_batch_df = csv_batch_df[csv_batch_df[selected_tissue_col].astype(str) == str(selected_tissue_value)].copy()
    csv_batch_df = _apply_custom_filters(csv_batch_df)

    # Load Annotation overlays
    overlay_groups = []
    selected_overlay_files = st.session_state.get("tab9_actual_overlay_files", [])
    for geojson_path in selected_overlay_files:
        if os.path.exists(geojson_path):
            try:
                overlay_groups.extend(_load_geojson_groups(Path(geojson_path)))
            except Exception:
                pass
    visible_overlay_options = [record["key"] for record in overlay_groups]

    # Track changes in selected geojson files to auto-enable their overlays
    prev_overlay_files = st.session_state.get("_tab9_actual_prev_overlay_files", [])
    new_files = [f for f in selected_overlay_files if f not in prev_overlay_files]
    if new_files:
        new_keys = []
        for f in new_files:
            if os.path.exists(f):
                try:
                    groups = _load_geojson_groups(Path(f))
                    new_keys.extend([g["key"] for g in groups])
                except Exception:
                    pass
        if new_keys:
            current_visible = list(st.session_state.get("tab9_actual_visible_overlays", []))
            if "tab9_actual_visible_overlays" not in st.session_state:
                st.session_state["tab9_actual_visible_overlays"] = visible_overlay_options
            else:
                for nk in new_keys:
                    if nk not in current_visible:
                        current_visible.append(nk)
                st.session_state["tab9_actual_visible_overlays"] = current_visible
                
    st.session_state["_tab9_actual_prev_overlay_files"] = list(selected_overlay_files)

    # Load and match cell segmentations
    cells_list = None
    load_cells = st.session_state.get("tab9_actual_load_cells", False)
    cell_seg_source = st.session_state.get("tab9_actual_cell_seg_source", "GeoJSON boundaries file")
    match_dist = st.session_state.get("tab9_actual_cell_match_dist", 30)
    max_voronoi_dist = st.session_state.get("tab9_actual_voronoi_max_dist", 40)
    crop_voronoi_by_annotations = st.session_state.get("tab9_actual_voronoi_crop_annotations", True)

    # Build active annotations union for Voronoi cell cropping
    active_annotations_union = None
    if load_cells and cell_seg_source == "Voronoi (Virtual Segmentation)" and crop_voronoi_by_annotations:
        visible_overlays = st.session_state.get("tab9_actual_visible_overlays", visible_overlay_options)
        active_geoms = [r["geometry"] for r in overlay_groups if r["key"] in visible_overlays]
        if active_geoms:
            try:
                active_annotations_union = unary_union(active_geoms)
            except Exception:
                pass

    if load_cells:
        import time
        t_start = time.perf_counter()
        if cell_seg_source == "GeoJSON boundaries file":
            selected_seg_file = st.session_state.get("tab9_actual_cell_seg_file")
            if selected_seg_file and os.path.exists(selected_seg_file):
                try:
                    with st.spinner("Loading cell boundaries..."):
                        t0 = time.perf_counter()
                        raw_cells = _load_cell_segmentations_cached(selected_seg_file, Path(selected_seg_file).stat().st_mtime)
                        t_load = time.perf_counter() - t0
                    with st.spinner("Matching polygons to CSV cells..."):
                        t0 = time.perf_counter()
                        cells_list = _match_cells_to_csv(
                            cells=raw_cells,
                            csv_df=csv_batch_df,
                            x_col=selected_x_col,
                            y_col=selected_y_col,
                            max_dist=match_dist
                        )
                        t_match = time.perf_counter() - t0
                    n_matched = sum(1 for c in cells_list if c.get("matched", False)) if cells_list else 0
                    msg = f"Loaded {len(raw_cells):,} cell boundaries ({t_load:.2f}s), matched {n_matched:,} cells ({t_match:.2f}s)."
                    st.sidebar.info(msg)
                    _log_console(f"{msg}")
                except Exception as exc:
                    st.sidebar.error(f"Failed to load/match cell segmentations: {exc}")
                    _log_console(f"[ERROR] Cell loading/matching failed: {exc}", level="error")
        elif cell_seg_source == "Voronoi (Virtual Segmentation)":
            try:
                with st.spinner("Generating Voronoi cells..."):
                    t0 = time.perf_counter()
                    cells_list = _generate_voronoi_cells(
                        csv_df=csv_batch_df,
                        x_col=selected_x_col,
                        y_col=selected_y_col,
                        max_dist=max_voronoi_dist,
                        active_annotations_union=active_annotations_union
                    )
                    t_voronoi = time.perf_counter() - t0
                n_vor = len(cells_list) if cells_list else 0
                msg = f"Generated {n_vor:,} Voronoi cells in {t_voronoi:.2f}s."
                st.sidebar.info(msg)
                _log_console(f"{msg}")
            except Exception as exc:
                st.sidebar.error(f"Failed to generate Voronoi cells: {exc}")
                _log_console(f"[ERROR] Voronoi generation failed: {exc}", level="error")

    # Resolve Layer 1, Layer 2, Layer 3 visibilities and opacities dynamically
    show_markers = True
    markers_opacity = 1.0
    show_annotation_overlays = True
    overlay_fill_alpha = 1.0
    show_cells = True
    cell_fill_alpha = 1.0

    # Layer 1
    l1_type = st.session_state.get("tab9_actual_layer_1", "Markers")
    l1_show = st.session_state.get("tab9_actual_show_layer_1", True)
    l1_opacity = st.session_state.get("tab9_actual_layer_1_opacity", 1.0)
    if l1_type == "Markers":
        show_markers = l1_show
        markers_opacity = l1_opacity
    elif l1_type == "Tissue Annotations":
        show_annotation_overlays = l1_show
        overlay_fill_alpha = l1_opacity
    elif l1_type == "Cell Segmentations":
        show_cells = l1_show
        cell_fill_alpha = l1_opacity

    # Layer 2
    l2_type = st.session_state.get("tab9_actual_layer_2", "Tissue Annotations")
    l2_show = st.session_state.get("tab9_actual_show_layer_2", True)
    l2_opacity = st.session_state.get("tab9_actual_layer_2_opacity", 1.0)
    if l2_type == "Markers":
        show_markers = l2_show
        markers_opacity = l2_opacity
    elif l2_type == "Tissue Annotations":
        show_annotation_overlays = l2_show
        overlay_fill_alpha = l2_opacity
    elif l2_type == "Cell Segmentations":
        show_cells = l2_show
        cell_fill_alpha = l2_opacity

    # Layer 3
    l3_type = st.session_state.get("tab9_actual_layer_3", "Cell Segmentations")
    l3_show = st.session_state.get("tab9_actual_show_layer_3", True)
    l3_opacity = st.session_state.get("tab9_actual_layer_3_opacity", 1.0)
    if l3_type == "Markers":
        show_markers = l3_show
        markers_opacity = l3_opacity
    elif l3_type == "Tissue Annotations":
        show_annotation_overlays = l3_show
        overlay_fill_alpha = l3_opacity
    elif l3_type == "Cell Segmentations":
        show_cells = l3_show
        cell_fill_alpha = l3_opacity

    # Combine with load checkbox
    show_cells = show_cells and load_cells

    # Resolve cell segmentation styling variables
    cell_color_mode = st.session_state.get("tab9_actual_cell_color_mode", "Single Color")
    cell_single_color = st.session_state.get("tab9_actual_cell_single_color", "#ffffff")
    cell_phenotype_col = st.session_state.get("tab9_actual_cell_phenotype_col", "None")
    if cell_phenotype_col == "None":
        cell_phenotype_col = None
        
    # Rebuild cell phenotype colors
    cell_phenotype_colors = {}
    default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]
    if cell_phenotype_col:
        unique_phenos = sorted(csv_batch_df[cell_phenotype_col].dropna().astype(str).unique().tolist()) if cell_phenotype_col in csv_batch_df.columns else []
        for idx, pheno in enumerate(unique_phenos):
            widget_key = f"tab9_actual_cell_color_val_{_sanitize_key(pheno)}"
            stored_color = st.session_state.get(widget_key)
            if not stored_color or not str(stored_color).startswith("#") or len(stored_color) < 4:
                default_color = default_palette[idx % len(default_palette)]
                st.session_state[widget_key] = default_color
                cell_phenotype_colors[pheno] = default_color
            else:
                cell_phenotype_colors[pheno] = stored_color
            
    cell_continuous_col = st.session_state.get("tab9_actual_cell_continuous_col", "None")
    if cell_continuous_col == "None":
        cell_continuous_col = None
    cell_continuous_cmap = st.session_state.get("tab9_actual_cell_continuous_cmap", "viridis")
    
    # Continuous limits
    lim_mode = st.session_state.get("tab9_actual_cell_lim_mode", "Auto (Min/Max in CSV)")
    if lim_mode == "Auto (Min/Max in CSV)" and cell_continuous_col and not csv_batch_df.empty:
        vals = pd.to_numeric(csv_batch_df[cell_continuous_col], errors="coerce").dropna()
        min_val = float(vals.min()) if not vals.empty else 0.0
        max_val = float(vals.max()) if not vals.empty else 1.0
        cell_continuous_limits = (min_val, max_val)
    else:
        l_min = float(st.session_state.get("tab9_actual_cell_lim_min", 0.0))
        l_max = float(st.session_state.get("tab9_actual_cell_lim_max", 1.0))
        cell_continuous_limits = (l_min, l_max)
        
    cell_compare_col1 = st.session_state.get("tab9_actual_cell_compare_col1", "None")
    if cell_compare_col1 == "None":
        cell_compare_col1 = None
    cell_compare_col2 = st.session_state.get("tab9_actual_cell_compare_col2", "None")
    if cell_compare_col2 == "None":
        cell_compare_col2 = None
        
    cell_compare_match_color = st.session_state.get("tab9_actual_cell_compare_match_color", "#00ff00")
    cell_compare_mismatch_color = st.session_state.get("tab9_actual_cell_compare_mismatch_color", "#ff0000")
    
    cell_line_width = int(st.session_state.get("tab9_actual_cell_line_width", 1))
    cell_outline_only = bool(st.session_state.get("tab9_actual_cell_outline_only", False))
    cell_outline_color_mode = st.session_state.get("tab9_actual_cell_outline_color_mode", "Match Cell Color")
    cell_outline_color = st.session_state.get("tab9_actual_cell_outline_color", "#ffffff")
    show_cell_legend = bool(st.session_state.get("tab9_actual_show_cell_legend", True))

    # Resolve annotation styling and masking variables
    overlay_style = st.session_state.get("tab9_actual_overlay_style", "transparent")
    overlay_line_width = int(st.session_state.get("tab9_actual_overlay_line_width", 2))
    
    mask_source = st.session_state.get("tab9_actual_mask_source", [])
    outside_mode = st.session_state.get("tab9_actual_outside_mode", "dim")
    outside_dim_strength = float(st.session_state.get("tab9_actual_dim_strength", 0.8))
    outside_custom_color = st.session_state.get("tab9_actual_outside_color", "#000000")

    # REORGANIZED 2-COLUMN LAYOUT
    col_controls, col_preview = st.columns([1, 1.2])
    
    with col_controls:
        control_tabs = st.tabs([
            "Data & ROI",
            "Markers",
            "Cell Segmentation",
            "Annotations & Masking",
            "Output & Legend"
        ])
        
        # 1. DATA & ROI TAB
        with control_tabs[0]:
            st.markdown("### Data & ROI Selection")
            
            st.markdown("##### Column Mappings")
            st.info(f"**Batch Column:** `{selected_batch_col}` | **X Column:** `{selected_x_col}` | **Y Column:** `{selected_y_col}`\n\n*(Change these in Visual Validation Settings above)*")
            
            tissue_options = ["None"] + cols
            selected_tissue_col = st.selectbox(
                "Tissue ROI Column",
                options=tissue_options,
                index=_get_widget_index("tab9_actual_tissue_col", tissue_options, tissue_options.index(tissue_col_default) if tissue_col_default in tissue_options else 0),
                key="tab9_actual_tissue_col",
            )
            
            st.divider()
            st.markdown("##### Batch, Image & ROI Selection")
            
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                if available_batches:
                    batch_value = st.selectbox(
                        "Batch / Image Selection",
                        options=available_batches,
                        index=_get_widget_index("tab9_actual_batch_value", available_batches, 0),
                        key="tab9_actual_batch_value",
                    )
                else:
                    st.text_input("Batch / Image Selection", value="", disabled=True)
                    batch_value = ""
            
            with c_sel2:
                if selected_tissue_col != "None" and selected_batch_col and batch_value:
                    selected_tissue_value = st.selectbox(
                        "Tissue ROI",
                        options=["All"] + tissue_values,
                        index=_get_widget_index("tab9_actual_tissue_value", ["All"] + tissue_values, 0),
                        key="tab9_actual_tissue_value",
                    )
                else:
                    st.text_input("Tissue ROI", value="N/A", disabled=True)
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                if image_files:
                    selected_image = st.selectbox(
                        "Matched ROI Image",
                        options=image_files,
                        index=_get_widget_index("tab9_actual_image_path", image_files, image_files.index(selected_image) if selected_image in image_files else 0),
                        key="tab9_actual_image_path",
                        format_func=lambda path: image_labels[image_files.index(path)] if path in image_files else path,
                        help="Auto-matched from the selected batch / ROI name. Override manually if needed.",
                    )
                else:
                    st.text_input("Matched ROI Image", value="", disabled=True)
                    selected_image = ""
            
            with c_res2:
                if roi_geojson_dir and os.path.exists(roi_geojson_dir):
                    if not cell_seg_files:
                        st.text_input("Segmentation File (Cell boundaries)", value="No files found", disabled=True)
                    else:
                        st.selectbox(
                            "Segmentation File (Cell boundaries)",
                            options=cell_seg_files,
                            index=_get_widget_index("tab9_actual_cell_seg_file", cell_seg_files, cell_seg_files.index(selected_seg_file) if selected_seg_file in cell_seg_files else 0),
                            format_func=lambda path: Path(path).name,
                            key="tab9_actual_cell_seg_file",
                            help="Matched based on the active batch."
                        )
                else:
                    st.text_input("Segmentation File (Cell boundaries)", value="No directory configured", disabled=True)
            
            st.divider()
            st.markdown("##### Annotations Resources")
            
            if global_ann_dir and os.path.exists(global_ann_dir):
                overlay_geojson_dir = global_ann_dir
                st.caption(f"**Annotation Directory:** `{overlay_geojson_dir}` (Inherited from global configuration)")
            else:
                overlay_geojson_dir = directory_input_with_browse(
                    label="Annotation Source Directory",
                    text_key="tab9_actual_annotation_dir",
                    default_value=st.session_state.get("tab9_actual_annotation_dir", str(project_root)),
                    browser_key_prefix="tree_tab9_actual_annotation_dir",
                    start_dir_fallback=str(project_root),
                    help_text="Directory containing tissue annotation GeoJSON files.",
                )
            
            if overlay_geojson_dir and os.path.exists(overlay_geojson_dir):
                overlay_geojson_files = _list_files_cached(overlay_geojson_dir, GEOJSON_EXTENSIONS, recursive=True)
                overlay_labels = [
                    str(Path(path).relative_to(overlay_geojson_dir)) if Path(path).is_relative_to(Path(overlay_geojson_dir)) else Path(path).name
                    for path in overlay_geojson_files
                ]
                st.multiselect(
                    "Annotation GeoJSON Files",
                    options=overlay_geojson_files,
                    default=[x for x in st.session_state.get("tab9_actual_overlay_files", []) if x in overlay_geojson_files],
                    key="tab9_actual_overlay_files",
                    format_func=lambda path: overlay_labels[overlay_geojson_files.index(path)] if path in overlay_geojson_files else path,
                    help="Select one or more annotation GeoJSON files manually.",
                )
            
            st.divider()
            st.markdown("##### Cell Filtration")
            
            enable_filter = st.checkbox(
                "Filter cells by CSV column values",
                value=st.session_state.get("tab9_actual_enable_cell_filter", False),
                key="tab9_actual_enable_cell_filter",
                help="Apply custom filters on the loaded cell table before rendering or computing statistics."
            )
            
            if enable_filter:
                num_filters = st.number_input(
                    "Number of Filters",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.get("tab9_actual_num_filters", 1),
                    step=1,
                    key="tab9_actual_num_filters"
                )
                
                for i in range(int(num_filters)):
                    st.markdown(f"**Filter {i+1}**")
                    f_col1, f_col2 = st.columns(2)
                    
                    filter_col = f_col1.selectbox(
                        "Column",
                        options=[""] + cols,
                        key=f"tab9_actual_filter_col_{i}",
                    )
                    
                    if filter_col:
                        col_series = df[filter_col].dropna()
                        is_numeric = pd.api.types.is_numeric_dtype(col_series)
                        unique_vals = col_series.unique()
                        is_cat = not is_numeric or len(unique_vals) <= 15
                        
                        filter_type = f_col2.selectbox(
                            "Type",
                            options=["Categorical", "Continuous"],
                            index=0 if is_cat else 1,
                            key=f"tab9_actual_filter_type_{i}",
                        )
                        
                        if filter_type == "Categorical":
                            sorted_unique_vals = sorted([str(x) for x in unique_vals])
                            st.multiselect(
                                "Allowed Values",
                                options=sorted_unique_vals,
                                key=f"tab9_actual_filter_vals_{i}",
                                help="Only show cells matching these values."
                            )
                        else:
                            op = st.selectbox(
                                "Condition",
                                options=["Between", "Above (>=)", "Below (<=)", "Equal (==)"],
                                key=f"tab9_actual_filter_op_{i}",
                            )
                            val_min = float(col_series.min()) if not col_series.empty else 0.0
                            val_max = float(col_series.max()) if not col_series.empty else 100.0
                            
                            if op == "Between":
                                st.slider(
                                    "Range",
                                    min_value=val_min,
                                    max_value=val_max,
                                    value=(val_min, val_max),
                                    key=f"tab9_actual_filter_range_{i}",
                                )
                            else:
                                st.number_input(
                                    "Threshold",
                                    min_value=val_min,
                                    max_value=val_max,
                                    value=val_min,
                                    key=f"tab9_actual_filter_thresh_{i}",
                                    format="%.4f",
                                )
            
            st.divider()
            st.markdown("##### Preview & Crop Options")
            preview_max_dim = int(
                st.slider(
                    "Preview Max Dimension (px)",
                    min_value=600,
                    max_value=6000,
                    value=1400,
                    step=100,
                    key="tab9_actual_preview_max_dim",
                )
            )
            crop_to_cells = st.checkbox(
                "Crop to selected cells",
                value=True,
                key="tab9_actual_crop_to_cells",
                help="Crop the image to the bounding box of the selected batch / tissue ROI cells.",
            )
            
            c_p1, c_p2 = st.columns(2)
            crop_padding = float(
                c_p1.number_input(
                    "Crop Padding (px)",
                    min_value=0.0,
                    value=150.0,
                    step=25.0,
                    key="tab9_actual_crop_padding",
                )
            )
            pixel_size_um = float(
                c_p2.number_input(
                    "Pixel Size (µm)",
                    min_value=0.001,
                    value=0.5083,
                    step=0.0001,
                    format="%.4f",
                    key="tab9_actual_pixel_size_um",
                )
            )
            
            crop_mode = "Full image"
            if crop_to_cells:
                crop_mode = st.selectbox(
                    "Crop Mode",
                    options=["Rectangular context", "Selected ROI only"],
                    index=0,
                    key="tab9_actual_crop_mode",
                    help="Rectangular context keeps the current bounding box crop. Selected ROI only masks the crop to a resolved tissue geometry so neighbouring ROI tissue is hidden.",
                )
                
            roi_only_source = "Auto"
            if crop_mode == "Selected ROI only":
                roi_only_source = st.selectbox(
                    "Selected ROI Isolation",
                    options=["Auto", "Selected annotation region", "Cell buffer region", "Matched ROI boundary"],
                    index=0,
                    key="tab9_actual_roi_only_source",
                    help="Auto prefers the visible annotation region you selected, then other overlapping tissue-like annotations, then a cell-derived tissue mask, then the ROI boundary box.",
                )
                
            st.caption(f"**ROI Images Source Directory:** `{image_dir}`")
            
            st.divider()
            st.markdown("##### Portability: Settings Export/Import")
            
            # Export Settings
            export_settings = {}
            for k in list(st.session_state.keys()):
                if k.startswith("tab9_actual_") and not k.endswith("_import") and not k.endswith("_uploader"):
                    val = st.session_state[k]
                    if isinstance(val, (str, int, float, bool, list, dict)) or val is None:
                        export_settings[k] = val
            
            try:
                export_json = json.dumps(export_settings, indent=2)
            except Exception as e:
                export_json = "{}"
                
            st.download_button(
                label="Export Actual Tissue Settings (JSON)",
                data=export_json,
                file_name="actual_tissue_settings.json",
                mime="application/json",
                help="Export all visualization configurations for the Actual Tissue tab to a standalone JSON file."
            )
            
            # Import Settings
            uploaded_file = st.file_uploader(
                "Import Actual Tissue Settings",
                type=["json"],
                key="tab9_actual_settings_import",
                help="Select a JSON file containing exported 'Actual Tissue' settings to restore them."
            )
            if uploaded_file is not None:
                if st.button("Apply Settings", type="primary", key="tab9_actual_apply_imported_settings_btn"):
                    try:
                        imported_data = json.load(uploaded_file)
                        if isinstance(imported_data, dict):
                            for k, v in imported_data.items():
                                if k.startswith("tab9_actual_"):
                                    st.session_state[k] = v
                            
                            # Force a project config save if project is active
                            if "active_project" in st.session_state and st.session_state["active_project"]:
                                from spacemap_gui.persistence import sync_project_state, save_active_project_state
                                sync_project_state()
                                save_active_project_state()
                                
                            st.success("Settings applied successfully! Reloading...")
                            st.rerun()
                        else:
                            st.error("Invalid settings file format (must be a JSON dictionary).")
                    except Exception as e:
                        st.error(f"Error applying settings: {e}")
            
            
        # 2. MARKERS TAB
        with control_tabs[1]:
            st.markdown("### Markers & Intensity Scaling")
            
            if not selected_image:
                st.warning("Select an image in the Data & ROI tab to view available markers.")
            else:
                image_path = Path(selected_image)
                channel_names, channel_colors, _channel_meta, _mpp = _get_channel_info_cached(
                    str(image_path),
                    image_path.stat().st_mtime,
                )
                channel_names = [str(name) for name in channel_names]
                
                if not channel_names:
                    st.error("Unable to extract channels from the selected image.")
                else:
                    saved_markers = st.session_state.get("tab9_actual_markers")
                    if saved_markers is not None:
                        default_markers = [m for m in saved_markers if m in channel_names]
                    else:
                        default_markers = _default_marker_selection(channel_names)
                    selected_markers = [
                        str(marker)
                        for marker in st.multiselect(
                            "Markers / Channels",
                            options=channel_names,
                            default=default_markers,
                            key="tab9_actual_markers",
                            help="Multiple channels are rendered additively so marker signals overlap instead of covering one another.",
                        )
                    ]
                    
                    st.divider()
                    st.markdown("##### Marker Legend Styling")
                    st.checkbox(
                        "Color Marker Legend labels with Marker colors on black background",
                        value=st.session_state.get("tab9_actual_marker_legend_colored_labels", False),
                        key="tab9_actual_marker_legend_colored_labels",
                        help="When checked, Marker Legend text will be rendered in the marker's own color on a black background."
                    )
                    
                    st.divider()
                    st.markdown("##### Marker Colors")
                    marker_color_map = {
                        name: _hex_from_rgb(channel_colors[idx]) if idx < len(channel_colors) else "#ffffff"
                        for idx, name in enumerate(channel_names)
                    }
                    if selected_markers:
                        marker_columns = st.columns(min(3, len(selected_markers)))
                        for idx, marker in enumerate(selected_markers):
                            widget_key = f"tab9_actual_marker_color_{_sanitize_key(marker)}"
                            stored_color = st.session_state.get(widget_key)
                            if not stored_color or not str(stored_color).startswith("#") or len(stored_color) < 4:
                                default_color = marker_color_map.get(marker, "#ffffff")
                                st.session_state[widget_key] = default_color
                            else:
                                default_color = stored_color
                            with marker_columns[idx % len(marker_columns)]:
                                marker_color_map[marker] = st.color_picker(marker, value=default_color, key=widget_key)
                    else:
                        st.info("Select at least one marker. The preview will fall back to the first channel if needed.")
                        
                    st.divider()
                    st.markdown("##### Marker Intensity Scaling")
                    intensity_col1, intensity_col2 = st.columns(2)
                    marker_low_percentile = float(
                        intensity_col1.number_input(
                            "Lower Percentile Clip",
                            min_value=0.0,
                            max_value=99.9,
                            value=float(st.session_state.get("tab9_actual_marker_low_percentile", 1.0)),
                            step=0.1,
                            format="%.1f",
                            key="tab9_actual_marker_low_percentile",
                            help="Pixels below this percentile are clipped before display scaling. Raise it to suppress dim background outliers.",
                        )
                    )
                    marker_high_percentile = float(
                        intensity_col2.number_input(
                            "Upper Percentile Clip",
                            min_value=min(marker_low_percentile + 0.1, 100.0),
                            max_value=100.0,
                            value=float(st.session_state.get("tab9_actual_marker_high_percentile", 99.9)),
                            step=0.1,
                            format="%.1f",
                            key="tab9_actual_marker_high_percentile",
                            help="Pixels above this percentile are clipped before display scaling. Lower it to suppress very bright outliers.",
                        )
                    )
                    marker_high_percentile = max(marker_high_percentile, marker_low_percentile + 0.1)
                    st.caption("Preview and publication composites are rendered as 8-bit RGB images. Native OME-TIFF and TIFF stack exports keep 16-bit marker layers.")

        # 3. CELL SEGMENTATION TAB (NEW)
        with control_tabs[2]:
            st.markdown("### Cell Segmentation Polygons")
            st.caption(
                "Overlay cell boundaries (segmentation polygons) on top of the tissue image, and color them by variables (e.g. phenotypes) or continuous values."
            )
            
            load_cells = st.checkbox(
                "Load Cell Segmentation Polygons",
                value=st.session_state.get("tab9_actual_load_cells", False),
                key="tab9_actual_load_cells",
                help="Scan the segmentation directory and overlay cell polygons on top of the image."
            )
            
            if load_cells:
                cell_seg_source_options = ["GeoJSON boundaries file", "Voronoi (Virtual Segmentation)"]
                cell_seg_source = st.selectbox(
                    "Segmentation Source",
                    options=cell_seg_source_options,
                    index=_get_widget_index("tab9_actual_cell_seg_source", cell_seg_source_options, 0),
                    key="tab9_actual_cell_seg_source",
                    help="Choose whether to load cell segmentation boundaries from a GeoJSON file or generate virtual Voronoi cells from coordinates."
                )
                
                if cell_seg_source == "GeoJSON boundaries file":
                    if not roi_geojson_dir or not os.path.exists(roi_geojson_dir):
                        st.error("Segmentation Directory is not configured. Set it in global Visual Validation Settings.")
                    elif not cell_seg_files:
                        st.warning(f"No cell segmentation GeoJSON files found in `{roi_geojson_dir}`.")
                    else:
                        st.slider(
                            "Coordinate Match Max Distance (px)",
                            min_value=5,
                            max_value=150,
                            value=int(st.session_state.get("tab9_actual_cell_match_dist", 30)),
                            step=5,
                            key="tab9_actual_cell_match_dist",
                            help="Maximum pixel distance allowed to match a polygon to a CSV cell row."
                        )
                        if cells_list:
                            n_matched = sum(1 for c in cells_list if c.get("matched", False))
                            st.success(f"Loaded {len(cells_list):,} cell polygons. Matched {n_matched:,} cells to CSV.")
                        else:
                            st.info("No segmentation file loaded or matched.")
                            
                elif cell_seg_source == "Voronoi (Virtual Segmentation)":
                    coords = csv_batch_df[[selected_x_col, selected_y_col]].dropna().to_numpy()
                    if len(coords) >= 2:
                        import numpy as np
                        from scipy import spatial
                        tree = spatial.cKDTree(coords)
                        dists, _ = tree.query(coords, k=2)
                        nearest_dists = dists[:, 1]
                        default_max_dist = float(np.percentile(nearest_dists, 99.0))
                    else:
                        default_max_dist = 50.0
                        
                    st.slider(
                        "Voronoi Max Boundary Distance (px)",
                        min_value=5.0,
                        max_value=max(300.0, default_max_dist * 2),
                        value=float(st.session_state.get("tab9_actual_voronoi_max_dist", default_max_dist)),
                        step=1.0,
                        key="tab9_actual_voronoi_max_dist",
                        help="Maximum distance allowed from a cell centroid. Automatically set to the 99th percentile of nearest cell-cell distances."
                    )
                    
                    st.checkbox(
                        "Crop Voronoi cells by loaded annotations",
                        value=st.session_state.get("tab9_actual_voronoi_crop_annotations", True),
                        key="tab9_actual_voronoi_crop_annotations",
                        help="Clip Voronoi cells to lie strictly within the selected tissue annotation boundaries."
                    )
                    
                    if cells_list:
                        st.success(f"Generated {len(cells_list):,} Voronoi cells.")
                
                if cells_list is not None:
                    st.divider()
                    st.markdown("##### Coloring & Styling")
                    
                    cell_color_mode_options = ["Single Color", "Categorical / Phenotype Column", "Continuous Column", "Column Comparison"]
                    cell_color_mode = st.selectbox(
                        "Coloring Mode",
                        options=cell_color_mode_options,
                        index=_get_widget_index("tab9_actual_cell_color_mode", cell_color_mode_options, 0),
                        key="tab9_actual_cell_color_mode"
                    )
                    
                    if cell_color_mode == "Single Color":
                        cell_single_color = st.color_picker(
                            "Cell Fill/Outline Color",
                            value=st.session_state.get("tab9_actual_cell_single_color", "#ffffff"),
                            key="tab9_actual_cell_single_color"
                        )
                    elif cell_color_mode == "Categorical / Phenotype Column":
                        pheno_cols = [c for c in cols if "phenotype" in c.lower() or "leiden" in c.lower()] + [c for c in cols if c not in ["x_centroid", "y_centroid", "cell_id", "batch_id"]]
                        default_pheno = pheno_cols[0] if pheno_cols else cols[0]
                        
                        cell_phenotype_col = st.selectbox(
                            "Phenotype Column",
                            options=pheno_cols,
                            index=_get_widget_index("tab9_actual_cell_phenotype_col", pheno_cols, pheno_cols.index(default_pheno) if default_pheno in pheno_cols else 0),
                            key="tab9_actual_cell_phenotype_col"
                        )
                        
                        if cell_phenotype_col:
                            unique_phenos = sorted(csv_batch_df[cell_phenotype_col].dropna().astype(str).unique().tolist())
                            n_phenos = len(unique_phenos)
                            
                            palette_options = ["tab10", "tab20", "Set1", "Set2", "Set3", "Dark2", "Paired", "rainbow", "Spectral", "Custom (Individual Only)"]
                            palette_choice = st.selectbox(
                                "Palette Schematic",
                                options=palette_options,
                                index=_get_widget_index("tab9_actual_cell_phenotype_palette", palette_options, 0),
                                key="tab9_actual_cell_phenotype_palette",
                                help="Choose a pre-defined palette. To customize individual colors manually, select 'Custom (Individual Only)'."
                            )
                            
                            import matplotlib.pyplot as plt
                            import matplotlib.colors as mcolors
                            import numpy as np
                            
                            generated_colors = {}
                            if palette_choice != "Custom (Individual Only)":
                                cmap = plt.get_cmap(palette_choice)
                                if hasattr(cmap, "colors") and len(cmap.colors) >= n_phenos:
                                    for idx, pheno in enumerate(unique_phenos):
                                        generated_colors[pheno] = mcolors.to_hex(cmap.colors[idx])
                                else:
                                    for idx, pheno in enumerate(unique_phenos):
                                        val = idx / max(1, n_phenos - 1)
                                        generated_colors[pheno] = mcolors.to_hex(cmap(val))
                                        
                            with st.expander("Customize Phenotype Colors", expanded=True):
                                default_palette = [
                                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                                    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                                    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
                                ]
                                for idx, pheno in enumerate(unique_phenos):
                                    widget_key = f"tab9_actual_cell_color_val_{_sanitize_key(pheno)}"
                                    if palette_choice != "Custom (Individual Only)" and pheno in generated_colors:
                                        st.session_state[widget_key] = generated_colors[pheno]
                                        
                                    default_color = default_palette[idx % len(default_palette)]
                                    if widget_key not in st.session_state or st.session_state[widget_key] is None or not str(st.session_state[widget_key]).startswith("#") or len(st.session_state[widget_key]) < 4:
                                        st.session_state[widget_key] = default_color
                                        
                                    cell_phenotype_colors[pheno] = st.color_picker(
                                        f"Color: {pheno}",
                                        key=widget_key
                                    )
                                    
                            show_cell_legend = st.checkbox(
                                "Show cell legend in publication figure",
                                value=st.session_state.get("tab9_actual_show_cell_legend", True),
                                key="tab9_actual_show_cell_legend"
                            )
                            
                    elif cell_color_mode == "Continuous Column":
                        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in ["x_centroid", "y_centroid", "cell_id"]]
                        cell_continuous_col = st.selectbox(
                            "Continuous Column",
                            options=numeric_cols,
                            index=_get_widget_index("tab9_actual_cell_continuous_col", numeric_cols, 0),
                            key="tab9_actual_cell_continuous_col"
                        )
                        
                        cmap_options = ["viridis", "plasma", "inferno", "magma", "cividis", "bwr", "seismic", "coolwarm", "RdBu", "jet", "rainbow", "Spectral", "YlGnBu", "Reds", "Greens", "Blues", "Oranges", "YlOrRd"]
                        cell_continuous_cmap = st.selectbox(
                            "Colormap",
                            options=cmap_options,
                            index=_get_widget_index("tab9_actual_cell_continuous_cmap", cmap_options, cmap_options.index("viridis")),
                            key="tab9_actual_cell_continuous_cmap"
                        )
                        
                        if cell_continuous_col and not csv_batch_df.empty:
                            vals = pd.to_numeric(csv_batch_df[cell_continuous_col], errors="coerce").dropna()
                            min_val = float(vals.min()) if not vals.empty else 0.0
                            max_val = float(vals.max()) if not vals.empty else 1.0
                            
                            lim_mode_options = ["Auto (Min/Max in CSV)", "Custom"]
                            c_lim_type = st.radio(
                                "Limits Mode",
                                options=lim_mode_options,
                                index=_get_widget_index("tab9_actual_cell_lim_mode", lim_mode_options, 0),
                                horizontal=True,
                                key="tab9_actual_cell_lim_mode"
                            )
                            if c_lim_type == "Auto (Min/Max in CSV)":
                                cell_continuous_limits = (min_val, max_val)
                                st.caption(f"Scaling range: `{min_val:.4f}` to `{max_val:.4f}`")
                            else:
                                c1_lim, c2_lim = st.columns(2)
                                l_min = c1_lim.number_input("Min limit", value=float(st.session_state.get("tab9_actual_cell_lim_min", min_val)), key="tab9_actual_cell_lim_min")
                                l_max = c2_lim.number_input("Max limit", value=float(st.session_state.get("tab9_actual_cell_lim_max", max_val)), key="tab9_actual_cell_lim_max")
                                cell_continuous_limits = (l_min, l_max)
                                
                    elif cell_color_mode == "Column Comparison":
                        col1, col2 = st.columns(2)
                        cell_compare_col1 = col1.selectbox(
                            "Comparison Column 1",
                            options=cols,
                            index=_get_widget_index("tab9_actual_cell_compare_col1", cols, 0),
                            key="tab9_actual_cell_compare_col1"
                        )
                        cell_compare_col2 = col2.selectbox(
                            "Comparison Column 2",
                            options=cols,
                            index=_get_widget_index("tab9_actual_cell_compare_col2", cols, 0),
                            key="tab9_actual_cell_compare_col2"
                        )
                        
                        color_col1, color_col2 = st.columns(2)
                        cell_compare_match_color = color_col1.color_picker(
                            "Match Color",
                            value=st.session_state.get("tab9_actual_cell_compare_match_color", "#00ff00"),
                            key="tab9_actual_cell_compare_match_color"
                        )
                        cell_compare_mismatch_color = color_col2.color_picker(
                            "Mismatch Color",
                            value=st.session_state.get("tab9_actual_cell_compare_mismatch_color", "#ff0000"),
                            key="tab9_actual_cell_compare_mismatch_color"
                        )
                        
                    st.divider()
                    st.markdown("##### Outline Configurations")
                    cell_line_width = st.slider(
                        "Outline Width",
                        min_value=0,
                        max_value=5,
                        value=int(st.session_state.get("tab9_actual_cell_line_width", 1)),
                        step=1,
                        key="tab9_actual_cell_line_width"
                    )
                    cell_outline_only = st.checkbox(
                        "Draw Outline Only (No Fill)",
                        value=st.session_state.get("tab9_actual_cell_outline_only", False),
                        key="tab9_actual_cell_outline_only"
                    )
                    
                    col_out1, col_out2 = st.columns(2)
                    outline_color_mode_options = ["Match Cell Color", "Custom Color"]
                    cell_outline_color_mode = col_out1.selectbox(
                        "Outline Color Mode",
                        options=outline_color_mode_options,
                        index=_get_widget_index("tab9_actual_cell_outline_color_mode", outline_color_mode_options, 0),
                        key="tab9_actual_cell_outline_color_mode",
                        help="Choose whether to outline cells using their cell/categorical/comparison color, or a custom color."
                    )
                    cell_outline_color = "#ffffff"
                    if cell_outline_color_mode == "Custom Color":
                        cell_outline_color = col_out2.color_picker(
                            "Outline Custom Color",
                            value=st.session_state.get("tab9_actual_cell_outline_color", "#ffffff"),
                            key="tab9_actual_cell_outline_color"
                        )

        # 4. ANNOTATIONS & MASKING TAB
        with control_tabs[3]:
            st.markdown("### Annotations & Masking")
            
            if global_ann_dir and os.path.exists(global_ann_dir):
                st.caption(f"**Annotation Directory:** `{global_ann_dir}` (Inherited from global configuration)")
            else:
                st.caption(f"**Annotation Directory:** `{st.session_state.get('tab9_actual_annotation_dir', '')}`")
            
            # Matched tissue boundary
            auto_roi_geojson = None
            if roi_geojson_dir:
                roi_geojson_files = _list_files_cached(roi_geojson_dir, GEOJSON_EXTENSIONS, recursive=True)
                auto_roi_geojson = _best_match_file(roi_geojson_files, _sample_prefix(str(batch_value)), None) if roi_geojson_files else None
                
            if not auto_roi_geojson:
                fallback_boundary_dir = project_root / "data" / "boundaries"
                if fallback_boundary_dir.exists():
                    roi_geojson_files = _list_files_cached(str(fallback_boundary_dir), GEOJSON_EXTENSIONS, recursive=True)
                    auto_roi_geojson = _best_match_file(roi_geojson_files, _sample_prefix(str(batch_value)), None) if roi_geojson_files else None
                    
            matched_roi_groups = _load_geojson_groups(Path(auto_roi_geojson)) if auto_roi_geojson else []
            matched_roi_record = _roi_record_for_selection(matched_roi_groups, selected_tissue_value)
            
            visible_overlays = st.multiselect(
                "Visible Annotation Regions",
                options=visible_overlay_options,
                default=[x for x in st.session_state.get("tab9_actual_visible_overlays", visible_overlay_options) if x in visible_overlay_options],
                key="tab9_actual_visible_overlays",
                format_func=lambda key: key.replace("::", " | "),
            )
            
            st.divider()
            st.markdown("##### Annotation Styling")
            
            overlay_style = st.selectbox(
                "Overlay Style",
                options=["transparent", "solid", "outline"],
                index=_get_widget_index("tab9_actual_overlay_style", ["transparent", "solid", "outline"], 0),
                key="tab9_actual_overlay_style",
                disabled=not show_annotation_overlays,
            )
            
            overlay_line_width = int(
                st.slider(
                    "Outline Width",
                    min_value=1,
                    max_value=8,
                    value=int(st.session_state.get("tab9_actual_overlay_line_width", 2)),
                    step=1,
                    key="tab9_actual_overlay_line_width",
                    disabled=not show_annotation_overlays,
                )
            )
            
            overlay_color_map = {record["key"]: record["color"] for record in overlay_groups}
            if visible_overlays:
                with st.expander("Annotation Colors", expanded=True):
                    overlay_color_columns = st.columns(min(3, len(visible_overlays)))
                    for idx, overlay_key in enumerate(visible_overlays):
                        widget_key = f"tab9_actual_overlay_color_{_sanitize_key(overlay_key)}"
                        default_color = st.session_state.get(widget_key, overlay_color_map.get(overlay_key, "#ff7f0e"))
                        with overlay_color_columns[idx % len(overlay_color_columns)]:
                            overlay_color_map[overlay_key] = st.color_picker(
                                overlay_key.replace("::", " | "),
                                value=default_color,
                                key=widget_key,
                            )
                            
            st.divider()
            st.markdown("##### Masking Options")
            mask_sources = ["None"]
            if matched_roi_record is not None:
                mask_sources.append("Matched tissue ROI")
            mask_sources.extend(visible_overlays)
            saved_mask = st.session_state.get("tab9_actual_mask_source", [])
            if isinstance(saved_mask, str):
                if saved_mask == "None" or not saved_mask:
                    default_mask = []
                else:
                    default_mask = [saved_mask]
            else:
                default_mask = [opt for opt in saved_mask if opt in mask_sources and opt != "None"]

            mask_source = st.multiselect(
                "Mask to Annotations",
                options=[opt for opt in mask_sources if opt != "None"],
                default=default_mask,
                key="tab9_actual_mask_source",
                help="Keep only pixels inside the selected annotations / ROI polygons (combines them using a union).",
            )
            
            c_m1, c_m2 = st.columns(2)
            outside_mode = c_m1.selectbox(
                "Outside Pixels",
                options=["dim", "black", "custom", "transparent"],
                key="tab9_actual_outside_mode",
            )
            outside_dim_strength = c_m2.slider(
                "Dim Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                key="tab9_actual_dim_strength",
                help="Used only when Outside Pixels is set to dim.",
            )
            
            outside_custom_color = "#000000"
            if outside_mode == "custom":
                outside_custom_color = st.color_picker(
                    "Outside Color",
                    value="#000000",
                    key="tab9_actual_outside_color",
                )
                
            st.caption(f"**Annotation Source Directory:** `{overlay_geojson_dir}`")

        # 5. OUTPUT & LEGEND TAB
        with control_tabs[4]:
            st.markdown("### Output & Legend Settings")
            
            output_dir = directory_input_with_browse(
                label="Output Directory",
                text_key="tab9_actual_output_dir",
                default_value=output_dir_default,
                browser_key_prefix="tab9_actual_output_dir",
                start_dir_fallback=str(project_root),
            )
            
            st.divider()
            st.markdown("##### Layer Ordering & Visibility")
            st.caption("Customize the drawing order (Z-order) and visibility of layers.")
            
            col_o1, col_o2, col_o3 = st.columns(3)
            layer_options = ["Markers", "Tissue Annotations", "Cell Segmentations", "None"]
            
            with col_o1:
                l1 = st.selectbox(
                    "Layer 1 (Bottom)",
                    options=layer_options,
                    index=_get_widget_index("tab9_actual_layer_1", layer_options, 0),
                    key="tab9_actual_layer_1",
                )
                show_l1 = st.checkbox(
                    "Show Layer 1",
                    value=st.session_state.get("tab9_actual_show_layer_1", True),
                    key="tab9_actual_show_layer_1",
                )
                opacity_l1 = st.slider(
                    "Layer 1 Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("tab9_actual_layer_1_opacity", 1.0)),
                    step=0.05,
                    key="tab9_actual_layer_1_opacity",
                    disabled=not show_l1 or l1 == "None",
                )
                
            with col_o2:
                l2 = st.selectbox(
                    "Layer 2 (Middle)",
                    options=layer_options,
                    index=_get_widget_index("tab9_actual_layer_2", layer_options, 1),
                    key="tab9_actual_layer_2",
                )
                show_l2 = st.checkbox(
                    "Show Layer 2",
                    value=st.session_state.get("tab9_actual_show_layer_2", True),
                    key="tab9_actual_show_layer_2",
                )
                opacity_l2 = st.slider(
                    "Layer 2 Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("tab9_actual_layer_2_opacity", 1.0)),
                    step=0.05,
                    key="tab9_actual_layer_2_opacity",
                    disabled=not show_l2 or l2 == "None",
                )
                
            with col_o3:
                l3 = st.selectbox(
                    "Layer 3 (Top)",
                    options=layer_options,
                    index=_get_widget_index("tab9_actual_layer_3", layer_options, 2),
                    key="tab9_actual_layer_3",
                )
                show_l3 = st.checkbox(
                    "Show Layer 3",
                    value=st.session_state.get("tab9_actual_show_layer_3", True),
                    key="tab9_actual_show_layer_3",
                )
                opacity_l3 = st.slider(
                    "Layer 3 Opacity",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("tab9_actual_layer_3_opacity", 1.0)),
                    step=0.05,
                    key="tab9_actual_layer_3_opacity",
                    disabled=not show_l3 or l3 == "None",
                )
            
            background_color = st.color_picker(
                "Canvas Background Color",
                value=st.session_state.get("tab9_actual_background_color", "#ffffff"),
                key="tab9_actual_background_color",
                help="Base background color for the tissue display (defaults to white). Visible when markers are hidden or translucent."
            )
            
            st.divider()
            st.markdown("##### Publication Adornments")
            show_marker_legend = st.checkbox(
                "Show marker legend",
                value=st.session_state.get("tab9_actual_show_marker_legend", True),
                key="tab9_actual_show_marker_legend"
            )
            show_annotation_legend = st.checkbox(
                "Show annotation legend",
                value=st.session_state.get("tab9_actual_show_annotation_legend", True),
                key="tab9_actual_show_annotation_legend"
            )
            show_phenotype_legend = st.checkbox(
                "Show phenotype legend",
                value=st.session_state.get("tab9_actual_show_phenotype_legend", True),
                key="tab9_actual_show_phenotype_legend"
            )
            
            col_l1, col_l2, col_l3 = st.columns(3)
            marker_legend_position = col_l1.selectbox(
                "Marker legend position",
                options=ADORNMENT_POSITIONS,
                index=_get_widget_index("tab9_actual_marker_legend_position", ADORNMENT_POSITIONS, ADORNMENT_POSITIONS.index("below image") if "below image" in ADORNMENT_POSITIONS else 0),
                key="tab9_actual_marker_legend_position"
            )
            annotation_legend_position = col_l2.selectbox(
                "Annotation legend position",
                options=ADORNMENT_POSITIONS,
                index=_get_widget_index("tab9_actual_annotation_legend_position", ADORNMENT_POSITIONS, ADORNMENT_POSITIONS.index("below image") if "below image" in ADORNMENT_POSITIONS else 0),
                key="tab9_actual_annotation_legend_position"
            )
            phenotype_legend_position = col_l3.selectbox(
                "Phenotype legend position",
                options=ADORNMENT_POSITIONS,
                index=_get_widget_index("tab9_actual_phenotype_legend_position", ADORNMENT_POSITIONS, ADORNMENT_POSITIONS.index("below image") if "below image" in ADORNMENT_POSITIONS else 0),
                key="tab9_actual_phenotype_legend_position"
            )
            
            st.divider()
            st.markdown("##### Scale Bar")
            show_scale_bar = st.checkbox(
                "Show scale bar",
                value=st.session_state.get("tab9_actual_show_scale_bar", True),
                key="tab9_actual_show_scale_bar"
            )
            auto_scale_bar = st.checkbox(
                "Auto scale bar length",
                value=st.session_state.get("tab9_actual_auto_scale_bar", True),
                key="tab9_actual_auto_scale_bar"
            )
            
            col_s1, col_s2 = st.columns(2)
            scale_bar_um = float(
                col_s1.number_input(
                    "Scale bar length (µm)",
                    min_value=1.0,
                    value=float(st.session_state.get("tab9_actual_scale_bar_um", 20.0)),
                    step=1.0,
                    key="tab9_actual_scale_bar_um",
                    disabled=auto_scale_bar,
                )
            )
            scale_bar_position = col_s2.selectbox(
                "Scale bar position",
                options=ADORNMENT_POSITIONS,
                index=_get_widget_index("tab9_actual_scale_bar_position", ADORNMENT_POSITIONS, ADORNMENT_POSITIONS.index("bottom-right") if "bottom-right" in ADORNMENT_POSITIONS else 0),
                key="tab9_actual_scale_bar_position"
            )
            
            st.divider()
            st.markdown("##### Document Settings")
            col_d1, col_d2 = st.columns(2)
            output_dpi = int(
                col_d1.number_input(
                    "DPI",
                    min_value=72,
                    max_value=1200,
                    value=int(st.session_state.get("tab9_actual_output_dpi", 300)),
                    step=25,
                    key="tab9_actual_output_dpi"
                )
            )
            adornment_background = col_d2.checkbox(
                "Use black adornment background",
                value=st.session_state.get("tab9_actual_adornment_background", True),
                key="tab9_actual_adornment_background"
            )
            
            c_ex1, c_ex2 = st.columns(2)
            save_pdf = c_ex1.checkbox(
                "Save PDF",
                value=st.session_state.get("tab9_actual_save_pdf", True),
                key="tab9_actual_save_pdf"
            )
            save_svg = c_ex2.checkbox(
                "Save SVG",
                value=st.session_state.get("tab9_actual_save_svg", True),
                key="tab9_actual_save_svg"
            )

            # Standalone Legends (Separate Figures)
            import io
            st.divider()
            st.markdown("##### Standalone Legends (Separate Figures)")
            st.caption("Generate and download individual legend blocks as standalone publication-ready figures.")
            
            # Resolve the items
            # 1. Markers
            marker_items_standalone = [(marker, st.session_state.get(f"tab9_actual_marker_color_{_sanitize_key(marker)}", "#ffffff")) for marker in st.session_state.get("tab9_actual_markers", [])]
            # 2. Annotations
            selected_overlay_files_standalone = st.session_state.get("tab9_actual_overlay_files", [])
            overlay_groups_standalone = []
            for geojson_path in selected_overlay_files_standalone:
                if os.path.exists(geojson_path):
                    try:
                        overlay_groups_standalone.extend(_load_geojson_groups(Path(geojson_path)))
                    except Exception:
                        pass
            visible_overlay_options_standalone = [record["key"] for record in overlay_groups_standalone]
            visible_overlays_standalone = st.session_state.get("tab9_actual_visible_overlays", visible_overlay_options_standalone)
            selected_overlay_records_standalone = [record for record in overlay_groups_standalone if record["key"] in visible_overlays_standalone]
            annotation_items_standalone = [
                (
                    str(record["key"]).replace("::", " | "),
                    str(st.session_state.get(f"tab9_actual_overlay_color_{_sanitize_key(record['key'])}", record["color"] or "#ffffff")),
                )
                for record in selected_overlay_records_standalone
            ]
            # 3. Phenotypes
            phenotype_items_standalone = []
            cell_phenotype_col_standalone = st.session_state.get("tab9_actual_cell_phenotype_col", "")
            if st.session_state.get("tab9_actual_load_cells", False) and st.session_state.get("tab9_actual_cell_color_mode") == "Categorical / Phenotype Column" and cell_phenotype_col_standalone:
                # Resolve unique phenotypes from CSV batch
                unique_phenos_standalone = []
                if selected_batch_col and batch_value:
                    csv_batch_df_standalone = df[df[selected_batch_col].astype(str) == str(batch_value)].copy()
                    if selected_tissue_col != "None" and selected_tissue_value != "All":
                        csv_batch_df_standalone = csv_batch_df_standalone[csv_batch_df_standalone[selected_tissue_col].astype(str) == str(selected_tissue_value)].copy()
                    csv_batch_df_standalone = _apply_custom_filters(csv_batch_df_standalone)
                    if cell_phenotype_col_standalone in csv_batch_df_standalone.columns:
                        unique_phenos_standalone = sorted(csv_batch_df_standalone[cell_phenotype_col_standalone].dropna().astype(str).unique().tolist())
                for pheno in unique_phenos_standalone:
                    widget_key = f"tab9_actual_cell_color_val_{_sanitize_key(pheno)}"
                    color = st.session_state.get(widget_key, "#ffffff")
                    phenotype_items_standalone.append((pheno, color))

            standalone_cols = st.columns(3)
            
            # Helper to convert PIL Image to bytes for download
            def img_to_bytes(img):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            # Render Marker Legend block
            if marker_items_standalone:
                with standalone_cols[0]:
                    st.markdown("**Markers Legend**")
                    marker_legend_colored_labels_standalone = st.session_state.get("tab9_actual_marker_legend_colored_labels", False)
                    marker_block = _make_legend_block("Markers", marker_items_standalone, (800, 800), use_background=True, color_labels_with_item_color=marker_legend_colored_labels_standalone)
                    if marker_block:
                        st.image(marker_block, width="stretch")
                        st.download_button(
                            label="Download Markers Legend",
                            data=img_to_bytes(marker_block),
                            file_name="markers_legend.png",
                            mime="image/png",
                            key="btn_download_marker_legend"
                        )
            
            # Render Annotations Legend block
            if annotation_items_standalone:
                with standalone_cols[1]:
                    st.markdown("**Annotations Legend**")
                    annotation_block = _make_legend_block("Annotations", annotation_items_standalone, (800, 800), use_background=True)
                    if annotation_block:
                        st.image(annotation_block, width="stretch")
                        st.download_button(
                            label="Download Annotations Legend",
                            data=img_to_bytes(annotation_block),
                            file_name="annotations_legend.png",
                            mime="image/png",
                            key="btn_download_annotation_legend"
                        )
                        
            # Render Phenotypes Legend block
            if phenotype_items_standalone:
                with standalone_cols[2]:
                    st.markdown("**Phenotypes Legend**")
                    phenotype_block = _make_legend_block("Phenotypes", phenotype_items_standalone, (800, 800), use_background=True)
                    if phenotype_block:
                        st.image(phenotype_block, width="stretch")
                        st.download_button(
                            label="Download Phenotypes Legend",
                            data=img_to_bytes(phenotype_block),
                            file_name="phenotypes_legend.png",
                            mime="image/png",
                            key="btn_download_phenotype_legend"
                        )

    # PREVIEW & EXPORT PANEL (Right Column)
    with col_preview:
        st.markdown("### Preview & Export Panel")
        
        if not selected_x_col or not selected_y_col:
            st.warning("Select X and Y coordinate columns in Data & ROI tab to enable Actual Tissue.")
            return
        if selected_x_col == selected_y_col:
            st.error("X Coordinate Column and Y Coordinate Column must be different.")
            return
        if not selected_batch_col:
            st.warning("Select a Batch ID column in Data & ROI tab to enable Actual Tissue.")
            return
        if not batch_value:
            st.warning("Select a batch value to continue.")
            return

        filtered_df = df[df[selected_batch_col].astype(str) == str(batch_value)].copy()
        if selected_tissue_col != "None" and selected_tissue_value != "All":
            filtered_df = filtered_df[filtered_df[selected_tissue_col].astype(str) == str(selected_tissue_value)].copy()

        filtered_df[selected_x_col] = pd.to_numeric(filtered_df[selected_x_col], errors="coerce")
        filtered_df[selected_y_col] = pd.to_numeric(filtered_df[selected_y_col], errors="coerce")
        filtered_df = filtered_df.dropna(subset=[selected_x_col, selected_y_col])
        filtered_df = _apply_custom_filters(filtered_df)

        if filtered_df.empty:
            st.warning("No cells remain after applying the selected batch / tissue filters.")
            return

        image_path = Path(selected_image)
        channel_names, channel_colors, _channel_meta, _mpp = _get_channel_info_cached(
            str(image_path),
            image_path.stat().st_mtime,
        )
        channel_names = [str(name) for name in channel_names]
        if not channel_names:
            st.error("Unable to extract channels from the selected image.")
            return

        selected_markers = st.session_state.get("tab9_actual_markers", [])
        if not selected_markers and channel_names:
            selected_markers = _default_marker_selection(channel_names)

        marker_color_map = {}
        for marker in selected_markers:
            widget_key = f"tab9_actual_marker_color_{_sanitize_key(marker)}"
            stored_color = st.session_state.get(widget_key)
            if not stored_color or not str(stored_color).startswith("#") or len(stored_color) < 4:
                marker_color_map[marker] = "#ffffff"
            else:
                marker_color_map[marker] = stored_color

        import time
        with st.spinner("Generating composite preview..."):
            t0 = time.perf_counter()
            composite_image, (full_h, full_w) = _compose_preview_image(
                image_path=image_path,
                channel_names=channel_names,
                color_map=marker_color_map,
                selected_markers=selected_markers,
                max_dim=preview_max_dim,
                low_percentile=marker_low_percentile,
                high_percentile=marker_high_percentile,
            )
            t_comp = time.perf_counter() - t0
        _log_console(f"Generated composite preview image in {t_comp:.2f}s")

        crop_box_full = _crop_box_from_cells(filtered_df, selected_x_col, selected_y_col, crop_padding, (full_h, full_w)) if crop_to_cells else None
        selected_overlay_records = [record for record in overlay_groups if record["key"] in visible_overlays]

        roi_only_annotation_key = "Auto from visible annotations"
        if crop_mode == "Selected ROI only" and roi_only_source in {"Auto", "Selected annotation region"} and selected_overlay_records:
            roi_only_annotation_options = ["Auto from visible annotations"] + [record["key"] for record in selected_overlay_records]
            st.session_state.setdefault("tab9_actual_roi_only_annotation_key", "Auto from visible annotations")
            roi_only_annotation_key = st.selectbox(
                "Selected ROI Annotation Source",
                options=roi_only_annotation_options,
                key="tab9_actual_roi_only_annotation_key",
                format_func=lambda key: key if key == "Auto from visible annotations" else key.replace("::", " | "),
                help="Choose which selected visible annotation region should define the ROI-only mask.",
            )

        transformed_roi_geom = matched_roi_record["geometry"] if matched_roi_record is not None else None
        overlay_geometries = {record["key"]: record["geometry"] for record in selected_overlay_records}
        roi_isolation_geom, roi_isolation_label = _resolve_roi_isolation_geometry(
            crop_mode=crop_mode,
            roi_only_source=roi_only_source,
            roi_only_annotation_key=roi_only_annotation_key,
            cells_df=filtered_df,
            x_col=selected_x_col,
            y_col=selected_y_col,
            padding_px=crop_padding,
            matched_roi_geom=transformed_roi_geom,
            selected_overlay_records=selected_overlay_records,
        )

        show_roi_outline = st.checkbox(
            "Show matched tissue ROI boundary",
            value=matched_roi_record is not None,
            disabled=matched_roi_record is None,
            key="tab9_actual_show_roi_outline",
        )
        roi_outline_color = st.color_picker(
            "Tissue ROI Color",
            value="#00ff80",
            key="tab9_actual_roi_color",
            disabled=matched_roi_record is None,
        )
        
        with st.spinner("Rendering overlays..."):
            t0 = time.perf_counter()
            rendered, crop_box_px, active_mask_geom = _render_actual_image(
                composite_image=composite_image,
                full_hw=(full_h, full_w),
                crop_box_full=crop_box_full,
                crop_mode=crop_mode,
                transformed_roi_geom=roi_isolation_geom,
                selected_overlay_records=selected_overlay_records,
                overlay_geometries=overlay_geometries,
                mask_source=mask_source,
                outside_mode=outside_mode,
                outside_custom_color=outside_custom_color,
                outside_dim_strength=outside_dim_strength,
                show_roi_outline=show_roi_outline,
                roi_outline_color=roi_outline_color,
                overlay_color_map=overlay_color_map,
                overlay_style=overlay_style,
                overlay_fill_alpha=overlay_fill_alpha,
                overlay_line_width=overlay_line_width,
                show_annotation_overlays=show_annotation_overlays,
                # Cell overlay properties
                show_cells=show_cells,
                cells_list=cells_list,
                cell_color_mode=cell_color_mode,
                cell_single_color=cell_single_color,
                cell_phenotype_col=cell_phenotype_col,
                cell_phenotype_colors=cell_phenotype_colors,
                cell_continuous_col=cell_continuous_col,
                cell_continuous_cmap=cell_continuous_cmap,
                cell_continuous_limits=cell_continuous_limits,
                cell_fill_alpha=cell_fill_alpha,
                cell_line_width=cell_line_width,
                cell_outline_only=cell_outline_only,
                layer_order=[
                    st.session_state.get("tab9_actual_layer_1", "Markers"),
                    st.session_state.get("tab9_actual_layer_2", "Tissue Annotations"),
                    st.session_state.get("tab9_actual_layer_3", "Cell Segmentations"),
                ],
                show_markers=show_markers,
                markers_opacity=markers_opacity,
                cell_compare_col1=cell_compare_col1,
                cell_compare_col2=cell_compare_col2,
                cell_compare_match_color=cell_compare_match_color,
                cell_compare_mismatch_color=cell_compare_mismatch_color,
                cell_outline_color_mode=cell_outline_color_mode,
                cell_outline_color=cell_outline_color,
                background_color=st.session_state.get("tab9_actual_background_color", "#ffffff"),
            )
            t_render = time.perf_counter() - t0
        _log_console(f"Rendered overlays and cell shapes in {t_render:.2f}s")
            
        effective_scale_bar_um = _nice_scale_bar_um(rendered.width, pixel_size_um) if auto_scale_bar else float(scale_bar_um)
        
        annotation_legend_items = [
            (
                str(record["key"]).replace("::", " | "),
                str(overlay_color_map.get(record["key"], record["color"]) or "#ffffff"),
            )
            for record in selected_overlay_records
        ]
        
        phenotype_legend_items = []
        if load_cells and cell_color_mode == "Categorical / Phenotype Column" and cell_phenotype_colors and show_cell_legend:
            for pheno_val, color in cell_phenotype_colors.items():
                phenotype_legend_items.append((str(pheno_val), color))

        crop_w_native = (crop_box_full[2] - crop_box_full[0]) if crop_box_full is not None else float(full_w)
        rendered_scale = rendered.width / float(max(crop_w_native, 1.0))
        preview_pixel_size_um = float(pixel_size_um) / rendered_scale

        show_phenotype_legend = st.session_state.get("tab9_actual_show_phenotype_legend", True)
        phenotype_legend_position = st.session_state.get("tab9_actual_phenotype_legend_position", "below image")
        marker_legend_colored_labels = st.session_state.get("tab9_actual_marker_legend_colored_labels", False)

        rendered = _draw_publication_adornments(
            image=rendered,
            marker_items=[(marker, marker_color_map.get(marker, "#ffffff")) for marker in selected_markers],
            annotation_items=annotation_legend_items,
            phenotype_items=phenotype_legend_items,
            show_marker_legend=bool(show_marker_legend),
            show_annotation_legend=bool(show_annotation_legend),
            show_phenotype_legend=bool(show_phenotype_legend),
            show_scale_bar=bool(show_scale_bar),
            marker_legend_position=str(marker_legend_position),
            annotation_legend_position=str(annotation_legend_position),
            phenotype_legend_position=str(phenotype_legend_position),
            scale_bar_position=str(scale_bar_position),
            scale_bar_um=effective_scale_bar_um,
            pixel_size_um=preview_pixel_size_um,
            use_background=bool(adornment_background),
            marker_legend_colored_labels=bool(marker_legend_colored_labels),
        )

        st.image(
            rendered,
            caption="Actual tissue preview",
            width='stretch',
        )

        st.divider()
        summary_cols = st.columns(4)
        summary_cols[0].metric("Cells", f"{len(filtered_df):,}")
        summary_cols[1].metric("Markers", str(len(selected_markers) if selected_markers else 1))
        summary_cols[2].metric("Overlay Regions", str(len(selected_overlay_records) + (1 if show_roi_outline and transformed_roi_geom is not None else 0)))
        summary_cols[3].metric("Preview Size", f"{rendered.width} x {rendered.height}")

        if load_cells and cells_list:
            if cell_color_mode == "Column Comparison" and cell_compare_col1 and cell_compare_col2:
                matches = 0
                mismatches = 0
                for cell in cells_list:
                    val1 = cell.get("csv_data", {}).get(cell_compare_col1)
                    val2 = cell.get("csv_data", {}).get(cell_compare_col2)
                    if val1 is not None and val2 is not None and not pd.isna(val1) and not pd.isna(val2):
                        match = False
                        try:
                            if np.isclose(float(val1), float(val2)):
                                match = True
                        except Exception:
                            pass
                        if not match and str(val1).strip() == str(val2).strip():
                            match = True
                        if match:
                            matches += 1
                        else:
                            mismatches += 1
                
                total_compare = matches + mismatches
                if total_compare > 0:
                    match_pct = (matches / total_compare) * 100
                    st.markdown("---")
                    st.markdown("##### Cell Comparison Statistics")
                    comp_cols = st.columns(3)
                    comp_cols[0].metric("Matches (Green)", f"{matches:,}")
                    comp_cols[1].metric("Mismatches (Red)", f"{mismatches:,}")
                    comp_cols[2].metric("Match Percentage", f"{match_pct:.2f}%")
                else:
                    st.markdown("---")
                    st.warning("No cells found with valid data for comparison columns.")

            elif cell_color_mode == "Categorical / Phenotype Column" and cell_phenotype_col:
                pheno_counts = {}
                total_valid = 0
                for cell in cells_list:
                    val = cell.get("csv_data", {}).get(cell_phenotype_col)
                    if val is not None and not pd.isna(val):
                        val_str = str(val).strip()
                        pheno_counts[val_str] = pheno_counts.get(val_str, 0) + 1
                        total_valid += 1
                
                if total_valid > 0:
                    st.markdown("---")
                    st.markdown("##### Phenotype Proportions")
                    sorted_phenos = sorted(pheno_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    list_items = []
                    for pheno_val, count in sorted_phenos:
                        pct = (count / total_valid) * 100
                        color = cell_phenotype_colors.get(pheno_val, "#777777") if cell_phenotype_colors else "#777777"
                        item_html = (
                            f'<div style="display: flex; align-items: center; margin-bottom: 4px;">'
                            f'<span style="display: inline-block; width: 12px; height: 12px; background-color: {color}; border-radius: 2px; margin-right: 8px;"></span>'
                            f'<span><strong>{pheno_val}</strong>: {count:,} cells ({pct:.2f}%)</span>'
                            f'</div>'
                        )
                        list_items.append(item_html)
                    
                    st.markdown("\n".join(list_items), unsafe_allow_html=True)
                else:
                    st.markdown("---")
                    st.warning("No cells found with valid phenotype data.")

        if auto_roi_geojson:
            st.caption(f"**Matched ROI boundary GeoJSON:** `{auto_roi_geojson}`")
        st.caption(f"**Matched image:** `{image_path}`")
        if crop_to_cells:
            st.caption(f"**Crop mode:** `{crop_mode}`")
        if crop_mode == "Selected ROI only" and roi_isolation_label:
            st.caption(f"**Selected ROI isolation source:** `{roi_isolation_label}`")

        # EXPORTS AND DOWNLOADS PANEL
        st.divider()
        png_data = _image_bytes_with_dpi(rendered, fmt="png", dpi=output_dpi)
        stem = _build_output_stem(batch_value, selected_tissue_value, selected_markers)

        native_prepare_key = f"_actual_native_png::{stem}"
        native_name_key = f"_actual_native_png_name::{stem}"

        native_png = st.session_state.get(native_prepare_key)
        native_png_name = st.session_state.get(native_name_key, f"{stem}_native.png")

        export_options = [
            "Download Preview PNG",
            "Download Publication PDF",
            "Download Publication SVG",
            "Download Native Resolution PNG",
            "Export Native OME-TIFF",
            "Export Native TIFF Stack",
            "Export 16-bit Composite TIFF",
            "Save Actual Tissue Figure + Summary",
        ]
        
        with st.expander("Save / Export Actions", expanded=True):
            export_choice = st.selectbox(
                "Save As",
                options=export_options,
                key="tab9_actual_export_choice",
            )
            st.caption("Choose an output type, then use the action button below.")

            if export_choice == "Download Preview PNG":
                st.download_button(
                    "Download Selected Export",
                    data=png_data,
                    file_name=f"{stem}.png",
                    mime="image/png",
                    key="tab9_actual_download_png",
                    width='stretch',
                )

            elif export_choice == "Download Publication PDF":
                pdf_data = _figure_bytes(rendered, fmt="pdf", dpi=output_dpi)
                st.download_button(
                    "Download Selected Export",
                    data=pdf_data,
                    file_name=f"{stem}.pdf",
                    mime="application/pdf",
                    key="tab9_actual_download_pdf",
                    width='stretch',
                )

            elif export_choice == "Download Publication SVG":
                svg_data = _figure_bytes(rendered, fmt="svg", dpi=output_dpi)
                st.download_button(
                    "Download Selected Export",
                    data=svg_data,
                    file_name=f"{stem}.svg",
                    mime="image/svg+xml",
                    key="tab9_actual_download_svg",
                    width='stretch',
                )

            elif export_choice == "Download Native Resolution PNG":
                if not isinstance(native_png, (bytes, bytearray)):
                    if st.button("Prepare Selected Export", key="tab9_actual_prepare_native_png", width='stretch'):
                        with st.spinner("Rendering native-resolution image..."):
                            native_composite, _ = _compose_preview_image(
                                image_path=image_path,
                                channel_names=channel_names,
                                color_map=marker_color_map,
                                selected_markers=selected_markers,
                                max_dim=None,
                                low_percentile=marker_low_percentile,
                                high_percentile=marker_high_percentile,
                            )
                            native_rendered, _native_crop_box_px, _native_mask_geom = _render_actual_image(
                                composite_image=native_composite,
                                full_hw=(full_h, full_w),
                                crop_box_full=crop_box_full,
                                crop_mode=crop_mode,
                                transformed_roi_geom=roi_isolation_geom,
                                selected_overlay_records=selected_overlay_records,
                                overlay_geometries=overlay_geometries,
                                mask_source=mask_source,
                                outside_mode=outside_mode,
                                outside_custom_color=outside_custom_color,
                                outside_dim_strength=outside_dim_strength,
                                show_roi_outline=show_roi_outline,
                                roi_outline_color=roi_outline_color,
                                overlay_color_map=overlay_color_map,
                                overlay_style=overlay_style,
                                overlay_fill_alpha=overlay_fill_alpha,
                                overlay_line_width=overlay_line_width,
                                show_annotation_overlays=show_annotation_overlays,
                                # Cell overlay properties
                                show_cells=show_cells,
                                cells_list=cells_list,
                                cell_color_mode=cell_color_mode,
                                cell_single_color=cell_single_color,
                                cell_phenotype_col=cell_phenotype_col,
                                cell_phenotype_colors=cell_phenotype_colors,
                                cell_continuous_col=cell_continuous_col,
                                cell_continuous_cmap=cell_continuous_cmap,
                                cell_continuous_limits=cell_continuous_limits,
                                cell_fill_alpha=cell_fill_alpha,
                                cell_line_width=cell_line_width,
                                cell_outline_only=cell_outline_only,
                                layer_order=[
                                    st.session_state.get("tab9_actual_layer_1", "Markers"),
                                    st.session_state.get("tab9_actual_layer_2", "Tissue Annotations"),
                                    st.session_state.get("tab9_actual_layer_3", "Cell Segmentations"),
                                ],
                                show_markers=show_markers,
                                markers_opacity=markers_opacity,
                                cell_compare_col1=cell_compare_col1,
                                cell_compare_col2=cell_compare_col2,
                                cell_compare_match_color=cell_compare_match_color,
                                cell_compare_mismatch_color=cell_compare_mismatch_color,
                                cell_outline_color_mode=cell_outline_color_mode,
                                cell_outline_color=cell_outline_color,
                                background_color=st.session_state.get("tab9_actual_background_color", "#ffffff"),
                            )
                            native_rendered = _draw_publication_adornments(
                                native_rendered,
                                marker_items=[(marker, marker_color_map.get(marker, "#ffffff")) for marker in selected_markers],
                                annotation_items=annotation_legend_items,
                                show_marker_legend=bool(show_marker_legend),
                                show_annotation_legend=bool(show_annotation_legend),
                                show_scale_bar=bool(show_scale_bar),
                                marker_legend_position=str(marker_legend_position),
                                annotation_legend_position=str(annotation_legend_position),
                                scale_bar_position=str(scale_bar_position),
                                scale_bar_um=effective_scale_bar_um,
                                pixel_size_um=float(pixel_size_um),
                                use_background=bool(adornment_background),
                            )
                            st.session_state[native_prepare_key] = _image_bytes_with_dpi(native_rendered, fmt="png", dpi=output_dpi)
                            st.session_state[native_name_key] = f"{stem}_native.png"
                            st.rerun()
                else:
                    st.download_button(
                        "Download Selected Export",
                        data=bytes(native_png),
                        file_name=str(native_png_name),
                        mime="image/png",
                        key="tab9_actual_download_native_png",
                        width='stretch',
                    )

            elif export_choice == "Export Native OME-TIFF":
                if st.button("Save Selected Export", key="tab9_actual_export_native_ome", width='stretch'):
                    try:
                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        layers, native_crop_box, native_mask_geom = _prepare_native_layers(
                            image_path=image_path,
                            channel_names=channel_names,
                            selected_markers=selected_markers,
                            crop_box_full=crop_box_full,
                            full_hw=(full_h, full_w),
                            crop_mode=crop_mode,
                            roi_isolation_geom=roi_isolation_geom,
                            selected_overlay_records=selected_overlay_records,
                            overlay_geometries=overlay_geometries,
                            mask_source=mask_source,
                        )
                        ome_path = out_dir / f"{stem}.ome.tiff"
                        export_markers = [marker for marker in selected_markers if marker in channel_names] or [channel_names[0]]
                        export_colors = [_rgb_tuple(marker_color_map.get(marker, "#ffffff")) for marker in export_markers]
                        save_roi_pyramidal_ome_tiff(layers, export_markers, export_colors, str(ome_path), float(pixel_size_um))
                        st.success(f"Saved native OME-TIFF to `{ome_path}`")
                    except Exception as exc:
                        st.error(f"Failed to export native OME-TIFF: {exc}")

            elif export_choice == "Export Native TIFF Stack":
                if st.button("Save Selected Export", key="tab9_actual_export_native_tiff", width='stretch'):
                    try:
                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        layers, native_crop_box, native_mask_geom = _prepare_native_layers(
                            image_path=image_path,
                            channel_names=channel_names,
                            selected_markers=selected_markers,
                            crop_box_full=crop_box_full,
                            full_hw=(full_h, full_w),
                            crop_mode=crop_mode,
                            roi_isolation_geom=roi_isolation_geom,
                            selected_overlay_records=selected_overlay_records,
                            overlay_geometries=overlay_geometries,
                            mask_source=mask_source,
                        )
                        tiff_path = out_dir / f"{stem}.tiff"
                        _save_plain_tiff_stack(layers, tiff_path)
                        st.success(f"Saved native TIFF stack to `{tiff_path}`")
                    except Exception as exc:
                        st.error(f"Failed to export native TIFF stack: {exc}")

            elif export_choice == "Export 16-bit Composite TIFF":
                if st.button("Save Selected Export", key="tab9_actual_export_rgb16_tiff", width='stretch'):
                    try:
                        out_dir = Path(output_dir)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        layers, native_crop_box, native_mask_geom = _prepare_native_layers(
                            image_path=image_path,
                            channel_names=channel_names,
                            selected_markers=selected_markers,
                            crop_box_full=crop_box_full,
                            full_hw=(full_h, full_w),
                            crop_mode=crop_mode,
                            roi_isolation_geom=roi_isolation_geom,
                            selected_overlay_records=selected_overlay_records,
                            overlay_geometries=overlay_geometries,
                            mask_source=mask_source,
                        )
                        export_markers = [marker for marker in selected_markers if marker in channel_names] or [channel_names[0]]
                        export_colors = [_rgb_tuple(marker_color_map.get(marker, "#ffffff")) for marker in export_markers]
                        rgb16 = _build_rgb16_composite(
                            layers,
                            export_colors,
                            low_percentile=marker_low_percentile,
                            high_percentile=marker_high_percentile,
                        )
                        rgb16_path = out_dir / f"{stem}_rgb16.tiff"
                        _save_rgb16_tiff(rgb16, rgb16_path)
                        st.success(f"Saved 16-bit composite TIFF to `{rgb16_path}`")
                    except Exception as exc:
                        st.error(f"Failed to export 16-bit composite TIFF: {exc}")

            else:
                if st.button("Save Selected Export", type="primary", key="tab9_actual_save_btn", width='stretch'):
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    png_path = out_dir / f"{stem}.png"
                    summary_path = out_dir / f"{stem}_summary.csv"
                    _save_png_with_dpi(rendered, png_path, output_dpi)
                    pdf_path = out_dir / f"{stem}.pdf"
                    svg_path = out_dir / f"{stem}.svg"
                    if save_pdf:
                        _save_publication_figure(rendered, pdf_path, fmt="pdf", dpi=output_dpi)
                    if save_svg:
                        _save_publication_figure(rendered, svg_path, fmt="svg", dpi=output_dpi)
                        
                    crop_left, crop_top, crop_right, crop_bottom = crop_box_full if crop_box_full is not None else (0.0, 0.0, float(full_w), float(full_h))
                    summary_df = pd.DataFrame(
                        [
                            {
                                "batch_column": str(selected_batch_col),
                                "batch_value": str(batch_value),
                                "tissue_column": str(selected_tissue_col),
                                "tissue_value": str(selected_tissue_value),
                                "image_path": str(image_path),
                                "roi_geojson_path": str(auto_roi_geojson) if auto_roi_geojson else "",
                                "overlay_geojson_files": ";".join(selected_overlay_files),
                                "visible_overlay_regions": ";".join(visible_overlays),
                                "mask_source": ";".join(mask_source) if isinstance(mask_source, list) else str(mask_source),
                                "markers": ";".join(selected_markers),
                                "marker_low_percentile": float(marker_low_percentile),
                                "marker_high_percentile": float(marker_high_percentile),
                                "pixel_size_um": float(pixel_size_um),
                                "dpi": int(output_dpi),
                                "show_marker_legend": bool(show_marker_legend),
                                "show_annotation_legend": bool(show_annotation_legend),
                                "marker_legend_position": str(marker_legend_position),
                                "annotation_legend_position": str(annotation_legend_position),
                                "show_scale_bar": bool(show_scale_bar),
                                "scale_bar_position": str(scale_bar_position),
                                "adornment_background": bool(adornment_background),
                                "scale_bar_um": float(effective_scale_bar_um),
                                "crop_to_cells": bool(crop_to_cells),
                                "crop_mode": str(crop_mode),
                                "roi_only_source": str(roi_only_source),
                                "roi_only_resolved_geometry": str(roi_isolation_label) if roi_isolation_label else "",
                                "show_annotation_overlays": bool(show_annotation_overlays),
                                "layer_order": ";".join([
                                    st.session_state.get("tab9_actual_layer_1", "Markers"),
                                    st.session_state.get("tab9_actual_layer_2", "Tissue Annotations"),
                                    st.session_state.get("tab9_actual_layer_3", "Cell Segmentations"),
                                ]),
                                "show_markers": bool(show_markers),
                                "markers_opacity": float(markers_opacity),
                                "crop_left_px": float(crop_left),
                                "crop_top_px": float(crop_top),
                                "crop_right_px": float(crop_right),
                                "crop_bottom_px": float(crop_bottom),
                                "has_active_mask": bool(active_mask_geom is not None and not active_mask_geom.is_empty),
                                "n_cells": int(len(filtered_df)),
                                "output_png": str(png_path),
                                "output_pdf": str(pdf_path) if save_pdf else "",
                                "output_svg": str(svg_path) if save_svg else "",
                            }
                        ]
                    )
                    summary_df.to_csv(summary_path, index=False)
                    st.success(f"Saved actual tissue figure to `{png_path}` and summary to `{summary_path}`.")
