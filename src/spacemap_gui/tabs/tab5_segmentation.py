import streamlit as st
import os
import csv
import re
import time
from pathlib import Path
from datetime import datetime
import tifffile
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError for large images
import json
import pandas as pd
import torch
from spacemap_gui.file_browser import render_file_browser
from spacemap_gui.ui_utils import (
    directory_input_with_browse,
    show_preflight_errors,
    validate_run_directories,
    validate_selected_files,
)
from spacemap_gui.process_log import record_process_event
from spacemap_gui.tabs import segmentation_spatial_ui as spatial_ui
from spacemap_gui.tabs import cell_tagging_ui

from spacemap.segmentation_cellpose import SegmentationConfig, segment_image_cellpose, get_channel_names
from spacemap.background_subtraction_utils import get_channel_info
from spacemap.utils import hex_to_rgb
import base64
from io import BytesIO
from shapely.geometry import shape, Polygon
import matplotlib.pyplot as plt
from PIL import ImageDraw
import seaborn as sns
from typing import Any, Dict, List, cast, Optional
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------
# Logging utilities (similar to Tab1 registration)
# -----------------------------------------------------------------------------

def get_segmentation_metadata_log(file_path: str) -> Dict[str, Any]:
    """
    Safely extract metadata from any TIFF/OME-TIFF/QPTIFF file for logging purposes.
    """
    result = {
        'file_size_gb': 'N/A',
        'dtype': 'N/A',
        'shape': 'N/A',
        'mpp': 'N/A',
        'pyramid_levels': 'N/A',
        'channels': 'N/A',
        'n_channels': 0
    }
    
    try:
        if not os.path.exists(file_path):
            result['file_size_gb'] = 'File Not Found'
            return result
        
        file_size_bytes = os.path.getsize(file_path)
        result['file_size_gb'] = f"{file_size_bytes / (1024**3):.3f}"
        
        with tifffile.TiffFile(file_path) as tif:
            first_page = tif.pages[0]
            result['dtype'] = str(first_page.dtype)
            result['shape'] = str(first_page.shape)
            
            page_shapes = []
            for page in tif.pages[:20]:
                page_shapes.append(f"({page.shape[0]}, {page.shape[1]})")
            
            unique_shapes = list(dict.fromkeys(page_shapes))
            if len(unique_shapes) > 1:
                result['pyramid_levels'] = ", ".join(unique_shapes[:5])
            else:
                result['pyramid_levels'] = "Single Level"
            
            try:
                if tif.ome_metadata:
                    ome_root = ET.fromstring(tif.ome_metadata)
                    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels_elem = ome_root.find('.//ome:Pixels', namespace)
                    if pixels_elem is None:
                        pixels_elem = ome_root.find('.//Pixels')
                    if pixels_elem is not None:
                        mpp_val = pixels_elem.attrib.get('PhysicalSizeX', None)
                        if mpp_val:
                            result['mpp'] = mpp_val
                
                if result['mpp'] == 'N/A':
                    res = first_page.resolution
                    if res and res[0] and res[1]:
                        px_per_unit = res[0] / res[1]
                        if px_per_unit > 0:
                            result['mpp'] = f"{25400.0 / px_per_unit:.4f}"
            except Exception:
                pass
            
            try:
                names, _, _, _ = get_channel_info(file_path)
                if names:
                    result['channels'] = "; ".join(names)
                    result['n_channels'] = len(names)
            except Exception:
                result['n_channels'] = len([p for p in tif.pages if p.shape[0] > 500])
                result['channels'] = f"{result['n_channels']} channels (names not extracted)"
                
    except Exception as e:
        result['file_size_gb'] = f"Extraction Failed: {str(e)[:50]}"
    
    return result


def create_seg_log_session_folder(output_dir: str) -> str:
    """Create a timestamped session folder inside output_dir/log/"""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "log", f"session_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def write_seg_image_log(log_dir: str, image_data: Dict[str, Any]):
    """Write individual image log CSV file."""
    image_name = image_data.get('Image_Name', 'unknown')
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_name)
    image_log_path = os.path.join(log_dir, f"image_{safe_name}.csv")
    
    with open(image_log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=image_data.keys())
        writer.writeheader()
        writer.writerow(image_data)


def append_to_seg_details(log_dir: str, image_data: Dict[str, Any], is_first: bool = False):
    """Append image data to the master images_details.csv file."""
    details_path = os.path.join(log_dir, "images_details.csv")
    
    mode = 'w' if is_first else 'a'
    write_header = is_first
    
    with open(details_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=image_data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(image_data)


def write_seg_session_stats(log_dir: str, stats: Dict[str, Any]):
    """Write session summary statistics to session_stats.csv."""
    stats_path = os.path.join(log_dir, "session_stats.csv")
    
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)


# Monkey patch for streamlit-drawable-canvas compatibility with newer Streamlit versions
# (shared implementation in spacemap_gui.ui_utils)
from spacemap_gui.ui_utils import patch_canvas_image_to_url
patch_canvas_image_to_url()

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False
    st_canvas = None

def image_to_base64(img):
    """Convert PIL Image to base64 string for st_canvas"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def geojson_to_fabric(geojson_data, scale_x, scale_y):
    """Convert GeoJSON to Fabric.js JSON format for st_canvas"""
    objects = []
    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data.get("features", [])
    elif geojson_data.get("type") == "Feature":
        features = [geojson_data]
    else:
        features = []
        
    for feature in features:
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        label = props.get("label", "ROI")
        
        gtype = geom.get("type")
        if gtype == "Polygon":
            coords = geom.get("coordinates", [])
            if not coords: continue
            ring = coords[0]
            path_cmds = []
            if ring:
                x0, y0 = ring[0]
                path_cmds.append(['M', x0 * scale_x, y0 * scale_y])
                for pt in ring[1:]:
                    x, y = pt
                    path_cmds.append(['L', x * scale_x, y * scale_y])
                path_cmds.append(['z'])
            obj = {
                "type": "path",
                "path": path_cmds,
                "fill": "rgba(255, 165, 0, 0.3)",
                "stroke": "#000000",
                "strokeWidth": 2,
                "opacity": 1,
                "closed": True
            }
            objects.append(obj)
        elif gtype == "MultiPolygon":
            coords = geom.get("coordinates", [])
            for poly in coords:
                if not poly: continue
                ring = poly[0]
                path_cmds = []
                if ring:
                    x0, y0 = ring[0]
                    path_cmds.append(['M', x0 * scale_x, y0 * scale_y])
                    for pt in ring[1:]:
                        x, y = pt
                        path_cmds.append(['L', x * scale_x, y * scale_y])
                    path_cmds.append(['z'])
                obj = {
                    "type": "path",
                    "path": path_cmds,
                    "fill": "rgba(255, 165, 0, 0.3)",
                    "stroke": "#000000",
                    "strokeWidth": 2,
                    "opacity": 1,
                    "closed": True
                }
                objects.append(obj)
    return {"version": "4.4.0", "objects": objects}

def save_drawn_rois(objects, labels, scale_x, scale_y, image_path, geojson_dir):
    features = []
    for obj, label in zip(objects, labels):
        if obj["type"] == "path":
            norm_obj = normalize_fabric_path_to_absolute(obj)
            path_data = norm_obj["path"]
            coords = []
            for cmd in path_data:
                if cmd[0] in ['M', 'L']:
                    coords.append([cmd[1] * scale_x, cmd[2] * scale_y])
            if coords:
                coords.append(coords[0])
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"label": label}
            }
            features.append(feature)
        elif obj["type"] == "rect":
            # Apply object scaling if present
            obj_scale_x = obj.get("scaleX", 1)
            obj_scale_y = obj.get("scaleY", 1)
            
            left = obj["left"] * scale_x
            top = obj["top"] * scale_y
            width = obj["width"] * obj_scale_x * scale_x
            height = obj["height"] * obj_scale_y * scale_y
            coords = [[
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height],
                [left, top]
            ]]
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {"label": label}
            }
            features.append(feature)
            
    feature_collection = {"type": "FeatureCollection", "features": features}
    stem = image_path.stem
    if stem.endswith('.ome'): stem = stem[:-4]
    if stem.endswith('_merged'): stem = stem[:-7]
    filename = f"{stem}_roi_bounding_boxes_coords.geojson"
    save_path = Path(geojson_dir) / filename
    with open(save_path, 'w') as f:
        json.dump(feature_collection, f)
    
    st.success(f"Saved {len(features)} ROIs to {save_path}")
    
    # Debug: Print bounds of saved ROIs
    # for i, feature in enumerate(features):
    #     try:
    #         geom = shape(feature['geometry'])
    #         st.write(f"ROI {i+1} Bounds: {geom.bounds}")
    #     except Exception:
    #         pass
            
    return str(save_path)


def _seg_mask_filename(image_path: Path) -> str:
    """Build a GeoJSON filename for segmentation masks.

    Parameters
    ----------
    image_path:
        Input image path.

    Returns
    -------
    str
        File name ending in `_seg.geojson`.
    """
    stem = image_path.stem
    if stem.endswith('.ome'):
        stem = stem[:-4]
    if stem.endswith('_merged'):
        stem = stem[:-7]
    return f"{stem}_roi_bounding_boxes_coords_seg.geojson"


def normalize_fabric_path_to_absolute(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize Fabric path object to absolute canvas pixel coordinates with left=0, top=0, scale=1."""
    if not isinstance(obj, dict) or obj.get("type") != "path":
        return obj
    path_cmds = obj.get("path", [])
    if not path_cmds:
        return obj

    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    scale_x_obj = float(obj.get("scaleX", 1.0))
    scale_y_obj = float(obj.get("scaleY", 1.0))

    # Collect all X and Y coordinates in command list to find min_cmd_x and min_cmd_y
    cmd_xs = []
    cmd_ys = []
    for cmd in path_cmds:
        if isinstance(cmd, list) and len(cmd) >= 3:
            c_type = cmd[0]
            if c_type in ("M", "L"):
                cmd_xs.append(float(cmd[1]))
                cmd_ys.append(float(cmd[2]))
            elif c_type == "Q" and len(cmd) >= 5:
                cmd_xs.extend([float(cmd[1]), float(cmd[3])])
                cmd_ys.extend([float(cmd[2]), float(cmd[4])])
            elif c_type == "C" and len(cmd) >= 7:
                cmd_xs.extend([float(cmd[1]), float(cmd[3]), float(cmd[5])])
                cmd_ys.extend([float(cmd[2]), float(cmd[4]), float(cmd[6])])

    if not cmd_xs or not cmd_ys:
        return obj

    min_cmd_x = min(cmd_xs)
    min_cmd_y = min(cmd_ys)

    path_offset = obj.get("pathOffset")
    pox = float(path_offset.get("x", 0.0)) if isinstance(path_offset, dict) else 0.0
    poy = float(path_offset.get("y", 0.0)) if isinstance(path_offset, dict) else 0.0

    if abs(left) > 1e-5 or abs(top) > 1e-5:
        def x_transform(x_val: float) -> float:
            return left + (x_val - min_cmd_x) * scale_x_obj
        def y_transform(y_val: float) -> float:
            return top + (y_val - min_cmd_y) * scale_y_obj
    elif abs(pox) > 1e-5 or abs(poy) > 1e-5:
        def x_transform(x_val: float) -> float:
            return x_val * scale_x_obj + pox
        def y_transform(y_val: float) -> float:
            return y_val * scale_y_obj + poy
    else:
        def x_transform(x_val: float) -> float:
            return x_val * scale_x_obj
        def y_transform(y_val: float) -> float:
            return y_val * scale_y_obj

    new_path = []
    for cmd in path_cmds:
        if isinstance(cmd, list) and len(cmd) >= 1:
            c_type = cmd[0]
            if c_type in ("M", "L") and len(cmd) >= 3:
                new_path.append([c_type, float(x_transform(float(cmd[1]))), float(y_transform(float(cmd[2])))])
            elif c_type == "Q" and len(cmd) >= 5:
                cx = x_transform(float(cmd[1]))
                cy = y_transform(float(cmd[2]))
                tx = x_transform(float(cmd[3]))
                ty = y_transform(float(cmd[4]))
                new_path.append([c_type, float(cx), float(cy), float(tx), float(ty)])
            elif c_type == "C" and len(cmd) >= 7:
                c1x = x_transform(float(cmd[1]))
                c1y = y_transform(float(cmd[2]))
                c2x = x_transform(float(cmd[3]))
                c2y = y_transform(float(cmd[4]))
                tx = x_transform(float(cmd[5]))
                ty = y_transform(float(cmd[6]))
                new_path.append([c_type, float(c1x), float(c1y), float(c2x), float(c2y), float(tx), float(ty)])
            elif c_type == "z":
                new_path.append(["z"])

    normalized = dict(obj)
    normalized["path"] = new_path
    normalized["left"] = 0
    normalized["top"] = 0
    normalized["scaleX"] = 1.0
    normalized["scaleY"] = 1.0
    normalized.pop("pathOffset", None)
    return normalized


def obj_to_shapely_polygon(obj, scale_x, scale_y):
    """Helper to convert a Fabric.js canvas object to a Shapely Polygon (with full transformation matrix and hole support)."""
    sx = float(scale_x) if float(scale_x) > 0 else 1.0
    sy = float(scale_y) if float(scale_y) > 0 else 1.0

    if isinstance(obj, dict) and obj.get("type") == "path":
        norm_obj = normalize_fabric_path_to_absolute(obj)
        path_data = norm_obj.get("path", [])

        def _transform_pt(px: float, py: float) -> list[float]:
            return [px * sx, py * sy]

        rings = []
        curr_ring = []
        for cmd in path_data:
            if isinstance(cmd, list) and len(cmd) >= 3:
                c_type = cmd[0]
                if c_type == 'M':
                    if len(curr_ring) >= 3:
                        rings.append(curr_ring)
                    curr_ring = [_transform_pt(float(cmd[1]), float(cmd[2]))]
                elif c_type == 'L':
                    curr_ring.append(_transform_pt(float(cmd[1]), float(cmd[2])))
                elif c_type == 'Q' and len(cmd) >= 5:
                    curr_ring.append(_transform_pt(float(cmd[3]), float(cmd[4])))
                elif c_type == 'C' and len(cmd) >= 7:
                    curr_ring.append(_transform_pt(float(cmd[5]), float(cmd[6])))
        if len(curr_ring) >= 3:
            rings.append(curr_ring)

        if not rings:
            return None
        shell = rings[0]
        if shell[0] != shell[-1]:
            shell.append(shell[0])
        holes = []
        for r in rings[1:]:
            if r[0] != r[-1]:
                r.append(r[0])
            holes.append(r)
        try:
            p = Polygon(shell, holes=holes if holes else None)
            return p if p.is_valid else p.buffer(0)
        except:
            return None
    elif obj.get("type") == "rect":
        obj_scale_x = obj.get("scaleX", 1.0)
        obj_scale_y = obj.get("scaleY", 1.0)
        left = obj["left"] * sx
        top = obj["top"] * sy
        width = obj["width"] * obj_scale_x * sx
        height = obj["height"] * obj_scale_y * sy
        coords = [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
            [left, top]
        ]
        try:
            p = Polygon(coords)
            return p if p.is_valid else p.buffer(0)
        except:
            return None
    elif obj.get("type") == "polygon":
        points = obj.get("points", [])
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        scale_x_obj = obj.get("scaleX", 1.0)
        scale_y_obj = obj.get("scaleY", 1.0)
        path_offset = obj.get("pathOffset", {"x": 0, "y": 0})
        ox = path_offset.get("x", 0)
        oy = path_offset.get("y", 0)
        
        coords = []
        for p in points:
            cx = (p["x"] - ox) * scale_x_obj + left
            cy = (p["y"] - oy) * scale_y_obj + top
            coords.append([cx * sx, cy * sy])
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            p = Polygon(coords)
            return p if p.is_valid else p.buffer(0)
        except:
            return None
    return None


def save_drawn_segmentation_mask(
    objects: list[dict[str, Any]],
    scale_x: float,
    scale_y: float,
    image_path: Path,
    geojson_dir: str | Path,
) -> str:
    """Save drawn segmentation mask polygons to GeoJSON.
    Subtracts exclusion (removal) polygons from inclusion (mask) polygons.
    """
    from shapely.ops import unary_union
    from shapely.geometry import shape, Polygon, MultiPolygon, mapping

    out_dir = Path(geojson_dir) / "new_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    inclusion_shapes = []
    exclusion_shapes = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        poly = obj_to_shapely_polygon(obj, scale_x, scale_y)
        if not poly or poly.is_empty:
            continue
            
        # Distinguish inclusion vs exclusion by color/stroke
        is_removal = (obj.get("fill") == "rgba(255, 0, 0, 0.3)" or obj.get("stroke") == "#ff0000")
        if is_removal:
            exclusion_shapes.append(poly)
        else:
            inclusion_shapes.append(poly)

    features: list[dict[str, Any]] = []
    if inclusion_shapes:
        inclusion_union = unary_union(inclusion_shapes)
        if exclusion_shapes:
            exclusion_union = unary_union(exclusion_shapes)
            subtracted = inclusion_union.difference(exclusion_union)
        else:
            subtracted = inclusion_union

        if not subtracted.is_empty:
            if isinstance(subtracted, Polygon):
                features.append({
                    "type": "Feature",
                    "geometry": mapping(subtracted),
                    "properties": {"label": "seg"},
                })
            elif isinstance(subtracted, MultiPolygon):
                for geom in subtracted.geoms:
                    if not geom.is_empty:
                        features.append({
                            "type": "Feature",
                            "geometry": mapping(geom),
                            "properties": {"label": "seg"},
                        })

    feature_collection = {"type": "FeatureCollection", "features": features}
    save_path = out_dir / _seg_mask_filename(image_path)
    with open(save_path, 'w') as f:
        json.dump(feature_collection, f)

    st.success(f"Saved {len(features)} subtracted segmentation mask polygons to {save_path}")
    return str(save_path)

@st.cache_data
def get_cached_channel_info(file_path: str, mtime: float = 0.0):
    """
    Get channel metadata from file. Cached to avoid expensive OME-TIFF scans.
    """
    return get_channel_info(file_path)

@st.cache_data
def get_cached_channel_image(file_path, channel_index, mtime: float = 0.0):
    """
    Load a single channel image from disk, downsampled for preview.
    Cached by Streamlit to avoid re-reading.
    """
    try:
        with tifffile.TiffFile(file_path) as tif:
            p0 = tif.pages[0]
            
            # Case 1: Multiple pages, one per channel
            if len(tif.pages) > 1 and p0.ndim == 2:
                h, w = p0.shape
                scale = max(h, w) // 1000
                if channel_index < len(tif.pages):
                    page = tif.pages[channel_index]
                    d = page.asarray()
                    if scale > 1:
                        d = d[::scale, ::scale]
                    return d, (h, w)
            
            # Case 2: Single page with (C, H, W) or (H, W, C)
            elif p0.ndim == 3:
                series = tif.series[0]
                data = series.asarray()
                data = np.squeeze(data)
                
                # Normalize dims
                if data.ndim == 3:
                    # Heuristic: The smallest dimension is likely the channel dimension
                    if data.shape[2] < data.shape[0] and data.shape[2] < data.shape[1]:
                         data = np.moveaxis(data, 2, 0)
                    elif data.shape[1] < data.shape[0] and data.shape[1] < data.shape[2]:
                         data = np.moveaxis(data, 1, 0)
                
                full_shape = data.shape[-2:]
                scale = max(full_shape) // 1000
                
                if channel_index < data.shape[0]:
                    ch_data = data[channel_index]
                    if scale > 1:
                        ch_data = ch_data[::scale, ::scale]
                    return ch_data, full_shape

    except Exception as e:
        print(f"Error loading channel {channel_index}: {e}")
    
    return None, (0, 0)

# Session-state caches that hold rendered preview artefacts. All of them are
# keyed by image path but none of them carry the file's timestamp, so they have
# to be dropped explicitly whenever the preview should be rebuilt from disk.
PREVIEW_CACHE_STATE_KEYS = (
    "tab4_seg_canvas_objects",
    "tab4_seg_canvas_reset",
    "tab4_seg_canvas_bg_cache",
    "tab4_seg_last_scale",
)

# Per-image widget state is keyed by file *name*, not by full path, so two
# directories holding a file of the same name would otherwise share channel
# picks, drawing tools and histogram ranges.
PREVIEW_WIDGET_KEY_PREFIXES = (
    "tab4_target_type_", "tab4_drawing_tool_", "tab4_tissue_source_",
    "tab4_auto_det_mode_", "tab4_he_auto_det_mode_", "tab4_auto_hist_log_",
    "tab4_canvas_ignore_sync_",
    "seg_tis_chk_", "seg_nuc_chk_", "seg_mem_chk_",
    "_last_z_", "_last_single_",
)

PREVIEW_BOUND_DIR_KEY = "_tab4_preview_bound_dir"

# --------------------------------------------------------------------------
# Batch stop control
# --------------------------------------------------------------------------
# The batch runs synchronously inside Streamlit's script thread, so while
# Cellpose is working the session cannot process a widget click and cannot
# update session state. Ctrl+C in the terminal does not help either: it goes to
# the Streamlit server's main thread, not to the worker running the batch.
#
# A file on disk is therefore the only channel that reaches a running batch. The
# Stop button writes the sentinel, and the batch loop checks for it between
# images and stops cleanly, keeping everything already written.
STOP_SENTINEL_NAME = ".spacemap_stop_segmentation"


def stop_sentinel_path(output_dir) -> Path:
    return Path(output_dir) / STOP_SENTINEL_NAME


def request_batch_stop(output_dir) -> bool:
    """Ask a running batch to stop after the image it is currently on."""
    try:
        path = stop_sentinel_path(output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(datetime.now().isoformat(), encoding="utf-8")
        print(f"[TAB5-SEGMENTATION] stop requested -> {path}", flush=True)
        return True
    except Exception as error:
        print(f"[TAB5-SEGMENTATION] could not request stop: {error}", flush=True)
        return False


def clear_batch_stop(output_dir) -> None:
    """Remove a stale sentinel so the next run is not cancelled immediately."""
    try:
        stop_sentinel_path(output_dir).unlink(missing_ok=True)
    except Exception:
        pass


def batch_stop_requested(output_dir) -> bool:
    try:
        return stop_sentinel_path(output_dir).exists()
    except Exception:
        return False


def clear_preview_caches(reason: str = "") -> None:
    """Drop every cached preview artefact so the editor re-reads from disk.

    Covers the two ``st.cache_data`` readers and the session-state caches that
    hold canvas backgrounds, drawn shapes and scale factors.
    """
    for cached in (get_cached_channel_info, get_cached_channel_image):
        try:
            cached.clear()
        except Exception:
            pass
    for key in PREVIEW_CACHE_STATE_KEYS:
        st.session_state.pop(key, None)
    for key in [
        name for name in list(st.session_state.keys())
        if isinstance(name, str) and name.startswith(PREVIEW_WIDGET_KEY_PREFIXES)
    ]:
        st.session_state.pop(key, None)
    if reason:
        print(f"[TAB5-SEGMENTATION] preview cache cleared ({reason})", flush=True)


def sync_preview_to_input_dir(input_dir: str, image_files: list) -> Optional[str]:
    """Keep the preview bound to the input directory that is actually selected.

    Two rules are enforced here, and both exist because the preview used to be
    able to show something that is not in the selected folder:

    * Changing the input directory drops every cached preview artefact, so a
      thumbnail, channel list or canvas background built from the previous
      folder can never be drawn against a file in the new one.
    * The active image is always one that exists in the *current* listing. The
      previous code kept ``tab4_active_img`` as long as it was truthy, so after
      switching directories the editor stayed on the old file.

    Returns the active image name, or None when the directory holds no images.
    """
    previous = st.session_state.get(PREVIEW_BOUND_DIR_KEY)
    if previous != input_dir:
        if previous is not None:
            clear_preview_caches(f"input directory changed to {input_dir}")
            st.session_state.pop("tab4_active_img", None)
        st.session_state[PREVIEW_BOUND_DIR_KEY] = input_dir

    active = st.session_state.get("tab4_active_img")
    if active not in image_files:
        active = image_files[0] if image_files else None
        if active is None:
            st.session_state.pop("tab4_active_img", None)
        else:
            st.session_state["tab4_active_img"] = active
    return active


# Reuse load_preview_channels from Tab 3 (or duplicate for independence)
# Duplicating to ensure Tab 4 is self-contained as requested
def load_preview_channels(file_path, indices_to_load):
    """
    Load specific channels for preview.
    Args:
        file_path: Path to image
        indices_to_load: List of channel indices to load
    Returns:
        dict: {index: numpy_array}
        tuple: full_shape
        bool: is_rgb
    """
    loaded_data = {}
    full_shape = (0, 0)
    is_rgb = False
    
    try:
        # Check for RGB first (single page RGB) - fast path, no caching needed usually or cache whole
        with tifffile.TiffFile(file_path) as tif:
            p0 = tif.pages[0]
            if p0.ndim == 3 and (p0.shape[-1] == 3 or p0.shape[0] == 3):
                full_shape = p0.shape
                is_rgb = True
                img = p0.asarray()
                if p0.shape[0] == 3:
                    img = np.moveaxis(img, 0, -1)
                
                # Subsample
                scale = max(img.shape) // 1000
                if scale > 1:
                    img = img[::scale, ::scale]
                
                # Normalize RGB
                if img.dtype != np.uint8:
                    p_min = np.nanmin(img)
                    p_max = np.nanpercentile(img, 99) 
                    if p_max > p_min:
                        img = np.clip((img - p_min) / (p_max - p_min) * 255, 0, 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                
                return {'rgb': img}, full_shape, True

        # Multi-channel logic using cached loader
        try:
            mtime = os.path.getmtime(file_path)
        except:
            mtime = 0.0
        for idx in indices_to_load:
            img, shape = get_cached_channel_image(file_path, idx, mtime=mtime)
            if img is not None:
                loaded_data[idx] = img
                full_shape = shape

    except Exception as e:
        print(f"Error load_preview_channels: {e}")
    return loaded_data, full_shape, is_rgb

def find_matched_mask_file(image_name: str, mask_save_dir: str) -> tuple[Optional[Path], bool]:
    """Search for a matched GeoJSON mask file for the given image.
    First checks `<mask_save_dir>/new_mask/<stem>_roi_bounding_boxes_coords_seg.geojson`.
    Then `<mask_save_dir>/new_mask/<stem>_roi_bounding_boxes_coords.geojson`.
    Then `<mask_save_dir>/<stem>_roi_bounding_boxes_coords_seg.geojson`.
    Then `<mask_save_dir>/<stem>_roi_bounding_boxes_coords.geojson`.
    
    Returns (path_to_file, is_new_mask).
    """
    stem = Path(image_name).stem
    if stem.endswith('.ome'): stem = stem[:-4]
    if stem.endswith('_merged'): stem = stem[:-7]
    
    base_dir = Path(mask_save_dir)
    new_mask_dir = base_dir / "new_mask"
    
    # Candidates in new_mask
    if new_mask_dir.exists():
        for filename in [f"{stem}_roi_bounding_boxes_coords_seg.geojson", f"{stem}_roi_bounding_boxes_coords.geojson"]:
            p = new_mask_dir / filename
            if p.exists():
                return p, True
                
    # Candidates in base_dir
    if base_dir.exists():
        for filename in [f"{stem}_roi_bounding_boxes_coords_seg.geojson", f"{stem}_roi_bounding_boxes_coords.geojson"]:
            p = base_dir / filename
            if p.exists():
                return p, False
                
    return None, False


HE_CHANNEL_ROLES = ("red", "green", "blue")


def get_he_channel_role(channel_name: str) -> str | None:
    """Return the H&E color role encoded by a registered channel name."""
    normalized = "_".join(
        part for part in re.split(r"[^a-z0-9]+", str(channel_name).lower()) if part
    )
    for role in HE_CHANNEL_ROLES:
        expected = f"he_{role}"
        if normalized == expected or normalized.endswith(f"_{expected}"):
            return role
    return None


def get_he_channel_indices(
    channel_names: list[str], selected_indices: Optional[list[int]] = None
) -> tuple[dict[str, int], list[str]]:
    """Resolve selected or available HE_Red/Green/Blue channel indices."""
    allowed_indices = set(selected_indices) if selected_indices is not None else set(range(len(channel_names)))
    role_indices: dict[str, int] = {}
    for index, channel_name in enumerate(channel_names):
        if index not in allowed_indices:
            continue
        role = get_he_channel_role(channel_name)
        if role is not None and role not in role_indices:
            role_indices[role] = index
    missing = [f"HE_{role.title()}" for role in HE_CHANNEL_ROLES if role not in role_indices]
    return role_indices, missing


def build_he_tissue_mask(
    he_planes: dict[str, np.ndarray],
    *,
    closing_radius: int = 5,
    hole_area: int = 3000,
    min_object_area: int = 3000,
) -> np.ndarray:
    """Build a refined tissue mask from registered H&E RGB layers.

    This follows the validated reference workflow: average H&E luminance,
    invert it so tissue is bright, apply Otsu thresholding, then close gaps and
    remove small holes and fragments.
    """
    missing = [role for role in HE_CHANNEL_ROLES if role not in he_planes]
    if missing:
        raise ValueError(f"Missing H&E layer(s): {', '.join(f'HE_{role.title()}' for role in missing)}")

    planes = [np.asarray(he_planes[role], dtype=np.float32) for role in HE_CHANNEL_ROLES]
    if any(plane.shape != planes[0].shape for plane in planes[1:]):
        raise ValueError("H&E layers have different image shapes and cannot be combined.")

    luminance = np.mean(np.stack(planes, axis=0), axis=0)
    inverted_luminance = float(np.nanmax(luminance)) - luminance
    if not np.isfinite(inverted_luminance).any() or float(np.nanmax(inverted_luminance)) <= float(np.nanmin(inverted_luminance)):
        raise ValueError("H&E layers have no usable intensity variation for tissue detection.")

    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

    mask = inverted_luminance > threshold_otsu(inverted_luminance)
    if closing_radius > 0:
        mask = binary_closing(mask, disk(int(closing_radius)))
    if hole_area > 0:
        mask = remove_small_holes(mask, area_threshold=int(hole_area))
    if min_object_area > 0:
        mask = remove_small_objects(mask, min_size=int(min_object_area))
    return np.asarray(mask, dtype=bool)


def get_preview_canvas_dimensions(mask_shape: tuple[int, int], display_width: int = 700) -> tuple[int, int]:
    """Return the display dimensions used by the segmentation canvas."""
    mask_height, mask_width = mask_shape
    if mask_height <= 0 or mask_width <= 0:
        return 1, 1
    canvas_width = min(mask_width, display_width)
    canvas_height = max(1, int(mask_height * (canvas_width / mask_width)))
    return canvas_width, canvas_height


def tissue_mask_to_canvas_objects(
    tissue_mask: np.ndarray,
    *,
    buffer_size: float = 0.0,
    min_area: float = 100.0,
    canvas_width: Optional[int] = None,
    canvas_height: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Convert a tissue mask into editable Fabric paths aligned to the canvas."""
    import cv2
    from shapely.geometry import MultiPolygon

    mask_uint8 = np.asarray(tissue_mask, dtype=np.uint8) * 255
    source_height, source_width = mask_uint8.shape[:2]
    default_canvas_width, default_canvas_height = get_preview_canvas_dimensions(
        (source_height, source_width)
    )
    target_width = canvas_width or default_canvas_width
    target_height = canvas_height or default_canvas_height
    scale_to_canvas_x = float(target_width) / float(source_width)
    scale_to_canvas_y = float(target_height) / float(source_height)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: list[dict[str, Any]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = contour.squeeze(axis=1)
        if len(points) < 3:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if buffer_size > 0:
            polygon = polygon.buffer(buffer_size)
        if polygon.is_empty:
            continue
        polygons = [polygon] if isinstance(polygon, Polygon) else list(polygon.geoms) if isinstance(polygon, MultiPolygon) else []
        for item in polygons:
            if item.is_empty or item.area < min_area:
                continue
            exterior = list(item.exterior.coords)
            if len(exterior) < 3:
                continue
            path = [["M", float(exterior[0][0] * scale_to_canvas_x), float(exterior[0][1] * scale_to_canvas_y)]]
            path.extend([["L", float(x * scale_to_canvas_x), float(y * scale_to_canvas_y)] for x, y in exterior[1:]])
            path.append(["z"])
            objects.append({
                "type": "path",
                "path": path,
                "fill": "rgba(255, 165, 0, 0.3)",
                "stroke": "#000000",
                "strokeWidth": 2,
                "opacity": 1,
                "closed": True,
            })
    return objects


def run_he_tissue_detection(
    image_path: str,
    he_indices: dict[str, int],
    *,
    closing_radius: int = 5,
    hole_area: int = 3000,
    min_object_area: int = 3000,
    buffer_size: float = 0.0,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Detect tissue from one image's selected H&E RGB channels."""
    try:
        mtime = os.path.getmtime(image_path)
    except OSError:
        mtime = 0.0

    he_planes: dict[str, np.ndarray] = {}
    for role, index in he_indices.items():
        plane, _ = get_cached_channel_image(image_path, index, mtime=mtime)
        if plane is not None:
            he_planes[role] = plane
    tissue_mask = build_he_tissue_mask(
        he_planes,
        closing_radius=closing_radius,
        hole_area=hole_area,
        min_object_area=min_object_area,
    )
    canvas_width, canvas_height = get_preview_canvas_dimensions(tissue_mask.shape)
    return tissue_mask_to_canvas_objects(
        tissue_mask,
        buffer_size=buffer_size,
        min_area=min_object_area,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    ), tissue_mask


def validate_batch_he_channels(
    image_files: list[str], input_dir: str
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    """Resolve H&E channels for every image and report images missing RGB layers."""
    resolved: dict[str, dict[str, int]] = {}
    missing_by_image: dict[str, list[str]] = {}
    for image_name in image_files:
        image_path = Path(input_dir) / image_name
        try:
            mtime = os.path.getmtime(image_path)
            channel_names, _, _, _ = get_cached_channel_info(str(image_path), mtime=mtime)
        except Exception:
            missing_by_image[image_name] = ["channel metadata"]
            continue
        he_indices, missing = get_he_channel_indices(channel_names)
        if missing:
            missing_by_image[image_name] = missing
        else:
            resolved[image_name] = he_indices
    return resolved, missing_by_image


def shapely_to_fabric_path(
    polygon: Any, scale_x: float, scale_y: float, removal: bool = False
) -> dict[str, Any]:
    """Convert a Shapely polygon back into a valid Fabric.js Path object with correct left, top, pathOffset geometry."""
    sx = scale_x if scale_x > 0 else 1.0
    sy = scale_y if scale_y > 0 else 1.0

    exterior = list(polygon.exterior.coords)
    canvas_pts = [[x / sx, y / sy] for x, y in exterior]
    xs = [p[0] for p in canvas_pts]
    ys = [p[1] for p in canvas_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    ox = (min_x + max_x) / 2.0
    oy = (min_y + max_y) / 2.0

    path = [["M", float(canvas_pts[0][0] - ox), float(canvas_pts[0][1] - oy)]]
    path.extend([["L", float(p[0] - ox), float(p[1] - oy)] for p in canvas_pts[1:]])
    path.append(["z"])

    for interior in getattr(polygon, "interiors", []):
        coords = list(interior.coords)
        if len(coords) >= 3:
            path.append(["M", float(coords[0][0] / sx - ox), float(coords[0][1] / sy - oy)])
            path.extend([["L", float(x / sx - ox), float(y / sy - oy)] for x, y in coords[1:]])
            path.append(["z"])

    return {
        "type": "path",
        "path": path,
        "left": float(min_x),
        "top": float(min_y),
        "scaleX": 1.0,
        "scaleY": 1.0,
        "pathOffset": {"x": float(ox), "y": float(oy)},
        "fill": "rgba(255, 0, 0, 0.3)" if removal else "rgba(255, 165, 0, 0.3)",
        "stroke": "#ff0000" if removal else "#000000",
        "strokeWidth": 2,
        "opacity": 1,
        "closed": True,
    }


def shapely_shapes_to_fabric_objects(
    shapes: list[Any], scale_x: float, scale_y: float, removal: bool
) -> list[dict[str, Any]]:
    if not shapes:
        return []
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    merged = unary_union(shapes)
    polygons = [merged] if isinstance(merged, Polygon) else list(merged.geoms) if isinstance(merged, MultiPolygon) else []
    converted: list[dict[str, Any]] = []
    for p in polygons:
        if not p.is_empty and len(p.exterior.coords) >= 3:
            converted.append(shapely_to_fabric_path(p, scale_x, scale_y, removal=removal))
    return converted


def merge_canvas_polygons(
    objects: list[dict[str, Any]], scale_x: float, scale_y: float
) -> list[dict[str, Any]]:
    """Union overlapping inclusion and removal polygons while preserving their roles."""
    inclusion_shapes = []
    exclusion_shapes = []
    for object_data in objects:
        if not isinstance(object_data, dict):
            continue
        polygon = obj_to_shapely_polygon(object_data, scale_x, scale_y)
        if polygon is None or polygon.is_empty:
            continue
        is_removal = (
            object_data.get("fill") == "rgba(255, 0, 0, 0.3)"
            or object_data.get("stroke") == "#ff0000"
        )
        (exclusion_shapes if is_removal else inclusion_shapes).append(polygon)

    return shapely_shapes_to_fabric_objects(inclusion_shapes, scale_x, scale_y, removal=False) + \
           shapely_shapes_to_fabric_objects(exclusion_shapes, scale_x, scale_y, removal=True)


def subtract_canvas_polygons(
    objects: list[dict[str, Any]], scale_x: float, scale_y: float
) -> list[dict[str, Any]]:
    """Subtract exclusion (removal) polygons from inclusion polygons and return updated canvas objects."""
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    sx = scale_x if scale_x > 0 else 1.0
    sy = scale_y if scale_y > 0 else 1.0

    inclusion_shapes = []
    exclusion_shapes = []
    for object_data in objects:
        if not isinstance(object_data, dict):
            continue
        poly = obj_to_shapely_polygon(object_data, sx, sy)
        if poly is None or poly.is_empty:
            continue
        is_removal = (
            object_data.get("fill") == "rgba(255, 0, 0, 0.3)"
            or object_data.get("stroke") == "#ff0000"
        )
        if is_removal:
            exclusion_shapes.append(poly)
        else:
            inclusion_shapes.append(poly)

    if not inclusion_shapes:
        return []

    inc_union = unary_union(inclusion_shapes)
    if exclusion_shapes:
        exc_union = unary_union(exclusion_shapes)
        subtracted = inc_union.difference(exc_union)
    else:
        subtracted = inc_union

    if subtracted.is_empty:
        return []

    polygons = (
        [subtracted]
        if isinstance(subtracted, Polygon)
        else list(subtracted.geoms)
        if isinstance(subtracted, MultiPolygon)
        else []
    )
    converted: list[dict[str, Any]] = []
    for polygon in polygons:
        if not polygon.is_empty and len(polygon.exterior.coords) >= 3:
            converted.append(shapely_to_fabric_path(polygon, sx, sy, removal=False))
    return converted



def run_auto_tissue_detection(image_path, channel_indices, threshold_min=20, threshold_max=255, merge_radius=15, buffer_size=10, min_area=100, use_zscore=False):
    """Detect tissue boundaries on selected channel(s) and generate Fabric.js path objects."""
    import cv2
    from shapely.geometry import Polygon, MultiPolygon
    
    try:
        mtime = os.path.getmtime(image_path)
    except:
        mtime = 0.0
        
    loaded_imgs = []
    for idx in channel_indices:
        img_data, _ = get_cached_channel_image(image_path, idx, mtime=mtime)
        if img_data is not None:
            loaded_imgs.append(img_data)
            
    if not loaded_imgs:
        return []
        
    if len(loaded_imgs) == 1:
        combined_img = loaded_imgs[0]
    else:
        combined_img = np.maximum.reduce(loaded_imgs)
        
    combined_img = combined_img.astype(np.float32)
    if use_zscore:
        mean = np.mean(combined_img)
        std = np.std(combined_img)
        z_img = (combined_img - mean) / (std + 1e-6)
        thresh = np.zeros(z_img.shape, dtype=np.uint8)
        thresh[(z_img >= threshold_min) & (z_img <= threshold_max)] = 255
    else:
        thresh = np.zeros(combined_img.shape, dtype=np.uint8)
        thresh[(combined_img >= threshold_min) & (combined_img <= threshold_max)] = 255
    
    if merge_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_radius * 2 + 1, merge_radius * 2 + 1))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_polygons = []
    
    thumb_h, thumb_w = combined_img.shape
    display_width = 700
    canvas_w = thumb_w
    canvas_h = thumb_h
    if thumb_w > display_width:
        ratio = display_width / thumb_w
        canvas_w = display_width
        canvas_h = int(thumb_h * ratio)
        
    scale_to_canvas_x = canvas_w / thumb_w
    scale_to_canvas_y = canvas_h / thumb_h
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        pts = cnt.squeeze(axis=1)
        if len(pts) < 3:
            continue
            
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
            
        if not poly.is_valid:
            continue
            
        if buffer_size > 0:
            poly = poly.buffer(buffer_size)
            
        if not poly.is_valid or poly.is_empty:
            continue
            
        if isinstance(poly, Polygon):
            polys = [poly]
        elif isinstance(poly, MultiPolygon):
            polys = list(poly.geoms)
        else:
            polys = []
            
        for p in polys:
            ext_coords = list(p.exterior.coords)
            canvas_pts = []
            for tx, ty in ext_coords:
                canvas_pts.append((tx * scale_to_canvas_x, ty * scale_to_canvas_y))
            detected_polygons.append(canvas_pts)
            
    objects = []
    for pts in detected_polygons:
        if len(pts) < 3:
            continue
        path_cmds = []
        x0, y0 = pts[0]
        path_cmds.append(['M', x0, y0])
        for x, y in pts[1:]:
            path_cmds.append(['L', x, y])
        path_cmds.append(['z'])
        
        obj = {
            "type": "path",
            "path": path_cmds,
            "fill": "rgba(255, 165, 0, 0.3)",
            "stroke": "#000000",
            "strokeWidth": 2,
            "opacity": 1,
            "closed": True
        }
        objects.append(obj)
        
    return objects

def calculate_batch_zscore_data(image_files, input_dir, tis_indices):
    """Calculate downsampled Z-scored images and their histograms for all files."""
    batch_data = {}
    all_hists = []
    
    # We define Z-score bins from -3 to 8
    bins = np.linspace(-3, 8, 100)
    
    for img_name in image_files:
        img_path = Path(input_dir) / img_name
        try:
            mtime = os.path.getmtime(str(img_path))
        except:
            mtime = 0.0
        loaded_imgs = []
        for idx in tis_indices:
            img_data, _ = get_cached_channel_image(str(img_path), idx, mtime=mtime)
            if img_data is not None:
                loaded_imgs.append(img_data)
        if not loaded_imgs:
            continue
            
        if len(loaded_imgs) == 1:
            combined = loaded_imgs[0]
        else:
            combined = np.maximum.reduce(loaded_imgs)
            
        combined = combined.astype(np.float32)
        mean = np.mean(combined)
        std = np.std(combined)
        
        # Z-score normalization
        z_img = (combined - mean) / (std + 1e-6)
        
        # Compute histogram of Z-scores
        hist_vals, _ = np.histogram(z_img.ravel(), bins=bins)
        all_hists.append(hist_vals)
        
        batch_data[img_name] = {
            'z_img': z_img,
            'mean': mean,
            'std': std
        }
        
    if all_hists:
        avg_hist = np.mean(all_hists, axis=0)
    else:
        avg_hist = np.zeros(len(bins) - 1)
        
    return batch_data, avg_hist, bins

def run_batch_tissue_detection(image_files, input_dir, mask_save_dir, tis_indices, z_min, z_max, merge_radius, buffer_size, min_area):
    """Detect tissue boundaries on all images using Z-scored thresholding and save them."""
    import cv2
    from shapely.geometry import Polygon, MultiPolygon
    
    success_count = 0
    
    for img_name in image_files:
        img_path = Path(input_dir) / img_name
        try:
            mtime = os.path.getmtime(str(img_path))
        except:
            mtime = 0.0
        loaded_imgs = []
        for idx in tis_indices:
            img_data, full_shape = get_cached_channel_image(str(img_path), idx, mtime=mtime)
            if img_data is not None:
                loaded_imgs.append(img_data)
        if not loaded_imgs:
            continue
            
        if len(loaded_imgs) == 1:
            combined_img = loaded_imgs[0]
        else:
            combined_img = np.maximum.reduce(loaded_imgs)
            
        combined_img = combined_img.astype(np.float32)
        mean = np.mean(combined_img)
        std = np.std(combined_img)
        z_img = (combined_img - mean) / (std + 1e-6)
        
        # Threshold on Z-score
        thresh = np.zeros(z_img.shape, dtype=np.uint8)
        thresh[(z_img >= z_min) & (z_img <= z_max)] = 255
        
        if merge_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_radius * 2 + 1, merge_radius * 2 + 1))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_polygons = []
        thumb_h, thumb_w = combined_img.shape
        display_width = 700
        canvas_w = thumb_w
        canvas_h = thumb_h
        if thumb_w > display_width:
            ratio = display_width / thumb_w
            canvas_w = display_width
            canvas_h = int(thumb_h * ratio)
            
        scale_to_canvas_x = canvas_w / thumb_w
        scale_to_canvas_y = canvas_h / thumb_h
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
                
            pts = cnt.squeeze(axis=1)
            if len(pts) < 3:
                continue
                
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
                
            if not poly.is_valid:
                continue
                
            if buffer_size > 0:
                poly = poly.buffer(buffer_size)
                
            if not poly.is_valid or poly.is_empty:
                continue
                
            if isinstance(poly, Polygon):
                polys = [poly]
            elif isinstance(poly, MultiPolygon):
                polys = list(poly.geoms)
            else:
                polys = []
                
            for p in polys:
                ext_coords = list(p.exterior.coords)
                canvas_pts = []
                for tx, ty in ext_coords:
                    canvas_pts.append((tx * scale_to_canvas_x, ty * scale_to_canvas_y))
                detected_polygons.append(canvas_pts)
                
        detected_objs = []
        for pts in detected_polygons:
            if len(pts) < 3:
                continue
            path_cmds = []
            x0, y0 = pts[0]
            path_cmds.append(['M', x0, y0])
            for x, y in pts[1:]:
                path_cmds.append(['L', x, y])
            path_cmds.append(['z'])
            
            obj = {
                "type": "path",
                "path": path_cmds,
                "fill": "rgba(255, 165, 0, 0.3)",
                "stroke": "#000000",
                "strokeWidth": 2,
                "opacity": 1,
                "closed": True
            }
            detected_objs.append(obj)
            
        # Save to session state
        st.session_state.setdefault('tab4_seg_canvas_objects', {})[str(img_path)] = detected_objs
        st.session_state.setdefault('tab4_seg_canvas_reset', {})[str(img_path)] = st.session_state.setdefault('tab4_seg_canvas_reset', {}).get(str(img_path), 0) + 1
        
        # Save to disk automatically
        full_h, full_w = full_shape[0], full_shape[1]
        if len(full_shape) == 3:
            if full_shape[0] < full_shape[1] and full_shape[0] < 100:
                full_h, full_w = full_shape[1], full_shape[2]
                
        scale_x = float(full_w) / float(canvas_w)
        scale_y = float(full_h) / float(canvas_h)
        
        save_drawn_segmentation_mask(
            objects=detected_objs,
            scale_x=scale_x,
            scale_y=scale_y,
            image_path=img_path,
            geojson_dir=mask_save_dir
        )
        
        success_count += 1
        
    return success_count


def run_batch_he_tissue_detection(
    image_files: list[str],
    input_dir: str,
    mask_save_dir: str,
    he_indices_by_image: dict[str, dict[str, int]],
    *,
    closing_radius: int = 5,
    hole_area: int = 3000,
    min_object_area: int = 3000,
    buffer_size: float = 0.0,
) -> tuple[int, dict[str, str]]:
    """Build and save H&E tissue masks for a validated batch of ROI images."""
    success_count = 0
    failures: dict[str, str] = {}
    for image_name in image_files:
        image_path = Path(input_dir) / image_name
        indices = he_indices_by_image.get(image_name)
        if not indices:
            failures[image_name] = "H&E channel mapping was not validated."
            continue
        try:
            mtime = os.path.getmtime(image_path)
            he_planes: dict[str, np.ndarray] = {}
            full_shape: tuple[int, ...] = (0, 0)
            for role, index in indices.items():
                plane, full_shape = get_cached_channel_image(str(image_path), index, mtime=mtime)
                if plane is None:
                    raise ValueError(f"Could not load HE_{role.title()}.")
                he_planes[role] = plane

            tissue_mask = build_he_tissue_mask(
                he_planes,
                closing_radius=closing_radius,
                hole_area=hole_area,
                min_object_area=min_object_area,
            )
            canvas_width, canvas_height = get_preview_canvas_dimensions(tissue_mask.shape)
            detected_objects = tissue_mask_to_canvas_objects(
                tissue_mask,
                buffer_size=buffer_size,
                min_area=min_object_area,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            if not detected_objects:
                raise ValueError("No tissue polygons were detected from the H&E layers.")

            full_height, full_width = full_shape[-2:]
            scale_x = float(full_width) / float(canvas_width)
            scale_y = float(full_height) / float(canvas_height)
            st.session_state.setdefault("tab4_seg_canvas_objects", {})[str(image_path)] = detected_objects
            st.session_state.setdefault("tab4_seg_canvas_reset", {})[str(image_path)] = (
                st.session_state.setdefault("tab4_seg_canvas_reset", {}).get(str(image_path), 0) + 1
            )
            save_drawn_segmentation_mask(
                objects=detected_objects,
                scale_x=scale_x,
                scale_y=scale_y,
                image_path=image_path,
                geojson_dir=mask_save_dir,
            )
            success_count += 1
        except Exception as error:
            failures[image_name] = str(error)
    return success_count, failures

def generate_summary_thumbnail(image_path, tis_indices, objects, scale_x, scale_y):
    """Generate a small PIL Image thumbnail with polygons overlaid."""
    if not tis_indices:
        return None
    try:
        mtime = os.path.getmtime(str(image_path))
    except:
        mtime = 0.0
    img_data, _ = get_cached_channel_image(str(image_path), tis_indices[0], mtime=mtime)
    if img_data is None:
        return None
        
    # Normalize img_data to 8-bit
    img_data = img_data.astype(np.float32)
    p99 = np.percentile(img_data, 99)
    p_min = np.min(img_data)
    if p99 > p_min:
        norm = np.clip((img_data - p_min) / (p99 - p_min) * 255, 0, 255).astype(np.uint8)
    else:
        norm = np.clip(img_data / (img_data.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        
    pil_img = Image.fromarray(norm).convert("RGB")
    
    # Draw polygons on it
    if objects:
        draw = ImageDraw.Draw(pil_img)
        canvas_width = pil_img.width
        canvas_height = pil_img.height
        display_width = 700
        if canvas_width > display_width:
            ratio = display_width / canvas_width
            canvas_width = display_width
            canvas_height = int(pil_img.height * ratio)
            
        s_x = pil_img.width / canvas_width
        s_y = pil_img.height / canvas_height
        
        # Check if there are removal shapes or a subtracted preview
        has_removal = any(
            isinstance(o, dict) and (o.get("fill") == "rgba(255, 0, 0, 0.3)" or o.get("stroke") == "#ff0000")
            for o in objects
        )
        if has_removal:
            try:
                subtracted_objs = subtract_canvas_polygons(objects, 1.0, 1.0)
                draw_objs = subtracted_objs if subtracted_objs else objects
            except:
                draw_objs = objects
        else:
            draw_objs = objects

        for obj in draw_objs:
            if isinstance(obj, dict) and obj.get('type') == 'path':
                norm_obj = normalize_fabric_path_to_absolute(obj)
                path_data = norm_obj.get('path', [])
                pts = []
                for cmd in path_data:
                    if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ('M', 'L'):
                        pts.append((float(cmd[1]) * s_x, float(cmd[2]) * s_y))
                if len(pts) >= 3:
                    is_removal = (obj.get("fill") == "rgba(255, 0, 0, 0.3)" or obj.get("stroke") == "#ff0000")
                    color = "red" if is_removal else "orange"
                    draw.polygon(pts, outline=color, width=2)
                    
    pil_img.thumbnail((150, 150))
    return pil_img

def _run_batch_segmentation_v2(to_proc, input_dir, mask_save_dir, output_dir, use_mask, create_log, model_type, custom_model_path, diameter, flow, tile_size, tile_overlap, cellpose_internal_tile_size, cellpose_batch_size, adaptive_gpu_memory, min_cellpose_internal_tile_size, min_tissue_fraction, tissue_crop_padding, gpu, save_geojson, save_npy, save_png, overwrite, use_annotation_mask=False,
                               cudnn_mode='auto', flow_device='auto'):
    """Process cellpose segmentation for checked images."""
    session_start_time = time.time()
    total_pairs = len(to_proc)
    record_process_event(
        "tab5",
        "Segmentation started",
        details={
            "images_selected": total_pairs,
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "model_type": model_type,
            "use_tissue_masks": use_mask,
            "outer_roi_tile_size": tile_size,
            "cellpose_internal_tile_size": cellpose_internal_tile_size,
            "adaptive_gpu_memory": adaptive_gpu_memory,
            "min_tissue_fraction": min_tissue_fraction,
            "tissue_crop_padding": tissue_crop_padding,
        },
    )
    
    # Progress widgets
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    status_container = st.status("Batch segmenting images...", expanded=True)
    
    log_dir = None
    if create_log:
        log_dir = create_seg_log_session_folder(output_dir)
        
    processed_count = 0
    failed_count = 0
    total_cells = 0
    is_first_log = True
    stopped_early = False

    # A sentinel left behind by an earlier cancelled run would otherwise stop
    # this one before it starts.
    clear_batch_stop(output_dir)

    try:
        for current_idx, p in enumerate(to_proc):
            if batch_stop_requested(output_dir):
                stopped_early = True
                remaining = total_pairs - current_idx
                status_container.write(
                    f"Stop requested — halting before image {current_idx + 1}/{total_pairs}. "
                    f"{remaining} image(s) not processed."
                )
                record_process_event(
                    "tab5", "Segmentation stopped by user", status="warning",
                    details={"processed": processed_count, "remaining": remaining},
                )
                break

            img_name = p['image']
            img_path = Path(input_dir) / img_name
            img_start_time = time.time()
            record_process_event(
                "tab5",
                "Segmentation image started",
                details={"image": img_name, "input": str(img_path), "input_metadata": get_segmentation_metadata_log(str(img_path))},
            )
            
            status_text.text(f"Processing image {current_idx + 1}/{total_pairs}: {img_name}")
            status_container.write(f"Segmenting {img_name}...")
            if gpu:
                status_container.write(
                    f"GPU memory safety: outer ROI tile {tile_size}px; Cellpose internal block {cellpose_internal_tile_size}px; batch size {cellpose_batch_size}."
                )
            if use_mask:
                status_container.write(
                    f"Mask-aware speed: skip tiles below {min_tissue_fraction:.3%} tissue coverage; crop tissue with {tissue_crop_padding}px padding."
                )
            
            # Channels Setup
            file_key = img_name.replace('.', '_').replace(' ', '_')
            nuclear_key = f"seg_nuclear_{file_key}"
            membrane_key = f"seg_membrane_{file_key}"
            
            try:
                mtime = os.path.getmtime(str(img_path))
            except:
                mtime = 0.0
            ch_names, _, _, _ = get_cached_channel_info(str(img_path), mtime=mtime)
            
            # Fallback to active image channel configuration if not defined or empty for current image
            active_img = st.session_state.get('tab4_active_img')
            if active_img:
                active_file_key = active_img.replace('.', '_').replace(' ', '_')
                active_nuc_key = f"seg_nuclear_{active_file_key}"
                active_mem_key = f"seg_membrane_{active_file_key}"
                
                current_nuc = st.session_state.get(nuclear_key)
                if (not current_nuc or not any(current_nuc.values())) and active_nuc_key in st.session_state:
                    st.session_state[nuclear_key] = st.session_state[active_nuc_key].copy()
                    
                current_mem = st.session_state.get(membrane_key)
                if (not current_mem or not any(current_mem.values())) and active_mem_key in st.session_state:
                    st.session_state[membrane_key] = st.session_state[active_mem_key].copy()

            nuc_ch_names = []
            mem_ch_names = []
            
            if nuclear_key in st.session_state:
                nuc_ch_names = [ch_names[int(i)] for i, sel in st.session_state[nuclear_key].items() if sel and int(i) < len(ch_names)]
            else:
                nuc_ch_names = [c for c in ch_names if 'dapi' in c.lower()]
                if not nuc_ch_names and ch_names:
                    nuc_ch_names = [ch_names[0]]
                    
            if membrane_key in st.session_state:
                mem_ch_names = [ch_names[int(i)] for i, sel in st.session_state[membrane_key].items() if sel and int(i) < len(ch_names)]
                
            dapi_ch = nuc_ch_names[0] if nuc_ch_names else 'DAPI'
            markers = mem_ch_names if mem_ch_names else None
            
            # Configuration
            seg_config = SegmentationConfig()
            seg_config.diameter = diameter
            seg_config.flow_threshold = flow
            seg_config.tile_size = tile_size
            seg_config.tile_overlap = tile_overlap
            seg_config.cellpose_internal_tile_size = cellpose_internal_tile_size
            seg_config.cellpose_batch_size = cellpose_batch_size
            seg_config.adaptive_gpu_memory = adaptive_gpu_memory
            seg_config.min_cellpose_internal_tile_size = min_cellpose_internal_tile_size
            seg_config.cudnn_mode = cudnn_mode
            seg_config.flow_threshold_device = flow_device
            seg_config.min_tissue_fraction = min_tissue_fraction
            seg_config.tissue_crop_padding = tissue_crop_padding
            seg_config.use_gpu = gpu
            seg_config.save_geojson = save_geojson
            seg_config.save_npy = save_npy
            seg_config.save_png = save_png
            
            if model_type == 'custom':
                seg_config.cellpose_model_path = custom_model_path or ""
                seg_config.cellpose_model_type = 'custom'
            else:
                seg_config.cellpose_model_type = model_type
                
            # Mask and spatial annotation. The annotation route restricts Cellpose
            # to one class of the reviewed GeoJSON and reuses the same file to tag
            # the extracted cells, so no separate annotation pass is needed.
            spatial = spatial_ui.build_spatial_config(img_name)
            if use_annotation_mask:
                if spatial["geojson_path"] and spatial["mask_classes"]:
                    seg_config.use_input_geojson = True
                    seg_config.input_geojson_path = spatial["geojson_path"]
                    seg_config.mask_class_names = list(spatial["mask_classes"])
                    status_container.write(
                        f"Segmenting inside '{', '.join(spatial['mask_classes'])}' "
                        f"from {os.path.basename(spatial['geojson_path'])}"
                    )
                else:
                    seg_config.use_input_geojson = False
                    status_container.write(f"Warning: no annotation mask for {img_name}. Running without mask.")
            elif use_mask:
                mask_file, _ = find_matched_mask_file(img_name, mask_save_dir)
                if mask_file and mask_file.exists():
                    seg_config.use_input_geojson = True
                    seg_config.input_geojson_path = str(mask_file)
                    status_container.write(f"Using mask: {mask_file.name}")
                else:
                    seg_config.use_input_geojson = False
                    status_container.write(f"Warning: Mask enabled but not found for {img_name}. Running without mask.")
            else:
                seg_config.use_input_geojson = False

            if spatial["geojson_path"] and (spatial["rules"] or spatial["edge_config"]):
                seg_config.annotation_geojson_path = spatial["geojson_path"]
                seg_config.spatial_rules = spatial["rules"]
                seg_config.spatial_tissue_label = spatial["tissue_label"]
                seg_config.spatial_min_cells_per_bed_target = spatial[
                    "min_cells_per_bed_target"
                ]
                seg_config.spatial_require_clean_path = spatial["require_clean_path"]
                seg_config.spatial_edge_config = spatial["edge_config"]
                status_container.write(
                    f"Spatial annotation: {len(spatial['rules'])} annotation rule(s) from "
                    f"{os.path.basename(spatial['geojson_path'])}"
                )
                
            image_log_data = {
                'Image_Name': img_name,
                'Date': datetime.now().strftime("%Y-%m-%d"),
                'Time': datetime.now().strftime("%H:%M:%S"),
                'Input_Path': str(img_path),
                'Status': 'In Progress',
                'Processing_Time_Sec': 0,
                'Cells_Detected': 0,
                'Error_Message': '',
                'Nuclear_Channels': "; ".join(nuc_ch_names) if nuc_ch_names else 'None',
                'Membrane_Channels': "; ".join(mem_ch_names) if mem_ch_names else 'None',
                'Model_Type': model_type,
                'Diameter': diameter,
                'Flow_Threshold': flow,
                'Tile_Size': tile_size,
                'Cellpose_Internal_Tile_Size': cellpose_internal_tile_size,
                'Cellpose_Batch_Size': cellpose_batch_size,
                'Adaptive_GPU_Memory': adaptive_gpu_memory,
                'Min_Tissue_Fraction': min_tissue_fraction,
                'Tissue_Crop_Padding': tissue_crop_padding,
                'Use_Mask': use_mask,
                'Mask_Path': str(seg_config.input_geojson_path) if seg_config.use_input_geojson else 'N/A',
            }
            
            try:
                results = segment_image_cellpose(
                    ome_tiff_path=str(img_path),
                    output_dir=output_dir,
                    config=seg_config,
                    markers=markers,
                    dapi_channel=dapi_ch,
                    overwrite=overwrite,
                    progress_callback=None
                )
                
                elapsed = time.time() - img_start_time
                cells_count = len(results.get('features_df', [])) if results else 0
                
                image_log_data['Status'] = 'Success'
                image_log_data['Processing_Time_Sec'] = f"{elapsed:.2f}"
                image_log_data['Cells_Detected'] = cells_count
                if results and 'features_csv' in results:
                    image_log_data['Output_Features_CSV'] = results['features_csv']
                if results and 'geojson' in results:
                    image_log_data['Output_GeoJSON'] = results.get('geojson', '')
                    
                processed_count += 1
                total_cells += cells_count
                status_container.write(f"Success: Detected {cells_count} cells.")
                record_process_event(
                    "tab5",
                    "Segmentation image complete",
                    status="complete",
                    duration_s=elapsed,
                    details={
                        "image": img_name,
                        "cells_detected": cells_count,
                        "features_csv": results.get("features_csv", "") if results else "",
                        "geojson": results.get("geojson", "") if results else "",
                    },
                )
                
            except Exception as e:
                elapsed = time.time() - img_start_time
                image_log_data['Status'] = 'Failed'
                image_log_data['Processing_Time_Sec'] = f"{elapsed:.2f}"
                image_log_data['Error_Message'] = str(e)[:200]
                if "out of memory" in str(e).lower() and "cuda" in str(e).lower():
                    memory_guidance = (
                        "GPU memory remained insufficient after automatic retries. "
                        "Set Cellpose Internal Block Size to 160 or 128, keep Cellpose GPU Batch Size at 1, "
                        "or turn off Use GPU to run this image on CPU."
                    )
                    status_container.write(f"Error segmenting {img_name}: {memory_guidance}")
                    st.warning(memory_guidance)
                else:
                    status_container.write(f"Error segmenting {img_name}: {e}")
                failed_count += 1
                record_process_event(
                    "tab5",
                    "Segmentation image failed",
                    message=str(e),
                    status="error",
                    duration_s=elapsed,
                    details={"image": img_name, "input": str(img_path)},
                )
                
            if create_log and log_dir:
                write_seg_image_log(log_dir, image_log_data)
                append_to_seg_details(log_dir, image_log_data, is_first=is_first_log)
                is_first_log = False
                
            progress_bar.progress(min((current_idx + 1) / total_pairs, 1.0))
            
        clear_batch_stop(output_dir)
        if stopped_early:
            status_container.update(label="Stopped by user", state="error")
            st.warning(
                f"Batch stopped by request. Processed: {processed_count}, Failed: {failed_count}, "
                f"Total Cells: {total_cells}. Everything already written was kept — re-run to continue "
                "(tick 'Overwrite' off to skip images that are already done)."
            )
        else:
            status_container.update(label="Complete!", state="complete")
            st.success(f"Batch completed. Processed: {processed_count}, Failed: {failed_count}, Total Cells: {total_cells}")
        record_process_event(
            "tab5",
            "Segmentation stopped" if stopped_early else "Segmentation complete",
            status="warning" if stopped_early else "complete",
            duration_s=time.time() - session_start_time,
            details={"images_processed": processed_count, "images_failed": failed_count, "cells_detected": total_cells, "output_directory": str(output_dir), "stopped_by_user": stopped_early},
        )


    except Exception as e:
        clear_batch_stop(output_dir)
        status_container.update(label="Error", state="error")
        st.error(f"Batch execution error: {e}")
        record_process_event(
            "tab5",
            "Segmentation batch failed",
            message=str(e),
            status="error",
            duration_s=time.time() - session_start_time,
            details={"output_directory": str(output_dir)},
        )

def render():
    """Page entry point: two subtabs over one shared project.

    The segmentation pipeline body is left exactly where it was and simply
    rendered inside the first subtab, so the existing layout, widget keys and
    control flow are untouched. Streamlit runs the body of every subtab on every
    script run, which is safe here because all the heavy work in both subtabs
    sits behind explicit Run buttons.
    """
    from spacemap_gui.ui_utils import require_active_project

    project_root = require_active_project()
    if project_root is None:
        return

    pipeline_tab, tagging_tab = st.tabs(
        ["Segmentation & feature extraction", "Spatial cell tagging"]
    )
    with pipeline_tab:
        _render_segmentation_pipeline()
    with tagging_tab:
        cell_tagging_ui.render_cell_tagging_panel(project_root)


def _render_segmentation_pipeline():
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    project_root_str = str(project_root)
    try:
        with open(project_root / "tab4_debug.txt", "a") as f:
            f.write(f"\n=== TAB5 RENDER AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} ===\n")
            f.write(f"active_project={st.session_state.get('active_project') is not None}\n")
            f.write(f"tab4_input_dir={repr(st.session_state.get('tab4_input_dir'))}\n")
            f.write(f"proj_dir_tab4_input_dir={repr(st.session_state.get('proj_dir_tab4_input_dir'))}\n")
            f.write(f"tab4_output_dir={repr(st.session_state.get('tab4_output_dir'))}\n")
            f.write(f"tab4_seg_mask_save_dir={repr(st.session_state.get('tab4_seg_mask_save_dir'))}\n")
            f.write(f"snapshot={repr(st.session_state.get('_dir_sync_snapshot', {}).get('tab4_input_dir'))}\n")
    except Exception as e:
        pass
    # print(f"[DEBUG TAB5] render() called. active_project={st.session_state.get('active_project') is not None}, tab4_input_dir={repr(st.session_state.get('tab4_input_dir'))}", flush=True)
    st.markdown("## Segmentation Pipeline")
    
    # Ensure project directories are synchronized and loaded automatically
    active_proj = st.session_state.get("active_project")
    updated = False
    if active_proj:
        mappings = active_proj.get("directory_mappings", {})
        for key in ["tab4_input_dir", "tab4_output_dir", "tab4_seg_mask_save_dir"]:
            proj_val = mappings.get(key)
            if proj_val and not st.session_state.get(key):
                st.session_state[key] = proj_val
                updated = True
 
    # Defaults Initialization
    defaults = {
        'tab4_input_dir': st.session_state.get('tab4_input_dir') or st.session_state.get('tab3_output_dir') or str(project_root / 'outputs/rois'),
        'tab4_output_dir': st.session_state.get('tab4_output_dir') or str(project_root / 'outputs/segmentation/cellpose'),
        'tab4_seg_mask_save_dir': st.session_state.get('tab4_seg_mask_save_dir') or str(project_root / 'data/boundaries'),
    }
    for k, v in defaults.items():
        if k not in st.session_state or not st.session_state[k]:
            st.session_state[k] = v
            updated = True
 
    if updated:
        st.rerun()
 
    # Header
    st.markdown("Segment cell boundaries using Cellpose with customizable tissue masks and parameters.")
 
    input_dir = st.session_state['tab4_input_dir']
    mask_save_dir = st.session_state['tab4_seg_mask_save_dir']
    output_dir = st.session_state['tab4_output_dir']
    
    input_path = Path(input_dir)
    image_files = []
    if input_path.exists():
        image_files = sorted([f for f in os.listdir(input_path) if f.endswith(('.ome.tiff', '.qptiff'))])
        
    # Pairs setup
    pairs = []
    for img in image_files:
        mask_file, is_new_mask = find_matched_mask_file(img, mask_save_dir)
        pairs.append({
            'image': img,
            'mask': mask_file.name if mask_file else None,
            'has_mask': mask_file is not None,
            'is_new_mask': is_new_mask
        })
        
    # Binds the preview to the selected folder and discards anything cached from
    # a previous one, so the editor can only ever show a file that is in the
    # directory named above.
    active_img = sync_preview_to_input_dir(input_dir, image_files)


    # Layout split
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Directories")
        # print(f"[DEBUG TAB5] before widget. tab4_input_dir={repr(st.session_state.get('tab4_input_dir'))}", flush=True)
        input_dir = directory_input_with_browse(
            label="Input Directory (ROIs)",
            text_key="tab4_input_dir",
            default_value=str(project_root / 'outputs/rois'),
            browser_key_prefix="tab4_input_dir",
            start_dir_fallback=str(project_root),
        )
        mask_save_dir = directory_input_with_browse(
            label="Drawn tissue mask directory (optional — this tab's own editor)",
            text_key="tab4_seg_mask_save_dir",
            default_value=str(project_root / 'data/boundaries'),
            browser_key_prefix="tab4_seg_mask_save_dir",
            start_dir_fallback=str(project_root),
            help_text=(
                "This is NOT the Annotation tab's GeoJSON. It is where the masks you draw or auto-detect in "
                "the Preview & Tissue Mask Editor below are saved and reloaded, as "
                "<stem>_roi_bounding_boxes_coords.geojson. It is only used when 'Segmentation region' is set "
                "to 'Drawn tissue mask (this tab)'.\n\n"
                "If you already reviewed the tissue in the Annotation tab, leave this alone and set "
                "'Segmentation region' to 'Annotation GeoJSON' instead — that path uses the single reviewed "
                "annotation file for both the segmentation mask and the cell spatial tags."
            ),
        )
        output_dir = directory_input_with_browse(
            label="Output Directory (Segmentation Results)",
            text_key="tab4_output_dir",
            default_value=str(project_root / 'outputs/segmentation/cellpose'),
            browser_key_prefix="tab4_output_dir",
            start_dir_fallback=str(project_root),
        )
        if "tab4_create_log" not in st.session_state:
            st.session_state["tab4_create_log"] = True
        create_log = st.checkbox("Create Processing Log (CSV)", key="tab4_create_log")
        

            
        
        st.subheader("2. Pairs")
        if not image_files:
            st.info("No images found in the Input Directory.")
        else:
            # Search Images
            search_query = st.text_input(
                "Search images...",
                value="",
                placeholder="Search image...",
                label_visibility="collapsed",
                key="tab4_pairs_search"
            )
            filtered_pairs = pairs
            if search_query:
                filtered_pairs = [p for p in pairs if search_query.lower() in p['image'].lower()]
                
            # Toggle Batch
            def _toggle_batch_seg():
                v = st.session_state.get('tab4_all_proc', False)
                for p in filtered_pairs: st.session_state[f"tab4_mp_{p['image']}"] = v
                
            c_all_chk, c_all_lbl = st.columns([0.08, 0.92])
            c_all_chk.checkbox("##", key="tab4_all_proc", on_change=_toggle_batch_seg, label_visibility="collapsed")
            c_all_lbl.markdown("**Select All for Batch**")
            
            # List of Pairs. Rendered inside a fixed-height scrolling container so
            # every image stays reachable - this used to stop at 30 and print
            # "... N more images ...", which made the rest unselectable.
            if not filtered_pairs:
                st.caption("No matching images found.")
            else:
                st.caption(f"{len(filtered_pairs)} image(s) — scroll for more.")
                list_container = st.container(height=420, border=True)
                for p in filtered_pairs:
                    img = p['image']
                    c_chk, c_btn = list_container.columns([0.08, 0.92])
                    k_proc = f"tab4_mp_{img}"
                    if k_proc not in st.session_state: st.session_state[k_proc] = False
                    c_chk.checkbox("##", key=k_proc, label_visibility="collapsed")
                    
                    is_active = (active_img == img)
                    btn_type = "primary" if is_active else "secondary"
                    stored_objs = st.session_state.setdefault('tab4_seg_canvas_objects', {}).get(str(Path(input_dir) / img), [])
                    has_polys = len([o for o in stored_objs if o.get('type') != 'image']) > 0
                    
                    if p['has_mask']:
                        mask_lbl = " (Masked)"
                    elif has_polys:
                        mask_lbl = " (Tissue Detected)"
                    else:
                        mask_lbl = ""
                    label = f"{img}{mask_lbl} (Active)" if is_active else f"{img}{mask_lbl}"
                    
                    if c_btn.button(label, key=f"tab4_btn_{img}", type=btn_type, width='stretch'):
                        st.session_state['tab4_active_img'] = img
                        st.rerun()




        # Channels selection
        if active_img:
            img_full_path = Path(input_dir) / active_img
            try:
                mtime = os.path.getmtime(str(img_full_path))
            except:
                mtime = 0.0
            st.subheader("3. Channels")
            try:
                ch_names, ch_colors, ch_meta, mpp = get_cached_channel_info(str(img_full_path), mtime=mtime)
            except:
                ch_names, ch_colors, ch_meta, mpp = [], [], [], 0.5083
                
            file_key = active_img.replace('.', '_').replace(' ', '_')
            nuclear_state_key = f'seg_nuclear_{file_key}'
            membrane_state_key = f'seg_membrane_{file_key}'
            tissue_state_key = f'seg_tissue_{file_key}'
            
            # Ensure keys are integers (since JSON loads keys as strings)
            for state_key in [nuclear_state_key, membrane_state_key, tissue_state_key]:
                if state_key in st.session_state and isinstance(st.session_state[state_key], dict):
                    st.session_state[state_key] = {int(k): v for k, v in st.session_state[state_key].items()}

            # Auto-load the mask when this image has not been loaded yet, or when
            # the file on disk changed since it was loaded. Without the timestamp
            # check an edit made in QuPath is never picked up: the session keeps
            # serving the copy it read the first time, even across tab reloads.
            st.session_state.setdefault('tab4_seg_canvas_objects', {})
            st.session_state.setdefault('_tab4_seg_mask_stamps', {})
            _matched_for_stamp, _ = find_matched_mask_file(active_img, mask_save_dir)
            try:
                _disk_stamp = os.path.getmtime(_matched_for_stamp) if _matched_for_stamp else None
            except OSError:
                _disk_stamp = None
            _loaded_stamp = st.session_state['_tab4_seg_mask_stamps'].get(str(img_full_path), "unset")
            _mask_changed = _loaded_stamp != "unset" and _disk_stamp != _loaded_stamp
            if _mask_changed:
                st.info(
                    f"`{Path(_matched_for_stamp).name if _matched_for_stamp else 'mask'}` changed on disk; "
                    "reloading it into the editor."
                )
            if str(img_full_path) not in st.session_state['tab4_seg_canvas_objects'] or _mask_changed:
                st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = []
                st.session_state['_tab4_seg_mask_stamps'][str(img_full_path)] = _disk_stamp
                matched_mask, is_new_mask = find_matched_mask_file(active_img, mask_save_dir)
                if matched_mask and matched_mask.exists():
                    try:
                        with open(matched_mask, 'r') as f:
                            geo_data = json.load(f)
                        
                        # Compute exact scale
                        try:
                            _, full_shape = get_cached_channel_image(str(img_full_path), 0, mtime=mtime)
                            full_h, full_w = full_shape[0], full_shape[1]
                            if full_w > 0:
                                display_width = 700
                                canvas_w = min(full_w, display_width)
                                calc_scale_x = float(full_w) / float(canvas_w)
                                calc_scale_y = float(full_h) / float(int(full_h * (canvas_w / full_w)))
                            else:
                                calc_scale_x, calc_scale_y = 1.0, 1.0
                        except:
                            calc_scale_x, calc_scale_y = 1.0, 1.0
                            
                        loaded_drawing = geojson_to_fabric(geo_data, 1 / calc_scale_x, 1 / calc_scale_y)
                        loaded_objs = [o for o in (loaded_drawing.get('objects', []) or []) if isinstance(o, dict) and o.get('type') != 'image']
                        if loaded_objs:
                            st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = loaded_objs
                    except Exception as e:
                        print(f"Error auto-loading mask for {active_img}: {e}")

            # Defaults.
            # Selections are stored per image as {channel index: bool}. Indices are
            # only meaningful within one panel, so a new image is seeded from the
            # last used selection matched BY CHANNEL NAME. Without that, moving to
            # an image with a different panel silently selects the wrong markers.
            def _seed_from_last_used(state_key: str, remembered_key: str, fallback_names: list) -> None:
                existing = st.session_state.get(state_key)
                if existing and any(existing.values()):
                    return
                remembered = [str(n) for n in st.session_state.get(remembered_key, [])]
                if remembered:
                    wanted = {n.lower() for n in remembered}
                    seeded = {i: (ch.lower() in wanted) for i, ch in enumerate(ch_names)}
                    if any(seeded.values()):
                        st.session_state[state_key] = seeded
                        return
                st.session_state[state_key] = {
                    i: any(f.lower() in ch.lower() for f in fallback_names)
                    for i, ch in enumerate(ch_names)
                }

            if ch_names:
                _seed_from_last_used(nuclear_state_key, "tab4_last_nuclear_names", ['DAPI'])
                _seed_from_last_used(
                    membrane_state_key, "tab4_last_membrane_names",
                    ['E-cadherin', 'Pan-Cytokeratin', 'CD31', 'Vimentin'],
                )

                _seed_from_last_used(tissue_state_key, "tab4_last_tissue_names", ['DAPI'])
                
            with st.expander("Channel Selections", expanded=True):
                c0, c1, c2, c3, c4 = st.columns([1, 1, 1, 4, 2])
                c0.markdown("**Nuc**")
                c1.markdown("**Mem**")
                c2.markdown("**Tis / H&E**")
                c3.markdown("**Channel**")
                c4.markdown("**Color**")
                st.caption("Use Tis for nuclear/marker tissue detection, or select HE_Red, HE_Green, and HE_Blue here to enable registered H&E tissue detection.")
                
                for i, (ch, col) in enumerate(zip(ch_names, ch_colors)):
                    c0, c1, c2, c3, c4 = st.columns([1, 1, 1, 4, 2])
                    
                    is_nuc = st.session_state[nuclear_state_key].get(i, False)
                    if c0.checkbox("##", value=is_nuc, key=f"seg_nuc_chk_{file_key}_{i}", label_visibility="collapsed"):
                        st.session_state[nuclear_state_key][i] = True
                    else:
                        st.session_state[nuclear_state_key][i] = False
                        
                    is_mem = st.session_state[membrane_state_key].get(i, False)
                    if c1.checkbox("##", value=is_mem, key=f"seg_mem_chk_{file_key}_{i}", label_visibility="collapsed"):
                        st.session_state[membrane_state_key][i] = True
                    else:
                        st.session_state[membrane_state_key][i] = False

                    is_tis = st.session_state[tissue_state_key].get(i, False)
                    if c2.checkbox("##", value=is_tis, key=f"seg_tis_chk_{file_key}_{i}", label_visibility="collapsed"):
                        st.session_state[tissue_state_key][i] = True
                    else:
                        st.session_state[tissue_state_key][i] = False
                        
                    c3.text(ch)
                    if isinstance(col, tuple):
                        hex_col = '#{:02x}{:02x}{:02x}'.format(*col)
                    else:
                        hex_col = col
                    c4.color_picker(f"cp_{file_key}_{i}", value=hex_col, disabled=True, label_visibility="collapsed")

                selected_membrane_channels = [
                    channel_name for index, channel_name in enumerate(ch_names)
                    if st.session_state[membrane_state_key].get(index, False)
                ]
                st.caption(
                    f"Cellpose uses one maximum-intensity membrane projection from {len(selected_membrane_channels)} selected Mem channel(s), plus the selected nuclear channel."
                )

                # Remember the selection by NAME so the next image inherits it even
                # when its panel has a different channel order.
                def _names_for(state_key: str) -> list:
                    return [
                        channel_name for index, channel_name in enumerate(ch_names)
                        if st.session_state[state_key].get(index, False)
                    ]

                st.session_state["tab4_last_nuclear_names"] = _names_for(nuclear_state_key)
                st.session_state["tab4_last_membrane_names"] = selected_membrane_channels
                st.session_state["tab4_last_tissue_names"] = _names_for(tissue_state_key)

                apply_column, save_column = st.columns(2)
                if apply_column.button(
                    "Apply this channel selection to all images",
                    key="tab4_apply_channels_all",
                    width="stretch",
                    help=(
                        "Copies the Nuc/Mem/Tis selection to every listed image, matched by channel "
                        "name so images with a different panel order still get the right markers."
                    ),
                ):
                    targets = [pair["image"] for pair in pairs]
                    wanted = {
                        "nuclear": {n.lower() for n in st.session_state["tab4_last_nuclear_names"]},
                        "membrane": {n.lower() for n in selected_membrane_channels},
                        "tissue": {n.lower() for n in st.session_state["tab4_last_tissue_names"]},
                    }
                    applied = 0
                    for target in targets:
                        target_key = target.replace('.', '_').replace(' ', '_')
                        target_path = Path(input_dir) / target
                        try:
                            target_mtime = os.path.getmtime(str(target_path))
                            target_names, _, _, _ = get_cached_channel_info(str(target_path), mtime=target_mtime)
                        except Exception:
                            continue
                        for role, prefix in (("nuclear", "seg_nuclear_"), ("membrane", "seg_membrane_"), ("tissue", "seg_tissue_")):
                            st.session_state[f"{prefix}{target_key}"] = {
                                index: (str(name).lower() in wanted[role])
                                for index, name in enumerate(target_names)
                            }
                        applied += 1
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab5_segmentation")
                    st.success(f"Applied the channel selection to {applied} image(s) and saved it to the project.")

                if save_column.button(
                    "Save channel selection",
                    key="tab4_save_channels",
                    width="stretch",
                    help="Writes the current per-image channel selection into the project configuration.",
                ):
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab5_segmentation")
                    st.success("Channel selection saved to the project configuration.")
                if len(selected_membrane_channels) > 6:
                    st.warning(
                        "More than 6 membrane channels are selected. This increases full-resolution disk I/O for every tissue tile and can make large images much slower. "
                        "Use a small, high-signal panel unless you specifically need more channels."
                    )
                if st.button("Use Recommended Fast Membrane Panel", key=f"tab4_fast_membrane_{file_key}", width="stretch"):
                    recommended_markers = {"e-cadherin", "pan-cytokeratin", "cd31", "vimentin"}
                    st.session_state[membrane_state_key] = {
                        index: channel_name.lower() in recommended_markers
                        for index, channel_name in enumerate(ch_names)
                    }
                    st.rerun()

            # Cellpose Parameters
            st.subheader("4. Cellpose Parameters")
            model_options = ['cyto', 'nuclei', 'cyto2', 'cp', 'custom']
            st.selectbox("Model Type", model_options, index=model_options.index(st.session_state.get("tab4_model_type", "cyto")), key="tab4_model_type")
            
            if st.session_state.get("tab4_model_type") == 'custom':
                directory_input_with_browse(
                    label="Custom Model Path",
                    text_key="tab4_custom_model",
                    default_value=st.session_state.get("tab4_custom_model", str(project_root / "data/seg_model/CP_20241014_094617")),
                    browser_key_prefix="tab4_custom_model",
                    start_dir_fallback=str(project_root),
                )
                
            c_p1, c_p2 = st.columns(2)
            c_p1.number_input("Diameter (px, 0=auto)", value=st.session_state.get("tab4_diameter", 0), key="tab4_diameter")
            c_p2.slider("Flow Threshold", 0.0, 1.0, st.session_state.get("tab4_flow", 0.8), 0.1, key="tab4_flow")
            
            c_t1, c_t2 = st.columns(2)
            c_t1.number_input("Outer ROI Tile Size (px)", value=st.session_state.get("tab4_tile_size", 2048), min_value=256, step=256, key="tab4_tile_size", help="For masked segmentation, SPACEMAP splits the large ROI into these outer tiles before stitching features and masks together.")
            c_t2.slider("Tile Overlap", 0.0, 0.5, st.session_state.get("tab4_tile_overlap", 0.1), 0.05, key="tab4_tile_overlap")
            
            st.checkbox("Use GPU", value=st.session_state.get("tab4_gpu", torch.cuda.is_available()), key="tab4_gpu")

            st.markdown("---")
            st.markdown("**Tissue mask & spatial annotation**")
            spatial_ui.render_spatial_panel(project_root, [p["image"] for p in pairs], input_dir)
            st.markdown("---")
            with st.expander("GPU Memory Safety", expanded=True):
                st.caption("The outer ROI tile controls stitching. The Cellpose internal block controls GPU memory. A 2048px outer ROI tile is safe when Cellpose uses small internal blocks.")
                memory_col1, memory_col2 = st.columns(2)
                internal_block_size = memory_col1.number_input(
                    "Cellpose Internal Block Size (px)",
                    min_value=128,
                    max_value=1024,
                    step=32,
                    value=st.session_state.get("tab4_cellpose_internal_tile_size", 224),
                    key="tab4_cellpose_internal_tile_size",
                    help="Cellpose splits each outer ROI tile into blocks of this size. Lower values use less GPU memory but take longer.",
                )
                memory_col2.number_input(
                    "Cellpose GPU Batch Size",
                    min_value=1,
                    max_value=64,
                    step=1,
                    value=st.session_state.get("tab4_cellpose_batch_size", 8),
                    key="tab4_cellpose_batch_size",
                    help=(
                        "Number of internal Cellpose blocks sent to the GPU together. 8 measured fastest on a "
                        "40 GB A100; 16 was no better. Batch size only helps once the flow check runs on the "
                        "GPU, because otherwise the CPU flow step dominates. It is reduced automatically when "
                        "free GPU memory is low, and again after any out-of-memory error."
                    ),
                )
                speed_col1, speed_col2 = st.columns(2)
                cudnn_options = ["auto", "enabled", "disabled"]
                saved_cudnn = st.session_state.get("tab4_cudnn_mode", "auto")
                speed_col1.selectbox(
                    "cuDNN",
                    cudnn_options,
                    index=cudnn_options.index(saved_cudnn) if saved_cudnn in cudnn_options else 0,
                    key="tab4_cudnn_mode",
                    help=(
                        "auto probes cuDNN once and falls back if the install is broken. Measured about 12% "
                        "faster than running without it. Set disabled only if you hit libcudnn errors."
                    ),
                )
                flow_options = ["auto", "gpu", "cpu"]
                saved_flow = st.session_state.get("tab4_flow_device", "auto")
                speed_col2.selectbox(
                    "Flow-error check device",
                    flow_options,
                    index=flow_options.index(saved_flow) if saved_flow in flow_options else 0,
                    key="tab4_flow_device",
                    help=(
                        "Where the flow threshold is applied after network prediction. This was the single "
                        "biggest cost in the old pipeline: moving it to the GPU alone was 1.6x faster, with "
                        "the same cells. auto uses the GPU whenever one is available."
                    ),
                )
                if int(internal_block_size) > 256 or int(st.session_state.get("tab4_cellpose_batch_size", 1)) > 1:
                    st.warning(
                        "These settings are high-memory. For large masked images, start with a 224px internal block and GPU batch size 1. "
                        "A 1024px block with batch size 3 can be slower and can exceed GPU memory."
                    )
                    def use_recommended_gpu_settings() -> None:
                        st.session_state["tab4_cellpose_internal_tile_size"] = 224
                        st.session_state["tab4_cellpose_batch_size"] = 1
                        st.session_state["tab4_adaptive_gpu_memory"] = True
                        st.session_state["tab4_min_cellpose_internal_tile_size"] = 128
                    st.button(
                        "Use Recommended Large-Image GPU Settings",
                        key="tab4_use_safe_gpu_settings",
                        width="stretch",
                        on_click=use_recommended_gpu_settings,
                    )
                st.checkbox(
                    "Retry automatically with smaller Cellpose blocks after GPU memory errors",
                    value=st.session_state.get("tab4_adaptive_gpu_memory", True),
                    key="tab4_adaptive_gpu_memory",
                    help="On a CUDA out-of-memory error, clears the GPU cache and retries the same tile with progressively smaller internal blocks.",
                )
                retry_options = [size for size in [128, 160, 192, 224, 256] if size <= int(internal_block_size)]
                retry_default = st.session_state.get("tab4_min_cellpose_internal_tile_size", 128)
                if retry_default not in retry_options:
                    retry_default = retry_options[0]
                st.select_slider(
                    "Smallest automatic retry block (px)",
                    options=retry_options,
                    value=retry_default,
                    key="tab4_min_cellpose_internal_tile_size",
                    help="The retry limit. Lower values are slower but more likely to run on memory-limited GPUs.",
                )
                st.number_input(
                    "Skip outer tiles below tissue coverage fraction",
                    min_value=0.0,
                    max_value=0.05,
                    step=0.001,
                    value=st.session_state.get("tab4_min_tissue_fraction", 0.002),
                    key="tab4_min_tissue_fraction",
                    help="Tiles with less tissue than this fraction are skipped. A small value avoids spending GPU time on nearly empty tile edges.",
                )
                st.number_input(
                    "Tissue crop padding inside each ROI tile (px)",
                    min_value=0,
                    max_value=512,
                    step=16,
                    value=st.session_state.get("tab4_tissue_crop_padding", 64),
                    key="tab4_tissue_crop_padding",
                    help="Extra pixels around the tissue bounding box. More padding protects cells near the boundary; less padding reduces GPU work.",
                )
            
            st.subheader("5. Output Formats")
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            c_opt1.checkbox("Save GeoJSON", value=st.session_state.get("tab4_save_geojson", True), key="tab4_save_geojson")
            c_opt2.checkbox("Save NPY (Mask)", value=st.session_state.get("tab4_save_npy", False), key="tab4_save_npy")
            c_opt3.checkbox("Save PNG (Vis)", value=st.session_state.get("tab4_save_png", False), key="tab4_save_png")
            st.checkbox(
                "Overwrite existing segmentation outputs",
                value=bool(st.session_state.get("tab4_overwrite", True)),
                key="tab4_overwrite",
                help=(
                    "On: every selected image is re-segmented and its outputs replaced. "
                    "Off: images that already have a features CSV (and the other selected output "
                    "formats) are skipped, so an interrupted batch can be resumed without redoing work."
                ),
            )

            # The stop control has to live outside the run branch so it is still
            # on the page while a batch is in flight, and it writes a file rather
            # than session state because the script thread is busy segmenting.
            if batch_stop_requested(output_dir):
                c_stop_msg, c_stop_undo = st.columns([3, 1])
                c_stop_msg.warning("Stop requested — the batch will halt before its next image.")
                if c_stop_undo.button("Cancel stop", key="tab4_cancel_stop", width='stretch'):
                    clear_batch_stop(output_dir)
                    st.rerun()
            elif st.button(
                "Stop batch after current image", key="tab4_stop_seg", width='stretch',
                help=(
                    "Asks a running batch to finish the image it is on, keep everything already written, and "
                    "stop before the next one. Cellpose cannot be interrupted mid-image, so this is not "
                    "instant on a large ROI.\n\n"
                    "Ctrl+C in the terminal does not work here: it reaches the Streamlit server, not the "
                    "worker running the batch. If the browser looks frozen while a batch runs that is "
                    "expected — click this, then wait for the current image to finish."
                ),
            ):
                if request_batch_stop(output_dir):
                    st.rerun()

            c_run_seg, c_save_proj = st.columns(2)
            with c_run_seg:
                if st.button("Run Segmentation (Selected)", type="primary", width='stretch'):
                    to_proc = [p for p in pairs if st.session_state.get(f"tab4_mp_{p['image']}")]
                    preflight_errors = validate_run_directories(
                        {"ROI input directory": input_dir},
                        output_dir,
                    )
                    preflight_errors.extend(
                        validate_selected_files(
                            [Path(input_dir) / pair["image"] for pair in to_proc],
                            item_name="image",
                        )
                    )

                    active_file_key = active_img.replace('.', '_').replace(' ', '_') if active_img else ""
                    active_nuclear_selection = st.session_state.get(
                        f"seg_nuclear_{active_file_key}", {}
                    )
                    images_without_nuclear_channel = []
                    for pair in to_proc:
                        image_key = pair["image"].replace('.', '_').replace(' ', '_')
                        nuclear_selection = st.session_state.get(
                            f"seg_nuclear_{image_key}", active_nuclear_selection
                        )
                        if not isinstance(nuclear_selection, dict) or not any(nuclear_selection.values()):
                            images_without_nuclear_channel.append(pair["image"])
                    if images_without_nuclear_channel:
                        preflight_errors.append(
                            "Select at least one nuclear channel in Channel Selections before segmenting."
                        )

                    model_type = st.session_state.get("tab4_model_type", "cyto")
                    custom_model_path = st.session_state.get("tab4_custom_model", "")
                    if model_type == "custom" and not Path(str(custom_model_path)).expanduser().exists():
                        preflight_errors.append("Choose an existing custom Cellpose model path.")

                    mask_source = st.session_state.get(spatial_ui.MASK_SOURCE_KEY, spatial_ui.MASK_SOURCE_NONE)
                    use_mask = mask_source == spatial_ui.MASK_SOURCE_DRAWN
                    use_annotation_mask = mask_source == spatial_ui.MASK_SOURCE_ANNOTATION
                    if use_mask:
                        mask_path = Path(mask_save_dir).expanduser()
                        if not mask_path.is_dir():
                            preflight_errors.append("Choose an existing tissue mask directory or turn off mask-based segmentation.")
                        else:
                            missing_masks = [
                                pair["image"] for pair in to_proc
                                if find_matched_mask_file(pair["image"], mask_save_dir)[0] is None
                            ]
                            if missing_masks:
                                preflight_errors.append(
                                    "No tissue mask was found for: " + ", ".join(missing_masks[:3])
                                )
                    elif use_annotation_mask:
                        unmatched = [
                            pair["image"] for pair in to_proc
                            if not spatial_ui.build_spatial_config(pair["image"])["geojson_path"]
                        ]
                        if unmatched:
                            preflight_errors.append(
                                "No annotation GeoJSON matched: " + ", ".join(unmatched[:3])
                                + ". Check the annotation directory or switch the segmentation region to 'No mask'."
                            )
                        without_class = [
                            pair["image"] for pair in to_proc
                            if spatial_ui.build_spatial_config(pair["image"])["geojson_path"]
                            and not spatial_ui.build_spatial_config(pair["image"])["mask_classes"]
                        ]
                        if without_class:
                            preflight_errors.append(
                                "Choose a mask polygon class for: " + ", ".join(without_class[:3])
                            )

                    if preflight_errors:
                        record_process_event(
                            "tab5",
                            "Segmentation preflight failed",
                            status="warning",
                            details={"errors": preflight_errors},
                        )
                        show_preflight_errors(preflight_errors)
                    else:
                        _run_batch_segmentation_v2(
                            to_proc=to_proc,
                            input_dir=input_dir,
                            mask_save_dir=mask_save_dir,
                            output_dir=output_dir,
                            use_mask=use_mask,
                            use_annotation_mask=use_annotation_mask,
                            create_log=create_log,
                            model_type=model_type,
                            custom_model_path=custom_model_path,
                            diameter=st.session_state.get("tab4_diameter", 0),
                            flow=st.session_state.get("tab4_flow", 0.8),
                            tile_size=st.session_state.get("tab4_tile_size", 2048),
                            tile_overlap=st.session_state.get("tab4_tile_overlap", 0.1),
                            cellpose_internal_tile_size=st.session_state.get("tab4_cellpose_internal_tile_size", 224),
                            cellpose_batch_size=st.session_state.get("tab4_cellpose_batch_size", 1),
                            adaptive_gpu_memory=st.session_state.get("tab4_adaptive_gpu_memory", True),
                            min_cellpose_internal_tile_size=st.session_state.get("tab4_min_cellpose_internal_tile_size", 128),
                            min_tissue_fraction=st.session_state.get("tab4_min_tissue_fraction", 0.002),
                            tissue_crop_padding=st.session_state.get("tab4_tissue_crop_padding", 64),
                            gpu=st.session_state.get("tab4_gpu", True),
                            save_geojson=st.session_state.get("tab4_save_geojson", True),
                            save_npy=st.session_state.get("tab4_save_npy", False),
                            save_png=st.session_state.get("tab4_save_png", False),
                            cudnn_mode=st.session_state.get("tab4_cudnn_mode", "auto"),
                            flow_device=st.session_state.get("tab4_flow_device", "auto"),
                            overwrite=st.session_state.get("tab4_overwrite", True),
                        )
            with c_save_proj:
                if st.button("Save Project", key="tab4_save_project_btn", width='stretch'):
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab5_segmentation")
                    st.success("Project settings saved to configuration file.")

    with col2:
        header_column, reload_column = st.columns([3, 1])
        header_column.subheader("Preview & Tissue Mask Editor")
        reload_column.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if reload_column.button(
            "Reload from disk", key="tab4_preview_reload", width="stretch",
            help=(
                "Discard every cached preview - channel lists, thumbnails, canvas backgrounds and drawn "
                "shapes - and re-read the image from disk. Use this after replacing or regenerating a file "
                "in the input directory while the app is running."
            ),
        ):
            clear_preview_caches("manual reload")
            st.rerun()

        if not image_files:
            st.info(f"No images found in `{input_dir or '(not set)'}`.")
        elif not active_img:
            st.info("Select an image from the list on the left to configure.")
        else:
            st.caption(f"Showing `{active_img}` from `{input_dir}`.")
            img_full_path = Path(input_dir) / active_img

            # Show the annotation polygon that will bound segmentation, so the
            # chosen class can be confirmed before a long run starts.
            if st.session_state.get(spatial_ui.MASK_SOURCE_KEY) == spatial_ui.MASK_SOURCE_ANNOTATION:
                with st.expander("Segmentation mask preview", expanded=True):
                    spatial_ui.render_mask_overlay(str(img_full_path), active_img)
            file_key = active_img.replace('.', '_').replace(' ', '_')
            
            # Integrated Toolbar: Action Target + Drawing Tool
            c_mode1, c_mode2 = st.columns([1, 1])
            with c_mode1:
                drawing_target = st.radio(
                    "Action Target",
                    ["Tissue (Inclusion)", "Artifact Hole (Exclusion)"],
                    horizontal=True,
                    key=f"tab4_target_type_{file_key}",
                    help="Tissue (Inclusion): Draw regions of valid tissue to segment.\nArtifact Hole (Exclusion): Draw background/artifact areas to subtract."
                )
            with c_mode2:
                drawing_tool = st.radio(
                    "Drawing Tool",
                    ["Select / Edit", "Polygon", "Freehand", "Rectangle", "Auto Detect"],
                    horizontal=True,
                    key=f"tab4_drawing_tool_{file_key}",
                    help="Select / Edit: Click shapes to move, reshape, or delete.\nPolygon: Click points to place vertices, double-click or right-click to finish.\nFreehand: Drag mouse contour.\nRectangle: Drag box.\nAuto Detect: Automatic thresholding or H&E tissue detection."
                )

            # Map drawing tool to Fabric.js mode
            if drawing_tool.startswith("Select"):
                drawing_mode = "transform"
            elif drawing_tool.startswith("Polygon"):
                drawing_mode = "polygon"
            elif drawing_tool.startswith("Freehand"):
                drawing_mode = "freedraw"
            elif drawing_tool.startswith("Rectangle"):
                drawing_mode = "rect"
            else:
                drawing_mode = "transform"

            # Fill / Stroke Color mapping
            if drawing_target.startswith("Tissue"):
                fill_color = "rgba(255, 165, 0, 0.3)"
                stroke_color = "#ff8c00"
            else:
                fill_color = "rgba(255, 0, 0, 0.3)"
                stroke_color = "#ff0000"

            # Predefined Mask quick-loader if present
            matched_mask, is_new_mask = find_matched_mask_file(active_img, mask_save_dir)
            if matched_mask and matched_mask.exists():
                c_pre1, c_pre2 = st.columns([3, 1])
                c_pre1.caption(f"Matched GeoJSON: `{matched_mask.name}` ({'Custom saved new_mask' if is_new_mask else 'Predefined original'})")
                if c_pre2.button("Load GeoJSON", key=f"tab4_btn_load_predef_{file_key}", help="Load existing GeoJSON mask onto canvas"):
                    with st.spinner("Loading predefined boundaries..."):
                        try:
                            with open(matched_mask, 'r') as f:
                                geo_data = json.load(f)
                            last_scale = st.session_state.get('tab4_seg_last_scale', {}).get(str(img_full_path), {'scale_x': 1.0, 'scale_y': 1.0})
                            loaded_drawing = geojson_to_fabric(geo_data, 1 / last_scale['scale_x'], 1 / last_scale['scale_y'])
                            loaded_objs = [o for o in (loaded_drawing.get('objects', []) or []) if isinstance(o, dict) and o.get('type') != 'image']
                            if loaded_objs:
                                current_objs = st.session_state.setdefault('tab4_seg_canvas_objects', {}).setdefault(str(img_full_path), [])
                                current_objs = [o for o in current_objs if isinstance(o, dict) and o.get('type') != 'image']
                                current_objs.extend(loaded_objs)
                                st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = current_objs
                                st.session_state[f"tab4_canvas_ignore_sync_{file_key}"] = True
                                st.session_state.setdefault('tab4_seg_canvas_reset', {})[str(img_full_path)] = st.session_state.setdefault('tab4_seg_canvas_reset', {}).get(str(img_full_path), 0) + 1
                                st.rerun()
                            else:
                                st.warning("No boundaries found in GeoJSON.")
                        except Exception as e:
                            st.error(f"Error loading predefined GeoJSON: {e}")

            # If Auto Tissue Detection is selected
            if drawing_tool.startswith("Auto Detect"):
                st.markdown("**Auto Tissue Detection Configurations**")
                
                tis_indices = [int(i) for i, sel in st.session_state.get(f'seg_tissue_{file_key}', {}).items() if sel]
                tissue_source = st.radio(
                    "Tissue detection source",
                    [
                        "Nuclear / selected marker channels",
                        "Registered H&E RGB (HE_Red, HE_Green, HE_Blue)",
                    ],
                    horizontal=True,
                    key=f"tab4_tissue_source_{file_key}",
                    help="Nuclear detection uses the selected Tis channels and intensity thresholds. H&E detection follows the registered H&E luminance-inversion and Otsu-mask workflow.",
                )

                if tissue_source.startswith("Registered H&E"):
                    active_he_indices, missing_he_layers = get_he_channel_indices(ch_names, tis_indices)
                    if missing_he_layers:
                        st.warning(
                            "Select the registered H&E layers in the Tis column to use this method. "
                            f"Missing from the current selection: {', '.join(missing_he_layers)}."
                        )
                    else:
                        st.success(
                            "H&E tissue source ready: "
                            + ", ".join(f"HE_{role.title()}" for role in HE_CHANNEL_ROLES)
                            + "."
                        )
                        st.caption(
                            "This method averages H&E luminance, inverts it so tissue is bright, applies Otsu thresholding, then closes gaps and removes small holes and fragments."
                        )
                        he_refine_a, he_refine_b, he_refine_c, he_refine_d = st.columns(4)
                        he_closing_radius = he_refine_a.slider(
                            "H&E gap-closing radius (px)",
                            min_value=0,
                            max_value=25,
                            value=5,
                            key=f"tab4_he_close_radius_{file_key}",
                        )
                        he_hole_area = he_refine_b.number_input(
                            "Fill holes below (px)",
                            min_value=0,
                            value=3000,
                            step=500,
                            key=f"tab4_he_hole_area_{file_key}",
                        )
                        he_min_object_area = he_refine_c.number_input(
                            "Keep tissue above (px)",
                            min_value=0,
                            value=3000,
                            step=500,
                            key=f"tab4_he_min_object_area_{file_key}",
                        )
                        he_buffer_size = he_refine_d.slider(
                            "Polygon buffer (px)",
                            min_value=0,
                            max_value=25,
                            value=0,
                            key=f"tab4_he_buffer_{file_key}",
                        )
                        detection_mode = st.radio(
                            "Tissue Detection Mode",
                            ["Single Image", "Batch (All Images)"],
                            horizontal=True,
                            key=f"tab4_he_auto_det_mode_{file_key}",
                        )

                        if detection_mode == "Single Image":
                            if st.button("Detect Tissue from H&E", type="primary", width="stretch", key=f"tab4_btn_run_he_{file_key}"):
                                with st.spinner("Detecting tissue from registered H&E layers..."):
                                    record_process_event(
                                        "tab5",
                                        "H&E tissue detection started",
                                        details={"image": active_img, "source": "HE_Red/HE_Green/HE_Blue"},
                                    )
                                    try:
                                        detected_objs, tissue_mask = run_he_tissue_detection(
                                            str(img_full_path),
                                            active_he_indices,
                                            closing_radius=int(he_closing_radius),
                                            hole_area=int(he_hole_area),
                                            min_object_area=int(he_min_object_area),
                                            buffer_size=float(he_buffer_size),
                                        )
                                        if detected_objs:
                                            st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = detected_objs
                                            st.session_state.setdefault('tab4_seg_canvas_reset', {})[str(img_full_path)] = st.session_state.setdefault('tab4_seg_canvas_reset', {}).get(str(img_full_path), 0) + 1
                                            record_process_event(
                                                "tab5",
                                                "H&E tissue detection complete",
                                                status="complete",
                                                details={"image": active_img, "polygons_detected": len(detected_objs)},
                                            )
                                            st.success(f"Detected {len(detected_objs)} H&E tissue polygon(s). Review, merge if needed, then save the mask.")
                                            st.rerun()
                                        else:
                                            st.warning("No H&E tissue polygons were detected. Try reducing the minimum tissue area or use nuclear-marker detection.")
                                    except Exception as error:
                                        record_process_event(
                                            "tab5",
                                            "H&E tissue detection failed",
                                            message=str(error),
                                            status="error",
                                            details={"image": active_img},
                                        )
                                        st.error(f"H&E tissue detection could not run: {error}. Check the selected H&E layers or use nuclear-marker detection.")
                        else:
                            st.caption("Before processing, SPACEMAP checks every image for HE_Red, HE_Green, and HE_Blue. Images missing any layer are not changed.")
                            if st.button("Validate and Run H&E Batch Tissue Detection", type="primary", width="stretch", key=f"tab4_btn_run_he_batch_{file_key}"):
                                with st.spinner("Validating registered H&E layers across all images..."):
                                    he_indices_by_image, missing_by_image = validate_batch_he_channels(image_files, input_dir)
                                if missing_by_image:
                                    record_process_event(
                                        "tab5",
                                        "H&E batch validation failed",
                                        status="warning",
                                        details={"images_missing_layers": missing_by_image},
                                    )
                                    missing_summary = "; ".join(
                                        f"{name}: {', '.join(missing)}"
                                        for name, missing in list(missing_by_image.items())[:8]
                                    )
                                    st.error(
                                        "H&E batch validation failed. No masks were changed. "
                                        f"Images missing required layers: {missing_summary}. Use nuclear-marker detection or process only H&E-compatible images."
                                    )
                                else:
                                    with st.spinner("Creating tissue masks from registered H&E layers..."):
                                        success_count, failures = run_batch_he_tissue_detection(
                                            image_files,
                                            input_dir,
                                            mask_save_dir,
                                            he_indices_by_image,
                                            closing_radius=int(he_closing_radius),
                                            hole_area=int(he_hole_area),
                                            min_object_area=int(he_min_object_area),
                                            buffer_size=float(he_buffer_size),
                                        )
                                    record_process_event(
                                        "tab5",
                                        "H&E batch tissue detection complete",
                                        status="complete" if not failures else "warning",
                                        details={
                                            "images_masked": success_count,
                                            "images_failed": failures,
                                            "source": "HE_Red/HE_Green/HE_Blue",
                                        },
                                    )
                                    if failures:
                                        failure_summary = "; ".join(
                                            f"{name}: {message}"
                                            for name, message in list(failures.items())[:5]
                                        )
                                        st.warning(f"Created {success_count} H&E tissue mask(s). Some images failed: {failure_summary}")
                                    else:
                                        st.success(f"Created and saved H&E tissue masks for all {success_count} images.")
                                    st.rerun()

                elif not tis_indices:
                    st.warning("Please check one or more channels under 'Tis' in the Channels list on the left for tissue detection.")
                else:
                    detection_mode = st.radio(
                        "Tissue Detection Mode",
                        ["Single Image", "Batch (All Images)"],
                        horizontal=True,
                        key=f"tab4_auto_det_mode_{file_key}"
                    )
                    
                    if detection_mode == "Single Image":
                        # Cache the combined raw image to avoid re-loading/re-blending on slider changes
                        cache_key = f"tab4_auto_combined_raw_{file_key}"
                        indices_cache_key = f"tab4_auto_combined_indices_{file_key}"
                        
                        cached_indices = st.session_state.get(indices_cache_key, [])
                        if cache_key not in st.session_state or cached_indices != tis_indices:
                            loaded_imgs = []
                            for idx in tis_indices:
                                # mtime must be passed, or this entry is keyed on the
                                # path alone and keeps serving the first version read.
                                img_data, _ = get_cached_channel_image(
                                    str(img_full_path), idx,
                                    mtime=os.path.getmtime(img_full_path) if os.path.exists(img_full_path) else 0.0,
                                )
                                if img_data is not None:
                                    loaded_imgs.append(img_data)
                            if loaded_imgs:
                                if len(loaded_imgs) == 1:
                                    combined_img = loaded_imgs[0]
                                else:
                                    combined_img = np.maximum.reduce(loaded_imgs)
                                st.session_state[cache_key] = combined_img
                                st.session_state[indices_cache_key] = tis_indices
                            else:
                                st.session_state[cache_key] = None
                                st.session_state[indices_cache_key] = []
                        
                        combined_img = st.session_state.get(cache_key)
                        if combined_img is not None:
                            # Determine threshold scale type (Z-Score or Raw Intensity)
                            thresh_type_key = f"tab4_auto_thresh_type_{file_key}"
                            if thresh_type_key not in st.session_state:
                                st.session_state[thresh_type_key] = "Z-Score"
                            thresh_type = st.radio("Intensity Scale", ["Z-Score", "Raw Intensity"], horizontal=True, key=thresh_type_key)
                            
                            # Determine range bounds (8-bit vs 16-bit vs other)
                            is_16bit = False
                            if np.issubdtype(combined_img.dtype, np.integer):
                                try:
                                    max_range = int(np.iinfo(combined_img.dtype).max)
                                except:
                                    max_range = 65535 if combined_img.dtype == np.uint16 else 255
                            else:
                                max_range = 255
                            is_16bit = (max_range > 255)
                            
                            merge_key = f"tab4_auto_merge_{file_key}"
                            buffer_key = f"tab4_auto_buffer_{file_key}"
                            min_area_key = f"tab4_auto_min_area_{file_key}"
                            
                            if merge_key not in st.session_state:
                                st.session_state[merge_key] = 6
                            if buffer_key not in st.session_state:
                                st.session_state[buffer_key] = 1
                            if min_area_key not in st.session_state:
                                st.session_state[min_area_key] = 100
                                
                            if thresh_type == "Z-Score":
                                default_min = 1.0
                                default_max = 4.0
                                slider_min_key = f"tab4_auto_thresh_z_min_{file_key}"
                                slider_max_key = f"tab4_auto_thresh_z_max_{file_key}"
                                slider_key = f"tab4_auto_thresh_z_range_slider_single_{file_key}"
                                num_min_key = f"tab4_auto_num_z_min_single_{file_key}"
                                num_max_key = f"tab4_auto_num_z_max_single_{file_key}"
                                
                                if slider_min_key not in st.session_state:
                                    st.session_state[slider_min_key] = default_min
                                if slider_max_key not in st.session_state:
                                    st.session_state[slider_max_key] = default_max
                                    
                                val_min = st.session_state.get(slider_min_key, default_min)
                                val_max = st.session_state.get(slider_max_key, default_max)
                                
                                last_slider = st.session_state.get(f"_last_single_z_slider_{file_key}", (val_min, val_max))
                                last_min = st.session_state.get(f"_last_single_z_min_{file_key}", val_min)
                                last_max = st.session_state.get(f"_last_single_z_max_{file_key}", val_max)
                                
                                current_slider = st.session_state.get(slider_key, (val_min, val_max))
                                current_min = st.session_state.get(num_min_key, val_min)
                                current_max = st.session_state.get(num_max_key, val_max)
                                
                                if current_slider != last_slider:
                                    val_min, val_max = current_slider
                                elif current_min != last_min:
                                    val_min = current_min
                                elif current_max != last_max:
                                    val_max = current_max
                                    
                                val_min = min(max(float(val_min), -3.0), 8.0)
                                val_max = min(max(float(val_max), -3.0), 8.0)
                                if val_min > val_max:
                                    val_min, val_max = val_max, val_min
                                    
                                st.session_state[slider_min_key] = val_min
                                st.session_state[slider_max_key] = val_max
                                st.session_state[f"_last_single_z_slider_{file_key}"] = (val_min, val_max)
                                st.session_state[f"_last_single_z_min_{file_key}"] = val_min
                                st.session_state[f"_last_single_z_max_{file_key}"] = val_max
                                
                                combined_img_float = combined_img.astype(np.float32)
                                mean = np.mean(combined_img_float)
                                std = np.std(combined_img_float)
                                z_img = (combined_img_float - mean) / (std + 1e-6)
                                plot_data = z_img[::10, ::10].ravel()
                                log_scale = False
                                
                            else:
                                # Raw Intensity
                                default_min = 159
                                default_max = 215
                                if is_16bit:
                                    default_min = int(default_min * 256)
                                    default_max = int(default_max * 256)
                                    
                                slider_min_key = f"tab4_auto_thresh_raw_min_{file_key}"
                                slider_max_key = f"tab4_auto_thresh_raw_max_{file_key}"
                                slider_key = f"tab4_auto_thresh_raw_range_slider_single_{file_key}"
                                num_min_key = f"tab4_auto_num_raw_min_single_{file_key}"
                                num_max_key = f"tab4_auto_num_raw_max_single_{file_key}"
                                
                                if slider_min_key not in st.session_state:
                                    st.session_state[slider_min_key] = default_min
                                if slider_max_key not in st.session_state:
                                    st.session_state[slider_max_key] = default_max
                                    
                                val_min = st.session_state.get(slider_min_key, default_min)
                                val_max = st.session_state.get(slider_max_key, default_max)
                                
                                last_slider = st.session_state.get(f"_last_single_raw_slider_{file_key}", (val_min, val_max))
                                last_min = st.session_state.get(f"_last_single_raw_min_{file_key}", val_min)
                                last_max = st.session_state.get(f"_last_single_raw_max_{file_key}", val_max)
                                
                                current_slider = st.session_state.get(slider_key, (val_min, val_max))
                                current_min = st.session_state.get(num_min_key, val_min)
                                current_max = st.session_state.get(num_max_key, val_max)
                                
                                if current_slider != last_slider:
                                    val_min, val_max = current_slider
                                elif current_min != last_min:
                                    val_min = current_min
                                elif current_max != last_max:
                                    val_max = current_max
                                    
                                val_min = min(max(int(val_min), 0), max_range)
                                val_max = min(max(int(val_max), 0), max_range)
                                if val_min > val_max:
                                    val_min, val_max = val_max, val_min
                                    
                                st.session_state[slider_min_key] = val_min
                                st.session_state[slider_max_key] = val_max
                                st.session_state[f"_last_single_raw_slider_{file_key}"] = (val_min, val_max)
                                st.session_state[f"_last_single_raw_min_{file_key}"] = val_min
                                st.session_state[f"_last_single_raw_max_{file_key}"] = val_max
                                
                                plot_data = combined_img[::10, ::10].ravel()
                                
                                c_log, c_blank = st.columns([1, 1])
                                log_scale = c_log.checkbox("Log scale X-axis", value=st.session_state.get(f"tab4_auto_hist_log_{file_key}", False), key=f"tab4_auto_hist_log_{file_key}")
                                
                            # Display smooth KDE histogram using Matplotlib and Seaborn
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            fig.patch.set_facecolor('#1a1c24')
                            ax.set_facecolor('#1a1c24')
                            ax.spines['bottom'].set_color('#cccccc')
                            ax.spines['left'].set_color('#cccccc')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.tick_params(colors='#cccccc')
                            ax.yaxis.label.set_color('#cccccc')
                            ax.xaxis.label.set_color('#cccccc')
                            ax.title.set_color('#cccccc')
                            
                            if log_scale:
                                plot_data = plot_data[plot_data > 0]
                                
                            try:
                                if len(plot_data) > 0:
                                    if log_scale:
                                        sns.kdeplot(plot_data, ax=ax, color="#FF4B4B", fill=True, log_scale=True)
                                    else:
                                        sns.kdeplot(plot_data, ax=ax, color="#FF4B4B", fill=True)
                                else:
                                    ax.text(0.5, 0.5, "No positive values for log scale", color='#cccccc', ha='center', va='center')
                            except Exception as e:
                                # Fallback: simple line plot of histogram
                                try:
                                    if thresh_type == "Z-Score":
                                        hist_vals, bin_edges = np.histogram(plot_data, bins=100, range=(-3, 8))
                                    else:
                                        hist_vals, bin_edges = np.histogram(plot_data, bins=100, range=(0, max_range))
                                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                                    ax.plot(bin_centers, hist_vals, color="#FF4B4B")
                                    ax.fill_between(bin_centers, hist_vals, color="#FF4B4B", alpha=0.3)
                                    if log_scale:
                                        ax.set_xscale('log')
                                except Exception as e:
                                    from spacemap_gui.ui_utils import log_exception
                                    log_exception("plotting histogram", e)
                                    
                            # Plot barriers in real-time
                            ax.axvline(x=val_min, color='#FFC048', linestyle='--', linewidth=1.5, label=f'Min: {val_min:.2f}' if thresh_type == "Z-Score" else f'Min: {val_min}')
                            ax.axvline(x=val_max, color='#1DD1A1', linestyle='--', linewidth=1.5, label=f'Max: {val_max:.2f}' if thresh_type == "Z-Score" else f'Max: {val_max}')
                            
                            ax.set_title("Marker Intensity Distribution (Smooth)" if thresh_type == "Raw Intensity" else "Marker Z-Score Distribution (Smooth)")
                            ax.set_xlabel("Intensity" if thresh_type == "Raw Intensity" else "Z-Score")
                            ax.set_ylabel("Density")
                            ax.legend(facecolor='#1a1c24', edgecolor='#cccccc', labelcolor='#cccccc')
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            c_a1, c_a2 = st.columns(2)
                            if thresh_type == "Z-Score":
                                thresh_range = c_a1.slider(
                                    "Z-Score Range",
                                    -3.0, 8.0,
                                    (val_min, val_max),
                                    step=0.1,
                                    key=slider_key
                                )
                            else:
                                thresh_range = c_a1.slider(
                                    "Intensity Range",
                                    0, max_range,
                                    (val_min, val_max),
                                    key=slider_key
                                )
                            auto_merge = c_a2.slider("Merge Radius (pixels)", 1, 50, st.session_state[merge_key], key=merge_key)
                            
                            c_m1, c_m2 = st.columns(2)
                            if thresh_type == "Z-Score":
                                manual_min = c_m1.number_input("Manual Min Z-Score", -3.0, 8.0, val_min, step=0.1, key=num_min_key)
                                manual_max = c_m2.number_input("Manual Max Z-Score", -3.0, 8.0, val_max, step=0.1, key=num_max_key)
                            else:
                                manual_min = c_m1.number_input("Manual Min Intensity", 0, max_range, val_min, step=1, key=num_min_key)
                                manual_max = c_m2.number_input("Manual Max Intensity", 0, max_range, val_max, step=1, key=num_max_key)
                                
                            c_a3, c_a4 = st.columns(2)
                            auto_buffer = c_a3.slider("Buffer Size (pixels)", 0, 50, st.session_state[buffer_key], key=buffer_key)
                            auto_min_area = c_a4.number_input("Min Area (px²)", value=st.session_state[min_area_key], min_value=1, key=min_area_key)
                            
                            if st.button("Run Auto Tissue Detection", type="primary", width='stretch', key=f"tab4_btn_run_auto_{file_key}"):
                                with st.spinner("Detecting tissue boundaries..."):
                                    detected_objs = run_auto_tissue_detection(
                                        image_path=str(img_full_path),
                                        channel_indices=tis_indices,
                                        threshold_min=thresh_range[0],
                                        threshold_max=thresh_range[1],
                                        merge_radius=auto_merge,
                                        buffer_size=auto_buffer,
                                        min_area=auto_min_area,
                                        use_zscore=(thresh_type == "Z-Score")
                                    )
                                    if detected_objs:
                                        # Auto erase previous polygons and draw the new ones
                                        st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = detected_objs
                                        st.session_state.setdefault('tab4_seg_canvas_reset', {})[str(img_full_path)] = st.session_state.setdefault('tab4_seg_canvas_reset', {}).get(str(img_full_path), 0) + 1
                                        st.rerun()
                                    else:
                                        st.warning("No tissue regions detected. Try adjusting the threshold range.")
                                        
                    else:
                        st.markdown("##### Batch Z-Score Tissue Detection")
                        
                        # Button to calculate
                        if st.button("Calculate Z-Scores & Average Histogram", key=f"tab4_btn_calc_z_{file_key}", width='stretch'):
                            with st.spinner("Calculating Z-scores across all images..."):
                                batch_data, avg_hist, bins = calculate_batch_zscore_data(image_files, input_dir, tis_indices)
                                st.session_state["tab4_batch_zscore_data"] = batch_data
                                st.session_state["tab4_batch_avg_hist"] = avg_hist
                                st.session_state["tab4_batch_hist_bins"] = bins
                                st.success(f"Calculated Z-scores for {len(batch_data)} images.")
                                st.rerun()
                                
                        avg_hist = st.session_state.get("tab4_batch_avg_hist")
                        bins = st.session_state.get("tab4_batch_hist_bins")
                        
                        # Set default values for range on Z-scale
                        z_min_key = f"tab4_auto_z_min_{file_key}"
                        z_max_key = f"tab4_auto_z_max_{file_key}"
                        if z_min_key not in st.session_state:
                            st.session_state[z_min_key] = 1.0
                        if z_max_key not in st.session_state:
                            st.session_state[z_max_key] = 4.0
                            
                        merge_key = f"tab4_auto_merge_batch_{file_key}"
                        buffer_key = f"tab4_auto_buffer_batch_{file_key}"
                        min_area_key = f"tab4_auto_min_area_batch_{file_key}"
                        
                        if merge_key not in st.session_state:
                            st.session_state[merge_key] = 6
                        if buffer_key not in st.session_state:
                            st.session_state[buffer_key] = 1
                        if min_area_key not in st.session_state:
                            st.session_state[min_area_key] = 100
                            
                        # Slider and Manual input sync
                        slider_z_key = f"tab4_auto_thresh_z_slider_{file_key}"
                        num_z_min_key = f"tab4_auto_num_z_min_{file_key}"
                        num_z_max_key = f"tab4_auto_num_z_max_{file_key}"
                        
                        val_z_min = st.session_state.get(z_min_key, 1.0)
                        val_z_max = st.session_state.get(z_max_key, 4.0)
                        
                        last_z_slider = st.session_state.get(f"_last_z_slider_{file_key}", (val_z_min, val_z_max))
                        last_z_min = st.session_state.get(f"_last_z_min_{file_key}", val_z_min)
                        last_z_max = st.session_state.get(f"_last_z_max_{file_key}", val_z_max)
                        
                        current_z_slider = st.session_state.get(slider_z_key, (val_z_min, val_z_max))
                        current_z_min = st.session_state.get(num_z_min_key, val_z_min)
                        current_z_max = st.session_state.get(num_z_max_key, val_z_max)
                        
                        if current_z_slider != last_z_slider:
                            val_z_min, val_z_max = current_z_slider
                        elif current_z_min != last_z_min:
                            val_z_min = current_z_min
                        elif current_z_max != last_z_max:
                            val_z_max = current_z_max
                            
                        val_z_min = min(max(float(val_z_min), -3.0), 8.0)
                        val_z_max = min(max(float(val_z_max), -3.0), 8.0)
                        if val_z_min > val_z_max:
                            val_z_min, val_z_max = val_z_max, val_z_min
                            
                        st.session_state[z_min_key] = val_z_min
                        st.session_state[z_max_key] = val_z_max
                        st.session_state[f"_last_z_slider_{file_key}"] = (val_z_min, val_z_max)
                        st.session_state[f"_last_z_min_{file_key}"] = val_z_min
                        st.session_state[f"_last_z_max_{file_key}"] = val_z_max
                        
                        # Render average Z-score histogram
                        if avg_hist is not None and bins is not None:
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            fig, ax = plt.subplots(figsize=(6, 2.5))
                            fig.patch.set_facecolor('#1a1c24')
                            ax.set_facecolor('#1a1c24')
                            ax.spines['bottom'].set_color('#cccccc')
                            ax.spines['left'].set_color('#cccccc')
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.tick_params(colors='#cccccc')
                            ax.yaxis.label.set_color('#cccccc')
                            ax.xaxis.label.set_color('#cccccc')
                            ax.title.set_color('#cccccc')
                            
                            ax.plot(bin_centers, avg_hist, color="#FF4B4B", linewidth=2)
                            ax.fill_between(bin_centers, avg_hist, color="#FF4B4B", alpha=0.3)
                            
                            # Barriers
                            ax.axvline(x=val_z_min, color='#FFC048', linestyle='--', linewidth=1.5, label=f'Min Z: {val_z_min:.2f}')
                            ax.axvline(x=val_z_max, color='#1DD1A1', linestyle='--', linewidth=1.5, label=f'Max Z: {val_z_max:.2f}')
                            
                            ax.set_title("Average Z-Score Distribution (All Batches)")
                            ax.set_xlabel("Z-Score")
                            ax.set_ylabel("Density")
                            ax.legend(facecolor='#1a1c24', edgecolor='#cccccc', labelcolor='#cccccc')
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("Click 'Calculate Z-Scores & Average Histogram' to initialize Z-score distribution visualization.")
                            
                        # Controls
                        c_a1, c_a2 = st.columns(2)
                        thresh_z_range = c_a1.slider(
                            "Z-Score Threshold Range",
                            -3.0, 8.0,
                            (val_z_min, val_z_max),
                            step=0.1,
                            key=slider_z_key
                        )
                        auto_merge = c_a2.slider("Merge Radius (pixels)", 1, 50, st.session_state[merge_key], key=merge_key)
                        
                        c_m1, c_m2 = st.columns(2)
                        manual_z_min = c_m1.number_input("Manual Min Z-Score", -3.0, 8.0, val_z_min, step=0.1, key=num_z_min_key)
                        manual_z_max = c_m2.number_input("Manual Max Z-Score", -3.0, 8.0, val_z_max, step=0.1, key=num_z_max_key)
                        
                        c_a3, c_a4 = st.columns(2)
                        auto_buffer = c_a3.slider("Buffer Size (pixels)", 0, 50, st.session_state[buffer_key], key=buffer_key)
                        auto_min_area = c_a4.number_input("Min Area (px²)", value=st.session_state[min_area_key], min_value=1, key=min_area_key)
                        
                        if st.button("Run Batch Tissue Detection", type="primary", width='stretch', key=f"tab4_btn_run_batch_auto_{file_key}"):
                            with st.spinner("Running batch tissue detection on all ROIs..."):
                                success_count = run_batch_tissue_detection(
                                    image_files=image_files,
                                    input_dir=input_dir,
                                    mask_save_dir=mask_save_dir,
                                    tis_indices=tis_indices,
                                    z_min=thresh_z_range[0],
                                    z_max=thresh_z_range[1],
                                    merge_radius=auto_merge,
                                    buffer_size=auto_buffer,
                                    min_area=auto_min_area
                                )
                                st.success(f"Batch tissue detection complete. Successfully masked {success_count} images.")
                                st.rerun()
                                

            # Preview Composition
            indices_to_load = set()
            active_channels = []
            
            for i, sel in st.session_state.get(f'seg_nuclear_{file_key}', {}).items():
                if sel:
                    idx = int(i)
                    indices_to_load.add(idx)
                    if idx < len(ch_names):
                        active_channels.append(f"{ch_names[idx]} (Nuc)")
            for i, sel in st.session_state.get(f'seg_membrane_{file_key}', {}).items():
                if sel:
                    idx = int(i)
                    indices_to_load.add(idx)
                    if idx < len(ch_names):
                        active_channels.append(f"{ch_names[idx]} (Mem)")
                        
            composite = None
            full_shape = (1000, 1000)
            if indices_to_load:
                img_data_dict, fs, is_rgb = load_preview_channels(str(img_full_path), list(indices_to_load))
                if img_data_dict:
                    full_shape = fs
                    h, w = img_data_dict[next(iter(img_data_dict))].shape
                    composite = np.zeros((h, w, 3), dtype=np.float32)
                    
                    for idx_raw, img in img_data_dict.items():
                        idx = int(idx_raw)
                        img = img.astype(np.float32)
                        p99 = np.percentile(img, 99)
                        if p99 > 0:
                            img = np.clip(img / p99, 0, 1)
                        else:
                            img = img / (img.max() + 1e-6)
                            
                        if idx < len(ch_colors):
                            hex_col = ch_colors[idx]
                            if isinstance(hex_col, tuple):
                                color = hex_col
                            else:
                                color = hex_to_rgb(hex_col)
                        else:
                            color = (255, 255, 255)
                            
                        for c in range(3):
                            composite[:, :, c] += img * color[c]
                    composite = np.clip(composite, 0, 255).astype(np.uint8)
                    
            if composite is None:
                composite = np.zeros((700, 700, 3), dtype=np.uint8)
                
            pil_img = Image.fromarray(composite)
            display_width = 700
            canvas_width = pil_img.width
            canvas_height = pil_img.height
            if canvas_width > display_width:
                ratio = display_width / canvas_width
                canvas_width = display_width
                canvas_height = int(pil_img.height * ratio)
                
            canvas_bg = pil_img.resize((canvas_width, canvas_height))
            if canvas_bg.mode != "RGB":
                canvas_bg = canvas_bg.convert("RGB")
                
            full_h, full_w = full_shape[0], full_shape[1]
            if len(full_shape) == 3:
                if full_shape[0] < full_shape[1] and full_shape[0] < 100:
                    full_h, full_w = full_shape[1], full_shape[2]
            final_scale_x = float(full_w) / float(canvas_width)
            final_scale_y = float(full_h) / float(canvas_height)
            
            st.session_state.setdefault('tab4_seg_last_scale', {})[str(img_full_path)] = {
                'scale_x': final_scale_x,
                'scale_y': final_scale_y,
            }
            
            if "tab4_seg_canvas_bg_cache" not in st.session_state:
                st.session_state["tab4_seg_canvas_bg_cache"] = {}
            bg_cache_key = (str(img_full_path), canvas_width, canvas_height, tuple(sorted(indices_to_load)))
            if bg_cache_key not in st.session_state["tab4_seg_canvas_bg_cache"]:
                bg_b64 = image_to_base64(canvas_bg)
                st.session_state["tab4_seg_canvas_bg_cache"][bg_cache_key] = {
                    "type": "image",
                    "version": "4.4.0",
                    "originX": "left",
                    "originY": "top",
                    "left": 0,
                    "top": 0,
                    "width": canvas_width,
                    "height": canvas_height,
                    "fill": "rgb(0,0,0)",
                    "stroke": None,
                    "strokeWidth": 0,
                    "scaleX": 1,
                    "scaleY": 1,
                    "opacity": 1,
                    "visible": True,
                    "src": bg_b64,
                    "selectable": False,
                    "evented": False,
                    "hoverCursor": "default",
                }
            bg_obj = st.session_state["tab4_seg_canvas_bg_cache"][bg_cache_key]
            
            st.session_state.setdefault('tab4_seg_canvas_objects', {})
            st.session_state.setdefault('tab4_seg_canvas_reset', {})
            
            reset_count = int(st.session_state['tab4_seg_canvas_reset'].get(str(img_full_path), 0))
            canvas_key = f"tab4_canvas_{file_key}_{reset_count}"
            stored_objs = st.session_state['tab4_seg_canvas_objects'].get(str(img_full_path), [])
            non_image_objects = [o for o in stored_objs if isinstance(o, dict) and o.get('type') != 'image']
            
            init_draw_key = f"tab4_init_drawing_{file_key}_{reset_count}"
            if init_draw_key not in st.session_state:
                st.session_state[init_draw_key] = {"version": "4.4.0", "objects": [bg_obj] + non_image_objects}
            initial_drawing = st.session_state[init_draw_key]
            
            action_message_key = f"tab4_canvas_action_message_{file_key}"
            if st.session_state.get(action_message_key):
                st.success(st.session_state.pop(action_message_key))
                
            st.caption("Orange shapes define valid tissue to segment. Red shapes define artifact holes to subtract. Select / Edit mode allows clicking any shape to move, adjust vertices, or delete.")

            _canvas = cast(Any, st_canvas)
            canvas_result = _canvas(
                fill_color=fill_color,
                stroke_width=2,
                stroke_color=stroke_color,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                initial_drawing=initial_drawing,
                key=canvas_key,
                display_toolbar=True,
            )

            def _extract_path_points_seg(obj: dict[str, Any]) -> list[tuple[float, float]]:
                pts: list[tuple[float, float]] = []
                norm_obj = normalize_fabric_path_to_absolute(obj)
                for cmd in norm_obj.get('path', []) or []:
                    if isinstance(cmd, list) and len(cmd) >= 3:
                        c_type = cmd[0]
                        if c_type in ('M', 'L'):
                            pts.append((float(cmd[1]), float(cmd[2])))
                        elif c_type == 'Q' and len(cmd) >= 5:
                            pts.append((float(cmd[3]), float(cmd[4])))
                        elif c_type == 'C' and len(cmd) >= 7:
                            pts.append((float(cmd[5]), float(cmd[6])))
                if len(pts) >= 3 and pts[0] != pts[-1]:
                    pts.append(pts[0])
                return pts
                
            def _is_valid_polygon_obj_seg(obj: dict[str, Any]) -> bool:
                if obj.get('type') == 'rect':
                    return True
                if obj.get('type') != 'path':
                    return False
                pts = _extract_path_points_seg(obj)
                if len(pts) < 3:
                    return False
                try:
                    return Polygon(pts).area > 0.0
                except:
                    return False

            # Real-time state synchronization with change-detection to prevent blinking loops
            ignore_sync_key = f"tab4_canvas_ignore_sync_{file_key}"
            if st.session_state.pop(ignore_sync_key, False):
                print(f"[TAB5 CANVAS LOG] Canvas reset frame for {active_img}: skipping stale browser state sync.", flush=True)
            elif canvas_result is not None and canvas_result.json_data is not None:
                raw_objects = canvas_result.json_data.get('objects', [])
                objs = [o for o in raw_objects if isinstance(o, dict) and o.get('type') != 'image']
                valid: list[dict[str, Any]] = []
                for o in objs:
                    if _is_valid_polygon_obj_seg(o):
                        # Ensure path objects have closed fill styling and normalized absolute coordinates
                        if o.get("type") == "path":
                            path_cmds = o.get("path", [])
                            if path_cmds and path_cmds[-1] != ["z"]:
                                path_cmds.append(["z"])
                            o["closed"] = True
                            o = normalize_fabric_path_to_absolute(o)
                        valid.append(o)
                curr_stored = st.session_state['tab4_seg_canvas_objects'].get(str(img_full_path), [])
                if json.dumps(valid, sort_keys=True) != json.dumps(curr_stored, sort_keys=True):
                    print(f"[TAB5 CANVAS LOG] Syncing {len(valid)} valid canvas shape(s) for {active_img} into session state.", flush=True)
                    st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = valid

            stored = st.session_state['tab4_seg_canvas_objects'].get(str(img_full_path), [])
            incls = len([o for o in stored if not (o.get('fill') == "rgba(255, 0, 0, 0.3)" or o.get('stroke') == "#ff0000")])
            excls = len(stored) - incls
            st.info(f"Tissue Polygons: {incls} inclusion (orange), {excls} artifact removal (red).")

            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.5, 1.5, 1])

            with btn_col1:
                if st.button("Merge / Subtract", help="Subtract red artifact holes from tissue and merge overlapping tissue polygons offline", width="stretch", key=f"tab4_sub_btn_{file_key}"):
                    if stored:
                        print(f"[TAB5 CANVAS LOG] 'Merge / Subtract' clicked. Processing {len(stored)} shape(s) offline for {active_img}...", flush=True)
                        subtracted = subtract_canvas_polygons(stored, 1.0, 1.0)
                        st.session_state.setdefault('tab4_seg_subtracted_preview', {})[str(img_full_path)] = subtracted
                        print(f"[TAB5 CANVAS LOG] 'Merge / Subtract' complete. Output: {len(subtracted)} clean tissue shape(s).", flush=True)
                        st.session_state[action_message_key] = f"Merge & Subtraction Preview updated! {incls} inclusion shape(s) - {excls} artifact removal shape(s) = {len(subtracted)} clean tissue polygon(s) in preview panel. Canvas polygons stay locked in place."
                        st.rerun()
                    else:
                        st.info("No polygons on canvas to process.")

            with btn_col2:
                if st.button("Delete Selected", help="Remove the selected or last drawn polygon from canvas", width="stretch", key=f"tab4_del_sel_btn_{file_key}"):
                    if stored:
                        print(f"[TAB5 CANVAS LOG] 'Delete Selected' clicked. Current stored shapes count: {len(stored)}", flush=True)
                        target_idx = -1
                        for idx_s, obj_s in enumerate(stored):
                            if isinstance(obj_s, dict) and (obj_s.get("active") or obj_s.get("selected") or obj_s.get("isEditing")):
                                target_idx = idx_s
                                break
                        if target_idx != -1:
                            deleted_obj = stored.pop(target_idx)
                            print(f"[TAB5 CANVAS LOG] Deleted ACTIVE selected shape at index {target_idx}: {deleted_obj.get('type')}", flush=True)
                        else:
                            deleted_obj = stored.pop()
                            print(f"[TAB5 CANVAS LOG] No active flag found; deleted LAST shape at index {len(stored)}: {deleted_obj.get('type')}", flush=True)
                        st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = stored
                        st.session_state[f"tab4_canvas_ignore_sync_{file_key}"] = True
                        st.session_state['tab4_seg_canvas_reset'][str(img_full_path)] = reset_count + 1
                        st.rerun()
                    else:
                        st.info("No shapes on canvas to delete.")

            with btn_col3:
                if st.button("Save Tissue Mask", type="primary", help="Apply artifact subtractions at full resolution and save tissue boundary GeoJSON for segmentation", width="stretch", key=f"tab4_save_mask_btn_{file_key}"):
                    save_dir = st.session_state['tab4_seg_mask_save_dir']
                    if not stored:
                        st.warning("No tissue polygons on canvas to save.")
                    else:
                        print(f"[TAB5 CANVAS LOG] 'Save Tissue Mask' clicked for {active_img}.", flush=True)
                        out_file = save_drawn_segmentation_mask(
                            stored,
                            final_scale_x,
                            final_scale_y,
                            img_full_path,
                            save_dir
                        )
                        print(f"[TAB5 CANVAS LOG] Tissue mask saved successfully to {out_file}.", flush=True)
                        st.session_state[action_message_key] = f"Saved full high-resolution tissue mask with subtractions applied to `{out_file}`."
                        st.rerun()

            with btn_col4:
                if st.button("Clear Canvas", help="Remove all shapes from current image canvas", width="stretch", key=f"tab4_clear_btn_{file_key}"):
                    print(f"[TAB5 CANVAS LOG] 'Clear Canvas' clicked for {active_img}. Clearing all shapes.", flush=True)
                    st.session_state['tab4_seg_canvas_objects'][str(img_full_path)] = []
                    st.session_state[f"tab4_canvas_ignore_sync_{file_key}"] = True
                    st.session_state['tab4_seg_canvas_reset'][str(img_full_path)] = reset_count + 1
                    st.rerun()

            st.caption("Info: Canvas polygons are locked in place. Orange shapes define valid tissue; red shapes define artifact holes to subtract. Merging and subtraction are computed offline for preview and scaled to full high resolution upon Save.")

            # Render Summary Thumbnails at the bottom of the right panel
            st.markdown("---")
            if image_files:
                tis_indices = [int(i) for i, sel in st.session_state.get(f'seg_tissue_{file_key}', {}).items() if sel]
                cols = st.columns(4)
                for idx, img in enumerate(image_files):
                    col = cols[idx % 4]
                    with col:
                        img_path = Path(input_dir) / img
                        # Find matching objects
                        objs = st.session_state.setdefault('tab4_seg_canvas_objects', {}).get(str(img_path), [])
                        
                        # Generate thumbnail
                        thumb = generate_summary_thumbnail(img_path, tis_indices, objs, 1.0, 1.0)
                        if thumb:
                            st.image(thumb, width='stretch', caption=img[:20] + "..." if len(img) > 23 else img)
                        else:
                            st.caption(f"{img[:10]} (No preview)")
                            
                        # Button to select as active
                        is_active = (active_img == img)
                        btn_type = "primary" if is_active else "secondary"
                        if st.button("Select", key=f"tab4_select_thumb_{img}", type=btn_type, width='stretch'):
                            st.session_state['tab4_active_img'] = img
                            st.rerun()

    # --- Post-Segmentation Analysis ---
    st.markdown("---")
    st.header("Post-Segmentation Analysis & Visualization")
    
    selected_file = active_img
    if selected_file:
        stem = Path(selected_file).stem
        if stem.endswith('.ome'): stem = stem[:-4]
        if stem.endswith('_merged'): stem = stem[:-7]
        
        out_dir = Path(st.session_state['tab4_output_dir'])
        default_features_path = out_dir / 'features' / f"{stem}_cell_features.csv"
        default_geojson_path = out_dir / 'boundaries' / f"{stem}_cell_boundaries.geojson"
        
        if 'tab4_last_file_for_csv' not in st.session_state:
            st.session_state['tab4_last_file_for_csv'] = None
        if 'tab4_csv_path' not in st.session_state:
            st.session_state['tab4_csv_path'] = str(default_features_path)
        if 'tab4_geojson_path' not in st.session_state:
            st.session_state['tab4_geojson_path'] = str(default_geojson_path)

        if st.session_state['tab4_last_file_for_csv'] != selected_file:
            st.session_state['tab4_csv_path'] = str(default_features_path)
            st.session_state['tab4_geojson_path'] = str(default_geojson_path)
            st.session_state['tab4_last_file_for_csv'] = selected_file
            st.session_state['tab4_csv_path_input'] = str(default_features_path)
            st.session_state['tab4_geojson_path_input'] = str(default_geojson_path)

        config_file = Path("src/spacemap_gui/interface_config.json")
        saved_csv_dir = None
        saved_geo_dir = None
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    cfg = json.load(f)
                    saved_csv_dir = cfg.get('tab4_csv_dir')
                    saved_geo_dir = cfg.get('tab4_geojson_dir')
            except:
                pass

        project_root_str = st.session_state.get('active_project_root', '.')
        project_root = Path(project_root_str)
        default_seg_dir = str(project_root / "outputs/segmentation/cellpose")

        if 'tab4_csv_dir' not in st.session_state:
             if saved_csv_dir and os.path.exists(saved_csv_dir):
                 st.session_state['tab4_csv_dir'] = saved_csv_dir
             else:
                 current_csv = st.session_state.get('tab4_csv_path', '')
                 if current_csv and os.path.exists(os.path.dirname(current_csv)):
                     st.session_state['tab4_csv_dir'] = os.path.dirname(current_csv)
                 elif os.path.exists(default_seg_dir):
                     st.session_state['tab4_csv_dir'] = default_seg_dir
                 else:
                     st.session_state['tab4_csv_dir'] = str(out_dir / 'features')

        if 'tab4_geojson_dir' not in st.session_state:
             if saved_geo_dir and os.path.exists(saved_geo_dir):
                 st.session_state['tab4_geojson_dir'] = saved_geo_dir
             else:
                 current_geo = st.session_state.get('tab4_geojson_path', '')
                 if current_geo and os.path.exists(os.path.dirname(current_geo)):
                     st.session_state['tab4_geojson_dir'] = os.path.dirname(current_geo)
                 elif os.path.exists(default_seg_dir):
                     st.session_state['tab4_geojson_dir'] = default_seg_dir
                 else:
                     st.session_state['tab4_geojson_dir'] = str(out_dir / 'boundaries')

        def update_config(key, value):
            config_file = Path("src/spacemap_gui/interface_config.json")
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        cfg = json.load(f)
                    cfg[key] = value
                    with open(config_file, 'w') as f:
                        json.dump(cfg, f, indent=4)
                except Exception as e:
                    print(f"Error saving config: {e}")

        st.markdown("##### Select Features CSV")
        prev_csv_dir = st.session_state.get('tab4_csv_dir', '')
        tab4_csv_dir = directory_input_with_browse(
            label="CSV Directory",
            text_key="tab4_csv_dir",
            default_value=str(out_dir / 'features'),
            browser_key_prefix="tab4_csv_dir",
            start_dir_fallback=str(project_root),
        )
        if tab4_csv_dir != prev_csv_dir:
            update_config('tab4_csv_dir', tab4_csv_dir)
        
        csv_dir = tab4_csv_dir
        csv_files = []
        if os.path.exists(csv_dir) and os.path.isdir(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            csv_files.sort()
        
        current_csv_name = os.path.basename(st.session_state.get('tab4_csv_path', ''))
        idx = 0
        if current_csv_name in csv_files:
            idx = csv_files.index(current_csv_name)
            
        selected_csv = st.selectbox("Select CSV File", csv_files, index=idx, key="tab4_single_csv_select")
        if selected_csv:
            st.session_state['tab4_csv_path'] = os.path.join(csv_dir, selected_csv)

        st.markdown("##### Select Boundaries GeoJSON")
        prev_geo_dir = st.session_state.get('tab4_geojson_dir', '')
        tab4_geojson_dir = directory_input_with_browse(
            label="GeoJSON Directory",
            text_key="tab4_geojson_dir",
            default_value=str(out_dir / 'boundaries'),
            browser_key_prefix="tab4_geojson_dir",
            start_dir_fallback=str(project_root),
        )
        if tab4_geojson_dir != prev_geo_dir:
            update_config('tab4_geojson_dir', tab4_geojson_dir)
            
        geo_dir = tab4_geojson_dir
        geo_files = []
        if os.path.exists(geo_dir) and os.path.isdir(geo_dir):
            geo_files = [f for f in os.listdir(geo_dir) if f.endswith('.geojson')]
            geo_files.sort()
            
        current_geo_name = os.path.basename(st.session_state.get('tab4_geojson_path', ''))
        g_idx = 0
        if current_geo_name in geo_files:
            g_idx = geo_files.index(current_geo_name)
            
        selected_geo = st.selectbox("Select GeoJSON File", geo_files, index=g_idx, key="tab4_single_geo_select")
        if selected_geo:
            st.session_state['tab4_geojson_path'] = os.path.join(geo_dir, selected_geo)

        features_path = Path(st.session_state['tab4_csv_path'])
        geojson_path = Path(st.session_state['tab4_geojson_path'])
        
        if not features_path.exists() or not features_path.is_file():
            if features_path.is_dir():
                st.warning(f"Features path is a directory. Please select a CSV file: {features_path}")
            else:
                st.info(f"Features file not found: {features_path}")

            if not geojson_path.exists():
                 st.info("No segmentation results found. Run segmentation first.")
        else:
            an_tab1, an_tab2, an_tab3 = st.tabs(["Segmentation Map", "Histograms", "Summary Table"])
            
            with an_tab1:
                st.subheader("Generate Segmentation Map")
                if geojson_path.exists() and geojson_path.is_file():
                    c_map1, c_map2 = st.columns(2)
                    poly_color = c_map1.color_picker("Polygon Fill Color", "#FF0000")
                    bg_color = c_map2.color_picker("Background Color", "#000000")
                    
                    if st.button("Generate Map"):
                        try:
                            with open(geojson_path, 'r') as f:
                                geo_data = json.load(f)
                            
                            min_x, min_y = float('inf'), float('inf')
                            max_x, max_y = float('-inf'), float('-inf')
                            
                            valid_features = []
                            for feat in geo_data['features']:
                                geom = shape(feat['geometry'])
                                if not geom.is_empty:
                                    b = geom.bounds
                                    min_x = min(min_x, b[0])
                                    min_y = min(min_y, b[1])
                                    max_x = max(max_x, b[2])
                                    max_y = max(max_y, b[3])
                                    valid_features.append(geom)
                            
                            if not valid_features:
                                st.warning("No valid geometries found in GeoJSON.")
                            else:
                                padding = 50
                                w = int(max_x - min_x + 2 * padding)
                                h = int(max_y - min_y + 2 * padding)
                                
                                offset_x = min_x - padding
                                offset_y = min_y - padding
                                
                                map_img = Image.new("RGB", (w, h), bg_color)
                                draw = ImageDraw.Draw(map_img)
                                
                                for geom in valid_features:
                                    if geom.geom_type == 'Polygon':
                                        coords = [(x - offset_x, y - offset_y) for x, y in geom.exterior.coords]
                                        draw.polygon(coords, fill=poly_color)
                                    elif geom.geom_type == 'MultiPolygon':
                                        for poly in geom.geoms:
                                            coords = [(x - offset_x, y - offset_y) for x, y in poly.exterior.coords]
                                            draw.polygon(coords, fill=poly_color)
                                
                                st.image(map_img, caption=f"Segmentation Map (Cropped to {w}x{h})", width="stretch")
                                
                                buf = BytesIO()
                                map_img.save(buf, format="PNG")
                                st.download_button(
                                    label="Download PNG",
                                    data=buf.getvalue(),
                                    file_name=f"{stem}_segmentation_map.png",
                                    mime="image/png"
                                )
                            
                        except Exception as e:
                            st.error(f"Error generating map: {e}")
                else:
                    if geojson_path.is_dir():
                        st.warning(f"GeoJSON path is a directory. Please select a file: {geojson_path}")
                    else:
                        st.warning(f"GeoJSON boundaries not found at {geojson_path}")

            with an_tab2:
                st.subheader("Feature Histograms")
                if features_path.exists():
                    df = pd.read_csv(features_path)
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    numeric_cols = [c for c in numeric_cols if c not in ['label', 'roi_id']]
                    
                    with st.expander("Histogram Settings", expanded=True):
                        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
                        bins = c_h1.slider("Number of Bins", min_value=5, max_value=200, value=50)
                        scale_type = c_h2.selectbox("X-Axis Scale", ["Linear", "Log"], index=0)
                        hist_color = c_h3.color_picker("Color", "#87CEEB")
                        plot_type = c_h4.selectbox("Plot Type", ["Histogram", "Smooth (KDE)", "Both"], index=0)
                    
                    selected_cols = st.multiselect("Select Variables", numeric_cols, default=['area'] if 'area' in numeric_cols else numeric_cols[:1])
                    
                    if selected_cols:
                        for col in selected_cols:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            
                            data = df[col]
                            if scale_type == "Log":
                                data = data[data > 0]
                            
                            try:
                                if plot_type == "Histogram":
                                    sns.histplot(data, bins=bins, color=hist_color, edgecolor='black', alpha=0.7, ax=ax, log_scale=(scale_type=="Log"))
                                elif plot_type == "Smooth (KDE)":
                                    sns.kdeplot(data, color=hist_color, fill=True, ax=ax, log_scale=(scale_type=="Log"))
                                elif plot_type == "Both":
                                    sns.histplot(data, bins=bins, color=hist_color, edgecolor='black', alpha=0.4, kde=True, ax=ax, log_scale=(scale_type=="Log"))
                            except Exception as e:
                                st.warning(f"Plotting error (falling back to basic hist): {e}")
                                ax.hist(data, bins=bins, color=hist_color, edgecolor='black', alpha=0.7)
                                if scale_type == "Log":
                                    ax.set_xscale('log')
                                
                            file_name = features_path.name
                            title = f"{file_name}\nDistribution of {col}"
                            
                            ax.set_title(title)
                            ax.set_xlabel(col)
                            ax.set_ylabel("Count")
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close(fig)
                else:
                    st.warning("Features CSV not found.")

            with an_tab3:
                st.subheader("Summary Statistics")
                df = pd.read_csv(features_path)
                st.write(f"Total Cells: {len(df)}")
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                selected_stats = st.multiselect("Select Variables for Table", numeric_cols, default=numeric_cols[:5])
                
                if selected_stats:
                    desc = df[selected_stats].describe()
                    st.dataframe(desc)

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
