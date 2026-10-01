import streamlit as st
import os
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError for large images
import json
from shapely.geometry import Polygon
from spacemap.registration import process_single_image_rois
from spacemap.utils import hex_to_rgb, extract_rois_with_geometry_from_geojson, resolve_roi_label
from spacemap.background_subtraction_utils import get_channel_info, load_preview_channels_compatible, is_supported_image_file, normalize_image_stem
import pandas as pd
import base64
from io import BytesIO
from PIL import ImageDraw
from spacemap_gui.ui_utils import directory_input_with_browse
from spacemap_gui.process_log import record_process_event
import cv2
import re
import time
from datetime import datetime
from typing import Any, cast, Optional, Dict
from spacemap_gui.tabs.tab1_registration import get_image_metadata_log, _prefix_keys
import csv


from spacemap_gui import terminal


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab3', message, step=None, level=level)


def sync_single_to_multi_dirs():
    if "tab3_input_dir" in st.session_state:
        st.session_state["tab3_multi_input_dir"] = st.session_state["tab3_input_dir"]
    if "tab3_geojson_dir" in st.session_state:
        st.session_state["tab3_multi_geojson_dir"] = st.session_state["tab3_geojson_dir"]
    if "tab3_output_dir" in st.session_state:
        st.session_state["tab3_multi_output_dir"] = st.session_state["tab3_output_dir"]


def sync_multi_to_single_dirs():
    if "tab3_multi_input_dir" in st.session_state:
        st.session_state["tab3_input_dir"] = st.session_state["tab3_multi_input_dir"]
    if "tab3_multi_geojson_dir" in st.session_state:
        st.session_state["tab3_geojson_dir"] = st.session_state["tab3_multi_geojson_dir"]
    if "tab3_multi_output_dir" in st.session_state:
        st.session_state["tab3_output_dir"] = st.session_state["tab3_multi_output_dir"]

# Tab 3 specific logging
def log_tab3(message: str):
    """Tab 3 progress line, printed only while the user is on Tab 3."""
    user_tab = os.environ.get("_spacemap_user_tab", "")
    if user_tab and user_tab != "tab3":
        return
    terminal.say("tab3", message)
    record_process_event("tab3", "ROI extraction step", message=message)

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

@st.cache_data
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
        loaded_data, full_shape, is_rgb = load_preview_channels_compatible(str(file_path), list(indices_to_load), max_dim=1000)
        return loaded_data, full_shape, is_rgb

    except Exception as e:
        _log_console(f"Error loading preview: {e}", level="error")
        return {}, (0,0), False
        
    return loaded_data, full_shape, is_rgb


def _norm_image_prefix(name: str) -> str:
    stem = normalize_image_stem(name)
    if stem.endswith('_merged'):
        stem = stem[:-7]
    return stem.strip().lower()


def _norm_geojson_prefix(name: str) -> str:
    stem = normalize_image_stem(name)
    suffix = '_roi_bounding_boxes_coords'
    if stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]
    if stem.endswith('_merged'):
        stem = stem[:-7]
    return stem.strip().lower()


@st.cache_data(show_spinner=False)
def _load_geojson_cached(path: str, mtime: float) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _count_rois_in_geojson(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        geo = _load_geojson_cached(str(path), path.stat().st_mtime)
        rois = extract_rois_with_geometry_from_geojson(geo)
        return len(rois)
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def _get_cached_channel_info(file_path: str, mtime: float):
    return get_channel_info(file_path)


@st.cache_data(show_spinner=False)
def _load_thumbnail_for_overlay(path: str, mtime: float, max_dim: int = 700) -> tuple[Image.Image, tuple[int, int]]:
    """Load a lightweight thumbnail (and original shape) for overlay preview."""
    preview_data, full_shape, is_rgb = load_preview_channels_compatible(path, [0], max_dim=max_dim)
    if is_rgb and 'rgb' in preview_data:
        arr = np.asarray(preview_data['rgb'])
    elif preview_data:
        arr = np.asarray(next(iter(preview_data.values())))
    else:
        arr = np.zeros((max_dim, max_dim), dtype=np.uint8)

    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            arr = np.moveaxis(arr, 0, -1)
        elif arr.shape[-1] != 3:
            arr = arr[0]

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            lo = float(np.percentile(arr, 1))
            hi = float(np.percentile(arr, 99))
            if hi <= lo:
                hi = lo + 1.0
            arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert('RGB')
    else:
        if arr.dtype != np.uint8:
            lo = float(np.percentile(arr, 1))
            hi = float(np.percentile(arr, 99))
            if hi <= lo:
                hi = lo + 1.0
            arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert('RGB')

    w, h = pil.size
    scale = max(w, h) / float(max_dim)
    if scale > 1.0:
        pil = pil.resize((int(w / scale), int(h / scale)))
    return pil, (int(full_shape[-2]), int(full_shape[-1]))


def _normalize_to_uint8(img: Any) -> np.ndarray:
    if img is None:
        return np.zeros((1, 1), dtype=np.uint8)
    arr = np.asarray(img)
    arr = np.squeeze(arr)
    if arr.dtype == np.uint8:
        return arr
    try:
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return out
    except Exception:
        out = arr.astype(np.float32)
        out = out - np.min(out)
        denom = np.max(out) - np.min(out)
        if denom <= 0:
            denom = 1.0
        out = np.clip(out / denom * 255.0, 0, 255).astype(np.uint8)
        return out


def _color_to_rgb(color_value: Any) -> tuple[int, int, int]:
    if isinstance(color_value, str):
        s = color_value.strip()
        if not s:
            return (255, 255, 255)
        if not s.startswith('#') and re.fullmatch(r"[0-9a-fA-F]{6}", s):
            s = f"#{s}"
        if s.startswith('#') and len(s) == 7:
            try:
                rgb = hex_to_rgb(s)
                return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            except Exception:
                return (255, 255, 255)
        return (255, 255, 255)
    if isinstance(color_value, (tuple, list)) and len(color_value) >= 3:
        try:
            r = int(color_value[0])
            g = int(color_value[1])
            b = int(color_value[2])
            return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        except Exception:
            return (255, 255, 255)
    return (255, 255, 255)


def _find_he_indices(channel_names: list[str]) -> dict[str, int]:
    idx_map: dict[str, int] = {}
    for i, name in enumerate(channel_names or []):
        n = (name or "").strip().lower()
        if n == "he_red" or n.endswith("he_red"):
            idx_map["r"] = i
        elif n == "he_green" or n.endswith("he_green"):
            idx_map["g"] = i
        elif n == "he_blue" or n.endswith("he_blue"):
            idx_map["b"] = i
    return idx_map


def _build_preview_pil(
    file_path: Path,
    channel_names: list[str],
    channel_colors: list[Any],
    preview_mode: str,
    preview_channels: list[str],
    max_width: int = 700,
) -> tuple[Image.Image, tuple[int, int]]:
    """Build a preview image for Multiple Images visualization/editing.

    Returns
    -------
    pil_img:
        Preview image.
    full_hw:
        Full-resolution (H, W) used for scale calculations.
    """
    he_idx = _find_he_indices(channel_names or [])

    indices_to_load: set[int] = set()
    if preview_mode == "H&E" and all(k in he_idx for k in ("r", "g", "b")):
        indices_to_load.update([he_idx["r"], he_idx["g"], he_idx["b"]])
    elif preview_mode == "Single Channel" and preview_channels:
        try:
            idx = (channel_names or []).index(preview_channels[0])
            indices_to_load.add(idx)
        except ValueError:
            pass
    else:
        for ch in preview_channels[:3]:
            try:
                idx = (channel_names or []).index(ch)
                indices_to_load.add(idx)
            except ValueError:
                continue

    if not indices_to_load:
        if all(k in he_idx for k in ("r", "g", "b")):
            indices_to_load.update([he_idx["r"], he_idx["g"], he_idx["b"]])
        elif channel_names:
            indices_to_load.add(0)

    loaded, full_shape, _is_rgb = load_preview_channels(str(file_path), sorted(list(indices_to_load)))
    loaded_any: dict[Any, Any] = cast(Any, loaded)
    full_h = int(full_shape[-2]) if full_shape else 0
    full_w = int(full_shape[-1]) if full_shape else 0

    if preview_mode == "H&E" and all(k in he_idx for k in ("r", "g", "b")):
        r = _normalize_to_uint8(loaded_any.get(he_idx["r"]))
        g = _normalize_to_uint8(loaded_any.get(he_idx["g"]))
        b = _normalize_to_uint8(loaded_any.get(he_idx["b"]))
        rgb = np.dstack([r, g, b]).astype(np.uint8)
        pil = Image.fromarray(rgb).convert("RGB")
    elif preview_mode == "Single Channel" and preview_channels:
        try:
            idx = (channel_names or []).index(preview_channels[0])
            gray = _normalize_to_uint8(loaded_any.get(idx))
            pil = Image.fromarray(gray).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (max_width, max_width), (0, 0, 0))
    else:
        # Composite (additive) using up to 3 selected channels
        if not preview_channels:
            # Fallback to H&E if available, otherwise first channel
            if all(k in he_idx for k in ("r", "g", "b")):
                r = _normalize_to_uint8(loaded_any.get(he_idx["r"]))
                g = _normalize_to_uint8(loaded_any.get(he_idx["g"]))
                b = _normalize_to_uint8(loaded_any.get(he_idx["b"]))
                rgb = np.dstack([r, g, b]).astype(np.uint8)
                pil = Image.fromarray(rgb).convert("RGB")
            else:
                pil = Image.new("RGB", (max_width, max_width), (0, 0, 0))
        else:
            # Use the first loaded channel as shape reference
            first_idx = None
            for ch in preview_channels[:3]:
                try:
                    first_idx = (channel_names or []).index(ch)
                    break
                except ValueError:
                    continue
            if first_idx is None or first_idx not in loaded:
                pil = Image.new("RGB", (max_width, max_width), (0, 0, 0))
            else:
                h, w = loaded_any[first_idx].shape
                comp = np.zeros((h, w, 3), dtype=np.float32)
                for ch in preview_channels[:3]:
                    try:
                        idx = (channel_names or []).index(ch)
                    except ValueError:
                        continue
                    if idx not in loaded:
                        continue
                    img_u8 = _normalize_to_uint8(loaded_any[idx]).astype(np.float32) / 255.0
                    col_idx = idx if idx < len(channel_colors or []) else None
                    rgb = _color_to_rgb(channel_colors[col_idx]) if col_idx is not None else (255, 255, 255)
                    comp[..., 0] += img_u8 * (rgb[0] / 255.0) * 255.0
                    comp[..., 1] += img_u8 * (rgb[1] / 255.0) * 255.0
                    comp[..., 2] += img_u8 * (rgb[2] / 255.0) * 255.0
                comp = np.clip(comp, 0, 255).astype(np.uint8)
                pil = Image.fromarray(comp).convert("RGB")

    # Constrain width for consistent UI/canvas sizing
    if pil.width > max_width:
        ratio = max_width / float(pil.width)
        pil = pil.resize((max_width, int(pil.height * ratio)))
    return pil, (full_h, full_w)

def render_single_image(project_root):
    # log_tab3("=== render() START ===")
    import time
    _start = time.perf_counter()
    
    # Defaults Initialization (Robust Pattern matching Multiple Images)
    defaults = {
        'tab3_input_dir': st.session_state.get('tab3_multi_input_dir', str(project_root)),
        'tab3_geojson_dir': st.session_state.get('tab3_multi_geojson_dir', str(project_root / 'data/boundaries')),
        'tab3_output_dir': st.session_state.get('tab3_multi_output_dir', str(project_root / 'outputs/rois')),
        'roi_mpp_input': 0.5083
    }
    
    # Apply defaults if key missing or empty
    for k, v in defaults.items():
        if k not in st.session_state or not st.session_state[k]:
            st.session_state[k] = v

    st.header("Single Image")

    st.markdown("Draw, edit, and extract ROIs for one image.")

    # Initialize variables
    selected_file = None
    file_path = None
    img_data_dict = {} 
    is_rgb = False
    full_shape = (0, 0)
    edited_channel_names = []
    edited_channel_colors = []
    edited_channel_meta = []
    show_overlay = False
    geojson_path = None
    input_path = None
    geojson_matches_image = False

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Directories")
        
        input_dir = directory_input_with_browse(
            label="Images Directory",
            text_key="tab3_input_dir",
            default_value=str(project_root),
            browser_key_prefix="tab3_input_dir",
            start_dir_fallback=str(project_root),
        )

        geojson_dir = directory_input_with_browse(
            label="GeoJSON Directory",
            text_key="tab3_geojson_dir",
            default_value=str(project_root / 'data/boundaries'),
            browser_key_prefix="tab3_geojson_dir",
            start_dir_fallback=str(project_root),
        )

        output_dir = directory_input_with_browse(
            label="Output Directory",
            text_key="tab3_output_dir",
            default_value=str(project_root / 'outputs/rois'),
            browser_key_prefix="tab3_output_dir",
            start_dir_fallback=str(project_root),
        )

        # Detail logs follow the tab's single "Record run log" switch at the top of the page.
        from spacemap_gui import run_log
        create_log = run_log.enabled("tab3")
        if not input_dir:
            st.info("Please select an input directory to start.")
            return

        input_path = Path(input_dir)
        
        st.subheader("2. Image Selection")
        if input_path.exists():
            files = sorted([f for f in os.listdir(input_path) if is_supported_image_file(f)])
            
            def on_file_change():
                keys_to_clear = ['roi_channels', 'roi_colors', 'roi_meta', 'roi_mpp', 'roi_visible', 'roi_editor']
                for k in keys_to_clear:
                    if k in st.session_state:
                        del st.session_state[k]
                for k in list(st.session_state.keys()):
                    if isinstance(k, str) and (k.startswith('roi_color_') or k.startswith('roi_name_') or k.startswith('roi_cp_') or k.startswith('roi_vis_') or k.startswith('roi_export_')):
                        del st.session_state[k]

                # Avoid carrying prior selections/overlays across images
                for k in ['roi_geojson_select', 'tab3_load_existing_canvas', 'roi_drawn_editor', 'tab3_canvas_objects']:
                    if k in st.session_state:
                        del st.session_state[k]
            
            # Add placeholder
            files_with_placeholder = ["Select an image..."] + files
            selected_file_raw = st.selectbox("Select Image", files_with_placeholder, key="roi_file_select", on_change=on_file_change)
            
            if selected_file_raw == "Select an image...":
                selected_file = None
            else:
                selected_file = selected_file_raw
                if st.session_state.get("tab3_last_selected_file") != selected_file:
                    log_tab3(f"Selected image: {selected_file}")
                    st.session_state["tab3_last_selected_file"] = selected_file
        else:
            st.error("Directory not found")
        
        if selected_file:
            file_path = input_path / selected_file
            
            # 1. ROI Definition & Parameters (Moved Up)
            st.markdown("### ROI Definition")
            
            geojson_dir_str = st.session_state.get('tab3_geojson_dir', str(project_root / 'data/boundaries'))
            geojson_dir_path = Path(geojson_dir_str)
            geojson_files = []
            if geojson_dir_path.exists():
                geojson_files = sorted([f for f in os.listdir(geojson_dir_path) if f.endswith('.geojson')])

            # Logic to find match
            expected_index = 0
            if geojson_files:
                # Try prefix match first (robust)
                img_prefix = _norm_image_prefix(selected_file)
                found_match = False
                for i, g in enumerate(geojson_files):
                    if _norm_geojson_prefix(g) == img_prefix:
                        expected_index = i
                        found_match = True
                        break
                
                if not found_match:
                    # Fallback to stem match (legacy)
                    stem = Path(selected_file).stem
                    if stem.endswith('.ome'): stem = stem[:-4]
                    if stem.endswith('_merged'): stem = stem[:-7]
                    expected_g = f"{stem}_roi_bounding_boxes_coords.geojson"
                    if expected_g in geojson_files:
                        expected_index = geojson_files.index(expected_g)

            selected_geojson = st.selectbox("Select GeoJSON File", geojson_files, index=expected_index, key="roi_geojson_select")
            
            # Define geojson_path based on selection
            if selected_geojson:
                geojson_path = geojson_dir_path / selected_geojson
                geojson_matches_image = _norm_geojson_prefix(selected_geojson) == _norm_image_prefix(selected_file)
            else:
                geojson_path = None
                geojson_matches_image = False
                
            st.markdown("### Parameters")
            mpp_val = st.number_input(
                "MPP (µm/px)", 
                value=st.session_state.get('roi_mpp', {'codex': 0.5083})['codex'], 
                key="roi_mpp_input", 
                help="Microns per pixel. Defines physical scale of the image, critical for area calculations."
            )
            mpp = {'codex': mpp_val}
            
            # 2. Get Metadata (Fast)
            st.subheader("Channel Configuration")
            
            if 'roi_channels' not in st.session_state:
                log_tab3(f"Loading channel info for: {selected_file}")
                try:
                    channel_names, channel_colors, channel_meta, mpp = get_channel_info(str(file_path))
                    if channel_names:
                        st.session_state['roi_channels'] = channel_names
                        st.session_state['roi_colors'] = channel_colors
                        st.session_state['roi_meta'] = channel_meta
                        st.session_state['roi_mpp'] = mpp
                    else:
                        raise ValueError("No channels found via get_channel_info")
                except Exception as e:
                    # Fallback if metadata fails - we might need to load image to guess
                    st.warning(f"Metadata extraction failed: {e}. Loading full image to inspect.")
                    # In this case we might have to load everything blindly
                    st.session_state['roi_channels'] = [] # Will be populated later if needed

            # 2. Determine Visibility & Load List
            if 'roi_visible' not in st.session_state:
                st.session_state['roi_visible'] = {}

            channels = st.session_state.get('roi_channels', [])
            colors = st.session_state.get('roi_colors', [])
            meta = st.session_state.get('roi_meta', [])
            
            # Initialize Visibility Defaults
            if channels and not st.session_state['roi_visible']:
                he_indices = []
                dapi_index = -1
                
                for i, ch in enumerate(channels):
                    ch_lower = ch.lower()
                    if 'he' in ch_lower or 'hematoxylin' in ch_lower or 'eosin' in ch_lower:
                        he_indices.append(i)
                    if 'dapi' in ch_lower:
                        dapi_index = i
                    st.session_state['roi_visible'][i] = False # Default all off
                
                if he_indices:
                    for idx in he_indices:
                        st.session_state['roi_visible'][idx] = True
                elif dapi_index != -1:
                    st.session_state['roi_visible'][dapi_index] = True
                elif len(channels) > 0:
                    st.session_state['roi_visible'][0] = True

            # Calculate indices to load
            # Load all visible channels + DAPI (if exists) + HE (if exists)
            indices_to_load = set()
            
            # Add currently visible
            for i, vis in st.session_state['roi_visible'].items():
                if vis:
                    indices_to_load.add(i)
            
            # Add DAPI and HE to "preload" list (so they are ready if toggled)
            for i, ch in enumerate(channels):
                ch_lower = ch.lower()
                if 'dapi' in ch_lower or 'he' in ch_lower or 'hematoxylin' in ch_lower or 'eosin' in ch_lower:
                    indices_to_load.add(i)
            
            # 3. Load Data (Lazy/Partial)
            img_data_dict, full_shape, is_rgb = load_preview_channels(str(file_path), sorted(list(indices_to_load)))
            
            if not img_data_dict and not is_rgb:
                st.error("Error loading image data.")
                st.stop()

            # Channel Table
            with st.expander("Edit Channels & Visibility", expanded=True):
                # Header
                c_exp, c_show, c_name, c_hex, c_color, c_dye = st.columns([0.7, 0.7, 3, 2, 1, 1.5])
                c_exp.markdown("**Export**")
                c_show.markdown("**Show**")
                c_name.markdown("**Channel**")
                c_hex.markdown("**Hex**")
                c_color.markdown("**Color**")
                c_dye.markdown("**Dye**")
                
                for i, (ch, col, m) in enumerate(zip(channels, colors, meta)):
                    # Color state
                    color_key = f"roi_color_{i}"
                    hex_col = '#{:02x}{:02x}{:02x}'.format(*col) if isinstance(col, tuple) else col
                    if _unset_colour(st.session_state.get(color_key), hex_col):
                        st.session_state[color_key] = hex_col
                        st.session_state.pop(f"roi_cp_{i}", None)
                        
                    # Name state (a blank saved name is filled from the image again)
                    name_key = f"roi_name_{i}"
                    if not str(st.session_state.get(name_key) or "").strip():
                        st.session_state[name_key] = ch
                        
                    # Export state
                    export_key = f"roi_export_{i}"
                    if export_key not in st.session_state:
                        st.session_state[export_key] = True

                    c_exp, c_show, c_name, c_hex, c_color, c_dye = st.columns([0.7, 0.7, 3, 2, 1, 1.5])
                    
                    # Export Checkbox
                    c_exp.checkbox("##", key=export_key, label_visibility="collapsed")
                    
                    # Visibility Checkbox
                    vis_key = f"roi_vis_{i}"
                    is_visible = st.session_state['roi_visible'].get(i, False)
                    
                    # If user toggles this, st.rerun() happens, indices_to_load updates, data loads.
                    if c_show.checkbox("##", value=is_visible, key=vis_key, label_visibility="collapsed"):
                        st.session_state['roi_visible'][i] = True
                    else:
                        st.session_state['roi_visible'][i] = False

                    # Name Input
                    new_name = c_name.text_input(f"n_{i}", key=name_key, label_visibility="collapsed")
                    
                    # Color picker sync
                    def update_color(k=color_key, idx=i):
                        st.session_state[k] = st.session_state[f"roi_cp_{idx}"]

                    picker_key = f"roi_cp_{i}"
                    if picker_key not in st.session_state:
                        st.session_state[picker_key] = st.session_state[color_key]
                    c_color.color_picker(f"cp_{i}", key=picker_key, on_change=update_color, label_visibility="collapsed")
                    c_hex.text(st.session_state[color_key])
                    c_dye.text(m.get('dye', '-'))
                    
                    edited_channel_names.append(new_name)
                    edited_channel_colors.append(hex_to_rgb(st.session_state[color_key]))
                    edited_channel_meta.append(m)

            # ROI Definition (Moved Up - Removed)


            # Parameters (Moved Up - Removed)

            
            # Output Directory & Log Inputs removed (moved to top)


            # Check if user is in Draw mode with drawn ROIs
            current_mode = st.session_state.get('tab3_mode', 'View/Import')
            drawn_rois = st.session_state.get('tab3_canvas_objects', {}).get(selected_file, [])
            
            if current_mode == "Draw ROIs" and drawn_rois:
                st.warning(f"You have {len(drawn_rois)} drawn ROI(s). Click 'Save & Extract' to use them, or switch to View/Import mode to use existing GeoJSON.")
                extract_btn = st.button("Save & Extract Drawn ROIs", type="primary", help="Save drawn shapes to GeoJSON and extract ROIs.")
            else:
                extract_btn = st.button("Extract ROIs", type="primary", help="Extract ROI images based on selected GeoJSON.")

            if extract_btn:
                status_container = st.status("Processing...", expanded=True)
                # Construct Channel Config
                keep_list = []
                rename_map = {}
                color_map = {}
                
                for i, orig_name in enumerate(channels):
                    # Use Export checkbox state instead of Visibility
                    if st.session_state.get(f"roi_export_{i}", True):
                        keep_list.append(orig_name)
                        
                        # Rename
                        new_name = st.session_state.get(f"roi_name_{i}", orig_name)
                        if new_name != orig_name:
                            rename_map[orig_name] = new_name
                        
                        # Color
                        hex_col = st.session_state.get(f"roi_color_{i}")
                        if hex_col:
                            # Map NEW name to color (or old name, backend checks both)
                            color_map[new_name] = hex_to_rgb(hex_col)

                channel_config = {
                    'keep': keep_list,
                    'rename': rename_map,
                    'colors': color_map
                }

                # If in Draw mode with drawn ROIs, save them first
                geojson_to_use = selected_geojson
                if current_mode == "Draw ROIs" and drawn_rois:
                    log_tab3(f"Auto-saving {len(drawn_rois)} drawn ROIs before extraction")
                    # Get labels from session state (stored by data editor on previous render)
                    stored_labels = st.session_state.get('tab3_roi_labels', {}).get(selected_file, {})
                    if stored_labels:
                        stored_labels = {int(k): v for k, v in stored_labels.items() if str(k).isdigit()}
                        st.session_state['tab3_roi_labels'][selected_file] = stored_labels
                    log_tab3(f"Stored labels in session state: {stored_labels}")
                    
                    # If no stored labels, check data editor session state directly
                    if not stored_labels and 'roi_drawn_editor' in st.session_state:
                        # st.data_editor stores edits in session state
                        # On rerun after editing, we may need to reconstruct from edited_rows
                        log_tab3(f"No stored labels, checking roi_drawn_editor")
                    
                    labels = [stored_labels.get(i, f"ROI {i+1}") for i in range(len(drawn_rois))]
                    log_tab3(f"Using labels: {labels}")
                    
                    # Use stored scale factors from canvas (set in Draw ROIs mode)
                    save_scale_x = st.session_state.get('tab3_canvas_scale_x', full_shape[1] / 700)
                    save_scale_y = st.session_state.get('tab3_canvas_scale_y', full_shape[0] / 700)
                    
                    # Save to the ROI save directory
                    save_dir = st.session_state.get('roi_save_dir', str(geojson_dir_path))
                    save_drawn_rois_for_extraction(drawn_rois, labels, save_scale_x, save_scale_y, file_path, save_dir)
                    
                    # Update the geojson to use
                    stem = Path(file_path).stem
                    if stem.endswith('.ome'): stem = stem[:-4]
                    if stem.endswith('_merged'): stem = stem[:-7]
                    geojson_to_use = f"{stem}_roi_bounding_boxes_coords.geojson"
                    status_container.write(f"Saved {len(drawn_rois)} drawn ROIs to {geojson_to_use}")

                run_roi_extraction(
                    str(file_path), 
                    str(geojson_dir_path) if not (current_mode == "Draw ROIs" and drawn_rois) else st.session_state.get('roi_save_dir', str(geojson_dir_path)), 
                    output_dir, 
                    mpp,
                    channel_config,
                    geojson_filename=geojson_to_use,
                    status_container=status_container,
                    create_log=create_log
                )

    with col2:
        st.subheader("Preview")
        if selected_file and (img_data_dict or is_rgb):
            
            # Debug Info (Collapsed)
            with st.expander("Image Debug Info", expanded=False):
                st.write(f"Is RGB: {is_rgb}")
                if not is_rgb:
                    st.write(f"Loaded Channels: {list(img_data_dict.keys())}")

            # Contrast Controls
            c_cont, c_inv, c_blend = st.columns(3)
            enhance_contrast = c_cont.checkbox("Enhance Contrast", value=True, key="tab3_enhance_contrast")
            invert_colors = c_inv.checkbox("Invert Colors", value=False, key="tab3_invert_colors")
            blending_mode = c_blend.selectbox("Blending", ["Additive", "Screen"], index=0, key="tab3_blending")

            # Generate Composite Image based on visibility
            if is_rgb:
                img_to_show = img_data_dict['rgb'].copy()
                
                if invert_colors:
                    if img_to_show.dtype == np.uint8:
                        img_to_show = 255 - img_to_show
                    else:
                        img_to_show = np.nanmax(img_to_show) - img_to_show

                if enhance_contrast and img_to_show.dtype == np.uint8:
                     p_min, p_max = np.nanmin(img_to_show), np.nanmax(img_to_show)
                     if p_max - p_min < 200 and p_max > p_min: 
                         img_to_show = np.clip((img_to_show - p_min) / (p_max - p_min) * 255, 0, 255).astype(np.uint8)

                caption = "RGB Preview"
            else:
                # Composite
                # Determine shape from first loaded channel
                if not img_data_dict:
                    st.warning("No channels loaded.")
                    st.stop()
                    
                first_key = next(iter(img_data_dict))
                h, w = img_data_dict[first_key].shape
                composite = np.zeros((h, w, 3), dtype=np.float32)
                active_channels = []
                
                # Ensure channels is defined (it should be from col1)
                if 'roi_channels' in st.session_state:
                    channels = st.session_state['roi_channels']
                else:
                    channels = []
                
                for i in range(len(channels)):
                    if st.session_state['roi_visible'].get(i, False):
                        # Check if data is loaded
                        if i in img_data_dict:
                            ch_data = img_data_dict[i].astype(np.float32)
                            
                            if invert_colors:
                                ch_data = np.nanmax(ch_data) - ch_data

                            if ch_data.max() > 0:
                                if enhance_contrast:
                                    p99 = np.percentile(ch_data, 99)
                                    if p99 > 0:
                                        ch_data = np.clip(ch_data / p99, 0, 1)
                                    else:
                                        ch_data = ch_data / ch_data.max()
                                else:
                                    ch_data = ch_data / ch_data.max()
                            
                            color_key = f"roi_color_{i}"
                            if color_key in st.session_state:
                                hex_col = st.session_state[color_key]
                                color = hex_to_rgb(hex_col)
                            elif i < len(edited_channel_colors):
                                color = edited_channel_colors[i]
                            else:
                                color = (255, 255, 255)
                            
                            if blending_mode == "Additive":
                                for c in range(3):
                                    composite[:, :, c] += ch_data * color[c]
                            else: # Screen
                                for c in range(3):
                                    layer_val = ch_data * color[c]
                                    current_val = composite[:, :, c]
                                    composite[:, :, c] = current_val + layer_val - (current_val * layer_val) / 255.0
                                
                            if i < len(edited_channel_names):
                                active_channels.append(edited_channel_names[i])
                            else:
                                active_channels.append(f"Ch{i}")
                        else:
                            # Data not loaded yet (shouldn't happen if logic is correct)
                            pass
                
                composite = np.clip(composite, 0, 255).astype(np.uint8)
                img_to_show = composite
                caption = f"Composite: {', '.join(active_channels)}" if active_channels else "No channels selected"

            # Convert to PIL
            pil_img = Image.fromarray(img_to_show)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            full_h, full_w = full_shape[0], full_shape[1]
            disp_w, disp_h = pil_img.size
            scale_factor_x = full_w / disp_w
            scale_factor_y = full_h / disp_h
            
            st.caption(f"{caption}. Scale: 1:{scale_factor_x:.1f}")

            # Mode Selection
            mode = st.radio("Mode", ["View/Import", "Draw ROIs"], horizontal=True, key="tab3_mode")
            if st.session_state.get("tab3_last_mode") != mode:
                log_tab3(f"Mode: {mode}")
                st.session_state["tab3_last_mode"] = mode

            if mode == "View/Import":
                # Overlay GeoJSON
                overlay_img = pil_img.copy()
                roi_data = []
                features = []
                geo_data = {}
                
                if "tab3_show_overlay" not in st.session_state:
                    st.session_state["tab3_show_overlay"] = True
                show_overlay = st.checkbox("Load & Overlay ROIs", key="tab3_show_overlay")

                if show_overlay and geojson_path and geojson_path.exists() and not geojson_matches_image:
                    st.caption("GeoJSON does not match selected image prefix; overlay disabled.")

                if show_overlay and geojson_path and geojson_path.exists() and geojson_matches_image:
                    try:
                        with open(str(geojson_path), 'r') as f:
                            geo_data = json.load(f)
                        
                        # Use the new geometry extraction to handle holes/polygons correctly
                        rois = extract_rois_with_geometry_from_geojson(geo_data)
                        
                        # Draw on overlay
                        # We need to convert PIL to numpy for cv2 drawing if we want to support holes easily,
                        # or use PIL ImageDraw with care.
                        # Let's use cv2 for robust polygon drawing (including holes) then convert back.
                        
                        overlay_np = np.array(overlay_img)
                        # Make sure it's writable
                        overlay_np = overlay_np.copy()
                        
                        for i, (bbox, label, geometry) in enumerate(rois):
                            roi_data.append({"ROI ID": i+1, "Label": label})
                            
                            # Scale geometry points
                            def scale_coords(coords):
                                return np.array(coords) / [scale_factor_x, scale_factor_y]

                            if geometry['type'] == 'Polygon':
                                rings = geometry['coordinates']
                                if not rings: continue
                                
                                # Exterior
                                ext = scale_coords(rings[0]).astype(np.int32)
                                cv2.polylines(overlay_np, [ext], True, (255, 0, 0), 2)
                                
                                # Holes (draw in different color or thinner?)
                                for ring in rings[1:]:
                                    hole = scale_coords(ring).astype(np.int32)
                                    cv2.polylines(overlay_np, [hole], True, (255, 255, 0), 1)
                                    
                            elif geometry['type'] == 'MultiPolygon':
                                for poly_coords in geometry['coordinates']:
                                    if not poly_coords: continue
                                    rings = poly_coords
                                    ext = scale_coords(rings[0]).astype(np.int32)
                                    cv2.polylines(overlay_np, [ext], True, (255, 0, 0), 2)
                                    for ring in rings[1:]:
                                        hole = scale_coords(ring).astype(np.int32)
                                        cv2.polylines(overlay_np, [hole], True, (255, 255, 0), 1)
                        
                        overlay_img = Image.fromarray(overlay_np)
                        st.caption(f"Overlaying ROIs from {geojson_path.name}")
                        
                    except Exception as e:
                        st.warning(f"Could not overlay GeoJSON: {e}")

                st.image(overlay_img, width=500)
                
                if roi_data:
                    st.markdown("### ROI Labels")
                    # We need to map edits back to the original features list order
                    # extract_rois_with_geometry_from_geojson returns ROIs in order of features usually
                    
                    # roi_data is already a list of dicts: {"ROI ID": i+1, "Label": label}
                    df = pd.DataFrame(roi_data)
                    edited_df = st.data_editor(df, key="roi_editor", hide_index=True)
                    
                    if st.button("Update Labels"):
                        if not geojson_path:
                            st.error("No GeoJSON selected.")
                            st.stop()
                        # Reload original to be safe
                        with open(str(geojson_path), 'r') as f:
                            geo_data = json.load(f)
                        features = geo_data.get("features", [])
                        
                        # We assume 1-to-1 mapping if we didn't filter features in extraction
                        # extract_rois_with_geometry_from_geojson skips invalid geometries.
                        # This might cause index mismatch if some features were skipped.
                        # But for now, let's assume valid GeoJSON.
                        
                        # Better approach: Use the label to find the feature? No, labels change.
                        # We should rely on the index in the dataframe matching the index in the extraction list.
                        # And we need to map extraction list index back to feature index.
                        # extract_rois... doesn't return original index.
                        
                        # Simple fix: Iterate features again and update if it matches logic.
                        # Or just update all features assuming they match.
                        
                        new_labels = edited_df["Label"].tolist()
                        
                        # Re-extract to get mapping
                        # This is inefficient but safe
                        current_rois = extract_rois_with_geometry_from_geojson(geo_data)
                        
                        if len(new_labels) == len(current_rois):
                            # We need to know which feature index corresponds to which ROI
                            # Let's manually iterate features to match the extraction logic
                            roi_idx = 0
                            for i, feature in enumerate(features):
                                geom = feature.get('geometry', {})
                                # Check if valid (same logic as utils)
                                coords = []
                                if geom.get('type') in ['Polygon', 'MultiPolygon']:
                                    # It was extracted
                                    if roi_idx < len(new_labels):
                                        if "properties" not in feature:
                                            feature["properties"] = {}
                                        feature["properties"]["label"] = new_labels[roi_idx]
                                        roi_idx += 1
                            
                            with open(str(geojson_path), 'w') as f:
                                json.dump(geo_data, f)
                            st.success("Labels updated!")
                            st.rerun()
                        else:
                            st.error("Mismatch in ROI counts. Cannot update safely.")

            elif mode == "Draw ROIs":
                if not HAS_CANVAS:
                    st.error("streamlit-drawable-canvas is not installed.")
                else:
                    st.info("Left-click to add points. Right-click to close polygon. Double-click to remove.")
                    
                    # Save Directory Selection
                    if 'roi_save_dir' not in st.session_state:
                        st.session_state['roi_save_dir'] = st.session_state.get('geojson_dir', str(project_root / 'data/boundaries'))

                    save_dir = directory_input_with_browse(
                        label="Save Directory",
                        text_key="roi_save_dir",
                        default_value=st.session_state.get('geojson_dir', str(project_root / 'data/boundaries')),
                        browser_key_prefix="tab3_roi_save_dir",
                        start_dir_fallback=str(project_root),
                    )

                    # Prepare initial drawing from existing GeoJSON if desired
                    initial_drawing = None
                    if geojson_path and geojson_path.exists() and geojson_matches_image:
                        if "tab3_load_existing_canvas" not in st.session_state:
                            st.session_state["tab3_load_existing_canvas"] = False
                        if st.checkbox("Load existing ROIs into canvas", key="tab3_load_existing_canvas"):
                            try:
                                with open(str(geojson_path), 'r') as f:
                                    geo_data = json.load(f)
                                initial_drawing, loaded_labels = geojson_to_fabric_with_labels(
                                    geo_data,
                                    scale_x=1 / scale_factor_x,
                                    scale_y=1 / scale_factor_y,
                                )

                                # Seed label state from GeoJSON so names persist.
                                if 'tab3_roi_labels' not in st.session_state:
                                    st.session_state['tab3_roi_labels'] = {}
                                existing = st.session_state['tab3_roi_labels'].get(selected_file, {})
                                if not existing and loaded_labels:
                                    st.session_state['tab3_roi_labels'][selected_file] = {
                                        i: str(lbl) for i, lbl in enumerate(loaded_labels)
                                    }
                            except Exception as e:
                                st.warning(f"Could not load existing ROIs: {e}")

                    # Persist per-image canvas objects so table edits can delete/highlight ROIs
                    if 'tab3_canvas_objects' not in st.session_state:
                        st.session_state['tab3_canvas_objects'] = {}
                    
                    # Track which ROIs are marked for deletion (by signature)
                    if 'tab3_deleted_rois' not in st.session_state:
                        st.session_state['tab3_deleted_rois'] = {}
                    if selected_file not in st.session_state['tab3_deleted_rois']:
                        st.session_state['tab3_deleted_rois'][selected_file] = set()

                    # Canvas
                    # We need a unique key for the canvas to reset when file changes
                    canvas_key = f"canvas_{selected_file}"
                    
                    # Resize image for canvas to avoid huge dimensions
                    # We use the same pil_img which is already scaled for display in st.image usually?
                    # No, pil_img is created from img_to_show which is full resolution of the loaded data (which might be subsampled).
                    # Let's ensure it fits in the column.
                    
                    # Calculate display width (approximate column width)
                    display_width = 700 # Standard Streamlit column width estimate
                    
                    canvas_width = pil_img.width
                    canvas_height = pil_img.height
                    
                    # If image is too wide, scale it down for canvas
                    if canvas_width > display_width:
                        ratio = display_width / canvas_width
                        canvas_width = display_width
                        canvas_height = int(pil_img.height * ratio)
                        
                    # We need to pass the image to the canvas. 
                    # If we resize, we must adjust scale factors for saving!
                    # Current scale_factor_x = full_w / disp_w (where disp_w is pil_img.width)
                    # If we resize pil_img for canvas, we introduce another scale factor.
                    
                    canvas_bg = pil_img.resize((canvas_width, canvas_height))
                    
                    # Update scale factors to include this resize
                    # Total scale = (full_w / pil_img.width) * (pil_img.width / canvas_width) = full_w / canvas_width
                    final_scale_x = full_w / canvas_width
                    final_scale_y = full_h / canvas_height
                    
                    # Store scale factors in session state for extraction button
                    st.session_state['tab3_canvas_scale_x'] = final_scale_x
                    st.session_state['tab3_canvas_scale_y'] = final_scale_y
                    
                    # Ensure RGB
                    if canvas_bg.mode != "RGB":
                        canvas_bg = canvas_bg.convert("RGB")
                    
                    # WORKAROUND: Embed image as a Fabric.js object to ensure visibility.
                    # Cache per-image+size to avoid re-encoding base64 on every rerun.
                    if "tab3_canvas_bg_cache" not in st.session_state:
                        st.session_state["tab3_canvas_bg_cache"] = {}
                    bg_cache_key = (selected_file, canvas_width, canvas_height)
                    if bg_cache_key not in st.session_state["tab3_canvas_bg_cache"]:
                        bg_b64 = image_to_base64(canvas_bg)
                        st.session_state["tab3_canvas_bg_cache"][bg_cache_key] = {
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

                    bg_obj = st.session_state["tab3_canvas_bg_cache"][bg_cache_key]

                    if initial_drawing is None:
                        initial_drawing = {"version": "4.4.0", "objects": []}

                    # Build initial_drawing without mutating/duplicating background.
                    existing_objects = initial_drawing.get("objects", []) or []
                    non_image_objects = [o for o in existing_objects if isinstance(o, dict) and o.get("type") != "image"]
                    initial_drawing = {"version": "4.4.0", "objects": [bg_obj] + non_image_objects}
                    
                    # --- UNIFIED DRAWING TOOLBAR ---
                    # We use a session state variable to track if we are in "Draw Mode"
                    # Default is "Edit/Select" (transform)
                    
                    if 'tab3_draw_active' not in st.session_state:
                        st.session_state['tab3_draw_active'] = False
                        
                    # Toolbar UI
                    c_tool_btn, c_tool_info = st.columns([1, 4])

                    draw_active = bool(st.session_state.get('tab3_draw_active', False))
                    toggle_label = "Stop Drawing" if draw_active else "Draw New ROI"
                    toggle_help = "Switch back to Edit/Select mode" if draw_active else "Switch to Polygon Drawing mode"
                    toggle_type = "primary" if draw_active else "secondary"
                    if c_tool_btn.button(toggle_label, type=toggle_type, help=toggle_help, key=f"tab3_draw_toggle_{selected_file}"):
                        draw_active = not draw_active
                        st.session_state['tab3_draw_active'] = draw_active

                    if draw_active:
                        c_tool_info.info("Left-click to add points. Right-click to close polygon.")
                        drawing_mode = "polygon"
                    else:
                        c_tool_info.info("Click ROI to select/edit. Drag to move. Press Delete/Backspace to remove.")
                        drawing_mode = "transform"

                    # Canvas Key with Reset
                    if 'canvas_reset_count' not in st.session_state:
                        st.session_state['canvas_reset_count'] = 0
                        
                    # Logic to handle "Hard Delete" from table
                    # If a delete happened in the table, we need to force a canvas reset
                    # to visually remove the object.
                    if st.session_state.get('tab3_force_canvas_reset', False):
                        st.session_state['canvas_reset_count'] += 1
                        st.session_state['tab3_force_canvas_reset'] = False
                        # We do NOT clear canvas_objects here, because we want to keep the remaining ones!
                        # But we DO need to rebuild initial_drawing from the remaining objects.
                        # This happens automatically below because we use tab3_canvas_objects to seed initial_drawing
                        # if we are not in a fresh state.
                        
                        # Wait, initial_drawing is only used on mount (new key).
                        # So incrementing key is correct.
                        # But we need to ensure initial_drawing contains the CURRENT valid objects.
                        
                        # Rebuild initial_drawing from session state objects
                        current_objs = st.session_state.get('tab3_canvas_objects', {}).get(selected_file, [])
                        
                        # We need to convert these back to Fabric JSON format if they aren't already?
                        # They are stored as Fabric objects from json_data.
                        
                        # Re-insert background
                        new_initial = {"version": "4.4.0", "objects": [bg_obj] + current_objs}
                        initial_drawing = new_initial
                        
                    
                    if st.button("Reset Canvas (Reload Image)"):
                        st.session_state['canvas_reset_count'] += 1
                        # Clear everything
                        if selected_file in st.session_state.get('tab3_canvas_objects', {}):
                            del st.session_state['tab3_canvas_objects'][selected_file]
                        if selected_file in st.session_state.get('tab3_roi_labels', {}):
                            del st.session_state['tab3_roi_labels'][selected_file]
                        if selected_file in st.session_state.get('tab3_deleted_rois', {}):
                            st.session_state['tab3_deleted_rois'][selected_file] = set()
                    
                    # If we have stored objects and we are resetting/reloading, use them
                    # This ensures that when we increment key (hard delete), we restore the valid objects
                    # BUT: If we are just re-rendering (no reset), we should NOT modify initial_drawing
                    # because st_canvas ignores it unless the key changes.
                    # However, if we are in a loop where st_canvas is re-mounting, we might need this.
                    # The issue is that if we append here, we might duplicate if initial_drawing already has them.
                    
                    if selected_file in st.session_state.get('tab3_canvas_objects', {}) and len(initial_drawing["objects"]) == 1:
                         # initial_drawing only has bg_obj
                         saved = st.session_state['tab3_canvas_objects'][selected_file]
                         # Deep copy to avoid reference issues
                         import copy
                         saved_copy = copy.deepcopy(saved)
                         initial_drawing["objects"].extend(saved_copy)

                    # Use a static key unless forced reset
                    # This prevents the canvas from re-mounting on every interaction
                    canvas_key = f"canvas_{selected_file}_{st.session_state['canvas_reset_count']}"

                    # Put the canvas inside a form so drawing does NOT trigger reruns.
                    # The user explicitly syncs the latest canvas state.
                    st.caption("Draw your ROIs, then click 'Sync Canvas' to register them.")
                    with st.form(key=f"tab3_canvas_form_{selected_file}_{st.session_state['canvas_reset_count']}"):
                        _canvas = cast(Any, st_canvas)
                        canvas_result = _canvas(
                            fill_color="rgba(255, 165, 0, 0.3)",
                            stroke_width=2,
                            stroke_color="#000000",
                            update_streamlit=True,
                            height=canvas_height,
                            width=canvas_width,
                            drawing_mode=drawing_mode,
                            initial_drawing=initial_drawing,
                            key=canvas_key,
                            display_toolbar=True,
                        )
                        sync_clicked = st.form_submit_button(
                            "Sync Canvas",
                            type="primary",
                            help="Click after drawing to register your ROIs",
                        )

                    def _extract_path_points(obj):
                        pts = []
                        for cmd in obj.get('path', []) or []:
                            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ('M', 'L'):
                                pts.append((float(cmd[1]), float(cmd[2])))
                        dedup = []
                        for p in pts:
                            if not dedup or p != dedup[-1]:
                                dedup.append(p)
                        return dedup

                    def _is_valid_polygon_obj(obj):
                        if obj.get('type') != 'path':
                            return False
                        pts = _extract_path_points(obj)
                        if len(pts) < 3:
                            return False
                        try:
                            return Polygon(pts).area > 0.0
                        except Exception:
                            return False

                    def _sig(obj):
                        # Signature used for de-dup/merge. Prefer the full path.
                        try:
                            return json.dumps(obj.get('path', []), separators=(',', ':'), ensure_ascii=False)
                        except Exception:
                            return str(obj.get('path', []))

                    # On explicit sync, parse canvas JSON and update stored objects.
                    if sync_clicked:
                        if canvas_result is None or canvas_result.json_data is None:
                            st.warning("Nothing to sync yet.")
                        else:
                            raw_objects = canvas_result.json_data.get("objects", [])
                            objects = [obj for obj in raw_objects if isinstance(obj, dict) and obj.get("type") != "image"]
                            log_tab3(f"Sync: Canvas returned {len(raw_objects)} total objects, {len(objects)} non-image objects")

                            valid_objects = []
                            seen = set()
                            for obj in objects:
                                if not _is_valid_polygon_obj(obj):
                                    continue
                                s = _sig(obj)
                                if s in seen:
                                    continue
                                seen.add(s)
                                valid_objects.append(obj)

                            st.session_state['tab3_canvas_objects'][selected_file] = valid_objects
                            log_tab3(f"Sync: Stored {len(valid_objects)} valid polygon ROIs")

                    # Render table from stored objects (stable across reruns)
                    stored_objects = st.session_state.get('tab3_canvas_objects', {}).get(selected_file, [])
                    valid_objects = []
                    seen = set()
                    for obj in stored_objects:
                        if not isinstance(obj, dict):
                            continue
                        if not _is_valid_polygon_obj(obj):
                            continue
                        s = _sig(obj)
                        if s in seen:
                            continue
                        seen.add(s)
                        valid_objects.append((obj, s))

                    if valid_objects:
                        st.markdown("### Drawn ROIs")

                        # Get existing labels from session state or create defaults
                        existing_labels = st.session_state.get('tab3_roi_labels', {}).get(selected_file, {})
                        if existing_labels:
                            existing_labels = {int(k): v for k, v in existing_labels.items() if str(k).isdigit()}
                            st.session_state['tab3_roi_labels'][selected_file] = existing_labels

                        # Initialize labels in session state
                        if 'tab3_roi_labels' not in st.session_state:
                            st.session_state['tab3_roi_labels'] = {}
                        if selected_file not in st.session_state['tab3_roi_labels']:
                            st.session_state['tab3_roi_labels'][selected_file] = {}

                        # Table header
                        h_id, h_label, h_del = st.columns([1, 4, 1])
                        h_id.markdown("**#**")
                        h_label.markdown("**Label**")
                        h_del.markdown("**Del**")

                        # Track which ROI to delete (if any)
                        idx_to_delete = -1
                        updated_labels = {}

                        # --- Smart Label Generation ---
                        # Ensure new ROIs get the first available "ROI X" label
                        final_labels = {}
                        used_names = set()
                        indices_to_assign = []

                        # 1. Lock in existing labels
                        for i in range(len(valid_objects)):
                            if i in existing_labels:
                                lbl = existing_labels[i]
                                final_labels[i] = lbl
                                used_names.add(lbl)
                            else:
                                indices_to_assign.append(i)

                        # 2. Assign new labels to missing indices
                        for i in indices_to_assign:
                            k = 1
                            while f"ROI {k}" in used_names:
                                k += 1
                            new_lbl = f"ROI {k}"
                            final_labels[i] = new_lbl
                            used_names.add(new_lbl)
                        # -----------------------------

                        for i, (obj, sig) in enumerate(valid_objects):
                            default_label = final_labels[i]

                            c_id, c_label, c_del = st.columns([1, 4, 1])
                            c_id.text(f"{i + 1}")

                            # Label input with unique key
                            label_key = f"roi_label_{selected_file}_{i}"
                            new_label = c_label.text_input(
                                f"label_{i}",
                                value=default_label,
                                key=label_key,
                                label_visibility="collapsed",
                            )
                            updated_labels[i] = new_label

                            # Delete button - Hard Delete
                            if c_del.button("Del", key=f"roi_del_{selected_file}_{i}", help=f"Delete ROI {i+1}"):
                                idx_to_delete = i

                        # Store updated labels
                        st.session_state['tab3_roi_labels'][selected_file] = updated_labels

                        # Handle deletion if requested
                        if idx_to_delete != -1:
                            log_tab3(f"Deleting ROI index {idx_to_delete}")
                            current_list = st.session_state['tab3_canvas_objects'][selected_file]
                            if idx_to_delete < len(current_list):
                                current_list.pop(idx_to_delete)
                                st.session_state['tab3_canvas_objects'][selected_file] = current_list

                                # Shift labels
                                new_labels = {}
                                for k, v in updated_labels.items():
                                    if k < idx_to_delete:
                                        new_labels[k] = v
                                    elif k > idx_to_delete:
                                        new_labels[k - 1] = v
                                st.session_state['tab3_roi_labels'][selected_file] = new_labels

                                # Trigger Hard Reset (needs an extra rerun because the reset logic is above)
                                st.session_state['tab3_force_canvas_reset'] = True
                                st.rerun()

                        if st.button("Save Drawn ROIs", width='stretch'):
                            objs_to_save = [obj for obj, _ in valid_objects]
                            labels = [updated_labels.get(i, f"ROI {i+1}") for i in range(len(objs_to_save))]
                            log_tab3(f"Saving {len(objs_to_save)} ROIs with labels: {labels}")
                            if file_path is None:
                                st.error("No image file selected; cannot save ROIs.")
                            else:
                                save_drawn_rois(
                                    objs_to_save,
                                    labels,
                                    final_scale_x,
                                    final_scale_y,
                                    file_path,
                                    st.session_state['roi_save_dir'],
                                )
                                st.rerun()

def detect_tma_cores(
    image: Image.Image,
    min_size: int,
    max_size: int,
    threshold_val: int,  # 0 for Otsu
    row_threshold_factor: float,
    rows_input: int,
    cols_input: int,
    row_format: str,     # "A-Z" or "1-N"
    col_format: str,     # "1-N"
) -> tuple[list[dict], list[str]]:
    # Convert PIL Image to OpenCV
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Smooth image
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Determine background type (dark vs light)
    mean_val = np.mean(gray)
    is_light_bg = mean_val > 127
    
    if is_light_bg:
        processed = 255 - blurred
    else:
        processed = blurred
        
    # Thresholding
    if threshold_val == 0:
        _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(processed, threshold_val, 255, cv2.THRESH_BINARY)
        
    # Clean up binary mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and collect bounding boxes
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Check dimensions
        if min_size <= w <= max_size and min_size <= h <= max_size:
            # Aspect ratio check
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:
                boxes.append((x, y, w, h))
                
    if not boxes:
        return [], []
        
    # Sort boxes: Non-overlapping filter
    # Sort boxes by area descending, and keep boxes only if they don't overlap significantly
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    filtered_boxes = []
    for box in boxes:
        overlap = False
        bx, by, bw, bh = box
        boxA = [bx, by, bx + bw, by + bh]
        for f_box in filtered_boxes:
            fx, fy, fw, fh = f_box
            boxB = [fx, fy, fx + fw, fy + fh]
            
            # Intersection
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            inter = max(0, xB - xA) * max(0, yB - yA)
            if inter > 0:
                areaA = bw * bh
                areaB = fw * fh
                if inter / areaA > 0.1 or inter / areaB > 0.1:
                    overlap = True
                    break
        if not overlap:
            filtered_boxes.append(box)
            
    if not filtered_boxes:
        return [], []
        
    # Sort bounding boxes top-to-bottom and left-to-right to assign grid labels
    centers = [(b[0] + b[2]/2.0, b[1] + b[3]/2.0, b) for b in filtered_boxes]
    centers = sorted(centers, key=lambda c: c[1])
    
    # Group into rows
    avg_h = np.mean([b[3] for b in filtered_boxes])
    row_gap_thresh = avg_h * row_threshold_factor
    
    rows = []
    current_row = []
    for c in centers:
        if not current_row:
            current_row.append(c)
        else:
            prev_y = current_row[-1][1]
            if c[1] - prev_y > row_gap_thresh:
                rows.append(current_row)
                current_row = [c]
            else:
                current_row.append(c)
    if current_row:
        rows.append(current_row)
        
    # Sort each row by X center
    for r in rows:
        r.sort(key=lambda c: c[0])
        
    # Generate labels and final rect objects
    detected_objs = []
    detected_labels = []
    
    for r_idx, row in enumerate(rows):
        if row_format == "A-Z" and r_idx < 26:
            row_lbl = chr(65 + r_idx)
        else:
            # Numbered rows need a separator: "R1" + "12" and "R11" + "2" would
            # both read "R112" and two cores would share one ROI name.
            row_lbl = f"R{r_idx+1}_C"

        for c_idx, (_, _, box) in enumerate(row):
            col_lbl = str(c_idx + 1)
            label = f"{row_lbl}{col_lbl}"
            
            bx, by, bw, bh = box
            obj = {
                "type": "rect",
                "left": int(bx),
                "top": int(by),
                "width": int(bw),
                "height": int(bh),
                "fill": "rgba(255, 165, 0, 0.3)",
                "stroke": "#00ff00",
                "strokeWidth": 2,
                "scaleX": 1.0,
                "scaleY": 1.0,
            }
            detected_objs.append(obj)
            detected_labels.append(label)
            
    return detected_objs, detected_labels


def draw_labels_on_image(image: Image.Image, objects: list, labels_dict: dict) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
        
    for i, obj in enumerate(objects):
        label = labels_dict.get(i, f"ROI {i + 1}")
        
        left, top = None, None
        obj_type = obj.get("type", "")
        
        if obj_type == "rect":
            left = obj.get("left", 0)
            top = obj.get("top", 0)
        elif obj_type == "path":
            path_cmds = obj.get("path", [])
            xs = [cmd[1] for cmd in path_cmds if len(cmd) > 2 and cmd[0] in ['M', 'L']]
            ys = [cmd[2] for cmd in path_cmds if len(cmd) > 2 and cmd[0] in ['M', 'L']]
            if xs and ys:
                left = min(xs)
                top = min(ys)
        elif obj_type == "polygon":
            points = obj.get("points", [])
            xs = [p.get("x", 0) for p in points]
            ys = [p.get("y", 0) for p in points]
            obj_left = obj.get("left", 0)
            obj_top = obj.get("top", 0)
            scale_x = obj.get("scaleX", 1.0)
            scale_y = obj.get("scaleY", 1.0)
            path_offset = obj.get("pathOffset", {"x": 0, "y": 0})
            ox = path_offset.get("x", 0)
            oy = path_offset.get("y", 0)
            if xs and ys:
                abs_xs = [(x - ox) * scale_x + obj_left for x in xs]
                abs_ys = [(y - oy) * scale_y + obj_top for y in ys]
                left = min(abs_xs)
                top = min(abs_ys)
                
        if left is not None and top is not None:
            draw.text((left + 3, top + 3), label, fill=(255, 0, 0), font=font)
            
    return img


def _derive_clean_name(filename: str, prefix: str, suffix: str) -> str:
    stem = Path(filename).stem
    if stem.endswith('.ome'):
        stem = stem[:-4]
    
    stem = stem.lower()
    prefix = prefix.lower()
    suffix = suffix.lower()
    
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    if suffix and stem.endswith(suffix):
        stem = stem[:-len(suffix)]
        
    return stem.strip()


def _friendly_shape_type(obj: dict[str, Any]) -> str:
    raw = obj.get("type") or ""
    if raw in ("path", "polygon"):
        return "Polygon"
    if raw == "rect":
        return "Rectangle"
    return str(raw).capitalize() if raw else "?"


def _unset_colour(saved, image_colour) -> bool:
    """True when a marker's colour should be taken from the image.

    Missing, or black while the image gives the marker a real colour: black
    would hide the marker in every viewer, and is what an interrupted save
    leaves behind.
    """
    text = str(saved or "").strip().lower()
    if not text:
        return True
    return text in ("#000000", "#000") and str(image_colour or "").strip().lower() not in ("#000000", "#000", "")


def _follow_pipeline_input() -> None:
    """Images Directory follows the last step that produced images (Tab 2, else Tab 1).

    A folder chosen by hand is kept; Tab 3's own output folder never is.
    See spacemap_gui.pipeline_folders.
    """
    from spacemap_gui.pipeline_folders import follow_pipeline

    mappings = dict((st.session_state.get("active_project") or {}).get("directory_mappings") or {})
    mappings.update({key: value for key, value in st.session_state.items()
                     if isinstance(key, str) and key.endswith("_dir") and not key.startswith("proj_dir_")
                     and isinstance(value, str)})
    current = st.session_state.get("tab3_multi_input_dir") or st.session_state.get("tab3_input_dir")
    value, note = follow_pipeline(current, mappings)
    if note:
        st.session_state["tab3_multi_input_dir"] = value
        st.session_state["tab3_input_dir"] = value
        log_tab3(note)
        st.caption(note)


def render_multiple_images(project_root):
    c_head = st.columns([1])[0]
    c_head.header("Multiple Images")
    st.markdown("Pair images with GeoJSONs. Configure channels on the left, edit ROIs on the right.")

    defaults = {
        'tab3_multi_input_dir': str(project_root),
        'tab3_multi_geojson_dir': str(project_root / 'data/boundaries'),
        'tab3_multi_output_dir': str(project_root / 'outputs/rois'),
        'tab3_multi_mpp': float(st.session_state.get('roi_mpp', {'codex': 0.5083}).get('codex', 0.5083)),
        'tab3_multi_export_markers': {},
        'tab3_multi_active_img': None,
        'tab3_multi_canvas_objects': {},
        'tab3_multi_roi_labels': {},
        'tab3_multi_canvas_reset': {},
        'tab3_multi_geojson_loaded_mtime': {},
        'tab3_multi_visible_channels': {},
        'tab3_multi_img_prefix': '',
        'tab3_multi_img_suffix': '_merged',
        'tab3_multi_geo_prefix': '',
        'tab3_multi_geo_suffix': '_roi_bounding_boxes_coords',
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    # Helper: Refresh Pairs
    def _refresh_pairs(input_dir: str, geojson_dir: str) -> list[dict[str, Any]]:
        input_path = Path(str(input_dir))
        geo_path = Path(str(geojson_dir))
        if not input_path.exists() or not geo_path.exists():
            return []

        img_files = sorted([f for f in os.listdir(input_path) if is_supported_image_file(f)])
        geo_files = sorted([f for f in os.listdir(geo_path) if f.endswith('.geojson')])

        img_prefix = st.session_state.get('tab3_multi_img_prefix', '')
        img_suffix = st.session_state.get('tab3_multi_img_suffix', '_merged')
        geo_prefix = st.session_state.get('tab3_multi_geo_prefix', '')
        geo_suffix = st.session_state.get('tab3_multi_geo_suffix', '_roi_bounding_boxes_coords')

        geo_by_clean: dict[str, str] = {}
        for g in geo_files:
            clean_geo = _derive_clean_name(g, geo_prefix, geo_suffix)
            geo_by_clean.setdefault(clean_geo, g)

        pairs: list[dict[str, Any]] = []
        for img in img_files:
            override_key = f"tab3_multi_override_{img}"
            override_val = st.session_state.get(override_key, "<Auto>")
            
            if override_val == "<Skip>":
                continue
                
            # Determine the active method for this image
            method_key = f"tab3_boundary_method_{img}"
            pair_settings = st.session_state.get('tab3_multi_pair_settings', {}).get(img, {})
            
            current_method = st.session_state.get(method_key) or pair_settings.get('boundary_method')
            if not current_method:
                # Default logic if not selected yet
                clean_img = _derive_clean_name(img, img_prefix, img_suffix)
                temp_gj = geo_by_clean.get(clean_img)
                if temp_gj and (geo_path / temp_gj).exists():
                    current_method = "Predefined GeoJSON boundaries"
                else:
                    current_method = "Polygon (left-click vertices, right-click to close)"
            
            chosen_geojson = None
            is_new_boundary = False
            
            if current_method == "Predefined GeoJSON boundaries":
                if override_val == "<Auto>":
                    clean_img = _derive_clean_name(img, img_prefix, img_suffix)
                    chosen_geojson = geo_by_clean.get(clean_img)
                else:
                    if override_val in geo_files:
                        chosen_geojson = override_val
            else:
                # Prioritize new_boundaries subfolder
                stem = Path(img).stem
                if stem.endswith('.ome'): stem = stem[:-4]
                if stem.endswith('_merged'): stem = stem[:-7]
                custom_name = f"{stem}_roi_bounding_boxes_coords.geojson"
                
                custom_path = geo_path / "new_boundaries" / custom_name
                if custom_path.exists():
                    chosen_geojson = custom_name
                    is_new_boundary = True
            
            if chosen_geojson:
                if is_new_boundary:
                    geojson_path = geo_path / "new_boundaries" / chosen_geojson
                else:
                    geojson_path = geo_path / chosen_geojson
                    
                roi_count = _count_rois_in_geojson(geojson_path)
                pairs.append({
                    'image': img, 
                    'geojson': chosen_geojson, 
                    'roi_count': roi_count, 
                    'is_new_boundary': is_new_boundary
                })
            else:
                pairs.append({
                    'image': img, 
                    'geojson': None, 
                    'roi_count': 0, 
                    'is_new_boundary': False
                })
        return pairs

    # State Initialization for Per-Pair Settings
    if 'tab3_multi_pair_settings' not in st.session_state:
        st.session_state['tab3_multi_pair_settings'] = {}

    col1, col2 = st.columns([1, 1])

    # --- LEFT COLUMN ---
    with col1:
        st.subheader("1. Directories")
        _follow_pipeline_input()
        input_dir = directory_input_with_browse(
            label="Images Directory", text_key="tab3_multi_input_dir", default_value=str(project_root),
            browser_key_prefix="tab3_multi_input_dir", start_dir_fallback=str(project_root)
        )
        st.session_state["tab3_input_dir"] = input_dir
        
        geojson_dir = directory_input_with_browse(
            label="GeoJSON Directory", text_key="tab3_multi_geojson_dir", default_value=str(project_root / 'data/boundaries'),
            browser_key_prefix="tab3_multi_geojson_dir", start_dir_fallback=str(project_root)
        )
        st.session_state["tab3_geojson_dir"] = geojson_dir
        
        output_dir = directory_input_with_browse(
            label="Output Directory", text_key="tab3_multi_output_dir", default_value=str(project_root / 'outputs/rois'),
            browser_key_prefix="tab3_multi_output_dir", start_dir_fallback=str(project_root)
        )
        st.session_state["tab3_output_dir"] = output_dir
        # Detail logs follow the tab's single "Record run log" switch at the top of the page.
        from spacemap_gui import run_log
        create_log = run_log.enabled("tab3")
        

        
        # Ensure output dir exists
        od = str(output_dir).strip()
        if od and st.session_state.get('tab3_multi_od_mkdir') != od:
             try: Path(od).expanduser().mkdir(parents=True, exist_ok=True); st.session_state['tab3_multi_od_mkdir'] = od
             except: pass

        st.subheader("2. Pairs")
        # Logic to find pairs
        input_path = Path(str(input_dir))
        geo_path = Path(str(geojson_dir))
        
        with st.expander("Suffix and Manual Pair Configuration", expanded=False):
            st.markdown("Configure prefix and suffix matching rules or set manual overrides.")
            
            c_p1, c_p2 = st.columns(2)
            st.session_state['tab3_multi_img_prefix'] = c_p1.text_input("Image Prefix", value=st.session_state['tab3_multi_img_prefix'])
            st.session_state['tab3_multi_img_suffix'] = c_p2.text_input("Image Suffix", value=st.session_state['tab3_multi_img_suffix'])
            
            c_p3, c_p4 = st.columns(2)
            st.session_state['tab3_multi_geo_prefix'] = c_p3.text_input("GeoJSON Prefix", value=st.session_state['tab3_multi_geo_prefix'])
            st.session_state['tab3_multi_geo_suffix'] = c_p4.text_input("GeoJSON Suffix", value=st.session_state['tab3_multi_geo_suffix'])
            
            if geo_path.exists():
                geo_files = sorted([f for f in os.listdir(geo_path) if f.endswith('.geojson')])
                img_files = sorted([f for f in os.listdir(input_path) if is_supported_image_file(f)]) if input_path.exists() else []
                
                if img_files:
                    st.caption(
                        "To point one image at a different GeoJSON, select it in the list "
                        "below and use the Boundaries file control next to the preview."
                    )

        pairs = _refresh_pairs(input_dir, geojson_dir)

        if not pairs:
            st.info("No pairs found.")
        else:
            # Search/Filter bar for practicality
            search_query = st.text_input(
                "Search pairs",
                value="",
                placeholder="Search image pairs...",
                label_visibility="collapsed",
                key="tab3_pairs_search"
            )
            
            filtered_pairs = pairs
            if search_query:
                filtered_pairs = [p for p in pairs if search_query.lower() in p['image'].lower()]
            
            # Batch Toggle
            def _toggle_batch():
                v = st.session_state.get('tab3_multi_all_proc', False)
                for p in filtered_pairs: st.session_state[f"tab3_mp_{p['image']}"] = v
                
            c_all_chk, c_all_lbl = st.columns([0.08, 0.92])
            c_all_chk.checkbox("##", key="tab3_multi_all_proc", on_change=_toggle_batch, label_visibility="collapsed")
            c_all_lbl.markdown("**Select All for Batch**")
            
            # List of Pairs
            if not filtered_pairs:
                st.caption("No matching pairs found.")
            else:
                for p in filtered_pairs[:30]: # Limit for perf
                    img = p['image']
                    c_chk, c_btn = st.columns([0.08, 0.92])
                    k_proc = f"tab3_mp_{img}"
                    if k_proc not in st.session_state: st.session_state[k_proc] = False
                    c_chk.checkbox("##", key=k_proc, label_visibility="collapsed")
                    
                    is_active = (st.session_state['tab3_multi_active_img'] == img)
                    btn_type = "primary" if is_active else "secondary"
                    roi_note = f"{p['roi_count']} ROIs" if p.get('geojson') else "no boundaries"
                    label = f"{img} - {roi_note}"

                    if c_btn.button(label, key=f"tab3_m_btn_{img}", type=btn_type, width='stretch'):
                        st.session_state['tab3_multi_active_img'] = img
                        st.rerun()
                    c_btn.caption(f"\u2514 {p['geojson']}" if p.get('geojson') else "\u2514 no matching GeoJSON")
                
                if len(filtered_pairs) > 30: 
                    st.caption(f"... {len(filtered_pairs)-30} more pairs ...")


        # Channel Config
        active_img = st.session_state.get('tab3_multi_active_img')
        if active_img:
             img_full_path = input_path / active_img
             st.subheader(f"3. Channels: {active_img}")
             
             try:
                 ch_names, ch_colors, ch_meta, mpp = _get_cached_channel_info(str(img_full_path), img_full_path.stat().st_mtime)
             except:
                 ch_names, ch_colors, ch_meta, mpp = [], [], [], 0.5083
             
             # Load Per-Pair Settings or Global
             pair_settings = st.session_state['tab3_multi_pair_settings'].get(active_img, {})
             
             # Visibility (Local to this active view)
             vis_key_store = f"tab3_m_vis_{active_img}"
             if vis_key_store not in st.session_state:
                 # Prefer saved settings if any
                 if 'vis_map' in pair_settings:
                    st.session_state[vis_key_store] = {int(k): v for k, v in pair_settings['vis_map'].items() if str(k).isdigit()}
                 else:
                    default_vis = {}
                    for i, ch in enumerate(ch_names):
                        c = ch.lower()
                        if 'he' in c or 'hematoxylin' in c or 'eosin' in c or 'dapi' in c: default_vis[i] = True
                        else: default_vis[i] = False
                    if not any(default_vis.values()) and ch_names: default_vis[0] = True
                    st.session_state[vis_key_store] = default_vis
             
             visible_map = st.session_state[vis_key_store]
             
             # Export Map (Per Pair or Global fallback)
             export_map_key = f"tab3_m_exp_map_{active_img}"
             if export_map_key not in st.session_state:
                 if 'export_map' in pair_settings:
                     st.session_state[export_map_key] = pair_settings['export_map']
                 else:
                     st.session_state[export_map_key] = st.session_state.get('tab3_multi_export_markers', {}).copy()

             current_export_map = st.session_state[export_map_key]

             rename_key = f"tab3_m_rename_{active_img}"
             colors_key = f"tab3_m_colors_{active_img}"
             if rename_key not in st.session_state:
                 st.session_state[rename_key] = dict(pair_settings.get('rename_map', {}))
             if colors_key not in st.session_state:
                 st.session_state[colors_key] = dict(pair_settings.get('color_map', {}))

             with st.expander("Markers", expanded=True):
                 st.caption(
                     "Extract: include this marker in the ROI files. "
                     "Preview: show it in the image on the right. "
                     "Name and colour are written into the extracted files."
                 )
                 h1, h2, h3, h4, h5 = st.columns([0.8, 0.8, 3, 1, 1.5])
                 h1.markdown("**Extract**")
                 h2.markdown("**Preview**")
                 h3.markdown("**Name**")
                 h4.markdown("**Colour**")
                 h5.markdown("**Dye**")

                 rename_map: dict[str, str] = {}
                 color_map: dict[str, tuple[int, int, int]] = {}

                 for i, (name, col, meta) in enumerate(zip(ch_names, ch_colors, ch_meta)):
                     k_exp = f"tab3_m_exp_{active_img}_{name}"
                     k_vis = f"tab3_m_vis_{active_img}_{i}"
                     k_col = f"tab3_m_col_{active_img}_{i}"
                     k_name = f"tab3_m_name_{active_img}_{i}"

                     if name not in current_export_map: current_export_map[name] = True

                     if isinstance(col, tuple): hex_c = '#{:02x}{:02x}{:02x}'.format(*col)
                     else: hex_c = col
                     # A blank name or a black colour is never a real choice: it is what an
                     # interrupted save left behind. Fill it from the image again, or the
                     # table shows empty boxes and black is written into the ROI files.
                     if not str(st.session_state.get(k_name) or "").strip():
                         st.session_state[k_name] = str(st.session_state[rename_key].get(name) or name)
                     if _unset_colour(st.session_state.get(k_col), hex_c):
                         st.session_state[k_col] = hex_c

                     r1, r2, r3, r4, r5 = st.columns([0.8, 0.8, 3, 1, 1.5])

                     checked = r1.checkbox("##", value=current_export_map[name], key=k_exp, label_visibility="collapsed")
                     current_export_map[name] = checked

                     if r2.checkbox("##", value=visible_map.get(i, False), key=k_vis, label_visibility="collapsed"):
                         visible_map[i] = True
                     else:
                         visible_map[i] = False

                     # Renaming here mirrors the registration tab: the typed name is what
                     # lands in the extracted OME-TIFF's channel metadata.
                     new_name = str(r3.text_input(f"name_{i}", key=k_name, label_visibility="collapsed") or name).strip() or name
                     chosen_hex = r4.color_picker("##", key=k_col, label_visibility="collapsed")
                     r5.text(meta.get('dye', '-'))

                     if new_name != name:
                         rename_map[name] = new_name
                     try:
                         color_map[new_name] = hex_to_rgb(chosen_hex)
                     except Exception:
                         pass

                 st.session_state[export_map_key] = current_export_map
                 st.session_state[vis_key_store] = visible_map
                 st.session_state[rename_key] = rename_map
                 st.session_state[colors_key] = color_map

        # --- Actions ---------------------------------------------------------
        # Below the marker list, so the last thing you set is the last thing you see.
        st.markdown("---")
        checked_pairs = [p for p in pairs if st.session_state.get(f"tab3_mp_{p['image']}")]
        ready_pairs = [p for p in checked_pairs if p.get('geojson')]
        missing = len(checked_pairs) - len(ready_pairs)

        if st.button(
            f"Extract ROIs ({len(checked_pairs)}/{len(pairs)})",
            type="primary",
            width='stretch',
            disabled=not checked_pairs,
            key="tab3_extract_rois",
            help="Extracts every image ticked in the list above, using that image's markers and boundaries.",
        ):
            _commit_all_pair_settings(pairs)
            if ready_pairs:
                _run_batch_extraction(ready_pairs, input_dir, geojson_dir, output_dir, create_log)
            else:
                st.warning("None of the ticked images have boundaries yet.")

        if missing:
            st.caption(f"{missing} ticked image(s) have no boundaries and will be skipped.")
        elif not checked_pairs:
            st.caption("Tick images in the list above to extract them.")

        if st.button(
            "Save Project",
            key="tab3_save_project",
            width='stretch',
            help="Stores directories, marker choices and ROI names for every image in this project.",
        ):
            _commit_all_pair_settings(pairs)
            from spacemap_gui.persistence import save_active_project_state
            save_active_project_state("tab3_roi_extraction")
            st.success("Saved.")


    # --- RIGHT COLUMN ---
    with col2:
        if not active_img:
            st.info("Select a pair to view.")
        else:
            # Load Data & Generate Preview (Manual matching Single Image)
            img_path = input_path / active_img
            
            # Preview Controls set to high-performance default constants (no display widgets needed)
            enhance_contrast = True
            invert_colors = False
            blending_mode = "Additive"

            # 2. Identify Visible Channels
            vis_map = st.session_state.get(f"tab3_m_vis_{active_img}", {})
            visible_indices = sorted([int(i) for i, v in vis_map.items() if v and str(i).isdigit()])
            
            if not visible_indices and not ch_names:
                # If no channels visible, maybe show nothing or placeholder
                thumb = None
                full_h, full_w = 1000, 1000
                st.info("No channels selected to show.")
            else:
                 try:
                     # 3. Load Data
                     # Allow loading if nothing selected? load_preview_channels handles empty list?
                     # If RGB, indices might be ignored or handled.
                     img_data_dict, full_shape, is_rgb = load_preview_channels(str(img_path), visible_indices)
                     
                     full_h, full_w = full_shape[0], full_shape[1]
                     
                     # 4. Generate Composite
                     if is_rgb:
                         img_to_show = img_data_dict.get('rgb', np.zeros((100,100,3), dtype=np.uint8)).copy()
                         if invert_colors:
                             if img_to_show.dtype == np.uint8: img_to_show = 255 - img_to_show
                             else: img_to_show = np.nanmax(img_to_show) - img_to_show
                         
                         if enhance_contrast and img_to_show.dtype == np.uint8:
                              p_min, p_max = np.nanmin(img_to_show), np.nanmax(img_to_show)
                              if p_max - p_min > 0 and p_max - p_min < 200:
                                  img_to_show = np.clip((img_to_show - p_min) / (p_max - p_min) * 255, 0, 255).astype(np.uint8)
                     else:
                         # Composite
                         if not img_data_dict:
                             # Should effectively be black
                             h, w = (100, 100)
                             if display_width := 800: pass # dummy
                             composite = np.zeros((h, w, 3), dtype=np.float32) # Fallback size? 
                             # Actually load_preview_channels usually returns at least shape if file is valid
                             # But if indices is empty, dict is empty.
                             # We need the shape. If dict empty, use cached shape or generic?
                             # Let's hope at least one channel is loaded or is_rgb.
                             # If nothing loaded, we can't determine shape easily without metadata logic.
                             # We'll assume at least one channel visible or handle graceful fail.
                             st.warning("Select 'Show' on at least one channel to view image.")
                             composite = np.zeros((512, 512, 3), dtype=np.uint8)
                         else:
                             first_key = next(iter(img_data_dict))
                             h, w = img_data_dict[first_key].shape
                             composite = np.zeros((h, w, 3), dtype=np.float32)
                             
                             for i in visible_indices:
                                 if i in img_data_dict:
                                     ch_data = img_data_dict[i].astype(np.float32)
                                     
                                     if invert_colors:
                                         ch_data = np.nanmax(ch_data) - ch_data
                                     
                                     if ch_data.max() > 0:
                                         if enhance_contrast:
                                             p99 = np.percentile(ch_data, 99)
                                             if p99 > 0: ch_data = np.clip(ch_data / p99, 0, 1)
                                             else: ch_data = ch_data / ch_data.max()
                                         else:
                                             ch_data = ch_data / ch_data.max()
                                     
                                     # Get Color
                                     # Multi keys: tab3_m_col_{active_img}_{i}
                                     # If not set, use default logic
                                     k_col = f"tab3_m_col_{active_img}_{i}"
                                     if k_col in st.session_state:
                                         hex_c = st.session_state[k_col]
                                         color = hex_to_rgb(hex_c)
                                     else:
                                         # Fallback
                                         if i < len(ch_colors):
                                              color = ch_colors[i]
                                              if isinstance(color, str): color = hex_to_rgb(color)
                                         else: color = (255, 255, 255)
                                     
                                     if blending_mode == "Additive":
                                         for c in range(3): composite[:, :, c] += ch_data * color[c]
                                     else:
                                         for c in range(3):
                                             val = ch_data * color[c]
                                             curr = composite[:, :, c]
                                             composite[:, :, c] = curr + val - (curr * val) / 255.0
                         
                         img_to_show = np.clip(composite, 0, 255).astype(np.uint8)

                     # Convert to PIL
                     pil_img = Image.fromarray(img_to_show)
                     if pil_img.mode != 'RGB': pil_img = pil_img.convert('RGB')
                     
                     # Downscale for display/canvas if huge
                     # Canvas performs better with manageable sizes
                     max_dim = 1000
                     if pil_img.width > max_dim or pil_img.height > max_dim:
                         pil_img.thumbnail((max_dim, max_dim))
                     
                     thumb = pil_img
                     
                 except Exception as e:
                     st.error(f"Error generating preview: {e}")
                     thumb = None
                     full_h, full_w = 1000, 1000

                     
            if thumb:
                # 5. Canvas Prep (Single Image "Draw ROIs" Logic)
                # Resize for display
                display_width = 700
                canvas_width = thumb.width
                canvas_height = thumb.height
                
                if canvas_width > display_width:
                    ratio = display_width / canvas_width
                    canvas_width = display_width
                    canvas_height = int(thumb.height * ratio)
                
                # Resize PIL
                canvas_bg = thumb.resize((canvas_width, canvas_height))
                
                # Update Scale Factors
                sX = full_w / canvas_width
                sY = full_h / canvas_height
                
                pair_dict = next((p for p in pairs if p['image'] == active_img), None)
                pair_gj = pair_dict.get('geojson') if pair_dict else None
                is_new_boundary = pair_dict.get('is_new_boundary', False) if pair_dict else False
                
                # Hydrate from saved settings if first load
                pair_settings = st.session_state.get('tab3_multi_pair_settings', {}).get(active_img, {})
                if active_img not in st.session_state['tab3_multi_canvas_objects']:
                    if 'canvas_objects' in pair_settings:
                        st.session_state['tab3_multi_canvas_objects'][active_img] = pair_settings['canvas_objects']
                        st.session_state['tab3_multi_roi_labels'][active_img] = {int(k): v for k, v in pair_settings.get('roi_labels', {}).items() if str(k).isdigit()}
                
                # Check Load from Disk
                if pair_gj:
                    if is_new_boundary:
                        gj_path = geo_path / "new_boundaries" / pair_gj
                    else:
                        gj_path = geo_path / pair_gj
                        
                    gj_mtime = gj_path.stat().st_mtime if gj_path.exists() else 0
                    last_load = st.session_state['tab3_multi_geojson_loaded_mtime'].get(active_img, 0)
                    
                    if gj_mtime != last_load:
                        try:
                            geo = _load_geojson_cached(str(gj_path), gj_mtime)
                            drawing, lbls = geojson_to_fabric_with_labels(geo, 1.0/sX, 1.0/sY)
                            f_objs = [o for o in (drawing.get('objects', []) or []) if isinstance(o, dict) and o.get('type') != 'image']
                            st.session_state['tab3_multi_canvas_objects'][active_img] = f_objs
                            
                            new_l = {}
                            for i, l in enumerate(lbls): new_l[i] = str(l)
                            st.session_state['tab3_multi_roi_labels'][active_img] = new_l
                            
                            st.session_state['tab3_multi_geojson_loaded_mtime'][active_img] = gj_mtime
                            st.session_state['tab3_multi_canvas_reset'][active_img] = str(datetime.now())
                        except: pass
                
                # Reset Logic
                force_reset = st.session_state['tab3_multi_canvas_reset'].get(active_img)
                ver = st.session_state.get(f"tab3_m_ver_{active_img}", 0)
                if force_reset:
                    ver += 1
                    st.session_state[f"tab3_m_ver_{active_img}"] = ver
                    del st.session_state['tab3_multi_canvas_reset'][active_img]

                st.markdown("---")
                st.subheader("ROIs")

                method_key = f"tab3_boundary_method_{active_img}"

                if method_key not in st.session_state:
                    saved_method = pair_settings.get('boundary_method')
                    if saved_method:
                        st.session_state[method_key] = saved_method
                    elif pair_gj and not is_new_boundary:
                        st.session_state[method_key] = BOUNDARY_METHOD_FILE
                    else:
                        st.session_state[method_key] = DRAW_SHAPE_METHODS["Polygon"]

                source_key = f"tab3_boundary_source_{active_img}"
                shape_key = f"tab3_boundary_shape_{active_img}"
                draw_key = f"tab3_draw_new_{active_img}"
                if source_key not in st.session_state:
                    st.session_state[source_key] = boundary_source_from_stored(st.session_state[method_key])
                if shape_key not in st.session_state:
                    st.session_state[shape_key] = boundary_shape_from_stored(st.session_state[method_key])

                boundary_source = st.radio(
                    "Where the ROIs come from",
                    options=BOUNDARY_SOURCES,
                    key=source_key,
                    horizontal=True,
                )

                if boundary_source == BOUNDARY_SOURCE_FILE:
                    boundary_method = BOUNDARY_METHOD_FILE
                    canvas_drawing_mode = "transform"
                elif boundary_source == BOUNDARY_SOURCE_TMA:
                    boundary_method = BOUNDARY_METHOD_TMA
                    canvas_drawing_mode = "transform"
                else:
                    # Drawing is an explicit action rather than a mode you can get stuck
                    # in: off, the canvas only lets you nudge shapes that already exist.
                    drawing_on = st.toggle(
                        "Draw a new ROI",
                        key=draw_key,
                        help="On: click the image to add an ROI. Off: drag an existing ROI to adjust it.",
                    )
                    if drawing_on:
                        shape = st.radio(
                            "Shape",
                            options=list(DRAW_SHAPE_METHODS),
                            key=shape_key,
                            horizontal=True,
                        )
                        boundary_method = DRAW_SHAPE_METHODS[shape]
                        canvas_drawing_mode = DRAW_SHAPE_CANVAS_MODES[shape]
                    else:
                        boundary_method = DRAW_SHAPE_METHODS[
                            st.session_state.get(shape_key, "Polygon")
                        ]
                        canvas_drawing_mode = "transform"

                st.session_state[method_key] = boundary_method

                if boundary_method == "Predefined GeoJSON boundaries":
                    if pair_gj:
                        original_path = geo_path / pair_gj
                        new_path = geo_path / "new_boundaries" / pair_gj
                        
                        gj_path = None
                        load_triggered = False
                        
                        if new_path.exists():
                            st.caption(f"Matched GeoJSON file: {pair_gj} (Found customized version in new_boundaries)")
                            c_load1, c_load2 = st.columns(2)
                            if c_load1.button("Load Original Predefined", key=f"tab3_load_gj_btn_{active_img}"):
                                gj_path = original_path
                                load_triggered = True
                            if c_load2.button("Load Customized Saved", key=f"tab3_load_gj_new_btn_{active_img}", type="primary"):
                                gj_path = new_path
                                load_triggered = True
                        else:
                            st.caption(f"Matched GeoJSON file: {pair_gj}")
                            if st.button("Load Boundaries from File", key=f"tab3_load_gj_btn_{active_img}"):
                                gj_path = original_path
                                load_triggered = True
                                
                        if load_triggered and gj_path and gj_path.exists():
                            try:
                                geo = _load_geojson_cached(str(gj_path), gj_path.stat().st_mtime)
                                drawing, lbls = geojson_to_fabric_with_labels(geo, 1.0/sX, 1.0/sY)
                                f_objs = [o for o in (drawing.get('objects', []) or []) if isinstance(o, dict) and o.get('type') != 'image']
                                st.session_state['tab3_multi_canvas_objects'][active_img] = f_objs
                                new_l = {}
                                for i, l in enumerate(lbls):
                                    new_l[i] = str(l)
                                st.session_state['tab3_multi_roi_labels'][active_img] = new_l
                                st.session_state[f"tab3_m_ver_{active_img}"] = st.session_state.get(f"tab3_m_ver_{active_img}", 0) + 1
                                st.success("Loaded boundaries successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading boundaries: {e}")
                    else:
                        st.warning("No matched GeoJSON file found for this image. Configure suffixes on the left.")

                elif boundary_method == "Auto TMA mode (automatic core detection)":
                    with st.expander("Auto TMA Core Detection Settings", expanded=True):
                        # Keys start with tab3_ so these settings are saved in the
                        # project file; the bare tma_ prefix belonged to no tab and
                        # was silently dropped on every save.
                        from spacemap_gui.ui_utils import seed_widget_state
                        tma_keys = {
                            "min": (f"tab3_tma_min_{active_img}", 30),
                            "max": (f"tab3_tma_max_{active_img}", 150),
                            "thresh": (f"tab3_tma_thresh_{active_img}", 0),
                            "gap": (f"tab3_tma_row_gap_{active_img}", 0.5),
                            "fmt": (f"tab3_tma_row_fmt_{active_img}", "A-Z"),
                            "prefix": (f"tab3_tma_prefix_{active_img}", ""),
                        }
                        for tma_key, tma_default in tma_keys.values():
                            seed_widget_state(tma_key, tma_default)
                        seed_widget_state(tma_keys["fmt"][0], "A-Z", options=["A-Z", "1-N"])

                        c_t1, c_t2 = st.columns(2)
                        min_size = c_t1.slider(
                            "Min Core Size (px)", min_value=10, max_value=300, step=5, key=tma_keys["min"][0],
                            help="Smallest blob, in thumbnail pixels, that counts as a tissue core. Raise it to ignore debris.",
                        )
                        max_size = c_t2.slider(
                            "Max Core Size (px)", min_value=50, max_value=500, step=5, key=tma_keys["max"][0],
                            help="Largest blob, in thumbnail pixels, that counts as one core. Lower it if touching cores are merged.",
                        )

                        c_t3, c_t4 = st.columns(2)
                        threshold_val = c_t3.slider(
                            "Sensitivity / Threshold (0=Auto)", min_value=0, max_value=255, step=5, key=tma_keys["thresh"][0],
                            help="Brightness cut-off that separates tissue from glass. 0 picks it automatically.",
                        )
                        row_gap_factor = c_t4.slider(
                            "Row Gap Factor", min_value=0.1, max_value=2.0, step=0.1, key=tma_keys["gap"][0],
                            help="How far apart, relative to core size, two cores must be vertically to start a new row.",
                        )

                        c_t5, c_t6 = st.columns(2)
                        row_format = c_t5.selectbox(
                            "Row Label Format", ["A-Z", "1-N"], key=tma_keys["fmt"][0],
                            help="Name cores by letter row (A1, B1, ...) or numbered row (R1_C1, R2_C1, ...).",
                        )
                        col_prefix = c_t6.text_input(
                            "Core Name Prefix", key=tma_keys["prefix"][0],
                            help="Optional text added to the start of every core name.",
                        )
                        
                        if st.button("Detect TMA Cores", key=f"run_tma_{active_img}", type="primary"):
                            detected_objs, detected_labels = detect_tma_cores(
                                image=canvas_bg,
                                min_size=min_size,
                                max_size=max_size,
                                threshold_val=threshold_val,
                                row_threshold_factor=row_gap_factor,
                                rows_input=8,
                                cols_input=12,
                                row_format=row_format,
                                col_format="1-N"
                            )
                            if detected_objs:
                                if col_prefix:
                                    detected_labels = [f"{col_prefix}{lbl}" for lbl in detected_labels]
                                st.session_state['tab3_multi_canvas_objects'][active_img] = detected_objs
                                new_l = {}
                                for i, l in enumerate(detected_labels):
                                    new_l[i] = str(l)
                                st.session_state['tab3_multi_roi_labels'][active_img] = new_l
                                st.session_state[f"tab3_m_ver_{active_img}"] = st.session_state.get(f"tab3_m_ver_{active_img}", 0) + 1
                                st.success(f"Detected {len(detected_objs)} cores!")
                                st.rerun()
                            else:
                                st.warning("No cores detected. Adjust size or threshold settings.")

                curr_objs = st.session_state['tab3_multi_canvas_objects'].get(active_img, [])
                labels_dict = st.session_state['tab3_multi_roi_labels'].get(active_img, {})
                if labels_dict:
                    labels_dict = {int(k): v for k, v in labels_dict.items() if str(k).isdigit()}
                    st.session_state['tab3_multi_roi_labels'][active_img] = labels_dict

                # Which ROI is outlined on the canvas. The table below sets this by index,
                # so duplicate names still highlight the row you actually clicked.
                sel_key = f"tab3_sel_roi_{active_img}"
                selected_roi_index = st.session_state.get(sel_key)
                if selected_roi_index is not None and not (0 <= int(selected_roi_index) < len(curr_objs)):
                    selected_roi_index = None
                    st.session_state[sel_key] = None

                # Generate visual representation string for caching and key-triggering
                vis_map = st.session_state.get(f"tab3_m_vis_{active_img}", {})
                visible_indices = sorted([int(i) for i, v in vis_map.items() if v and str(i).isdigit()])
                enhance_contrast = True
                invert_colors = False
                blending_mode = "Additive"
                
                vis_state_str = "_".join(map(str, visible_indices))
                
                # Build Cache Key including labels and shape count to reload on edit/sync
                labels_tuple = tuple(sorted((int(k), v) for k, v in labels_dict.items() if str(k).isdigit()))
                bg_cache_key = (
                    active_img,
                    canvas_width,
                    canvas_height,
                    vis_state_str,
                    labels_tuple,
                    len(curr_objs)
                )
                
                # Create or fetch bg_obj
                if "tab3_canvas_bg_cache" not in st.session_state:
                    st.session_state["tab3_canvas_bg_cache"] = {}
                
                if bg_cache_key not in st.session_state["tab3_canvas_bg_cache"]:
                    canvas_bg_with_labels = draw_labels_on_image(canvas_bg, curr_objs, labels_dict)
                    bg_b64 = image_to_base64(canvas_bg_with_labels)
                    st.session_state["tab3_canvas_bg_cache"][bg_cache_key] = {
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
                
                bg_obj = st.session_state["tab3_canvas_bg_cache"][bg_cache_key]

                # Reconstruct display objects and append behind background
                display_objs = []
                for i, obj in enumerate(curr_objs):
                    obj_copy = obj.copy()
                    if selected_roi_index is not None and i == int(selected_roi_index):
                        obj_copy["stroke"] = "#ff0000"
                        obj_copy["strokeWidth"] = 4
                    else:
                        obj_copy["stroke"] = "#00ff00"
                        obj_copy["strokeWidth"] = 2
                    display_objs.append(obj_copy)

                initial_drawing = {
                    "version": "4.4.0",
                    "objects": [bg_obj] + display_objs
                }

                # The canvas remounts whenever this key changes, and remounting throws away
                # any edit that has not been synced. So the key must depend only on things
                # that really do require a fresh canvas — the image, an explicit reset, the
                # drawing mode, the visible channels and how many shapes we pushed in.
                # It must NOT depend on ROI names: renaming used to remount the canvas and
                # silently discard a resize the user had just made.
                c_key = f"can_m_{active_img}_{ver}_{canvas_drawing_mode}_{vis_state_str}_{len(display_objs)}"

                if HAS_CANVAS:
                    # update_streamlit=True is the component's normal mode: it reports each
                    # completed edit (fabric's mouse:up) back to Python by itself. Trying to
                    # keep it silent and pull on demand does not work — the component only
                    # ever pushes on its own terms, so a "Sync" button had nothing to read.
                    res = st_canvas(
                        fill_color="rgba(255,165,0,0.3)", stroke_color="#00ff00", stroke_width=2,
                        update_streamlit=True, height=canvas_height, width=canvas_width,
                        drawing_mode=canvas_drawing_mode, key=c_key, initial_drawing=initial_drawing,
                        display_toolbar=True
                    )

                    if res is not None and res.json_data:
                        raw = res.json_data.get("objects", []) or []
                        objs = [o for o in raw if isinstance(o, dict) and o.get("type") != "image"]
                        for o in objs:
                            o["stroke"] = "#00ff00"
                            o["strokeWidth"] = 2

                        previous = st.session_state['tab3_multi_canvas_objects'].get(active_img, [])
                        # Only act on a real geometry change. Without this the commit would
                        # re-trigger on every rerun and spin.
                        if _canvas_geometry_signature(objs) != _canvas_geometry_signature(previous):
                            st.session_state['tab3_multi_canvas_objects'][active_img] = objs

                            # Names follow their shape by position; a newly drawn shape gets
                            # a default name.
                            st.session_state['tab3_multi_roi_labels'][active_img] = {
                                i: labels_dict.get(i, f"ROI {i + 1}") for i in range(len(objs))
                            }

                            if len(objs) != len(previous):
                                st.session_state[sel_key] = None
                            st.rerun()

                    if canvas_drawing_mode == "polygon":
                        # The component disables reporting in polygon mode, so a finished
                        # polygon only reaches the tab via the toolbar's send arrow.
                        st.caption(
                            "Polygons only register when you press the download arrow on the "
                            "canvas toolbar. Save ROIs then writes them to the GeoJSON."
                        )
                    else:
                        st.caption("Edits apply as you make them. Save ROIs writes them to the GeoJSON.")

                canvas_objs = st.session_state['tab3_multi_canvas_objects'].get(active_img, [])
                if canvas_objs:
                    st.markdown(f"**{len(canvas_objs)} ROIs** - click a row to outline it above")

                    rows = []
                    for i, obj in enumerate(canvas_objs):
                        w, h = _roi_extent(obj, sX, sY)
                        rows.append({
                            "Name": labels_dict.get(i, f"ROI {i + 1}"),
                            "Shape": _friendly_shape_type(obj),
                            "Size (px)": f"{w:,} x {h:,}" if w and h else "-",
                        })

                    table_event = st.dataframe(
                        pd.DataFrame(rows),
                        hide_index=True,
                        width='stretch',
                        on_select="rerun",
                        selection_mode="single-row",
                        key=f"tab3_roi_table_{active_img}",
                    )

                    picked = []
                    try:
                        picked = list(table_event.selection.rows)
                    except Exception:
                        picked = []
                    picked_index = int(picked[0]) if picked else None
                    if picked_index != st.session_state.get(sel_key):
                        st.session_state[sel_key] = picked_index
                        st.rerun()

                    if picked_index is not None:
                        current_name = labels_dict.get(picked_index, f"ROI {picked_index + 1}")
                        roi_name_key = f"tab3_roi_name_{active_img}_{picked_index}"
                        if roi_name_key not in st.session_state:
                            st.session_state[roi_name_key] = current_name

                        c_name, c_del = st.columns([3, 1])
                        typed = str(c_name.text_input(
                            "Name of the selected ROI",
                            key=roi_name_key,
                            help="This becomes the ROI's name in the GeoJSON and in the extracted file name.",
                        ) or "").strip()

                        if typed and typed != current_name:
                            labels_dict[picked_index] = typed
                            st.session_state['tab3_multi_roi_labels'][active_img] = labels_dict
                            st.rerun()

                        c_del.markdown("&nbsp;", unsafe_allow_html=True)
                        if c_del.button("Delete", key=f"tab3_roi_del_{active_img}_{picked_index}", width='stretch'):
                            remaining = [o for j, o in enumerate(canvas_objs) if j != picked_index]
                            relabelled = {}
                            for j in range(len(canvas_objs)):
                                if j == picked_index:
                                    continue
                                target = j if j < picked_index else j - 1
                                relabelled[target] = labels_dict.get(j, f"ROI {j + 1}")
                            st.session_state['tab3_multi_canvas_objects'][active_img] = remaining
                            st.session_state['tab3_multi_roi_labels'][active_img] = relabelled
                            st.session_state[sel_key] = None
                            st.session_state[f"tab3_m_ver_{active_img}"] = st.session_state.get(f"tab3_m_ver_{active_img}", 0) + 1
                            st.rerun()
                else:
                    st.caption("No ROIs yet. Load a boundaries file, or switch to Draw by hand.")


                active_pair = next((q for q in pairs if q.get('image') == active_img), None)
                source_geojson = None
                if active_pair and active_pair.get('geojson'):
                    _geo_base = Path(geojson_dir)
                    source_geojson = (
                        _geo_base / "new_boundaries" / active_pair['geojson']
                        if active_pair.get('is_new_boundary')
                        else _geo_base / active_pair['geojson']
                    )

                _geo_dir_path = Path(geojson_dir)
                _geo_choices = ["<Auto>", "<Skip>"] + (
                    sorted(f for f in os.listdir(_geo_dir_path) if f.endswith('.geojson'))
                    if _geo_dir_path.exists() else []
                )
                _override_key = f"tab3_multi_override_{active_img}"
                if st.session_state.get(_override_key) not in _geo_choices:
                    st.session_state[_override_key] = "<Auto>"

                def _forget_loaded_boundaries(image=active_img):
                    """Changing the file must re-read it, not reuse what is already on screen."""
                    for store in ('tab3_multi_canvas_objects', 'tab3_multi_roi_labels',
                                  'tab3_multi_geojson_loaded_mtime'):
                        st.session_state.get(store, {}).pop(image, None)
                    st.session_state[f"tab3_m_ver_{image}"] = st.session_state.get(f"tab3_m_ver_{image}", 0) + 1

                st.selectbox(
                    "Boundaries file",
                    options=_geo_choices,
                    key=_override_key,
                    on_change=_forget_loaded_boundaries,
                    help="<Auto> matches by name, <Skip> leaves this image out. "
                         "Picking a file re-reads it from disk.",
                )

                if source_geojson is not None:
                    st.caption(f"Reading: {source_geojson.name}")
                    save_as_new = st.checkbox(
                        "Save as a new file instead of replacing this one",
                        key=f"tab3_save_as_new_{active_img}",
                        help=f"Unticked, Save ROIs overwrites {source_geojson.name}.",
                    )
                else:
                    st.caption("No boundaries file yet — Save ROIs will create one.")
                    save_as_new = True

                col_c1, col_c2 = st.columns(2)

                if col_c1.button("Clear ROIs", key=f"clear_rois_btn_{active_img}", width='stretch'):
                    st.session_state['tab3_multi_canvas_objects'][active_img] = []
                    st.session_state['tab3_multi_roi_labels'][active_img] = {}
                    st.session_state[f"tab3_m_ver_{active_img}"] = st.session_state.get(f"tab3_m_ver_{active_img}", 0) + 1
                    st.rerun()

                if col_c2.button("Save ROIs", key=f"save_rois_btn_{active_img}", type="primary", width='stretch',
                                 help="Writes the ROI names and shapes above to a GeoJSON file."):
                    if not canvas_objs:
                        st.warning("There are no ROIs to save.")
                    else:
                        labels_list = [labels_dict.get(i, f"ROI {i+1}") for i in range(len(canvas_objs))]
                        try:
                            if save_as_new or source_geojson is None:
                                target = None
                                out_dir = Path(geojson_dir) / "new_boundaries"
                                out_dir.mkdir(parents=True, exist_ok=True)
                            else:
                                target = source_geojson
                                out_dir = source_geojson.parent

                            saved_gj = save_drawn_rois_for_extraction(
                                objects=canvas_objs,
                                labels=labels_list,
                                scale_x=sX,
                                scale_y=sY,
                                image_path=img_path,
                                geojson_dir=out_dir,
                                output_path=target,
                            )
                            st.success(f"Saved {len(canvas_objs)} ROIs to {saved_gj.name}")
                            st.session_state['tab3_multi_geojson_loaded_mtime'][active_img] = saved_gj.stat().st_mtime
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not save the ROIs: {e}")
            else:
                 pass


def _commit_pair_settings(image_name: str) -> dict[str, Any]:
    """Fold one image's widget state into the per-pair settings the project stores.

    This is what the old "Save Pair Settings" button did. It now runs automatically from
    Save Project and from Extract ROIs, so settings are one save action and writing
    boundary files stays a separate, explicit step. Existing keys are preserved, so
    settings saved by older versions of the tab survive untouched.
    """
    all_settings = st.session_state.setdefault('tab3_multi_pair_settings', {})
    settings = dict(all_settings.get(image_name, {}))

    for field, session_key in (
        ('export_map', f"tab3_m_exp_map_{image_name}"),
        ('vis_map', f"tab3_m_vis_{image_name}"),
        ('rename_map', f"tab3_m_rename_{image_name}"),
        ('color_map', f"tab3_m_colors_{image_name}"),
        ('boundary_method', f"tab3_boundary_method_{image_name}"),
    ):
        value = st.session_state.get(session_key)
        if value is not None:
            settings[field] = value

    for field, store_key in (
        ('canvas_objects', 'tab3_multi_canvas_objects'),
        ('roi_labels', 'tab3_multi_roi_labels'),
    ):
        value = st.session_state.get(store_key, {}).get(image_name)
        if value is not None:
            settings[field] = value

    all_settings[image_name] = settings
    return settings


def _commit_all_pair_settings(pairs) -> None:
    """Commit settings for every listed image, not just the one on screen."""
    for pair in pairs or []:
        name = pair.get('image') if isinstance(pair, dict) else pair
        if name:
            _commit_pair_settings(str(name))


# Stored values for tab3_boundary_method_<image>. These strings are also what older
# project files contain and what _refresh_pairs matches on, so they must not change.
BOUNDARY_METHOD_FILE = "Predefined GeoJSON boundaries"
BOUNDARY_METHOD_TMA = "Auto TMA mode (automatic core detection)"
BOUNDARY_METHOD_LEGACY_TRANSFORM = "Select / Edit / Move existing boundaries"
DRAW_SHAPE_METHODS = {
    "Polygon": "Polygon (left-click vertices, right-click to close)",
    "Freehand": "Freehand (drag to scribble outline, release to commit)",
    "Rectangle": "Rectangle (click-and-drag bounding box)",
}
DRAW_SHAPE_CANVAS_MODES = {"Polygon": "polygon", "Freehand": "freedraw", "Rectangle": "rect"}

BOUNDARY_SOURCE_FILE = "Boundaries file"
BOUNDARY_SOURCE_DRAW = "Draw by hand"
BOUNDARY_SOURCE_TMA = "Detect TMA cores"
BOUNDARY_SOURCES = [BOUNDARY_SOURCE_FILE, BOUNDARY_SOURCE_DRAW, BOUNDARY_SOURCE_TMA]


def boundary_source_from_stored(stored) -> str:
    """Map a stored boundary method onto the three-way "where do ROIs come from" choice.

    Older projects can hold "Select / Edit / Move existing boundaries", which described a
    canvas mode rather than a source. Those ROIs were drawn by hand, so that is where it
    lands; selecting an ROI is now the table's job, not a mode.
    """
    value = str(stored or "")
    if value == BOUNDARY_METHOD_FILE:
        return BOUNDARY_SOURCE_FILE
    if value == BOUNDARY_METHOD_TMA:
        return BOUNDARY_SOURCE_TMA
    return BOUNDARY_SOURCE_DRAW


def boundary_shape_from_stored(stored, default: str = "Polygon") -> str:
    """Recover which drawing shape a stored boundary method referred to."""
    value = str(stored or "")
    for shape, method in DRAW_SHAPE_METHODS.items():
        if method == value:
            return shape
    return default


def _canvas_geometry_signature(objects) -> list:
    """Signature of ROI geometry only, ignoring styling this tab overwrites itself.

    Used to tell a genuinely new canvas push apart from the component echoing back the
    value it already reported, which is what arrives on the first pass after Sync.
    """
    ignored = {"stroke", "strokeWidth", "fill", "opacity", "selectable", "evented",
               "hoverCursor", "version"}
    signature = []
    for obj in objects or []:
        if not isinstance(obj, dict) or obj.get("type") == "image":
            continue
        signature.append(tuple(sorted(
            (key, json.dumps(value, sort_keys=True, default=str))
            for key, value in obj.items() if key not in ignored
        )))
    return signature


def _roi_extent(obj: dict[str, Any], scale_x: float = 1.0, scale_y: float = 1.0):
    """Width and height of one canvas ROI in full-resolution image pixels.

    Returns ``(None, None)`` for a shape whose geometry cannot be read, so the table can
    show a placeholder instead of failing to render.
    """
    try:
        kind = obj.get("type")
        if kind == "rect":
            width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
            height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
        elif kind == "path":
            xs = [float(c[1]) for c in obj.get("path", []) if len(c) >= 3 and c[0] in ("M", "L")]
            ys = [float(c[2]) for c in obj.get("path", []) if len(c) >= 3 and c[0] in ("M", "L")]
            if not xs or not ys:
                return None, None
            width, height = max(xs) - min(xs), max(ys) - min(ys)
        elif kind in ("polygon", "circle", "ellipse"):
            points = obj.get("points") or []
            if points:
                xs = [float(pt["x"]) for pt in points]
                ys = [float(pt["y"]) for pt in points]
                width, height = max(xs) - min(xs), max(ys) - min(ys)
            else:
                width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
                height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))
        else:
            return None, None

        return int(round(width * float(scale_x))), int(round(height * float(scale_y)))
    except Exception:
        return None, None


def _run_batch_extraction(to_process, input_dir, geojson_dir, output_dir, create_log):
    status = st.status("Batch Processing...", expanded=True)
    input_p = Path(str(input_dir))
    geo_p = Path(str(geojson_dir))
    out_p = Path(str(output_dir))
    mpp = {'codex': float(st.session_state.get('tab3_multi_mpp', 0.5083))}
    
    global_export = st.session_state.get('tab3_multi_export_markers', {})
    record_process_event(
        "tab3",
        "ROI batch extraction started",
        details={"images_selected": len(to_process), "input_directory": str(input_p), "output_directory": str(out_p)},
    )

    log_file = None
    log_writer = None
    if create_log:
        try:
            ts = datetime.now().strftime('%y%m%d_%H%M%S')
            ldir = out_p / "log" / f"roi_multi_{ts}"
            ldir.mkdir(parents=True, exist_ok=True)
            log_file = open(ldir / "batch_log.csv", "w", newline="", encoding="utf-8")
        except: pass

    try:
        for idx, p in enumerate(to_process):
            img = p['image']
            
            if not p.get('geojson'):
                status.write(f"Skipping {img}: No matched GeoJSON file.")
                row = {"Image": img, "Status": "Skipped: No GeoJSON file"}
                if log_file:
                    if not log_writer:
                        log_writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
                        log_writer.writeheader()
                    log_writer.writerow(row)
                    log_file.flush()
                continue

            status.write(f"Processing {img}...")
            _log_console(f"Processing {img}...")
            
            row = {"Image": img, "Status": "Pending"}
            try:
                ip = input_p / img
                if p.get('is_new_boundary'):
                    gp = geo_p / "new_boundaries" / p['geojson']
                else:
                    gp = geo_p / p['geojson']
                
                pair_settings = st.session_state.get('tab3_multi_pair_settings', {}).get(img, {})
                ch_names, _, _, _ = _get_cached_channel_info(str(ip), ip.stat().st_mtime)
                
                if 'export_map' in pair_settings:
                    exp_map = pair_settings['export_map']
                else:
                    exp_map = global_export
                
                keep = [c for c in ch_names if exp_map.get(c, True)]
                # Carry the marker renames and colours chosen for this image, falling back
                # to whatever the tab is holding live if the pair was never committed.
                rename_map = pair_settings.get('rename_map')
                if rename_map is None:
                    rename_map = st.session_state.get(f"tab3_m_rename_{img}", {})
                color_map = pair_settings.get('color_map')
                if color_map is None:
                    color_map = st.session_state.get(f"tab3_m_colors_{img}", {})
                conf = {
                    'keep': keep,
                    'rename': dict(rename_map or {}),
                    'colors': {k: tuple(v) for k, v in (color_map or {}).items()},
                }
                
                t0 = time.time()
                process_single_image_rois(str(ip), str(gp), str(out_p), mpp, channel_config=conf)
                row["Status"] = "Success"
                row["Duration"] = f"{time.time()-t0:.2f}s"
                record_process_event(
                    "tab3",
                    "ROI batch image complete",
                    status="complete",
                    duration_s=time.time() - t0,
                    details={"input": str(ip), "geojson": str(gp), "output_directory": str(out_p)},
                )
                
            except Exception as e:
                row["Status"] = f"Error: {e}"
                _log_console(f"Error {img}: {e}", level="error")
                record_process_event(
                    "tab3",
                    "ROI batch image failed",
                    message=str(e),
                    status="error",
                    details={"image": img, "output_directory": str(out_p)},
                )
            
            if log_file:
                if not log_writer:
                    log_writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
                    log_writer.writeheader()
                log_writer.writerow(row)
                log_file.flush()
                
        status.update(label="Complete!", state="complete")
        st.success("Batch finished.")
        record_process_event("tab3", "ROI batch extraction complete", status="complete", details={"output_directory": str(out_p)})
    finally:
        if log_file: log_file.close()


def render():
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return
    st.markdown("## ROI Extraction")
    render_multiple_images(project_root)


def image_to_base64(img: Image.Image) -> str:
    """Convert a PIL image to a base64 data URI.

    Parameters
    ----------
    img:
        Image to encode.

    Returns
    -------
    str
        Base64-encoded PNG image as a data URI.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def save_drawn_rois(
    objects: list[dict[str, Any]],
    labels: list[str],
    scale_x: float,
    scale_y: float,
    image_path: str | Path,
    geojson_dir: str | Path,
) -> None:
    """Save drawn ROIs from the canvas into a GeoJSON file.

    Parameters
    ----------
    objects:
        Fabric.js objects from the drawable canvas (typically `type == "path"`).
    labels:
        ROI labels aligned to `objects`.
    scale_x:
        X scale factor from canvas pixels to full-resolution image pixels.
    scale_y:
        Y scale factor from canvas pixels to full-resolution image pixels.
    image_path:
        Path to the source image.
    geojson_dir:
        Output directory for the GeoJSON.

    Returns
    -------
    None
    """
    features = []
    for obj, label in zip(objects, labels):
        if obj["type"] == "path":
            # Convert path to polygon
            # path is list of ['M', x, y], ['L', x, y], ...
            path_data = obj["path"]
            coords = []
            for cmd in path_data:
                if cmd[0] in ['M', 'L']:
                    coords.append([cmd[1] * scale_x, cmd[2] * scale_y])

            # Skip degenerate shapes (e.g., accidental lines)
            if len(coords) < 3:
                continue
            
            # Close polygon
            if coords:
                coords.append(coords[0])
                
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {"label": label, "name": label}
            }
            features.append(feature)
            
        elif obj["type"] == "rect":
            obj_scale_x = obj.get("scaleX", 1.0)
            obj_scale_y = obj.get("scaleY", 1.0)
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
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords
                },
                "properties": {"label": label, "name": label}
            }
            features.append(feature)

        elif obj["type"] == "polygon":
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
                coords.append([cx * scale_x, cy * scale_y])
            if len(coords) < 3:
                continue
            if coords:
                coords.append(coords[0])
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": {"label": label, "name": label}
            }
            features.append(feature)
            
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Naming convention
    image_path = Path(image_path)
    stem = normalize_image_stem(image_path)
    if stem.endswith('_merged'): stem = stem[:-7]
    
    filename = f"{stem}_roi_bounding_boxes_coords.geojson"
    out_dir = Path(str(geojson_dir).strip()).expanduser()
    if str(out_dir) == "":
        st.error("Save directory is empty. Please choose a valid Save Directory.")
        return

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Could not create save directory: {out_dir} ({e})")
        return

    save_path = out_dir / filename

    try:
        with open(save_path, 'w') as f:
            json.dump(feature_collection, f)
    except Exception as e:
        st.error(f"Could not write GeoJSON: {save_path} ({e})")
        return
        
    st.success(f"Saved {len(features)} ROIs to {save_path}")


def save_drawn_rois_for_extraction(objects, labels, scale_x, scale_y, image_path, geojson_dir, output_path=None):
    """Write ROIs to GeoJSON and return the path written.

    ``output_path`` writes exactly there, which is how "Save ROIs" replaces the file the
    boundaries came from. Without it the name is derived from the image and written into
    ``geojson_dir``, as before.
    """
    features = []
    for obj, label in zip(objects, labels):
        if obj["type"] == "path":
            path_data = obj["path"]
            coords = []
            for cmd in path_data:
                if cmd[0] in ['M', 'L']:
                    coords.append([cmd[1] * scale_x, cmd[2] * scale_y])
            if len(coords) < 3:
                continue
            if coords:
                coords.append(coords[0])
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"label": label, "name": label}
            }
            features.append(feature)
        elif obj["type"] == "rect":
            obj_scale_x = obj.get("scaleX", 1.0)
            obj_scale_y = obj.get("scaleY", 1.0)
            left = obj["left"] * scale_x
            top = obj["top"] * scale_y
            width = obj["width"] * obj_scale_x * scale_x
            height = obj["height"] * obj_scale_y * scale_y
            coords = [[
                [left, top], [left + width, top],
                [left + width, top + height], [left, top + height], [left, top]
            ]]
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {"label": label, "name": label}
            }
            features.append(feature)
        elif obj["type"] == "polygon":
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
                coords.append([cx * scale_x, cy * scale_y])
            if len(coords) < 3:
                continue
            if coords:
                coords.append(coords[0])
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"label": label, "name": label}
            }
            features.append(feature)
            
    feature_collection = {"type": "FeatureCollection", "features": features}
    
    if output_path is not None:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stem = normalize_image_stem(image_path)
        if stem.endswith('_merged'): stem = stem[:-7]

        filename = f"{stem}_roi_bounding_boxes_coords.geojson"
        out_dir = Path(geojson_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / filename

    with open(save_path, 'w') as f:
        json.dump(feature_collection, f)
    
    log_tab3(f"Saved {len(features)} ROIs to {save_path}")
    return save_path


def apply_roi_changes(geojson_path):
    """
    Apply changes from st.data_editor state to the GeoJSON file.
    Returns True if changes were applied, False otherwise.
    """
    if "roi_editor" not in st.session_state:
        return False
        
    changes = st.session_state["roi_editor"]
    edited_rows = changes.get("edited_rows", {})
    
    if not edited_rows:
        return False
        
    try:
        with open(geojson_path, 'r') as f:
            geo_data = json.load(f)
            
        features = geo_data.get("features", [])
        updated = False
        
        for idx_str, new_values in edited_rows.items():
            idx = int(idx_str)
            if idx < len(features):
                if "Label" in new_values:
                    new_label = new_values["Label"]
                    if "properties" not in features[idx]:
                        features[idx]["properties"] = {}
                    features[idx]["properties"]["label"] = new_label
                    updated = True
                    
        if updated:
            with open(geojson_path, 'w') as f:
                json.dump(geo_data, f)
            return True
            
    except Exception as e:
        st.error(f"Error applying ROI changes: {e}")
        return False
    
    return False

def run_roi_extraction(input_path, geojson_dir, rois_dir, mpp, channel_config, geojson_filename=None, status_container=None, create_log=False):
    def log_step(msg):
        ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        full_msg = f"{ts} {msg}"
        if status_container:
            status_container.write(full_msg)
        _log_console(msg)
        record_process_event("tab3", "ROI extraction step", message=msg)

    # Logging Setup
    session_dir: Optional[str] = None
    log_path: Optional[str] = None
    session_start_dt = datetime.now()
    log_data: Dict[str, str] = {}
    
    # Ensure output dir exists
    rois_dir_path = Path(rois_dir)
    rois_dir_path.mkdir(parents=True, exist_ok=True)
    
    if create_log:
        log_root = rois_dir_path / "log"
        log_root.mkdir(exist_ok=True)
        session_dir = log_root / f"roi_session_{session_start_dt.strftime('%y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True)
        log_path = session_dir / "process_log.csv"
        
        # Initial Log Data
        log_data = {
            "Date": session_start_dt.strftime("%Y-%m-%d"),
            "Time": session_start_dt.strftime("%H:%M:%S"),
            "Input_File": str(input_path),
            "GeoJSON_Dir": str(geojson_dir),
            "Output_Dir": str(rois_dir),
        }
        # Input Metadata
        try:
            in_meta = get_image_metadata_log(str(input_path))
            log_data.update(_prefix_keys("Input_", in_meta))
        except Exception as e:
            from spacemap_gui.ui_utils import log_exception
            log_exception("reading input image metadata", e)

    if status_container:
        status_container.update(label="Extracting ROIs...", state="running")
    
    log_step(f"Input image: {Path(input_path).name}")
    log_step(f"GeoJSON dir: {geojson_dir}")
    record_process_event(
        "tab3",
        "ROI extraction started",
        details={"input": str(input_path), "geojson_directory": str(geojson_dir), "output_directory": str(rois_dir)},
    )
    start_time = time.time()
    
    try:
        # Determine GeoJSON Path for editing application
        current_geojson_path = None
        if geojson_filename:
            current_geojson_path = Path(geojson_dir) / geojson_filename
        else:
            stem = Path(input_path).stem
            if stem.endswith('.ome'): stem = stem[:-4]
            if stem.endswith('_merged'): stem = stem[:-7]
            current_geojson_path = Path(geojson_dir) / f"{stem}_roi_bounding_boxes_coords.geojson"

        if current_geojson_path and current_geojson_path.exists():
            log_step(f"Target GeoJSON: {current_geojson_path.name}")
            # Apply edits
            if apply_roi_changes(current_geojson_path):
                 log_step("Applied pending label changes to GeoJSON.")
        else:
            log_step(f"Warning: GeoJSON not found at {current_geojson_path}, proceeding scan...")

        # Process
        process_single_image_rois(
            str(input_path), 
            str(current_geojson_path), 
            str(rois_dir), 
            mpp, 
            channel_config=channel_config
        )
        
        elapsed = time.time() - start_time
        log_step(f"Done in {elapsed:.2f}s")
        
        if status_container:
            status_container.update(label="Extraction Complete!", state="complete", expanded=False)
        st.success(f"ROIs extracted to {rois_dir}")
        st.toast("ROI Extraction Complete!")
        # Update legacy/persistence
        st.session_state['roi_output_dir'] = str(rois_dir) 
        record_process_event(
            "tab3",
            "ROI extraction complete",
            status="complete",
            duration_s=elapsed,
            details={"input": str(input_path), "output_directory": str(rois_dir), "input_metadata": get_image_metadata_log(str(input_path))},
        )

        if create_log and log_path:
            log_data["Processing_Time_Sec"] = f"{elapsed:.3f}"
            log_data["Status"] = "Success"
            
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
                writer.writeheader()
                writer.writerow(log_data)
                
    except Exception as e:
        if status_container:
            status_container.update(label="Error", state="error")
        st.error(f"Error extracting ROIs: {e}")
        log_step(f"ERROR: {e}")
        record_process_event(
            "tab3",
            "ROI extraction failed",
            message=str(e),
            status="error",
            duration_s=time.time() - start_time,
            details={"input": str(input_path), "output_directory": str(rois_dir)},
        )
        
        if create_log and log_path:
            log_data["Status"] = f"Error: {e}"
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
                writer.writeheader()
                writer.writerow(log_data)

def geojson_to_fabric_with_labels(
    geojson_data: dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[dict[str, Any], list[str]]:
    objects: list[dict[str, Any]] = []
    labels: list[str] = []

    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data.get("features", [])
    elif geojson_data.get("type") == "Feature":
        features = [geojson_data]
    else:
        features = []

    for feature in features:
        geom = feature.get("geometry", {}) or {}
        props = feature.get("properties", {}) or {}
        label = resolve_roi_label(props, len(labels))
        gtype = geom.get("type")

        polys: list[list[list[float]]] = []
        if gtype == "Polygon":
            coords = geom.get("coordinates", []) or []
            if coords:
                polys = [coords[0]]
        elif gtype == "MultiPolygon":
            coords = geom.get("coordinates", []) or []
            for poly in coords:
                if poly and poly[0]:
                    polys.append(poly[0])

        for ring in polys:
            if not ring:
                continue
            path_cmds: list[list[Any]] = []
            x0, y0 = ring[0][0], ring[0][1]
            path_cmds.append(["M", float(x0) * scale_x, float(y0) * scale_y])
            for pt in ring[1:]:
                if not pt or len(pt) < 2:
                    continue
                x, y = pt[0], pt[1]
                path_cmds.append(["L", float(x) * scale_x, float(y) * scale_y])
            path_cmds.append(["z"])

            obj = {
                "type": "path",
                "path": path_cmds,
                "fill": "rgba(255, 165, 0, 0.3)",
                "stroke": "#000000",
                "strokeWidth": 2,
                "opacity": 1,
                "closed": True,
                "label": label,
            }
            objects.append(obj)
            labels.append(label)

    drawing = {"version": "4.4.0", "objects": objects}
    return drawing, labels


def geojson_to_fabric(geojson_data, scale_x=1.0, scale_y=1.0):
    drawing, _labels = geojson_to_fabric_with_labels(geojson_data, scale_x=float(scale_x), scale_y=float(scale_y))
    return drawing

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
