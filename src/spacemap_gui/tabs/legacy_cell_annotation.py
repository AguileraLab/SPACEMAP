"""Cell-level annotation: map tissue GeoJSON annotations onto segmented cells.

Rendered as the last subtab of the Annotation tab (``tabs/tab4_annotation``).
It is scheduled to move into the Segmentation tab, which is why it lives in its
own module rather than inside the tissue-annotation package.
"""

import streamlit as st
from datetime import datetime
import os
import time
from pathlib import Path
import json
import re
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from spacemap_gui.file_browser import render_file_browser
from spacemap.utils import hex_to_rgb
from spacemap_gui.ui_utils import directory_input_with_browse, st_file_selector
from spacemap_gui.process_log import record_process_event

# Lazy-load sentinels – set by _ensure_heavy_imports()
_HEAVY_LOADED = False
tifffile = None  # type: ignore[assignment]
cv2 = None       # type: ignore[assignment]
shape = None     # type: ignore[assignment]
mapping = None   # type: ignore[assignment]
Polygon = None   # type: ignore[assignment]
unary_union = None  # type: ignore[assignment]
make_valid = None   # type: ignore[assignment]
plt = None       # type: ignore[assignment]
sns = None       # type: ignore[assignment]
stats = None     # type: ignore[assignment]
extract_polygons_simple = None  # type: ignore[assignment]
save_compact_geojson = None     # type: ignore[assignment]
efficient_union = None          # type: ignore[assignment]
merge_polygons = None           # type: ignore[assignment]
split_sort_polygons = None      # type: ignore[assignment]
classify_point_in_polygon_lists = None  # type: ignore[assignment]
get_channel_info = None  # type: ignore[assignment]


def _ensure_heavy_imports():
    """Import heavy scientific libraries on first call; no-op afterwards."""
    global _HEAVY_LOADED
    if _HEAVY_LOADED:
        return
    global tifffile, cv2, shape, mapping, Polygon, unary_union, make_valid
    global plt, sns, stats
    global extract_polygons_simple, save_compact_geojson, efficient_union
    global merge_polygons, split_sort_polygons, classify_point_in_polygon_lists
    global get_channel_info

    import tifffile as _tifffile
    import cv2 as _cv2
    from shapely.geometry import shape as _shape, mapping as _mapping, Polygon as _Polygon
    from shapely.ops import unary_union as _uu
    from shapely.validation import make_valid as _mv
    from spacemap.background_subtraction_utils import get_channel_info as _gci
    from spacemap.annotation_utils import (
        extract_polygons_simple as _eps,
        save_compact_geojson as _scg,
        efficient_union as _eu,
        merge_polygons as _mp,
        split_sort_polygons as _ssp,
        classify_point_in_polygon_lists as _cppl,
    )
    import matplotlib.pyplot as _plt
    import seaborn as _sns
    from scipy import stats as _stats

    tifffile = _tifffile
    cv2 = _cv2
    shape = _shape
    mapping = _mapping
    Polygon = _Polygon
    unary_union = _uu
    make_valid = _mv
    get_channel_info = _gci
    extract_polygons_simple = _eps
    save_compact_geojson = _scg
    efficient_union = _eu
    merge_polygons = _mp
    split_sort_polygons = _ssp
    classify_point_in_polygon_lists = _cppl
    plt = _plt
    sns = _sns
    stats = _stats
    _HEAVY_LOADED = True


# ==============================================================================
# CORE MATCHING ALGORITHMS (SPACEMAP STANDARD)
# ==============================================================================

def log_step(step_name: str):
    """Log to console with timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {step_name}")
    record_process_event("tab4", "Annotation step", message=step_name)


def get_normalized_data(
    dataframe: pd.DataFrame,
    markers: list[str],
    normalization: str,
    cofactor: float,
) -> tuple[dict[str, pd.Series], str]:
    """Return marker series normalized for the Tab 5 comparison plots."""
    normalized: dict[str, pd.Series] = {}
    for marker in markers:
        values = pd.to_numeric(dataframe[marker], errors="coerce")
        if normalization == "Z-Score":
            standard_deviation = values.std()
            values = (values - values.mean()) / standard_deviation if standard_deviation else values * 0
        elif normalization == "Min-Max":
            value_range = values.max() - values.min()
            values = (values - values.min()) / value_range if value_range else values * 0
        elif normalization == "ArcSinh":
            values = np.arcsinh(values / max(float(cofactor), 1e-12))
        elif normalization == "Log10":
            values = np.log10(values.where(values > 0))
        normalized[marker] = values.dropna()

    labels = {
        "None": "Expression",
        "Z-Score": "Z-score",
        "Min-Max": "Min-max scaled expression",
        "ArcSinh": f"ArcSinh(expression / {cofactor:g})",
        "Log10": "Log10 expression",
    }
    return normalized, labels.get(normalization, "Expression")


def list_geojson_files(directory: str) -> list[str]:
    """List all GeoJSON files in the directory."""
    if not directory or not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.geojson', '.json'))
    ])


def load_geojson_labels_cached(file_path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return _load_geojson_labels_cached__impl(file_path, file_signature(file_path))


@st.cache_data
def _load_geojson_labels_cached__impl(file_path: str, signature) -> list[str]:
    """Extract unique classification names from GeoJSON (cached)."""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    features = data.get('features', [])
    if not features and data.get('type') == 'Feature':
        features = [data]
        
    labels = set()
    for feat in features:
        props = feat.get('properties', {})
        cls = props.get('classification', {})
        name = cls.get('name') if isinstance(cls, dict) else props.get('name')
        if name:
            labels.add(name)
            
    return sorted(list(labels))


def load_geojson_features_cached(file_path):
    """Cache-safe wrapper: re-reads when the file on disk changes."""
    from spacemap_gui.ui_utils import file_signature
    return _load_geojson_features_cached__impl(file_path, file_signature(file_path))


@st.cache_data
def _load_geojson_features_cached__impl(file_path: str, signature) -> list[dict]:
    """Load and cache GeoJSON features."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    features = data.get('features', [])
    if not features and data.get('type') == 'Feature':
        features = [data]
    return features


@st.cache_data
def get_cached_channel_image(file_path, channel_index):
    """
    Load a single channel image from disk, downsampled for preview.
    Cached by Streamlit to avoid re-reading.
    """
    try:
        _ensure_heavy_imports()
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


def load_preview_channels(file_path: str, indices_to_load: list) -> tuple:
    """Load specific channels for preview (subsampled)."""
    loaded_data = {}
    full_shape = (0, 0)
    is_rgb = False
    
    try:
        _ensure_heavy_imports()
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
        for idx in indices_to_load:
            img, shape = get_cached_channel_image(file_path, idx)
            if img is not None:
                loaded_data[idx] = img
                full_shape = shape

    except Exception as e:
        print(f"Error load_preview_channels: {e}")
    return loaded_data, full_shape, is_rgb


def _normalize_filename_for_matching(filename: str) -> str:
    """Normalize filename for matching by removing extension and standard suffixes."""
    name = Path(filename).name.lower()
    for ext in ['.geojson', '.json', '.csv', '.ome.tiff', '.ome.tif', '.tiff', '.tif']:
        if name.endswith(ext):
            name = name[:-len(ext)]
            
    # Retrieve prefixes and suffixes to strip from session_state
    strip_suffixes = []
    if 'annot_strip_suffixes' in st.session_state:
        raw_s = st.session_state['annot_strip_suffixes']
        if raw_s:
            strip_suffixes = [s.strip().lower() for s in raw_s.split(",") if s.strip()]
    else:
        strip_suffixes = ['_merged', '_zones', '_fully_annotated', '_cell_features', '_features', '_corrected', '_corr', '_cleaned', '_cellpose_features', '_cellpose']
        
    strip_prefixes = []
    if 'annot_strip_prefixes' in st.session_state:
        raw_p = st.session_state['annot_strip_prefixes']
        if raw_p:
            strip_prefixes = [p.strip().lower() for p in raw_p.split(",") if p.strip()]

    # Strip prefixes first
    for pref in strip_prefixes:
        if name.startswith(pref):
            name = name[len(pref):]
            
    # Strip suffixes (longer first)
    for suff in sorted(strip_suffixes, key=len, reverse=True):
        if name.endswith(suff):
            name = name[:-len(suff)]
            
    return name


def _tokenize_filename_for_matching(filename: str) -> tuple[list[str], list[str]]:
    """Tokenize filename into numbers and word components."""
    name = _normalize_filename_for_matching(filename)
    numbers = re.findall(r'\d+', name)
    words = re.findall(r'[a-zA-Z]+', name)
    return numbers, words


def _build_pairing_entry(filename: str) -> dict:
    """Create token payload for pairing similarity evaluation."""
    numbers, words = _tokenize_filename_for_matching(filename)
    return {
        "filename": filename,
        "normalized": _normalize_filename_for_matching(filename),
        "numbers": set(numbers),
        "words": set(words)
    }


def _score_pairing_entries(left: dict, right: dict) -> float:
    """Score matching strength based on matching tokens and ratio."""
    # 1. Exact match
    if left["normalized"] == right["normalized"]:
        return 10.0
        
    # 2. Match exact numbers (ROI / Patient indices)
    num_intersect = left["numbers"].intersection(right["numbers"])
    if left["numbers"] and right["numbers"] and not num_intersect:
        return 0.0 # Conflicting numbers (e.g. ROI 1 vs ROI 2) -> mismatch
        
    score = 0.0
    if num_intersect:
        score += len(num_intersect) * 2.0
        
    # 3. String ratio
    ratio = SequenceMatcher(None, left["normalized"], right["normalized"]).ratio()
    score += ratio * 3.0
    
    return score


def _derive_pair_label(csv_entry: dict, geo_entry: dict) -> str:
    """Determine the prefix key for a pair."""
    c_norm = csv_entry["normalized"]
    g_norm = geo_entry["normalized"]
    # Return shortest common prefix/string
    if len(c_norm) < len(g_norm) and c_norm in g_norm:
        return csv_entry["filename"].split('.')[0]
    return geo_entry["filename"].split('.')[0]


def discover_best_file_pairs(csv_files: list[str], geojson_files: list[str]) -> tuple[list[dict], list[str], list[str]]:
    """Pair CSV files with corresponding GeoJSON files."""
    if not csv_files or not geojson_files:
        return [], csv_files, geojson_files
        
    csv_entries = [_build_pairing_entry(f) for f in csv_files]
    geo_entries = [_build_pairing_entry(f) for f in geojson_files]
    
    pairs = []
    matched_csv = set()
    matched_geo = set()
    
    # Priority matching: find best scores
    candidates = []
    for c_idx, c_ent in enumerate(csv_entries):
        for g_idx, g_ent in enumerate(geo_entries):
            score = _score_pairing_entries(c_ent, g_ent)
            if score > 1.5:
                candidates.append((score, c_idx, g_idx))
                
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for score, c_idx, g_idx in candidates:
        if c_idx in matched_csv or g_idx in matched_geo:
            continue
            
        c_ent = csv_entries[c_idx]
        g_ent = geo_entries[g_idx]
        
        prefix = _derive_pair_label(c_ent, g_ent)
        pairs.append({
            "Prefix": prefix,
            "CSV File": c_ent["filename"],
            "GeoJSON File": g_ent["filename"]
        })
        
        matched_csv.add(c_idx)
        matched_geo.add(g_idx)
        
    unmatched_csv = [csv_files[i] for i in range(len(csv_files)) if i not in matched_csv]
    unmatched_geo = [geojson_files[i] for i in range(len(geojson_files)) if i not in matched_geo]
    
    return pairs, unmatched_csv, unmatched_geo


def find_best_match_for_prefix(prefix: str, files_list: list[str]) -> str:
    """Find the best matching file in the list for a given sample prefix."""
    if not files_list:
        return "No Match Found"
        
    # 1. Exact or substring match (fast path)
    for f in files_list:
        if prefix.lower() in f.lower() or f.lower() in prefix.lower():
            return f
            
    # 2. Normalized comparison
    norm_prefix = _normalize_filename_for_matching(prefix)
    for f in files_list:
        norm_f = _normalize_filename_for_matching(f)
        if norm_f in norm_prefix or norm_prefix in norm_f:
            return f
            
    # 3. Fuzzy matching
    best_match = "No Match Found"
    best_score = 0.0
    threshold = float(st.session_state.get('annot_fuzzy_threshold', 0.3))
    
    numbers_prefix = set(re.findall(r'\d+', norm_prefix))
    
    for f in files_list:
        norm_f = _normalize_filename_for_matching(f)
        
        # Check number overlap
        numbers_f = set(re.findall(r'\d+', norm_f))
        if numbers_prefix and numbers_f and not numbers_prefix.intersection(numbers_f):
            continue
            
        ratio = SequenceMatcher(None, norm_prefix, norm_f).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = f
            
    if best_score >= threshold:
        return best_match
        
    return "No Match Found"


def find_matching_csv(link_id: str, csv_files: list[str]) -> str:
    """Find a segmentation CSV file that matches a prefix link ID."""
    res = find_best_match_for_prefix(link_id, csv_files)
    return "" if res == "No Match Found" else res


def find_matching_image(link_id: str, image_files: list[str]) -> str:
    """Find a background image file that matches a prefix link ID."""
    res = find_best_match_for_prefix(link_id, image_files)
    return "" if res == "No Match Found" else res



# ==============================================================================
# PIPELINE STAGE 1: MERGE & CLEAN BACKEND
# ==============================================================================

def process_merge_and_clean(image_stem: str, geojson_files_with_actions: list, 
                           output_label: str, min_area: int, include_original: bool,
                           output_dir: str) -> bool:
    """Perform merge and clean operation for a single image prefix."""
    _ensure_heavy_imports()
    polys_to_merge = []
    polys_to_subtract = []
    keep_polys_by_name = {}
    original_features = []
    
    for fpath, file_actions in geojson_files_with_actions:
        merge_annots = {x.lower() for x in file_actions.get("merge", [])}
        subtract_annots = {x.lower() for x in file_actions.get("subtract", [])}
        keep_annots = {x.lower() for x in file_actions.get("keep", [])}
        
        try:
            with open(fpath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            features = data.get('features', [])
            if not features and data.get('type') == 'Feature':
                features = [data]
            
            for feat in features:
                props = feat.get('properties', {})
                cls = props.get('classification', {})
                name = cls.get('name') if isinstance(cls, dict) else props.get('name')
                if not name:
                    continue
                
                geom = shape(feat['geometry'])
                if not geom.is_valid:
                    geom = make_valid(geom)
                
                color_rgb = cls.get('colorRGB') if isinstance(cls, dict) else props.get('colorRGB', -16711936)
                
                name_lower = name.lower()
                if name_lower in merge_annots:
                    polys_to_merge.append(geom)
                    if include_original:
                        original_features.append(feat)
                
                if name_lower in subtract_annots:
                    polys_to_subtract.append(geom)
                    if include_original:
                        original_features.append(feat)
                
                if name_lower in keep_annots:
                    keep_polys_by_name.setdefault(name, []).append((geom, color_rgb))
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    final_features = []
    tissue_geom = None
    
    if polys_to_merge:
        tissue_geom = unary_union(polys_to_merge)
    
    if tissue_geom and polys_to_subtract:
        subtracted_geom = unary_union(polys_to_subtract)
        try:
            tissue_geom = tissue_geom.difference(subtracted_geom)
        except:
            tissue_geom = make_valid(tissue_geom).difference(make_valid(subtracted_geom))
            
    if tissue_geom and not tissue_geom.is_empty:
        if not tissue_geom.is_valid:
            tissue_geom = make_valid(tissue_geom)
        geoms = list(tissue_geom.geoms) if tissue_geom.geom_type == 'MultiPolygon' else [tissue_geom]
        valid_geoms = [g for g in geoms if g.area >= min_area]
        
        if valid_geoms:
            tissue_geom = unary_union(valid_geoms)
            feat = {
                "type": "Feature",
                "geometry": mapping(tissue_geom),
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": output_label, "colorRGB": -16711936},
                    "isLocked": False
                }
            }
            final_features.append(feat)
        else:
            tissue_geom = None
            
    # Process Keep separate annotations
    for annot_name, polys_and_colors in keep_polys_by_name.items():
        if not polys_and_colors:
            continue
        color_rgb = polys_and_colors[0][1]
        all_geoms = [g for g, c in polys_and_colors]
        merged_keep = unary_union(all_geoms)
        if not merged_keep.is_valid:
            merged_keep = make_valid(merged_keep)
            
        if polys_to_subtract:
            subtracted_geom = unary_union(polys_to_subtract)
            try:
                merged_keep = merged_keep.difference(subtracted_geom)
            except:
                merged_keep = make_valid(merged_keep).difference(make_valid(subtracted_geom))
                
        if tissue_geom and not tissue_geom.is_empty:
            try:
                merged_keep = merged_keep.intersection(tissue_geom)
            except:
                merged_keep = make_valid(merged_keep).intersection(make_valid(tissue_geom))
                
        if merged_keep.is_empty:
            continue
            
        feat = {
            "type": "Feature",
            "geometry": mapping(merged_keep),
            "properties": {
                "objectType": "annotation",
                "classification": {"name": annot_name, "colorRGB": color_rgb},
                "isLocked": False
            }
        }
        final_features.append(feat)
        
    if include_original:
        final_features.extend(original_features)
        
    if final_features:
        out_name = f"{image_stem}_merged.geojson"
        out_path = Path(output_dir) / out_name
        os.makedirs(output_dir, exist_ok=True)
        
        fc = {"type": "FeatureCollection", "features": final_features}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(fc, f)
        return True
    return False


# ==============================================================================
# PIPELINE STAGE 2: CREATE ZONES BACKEND
# ==============================================================================

def generate_zones_for_geojson(fpath: str, features_df: pd.DataFrame, tissue_label: str = "tissue") -> bool:
    """Generate inner/outer buffer zones for compartments and merge zones of the same level."""
    try:
        _ensure_heavy_imports()
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        features = data.get("features", [])
        
        # Group existing polygons
        polys_by_class = {}
        for feat in features:
            props = feat.get("properties", {})
            cls = props.get("classification", {}).get("name", "").lower()
            if not cls:
                cls = props.get("name", "").lower()
            geom = shape(feat.get("geometry", {}))
            if geom.is_empty:
                continue
            if cls:
                polys_by_class.setdefault(cls, []).append(geom)
                
        # Union tissue boundary for clipping if it exists
        tissue_boundary = Polygon()
        tissue_keys = [k for k in polys_by_class.keys() if tissue_label.lower() in k]
        if tissue_keys:
            tissue_polys = [g.simplify(10.0) for g in polys_by_class[tissue_keys[0]]]
            tissue_boundary = efficient_union(tissue_polys)
            if not tissue_boundary.is_valid:
                tissue_boundary = make_valid(tissue_boundary)
                
        new_features = []
        buf_resolution = 3
        
        # Determine maximum inner and outer zones defined
        max_inner = 0
        max_outer = 0
        active_configs = []
        
        if features_df is not None and not features_df.empty:
            for _, row in features_df.iterrows():
                if not bool(row.get("Create Zones", False)):
                    continue
                lbl = row.get("Annotation Label", "")
                if not lbl or str(lbl).lower() not in polys_by_class:
                    continue
                active_configs.append(row)
                max_inner = max(max_inner, int(row.get("Inner Zones", 0)))
                max_outer = max(max_outer, int(row.get("Outer Zones", 0)))
                
        # Process Outer Zones level by level
        for lvl in range(1, max_outer + 1):
            lvl_polys = []
            for row in active_configs:
                num_outer = int(row.get("Outer Zones", 0))
                if lvl > num_outer:
                    continue
                lbl = row["Annotation Label"]
                lbl_lower = lbl.lower()
                source_polys = polys_by_class[lbl_lower]
                simplified = [g.simplify(3.0) for g in source_polys if not g.is_empty]
                source_union = efficient_union(simplified)
                if source_union.is_empty:
                    continue
                if not source_union.is_valid:
                    source_union = make_valid(source_union)
                source_union = source_union.simplify(2.0)
                
                buffer_size = int(row.get("Zone Width (px)", 100))
                
                # Outer zone geometry at lvl
                dist_curr = lvl * buffer_size
                dist_prev = (lvl - 1) * buffer_size
                
                curr_geom = source_union.buffer(dist_curr, resolution=buf_resolution).simplify(2.0)
                if lvl == 1:
                    prev_geom = source_union
                else:
                    prev_geom = source_union.buffer(dist_prev, resolution=buf_resolution).simplify(2.0)
                    
                try:
                    zone = curr_geom.difference(prev_geom)
                except:
                    curr_geom = make_valid(curr_geom)
                    prev_geom = make_valid(prev_geom)
                    zone = curr_geom.difference(prev_geom)
                    
                if not tissue_boundary.is_empty:
                    try:
                        zone = zone.intersection(tissue_boundary)
                    except:
                        zone = make_valid(zone).intersection(tissue_boundary)
                        
                if not zone.is_empty:
                    lvl_polys.append(zone)
                    
            if lvl_polys:
                merged_lvl = efficient_union(lvl_polys)
                if not merged_lvl.is_empty:
                    if not merged_lvl.is_valid:
                        merged_lvl = make_valid(merged_lvl)
                    name = f"outer_{lvl}"
                    new_features.append({
                        "type": "Feature",
                        "geometry": mapping(merged_lvl),
                        "properties": {
                            "objectType": "annotation",
                            "classification": {"name": name, "colorRGB": -32768}
                        }
                    })
                    
        # Process Inner Zones level by level
        for lvl in range(1, max_inner + 1):
            lvl_polys = []
            for row in active_configs:
                num_inner = int(row.get("Inner Zones", 0))
                if lvl > num_inner:
                    continue
                lbl = row["Annotation Label"]
                lbl_lower = lbl.lower()
                source_polys = polys_by_class[lbl_lower]
                simplified = [g.simplify(3.0) for g in source_polys if not g.is_empty]
                source_union = efficient_union(simplified)
                if source_union.is_empty:
                    continue
                if not source_union.is_valid:
                    source_union = make_valid(source_union)
                source_union = source_union.simplify(2.0)
                
                buffer_size = int(row.get("Zone Width (px)", 100))
                
                # Inner zone geometry at lvl
                dist_curr = -lvl * buffer_size
                dist_prev = -(lvl - 1) * buffer_size
                
                curr_geom = source_union.buffer(dist_curr, resolution=buf_resolution).simplify(2.0)
                if not curr_geom.is_valid:
                    curr_geom = make_valid(curr_geom)
                if lvl == 1:
                    prev_geom = source_union
                else:
                    prev_geom = source_union.buffer(dist_prev, resolution=buf_resolution).simplify(2.0)
                    if not prev_geom.is_valid:
                        prev_geom = make_valid(prev_geom)
                        
                if prev_geom.is_empty:
                    zone = Polygon()
                else:
                    if curr_geom.is_empty:
                        zone = prev_geom
                    else:
                        try:
                            zone = prev_geom.difference(curr_geom)
                        except:
                            prev_geom = make_valid(prev_geom)
                            curr_geom = make_valid(curr_geom)
                            zone = prev_geom.difference(curr_geom)
                            
                if not zone.is_empty:
                    lvl_polys.append(zone)
                    
            if lvl_polys:
                merged_lvl = efficient_union(lvl_polys)
                if not merged_lvl.is_empty:
                    if not merged_lvl.is_valid:
                        merged_lvl = make_valid(merged_lvl)
                    name = f"inner_{lvl}"
                    new_features.append({
                        "type": "Feature",
                        "geometry": mapping(merged_lvl),
                        "properties": {
                            "objectType": "annotation",
                            "classification": {"name": name, "colorRGB": -16711936}
                        }
                    })
                    
        if new_features:
            features.extend(new_features)
            data["features"] = features
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        return True
    except Exception as e:
        print(f"Error generating zones: {e}")
        return False


# ==============================================================================
# PIPELINE STAGE 3: CELL ANNOTATION BACKEND
# ==============================================================================

def process_cell_annotation_dynamic(
    csv_path: str,
    geojson_path: str,
    output_path: str,
    features_df: pd.DataFrame,
    x_col_name: str = None,
    y_col_name: str = None,
    tissue_label: str = "tissue",
    edge_config: dict = None
) -> bool:
    """Annotate segmented cells with spatial information from a GeoJSON.

    The implementation lives in :mod:`spacemap.cell_spatial_annotation` so that
    this subtab, the Segmentation tab (which annotates inline, right after
    feature extraction) and the notebooks all produce identical columns.
    """
    try:
        from spacemap.cell_spatial_annotation import (
            EdgeTagConfig,
            annotate_cell_csv,
            rules_from_records,
        )

        records = features_df.to_dict("records") if features_df is not None and not features_df.empty else []
        rules = rules_from_records(records)

        edge = None
        if edge_config:
            edge = EdgeTagConfig(
                reference=str(edge_config.get("reference", tissue_label)),
                buffer_px=float(edge_config.get("buffer", 100.0)),
                column=str(edge_config.get("column", "edge_cell")),
            )

        summary = annotate_cell_csv(
            csv_path=csv_path,
            geojson_path=geojson_path,
            output_path=output_path,
            rules=rules,
            x_col=x_col_name,
            y_col=y_col_name,
            tissue_label=tissue_label,
            edge_config=edge,
        )
        for warning in summary.get("warnings", []):
            print(f"  Annotation warning: {warning}")
        return True
    except Exception as e:
        print(f"Error annotating cell CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# PIPELINE STAGE 4: EDIT CELL ANNOTATION BACKEND
# ==============================================================================

def perform_edit_annotation_update(csv_mode: str, edit_pairs: list, target_col: str, geojson_name: str, update_type: str, x_col: str, y_col: str, batch_col: str = None, cell_id_col: str = None, unified_csv_path: str = None, overwrite: bool = True):
    """Update or overwrite specific annotation columns on cell level in existing CSV files."""
    _ensure_heavy_imports()
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    
    if not edit_pairs:
        st.error("No matched pairs to process.")
        return False
        
    try:
        progress = st.progress(0)
        status = st.empty()
        
        def compute_updates_for_cells(x_coords, y_coords, geojson_path, name_to_match, up_type):
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geo_data = json.load(f)
            features = geo_data.get('features', [])
            
            matched_geoms = []
            for feat in features:
                props = feat.get('properties', {})
                cls = props.get('classification', {})
                name = cls.get('name') if isinstance(cls, dict) else props.get('name')
                if name and name.lower() == name_to_match.lower():
                    try:
                        geom = shape(feat.get('geometry'))
                        if not geom.is_valid:
                            geom = make_valid(geom)
                        matched_geoms.append(geom)
                    except Exception as e:
                        from spacemap_gui.ui_utils import log_exception
                        log_exception("parsing matched geometry", e)
            
            if not matched_geoms:
                if up_type == "Containment (1/0)":
                    return np.zeros(len(x_coords), dtype=int)
                else:
                    return np.full(len(x_coords), np.nan)
                    
            merged = unary_union(matched_geoms)
            if not merged.is_valid:
                merged = make_valid(merged)
                
            poly_list = split_sort_polygons(merged)
            geoms = [p['geom'] for p in poly_list]
            tree = STRtree(geoms) if geoms else None
            
            results = []
            for px, py in zip(x_coords, y_coords):
                pt = Point(px, py)
                if up_type == "Containment (1/0)":
                    inside = 0
                    if tree:
                        candidates = tree.query(pt)
                        for c_idx in candidates:
                            if poly_list[c_idx]['geom'].contains(pt):
                                inside = 1
                                break
                    results.append(inside)
                elif up_type == "Distance":
                    results.append(np.nan if merged.is_empty else merged.boundary.distance(pt))
                elif up_type == "Signed Distance":
                    if merged.is_empty:
                        results.append(np.nan)
                    else:
                        dist = merged.boundary.distance(pt)
                        inside = False
                        if tree:
                            candidates = tree.query(pt)
                            for c_idx in candidates:
                                if poly_list[c_idx]['geom'].contains(pt):
                                    inside = True
                                    break
                        results.append(dist if inside else -dist)
            return results

        if csv_mode == "Unified CSV File (All batches combined)":
            status.text("Loading unified CSV file...")
            df = pd.read_csv(unified_csv_path)
            
            if target_col not in df.columns:
                df[target_col] = 0 if update_type == "Containment (1/0)" else np.nan
                
            for idx, pair in enumerate(edit_pairs):
                b_id = pair["Batch ID"]
                g_path = pair["GeoJSON File"]
                
                status.text(f"Updating batch {b_id} using {os.path.basename(g_path)}...")
                
                batch_mask = df[batch_col].astype(str) == str(b_id)
                sub_df = df[batch_mask]
                if sub_df.empty:
                    continue
                    
                x_coords = sub_df[x_col].tolist()
                y_coords = sub_df[y_col].tolist()
                
                updates = compute_updates_for_cells(x_coords, y_coords, g_path, geojson_name, update_type)
                df.loc[batch_mask, target_col] = updates
                progress.progress((idx + 1) / len(edit_pairs))
                
            status.text("Saving updated unified CSV...")
            if overwrite:
                df.to_csv(unified_csv_path, index=False)
            else:
                out_path = unified_csv_path.replace(".csv", "_updated.csv")
                df.to_csv(out_path, index=False)
                st.info(f"Saved to {out_path}")
        else:
            for idx, pair in enumerate(edit_pairs):
                csv_path = pair["CSV File"]
                g_path = pair["GeoJSON File"]
                
                status.text(f"Processing {os.path.basename(csv_path)}...")
                df = pd.read_csv(csv_path)
                
                x_coords = df[x_col].tolist()
                y_coords = df[y_col].tolist()
                
                updates = compute_updates_for_cells(x_coords, y_coords, g_path, geojson_name, update_type)
                df[target_col] = updates
                
                if overwrite:
                    df.to_csv(csv_path, index=False)
                else:
                    out_path = csv_path.replace(".csv", "_updated.csv")
                    df.to_csv(out_path, index=False)
                    
                progress.progress((idx + 1) / len(edit_pairs))
                
        status.text("Annotation update completed!")
        progress.progress(1.0)
        return True
    except Exception as e:
        st.error(f"Error during update: {e}")
        return False


# ==============================================================================
# UI SCREENS
# ==============================================================================

def get_available_polygon_classes(rules_df: pd.DataFrame, tissue_label: str) -> list[str]:
    """Helper to dynamically compute all available polygon classes (only tissue & keep-separate & zones)."""
    classes = [tissue_label]
    base_compartments = {tissue_label.lower()}
    if rules_df is not None and not rules_df.empty:
        for _, row in rules_df.iterrows():
            lbl = row.get("Label", "")
            if not lbl:
                continue
            lbl_lower = lbl.lower()
            if lbl_lower != tissue_label.lower():
                if bool(row.get("Keep Separate", True)):
                    classes.append(lbl)
                    base_compartments.add(lbl_lower)
                    
    # Find max inner/outer zones from features_df in session state
    features_df = st.session_state.get('annot_features')
    max_inner = 0
    max_outer = 0
    if features_df is not None and not features_df.empty:
        for _, row in features_df.iterrows():
            lbl = row.get("Annotation Label", "")
            if not lbl or str(lbl).lower() not in base_compartments:
                continue
            if bool(row.get("Create Zones", False)):
                max_inner = max(max_inner, int(row.get("Inner Zones", 0)))
                max_outer = max(max_outer, int(row.get("Outer Zones", 0)))
                
    for i in range(max_inner):
        classes.append(f"inner_{i+1}")
    for i in range(max_outer):
        classes.append(f"outer_{i+1}")
        
    # Remove duplicates but preserve order
    seen = set()
    unique_classes = []
    for c in classes:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique_classes.append(c)
    return unique_classes


def generate_default_features(rules_df: pd.DataFrame, tissue_label: str) -> pd.DataFrame:
    """Generate default rows for the feature extraction configuration matrix."""
    avail = get_available_polygon_classes(rules_df, tissue_label)
    rows = []
    for c in avail:
        is_tissue = (c.lower() == tissue_label.lower())
        rows.append({
            "Annotation Label": c,
            "Binary (0/1)": True,
            "ROI Size": is_tissue,
            "ROI ID": is_tissue,
            "Distance": is_tissue,
            "Bed (Use 2 Annotations)": False,
            "Bed Comp. Target": "",
            "Group by Tissue ROI": False,
            "Create Zones": False,
            "Zone Width (px)": 100,
            "Inner Zones": 1,
            "Outer Zones": 1
        })
    return pd.DataFrame(rows)


def align_features_df(features_df: pd.DataFrame, rules_df: pd.DataFrame, tissue_label: str) -> pd.DataFrame:
    """Align features matrix rows on scan/rules updates, keeping checked states intact."""
    avail = get_available_polygon_classes(rules_df, tissue_label)
    if features_df is None or features_df.empty:
        return generate_default_features(rules_df, tissue_label)
        
    existing_config = {}
    for _, row in features_df.iterrows():
        lbl = row.get("Annotation Label")
        if lbl:
            existing_config[lbl.lower()] = row.to_dict()
            
    aligned_rows = []
    for c in avail:
        c_lower = c.lower()
        is_tissue = (c_lower == tissue_label.lower())
        if c_lower in existing_config:
            row_dict = existing_config[c_lower]
            aligned_rows.append({
                "Annotation Label": c,
                "Binary (0/1)": bool(row_dict.get("Binary (0/1)", True)),
                "ROI Size": bool(row_dict.get("ROI Size", is_tissue)),
                "ROI ID": bool(row_dict.get("ROI ID", is_tissue)),
                "Distance": bool(row_dict.get("Distance", is_tissue)),
                "Bed (Use 2 Annotations)": bool(row_dict.get("Bed (Use 2 Annotations)", False)),
                "Bed Comp. Target": str(row_dict.get("Bed Comp. Target", "")),
                "Group by Tissue ROI": bool(row_dict.get("Group by Tissue ROI", False)),
                "Create Zones": bool(row_dict.get("Create Zones", False)),
                "Zone Width (px)": int(row_dict.get("Zone Width (px)", 100)),
                "Inner Zones": int(row_dict.get("Inner Zones", 1)),
                "Outer Zones": int(row_dict.get("Outer Zones", 1))
            })
        else:
            aligned_rows.append({
                "Annotation Label": c,
                "Binary (0/1)": True,
                "ROI Size": is_tissue,
                "ROI ID": is_tissue,
                "Distance": is_tissue,
                "Bed (Use 2 Annotations)": False,
                "Bed Comp. Target": "",
                "Group by Tissue ROI": False,
                "Create Zones": False,
                "Zone Width (px)": 100,
                "Inner Zones": 1,
                "Outer Zones": 1
            })
    return pd.DataFrame(aligned_rows)


def render_before_after_visualization(viz_lid: str, img_path: str):
    """Plot original GeoJSONs and processed GeoJSONs side-by-side on background image thumbnail."""
    _ensure_heavy_imports()
    from matplotlib.patches import Polygon as MatplotlibPolygon
    
    try:
        ch_names, ch_colors, _, _ = get_channel_info(img_path)
        dapi_idx = 0
        for ch_idx, ch_name in enumerate(ch_names):
            if 'dapi' in ch_name.lower() or 'dna' in ch_name.lower():
                dapi_idx = ch_idx
                break
                
        img_data, full_shape, _ = load_preview_channels(img_path, [dapi_idx])
        bg_img = img_data[dapi_idx] if img_data else None
    except Exception as e:
        st.error(f"Error loading background image preview: {e}")
        return
        
    if bg_img is None:
        st.warning("Background image could not be loaded.")
        return
        
    # Normalize bg_img
    bg_img = ((bg_img - bg_img.min()) / (bg_img.max() - bg_img.min()) * 255).astype(np.uint8)
    
    h_disp, w_disp = bg_img.shape[:2]
    scale_x = w_disp / full_shape[1]
    scale_y = h_disp / full_shape[0]
    
    # Load Geometries
    grouped_files = st.session_state.get('annot_grouped_files', {})
    if viz_lid not in grouped_files:
        st.warning(f"No matched files for group {viz_lid}")
        return
        
    # Before features
    before_feats = []
    for fname, dir_idx in grouped_files[viz_lid]:
        fpath = os.path.join(st.session_state.annot_input_dirs[dir_idx], fname)
        if os.path.exists(fpath):
            before_feats.extend(load_geojson_features_cached(fpath))
            
    # After features
    after_feats = []
    after_path = os.path.join(st.session_state.annot_output_dir, f"{viz_lid}_merged.geojson")
    if os.path.exists(after_path):
        after_feats.extend(load_geojson_features_cached(after_path))
        
    # Get color mapping
    rules_df = st.session_state.annot_rules
    class_colors = {}
    if not rules_df.empty:
        for _, r in rules_df.iterrows():
            class_colors[r["Label"].lower()] = r["Color"]
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Render background
    ax1.imshow(bg_img, cmap='gray')
    ax2.imshow(bg_img, cmap='gray')
    
    def draw_geoms_on_ax(ax, feats):
        for feat in feats:
            props = feat.get('properties', {})
            cls = props.get('classification', {})
            name = cls.get('name') if isinstance(cls, dict) else props.get('name')
            
            color_hex = "#00FF00"
            if name:
                color_hex = class_colors.get(name.lower(), color_hex)
                
            geom = shape(feat['geometry'])
            polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
            
            for p in polys:
                coords = list(p.exterior.coords)
                scaled_coords = [(pt[0]*scale_x, pt[1]*scale_y) for pt in coords]
                if len(scaled_coords) > 2:
                    poly_patch = MatplotlibPolygon(scaled_coords, closed=True, facecolor=color_hex, edgecolor=color_hex, alpha=0.4, linewidth=1)
                    ax.add_patch(poly_patch)
                    
    draw_geoms_on_ax(ax1, before_feats)
    draw_geoms_on_ax(ax2, after_feats)
    
    ax1.set_title("Before: Input GeoJSONs")
    ax2.set_title("After: Merged & Zones")
    
    for ax in [ax1, ax2]:
        ax.axis("off")
        
    plt.tight_layout()
    st.pyplot(fig)


def render_unified_annotation(project_root: Path):
    """Consolidated UI combining Merge, Zones, and Cell Annotation."""
    render_start = time.perf_counter()
    
    defaults = {
        'annot_input_dirs': [str(project_root / 'data/annotations')] * 3,
        'annot_image_dir': str(project_root / 'data/codex'),
        'cell_seg_dir': st.session_state.get('tab4_csv_dir') or st.session_state.get('tab4_output_dir') or str(project_root / 'outputs/segmentation/cellpose/features'),
        'roi_output_dir': st.session_state.get('tab3_output_dir') or str(project_root / 'outputs/rois'),
        'annot_output_dir': str(project_root / 'outputs/processed_annotations/stage1a_merged_tissue'),
        'cell_out_dir': str(project_root / 'outputs/annotated_csv'),
        'annot_merged_name': 'tissue',
        'annot_rules': pd.DataFrame(columns=["Label", "Primary Action", "Keep Separate", "Color"]),
        'annot_features': pd.DataFrame(columns=["Annotation Label", "Binary (0/1)", "ROI Size", "ROI ID", "Distance", "Bed (Use 2 Annotations)", "Bed Comp. Target", "Group by Tissue ROI", "Create Zones", "Zone Width (px)", "Inner Zones", "Outer Zones"]),
        'annot_active_group_idx': None,
        'annot_all_discovered': {},
        'annot_file_inclusion': {},
        'cell_x_col': 'x',
        'cell_y_col': 'y',
        'cell_containment_col': 'tissue',
        'cell_containment_annot': 'tissue',
        'cell_calc_dist': True,
        'cell_dist_ref': 'Tumor',
        'cell_edge_tag': False,
        'cell_edge_ref': 'tissue',
        'cell_edge_buffer': 100.0,
        'cell_edge_col': 'edge_cell',
        'cell_bed_comparisons': [],
        'annot_match_regex': r'^(P\d+(_P\d+)?_roi_\d+)',
        'annot_strip_suffixes': '_merged, _zones, _fully_annotated, _cell_features, _features, _corrected, _corr, _cleaned, _cellpose_features, _cellpose',
        'annot_strip_prefixes': '',
        'annot_fuzzy_threshold': 0.3,
        'annot_file_label_map': {}
    }
    
    keys_list = list(st.session_state.keys())
    updated = False
    for k, v in defaults.items():
        if k not in keys_list:
            st.session_state[k] = v
            print(f"DEBUG LOOP: {k} not in keys_list. Initializing to {v}", flush=True)
            updated = True
        else:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame):
                continue
            if isinstance(val, list) and val == [] and v != []:
                st.session_state[k] = v
                print(f"DEBUG LOOP: {k} is empty list. Initializing to {v}", flush=True)
                updated = True
            elif isinstance(val, str) and val == "" and v != "":
                st.session_state[k] = v
                print(f"DEBUG LOOP: {k} is empty string. Initializing to {v}", flush=True)
                updated = True
            
    if updated:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TAB4-LEGACY: Initialized default configuration values", flush=True)
        st.rerun()
            
    if 'annot_rules' in keys_list and isinstance(st.session_state.annot_rules, pd.DataFrame):
        cols_to_drop = [c for c in ["Create Zones", "Zone Width (px)", "Inner Zones", "Outer Zones"] if c in st.session_state.annot_rules.columns]
        if cols_to_drop:
            st.session_state.annot_rules = st.session_state.annot_rules.drop(columns=cols_to_drop)

    # Align features automatically on load/rerun
    st.session_state.annot_features = align_features_df(
        st.session_state.annot_features,
        st.session_state.annot_rules,
        st.session_state.annot_merged_name
    )
    # Initialize local variables to prevent UnboundLocalError in conditionally executed blocks
    grouped_files = st.session_state.get('annot_grouped_files', {})
    btn_autofill = False
    btn_clear = False
    save_btn = False
    run_btn = False
    edited_rules = pd.DataFrame()
    edited_features = pd.DataFrame()
    edited_pairs = pd.DataFrame()

    st.subheader("1. Setup & Map Annotations")
        
    left_col, right_col = st.columns([1.3, 0.7])
    
    with left_col:
        # === SECTION 1: DIRECTORIES ===
        st.subheader("1. Directories")
        
        st.markdown("**Input Directories** (Scan for annotation GeoJSON files)")
        dirs_to_remove = []
        for i in range(len(st.session_state.annot_input_dirs)):
            is_browsing = st.session_state.get('annot_browse_idx') == i
            if not is_browsing:
                c1, c2, c3 = st.columns([5, 1, 1])
                in_key = f"annot_in_{i}"
                
                # Safely pop/get widget state value to prevent Streamlit widget from reverting
                default_val = st.session_state.annot_input_dirs[i]
                val_in_state = st.session_state.pop(in_key, default_val)
                
                new_dir = c1.text_input(
                    "Directory", 
                    value=val_in_state,
                    key=in_key, 
                    label_visibility="collapsed"
                )
                if new_dir != st.session_state.annot_input_dirs[i]:
                    temp = list(st.session_state.annot_input_dirs)
                    temp[i] = new_dir
                    st.session_state.annot_input_dirs = temp
                if c2.button("Browse", key=f"annot_browse_{i}"):
                    st.session_state.annot_browse_idx = i
                    st.rerun()
                if c3.button("Remove", key=f"annot_del_{i}"):
                    dirs_to_remove.append(i)
            else:
                idx = st.session_state.annot_browse_idx
                path = render_file_browser(
                    st.session_state.annot_input_dirs[idx], 
                    f"annot_browser_{idx}", 
                    'directory'
                )
                if path == "CANCEL":
                    st.session_state.annot_browse_idx = None
                    st.rerun()
                elif path:
                    st.session_state.annot_input_dirs[idx] = path
                    st.session_state[f"annot_in_{idx}"] = path
                    st.session_state.annot_browse_idx = None
                    st.rerun()
                    
        if dirs_to_remove:
            temp = list(st.session_state.annot_input_dirs)
            for ri in sorted(dirs_to_remove, reverse=True):
                temp.pop(ri)
            st.session_state.annot_input_dirs = temp
            st.rerun()
            
        if st.button("+ Add Directory", key="add_dir_btn"):
            st.session_state.annot_input_dirs.append("")
            st.rerun()
            
        directory_input_with_browse(
            label="Cell Segmentation CSV Directory",
            text_key="cell_seg_dir",
            default_value=st.session_state.cell_seg_dir,
            browser_key_prefix="cell_seg_dir_ann",
        )
        
        directory_input_with_browse(
            label="ROIs Directory",
            text_key="roi_output_dir",
            default_value=st.session_state.roi_output_dir,
            browser_key_prefix="roi_dir_ann",
        )
        
        directory_input_with_browse(
            label="Merged GeoJSON Output Directory",
            text_key="annot_output_dir",
            default_value=st.session_state.annot_output_dir,
            browser_key_prefix="ann_out_dir",
        )
        
        directory_input_with_browse(
            label="Annotated CSV Output Directory",
            text_key="cell_out_dir",
            default_value=st.session_state.cell_out_dir,
            browser_key_prefix="cell_out_dir_ann",
        )
        
        with st.expander("🔍 Advanced Matching Settings", expanded=False):
            st.text_input(
                "Prefix Extraction Regex",
                value=st.session_state.get('annot_match_regex', r'^(P\d+(_P\d+)?_roi_\d+)'),
                key="annot_match_regex",
                help="Regex to extract prefix (link ID) from GeoJSON filenames to group them."
            )
            st.text_input(
                "Suffixes to Strip (comma-separated)",
                value=st.session_state.get('annot_strip_suffixes', '_merged, _zones, _fully_annotated, _cell_features, _features, _corrected, _corr, _cleaned, _cellpose_features, _cellpose'),
                key="annot_strip_suffixes",
                help="Suffixes to strip from filenames (without extension) when matching."
            )
            st.text_input(
                "Prefixes to Strip (comma-separated)",
                value=st.session_state.get('annot_strip_prefixes', ''),
                key="annot_strip_prefixes",
                help="Prefixes to strip from filenames when matching."
            )
            st.slider(
                "Fuzzy Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get('annot_fuzzy_threshold', 0.3)),
                step=0.05,
                key="annot_fuzzy_threshold",
                help="Minimum similarity score (0 to 1) for SequenceMatcher ratio to match GeoJSON with CSV."
            )

            
        # Scan files button
        if st.button("Scan Directories", type="primary", width='stretch', key="annot_scan_btn"):
            if not any(st.session_state.annot_input_dirs):
                st.warning("Please provide at least one annotation directory.")
            else:
                with st.spinner("Scanning annotation directories..."):
                    all_discovered = {}
                    unique_labels = set()
                    
                    for dir_idx, d_path in enumerate(st.session_state.annot_input_dirs):
                        if not d_path or not os.path.isdir(d_path):
                            continue
                        files = list_geojson_files(d_path)
                        all_discovered[dir_idx] = files
                        
                        for f in files:
                            fpath = os.path.join(d_path, f)
                            try:
                                labels = load_geojson_labels_cached(fpath)
                                unique_labels.update(labels)
                            except Exception as e:
                                from spacemap_gui.ui_utils import log_exception
                                log_exception(f"loading labels from {f}", e)
                                
                    # Image group matching (link IDs / slide IDs matching)
                    grouped_files = {}
                    file_label_map = {}
                    
                    pattern = st.session_state.get('annot_match_regex', r'^(P\d+(_P\d+)?_roi_\d+)')
                    try:
                        pat_compiled = re.compile(pattern, re.IGNORECASE)
                    except re.error as e:
                        st.warning(f"Invalid regex pattern, falling back to default: {e}")
                        pattern = r'^(P\d+(_P\d+)?_roi_\d+)'
                        pat_compiled = re.compile(pattern, re.IGNORECASE)
                    
                    def extract_prefix(filename, pat_comp):
                        name = Path(filename).name
                        for ext in ['.geojson', '.json', '.csv', '.ome.tiff', '.ome.tif', '.tiff', '.tif']:
                            if name.lower().endswith(ext):
                                name = name[:-len(ext)]
                                break
                        m = pat_comp.match(name)
                        if m:
                            return m.group(1) if m.groups() else m.group(0)
                        # Fallback for s10r1 style
                        m2 = re.match(r'^([a-zA-Z0-9]+(?:_roi_\d+|_r\d+|_region_\d+)?)', name, re.IGNORECASE)
                        if m2:
                            return m2.group(1)
                        # Default split
                        parts = name.split('_')
                        if parts:
                            return parts[0]
                        return name

                    # Collect all unique prefixes across all sources to align them
                    all_prefixes = set()
                    
                    # 1. Prefixes from GeoJSONs
                    for dir_idx, files in all_discovered.items():
                        for f in files:
                            p = extract_prefix(f, pat_compiled)
                            if p:
                                all_prefixes.add(p)
                                
                    # 2. Prefixes from CSVs
                    csv_files = []
                    if os.path.isdir(st.session_state.cell_seg_dir):
                        csv_files = sorted([f for f in os.listdir(st.session_state.cell_seg_dir) if f.endswith('.csv')])
                        for f in csv_files:
                            p = extract_prefix(f, pat_compiled)
                            if p:
                                all_prefixes.add(p)
                                
                    # 3. Prefixes from Images
                    image_files = []
                    for s_dir in [st.session_state.roi_output_dir, st.session_state.annot_image_dir]:
                        if s_dir and os.path.isdir(s_dir):
                            image_files.extend([f for f in os.listdir(s_dir) if f.endswith(('.qptiff', '.ome.tiff', '.tiff', '.tif'))])
                    image_files = sorted(list(set(image_files)))
                    for f in image_files:
                        p = extract_prefix(f, pat_compiled)
                        if p:
                            all_prefixes.add(p)
                            
                    matched_pairs_list = []
                    for lid in sorted(list(all_prefixes)):
                        row_data = {"Prefix": lid}
                        
                        # Match GeoJSON for each input directory
                        for dir_idx, d_path in enumerate(st.session_state.annot_input_dirs):
                            col_name = f"GeoJSON: {os.path.basename(d_path)} (Dir {dir_idx})"
                            if d_path and os.path.isdir(d_path):
                                files_in_dir = all_discovered.get(dir_idx, [])
                                row_data[col_name] = find_best_match_for_prefix(lid, files_in_dir)
                            else:
                                row_data[col_name] = "No Match Found"
                                
                        # Match CSV
                        row_data["Matched CSV File"] = find_best_match_for_prefix(lid, csv_files)
                        
                        # Match Image
                        row_data["Matched Image File"] = find_best_match_for_prefix(lid, image_files)
                        
                        # Process checkbox
                        cb_key = f"annot_proc_group_{lid}"
                        row_data["Process"] = st.session_state.get(cb_key, True)
                        
                        matched_pairs_list.append(row_data)
                        
                    # Build grouped_files map based on the matched_pairs_list
                    for row in matched_pairs_list:
                        lid = row["Prefix"]
                        grouped_files[lid] = []
                        for dir_idx, d_path in enumerate(st.session_state.annot_input_dirs):
                            col_name = f"GeoJSON: {os.path.basename(d_path)} (Dir {dir_idx})"
                            fname = row.get(col_name)
                            if fname and fname != "No Match Found":
                                grouped_files[lid].append((fname, dir_idx))
                                
                    # Populate file_label_map for all found GeoJSONs
                    for dir_idx, files in all_discovered.items():
                        for f in files:
                            try:
                                labels = load_geojson_labels_cached(os.path.join(st.session_state.annot_input_dirs[dir_idx], f))
                                file_label_map[f"{dir_idx}|{f}"] = labels
                            except Exception as e:
                                from spacemap_gui.ui_utils import log_exception
                                log_exception(f"loading labels from {f}", e)
                                
                    st.session_state.annot_all_discovered = all_discovered
                    st.session_state.annot_grouped_files = grouped_files
                    st.session_state.annot_file_label_map = file_label_map
                    st.session_state.annot_matched_pairs_df = pd.DataFrame(matched_pairs_list)
                    
                    # Update rules dataframe
                    current_rules = st.session_state.annot_rules
                    existing_labels = set(current_rules["Label"].tolist()) if not current_rules.empty else set()
                    new_labels = unique_labels - existing_labels
                    
                    if new_labels:
                        new_rows = []
                        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#C0C0C0', '#808080']
                        for i, lbl in enumerate(new_labels):
                            default_action = "None"
                            default_keep = True
                            if "tumor" in lbl.lower():
                                default_action = "Merge"
                                default_keep = False
                            if "necro" in lbl.lower() or "artifact" in lbl.lower():
                                default_action = "Subtract"
                                default_keep = False
                                
                            new_rows.append({
                                "Label": lbl,
                                "Primary Action": default_action,
                                "Keep Separate": default_keep,
                                "Color": colors[i % len(colors)]
                            })
                            
                        if current_rules.empty:
                            st.session_state.annot_rules = pd.DataFrame(new_rows)
                        else:
                            st.session_state.annot_rules = pd.concat([current_rules, pd.DataFrame(new_rows)], ignore_index=True)
                            
                    st.session_state.annot_features = align_features_df(st.session_state.annot_features, st.session_state.annot_rules, st.session_state.annot_merged_name)
                    st.success("Scan completed successfully!")
                    st.rerun()
                           # Rules configuration
        if not st.session_state.annot_rules.empty:
            with st.form("annotation_settings_form"):
                st.caption("💡 **Tip:** Edit rules, features, and colors offline, then click **Save Project** at the bottom to save your changes.")
                
                st.markdown("**Annotation Rules**")
                rule_config = {
                    "Label": st.column_config.TextColumn("Label", disabled=True, width="medium"),
                    "Primary Action": st.column_config.SelectboxColumn(
                        "Primary Action",
                        options=["Merge", "Subtract", "None"],
                        help="Merge or Subtract from main tissue mask.",
                        width="small"
                    ),
                    "Keep Separate": st.column_config.CheckboxColumn(
                        "Keep Separate",
                        help="Also save this annotation as a separate layer in the output.",
                        width="small"
                    )
                }
                
                # Filter rules to only have rules_config keys to prevent UI issues
                display_rules = st.session_state.annot_rules[[col for col in rule_config.keys() if col in st.session_state.annot_rules.columns]]
                
                edited_rules = st.data_editor(
                    display_rules,
                    column_config=rule_config,
                    hide_index=True,
                    key="annot_rules_editor",
                    width='stretch'
                )
                
                # Color selection grid
                st.markdown("🎨 **Label Colors**")
                color_cols = st.columns(4)
                for idx, row in display_rules.iterrows():
                    lbl = row["Label"]
                    orig_row = st.session_state.annot_rules[st.session_state.annot_rules["Label"] == lbl]
                    col_hex = orig_row["Color"].values[0] if not orig_row.empty else "#FFFFFF"
                    if not isinstance(col_hex, str) or not col_hex.startswith("#") or len(col_hex) != 7:
                        col_hex = "#FFFFFF"
                    with color_cols[idx % 4]:
                        st.color_picker(lbl, value=col_hex, key=f"annot_color_{lbl}")
                
                st.session_state.annot_merged_name = st.text_input(
                    "Merged Polygon Classification Name",
                    value=st.session_state.annot_merged_name,
                    key="annot_merged_name_widget",
                    help="Name for the output polygon created by merging/subtracting annotations (e.g., 'tissue')."
                )
                        
                # === CELL FEATURE EXTRACTION CONFIGURATION ===
                st.subheader("2. Cell Feature Extraction Settings")
                with st.expander("Configure Cell Feature Extraction", expanded=True):
                    # Coordinates detection
                    detected_cols = []
                    csv_dir = st.session_state.cell_seg_dir
                    if csv_dir and os.path.isdir(csv_dir):
                        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])
                        if csv_files:
                            first_csv = os.path.join(csv_dir, csv_files[0])
                            try:
                                df_temp = pd.read_csv(first_csv, nrows=0)
                                detected_cols = list(df_temp.columns)
                            except Exception as e:
                                from spacemap_gui.ui_utils import log_exception
                                log_exception("reading CSV header for column detection", e)
                                
                    default_cols = ['x', 'y', 'x_centroid', 'y_centroid', 'centroid_x', 'centroid_y']
                    all_cols = []
                    if detected_cols:
                        all_cols = detected_cols
                    else:
                        all_cols = default_cols
                        
                    for c_val in [st.session_state.cell_x_col, st.session_state.cell_y_col]:
                        if c_val and c_val not in all_cols:
                            all_cols.append(c_val)
                            
                    try:
                        x_idx = all_cols.index(st.session_state.cell_x_col)
                    except ValueError:
                        x_idx = 0
                        for i, col in enumerate(all_cols):
                            if col.lower() in ['x', 'x_centroid', 'centroid_x']:
                                x_idx = i
                                break
                    try:
                        y_idx = all_cols.index(st.session_state.cell_y_col)
                    except ValueError:
                        y_idx = 0
                        for i, col in enumerate(all_cols):
                            if col.lower() in ['y', 'y_centroid', 'centroid_y']:
                                y_idx = i
                                break
                                
                    c_c1, c_c2 = st.columns(2)
                    cell_x_col = c_c1.selectbox("X Coordinate Column", options=all_cols, index=x_idx, key="cell_x_col_widget")
                    cell_y_col = c_c2.selectbox("Y Coordinate Column", options=all_cols, index=y_idx, key="cell_y_col_widget")
                    
                    # Feature extraction config table
                    avail_classes = get_available_polygon_classes(st.session_state.annot_rules, st.session_state.annot_merged_name)
                    
                    st.markdown("**Feature Extraction Configuration Table**")
                    
                    # Collapsible explanations for column parameters
                    with st.expander("Column Explanations: Bed Comp. Target & Group by Tissue ROI"):
                        st.markdown("""
                        * **Bed Comp. Target (Bed Proximity Competition)**: When Bed is enabled, this specifies the target compartment that competes against the current row's compartment. The cell features will compute which of these two compartments is closer to each individual cell.
                        * **Group by Tissue ROI**: When enabled, the algorithm groups cells and targets by their containing tissue region (ROI) before calculating distances or Bed values. This prevents cells in one tissue region from being associated with compartments in an unrelated tissue region.
                        """)
                    
                    c_auto, c_clear = st.columns([2, 1])
                    btn_autofill = c_auto.form_submit_button("Autofill Features from Rules", key="btn_autofill_features")
                    btn_clear = c_clear.form_submit_button("Clear Features", key="btn_clear_features")
                        
                    features_df = st.session_state.annot_features
                    for col_name in ["Annotation Label", "Binary (0/1)", "ROI Size", "ROI ID", "Distance", "Bed (Use 2 Annotations)", "Bed Comp. Target", "Group by Tissue ROI", "Create Zones", "Zone Width (px)", "Inner Zones", "Outer Zones"]:
                        if col_name not in features_df.columns:
                            if col_name in ["Annotation Label", "Bed Comp. Target"]:
                                features_df[col_name] = ""
                            elif col_name in ["Zone Width (px)", "Inner Zones", "Outer Zones"]:
                                features_df[col_name] = 100 if col_name == "Zone Width (px)" else 1
                            else:
                                features_df[col_name] = False
                            
                    edited_features = st.data_editor(
                        features_df,
                        column_config={
                            "Annotation Label": st.column_config.TextColumn("Annotation Label", disabled=True, width="medium"),
                            "Binary (0/1)": st.column_config.CheckboxColumn(
                                "Binary (0/1)",
                                help="Check to generate a 0/1 containment feature for this compartment.",
                                width="small"
                            ),
                            "ROI Size": st.column_config.CheckboxColumn(
                                "ROI Size",
                                help="Check to calculate containing ROI area.",
                                width="small"
                            ),
                            "ROI ID": st.column_config.CheckboxColumn(
                                "ROI ID",
                                help="Check to calculate containing ROI index (1-based).",
                                width="small"
                            ),
                            "Distance": st.column_config.CheckboxColumn(
                                "Distance",
                                help="Check to calculate signed distance to this compartment boundary.",
                                width="small"
                            ),
                            "Bed (Use 2 Annotations)": st.column_config.CheckboxColumn(
                                "Bed (Use 2 Annotations)",
                                help="Check to compute Bed competition proximity feature against a target.",
                                width="small"
                            ),
                            "Bed Comp. Target": st.column_config.SelectboxColumn(
                                "Bed Comp. Target",
                                options=[""] + avail_classes,
                                help="Target compartment to compete against in proximity Bed calculation.",
                                width="medium"
                            ),
                            "Group by Tissue ROI": st.column_config.CheckboxColumn(
                                "Group by Tissue ROI",
                                help="Group cells and targets by tissue ROI before calculating distances or Bed values.",
                                width="small"
                            ),
                            "Create Zones": st.column_config.CheckboxColumn(
                                "Create Zones",
                                help="Check to generate inner/outer buffer zones around this compartment.",
                                width="small"
                            ),
                            "Zone Width (px)": st.column_config.NumberColumn(
                                "Zone Width (px)",
                                help="Width of each zone buffer in pixels.",
                                min_value=1,
                                max_value=10000,
                                step=1,
                                width="small"
                            ),
                            "Inner Zones": st.column_config.NumberColumn(
                                "Inner Zones",
                                help="Number of inner/negative buffer zones.",
                                min_value=0,
                                max_value=20,
                                step=1,
                                width="small"
                            ),
                            "Outer Zones": st.column_config.NumberColumn(
                                "Outer Zones",
                                help="Number of outer/positive buffer zones.",
                                min_value=0,
                                max_value=20,
                                step=1,
                                width="small"
                            )
                        },
                        hide_index=True,
                        key="annot_features_editor",
                        width='stretch'
                    )
                    
                    # Edge Cell Tagging Config
                    st.markdown("---")
                    st.markdown("**Edge Cell Tagging Settings**")
                    cell_edge_tag = st.checkbox(
                        "Enable Edge Cell Tagging",
                        value=st.session_state.get('cell_edge_tag', False),
                        help="Check to flag cells located near the boundary of a reference compartment.",
                        key="cell_edge_tag_widget"
                    )
                    
                    if cell_edge_tag:
                        ec1, ec2, ec3 = st.columns(3)
                        cell_edge_ref = ec1.text_input(
                            "Edge Reference Compartment",
                            value=st.session_state.get('cell_edge_ref', st.session_state.annot_merged_name),
                            help="Name of the classification layer (e.g. tissue, tumor) used as boundary reference.",
                            key="cell_edge_ref_widget"
                        )
                        
                        cell_edge_buffer = ec2.number_input(
                            "Edge Buffer Width (px)",
                            value=float(st.session_state.get('cell_edge_buffer', 100.0)),
                            min_value=0.0,
                            max_value=10000.0,
                            step=1.0,
                            help="Cells within this distance from the boundary will be flagged.",
                            key="cell_edge_buffer_widget"
                        )
                        
                        cell_edge_col = ec3.text_input(
                            "Edge Column Name",
                            value=st.session_state.get('cell_edge_col', 'edge_cell'),
                            help="Name of the output column in the annotated CSV.",
                            key="cell_edge_col_widget"
                        )
                # === SECTION 3: IMAGE GROUPS & RUN ===
                matched_df = st.session_state.get('annot_matched_pairs_df')
                if matched_df is not None and not matched_df.empty:
                    st.subheader("3. Matched File Groups")
                    
                    csv_files = []
                    if os.path.isdir(st.session_state.cell_seg_dir):
                        csv_files = sorted([f for f in os.listdir(st.session_state.cell_seg_dir) if f.endswith('.csv')])
                        
                    image_files = []
                    for s_dir in [st.session_state.roi_output_dir, st.session_state.annot_image_dir]:
                        if s_dir and os.path.isdir(s_dir):
                            image_files.extend([f for f in os.listdir(s_dir) if f.endswith(('.qptiff', '.ome.tiff', '.tiff', '.tif'))])
                    image_files = sorted(list(set(image_files)))
                        
                    # Build column configuration dynamically
                    column_config = {
                        "Prefix": st.column_config.TextColumn("Image Prefix", disabled=True),
                        "Matched CSV File": st.column_config.SelectboxColumn(
                            "Matched CSV",
                            options=["No Match Found"] + csv_files,
                            required=False,
                            help="Select the matching segmentation CSV file for this group."
                        ),
                        "Matched Image File": st.column_config.SelectboxColumn(
                            "Matched Image",
                            options=["No Match Found"] + image_files,
                            required=False,
                            help="Select the matching background image (OME-TIFF) for this group."
                        ),
                        "Process": st.column_config.CheckboxColumn("Process")
                    }
                    
                    all_discovered = st.session_state.get('annot_all_discovered', {})
                    # Add dynamic GeoJSON selectbox columns
                    for dir_idx, d_path in enumerate(st.session_state.annot_input_dirs):
                        if d_path and os.path.isdir(d_path):
                            col_name = f"GeoJSON: {os.path.basename(d_path)} (Dir {dir_idx})"
                            files_in_dir = all_discovered.get(dir_idx, [])
                            column_config[col_name] = st.column_config.SelectboxColumn(
                                f"GeoJSON: {os.path.basename(d_path)}",
                                options=["No Match Found"] + files_in_dir,
                                required=False,
                                help=f"Select the matching GeoJSON file from directory {d_path}."
                            )
                            
                    edited_pairs = st.data_editor(
                        matched_df,
                        column_config=column_config,
                        hide_index=True,
                        key="annot_pairs_editor",
                        width='stretch'
                    )
                    st.session_state.annot_matched_pairs_df = edited_pairs
                    
                    # Update reconstructed grouped_files based on edited_pairs in session state
                    reconstructed_grouped_files = {}
                    for _, row in edited_pairs.iterrows():
                        lid = row["Prefix"]
                        reconstructed_grouped_files[lid] = []
                        for dir_idx, d_path in enumerate(st.session_state.annot_input_dirs):
                            col_name = f"GeoJSON: {os.path.basename(d_path)} (Dir {dir_idx})"
                            if col_name in row:
                                fname = row[col_name]
                                if fname and fname != "No Match Found":
                                    reconstructed_grouped_files[lid].append((fname, dir_idx))
                    st.session_state.annot_grouped_files = reconstructed_grouped_files
                    
                    st.markdown("---")
                    
                    c_run_ann, c_save_proj = st.columns(2)
                    with c_run_ann:
                        run_btn = st.form_submit_button("Run Unified Annotation", type="primary", width="stretch", key="btn_run_un_ann")
                    save_container = c_save_proj
                else:
                    run_btn = False

                save_button = (
                    save_container.form_submit_button
                    if matched_df is not None and not matched_df.empty
                    else st.form_submit_button
                )
                save_btn = save_button("Save Project", width="stretch", key="tab5_unified_save_project_btn")

            # Outside the form: process submit triggers
            if btn_autofill or btn_clear or save_btn or run_btn:
                # 1. Update rules and colors
                updated_rules = edited_rules.copy()
                color_list = []
                for idx, row in updated_rules.iterrows():
                    lbl = row["Label"]
                    picker_key = f"annot_color_{lbl}"
                    if picker_key in st.session_state:
                        color_list.append(st.session_state[picker_key])
                    else:
                        orig_row = st.session_state.annot_rules[st.session_state.annot_rules["Label"] == lbl]
                        orig_color = orig_row["Color"].values[0] if not orig_row.empty else "#FFFFFF"
                        color_list.append(orig_color)
                updated_rules["Color"] = color_list
                st.session_state.annot_rules = updated_rules

                # 2. Update merged name
                st.session_state.annot_merged_name = st.session_state.annot_merged_name_widget

                # 3. Update X and Y columns
                st.session_state.cell_x_col = st.session_state.cell_x_col_widget
                st.session_state.cell_y_col = st.session_state.cell_y_col_widget

                # 4. Update Edge Tagging settings
                st.session_state.cell_edge_tag = st.session_state.cell_edge_tag_widget
                if st.session_state.cell_edge_tag:
                    if "cell_edge_ref_widget" in st.session_state:
                        st.session_state.cell_edge_ref = st.session_state.cell_edge_ref_widget
                    if "cell_edge_buffer_widget" in st.session_state:
                        st.session_state.cell_edge_buffer = st.session_state.cell_edge_buffer_widget
                    if "cell_edge_col_widget" in st.session_state:
                        st.session_state.cell_edge_col = st.session_state.cell_edge_col_widget

                # 5. Update Matched File Groups Process choices
                if matched_df is not None and not matched_df.empty:
                    for _, row in edited_pairs.iterrows():
                        st.session_state[f"annot_proc_group_{row['Prefix']}"] = row["Process"]
                
                # Perform the specific button actions
                if btn_autofill:
                    st.session_state.annot_features = generate_default_features(st.session_state.annot_rules, st.session_state.annot_merged_name)
                    st.success("Features autofilled from rules!")
                    st.rerun()
                elif btn_clear:
                    cleared_df = edited_features.copy()
                    for col in ["Binary (0/1)", "ROI Size", "ROI ID", "Distance", "Bed (Use 2 Annotations)", "Group by Tissue ROI", "Create Zones"]:
                        if col in cleared_df.columns:
                            cleared_df[col] = False
                    for col in ["Zone Width (px)", "Inner Zones", "Outer Zones"]:
                        if col in cleared_df.columns:
                            cleared_df[col] = 100 if col == "Zone Width (px)" else 1
                    if "Bed Comp. Target" in cleared_df.columns:
                        cleared_df["Bed Comp. Target"] = ""
                    st.session_state.annot_features = cleared_df
                    st.success("Feature table cleared!")
                    st.rerun()
                elif save_btn:
                    st.session_state.annot_features = edited_features
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab4_annotation")
                    st.success("Project settings saved to configuration file.")
                    st.rerun()
                elif run_btn:
                    st.session_state.annot_features = edited_features
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab4_annotation")
                    
            if run_btn:
                grouped_files = st.session_state.get('annot_grouped_files', {})
                to_proc = [row["Prefix"] for _, row in edited_pairs.iterrows() if row["Process"]]
                if not to_proc:
                    st.warning("No image groups checked for processing.")
                    record_process_event(
                        "tab4",
                        "Annotation preflight failed",
                        status="warning",
                        details={"reason": "No image groups selected"},
                    )
                else:
                    progress = st.progress(0)
                    status = st.empty()
                    step_status = st.empty()
                    run_started_at = time.time()
                    record_process_event(
                        "tab4",
                        "Annotation started",
                        details={
                            "groups_selected": len(to_proc),
                            "output_directory": st.session_state.annot_output_dir,
                            "cell_output_directory": st.session_state.cell_out_dir,
                        },
                    )
                    
                    def log_step(msg: str):
                        ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                        full_msg = f"{ts} {msg}"
                        print(full_msg)
                        step_status.write(full_msg)
                        status.text(msg)
                        record_process_event("tab4", "Annotation step", message=msg)
                        
                    log_step(f"Starting unified processing for {len(to_proc)} groups...")
                    success_count = 0
                    file_label_map = st.session_state.get("annot_file_label_map", {})
                    
                    for idx, lid in enumerate(to_proc):
                        log_step(f"Processing group [{idx+1}/{len(to_proc)}]: {lid}")
                        
                        group_files_list = grouped_files.get(lid, [])
                        files_payload = []
                        for fname, dir_idx in group_files_list:
                            annot_actions = {"merge": [], "subtract": [], "keep": []}
                            map_key = f"{dir_idx}|{fname}"
                            labels = file_label_map.get(map_key, file_label_map.get(fname, []))
                            if not labels:
                                try:
                                    labels = load_geojson_labels_cached(os.path.join(st.session_state.annot_input_dirs[dir_idx], fname))
                                    file_label_map[map_key] = labels
                                except Exception as le:
                                    print(f"Error loading labels for {fname}: {le}")
                                    
                            for l in labels:
                                rule = st.session_state.annot_rules[st.session_state.annot_rules["Label"].str.lower() == l.lower()]
                                if not rule.empty:
                                    p_act = rule.iloc[0]['Primary Action']
                                    k_sep = rule.iloc[0]['Keep Separate']
                                    if p_act == "Merge": annot_actions["merge"].append(l)
                                    elif p_act == "Subtract": annot_actions["subtract"].append(l)
                                    if k_sep: annot_actions["keep"].append(l)
                                    
                            fpath = os.path.join(st.session_state.annot_input_dirs[dir_idx], fname)
                            files_payload.append((fpath, annot_actions))
                            
                        # 1. Merge & Clean
                        merged_geojson_path = os.path.join(st.session_state.annot_output_dir, f"{lid}_merged.geojson")
                        merge_ok = process_merge_and_clean(
                            image_stem=lid,
                            geojson_files_with_actions=files_payload,
                            output_label=st.session_state.annot_merged_name,
                            min_area=300000,
                            include_original=False,
                            output_dir=st.session_state.annot_output_dir
                        )
                        
                        if merge_ok and os.path.exists(merged_geojson_path):
                            log_step(f"  Merged GeoJSON created: {os.path.basename(merged_geojson_path)}")
                            
                            # 2. Create Zones (if configured)
                            has_zones = False
                            if "Create Zones" in st.session_state.annot_features.columns:
                                has_zones = st.session_state.annot_features["Create Zones"].any()
                            if has_zones:
                                log_step("  Generating buffer zones...")
                                zone_ok = generate_zones_for_geojson(
                                    fpath=merged_geojson_path,
                                    features_df=st.session_state.annot_features,
                                    tissue_label=st.session_state.annot_merged_name
                                )
                                if zone_ok:
                                    log_step("  Buffer zones appended to GeoJSON successfully.")
                                else:
                                    log_step("  Warning: Failed to generate buffer zones.")
                                    
                            # 3. Cell Annotation (if matched CSV found)
                            csv_match = next((row["Matched CSV File"] for _, row in edited_pairs.iterrows() if row["Prefix"] == lid), None)
                            if csv_match and csv_match != "No Match Found":
                                csv_path = os.path.join(st.session_state.cell_seg_dir, csv_match)
                                cell_out_path = os.path.join(st.session_state.cell_out_dir, f"{lid}_fully_annotated.csv")
                                log_step(f"  Annotating cells using {csv_match}...")
                                
                                edge_config = None
                                if st.session_state.get('cell_edge_tag', False):
                                    edge_config = {
                                        "reference": st.session_state.get('cell_edge_ref', 'tissue'),
                                        "buffer": float(st.session_state.get('cell_edge_buffer', 100.0)),
                                        "column": st.session_state.get('cell_edge_col', 'edge_cell')
                                    }
                                
                                annot_ok = process_cell_annotation_dynamic(
                                    csv_path=csv_path,
                                    geojson_path=merged_geojson_path,
                                    output_path=cell_out_path,
                                    features_df=st.session_state.annot_features,
                                    x_col_name=st.session_state.cell_x_col,
                                    y_col_name=st.session_state.cell_y_col,
                                    tissue_label=st.session_state.annot_merged_name,
                                    edge_config=edge_config
                                )
                                if annot_ok:
                                    log_step(f"  SUCCESS: Annotated cell CSV saved to {os.path.basename(cell_out_path)}")
                                    success_count += 1
                                else:
                                    log_step(f"  FAILED: Cell annotation failed for {lid}")
                            else:
                                log_step(f"  No matching segmentation CSV found for {lid}. Skipping cell annotation.")
                                success_count += 1
                        else:
                            log_step(f"  FAILED: Could not merge annotations for {lid}")
                            
                        progress.progress((idx + 1) / len(to_proc))
                        
                    log_step("Unified processing completed!")
                    progress.progress(1.0)
                    st.success(f"Successfully processed {success_count} groups.")
                    record_process_event(
                        "tab4",
                        "Annotation complete",
                        status="complete",
                        duration_s=time.time() - run_started_at,
                        details={
                            "groups_selected": len(to_proc),
                            "groups_completed": success_count,
                            "output_directory": st.session_state.annot_output_dir,
                            "cell_output_directory": st.session_state.cell_out_dir,
                        },
                    )

        # === SECTION 4: VISUALIZATION EXPANDER ===
        st.subheader("4. Visualize Annotation Changes")
        with st.expander("Show Before / After Spatial Comparison", expanded=False):
            link_ids = sorted(grouped_files.keys())
            if not link_ids:
                st.info("No matched file groups discovered yet. Scan directories first.")
            else:
                viz_sel = st.selectbox("Select Image Group to Compare", options=link_ids, key="before_after_viz_sel")
                
                # Try to locate background image in ROIs Directory or Image Directory
                img_path = None
                matched_df = st.session_state.get('annot_matched_pairs_df')
                img_match = None
                if matched_df is not None and not matched_df.empty:
                    row = matched_df[matched_df["Prefix"] == viz_sel]
                    if not row.empty:
                        img_match = row.iloc[0]["Matched Image File"]
                        if img_match == "No Match Found":
                            img_match = None
                
                if img_match:
                    for s_dir in [st.session_state.roi_output_dir, st.session_state.annot_image_dir]:
                        if s_dir and os.path.isdir(s_dir):
                            test_path = os.path.join(s_dir, img_match)
                            if os.path.isfile(test_path):
                                img_path = test_path
                                break
                else:
                    for s_dir in [st.session_state.roi_output_dir, st.session_state.annot_image_dir]:
                        if s_dir and os.path.isdir(s_dir):
                            all_imgs = os.listdir(s_dir)
                            match = find_matching_image(viz_sel, [f for f in all_imgs if f.endswith(('.qptiff', '.ome.tiff', '.tiff', '.tif'))])
                            if match:
                                img_path = os.path.join(s_dir, match)
                                break
                
                if img_path and os.path.isfile(img_path):
                    if st.button("Generate Before/After Comparison Plot", key="btn_gen_before_after"):
                        with st.spinner("Generating comparison plots..."):
                            render_before_after_visualization(viz_sel, img_path)
                else:
                    st.warning("No matched background image found in Image Directory.")

    # === RIGHT COLUMN: VISUALIZATION ===
    with right_col:
        st.subheader("Visualization")
        _ensure_heavy_imports()
        
        grouped_files = st.session_state.get('annot_grouped_files', {})
        link_ids = sorted(grouped_files.keys())
        
        if not link_ids:
            st.info("No matched file groups discovered yet. Scan directories first.")
        else:
            viz_lid = st.selectbox(
                "Select File Group to Visualize", 
                options=link_ids, 
                key="annot_viz_group_sel_widget"
            )
            
            if viz_lid:
                st.markdown(f"**Viewing:** {viz_lid}")
                viz_mode = st.radio(
                    "Annotation View", 
                    ["Input (Before)", "Output (After)", "Side-by-Side"], 
                    horizontal=True, 
                    key="annot_viz_mode_widget"
                )
                
                # Locate background image in ROIs Directory or Image Directory
                img_path = None
                matched_df = st.session_state.get('annot_matched_pairs_df')
                img_match = None
                if matched_df is not None and not matched_df.empty:
                    row = matched_df[matched_df["Prefix"] == viz_lid]
                    if not row.empty:
                        img_match = row.iloc[0]["Matched Image File"]
                        if img_match == "No Match Found":
                            img_match = None
                
                if img_match:
                    for s_dir in [st.session_state.roi_output_dir, st.session_state.annot_image_dir]:
                        if s_dir and os.path.isdir(s_dir):
                            test_path = os.path.join(s_dir, img_match)
                            if os.path.isfile(test_path):
                                img_path = test_path
                                break
                else:
                    for s_dir in [st.session_state.roi_output_dir, st.session_state.annot_image_dir]:
                        if s_dir and os.path.isdir(s_dir):
                            all_imgs = os.listdir(s_dir)
                            match = find_matching_image(viz_lid, [f for f in all_imgs if f.endswith(('.qptiff', '.ome.tiff', '.tiff', '.tif'))])
                            if match:
                                img_path = os.path.join(s_dir, match)
                                break
                        
                if img_path and os.path.isfile(img_path):
                    try:
                        ch_names, ch_colors, ch_meta, _ = get_channel_info(img_path)
                        
                        # Find default channel (DAPI or DNA)
                        default_sel = []
                        for ch_idx, ch_name in enumerate(ch_names):
                            if 'dapi' in ch_name.lower() or 'dna' in ch_name.lower():
                                default_sel.append(ch_name)
                        if not default_sel and ch_names:
                            default_sel.append(ch_names[0])
                            
                        selected_channels = st.multiselect(
                            "Select Markers/Channels to Visualize",
                            options=ch_names,
                            default=default_sel,
                            key=f"annot_viz_channels_{viz_lid}_widget"
                        )
                        
                        indices_to_load = [ch_names.index(ch) for ch in selected_channels]
                        composite = None
                        full_shape = (1000, 1000)
                        if indices_to_load:
                            with st.spinner("Loading preview channels..."):
                                img_data_dict, fs, is_rgb = load_preview_channels(str(img_path), indices_to_load)
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
                            composite = np.zeros((512, 512, 3), dtype=np.uint8)
                            
                        h_disp, w_disp = composite.shape[:2]
                        scale_x = w_disp / full_shape[1]
                        scale_y = h_disp / full_shape[0]
                        
                        # Colors mapping
                        rules_df = st.session_state.annot_rules
                        class_colors = {}
                        if not rules_df.empty:
                            for _, r in rules_df.iterrows():
                                class_colors[r["Label"].lower()] = r["Color"]
                                
                        # Get input geojson features
                        input_features = []
                        if viz_lid in grouped_files:
                            for fname, dir_idx in grouped_files[viz_lid]:
                                fpath = os.path.join(st.session_state.annot_input_dirs[dir_idx], fname)
                                if os.path.isfile(fpath):
                                    input_features.extend(load_geojson_features_cached(fpath))
                                    
                        # Get output/merged geojson features
                        out_features = []
                        out_fpath = os.path.join(st.session_state.annot_output_dir, f"{viz_lid}_merged.geojson")
                        out_exists = os.path.isfile(out_fpath)
                        if out_exists:
                            out_features.extend(load_geojson_features_cached(out_fpath))
                            
                        def draw_features_on_image(pil_img, geojson_features, class_colors, scale_x, scale_y):
                            draw = ImageDraw.Draw(pil_img, 'RGBA')
                            for feat in geojson_features:
                                props = feat.get('properties', {})
                                cls = props.get('classification', {})
                                name = cls.get('name') if isinstance(cls, dict) else props.get('name')
                                
                                color_hex = "#00FF00"
                                if name:
                                    color_hex = class_colors.get(name.lower(), color_hex)
                                fill_c = hex_to_rgb(color_hex) + (40,)
                                outline_c = hex_to_rgb(color_hex) + (255,)
                                
                                try:
                                    geom = shape(feat['geometry'])
                                    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
                                    for p in polys:
                                        coords = list(p.exterior.coords)
                                        scaled_coords = [(pt[0]*scale_x, pt[1]*scale_y) for pt in coords]
                                        if len(scaled_coords) > 2:
                                            draw.polygon(scaled_coords, fill=fill_c, outline=outline_c)
                                except Exception:
                                    pass
                                    
                        if viz_mode == "Input (Before)":
                            pil_img = Image.fromarray(composite).convert("RGB")
                            draw_features_on_image(pil_img, input_features, class_colors, scale_x, scale_y)
                            st.markdown("**Preview (Before: Input Annotations)**")
                            st.image(pil_img, width="stretch")
                            
                        elif viz_mode == "Output (After)":
                            if not out_exists:
                                st.info("Output results not found. Click 'Run Unified Annotation' to generate output results.")
                            else:
                                pil_img = Image.fromarray(composite).convert("RGB")
                                draw_features_on_image(pil_img, out_features, class_colors, scale_x, scale_y)
                                st.markdown("**Preview (After: Merged & Processed Results)**")
                                st.image(pil_img, width="stretch")
                                
                        elif viz_mode == "Side-by-Side":
                            c1, c2 = st.columns(2)
                            with c1:
                                pil_img1 = Image.fromarray(composite).convert("RGB")
                                draw_features_on_image(pil_img1, input_features, class_colors, scale_x, scale_y)
                                st.markdown("**Before: Input Annotations**")
                                st.image(pil_img1, width="stretch")
                            with c2:
                                if not out_exists:
                                    st.info("Output results not found. Run unified annotation to see them.")
                                else:
                                    pil_img2 = Image.fromarray(composite).convert("RGB")
                                    draw_features_on_image(pil_img2, out_features, class_colors, scale_x, scale_y)
                                    st.markdown("**After: Merged & Processed Results**")
                                    st.image(pil_img2, width="stretch")
                                    
                    except Exception as e:
                        st.error(f"Error visualizing: {e}")
                else:
                    st.warning("No matched background image found in Image Directory.")


def render_edit_annotation(project_root: Path):
    """Subtab 2: Edit & Update Cell Annotations dynamically from GeoJSON files."""
    st.header("2. Edit Annotation")
    st.markdown("Update or overwrite specific annotation columns (like `tissue`) on cell level in existing CSV files (unified or individual) using updated GeoJSON annotations.")
    
    # Initialize defaults
    keys_list = list(st.session_state.keys())
    if 'edit_csv_mode' not in keys_list:
        st.session_state.edit_csv_mode = "Unified CSV File (All batches combined)"
    if 'edit_unified_csv_path' not in keys_list:
        st.session_state.edit_unified_csv_path = str(project_root / 'outputs/batch_corrected/batch_corrected_data.csv')
    if 'edit_batch_col' not in keys_list:
        st.session_state.edit_batch_col = 'batch_id'
    if 'edit_cell_id_col' not in keys_list:
        st.session_state.edit_cell_id_col = 'cell_id'
    if 'edit_individual_csv_dir' not in keys_list:
        st.session_state.edit_individual_csv_dir = st.session_state.get('cell_out_dir') or str(project_root / 'outputs/annotated_csv')
    if 'edit_geojson_dir' not in keys_list:
        st.session_state.edit_geojson_dir = st.session_state.get('annot_output_dir') or str(project_root / 'outputs/processed_annotations/stage1a_merged_tissue')
    if 'edit_target_col' not in keys_list:
        st.session_state.edit_target_col = 'tissue'
    if 'edit_geojson_name' not in keys_list:
        st.session_state.edit_geojson_name = 'tissue'
    if 'edit_update_type' not in keys_list:
        st.session_state.edit_update_type = "Containment (1/0)"
    if 'edit_x_col' not in keys_list:
        st.session_state.edit_x_col = st.session_state.get('cell_x_col', 'x')
    if 'edit_y_col' not in keys_list:
        st.session_state.edit_y_col = st.session_state.get('cell_y_col', 'y')
    if 'edit_overwrite' not in keys_list:
        st.session_state.edit_overwrite = True
    if 'edit_pairs' not in keys_list:
        st.session_state.edit_pairs = []

    st.session_state.edit_csv_mode = st.radio(
        "CSV File Mode",
        ["Unified CSV File (All batches combined)", "Individual CSV Files (Single CSV per batch)"],
        key="edit_csv_mode_widget"
    )
    
    st.subheader("1. File Setup")
    
    if st.session_state.edit_csv_mode == "Unified CSV File (All batches combined)":
        st.markdown("**Unified CSV File Path**")
        c1, c2 = st.columns([4, 1])
        with c1:
            st.session_state.edit_unified_csv_path = st.text_input("Unified CSV Path", value=st.session_state.edit_unified_csv_path, label_visibility="collapsed")
        with c2:
            if st.button("Select File", key="btn_browse_unified_csv"):
                st.session_state.show_unified_csv_selector = True
                st.rerun()
                
        if st.session_state.get('show_unified_csv_selector'):
            confirmed = st_file_selector("edit_unified_csv_path", label="Choose CSV File", key_prefix="edit_csv", extensions=[".csv"])
            if confirmed:
                st.session_state.show_unified_csv_selector = False
                st.rerun()
                
        c_b1, c_b2 = st.columns(2)
        st.session_state.edit_batch_col = c_b1.text_input("Batch ID Column (in CSV)", value=st.session_state.edit_batch_col)
        st.session_state.edit_cell_id_col = c_b2.text_input("Cell ID Column (in CSV)", value=st.session_state.edit_cell_id_col)
    else:
        directory_input_with_browse(
            label="Individual CSVs Directory",
            text_key="edit_individual_csv_dir",
            default_value=st.session_state.edit_individual_csv_dir,
            browser_key_prefix="edit_ind_csv_dir",
        )
        
    directory_input_with_browse(
        label="GeoJSON Directory (Updated boundaries)",
        text_key="edit_geojson_dir",
        default_value=st.session_state.edit_geojson_dir,
        browser_key_prefix="edit_geo_dir",
    )
    
    if st.button("Scan & Match Files/Batches", type="primary", key="btn_edit_match"):
        with st.spinner("Matching CSV data with GeoJSON files..."):
            geojson_files = sorted([f for f in os.listdir(st.session_state.edit_geojson_dir) if f.endswith('.geojson')])
            
            pairs = []
            if st.session_state.edit_csv_mode == "Unified CSV File (All batches combined)":
                if not os.path.exists(st.session_state.edit_unified_csv_path):
                    st.error("Unified CSV file not found.")
                else:
                    try:
                        df_batches = pd.read_csv(st.session_state.edit_unified_csv_path, usecols=[st.session_state.edit_batch_col])
                        unique_batches = sorted(df_batches[st.session_state.edit_batch_col].dropna().astype(str).unique().tolist())
                        
                        for b in unique_batches:
                            matched_gj = ""
                            for gj in geojson_files:
                                if b.lower() in gj.lower() or _normalize_filename_for_matching(b) in _normalize_filename_for_matching(gj):
                                    matched_gj = os.path.join(st.session_state.edit_geojson_dir, gj)
                                    break
                            pairs.append({
                                "Batch ID": b,
                                "GeoJSON File": matched_gj or "No Match Found"
                            })
                    except Exception as e:
                        st.error(f"Error reading CSV batches: {e}")
            else:
                csv_files = sorted([f for f in os.listdir(st.session_state.edit_individual_csv_dir) if f.endswith('.csv')])
                matched_pairs, _, _ = discover_best_file_pairs(csv_files, geojson_files)
                for pair in matched_pairs:
                    pairs.append({
                        "CSV File": os.path.join(st.session_state.edit_individual_csv_dir, pair["CSV File"]),
                        "GeoJSON File": os.path.join(st.session_state.edit_geojson_dir, pair["GeoJSON File"])
                    })
            st.session_state.edit_pairs = pairs
            if pairs:
                st.success(f"Matched {len(pairs)} batches/files successfully.")
            else:
                st.warning("No matched batches or files found.")

    if st.session_state.edit_pairs:
        st.subheader("2. Matched Batches/Files")
        pairs_df = pd.DataFrame(st.session_state.edit_pairs)
        
        if st.session_state.edit_csv_mode == "Unified CSV File (All batches combined)":
            edited_pairs = st.data_editor(
                pairs_df,
                column_config={
                    "Batch ID": st.column_config.TextColumn("Batch/Slide ID", disabled=True),
                    "GeoJSON File": st.column_config.TextColumn("Matched GeoJSON File", disabled=True)
                },
                hide_index=True,
                key="edit_unified_pairs_editor",
                width='stretch'
            )
        else:
            edited_pairs = st.data_editor(
                pairs_df,
                column_config={
                    "CSV File": st.column_config.TextColumn("CSV File Path", disabled=True),
                    "GeoJSON File": st.column_config.TextColumn("Matched GeoJSON File", disabled=True)
                },
                hide_index=True,
                key="edit_individual_pairs_editor",
                width='stretch'
            )
            
        st.subheader("3. Update Configuration")
        c_p1, c_p2 = st.columns(2)
        st.session_state.edit_target_col = c_p1.text_input("CSV Target Column", value=st.session_state.edit_target_col)
        st.session_state.edit_geojson_name = c_p2.text_input("GeoJSON Classification Name", value=st.session_state.edit_geojson_name)
        
        st.session_state.edit_update_type = c_p1.selectbox(
            "Update Calculation Type",
            ["Containment (1/0)", "Signed Distance", "Distance"],
            index=["Containment (1/0)", "Signed Distance", "Distance"].index(st.session_state.edit_update_type)
        )
        st.session_state.edit_overwrite = st.checkbox("Overwrite original CSV file(s)", value=st.session_state.edit_overwrite)
        
        with st.expander("Coordinates Column Names"):
            c_x1, c_x2 = st.columns(2)
            st.session_state.edit_x_col = c_x1.text_input("X coordinate column", value=st.session_state.edit_x_col)
            st.session_state.edit_y_col = c_x2.text_input("Y coordinate column", value=st.session_state.edit_y_col)
            
        st.markdown("---")
        
        run_edit_btn = st.button("Update Cell Annotations", type="primary", width='stretch', key="btn_run_edit_ann")
                
        if run_edit_btn:
            valid_pairs = [p for p in st.session_state.edit_pairs if p["GeoJSON File"] != "No Match Found"]
            if not valid_pairs:
                st.error("No valid file pairs matched. Ensure GeoJSON files are present and match batch names.")
            else:
                with st.spinner("Updating cell annotations..."):
                    ok = perform_edit_annotation_update(
                        csv_mode=st.session_state.edit_csv_mode,
                        edit_pairs=valid_pairs,
                        target_col=st.session_state.edit_target_col,
                        geojson_name=st.session_state.edit_geojson_name,
                        update_type=st.session_state.edit_update_type,
                        x_col=st.session_state.edit_x_col,
                        y_col=st.session_state.edit_y_col,
                        batch_col=st.session_state.edit_batch_col if st.session_state.edit_csv_mode == "Unified CSV File (All batches combined)" else None,
                        cell_id_col=st.session_state.edit_cell_id_col if st.session_state.edit_csv_mode == "Unified CSV File (All batches combined)" else None,
                        unified_csv_path=st.session_state.edit_unified_csv_path if st.session_state.edit_csv_mode == "Unified CSV File (All batches combined)" else None,
                        overwrite=st.session_state.edit_overwrite
                    )
                    if ok:
                        st.success("Cell annotations updated successfully!")


def render_visualization(project_root: Path):
    """Subtab 3: Visualization & Analysis."""
    st.header("3. Visualization & Analysis")
    st.markdown("Visualize spatial distribution of cells and analyze marker expression.")
    
    # Initialize state
    if not st.session_state.get('vis_csv_dir'):
        st.session_state.vis_csv_dir = str(project_root / 'outputs/annotated_csv')
    if not st.session_state.get('vis_img_dir'):
        st.session_state.vis_img_dir = str(project_root / 'outputs/rois')
    if 'vis_filters' not in list(st.session_state.keys()):
        st.session_state.vis_filters = []
        
    st.subheader("1. Data Selection")
    c1, c2 = st.columns([3, 1])
    
    default_val = st.session_state.vis_csv_dir
    val_in_state = st.session_state.pop("vis_csv_dir_in", default_val)
    csv_dir = c1.text_input("Annotated CSV Directory", value=val_in_state, key="vis_csv_dir_in")
    if csv_dir != st.session_state.vis_csv_dir:
        st.session_state.vis_csv_dir = csv_dir
        
    if c2.button("Browse", key="vis_browse_csv"):
        st.session_state.show_vis_csv_browser = True
        
    if st.session_state.get('show_vis_csv_browser'):
        path = render_file_browser(st.session_state.vis_csv_dir, "vis_csv_browser", 'directory')
        if path == "CANCEL":
            st.session_state.show_vis_csv_browser = False
            st.rerun()
        elif path:
            st.session_state.vis_csv_dir = path
            st.session_state["vis_csv_dir_in"] = path
            st.session_state.show_vis_csv_browser = False
            st.rerun()
            
    csv_files = []
    if os.path.isdir(st.session_state.vis_csv_dir):
        csv_files = sorted([f for f in os.listdir(st.session_state.vis_csv_dir) if f.endswith('.csv')])
        
    selected_csv = st.selectbox("Select CSV File", csv_files, key="vis_csv_select")
    if not selected_csv:
        st.info("Please select a CSV file to proceed.")
        return
        
    def load_annotated_csv_cached(path):
        """Cache-safe wrapper: re-reads when the file on disk changes."""
        from spacemap_gui.ui_utils import file_signature
        return _load_annotated_csv_cached__impl(path, file_signature(path))


    @st.cache_data
    def _load_annotated_csv_cached__impl(path, signature):
        return pd.read_csv(path)
        
    try:
        df = load_annotated_csv_cached(os.path.join(st.session_state.vis_csv_dir, selected_csv))
        st.caption(f"Loaded {len(df)} cells with {len(df.columns)} columns.")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    st.markdown("### Image Selection")
    c3, c4 = st.columns([3, 1])
    
    default_val = st.session_state.vis_img_dir
    val_in_state = st.session_state.pop("vis_img_dir_in", default_val)
    img_dir = c3.text_input("Image Directory", value=val_in_state, key="vis_img_dir_in")
    if img_dir != st.session_state.vis_img_dir:
        st.session_state.vis_img_dir = img_dir
        
    if c4.button("Browse", key="vis_browse_img"):
        st.session_state.show_vis_img_browser = True
        
    if st.session_state.get('show_vis_img_browser'):
        path = render_file_browser(st.session_state.vis_img_dir, "vis_img_browser", 'directory')
        if path == "CANCEL":
            st.session_state.show_vis_img_browser = False
            st.rerun()
        elif path:
            st.session_state.vis_img_dir = path
            st.session_state["vis_img_dir_in"] = path
            st.session_state.show_vis_img_browser = False
            st.rerun()
            
    img_files = []
    if os.path.isdir(st.session_state.vis_img_dir):
        img_files = sorted([f for f in os.listdir(st.session_state.vis_img_dir) if f.endswith(('.qptiff', '.ome.tiff', '.tiff'))])
        
    default_img_idx = 0
    stem = Path(selected_csv).stem.replace('_fully_annotated', '').replace('_cell_features', '')
    for i, f in enumerate(img_files):
        if stem in f:
            default_img_idx = i
            break
            
    selected_img = st.selectbox("Select Background Image", img_files, index=default_img_idx, key="vis_img_select")
    
    st.subheader("2. Cell Filtering")
    # Filters management
    col_f1, col_f2 = st.columns([2, 1])
    filt_col = col_f1.selectbox("Filter Column", sorted(df.columns), key="vis_filt_col")
    filt_op = col_f2.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="vis_filt_op")
    
    if pd.api.types.is_numeric_dtype(df[filt_col]):
        filt_val = st.number_input("Value", value=0.0, key="vis_filt_val_num")
    else:
        unique_vals = sorted(df[filt_col].astype(str).unique())
        if len(unique_vals) < 50:
            filt_val = st.selectbox("Value", unique_vals, key="vis_filt_val_cat")
        else:
            filt_val = st.text_input("Value", key="vis_filt_val_text")
            
    if st.button("Add Filter", key="btn_add_filter"):
        st.session_state.vis_filters.append({"col": filt_col, "op": filt_op, "val": filt_val})
        
    # Display active filters
    if st.session_state.vis_filters:
        st.markdown("**Active Filters:**")
        for idx, f in enumerate(st.session_state.vis_filters):
            c_d1, c_d2 = st.columns([5, 1])
            c_d1.text(f"- {f['col']} {f['op']} {f['val']}")
            if c_d2.button("Remove", key=f"del_filt_{idx}"):
                st.session_state.vis_filters.pop(idx)
                st.rerun()
                
    # Apply filters
    filtered_df = df.copy()
    for f in st.session_state.vis_filters:
        col, op, val = f['col'], f['op'], f['val']
        if op == "==": filtered_df = filtered_df[filtered_df[col] == val]
        elif op == "!=": filtered_df = filtered_df[filtered_df[col] != val]
        elif op == ">": filtered_df = filtered_df[filtered_df[col] > val]
        elif op == "<": filtered_df = filtered_df[filtered_df[col] < val]
        elif op == ">=": filtered_df = filtered_df[filtered_df[col] >= val]
        elif op == "<=": filtered_df = filtered_df[filtered_df[col] <= val]
        elif op == "contains": filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val))]
        
    st.caption(f"Cells remaining: {len(filtered_df)} / {len(df)}")
    
    # Coordinates mapping
    x_col = 'x'
    y_col = 'y'
    for c in ['x_centroid', 'centroid_x', 'centroid_y', 'y_centroid']:
        if c in df.columns:
            if 'x' in c: x_col = c
            if 'y' in c: y_col = c

    # Plot Settings
    st.subheader("3. Spatial Map")
    col_viz1, col_viz2 = st.columns([1, 2])
    
    with col_viz1:
        color_col = st.selectbox("Color By Column", ["None"] + sorted(df.columns), key="vis_color_col")
        pt_size = st.slider("Point Size", 1, 20, 5)
        alpha = st.slider("Alpha", 0.1, 1.0, 0.8)
        
        st.markdown("**Image Channels**")
        selected_channels = []
        if selected_img:
            img_path = os.path.join(st.session_state.vis_img_dir, selected_img)
            try:
                ch_names, ch_colors, _, _ = get_channel_info(img_path)
                for i, name in enumerate(ch_names):
                    if st.checkbox(name, value=(i==0), key=f"vis_ch_{i}"):
                        selected_channels.append(i)
            except Exception as e:
                st.error(f"Error reading image: {e}")
                
    with col_viz2:
        run_map_btn = st.button("Generate Spatial Map", type="primary", width='stretch', key="btn_run_vis_map")
                
        if run_map_btn:
            with st.spinner("Generating map..."):
                bg_img = None
                if selected_img and selected_channels:
                    img_data, full_shape, is_rgb = load_preview_channels(img_path, selected_channels)
                    if is_rgb:
                        bg_img = img_data['rgb']
                    else:
                        h, w = img_data[selected_channels[0]].shape
                        composite = np.zeros((h, w, 3), dtype=np.float32)
                        for idx in selected_channels:
                            img = img_data[idx].astype(np.float32)
                            p99 = np.percentile(img, 99)
                            if p99 > 0: img = np.clip(img / p99, 0, 1)
                            col = ch_colors[idx]
                            if not isinstance(col, tuple): col = hex_to_rgb(col)
                            for c in range(3):
                                composite[:, :, c] += img * col[c]
                        bg_img = np.clip(composite, 0, 255).astype(np.uint8)
                        
                fig, ax = plt.subplots(figsize=(10, 10))
                scale_x, scale_y = 1.0, 1.0
                if bg_img is not None:
                    ax.imshow(bg_img)
                    h_disp, w_disp = bg_img.shape[:2]
                    scale_x = w_disp / full_shape[1]
                    scale_y = h_disp / full_shape[0]
                else:
                    ax.invert_yaxis()
                    
                x = filtered_df[x_col] * scale_x
                y = filtered_df[y_col] * scale_y
                
                if color_col != "None":
                    if pd.api.types.is_numeric_dtype(filtered_df[color_col]) and len(filtered_df[color_col].unique()) > 20:
                        sc = ax.scatter(x, y, c=filtered_df[color_col], s=pt_size, alpha=alpha, cmap='viridis')
                        plt.colorbar(sc, ax=ax, label=color_col)
                    else:
                        cats = filtered_df[color_col].unique()
                        palette = sns.color_palette("husl", len(cats))
                        for i, cat in enumerate(cats):
                            mask = filtered_df[color_col] == cat
                            ax.scatter(x[mask], y[mask], color=palette[i], s=pt_size, alpha=alpha, label=str(cat))
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.scatter(x, y, c='red', s=pt_size, alpha=alpha)
                    
                ax.set_title(f"Spatial Distribution: {len(filtered_df)} cells")
                ax.axis('off')
                st.pyplot(fig)

    st.subheader("4. Marker Analysis")
    if 'vis_hist_groups' not in list(st.session_state.keys()):
        st.session_state.vis_hist_groups = []
        
    c_h1, c_h2 = st.columns([1, 3])
    with c_h1:
        exclude = ['x', 'y', 'id', 'label', 'area', 'eccentricity', 'solidity', 'extent', 'orientation']
        marker_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not any(x in c.lower() for x in exclude)]
        selected_markers = st.multiselect("Select Markers (Max 10)", marker_candidates, max_selections=10)
        norm_method = st.radio("Normalization", ["None", "Z-Score", "Min-Max", "ArcSinh", "Log10"], index=1)
        
        cofactor = 5.0
        if norm_method == "ArcSinh":
            cofactor = st.number_input("Cofactor", value=5.0, min_value=0.1)
            
        bins = st.slider("Bins", 10, 100, 50)
        hist_alpha = st.slider("Transparency", 0.1, 1.0, 0.5)
        
        c_disp1, c_disp2 = st.columns(2)
        show_hist = c_disp1.checkbox("Show Bars", value=True)
        show_kde = c_disp2.checkbox("Show Curve", value=True)
        log_scale = st.checkbox("Log Scale X", value=False)
        
        st.markdown("---")
        st.markdown("**Comparison Groups**")
        new_grp_name = st.text_input("New Group Name", f"Group {chr(66 + len(st.session_state.vis_hist_groups))}")
        if st.button("Add Group"):
            st.session_state.vis_hist_groups.append({'name': new_grp_name, 'filters': []})
            st.rerun()
            
        groups_to_remove = []
        for i, grp in enumerate(st.session_state.vis_hist_groups):
            with st.expander(f"Configure {grp['name']}", expanded=False):
                if st.button(f"Remove {grp['name']}", key=f"rm_grp_{i}"):
                    groups_to_remove.append(i)
                    continue
                c_fb1, c_fb2 = st.columns([2, 1])
                fb_col = c_fb1.selectbox("Col", sorted(df.columns), key=f"vis_filter_{i}_col")
                fb_op = c_fb2.selectbox("Op", ["==", "!=", ">", "<", ">=", "<=", "contains"], key=f"vis_filter_{i}_op")
                
                if pd.api.types.is_numeric_dtype(df[fb_col]):
                    fb_val = st.number_input("Val", value=0.0, key=f"vis_filter_{i}_val_num")
                else:
                    unique_vals = sorted(df[fb_col].astype(str).unique())
                    if len(unique_vals) < 50:
                        fb_val = st.selectbox("Val", unique_vals, key=f"vis_filter_{i}_val_cat")
                    else:
                        fb_val = st.text_input("Val", key=f"vis_filter_{i}_val_text")
                        
                if st.button("Add Filter", key=f"add_filter_{i}"):
                    grp['filters'].append({"col": fb_col, "op": fb_op, "val": fb_val})
                    
                if grp['filters']:
                    for j, f in enumerate(grp['filters']):
                        c_d1, c_d2 = st.columns([4, 1])
                        c_d1.text(f"{f['col']} {f['op']} {f['val']}")
                        if c_d2.button("X", key=f"del_filter_{i}_{j}"):
                            grp['filters'].pop(j)
                            st.rerun()
                            
        if groups_to_remove:
            for i in sorted(groups_to_remove, reverse=True):
                st.session_state.vis_hist_groups.pop(i)
            st.rerun()
            
    with c_h2:
        if selected_markers:
            all_groups = [{'name': 'Group A (Main)', 'filters': st.session_state.vis_filters}]
            all_groups.extend(st.session_state.vis_hist_groups)
            
            num_plots = len(all_groups)
            cols = 4
            rows = (num_plots + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True, squeeze=False)
            axes = axes.flatten()
            
            for idx, grp in enumerate(all_groups):
                ax = axes[idx]
                grp_df = df.copy()
                for f in grp['filters']:
                    col, op, val = f['col'], f['op'], f['val']
                    if op == "==": grp_df = grp_df[grp_df[col] == val]
                    elif op == "!=": grp_df = grp_df[grp_df[col] != val]
                    elif op == ">": grp_df = grp_df[grp_df[col] > val]
                    elif op == "<": grp_df = grp_df[grp_df[col] < val]
                    elif op == ">=": grp_df = grp_df[grp_df[col] >= val]
                    elif op == "<=": grp_df = grp_df[grp_df[col] <= val]
                    elif op == "contains": grp_df = grp_df[grp_df[col].astype(str).str.contains(str(val))]
                    
                data_map, xlabel = get_normalized_data(grp_df, selected_markers, norm_method, cofactor)
                for m, d in data_map.items():
                    if show_hist:
                        sns.histplot(d, bins=bins, kde=False, label=m, alpha=hist_alpha, element="step", ax=ax, stat="density")
                    if show_kde:
                        sns.kdeplot(d, label=m if not show_hist else None, ax=ax)
                ax.set_title(f"{grp['name']}\nN={len(grp_df)}")
                ax.set_xlabel(xlabel)
                if idx == 0:
                    ax.legend()
                if log_scale: ax.set_xscale('log')
                
            for i in range(num_plots, len(axes)):
                axes[i].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Select markers to plot histograms.")


def render():
    """Render the cell-level annotation workflow inside the Annotation tab."""
    import streamlit as st
    from pathlib import Path
    from spacemap_gui.ui_utils import require_active_project
    project_root = require_active_project()
    if project_root is None:
        return

    _ensure_heavy_imports()
    st.markdown("### Cell Annotation Pipeline")

    t1, t2, t3 = st.tabs(["1. Map & Annotate Cells", "2. Edit Annotation", "3. Visualization"])
    
    with t1:
        render_unified_annotation(project_root)
    with t2:
        render_edit_annotation(project_root)
    with t3:
        render_visualization(project_root)

if __name__ == "__main__":
    from spacemap_gui.ui_utils import require_active_project
    require_active_project()
    render()
