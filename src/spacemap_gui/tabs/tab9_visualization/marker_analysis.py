"""Marker analysis tools for image-based marker intensity QC and export."""

from __future__ import annotations

import copy
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import MultiPolygon, Polygon, shape

Image.MAX_IMAGE_PIXELS = None

from spacemap_gui.ui_utils import directory_input_with_browse
from spacemap.background_subtraction_utils import (
    get_channel_info,
    is_supported_image_file,
    load_preview_channels_compatible,
    normalize_image_stem,
    read_channel_image,
)

from ..tab3_roi_extraction import HAS_CANVAS, image_to_base64, st_canvas

WHOLE_IMAGE_LABEL = "Whole image"
TRANSFORM_OPTIONS = ("Linear", "Log1p", "Log10", "Arcsinh")
DEFAULT_PREVIEW_MAX_DIM = 1400
DEFAULT_HISTOGRAM_BINS = 128
DEFAULT_HIST_SAMPLE_CAP = 250_000
DEFAULT_RAW_EXPORT_CAP = 250_000
DEFAULT_BATCH_COMPILED_SAMPLE_CAP = 150_000


@dataclass
class RegionSpec:
    label: str
    geometry: Polygon | MultiPolygon | None
    source: str


@dataclass(frozen=True)
class HistogramFigureOptions:
    layout_mode: str
    show_bars: bool
    show_curve: bool
    density: bool
    legend_outside: bool
    width_inches: float
    height_inches: float
    dpi: int
    save_separate_markers: bool


@dataclass
class BatchCompiledChunk:
    image_name: str
    region_label: str
    marker_name: str
    pixel_count: int
    value_sum: float
    value_sum_sq: float
    min_value: float
    max_value: float
    sample_values: np.ndarray
    sample_weight: float
    fluorophore: str
    exposure: str
    excitation_wavelength: str
    emission_wavelength: str


@dataclass
class BatchImageJobResult:
    image_name: str
    summary_df: pd.DataFrame
    histogram_df: pd.DataFrame
    raw_paths: list[Path]
    compiled_chunks: list[BatchCompiledChunk]


def _read_channel_info(file_path: str):
    return get_channel_info(file_path)


def _get_image_shape(file_path: str) -> tuple[int, int]:
    loaded, full_shape, _is_rgb = load_preview_channels_compatible(file_path, [0], max_dim=64)
    if full_shape and len(full_shape) >= 2 and int(full_shape[-2]) > 0 and int(full_shape[-1]) > 0:
        return int(full_shape[-2]), int(full_shape[-1])
    if loaded:
        first = np.asarray(next(iter(loaded.values())))
        if first.ndim >= 2:
            return int(first.shape[-2]), int(first.shape[-1])
    return 0, 0


@st.cache_data(show_spinner=False)
def _get_cached_channel_info(file_path: str, mtime: float):
    return _read_channel_info(file_path)


@st.cache_data(show_spinner=False)
def _load_geojson_cached(path: str, mtime: float) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def _get_image_shape_cached(file_path: str, mtime: float) -> tuple[int, int]:
    return _get_image_shape(file_path)


def _list_image_files(directory: str) -> list[str]:
    try:
        path = Path(str(directory))
        if not path.exists():
            return []
        return sorted([item.name for item in path.iterdir() if item.is_file() and is_supported_image_file(item.name)])
    except Exception:
        return []


def _list_geojson_files(directory: str) -> list[str]:
    try:
        path = Path(str(directory))
        if not path.exists():
            return []
        return sorted([item.name for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".geojson"])
    except Exception:
        return []


def _norm_image_prefix(name: str) -> str:
    stem = normalize_image_stem(name)
    if stem.endswith("_merged"):
        stem = stem[:-7]
    return stem.strip().lower()


def _norm_geojson_prefix(name: str) -> str:
    stem = normalize_image_stem(name)
    suffix = "_roi_bounding_boxes_coords"
    if stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]
    if stem.endswith("_merged"):
        stem = stem[:-7]
    return stem.strip().lower()


def _matched_geojson_name(image_name: str, geojson_files: list[str]) -> str | None:
    image_prefix = _norm_image_prefix(image_name)
    for geojson_name in geojson_files:
        if _norm_geojson_prefix(geojson_name) == image_prefix:
            return geojson_name
    return geojson_files[0] if geojson_files else None


def _color_to_rgb(color_value: Any) -> tuple[int, int, int]:
    if isinstance(color_value, str):
        value = color_value.strip().lstrip("#")
        if len(value) == 6:
            try:
                return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
            except Exception:
                return (255, 255, 255)
        return (255, 255, 255)
    if isinstance(color_value, (tuple, list)) and len(color_value) >= 3:
        try:
            red = max(0, min(255, int(color_value[0])))
            green = max(0, min(255, int(color_value[1])))
            blue = max(0, min(255, int(color_value[2])))
            return red, green, blue
        except Exception:
            return (255, 255, 255)
    return (255, 255, 255)


def _color_to_hex(color_value: Any) -> str:
    red, green, blue = _color_to_rgb(color_value)
    return f"#{red:02X}{green:02X}{blue:02X}"


def _normalize_to_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if array.dtype == np.uint8:
        return array
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros(array.shape, dtype=np.uint8)
    values = array[finite_mask].astype(np.float32, copy=False)
    low = float(np.percentile(values, 1))
    high = float(np.percentile(values, 99))
    if high <= low:
        high = low + 1.0
    normalized = np.zeros(array.shape, dtype=np.float32)
    normalized[finite_mask] = np.clip((array[finite_mask] - low) / (high - low), 0.0, 1.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _transform_values(values: np.ndarray, method: str, cofactor: float) -> np.ndarray:
    clean = np.asarray(values, dtype=np.float64)
    if clean.size == 0:
        return clean
    if method == "Log1p":
        return np.log1p(np.clip(clean, 0, None))
    if method == "Log10":
        return np.log10(np.clip(clean, 1e-9, None))
    if method == "Arcsinh":
        divisor = max(float(cofactor), 1e-9)
        return np.arcsinh(clean / divisor)
    return clean


def _transform_axis_label(method: str, cofactor: float) -> str:
    if method == "Log1p":
        return "log1p(Intensity)"
    if method == "Log10":
        return "log10(Intensity)"
    if method == "Arcsinh":
        return f"arcsinh(Intensity / {cofactor:g})"
    return "Intensity"


def _choose_indices(total_size: int, limit: int) -> np.ndarray:
    if total_size <= 0:
        return np.array([], dtype=np.int64)
    if limit <= 0 or total_size <= limit:
        return np.arange(total_size, dtype=np.int64)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(total_size, size=int(limit), replace=False).astype(np.int64))


def _sample_array(values: np.ndarray, limit: int) -> np.ndarray:
    if values.size <= 0:
        return values
    if limit <= 0 or values.size <= limit:
        return values
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(values.size, size=int(limit), replace=False))
    return values[indices]


def _safe_slug(text: str) -> str:
    chars = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "region"


def _build_preview_pil(
    file_path: Path,
    channel_names: list[str],
    channel_colors: list[Any],
    preview_mode: str,
    preview_markers: list[str],
    max_width: int = 720,
) -> tuple[Image.Image, tuple[int, int]]:
    index_lookup = {name: idx for idx, name in enumerate(channel_names)}
    chosen_markers = [name for name in preview_markers if name in index_lookup]
    if not chosen_markers and channel_names:
        chosen_markers = channel_names[: min(3, len(channel_names))]

    indices = [index_lookup[name] for name in chosen_markers if name in index_lookup]
    if not indices and channel_names:
        indices = [0]

    loaded, full_shape, _is_rgb = load_preview_channels_compatible(str(file_path), indices, max_dim=DEFAULT_PREVIEW_MAX_DIM)
    full_h = int(full_shape[-2]) if full_shape else 0
    full_w = int(full_shape[-1]) if full_shape else 0

    if not loaded:
        return Image.new("RGB", (max_width, max_width), (0, 0, 0)), (full_h, full_w)

    if preview_mode == "Single Channel" or len(indices) == 1:
        plane = _normalize_to_uint8(loaded[indices[0]])
        preview = Image.fromarray(plane).convert("RGB")
    else:
        reference = np.asarray(loaded[indices[0]])
        height, width = int(reference.shape[-2]), int(reference.shape[-1])
        composite = np.zeros((height, width, 3), dtype=np.float32)
        for marker_name in chosen_markers[:3]:
            marker_index = index_lookup.get(marker_name)
            if marker_index is None or marker_index not in loaded:
                continue
            plane = _normalize_to_uint8(loaded[marker_index]).astype(np.float32) / 255.0
            color = np.asarray(_color_to_rgb(channel_colors[marker_index]), dtype=np.float32) / 255.0
            composite += plane[..., None] * color
        preview = Image.fromarray((np.clip(composite, 0.0, 1.0) * 255.0).astype(np.uint8))

    if preview.mode != "RGB":
        preview = preview.convert("RGB")
    if preview.width > max_width:
        ratio = max_width / float(preview.width)
        preview = preview.resize((max_width, max(1, int(preview.height * ratio))))
    return preview, (full_h, full_w)


def _unique_region_label(label: str, seen: dict[str, int]) -> str:
    count = seen.get(label, 0) + 1
    seen[label] = count
    if count == 1:
        return label
    return f"{label} ({count})"


def _load_regions_from_geojson(path: Path) -> list[RegionSpec]:
    if not path.exists():
        return []
    try:
        geojson = _load_geojson_cached(str(path), path.stat().st_mtime)
    except Exception:
        return []

    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", []) or []
    elif geojson.get("type") == "Feature":
        features = [geojson]
    else:
        features = []

    regions: list[RegionSpec] = []
    label_counts: dict[str, int] = {}
    for feature in features:
        geometry_data = feature.get("geometry") or {}
        if geometry_data.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            geometry = shape(geometry_data)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        properties = feature.get("properties") or {}
        label = str(properties.get("label") or f"ROI {len(regions) + 1}")
        label = _unique_region_label(label, label_counts)
        regions.append(RegionSpec(label=label, geometry=cast(Polygon | MultiPolygon, geometry), source="GeoJSON"))
    return regions


def _iter_polygons(geometry: Polygon | MultiPolygon | None) -> list[Polygon]:
    if geometry is None:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return []


def _clip_bounds(bounds: tuple[float, float, float, float], full_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    x_min, y_min, x_max, y_max = bounds
    full_h, full_w = full_shape
    x1 = max(0, min(full_w, int(np.floor(x_min))))
    y1 = max(0, min(full_h, int(np.floor(y_min))))
    x2 = max(0, min(full_w, int(np.ceil(x_max))))
    y2 = max(0, min(full_h, int(np.ceil(y_max))))
    if x2 <= x1 or y2 <= y1:
        return None
    return y1, y2, x1, x2


def _geometry_mask(geometry: Polygon | MultiPolygon, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y1, y2, x1, x2 = bbox
    width = max(1, int(x2 - x1))
    height = max(1, int(y2 - y1))
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in _iter_polygons(geometry):
        exterior = [(float(x) - x1, float(y) - y1) for x, y in polygon.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=1, outline=1)
        for interior in polygon.interiors:
            hole = [(float(x) - x1, float(y) - y1) for x, y in interior.coords]
            if len(hole) >= 3:
                draw.polygon(hole, fill=0, outline=0)
    return np.asarray(mask_image, dtype=bool)


def _overlay_regions_on_preview(
    preview_image: Image.Image,
    regions: list[RegionSpec],
    full_shape: tuple[int, int],
) -> Image.Image:
    if not regions:
        return preview_image
    full_h, full_w = full_shape
    if full_h <= 0 or full_w <= 0:
        return preview_image
    scale_x = preview_image.width / float(full_w)
    scale_y = preview_image.height / float(full_h)
    overlay = preview_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for region in regions:
        if region.geometry is None:
            continue
        for polygon in _iter_polygons(region.geometry):
            exterior = [(float(x) * scale_x, float(y) * scale_y) for x, y in polygon.exterior.coords]
            if len(exterior) >= 3:
                draw.polygon(exterior, fill=(255, 165, 0, 60), outline=(0, 0, 0, 220))
            for interior in polygon.interiors:
                hole = [(float(x) * scale_x, float(y) * scale_y) for x, y in interior.coords]
                if len(hole) >= 3:
                    draw.polygon(hole, fill=(0, 0, 0, 0), outline=(0, 0, 0, 80))
        try:
            centroid = region.geometry.centroid
            draw.text((float(centroid.x) * scale_x + 4, float(centroid.y) * scale_y + 4), region.label, fill=(255, 255, 255, 255))
        except Exception:
            continue
    return overlay.convert("RGB")


def _canvas_points_from_object(obj: dict[str, Any], scale_x: float, scale_y: float) -> list[tuple[float, float]]:
    if obj.get("type") == "path":
        points = []
        for command in obj.get("path", []) or []:
            if isinstance(command, list) and len(command) >= 3 and command[0] in {"M", "L"}:
                points.append((float(command[1]) * scale_x, float(command[2]) * scale_y))
        return points
    if obj.get("type") == "rect":
        left = float(obj.get("left", 0.0)) * scale_x
        top = float(obj.get("top", 0.0)) * scale_y
        width = float(obj.get("width", 0.0)) * scale_x
        height = float(obj.get("height", 0.0)) * scale_y
        return [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ]
    return []


def _is_valid_canvas_object(obj: dict[str, Any]) -> bool:
    if obj.get("type") == "rect":
        return float(obj.get("width", 0.0)) > 0 and float(obj.get("height", 0.0)) > 0
    if obj.get("type") != "path":
        return False
    points = _canvas_points_from_object(obj, 1.0, 1.0)
    if len(points) < 3:
        return False
    try:
        polygon = Polygon(points)
        return polygon.area > 0.0
    except Exception:
        return False


def _canvas_object_signature(obj: dict[str, Any]) -> str:
    try:
        return json.dumps(obj.get("path", obj), separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return str(obj)


def _regions_from_canvas_objects(
    objects: list[dict[str, Any]],
    labels: list[str],
    scale_x: float,
    scale_y: float,
) -> list[RegionSpec]:
    regions: list[RegionSpec] = []
    label_counts: dict[str, int] = {}
    for obj, label in zip(objects, labels):
        points = _canvas_points_from_object(obj, scale_x, scale_y)
        if len(points) < 3:
            continue
        try:
            polygon = Polygon(points)
        except Exception:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0.0:
            continue
        region_label = _unique_region_label(str(label or f"ROI {len(regions) + 1}"), label_counts)
        regions.append(RegionSpec(label=region_label, geometry=polygon, source="Drawn"))
    return regions


def _build_marker_table(
    prefix: str,
    context_key: str,
    channel_names: list[str],
    channel_colors: list[Any],
    channel_metadata: list[dict[str, Any]],
) -> tuple[list[str], pd.DataFrame]:
    selected_key = f"{prefix}_selected_markers"
    existing_selection = [name for name in st.session_state.get(selected_key, channel_names) if name in channel_names]
    if not existing_selection:
        existing_selection = list(channel_names)

    rows = []
    for idx, name in enumerate(channel_names):
        meta = channel_metadata[idx] if idx < len(channel_metadata) else {}
        rows.append(
            {
                "Analyze": name in existing_selection,
                "Marker": name,
                "Color": _color_to_hex(channel_colors[idx] if idx < len(channel_colors) else (255, 255, 255)),
                "Fluorophore": str(meta.get("dye", "-")),
                "Exposure": str(meta.get("exposure", "-")),
                "Excitation (nm)": str(meta.get("excitation_wavelength", "-")),
                "Emission (nm)": str(meta.get("emission_wavelength", "-")),
            }
        )

    edited = st.data_editor(
        pd.DataFrame(rows),
        key=f"{prefix}_marker_editor_{context_key}",
        hide_index=True,
        disabled=["Marker", "Color", "Fluorophore", "Exposure", "Excitation (nm)", "Emission (nm)"],
    )
    selected_markers = edited.loc[edited["Analyze"], "Marker"].tolist() if not edited.empty else []
    st.session_state[selected_key] = selected_markers
    return selected_markers, edited


def _build_region_preview_regions(
    mode: str,
    full_shape: tuple[int, int],
    geojson_path: Path | None,
    drawn_regions: list[RegionSpec],
) -> list[RegionSpec]:
    if mode == "GeoJSON" and geojson_path is not None:
        return _load_regions_from_geojson(geojson_path)
    if mode == "Draw polygons":
        return drawn_regions
    return [RegionSpec(label=WHOLE_IMAGE_LABEL, geometry=None, source="Whole image")]


def _selected_regions_from_labels(regions: list[RegionSpec], selected_labels: list[str]) -> list[RegionSpec]:
    if not regions:
        return [RegionSpec(label=WHOLE_IMAGE_LABEL, geometry=None, source="Whole image")]
    if not selected_labels:
        return regions
    selected_set = set(selected_labels)
    chosen = [region for region in regions if region.label in selected_set]
    return chosen or regions


def _extract_combined_values(
    image_path: Path,
    channel_index: int,
    full_shape: tuple[int, int],
    regions: list[RegionSpec],
    use_thumbnail_for_whole: bool,
) -> np.ndarray:
    if not regions or (len(regions) == 1 and regions[0].geometry is None):
        if use_thumbnail_for_whole:
            loaded, _shape, _is_rgb = load_preview_channels_compatible(str(image_path), [channel_index], max_dim=DEFAULT_PREVIEW_MAX_DIM)
            preview = np.asarray(loaded.get(channel_index, np.zeros((1, 1), dtype=np.uint16)))
            return np.ravel(preview)
        full = np.asarray(read_channel_image(str(image_path), channel_index))
        return np.ravel(full)

    values = []
    for region in regions:
        if region.geometry is None:
            continue
        bbox = _clip_bounds(region.geometry.bounds, full_shape)
        if bbox is None:
            continue
        crop = np.asarray(read_channel_image(str(image_path), channel_index, region=bbox))
        mask = _geometry_mask(region.geometry, bbox)
        if crop.shape[:2] != mask.shape:
            continue
        region_values = crop[mask]
        if region_values.size > 0:
            values.append(np.asarray(region_values).ravel())
    if not values:
        return np.array([], dtype=np.float64)
    return np.concatenate(values)


def _build_summary_row(
    image_name: str,
    marker_name: str,
    values: np.ndarray,
    metadata: dict[str, Any],
    region_label: str,
    resolution_label: str,
) -> dict[str, Any]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "Image": image_name,
            "Region": region_label,
            "Marker": marker_name,
            "Pixels": 0,
            "Min": np.nan,
            "Max": np.nan,
            "Dynamic Range": np.nan,
            "Mean": np.nan,
            "Median": np.nan,
            "Std": np.nan,
            "P95": np.nan,
            "P99": np.nan,
            "Fluorophore": metadata.get("dye", "-"),
            "Exposure": metadata.get("exposure", "-"),
            "Excitation (nm)": metadata.get("excitation_wavelength", "-"),
            "Emission (nm)": metadata.get("emission_wavelength", "-"),
            "Analysis Resolution": resolution_label,
        }
    minimum = float(np.min(clean))
    maximum = float(np.max(clean))
    return {
        "Image": image_name,
        "Region": region_label,
        "Marker": marker_name,
        "Pixels": int(clean.size),
        "Min": minimum,
        "Max": maximum,
        "Dynamic Range": maximum - minimum,
        "Mean": float(np.mean(clean)),
        "Median": float(np.median(clean)),
        "Std": float(np.std(clean)),
        "P95": float(np.percentile(clean, 95)),
        "P99": float(np.percentile(clean, 99)),
        "Fluorophore": metadata.get("dye", "-"),
        "Exposure": metadata.get("exposure", "-"),
        "Excitation (nm)": metadata.get("excitation_wavelength", "-"),
        "Emission (nm)": metadata.get("emission_wavelength", "-"),
        "Analysis Resolution": resolution_label,
    }


def _build_marker_color_map(channel_names: list[str], channel_colors: list[Any]) -> dict[str, tuple[float, float, float]]:
    fallback_palette = plt.cm.tab20(np.linspace(0, 1, max(len(channel_names), 1)))
    color_map: dict[str, tuple[float, float, float]] = {}
    for idx, marker_name in enumerate(channel_names):
        rgb = np.asarray(_color_to_rgb(channel_colors[idx] if idx < len(channel_colors) else (255, 255, 255)), dtype=np.float32) / 255.0
        if np.all(rgb >= 0.95):
            fallback = np.asarray(fallback_palette[idx % len(fallback_palette)][:3], dtype=np.float32)
            color_map[marker_name] = (float(fallback[0]), float(fallback[1]), float(fallback[2]))
        else:
            color_map[marker_name] = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return color_map


def _build_nominal_color_map(names: list[str]) -> dict[str, tuple[float, float, float]]:
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(names), 1)))
    return {
        str(name): (float(color[0]), float(color[1]), float(color[2]))
        for name, color in zip(names, palette)
    }


def _sanitize_selection_state(key: str, options: list[str], default_selection: list[str]) -> None:
    if key not in st.session_state:
        st.session_state[key] = list(default_selection)
        return
    st.session_state[key] = [value for value in st.session_state.get(key, []) if value in options]


def _build_histogram_frame_from_samples(
    sample_map: dict[str, np.ndarray],
    transform: str,
    cofactor: float,
    bins: int,
    smooth_sigma: float,
    image_name: str,
    region_label: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for marker_name, values in sample_map.items():
        records.extend(
            _histogram_records(
                image_name=image_name,
                region_label=region_label,
                marker_name=marker_name,
                values=values,
                transform=transform,
                cofactor=cofactor,
                bins=bins,
                smooth_sigma=smooth_sigma,
            )
        )
    return pd.DataFrame(records)


def _build_histogram_figure(
    histogram_df: pd.DataFrame,
    transform: str,
    cofactor: float,
    marker_color_map: dict[str, tuple[float, float, float]],
    figure_options: HistogramFigureOptions,
    title: str,
    series_column: str = "Marker",
    legend_title: str = "Markers",
):
    if histogram_df is None or histogram_df.empty:
        return None
    if not figure_options.show_bars and not figure_options.show_curve:
        return None

    if series_column not in histogram_df.columns:
        return None

    series_names = list(dict.fromkeys(histogram_df[series_column].astype(str).tolist()))
    if not series_names:
        return None

    y_col = "Density" if figure_options.density else "Count"
    smooth_col = "Smooth Density" if figure_options.density else "Smooth Count"

    def _plot_series(ax, series_name: str, series_df: pd.DataFrame, include_legend: bool) -> bool:
        group = series_df.sort_values("Bin Center")
        if group.empty:
            return False
        edges = np.concatenate([group["Bin Left"].to_numpy(dtype=float), [float(group["Bin Right"].iloc[-1])]])
        color = marker_color_map.get(series_name, (0.2, 0.4, 0.8))
        plotted_local = False
        if figure_options.show_bars:
            ax.stairs(
                group[y_col].to_numpy(dtype=float),
                edges,
                fill=True,
                color=color,
                alpha=0.18,
                linewidth=1.2,
                label=series_name if include_legend and not figure_options.show_curve else None,
            )
            plotted_local = True
        if figure_options.show_curve:
            ax.plot(
                group["Bin Center"].to_numpy(dtype=float),
                group[smooth_col].to_numpy(dtype=float),
                color=color,
                linewidth=2.0,
                label=series_name if include_legend else None,
            )
            plotted_local = True
        return plotted_local

    if figure_options.layout_mode == "Stacked":
        figure_height = max(float(figure_options.height_inches), 2.2 * len(series_names))
        fig, axes = plt.subplots(
            len(series_names),
            1,
            sharex=True,
            figsize=(float(figure_options.width_inches), figure_height),
            squeeze=False,
        )
        axes_list = axes.flatten()
        plotted = False
        for axis, series_name in zip(axes_list, series_names):
            series_df = histogram_df[histogram_df[series_column].astype(str) == series_name]
            plotted = _plot_series(axis, series_name, series_df, include_legend=False) or plotted
            axis.set_ylabel("Density" if figure_options.density else "Count")
            axis.set_title(series_name, loc="left", color=marker_color_map.get(series_name, (0.2, 0.4, 0.8)), fontsize=11)
            axis.grid(alpha=0.2)
        axes_list[-1].set_xlabel(_transform_axis_label(transform, cofactor))
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig, ax = plt.subplots(figsize=(float(figure_options.width_inches), float(figure_options.height_inches)))
        plotted = False
        for series_name in series_names:
            series_df = histogram_df[histogram_df[series_column].astype(str) == series_name]
            plotted = _plot_series(ax, series_name, series_df, include_legend=True) or plotted
        ax.set_xlabel(_transform_axis_label(transform, cofactor))
        ax.set_ylabel("Density" if figure_options.density else "Count")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        if plotted:
            if figure_options.legend_outside:
                ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
                fig.tight_layout(rect=(0, 0, 0.8, 1))
            else:
                ax.legend(title=legend_title, loc="upper right", frameon=False, fontsize=9)
                fig.tight_layout()

    if not plotted:
        plt.close(fig)
        return None

    fig.set_dpi(int(figure_options.dpi))
    return fig


def _histogram_records(
    image_name: str,
    region_label: str,
    marker_name: str,
    values: np.ndarray,
    transform: str,
    cofactor: float,
    bins: int,
    smooth_sigma: float,
) -> list[dict[str, Any]]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return []
    transformed = _transform_values(clean, transform, cofactor)
    transformed = transformed[np.isfinite(transformed)]
    if transformed.size == 0:
        return []
    counts, edges = np.histogram(transformed, bins=int(bins), density=False)
    widths = np.diff(edges)
    density = counts / max(float(counts.sum()), 1.0) / np.where(widths == 0, 1.0, widths)
    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=float(smooth_sigma)) if smooth_sigma > 0 else counts.astype(np.float64)
    smooth_density = smoothed / max(float(counts.sum()), 1.0) / np.where(widths == 0, 1.0, widths)
    records = []
    for idx, count in enumerate(counts):
        records.append(
            {
                "Image": image_name,
                "Region": region_label,
                "Marker": marker_name,
                "Transform": transform,
                "Bin Left": float(edges[idx]),
                "Bin Right": float(edges[idx + 1]),
                "Bin Center": float((edges[idx] + edges[idx + 1]) / 2.0),
                "Count": int(count),
                "Density": float(density[idx]),
                "Smooth Count": float(smoothed[idx]),
                "Smooth Density": float(smooth_density[idx]),
            }
        )
    return records


def _figure_to_buffer(figure: plt.Figure, fmt: str, dpi: int) -> io.BytesIO:
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": fmt, "bbox_inches": "tight"}
    if fmt in {"png", "pdf"}:
        save_kwargs["dpi"] = int(dpi)
    figure.savefig(buffer, **save_kwargs)
    buffer.seek(0)
    return buffer


def _render_histogram_downloads(figure: plt.Figure, file_stem: str, dpi: int, key_prefix: str) -> None:
    st.markdown("##### Download Figure")
    col_png, col_pdf, col_svg = st.columns(3)
    with col_png:
        st.download_button(
            "Download PNG",
            _figure_to_buffer(figure, "png", dpi),
            file_name=f"{file_stem}.png",
            mime="image/png",
            key=f"{key_prefix}_png_download",
        )
    with col_pdf:
        st.download_button(
            "Download PDF",
            _figure_to_buffer(figure, "pdf", dpi),
            file_name=f"{file_stem}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf_download",
        )
    with col_svg:
        st.download_button(
            "Download SVG",
            _figure_to_buffer(figure, "svg", dpi),
            file_name=f"{file_stem}.svg",
            mime="image/svg+xml",
            key=f"{key_prefix}_svg_download",
        )


def _save_figure_variants(figure: plt.Figure, output_dir: Path, file_stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in ("png", "pdf", "svg"):
        target_path = output_dir / f"{file_stem}.{fmt}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt in {"png", "pdf"}:
            save_kwargs["dpi"] = int(dpi)
        figure.savefig(target_path, format=fmt, **save_kwargs)
        saved_paths.append(target_path)
    return saved_paths


def _export_histogram_figures(
    histogram_df: pd.DataFrame,
    output_dir: Path,
    marker_color_map: dict[str, tuple[float, float, float]],
    transform: str,
    cofactor: float,
    figure_options: HistogramFigureOptions,
    file_stem: str,
    title: str,
) -> list[Path]:
    if histogram_df.empty:
        return []

    saved_paths: list[Path] = []
    combined_figure = _build_histogram_figure(
        histogram_df=histogram_df,
        transform=transform,
        cofactor=cofactor,
        marker_color_map=marker_color_map,
        figure_options=figure_options,
        title=title,
        series_column="Marker",
        legend_title="Markers",
    )
    if combined_figure is not None:
        layout_slug = "stacked" if figure_options.layout_mode == "Stacked" else "overlay"
        saved_paths.extend(_save_figure_variants(combined_figure, output_dir, f"{file_stem}__{layout_slug}", figure_options.dpi))
        plt.close(combined_figure)

    if figure_options.save_separate_markers:
        separate_options = HistogramFigureOptions(
            layout_mode="Overlay",
            show_bars=figure_options.show_bars,
            show_curve=figure_options.show_curve,
            density=figure_options.density,
            legend_outside=False,
            width_inches=figure_options.width_inches,
            height_inches=figure_options.height_inches,
            dpi=figure_options.dpi,
            save_separate_markers=False,
        )
        for marker_name in list(dict.fromkeys(histogram_df["Marker"].astype(str).tolist())):
            marker_df = histogram_df[histogram_df["Marker"].astype(str) == marker_name]
            marker_figure = _build_histogram_figure(
                histogram_df=marker_df,
                transform=transform,
                cofactor=cofactor,
                marker_color_map=marker_color_map,
                figure_options=separate_options,
                title=f"{title} | {marker_name}",
                series_column="Marker",
                legend_title="Markers",
            )
            if marker_figure is None:
                continue
            saved_paths.extend(
                _save_figure_variants(
                    marker_figure,
                    output_dir,
                    f"{file_stem}__marker_{_safe_slug(marker_name)}",
                    figure_options.dpi,
                )
            )
            plt.close(marker_figure)
    return saved_paths


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> list[float]:
    clean_values = np.asarray(values, dtype=np.float64)
    clean_weights = np.asarray(weights, dtype=np.float64)
    finite_mask = np.isfinite(clean_values) & np.isfinite(clean_weights) & (clean_weights > 0)
    clean_values = clean_values[finite_mask]
    clean_weights = clean_weights[finite_mask]
    if clean_values.size == 0:
        return [float("nan") for _ in quantiles]
    order = np.argsort(clean_values)
    sorted_values = clean_values[order]
    sorted_weights = clean_weights[order]
    cumulative = np.cumsum(sorted_weights)
    total_weight = float(cumulative[-1])
    if total_weight <= 0:
        return [float("nan") for _ in quantiles]
    return [
        float(np.interp(float(q) * total_weight, cumulative, sorted_values))
        for q in quantiles
    ]


def _aggregate_compiled_chunks(chunks: list[BatchCompiledChunk]) -> pd.DataFrame:
    grouped: dict[tuple[str, str], list[BatchCompiledChunk]] = {}
    for chunk in chunks:
        grouped.setdefault((chunk.region_label, chunk.marker_name), []).append(chunk)

    rows: list[dict[str, Any]] = []
    for (region_label, marker_name), group in grouped.items():
        total_pixels = int(sum(chunk.pixel_count for chunk in group))
        total_sum = float(sum(chunk.value_sum for chunk in group))
        total_sum_sq = float(sum(chunk.value_sum_sq for chunk in group))
        exact_mean = total_sum / total_pixels if total_pixels > 0 else float("nan")
        variance = (total_sum_sq / total_pixels) - (exact_mean ** 2) if total_pixels > 0 else float("nan")
        exact_std = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else float("nan")
        image_names = sorted({chunk.image_name for chunk in group})
        samples = [chunk.sample_values.astype(np.float64, copy=False) for chunk in group if chunk.sample_values.size > 0]
        sample_weights = [
            np.full(chunk.sample_values.size, float(chunk.sample_weight), dtype=np.float64)
            for chunk in group
            if chunk.sample_values.size > 0
        ]
        compiled_sample_pixels = int(sum(chunk.sample_values.size for chunk in group))
        if samples and sample_weights:
            sample_values = np.concatenate(samples)
            weight_values = np.concatenate(sample_weights)
            approx_median, approx_p95, approx_p99 = _weighted_quantiles(sample_values, weight_values, [0.5, 0.95, 0.99])
        else:
            approx_median, approx_p95, approx_p99 = float("nan"), float("nan"), float("nan")

        rows.append(
            {
                "Region": region_label,
                "Marker": marker_name,
                "Images Used": "; ".join(image_names),
                "Image Count": len(image_names),
                "Total Pixels": total_pixels,
                "Compiled Sample Pixels": compiled_sample_pixels,
                "Min": float(min(chunk.min_value for chunk in group)) if group else float("nan"),
                "Max": float(max(chunk.max_value for chunk in group)) if group else float("nan"),
                "Dynamic Range": float(max(chunk.max_value for chunk in group) - min(chunk.min_value for chunk in group)) if group else float("nan"),
                "Mean": exact_mean,
                "Std": exact_std,
                "Approx Median": approx_median,
                "Approx P95": approx_p95,
                "Approx P99": approx_p99,
                "Fluorophores": "; ".join(sorted({str(chunk.fluorophore) for chunk in group if str(chunk.fluorophore) not in {"", "-"}})),
                "Exposures": "; ".join(sorted({str(chunk.exposure) for chunk in group if str(chunk.exposure) not in {"", "-"}})),
                "Excitation (nm)": "; ".join(sorted({str(chunk.excitation_wavelength) for chunk in group if str(chunk.excitation_wavelength) not in {"", "-"}})),
                "Emission (nm)": "; ".join(sorted({str(chunk.emission_wavelength) for chunk in group if str(chunk.emission_wavelength) not in {"", "-"}})),
                "Quantile Basis": "Weighted compiled per-image samples",
            }
        )
    return pd.DataFrame(rows)


def _build_cross_image_histogram_df(
    chunks: list[BatchCompiledChunk],
    transform: str,
    cofactor: float,
    bins: int,
    smooth_sigma: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.sample_values.size == 0:
            continue
        records.extend(
            _histogram_records(
                image_name=chunk.image_name,
                region_label=chunk.region_label,
                marker_name=chunk.marker_name,
                values=chunk.sample_values,
                transform=transform,
                cofactor=cofactor,
                bins=bins,
                smooth_sigma=smooth_sigma,
            )
        )
    return pd.DataFrame(records)


def _export_cross_image_histogram_figures(
    histogram_df: pd.DataFrame,
    output_dir: Path,
    transform: str,
    cofactor: float,
    figure_options: HistogramFigureOptions,
) -> list[Path]:
    if histogram_df.empty:
        return []
    image_color_map = _build_nominal_color_map(sorted({str(name) for name in histogram_df["Image"].astype(str).tolist()}))
    saved_paths: list[Path] = []
    for region_label in list(dict.fromkeys(histogram_df["Region"].astype(str).tolist())):
        region_df = histogram_df[histogram_df["Region"].astype(str) == region_label]
        for marker_name in list(dict.fromkeys(region_df["Marker"].astype(str).tolist())):
            marker_df = region_df[region_df["Marker"].astype(str) == marker_name]
            if marker_df.empty:
                continue
            figure = _build_histogram_figure(
                histogram_df=marker_df,
                transform=transform,
                cofactor=cofactor,
                marker_color_map=image_color_map,
                figure_options=figure_options,
                title=f"{marker_name} across images | {region_label}",
                series_column="Image",
                legend_title="Images",
            )
            if figure is None:
                continue
            layout_slug = "stacked" if figure_options.layout_mode == "Stacked" else "overlay"
            saved_paths.extend(
                _save_figure_variants(
                    figure,
                    output_dir,
                    f"cross_image_histogram__{_safe_slug(region_label)}__marker_{_safe_slug(marker_name)}__{layout_slug}",
                    figure_options.dpi,
                )
            )
            plt.close(figure)
    return saved_paths


def _run_batch_image_job(
    image_path: Path,
    output_root: Path,
    selected_marker_names: list[str],
    region_specs: list[RegionSpec],
    transform: str,
    cofactor: float,
    bins: int,
    smooth_sigma: float,
    export_raw: bool,
    raw_pixel_cap: int,
    figure_options: HistogramFigureOptions,
    compiled_sample_cap: int,
) -> BatchImageJobResult:
    summary_df, histogram_df, raw_paths, compiled_chunks = _export_image_reports(
        image_path=image_path,
        output_root=output_root,
        selected_marker_names=selected_marker_names,
        region_specs=region_specs,
        transform=transform,
        cofactor=cofactor,
        bins=bins,
        smooth_sigma=smooth_sigma,
        export_raw=export_raw,
        raw_pixel_cap=raw_pixel_cap,
        figure_options=figure_options,
        compiled_sample_cap=compiled_sample_cap,
    )
    return BatchImageJobResult(
        image_name=image_path.name,
        summary_df=summary_df,
        histogram_df=histogram_df,
        raw_paths=raw_paths,
        compiled_chunks=compiled_chunks,
    )


def _analysis_signature(
    image_path: Path,
    marker_names: list[str],
    region_labels: list[str],
    use_thumbnail_for_whole: bool,
) -> str:
    payload = {
        "image": str(image_path),
        "markers": marker_names,
        "regions": region_labels,
        "thumbnail": use_thumbnail_for_whole,
    }
    return json.dumps(payload, sort_keys=True)


def _render_canvas_editor(
    prefix: str,
    image_name: str,
    preview_image: Image.Image,
    full_shape: tuple[int, int],
) -> list[RegionSpec]:
    if not HAS_CANVAS:
        st.error("streamlit-drawable-canvas is not installed.")
        return []

    objects_store_key = f"{prefix}_canvas_objects"
    labels_store_key = f"{prefix}_canvas_labels"
    reset_store_key = f"{prefix}_canvas_reset_counts"
    draw_mode_key = f"{prefix}_draw_active"
    background_cache_key = f"{prefix}_canvas_backgrounds"

    if objects_store_key not in st.session_state:
        st.session_state[objects_store_key] = {}
    if labels_store_key not in st.session_state:
        st.session_state[labels_store_key] = {}
    if reset_store_key not in st.session_state:
        st.session_state[reset_store_key] = {}
    if draw_mode_key not in st.session_state:
        st.session_state[draw_mode_key] = True
    if background_cache_key not in st.session_state:
        st.session_state[background_cache_key] = {}

    objects_by_image = st.session_state[objects_store_key]
    labels_by_image = st.session_state[labels_store_key]
    reset_counts = st.session_state[reset_store_key]
    backgrounds = st.session_state[background_cache_key]

    canvas_width = preview_image.width
    canvas_height = preview_image.height
    full_h, full_w = full_shape
    scale_x = full_w / float(max(canvas_width, 1)) if full_w else 1.0
    scale_y = full_h / float(max(canvas_height, 1)) if full_h else 1.0

    cache_key = (image_name, canvas_width, canvas_height)
    if cache_key not in backgrounds:
        backgrounds[cache_key] = {
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
            "src": image_to_base64(preview_image),
            "selectable": False,
            "evented": False,
            "hoverCursor": "default",
        }

    if image_name not in reset_counts:
        reset_counts[image_name] = 0

    tool_col, info_col = st.columns([1, 3])
    draw_active = bool(st.session_state.get(draw_mode_key, True))
    if tool_col.button(
        "Stop Drawing" if draw_active else "Draw New ROI",
        key=f"{prefix}_toggle_draw_{image_name}",
        type="primary" if draw_active else "secondary",
    ):
        draw_active = not draw_active
        st.session_state[draw_mode_key] = draw_active

    if draw_active:
        info_col.info("Left-click to add points. Right-click closes a polygon. Click Sync to register updates.")
        drawing_mode = "polygon"
    else:
        info_col.info("Use transform mode to select or delete shapes, then click Sync.")
        drawing_mode = "transform"

    if st.button("Reset Canvas", key=f"{prefix}_reset_canvas_{image_name}"):
        objects_by_image.pop(image_name, None)
        labels_by_image.pop(image_name, None)
        reset_counts[image_name] = int(reset_counts.get(image_name, 0)) + 1
        st.rerun()

    stored_objects = copy.deepcopy(objects_by_image.get(image_name, []))
    initial_drawing = {"version": "4.4.0", "objects": [backgrounds[cache_key]] + stored_objects}
    canvas_key = f"{prefix}_canvas_{image_name}_{reset_counts[image_name]}"

    st.caption("Canvas updates are applied when you click Sync Canvas.")
    with st.form(key=f"{prefix}_canvas_form_{image_name}_{reset_counts[image_name]}"):
        canvas = cast(Any, st_canvas)
        result = canvas(
            fill_color="rgba(255, 165, 0, 0.30)",
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
        sync_clicked = st.form_submit_button("Sync Canvas", type="primary")

    if sync_clicked and result is not None and result.json_data is not None:
        raw_objects = result.json_data.get("objects", []) or []
        valid_objects = []
        seen_signatures: set[str] = set()
        for obj in raw_objects:
            if not isinstance(obj, dict) or obj.get("type") == "image":
                continue
            if not _is_valid_canvas_object(obj):
                continue
            signature = _canvas_object_signature(obj)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            valid_objects.append(obj)
        objects_by_image[image_name] = valid_objects

    current_objects = [obj for obj in objects_by_image.get(image_name, []) if isinstance(obj, dict) and _is_valid_canvas_object(obj)]
    if not current_objects:
        st.caption("No drawn polygons synced yet.")
        return []

    existing_labels = labels_by_image.get(image_name, {})
    updated_labels: dict[int, str] = {}
    delete_index = -1

    h1, h2, h3 = st.columns([1, 4, 1])
    h1.markdown("**#**")
    h2.markdown("**Label**")
    h3.markdown("**Del**")

    for idx, _obj in enumerate(current_objects):
        default_label = str(existing_labels.get(idx, f"ROI {idx + 1}"))
        c1, c2, c3 = st.columns([1, 4, 1])
        c1.text(str(idx + 1))
        updated_labels[idx] = c2.text_input(
            f"ROI label {idx + 1}",
            value=default_label,
            key=f"{prefix}_roi_label_{image_name}_{idx}",
            label_visibility="collapsed",
        )
        if c3.button("Del", key=f"{prefix}_roi_delete_{image_name}_{idx}"):
            delete_index = idx

    labels_by_image[image_name] = updated_labels
    if delete_index >= 0:
        current_objects.pop(delete_index)
        objects_by_image[image_name] = current_objects
        new_labels = {}
        for idx, label in updated_labels.items():
            if idx < delete_index:
                new_labels[idx] = label
            elif idx > delete_index:
                new_labels[idx - 1] = label
        labels_by_image[image_name] = new_labels
        reset_counts[image_name] = int(reset_counts.get(image_name, 0)) + 1
        st.rerun()

    labels = [updated_labels.get(idx, f"ROI {idx + 1}") for idx in range(len(current_objects))]
    return _regions_from_canvas_objects(current_objects, labels, scale_x, scale_y)


def _metadata_lookup(channel_names: list[str], channel_metadata: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for idx, name in enumerate(channel_names):
        lookup[name] = channel_metadata[idx] if idx < len(channel_metadata) else {}
    return lookup


def _render_single_image(project_root: Path) -> None:
    default_image_dir = str(project_root)
    default_geojson_dir = str(project_root / "data" / "boundaries")
    default_output_dir = str(project_root / "outputs" / "marker_analysis")

    col_header, col_reset = st.columns([4, 1])
    col_header.header("Single Image")
    if col_reset.button("Reset", key="tab9_marker_single_reset"):
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith("tab9_marker_single_"):
                del st.session_state[key]
        st.rerun()

    controls_col, preview_col = st.columns([1.05, 1.2])

    with controls_col:
        image_dir = directory_input_with_browse(
            label="Images Directory",
            text_key="tab9_marker_single_image_dir",
            default_value=default_image_dir,
            browser_key_prefix="tab9_marker_single_image_dir",
            start_dir_fallback=str(project_root),
        )
        geojson_dir = directory_input_with_browse(
            label="GeoJSON Directory",
            text_key="tab9_marker_single_geojson_dir",
            default_value=default_geojson_dir,
            browser_key_prefix="tab9_marker_single_geojson_dir",
            start_dir_fallback=str(project_root),
        )
        output_dir = directory_input_with_browse(
            label="Output Directory",
            text_key="tab9_marker_single_output_dir",
            default_value=default_output_dir,
            browser_key_prefix="tab9_marker_single_output_dir",
            start_dir_fallback=str(project_root),
        )

        image_files = _list_image_files(image_dir)
        if not image_files:
            st.info("Select an image directory containing OME-TIFF or QPTIFF files.")
            return

        selected_image = st.selectbox(
            "Image",
            options=image_files,
            key="tab9_marker_single_image",
        )
        image_path = Path(image_dir) / selected_image

        try:
            channel_names, channel_colors, channel_meta, _mpp = _get_cached_channel_info(str(image_path), image_path.stat().st_mtime)
        except Exception as exc:
            st.error(f"Failed to read image metadata: {exc}")
            return

        if not channel_names:
            st.warning("No markers were found in the selected image.")
            return

        st.markdown("### Marker Metadata")
        selected_markers, marker_table = _build_marker_table(
            "tab9_marker_single",
            normalize_image_stem(selected_image),
            channel_names,
            channel_colors,
            channel_meta,
        )
        marker_color_map = _build_marker_color_map(channel_names, channel_colors)
        if not selected_markers:
            st.info("Enable at least one marker for analysis or export.")

        preview_options = [name for name in channel_names if name in selected_markers] or list(channel_names)
        default_preview = preview_options[: min(3, len(preview_options))]
        preview_markers = st.multiselect(
            "Preview markers",
            options=channel_names,
            default=default_preview,
            max_selections=3,
            key="tab9_marker_single_preview_markers",
        )
        preview_mode = st.selectbox(
            "Preview mode",
            options=["Composite", "Single Channel"],
            index=0 if len(preview_markers) != 1 else 1,
            key="tab9_marker_single_preview_mode",
        )

        region_mode = st.radio(
            "Region source",
            options=["Whole image", "GeoJSON", "Draw polygons"],
            horizontal=True,
            key="tab9_marker_single_region_mode",
        )

        geojson_path: Path | None = None
        loaded_geojson_regions: list[RegionSpec] = []
        if region_mode == "GeoJSON":
            geojson_files = _list_geojson_files(geojson_dir)
            if not geojson_files:
                st.warning("No GeoJSON files found in the selected directory.")
            else:
                default_geojson = _matched_geojson_name(selected_image, geojson_files)
                default_index = geojson_files.index(default_geojson) if default_geojson in geojson_files else 0
                selected_geojson = st.selectbox(
                    "GeoJSON file",
                    options=geojson_files,
                    index=default_index,
                    key="tab9_marker_single_geojson_file",
                )
                geojson_path = Path(geojson_dir) / selected_geojson
                loaded_geojson_regions = _load_regions_from_geojson(geojson_path)
                if not loaded_geojson_regions:
                    st.warning("The selected GeoJSON did not contain usable polygon regions.")

        histogram_options = selected_markers or channel_names
        default_histogram_markers = histogram_options[: min(4, len(histogram_options))]
        _sanitize_selection_state("tab9_marker_single_hist_markers", histogram_options, default_histogram_markers)
        h_sel_all, h_sel_clear = st.columns(2)
        if h_sel_all.button("Select All Histogram Markers", key="tab9_marker_single_hist_select_all"):
            st.session_state["tab9_marker_single_hist_markers"] = list(histogram_options)
            st.rerun()
        if h_sel_clear.button("Clear Histogram Markers", key="tab9_marker_single_hist_clear"):
            st.session_state["tab9_marker_single_hist_markers"] = []
            st.rerun()
        histogram_markers = st.multiselect(
            "Histogram markers",
            options=histogram_options,
            key="tab9_marker_single_hist_markers",
        )
        transform = st.selectbox("Histogram X-axis", options=list(TRANSFORM_OPTIONS), key="tab9_marker_single_transform")
        cofactor = st.number_input(
            "Arcsinh cofactor",
            min_value=0.1,
            value=5.0,
            step=0.5,
            key="tab9_marker_single_cofactor",
        )
        bins = st.slider("Histogram bins", 16, 512, DEFAULT_HISTOGRAM_BINS, step=16, key="tab9_marker_single_bins")
        smooth_sigma = st.slider("Smoothing", 0.0, 6.0, 1.5, step=0.5, key="tab9_marker_single_smoothing")
        hist_sample_cap = st.number_input(
            "Histogram pixel cap",
            min_value=10_000,
            max_value=2_000_000,
            value=DEFAULT_HIST_SAMPLE_CAP,
            step=10_000,
            key="tab9_marker_single_hist_cap",
            help="Interactive histograms use sampled pixels after summary stats are computed.",
        )
        use_thumbnail_for_whole = st.checkbox(
            "Use thumbnail resolution for whole-image preview analysis",
            value=True,
            key="tab9_marker_single_thumbnail_preview",
            help="Interactive summary and histograms stay responsive on large images. Export still uses the full-resolution image.",
        )
        fig_opt1, fig_opt2, fig_opt3 = st.columns(3)
        show_hist_bars = fig_opt1.checkbox("Show bars", value=True, key="tab9_marker_single_hist_bars")
        show_hist_curve = fig_opt2.checkbox("Show smooth curve", value=True, key="tab9_marker_single_hist_curve")
        density = fig_opt3.checkbox("Density y-axis", value=True, key="tab9_marker_single_hist_density")
        fig_layout_cols = st.columns([1.2, 1])
        layout_mode = fig_layout_cols[0].radio(
            "Histogram layout",
            options=["Overlay", "Stacked"],
            horizontal=True,
            key="tab9_marker_single_hist_layout",
        )
        legend_outside = fig_layout_cols[1].checkbox(
            "Legend outside right",
            value=True,
            key="tab9_marker_single_hist_legend_outside",
            disabled=layout_mode == "Stacked",
        )
        fig_size_cols = st.columns(3)
        fig_width = fig_size_cols[0].number_input("Figure width (in)", min_value=4.0, max_value=20.0, value=10.0, step=0.5, key="tab9_marker_single_hist_width")
        fig_height = fig_size_cols[1].number_input("Figure height (in)", min_value=3.0, max_value=20.0, value=4.8, step=0.5, key="tab9_marker_single_hist_height")
        fig_dpi = int(fig_size_cols[2].number_input("Figure DPI", min_value=72, max_value=1200, value=300, step=25, key="tab9_marker_single_hist_dpi"))
        save_separate_hist_figures = st.checkbox(
            "Save separate figure per marker during export",
            value=True,
            key="tab9_marker_single_hist_save_separate",
        )
        histogram_figure_options = HistogramFigureOptions(
            layout_mode=layout_mode,
            show_bars=bool(show_hist_bars),
            show_curve=bool(show_hist_curve),
            density=bool(density),
            legend_outside=bool(legend_outside and layout_mode == "Overlay"),
            width_inches=float(fig_width),
            height_inches=float(fig_height),
            dpi=int(fig_dpi),
            save_separate_markers=bool(save_separate_hist_figures),
        )

    with preview_col:
        preview_image, full_shape = _build_preview_pil(image_path, channel_names, channel_colors, preview_mode, preview_markers)
        if full_shape[0] <= 0 or full_shape[1] <= 0:
            full_shape = _get_image_shape_cached(str(image_path), image_path.stat().st_mtime)
        drawn_regions: list[RegionSpec] = []
        preview_regions: list[RegionSpec] = []

        if region_mode == "Draw polygons":
            drawn_regions = _render_canvas_editor("tab9_marker_single", selected_image, preview_image, full_shape)
            preview_regions = drawn_regions
        else:
            preview_regions = _build_region_preview_regions(region_mode, full_shape, geojson_path, drawn_regions)
            overlay = _overlay_regions_on_preview(preview_image, preview_regions, full_shape)
            st.image(overlay, caption=f"Preview: {selected_image}")

        available_region_labels = [region.label for region in preview_regions if region.geometry is not None]
        if available_region_labels:
            selected_region_labels = st.multiselect(
                "Regions to analyze/export",
                options=available_region_labels,
                default=available_region_labels,
                key="tab9_marker_single_regions",
            )
        else:
            selected_region_labels = []

        effective_regions = _selected_regions_from_labels(preview_regions, selected_region_labels)
        region_descriptor = ", ".join([region.label for region in effective_regions]) or WHOLE_IMAGE_LABEL

        analysis_key = "tab9_marker_single_analysis"
        current_signature = _analysis_signature(image_path, histogram_markers, [region.label for region in effective_regions], use_thumbnail_for_whole)

        if st.button("Analyze Selected Markers", type="primary", key="tab9_marker_single_run"):
            metadata_lookup = _metadata_lookup(channel_names, channel_meta)
            index_lookup = {name: idx for idx, name in enumerate(channel_names)}
            summary_rows = []
            sample_map: dict[str, np.ndarray] = {}
            resolution_label = "Thumbnail" if use_thumbnail_for_whole and (not effective_regions or effective_regions[0].geometry is None) else "Full"

            for marker_name in histogram_markers:
                if marker_name not in index_lookup:
                    continue
                values = _extract_combined_values(
                    image_path,
                    index_lookup[marker_name],
                    full_shape,
                    effective_regions,
                    use_thumbnail_for_whole=use_thumbnail_for_whole,
                )
                summary_rows.append(
                    _build_summary_row(
                        image_name=selected_image,
                        marker_name=marker_name,
                        values=values,
                        metadata=metadata_lookup.get(marker_name, {}),
                        region_label=region_descriptor,
                        resolution_label=resolution_label,
                    )
                )
                sample_map[marker_name] = _sample_array(np.asarray(values).ravel(), int(hist_sample_cap))

            st.session_state[analysis_key] = {
                "signature": current_signature,
                "summary": pd.DataFrame(summary_rows),
                "sample_map": sample_map,
            }

        analysis_state = st.session_state.get(analysis_key)
        if analysis_state and analysis_state.get("signature") == current_signature:
            summary_df = analysis_state.get("summary")
            sample_map = analysis_state.get("sample_map", {})
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                st.markdown("### Marker Summary")
                st.dataframe(summary_df, hide_index=True)
            preview_histogram_df = _build_histogram_frame_from_samples(
                sample_map=sample_map,
                transform=transform,
                cofactor=float(cofactor),
                bins=int(bins),
                smooth_sigma=float(smooth_sigma),
                image_name=selected_image,
                region_label=region_descriptor,
            )
            figure = _build_histogram_figure(
                histogram_df=preview_histogram_df,
                transform=transform,
                cofactor=float(cofactor),
                marker_color_map=marker_color_map,
                figure_options=histogram_figure_options,
                title=f"{selected_image} | {region_descriptor}",
            )
            if figure is not None:
                st.markdown("### Histogram")
                st.pyplot(figure)
                _render_histogram_downloads(
                    figure,
                    file_stem=f"{normalize_image_stem(selected_image)}__{_safe_slug(region_descriptor)}__histogram",
                    dpi=histogram_figure_options.dpi,
                    key_prefix="tab9_marker_single_hist",
                )
                plt.close(figure)
            elif not histogram_figure_options.show_bars and not histogram_figure_options.show_curve:
                st.info("Enable bars, the smoothed curve, or both to visualize the histogram.")
        elif histogram_markers:
            st.info("Click Analyze Selected Markers to compute dynamic range, percentiles, and histograms.")

        st.markdown("### Export")
        export_raw = st.checkbox("Export raw pixel intensities", value=True, key="tab9_marker_single_export_raw")
        raw_pixel_cap = st.number_input(
            "Raw export pixel cap per region (0 = all)",
            min_value=0,
            max_value=5_000_000,
            value=DEFAULT_RAW_EXPORT_CAP,
            step=10_000,
            key="tab9_marker_single_raw_cap",
        )
        if st.button("Export Reports", key="tab9_marker_single_export", type="primary"):
            if not selected_markers:
                st.warning("Select at least one marker before exporting.")
            else:
                output_root = Path(output_dir).expanduser()
                output_root.mkdir(parents=True, exist_ok=True)
                with st.status("Exporting marker analysis...", expanded=True) as status:
                    try:
                        summary_df, histogram_df, raw_paths, _compiled_chunks = _export_image_reports(
                            image_path=image_path,
                            output_root=output_root,
                            selected_marker_names=selected_markers,
                            region_specs=effective_regions,
                            transform=transform,
                            cofactor=float(cofactor),
                            bins=int(bins),
                            smooth_sigma=float(smooth_sigma),
                            export_raw=bool(export_raw),
                            raw_pixel_cap=int(raw_pixel_cap),
                            figure_options=histogram_figure_options,
                        )
                        status.write(f"Summary rows: {len(summary_df)}")
                        status.write(f"Histogram rows: {len(histogram_df)}")
                        for raw_path in raw_paths:
                            status.write(f"Raw intensities: {raw_path.name}")
                        status.write(f"Histogram figures: {(output_root / normalize_image_stem(image_path) / 'figures')}")
                        status.update(label="Marker analysis export complete", state="complete")
                        st.success(f"Reports exported to {output_root / normalize_image_stem(image_path)}")
                    except Exception as exc:
                        status.update(label="Marker analysis export failed", state="error")
                        st.error(f"Export failed: {exc}")


def _export_image_reports(
    image_path: Path,
    output_root: Path,
    selected_marker_names: list[str],
    region_specs: list[RegionSpec],
    transform: str,
    cofactor: float,
    bins: int,
    smooth_sigma: float,
    export_raw: bool,
    raw_pixel_cap: int,
    figure_options: HistogramFigureOptions | None = None,
    compiled_sample_cap: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path], list[BatchCompiledChunk]]:
    channel_names, channel_colors, channel_meta, _mpp = _get_cached_channel_info(str(image_path), image_path.stat().st_mtime)
    index_lookup = {name: idx for idx, name in enumerate(channel_names)}
    meta_lookup = _metadata_lookup(channel_names, channel_meta)
    marker_color_map = _build_marker_color_map(channel_names, channel_colors)
    selected_available = [name for name in selected_marker_names if name in index_lookup]
    if not selected_available:
        raise ValueError("None of the selected markers were available in this image.")

    if figure_options is None:
        figure_options = HistogramFigureOptions(
            layout_mode="Overlay",
            show_bars=True,
            show_curve=True,
            density=True,
            legend_outside=True,
            width_inches=10.0,
            height_inches=4.8,
            dpi=300,
            save_separate_markers=True,
        )

    full_shape = _get_image_shape_cached(str(image_path), image_path.stat().st_mtime)
    if full_shape[0] <= 0 or full_shape[1] <= 0:
        fallback = np.asarray(read_channel_image(str(image_path), index_lookup[selected_available[0]]))
        full_shape = (int(fallback.shape[-2]), int(fallback.shape[-1]))
    image_output_dir = output_root / normalize_image_stem(image_path)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for marker_name in selected_available:
        idx = index_lookup[marker_name]
        metadata = meta_lookup.get(marker_name, {})
        metadata_rows.append(
            {
                "Marker Index": idx,
                "Marker": marker_name,
                "Color": _color_to_hex(channel_colors[idx] if idx < len(channel_colors) else (255, 255, 255)),
                "Fluorophore": metadata.get("dye", "-"),
                "Exposure": metadata.get("exposure", "-"),
                "Excitation (nm)": metadata.get("excitation_wavelength", "-"),
                "Emission (nm)": metadata.get("emission_wavelength", "-"),
            }
        )
    pd.DataFrame(metadata_rows).to_csv(image_output_dir / "marker_metadata.csv", index=False)

    effective_regions = region_specs or [RegionSpec(label=WHOLE_IMAGE_LABEL, geometry=None, source="Whole image")]
    summary_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    compiled_chunks: list[BatchCompiledChunk] = []

    for region in effective_regions:
        raw_frame: pd.DataFrame | None = None
        if region.geometry is None:
            region_shape = full_shape
            total_pixels = int(region_shape[0] * region_shape[1])
            raw_indices = _choose_indices(total_pixels, raw_pixel_cap)
            y_coords, x_coords = np.unravel_index(raw_indices, region_shape) if export_raw else (np.array([]), np.array([]))
            raw_columns: dict[str, Any] = {
                "image": np.repeat(image_path.name, len(raw_indices)),
                "region": np.repeat(region.label, len(raw_indices)),
                "y": y_coords,
                "x": x_coords,
            } if export_raw else {}
            mask = None
            bbox = None
        else:
            bbox = _clip_bounds(region.geometry.bounds, full_shape)
            if bbox is None:
                continue
            mask = _geometry_mask(region.geometry, bbox)
            flat_region_indices = np.flatnonzero(mask.ravel())
            if flat_region_indices.size == 0:
                continue
            chosen = _choose_indices(flat_region_indices.size, raw_pixel_cap)
            sampled_flat = flat_region_indices[chosen] if export_raw else np.array([], dtype=np.int64)
            local_y, local_x = np.unravel_index(sampled_flat, mask.shape) if export_raw else (np.array([]), np.array([]))
            y1, y2, x1, x2 = bbox
            raw_columns = {
                "image": np.repeat(image_path.name, len(sampled_flat)),
                "region": np.repeat(region.label, len(sampled_flat)),
                "y": local_y + y1,
                "x": local_x + x1,
            } if export_raw else {}
            raw_indices = sampled_flat

        for marker_name in selected_available:
            marker_index = index_lookup[marker_name]
            metadata = meta_lookup.get(marker_name, {})
            if region.geometry is None:
                channel_data = np.asarray(read_channel_image(str(image_path), marker_index))
                values = channel_data.ravel()
                sampled_values = values[raw_indices] if export_raw else np.array([], dtype=channel_data.dtype)
            else:
                assert bbox is not None and mask is not None
                channel_data = np.asarray(read_channel_image(str(image_path), marker_index, region=bbox))
                if channel_data.shape[:2] != mask.shape:
                    continue
                values = channel_data[mask]
                sampled_values = channel_data.ravel()[raw_indices] if export_raw else np.array([], dtype=channel_data.dtype)

            summary_rows.append(
                _build_summary_row(
                    image_name=image_path.name,
                    marker_name=marker_name,
                    values=np.asarray(values),
                    metadata=metadata,
                    region_label=region.label,
                    resolution_label="Full",
                )
            )
            histogram_rows.extend(
                _histogram_records(
                    image_name=image_path.name,
                    region_label=region.label,
                    marker_name=marker_name,
                    values=np.asarray(values),
                    transform=transform,
                    cofactor=cofactor,
                    bins=bins,
                    smooth_sigma=smooth_sigma,
                )
            )
            if compiled_sample_cap > 0:
                values_float = np.asarray(values, dtype=np.float64)
                compiled_sample = _sample_array(np.asarray(values).ravel(), int(compiled_sample_cap))
                sample_weight = float(values.size) / float(compiled_sample.size) if compiled_sample.size > 0 else 0.0
                compiled_chunks.append(
                    BatchCompiledChunk(
                        image_name=image_path.name,
                        region_label=region.label,
                        marker_name=marker_name,
                        pixel_count=int(values.size),
                        value_sum=float(np.sum(values_float, dtype=np.float64)) if values.size > 0 else 0.0,
                        value_sum_sq=float(np.sum(values_float * values_float, dtype=np.float64)) if values.size > 0 else 0.0,
                        min_value=float(np.min(values)) if values.size > 0 else float("nan"),
                        max_value=float(np.max(values)) if values.size > 0 else float("nan"),
                        sample_values=np.asarray(compiled_sample, dtype=np.float64),
                        sample_weight=sample_weight,
                        fluorophore=str(metadata.get("dye", "-")),
                        exposure=str(metadata.get("exposure", "-")),
                        excitation_wavelength=str(metadata.get("excitation_wavelength", "-")),
                        emission_wavelength=str(metadata.get("emission_wavelength", "-")),
                    )
                )
            if export_raw:
                raw_columns[marker_name] = sampled_values

        if export_raw and raw_columns:
            raw_frame = pd.DataFrame(raw_columns)
            raw_path = image_output_dir / f"marker_intensities__{_safe_slug(region.label)}.csv.gz"
            raw_frame.to_csv(raw_path, index=False, compression="gzip")
            raw_paths.append(raw_path)

    summary_df = pd.DataFrame(summary_rows)
    histogram_df = pd.DataFrame(histogram_rows)
    summary_df.to_csv(image_output_dir / "marker_summary.csv", index=False)
    histogram_df.to_csv(image_output_dir / "marker_histograms.csv", index=False)

    if not histogram_df.empty:
        figures_dir = image_output_dir / "figures"
        for region_label in list(dict.fromkeys(histogram_df["Region"].astype(str).tolist())):
            region_histogram_df = histogram_df[histogram_df["Region"].astype(str) == region_label].copy()
            if region_histogram_df.empty:
                continue
            _export_histogram_figures(
                histogram_df=region_histogram_df,
                output_dir=figures_dir,
                marker_color_map=marker_color_map,
                transform=transform,
                cofactor=cofactor,
                figure_options=figure_options,
                file_stem=f"marker_histogram__{_safe_slug(region_label)}",
                title=f"{image_path.name} | {region_label}",
            )
    return summary_df, histogram_df, raw_paths, compiled_chunks


def _resolve_batch_regions(
    mode: str,
    image_name: str,
    full_shape: tuple[int, int],
    geojson_dir: str,
    shared_geojson_path: Path | None,
    shared_drawn_regions: list[RegionSpec],
) -> list[RegionSpec]:
    if mode == "Whole image":
        return [RegionSpec(label=WHOLE_IMAGE_LABEL, geometry=None, source="Whole image")]
    if mode == "Matched GeoJSON":
        geojson_files = _list_geojson_files(geojson_dir)
        matched = _matched_geojson_name(image_name, geojson_files)
        if not matched:
            raise FileNotFoundError(f"No matching GeoJSON found for {image_name}")
        return _load_regions_from_geojson(Path(geojson_dir) / matched)
    if mode == "Shared GeoJSON":
        if shared_geojson_path is None:
            raise FileNotFoundError("Select a shared GeoJSON file.")
        return _load_regions_from_geojson(shared_geojson_path)
    if mode == "Shared Drawn Polygons":
        if not shared_drawn_regions:
            raise ValueError("Draw at least one polygon on the active image before batch export.")
        return shared_drawn_regions
    return [RegionSpec(label=WHOLE_IMAGE_LABEL, geometry=None, source="Whole image")]


def _render_batch(project_root: Path) -> None:
    default_image_dir = str(project_root)
    default_geojson_dir = str(project_root / "data" / "boundaries")
    default_output_dir = str(project_root / "outputs" / "marker_analysis_batch")

    col_header, col_reset = st.columns([4, 1])
    col_header.header("Multiple Images")
    if col_reset.button("Reset", key="tab9_marker_batch_reset"):
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith("tab9_marker_batch_"):
                del st.session_state[key]
        st.rerun()

    controls_col, preview_col = st.columns([1.05, 1.2])

    with controls_col:
        image_dir = directory_input_with_browse(
            label="Images Directory",
            text_key="tab9_marker_batch_image_dir",
            default_value=default_image_dir,
            browser_key_prefix="tab9_marker_batch_image_dir",
            start_dir_fallback=str(project_root),
        )
        geojson_dir = directory_input_with_browse(
            label="GeoJSON Directory",
            text_key="tab9_marker_batch_geojson_dir",
            default_value=default_geojson_dir,
            browser_key_prefix="tab9_marker_batch_geojson_dir",
            start_dir_fallback=str(project_root),
        )
        output_dir = directory_input_with_browse(
            label="Output Directory",
            text_key="tab9_marker_batch_output_dir",
            default_value=default_output_dir,
            browser_key_prefix="tab9_marker_batch_output_dir",
            start_dir_fallback=str(project_root),
        )

        image_files = _list_image_files(image_dir)
        if not image_files:
            st.info("Select an image directory to enable batch export.")
            return

        selected_images = st.multiselect(
            "Images to export",
            options=image_files,
            default=image_files[: min(3, len(image_files))],
            key="tab9_marker_batch_images",
        )
        preview_candidates = selected_images or image_files
        active_image = st.selectbox(
            "Active preview image",
            options=preview_candidates,
            key="tab9_marker_batch_active_image",
        )
        active_path = Path(image_dir) / active_image

        try:
            channel_names, channel_colors, channel_meta, _mpp = _get_cached_channel_info(str(active_path), active_path.stat().st_mtime)
        except Exception as exc:
            st.error(f"Failed to read marker metadata for {active_image}: {exc}")
            return

        if not channel_names:
            st.warning("No markers were found in the active image.")
            return

        st.markdown("### Batch Marker Selection")
        selected_markers, _marker_table = _build_marker_table(
            "tab9_marker_batch",
            normalize_image_stem(active_image),
            channel_names,
            channel_colors,
            channel_meta,
        )
        st.caption("Batch export matches markers by name. Missing markers in a given image are skipped and noted in the batch manifest.")

        preview_options = [name for name in channel_names if name in selected_markers] or list(channel_names)
        preview_markers = st.multiselect(
            "Preview markers",
            options=channel_names,
            default=preview_options[: min(3, len(preview_options))],
            max_selections=3,
            key="tab9_marker_batch_preview_markers",
        )
        preview_mode = st.selectbox(
            "Preview mode",
            options=["Composite", "Single Channel"],
            index=0 if len(preview_markers) != 1 else 1,
            key="tab9_marker_batch_preview_mode",
        )

        batch_region_mode = st.radio(
            "Batch region source",
            options=["Whole image", "Matched GeoJSON", "Shared GeoJSON", "Shared Drawn Polygons"],
            key="tab9_marker_batch_region_mode",
        )

        shared_geojson_path: Path | None = None
        if batch_region_mode == "Shared GeoJSON":
            geojson_files = _list_geojson_files(geojson_dir)
            if not geojson_files:
                st.warning("No GeoJSON files found for shared-region batch export.")
            else:
                shared_geojson_name = st.selectbox(
                    "Shared GeoJSON file",
                    options=geojson_files,
                    key="tab9_marker_batch_shared_geojson",
                )
                shared_geojson_path = Path(geojson_dir) / shared_geojson_name

        transform = st.selectbox("Histogram/report transform", options=list(TRANSFORM_OPTIONS), key="tab9_marker_batch_transform")
        cofactor = st.number_input("Arcsinh cofactor", min_value=0.1, value=5.0, step=0.5, key="tab9_marker_batch_cofactor")
        bins = st.slider("Histogram bins", 16, 512, DEFAULT_HISTOGRAM_BINS, step=16, key="tab9_marker_batch_bins")
        smooth_sigma = st.slider("Histogram smoothing", 0.0, 6.0, 1.5, step=0.5, key="tab9_marker_batch_smoothing")
        export_raw = st.checkbox("Export raw pixel intensities", value=True, key="tab9_marker_batch_export_raw")
        raw_pixel_cap = st.number_input(
            "Raw export pixel cap per region (0 = all)",
            min_value=0,
            max_value=5_000_000,
            value=DEFAULT_RAW_EXPORT_CAP,
            step=10_000,
            key="tab9_marker_batch_raw_cap",
        )
        batch_fig_opt1, batch_fig_opt2, batch_fig_opt3 = st.columns(3)
        batch_show_bars = batch_fig_opt1.checkbox("Show bars in saved figures", value=True, key="tab9_marker_batch_hist_bars")
        batch_show_curve = batch_fig_opt2.checkbox("Show smooth curve in saved figures", value=True, key="tab9_marker_batch_hist_curve")
        batch_density = batch_fig_opt3.checkbox("Density y-axis in saved figures", value=True, key="tab9_marker_batch_hist_density")
        batch_layout_cols = st.columns([1.2, 1])
        batch_layout_mode = batch_layout_cols[0].radio(
            "Saved histogram layout",
            options=["Overlay", "Stacked"],
            horizontal=True,
            key="tab9_marker_batch_hist_layout",
        )
        batch_legend_outside = batch_layout_cols[1].checkbox(
            "Legend outside right",
            value=True,
            key="tab9_marker_batch_hist_legend_outside",
            disabled=batch_layout_mode == "Stacked",
        )
        batch_fig_size_cols = st.columns(3)
        batch_fig_width = batch_fig_size_cols[0].number_input("Saved figure width (in)", min_value=4.0, max_value=20.0, value=10.0, step=0.5, key="tab9_marker_batch_hist_width")
        batch_fig_height = batch_fig_size_cols[1].number_input("Saved figure height (in)", min_value=3.0, max_value=20.0, value=4.8, step=0.5, key="tab9_marker_batch_hist_height")
        batch_fig_dpi = int(batch_fig_size_cols[2].number_input("Saved figure DPI", min_value=72, max_value=1200, value=300, step=25, key="tab9_marker_batch_hist_dpi"))
        batch_save_separate = st.checkbox(
            "Also save separate figure per marker",
            value=True,
            key="tab9_marker_batch_hist_save_separate",
        )
        batch_histogram_figure_options = HistogramFigureOptions(
            layout_mode=batch_layout_mode,
            show_bars=bool(batch_show_bars),
            show_curve=bool(batch_show_curve),
            density=bool(batch_density),
            legend_outside=bool(batch_legend_outside and batch_layout_mode == "Overlay"),
            width_inches=float(batch_fig_width),
            height_inches=float(batch_fig_height),
            dpi=int(batch_fig_dpi),
            save_separate_markers=bool(batch_save_separate),
        )
        batch_exec_cols = st.columns(3)
        run_parallel = batch_exec_cols[0].checkbox("Process in parallel", value=True, key="tab9_marker_batch_parallel")
        max_workers_default = min(max(len(selected_images), 1), max(1, min(8, os.cpu_count() or 4)))
        parallel_workers = int(batch_exec_cols[1].number_input("Parallel workers", min_value=1, max_value=max(1, os.cpu_count() or 8), value=max_workers_default, step=1, key="tab9_marker_batch_workers"))
        compiled_sample_cap = int(batch_exec_cols[2].number_input(
            "Compiled sample cap",
            min_value=10_000,
            max_value=2_000_000,
            value=DEFAULT_BATCH_COMPILED_SAMPLE_CAP,
            step=10_000,
            key="tab9_marker_batch_compiled_sample_cap",
            help="Per image-marker-region weighted sample used for combined cross-image marker summaries and comparison histograms.",
        ))
        build_combined_report = st.checkbox(
            "Write combined cross-image marker report",
            value=True,
            key="tab9_marker_batch_build_combined",
            help="Creates one root-level compiled marker CSV plus cross-image marker histograms across all selected images.",
        )

    with preview_col:
        preview_image, full_shape = _build_preview_pil(active_path, channel_names, channel_colors, preview_mode, preview_markers)
        if full_shape[0] <= 0 or full_shape[1] <= 0:
            full_shape = _get_image_shape_cached(str(active_path), active_path.stat().st_mtime)

        shared_drawn_regions: list[RegionSpec] = []
        preview_regions: list[RegionSpec] = []

        if batch_region_mode == "Shared Drawn Polygons":
            shared_drawn_regions = _render_canvas_editor("tab9_marker_batch", active_image, preview_image, full_shape)
            preview_regions = shared_drawn_regions
        elif batch_region_mode == "Matched GeoJSON":
            matched_name = _matched_geojson_name(active_image, _list_geojson_files(geojson_dir))
            if matched_name is not None:
                preview_regions = _load_regions_from_geojson(Path(geojson_dir) / matched_name)
            overlay = _overlay_regions_on_preview(preview_image, preview_regions, full_shape)
            st.image(overlay, caption=f"Preview: {active_image}")
        elif batch_region_mode == "Shared GeoJSON":
            if shared_geojson_path is not None:
                preview_regions = _load_regions_from_geojson(shared_geojson_path)
            overlay = _overlay_regions_on_preview(preview_image, preview_regions, full_shape)
            st.image(overlay, caption=f"Preview: {active_image}")
        else:
            st.image(preview_image, caption=f"Preview: {active_image}")

        if preview_regions:
            st.caption(f"Active preview contains {len(preview_regions)} region(s).")

        if st.button("Export Reports For Selected Images", type="primary", key="tab9_marker_batch_export"):
            if not selected_images:
                st.warning("Select at least one image for batch export.")
            elif not selected_markers:
                st.warning("Select at least one marker for batch export.")
            else:
                output_root = Path(output_dir).expanduser()
                output_root.mkdir(parents=True, exist_ok=True)
                manifest_rows = []
                combined_summary_frames: list[pd.DataFrame] = []
                combined_histogram_frames: list[pd.DataFrame] = []
                compiled_chunks: list[BatchCompiledChunk] = []
                with st.status("Running batch marker analysis...", expanded=True) as status:
                    job_specs: list[tuple[str, Path, list[RegionSpec]]] = []
                    for image_name in selected_images:
                        image_path = Path(image_dir) / image_name
                        try:
                            image_shape = _get_image_shape_cached(str(image_path), image_path.stat().st_mtime)
                            regions = _resolve_batch_regions(
                                mode=batch_region_mode,
                                image_name=image_name,
                                full_shape=image_shape,
                                geojson_dir=geojson_dir,
                                shared_geojson_path=shared_geojson_path,
                                shared_drawn_regions=shared_drawn_regions,
                            )
                            job_specs.append((image_name, image_path, regions))
                        except Exception as exc:
                            manifest_rows.append(
                                {
                                    "Image": image_name,
                                    "Status": f"Failed: {exc}",
                                    "Markers Requested": len(selected_markers),
                                    "Summary Rows": 0,
                                    "Histogram Rows": 0,
                                    "Raw Files": 0,
                                    "Output Directory": str(output_root / normalize_image_stem(image_path)),
                                }
                            )

                    status.write(
                        f"Prepared {len(job_specs)} image job(s). "
                        f"Execution: {'parallel' if run_parallel and len(job_specs) > 1 else 'serial'} with {max(1, min(parallel_workers, len(job_specs) if job_specs else 1))} worker(s)."
                    )

                    def _consume_result(result: BatchImageJobResult) -> None:
                        combined_summary_frames.append(result.summary_df)
                        combined_histogram_frames.append(result.histogram_df)
                        compiled_chunks.extend(result.compiled_chunks)
                        manifest_rows.append(
                            {
                                "Image": result.image_name,
                                "Status": "Success",
                                "Markers Requested": len(selected_markers),
                                "Summary Rows": len(result.summary_df),
                                "Histogram Rows": len(result.histogram_df),
                                "Raw Files": len(result.raw_paths),
                                "Output Directory": str(output_root / normalize_image_stem(result.image_name)),
                            }
                        )

                    if run_parallel and len(job_specs) > 1:
                        worker_count = max(1, min(int(parallel_workers), len(job_specs)))
                        with ThreadPoolExecutor(max_workers=worker_count) as executor:
                            future_map = {
                                executor.submit(
                                    _run_batch_image_job,
                                    image_path,
                                    output_root,
                                    selected_markers,
                                    regions,
                                    transform,
                                    float(cofactor),
                                    int(bins),
                                    float(smooth_sigma),
                                    bool(export_raw),
                                    int(raw_pixel_cap),
                                    batch_histogram_figure_options,
                                    int(compiled_sample_cap) if build_combined_report else 0,
                                ): image_name
                                for image_name, image_path, regions in job_specs
                            }
                            for future in as_completed(future_map):
                                image_name = future_map[future]
                                status.write(f"Completed {image_name}")
                                try:
                                    _consume_result(future.result())
                                except Exception as exc:
                                    manifest_rows.append(
                                        {
                                            "Image": image_name,
                                            "Status": f"Failed: {exc}",
                                            "Markers Requested": len(selected_markers),
                                            "Summary Rows": 0,
                                            "Histogram Rows": 0,
                                            "Raw Files": 0,
                                            "Output Directory": str(output_root / normalize_image_stem(image_name)),
                                        }
                                    )
                    else:
                        for image_name, image_path, regions in job_specs:
                            status.write(f"Processing {image_name}...")
                            try:
                                result = _run_batch_image_job(
                                    image_path=image_path,
                                    output_root=output_root,
                                    selected_marker_names=selected_markers,
                                    region_specs=regions,
                                    transform=transform,
                                    cofactor=float(cofactor),
                                    bins=int(bins),
                                    smooth_sigma=float(smooth_sigma),
                                    export_raw=bool(export_raw),
                                    raw_pixel_cap=int(raw_pixel_cap),
                                    figure_options=batch_histogram_figure_options,
                                    compiled_sample_cap=int(compiled_sample_cap) if build_combined_report else 0,
                                )
                                _consume_result(result)
                            except Exception as exc:
                                manifest_rows.append(
                                    {
                                        "Image": image_name,
                                        "Status": f"Failed: {exc}",
                                        "Markers Requested": len(selected_markers),
                                        "Summary Rows": 0,
                                        "Histogram Rows": 0,
                                        "Raw Files": 0,
                                        "Output Directory": str(output_root / normalize_image_stem(image_name)),
                                    }
                                )

                    if build_combined_report and compiled_chunks:
                        combined_dir = output_root / "combined"
                        combined_dir.mkdir(parents=True, exist_ok=True)
                        if combined_summary_frames:
                            all_summary_df = pd.concat(combined_summary_frames, ignore_index=True)
                            all_summary_path = combined_dir / "all_image_marker_summary.csv"
                            all_summary_df.to_csv(all_summary_path, index=False)
                            status.write(f"All-image summary: {all_summary_path}")
                        if combined_histogram_frames:
                            all_hist_df = pd.concat(combined_histogram_frames, ignore_index=True)
                            all_hist_path = combined_dir / "all_image_marker_histograms.csv"
                            all_hist_df.to_csv(all_hist_path, index=False)
                            status.write(f"All-image hist rows: {all_hist_path}")

                        compiled_summary_df = _aggregate_compiled_chunks(compiled_chunks)
                        compiled_summary_path = combined_dir / "compiled_marker_summary.csv"
                        compiled_summary_df.to_csv(compiled_summary_path, index=False)
                        status.write(f"Compiled marker summary: {compiled_summary_path}")

                        cross_image_hist_df = _build_cross_image_histogram_df(
                            chunks=compiled_chunks,
                            transform=transform,
                            cofactor=float(cofactor),
                            bins=int(bins),
                            smooth_sigma=float(smooth_sigma),
                        )
                        cross_hist_path = combined_dir / "cross_image_marker_histograms.csv"
                        cross_image_hist_df.to_csv(cross_hist_path, index=False)
                        status.write(f"Cross-image histogram rows: {cross_hist_path}")

                        saved_cross_figures = _export_cross_image_histogram_figures(
                            histogram_df=cross_image_hist_df,
                            output_dir=combined_dir / "figures",
                            transform=transform,
                            cofactor=float(cofactor),
                            figure_options=batch_histogram_figure_options,
                        )
                        status.write(f"Cross-image histogram figures: {len(saved_cross_figures)} file(s)")

                    manifest = pd.DataFrame(manifest_rows)
                    manifest_path = output_root / "batch_manifest.csv"
                    manifest.to_csv(manifest_path, index=False)
                    status.write(f"Batch manifest: {manifest_path}")
                    status.update(label="Batch marker analysis complete", state="complete")
                st.success(f"Batch reports exported to {output_root}")
                st.dataframe(pd.DataFrame(manifest_rows), hide_index=True)


def render_marker_analysis(project_root: Path) -> None:
    st.markdown("### Marker Analysis")
    st.markdown(
        "Inspect raw image-marker intensities from OME-TIFF and QPTIFF inputs, including histogram QC, dynamic range summaries, ROI-restricted exports, and per-image batch reports."
    )

    tab_single, tab_batch = st.tabs(["Single Image", "Multiple Images"])
    with tab_single:
        _render_single_image(project_root)
    with tab_batch:
        _render_batch(project_root)
