"""Unsupervised tissue macro-structure classification and spatial annotation.

Backend for the SPACEMAP Annotation tab.  Every function here is pure Python /
NumPy so that the Streamlit layer stays a thin controller: it collects
parameters, calls one of the stage functions below, and renders the returned
arrays, data frames and matplotlib figures.

Pipeline stages (mirrors ``notebooks/new/tissue_annotation_v2.ipynb``):

1. :func:`detect_tissue_mask`      - virtual H&E optical-density tissue masking.
2. :func:`detect_artifacts`        - out-of-focus, flat-bright and filamentous
   lint detection, subtracted from the reviewed tissue mask.
3. :func:`build_feature_stack` + :func:`run_pixel_clustering` +
   :func:`build_cluster_profiles` + :func:`export_cluster_partition` - pixel
   classification of the reviewed valid tissue.
4. :func:`export_domain_partition` - merge clusters into biological domains and
   export the final gapless MultiPolygon GeoJSON.

All heavy image reads go through :class:`PyramidImageReader`, which reads the
requested OME-TIFF pyramid level directly instead of decoding full resolution
and striding.  That is both faster and anti-aliased (block averaged).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import tifffile

import rasterio.features
from shapely.affinity import scale as scale_geometry
from shapely.affinity import translate as translate_geometry
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, remove_small_holes, skeletonize
from sklearn.cluster import MiniBatchKMeans

import matplotlib.colors as mcolors

__all__ = [
    "HE_CHANNEL_PATTERNS",
    "PyramidImageReader",
    "DEFAULT_TISSUE_CONFIG",
    "DEFAULT_ARTIFACT_CONFIG",
    "DEFAULT_CLUSTERING_CONFIG",
    "DEFAULT_POOLING_CONFIG",
    "is_he_channel",
    "split_he_and_codex_channels",
    "detect_tissue_mask",
    "detect_blood_clot",
    "detect_artifacts",
    "load_geojson_mask",
    "mask_to_geojson_features",
    "region_features_from_labels",
    "write_feature_collection",
    "write_artifact_manifest",
    "ARTIFACT_REGION_SCHEMA",
    "ARTIFACT_MANIFEST_SCHEMA",
    "read_geojson_class_names",
    "build_feature_stack",
    "run_pixel_clustering",
    "build_cluster_profiles",
    "export_cluster_partition",
    "export_domain_partition",
    "pool_geojson_classes",
    "generate_zone_features",
    "qupath_color_from_hex",
    "assign_distinct_colors",
    "deterministic_class_hex_color",
    "DISTINCT_PALETTE",
    "build_cluster_palette",
]


# ==========================================================================
# Constants and small helpers
# ==========================================================================

HE_CHANNEL_PATTERNS: tuple[str, ...] = ("he_", "he ", "h&e", "he_red", "he_green", "he_blue")

OME_NAMESPACE = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

_NEUTRAL_CLASS_NAMES = {"low_signal_tissue", "background", "non-tissue", "unassigned"}


def is_he_channel(name: str, patterns: Sequence[str] = HE_CHANNEL_PATTERNS) -> bool:
    """Return True when a channel name looks like a virtual H&E layer."""
    lowered = str(name).strip().lower()
    return any(lowered.startswith(pattern) or lowered == pattern.strip("_") for pattern in patterns)


def split_he_and_codex_channels(
    channel_names: Sequence[str],
    patterns: Sequence[str] = HE_CHANNEL_PATTERNS,
) -> tuple[list[int], list[int]]:
    """Split channel indices into (H&E indices, CODEX indices)."""
    he_indices = [index for index, name in enumerate(channel_names) if is_he_channel(name, patterns)]
    codex_indices = [index for index in range(len(channel_names)) if index not in he_indices]
    return he_indices, codex_indices


def qupath_color_from_hex(hex_str: str) -> int:
    """Convert ``#RRGGBB`` to the signed ARGB integer QuPath stores.

    QuPath packs a classification colour as ARGB and reads the top byte as the
    alpha, so the opaque byte has to be set. Leaving it clear gives a fully
    transparent colour, and the annotation then draws with no colour at all.
    That went unnoticed for a long time because it only bites half the palette:
    a colour whose red channel is 0x80 or above already wrapped to a negative
    number whose top byte happened to be 0xff, so yellow came out right while
    green came out invisible.
    """
    rgb_int = int(str(hex_str).lstrip("#"), 16) & 0xFFFFFF
    return (0xFF000000 | rgb_int) - 0x100000000


# Maximally distinct qualitative palette (Trubetskoy's 20-colour set, reordered
# so the first entries are as far apart as possible). Used for tissue domains so
# that neighbouring annotations never come out in near-identical hues.
DISTINCT_PALETTE: tuple[str, ...] = (
    "#e6194b",  # red
    "#4363d8",  # blue
    "#3cb44b",  # green
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#469990",  # teal
    "#9a6324",  # brown
    "#ffe119",  # yellow
    "#000075",  # navy
    "#800000",  # maroon
    "#fabed4",  # pink
    "#aaffc3",  # mint
    "#808000",  # olive
    "#dcbeff",  # lavender
    "#ffd8b1",  # apricot
    "#a9a9a9",  # grey
    "#fffac8",  # beige
)


def assign_distinct_colors(
    names: Sequence[str],
    existing: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Give every name a visually distinct colour, honouring existing choices.

    Colours already chosen (for example with the colour picker) are kept and
    removed from the available pool, so the remaining names still get hues that
    are far from each other.  Once the palette is exhausted the deterministic
    hash colour is used as an overflow.
    """
    assigned = {str(name): str(color) for name, color in (existing or {}).items() if color}
    used = {color.lower() for color in assigned.values()}
    available = [color for color in DISTINCT_PALETTE if color.lower() not in used]

    result: dict[str, str] = {}
    for name in names:
        name = str(name)
        if name in assigned:
            result[name] = assigned[name]
            continue
        if str(name).strip().lower() in _NEUTRAL_CLASS_NAMES:
            result[name] = "#888888"
            continue
        if available:
            result[name] = available.pop(0)
        else:
            result[name] = deterministic_class_hex_color(name, seed)
        used.add(result[name].lower())
    return result


def deterministic_class_hex_color(name: str, seed: int = 42) -> str:
    """Return a stable, high-saturation hex color for a class name."""
    if str(name).strip().lower() in _NEUTRAL_CLASS_NAMES:
        return "#888888"
    hash_bytes = hashlib.md5(f"{seed}_{name}".encode("utf-8")).digest()
    hash_int = int.from_bytes(hash_bytes[:4], "big")
    hue = (hash_int % 360) / 360.0
    saturation = 0.75 + 0.25 * ((hash_int >> 8) % 100) / 100.0
    value = 0.85 + 0.15 * ((hash_int >> 16) % 100) / 100.0
    return mcolors.to_hex(mcolors.hsv_to_rgb((hue, saturation, value)))


def build_cluster_palette(n_clusters: int, include_low_signal: bool = False) -> list[str]:
    """High-contrast qualitative palette so adjacent cluster IDs stay distinct."""
    import seaborn as sns

    if n_clusters <= 20:
        base_palette = sns.color_palette("tab20", 20)
        interleave_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        palette = [base_palette[index] for index in interleave_order[:n_clusters]]
    else:
        palette = sns.color_palette("gist_ncar", n_clusters)
    colors = [mcolors.to_hex(color) for color in palette]
    if include_low_signal:
        colors.append("#888888")
    return colors


def _gaussian_kernel_size(sigma: float) -> tuple[int, int]:
    radius = int(4 * sigma + 0.5)
    return (2 * radius + 1, 2 * radius + 1)


def _remove_small_components(mask: np.ndarray, min_area_px: int, connectivity: int = 4) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    keep = np.zeros(count, dtype=bool)
    keep[np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area_px)[0] + 1] = True
    return keep[labels]


# ==========================================================================
# Focus measurement helpers (shared by the tissue pass and the artifact step)
# ==========================================================================

# Tile size and working resolution of the per-marker focus grid. These are an
# implementation detail of marker attribution rather than a user setting: the
# grid only has to be fine enough to say *where* in a region a marker lost
# focus, and coarse enough that 40 channels of it cost nothing to keep.
ATTRIBUTION_BIN_UM = 32.0
ATTRIBUTION_MAX_SIDE_PX = 3000

# Lengths are configured in microns so that the same settings mean the same
# thing at any ``blur_downsample``. Configs saved before that switch carry only
# the pixel-valued key, which is converted with the pixel size in force at the
# time, so an existing project keeps behaving exactly as it did.
_LEGACY_PIXEL_SCALE_KEYS: dict[str, str] = {
    "focus_detail_um": "laplacian_ksize",
    "focus_window_um": "window_size",
    "noise_smoothing_um": "gauss_pre_sigma",
    "region_smoothing_um": "gauss_post_sigma",
    "fill_holes_um": "morph_close_radius",
    "remove_specks_um": "morph_open_radius",
    "bright_fill_holes_um": "bright_morph_close_radius",
    "bright_remove_specks_um": "bright_morph_open_radius",
    "lint_join_gaps_um": "lint_morph_close_radius",
    "lint_remove_specks_um": "lint_morph_open_radius",
}


def _odd_kernel(value: float, minimum: int = 1, maximum: int = 999) -> int:
    """Nearest odd integer within ``[minimum, maximum]``.

    OpenCV rejects even Laplacian and box-filter kernel sizes. A config restored
    from an older project, or hand-edited on disk, can easily hold one, and the
    resulting exception is opaque; rounding to odd here is the cheap fix.
    """
    size = int(round(float(value)))
    if size % 2 == 0:
        size += 1
    return int(np.clip(size, int(minimum), int(maximum)))


def _disk_kernel(radius_px: float) -> np.ndarray:
    return disk(int(max(round(float(radius_px)), 1))).astype(np.uint8)


def _scale_um(settings: dict[str, Any], key: str, pixel_size_um: float) -> float:
    """Length in microns for one scale parameter, honouring legacy pixel keys."""
    value = settings.get(key)
    if value is not None:
        return float(value)
    legacy_key = _LEGACY_PIXEL_SCALE_KEYS.get(key)
    if legacy_key is not None and settings.get(legacy_key) is not None:
        return float(settings[legacy_key]) * float(pixel_size_um)
    return float(DEFAULT_ARTIFACT_CONFIG.get(key, 0.0))


def _focus_map(
    image: np.ndarray,
    *,
    pre_sigma_px: float,
    ksize: int,
    window_px: int,
    post_sigma_px: float,
) -> np.ndarray:
    """Per-pixel sharpness: the local variance of the Laplacian, contrast-normalized.

    The Laplacian responds to edges, so its local variance is high where detail
    is crisp and low where defocus has smeared it out. Dividing by the local
    mean squared makes the score independent of exposure, so a dim sharp field
    and a bright sharp field score alike; both inputs are on a 0-255 scale,
    which is what makes the ``+ 1`` floor a meaningful one.

    The score is only ever compared against another score computed the same way
    on the same image, never against an absolute number, because its magnitude
    depends on the content as much as on the focus.
    """
    source = np.asarray(image, dtype=np.float32)
    working = source
    if pre_sigma_px > 0:
        working = cv2.GaussianBlur(
            source, _gaussian_kernel_size(pre_sigma_px), float(pre_sigma_px), borderType=cv2.BORDER_REFLECT
        )
    laplacian = cv2.Laplacian(working, cv2.CV_32F, ksize=_odd_kernel(ksize, 1, 31))
    window = (int(window_px), int(window_px))
    mean_square = cv2.boxFilter(laplacian * laplacian, ddepth=-1, ksize=window, borderType=cv2.BORDER_REFLECT)
    mean = cv2.boxFilter(laplacian, ddepth=-1, ksize=window, borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(mean_square - mean * mean, 0.0)
    local_mean = cv2.boxFilter(source, ddepth=-1, ksize=window, borderType=cv2.BORDER_REFLECT)
    focus = variance / (local_mean * local_mean + 1.0)
    if post_sigma_px > 0:
        focus = cv2.GaussianBlur(
            focus, _gaussian_kernel_size(post_sigma_px), float(post_sigma_px), borderType=cv2.BORDER_REFLECT
        )
    return focus


def _attribution_grid_geometry(shape: tuple[int, int], pixel_size_um: float) -> tuple[tuple[int, int], int]:
    """Tile-grid shape and the decimation used to fill it."""
    height, width = int(shape[0]), int(shape[1])
    bin_px = max(int(round(ATTRIBUTION_BIN_UM / max(float(pixel_size_um), 1e-6))), 1)
    grid_shape = (max(height // bin_px, 1), max(width // bin_px, 1))
    decimate = max(int(np.ceil(max(height, width) / float(ATTRIBUTION_MAX_SIDE_PX))), 1)
    return grid_shape, decimate


def _channel_focus_tiles(
    channel: np.ndarray,
    grid_shape: tuple[int, int],
    *,
    pre_sigma_px: float,
    ksize: int,
    decimate: int,
) -> np.ndarray:
    """Tile-averaged sharpness for one marker channel.

    Only the *relative* sharpness of a channel against its own sharp tiles is
    used downstream, so this runs on a decimated copy. A Laplacian at full
    working resolution on every channel of a 40-plex ROI would dominate the
    runtime of the whole step, and decimation removes the same high frequencies
    from every channel, which leaves that comparison intact.

    The mean of the squared Laplacian stands in for its variance here: over a
    tile the Laplacian averages to approximately zero, so the two agree, and
    this way costs one pass instead of two.
    """
    grid_height, grid_width = int(grid_shape[0]), int(grid_shape[1])
    working = channel
    sigma = float(pre_sigma_px)
    if decimate > 1:
        target = (max(channel.shape[1] // decimate, grid_width), max(channel.shape[0] // decimate, grid_height))
        working = cv2.resize(channel, target, interpolation=cv2.INTER_AREA)
        sigma = sigma / float(decimate)
    working = np.asarray(working, dtype=np.float32)
    if sigma > 0:
        working = cv2.GaussianBlur(
            working, _gaussian_kernel_size(sigma), sigma, borderType=cv2.BORDER_REFLECT
        )
    laplacian = cv2.Laplacian(working, cv2.CV_32F, ksize=_odd_kernel(ksize, 1, 31))
    np.multiply(laplacian, laplacian, out=laplacian)
    energy = cv2.resize(laplacian, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
    intensity = cv2.resize(working, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
    return (energy / (intensity * intensity + 1.0)).astype(np.float32)


# ==========================================================================
# Pyramid-aware OME-TIFF reader
# ==========================================================================

_LAYOUT_CHANNEL_SERIES = "channel_series"   # series index = channel * n_levels + level
_LAYOUT_SINGLE_SERIES = "single_series"     # one multi-channel series with .levels
_LAYOUT_PAGES = "pages"                     # flat pages, strided reads only


class PyramidImageReader:
    """Read OME-TIFF channels at a requested downsample factor.

    The reader inspects the file header once and then serves reads from the
    matching pyramid level, which avoids decoding full resolution data for a
    2x/4x/8x view.  Three on-disk layouts are handled:

    ``channel_series``
        Each channel is written as its own series followed by its pyramid
        levels (the layout produced by the SPACEMAP registration/ROI writers
        and assumed by the reference notebook).
    ``single_series``
        One multi-channel series exposing ``series.levels``.
    ``pages``
        No usable pyramid; full-resolution pages are read and strided.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        with tifffile.TiffFile(self.path) as tif:
            self.channel_names = self._read_channel_names(tif)
            self.n_channels = len(self.channel_names)
            self._detect_layout(tif)

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _read_channel_names(tif: tifffile.TiffFile) -> list[str]:
        ome_xml = getattr(tif, "ome_metadata", None)
        if ome_xml:
            try:
                root = ET.fromstring(ome_xml)
                names = [
                    channel.get("Name") or f"Channel_{index}"
                    for index, channel in enumerate(root.findall(".//ome:Channel", OME_NAMESPACE))
                ]
                if names:
                    return names
            except ET.ParseError:
                pass
        # Fall back to one name per full-resolution page.
        page_count = len(tif.pages)
        return [f"Channel_{index}" for index in range(page_count)]

    @staticmethod
    def _spatial_shape(shape_value, axes: str | None) -> tuple[int, int]:
        axes = axes or ""
        if "Y" in axes and "X" in axes:
            return int(shape_value[axes.index("Y")]), int(shape_value[axes.index("X")])
        if len(shape_value) >= 2:
            return int(shape_value[-2]), int(shape_value[-1])
        return 0, 0

    def _detect_layout(self, tif: tifffile.TiffFile) -> None:
        n_series = len(tif.series)
        self._levels_per_channel = 1
        self._channel_axis = None
        self._series_axes = ""

        if self.n_channels > 0 and n_series >= self.n_channels and n_series % self.n_channels == 0:
            levels = n_series // self.n_channels
            if levels > 1:
                shapes = [self._spatial_shape(tif.series[i].shape, getattr(tif.series[i], "axes", "")) for i in range(levels)]
                decreasing = all(shapes[i + 1][0] < shapes[i][0] for i in range(levels - 1))
                if decreasing:
                    self.layout = _LAYOUT_CHANNEL_SERIES
                    self._levels_per_channel = levels
                    self._level_shapes = shapes
                    self.full_shape = shapes[0]
                    return

        primary = max(tif.series, key=lambda series: int(np.prod(getattr(series, "shape", (0,))))) if n_series else None
        if primary is not None:
            axes = getattr(primary, "axes", "") or ""
            channel_axis = next((axes.index(name) for name in ("C", "I") if name in axes), None)
            if channel_axis is not None and len(primary.shape) >= 3:
                levels = list(getattr(primary, "levels", []) or [primary])
                self.layout = _LAYOUT_SINGLE_SERIES
                self._series_axes = axes
                self._channel_axis = channel_axis
                self._levels_per_channel = len(levels)
                self._level_shapes = [self._spatial_shape(level.shape, getattr(level, "axes", axes)) for level in levels]
                self.full_shape = self._level_shapes[0]
                return

        self.layout = _LAYOUT_PAGES
        page = tif.pages[0]
        self._level_shapes = [self._spatial_shape(page.shape, getattr(page, "axes", ""))]
        self.full_shape = self._level_shapes[0]

    # -- public API -----------------------------------------------------------

    @property
    def is_pyramidal(self) -> bool:
        return self._levels_per_channel > 1

    @property
    def level_count(self) -> int:
        return self._levels_per_channel

    @property
    def level_shapes(self) -> list[tuple[int, int]]:
        return list(self._level_shapes)

    def target_shape(self, factor: int) -> tuple[int, int]:
        """Shape a channel read at ``factor`` will return."""
        factor = max(int(factor), 1)
        level_index = self._level_index_for(factor)
        level_height, level_width = self._level_shapes[level_index]
        level_factor = 2 ** level_index
        if level_factor == factor:
            return level_height, level_width
        # Non-power-of-two requests (notably QuPath's default 5x background
        # threshold downsample) must not silently fall back to the 4x pyramid.
        full_height, full_width = self.full_shape
        return (
            max(int(round(full_height / factor)), 1),
            max(int(round(full_width / factor)), 1),
        )

    def _level_index_for(self, factor: int) -> int:
        if self._levels_per_channel <= 1 or factor <= 1:
            return 0
        wanted = int(np.floor(np.log2(factor)))
        return int(min(max(wanted, 0), self._levels_per_channel - 1))

    @contextmanager
    def open(self):
        """Open the file once for a multi-channel read loop."""
        with tifffile.TiffFile(self.path) as tif:
            yield tif

    def read_channel(self, tif: tifffile.TiffFile, channel_index: int, factor: int) -> np.ndarray:
        """Read one channel at ``factor`` downsampling as ``float32``."""
        factor = max(int(factor), 1)
        level_index = self._level_index_for(factor)
        level_factor = 2 ** level_index

        if self.layout == _LAYOUT_CHANNEL_SERIES:
            series_index = channel_index * self._levels_per_channel + level_index
            data = np.asarray(tif.series[series_index].asarray())
        elif self.layout == _LAYOUT_SINGLE_SERIES:
            primary = max(tif.series, key=lambda series: int(np.prod(getattr(series, "shape", (0,)))))
            levels = list(getattr(primary, "levels", []) or [primary])
            level = levels[min(level_index, len(levels) - 1)]
            selector: list[Any] = []
            for axis_position, axis_name in enumerate(self._series_axes):
                if axis_name in ("Y", "X"):
                    selector.append(slice(None))
                elif axis_position == self._channel_axis:
                    selector.append(channel_index)
                else:
                    selector.append(0)
            # Read through Zarr so only this channel's chunks are decoded.
            # `level.asarray()[selector]` would decode every channel of the level
            # on every call, turning a C-channel read loop into O(C^2) work.
            try:
                import zarr

                store = zarr.open(level.aszarr(), mode="r")
                if not hasattr(store, "shape"):
                    store = store["0"] if "0" in store else store[list(store.array_keys())[0]]
                data = np.asarray(store[tuple(selector)])
            except Exception:
                data = np.asarray(level.asarray()[tuple(selector)])
        else:
            page_index = min(channel_index, len(tif.pages) - 1)
            data = np.asarray(tif.pages[page_index].asarray())
            if data.ndim == 3:
                data = data[channel_index] if data.shape[0] > channel_index else data[..., channel_index]
            level_factor = 1

        data = np.squeeze(data)
        if data.ndim != 2:
            data = data.reshape(data.shape[-2], data.shape[-1])

        if level_factor != factor:
            target_h, target_w = self.target_shape(factor)
            if data.shape != (target_h, target_w):
                data = cv2.resize(
                    data.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA
                )
        return np.ascontiguousarray(data, dtype=np.float32)

    def read_channels(self, channel_indices: Sequence[int], factor: int, progress=None) -> np.ndarray:
        """Read several channels into one ``(C, H, W)`` float32 stack."""
        height, width = self.target_shape(factor)
        stack = np.empty((len(channel_indices), height, width), dtype=np.float32)
        with self.open() as tif:
            for position, channel_index in enumerate(channel_indices):
                plane = self.read_channel(tif, channel_index, factor)
                if plane.shape != (height, width):
                    plane = cv2.resize(plane, (width, height), interpolation=cv2.INTER_AREA)
                stack[position] = plane
                if progress is not None:
                    progress(position + 1, len(channel_indices), self.channel_names[channel_index])
        return stack


# ==========================================================================
# GeoJSON helpers
# ==========================================================================

def _polygons_from_mask(mask: np.ndarray, scale_factor: float, simplify_px: float = 2.0) -> list:
    """Vectorize a binary mask into shapely polygons with holes preserved."""
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list = []
    if hierarchy is None:
        return polygons
    hierarchy = hierarchy[0]
    contour_index = 0
    while contour_index >= 0:
        exterior = cv2.approxPolyDP(contours[contour_index], simplify_px, closed=True).reshape(-1, 2)
        if len(exterior) >= 3:
            exterior_coords = [(float(point[0]) * scale_factor, float(point[1]) * scale_factor) for point in exterior]
            holes = []
            child = hierarchy[contour_index][2]
            while child >= 0:
                hole = cv2.approxPolyDP(contours[child], simplify_px, closed=True).reshape(-1, 2)
                if len(hole) >= 3:
                    holes.append([(float(point[0]) * scale_factor, float(point[1]) * scale_factor) for point in hole])
                child = hierarchy[child][0]
            try:
                polygon = make_valid(Polygon(exterior_coords, holes))
                if not polygon.is_empty:
                    polygons.append(polygon)
            except Exception:
                pass
        contour_index = hierarchy[contour_index][0]
    return polygons


def mask_to_geojson_features(
    mask: np.ndarray,
    scale_factor: float,
    class_name: str,
    color_rgb: int | None = None,
    simplify_px: float = 2.0,
    merge: bool = True,
    locked: bool = False,
    extra_properties: dict | None = None,
) -> list[dict]:
    """Vectorize a binary mask into QuPath-compatible GeoJSON features."""
    polygons = _polygons_from_mask(mask, scale_factor, simplify_px=simplify_px)
    if not polygons:
        return []
    if color_rgb is None:
        color_rgb = qupath_color_from_hex(deterministic_class_hex_color(class_name))

    def _feature(geometry) -> dict:
        properties = {
            "objectType": "annotation",
            "classification": {"name": class_name, "colorRGB": int(color_rgb)},
            "name": class_name,
            "isLocked": bool(locked),
        }
        if extra_properties:
            properties.update(extra_properties)
        return {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": mapping(geometry),
            "properties": properties,
        }

    if merge:
        merged = make_valid(unary_union(polygons))
        if merged.is_empty:
            return []
        return [_feature(merged)]
    return [_feature(polygon) for polygon in polygons if not polygon.is_empty]


def write_feature_collection(features: Sequence[dict], output_path: str | os.PathLike) -> float:
    """Write a FeatureCollection and return its size in MB."""
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump({"type": "FeatureCollection", "features": list(features)}, file_handle, separators=(",", ":"))
    return os.path.getsize(output_path) / (1024 ** 2)


# ==========================================================================
# Per-region artifact export (QuPath review and automated review)
# ==========================================================================

ARTIFACT_REGION_SCHEMA = "spacemap.artifact-region/1"
ARTIFACT_MANIFEST_SCHEMA = "spacemap.artifact-manifest/1"


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars and arrays into plain JSON types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def region_features_from_labels(
    labels: np.ndarray,
    records: Sequence[dict[str, Any]],
    scale_factor: float,
    class_name: str,
    color_rgb: int | None = None,
    simplify_px: float = 2.0,
    locked: bool = False,
) -> list[dict[str, Any]]:
    """One GeoJSON feature per detected region, each carrying its own evidence.

    A single merged MultiPolygon cannot be reviewed region by region: accepting
    one blurry patch while rejecting another needs them to be separate objects,
    each holding the measurements and the marker evidence that produced it.

    Numeric evidence is written to ``properties.measurements``, which QuPath
    shows in its measurement table, and the full structured record to
    ``properties.spacemap`` for the automated review pass.
    """
    if color_rgb is None:
        color_rgb = qupath_color_from_hex(deterministic_class_hex_color(class_name))
    features: list[dict[str, Any]] = []
    for record in records:
        left, top, width, height = record["bbox_px"]
        roi = labels[top:top + height, left:left + width] == int(record["label"])
        polygons = _polygons_from_mask(roi, float(scale_factor), simplify_px=float(simplify_px))
        if not polygons:
            continue
        geometry = make_valid(unary_union(polygons))
        if geometry.is_empty:
            continue
        geometry = translate_geometry(
            geometry, xoff=float(left) * float(scale_factor), yoff=float(top) * float(scale_factor)
        )
        feature_id = str((record.get("spacemap") or {}).get("region_id") or uuid.uuid4())
        properties: dict[str, Any] = {
            "objectType": "annotation",
            "classification": {"name": class_name, "colorRGB": int(color_rgb)},
            "name": str(record.get("name", class_name)),
            "isLocked": bool(locked),
            "measurements": _json_safe(record.get("measurements") or {}),
        }
        if record.get("spacemap"):
            properties["spacemap"] = _json_safe(record["spacemap"])
        record["feature_id"] = feature_id
        features.append({
            "type": "Feature",
            "id": feature_id,
            "geometry": mapping(geometry),
            "properties": properties,
        })
    return features


def write_artifact_manifest(
    output_path: str | os.PathLike,
    *,
    sample_id: str,
    regions: dict[str, Sequence[dict[str, Any]]],
    exports: dict[str, str],
    stats: dict[str, Any],
    parameters: dict[str, Any],
    marker_names: Sequence[str],
    image_path: str | None = None,
) -> str:
    """Write one machine-readable index of every artifact region in a sample.

    The GeoJSON layers carry the same evidence inline, but a reviewer that only
    needs to decide accept-or-reject should not have to parse geometry to find
    it. This file lists every region once, with its measurements, its ranked
    candidate markers, and the layer and feature it corresponds to.
    """
    from datetime import datetime, timezone

    entries: list[dict[str, Any]] = []
    for artifact_type, records in regions.items():
        for record in records:
            block = dict(record.get("spacemap") or {})
            block["geojson_file"] = os.path.basename(exports.get(artifact_type, "")) or None
            block["feature_id"] = record.get("feature_id")
            block["measurements"] = record.get("measurements", {})
            entries.append(block)
    entries.sort(key=lambda entry: -float(entry.get("area_um2") or 0.0))

    payload = _json_safe({
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "sample_id": sample_id,
        "image_path": image_path,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "markers_used": list(marker_names),
        "parameters": parameters,
        "statistics": stats,
        "layers": exports,
        "region_fields": {
            "artifact_type": "blur, bright or lint.",
            "severity": "0-1, how far past the detection threshold the region sits. Blur only.",
            "confidence": "Fraction of the available independent tests that flagged this region. Blur only.",
            "candidate_markers": (
                "Markers ranked by the evidence that produced the call: lowest focus ratio first for "
                "blur, highest robust-Z first for bright and lint. A focus ratio of 1.0 means the "
                "marker is as sharp here as it is in its own sharpest tissue elsewhere."
            ),
            "review": (
                "Left as status 'pending' by detection. A reviewer sets status to 'reviewed' and "
                "decision to 'accept' (a true artifact, keep it excluded) or 'reject' (a false "
                "positive, return it to valid tissue)."
            ),
        },
        "regions": entries,
    })
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return output_path


def _iter_geojson_features(geojson_path: str | os.PathLike) -> list[dict]:
    with open(str(geojson_path), "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    features = data.get("features", [])
    if not features and data.get("type") == "Feature":
        features = [data]
    return features


def feature_class_name(feature: dict) -> str:
    """Return the QuPath classification name of one GeoJSON feature."""
    properties = feature.get("properties", {}) or {}
    classification = properties.get("classification")
    if isinstance(classification, dict):
        name = classification.get("name")
        if name:
            return str(name)
    if isinstance(classification, str) and classification:
        return classification
    return str(properties.get("name") or "Unclassified")


def read_geojson_class_names(geojson_path: str | os.PathLike) -> list[str]:
    """Return the sorted unique classification names present in a GeoJSON."""
    try:
        return sorted({feature_class_name(feature) for feature in _iter_geojson_features(geojson_path)})
    except Exception:
        return []


def load_geojson_mask(
    geojson_path: str | os.PathLike,
    out_shape: tuple[int, int],
    scale_factor: float,
    class_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Rasterize a QuPath GeoJSON onto a downsampled grid.

    ``scale_factor`` is the downsample factor of the target grid relative to
    full resolution, so full-resolution coordinates are divided by it.  This is
    always read fresh from disk, which is what lets a QuPath edit round-trip
    back into the pipeline.
    """
    wanted = {str(name).lower() for name in class_names} if class_names else None
    geometries = []
    for feature in _iter_geojson_features(geojson_path):
        if wanted is not None and feature_class_name(feature).lower() not in wanted:
            continue
        geometry_dict = feature.get("geometry")
        if not geometry_dict:
            continue
        try:
            geometry = shape(geometry_dict)
            if not geometry.is_empty:
                geometries.append(geometry if geometry.is_valid else make_valid(geometry))
        except Exception:
            continue

    mask = np.zeros(out_shape, dtype=bool)
    if not geometries:
        return mask

    merged = unary_union(geometries)
    parts = [merged] if merged.geom_type == "Polygon" else list(getattr(merged, "geoms", []))
    shapes_to_rasterize = []
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        exterior = [(x / scale_factor, y / scale_factor) for x, y in part.exterior.coords]
        interiors = [[(x / scale_factor, y / scale_factor) for x, y in ring.coords] for ring in part.interiors]
        try:
            shapes_to_rasterize.append((Polygon(exterior, interiors), 1))
        except Exception:
            continue
    if not shapes_to_rasterize:
        return mask
    return rasterio.features.rasterize(
        shapes_to_rasterize, out_shape=out_shape, fill=0, default_value=1, dtype=np.uint8
    ).astype(bool)


# ==========================================================================
# Stage 1 - H&E optical-density tissue masking
# ==========================================================================

DEFAULT_TISSUE_CONFIG: dict[str, Any] = {
    "blur_downsample": 2,
    "microns_per_pixel": 0.5083,
    "clamp_sigma": 4.0,
    "tissue_threshold": 10.0,
    "tissue_morph_radius": 11,
    "min_tissue_island_px": 5000,
    "he_white_percentile": 99.5,
    "he_od_threshold_factor": 0.85,
    "he_valid_min_intensity": 0.0,
    "tissue_mask_mode": "he_only",
    "codex_threshold_mode": "otsu",
    "he_codex_edge_rescue_um": 20.0,
    "tissue_border_exclusion_um": 20.0,
    "bright_breadth_channel_sigma": 3.0,
    "robust_signal_clip_z": 50.0,
    "boundary_simplify_px": 2.0,
    # Blood-clot detection (off by default; H&E only)
    "detect_blood_clot": False,
    "clot_eosin_reference_percentile": 25.0,
    "clot_eosin_ratio": 2.0,
    "clot_nuclear_percentile": 50.0,
    "clot_nuclei_reference_percentile": 75.0,
    "clot_max_nuclei_ratio": 0.25,
    "clot_rim_nuclei_ratio": 3.0,
    "clot_eosin_smooth_ratio": 0.75,
    "clot_texture_window_um": 40.0,
    "clot_morph_radius_um": 20.0,
    "clot_min_area_mm2": 0.05,
    "clot_exclude_from_tissue": True,
}

TISSUE_MASK_MODES = ("he_only", "he_with_codex_edge_rescue", "he_codex_union", "codex_only")

# Modes that do not use the virtual H&E optical density at all. Useful for
# images that were never given virtual H&E layers, where a marker projection
# (DAPI on its own works well) is the only way to find tissue.
CODEX_ONLY_MODES = ("codex_only",)


def _clamp_and_normalize(array: np.ndarray, sigma: float) -> np.ndarray:
    mean, std = float(array.mean()), float(array.std())
    array = np.clip(array, 0, mean + sigma * std)
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) / (high - low) * 255.0
    return array.astype(np.float32)


def _robust_channel_z(array: np.ndarray, clip_z: float, out: np.ndarray | None = None) -> np.ndarray:
    """Robust per-channel Z map, computed in place to limit peak memory.

    A 2x-downsampled 40-channel ROI plane is ~240 MB; doing this with temporaries
    allocates three of those per channel, which dominates the artifact step.
    """
    sample = array[::4, ::4]
    sample = sample[np.isfinite(sample) & (sample > 0)]
    if out is None:
        out = np.empty_like(array, dtype=np.float32)
    if sample.size == 0:
        out[:] = 0.0
        return out
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma < 1e-6:
        robust_sigma = max(float(sample.std()), 1.0)
    np.subtract(array, median, out=out)
    np.divide(out, robust_sigma, out=out)
    np.clip(out, 0.0, clip_z, out=out)
    return out


def detect_blood_clot(
    he_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    config: dict[str, Any] | None = None,
    microns_per_pixel: float = 0.5083,
    downsample: int = 1,
    sample_id: str = "",
) -> dict[str, Any]:
    """Find blood clot / haemorrhage from the virtual H&E alone.

    A clot is a dense eosinophilic mass with essentially no nuclei in it, so it
    shows three properties together that real tissue does not:

    * **eosin-rich** - clearly pinker than the pale end of the tissue on the
      same slide;
    * **nuclei-free** - almost none of its area is covered by nuclei;
    * **structureless** - the eosin itself is smooth, with none of the fibre
      striations that run through collagen and muscle.

    No one of these is enough. Eosin alone also selects muscle and dense
    collagen, which are legitimately pink. Absence of nuclei alone also selects
    lumen, fat and out-of-focus blanks. Smooth eosin alone selects anything
    homogeneous. Requiring all three is what separates a clot from tissue that
    merely stains pink.

    Every test is expressed against a reference measured on the same slide, and
    each has a meaningful zero point, so the detector can return nothing when
    there is no clot. Bare percentile cuts cannot: they always select their own
    tail, which on a slide with no clot means the most clot-like real tissue.

    Nuclei are measured as the *share of the area they cover*, not as the local
    variability of the haematoxylin channel. In virtual H&E the haematoxylin is
    frequently clipped to zero over a good part of the tissue, and a variability
    measure reads that clipping as "no nuclei here" when it only means the
    reconstruction ran out of range.

    Returns the mask, the intermediate maps for plotting, per-component
    measurements, and one record per accepted region carrying the evidence that
    produced it.
    """
    settings = {**DEFAULT_TISSUE_CONFIG, **(config or {})}
    empty = np.zeros(tissue_mask.shape, dtype=bool)
    result: dict[str, Any] = {
        "clot_mask": empty, "eosin": None, "nuclear_texture": None, "nuclear_coverage": None,
        "eosin_texture": None, "components": [], "regions": [], "region_labels": None,
        "stats": {"clot_percent": 0.0, "clot_pixels": 0, "clot_islands": 0,
                  "clot_area_mm2": 0.0, "largest_island_mm2": 0.0},
    }
    if he_rgb is None or not tissue_mask.any():
        return result

    from skimage.color import rgb2hed

    stains = rgb2hed(np.clip(np.asarray(he_rgb, dtype=np.float32), 0.0, 1.0))
    haematoxylin = stains[..., 0].astype(np.float32)
    eosin = stains[..., 1].astype(np.float32)

    # The window is specified in microns, not pixels, because its job is to span
    # several nuclei. In pixels its physical meaning would change with the
    # masking downsample: 9 px covers 73 um at 16x but only 9 um at 2x, which is
    # narrower than a single nucleus and measures nothing.
    pixel_um = float(microns_per_pixel) * max(int(downsample), 1)
    window = _odd_kernel(float(settings["clot_texture_window_um"]) / max(pixel_um, 1e-6), 3, 999)

    def windowed_mean(field: np.ndarray) -> np.ndarray:
        return cv2.blur(np.asarray(field, dtype=np.float32), (window, window))

    inside = tissue_mask
    nuclear_cut = float(np.percentile(haematoxylin[inside], float(settings["clot_nuclear_percentile"])))
    nuclear_coverage = windowed_mean((haematoxylin > nuclear_cut).astype(np.float32))
    eosin_texture = np.sqrt(np.clip(
        windowed_mean(eosin * eosin) - windowed_mean(eosin) ** 2, 0.0, None
    ))
    # Kept for the diagnostic figure and for anyone comparing with older runs.
    nuclear_texture = np.sqrt(np.clip(
        windowed_mean(haematoxylin * haematoxylin) - windowed_mean(haematoxylin) ** 2, 0.0, None
    ))

    # The pale end of the tissue, not its middle: a clot large enough to occupy
    # much of the slide would drag a median up towards its own value and then
    # fail to clear a threshold defined from it.
    eosin_reference = max(
        float(np.percentile(eosin[inside], float(settings["clot_eosin_reference_percentile"]))), 1e-9
    )
    texture_reference = float(np.median(eosin_texture[inside]))

    # Nucleated tissue on this same slide sets the bar, rather than a fixed
    # percentage: how much of an area nuclei cover depends on the tissue type and
    # on how the virtual H&E was rendered, so an absolute number is a different
    # demand on every slide. A high percentile is used so that a clot occupying
    # most of the tissue still cannot pull the reference down to its own value.
    nuclei_reference = max(float(np.percentile(
        nuclear_coverage[inside], float(settings["clot_nuclei_reference_percentile"])
    )), 1e-6)
    coverage_allowed = float(settings["clot_max_nuclei_ratio"]) * nuclei_reference

    pink = inside & (eosin >= float(settings["clot_eosin_ratio"]) * eosin_reference)
    nuclei_free = inside & (nuclear_coverage <= coverage_allowed)
    # A reference that has collapsed to zero carries no information: the eosin
    # channel is flat everywhere, so "smoother than usual" cannot be asked.
    smoothness_used = texture_reference > 0.02 * eosin_reference
    smooth = (
        inside & (eosin_texture <= float(settings["clot_eosin_smooth_ratio"]) * texture_reference)
        if smoothness_used else inside.copy()
    )

    candidate = pink & nuclei_free & smooth
    # Scattered specks are removed before the rim is recovered, not after. A
    # dilation cannot tell a real clot's core from a chance gap between nuclei,
    # so anything it is allowed to grow it will grow.
    speck_radius = max(int(round(0.25 * window)), 1)
    if speck_radius > 0 and candidate.any():
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN,
                                     _disk_kernel(speck_radius)) > 0
    if candidate.any():
        # Both windowed tests shrink every region by half a window, which eats
        # the rim of a real clot. Grow the core back, but only across tissue that
        # is pink and still well short of the nuclei density of normal tissue:
        # at a genuine clot's edge the window straddles the boundary and reads a
        # little above the limit, which is what this allowance is for.
        recovery = _disk_kernel(max(round(0.5 * window), 1))
        grown = cv2.dilate(candidate.astype(np.uint8), recovery) > 0
        rim = pink & (nuclear_coverage <= float(settings["clot_rim_nuclei_ratio"]) * coverage_allowed)
        # A union, never a replacement: this step adds back the rim the windows
        # ate, so it must not remove anything the three tests agreed on.
        candidate = candidate | (grown & rim)

    # Also physical: a morphology radius in pixels would erase different real-world
    # structures at every downsample, and silently act as a size floor.
    radius = int(round(float(settings["clot_morph_radius_um"]) / max(pixel_um, 1e-6)))
    if radius > 0:
        kernel = _disk_kernel(radius)
        candidate_u8 = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel)
        candidate = candidate_u8 > 0

    pixel_mm2 = (pixel_um * pixel_um) / 1e6
    min_area_px = int(float(settings["clot_min_area_mm2"]) / pixel_mm2) if pixel_mm2 > 0 else 0

    count, labels, stats_array, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    components = []
    for index in range(1, count):
        area_px = int(stats_array[index, cv2.CC_STAT_AREA])
        area_mm2 = area_px * pixel_mm2
        kept = area_px >= min_area_px
        keep[index] = kept
        components.append({
            "component": index, "area_px": area_px, "area_mm2": round(area_mm2, 4),
            "width_px": int(stats_array[index, cv2.CC_STAT_WIDTH]),
            "height_px": int(stats_array[index, cv2.CC_STAT_HEIGHT]),
            "kept": bool(kept),
        })
    clot_mask = keep[labels] if count > 1 else empty

    parameter_summary = {
        "clot_eosin_ratio": float(settings["clot_eosin_ratio"]),
        "clot_max_nuclei_ratio": float(settings["clot_max_nuclei_ratio"]),
        "nuclei_coverage_allowed": round(coverage_allowed, 5),
        "clot_eosin_smooth_ratio": float(settings["clot_eosin_smooth_ratio"]) if smoothness_used else None,
        "clot_min_area_mm2": float(settings["clot_min_area_mm2"]),
        "eosin_reference": round(eosin_reference, 6),
        "nuclear_threshold": round(nuclear_cut, 6),
        "microns_per_pixel_effective": round(pixel_um, 4),
    }

    region_count, region_labels, regions = _component_records(clot_mask, pixel_um, downsample)
    eosin_ratio_map = eosin / eosin_reference
    for record in regions:
        region = region_labels == record["label"]
        eosin_ratio = float(np.median(eosin_ratio_map[region]))
        coverage_value = float(np.median(nuclear_coverage[region]))
        smoothness = (
            float(np.median(eosin_texture[region]) / texture_reference)
            if smoothness_used and texture_reference > 0 else None
        )
        # How far past each gate the region sits, worst case, so a region that
        # only just qualifies is not reported with the same weight as an obvious
        # haemorrhage.
        margins = [
            eosin_ratio / max(float(settings["clot_eosin_ratio"]), 1e-9),
            coverage_allowed / max(coverage_value, 1e-6),
        ]
        if smoothness is not None:
            margins.append(float(settings["clot_eosin_smooth_ratio"]) / max(smoothness, 1e-6))
        confidence = float(np.clip((min(margins) - 1.0) / 2.0, 0.0, 1.0))
        record["measurements"] = {
            "area_mm2": round(record["area_um2"] / 1e6, 4),
            "area_um2": round(record["area_um2"], 1),
            "eosin_ratio": round(eosin_ratio, 3),
            "nuclear_coverage": round(coverage_value, 4),
            "eosin_smoothness_ratio": round(smoothness, 3) if smoothness is not None else -1.0,
            "confidence": round(confidence, 3),
        }
        record["spacemap"] = {
            "schema": ARTIFACT_REGION_SCHEMA,
            "artifact_type": "blood_clot",
            "region_id": f"{sample_id + '_' if sample_id else ''}clot_{record['label']:03d}",
            "sample_id": sample_id or None,
            "detector": "eosin_rich_nuclei_free_structureless",
            "area_um2": round(record["area_um2"], 1),
            "centroid_full_res_px": record["centroid_full_res_px"],
            "bbox_full_res_px": record["bbox_full_res_px"],
            "confidence": round(confidence, 3),
            "evidence": {
                "eosin_ratio": round(eosin_ratio, 3),
                "eosin_ratio_required": float(settings["clot_eosin_ratio"]),
                "nuclear_coverage": round(coverage_value, 4),
                "nuclear_coverage_allowed": round(coverage_allowed, 5),
                "nuclear_coverage_in_normal_tissue": round(nuclei_reference, 5),
                "eosin_smoothness_ratio": round(smoothness, 3) if smoothness is not None else None,
                "eosin_smoothness_allowed": (
                    float(settings["clot_eosin_smooth_ratio"]) if smoothness_used else None
                ),
                "smoothness_test_used": bool(smoothness_used),
            },
            "candidate_markers": [],
            "parameters": parameter_summary,
            "review": {"status": "pending", "decision": None, "reviewer": None, "notes": None},
        }
        record["name"] = f"Blood clot {record['label']} ({record['area_um2'] / 1e6:.3f} mm2)"

    kept_areas = [component["area_mm2"] for component in components if component["kept"]]
    result.update({
        "clot_mask": clot_mask,
        "eosin": eosin,
        "nuclear_texture": nuclear_texture,
        "nuclear_coverage": nuclear_coverage,
        "eosin_texture": eosin_texture,
        "components": sorted(components, key=lambda component: -component["area_px"]),
        "regions": regions,
        "region_labels": region_labels,
        "parameters": parameter_summary,
        "stats": {
            "clot_percent": float(100.0 * clot_mask.sum() / max(int(tissue_mask.sum()), 1)),
            "clot_pixels": int(clot_mask.sum()),
            "clot_islands": int(len(kept_areas)),
            "clot_area_mm2": round(float(sum(kept_areas)), 4),
            "largest_island_mm2": round(float(max(kept_areas)), 4) if kept_areas else 0.0,
            "rejected_islands": int(len(components) - len(kept_areas)),
            "eosin_reference": round(eosin_reference, 6),
            "nuclear_threshold": round(nuclear_cut, 6),
            "eosin_texture_reference": round(texture_reference, 6),
            "nuclei_coverage_reference": round(nuclei_reference, 5),
            "nuclei_coverage_allowed": round(coverage_allowed, 5),
            "smoothness_test_used": bool(smoothness_used),
            "pink_percent": float(100.0 * pink.sum() / max(int(tissue_mask.sum()), 1)),
            "nuclei_free_percent": float(100.0 * nuclei_free.sum() / max(int(tissue_mask.sum()), 1)),
            "min_area_px": int(min_area_px),
            "texture_window_px": int(window),
            "morph_radius_px": int(radius),
        },
    })
    return result


def detect_tissue_mask(
    reader: PyramidImageReader,
    config: dict[str, Any],
    he_indices: Sequence[int] | None = None,
    codex_indices: Sequence[int] | None = None,
    progress=None,
    need_codex_maps: bool = False,
) -> dict[str, Any]:
    """Separate tissue from slide background using the virtual H&E channels.

    Each H&E channel is normalized against its own white-background reference,
    converted to optical density and combined into a stain-strength map before
    Otsu thresholding.  Zero-valued acquisition borders are excluded explicitly
    so black scanner frames and CODEX-only lint cannot enter the tissue mask.

    In the default ``he_only`` mode the CODEX markers contribute nothing to the
    mask, so they are not read at all: only the three virtual H&E channels are
    decoded. That is the difference between three reads and one read per marker,
    which on a 40-channel ROI is the difference between about a second and about
    half a minute. Pass ``need_codex_maps=True`` (as the artifact step does) when
    the per-marker projection and robust-Z maps are required.

    Returns a dictionary with the mask, the intermediate maps that later stages
    reuse (``codex_projection``, ``codex_robust_max``, ``he_gray``,
    ``he_rgb``), and a ``stats`` block for the UI.
    """
    settings = {**DEFAULT_TISSUE_CONFIG, **(config or {})}
    factor = int(settings["blur_downsample"])
    microns_per_pixel = float(settings["microns_per_pixel"])
    clip_z = float(settings["robust_signal_clip_z"])
    breadth_sigma = float(settings["bright_breadth_channel_sigma"])

    if he_indices is None or codex_indices is None:
        detected_he, detected_codex = split_he_and_codex_channels(reader.channel_names)
        he_indices = detected_he if he_indices is None else he_indices
        codex_indices = detected_codex if codex_indices is None else codex_indices
    he_indices = list(he_indices)
    codex_indices = list(codex_indices)

    mode_needs_codex = str(settings["tissue_mask_mode"]) != "he_only"
    mode_is_codex_only = str(settings["tissue_mask_mode"]) in CODEX_ONLY_MODES
    has_he_stack = len(he_indices) >= 3
    # Without a usable H&E stack the CODEX projection is the only signal left.
    read_codex = bool(codex_indices) and (need_codex_maps or mode_needs_codex or not has_he_stack)
    # In codex_only mode the H&E layers no longer drive the mask, but they are
    # still worth reading when present: step 2 uses the H&E focus map as its
    # blur reference. Three extra reads is a negligible cost.

    he_channels: list[np.ndarray] = []
    codex_projection: np.ndarray | None = None
    codex_robust_max: np.ndarray | None = None
    robust_scratch: np.ndarray | None = None
    channel_positive_count: np.ndarray | None = None
    # Per-marker sharpness and signal, at tile resolution. Built here because
    # this is the only pass that decodes every channel; keeping them costs a few
    # megabytes and saves step 2 a second full read of the image.
    channel_focus_grid: np.ndarray | None = None
    channel_signal_grid: np.ndarray | None = None
    attribution_grid: tuple[int, int] | None = None
    attribution_decimate = 1
    codex_slot = {index: position for position, index in enumerate(codex_indices)}
    attribution_pixel_um = microns_per_pixel * factor
    attribution_pre_sigma_px = float(DEFAULT_ARTIFACT_CONFIG["noise_smoothing_um"]) / max(attribution_pixel_um, 1e-6)
    attribution_ksize = _odd_kernel(
        float(DEFAULT_ARTIFACT_CONFIG["focus_detail_um"]) / max(attribution_pixel_um, 1e-6), 1, 31
    )

    # A channel may legitimately appear in both lists: the H&E layers are always
    # needed for the optical-density mask, and the user may additionally include
    # them in the CODEX projection used for artifact seeding.
    codex_set = set(codex_indices) if read_codex else set()
    he_set = set(he_indices)
    channels_to_read = sorted(codex_set | he_set)
    total = len(channels_to_read)
    with reader.open() as tif:
        for position, index in enumerate(channels_to_read):
            channel = reader.read_channel(tif, index, factor)
            if index in he_set:
                he_channels.append(channel)
            if index in codex_set:
                slot = codex_slot[index]
                if codex_robust_max is None:
                    codex_robust_max = _robust_channel_z(channel, clip_z)
                    robust_z = codex_robust_max
                    channel_positive_count = (codex_robust_max >= breadth_sigma).astype(np.uint8)
                    codex_projection = channel.copy()
                    robust_scratch = np.empty_like(codex_robust_max)
                    attribution_grid, attribution_decimate = _attribution_grid_geometry(
                        channel.shape, attribution_pixel_um
                    )
                    channel_focus_grid = np.zeros((len(codex_indices), *attribution_grid), dtype=np.float32)
                    channel_signal_grid = np.zeros((len(codex_indices), *attribution_grid), dtype=np.float32)
                else:
                    robust_z = _robust_channel_z(channel, clip_z, out=robust_scratch)
                    channel_positive_count += (robust_z >= breadth_sigma).astype(np.uint8)
                    np.maximum(codex_robust_max, robust_z, out=codex_robust_max)
                    np.maximum(codex_projection, channel, out=codex_projection)
                # The maximum projection above is blind to defocus that hits
                # only some cycles: one channel that stayed sharp keeps the
                # projection sharp. Scoring each marker separately here is what
                # lets step 2 see that, and name the markers responsible.
                channel_focus_grid[slot] = _channel_focus_tiles(
                    channel, attribution_grid,
                    pre_sigma_px=attribution_pre_sigma_px,
                    ksize=attribution_ksize,
                    decimate=attribution_decimate,
                )
                channel_signal_grid[slot] = cv2.resize(
                    robust_z, (attribution_grid[1], attribution_grid[0]), interpolation=cv2.INTER_AREA
                )
            if progress is not None:
                progress(position + 1, total, reader.channel_names[index])

    if codex_projection is None and not he_channels:
        raise ValueError("This image has neither virtual H&E nor CODEX channels to work from.")

    if codex_projection is not None:
        codex_projection = _clamp_and_normalize(codex_projection, float(settings["clamp_sigma"]))
        height, width = codex_projection.shape
    else:
        height, width = he_channels[0].shape

    he_gray = None
    he_stain_score = None
    he_valid_pixels = None
    he_white_references: list[float] = []
    he_rgb = None
    valid_min = float(settings["he_valid_min_intensity"])

    if len(he_channels) >= 3:
        he_stain_score = np.zeros_like(he_channels[0], dtype=np.float32)
        he_valid_pixels = np.ones_like(he_channels[0], dtype=bool)
        for channel in he_channels[:3]:
            sample = channel[::4, ::4]
            positive = sample[np.isfinite(sample) & (sample > valid_min)]
            white_reference = float(np.percentile(positive, float(settings["he_white_percentile"]))) if positive.size else 1.0
            white_reference = max(white_reference, 1.0)
            he_white_references.append(white_reference)
            he_valid_pixels &= np.isfinite(channel) & (channel > valid_min)
            normalized = np.clip((channel + 1.0) / (white_reference + 1.0), 1e-4, 1.0)
            he_stain_score += -np.log(normalized)
        he_stain_score /= 3.0
        stain_scale = float(np.percentile(he_stain_score[he_valid_pixels][::16], 99.5))
        he_gray = np.clip(he_stain_score / max(stain_scale, 1e-6) * 255.0, 0, 255).astype(np.float32)
        he_rgb = np.stack(
            [
                np.clip(channel / reference, 0.0, 1.0)
                for channel, reference in zip(he_channels[:3], he_white_references)
            ],
            axis=-1,
        )
    elif he_channels:
        he_gray = _clamp_and_normalize(he_channels[0], float(settings["clamp_sigma"]))

    codex_threshold = None
    if codex_projection is not None:
        if str(settings.get("codex_threshold_mode", "fixed")).lower() == "otsu":
            # A fixed intensity cut does not transfer between panels or
            # exposures; Otsu on the normalized projection adapts to the image.
            sample = codex_projection[::4, ::4]
            sample = sample[np.isfinite(sample)]
            codex_threshold = float(threshold_otsu(sample)) if sample.size else float(settings["tissue_threshold"])
        else:
            codex_threshold = float(settings["tissue_threshold"])
        codex_mask = codex_projection >= codex_threshold
    else:
        codex_mask = np.zeros((height, width), dtype=bool)
    otsu_value = None
    he_od_threshold = None
    mode = str(settings["tissue_mask_mode"])

    if mode_is_codex_only:
        if codex_projection is None:
            raise ValueError(
                "codex_only tissue detection needs at least one marker channel, but none were selected."
            )
        tissue_mask = codex_mask.copy()
    elif he_stain_score is not None:
        valid_values = he_stain_score[he_valid_pixels]
        otsu_value = float(threshold_otsu(valid_values[::16]))
        he_od_threshold = otsu_value * float(settings["he_od_threshold_factor"])
        he_mask = (he_stain_score >= he_od_threshold) & he_valid_pixels
        if mode == "he_only":
            tissue_mask = he_mask.copy()
        elif mode == "he_with_codex_edge_rescue":
            rescue_px = int(np.ceil(float(settings["he_codex_edge_rescue_um"]) / (microns_per_pixel * factor)))
            rescue_zone = cv2.dilate(he_mask.astype(np.uint8), disk(max(rescue_px, 1)).astype(np.uint8)) > 0
            tissue_mask = he_mask | (codex_mask & rescue_zone)
        elif mode == "he_codex_union":
            tissue_mask = he_mask | codex_mask
        else:
            raise ValueError(f"Unknown tissue mask mode: {mode}")
    else:
        tissue_mask = codex_mask.copy()

    border_px = int(np.ceil(float(settings["tissue_border_exclusion_um"]) / (microns_per_pixel * factor)))
    if border_px > 0:
        tissue_mask[:border_px, :] = False
        tissue_mask[-border_px:, :] = False
        tissue_mask[:, :border_px] = False
        tissue_mask[:, -border_px:] = False

    morphology_kernel = disk(int(settings["tissue_morph_radius"])).astype(np.uint8)
    tissue_u8 = tissue_mask.astype(np.uint8)
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, morphology_kernel)
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, morphology_kernel)
    tissue_mask = _remove_small_components(tissue_u8 > 0, int(settings["min_tissue_island_px"]), connectivity=4)
    if not tissue_mask.any():
        tissue_mask = tissue_u8 > 0

    if border_px > 0:
        tissue_mask[:border_px, :] = False
        tissue_mask[-border_px:, :] = False
        tissue_mask[:, :border_px] = False
        tissue_mask[:, -border_px:] = False

    # Blood clot runs here rather than in the artifact step because it needs only
    # the virtual H&E, which this pass has already decoded. The artifact step
    # decodes every selected marker, which for this purpose would be wasted work.
    clot = None
    if bool(settings.get("detect_blood_clot", False)) and he_rgb is not None:
        clot = detect_blood_clot(
            he_rgb, tissue_mask, settings,
            microns_per_pixel=microns_per_pixel, downsample=factor,
            sample_id=str(settings.get("sample_id", "")),
        )
        if bool(settings.get("clot_exclude_from_tissue", True)) and clot["clot_mask"].any():
            tissue_mask = tissue_mask & ~clot["clot_mask"]

    return {
        "tissue_mask": tissue_mask,
        "clot": clot,
        "clot_mask": clot["clot_mask"] if clot else None,
        "codex_projection": codex_projection,
        "codex_robust_max": codex_robust_max,
        "channel_positive_count": channel_positive_count,
        "channel_focus_grid": channel_focus_grid,
        "channel_signal_grid": channel_signal_grid,
        "codex_channel_names": [reader.channel_names[index] for index in codex_indices],
        "attribution_grid_shape": attribution_grid,
        "he_gray": he_gray,
        "he_stain_score": he_stain_score,
        "he_valid_pixels": he_valid_pixels,
        "he_rgb": he_rgb,
        "he_white_references": he_white_references,
        "border_px": border_px,
        "downsample": factor,
        "shape": (height, width),
        "codex_indices": codex_indices,
        "he_indices": he_indices,
        "stats": {
            "tissue_percent": float(100.0 * tissue_mask.mean()),
            "tissue_pixels": int(tissue_mask.sum()),
            "otsu": otsu_value,
            "he_od_threshold": he_od_threshold,
            "mode": mode,
            "codex_threshold": codex_threshold,
            "codex_threshold_mode": str(settings.get("codex_threshold_mode", "fixed")),
            "he_channels_used": len(he_channels),
            "he_white_references": [round(value, 1) for value in he_white_references],
            **({"clot_" + k.removeprefix("clot_"): v for k, v in clot["stats"].items()} if clot else {}),
        },
    }


# ==========================================================================
# Stage 2 - blur, flat-bright and filamentous lint artifacts
# ==========================================================================

DEFAULT_ARTIFACT_CONFIG: dict[str, Any] = {
    # --- Out-of-focus tissue: what counts as blurry ---
    "blur_reference_mode": "he_and_self",
    "blur_threshold_mode": "auto",
    "blur_sensitivity": 1.0,
    # The step the tab's simple control sits on; it is what produces
    # blur_sensitivity above, and is stored so a project reopens on it.
    "blur_sensitivity_level": 4,
    "blur_min_separation": 4.0,
    "max_blur_fraction": 0.30,
    "blur_ratio_threshold": 0.35,
    "blur_evidence_rule": "either",
    "sharp_reference_percentile": 75.0,
    "he_structure_percentile": 15.0,
    # --- Out-of-focus tissue: where blur may be judged at all ---
    "require_codex_signal": True,
    "codex_signal_z": 3.0,
    "signal_support_um": 25.0,
    # --- Out-of-focus tissue: per-marker consensus ---
    "use_marker_consensus": True,
    "marker_consensus_fraction": 0.40,
    "marker_consensus_min_markers": 4,
    "marker_signal_z": 2.0,
    # --- Out-of-focus tissue: measurement scales, in microns so that they mean
    # the same thing at any blur_downsample ---
    "focus_detail_um": 5.0,
    "focus_window_um": 32.0,
    "noise_smoothing_um": 1.5,
    "region_smoothing_um": 5.0,
    "fill_holes_um": 15.0,
    "remove_specks_um": 7.0,
    "expand_microns": 15.0,
    "min_area_microns2": 2000.0,
    # --- Flat bright artifacts ---
    "detect_bright_artifacts": True,
    "use_breadth_signal": False,
    "bright_breadth_fraction": 0.50,
    "bright_flat_mean_percentile": 99.5,
    "bright_flat_texture_percentile": 2.0,
    "bright_fill_holes_um": 9.0,
    "bright_remove_specks_um": 2.0,
    "bright_expand_microns": 10.0,
    "min_area_microns2_bright": 20.0,
    # --- Filamentous lint ---
    "detect_filamentous_lint": True,
    "lint_robust_z_threshold": 12.0,
    "lint_he_exclusion_margin_um": 5.0,
    "lint_min_area_microns2": 20.0,
    "lint_max_area_microns2": 100000.0,
    "lint_min_length_microns": 75.0,
    "lint_min_width_microns": 5.0,
    "lint_max_width_microns": 60.0,
    "lint_min_aspect_ratio": 4.0,
    "lint_join_gaps_um": 3.0,
    "lint_remove_specks_um": 0.0,
    "lint_expand_microns": 5.0,
    "lint_seam_max_width_microns": 8.0,
    "lint_seam_min_span_ratio": 5.0,
    # --- Output ---
    "boundary_simplify_px": 2.0,
    "write_review_manifest": True,
}


def _separated_threshold(
    values: np.ndarray,
    *,
    min_gap: float,
    max_share: float,
    sensitivity: float = 1.0,
) -> tuple[float | None, dict[str, Any]]:
    """A cut between a sharp and a soft population, or nothing if there is only one.

    Sharpness inside real tissue varies several-fold for honest reasons: a
    marker decorates some structures and not others, and a gland wall carries
    more detail than the stroma beside it. A fixed cut on that continuum always
    removes its lower tail, which on a slide that is entirely in focus means
    deleting the least textured third of the tissue.

    Defocus does not look like that. It moves a whole region far down the scale
    and leaves a distinct second mode, so the test for "is any of this blurry"
    is whether two populations exist at all. Otsu proposes the split; the gap
    between the two medians decides whether to believe it. A slide with no
    defocus produces a gap near the natural spread of the tissue and is
    reported, correctly, as having no blur.
    """
    diagnostics: dict[str, Any] = {"gap": None, "share": None, "separability": None, "cut": None}
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size < 200:
        return None, diagnostics
    logs = np.log(finite)
    # Otsu is a histogram method; a sample of a few hundred thousand points
    # describes the same histogram as tens of millions and costs far less.
    stride = max(logs.size // 200_000, 1)
    try:
        cut = float(threshold_otsu(logs[::stride]))
    except Exception:
        return None, diagnostics
    soft, sharp = logs <= cut, logs > cut
    if soft.sum() < 10 or sharp.sum() < 10:
        return None, diagnostics

    gap = float(np.exp(float(np.median(logs[sharp])) - float(np.median(logs[soft]))))
    share = float(soft.mean())
    weight = share * (1.0 - share)
    between = weight * (float(logs[soft].mean()) - float(logs[sharp].mean())) ** 2
    separability = float(between / max(float(logs.var()), 1e-12))
    diagnostics.update({"gap": round(gap, 3), "share": round(share, 4),
                        "separability": round(separability, 3), "cut": round(float(np.exp(cut)), 5)})
    if gap < float(min_gap) or share > float(max_share):
        return None, diagnostics
    return float(np.exp(cut)) * float(sensitivity), diagnostics

def _component_records(mask: np.ndarray, pixel_size_um: float, downsample: int) -> tuple[int, np.ndarray, list[dict[str, Any]]]:
    """Label a binary mask and describe each connected region geometrically."""
    count, labels, stats_array, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    pixel_area_um2 = float(pixel_size_um) ** 2
    scale = float(max(int(downsample), 1))
    records: list[dict[str, Any]] = []
    for label_id in range(1, count):
        left = int(stats_array[label_id, cv2.CC_STAT_LEFT])
        top = int(stats_array[label_id, cv2.CC_STAT_TOP])
        width = int(stats_array[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats_array[label_id, cv2.CC_STAT_HEIGHT])
        area_px = int(stats_array[label_id, cv2.CC_STAT_AREA])
        centroid_x, centroid_y = float(centroids[label_id][0]), float(centroids[label_id][1])
        records.append({
            "label": int(label_id),
            "bbox_px": (left, top, width, height),
            "area_px": area_px,
            "area_um2": float(area_px * pixel_area_um2),
            "equivalent_diameter_um": float(2.0 * np.sqrt(area_px * pixel_area_um2 / np.pi)),
            "centroid_full_res_px": [round(centroid_x * scale, 1), round(centroid_y * scale, 1)],
            "bbox_full_res_px": [
                round(left * scale, 1), round(top * scale, 1),
                round(width * scale, 1), round(height * scale, 1),
            ],
        })
    return count, labels, records


def _region_focus_evidence(
    tiles: np.ndarray,
    marker_ratio_grid: np.ndarray | None,
    signal_tiles: np.ndarray | None,
    marker_names: Sequence[str],
    threshold: float,
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], int, int]:
    """Rank markers by how far each one has lost focus inside one region.

    A marker is only judged where it actually has signal, and only against its
    own sharp tiles elsewhere in the same image. That is what separates "this
    cycle went out of focus" from "this antibody simply has no fine structure",
    which a single pooled projection cannot tell apart.
    """
    if marker_ratio_grid is None or signal_tiles is None or not tiles.any():
        return [], 0, 0
    tile_total = float(tiles.sum())
    ranked: list[dict[str, Any]] = []
    for channel in range(marker_ratio_grid.shape[0]):
        selected = tiles & signal_tiles[channel]
        if not selected.any():
            continue
        values = marker_ratio_grid[channel][selected]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ratio = float(np.median(values))
        ranked.append({
            "marker": str(marker_names[channel]) if channel < len(marker_names) else f"channel_{channel}",
            "focus_ratio": round(ratio, 4),
            "out_of_focus": bool(ratio < threshold),
            "coverage": round(float(selected.sum()) / max(tile_total, 1.0), 3),
        })
    ranked.sort(key=lambda entry: entry["focus_ratio"])
    markers_with_signal = len(ranked)
    markers_out_of_focus = sum(1 for entry in ranked if entry["out_of_focus"])
    return ranked[:top_k], markers_with_signal, markers_out_of_focus


def _region_signal_evidence(
    tiles: np.ndarray,
    signal_grid: np.ndarray | None,
    marker_names: Sequence[str],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank markers by how brightly each one stands out inside one region."""
    if signal_grid is None or not tiles.any():
        return []
    ranked: list[dict[str, Any]] = []
    for channel in range(signal_grid.shape[0]):
        values = signal_grid[channel][tiles]
        if values.size == 0:
            continue
        ranked.append({
            "marker": str(marker_names[channel]) if channel < len(marker_names) else f"channel_{channel}",
            "signal_z": round(float(np.median(values)), 3),
        })
    ranked.sort(key=lambda entry: -entry["signal_z"])
    return ranked[:top_k]


def detect_artifacts(
    tissue_mask: np.ndarray,
    codex_projection: np.ndarray,
    he_gray: np.ndarray | None,
    codex_robust_max: np.ndarray | None,
    channel_positive_count: np.ndarray | None,
    n_codex_channels: int,
    config: dict[str, Any],
    microns_per_pixel: float,
    downsample: int,
    border_px: int = 0,
    *,
    marker_names: Sequence[str] | None = None,
    channel_focus_grid: np.ndarray | None = None,
    channel_signal_grid: np.ndarray | None = None,
    sample_id: str = "",
) -> dict[str, Any]:
    """Detect out-of-focus tissue, flat bright glare and filamentous lint.

    Blur is evaluated only inside the reviewed tissue mask.  Filamentous lint is
    seeded strictly *outside* tissue, so marker-positive glands, vessels and
    stromal tracks are never classified as lint.  Only the lint that overlaps
    tissue is subtracted; the full lint mask is exported separately for review.

    Out-of-focus tissue is judged by two independent lines of evidence, because
    each one alone has a failure mode that shows up on real slides:

    * **Pooled sharpness against a reference.** The maximum projection of the
      selected markers is scored for local sharpness and compared either with
      the virtual H&E of the same spot or with the sharpest tissue elsewhere in
      the same image. This catches whole regions that went soft.
    * **Per-marker consensus.** Every marker is scored on its own coarse grid
      and compared against its *own* sharp tiles. A region is out of focus when
      most of the markers that have signal there have individually lost focus.
      The pooled projection cannot see this: it is a maximum, so one sharp
      channel such as DAPI hides twenty blurred cycles behind it.

    Both are confined to tissue that actually carries marker signal. Without
    that gate, antibody-negative stroma scores near-zero sharpness for the
    honest reason that nothing stained there, and gets deleted as blur.

    Every returned region carries the measurements and the ranked marker
    evidence that produced it, so a reviewer, or the AI review pass, can accept
    or reject each one on its own merits.
    """
    settings = {**DEFAULT_ARTIFACT_CONFIG, **(config or {})}
    height, width = tissue_mask.shape[:2]
    pixel_size_um = float(microns_per_pixel) * max(int(downsample), 1)
    pixel_area_um2 = pixel_size_um ** 2
    px_per_um = 1.0 / pixel_size_um
    marker_names = list(marker_names or [])

    # The raw config, not the merged settings, so that a project saved before
    # lengths moved to microns still resolves through its pixel-valued keys.
    user_config = dict(config or {})

    def length_px(key: str) -> float:
        return _scale_um(user_config, key, pixel_size_um) * px_per_um

    def area_px(key: str) -> int:
        return int(np.ceil(float(settings[key]) / pixel_area_um2))

    detail_ksize = _odd_kernel(length_px("focus_detail_um"), 1, 31)
    focus_window_px = _odd_kernel(length_px("focus_window_um"), 3, 999)
    pre_sigma_px = max(length_px("noise_smoothing_um"), 0.0)
    post_sigma_px = max(length_px("region_smoothing_um"), 0.0)
    blur_threshold = float(settings["blur_ratio_threshold"])
    reference_percentile = float(
        user_config.get("sharp_reference_percentile")
        or user_config.get("self_reference_percentile")
        or settings["sharp_reference_percentile"]
    )

    codex_focus = _focus_map(
        codex_projection, pre_sigma_px=pre_sigma_px, ksize=detail_ksize,
        window_px=focus_window_px, post_sigma_px=post_sigma_px,
    )
    he_focus = (
        _focus_map(he_gray, pre_sigma_px=pre_sigma_px, ksize=detail_ksize,
                   window_px=focus_window_px, post_sigma_px=post_sigma_px)
        if he_gray is not None else None
    )

    # --- where blur is allowed to be judged at all ---------------------------
    evaluate_mask = tissue_mask.copy()
    signal_support = None
    if bool(settings.get("require_codex_signal", True)) and codex_robust_max is not None:
        signal_support = codex_robust_max >= float(settings.get("codex_signal_z", 3.0))
        support_px = int(round(length_px("signal_support_um")))
        if support_px > 0:
            signal_support = cv2.morphologyEx(
                signal_support.astype(np.uint8), cv2.MORPH_CLOSE, _disk_kernel(support_px)
            ) > 0
        evaluate_mask &= signal_support

    he_has_structure = None
    if he_focus is not None:
        inside_he = he_focus[tissue_mask]
        structure_floor = (
            float(np.percentile(inside_he, float(settings.get("he_structure_percentile", 15.0))))
            if inside_he.size else 0.0
        )
        he_has_structure = he_focus >= structure_floor

    # --- arm 1: pooled sharpness against a reference -------------------------
    if "blur_reference_mode" in user_config:
        mode = str(user_config["blur_reference_mode"])
    elif "use_he_reference" in user_config:
        mode = "he" if bool(user_config["use_he_reference"]) else "self"
    else:
        mode = str(settings["blur_reference_mode"])
    if he_focus is None and mode in ("he", "he_and_self", "he_or_self"):
        mode = "self"
    uses_he = mode in ("he", "he_and_self", "he_or_self")
    uses_self = mode in ("self", "he_and_self", "he_or_self")

    reference_domain = evaluate_mask if evaluate_mask.any() else tissue_mask
    ratio_arms: list[np.ndarray] = []
    ratio_he = None
    ratio_self = None
    self_reference = None
    def anchored(score: np.ndarray) -> tuple[np.ndarray, float]:
        """Rescale a sharpness score so 1.0 means "as sharp as this image gets".

        Both scores below are read against a high percentile of themselves over
        the evaluated tissue rather than against a fixed number. A raw focus
        value has no absolute meaning - it depends on the panel, the exposure
        and, for the H&E ratio, on two different imaging modalities - so a fixed
        cut transfers between slides only by luck. The percentile deliberately
        sits below the top of the distribution: strong edges such as a tissue
        border or a vessel wall produce focus values several times higher than
        ordinary sharp tissue, and anchoring on those would make everything else
        look blurry.
        """
        values = score[reference_domain]
        anchor = float(np.percentile(values, reference_percentile)) if values.size else 0.0
        anchor = max(anchor, 1e-12)
        return score / anchor, anchor

    if uses_he and he_focus is not None:
        he_inside = he_focus[reference_domain]
        # A data-driven floor, not a fixed epsilon: the magnitude of a focus
        # score depends on the image, so a constant would be a different-sized
        # guard on every slide.
        he_floor = max(float(np.percentile(he_inside, 5.0)) if he_inside.size else 0.0, 1e-8)
        ratio_he, _ = anchored(codex_focus / np.maximum(he_focus, he_floor))
    if uses_self:
        ratio_self, self_reference = anchored(codex_focus)

    # The ratio actually used for reporting: whichever arm speaks against the
    # region most strongly, so a stated severity matches what triggered it.
    reported_ratio = None
    for candidate in (ratio_he, ratio_self):
        if candidate is None:
            continue
        reported_ratio = candidate if reported_ratio is None else np.minimum(reported_ratio, candidate)
    if reported_ratio is None:
        reported_ratio = np.ones_like(codex_focus, dtype=np.float32)

    sensitivity = float(settings.get("blur_sensitivity", 1.0))
    automatic = str(settings.get("blur_threshold_mode", "auto")).lower() == "auto"
    if automatic:
        pooled_threshold, pooled_separation = _separated_threshold(
            reported_ratio[reference_domain],
            min_gap=float(settings["blur_min_separation"]),
            max_share=float(settings["max_blur_fraction"]),
            sensitivity=sensitivity,
        )
    else:
        pooled_threshold = float(settings["blur_ratio_threshold"]) * sensitivity
        pooled_separation = {"gap": None, "share": None, "separability": None, "cut": None}
    # Still needed for reporting a severity and for labelling each marker, even
    # when the pooled test found nothing to cut.
    blur_threshold = (
        pooled_threshold if pooled_threshold is not None
        else float(settings["blur_ratio_threshold"]) * sensitivity
    )
    marker_focus_threshold = float(settings["blur_ratio_threshold"]) * sensitivity

    if pooled_threshold is not None:
        if ratio_he is not None:
            arm = (ratio_he < pooled_threshold) & evaluate_mask
            if he_has_structure is not None:
                arm &= he_has_structure
            ratio_arms.append(arm)
        if ratio_self is not None:
            ratio_arms.append((ratio_self < pooled_threshold) & evaluate_mask)

    if not ratio_arms:
        blur_from_ratio = np.zeros_like(tissue_mask, dtype=bool)
    elif mode == "he_and_self" and len(ratio_arms) == 2:
        blur_from_ratio = ratio_arms[0] & ratio_arms[1]
    else:
        blur_from_ratio = ratio_arms[0]
        for arm in ratio_arms[1:]:
            blur_from_ratio = blur_from_ratio | arm

    # --- arm 2: per-marker consensus ----------------------------------------
    consensus_map = None
    consensus_threshold = None
    marker_ratio_grid = None
    signal_tiles = None
    grid_shape = None
    blur_from_markers = np.zeros_like(tissue_mask, dtype=bool)
    consensus_available = (
        bool(settings.get("use_marker_consensus", True))
        and channel_focus_grid is not None
        and channel_signal_grid is not None
        and channel_focus_grid.ndim == 3
        and channel_focus_grid.shape[0] >= int(settings.get("marker_consensus_min_markers", 4))
    )
    if consensus_available:
        grid_shape = (int(channel_focus_grid.shape[1]), int(channel_focus_grid.shape[2]))
        grid_size = (grid_shape[1], grid_shape[0])
        evaluate_tiles = cv2.resize(
            evaluate_mask.astype(np.float32), grid_size, interpolation=cv2.INTER_AREA
        ) >= 0.5
        marker_signal_z = float(settings.get("marker_signal_z", 2.0))
        signal_tiles = channel_signal_grid >= marker_signal_z
        references = np.full(channel_focus_grid.shape[0], np.nan, dtype=np.float32)
        for channel in range(channel_focus_grid.shape[0]):
            selected = signal_tiles[channel] & evaluate_tiles
            values = channel_focus_grid[channel][selected]
            values = values[np.isfinite(values) & (values > 0)]
            if values.size >= 8:
                references[channel] = float(np.percentile(values, reference_percentile))
        with np.errstate(invalid="ignore", divide="ignore"):
            marker_ratio_grid = channel_focus_grid / references[:, None, None]
        usable = signal_tiles & np.isfinite(marker_ratio_grid)
        markers_present = usable.sum(axis=0)
        # Each marker is judged against its own sharp tiles, so this cut is its
        # own number. Reusing the pooled cut would be a category error: that one
        # is chosen from the distribution of a maximum projection, which is a
        # different quantity on a different scale.
        markers_soft = (usable & (marker_ratio_grid < marker_focus_threshold)).sum(axis=0)
        minimum_markers = int(settings.get("marker_consensus_min_markers", 4))
        consensus_grid = np.where(
            markers_present >= minimum_markers,
            markers_soft / np.maximum(markers_present, 1),
            0.0,
        ).astype(np.float32)
        consensus_map = cv2.resize(consensus_grid, (width, height), interpolation=cv2.INTER_LINEAR)
        consensus_threshold = float(settings.get("marker_consensus_fraction", 0.5)) / max(sensitivity, 1e-6)
        # In automatic mode both arms answer to the same question first: does
        # this image contain a distinct out-of-focus population at all? That is
        # asked of the pooled sharpness, which is a continuous quantity with a
        # well-behaved distribution. The share of markers is a bounded ratio full
        # of ties at 0 and 1, and a split of it says more about the ties than
        # about the image.
        if automatic and pooled_threshold is None:
            consensus_threshold = None
        if consensus_threshold is not None:
            blur_from_markers = (consensus_map >= consensus_threshold) & evaluate_mask

    rule = str(settings.get("blur_evidence_rule", "either"))
    if not consensus_available or rule == "pooled_only":
        blur_seed = blur_from_ratio
    elif rule == "markers_only":
        blur_seed = blur_from_markers
    elif rule == "both":
        blur_seed = blur_from_ratio & blur_from_markers
    else:
        blur_seed = blur_from_ratio | blur_from_markers

    # --- clean up, filter by size, then add the safety margin ----------------
    # Size filtering happens *before* the margin is added, otherwise the margin
    # inflates every speck past the minimum area and the setting does nothing.
    blur_u8 = blur_seed.astype(np.uint8)
    close_px = int(round(length_px("fill_holes_um")))
    open_px = int(round(length_px("remove_specks_um")))
    if close_px > 0:
        blur_u8 = cv2.morphologyEx(blur_u8, cv2.MORPH_CLOSE, _disk_kernel(close_px))
    if open_px > 0:
        blur_u8 = cv2.morphologyEx(blur_u8, cv2.MORPH_OPEN, _disk_kernel(open_px))
    blur_core = (blur_u8 > 0) & tissue_mask
    blur_core = _remove_small_components(blur_core, area_px("min_area_microns2"), connectivity=8)
    blur_refined = blur_core
    expand_px = int(np.ceil(float(settings["expand_microns"]) * px_per_um))
    if float(settings["expand_microns"]) > 0 and expand_px > 0 and blur_core.any():
        blur_refined = (cv2.dilate(blur_core.astype(np.uint8), _disk_kernel(expand_px)) > 0) & tissue_mask

    # --- flat bright artifacts inside tissue ---------------------------------
    bright_core = np.zeros_like(tissue_mask, dtype=bool)
    bright_mask = np.zeros_like(tissue_mask, dtype=bool)
    bright_breadth = None
    if bool(settings["detect_bright_artifacts"]):
        if bool(settings["use_breadth_signal"]) and channel_positive_count is not None and n_codex_channels > 0:
            bright_breadth = channel_positive_count.astype(np.float32) / float(n_codex_channels)
            breadth_mask = (bright_breadth >= float(settings["bright_breadth_fraction"])) & tissue_mask
        else:
            breadth_mask = np.zeros_like(tissue_mask, dtype=bool)

        window = (focus_window_px, focus_window_px)
        local_mean_codex = cv2.boxFilter(
            np.asarray(codex_projection, dtype=np.float32), ddepth=-1, ksize=window, borderType=cv2.BORDER_REFLECT
        )
        mean_values = local_mean_codex[tissue_mask]
        focus_values = codex_focus[tissue_mask]
        bright_mean_threshold = (
            float(np.percentile(mean_values, float(settings["bright_flat_mean_percentile"])))
            if mean_values.size else np.inf
        )
        flat_texture_threshold = (
            float(np.percentile(focus_values, float(settings["bright_flat_texture_percentile"])))
            if focus_values.size else -np.inf
        )
        # A field with no spread is bright and textureless everywhere by
        # definition: both percentile cuts collapse onto the one value present
        # and the test would take the whole tissue. There is nothing to find.
        intensity_spread = (
            float(np.percentile(mean_values, 99.0) - np.percentile(mean_values, 1.0))
            if mean_values.size else 0.0
        )
        if intensity_spread <= 1e-6:
            flat_glare = np.zeros_like(tissue_mask, dtype=bool)
        else:
            flat_glare = (
                (local_mean_codex >= bright_mean_threshold)
                & (codex_focus <= flat_texture_threshold)
                & tissue_mask
            )
        bright_u8 = (breadth_mask | flat_glare).astype(np.uint8)
        bright_close_px = int(round(length_px("bright_fill_holes_um")))
        bright_open_px = int(round(length_px("bright_remove_specks_um")))
        if bright_close_px > 0:
            bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_CLOSE, _disk_kernel(bright_close_px))
        if bright_open_px > 0:
            bright_u8 = cv2.morphologyEx(bright_u8, cv2.MORPH_OPEN, _disk_kernel(bright_open_px))
        bright_core = (bright_u8 > 0) & tissue_mask
        bright_core = _remove_small_components(bright_core, area_px("min_area_microns2_bright"), connectivity=8)
        bright_mask = bright_core
        bright_expand_px = int(np.ceil(float(settings["bright_expand_microns"]) * px_per_um))
        if float(settings["bright_expand_microns"]) > 0 and bright_expand_px > 0 and bright_core.any():
            bright_mask = (cv2.dilate(bright_core.astype(np.uint8), _disk_kernel(bright_expand_px)) > 0) & tissue_mask

    # --- H&E-external filamentous lint ---------------------------------------
    lint_mask = np.zeros_like(tissue_mask, dtype=bool)
    lint_seed_labels = np.zeros(tissue_mask.shape, dtype=np.int32)
    lint_records: list[dict[str, Any]] = []
    lint_shape_by_label: dict[int, dict[str, Any]] = {}
    if bool(settings["detect_filamentous_lint"]) and codex_robust_max is not None:
        exclusion_px = int(np.ceil(float(settings["lint_he_exclusion_margin_um"]) * px_per_um))
        tissue_exclusion = cv2.dilate(tissue_mask.astype(np.uint8), _disk_kernel(exclusion_px)) > 0
        lint_candidate = (codex_robust_max >= float(settings["lint_robust_z_threshold"])) & (~tissue_exclusion)
        if border_px > 0:
            lint_candidate[:border_px, :] = False
            lint_candidate[-border_px:, :] = False
            lint_candidate[:, :border_px] = False
            lint_candidate[:, -border_px:] = False

        candidate_u8 = lint_candidate.astype(np.uint8)
        lint_close_px = int(round(length_px("lint_join_gaps_um")))
        lint_open_px = int(round(length_px("lint_remove_specks_um")))
        if lint_close_px > 0:
            candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, _disk_kernel(lint_close_px))
        if lint_open_px > 0:
            candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, _disk_kernel(lint_open_px))

        count, labels, stats_array, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
        min_aspect = float(settings["lint_min_aspect_ratio"])
        for component_id in range(1, count):
            x = stats_array[component_id, cv2.CC_STAT_LEFT]
            y = stats_array[component_id, cv2.CC_STAT_TOP]
            component_width = stats_array[component_id, cv2.CC_STAT_WIDTH]
            component_height = stats_array[component_id, cv2.CC_STAT_HEIGHT]
            component_area_px = stats_array[component_id, cv2.CC_STAT_AREA]
            area_um2 = component_area_px * pixel_area_um2
            if area_um2 < float(settings["lint_min_area_microns2"]) or area_um2 > float(settings["lint_max_area_microns2"]):
                continue

            component_roi = labels[y:y + component_height, x:x + component_width] == component_id
            contours, _ = cv2.findContours(component_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            rect_width, rect_height = cv2.minAreaRect(contour)[1]
            major_axis = max(rect_width, rect_height, 1.0)
            minor_axis = max(min(rect_width, rect_height), 1.0)
            rotated_aspect = major_axis / minor_axis

            skeleton_length = float(skeletonize(component_roi).sum())
            mean_width_px = component_area_px / max(skeleton_length, 1.0)
            length_um = max(major_axis, skeleton_length) * pixel_size_um
            width_um = mean_width_px * pixel_size_um
            filamentarity = skeleton_length / max(mean_width_px, 1.0)
            horizontal_span_ratio = component_width / max(component_height, 1)
            seam_like = (
                horizontal_span_ratio >= float(settings["lint_seam_min_span_ratio"])
                and width_um <= float(settings["lint_seam_max_width_microns"])
            )
            accepted = (
                length_um >= float(settings["lint_min_length_microns"])
                and float(settings["lint_min_width_microns"]) <= width_um <= float(settings["lint_max_width_microns"])
                and rotated_aspect >= min_aspect
                and filamentarity >= min_aspect
                and not seam_like
            )
            measurements = {
                "component": int(component_id),
                "area_um2": float(area_um2),
                "length_um": float(length_um),
                "width_um": float(width_um),
                "aspect_ratio": float(rotated_aspect),
                "filamentarity": float(filamentarity),
                "horizontal_span_ratio": float(horizontal_span_ratio),
                "seam_like": bool(seam_like),
                "accepted": bool(accepted),
            }
            lint_records.append(measurements)
            if accepted:
                window_slice = (slice(y, y + component_height), slice(x, x + component_width))
                lint_mask[window_slice][component_roi] = True
                lint_seed_labels[window_slice][component_roi] = component_id
                lint_shape_by_label[int(component_id)] = measurements

        if float(settings["lint_expand_microns"]) > 0 and lint_mask.any():
            lint_expand_px = int(np.ceil(float(settings["lint_expand_microns"]) * px_per_um))
            if lint_expand_px > 0:
                lint_mask = cv2.dilate(lint_mask.astype(np.uint8), _disk_kernel(lint_expand_px)) > 0

    lint_components = pd.DataFrame(lint_records)
    if not lint_components.empty:
        lint_components = lint_components.sort_values(["accepted", "length_um"], ascending=[False, False])

    lint_in_tissue = lint_mask & tissue_mask
    artifact_mask = blur_refined | bright_mask | lint_in_tissue
    valid_tissue_mask = tissue_mask & (~artifact_mask)

    # --- per-region evidence -------------------------------------------------
    def tiles_for(labels_grid: np.ndarray | None, label_id: int, record: dict[str, Any]) -> np.ndarray:
        if labels_grid is None or grid_shape is None:
            return np.zeros((1, 1), dtype=bool)
        tiles = labels_grid == label_id
        if not tiles.any():
            # A region smaller than one attribution tile still deserves evidence:
            # fall back to the single tile holding its centroid.
            scale = float(max(int(downsample), 1))
            column = int(np.clip(record["centroid_full_res_px"][0] / scale / max(width, 1) * grid_shape[1], 0, grid_shape[1] - 1))
            row = int(np.clip(record["centroid_full_res_px"][1] / scale / max(height, 1) * grid_shape[0], 0, grid_shape[0] - 1))
            tiles = np.zeros(grid_shape, dtype=bool)
            tiles[row, column] = True
        return tiles

    def grid_labels_of(labels: np.ndarray) -> np.ndarray | None:
        if grid_shape is None:
            return None
        return cv2.resize(
            labels.astype(np.int32), (grid_shape[1], grid_shape[0]), interpolation=cv2.INTER_NEAREST
        )

    def region_identifier(artifact_type: str, label_id: int) -> str:
        stem = f"{sample_id}_" if sample_id else ""
        return f"{stem}{artifact_type}_{label_id:03d}"

    parameter_summary = {
        "blur_ratio_threshold": blur_threshold,
        "blur_reference_mode": mode,
        "blur_evidence_rule": rule if consensus_available else "pooled_only",
        "marker_consensus_fraction": float(settings.get("marker_consensus_fraction", 0.5)),
        "codex_signal_z": float(settings.get("codex_signal_z", 3.0)) if signal_support is not None else None,
        "expand_microns": float(settings["expand_microns"]),
        "min_area_microns2": float(settings["min_area_microns2"]),
        "microns_per_pixel_effective": round(pixel_size_um, 4),
    }

    blur_count, blur_labels, blur_regions = _component_records(blur_refined, pixel_size_um, downsample)
    blur_grid_labels = grid_labels_of(blur_labels)
    for record in blur_regions:
        region = blur_labels == record["label"]
        core = region & blur_core
        measured = core if core.any() else region
        ratio_values = reported_ratio[measured]
        median_ratio = float(np.median(ratio_values)) if ratio_values.size else float("nan")
        consensus_value = float(np.median(consensus_map[measured])) if consensus_map is not None and measured.any() else None
        tiles = tiles_for(blur_grid_labels, record["label"], record)
        candidates, markers_with_signal, markers_soft = _region_focus_evidence(
            tiles, marker_ratio_grid, signal_tiles, marker_names, marker_focus_threshold,
        )
        arms_available = int(bool(ratio_arms)) + int(consensus_available)
        arms_fired = int(bool(np.any(blur_from_ratio & measured))) + int(
            consensus_available and bool(np.any(blur_from_markers & measured))
        )
        # A region the per-marker test found but the pooled test did not would
        # otherwise report zero severity, which reads as "barely blurry" when it
        # is in fact "most of the markers here are out of focus".
        pooled_severity = (
            float(np.clip(1.0 - median_ratio / max(blur_threshold, 1e-9), 0.0, 1.0))
            if np.isfinite(median_ratio) else 0.0
        )
        marker_severity = float(consensus_value) if (consensus_value is not None and markers_soft) else 0.0
        severity = max(pooled_severity, marker_severity)
        record["measurements"] = {
            "area_um2": round(record["area_um2"], 1),
            "equivalent_diameter_um": round(record["equivalent_diameter_um"], 1),
            "median_focus_ratio": round(median_ratio, 4) if np.isfinite(median_ratio) else -1.0,
            "blur_severity": round(severity, 3),
            "marker_consensus": round(consensus_value, 3) if consensus_value is not None else -1.0,
            "markers_with_signal": int(markers_with_signal),
            "markers_out_of_focus": int(markers_soft),
            "evidence_arms_fired": int(arms_fired),
        }
        record["spacemap"] = {
            "schema": ARTIFACT_REGION_SCHEMA,
            "artifact_type": "blur",
            "region_id": region_identifier("blur", record["label"]),
            "sample_id": sample_id or None,
            "detector": "focus_ratio_and_marker_consensus",
            "area_um2": round(record["area_um2"], 1),
            "centroid_full_res_px": record["centroid_full_res_px"],
            "bbox_full_res_px": record["bbox_full_res_px"],
            "severity": round(severity, 3),
            "confidence": round(arms_fired / max(arms_available, 1), 3),
            "evidence": {
                "median_focus_ratio": round(median_ratio, 4) if np.isfinite(median_ratio) else None,
                "focus_ratio_threshold": round(marker_focus_threshold, 5),
                "pooled_threshold": round(float(blur_threshold), 5),
                "reference_mode": mode,
                "marker_consensus": round(consensus_value, 3) if consensus_value is not None else None,
                "markers_with_signal": int(markers_with_signal),
                "markers_out_of_focus": int(markers_soft),
                "pooled_severity": round(pooled_severity, 3),
                "marker_severity": round(marker_severity, 3),
                "arms_fired": int(arms_fired),
                "arms_available": int(arms_available),
            },
            "candidate_markers": candidates,
            "parameters": parameter_summary,
            "review": {"status": "pending", "decision": None, "reviewer": None, "notes": None},
        }
        record["name"] = f"Blur {record['label']} ({record['area_um2'] / 1e6:.3f} mm2)"

    bright_count, bright_labels, bright_regions = _component_records(bright_mask, pixel_size_um, downsample)
    bright_grid_labels = grid_labels_of(bright_labels)
    for record in bright_regions:
        region = bright_labels == record["label"]
        core = region & bright_core
        measured = core if core.any() else region
        tiles = tiles_for(bright_grid_labels, record["label"], record)
        candidates = _region_signal_evidence(tiles, channel_signal_grid, marker_names)
        breadth_value = float(np.median(bright_breadth[measured])) if bright_breadth is not None and measured.any() else None
        record["measurements"] = {
            "area_um2": round(record["area_um2"], 1),
            "equivalent_diameter_um": round(record["equivalent_diameter_um"], 1),
            "marker_breadth_fraction": round(breadth_value, 3) if breadth_value is not None else -1.0,
            "top_marker_signal_z": candidates[0]["signal_z"] if candidates else -1.0,
        }
        record["spacemap"] = {
            "schema": ARTIFACT_REGION_SCHEMA,
            "artifact_type": "bright",
            "region_id": region_identifier("bright", record["label"]),
            "sample_id": sample_id or None,
            "detector": "bright_and_textureless",
            "area_um2": round(record["area_um2"], 1),
            "centroid_full_res_px": record["centroid_full_res_px"],
            "bbox_full_res_px": record["bbox_full_res_px"],
            "evidence": {
                "brightness_percentile": float(settings["bright_flat_mean_percentile"]),
                "texture_percentile": float(settings["bright_flat_texture_percentile"]),
                "marker_breadth_fraction": round(breadth_value, 3) if breadth_value is not None else None,
            },
            "candidate_markers": candidates,
            "parameters": parameter_summary,
            "review": {"status": "pending", "decision": None, "reviewer": None, "notes": None},
        }
        record["name"] = f"Bright {record['label']} ({record['area_um2']:.0f} um2)"

    lint_count, lint_labels, lint_regions = _component_records(lint_mask, pixel_size_um, downsample)
    lint_grid_labels = grid_labels_of(lint_labels)
    for record in lint_regions:
        region = lint_labels == record["label"]
        seed_ids = [int(value) for value in np.unique(lint_seed_labels[region]) if value > 0]
        shapes = [lint_shape_by_label[value] for value in seed_ids if value in lint_shape_by_label]
        tiles = tiles_for(lint_grid_labels, record["label"], record)
        candidates = _region_signal_evidence(tiles, channel_signal_grid, marker_names)
        length_um = max((shape_record["length_um"] for shape_record in shapes), default=0.0)
        width_um = float(np.mean([shape_record["width_um"] for shape_record in shapes])) if shapes else 0.0
        aspect = max((shape_record["aspect_ratio"] for shape_record in shapes), default=0.0)
        filamentarity = max((shape_record["filamentarity"] for shape_record in shapes), default=0.0)
        record["measurements"] = {
            "area_um2": round(record["area_um2"], 1),
            "length_um": round(length_um, 1),
            "width_um": round(width_um, 2),
            "aspect_ratio": round(aspect, 2),
            "filamentarity": round(filamentarity, 2),
            "overlaps_tissue": float(bool(np.any(region & tissue_mask))),
            "top_marker_signal_z": candidates[0]["signal_z"] if candidates else -1.0,
        }
        record["spacemap"] = {
            "schema": ARTIFACT_REGION_SCHEMA,
            "artifact_type": "lint",
            "region_id": region_identifier("lint", record["label"]),
            "sample_id": sample_id or None,
            "detector": "external_filament_shape",
            "area_um2": round(record["area_um2"], 1),
            "centroid_full_res_px": record["centroid_full_res_px"],
            "bbox_full_res_px": record["bbox_full_res_px"],
            "evidence": {
                "length_um": round(length_um, 1),
                "width_um": round(width_um, 2),
                "aspect_ratio": round(aspect, 2),
                "filamentarity": round(filamentarity, 2),
                "seeded_outside_tissue": True,
                "overlaps_tissue": bool(np.any(region & tissue_mask)),
                "marker_count_in_grid": len(candidates),
            },
            "candidate_markers": candidates,
            "parameters": parameter_summary,
            "review": {"status": "pending", "decision": None, "reviewer": None, "notes": None},
        }
        record["name"] = f"Lint {record['label']} ({length_um:.0f} um)"

    tissue_pixels = max(int(tissue_mask.sum()), 1)
    evaluated_percent = 100.0 * int(evaluate_mask.sum()) / tissue_pixels
    return {
        "blur_mask": blur_refined,
        "blur_core_mask": blur_core,
        "bright_mask": bright_mask,
        "lint_mask": lint_mask,
        "lint_in_tissue": lint_in_tissue,
        "artifact_mask": artifact_mask,
        "valid_tissue_mask": valid_tissue_mask,
        "codex_focus": codex_focus,
        "blur_ratio_map": reported_ratio,
        "consensus_map": consensus_map,
        "evaluate_mask": evaluate_mask,
        "lint_components": lint_components,
        "region_labels": {"blur": blur_labels, "bright": bright_labels, "lint": lint_labels},
        "regions": {"blur": blur_regions, "bright": bright_regions, "lint": lint_regions},
        "parameters": parameter_summary,
        "stats": {
            "blurry_percent": 100.0 * int(blur_refined.sum()) / tissue_pixels,
            "bright_percent": 100.0 * int(bright_mask.sum()) / tissue_pixels,
            "lint_total_px": int(lint_mask.sum()),
            "lint_tissue_percent": 100.0 * int(lint_in_tissue.sum()) / tissue_pixels,
            "artifact_percent": 100.0 * int(artifact_mask.sum()) / tissue_pixels,
            "valid_percent": 100.0 * int(valid_tissue_mask.sum()) / tissue_pixels,
            "valid_pixels": int(valid_tissue_mask.sum()),
            "accepted_lint": int(lint_components["accepted"].sum()) if not lint_components.empty else 0,
            "rejected_seams": int(lint_components["seam_like"].sum()) if not lint_components.empty else 0,
            "blur_regions": len(blur_regions),
            "bright_regions": len(bright_regions),
            "lint_regions": len(lint_regions),
            "focus_evaluated_percent": evaluated_percent,
            "self_reference_focus": self_reference,
            "blur_threshold_used": round(float(blur_threshold), 5),
            "blur_threshold_mode": "auto" if automatic else "manual",
            "blur_population_found": bool(pooled_threshold is not None or consensus_threshold is not None),
            "blur_separation_gap": pooled_separation.get("gap"),
            "blur_separation_share": pooled_separation.get("share"),
            "blur_separation_required": float(settings["blur_min_separation"]),
            "marker_focus_threshold": round(marker_focus_threshold, 5),
            "blur_sensitivity": sensitivity,
            "marker_consensus_used": bool(consensus_available),
            "blur_reference_mode": mode,
        },
    }


# ==========================================================================
# Stage 3 - feature engineering, clustering and high-resolution export
# ==========================================================================

DEFAULT_CLUSTERING_CONFIG: dict[str, Any] = {
    "clustering_downsample": 4,
    "microns_per_pixel": 0.5083,
    "random_seed": 42,
    "arcsinh_mode": "dynamic",
    "arcsinh_cofactor": 5.0,
    "arcsinh_dynamic_k": 3.0,
    "arcsinh_dynamic_min": 5.0,
    "percentile_threshold": 99.9,
    "sigma_spatial_smoothing": 0.8,
    "use_top_k_masking": True,
    "top_k_mode": "auto",
    "top_k_markers": 5,
    "top_k_auto_min": 2,
    "top_k_auto_max": 10,
    "low_signal_handling": "include",
    "low_signal_quantile": 0.05,
    "n_clusters": 20,
    "kmeans_batch_size": 2048,
    "kmeans_n_init": 5,
    "kmeans_max_iter": 300,
    "kmeans_fit_sample": 0,
    # Size-aware balanced heatmap
    "size_shrinkage_prior_fraction": 0.25,
    "balanced_marker_quantile": 0.90,
    "balanced_min_z_scale": 0.50,
    "balanced_color_limit": 2.50,
    # High-resolution export
    "fine_export_smooth_sigma": 2.0,
    "fine_export_max_sigma": 6.0,
    "max_raw_polygons": 20000,
    "merged_valid_close_um": 4.0,
    "merged_valid_fill_holes_um2": 100.0,
}


def _find_elbow_k(values: np.ndarray, k_min: int, k_max: int) -> int:
    """Maximum-distance-to-chord elbow of a decaying scree curve."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    k_min = max(1, min(k_min, n))
    k_max = max(k_min, min(k_max, n))
    xs = np.arange(1, n + 1, dtype=np.float64)
    x_norm = (xs - xs[0]) / (xs[-1] - xs[0] + 1e-12)
    y_norm = (values - values.min()) / (values.max() - values.min() + 1e-12)
    line_vector = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    line_vector = line_vector / (np.linalg.norm(line_vector) + 1e-12)
    points = np.stack([x_norm - x_norm[0], y_norm - y_norm[0]], axis=1)
    projection_length = points @ line_vector
    projected = np.outer(projection_length, line_vector)
    distance = np.linalg.norm(points - projected, axis=1)
    distance[: k_min - 1] = -np.inf
    distance[k_max:] = -np.inf
    return int(np.argmax(distance)) + 1


def build_feature_stack(
    reader: PyramidImageReader,
    marker_indices: Sequence[int],
    valid_tissue_mask: np.ndarray,
    config: dict[str, Any],
    progress=None,
) -> dict[str, Any]:
    """Build the normalized Z-score feature matrix over valid tissue pixels.

    Steps 1.3-1.5 of the reference notebook: pyramid read at the clustering
    downsample, dynamic (per-marker) arcsinh, robust percentile scaling,
    optional spatial smoothing, Z-scoring across tissue pixels, and optional
    Top-K marker masking with a data-driven elbow estimate for K.

    The transformations run in place on one ``(C, H, W)`` float32 array so a
    35-40 channel ROI needs roughly a third of the peak memory of the notebook.
    """
    settings = {**DEFAULT_CLUSTERING_CONFIG, **(config or {})}
    factor = int(settings["clustering_downsample"])
    marker_indices = list(marker_indices)
    marker_names = [reader.channel_names[index] for index in marker_indices]

    image = reader.read_channels(marker_indices, factor, progress=progress)
    n_markers, height, width = image.shape
    if valid_tissue_mask.shape != (height, width):
        valid_tissue_mask = cv2.resize(
            valid_tissue_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    raw_for_cofactors = image  # cofactors are computed before the in-place transform
    cofactors: dict[str, float] = {}
    if str(settings["arcsinh_mode"]).lower() == "dynamic":
        for channel in range(n_markers):
            tissue_pixels = raw_for_cofactors[channel][valid_tissue_mask]
            cofactor = float(settings["arcsinh_cofactor"])
            if tissue_pixels.size > 100:
                p75 = np.percentile(tissue_pixels, 75)
                background = tissue_pixels[tissue_pixels <= p75]
                if background.size > 50:
                    mad = float(np.median(np.abs(background - np.median(background))))
                    sigma_bg = 1.4826 * mad
                    cofactor = max(float(settings["arcsinh_dynamic_min"]), float(settings["arcsinh_dynamic_k"]) * sigma_bg)
            cofactors[marker_names[channel]] = round(cofactor, 2)
    else:
        for name in marker_names:
            cofactors[name] = float(settings["arcsinh_cofactor"])

    raw_preview = {name: image[index].copy() for index, name in enumerate(marker_names[:4])}

    # Arcsinh + robust percentile scaling + spatial smoothing, in place.
    percentile = float(settings["percentile_threshold"])
    smoothing_sigma = float(settings["sigma_spatial_smoothing"])
    for channel in range(n_markers):
        plane = image[channel]
        np.arcsinh(plane / cofactors[marker_names[channel]], out=plane)
        channel_min = float(plane.min())
        channel_max = float(np.percentile(plane, percentile))
        if channel_max - channel_min > 1e-8:
            plane -= channel_min
            plane /= (channel_max - channel_min)
            np.clip(plane, 0.0, 1.0, out=plane)
        else:
            plane[:] = 0.0
        if smoothing_sigma > 0:
            image[channel] = gaussian(plane, sigma=smoothing_sigma, preserve_range=True).astype(np.float32)

    normalized_preview = {name: image[index].copy() for index, name in enumerate(marker_names[:4])}

    pixels = image.reshape(n_markers, -1).T
    valid_flat = valid_tissue_mask.reshape(-1)
    valid_indices = np.where(valid_flat)[0]
    if valid_indices.size == 0:
        raise ValueError("The reviewed valid-tissue mask is empty; nothing to cluster.")

    valid_pixels = pixels[valid_indices]
    pixels_z_valid = np.zeros_like(valid_pixels, dtype=np.float32)
    for channel in range(n_markers):
        values = valid_pixels[:, channel]
        mean, std = float(values.mean()), float(values.std())
        if std > 1e-8:
            pixels_z_valid[:, channel] = (values - mean) / std

    mean_sorted_profile = np.sort(pixels_z_valid, axis=1)[:, ::-1].mean(axis=0)
    auto_k = _find_elbow_k(
        mean_sorted_profile, int(settings["top_k_auto_min"]), int(settings["top_k_auto_max"])
    )

    if bool(settings["use_top_k_masking"]):
        effective_k = auto_k if str(settings["top_k_mode"]) == "auto" else int(settings["top_k_markers"])
        effective_k = int(min(max(effective_k, 1), n_markers))
        sorted_index = np.argsort(pixels_z_valid, axis=1)
        top_k_index = sorted_index[:, -effective_k:]
        rows = np.arange(pixels_z_valid.shape[0])[:, None]
        selection_mask = np.zeros_like(pixels_z_valid, dtype=bool)
        selection_mask[rows, top_k_index] = True
        features = np.zeros_like(pixels_z_valid)
        features[selection_mask] = pixels_z_valid[selection_mask]
    else:
        effective_k = int(settings["top_k_markers"])
        selection_mask = np.zeros_like(pixels_z_valid, dtype=bool)
        features = pixels_z_valid

    # Optional Studio features use the same normalized planes and scope statistics.
    spatial = settings.get("spatial_features", [])
    feature_names = list(marker_names)
    if spatial:
        from scipy.ndimage import uniform_filter, sobel
        extra = []
        for channel, name in enumerate(marker_names):
            plane = image[channel]
            for metric in spatial:
                if metric == "texture":
                    mean = uniform_filter(plane, size=5, mode="reflect")
                    field = np.sqrt(np.maximum(0, uniform_filter(plane * plane, size=5, mode="reflect") - mean * mean))
                elif metric == "edges":
                    field = np.hypot(sobel(plane, axis=0, mode="reflect"), sobel(plane, axis=1, mode="reflect"))
                else:
                    raise ValueError("Unknown classification feature.")
                column = field.ravel()[valid_indices]
                std = float(column.std())
                extra.append((column - column.mean()) / std if std > 1e-8 else np.zeros_like(column))
                feature_names.append(name + ":" + metric)
        features = np.column_stack([features, *extra]).astype(np.float32)

    return {
        "feature_names": feature_names,
        "marker_names": marker_names,
        "marker_indices": marker_indices,
        "shape": (height, width),
        "valid_tissue_mask": valid_tissue_mask,
        "valid_indices": valid_indices,
        "pixels_original_valid": valid_pixels,
        "pixels_z_valid": pixels_z_valid,
        "features": features,
        "selection_mask": selection_mask,
        "cofactors": cofactors,
        "mean_sorted_profile": mean_sorted_profile,
        "top_k_auto": int(auto_k),
        "top_k_effective": int(effective_k),
        "raw_preview": raw_preview,
        "normalized_preview": normalized_preview,
        "downsample": factor,
    }


def run_pixel_clustering(feature_stack: dict[str, Any], config: dict[str, Any], *, predictor=None) -> dict[str, Any]:
    """MiniBatchKMeans over valid-tissue pixel feature vectors (Step 2.1)."""
    settings = {**DEFAULT_CLUSTERING_CONFIG, **(config or {})}
    features = feature_stack["features"]
    n_clusters = int(settings["n_clusters"])
    seed = int(settings["random_seed"])

    if str(settings["low_signal_handling"]) == "reserve":
        mean_signal = features.mean(axis=1)
        threshold = float(np.quantile(mean_signal, float(settings["low_signal_quantile"])))
        low_signal_mask = mean_signal <= threshold
    else:
        low_signal_mask = np.zeros(len(features), dtype=bool)
    kmeans_mask = ~low_signal_mask
    fit_features = features[kmeans_mask]

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=int(settings["kmeans_batch_size"]),
        random_state=seed,
        n_init=int(settings["kmeans_n_init"]),
        max_iter=int(settings["kmeans_max_iter"]),
        verbose=0,
    )

    fit_sample = int(settings.get("kmeans_fit_sample", 0) or 0)
    if 0 < fit_sample < fit_features.shape[0]:
        rng = np.random.default_rng(seed)
        subset = rng.choice(fit_features.shape[0], size=fit_sample, replace=False)
        kmeans.fit(fit_features[subset])
        fitted_labels = predictor(fit_features, kmeans.cluster_centers_) if predictor else kmeans.predict(fit_features)
    else:
        kmeans.fit(fit_features)
        fitted_labels = predictor(fit_features, kmeans.cluster_centers_) if predictor else kmeans.predict(fit_features)

    labels_valid = np.empty(len(features), dtype=np.int32)
    labels_valid[kmeans_mask] = fitted_labels
    if str(settings["low_signal_handling"]) == "reserve":
        labels_valid[low_signal_mask] = n_clusters
        reserved_low_signal_cluster: int | None = n_clusters
    else:
        reserved_low_signal_cluster = None

    height, width = feature_stack["shape"]
    cluster_map = np.full((height, width), -1, dtype=np.int32)
    cluster_map.flat[feature_stack["valid_indices"]] = labels_valid

    palette = build_cluster_palette(n_clusters, include_low_signal=reserved_low_signal_cluster is not None)
    unique, counts = np.unique(labels_valid, return_counts=True)
    summary = pd.DataFrame({
        "cluster": unique,
        "pixels": counts,
        "tissue_percent": 100.0 * counts / max(len(labels_valid), 1),
    })

    return {
        "labels_valid": labels_valid,
        "cluster_map": cluster_map,
        "cluster_colors": palette,
        "reserved_low_signal_cluster": reserved_low_signal_cluster,
        "low_signal_mask": low_signal_mask,
        "summary": summary,
        "n_clusters": n_clusters,
    }


def build_cluster_profiles(
    feature_stack: dict[str, Any],
    clustering: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Cluster signatures: absolute intensity, mean Z, and size-aware balanced.

    The size-aware view shrinks unstable means from scarce clusters toward the
    tissue-wide marker center and rescales color contrast per marker, so a
    cluster covering 0.2% of the tissue cannot dominate the palette while a
    dominant cluster washes out.  This is the matrix the interactive merge
    heatmap uses; the annotations printed on it remain interpretable Z-scores.
    """
    settings = {**DEFAULT_CLUSTERING_CONFIG, **(config or {})}
    marker_names = feature_stack["marker_names"]
    labels = clustering["labels_valid"]

    absolute_profiles = (
        pd.DataFrame(feature_stack["pixels_original_valid"], columns=marker_names)
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )
    z_profiles = (
        pd.DataFrame(feature_stack["pixels_z_valid"], columns=marker_names)
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )

    cluster_sizes = (
        pd.Series(labels).value_counts().reindex(z_profiles.index).fillna(0).astype(np.int64)
    )
    cluster_fractions = cluster_sizes / max(int(cluster_sizes.sum()), 1)
    positive_sizes = cluster_sizes[cluster_sizes > 0]
    median_cluster_size = float(positive_sizes.median()) if not positive_sizes.empty else 1.0
    shrinkage_prior = max(1.0, median_cluster_size * float(settings["size_shrinkage_prior_fraction"]))
    reliability = cluster_sizes / (cluster_sizes + shrinkage_prior)

    marker_center = z_profiles.mul(cluster_sizes, axis=0).sum(axis=0) / max(float(cluster_sizes.sum()), 1.0)
    shrunk_profiles = marker_center + z_profiles.sub(marker_center, axis=1).mul(reliability, axis=0)
    shrunk_deviation = shrunk_profiles.sub(marker_center, axis=1)
    color_limit = float(settings["balanced_color_limit"])
    marker_scale = (
        shrunk_deviation.abs()
        .quantile(float(settings["balanced_marker_quantile"]), axis=0)
        .clip(lower=float(settings["balanced_min_z_scale"]))
    )
    balanced_scores = shrunk_deviation.divide(marker_scale, axis=1).clip(-color_limit, color_limit)

    size_summary = pd.DataFrame({
        "pixels": cluster_sizes,
        "tissue_percent": 100.0 * cluster_fractions,
        "reliability": reliability,
    })

    return {
        "absolute_profiles": absolute_profiles,
        "z_profiles": z_profiles,
        "shrunk_profiles": shrunk_profiles,
        "balanced_scores": balanced_scores,
        "size_summary": size_summary,
        "marker_center": marker_center,
        "color_limit": color_limit,
        "median_cluster_size": median_cluster_size,
        "shrinkage_prior": shrinkage_prior,
    }


def _smooth_label_partition(
    label_map: np.ndarray,
    tissue_mask: np.ndarray,
    class_ids: Sequence[int],
    sigma: float,
) -> np.ndarray:
    """Winner-take-all smoothing with bounded RAM and complete tissue coverage."""
    if sigma <= 0:
        result = label_map.astype(np.int16, copy=True)
        result[~tissue_mask] = -1
        return result

    best_scores = np.full(label_map.shape, -np.inf, dtype=np.float32)
    result = np.full(label_map.shape, -1, dtype=np.int16)
    for class_id in class_ids:
        class_mask = (label_map == class_id).astype(np.float32)
        score = cv2.GaussianBlur(class_mask, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        update = tissue_mask & (score > best_scores)
        best_scores[update] = score[update]
        result[update] = class_id
        del class_mask, score, update

    unresolved = tissue_mask & (result < 0)
    result[unresolved] = label_map[unresolved].astype(np.int16)
    result[~tissue_mask] = -1
    return result


def _partition_shapes(label_map: np.ndarray, class_name_by_id: dict[int, str]):
    """Polygonize the whole integer partition once so shared edges stay identical."""
    valid = label_map >= 0
    for geometry, value in rasterio.features.shapes(label_map.astype(np.int16), mask=valid, connectivity=8):
        class_id = int(value)
        if class_id in class_name_by_id:
            yield class_id, class_name_by_id[class_id], geometry


def _scaled_geometry(geometry, raster_scale: float):
    polygon = scale_geometry(shape(geometry), xfact=raster_scale, yfact=raster_scale, origin=(0, 0))
    return make_valid(polygon)


def export_cluster_partition(
    clustering: dict[str, Any],
    feature_stack: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | os.PathLike,
    sample_id: str,
    excluded_clusters: Sequence[int] = (),
) -> dict[str, Any]:
    """Export a gapless high-resolution cluster GeoJSON (Step 2.3).

    Class scores are Gaussian smoothed sequentially and resolved with
    winner-take-all assignment, so every valid-tissue pixel belongs to exactly
    one cluster and neighbouring polygons share identical edges.  Independent
    per-class vector polishing is deliberately not applied: it moves shared
    edges separately and opens cracks between classes.
    """
    settings = {**DEFAULT_CLUSTERING_CONFIG, **(config or {})}
    cluster_map = clustering["cluster_map"]
    valid_tissue_mask = feature_stack["valid_tissue_mask"]
    export_scale = int(settings["clustering_downsample"])
    microns_per_pixel = float(settings["microns_per_pixel"])
    colors = clustering["cluster_colors"]
    reserved = clustering["reserved_low_signal_cluster"]

    active_class_ids = sorted(int(value) for value in np.unique(cluster_map) if value >= 0)
    class_name_by_id = {
        class_id: ("Low_Signal_Tissue" if reserved is not None and class_id == reserved else f"Cluster_{class_id}")
        for class_id in active_class_ids
    }

    smoothing_sigma = float(settings["fine_export_smooth_sigma"])
    max_sigma = float(settings["fine_export_max_sigma"])
    max_polygons = int(settings["max_raw_polygons"])
    retries: list[str] = []
    while True:
        export_label_map = _smooth_label_partition(cluster_map, valid_tissue_mask, active_class_ids, smoothing_sigma)
        shape_records = list(_partition_shapes(export_label_map, class_name_by_id))
        if len(shape_records) <= max_polygons or smoothing_sigma >= max_sigma:
            break
        next_sigma = min(max_sigma, smoothing_sigma * 1.5)
        retries.append(f"{len(shape_records):,} fragments at sigma={smoothing_sigma:.2f}; retried at sigma={next_sigma:.2f}")
        smoothing_sigma = next_sigma

    uncovered = int(np.sum(valid_tissue_mask & (export_label_map < 0)))
    outside = int(np.sum((export_label_map >= 0) & (~valid_tissue_mask)))
    if uncovered or outside:
        raise RuntimeError(
            f"Cluster partition is not gapless within valid tissue (uncovered={uncovered:,}, outside={outside:,})."
        )

    polygons_by_class: dict[int, list] = defaultdict(list)
    for class_id, _class_name, geometry in shape_records:
        try:
            polygon = _scaled_geometry(geometry, export_scale)
            if not polygon.is_empty:
                polygons_by_class[class_id].append(polygon)
        except Exception:
            continue

    export_resolution_um = export_scale * microns_per_pixel
    features: list[dict] = []
    for class_id in active_class_ids:
        polygons = polygons_by_class.get(class_id, [])
        if not polygons:
            continue
        merged = make_valid(unary_union(polygons))
        if merged.is_empty:
            continue
        color_index = min(class_id, len(colors) - 1)
        class_name = class_name_by_id[class_id]
        features.append({
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": mapping(merged),
            "properties": {
                "objectType": "annotation",
                "classification": {"name": class_name, "colorRGB": int(qupath_color_from_hex(colors[color_index]))},
                "name": class_name,
                "clusterId": int(class_id),
                "exportResolutionMicrons": export_resolution_um,
                "isLocked": False,
            },
        })

    output_dir = str(output_dir)
    clusters_path = os.path.join(output_dir, f"{sample_id}_raw_macro_clusters.geojson")
    clusters_size_mb = write_feature_collection(features, clusters_path)

    # Reviewed valid tissue after removing clusters flagged as residual lint.
    excluded_ids = sorted({int(value) for value in excluded_clusters})
    merged_valid_mask = valid_tissue_mask & (~np.isin(cluster_map, excluded_ids))
    close_radius_px = int(np.ceil(float(settings["merged_valid_close_um"]) / (microns_per_pixel * export_scale)))
    if close_radius_px > 0:
        merged_valid_mask = cv2.morphologyEx(
            merged_valid_mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(close_radius_px).astype(np.uint8)
        ) > 0
        merged_valid_mask &= valid_tissue_mask
    fill_hole_pixels = int(np.ceil(float(settings["merged_valid_fill_holes_um2"]) / (microns_per_pixel * export_scale) ** 2))
    if fill_hole_pixels > 0:
        merged_valid_mask = remove_small_holes(merged_valid_mask, area_threshold=fill_hole_pixels, connectivity=1)
        merged_valid_mask &= valid_tissue_mask

    merged_shapes = [
        _scaled_geometry(geometry, export_scale)
        for geometry, value in rasterio.features.shapes(
            merged_valid_mask.astype(np.uint8), mask=merged_valid_mask, connectivity=8
        )
        if int(value) == 1
    ]
    merged_features: list[dict] = []
    if merged_shapes:
        merged_geometry = make_valid(unary_union(merged_shapes))
        if not merged_geometry.is_empty:
            merged_features.append({
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(merged_geometry),
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": "Valid_Tissue_From_Clusters", "colorRGB": -16711936},
                    "name": "HighRes_Valid_Tissue_From_Clusters",
                    "excludedClusterIds": excluded_ids,
                    "exportResolutionMicrons": export_resolution_um,
                    "isLocked": False,
                },
            })
    merged_valid_path = os.path.join(output_dir, f"{sample_id}_valid_tissue_from_clusters_highres.geojson")
    merged_valid_size_mb = write_feature_collection(merged_features, merged_valid_path)

    return {
        "export_label_map": export_label_map,
        "class_name_by_id": class_name_by_id,
        "clusters_path": clusters_path,
        "clusters_size_mb": clusters_size_mb,
        "merged_valid_path": merged_valid_path,
        "merged_valid_size_mb": merged_valid_size_mb,
        "smoothing_sigma": smoothing_sigma,
        "polygon_count": len(shape_records),
        "export_resolution_um": export_resolution_um,
        "excluded_clusters": excluded_ids,
        "retries": retries,
    }


def export_domain_partition(
    cluster_export: dict[str, Any],
    cluster_to_domain: dict[int, str],
    feature_stack: dict[str, Any],
    clustering: dict[str, Any],
    config: dict[str, Any],
    output_dir: str | os.PathLike,
    sample_id: str,
    domain_colors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Merge clusters into biological domains and export the final GeoJSON.

    The domain partition is derived from the exact smoothed cluster partition
    used in :func:`export_cluster_partition`, so domains inherit the same
    gapless shared boundaries instead of being re-derived at a different
    resolution.
    """
    settings = {**DEFAULT_CLUSTERING_CONFIG, **(config or {})}
    export_scale = int(settings["clustering_downsample"])
    microns_per_pixel = float(settings["microns_per_pixel"])

    base_partition = cluster_export["export_label_map"]
    domain_order: list[str] = []
    for cluster_id in sorted(cluster_to_domain):
        name = str(cluster_to_domain[cluster_id]).strip()
        if name and name not in domain_order:
            domain_order.append(name)
    if not domain_order:
        raise ValueError("No cluster-to-domain assignments were provided.")

    # Maximally distinct hues so neighbouring domains never look alike, while
    # any colour the user picked is preserved.
    colors = assign_distinct_colors(domain_order, existing=domain_colors, seed=int(settings["random_seed"]))

    domain_to_id = {name: index for index, name in enumerate(domain_order)}
    domain_id_to_name = {index: name for name, index in domain_to_id.items()}

    domain_label_map = np.full(base_partition.shape, -1, dtype=np.int16)
    for cluster_id, domain_name in cluster_to_domain.items():
        domain_name = str(domain_name).strip()
        if not domain_name:
            continue
        domain_label_map[base_partition == int(cluster_id)] = domain_to_id[domain_name]

    polygons_by_domain: dict[str, list] = defaultdict(list)
    for _domain_id, domain_name, geometry in _partition_shapes(domain_label_map, domain_id_to_name):
        try:
            polygon = _scaled_geometry(geometry, export_scale)
            if not polygon.is_empty:
                polygons_by_domain[domain_name].append(polygon)
        except Exception:
            continue

    features: list[dict] = []
    for domain_name in domain_order:
        polygons = polygons_by_domain.get(domain_name, [])
        if not polygons:
            continue
        merged = make_valid(unary_union(polygons))
        if merged.is_empty:
            continue
        features.append({
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": mapping(merged),
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": domain_name,
                    "colorRGB": int(qupath_color_from_hex(colors[domain_name])),
                },
                "name": domain_name,
                "exportResolutionMicrons": export_scale * microns_per_pixel,
                "isLocked": False,
            },
        })

    domains_path = os.path.join(str(output_dir), f"{sample_id}_macro_domains.geojson")
    size_mb = write_feature_collection(features, domains_path)

    marker_names = feature_stack["marker_names"]
    domain_labels_valid = np.array([
        str(cluster_to_domain.get(int(cluster), f"Cluster_{int(cluster)}"))
        for cluster in clustering["labels_valid"]
    ])
    domain_profiles = (
        pd.DataFrame(feature_stack["pixels_original_valid"], columns=marker_names)
        .assign(domain=domain_labels_valid)
        .groupby("domain")
        .mean()
    )
    domain_z_profiles = (
        pd.DataFrame(feature_stack["pixels_z_valid"], columns=marker_names)
        .assign(domain=domain_labels_valid)
        .groupby("domain")
        .mean()
    )

    return {
        "domains_path": domains_path,
        "size_mb": size_mb,
        "domain_label_map": domain_label_map,
        "domain_order": domain_order,
        "domain_colors": colors,
        "domain_profiles": domain_profiles,
        "domain_z_profiles": domain_z_profiles,
        "export_resolution_um": export_scale * microns_per_pixel,
    }


# ==========================================================================
# Stage 4 - pooling merged domains and buffer zones
# ==========================================================================

DEFAULT_POOLING_CONFIG: dict[str, Any] = {
    "simplify_tolerance_px": 0.0,
    "min_area_px": 0.0,
    "buffer_resolution": 3,
}


def _load_class_geometries(geojson_path: str | os.PathLike) -> dict[str, list]:
    """Group polygons in a GeoJSON by lower-cased classification name."""
    grouped: dict[str, list] = {}
    for feature in _iter_geojson_features(geojson_path):
        geometry_dict = feature.get("geometry")
        if not geometry_dict:
            continue
        try:
            geometry = shape(geometry_dict)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        grouped.setdefault(feature_class_name(feature), []).append(geometry)
    return grouped


def pool_geojson_classes(
    geojson_path: str | os.PathLike,
    pool_rules: Sequence[dict[str, Any]],
    output_path: str | os.PathLike,
    keep_source_classes: bool = True,
    config: dict[str, Any] | None = None,
    class_colors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Pool existing annotation classes into new named polygons.

    ``pool_rules`` is a list of ``{"name": str, "sources": [class names],
    "subtract": [class names]}`` records.  A rule that pools Epithelia, Stroma
    and Endothelia into ``tissue`` produces one extra polygon while the source
    classes are preserved when ``keep_source_classes`` is True.

    The source GeoJSON is always read from disk, never from cached state, so a
    QuPath refinement made between steps is respected.
    """
    settings = {**DEFAULT_POOLING_CONFIG, **(config or {})}
    grouped = _load_class_geometries(geojson_path)
    lookup = {name.lower(): geometries for name, geometries in grouped.items()}
    colors = dict(class_colors or {})

    features: list[dict] = []
    if keep_source_classes:
        features.extend(_iter_geojson_features(geojson_path))

    simplify_tolerance = float(settings["simplify_tolerance_px"])
    min_area = float(settings["min_area_px"])
    created: list[dict[str, Any]] = []

    for rule in pool_rules:
        name = str(rule.get("name", "")).strip()
        sources = [str(value).lower() for value in rule.get("sources", []) if str(value).strip()]
        subtract = [str(value).lower() for value in rule.get("subtract", []) if str(value).strip()]
        if not name or not sources:
            continue

        geometries: list = []
        for source in sources:
            geometries.extend(lookup.get(source, []))
        if not geometries:
            created.append({"name": name, "sources": sources, "area": 0.0, "status": "no source polygons"})
            continue

        merged = make_valid(unary_union(geometries))
        if subtract:
            subtract_geometries: list = []
            for source in subtract:
                subtract_geometries.extend(lookup.get(source, []))
            if subtract_geometries:
                merged = make_valid(merged.difference(make_valid(unary_union(subtract_geometries))))
        if simplify_tolerance > 0:
            merged = merged.simplify(simplify_tolerance, preserve_topology=True)
        if min_area > 0 and merged.geom_type == "MultiPolygon":
            parts = [part for part in merged.geoms if part.area >= min_area]
            merged = make_valid(unary_union(parts)) if parts else merged
        if merged.is_empty:
            created.append({"name": name, "sources": sources, "area": 0.0, "status": "empty after subtraction"})
            continue

        color_hex = colors.get(name, deterministic_class_hex_color(name))
        features.append({
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": mapping(merged),
            "properties": {
                "objectType": "annotation",
                "classification": {"name": name, "colorRGB": int(qupath_color_from_hex(color_hex))},
                "name": name,
                "pooledFrom": sources,
                "isLocked": False,
            },
        })
        # Make the pooled result available as a zone source in the same run.
        lookup[name.lower()] = [merged]
        created.append({"name": name, "sources": sources, "area": float(merged.area), "status": "created"})

    size_mb = write_feature_collection(features, output_path)
    return {"output_path": str(output_path), "size_mb": size_mb, "created": created, "features": features}


def generate_zone_features(
    geojson_path: str | os.PathLike,
    zone_rules: Sequence[dict[str, Any]],
    clip_class: str | None = None,
    config: dict[str, Any] | None = None,
) -> list[dict]:
    """Build inner/outer buffer zones around annotation classes.

    ``zone_rules`` entries look like ``{"label": "Epithelia", "width_px": 100,
    "inner": 2, "outer": 1}``.  Zones at the same level are merged across all
    source labels so that ``outer_1`` is a single, consistent annotation, which
    matches how the cell-annotation step consumes them.
    """
    settings = {**DEFAULT_POOLING_CONFIG, **(config or {})}
    resolution = int(settings["buffer_resolution"])
    grouped = _load_class_geometries(geojson_path)
    lookup = {name.lower(): geometries for name, geometries in grouped.items()}

    clip_geometry = None
    if clip_class:
        clip_polygons = lookup.get(str(clip_class).lower())
        if clip_polygons:
            clip_geometry = make_valid(unary_union([polygon.simplify(10.0) for polygon in clip_polygons]))

    active_rules = []
    for rule in zone_rules:
        label = str(rule.get("label", "")).strip()
        if not label or label.lower() not in lookup:
            continue
        if int(rule.get("inner", 0)) <= 0 and int(rule.get("outer", 0)) <= 0:
            continue
        active_rules.append(rule)
    if not active_rules:
        return []

    max_inner = max(int(rule.get("inner", 0)) for rule in active_rules)
    max_outer = max(int(rule.get("outer", 0)) for rule in active_rules)

    def source_union(label: str):
        polygons = [polygon.simplify(3.0) for polygon in lookup[label.lower()] if not polygon.is_empty]
        if not polygons:
            return None
        merged = make_valid(unary_union(polygons))
        return merged.simplify(2.0) if not merged.is_empty else None

    new_features: list[dict] = []

    for level in range(1, max_outer + 1):
        level_polygons = []
        for rule in active_rules:
            if level > int(rule.get("outer", 0)):
                continue
            merged = source_union(str(rule["label"]))
            if merged is None:
                continue
            width = float(rule.get("width_px", 100))
            current = merged.buffer(level * width, resolution=resolution).simplify(2.0)
            previous = merged if level == 1 else merged.buffer((level - 1) * width, resolution=resolution).simplify(2.0)
            zone = make_valid(current).difference(make_valid(previous))
            if clip_geometry is not None and not clip_geometry.is_empty:
                zone = make_valid(zone).intersection(clip_geometry)
            if not zone.is_empty:
                level_polygons.append(zone)
        if level_polygons:
            merged_level = make_valid(unary_union(level_polygons))
            if not merged_level.is_empty:
                name = f"outer_{level}"
                new_features.append({
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": mapping(merged_level),
                    "properties": {
                        "objectType": "annotation",
                        "classification": {"name": name, "colorRGB": int(qupath_color_from_hex(deterministic_class_hex_color(name)))},
                        "name": name,
                        "isLocked": False,
                    },
                })

    for level in range(1, max_inner + 1):
        level_polygons = []
        for rule in active_rules:
            if level > int(rule.get("inner", 0)):
                continue
            merged = source_union(str(rule["label"]))
            if merged is None:
                continue
            width = float(rule.get("width_px", 100))
            current = make_valid(merged.buffer(-level * width, resolution=resolution).simplify(2.0))
            previous = merged if level == 1 else make_valid(merged.buffer(-(level - 1) * width, resolution=resolution).simplify(2.0))
            if previous.is_empty:
                continue
            zone = previous if current.is_empty else make_valid(previous).difference(current)
            if not zone.is_empty:
                level_polygons.append(zone)
        if level_polygons:
            merged_level = make_valid(unary_union(level_polygons))
            if not merged_level.is_empty:
                name = f"inner_{level}"
                new_features.append({
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": mapping(merged_level),
                    "properties": {
                        "objectType": "annotation",
                        "classification": {"name": name, "colorRGB": int(qupath_color_from_hex(deterministic_class_hex_color(name)))},
                        "name": name,
                        "isLocked": False,
                    },
                })

    return new_features
