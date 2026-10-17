"""Render viewer tiles from an :class:`~spacemap.viewer.sources.ImageSource`.

The viewer asks for square tiles addressed like a map: ``(z, x, y)`` where the
tile covers ``TILE * 2**z`` full-resolution pixels per side (``z = 0`` is full
resolution). For each tile the renderer picks the pyramid level closest to the
requested resolution without being coarser, reads only that region, blends the
chosen channels in their colours and contrast, and encodes PNG for lossless
detail or JPEG for compact overviews.

Raw channel tiles are kept in a memory cache, so changing a colour or the
contrast re-renders from memory instead of reading the file again.
"""

from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .sources import ImageInfo, open_source

TILE = 512
_CACHE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ChannelStyle:
    index: int
    color: tuple[float, float, float]
    low: float
    high: float
    gamma: float = 1.0


def parse_color(value: str) -> tuple[float, float, float]:
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Colour must be six hex digits, got {value!r}")
    return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def max_zoom(info: ImageInfo) -> int:
    """The coarsest z at which the whole image fits in one tile."""
    longest = max(info.width, info.height)
    return max(0, math.ceil(math.log2(max(longest / TILE, 1))))


def tile_bounds(info: ImageInfo, z: int, x: int, y: int) -> tuple[int, int, int, int, int, int]:
    """Full-resolution region of a tile and its output size (clipped at edges)."""
    scale = 2**z
    x0, y0 = x * TILE * scale, y * TILE * scale
    x1, y1 = min(info.width, x0 + TILE * scale), min(info.height, y0 + TILE * scale)
    out_w = max(0, math.ceil((x1 - x0) / scale))
    out_h = max(0, math.ceil((y1 - y0) / scale))
    return x0, y0, x1, y1, out_w, out_h


def _pick_level(info: ImageInfo, scale: float) -> int:
    """Coarsest pyramid level whose resolution is at least what the tile needs."""
    best = 0
    for index, level in enumerate(info.levels):
        if max(level.downsample_x, level.downsample_y) <= scale * 1.001:
            best = index
    return best


# ---------------------------------------------------------------------------
# Raw-tile memory cache (shared by all threads)
# ---------------------------------------------------------------------------


class _TileCache:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.used = 0
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key, value: np.ndarray) -> None:
        with self._lock:
            if key in self._items:
                return
            self._items[key] = value
            self.used += value.nbytes
            while self.used > self.budget and self._items:
                _, old = self._items.popitem(last=False)
                self.used -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.used = 0


RAW_CACHE = _TileCache(_CACHE_BYTES)


def _resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    if array.shape[1] == width and array.shape[0] == height:
        return array
    try:
        import cv2

        return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        if array.ndim == 2 and array.dtype == np.uint16:
            image = Image.fromarray(array.astype(np.int32), mode="I")
            return np.asarray(image.resize((width, height), Image.BOX)).astype(np.uint16)
        return np.asarray(Image.fromarray(array).resize((width, height), Image.BOX))


def raw_tile(path: str, info: ImageInfo, z: int, x: int, y: int, channel: int) -> np.ndarray:
    """One channel (or RGB) of a tile at the tile's own resolution."""
    key = (path, info.fingerprint, z, x, y, channel)
    cached = RAW_CACHE.get(key)
    if cached is not None:
        return cached
    x0, y0, x1, y1, out_w, out_h = tile_bounds(info, z, x, y)
    shape = (out_h, out_w, 3) if info.rgb else (out_h, out_w)
    if out_w == 0 or out_h == 0:
        return np.zeros(shape, dtype=info.dtype)
    level_index = _pick_level(info, 2**z)
    level = info.levels[level_index]
    source = open_source(path)
    region = source.read_region(
        level_index,
        channel,
        math.floor(x0 / level.downsample_x),
        math.floor(y0 / level.downsample_y),
        math.ceil(x1 / level.downsample_x),
        math.ceil(y1 / level.downsample_y),
    )
    region = _resize(np.ascontiguousarray(region), out_w, out_h)
    RAW_CACHE.put(key, region)
    return region


def composite(tiles: list[np.ndarray], styles: list[ChannelStyle]) -> np.ndarray:
    """Additive blend of single-channel tiles into an RGB uint8 image."""
    if not tiles:
        raise ValueError("No channels to draw.")
    height, width = tiles[0].shape[:2]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    for data, style in zip(tiles, styles):
        span = max(float(style.high) - float(style.low), 1e-6)
        value = (data.astype(np.float32) - float(style.low)) / span
        np.clip(value, 0.0, 1.0, out=value)
        if style.gamma != 1.0:
            np.power(value, 1.0 / style.gamma, out=value)
        rgb += value[..., None] * np.asarray(style.color, dtype=np.float32)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def checker_tile(jpeg: bytes, info: ImageInfo, z: int, x: int, y: int, square: int) -> bytes:
    """*jpeg* as a PNG with every other *square* (full-resolution pixels) see-through: a checkerboard over the image
    below, to compare a registered image with the one it was aligned to."""
    import io

    from PIL import Image

    x0, y0, _, _, width, height = tile_bounds(info, z, x, y)
    scale = 2**z
    with Image.open(io.BytesIO(jpeg)) as picture:
        rgba = np.array(picture.convert("RGBA"))
    height, width = rgba.shape[:2]
    cols = ((x0 + (np.arange(width) + 0.5) * scale) // square).astype(np.int64)
    rows = ((y0 + (np.arange(height) + 0.5) * scale) // square).astype(np.int64)
    rgba[..., 3] = np.where((rows[:, None] + cols[None, :]) % 2 == 0, 255, 0).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(out, format="PNG", compress_level=3)
    return out.getvalue()


def render_tile(
    path: str, info: ImageInfo, z: int, x: int, y: int, styles: list[ChannelStyle] | None, *, quality: int = 92,
    lossless: bool = False,
) -> bytes:
    """A finished display tile. PNG preserves the composed RGB pixels exactly.

    This affects display encoding only; measurements read the original channels.
    RGB images ignore *styles*.
    """
    encode = encode_png if lossless else lambda image: encode_jpeg(image, quality=quality)
    if info.rgb:
        image = raw_tile(path, info, z, x, y, 0)
        if image.dtype != np.uint8:
            image = np.clip(image / max(float(image.max()), 1.0) * 255.0, 0, 255).astype(np.uint8)
    else:
        if styles == []:  # explicitly all markers off; do not silently display channel zero
            *_, width, height = tile_bounds(info, z, x, y)
            return encode(np.zeros((height, width, 3), dtype=np.uint8))
        styles = styles or []
        valid = [style for style in styles if 0 <= style.index < info.channel_count]
        if not valid:
            valid = [
                ChannelStyle(
                    0,
                    (1.0, 1.0, 1.0),
                    0.0,
                    float(np.iinfo(info.dtype).max) if np.issubdtype(np.dtype(info.dtype), np.integer) else 1.0,
                )
            ]
        tiles = [raw_tile(path, info, z, x, y, style.index) for style in valid]
        image = composite(tiles, valid)
    return encode(image)


def encode_jpeg(image: np.ndarray, *, quality: int = 92) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    if image.size == 0:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
    # Colour subsampling smears small coloured features even at high quality.
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def encode_png(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    if image.size == 0:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Suggested contrast
# ---------------------------------------------------------------------------

_STATS: dict = {}
_STATS_LOCK = threading.Lock()


def channel_limits(path: str, info: ImageInfo, channel: int) -> tuple[float, float]:
    """Suggested (low, high) display limits: 1st and 99.8th percentile of tissue.

    Computed once per channel from the largest pyramid level at most 3,000 px
    on the long side, so it costs a fraction of a second even on whole slides.
    """
    key = (path, info.fingerprint, channel)
    with _STATS_LOCK:
        if key in _STATS:
            return _STATS[key]
    level_index = len(info.levels) - 1
    for index, level in enumerate(info.levels):
        if max(level.width, level.height) <= 3000:
            level_index = index
            break
    level = info.levels[level_index]
    data = open_source(path).read_region(level_index, channel, 0, 0, level.width, level.height)
    # A regular sample of at most 4 million pixels, taken before any conversion:
    # converting a large ROI's whole level (30 million pixels for PDAC's s8r1) cost seconds.
    flat = data.reshape(-1)
    values = flat[:: max(1, flat.size // 4_000_000)].astype(np.float32)
    nonzero = values[values > 0]
    sample = nonzero if nonzero.size > 1000 else values
    if sample.size > 2_000_000:
        sample = sample[:: sample.size // 2_000_000 + 1]
    low = float(np.percentile(sample, 1.0)) if sample.size else 0.0
    high = float(np.percentile(sample, 99.8)) if sample.size else 1.0
    if high <= low:  # flat channel (a mask, or one bright value): show its whole range
        low = min(float(values.min()), 0.0) if values.size else 0.0
        high = max(float(values.max()) if values.size else 1.0, low + 1.0)
    with _STATS_LOCK:
        _STATS[key] = (low, high)
    return low, high


_HISTOGRAMS: OrderedDict = OrderedDict()
_REGION_PIXELS = 2048  # longest side a region is read at for its histogram


def _shape_list(region) -> list:
    """*region* as a list of Shapely geometries (None or empty: the whole image)."""
    if region is None:
        return []
    return [g for g in (region if isinstance(region, (list, tuple)) else [region]) if g is not None and not g.is_empty]


def _region_mask(shapes, out_shape, transform):
    """One deterministic windowed rasterization convention for every population.

    GDAL's all-touched edge handling can depend on clipping at window bounds.
    Use the same 1,024-pixel windows for dense and streamed reads, including holes.
    """
    import rasterio.features
    from affine import Affine
    from shapely.geometry import box
    mask = np.zeros(out_shape, dtype=bool)
    for y in range(0, out_shape[0], 1024):
        for x in range(0, out_shape[1], 1024):
            height, width = min(1024, out_shape[0]-y), min(1024, out_shape[1]-x)
            window_transform = transform * Affine.translation(x,y)
            left, top = window_transform * (0,0)
            right, bottom = window_transform * (width,height)
            bounds = box(left,top,right,bottom)
            nearby = [g for g in shapes if g.intersects(bounds)]
            if nearby:
                mask[y:y+height,x:x+width] = rasterio.features.rasterize(
                    [(g,1) for g in nearby], out_shape=(height,width),
                    transform=window_transform, fill=0, dtype="uint8", all_touched=True).astype(bool)
    return mask


def marker_pixels(path: str, info: ImageInfo, channel: int, *, region=None, native=False) -> tuple[np.ndarray, int]:
    """One marker's pixels and the pyramid level they come from.

    The whole image is read from its largest level of at most 3,000 px on the
    long side; *region* (one or more Shapely geometries in full-resolution
    pixels) from the finest level at which the region is at most 2,048 px, so
    a small region is measured at full resolution.
    """
    source = open_source(path)
    shapes = _shape_list(region)
    if not shapes:
        level_index = len(info.levels) - 1
        for index, level in enumerate(info.levels):
            if max(level.width, level.height) <= 3000:
                level_index = index
                break
        if native:
            level_index = 0
        level = info.levels[level_index]
        _check_population_size(level.width, level.height)
        return source.read_region(level_index, channel, 0, 0, level.width, level.height).reshape(-1), level_index
    import rasterio.features
    from affine import Affine

    bounds = np.array([g.bounds for g in shapes])
    x0, y0 = (int(math.floor(v)) for v in bounds[:, :2].min(axis=0))
    x1, y1 = (int(math.floor(v)) for v in bounds[:, 2:].max(axis=0))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(info.width, x1 + 1), min(info.height, y1 + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("The shape is outside the image.")
    level_index = 0
    for index, level in enumerate(info.levels):
        level_index = index
        if max((x1 - x0) / level.downsample_x, (y1 - y0) / level.downsample_y) <= _REGION_PIXELS:
            break
    if native:
        level_index = 0
    level = info.levels[level_index]
    dx, dy = level.downsample_x, level.downsample_y
    lx0, ly0 = int(x0 / dx), int(y0 / dy)
    lx1, ly1 = max(lx0 + 1, int(math.ceil(x1 / dx))), max(ly0 + 1, int(math.ceil(y1 / dy)))
    _check_population_size(lx1 - lx0, ly1 - ly0)
    data = source.read_region(level_index, channel, lx0, ly0, lx1, ly1)
    inside = _region_mask(shapes, data.shape[:2], Affine(dx, 0, lx0 * dx, 0, dy, ly0 * dy))
    values = data[inside]
    if values.size == 0:
        raise ValueError("No pixels inside the shape.")
    return values, level_index


_POPULATIONS = _TileCache(96 * 1024 * 1024)
_BACKGROUNDS: OrderedDict = OrderedDict()


def _check_population_size(width, height):
    if width * height > 64_000_000:
        raise ValueError("This population exceeds 64 million uncompressed pixels. Full resolution streams 8/16-bit integer images; for floating-point images choose a smaller annotation region or a coarser pyramid.")


def _integer_counts(values, dtype):
    limits = np.iinfo(dtype)
    data = np.asarray(values, dtype=np.int64).ravel() - limits.min
    return np.bincount(data, minlength=limits.max - limits.min + 1)


def _compact_population(counts, dtype):
    from spacemap.thresholding_engine import HistogramPopulation
    occupied = np.flatnonzero(counts)
    return HistogramPopulation((occupied + np.iinfo(dtype).min).astype(dtype), counts[occupied])


def _native_population(path, info, channel, shapes):
    """Read native uint8/uint16/int16 pixels once in bounded windows.

    Counts are lossless, including zeros and every pixel inside polygon masks.
    No image-sized plane, flattened copy or float display array is allocated.
    """
    import rasterio.features
    from affine import Affine
    from shapely.geometry import box
    dtype = np.dtype(info.dtype)
    limits = np.iinfo(dtype)
    counts = np.zeros(limits.max - limits.min + 1, dtype=np.int64)
    source = open_source(path)
    x0 = y0 = 0
    x1, y1 = info.width, info.height
    if shapes:
        bounds = np.array([g.bounds for g in shapes])
        x0, y0 = np.maximum(0, np.floor(bounds[:, :2].min(axis=0))).astype(int)
        x1, y1 = np.minimum([info.width, info.height], np.floor(bounds[:, 2:].max(axis=0)) + 1).astype(int)
    for y in range(y0, y1, 1024):
        for x in range(x0, x1, 1024):
            right, bottom = min(x + 1024, x1), min(y + 1024, y1)
            inside = None
            if shapes:
                window = box(x, y, right, bottom)
                nearby = [g for g in shapes if g.intersects(window)]
                if not nearby:
                    continue
                inside = _region_mask(nearby, (bottom-y,right-x), Affine.translation(x,y))
                if not inside.any():
                    continue
            data = source.read_region(0, channel, x, y, right, bottom)
            counts += _integer_counts(data[inside] if inside is not None else data, dtype)
    if not counts.any():
        raise ValueError("No pixels inside the selected image region.")
    return _compact_population(counts, dtype)


def threshold_population(path, info, channel, *, region=None, native=False):
    import hashlib
    shapes = _shape_list(region)
    geometry = hashlib.sha1(b"".join(g.wkb for g in shapes)).hexdigest() if shapes else ""
    key = (path, info.fingerprint, channel, geometry, bool(native))
    entry = _POPULATIONS.get(key)
    if entry is None:
        dtype = np.dtype(info.dtype)
        compact = dtype.kind in "iu" and dtype.itemsize <= 2
        if native and compact:
            values, level = _native_population(path, info, channel, shapes), 0
        else:
            values, level = marker_pixels(path, info, channel, region=shapes, native=native)
            if compact:
                values = _compact_population(_integer_counts(values,dtype),dtype)
        # Cache a level index beside the array without changing its dtype.
        entry = _Population(values, level)
        _POPULATIONS.put(key, entry)
    return entry.values, entry.level, key


@dataclass
class _Population:
    values: np.ndarray
    level: int

    @property
    def nbytes(self):
        return self.values.nbytes


def background_level(path: str, info: ImageInfo, channel: int, method: str = "Triangle", *, region=None, params=None, native=False) -> dict:
    """Shared raw threshold on a named pixel population; failed methods never apply."""
    from spacemap.thresholding_engine import calculate, parameters, method_name, ENGINE_VERSION
    params = parameters(params)
    method = method_name(method)
    values, level_index, population_key = threshold_population(path, info, channel, region=region, native=native)
    key = (*population_key, ENGINE_VERSION, method, tuple(params.items()))
    with _STATS_LOCK:
        if key in _BACKGROUNDS:
            _BACKGROUNDS.move_to_end(key)
            return _BACKGROUNDS[key]
    found = calculate(values, method, params=params)
    result = {**found, "pixels": found["n_pixels"],
              "downsample": float(info.levels[level_index].downsample_x),
              "region": bool(_shape_list(region)), "fallback": "none"}
    with _STATS_LOCK:
        _BACKGROUNDS[key] = result
        while len(_BACKGROUNDS) > 1024:
            _BACKGROUNDS.popitem(last=False)
    return result


def threshold_benchmark(path, info, channel, methods, **options):
    if not methods:
        raise ValueError("Choose at least one threshold method.")
    return _pixel_histograms(path, info, channel, methods, **options)


def _pixel_histograms(path, info, channel, methods=(), *, region=None, params=None, native=False, scale="Linear", cofactor=5.0, bins=256):
    """Whole-image and annotation histograms on a common display axis.

    Cutoffs always use untransformed samples. Nonpositive pixels count toward
    estimation but are omitted from Log10 plots, and explicitly counted.
    """
    from spacemap.thresholding_engine import project, method_name, parameters, ENGINE_VERSION, HistogramPopulation
    methods = list(dict.fromkeys(method_name(m) for m in methods))
    params = parameters(params)
    populations = [("Whole image", None)]
    if _shape_list(region):
        populations.append(("Annotation", region))
    samples = []
    for name, shapes in populations:
        values, level, _ = threshold_population(path, info, channel, region=shapes, native=native)
        weights = values.counts if isinstance(values, HistogramPopulation) else None
        finite = values.values if weights is not None else values[np.isfinite(values)]
        total = int(finite.size) if weights is None else int(weights.sum())
        shown = project(finite, scale, cofactor)
        valid = np.isfinite(shown)
        plotted = shown[valid]
        plotted_weights = None if weights is None else weights[valid]
        plotted_count = plotted.size if weights is None else int(plotted_weights.sum())
        gates = []
        for method in methods:
            try:
                result = background_level(path, info, channel, method, region=shapes, params=params, native=native)
                value = float(project(result["threshold"], scale, cofactor))
                gates.append({**result, "coordinate": value if np.isfinite(value) else None,
                              "positive_fraction": float(np.mean(finite > result["threshold"])) if weights is None else float(weights[finite > result["threshold"]].sum() / total)})
            except ValueError as error:
                gates.append({"method": method, "error": str(error)})
        samples.append({"name": name, "values": plotted, "weights": plotted_weights, "pixels": total,
                        "omitted": int(total - plotted_count), "thresholds": gates,
                        "downsample": float(info.levels[level].downsample_x)})
    nonempty = [s["values"] for s in samples if s["values"].size]
    low = min(float(v.min()) for v in nonempty) if nonempty else 0.0
    high = max(float(v.max()) for v in nonempty) if nonempty else 1.0
    # The measured range defines the bins in both viewers. A manual cutoff
    # outside that range must not reshape the distribution or hide its detail.
    if high <= low:
        high = low + 1.0
    edges = np.linspace(low, high, bins + 1)
    for sample in samples:
        sample["counts"] = np.histogram(sample.pop("values"), bins=edges, weights=sample.pop("weights"))[0].tolist()
    return {"channel": channel, "scale": scale, "cofactor": cofactor, "axis": [low, high],
            "edges": edges.tolist(), "populations": samples, "engine": ENGINE_VERSION,
            "parameters": params, "native": bool(native)}


def channel_histogram(path: str, info: ImageInfo, channel: int, *, region=None, bins=256,
                      native=False, scale="Linear", cofactor=5.0) -> dict:
    """The Image Layer and benchmark share populations, display bins and range.

    All finite pixels are counted, with no bright-tail clipping. Statistics and
    suggested brightness limits remain in raw units, regardless of plot scale.
    """
    from spacemap.thresholding_engine import HistogramPopulation, _percentile

    values, _, population_key = threshold_population(path, info, channel, region=region, native=native)
    key = (*population_key, bins, scale, cofactor)
    with _STATS_LOCK:
        if key in _HISTOGRAMS:
            _HISTOGRAMS.move_to_end(key)
            return _HISTOGRAMS[key]
    distribution = _pixel_histograms(path, info, channel, region=region, native=native,
                                     scale=scale, cofactor=cofactor, bins=bins)
    population = distribution["populations"][-1]
    weights = values.counts if isinstance(values, HistogramPopulation) else None
    finite = values.values if weights is not None else values[np.isfinite(values)]
    vmin, vmax = float(finite.min()), float(finite.max())
    nonzero = finite > 0
    count = int(nonzero.sum()) if weights is None else int(weights[nonzero].sum())
    sample = finite[nonzero] if count > 1000 else finite
    sample_weights = weights[nonzero] if weights is not None and count > 1000 else weights
    low, high = [_percentile(sample, q, sample_weights) for q in (1.0, 99.8)]
    if high <= low:
        low, high = min(vmin, 0.0), max(vmax, min(vmin, 0.0) + 1.0)
    result = {**population, "axis": distribution["axis"], "edges": distribution["edges"],
              "scale": scale, "cofactor": cofactor, "native": bool(native),
              "min": vmin, "max": vmax, "mean": float(np.average(finite, weights=weights)),
              "median": _percentile(finite, 50, weights), "suggested": [low, high]}
    with _STATS_LOCK:
        _HISTOGRAMS[key] = result
        while len(_HISTOGRAMS) > 256:
            _HISTOGRAMS.popitem(last=False)
    return result


def auto_display_range(path, info, channel, *, region=None, native=False, mode="both",
                       low=0.0, method="Triangle", params=None, upper_percentile=99.8):
    """Display saturation from pixels above the raw low level; never a cell gate.

    No qualifying pixels is an explicit error, not an invented low+1 range.
    Integer populations use exact weighted quantiles without expanding pixels.
    """
    from spacemap.thresholding_engine import HistogramPopulation, _percentile

    if mode not in ("both", "upper"):
        raise ValueError("Display adjustment must be both or upper.")
    if not np.isfinite(upper_percentile) or not 0 < upper_percentile <= 100:
        raise ValueError("Upper percentile must be greater than 0 and at most 100.")
    if mode == "both":
        found = background_level(path, info, channel, method, region=region, params=params, native=native)
        low = found["threshold"]
    if not np.isfinite(low):
        raise ValueError("The lower display level must be finite.")
    values, level, _ = threshold_population(path, info, channel, region=region, native=native)
    weights = values.counts if isinstance(values, HistogramPopulation) else None
    finite = values.values if weights is not None else values[np.isfinite(values)]
    positive = finite > low
    if not positive.any():
        raise ValueError("No pixels above the lower level in this area. Lower that level or choose another area.")
    sample, sample_weights = finite[positive], None if weights is None else weights[positive]
    high = _percentile(sample, upper_percentile, sample_weights)
    return {"low": float(low), "high": high, "upper_percentile": upper_percentile,
            "signal_pixels": int(positive.sum()) if weights is None else int(sample_weights.sum()),
            "pixels": int(finite.size) if weights is None else int(weights.sum()),
            "downsample": float(info.levels[level].downsample_x), "mode": mode,
            "display_only_upper": True}
