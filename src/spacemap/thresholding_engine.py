"""Deterministic intensity thresholds shared by Background, the viewer and gating.

ImageJ's 256-bin mapping is retained for classic compatibility. Inputs are never
silently filtered to positive values. Display transforms are a separate operation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

THRESHOLD_METHODS = ("Triangle", "Geometric elbow", "Otsu", "Yen", "Li", "IsoData", "Mean", "Minimum", "Percentile", "Manual")
METHOD_ALIASES = {"Li (Min Cross-Entropy)": "Li", "IsoData (Ridler-Calvard)": "IsoData", "Minimum (Bimodal Valley)": "Minimum"}
PARAMETER_DEFAULTS = {"percentile": 95.0, "manual": 0.0}
MIN_THRESHOLD = 5.0
_FALLBACK_THRESHOLD = 150.0
ENGINE_VERSION = "shared-imagej-1"


@dataclass(frozen=True)
class HistogramPopulation:
    """Lossless integer-value counts; no need to retain every image pixel."""
    values: np.ndarray
    counts: np.ndarray

    @property
    def nbytes(self):
        return self.values.nbytes + self.counts.nbytes

    @property
    def size(self):
        return int(self.counts.sum())


def _percentile(values, percentile, weights=None):
    if weights is None:
        return float(np.percentile(values, percentile))
    # NumPy's default linear quantile, evaluated at exact order statistics.
    position = (int(weights.sum()) - 1) * percentile / 100.0
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    cumulative = np.cumsum(weights, dtype=np.int64)
    a, b = values[np.searchsorted(cumulative, [lower, upper], side="right")].astype(float)
    return float(a + (b - a) * (position - lower))


def method_name(method):
    name = METHOD_ALIASES.get(method, method)
    if name not in THRESHOLD_METHODS:
        raise ValueError(f"Unknown threshold method: {method}")
    return name


def parameters(options=None):
    options = options or {}
    result = {key: float(options.get(key, default)) for key, default in PARAMETER_DEFAULTS.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Threshold parameters must be finite.")
    if not 0 <= result["percentile"] <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    return result


def catalog():
    return {"methods": list(THRESHOLD_METHODS), "parameters": dict(PARAMETER_DEFAULTS), "engine": ENGINE_VERSION}


def _imagej_bilevel(histogram: np.ndarray) -> int | None:
    """ImageJ ``AutoThresholder.bilevel`` with its default subtract-one rule."""
    occupied = np.flatnonzero(histogram > 0)
    if occupied.size == 1:
        return int(occupied[0]) - 1
    if occupied.size == 2:
        return int(occupied[1]) - 1
    return None


def _imagej_triangle(histogram: np.ndarray) -> int:
    """Exact port of ImageJ's 256-bin ``AutoThresholder.Triangle``."""
    data = np.asarray(histogram, dtype=np.int64).copy()
    bilevel = _imagej_bilevel(data)
    if bilevel is not None and bilevel >= 0:
        return bilevel

    nonzero = np.flatnonzero(data > 0)
    if not nonzero.size:
        return 0
    minimum = max(int(nonzero[0]) - 1, 0)
    maximum2 = min(int(nonzero[-1]) + 1, data.size - 1)
    maximum = int(np.argmax(data))

    inverted = (maximum - minimum) < (maximum2 - maximum)
    if inverted:
        data = data[::-1].copy()
        minimum = data.size - 1 - maximum2
        maximum = data.size - 1 - maximum

    if minimum == maximum:
        split = minimum
    else:
        nx = float(data[maximum])
        ny = float(minimum - maximum)
        norm = float(np.hypot(nx, ny))
        if norm == 0:
            split = minimum
        else:
            nx /= norm
            ny /= norm
            distance0 = nx * minimum + ny * float(data[minimum])
            split = minimum
            split_distance = 0.0
            for index in range(minimum + 1, maximum + 1):
                distance = nx * index + ny * float(data[index]) - distance0
                if distance > split_distance:
                    split = index
                    split_distance = distance
            split -= 1
    return data.size - 1 - split if inverted else split


def _imagej_li(histogram: np.ndarray) -> int:
    """Port ImageJ's histogram-based Li implementation."""
    data = np.asarray(histogram, dtype=np.float64)
    bins = np.arange(data.size, dtype=np.float64)
    total = float(data.sum())
    if total <= 0:
        return 0
    new_threshold = float(np.dot(bins, data) / total)
    threshold = int(new_threshold + 0.5)
    for _ in range(10_000):
        old_threshold = new_threshold
        threshold = int(old_threshold + 0.5)
        threshold = min(max(threshold, 0), data.size - 1)
        background = data[: threshold + 1]
        foreground = data[threshold + 1 :]
        mean_background = (
            float(np.dot(bins[: threshold + 1], background) / background.sum())
            if background.sum() else 0.0
        )
        mean_foreground = (
            float(np.dot(bins[threshold + 1 :], foreground) / foreground.sum())
            if foreground.sum() else 0.0
        )
        if mean_background <= 0 or mean_foreground <= 0 or mean_background == mean_foreground:
            return threshold
        temp = ((mean_background - mean_foreground) /
                (np.log(mean_background) - np.log(mean_foreground)))
        new_threshold = float(int(temp - 0.5) if temp < -np.finfo(float).eps else int(temp + 0.5))
        if abs(new_threshold - old_threshold) <= 0.5:
            return threshold
    raise RuntimeError("Li threshold did not converge after 10,000 iterations")


def _imagej_histogram(
    values: np.ndarray,
    value_range: tuple[float, float] | None,
    imagej_dtype,
    weights=None,
) -> tuple[np.ndarray, float, float]:
    """Convert pixels to the 8-bit histogram used by ``ImageProcessor``.

    QuPath's extracted CODEX channels are unsigned 16-bit ``ShortProcessor``
    images. ImageJ first resets the display range to the request rectangle's
    actual minimum and maximum, then converts to byte with the same rounding as
    ``ShortProcessor.create8BitImage``.
    """
    minimum, maximum = value_range or (float(values.min()), float(values.max()))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        minimum, maximum = float(values.min()), float(values.max())
    histogram = np.zeros(256, dtype=np.int64)
    if maximum <= minimum:
        histogram[0] = values.size if weights is None else int(weights.sum())
    else:
        dtype = np.dtype(imagej_dtype) if imagej_dtype is not None else values.dtype
        if dtype.kind == "u" and dtype.itemsize == 2:
            scale = 256.0 / (maximum - minimum + 1.0)
        else:
            scale = 255.0 / (maximum - minimum)
        # Chunking avoids a second multi-gigabyte conversion array for a
        # full-resolution tissue ROI.
        for start in range(0, values.size, 10_000_000):
            block = values[start : start + 10_000_000]
            converted = np.clip(
                (block - minimum) * scale + 0.5, 0, 255
            ).astype(np.uint8)
            histogram += np.bincount(converted, minlength=256,
                weights=None if weights is None else weights[start:start + 10_000_000]).astype(np.int64)
    return histogram, float(minimum), float(maximum)


def _imagej_threshold_bin(histogram: np.ndarray, method: str) -> int:
    """Threshold a 256-bin histogram, matching ImageJ for the default methods."""
    from skimage import filters

    bilevel = _imagej_bilevel(histogram)
    if bilevel is not None and bilevel >= 0:
        return bilevel
    if method == "Triangle":
        return _imagej_triangle(histogram)
    if method == "Mean":
        total = int(histogram.sum())
        return int(np.dot(np.arange(256, dtype=np.int64), histogram) // total) if total else 0
    if method == "Li":
        return _imagej_li(histogram)

    centers = np.arange(256, dtype=np.float64)
    histogram_input = (histogram.astype(np.float64), centers)
    if method == "Otsu":
        return int(filters.threshold_otsu(hist=histogram_input))
    if method == "Yen":
        return int(filters.threshold_yen(hist=histogram_input))
    if method == "IsoData":
        return int(filters.threshold_isodata(hist=histogram_input))
    if method == "Minimum":
        return int(filters.threshold_minimum(hist=histogram_input))
    raise ValueError(f"Unknown threshold method: {method}")


def _legacy_estimate(
    pixels: np.ndarray,
    method: str = "Geometric elbow",
    *,
    value_range: tuple[float, float] | None = None,
    imagej_dtype=None,
    weights=None,
) -> dict:
    """Estimate one marker's background level from its pixel values.

    Returns the threshold plus the smoothed histogram and its peak, so the UI
    can show what the decision was based on instead of only the number.
    """
    from scipy.ndimage import gaussian_filter1d
    original = np.asarray(pixels)
    values = original.ravel()
    if np.issubdtype(values.dtype, np.floating):
        all_finite = all(
            np.isfinite(values[start : start + 10_000_000]).all()
            for start in range(0, values.size, 10_000_000)
        )
        if not all_finite:
            values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "threshold": _FALLBACK_THRESHOLD,
            "peak": 0.0,
            "histogram": np.array([]),
            "method": method,
            "n_pixels": 0,
            "fallback_status": "empty_pixels_fallback_150",
            "threshold_error": "No finite pixels were available.",
        }

    fallback_status = "none"
    threshold_error = ""
    smooth = np.array([], dtype=float)
    peak = 0.0
    if method == "Geometric elbow":
        # A clipped histogram is appropriate for the custom elbow only; the
        # ImageJ methods below deliberately use the complete 256-bin range.
        p99 = _percentile(values, 99, weights)
        keep = values <= max(p99, 10.0)
        clipped = values[keep]
        counts = (
            np.bincount(np.clip(clipped, 0, None).astype(np.int64), weights=None if weights is None else weights[keep])
            if clipped.size else np.array([0])
        )
        smooth = gaussian_filter1d(counts.astype(float), sigma=2.0)
        peak = float(np.argmax(smooth)) if smooth.size else 0.0
        # Walk from the noise peak to the tail and take the point furthest from
        # the straight line joining them: where the noise mode stops falling
        # steeply and the real signal begins.
        tail = smooth.size - 1
        if tail <= peak:
            threshold = max(peak + 1.0, MIN_THRESHOLD)
        else:
            x = np.arange(int(peak), tail + 1, dtype=float)
            y = smooth[int(peak): tail + 1]
            xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
            yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
            p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
            span = np.linalg.norm(p2 - p1)
            if span <= 0:
                threshold = max(float(peak) + 1.0, MIN_THRESHOLD)
            else:
                dist = np.abs(np.cross(p2 - p1, p1 - np.stack([xn, yn], axis=1))) / span
                threshold = max(float(x[int(np.argmax(dist))]), MIN_THRESHOLD)
    else:
        # The Groovy reference calls ImageProcessor.setAutoThreshold(), which
        # thresholds a display-scaled 8-bit histogram and maps the selected bin
        # back into the original intensity range. Running skimage directly on
        # raw float pixels is not equivalent and was the main source of the GUI
        # discrepancy.
        try:
            imagej_histogram, range_minimum, range_maximum = _imagej_histogram(
                values, value_range, imagej_dtype or original.dtype, weights=weights
            )
            threshold_bin = _imagej_threshold_bin(imagej_histogram, method)
            threshold = (
                range_minimum
                if range_maximum <= range_minimum
                else range_minimum + (float(threshold_bin) / 255.0) *
                (range_maximum - range_minimum)
            )
            smooth = gaussian_filter1d(imagej_histogram.astype(float), sigma=2.0)
            peak_bin = int(np.argmax(smooth)) if smooth.size else 0
            peak = (
                range_minimum
                if range_maximum <= range_minimum
                else range_minimum + (float(peak_bin) / 255.0) *
                (range_maximum - range_minimum)
            )
        except Exception as error:
            threshold = _FALLBACK_THRESHOLD
            fallback_status = "method_error_fallback_150"
            threshold_error = f"{type(error).__name__}: {error}"

    return {
        "threshold": float(threshold),
        "peak": peak,
        "histogram": smooth,
        "method": method,
        "n_pixels": int(values.size) if weights is None else int(weights.sum()),
        "fallback_status": fallback_status,
        "threshold_error": threshold_error,
    }



def calculate(values, method="Triangle", *, params=None, value_range=None, imagej_dtype=None):
    """Return a raw cutoff and provenance, or fail explicitly; never apply a fallback.

    The same values, dtype, method and parameters produce the same result in every
    caller. Pixel and cell-mean populations are intentionally not interchangeable.
    """
    method = method_name(method)
    params = parameters(params)
    weights = values.counts if isinstance(values, HistogramPopulation) else None
    original = values.values if weights is not None else np.asarray(values)
    data = original.reshape(-1)
    if weights is None:
        data = data[np.isfinite(data)]
    if not data.size:
        raise ValueError("No finite measurements in the selected population.")
    low, high = float(data.min()), float(data.max())
    if method == "Manual":
        threshold = params["manual"]
    elif method == "Percentile":
        threshold = _percentile(data, params["percentile"], weights)
    elif low == high:
        threshold = low
    else:
        # The classic elbow uses integer intensity bins. Bound memory and preserve
        # corrected/negative floating point measurements by an invertible linear
        # mapping. No nonlinear display transform participates in estimation.
        scaled = method == "Geometric elbow" and (low < 0 or high > 65535 or not np.issubdtype(data.dtype, np.integer))
        work = (data - low) * (65535.0 / (high - low)) if scaled else data
        found = _legacy_estimate(work, method, value_range=value_range, imagej_dtype=imagej_dtype or original.dtype, weights=weights)
        if found["fallback_status"] != "none":
            raise ValueError(f"{method} could not estimate this population: {found['threshold_error']}")
        threshold = float(found["threshold"])
        if scaled:
            threshold = low + threshold * (high - low) / 65535.0
    if not np.isfinite(threshold):
        raise ValueError(f"{method} returned a non-finite cutoff.")
    return {"threshold": float(threshold), "method": method, "n_pixels": int(data.size) if weights is None else int(weights.sum()),
            "parameters": params, "engine": ENGINE_VERSION,
            "fallback_status": "none", "threshold_error": ""}


def estimate_background(values, method="Geometric elbow", *, params=None, value_range=None, imagej_dtype=None):
    """Classic export adapter; retained diagnostics explicitly flag legacy fallbacks."""
    method = method_name(method)
    try:
        result = calculate(values, method, params=params, value_range=value_range, imagej_dtype=imagej_dtype)
    except ValueError as error:
        finite = np.asarray(values).reshape(-1)
        finite = finite[np.isfinite(finite)]
        return {"threshold": _FALLBACK_THRESHOLD, "peak": 0.0, "histogram": np.array([]),
                "method": method, "n_pixels": int(finite.size), "threshold_error": str(error),
                "fallback_status": "method_error_fallback_150" if finite.size else "empty_pixels_fallback_150"}
    # Plot diagnostics need not run the threshold method a second time.
    finite = np.asarray(values).reshape(-1)
    finite = finite[np.isfinite(finite)]
    counts, edges = np.histogram(finite, bins=256)
    result.update(histogram=counts, peak=float((edges[np.argmax(counts)] + edges[np.argmax(counts)+1])/2))
    return result


def project(values, scale="Linear", cofactor=5.0):
    """Display coordinates only. Log10 omits non-positive values (no hidden offset)."""
    data = np.asarray(values, dtype=float)
    if scale == "Linear":
        return data
    if scale == "Log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(data > 0, np.log10(data), np.nan)
    if scale == "Arcsinh":
        if not np.isfinite(cofactor) or cofactor <= 0:
            raise ValueError("Arcsinh cofactor must be positive and finite.")
        return np.arcsinh(data / cofactor)
    raise ValueError("Choose Linear, Log10 or Arcsinh.")
