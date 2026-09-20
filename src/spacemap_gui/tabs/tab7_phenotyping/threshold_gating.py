"""
Threshold Gating Subtab
Interactive flow cytometry-style gating for marker thresholding.
Compatible with SCIMAP and Resolve phenotyping workflows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import (
    logger,
    Timer,
    get_marker_list,
    project_root_calc,
    _normalize_marker_name,
    _load_tab7_state,
    _update_tab7_state,
    AUTO_THRESHOLD_METHODS,
    get_marker_workspace_panel,
    render_method_marker_selection,
)

try:
    from streamlit_plotly_events import plotly_events
    _HAS_PLOTLY_EVENTS = True
except ImportError:
    plotly_events = None
    _HAS_PLOTLY_EVENTS = False

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except ImportError:
    gaussian_kde = None
    _HAS_SCIPY = False

try:
    from scipy.optimize import brentq as _brentq
except ImportError:
    _brentq = None


# ==============================================================================
# Logicle (Biexponential) Transform — Gold Standard for Flow Cytometry
# ==============================================================================

# Module-level LUT cache for Logicle — avoids recomputing brentq and the 8K-point
# lookup table on every transform call.
_logicle_lut_cache: dict[tuple[float, float, float, float], tuple[np.ndarray, np.ndarray, tuple]] = {}


def _logicle_biexp_params(T: float, W: float, M: float = 4.5, A: float = 0.0):
    """Compute biexponential parameters a, b, c, d, f from Logicle T, W, M, A.

    The biexponential function is  B(y) = a·e^(b·y) − c·e^(−d·y) + f
    and the Logicle is its inverse: S(x) = B⁻¹(x).
    """
    w = W / (M + A)
    x2 = A / (M + A)
    x1 = x2 + w
    x0 = x2 + 2 * w
    b = (M + A) * np.log(10)

    # Solve for d: 2·(ln d − ln b) + b·(w + x2) − d·(w + x2) = 0
    if _brentq is not None:
        def _f(d_val):
            return 2.0 * (np.log(d_val) - np.log(b)) + b * (w + x2) - d_val * (w + x2)
        d = _brentq(_f, 1e-12, b, xtol=1e-14)
    else:
        # Newton approx fallback
        d = b
        for _ in range(200):
            fd = 2.0 * (np.log(d) - np.log(b)) + b * (w + x2) - d * (w + x2)
            fd_prime = 2.0 / d - (w + x2)
            d -= fd / fd_prime
            if abs(fd) < 1e-14:
                break

    c_a = np.exp(x0 * (b + d))
    f_val = -np.exp(b * x1) + c_a * np.exp(-d * x1)
    a = T / (np.exp(b) - c_a * np.exp(-d) + f_val)
    c = c_a * a
    f = f_val * a
    return a, b, c, d, f


def _get_logicle_lut(T: float, W: float, M: float = 4.5, A: float = 0.0,
                     n_lut: int = 8192) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Return (x_grid, y_grid, params) for the Logicle LUT, cached."""
    key = (T, W, M, A)
    if key in _logicle_lut_cache:
        return _logicle_lut_cache[key]

    params = _logicle_biexp_params(T, W, M, A)
    a, b, c, d, f = params
    y_lo = -A / (M + A) if (M + A) > 0 else 0.0
    y_grid = np.linspace(y_lo, 1.0, n_lut)
    x_grid = a * np.exp(b * y_grid) - c * np.exp(-d * y_grid) + f
    result = (x_grid, y_grid, params)
    _logicle_lut_cache[key] = result
    return result


def _logicle_forward(values: np.ndarray, T: float, W: float, M: float = 4.5,
                     A: float = 0.0, n_lut: int = 8192) -> np.ndarray:
    """Fast Logicle (biexponential) transform via interpolation lookup table."""
    x_grid, y_grid, _ = _get_logicle_lut(T, W, M, A, n_lut)
    return np.interp(values, x_grid, y_grid).astype(np.float64)


def _logicle_inverse_scalar(y: float, T: float, W: float, M: float = 4.5,
                            A: float = 0.0) -> float:
    """Inverse Logicle for a single y value (display → linear)."""
    _, _, (a, b, c, d, f) = _get_logicle_lut(T, W, M, A)
    return float(a * np.exp(b * y) - c * np.exp(-d * y) + f)


def estimate_logicle_params(values: np.ndarray):
    """Auto-estimate Logicle T and W from data.

    T  = 1.1 × max |x|  (top of scale)
    W  = estimated from the spread of the near-zero population
         so the linear region covers the noise floor.
    """
    clean = values[np.isfinite(values)]
    if len(clean) < 20:
        return 262144.0, 0.5  # safe defaults

    T = float(np.max(np.abs(clean))) * 1.1
    if T < 1.0:
        T = 1.0

    # Estimate W from the negative or near-zero population
    neg = clean[clean < 0]
    if len(neg) > 10:
        r = float(np.percentile(np.abs(neg), 95))
        W = (4.5 - np.log10(T / max(r, 1e-10))) / 2.0
        W = max(0.1, min(W, 2.0))
    else:
        # No negatives; estimate from noise floor (bottom 10 %)
        low = clean[clean < np.percentile(clean, 10)]
        if len(low) > 5:
            spread = float(np.std(low))
            W = max(0.1, min(2.0 * spread / T * 4.5, 2.0))
        else:
            W = 0.5
    return T, W


def estimate_arcsinh_cofactor(values: np.ndarray) -> float:
    """Auto-estimate Arcsinh cofactor from data noise floor.

    Uses the median absolute deviation (MAD) of the low-intensity
    population as a robust noise estimator.  The cofactor is set so
    that noise sits in the linear region of arcsinh (≈ [−1, 1]).
    """
    clean = values[np.isfinite(values)]
    if len(clean) < 20:
        return 5.0

    # Use bottom 50 % as "noise"
    median_val = float(np.median(clean))
    low = clean[clean <= median_val]
    if len(low) > 10:
        mad = float(np.median(np.abs(low - np.median(low))))
        cofactor = max(mad * 1.4826, 0.01)  # MAD → σ conversion
    else:
        cofactor = max(float(np.percentile(np.abs(clean), 5)), 0.01)
    return round(cofactor, 4)


# ==============================================================================
# Data Transformations
# ==============================================================================

class DataTransform:
    """Data transformation utilities for visualization with inverse for thresholds.

    Each entry carries:
        forward(values, cofactor_or_params) → transformed array
        inverse(t, cofactor_or_params)      → original-scale scalar
        description                         → tooltip for the UI
        needs_cofactor                      → show cofactor spinner
        needs_logicle                       → show T / W spinners
        auto_cofactor                       → auto-estimate cofactor from data
    """

    TRANSFORMS: dict[str, dict[str, Any]] = {
        "Linear": {
            "forward": lambda x, cofactor: x,
            "inverse": lambda t, cofactor: t,
            "description": "No transformation (original scale)",
            "needs_cofactor": False,
            "needs_logicle": False,
        },
        "Logicle": {
            # Forward & inverse are special-cased (they need T & W, not a scalar cofactor).
            # Placeholder lambdas – overridden in transform() / inverse_transform().
            "forward": None,
            "inverse": None,
            "description": (
                "Biexponential (Parks et al.). Gold standard in flow cytometry — "
                "linear near zero, logarithmic for high values.  Best for gating."
            ),
            "needs_cofactor": False,
            "needs_logicle": True,
        },
        "Arcsinh": {
            "forward": lambda x, cofactor: np.arcsinh(x / cofactor),
            "inverse": lambda t, cofactor: np.sinh(t) * cofactor,
            "description": "Inverse hyperbolic sine — standard for CyTOF (cofactor ≈ 5).",
            "needs_cofactor": True,
            "needs_logicle": False,
        },
        "Arcsinh (auto)": {
            "forward": lambda x, cofactor: np.arcsinh(x / cofactor),
            "inverse": lambda t, cofactor: np.sinh(t) * cofactor,
            "description": (
                "Arcsinh with auto-estimated cofactor from your data's noise floor. "
                "Better separation than fixed cofactor for imaging data."
            ),
            "needs_cofactor": False,   # auto-estimated
            "needs_logicle": False,
            "auto_cofactor": True,
        },
        "Log10": {
            "forward": lambda x, cofactor: np.log10(np.maximum(x, 1e-10)),
            "inverse": lambda t, cofactor: np.power(10, t),
            "description": "Log base 10 (clips negatives)",
            "needs_cofactor": False,
            "needs_logicle": False,
        },
        "Log1p": {
            "forward": lambda x, cofactor: np.log1p(np.maximum(x, 0)),
            "inverse": lambda t, cofactor: np.expm1(t),
            "description": "Log(1 + x) — handles zeros gracefully",
            "needs_cofactor": False,
            "needs_logicle": False,
        },
        "Sqrt": {
            "forward": lambda x, cofactor: np.sqrt(np.maximum(x, 0)),
            "inverse": lambda t, cofactor: t ** 2,
            "description": "Square root",
            "needs_cofactor": False,
            "needs_logicle": False,
        },
    }

    @classmethod
    def get_names(cls) -> list[str]:
        return list(cls.TRANSFORMS.keys())

    @classmethod
    def transform(cls, values: np.ndarray, method: str, cofactor: float = 5.0,
                  logicle_T: float | None = None, logicle_W: float | None = None) -> np.ndarray:
        if method == "Logicle":
            T = logicle_T if logicle_T is not None else 262144.0
            W = logicle_W if logicle_W is not None else 0.5
            return _logicle_forward(values, T, W)
        if method == "Arcsinh (auto)":
            # cofactor should already be auto-estimated before this call
            return np.arcsinh(values / cofactor)
        if method not in cls.TRANSFORMS or cls.TRANSFORMS[method]["forward"] is None:
            return values
        return cls.TRANSFORMS[method]["forward"](values, cofactor)

    @classmethod
    def inverse_transform(cls, threshold: float, method: str, cofactor: float = 5.0,
                          logicle_T: float | None = None, logicle_W: float | None = None) -> float:
        if method == "Logicle":
            T = logicle_T if logicle_T is not None else 262144.0
            W = logicle_W if logicle_W is not None else 0.5
            return _logicle_inverse_scalar(threshold, T, W)
        if method == "Arcsinh (auto)":
            return float(np.sinh(threshold) * cofactor)
        if method not in cls.TRANSFORMS or cls.TRANSFORMS[method]["inverse"] is None:
            return threshold
        return float(cls.TRANSFORMS[method]["inverse"](threshold, cofactor))

    @classmethod
    def get_description(cls, method: str) -> str:
        return cls.TRANSFORMS.get(method, {}).get("description", "")

    @classmethod
    def needs_cofactor(cls, method: str) -> bool:
        return cls.TRANSFORMS.get(method, {}).get("needs_cofactor", False)

    @classmethod
    def needs_logicle(cls, method: str) -> bool:
        return cls.TRANSFORMS.get(method, {}).get("needs_logicle", False)

    @classmethod
    def has_auto_cofactor(cls, method: str) -> bool:
        return cls.TRANSFORMS.get(method, {}).get("auto_cofactor", False)


# ==============================================================================
# Visualization Functions
# ==============================================================================

# -- Cached heavy computations (survive across Streamlit reruns) ---------------
# We use a manual dict cache in session_state.  This is faster than
# @st.cache_data for our use case because:
#   • No serialisation/hashing overhead for large numpy arrays.
#   • Survives across reruns within the same session (like st.cache_data).
#   • Instant dict-key lookup (string) vs blake2b hash of 150K floats.
# The cache key encodes: marker(s), batch, transform, cofactor/logicle params.
# The cache is automatically invalidated when data is reloaded (load_btn resets it).

_CACHE_KEY = "thresh_compute_cache"   # session_state key for the compute cache
_CACHE_MAX = 128                       # max entries before evicting oldest


def _get_cache() -> dict[str, Any]:
    if _CACHE_KEY not in st.session_state:
        st.session_state[_CACHE_KEY] = {}
    return st.session_state[_CACHE_KEY]


def _clear_cache() -> None:
    st.session_state[_CACHE_KEY] = {}


def _cache_get(key: str) -> Any:
    return _get_cache().get(key)


def _cache_put(key: str, value: Any) -> None:
    cache = _get_cache()
    if len(cache) >= _CACHE_MAX:
        # Evict oldest entry (first key in dict — insertion order)
        oldest = next(iter(cache))
        del cache[oldest]
    cache[key] = value


def _make_kde_key(marker: str, batch: str, n: int,
                  transform: str, cofactor: float,
                  logicle_T: float | None, logicle_W: float | None) -> str:
    return f"kde|{marker}|{batch}|{n}|{transform}|{cofactor}|{logicle_T}|{logicle_W}"


def _make_scatter_key(mx: str, my: str, batch: str, n: int,
                      transform: str, cofactor: float,
                      logicle_T: float | None, logicle_W: float | None) -> str:
    return f"scat|{mx}|{my}|{batch}|{n}|{transform}|{cofactor}|{logicle_T}|{logicle_W}"


def _make_param_key(kind: str, marker: str, n: int) -> str:
    return f"param|{kind}|{marker}|{n}"


def _compute_kde(
    values: np.ndarray,
    transform_method: str,
    cofactor: float,
    logicle_T: float | None,
    logicle_W: float | None,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute KDE x_grid and density."""
    clean = values[~np.isnan(values) & ~np.isinf(values)]
    if len(clean) < 10:
        return None

    transformed = DataTransform.transform(clean, transform_method, cofactor,
                                          logicle_T=logicle_T, logicle_W=logicle_W)
    transformed = transformed[~np.isnan(transformed) & ~np.isinf(transformed)]
    if len(transformed) < 10:
        return None

    x_min = float(np.percentile(transformed, 0.5))
    x_max = float(np.percentile(transformed, 99.5))
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, n_points)

    if _HAS_SCIPY and gaussian_kde is not None:
        try:
            kde = gaussian_kde(transformed, bw_method="scott")
            density = kde(x_grid)
        except Exception:
            hist, bin_edges = np.histogram(transformed, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x_grid, bin_centers, hist)
    else:
        hist, bin_edges = np.histogram(transformed, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = np.interp(x_grid, bin_centers, hist)

    return x_grid, density


def _compute_scatter_data(
    values_x: np.ndarray,
    values_y: np.ndarray,
    transform_method: str,
    cofactor: float,
    logicle_T: float | None,
    logicle_W: float | None,
    max_points: int = 10000,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute transformed scatter points."""
    mask = ~np.isnan(values_x) & ~np.isinf(values_x) & ~np.isnan(values_y) & ~np.isinf(values_y)
    x_clean = values_x[mask]
    y_clean = values_y[mask]
    if len(x_clean) < 10:
        return None
    if len(x_clean) > max_points:
        idx = np.random.choice(len(x_clean), max_points, replace=False)
        x_clean = x_clean[idx]
        y_clean = y_clean[idx]
    x_t = DataTransform.transform(x_clean, transform_method, cofactor,
                                  logicle_T=logicle_T, logicle_W=logicle_W)
    y_t = DataTransform.transform(y_clean, transform_method, cofactor,
                                  logicle_T=logicle_T, logicle_W=logicle_W)
    return x_t, y_t


def _data_fingerprint(marker: str, batch: str, n: int) -> str:
    """Quick fingerprint for cache keys (avoids hashing entire arrays)."""
    return f"{marker}|{batch}|{n}"


# -- Figure builders (thin wrappers — the heavy lifting is cached above) -------

def build_kde_histogram(
    values: np.ndarray,
    marker_name: str,
    threshold: float | None = None,
    transform_method: str = "Linear",
    cofactor: float = 5.0,
    n_points: int = 500,
    square_aspect: bool = False,
    logicle_T: float | None = None,
    logicle_W: float | None = None,
    batch: str = "_universal_",
) -> go.Figure:
    """Build a smooth KDE histogram with optional threshold line."""
    # Check manual cache first
    cache_key = _make_kde_key(marker_name, batch, len(values),
                              transform_method, cofactor, logicle_T, logicle_W)
    result = _cache_get(cache_key)
    if result is None:
        result = _compute_kde(values, transform_method, cofactor,
                              logicle_T, logicle_W, n_points)
        if result is not None:
            _cache_put(cache_key, result)

    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    x_grid, density = result
    fig = go.Figure()

    # KDE fill
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=density,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#4A90D9", width=2),
        fillcolor="rgba(74, 144, 217, 0.3)",
        name=marker_name,
        hovertemplate=f"{marker_name}: %{{x:.3f}}<extra></extra>",
    ))

    # Threshold line
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line=dict(color="#E74C3C", width=2, dash="dash"),
            annotation_text=f"Threshold: {threshold:.3f}",
            annotation_position="top",
        )

    transform_label = f" ({transform_method})" if transform_method != "Linear" else ""
    plot_size = 500 if square_aspect else 350
    layout_kwargs: dict[str, Any] = dict(
        title=dict(text=f"{marker_name}{transform_label}", font=dict(size=14)),
        xaxis_title=f"Intensity{transform_label}",
        yaxis_title="Density",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        height=plot_size,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if square_aspect:
        layout_kwargs["width"] = plot_size
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

    return fig


def build_bivariate_scatter(
    values_x: np.ndarray,
    values_y: np.ndarray,
    marker_x: str,
    marker_y: str,
    threshold_x: float | None = None,
    threshold_y: float | None = None,
    transform_method: str = "Linear",
    cofactor: float = 5.0,
    max_points: int = 10000,
    square_aspect: bool = False,
    logicle_T: float | None = None,
    logicle_W: float | None = None,
    batch: str = "_universal_",
) -> go.Figure:
    """Build a 2D scatter plot for bivariate gating."""
    # Check manual cache first
    cache_key = _make_scatter_key(marker_x, marker_y, batch, len(values_x),
                                  transform_method, cofactor, logicle_T, logicle_W)
    result = _cache_get(cache_key)
    if result is None:
        result = _compute_scatter_data(values_x, values_y, transform_method, cofactor,
                                       logicle_T, logicle_W, max_points)
        if result is not None:
            _cache_put(cache_key, result)

    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    x_trans, y_trans = result

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_trans,
        y=y_trans,
        mode="markers",
        marker=dict(size=3, color="rgba(74, 144, 217, 0.4)", line=dict(width=0)),
        hovertemplate=f"{marker_x}: %{{x:.3f}}<br>{marker_y}: %{{y:.3f}}<extra></extra>",
    ))

    if threshold_x is not None:
        fig.add_vline(x=threshold_x, line=dict(color="#E74C3C", width=2, dash="dash"))
    if threshold_y is not None:
        fig.add_hline(y=threshold_y, line=dict(color="#E74C3C", width=2, dash="dash"))

    transform_label = f" ({transform_method})" if transform_method != "Linear" else ""
    plot_size = 500 if square_aspect else 400
    layout_kwargs: dict[str, Any] = dict(
        title=dict(text=f"{marker_x} vs {marker_y}{transform_label}", font=dict(size=14)),
        xaxis_title=f"{marker_x}{transform_label}",
        yaxis_title=f"{marker_y}{transform_label}",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=50),
        height=plot_size,
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if square_aspect:
        layout_kwargs["width"] = plot_size
        layout_kwargs["yaxis"] = dict(scaleanchor="x", scaleratio=1)
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")

    return fig


# ==============================================================================
# Threshold Management
# ==============================================================================

def _get_csv_hash(csv_path: str) -> str:
    """Compute a hash of CSV file path + modification time for caching."""
    if not csv_path or not os.path.exists(csv_path):
        return ""
    try:
        mtime = os.path.getmtime(csv_path)
        return hashlib.md5(f"{csv_path}:{mtime}".encode()).hexdigest()[:12]
    except Exception:
        return ""


def get_cached_threshold_markers(csv_path: str) -> list[str]:
    csv_hash = _get_csv_hash(csv_path)
    if not csv_hash:
        return []
    state = _load_tab7_state()
    cache = state.get("threshold_marker_cache", {})
    entry = cache.get(csv_hash, {})
    return entry.get("markers", [])


def set_cached_threshold_markers(csv_path: str, markers: list[str]) -> None:
    csv_hash = _get_csv_hash(csv_path)
    if not csv_hash:
        return
    state = _load_tab7_state()
    cache = state.get("threshold_marker_cache", {})
    cache[csv_hash] = {"markers": markers, "path": csv_path}
    _update_tab7_state({"threshold_marker_cache": cache})


def get_cached_thresholds(csv_path: str) -> dict[str, dict[str, float]]:
    csv_hash = _get_csv_hash(csv_path)
    if not csv_hash:
        return {}
    state = _load_tab7_state()
    cache = state.get("threshold_value_cache", {})
    return cache.get(csv_hash, {})


def set_cached_thresholds(csv_path: str, thresholds: dict[str, dict[str, float]]) -> None:
    csv_hash = _get_csv_hash(csv_path)
    if not csv_hash:
        return
    state = _load_tab7_state()
    cache = state.get("threshold_value_cache", {})
    cache[csv_hash] = thresholds
    _update_tab7_state({"threshold_value_cache": cache})


def detect_batch_column(df: pd.DataFrame) -> str | None:
    candidates = ["batch_id", "Batch_ID", "BatchID", "File Name", "file_name", "FileName", "sample_id", "Sample_ID"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        col_lower = col.lower()
        if "batch" in col_lower or "file" in col_lower:
            return col
    return None


def has_corrected_markers(markers: list[str]) -> bool:
    return any(m.endswith("_corr") for m in markers)


# ==============================================================================
# CSV Import/Export
# ==============================================================================

def export_thresholds_to_csv(
    thresholds: dict[str, dict[str, float]],
    output_path: str,
    use_batch: bool = True,
) -> bool:
    """Export threshold values to CSV compatible with SCIMAP/Resolve tabs."""
    rows = []
    for marker, batch_thresholds in thresholds.items():
        if use_batch:
            # Per-batch export: one row per marker × batch
            for batch_id, value in batch_thresholds.items():
                if batch_id != "_universal_":
                    rows.append({"marker": marker, "threshold": value, "batch_id": batch_id})
        else:
            # Universal export: ONE row per marker with the best threshold.
            # Prefer the _universal_ key; otherwise take max non-zero value.
            universal_val = batch_thresholds.get("_universal_")
            if universal_val is not None and universal_val != 0.0:
                best = universal_val
            else:
                nonzero = [v for v in batch_thresholds.values() if v != 0.0]
                best = max(nonzero) if nonzero else 0.0
            rows.append({"marker": marker, "threshold": best})

    if not rows:
        return False

    df = pd.DataFrame(rows)
    if use_batch and "batch_id" in df.columns:
        df = df[["marker", "threshold", "batch_id"]]
    else:
        df = df[["marker", "threshold"]]

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Failed to export thresholds: {e}")
        return False


def import_thresholds_from_csv(
    csv_path: str,
) -> tuple[dict[str, dict[str, float]], bool]:
    """Import threshold values from CSV file."""
    if not os.path.exists(csv_path):
        return {}, False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read threshold CSV: {e}")
        return {}, False

    df.columns = [c.lower().strip() for c in df.columns]

    if "marker" not in df.columns or "threshold" not in df.columns:
        logger.error("Threshold CSV must have 'marker' and 'threshold' columns")
        return {}, False

    has_batch = "batch_id" in df.columns

    thresholds: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        marker = str(row["marker"]).strip()
        try:
            value = float(row["threshold"])
        except (ValueError, TypeError):
            continue

        if has_batch:
            batch_id = str(row["batch_id"]).strip()
        else:
            batch_id = "_universal_"

        if marker not in thresholds:
            thresholds[marker] = {}
        thresholds[marker][batch_id] = value

    return thresholds, has_batch


# ==============================================================================
# Timing helper
# ==============================================================================

def _tlog(msg: str, t0: float | None = None) -> float:
    """Print a timestamped log line to the console and return current time."""
    now = time.time()
    elapsed = f" [{(now - t0)*1000:.0f} ms]" if t0 is not None else ""
    print(f"[THRESH {time.strftime('%H:%M:%S')}]{elapsed} {msg}", flush=True)
    return now


# ==============================================================================
# Main Render Function
# ==============================================================================

def render_threshold_tab(project_root: Path, global_input_path: str | None) -> None:
    """Render the Threshold gating subtab."""
    _t0 = _tlog("render_threshold_tab START")

    st.subheader("Threshold Gating")
    st.caption("Interactive flow cytometry-style gating for marker thresholding. "
               "Set thresholds visually and export for use in SCIMAP / Resolve tabs.")

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------
    if not _HAS_PLOTLY_EVENTS:
        st.error(
            "Missing dependency: streamlit-plotly-events. "
            "Install with: pip install streamlit-plotly-events"
        )
        return

    if not global_input_path or not os.path.exists(global_input_path):
        st.warning("Please select a unified CSV file in the Global Input Configuration above.")
        return

    # ------------------------------------------------------------------
    # Session state initialisation
    # ------------------------------------------------------------------
    _defaults: dict[str, Any] = {
        "thresh_selected_markers": [],
        "thresh_data": None,
        "thresh_values": {},
        "thresh_active_marker": None,
        "thresh_active_batch": None,
        "thresh_loaded_path": None,
        "thresh_click_version": 0,
    }
    for key, default in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ==================================================================
    # 1. Data Source
    # ==================================================================
    st.markdown("#### 1. Data Source")
    st.caption(f"**Using:** {global_input_path}")

    col_sample, col_load = st.columns([4, 1])
    with col_sample:
        subsample_cap = st.number_input(
            "Subsampling Cap",
            min_value=10000,
            max_value=500000,
            value=150000,
            step=10000,
            key="thresh_subsample_cap",
            help="Maximum number of cells to load for visualization.",
        )

    needs_load = (
        st.session_state.thresh_data is None
        or st.session_state.thresh_loaded_path != global_input_path
    )

    with col_load:
        # Match the vertical alignment to the number_input (label + input)
        st.markdown(
            "<div style='height:28px'></div>",
            unsafe_allow_html=True,
        )
        load_btn = st.button(
            "Load Data" if needs_load else "Reload Data",
            key="thresh_load_btn",
            type="primary" if needs_load else "secondary",
            width="stretch",
        )

    # Auto-load when path changes
    if needs_load and st.session_state.thresh_loaded_path != global_input_path:
        load_btn = True

    if load_btn:
        with st.spinner("Loading data..."):
            try:
                csv_path = global_input_path
                _t1 = _tlog("Load: reading header")

                df_header = pd.read_csv(csv_path, nrows=0)
                available_markers = get_marker_list(df_header)
                _t1 = _tlog(f"Load: header done, {len(available_markers)} markers", _t1)

                cached_markers = get_cached_threshold_markers(csv_path)
                if cached_markers:
                    cached_markers = [m for m in cached_markers if m in available_markers]
                _t1 = _tlog(f"Load: cached markers = {len(cached_markers) if cached_markers else 0}", _t1)

                total_rows = sum(1 for _ in open(csv_path)) - 1
                _t1 = _tlog(f"Load: counted {total_rows} rows", _t1)
                if total_rows > subsample_cap:
                    skip_rows = sorted(
                        np.random.choice(
                            range(1, total_rows + 1),
                            total_rows - int(subsample_cap),
                            replace=False,
                        )
                    )
                    df = pd.read_csv(csv_path, skiprows=skip_rows)
                else:
                    df = pd.read_csv(csv_path)

                _t1 = _tlog(f"Load: CSV read done, {len(df)} rows", _t1)
                st.session_state.thresh_data = df
                st.session_state.thresh_available_markers = available_markers
                st.session_state.thresh_loaded_path = csv_path
                _clear_cache()  # invalidate all compute caches for new data

                cached_thresholds = get_cached_thresholds(csv_path)
                if cached_thresholds:
                    st.session_state.thresh_values = cached_thresholds
                _t1 = _tlog("Load: cache restored", _t1)

                if cached_markers:
                    st.session_state.thresh_selected_markers = cached_markers

                st.success(f"Loaded {len(df):,} cells with {len(available_markers)} markers.")
                _tlog("Load: COMPLETE", _t1)
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                return

    # Guard: nothing to show until data is loaded
    if st.session_state.thresh_data is None:
        return

    # ==================================================================
    # 2. Marker Selection
    # ==================================================================
    st.divider()
    st.markdown("#### 2. Marker Selection")

    available_markers: list[str] = st.session_state.get("thresh_available_markers", [])

    shared_marker_panel, marker_workspace_note = get_marker_workspace_panel(
        available_markers,
        global_input_path,
    )
    _, selected, marker_workspace_note = render_method_marker_selection(
        key_prefix="thresh",
        method_name="Threshold Gating",
        full_input_path=global_input_path,
        available_columns=available_markers,
        shared_panel=shared_marker_panel,
        marker_workspace_note=marker_workspace_note,
        fallback_markers=[],
        saved_markers=st.session_state.get("thresh_selected_markers", []),
    )
    st.session_state.thresh_selected_markers = list(selected)
    if global_input_path:
        # Keep the small threshold cache compatible with older projects; the
        # authoritative per-method settings are also saved in project JSON by
        # render_method_marker_selection().
        set_cached_threshold_markers(global_input_path, list(selected))

    # Use the confirmed marker selection (from session state)
    selected = st.session_state.thresh_selected_markers
    selected = [m for m in selected if m in available_markers]

    if has_corrected_markers(selected):
        st.info(
            "Corrected markers detected (_corr suffix). "
            "Consider using universal thresholds (disable batch mode)."
        )


    st.caption(f"**Selected:** {len(selected)} markers")

    if not selected:
        st.info("Select at least one marker to proceed.")
        return

    # ------------------------------------------------------------------
    # Wrap Sections 3-6 in st.fragment for PARTIAL reruns.
    # When the user changes Transform, Viz mode, Batch selector,
    # Cofactor, etc., ONLY this fragment re-executes.
    # The full 8-tab app.py rerun is skipped entirely.
    # ------------------------------------------------------------------
    @st.fragment
    def _interactive_section():
        _ft0 = _tlog("fragment: _interactive_section START")
        # Read shared state (safe for both full and partial reruns)
        selected = st.session_state.thresh_selected_markers
        df = st.session_state.thresh_data
        
        if selected:
            valid_mask = ~df[selected].apply(pd.to_numeric, errors='coerce').isna().any(axis=1)
            df = df[valid_mask].copy()

        # ==================================================================
        # 3. Settings (Batch + Transformation)
        # ==================================================================
        st.divider()
        st.markdown("#### 3. Settings")

        detected_batch_col = detect_batch_column(df)

        use_batch = st.checkbox(
            "Enable Batch Mode",
            value=detected_batch_col is not None,
            key="thresh_use_batch",
            help="Enable per-batch thresholding. Disable for universal thresholds.",
        )

        col_batch, col_transform = st.columns(2)

        with col_batch:
            if use_batch:
                batch_options = [c for c in df.columns if c not in selected]
                batch_col_idx = 0
                if detected_batch_col and detected_batch_col in batch_options:
                    batch_col_idx = batch_options.index(detected_batch_col)

                batch_col = st.selectbox(
                    "Batch Column",
                    options=batch_options,
                    index=batch_col_idx,
                    key="thresh_batch_col",
                )
                batch_values = sorted(df[batch_col].dropna().unique().astype(str).tolist())
            else:
                batch_col = None
                batch_values = ["_universal_"]

        with col_transform:
            transform = st.selectbox(
                "Transformation",
                options=DataTransform.get_names(),
                index=0,
                key="thresh_transform",
                help=DataTransform.get_description(
                    st.session_state.get("thresh_transform", "Linear")
                ),
            )

            cofactor = 5.0
            logicle_T: float | None = None
            logicle_W: float | None = None

            if DataTransform.needs_cofactor(transform):
                # Manual Arcsinh cofactor
                cofactor = st.number_input(
                    "Cofactor",
                    min_value=0.01,
                    max_value=10000.0,
                    value=5.0,
                    step=0.5,
                    key="thresh_cofactor",
                    help=(
                        "Divide data by this value before arcsinh.  "
                        "CyTOF ≈ 5, CODEX fluorescence ≈ 100-500, normalized ≈ 0.2-1."
                    ),
                )
            elif DataTransform.has_auto_cofactor(transform):
                # Auto-estimate cofactor from data — CACHED
                first_marker = selected[0] if selected else None
                if first_marker and first_marker in df.columns:
                    pk = _make_param_key("arcsinh_cofactor", first_marker, len(df))
                    cofactor = _cache_get(pk)
                    if cofactor is None:
                        cofactor = estimate_arcsinh_cofactor(df[first_marker].values)
                        _cache_put(pk, cofactor)
                else:
                    cofactor = 5.0
                st.caption(f"Auto-estimated cofactor: **{cofactor:.4f}**")
            elif DataTransform.needs_logicle(transform):
                # Logicle T / W parameters — CACHED estimation
                first_marker = selected[0] if selected else None
                if first_marker and first_marker in df.columns:
                    pk = _make_param_key("logicle_params", first_marker, len(df))
                    cached_lp = _cache_get(pk)
                    if cached_lp is not None:
                        auto_T, auto_W = cached_lp
                    else:
                        auto_T, auto_W = estimate_logicle_params(df[first_marker].values)
                        _cache_put(pk, (auto_T, auto_W))
                else:
                    auto_T, auto_W = 262144.0, 0.5
                tc1, tc2 = st.columns(2)
                with tc1:
                    logicle_T = st.number_input(
                        "T (scale top)",
                        min_value=1.0,
                        max_value=1e8,
                        value=float(auto_T),
                        step=1000.0,
                        format="%.1f",
                        key="thresh_logicle_T",
                        help="Maximum data value. Auto-estimated from your data.",
                    )
                with tc2:
                    logicle_W = st.number_input(
                        "W (linearisation)",
                        min_value=0.05,
                        max_value=3.0,
                        value=float(auto_W),
                        step=0.05,
                        format="%.2f",
                        key="thresh_logicle_W",
                        help=(
                            "Width of the linear region near zero.  "
                            "Larger → wider linear zone → better neg/pos spread."
                        ),
                    )

        # ==================================================================
        # 4. Threshold Table
        # ==================================================================
        st.divider()
        st.markdown("#### 4. Threshold Values")

        thresh_values: dict[str, dict[str, float]] = st.session_state.thresh_values
        for marker in selected:
            if marker not in thresh_values:
                thresh_values[marker] = {}
            for batch in batch_values:
                if batch not in thresh_values[marker]:
                    thresh_values[marker][batch] = 0.0

        _is_transformed = transform != "Linear"

        if use_batch and len(batch_values) > 1:
            # -- Matrix view (rows = markers, cols = batches) --
            st.caption(
                "Edit the Linear column to set thresholds. "
                + ("The Transformed column is shown for reference only." if _is_transformed else "")
            )

            table_data = []
            for marker in selected:
                row: dict[str, Any] = {"Marker": marker}
                for batch in batch_values:
                    lin_val = round(thresh_values.get(marker, {}).get(batch, 0.0), 4)
                    row[batch] = lin_val
                table_data.append(row)

            table_df = pd.DataFrame(table_data)
            col_config: dict[str, Any] = {
                "Marker": st.column_config.TextColumn("Marker", disabled=True),
            }
            edited_df = st.data_editor(
                table_df,
                key="thresh_table_editor",
                width="stretch",
                hide_index=True,
                column_config=col_config,
            )

            # Write edits back
            for _, row in edited_df.iterrows():
                marker = row["Marker"]
                for batch in batch_values:
                    if batch in row:
                        try:
                            thresh_values[marker][batch] = float(row[batch])
                        except (ValueError, TypeError):
                            pass

        else:
            # -- Simple list view (universal thresholds) --
            if _is_transformed:
                st.caption(
                    f"Linear = original intensity (used by SCIMAP / Resolve). "
                    f"Transformed = {transform} scale (shown on plots)."
                )
            table_data = []
            for marker in selected:
                lin_val = thresh_values.get(marker, {}).get("_universal_", 0.0)
                row_dict: dict[str, Any] = {
                    "Marker": marker,
                    "Linear": round(lin_val, 4),
                }
                if _is_transformed:
                    row_dict[f"Transformed ({transform})"] = round(
                        float(DataTransform.transform(np.array([lin_val]), transform, cofactor,
                              logicle_T=logicle_T, logicle_W=logicle_W)[0]), 4
                    )
                table_data.append(row_dict)

            table_df = pd.DataFrame(table_data)

            col_config_simple: dict[str, Any] = {
                "Marker": st.column_config.TextColumn("Marker", disabled=True),
                "Linear": st.column_config.NumberColumn("Linear", format="%.4f"),
            }
            if _is_transformed:
                col_config_simple[f"Transformed ({transform})"] = st.column_config.NumberColumn(
                    f"Transformed ({transform})", format="%.4f", disabled=True,
                )

            edited_df = st.data_editor(
                table_df,
                key="thresh_table_editor_simple",
                width="stretch",
                hide_index=True,
                column_config=col_config_simple,
            )

            for _, row in edited_df.iterrows():
                marker = row["Marker"]
                try:
                    thresh_values[marker]["_universal_"] = float(row["Linear"])
                except (ValueError, TypeError):
                    pass

        st.session_state.thresh_values = thresh_values
        _tlog("Section 4 (table) done", _t0)

        # NOTE: persistent cache is only written on explicit actions (click, export, import)
        # to avoid slow disk I/O on every widget interaction.

        # ==================================================================
        # 5. Interactive Gating
        # ==================================================================
        st.divider()
        st.markdown("#### 5. Interactive Gating")

        # -- Controls row --
        ctrl_cols = st.columns([1.2, 2.4, 0.4])

        with ctrl_cols[0]:
            viz_mode = st.radio(
                "Visualization",
                options=["Histogram", "Scatter"],
                key="thresh_viz_mode",
                horizontal=True,
            )

        with ctrl_cols[1]:
            # Always initialise scatter variables so they exist in both branches
            marker_x = selected[0] if selected else None
            marker_y = selected[1] if len(selected) > 1 else None

            if viz_mode == "Histogram":
                active_marker = st.selectbox(
                    "Marker",
                    options=selected,
                    key="thresh_hist_marker",
                )
                st.session_state.thresh_active_marker = active_marker
            else:
                mc = st.columns(2)
                with mc[0]:
                    marker_x = st.selectbox(
                        "X-axis Marker", options=selected, key="thresh_scatter_x"
                    )
                with mc[1]:
                    y_options = [m for m in selected if m != marker_x]
                    marker_y = st.selectbox(
                        "Y-axis Marker",
                        options=y_options if y_options else selected,
                        key="thresh_scatter_y",
                    )

        with ctrl_cols[2]:
            square_aspect = st.checkbox(
                "Square",
                value=False,
                key="thresh_square_aspect",
                help="Force equal width and height on the plot.",
            )

        # Batch selector (only when batch mode active)
        if use_batch and len(batch_values) > 1:
            active_batch = st.selectbox(
                "Batch for Visualization",
                options=batch_values,
                key="thresh_viz_batch",
            )
        else:
            active_batch = "_universal_"

        st.session_state.thresh_active_batch = active_batch

        # Subset data
        if use_batch and batch_col:
            mask = df[batch_col].astype(str) == active_batch
            viz_df = df[mask]
        else:
            viz_df = df

        # -- Click version keeps the plotly_events key unique after each click,
        #    which forces Streamlit to remount the component with the updated
        #    figure (including the new threshold line) instead of showing a
        #    stale cached version.
        click_ver = st.session_state.get("thresh_click_version", 0)

        # -- Render plot --
        _t_viz = _tlog("Section 5: building visualization")
        if viz_mode == "Histogram":
            marker = st.session_state.thresh_active_marker
            if marker and marker in viz_df.columns:
                values = viz_df[marker].values
                _tlog(f"Histogram: extracting {len(values)} values for {marker}", _t_viz)
                current_thresh = thresh_values.get(marker, {}).get(active_batch)

                if current_thresh and current_thresh != 0.0:
                    display_thresh = float(
                        DataTransform.transform(np.array([current_thresh]), transform, cofactor,
                                               logicle_T=logicle_T, logicle_W=logicle_W)[0]
                    )
                else:
                    display_thresh = None

                _t_kde = _tlog("Histogram: building KDE figure")
                fig = build_kde_histogram(
                    values,
                    marker,
                    threshold=display_thresh,
                    transform_method=transform,
                    cofactor=cofactor,
                    square_aspect=square_aspect,
                    logicle_T=logicle_T,
                    logicle_W=logicle_W,
                    batch=active_batch,
                )
                _tlog("Histogram: KDE figure built", _t_kde)

                st.caption("Click on the histogram to set the threshold.")
                _t_pe = _tlog("Histogram: rendering plotly_events")
                clicked = plotly_events(
                    fig,
                    click_event=True,
                    key=f"thresh_hist_click_{marker}_{active_batch}_v{click_ver}",
                )
                _tlog("Histogram: plotly_events rendered", _t_pe)

                if clicked and len(clicked) > 0:
                    click_x = clicked[0].get("x")
                    if click_x is not None:
                        linear_thresh = DataTransform.inverse_transform(
                            float(click_x), transform, cofactor,
                            logicle_T=logicle_T, logicle_W=logicle_W
                        )
                        thresh_values[marker][active_batch] = linear_thresh
                        st.session_state.thresh_values = thresh_values
                        # Persist immediately
                        if st.session_state.thresh_loaded_path:
                            set_cached_thresholds(
                                st.session_state.thresh_loaded_path, thresh_values
                            )
                        # Bump version so the next render gets a fresh plotly_events key
                        st.session_state.thresh_click_version = click_ver + 1
                        st.rerun(scope="fragment")

        else:  # Scatter
            _tlog(f"Scatter mode: marker_x={marker_x!r}, marker_y={marker_y!r}")
            _tlog(f"Scatter mode: marker_x in columns={marker_x in viz_df.columns if marker_x else False}, marker_y in columns={marker_y in viz_df.columns if marker_y else False}")
            if marker_x and marker_y and marker_x in viz_df.columns and marker_y in viz_df.columns:
                values_x = viz_df[marker_x].values
                values_y = viz_df[marker_y].values

                thresh_x = thresh_values.get(marker_x, {}).get(active_batch)
                thresh_y = thresh_values.get(marker_y, {}).get(active_batch)

                display_thresh_x = (
                    float(DataTransform.transform(np.array([thresh_x]), transform, cofactor,
                          logicle_T=logicle_T, logicle_W=logicle_W)[0])
                    if thresh_x and thresh_x != 0.0
                    else None
                )
                display_thresh_y = (
                    float(DataTransform.transform(np.array([thresh_y]), transform, cofactor,
                          logicle_T=logicle_T, logicle_W=logicle_W)[0])
                    if thresh_y and thresh_y != 0.0
                    else None
                )

                _t_scatter = _tlog("Scatter: building figure")
                fig = build_bivariate_scatter(
                    values_x,
                    values_y,
                    marker_x,
                    marker_y,
                    threshold_x=display_thresh_x,
                    threshold_y=display_thresh_y,
                    transform_method=transform,
                    cofactor=cofactor,
                    square_aspect=square_aspect,
                    logicle_T=logicle_T,
                    logicle_W=logicle_W,
                    batch=active_batch,
                )
                _tlog("Scatter: figure built", _t_scatter)

                st.caption("Click on the plot to set thresholds for both markers.")
                _t_spe = _tlog("Scatter: rendering plotly_events")
                clicked = plotly_events(
                    fig,
                    click_event=True,
                    key=f"thresh_scatter_click_{marker_x}_{marker_y}_{active_batch}_v{click_ver}",
                )
                _tlog("Scatter: plotly_events rendered", _t_spe)

                if clicked and len(clicked) > 0:
                    click_x = clicked[0].get("x")
                    click_y = clicked[0].get("y")
                    if click_x is not None:
                        linear_thresh_x = DataTransform.inverse_transform(
                            float(click_x), transform, cofactor,
                            logicle_T=logicle_T, logicle_W=logicle_W
                        )
                        thresh_values[marker_x][active_batch] = linear_thresh_x
                    if click_y is not None:
                        linear_thresh_y = DataTransform.inverse_transform(
                            float(click_y), transform, cofactor,
                            logicle_T=logicle_T, logicle_W=logicle_W
                        )
                        thresh_values[marker_y][active_batch] = linear_thresh_y
                    st.session_state.thresh_values = thresh_values
                    if st.session_state.thresh_loaded_path:
                        set_cached_thresholds(
                            st.session_state.thresh_loaded_path, thresh_values
                        )
                    st.session_state.thresh_click_version = click_ver + 1
                    st.rerun(scope="fragment")
            else:
                _tlog(f"Scatter SKIPPED — condition failed. marker_x={marker_x!r}, marker_y={marker_y!r}")
                if marker_x and marker_y:
                    _tlog(f"  viz_df columns sample: {list(viz_df.columns[:10])}")
                st.warning("Cannot render scatter: select two valid markers that exist in the loaded data.")

        # ==================================================================
        # 6. Export / Import
        # ==================================================================
        st.divider()
        st.markdown("#### 6. Export / Import")

        exp_col, imp_col = st.columns(2)

        with exp_col:
            default_export = str(
                project_root / "outputs" / "phenotyping" / "thresholds_gating.csv"
            )
            export_path = st.text_input(
                "Export Path",
                value=default_export,
                key="thresh_export_path",
            )
            if st.button("Export to CSV", key="thresh_export_btn", width="stretch"):
                if export_thresholds_to_csv(thresh_values, export_path, use_batch=use_batch):
                    st.success(f"Exported to: {export_path}")
                else:
                    st.error("Export failed. Check path and permissions.")

        with imp_col:
            import_file = st.file_uploader(
                "Import CSV",
                type=["csv"],
                key="thresh_import_file",
            )
            if import_file is not None:
                temp_dir = project_root / "outputs" / "phenotyping"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = str(temp_dir / "_temp_import.csv")
                with open(temp_path, "wb") as f:
                    f.write(import_file.getvalue())

                imported, has_batch_import = import_thresholds_from_csv(temp_path)
                if imported:
                    for marker, batches in imported.items():
                        if marker not in thresh_values:
                            thresh_values[marker] = {}
                        thresh_values[marker].update(batches)
                    st.session_state.thresh_values = thresh_values
                    if st.session_state.thresh_loaded_path:
                        set_cached_thresholds(
                            st.session_state.thresh_loaded_path, thresh_values
                        )
                    st.success(f"Imported thresholds for {len(imported)} markers.")
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                else:
                    st.error("Import failed. CSV must contain 'marker' and 'threshold' columns.")

        if has_corrected_markers(selected):
            st.warning(
                "Corrected markers detected. When importing thresholds into SCIMAP or Resolve, "
                "ensure the target markers are the corrected versions (with _corr suffix)."
            )


        _tlog("fragment: _interactive_section END", _ft0)

    _interactive_section()

    _tlog("render_threshold_tab END", _t0)
