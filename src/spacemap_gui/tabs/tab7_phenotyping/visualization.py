"""
Visualization Subtab
Publication-ready visualizations for phenotyping comparisons (Step8).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from spacemap_gui.ui_utils import clamp_numeric_widget_state
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except Exception:
    linkage = None
    leaves_list = None

try:
    from spacemap_gui.cell_visualizer import CellVisualizer
except ImportError:
    CellVisualizer = None

from .common import get_named_marker_panels, logger, _extract_markers_from_notebook
from .cell_visualization import render_cell_visualization
from spacemap_gui import terminal

NOTEBOOK_NAME = "Step8_Comparison.ipynb"
_VIZ_SHARED_DF_KEY = "viz_shared_df"
_VIZ_SHARED_PATH_KEY = "viz_shared_df_path"
_VIZ_SHARED_MTIME_KEY = "viz_shared_df_mtime"


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _log_console(message: str, level: str = "info") -> None:
    """One timestamped terminal line for this tab (see spacemap_gui.terminal)."""
    terminal.say('tab7', message, step='Visualization', level=level)


def _log_terminal_event(message: str) -> None:
    _log_console(message)


def _get_csv_columns_fast(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _get_loaded_visualization_df(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    loaded_path = st.session_state.get(_VIZ_SHARED_PATH_KEY)
    loaded_mtime = st.session_state.get(_VIZ_SHARED_MTIME_KEY)
    df = st.session_state.get(_VIZ_SHARED_DF_KEY)
    if loaded_path != path or not isinstance(df, pd.DataFrame):
        return None
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return None
    if loaded_mtime != mtime:
        return None
    return df


def _set_loaded_visualization_df(path: str, df: pd.DataFrame) -> None:
    st.session_state[_VIZ_SHARED_DF_KEY] = df
    st.session_state[_VIZ_SHARED_PATH_KEY] = path
    st.session_state[_VIZ_SHARED_MTIME_KEY] = os.path.getmtime(path)


def _clear_loaded_visualization_df() -> None:
    st.session_state.pop(_VIZ_SHARED_DF_KEY, None)
    st.session_state.pop(_VIZ_SHARED_PATH_KEY, None)
    st.session_state.pop(_VIZ_SHARED_MTIME_KEY, None)


def _extract_list_from_notebook(var_names: list[str]) -> list[str]:
    """Extract a list literal from the Step8 notebook.

    Parameters
    ----------
    var_names : list[str]
        Variable names to search for.

    Returns
    -------
    list[str]
        Parsed list or empty list.
    """
    try:
        notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / NOTEBOOK_NAME
        if not notebook_path.exists():
            return []
        with open(notebook_path, "r") as f:
            nb = json.load(f)

        def _extract_list(var_name: str, cell_source: str) -> list[str]:
            m = re.search(rf"{re.escape(var_name)}\s*=\s*\[(.*?)\]", cell_source, flags=re.S)
            if not m:
                return []
            inside = m.group(1)
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", inside)
            out: list[str] = []
            seen: set[str] = set()
            for t in tokens:
                t = str(t).strip()
                if not t or t in seen:
                    continue
                out.append(t)
                seen.add(t)
            return out

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source:
                continue
            for var in var_names:
                if var in source:
                    values = _extract_list(var, source)
                    if values:
                        return values
        return []
    except Exception:
        return []


def _extract_scalar_from_notebook(var_name: str) -> str | None:
    try:
        notebook_path = Path(__file__).resolve().parents[3] / "notebooks" / NOTEBOOK_NAME
        if not notebook_path.exists():
            return None
        with open(notebook_path, "r") as f:
            nb = json.load(f)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source or var_name not in source:
                continue
            m = re.search(rf"{re.escape(var_name)}\s*=\s*([^\n#]+)", source)
            if m:
                return m.group(1).strip().strip("'\"")
        return None
    except Exception:
        return None


def _extract_list_defaults_from_notebook(var_name: str) -> list[str]:
    values = _extract_list_from_notebook([var_name])
    return values or []


@st.cache_data(show_spinner=False)
def _cached_read_columns(
    path: str, mtime: float, columns: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Load a CSV file with Streamlit caching.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    mtime : float
        File modification time for cache invalidation.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    _ = mtime
    # Reads through the shared loader so the Parquet fast copy is used when one
    # exists, and so naming columns actually skips the rest on disk.
    from spacemap import dataio

    return dataio.load(path, columns=list(columns) if columns else None)


def _get_unified_df(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a cached unified dataframe for the given path.

    Parameters
    ----------
    path : str
        Path to the unified CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Unified CSV file not found.")
    mtime = os.path.getmtime(path)
    if columns:
        return _cached_read_columns(path, mtime, tuple(columns))

    # A full-table read skips Streamlit's cache on purpose: it pickles what it
    # stores and unpickles on every hit, which costs twenty to thirty seconds
    # on a frame this size. The caller keeps it in session_state instead.
    from spacemap import dataio

    return dataio.load(path)


def _get_cached_unified_df(path: str) -> pd.DataFrame:
    """Return unified dataframe from session cache when possible."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError("Unified CSV file not found.")

    loaded_df = _get_loaded_visualization_df(path)
    if isinstance(loaded_df, pd.DataFrame):
        return loaded_df

    mtime = os.path.getmtime(path)
    cache_key = f"viz_unified::{path}::{mtime}"
    if st.session_state.get("viz_unified_cache_key") == cache_key:
        cached_df = st.session_state.get("viz_unified_cache_df")
        if isinstance(cached_df, pd.DataFrame):
            _set_loaded_visualization_df(path, cached_df)
            return cached_df

    _log_terminal_event(f"Loading unified CSV into memory: {os.path.basename(path)}")
    t0 = time.perf_counter()
    df = _get_unified_df(path)
    dt = time.perf_counter() - t0
    _log_terminal_event(f"Loaded {len(df):,} rows, {len(df.columns)} columns in {dt:.2f}s")

    st.session_state["viz_unified_cache_key"] = cache_key
    st.session_state["viz_unified_cache_df"] = df
    _set_loaded_visualization_df(path, df)
    return df


def _cluster_order(values: pd.DataFrame, axis: int) -> list[str]:
    if linkage is None or leaves_list is None:
        return list(values.index if axis == 0 else values.columns)
    data = values.values if axis == 0 else values.values.T
    if data.shape[0] <= 1:
        return list(values.index if axis == 0 else values.columns)
    try:
        link = linkage(data, method="average", metric="euclidean")
        order = leaves_list(link)
        labels = list(values.index if axis == 0 else values.columns)
        return [labels[i] for i in order]
    except Exception:
        return list(values.index if axis == 0 else values.columns)


def _apply_custom_order(items: list[str], custom_list: list[str]) -> list[str]:
    ordered = [i for i in custom_list if i in items]
    remainder = [i for i in items if i not in ordered]
    return ordered + remainder


def _build_palette(colors: list[str]) -> list[str]:
    return colors if colors else sns.color_palette("tab10").as_hex()


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "heatmap"


def _strip_suffixes(value: str, suffixes: list[str]) -> str:
    if not suffixes:
        return value
    cleaned = str(value)
    for suffix in suffixes:
        if suffix and cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned


def _get_filterable_columns(columns: list[str], excluded: list[str] | None = None) -> list[str]:
    excluded_set = {str(col) for col in (excluded or [])}
    blocked_prefixes = ("umap_", "leiden_")
    blocked_suffixes = ("_norm",)

    filterable: list[str] = []
    for column in columns:
        column_str = str(column)
        column_lower = column_str.lower()
        if column_str in excluded_set:
            continue
        if column_lower.startswith(blocked_prefixes):
            continue
        if column_lower.endswith(blocked_suffixes):
            continue
        filterable.append(column_str)
    return filterable


def _get_low_cardinality_filter_columns(
    df: pd.DataFrame,
    columns: list[str],
    excluded: list[str] | None = None,
    max_unique: int = 100,
) -> list[str]:
    candidate_columns = _get_filterable_columns(columns, excluded=excluded)
    low_cardinality: list[str] = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        try:
            unique_count = int(non_null.nunique(dropna=True))
        except Exception:
            continue
        if unique_count == 0 or unique_count > max_unique:
            continue
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series):
            low_cardinality.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series) and unique_count <= min(20, max_unique):
            low_cardinality.append(column)
    return low_cardinality


def _get_filter_value_options(df: pd.DataFrame, column: str, limit: int = 500) -> list[str]:
    if column not in df.columns:
        return []
    try:
        values = (
            df[column]
            .dropna()
            .astype(str)
            .map(str.strip)
        )
        values = values[values != ""]
        unique_values = sorted(values.unique().tolist())
        return unique_values[:limit]
    except Exception:
        return []


def _read_heatmap_filter_specs(n_filters: int) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    for idx in range(n_filters):
        column = str(st.session_state.get(f"viz_heatmap_filter_col_{idx}", "") or "").strip()
        selected_values = st.session_state.get(f"viz_heatmap_filter_vals_{idx}", [])
        if not column or not isinstance(selected_values, list) or not selected_values:
            continue
        cleaned_values = [str(value) for value in selected_values if str(value).strip()]
        if cleaned_values:
            specs.append((column, cleaned_values))
    return specs


def _apply_value_filters(df: pd.DataFrame, filter_specs: list[tuple[str, list[str]]]) -> tuple[pd.DataFrame, list[str]]:
    filtered_df = df.copy()
    summary_lines: list[str] = []
    for column, selected_values in filter_specs:
        if column not in filtered_df.columns:
            summary_lines.append(f"Skipped '{column}' (column not found)")
            continue
        before_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected_values)].copy()
        after_count = len(filtered_df)
        preview = ", ".join(selected_values[:5])
        if len(selected_values) > 5:
            preview = f"{preview}, ..."
        summary_lines.append(f"{column} in [{preview}] ({before_count:,} -> {after_count:,})")
    return filtered_df, summary_lines


def _prepare_filtered_df(df: pd.DataFrame, gt_col: str) -> pd.DataFrame:
    if gt_col not in df.columns:
        raise ValueError(f"Ground truth column '{gt_col}' not found.")
    work = df.copy()
    work = work[~work[gt_col].isin(["unclassified"])].copy()
    work = work.dropna(subset=[gt_col])
    return work


def _compute_method_metrics(df: pd.DataFrame, gt_col: str, methods: list[str]) -> list[dict[str, Any]]:
    results = []
    available_methods = [m for m in methods if m in df.columns]
    for method in available_methods:
        method_df = df[df[method].notna() & (df[method] != "unknown") & df[gt_col].notna()].copy()
        if method_df.empty:
            continue
        y_true = method_df[gt_col]
        y_pred = method_df[method]
        labels = sorted(list(set(y_true) | set(y_pred)))
        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        results.append(
            {
                "method": method,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_weighted": precision_w,
                "recall_weighted": recall_w,
                "f1_weighted": f1_w,
                "cohen_kappa": cohen_kappa_score(y_true, y_pred),
                "matthews_corr": matthews_corrcoef(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
                "labels": labels,
                "per_class_precision": precision_per,
                "per_class_recall": recall_per,
                "per_class_f1": f1_per,
                "per_class_support": support_per,
                "n_cells": len(method_df),
                "coverage": (len(method_df) / len(df)) * 100,
            }
        )
    return results


def _collect_filtered_cell_types(results: list[dict[str, Any]], exclude_list: list[str]) -> list[str]:
    """Collect unique cell types across results, excluding unwanted phenotypes.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    exclude_list : list[str]
        Phenotypes to exclude.

    Returns
    -------
    list[str]
        Filtered, sorted cell types.
    """
    cell_types = {label for r in results for label in r.get("labels", [])}
    return sorted([ct for ct in cell_types if ct not in exclude_list])


def _build_per_class_matrix(
    results: list[dict[str, Any]],
    metric_key: str,
    cell_types: list[str],
) -> pd.DataFrame:
    """Build per-class metric matrix for a given metric.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    metric_key : str
        Key in results containing per-class metric array.
    cell_types : list[str]
        Ordered list of cell types.

    Returns
    -------
    pd.DataFrame
        Metric matrix with cell types as rows and methods as columns.
    """
    methods = [r["method"] for r in results]
    matrix = pd.DataFrame(index=cell_types, columns=methods, dtype=float)
    for result in results:
        method = result["method"]
        labels = result.get("labels", [])
        values = result.get(metric_key, [])
        for i, label in enumerate(labels):
            if label in cell_types and i < len(values):
                matrix.loc[label, method] = values[i]
    return matrix


def _plot_comprehensive_performance(
    results: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    """Create a comprehensive performance comparison figure.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    output_dir : Path
        Output directory for the figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple[Figure, Path]
        Figure and saved figure path.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 14))
    fig.suptitle("Comprehensive Phenotyping Validation", fontsize=16, fontweight="bold")

    methods = [r["method"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    bars = ax1.bar(methods, accuracies, color="skyblue", alpha=0.8, edgecolor="black")
    ax1.set_title("Overall Accuracy Comparison", fontweight="bold")
    ax1.set_ylabel("Accuracy", fontweight="bold")
    ax1.set_ylim(0, 1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}", ha="center", va="bottom")

    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cohen_kappa"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa"]
    x = np.arange(len(methods))
    width = 0.18
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [r[metric] for r in results]
        ax2.bar(x + i * width - 2 * width, values, width, label=name, alpha=0.85)
    ax2.set_title("Multiple Metrics Comparison", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_xlabel("Methods", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10, edgecolor="black")

    n_cells = [r["n_cells"] for r in results]
    bars = ax3.bar(methods, n_cells, color="orange", alpha=0.8, edgecolor="black")
    ax3.set_title("Number of Cells Analyzed", fontweight="bold")
    ax3.set_ylabel("Number of Cells", fontweight="bold")
    ax3.set_xlabel("Methods", fontweight="bold")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01 * max(n_cells), f"{int(height):,}", ha="center", va="bottom")

    ax4.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "comprehensive_performance_comparison.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def _plot_spider_web_performance(
    results: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    """Create a spider web performance comparison figure.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    output_dir : Path
        Output directory for the figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple[Figure, Path]
        Figure and saved figure path.
    """
    metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cohen_kappa"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    num_vars = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    for idx, result in enumerate(results):
        values = [result.get(metric, 0) for metric in metrics]
        if "cohen_kappa" in metrics:
            kappa_idx = metrics.index("cohen_kappa")
            values[kappa_idx] = (values[kappa_idx] + 1) / 2
        values += values[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, values, "o-", color=color, linewidth=2.5, markersize=7, label=result["method"])
        ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    ax.grid(color="grey", linestyle="--", linewidth=0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=12)
    ax.tick_params(axis="x", pad=25)

    plt.suptitle("Multi-Dimensional Performance Comparison", fontsize=18, fontweight="bold", y=1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=11)

    fig.tight_layout(pad=2)
    fig_path = output_dir / "spider_web_performance_comparison.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def _plot_confusion_matrices(
    results: list[dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    """Create confusion matrices for each method.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    output_dir : Path
        Output directory for the figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple[Figure, Path]
        Figure and saved figure path.
    """
    n_methods = len(results)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows), squeeze=False)
    axes = axes.flatten()

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    for i, result in enumerate(results):
        ax = axes[i]
        cm = result["confusion_matrix"]
        labels = result["labels"]
        im = ax.imshow(cm, cmap="Blues")
        for i_cm in range(len(labels)):
            for j_cm in range(len(labels)):
                ax.text(j_cm, i_cm, cm[i_cm, j_cm], ha="center", va="center", color="black", fontweight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{result['method']}\n(Acc: {result['accuracy']:.3f})", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_methods, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "comprehensive_confusion_matrices.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def _plot_per_class_heatmaps(
    results: list[dict[str, Any]],
    cell_types: list[str],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    """Create precision/recall/F1 heatmaps for per-class performance.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    cell_types : list[str]
        Ordered list of cell types.
    output_dir : Path
        Output directory for the figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple[Figure, Path]
        Figure and saved figure path.
    """
    precision_matrix = _build_per_class_matrix(results, "per_class_precision", cell_types)
    recall_matrix = _build_per_class_matrix(results, "per_class_recall", cell_types)
    f1_matrix = _build_per_class_matrix(results, "per_class_f1", cell_types)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, max(8, len(cell_types) * 0.5)))
    fig.suptitle(
        f"Per-Class Performance (Filtered: {len(cell_types)} cell types)",
        fontsize=16,
        fontweight="bold",
    )

    sns.heatmap(precision_matrix, annot=True, fmt=".2f", cmap="Reds", ax=ax1, vmin=0, vmax=1)
    ax1.set_title("Precision by Cell Type")
    sns.heatmap(recall_matrix, annot=True, fmt=".2f", cmap="Greens", ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Recall by Cell Type")
    sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax3, vmin=0, vmax=1)
    ax3.set_title("F1-Score by Cell Type")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Method")
        ax.set_ylabel("Cell Type" if ax == ax1 else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "comprehensive_per_class_analysis.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


def _plot_average_per_class_metrics(
    results: list[dict[str, Any]],
    cell_types: list[str],
    output_dir: Path,
    dpi: int,
) -> tuple[Figure, Path]:
    """Create average per-class metric plots with error bars.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Computed method results.
    cell_types : list[str]
        Ordered list of cell types.
    output_dir : Path
        Output directory for the figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    tuple[Figure, Path]
        Figure and saved figure path.
    """
    methods = [r["method"] for r in results]
    method_metrics: dict[str, dict[str, list[float]]] = {m: {"precision": [], "recall": [], "f1": []} for m in methods}

    for result in results:
        method = result["method"]
        labels = result.get("labels", [])
        for i, label in enumerate(labels):
            if label in cell_types:
                method_metrics[method]["precision"].append(result["per_class_precision"][i])
                method_metrics[method]["recall"].append(result["per_class_recall"][i])
                method_metrics[method]["f1"].append(result["per_class_f1"][i])

    metric_stats: dict[str, dict[str, float]] = {}
    for method in methods:
        metric_stats[method] = {
            "precision_mean": float(np.mean(method_metrics[method]["precision"])) if method_metrics[method]["precision"] else 0,
            "precision_std": float(np.std(method_metrics[method]["precision"])) if method_metrics[method]["precision"] else 0,
            "recall_mean": float(np.mean(method_metrics[method]["recall"])) if method_metrics[method]["recall"] else 0,
            "recall_std": float(np.std(method_metrics[method]["recall"])) if method_metrics[method]["recall"] else 0,
            "f1_mean": float(np.mean(method_metrics[method]["f1"])) if method_metrics[method]["f1"] else 0,
            "f1_std": float(np.std(method_metrics[method]["f1"])) if method_metrics[method]["f1"] else 0,
        }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Average Per-Class Metrics Across Methods (Filtered: {len(cell_types)} cell types)",
        fontsize=16,
        fontweight="bold",
    )

    x = np.arange(len(methods))
    width = 0.6

    precision_means = [metric_stats[m]["precision_mean"] for m in methods]
    precision_stds = [metric_stats[m]["precision_std"] for m in methods]
    ax1.bar(x, precision_means, width, yerr=precision_stds, capsize=8, alpha=0.8, edgecolor="black", linewidth=1.5, color="coral", error_kw={"linewidth": 2})
    ax1.set_ylabel("Average Precision", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Methods", fontweight="bold", fontsize=12)
    ax1.set_title("Average Precision by Cell Type", fontweight="bold", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    recall_means = [metric_stats[m]["recall_mean"] for m in methods]
    recall_stds = [metric_stats[m]["recall_std"] for m in methods]
    ax2.bar(x, recall_means, width, yerr=recall_stds, capsize=8, alpha=0.8, edgecolor="black", linewidth=1.5, color="lightgreen", error_kw={"linewidth": 2})
    ax2.set_ylabel("Average Recall", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Methods", fontweight="bold", fontsize=12)
    ax2.set_title("Average Recall by Cell Type", fontweight="bold", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    f1_means = [metric_stats[m]["f1_mean"] for m in methods]
    f1_stds = [metric_stats[m]["f1_std"] for m in methods]
    ax3.bar(x, f1_means, width, yerr=f1_stds, capsize=8, alpha=0.8, edgecolor="black", linewidth=1.5, color="skyblue", error_kw={"linewidth": 2})
    ax3.set_ylabel("Average F1-Score", fontweight="bold", fontsize=12)
    ax3.set_xlabel("Methods", fontweight="bold", fontsize=12)
    ax3.set_title("Average F1-Score by Cell Type", fontweight="bold", fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha="right")
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = output_dir / "average_per_class_metrics_with_errorbars.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    return fig, fig_path


# =============================================================================
# UMAP EXPLORER - Publication-ready UMAP visualization & computation
# =============================================================================

# Available colormaps grouped by type for the UI
_CATEGORICAL_PALETTES = [
    "tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "Dark2",
    "Pastel1", "Pastel2", "Accent", "Bold",
]

_CONTINUOUS_CMAPS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "Reds", "Blues", "Greens", "Purples", "Oranges",
    "YlOrRd", "YlGnBu", "RdYlBu", "coolwarm", "bwr",
    "Spectral", "RdBu_r", "PiYG",
]


def _classify_column(series: pd.Series, max_categories: int = 50) -> str:
    """Classify a column as 'categorical' or 'continuous'."""
    if series.dtype == object or series.dtype.name == "category":
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    n_unique = series.nunique(dropna=True)
    if n_unique <= max_categories and pd.api.types.is_integer_dtype(series):
        return "categorical"
    return "continuous"


@st.cache_data(show_spinner="Computing UMAP embedding...")
def _compute_umap_embedding(
    marker_values_bytes: bytes,
    n_cells: int,
    n_markers: int,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    n_pcs: int,
    metric: str,
    random_seed: int,
    subsample_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute UMAP embedding from marker matrix. Returns (umap_coords, subsample_indices_or_None).

    The marker_values_bytes is the raw bytes of the numpy float32 array for cache hashing.
    """
    import scanpy as sc

    logger.info("=" * 80)
    logger.info("UMAP EXPLORER — Compute Embedding")
    logger.info("=" * 80)
    logger.info(f"Input matrix: {n_cells:,} cells x {n_markers} markers")
    logger.info(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, "
                f"metric={metric}, n_pcs={n_pcs}, seed={random_seed}")
    logger.info(f"Subsample: {subsample_size:,}" if subsample_size > 0 else "Subsample: OFF (all cells)")

    _t0 = time.time()

    X = np.frombuffer(marker_values_bytes, dtype=np.float32).reshape(n_cells, n_markers)

    need_subsample = subsample_size > 0 and n_cells > subsample_size
    if need_subsample:
        np.random.seed(random_seed)
        sub_idx = np.random.choice(n_cells, subsample_size, replace=False)
        sub_idx.sort()
        X_sub = X[sub_idx]
        logger.info(f"Step 1/5: Subsampled {subsample_size:,} / {n_cells:,} cells ({subsample_size/n_cells*100:.1f}%)")
    else:
        sub_idx = None
        X_sub = X
        logger.info(f"Step 1/5: No subsampling — using all {n_cells:,} cells")

    adata = sc.AnnData(X=X_sub.copy())
    adata.obs_names_make_unique()

    actual_pcs = min(n_pcs, min(adata.shape) - 1)
    logger.info(f"Step 2/5: Running PCA with {actual_pcs} components on {X_sub.shape[0]:,} cells...")
    _t1 = time.time()
    sc.tl.pca(adata, n_comps=actual_pcs, svd_solver="randomized", random_state=random_seed)
    logger.info(f"         PCA completed in {time.time() - _t1:.2f}s")

    logger.info(f"Step 3/5: Building neighbor graph (n_neighbors={n_neighbors}, metric={metric})...")
    _t2 = time.time()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_pcs, random_state=random_seed)
    logger.info(f"         Neighbors completed in {time.time() - _t2:.2f}s")

    logger.info(f"Step 4/5: Computing UMAP embedding (min_dist={min_dist})...")
    _t3 = time.time()
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_seed, n_components=n_components)
    logger.info(f"         UMAP completed in {time.time() - _t3:.2f}s")

    umap_coords_sub = adata.obsm["X_umap"]

    if need_subsample:
        # Project remaining cells via k-NN in PCA space
        from sklearn.neighbors import KNeighborsRegressor

        logger.info(f"Step 5/5: Projecting remaining {n_cells - subsample_size:,} cells via k-NN in PCA space...")
        _t4 = time.time()

        # Full PCA projection
        X_centered = X - np.mean(X_sub, axis=0)
        pca_full = X_centered @ adata.varm["PCs"]

        pca_train = pca_full[sub_idx]
        knn = KNeighborsRegressor(n_neighbors=min(15, len(pca_train)), weights="distance")
        knn.fit(pca_train, umap_coords_sub)

        all_umap = knn.predict(pca_full)
        # Overwrite subsample positions with exact values
        all_umap[sub_idx] = umap_coords_sub
        logger.info(f"         k-NN projection completed in {time.time() - _t4:.2f}s")
        logger.info(f"COMPLETED: UMAP embedding for {n_cells:,} cells in {time.time() - _t0:.2f}s total")
        logger.info(f"  Coordinate range — X: [{all_umap[:, 0].min():.2f}, {all_umap[:, 0].max():.2f}], "
                    f"Y: [{all_umap[:, 1].min():.2f}, {all_umap[:, 1].max():.2f}]")
        logger.info("=" * 80)
        return all_umap.astype(np.float32), sub_idx
    else:
        _umap_sub = np.asarray(umap_coords_sub)
        logger.info(f"Step 5/5: No projection needed (no subsampling)")
        logger.info(f"COMPLETED: UMAP embedding for {n_cells:,} cells in {time.time() - _t0:.2f}s total")
        logger.info(f"  Coordinate range — X: [{_umap_sub[:, 0].min():.2f}, {_umap_sub[:, 0].max():.2f}], "
                    f"Y: [{_umap_sub[:, 1].min():.2f}, {_umap_sub[:, 1].max():.2f}]")
        logger.info("=" * 80)
        return _umap_sub.astype(np.float32), None


def _make_umap_figure_categorical(
    umap_x: np.ndarray,
    umap_y: np.ndarray,
    labels: np.ndarray,
    palette_name: str,
    custom_color_map: dict | None,
    point_size: float,
    alpha: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
    title: str,
    show_legend: bool,
    legend_fontsize: int,
    legend_ncol: int,
    legend_markerscale: float,
    label_fontsize: int,
    title_fontsize: int,
    axis_off: bool,
    rasterized: bool,
    background_color: str,
    grid_alpha: float,
    order: list[str] | None,
) -> Figure:
    """Create a publication-quality UMAP scatter colored by categorical variable."""
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    unique_labels = sorted(pd.Series(labels).dropna().unique(), key=str)
    if order:
        ordered = [l for l in order if l in unique_labels]
        remainder = [l for l in unique_labels if l not in ordered]
        unique_labels = ordered + remainder

    # Build color map
    if custom_color_map:
        color_map = custom_color_map
    else:
        try:
            pal = sns.color_palette(palette_name, n_colors=max(len(unique_labels), 1))
        except Exception:
            pal = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))
        color_map = {label: pal[i % len(pal)] for i, label in enumerate(unique_labels)}

    # Plot each category (last plotted on top)
    for label in unique_labels:
        mask = labels == label
        if mask.sum() == 0:
            continue
        c = color_map.get(label, "#999999")
        ax.scatter(
            umap_x[mask], umap_y[mask],
            c=[c], s=point_size, alpha=alpha,
            label=f"{label} ({mask.sum():,})",
            edgecolors="none", rasterized=rasterized,
        )

    ax.set_xlabel("UMAP 1", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)

    if axis_off:
        ax.axis("off")
    else:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=label_fontsize - 2)

    if show_legend and len(unique_labels) <= 60:
        leg = ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=legend_fontsize, ncol=legend_ncol,
            markerscale=legend_markerscale, frameon=True,
            edgecolor="black", fancybox=False,
        )
        leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    return fig


def _make_umap_figure_continuous(
    umap_x: np.ndarray,
    umap_y: np.ndarray,
    values: np.ndarray,
    cmap_name: str,
    point_size: float,
    alpha: float,
    fig_width: float,
    fig_height: float,
    dpi: int,
    title: str,
    colorbar_label: str,
    vmin: float | None,
    vmax: float | None,
    label_fontsize: int,
    title_fontsize: int,
    axis_off: bool,
    rasterized: bool,
    background_color: str,
    grid_alpha: float,
) -> Figure:
    """Create publication-quality UMAP scatter colored by continuous variable."""
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    scatter = ax.scatter(
        umap_x, umap_y,
        c=values, cmap=cmap_name,
        s=point_size, alpha=alpha,
        vmin=vmin, vmax=vmax,
        edgecolors="none", rasterized=rasterized,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=label_fontsize, fontweight="bold")
    cbar.ax.tick_params(labelsize=label_fontsize - 2)

    ax.set_xlabel("UMAP 1", fontsize=label_fontsize, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=label_fontsize, fontweight="bold")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)

    if axis_off:
        ax.axis("off")
    else:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=label_fontsize - 2)

    fig.tight_layout()
    return fig


def _render_umap_explorer(
    project_root: Path,
    unified_csv_path: str,
    available_columns: list[str],
    cached_params: dict[str, Any],
) -> None:
    """Render the UMAP Explorer subtab inside Visualization."""
    st.markdown("### UMAP Explorer")
    st.caption(
        "Compute or load UMAP embeddings and create publication-ready scatter plots "
        "colored by any categorical or continuous column in the unified CSV."
    )

    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.error("Unified CSV file not found. Please set the global input path first.")
        return

    loaded_df = _get_loaded_visualization_df(unified_csv_path)
    if not isinstance(loaded_df, pd.DataFrame):
        st.info("Load the unified CSV once using 'Unified CSV Loader' above to activate UMAP Explorer.")
        return

    # ---- Load full unified data (cached) ----
    logger.info("UMAP Explorer: Loading unified CSV: %s", os.path.basename(unified_csv_path))
    try:
        df_full = loaded_df
    except Exception as e:
        st.error(f"Failed to load unified CSV: {e}")
        logger.error("UMAP Explorer: Failed to load CSV — %s", e)
        return

    n_total_cells = len(df_full)
    logger.info("UMAP Explorer: Loaded %s cells, %s columns", f"{n_total_cells:,}", len(df_full.columns))

    # ==================================================================
    # Section 0: DATA FILTERING
    # ==================================================================
    st.markdown("---")
    st.markdown("#### 0. Data Filtering")
    st.caption(
        "Apply filters to select a subset of cells before UMAP computation "
        "or visualization. Only the filtered cells will be processed."
    )

    # Determine filterable columns (categorical-like: object, bool, or numeric with <= 50 unique)
    _MAX_FILTER_CATS = 50
    filterable_cols: list[str] = []
    for _fc in df_full.columns:
        if df_full[_fc].dtype == object or df_full[_fc].dtype.name == "category":
            if df_full[_fc].nunique(dropna=True) <= _MAX_FILTER_CATS:
                filterable_cols.append(_fc)
        elif pd.api.types.is_bool_dtype(df_full[_fc]):
            filterable_cols.append(_fc)
        elif pd.api.types.is_numeric_dtype(df_full[_fc]):
            if df_full[_fc].nunique(dropna=True) <= _MAX_FILTER_CATS:
                filterable_cols.append(_fc)

    # Number of active filters (persisted in session state)
    if "umap_n_filters" not in st.session_state:
        st.session_state["umap_n_filters"] = 1
    n_filters: int = st.session_state["umap_n_filters"]

    # Collect filter specs
    filter_specs: list[tuple[str, list[str]]] = []

    for _fi in range(n_filters):
        filt_label = f"Filter {_fi + 1}" if n_filters > 1 else "Filter"
        _fc1, _fc2 = st.columns([1, 2])

        with _fc1:
            # Smart default: first filter defaults to 'tissue' if it exists
            _def_idx = 0
            _cached_col = cached_params.get(f"umap_filter_col_{_fi}")
            if _cached_col and _cached_col in filterable_cols:
                _def_idx = filterable_cols.index(_cached_col)
            elif _fi == 0 and "tissue" in filterable_cols:
                _def_idx = filterable_cols.index("tissue")

            _sel_col = st.selectbox(
                f"{filt_label} — Column",
                options=filterable_cols if filterable_cols else ["(no filterable columns)"],
                index=min(_def_idx, max(len(filterable_cols) - 1, 0)),
                key=f"umap_filt_col_{_fi}",
            )

        with _fc2:
            if _sel_col and _sel_col in df_full.columns:
                _uniq_vals = sorted(df_full[_sel_col].dropna().unique(), key=str)
                _uniq_str = [str(v) for v in _uniq_vals]

                # Smart default values
                _cached_vals = cached_params.get(f"umap_filter_vals_{_fi}", [])
                if not _cached_vals:
                    # First filter on 'tissue' → default to "1" if available
                    if _fi == 0 and _sel_col == "tissue":
                        if "1" in _uniq_str:
                            _cached_vals = ["1"]
                        elif 1 in _uniq_vals:
                            _cached_vals = ["1"]

                _default_sel = [v for v in _cached_vals if v in _uniq_str]

                _sel_vals = st.multiselect(
                    f"{filt_label} — Values (keep)",
                    options=_uniq_str,
                    default=_default_sel,
                    key=f"umap_filt_vals_{_fi}",
                    help=f"Select which {_sel_col} values to keep. Leave empty = no filter on this column.",
                )

                if _sel_vals:
                    filter_specs.append((_sel_col, _sel_vals))

    # Add / remove filter buttons
    _btn_c1, _btn_c2, _ = st.columns([1, 1, 4])
    with _btn_c1:
        if st.button("+ Add Filter", key="umap_add_filter"):
            st.session_state["umap_n_filters"] = n_filters + 1
            st.rerun()
    with _btn_c2:
        if n_filters > 1:
            if st.button("- Remove Filter", key="umap_rm_filter"):
                st.session_state["umap_n_filters"] = max(1, n_filters - 1)
                st.rerun()

    # Apply filters
    df = df_full.copy()
    _filter_summary: list[str] = []
    for _fcol, _fvals in filter_specs:
        _before = len(df)
        df = df[df[_fcol].astype(str).isin(_fvals)]
        _after = len(df)
        _filter_summary.append(f"`{_fcol}` in {_fvals}  ({_before:,} → {_after:,})")

    # Track original row indices (positions in df_full) for CSV write-back
    original_indices = df.index.values.copy()
    n_filtered_cells = len(df)

    if filter_specs:
        st.success(f"**{n_filtered_cells:,}** / {n_total_cells:,} cells after filtering.")
        for _part in _filter_summary:
            st.caption(f"  {_part}")
        logger.info("UMAP Explorer: Filters applied — %s / %s cells retained", f"{n_filtered_cells:,}", f"{n_total_cells:,}")
        for _part in _filter_summary:
            logger.info("  Filter: %s", _part.replace("`", ""))
        # Cache filter params
        for _fi in range(n_filters):
            cached_params[f"umap_filter_col_{_fi}"] = st.session_state.get(f"umap_filt_col_{_fi}", "")
            cached_params[f"umap_filter_vals_{_fi}"] = st.session_state.get(f"umap_filt_vals_{_fi}", [])
    else:
        st.info(f"No filters applied. Using all **{n_total_cells:,}** cells.")
        logger.info("UMAP Explorer: No filters — using all %s cells", f"{n_total_cells:,}")

    if n_filtered_cells == 0:
        st.error("No cells remain after filtering. Adjust filter criteria.")
        return

    # Reset index for clean positional work but remember originals
    df = df.reset_index(drop=True)

    # Detect existing UMAP columns
    umap_x_col = None
    umap_y_col = None
    for xc, yc in [("umap_x", "umap_y"), ("UMAP_1", "UMAP_2"), ("umap_1", "umap_2"), ("UMAP1", "UMAP2")]:
        if xc in df.columns and yc in df.columns:
            umap_x_col, umap_y_col = xc, yc
            break

    has_existing_umap = umap_x_col is not None

    # ---- Section 1: UMAP Coordinates ----
    st.markdown("---")
    st.markdown("#### 1. UMAP Coordinates")

    coord_tab_load, coord_tab_compute = st.tabs(["Load Existing", "Compute New"])

    # Session state for UMAP data
    if "umap_explorer_x" not in st.session_state:
        st.session_state["umap_explorer_x"] = None
        st.session_state["umap_explorer_y"] = None
        st.session_state["umap_explorer_index"] = None   # original row indices into df_full
        st.session_state["umap_explorer_source"] = None
        st.session_state["umap_explorer_n_cells"] = None

    with coord_tab_load:
        if has_existing_umap:
            st.success(f"Found existing UMAP columns: **{umap_x_col}**, **{umap_y_col}**")
            valid_mask = df[umap_x_col].notna() & df[umap_y_col].notna()
            n_valid = valid_mask.sum()
            n_nan = (~valid_mask).sum()
            st.write(f"Valid coordinates: **{n_valid:,}** / {n_filtered_cells:,} cells (after filtering)")
            if n_nan > 0:
                st.warning(f"{n_nan:,} filtered cells have missing UMAP coordinates and will be excluded from plots.")

            if st.button("Load UMAP from CSV", key="umap_load_btn"):
                logger.info("UMAP Explorer: Loading existing UMAP coordinates from columns '%s', '%s'", umap_x_col, umap_y_col)
                logger.info("  Valid coordinates: %s / %s cells", f"{n_valid:,}", f"{n_filtered_cells:,}")
                st.session_state["umap_explorer_x"] = df.loc[valid_mask, umap_x_col].values.astype(np.float32)
                st.session_state["umap_explorer_y"] = df.loc[valid_mask, umap_y_col].values.astype(np.float32)
                st.session_state["umap_explorer_index"] = np.asarray(original_indices)[np.asarray(valid_mask.values)]  # map to df_full rows
                st.session_state["umap_explorer_source"] = "loaded"
                st.session_state["umap_explorer_n_cells"] = n_filtered_cells
                logger.info("  UMAP coordinates loaded successfully")
                st.success(f"Loaded {n_valid:,} UMAP coordinates.")
                st.rerun()
        else:
            st.info("No existing UMAP coordinates found in the filtered data. Use **Compute New** to generate them.")

    with coord_tab_compute:
        st.markdown("##### UMAP Computation Parameters")

        # Marker selection for UMAP computation
        metadata_exclude = {
            "cell_id", "batch_id", "tissue", "x_centroid", "y_centroid",
            "area", "perimeter", "eccentricity", "solidity", "extent",
            "major_axis", "minor_axis", "orientation", "label", "index",
        }
        marker_candidates = [
            c for c in df.columns
            if c.lower() not in {m.lower() for m in metadata_exclude}
            and not c.startswith("umap_") and not c.startswith("UMAP")
            and not c.startswith("phenotype_")
            and not c.startswith("leiden_") and not c.startswith("som_")
            and not c.startswith("meta_cluster")
            and not c.endswith("_norm")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        # Smart defaults: prefer _corr markers
        default_markers = [m for m in marker_candidates if m.endswith("_corr")]
        if not default_markers:
            default_markers = marker_candidates[:20]

        with st.form("umap_compute_form", clear_on_submit=False):
            selected_markers = st.multiselect(
                "Markers for UMAP",
                options=marker_candidates,
                default=[m for m in cached_params.get("umap_markers", default_markers) if m in marker_candidates],
                help="Markers used as input features for UMAP dimensionality reduction.",
                key="umap_compute_markers",
            )

            cp1, cp2, cp3 = st.columns(3)
            with cp1:
                n_neighbors = st.number_input(
                    "n_neighbors", min_value=2, max_value=200,
                    value=int(cached_params.get("umap_n_neighbors", 15)),
                    help="Larger = more global structure; smaller = more local detail.",
                    key="umap_n_neighbors",
                )
                n_pcs = st.number_input(
                    "PCA components", min_value=2, max_value=100,
                    value=int(cached_params.get("umap_n_pcs", 20)),
                    help="Number of principal components before UMAP.",
                    key="umap_n_pcs",
                )
            with cp2:
                min_dist = st.number_input(
                    "min_dist", min_value=0.0, max_value=1.0,
                    value=float(cached_params.get("umap_min_dist", 0.1)),
                    step=0.05, format="%.2f",
                    help="Minimum distance between points. Lower = tighter clusters.",
                    key="umap_min_dist",
                )
                metric = st.selectbox(
                    "Distance metric",
                    ["euclidean", "cosine", "manhattan", "correlation"],
                    index=["euclidean", "cosine", "manhattan", "correlation"].index(
                        cached_params.get("umap_metric", "euclidean")
                    ),
                    key="umap_metric",
                )
            with cp3:
                # Bound follows the loaded CSV; a value remembered from a larger
                # file (or from the Tab 9 explorer, which shares this key) has to
                # be pulled into range before the widget is built.
                subsample_default = clamp_numeric_widget_state(
                    "umap_subsample", 0, int(n_filtered_cells),
                    fallback=int(cached_params.get("umap_subsample", min(150000, n_filtered_cells))),
                )
                subsample = st.number_input(
                    "Subsample size (0=all)",
                    min_value=0, max_value=n_filtered_cells,
                    value=int(subsample_default or 0),
                    step=10000,
                    help="Compute the UMAP on this many randomly chosen cells, then place every other cell next to its most similar computed cells. Faster on large tables. Set 0 to use all cells.",
                    key="umap_subsample",
                )
                random_seed = st.number_input(
                    "Random seed", min_value=0, max_value=99999,
                    value=int(cached_params.get("umap_seed", 42)),
                    key="umap_seed",
                )

            # Normalization toggle
            apply_zscore = st.checkbox(
                "Z-score normalize markers before UMAP",
                value=cached_params.get("umap_zscore", True),
                help="Recommended when markers have different scales.",
                key="umap_zscore_cb",
            )

            compute_btn = st.form_submit_button("Compute UMAP", type="primary")

        if compute_btn:
            if not selected_markers:
                st.error("Please select at least one marker for UMAP computation.")
            else:
                try:
                    logger.info("UMAP Explorer: Preparing data for computation")
                    logger.info("  Markers (%d): %s%s", len(selected_markers),
                                ", ".join(selected_markers[:8]),
                                f"... +{len(selected_markers)-8} more" if len(selected_markers) > 8 else "")
                    X_raw = df[selected_markers].fillna(0).values.astype(np.float32)
                    logger.info("  Matrix shape: %s x %s  (%.1f MB)",
                                f"{X_raw.shape[0]:,}", X_raw.shape[1],
                                X_raw.nbytes / 1024 / 1024)

                    if apply_zscore:
                        from sklearn.preprocessing import StandardScaler
                        logger.info("  Applying Z-score normalization...")
                        X_raw = StandardScaler().fit_transform(X_raw).astype(np.float32)
                        logger.info("  Z-score normalization complete")
                    else:
                        logger.info("  Z-score normalization: OFF")

                    marker_bytes = X_raw.tobytes()

                    with st.spinner(f"Computing UMAP on {n_filtered_cells:,} filtered cells ({len(selected_markers)} markers)..."):
                        umap_coords, sub_idx = _compute_umap_embedding(
                            marker_values_bytes=marker_bytes,
                            n_cells=n_filtered_cells,
                            n_markers=len(selected_markers),
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            n_components=2,
                            n_pcs=n_pcs,
                            metric=metric,
                            random_seed=random_seed,
                            subsample_size=subsample,
                        )

                    st.session_state["umap_explorer_x"] = umap_coords[:, 0]
                    st.session_state["umap_explorer_y"] = umap_coords[:, 1]
                    st.session_state["umap_explorer_index"] = original_indices  # map back to df_full rows
                    st.session_state["umap_explorer_source"] = "computed"
                    st.session_state["umap_explorer_n_cells"] = n_filtered_cells

                    # Cache selected params
                    cached_params["umap_markers"] = selected_markers
                    cached_params["umap_n_neighbors"] = n_neighbors
                    cached_params["umap_n_pcs"] = n_pcs
                    cached_params["umap_min_dist"] = min_dist
                    cached_params["umap_metric"] = metric
                    cached_params["umap_subsample"] = subsample
                    cached_params["umap_seed"] = random_seed
                    cached_params["umap_zscore"] = apply_zscore

                    logger.info("UMAP Explorer: Coordinates stored in session state")
                    st.success(f"UMAP computed for **{n_filtered_cells:,}** filtered cells.")
                    if sub_idx is not None:
                        st.caption(f"Direct UMAP on {len(sub_idx):,} subsampled cells; remaining projected via k-NN.")
                    st.rerun()
                except ImportError:
                    st.error("scanpy is required for UMAP computation. Install with: `pip install scanpy`")
                    logger.error("UMAP Explorer: scanpy not installed")
                except Exception as e:
                    st.error(f"UMAP computation failed: {e}")
                    logger.error(f"UMAP Explorer: Computation error — {e}")

    # ---- Early exit if no coordinates available ----
    ux = st.session_state.get("umap_explorer_x")
    uy = st.session_state.get("umap_explorer_y")
    u_idx = st.session_state.get("umap_explorer_index")   # original row indices in df_full
    u_source = st.session_state.get("umap_explorer_source")

    # Validate cached UMAP coordinates and indices to prevent out-of-bounds IndexError
    if ux is not None and u_idx is not None:
        is_invalid = False
        try:
            if len(ux) != len(u_idx):
                is_invalid = True
            elif np.any(u_idx >= len(df_full)) or np.any(u_idx < 0):
                is_invalid = True
        except Exception:
            is_invalid = True

        if is_invalid:
            logger.info("UMAP Explorer: Stale/out-of-bounds UMAP coordinates detected. Resetting cached coordinates.")
            st.session_state["umap_explorer_x"] = None
            st.session_state["umap_explorer_y"] = None
            st.session_state["umap_explorer_index"] = None
            st.session_state["umap_explorer_source"] = None
            st.session_state["umap_explorer_n_cells"] = None
            ux, uy, u_idx, u_source = None, None, None, None

    if ux is None or uy is None:
        st.markdown("---")
        st.info("Load or compute UMAP coordinates above to unlock plotting and export options.")
        return

    n_plot_cells = len(ux)
    st.markdown("---")
    st.markdown(f"#### 2. Visualization  &mdash;  {n_plot_cells:,} cells  ({u_source})")

    # Build sub-dataframe aligned to UMAP coordinates using df_full original indices
    if u_idx is not None:
        try:
            df_plot = df_full.iloc[u_idx].reset_index(drop=True).copy()
        except Exception as exc:
            logger.error(f"UMAP Explorer: Failed to index df_full with u_idx: {exc}")
            st.error(f"Failed to align UMAP coordinates: {exc}. Please reload or recompute UMAP.")
            df_plot = df.reset_index(drop=True).copy()
    else:
        df_plot = df.reset_index(drop=True).copy()

    # ---- Color column selection ----
    all_cols = list(df_plot.columns)
    phenotype_cols = [c for c in all_cols if c.startswith("phenotype_")]
    cluster_cols = [c for c in all_cols if "cluster" in c.lower()]
    numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_plot[c])]
    suggested_order = phenotype_cols + cluster_cols + [c for c in all_cols if c not in phenotype_cols + cluster_cols]

    color_col = st.selectbox(
        "Color by column",
        options=suggested_order,
        index=0 if suggested_order else 0,
        key="umap_color_col",
    )

    # Detect column type
    col_type = _classify_column(df_plot[color_col])

    # ---- Category include / exclude (for categorical columns) ----
    _excluded_categories: list[str] = []
    if col_type == "categorical":
        _all_cats = sorted(df_plot[color_col].dropna().astype(str).unique(), key=str)
        with st.expander("Include / Exclude categories", expanded=False):
            st.caption(
                "Deselect categories you want to **hide** from the UMAP plot. "
                "Only selected categories will be plotted."
            )
            _cached_inc = cached_params.get(f"umap_inc_cats_{color_col}")
            _default_sel = _cached_inc if (_cached_inc is not None and set(_cached_inc).issubset(set(_all_cats))) else _all_cats
            _included_cats = st.multiselect(
                "Categories to show",
                options=_all_cats,
                default=_default_sel,
                key=f"umap_inc_cats_{color_col}",
                help="Deselect categories to hide them from the plot.",
            )
            _excluded_categories = [c for c in _all_cats if c not in _included_cats]
            if _excluded_categories:
                st.info(f"Hiding {len(_excluded_categories)} categor{'y' if len(_excluded_categories)==1 else 'ies'}: {', '.join(_excluded_categories[:10])}{'...' if len(_excluded_categories)>10 else ''}")
                logger.info("UMAP Explorer: Excluding %d categories from '%s': %s",
                            len(_excluded_categories), color_col,
                            ", ".join(_excluded_categories[:10]))
            cached_params[f"umap_inc_cats_{color_col}"] = _included_cats

    # ---- Subsample for display ----
    st.markdown("##### Figure Settings")

    with st.form("umap_plot_form", clear_on_submit=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            display_default = clamp_numeric_widget_state(
                "umap_display_sub", 0, int(n_plot_cells),
                fallback=int(cached_params.get("umap_display_sub", 0)),
            )
            display_subsample = st.number_input(
                "Cells to plot (0=all)", min_value=0, max_value=n_plot_cells,
                value=int(display_default or 0),
                step=50000, key="umap_display_sub",
                help="Subsample for display only (does not affect saved coordinates).",
            )
            point_size = st.number_input(
                "Point size", min_value=0.05, max_value=50.0,
                value=float(cached_params.get("umap_pt_size", 0.5)),
                step=0.1, format="%.2f", key="umap_pt_size",
            )
            alpha_val = st.slider(
                "Opacity", 0.01, 1.0,
                value=float(cached_params.get("umap_alpha", 0.7)),
                step=0.05, key="umap_alpha",
            )
        with fc2:
            fig_width = st.number_input(
                "Figure width (in)", min_value=4.0, max_value=30.0,
                value=float(cached_params.get("umap_fig_w", 10.0)),
                step=0.5, key="umap_fig_w",
            )
            fig_height = st.number_input(
                "Figure height (in)", min_value=4.0, max_value=30.0,
                value=float(cached_params.get("umap_fig_h", 8.0)),
                step=0.5, key="umap_fig_h",
            )
            fig_dpi = st.number_input(
                "DPI", min_value=72, max_value=1200,
                value=int(cached_params.get("umap_dpi", 300)),
                step=50, key="umap_dpi",
            )
        with fc3:
            title_fontsize = st.number_input(
                "Title font size", 8, 36,
                value=int(cached_params.get("umap_title_fs", 16)),
                key="umap_title_fs",
            )
            label_fontsize = st.number_input(
                "Label font size", 6, 28,
                value=int(cached_params.get("umap_label_fs", 12)),
                key="umap_label_fs",
            )
            rasterize = st.checkbox(
                "Rasterize points",
                value=cached_params.get("umap_raster", True),
                help="Rasterize scatter points in vector output for smaller PDF/SVG files.",
                key="umap_raster",
            )

        # Advanced styling
        with st.expander("Advanced Styling", expanded=False):
            adv1, adv2 = st.columns(2)
            with adv1:
                bg_color = st.selectbox(
                    "Background",
                    ["white", "#f0f0f0", "black", "#1a1a2e", "none"],
                    index=["white", "#f0f0f0", "black", "#1a1a2e", "none"].index(
                        cached_params.get("umap_bg", "white")
                    ),
                    key="umap_bg",
                )
                grid_alpha = st.slider(
                    "Grid opacity", 0.0, 1.0,
                    value=float(cached_params.get("umap_grid_a", 0.3)),
                    step=0.05, key="umap_grid_a",
                )
                axis_off = st.checkbox(
                    "Hide axes",
                    value=cached_params.get("umap_axis_off", False),
                    key="umap_axis_off",
                )
            with adv2:
                if col_type == "categorical":
                    show_legend = st.checkbox("Show legend", value=cached_params.get("umap_legend", True), key="umap_legend")
                    legend_fs = st.number_input("Legend font size", 4, 20, value=int(cached_params.get("umap_leg_fs", 8)), key="umap_leg_fs")
                    legend_ncol = st.number_input("Legend columns", 1, 5, value=int(cached_params.get("umap_leg_nc", 1)), key="umap_leg_nc")
                    legend_ms = st.number_input("Legend marker scale", 1.0, 15.0, value=float(cached_params.get("umap_leg_ms", 4.0)), step=0.5, key="umap_leg_ms")
                else:
                    show_legend = False
                    legend_fs = 8
                    legend_ncol = 1
                    legend_ms = 4.0

        # Palette / colormap — initialize defaults to avoid unbound references
        palette_name = "tab10"
        cmap_name = "viridis"
        use_custom_vmin = False
        use_custom_vmax = False
        custom_vmin: float | None = None
        custom_vmax: float | None = None

        if col_type == "categorical":
            palette_name = st.selectbox(
                "Color palette",
                _CATEGORICAL_PALETTES,
                index=_CATEGORICAL_PALETTES.index(cached_params.get("umap_cat_pal", "tab10")) if cached_params.get("umap_cat_pal", "tab10") in _CATEGORICAL_PALETTES else 0,
                key="umap_cat_pal",
            )
        else:
            cmap_name = st.selectbox(
                "Colormap",
                _CONTINUOUS_CMAPS,
                index=_CONTINUOUS_CMAPS.index(cached_params.get("umap_cont_cmap", "viridis")) if cached_params.get("umap_cont_cmap", "viridis") in _CONTINUOUS_CMAPS else 0,
                key="umap_cont_cmap",
            )
            vmin_max_c1, vmin_max_c2 = st.columns(2)
            with vmin_max_c1:
                use_custom_vmin = st.checkbox("Custom vmin", value=False, key="umap_use_vmin")
                custom_vmin = st.number_input("vmin", value=0.0, key="umap_vmin") if use_custom_vmin else None
            with vmin_max_c2:
                use_custom_vmax = st.checkbox("Custom vmax", value=False, key="umap_use_vmax")
                custom_vmax = st.number_input("vmax", value=1.0, key="umap_vmax") if use_custom_vmax else None

        custom_title = st.text_input(
            "Plot title (leave blank for auto)",
            value=cached_params.get("umap_title", ""),
            key="umap_title_input",
        )

        plot_btn = st.form_submit_button("Generate UMAP Plot", type="primary")

    if plot_btn:
        logger.info("UMAP Explorer: Generating plot — color by '%s' (%s)", color_col, col_type)

        # Apply display subsample
        if display_subsample > 0 and display_subsample < n_plot_cells:
            np.random.seed(42)
            vis_idx = np.random.choice(n_plot_cells, display_subsample, replace=False)
            vis_idx.sort()
        else:
            vis_idx = np.arange(n_plot_cells)

        ux_vis = ux[vis_idx]
        uy_vis = uy[vis_idx]
        color_data = df_plot[color_col].iloc[vis_idx]

        # Apply category inclusion filter — keep only user-selected categories.
        # This also drops NA / blank cells that are not part of _included_cats,
        # even when _excluded_categories is empty (NA is never listed as a
        # selectable category because _all_cats uses dropna()).
        if col_type == "categorical":
            _has_na = color_data.isna().any() or (color_data.astype(str).str.strip() == "").any()
            if _excluded_categories or _has_na:
                _inc_set = set(_included_cats) if '_included_cats' in dir() and _included_cats else None
                if _inc_set:
                    _keep_mask = np.asarray(color_data.fillna("").astype(str).isin(_inc_set))
                else:
                    _keep_mask = np.asarray(~color_data.astype(str).isin(_excluded_categories))
                vis_idx = vis_idx[_keep_mask]
                ux_vis = ux_vis[_keep_mask]
                uy_vis = uy_vis[_keep_mask]
                color_data = color_data[_keep_mask]
                logger.info("  After category filtering: %s cells for plot", f"{len(vis_idx):,}")

        auto_title = f"UMAP — {color_col}  (n={len(vis_idx):,})"
        title = str(custom_title).strip() if custom_title else auto_title
        if not title:
            title = auto_title

        logger.info("  Plot settings: %s cells, size=%.2f, alpha=%.2f, %dx%d in, %d DPI",
                    f"{len(vis_idx):,}", point_size, alpha_val, int(fig_width), int(fig_height), fig_dpi)

        try:
            if col_type == "categorical":
                labels = np.asarray(color_data.fillna("NA").astype(str).values)
                fig = _make_umap_figure_categorical(
                    umap_x=ux_vis, umap_y=uy_vis, labels=labels,
                    palette_name=palette_name, custom_color_map=None,
                    point_size=point_size, alpha=alpha_val,
                    fig_width=fig_width, fig_height=fig_height, dpi=fig_dpi,
                    title=title, show_legend=show_legend,
                    legend_fontsize=legend_fs, legend_ncol=legend_ncol,
                    legend_markerscale=legend_ms, label_fontsize=label_fontsize,
                    title_fontsize=title_fontsize, axis_off=axis_off,
                    rasterized=rasterize, background_color=bg_color,
                    grid_alpha=grid_alpha, order=None,
                )
            else:
                vals = np.asarray(pd.to_numeric(color_data, errors="coerce").fillna(0).values)
                fig = _make_umap_figure_continuous(
                    umap_x=ux_vis, umap_y=uy_vis, values=vals,
                    cmap_name=cmap_name, point_size=point_size, alpha=alpha_val,
                    fig_width=fig_width, fig_height=fig_height, dpi=fig_dpi,
                    title=title, colorbar_label=color_col,
                    vmin=custom_vmin if use_custom_vmin else None,
                    vmax=custom_vmax if use_custom_vmax else None,
                    label_fontsize=label_fontsize, title_fontsize=title_fontsize,
                    axis_off=axis_off, rasterized=rasterize,
                    background_color=bg_color, grid_alpha=grid_alpha,
                )

            st.pyplot(fig)
            logger.info("  Plot rendered successfully")

            # Download buttons
            st.markdown("##### Export Figure")
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=fig_dpi, bbox_inches="tight")
                buf_png.seek(0)
                st.download_button("Download PNG", buf_png, file_name="umap_plot.png", mime="image/png", key="umap_dl_png")
            with dl2:
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format="pdf", dpi=fig_dpi, bbox_inches="tight")
                buf_pdf.seek(0)
                st.download_button("Download PDF", buf_pdf, file_name="umap_plot.pdf", mime="application/pdf", key="umap_dl_pdf")
            with dl3:
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format="svg", bbox_inches="tight")
                buf_svg.seek(0)
                st.download_button("Download SVG", buf_svg, file_name="umap_plot.svg", mime="image/svg+xml", key="umap_dl_svg")

            # Save to disk
            output_dir = project_root / "outputs" / "phenotyping" / "umap_plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_col = _sanitize_filename(color_col)
            save_path = output_dir / f"umap_{safe_col}.png"
            fig.savefig(save_path, dpi=fig_dpi, bbox_inches="tight")
            logger.info("  Auto-saved to: %s", save_path)
            st.caption(f"Auto-saved: `{save_path}`")

            plt.close(fig)

        except Exception as e:
            st.error(f"Plot generation failed: {e}")
            logger.error(f"UMAP Explorer: Plot error — {e}")

    # ---- Section 3: Save coordinates to CSV ----
    st.markdown("---")
    st.markdown("#### 3. Save UMAP Coordinates to Unified CSV")
    st.caption(
        "Write the current UMAP coordinates back into the unified CSV file "
        "so they are available for downstream analysis. "
        "Filtered cells get coordinates; unfiltered cells get NaN."
    )

    _u_idx_len = len(u_idx) if u_idx is not None else 0
    _is_partial = (u_idx is not None and _u_idx_len < n_total_cells)
    if _is_partial:
        st.info(
            f"Coordinates available for **{_u_idx_len:,}** / {n_total_cells:,} cells. "
            "Unmatched rows will be written as NaN."
        )

    with st.form("umap_save_form", clear_on_submit=False):
        sv_c1, sv_c2 = st.columns(2)
        with sv_c1:
            umap_x_save_col = st.text_input(
                "UMAP X column name",
                value=cached_params.get("umap_save_x_name", "umap_x"),
                key="umap_save_x_name",
                help="Column name for the X coordinate in the unified CSV.",
            )
        with sv_c2:
            umap_y_save_col = st.text_input(
                "UMAP Y column name",
                value=cached_params.get("umap_save_y_name", "umap_y"),
                key="umap_save_y_name",
                help="Column name for the Y coordinate in the unified CSV.",
            )

        overwrite_existing = st.checkbox(
            "Overwrite existing columns if they already exist",
            value=True,
            key="umap_save_overwrite",
            help="Uncheck to abort if the target columns already exist in the CSV.",
        )

        save_btn = st.form_submit_button("Save Coordinates to CSV", type="primary")

    if save_btn:
        try:
            logger.info("UMAP Explorer: Saving coordinates to CSV")
            logger.info("  Column names: '%s', '%s'", umap_x_save_col, umap_y_save_col)
            df_write = pd.read_csv(unified_csv_path, low_memory=False)

            # Safety check
            if not overwrite_existing:
                conflict = [c for c in [umap_x_save_col, umap_y_save_col] if c in df_write.columns]
                if conflict:
                    st.error(f"Columns already exist and overwrite is disabled: {conflict}")
                    st.stop()

            # Write coordinates at the correct original row positions
            df_write[umap_x_save_col] = np.nan
            df_write[umap_y_save_col] = np.nan

            if u_idx is not None:
                # u_idx contains original row indices in df_full / df_write
                valid_idx = u_idx[u_idx < len(df_write)]
                df_write.loc[valid_idx, umap_x_save_col] = ux[:len(valid_idx)]
                df_write.loc[valid_idx, umap_y_save_col] = uy[:len(valid_idx)]
            else:
                df_write[umap_x_save_col] = ux
                df_write[umap_y_save_col] = uy

            df_write.to_csv(unified_csv_path, index=False)

            n_written = int((~df_write[umap_x_save_col].isna()).sum())
            logger.info("  Saved %s coordinates in %s rows to %s",
                        f"{n_written:,}", f"{len(df_write):,}", os.path.basename(unified_csv_path))
            st.success(
                f"Saved **{umap_x_save_col}** and **{umap_y_save_col}** "
                f"to `{os.path.basename(unified_csv_path)}`  "
                f"({n_written:,} coordinates in {len(df_write):,} rows)."
            )
            # Cache column names
            cached_params["umap_save_x_name"] = umap_x_save_col
            cached_params["umap_save_y_name"] = umap_y_save_col

            # Invalidate cached df so next load picks up new columns
            for k in list(st.session_state.keys()):
                if "viz_unified" in str(k):
                    del st.session_state[k]
        except Exception as e:
            st.error(f"Failed to save coordinates: {e}")
            logger.error(f"UMAP Explorer: Save error — {e}")

    # ---- Section 4: Quick stats ----
    with st.expander("UMAP Coordinate Statistics", expanded=False):
        stat_df = pd.DataFrame({
            "UMAP_X": ux, "UMAP_Y": uy,
        })
        st.write(stat_df.describe().round(4))
        if u_idx is not None:
            st.caption(f"Covers {_u_idx_len:,} / {n_total_cells:,} total cells in the unified CSV.")


def render_visualization(project_root: Path, global_input_path: str | None) -> None:
    st.subheader("Visualization")
    st.markdown("Publication-ready comparative visualizations (Step8).")

    cache_dir = project_root / "outputs/phenotyping/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    viz_cache = cache_dir / "visualization_params.json"

    cached_params: dict[str, Any] = {}
    if viz_cache.exists():
        try:
            with open(viz_cache, "r") as f:
                cached_params = json.load(f)
        except Exception:
            cached_params = {}

    unified_csv_path = global_input_path or ""
    available_columns: list[str] = _get_csv_columns_fast(unified_csv_path)

    st.markdown("#### Unified CSV Loader")
    if not unified_csv_path or not os.path.exists(unified_csv_path):
        st.warning("Unified CSV file not found. Please set it in Global Input Configuration.")
    else:
        loaded_df = _get_loaded_visualization_df(unified_csv_path)
        c_load1, c_load2, c_load3 = st.columns([1, 1, 4])
        with c_load1:
            if st.button("Load Unified CSV", key="viz_shared_load_btn", type="primary"):
                try:
                    with st.spinner("Loading unified CSV into memory..."):
                        loaded_df = _get_cached_unified_df(unified_csv_path)
                    st.success(f"Loaded {len(loaded_df):,} rows into memory.")
                except Exception as exc:
                    st.error(f"Failed to load unified CSV: {exc}")
        with c_load2:
            if st.button("Unload CSV", key="viz_shared_unload_btn"):
                _clear_loaded_visualization_df()
                _log_terminal_event("Released unified CSV from visualization memory cache")
                st.success("Unified CSV cache cleared.")
                st.rerun()
        with c_load3:
            if isinstance(loaded_df, pd.DataFrame):
                st.caption(f"Loaded in memory: {len(loaded_df):,} rows | {len(loaded_df.columns)} columns")
            else:
                st.caption("CSV not loaded yet. Tabs open normally; click 'Load Unified CSV' once to reuse across all Visualization subtabs.")

    sub_tabs = st.tabs(["Proportion", "Pairwise", "F1 Score", "Heatmap", "Visual Validation", "UMAP Explorer", "Cell Visualization"])

    output_dir_default = cached_params.get(
        "viz_output_dir",
        str(project_root / "outputs/phenotyping/comparison"),
    )

    with sub_tabs[0]:
        st.markdown("### Proportion")
        st.caption("Phenotype proportion comparison across methods (Step8 cell 5).")

        notebook_methods = _extract_list_from_notebook(["columns_to_process", "PHENOTYPING_METHODS"])
        if "viz_prop_methods" not in st.session_state or st.session_state.get("viz_prop_last_file") != unified_csv_path:
            defaults = [m for m in notebook_methods if m in available_columns]
            if not defaults:
                defaults = [m for m in available_columns if m.startswith("phenotype_")]
            st.session_state["viz_prop_methods"] = defaults
            st.session_state["viz_prop_last_file"] = unified_csv_path

        with st.form("viz_prop_form", clear_on_submit=False):
            _viz_prop_opts = [c for c in available_columns if c.startswith("phenotype_")] or available_columns
            methods = st.multiselect(
                "Phenotype Columns",
                options=_viz_prop_opts,
                default=[m for m in st.session_state.get("viz_prop_methods", []) if m in _viz_prop_opts],
                key="viz_prop_methods_select",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                dpi_val = st.number_input(
                    "Figure DPI",
                    min_value=100,
                    max_value=600,
                    value=int(cached_params.get("viz_prop_dpi", 300)),
                    step=50,
                    key="viz_prop_dpi",
                )
            with col2:
                fig_width = st.number_input(
                    "Figure Width",
                    min_value=6.0,
                    max_value=30.0,
                    value=float(cached_params.get("viz_prop_width", 22.0)),
                    step=1.0,
                    key="viz_prop_width",
                )
            with col3:
                fig_height = st.number_input(
                    "Figure Height",
                    min_value=4.0,
                    max_value=15.0,
                    value=float(cached_params.get("viz_prop_height", 8.0)),
                    step=0.5,
                    key="viz_prop_height",
                )

            palette_mode = st.selectbox(
                "Color Palette",
                ["tab10", "Set2", "husl", "viridis", "manual"],
                index=["tab10", "Set2", "husl", "viridis", "manual"].index(cached_params.get("viz_prop_palette", "tab10")),
                key="viz_prop_palette",
            )

            custom_colors = []
            if palette_mode == "manual":
                for method in methods:
                    key = f"viz_prop_color_{method}"
                    default_color = cached_params.get(key, "#1f77b4")
                    custom_colors.append(st.color_picker(f"{method} Color", value=default_color, key=key))

            output_dir = st.text_input(
                "Output Directory",
                value=output_dir_default,
                key="viz_prop_output_dir",
            )

            save_cfg = st.form_submit_button("Save Configuration", type="primary")
            if save_cfg:
                st.session_state["viz_prop_methods"] = methods
                cached_params.update(
                    {
                        "viz_prop_methods": methods,
                        "viz_prop_dpi": dpi_val,
                        "viz_prop_width": fig_width,
                        "viz_prop_height": fig_height,
                        "viz_prop_palette": palette_mode,
                        "viz_output_dir": output_dir,
                    }
                )
                for method, color in zip(methods, custom_colors):
                    cached_params[f"viz_prop_color_{method}"] = color
                with open(viz_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
                st.success("Configuration saved.")

        if st.button("Generate Proportion Figure", type="primary", key="viz_prop_run"):
            log_lines: list[str] = []

            def _log(msg: str) -> None:
                line = f"[{_timestamp()}] {msg}"
                log_lines.append(line)
                _log_console(line)
                logger.info(line)

            if not unified_csv_path or not os.path.exists(unified_csv_path):
                st.error("Unified CSV file not found.")
            elif not st.session_state.get("viz_prop_methods"):
                st.error("Select at least one phenotype column.")
            else:
                _log("Starting proportion analysis")
                methods = st.session_state.get("viz_prop_methods", [])
                df = _get_cached_unified_df(unified_csv_path).copy()
                _log(f"Loaded data: {len(df):,} rows")
                records = []
                for method in methods:
                    if method not in df.columns:
                        continue
                    vc = df[method].value_counts(dropna=False)
                    total = int(vc.sum())
                    for phenotype, count in vc.items():
                        if pd.isna(phenotype):
                            continue
                        proportion = count / total if total else 0
                        records.append(
                            {
                                "method": method,
                                "phenotype": str(phenotype),
                                "count": int(count),
                                "total_cells": total,
                                "proportion": proportion,
                            }
                        )
                summary_long = pd.DataFrame(records)
                if summary_long.empty:
                    st.warning("No phenotypes found for the selected columns.")
                    _log("No phenotypes found for selected columns")
                else:
                    phenotype_order = (
                        summary_long.groupby("phenotype")["proportion"]
                        .mean()
                        .sort_values(ascending=False)
                        .index.tolist()
                    )
                    output_dir = cached_params.get("viz_output_dir", output_dir_default)
                    os.makedirs(output_dir, exist_ok=True)
                    long_path = Path(output_dir) / "phenotype_proportions_long.csv"
                    pivot_path = Path(output_dir) / "phenotype_proportions_pivot.csv"
                    summary_long.to_csv(long_path, index=False)
                    pivot_prop = summary_long.pivot(index="phenotype", columns="method", values="proportion").fillna(0)
                    pivot_prop.to_csv(pivot_path)
                    _log(f"Saved tables to {output_dir}")

                    palette_mode = cached_params.get("viz_prop_palette", "tab10")
                    if palette_mode == "manual":
                        palette = _build_palette([cached_params.get(f"viz_prop_color_{m}", "#1f77b4") for m in methods])
                    else:
                        palette = sns.color_palette(palette_mode, n_colors=len(methods))

                    fig, ax = plt.subplots(figsize=(cached_params.get("viz_prop_width", 22.0), cached_params.get("viz_prop_height", 8.0)))
                    sns.barplot(
                        data=summary_long,
                        x="phenotype",
                        y="proportion",
                        hue="method",
                        order=phenotype_order,
                        palette=palette,
                        ax=ax,
                    )
                    ax.set_title("Phenotype Proportions Across Methods")
                    ax.set_ylabel("Proportion of Cells")
                    ax.set_xlabel("Phenotype")
                    ax.tick_params(axis="x", rotation=45)
                    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
                    fig.tight_layout()
                    fig_path = Path(output_dir) / "phenotype_proportions_grouped.png"
                    fig.savefig(fig_path, dpi=int(cached_params.get("viz_prop_dpi", 300)))
                    st.pyplot(fig)
                    st.success(f"Saved figures and tables to {output_dir}")
                    _log(f"Saved figure: {fig_path}")

            if log_lines:
                st.markdown("#### Proportion Log")
                st.code("\n".join(log_lines))

    with sub_tabs[1]:
        st.markdown("### Pairwise")
        st.caption("Pairwise phenotype agreement heatmap (Step8 cell 7).")

        notebook_methods = _extract_list_from_notebook(["METHODS_TO_COMPARE", "PHENOTYPING_METHODS"])
        if "viz_pair_methods" not in st.session_state or st.session_state.get("viz_pair_last_file") != unified_csv_path:
            defaults = [m for m in notebook_methods if m in available_columns]
            if not defaults:
                defaults = [m for m in available_columns if m.startswith("phenotype_")]
            st.session_state["viz_pair_methods"] = defaults
            st.session_state["viz_pair_last_file"] = unified_csv_path

        with st.form("viz_pair_form", clear_on_submit=False):
            _viz_pair_opts = [c for c in available_columns if c.startswith("phenotype_")] or available_columns
            methods = st.multiselect(
                "Phenotype Columns",
                options=_viz_pair_opts,
                default=[m for m in st.session_state.get("viz_pair_methods", []) if m in _viz_pair_opts],
                key="viz_pair_methods_select",
            )
            dpi_val = st.number_input(
                "Figure DPI",
                min_value=100,
                max_value=600,
                value=int(cached_params.get("viz_pair_dpi", 300)),
                step=50,
                key="viz_pair_dpi",
            )
            output_dir = st.text_input(
                "Output Directory",
                value=output_dir_default,
                key="viz_pair_output_dir",
            )

            save_cfg = st.form_submit_button("Save Configuration", type="primary")
            if save_cfg:
                st.session_state["viz_pair_methods"] = methods
                cached_params.update(
                    {
                        "viz_pair_methods": methods,
                        "viz_pair_dpi": dpi_val,
                        "viz_output_dir": output_dir,
                    }
                )
                with open(viz_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
                st.success("Configuration saved.")

        if st.button("Generate Pairwise Heatmap", type="primary", key="viz_pair_run"):
            log_lines: list[str] = []

            def _log(msg: str) -> None:
                line = f"[{_timestamp()}] {msg}"
                log_lines.append(line)
                _log_console(line)
                logger.info(line)

            if not unified_csv_path or not os.path.exists(unified_csv_path):
                st.error("Unified CSV file not found.")
            elif not st.session_state.get("viz_pair_methods"):
                st.error("Select at least one phenotype column.")
            else:
                _log("Starting pairwise similarity analysis")
                methods = st.session_state.get("viz_pair_methods", [])
                df = _get_cached_unified_df(unified_csv_path).copy()
                _log(f"Loaded data: {len(df):,} rows")
                df = df[methods]
                similarity_matrix = pd.DataFrame(
                    np.zeros((len(methods), len(methods))),
                    index=methods,
                    columns=methods,
                )
                for i, m1 in enumerate(methods):
                    for m2 in methods[i:]:
                        filtered = df[[m1, m2]].dropna()
                        s1 = filtered[m1]
                        if isinstance(s1, pd.DataFrame):
                            s1 = s1.iloc[:, 0]
                        s2 = filtered[m2]
                        if isinstance(s2, pd.DataFrame):
                            s2 = s2.iloc[:, 0]
                        similarity = float(((s1 == s2).sum() / len(s1)) * 100) if len(s1) else 0
                        similarity_matrix.at[m1, m2] = similarity
                        similarity_matrix.at[m2, m1] = similarity
                _log("Computed similarity matrix")

                output_dir = cached_params.get("viz_output_dir", output_dir_default)
                os.makedirs(output_dir, exist_ok=True)
                matrix_csv_path = Path(output_dir) / "pairwise_similarity_matrix.csv"
                similarity_matrix.to_csv(matrix_csv_path)
                _log(f"Saved matrix: {matrix_csv_path}")

                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    similarity_matrix,
                    mask=mask,
                    annot=True,
                    fmt=".1f",
                    annot_kws={"size": 12},
                    cmap="YlGnBu",
                    linewidths=0.5,
                    vmin=0,
                    vmax=100,
                    cbar_kws={"label": "Similarity Percentage (%)"},
                    ax=ax,
                )
                ax.set_title("Pairwise Method Similarity (% of matching phenotypes)")
                ax.set_xlabel("")
                ax.set_ylabel("")
                fig.tight_layout()
                heatmap_path = Path(output_dir) / "pairwise_similarity_heatmap.png"
                fig.savefig(heatmap_path, dpi=int(cached_params.get("viz_pair_dpi", 300)))
                st.pyplot(fig)
                st.success(f"Saved heatmap and CSV to {output_dir}")
                _log(f"Saved heatmap: {heatmap_path}")

            if log_lines:
                st.markdown("#### Pairwise Log")
                st.code("\n".join(log_lines))

    with sub_tabs[2]:
        st.markdown("### F1 Score")
        st.caption("Validation against ground truth with optional consensus analysis (Step8 cell 8).")

        notebook_methods = _extract_list_from_notebook(["PHENOTYPING_METHODS"])
        notebook_filter = _extract_list_defaults_from_notebook("FILTER_OUT_PHENOTYPES")
        gt_default = _extract_scalar_from_notebook("GROUND_TRUTH_COLUMN") or "phenotype_ref1"
        figure_options = [
            "Per-class F1 heatmap",
            "Comprehensive performance comparison",
            "Spider web performance comparison",
            "Confusion matrices",
            "Per-class precision/recall/F1 heatmaps",
            "Average per-class metrics (error bars)",
        ]
        default_figures = cached_params.get("viz_f1_figures", ["Per-class F1 heatmap"])
        if not isinstance(default_figures, list):
            default_figures = ["Per-class F1 heatmap"]
        default_figures = [f for f in default_figures if f in figure_options]

        if "viz_f1_methods" not in st.session_state or st.session_state.get("viz_f1_last_file") != unified_csv_path:
            defaults = [m for m in notebook_methods if m in available_columns]
            if not defaults:
                defaults = [m for m in available_columns if m.startswith("phenotype_")]
            st.session_state["viz_f1_methods"] = defaults
            st.session_state["viz_f1_last_file"] = unified_csv_path

        # Auto-detect consensus column from unified CSV
        _consensus_flag_candidates = [c for c in available_columns if c.lower() in ("consensus", "is_consensus")]
        if not _consensus_flag_candidates:
            _consensus_flag_candidates = [c for c in available_columns if "consensus" in c.lower() and c not in [cc for cc in available_columns if cc.startswith("phenotype_")]]
        _default_consensus_flag = "consensus" if "consensus" in available_columns else (_consensus_flag_candidates[0] if _consensus_flag_candidates else "consensus")

        # Auto-detect default consensus phenotype column
        _default_consensus_pheno = "phenotype_ml_consensus" if "phenotype_ml_consensus" in available_columns else (gt_default)

        with st.form("viz_f1_form", clear_on_submit=False):
            _viz_f1_opts = [c for c in available_columns if c.startswith("phenotype_")] or available_columns
            methods = st.multiselect(
                "Phenotype Columns",
                options=_viz_f1_opts,
                default=[m for m in st.session_state.get("viz_f1_methods", []) if m in _viz_f1_opts],
                key="viz_f1_methods_select",
            )
            gt_candidates = [c for c in available_columns if c.startswith("phenotype_")]
            if gt_default not in gt_candidates and gt_candidates:
                gt_default = gt_candidates[0]
            ground_truth_col = st.selectbox(
                "Ground Truth Column",
                options=gt_candidates or available_columns,
                index=gt_candidates.index(gt_default) if gt_default in gt_candidates else 0,
                key="viz_f1_gt_col",
            )
            dataset_id_col = st.text_input(
                "Dataset ID Column",
                value=cached_params.get("viz_f1_dataset_col", "batch_id"),
                key="viz_f1_dataset_col",
                help="Column identifying batches/samples. Used for per-dataset breakdowns if needed.",
            )

            st.markdown("**Consensus / Non-Consensus Stratification**")
            f1_c1, f1_c2 = st.columns(2)
            with f1_c1:
                include_consensus = st.checkbox(
                    "Include Consensus Analysis",
                    value=bool(cached_params.get("viz_f1_include_consensus", False)),
                    key="viz_f1_include_consensus",
                    help="Add a 'Consensus' entry to the F1 analysis using only cells flagged as consensus (value=1).",
                )
            with f1_c2:
                include_nonconsensus = st.checkbox(
                    "Include Non-Consensus Analysis",
                    value=bool(cached_params.get("viz_f1_include_nonconsensus", False)),
                    key="viz_f1_include_nonconsensus",
                    help="Add a 'Non-Consensus' entry to the F1 analysis using only cells flagged as non-consensus (value=0).",
                )

            # Conditional consensus configuration
            if include_consensus or include_nonconsensus:
                st.markdown("**Consensus Column Configuration**")
                _cons_col_options = _consensus_flag_candidates if _consensus_flag_candidates else available_columns
                _cons_col_default = cached_params.get("viz_f1_consensus_flag_col", _default_consensus_flag)
                if _cons_col_default not in _cons_col_options:
                    _cons_col_options = [_cons_col_default] + _cons_col_options
                consensus_flag_col = st.selectbox(
                    "Consensus Flag Column",
                    options=_cons_col_options,
                    index=_cons_col_options.index(_cons_col_default) if _cons_col_default in _cons_col_options else 0,
                    key="viz_f1_consensus_flag_col",
                    help="Binary column where 1 = consensus cell, 0 = non-consensus cell.",
                )
                _pheno_candidates = [c for c in available_columns if c.startswith("phenotype_")]
                _cons_pheno_default = cached_params.get("viz_f1_consensus_pheno_col", _default_consensus_pheno)
                if _cons_pheno_default not in _pheno_candidates and _pheno_candidates:
                    _cons_pheno_default = _pheno_candidates[0]
                consensus_phenotype_col = st.selectbox(
                    "Consensus Phenotype Column",
                    options=_pheno_candidates or available_columns,
                    index=_pheno_candidates.index(_cons_pheno_default) if _cons_pheno_default in _pheno_candidates else 0,
                    key="viz_f1_consensus_pheno_col",
                    help="Phenotype column used for consensus/non-consensus cells (e.g. 'phenotype_ml_consensus').",
                )
                st.caption(f"Consensus cells: {consensus_flag_col} == 1 → phenotypes from '{consensus_phenotype_col}'")
                st.caption(f"Non-consensus cells: {consensus_flag_col} == 0 → phenotypes from '{consensus_phenotype_col}'")
            else:
                consensus_flag_col = _default_consensus_flag
                consensus_phenotype_col = _default_consensus_pheno

            st.divider()
            output_dir = st.text_input(
                "Output Directory",
                value=output_dir_default,
                key="viz_f1_output_dir",
            )
            f1_dpi_c1, f1_dpi_c2 = st.columns(2)
            with f1_dpi_c1:
                dpi_val = st.number_input(
                    "Figure DPI",
                    min_value=100,
                    max_value=600,
                    value=int(cached_params.get("viz_f1_dpi", 300)),
                    step=50,
                    key="viz_f1_dpi",
                )
            with f1_dpi_c2:
                fig_row_height = st.number_input(
                    "Row Height (inches)",
                    min_value=0.2,
                    max_value=1.5,
                    value=float(cached_params.get("viz_f1_row_height", 0.45)),
                    step=0.05,
                    key="viz_f1_row_height",
                    help="Height per cell-type row in heatmap figures.",
                )

            st.markdown("**Publication Column Sizing**")
            f1_lock_c1, f1_lock_c2 = st.columns([1, 1])
            with f1_lock_c1:
                lock_col_width = st.checkbox(
                    "Lock Column Width",
                    value=bool(cached_params.get("viz_f1_lock_col_width", False)),
                    key="viz_f1_lock_col_width",
                    help="When enabled, every method column has the same fixed width regardless of how many methods are selected. Ideal for publication consistency.",
                )
            with f1_lock_c2:
                col_width_inches = st.number_input(
                    "Column Width (inches)",
                    min_value=0.5,
                    max_value=6.0,
                    value=float(cached_params.get("viz_f1_col_width", 2.0)),
                    step=0.25,
                    key="viz_f1_col_width",
                    help="Fixed width per method column when Lock Column Width is enabled.",
                    disabled=not lock_col_width,
                )

            st.markdown("**Heatmap Cell Annotations**")
            f1_annot_c1, f1_annot_c2 = st.columns([1, 1])
            with f1_annot_c1:
                show_annot = st.checkbox(
                    "Show Cell Values",
                    value=bool(cached_params.get("viz_f1_show_annot", True)),
                    key="viz_f1_show_annot",
                    help="Display numeric values inside heatmap cells.",
                )
            with f1_annot_c2:
                annot_fontsize = st.number_input(
                    "Annotation Font Size",
                    min_value=4,
                    max_value=24,
                    value=int(cached_params.get("viz_f1_annot_fontsize", 8)),
                    step=1,
                    key="viz_f1_annot_fontsize",
                    help="Font size for the numeric values inside heatmap cells.",
                    disabled=not show_annot,
                )

            selected_figures = st.multiselect(
                "Figures to Generate",
                options=figure_options,
                default=default_figures,
                key="viz_f1_figures",
            )

            save_cfg = st.form_submit_button("Save Configuration", type="primary")
            if save_cfg:
                st.session_state["viz_f1_methods"] = methods
                cached_params.update(
                    {
                        "viz_f1_methods": methods,
                        "viz_f1_gt_col": ground_truth_col,
                        "viz_f1_dataset_col": dataset_id_col,
                        "viz_f1_include_consensus": include_consensus,
                        "viz_f1_include_nonconsensus": include_nonconsensus,
                        "viz_f1_consensus_flag_col": consensus_flag_col,
                        "viz_f1_consensus_pheno_col": consensus_phenotype_col,
                        "viz_f1_dpi": dpi_val,
                        "viz_f1_row_height": float(fig_row_height),
                        "viz_f1_lock_col_width": lock_col_width,
                        "viz_f1_col_width": float(col_width_inches),
                        "viz_f1_show_annot": show_annot,
                        "viz_f1_annot_fontsize": int(annot_fontsize),
                        "viz_f1_figures": selected_figures,
                        "viz_output_dir": output_dir,
                    }
                )
                with open(viz_cache, "w") as f:
                    json.dump(cached_params, f, indent=2)
                st.success("Configuration saved.")

        st.markdown("#### Filter Out Phenotypes")
        if st.button("Scan Phenotypes", key="viz_f1_scan"):
            if not unified_csv_path or not os.path.exists(unified_csv_path):
                st.error("Unified CSV file not found.")
            elif not st.session_state.get("viz_f1_methods"):
                st.error("Select phenotype columns before scanning.")
            else:
                df = _get_cached_unified_df(unified_csv_path).copy()
                methods = st.session_state.get("viz_f1_methods", [])
                phenotypes = set()
                for method in methods:
                    if method in df.columns:
                        phenotypes.update(df[method].dropna().astype(str).unique().tolist())
                phenotypes = sorted(phenotypes)
                rows = []
                for pheno in phenotypes:
                    row = {"Exclude": pheno in notebook_filter, "Phenotype": pheno}
                    for method in methods:
                        if method in df.columns:
                            vc = df[method].value_counts(dropna=False)
                            total = int(vc.sum()) if len(vc) else 0
                            count = int(vc.get(pheno, 0))
                            row[method] = f"{(count / total * 100):.2f}%" if total else "0.00%"
                    rows.append(row)
                st.session_state["viz_f1_filter_table"] = pd.DataFrame(rows)

        filter_table = st.session_state.get("viz_f1_filter_table")
        if isinstance(filter_table, pd.DataFrame) and not filter_table.empty:
            with st.form("viz_f1_filter_form", clear_on_submit=False):
                edited = st.data_editor(
                    filter_table,
                    column_config={
                        "Exclude": st.column_config.CheckboxColumn("Exclude", default=False),
                        "Phenotype": st.column_config.TextColumn("Phenotype", disabled=True),
                    },
                    disabled=["Phenotype"],
                    hide_index=True,
                    width="stretch",
                    key="viz_f1_filter_editor",
                )
                save_filter = st.form_submit_button("Save Filter Selection", type="primary")
                if save_filter:
                    st.session_state["viz_f1_filter_table"] = edited
                    exclude_list = edited.loc[edited["Exclude"], "Phenotype"].astype(str).tolist()
                    st.session_state["viz_f1_exclude"] = exclude_list
                    st.success("Filter selection saved.")
        else:
            st.info("Scan phenotypes to build the filter table.")

        if st.button("Run F1 Analysis", type="primary", key="viz_f1_run"):
            log_lines: list[str] = []

            def _log(msg: str) -> None:
                line = f"[{_timestamp()}] {msg}"
                log_lines.append(line)
                _log_console(line)
                logger.info(line)

            if not unified_csv_path or not os.path.exists(unified_csv_path):
                st.error("Unified CSV file not found.")
            elif not st.session_state.get("viz_f1_methods"):
                st.error("Select phenotype columns.")
            else:
                _log("Starting F1 analysis")
                df = _get_cached_unified_df(unified_csv_path).copy()
                methods = st.session_state.get("viz_f1_methods", [])
                ground_truth_col = cached_params.get("viz_f1_gt_col", gt_default)
                output_dir = cached_params.get("viz_output_dir", output_dir_default)
                os.makedirs(output_dir, exist_ok=True)
                exclude_list = st.session_state.get("viz_f1_exclude", notebook_filter)

                # Consensus stratification settings
                _f1_include_consensus = bool(cached_params.get("viz_f1_include_consensus", False))
                _f1_include_nonconsensus = bool(cached_params.get("viz_f1_include_nonconsensus", False))
                _f1_consensus_flag_col = cached_params.get("viz_f1_consensus_flag_col", "consensus")
                _f1_consensus_pheno_col = cached_params.get("viz_f1_consensus_pheno_col", "phenotype_ml_consensus")

                status = st.status("Running F1 analysis...", expanded=True)
                status.write(f"[{_timestamp()}] Preparing data")
                df = _prepare_filtered_df(df, ground_truth_col)
                _log(f"Prepared data: {len(df):,} rows")

                if exclude_list:
                    df = df[~df[ground_truth_col].isin(exclude_list)].copy()
                    _log(f"Filtered out phenotypes: {len(exclude_list)}")
                status.write(f"[{_timestamp()}] Computing metrics for {len(methods)} methods")

                individual_results = _compute_method_metrics(df, ground_truth_col, methods)
                if not individual_results:
                    status.warning("No valid method results found.")
                    _log("No valid method results found")
                    status.update(label="F1 analysis finished", state="complete")
                    return
                _log(f"Computed results for {len(individual_results)} methods")

                # --- Consensus / Non-Consensus stratified analysis ---
                if (_f1_include_consensus or _f1_include_nonconsensus) and _f1_consensus_flag_col in df.columns and _f1_consensus_pheno_col in df.columns:
                    status.write(f"[{_timestamp()}] Computing consensus-stratified metrics...")
                    _cons_flag = df[_f1_consensus_flag_col].astype(float)

                    if _f1_include_consensus:
                        df_cons = df[_cons_flag == 1].copy()
                        if not df_cons.empty and _f1_consensus_pheno_col in df_cons.columns:
                            cons_results = _compute_method_metrics(df_cons, ground_truth_col, [_f1_consensus_pheno_col])
                            for r in cons_results:
                                r["method"] = f"{_f1_consensus_pheno_col} (Consensus)"
                            individual_results.extend(cons_results)
                            _log(f"Consensus analysis: {len(df_cons):,} cells")
                        else:
                            _log("Consensus analysis skipped: no consensus cells or phenotype column missing")

                    if _f1_include_nonconsensus:
                        df_noncons = df[_cons_flag == 0].copy()
                        if not df_noncons.empty and _f1_consensus_pheno_col in df_noncons.columns:
                            noncons_results = _compute_method_metrics(df_noncons, ground_truth_col, [_f1_consensus_pheno_col])
                            for r in noncons_results:
                                r["method"] = f"{_f1_consensus_pheno_col} (Non-Consensus)"
                            individual_results.extend(noncons_results)
                            _log(f"Non-consensus analysis: {len(df_noncons):,} cells")
                        else:
                            _log("Non-consensus analysis skipped: no non-consensus cells or phenotype column missing")
                elif (_f1_include_consensus or _f1_include_nonconsensus):
                    missing_cols = []
                    if _f1_consensus_flag_col not in df.columns:
                        missing_cols.append(_f1_consensus_flag_col)
                    if _f1_consensus_pheno_col not in df.columns:
                        missing_cols.append(_f1_consensus_pheno_col)
                    _log(f"Consensus stratification skipped: missing columns {missing_cols}")
                    status.warning(f"Consensus columns not found in data: {missing_cols}. Stratification skipped.")

                summary_df = pd.DataFrame(individual_results)
                st.markdown("#### Method Summary")
                st.dataframe(
                    summary_df[["method", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "cohen_kappa", "matthews_corr", "n_cells", "coverage"]],
                    width="stretch",
                    hide_index=True,
                )

                selected_figures = st.session_state.get("viz_f1_figures", default_figures)
                if not selected_figures:
                    st.warning("No figures selected. Choose at least one figure to generate.")
                    _log("No figures selected")
                else:
                    cell_types = _collect_filtered_cell_types(individual_results, exclude_list)

                    # Publication sizing parameters
                    _f1_dpi = int(cached_params.get("viz_f1_dpi", 300))
                    _f1_row_h = float(cached_params.get("viz_f1_row_height", 0.45))
                    _f1_lock = bool(cached_params.get("viz_f1_lock_col_width", False))
                    _f1_col_w = float(cached_params.get("viz_f1_col_width", 2.0))
                    _f1_show_annot = bool(cached_params.get("viz_f1_show_annot", True))
                    _f1_annot_fs = int(cached_params.get("viz_f1_annot_fontsize", 8))
                    _f1_annot_kws = {"size": _f1_annot_fs} if _f1_show_annot else None
                    _n_methods = len(individual_results)
                    _n_rows_ct = max(1, len(cell_types))
                    _ylabel_pad = 2.5  # inches reserved for y-axis labels

                    def _heatmap_figsize(n_cols: int, n_rows: int) -> tuple[float, float]:
                        """Compute heatmap figure size respecting lock-column-width setting."""
                        h = max(6.0, n_rows * _f1_row_h)
                        if _f1_lock:
                            w = _ylabel_pad + n_cols * _f1_col_w
                        else:
                            w = max(8.0, _ylabel_pad + n_cols * 1.8)
                        return (w, h)

                    if "Per-class F1 heatmap" in selected_figures:
                        f1_matrix = _build_per_class_matrix(individual_results, "per_class_f1", cell_types)
                        if not f1_matrix.empty:
                            _fw, _fh = _heatmap_figsize(f1_matrix.shape[1], f1_matrix.shape[0])
                            fig, ax = plt.subplots(figsize=(_fw, _fh))
                            sns.heatmap(
                                f1_matrix, annot=_f1_show_annot, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
                                square=_f1_lock,
                                annot_kws=_f1_annot_kws,
                            )
                            ax.set_title("Per-Class F1 Scores", fontsize=14, fontweight="bold")
                            ax.set_xlabel("Method", fontsize=11, fontweight="bold")
                            ax.set_ylabel("Cell Type", fontsize=11, fontweight="bold")
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
                            fig.tight_layout()
                            heatmap_path = Path(output_dir) / "per_class_f1_scores.png"
                            fig.savefig(heatmap_path, dpi=_f1_dpi, bbox_inches="tight")
                            st.pyplot(fig)
                            plt.close(fig)
                            _log(f"Saved heatmap: {heatmap_path}")
                        else:
                            _log("Per-class F1 heatmap skipped: no cell types after filtering")

                    if "Comprehensive performance comparison" in selected_figures:
                        fig, fig_path = _plot_comprehensive_performance(
                            individual_results, Path(output_dir), _f1_dpi,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        _log(f"Saved comprehensive comparison: {fig_path}")

                    if "Spider web performance comparison" in selected_figures:
                        fig, fig_path = _plot_spider_web_performance(
                            individual_results, Path(output_dir), _f1_dpi,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        _log(f"Saved spider web comparison: {fig_path}")

                    if "Confusion matrices" in selected_figures:
                        fig, fig_path = _plot_confusion_matrices(
                            individual_results, Path(output_dir), _f1_dpi,
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        _log(f"Saved confusion matrices: {fig_path}")

                    if "Per-class precision/recall/F1 heatmaps" in selected_figures:
                        if cell_types:
                            _prec_m = _build_per_class_matrix(individual_results, "per_class_precision", cell_types)
                            _rec_m = _build_per_class_matrix(individual_results, "per_class_recall", cell_types)
                            _f1m = _build_per_class_matrix(individual_results, "per_class_f1", cell_types)
                            _single_w, _single_h = _heatmap_figsize(_n_methods, _n_rows_ct)
                            _triple_w = _single_w * 3 if _f1_lock else 24
                            _triple_h = _single_h if _f1_lock else max(8, _n_rows_ct * 0.5)
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(_triple_w, _triple_h))
                            fig.suptitle(
                                f"Per-Class Performance (Filtered: {len(cell_types)} cell types)",
                                fontsize=16, fontweight="bold",
                            )
                            sns.heatmap(_prec_m, annot=_f1_show_annot, fmt=".2f", cmap="Reds", ax=ax1, vmin=0, vmax=1, square=_f1_lock, annot_kws=_f1_annot_kws)
                            ax1.set_title("Precision by Cell Type", fontweight="bold")
                            sns.heatmap(_rec_m, annot=_f1_show_annot, fmt=".2f", cmap="Greens", ax=ax2, vmin=0, vmax=1, square=_f1_lock, annot_kws=_f1_annot_kws)
                            ax2.set_title("Recall by Cell Type", fontweight="bold")
                            sns.heatmap(_f1m, annot=_f1_show_annot, fmt=".2f", cmap="Blues", ax=ax3, vmin=0, vmax=1, square=_f1_lock, annot_kws=_f1_annot_kws)
                            ax3.set_title("F1-Score by Cell Type", fontweight="bold")
                            for _ax in (ax1, ax2, ax3):
                                _ax.set_xlabel("Method", fontsize=10, fontweight="bold")
                                _ax.set_ylabel("Cell Type" if _ax == ax1 else "", fontsize=10, fontweight="bold")
                                _ax.set_xticklabels(_ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
                                _ax.set_yticklabels(_ax.get_yticklabels(), fontsize=9)
                            fig.tight_layout(rect=(0, 0, 1, 0.96))
                            fig_path = Path(output_dir) / "comprehensive_per_class_analysis.png"
                            fig.savefig(fig_path, dpi=_f1_dpi, bbox_inches="tight")
                            st.pyplot(fig)
                            plt.close(fig)
                            _log(f"Saved per-class heatmaps: {fig_path}")
                        else:
                            _log("Per-class heatmaps skipped: no cell types after filtering")

                    if "Average per-class metrics (error bars)" in selected_figures:
                        if cell_types:
                            fig, fig_path = _plot_average_per_class_metrics(
                                individual_results, cell_types, Path(output_dir), _f1_dpi,
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                            _log(f"Saved average per-class metrics: {fig_path}")
                        else:
                            _log("Average per-class metrics skipped: no cell types after filtering")

                status.update(label="F1 analysis complete", state="complete")
                _log("F1 analysis complete")

            if log_lines:
                st.markdown("#### F1 Log")
                st.code("\n".join(log_lines))

    with sub_tabs[3]:
        st.markdown("### Heatmap")
        st.caption("Publication-ready phenotype × marker heatmaps.")

        phenotype_candidates = [
            c for c in available_columns
            if c.lower().startswith("phenotype_") or c.lower().endswith("_phenotype")
        ]
        non_phenotype_candidates = [c for c in available_columns if c not in phenotype_candidates]
        heatmap_group_candidates = phenotype_candidates + non_phenotype_candidates
        notebook_default_pheno = _extract_scalar_from_notebook("PHENOTYPE_COLUMN")
        default_pheno = (
            notebook_default_pheno
            if notebook_default_pheno in heatmap_group_candidates
            else (
                phenotype_candidates[0]
                if phenotype_candidates
                else (heatmap_group_candidates[0] if heatmap_group_candidates else "")
            )
        )

        notebook_markers = _extract_list_from_notebook(["MARKERS"])
        metadata_cols = [
            "cell_id", "batch_id", "tissue", "x_centroid", "y_centroid", "area", "perimeter",
            "eccentricity", "solidity", "extent", "major_axis", "minor_axis", "orientation", "label", "index",
        ]
        metadata_lower = [m.lower() for m in metadata_cols]
        marker_options: list[str] = [
            c for c in available_columns
            if c.lower() not in metadata_lower
            and not c.startswith("phenotype_")
            and not c.startswith("umap_")
            and not c.startswith("leiden_")
            and not c.endswith("_norm")
        ]

        marker_candidates = [m for m in notebook_markers if m in marker_options]
        if not marker_candidates:
            marker_candidates = marker_options

        shared_marker_panels, _ = get_named_marker_panels(marker_options, unified_csv_path)
        shared_panel_key = "tab7_viz_heatmap_shared_panel_name"
        if shared_marker_panels:
            shared_panel_names = list(shared_marker_panels)
            if st.session_state.get(shared_panel_key) not in shared_marker_panels:
                st.session_state[shared_panel_key] = shared_panel_names[0]

            def _load_shared_heatmap_panel() -> None:
                panel_name = st.session_state.get(shared_panel_key, shared_panel_names[0])
                panel_values = list(shared_marker_panels.get(panel_name, []))
                st.session_state["viz_heatmap_marker_multiselect"] = panel_values
                st.session_state["viz_heatmap_confirmed_markers"] = panel_values
                st.session_state["viz_heatmap_markers"] = panel_values

            selected_shared_panel = st.selectbox(
                "Shared marker panel",
                shared_panel_names,
                key=shared_panel_key,
                on_change=_load_shared_heatmap_panel,
                help="Select a named panel created in the shared marker workspace.",
            )
            marker_options = list(shared_marker_panels[selected_shared_panel])
            marker_candidates = list(marker_options)

        if "viz_heatmap_confirmed_markers" not in st.session_state or st.session_state.get("viz_heatmap_last_file") != unified_csv_path:
            st.session_state["viz_heatmap_confirmed_markers"] = marker_candidates
            st.session_state["viz_heatmap_markers"] = marker_candidates
            st.session_state["viz_heatmap_last_file"] = unified_csv_path

        pending_heatmap_selection = st.session_state.pop("viz_heatmap_marker_pending_selection", None)
        if isinstance(pending_heatmap_selection, list):
            st.session_state["viz_heatmap_confirmed_markers"] = pending_heatmap_selection
            st.session_state["viz_heatmap_markers"] = pending_heatmap_selection
            st.session_state["viz_heatmap_marker_multiselect"] = pending_heatmap_selection

        if "viz_heatmap_pheno_cols" not in st.session_state or st.session_state.get("viz_heatmap_last_file_pheno") != unified_csv_path:
            st.session_state["viz_heatmap_pheno_cols"] = [default_pheno] if default_pheno else []
            st.session_state["viz_heatmap_last_file_pheno"] = unified_csv_path

        cached_heatmap_filters = cached_params.get("viz_heatmap_filters", [])
        if st.session_state.get("viz_heatmap_filter_file") != unified_csv_path:
            previous_filter_count = int(st.session_state.get("viz_heatmap_n_filters", 0) or 0)
            initial_filter_count = max(1, len(cached_heatmap_filters))
            st.session_state["viz_heatmap_n_filters"] = initial_filter_count
            for idx in range(initial_filter_count):
                spec = cached_heatmap_filters[idx] if idx < len(cached_heatmap_filters) else {}
                st.session_state[f"viz_heatmap_filter_col_{idx}"] = str(spec.get("column", "") or "")
                st.session_state[f"viz_heatmap_filter_vals_{idx}"] = [
                    str(value) for value in spec.get("values", []) if str(value).strip()
                ]
            for idx in range(initial_filter_count, previous_filter_count):
                st.session_state.pop(f"viz_heatmap_filter_col_{idx}", None)
                st.session_state.pop(f"viz_heatmap_filter_vals_{idx}", None)
            st.session_state["viz_heatmap_filter_file"] = unified_csv_path

        heatmap_filter_df: pd.DataFrame | None = None
        heatmap_filter_columns: list[str] = []
        if unified_csv_path and os.path.exists(unified_csv_path):
            try:
                heatmap_filter_df = _get_cached_unified_df(unified_csv_path)
                heatmap_filter_columns = _get_low_cardinality_filter_columns(
                    heatmap_filter_df,
                    available_columns,
                    excluded=marker_options,
                )
            except Exception as exc:
                st.warning(f"Unable to load filter values for heatmap filtering: {exc}")

        st.markdown("#### Data Filters")
        st.caption("Optionally restrict the rows used for heatmap generation. Each filter keeps only rows matching the selected values.")
        if heatmap_filter_df is None:
            st.info("Load a valid unified CSV to enable heatmap filtering.")
        elif not heatmap_filter_columns:
            st.info("No categorical or low-cardinality columns were found for heatmap filtering.")
        else:
            n_heatmap_filters = max(1, int(st.session_state.get("viz_heatmap_n_filters", 1) or 1))
            preview_filter_specs: list[tuple[str, list[str]]] = []

            for filter_idx in range(n_heatmap_filters):
                column_key = f"viz_heatmap_filter_col_{filter_idx}"
                values_key = f"viz_heatmap_filter_vals_{filter_idx}"
                current_column = str(st.session_state.get(column_key, "") or "")
                if current_column and current_column not in heatmap_filter_columns:
                    st.session_state[column_key] = ""
                    current_column = ""

                st.markdown(f"**Filter {filter_idx + 1}**")
                filter_col_a, filter_col_b = st.columns([2, 3])
                with filter_col_a:
                    selected_filter_column = st.selectbox(
                        f"Column {filter_idx + 1}",
                        options=[""] + heatmap_filter_columns,
                        key=column_key,
                        format_func=lambda value: "Select a column" if value == "" else value,
                    )

                value_options = _get_filter_value_options(heatmap_filter_df, selected_filter_column) if selected_filter_column else []
                current_values = st.session_state.get(values_key, [])
                cleaned_values = [value for value in current_values if value in value_options]
                if cleaned_values != current_values:
                    st.session_state[values_key] = cleaned_values

                with filter_col_b:
                    selected_filter_values = st.multiselect(
                        f"Values {filter_idx + 1}",
                        options=value_options,
                        key=values_key,
                        help="Select one or more values to keep for this column. Leave empty to disable this filter.",
                    )

                if selected_filter_column and selected_filter_values:
                    preview_filter_specs.append((selected_filter_column, selected_filter_values))

            filter_btn_1, filter_btn_2, filter_btn_3 = st.columns([1, 1, 1])
            with filter_btn_1:
                if st.button("+ Add Filter", key="viz_heatmap_add_filter"):
                    st.session_state["viz_heatmap_n_filters"] = n_heatmap_filters + 1
                    st.rerun()
            with filter_btn_2:
                if n_heatmap_filters > 1 and st.button("- Remove Filter", key="viz_heatmap_remove_filter"):
                    last_idx = n_heatmap_filters - 1
                    st.session_state.pop(f"viz_heatmap_filter_col_{last_idx}", None)
                    st.session_state.pop(f"viz_heatmap_filter_vals_{last_idx}", None)
                    st.session_state["viz_heatmap_n_filters"] = max(1, n_heatmap_filters - 1)
                    st.rerun()
            with filter_btn_3:
                if st.button("Clear Filters", key="viz_heatmap_clear_filters"):
                    for idx in range(n_heatmap_filters):
                        st.session_state[f"viz_heatmap_filter_col_{idx}"] = ""
                        st.session_state[f"viz_heatmap_filter_vals_{idx}"] = []
                    st.rerun()

            if preview_filter_specs:
                preview_df, preview_summary = _apply_value_filters(heatmap_filter_df, preview_filter_specs)
                st.success(f"{len(preview_df):,} / {len(heatmap_filter_df):,} rows currently match the active heatmap filters.")
                for summary_line in preview_summary:
                    st.caption(summary_line)
            else:
                st.info(f"No heatmap filters applied. Using all {len(heatmap_filter_df):,} rows.")

        st.markdown("#### Marker Configuration")
        with st.form("viz_heatmap_marker_form", clear_on_submit=False):
            st.markdown("**Available Markers** (auto-selected from notebook)")

            hm_col1, hm_col2 = st.columns(2)
            with hm_col1:
                hm_select_all_markers = st.form_submit_button("Select all markers")
            with hm_col2:
                hm_clear_all = st.form_submit_button("Clear all")

            current_selection_hm = [
                m for m in st.session_state.get("viz_heatmap_confirmed_markers", [])
                if m in marker_options
            ]
            viz_hm_multi = st.session_state.get("viz_heatmap_marker_multiselect", [])
            if "viz_heatmap_marker_multiselect" not in st.session_state or (not viz_hm_multi and current_selection_hm):
                st.session_state["viz_heatmap_marker_multiselect"] = current_selection_hm
            else:
                st.session_state["viz_heatmap_marker_multiselect"] = [
                    m for m in st.session_state.get("viz_heatmap_marker_multiselect", [])
                    if m in marker_options
                ]

            selected_markers_hm = st.multiselect(
                "Select Markers",
                marker_options,
                help="Choose or trim markers from the selected named panel.",
                key="viz_heatmap_marker_multiselect",
            )

            hm_save_markers = st.form_submit_button("Save Marker Selection", type="primary")
            if hm_save_markers:
                st.session_state["viz_heatmap_confirmed_markers"] = selected_markers_hm
                st.session_state["viz_heatmap_markers"] = selected_markers_hm
                from spacemap_gui.persistence import save_active_project_state
                save_active_project_state("tab7_phenotyping")
                st.success(f"Saved {len(selected_markers_hm)} markers and project saved.")

            if hm_select_all_markers:
                st.session_state["viz_heatmap_marker_pending_selection"] = list(marker_options)
                st.session_state["viz_heatmap_confirmed_markers"] = list(marker_options)
                st.session_state["viz_heatmap_markers"] = list(marker_options)
                st.rerun()

            if hm_clear_all:
                st.session_state["viz_heatmap_marker_pending_selection"] = []
                st.session_state["viz_heatmap_confirmed_markers"] = []
                st.session_state["viz_heatmap_markers"] = []
                st.rerun()

        st.caption(f"Confirmed markers: {len(st.session_state.get('viz_heatmap_confirmed_markers', []))}")

        with st.form("viz_heatmap_form", clear_on_submit=False):
            _hm_pheno_opts = heatmap_group_candidates or available_columns
            pheno_cols = st.multiselect(
                "Phenotype Columns",
                options=_hm_pheno_opts,
                default=[m for m in st.session_state.get("viz_heatmap_pheno_cols", []) if m in _hm_pheno_opts],
                key="viz_heatmap_pheno_cols_select",
            )
            st.caption("Columns prefixed with 'phenotype_' or suffixed with '_phenotype' are shown first.")

            col1, col2, col3 = st.columns(3)
            with col1:
                dpi_val = st.number_input(
                    "DPI",
                    min_value=100,
                    max_value=600,
                    value=int(cached_params.get("viz_heatmap_dpi", 300)),
                    step=50,
                    key="viz_heatmap_dpi",
                )
            with col2:
                fig_width = st.number_input(
                    "Figure Width",
                    min_value=6.0,
                    max_value=30.0,
                    value=float(cached_params.get("viz_heatmap_width", 12.0)),
                    step=1.0,
                    key="viz_heatmap_width",
                )
            with col3:
                fig_height = st.number_input(
                    "Figure Height",
                    min_value=6.0,
                    max_value=30.0,
                    value=float(cached_params.get("viz_heatmap_height", 10.0)),
                    step=1.0,
                    key="viz_heatmap_height",
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                cmap = st.selectbox(
                    "Color Map",
                    ["bwr", "coolwarm", "vlag", "icefire", "RdBu_r"],
                    index=["bwr", "coolwarm", "vlag", "icefire", "RdBu_r"].index(cached_params.get("viz_heatmap_cmap", "bwr")),
                    key="viz_heatmap_cmap",
                )
            with col5:
                zmin = st.number_input(
                    "Z-Score Min",
                    min_value=-10.0,
                    max_value=0.0,
                    value=float(cached_params.get("viz_heatmap_zmin", -4.0)),
                    step=0.5,
                    key="viz_heatmap_zmin",
                )
            with col6:
                zmax = st.number_input(
                    "Z-Score Max",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(cached_params.get("viz_heatmap_zmax", 4.0)),
                    step=0.5,
                    key="viz_heatmap_zmax",
                )

            col7, col8, col9 = st.columns(3)
            with col7:
                cluster_x = st.checkbox(
                    "Cluster X Axis",
                    value=bool(cached_params.get("viz_heatmap_cluster_x", False)),
                    key="viz_heatmap_cluster_x",
                )
            with col8:
                cluster_y = st.checkbox(
                    "Cluster Y Axis",
                    value=bool(cached_params.get("viz_heatmap_cluster_y", False)),
                    key="viz_heatmap_cluster_y",
                )
            with col9:
                flip_axes = st.checkbox(
                    "Flip Axes",
                    value=bool(cached_params.get("viz_heatmap_flip", False)),
                    key="viz_heatmap_flip",
                )

            col10, col11 = st.columns(2)
            with col10:
                square_cells = st.checkbox(
                    "Square Cells",
                    value=bool(cached_params.get("viz_heatmap_square", False)),
                    key="viz_heatmap_square",
                )
            with col11:
                x_tick_rotation = st.number_input(
                    "X Tick Rotation",
                    min_value=0,
                    max_value=90,
                    value=int(cached_params.get("viz_heatmap_xtick_rotation", 45)),
                    step=5,
                    key="viz_heatmap_xtick_rotation",
                )

            col12, col13 = st.columns(2)
            with col12:
                marker_order_mode = st.selectbox(
                    "Marker Order",
                    ["Notebook", "Alphabetical", "Custom"],
                    index=["Notebook", "Alphabetical", "Custom"].index(cached_params.get("viz_heatmap_marker_order", "Notebook")),
                    key="viz_heatmap_marker_order",
                )
            with col13:
                phenotype_order_mode = st.selectbox(
                    "Phenotype Order",
                    ["Alphabetical", "Custom"],
                    index=["Alphabetical", "Custom"].index(cached_params.get("viz_heatmap_pheno_order", "Alphabetical")),
                    key="viz_heatmap_pheno_order",
                )

            default_group_cols = []
            if "batch_id" in available_columns:
                default_group_cols = ["batch_id"]
            _cached_group = cached_params.get("viz_heatmap_groupby_cols", default_group_cols)
            # Filter cached defaults to only columns present in the current CSV
            _safe_group_default = [c for c in _cached_group if c in available_columns]
            groupby_cols = st.multiselect(
                "Z-Score Grouping Columns",
                options=available_columns,
                default=_safe_group_default,
                key="viz_heatmap_groupby_cols",
            )

            marker_label_suffixes = st.text_input(
                "Marker Label Suffixes to Hide (comma-separated)",
                value=str(cached_params.get("viz_heatmap_marker_suffixes", "_corr")),
                key="viz_heatmap_marker_suffixes",
            )

            top_n_markers = st.number_input(
                "Top Markers in Summary Table",
                min_value=5,
                max_value=10,
                value=int(cached_params.get("viz_heatmap_summary_top_n", 5)),
                step=1,
                key="viz_heatmap_summary_top_n",
                help="Number of highest z-scored markers to include per phenotype in the summary table.",
            )

            marker_custom = st.text_area(
                "Custom Marker Order (comma-separated)",
                value=str(cached_params.get("viz_heatmap_marker_custom", "")),
                key="viz_heatmap_marker_custom",
            )
            phenotype_custom = st.text_area(
                "Custom Phenotype Order (comma-separated)",
                value=str(cached_params.get("viz_heatmap_pheno_custom", "")),
                key="viz_heatmap_pheno_custom",
            )

            corrected_only = st.checkbox(
                "Use corrected==1 if available",
                value=bool(cached_params.get("viz_heatmap_corrected_only", True)),
                key="viz_heatmap_corrected_only",
            )

            output_dir = st.text_input(
                "Output Directory",
                value=output_dir_default,
                key="viz_heatmap_output_dir",
            )

            save_cfg = st.form_submit_button("Save Configuration", type="primary")
            generate_now = st.form_submit_button("Generate Heatmap")
            if save_cfg:
                saved_heatmap_filters = [
                    {"column": column, "values": values}
                    for column, values in _read_heatmap_filter_specs(
                        int(st.session_state.get("viz_heatmap_n_filters", 1) or 1)
                    )
                ]
                st.session_state["viz_heatmap_markers"] = st.session_state.get("viz_heatmap_confirmed_markers", [])
                st.session_state["viz_heatmap_pheno_cols"] = pheno_cols
                cached_params.update(
                    {
                        "viz_heatmap_dpi": dpi_val,
                        "viz_heatmap_width": fig_width,
                        "viz_heatmap_height": fig_height,
                        "viz_heatmap_cmap": cmap,
                        "viz_heatmap_zmin": zmin,
                        "viz_heatmap_zmax": zmax,
                        "viz_heatmap_cluster_x": cluster_x,
                        "viz_heatmap_cluster_y": cluster_y,
                        "viz_heatmap_flip": flip_axes,
                        "viz_heatmap_square": square_cells,
                        "viz_heatmap_xtick_rotation": int(x_tick_rotation),
                        "viz_heatmap_marker_order": marker_order_mode,
                        "viz_heatmap_pheno_order": phenotype_order_mode,
                        "viz_heatmap_marker_custom": marker_custom,
                        "viz_heatmap_pheno_custom": phenotype_custom,
                        "viz_heatmap_corrected_only": corrected_only,
                        "viz_heatmap_groupby_cols": groupby_cols,
                        "viz_heatmap_marker_suffixes": marker_label_suffixes,
                        "viz_heatmap_summary_top_n": int(top_n_markers),
                        "viz_heatmap_filters": saved_heatmap_filters,
                        "viz_heatmap_n_filters": int(st.session_state.get("viz_heatmap_n_filters", 1) or 1),
                        "viz_output_dir": output_dir,
                    }
                )
                try:
                    with open(viz_cache, "w") as f:
                        json.dump(cached_params, f, indent=2)
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                except Exception as ex:
                    logger.warning(f"Failed to auto-save visualization cache or project state: {ex}")
                st.success("Configuration saved.")

        if generate_now:
            pheno_cols = st.session_state.get("viz_heatmap_pheno_cols_select", [])
            st.session_state["viz_heatmap_pheno_cols"] = pheno_cols
            st.session_state["viz_heatmap_markers"] = st.session_state.get("viz_heatmap_confirmed_markers", [])

            dpi_val = int(st.session_state.get("viz_heatmap_dpi", cached_params.get("viz_heatmap_dpi", 300)))
            fig_width = float(st.session_state.get("viz_heatmap_width", cached_params.get("viz_heatmap_width", 12.0)))
            fig_height = float(st.session_state.get("viz_heatmap_height", cached_params.get("viz_heatmap_height", 10.0)))
            cmap = str(st.session_state.get("viz_heatmap_cmap", cached_params.get("viz_heatmap_cmap", "bwr")))
            zmin = float(st.session_state.get("viz_heatmap_zmin", cached_params.get("viz_heatmap_zmin", -4.0)))
            zmax = float(st.session_state.get("viz_heatmap_zmax", cached_params.get("viz_heatmap_zmax", 4.0)))
            cluster_x = bool(st.session_state.get("viz_heatmap_cluster_x", cached_params.get("viz_heatmap_cluster_x", False)))
            cluster_y = bool(st.session_state.get("viz_heatmap_cluster_y", cached_params.get("viz_heatmap_cluster_y", False)))
            flip_axes = bool(st.session_state.get("viz_heatmap_flip", cached_params.get("viz_heatmap_flip", False)))
            square_cells = bool(st.session_state.get("viz_heatmap_square", cached_params.get("viz_heatmap_square", False)))
            x_tick_rotation = int(st.session_state.get("viz_heatmap_xtick_rotation", cached_params.get("viz_heatmap_xtick_rotation", 45)))
            marker_order_mode = str(st.session_state.get("viz_heatmap_marker_order", cached_params.get("viz_heatmap_marker_order", "Notebook")))
            phenotype_order_mode = str(st.session_state.get("viz_heatmap_pheno_order", cached_params.get("viz_heatmap_pheno_order", "Alphabetical")))
            marker_custom = str(st.session_state.get("viz_heatmap_marker_custom", cached_params.get("viz_heatmap_marker_custom", "")))
            phenotype_custom = str(st.session_state.get("viz_heatmap_pheno_custom", cached_params.get("viz_heatmap_pheno_custom", "")))
            corrected_only = bool(st.session_state.get("viz_heatmap_corrected_only", cached_params.get("viz_heatmap_corrected_only", True)))
            groupby_cols = st.session_state.get("viz_heatmap_groupby_cols", cached_params.get("viz_heatmap_groupby_cols", []))
            marker_label_suffixes = str(
                st.session_state.get("viz_heatmap_marker_suffixes", cached_params.get("viz_heatmap_marker_suffixes", "_corr"))
            )
            top_n_markers = int(
                st.session_state.get("viz_heatmap_summary_top_n", cached_params.get("viz_heatmap_summary_top_n", 5))
            )
            output_dir = str(st.session_state.get("viz_heatmap_output_dir", cached_params.get("viz_output_dir", output_dir_default)))
            active_heatmap_filters = _read_heatmap_filter_specs(
                int(st.session_state.get("viz_heatmap_n_filters", cached_params.get("viz_heatmap_n_filters", 1)) or 1)
            )
            if not unified_csv_path or not os.path.exists(unified_csv_path):
                st.error("Unified CSV file not found.")
            elif not st.session_state.get("viz_heatmap_pheno_cols"):
                st.error("Select at least one phenotype column.")
            elif not st.session_state.get("viz_heatmap_markers"):
                st.error("Select at least one marker.")
            else:
                df = _get_cached_unified_df(unified_csv_path).copy()
                if corrected_only and "corrected" in df.columns:
                    df = df[df["corrected"] == 1].copy()

                if active_heatmap_filters:
                    df, filter_summary = _apply_value_filters(df, active_heatmap_filters)
                    if filter_summary:
                        st.caption("Applied heatmap filters:")
                        for summary_line in filter_summary:
                            st.caption(summary_line)
                    if df.empty:
                        st.error("No rows remain after applying the heatmap filters. Adjust the filter criteria.")
                        return

                markers_selected = [m for m in st.session_state.get("viz_heatmap_markers", []) if m in df.columns]
                if not markers_selected:
                    st.error("Selected markers are not available in the unified CSV.")
                else:
                    output_dir = output_dir
                    os.makedirs(output_dir, exist_ok=True)

                    marker_custom_list = [m.strip() for m in marker_custom.split(",") if m.strip()]
                    pheno_custom_list = [p.strip() for p in phenotype_custom.split(",") if p.strip()]

                    suffix_list = [s.strip() for s in marker_label_suffixes.split(",") if s.strip()]
                    groupby_cols = [c for c in (groupby_cols or []) if c in df.columns]
                    for phenotype_col in st.session_state.get("viz_heatmap_pheno_cols", []):
                        if phenotype_col not in df.columns:
                            st.warning(f"Column '{phenotype_col}' not found. Skipping.")
                            continue

                        data_cols = [phenotype_col] + markers_selected
                        if groupby_cols:
                            data_cols = list(dict.fromkeys(data_cols + groupby_cols))
                        data_subset = df[data_cols].copy()
                        data_subset = data_subset.dropna(subset=[phenotype_col])

                        if groupby_cols:
                            means = data_subset.groupby(groupby_cols)[markers_selected].transform("mean")
                            stds = data_subset.groupby(groupby_cols)[markers_selected].transform("std").replace(0, np.nan)
                            z_df = (data_subset[markers_selected] - means) / stds
                        else:
                            means = data_subset[markers_selected].mean(axis=0)
                            stds = data_subset[markers_selected].std(axis=0).replace(0, np.nan)
                            z_df = (data_subset[markers_selected] - means) / stds
                        z_df = z_df.fillna(0)
                        z_df[phenotype_col] = data_subset[phenotype_col].values

                        mean_z = z_df.groupby(phenotype_col)[markers_selected].mean()
                        mean_z.index.name = "phenotype"

                        phenotypes = list(mean_z.index)
                        markers = list(mean_z.columns)

                        if marker_order_mode == "Notebook":
                            markers = [m for m in notebook_markers if m in markers] + [m for m in markers if m not in notebook_markers]
                        elif marker_order_mode == "Alphabetical":
                            markers = sorted(markers)
                        elif marker_order_mode == "Custom" and marker_custom_list:
                            markers = _apply_custom_order(markers, marker_custom_list)

                        if phenotype_order_mode == "Alphabetical":
                            phenotypes = sorted(phenotypes)
                        elif phenotype_order_mode == "Custom" and pheno_custom_list:
                            phenotypes = _apply_custom_order(phenotypes, pheno_custom_list)

                        mean_z = mean_z.loc[phenotypes, markers]
                        heat_data = mean_z.T

                        if cluster_x:
                            x_order = _cluster_order(heat_data, axis=1)
                            heat_data = heat_data.loc[:, x_order]
                        if cluster_y:
                            y_order = _cluster_order(heat_data, axis=0)
                            heat_data = heat_data.loc[y_order, :]

                        if flip_axes:
                            heat_data = heat_data.T

                        # Prepare tick labels to ensure one tick per label and stripped suffixes
                        if flip_axes:
                            x_labels = [_strip_suffixes(m, suffix_list) for m in heat_data.columns]
                            y_labels = heat_data.index
                        else:
                            x_labels = heat_data.columns
                            y_labels = [_strip_suffixes(m, suffix_list) for m in heat_data.index]

                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        sns.heatmap(
                            heat_data,
                            cmap=cmap,
                            center=0,
                            vmin=zmin,
                            vmax=zmax,
                            linewidths=0.5,
                            linecolor="lightgray",
                            cbar_kws={"label": "Mean Z-score"},
                            square=square_cells,
                            ax=ax,
                            xticklabels=x_labels,
                            yticklabels=y_labels,
                        )
                        title = f"Heatmap of {phenotype_col}: Mean Marker Z-Scores"
                        ax.set_title(title, fontsize=14, pad=16)
                        ax.set_xlabel("Phenotype" if not flip_axes else "Marker")
                        ax.set_ylabel("Marker" if not flip_axes else "Phenotype")

                        plt.xticks(rotation=x_tick_rotation, ha="right")
                        plt.yticks(rotation=0)
                        plt.tight_layout()

                        filename = f"heatmap_{_sanitize_filename(phenotype_col)}.png"
                        output_path = Path(output_dir) / filename
                        fig.savefig(output_path, dpi=int(dpi_val), bbox_inches="tight")
                        st.success(f"Saved: {output_path}")
                        st.pyplot(fig)

                        summary_rows: list[dict[str, Any]] = []
                        top_n_effective = max(5, min(10, int(top_n_markers)))
                        for phenotype_name, phenotype_row in mean_z.iterrows():
                            marker_scores = phenotype_row.nlargest(min(top_n_effective, len(phenotype_row)))
                            row: dict[str, Any] = {"Phenotype": str(phenotype_name)}
                            for rank, (marker_name, marker_score) in enumerate(marker_scores.items(), start=1):
                                row[f"Top {rank} Marker"] = _strip_suffixes(str(marker_name), suffix_list)
                                row[f"Top {rank} Z-score"] = float(marker_score)
                            summary_rows.append(row)

                        summary_df = pd.DataFrame(summary_rows)
                        st.markdown(f"##### Summary Table — {phenotype_col}")
                        st.dataframe(summary_df, width="stretch", hide_index=True)

                        summary_path = Path(output_dir) / f"heatmap_summary_{_sanitize_filename(phenotype_col)}.csv"
                        summary_df.to_csv(summary_path, index=False)
                        st.caption(f"Summary saved: `{summary_path}`")

                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png", dpi=int(dpi_val), bbox_inches="tight")
                        buffer.seek(0)
                        st.download_button(
                            label=f"Download {phenotype_col} Heatmap",
                            data=buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"viz_heatmap_download_{phenotype_col}",
                        )
                        plt.close(fig)

    with sub_tabs[4]:
        st.markdown("### Visual Validation")
        st.caption("Inspect random cells from specific phenotypes to validate assignments.")

        # --- LOAD DATA (Shared) ---
        if not unified_csv_path or not os.path.exists(unified_csv_path):
            st.error("Unified CSV file not found. Please ensure global input path is set.")
        else:
            # 1. MARKER SELECTION (Exact Match to Leiden Logic)
            st.markdown("##### Marker Configuration")
            
            # Helper to get markers
            notebook_corr_markers, notebook_non_corr_markers = _extract_markers_from_notebook("Step6A_phenotype_leiden.ipynb")
            
            # Get available columns from CSV
            df_viz_val = _get_loaded_visualization_df(unified_csv_path)
            if isinstance(df_viz_val, pd.DataFrame):
                available_cols = list(df_viz_val.columns)
            else:
                available_cols = available_columns
                df_viz_val = pd.DataFrame()
                st.info("Load the unified CSV once using 'Unified CSV Loader' above to enable Visual Validation preprocessing.")

            # Filter markers (Excluding metadata)
            metadata_cols = ['cell_id', 'batch_id', 'tissue', 'x_centroid', 'y_centroid', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'major_axis', 'minor_axis', 'orientation', 'label', 'index']
            metadata_lower = [m.lower() for m in metadata_cols]
            
            all_marker_candidates = [
                c for c in available_cols
                if c.lower() not in metadata_lower
                and not c.startswith("phenotype_")
                and not c.startswith('umap_')
                and not c.startswith('leiden_')
                and not c.endswith('_norm') # Match Leiden logic
            ]

            shared_marker_panels, _ = get_named_marker_panels(all_marker_candidates, unified_csv_path)
            shared_panel_key = "tab7_viz_validation_shared_panel_name"
            if shared_marker_panels:
                shared_panel_names = list(shared_marker_panels)
                if st.session_state.get(shared_panel_key) not in shared_marker_panels:
                    st.session_state[shared_panel_key] = shared_panel_names[0]

                def _load_shared_validation_panel() -> None:
                    panel_name = st.session_state.get(shared_panel_key, shared_panel_names[0])
                    panel_values = list(shared_marker_panels.get(panel_name, []))
                    st.session_state['viz_val_marker_multiselect'] = panel_values
                    st.session_state['viz_val_confirmed_markers'] = panel_values

                selected_shared_panel = st.selectbox(
                    "Shared marker panel",
                    shared_panel_names,
                    key=shared_panel_key,
                    on_change=_load_shared_validation_panel,
                    help="Select a named panel created in the shared marker workspace.",
                )
                all_marker_candidates = list(shared_marker_panels[selected_shared_panel])
            
            # Auto-select markers from notebook if needed
            expected_markers = []
            for m in notebook_corr_markers:
                corr_col = f"{m}_corr"
                if corr_col in all_marker_candidates:
                    expected_markers.append(corr_col)
            for m in notebook_non_corr_markers:
                if m in all_marker_candidates:
                    expected_markers.append(m)

            # Initialize session state for confirmed markers (Leiden pattern)
            # Use a visualization-specific key to avoid conflict with Leiden tab
            if 'viz_val_confirmed_markers' not in st.session_state or st.session_state.get('viz_val_last_file') != unified_csv_path:
                 st.session_state['viz_val_confirmed_markers'] = expected_markers
                 st.session_state['viz_val_last_file'] = unified_csv_path
                 if expected_markers:
                     st.info(f"Auto-selected {len(expected_markers)} markers from notebook configuration")

            pending_marker_selection = st.session_state.pop('viz_val_marker_pending_selection', None)
            if isinstance(pending_marker_selection, list):
                st.session_state['viz_val_confirmed_markers'] = pending_marker_selection
                st.session_state['viz_val_marker_multiselect'] = pending_marker_selection
            
            # --- MARKER SELECTION FORM (Offline) ---
            with st.form("viz_val_marker_form", clear_on_submit=False):
                st.markdown("**Available Markers** (auto-selected from notebook)")
                
                # Show quick filter options
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    select_all_all = st.form_submit_button("Select all markers")
                with f_col2:
                    clear_all = st.form_submit_button("Clear all")
                
                # Current selection
                current_selection_vv = [m for m in st.session_state.get('viz_val_confirmed_markers', []) if m in all_marker_candidates]
                viz_val_multi = st.session_state.get('viz_val_marker_multiselect', [])
                if 'viz_val_marker_multiselect' not in st.session_state or (not viz_val_multi and current_selection_vv):
                    st.session_state['viz_val_marker_multiselect'] = current_selection_vv
                else:
                    st.session_state['viz_val_marker_multiselect'] = [
                        m for m in st.session_state.get('viz_val_marker_multiselect', [])
                        if m in all_marker_candidates
                    ]
                
                selected_markers_vv = st.multiselect(
                    "Select Markers",
                    all_marker_candidates,
                    help="Choose or trim markers from the selected named panel.",
                    key="viz_val_marker_multiselect"
                )
                
                save_markers_btn = st.form_submit_button("Save Marker Selection", type="primary")
                
                if save_markers_btn:
                    st.session_state['viz_val_confirmed_markers'] = selected_markers_vv
                    from spacemap_gui.persistence import save_active_project_state
                    save_active_project_state("tab7_phenotyping")
                    st.success(f"Saved {len(selected_markers_vv)} markers and project saved.")
                
                if select_all_all:
                    st.session_state['viz_val_marker_pending_selection'] = list(all_marker_candidates)
                    st.session_state['viz_val_confirmed_markers'] = list(all_marker_candidates)
                    st.rerun()
                
                if clear_all:
                    st.session_state['viz_val_marker_pending_selection'] = []
                    st.session_state['viz_val_confirmed_markers'] = []
                    st.rerun()

            # Use confirmed markers
            markers_final = st.session_state.get('viz_val_confirmed_markers', [])
            st.caption(f"Confirmed markers: {len(markers_final)}")


            # 2. DATA LOADING & NORMALIZATION
            st.divider()
            st.markdown("##### Data Loading & Normalization")
            
            c_norm1, c_norm2 = st.columns(2)
            with c_norm1:
                enable_znorm = st.checkbox(
                    "Enable Z-Score Normalization", 
                    value=True, 
                    help="Normalize marker expression per batch.",
                    key="viz_val_enable_znorm"
                )
            with c_norm2:
                batch_col_vv = st.selectbox(
                    "Batch Column",
                    [c for c in available_cols if 'batch' in c.lower()] or ['batch_id'],
                    key="viz_val_batch_col"
                )
            
            if st.button("Load & Normalize Data", type="primary", key="viz_val_load_btn"):
                if not markers_final:
                    st.error("Please save marker selection first.")
                else:
                    try:
                        with st.spinner("Loading and normalizing data..."):
                            # Load raw DF
                            df_raw = _get_cached_unified_df(unified_csv_path)
                            
                            # Identify necessary columns
                            meta_needed = ['cell_id', 'x_centroid', 'y_centroid', batch_col_vv]
                            # Start with selected markers
                            cols_to_keep = markers_final + meta_needed
                            
                            # We can't know the Phenotype column yet as it's selected in next step.
                            # So we must keep ALL phenotype columns or the full DF.
                            # Keeping full DF is safer but memory heavy. 
                            # Let's keep markers + metadata + all 'phenotype_' columns.
                            pheno_candidates = [c for c in df_raw.columns if c.startswith('phenotype_')]
                            cols_to_keep += pheno_candidates
                            
                            # Filter valid columns
                            cols_to_keep = list(dict.fromkeys([c for c in cols_to_keep if c in df_raw.columns]))
                            
                            # Create Working DF
                            viz_df = df_raw[cols_to_keep].copy()
                            
                            # FORCE NUMERIC CONVERSION for Markers
                            # This fixes the "dtype object" nlargest error
                            for m in markers_final:
                                if m in viz_df.columns:
                                    viz_df[m] = pd.to_numeric(viz_df[m], errors='coerce').fillna(0)
                            
                            # PRE-CALCULATE BATCH STATS (Universal Normalization)
                            batch_stats = {}
                            if enable_znorm:
                                if batch_col_vv not in viz_df.columns:
                                    st.warning(f"Batch column {batch_col_vv} not found. Skipping normalization.")
                                else:
                                    # Calculate means/stds per batch for selected markers
                                    # Groupby is efficient enough here
                                    # We store them in a dict for fast lookup
                                    grouped = viz_df.groupby(batch_col_vv)[markers_final]
                                    means = grouped.mean()
                                    stds = grouped.std().replace(0, 1)
                                    
                                    # Convert to dict of dicts for safe access
                                    # {batch_id: {marker: mean}}
                                    batch_stats['means'] = means.to_dict(orient='index')
                                    batch_stats['stds'] = stds.to_dict(orient='index')
                            
                            # Store in session state
                            st.session_state['viz_val_data_ready'] = True
                            st.session_state['viz_val_cached_df'] = viz_df
                            st.session_state['viz_val_batch_stats'] = batch_stats
                            st.session_state['viz_val_loaded_markers'] = markers_final # Track which markers were loaded
                            st.session_state['viz_val_loaded_batch_col'] = batch_col_vv
                            
                        st.success("Data loaded and normalized successfully!")
                        st.rerun() # Refresh to show next section
                        
                    except Exception as e:
                        st.error(f"Error loading data: {e}")
                        logger.error(f"VizVal Load Error: {e}")

            # 3. PHENOTYPE SELECTION (Gated)
            if st.session_state.get('viz_val_data_ready', False):
                st.divider()
                st.markdown("##### Visualization Target")
                
                viz_df = st.session_state['viz_val_cached_df']
                full_df = _get_cached_unified_df(unified_csv_path)
                loaded_markers = st.session_state.get('viz_val_loaded_markers', [])
                
                # Check if markers changed since load
                if loaded_markers != markers_final:
                    st.warning("Marker selection has changed. Please reload data above.")
                else:
                    phenotype_first_cols = [
                        c for c in full_df.columns
                        if c.lower().startswith("phenotype_") or c.lower().endswith("_phenotype")
                    ]
                    remaining_cols = [c for c in full_df.columns if c not in phenotype_first_cols]
                    pheno_cols = phenotype_first_cols + remaining_cols
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if pheno_cols:
                            selected_pheno_col = st.selectbox(
                                "Phenotype Column",
                                pheno_cols,
                                index=0,
                                key="viz_val_pheno_col"
                            )
                            st.caption("Columns prefixed with 'phenotype_' or suffixed with '_phenotype' are shown first.")
                        else:
                            selected_pheno_col = None
                            st.warning("No columns available for phenotype selection.")
                    with c2:
                         if selected_pheno_col in full_df.columns:
                             unique_phenos = sorted(full_df[selected_pheno_col].dropna().astype(str).unique().tolist())
                             selected_phenotype = st.selectbox(
                                 "Select Phenotype Class",
                                 unique_phenos,
                                 key="viz_val_target_pheno"
                             )
                         else:
                             selected_phenotype = None
                    
                    n_view = st.number_input("Number of Random Cells", 1, 50, 5, key="viz_val_n_cells")
                    batch_col_loaded = st.session_state.get('viz_val_loaded_batch_col', batch_col_vv)
                    available_viz_batches = []
                    if batch_col_loaded in viz_df.columns:
                        available_viz_batches = sorted(viz_df[batch_col_loaded].dropna().astype(str).unique().tolist())
                    viz_batch_scope = st.radio(
                        "Batch Scope",
                        ["All matched batches", "Specific batch"],
                        horizontal=True,
                        key="viz_val_batch_scope",
                    )
                    selected_viz_batch = None
                    if viz_batch_scope == "Specific batch" and available_viz_batches:
                        selected_viz_batch = st.selectbox(
                            "Batch",
                            available_viz_batches,
                            key="viz_val_batch_sel",
                        )
                    
                    # EXPORT SETTINGS
                    st.markdown("##### Export Settings")
                    default_out = project_root / "outputs" / "visual_validation"
                    export_dir_str = st.text_input("Output Directory", str(default_out), key="viz_val_out_dir")
                    
                    if st.button("Visualize & Save", type="primary", key="viz_val_run"):
                         if selected_pheno_col not in full_df.columns:
                             st.error("Selected phenotype column is not available in the unified CSV.")
                             subset_df = pd.DataFrame()
                         else:
                             filter_mask = full_df[selected_pheno_col].astype(str) == str(selected_phenotype)
                             subset_df = viz_df.loc[filter_mask]
                         
                         if len(subset_df) == 0:
                             st.warning("No cells found for this phenotype.")
                         else:
                             # Initialize Export
                             timestamp = int(time.time())
                             export_dir = Path(export_dir_str)
                             try:
                                 export_dir.mkdir(parents=True, exist_ok=True)
                             except Exception as e:
                                 st.error(f"Could not create output directory: {e}")
                                 export_dir = None
                             
                             # Initialize Visualizer
                             viz_roi = st.session_state.get('viz_roi_dir')
                             viz_seg = st.session_state.get('viz_seg_dir')
                             
                             if not viz_roi or not viz_seg or not os.path.exists(viz_roi):
                                 st.error("ROI/Seg directories not configured.")
                             else:
                                 try:
                                     viz = CellVisualizer(viz_roi, viz_seg)
                                     batch_col = st.session_state.get('viz_val_loaded_batch_col', 'batch_id')
                                     if not batch_col or batch_col not in subset_df.columns:
                                         batch_col = next((c for c in ['batch_id', 'BatchID', 'Point', 'sample_id'] if c in subset_df.columns), 'batch_id')
                                     
                                     if batch_col not in subset_df.columns:
                                         st.error(f"Batch column '{batch_col}' not found in results.")
                                         valid_subset = pd.DataFrame()
                                     else:
                                         # Match files
                                         unique_batches = subset_df[batch_col].unique().astype(str).tolist()
                                         matches = viz.match_file_triads(unique_batches)
                                         
                                         valid_subset = subset_df[subset_df[batch_col].astype(str).isin(matches.keys())]
                                         if viz_batch_scope == "Specific batch" and selected_viz_batch:
                                             valid_subset = valid_subset[valid_subset[batch_col].astype(str) == str(selected_viz_batch)]
                                     
                                     if len(valid_subset) == 0:
                                          st.warning("No matching images for these batches.")
                                     else:
                                          n_actual = min(n_view, len(valid_subset))
                                          random_sample = valid_subset.sample(n=n_actual, random_state=timestamp)
                                          random_sample = random_sample.sort_values(batch_col)
                                          sampled_batches = random_sample[batch_col].astype(str).drop_duplicates().tolist()
                                          prewarm_fn = getattr(viz, 'prewarm_batch', None)
                                          if callable(prewarm_fn):
                                              with st.spinner(f"Preloading {len(sampled_batches)} batch cache(s)..."):
                                                  for prewarm_batch_id in sampled_batches:
                                                      prewarm_fn(prewarm_batch_id)
                                          
                                          st.info(f"Visualizing {n_actual} cells... (Results saved to {export_dir.name})")
                                          
                                          batch_stats = st.session_state.get('viz_val_batch_stats', {})
                                          has_stats = 'means' in batch_stats
                                          
                                          cols = st.columns(min(n_actual, 3))
                                          idx_plot = 0
                                          summary_rows = []
                                          mic_scale = 0.5083
                                          
                                          for _, cell_row in random_sample.iterrows():
                                               try:
                                                   batch_val = cell_row[batch_col]
                                                   
                                                   # Get Raw Values & FORCE NUMERIC
                                                   row_vals_raw = cell_row[loaded_markers]
                                                   row_vals = pd.to_numeric(row_vals_raw, errors='coerce').fillna(0)
                                                   
                                                   if row_vals.dtype == 'object':
                                                        st.error(f"Failed to convert row values to numeric. Dtype remains object.")
                                                        continue

                                                   # Apply Normalization
                                                   if has_stats and batch_val in batch_stats['means']:
                                                       b_mean = pd.Series(batch_stats['means'][batch_val])
                                                       b_std = pd.Series(batch_stats['stds'][batch_val])
                                                       
                                                       b_mean = b_mean.reindex(row_vals.index).fillna(0)
                                                       b_std = b_std.reindex(row_vals.index).replace(0, 1).fillna(1)
                                                       
                                                       z_scores = (row_vals - b_mean) / b_std
                                                       z_scores = pd.to_numeric(z_scores, errors='coerce').fillna(0)
                                                       top_5 = z_scores.nlargest(5)
                                                   else:
                                                       top_5 = row_vals.nlargest(5)
                                                   
                                                   # Format & Collect Data
                                                   top_5_list = []
                                                   suffix_list = ["_corr", "_norm", "_log"]
                                                   top_5_clean = []
                                                   
                                                   # For CSV Summary
                                                   row_data = {
                                                       'cell_id': cell_row.get('cell_id'),
                                                       'batch_id': batch_val,
                                                       'x_centroid_px': cell_row.get('x_centroid'),
                                                       'y_centroid_px': cell_row.get('y_centroid'),
                                                       'x_centroid_mic_m': cell_row.get('x_centroid', 0) * mic_scale,
                                                       'y_centroid_mic_m': cell_row.get('y_centroid', 0) * mic_scale,
                                                       'phenotype_column': selected_pheno_col,
                                                       'phenotype_class': selected_phenotype
                                                   }
                                                   
                                                   score_idx = 1
                                                   for marker, score in top_5.items():
                                                        display_name = str(marker)
                                                        for s in suffix_list:
                                                             display_name = display_name.replace(s, "")
                                                        top_5_list.append(f"{display_name} ({float(score):.2f})")
                                                        top_5_clean.append(display_name)
                                                        
                                                        # Add to CSV row
                                                        row_data[f'Top_{score_idx}_Marker'] = display_name
                                                        row_data[f'Top_{score_idx}_Score'] = float(score)
                                                        score_idx += 1
                                                   
                                                   summary_rows.append(row_data)

                                                   # Image Generation
                                                   cid = cell_row['cell_id']
                                                   
                                                   cell_ctx = viz.get_cell_context(
                                                       batch_id=str(batch_val),
                                                       x=cell_row['x_centroid'],
                                                       y=cell_row['y_centroid'],
                                                       requested_channels=top_5_clean,
                                                   )
                                                   
                                                   available_channels = cell_ctx.get('channel_names', [])
                                                   indices_to_plot = []
                                                   names_to_plot = []
                                                   
                                                   if available_channels:
                                                       chan_map = {c.lower(): i for i, c in enumerate(available_channels)}
                                                       for m_name in top_5_clean:
                                                           idx = -1
                                                           if m_name in available_channels:
                                                               idx = available_channels.index(m_name)
                                                           elif m_name.lower() in chan_map:
                                                               idx = chan_map[m_name.lower()]
                                                           
                                                           if idx != -1:
                                                               indices_to_plot.append(idx)
                                                               names_to_plot.append(m_name)
                                                   
                                                   if indices_to_plot:
                                                       fig = viz.generate_visualization(
                                                            cell_data=cell_ctx,
                                                            marker_indices=indices_to_plot,
                                                            marker_names=names_to_plot,
                                                            display_size=40,
                                                            show_neighbors=True
                                                       )
                                                       
                                                       # Save Figure
                                                       if export_dir:
                                                           safe_pheno = "".join(x for x in selected_phenotype if x.isalnum() or x in "_-")
                                                           fname = f"vv_{selected_pheno_col}_{safe_pheno}_cell_{cid}_{timestamp}.png"
                                                           save_path = export_dir / fname
                                                           fig.savefig(save_path, bbox_inches='tight')
                                                       
                                                       with cols[idx_plot % 3]:
                                                            st.pyplot(fig)
                                                            st.caption(f"**Cell {cid}**\n{', '.join(top_5_list)}")
                                                            plt.close(fig)
                                                       idx_plot += 1
                                                   else:
                                                       st.warning(f"Cell {cid}: Markers not found in image channels.")
                                               
                                               except Exception as inner_e:
                                                   st.error(f"Error processing cell {cell_row.get('cell_id', 'unknown')}: {inner_e}")
                                                   _log_console(f"!!! VIZ ERROR: {inner_e}", level="error")
                                          
                                          # SUMMARY TABLE & CSV
                                          if summary_rows:
                                              st.markdown("##### Summary Table")
                                              summary_df = pd.DataFrame(summary_rows)
                                              st.dataframe(summary_df)
                                              
                                              if export_dir:
                                                  safe_pheno_col = "".join(x for x in selected_pheno_col if x.isalnum() or x in "_-")
                                                  safe_pheno = "".join(x for x in selected_phenotype if x.isalnum() or x in "_-")
                                                  csv_name = f"vv_summary_{safe_pheno_col}_{safe_pheno}_{timestamp}.csv"
                                                  csv_path = export_dir / csv_name
                                                  summary_df.to_csv(csv_path, index=False)
                                                  st.success(f"Saved summary to: `{csv_path}`")
                                                  st.success(f"Saved {idx_plot} images to: `{export_dir}`")

                                 except Exception as e:
                                     st.error(f"Visualization Error: {e}")
                                     logger.error(f"Outer Viz Error: {e}")

            else:
                st.info("Please load and normalize data to proceed.")

    # =========================================================================
    # UMAP EXPLORER SUBTAB
    # =========================================================================
    with sub_tabs[5]:
        _render_umap_explorer(project_root, unified_csv_path, available_columns, cached_params)

    # =========================================================================
    # CELL VISUALIZATION SUBTAB
    # =========================================================================
    with sub_tabs[6]:
        render_cell_visualization(
            project_root,
            unified_csv_path,
            available_columns,
            cached_params,
            loaded_df=_get_loaded_visualization_df(unified_csv_path),
        )

    try:
        for key, value in st.session_state.items():
            if not isinstance(key, str):
                continue
            if not (key.startswith("viz_") or key.startswith("cellviz_")):
                continue
            if isinstance(value, pd.DataFrame):
                continue
            cached_params[key] = value
        with open(viz_cache, "w") as f:
            json.dump(cached_params, f, indent=2)
    except Exception:
        pass
