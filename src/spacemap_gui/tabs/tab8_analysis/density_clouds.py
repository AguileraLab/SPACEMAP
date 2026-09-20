"""Reference-style tissue density clouds (SPACEMAP Figure 2E look).

Cloned from ``test_codes/SPACEMAP_Figure2D_E_triads_v1.py`` so the GUI
reproduces the published figure exactly: black background, translucent tissue
annotation layers read from the merged GeoJSON files, a white outline around the
selected tissue component and a Gaussian-smoothed density cloud drawn on a color
scale shared by every panel.

The API takes pre-computed point sets, so any panel can reuse it: the Density
subtab feeds phenotype centroids (CD8, CD4, DC, ...) and the Triad subtab can
feed triad centroids without changing a single style constant.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter
from shapely import contains_xy, make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

# Style constants copied verbatim from the reference script.
TISSUE_COLORS: dict[str, str] = {
    "tumor": "#D94B4B",
    "epithelia": "#55C667",
    "stroma": "#56B4E9",
    "endothelial": "#D8B365",
}
DEFAULT_LAYERS = ["tumor", "epithelia", "stroma"]
CLOUD_STOPS = ("#5E239D", "#F8961E", "#FFF75E")
CLOUD_BASE = "#00000000"
CLOUD_HIGHLIGHT = "#FFF75E"
EXTRA_LAYER_PALETTE = ["#D8B365", "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F"]
DEFAULT_TISSUE_LAYER = "tissue"
DEFAULT_FILE_PATTERN = "{batch}_merged.geojson"


@dataclass
class CloudStyle:
    """Every knob the Figure 2E renderer exposes; defaults reproduce the paper."""

    layer_colors: dict[str, str] = field(default_factory=dict)
    layer_alpha: float = 0.46
    background: str = "black"
    text_color: str = "white"
    show_boundary: bool = True
    boundary_color: str = "white"
    boundary_alpha: float = 0.85
    boundary_width: float = 0.7
    show_cloud: bool = True
    cloud_stops: Sequence[str] = CLOUD_STOPS
    grid_size: int = 360
    sigma_um: float = 55.0
    show_points: bool = True
    point_color: str = CLOUD_HIGHLIGHT
    point_size: float = 2.5
    point_alpha: float = 0.22
    panel_width: float = 3.5
    panel_height: float = 5.25
    title_fontsize: float = 11.5
    suptitle_fontsize: float = 13.0
    legend_fontsize: float = 8.5
    colorbar_fontsize: float = 9.0
    colorbar_label: str = "Smoothed density (shared scale)"
    show_colorbar: bool = True
    show_legend: bool = True
    cloud_label: str = "Density cloud"
    point_label: str = "Cell centroids"
    count_noun: str = "cells"
    dpi: int = 300


def flatten_polygons(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        result: list[Polygon] = []
        for part in geometry.geoms:
            result.extend(flatten_polygons(part))
        return result
    return []


def safe_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return GeometryCollection()
    return make_valid(geometry) if not geometry.is_valid else geometry


def _load_annotation_geometries(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    grouped = defaultdict(list)
    for feature in payload.get("features", []):
        properties = feature.get("properties", {}) or {}
        name = (properties.get("classification") or {}).get("name") or properties.get("name") or "unknown"
        try:
            geometry = safe_geometry(shape(feature["geometry"]))
        except Exception:
            continue
        if not geometry.is_empty:
            grouped[str(name).lower()].append(geometry)
    return {name: safe_geometry(unary_union(items)) for name, items in grouped.items()}


@st.cache_data(show_spinner=False, max_entries=8)
def load_annotation_geometries(path: str, mtime: float) -> dict[str, Any]:
    """Cached GeoJSON reader. ``mtime`` busts the cache when the file changes."""

    return _load_annotation_geometries(Path(path))


def annotation_path(annotation_dir: str, batch: str, pattern: str = DEFAULT_FILE_PATTERN) -> Path:
    name = str(pattern or DEFAULT_FILE_PATTERN).replace("{batch}", str(batch))
    return Path(annotation_dir) / name


def load_annotations_for_batch(annotation_dir: str, batch: str, pattern: str = DEFAULT_FILE_PATTERN) -> dict[str, Any]:
    path = annotation_path(annotation_dir, batch, pattern)
    if not path.exists():
        return {}
    return load_annotation_geometries(str(path), path.stat().st_mtime)


def default_layer_color(name: str, index: int = 0) -> str:
    key = str(name).lower()
    if key in TISSUE_COLORS:
        return TISSUE_COLORS[key]
    return EXTRA_LAYER_PALETTE[index % len(EXTRA_LAYER_PALETTE)]


def draw_geometry(ax, geometry, facecolor, edgecolor="none", alpha=1.0, linewidth=0.5, zorder=1):
    for polygon in flatten_polygons(geometry):
        xy = np.asarray(polygon.exterior.coords)
        ax.add_patch(
            MplPolygon(
                xy,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
        )
        for interior in polygon.interiors:
            hole = np.asarray(interior.coords)
            ax.add_patch(
                MplPolygon(hole, closed=True, facecolor=ax.get_facecolor(), edgecolor="none", zorder=zorder + 0.01)
            )


def choose_tissue_component(tissue_geometry, coords: np.ndarray):
    """Pick the tissue polygon holding most of ``coords`` (reference behaviour)."""

    polygons = flatten_polygons(tissue_geometry)
    if not polygons:
        return None, []
    points = np.asarray(coords, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None, []
    sample = points[:: max(1, len(points) // 2500)]
    scores = [int(np.sum(contains_xy(polygon, sample[:, 0], sample[:, 1]))) for polygon in polygons]
    if not scores or max(scores) == 0:
        return None, scores
    return polygons[int(np.argmax(scores))], scores


def density_field(points: np.ndarray, bounds, grid_size: int, sigma_px: float):
    x0, y0, x1, y1 = bounds
    if not len(points):
        return (
            np.zeros((grid_size, grid_size)),
            np.linspace(x0, x1, grid_size),
            np.linspace(y0, y1, grid_size),
        )
    histogram, x_edges, y_edges = np.histogram2d(
        points[:, 0], points[:, 1], bins=grid_size, range=[[x0, x1], [y0, y1]]
    )
    bin_px = max((x1 - x0) / grid_size, (y1 - y0) / grid_size)
    field_values = gaussian_filter(histogram, sigma=max(0.5, sigma_px / bin_px), mode="constant")
    return field_values.T, (x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2


def cloud_colormap(stops: Sequence[str]) -> LinearSegmentedColormap:
    colors = [CLOUD_BASE, *[str(stop) for stop in stops if str(stop)]]
    if len(colors) < 2:
        colors = [CLOUD_BASE, CLOUD_HIGHLIGHT]
    return LinearSegmentedColormap.from_list("spacemap_cloud", colors)


def compute_cloud_fields(
    panels: list[dict[str, Any]],
    component,
    style: CloudStyle,
    pixel_size_um: float,
) -> dict[str, Any]:
    """Smoothed density per panel, masked to the component, on one shared scale."""

    bounds = component.bounds
    fields: dict[str, np.ndarray] = {}
    points_by_panel: dict[str, np.ndarray] = {}
    grid_x = grid_y = None
    sigma_px = float(style.sigma_um) / float(pixel_size_um) if float(pixel_size_um) > 0 else float(style.sigma_um)

    for panel in panels:
        label = str(panel["label"])
        points = np.asarray(panel.get("points", np.empty((0, 2))), dtype=float)
        if points.size:
            points = points[np.isfinite(points).all(axis=1)]
            points = points[contains_xy(component, points[:, 0], points[:, 1])]
        else:
            points = np.empty((0, 2), dtype=float)
        points_by_panel[label] = points
        fields[label], grid_x, grid_y = density_field(points, bounds, int(style.grid_size), sigma_px)

    if grid_x is None or grid_y is None:
        return {"bounds": bounds, "fields": {}, "points": points_by_panel, "shared_max": 0.0}

    xx, yy = np.meshgrid(grid_x, grid_y)
    mask = contains_xy(component, xx, yy)
    for label in list(fields):
        fields[label] = np.where(mask, fields[label], np.nan)

    finite_maxima = [float(np.nanmax(values)) for values in fields.values() if np.isfinite(values).any()]
    shared_max = max(finite_maxima) if finite_maxima else 0.0
    return {
        "bounds": bounds,
        "fields": fields,
        "points": points_by_panel,
        "shared_max": float(shared_max) if shared_max > 0 else 1.0,
    }


def legend_handles(layer_names: Sequence[str], style: CloudStyle) -> list:
    handles = [
        Patch(
            facecolor=style.layer_colors.get(name, default_layer_color(name, index)),
            alpha=float(style.layer_alpha),
            label=str(name).capitalize(),
        )
        for index, name in enumerate(layer_names)
    ]
    if style.show_cloud:
        handles.append(Patch(facecolor=list(style.cloud_stops)[-1], label=style.cloud_label))
    if style.show_points:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=5,
                markerfacecolor=style.point_color,
                markeredgecolor="none",
                label=style.point_label,
            )
        )
    return handles


def _draw_panel(ax, label: str, shared: dict[str, Any], component, layers: dict[str, Any], style: CloudStyle):
    bounds = shared["bounds"]
    ax.set_facecolor(style.background)
    for index, (name, geometry) in enumerate(layers.items()):
        draw_geometry(
            ax,
            geometry,
            style.layer_colors.get(name, default_layer_color(name, index)),
            alpha=float(style.layer_alpha),
            zorder=1,
        )
    if style.show_boundary:
        draw_geometry(
            ax,
            component,
            "none",
            edgecolor=style.boundary_color,
            alpha=float(style.boundary_alpha),
            linewidth=float(style.boundary_width),
            zorder=2,
        )

    image = None
    if style.show_cloud and label in shared["fields"]:
        image = ax.imshow(
            shared["fields"][label],
            extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
            origin="lower",
            cmap=cloud_colormap(style.cloud_stops),
            norm=Normalize(0, shared["shared_max"]),
            interpolation="bilinear",
            zorder=3,
        )

    points = shared["points"].get(label, np.empty((0, 2)))
    if style.show_points and len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=float(style.point_size),
            c=style.point_color,
            alpha=float(style.point_alpha),
            linewidths=0,
            zorder=4,
        )

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[3], bounds[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{label}\n{len(points):,} {style.count_noun}",
        color=style.text_color,
        fontweight="bold",
        fontsize=float(style.title_fontsize),
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def render_cloud_figure(
    panels: list[dict[str, Any]],
    component,
    layers: dict[str, Any],
    style: CloudStyle,
    shared: dict[str, Any],
    suptitle: str | None = None,
    labels: Sequence[str] | None = None,
):
    """Draw one row of density-cloud panels in the Figure 2E style."""

    panel_labels = [str(label) for label in (labels if labels is not None else [panel["label"] for panel in panels])]
    if not panel_labels:
        return None

    fig, axes = plt.subplots(
        1,
        len(panel_labels),
        figsize=(float(style.panel_width) * len(panel_labels), float(style.panel_height)),
        facecolor=style.background,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.flatten()
    image = None
    for ax, label in zip(axes, panel_labels):
        image = _draw_panel(ax, label, shared, component, layers, style)

    if image is not None and style.show_colorbar:
        colorbar = fig.colorbar(image, ax=list(axes), fraction=0.018 if len(panel_labels) > 1 else 0.04, pad=0.01)
        colorbar.set_label(style.colorbar_label, color=style.text_color, fontsize=float(style.colorbar_fontsize))
        colorbar.ax.tick_params(colors=style.text_color, labelsize=float(style.colorbar_fontsize))

    if style.show_legend:
        handles = legend_handles(list(layers), style)
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.48, 0.015),
                ncol=min(len(handles), 5),
                frameon=False,
                labelcolor=style.text_color,
                fontsize=float(style.legend_fontsize),
            )

    if suptitle:
        fig.suptitle(suptitle, color=style.text_color, fontsize=float(style.suptitle_fontsize), fontweight="bold")
    return fig


def save_cloud_figure(fig, path: Path, style: CloudStyle, also_svg: bool = True) -> list[str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = [str(path)]
    fig.savefig(path, dpi=int(style.dpi), bbox_inches="tight", facecolor=style.background)
    if also_svg:
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", facecolor=style.background)
        written.append(str(svg_path))
    return written
