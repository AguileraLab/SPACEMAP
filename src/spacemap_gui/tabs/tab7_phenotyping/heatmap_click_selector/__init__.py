from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_BUILD_DIR = Path(__file__).resolve().parent / "frontend" / "build"
_COMPONENT_AVAILABLE = _BUILD_DIR.exists()
_component_func = None

if _COMPONENT_AVAILABLE:
    _component_func = components.declare_component(
        "tab7_heatmap_click_selector",
        path=str(_BUILD_DIR),
    )


def heatmap_click_selector_available() -> bool:
    """Return whether the built local heatmap selector frontend is present."""
    return _COMPONENT_AVAILABLE and _component_func is not None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)

    # NumPy arrays and many Plotly array containers expose ``tolist``.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_safe(to_list())
        except Exception:
            pass

    # NumPy scalar types expose ``item``.
    to_item = getattr(value, "item", None)
    if callable(to_item):
        try:
            return to_item()
        except Exception:
            pass

    return value


def render_heatmap_click_selector(
    *,
    figure: Any,
    available_phenotypes: list[str],
    cluster_display_labels: dict[Any, str] | None = None,
    cluster_marker_scores: dict[Any, dict[str, float]] | None = None,
    reset_counter: int = 0,
    cluster_axis: str = "x",
    key: str | None = None,
):
    """Render the heatmap and return the buffered assignment payload, if any.

    ``cluster_axis`` states which axis carries the clusters. ``"x"`` (the
    default) draws vertical highlight frames over cluster columns; ``"y"`` draws
    horizontal frames over cluster rows.
    """
    if _component_func is None:
        return None

    figure_dict = figure.to_dict() if hasattr(figure, "to_dict") else figure
    figure_dict = json.loads(json.dumps(_json_safe(figure_dict)))
    normalized_labels = {
        str(cluster_id): str(label)
        for cluster_id, label in (cluster_display_labels or {}).items()
    }
    normalized_marker_scores = {
        str(cluster_id): {
            str(marker): float(score)
            for marker, score in marker_scores.items()
        }
        for cluster_id, marker_scores in (cluster_marker_scores or {}).items()
    }

    return _component_func(
        figure=figure_dict,
        availablePhenotypes=[str(value) for value in available_phenotypes],
        clusterDisplayLabels=normalized_labels,
        clusterMarkerScores=normalized_marker_scores,
        resetCounter=int(reset_counter),
        clusterAxis="y" if str(cluster_axis) == "y" else "x",
        key=key,
        default=None,
    )