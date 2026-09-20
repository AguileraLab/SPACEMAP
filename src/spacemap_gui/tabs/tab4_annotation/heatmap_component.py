"""Click-to-merge heatmap component for the Annotation tab.

Reuses the built frontend that Tab 7 ships (``tab7_phenotyping/
heatmap_click_selector/frontend/build``) but declares it under its own
component name.  Declaring it here rather than importing Tab 7's package keeps
the Annotation tab from executing the whole phenotyping page on import, while
still guaranteeing both tabs behave identically: click cells to buffer a
selection, then apply one label to every selected column at once.

If the frontend has not been built in this installation the caller falls back
to a static Plotly chart plus the assignment table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

_BUILD_DIR = (
    Path(__file__).resolve().parents[1]
    / "tab7_phenotyping"
    / "heatmap_click_selector"
    / "frontend"
    / "build"
)
_COMPONENT_AVAILABLE = _BUILD_DIR.exists()
_component_func = None

if _COMPONENT_AVAILABLE:
    try:
        _component_func = components.declare_component(
            "tab4_annotation_heatmap_click_selector",
            path=str(_BUILD_DIR),
        )
    except Exception:  # pragma: no cover - a broken build must not break the tab
        _COMPONENT_AVAILABLE = False
        _component_func = None


def heatmap_click_selector_available() -> bool:
    """Whether the built click-selector frontend is usable."""
    return _COMPONENT_AVAILABLE and _component_func is not None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _json_safe(to_list())
        except Exception:
            pass
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
    cluster_axis: str = "y",
    key: str | None = None,
):
    """Render the heatmap and return the buffered assignment payload, if any.

    This tab lays clusters out on the Y axis so the interactive heatmap matches
    the seaborn clustermap above it; the highlight frames are therefore
    horizontal bands over cluster rows.
    """
    if _component_func is None:
        return None

    figure_dict = figure.to_dict() if hasattr(figure, "to_dict") else figure
    figure_dict = json.loads(json.dumps(_json_safe(figure_dict)))
    return _component_func(
        figure=figure_dict,
        availablePhenotypes=[str(value) for value in available_phenotypes],
        clusterDisplayLabels={
            str(cluster_id): str(label)
            for cluster_id, label in (cluster_display_labels or {}).items()
        },
        clusterMarkerScores={
            str(cluster_id): {str(marker): float(score) for marker, score in scores.items()}
            for cluster_id, scores in (cluster_marker_scores or {}).items()
        },
        resetCounter=int(reset_counter),
        clusterAxis="y" if str(cluster_axis) == "y" else "x",
        key=key,
        default=None,
    )
