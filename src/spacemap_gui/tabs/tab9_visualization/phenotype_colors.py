"""Shared phenotype colours for Tab 9 visualizations.

The canonical values deliberately use the keys originally created by the
Actual Tissue / Cell Visualization panels.  Panel-specific widget keys are
kept separate: Streamlit executes every loaded tab on a rerun and rejects two
widgets with the same key, even when the widgets appear in different tabs.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from typing import Any

from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import pandas as pd
import streamlit as st

from spacemap.acc_utils import build_color_mapping


_CANONICAL_COLOR_PREFIX = "tab9_actual_cell_color_val_"
_PALETTE_STATE_VERSION_KEY = "tab9_actual_phenotype_color_state_version"
_PALETTE_STATE_VERSION = 1


def sanitize_color_key(value: Any) -> str:
    """Return the legacy-safe key fragment shared with the tissue panels."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "item"


def canonical_color_key(label: Any) -> str:
    """Return the durable session-state key for one phenotype label."""
    return f"{_CANONICAL_COLOR_PREFIX}{sanitize_color_key(label)}"


def _normalise_hex(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    try:
        return to_hex(str(value), keep_alpha=False)
    except (TypeError, ValueError):
        return None


def initial_category_colors(
    labels: Sequence[Any],
    stored_colors: MutableMapping[str, Any] | dict[str, Any],
    *,
    migrate_legacy_all_black: bool = False,
) -> dict[str, str]:
    """Resolve saved choices, then house colours, then stable fallback colours."""
    clean_labels = sorted({str(label) for label in labels}, key=str.casefold)
    valid: dict[str, str] = {}
    for label in clean_labels:
        color = _normalise_hex(stored_colors.get(label))
        if color is not None:
            valid[label] = color

    if (
        migrate_legacy_all_black
        and len(valid) == len(clean_labels)
        and valid
        and all(color == "#000000" for color in valid.values())
    ):
        valid = {}

    return build_color_mapping(clean_labels, existing=valid)


def category_labels_for_columns(frame: pd.DataFrame | None, columns: Sequence[str]) -> list[str]:
    """Collect non-empty categorical labels across the selected columns."""
    if frame is None or frame.empty:
        return []
    labels: set[str] = set()
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        labels.update(value for value in values.unique() if value)
    return sorted(labels, key=str.casefold)


def shared_phenotype_colors(
    labels: Sequence[Any],
    state: MutableMapping[str, Any] | None = None,
) -> dict[str, str]:
    """Read and initialize the canonical shared palette in ``state``."""
    state = st.session_state if state is None else state
    clean_labels = sorted({str(label) for label in labels}, key=str.casefold)
    try:
        palette_state_version = int(state.get(_PALETTE_STATE_VERSION_KEY, 0) or 0)
    except (TypeError, ValueError):
        palette_state_version = 0
    colors = initial_category_colors(
        clean_labels,
        {label: state.get(canonical_color_key(label)) for label in clean_labels},
        migrate_legacy_all_black=palette_state_version < _PALETTE_STATE_VERSION,
    )
    for label, color in colors.items():
        state[canonical_color_key(label)] = color
    state[_PALETTE_STATE_VERSION_KEY] = _PALETTE_STATE_VERSION
    return colors


def render_phenotype_color_controls(
    labels: Sequence[Any],
    *,
    panel_key: str,
    title: str = "Phenotype colors",
    expanded: bool = False,
) -> dict[str, str]:
    """Render synchronized, editable color pickers and return their mapping."""
    clean_labels = sorted({str(label) for label in labels}, key=str.casefold)
    if not clean_labels:
        return {}

    colors = shared_phenotype_colors(clean_labels)
    house_colors = build_color_mapping(clean_labels)
    with st.expander(title, expanded=expanded):
        st.caption(
            "Known phenotypes start with the Cell Visualization house scheme. "
            "Changes are shared with the other loaded visualization panels."
        )
        if st.button("Reset to house colour scheme", key=f"{panel_key}_reset_house_colors"):
            for label in clean_labels:
                color = house_colors[label]
                st.session_state[canonical_color_key(label)] = color
                colors[label] = color

        picker_columns = st.columns(4)
        for index, label in enumerate(clean_labels):
            canonical_key = canonical_color_key(label)
            canonical_value = _normalise_hex(st.session_state.get(canonical_key)) or colors[label]
            fragment = sanitize_color_key(label)
            widget_key = f"{panel_key}_phenotype_color_{fragment}_{index}"
            sync_key = f"_{widget_key}_canonical"

            # If another panel changed the canonical value, refresh this panel's
            # widget before it is instantiated. If the picker itself changed,
            # its previous sync marker still equals the old canonical value and
            # its new widget value is preserved below.
            if widget_key not in st.session_state or st.session_state.get(sync_key) != canonical_value:
                st.session_state[widget_key] = canonical_value

            with picker_columns[index % len(picker_columns)]:
                chosen = st.color_picker(label, key=widget_key)
            chosen = _normalise_hex(chosen) or canonical_value
            st.session_state[canonical_key] = chosen
            st.session_state[sync_key] = chosen
            colors[label] = chosen

    return colors


def apply_category_tick_colors(axis: Any, color_map: dict[str, str], *, which: str) -> None:
    """Color categorical tick labels without changing a continuous heatmap scale."""
    tick_labels = axis.get_xticklabels() if which == "x" else axis.get_yticklabels()
    for tick_label in tick_labels:
        label = str(tick_label.get_text())
        if label in color_map:
            tick_label.set_color(color_map[label])
            tick_label.set_fontweight("semibold")


def add_category_color_strip(
    axis: Any,
    labels: Sequence[Any],
    color_map: dict[str, str],
    *,
    which: str,
) -> None:
    """Add a compact phenotype annotation strip to a heatmap axis."""
    clean_labels = [str(label) for label in labels]
    if which == "x":
        transform = axis.get_xaxis_transform()
        for index, label in enumerate(clean_labels):
            axis.add_patch(
                Rectangle(
                    (index, 1.005),
                    1.0,
                    0.025,
                    transform=transform,
                    clip_on=False,
                    facecolor=color_map.get(label, "#777777"),
                    edgecolor="none",
                )
            )
    elif which == "y":
        transform = axis.get_yaxis_transform()
        for index, label in enumerate(clean_labels):
            axis.add_patch(
                Rectangle(
                    (-0.03, index),
                    0.025,
                    1.0,
                    transform=transform,
                    clip_on=False,
                    facecolor=color_map.get(label, "#777777"),
                    edgecolor="none",
                )
            )
    else:
        raise ValueError("which must be 'x' or 'y'")
