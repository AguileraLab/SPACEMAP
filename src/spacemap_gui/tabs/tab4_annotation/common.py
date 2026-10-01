"""Shared state, image previews and UI helpers for the Annotation tab.

The Annotation tab drives ``spacemap.tissue_annotation_utils`` through four
sequential subtabs.  Everything that more than one subtab needs - directory
handling, image discovery, the pyramid reader cache, GeoJSON overlays, the
parameter-form helper and figure saving - lives here so each subtab file stays
focused on its own step.

Two conventions are enforced across the tab:

* **Nothing heavy runs on a widget change.** Parameter panels are Streamlit
  forms; the pipeline only executes when an explicit Run button is pressed.
* **Hand-offs always come from disk.** Every step that consumes the previous
  step's GeoJSON re-reads the file, so a QuPath refinement made between steps
  is always picked up instead of a stale in-memory mask.
"""

from __future__ import annotations

from spacemap_gui import terminal
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import streamlit as st

from spacemap_gui.process_log import record_process_event
from spacemap_gui.ui_utils import directory_input_with_browse

TAB_CONTEXT = "tab4"

IMAGE_EXTENSIONS = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".qptiff")

# Runtime products (masks, feature matrices, fitted models) are numpy-heavy and
# must never reach the project JSON.  The leading underscore keeps them out of
# ``persistence.get_tab_name_from_key`` and therefore out of the saved config.
RUNTIME_KEY = "_annot_ta_runtime_cache"

CONFIG_KEYS = (
    "annot_ta_tissue_config",
    "annot_ta_artifact_config",
    "annot_ta_pieces_config",
    "annot_ta_cluster_config",
    "annot_ta_pool_config",
)


# ==========================================================================
# Logging
# ==========================================================================

def log_annotation(message: str, subtab: str = "") -> None:
    """Print a timestamped line and record it in the page process log."""
    terminal.say("tab4", message, step=subtab or None)
    record_process_event(TAB_CONTEXT, "Tissue annotation step", message=message)


# ==========================================================================
# Session state
# ==========================================================================

def default_state(project_root: Path) -> dict[str, Any]:
    """Directory and configuration defaults for a freshly loaded project."""
    from spacemap.tissue_annotation_utils import (
        DEFAULT_ARTIFACT_CONFIG,
        DEFAULT_CLUSTERING_CONFIG,
        DEFAULT_POOLING_CONFIG,
        DEFAULT_TISSUE_CONFIG,
    )
    from spacemap.tissue_continuity import DEFAULT_PIECE_CONFIG

    rois_dir = st.session_state.get("tab3_output_dir") or str(project_root / "outputs" / "rois")
    annotations_root = project_root / "outputs" / "processed_annotations"
    return {
        "annot_ta_input_dir": rois_dir,
        "annot_ta_output_dir": str(annotations_root / "stage0_tissue_annotation"),
        "annot_ta_pool_input_dir": str(annotations_root / "stage0_tissue_annotation"),
        "annot_ta_pool_output_dir": str(annotations_root / "stage1b_compartments_zones"),
        "annot_ta_sample": "",
        "annot_ta_batch_samples": [],
        "annot_ta_cluster_markers": [],
        "annot_ta_artifact_markers": [],
        "annot_ta_he_channels": [],
        "annot_ta_tissue_codex_markers": [],
        "annot_ta_tissue_config": dict(DEFAULT_TISSUE_CONFIG),
        "annot_ta_artifact_config": dict(DEFAULT_ARTIFACT_CONFIG),
        "annot_ta_pieces_config": dict(DEFAULT_PIECE_CONFIG),
        "annot_ta_cluster_config": dict(DEFAULT_CLUSTERING_CONFIG),
        "annot_ta_pool_config": dict(DEFAULT_POOLING_CONFIG),
        "annot_ta_domain_map": {},
        "annot_ta_domain_colors": {},
        "annot_ta_domain_names": [],
        "annot_ta_excluded_clusters": [],
        "annot_ta_pool_rules": [],
        "annot_ta_zone_rules": [],
        "annot_ta_save_figures": True,
    }


def ensure_state(project_root: Path) -> None:
    """Install missing defaults without overwriting restored project values."""
    defaults = default_state(project_root)
    for key, value in defaults.items():
        if key not in st.session_state or st.session_state[key] in (None, ""):
            st.session_state[key] = value
        elif isinstance(value, dict) and isinstance(st.session_state.get(key), dict):
            # Fill in parameters added after a project was first saved.
            merged = {**value, **st.session_state[key]}
            if merged != st.session_state[key]:
                st.session_state[key] = merged


def get_config(key: str) -> dict[str, Any]:
    value = st.session_state.get(key)
    return dict(value) if isinstance(value, dict) else {}


def save_config(key: str, config: dict[str, Any]) -> None:
    """Persist one stage config into session state and the project JSON."""
    st.session_state[key] = dict(config)
    save_project_settings()


def save_project_settings() -> None:
    """Write the current tab settings into ``<project>.json``."""
    try:
        from spacemap_gui.persistence import save_active_project_state

        save_active_project_state("tab4_annotation")
    except Exception as error:  # pragma: no cover - persistence must never block work
        from spacemap_gui.ui_utils import log_exception

        log_exception("saving annotation tab settings", error)


def runtime_store(sample_id: str) -> dict[str, Any]:
    """Per-sample scratch space for masks and feature matrices."""
    cache = st.session_state.setdefault(RUNTIME_KEY, {})
    return cache.setdefault(str(sample_id), {})


def clear_runtime(sample_id: str | None = None) -> None:
    cache = st.session_state.setdefault(RUNTIME_KEY, {})
    if sample_id is None:
        cache.clear()
    else:
        cache.pop(str(sample_id), None)


# ==========================================================================
# Image discovery and readers
# ==========================================================================

def sample_id_from_filename(filename: str) -> str:
    """Strip image extensions so ``P28_roi_1.ome.tiff`` becomes ``P28_roi_1``."""
    name = Path(filename).name
    lowered = name.lower()
    for extension in IMAGE_EXTENSIONS:
        if lowered.endswith(extension):
            return name[: -len(extension)]
    return Path(name).stem


@st.cache_data(show_spinner=False)
def list_images(directory: str, mtime: float) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory)
        if name.lower().endswith(IMAGE_EXTENSIONS) and not name.startswith(".")
    )


def discover_images(directory: str) -> list[str]:
    try:
        mtime = os.path.getmtime(directory) if directory and os.path.isdir(directory) else 0.0
    except OSError:
        mtime = 0.0
    return list_images(directory, mtime)


@st.cache_resource(show_spinner=False)
def get_reader(image_path: str, mtime: float):
    """Cache one :class:`PyramidImageReader` per file (header read only)."""
    from spacemap.tissue_annotation_utils import PyramidImageReader

    return PyramidImageReader(image_path)


def open_reader(image_path: str):
    try:
        mtime = os.path.getmtime(image_path)
    except OSError:
        mtime = 0.0
    return get_reader(image_path, mtime)


def resolve_image_path(sample_id: str) -> str | None:
    """Return the image file backing ``sample_id`` in the configured input directory."""
    input_dir = st.session_state.get("annot_ta_input_dir", "")
    for filename in discover_images(input_dir):
        if sample_id_from_filename(filename) == sample_id:
            return os.path.join(input_dir, filename)
    return None


def output_dir_for(sample_id: str = "", subdir: str = "") -> str:
    base = st.session_state.get("annot_ta_output_dir", "")
    path = Path(base)
    if subdir:
        path = path / subdir
    if sample_id:
        path = path / sample_id
    return str(path)


# The output folder holds the annotation itself; everything else this tab writes
# is working material for review and lives one level down. Without the split a
# five-step run leaves a dozen layers per sample side by side and the one file
# the next stage actually opens is impossible to pick out.
INTERMEDIATE_DIRNAME = "intermediate"

# Layers the rest of the pipeline and the other tabs open by name: the valid
# tissue that carries through the steps, and the products each later step hands
# to segmentation and visualization. These stay in the output folder.
PRIMARY_GEOJSON_SUFFIXES = frozenset({
    "valid_tissue",
    "valid_tissue_from_clusters_highres",
    "macro_domains",
    "zones",
})

# Review and diagnostic layers, plus the boundary that step 2 consumes. The
# boundary is here rather than beside the annotation because it is step 1's
# working output, not the annotation: the annotation is the valid tissue.
INTERMEDIATE_GEOJSON_SUFFIXES = frozenset({
    "tissue_boundary",
    "blood_clot",
    "detected_blur",
    "detected_bright",
    "detected_lint",
    "detected_artifacts",
    "tissue_pieces",
    "raw_macro_clusters",
})


def intermediate_dir(directory: str | None = None) -> str:
    base = directory or st.session_state.get("annot_ta_output_dir", "")
    return os.path.join(base, INTERMEDIATE_DIRNAME)


def geojson_path(sample_id: str, suffix: str, directory: str | None = None) -> str:
    """Where one annotation layer belongs, for both reading and writing."""
    base = directory or st.session_state.get("annot_ta_output_dir", "")
    name = f"{sample_id}_{suffix}.geojson"
    if suffix in PRIMARY_GEOJSON_SUFFIXES:
        return os.path.join(base, name)
    canonical = os.path.join(base, INTERMEDIATE_DIRNAME, name)
    if not os.path.isfile(canonical):
        # A project written before the split keeps opening from where its files
        # already are, so nothing has to be regenerated to keep working.
        legacy = os.path.join(base, name)
        if os.path.isfile(legacy):
            return legacy
    return canonical


def intermediate_file(sample_id: str, filename: str, directory: str | None = None) -> str:
    """Path for a non-GeoJSON side file, such as the artifact review manifest."""
    base = directory or st.session_state.get("annot_ta_output_dir", "")
    canonical = os.path.join(base, INTERMEDIATE_DIRNAME, f"{sample_id}_{filename}")
    if not os.path.isfile(canonical):
        legacy = os.path.join(base, f"{sample_id}_{filename}")
        if os.path.isfile(legacy):
            return legacy
    return canonical


def tidy_output_directory(directory: str | None = None) -> int:
    """Move review layers left in the output folder by an older build down a level.

    Called at the start of a run so a project reorganises itself the first time
    it is used again, rather than needing a migration the user has to know about.
    Only files this tab writes are touched, and every one of them is reproduced
    by re-running the step that made it.
    """
    base = directory or st.session_state.get("annot_ta_output_dir", "")
    if not base or not os.path.isdir(base):
        return 0
    target = os.path.join(base, INTERMEDIATE_DIRNAME)
    moved = 0
    for name in sorted(os.listdir(base)):
        source = os.path.join(base, name)
        if not os.path.isfile(source):
            continue
        stem, extension = os.path.splitext(name)
        if extension.lower() == ".geojson":
            if not any(stem.endswith(f"_{suffix}") for suffix in INTERMEDIATE_GEOJSON_SUFFIXES):
                continue
        elif not stem.endswith("_artifact_review") or extension.lower() != ".json":
            continue
        os.makedirs(target, exist_ok=True)
        destination = os.path.join(target, name)
        try:
            if os.path.isfile(destination):
                # The copy below is the canonical one; the stray beside it is a
                # leftover of the same layer, so it goes.
                os.remove(source)
            else:
                os.replace(source, destination)
            moved += 1
        except OSError:
            continue
    return moved


# ==========================================================================
# Preview rendering
# ==========================================================================

def _stretch_to_uint8(array: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low = float(np.percentile(array, low_percentile))
    high = float(np.percentile(array, high_percentile))
    if high <= low:
        high = low + 1.0
    return np.clip((array - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


@st.cache_data(show_spinner=False, max_entries=16)
def build_thumbnail(image_path: str, mtime: float, mode: str, channel: str = "", max_dim: int = 900):
    """Build a preview thumbnail from the smallest usable pyramid level.

    ``mode`` is ``"H&E"``, ``"CODEX max"`` or ``"Single channel"``.  Reading the
    coarsest level that still exceeds ``max_dim`` keeps this fast even for
    17k x 14k ROIs.
    """
    from spacemap.tissue_annotation_utils import split_he_and_codex_channels

    reader = open_reader(image_path)
    factor = 1
    while True:
        height, width = reader.target_shape(factor * 2)
        if max(height, width) < max_dim or factor * 2 > 2 ** max(reader.level_count - 1, 0):
            break
        factor *= 2
    factor = max(factor, 1)

    he_indices, codex_indices = split_he_and_codex_channels(reader.channel_names)
    with reader.open() as tif:
        if mode == "H&E" and len(he_indices) >= 3:
            channels = [reader.read_channel(tif, index, factor) for index in he_indices[:3]]
            rgb = np.dstack([
                np.clip(plane / max(float(np.percentile(plane, 99.5)), 1.0), 0, 1) * 255.0
                for plane in channels
            ]).astype(np.uint8)
            return rgb, reader.target_shape(factor)
        if mode == "Single channel" and channel in reader.channel_names:
            plane = reader.read_channel(tif, reader.channel_names.index(channel), factor)
            gray = _stretch_to_uint8(plane)
            return np.dstack([gray] * 3), reader.target_shape(factor)

        projection = None
        for index in (codex_indices or list(range(reader.n_channels)))[:24]:
            plane = reader.read_channel(tif, index, factor)
            projection = plane if projection is None else np.maximum(projection, plane)
        gray = _stretch_to_uint8(projection if projection is not None else np.zeros((16, 16), dtype=np.float32))
        return np.dstack([gray] * 3), reader.target_shape(factor)


def render_image_preview(image_path: str, key_prefix: str, max_dim: int = 900) -> None:
    """Render the shared right-hand preview panel used by every subtab."""
    reader = open_reader(image_path)
    mode = st.radio(
        "Preview layer",
        ["H&E", "CODEX max", "Single channel"],
        horizontal=True,
        key=f"{key_prefix}_preview_mode",
        help=(
            "H&E renders the three virtual H&E channels as an RGB image. "
            "CODEX max shows the maximum projection across CODEX markers. "
            "Single channel shows one selected marker."
        ),
    )
    channel = ""
    if mode == "Single channel":
        channel = st.selectbox(
            "Channel", reader.channel_names, key=f"{key_prefix}_preview_channel",
            help="Marker channel rendered in the preview.",
        )
    try:
        mtime = os.path.getmtime(image_path)
    except OSError:
        mtime = 0.0
    thumbnail, shape = build_thumbnail(image_path, mtime, mode, channel, max_dim)
    st.image(thumbnail, width="stretch", caption=f"{Path(image_path).name} - preview grid {shape[0]} x {shape[1]}")


@st.cache_data(show_spinner=False, max_entries=24)
def _render_polygon_overlay(
    image_path: str,
    geojson_paths: tuple[str, ...],
    stamps: tuple[float, ...],
    max_dim: int,
    fill_alpha: int,
    line_width: int,
    class_colors_items: tuple[tuple[str, str], ...],
    preview_mode: str,
    class_filter: tuple[str, ...] | None = None,
) -> tuple[bytes, dict[str, str]]:
    """Rasterize the overlay once per (image, files, settings) combination.

    Every subtab of the page runs on every rerun, so without this cache four
    overlays would be redrawn from GeoJSON on each heatmap click. ``stamps``
    carries the files' modification times so a QuPath edit still invalidates it.
    """
    image, colors = _build_polygon_overlay(
        image_path, list(geojson_paths), max_dim, fill_alpha, line_width,
        dict(class_colors_items), preview_mode, class_filter,
    )
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), colors


def overlay_polygons_on_thumbnail(
    image_path: str,
    geojson_paths: Sequence[str],
    max_dim: int = 900,
    fill_alpha: int = 70,
    line_width: int = 2,
    class_colors: dict[str, str] | None = None,
    preview_mode: str = "H&E",
    class_filter: Sequence[str] | None = None,
):
    """Draw colored annotation polygons over an image thumbnail.

    ``class_filter`` restricts the drawing to specific classification names,
    which the Segmentation tab uses to show only the polygon chosen as the
    segmentation mask.

    Returns ``(PNG bytes, {class name: hex color})`` so the caller can render a
    matching legend. The result is cached against the source files' timestamp
    and size, so a re-run that rewrites a layer always redraws it.
    """
    paths = tuple(str(path) for path in geojson_paths if path)
    # Size as well as modification time: a re-run that rewrites a layer
    # within the same clock tick still changes its length, and some network
    # filesystems only keep whole-second timestamps.
    stamps = tuple(
        (os.path.getmtime(path), os.path.getsize(path)) if os.path.isfile(path) else (0.0, 0)
        for path in paths
    )
    return _render_polygon_overlay(
        image_path, paths, stamps, max_dim, fill_alpha, line_width,
        tuple(sorted((class_colors or {}).items())), preview_mode,
        tuple(str(name) for name in class_filter) if class_filter else None,
    )


def _build_polygon_overlay(
    image_path: str,
    geojson_paths: Sequence[str],
    max_dim: int,
    fill_alpha: int,
    line_width: int,
    class_colors: dict[str, str] | None,
    preview_mode: str,
    class_filter: Sequence[str] | None = None,
):
    """Draw the annotation polygons over an image thumbnail."""
    from PIL import Image, ImageDraw
    from shapely.geometry import shape as shapely_shape

    from spacemap.tissue_annotation_utils import (
        _iter_geojson_features,
        assign_distinct_colors,
        feature_class_name,
    )

    try:
        mtime = os.path.getmtime(image_path)
    except OSError:
        mtime = 0.0
    thumbnail, _ = build_thumbnail(image_path, mtime, preview_mode, "", max_dim)
    reader = open_reader(image_path)
    full_height, full_width = reader.full_shape

    pil_image = Image.fromarray(np.asarray(thumbnail, dtype=np.uint8)).convert("RGB")
    scale_x = pil_image.width / max(full_width, 1)
    scale_y = pil_image.height / max(full_height, 1)

    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Collect every class first so colours can be spread across the palette
    # instead of being assigned one hash at a time as polygons are met.
    loaded: list[tuple[str, list]] = []
    class_order: list[str] = []
    for path in geojson_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            features = _iter_geojson_features(path)
        except Exception:
            continue
        wanted = {str(name).lower() for name in class_filter} if class_filter else None
        if wanted is not None:
            features = [f for f in features if feature_class_name(f).lower() in wanted]
        loaded.append((path, features))
        for feature in features:
            name = feature_class_name(feature)
            if name not in class_order:
                class_order.append(name)
    colors: dict[str, str] = assign_distinct_colors(class_order, existing=class_colors)

    for _path, features in loaded:
        for feature in features:
            name = feature_class_name(feature)
            hex_color = colors.get(name, "#888888")
            rgb = tuple(int(hex_color.lstrip("#")[index:index + 2], 16) for index in (0, 2, 4))
            try:
                geometry = shapely_shape(feature["geometry"])
            except Exception:
                continue
            parts = list(geometry.geoms) if geometry.geom_type in ("MultiPolygon", "GeometryCollection") else [geometry]
            for part in parts:
                if getattr(part, "geom_type", "") != "Polygon":
                    continue
                exterior = [(x * scale_x, y * scale_y) for x, y in part.exterior.coords]
                if len(exterior) < 3:
                    continue
                draw.polygon(exterior, fill=rgb + (fill_alpha,), outline=rgb + (255,), width=line_width)
                for ring in part.interiors:
                    hole = [(x * scale_x, y * scale_y) for x, y in ring.coords]
                    if len(hole) >= 3:
                        draw.polygon(hole, fill=(0, 0, 0, 0), outline=rgb + (200,), width=max(line_width - 1, 1))

    return Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB"), colors


def render_class_legend(colors: dict[str, str]) -> None:
    if not colors:
        return
    chips = " ".join(
        f"<span style='display:inline-block;margin:2px 8px 2px 0;'>"
        f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
        f"border:1px solid #666;vertical-align:middle;margin-right:4px;'></span>"
        f"<span style='vertical-align:middle;font-size:13px;'>{name}</span></span>"
        for name, color in sorted(colors.items())
    )
    st.markdown(chips, unsafe_allow_html=True)


# ==========================================================================
# Parameter forms
# ==========================================================================

def render_parameter_form(
    *,
    form_key: str,
    spec: Sequence[dict[str, Any]],
    config: dict[str, Any],
    columns: int = 2,
    submit_label: str = "Save settings",
    extra_controls: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Render a grouped parameter panel inside a form.

    Nothing recomputes while the user edits values: Streamlit buffers all
    widgets in the form and only reruns once, on submit.  Every entry in
    ``spec`` carries a ``help`` string which Streamlit renders as the small
    ``(?)`` marker used throughout the pipeline.

    Each spec entry is ``{"key", "label", "help", "type"}`` plus type-specific
    fields.  Supported types: ``number``, ``int``, ``slider``, ``select``,
    ``multiselect``, ``checkbox``, ``text`` and ``header`` (a section title).
    """
    updated = dict(config)
    with st.form(form_key):
        column_objects: list[Any] = []
        position = 0
        for entry in spec:
            entry_type = entry.get("type", "number")
            if entry_type == "header":
                st.markdown(f"**{entry['label']}**")
                if entry.get("help"):
                    st.caption(entry["help"])
                column_objects = []
                position = 0
                continue

            if not column_objects or position % columns == 0:
                column_objects = st.columns(columns)
                position = 0
            target = column_objects[position % columns]
            position += 1

            key = entry["key"]
            widget_key = f"{form_key}__{key}"
            current = config.get(key, entry.get("default"))
            help_text = entry.get("help", "")

            with target:
                if entry_type == "checkbox":
                    updated[key] = st.checkbox(entry["label"], value=bool(current), key=widget_key, help=help_text)
                elif entry_type == "select":
                    options = list(entry["options"])
                    index = options.index(current) if current in options else 0
                    updated[key] = st.selectbox(entry["label"], options, index=index, key=widget_key, help=help_text)
                elif entry_type == "multiselect":
                    options = list(entry["options"])
                    default = [value for value in (current or []) if value in options]
                    updated[key] = st.multiselect(entry["label"], options, default=default, key=widget_key, help=help_text)
                elif entry_type == "text":
                    updated[key] = st.text_input(entry["label"], value=str(current or ""), key=widget_key, help=help_text)
                elif entry_type == "slider":
                    updated[key] = st.slider(
                        entry["label"],
                        min_value=float(entry.get("min", 0.0)),
                        max_value=float(entry.get("max", 1.0)),
                        value=float(current if current is not None else entry.get("default", 0.0)),
                        step=float(entry.get("step", 0.01)),
                        key=widget_key,
                        help=help_text,
                    )
                elif entry_type == "int":
                    updated[key] = int(st.number_input(
                        entry["label"],
                        min_value=int(entry.get("min", 0)),
                        max_value=int(entry.get("max", 1_000_000)),
                        value=int(current if current is not None else entry.get("default", 0)),
                        step=int(entry.get("step", 1)),
                        key=widget_key,
                        help=help_text,
                    ))
                else:
                    updated[key] = float(st.number_input(
                        entry["label"],
                        min_value=float(entry.get("min", 0.0)),
                        max_value=float(entry.get("max", 1e9)),
                        value=float(current if current is not None else entry.get("default", 0.0)),
                        step=float(entry.get("step", 0.1)),
                        format=entry.get("format", "%.4g"),
                        key=widget_key,
                        help=help_text,
                    ))

        if extra_controls is not None:
            extra_controls()
        submitted = st.form_submit_button(submit_label, width="stretch")
    return updated, submitted


# ==========================================================================
# Figures
# ==========================================================================

def save_figure(figure, sample_id: str, name: str) -> str | None:
    """Save a matplotlib figure into ``<output>/figures/<sample>/`` as PNG + PDF."""
    if not st.session_state.get("annot_ta_save_figures", True):
        return None
    directory = output_dir_for(sample_id, "figures")
    try:
        os.makedirs(directory, exist_ok=True)
        png_path = os.path.join(directory, f"{name}.png")
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        figure.savefig(os.path.join(directory, f"{name}.pdf"), bbox_inches="tight")
        return png_path
    except Exception as error:
        from spacemap_gui.ui_utils import log_exception

        log_exception(f"saving figure {name}", error)
        return None


def show_figure(figure, sample_id: str, name: str) -> None:
    """Render a figure in the UI and persist it alongside the GeoJSON outputs."""
    if figure is None:
        # Builders return None when there is nothing to draw (no clusters, no
        # assignments yet); skipping is the correct result, not an error.
        return
    st.pyplot(figure, width="stretch")
    save_figure(figure, sample_id, name)
    import matplotlib.pyplot as plt

    plt.close(figure)


def show_cached_figure(sample_id: str, name: str, signature: str, builder: Callable[[], Any]) -> bool:
    """Render a figure once per ``signature`` and re-show it as a cached image.

    Large spatial maps are expensive to draw and to write as PNG + PDF. Calling
    ``st.pyplot`` on every rerun would repeat that work on every heatmap click,
    which is what made domain assignment feel slow. Rasterizing once and
    serving the bytes afterwards keeps interaction instant while the figure on
    disk stays in step with what is displayed.

    Returns True when something was rendered.
    """
    from io import BytesIO

    import matplotlib.pyplot as plt

    cache = runtime_store(sample_id).setdefault("figure_cache", {})
    entry = cache.get(name)
    if entry is None or entry.get("signature") != signature:
        figure = builder()
        if figure is None:
            cache[name] = {"signature": signature, "png": None}
        else:
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            save_figure(figure, sample_id, name)
            plt.close(figure)
            cache[name] = {"signature": signature, "png": buffer.getvalue()}
        entry = cache[name]

    if entry.get("png"):
        st.image(entry["png"], width="stretch")
        return True
    return False


def save_table(dataframe, sample_id: str, name: str) -> str | None:
    """Save a results table next to the figures so it can be reported later."""
    if dataframe is None or getattr(dataframe, "empty", True):
        return None
    directory = output_dir_for(sample_id, "figures")
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.csv")
        dataframe.to_csv(path)
        return path
    except Exception as error:
        from spacemap_gui.ui_utils import log_exception

        log_exception(f"saving table {name}", error)
        return None


# ==========================================================================
# Shared UI blocks
# ==========================================================================

def render_directory_block(include_pool_dirs: bool = True) -> None:
    """Directory selectors for the whole tab.

    Streamlit renders every subtab of an ``st.tabs`` group in the same script
    run, so these widgets must be created exactly once, at page level, or their
    keys collide.  The subtabs read the resulting session-state values.
    """
    directory_input_with_browse(
        label="Input image directory (ROI images)",
        text_key="annot_ta_input_dir",
        default_value=st.session_state.get("annot_ta_input_dir", ""),
        browser_key_prefix="annot_ta_input_dir_browse",
        help_text=(
            "Folder holding the pyramidal OME-TIFF ROI images. This is normally the "
            "ROI Extraction output directory."
        ),
    )
    directory_input_with_browse(
        label="Tissue annotation output directory",
        text_key="annot_ta_output_dir",
        default_value=st.session_state.get("annot_ta_output_dir", ""),
        browser_key_prefix="annot_ta_output_dir_browse",
        help_text=(
            "GeoJSON files, figures and tables from steps 1-3 are written here. "
            "Open these files in QuPath to refine them between steps."
        ),
    )
    if include_pool_dirs:
        directory_input_with_browse(
            label="Domain GeoJSON directory (input for pooling)",
            text_key="annot_ta_pool_input_dir",
            default_value=st.session_state.get("annot_ta_pool_input_dir", ""),
            browser_key_prefix="annot_ta_pool_input_browse",
            help_text="Folder containing the merged-domain GeoJSON files produced by step 3 (or refined in QuPath).",
        )
        directory_input_with_browse(
            label="Pooled + zoned GeoJSON output directory",
            text_key="annot_ta_pool_output_dir",
            default_value=st.session_state.get("annot_ta_pool_output_dir", ""),
            browser_key_prefix="annot_ta_pool_output_browse",
            help_text="Final annotation GeoJSON files used by the downstream cell annotation step.",
        )


def render_sample_selector(key_prefix: str = "annot_ta", allow_batch: bool = True) -> tuple[str, list[str]]:
    """Select the working image plus an optional batch selection.

    Like :func:`render_directory_block` this belongs at page level: one image
    choice drives every subtab, and duplicating the widget across subtabs would
    both collide on keys and let the last-rendered subtab silently win.

    Returns ``(sample_id, batch_sample_ids)``.
    """
    input_dir = st.session_state.get("annot_ta_input_dir", "")
    filenames = discover_images(input_dir)
    if not filenames:
        st.warning(f"No images found in `{input_dir or '(not set)'}`.")
        return "", []

    samples = [sample_id_from_filename(name) for name in filenames]
    prune_widget_selection(f"{key_prefix}_sample_select", samples)
    prune_widget_selection(f"{key_prefix}_batch_select", samples)
    current = st.session_state.get("annot_ta_sample", "")
    index = samples.index(current) if current in samples else 0
    sample_id = st.selectbox(
        "Image",
        samples,
        index=index,
        key=f"{key_prefix}_sample_select",
        help="The image this step works on. Steps 1, 2 and 4 can also run in batch over several images.",
    )
    if sample_id != st.session_state.get("annot_ta_sample"):
        st.session_state["annot_ta_sample"] = sample_id

    batch: list[str] = []
    if allow_batch:
        saved = [value for value in st.session_state.get("annot_ta_batch_samples", []) if value in samples]
        batch = st.multiselect(
            "Batch selection (optional)",
            samples,
            default=saved or [sample_id],
            key=f"{key_prefix}_batch_select",
            help=(
                "Images processed by the batch Run button of this step. "
                "Batch runs reuse the parameters shown on the left for every selected image."
            ),
        )
        st.session_state["annot_ta_batch_samples"] = batch
    return sample_id, batch


def prune_widget_selection(key: str, options: Sequence[Any]) -> None:
    """Drop stored widget values that are no longer valid options.

    Streamlit rejects a session-state value that is not in a select widget's
    option list.  Because this tab's options depend on the selected image and on
    which output files exist yet, stored selections routinely go stale between
    renders; pruning them here keeps the widget from raising.
    """
    stored = st.session_state.get(key)
    option_set = set(options)
    if isinstance(stored, list):
        pruned = [value for value in stored if value in option_set]
        if len(pruned) != len(stored):
            st.session_state[key] = pruned
    elif stored is not None and stored not in option_set:
        st.session_state.pop(key, None)


def sticky_file_selectbox(
    *,
    label: str,
    candidates: Sequence[str],
    preferred: str,
    key: str,
    help_text: str,
) -> str:
    """A file selectbox that snaps to ``preferred`` the first time it appears.

    A plain ``st.selectbox`` keyed in session state ignores its ``index``
    argument once the key exists, so after an earlier step finally produces its
    output file the widget would keep pointing at whatever was selected before
    that file existed.  This snaps to the expected hand-off file exactly once,
    when it first shows up on disk, and respects any later manual choice.
    """
    candidates = list(candidates)
    if not candidates:
        return ""
    seen_key = f"_{key}__preferred_seen"
    if preferred in candidates:
        if not st.session_state.get(seen_key, False):
            st.session_state[key] = preferred
            st.session_state[seen_key] = True
    else:
        st.session_state[seen_key] = False

    current = st.session_state.get(key)
    if current not in candidates:
        st.session_state.pop(key, None)
        current = None
    index = candidates.index(current) if current in candidates else 0
    return st.selectbox(label, candidates, index=index, key=key, help=help_text)


def describe_output(path: str, label: str) -> None:
    """Consistent success/absent messaging for a produced GeoJSON."""
    if path and os.path.isfile(path):
        size_kb = os.path.getsize(path) / 1024.0
        st.success(f"{label}: `{path}` ({size_kb:,.0f} KB)")
    else:
        st.info(f"{label} has not been generated yet.")


def qupath_hint(path: str, next_step: str) -> None:
    st.info(
        f"**Review step:** open `{os.path.basename(path)}` in QuPath, refine it if needed, and save it "
        f"under the same name. {next_step} always re-reads this file from disk, so your edits are used."
    )
