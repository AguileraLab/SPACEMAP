"""Subtab 2b - turn the valid tissue into reviewed, analysable tissue pieces.

A fragmented section carries several physically separate scraps of biopsy, and a
single scrap is sometimes two lobes joined by a thin bridge. Both matter for
every distance and neighbourhood measured later, so this step makes them
explicit before any cell is annotated:

* split the valid tissue into its separate pieces and measure each one,
* report the narrow bridges (necks) inside a piece, and optionally cut there,
* drop pieces too small - by area, or by the widest circle that fits inside
  them - to support the analysis radius the study needs.

Nothing is inferred from cell positions: a sparsely populated region is still
tissue, and only the tissue polygon decides where a gap is. The step reads its
input GeoJSON from disk, so a QuPath correction made after step 2 is honoured.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from spacemap_gui.process_log import record_process_event

from .common import (
    TAB_CONTEXT,
    describe_output,
    geojson_path,
    log_annotation,
    overlay_polygons_on_thumbnail,
    render_class_legend,
    render_parameter_form,
    resolve_image_path,
    runtime_store,
    save_config,
    save_table,
)

CONFIG_KEY = "annot_ta_pieces_config"

OUTPUT_SUFFIX = "tissue_pieces"

# The kept pieces keep the class name of the file they came from, so this output
# is a drop-in replacement for the valid-tissue GeoJSON everywhere downstream.
KEPT_CLASS = "Valid_Tissue"
REJECTED_CLASS = "Rejected_Tissue"
NECK_CLASS = "Detected_Neck"
KEPT_COLOR = "#2ecc71"
REJECTED_COLOR = "#9e9e9e"
NECK_COLOR = "#e91e63"

# The step reads the annotation the previous steps have been building and writes
# it back, so there is nothing to choose: one file carries through the tab.
SOURCE_SUFFIX = "valid_tissue"
REVIEW_SUFFIX = "tissue_pieces"


def _parameter_spec(microns_per_pixel: float) -> list[dict]:
    """Form layout. Thresholds are stated in real units and converted with the MPP."""
    return [
        {"type": "header", "label": "Narrow bridges (necks)",
         "help": (
             "A neck is a narrow strip of tissue that separates two larger sections of the same "
             "piece, so the tissue widens out again on both sides of it. Picture flooding one side: "
             "water crosses a wide bridge freely but barely trickles through a narrow one. A bleb or "
             "a shred hanging off the edge of a mass is not a neck, because only one side is real "
             "tissue - which is why both sides have to pass the size tests below."
         )},
        {"key": "detect_necks", "label": "Detect necks", "type": "checkbox", "default": True,
         "help": "Report the bridges. Detection on its own never changes the tissue."},
        {"key": "neck_method", "label": "How bridges are measured", "type": "select",
         "options": ["raster", "polygon"],
         "help": (
             "Finding a bridge means opening the piece with discs of many sizes and seeing when it "
             "falls in two. 'raster' does that on a small binary image, which is what makes the step "
             "usable on a hand-drawn outline carrying tens of thousands of vertices: seconds rather "
             "than minutes per piece. Only the search moves; what is cut out of the tissue is still "
             "taken out of the polygon you drew, so the outline keeps its own precision everywhere "
             "except across the cut. 'polygon' is the older all-vector search, kept for comparison."
         )},
        {"key": "neck_min_width_um", "label": "Widest bridge still called a neck (\u00b5m)",
         "type": "number", "default": 100.0, "min": 1.0, "max": 2000.0, "step": 10.0,
         "help": (
             "A bridge wider than this is treated as ordinary continuous tissue. Set it to the "
             "largest neighbourhood radius the study uses: tissue narrower than your measuring "
             "circle cannot support it. Each bridge's own width is then measured, so a neck found "
             "at this setting is usually reported much narrower."
         )},
        {"key": "neck_min_lobe_area_mm2", "label": "Each side must be at least (mm\u00b2)",
         "type": "number", "default": 0.02, "min": 0.0, "max": 10.0, "step": 0.005, "format": "%.4f",
         "help": (
             "The size test that separates a real neck from a bleb. 0.02 mm\u00b2 is a patch roughly "
             "160 \u00b5m across - an island of tissue rather than a sliver. Raise it to report only "
             "bridges between two major sections."
         )},
        {"key": "neck_min_lobe_fraction", "label": "...and at least this share of the piece",
         "type": "slider", "default": 0.02, "min": 0.0, "max": 0.5, "step": 0.01,
         "help": (
             "A backstop for small pieces, where an absolute area alone would accept almost any "
             "bump. 0.02 means the smaller side must hold at least 2% of the piece."
         )},
        {"key": "cut_at_necks", "label": "Cut pieces at their necks", "type": "checkbox", "default": False,
         "help": (
             "Split a piece into separate lobes at the narrowest point of the bridge. Leave off to "
             "report necks without changing the tissue; turn on when two lobes should be treated as "
             "independent pieces."
         )},

        {"type": "header", "label": "Size filters",
         "help": (
             "Small scraps cannot support radius-based measurements, and their cells otherwise enter "
             "every per-piece average. Rejected pieces are kept in the file for review, not deleted."
         )},
        {"key": "filter_by_area", "label": "Reject by area", "type": "checkbox", "default": True,
         "help": "Drop pieces smaller than the area below."},
        {"key": "min_area_mm2", "label": "Minimum piece area (mm²)", "type": "number",
         "default": 0.05, "min": 0.0, "max": 100.0, "step": 0.005, "format": "%.4f",
         "help": (
             f"At {microns_per_pixel:g} µm/pixel, 0.05 mm² is about "
             f"{0.05 * 1e6 / max(microns_per_pixel, 1e-9) ** 2:,.0f} pixels², a disc roughly "
             "126 \u00b5m across."
         )},
        {"key": "filter_by_inscribed_circle", "label": "Reject by widest fitting circle",
         "type": "checkbox", "default": False,
         "help": (
             "Reject a piece unless a circle of the diameter below fits inside it without leaving "
             "tissue. Catches long thin shreds that pass an area test but cannot host a single "
             "complete neighbourhood."
         )},
        {"key": "min_inscribed_diameter_um", "label": "Circle must fit (\u00b5m across)", "type": "number",
         "default": 100.0, "min": 1.0, "max": 5000.0, "step": 10.0,
         "help": "Usually the same as the largest neighbourhood diameter used in the analysis."},

        {"type": "header", "label": "Geometry clean-up",
         "help": "Optional tidying applied before the pieces are measured."},
        {"key": "fill_holes_below_um2", "label": "Fill holes below (\u00b5m\u00b2)", "type": "number",
         "default": 0.0, "min": 0.0, "max": 1e6, "step": 10.0,
         "help": (
             "Close specks left inside a piece by segmentation. Leave at 0 to keep every hole: a real "
             "lumen is a void that a straight line should not cross."
         )},
        {"key": "simplify_px", "label": "Simplify outline (pixels)", "type": "number",
         "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.5,
         "help": "Reduces vertex count. 0 keeps the outline exactly as detected."},
        {"key": "microns_per_pixel", "label": "Micrometres per pixel", "type": "number",
         "default": float(microns_per_pixel), "min": 0.0001, "max": 100.0, "step": 0.0001,
         "format": "%.4f",
         "help": "Image calibration used for every physical threshold above. Taken from step 1."},
    ]


# ==========================================================================
# Pipeline
# ==========================================================================

# ==========================================================================
# Sources
# ==========================================================================

@st.cache_data(show_spinner=False, max_entries=4)
def _read_annotation(path: str, mtime: float, size: int) -> dict[str, dict]:
    """Every polygon in a GeoJSON, grouped by class, with the colour it was drawn in.

    A hand-drawn annotation commonly holds one feature per class, and that
    feature is often a GeometryCollection rather than a MultiPolygon, so the
    parts are flattened rather than read off the geometry type. Keyed on the
    file's timestamp and size because a 27 MB annotation takes several seconds
    to parse and the panel re-renders on every widget touch.
    """
    from shapely.geometry import shape

    from spacemap.tissue_continuity import _as_polygons

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    features = data.get("features", []) if isinstance(data, dict) else data
    grouped: dict[str, dict] = {}
    for feature in features:
        properties = feature.get("properties") or {}
        classification = properties.get("classification")
        if isinstance(classification, dict):
            name, colour = classification.get("name"), classification.get("colorRGB")
        else:
            name, colour = classification, None
        name = str(name or properties.get("name") or "Unclassified")
        try:
            parts = _as_polygons(shape(feature["geometry"]))
        except Exception:
            continue
        if not parts:
            continue
        entry = grouped.setdefault(name, {"parts": [], "color": None})
        entry["parts"].extend(parts)
        if entry["color"] is None and colour is not None:
            entry["color"] = int(colour)
    return grouped


def _annotation_entries(path: str) -> dict[str, dict]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        stat = os.stat(path)
        return _read_annotation(path, stat.st_mtime, stat.st_size)
    except (OSError, ValueError):
        return {}


def load_classes(path: str) -> dict[str, list]:
    """Polygons grouped by class, or an empty mapping when the file is unusable."""
    return {name: entry["parts"] for name, entry in _annotation_entries(path).items()}


def load_class_colors(path: str) -> dict[str, int | None]:
    """The colour each class was drawn in. ``None`` where the file carried none."""
    return {name: entry.get("color") for name, entry in _annotation_entries(path).items()}


def _hex_from_qupath(value: int) -> str:
    return f"#{int(value) & 0xFFFFFF:06x}"


def resolve_class_colours(names, source_colours: dict) -> dict[str, int]:
    """A colour for every class: the one it was drawn in, or a distinct new one.

    Hand-drawn exports frequently leave ``colorRGB`` empty on every class but
    one, so writing the file back as-is would give most of the annotation no
    colour at all. The classes that do carry one keep it exactly, and are held
    out of the pool, so the colours invented for the rest stay far from them and
    from each other.
    """
    from spacemap.tissue_annotation_utils import assign_distinct_colors, qupath_color_from_hex

    names = [str(name) for name in names]
    existing = {
        name: _hex_from_qupath(colour)
        for name, colour in source_colours.items() if colour is not None
    }
    palette = assign_distinct_colors(names, existing=existing)
    resolved: dict[str, int] = {}
    for name in names:
        colour = source_colours.get(name)
        resolved[name] = (
            int(colour) if colour is not None
            else int(qupath_color_from_hex(palette[name]))
        )
    return resolved


def annotation_candidates(project_root: Path, sample_id: str) -> tuple[str, list[str]]:
    """A folder of hand-drawn annotations and the files in it, sample first."""
    for folder in (project_root / "data" / "annotations", project_root / "annotations"):
        if folder.is_dir():
            names = sorted(name for name in os.listdir(folder) if name.lower().endswith(".geojson"))
            preferred = [name for name in names if name.startswith(str(sample_id))]
            return str(folder), preferred + [name for name in names if name not in preferred]
    return "", []


def _polygons_of(path: str, class_names: tuple[str, ...] | None = None):
    """Every polygon in one GeoJSON, optionally restricted to some classes."""
    from shapely.geometry import shape

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    features = data.get("features", []) if isinstance(data, dict) else data
    geometries = []
    for feature in features:
        if class_names is not None:
            classification = (feature.get("properties") or {}).get("classification") or {}
            name = classification.get("name") if isinstance(classification, dict) else classification
            if name not in class_names:
                continue
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            continue
        if not geometry.is_empty and geometry.geom_type in ("Polygon", "MultiPolygon"):
            geometries.append(geometry)
    return geometries


def _load_source_geometry(sample_id: str, source: dict | None = None):
    """The tissue to split, the other classes to carry with it, and where it came from.

    Re-running has to start from the same tissue every time. Reading the
    annotation alone would not do that: this step has already written its own
    result there, so a second run would filter an already-filtered mask and a
    loosened size threshold could never bring a piece back. The review layer
    holds both what was kept and what was dropped, so when it exists it restores
    exactly the tissue the previous run was given.
    """
    from shapely.ops import unary_union

    source = source or {}
    custom_path = str(source.get("path") or "")
    if custom_path:
        classes = load_classes(custom_path)
        if not classes:
            raise ValueError(f"'{os.path.basename(custom_path)}' holds no usable polygons.")
        chosen = str(source.get("class_name") or "")
        if chosen not in classes:
            raise ValueError(
                f"'{chosen}' is not one of the classes in {os.path.basename(custom_path)}: "
                f"{', '.join(sorted(classes))}."
            )
        others = {name: parts for name, parts in classes.items() if name != chosen}
        # No thinning here. The outline that comes in is the one that goes out:
        # the neck search runs on a raster of its own making, so a detailed
        # outline no longer costs anything, and simplifying it would hand back a
        # coarser annotation than the one that was drawn.
        return unary_union(classes[chosen]), others, custom_path

    review_path = geojson_path(sample_id, REVIEW_SUFFIX)
    if os.path.isfile(review_path):
        geometries = _polygons_of(review_path, (KEPT_CLASS, REJECTED_CLASS))
        if geometries:
            return unary_union(geometries), {}, review_path

    path = geojson_path(sample_id, SOURCE_SUFFIX)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"'{os.path.basename(path)}' not found for '{sample_id}'. Run step 1 and step 2 first."
        )
    geometries = _polygons_of(path)
    if not geometries:
        raise ValueError(f"'{os.path.basename(path)}' contains no usable polygons.")
    return unary_union(geometries), {}, path


def _run_single(sample_id: str, config: dict, source: dict | None = None) -> dict:
    """Split, measure and filter one image's tissue, then write the GeoJSON."""
    from shapely.ops import unary_union

    from spacemap.tissue_annotation_utils import (
        qupath_color_from_hex,
        write_feature_collection,
    )
    from spacemap.tissue_continuity import analyse_tissue_pieces, px2_to_mm2

    source = source or {}
    geometry, others, source_path = _load_source_geometry(sample_id, source)
    result = analyse_tissue_pieces(geometry, config)

    features = []
    for record in result["kept"]:
        features.append(_feature(record, KEPT_CLASS, KEPT_COLOR, qupath_color_from_hex))
    for record in result["rejected"]:
        features.append(_feature(record, REJECTED_CLASS, REJECTED_COLOR, qupath_color_from_hex))
    # Markers make the detection checkable: open the file in QuPath and every
    # neck is drawn where it was found, labelled with the width measured there.
    for index, neck in enumerate(result["necks"], start=1):
        features.append(_neck_feature(neck, index, qupath_color_from_hex))

    output_path = geojson_path(sample_id, OUTPUT_SUFFIX)
    write_feature_collection(features, output_path)

    # The annotation itself keeps only what survived, so the next step and the
    # tabs after it read the same single file they have been reading all along.
    kept_class = str(source.get("class_name") or KEPT_CLASS)
    custom = bool(source.get("path"))
    microns_per_pixel = float(config.get("microns_per_pixel", 0.5083))
    source_colours = resolve_class_colours(
        [kept_class] + sorted(others), load_class_colors(str(source.get("path") or "")),
    )
    kept_union = unary_union([record["geom"] for record in result["kept"]]) if result["kept"] else None

    if custom:
        # One feature per class, the way the file arrived. A hand-drawn class is
        # a single annotation that happens to have many parts, and exploding it
        # into thousands of separate objects would change what the file is, not
        # just where its edges fall.
        annotation_features = [
            _union_feature(kept_union, kept_class, microns_per_pixel,
                           source_colours, qupath_color_from_hex)
        ] if kept_union is not None else []
    else:
        annotation_features = [
            _feature(record, kept_class, KEPT_COLOR, qupath_color_from_hex)
            for record in result["kept"]
        ]

    # Every other class in the source file is cut to what is left of the tissue.
    # One intersection does both halves of that: a polygon inside a piece that
    # was dropped disappears, and a polygon lying across a neck that was cut is
    # parted along the same line. Anything else would let the contents of the
    # annotation disagree with its container.
    carried: list[dict[str, Any]] = []
    if others and bool(source.get("apply_to_others", True)):
        for name, parts in sorted(others.items()):
            whole = unary_union(parts)
            # Nothing kept means nothing survives inside it either.
            clipped = whole.intersection(kept_union) if kept_union is not None else whole.difference(whole)
            pieces = _as_polygon_list(clipped)
            if pieces:
                annotation_features.append(
                    _union_feature(clipped, name, microns_per_pixel,
                                   source_colours, qupath_color_from_hex)
                )
            carried.append({
                "class": name,
                "before_mm2": round(px2_to_mm2(whole.area, microns_per_pixel), 5),
                "after_mm2": round(px2_to_mm2(clipped.area, microns_per_pixel), 5),
                "parts": len(pieces),
            })

    annotation_path = geojson_path(sample_id, SOURCE_SUFFIX)
    write_feature_collection(annotation_features, annotation_path)

    result["output_path"] = output_path
    result["annotation_path"] = annotation_path
    result["source_path"] = source_path
    result["carried_classes"] = carried
    result["kept_class"] = kept_class
    return result


def _as_polygon_list(geometry) -> list:
    from spacemap.tissue_continuity import _as_polygons

    return [] if geometry is None or geometry.is_empty else _as_polygons(geometry)


def _union_feature(geometry, class_name: str, microns_per_pixel: float,
                   source_colours: dict, color_fn) -> dict:
    """One class, kept whole, cut to the tissue that survived."""
    from shapely.geometry import mapping
    from shapely.ops import unary_union

    from spacemap.tissue_annotation_utils import deterministic_class_hex_color
    from spacemap.tissue_continuity import _as_polygons, px2_to_mm2

    colour = source_colours.get(class_name)
    if colour is None:
        colour = int(color_fn(deterministic_class_hex_color(str(class_name))))

    # Polygons only. Intersecting against the kept tissue leaves the odd point
    # and line where boundaries graze, and a hand-drawn class often arrives as a
    # GeometryCollection carrying a few of those already. Writing them back would
    # hand the next step an annotation with no area in it.
    parts = _as_polygons(geometry)
    cleaned = unary_union(parts) if parts else geometry
    return {
        "type": "Feature",
        "geometry": mapping(cleaned),
        "properties": {
            "objectType": "annotation",
            "classification": {"name": class_name, "colorRGB": colour},
            "name": class_name,
            "isLocked": False,
            "area_mm2": round(float(px2_to_mm2(cleaned.area, microns_per_pixel)), 6),
            "parts": int(len(parts)),
            "cropped_to_kept_tissue": True,
        },
    }


def _feature(record: dict, class_name: str, hex_color: str, color_fn) -> dict:
    """One GeoJSON feature carrying the piece measurements as properties."""
    from shapely.geometry import mapping

    properties = {
        "objectType": "annotation",
        "classification": {"name": class_name, "colorRGB": int(color_fn(hex_color))},
        "name": class_name,
        "isLocked": False,
        "piece_id": int(record["piece_id"]),
        "area_mm2": round(float(record["area_mm2"]), 6),
        "inscribed_diameter_um": round(float(record["inscribed_d_um"]), 1),
        "circularity": round(float(record["circularity"]), 4),
    }
    if record.get("reason"):
        properties["rejected_because"] = str(record["reason"])
    return {
        "type": "Feature",
        "geometry": mapping(record["geom"]),
        "properties": properties,
    }


def _neck_feature(neck: dict, index: int, color_fn) -> dict:
    """A small marker drawn across the bridge, carrying its measurements."""
    from shapely.geometry import mapping

    sides = neck.get("side_areas_mm2", [0.0, 0.0])
    return {
        "type": "Feature",
        "geometry": mapping(neck["marker"]),
        "properties": {
            "objectType": "annotation",
            "classification": {"name": NECK_CLASS, "colorRGB": int(color_fn(NECK_COLOR))},
            "name": NECK_CLASS,
            "isLocked": False,
            "neck_id": int(index),
            "source_piece": int(neck.get("source_piece", 0)),
            "bridge_width_um": round(float(neck["bridge_width_um"]), 1),
            "n_strands": int(neck.get("n_strands", 1)),
            "larger_side_mm2": round(float(sides[0]), 6),
            "smaller_side_mm2": round(float(sides[1]), 6) if len(sides) > 1 else 0.0,
        },
    }


def _results_table(result: dict):
    import pandas as pd

    rows = []
    for status, records in (("kept", result["kept"]), ("rejected", result["rejected"])):
        for record in records:
            rows.append({
                "piece": record["piece_id"],
                "status": status,
                "area (mm²)": round(record["area_mm2"], 5),
                "widest circle (µm)": round(record["inscribed_d_um"], 1),
                "effective radius (µm)": round(record["effective_r_um"], 1),
                "circularity": round(record["circularity"], 3),
                "holes": record["n_holes"],
                "reason": record.get("reason", ""),
            })
    return pd.DataFrame(rows)


def _necks_table(result: dict):
    import pandas as pd

    return pd.DataFrame([
        {
            "piece": neck["source_piece"],
            "piece area (mm²)": round(neck["area_mm2"], 5),
            "bridge width (µm)": round(neck["bridge_width_um"], 1),
            "strands": neck["n_strands"],
            "parted": "yes" if neck["separated"] else "no",
            "larger side (mm²)": round(neck["side_areas_mm2"][0], 5),
            "smaller side (mm²)": round(neck["side_areas_mm2"][1], 5),
            "smaller side (% of piece)": round(100.0 * neck["smaller_side_fraction"], 1),
            "x": round(neck["x"]),
            "y": round(neck["y"]),
        }
        for neck in result["necks"]
    ])


# ==========================================================================
# UI
# ==========================================================================

def render(project_root: Path, sample_id: str, batch_samples: list[str]) -> None:
    st.markdown("### Step 2b - Tissue pieces, necks and size filters")
    st.caption(
        "Splits the reviewed valid tissue into its separate scraps, measures each one, reports narrow "
        "bridges, and drops pieces too small to support the analysis radius. Everything is decided from "
        "the tissue polygon alone - never from how many cells happen to sit there."
    )

    from spacemap.tissue_continuity import DEFAULT_PIECE_CONFIG

    tissue_config = st.session_state.get("annot_ta_tissue_config", {}) or {}
    microns_per_pixel = float(tissue_config.get("microns_per_pixel", DEFAULT_PIECE_CONFIG["microns_per_pixel"]))

    config = {**DEFAULT_PIECE_CONFIG, **(st.session_state.get(CONFIG_KEY) or {})}
    config.setdefault("microns_per_pixel", microns_per_pixel)

    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown("#### Source")
        use_custom = st.checkbox(
            "Split a different GeoJSON instead of this project's annotation",
            key="annot_ta_s2b_use_custom",
            help=(
                "Off: splits the annotation steps 1 and 2 have been building and writes the pieces it "
                "keeps back into it. On: reads any GeoJSON, such as a hand-drawn annotation holding "
                "several classes, and lets you choose which class is the tissue."
            ),
        )
        source: dict[str, Any] = {}
        if not use_custom:
            st.caption(
                f"Splits `{sample_id}_{SOURCE_SUFFIX}.geojson`. The file is re-read from disk each "
                "run, so a QuPath correction is picked up."
            )
        else:
            default_folder, names = annotation_candidates(project_root, sample_id)
            folder = st.text_input(
                "Folder", value=st.session_state.get("annot_ta_s2b_folder", default_folder),
                key="annot_ta_s2b_folder",
                help="Any folder holding GeoJSON annotations.",
            )
            if folder != default_folder and os.path.isdir(folder):
                names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".geojson"))
                preferred = [n for n in names if n.startswith(str(sample_id))]
                names = preferred + [n for n in names if n not in preferred]
            if not names:
                st.warning(f"No GeoJSON files in `{folder}`.")
                return
            filename = st.selectbox("GeoJSON file", names, key="annot_ta_s2b_file")
            custom_path = os.path.join(folder, filename)
            classes = load_classes(custom_path)
            if not classes:
                st.warning(f"`{filename}` holds no usable polygons.")
                return
            summary = ", ".join(f"{name} ({len(parts)})" for name, parts in sorted(classes.items()))
            st.caption(f"Classes in this file: {summary}")
            class_name = st.selectbox(
                "Polygon class to split", sorted(classes), key="annot_ta_s2b_class",
                help="The containing class, the one the size filters and neck cuts apply to.",
            )
            apply_to_others = st.checkbox(
                "Apply the same edits to the other classes", value=True,
                key="annot_ta_s2b_apply_others",
                help=(
                    "Cuts every other class to whatever is left of the chosen one. A polygon inside a "
                    "piece that gets dropped disappears with it, and a polygon lying across a neck that "
                    "gets cut is parted along the same line, so the contents never disagree with their "
                    "container."
                ),
            )
            source = {"path": custom_path, "class_name": class_name,
                      "apply_to_others": bool(apply_to_others)}
            st.session_state["_annot_ta_s2b_preview_source"] = source

        st.markdown("#### Parameters")
        updated_config, submitted = render_parameter_form(
            form_key="annot_ta_s2b_form",
            spec=_parameter_spec(microns_per_pixel),
            config=config,
            submit_label="Save tissue piece settings",
        )
        if submitted:
            save_config(CONFIG_KEY, updated_config)
            st.success("Tissue piece settings saved to the project configuration.")
            config = updated_config

        area_px2 = float(config.get("min_area_mm2", 0.0)) * 1e6 / max(float(config.get("microns_per_pixel", 1.0)), 1e-9) ** 2
        st.caption(
            f"Current area threshold: **{config.get('min_area_mm2', 0):g} mm²** "
            f"= {area_px2:,.0f} pixels² at {config.get('microns_per_pixel', 0):g} µm/pixel."
        )

        st.markdown("#### Run")
        run_column, batch_column = st.columns(2)
        run_single = run_column.button(
            "Build tissue pieces (this image)", type="primary", width="stretch",
            key="annot_ta_s2b_run",
        )
        run_batch = batch_column.button(
            f"Run batch ({len(batch_samples)} images)", width="stretch",
            key="annot_ta_s2b_run_batch", disabled=not batch_samples,
            help="Applies the same settings to every selected image, each using its own tissue GeoJSON.",
        )

        if run_single or run_batch:
            targets = batch_samples if run_batch else [sample_id]
            progress_bar = st.progress(0.0)
            status = st.empty()
            started = time.time()
            record_process_event(
                TAB_CONTEXT, "Tissue piece analysis started",
                details={"samples": len(targets), "source": SOURCE_SUFFIX},
            )
            completed = 0
            for index, target in enumerate(targets):
                try:
                    status.write(f"Processing {target} ({index + 1}/{len(targets)})...")
                    result = _run_single(target, config, source)
                    runtime_store(target)["tissue_pieces"] = result
                    log_annotation(
                        f"{target}: {len(result['kept'])} pieces kept, "
                        f"{len(result['rejected'])} rejected, {len(result['necks'])} necks",
                        subtab="s2b",
                    )
                    completed += 1
                except Exception as error:
                    st.error(f"{target}: {error}")
                    log_annotation(f"{target}: tissue piece analysis failed - {error}", subtab="s2b")
                progress_bar.progress((index + 1) / max(len(targets), 1))
            status.write(
                f"Finished {completed}/{len(targets)} image(s) in {time.time() - started:.1f}s."
            )

    with right:
        st.markdown("#### Result")
        output_path = geojson_path(sample_id, OUTPUT_SUFFIX)
        annotation_path = geojson_path(sample_id, SOURCE_SUFFIX)
        describe_output(annotation_path, "Valid tissue (the annotation)")
        describe_output(output_path, "Pieces, rejects and necks (review layer)")

        image_path = resolve_image_path(sample_id)
        result = runtime_store(sample_id).get("tissue_pieces")
        if result is None:
            # Draw the class that is about to be split, so the source can be
            # checked against the image before anything is cut.
            chosen = st.session_state.get("_annot_ta_s2b_preview_source") or {}
            if image_path and chosen.get("path"):
                try:
                    image, colors = overlay_polygons_on_thumbnail(
                        image_path, [chosen["path"]],
                        class_filter=(chosen["class_name"],), fill_alpha=60,
                    )
                    st.image(
                        image, width="stretch",
                        caption=f"{chosen['class_name']} from {os.path.basename(chosen['path'])}",
                    )
                    render_class_legend(colors)
                except Exception as error:
                    st.warning(f"Could not draw the selected class: {error}")
            st.info("Run the step to see the pieces, their measurements and any necks.")
            return

        kept_area = sum(record["area_mm2"] for record in result["kept"])
        rejected_area = sum(record["area_mm2"] for record in result["rejected"])
        total_area = kept_area + rejected_area
        metrics = st.columns(4)
        metrics[0].metric("Pieces kept", len(result["kept"]))
        metrics[1].metric("Rejected", len(result["rejected"]))
        metrics[2].metric("Necks found", len(result["necks"]))
        metrics[3].metric(
            "Tissue kept",
            f"{(100.0 * kept_area / total_area) if total_area else 0.0:.1f}%",
            help=f"{kept_area:.3f} mm² kept, {rejected_area:.4f} mm² dropped.",
        )

        carried = result.get("carried_classes") or []
        views = ["Pieces and necks"] + (["Edited annotation"] if carried else [])
        view = views[0]
        if len(views) > 1:
            view = st.radio("Show", views, horizontal=True, key="annot_ta_s2b_preview_layer")
        drawn_path = output_path if view == "Pieces and necks" else annotation_path
        caption = (
            "Green: kept. Grey: rejected. Pink: a detected neck."
            if view == "Pieces and necks"
            else "Every class in the source, cut to the tissue that survived."
        )
        if image_path and os.path.isfile(drawn_path):
            try:
                image, colors = overlay_polygons_on_thumbnail(
                    image_path, [drawn_path],
                    class_colors={
                        KEPT_CLASS: KEPT_COLOR,
                        REJECTED_CLASS: REJECTED_COLOR,
                        NECK_CLASS: NECK_COLOR,
                    },
                    fill_alpha=70,
                )
                st.image(image, width="stretch", caption=caption)
                render_class_legend(colors)
            except Exception as error:
                st.warning(f"Could not render the piece overlay: {error}")

        if carried:
            import pandas as pd

            st.caption(
                f"Carried through and cut to the kept tissue, written into "
                f"`{os.path.basename(annotation_path)}` alongside the "
                f"{result.get('kept_class', KEPT_CLASS)} pieces."
            )
            st.dataframe(
                pd.DataFrame(carried).rename(columns={
                    "class": "class", "before_mm2": "before (mm²)",
                    "after_mm2": "after (mm²)", "parts": "parts",
                }),
                hide_index=True, width="stretch",
            )

        table = _results_table(result)
        if not table.empty:
            st.markdown("**Pieces**")
            st.dataframe(table, width="stretch", hide_index=True)
            save_table(table.set_index("piece"), sample_id, "tissue_pieces")

        necks = _necks_table(result)
        if not necks.empty:
            st.markdown("**Necks**")
            st.caption(
                "A neck is a bridge narrow enough to separate two sections that are each real tissue, "
                "so a bleb hanging off a mass is not reported. Each width is measured rather than "
                "assumed: the piece is opened at a range of widths and the narrowest one that still "
                "parts those two sections is the bridge's own width. A bridge only a few micrometres "
                "wide was more likely created by mask smoothing than observed - check it against the "
                "H&E before trusting it. Two sections often hang together by more than one strand - a "
                "slit that stays joined at both ends - and the width shown is the one at which all of "
                "them give way."
            )
            st.dataframe(necks, width="stretch", hide_index=True)
            save_table(necks, sample_id, "tissue_necks")
        elif result["settings"].get("detect_necks", True):
            st.success("No necks narrower than the accepted width were found in this image.")
