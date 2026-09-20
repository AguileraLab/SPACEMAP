"""Subtab 2 - remove blurry regions, flat bright glare and filamentous lint."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from spacemap_gui.process_log import record_process_event

from . import figures
from .common import (
    TAB_CONTEXT,
    describe_output,
    geojson_path,
    intermediate_file,
    log_annotation,
    open_reader,
    output_dir_for,
    overlay_polygons_on_thumbnail,
    qupath_hint,
    prune_widget_selection,
    render_class_legend,
    render_image_preview,
    render_parameter_form,
    resolve_image_path,
    runtime_store,
    save_config,
    save_project_settings,
    save_table,
    show_cached_figure,
    tidy_output_directory,
)

CONFIG_KEY = "annot_ta_artifact_config"
MARKERS_KEY = "annot_ta_artifact_markers"

# One GeoJSON per artifact class, plus their union and the cleaned tissue, so each
# can be reviewed or corrected independently in QuPath. A None tolerance means
# "use the configured boundary simplification"; the colours match the diagnostic
# figure so a layer looks the same in QuPath as it does in the plot.
#
# The last field names the region set to export feature-by-feature. Those layers
# carry one annotation per detected region, each with its own measurements and
# the markers that produced it, because a single merged shape cannot be accepted
# in one place and rejected in another. Valid tissue and the merged artifact
# union stay as single shapes: they are masks, not findings.
ARTIFACT_EXPORTS = (
    ("valid_tissue_mask", "valid_tissue", "Valid_Tissue", "#2ecc71", None, None),
    ("blur_mask", "detected_blur", "Detected_Blur", "#ffff00", None, "blur"),
    ("bright_mask", "detected_bright", "Detected_Bright", "#ff00ff", None, "bright"),
    ("lint_mask", "detected_lint", "Detected_Lint", "#00ffff", 1.0, "lint"),
    ("artifact_mask", "detected_artifacts", "Detected_Artifacts", "#ff5252", None, None),
)

# The stored value is the canonical one the detector understands; the label is
# what the form shows. Keeping the two apart means the saved project config is
# still readable by the notebook and by anyone calling the science layer directly.
BLUR_RULE_CHOICES = {
    "Either test flags it (recommended)": "either",
    "Both tests must agree (fewest false positives)": "both",
    "Overall sharpness only": "pooled_only",
    "Per-marker test only": "markers_only",
}

BLUR_REFERENCE_CHOICES = {
    "Both references must agree (recommended)": "he_and_self",
    "Either reference (catches more)": "he_or_self",
    "The H&E at the same spot only": "he",
    "The sharp tissue in this image only": "self",
}

BLUR_MODE_CHOICES = {
    "Find the cut-off automatically (recommended)": "auto",
    "Use the fixed cut-off below": "manual",
}

# The whole simple control is this one dial. Each step multiplies the cut-off the
# detector worked out for itself, so the two refinement buttons move by one step
# and re-run; nothing else has to be understood to use the tab.
# Each level carries two multipliers. The first moves the cut-off the detector
# works out for itself. The second moves how separated the sharp and soft parts
# of the tissue must be before any of it is called blurry at all, because on a
# slide that has no distinct out-of-focus region that second test is what
# decides the answer, and a control that could not reach it would appear to do
# nothing however far it was moved.
SENSITIVITY_LEVELS = {
    1: ("Very cautious", 0.50, 1.50),
    2: ("Cautious", 0.63, 1.35),
    3: ("Slightly cautious", 0.79, 1.20),
    4: ("Balanced", 1.00, 1.00),
    5: ("Slightly keen", 1.26, 0.90),
    6: ("Keen", 1.59, 0.825),
    7: ("Very keen", 2.00, 0.75),
}
DEFAULT_SENSITIVITY_LEVEL = 4


def sensitivity_level(config: dict) -> int:
    level = int(config.get("blur_sensitivity_level", DEFAULT_SENSITIVITY_LEVEL) or DEFAULT_SENSITIVITY_LEVEL)
    return max(1, min(7, level))


def sensitivity_scale(config: dict) -> float:
    return SENSITIVITY_LEVELS[sensitivity_level(config)][1]


def run_config_for(config: dict) -> dict:
    """The config a run uses: the saved settings with the simple control applied."""
    _, cut_scale, separation_scale = SENSITIVITY_LEVELS[sensitivity_level(config)]
    base_separation = float(config.get("blur_min_separation", 4.0))
    base_share = float(config.get("max_blur_fraction", 0.30))
    return {
        **config,
        "blur_sensitivity": cut_scale,
        "blur_min_separation": base_separation * separation_scale,
        # The share guard deliberately does not move with the slider. It is the
        # backstop that keeps a keen setting from calling a whole slide blurry,
        # and a control that could switch it off would make the other settings
        # meaningless at the top of the range.
        "max_blur_fraction": base_share,
    }


NOTE = (
    "This affects the overall-sharpness test only: the per-marker grid is built during the image read in step 1 and uses the standard scale."
)

PARAMETER_SPEC = [
    {"type": "header", "label": "Out-of-focus regions",
     "help": (
         "Finds parts of the tissue where the CODEX scan came out blurry, so they are not analysed as "
         "if they were sharp. Two separate tests are run. The first asks whether the area as a whole is "
         "less sharp than it should be. The second asks each marker on its own whether it lost focus "
         "here, which is the only way to catch a cycle that drifted while the rest of the panel stayed "
         "sharp. Both are only applied where the markers actually have signal, so tissue that simply "
         "did not stain is never mistaken for blur. Start with a preset, run once, and look at the "
         "yellow overlay and the sharpness histogram."
     )},
    {
        "key": "blur_threshold_mode", "label": "How the cut-off is chosen", "type": "select",
        "options": list(BLUR_MODE_CHOICES),
        "help": (
            "Automatic looks at the spread of sharpness across the tissue and only cuts when it finds "
            "two separate populations, a sharp one and a soft one. That is what lets it answer 'nothing "
            "here is blurry' on a slide that is entirely in focus. A fixed cut-off cannot: sharpness "
            "varies several-fold across normal tissue, so any fixed number removes its lower tail."
        ),
    },
    {
        "key": "blur_min_separation", "label": "Separation needed before anything is cut", "type": "number",
        "min": 1.5, "max": 20.0, "step": 0.5,
        "help": (
            "How far apart the soft and sharp populations must be before blur is reported at all. 4 means "
            "the soft group has to be four times less sharp than the sharp group. Ordinary tissue texture "
            "spans about two to three times, so anything below 3 starts cutting into normal variation. "
            "Automatic mode only."
        ),
    },
    {
        "key": "max_blur_fraction", "label": "Never call more than this share blurry", "type": "number",
        "min": 0.05, "max": 1.0, "step": 0.05,
        "help": (
            "A safety valve. If the split would put more than this share of the tissue in the soft group, "
            "the result is treated as no detection rather than as a slide that is mostly out of focus. "
            "Automatic mode only."
        ),
    },
    {
        "key": "blur_ratio_threshold", "label": "Fixed cut-off (manual mode)", "type": "number",
        "min": 0.0, "max": 2.0, "step": 0.05,
        "help": (
            "Only used when the cut-off above is set to fixed. 1.0 means 'as sharp as this image gets'; "
            "0.35 removes anything less than 35% as sharp as the sharp tissue on the same slide. It is "
            "also the cut used to decide whether one individual marker has lost focus, in both modes."
        ),
    },
    {
        "key": "blur_evidence_rule", "label": "How an area gets flagged", "type": "select",
        "options": list(BLUR_RULE_CHOICES),
        "help": (
            "Which of the two tests has to fire. 'Either' catches the most, including single blurred "
            "cycles. 'Both must agree' is the safest if valid tissue is being taken. The per-marker test "
            "needs at least four markers selected; below that only the overall test runs."
        ),
    },
    {
        "key": "marker_consensus_fraction", "label": "Share of markers that must be soft", "type": "number",
        "min": 0.0, "max": 1.0, "step": 0.05,
        "help": (
            "For the per-marker test. 0.5 means half the markers that have signal in that spot must have "
            "individually lost focus. LOWER it to catch defocus that hit only a few cycles; raise it to "
            "demand that most of the panel agrees."
        ),
    },
    {
        "key": "blur_reference_mode", "label": "What overall sharpness is compared against", "type": "select",
        "options": list(BLUR_REFERENCE_CHOICES),
        "help": (
            "The first test needs something to call sharp. The H&E of the same spot shows what detail is "
            "really there; the sharp tissue elsewhere in the same image shows what this scan can achieve. "
            "Requiring both keeps featureless-but-focused tissue. If the image has no usable H&E layers, "
            "this falls back to the image on its own automatically."
        ),
    },
    {
        "key": "require_codex_signal", "label": "Only judge tissue that has marker signal", "type": "checkbox",
        "help": (
            "Leave this on. Antibodies only stain their targets, so collagen, muscle and fat can be "
            "perfectly in focus and still carry almost no CODEX signal. Without this gate those areas "
            "score as unsharp and get deleted as blur, which is the most common cause of valid tissue "
            "going missing."
        ),
    },
    {"key": "codex_signal_z", "label": "How much signal counts as stained", "type": "number",
     "min": 0.0, "max": 20.0, "step": 0.5,
     "help": "How far above background a marker must rise for a spot to be judged at all. Lower it if stained tissue is being skipped."},
    {"key": "expand_microns", "label": "Safety margin (µm)", "type": "number", "min": 0.0, "max": 500.0, "step": 1.0,
     "help": "Grows each blurry region outwards a little before it is removed, so the fuzzy edge goes too. This is added after the size filter below."},
    {"key": "min_area_microns2", "label": "Ignore blur smaller than (um^2)", "type": "number", "min": 0.0, "max": 1e6, "step": 100.0,
     "help": "Blurry patches smaller than this are left alone. 2000 um^2 is roughly a 45 x 45 um patch. Raise it to keep only large out-of-focus areas."},

    {"type": "header", "label": "Out-of-focus regions - fine tuning",
     "help": "Sensible for most slides as they are. All lengths are in microns, so they mean the same thing at any masking downsample."},
    {"key": "sharp_reference_percentile", "label": "What counts as sharp (percentile)", "type": "number",
     "min": 50.0, "max": 99.0, "step": 1.0,
     "help": (
         "Which part of the image is taken as the sharp benchmark. 75 means the sharpest quarter. Going "
         "much higher anchors on strong edges such as the tissue border, which makes ordinary tissue look "
         "soft by comparison. Lower it only if you know most of the slide is blurry."
     )},
    {"key": "he_structure_percentile", "label": "Skip the plainest tissue (percentile)", "type": "number",
     "min": 0.0, "max": 60.0, "step": 1.0,
     "help": "The blankest areas of the H&E are skipped, because comparing sharpness where there is no detail is meaningless. 15 skips the plainest 15%."},
    {"key": "marker_signal_z", "label": "Per-marker signal cutoff", "type": "number", "min": 0.0, "max": 20.0, "step": 0.5,
     "help": "How much signal one marker needs in a spot before its focus is judged there. Rarely needs changing."},
    {"key": "marker_consensus_min_markers", "label": "Markers needed for the per-marker test", "type": "int",
     "min": 2, "max": 60, "step": 1,
     "help": "Below this many markers with signal, the per-marker test is not trusted and only the overall test is used."},
    {"key": "focus_detail_um", "label": "Detail scale (µm)", "type": "number", "min": 0.5, "max": 20.0, "step": 0.5,
     "help": "How fine the detail is that counts as sharp. Larger looks at coarser structure. Rarely needs changing." + NOTE},
    {"key": "focus_window_um", "label": "Area examined at once (µm)", "type": "number", "min": 5.0, "max": 200.0, "step": 1.0,
     "help": "How big a patch is judged as a unit. Larger gives smoother, blockier blur regions."},
    {"key": "noise_smoothing_um", "label": "Noise smoothing before (µm)", "type": "number", "min": 0.0, "max": 10.0, "step": 0.5,
     "help": "Removes speckle noise before sharpness is measured. Raise slightly for very noisy images." + NOTE},
    {"key": "region_smoothing_um", "label": "Smoothing after (µm)", "type": "number", "min": 0.0, "max": 30.0, "step": 0.5,
     "help": "Merges the result into solid patches instead of scattered dots. Raise if blur comes out speckled."},
    {"key": "signal_support_um", "label": "Bridge gaps in signal (µm)", "type": "number", "min": 0.0, "max": 200.0, "step": 1.0,
     "help": "Closes small unstained gaps so a judged region is not punched full of holes."},
    {"key": "fill_holes_um", "label": "Fill small holes (µm)", "type": "number", "min": 0.0, "max": 100.0, "step": 1.0,
     "help": "Closes small gaps inside a blurry patch so it becomes one region."},
    {"key": "remove_specks_um", "label": "Remove small specks (µm)", "type": "number", "min": 0.0, "max": 100.0, "step": 1.0,
     "help": "Deletes tiny isolated blur spots. Raise if you get lots of small dots."},

    {"type": "header", "label": "Bright glare and debris",
     "help": (
         "Finds spots inside the tissue that are very bright but completely smooth, which is what "
         "glare and debris look like. Real tissue is bright *and* detailed, so it is kept."
     )},
    {"key": "detect_bright_artifacts", "label": "Look for bright glare", "type": "checkbox",
     "help": "Turn off if you only want blur and lint removed."},
    {
        "key": "use_breadth_signal", "label": "Also flag things bright in many markers at once", "type": "checkbox",
        "help": (
            "Off by default. Debris tends to glow in every channel at once, unlike a real cell. "
            "Useful when you have obvious autofluorescent junk, but dense real tissue can trip it, so "
            "check the overlay after turning it on."
        ),
    },
    {"key": "bright_breadth_fraction", "label": "How many markers must be bright", "type": "number", "min": 0.0, "max": 1.0, "step": 0.05,
     "help": "Only used by the option above. 0.5 means half the markers must be bright at the same spot."},
    {"key": "bright_flat_mean_percentile", "label": "How bright it must be", "type": "number", "min": 50.0, "max": 100.0, "step": 0.1,
     "help": "99.5 means 'only the brightest 0.5% of the tissue'. LOWER it to catch more glare; raise it to be safer."},
    {"key": "bright_flat_texture_percentile", "label": "How smooth it must be", "type": "number", "min": 0.0, "max": 50.0, "step": 0.5,
     "help": "The second half of the test: it must also be featureless. Raise it to catch more, lower it if real tissue is being taken."},
    {"key": "bright_fill_holes_um", "label": "Fill small holes (µm)", "type": "number", "min": 0.0, "max": 100.0, "step": 1.0,
     "help": "Closes gaps inside a bright patch."},
    {"key": "bright_remove_specks_um", "label": "Remove small specks (µm)", "type": "number", "min": 0.0, "max": 100.0, "step": 1.0,
     "help": "Deletes tiny isolated bright dots."},
    {"key": "bright_expand_microns", "label": "Safety margin (µm)", "type": "number", "min": 0.0, "max": 200.0, "step": 1.0,
     "help": "Grows each bright region outwards a little before it is removed."},
    {"key": "min_area_microns2_bright", "label": "Ignore glare smaller than (um^2)", "type": "number", "min": 0.0, "max": 1e6, "step": 5.0,
     "help": "Bright patches smaller than this are left alone."},

    {"type": "header", "label": "Lint and fibres",
     "help": (
         "Finds threads of dust and fabric lying on the slide. These are only looked for *outside* the "
         "tissue, so glands, vessels and stroma are never mistaken for a fibre. A fibre is long, thin "
         "and bright; the settings below describe that shape in real-world sizes."
     )},
    {"key": "detect_filamentous_lint", "label": "Look for lint", "type": "checkbox",
     "help": "Turn off if the slide has no fibre contamination."},
    {
        "key": "lint_robust_z_threshold", "label": "How bright a fibre must be", "type": "number",
        "min": 1.0, "max": 100.0, "step": 0.5,
        "help": (
            "How much a spot must stand out in any single marker to be considered. LOWER it to find "
            "faint fibres, at the cost of more false candidates. This is the main knob if lint is missed."
        ),
    },
    {"key": "lint_he_exclusion_margin_um", "label": "Keep this far from tissue (µm)", "type": "number", "min": 0.0, "max": 200.0, "step": 1.0,
     "help": "A fibre must start at least this far outside the tissue. Raise it if fibres near the tissue edge are eating into it."},
    {"key": "lint_min_area_microns2", "label": "Smallest fibre (um^2)", "type": "number", "min": 0.0, "max": 1e6, "step": 5.0,
     "help": "Anything smaller than this is ignored as dust."},
    {"key": "lint_max_area_microns2", "label": "Largest fibre (um^2)", "type": "number", "min": 0.0, "max": 1e8, "step": 1000.0,
     "help": "Anything bigger is treated as a genuine region of signal, not a fibre."},
    {"key": "lint_min_length_microns", "label": "Shortest fibre (µm)", "type": "number", "min": 0.0, "max": 5000.0, "step": 5.0,
     "help": "How long a thread has to be. Raise it to only catch long obvious fibres."},
    {"key": "lint_min_width_microns", "label": "Thinnest fibre (µm)", "type": "number", "min": 0.0, "max": 500.0, "step": 0.5,
     "help": "Raising this is the single most effective way to stop thin scan lines being called lint."},
    {"key": "lint_max_width_microns", "label": "Thickest fibre (µm)", "type": "number", "min": 0.0, "max": 1000.0, "step": 1.0,
     "help": "Anything wider than this is too thick to be a thread."},
    {"key": "lint_min_aspect_ratio", "label": "How long compared to wide", "type": "number", "min": 1.0, "max": 100.0, "step": 0.5,
     "help": "A fibre must be at least this many times longer than it is wide. Raise it to accept only clearly thread-like shapes."},
    {"key": "lint_seam_max_width_microns", "label": "Scan-line width (µm)", "type": "number", "min": 0.0, "max": 200.0, "step": 0.5,
     "help": "Thin horizontal stripes narrower than this are rejected as scanner seams rather than lint."},
    {"key": "lint_seam_min_span_ratio", "label": "Scan-line flatness", "type": "number", "min": 1.0, "max": 100.0, "step": 0.5,
     "help": "How wide-and-flat a stripe must be to count as a scanner seam. Works together with the setting above."},
    {"key": "lint_join_gaps_um", "label": "Join broken threads (µm)", "type": "number", "min": 0.0, "max": 50.0, "step": 0.5,
     "help": "Bridges small breaks so a fibre stays one piece instead of many fragments."},
    {"key": "lint_remove_specks_um", "label": "Remove specks (µm)", "type": "number", "min": 0.0, "max": 50.0, "step": 0.5,
     "help": "Deletes isolated bright dots before shapes are measured."},
    {"key": "lint_expand_microns", "label": "Safety margin (µm)", "type": "number", "min": 0.0, "max": 200.0, "step": 0.5,
     "help": "Grows each accepted fibre outwards a little before it is removed."},

    {"type": "header", "label": "Output",
     "help": "How the saved outlines are drawn, and what is written alongside them for review."},
    {"key": "boundary_simplify_px", "label": "Outline smoothing (px)", "type": "number", "min": 0.0, "max": 20.0, "step": 0.5,
     "help": "Higher gives simpler outlines and smaller files, at the cost of fine detail."},
    {"key": "write_review_manifest", "label": "Write the review manifest", "type": "checkbox",
     "help": (
         "Writes one JSON file listing every detected region with its measurements and the markers that "
         "led to it, for the review pass to read without opening the images. The same information is "
         "also inside each GeoJSON feature."
     )},
]


def _choice_label(value, choices: dict[str, str]) -> str:
    for label, canonical in choices.items():
        if canonical == value:
            return label
    return next(iter(choices))


def _config_for_form(config: dict) -> dict:
    """Swap canonical selector values for the labels the form shows."""
    display = dict(config)
    display["blur_evidence_rule"] = _choice_label(config.get("blur_evidence_rule"), BLUR_RULE_CHOICES)
    display["blur_reference_mode"] = _choice_label(config.get("blur_reference_mode"), BLUR_REFERENCE_CHOICES)
    display["blur_threshold_mode"] = _choice_label(config.get("blur_threshold_mode"), BLUR_MODE_CHOICES)
    return display


def _config_from_form(updated: dict) -> dict:
    """Swap the form's labels back for the values the detector understands."""
    resolved = dict(updated)
    resolved["blur_evidence_rule"] = BLUR_RULE_CHOICES.get(updated.get("blur_evidence_rule"), "either")
    resolved["blur_reference_mode"] = BLUR_REFERENCE_CHOICES.get(updated.get("blur_reference_mode"), "he_and_self")
    resolved["blur_threshold_mode"] = BLUR_MODE_CHOICES.get(updated.get("blur_threshold_mode"), "auto")
    return resolved


def _marker_options(sample_id: str) -> tuple[list[str], list[str], list[str]]:
    """Return (all channel names, default artifact markers, H&E channel names)."""
    from spacemap.tissue_annotation_utils import split_he_and_codex_channels

    image_path = resolve_image_path(sample_id)
    if not image_path:
        return [], [], []
    reader = open_reader(image_path)
    he_indices, codex_indices = split_he_and_codex_channels(reader.channel_names)
    names = list(reader.channel_names)
    return names, [names[i] for i in codex_indices], [names[i] for i in he_indices]


def _load_signal_maps(sample_id: str, tissue_config: dict, marker_names: list[str], status=None) -> dict:
    """Compute (or reuse) the per-pixel signal maps step 2 needs.

    These maps come from the same reader pass as step 1.  They are cached in
    the session runtime keyed by the parameters that affect them, so changing
    only an artifact threshold does not re-read 40 channels.
    """
    from spacemap.tissue_annotation_utils import detect_tissue_mask, split_he_and_codex_channels

    image_path = resolve_image_path(sample_id)
    if not image_path:
        raise FileNotFoundError(f"No image file found for sample '{sample_id}'.")
    reader = open_reader(image_path)
    he_indices, _ = split_he_and_codex_channels(reader.channel_names)
    selected_indices = [reader.channel_names.index(name) for name in marker_names if name in reader.channel_names]
    if not selected_indices:
        raise ValueError("Select at least one marker channel for artifact detection.")

    signature = json.dumps(
        # The version is part of the key so a session that cached maps from an
        # earlier build, before the per-marker focus grids existed, recomputes
        # instead of silently running without them.
        {"version": 2, "config": tissue_config, "markers": sorted(selected_indices), "path": image_path,
         "mtime": os.path.getmtime(image_path)},
        sort_keys=True, default=str,
    )
    store = runtime_store(sample_id)
    cached = store.get("signal_maps")
    if cached and cached.get("signature") == signature:
        return cached["result"]

    def progress(done: int, total: int, name: str) -> None:
        if status is not None:
            status.write(f"{sample_id}: reading channel {done}/{total} ({name})")

    # The artifact detectors need the per-marker projection, robust-Z and focus
    # maps, so this pass does read every selected marker channel.
    result = detect_tissue_mask(
        reader, tissue_config, he_indices=he_indices, codex_indices=selected_indices,
        progress=progress, need_codex_maps=True,
    )
    store["signal_maps"] = {"signature": signature, "result": result}
    return result


def _regions_from_geojson(path: str) -> list[dict]:
    """Read back the region records a previously written layer carries.

    Blood clot is detected in step 1 and its layer already holds the evidence
    for each region, so the manifest can quote it rather than recompute it.
    """
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            collection = json.load(handle)
    except (OSError, ValueError):
        return []
    records = []
    for feature in collection.get("features") or []:
        properties = feature.get("properties") or {}
        block = properties.get("spacemap")
        if not block:
            continue
        records.append({
            "spacemap": block,
            "measurements": properties.get("measurements") or {},
            "feature_id": feature.get("id"),
        })
    return records


def _run_single(sample_id: str, tissue_config: dict, artifact_config: dict, marker_names: list[str], status=None) -> dict:
    """Detect artifacts for one sample and write the per-class GeoJSON layers."""
    from spacemap.tissue_annotation_utils import (
        detect_artifacts,
        load_geojson_mask,
        mask_to_geojson_features,
        qupath_color_from_hex,
        region_features_from_labels,
        write_artifact_manifest,
        write_feature_collection,
    )

    maps = _load_signal_maps(sample_id, tissue_config, marker_names, status=status)
    downsample = int(tissue_config.get("blur_downsample", 2))
    height, width = maps["shape"]

    boundary_path = geojson_path(sample_id, "tissue_boundary")
    if not os.path.isfile(boundary_path):
        raise FileNotFoundError(
            f"Tissue boundary GeoJSON not found for '{sample_id}'. Run step 1 first, or place the file at {boundary_path}."
        )
    # Always reload from disk so a QuPath correction made after step 1 is honoured.
    tissue_mask = load_geojson_mask(boundary_path, (height, width), float(downsample))
    if not tissue_mask.any():
        raise ValueError(f"The tissue boundary GeoJSON for '{sample_id}' rasterized to an empty mask.")

    # The reader returns the marker names in the same order as the rows of the
    # per-marker focus grid, so evidence lines up with names without a lookup.
    grid_marker_names = list(maps.get("codex_channel_names") or marker_names)

    artifacts = detect_artifacts(
        tissue_mask=tissue_mask,
        codex_projection=maps["codex_projection"],
        he_gray=maps["he_gray"],
        codex_robust_max=maps["codex_robust_max"],
        channel_positive_count=maps["channel_positive_count"],
        n_codex_channels=len(marker_names),
        config=artifact_config,
        microns_per_pixel=float(tissue_config.get("microns_per_pixel", 0.5083)),
        downsample=downsample,
        border_px=int(maps.get("border_px", 0)),
        marker_names=grid_marker_names,
        channel_focus_grid=maps.get("channel_focus_grid"),
        channel_signal_grid=maps.get("channel_signal_grid"),
        sample_id=sample_id,
    )

    output_dir = st.session_state.get("annot_ta_output_dir", "")
    os.makedirs(output_dir, exist_ok=True)
    tidy_output_directory(output_dir)
    simplify = float(artifact_config.get("boundary_simplify_px", 2.0))

    written: dict[str, str] = {}
    region_paths: dict[str, str] = {}
    for mask_key, suffix, class_name, hex_color, tolerance, region_key in ARTIFACT_EXPORTS:
        mask = artifacts.get(mask_key)
        path = geojson_path(sample_id, suffix)
        simplify_px = tolerance if tolerance is not None else simplify
        color = int(qupath_color_from_hex(hex_color))
        if region_key is not None:
            features = region_features_from_labels(
                artifacts["region_labels"][region_key], artifacts["regions"][region_key],
                float(downsample), class_name, color_rgb=color, simplify_px=simplify_px,
            )
            region_paths[region_key] = path
        else:
            features = (
                mask_to_geojson_features(
                    mask, float(downsample), class_name, color_rgb=color,
                    simplify_px=simplify_px, merge=True,
                )
                if mask is not None and mask.any()
                else []
            )
        # Always write, even when empty, so a stale file from an earlier run can
        # never be mistaken for the current result.
        write_feature_collection(features, path)
        written[suffix] = path

    manifest_path = None
    if bool(artifact_config.get("write_review_manifest", True)):
        # Blood clot is found in step 1, but a reviewer wants one index per
        # sample rather than one per step, so its regions are folded in here
        # from the layer step 1 wrote.
        manifest_regions = dict(artifacts["regions"])
        clot_path = geojson_path(sample_id, "blood_clot")
        clot_regions = _regions_from_geojson(clot_path)
        if clot_regions:
            manifest_regions["blood_clot"] = clot_regions
            region_paths["blood_clot"] = clot_path
        manifest_path = write_artifact_manifest(
            intermediate_file(sample_id, "artifact_review.json"),
            sample_id=sample_id,
            regions=manifest_regions,
            exports=region_paths,
            stats=artifacts["stats"],
            parameters=artifacts["parameters"],
            marker_names=grid_marker_names,
            image_path=resolve_image_path(sample_id),
        )

    # The label planes have done their job once the layers are written, and they
    # are full int32 images: on a large ROI, keeping three of them in the session
    # for the rest of the run costs hundreds of megabytes for nothing.
    artifacts.pop("region_labels", None)
    artifacts["tissue_mask"] = tissue_mask
    artifacts["maps"] = maps
    artifacts["exports"] = written
    artifacts["manifest_path"] = manifest_path
    artifacts["valid_path"] = written["valid_tissue"]
    artifacts["lint_path"] = written["detected_lint"]
    artifacts["run_token"] = uuid.uuid4().hex
    return artifacts


def _region_table(artifacts: dict, artifact_type: str) -> pd.DataFrame:
    """Flatten one class of detected regions into a reviewable table."""
    rows = []
    for record in artifacts.get("regions", {}).get(artifact_type, []):
        block = record.get("spacemap", {})
        evidence = block.get("evidence", {})
        candidates = block.get("candidate_markers", [])
        row = {
            "region": block.get("region_id", ""),
            "area_um2": round(float(record.get("area_um2", 0.0)), 1),
            "width_um": round(float(record.get("equivalent_diameter_um", 0.0)), 1),
        }
        if artifact_type == "blur":
            row.update({
                "severity": block.get("severity"),
                "confidence": block.get("confidence"),
                "markers_soft": f"{evidence.get('markers_out_of_focus', 0)}/{evidence.get('markers_with_signal', 0)}",
                "candidate_markers": ", ".join(
                    f"{entry['marker']} ({entry['focus_ratio']:.2f})" for entry in candidates[:5]
                ),
            })
        else:
            row.update({
                "candidate_markers": ", ".join(
                    f"{entry['marker']} (z={entry['signal_z']:.1f})" for entry in candidates[:5]
                ),
            })
            if artifact_type == "lint":
                row["length_um"] = evidence.get("length_um")
                row["overlaps_tissue"] = evidence.get("overlaps_tissue")
        rows.append(row)
    return pd.DataFrame(rows)


def render(project_root: Path, sample_id: str, batch_samples: list[str]) -> None:
    st.markdown("### Step 2 - Remove blur, bright artifacts and lint")
    st.caption(
        "Reloads the tissue boundary from disk (so QuPath corrections count), scores out-of-focus tissue, "
        "flat bright glare and external filamentous lint, then writes the clean valid-tissue GeoJSON. "
        "Each detected region is exported as its own annotation carrying the measurements and the marker "
        "evidence behind it, so it can be accepted or rejected one region at a time."
    )

    tissue_config = st.session_state.get("annot_ta_tissue_config", {})
    config = st.session_state.get(CONFIG_KEY, {})
    all_names, codex_defaults, he_names = _marker_options(sample_id)

    left, right = st.columns([1.25, 1.0])

    with left:
        st.markdown("#### Marker channels")
        st.caption(
            "Only the selected markers contribute to the CODEX projection, the robust-Z map and the "
            "per-marker focus grid that seed artifact detection. The three virtual H&E layers are excluded "
            "by default; include them only if you want H&E intensity itself to seed artifacts."
        )
        saved_markers = [name for name in st.session_state.get(MARKERS_KEY, []) if name in all_names]
        if "_annot_ta_s2_markers_pending" in st.session_state:
            st.session_state["annot_ta_s2_markers"] = st.session_state.pop("_annot_ta_s2_markers_pending")
        widget_value = st.session_state.get("annot_ta_s2_markers")
        if isinstance(widget_value, list) and any(name not in all_names for name in widget_value):
            st.session_state["annot_ta_s2_markers"] = [name for name in widget_value if name in all_names]
        selected_markers = st.multiselect(
            "Markers used for artifact detection",
            all_names,
            default=saved_markers or codex_defaults,
            key="annot_ta_s2_markers",
            help=(
                "Channels feeding the artifact detectors. Drop noisy or empty channels to reduce false lint "
                "seeds. Keeping most of the panel makes the per-marker blur test stronger, since it needs "
                "several markers with signal in a spot before it will judge it."
            ),
        )
        marker_columns = st.columns(3)
        if marker_columns[0].button("All CODEX", key="annot_ta_s2_markers_codex", width="stretch"):
            # The widget key wins over `default` once it exists, so a preset has
            # to write the widget key as well as the canonical saved value.
            st.session_state[MARKERS_KEY] = codex_defaults
            st.session_state["_annot_ta_s2_markers_pending"] = codex_defaults
            st.rerun()
        if marker_columns[1].button("All channels", key="annot_ta_s2_markers_all", width="stretch"):
            st.session_state[MARKERS_KEY] = all_names
            st.session_state["_annot_ta_s2_markers_pending"] = all_names
            st.rerun()
        if marker_columns[2].button("Save selection", key="annot_ta_s2_markers_save", width="stretch"):
            st.session_state[MARKERS_KEY] = selected_markers
            save_project_settings()
            st.success("Marker selection saved.")
        if he_names:
            st.caption(f"Detected virtual H&E layers: {', '.join(he_names)}")

        st.markdown("#### Detect artifacts")
        # The two nudge buttons write here before rerunning: a widget key wins
        # over its own `value` once it exists, so the slider has to be told.
        if "_annot_ta_s2_sensitivity_pending" in st.session_state:
            st.session_state["annot_ta_s2_sensitivity"] = st.session_state.pop(
                "_annot_ta_s2_sensitivity_pending")
        level = int(st.slider(
            "Blur sensitivity", min_value=1, max_value=7, step=1,
            value=sensitivity_level(config), key="annot_ta_s2_sensitivity",
            help=(
                "The whole simple control. The detector works out its own cut-off from the spread of "
                "sharpness in your tissue, and separately checks that a distinct out-of-focus part "
                "exists at all before removing anything. This moves both: 4 leaves them as found, 1 is "
                "stricter on each, 7 is looser on each. If nothing is found even at 7, the tissue has "
                "no region whose sharpness stands apart from the rest, and the honest answer is that "
                "none of it is out of focus."
            ),
        ))
        if level != sensitivity_level(config):
            config = {**config, "blur_sensitivity_level": level}
            save_config(CONFIG_KEY, config)
        level_name, level_scale, separation_scale = SENSITIVITY_LEVELS[level]
        separation = float(config.get("blur_min_separation", 4.0)) * separation_scale
        st.caption(
            f"**{level_name}** - multiplies the automatic cut-off by {level_scale:g}, and asks for a "
            f"{separation:.1f}x separation between sharp and soft tissue before anything is called "
            "blurry. Press Detect, look at the yellow overlay, then move the slider or use the two "
            "buttons underneath."
        )
        run_column, batch_column = st.columns(2)
        run_single = run_column.button(
            "Detect artifacts (this image)", type="primary", width="stretch", key="annot_ta_s2_run")
        run_batch = batch_column.button(
            f"Run batch ({len(batch_samples)} images)", width="stretch", key="annot_ta_s2_run_batch",
            disabled=not batch_samples,
            help="Runs artifact detection for every selected image using each image's own tissue boundary GeoJSON.",
        )

        fewer_column, more_column = st.columns(2)
        if fewer_column.button(
            "Flagged too much", width="stretch", key="annot_ta_s2_less", disabled=level <= 1,
            help="Takes the sensitivity down one step and runs this image again.",
        ):
            save_config(CONFIG_KEY, {**config, "blur_sensitivity_level": level - 1})
            st.session_state["_annot_ta_s2_sensitivity_pending"] = level - 1
            st.session_state["_annot_ta_s2_rerun"] = True
            st.rerun()
        if more_column.button(
            "Missed blurry areas", width="stretch", key="annot_ta_s2_more", disabled=level >= 7,
            help="Takes the sensitivity up one step and runs this image again.",
        ):
            save_config(CONFIG_KEY, {**config, "blur_sensitivity_level": level + 1})
            st.session_state["_annot_ta_s2_sensitivity_pending"] = level + 1
            st.session_state["_annot_ta_s2_rerun"] = True
            st.rerun()

        # Pressing one of the two buttons above saves the new level and comes back
        # round; this is where that trip turns into an actual run.
        if st.session_state.pop("_annot_ta_s2_rerun", False):
            run_single = True

        with st.expander("Expert settings", expanded=False):
            st.caption(
                "Everything the three detectors do, in full. The control above only moves the blur "
                "cut-off; these change how each test is measured and what the other two detectors look "
                "for. Nothing here needs changing for a normal run."
            )
            updated_config, submitted = render_parameter_form(
                form_key="annot_ta_s2_form",
                spec=PARAMETER_SPEC,
                config=_config_for_form(config),
                submit_label="Save artifact detection settings",
            )
            if submitted:
                st.session_state[MARKERS_KEY] = selected_markers
                resolved = _config_from_form(updated_config)
                save_config(CONFIG_KEY, resolved)
                st.success("Artifact detection settings saved to the project configuration.")
                config = resolved

        if run_single or run_batch:
            targets = batch_samples if run_batch else [sample_id]
            progress_bar = st.progress(0.0)
            status = st.empty()
            started = time.time()
            record_process_event(
                TAB_CONTEXT, "Artifact detection started",
                details={"samples": len(targets), "markers": len(selected_markers)},
            )
            completed = 0
            for index, target in enumerate(targets):
                try:
                    status.write(f"Processing {target} ({index + 1}/{len(targets)})...")
                    run_config = run_config_for(config)
                    artifacts = _run_single(target, tissue_config, run_config, selected_markers, status=status)
                    runtime_store(target)["artifacts"] = artifacts
                    log_annotation(
                        f"{target}: valid tissue {artifacts['stats']['valid_percent']:.2f}% "
                        f"(artifacts {artifacts['stats']['artifact_percent']:.2f}%, "
                        f"{artifacts['stats']['blur_regions']} blur regions)",
                        subtab="s2",
                    )
                    completed += 1
                except Exception as error:
                    st.error(f"{target}: {error}")
                    log_annotation(f"{target}: artifact detection failed - {error}", subtab="s2")
                progress_bar.progress((index + 1) / len(targets))
            status.empty()
            record_process_event(
                TAB_CONTEXT, "Artifact detection complete", status="complete",
                duration_s=time.time() - started,
                details={"samples": len(targets), "completed": completed},
            )
            st.success(f"Artifact detection finished for {completed}/{len(targets)} image(s).")

        artifacts = runtime_store(sample_id).get("artifacts")
        if artifacts is not None:
            stats = artifacts["stats"]
            metrics = st.columns(4)
            metrics[0].metric("Clean valid tissue", f"{stats['valid_percent']:.2f}%")
            metrics[1].metric("Blurry", f"{stats['blurry_percent']:.2f}%")
            metrics[2].metric("Bright/flat", f"{stats['bright_percent']:.2f}%")
            metrics[3].metric("Lint overlap", f"{stats['lint_tissue_percent']:.2f}%")
            if not stats.get("blur_population_found", True):
                gap = stats.get("blur_separation_gap")
                required = stats.get("blur_separation_required")
                st.info(
                    "**No out-of-focus region found.** The sharpness of this tissue forms a single "
                    f"population: its softest part is {gap}x less sharp than its sharpest, against the "
                    f"{required}x separation a genuinely defocused area shows. Nothing was removed as "
                    "blur. If you can see blur that was missed, press *Missed blurry areas*."
                    if gap is not None else
                    "**No out-of-focus region found.** Nothing was removed as blur."
                )
            st.caption(
                f"Regions found - blur: {stats.get('blur_regions', 0):,} · "
                f"bright: {stats.get('bright_regions', 0):,} · lint: {stats.get('lint_regions', 0):,}. "
                f"Focus judged on {stats.get('focus_evaluated_percent', 0.0):.1f}% of the tissue "
                f"(the rest carries too little marker signal to judge). "
                f"Per-marker test: {'on' if stats.get('marker_consensus_used') else 'off, too few markers'}. "
                f"Accepted lint components: {stats['accepted_lint']:,}, rejected seam-like candidates: "
                f"{stats['rejected_seams']:,}."
            )

            show_cached_figure(
                sample_id, "step2_artifacts", artifacts["run_token"],
                lambda: figures.artifact_figure(
                    artifacts["tissue_mask"], artifacts,
                    artifacts["maps"].get("codex_robust_max"),
                    artifacts["maps"].get("he_gray"),
                    artifacts["maps"]["codex_projection"],
                    float(config.get("lint_robust_z_threshold", 12.0)),
                    sample_id,
                ),
            )
            show_cached_figure(
                sample_id, "step2_blur_scoring", artifacts["run_token"],
                lambda: figures.blur_diagnostic_figure(artifacts, sample_id),
            )

            for artifact_type, title in (
                ("blur", "Blurry regions"),
                ("bright", "Bright / flat regions"),
            ):
                table = _region_table(artifacts, artifact_type)
                if not table.empty:
                    with st.expander(f"{title} and the markers behind them ({len(table):,})", expanded=False):
                        st.dataframe(table.head(200), hide_index=True, width="stretch")
                        saved = save_table(table, sample_id, f"step2_{artifact_type}_regions")
                        if saved:
                            st.caption(f"Full table saved to `{saved}`")

            lint_components = artifacts.get("lint_components")
            if lint_components is not None and not lint_components.empty:
                with st.expander(f"Lint candidate measurements ({len(lint_components):,} components)", expanded=False):
                    st.dataframe(lint_components.head(200), hide_index=True, width="stretch")
                    saved = save_table(lint_components, sample_id, "step2_lint_components")
                    if saved:
                        st.caption(f"Full table saved to `{saved}`")

        st.markdown("##### Output files")
        st.caption(
            "Each artifact class is exported on its own so it can be reviewed or corrected independently "
            "in QuPath, alongside their union and the cleaned tissue. Blur, bright and lint layers hold "
            "one annotation per region; open the measurement table in QuPath to see the evidence."
        )
        describe_output(geojson_path(sample_id, "valid_tissue"), "Valid tissue (the annotation)")
        for suffix, label in (
            ("detected_blur", "Blurry regions"),
            ("detected_bright", "Bright / flat artifacts"),
            ("detected_lint", "Filamentous lint"),
            ("detected_artifacts", "All artifacts merged"),
        ):
            describe_output(geojson_path(sample_id, suffix), label)
        describe_output(
            intermediate_file(sample_id, "artifact_review.json"),
            "Review manifest (every region, with its evidence)",
        )
        if os.path.isfile(geojson_path(sample_id, "valid_tissue")):
            qupath_hint(geojson_path(sample_id, "valid_tissue"), "Step 3")

    with right:
        st.markdown("#### Preview")
        image_path = resolve_image_path(sample_id)
        if not image_path:
            st.info("Select an image to preview.")
            return

        layer_options = {
            "Valid tissue": geojson_path(sample_id, "valid_tissue"),
            "Blurry regions": geojson_path(sample_id, "detected_blur"),
            "Bright / flat": geojson_path(sample_id, "detected_bright"),
            "Filamentous lint": geojson_path(sample_id, "detected_lint"),
            "All artifacts": geojson_path(sample_id, "detected_artifacts"),
            "Tissue boundary": geojson_path(sample_id, "tissue_boundary"),
        }
        available = [name for name, path in layer_options.items() if os.path.isfile(path)]
        if available:
            prune_widget_selection("annot_ta_s2_layers", available)
            chosen = st.multiselect(
                "Overlay layers",
                available,
                default=[name for name in ("Valid tissue", "Filamentous lint") if name in available],
                key="annot_ta_s2_layers",
                help="Polygons are read fresh from disk every render, so QuPath edits appear immediately.",
            )
            if chosen:
                image, colors = overlay_polygons_on_thumbnail(
                    image_path,
                    [layer_options[name] for name in chosen],
                    class_colors={
                        "Valid_Tissue": "#2ecc71",
                        "Detected_Blur": "#ffff00",
                        "Detected_Bright": "#ff00ff",
                        "Detected_Lint": "#00ffff",
                        "Detected_Artifacts": "#ff5252",
                        "Tissue_Boundary": "#ffb400",
                    },
                )
                st.image(image, width="stretch", caption=f"{sample_id} - {', '.join(chosen)}")
                render_class_legend(colors)
            else:
                render_image_preview(image_path, "annot_ta_s2")
        else:
            render_image_preview(image_path, "annot_ta_s2")

        st.caption(f"Figures and tables for this image: `{output_dir_for(sample_id, 'figures')}`")
