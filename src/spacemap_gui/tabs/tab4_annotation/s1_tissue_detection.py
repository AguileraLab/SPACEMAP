"""Subtab 1 - detect the valid tissue mask from the virtual H&E channels."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import streamlit as st

from spacemap_gui.process_log import record_process_event

from . import figures
from .common import (
    TAB_CONTEXT,
    describe_output,
    geojson_path,
    log_annotation,
    open_reader,
    output_dir_for,
    overlay_polygons_on_thumbnail,
    tidy_output_directory,
    qupath_hint,
    render_class_legend,
    render_image_preview,
    render_parameter_form,
    resolve_image_path,
    runtime_store,
    save_config,
    save_project_settings,
    show_cached_figure,
)

CONFIG_KEY = "annot_ta_tissue_config"

PARAMETER_SPEC = [
    {"type": "header", "label": "Resolution", "help": "Which pyramid level the masking runs on."},
    {
        "key": "blur_downsample", "label": "Masking downsample (x)", "type": "int", "min": 1, "max": 16, "step": 1,
        "help": (
            "Pyramid level used for tissue masking and artifact detection. 2 means the 2x level is read "
            "directly from the image pyramid, so no full-resolution decode happens. Higher values are "
            "faster and coarser."
        ),
    },
    {
        "key": "microns_per_pixel", "label": "Pixel size (um/px at full res)", "type": "number",
        "min": 0.0001, "max": 10.0, "step": 0.0001, "format": "%.4f",
        "help": "Full-resolution pixel size. All micron-based thresholds in this tab are converted using this value.",
    },
    {"type": "header", "label": "H&E optical density", "help": "How stained tissue is separated from slide background."},
    {
        "key": "he_white_percentile", "label": "White reference percentile", "type": "number",
        "min": 50.0, "max": 100.0, "step": 0.1,
        "help": (
            "Per-channel intensity percentile treated as the white background reference before converting "
            "to optical density. Lower it if the slide background is unusually dark."
        ),
    },
    {
        "key": "he_od_threshold_factor", "label": "Otsu threshold factor", "type": "number",
        "min": 0.1, "max": 2.0, "step": 0.05,
        "help": (
            "Multiplier applied to the automatic Otsu optical-density threshold. Below 1.0 the mask becomes "
            "more inclusive (captures faint tissue); above 1.0 it becomes stricter."
        ),
    },
    {
        "key": "he_valid_min_intensity", "label": "Minimum valid H&E intensity", "type": "number",
        "min": 0.0, "max": 65535.0, "step": 1.0,
        "help": "Pixels at or below this intensity in any H&E channel are treated as acquisition padding and excluded.",
    },
    {"type": "header", "label": "Mask composition", "help": "Whether CODEX signal may contribute to the tissue mask."},
    {
        "key": "tissue_mask_mode", "label": "Tissue mask mode", "type": "select",
        "options": ["he_only", "he_with_codex_edge_rescue", "he_codex_union", "codex_only"],
        "help": (
            "he_only uses H&E alone and prevents marker-only lint from becoming tissue (recommended when "
            "virtual H&E layers exist). he_with_codex_edge_rescue additionally accepts marker signal within "
            "a margin of an H&E-positive boundary. he_codex_union accepts any marker-positive pixel and is "
            "the most permissive. codex_only ignores H&E entirely and thresholds the marker projection, "
            "which is the option to use when the image has no virtual H&E layers."
        ),
    },
    {
        "key": "codex_threshold_mode", "label": "Marker threshold", "type": "select",
        "options": ["otsu", "fixed"],
        "help": (
            "How the marker projection is thresholded in the marker-based modes. otsu adapts to each image "
            "and transfers across panels and exposures; fixed uses the value below unchanged."
        ),
    },
    {
        "key": "he_codex_edge_rescue_um", "label": "CODEX edge rescue margin (µm)", "type": "number",
        "min": 0.0, "max": 500.0, "step": 1.0,
        "help": "Only used in he_with_codex_edge_rescue mode: how far outside the H&E boundary CODEX signal may rescue tissue.",
    },
    {
        "key": "tissue_threshold", "label": "Fixed marker threshold", "type": "number",
        "min": 0.0, "max": 255.0, "step": 1.0,
        "help": "Intensity cut on the normalized marker max projection (0-255). Only used when the marker threshold is set to fixed.",
    },
    {"type": "header", "label": "Cleanup", "help": "Morphology and border handling applied to the raw mask."},
    {
        "key": "tissue_morph_radius", "label": "Morphology radius (px)", "type": "int", "min": 0, "max": 100, "step": 1,
        "help": (
            "Radius of the closing then opening disk applied at the masking resolution. Larger values fill "
            "gaps inside tissue and remove speckle, but round off fine tissue edges."
        ),
    },
    {
        "key": "min_tissue_island_px", "label": "Minimum tissue island (px)", "type": "int",
        "min": 0, "max": 10_000_000, "step": 100,
        "help": "Connected components smaller than this (at the masking resolution) are discarded as debris.",
    },
    {
        "key": "tissue_border_exclusion_um", "label": "Border exclusion (µm)", "type": "number",
        "min": 0.0, "max": 1000.0, "step": 1.0,
        "help": "Strip removed from all four image edges, which drops black scanner frames and edge artifacts.",
    },
    {"type": "header", "label": "Blood clot / haemorrhage",
     "help": (
         "Finds solid masses of blood: intensely pink, with almost no cell nuclei in them, and "
         "structureless. All three have to hold at once, because pink alone also describes muscle and "
         "collagen, and nuclei-free alone also describes lumen, fat and blank areas. Uses the virtual "
         "H&E only. Each setting is read against the same slide, so if there is no clot the answer is "
         "none rather than the most clot-like tissue present."
     )},
    {
        "key": "detect_blood_clot", "label": "Detect blood clot", "type": "checkbox",
        "help": (
            "Off by default. Turn on to look for blood clot. It reads no extra channels, so it adds "
            "almost no time. Start with the defaults, run once, and look at the red overlay in the "
            "preview before changing anything. Finding nothing is a normal and correct result."
        ),
    },
    {
        "key": "clot_eosin_ratio", "label": "How much pinker than pale tissue", "type": "number",
        "min": 1.0, "max": 10.0, "step": 0.1,
        "help": (
            "2.0 means a clot must be at least twice as pink as the pale end of the tissue on this "
            "slide. LOWER it (1.5-1.7) if an obvious clot is being missed. RAISE it (2.5-3) if normal "
            "pink tissue is being flagged. This is a ratio, not a percentile, so it does not select "
            "anything at all when nothing is pink enough."
        ),
    },
    {
        "key": "clot_max_nuclei_ratio", "label": "How much emptier of nuclei than normal tissue", "type": "number",
        "min": 0.01, "max": 2.0, "step": 0.05,
        "help": (
            "The test that separates a clot from muscle and fibrous stroma. 0.25 means nuclei may cover "
            "at most a quarter of what they cover in ordinary tissue here. RAISE it (0.4-0.5) if a real "
            "clot is only partly found. LOWER it (0.15) if nucleated tissue is still being taken."
        ),
    },
    {
        "key": "clot_eosin_smooth_ratio", "label": "How much smoother than normal tissue", "type": "number",
        "min": 0.05, "max": 3.0, "step": 0.05,
        "help": (
            "A clot is a featureless mass; collagen and muscle carry visible fibre striations. 0.75 "
            "means the pink must vary at most three quarters as much as it does across the tissue "
            "generally. Skipped automatically when the H&E has no texture to compare against."
        ),
    },
    {
        "key": "clot_min_area_mm2", "label": "Ignore clots smaller than (mm^2)", "type": "number",
        "min": 0.0, "max": 100.0, "step": 0.01, "format": "%.3f",
        "help": (
            "Small scattered pockets of blood are ignored; only masses this large are kept. "
            "0.05 mm^2 is about a 0.2 x 0.2 mm patch. RAISE it to keep only the one big clot. "
            "Set 0 to keep every island found. Check the island table under the metrics to see the "
            "sizes actually present and pick a number between them."
        ),
    },
    {
        "key": "clot_texture_window_um", "label": "Nuclei-checking window (µm)", "type": "number",
        "min": 5.0, "max": 300.0, "step": 5.0,
        "help": (
            "How large an area is examined when asking 'are there nuclei here?'. It has to be wide "
            "enough to contain several nuclei, so 30-60 um. Usually leave at 40. Too small and nothing "
            "is ever detected, because a tiny window fits inside a single nucleus."
        ),
    },
    {
        "key": "clot_morph_radius_um", "label": "Smoothing radius (µm)", "type": "number",
        "min": 0.0, "max": 200.0, "step": 5.0,
        "help": (
            "Tidies the outline: fills small holes inside the clot and removes thin wisps around it. "
            "Also acts as a size floor, so anything narrower than this disappears before the minimum "
            "area is applied. LOWER it if small clots vanish entirely; raise it for a cleaner outline."
        ),
    },
    {
        "key": "clot_nuclei_reference_percentile", "label": "What counts as normal nuclei density (percentile)",
        "type": "number", "min": 50.0, "max": 99.0, "step": 5.0,
        "help": (
            "Which part of the slide is taken as ordinary nucleated tissue, the benchmark the setting "
            "above is measured against. 75 means the upper quarter. Leave it unless a clot covers most "
            "of the slide, in which case raise it."
        ),
    },
    {
        "key": "clot_eosin_reference_percentile", "label": "What counts as pale tissue (percentile)",
        "type": "number", "min": 1.0, "max": 50.0, "step": 5.0,
        "help": (
            "Which part of the slide is taken as the pale end, the benchmark for pinkness. 25 means the "
            "palest quarter. Leave it unless a clot covers most of the slide, in which case lower it."
        ),
    },
    {
        "key": "clot_exclude_from_tissue", "label": "Remove clot from the tissue boundary", "type": "checkbox",
        "help": (
            "On: the clot is cut out of the tissue, so later steps never analyse it. "
            "Off: the clot is still saved as its own layer for you to look at, but the tissue "
            "boundary is left exactly as it was. Turn it off the first time, check the overlay, then "
            "turn it on once you are happy with the settings."
        ),
    },
    {"type": "header", "label": "Signal statistics", "help": "Robust statistics reused by the artifact step."},
    {
        "key": "clamp_sigma", "label": "Projection clamp (sigma)", "type": "number", "min": 0.5, "max": 20.0, "step": 0.5,
        "help": "The CODEX max projection is clipped at mean + this many standard deviations before normalizing to 0-255.",
    },
    {
        "key": "robust_signal_clip_z", "label": "Robust Z clip", "type": "number", "min": 5.0, "max": 200.0, "step": 5.0,
        "help": "Upper clip on the per-channel robust Z map. This map seeds filamentous lint detection in step 2.",
    },
    {
        "key": "bright_breadth_channel_sigma", "label": "Breadth robust-Z threshold", "type": "number",
        "min": 1.0, "max": 20.0, "step": 0.5,
        "help": "A channel counts as strongly positive at a pixel above this robust Z. Used by the optional breadth-based bright artifact rule in step 2.",
    },
    {
        "key": "boundary_simplify_px", "label": "Boundary simplification (px)", "type": "number",
        "min": 0.0, "max": 20.0, "step": 0.5,
        "help": "Douglas-Peucker tolerance applied when vectorizing the mask. Higher values give smaller GeoJSON files with coarser outlines.",
    },
]


HE_CHANNELS_KEY = "annot_ta_he_channels"
CODEX_MARKERS_KEY = "annot_ta_tissue_codex_markers"

NONE_LABEL = "(none)"


def resolve_tissue_channels(sample_id: str) -> tuple[list[int], list[int]]:
    """Channel indices the tissue step should use, honouring any manual choice.

    Returns ``(he_indices, codex_indices)``. Falls back to name-based detection
    for H&E, and to every marker for the marker projection.
    """
    from spacemap.tissue_annotation_utils import split_he_and_codex_channels

    image_path = resolve_image_path(sample_id)
    if not image_path:
        return [], []
    reader = open_reader(image_path)
    names = list(reader.channel_names)
    detected_he, detected_codex = split_he_and_codex_channels(names)

    chosen_he = [name for name in st.session_state.get(HE_CHANNELS_KEY, []) if name in names]
    he_indices = [names.index(name) for name in chosen_he] if chosen_he else list(detected_he)

    chosen_markers = [name for name in st.session_state.get(CODEX_MARKERS_KEY, []) if name in names]
    codex_indices = [names.index(name) for name in chosen_markers] if chosen_markers else list(detected_codex)
    return he_indices, codex_indices


def _render_channel_panel(sample_id: str, config: dict) -> None:
    """Let the user name the H&E layers, or switch to a marker-based method."""
    from spacemap.tissue_annotation_utils import split_he_and_codex_channels

    image_path = resolve_image_path(sample_id)
    if not image_path:
        return
    reader = open_reader(image_path)
    names = list(reader.channel_names)
    detected_he, detected_codex = split_he_and_codex_channels(names)
    detected_he_names = [names[index] for index in detected_he]
    mode = str(config.get("tissue_mask_mode", "he_only"))

    st.markdown("#### Tissue source channels")
    if detected_he_names:
        st.caption(
            f"Detected virtual H&E layers by name: {', '.join(detected_he_names)}. "
            "Override them below if the channels are named differently."
        )
    else:
        st.warning(
            "No virtual H&E layers were detected by name. Either name them below, or set the tissue mask "
            "mode to `codex_only` and pick the markers to threshold instead (DAPI alone usually works well)."
        )

    options = [NONE_LABEL] + names
    saved = [name for name in st.session_state.get(HE_CHANNELS_KEY, []) if name in names]
    if not saved:
        saved = detected_he_names
    defaults = (saved + [NONE_LABEL, NONE_LABEL, NONE_LABEL])[:3]

    columns = st.columns(3)
    picked: list[str] = []
    for position, (column, role) in enumerate(zip(columns, ("Red", "Green", "Blue"))):
        current = defaults[position] if defaults[position] in options else NONE_LABEL
        with column:
            choice = st.selectbox(
                f"H&E {role} layer",
                options,
                index=options.index(current),
                key=f"annot_ta_s1_he_{role.lower()}",
                help=(
                    f"Channel holding the {role.lower()} component of the virtual H&E image. "
                    "All three are needed for optical-density masking."
                ),
            )
        if choice != NONE_LABEL:
            picked.append(choice)

    if len(picked) < 3 and mode != "codex_only":
        st.warning(
            f"Only {len(picked)} H&E layer(s) selected. Optical-density masking needs three; "
            "with fewer, detection falls back to the marker projection."
        )

    marker_options = names
    saved_markers = [name for name in st.session_state.get(CODEX_MARKERS_KEY, []) if name in marker_options]
    if not saved_markers:
        dapi = [name for name in marker_options if "dapi" in name.lower() or "dna" in name.lower()]
        saved_markers = dapi or [names[index] for index in detected_codex]
    marker_help = (
        "Markers combined into the max projection that is thresholded when the mask mode uses markers. "
        "DAPI alone is a good default: it is present in every cell and rarely saturates."
    )
    # A preset button stages its value here and reruns; applying it before the
    # widget is instantiated is the only legal way to change a widget's value.
    if "_annot_ta_s1_codex_markers_pending" in st.session_state:
        st.session_state["annot_ta_s1_codex_markers"] = st.session_state.pop(
            "_annot_ta_s1_codex_markers_pending"
        )
    stored = st.session_state.get("annot_ta_s1_codex_markers")
    if isinstance(stored, list) and any(name not in marker_options for name in stored):
        st.session_state["annot_ta_s1_codex_markers"] = [n for n in stored if n in marker_options]
    chosen_markers = st.multiselect(
        "Marker channels for marker-based detection",
        marker_options,
        default=saved_markers,
        key="annot_ta_s1_codex_markers",
        help=marker_help,
    )

    button_columns = st.columns([1, 1, 2])
    if button_columns[0].button("Use DAPI only", key="annot_ta_s1_markers_dapi", width="stretch"):
        dapi = [name for name in marker_options if "dapi" in name.lower() or "dna" in name.lower()]
        st.session_state[CODEX_MARKERS_KEY] = dapi or marker_options[:1]
        st.session_state["_annot_ta_s1_codex_markers_pending"] = st.session_state[CODEX_MARKERS_KEY]
        st.rerun()
    if button_columns[1].button("Save channels", key="annot_ta_s1_save_channels", width="stretch"):
        st.session_state[HE_CHANNELS_KEY] = picked
        st.session_state[CODEX_MARKERS_KEY] = chosen_markers
        save_project_settings()
        st.success("Tissue source channels saved.")

    st.session_state[HE_CHANNELS_KEY] = picked
    st.session_state[CODEX_MARKERS_KEY] = chosen_markers

    if mode == "codex_only":
        st.info(f"Mode `codex_only`: tissue is thresholded from {len(chosen_markers)} marker channel(s); H&E is ignored.")
    elif len(picked) >= 3:
        st.caption(f"Mode `{mode}`: tissue comes from the optical density of {', '.join(picked)}.")


def _run_single(sample_id: str, config: dict, status=None) -> dict:
    """Detect tissue for one sample and write its boundary GeoJSON."""
    from spacemap.tissue_annotation_utils import (
        detect_tissue_mask,
        mask_to_geojson_features,
        qupath_color_from_hex,
        region_features_from_labels,
        write_feature_collection,
    )

    image_path = resolve_image_path(sample_id)
    if not image_path:
        raise FileNotFoundError(f"No image file found for sample '{sample_id}'.")

    reader = open_reader(image_path)

    def progress(done: int, total: int, name: str) -> None:
        if status is not None:
            status.write(f"{sample_id}: reading channel {done}/{total} ({name})")

    he_indices, codex_indices = resolve_tissue_channels(sample_id)
    # need_codex_maps stays False: in he_only mode the mask comes from the
    # selected H&E channels alone, so the marker channels are never decoded here.
    # The sample name travels with the config so each detected clot region can
    # carry an identifier the review pass can refer back to.
    result = detect_tissue_mask(
        reader, {**config, "sample_id": sample_id},
        he_indices=he_indices, codex_indices=codex_indices, progress=progress,
    )

    output_dir = st.session_state.get("annot_ta_output_dir", "")
    # A project written by an older build has its review layers loose beside the
    # annotation; this puts them where they now belong before anything is added.
    tidy_output_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    features = mask_to_geojson_features(
        result["tissue_mask"],
        scale_factor=float(config.get("blur_downsample", 2)),
        class_name="Tissue_Boundary",
        color_rgb=-16711936,
        simplify_px=float(config.get("boundary_simplify_px", 2.0)),
        merge=False,
        locked=True,
    )
    boundary_path = geojson_path(sample_id, "tissue_boundary")
    size_mb = write_feature_collection(features, boundary_path)

    # The annotation itself starts here. Step 2 rewrites this same file with the
    # artifacts taken out, step 3 adds its classes to it, and so on, so the
    # output folder always holds one file that is the current annotation while
    # the layers that explain it stay in the subfolder.
    valid_features = mask_to_geojson_features(
        result["tissue_mask"],
        scale_factor=float(config.get("blur_downsample", 2)),
        class_name="Valid_Tissue",
        color_rgb=int(qupath_color_from_hex("#2ecc71")),
        simplify_px=float(config.get("boundary_simplify_px", 2.0)),
        merge=True,
    )
    valid_path = geojson_path(sample_id, "valid_tissue")
    write_feature_collection(valid_features, valid_path)
    result["valid_path"] = valid_path

    # The clot is always exported when detection is on, even if empty, so a stale
    # file from a previous run can never be mistaken for the current result.
    if bool(config.get("detect_blood_clot", False)):
        # One annotation per clot, each carrying the three measurements that
        # produced it, so a reviewer can accept one and reject another.
        clot = result.get("clot") or {}
        regions = clot.get("regions") or []
        labels = clot.get("region_labels")
        clot_features = (
            region_features_from_labels(
                labels, regions,
                float(config.get("blur_downsample", 2)),
                "Blood_Clot",
                color_rgb=int(qupath_color_from_hex("#e6194b")),
                simplify_px=float(config.get("boundary_simplify_px", 2.0)),
            )
            if labels is not None and regions
            else []
        )
        result["clot_path"] = geojson_path(sample_id, "blood_clot")
        write_feature_collection(clot_features, result["clot_path"])

    result["boundary_path"] = boundary_path
    result["run_token"] = uuid.uuid4().hex
    result["boundary_size_mb"] = size_mb
    result["image_path"] = image_path
    return result


def render(project_root: Path, sample_id: str, batch_samples: list[str]) -> None:
    st.markdown("### Step 1 - Detect valid tissue from H&E")
    st.caption(
        "Separates stained tissue from slide background using the virtual H&E channels only. "
        "The output GeoJSON is the starting point for artifact removal and can be corrected in QuPath first."
    )


    config = st.session_state.get(CONFIG_KEY, {})
    left, right = st.columns([1.25, 1.0])

    with left:
        _render_channel_panel(sample_id, config)

        st.markdown("#### Parameters")
        updated_config, submitted = render_parameter_form(
            form_key="annot_ta_s1_form",
            spec=PARAMETER_SPEC,
            config=config,
            submit_label="Save tissue detection settings",
        )
        if submitted:
            save_config(CONFIG_KEY, updated_config)
            st.success("Tissue detection settings saved to the project configuration.")
            config = updated_config

        image_path = resolve_image_path(sample_id)
        if image_path:
            reader = open_reader(image_path)
            with st.expander("Image structure", expanded=False):
                st.write(
                    f"**Layout:** `{reader.layout}` · **Channels:** {reader.n_channels} · "
                    f"**Pyramid levels:** {reader.level_count}"
                )
                st.write("**Level shapes:** " + ", ".join(f"{h}x{w}" for h, w in reader.level_shapes))
                from spacemap.tissue_annotation_utils import split_he_and_codex_channels

                he_indices, codex_indices = split_he_and_codex_channels(reader.channel_names)
                st.write("**Detected H&E channels:** " + (", ".join(reader.channel_names[i] for i in he_indices) or "none"))
                st.caption(f"{len(codex_indices)} CODEX marker channels available for later steps.")
                if len(he_indices) < 3:
                    st.warning(
                        "Fewer than three virtual H&E channels were detected. Tissue masking will fall back to "
                        "the CODEX projection, which is less specific."
                    )

        st.markdown("#### Run")
        run_column, batch_column = st.columns(2)
        run_single = run_column.button("Detect tissue (this image)", type="primary", width="stretch", key="annot_ta_s1_run")
        run_batch = batch_column.button(
            f"Run batch ({len(batch_samples)} images)", width="stretch", key="annot_ta_s1_run_batch",
            disabled=not batch_samples,
            help="Runs tissue detection with the saved parameters for every image selected above.",
        )

        if run_single or run_batch:
            targets = batch_samples if run_batch else [sample_id]
            progress_bar = st.progress(0.0)
            status = st.empty()
            started = time.time()
            record_process_event(
                TAB_CONTEXT, "Tissue detection started",
                details={"samples": len(targets), "mode": config.get("tissue_mask_mode")},
            )
            completed = 0
            for index, target in enumerate(targets):
                try:
                    status.write(f"Processing {target} ({index + 1}/{len(targets)})...")
                    result = _run_single(target, config, status=status)
                    runtime_store(target)["tissue"] = result
                    runtime_store(target).pop("artifacts", None)
                    log_annotation(
                        f"{target}: tissue {result['stats']['tissue_percent']:.1f}% -> {result['boundary_path']}",
                        subtab="s1",
                    )
                    completed += 1
                except Exception as error:
                    st.error(f"{target}: {error}")
                    log_annotation(f"{target}: tissue detection failed - {error}", subtab="s1")
                progress_bar.progress((index + 1) / len(targets))
            status.empty()
            record_process_event(
                TAB_CONTEXT, "Tissue detection complete", status="complete",
                duration_s=time.time() - started,
                details={"samples": len(targets), "completed": completed},
            )
            st.success(f"Tissue detection finished for {completed}/{len(targets)} image(s).")

        result = runtime_store(sample_id).get("tissue")
        boundary_path = geojson_path(sample_id, "tissue_boundary")

        if result is not None:
            stats = result["stats"]
            metric_columns = st.columns(3)
            metric_columns[0].metric("Tissue coverage", f"{stats['tissue_percent']:.1f}%")
            metric_columns[1].metric("Tissue pixels", f"{stats['tissue_pixels']:,}")
            metric_columns[2].metric(
                "OD threshold",
                f"{stats['he_od_threshold']:.4f}" if stats.get("he_od_threshold") is not None else "n/a",
            )
            if stats.get("he_white_references"):
                st.caption(f"H&E white references: {stats['he_white_references']} · Otsu: {stats['otsu']:.4f}")
            clot = result.get("clot")
            if clot is not None:
                clot_stats = clot["stats"]
                st.markdown("##### Blood clot")
                clot_metrics = st.columns(3)
                clot_metrics[0].metric("Clot area", f"{clot_stats['clot_area_mm2']:.2f} mm²")
                clot_metrics[1].metric("Share of tissue", f"{clot_stats['clot_percent']:.1f}%")
                clot_metrics[2].metric("Islands kept", f"{clot_stats['clot_islands']}")
                if clot_stats["clot_islands"]:
                    st.caption(
                        f"Largest island {clot_stats['largest_island_mm2']:.2f} mm² · "
                        f"{clot_stats['rejected_islands']} island(s) below the minimum area were discarded · "
                        f"eosin cut {clot_stats['eosin_cut']:.4f}, texture cut {clot_stats['texture_cut']:.4f}"
                    )
                    if config.get("clot_exclude_from_tissue", True):
                        st.caption("The clot has been subtracted from the tissue boundary.")
                else:
                    st.caption(
                        "No clot island passed the minimum area. Lower the minimum area, or the eosin "
                        "percentile, if a clot is visibly present."
                    )
                components = clot.get("components") or []
                if components:
                    import pandas as pd

                    with st.expander(f"Clot island measurements ({len(components)})", expanded=False):
                        st.dataframe(pd.DataFrame(components), hide_index=True, width="stretch")

            show_cached_figure(
                sample_id, "step1_tissue_mask", result["run_token"],
                lambda: figures.tissue_mask_figure(result, sample_id),
            )

        describe_output(geojson_path(sample_id, "valid_tissue"), "Valid tissue (the annotation)")
        describe_output(boundary_path, "Tissue boundary (working layer for step 2)")
        if os.path.isfile(geojson_path(sample_id, "blood_clot")):
            describe_output(geojson_path(sample_id, "blood_clot"), "Blood clot GeoJSON")
        if os.path.isfile(boundary_path):
            qupath_hint(boundary_path, "Step 2")

    with right:
        st.markdown("#### Preview")
        image_path = resolve_image_path(sample_id)
        if not image_path:
            st.info("Select an image to preview.")
            return

        overlay_boundary = st.checkbox(
            "Overlay detected tissue boundary",
            value=True,
            key="annot_ta_s1_overlay",
            help="Draws the tissue boundary GeoJSON on the thumbnail. The file is re-read from disk each time.",
        )
        clot_path = geojson_path(sample_id, "blood_clot")
        overlay_clot = False
        if os.path.isfile(clot_path):
            overlay_clot = st.checkbox(
                "Overlay detected blood clot",
                value=True,
                key="annot_ta_s1_overlay_clot",
                help="Draws the blood clot GeoJSON so it can be checked against the H&E before step 2.",
            )

        layers = []
        if overlay_boundary and os.path.isfile(geojson_path(sample_id, "tissue_boundary")):
            layers.append(geojson_path(sample_id, "tissue_boundary"))
        if overlay_clot:
            layers.append(clot_path)

        if layers:
            image, colors = overlay_polygons_on_thumbnail(
                image_path,
                layers,
                class_colors={"Tissue_Boundary": "#00ff00", "Blood_Clot": "#e6194b"},
            )
            st.image(image, width="stretch", caption=f"{sample_id} - tissue detection")
            render_class_legend(colors)
        else:
            render_image_preview(image_path, "annot_ta_s1")

        st.caption(f"Figures and tables for this image: `{output_dir_for(sample_id, 'figures')}`")
