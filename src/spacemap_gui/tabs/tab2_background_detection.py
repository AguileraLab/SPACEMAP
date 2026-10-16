"""Detect each marker's background from the image, when no blank cycles exist.

Blank-cycle subtraction is the better route when the panel provides blanks.
This subtab covers the case where it does not: the background is estimated from
every marker's own pixel histogram, reviewed, corrected by hand if needed, and
then either written as a threshold table for the RESOLVE phenotyping step or
baked into a background-subtracted copy of the image.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from spacemap.background_detection import (
    DEFAULT_MIN_COMPONENT_AREA_PX,
    THRESHOLD_METHODS,
    align_region_aggregate,
    detect_backgrounds,
    geojson_component_summary,
    image_levels,
    mask_from_geojson,
    masks_from_geojson,
    otsu_tissue_mask,
    prepare_threshold_export,
    subtract_background_image,
    threshold_csv_name,
    write_threshold_csv,
)
from spacemap_gui.process_log import record_process_event
from spacemap_gui.ui_utils import directory_input_with_browse

def _threshold_parameters(prefix):
    """The same bounded configuration as Studio; shared through session state."""
    from spacemap.thresholding_engine import parameters
    with st.expander("Percentile / Manual parameters"):
        percentile = st.number_input("Percentile", min_value=0.0, max_value=100.0, value=95.0, key=prefix + "_percentile")
        manual = st.number_input("Manual cutoff", value=0.0, key=prefix + "_manual")
    params = parameters({"percentile": percentile, "manual": manual})
    st.session_state["thresholding_parameters"] = params
    return params


TABLE_KEY = "tab2_bgdet_table"
MASK_KEY = "tab2_bgdet_mask"
BATCH_KEY = "tab2_bgdet_batch_table"
IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif")
POLYGON_OUTPUT_OPTIONS = {
    "One QuPath-compatible value (recommended)": "combined",
    "Separate annotation columns": "individual",
}
POLYGON_MERGE_OPTIONS = {
    "Mean": "mean",
    "Median": "median",
    "5th percentile": "p05",
    "25th percentile": "p25",
    "75th percentile": "p75",
}


def _images_in(directory: str) -> list[str]:
    path = Path(directory)
    if not path.is_dir():
        return []
    return sorted(f.name for f in path.iterdir() if f.name.lower().endswith(IMAGE_SUFFIXES))


def _stem(filename: str) -> str:
    name = Path(filename).name
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


@st.cache_resource(show_spinner=False)
def _reader(path: str, mtime: float):
    del mtime
    from spacemap.tissue_annotation_utils import PyramidImageReader

    return PyramidImageReader(path)


def _open(path: str):
    return _reader(path, os.path.getmtime(path))


@st.cache_data(show_spinner=False)
def _plane(path: str, mtime: float, channel: str, downsample: int) -> np.ndarray:
    """One channel at one pyramid level. Cached because the review panel reads
    the same plane repeatedly while the user drags the manual threshold."""
    del mtime
    import tifffile

    reader = _open(path)
    index = list(reader.channel_names).index(channel)
    with tifffile.TiffFile(path) as tif:
        return reader.read_channel(tif, index, int(downsample))


def _geojson_classes(path: str) -> list[str]:
    """Classification names available in one annotation GeoJSON."""
    if not path or not os.path.isfile(path):
        return []
    try:
        import json

        with open(path) as handle:
            collection = json.load(handle)
        return sorted({
            str(((feature.get("properties") or {}).get("classification") or {}).get("name", "")).strip()
            for feature in collection.get("features", [])
        } - {""})
    except Exception:
        return []


def _polygon_mask_note(
    region_masks: dict[str, np.ndarray],
    downsample: int,
    source_name: str,
    component_summary: dict | None = None,
) -> str:
    areas = [int(mask.sum()) for mask in region_masks.values() if mask.any()]
    if not areas:
        return f"0 valid polygons from {source_name}"
    note = (
        f"{len(areas)} annotation object(s) from {source_name}; "
        f"area {min(areas):,}-{max(areas):,} pixels at {downsample}x downsample"
    )
    if component_summary:
        note += (
            f"; {int(component_summary['kept_components'])} disconnected component(s) kept, "
            f"{int(component_summary['ignored_components'])} below "
            f"{float(component_summary['min_component_area_px']):,.0f} px² ignored"
        )
    return note


def _build_mask(image_path: str, geojson: str, classes: list[str], downsample: int,
                reference_plane: np.ndarray, min_component_area_px: float = 0.0,
                ) -> tuple[np.ndarray | None, str, str, str]:
    if geojson and os.path.isfile(geojson):
        try:
            mask = mask_from_geojson(
                geojson, reference_plane.shape, downsample,
                classes=classes or None,
                min_component_area_px=min_component_area_px,
            )
            if mask.any():
                return (
                    mask,
                    f"{mask.mean():.1%} of the frame, from {Path(geojson).name}",
                    "geojson_pooled_pixels",
                    "none",
                )
            selected = ", ".join(classes) if classes else "all classes"
            raise ValueError(
                f"{Path(geojson).name} has no valid in-frame pixels for {selected}. "
                "Choose a class present in this image or inspect the annotation coordinates."
            )
        except Exception as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(f"Could not read {Path(geojson).name}: {error}") from error
    try:
        mask = otsu_tissue_mask(reference_plane)
        return (
            mask,
            f"{mask.mean():.1%} of the frame, from an automatic Otsu mask",
            "otsu_pooled_pixels",
            "otsu_no_geojson",
        )
    except Exception as error:
        return (
            None,
            f"Otsu fallback failed ({error}); using the whole frame",
            "whole_frame_pixels",
            "whole_frame_after_otsu_failure",
        )


def render(project_root: Path) -> None:
    st.markdown(
        "Detect marker background and create threshold tables for RESOLVE."
    )
    threshold_dir = directory_input_with_browse(
        label="Project threshold tables directory (shared with RESOLVE)",
        text_key="threshold_dir",
        default_value=str(
            st.session_state.get("threshold_dir", project_root / "data" / "thresholds")
        ),
        browser_key_prefix="tab2_shared_threshold_dir",
        start_dir_fallback=str(project_root),
        help_text=(
            "Tab 2 writes both single-image and batch threshold CSVs here, and RESOLVE "
            "reads the same project-level directory. The default is "
            "<project>/data/thresholds. Changes are synchronized with Workspace."
        ),
    )
    st.caption(
        "This is the shared project destination for both modes below. It is saved in the "
        "project JSON as `directory_mappings.threshold_dir`."
    )
    single_tab, batch_tab = st.tabs(["Single Image", "Multiple Images"])
    with single_tab:
        _render_single(project_root, threshold_dir)
    with batch_tab:
        _render_batch(project_root, threshold_dir)


def _render_single(project_root: Path, threshold_dir: str) -> None:
    left, right = st.columns([1.1, 1.0])

    with left:
        st.subheader("1. Image and region")
        image_dir = directory_input_with_browse(
            label="Image directory",
            text_key="tab2_bgdet_image_dir",
            default_value=str(st.session_state.get("tab2_bgdet_image_dir", project_root / "outputs/rois")),
            browser_key_prefix="tab2_bgdet_image_dir",
            start_dir_fallback=str(project_root),
            help_text="Folder holding the ROI images to measure.",
        )
        images = _images_in(image_dir)
        if not images:
            st.warning("No images found in that folder.")
            return
        image_name = st.selectbox("Image", images, key="tab2_bgdet_image")
        image_path = str(Path(image_dir) / image_name)
        reader = _open(image_path)
        channels = list(reader.channel_names)
        st.caption(f"{len(channels)} channels · {reader.level_count} pyramid level(s) · {reader.level_shapes[0][0]}x{reader.level_shapes[0][1]}")

        annotation_dir = directory_input_with_browse(
            label="Tissue annotation folder (optional but recommended)",
            text_key="tab2_bgdet_annot_dir",
            default_value=str(st.session_state.get("tab2_bgdet_annot_dir", project_root / "outputs/processed_annotations")),
            browser_key_prefix="tab2_bgdet_annot_dir",
            start_dir_fallback=str(project_root),
            help_text=(
                "GeoJSON tissue polygons from the Annotation step. Measuring inside real tissue keeps empty "
                "slide out of the histogram, which otherwise drags the background estimate far too low."
            ),
        )
        stem = _stem(image_name)
        candidates = []
        if Path(annotation_dir).is_dir():
            candidates = sorted(f.name for f in Path(annotation_dir).iterdir()
                                if f.suffix == ".geojson" and stem in f.name)
        geojson_name = st.selectbox(
            "Tissue polygons", ["(none - automatic Otsu fallback)"] + candidates,
            index=1 if candidates else 0, key="tab2_bgdet_geojson",
        )
        geojson_path = "" if geojson_name.startswith("(none") else str(Path(annotation_dir) / geojson_name)

        classes: list[str] = []
        if geojson_path:
            names = _geojson_classes(geojson_path)
            if names:
                default = [n for n in names if n.lower() == "tissue"] or names[:1]
                classes = st.multiselect(
                    "Which polygon classes count as tissue", names, default=default,
                    key="tab2_bgdet_classes",
                    help="Usually the whole-tissue polygon. Restricting to a sub-region is useful for spot checks.",
                )
        min_component_area_px = float(st.number_input(
            "Ignore disconnected pieces smaller than (full-resolution px²)",
            min_value=0,
            value=int(DEFAULT_MIN_COMPONENT_AREA_PX),
            step=1_000,
            key="tab2_bgdet_min_component_area",
            disabled=not bool(geojson_path),
            help=(
                "Applied to each disconnected piece before rasterization. The area is measured "
                "in level-0 image pixels, so changing downsample does not change which pieces are kept."
            ),
        ))
        component_summary = None
        if geojson_path:
            component_summary = geojson_component_summary(
                geojson_path, classes or None, min_component_area_px
            )
            retained = (
                100.0 * float(component_summary["kept_area_px"])
                / max(float(component_summary["total_area_px"]), 1.0)
            )
            st.caption(
                f"{int(component_summary['kept_components'])} component(s) kept; "
                f"{int(component_summary['ignored_components'])} ignored; "
                f"{retained:.3f}% of selected annotation area retained."
            )

        st.subheader("2. Method")
        levels = image_levels(image_path)
        with st.expander("Image details", expanded=False):
            st.dataframe(levels, width="stretch", hide_index=True)
            st.caption(
                "Measuring on a lower level is much faster, but averaging narrows the pixel "
                "distribution, so a threshold at 5x is not identical to one at full resolution. "
                "Keep the level fixed across images you intend to compare."
            )
        downsample_options = [1, 2, 4, 5, 8, 16]
        downsample = int(st.select_slider(
            "Downsample", options=downsample_options, value=5,
            format_func=lambda factor: (
                f"{factor}x — {reader.target_shape(factor)[1]}x{reader.target_shape(factor)[0]}"
                + (" (Groovy default)" if factor == 5 else "")
            ),
            key="tab2_bgdet_downsample",
            help=(
                "The reference Groovy script uses thresholdDownsample=5. Use 5x when comparing "
                "values; changing resolution changes the pixel histogram."
            ),
        ))

        method_column, output_column = st.columns(2)
        method = method_column.selectbox(
            "Threshold method", list(THRESHOLD_METHODS),
            index=list(THRESHOLD_METHODS).index("Triangle"),
            key="tab2_bgdet_method",
            help=(
                "Triangle is the default because it reproduces manually gated thresholds most closely on "
                "CODEX panels. Geometric elbow finds the shoulder of the noise mode and returns markedly "
                "lower values. Compare a few markers before committing."
            ),
        )
        threshold_params = _threshold_parameters("tab2_threshold")
        output_label = output_column.selectbox(
            "CSV output",
            list(POLYGON_OUTPUT_OPTIONS),
            key="tab2_bgdet_polygon_output",
            help=(
                "The QuPath-compatible option averages annotation-object thresholds into one "
                "Polygon_1. Separate output retains one column per GeoJSON annotation feature."
            ),
        )
        output_layout = POLYGON_OUTPUT_OPTIONS[output_label]
        export_mode = "individual"
        if output_layout == "combined":
            merge_label = st.selectbox(
                "Combine using",
                list(POLYGON_MERGE_OPTIONS),
                key="tab2_bgdet_polygon_merge",
                help="Mean exactly matches the averaging in the Groovy reference script.",
            )
            export_mode = POLYGON_MERGE_OPTIONS[merge_label]
            st.caption("The CSV will contain one Polygon_1 value per marker.")
        else:
            st.caption(
                "The CSV will contain Polygon_1, Polygon_2, ...; one column for each "
                "selected GeoJSON annotation feature."
            )

        st.subheader("3. Detect")
        markers = st.multiselect(
            "Markers", channels, default=[c for c in channels if not c.upper().startswith("HE_")],
            key="tab2_bgdet_markers",
        )
        if st.button("Detect background for all selected markers", type="primary",
                     key="tab2_bgdet_run", width="stretch", disabled=not markers):
            progress = st.progress(0.0)
            status = st.empty()
            started = time.time()
            try:
                reference = _plane(image_path, os.path.getmtime(image_path), markers[0], downsample)
                mask, mask_note, source_mode, fallback_status = _build_mask(
                    image_path, geojson_path, classes, downsample, reference,
                    min_component_area_px,
                )
                st.session_state[MASK_KEY] = mask
                per_region = None
                effective_region_unit = "pooled"
                if geojson_path:
                    per_region = masks_from_geojson(
                        geojson_path, reference.shape, downsample, classes or None,
                        region_unit="feature",
                        min_component_area_px=min_component_area_px,
                    )
                    if not per_region:
                        raise ValueError(
                            "The selected polygons contain no pixels at this resolution. "
                            "Use a finer level or choose another class."
                        )
                    mask_note = _polygon_mask_note(
                        per_region, downsample, Path(geojson_path).name,
                        component_summary,
                    )
                    source_mode = "qupath_annotation_thresholds"
                    fallback_status = "none"
                    effective_region_unit = "feature"
                elif output_layout == "individual":
                    st.warning(
                        "Separate Polygon columns need a GeoJSON file. This image will use the "
                        "automatic mask and save one Polygon_1 value."
                    )
                status.write(f"Measuring inside {mask_note}")
                table = detect_backgrounds(
                    image_path, markers=markers, method=method, downsample=int(downsample), params=threshold_params,
                    mask=mask, region_masks=per_region,
                    aggregate=export_mode,
                    progress=lambda done, total, name: (
                        progress.progress(done / total), status.write(f"{name} ({done}/{total})")),
                )
                st.session_state["thresholding_results"] = {"image": str(image_path), "method": method, "parameters": threshold_params, "table": table}
                table.insert(0, "image", stem)
                table["manual_override"] = np.nan
                table["mask"] = mask_note
                table["source_mode"] = source_mode
                table["mask_fallback_status"] = fallback_status
                threshold_fallback = table["threshold_fallback_status"].fillna("none").astype(str)
                if fallback_status == "none":
                    table["fallback_status"] = threshold_fallback
                else:
                    table["fallback_status"] = threshold_fallback.map(
                        lambda value: f"mask={fallback_status}" if value == "none"
                        else f"mask={fallback_status}; threshold={value}"
                    )
                table["region_unit"] = effective_region_unit
                st.session_state[TABLE_KEY] = table
                status.empty()
                progress.empty()
                st.success(f"Measured {len(table)} marker(s) in {time.time() - started:.1f}s inside {mask_note}.")
                fallback_markers = table.loc[
                    table["threshold_fallback_status"].ne("none"), "Marker"
                ].astype(str).tolist()
                if fallback_markers:
                    st.warning(
                        "Threshold detection fell back to 150 for: " + ", ".join(fallback_markers)
                        + ". Review the method and enter manual overrides; export is blocked until these "
                          "fallbacks are resolved."
                    )
                record_process_event("tab2", "Background detection complete", status="complete",
                                     duration_s=time.time() - started,
                                     details={"image": stem, "method": method, "markers": len(table)})
            except Exception as error:
                status.empty()
                st.error(f"Detection failed: {error}")
                record_process_event("tab2", "Background detection failed", status="error", message=str(error))

        table = st.session_state.get(TABLE_KEY)
        if isinstance(table, pd.DataFrame) and not table.empty and table["image"].iloc[0] == stem:
            table = align_region_aggregate(table, export_mode=export_mode)
            table["final"] = table["manual_override"].fillna(table["threshold"])
            st.session_state[TABLE_KEY] = table
            st.subheader("4. Review and correct")
            st.caption(
                "Edit the override column to replace any automatic value. Blank means keep the detected one. "
                "For separate output, an override replaces all Polygon columns for that marker."
            )
            if "region_spread" in table.columns and table["region_spread"].max() > 3:
                worst = table.loc[table["region_spread"].idxmax()]
                st.warning(
                    f"Background varies up to {worst['region_spread']:.0f}x between annotations "
                    f"(worst: {worst['Marker']}). Check that you selected the correct polygon class. "
                    "For one combined value, Median or a lower percentile is less affected by a high polygon."
                )
            edited = st.data_editor(
                table[["Marker", "threshold", "manual_override", "peak", "n_pixels"]],
                key="tab2_bgdet_editor", width="stretch", hide_index=True,
                disabled=["Marker", "threshold", "peak", "n_pixels"],
                column_config={
                    "threshold": st.column_config.NumberColumn("detected", format="%.1f"),
                    "manual_override": st.column_config.NumberColumn("your value", format="%.1f"),
                    "peak": st.column_config.NumberColumn("noise peak", format="%.0f"),
                    "n_pixels": st.column_config.NumberColumn("pixels measured", format="%d"),
                },
            )
            merged = table.copy()
            merged["manual_override"] = edited["manual_override"].to_numpy()
            merged["final"] = merged["manual_override"].fillna(merged["threshold"])
            st.session_state[TABLE_KEY] = merged

            st.subheader("5. Save")
            st.caption(f"Threshold tables will be written to `{threshold_dir}`.")
            save_column, image_column = st.columns(2)
            if save_column.button("Save threshold table", key="tab2_bgdet_save_csv", width="stretch"):
                try:
                    path = write_threshold_csv(
                        merged,
                        Path(threshold_dir) / threshold_csv_name(stem),
                        stem,
                        export_mode=export_mode,
                    )
                    st.success(f"Saved {path}")
                    st.caption(
                        "RESOLVE will find this file automatically."
                    )
                except Exception as error:
                    st.error(f"Could not export thresholds: {error}")

            if image_column.button("Save output image", key="tab2_bgdet_save_image",
                                   width="stretch",
                                   help=(
                                       "Saves a pyramidal OME-TIFF with background removed and "
                                       "channel metadata preserved."
                                   )):
                output_dir = Path(st.session_state.get("tab2_output_dir",
                                                       project_root / "outputs/background_subtracted"))
                target = output_dir / f"{stem}_bg_subtracted.ome.tiff"
                progress = st.progress(0.0)
                status = st.empty()
                started = time.time()
                try:
                    prepared = prepare_threshold_export(merged, export_mode=export_mode)
                    thresholds = dict(zip(prepared["Marker"], prepared["threshold_final"].astype(float)))
                    subtract_background_image(
                        image_path, thresholds, target,
                        progress=lambda done, total, name: (
                            progress.progress(done / total), status.write(f"{name} ({done}/{total})")),
                    )
                    status.empty()
                    progress.empty()
                    st.success(f"Saved {target} in {time.time() - started:.0f}s")
                    record_process_event("tab2", "Background-subtracted image written", status="complete",
                                         duration_s=time.time() - started, details={"output": str(target)})
                except Exception as error:
                    status.empty()
                    st.error(f"Could not write the image: {error}")

    with right:
        st.subheader("Inspect one marker")
        table = st.session_state.get(TABLE_KEY)
        if not isinstance(table, pd.DataFrame) or table.empty:
            st.info("Run detection to inspect a marker here.")
            return
        marker = st.selectbox("Marker", table["Marker"].tolist(), key="tab2_bgdet_inspect")
        row = table.loc[table["Marker"].eq(marker)].iloc[0]
        detected = float(row["threshold"])
        current = float(row.get("final", detected))
        value = st.slider(
            "Threshold", 0.0, float(max(detected * 6.0, current * 2.0, 50.0)), float(current),
            step=max(round(detected / 100.0, 2), 0.1), key="tab2_bgdet_slider",
            help="Drag to test a different cut. The panels below redraw on the downsampled view, not full resolution.",
        )
        plane = _plane(image_path, os.path.getmtime(image_path), marker, downsample)
        mask = st.session_state.get(MASK_KEY)
        pixels = plane[mask] if isinstance(mask, np.ndarray) and mask.shape == plane.shape else plane.ravel()

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(5.4, 6.4))
        upper = float(np.percentile(pixels, 99.5)) if pixels.size else 1.0
        axes[0].hist(pixels[pixels <= upper], bins=150, color="#7f8c8d")
        axes[0].axvline(detected, color="#2980b9", ls="--", lw=1.6, label=f"detected {detected:.1f}")
        axes[0].axvline(value, color="#c0392b", ls="-", lw=1.6, label=f"in use {value:.1f}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("intensity")
        axes[0].set_ylabel("pixels (log)")
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title(f"{marker} — pixels measured", fontsize=10)
        kept = float((pixels > value).mean()) if pixels.size else 0.0
        axes[1].imshow(np.clip(plane.astype(np.float32) - value, 0, None),
                       cmap="magma", vmax=max(upper - value, 1.0))
        axes[1].set_title(f"after subtracting {value:.1f} — {kept:.1%} of pixels remain", fontsize=10)
        axes[1].axis("off")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        if abs(value - current) > 1e-9:
            if st.button("Use this value for " + marker, key="tab2_bgdet_apply", width="stretch"):
                updated = st.session_state[TABLE_KEY].copy()
                updated.loc[updated["Marker"].eq(marker), "manual_override"] = value
                updated["final"] = updated["manual_override"].fillna(updated["threshold"])
                st.session_state[TABLE_KEY] = updated
                st.rerun()


def _common_markers(paths: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Markers present in every selected image, plus what each one is missing.

    Reported rather than silently intersected: a marker absent from one image is
    usually a panel difference worth knowing about before it disappears from the
    run.
    """
    per_image = {}
    for path in paths:
        try:
            per_image[Path(path).name] = list(_open(path).channel_names)
        except Exception:
            per_image[Path(path).name] = []
    present = [set(v) for v in per_image.values() if v]
    common = sorted(set.intersection(*present)) if present else []
    missing = {name: sorted(set(common).symmetric_difference(channels) & set(channels))
               for name, channels in per_image.items()}
    return common, {k: v for k, v in missing.items() if v}


def _render_batch(project_root: Path, threshold_dir: str) -> None:
    st.markdown(
        "Apply one set of background-detection settings to multiple images."
    )

    image_dir = directory_input_with_browse(
        label="Image directory",
        text_key="tab2_bgdet_batch_dir",
        default_value=str(st.session_state.get("tab2_bgdet_batch_dir",
                                               st.session_state.get("tab2_bgdet_image_dir",
                                                                    project_root / "outputs/rois"))),
        browser_key_prefix="tab2_bgdet_batch_dir",
        start_dir_fallback=str(project_root),
    )
    available = _images_in(image_dir)
    if not available:
        st.warning("No images found in that folder.")
        return
    chosen = st.multiselect("Images", available, default=available, key="tab2_bgdet_batch_images")
    if not chosen:
        st.info("Select at least one image.")
        return
    paths = [str(Path(image_dir) / name) for name in chosen]

    common, differences = _common_markers(paths)
    if not common:
        st.error("These images share no channel names. Check that they come from the same panel.")
        return
    if differences:
        st.warning(
            "Some images carry channels the others do not; only the shared ones are offered. "
            + "; ".join(f"**{k}**: {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}"
                        for k, v in list(differences.items())[:4])
        )
    st.caption(f"{len(common)} channel(s) common to all {len(chosen)} image(s).")

    markers = st.multiselect(
        "Markers to measure", common,
        default=[c for c in common if not c.upper().startswith("HE_")],
        key="tab2_bgdet_batch_markers",
        help="Starts from every shared channel except the virtual H&E layers. Remove any you do not need.",
    )

    annotation_dir = directory_input_with_browse(
        label="Tissue annotation folder",
        text_key="tab2_bgdet_batch_annot",
        default_value=str(st.session_state.get("tab2_bgdet_batch_annot",
                                               project_root / "outputs/processed_annotations")),
        browser_key_prefix="tab2_bgdet_batch_annot",
        start_dir_fallback=str(project_root),
        help_text="Each image is matched to a GeoJSON whose name contains its stem. Unmatched images fall "
                  "back to an automatic Otsu mask, which is less reliable.",
    )
    matches = {}
    if Path(annotation_dir).is_dir():
        for name in chosen:
            stem = _stem(name)
            hits = sorted(f.name for f in Path(annotation_dir).iterdir()
                          if f.suffix == ".geojson" and stem in f.name)
            matches[name] = hits[0] if hits else ""

    classes_by_image: dict[str, list[str]] = {}
    for name in chosen:
        match = matches.get(name, "")
        geojson_path = str(Path(annotation_dir) / match) if match else ""
        classes_by_image[name] = _geojson_classes(geojson_path)
    available_classes = sorted({c for values in classes_by_image.values() for c in values})
    default_classes = [c for c in available_classes if c.lower() == "tissue"]
    if not default_classes and available_classes:
        default_classes = available_classes[:1]
    selected_classes = st.multiselect(
        "Polygon classes counted as tissue",
        available_classes,
        default=default_classes,
        key="tab2_bgdet_batch_classes",
        help=(
            "Classes are discovered from every matched GeoJSON. Leave empty to use every polygon in each "
            "matched file. A matched file with none of the selected classes fails explicitly rather than "
            "silently measuring the whole frame. Unmatched files use the Otsu fallback."
        ),
    )
    min_component_area_px = float(st.number_input(
        "Ignore disconnected pieces smaller than (full-resolution px²)",
        min_value=0,
        value=int(DEFAULT_MIN_COMPONENT_AREA_PX),
        step=1_000,
        key="tab2_bgdet_batch_min_component_area",
        help=(
            "Applied before rasterization and shared by every image. The default removes tiny "
            "annotation fragments while retaining the main tissue areas."
        ),
    ))
    component_summaries: dict[str, dict | None] = {}
    for name in chosen:
        match = matches.get(name, "")
        component_summaries[name] = (
            geojson_component_summary(
                Path(annotation_dir) / match,
                selected_classes or None,
                min_component_area_px,
            )
            if match else None
        )
    st.dataframe(
        pd.DataFrame([{
            "image": name,
            "tissue polygons": matches.get(name) or "(none - Otsu fallback)",
            "available classes": ", ".join(classes_by_image[name]) or "(none discovered)",
            "selected classes present": (
                ", ".join(c for c in selected_classes if c in classes_by_image[name])
                if selected_classes and matches.get(name)
                else ("all" if matches.get(name) else "Otsu fallback")
            ) or "(none - this image will fail)",
            "components kept": (
                int(component_summaries[name]["kept_components"])
                if component_summaries[name] else "—"
            ),
            "small components ignored": (
                int(component_summaries[name]["ignored_components"])
                if component_summaries[name] else "—"
            ),
        } for name in chosen]),
        width="stretch", hide_index=True,
    )

    settings = st.columns(2)
    method = settings[0].selectbox(
        "Threshold method", list(THRESHOLD_METHODS),
        index=list(THRESHOLD_METHODS).index("Triangle"), key="tab2_bgdet_batch_method")
    threshold_params = _threshold_parameters("tab2_batch_threshold")
    downsample = settings[1].select_slider(
        "Downsample", options=[1, 2, 4, 5, 8, 16], value=5,
        key="tab2_bgdet_batch_downsample",
        format_func=lambda factor: f"{factor}x" + (" (Groovy default)" if factor == 5 else ""),
        help="Use 5x to match thresholdDownsample=5 in the reference Groovy script.")
    batch_output_label = st.selectbox(
        "CSV output",
        list(POLYGON_OUTPUT_OPTIONS),
        key="tab2_bgdet_batch_polygon_output",
        help=(
            "The QuPath-compatible option averages annotation-object thresholds into one "
            "Polygon_1. Separate output keeps one column per GeoJSON annotation feature."
        ),
    )
    batch_output_layout = POLYGON_OUTPUT_OPTIONS[batch_output_label]
    batch_export_mode = "individual"
    if batch_output_layout == "combined":
        batch_merge_label = st.selectbox(
            "Combine using",
            list(POLYGON_MERGE_OPTIONS),
            key="tab2_bgdet_batch_polygon_merge",
            help="Mean exactly matches the averaging in the Groovy reference script.",
        )
        batch_export_mode = POLYGON_MERGE_OPTIONS[batch_merge_label]
        st.caption("Each image CSV will contain one Polygon_1 value per marker.")
    else:
        st.caption(
            "Each image CSV will contain Polygon_1, Polygon_2, ...; one column for each "
            "selected GeoJSON annotation feature."
        )
        unmatched = [name for name in chosen if not matches.get(name)]
        if unmatched:
            st.warning(
                "Images without a matching GeoJSON will have only Polygon_1: "
                + ", ".join(unmatched)
            )

    if st.button(f"Measure {len(chosen)} image(s)", type="primary", key="tab2_bgdet_batch_run",
                 width="stretch", disabled=not markers):
        progress = st.progress(0.0)
        status = st.empty()
        started = time.time()
        results, failures = [], []
        for position, name in enumerate(chosen):
            path = str(Path(image_dir) / name)
            stem = _stem(name)
            status.write(f"{stem} ({position + 1}/{len(chosen)})")
            try:
                reference = _plane(path, os.path.getmtime(path), markers[0], downsample)
                geojson = str(Path(annotation_dir) / matches[name]) if matches.get(name) else ""
                classes = list(selected_classes)
                mask, note, source_mode, fallback_status = _build_mask(
                    path, geojson, classes, downsample, reference,
                    min_component_area_px,
                )
                per_region = None
                effective_region_unit = "pooled"
                if geojson:
                    per_region = masks_from_geojson(
                        geojson, reference.shape, downsample, classes or None,
                        region_unit="feature",
                        min_component_area_px=min_component_area_px,
                    )
                    if not per_region:
                        raise ValueError(
                            "The selected polygons contain no pixels at this resolution. "
                            "Use a finer downsample or choose another class."
                        )
                    note = _polygon_mask_note(
                        per_region, downsample, Path(geojson).name,
                        component_summaries.get(name),
                    )
                    source_mode = "qupath_annotation_thresholds"
                    fallback_status = "none"
                    effective_region_unit = "feature"
                table = detect_backgrounds(
                    path, markers=markers, method=method, params=threshold_params,
                    downsample=int(downsample), mask=mask, region_masks=per_region,
                    aggregate=batch_export_mode,
                )
                table.insert(0, "image", stem)
                table["mask"] = note
                table["source_mode"] = source_mode
                table["mask_fallback_status"] = fallback_status
                threshold_fallback = table["threshold_fallback_status"].fillna("none").astype(str)
                if fallback_status == "none":
                    table["fallback_status"] = threshold_fallback
                else:
                    table["fallback_status"] = threshold_fallback.map(
                        lambda value: f"mask={fallback_status}" if value == "none"
                        else f"mask={fallback_status}; threshold={value}"
                    )
                table["region_unit"] = effective_region_unit
                table["manual_override"] = np.nan
                results.append(table)
            except Exception as error:
                failures.append(f"{stem}: {error}")
            progress.progress((position + 1) / len(chosen))
        status.empty()
        progress.empty()
        if results:
            combined = pd.concat(results, ignore_index=True)
            combined["final"] = combined["threshold"]
            st.session_state[BATCH_KEY] = combined
            st.session_state["thresholding_results"] = {"method": method, "parameters": threshold_params, "table": combined}
            st.success(f"Measured {combined['image'].nunique()} image(s) in {time.time() - started:.0f}s.")
            record_process_event("tab2", "Batch background detection complete", status="complete",
                                 duration_s=time.time() - started,
                                 details={"images": combined["image"].nunique(), "markers": len(markers)})
        for failure in failures:
            st.error(failure)

    combined = st.session_state.get(BATCH_KEY)
    if not isinstance(combined, pd.DataFrame) or combined.empty:
        return
    combined = align_region_aggregate(combined, export_mode=batch_export_mode)
    combined["final"] = combined["manual_override"].fillna(combined["threshold"])
    st.session_state[BATCH_KEY] = combined

    st.subheader("Review and override")
    st.caption(
        "Blank means keep the detected value. A manual value is required for any marker whose "
        "detector used the 150 fallback. For separate output, an override replaces all Polygon "
        "columns for that marker."
    )
    editable_columns = [
        "image", "Marker", "threshold", "manual_override",
        "threshold_fallback_status",
    ]
    edited = st.data_editor(
        combined[editable_columns],
        key="tab2_bgdet_batch_editor",
        width="stretch",
        hide_index=True,
        disabled=["image", "Marker", "threshold", "threshold_fallback_status"],
        column_config={
            "threshold": st.column_config.NumberColumn("detected", format="%.1f"),
            "manual_override": st.column_config.NumberColumn("your value", format="%.1f"),
            "threshold_fallback_status": st.column_config.TextColumn("detector status"),
        },
    )
    combined = combined.copy()
    combined["manual_override"] = pd.to_numeric(
        edited["manual_override"], errors="coerce"
    ).to_numpy()
    combined["final"] = combined["manual_override"].fillna(combined["threshold"])
    st.session_state[BATCH_KEY] = combined

    fallback_rows = combined["threshold_fallback_status"].fillna("none").ne("none")
    unresolved_fallback = fallback_rows & combined["manual_override"].isna()
    if unresolved_fallback.any():
        fallback_summary = combined.loc[unresolved_fallback, ["image", "Marker"]].astype(str)
        st.warning(
            "Threshold detection used the 150 fallback and still needs a manual override for "
            + ", ".join(
                f"{row.image}/{row.Marker}" for row in fallback_summary.itertuples(index=False)
            )
            + ". Threshold-table and background-subtracted-image export are blocked for those images."
        )
    elif fallback_rows.any():
        st.info(
            "Every detector fallback has a manual override. Export will use those overrides."
        )

    st.subheader("Detection details")
    provenance_columns = [
        c for c in ("image", "mask", "source_mode", "fallback_status", "region_unit")
        if c in combined.columns
    ]
    if provenance_columns:
        st.dataframe(
            combined[provenance_columns].drop_duplicates(),
            width="stretch", hide_index=True,
        )
    wide = combined.pivot_table(index="Marker", columns="image", values="threshold")
    spread = (wide.max(axis=1) / wide.min(axis=1).replace(0, np.nan))
    st.dataframe(wide.round(1), width="stretch")
    unstable = spread[spread > 5].sort_values(ascending=False)
    if not unstable.empty:
        st.warning(
            "These markers vary more than 5x between images: "
            + ", ".join(f"{m} ({v:.0f}x)" for m, v in unstable.head(6).items())
            + ". Large swings usually mean uneven staining or a mask that caught different tissue."
        )

    st.subheader("Save")
    st.caption(f"Batch threshold tables will be written to `{threshold_dir}`.")
    csv_column, image_column = st.columns(2)
    if csv_column.button("Save every threshold table", key="tab2_bgdet_batch_save_csv", width="stretch"):
        written, failures = [], []
        for stem, group in combined.groupby("image"):
            try:
                written.append(write_threshold_csv(
                    group,
                    Path(threshold_dir) / threshold_csv_name(str(stem)),
                    export_mode=batch_export_mode,
                ))
            except Exception as error:
                failures.append(f"{stem}: {error}")
        if written:
            st.success(f"Wrote {len(written)} table(s) to {threshold_dir}")
            st.code("\n".join(str(w) for w in written))
        for failure in failures:
            st.error(failure)

    image_dir_out = st.session_state.get("tab2_output_dir", project_root / "outputs/background_subtracted")
    if image_column.button("Save all output images", key="tab2_bgdet_batch_save_img",
                           width="stretch",
                           help=(
                               "Saves one pyramidal OME-TIFF per image with background removed, "
                               "values clipped at zero, and channel metadata preserved."
                           )):
        progress = st.progress(0.0)
        status = st.empty()
        started = time.time()
        written, failures = [], []
        stems = list(combined["image"].unique())
        for position, stem in enumerate(stems):
            source = next((str(Path(image_dir) / n) for n in chosen if _stem(n) == stem), None)
            if source is None:
                failures.append(f"{stem}: source image not found")
                continue
            group = combined.loc[combined["image"].eq(stem)]
            target = Path(image_dir_out) / f"{stem}_bg_subtracted.ome.tiff"
            try:
                prepared = prepare_threshold_export(group, export_mode=batch_export_mode)
                thresholds = dict(zip(prepared["Marker"], prepared["threshold_final"].astype(float)))
                status.write(f"{stem} ({position + 1}/{len(stems)})")
                subtract_background_image(
                    source, thresholds, target,
                    progress=lambda done, total, name, s=stem: status.write(
                        f"{s}: {name} ({done}/{total})"))
                written.append(target)
            except Exception as error:
                failures.append(f"{stem}: {error}")
            progress.progress((position + 1) / len(stems))
        status.empty()
        progress.empty()
        if written:
            st.success(f"Wrote {len(written)} image(s) in {time.time() - started:.0f}s")
            st.code("\n".join(str(w) for w in written))
        for failure in failures:
            st.error(failure)
