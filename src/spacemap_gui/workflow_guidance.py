"""Consistent, lightweight workflow guidance for every SPACEMAP pipeline page."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import streamlit as st


PIPELINE_GUIDES: dict[str, dict[str, object]] = {
    "tab0": {
        "title": "Project Workspace",
        "purpose": "Create or load an analysis project, then confirm the shared directory map before processing data.",
        "inputs": (),
        "outputs": (),
        "next_step": "Save the project after changing directories, then begin Registration.",
        "before_run": "Confirm that raw CODEX and H&E folders point to the intended study data.",
    },
    "tab1": {
        "title": "Registration",
        "purpose": "Align each CODEX image with its matching H&E image and create registered image stacks.",
        "inputs": (("CODEX images", "batch_codex_dir"), ("H&E images", "batch_he_dir")),
        "outputs": (("Registered images", "tab1_batch_output_dir"),),
        "next_step": "Use the registered images for Background Subtraction.",
        "before_run": "Confirm at least one matched pair is selected and that its CODEX and H&E MPP values are correct.",
    },
    "tab2": {
        "title": "Background Subtraction",
        "purpose": "Remove background signal from registered images or raw CODEX images.",
        "inputs": (("Registered or raw images", "tab2_input_dir_reg"),),
        "outputs": (("Background-subtracted images", "tab2_output_dir"),),
        "next_step": "Use the corrected images for ROI Extraction.",
        "before_run": "Select an image, detect channels, and confirm the blank-channel assignments.",
    },
    "tab3": {
        "title": "ROI Extraction",
        "purpose": "Extract regions of interest from corrected images using existing or drawn GeoJSON boundaries.",
        "inputs": (("Images", "tab3_input_dir"), ("ROI boundaries", "tab3_geojson_dir")),
        "outputs": (("ROI images", "tab3_output_dir"),),
        "next_step": "Use ROI images for Annotation, then Segmentation.",
        "before_run": "Confirm the selected image, boundary file, exported channels, and ROI output directory.",
    },
    "tab4": {
        "title": "Annotation",
        "purpose": "Discover tissue macro-structures from the imaging data, review them in QuPath, and export reviewed tissue annotations and zones.",
        "inputs": (("ROI images", "annot_ta_input_dir"), ("Segmentation features, cell annotation only", "cell_seg_dir")),
        "outputs": (("Tissue annotation GeoJSON", "annot_ta_output_dir"), ("Pooled annotations and zones", "annot_ta_pool_output_dir")),
        "next_step": "Use the reviewed tissue annotations for Segmentation and cell annotation.",
        "before_run": "Choose the ROI image, confirm the H&E channels were detected, and save each step's parameters before running it.",
    },
    "tab5": {
        "title": "Segmentation",
        "purpose": "Find cells inside the reviewed tissue, extract their features, and tag each cell with the tissue domain it falls in.",
        "inputs": (("ROI images", "tab4_input_dir"), ("Tissue annotations", "tab4_annot_geojson_dir")),
        "outputs": (("Segmentation results", "tab4_output_dir"),),
        "next_step": "The annotated cell CSV files go straight into Batch Correction.",
        "before_run": "Select images and a nuclear channel, then confirm the annotation match and the mask polygon class for every selected image.",
    },
    "tab6": {
        "title": "Batch Correction",
        "purpose": "Combine annotated cell CSV files, apply quality filters, and correct batch effects.",
        "inputs": (("Annotated cell CSV files", "bc_input_dir"),),
        "outputs": (("Batch-corrected CSV files", "bc_output_dir"),),
        "next_step": "Use the corrected CSV for Phenotyping, Spatial Analysis, and Visualization.",
        "before_run": "Scan files, select files and markers, then review filtration counts before running correction.",
    },
    "tab7": {
        "title": "Phenotyping",
        "purpose": "Assign cell phenotypes using clustering, thresholding, machine learning, or manual review.",
        "inputs": (("Batch-corrected CSV files", "global_pheno_dir"),),
        "outputs": (("Phenotyping outputs", "phenotyping_output_dir"),),
        "next_step": "Use phenotype assignments for Spatial Analysis and Visualization.",
        "before_run": "Choose the unified CSV and verify that selected marker columns are present before running a phenotyping method.",
    },
    "tab8": {
        "title": "Spatial Analysis",
        "purpose": "Measure spatial relationships, neighborhoods, density, interactions, triads, and clonality.",
        "inputs": (("Batch-corrected or phenotyped CSV", "spatial_input_dir"),),
        "outputs": (("Spatial-analysis outputs", "spatial_output_dir"),),
        "next_step": "Review spatial outputs or move to Visualization for publication figures.",
        "before_run": "Choose a unified CSV and confirm coordinate and phenotype columns in the selected analysis tool.",
    },
    "tab9": {
        "title": "Visualization",
        "purpose": "Create publication-ready figures and visual validation views from processed image and cell data.",
        "inputs": (("Batch-corrected or phenotyped CSV", "global_visualization_dir"), ("ROI images", "tab9_actual_image_dir")),
        "outputs": (("Figure outputs", "tab9_actual_output_dir"),),
        "next_step": "Export the selected figures and retain the Process Log with the analysis record.",
        "before_run": "Choose the unified CSV, verify image/overlay locations, and confirm the output folder before exporting figures.",
    },
}


def get_directory_readiness(directory_value: object) -> tuple[str, str]:
    """Return a short directory state without scanning its contents recursively."""
    if not directory_value or not str(directory_value).strip():
        return "Not configured", ""

    directory = Path(str(directory_value)).expanduser()
    try:
        if not directory.exists():
            return "Missing", str(directory)
        if not directory.is_dir():
            return "Not a directory", str(directory)
        try:
            next(directory.iterdir())
        except StopIteration:
            return "Empty", str(directory)
        return "Ready", str(directory)
    except OSError:
        return "Unavailable", str(directory)


def format_directory_path(directory: str, project_root: str | Path | None = None) -> str:
    """Format a directory compactly for the guide while keeping it recognizable."""
    if not directory:
        return ""

    path = Path(directory)
    if project_root:
        try:
            return f"project/{path.relative_to(Path(project_root))}"
        except ValueError:
            pass

    parts = path.parts
    if len(parts) > 4:
        return str(Path("...", *parts[-3:]))
    return str(path)


def _render_directory_list(items: Iterable[tuple[str, str]]) -> None:
    project_root = st.session_state.get("active_project_root")
    for label, session_key in items:
        status, directory = get_directory_readiness(st.session_state.get(session_key))
        st.markdown(f"**{label}:** {status}")
        if directory:
            st.caption(format_directory_path(directory, project_root))


def render_workflow_guidance(context: str) -> None:
    """Render the consistent top-of-page guide for the current pipeline stage."""
    guide = PIPELINE_GUIDES.get(context)
    if guide is None:
        return

    inputs = guide["inputs"]
    outputs = guide["outputs"]
    with st.expander(f"Step Guide: {guide['title']}", expanded=True):
        st.caption(str(guide["purpose"]))
        input_column, output_column, next_column = st.columns((1, 1, 1))
        with input_column:
            st.markdown("**Inputs**")
            if inputs:
                _render_directory_list(inputs)  # type: ignore[arg-type]
            else:
                st.caption("Project configuration and directory mapping.")
        with output_column:
            st.markdown("**Outputs**")
            if outputs:
                _render_directory_list(outputs)  # type: ignore[arg-type]
            else:
                st.caption("Saved project configuration and shared pipeline settings.")
        with next_column:
            st.markdown("**Next**")
            st.write(str(guide["next_step"]))

        st.info(f"Before running: {guide['before_run']}")