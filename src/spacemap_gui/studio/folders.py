"""Each Studio step's folders: where it reads from and writes to, shown and changed in the step itself.

A folder is shown as the step itself works it out (so what you see is what the step uses) and changed
through the project settings classic SPACEMAP uses for it; a folder another step sets (Phenotyping reads
Tab 3's ROI images, say) is shown with where to change it. Folders are inside the project; a new one is
made by the step when it first writes there.
"""

import os
from pathlib import Path

from .settings import project_lock, read_config

DEPTH = 3
SKIP = {"intermediate", "intermediate_images", "cache", "viewer_cache", "jobs", "__pycache__", "previous"}

# step -> [(label, "in"/"out", settings written, settings read, fallback, note when set elsewhere)]
FOLDERS = {
    "tab2": [
        ("Images", "in", ("tab2_bgdet_image_dir",), (), "outputs/rois", ""),
        ("Threshold tables", "out", ("threshold_dir",), ("threshold_dir",), "data/thresholds", ""),
        ("Background removed", "out", ("tab2_output_dir", "bg_subtraction_output_dir"), ("tab2_output_dir", "bg_subtraction_output_dir"),
         "outputs/background_subtracted", ""),
    ],
    "tab3": [
        ("Images", "in", ("tab3_input_dir",), (), "outputs/registered", ""),
        ("ROI files", "in", ("tab3_geojson_dir",), ("tab3_geojson_dir",), "data/boundaries", ""),
        ("ROI images", "out", ("tab3_output_dir", "roi_output_dir"), ("tab3_output_dir", "roi_output_dir"), "outputs/rois", ""),
    ],
    "tab4": [
        ("ROI images", "in", ("annot_ta_input_dir",), ("annot_ta_input_dir", "tab3_output_dir"), "outputs/rois", ""),
        ("Annotations", "out", ("annot_ta_output_dir",), ("annot_ta_output_dir",), "outputs/processed_annotations/stage0_tissue_annotation", ""),
        ("Zones", "out", ("annot_ta_pool_output_dir",), ("annot_ta_pool_output_dir",), "outputs/processed_annotations/stage1b_compartments_zones", ""),
    ],
    "tab5": [
        ("ROI images", "in", ("tab4_input_dir",), ("tab4_input_dir", "tab3_output_dir"), "outputs/rois", ""),
        ("Cell measurements", "out", ("tab4_output_dir", "cell_seg_dir"), ("tab4_output_dir", "cell_seg_dir"), "outputs/segmentation", ""),
        ("Cell outlines", "out", ("tab4_geojson_dir",), ("tab4_geojson_dir",), "outputs/segmentation/cellpose/boundaries", ""),
    ],
    "tab6": [
        ("Cell measurements", "in", ("bc_input_dir",), ("bc_input_dir",), "outputs/annotated_csv", ""),
        ("Combined measurements", "out", ("bc_output_dir",), ("bc_output_dir",), "outputs/batch_corrected", ""),
    ],
    "tab7": [
        ("ROI images", "in", (), ("tab3_output_dir",), "outputs/rois", "Set in the Regions of interest step (its ROI images)."),
        ("Phenotyping results", "out", ("phenotyping_output_dir",), ("phenotyping_output_dir",), "outputs/phenotyping", ""),
    ],
    "tab8": [
        ("Measurements", "in", ("spatial_input_dir",), ("spatial_input_dir", "bc_output_dir"), "outputs/batch_corrected", ""),
        ("Spatial results", "out", ("spatial_output_dir",), ("spatial_output_dir",), "outputs/spatial_analysis", ""),
    ],
    "tab9": [
        ("Measurements", "in", ("global_visualization_dir",), ("global_visualization_dir", "bc_output_dir"), "outputs/batch_corrected", ""),
        ("Figures", "out", ("tab9_actual_output_dir",), ("tab9_actual_output_dir",), "outputs/spatial_analysis", ""),
    ],
}


def _mappings(root, config):
    return {
        key: str(value).replace("${PROJECT_ROOT}", str(root))
        for key, value in ((config or {}).get("directory_mappings") or {}).items()
        if isinstance(value, str) and value
    }


def _shown(root, folder):
    try:
        return str(Path(folder).resolve().relative_to(Path(root).resolve())) or "."
    except ValueError:
        return str(folder)


def _current(root, config, step, label, read, fallback):
    """The folder the step itself uses (Background's and ROI's own rules for their images)."""
    from . import background, rois
    from .steps import _folder

    mappings = _mappings(root, config)
    if step == "tab2" and label == "Images":
        return background.folders(root, mappings, config)[0]
    if step == "tab3" and label == "Images":
        return rois.folders(root, mappings, config)[0]
    return _folder(root, mappings, read, fallback)


def project_folders(root):
    """The project's folders a few levels deep (caches and job folders left out), relative to the project."""
    root = Path(root).resolve()
    found = []
    for current, names, _ in os.walk(root):
        level = len(Path(current).relative_to(root).parts)
        names[:] = sorted(name for name in names if not name.startswith(".") and name not in SKIP)
        if level >= DEPTH:
            names[:] = []
        if level:
            found.append(str(Path(current).relative_to(root)))
    return found[:400]


def listing(root, step):
    """The step's folders: label, in or out, the folder in use, whether it exists and can be changed here."""
    if step not in FOLDERS:
        return {"step": step, "folders": []}
    config = read_config(root)
    rows = []
    for label, role, write, read, fallback, note in FOLDERS[step]:
        folder = _current(root, config, step, label, read, fallback)
        rows.append({"label": label, "role": role, "folder": _shown(root, folder), "exists": Path(folder).is_dir(),
                     "editable": bool(write), "note": note})
    return {"step": step, "folders": rows, "choices": project_folders(root)}


def change(root, step, label, value):
    """Point one of the step's folders at *value* (inside the project), as classic SPACEMAP saves it."""
    from spacemap_gui import persistence

    entry = next((item for item in FOLDERS.get(step, []) if item[0] == label), None)
    if entry is None or not entry[2]:
        raise ValueError("This folder cannot be changed here.")
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("Choose a folder inside the project.")
    value = value.strip()
    if Path(value).is_absolute() or value.startswith("~"):
        raise ValueError("Choose a folder inside the project.")
    value = value.rstrip("/")
    base = Path(root).resolve()
    target = (base / value).resolve()
    if base != target and base not in target.parents:
        raise ValueError("Choose a folder inside the project.")
    if entry[1] == "in" and not target.is_dir():
        raise ValueError(f"{value} does not exist: a step reads from a folder that is there.")
    with project_lock(root):
        config = read_config(root)
        for key in entry[2]:
            persistence.update_stored_config_key(config, key, "${PROJECT_ROOT}/" + Path(value).as_posix())
        persistence.save_project_config(str(root), config)
    return listing(root, step)
