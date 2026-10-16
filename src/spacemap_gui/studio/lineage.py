"""The project's lineage: every ROI traced back to its slide and to the images that made the slide.

Each step names its outputs after its inputs, so the chain is read from the files
the steps have written, for any project, with nothing to set up:

* Registration (Tab 1) pairs ``<slide>_codex`` with ``<slide>_he`` and writes
  ``<slide>_merged.ome.tiff``.
* The ROI step (Tab 3) keeps the slide's ROIs in
  ``<slide>_roi_bounding_boxes_coords.geojson`` and cuts each to
  ``<slide>_<ROI>.ome.tiff``.
* Segmentation (Tab 5) writes one cell table per ROI image, named after it.
* The combined table (Tab 6) keeps each cell's table as its batch.

The parent slide is available as a correction unit; the user must decide whether
it represents the technical batch in their experiment. An ROI found in no ROI file gets its slide from its name
(``<slide>_roi_<n>``) and is marked so, for the user to check.
"""

from pathlib import Path
import re

ROI_FILE = "_roi_bounding_boxes_coords.geojson"
BY_NAME = re.compile(r"_roi_?\d+$", re.IGNORECASE)
_names = {}  # ROI file -> (stamp, ROI names): reading a large ROI file takes a second


def _roi_names(path):
    from . import rois

    info = path.stat()
    stamp = (info.st_mtime_ns, info.st_size)
    cached = _names.get(str(path))
    if cached is None or cached[0] != stamp:
        cached = _names[str(path)] = (stamp, rois.roi_names(path))
    return cached[1]


def _path(root, value):
    path = Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()
    return path if path.is_absolute() else Path(root) / path


def _folder(root, mappings, keys, fallback):
    value = next((mappings[key] for key in keys if (mappings or {}).get(key)), None)
    return _path(root, value) if value else Path(root) / fallback


def _files(folder, suffixes):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.name.lower().endswith(suffixes))


def cell_tables(root, mappings):
    """The segmentation's cell tables, one per ROI image: {ROI: path}."""
    from .cell_inputs import table_index

    folder = _folder(root, mappings, ("tab4_output_dir", "segmentation_output_dir"), "outputs/segmentation")
    return {roi: paths[0] for roi, paths in table_index(folder).items() if len(paths) == 1}


def trace(root, mappings):
    """Every slide with its source images and ROIs, and every ROI with its slide, image and cell table."""
    from spacemap.viewer.catalog import sample_of

    from . import registration, rois

    root = Path(root)
    pairs = registration.pairs(root, mappings)
    _, _, registered = registration.folders(root, mappings)
    from .settings import read_config

    config = read_config(root)
    _, shapes, roi_images = rois.folders(root, mappings, config)
    slides = {}

    def slide(name):
        if name not in slides:
            pair = pairs.get(name, {})
            merged = registration.output_for(registered, name)
            slides[name] = {
                "slide": name,
                "codex": Path(pair["codex"]).name if pair.get("codex") else "",
                "he": Path(pair["he"]).name if pair.get("he") else "",
                "registered": merged.name if merged.is_file() else "",
                "rois": [],
            }
        return slides[name]

    for name in pairs:
        slide(name)
    found = {}
    # A manually matched ROI file can have any name. Its parent is the image selected
    # in the ROI step, not the GeoJSON's filename and not an underscore-split heuristic.
    for image, filename in (rois._saved(config, rois.FILES_KEY, {}) or {}).items():
        path = Path(shapes) / filename
        if not path.is_file():
            continue
        name = rois.stem(image)
        for label in _roi_names(path):
            roi = f"{name}_{rois.safe_label(label)}"
            found[roi] = {"roi": roi, "slide": name, "label": label, "found": "roi step", "source": path.name}
    for path in _files(shapes, (ROI_FILE,)):
        name = path.name[: -len(ROI_FILE)]
        for label in _roi_names(path):
            roi = f"{name}_{rois.safe_label(label)}"
            found.setdefault(roi, {"roi": roi, "slide": name, "label": label, "found": "roi step", "source": path.name})
    images = {sample_of(path.name): path.name for path in _files(roi_images, (".ome.tiff", ".ome.tif"))}
    from . import cell_inputs
    if (mappings or {}).get("bc_images_dir"):
        images.update({roi: Path(path).name for roi, path in cell_inputs.images(root, mappings).items()})
    tables = cell_tables(root, mappings)
    if (mappings or {}).get("bc_input_dir"):
        for roi, paths in cell_inputs.table_index(_path(root, mappings["bc_input_dir"])).items():
            if len(paths) == 1:
                tables[roi] = paths[0]
    for roi in sorted((set(images) | set(tables)) - set(found)):
        if roi not in found:
            by_name = BY_NAME.sub("", roi)
            found[roi] = {"roi": roi, "slide": by_name if by_name != roi else roi, "label": "", "found": "name"}
    for roi, item in sorted(found.items()):
        item["image"] = images.get(roi, "")
        item["table"] = Path(tables[roi]).name if roi in tables else ""
        slide(item["slide"])["rois"].append(roi)
    return {"slides": [slides[name] for name in sorted(slides)], "rois": [found[roi] for roi in sorted(found)]}


def slide_of(root, mappings):
    """{ROI: slide} for every ROI the project knows."""
    return {item["roi"]: item["slide"] for item in trace(root, mappings)["rois"]}
