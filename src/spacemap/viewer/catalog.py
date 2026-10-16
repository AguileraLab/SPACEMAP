"""Everything a project can show in the viewer, gathered from the tabs' output folders.

Folders come from the project file's directory settings where they exist, then
the standard project layout. Files are linked by sample name: ``P11_codex.qptiff``,
``P11_he.qptiff`` and ``P11_merged.ome.tiff`` are sample ``P11``; ROI ``s4r2``
links ``s4r2.ome.tiff``, ``s4r2.ome_cell_boundaries.geojson`` and the annotation
files that start with ``s4r2``, and its cells are the table rows whose
``batch_id`` is ``s4r2``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .sources import is_viewable

#: (group shown to the user, directory-setting keys, fallback folder in the project)
IMAGE_GROUPS = [
    ("Raw CODEX", ("batch_codex_dir",), "data/codex"),
    ("Raw H&E", ("batch_he_dir",), "data/he"),
    ("Registered", ("tab1_batch_output_dir", "registration_output_dir"), "outputs/registered"),
    ("Registration intermediates", (), "outputs/registered/intermediate_images"),
    ("Background subtracted", ("tab2_output_dir", "bg_subtraction_output_dir"), "outputs/background_subtracted"),
    ("ROIs", ("tab3_output_dir", "roi_output_dir"), "outputs/rois"),
]
# Tab 5 writes cell outlines to ``tab4_geojson_dir`` (a legacy name). Not
# ``batch_geojson_dir``: that is Tab 3's ROI boundary folder, whose files carry a
# sample's name but hold ROIs, not cells.
SEGMENTATION_FOLDERS = (
    ("tab4_geojson_dir", "cell_seg_dir"),
    ("outputs/segmentation/geojson", "outputs/segmentation/cellpose/boundaries", "outputs/segmentation"),
)
ANNOTATION_FOLDERS = (
    ("annot_output_dir", "tab4_annot_geojson_dir", "annot_ta_output_dir", "zone_output_dir"),
    (
        "outputs/annotations",
        "outputs/processed_annotations",
        "outputs/tissue_classification",
        "outputs/tissue_annotation",
        "outputs/tissue_annotation2",
        "outputs/tissue_annotation3",
        "outputs/tissue_annotation_professional",
        "data/annotations",
    ),
)
_SAMPLE_SUFFIXES = (
    "_cell_features_tagged",
    "_cell_boundaries",
    "_cell_features",
    ".ome",
    "_codex",
    "_he",
    "_merged",
    "_all_good",
    "_registered",
    "_bg_subtracted",
    "_fully_annotated",
    "_annotated",  # Tab 5's cell annotation: <roi>.ome_cell_features_annotated.csv
    "_tissue_classes",
    "_tissue_boundary",
    "_macro_domains",
    "_raw_macro_clusters",
    "_raw_clusters",
)


@dataclass
class ImageEntry:
    group: str
    name: str
    path: str
    sample: str


@dataclass
class ImageBundle:
    """One image with everything that can be drawn on it."""

    image: ImageEntry
    segmentation: str | None = None
    annotations: list[tuple[str, str]] = field(default_factory=list)
    cell_tables: list[tuple[str, str]] = field(default_factory=list)


def sample_of(filename: str) -> str:
    """Sample or ROI name from a file name, e.g. ``P11_codex.qptiff`` -> ``P11``."""
    name = Path(filename).name
    name = re.split(r" - ", name, maxsplit=1)[0]  # QuPath: "s10r1.ome.tiff - Merged Image.geojson"
    for suffix in (".parquet", ".geojson", ".json", ".qptiff", ".tiff", ".tif", ".svs", ".csv"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    changed = True
    while changed:
        changed = False
        for suffix in _SAMPLE_SUFFIXES:
            if name.lower().endswith(suffix.lower()):
                name = name[: -len(suffix)]
                changed = True
    return name


def _folders(project_root: Path, mappings: dict, keys, fallbacks) -> list[Path]:
    found: list[Path] = []
    for key in keys:
        value = str((mappings or {}).get(key) or "").strip()
        if value:
            found.append(Path(value))
    for fallback in [fallbacks] if isinstance(fallbacks, str) else fallbacks:
        found.append(project_root / fallback)
    unique: list[Path] = []
    for folder in found:
        if folder.is_dir() and folder.resolve() not in [u.resolve() for u in unique]:
            unique.append(folder)
    return unique


def list_images(project_root: str | Path, mappings: dict | None = None) -> list[ImageEntry]:
    """Every image of the project, in pipeline order, each under the step that made it.

    When one folder is set as the output of several steps (for example Tab 2's
    output pointing at Tab 3's ROI folder), its images belong to the latest of
    those steps: they are what that step produced.
    """
    project_root = Path(project_root)
    claimed: dict[str, tuple[int, ImageEntry]] = {}
    for rank, (group, keys, fallback) in reversed(list(enumerate(IMAGE_GROUPS))):
        for folder in _folders(project_root, mappings or {}, keys, fallback):
            for path in sorted(folder.iterdir()):
                if not path.is_file() or not is_viewable(path):
                    continue
                resolved = str(path.resolve())
                if resolved not in claimed:
                    claimed[resolved] = (rank, ImageEntry(group, path.name, str(path), sample_of(path.name)))
    return [entry for _, entry in sorted(claimed.values(), key=lambda item: (item[0], item[1].name))]


_CELL_LABEL = re.compile(rb'"(?:label|cell_label)"\s*:\s*\d')


def holds_cells(path: str | Path) -> bool:
    """True when a GeoJSON's first features carry cell numbers (``label``), as Tab 5 writes them."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(256 * 1024)
    except OSError:
        return False
    return bool(_CELL_LABEL.search(head))


def find_segmentation(project_root: str | Path, mappings: dict | None, sample: str) -> str | None:
    """The cell segmentation (Tab 5 outlines) for *sample*, or None.

    Files named as cell outlines are preferred in every folder before any other
    name is considered, and a file is only taken if it holds numbered cells.
    """
    keys, fallbacks = SEGMENTATION_FOLDERS
    matches = []
    folders = _folders(Path(project_root), mappings or {}, keys, fallbacks)
    output = str((mappings or {}).get("tab4_output_dir") or "").replace("${PROJECT_ROOT}", str(project_root)).strip()
    if output and (Path(output) / "geojson").is_dir():
        folders.insert(0, Path(output) / "geojson")  # where Tab 5 writes the outlines of its output folder
    for folder in folders:
        matches += [path for path in sorted(folder.glob(f"{sample}*.geojson")) if sample_of(path.name) == sample]
    for path in sorted(matches, key=lambda p: "cell" not in p.name.lower()):
        if holds_cells(path):
            return str(path)
    return None


def annotation_files(folder: str | Path):
    """Current GeoJSON files, without descending into Studio history or caches.

    History stores a directory named after each annotation, including its
    .geojson suffix. Matching the suffix alone can return directories as layers.
    Pruning also avoids scanning every saved revision on each image open.
    """
    folder = Path(folder)
    internal = {"studio_history", "viewer_cache"}
    if internal.intersection(folder.parts):
        return
    for directory, folders, files in os.walk(folder):
        folders[:] = [name for name in folders if name not in internal]
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() == ".geojson" and path.is_file():
                yield path


def find_annotations(project_root: str | Path, mappings: dict | None, sample: str) -> list[tuple[str, str]]:
    """(label, path) of every annotation file for *sample*, labelled by folder."""
    keys, fallbacks = ANNOTATION_FOLDERS
    project_root = Path(project_root)
    results: list[tuple[str, str]] = []
    for folder in _folders(project_root, mappings or {}, keys, fallbacks):
        for path in sorted(annotation_files(folder)):
            if sample_of(path.name) != sample or "cell_boundaries" in path.name:
                continue
            try:
                where = path.parent.relative_to(project_root)
            except ValueError:
                where = path.parent
            results.append((f"{where} / {path.name}", str(path)))
    return results


def find_cell_tables(project_root: str | Path, mappings: dict | None, unified_csv: str | None) -> list[tuple[str, str]]:
    """Per-cell tables that can colour cells: the unified table and spatial results."""
    project_root = Path(project_root)
    tables: list[tuple[str, str]] = []
    if unified_csv and Path(unified_csv).is_file():
        tables.append(("Unified cell table", str(unified_csv)))
    for path in sorted((project_root / "outputs" / "spatial_analysis").rglob("*cell_level*.csv")):
        tables.append((f"Spatial analysis / {path.name}", str(path)))
    return tables


def background_image_for(project_root: str | Path, mappings: dict | None, entry: ImageEntry) -> str | None:
    """A matching Tab 2 output, identified by its explicit output suffix.

    Folder membership alone is insufficient: projects can point Tab 2 at their
    ROI folder even when those images have not had background removed.
    """
    if "_bg_subtracted.ome." in entry.name.lower():
        return entry.path
    _, keys, fallback = next(group for group in IMAGE_GROUPS if group[0] == "Background subtracted")
    matches = {}
    for folder in _folders(Path(project_root), mappings or {}, keys, fallback):
        for path in folder.glob("*_bg_subtracted.ome.tif*"):
            if path.is_file() and sample_of(path.name) == entry.sample:
                matches[str(path.resolve())] = str(path)
    # Different candidates need an explicit user choice rather than guessing.
    return next(iter(matches.values())) if len(matches) == 1 else None


def find_point_tables(project_root: str | Path) -> list[tuple[str, str]]:
    """Tables of positions that are not cells, for density maps: triads found by Tab 8."""
    base = Path(project_root) / "outputs" / "spatial_analysis"
    if not base.is_dir():
        return []
    return [(f"Triads / {path.name}", str(path)) for path in sorted(base.rglob("triad_instances*.csv"))]


def bundle_for(
    project_root: str | Path, mappings: dict | None, entry: ImageEntry, unified_csv: str | None
) -> ImageBundle:
    bundle = ImageBundle(entry)
    bundle.segmentation = find_segmentation(project_root, mappings, entry.sample)
    bundle.annotations = find_annotations(project_root, mappings, entry.sample)
    if bundle.segmentation:
        bundle.cell_tables = find_cell_tables(project_root, mappings, unified_csv)
    return bundle


# What each tab most likely wants to see first. Tabs are named as app.py names them.
_START_GROUP = {"tab0": "ROIs", "tab1": "Registered", "tab2": "Background subtracted"}
_NEIGHBOURHOOD_COLUMNS = (
    "neighborhood",
    "neighbourhood",
    "CN_name_merged",
    "cn_name_merged",
    "CN_name_original",
    "cn_name_original",
    "neighborhood_cluster",
    "CN_cluster_merged",
    "CN_cluster",
    "cn_cluster",
)
_REFERENCE = re.compile(r"(ref|manual)", re.IGNORECASE)


def start_group(context: str) -> str:
    """The image group the picker starts on for a tab."""
    return _START_GROUP.get(context, "ROIs")


def suggest_colouring(context: str, tables: list[tuple[str, list[str]]]) -> tuple[int, str]:
    """(table index, column) the cells are coloured by when the viewer opens from *context*.

    Tabs 7 and 9 show the newest phenotype column, Tab 8 the neighbourhoods;
    elsewhere cells are outlines only (``(0, "")``).
    """
    if context in ("tab7", "tab9"):
        for index, (_, columns) in enumerate(tables):
            phenotypes = [c for c in columns if c.startswith("phenotype_") and not _REFERENCE.search(c)]
            if phenotypes:
                return index, phenotypes[-1]
            if "consensus" in columns:
                return index, "consensus"
    if context == "tab8":
        ranked = sorted(range(len(tables)), key=lambda i: not re.search(r"neighbo", tables[i][0], re.IGNORECASE))
        for index in ranked:
            columns = tables[index][1]
            for name in _NEIGHBOURHOOD_COLUMNS:
                if name in columns:
                    return index, name
            loose = [c for c in columns if re.match(r"(cn_|neighbo)", c, re.IGNORECASE)]
            if loose:
                return index, loose[0]
    return 0, ""


# Images added by hand from outside the project (a scan elsewhere on the
# cluster, a collaborator's file). Kept per project in the user's SPACEMAP
# state folder, so nothing is written into the project itself.
ADDED_GROUP = "Added images"


def _added_store(project_root: str | Path | None) -> Path:
    """The list of added images: in the project (so it follows the project), else in the user's settings."""
    if project_root:
        return Path(project_root) / "outputs" / "viewer" / "added_images.json"
    from spacemap.userpaths import state_dir

    return Path(state_dir()) / "viewer_added_images.json"


def _added_paths(project_root: str | Path | None) -> list[str]:
    import json

    try:
        data = json.loads(_added_store(project_root).read_text())
    except (OSError, ValueError):
        return []
    images = data.get("images") if isinstance(data, dict) else None
    return [str(p) for p in images] if isinstance(images, list) else []


def _added_write(project_root: str | Path | None, paths: list[str]) -> None:
    import json
    import os

    path = _added_store(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"images": paths}, indent=1))
    os.replace(temporary, path)


def added_images(project_root: str | Path | None) -> list[ImageEntry]:
    """Images added to this project's list by hand (files that no longer exist are skipped)."""
    return [
        ImageEntry(ADDED_GROUP, Path(p).name, p, sample_of(Path(p).name))
        for p in _added_paths(project_root)
        if Path(p).is_file()
    ]


def add_image(project_root: str | Path | None, path: str | Path) -> ImageEntry:
    """Add an image from anywhere to this project's list. Raises ValueError if it cannot be shown."""
    from .sources import image_info

    candidate = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if not candidate.is_file():
        raise ValueError(f"No file at {candidate}.")
    if not is_viewable(candidate):
        raise ValueError(f"{candidate.name} is not an image the viewer can open (QPTIFF, OME-TIFF, TIFF or SVS).")
    try:
        image_info(candidate)
    except Exception as exc:
        raise ValueError(f"{candidate.name} could not be read as an image: {exc}") from exc
    resolved = str(candidate.resolve())
    listed = _added_paths(project_root)
    if resolved not in listed:
        _added_write(project_root, listed + [resolved])
    return ImageEntry(ADDED_GROUP, candidate.name, resolved, sample_of(candidate.name))


def remove_image(project_root: str | Path | None, path: str | Path) -> None:
    """Take an added image off the list (the file itself is not touched)."""
    listed = _added_paths(project_root)
    kept = [p for p in listed if p != str(Path(path).resolve())]
    if kept != listed:
        _added_write(project_root, kept)
