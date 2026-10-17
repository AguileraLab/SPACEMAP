"""The cell segmentations an image shows in Layers: the Segmentation step's own result first, then any the user
imports (another run, a colleague's outlines). Imported ones are remembered per image in
``outputs/studio/imported_cells/<image-key>.json`` and can be taken off again; their files are never changed."""

import json
from pathlib import Path


def _record(root, image):
    from .annotation_inputs import image_key

    return Path(root) / "outputs/studio/imported_cells" / (image_key(image) + ".json")


def imported(root, image):
    """The segmentation files imported for *image* that are still there."""
    try:
        paths = json.loads(_record(root, image).read_text())
    except (OSError, ValueError):
        return []
    return [path for path in paths if isinstance(path, str) and Path(path).is_file()]


def _save(root, image, paths):
    from .annotation_edits import _atomic
    from .settings import project_lock

    with project_lock(root):
        _atomic(_record(root, image), json.dumps(list(dict.fromkeys(paths)), indent=2).encode())


def own(root, config, mappings, image, sample):
    """The Segmentation step's outlines for this image (its Outlines folder), when they exist."""
    from spacemap.viewer.catalog import holds_cells, sample_of

    from . import segmentation

    folder = segmentation.step_folders(root, config, mappings)["geojson"]
    exact = segmentation.results(folder.parent, Path(image).name, folder)[1]
    if exact.is_file() and holds_cells(exact):
        return str(exact)
    if folder.is_dir():
        for path in sorted(folder.glob(f"{sample}*.geojson"), key=lambda p: "cell" not in p.name.lower()):
            if sample_of(path.name) == sample and holds_cells(path):
                return str(path)
    return None


def candidates(root, config, mappings, image, sample):
    """Segmentation files of this sample the project holds (Segmentation's folders, classic Tab 5's, other runs under
    outputs/), newest first, as {"name": shown name, "path"}."""
    from spacemap.viewer.catalog import SEGMENTATION_FOLDERS, _folders, holds_cells, sample_of

    from . import segmentation

    places = segmentation.step_folders(root, config, mappings)
    folders = [places["geojson"], places["output"] / "geojson", places["output"]]
    resolved = {k: str(v).replace("${PROJECT_ROOT}", str(root)) for k, v in (mappings or {}).items() if isinstance(v, str)}
    folders += _folders(Path(root), resolved, *SEGMENTATION_FOLDERS)
    outputs = Path(root) / "outputs"
    if outputs.is_dir():
        for top in outputs.iterdir():
            if top.is_dir() and "segment" in top.name.lower():
                folders += [top, top / "geojson"] + [p for p in top.glob("*/geojson") if p.is_dir()]
    seen, found = set(), []
    for folder in folders:
        if not Path(folder).is_dir():
            continue
        for path in Path(folder).glob(f"{sample}*.geojson"):
            key = str(path.resolve())
            if key in seen or sample_of(path.name) != sample or "previous" in path.parts:
                continue
            seen.add(key)
            if holds_cells(path):
                found.append(path)
    found.sort(key=lambda p: -p.stat().st_mtime)
    shown = lambda p: str(p.relative_to(root)) if Path(root) in p.parents else str(p)
    return [{"name": shown(p), "path": str(p.resolve())} for p in found]


def _tables_record(root, image):
    return _record(root, image).with_suffix(".tables.json")


def _cells_name(path):
    """An outlines or cell table file's name without Studio's drop prefix or the cell suffixes, to pair them:
    1a2b3c4d_P11_roi_1.ome_cell_boundaries.geojson and P11_roi_1.ome_cell_features.csv are both p11_roi_1.ome."""
    import re

    name = re.sub(r"^[0-9a-f]{8}_", "", Path(str(path)).name).lower()
    return re.sub(r"(_cell_boundaries|_cell_features|_cells|_boundaries|_features)?\.(geojson|json|csv|tsv|parquet)$", "", name)


def table_of(root, image, path):
    """The cell table of an imported segmentation: the one given with it (dropped together), else the table of the
    same name next to it (<run>/<image>_cell_features.csv beside <run>/geojson/), else None."""
    try:
        chosen = json.loads(_tables_record(root, image).read_text()).get(str(Path(path).resolve()))
    except (OSError, ValueError, AttributeError):
        chosen = None
    if chosen and Path(chosen).is_file():
        return chosen
    outlines = Path(path)
    places = [outlines.parent] + ([outlines.parent.parent] if outlines.parent.name == "geojson" else [])
    for folder in places:
        for found in sorted(folder.glob("*")):
            if found.suffix.lower() in (".csv", ".tsv", ".parquet") and _cells_name(found) == _cells_name(outlines):
                return str(found)
    return None


def add(root, image, path, table=None):
    from spacemap.viewer.catalog import holds_cells

    path = Path(path).expanduser()
    if not path.is_file() or path.suffix.lower() != ".geojson":
        raise ValueError("Choose a GeoJSON file of cell outlines.")
    if not holds_cells(path):
        raise ValueError(f"{path.name} has no numbered cells (each outline needs a label).")
    _save(root, image, imported(root, image) + [str(path.resolve())])
    if table:
        from .annotation_edits import _atomic

        if not Path(table).is_file():
            raise ValueError(f"{Path(table).name} is not there any more.")
        try:
            known = json.loads(_tables_record(root, image).read_text())
        except (OSError, ValueError):
            known = {}
        known[str(path.resolve())] = str(Path(table).resolve())
        _atomic(_tables_record(root, image), json.dumps(known, indent=2).encode())
    return str(path.resolve())


def remove(root, image, path):
    _save(root, image, [p for p in imported(root, image) if Path(p).resolve() != Path(path).resolve()])


def _with_roi(table, roi, cache):
    """A per-image cell table as the viewer reads tables (cells found by ROI and label): the table itself when it
    has a batch_id column, else a cached Parquet copy that adds it (made again when the table changes)."""
    from hashlib import sha256

    import pandas as pd

    from spacemap import dataio

    table = Path(table)
    if "batch_id" in dataio.columns(str(table)):
        return str(table)
    stamp = table.stat()
    target = Path(cache) / "roi_tables" / (sha256(f"{table.resolve()}|{stamp.st_mtime_ns}|{stamp.st_size}".encode()).hexdigest()[:24] + ".parquet")
    if not target.is_file():
        frame = pd.read_csv(table, low_memory=False)
        frame.insert(0, "batch_id", str(roi))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name("." + target.name + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    return str(target)


def own_tables(root, config, mappings, image_name, roi, cache):
    """The tables that belong to the Segmentation step's own cells of this image: its tagged table (Cell tagging's
    output, when made after the cells) and its cell table, as [(name, path)], the one to colour by first."""
    from spacemap_gui.tabs import cell_tagging_ui as tab5

    from . import segmentation, tagging

    places = segmentation.step_folders(root, config, mappings)
    table = segmentation.results(places["output"], image_name)[0]
    if not table.is_file():
        return []
    found = []
    try:
        state = tagging.values(root, config)
        tagged = Path(state[tab5.OUTPUT_DIR_KEY]) / f"{tab5.sample_id_from_csv(table.name, '')}{state[tab5.OUTPUT_SUFFIX_KEY]}.csv"
        if tagged.is_file() and tagged.stat().st_mtime >= table.stat().st_mtime:
            found.append(("Tagged cells (Cell tagging)", _with_roi(tagged, roi, cache)))
    except (OSError, ValueError, KeyError):
        pass
    found.append(("This segmentation's cells", _with_roi(table, roi, cache)))
    return found
