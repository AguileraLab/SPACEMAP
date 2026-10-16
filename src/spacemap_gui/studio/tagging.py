"""Studio's Cell tagging step: Tab 5's standalone spatial cell tagging, with its saved settings.

Each cell gets columns from the annotation it sits in: 0/1 membership of each class, which polygon
(ROI) and its size, a signed distance, the bed (which of chosen classes it is closer to) and an
edge flag, all inside the cell's own tissue piece. Cell tables are either one CSV per ROI or one
combined CSV with a column naming the ROI. Every table is matched to its annotation GeoJSON by
name, or to a file chosen by hand. Settings are Tab 5's ``seg_tag_*`` keys, so classic Tab 5's
"Cell tagging" page shows and uses the same choices. Tagged tables are new CSVs (the input tables
are never changed), as classic tagging writes them: batch correction reads them from there.
"""

from functools import lru_cache
from pathlib import Path

from spacemap_gui.tabs import cell_tagging_ui as tab5

SECTION = "tab5_segmentation"
SHARED_GEOJSON_KEY = "tab4_annot_geojson_dir"
FOLDERS = {
    "csv": (tab5.CSV_DIR_KEY, "outputs/segmentation/cellpose/features"),
    "merged": (tab5.MERGED_DIR_KEY, "outputs/segmentation/cellpose/features"),
    "output": (tab5.OUTPUT_DIR_KEY, "outputs/segmentation/tagged_features"),
    "geojson": (tab5.GEOJSON_DIR_KEY, "outputs/processed_annotations/stage1b_compartments_zones"),
}
DEFAULTS = {
    tab5.MODE_KEY: tab5.MODE_PER_IMAGE,
    tab5.CSV_SUFFIX_KEY: tab5.DEFAULT_CSV_SUFFIX,
    tab5.MERGED_FILE_KEY: "",
    tab5.BATCH_COLUMN_KEY: "",
    tab5.GEOJSON_OVERRIDE_KEY: False,
    tab5.GEOJSON_MATCHES_KEY: {},
    tab5.OUTPUT_SUFFIX_KEY: tab5.DEFAULT_OUTPUT_SUFFIX,
    tab5.X_COLUMN_KEY: "",
    tab5.Y_COLUMN_KEY: "",
    tab5.TISSUE_LABEL_KEY: tab5.DEFAULT_TISSUE_LABEL,
    tab5.RULES_KEY: [],
    tab5.BED_LABELS_KEY: [],
    tab5.EDGE_ENABLED_KEY: False,
    tab5.EDGE_REFERENCE_KEY: tab5.DEFAULT_TISSUE_LABEL,
    tab5.EDGE_BUFFER_KEY: 100.0,
    tab5.EDGE_COLUMN_KEY: "edge_cell",
    tab5.MIN_BED_CELLS_KEY: tab5.DEFAULT_MIN_CELLS_PER_BED_TARGET,
    tab5.CLEAN_PATH_KEY: False,
    tab5.OVERWRITE_COLUMNS_KEY: True,
    tab5.USE_GPU_KEY: False,
    "seg_tag_cpu_workers": 0,
    # Studio: the user's column names, units, one "bed" column, and the Annotation step's exports as the annotations.
    "seg_tag_column_names": {},
    "seg_tag_units": "um",
    "seg_tag_bed_column": "bed",
    "seg_tag_use_annotation_export": True,
    # Where each table's annotations come from when none is picked for it: "export" (the Annotation step's export,
    # else the folder's file by name), "imported" (the GeoJSON imported for its image), "folder" (by name); "": export
    # unless the old exports switch is off.
    "seg_tag_annotation_source": "",
}
# Distances that stay inside the cell's own tissue piece (a straight line that never leaves it) are what the user
# asked for; Studio starts with them on.
DEFAULTS[tab5.CLEAN_PATH_KEY] = True
FOLLOW_KEY = "seg_tag_follow_segmentation"  # on: the tables are the Segmentation step's output, wherever it is
DEFAULTS[FOLLOW_KEY] = True
NAMES_KEY, UNITS_KEY, BED_COLUMN_KEY, USE_EXPORT_KEY = ("seg_tag_column_names", "seg_tag_units", "seg_tag_bed_column",
                                                        "seg_tag_use_annotation_export")
SOURCE_KEY = "seg_tag_annotation_source"
SOURCES = ("", "export", "imported", "folder")
FOLDER_KEYS = {key for key, _ in FOLDERS.values()}
TEXT_LIMIT = 128


def classes_of(path):
    """An annotation file's class names (re-read when the file changes); Studio's own cache, not Streamlit's."""
    import os

    if not path or not os.path.isfile(path):
        return []
    stamp = os.stat(path)
    return list(_classes(str(path), stamp.st_mtime_ns, stamp.st_size))


@lru_cache(maxsize=256)
def _classes(path, modified, size):
    from spacemap.cell_spatial_annotation import list_annotation_classes

    return tuple(list_annotation_classes(path))


def _path(root, value):
    return Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()


TABLE_SUFFIXES = (".csv", ".parquet")


def _csvs(folder):
    """The cell tables in a folder: CSV or Parquet (not the Parquet twins Studio keeps beside CSVs)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TABLE_SUFFIXES
                  and not p.name.startswith(".") and not p.name.lower().endswith(".csv.parquet"))


def _columns(path):
    from spacemap import dataio

    return list(dataio.columns(str(path), include_pending=False)) if Path(path).is_file() else []


def _batch_values(path, column):
    from spacemap import dataio

    if not column or not Path(path).is_file():
        return []
    return sorted(map(str, dataio.load(str(path), columns=[column], apply_labels=False)[column].dropna().astype(str).unique()))


def read_table(path):
    """A whole cell table, CSV or Parquet (Parquet reads much faster)."""
    import pandas as pd

    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)


def write_table(frame, path):
    """Written in the format its name says (a Parquet input gives a Parquet output)."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _extension(name):
    return ".parquet" if str(name).lower().endswith(".parquet") else ".csv"


def values(root, config):
    """Tab 5's tagging settings for the project, with its defaults; folders as paths."""
    saved = ((config or {}).get("tab_configurations") or {}).get(SECTION) or {}
    mappings = (config or {}).get("directory_mappings") or {}
    found = {key: saved.get(key, default) for key, default in DEFAULTS.items()}
    for kind, (key, fallback) in FOLDERS.items():
        found[key] = str(_path(root, saved[key])) if saved.get(key) else str(Path(root) / fallback)
    if not saved.get(tab5.CSV_DIR_KEY):
        # Where segmentation wrote its tables: Tab 5's CSV folder, else its output folder, when they hold CSVs.
        for key in ("tab4_csv_dir", "tab4_output_dir"):
            if mappings.get(key) and _csvs(_path(root, mappings[key])):
                found[tab5.CSV_DIR_KEY] = str(_path(root, mappings[key]))
                break
    from . import segmentation

    produced = segmentation.step_folders(root, config or {}, mappings)["output"]
    follow = saved.get(FOLLOW_KEY)
    if follow is None:  # never chosen: follow, unless the folder used so far holds tables and Segmentation's has none
        follow = not (_csvs(Path(found[tab5.CSV_DIR_KEY])) and not _csvs(produced))
    found[FOLLOW_KEY] = bool(follow)
    if follow and found[tab5.MODE_KEY] == tab5.MODE_PER_IMAGE:
        found[tab5.CSV_DIR_KEY] = str(produced)  # the tables Segmentation writes now, even after its Output changed
    if not found[tab5.GEOJSON_OVERRIDE_KEY]:
        shared = mappings.get(SHARED_GEOJSON_KEY) or saved.get(SHARED_GEOJSON_KEY)
        found[tab5.GEOJSON_DIR_KEY] = str(_path(root, shared)) if shared else found[tab5.GEOJSON_DIR_KEY]
    return found


def _shown(root, folder):
    try:
        return str(Path(folder).resolve().relative_to(Path(root).resolve())) or "."
    except ValueError:
        return str(folder)


def folders_with(root, suffix, depth=4):
    """The project's folders (a few levels deep) holding files ending with *suffix*, relative to the project."""
    import os

    root = Path(root).resolve()
    found = []
    for current, names, files in os.walk(root):
        level = len(Path(current).relative_to(root).parts)
        names[:] = sorted(name for name in names if not name.startswith(".") and name not in {"intermediate", "intermediate_images"})
        if level >= depth:
            names[:] = []
        if any(name.lower().endswith(suffix) and not name.startswith(".") for name in files):
            found.append(str(Path(current).relative_to(root)))
    return sorted(found)


def sources(root, state):
    """The cell tables to tag and the annotation each one uses.

    Returns {"mode", "tables" (per ROI) or "table"/"columns"/"ids" (combined), "rows": [{source, sample,
    annotation, chosen, classes}], "classes", "geojson_files", "columns", "coordinates"}.
    """
    geojson_dir = state[tab5.GEOJSON_DIR_KEY]
    matches = state.get(tab5.GEOJSON_MATCHES_KEY) or {}
    found = {"mode": state[tab5.MODE_KEY], "rows": [], "columns": []}
    if state[tab5.MODE_KEY] == tab5.MODE_MERGED:
        folder = Path(state[tab5.MERGED_DIR_KEY])
        tables = _csvs(folder)
        name = state[tab5.MERGED_FILE_KEY] if state[tab5.MERGED_FILE_KEY] in tables else (tables[0] if tables else "")
        found.update(tables=tables, table=name)
        if name:
            columns = _columns(folder / name)
            guesses = [c for c in columns if str(c).lower() in {"batch", "batch_id", "image", "image_id", "sample", "sample_id", "roi", "roi_id", "source"}]
            batch = state[tab5.BATCH_COLUMN_KEY] if state[tab5.BATCH_COLUMN_KEY] in columns else (guesses[0] if guesses else "")
            found.update(columns=columns, batch_column=batch)
            ids = _batch_values(folder / name, batch) if batch else []
            pairs = [(value, value) for value in ids]
        else:
            pairs = []
    else:
        folder = Path(state[tab5.CSV_DIR_KEY])
        tables = _csvs(folder)
        found["tables"] = tables
        if tables:
            found["columns"] = _columns(folder / tables[0])
        pairs = [(name, tab5.sample_id_from_csv(name, state[tab5.CSV_SUFFIX_KEY])) for name in tables]
    classes = []
    exports, images = _exports_and_images(root)
    for source, sample in pairs:
        chosen = matches.get(source) or matches.get(sample)
        path, exported = annotation_path(state, sample, chosen, exports)
        names = classes_of(path)
        classes += [name for name in names if name not in classes]
        imported = (exports.get(IMPORTED) or {}).get(_sample_key(sample), [])
        found["rows"].append({"source": source, "sample": sample, "annotation": _shown_name(path) if path else None,
                              "chosen": chosen or "", "classes": names, "exported": exported,
                              "origin": "export" if exported else "imported" if path in imported else "folder" if path else "",
                              "imported": [["imported:" + Path(p).name, _shown_name(p)] for p in imported],
                              "image": images.get(_sample_key(sample))})
    found["classes"] = classes
    found["geojson_files"] = sorted(p.name for p in Path(geojson_dir).glob("*.geojson")) if Path(geojson_dir).is_dir() else []
    x, y = _coordinates(found["columns"], state)
    found["coordinates"] = {"x": x, "y": y}
    return found


def _sample_key(name):
    from spacemap.viewer.catalog import sample_of

    return sample_of(Path(str(name)).name)


IMPORTED = "__imported__"  # in the exports map: {sample: [imported GeoJSON copies]}


def _shown_name(path):
    """A file's name as the lists show it: an imported copy without Studio's id in front."""
    name = Path(str(path)).name
    return name.split("_", 1)[-1] if "imported_annotations" in Path(str(path)).parts else name


def _exports_and_images(root):
    """The Annotation step's exported GeoJSON per sample (with the GeoJSON imported for each image under IMPORTED),
    and the Segmentation step's image per sample."""
    from . import annotation, segmentation
    from .settings import read_config

    config = read_config(root)
    mappings = config.get("directory_mappings") or {}
    output = annotation.main_output(root, mappings)  # the export stays in the output folder itself
    exports = {}
    folder = Path(output) / "studio_export"
    if folder.is_dir():
        for path in folder.glob("*_annotations.geojson"):
            exports[_sample_key(path.name[: -len("_annotations.geojson")])] = str(path)
    from . import annotation_inputs, annotation_names

    imported = {}
    for image in annotation_names._annotation_images(root, config):
        files = sorted(annotation_inputs.imported(root, image), key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
        if files:
            imported[_sample_key(image.name)] = [str(p) for p in files]
    exports[IMPORTED] = imported
    images = {}
    places = segmentation.step_folders(root, config, mappings)
    if places["images"].is_dir():
        for path in places["images"].iterdir():
            if path.name.lower().endswith(segmentation.IMAGE_SUFFIXES):
                images.setdefault(_sample_key(path.name), path.name)
    return exports, images


def annotation_path(state, sample, chosen, exports):
    """The annotation file for a table: one picked by hand (a folder file, or "imported:<copy>"), else by the source
    chosen for all tables: the Annotation step's export (else the folder's file by name), the GeoJSON imported for
    its image (the newest), or the folder's file by name."""
    imported = (exports.get(IMPORTED) or {}).get(_sample_key(sample), [])
    if chosen and chosen.startswith("imported:"):
        return next((p for p in imported if Path(p).name == chosen[len("imported:"):]), None), False
    source = state.get(SOURCE_KEY) or ("export" if state.get(USE_EXPORT_KEY, True) else "folder")
    if not chosen and source == "imported":
        return (imported[0] if imported else None), False
    if not chosen and source == "export":
        exported = exports.get(_sample_key(sample))
        if exported:
            return exported, True
    return tab5.match_geojson(sample, state[tab5.GEOJSON_DIR_KEY], chosen), False


def microns_for(root, sample):
    """Microns per pixel for a table's cells: its image's, else the project's CODEX pixel size."""
    from spacemap.viewer.sources import image_info

    from . import segmentation
    from .settings import read_config

    config = read_config(root)
    _, images = _exports_and_images(root)
    name = images.get(_sample_key(sample))
    if name:
        places = segmentation.step_folders(root, config, config.get("directory_mappings") or {})
        try:
            size = image_info(str(places["images"] / name)).pixel_size_um
            if size:
                return float(size)
        except Exception:
            pass
    try:
        return float(config.get("mpp_codex") or 0) or None
    except (TypeError, ValueError):
        return None


def _coordinates(columns, state):
    import pandas as pd

    from spacemap.cell_spatial_annotation import resolve_coordinate_columns

    frame = pd.DataFrame(columns=list(columns))
    return resolve_coordinate_columns(frame, state.get(tab5.X_COLUMN_KEY) or None, state.get(tab5.Y_COLUMN_KEY) or None)


def rules(found, state):
    """Tab 5's rule rows for the classes found (saved choices kept, new classes added)."""
    return tab5.align_rules(found["classes"], state[tab5.TISSUE_LABEL_KEY], state.get(tab5.RULES_KEY) or [])


def planned_columns(state, records=None):
    """The columns tagging adds, with the user's names."""
    from .tag_columns import planned

    _, edge, *_ = tab5.build_engine_inputs(state)
    records = tab5.normalize_rule_records(records if records is not None else state.get(tab5.RULES_KEY) or [])
    return planned([r for r in records if any(r.get(k) for k in ("Binary (0/1)", "ROI ID", "ROI Size", "Distance"))],
                   state.get(NAMES_KEY) or {}, tab5.selected_bed_labels(state=state), state.get(BED_COLUMN_KEY, ""),
                   edge.column if edge else "")


def tag_frame(frame, geojson, state, x, y, microns, progress=None):
    """Tag one table's cells (in memory) and give the columns the user's names and units. Returns (frame, columns,
    warnings)."""
    from spacemap.cell_spatial_annotation import annotate_cells

    from .tag_columns import finish

    engine_rules, edge, tissue_label, minimum, clean = tab5.build_engine_inputs(state)
    beds = tab5.selected_bed_labels(state=state)
    options = dict(x_col=x, y_col=y, tissue_label=tissue_label, min_cells_per_bed_target=minimum, require_clean_path=clean,
                   overwrite_existing=bool(state.get(tab5.OVERWRITE_COLUMNS_KEY, True)), use_gpu=bool(state.get(tab5.USE_GPU_KEY)),
                   cpu_workers=state.get("seg_tag_cpu_workers", 0), independent_distances=True, progress=progress)
    # With "inside the piece", each annotation's distance is measured on its own: only the tissue piece's edge stops
    # the straight line, never another annotation (a Stroma cell still gets its distance to Tumor). Beds keep the
    # engine's rule, in which the nearest annotation boundary along the line wins.
    annotated, summary = annotate_cells(frame, geojson, engine_rules, edge_config=edge, bed_labels=beds, **options)
    labels = list(dict.fromkeys([rule.label for rule in engine_rules] + list(beds)))
    annotated, columns, scales = finish(annotated, labels, state.get(NAMES_KEY) or {}, state.get(UNITS_KEY, "um"), microns,
                                        beds, state.get(BED_COLUMN_KEY, ""), edge.column if edge else "")
    warnings = list(summary.get("warnings", []))
    if state.get(UNITS_KEY, "um") == "um" and not microns:
        warnings.append("No pixel size for this image: distances and sizes stay in pixels.")
    return annotated, columns, warnings, scales


def check(changes):
    """Only Tab 5's tagging settings, with sane values."""
    if not isinstance(changes, dict) or set(changes) - set(DEFAULTS) - FOLDER_KEYS:
        raise ValueError("Only the cell tagging settings can be changed here.")
    if tab5.CSV_DIR_KEY in changes and FOLLOW_KEY not in changes:
        changes = {**changes, FOLLOW_KEY: False}  # a folder picked by hand: no longer following Segmentation
    if tab5.GEOJSON_DIR_KEY in changes and tab5.GEOJSON_OVERRIDE_KEY not in changes:
        changes = {**changes, tab5.GEOJSON_OVERRIDE_KEY: True}  # an Annotations folder typed here wins over the shared one
    if FOLLOW_KEY in changes and not isinstance(changes[FOLLOW_KEY], bool):
        raise ValueError("Follow Segmentation is on or off.")
    for key, value in changes.items():
        if key == tab5.MODE_KEY and value not in tab5.MODES:
            raise ValueError("Choose one CSV per ROI or one combined CSV.")
        elif key == NAMES_KEY:
            from .tag_columns import check_names
            check_names(value)
        elif key == UNITS_KEY:
            if value not in ("um", "px"):
                raise ValueError("Choose µm or pixels.")
        elif key == BED_COLUMN_KEY:
            from .tag_columns import NAME
            if not isinstance(value, str) or (value and not NAME.match(value)):
                raise ValueError("The bed column's name starts with a letter or _, then letters, digits, _ or . (up to 64).")
        elif key == "seg_tag_cpu_workers":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 256:
                raise ValueError("CPU cores must be a whole number from 0 (Auto) to 256.")
        elif key in {tab5.GEOJSON_OVERRIDE_KEY, tab5.EDGE_ENABLED_KEY, tab5.CLEAN_PATH_KEY, tab5.OVERWRITE_COLUMNS_KEY, tab5.USE_GPU_KEY,
                     USE_EXPORT_KEY}:
            if not isinstance(value, bool):
                raise ValueError("Choose on or off.")
        elif key == SOURCE_KEY:
            if value not in SOURCES:
                raise ValueError("Choose the export, the imported files or the folder.")
        elif key == tab5.EDGE_BUFFER_KEY:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 100000:
                raise ValueError("The edge distance must be 0 to 100,000 pixels.")
        elif key == tab5.MIN_BED_CELLS_KEY:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100000:
                raise ValueError("The smallest bed must be 1 to 100,000 cells.")
        elif key == tab5.RULES_KEY:
            if not isinstance(value, list) or len(value) > 200 or not all(isinstance(item, dict) for item in value):
                raise ValueError("Send the tagging rules as a list.")
        elif key == tab5.BED_LABELS_KEY:
            if not isinstance(value, list) or len(value) > 50 or not all(isinstance(item, str) and 0 < len(item) <= TEXT_LIMIT for item in value):
                raise ValueError("Choose the bed annotations by name.")
            if len(value) == 1:
                raise ValueError("Beds compare at least two annotations.")
        elif key == tab5.GEOJSON_MATCHES_KEY:
            if not isinstance(value, dict) or len(value) > 5000 or not all(
                isinstance(k, str) and isinstance(v, str) and len(k) <= 512 and len(v) <= 300 and "/" not in v and "\\" not in v
                for k, v in value.items()
            ):
                raise ValueError("Choose an annotation file from the folder for each table.")
        elif key == tab5.OUTPUT_SUFFIX_KEY:
            if not isinstance(value, str) or not value.strip() or len(value) > 64 or any(mark in value for mark in "/\\"):
                raise ValueError("The output suffix cannot be empty: it keeps the tagged tables apart from the originals.")
        elif key == FOLLOW_KEY:
            pass  # checked above (on or off)
        elif key in FOLDER_KEYS:
            if not isinstance(value, str) or not value.strip() or len(value) > 1024 or (not Path(value).is_absolute() and ".." in Path(value).parts):
                raise ValueError("Type the folder's full path (or a path inside the project).")
        elif not isinstance(value, str) or len(value) > 512:
            raise ValueError(f"Check {key}.")
    return changes


def save(root, changes, expected):
    """Save tagging settings as classic Tab 5 would; folders go in as ``${PROJECT_ROOT}/...``."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    changes = check(changes)
    with project_lock(root):
        config = read_config(root)
        current = values(root, config)
        if expected != revision(current):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        for key, value in changes.items():
            if key == tab5.RULES_KEY:
                value = tab5.normalize_rule_records(value)
            if key in FOLDER_KEYS:  # kept as ${PROJECT_ROOT}/... inside the project, else as the full path
                typed = Path(value.strip()).expanduser()
                full = (typed if typed.is_absolute() else Path(root) / typed).resolve()
                if key != tab5.OUTPUT_DIR_KEY and not full.is_dir():
                    raise ValueError(f"{full} is not a folder.")
                if not full.parent.is_dir():
                    raise ValueError(f"{full.parent} is not a folder.")
                base = Path(root).resolve()
                value = "${PROJECT_ROOT}/" + full.relative_to(base).as_posix() if base in full.parents or full == base else str(full)
            persistence.update_stored_config_key(config, key, value)
        try:
            persistence.save_project_config(str(root), config, raise_errors=True)
        except OSError as exc:
            import errno

            reasons = {
                errno.EDQUOT: "Storage quota exceeded. Free storage or request a quota increase",
                errno.ENOSPC: "The project filesystem is full. Free space before saving",
                errno.EACCES: "The project folder is not writable by your account",
                errno.EPERM: "The project folder is not writable by your account",
            }
            reason = reasons.get(exc.errno, str(exc))
            raise ValueError(f"Cell tagging settings were not saved. {reason}.") from exc
        return values(root, read_config(root))


def outputs(state, found):
    """Which tables are already tagged: source -> tagged file name (when it exists)."""
    folder = Path(state[tab5.OUTPUT_DIR_KEY])
    suffix = state[tab5.OUTPUT_SUFFIX_KEY]
    if found["mode"] == tab5.MODE_MERGED:
        name = f"{Path(found.get('table') or '').stem}{suffix}{_extension(found.get('table'))}" if found.get("table") else ""
        return {"*": name} if name and (folder / name).exists() else {}
    done = {}
    for row in found["rows"]:
        name = f"{tab5.sample_id_from_csv(row['source'], '')}{suffix}{_extension(row['source'])}"
        if (folder / name).exists():
            done[row["source"]] = name
    return done


def run(settings, folder, report):
    """Tag the listed tables (a job): each with its own annotation, columns named and converted as chosen. The input
    tables are never changed; tagged tables are new CSVs in the output folder."""
    from .tagging_outputs import StagedTables

    state = settings["state"]
    found = settings["found"]
    x, y = found["coordinates"]["x"], found["coordinates"]["y"]
    if not x or not y:
        raise ValueError("Choose the columns that hold the cell coordinates.")
    out_dir = Path(state[tab5.OUTPUT_DIR_KEY])
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = state[tab5.MODE_KEY] == tab5.MODE_MERGED
    if merged:
        report(f"Reading combined table: {found['table']}", 0)
    combined = read_table(Path(state[tab5.MERGED_DIR_KEY]) / found["table"]) if merged else None
    wanted = set(settings.get("tables") or [])
    rows = [row for row in found["rows"] if not wanted or row["source"] in wanted]
    with StagedTables(out_dir, folder) as staged:
        return _run_staged(settings, rows, combined, staged, report)


def _run_staged(settings, rows, combined, staged, report):
    import pandas as pd
    import time
    from spacemap.spatial_visibility import cpu_workers

    state, found = settings["state"], settings["found"]
    root = settings.get("root")
    x, y = found["coordinates"]["x"], found["coordinates"]["y"]
    exports = {_sample_key(k): v for k, v in (settings.get("exports") or {}).items()}
    out_dir, suffix = Path(state[tab5.OUTPUT_DIR_KEY]), state[tab5.OUTPUT_SUFFIX_KEY]
    merged = combined is not None
    # Convert and group once, instead of rescanning the combined cohort per image.
    groups = combined.groupby(combined[found["batch_column"]].astype(str), sort=False).indices if merged else {}
    records, parts = [], []
    if state.get(tab5.CLEAN_PATH_KEY):
        report(f"Cell tagging: up to {cpu_workers(state.get('seg_tag_cpu_workers', 0))} CPU workers for gap-aware distances.", 0)
        report("Gap-aware paths and bed assignment use CPU. GPU acceleration applies to eligible ordinary boundary distances. Progress tracks stages, not remaining time.")
    for number, row in enumerate(rows, start=1):
        started = time.perf_counter()
        def image_report(message, fraction=None):
            progress = None if fraction is None else .98 * (number - 1 + fraction) / max(len(rows), 1)
            report(f"[{time.strftime('%H:%M:%S')}] {row['source']} ({number}/{len(rows)}): {message}", progress)

        image_report("Loading cell table", 0)
        record = {"Source": row["source"]}
        geojson, _ = annotation_path(state, row["sample"], row["chosen"] or None, exports)
        if not geojson:
            records.append({**record, "Status": "skipped: no annotation"})
            image_report("Skipped: no annotation selected", 1)
            continue
        try:
            if merged:
                frame = combined.iloc[groups.get(row["source"], [])]
            else:
                frame = read_table(Path(state[tab5.CSV_DIR_KEY]) / row["source"])
            image_report(f"Loaded {len(frame):,} cells; annotation {Path(geojson).name}", .03)
            microns = microns_for(root, row["sample"]) if root else None
            def measurements(message, fraction=None):
                image_report(message, None if fraction is None else .04 + .90 * fraction)

            tagged, columns, warnings, _ = tag_frame(frame, geojson, state, x, y, microns, progress=measurements)
            for warning in warnings:
                image_report(f"Warning: {warning}")
            if merged:
                parts.append(tagged)
            else:
                output = out_dir / f"{tab5.sample_id_from_csv(row['source'], '')}{suffix}{_extension(row['source'])}"
                image_report(f"Writing staged table: {output.name}", .96)
                staged.stage(tagged, output, write_table)
            records.append({**record, "Status": "ok", "Cells": int(len(tagged)), "Columns": len(columns),
                            "Annotation": Path(geojson).name, "Warnings": "; ".join(warnings)})
            image_report(f"Tagged {len(tagged):,} cells, {len(columns)} columns ({time.perf_counter() - started:.1f}s)", 1)
        except Exception as error:  # one table failing does not stop the others
            records.append({**record, "Status": f"failed: {error}"})
            image_report(f"Failed: {error}", 1)
    if merged and parts:
        output = out_dir / f"{Path(found['table']).stem}{suffix}{_extension(found['table'])}"
        report(f"Writing combined tagged table: {output.name}", .98)
        staged.stage(pd.concat(parts), output, write_table)
    written = staged.publish(report)
    failed = [record for record in records if str(record.get("Status", "")).startswith("failed")]
    return {"records": records, "outputs": written, "count": len(written), "failed": len(failed)}


# ---------------------------------------------------------------- preview on the image (nothing is written)
PREFIX = "preview:"  # a cell layer coloured by "preview:<column>" reads the column from here
_PREVIEWS = {}
_PREVIEW_LIMIT = 4


def _row_for(found, image_name):
    from spacemap.viewer.catalog import sample_of

    wanted = sample_of(image_name)
    return next((row for row in found["rows"] if sample_of(row["sample"]) == wanted), None)


def preview(root, state, image_path, roi, bounds=None):
    """Tag the open ROI's cells in memory with the current settings; the table on disk is not changed.

    Returns the columns tagging would add, the number of cells, the annotation used and its warnings.
    """
    import numpy as np
    import pandas as pd

    found = sources(root, state)
    row = _row_for(found, Path(image_path).name)
    if row is None:
        raise ValueError(f"No cell table for {Path(image_path).name} in the chosen folder.")
    exports, _ = _exports_and_images(root)
    geojson, _ = annotation_path(state, row["sample"], row["chosen"] or None, exports)
    if not geojson:
        raise ValueError(f"No annotation file for {row['source']}: pick one in Annotations.")
    x, y = found["coordinates"]["x"], found["coordinates"]["y"]
    if not x or not y:
        raise ValueError("Choose the columns that hold the cell coordinates.")
    if found["mode"] == tab5.MODE_MERGED:
        frame = read_table(Path(state[tab5.MERGED_DIR_KEY]) / found["table"])
        frame = frame[frame[found["batch_column"]].astype(str) == row["source"]]
    else:
        frame = read_table(Path(state[tab5.CSV_DIR_KEY]) / row["source"])
    if "label" not in frame:
        raise ValueError(f"{row['source']} has no label column, so its cells cannot be matched to the outlines.")
    total = len(frame)
    if bounds:
        # Only the cells in view: quicker on a large image (distances still use the whole annotation).
        x0, y0, width, height = (float(v) for v in bounds)
        xs, ys = pd.to_numeric(frame[x], errors="coerce"), pd.to_numeric(frame[y], errors="coerce")
        frame = frame[(xs >= x0) & (xs <= x0 + width) & (ys >= y0) & (ys <= y0 + height)]
        if frame.empty:
            raise ValueError("No cells in view: zoom out, or preview the whole image.")
    annotated, columns, warnings, scales = tag_frame(frame, geojson, state, x, y, microns_for(root, row["sample"]))
    summary = {"warnings": warnings}
    columns = [column for column in columns if column in annotated]
    if not columns:
        raise ValueError("Choose at least one column to add, then preview.")
    key = (str(image_path), str(roi))
    version = _PREVIEWS.get(key, {}).get("version", 0) + 1
    if len(_PREVIEWS) >= _PREVIEW_LIMIT and key not in _PREVIEWS:
        _PREVIEWS.pop(next(iter(_PREVIEWS)))
    _PREVIEWS[key] = {
        "version": version,
        "labels": pd.to_numeric(annotated["label"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
        "columns": {column: annotated[column].to_numpy() for column in columns},
        "scales": set(scales),  # distances and sizes: always a colour scale
    }
    return {"columns": columns, "cells": int(len(annotated)), "of": int(total), "annotation": Path(geojson).name, "table": row["source"],
            "version": version, "warnings": list(summary.get("warnings", []))[:10],
            "summary": {column: _describe(annotated[column], column in scales) for column in columns}}


def _describe(values, scale=False):
    """A column in a line: numbers as lowest / middle / highest and blanks; categories as counts of each value.
    Measurements (*scale*: distances, sizes) are always numbers."""
    import pandas as pd

    blank = int(values.isna().sum())
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.notna().sum() == values.notna().sum() and values.notna().any() and (scale or values.nunique() > 12):
        return {"kind": "numeric", "blank": blank, "min": float(numbers.min()), "median": float(numbers.median()), "max": float(numbers.max())}
    counts = values.astype(object).where(values.notna(), None).value_counts(dropna=True).head(12)
    return {"kind": "categorical", "blank": blank, "counts": [[str(key), int(count)] for key, count in counts.items()]}


def preview_values(image, roi, column):
    """(labels, values, version) of a previewed column for the cell layer of *image* / *roi*, or None."""
    found = _PREVIEWS.get((str(image), str(roi)))
    if not found or column not in found["columns"]:
        return None
    return found["labels"], found["columns"][column], found["version"]


def preview_is_scale(image, roi, column):
    """Whether a previewed column is a measurement (a distance or a size), shown as a colour scale."""
    found = _PREVIEWS.get((str(image), str(roi)))
    return bool(found and column in found.get("scales", ()))


def clear_preview(image, roi):
    _PREVIEWS.pop((str(image), str(roi)), None)
