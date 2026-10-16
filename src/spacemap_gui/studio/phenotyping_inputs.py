"""The combined measurement table shared by Batch correction, Studio and classic Phenotyping."""

from pathlib import Path

from .settings import project_lock, read_config


def _path(root, value):
    path = Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def folders(root, mappings, table=None):
    from .cell_inputs import image_folder

    mappings = {**mappings, **read_config(root).get("directory_mappings", {})}
    return {
        "images": str(_path(root, mappings["viz_roi_dir"])) if mappings.get("viz_roi_dir") else str(image_folder(root, mappings)),
        "tables": str(_path(root, mappings.get("global_pheno_dir") or (Path(table).parent if table else mappings.get("bc_output_dir") or "outputs/batch_corrected"))),
        "output": str(_path(root, mappings.get("phenotyping_output_dir") or "outputs/phenotyping")),
    }


def table_files(folder):
    """CSV and direct Parquet inputs; a CSV cache is not a second dataset."""
    folder = Path(folder)
    return [str(path) for path in sorted(folder.iterdir(), key=lambda p: p.name, reverse=True)
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
            and not path.name.startswith(".")
            and not (path.name.endswith(".csv.parquet") and path.with_suffix("").is_file())] if folder.is_dir() else []


def column_roles(root, table, available=None):
    from spacemap import dataio
    from .correction import last_run

    columns = set(available if available is not None else dataio.columns(table)) if table else set()
    config = read_config(root)
    saved = config.get("tab_configurations", {}).get("tab7_phenotyping", {}).get("studio_table_columns", {})
    if table and saved.get("table") and _path(root, saved["table"]) == Path(table).resolve():
        return {key: saved.get(key, "") if saved.get(key, "") in columns else "" for key in ("roi", "slide", "cell_id")}
    manifest = last_run(table) or {}
    bc = manifest.get("config", {})
    return {key: next((c for c in candidates if c in columns), "") for key, candidates in {
        "roi": [bc.get("batch_id_col"), "batch_id", "roi", "roi_id"],
        "slide": ["slide", "slide_id"],
        "cell_id": [bc.get("cell_id_col"), "cell_id"],
    }.items()}


def segmentation_column(root, table):
    from spacemap import dataio
    saved = read_config(root).get("tab_configurations", {}).get("tab7_phenotyping", {}).get("studio_table_columns", {})
    if table and saved.get("table") and _path(root, saved["table"]) == Path(table).resolve() and "label" in saved:
        return saved["label"]
    return "label" if table and "label" in dataio.columns(table) else ""


def save_roles(root, table, roles):
    from spacemap import dataio
    from spacemap_gui import persistence

    if not table or not isinstance(roles, dict) or set(roles) not in ({"roi", "slide", "cell_id"}, {"roi", "slide", "cell_id", "label"}):
        raise ValueError("Choose the ROI, Slide and Cell ID columns for the combined table.")
    columns = dataio.columns(table)
    if any(not isinstance(v, str) or v and v not in columns for v in roles.values()):
        raise ValueError("Choose columns that exist in the combined table.")
    if not roles["roi"] or not roles["cell_id"]:
        raise ValueError("Choose an ROI column and a unique Cell ID column. Slide is optional.")
    identities = [roles[key] for key in ("roi", "slide", "cell_id") if roles[key]]
    if len(set(identities)) != len(identities):
        raise ValueError("ROI, Slide and Cell ID must use different columns.")
    ids = dataio.load(table, columns=[roles["cell_id"]])[roles["cell_id"]]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.astype(str).duplicated().any():
        raise ValueError("The Cell ID column has missing or repeated IDs. Use the unique IDs from the merged table.")
    with project_lock(root):
        config = read_config(root)
        config.setdefault("tab_configurations", {}).setdefault("tab7_phenotyping", {})["studio_table_columns"] = {
            "table": str(Path(table).resolve()), **roles}
        persistence.save_project_config(str(root), config)


def use_table(channel, path):
    """Persist an explicitly selected input and immediately use it in this Studio session."""
    from spacemap import dataio
    from spacemap_gui import persistence
    from spacemap_gui.viewer import registry

    project = registry.session_project(channel) or {}
    root = project.get("root")
    if not root or project.get("read_only"):
        raise ValueError("Open an editable project before choosing its cell table.")
    path = _path(root, path)
    if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError("Choose an existing combined CSV or Parquet cell table.")
    dataio.columns(str(path))  # fail before saving an unreadable file
    with project_lock(root):
        config = read_config(root)
        for key, value in {"global_pheno_dir": str(path.parent), "global_pheno_file": path.name,
                           "global_pheno_dir_user_override": True, "proj_unified_csv_filename": path.name}.items():
            persistence.update_stored_config_key(config, key, value)
        persistence.save_project_config(str(root), config)
    registry.session_use_table(channel, str(path))
    return str(path)


def save_folders(root, changes):
    from spacemap_gui import persistence

    keys = {"images": "viz_roi_dir", "tables": "global_pheno_dir", "output": "phenotyping_output_dir"}
    if not isinstance(changes, dict) or set(changes) - set(keys):
        raise ValueError("Choose an Images, Tables or Output folder.")
    checked = {}
    for name, value in changes.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise ValueError("Enter a folder path.")
        path = _path(root, value.strip())
        if path.exists() and not path.is_dir():
            raise ValueError(f"{path.name} is a file. Enter its folder here.")
        if name != "output" and not path.is_dir():
            raise ValueError(f"The {name} folder does not exist.")
        checked[keys[name]] = str(path)
    with project_lock(root):
        config = read_config(root)
        for key, value in checked.items():
            persistence.update_stored_config_key(config, key, value)
        if "tables" in changes:
            persistence.update_stored_config_key(config, "global_pheno_dir_user_override", True)
        persistence.save_project_config(str(root), config)


def plan(root, mappings, table=None):
    from spacemap import dataio
    from spacemap.viewer.catalog import sample_of
    from . import cell_inputs, phenotyping_overview

    locations = folders(root, mappings, table)
    warnings = []
    try:
        images = cell_inputs.images(root, {"bc_images_dir": locations["images"]})
    except (OSError, ValueError) as error:
        images = {}
        warnings.append(str(error))
    available, roles = [], {"roi": "", "slide": "", "cell_id": ""}
    info, results, methods = {"table": "", "cells": 0, "rois": [], "phenotypes": []}, {}, []
    try:
        available = dataio.columns(table) if table and Path(table).is_file() else []
        roles = column_roles(root, table, available)
        info = phenotyping_overview.overview(table, roi_column=roles["roi"], slide_column=roles["slide"])
        methods = phenotyping_overview.method_columns(root, table, available)
        results = phenotyping_overview.results(table, suffix=None, roi_column=roles["roi"],
                                               extra_columns=[c for m in methods for c in m["columns"]])
    except (OSError, ValueError, KeyError) as error:
        warnings.append(str(error))
    counts = {sample_of(item["roi"]): item["cells"] for item in info["rois"]}
    slides = {sample_of(item["roi"]): item.get("slide", "") for item in info["rois"]}
    rows = [{"roi": roi, "image": Path(images[roi]).name if roi in images else "",
             "image_path": images.get(roi, ""), "cells": counts.get(roi, 0), "slide": slides.get(roi, ""),
             "results": {column: sum(n for key, n in seen.items() if sample_of(key) == roi)
                         for column, seen in results.items()}}
            for roi in sorted(set(images) | set(counts))]
    return {**info, "path": str(table or ""), "folders": locations,
            "segmentation_label": segmentation_column(root, table) if available else "",
            "tables": table_files(locations["tables"]), "rows": rows,
            "cell_id_column": roles["cell_id"], "columns": available, "roles": roles,
            "result_columns": list(results), "methods": methods, "warnings": warnings}


def combined_table(channel):
    from spacemap_gui.viewer import registry

    project = registry.session_project(channel) or {}
    table = project.get("unified")
    if not table or not Path(table).is_file():
        raise ValueError("Choose the combined cell table at the top of Phenotyping, or use the result from Batch correction.")
    return str(table)
