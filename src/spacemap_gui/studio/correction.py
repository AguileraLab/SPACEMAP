"""Studio's batch correction: classic Tab 6's correction run as a job, from the cell tables and Samples.

The steps, settings and defaults are Tab 6's (``tabs/tab6_batch_correction.py``:
the setup's Load and the Run of "QC & Correction"), in the same order, and the
settings are Tab 6's own (``bc_config``), so classic and Studio share them:

1. Read each ROI's cell table (Tab 6's input folder, else the segmentation's),
   give its cells the ROI's Samples columns (slide, outcome...), and name the
   batch after the ROI, as the combined table does.
2. Give every cell an ID with Tab 6's strategy (Tab 6's own function).
3. Filter as Tab 6 does: the tissue column (only tables that have it), cell area
   between two percentiles, and cells that are outliers in most markers.
4. Correct with Tab 6's ``BatchCorrector`` between the chosen units (the slide
   by default): arcsinh, then a SOM of the cells so that matched populations
   are compared, then cyCombine's ComBat within each SOM group (default).
   SPACEMAP's monotonic variant and legacy approximation remain available.
5. Measure pooled EMD reduction and relative MAD change on up to 10,000 cells;
   these are diagnostics, not cyCombine population-level benchmark scores.
6. Optional post-correction QC; then save every loaded cell with its corrected
   values as ``batch_corrected_merged_<time>.parquet``. Studio saves Parquet,
   never CSV: a CSV (for classic) is written only when the user asks, from
   Jobs > Saved results. Nothing is overwritten.
"""

from datetime import datetime
from pathlib import Path
import json
import re

import numpy as np

UNIT_AS_IS = "(use batch column as-is)"
MODES = ("cycombine", "monotonic", "per_cluster")
# Tab 6's defaults, as its run reads them (bc_config.get(key, default)).
DEFAULTS = {
    "cell_id_col": "cell_id",
    "batch_id_col": "batch_id",
    "cell_id_strategy": "Auto-generate global (cell_0, cell_1, ...)",
    "local_cell_id_col": "label",
    "tissue_col": "None",
    "tissue_val": 1,
    "area_col": "area",
    "area_min_pct": 0.1,
    "area_max_pct": 99.9,
    "outlier_low": 0.1,
    "outlier_high": 99.9,
    "arcsinh_cofactor": 5.0,
    "norm_method": "scale",
    "som_grid_dim": 8,
    "som_iterations": 100000,
    "som_rlen": 10,
    "cycombine_seed": 473,
    "min_cells_per_cluster": 10,
    "covariates": [],
    "parametric": True,
    "correction_mode": "cycombine",
    "batch_unit_col": UNIT_AS_IS,
    "batch_unit_pattern": "",
    "corrected_suffix": "_corr",
    "enable_post_qc": False,
    "filter_overcorrected": False,
    "qc_percentile_threshold": 95.0,
    "filter_distance": False,
    "max_correction_distance": 10.0,
    "add_corrected_indicator": False,
    "corrected_indicator_col": "corrected",
    "exclude_markers": ["DAPI", "Blank", "Empty", "DNA"],
}
# What Studio lets the user change, with the checks Tab 6's widgets apply.
EDITABLE = {
    "norm_method": lambda v: v in ("scale", "rank", "none"),
    "som_rlen": lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 1000,
    "cycombine_seed": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 2147483647,
    "correction_mode": lambda v: v in MODES,
    "batch_unit_col": lambda v: isinstance(v, str) and 0 < len(v) <= 128,
    "covariates": lambda v: isinstance(v, list) and all(isinstance(c, str) for c in v) and len(v) <= 20,
    "arcsinh_cofactor": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 65535,
    "som_grid_dim": lambda v: isinstance(v, int) and not isinstance(v, bool) and 2 <= v <= 50,
    "som_iterations": lambda v: isinstance(v, int) and not isinstance(v, bool) and 1000 <= v <= 500000,
    "min_cells_per_cluster": lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 500,
    "parametric": lambda v: isinstance(v, bool),
    "corrected_suffix": lambda v: isinstance(v, str) and re.fullmatch(r"_[A-Za-z0-9_]{1,30}", v) is not None,
}


def _tab6(config):
    return (config.get("tab_configurations") or {}).get("tab6_batch_correction") or {}


def config_of(config):
    """Tab 6's correction settings with its defaults filled in."""
    saved = _tab6(config).get("bc_config")
    return {**DEFAULTS, **(saved if isinstance(saved, dict) else {})}


def _path(root, value):
    path = Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def output_dir(root, mappings):
    value = (mappings or {}).get("bc_output_dir")
    return _path(root, value) if value else Path(root) / "outputs" / "batch_corrected"


def input_folder(root, mappings):
    """Explicit input, else tagged measurements when available, else Segmentation."""
    value = (mappings or {}).get("bc_input_dir")
    if value:
        return _path(root, value)
    for tagged in (_tagged_folder(root, mappings), Path(root) / "outputs" / "annotated_csv"):
        if tagged.is_dir() and any(p.suffix.lower() in (".csv", ".parquet") for p in tagged.iterdir() if p.is_file()):
            return tagged
    segmentation = next(((mappings or {})[key] for key in ("tab4_output_dir", "segmentation_output_dir") if (mappings or {}).get(key)), None)
    return _path(root, segmentation) if segmentation else Path(root) / "outputs" / "segmentation"


def _tagged_folder(root, mappings):
    from .settings import read_config

    saved = (read_config(root).get("tab_configurations") or {}).get("tab5_segmentation") or {}
    return _path(root, saved.get("seg_tag_output_dir") or (mappings or {}).get("seg_tag_output_dir") or "outputs/segmentation/tagged_features")


def tables(root, mappings):
    """Matched measurement tables for the selected ROI image directory."""
    from .cell_inputs import matches
    return {row["roi"]: row["path"] for row in matches(root, mappings, input_folder(root, mappings)) if row["path"]}


def save_folders(root, changes):
    """The same full-path directory controls as Annotation and Segmentation."""
    from spacemap_gui import persistence
    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - {"images", "input", "output"}:
        raise ValueError("Choose the Images, Cell tables or Output folder.")
    base = Path(root).resolve()
    with project_lock(root):
        config = read_config(root)
        for kind, value in changes.items():
            if not isinstance(value, str) or not value.strip() or len(value) > 1024:
                raise ValueError("Type the folder’s full path.")
            target = _path(base, value.strip())
            if (kind != "output" and not target.is_dir()) or (target.exists() and not target.is_dir()):
                raise ValueError(f"{target} is not a folder.")
            if not target.parent.is_dir():
                raise ValueError(f"{target.parent} is not a folder.")
            stored = "${PROJECT_ROOT}/" + target.relative_to(base).as_posix() if target.is_relative_to(base) else str(target)
            persistence.update_stored_config_key(config, {"images": "bc_images_dir", "input": "bc_input_dir", "output": "bc_output_dir"}[kind], stored)
        persistence.save_project_config(str(root), config)
    return config


def _stamp(path):
    info = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _saved_results(root, mappings, table):
    """Completed outputs with manifests, newest first, including the active table."""
    folder = output_dir(root, mappings)
    paths = list(folder.glob("batch_corrected_merged_*.parquet")) if folder.is_dir() else []
    if table and Path(table).is_file():
        paths.append(Path(table))
    for path in sorted(set(paths), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        manifest = last_run(path)
        if manifest:
            yield path, manifest


_HEADERS = {}  # path -> (size, modified, columns)


def _header(path):
    import pandas as pd

    info = Path(path).stat()
    cached = _HEADERS.get(path)
    if cached is None or cached[:2] != (info.st_size, info.st_mtime_ns):
        if Path(path).suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            columns = pq.read_schema(path).names
        else:
            columns = list(pd.read_csv(path, nrows=0).columns)
        cached = _HEADERS[path] = (info.st_size, info.st_mtime_ns, columns)
    return cached[2]


def _headers(paths):
    """Every table's columns: read side by side (each is a file open on the project's disk), then remembered."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(paths, pool.map(_header, paths)))


def last_run(table):
    """The ROIs, unit and markers of the correction that wrote *table* (saved beside it), or None."""
    import json

    if not table:
        return None
    try:
        found = json.loads(Path(table).with_suffix(".json").read_text())
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and isinstance(found.get("rois"), list) else None


def plan(root, mappings, config, table=None):
    """What a run would use: tables (one row per ROI, with its slide and state), markers, units, methods and settings,
    for the page. Completed output manifests provide saved-result status independently of adoption."""
    from spacemap_gui.tabs.tab6_batch_correction import _pick_batch_marker_defaults

    from . import samples, cell_inputs

    settings = config_of(config)
    matched = cell_inputs.matches(root, mappings, input_folder(root, mappings))
    found = {row["roi"]: row["path"] for row in matched if row["path"]}
    headers = _headers(list(found.values()))
    common = [name for name in next(iter(headers.values()), []) if all(name in h for h in headers.values())]
    sheet = samples.table(root, mappings)
    columns = [name for name in sheet["columns"] if name != "roi"]
    unit_columns = [name for name in columns if not any(row.get(name) == "Varies by cell" for row in sheet["rows"])]
    saved_choice = _tab6(config).get("bc_selected_markers")
    saved = [m for m in (saved_choice or []) if m in common]
    markers = saved if isinstance(saved_choice, list) else _pick_batch_marker_defaults(common, settings, columns)
    markers = [m for m in markers if not m.endswith(settings.get("corrected_suffix") or "_corr") and not m.endswith("_corr")]
    rois = list(found)
    units = [{"column": UNIT_AS_IS, "name": "ROI", "units": len(rois)}] + [
        {"column": name, "name": name, "units": len({row[name] for row in sheet["rows"] if row["roi"] in found})}
        for name in unit_columns
    ]
    unit = settings["batch_unit_col"]
    if unit != UNIT_AS_IS and unit not in unit_columns:
        unit = "slide" if "slide" in columns else UNIT_AS_IS
    chosen = _tab6(config).get("bc_config") if isinstance(_tab6(config).get("bc_config"), dict) else {}
    if unit == UNIT_AS_IS and "slide" in columns and not chosen.get("batch_unit_col"):
        unit = "slide"  # never set: correct between slides, as Tab 6's own guidance says
    # Never chosen: cyCombine's own method, Tab 6's default.
    mode = chosen.get("correction_mode") or "cycombine"
    folder = input_folder(root, mappings)
    tagged_folders = {_tagged_folder(root, mappings).resolve(), (Path(root) / "outputs/annotated_csv").resolve()}
    sheet_of = {row["roi"]: {k: v for k, v in row.items() if not k.startswith("_") and k != "roi"} for row in sheet["rows"]}
    last = last_run(table)
    from .sample_links import read as read_links
    links = read_links(root)
    records = {}
    for output, manifest in _saved_results(root, mappings, table):
        for roi in manifest["rois"]:
            records.setdefault(roi, (output, manifest))
    states = {}
    current_inputs = {roi: _stamp(path) for roi, path in found.items()}
    for roi, path in found.items():
        record = records.get(roi)
        status, count, name = "not_run", None, ""
        if record:
            output, manifest = record
            name = output.name
            stamp = (manifest.get("inputs") or {}).get(roi)
            count = (manifest.get("corrected_per_roi") or {}).get(roi)
            unchanged = all(current_inputs.get(key) == value for key, value in (manifest.get("inputs") or {}).items())
            used_columns = [manifest.get("unit"), *(manifest.get("config", {}).get("covariates") or [])]
            unchanged = unchanged and all(
                str(sheet_of.get(key, {}).get(column, "")) == str(manifest.get("samples", {}).get(key, {}).get(column, ""))
                for key in manifest["rois"] for column in used_columns if column and column != UNIT_AS_IS
            )
            unchanged = unchanged and (manifest.get("sample_links") or []) == links
            status = "legacy" if not stamp else "saved" if unchanged else "stale"
            if count == 0 and status == "saved":
                status = "excluded"
        states[roi] = {"status": status, "result": name, "corrected_cells": count, "corrected": status == "saved"}
    rows = [{**row, "table": Path(row["path"]).name if row["path"] else "", "slide": sheet_of.get(row["roi"], {}).get("slide", ""),
             "values": sheet_of.get(row["roi"], {}),
             "tagged": bool(row["path"]) and Path(row["path"]).resolve().parent in tagged_folders,
             **states.get(row["roi"], {"status": row["match"], "corrected": False})} for row in matched]
    return {
        "rows": rows,
        "last": {"name": Path(table).name, "rois": len(last["rois"]), "unit": last.get("unit", "")} if last else None,
        "tables": len(found),
        "folder": str(folder),
        "folders": {"images": str(cell_inputs.image_folder(root, mappings)), "input": str(folder), "output": str(output_dir(root, mappings))},
        "rois": rois,
        "missing": [row["roi"] for row in matched if row["match"] == "missing"][:50],
        "common": common,
        "markers": markers,
        "saved_markers": isinstance(saved_choice, list),
        "units": units,
        "unit": unit,
        "covariate_options": [name for name in columns if name != "slide"],
        "settings": {**{key: settings[key] for key in EDITABLE}, "correction_mode": mode},
        "filters": {key: settings[key] for key in ("tissue_col", "tissue_val", "area_col", "area_min_pct", "area_max_pct", "outlier_low", "outlier_high", "enable_post_qc")},
        "output": str(output_dir(root, mappings).relative_to(root))
        if output_dir(root, mappings).is_relative_to(root)
        else "",
    }


def save(root, changes, markers=None):
    """Save Studio's choices in Tab 6's settings, where classic Tab 6 reads them."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - set(EDITABLE):
        raise ValueError("Only the correction's own settings can be changed here.")
    for key, value in changes.items():
        if not EDITABLE[key](value):
            raise ValueError(f"{key.replace('_', ' ')} has a value Tab 6 does not accept.")
    if markers is not None and not (isinstance(markers, list) and all(isinstance(m, str) for m in markers)):
        raise ValueError("Choose the markers to correct.")
    with project_lock(root):
        config = read_config(root)
        saved = _tab6(config).get("bc_config")
        persistence.update_stored_config_key(
            config, "bc_config", {**(saved if isinstance(saved, dict) else {}), **changes}
        )
        if markers is not None:
            persistence.update_stored_config_key(config, "bc_selected_markers", list(markers))
        persistence.save_project_config(str(root), config)
        return config_of(read_config(root))


def _load(found, sheet, batch_col, config, report, links=()):
    """Tab 6's Load: each table with the ROI's Samples columns and its batch; cell IDs; the filter-column tables."""
    import pandas as pd

    from spacemap_gui.tabs.tab6_batch_correction import _assign_batch_correction_cell_ids

    frames, with_filter = [], set()
    tissue = config.get("tissue_col", "None")
    for index, (roi, path) in enumerate(found.items()):
        report(f"{roi}: reading cells ({index + 1} of {len(found)})…", 0.02 + 0.18 * index / max(len(found), 1))
        from .cell_inputs import read_frame
        frame = read_frame(path)
        targets = {rule["target"]: frame[rule["target"]].copy() for rule in links if rule["target"] in frame}
        matching_keys = read_frame(path, columns=list(targets), text_columns=list(targets)).to_dict('series') if targets else {}
        name = Path(path).name
        if tissue and tissue != "None" and tissue in frame.columns:
            with_filter.add(name)
        for column, value in sheet.get(roi, {}).items():
            frame[column] = value
        for column, values in targets.items():
            frame[column] = values
        from .sample_links import apply
        frame = apply(frame, roi, links, matching_keys=matching_keys)
        frame[batch_col] = roi  # the ROI, as the combined table names a cell's batch
        frame["original_filename"] = name
        frames.append(frame)
    if not frames:
        raise ValueError("No cell tables to correct: run segmentation first.")
    loaded = pd.concat(frames, ignore_index=True)
    return _assign_batch_correction_cell_ids(loaded, config, batch_col), with_filter


def _clean(loaded, config, markers, with_filter):
    """Tab 6's filters, in its order: filter column, area percentiles, outliers in most markers."""
    import numbers

    import pandas as pd

    filtered, steps = loaded, []
    tissue, value = config.get("tissue_col", "None"), config.get("tissue_val", 1)
    if tissue and tissue != "None" and tissue in filtered.columns:
        before = len(filtered)
        lacking = ~filtered["original_filename"].isin(with_filter)
        try:
            match = filtered[tissue] == value
            if isinstance(value, numbers.Number):
                match |= pd.to_numeric(filtered[tissue], errors="coerce") == float(value)
        except Exception:  # noqa: BLE001 - Tab 6 falls back to comparing text
            match = filtered[tissue].astype(str) == str(value)
        filtered = filtered[lacking | match]
        steps.append({"filter": f"{tissue} = {value}", "before": before, "after": len(filtered)})
    area = config.get("area_col", "area")
    if area in filtered.columns and len(filtered) > 0:
        numbers = pd.to_numeric(filtered[area], errors="coerce").replace([np.inf, -np.inf], np.nan)
        measured = numbers.dropna()
        if measured.empty:
            raise ValueError(f"The area column {area} has no finite numbers. Review cell filters in classic Batch correction.")
        low = np.percentile(measured, config.get("area_min_pct", 0.1))
        high = np.percentile(measured, config.get("area_max_pct", 99.9))
        before = len(filtered)
        filtered = filtered[(numbers >= low) & (numbers <= high)]
        steps.append({"filter": f"{area} between {low:.2f} and {high:.2f}", "before": before, "after": len(filtered)})
    present = [m for m in markers if m in filtered.columns]
    if present and len(filtered) > 0:
        before = len(filtered)
        values = pd.concat([pd.to_numeric(filtered[m], errors="coerce") for m in present], axis=1).fillna(0)
        high = (values > values.quantile(config.get("outlier_high", 99.9) / 100.0)).sum(axis=1)
        low = (values < values.quantile(config.get("outlier_low", 0.1) / 100.0)).sum(axis=1)
        outlier = ((high / len(present)) > 0.5) | ((low / len(present)) > 0.5)
        filtered = filtered[~outlier]
        steps.append({"filter": "outliers in most markers", "before": before, "after": len(filtered)})
    return filtered.reset_index(drop=True), steps


def job_settings(root, mappings, saved, markers, rois=None):
    """What a correction job needs: the saved settings, the markers, the ROIs' tables (*rois*, else every ROI's) and
    the Samples sheet."""
    from . import samples, sample_links

    found = tables(root, mappings)
    if rois is not None:
        unknown = [roi for roi in rois if roi not in found]
        if unknown:
            raise ValueError("No cell table for " + ", ".join(map(str, unknown[:4])) + ".")
        found = {roi: path for roi, path in found.items() if roi in set(rois)}
        if len(found) < 2:
            raise ValueError("Tick at least two tables: the correction compares them.")
    return {
        "project_root": str(root),
        "config": saved,
        "markers": markers,
        "tables": found,
        "samples": samples.values(root, mappings),
        "sample_links": sample_links.read(root),
        "output_dir": str(output_dir(root, mappings)),
    }


def save_table(frame, folder, stem="batch_corrected_merged"):
    """Save *frame* as ``<stem>_<time>.parquet`` in *folder* (a new name each time); return its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    target = folder / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.parquet"
    # Text columns may mix numbers and words (Samples columns): kept as text so Parquet takes them.
    frame = frame.copy()
    for column in frame.columns[frame.dtypes == object]:
        frame[column] = frame[column].astype("string")
    temporary = target.with_name("." + target.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(target)
    return target


def run(settings, folder, report):
    """The job: load, filter, correct between units, measure and save, as classic Tab 6 does."""
    import pandas as pd

    from spacemap.batch_correction_utils import (
        BatchCorrector,
        apply_post_correction_qc_filters,
        calculate_emd_reduction,
        calculate_mad_score,
        detect_overcorrected_cells,
    )

    root = Path(settings["project_root"])
    config, markers = settings["config"], list(settings["markers"])
    found, sheet = settings["tables"], settings["samples"]
    inputs = {roi: _stamp(path) for roi, path in found.items()}
    batch_col, cell_col = config.get("batch_id_col", "batch_id"), config.get("cell_id_col", "cell_id")
    reserved = {cell_col, batch_col, "label", "original_filename", *markers}
    collisions = sorted({column for values in sheet.values() for column in values if column in reserved})
    if collisions:
        raise ValueError("Rename these Samples columns so they do not replace cell IDs or marker measurements: " + ", ".join(collisions))
    loaded, with_filter = _load(found, sheet, batch_col, config, report, settings.get("sample_links", []))
    if loaded[cell_col].isna().any() or loaded[cell_col].duplicated().any():
        raise ValueError("Cell IDs must be unique. Choose global IDs or ROI + local cell ID in classic Batch correction.")
    report(f"Filtering {len(loaded):,} cells as Tab 6 does…", 0.22)
    cleaned, filters = _clean(loaded, config, markers, with_filter)
    work = cleaned.copy()
    if batch_col != "batch":
        work = work.rename(columns={batch_col: "batch"})
    unit = config.get("batch_unit_col", UNIT_AS_IS)
    if unit and unit != UNIT_AS_IS:
        if unit not in work.columns:
            raise ValueError(f"The cells have no {unit} column: add it in Samples first.")
        if work[unit].isna().any() or work[unit].astype(str).str.strip().eq("").any():
            raise ValueError(f"Some cells have no {unit}. Complete this column in Samples before correcting.")
        work["batch"] = work[unit].astype(str)
    missing = [c for c in [cell_col, "batch", *markers] if c not in work.columns]
    if missing:
        raise ValueError(f"These columns are missing from the cell tables: {', '.join(missing[:8])}.")
    work.loc[:, markers] = work[markers].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    work.loc[:, "batch"] = work["batch"].astype(str)
    valid = ~work[markers].isna().any(axis=1)
    work = work.loc[valid].copy()
    filters.append({"filter": "finite marker measurements", "before": len(cleaned), "after": len(work)})
    units = sorted(work["batch"].unique())
    if len(units) < 2:
        raise ValueError(f"There is only one {unit if unit != UNIT_AS_IS else 'batch'}: nothing to correct between.")
    mode = config.get("correction_mode", "cycombine")
    if len(set(markers)) != len(markers) or len(markers) < 2:
        raise ValueError("Choose at least two distinct measured markers for ComBat’s empirical Bayes estimates.")
    if any(m.endswith(config.get("corrected_suffix") or "_corr") or m.endswith("_corr") for m in markers):
        raise ValueError("Choose raw measured markers, not already corrected columns (arcsinh is applied once).")
    covariates = list(config.get("covariates") or [])
    if covariates:
        from spacemap.cycombine import _confounded, _dummies
        missing_covars = [c for c in covariates if c not in work.columns]
        if missing_covars:
            raise ValueError("Missing biological covariates: " + ", ".join(missing_covars))
        if work[covariates].isna().any().any() or work[covariates].astype(str).apply(lambda c: c.str.strip().eq("")).any().any():
            raise ValueError("Complete the selected biological covariates for every cell first.")
        codes = pd.factorize(pd.MultiIndex.from_frame(work[covariates].astype(str)))[0].astype(str)
        if len(set(codes)) > 1 and _confounded(work["batch"].to_numpy(), _dummies(codes, sorted(set(codes)))):
            raise ValueError("The biological covariate is confounded with the correction unit: their effects cannot be separated. Review Samples and Correct between; cyCombine cannot protect this design.")
    label = {"cycombine": "cyCombine", "monotonic": "monotonic", "per_cluster": "earlier cyCombine approximation"}[mode]
    report(f"Correcting {len(markers)} markers between {len(units)} units ({label})…", 0.3)
    corrector = BatchCorrector(
        markers=markers,
        normalization_method=config.get("norm_method", "scale"),
        som_grid_size=(int(config["som_grid_dim"]), int(config["som_grid_dim"])),
        som_iterations=int(config["som_iterations"]),
        random_seed=42,
        verbose=False,
        mode="som_clustered",
        min_cells_per_cluster=int(config["min_cells_per_cluster"]),
        covariates=list(config.get("covariates") or []),
        parametric=bool(config.get("parametric", True)),
        arcsinh_cofactor=float(config.get("arcsinh_cofactor", 5.0)),
        correction_mode=mode,
        som_rlen=int(config.get("som_rlen", 10)),
        cycombine_seed=int(config.get("cycombine_seed", 473)),
    )
    corrected = corrector.correct_batch_effects(work)
    report("Measuring pooled distribution changes (EMD, MAD)…", 0.85)
    if len(work) > 10000:
        sample = work.sample(n=10000, random_state=42)
        sample_corrected = corrected[corrected[cell_col].isin(sample[cell_col])]
    else:
        sample, sample_corrected = work, corrected
    cofactor = float(config.get("arcsinh_cofactor", 5.0))
    emd = float(calculate_emd_reduction(sample, sample_corrected, markers, batch_col="batch", cofactor=cofactor))
    mad = float(calculate_mad_score(sample, sample_corrected, markers, batch_col="batch", cofactor=cofactor))
    retained, qc = corrected.copy(), None
    if config.get("enable_post_qc"):
        flagged = detect_overcorrected_cells(
            data=corrected,
            markers=markers,
            batch_col="batch",
            percentile_threshold=float(config.get("qc_percentile_threshold", 95.0)),
            cofactor=cofactor,
        )
        thresholds = {
            "enable_post_correction_qc": True,
            "filter_overcorrected": bool(config.get("filter_overcorrected")),
        }
        if config.get("filter_distance"):
            thresholds["max_correction_distance"] = float(config.get("max_correction_distance", 10.0))
        retained, qc = apply_post_correction_qc_filters(data=flagged, qc_thresholds=thresholds)
        qc = {key: value.item() if isinstance(value, np.generic) else value for key, value in qc.items()}
    if batch_col != "batch":
        retained = retained.rename(columns={"batch": batch_col})
    suffix = str(config.get("corrected_suffix") or "_corr")
    final = [f"{m}{suffix}" for m in markers]
    retained = retained.rename(columns={f"{m}_corr": f"{m}{suffix}" for m in markers if suffix != "_corr"})
    # Every loaded cell is kept; cells filtered out have no corrected values (Tab 6's output).
    base = loaded.drop(columns=[c for c in final if c in loaded.columns])
    result = base.merge(retained[[cell_col, *final]], on=cell_col, how="left", validate="one_to_one")
    if config.get("add_corrected_indicator") and config.get("corrected_indicator_col"):
        indicator = config["corrected_indicator_col"]
        result[indicator] = result[cell_col].isin(retained[cell_col]).astype(int)
    report("Saving the combined table (Parquet)…", 0.92)
    target = save_table(result, output_dir(root, {"bc_output_dir": settings.get("output_dir")}))
    counts = result.loc[result[final].notna().all(axis=1)].groupby(batch_col).size().to_dict()
    summary = corrector.run_summary
    combat = summary.get("combat", {})
    warnings = []
    if combat.get("n_clusters_covar_fallback"):
        warnings.append(f"Biological covariates were omitted in {combat['n_clusters_covar_fallback']} SOM groups because they were confounded or too imbalanced.")
    if combat.get("n_cells_one_batch"):
        warnings.append(f"{combat['n_cells_one_batch']} cells were in single-batch SOM groups and kept their arcsinh measurements.")
    if combat.get("n_cells_kept_as_measured"):
        warnings.append(f"{combat['n_cells_kept_as_measured']} cells retained at least one measured value because ComBat could not estimate it.")
    manifest = {"version": 2, "rois": list(found), "unit": unit, "markers": markers, "mode": mode,
                "config": config, "inputs": inputs, "samples": sheet, "sample_links": settings.get("sample_links", []), "filters": filters,
                "post_qc": qc,
                "corrected_per_roi": {roi: int(counts.get(roi, 0)) for roi in found},
                "summary": summary, "warnings": warnings,
                "diagnostics": {"scope": "pooled batches; not cyCombine population-level benchmark scores",
                                "emd_reduction": round(emd, 4), "relative_mad_change": round(mad, 4)}}
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(target.with_suffix(".json"))
    return {
        "output": str(target),
        "name": target.name,
        "cells": int(len(result)),
        "corrected_cells": int(len(retained)),
        "filters": filters,
        "markers": markers,
        "unit": unit,
        "units": [str(u) for u in units],
        "mode": mode,
        "emd_reduction": round(emd, 4),
        "mad": round(mad, 4),
        "qc": qc,
        "warnings": warnings,
        "summary": summary,
        "tables": len(found),
    }
