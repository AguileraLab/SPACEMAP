"""Small step work lists, based on files on disk rather than session flags."""

import os
import csv
import json
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from spacemap.viewer.catalog import sample_of
from . import annotation, background, rois
from .shell import STEPS
from .settings import read_config

# Existing directory keys, with the usual layout as a fallback. No recursive
# scans: large image/segmentation contents are never read to build the sidebar.
LAYOUT = {
    "tab0": ([], [("Project settings", (), ".")]),
    "tab1": (
        [("CODEX", ("batch_codex_dir",), "data/codex"), ("H&E", ("batch_he_dir",), "data/he")],
        [("Registered images", ("tab1_batch_output_dir", "registration_output_dir"), "outputs/registered")],
    ),
    # Tab 2 measures the images in its own folder (the ROIs unless chosen otherwise): see background.folders.
    "tab2": (
        [("Images", ("tab3_output_dir", "roi_output_dir"), "outputs/rois")],
        [
            ("Threshold tables", ("threshold_dir",), "data/thresholds"),
            ("Background removed", ("tab2_output_dir", "bg_subtraction_output_dir"), "outputs/background_subtracted"),
        ],
    ),
    "tab3": (
        [("Images", ("tab3_input_dir",), "outputs/background_subtracted")],
        [("ROI images", ("tab3_output_dir", "roi_output_dir"), "outputs/rois")],
    ),
    "tab4": (
        [("ROI images", ("annot_ta_input_dir", "tab3_output_dir"), "outputs/rois")],
        [
            (
                "Annotations",
                ("annot_ta_output_dir", "tab4_annot_geojson_dir"),
                "outputs/processed_annotations/stage0_tissue_annotation",
            ),
            ("Zones", ("annot_ta_pool_output_dir",), "outputs/processed_annotations/stage1b_compartments_zones"),
        ],
    ),
    "tab5": (
        [("ROI images", ("tab4_input_dir", "tab3_output_dir"), "outputs/rois")],
        [
            ("Cell measurements", ("tab4_output_dir", "cell_seg_dir"), "outputs/segmentation"),
            ("Cell outlines", ("tab4_geojson_dir",), "outputs/segmentation/cellpose/boundaries"),
        ],
    ),
    # Cell tagging keeps its folders in Tab 5's own settings (tagging.values); these are the usual places.
    "tagging": (
        [("Cell tables", ("tab4_csv_dir", "tab4_output_dir"), "outputs/segmentation/cellpose/features")],
        [("Tagged tables", (), "outputs/segmentation/tagged_features")],
    ),
    "tab6": (
        [("Cell measurements", ("bc_input_dir",), "outputs/annotated_csv")],
        [("Combined measurements", ("bc_output_dir",), "outputs/batch_corrected")],
    ),
    "tab7": (
        [("ROI images", ("tab3_output_dir",), "outputs/rois")],
        [("Phenotyping results", ("phenotyping_output_dir",), "outputs/phenotyping")],
    ),
    "tab8": (
        [("Measurements", ("spatial_input_dir", "bc_output_dir"), "outputs/batch_corrected")],
        [("Spatial results", ("spatial_output_dir",), "outputs/spatial_analysis")],
    ),
    "tab9": (
        [("Measurements", ("global_visualization_dir", "bc_output_dir"), "outputs/batch_corrected")],
        [("Figures", ("tab9_actual_output_dir",), "outputs/spatial_analysis")],
    ),
}
DONE_LABELS = {
    "tab1": "Registered",
    "tab4": "Annotated",
    "tab5": "Segmented",
    "tagging": "Tagged",
    "tab8": "Analysed",
    "tab9": "Figures saved",
}
SUFFIXES = {".tif", ".tiff", ".qptiff", ".svs", ".csv", ".parquet", ".geojson", ".png", ".pdf", ".svg"}


@lru_cache(maxsize=8)
def _runs(path, size, modified):
    if size > 8_000_000:
        return []
    try:
        with open(path, newline="") as handle:
            return list(deque(csv.DictReader(handle), maxlen=200))
    except (OSError, csv.Error):
        return []


def last_run(root, step):
    from spacemap_gui.run_log import TAB_LABELS

    path = root / "logs/run_log.csv"
    if not path.exists():
        return None
    stamp = path.stat()
    return next(
        (
            row
            for row in reversed(_runs(str(path), stamp.st_size, stamp.st_mtime_ns))
            if row.get("tab") == TAB_LABELS.get(step)
        ),
        None,
    )


def _mentioned_files(value, key=""):
    if isinstance(value, dict):
        return set().union(*(_mentioned_files(item, str(name)) for name, item in value.items())) if value else set()
    if isinstance(value, list):
        return set().union(*(_mentioned_files(item, key) for item in value)) if value else set()
    if isinstance(value, str) and any(word in key.lower() for word in ("file", "image", "roi", "sample", "path")):
        return {Path(value).name}
    return set()


def _folder(root, mappings, keys, fallback):
    value = next((mappings[key] for key in keys if mappings.get(key)), None)
    path = Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser() if value else root / fallback
    return path


def _files(folder, limit=10000):
    files = []
    truncated = False
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.name)
                if (
                    path.suffix.lower() in SUFFIXES
                    and entry.is_file()
                    and not entry.name.endswith((".csv.parquet", ".labels.parquet"))
                ):
                    files.append(entry.name)
                    if len(files) >= limit:
                        truncated = True
                        break
    except (OSError, TypeError):
        pass
    return sorted(files), truncated


@lru_cache(maxsize=4)
def _phenotyped(table: str, stamp: str) -> dict[str, list[str]]:
    """ROI name -> method columns (phenotype_*, not reference columns) with values for its cells.

    Phenotyping run in the classic tab leaves no Studio job, so without this
    every ROI of a phenotyped project was listed as "Not run". *stamp* (the
    table's fingerprint and its saved result columns) makes the cache follow
    changes to the table. Each distinct value is judged once, so a table of
    millions of cells takes about a second.
    """
    from spacemap import dataio
    from spacemap.viewer.reference import is_reference_column

    columns = [c for c in dataio.columns(table) if c.casefold().startswith("phenotype_") and not is_reference_column(c)]
    if not columns:
        return {}
    # With the labels applied, the result columns Studio's jobs saved beside the table are read too.
    frame = dataio.load(table, columns=["batch_id", *columns])
    rois, names = pd.factorize(frame["batch_id"].astype(str))
    found = {str(name): [] for name in names}
    for column in columns:
        if column not in frame.columns:
            continue
        codes, values = pd.factorize(frame[column])  # blanks (NaN) get -1
        blank = np.array([not str(value).strip() for value in values], dtype=bool)
        filled = codes >= 0
        if blank.any():
            filled &= ~blank[np.clip(codes, 0, None)]
        for roi in np.flatnonzero(np.bincount(rois[filled], minlength=len(names))):
            found[str(names[roi])].append(column)
    return {roi: methods for roi, methods in found.items() if methods}


def _table_stamp(table):
    """The table's fingerprint with its saved result columns and labels (kept beside it)."""
    from spacemap.viewer.sources import fingerprint

    parts = [fingerprint(table)]
    for suffix in (".pending", ".labels.parquet"):
        beside = Path(str(table) + suffix)
        if beside.exists():
            parts.append(str(max([beside.stat().st_mtime_ns] + [p.stat().st_mtime_ns for p in beside.glob("*")])))
    return "-".join(parts)


def work_list(channel, step):
    from spacemap_gui.viewer import registry
    from spacemap_gui.viewer.ui import window_images
    from .jobs import recent

    if step not in LAYOUT:
        raise ValueError("Choose an analysis step.")
    if registry.session_view(channel, mark_seen=False) is None:
        raise ValueError("Open Studio from SPACEMAP first.")
    project = registry.session_project(channel) or {}
    if not project.get("root"):
        return {"step": step, "items": [], "outputs": [], "note": "Load a project in classic SPACEMAP first."}
    root = Path(project["root"])
    config = read_config(root)
    if step == "tagging":
        return _tagging_list(channel, root, config)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    inputs, outputs = LAYOUT[step]
    groups = []
    truncated = False
    found_outputs = []
    for label, keys, fallback in inputs:
        eligible = (
            {".tif", ".tiff", ".qptiff", ".svs"}
            if step in {"tab1", "tab2", "tab3", "tab4", "tab5", "tab7"}
            else {".csv", ".parquet"}
        )
        folder = _folder(root, mappings, keys, fallback)
        if step == "tab2":
            folder = background.folders(root, mappings, config)[0]
        if step == "tab3":
            folder = rois.folders(root, mappings, config)[0]  # Tab 3's own rule: its chosen folder, else the pipeline's
        names, cut = _files(folder)
        names = [name for name in names if Path(name).suffix.lower() in eligible]
        truncated |= cut
        groups.append((label, names, folder))
    for label, keys, fallback in outputs:
        folder = _folder(root, mappings, keys, fallback)
        names, cut = _files(folder)
        truncated |= cut
        found_outputs.extend({"name": name, "kind": label, "sample": sample_of(name)} for name in names)
    images = window_images(channel) if step in {"tab1", "tab2", "tab3", "tab4", "tab5", "tab7"} else []
    rows = {}
    locations = {}  # sample -> the input files' paths (kept here: the browser gets ids and names only)
    for kind, names, folder in groups:
        for name in names:
            locations.setdefault(sample_of(name), set()).add(str((folder / name).resolve()))
            sample = sample_of(name)
            row = rows.setdefault(
                sample, {"name": sample, "inputs": [], "state": "not_run", "state_label": "Not run", "outputs": []}
            )
            row["inputs"].append({"name": name, "kind": kind})
    if step == "tab0":
        rows[root.name] = {
            "name": root.name,
            "inputs": [],
            "state": "done" if config else "not_run",
            "state_label": "Saved" if config else "Not saved",
            "outputs": [],
        }
    if not rows and step != "tab0":
        # Inputs moved or cleaned up, results kept (a combined table): list the results.
        for item in found_outputs:
            rows.setdefault(
                item["sample"],
                {"name": item["sample"], "inputs": [], "state": "not_run", "state_label": "Not run", "outputs": []},
            )
    for row in rows.values():
        # An ROI cut from an image (P11_roi_1 from P11) is that image's Tab 3 output.
        row["outputs"] = [
            item["name"]
            for item in found_outputs
            if item["sample"] == row["name"] or (step == "tab3" and item["sample"].startswith(row["name"] + "_roi"))
        ]
        if row["outputs"]:
            row.update(state="done", state_label=DONE_LABELS.get(step, "Output found"))
        elif step == "tab6" and found_outputs:
            # Batch correction combines every table into one result.
            row.update(state="done", state_label="Combined")
        if step == "tab3" and row["inputs"]:
            _, shapes_dir, extracted_dir = rois.folders(root, mappings, config)
            image = row["inputs"][0]["name"]
            names = rois.roi_names(rois.roi_file(shapes_dir, image))
            done = len(rois.extracted(extracted_dir, image, names))
            if names and done == len(names):
                row.update(state="done", state_label=f"{done} ROI{'s' if done > 1 else ''} extracted")
            elif names:
                row.update(state="partial", state_label=f"{done} of {len(names)} ROIs extracted")
        if step == "tab4" and row["inputs"]:
            _, annotated = annotation.folders(root, mappings)
            sample = annotation.sample_id(row["inputs"][0]["name"])
            if annotation.zones(annotation.zones_folder(root, mappings), sample).exists():
                row.update(state="done", state_label="Zones")
            elif annotation.domains(annotated, sample).exists():
                row.update(state="done", state_label="Domains exported")
            elif annotation.classes(annotated, sample).exists():
                row.update(state="done", state_label="Classes exported")
            elif annotation.valid_tissue(annotated, sample).exists():
                row.update(state="done", state_label="Tissue found")
        if step == "tab2" and row["inputs"]:
            _, tables, removed = background.folders(root, mappings, config)
            image = row["inputs"][0]["name"]
            table, output = background.table_path(tables, image), background.output_path(removed, image)
            row["outputs"] = [path.name for path in (table, output) if path.exists()]
            if output.exists():
                row.update(state="done", state_label="Background removed")
            elif table.exists():
                row.update(state="partial", state_label="Levels saved")
            else:
                row.update(state="not_run", state_label="Not run")
        if step == "tab1" and {item["kind"] for item in row["inputs"]} != {"CODEX", "H&E"}:
            row.update(state="needs_review", state_label="Missing image pair")
        candidates = [entry for entry in images if entry["sample"] == row["name"]]
        # The same name can be in two folders (a new registration folder, say): open the file this row lists.
        listed = [entry for entry in candidates if _image_path(channel, entry["id"]) in locations.get(row["name"], ())]
        if listed or candidates:
            row["image"] = (listed or candidates)[-1]
    if step == "tab7" and project.get("unified") and Path(project["unified"]).is_file():
        try:
            methods = _phenotyped(str(project["unified"]), _table_stamp(project["unified"]))
        except (OSError, ValueError, KeyError):
            methods = {}
        for row in rows.values():
            found = methods.get(row["name"])
            if found:
                row.update(
                    state="done",
                    state_label=f"Phenotyped ({len(found)} method{'s' if len(found) > 1 else ''})",
                    methods=found,
                )
    if step == "tab7":
        jobs = recent(root)
        latest = next((job for job in jobs if job["kind"] in {"resolve_all", "mr_ml"}), None)
        if latest:
            for row in rows.values():
                if latest["status"] == "done" and row["name"] in latest.get("result", {}).get("roi_names", []):
                    row.update(state="done", state_label="Result saved")
                elif latest["status"] == "failed":
                    row.update(state="failed", state_label="Run failed")
                elif latest["status"] in {"starting", "running", "stopping"}:
                    row.update(state="running", state_label="Working")
                elif latest["status"] == "cancelled":
                    row.update(state="needs_review", state_label="Run stopped")
    run = last_run(root, step)
    if run and run.get("status") in {"failed", "error", "cancelled", "interrupted"}:
        try:
            mentioned = _mentioned_files(json.loads(run.get("settings_json") or "{}"))
        except ValueError:
            mentioned = set()
        for row in rows.values():
            if row["name"] in mentioned or any(item["name"] in mentioned for item in row["inputs"]):
                row.update(
                    state="failed" if run["status"] in {"failed", "error"} else "needs_review",
                    state_label="Run failed" if run["status"] in {"failed", "error"} else "Run stopped",
                )
    return {
        "step": step,
        "items": list(rows.values()),
        "outputs": found_outputs[:500],
        "truncated": truncated,
        "note": "States come from saved files and jobs. An output still needs your quality review.",
        "last_run": {"status": run.get("status"), "action": run.get("action"), "message": run.get("error", "")[:500]}
        if run
        else None,
        "actions": {
            "preview": bool(images),
            "run": "resolve_all" if step == "tab7" else "classic",
            "settings": "resolve_config" if step == "tab7" else "classic",
        },
    }


def _image_path(channel, image_id):
    from spacemap_gui.viewer import registry

    entries = registry.session_images(channel) or []
    try:
        return str(Path(entries[image_id]["path"]).resolve())
    except (IndexError, KeyError, TypeError, OSError):
        return None


def _tagging_list(channel, root, config):
    """The Cell tagging step's items: each cell table (or each ROI of a combined table), tagged or not."""
    from spacemap_gui.tabs import cell_tagging_ui as tab5
    from spacemap_gui.viewer.ui import window_images

    from . import tagging

    state = tagging.values(root, config)
    found = tagging.sources(root, state)
    tagged = tagging.outputs(state, found)
    images = window_images(channel)
    folder = Path(state[tab5.OUTPUT_DIR_KEY])
    names, truncated = _files(folder) if folder.is_dir() else ([], False)
    rows = []
    for row in found["rows"]:
        name = sample_of(row["sample"])
        output = tagged.get("*") or tagged.get(row["source"])
        item = {"name": name, "inputs": [{"name": row["source"], "kind": "Cell table"}], "outputs": [output] if output else [],
                "annotation": row["annotation"]}
        if not row["annotation"]:
            item.update(state="needs_review", state_label="No annotation")
        elif output:
            item.update(state="done", state_label="Tagged")
        else:
            item.update(state="not_run", state_label="Not tagged")
        candidates = [entry for entry in images if entry["sample"] == name]
        if candidates:
            item["image"] = candidates[-1]
        rows.append(item)
    return {
        "step": "tagging",
        "items": rows,
        "outputs": [{"name": name, "kind": "Tagged tables", "sample": sample_of(name)} for name in names if name.endswith(".csv")][:500],
        "truncated": truncated,
        "note": "States come from saved files: a table is tagged when its tagged copy exists.",
        "last_run": None,
        "actions": {"preview": bool(images), "run": "classic", "settings": "classic"},
    }


def overview(channel):
    """One state per step for the Studio rail: done, partial or none."""
    states = {}
    for step, *_ in STEPS:
        try:
            items = work_list(channel, step)["items"]
        except (ValueError, OSError):
            states[step] = "unknown"
            continue
        done = sum(item["state"] == "done" for item in items)
        started = done or any(item["state"] == "partial" for item in items)
        states[step] = "none" if not started else "done" if done == len(items) else "partial"
    return {"states": states}


def project_status(channel):
    """The Project step: each sample's state in every step, the step folders, and the project file."""
    import time

    from spacemap_gui.viewer import registry

    from .settings import config_path

    if registry.session_view(channel, mark_seen=False) is None:
        raise ValueError("Open Studio from SPACEMAP first.")
    project = registry.session_project(channel) or {}
    if not project.get("root"):
        return {"project": None}
    root = Path(project["root"])
    config = read_config(root)
    mappings = {**project.get("mappings", {}), **config.get("directory_mappings", {})}
    samples, folders = {}, []
    for step, name, *_ in STEPS[1:]:
        try:
            items = work_list(channel, step)["items"]
        except (ValueError, OSError):
            items = []
        for item in items:
            samples.setdefault(item["name"], {})[step] = {"state": item["state"], "label": item["state_label"]}
        inputs, outputs = LAYOUT[step]
        for kind, (label, keys, fallback) in [("in", entry) for entry in inputs] + [
            ("out", entry) for entry in outputs
        ]:
            folder = _folder(root, mappings, keys, fallback)
            names, cut = _files(folder) if folder.is_dir() else ([], False)
            try:
                shown = str(folder.relative_to(root))
            except ValueError:
                shown = str(folder)
            folders.append(
                {
                    "step": step,
                    "step_name": name,
                    "label": label,
                    "kind": kind,
                    "path": shown,
                    "exists": folder.is_dir(),
                    "files": len(names),
                    "more": cut,
                }
            )
    path = config_path(root)
    saved = path.stat() if path.exists() else None
    return {
        "project": {
            "name": root.name,
            "root": str(root),
            "file": path.name,
            "saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(saved.st_mtime)) if saved else None,
            "size_kb": round(saved.st_size / 1024) if saved else 0,
            "read_only": bool(project.get("read_only")),
        },
        "steps": [{"id": step, "name": name} for step, name, *_ in STEPS[1:]],
        "samples": [{"name": sample, "steps": states} for sample, states in sorted(samples.items())],
        "folders": folders,
    }
