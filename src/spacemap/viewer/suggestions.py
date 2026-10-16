"""On-demand reference suggestions. No work runs while an image opens.

Distance-weighted neighbour votes are estimates, not calibrated probabilities.
Only the chosen ROI, label column and measurement source teach the model.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import cell_profiles, cells, reference
from .sources import fingerprint, image_info, open_source

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cell-suggestions")
_LOCK = threading.RLock()
_JOBS = OrderedDict()
_MODELS = cells._Cache(4)
_PIXELS = cells._Cache(2)
_TABLES = cells._Cache(2)
_EMPTY = {"", "nan", "none", "unknown", "unassigned", "unclassified", "unlabeled", "unlabelled"}


def named(values):
    return np.asarray(["" if pd.isna(v) or str(v).strip().casefold() in _EMPTY else str(v).strip() for v in values])


def revision(settings):
    paths = [settings["table"], reference.path_for(settings["table"]), settings["image"]]
    if settings["source"] != "table":
        paths += [settings["pyramid"], settings["segmentation"]]
    return tuple(fingerprint(p) if Path(p).exists() else "" for p in paths) + cells._edits_token(settings["table"])


def _keys(settings):
    return {k: settings[k] for k in ("roi_column", "label_column")}


def _table_values(settings, positions, *, cache=False):
    """One selected-row read; overlay edited marker columns if present."""
    from spacemap import labels, pending_columns

    table, markers = settings["table"], settings["markers"]
    pending = [p for p in pending_columns.pending(table) if not p["stale"] and p["column"] in markers]
    edited = {p["column"] for p in pending}
    edits = labels.load(table)
    if not edits.empty:
        edits = edits[edits["column"].isin(markers)]
        edited.update(edits["column"])
    key = (
        table,
        fingerprint(table),
        settings["roi"],
        tuple(markers),
        tuple((p["column"], fingerprint(p["file"])) for p in pending),
        int(pd.util.hash_pandas_object(edits, index=False).sum()) if not edits.empty else 0,
    )
    cached = _TABLES.get(key)
    if cached is not None:
        rows, values = cached
        locations = np.searchsorted(rows, positions)
        if np.all(locations < len(rows)) and np.array_equal(rows[locations], positions):
            return values[locations]
    raw = cells.fast_rows(table, positions, markers)
    result = np.column_stack(
        [
            pd.to_numeric(
                pd.Series(
                    cells.values_at_rows(table, positions, marker)
                    if raw is None or marker in edited
                    else raw.column(marker).to_numpy()
                ),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            for marker in markers
        ]
    )
    if cache:
        _TABLES.put(key, (positions.copy(), result))
    return result


def _pixel_key(settings):
    return (
        settings["table"],
        fingerprint(settings["table"]),
        settings["roi"],
        settings["roi_column"],
        settings["label_column"],
        settings["image"],
        fingerprint(settings["image"]),
        fingerprint(settings["pyramid"]),
        fingerprint(settings["segmentation"]),
        tuple(settings["markers"]),
        tuple(sorted(settings["backgrounds"].items())),
        settings["corrected"],
    )


def _profiles(settings, labels, progress):
    cached = _PIXELS.get(_pixel_key(settings))
    if cached is not None:
        ids, values = cached
        places = np.searchsorted(ids, labels)
        good = (places < len(ids)) & (ids[np.minimum(places, len(ids) - 1)] == labels)
        result = np.full((len(labels), len(settings["markers"])), np.nan)
        result[good] = values[places[good]]
        return result
    values = []
    for i, label in enumerate(labels):
        progress(f"Reading example cells: {i + 1} of {len(labels)}")
        profile = cell_profiles.image_profile(
            settings["image"],
            settings["pyramid"],
            settings["segmentation"],
            int(label),
            settings["backgrounds"],
            corrected=settings["corrected"],
        )
        while profile["status"] == "preparing":
            progress(profile.get("message", "Preparing cell measurements..."))
            time.sleep(0.25)
            profile = cell_profiles.image_profile(
                settings["image"],
                settings["pyramid"],
                settings["segmentation"],
                int(label),
                settings["backgrounds"],
                corrected=settings["corrected"],
            )
        if profile["status"] != "ready":
            raise ValueError(profile.get("message", "Open a cell's image measurements first, then try again."))
        by_name = {m["marker"]: m["signal"] for m in profile["markers"]}
        values.append([by_name.get(marker, np.nan) for marker in settings["markers"]])
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self, ids, values, names):
        valid = np.all(np.isfinite(values), axis=1)
        self.omitted = int((~valid).sum())
        self.ids, values, names = ids[valid], values[valid], names[valid]
        self.classes, counts = np.unique(names, return_counts=True)
        if len(self.classes) < 2 or np.any(counts < 2):
            raise ValueError(
                "Add at least two checked examples of each phenotype, with at least two phenotypes in this ROI."
            )
        values = np.arcsinh(values / 5.0)
        self.mean, self.scale = values.mean(axis=0), values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        self.tree = cKDTree((values - self.mean) / self.scale)
        self.codes = np.searchsorted(self.classes, names)

    def predict(self, values, ids):
        values = np.asarray(values, dtype=float)
        valid = np.all(np.isfinite(values), axis=1)
        output = np.full((len(values), len(self.classes)), np.nan)
        if not valid.any():
            return output
        x = (np.arcsinh(values[valid] / 5.0) - self.mean) / self.scale
        distances, places = self.tree.query(x, k=min(8, len(self.ids)), workers=1)
        weights = 1 / (1 + distances)
        # Never let a cell vote for its own saved phenotype.
        weights[self.ids[places] == np.asarray(ids)[valid, None]] = 0
        # Keep up to seven other examples; additive smoothing avoids certainty.
        weights[np.cumsum(weights > 0, axis=1) > 7] = 0
        votes = np.full((len(x), len(self.classes)), 0.5)
        rows = np.repeat(np.arange(len(x)), places.shape[1])
        np.add.at(votes, (rows, self.codes[places].ravel()), weights.ravel())
        output[valid] = votes / votes.sum(axis=1, keepdims=True)
        return output


def _model(settings, stamp, progress):
    key = (json.dumps(settings, sort_keys=True), stamp)
    cached = _MODELS.get(key)
    if cached is not None:
        return cached
    table, roi = settings["table"], settings["roi"]
    positions, ids = cells.roi_index(table, roi, **_keys(settings))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Repeated cell numbers were found in this ROI. Check the table before using suggestions.")
    names = named(cells.values_at_rows(table, positions, settings["column"]))
    with reference.database(table) as db:
        flagged = (
            [
                r[0]
                for r in db.execute(
                    "SELECT label FROM reviews WHERE roi=? AND column_name=? AND flagged=1", (roi, settings["column"])
                )
            ]
            if db is not None
            else []
        )
    keep = (names != "") & ~np.isin(ids, flagged) & (ids > 0)
    classes, counts = np.unique(names[keep], return_counts=True)
    unready = classes[counts < 2].tolist()
    keep &= ~np.isin(names, unready)
    classes, counts = classes[counts >= 2], counts[counts >= 2]
    if len(classes) < 2:
        raise ValueError(
            "Add at least two checked examples of each phenotype, with at least two phenotypes in this ROI."
        )
    available_examples = int(keep.sum())
    if len(classes) > 1000:
        raise ValueError("This column has too many different names. Choose a phenotype label column.")
    # A large consensus column must not create a million-example search tree.
    # Deterministic sampling is disclosed in the result and the viewer.
    per_class = max(2, min(100, 2000 // len(classes)))
    rng = np.random.default_rng(0)
    for category, count in zip(classes, counts):
        if count > per_class:
            candidates = np.flatnonzero(keep & (names == category))
            keep[candidates] = False
            keep[rng.choice(candidates, per_class, replace=False)] = True
    progress(f"Learning from {int(keep.sum())} labelled cells...")
    if settings["source"] == "table":
        values = _table_values(settings, positions[keep])
    elif keep.sum() > 256:
        pixel_ids, pixel_values = _all_pixels(settings, progress)
        values = pixel_values[np.searchsorted(pixel_ids, ids[keep])]
    else:
        values = _profiles(settings, ids[keep], progress)
    model = Model(ids[keep], values, names[keep])
    model.provenance = {
        "column": settings["column"],
        "roi": roi,
        "table": Path(table).name,
        "source": settings["source"],
        "examples": len(model.ids),
        "available_examples": available_examples,
        "markers": settings["markers"],
        "backgrounds": settings["backgrounds"],
        "image": Path(settings["image"]).name,
        "flagged_excluded": int(np.isin(ids, flagged).sum()),
        "missing_excluded": model.omitted,
        "needs_examples": unready,
        "method": "Seven nearest examples; weighted votes with smoothing",
        "revision": hashlib.sha256(repr(key).encode()).hexdigest(),
    }
    _MODELS.put(key, model)
    return model


def _all_pixels(settings, progress):
    """One full-resolution tile pass, reused for subsequent scans and clicks."""
    key = _pixel_key(settings)
    cached = _PIXELS.get(key)
    if cached is not None:
        return cached
    # Validate alignment/source freshness using the same rules as clicked cells.
    _, ids = cells.roi_index(settings["table"], settings["roi"], **_keys(settings))
    ids = np.unique(ids[ids > 0])
    _profiles(settings, ids[:1], progress)
    source, masks = open_source(settings["image"]), open_source(settings["pyramid"])
    info = source.info
    channels = [info.channel_names.index(m) for m in settings["markers"]]
    sums = np.zeros((len(ids), len(channels)), dtype=np.float64)
    counts = np.zeros_like(sums)
    total = ((info.width + 511) // 512) * ((info.height + 511) // 512)
    done = 0
    for y in range(0, info.height, 512):
        for x in range(0, info.width, 512):
            progress(f"Measuring image cells: {round(100 * done / total)}%. You can keep using the viewer.")
            bounds = (x, y, min(x + 512, info.width), min(y + 512, info.height))
            mask = masks.read_region(0, 0, *bounds).ravel()
            locations = np.searchsorted(ids, mask)
            keep = (locations < len(ids)) & (ids[np.minimum(locations, len(ids) - 1)] == mask) & (mask > 0)
            if keep.any():
                unique, inverse = np.unique(locations[keep], return_inverse=True)
                for j, (channel, marker) in enumerate(zip(channels, settings["markers"])):
                    pixels = source.read_region(0, channel, *bounds).ravel()[keep].astype(np.float64)
                    finite = np.isfinite(pixels)
                    background = 0 if settings["corrected"] else settings["backgrounds"][marker]
                    sums[unique, j] += np.bincount(
                        inverse[finite], weights=np.maximum(pixels[finite] - background, 0), minlength=len(unique)
                    )
                    counts[unique, j] += np.bincount(inverse[finite], minlength=len(unique))
            done += 1
    values = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    result = (ids, values)
    _PIXELS.put(key, result)
    return result


def run(settings, *, label=None, scan=False, progress=lambda text: None):
    stamp = revision(settings)
    model = _model(settings, stamp, progress)
    if scan:
        positions, ids = cells.roi_index(settings["table"], settings["roi"], **_keys(settings))
        unlabelled = (named(cells.values_at_rows(settings["table"], positions, settings["destination"])) == "") & (
            ids > 0
        )
        if not unlabelled.any():
            values = np.empty((0, len(settings["markers"])))
        elif settings["source"] == "table":
            values = _table_values(settings, positions, cache=True)[unlabelled]
        else:
            pixel_ids, all_values = _all_pixels(settings, progress)
            values = all_values[np.searchsorted(pixel_ids, ids[unlabelled])]
        ids = ids[unlabelled]
        scores = np.full(len(ids), np.nan)
        for start in range(0, len(ids), 8192):
            progress(f"Finding cells to check: {round(100 * start / max(1, len(ids)))}%")
            probabilities = model.predict(values[start : start + 8192], ids[start : start + 8192])
            scores[start : start + 8192] = probabilities.max(axis=1)
        valid = np.isfinite(scores)
        order = np.lexsort((ids[valid], scores[valid]))
        result = {
            "status": "ready",
            "labels": ids[valid][order],
            "scores": scores[valid][order],
            "omitted": int((~valid).sum()),
            "provenance": model.provenance,
        }
    else:
        ids, positions = cells.assignment_rows(settings["table"], settings["roi"], [label], **_keys(settings))
        if not len(ids):
            raise ValueError("This cell has no row in the selected table.")
        values = (
            _table_values(settings, positions) if settings["source"] == "table" else _profiles(settings, ids, progress)
        )
        probabilities = model.predict(values, ids)[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("This cell is missing marker measurements, so no suggestion is shown.")
        order = np.argsort(-probabilities, kind="stable")[:3]
        result = {
            "status": "ready",
            "label": int(label),
            "choices": [{"phenotype": str(model.classes[i]), "probability": float(probabilities[i])} for i in order],
            "probabilities": [{"phenotype": str(name), "probability": float(probabilities[i])}
                              for i, name in enumerate(model.classes)],
            "provenance": model.provenance,
        }
    if revision(settings) != stamp:
        raise ValueError("The labels or measurements changed. Try again to use the latest values.")
    return result


def request(settings, *, label=None, scan=False, exclude=(), retry=False, cancel=False):
    key = (json.dumps(settings, sort_keys=True), revision(settings), None if scan else label, scan)
    with _LOCK:
        job = _JOBS.get(key)
        if cancel:
            for old, pending in _JOBS.items():
                if old[0] == key[0] and old[3] == scan:
                    pending["cancel"].set()
            return {"status": "cancelled", "message": "Stopped finding cells."}
        if job is None or (retry and job["result"]["status"] in {"error", "cancelled"}):
            if sum(j["result"]["status"] == "preparing" for j in _JOBS.values()) >= 4:
                return {"status": "preparing", "message": "Finishing earlier cell measurements..."}
            job = {
                "result": {"status": "preparing", "message": "Preparing suggestions..."},
                "cancel": threading.Event(),
            }
            _JOBS[key] = job

            def work():
                def progress(message):
                    if job["cancel"].is_set():
                        raise InterruptedError("Stopped finding cells.")
                    job["result"] = {"status": "preparing", "message": message}

                try:
                    job["result"] = run(settings, label=label, scan=scan, progress=progress)
                except Exception as exc:
                    job["result"] = {
                        "status": "cancelled" if isinstance(exc, InterruptedError) else "error",
                        "message": str(exc),
                    }

            _POOL.submit(work)
        _JOBS.move_to_end(key)
        for old in list(_JOBS):
            if len(_JOBS) <= 16:
                break
            if _JOBS[old]["result"]["status"] != "preparing":
                del _JOBS[old]
        result = job["result"]
        if scan and result["status"] == "ready":
            keep = ~np.isin(result["labels"], exclude)
            index = np.flatnonzero(keep)
            first = int(index[0]) if len(index) else None
            return {
                "status": "ready",
                "label": int(result["labels"][first]) if first is not None else None,
                "remaining": int(keep.sum()),
                "omitted": result["omitted"],
                "provenance": result["provenance"],
            }
        return result


def settings_for(layer, table, body, markers):
    source = body.get("source", "table")
    if source not in {"table", "image", "corrected"}:
        raise ValueError("Choose saved measurements or image pixels.")
    image = layer.get("background_image") if source == "corrected" else layer["image"]
    if not image:
        raise ValueError("No background-subtracted image was found for this ROI.")
    corrected = source == "corrected" or "_bg_subtracted.ome." in Path(image).name.lower()
    backgrounds = body.get("backgrounds") or {}
    if source != "table":
        markers = [m for m in image_info(image).channel_names if not m.upper().startswith("HE_")]
        if not corrected and any(m not in backgrounds or not np.isfinite(float(backgrounds[m])) for m in markers):
            raise ValueError(
                "Set a background level for every marker with Auto all markers, or choose saved measurements."
            )
    if not markers:
        raise ValueError("No matching marker measurements were found.")
    if len(set(markers)) != len(markers):
        raise ValueError("The marker names must be unique before using suggestions.")
    column, destination = str(body.get("column") or "").strip(), str(body.get("destination") or "").strip()
    if not column or not destination:
        raise ValueError("Choose where to save labels and which label column to learn from.")
    return {
        "table": table,
        "roi": layer["roi"],
        "column": column,
        "destination": destination,
        "source": source,
        "image": image,
        "pyramid": layer["pyramid"],
        "segmentation": layer.get("segmentation") or cells._recorded_source(Path(layer["pyramid"])).get("source"),
        "corrected": corrected,
        "backgrounds": {m: float(backgrounds[m]) for m in markers} if source == "image" and not corrected else {},
        "markers": markers,
        "roi_column": layer.get("roi_column", "batch_id"),
        "label_column": layer.get("label_column", "label"),
    }
