"""Marker-bar views, independent of raw thresholds and phenotype predictions.

Z1 uses the whole ROI's saved measurements. The other operations act on the
clicked cell. Normal CDF and NLog match RESOLVE, including its tail floor.
"""

from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pandas as pd
from scipy.special import ndtr

from spacemap import dataio, labels, pending_columns
from spacemap.ph_thres_utils import NORMALIZATION_EPSILON
from . import cells, consensus_model
from .sources import fingerprint

ORDER = ("Arcsinh", "Z1", "Z2", "CDF", "NLog")
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="marker-view")
_LOCK = threading.RLock()
_STATS = cells._Cache(8)
_TOKENS = cells._Cache(16)


def steps(requested):
    if not isinstance(requested, list) or any(name not in ("Raw", *ORDER) for name in requested):
        raise ValueError("Choose Raw, Arcsinh, Z1, Z2, CDF or NLog.")
    if len(set(requested)) != len(requested) or ("Raw" in requested and len(requested) != 1):
        raise ValueError("Raw is used alone. Select each other step once.")
    return [name for name in ORDER if name in requested]


def revision(settings):
    table, markers = settings["table"], settings["markers"]
    key = (table, fingerprint(table), cells._edits_token(table), tuple(markers))
    cached = _TOKENS.get(key)
    if cached is not None:
        return cached
    edits = labels.load(table)
    if not edits.empty:
        edits = edits[edits["column"].isin(markers)]
    value = (
        table,
        fingerprint(table),
        tuple(markers),
        tuple(
            (p["column"], fingerprint(p["file"]))
            for p in pending_columns.pending(table)
            if p["column"] in markers and not p["stale"]
        ),
        int(pd.util.hash_pandas_object(edits, index=False).sum()) if not edits.empty else 0,
    )
    _TOKENS.put(key, value)
    return value


def statistics(settings):
    """Both raw and Arcsinh statistics in bounded batches, once per marker revision."""
    rows, _ = cells.roi_index(
        settings["table"], settings["roi"], roi_column=settings["roi_column"], label_column=settings["label_column"]
    )
    if not len(rows):
        raise ValueError("No cells from this ROI were found in the selected table.")
    count = np.zeros(len(settings["markers"]), dtype=np.int64)
    means = np.zeros((2, len(count)))
    squares = means.copy()
    if dataio.fast_copy_info(settings["table"]).get("current"):
        batches = (
            consensus_model.selected_values(settings, rows[i : i + 65536], dtype=np.float64)
            for i in range(0, len(rows), 65536)
        )
    else:
        # One CSV pass, rather than rereading the CSV for every batch.
        values = consensus_model.selected_values(settings, rows, streaming=True, dtype=np.float64)
        batches = (values[i : i + 65536] for i in range(0, len(values), 65536))
    for raw in batches:
        finite = np.isfinite(raw)
        counts = finite.sum(axis=0)
        combined = count + counts
        for index, values in enumerate((raw, np.arcsinh(raw / 5.0))):
            batch_mean = np.where(finite, values, 0).sum(axis=0) / np.maximum(counts, 1)
            centered = np.where(finite, values - batch_mean, 0)
            delta = batch_mean - means[index]
            squares[index] += (centered * centered).sum(axis=0) + delta * delta * count * counts / np.maximum(
                combined, 1
            )
            means[index] += delta * counts / np.maximum(combined, 1)
        count = combined
    scales = np.sqrt(squares / np.maximum(count, 1))
    scales[scales == 0] = NORMALIZATION_EPSILON
    return means, scales, count


def transform(values, selected, stats=None):
    values = np.asarray(values, dtype=np.float64).copy()
    for step in selected:
        if step == "Arcsinh":
            values = np.arcsinh(values / 5.0)
        elif step == "Z1":
            means, scales, counts = stats
            index = int("Arcsinh" in selected)
            values = (values - means[index]) / scales[index]
            values[counts == 0] = np.nan
        elif step == "Z2":
            finite = np.isfinite(values)
            if finite.any():
                spread = values[finite].std() or NORMALIZATION_EPSILON
                values = (values - values[finite].mean()) / spread
        elif step == "CDF":
            values = ndtr(values)
        elif step == "NLog":
            values = -np.log(np.maximum(1 - values, NORMALIZATION_EPSILON))
    return values


def apply_profile(profile, requested, settings=None):
    selected = steps(requested)
    if profile["status"] != "ready" or not selected:
        return profile
    stats = None
    if "Z1" in selected:
        if profile["source"] != "table" or not settings:
            raise ValueError("Z1 compares cells in this ROI. Choose Saved measurements to use it.")
        stamp = revision(settings)
        key = (stamp, settings["roi"], settings["roi_column"], settings["label_column"])
        with _LOCK:
            job = _STATS.get(key)
            if job is None:
                job = {"status": "preparing", "message": "Preparing marker comparisons for this ROI once..."}
                _STATS.put(key, job)

                def work():
                    try:
                        values = statistics(settings)
                        if revision(settings) != stamp:
                            raise ValueError("Marker measurements changed. Click the cell again.")
                        job.update(stats=values, status="ready")
                    except Exception as exc:
                        job.update(status="error", message=str(exc))

                _POOL.submit(work)
        if job["status"] != "ready":
            return {"status": job["status"], "message": job["message"], "source": profile["source"]}
        lookup = {name: i for i, name in enumerate(settings["markers"])}
        indices = [lookup[row["marker"]] for row in profile["markers"]]
        means, scales, counts = job["stats"]
        stats = (means[:, indices], scales[:, indices], counts[indices])
    raw = [row.get("signal") if row.get("signal") is not None else row["mean"] for row in profile["markers"]]
    values = transform(raw, selected, stats)
    result = {
        **profile,
        "steps": selected,
        "markers": [
            {**row, "view": float(value) if np.isfinite(value) else None}
            for row, value in zip(profile["markers"], values)
        ],
    }
    result["view_note"] = (
        ("Z1 compares with saved cells in this ROI. " if "Z1" in selected else "")
        + ("Z2 compares markers within this cell. " if "Z2" in selected else "")
        + ("CDF and NLog follow RESOLVE. " if "CDF" in selected or "NLog" in selected else "")
        + "Display only; saved measurements and thresholds stay unchanged."
    )
    if "NLog" in selected and "CDF" not in selected:
        result["view_note"] += " NLog is usually used after CDF; values at or above one share the same upper limit."
    return result
