"""Persistent, bounded Random Forest training from saved consensus measurements.

Training runs in a child process. Viewer requests load the private cache once
and predict without reading images or relearning from destination-column edits.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pandas as pd

from spacemap import dataio, labels, pending_columns, userpaths
from . import cells, reference
from .sources import fingerprint
from spacemap.sharing import open_lock

COLUMN = "phenotype_ml_consensus"
FORMAT = 1
TREES = 96
PER_CLASS = 2000
MAX_EXAMPLES = 50000
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="consensus-model")
_LOCK = threading.RLock()
_JOBS = OrderedDict()
_STAMPS = cells._Cache(8)
_ROWS = cells._Cache(512)
_SCANS = cells._Cache(2)
_EMPTY = {"", "nan", "none", "unknown", "unassigned", "unclassified", "unlabeled", "unlabelled"}


@lru_cache(maxsize=1)
def _version():
    from importlib.metadata import version

    return version("scikit-learn")


def _pending(table, column):
    return next((p for p in pending_columns.pending(table) if p["column"] == column and not p["stale"]), None)


def signature(settings):
    table, markers = settings["table"], settings["markers"]
    watched = [COLUMN, *markers]
    db = reference.path_for(table)
    key = (
        table,
        fingerprint(table),
        tuple(markers),
        cells._edits_token(table),
        fingerprint(db) if db.exists() else "",
        settings["roi_column"],
        settings["label_column"],
    )
    cached = _STAMPS.get(key)
    if cached is not None:
        return cached
    edits = labels.load(table)
    if not edits.empty:
        edits = edits[edits["column"].isin(watched)]
    with reference.database(table) as connection:
        flags = (
            [
                list(r)
                for r in connection.execute(
                    "SELECT roi,label FROM reviews WHERE column_name=? AND flagged=1 ORDER BY roi,label", (COLUMN,)
                )
            ]
            if connection
            else []
        )
    payload = {
        "format": FORMAT,
        "sklearn": _version(),
        "table": str(Path(table).resolve()),
        "fingerprint": fingerprint(table),
        "markers": markers,
        "roi_column": settings["roi_column"],
        "label_column": settings["label_column"],
        "edits": int(pd.util.hash_pandas_object(edits, index=False).sum()) if not edits.empty else 0,
        "pending": [
            (p["column"], fingerprint(p["file"]))
            for p in pending_columns.pending(table)
            if p["column"] in watched and not p["stale"]
        ],
        "flags": flags,
    }
    result = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _STAMPS.put(key, result)
    return result


def settings_for(layer, table, markers, destination):
    if COLUMN not in dataio.columns(table):
        raise ValueError("Choose a cell table containing phenotype_ml_consensus to use the saved model.")
    markers = sorted(set(markers))
    if not markers:
        raise ValueError("The selected table has no matching marker measurements.")
    return {
        "table": str(Path(table).resolve()),
        "roi": layer["roi"],
        "markers": markers,
        "destination": str(destination or "phenotype_ref1"),
        "roi_column": layer.get("roi_column", "batch_id"),
        "label_column": layer.get("label_column", "label"),
    }


def path_for(settings):
    return userpaths.models_dir() / "viewer_consensus" / (signature(settings) + ".joblib")


def _batches(table, columns):
    """At most one batch of the source table in memory; never build a fast copy."""
    import pyarrow.parquet as pq

    fast = dataio.fast_copy_info(table)
    offset = 0
    if fast.get("current"):
        for batch in pq.ParquetFile(fast["twin"]).iter_batches(batch_size=65536, columns=columns):
            yield offset, batch
            offset += len(batch)
    else:
        for batch in pd.read_csv(table, usecols=columns, chunksize=65536):
            yield offset, batch
            offset += len(batch)


def _edited_column(table, column):
    frame = labels.load(table)
    return frame[frame["column"] == column].sort_values("row") if not frame.empty else frame


def _overlay(values, offset, edits):
    if not edits.empty:
        wanted = edits[(edits["row"] >= offset) & (edits["row"] < offset + len(values))]
        values[wanted["row"].to_numpy(dtype=int) - offset] = wanted["value"].to_numpy()
    return values


def _numpy(column):
    return column.to_numpy() if isinstance(column, pd.Series) else column.to_numpy(zero_copy_only=False)


def _label_batches(settings):
    import pyarrow.parquet as pq

    table = settings["table"]
    column = settings.get("training_column", COLUMN)
    pending = _pending(table, column)
    edits = _edited_column(table, column)
    if pending:
        offset = 0
        for batch in pq.ParquetFile(pending["file"]).iter_batches(batch_size=65536, columns=["value"]):
            values = batch.column(0).to_numpy(zero_copy_only=False).astype(object)
            yield offset, _overlay(values, offset, edits)
            offset += len(values)
    else:
        native = column in dataio.columns(table, include_pending=False)
        for offset, batch in _batches(table, [column if native else settings["label_column"]]):
            if native:
                values = _numpy(batch[column]).astype(object)
            else:
                values = np.full(len(batch), None, dtype=object)
            yield offset, _overlay(values, offset, edits)


def _sample(settings, progress, *, limit=MAX_EXAMPLES):
    """Uniform random-priority reservoir per phenotype, including every ROI."""
    table = settings["table"]
    with reference.database(table) as db:
        flags = (
            list(
                db.execute(
                    "SELECT roi,label FROM reviews WHERE column_name=? AND flagged=1",
                    (settings.get("training_column", COLUMN),),
                )
            )
            if db
            else []
        )
    flagged = []
    for roi in sorted({r[0] for r in flags}):
        _, positions = cells.assignment_rows(
            table,
            roi,
            [r[1] for r in flags if r[0] == roi],
            roi_column=settings["roi_column"],
            label_column=settings["label_column"],
        )
        flagged.extend(positions.tolist())
    reservoirs, counts = {}, {}
    rng = np.random.default_rng(0)
    for offset, values in _label_batches(settings):
        text = pd.Series(values).fillna("").astype(str).str.strip()
        keep = ~text.str.lower().isin(_EMPTY).to_numpy()
        if settings.get("checked_rows") is not None:
            keep &= np.isin(np.arange(offset, offset + len(values)), settings["checked_rows"])
        if flagged:
            keep &= ~np.isin(np.arange(offset, offset + len(values)), flagged)
        rows = np.flatnonzero(keep)
        names = text.to_numpy()[rows]
        classes, codes, group_counts = np.unique(names, return_inverse=True, return_counts=True)
        priorities = rng.random(len(rows))
        for i, (name, count) in enumerate(zip(classes, group_counts)):
            selected = codes == i
            positions, ranks = rows[selected] + offset, priorities[selected]
            counts[str(name)] = counts.get(str(name), 0) + int(count)
            old_positions, old_ranks = reservoirs.get(str(name), (np.empty(0, np.int64), np.empty(0)))
            positions, ranks = np.concatenate([old_positions, positions]), np.concatenate([old_ranks, ranks])
            if len(positions) > PER_CLASS:
                chosen = np.argpartition(ranks, -PER_CLASS)[-PER_CLASS:]
                positions, ranks = positions[chosen], ranks[chosen]
            reservoirs[str(name)] = positions, ranks
        if len(reservoirs) > 1000:
            raise ValueError("The consensus column has too many different names for a phenotype model.")
        progress(f"Reading consensus labels: {offset + len(values):,} cells checked. The viewer stays available.")
    usable = [name for name in sorted(reservoirs) if counts[name] >= 2]
    if len(usable) < 2:
        raise ValueError("The consensus column needs at least two phenotypes with two examples each.")
    cap = min(PER_CLASS, max(2, limit // len(usable)))
    positions, targets = [], []
    for name in usable:
        rows, ranks = reservoirs[name]
        chosen = np.argsort(-ranks, kind="stable")[:cap]
        positions.extend(rows[chosen])
        targets.extend([name] * len(chosen))
    positions, targets = np.asarray(positions, np.int64), np.asarray(targets)
    order = np.argsort(positions)
    return positions[order], targets[order], counts, len(flagged)


def selected_values(settings, positions, *, streaming=False, dtype=np.float32):
    table, markers = settings["table"], settings["markers"]
    positions = np.asarray(positions, np.int64)
    output = np.full((len(positions), len(markers)), np.nan, dtype=dtype)
    if not len(positions):
        return output
    available = set(dataio.columns(table, include_pending=False))
    native = [c for c in markers if c in available]
    raw = None if streaming else cells.fast_rows(table, positions, native)
    if raw is not None:
        for j, marker in enumerate(markers):
            if marker in native:
                output[:, j] = pd.to_numeric(pd.Series(_numpy(raw[marker])), errors="coerce").to_numpy(dtype)
    else:
        import pyarrow as pa

        order = np.argsort(positions)
        sorted_positions = positions[order]
        for offset, batch in _batches(table, native or [settings["label_column"]]):
            left, right = np.searchsorted(sorted_positions, [offset, offset + len(batch)])
            if left == right:
                continue
            local = sorted_positions[left:right] - offset
            chosen = batch.iloc[local] if isinstance(batch, pd.DataFrame) else batch.take(pa.array(local))
            for j, marker in enumerate(markers):
                if marker in native:
                    output[order[left:right], j] = pd.to_numeric(
                        pd.Series(_numpy(chosen[marker])), errors="coerce"
                    ).to_numpy(dtype)
    edits = labels.load(table)
    edited = set(edits["column"]) if not edits.empty else set()
    edited.update(p["column"] for p in pending_columns.pending(table) if not p["stale"])
    for j, marker in enumerate(markers):
        if marker in edited:
            output[:, j] = pd.to_numeric(
                pd.Series(cells.values_at_rows(table, positions, marker)), errors="coerce"
            ).to_numpy(dtype)
    return output


def train(settings, progress=lambda text: None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    stamp = signature(settings)
    rows, targets, counts, flagged = _sample(settings, progress)
    progress(f"Reading saved marker values for {len(rows):,} example cells...")
    values = selected_values(settings, rows, streaming=True)
    finite = np.all(np.isfinite(values), axis=1)
    missing = int((~finite).sum())
    rows, targets, values = rows[finite], targets[finite], values[finite]
    classes, numbers = np.unique(targets, return_counts=True)
    keep = np.isin(targets, classes[numbers >= 2])
    rows, targets, values = rows[keep], targets[keep], values[keep]
    if len(np.unique(targets)) < 2:
        raise ValueError("Too few consensus-labelled cells have complete marker measurements.")
    progress(f"Training the saved model once, using {len(rows):,} example cells...")
    model = RandomForestClassifier(
        n_estimators=TREES,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=2,
        bootstrap=True,
        oob_score=True,
    )
    model.fit(values, targets)
    oob = model.oob_decision_function_.copy()
    valid_oob = np.isfinite(oob).all(axis=1) & (oob.sum(axis=1) > 0)
    score = (
        float(balanced_accuracy_score(targets[valid_oob], model.classes_[oob[valid_oob].argmax(axis=1)]))
        if valid_oob.any()
        else None
    )
    model.n_jobs = 1  # avoids starting a thread team for every clicked cell
    if stamp != signature(settings):
        raise ValueError(
            "Consensus labels or marker measurements changed during training. Try preparing the model again."
        )
    provenance = {
        "method": "Random Forest",
        "column": COLUMN,
        "source": "table",
        "roi": "all ROIs in this table",
        "table": Path(settings["table"]).name,
        "markers": settings["markers"],
        "examples": len(rows),
        "available_examples": sum(counts.values()),
        "flagged_excluded": flagged,
        "missing_excluded": missing,
        "needs_examples": sorted(set(counts) - set(model.classes_)),
        "revision": stamp,
        "oob_balanced_accuracy": score,
        "trees": TREES,
        "training_counts": dict(zip(*np.unique(targets, return_counts=True))),
    }
    provenance["training_counts"] = {str(k): int(v) for k, v in provenance["training_counts"].items()}
    return {"format": FORMAT, "revision": stamp, "model": model, "rows": rows, "oob": oob, "provenance": provenance}


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def _load_or_train(settings, target, job):
    import fcntl
    import joblib

    target.parent.mkdir(parents=True, exist_ok=True)
    with open_lock(target.with_suffix(".lock")) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            try:
                artifact = joblib.load(target)
                if artifact["format"] == FORMAT and artifact["revision"] == signature(settings):
                    job["artifact"] = artifact
                    job["status"] = "ready"
                    return
            except Exception:
                pass  # incomplete/incompatible private cache: rebuild atomically
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=target.parent) as folder:
            spec, status, log = [Path(folder) / name for name in ("settings.json", "status.json", "worker.log")]
            spec.write_text(json.dumps(settings))
            env = os.environ.copy()
            root = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "2"
            with log.open("w") as output:
                process = subprocess.Popen(
                    [sys.executable, "-m", __name__, str(spec), str(target), str(status)],
                    stdout=output,
                    stderr=output,
                    env=env,
                )
                job["progress_file"] = status
                returncode = process.wait()
            if returncode:
                raise ValueError(
                    json.loads(status.read_text()).get("message", "Model preparation failed.")
                    if status.exists()
                    else "Model preparation failed. Check that the cell table is readable and enough memory is available."
                )
            artifact = joblib.load(target)
            if artifact["revision"] != signature(settings):
                raise ValueError(
                    "The source changed while the model was preparing. Retry to use the latest measurements."
                )
            job["artifact"] = artifact
            job["status"] = "ready"


def ensure(settings, *, retry=False):
    target = path_for(settings)
    key = str(target)
    with _LOCK:
        job = _JOBS.get(key)
        if job is None or retry and job["status"] == "error":
            if sum(j["status"] == "preparing" for j in _JOBS.values()) >= 3:
                return {"status": "preparing", "message": "Waiting for another cell table's model to finish..."}
            job = {"status": "preparing", "message": "Loading the saved consensus model..."}
            _JOBS[key] = job

            def work():
                try:
                    _load_or_train(settings, target, job)
                except Exception as exc:
                    job.update(status="error", message=str(exc))

            _POOL.submit(work)
        _JOBS.move_to_end(key)
        for old in list(_JOBS):
            if len(_JOBS) <= 3:
                break
            if _JOBS[old]["status"] != "preparing":
                del _JOBS[old]
        return job


def status(job):
    if job["status"] == "ready":
        return {"status": "ready", "provenance": job["artifact"]["provenance"]}
    progress = job.get("progress_file")
    if job["status"] == "preparing" and progress and progress.exists():
        try:
            return {"status": "preparing", "message": json.loads(progress.read_text())["message"]}
        except (OSError, ValueError, KeyError):
            pass
    return {"status": job["status"], "message": job["message"]}


def predict(artifact, values, rows):
    probabilities = np.full((len(values), len(artifact["model"].classes_)), np.nan)
    valid = np.isfinite(values).all(axis=1)
    if valid.any():
        probabilities[valid] = artifact["model"].predict_proba(values[valid])
    trained = artifact["rows"]
    places = np.searchsorted(trained, rows)
    own = (places < len(trained)) & (trained[np.minimum(places, len(trained) - 1)] == rows) & valid
    # For sampled training cells use only trees that did not train on that row.
    probabilities[own] = artifact["oob"][places[own]]
    probabilities[probabilities.sum(axis=1) <= 0] = np.nan
    return probabilities


def request(settings, *, label=None, prepare=False, scan=False, exclude=(), retry=False, cancel=False):
    job = ensure(settings, retry=retry)
    if job["status"] != "ready" or prepare:
        return status(job)
    artifact = job["artifact"]
    stamp = signature(settings)
    if scan:
        with _LOCK:
            return _scan(settings, artifact, exclude, cancel, retry)
    ids, rows = cells.assignment_rows(
        settings["table"],
        settings["roi"],
        [label],
        roi_column=settings["roi_column"],
        label_column=settings["label_column"],
    )
    if len(rows) != 1:
        raise ValueError("This cell has no unique row in the selected table.")
    key = (stamp, int(rows[0]))
    probabilities = _ROWS.get(key)
    if probabilities is None:
        probabilities = predict(artifact, selected_values(settings, rows), rows)[0]
        _ROWS.put(key, probabilities)
    if not np.isfinite(probabilities).all():
        raise ValueError(
            "This cell is missing marker measurements or independent training votes, so no suggestion is shown."
        )
    if signature(settings) != stamp:
        raise ValueError("The source measurements changed. Click the cell again to use the updated model.")
    order = np.argsort(-probabilities, kind="stable")[:3]
    return {
        "status": "ready",
        "label": int(label),
        "choices": [
            {"phenotype": str(artifact["model"].classes_[i]), "probability": float(probabilities[i])} for i in order
        ],
        "provenance": artifact["provenance"],
        "probabilities": [{"phenotype": str(name), "probability": float(probabilities[i])}
                          for i, name in enumerate(artifact["model"].classes_)],
    }


def _scan(settings, artifact, exclude, cancel, retry, revision_check=None, predictor=None):
    from .suggestions import named

    table, roi = settings["table"], settings["roi"]
    key = (str(Path(table).resolve()), fingerprint(table), settings["roi_column"], settings["label_column"],
           artifact["revision"], roi, settings["destination"], cells._edits_token(table))
    job = _SCANS.get(key)
    if cancel:
        if job:
            job["cancel"].set()
        return {"status": "cancelled", "message": "Stopped finding cells."}
    if job is None or retry and job["status"] in {"error", "cancelled"}:
        job = {"status": "preparing", "message": "Finding unlabelled cells to check...", "cancel": threading.Event()}
        _SCANS.put(key, job)

        def work():
            try:
                rows, ids = cells.roi_index(
                    table, roi, roi_column=settings["roi_column"], label_column=settings["label_column"]
                )
                keep = (named(cells.values_at_rows(table, rows, settings["destination"])) == "") & (ids > 0)
                rows, ids = rows[keep], ids[keep]
                # Read marker columns once for the ROI, sharing the existing feature cache.
                from .suggestions import _table_values

                legacy = {**settings, "source": "table"}
                values = (
                    _table_values(legacy, rows, cache=True) if len(rows) else np.empty((0, len(settings["markers"])))
                )
                scores = np.full(len(rows), np.nan)
                for start in range(0, len(rows), 8192):
                    if job["cancel"].is_set():
                        raise InterruptedError("Stopped finding cells.")
                    job["message"] = f"Finding cells to check: {round(100 * start / max(1, len(rows)))}%"
                    scores[start : start + 8192] = (predictor or predict)(
                        artifact, values[start : start + 8192], rows[start : start + 8192]
                    ).max(axis=1)
                unchanged = revision_check() if revision_check else signature(settings) == artifact["revision"]
                if key[-1] != cells._edits_token(table) or not unchanged:
                    raise ValueError("Labels or measurements changed. Start the search again.")
                valid = np.isfinite(scores)
                order = np.lexsort((ids[valid], scores[valid]))
                job.update(ids=ids[valid][order], omitted=int((~valid).sum()), status="ready")
            except Exception as exc:
                job.update(status="cancelled" if isinstance(exc, InterruptedError) else "error", message=str(exc))

        _POOL.submit(work)
    if job["status"] != "ready":
        return {"status": job["status"], "message": job["message"]}
    remaining = job["ids"][~np.isin(job["ids"], exclude)]
    return {
        "status": "ready",
        "label": int(remaining[0]) if len(remaining) else None,
        "remaining": len(remaining),
        "omitted": job["omitted"],
        "provenance": artifact["provenance"],
    }


if __name__ == "__main__":
    import joblib

    settings_path, target, status_path = map(Path, sys.argv[1:])
    try:
        artifact = train(
            json.loads(settings_path.read_text()), lambda message: _write_json(status_path, {"message": message})
        )
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        joblib.dump(artifact, temporary, compress=3)
        temporary.replace(target)
    except Exception as exc:
        _write_json(status_path, {"message": str(exc)})
        raise
