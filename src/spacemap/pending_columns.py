"""Whole columns saved in the interface, waiting to be written into the CSV.

Saving a phenotype column used to rewrite the whole unified table -- minutes
for a 13 GB CSV, on every save. A save now stores just the columns it changed,
one small Parquet file each, beside the table. Every read through
:mod:`spacemap.dataio` merges them in, so the interface shows the new values at
once. Writing them into the CSV is a separate step, :func:`commit`, which runs
in its own process (:func:`start_commit`) so it finishes even if the interface
is closed or restarted, and which writes pending Manual Reference labels in the
same pass.

Layout, beside ``table.csv``::

    table.csv.pending/
        _anchor.parquet          cell IDs in table order (the ID column named below)
        <column>.parquet         one full-length column each
        commit_status.json       progress of a running or finished write

**Safety.** Each file records the table's size and modification time. While
they match, columns are applied by row position. If the CSV is rewritten by
anything else, the columns are *not* applied -- :func:`pending` reports them as
``stale`` -- until the user chooses :func:`reapply_by_id`, which maps them onto
the new table through the stored cell IDs, or :func:`discard`. Nothing is ever
applied to the wrong cells, and nothing is dropped without the user choosing to.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STORE_SUFFIX = ".pending"
ANCHOR_FILE = "_anchor.parquet"
STATUS_FILE = "commit_status.json"
_META_KEY = b"spacemap_pending"


class PendingColumnsError(RuntimeError):
    """Raised when pending columns cannot be stored or applied safely."""


# ---------------------------------------------------------------------------
# Paths and fingerprints
# ---------------------------------------------------------------------------

def store_dir(table_path: str | Path) -> Path:
    source = Path(table_path).expanduser()
    beside = source.with_name(source.name + STORE_SUFFIX)
    if os.access(source.parent, os.W_OK):
        return beside
    from hashlib import sha1

    digest = sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
    root = Path(os.environ.get("SPACEMAP_CACHE_DIR", Path.home() / ".cache" / "spacemap")).expanduser()
    return root / "pending" / f"{source.name}.{digest}{STORE_SUFFIX}"


def _fingerprint(table_path: Path) -> str:
    stat = Path(table_path).stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _safe_name(column: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", column).strip("._") or "column"


def _text_or_missing(value: Any):
    """Text for a stored label; None for any kind of missing value (NaN, None, pd.NA)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def _column_file(table_path: Path, column: str) -> Path:
    return store_dir(table_path) / f"{_safe_name(column)}.parquet"


def _write_parquet(frame: pd.DataFrame, target: Path, meta: Mapping[str, Any]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({_META_KEY: json.dumps(dict(meta), default=str).encode("utf-8")})
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="snappy")
    os.replace(temporary, target)


def _read_meta(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    raw = (pq.read_schema(path).metadata or {}).get(_META_KEY)
    return json.loads(raw.decode("utf-8")) if raw else {}


# ---------------------------------------------------------------------------
# What is pending
# ---------------------------------------------------------------------------

def pending(table_path: str | Path) -> list[dict[str, Any]]:
    """Every pending column: name, rows, saved_at, source, and whether it is stale."""
    source = Path(table_path).expanduser()
    directory = store_dir(source)
    if not directory.is_dir() or not source.is_file():
        return []
    current = _fingerprint(source)
    items = []
    for path in sorted(directory.glob("*.parquet")):
        if path.name == ANCHOR_FILE:
            continue
        try:
            meta = _read_meta(path)
        except Exception as exc:
            logger.warning("Unreadable pending column %s: %s", path.name, exc)
            continue
        if not meta.get("column"):
            continue
        items.append({**meta, "file": str(path), "stale": meta.get("fingerprint") != current})
    return items


def pending_names(table_path: str | Path, *, include_stale: bool = False) -> list[str]:
    return [item["column"] for item in pending(table_path) if include_stale or not item["stale"]]


def has_stale(table_path: str | Path) -> bool:
    return any(item["stale"] for item in pending(table_path))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save(
    table_path: str | Path,
    columns: Mapping[str, pd.Series | np.ndarray | Sequence],
    *,
    anchor_column: str,
    anchor_values: pd.Series | np.ndarray | Sequence,
    source_note: str = "",
) -> list[str]:
    """Store full-length columns for *table_path*. Returns the names stored.

    *anchor_values* are the table's cell IDs in table order; they are what lets a
    pending column be re-applied if the table is rewritten before it is written.
    """
    from . import labels
    with labels.write_lock(table_path):
        source = Path(table_path).expanduser()
        if not source.is_file():
            raise PendingColumnsError(f"{source} does not exist.")
        if has_stale(source):
            raise PendingColumnsError(
                "Earlier saved changes for this table no longer match it (the CSV was rewritten). "
                "Re-apply them by cell ID or discard them before saving new ones."
            )
        return _store(source, columns, anchor_column=anchor_column, anchor_values=anchor_values,
                      source_note=source_note)


def _store(
    source: Path,
    columns: Mapping[str, pd.Series | np.ndarray | Sequence],
    *,
    anchor_column: str,
    anchor_values: pd.Series | np.ndarray | Sequence,
    source_note: str = "",
) -> list[str]:
    """Write columns against the table as it is now (no staleness check)."""
    anchor = pd.Series(np.asarray(anchor_values, dtype=object) if not isinstance(anchor_values, pd.Series)
                       else anchor_values.to_numpy(), name="anchor")
    rows = len(anchor)
    if anchor.duplicated().any():
        raise PendingColumnsError(f"The cell ID column '{anchor_column}' has duplicate values.")
    fingerprint = _fingerprint(source)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    anchor_path = store_dir(source) / ANCHOR_FILE
    anchor_meta = {"fingerprint": fingerprint, "rows": rows, "anchor_column": anchor_column}
    rewrite_anchor = True
    if anchor_path.is_file():
        try:
            rewrite_anchor = _read_meta(anchor_path) != anchor_meta
        except Exception:
            rewrite_anchor = True
    if rewrite_anchor:
        _write_parquet(pd.DataFrame({"anchor": anchor.astype(str)}), anchor_path, anchor_meta)

    stored = []
    for name, values in columns.items():
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        if len(series) != rows:
            raise PendingColumnsError(f"Column '{name}' has {len(series)} values; the table has {rows} rows.")
        frame = pd.DataFrame({"value": series.reset_index(drop=True)})
        if frame["value"].dtype == object:
            # Parquet needs one type per column; keep missing values missing.
            frame["value"] = frame["value"].map(_text_or_missing)
        _write_parquet(
            frame,
            _column_file(source, name),
            {"column": name, "fingerprint": fingerprint, "rows": rows, "saved_at": stamp,
             "source": source_note, "anchor_column": anchor_column},
        )
        stored.append(name)
    logger.info("Saved %d pending column(s) for %s: %s", len(stored), source.name, ", ".join(stored))
    return stored


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def apply_to(frame: pd.DataFrame, table_path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Overlay current (non-stale) pending columns onto a frame in table order.

    With *columns*, only those are added or replaced, so a narrow read stays
    narrow; without, every pending column is applied.
    """
    wanted = set(columns) if columns else None
    for item in pending(table_path):
        name = item["column"]
        if item["stale"] or (wanted is not None and name not in wanted):
            continue
        if int(item.get("rows", -1)) != len(frame):
            logger.warning("Pending column %s does not match the table length; not applied.", name)
            continue
        values = pd.read_parquet(item["file"])["value"]
        frame[name] = values.to_numpy()
    return frame


def reapply_by_id(table_path: str | Path, *, read_anchor: Callable[[Path, str], pd.Series] | None = None) -> int:
    """Map stale pending columns onto the rewritten table through the stored cell IDs.

    Cells missing from the new table are dropped; cells new to it keep their
    current value. Returns how many columns were re-applied.
    """
    source = Path(table_path).expanduser()
    directory = store_dir(source)
    anchor_path = directory / ANCHOR_FILE
    stale = [item for item in pending(source) if item["stale"]]
    if not stale:
        return 0
    if not anchor_path.is_file():
        raise PendingColumnsError("The stored cell IDs are missing, so the changes cannot be re-applied.")
    anchor_meta = _read_meta(anchor_path)
    anchor_column = anchor_meta["anchor_column"]
    old_ids = pd.read_parquet(anchor_path)["anchor"].astype(str).to_numpy()

    from . import dataio

    if read_anchor is None:
        new_frame = dataio.load(source, columns=[anchor_column], apply_labels=False)
        new_ids = new_frame[anchor_column].astype(str)
    else:
        new_ids = read_anchor(source, anchor_column).astype(str)
    if new_ids.duplicated().any():
        raise PendingColumnsError(f"The rewritten table has duplicate '{anchor_column}' values.")
    positions = pd.Series(np.arange(len(old_ids)), index=old_ids)
    take = positions.reindex(new_ids.to_numpy()).to_numpy()
    found = ~np.isnan(take)

    for item in stale:
        name = item["column"]
        old_values = pd.read_parquet(item["file"])["value"].to_numpy(dtype=object)
        if name in dataio.columns(source, include_pending=False):
            merged = dataio.load(source, columns=[name], apply_labels=False)[name].to_numpy(dtype=object)
        else:
            merged = np.full(len(new_ids), None, dtype=object)
        merged = merged.copy()
        merged[found] = old_values[take[found].astype(int)]
        _store(
            source,
            {name: pd.Series(merged)},
            anchor_column=anchor_column,
            anchor_values=new_ids,
            source_note=f"{item.get('source', '')} (re-applied by {anchor_column})".strip(),
        )
    return len(stale)


def discard(table_path: str | Path, columns: Sequence[str] | None = None) -> int:
    """Delete pending columns (all, or the named ones). Returns how many."""
    removed = 0
    for item in pending(table_path):
        if columns is None or item["column"] in columns:
            Path(item["file"]).unlink(missing_ok=True)
            removed += 1
    directory = store_dir(table_path)
    if directory.is_dir() and not any(p for p in directory.glob("*.parquet") if p.name != ANCHOR_FILE):
        (directory / ANCHOR_FILE).unlink(missing_ok=True)
    return removed


# ---------------------------------------------------------------------------
# Writing into the CSV
# ---------------------------------------------------------------------------

def commit(table_path: str | Path, *, progress: Callable[[str], None] | None = None,
           rebuild_fast_copy: bool = True) -> dict[str, Any]:
    """Write saved changes without racing a viewer label save."""
    from . import labels

    with labels.write_lock(table_path):
        return _commit(table_path, progress=progress, rebuild_fast_copy=rebuild_fast_copy)


def _commit(table_path: str | Path, *, progress: Callable[[str], None] | None = None,
            rebuild_fast_copy: bool = True) -> dict[str, Any]:
    """Write pending columns and pending labels into the CSV in one pass.

    The CSV is replaced atomically, so an interrupted write leaves the old CSV
    and every pending change in place, ready to try again.
    """
    from . import dataio, labels

    source = Path(table_path).expanduser()

    def say(message: str) -> None:
        logger.info(message)
        if progress is not None:
            try:
                progress(message)
            except Exception:
                pass

    items = [item for item in pending(source) if not item["stale"]]
    if any(item["stale"] for item in pending(source)):
        raise PendingColumnsError("Some saved changes no longer match the CSV. Re-apply or discard them first.")
    label_columns = labels.pending_columns(source)
    report: dict[str, Any] = {"columns": [item["column"] for item in items], "labels": labels.pending_count(source),
                              "written": False}
    if not items and not report["labels"]:
        return report

    say(f"Reading {source.name}...")
    frame = dataio.load(source, apply_labels=False)
    if items:
        say(f"Adding {len(items)} saved column(s): {', '.join(report['columns'])}")
        frame = apply_to(frame, source, report["columns"])
    if report["labels"]:
        say(f"Adding {report['labels']} saved label(s) in {', '.join(label_columns)}")
        frame = labels.apply_to(frame, source, columns=label_columns)

    parquet = dataio.is_parquet_table(source)  # a table Studio saved: it stays Parquet
    say("Writing the table..." if parquet else "Writing the CSV (the slow part)...")
    temporary = source.with_name(f".{source.name}.{os.getpid()}.tmp")
    try:
        if parquet:
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        os.replace(temporary, source)
    except Exception as exc:
        Path(temporary).unlink(missing_ok=True)
        raise PendingColumnsError(f"Could not write {source}: {exc}") from exc
    report["written"] = True

    # Everything pending is in the CSV now; remove it before anything reads it
    # against the new fingerprint.
    discard(source, report["columns"])
    labels.clear(source)
    if rebuild_fast_copy:
        say("Refreshing the fast copy...")
        try:
            dataio.ensure_fast_copy(source, force=True, progress=progress)
        except Exception as exc:
            say(f"Fast copy not refreshed ({exc}); it will be rebuilt on next use.")
    say("Done.")
    return report


def commit_status(table_path: str | Path) -> dict[str, Any]:
    """State of the latest write: idle, writing, done or failed."""
    path = store_dir(table_path) / STATUS_FILE
    if not path.is_file():
        return {"state": "idle"}
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "idle"}
    if status.get("state") == "writing" and status.get("hostname") == socket.gethostname():
        try:
            import psutil

            alive = psutil.pid_exists(int(status.get("pid", -1)))
        except Exception:
            alive = True
        if not alive:
            status.update(state="failed", message="The write stopped before finishing. Nothing was lost; try again.")
    return status


def _write_status(table_path: Path, **fields: Any) -> None:
    path = store_dir(table_path) / STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(fields, default=str), encoding="utf-8")
    os.replace(temporary, path)


def start_commit(table_path: str | Path) -> dict[str, Any]:
    """Run :func:`commit` in its own process so it survives an interface restart."""
    source = Path(table_path).expanduser()
    status = commit_status(source)
    if status.get("state") == "writing":
        return status
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    log_path = store_dir(source) / "commit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "spacemap.pending_columns", "commit", str(source)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
    fields = {"state": "writing", "pid": process.pid, "hostname": socket.gethostname(),
              "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "message": "Starting..."}
    _write_status(source, **fields)
    return fields


def _run_commit_process(table_path: str) -> int:
    source = Path(table_path)
    base = {"state": "writing", "pid": os.getpid(), "hostname": socket.gethostname(),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}

    def progress(message: str) -> None:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}", flush=True)
        _write_status(source, **base, message=message)

    try:
        report = commit(source, progress=progress)
    except Exception as exc:
        _write_status(source, **{**base, "state": "failed"}, message=str(exc),
                      finished_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        print(f"FAILED: {exc}", flush=True)
        return 1
    _write_status(source, **{**base, "state": "done"},
                  message=f"Wrote {len(report['columns'])} column(s) and {report['labels']} label(s).",
                  finished_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "commit":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
        raise SystemExit(_run_commit_process(sys.argv[2]))
    raise SystemExit("usage: python -m spacemap.pending_columns commit <table.csv>")
