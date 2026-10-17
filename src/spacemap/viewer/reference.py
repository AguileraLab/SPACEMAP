"""Small durable receipts and review flags for viewer label edits.

Cell values still use the shared label store. Receipts make retrying a lost
save response safe, including after restarting the app. Review identity is
ROI + cell number + column, so writing pending labels into CSV retains flags.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

from spacemap import labels as label_store
from . import cells
from .sources import fingerprint

_PRESETS = cells._Cache(16)


def phenotype_names(table, column="phenotype_ml_consensus"):
    """Exact names across a loaded table, in small chunks, with pending edits.

    Ordinary reference edits do not invalidate the consensus-name cache.
    """
    from spacemap import dataio, pending_columns
    import pyarrow.parquet as pq

    if column not in dataio.columns(table):
        return []
    pending = next((p for p in pending_columns.pending(table) if p["column"] == column and not p["stale"]), None)
    edits = label_store.load(table)
    if not edits.empty:
        edits = edits[edits["column"] == column].sort_values("row")
    edit_stamp = int(pd.util.hash_pandas_object(edits, index=False).sum()) if not edits.empty else 0
    key = (str(table), fingerprint(table), column, fingerprint(pending["file"]) if pending else "", edit_stamp)
    cached = _PRESETS.get(key)
    if cached is not None:
        return cached
    fast = dataio.fast_copy_info(table)
    parquet = pending["file"] if pending else str(table) if Path(table).suffix.lower() == ".parquet" else fast["twin"] if fast.get("current") else None
    native = column in dataio.columns(table, include_pending=False)
    if not pending and not native:
        batches = ()
    elif parquet:
        batches = (
            batch.column(0).to_numpy(zero_copy_only=False)
            for batch in pq.ParquetFile(parquet).iter_batches(
                batch_size=65536, columns=["value" if pending else column]
            )
        )
    else:
        batches = (chunk[column].to_numpy() for chunk in pd.read_csv(table, usecols=[column], chunksize=65536))
    names, offset = set(), 0
    if not pending and not native and not edits.empty:
        names.update(str(v).strip() for v in pd.unique(edits["value"]) if pd.notna(v) and str(v).strip())
    for batch in batches:
        values = np.asarray(batch, dtype=object).copy()
        if not edits.empty:
            selected = edits[(edits["row"] >= offset) & (edits["row"] < offset + len(values))]
            values[selected["row"].to_numpy(dtype=int) - offset] = selected["value"].to_numpy()
        names.update(str(v).strip() for v in pd.unique(values) if pd.notna(v) and str(v).strip())
        offset += len(values)
    result = sorted(names, key=str.casefold)
    _PRESETS.put(key, result)
    return result


#: A manual-reference column: "ref", "ref1", "reference" or "manual" as a whole
#: word of the name (phenotype_ref1, manual_reference...). Matching "ref"
#: anywhere took phenotype_ml_consensus_refined ("refined") and file_prefix
#: ("prefix") for reference columns, and made them the default for labelling.
REFERENCE_COLUMN = re.compile(r"(?:^|_)(?:ref\d*|reference|manual)(?:_|\d|$)", re.IGNORECASE)


def is_reference_column(name) -> bool:
    return bool(REFERENCE_COLUMN.search(str(name)))


def columns(table):
    """Manual columns include explicitly registered custom names, not just ref*."""
    from spacemap import dataio
    with database(table) as db:
        registered = {row[0] for row in db.execute("SELECT DISTINCT column_name FROM reviews")} if db else set()
        if db and db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='manual_columns'").fetchone():
            registered.update(row[0] for row in db.execute("SELECT name FROM manual_columns"))
    return [name for name in dataio.columns(table) if is_reference_column(name) or name in registered]


def register_column(table, column):
    with database(table, write=True) as db:
        db.execute("INSERT OR IGNORE INTO manual_columns(name) VALUES (?)", (column,))
        db.commit()


def validate_column(table, column, **keys):
    """A custom label name must not overwrite cell identities or measurements."""
    from spacemap import dataio
    if not isinstance(column, str) or not column.strip() or len(column) > 128 or column != column.strip() or re.search(r"[\x00-\x1f/\\]", column):
        raise ValueError("Enter a column name of 1–128 characters without slashes or line breaks.")
    if column in {keys.get("roi_column", "batch_id"), keys.get("label_column", "label"), "cell_id", "x_centroid", "y_centroid"}:
        raise ValueError("Choose a reference column, not a cell identity or position column.")
    if column in dataio.columns(table) and column not in columns(table):
        raise ValueError("That name is already used by another column. Choose a new reference name.")
    return column


def options(tables, selected):
    from spacemap import dataio

    references = columns(selected)
    choices = list(dict.fromkeys([c for c in dataio.columns(selected) if c.casefold().startswith("phenotype_")] + references))
    names, sources, unavailable = set(), [], []
    for table in dict.fromkeys(tables):
        try:
            found = phenotype_names(table)
            if found:
                names.update(found)
                sources.append(Path(table).name)
        except (OSError, ValueError, KeyError):
            unavailable.append(Path(table).name)
    return {
        "columns": sorted(choices, key=str.casefold),
        "reference_columns": sorted(references, key=str.casefold),
        "phenotypes": sorted(names, key=str.casefold),
        "sources": sources,
        "unavailable": unavailable,
    }


def path_for(table):
    return Path(str(label_store.sidecar_path(table)) + ".review.sqlite").resolve()


@contextmanager
def database(table, *, write=False):
    path = path_for(table)
    if not write and not path.exists():
        yield None
        return
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path if write else path.as_uri() + "?mode=ro", uri=not write, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        if write:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, signature TEXT, fingerprint TEXT,
                    updates TEXT, result TEXT, done INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS reviews (
                    roi TEXT, label INTEGER, column_name TEXT, value TEXT,
                    saved_at TEXT, source TEXT, flagged INTEGER DEFAULT 0,
                    PRIMARY KEY (roi,label,column_name));
                CREATE INDEX IF NOT EXISTS review_recent ON reviews(roi,column_name,saved_at DESC);
                CREATE INDEX IF NOT EXISTS review_flagged ON reviews(roi,column_name,flagged,saved_at DESC);
                CREATE TABLE IF NOT EXISTS checked (
                    roi TEXT,label INTEGER,column_name TEXT,checked INTEGER,checked_at TEXT,
                    PRIMARY KEY(roi,label,column_name));
                CREATE TABLE IF NOT EXISTS manual_columns (name TEXT PRIMARY KEY);
            """)
        yield db
    finally:
        db.close()


def save(table, roi, column, values, request_id, *, source="typed", expected=None, provenance=None, **keys):
    """Assign one batch once; retries return its original Undo information."""
    if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
        raise ValueError("The save needs a valid request number.")
    if column in {
        keys.get("roi_column", "batch_id"),
        keys.get("label_column", "label"),
        "cell_id",
        "x_centroid",
        "y_centroid",
    }:
        raise ValueError("Choose a phenotype column, not a cell number or position column.")
    if source not in {"typed", "accepted", "undo"}:
        raise ValueError("Unknown assignment source.")
    values = {str(int(k)): "" if v is None else str(v) for k, v in values.items()}
    receipt_inputs = [roi, column, values, source, expected]
    if provenance is not None:
        if not isinstance(provenance, dict) or len(json.dumps(provenance)) > 16000:
            raise ValueError("The suggestion details could not be saved.")
        receipt_inputs.append(provenance)
    signature = hashlib.sha256(json.dumps(receipt_inputs, sort_keys=True).encode()).hexdigest()
    with label_store.write_lock(table), database(table, write=True) as db:
        receipt = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        if receipt:
            if receipt["signature"] != signature:
                raise ValueError("This save number was already used for different changes.")
            result = json.loads(receipt["result"])
            if receipt["done"]:
                return result
            if receipt["fingerprint"] != fingerprint(table):
                raise ValueError("The table changed before this save finished. Reopen the image and check the labels.")
            updates = json.loads(receipt["updates"])
        else:
            labels, rows = cells.assignment_rows(table, roi, values, **keys)
            previous = cells.values_at_rows(table, rows, column)
            if expected is not None:
                if not isinstance(expected, dict) or any(
                    str(label) not in expected or ("" if pd.isna(old) else str(old)) != str(expected[str(label)] or "")
                    for label, old in zip(labels, previous)
                ):
                    raise ValueError("Some labels changed after this save. Review those cells before undoing.")
            stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            updates = [
                {"row": int(row), "label": int(label), "column": column, "value": values[str(label)], "saved_at": stamp}
                for row, label in zip(rows, labels)
            ]
            result = {
                "assigned": len(updates),
                "labels": labels.tolist(),
                "previous": {str(label): None if pd.isna(old) else str(old) for label, old in zip(labels, previous)},
            }
            if provenance is not None:
                result["suggestion"] = provenance
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,0)",
                (request_id, signature, fingerprint(table), json.dumps(updates), json.dumps(result)),
            )
            db.commit()  # preserve the original Undo values before changing labels
        label_store.append(table, updates)
        for update in updates:
            db.execute(
                "UPDATE checked SET checked=0 WHERE roi=? AND label=? AND column_name=?", (roi, update["label"], column)
            )
            db.execute(
                """INSERT INTO reviews (roi,label,column_name,value,saved_at,source)
                          VALUES (?,?,?,?,?,?) ON CONFLICT(roi,label,column_name) DO UPDATE SET
                          value=excluded.value,saved_at=excluded.saved_at,source=excluded.source
                          WHERE excluded.saved_at >= reviews.saved_at""",
                (roi, update["label"], column, update["value"], update["saved_at"], source),
            )
        db.execute("UPDATE requests SET done=1 WHERE id=?", (request_id,))
        db.commit()
        return result


def set_flag(table, roi, column, label, flagged, **keys):
    with label_store.write_lock(table):
        labels, rows = cells.assignment_rows(table, roi, [label], **keys)
        if not len(rows):
            raise ValueError("This cell has no row in the selected table.")
        value = cells.values_at_rows(table, rows, column)[0]
        value = "" if pd.isna(value) else str(value)
        stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with database(table, write=True) as db:
            db.execute(
                "INSERT INTO checked VALUES (?,?,?,?,?) ON CONFLICT(roi,label,column_name) DO UPDATE SET checked=excluded.checked,checked_at=excluded.checked_at",
                (roi, int(label), column, int(not flagged), stamp),
            )
            db.execute(
                """INSERT INTO reviews VALUES (?,?,?,?,?,'review',?)
                          ON CONFLICT(roi,label,column_name) DO UPDATE SET flagged=excluded.flagged""",
                (roi, int(label), column, value, stamp, int(bool(flagged))),
            )
            db.commit()
    return {"label": int(label), "flagged": bool(flagged), "checked": not bool(flagged)}


def get_flag(table, roi, column, label):
    with database(table) as db:
        row = (
            db.execute(
                "SELECT flagged FROM reviews WHERE roi=? AND column_name=? AND label=?", (roi, column, int(label))
            ).fetchone()
            if db is not None
            else None
        )
        checked = (
            db.execute(
                "SELECT checked FROM checked WHERE roi=? AND column_name=? AND label=?", (roi, column, int(label))
            ).fetchone()
            if db and db.execute("SELECT name FROM sqlite_master WHERE name='checked'").fetchone()
            else None
        )
    return {
        "label": int(label),
        "flagged": bool(row[0]) if row else False,
        "checked": bool(checked[0]) if checked else False,
    }


def review(table, roi, column, *, mode="recent", offset=0, limit=30, **keys):
    limit, offset = min(100, max(1, int(limit))), max(0, int(offset))
    with database(table) as db:
        if mode == "all":
            labels, values = cells.roi_values(table, roi, column, **keys)
            keep = ~pd.isna(values) & np.asarray(
                [str(v).strip() not in {"", "Unassigned", "Unknown", "Unclassified"} for v in values]
            )
            selected = labels[keep][offset : offset + limit]
            items = [
                {"label": int(label), "value": str(value), "flagged": False}
                for label, value in zip(selected, values[keep][offset : offset + limit])
            ]
            if db is not None and len(selected):
                flags = {
                    int(row[0])
                    for row in db.execute(
                        "SELECT label FROM reviews WHERE roi=? AND column_name=? AND flagged=1", (roi, column)
                    )
                }
                for item in items:
                    item["flagged"] = item["label"] in flags
            total = int(keep.sum())
        elif mode in {"recent", "flagged"}:
            predicate = "roi=? AND column_name=?" + (" AND flagged=1" if mode == "flagged" else "")
            rows = (
                list(
                    db.execute(
                        f"SELECT label,source,saved_at,flagged FROM reviews WHERE {predicate} ORDER BY saved_at DESC,label LIMIT ? OFFSET ?",
                        (roi, column, limit, offset),
                    )
                )
                if db is not None
                else []
            )
            total = (
                db.execute(f"SELECT count(*) FROM reviews WHERE {predicate}", (roi, column)).fetchone()[0]
                if db is not None
                else 0
            )
            items = [dict(row) for row in rows]
            # Report current values, including edits made in the classic GUI.
            labels, positions = cells.assignment_rows(table, roi, [i["label"] for i in items], **keys)
            current = dict(zip(labels.tolist(), cells.values_at_rows(table, positions, column)))
            for item in items:
                value = current.get(item["label"])
                item["value"] = "" if pd.isna(value) else str(value)
                item["flagged"] = bool(item["flagged"])
                item["missing"] = item["label"] not in current
        else:
            raise ValueError("Choose recent cells, all assigned cells, or cells needing review.")
    return {"items": items, "total": int(total), "offset": offset, "limit": limit}


def save_groups(table, groups, column, value, request_id, *, expected=None, **keys):
    """One durable save for an exact selection spanning several ROIs.

    Resolve every ROI/ID pair before appending anything. The label store remains
    the single writer, and the receipt retains previous values for guarded Undo.
    """
    validate_column(table, column, **keys)
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("Enter a phenotype name of up to 128 characters.")
    if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
        raise ValueError("The save needs a valid request number.")
    payload = [{"roi": group["roi"], "ids": [int(v) for v in group["ids"]]} for group in groups]
    signature = hashlib.sha256(json.dumps([payload, column, value, expected], sort_keys=True).encode()).hexdigest()
    with label_store.write_lock(table), database(table, write=True) as db:
        receipt = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        if receipt:
            if receipt["signature"] != signature:
                raise ValueError("This save number was already used for another selection.")
            result = json.loads(receipt["result"])
            if receipt["done"]:
                return result
            if receipt["fingerprint"] != fingerprint(table):
                raise ValueError("The table changed during this save. Review the selected cells.")
            updates = json.loads(receipt["updates"])
        else:
            updates = []
            previous = []
            stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            for group in payload:
                ids, rows = cells.assignment_rows(table, group["roi"], group["ids"], **keys)
                if set(ids) != set(group["ids"]):
                    raise ValueError("Some selected cells have no unique matching row. Nothing was saved.")
                old = cells.values_at_rows(table, rows, column)
                for label, row, before in zip(ids, rows, old):
                    before = None if pd.isna(before) else str(before)
                    key = group["roi"] + "\x1f" + str(label)
                    if expected is not None and (key not in expected or str(before or "") != str(expected[key] or "")):
                        raise ValueError("Some selected labels changed again. Their newer labels were kept.")
                    previous.append({"roi": group["roi"], "label": int(label), "value": before})
                    updates.append(
                        {
                            "row": int(row),
                            "roi": group["roi"],
                            "label": int(label),
                            "column": column,
                            "value": value,
                            "saved_at": stamp,
                        }
                    )
            if len({update["row"] for update in updates}) != len(updates):
                raise ValueError("The selection repeats a cell row.")
            result = {"assigned": len(updates), "previous": previous}
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,0)",
                (request_id, signature, fingerprint(table), json.dumps(updates), json.dumps(result)),
            )
            db.commit()
        label_store.append(table, updates)
        db.execute("INSERT OR IGNORE INTO manual_columns(name) VALUES (?)", (column,))
        for update in updates:
            db.execute(
                "UPDATE checked SET checked=0 WHERE roi=? AND label=? AND column_name=?",
                (update["roi"], update["label"], column),
            )
            db.execute(
                """INSERT INTO reviews (roi,label,column_name,value,saved_at,source) VALUES (?,?,?,?,?,'typed')
                          ON CONFLICT(roi,label,column_name) DO UPDATE SET value=excluded.value,saved_at=excluded.saved_at,source=excluded.source""",
                (update["roi"], update["label"], column, update["value"], update["saved_at"]),
            )
        db.execute("UPDATE requests SET done=1 WHERE id=?", (request_id,))
        db.commit()
        return result


def undo_groups(table, column, previous, expected, request_id, **keys):
    """Check all expected values before atomically restoring a multi-ROI edit."""
    signature = hashlib.sha256(json.dumps([column, previous, expected], sort_keys=True).encode()).hexdigest()
    with label_store.write_lock(table), database(table, write=True) as db:
        receipt = db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        if receipt:
            if receipt["signature"] != signature:
                raise ValueError("This Undo request was already used for another selection.")
            result = json.loads(receipt["result"])
            if receipt["done"]:
                return result
            if receipt["fingerprint"] != fingerprint(table):
                raise ValueError("The table changed before Undo finished.")
            updates = json.loads(receipt["updates"])
        else:
            updates = []
            stamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            for roi in dict.fromkeys(item["roi"] for item in previous):
                items = [item for item in previous if item["roi"] == roi]
                ids = [item["label"] for item in items]
                found, rows = cells.assignment_rows(table, roi, ids, **keys)
                if set(found) != set(ids):
                    raise ValueError("Some selected cells changed. Nothing was undone.")
                current = cells.values_at_rows(table, rows, column)
                values = {item["label"]: item["value"] for item in items}
                for label, row, value in zip(found, rows, current):
                    if ("" if pd.isna(value) else str(value)) != expected:
                        raise ValueError("Some labels changed again. Their newer labels were kept.")
                    updates.append(
                        {
                            "row": int(row),
                            "roi": roi,
                            "label": int(label),
                            "column": column,
                            "value": values[int(label)] or "",
                            "saved_at": stamp,
                        }
                    )
            result = {"restored": len(updates)}
            db.execute(
                "INSERT INTO requests VALUES (?,?,?,?,?,0)",
                (request_id, signature, fingerprint(table), json.dumps(updates), json.dumps(result)),
            )
            db.commit()
        label_store.append(table, updates)
        for update in updates:
            db.execute(
                "UPDATE checked SET checked=0 WHERE roi=? AND label=? AND column_name=?",
                (update["roi"], update["label"], column),
            )
            db.execute(
                "UPDATE reviews SET value=?,saved_at=?,source='undo' WHERE roi=? AND label=? AND column_name=?",
                (update["value"], update["saved_at"], update["roi"], update["label"], column),
            )
        db.execute("UPDATE requests SET done=1 WHERE id=?", (request_id,))
        db.commit()
        return result
