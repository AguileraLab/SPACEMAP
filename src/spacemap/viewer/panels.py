"""Validated, revision-checked edits to the phenotype panel shared with Tab 7."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from spacemap.sharing import open_lock


def read(path):
    content = Path(path).read_bytes()
    return {"panel": json.loads(content), "revision": hashlib.sha256(content).hexdigest()}


def validate(payload, available):
    if not isinstance(payload, dict):
        raise ValueError("Expected a phenotype panel.")
    columns, rows = payload.get("criteria_cols"), payload.get("table")
    if not isinstance(columns, list) or not 1 <= len(columns) <= 20:
        raise ValueError("Use between one and twenty criteria groups.")
    if any(not isinstance(c, str) or not c.strip() or len(c) > 100 for c in columns):
        raise ValueError("Criteria groups need short, non-empty names.")
    if len(set(columns)) != len(columns) or set(columns) & {"Phenotype", "Supportive", "Must not have"}:
        raise ValueError("Criteria group names must be distinct.")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError("Use between one and 500 phenotypes.")
    names, cleaned = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Expected a phenotype row.")
        name = str(row.get("Phenotype") or "").strip()
        if not name or len(name) > 200 or name in names:
            raise ValueError("Phenotypes need unique, non-empty names (up to 200 characters).")
        names.add(name)
        record = {"Phenotype": name}
        for column in [*columns, "Supportive", "Must not have"]:
            markers = row.get(column) or []
            if not isinstance(markers, list) or any(not isinstance(m, str) for m in markers):
                raise ValueError("Marker groups must be lists of marker names.")
            if any(m not in available for m in markers):
                raise ValueError(f"Unknown marker in {name}, {column}.")
            record[column] = list(dict.fromkeys(markers))
        cleaned.append(record)
    if not any(row[c] for row in cleaned for c in columns):
        raise ValueError("At least one phenotype needs a criterion.")
    return {"criteria_cols": columns, "table": cleaned}


@contextmanager
def _locked(path):
    import fcntl

    with open_lock(Path(str(path) + ".lock")) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save(path, payload, revision, available):
    path = Path(path)
    panel = validate(payload, set(available))
    with _locked(path):
        current = read(path)
        if current["revision"] != revision:
            raise ValueError("The panel changed elsewhere. Reopen the editor before saving.")
        backup = path.with_name(path.name + ".before-viewer-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        shutil.copy2(path, backup)
        fd, name = tempfile.mkstemp(prefix=".panel-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({**current["panel"], **panel}, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
            from spacemap.sharing import share
            share(path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return {**read(path), "backup": backup.name}
