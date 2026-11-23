"""Projected analysis over separate image tables; the manifest contains paths only.

Image and cell IDs are namespaced in memory. Measurements and result journals
remain beside each original file; no merged measurement table is created.
"""
import json
from pathlib import Path

import pandas as pd

SUFFIX = ".spatial-tables.json"


def is_collection(path):
    return str(path).endswith(SUFFIX)


def entries(path, images=None):
    value = json.loads(Path(path).read_text())
    rows = value.get("tables", [])
    if value.get("format") != "spacemap-image-tables-v1" or not isinstance(rows, list):
        raise ValueError("Invalid spatial image-table collection.")
    if len({r["image"] for r in rows}) != len(rows) or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Each spatial image must have its own cell table.")
    if any(is_collection(r["path"]) or not Path(r["path"]).is_file() for r in rows):
        raise ValueError("A spatial cell table is missing. Check its pairing in Phenotyping.")
    if images:
        unknown = set(map(str, images)) - {r["image"] for r in rows}
        if unknown:
            raise ValueError("No paired cell table for: " + ", ".join(sorted(unknown)))
        rows = [r for r in rows if r["image"] in set(map(str, images))]
    return rows


def columns(path, include_pending=True):
    from . import dataio
    return list(dict.fromkeys(["batch_id", "cell_id", *[c for r in entries(path)
        for c in dataio.columns(r["path"], include_pending=include_pending)]]))


def validate(path, required, images=None, *, anchors=False):
    """Check every selected file, not just the union of available columns."""
    from . import dataio
    for row in entries(path, images):
        available = dataio.columns(row["path"])
        missing = set(required) - set(available) - {"batch_id", "cell_id"}
        if missing:
            raise ValueError(f"{row['image']}: no {', '.join(sorted(missing))} column. Choose columns present in every selected image table.")
        if anchors and (not row.get("anchor") or row["anchor"] not in available):
            raise ValueError(f"{row['image']}: choose a unique Cell ID in Phenotyping's Column roles.")
        if anchors:
            ids = dataio.load(row['path'], columns=[row['anchor']], build_fast_copy=False)[row['anchor']].astype('string')
            if ids.isna().any() or ids.str.strip().eq('').any() or ids.duplicated().any():
                raise ValueError(f"{row['image']}: Cell IDs are missing or repeated. Choose unique IDs in Column roles.")


def load(path, columns=None, *, images=None, **kwargs):
    from . import dataio
    wanted = list(dict.fromkeys(columns)) if columns else globals()["columns"](path, kwargs.get("apply_labels", True))
    missing = set(wanted) - set(globals()["columns"](path, kwargs.get("apply_labels", True)))
    if missing:
        raise dataio.DataIOError("No column(s): " + ", ".join(sorted(missing)))
    parts = []
    for row in entries(path, images):
        available = dataio.columns(row["path"], include_pending=kwargs.get("apply_labels", True))
        names = [c for c in wanted if c in available and c not in {"batch_id", "cell_id"}]
        anchor = row.get("anchor")
        if "cell_id" in wanted and anchor in available:
            names.append(anchor)
        # A virtual-ID-only request still needs a physical column to count rows.
        frame = dataio.load(row["path"], columns=list(dict.fromkeys(names)) or available[:1], **kwargs)
        if "cell_id" in wanted:
            ids = frame[anchor] if anchor in frame else pd.Series(range(len(frame)))
            prefix = json.dumps(row["image"], ensure_ascii=False) + ":"
            frame["cell_id"] = prefix + ids.astype("string")
        if "batch_id" in wanted:
            frame["batch_id"] = row["image"]
        parts.append(frame.reindex(columns=wanted))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=wanted)


def source_for_image(path, image):
    return entries(path, [str(image)])[0]["path"]
