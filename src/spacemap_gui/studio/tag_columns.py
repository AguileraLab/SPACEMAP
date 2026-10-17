"""Cell tagging columns as the user names them, in the units they choose.

Classic tagging (``spacemap.cell_spatial_annotation.annotate_cells``) writes fixed names in image pixels:
``<label>`` (inside: 1/0), ``<label>_roi`` (which piece), ``<label>_roi_size`` (its area, px²),
``<label>_distance`` (signed: negative inside, positive outside), ``<label>_bed`` / ``<label>_bed_id`` and the edge
flag. Studio runs the same engine, then gives each column the user's name, converts distances to µm and areas to
µm² when asked (with the image's pixel size), and can add one ``bed`` column naming the annotation each cell was
given to. Names are kept per annotation and kind: ``{"<label>|<kind>": "<column>"}``.
"""

import re

KINDS = ("inside", "roi", "roi_size", "distance", "bed", "bed_id")
UNITS = ("um", "px")
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")


def engine_name(label, kind):
    """The column the engine writes for *label* and *kind*."""
    return {"inside": label, "roi": f"{label}_roi", "roi_size": f"{label}_roi_size", "distance": f"{label}_distance",
            "bed": f"{label}_bed", "bed_id": f"{label}_bed_id"}[kind]


def column_name(names, label, kind):
    """The user's name for a column, else the engine's (a space or other mark becomes _)."""
    chosen = (names or {}).get(f"{label}|{kind}")
    if chosen:
        return chosen
    return re.sub(r"[^A-Za-z0-9_.]+", "_", engine_name(label, kind)).strip("_") or kind


def check_names(names):
    if not isinstance(names, dict) or len(names) > 2000:
        raise ValueError("Send the column names as a list of name choices.")
    for key, value in names.items():
        label, _, kind = str(key).rpartition("|")
        if not label or kind not in KINDS:
            raise ValueError("Unknown column.")
        if value and not NAME.match(str(value)):
            raise ValueError(f"{value}: a column name starts with a letter or _, then letters, digits, _ or . (up to 64).")
    return names


def finish(frame, labels, names, units, microns, bed_labels=(), bed_column="", edge_column=""):
    """Rename the engine's columns, convert units, add the one-column bed. Returns (frame, columns added, the
    columns that are measurements: distances and sizes, shown as a colour scale)."""
    import numpy as np
    import pandas as pd

    renamed, added, scales = {}, [], []
    for label in labels:
        for kind in KINDS:
            engine = engine_name(label, kind)
            if engine not in frame:
                continue
            if units == "um" and microns:
                if kind == "distance":
                    frame[engine] = pd.to_numeric(frame[engine], errors="coerce") * float(microns)
                elif kind == "roi_size":
                    frame[engine] = pd.to_numeric(frame[engine], errors="coerce") * float(microns) ** 2
            target = column_name(names, label, kind)
            if target != engine:
                renamed[engine] = target
            added.append(target)
            if kind in ("distance", "roi_size"):
                scales.append(target)
    frame = frame.rename(columns=renamed)
    if bed_column and bed_labels:
        bed = pd.Series(np.nan, index=frame.index, dtype=object)
        for label in bed_labels:
            column = column_name(names, label, "bed")
            if column in frame:
                bed[pd.to_numeric(frame[column], errors="coerce") == 1] = label
        frame[bed_column] = bed
        added.append(bed_column)
    if edge_column and edge_column in frame:
        added.append(edge_column)
    duplicates = sorted({name for name in added if added.count(name) > 1})
    if duplicates:
        raise ValueError("Two columns would have the same name: " + ", ".join(duplicates) + ". Rename one.")
    return frame, list(dict.fromkeys(added)), scales


def planned(rules, names, bed_labels=(), bed_column="", edge_column=""):
    """The columns a run adds, in order, for the page to show before running."""
    kinds = {"Binary (0/1)": "inside", "ROI ID": "roi", "ROI Size": "roi_size", "Distance": "distance"}
    columns = []
    for record in rules:
        label = record["Annotation Label"]
        columns += [column_name(names, label, kind) for field, kind in kinds.items() if record.get(field)]
    for label in bed_labels or []:
        columns += [column_name(names, label, "bed"), column_name(names, label, "bed_id")]
    if bed_column and bed_labels:
        columns.append(bed_column)
    if edge_column:
        columns.append(edge_column)
    return list(dict.fromkeys(columns))
