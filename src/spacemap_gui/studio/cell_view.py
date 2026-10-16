"""How a cell layer is shown: which tables belong to its cells, what kind each column is, and which cells pass the
layer's filters.

A table belongs to a segmentation's cells when, for this image, the same cell numbers sit at the same places (checked
against the segmentation's own table), or, without positions, when it numbers as many cells. A table
made from another segmentation is not offered: its values would land on the wrong cells.
"""

import json
from pathlib import Path

import numpy as np

_KINDS, _MATCHES, _KEEP = {}, {}, {}
POSITION = (("x_centroid", "y_centroid"), ("centroid_x", "centroid_y"), ("X", "Y"), ("x", "y"))


def _stamp(path):
    from spacemap.viewer.sources import fingerprint

    return fingerprint(str(path))


def column_kinds(table):
    """{column: "value" (numbers) or "group" (text, categories)}, from the table's schema (or a sample of a CSV)."""
    from spacemap import dataio

    from spacemap.viewer.cells import _edits_token
    key = (str(table), _stamp(table), _edits_token(str(table)))
    if key in _KINDS:
        return _KINDS[key]
    kinds = {}
    parquet = None
    for candidate in (Path(table), dataio.fast_copy_path(table)):
        if candidate and Path(candidate).is_file() and str(candidate).endswith(".parquet"):
            parquet = candidate
            break
    if parquet is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for field in pq.read_schema(parquet):
            numeric = pa.types.is_integer(field.type) or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type)
            kinds[field.name] = "value" if numeric else "group"
    else:
        import pandas as pd

        sample = pd.read_csv(table, nrows=2000, low_memory=False)
        kinds = {c: "value" if pd.api.types.is_numeric_dtype(sample[c]) else "group" for c in sample.columns}
    from spacemap import pending_columns
    import pyarrow as pa
    import pyarrow.parquet as pq

    for item in pending_columns.pending(table):
        if not item["stale"]:
            dtype = pq.read_schema(item["file"]).field("value").type
            kinds[item["column"]] = "value" if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype) else "group"
    for name in dataio.columns(table):
        kinds.setdefault(name, "group")
    if len(_KINDS) > 32:
        _KINDS.clear()
    _KINDS[key] = kinds
    return kinds


def _rows(table, roi, columns, roi_column="batch_id", label_column="label"):
    from spacemap.viewer import cells
    from . import data_view
    import pandas as pd

    selected = data_view._table(str(table), roi_column, str(roi))
    positions = selected.file_rows(np.arange(selected.count))
    frame = {"label": pd.to_numeric(pd.Series(cells.values_at_rows(str(table), positions, label_column)), errors="coerce").to_numpy(float)}
    for column in columns:
        frame[column] = pd.to_numeric(pd.Series(cells.values_at_rows(str(table), positions, column)), errors="coerce").to_numpy(float)
    return frame


def matches(reference, table, roi, cells_count=None, roi_column="batch_id", label_column="label", reference_roi=None):
    """(True/False, reason): whether *table*'s rows for *roi* are the same cells as *reference*'s (the segmentation's
    own table), by cell number and place; without positions, by count against *cells_count*."""
    from spacemap import dataio

    key = (str(reference), str(table), str(roi), _stamp(table), _stamp(reference) if reference else None, cells_count, roi_column, label_column, reference_roi)
    if key in _MATCHES:
        return _MATCHES[key]
    try:
        columns = set(dataio.columns(str(table)))
        if roi_column not in columns or label_column not in columns:
            found = (False, "choose ROI and Segmentation label columns to find these cells")
        else:
            pair = next((p for p in POSITION if set(p) <= columns), None)
            mine = _rows(table, roi, list(pair) if pair else [], roi_column, label_column)
            if not len(mine["label"]):
                found = (False, "it has no cells of this image")
            elif not np.isfinite(mine["label"]).all() or (mine["label"] <= 0).any() or (mine["label"] != np.floor(mine["label"])).any() or len(np.unique(mine["label"])) != len(mine["label"]):
                found = (False, "segmentation labels must be unique positive whole numbers within each ROI")
            elif reference and pair and set(pair) <= set(dataio.columns(str(reference))):
                theirs = _rows(reference, reference_roi or roi, list(pair))
                import pandas as pd
                identity = pd.Index(theirs["label"])
                if not identity.is_unique:
                    found = (False, "the segmentation's cell table has repeated labels")
                else:
                    positions = identity.get_indexer(mine["label"])
                    known = positions >= 0
                    same = np.zeros(len(positions), bool)
                    same[known] = (np.abs(mine[pair[0]][known] - theirs[pair[0]][positions[known]]) <= 3) & (np.abs(mine[pair[1]][known] - theirs[pair[1]][positions[known]]) <= 3)
                    found = (bool(same.all()), "Every linked cell has the same label and position" if same.all() else
                             f"{int((~same).sum()):,} cells have different labels or positions: it was made from another segmentation")
            elif cells_count:
                count = len(mine["label"])
                found = (abs(count - cells_count) <= max(2, 0.01 * cells_count),
                         f"{count:,} rows for {cells_count:,} cells")
            else:
                found = (True, "not checked (no positions to compare)")
    except (OSError, ValueError, KeyError) as error:
        found = (False, f"it could not be read ({error})")
    if len(_MATCHES) > 128:
        _MATCHES.clear()
    _MATCHES[key] = found
    return found


def parse_filters(text):
    """[{"col", "values": [...]}] or [{"col", "lo", "hi"}]: the cells shown are those passing every filter."""
    if not text:
        return []
    filters = json.loads(text)
    if not isinstance(filters, list) or len(filters) > 12:
        raise ValueError("Filters are a short list.")
    clean = []
    for item in filters:
        if not isinstance(item, dict) or not isinstance(item.get("col"), str):
            raise ValueError("Each filter names a column.")
        if "values" in item:
            clean.append({"col": item["col"], "values": [str(v) for v in item["values"]][:20000]})
        elif "contains" in item:
            clean.append({"col": item["col"], "contains": str(item["contains"])[:200]})
        else:
            lo, hi = item.get("lo"), item.get("hi")
            clean.append({"col": item["col"], "lo": None if lo in (None, "") else float(lo), "hi": None if hi in (None, "") else float(hi)})
    return clean


def keep(table, roi, filters, roi_column="batch_id", label_column="label"):
    """A lookup by cell number: True for the cells passing every filter (None when there is no filter)."""
    from spacemap.viewer import cells

    if not filters:
        return None, 0, 0
    key = (str(table), _stamp(table), cells._edits_token(str(table)), str(roi), roi_column, label_column, json.dumps(filters, sort_keys=True))
    if key in _KEEP:
        return _KEEP[key]
    labels, _ = cells.roi_values(str(table), str(roi), filters[0]["col"], roi_column=roi_column, label_column=label_column)
    labels = np.asarray(labels, np.int64)
    passing = np.ones(len(labels), bool)
    for item in filters:
        _, values = cells.roi_values(str(table), str(roi), item["col"], roi_column=roi_column, label_column=label_column)
        if "values" in item:
            passing &= np.isin(np.asarray(values, dtype=object).astype(str), item["values"])
        elif "contains" in item:
            wanted = item["contains"].lower()
            passing &= np.array([wanted in str(v).lower() if v is not None else False for v in values], bool)
        else:
            numbers = np.asarray([float(v) if v is not None and v == v else np.nan for v in values], float)
            ok = np.isfinite(numbers)
            if item["lo"] is not None:
                ok &= numbers >= item["lo"]
            if item["hi"] is not None:
                ok &= numbers <= item["hi"]
            passing &= ok
    size = int(labels.max()) + 1 if len(labels) else 1
    lookup = np.zeros(size, bool)
    lookup[labels[passing & (labels >= 0)]] = True
    result = (lookup, int(passing.sum()), int(len(labels)))
    if len(_KEEP) > 32:
        _KEEP.clear()
    _KEEP[key] = result
    return result
