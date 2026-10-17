"""The data viewer: a cell table read in pages, filtered, sorted, edited and exported, linked to the image's cells.

A table is never read whole. Its Parquet form (the file itself, or the twin of a CSV) is read by column and by row
group: the image column first (to find this image's rows), then only the columns a filter, the sort or the page
needs, and for a page only the row groups holding its rows. A large CSV without its twin yet is not read: the twin
is made in the background and the table opens when it is ready. A small CSV (no twin needed) is read into memory.

Public Data operations delegate to data_editor: paged reads, disk-backed filtering and undo, and atomic
batched saves with backups. The private measurement reader below remains for sampling and analysis.
Staged Data edits become available to other modules after Save.

Filters are ``[{"col", "values": [...]}]`` (groups), ``[{"col", "lo", "hi"}]`` (a range) or ``[{"col", "contains"}]``
(text), the same form the cell layers use, so a data filter can be shown on the image as it is.
"""

from collections import OrderedDict
import json
from pathlib import Path
import threading

import numpy as np

_TABLES = OrderedDict()   # (path, fingerprint, roi column, roi value) -> _Table
_LOCK = threading.Lock()
PAGE = 200
COLUMNS_KEPT = 12         # columns kept read per open table (the least recently used are dropped)


class Preparing(ValueError):
    """A large CSV whose Parquet twin is being made: the table opens when it is ready."""


def _parquet_of(path):
    """The table as Parquet: itself, or its twin once ready. A large CSV without one starts it and raises Preparing;
    a small CSV (or twins switched off) gives None (read into memory)."""
    from spacemap import dataio

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return path
    info = dataio.fast_copy_info(path)
    if info["exists"] and info["current"] and not dataio.is_building(path):
        return Path(info["twin"])
    if not info["worth_building"] or not info["enabled"]:
        return None
    error = dataio.build_error(path)
    if error and not dataio.is_building(path):
        raise ValueError(f"The fast copy of {path.name} could not be made: {error}")
    dataio.start_fast_copy(path)
    raise Preparing(f"Making a fast copy of {path.name} ({info['source_bytes'] / 1e9:.1f} GB) the first time it is opened; "
                    "it opens by itself when ready.")


def _key(path, roi_column, roi_value):
    from spacemap.viewer.sources import fingerprint
    from spacemap.viewer.cells import _edits_token

    return (str(path), fingerprint(str(path)), roi_column or "", None if roi_value is None else str(roi_value), _edits_token(str(path)))


def _as_text(array):
    """Values as the text the viewer shows and compares (whole numbers without ".0")."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_floating(array.type):
        whole = pc.equal(array, pc.floor(array))
        array = pc.if_else(pc.fill_null(whole, False), pc.cast(pc.cast(array, pa.int64(), safe=False), pa.string()),
                           pc.cast(array, pa.string()))
        return array
    if pa.types.is_dictionary(array.type):
        array = pc.cast(array, array.type.value_type)
    return pc.cast(array, pa.string())


class _Table:
    """One table (or one image's rows of it): which rows of the file, then each column read when it is needed."""

    def __init__(self, path, roi_column, roi_value):
        import pyarrow as pa
        import pyarrow.compute as pc

        self.path = str(path)
        self.lock = threading.RLock()
        parquet = _parquet_of(path)
        if parquet is not None:
            import pyarrow.parquet as pq

            self.file = pq.ParquetFile(parquet)
            self.memory = None
            self.schema = self.file.schema_arrow
            sizes = [self.file.metadata.row_group(i).num_rows for i in range(self.file.num_row_groups)]
            self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            self.length = int(self.file.metadata.num_rows)
        else:
            import pyarrow.csv as pacsv

            sep = "\t" if Path(path).suffix.lower() in (".tsv", ".txt") else ","
            self.file = None
            self.memory = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter=sep))
            self.schema = self.memory.schema
            self.length = self.memory.num_rows
        self.names = [n for n in self.schema.names if not n.startswith("__index_level_")]
        from spacemap import dataio, labels, pending_columns
        self.overlays = set(pending_columns.pending_names(path)) | set(labels.pending_columns(path))
        self.names = list(dict.fromkeys([*self.names, *dataio.columns(str(path))]))
        self.kinds = {f.name: "value" if (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type))
                      else "group" for f in self.schema if f.name in self.names}
        for item in pending_columns.pending(path):
            if not item["stale"]:
                import pyarrow.parquet as pq
                dtype = pq.read_schema(item["file"]).field("value").type
                self.kinds[item["column"]] = "value" if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype) else "group"
        for name in self.names:
            self.kinds.setdefault(name, "group")
        self.rows = None  # the file's row numbers of these rows (None: all of them)
        if roi_column and roi_value is not None:
            if roi_column not in self.names:
                raise ValueError(f"The table has no column {roi_column}.")
            keep = pc.equal(_as_text(self._whole(roi_column)), str(roi_value))
            self.rows = np.flatnonzero(np.asarray(pc.fill_null(keep, False))).astype(np.int64)
        self.count = self.length if self.rows is None else len(self.rows)
        self.columns = OrderedDict()   # column -> numpy array (numbers) or object array (text), for these rows
        self.views = OrderedDict()     # (filters, sort) -> positions among these rows, in view order

    def _whole(self, column):
        """One column of the whole file, as one Arrow array."""
        import pyarrow as pa

        if column in self.overlays:
            from spacemap.viewer.cells import values_at_rows
            values = values_at_rows(self.path, np.arange(self.length), column)
            return pa.array(values.tolist(), from_pandas=True)
        if self.memory is not None:
            found = self.memory.column(column)
        else:
            found = self.file.read(columns=[column]).column(column)
        return found.combine_chunks() if isinstance(found, pa.ChunkedArray) else found

    def column(self, name):
        """The saved column for these rows (numbers as floats, text as an Arrow string array)."""
        with self.lock:
            if name in self.columns:
                self.columns.move_to_end(name)
                return self.columns[name]
            if name not in self.names:
                raise ValueError(f"The table has no column {name}.")
            whole = self._whole(name)
            if self.rows is not None:
                whole = whole.take(self.rows)
            values = _values(whole, self.kinds[name] == "value")
            self.columns[name] = values
            while len(self.columns) > COLUMNS_KEPT:
                self.columns.popitem(last=False)
            return values

    def _position(self, row):
        if self.rows is None:
            return row if 0 <= row < self.length else None
        at = int(np.searchsorted(self.rows, row))
        return at if at < len(self.rows) and self.rows[at] == row else None

    def file_rows(self, positions):
        positions = np.asarray(positions, np.int64)
        return positions if self.rows is None else self.rows[positions]

    def take(self, positions, columns):
        """{column: values} for these positions (in their order), reading only the row groups that hold them."""
        import pyarrow as pa

        positions = np.asarray(positions, np.int64)
        result = {}
        missing = []
        for name in columns:
            if name in self.columns:
                result[name] = _pick(self.columns[name], positions)
            elif name in self.overlays:
                from spacemap.viewer.cells import values_at_rows
                rows = self.file_rows(positions)
                array = pa.array(values_at_rows(self.path, rows, name).tolist(), from_pandas=True)
                values = _values(array, self.kinds[name] == "value")
                result[name] = values
            else:
                missing.append(name)
        if missing and len(positions):
            rows = self.file_rows(positions)
            order = np.argsort(rows, kind="stable")
            ranked = rows[order]
            if self.memory is not None:
                part = self.memory.select(missing).take(pa.array(ranked))
            else:
                groups = np.searchsorted(self.starts, ranked, side="right") - 1
                pieces = []
                for group in np.unique(groups):
                    local = ranked[groups == group] - self.starts[group]
                    pieces.append(self.file.read_row_group(int(group), columns=missing).take(pa.array(local)))
                part = pa.concat_tables(pieces)
            back = np.empty_like(order)
            back[order] = np.arange(len(order))
            for name in missing:
                values = _pick(_values(part.column(name).combine_chunks(), self.kinds[name] == "value"), back)
                result[name] = values
        elif missing:
            for name in missing:
                result[name] = np.empty(0, float) if self.kinds[name] == "value" else []
        return {name: values if isinstance(values, (np.ndarray, list)) else values.to_pylist() for name, values in result.items()}


def _values(array, numeric):
    """Numbers as a float array (empty is NaN); anything else as an Arrow string array."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if numeric:
        if pa.types.is_boolean(array.type):
            array = pc.cast(array, pa.int8())
        return np.asarray(pc.cast(array, pa.float64()).to_numpy(zero_copy_only=False), float)
    return _as_text(array)


def _pick(values, positions):
    import pyarrow as pa

    return values[positions] if isinstance(values, np.ndarray) else values.take(pa.array(np.asarray(positions, np.int64)))


def _table(path, roi_column="", roi_value=None):
    key = _key(path, roi_column, roi_value)
    with _LOCK:
        if key in _TABLES:
            _TABLES.move_to_end(key)
            return _TABLES[key]
    table = _Table(path, roi_column, roi_value)
    with _LOCK:
        _TABLES[key] = table
        while len(_TABLES) > 4:
            _TABLES.popitem(last=False)
    return table


def _text_numbers(text):
    """A text column read as numbers (a value that is not a number is empty)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    try:
        return np.asarray(pc.cast(text, pa.float64()).to_numpy(zero_copy_only=False), float)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return np.array([_number(v) for v in text.to_pylist()], float)


def _mask(table, filters):
    import pyarrow as pa
    import pyarrow.compute as pc

    keep = np.ones(table.count, bool)
    for item in filters or []:
        column = item.get("col")
        if column not in table.names:
            continue
        values = table.column(column)
        text = values if not isinstance(values, np.ndarray) else None
        if "values" in item or "contains" in item:
            if text is None:
                text = _as_text(pa.array(values, pa.float64(), from_pandas=True))
            if "values" in item:
                found = pc.is_in(text, value_set=pa.array([str(v) for v in item["values"]], pa.string()))
            else:
                found = pc.match_substring(text, str(item["contains"]), ignore_case=True)
            keep &= np.asarray(pc.fill_null(found, False))
        else:
            numbers = values if text is None else _text_numbers(text)
            ok = np.isfinite(numbers)
            if item.get("lo") is not None:
                ok &= numbers >= float(item["lo"])
            if item.get("hi") is not None:
                ok &= numbers <= float(item["hi"])
            keep &= ok
    return keep


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _view(table, filters, sort):
    """Positions among the table's rows that match, in view order (cached for paging)."""
    key = json.dumps([filters or [], sort or None], sort_keys=True, default=str)
    with table.lock:
        if key in table.views:
            table.views.move_to_end(key)
            return table.views[key]
    positions = np.flatnonzero(_mask(table, filters))
    if sort and sort.get("col") in table.names and len(positions):
        values = _pick(table.column(sort["col"]), positions)
        if isinstance(values, np.ndarray):
            order = np.argsort(-values if sort.get("desc") else values, kind="stable")
            order = np.concatenate([order[np.isfinite(values[order])], order[~np.isfinite(values[order])]])  # empty last
        else:
            import pyarrow.compute as pc

            order = np.asarray(pc.array_sort_indices(values, order="descending" if sort.get("desc") else "ascending",
                                                     null_placement="at_end"))
        positions = positions[order]
    with table.lock:
        table.views[key] = positions
        while len(table.views) > 6:
            table.views.popitem(last=False)
    return positions


def _cell(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    return None if text in ("nan", "<NA>", "None") else text


# Public Data-panel operations use a bounded disk-backed workspace. Keep the
# lightweight measurement reader above for cluster sampling and other readers.
from .data_editor import (info, query, summary, edit, save, export, export_chunks,
                          find, labels_of, mutate, history, values_at_rows, current_name)
