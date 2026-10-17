"""The cell layer: segmentation drawn as outlines or filled, coloured by any cell column.

Cell outlines (QuPath-style GeoJSON, one polygon per cell with a ``label``)
are turned once into a *label pyramid*: an image whose pixel values are cell
numbers, with lower resolutions for zooming out. Measured on a 74,680-cell ROI:
41 MB of GeoJSON becomes an 8 MB pyramid in about 5 seconds, and it is rebuilt
automatically when the segmentation file changes.

Colouring a million cells then only needs a colour table indexed by cell number
(``lut[label] -> RGBA``), built from any per-cell table that has the ROI name and
the cell label -- the unified table, the neighbourhood table, and so on.
Switching the colouring is instant: tiles are recoloured, not redrawn.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .render import TILE, tile_bounds
from .sources import ImageInfo, fingerprint, open_source

LABEL_SUFFIX = ".cell_labels.tif"
_BUILD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Label pyramid
# ---------------------------------------------------------------------------


def label_pyramid_path(cache_dir: str | Path, geojson_path: str | Path) -> Path:
    name = Path(geojson_path).name
    for suffix in (".geojson", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return Path(cache_dir) / f"{name}{LABEL_SUFFIX}"


def _recorded_source(pyramid: Path) -> dict:
    import tifffile

    try:
        with tifffile.TiffFile(pyramid) as tif:
            return json.loads(tif.pages[0].description or "{}").get("spacemap_labels", {})
    except Exception:
        return {}


def _up_to_date(target: Path, wanted: dict) -> bool:
    """The pyramid on disk was built from this outline file (same content, image size) and is usable.

    Sources are compared by where they point: two storage aliases can refer to the
    same physical folder, and comparing the text rebuilt every ROI's outlines
    (minutes for a large one) whenever SPACEMAP used the other spelling.
    """
    recorded = _recorded_source(target) if target.is_file() else {}
    if not recorded or any(recorded.get(key) != wanted[key] for key in ("fingerprint", "width", "height")):
        return False
    try:
        return os.path.realpath(recorded.get("source", "")) == os.path.realpath(wanted["source"])
    except (OSError, ValueError):
        return recorded.get("source") == wanted["source"]


def ensure_label_pyramid(geojson_path: str | Path, width: int, height: int, cache_dir: str | Path) -> Path:
    """The label pyramid for *geojson_path*, building or rebuilding it if needed."""
    geojson_path = Path(geojson_path)
    target = label_pyramid_path(cache_dir, geojson_path)
    wanted = {
        "source": str(geojson_path),
        "fingerprint": fingerprint(geojson_path),
        "width": int(width),
        "height": int(height),
    }
    with _BUILD_LOCK:
        if _up_to_date(target, wanted):
            return target
        build_label_pyramid(geojson_path, width, height, target, wanted)
    return target


_BUILDING: dict[str, threading.Thread] = {}

# A ROI's outline file can be several hundred MB. Reading and drawing it holds
# Python's global lock for a minute or more, which would freeze the whole
# interface (pages, viewer, tiles). So the build runs in a separate process.
_BUILD_SCRIPT = (
    "import sys; from spacemap.viewer.cells import ensure_label_pyramid; "
    "ensure_label_pyramid(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])"
)


def _build_in_process(geojson_path: Path, width: int, height: int, cache_dir: str | Path) -> None:
    """Build a label pyramid in a child process; waits for it (the wait does not hold the lock)."""
    source_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH", "")]))
    try:
        done = subprocess.run(
            [sys.executable, "-c", _BUILD_SCRIPT, str(geojson_path), str(int(width)), str(int(height)), str(cache_dir)],
            env=env,
            capture_output=True,
            text=True,
        )
    except OSError:
        done = None
    if done is None or done.returncode != 0:
        wanted = {
            "source": str(geojson_path),
            "fingerprint": fingerprint(geojson_path),
            "width": int(width),
            "height": int(height),
        }
        if _up_to_date(label_pyramid_path(cache_dir, geojson_path), wanted):
            return  # another SPACEMAP process built it meanwhile
        detail = (done.stderr or "").strip().splitlines()[-1:] if done is not None else []
        print(
            f"SPACEMAP viewer: building cell outlines in a separate process failed "
            f"({detail[0] if detail else 'could not start'}); building them here instead.",
            file=sys.stderr,
        )
        ensure_label_pyramid(geojson_path, width, height, cache_dir)


def prepare_label_pyramid(
    geojson_path: str | Path, width: int, height: int, cache_dir: str | Path
) -> tuple[Path, bool]:
    """(label pyramid path, ready now). When it needs building, it is built in the background.

    Opening a ROI for the first time then shows the image at once; the cells
    appear when their outlines are ready (see pyramid_ready).
    """
    geojson_path = Path(geojson_path)
    target = label_pyramid_path(cache_dir, geojson_path)
    wanted = {
        "source": str(geojson_path),
        "fingerprint": fingerprint(geojson_path),
        "width": int(width),
        "height": int(height),
    }
    if _up_to_date(target, wanted):
        return target, True
    with _BUILD_LOCK:
        running = _BUILDING.get(str(target))
        if running is None or not running.is_alive():
            thread = threading.Thread(
                target=_build_in_process,
                args=(geojson_path, width, height, cache_dir),
                name=f"spacemap-outlines-{target.stem}",
                daemon=True,
            )
            _BUILDING[str(target)] = thread
            thread.start()
    return target, False


def pyramid_ready(pyramid: str | Path) -> bool:
    """True when a label pyramid exists and is not being (re)built."""
    running = _BUILDING.get(str(pyramid))
    return Path(pyramid).is_file() and not (running is not None and running.is_alive())


def pyramid_cell_count(pyramid: str | Path) -> int | None:
    """How many cells were drawn into a label pyramid (None if it predates the count)."""
    count = _recorded_source(Path(pyramid)).get("cells")
    return int(count) if count is not None else None


def build_label_pyramid(geojson_path: Path, width: int, height: int, target: Path, meta: dict) -> Path:
    import rasterio.features
    import tifffile
    from affine import Affine

    features = json.loads(Path(geojson_path).read_text())
    features = features["features"] if isinstance(features, dict) else features

    drawn = [0]

    def shapes():
        for feature in features:
            props = feature.get("properties") or {}
            label = props.get("label", props.get("cell_label"))
            geometry = feature.get("geometry")
            if label is None or not geometry:
                continue
            drawn[0] += 1
            yield geometry, int(label)

    labels = rasterio.features.rasterize(
        shapes(), out_shape=(int(height), int(width)), fill=0, transform=Affine.identity(), dtype="uint32"
    )
    meta = {**meta, "cells": drawn[0]}
    levels = []
    level = labels
    while max(level.shape) > TILE:
        level = level[::2, ::2]  # nearest: keeps real cell numbers
        levels.append(level)
    target.parent.mkdir(parents=True, exist_ok=True)
    # One temporary name per build: two SPACEMAP processes building the same
    # outlines at once moved each other's file away.
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    options = dict(tile=(256, 256), compression=("deflate", 1), photometric="minisblack")
    with tifffile.TiffWriter(temporary, bigtiff=True) as writer:
        writer.write(
            labels, subifds=len(levels), description=json.dumps({"spacemap_labels": meta}), metadata=None, **options
        )
        for lower in levels:
            writer.write(lower, subfiletype=1, **options)
    temporary.replace(target)
    return target


# ---------------------------------------------------------------------------
# Label tiles (nearest resampling, cached)
# ---------------------------------------------------------------------------


class _Cache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key, value) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)


_LABEL_TILES = _Cache(1024)


def label_tile(pyramid: str, image: ImageInfo, z: int, x: int, y: int) -> np.ndarray:
    """Cell numbers for one viewer tile of the image the pyramid belongs to."""
    labels_info = open_source(pyramid).info
    key = (pyramid, labels_info.fingerprint, z, x, y)
    cached = _LABEL_TILES.get(key)
    if cached is not None:
        return cached
    x0, y0, x1, y1, out_w, out_h = tile_bounds(image, z, x, y)
    if out_w == 0 or out_h == 0:
        return np.zeros((0, 0), dtype=np.uint32)
    scale = 2**z
    level_index = 0
    for index, level in enumerate(labels_info.levels):
        if level.downsample_x <= scale * 1.001:
            level_index = index
    level = labels_info.levels[level_index]
    lx0, ly0 = math.floor(x0 / level.downsample_x), math.floor(y0 / level.downsample_y)
    lx1, ly1 = math.ceil(x1 / level.downsample_x), math.ceil(y1 / level.downsample_y)
    region = open_source(pyramid).read_region(level_index, 0, lx0, ly0, lx1, ly1)
    if region.shape != (out_h, out_w):
        rows = np.minimum((np.arange(out_h) * region.shape[0] / out_h).astype(int), region.shape[0] - 1)
        cols = np.minimum((np.arange(out_w) * region.shape[1] / out_w).astype(int), region.shape[1] - 1)
        region = region[np.ix_(rows, cols)] if region.size else np.zeros((out_h, out_w), np.uint32)
    region = np.ascontiguousarray(region, dtype=np.uint32)
    _LABEL_TILES.put(key, region)
    return region


def outline_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels on the edge of a cell (next to a different cell or background)."""
    mask = np.zeros(labels.shape, dtype=bool)
    if labels.size == 0:
        return mask
    mask[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    mask[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    mask[1:, :] |= labels[1:, :] != labels[:-1, :]
    mask[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return mask & (labels > 0)


def cell_outline(pyramid: str | Path, label: int, x: float, y: float, *, window: int = 64,
                 largest: int = 2048) -> list[list[list[float]]]:
    """The boundary of cell *label* near (x, y), traced on the full-resolution label image.

    Used to highlight a clicked cell. A small window around the point is read
    and grown while the cell touches its edge. Returns up to three rings of
    [x, y] points in image pixels, largest first; empty when the cell is not there.
    """
    from skimage import measure

    source = open_source(pyramid)
    height, width = source.info.height, source.info.width
    half = int(window)
    while True:
        x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
        x1, y1 = min(width, int(x) + half), min(height, int(y) + half)
        if x1 <= x0 or y1 <= y0:
            return []
        mask = np.asarray(source.read_region(0, 0, x0, y0, x1, y1)) == int(label)
        if not mask.any():
            return []
        touches = (mask[0].any() and y0 > 0) or (mask[-1].any() and y1 < height) or \
            (mask[:, 0].any() and x0 > 0) or (mask[:, -1].any() and x1 < width)
        if not touches or 2 * half >= largest:
            break
        half *= 2
    contours = measure.find_contours(np.pad(mask, 1).astype(np.uint8), 0.5)
    # Padded index p is pixel p - 1, whose centre is at p - 0.5 in image pixels.
    rings = [[[round(float(c) - 0.5 + x0, 1), round(float(r) - 0.5 + y0, 1)] for r, c in contour] for contour in contours]
    return sorted(rings, key=len, reverse=True)[:3]


def render_cell_tile(
    labels: np.ndarray,
    lut: np.ndarray | None,
    *,
    mode: str = "outline",
    outline_rgba: tuple[int, int, int, int] = (255, 255, 0, 255),
    z: int = 0,
) -> np.ndarray:
    """RGBA tile. *lut* maps cell number -> RGBA; None draws outlines only.

    Zoomed out beyond 2x, a cell is a few pixels wide and its outline would hide
    its colour, so cells are drawn filled; outlines appear as you zoom in.
    """
    height, width = labels.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if labels.size == 0:
        return rgba
    if z >= 2:
        if lut is None:
            rgba[labels > 0] = outline_rgba
            return rgba
        mode = "fill"
    inside = labels > 0
    if lut is not None and mode in ("fill", "both"):
        safe = np.where(labels < len(lut), labels, 0)
        rgba[inside] = lut[safe[inside]]
    if mode in ("outline", "both") or lut is None:
        edges = outline_mask(labels)
        if lut is not None and mode == "outline":
            safe = np.where(labels < len(lut), labels, 0)
            colours = lut[safe[edges]].copy()
            colours[:, 3] = np.where(colours[:, 3] > 0, 255, 0)
            rgba[edges] = colours
        else:
            rgba[edges] = outline_rgba
    return rgba


# ---------------------------------------------------------------------------
# Per-cell tables: row index, column values, assignments, cell details
# ---------------------------------------------------------------------------

_INDEX = _Cache(4)
_VALUES = _Cache(64)
_KEY_LOCKS: dict = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _key_lock(key) -> threading.Lock:
    """One lock per cache key: the browser asks for many tiles at once, and the
    first one reads the table while the others wait for its result."""
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            if len(_KEY_LOCKS) > 512:
                _KEY_LOCKS.clear()
            lock = _KEY_LOCKS[key] = threading.Lock()
        return lock


def _edits_token(table_path: str) -> tuple:
    """Changes whenever a column or label is saved beside the table."""
    from spacemap import labels as label_store
    from spacemap import pending_columns

    stamps = []
    for candidate in (pending_columns.store_dir(table_path), label_store.sidecar_path(table_path)):
        try:
            stamps.append(Path(candidate).stat().st_mtime_ns)
        except OSError:
            stamps.append(0)
    return tuple(stamps)


def roi_index(
    table_path: str, roi: str, *, roi_column: str = "batch_id", label_column: str = "label"
) -> tuple[np.ndarray, np.ndarray]:
    """(row positions, cell labels) of one ROI's cells, in table order."""
    import pandas as pd

    from spacemap import dataio

    key = (table_path, fingerprint(table_path), roi_column, label_column)
    roi_key = key + (str(roi),)
    cached = _INDEX.get(roi_key)
    if cached is not None:
        return cached
    index = _INDEX.get(key)
    if index is None:
        with _key_lock(("index",) + key):
            cached = _INDEX.get(roi_key)
            if cached is not None:
                return cached
            index = _INDEX.get(key)
            if index is None:
                # Opening one ROI must not convert/group millions of unrelated
                # rows in Python: that holds the GIL and delays image tiles.
                selected = _parquet_roi_index(table_path, str(roi), roi_column, label_column)
                if selected is not None:
                    _INDEX.put(roi_key, selected)
                    return selected
                frame = dataio.load(table_path, columns=[roi_column, label_column], apply_labels=False)
                labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy()
                groups = frame.groupby(frame[roi_column].astype(str), sort=False).indices
                index = {}
                for name, positions in groups.items():
                    positions = np.asarray(positions, dtype=np.int64)
                    keep = np.isfinite(labels[positions])
                    index[str(name)] = (positions[keep], labels[positions[keep]].astype(np.int64))
                _INDEX.put(key, index)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    return index.get(str(roi), empty)


def _parquet_roi_index(table_path: str, roi: str, roi_column: str, label_column: str):
    """Read only matching ROI rows from a current fast copy, retaining CSV positions.

    Row-group statistics skip unrelated groups; mixed groups are filtered in
    Arrow before conversion to Python. Numeric ROI names use the existing
    pandas path to preserve its exact string conversion rules.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from spacemap import dataio

    info = dataio.fast_copy_info(table_path)
    if not info.get("enabled") or not info.get("current"):
        return None
    try:
        parquet = pq.ParquetFile(info["twin"])
        field = parquet.schema_arrow.field(roi_column)
        dtype = field.type.value_type if pa.types.is_dictionary(field.type) else field.type
        if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
            return None
        column_index = parquet.schema.names.index(roi_column)
        positions, labels, offset = [], [], 0
        for number in range(parquet.num_row_groups):
            group = parquet.metadata.row_group(number)
            stats = group.column(column_index).statistics
            # Preserve the existing reader's spelling of missing entries.
            if roi in {"nan", "None", "<NA>"}:
                return None
            if stats and stats.has_min_max and not (stats.min <= roi <= stats.max):
                offset += group.num_rows
                continue
            frame = parquet.read_row_group(number, columns=[roi_column, label_column])
            names = frame.column(roi_column).cast(pa.string())
            within = pc.indices_nonzero(pc.fill_null(pc.equal(names, roi), False)).to_numpy()
            if within.size:
                raw = frame.column(label_column).take(pa.array(within)).to_numpy(zero_copy_only=False)
                numbers = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy()
                keep = np.isfinite(numbers)
                positions.append(within[keep].astype(np.int64) + offset)
                labels.append(numbers[keep].astype(np.int64))
            offset += group.num_rows
        return (
            (np.concatenate(positions), np.concatenate(labels))
            if positions
            else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        )
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException):
        return None  # unreadable copy: retain the normal table reader's fallback


def fast_rows(table_path: str, positions: np.ndarray, columns: list[str]):
    """Rows *positions* (table order) of *columns* from the table's fast copy, or None.

    Reads only the Parquet row groups that hold those rows. Filtering the copy
    by ROI name instead reads the columns for every cell of every ROI: for
    PDAC (8.6 million cells) opening live RESOLVE on one ROI took over a minute.
    The fast copy keeps the table's row order.
    """
    from spacemap import dataio

    positions = np.asarray(positions, dtype=np.int64)
    info = dataio.fast_copy_info(table_path)
    if not positions.size or not info.get("current"):
        return None
    return parquet_rows(info["twin"], positions, columns)


def parquet_rows(path, positions, columns):
    """Selected rows from a Parquet table or pending full-length column."""
    positions = np.asarray(positions, dtype=np.int64)
    if not positions.size:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        parquet = pq.ParquetFile(path)
        sizes = np.array([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
        starts = np.concatenate([[0], np.cumsum(sizes)])
        if positions.max() >= starts[-1]:
            return None
        group_of = np.searchsorted(starts, positions, side="right") - 1
        groups = np.unique(group_of)
        table = parquet.read_row_groups(groups.tolist(), columns=list(columns))
        offsets = np.concatenate([[0], np.cumsum(sizes[groups])])
        local = offsets[np.searchsorted(groups, group_of)] + (positions - starts[group_of])
        if local.size and local[-1] - local[0] == local.size - 1 and np.all(np.diff(local) == 1):
            return table.slice(int(local[0]), int(local.size))  # one block of rows: no copy
        return table.take(pa.array(local))
    except Exception:
        return None


def roi_values(
    table_path: str, roi: str, column: str, *, roi_column: str = "batch_id", label_column: str = "label"
) -> tuple[np.ndarray, np.ndarray]:
    """(labels, values) of *column* for one ROI, including saved-but-unwritten edits."""
    key = (table_path, fingerprint(table_path), _edits_token(table_path), roi, column, roi_column, label_column)
    cached = _VALUES.get(key)
    if cached is not None:
        return cached
    with _key_lock(("values",) + key):
        cached = _VALUES.get(key)
        if cached is not None:
            return cached
        return _read_roi_values(table_path, roi, column, key, roi_column=roi_column, label_column=label_column)


def _read_roi_values(
    table_path: str, roi: str, column: str, key: tuple, *, roi_column: str, label_column: str
) -> tuple[np.ndarray, np.ndarray]:
    positions, labels = roi_index(table_path, roi, roi_column=roi_column, label_column=label_column)
    values = values_at_rows(table_path, positions, column)
    result = (labels, values)
    _VALUES.put(key, result)
    return result


def values_at_rows(table_path, positions, column):
    """Read only needed rows, then overlay full-column and per-cell edits."""
    from spacemap import dataio, labels as label_store, pending_columns

    positions = np.asarray(positions, dtype=np.int64)
    values = np.full(len(positions), None, dtype=object)
    if not len(positions):
        return values
    source = next((p for p in pending_columns.pending(table_path) if p["column"] == column and not p["stale"]), None)
    if source:
        table = parquet_rows(source["file"], positions, ["value"])
        if table is None:
            raise ValueError("Could not read the saved column. Please try again.")
        values = table.column(0).to_numpy(zero_copy_only=False).astype(object)
    elif column in dataio.columns(table_path, include_pending=False):
        table = fast_rows(table_path, positions, [column])
        if table is not None and table.num_columns:
            values = table.column(0).to_numpy(zero_copy_only=False).astype(object)
        else:
            values = (
                dataio.load(table_path, columns=[column], apply_labels=False)[column]
                .to_numpy()[positions]
                .astype(object)
            )
    edits = label_store.load(table_path)
    if not edits.empty:
        edits = edits[edits["column"] == column]
        order = np.argsort(positions)
        slots = np.searchsorted(positions[order], edits["row"].to_numpy()).clip(0, len(positions) - 1)
        match = positions[order][slots] == edits["row"].to_numpy()
        texts = edits["value"].to_numpy()[match]
        values[order[slots[match]]] = np.where(pd.isna(texts) | (texts == ""), None, texts)
    return values


_ASSIGN_INDEX = _Cache(8)


def assignment_rows(table_path, roi, wanted, **keys):
    """Match stable cell numbers once, without sorting the ROI for each click."""
    wanted = np.asarray(sorted({int(v) for v in wanted if int(v) > 0}), dtype=np.int64)
    if not len(wanted):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    key = (table_path, fingerprint(table_path), roi, tuple(sorted(keys.items())))
    cached = _ASSIGN_INDEX.get(key)
    if cached is None:
        positions, labels = roi_index(table_path, roi, **keys)
        order = np.argsort(labels)
        sorted_labels, sorted_rows = labels[order], positions[order]
        cached = sorted_labels, sorted_rows
        _ASSIGN_INDEX.put(key, cached)
    sorted_labels, sorted_rows = cached
    if not len(sorted_labels) or not len(wanted):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    left = np.searchsorted(sorted_labels, wanted)
    right = np.searchsorted(sorted_labels, wanted, side="right")
    if np.any(right - left > 1):
        raise ValueError("Some cell numbers are repeated in this ROI. Fix the cell table before assigning labels.")
    found = right > left
    return wanted[found], sorted_rows[left[found]]


def assign(
    table_path: str,
    roi: str,
    labels_to_assign,
    column: str,
    value,
    *,
    roi_column: str = "batch_id",
    label_column: str = "label",
) -> dict:
    """Give cells a value in *column* through the fast label store.

    Returns how many cells were assigned and each cell's previous value, so the
    viewer can undo. An empty *value* clears the cells.
    """
    from spacemap import labels as label_store

    with label_store.write_lock(table_path):
        wanted, rows = assignment_rows(
            table_path, roi, labels_to_assign, roi_column=roi_column, label_column=label_column
        )
        previous = {
            str(label): None if pd.isna(old) else str(old)
            for label, old in zip(wanted, values_at_rows(table_path, rows, column))
        }
        text = "" if value is None else str(value)
        label_store.append(table_path, [{"row": int(r), "column": column, "value": text} for r in rows])
        return {"assigned": int(len(rows)), "previous": previous}


def assign_many(table_path: str, roi: str, values_by_label: dict, column: str, **keys) -> int:
    """Undo support: give each cell its own value."""
    count = 0
    grouped: dict[str, list[int]] = {}
    for label, value in values_by_label.items():
        grouped.setdefault("" if value is None else str(value), []).append(int(label))
    for value, labels in grouped.items():
        count += assign(table_path, roi, labels, column, value, **keys)["assigned"]
    return count


def cell_details(
    table_path: str, roi: str, label: int, *, roi_column: str = "batch_id", label_column: str = "label"
) -> dict:
    """Every column of one cell's row, including saved-but-unwritten edits."""
    from spacemap import dataio
    from spacemap import labels as label_store
    from spacemap import pending_columns

    positions, labels = roi_index(table_path, roi, roi_column=roi_column, label_column=label_column)
    match = np.flatnonzero(labels == int(label))
    if match.size == 0:
        return {}
    row = int(positions[match[0]])
    record: dict = {}
    info = dataio.fast_copy_info(table_path)
    if info.get("current"):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(info["twin"])
        start = 0
        for group in range(parquet.num_row_groups):
            size = parquet.metadata.row_group(group).num_rows
            if row < start + size:
                record = parquet.read_row_group(group).slice(row - start, 1).to_pylist()[0]
                break
            start += size
    else:
        # Without a current fast copy a single row would need the whole CSV
        # parsed; prepare the copy and report what is known.
        dataio.start_fast_copy(table_path)
        record = {"label": int(label), "_note": "Cell details appear once the fast copy is ready."}
    for item in pending_columns.pending(table_path):
        if not item["stale"]:
            record[item["column"]] = _value_at(item["file"], row)
    edits = label_store.load(table_path)
    if not edits.empty:
        for _, edit in edits[edits["row"] == row].iterrows():
            record[str(edit["column"])] = edit["value"] or None
    return {key: _jsonable(value) for key, value in record.items()}


def _value_at(path: str, row: int):
    """One value of a full-length pending-column file."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    start = 0
    for group in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if row < start + size:
            return parquet.read_row_group(group, columns=["value"]).column(0)[row - start].as_py()
        start += size
    return None


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def categorical_lut(
    labels: np.ndarray,
    values,
    *,
    colors: dict[str, str] | None = None,
    hidden: set[str] | None = None,
    alpha: int = 255,
) -> tuple[np.ndarray, list[dict]]:
    """Colour table and legend for a categorical column (phenotypes, clusters, neighbourhoods)."""
    import pandas as pd

    from spacemap.acc_utils import build_color_mapping

    text = pd.Series(values).astype(object).where(pd.notna(values), None)
    text = text.map(lambda v: None if v is None or str(v).strip() in ("", "nan", "None") else str(v).strip())
    counts = text.value_counts()
    categories = list(counts.index)
    mapping = build_color_mapping(sorted(categories, key=str.casefold), existing=dict(colors or {}))
    hidden = hidden or set()
    size = int(labels.max()) + 1 if labels.size else 1
    lut = np.zeros((size, 4), dtype=np.uint8)
    rgba_by_category = {}
    for category in categories:
        hex_color = mapping.get(category, "#888888").lstrip("#")
        rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
        rgba_by_category[category] = rgb + [0 if category in hidden else alpha]
    codes = text.map(lambda v: v if v is not None else "\0")
    for category, rgba in rgba_by_category.items():
        lut[labels[(codes == category).to_numpy()]] = rgba
    legend = [
        {
            "value": category,
            "color": f"#{mapping.get(category, '#888888').lstrip('#')}",
            "count": int(counts[category]),
            "hidden": category in hidden,
        }
        for category in categories
    ]
    return lut, legend


def numeric_lut(
    labels: np.ndarray,
    values,
    *,
    colormap: str = "viridis",
    alpha: int = 255,
    low: float | None = None,
    high: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Colour table for a numeric column (a marker's intensity, a distance)."""
    import pandas as pd
    from matplotlib import colormaps

    numbers = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = numbers[np.isfinite(numbers)]
    lo = float(np.percentile(finite, 1)) if low is None and finite.size else float(low or 0.0)
    hi = float(np.percentile(finite, 99)) if high is None and finite.size else float(high or 1.0)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((numbers - lo) / (hi - lo), 0, 1)
    colours = (colormaps[colormap](np.nan_to_num(scaled))[:, :3] * 255).astype(np.uint8)
    size = int(labels.max()) + 1 if labels.size else 1
    lut = np.zeros((size, 4), dtype=np.uint8)
    valid = np.isfinite(numbers)
    lut[labels[valid], :3] = colours[valid]
    lut[labels[valid], 3] = alpha
    return lut, {"min": lo, "max": hi, "colormap": colormap}


def is_numeric(values) -> bool:
    import pandas as pd

    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) > 24
    return False


# ---------------------------------------------------------------------------
# Density maps (as Tab 9's virtual tissue: binned centroids, Gaussian blur)
# ---------------------------------------------------------------------------

_DENSITY = _Cache(32)
_GRID_BINS = 768  # bins along the image's long side
CENTROID_COLUMNS = (
    ("x_centroid", "y_centroid"),
    ("X_centroid", "Y_centroid"),
    ("centroid_x", "centroid_y"),
    ("x", "y"),
)


def centroid_columns(table_path: str) -> tuple[str, str] | None:
    from spacemap import dataio

    available = set(dataio.columns(table_path))
    for x_col, y_col in CENTROID_COLUMNS:
        if x_col in available and y_col in available:
            return x_col, y_col
    return None


#: Tables of positions that are not cells (Tab 8's triad instances): x, y columns.
POINT_COLUMNS = (("triad_centroid_x", "triad_centroid_y"),)
_POINT_FRAMES = _Cache(4)
_POINT_ROI_COLUMNS = ("batch_id", "roi", "batch", "sample", "roi_id")


def point_columns(table_path: str) -> tuple[str, str] | None:
    from spacemap import dataio

    available = set(dataio.columns(table_path))
    for x_col, y_col in POINT_COLUMNS:
        if x_col in available and y_col in available:
            return x_col, y_col
    return None


def roi_points(table_path: str, roi: str, columns: list[str]) -> dict[str, np.ndarray]:
    """Rows of a point table (triads) that belong to *roi*, matched on its batch or ROI column."""
    key = (table_path, fingerprint(table_path))
    frame = _POINT_FRAMES.get(key)
    if frame is None:
        frame = pd.read_csv(table_path, low_memory=False)
        _POINT_FRAMES.put(key, frame)
    mask = np.zeros(len(frame), dtype=bool)
    for column in _POINT_ROI_COLUMNS:
        if column in frame.columns:
            mask |= (frame[column].astype(str) == str(roi)).to_numpy()
    rows = frame.loc[mask]
    return {column: rows[column].to_numpy() for column in columns if column in rows.columns}


def group_counts(table_path: str, roi: str, column: str, **keys) -> list[tuple[str, int]]:
    """Each value of *column* in the ROI and how often it occurs (cells or triads), commonest first."""
    if point_columns(table_path):
        values = roi_points(table_path, roi, [column]).get(column, np.array([]))
    else:
        values = roi_values(table_path, roi, column, **keys)[1]
    text = pd.Series(values).dropna().astype(str).str.strip()
    counts = text[text != ""].value_counts()
    return [(str(value), int(count)) for value, count in counts.items()]


def density_grid(
    table_path: str,
    roi: str,
    image: ImageInfo,
    *,
    filter_column: str = "",
    filter_values: tuple = (),
    value_column: str = "",
    radius_um: float = 50.0,
    weighted: bool = False,
    roi_column: str = "batch_id",
    label_column: str = "label",
) -> dict:
    """A smoothed map of the ROI's cells, cached.

    Without *value_column*: cells per mm^2 (of the cells whose *filter_column*
    is one of *filter_values*, or of all cells). With it: the local average of
    that column, optionally weighted by how dense the cells are, as in Tab 9's
    virtual tissue. Returns the grid (NaN where there are no cells), the bin
    size in image pixels, the colour range and the units.
    """
    from scipy.ndimage import gaussian_filter

    points = point_columns(table_path)
    columns = points or centroid_columns(table_path)
    if columns is None:
        raise ValueError("The table has no position columns (x_centroid, y_centroid).")
    keys = dict(roi_column=roi_column, label_column=label_column)
    stamp = (
        table_path,
        fingerprint(table_path),
        _edits_token(table_path),
        roi,
        image.fingerprint,
        filter_column,
        tuple(filter_values),
        value_column,
        float(radius_um),
        bool(weighted),
    )
    cached = _DENSITY.get(stamp)
    if cached is not None:
        return cached
    if points:
        rows = roi_points(table_path, roi, [columns[0], columns[1]] + [c for c in (filter_column, value_column) if c])

        def read(column):
            return rows.get(column, np.array([]))
    else:

        def read(column):
            return roi_values(table_path, roi, column, **keys)[1]

    xs = np.asarray(pd.to_numeric(pd.Series(read(columns[0])), errors="coerce"), dtype=float)
    ys = np.asarray(pd.to_numeric(pd.Series(read(columns[1])), errors="coerce"), dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys)
    if filter_column and filter_values:
        groups = read(filter_column)
        wanted = {str(v) for v in filter_values}
        keep &= np.array([str(v).strip() in wanted for v in groups], dtype=bool)
    weights = None
    if value_column:
        weights = np.asarray(pd.to_numeric(pd.Series(read(value_column)), errors="coerce"), dtype=float)
        keep &= np.isfinite(weights)
    bin_px = max(image.width, image.height) / _GRID_BINS
    nx, ny = max(1, int(math.ceil(image.width / bin_px))), max(1, int(math.ceil(image.height / bin_px)))
    extent = [[0, ny * bin_px], [0, nx * bin_px]]
    counts, _, _ = np.histogram2d(ys[keep], xs[keep], bins=[ny, nx], range=extent)
    pixel_um = image.pixel_size_um or 1.0
    sigma = max(float(radius_um) / pixel_um / bin_px, 0.5)
    count_blur = gaussian_filter(counts, sigma=sigma, mode="constant")
    if weights is None:
        bin_mm2 = (bin_px * pixel_um / 1000.0) ** 2
        grid = count_blur / bin_mm2
        things = "triads" if points and "triad" in Path(table_path).name else ("points" if points else "cells")
        units = f"{things}/mm²" if image.pixel_size_um else f"{things} per bin"
    else:
        sums, _, _ = np.histogram2d(ys[keep], xs[keep], bins=[ny, nx], range=extent, weights=weights[keep])
        value_blur = gaussian_filter(sums, sigma=sigma, mode="constant")
        grid = np.divide(value_blur, count_blur, out=np.zeros_like(value_blur), where=count_blur > 1e-9)
        if weighted and count_blur.max() > 0:
            grid = grid * (count_blur / count_blur.max())
        units = value_column
    grid = grid.astype(np.float32)
    grid[count_blur < 1e-3 * max(float(count_blur.max()), 1e-12)] = np.nan
    finite = grid[np.isfinite(grid)]
    if finite.size:
        low, high = float(np.percentile(finite, 1)), float(np.percentile(finite, 99.5))
    else:
        low, high = 0.0, 1.0
    if weights is None:
        low = 0.0
    if high <= low:
        high = low + 1.0
    result = {
        "grid": grid,
        "bin_px": bin_px,
        "min": low,
        "max": high,
        "units": units,
        "kind": "density" if weights is None else "average",
        "cells": int(keep.sum()),
        "radius_um": float(radius_um),
    }
    _DENSITY.put(stamp, result)
    return result


_COLOUR_TABLES: dict[str, np.ndarray] = {}


def _colour_table(name: str) -> np.ndarray:
    """256 RGBA colours of a matplotlib colour map, built once."""
    if name not in _COLOUR_TABLES:
        from matplotlib import colormaps

        try:
            cmap = colormaps[name]
        except KeyError:
            cmap = colormaps["inferno"]
        _COLOUR_TABLES[name] = (cmap(np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)
    return _COLOUR_TABLES[name]


def render_density_tile(
    density: dict, image: ImageInfo, z: int, x: int, y: int, *, colormap: str = "inferno"
) -> np.ndarray:
    """RGBA tile of a density grid: colour by value, fading out where there are few cells."""
    import cv2

    x0, y0, x1, y1, out_w, out_h = tile_bounds(image, z, x, y)
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    if out_w == 0 or out_h == 0:
        return rgba
    scale = 2**z
    bin_px = density["bin_px"]
    cols = ((x0 + (np.arange(out_w) + 0.5) * scale) / bin_px - 0.5).astype(np.float32)
    rows = ((y0 + (np.arange(out_h) + 0.5) * scale) / bin_px - 0.5).astype(np.float32)
    map_x = np.broadcast_to(cols[None, :], (out_h, out_w)).copy()
    map_y = np.broadcast_to(rows[:, None], (out_h, out_w)).copy()
    grid = density["grid"]
    if "_filled" not in density:
        density["_filled"] = np.nan_to_num(grid, nan=0.0).astype(np.float32)
        density["_present"] = np.isfinite(grid).astype(np.float32)
    values = cv2.remap(
        density["_filled"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    present = cv2.remap(
        density["_present"], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    norm = np.clip((values - density["min"]) / (density["max"] - density["min"]), 0.0, 1.0)
    colour = str(colormap).lstrip("#")
    if len(colour) == 6 and all(c in "0123456789abcdefABCDEF" for c in colour):
        # One colour, from clear to solid: several maps can be laid over each other
        # and still show where each group is and where they overlap.
        rgba[..., :3] = [int(colour[i : i + 2], 16) for i in (0, 2, 4)]
        rgba[..., 3] = (np.power(norm, 0.7) * present * 235).astype(np.uint8)
        return rgba
    rgba[...] = _colour_table(colormap)[(norm * 255).astype(np.uint8)]
    fade = np.clip(norm * 4.0, 0.0, 1.0) if density.get("kind") == "density" else 1.0  # few cells fade out
    rgba[..., 3] = (fade * present * 235).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Marker ranking for one cell (as RESOLVE scores markers)
# ---------------------------------------------------------------------------

_MATRICES = _Cache(8)
ARCSINH_COFACTOR = 5.0  # RESOLVE's default


def roi_matrix(
    table_path: str, roi: str, columns: list[str], *, roi_column: str = "batch_id", label_column: str = "label"
) -> tuple:
    """(labels, values [cells x columns], columns found, Arcsinh means, Arcsinh SDs) for one ROI.

    Read in one pass from the fast copy and cached, so ranking a clicked cell's
    markers is only arithmetic on one row.
    """
    from spacemap import dataio

    columns = [c for c in dict.fromkeys(columns) if c]
    key = (table_path, fingerprint(table_path), _edits_token(table_path), roi, tuple(columns))
    cached = _MATRICES.get(key)
    if cached is not None:
        return cached
    with _key_lock(("matrix",) + key):
        cached = _MATRICES.get(key)
        if cached is not None:
            return cached
        positions, labels = roi_index(table_path, roi, roi_column=roi_column, label_column=label_column)
        available = set(dataio.columns(table_path))
        found = [c for c in columns if c in available]
        values = None
        table = fast_rows(table_path, positions, found) if found else None
        if table is not None:
            values = np.column_stack(
                [
                    pd.to_numeric(pd.Series(table.column(c).to_numpy(zero_copy_only=False)), errors="coerce").to_numpy(
                        dtype=np.float32
                    )
                    for c in found
                ]
            )
        if values is None:
            values = (
                np.column_stack(
                    [
                        pd.to_numeric(
                            pd.Series(
                                roi_values(table_path, roi, c, roi_column=roi_column, label_column=label_column)[1]
                            ),
                            errors="coerce",
                        ).to_numpy(dtype=np.float32)
                        for c in found
                    ]
                )
                if found
                else np.zeros((len(labels), 0), np.float32)
            )
        transformed = np.arcsinh(values / ARCSINH_COFACTOR)
        means = np.nanmean(transformed, axis=0) if len(transformed) else np.zeros(len(found))
        stds = np.nanstd(transformed, axis=0) if len(transformed) else np.ones(len(found))
        stds[~np.isfinite(stds) | (stds == 0)] = 1e-9
        result = (labels, values, found, means, stds)
        _MATRICES.put(key, result)
        return result


def marker_ranking(table_path: str, roi: str, label: int, markers: list[str], **keys) -> list[dict]:
    """Each marker of one cell as raw, Arcsinh, Z1 and Z2 values, highest Z1 first.

    As RESOLVE with batches grouped by ROI: Arcsinh is ``arcsinh(raw / 5)``;
    Z1 is the marker's z-score among this ROI's cells (how unusually high the
    cell is for that marker); Z2 is the z-score of the cell's Z1 values across
    its own markers (which markers stand out within the cell).
    """
    labels, values, found, means, stds = roi_matrix(table_path, roi, markers, **keys)
    rows = np.flatnonzero(labels == int(label))
    if not rows.size or not found:
        return []
    raw = values[rows[0]]
    transformed = np.arcsinh(raw / ARCSINH_COFACTOR)
    z1 = (transformed - means) / stds
    finite = np.isfinite(z1)
    spread = float(np.nanstd(z1[finite])) if finite.sum() > 1 else 0.0
    z2 = (z1 - float(np.nanmean(z1[finite]))) / spread if spread > 0 else np.zeros_like(z1)
    ranking = [
        {"marker": m, "raw": float(r), "asinh": float(a), "z1": float(a1), "z2": float(a2)}
        for m, r, a, a1, a2 in zip(found, raw, transformed, z1, z2)
        if np.isfinite(r)
    ]
    ranking.sort(key=lambda item: item["z1"], reverse=True)
    return ranking
