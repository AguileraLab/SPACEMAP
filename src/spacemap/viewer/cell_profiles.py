"""On-demand cell measurements, with a small disk index of complete cell bounds.

The index is prepared in a child process only when image measurements are
requested. It reads one GeoJSON feature at a time, and includes all parts of a
cell. Measurements use full-resolution pixels labelled with that cell number,
so neither neighbouring cells nor holes in the outline contribute.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import numpy as np

from .sources import fingerprint, image_info, open_source
from spacemap.sharing import open_lock


class _JSONReader:
    """Incremental JSON values; memory is bounded by the largest feature."""

    def __init__(self, handle, chunk_size=65536):
        self.handle, self.chunk_size = handle, chunk_size
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def more(self):
        self.buffer = self.buffer[self.position :]
        self.position = 0
        chunk = self.handle.read(self.chunk_size)
        self.buffer += chunk
        self.eof = not chunk

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                raise ValueError("The cell outline file ends unexpectedly.")
            self.more()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError("The cell outline file is not valid GeoJSON.")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A number may continue across a read boundary, including a
                # decimal point or exponent that the decoder cannot consume yet.
                if not self.eof and (end == len(self.buffer) or self.buffer[end] not in " \t\r\n,]}:"):
                    self.more()
                    continue
                self.position = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("The cell outline file is not valid GeoJSON.") from None
                self.more()


def iter_features(handle, chunk_size=65536):
    reader = _JSONReader(handle, chunk_size)

    def array():
        reader.expect("[")
        if reader.peek() != "]":
            while True:
                yield reader.value()
                if reader.peek() == "]":
                    break
                reader.expect(",")
        reader.expect("]")

    if reader.peek() == "[":
        yield from array()
        return
    reader.expect("{")
    feature = {}
    if reader.peek() == "}":
        return
    while True:
        key = reader.value()
        reader.expect(":")
        if key == "features":
            yield from array()
        else:
            feature[key] = reader.value()
        if reader.peek() == "}":
            break
        reader.expect(",")
    if feature.get("type") == "Feature":
        yield feature


def _geometry_bounds(geometry):
    if not isinstance(geometry, dict):
        return None
    bounds = [math.inf, math.inf, -math.inf, -math.inf]

    def coordinates(node):
        if not isinstance(node, (list, tuple)) or not node:
            return
        if isinstance(node[0], (int, float)) and len(node) >= 2:
            x, y = float(node[0]), float(node[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("A cell outline has an invalid coordinate.")
            bounds[0], bounds[1] = min(bounds[0], x), min(bounds[1], y)
            bounds[2], bounds[3] = max(bounds[2], x), max(bounds[3], y)
        else:
            for child in node:
                coordinates(child)

    if geometry.get("type") == "GeometryCollection":
        for child in geometry.get("geometries", []):
            child_bounds = _geometry_bounds(child)
            if child_bounds:
                coordinates([child_bounds[:2], child_bounds[2:]])
    elif geometry.get("type") in ("Polygon", "MultiPolygon"):
        coordinates(geometry.get("coordinates"))
    return tuple(bounds) if math.isfinite(bounds[0]) else None


def _signature(source):
    return str(Path(source).resolve()) + "|" + fingerprint(source)


def index_path(source):
    from spacemap.userpaths import cache_dir

    key = hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[:24]
    return Path(cache_dir()) / "viewer" / "cell_bounds" / (key + ".sqlite")


@contextmanager
def _connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        yield db
    finally:
        db.close()


def index_current(source, target):
    if not Path(target).is_file():
        return False
    try:
        with _connect(target) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        return bool(row and row[0] == _signature(source))
    except (OSError, sqlite3.Error):
        return False


def build_index(source, target):
    """Build atomically, with one writer per source even across app processes."""
    import fcntl

    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open_lock(target.with_suffix(".lock")) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if index_current(source, target):
            return
        signature = _signature(source)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        db = sqlite3.connect(temporary)
        try:
            with db:
                db.execute("CREATE TABLE bounds (label INTEGER PRIMARY KEY, x0 REAL, y0 REAL, x1 REAL, y1 REAL)")
                db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                statement = (
                    "INSERT INTO bounds VALUES (?, ?, ?, ?, ?) ON CONFLICT(label) DO UPDATE SET "
                    "x0=min(x0,excluded.x0), y0=min(y0,excluded.y0), "
                    "x1=max(x1,excluded.x1), y1=max(y1,excluded.y1)"
                )
                batch = []
                with source.open() as handle:
                    for feature in iter_features(handle):
                        if not isinstance(feature, dict):
                            continue
                        props = feature.get("properties") or {}
                        label = props.get("label", props.get("cell_label"))
                        try:
                            number = int(label)
                            if number <= 0 or float(label) != number:
                                continue
                        except (ValueError, TypeError, OverflowError):
                            continue
                        bounds = _geometry_bounds(feature.get("geometry"))
                        if bounds:
                            batch.append((number, *bounds))
                        if len(batch) >= 5000:
                            db.executemany(statement, batch)
                            batch.clear()
                db.executemany(statement, batch)
                if signature != _signature(source):
                    raise ValueError("The cell outlines changed while preparing measurements. Try again.")
                db.execute("INSERT INTO metadata VALUES ('source', ?)", (signature,))
            db.close()
            os.replace(temporary, target)
        finally:
            db.close()
            temporary.unlink(missing_ok=True)


_JOBS = {}
_LOCK = threading.Lock()
_WORKERS = threading.Semaphore(1)


def prepare_index(source, *, retry=False):
    """Return (path, status) without waiting for a large outline file to parse."""
    target = index_path(source)
    if index_current(source, target):
        return target, {"status": "ready"}
    key = _signature(source)
    with _LOCK:
        job = _JOBS.get(key)
        if job and (job["status"] == "ready" or (retry and job["status"] == "error")):
            job = None
        if job is None:
            job = _JOBS[key] = {
                "status": "preparing",
                "message": "Preparing cell outlines for measurements (first time only)...",
            }

            def work():
                try:
                    with _WORKERS:
                        env = dict(os.environ)
                        env["PYTHONPATH"] = os.pathsep.join(
                            filter(None, [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")])
                        )
                        done = subprocess.run(
                            [sys.executable, "-m", __name__, str(source), str(target)],
                            env=env,
                            capture_output=True,
                            text=True,
                        )
                    if done.returncode or not index_current(source, target):
                        raise ValueError(
                            "Could not prepare the cell outlines. Check the outline file and available disk space."
                        )
                    with _LOCK:
                        job.update(status="ready", message="Cell outlines ready.")
                except Exception as exc:
                    with _LOCK:
                        job.update(status="error", message=str(exc))

            threading.Thread(target=work, name="spacemap-cell-profile-index", daemon=True).start()
        return target, dict(job)


_PROFILES = OrderedDict()
_PROFILE_LOCK = threading.Lock()


@lru_cache(maxsize=32)
def _mask_source(pyramid, stamp):
    from . import cells

    return cells._recorded_source(Path(pyramid))


def image_profile(image, pyramid, source, label, backgrounds, *, corrected=False, retry=False):
    """Exact full-resolution masked means, signal above background and coverage."""
    from . import cells

    if not cells.pyramid_ready(pyramid):
        return {"status": "preparing", "message": "The cell outlines are still being prepared."}
    meta = _mask_source(str(pyramid), fingerprint(pyramid))
    if (
        not meta
        or meta.get("fingerprint") != fingerprint(source)
        or Path(meta["source"]).resolve() != Path(source).resolve()
    ):
        raise ValueError("The displayed cell outlines are out of date. Reopen the image after they are rebuilt.")
    target, progress = prepare_index(source, retry=retry)
    if progress["status"] != "ready":
        return progress
    info, label_info = image_info(image), image_info(pyramid)
    if info.rgb or (info.width, info.height) != (label_info.width, label_info.height):
        raise ValueError("This marker image does not line up with the cell outlines.")
    if len(set(info.channel_names)) != len(info.channel_names):
        raise ValueError("This image has repeated marker names; choose an image with unique marker names.")
    markers = [(i, name) for i, name in enumerate(info.channel_names) if not name.upper().startswith("HE_")]
    levels = {}
    for _, name in markers:
        if corrected:
            levels[name] = 0.0
        elif name in backgrounds:
            value = float(backgrounds[name])
            if not math.isfinite(value):
                raise ValueError("Background levels must be finite numbers.")
            levels[name] = value
    key = (
        str(Path(image).resolve()),
        info.fingerprint,
        fingerprint(pyramid),
        _signature(source),
        int(label),
        tuple(sorted(levels.items())),
        corrected,
    )
    with _PROFILE_LOCK:
        cached = _PROFILES.get(key)
        if cached is not None:
            _PROFILES.move_to_end(key)
            return cached
    with _connect(target) as db:
        bounds = db.execute("SELECT x0,y0,x1,y1 FROM bounds WHERE label=?", (int(label),)).fetchone()
    if bounds is None:
        raise ValueError("No outline was found for this cell.")
    x0, y0 = max(0, math.floor(bounds[0])), max(0, math.floor(bounds[1]))
    x1, y1 = min(info.width, math.ceil(bounds[2]) + 1), min(info.height, math.ceil(bounds[3]) + 1)
    if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > 16_000_000:
        raise ValueError("This cell outline is too large or outside the image. Check its segmentation.")
    source_image, labels_image = open_source(image), open_source(pyramid)
    mask = labels_image.read_region(0, 0, x0, y0, x1, y1) == int(label)
    count = int(mask.sum())
    if not count:
        raise ValueError("This outline contains no pixels for the selected cell.")
    measurements = []
    for channel, marker in markers:
        pixels = source_image.read_region(0, channel, x0, y0, x1, y1)[mask].astype(np.float64)
        pixels = pixels[np.isfinite(pixels)]
        if not pixels.size:
            continue
        background = levels.get(marker)
        measurements.append(
            {
                "marker": marker,
                "mean": float(pixels.mean()),
                "background": background,
                "signal": float(np.maximum(pixels - background, 0).mean()) if background is not None else None,
                "positive_percent": float(100 * np.mean(pixels > background)) if background is not None else None,
                "valid_pixels": int(pixels.size),
            }
        )
    result = {
        "status": "ready",
        "source": "corrected" if corrected else "image",
        "label": int(label),
        "image": Path(image).name,
        "pixels": count,
        "resolution": "Full resolution",
        "markers": measurements,
        "missing_backgrounds": [name for _, name in markers if name not in levels],
    }
    with _PROFILE_LOCK:
        _PROFILES[key] = result
        while len(_PROFILES) > 256:
            _PROFILES.popitem(last=False)
    return result


if __name__ == "__main__":
    build_index(Path(sys.argv[1]), Path(sys.argv[2]))
