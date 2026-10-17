"""Live RESOLVE in the viewer: one engine per ROI, kept in memory while thresholds are adjusted.

A session reads the ROI's raw marker values once (full precision, so phenotypes
equal Tab 7's), starts from RESOLVE's saved threshold table (or a suggested
level where none is saved), and then answers each threshold change with new
phenotypes, counts and cell colours. Thresholds are saved back into the same
table Tab 7 reads, for this ROI only.

The spec a session is opened from is registered by the interface (never sent
by the browser): the table, the ROI, the panel file, the threshold table, the
RESOLVE settings file and the phenotype colours.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.types as pa_types

from spacemap.resolve_live import LiveResolve, Panel, Result, Settings

#: Columns of RESOLVE's threshold table that are not markers.
TABLE_INFO_COLUMNS = ("batch_id", "Threshold_File", "Threshold_Source", "Threshold_Aggregation", "Polygon_Count")
SAVED_SOURCE = "Viewer (live RESOLVE)"
_MAX_SESSIONS = 3  # the largest PDAC ROI needs about 0.7 GB


@dataclass
class Session:
    spec: dict
    roi: str
    markers: list[str]
    labels: np.ndarray  # cell label of each ROI row (table order)
    raw: np.ndarray  # raw marker values, ROI rows x markers
    engine: LiveResolve
    thresholds: dict[str, float]
    source: dict[str, str]  # per marker: "saved" or "suggested"
    result: Result | None = None
    version: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_ms: float = 0.0
    histogram_transforms: dict = field(default_factory=dict)
    histogram_views: dict = field(default_factory=dict)
    panel_stamp: tuple = ()
    panel_checked_at: float = 0.0


_SESSIONS: OrderedDict[str, Session] = OrderedDict()
_LOCK = threading.Lock()


def _key(spec: dict) -> str:
    return json.dumps({k: spec.get(k) for k in ("table", "roi", "panel", "thresholds", "settings")}, sort_keys=True)


def read_panel(path: str | Path) -> Panel:
    return Panel.from_payload(json.loads(Path(path).read_text()))


def read_settings(path: str | Path | None) -> Settings:
    try:
        return Settings.from_config(json.loads(Path(path).read_text())) if path else Settings()
    except (OSError, ValueError):
        return Settings()


def table_markers(thresholds_path: str | Path | None) -> list[str]:
    """The markers of RESOLVE's threshold table (Tab 7's selected markers), in table order."""
    try:
        columns = pd.read_csv(thresholds_path, nrows=0).columns if thresholds_path else []
    except (OSError, ValueError):
        return []
    return [c for c in columns if c not in TABLE_INFO_COLUMNS]


def saved_thresholds(thresholds_path: str | Path | None, roi: str) -> dict[str, float]:
    try:
        table = pd.read_csv(thresholds_path) if thresholds_path else None
    except (OSError, ValueError):
        return {}
    if table is None or "batch_id" not in table.columns:
        return {}
    row = table[table["batch_id"].astype(str) == str(roi)]
    if row.empty:
        return {}
    values = {}
    for column in table.columns:
        if column in TABLE_INFO_COLUMNS:
            continue
        value = pd.to_numeric(row[column].iloc[0], errors="coerce")
        if pd.notna(value):
            values[column] = float(value)
    return values


def suggest_threshold(values: np.ndarray) -> float:
    """A raw-cell Otsu starting gate, using the same engine as the other panels."""
    from spacemap.thresholding_engine import calculate
    return calculate(values, "Otsu")["threshold"]


def _read_raw(table: str, roi: str, markers: list[str], roi_column: str, label_column: str):
    """(labels, raw values float64) for one ROI, in table order."""
    from spacemap import dataio
    from spacemap.viewer.cells import fast_rows, roi_index

    positions, labels = roi_index(table, roi, roi_column=roi_column, label_column=label_column)
    found = fast_rows(table, positions, markers)
    if found is not None:
        # Filled by column into column order, which the engine keeps (no further copies).
        values = np.empty((found.num_rows, len(markers)), dtype=float, order="F")
        for i, marker in enumerate(markers):
            column = found.column(marker)
            values[:, i] = (
                column.to_numpy(zero_copy_only=False)
                if pa_types.is_floating(column.type) or pa_types.is_integer(column.type)
                else pd.to_numeric(pd.Series(column.to_numpy(zero_copy_only=False)), errors="coerce")
            )
    else:
        frame = dataio.load(table, columns=markers)
        values = frame[markers].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[positions]
    return labels, values


def open_session(spec: dict) -> Session:
    """The session for *spec*, created (reading the ROI) the first time."""
    key = _key(spec)
    with _LOCK:
        session = _SESSIONS.get(key)
        if session is not None and time.monotonic() - session.panel_checked_at < 2:
            _SESSIONS.move_to_end(key)
            return session
    stat = Path(spec["panel"]).stat()
    settings_path = (
        Path(spec["settings"]) if spec.get("settings") else Path(spec["panel"]).parent / "resolve_params.json"
    )
    settings_stat = settings_path.stat() if settings_path.exists() else None
    stamp = (
        stat.st_mtime_ns,
        stat.st_size,
        (settings_stat.st_mtime_ns, settings_stat.st_size) if settings_stat else None,
    )
    with _LOCK:
        if session is not None and session.panel_stamp == stamp:
            session.panel_checked_at = time.monotonic()
            _SESSIONS.move_to_end(key)
            return session
    panel = read_panel(spec["panel"])
    settings = read_settings(settings_path if settings_path.exists() else None)
    available = set(_columns(spec["table"]))
    wanted = list(
        dict.fromkeys(
            table_markers(spec.get("thresholds"))
            + sorted(
                {m for groups in panel.definitions.values() for clause in groups for m in clause}
                | {m for markers in panel.supportive.values() for m in markers}
                | {m for markers in panel.exclusions.values() for m in markers}
            )
        )
    )
    markers = [m for m in wanted if m in available]
    if not markers:
        raise ValueError("None of the panel's markers is a column of the cell table.")
    labels, raw = _read_raw(
        spec["table"], spec["roi"], markers, spec.get("roi_column", "batch_id"), spec.get("label_column", "label")
    )
    saved = saved_thresholds(spec.get("thresholds"), spec["roi"])
    # Preserve this ROI's unsaved threshold edits when its shared panel changes.
    previous_source = {}
    if session is not None:
        with session.lock:
            saved.update(session.thresholds)
            previous_source = dict(session.source)
    thresholds, source = {}, {}
    for i, marker in enumerate(markers):
        if marker in saved:
            thresholds[marker], source[marker] = saved[marker], previous_source.get(marker, "saved")
        else:
            thresholds[marker], source[marker] = suggest_threshold(raw[:, i]), "suggested"
    previous_version = session.version if session is not None else 0
    session = Session(
        spec=spec,
        roi=spec["roi"],
        markers=markers,
        labels=labels,
        raw=raw,
        engine=LiveResolve(raw, markers, panel, settings),
        thresholds=thresholds,
        source=source,
        panel_stamp=stamp,
        panel_checked_at=time.monotonic(),
        version=previous_version,
    )
    assign(session, {})
    with _LOCK:
        _SESSIONS[key] = session
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
    return session


def _columns(table: str) -> list[str]:
    from spacemap import dataio

    return list(dataio.columns(table))


def invalidate_panel(path):
    """Recheck the shared panel on the next request without discarding raw edits."""
    with _LOCK:
        for session in _SESSIONS.values():
            if Path(session.spec["panel"]).resolve() == Path(path).resolve():
                session.panel_checked_at = 0.0


def invalidate_settings(path):
    """A saved Studio setting becomes visible on the next Live RESOLVE request."""
    target = Path(path).resolve()
    with _LOCK:
        for session in _SESSIONS.values():
            candidate = (
                Path(session.spec["settings"])
                if session.spec.get("settings")
                else Path(session.spec["panel"]).parent / "resolve_params.json"
            )
            if candidate.resolve() == target:
                session.panel_checked_at = 0.0


def assign(session: Session, changes: dict[str, float]) -> dict:
    """Apply threshold changes and re-run RESOLVE; returns counts and the new version."""
    with session.lock:
        checked = {}
        for marker, value in (changes or {}).items():
            if marker not in session.thresholds:
                raise ValueError(f"{marker} is not one of this panel's markers.")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("Thresholds must be finite raw values.")
            checked[marker] = value
        for marker, value in checked.items():
            session.thresholds[marker] = value
            session.source[marker] = "changed"
        started = time.perf_counter()
        session.result = session.engine.assign(session.thresholds)
        session.last_ms = (time.perf_counter() - started) * 1000
        session.version += 1
        return summary(session)


def summary(session: Session) -> dict:
    counts = dict(sorted(session.result.counts.items(), key=lambda item: -item[1])) if session.result else {}
    return {
        "version": session.version,
        "counts": counts,
        "cells": int(sum(counts.values())),
        "left_out": int((~session.engine.valid).sum()),
        "ms": round(session.last_ms, 1),
    }


def phenotype_lut(session: Session, colors: dict[str, str] | None = None, hidden: set[str] | None = None, near=False):
    """Colour table (cell label -> RGBA) for the current phenotypes, and its legend.

    Built from the engine's integer codes, so it keeps up with a dragged
    threshold on a million cells (mapping names through pandas took about a second).
    """
    from spacemap.acc_utils import build_color_mapping

    result, hidden = session.result, hidden or set()
    names = list(result.names)
    counts = result.counts
    mapping = build_color_mapping(sorted(counts, key=str.casefold), existing=dict(colors or {}))
    palette = np.zeros((len(names) + 1, 4), dtype=np.uint8)  # the extra row: cells left out
    for code, name in enumerate(names):
        colour = mapping.get(name, "#888888").lstrip("#")
        palette[code] = [int(colour[i : i + 2], 16) for i in (0, 2, 4)] + [0 if name in hidden else 255]
    row_codes = np.full(session.labels.shape[0], len(names), dtype=np.int64)
    row_codes[session.engine.valid] = result.codes
    lut = np.zeros((int(session.labels.max()) + 1 if session.labels.size else 1, 4), dtype=np.uint8)
    lut[session.labels] = palette[row_codes]
    if near:
        close = np.zeros(session.labels.size, dtype=bool)
        for i, marker in enumerate(session.markers):
            threshold = session.thresholds[marker]
            close |= np.abs(session.raw[:, i] - threshold) <= abs(threshold) * 0.1
        close &= session.engine.valid
        visible = lut[session.labels, 3] > 0
        lut[session.labels[close & visible]] = [255, 212, 0, 255]
        lut[session.labels[~close & visible], 3] = 50
    legend = [
        {
            "value": name,
            "color": "#" + mapping.get(name, "#888888").lstrip("#"),
            "count": int(count),
            "hidden": name in hidden,
        }
        for name, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
    return lut, legend


def histogram(session: Session, marker: str, bins: int = 64) -> dict:
    """The ROI's cells for one marker on an arcsinh(x/5) axis, with the threshold and positive count."""
    i = session.markers.index(marker)
    values = session.raw[:, i]
    values = values[np.isfinite(values)]
    scaled = np.arcsinh(np.clip(values, 0, None) / 5.0)
    top = float(np.percentile(scaled, 99.9)) if scaled.size else 1.0
    counts, edges = np.histogram(np.clip(scaled, 0, top), bins=bins, range=(0.0, max(top, 1e-6)))
    threshold = session.thresholds[marker]
    return {
        "marker": marker,
        "edges": (np.sinh(edges) * 5.0).round(3).tolist(),
        "counts": counts.tolist(),
        "threshold": threshold,
        "source": session.source.get(marker, ""),
        "positive": int((values > threshold).sum()),
        "total": int(values.size),
    }


def histogram_transform(session: Session, marker: str, names=(), cofactor=5.0):
    from .transforms import Transform

    key = (marker, tuple(names), float(cofactor))
    with session.lock:
        if key not in session.histogram_transforms:
            fitted = Transform.fit(session.raw[:, session.markers.index(marker)], names, cofactor)
            if len(session.histogram_transforms) >= 32:
                session.histogram_transforms.pop(next(iter(session.histogram_transforms)))
            session.histogram_transforms[key] = fitted
        return session.histogram_transforms[key]


def transformed_histogram(session: Session, marker: str, names=(), cofactor=5.0):
    from . import transforms

    transform = histogram_transform(session, marker, names, cofactor)
    key = (marker, tuple(names), float(cofactor))
    values = session.raw[:, session.markers.index(marker)]
    with session.lock:
        threshold = session.thresholds[marker]
        if key not in session.histogram_views:
            if len(session.histogram_views) >= 32:
                session.histogram_views.pop(next(iter(session.histogram_views)))
            session.histogram_views[key] = transforms.histogram(values, transform, threshold)
        return {
            **session.histogram_views[key],
            "threshold": threshold,
            "transformed_threshold": float(transform.forward(threshold)),
            "positive": int(np.count_nonzero(values > threshold)),
        }


def explain(session: Session, label: int) -> dict:
    """Why the cell with this label got its phenotype (see LiveResolve.explain)."""
    rows = np.flatnonzero(session.labels == int(label))
    if not rows.size:
        raise ValueError(f"Cell {label} is not in this ROI's table.")
    with session.lock:
        why = session.engine.explain(int(rows[0]), session.thresholds, session.result)
    why["label"] = int(label)
    return why


def save(session: Session) -> Path:
    """Write this ROI's thresholds into RESOLVE's threshold table (a backup is kept once per day)."""
    path = Path(session.spec["thresholds"])
    if path.is_file():
        backup = path.with_name(f"{path.name}.before-viewer-{time.strftime('%Y%m%d')}")
        if not backup.exists():
            shutil.copy2(path, backup)
        table = pd.read_csv(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(columns=["batch_id", *session.markers])
    row = {"batch_id": session.roi, **session.thresholds}
    if "Threshold_Source" in table.columns or not path.is_file():
        row["Threshold_Source"] = SAVED_SOURCE
    for column in row:
        if column not in table.columns:
            table[column] = np.nan
    match = table["batch_id"].astype(str) == str(session.roi)
    if match.any():
        for column, value in row.items():
            table.loc[match, column] = value
    else:
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)
    for marker in session.source:
        session.source[marker] = "saved"
    return path


def clear() -> None:
    with _LOCK:
        _SESSIONS.clear()
