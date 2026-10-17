"""Studio's Batch correction step: how well Tab 6's correction worked, from its saved table.

Tab 6 writes one combined table holding each marker as measured and its
corrected value (``<marker>_corr``). This reads a sample of it from the fast
copy, up to a few hundred cells per batch, and reports Tab 6's own measures:
EMD reduction (how much closer the batches' distributions came; cyCombine's
1 - EMD after / EMD before) and the MAD score (how much each batch's spread
changed), overall and per marker, plus each batch's
median before and after for a heatmap. ``tests/test_studio_batches.py``
checks the measures equal Tab 6's ``calculate_emd_reduction`` and
``calculate_mad_score``.

These are pooled distribution diagnostics, not cyCombine's population-stratified
benchmark scores. They cannot establish biological preservation by themselves.
"""

from collections import OrderedDict
from itertools import combinations
from pathlib import Path
import threading
import time

import numpy as np

PER_BATCH = 300
HISTOGRAM_BINS = 40
_cache = OrderedDict()
_lock = threading.Lock()


def _tab6(config):
    return (config.get("tab_configurations") or {}).get("tab6_batch_correction") or {}


def settings(config):
    saved = _tab6(config)
    bc = saved.get("bc_config") if isinstance(saved.get("bc_config"), dict) else {}
    batch = (saved.get("bc_unified_batch_col") if saved.get("bc_input_mode") == "unified" else None) or bc.get(
        "batch_id_col", "batch_id"
    )
    unit = bc.get("batch_unit_col")
    return {
        "batch_column": str(batch or "batch_id"),
        # What the correction removed differences between (Tab 6's correction unit, the slide say).
        "unit_column": str(unit) if unit and unit != "(use batch column as-is)" else "",
        "method": str(bc.get("method", "Combat")),
        "suffix": str(bc.get("corrected_suffix") or "_corr"),
        "mode": str(bc.get("correction_mode", "cycombine")),
        "cofactor": float(bc.get("arcsinh_cofactor", 5.0) or 5.0),
    }


def _chunks(table, columns=None):
    """The table's names and its rows in chunks: from the fast copy, or a small CSV as it is."""
    import pandas as pd
    import pyarrow.parquet as pq

    from spacemap import dataio

    table = Path(table)
    if table.suffix.lower() == ".parquet":
        source = pq.ParquetFile(table)
        return source.schema_arrow.names, (chunk.to_pandas() for chunk in source.iter_batches(batch_size=200_000, columns=columns))
    if dataio.fast_copy_info(table).get("current"):
        source = pq.ParquetFile(dataio.fast_copy_path(table))
        names = source.schema_arrow.names
        return names, (chunk.to_pandas() for chunk in source.iter_batches(batch_size=200_000, columns=columns))
    if table.stat().st_size > dataio.FAST_COPY_MIN_BYTES:
        raise ValueError("Open the combined table in Studio first so its fast copy is ready.")
    names = list(pd.read_csv(table, nrows=0).columns)
    return names, pd.read_csv(table, usecols=columns, chunksize=200_000) if columns else iter(())


def sample(table, batch_column, extra=(), suffix="_corr"):
    """Up to PER_BATCH cells of every batch, with every marker and its corrected value (and the *extra* columns).

    Corrected columns (Tab 6's *suffix*) come back as ``<marker>_corr``.
    """
    import pandas as pd

    stamp = Path(table).stat().st_mtime_ns
    key = (str(table), stamp, batch_column, tuple(extra), suffix)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    started = time.perf_counter()
    names, _ = _chunks(table)
    if batch_column not in names:
        raise ValueError(f"The combined table has no '{batch_column}' column.")
    markers = [name[: -len(suffix)] for name in names if name.endswith(suffix) and name[: -len(suffix)] in names]
    if not markers:
        raise ValueError(f"The combined table has no corrected markers (<marker>{suffix}): run batch correction first.")
    batches = pd.concat(chunk[batch_column].astype(str) for chunk in _chunks(table, [batch_column])[1])
    counts = batches.value_counts()
    chance = (PER_BATCH / counts).clip(upper=1.0)
    rng = np.random.default_rng(42)
    parts = []
    extra = [name for name in extra if name in names and name != batch_column]
    for frame in _chunks(table, [batch_column, *extra, *markers, *[m + suffix for m in markers]])[1]:
        frame = frame.rename(columns={m + suffix: m + "_corr" for m in markers})
        frame[batch_column] = frame[batch_column].astype(str)
        keep = rng.random(len(frame)) < frame[batch_column].map(chance).to_numpy()
        parts.append(frame[keep])
    found = {
        "frame": pd.concat(parts, ignore_index=True),
        "markers": markers,
        "counts": counts,
        "cells": int(len(batches)),
        "seconds": round(time.perf_counter() - started, 1),
    }
    with _lock:
        _cache[key] = found
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return found


def measures(frame, markers, batch_column, cofactor):
    """Tab 6's EMD reduction and MAD score, overall and per marker, from per-batch values."""
    from scipy.stats import wasserstein_distance

    groups = {name: part for name, part in frame.groupby(batch_column, sort=True)}
    names = list(groups)
    per_marker, before_all, after_all, mad_all = [], [], [], []
    for marker in markers:
        pairs = {b: _paired(groups[b], marker) for b in names}
        raw = {b: pairs[b][marker] for b in names}
        corr = {b: pairs[b][marker + "_corr"] for b in names}
        before = {b: np.arcsinh(_numbers(raw[b]) / cofactor) for b in names if len(raw[b]) and len(corr[b])}
        after = {b: corr[b].to_numpy(dtype=float) for b in before}
        emd_before, emd_after = [], []
        for first, second in combinations([b for b in names if b in before], 2):
            emd_before.append(wasserstein_distance(before[first], before[second]))
            emd_after.append(wasserstein_distance(after[first], after[second]))
        mads = []
        for b in before:
            spread_before = np.median(np.abs(before[b] - np.median(before[b])))
            spread_after = np.median(np.abs(after[b] - np.median(after[b])))
            if spread_before > 0:
                mads.append(abs(spread_before - spread_after) / spread_before)
        before_all += emd_before
        after_all += emd_after
        mad_all += mads
        mean_before = float(np.mean(emd_before)) if emd_before else 0.0
        per_marker.append(
            {
                "marker": marker,
                "emd_reduction": round(1.0 - float(np.mean(emd_after)) / mean_before, 4) if mean_before else 0.0,
                "mad": round(float(np.mean(mads)), 4) if mads else 0.0,
            }
        )
    mean_before = float(np.mean(before_all)) if before_all else 0.0
    overall = {
        "emd_reduction": round(1.0 - float(np.mean(after_all)) / mean_before, 4) if mean_before else 0.0,
        "mad": round(float(np.mean(mad_all)), 4) if mad_all else 0.0,
    }
    return overall, per_marker


def histograms(frame, markers, unit_column, cofactor, bins=HISTOGRAM_BINS):
    """Each marker's distribution in each unit, as measured (arcsinh) and corrected, on one scale per marker.

    Each unit's row is the share of its sampled cells in each bin; values beyond the scale count in the end bins.
    """
    units = sorted(frame[unit_column].astype(str).unique())
    found = {"units": units, "bins": bins, "markers": {}}
    for marker in markers:
        paired = _paired(frame, marker)
        before = np.arcsinh(paired[marker].to_numpy(dtype=float) / cofactor)
        after = paired[marker + "_corr"].to_numpy(dtype=float)
        masks = [(paired[unit_column].astype(str) == unit).to_numpy() for unit in units]
        both = np.concatenate([before, after])
        low, high = (float(both.min()), float(np.percentile(both, 99.5))) if both.size else (0.0, 1.0)
        if high <= low:
            high = low + 1.0
        edges = np.linspace(low, high, bins + 1)
        shares = {}
        for when, values in (("before", before), ("after", after)):
            shares[when] = [
                np.round(
                    np.histogram(np.clip(values[mask], low, high), bins=edges)[0] / max(int(mask.sum()), 1), 4
                ).tolist()
                for mask in masks
            ]
        found["markers"][marker] = {"range": [round(low, 4), round(high, 4)], **shares}
    return found


def _numbers(values):
    """Tab 6's reading of a measured column: numbers, blanks as 0."""
    import pandas as pd

    return pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(dtype=float)


def _paired(frame, marker):
    """Compare the same cells, excluding missing/invalid raw or corrected values."""
    import pandas as pd

    columns = [marker, marker + "_corr"]
    numbers = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numbers.to_numpy(dtype=float)).all(axis=1)
    result = frame.loc[valid].copy()
    result[columns] = numbers.loc[valid]
    return result


def check(table, config):
    """The Batch correction step's summary, measures and before/after heatmap for *table*; cached."""
    values = settings(config)
    from .correction import last_run
    saved = last_run(table)
    if saved and isinstance(saved.get("config"), dict):
        values = settings({"tab_configurations": {"tab6_batch_correction": {"bc_config": saved["config"]}}})
        values["method"] = {"cycombine": "cyCombine (Python port)", "monotonic": "SPACEMAP monotonic",
                            "per_cluster": "Legacy approximation"}.get(saved.get("mode"), saved.get("mode", "ComBat"))
    key = ("check", str(table), Path(table).stat().st_mtime_ns, tuple(sorted(values.items())))
    with _lock:
        if key in _cache:
            return _cache[key]
    result = _check(table, values)
    with _lock:
        _cache[key] = result
    return result


def _check(table, values):
    unit = values["unit_column"]
    found = sample(table, values["batch_column"], (unit,) if unit else (), values.get("suffix", "_corr"))
    frame, markers, batch = found["frame"], found["markers"], values["batch_column"]
    started = time.perf_counter()
    # Measured between the units the correction removed differences between, as Tab 6 measures its run.
    measured_by = unit if unit and unit in frame.columns else batch
    frame[measured_by] = frame[measured_by].astype(str)
    overall, per_marker = measures(frame, markers, measured_by, values["cofactor"])
    units = frame.groupby(batch, sort=True)[measured_by].first() if measured_by != batch else None
    import pandas as pd
    paired = {marker: _paired(frame, marker) for marker in markers}
    before = pd.DataFrame({m: np.arcsinh(paired[m][m] / values["cofactor"]).groupby(paired[m][batch]).median() for m in markers})
    after = pd.DataFrame({m: paired[m].groupby(batch)[m + "_corr"].median() for m in markers})
    before = before.reindex(sorted(frame[batch].unique()))
    after = after.reindex(before.index)
    path = Path(table)
    return {
        **values,
        "table": path.name,
        "cells": found["cells"],
        "batches": [{"name": str(name), "cells": int(count)} for name, count in found["counts"].sort_index().items()],
        "markers": markers,
        "overall": overall,
        "per_marker": per_marker,
        "heatmap": {
            "batches": [str(name) for name in before.index],
            "markers": markers,
            "before": before.round(4).astype(object).where(before.notna(), None).values.tolist(),
            "after": after.round(4).astype(object).where(after.notna(), None).values.tolist(),
        },
        "histograms": histograms(frame, markers, measured_by, values["cofactor"]),
        "measured_by": measured_by,
        "unit_of": {str(k): str(v) for k, v in units.items()} if units is not None else {},
        "sampled": int(len(frame)),
        "read_seconds": found["seconds"],
        "ms": round(1000 * (time.perf_counter() - started)),
    }
