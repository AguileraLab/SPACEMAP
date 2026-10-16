"""Name suggestions for clusters, from the user's reference cells and from their markers.

For a finished clustering (its column of groups) and a reference column the
user labelled (``phenotype_ref...``):

* **Reference cells in each cluster**: how many reference cells of each label
  fall in it, as a share of all that label's reference cells ("62% of the
  CD8 T references are here") and of the cluster's reference cells. The three
  labels with the largest share of their own total are shown per cluster.
* **By markers**: the reference suggestion model (``spacemap.viewer.suggestions.Model``,
  the one that suggests a phenotype while labelling cells: neighbours among
  the reference cells, markers arcsinh-scaled) votes on a sample of each
  cluster's cells; the mean vote is how much the cluster looks like each label.

A label's score for a cluster weighs both: the marker vote, and the share of
the cluster's reference cells with that label, trusted more the more reference
cells the cluster holds. The suggestions are the three best scores.
"""

from collections import OrderedDict
import threading
import json

import numpy as np
import pandas as pd


def owns_output(root, table, cluster_job_id, output):
    """A successful name save for this exact run owns its phenotype output."""
    from . import jobs
    for job in jobs.recent(root, limit=300):
        result = job.get("result") or {}
        if job.get("status") != "done":
            continue
        if job.get("kind") != "name_clusters":
            outputs = [*(result.get("columns") or [result.get("column")])]
            if result.get("overwrite"):
                outputs.append(result.get("phenotype_column"))
            if result.get("table") == str(table) and output in outputs:
                return job["id"] == cluster_job_id
            continue
        if result.get("column") != output:
            continue
        try:
            settings = json.loads((jobs.directory(root, job["id"]) / "request.json").read_text())["settings"]
        except (ValueError, OSError, KeyError):
            continue
        # The latest writer is the owner, even if another run used this name earlier.
        if settings.get("table") == str(table):
            return settings.get("cluster_job_id") == cluster_job_id
    return False


def saved_names(root, table, clusters_column, *, cluster_job_id=None):
    """The latest successful full name mapping, including older jobs' request files."""
    from . import jobs
    for job in jobs.recent(root, limit=300):
        result = job.get("result") or {}
        if (job.get("status") == "done" and result.get("clusters") and result.get("table") == str(table)
                and result.get("column") == clusters_column):
            # A new clustering may reuse its column; names from earlier membership are stale.
            break
        if job.get("kind") != "name_clusters" or job.get("status") != "done":
            continue
        try:
            settings = json.loads((jobs.directory(root, job["id"]) / "request.json").read_text()).get("settings") or {}
        except (ValueError, OSError):
            continue
        if settings.get("table") == str(table) and settings.get("clusters_col") == clusters_column:
            if cluster_job_id and settings.get("cluster_job_id") != cluster_job_id:
                continue
            return {"names":settings.get("names") or {}, "column":result.get("column"), "job_id":job["id"]}
    return {"names":{}, "column":None, "job_id":None}

SAMPLE = 400  # cells per cluster asked of the model
PER_LABEL = 2000  # reference cells per label the model learns from
TRUST = 10  # reference cells at which their share counts half
_CACHE = OrderedDict()
_LOCK = threading.Lock()


def reference_columns(table):
    from spacemap.viewer.reference import columns
    return columns(table)


def _model_columns(markers, available, suffix):
    """The measured columns the model reads: the raw marker when the table has it (corrected values are already
    arcsinh-scaled, and the model scales them itself)."""
    from .markers import base_of

    present = set(available)
    return [
        base_of(m, suffix) if base_of(m, suffix) in present else m
        for m in markers
        if m in present or base_of(m, suffix) in present
    ]


def suggest(table, clusters_column, reference_column, markers, suffix="_corr", *, cached_only=False):
    """{"labels": [...], "labelled": n, "clusters": {cluster: {...}}} for *clusters_column* of *table*; cached."""
    from spacemap import dataio
    from spacemap.viewer.sources import fingerprint
    from spacemap.viewer.suggestions import Model, named

    available = dataio.columns(table)
    for column in (clusters_column, reference_column):
        if column not in available:
            raise ValueError(f"The table has no {column} column.")
    columns = _model_columns(markers, available, suffix)
    from spacemap.viewer.cells import _edits_token
    key = (str(table), fingerprint(table), _edits_token(str(table)), clusters_column, reference_column, tuple(columns))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    if cached_only:
        return None
    frame = dataio.load(table, columns=list(dict.fromkeys([clusters_column, reference_column])))
    groups = frame[clusters_column]
    labels = pd.Series(named(frame[reference_column]), index=frame.index)
    labelled = labels != ""
    totals = labels[labelled].value_counts()
    if totals.empty:
        raise ValueError(f"{reference_column} has no labelled cells yet.")
    counts = pd.crosstab(groups[labelled], labels[labelled])
    # By markers: the reference model votes on a sample of each cluster's cells.
    votes, why = None, ""
    try:
        rng = np.random.default_rng(0)
        taught = np.concatenate(
            [rng.permutation(np.flatnonzero((labels == name).to_numpy()))[:PER_LABEL] for name in totals.index]
        )
        samples = {}
        for group, rows in groups.groupby(groups).groups.items():
            rows = frame.index.get_indexer(rows)
            samples[group] = rows if len(rows) <= SAMPLE else rng.choice(rows, SAMPLE, replace=False)
        needed = np.unique(np.concatenate([taught, *samples.values()]))
        from . import data_view
        measured = data_view._table(str(table)).take(needed, columns)
        values = pd.DataFrame(measured).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        model = Model(taught, values[np.searchsorted(needed, taught)], labels.to_numpy()[taught])
        votes = {}
        for group, sample in samples.items():
            predicted = model.predict(values[np.searchsorted(needed, sample)], sample)
            mean = np.nanmean(predicted, axis=0)
            votes[str(group)] = {
                str(name): float(value) for name, value in zip(model.classes, mean) if np.isfinite(value)
            }
    except ValueError as error:  # too few examples of a label: the reference share still helps
        why = str(error)
    found = {}
    for group in pd.unique(groups.dropna()):
        name = str(group)
        here = counts.loc[group] if group in counts.index else pd.Series(dtype=float)
        n = int(here.sum())
        top = sorted(
            (
                {
                    "label": str(label),
                    "cells": int(here[label]),
                    "of_label": round(float(here[label]) / float(totals[label]), 4),
                    "of_cluster": round(float(here[label]) / n, 4) if n else 0.0,
                }
                for label in here.index
                if here[label] > 0
            ),
            key=lambda item: (-item["of_label"], -item["cells"]),
        )
        trust = n / (n + TRUST)
        by_markers = (votes or {}).get(name, {})
        scores = []
        for label in totals.index:
            share = float(here.get(label, 0)) / n if n else 0.0
            marker = by_markers.get(str(label))
            if marker is None:
                score = share * trust
            else:
                weight = 0.5 * trust  # the reference cells count more the more of them the cluster holds
                score = (1 - weight) * marker + weight * share
            scores.append(
                {
                    "label": str(label),
                    "score": round(score, 4),
                    "by_markers": round(marker, 4) if marker is not None else None,
                    "reference_cells": int(here.get(label, 0)),
                    "of_label": round(float(here.get(label, 0)) / float(totals[label]), 4),
                }
            )
        scores.sort(key=lambda item: -item["score"])
        found[name] = {"labelled": n, "top": top[:3], "suggestions": scores[:3]}
    result = {
        "reference_column": reference_column,
        "labels": [{"label": str(label), "cells": int(count)} for label, count in totals.items()],
        "labelled": int(labelled.sum()),
        "by_markers": votes is not None,
        "note": why,
        "clusters": found,
    }
    with _LOCK:
        _CACHE[key] = result
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    return result
