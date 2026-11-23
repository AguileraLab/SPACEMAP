"""Cluster names from saved Manual reference models or labelled reference cells.

Saved-model mode predicts sampled cells using the shared feature panel and the
model's preprocessing, adapting from retained training examples when needed.
Model votes, manual counts and saved validation remain separate evidence.

Reference-cell mode combines the following for a finished clustering and a
reference column the user labelled (``phenotype_ref...``):

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
    for job in jobs.history(root):
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
    for job in jobs.history(root):
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


def _feature_panel(meta, table, features=None, suffix="_corr"):
    """Keep the model's measurement columns, limited to markers in this run."""
    from . import reference_models
    from .markers import base_of
    present = reference_models._present(meta, table)
    if features is None:
        return present
    selected = {base_of(name, suffix) for name in features}
    return [name for name in present if base_of(name, suffix) in selected]


def suggestion_sources(root, table, features=None, suffix="_corr"):
    """List compatible saved reference models using metadata, never load weights."""
    from . import reference_models, reference_setup

    configured = reference_setup.options(root, table)
    local = reference_models.available(table)
    models = []
    seen = set()
    def entry(item, identifier, key=""):
        present = _feature_panel(item, table, features, suffix)
        missing = [name for name in item.get("markers", []) if name not in present]
        compatible = bool(present) and (not missing or bool(item.get("adaptable") and item.get("allow_missing", True)))
        return {"id": identifier, "key": key, "name": item.get("name") or item.get("column", ""),
                "column": item.get("column", ""), "method": item.get("method", ""), "trained": item.get("trained", ""),
                "features": present, "total_features": len(item.get("markers", [])), "missing": missing,
                "compatible": compatible}
    for item in local:
        identifier = item["model_id"]
        models.append(entry(item, identifier))
        seen.add(identifier)
    for item in configured["models"]:
        if item["id"] not in seen:
            models.append(entry(item, item["id"], item["key"]))
    active = configured.get("base_model_id") or ""
    return {"reference_columns": reference_columns(table), "models": models,
            "active_model": reference_models.current_model_id(active) if active else ""}


def suggest_model(table, clusters_column, model_id, reference_column="", *, features=None, suffix="_corr", cached_only=False):
    """Sample each cluster with the saved model, retaining its fitted preprocessing.

    The cache can be read during AI prompt preparation without reading cell rows
    or deserializing a model. Scores describe model votes, not naming accuracy.
    """
    from spacemap import dataio
    from spacemap.viewer.cells import _edits_token
    from spacemap.viewer.sources import fingerprint
    from . import reference_models, subphenotype_guidance

    identifier = reference_models.current_model_id(model_id)
    meta = reference_models.metadata(identifier)
    available = dataio.columns(table)
    if clusters_column not in available:
        raise ValueError("This clustering column is no longer available. Run clustering again.")
    if reference_column and reference_column not in reference_columns(table):
        raise ValueError("Choose an available manual-reference column.")
    reference_column = reference_column or (meta.get("column") if meta.get("column") in reference_columns(table) else "")
    present = _feature_panel(meta, table, features, suffix)
    if not present:
        raise ValueError("No model features match this clustering. Choose another model.")
    path = reference_models.model_path(identifier)
    key = ("saved_model", str(path), fingerprint(path), fingerprint(path.with_suffix('.json')),
           str(table), fingerprint(table), _edits_token(str(table)), clusters_column, reference_column, tuple(present))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    if cached_only:
        return None
    groups = dataio.load(table, columns=[clusters_column])[clusters_column]
    if not groups.notna().any():
        raise ValueError("This clustering has no cells to name.")
    found = subphenotype_guidance.build(table, groups, {"model_id": identifier, "reference_column": reference_column,
                                                      "model_features": present}, lambda *_: None)
    found["model"] = {"id": identifier, "name": meta.get("name") or meta.get("method", "Model"),
                      "method": meta.get("method"), "column": meta.get("column"), "trained": meta.get("trained"),
                      "validation_held_out": bool(meta.get("validation_held_out"))}
    with _LOCK:
        _CACHE[key] = found
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    return found


def evidence(hints, clusters=None):
    """Bounded, source-labelled evidence shared with the assistant."""
    known = hints["clusters"]
    chosen = list(dict.fromkeys(clusters or known))[:80]
    selected = {name: known[name] for name in chosen if name in known}
    labels = {s["label"] for info in selected.values() for s in info.get("suggestions", [])}
    validation = hints.get("validation")
    if validation:
        validation = {**validation, "phenotypes": [p for p in validation.get("phenotypes", []) if p["phenotype"] in labels]}
    return {"column": hints.get("reference_column", ""), "model": hints.get("model"),
            "features": hints.get("features", []), "missing_features": hints.get("missing", []),
            "validation": validation,
            "coverage": f"{len(selected)} of {len(known)} clusters; request other IDs with cluster_reference when needed.",
            "score_meaning": "Mean model probability on sampled cells; not independent accuracy." if hints.get("model_id") else
                             "Reference-cell and marker-neighbour evidence; not independent accuracy.",
            "per_cluster": {cluster: {"suggested": info.get("suggestions", []),
                "reference_cells": info.get("top", []), "sampled": info.get("model_sampled"), "scored": info.get("model_scored")}
                for cluster, info in selected.items()}}


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
