"""One display calculation for clustering; never transform normalized columns again."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from spacemap.cluster_profiles import summarize

VERSION = 2
METHODS = {"leiden", "som", "vortex", "subphenotype", "consensus_ml"}
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cluster-profiles")
_LOCK = threading.Lock()
_UPGRADES = {}


def transformation(config, markers, available, cofactor=5.0):
    """Describe the saved run, never the editable settings for the next run."""
    corrected = [m for m in markers if m.endswith("_corr")]
    source = "Corrected (arcsinh)" if len(corrected) == len(markers) else "Raw + corrected (arcsinh)" if corrected else "Measured values"
    steps = []
    if corrected and config.get("norm_sinh"):
        steps.append(f"Back to measured scale (sinh × {cofactor:g})")
    within = config.get("batch_id_col") if config.get("batch_id_col") in available else ""
    if config.get("norm_z1"):
        steps.append("Z1" + (f" within {within}" if within else ""))
    for key, label in [("norm_z2", "Z2"), ("norm_cdf", "CDF"), ("norm_nlog", "−log(1−CDF)")]:
        if config.get(key):
            steps.append(label)
    return {"input": source, "steps": steps, "within": within or "whole table",
            "label": source + (" → " + " → ".join(steps) if steps else "; no additional transformation")}


def build(frame, columns, group, *, config=None, markers=None, cofactor=5.0, label=None,
          name=lambda value: "Low expression" if int(value) == -1 else f"Cluster {int(value)}",
          exclude_low=True):
    """Return cluster list and both summaries. Inputs must already be on the intended scale."""
    columns = list(columns)
    if not columns or any(c not in frame for c in columns):
        raise ValueError("The saved transformed marker columns are missing; the heatmap cannot be reconstructed accurately.")
    names = [c[:-5] if c.endswith("_norm") else c for c in columns]
    marker_names = markers or names
    transform = transformation(config or {}, marker_names, frame.columns, cofactor)
    if label:
        transform = {"label": label}
    valid = frame.loc[frame[group].notna() & (frame[group].ne(-1) if exclude_low else True), [group, *columns]]
    clusters = [{"id": int(g) if isinstance(g, (int, float, np.integer)) else str(g),
                 "name": name(g), "cells": int(n), "top_markers": []}
                for g, n in frame[group].value_counts().sort_index().items()]
    summaries = {}
    for aggregation in ("median", "mean"):
        values, z = summarize(valid, group, columns, aggregation)
        values.columns = z.columns = names
        tops = {name(g): z.loc[g].sort_values(ascending=False).index[:5].tolist() for g in z.index}
        summaries[aggregation] = {
            "rows": [name(g) for g in values.index], "columns": names,
            "values": np.round(z.to_numpy(dtype=float), 3).tolist(),
            "raw": np.round(values.to_numpy(dtype=float), 4).tolist(),
            "top_markers": tops, "aggregation": aggregation,
            "note": f"{aggregation.title()} · {transform['label']} · colour: marker z-score across clusters",
            "raw_label": aggregation.title() + " after the selected transformations",
        }
    profile = {**summaries["median"], "summaries": summaries, "version": VERSION, "transformation": transform}
    for cluster in clusters:
        cluster["top_markers"] = profile["top_markers"].get(cluster["name"], [])
    return clusters, profile


@lru_cache(maxsize=128)
def _orders(values):
    from scipy.cluster.hierarchy import leaves_list, linkage
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        return [], []
    order = lambda x: leaves_list(linkage(x, method="ward")).astype(int).tolist() if len(x) > 1 else list(range(len(x)))
    return order(matrix), order(matrix.T)


def choose(result, aggregation):
    profile = result.get("profile") or {}
    selected = (profile.get("summaries") or {}).get(aggregation)
    if not selected:
        return result
    row_order, column_order = _orders(tuple(tuple(row) for row in selected["values"]))
    return {**result, "profile": {**profile, **selected, "row_order": row_order, "column_order": column_order},
            **({"phenotypes": [{**c, "top_markers": selected["top_markers"].get(c["name"], [])}
                                for c in result["phenotypes"]]} if "phenotypes" in result else {}),
            "clusters": [{**c, "top_markers": selected["top_markers"].get(c["name"], [])}
                         for c in result.get("clusters", [])]}


def preference(root, method, aggregation=None):
    from .settings import project_lock
    from spacemap_workflows.leiden_job import write_json

    path = Path(root) / "outputs/studio/heatmap_settings.json"
    if aggregation is None:
        saved = json.loads(path.read_text()) if path.exists() else {}
        return saved.get(method, "median")
    with project_lock(root):
        saved = json.loads(path.read_text()) if path.exists() else {}
        if aggregation is not None:
            if method not in METHODS or aggregation not in {"median", "mean"}:
                raise ValueError("Choose Median or Mean for a phenotyping heatmap.")
            saved[method] = aggregation
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, saved)
        return saved.get(method, "median")


def _upgrade(folder, result):
    """Use immutable Leiden output, including its saved _norm values; no reclustering."""
    import pyarrow.parquet as pq

    settings = json.loads((folder / "request.json").read_text())["settings"]
    pipeline = settings["pipeline"]
    source = folder / "results.parquet"
    columns = [m + "_norm" for m in pipeline["markers_selected"]]
    available = pq.read_schema(source).names
    wanted = list(dict.fromkeys(["leiden_cluster", *columns,
        *([pipeline["batch_id_col"]] if pipeline.get("batch_id_col") in available else [])]))
    frame = pd.read_parquet(source, columns=wanted)
    clusters, profile = build(frame, columns, "leiden_cluster", config=pipeline,
                              markers=pipeline["markers_selected"], cofactor=pipeline.get("cofactor", 5.0))
    # Keep all original job metadata and names; only replace the display summaries.
    return {**result, "clusters": clusters, "profile": profile}


def for_job(root, folder, job):
    """Upgrade legacy Leiden profiles in the background once; all readers see the same choice."""
    result = job.get("result") or {}
    if job.get("status") != "done" or job.get("kind") not in METHODS or not result.get("profile"):
        return job
    profile = result["profile"]
    if profile.get("version") != VERSION and job["kind"] == "leiden":
        # Fingerprints allow recovery if a missing or replaced artifact is restored.
        paths = [folder / "results.parquet", folder / "request.json", folder / "status.json"]
        if not all(p.is_file() for p in paths):
            return {**job, "result": {**result, "profile": None,
                "profile_notice": "This older run has no saved transformed values. Run clustering again to build an accurate heatmap.",
                "clusters": [{**c, "top_markers": []} for c in result.get("clusters", [])]}}
        key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        with _LOCK:
            future = _UPGRADES.get(key)
            if future is None:
                future = _UPGRADES[key] = _POOL.submit(_upgrade, folder, deepcopy(result))
                if len(_UPGRADES) > 40:
                    for old in list(_UPGRADES):
                        if old != key and _UPGRADES[old].done():
                            del _UPGRADES[old]
                            break
        if not future.done():
            return {**job, "result": {**result, "profile": None, "profile_pending": True,
                    "profile_notice": "Updating the heatmap from this run’s saved transformations…",
                    "clusters": [{**c, "top_markers": []} for c in result.get("clusters", [])]}}
        try:
            result = future.result()
        except (ValueError, OSError, KeyError) as exc:
            return {**job, "result": {**result, "profile": None,
                "profile_notice": f"The saved heatmap could not be updated: {exc}",
                "clusters": [{**c, "top_markers": []} for c in result.get("clusters", [])]}}
    elif profile.get("version") != VERSION:
        result = {**result, "profile_notice": "Older heatmap: run clustering again to enable Median / Mean using the saved transformations."}
    return {**job, "result": choose(result, preference(root, job["kind"]))}
