"""Studio's Leiden method (Tab 7): run Tab 7's Leiden pipeline as a job, then name the clusters.

The settings are Tab 7's own, in ``outputs/phenotyping/cache/leiden_params.json``
(what classic Leiden reads and saves). The job runs Tab 7's headless pipeline
(``spacemap_workflows.leiden_pipeline``) on the combined table, read from its
fast copy, and saves the cluster of every cell as a new column
``<output>_clusters``, with each cluster's size and strongest markers. Naming
the clusters saves a second new column, ``<output>``, with the phenotype of
each cell. Explicit overwrite replaces these two columns and resets old names
to the new cluster numbers; other columns are kept.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

LOW_EXPRESSION = -1
# Tab 7's defaults (classic leiden_clustering.py).
DEFAULTS = {
    "markers": [],
    "cell_id_col": "cell_id",
    "batch_id_col": "",
    "enable_filter": False,
    "filter_col": "",
    "filter_val": "1",
    "resolution": 2.8,
    "n_neighbors": 15,
    "n_pcs": 15,
    "n_clusters_target": 45,
    "subsample_size": 200000,
    "use_intelligent": True,
    "compute_umap": False,
    "meta_clusters": 0,
    "use_gpu": True,
    "random_seed": 42,
    "norm_z1": True,
    "norm_z2": False,
    "norm_cdf": False,
    "norm_nlog": False,
    "norm_sinh": True,
    "enable_low_expr_filter": True,
    "low_expr_metric": "mean_zscore",
    "low_expr_threshold": -0.75,
    "output_col": "phenotype_leiden",
}
# What Studio shows, with limits; the rest stay as Tab 7 has them.
EDITABLE = {
    "markers": (list, None, None),
    "resolution": (float, 0.1, 5.0),
    "n_neighbors": (int, 2, 200),
    "subsample_size": (int, 1000, 10_000_000),
    "use_gpu": (bool, None, None),
    "enable_low_expr_filter": (bool, None, None),
    "low_expr_threshold": (float, -10.0, 100000.0),
    "output_col": (str, None, None),
    # Tab 7's transformation and normalisation, in its order: sinh back, Z1 (per batch), Z2, CDF, -log.
    "norm_sinh": (bool, None, None),
    "norm_z1": (bool, None, None),
    "norm_z2": (bool, None, None),
    "norm_cdf": (bool, None, None),
    "norm_nlog": (bool, None, None),
    "batch_id_col": ("column", None, None),
}
OUTPUT = re.compile(r"phenotype_[A-Za-z0-9_]{1,100}")
COLUMN = re.compile(r"[A-Za-z][A-Za-z0-9 _.-]{0,127}")


def check_normalisation(values):
    """Tab 7's rule: -log comes after the CDF (it is the log of a 0-1 value)."""
    if values.get("norm_nlog") and not values.get("norm_cdf"):
        raise ValueError("Turn on CDF before -log, as in Tab 7.")


def cache_path(root):
    return Path(root) / "outputs" / "phenotyping" / "cache" / "leiden_params.json"


def values(root):
    path = cache_path(root)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    found = {**DEFAULTS, **(saved if isinstance(saved, dict) else {})}
    if saved.get("low_expr_threshold") is None and saved.get("min_marker_sum") is not None:
        found["low_expr_threshold"] = saved["min_marker_sum"]
    return found


def check(changes):
    if not isinstance(changes, dict) or set(changes) - set(EDITABLE):
        raise ValueError("Unknown Leiden setting.")
    for key, value in changes.items():
        kind, low, high = EDITABLE[key]
        if kind is list:
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                raise ValueError("Choose at least one marker.")
        elif kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Choose on or off for {key}.")
        elif kind == "column":
            if not isinstance(value, str) or (value and not COLUMN.fullmatch(value)):
                raise ValueError("Choose a column of the table, or none.")
        elif kind is str:
            if not isinstance(value, str) or not OUTPUT.fullmatch(value):
                raise ValueError("Name the result phenotype_ followed by letters, numbers or underscores.")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
        elif kind is int and int(value) != value:
            raise ValueError(f"{key} must be a whole number.")
    return changes


def save(root, changes, expected):
    """Merge *changes* into Tab 7's Leiden settings file, as classic's Save Parameters writes it."""
    from .settings import project_lock, revision

    check(changes)
    with project_lock(root):
        if expected != revision(values(root)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        path = cache_path(root)
        saved = json.loads(path.read_text()) if path.is_file() else {}
        saved.update(
            {
                key: value if EDITABLE[key][0] in (list, str, "column") else EDITABLE[key][0](value)
                for key, value in changes.items()
            }
        )
        check_normalisation({**values(root), **saved})
        if "low_expr_threshold" in changes:
            saved["min_marker_sum"] = saved["low_expr_threshold"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(path)
        return values(root)


def pipeline_settings(values, table, cofactor):
    """The request Tab 7's Leiden run builds (classic leiden_clustering.py), for *table*."""
    return {
        "full_input_path": str(table),
        "markers_selected": list(values["markers"]),
        "cell_id_col": values["cell_id_col"],
        "batch_id_col": values["batch_id_col"],
        "enable_filter": bool(values["enable_filter"]),
        "filter_col": values["filter_col"] if values["enable_filter"] else None,
        "filter_val": values["filter_val"] if values["enable_filter"] else None,
        "norm_z1": values["norm_z1"],
        "norm_z2": values["norm_z2"],
        "norm_cdf": values["norm_cdf"],
        "norm_nlog": values["norm_nlog"],
        "norm_sinh": values["norm_sinh"],
        "low_expr_metric": values["low_expr_metric"],
        "enable_low_expr_filter": bool(values["enable_low_expr_filter"]),
        "low_expr_threshold": float(values["low_expr_threshold"]),
        "resolution": float(values["resolution"]),
        "n_neighbors": int(values["n_neighbors"]),
        "n_pcs": int(values["n_pcs"]),
        "use_intelligent": bool(values["use_intelligent"]),
        "subsample_size": int(values["subsample_size"]),
        "random_seed": int(values["random_seed"]),
        "compute_umap": bool(values["compute_umap"]),
        "meta_clusters": int(values["meta_clusters"]),
        "use_gpu": bool(values["use_gpu"]),
        "output_col_name": values["output_col"],
        "cofactor": float(cofactor),
        "fresh_assignment_mode": False,
    }


@contextmanager
def _given_input(path, frame):
    """Tab 7's pipeline reads its input with pandas.read_csv; hand it the columns already read.

    This runs in the job's own worker process, so nothing else sees the swap.
    """
    original = pd.read_csv

    def read(source, *args, **kwargs):
        return frame.copy() if str(source) == str(path) else original(source, *args, **kwargs)

    pd.read_csv = read
    try:
        yield
    finally:
        pd.read_csv = original


class _Log:
    """The pipeline's logger, passed on to the job's progress line."""

    def __init__(self, report):
        self.report = report

    def _say(self, message, *args):
        text = str(message) % args if args else str(message)
        if text.strip("= "):
            self.report(text[:200])

    info = warning = error = debug = _say

    def protect_output(self):
        protect = getattr(self.report, "protect_output", None)
        if protect is not None:
            protect()


def cluster_name(cluster):
    return "Low expression" if int(cluster) == LOW_EXPRESSION else f"Cluster {int(cluster)}"


def run(settings, folder, report):
    """The job: Tab 7's Leiden on the combined table, saved as ``<output>_clusters``."""
    from spacemap import dataio
    from spacemap_workflows.leiden_pipeline import _compute

    from .tasks import save_result, validate_table

    validate_table(settings)
    table, request = settings["table"], settings["pipeline"]
    anchor_column = settings.get("cell_id_col", "cell_id")
    wanted = [anchor_column, *request["markers_selected"]]
    for key in ("batch_id_col", "filter_col"):
        if request.get(key) and request[key] not in wanted:
            wanted.append(request[key])
    roi_column = settings.get("roi_col") or "batch_id"  # the combined table's ROI of each cell
    if roi_column not in wanted and roi_column in dataio.columns(table):
        wanted.append(roi_column)
    report("Reading the combined table…", 0.02)
    frame = dataio.load(table, columns=wanted)
    with _given_input(request["full_input_path"], frame):
        found = _compute(
            {"project_root": settings["project_root"], "settings": request, "audit": None},
            Path(folder),
            _Log(report),
            {},
        )
    results = pd.read_parquet(found["leiden_temp_file"])
    clusters = results.set_index(results[anchor_column].astype(str))["leiden_cluster"]
    labels = (
        frame[anchor_column]
        .astype(str)
        .map(clusters)
        .map(lambda value: cluster_name(value) if pd.notna(value) else None)
    )
    summary, profile = describe(results, found["leiden_norm_cols"], request)
    output = settings["output_col"]
    columns = {output: labels}
    overwrite = bool(settings.get("overwrite_columns"))
    if overwrite:
        # Cluster numbers change between runs: previous phenotype names cannot be reused.
        columns[settings["phenotype_col"]] = labels
    saved = save_result(settings, columns, frame[anchor_column], report)
    return {
        **saved,
        "clusters": summary,
        "profile": profile,
        "where": where(labels, frame[roi_column]) if roi_column in frame.columns else {},
        "markers": list(request["markers_selected"]),
        "phenotype_column": settings["phenotype_col"],
        "overwrite": overwrite,
        "n_clusters": found["leiden_n_clusters"],
    }


def where(labels, rois, limit=60):
    """{group: [{"roi", "cells"}, ...]}: where each group's cells are, most first, for "where is it" on the page."""
    frame = pd.DataFrame({"group": labels.to_numpy(), "roi": rois.astype(str).to_numpy()}).dropna(subset=["group"])
    counts = frame.groupby(["group", "roi"]).size().rename("cells").reset_index()
    return {
        str(group): [
            {"roi": str(row.roi), "cells": int(row.cells)}
            for row in part.sort_values(["cells", "roi"], ascending=[False, True]).head(limit).itertuples()
        ]
        for group, part in counts.groupby("group")
    }


def heatmap(frame, columns, rows, note, raw=None):
    """A profile for Studio's clickable heatmap: *frame* (groups x columns) compared across the groups."""
    spread = frame.std(axis=0).replace(0, 1).fillna(1)
    compared = ((frame - frame.mean(axis=0)) / spread).fillna(0)
    found = {
        "rows": list(rows),
        "columns": [str(name) for name in columns],
        "values": np.round(compared.to_numpy(dtype=float), 3).tolist(),
        "note": note,
    }
    if raw is not None:
        found["raw"] = np.round(raw.to_numpy(dtype=float), 4).tolist()
    return found


def describe(results, columns, config):
    """Summarize the exact transformed columns used by classic's Leiden pipeline."""
    from .cluster_profiles import build

    return build(results, columns, "leiden_cluster", config=config,
                 markers=config["markers_selected"], cofactor=config["cofactor"])


def name_clusters(settings, folder, report):
    """The job: each cell's cluster name mapped to the names given, saved as a new column."""
    from spacemap import dataio

    from .tasks import save_result, validate_table

    validate_table(settings)
    table, anchor_column = settings["table"], settings.get("cell_id_col", "cell_id")
    report("Reading the clusters…", 0.1)
    frame = dataio.load(table, columns=[anchor_column, settings["clusters_col"]])
    names = {str(key): str(value) for key, value in settings["names"].items()}
    phenotypes = frame[settings["clusters_col"]].map(
        lambda value: names.get(str(value), value) if pd.notna(value) else None
    )
    saved = save_result(settings, {settings["output_col"]: phenotypes}, frame[anchor_column], report)
    counts = phenotypes.value_counts()
    return {**saved, "phenotypes": {str(key): int(value) for key, value in counts.items()}, "names":names,
            "clusters_column":settings["clusters_col"], "cluster_job_id":settings.get("cluster_job_id")}


def check_names(names, clusters):
    """{cluster name: phenotype} for the clusters of a Leiden result."""
    if not isinstance(names, dict) or not names:
        raise ValueError("Name at least one cluster.")
    known = {item["name"] for item in clusters}
    for key, value in names.items():
        if key not in known:
            raise ValueError(f"There is no {key} in this result.")
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 60:
            raise ValueError(f"Give {key} a phenotype name of up to 60 characters.")
    return {key: value.strip() for key, value in names.items()}
