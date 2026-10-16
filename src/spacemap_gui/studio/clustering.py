"""Studio's SOM (and next VORTEX) clustering for Tab 7: classic's settings file, a job, and the explorer's result.

Each method keeps classic's settings file in ``outputs/phenotyping/cache`` (what classic reads and saves)
and runs classic's steps without Streamlit (``spacemap_workflows.som_pipeline``). The job saves every
cell's group in a new column ``<output>_clusters`` ("Cluster <n>", or "Low expression"), with each
group's size, its five most raised markers, a heatmap of the groups and where each group is (cells per
ROI), as Leiden's does, so the same explorer names them into ``<output>``. Earlier columns are kept unless overwrite is explicitly selected.
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

OUTPUT = re.compile(r"phenotype_[A-Za-z0-9_]{1,100}")


def _som_defaults():
    from spacemap_workflows.som_pipeline import DEFAULTS

    return DEFAULTS


def _vortex_defaults():
    from spacemap_workflows.vortex_pipeline import DEFAULTS

    return DEFAULTS


_NORMALISATION = {
    "norm_sinh": (bool, None, None),
    "norm_z1": (bool, None, None),
    "norm_z2": (bool, None, None),
    "norm_cdf": (bool, None, None),
    "norm_nlog": (bool, None, None),
    "batch_id_col": ("column", None, None),
}

METHODS = {
    "vortex": {
        "file": "vortex_params.json",
        "name": "VORTEX",
        "defaults": _vortex_defaults,
        "group": "vortex_cluster",
        "editable": {
            "markers": (list, None, None),
            "auto_select_k": (bool, None, None),
            "vortex_k": (int, 2, 500),
            "n_clusters_target": (int, 2, 500),
            "subsample_size": (int, 1000, 10_000_000),
            "use_gpu": (bool, None, None),
            "enable_low_expr_filter": (bool, None, None),
            "low_expr_threshold": (float, -10.0, 100000.0),
            "output_col": (str, None, None),
            **_NORMALISATION,
        },
    },
    "som": {
        "file": "som_params.json",
        "name": "SOM",
        "defaults": _som_defaults,
        "group": "meta_cluster",
        # What Studio shows, with classic's limits; the rest stay as Tab 7 has them.
        "editable": {
            "markers": (list, None, None),
            "grid_dim": (int, 2, 200),
            "n_meta_target": (int, 2, 500),
            "mode": (("fast", "balanced", "high_quality"), None, None),
            "subsample_size": (int, 1000, 10_000_000),
            "enable_low_expr_filter": (bool, None, None),
            "low_expr_threshold": (float, -10.0, 100000.0),
            "output_col": (str, None, None),
            "norm_sinh": (bool, None, None),
            "norm_z1": (bool, None, None),
            "norm_z2": (bool, None, None),
            "norm_cdf": (bool, None, None),
            "norm_nlog": (bool, None, None),
            "batch_id_col": ("column", None, None),
        },
    },
}


def _method(method):
    if method not in METHODS:
        raise ValueError("Unknown clustering method.")
    return METHODS[method]


def cache_path(root, method):
    return Path(root) / "outputs" / "phenotyping" / "cache" / _method(method)["file"]


def values(root, method):
    path = cache_path(root, method)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    saved = saved if isinstance(saved, dict) else {}
    found = {**_method(method)["defaults"](), **{key: value for key, value in saved.items() if value is not None}}
    if saved.get("low_expr_threshold") is None and saved.get("min_marker_sum") is not None:
        found["low_expr_threshold"] = saved["min_marker_sum"]
    if found.get("low_expr_threshold") is None:  # classic: the metric's default cut
        from spacemap_gui.tabs.tab7_phenotyping.common import get_low_expression_default_threshold

        found["low_expr_threshold"] = get_low_expression_default_threshold(found.get("low_expr_metric", "mean_zscore"))
    return found


def check(method, changes):
    editable = _method(method)["editable"]
    if not isinstance(changes, dict) or set(changes) - set(editable):
        raise ValueError("Unknown setting.")
    for key, value in changes.items():
        kind, low, high = editable[key]
        if isinstance(kind, tuple):
            if value not in kind:
                raise ValueError(f"Choose one of {', '.join(kind)} for {key}.")
        elif kind is list:
            if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
                raise ValueError("Choose at least one marker.")
        elif kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"Choose on or off for {key}.")
        elif kind == "column":
            from .leiden import COLUMN

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


def save(root, method, changes, expected):
    """Merge *changes* into classic's settings file for *method*, as its Save writes it."""
    from .settings import project_lock, revision

    check(method, changes)
    editable = _method(method)["editable"]
    with project_lock(root):
        if expected != revision(values(root, method)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        path = cache_path(root, method)
        saved = json.loads(path.read_text()) if path.is_file() else {}
        for key, value in changes.items():
            kind = editable[key][0]
            saved[key] = value if kind in (list, str, "column") or isinstance(kind, tuple) else kind(value)
        if "low_expr_threshold" in changes:
            saved["min_marker_sum"] = saved["low_expr_threshold"]
        from .leiden import check_normalisation

        check_normalisation({**values(root, method), **saved})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(path)
        return values(root, method)


def group_name(value):
    return "Low expression" if int(value) == -1 else f"Cluster {int(value)}"


def describe(results, columns, markers, group_column, config=None, cofactor=5.0):
    from .cluster_profiles import build

    return build(results, columns, group_column, config=config, markers=markers, cofactor=cofactor)


def run(settings, folder, report):
    """The job: classic's SOM or VORTEX on the combined table, saved as ``<output>_clusters``."""
    from spacemap import dataio
    from spacemap_workflows import som_pipeline, vortex_pipeline

    from .leiden import _Log

    from .leiden import where
    from .tasks import save_result, validate_table

    validate_table(settings)
    table, config = settings["table"], settings["config"]
    anchor = settings.get("cell_id_col", "cell_id")
    available = dataio.columns(table)
    roi = settings.get("roi_col") or "batch_id"
    wanted = [anchor, *config["markers"]]
    for name in (config.get("batch_id_col"), config.get("filter_col") if config.get("enable_filter") else None, roi):
        if name and name in available and name not in wanted:
            wanted.append(name)
    report("Reading the combined table…", 0.02)
    frame = dataio.load(table, columns=wanted)
    method = settings.get("method", "som")
    group = METHODS[method]["group"]
    grid = k = None
    if method == "vortex":
        results, columns, cell_col, k = vortex_pipeline.compute(
            frame, config, float(settings["cofactor"]), report, logger=_Log(report)
        )
    else:
        results, columns, cell_col, grid = som_pipeline.compute(frame, config, float(settings["cofactor"]), report)
    by_cell = results.set_index(results[cell_col].astype(str))[group]
    labels = frame[anchor].astype(str).map(by_cell).map(lambda value: group_name(value) if pd.notna(value) else None)
    report("Describing the groups…", 0.9)
    clusters, profile = describe(results, columns, config["markers"], group, config, float(settings["cofactor"]))
    columns = {settings["output_col"]: labels}
    overwrite = bool(settings.get("overwrite_columns"))
    if overwrite:
        columns[settings["phenotype_col"]] = labels
    saved = save_result(settings, columns, frame[anchor], report)
    return {
        **saved,
        "clusters": clusters,
        "profile": profile,
        "where": where(labels, frame[roi]) if roi in frame.columns else {},
        "markers": list(config["markers"]),
        "phenotype_column": settings["phenotype_col"],
        "overwrite": overwrite,
        "grid": [int(v) for v in np.atleast_1d(grid)] if grid is not None else None,
        "k": k,
        "n_clusters": int(results[group].nunique()),
    }
