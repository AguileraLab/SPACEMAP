"""Studio's neighbourhoods (Tab 8): Tab 8's cellular neighbourhoods as a job, then name them.

Each cell's neighbourhood is the mix of phenotypes among its k nearest cells in
the same slide and tissue region (Tab 8's ``_compute_neighborhood_features``),
grouped into neighbourhoods with the MiniBatchKMeans settings Tab 8 uses. The
settings are Tab 8's own keys (tab8_nb_*). The job saves every cell's
neighbourhood in ``<name>_cn``; naming the neighbourhoods saves ``<name>``.
Explicit overwrite replaces both outputs and resets old names to the new CN
groups. Studio saves columns beside the combined table, as its other results.
"""

import re

import numpy as np
import pandas as pd

# Studio setting -> (Tab 8 key, default, lowest, highest); None limits for text and switches.
KEYS = {
    "phenotype_col": ("tab8_nb_pheno_col", "", None, None),
    "batch_col": ("tab8_nb_batch_col", "batch_id", None, None),
    "roi_col": ("tab8_nb_roi_col", "tissue_roi", None, None),
    "k_neighbors": ("tab8_nb_k", 10, 3, 100),
    "n_neighborhoods": ("tab8_nb_kmeans_k", 10, 2, 50),
    "random_state": ("tab8_nb_random", 42, 0, 100000),
    "min_cells": ("tab8_nb_min_cells", 20, 5, 5000),
    "normalize": ("tab8_nb_scale", True, None, None),
    "output_col": ("tab8_nb_out_col", "neighborhood", None, None),
}
OUTPUT = re.compile(r"neighbou?rhoods?(_[A-Za-z0-9_]{1,100})?")
RESULT_OUTPUT = re.compile(OUTPUT.pattern + r"(_cn)?")


def _tab8(config):
    return (config.get("tab_configurations") or {}).get("tab8_analysis") or {}


def values(config):
    saved = _tab8(config)
    return {
        name: saved.get(key, default) if saved.get(key) not in (None, "None") else default
        for name, (key, default, *_) in KEYS.items()
    }


def check(changes):
    if not isinstance(changes, dict) or set(changes) - set(KEYS):
        raise ValueError("Unknown neighbourhood setting.")
    changes = dict(changes)
    if isinstance(changes.get("output_col"), str):
        changes["output_col"] = changes["output_col"].strip()
    for name, value in changes.items():
        _, default, low, high = KEYS[name]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"Choose on or off for {name}.")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be a whole number from {low} to {high}.")
        elif name == "output_col":
            if not isinstance(value, str) or not OUTPUT.fullmatch(value):
                raise ValueError(
                    "Name the result neighborhood, or neighborhood_ followed by letters, numbers or underscores."
                )
        elif not isinstance(value, str) or len(value) > 256:
            raise ValueError(f"Choose a column for {name}.")
    return changes


def save(root, changes, expected):
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    changes = check(changes)
    with project_lock(root):
        config = read_config(root)
        if expected != revision(values(config)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        for name, value in changes.items():
            persistence.update_stored_config_key(config, KEYS[name][0], value)
        persistence.save_project_config(str(root), config)
        return values(read_config(root))


def run(settings, folder, report):
    """The job: Tab 8's neighbourhood features and clustering, saved as ``<name>_cn``."""
    from sklearn.cluster import MiniBatchKMeans

    from spacemap import dataio
    from spacemap_gui.tabs.tab8_analysis.neighborhoods import _build_profile_summary, _compute_neighborhood_features

    from .tasks import save_result, validate_table

    validate_table(settings)
    chosen = settings["values"]
    anchor_column = settings.get("cell_id_col", "cell_id")
    grouping = [chosen[name] for name in ("batch_col", "roi_col") if chosen[name]]
    report("Reading the combined table…", 0.05)
    images = settings.get("images") or []
    frame = dataio.load(
        settings["table"],
        columns=list(dict.fromkeys([anchor_column, "x_centroid", "y_centroid", chosen["phenotype_col"], *grouping,
                                    *([settings["image_col"]] if images else [])])),
    )
    work = frame
    if images:  # only the chosen images' cells; the others get no neighbourhood (the column still covers every row)
        work = frame[frame[settings["image_col"]].astype(str).isin(images)]
        if work.empty:
            raise ValueError("None of the chosen images is in the combined table.")
    report("Counting each cell's neighbours…", 0.2)
    features, columns, _ = _compute_neighborhood_features(
        df=work,
        phenotype_col=chosen["phenotype_col"],
        batch_col=chosen["batch_col"],
        roi_col=chosen["roi_col"],
        k_neighbors=int(chosen["k_neighbors"]),
        min_cells_per_region=int(chosen["min_cells"]),
    )
    grouped = features.dropna(subset=columns).copy() if columns else pd.DataFrame()
    if grouped.empty:
        raise ValueError("No region has enough cells with a phenotype for neighbourhoods.")
    # Tab 8's clustering, as its Run Neighborhood Detection does it.
    x = grouped[columns].to_numpy(dtype=float)
    if chosen["normalize"]:
        spread = np.nanstd(x, axis=0)
        spread[spread <= 0] = 1.0
        x = (x - np.nanmean(x, axis=0)) / spread
    count = int(min(int(chosen["n_neighborhoods"]), len(grouped)))
    if count < 2:
        raise ValueError("Need at least two cells to group into neighbourhoods.")
    report("Grouping the neighbourhoods…", 0.6)
    model = MiniBatchKMeans(n_clusters=count, random_state=int(chosen["random_state"]), batch_size=1024, n_init=10)
    grouped["CN_cluster"] = model.fit_predict(x)
    profiles = grouped.groupby("CN_cluster", dropna=False)[columns].mean()
    profiles.columns = [name.replace("freq_", "") for name in profiles.columns]
    summary = _build_profile_summary(profiles, grouped)
    labels = pd.Series(None, index=frame.index, dtype=object)
    labels.loc[grouped.index] = grouped["CN_cluster"].map(lambda value: "CN " + str(int(value)))
    outputs = {settings["output_col"]: labels}
    overwrite = bool(settings.get("overwrite_columns"))
    if overwrite:
        # Membership can change on rerun; earlier descriptive names are now stale.
        outputs[settings["names_col"]] = labels
    saved = save_result(settings, outputs, frame[anchor_column], report)
    clusters = [
        {
            "id": int(row.CN_cluster),
            "name": "CN " + str(int(row.CN_cluster)),
            "cells": int(row.n_cells),
            "top_markers": [part.rsplit(" (", 1)[0] for part in str(row.top_phenotypes).split(", ") if part],
        }
        for row in summary.itertuples()
    ]
    from .leiden import heatmap, where

    return {
        **saved,
        "clusters": clusters,
        "where": where(labels, frame[chosen["batch_col"]]) if chosen["batch_col"] in frame.columns else {},
        "profile": {**heatmap(
            profiles,
            profiles.columns,
            ["CN " + str(int(value)) for value in profiles.index],
            "Share of each phenotype among the neighbours, compared across the neighbourhoods",
            raw=profiles,
        ), "aggregation": "mean", "raw_label": "Mean neighbourhood share"},
        "markers": settings["markers"],
        "phenotype_column": settings["names_col"],
        "overwrite": overwrite,
        "source_column": chosen["phenotype_col"],
        "images": images,
    }
