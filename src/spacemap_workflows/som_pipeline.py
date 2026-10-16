"""Tab 7's SOM clustering with no Streamlit calls: classic's "Run SOM Clustering Pipeline", step for step.

The steps and defaults are classic Tab 7's (``tabs/tab7_phenotyping/som_clustering.py``), in order:

1. Keep the cells passing the optional metadata filter; drop cells missing a marker value.
2. Normalise the markers with Tab 7's flexible normalisation (marker-wise z-score by default), within
   each batch when the table has the batch column; corrected markers (``_corr``, arcsinh values) are
   first taken back with sinh x cofactor, as classic does.
3. Keep the cells whose low-expression score is at or below the cut apart: they form the low-expression
   group (-1), which classic calls "Others" by default.
4. Scale the normalised values, train Tab 7's SOM (``ph_som_utils.train_som_advanced``: the grid, mode,
   subsample and early stop as set), then merge the SOM nodes into meta-clusters
   (``characterize_som_clusters`` and ``meta_cluster_clusters``).

``compute`` returns every cell's SOM node and meta-cluster, with the normalised columns.
"""

import numpy as np
import pandas as pd

LOW_EXPRESSION = -1
# Classic's defaults (som_clustering.py, its "som_config").
DEFAULTS = {
    "markers": [],
    "cell_id_col": "cell_id",
    "batch_id_col": "batch_id",
    "enable_filter": False,
    "filter_col": "tissue",
    "filter_val": "1",
    "auto_grid": False,
    "grid_dim": 30,
    "target_cells_per_node": 1800,
    "min_grid": 20,
    "max_grid": 70,
    "mode": "high_quality",
    "subsample_size": 150000,
    "subsample_method": "geosketch",
    "early_stop": True,
    "early_stop_min_improvement": 0.01,
    "early_stop_min_epoch_fraction": 0.3,
    "random_seed": 42,
    "n_meta_target": 45,
    "meta_method": "average",
    "meta_metric": "correlation",
    "norm_z1": True,
    "norm_z2": False,
    "norm_cdf": False,
    "norm_nlog": False,
    "norm_sinh": True,
    "enable_low_expr_filter": True,
    "low_expr_metric": "mean_zscore",
    "low_expr_threshold": -0.75,
    "output_col": "phenotype_som",
    "cluster_col": "som_cluster",
}
# Classic's SOM modes.
MODES = {
    "fast": {"iter_factor": 0.6, "start_sigma": 1.3, "lr": 0.08, "batch_size": 4096},
    "balanced": {"iter_factor": 1.0, "start_sigma": 1.5, "lr": 0.10, "batch_size": 4096},
    "high_quality": {"iter_factor": 1.5, "start_sigma": 1.8, "lr": 0.12, "batch_size": 3072},
}


def normalise(frame, config, markers, cofactor):
    """Tab 7's normalisation for clustering: (the frame with the normalised columns, their names)."""
    from spacemap import ph_leiden_utils

    source = frame
    if config["norm_sinh"]:
        source = frame.copy()
        for marker in markers:
            if marker.endswith("_corr") and marker in source.columns:
                source[marker] = np.sinh(source[marker]) * cofactor
    batch = config.get("batch_id_col") if config.get("batch_id_col") in source.columns else None
    subset = ph_leiden_utils.apply_flexible_normalization(
        df=source,
        marker_cols=markers,
        group_cols=[batch] if batch else [],
        suffix="_norm",
        apply_z1=config["norm_z1"],
        apply_z2=config["norm_z2"],
        apply_cdf=config["norm_cdf"],
        apply_nlog=config["norm_nlog"],
    )
    columns = list(subset.columns)
    found = pd.concat([frame, subset], axis=1)
    found[columns] = found[columns].fillna(0)
    return found, columns


def set_low_apart(found, config, markers, columns):
    """(cells to cluster, low-expression cells) as Tab 7 screens them."""
    from spacemap_gui.tabs.tab7_phenotyping.common import compute_low_expression_score
    from spacemap_gui.tabs.tab7_phenotyping.som_clustering import _coerce_som_low_expression_threshold

    scores = compute_low_expression_score(
        df=found, raw_marker_cols=markers, norm_marker_cols=columns, metric=config["low_expr_metric"]
    )
    if not config["enable_low_expr_filter"]:
        return found, found.iloc[0:0]
    cut = _coerce_som_low_expression_threshold(config["low_expr_threshold"], config["low_expr_metric"])
    low = scores <= cut
    return found[~low].copy(), found[low].copy()


def prepare(frame, config, cofactor, report):
    """Steps 1 to 3: (cells to cluster, low-expression cells, normalised column names, cell ID column)."""
    from spacemap_gui.tabs.tab7_phenotyping.common import (
        apply_data_filter,
        ensure_cell_id_anchor,
        normalize_optional_column_name,
    )

    markers = list(config["markers"])
    frame, cell_col, _ = ensure_cell_id_anchor(frame, normalize_optional_column_name(config.get("cell_id_col")))
    frame = apply_data_filter(
        frame,
        {
            "enable_filter": config.get("enable_filter", False),
            "filter_col": config.get("filter_col", "tissue"),
            "filter_val": config.get("filter_val", "1"),
        },
        logger_fn=lambda text: report(str(text)[:200], 0.1),
    )
    missing = [m for m in markers if m not in frame.columns]
    if missing:
        raise ValueError(f"The table has no {', '.join(missing[:5])}.")
    frame = frame[~frame[markers].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)].copy()
    if frame.empty:
        raise ValueError("No cells have a value for every chosen marker.")
    report(f"Normalising {len(markers)} markers of {len(frame):,} cells…", 0.15)
    found, columns = normalise(frame, config, markers, cofactor)
    kept, low = set_low_apart(found, config, markers, columns)
    if kept.empty:
        raise ValueError("Every cell is in the low-expression group: lower the cut or turn it off.")
    return kept, low, columns, cell_col


def compute(frame, config, cofactor, report):
    """Classic's SOM run on *frame*: every cell with ``som_cluster`` and ``meta_cluster`` (-1: low expression)."""
    from sklearn.preprocessing import StandardScaler

    from spacemap import ph_som_utils

    config = {**DEFAULTS, **config}
    kept, low, columns, cell_col = prepare(frame, config, cofactor, report)
    nodes = int(config["min_grid"]) ** 2 if config["auto_grid"] else int(config["grid_dim"]) ** 2
    if len(kept) < nodes:
        raise ValueError(
            f"Only {len(kept):,} cells are left to cluster, fewer than the {nodes:,} SOM nodes: use a smaller grid."
        )
    scaled = StandardScaler().fit_transform(kept[columns].values)
    adaptive = {"auto": bool(config["auto_grid"]), "mode": config["mode"], "base_epochs": 80}
    if config["auto_grid"]:
        adaptive.update(
            {
                "target_cells_per_node": config["target_cells_per_node"],
                "min_grid": config["min_grid"],
                "max_grid": config["max_grid"],
                "aspect": 1.0,
            }
        )
        side = 50
    else:
        side = int(config["grid_dim"])
    report(f"Training the SOM on {len(kept):,} cells…", 0.3)
    trained = ph_som_utils.train_som_advanced(
        clustering_data_scaled=scaled,
        data_df=kept,
        cell_id_column=cell_col,
        adaptive_config=adaptive,
        mode_params=MODES[config["mode"]],
        som_x_dim=side,
        som_y_dim=side,
        subsample_size=int(config["subsample_size"]),
        subsample_method=config["subsample_method"],
        early_stop=bool(config["early_stop"]),
        early_stop_min_improvement=float(config["early_stop_min_improvement"]),
        early_stop_min_epoch_fraction=float(config["early_stop_min_epoch_fraction"]),
        random_seed=int(config["random_seed"]),
        eval_size=min(4000, len(scaled)),
        second_stage_scaling=False,
        qe_scaling_strategy="none",
    )
    kept = kept.drop(columns=["som_cluster"], errors="ignore").merge(
        trained["som_clusters_df"], on=cell_col, how="left"
    )
    report(f"Merging the SOM nodes into {int(config['n_meta_target'])} meta-clusters…", 0.8)
    described = ph_som_utils.characterize_som_clusters(kept, kept["som_cluster"].values, columns)
    try:
        _, nodes_to_meta = ph_som_utils.meta_cluster_clusters(
            described.copy(),
            target_n=int(config["n_meta_target"]),
            method=config["meta_method"],
            metric=config["meta_metric"],
        )
    except ValueError as error:
        raise ValueError(
            f"The SOM nodes could not be compared ({config['meta_metric']} distance): too few markers, or markers "
            f"that rise and fall together. Choose more markers, or another merge method in classic Tab 7. ({error})"
        ) from error
    kept["meta_cluster"] = kept["som_cluster"].map(nodes_to_meta)
    if len(low):
        low = low.copy()
        low["som_cluster"] = LOW_EXPRESSION
        low["meta_cluster"] = LOW_EXPRESSION
    results = pd.concat([kept, low], axis=0).sort_index(kind="stable") if len(low) else kept
    return results, columns, cell_col, trained.get("grid_size")
