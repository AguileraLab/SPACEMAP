"""Tab 7's VORTEX clustering with no Streamlit calls: classic's "Run Vortex Clustering", step for step.

The steps and defaults are classic Tab 7's (``tabs/tab7_phenotyping/vortex_clustering.py``), in order:

1. Keep the cells whose filter column equals the filter value (when the filter is on); set aside cells
   missing a marker value.
2. Normalise the markers with Tab 7's flexible normalisation (within the batch column when the table has
   it); corrected markers (``_corr``) are first taken back with sinh x cofactor.
3. Keep the dim cells (low-expression score at or below the cut) apart as the low-expression group (-1).
4. When "choose K" is on, scan the candidate K values on a sketch of the cells and take the one
   ``recommend_vortex_k`` picks for the target number of clusters; else use the K given.
5. Run ``intelligent_vortex_clustering`` (sketching, KNN back-filling). UMAP is not computed: it does not
   change the clusters (classic's default is off too).

``compute`` returns every clustered cell with ``vortex_cluster`` (-1: low expression), the normalised column
names, the cell ID column and the K used.
"""

import numpy as np
import pandas as pd

LOW_EXPRESSION = -1
# Classic's defaults (vortex_clustering.py, cached_params.get(key, default)).
DEFAULTS = {
    "markers": [],
    "cell_id_col": "cell_id",
    "batch_id_col": "",
    "enable_filter": False,
    "filter_col": "tissue",
    "filter_val": "1",
    "auto_select_k": True,
    "vortex_k": 35,
    "n_clusters_target": 45,
    "k_candidates_text": "10, 15, 20, 25, 30, 35, 45, 60",
    "k_scan_sketch_size": 25000,
    "subsample_size": 150000,
    "subsample_method": "geosketch",
    "use_gpu": True,
    "max_geosketch_input": 0,
    "meta_clusters": 0,
    "knn_neighbors": 15,
    "min_cluster_size": 1,
    "random_seed": 42,
    "norm_z1": True,
    "norm_z2": False,
    "norm_cdf": False,
    "norm_nlog": False,
    "norm_sinh": True,
    "enable_low_expr_filter": True,
    "low_expr_metric": "mean_zscore",
    "low_expr_threshold": None,
    "output_col": "phenotype_vortex",
}


def _candidates(text):
    values = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit() and int(part) > 1:
            values.append(int(part))
    return sorted(set(values)) or [10, 15, 20, 25, 30, 35, 45, 60]


def _filter(frame, config):
    column, value = config.get("filter_col"), config.get("filter_val")
    if not (config.get("enable_filter") and column and value is not None and column in frame.columns):
        return frame
    try:  # classic compares numbers as numbers, else as text
        if pd.api.types.is_numeric_dtype(frame[column]):
            number = float(value)
            if pd.api.types.is_integer_dtype(frame[column]):
                number = int(number)
            return frame[frame[column] == number]
    except (TypeError, ValueError):
        pass
    return frame[frame[column].astype(str) == str(value)]


def compute(frame, config, cofactor, report, logger=None):
    """Classic's VORTEX run on *frame*."""
    from spacemap import ph_vortex_utils
    from spacemap_gui.tabs.tab7_phenotyping.common import (
        compute_low_expression_score,
        ensure_cell_id_anchor,
        get_low_expression_default_threshold,
        normalize_optional_column_name,
    )

    config = {**DEFAULTS, **{key: value for key, value in config.items() if value is not None}}
    markers = list(config["markers"])
    frame, cell_col, _ = ensure_cell_id_anchor(frame, normalize_optional_column_name(config.get("cell_id_col")))
    frame = _filter(frame, config)
    missing = [m for m in markers if m not in frame.columns]
    if missing:
        raise ValueError(f"The table has no {', '.join(missing[:5])}.")
    numbers = frame[markers].apply(pd.to_numeric, errors="coerce")
    frame = frame[~(numbers.isna().any(axis=1) | np.isinf(numbers).any(axis=1))].copy()
    if frame.empty:
        raise ValueError("No cells have a value for every chosen marker.")
    report(f"Normalising {len(markers)} markers of {len(frame):,} cells…", 0.12)
    source = frame
    if config["norm_sinh"]:
        source = frame.copy()
        for marker in markers:
            if marker.endswith("_corr"):
                source[marker] = np.sinh(source[marker]) * cofactor
    batch = config.get("batch_id_col")
    subset = ph_vortex_utils.apply_flexible_normalization(
        df=source,
        marker_cols=markers,
        group_cols=[batch] if batch and batch in source.columns else [],
        suffix="_norm",
        apply_z1=config["norm_z1"],
        apply_z2=config["norm_z2"],
        apply_cdf=config["norm_cdf"],
        apply_nlog=config["norm_nlog"],
    )
    columns = list(subset.columns)
    data = pd.concat([frame, subset], axis=1)
    data[columns] = data[columns].fillna(0)
    scores = compute_low_expression_score(
        df=data, raw_marker_cols=markers, norm_marker_cols=columns, metric=config["low_expr_metric"]
    )
    low = data.iloc[0:0]
    if config["enable_low_expr_filter"]:
        cut = config.get("low_expr_threshold")
        cut = float(cut) if cut is not None else float(get_low_expression_default_threshold(config["low_expr_metric"]))
        dim = (scores <= cut).to_numpy()
        low, data = data[dim].copy(), data[~dim].copy()
    if data.empty:
        raise ValueError("Every cell is in the low-expression group: lower the cut or turn it off.")
    k = int(config["vortex_k"])
    if config["auto_select_k"]:
        report("Choosing K: scanning " + ", ".join(map(str, _candidates(config["k_candidates_text"]))) + "…", 0.25)
        scan = ph_vortex_utils.evaluate_vortex_k_grid(
            df=data,
            norm_cols=columns,
            k_values=_candidates(config["k_candidates_text"]),
            sketch_size=int(config["k_scan_sketch_size"]),
            subsample_method=config["subsample_method"],
            min_cluster_size=int(config["min_cluster_size"]),
            random_seed=int(config["random_seed"]),
            logger=logger,
            use_gpu=bool(config["use_gpu"]),
            max_geosketch_input=int(config["max_geosketch_input"]),
        )
        k = int(
            ph_vortex_utils.recommend_vortex_k(results_df=scan, target_n_clusters=int(config["n_clusters_target"]))[
                "k_neighbors"
            ]
        )
    report(f"VORTEX clustering with K = {k} on {len(data):,} cells…", 0.45)
    assignments, _, _ = ph_vortex_utils.intelligent_vortex_clustering(
        df=data,
        norm_cols=columns,
        vortex_params={
            "k_neighbors": k,
            "subsample_size": int(config["subsample_size"]),
            "subsample_method": config["subsample_method"],
            "random_seed": int(config["random_seed"]),
            "knn_neighbors": int(config["knn_neighbors"]),
            "min_cluster_size": int(config["min_cluster_size"]),
            "generate_umap": False,
            "umap_neighbors": 15,
            "meta_clusters": int(config["meta_clusters"]),
            "use_gpu": bool(config["use_gpu"]),
            "max_geosketch_input": int(config["max_geosketch_input"]),
        },
        logger=logger,
    )
    data["vortex_cluster"] = np.asarray(assignments)
    if len(low):
        low = low.copy()
        low["vortex_cluster"] = LOW_EXPRESSION
        data = pd.concat([data, low], axis=0).sort_index(kind="stable")
    return data, columns, cell_col, k
