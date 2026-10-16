"""Leiden computation and durable output, with no Streamlit calls during a run."""

import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from spacemap import ph_leiden_utils
from spacemap_gui.tabs.tab7_phenotyping.common import (
    cluster_sort_key,
    compute_low_expression_score,
    ensure_cell_id_anchor,
    finish_phenotyping_run_audit,
    get_low_expression_default_threshold,
    get_low_expression_metric_label,
    normalize_cluster_id,
    normalize_optional_column_name,
    write_dataframe_parquet_atomic,
)
from spacemap_workflows.leiden_job import stage, write_json

LEIDEN_LOW_EXPR_CLUSTER_ID = -1
LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE = "Others"
LEIDEN_LOW_EXPR_LABEL = "Low expression bucket"


def run_pipeline(request, job_dir, logger):
    """Run from an immutable request; audit success/failure in the worker."""
    audit_result = {}
    try:
        result = _compute(request, Path(job_dir), logger, audit_result)
    except BaseException as exc:
        finish_phenotyping_run_audit(
            request.get("audit"),
            status="failed" if isinstance(exc, Exception) else "aborted",
            input_rows=audit_result.get("input_rows"),
            output_column=request["settings"]["output_col_name"],
            cluster_column="leiden_cluster",
            error=f"{type(exc).__name__}: {exc}",
            run_logger=logger,
        )
        raise
    finish_phenotyping_run_audit(
        request.get("audit"),
        status="success",
        output_path=result["leiden_temp_file"],
        input_rows=audit_result["input_rows"],
        output_rows=audit_result["output_rows"],
        output_column=request["settings"]["output_col_name"],
        cluster_column="leiden_cluster",
        phenotype_map=result["leiden_phenotype_map"],
        extra={
            "cluster_count": result["leiden_n_clusters"],
            "low_expression_cells": result["leiden_low_expr_count"],
        },
        run_logger=logger,
    )
    return result


def _compute(request, job_dir, logger, audit_result):
    project_root = Path(request["project_root"])
    settings = request["settings"]
    full_input_path = settings["full_input_path"]
    markers_selected = settings["markers_selected"]
    cell_id_col = settings["cell_id_col"]
    batch_id_col = settings["batch_id_col"]
    enable_filter = settings["enable_filter"]
    filter_col = settings["filter_col"]
    filter_val = settings["filter_val"]
    norm_z1 = settings["norm_z1"]
    norm_z2 = settings["norm_z2"]
    norm_cdf = settings["norm_cdf"]
    norm_nlog = settings["norm_nlog"]
    norm_sinh = settings["norm_sinh"]
    cofactor = settings["cofactor"]
    low_expr_metric = settings["low_expr_metric"]
    enable_low_expr_filter = settings["enable_low_expr_filter"]
    low_expr_threshold = settings["low_expr_threshold"]
    resolution = settings["resolution"]
    n_neighbors = settings["n_neighbors"]
    n_pcs = settings["n_pcs"]
    use_intelligent = settings["use_intelligent"]
    subsample_size = settings["subsample_size"]
    random_seed = settings["random_seed"]
    compute_umap = settings["compute_umap"]
    meta_clusters = settings["meta_clusters"]
    use_gpu = settings["use_gpu"]
    fresh_assignment_mode = settings["fresh_assignment_mode"]

    # Step 1: Load Data
    logger.info("Step 1: Loading data from %s", full_input_path)
    df = pd.read_csv(full_input_path, low_memory=False)
    audit_result["input_rows"] = len(df)
    df, cell_id_anchor_col, using_internal_cell_anchor = ensure_cell_id_anchor(
        df,
        normalize_optional_column_name(cell_id_col),
    )
    logger.info(f"Loaded {len(df):,} rows from input file")

    # Step 2: Apply Filtering (match notebook FILTER_CRITERIA)
    if enable_filter and filter_col and filter_val:
        if filter_col in df.columns:
            original_len = len(df)
            try:
                # Try numeric comparison first
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    val_num = float(filter_val)
                    if pd.api.types.is_integer_dtype(df[filter_col]):
                        val_num = int(val_num)
                    df = df[df[filter_col] == val_num]
                else:
                    df = df[df[filter_col].astype(str) == str(filter_val)]
            except (TypeError, ValueError):
                df = df[df[filter_col].astype(str) == str(filter_val)]

            logger.info(
                f"Filter applied: {filter_col}=={filter_val}: {original_len:,} -> {len(df):,} cells"
            )
        else:
            logger.warning(
                f"Filter column '{filter_col}' not found. Using full dataset."
            )
    else:
        logger.info("No filtering applied. Using full dataset.")

    # Validate Cell ID Column
    if using_internal_cell_anchor:
        logger.info(
            "No valid Cell ID column selected/found. Using an internal row-order anchor for this Leiden run."
        )
    else:
        n_unique = df[cell_id_anchor_col].nunique()
        logger.info(f"Cell ID column '{cell_id_anchor_col}': {n_unique:,} unique cells")

    # Check marker availability
    missing_markers = [m for m in markers_selected if m not in df.columns]
    if missing_markers:
        raise ValueError(f"Missing markers in dataset: {missing_markers}")

    # Drop rows containing NaN/Inf in selected markers
    if markers_selected:
        original_len_nan = len(df)
        valid_mask = ~df[markers_selected].apply(
            pd.to_numeric, errors="coerce"
        ).isna().any(axis=1)
        df = df[valid_mask].copy()
        logger.info(
            f"Filtered out {original_len_nan - len(df):,} cells containing NaN/Inf in selected markers. Remaining: {len(df):,} cells."
        )
        if len(df) == 0:
            raise ValueError(
                "No cells remain after filtering out NaN/Inf values in selected markers."
            )

    logger.info(f"Processing {len(df):,} cells with {len(markers_selected)} markers")
    logger.info(
        f"Markers: {', '.join(markers_selected[:5])}{'...' if len(markers_selected) > 5 else ''}"
    )

    # Step 3: Normalization (Match Notebook Pipeline)
    with stage(logger, "Normalization"):
        logger.info("Step 2: Applying normalization (notebook synchronized)")
        logger.info(
            f"Normalization flags: Z1={norm_z1}, Z2={norm_z2}, CDF={norm_cdf}, NLog={norm_nlog}, Sinh={norm_sinh}"
        )

        df_input = df
        if norm_sinh:
            df_input = df.copy()
            for m in markers_selected:
                if m.endswith("_corr") and m in df_input.columns:
                    df_input[m] = np.sinh(df_input[m]) * cofactor

        # Use ph_leiden_utils to ensure Batch-WISE normalization (matching notebook)
        if ph_leiden_utils:
            # Detect batch column for grouping
            batch_col = batch_id_col if batch_id_col in df_input.columns else None
            group_cols = [batch_col] if batch_col else []

            if not group_cols:
                logger.warning(
                    "No 'batch_id' column found (or selected) - performing global normalization (Notebook uses batch-wise)"
                )
            else:
                logger.info(f"Normalizing with grouping: {group_cols}")

            norm_subset = ph_leiden_utils.apply_flexible_normalization(
                df=df_input,
                marker_cols=markers_selected,
                group_cols=group_cols,
                suffix="_norm",
                apply_z1=norm_z1,
                apply_z2=norm_z2,
                apply_cdf=norm_cdf,
                apply_nlog=norm_nlog,
            )

            # Merge back results (on the original raw/un-sinh scaled df)
            norm_data = df.copy()
            norm_cols = list(norm_subset.columns)
            # Efficiently assign new columns
            norm_data = pd.concat([norm_data, norm_subset], axis=1)
        else:
            # Fallback if utils not available (should not happen in correct env)
            logger.warning(
                "ph_leiden_utils not found, falling back to global normalization"
            )
            norm_data = df.copy()
            norm_cols = []
            for m in markers_selected:
                col_data = df_input[m].fillna(0).copy()
                if norm_nlog:
                    col_data = np.log1p(col_data)
                if norm_cdf:
                    col_data = col_data.rank(pct=True)
                if norm_z1:
                    std_val = col_data.std()
                    col_data = (col_data - col_data.mean()) / (
                        std_val if std_val > 1e-9 else 1
                    )
                col_data = col_data.fillna(0)
                norm_col = f"{m}_norm"
                norm_data[norm_col] = col_data
                norm_cols.append(norm_col)

        # Ensure matrix is clean
        norm_data[norm_cols] = norm_data[norm_cols].fillna(0)
        logger.info(f"Normalization complete: {len(norm_cols)} markers normalized")

        raw_marker_values = (
            norm_data[markers_selected].apply(pd.to_numeric, errors="coerce").fillna(0)
        )
        norm_data["leiden_marker_sum_raw"] = raw_marker_values.sum(axis=1)
        low_expr_scores = compute_low_expression_score(
            df=norm_data,
            raw_marker_cols=markers_selected,
            norm_marker_cols=norm_cols,
            metric=low_expr_metric,
        )
        norm_data["leiden_low_expr_score"] = low_expr_scores
        low_expr_data = norm_data.iloc[0:0].copy()
        low_expr_count = 0

        # Apply low-expression filtering using the shared cell-level score.
        if enable_low_expr_filter:
            pre_filter = len(norm_data)
            low_expr_thresh_val = (
                float(low_expr_threshold)
                if low_expr_threshold is not None
                else float(get_low_expression_default_threshold(low_expr_metric))
            )
            low_expr_mask = low_expr_scores <= low_expr_thresh_val
            low_expr_count = int(low_expr_mask.sum())
            low_expr_data = norm_data[low_expr_mask].copy()
            norm_data = norm_data[~low_expr_mask].copy()
            df = norm_data  # Update main df reference to filtered version
            logger.info(
                "Low expression filter (%s <= %.4f): %s/%s cells preserved in the low-expression bucket; %s cells sent to Leiden clustering",
                get_low_expression_metric_label(low_expr_metric),
                low_expr_thresh_val,
                f"{low_expr_count:,}",
                f"{pre_filter:,}",
                f"{len(norm_data):,}",
            )
            if norm_data.empty:
                raise ValueError(
                    "All cells fell into the low-expression bucket. Lower the expression threshold or disable low-expression screening."
                )
        else:
            logger.info("Low expression filtering is disabled.")
            df = norm_data

    # Step 4: Leiden Clustering (Intelligent or Direct Scanpy)
    with stage(logger, "Leiden Clustering"):
        logger.info("Step 3: Performing Leiden clustering")
        logger.info(
            f"Parameters: resolution={resolution}, neighbors={n_neighbors}, PCs={n_pcs}"
        )
        logger.info(
            f"Dataset size: {len(df):,} cells | Intelligent mode: {use_intelligent}"
        )

        leiden_params = {
            "n_pcs": n_pcs,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
            "subsample_size": subsample_size,
            "random_seed": random_seed,
            "compute_umap": compute_umap,
            "meta_clusters": int(meta_clusters),
            "use_gpu": use_gpu,
        }

        # Check dataset size and intelligent clustering flag
        if use_intelligent and len(df) > 50000 and ph_leiden_utils is not None:
            logger.info(
                "Using intelligent_leiden_clustering (geometric sketching for large datasets)"
            )
            cluster_labels, umap_coords = ph_leiden_utils.intelligent_leiden_clustering(
                norm_data, norm_cols, leiden_params, logger
            )
            df["leiden_cluster"] = cluster_labels
            if umap_coords is not None:
                df["umap_x"] = umap_coords[:, 0]
                df["umap_y"] = umap_coords[:, 1]
        else:
            if use_intelligent and len(df) <= 50000:
                logger.info("Dataset ≤50k cells: using direct Scanpy approach")
            else:
                logger.info("Using direct Scanpy approach (standard Leiden)")

            import scanpy as sc

            adata = sc.AnnData(norm_data[norm_cols].fillna(0).values)
            adata.var_names = norm_cols
            adata.obs_names_make_unique()

            # PCA
            max_pcs = min(adata.shape) - 1
            actual_n_pcs = min(n_pcs, max_pcs)
            sc.tl.pca(
                adata,
                n_comps=actual_n_pcs,
                svd_solver="randomized",
                random_state=random_seed,
            )

            # Neighbors
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                n_pcs=actual_n_pcs,
                random_state=random_seed,
            )

            # Leiden
            sc.tl.leiden(
                adata,
                resolution=resolution,
                random_state=random_seed,
                key_added="leiden",
            )

            # UMAP (visualisation only, so optional)
            if compute_umap:
                sc.tl.umap(adata, random_state=random_seed)

            # Assign results
            df["leiden_cluster"] = adata.obs["leiden"].astype(int).values
            if compute_umap:
                df["umap_x"] = adata.obsm["X_umap"][:, 0]
                df["umap_y"] = adata.obsm["X_umap"][:, 1]

    df_results = df.copy()
    df_results["leiden_low_expression"] = False
    df_results["leiden_filter_status"] = "clustered"

    if low_expr_count:
        low_expr_results = low_expr_data.copy()
        low_expr_results["leiden_cluster"] = LEIDEN_LOW_EXPR_CLUSTER_ID
        low_expr_results["leiden_low_expression"] = True
        low_expr_results["leiden_filter_status"] = "low_expression"
        if "umap_x" in df_results.columns:
            low_expr_results["umap_x"] = np.nan
            low_expr_results["umap_y"] = np.nan
        df_results = pd.concat([df_results, low_expr_results], axis=0).sort_index(
            kind="stable"
        )
        logger.info(
            "Preserved %s low-expression cells as %s with default phenotype '%s'.",
            f"{low_expr_count:,}",
            LEIDEN_LOW_EXPR_LABEL,
            LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE,
        )

    n_clusters = df["leiden_cluster"].nunique()
    logger.info(f"Clustering complete: {n_clusters} clusters identified")
    logger.info("=" * 80)

    # Commit results before telling the UI that the run completed. Unique run
    # paths preserve earlier results even if a later run fails.
    protect_output = getattr(logger, "protect_output", None)
    if protect_output is not None:
        protect_output()
    temp_file = job_dir / "results.parquet"
    logger.info("Saving Leiden results to %s", temp_file)
    write_dataframe_parquet_atomic(df_results, temp_file)
    initial_phenotype_map = {
        normalize_cluster_id(cluster_id): "Unassigned"
        for cluster_id in sorted(
            df["leiden_cluster"].dropna().unique().tolist(), key=cluster_sort_key
        )
        if normalize_cluster_id(cluster_id) is not None
    }
    if low_expr_count and not fresh_assignment_mode:
        initial_phenotype_map[LEIDEN_LOW_EXPR_CLUSTER_ID] = (
            LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE
        )
    file_hash = hashlib.md5(str(full_input_path).encode()).hexdigest()
    try:
        cache_dir = project_root / "outputs" / "phenotyping" / "leiden_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_dataframe_parquet_atomic(
            df_results, cache_dir / f"res_{file_hash}.parquet"
        )
        write_json(
            cache_dir / f"meta_{file_hash}.json",
            {
                "markers": markers_selected,
                "norm_cols": norm_cols,
                "n_clusters": int(n_clusters),
                "timestamp": time.time(),
                "source_file": str(full_input_path),
            },
        )
    except Exception as exc:
        logger.warning(
            "Could not update legacy Leiden cache (run results are saved): %s", exc
        )
    audit_result["output_rows"] = len(df_results)
    return {
        "leiden_temp_file": str(temp_file),
        "leiden_markers": markers_selected,
        "leiden_norm_cols": norm_cols,
        "leiden_n_clusters": int(n_clusters),
        "leiden_phenotype_map": initial_phenotype_map,
        "leiden_has_run": True,
        "leiden_run_input_path": full_input_path,
        "leiden_run_id": job_dir.name,
        "leiden_loaded_hash": file_hash,
        "leiden_loaded_source_path": full_input_path,
        "leiden_low_expr_cluster_id": LEIDEN_LOW_EXPR_CLUSTER_ID
        if low_expr_count
        else None,
        "leiden_low_expr_count": int(low_expr_count),
        "leiden_low_expr_metric": low_expr_metric if enable_low_expr_filter else None,
        "leiden_low_expr_threshold": float(low_expr_threshold)
        if enable_low_expr_filter
        else None,
        "leiden_low_expr_default_pheno": LEIDEN_LOW_EXPR_DEFAULT_PHENOTYPE,
    }
