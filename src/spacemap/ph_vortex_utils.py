"""
Vortex (X-shift) phenotyping utilities for the SPACEMAP pipeline.

This module adapts the X-shift / Vortex density-gradient clustering approach to
the same workflow used by the existing Leiden and SOM notebooks: shared
normalization, cluster characterization, clustered heatmaps, interactive
phenotype assignment, QC plots, and unified-output writing.
"""

import os
from concurrent.futures import ProcessPoolExecutor

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import davies_bouldin_score, silhouette_score

warnings.filterwarnings("ignore")

try:
    import scanpy as sc
except ImportError:
    sc = None
    print("Warning: scanpy not installed. UMAP generation will be unavailable.")

try:
    import geosketch

    GEOSKETCH_AVAILABLE = True
except ImportError:
    GEOSKETCH_AVAILABLE = False
    geosketch = None
    print("Warning: geosketch not installed. Falling back to random subsampling.")

# GPU support, fast sketching, and the k-NN backfill are shared with the other
# phenotyping methods; see ph_accel_utils for the implementations.
from spacemap.ph_accel_utils import (  # noqa: F401  (re-exported for notebooks)
    GPU_AVAILABLE,
    _gpu_chunk_rows,
    geosketch_gs_fast,
    gpu_device_name,
    knn_backfill as _knn_backfill,
    knn_gpu,
    resolve_use_gpu as _resolve_use_gpu,
    torch,
)


def _log(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message, flush=True)


def apply_flexible_normalization(*args, **kwargs):
    from spacemap.ph_leiden_utils import apply_flexible_normalization as _shared_apply_flexible_normalization

    return _shared_apply_flexible_normalization(*args, **kwargs)


def load_or_compute_normalized(*args, **kwargs):
    from spacemap.ph_leiden_utils import load_or_compute_normalized as _shared_load_or_compute_normalized

    return _shared_load_or_compute_normalized(*args, **kwargs)


def characterize_clusters(data, marker_cols, clusters=None, cluster_col=None):
    """Shared cluster characterization wrapper for Vortex notebooks."""
    from spacemap.ph_leiden_utils import characterize_clusters as _shared_characterize_clusters

    return _shared_characterize_clusters(
        data=data,
        marker_cols=marker_cols,
        clusters=clusters,
        cluster_col=cluster_col,
    )


def create_cluster_heatmap(
    char_df,
    plots_dir,
    cluster_col,
    title="Cluster Expression Heatmap",
    filename="cluster_expression_heatmap.png",
):
    """Shared cluster heatmap wrapper for Vortex notebooks."""
    from spacemap.ph_leiden_utils import create_cluster_heatmap as _shared_create_cluster_heatmap

    return _shared_create_cluster_heatmap(
        char_df=char_df,
        plots_dir=plots_dir,
        cluster_col=cluster_col,
        title=title,
        filename=filename,
    )


def geometric_sketch_selection_numpy(pca_coords, n_select, random_seed):
    np.random.seed(random_seed)
    n_cells, n_dims = pca_coords.shape

    if n_select >= n_cells:
        return np.arange(n_cells)

    selected_indices = []
    n_bins_per_dim = max(2, int(np.sqrt(n_select)))

    for dim in range(min(3, n_dims)):  # Use first 3 PCs for binning
        pc_values = pca_coords[:, dim]
        bins = np.linspace(pc_values.min(), pc_values.max(), n_bins_per_dim + 1)

        for i in range(n_bins_per_dim):
            bin_mask = (pc_values >= bins[i]) & (pc_values < bins[i + 1])
            if i == n_bins_per_dim - 1:
                bin_mask = (pc_values >= bins[i]) & (pc_values <= bins[i + 1])

            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) > 0:
                n_from_bin = max(1, len(bin_indices) // n_bins_per_dim)
                sampled = np.random.choice(bin_indices, 
                                         size=min(n_from_bin, len(bin_indices)), 
                                         replace=False)
                selected_indices.extend(sampled)

    selected_indices = list(set(selected_indices))

    if len(selected_indices) > n_select:
        selected_indices = np.random.choice(selected_indices, size=n_select, replace=False)
    elif len(selected_indices) < n_select:
        remaining_indices = np.setdiff1d(np.arange(n_cells), selected_indices)
        n_additional = n_select - len(selected_indices)
        if len(remaining_indices) > 0:
            additional = np.random.choice(remaining_indices, 
                                        size=min(n_additional, len(remaining_indices)), 
                                        replace=False)
            selected_indices.extend(additional)

    return np.array(selected_indices[:n_select])


# Optional cap on how many rows the sketcher sees. This is a last-resort escape hatch
# for memory pressure, NOT a speed knob: geosketch's value to X-shift comes from
# flattening density, and that flattening scales with the input:output ratio. Capping
# the input close to the sketch size forces the sketcher to keep nearly every candidate
# cell, which reproduces the raw density distribution and collapses the clustering into
# one dominant cluster. `geosketch_gs_fast` sketches the full 5.6M-cell matrix in under
# 20s, so the default is no cap. 0/None disables it.
GEOSKETCH_MAX_INPUT_ROWS = 0


def geometric_sketch_selection(
    data_matrix,
    n_select,
    random_seed,
    method="geosketch",
    max_geosketch_input=GEOSKETCH_MAX_INPUT_ROWS,
    logger=None,
    use_fast_geosketch=True,
):
    """Density-aware or random subsampling for scalable X-shift execution."""
    rng = np.random.default_rng(random_seed)
    n_cells = data_matrix.shape[0]

    if n_select >= n_cells:
        return np.arange(n_cells)

    if method == "geosketch" and (use_fast_geosketch or (GEOSKETCH_AVAILABLE and geosketch is not None)):
        # Pre-sample so the sketcher never sees more rows than it needs. A uniform
        # sample still spans the same expression space, so the covering is preserved.
        candidate_indices = None
        if max_geosketch_input:
            cap = max(int(max_geosketch_input), int(n_select) * 2)
            if n_cells > cap:
                candidate_indices = np.sort(rng.choice(n_cells, cap, replace=False))
                _log(
                    logger,
                    f"Pre-sampling {cap:,} of {n_cells:,} cells before geometric sketching.",
                )

        working = data_matrix if candidate_indices is None else data_matrix[candidate_indices]

        if working.shape[1] > 10:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(10, working.shape[1]), random_state=random_seed)
            working = pca.fit_transform(working)

        if use_fast_geosketch:
            selected = geosketch_gs_fast(working, int(n_select), seed=random_seed)
        else:
            selected = np.asarray(geosketch.gs(working, n_select, replace=False, seed=random_seed))
        return selected if candidate_indices is None else candidate_indices[selected]

    elif method in ("geosketch_fast", "geosketch (fast)"):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, data_matrix.shape[1]), random_state=random_seed)
        data_matrix_pca = pca.fit_transform(data_matrix)
        return geometric_sketch_selection_numpy(data_matrix_pca, n_select, random_seed)

    return rng.choice(n_cells, n_select, replace=False)


def _safe_tree_query(tree, data_matrix, k_neighbors, query_workers=None):
    effective_k = min(int(k_neighbors) + 1, data_matrix.shape[0])
    if effective_k <= 1:
        return np.zeros((data_matrix.shape[0], 0)), np.zeros((data_matrix.shape[0], 0), dtype=int)

    workers = -1 if query_workers is None else max(1, int(query_workers))
    try:
        distances, indices = tree.query(data_matrix, k=effective_k, workers=workers)
    except TypeError:
        distances, indices = tree.query(data_matrix, k=effective_k)

    if effective_k == 2:
        distances = distances[:, np.newaxis] if distances.ndim == 1 else distances
        indices = indices[:, np.newaxis] if indices.ndim == 1 else indices

    return distances[:, 1:], indices[:, 1:]


def _resolve_pointer_roots(pointers):
    roots = np.asarray(pointers, dtype=int).copy()
    while True:
        next_roots = roots[roots]
        if np.array_equal(next_roots, roots):
            return roots
        roots = next_roots


def _merge_small_clusters(cluster_assignments, data_matrix, min_cluster_size, logger=None):
    if min_cluster_size <= 1:
        return cluster_assignments

    labels = np.asarray(cluster_assignments).copy()
    counts = pd.Series(labels).value_counts()
    small_labels = counts[counts < min_cluster_size].index.tolist()

    if not small_labels:
        return labels

    large_labels = counts[counts >= min_cluster_size].index.tolist()
    if not large_labels:
        return labels

    large_centroids = {
        label: data_matrix[labels == label].mean(axis=0)
        for label in large_labels
    }

    merged_cells = 0
    for label in small_labels:
        mask = labels == label
        if not mask.any():
            continue
        center = data_matrix[mask].mean(axis=0)
        target_label = min(
            large_labels,
            key=lambda other: np.linalg.norm(center - large_centroids[other]),
        )
        labels[mask] = target_label
        merged_cells += int(mask.sum())

    unique_labels = sorted(np.unique(labels))
    remap = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    remapped = np.array([remap[label] for label in labels], dtype=int)

    _log(
        logger,
        f"Merged {len(small_labels)} small clusters ({merged_cells:,} cells) using min_cluster_size={min_cluster_size}.",
    )
    return remapped


def _core_vortex_gpu_lowmem(data_matrix, k_neighbors, logger=None, device=0):
    """Memory-lean GPU density-gradient step for large sketches.

    Identical maths to :func:`_core_vortex_gpu`, but never materializes the full
    (n_cells x k) neighbour table: it sweeps the data twice, first to get each
    cell's k-th neighbour distance (the density), then to pick each cell's
    steepest-ascent neighbour. Peak memory becomes O(n_cells) plus one chunk of
    pairwise distances, which is what makes large subsample sizes viable.
    """
    dev = torch.device(f"cuda:{device}")
    n_cells = data_matrix.shape[0]
    matrix_t = torch.as_tensor(np.ascontiguousarray(data_matrix), device=dev, dtype=torch.float32)
    squared_norms = (matrix_t * matrix_t).sum(1)
    k_eff = int(min(k_neighbors + 1, n_cells))
    chunk = _gpu_chunk_rows(n_cells, budget_fraction=0.15)

    _log(
        logger,
        f"Computing exact k-NN on GPU for {n_cells:,} cells (k={k_neighbors}, memory-lean two-pass)...",
    )

    def _chunk_distances(start, end):
        query = matrix_t[start:end]
        dist = squared_norms.unsqueeze(0) - 2.0 * (query @ matrix_t.T)
        dist += (query * query).sum(1, keepdim=True)
        return dist

    densities = torch.empty(n_cells, device=dev)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        dist = _chunk_distances(start, end)
        kth = dist.topk(k_eff, dim=1, largest=False, sorted=True)[0][:, -1]
        densities[start:end] = 1.0 / (kth.clamp_min(0).sqrt() + 1e-9)
        del dist, kth

    _log(logger, "Tracing density gradients to local maxima...")
    pointers = torch.arange(n_cells, device=dev)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        dist = _chunk_distances(start, end)
        indices = dist.topk(k_eff, dim=1, largest=False, sorted=True)[1][:, 1:]
        if indices.shape[1] > 0:
            best = indices.gather(1, densities[indices].argmax(1, keepdim=True)).squeeze(1)
            segment = pointers[start:end]
            mask = densities[best] > densities[start:end]
            segment[mask] = best[mask]
            pointers[start:end] = segment
        del dist, indices

    while True:
        next_pointers = pointers[pointers]
        if torch.equal(next_pointers, pointers):
            break
        pointers = next_pointers

    _unique_roots, assignments = torch.unique(pointers, return_inverse=True)
    result = assignments.cpu().numpy().astype(int)
    del matrix_t, squared_norms, densities, pointers, assignments
    torch.cuda.empty_cache()
    return result


def _gpu_neighbor_table_bytes(n_cells, k_neighbors):
    """Bytes the single-pass path would need for its (n x k) distance+index table."""
    # float32 distances + int64 indices, allocated twice during topk.
    return int(n_cells) * int(k_neighbors + 1) * (4 + 8) * 2


def _core_vortex_gpu(data_matrix, k_neighbors, logger=None, device=0):
    """GPU implementation of the density-gradient step. Returns cluster assignments."""
    n_cells = data_matrix.shape[0]

    # Large sketches (or large K) would need an (n x k) neighbour table that does
    # not fit in device memory; fall back to the two-pass formulation there.
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        free_bytes = 8 * 1024 ** 3
    if _gpu_neighbor_table_bytes(n_cells, k_neighbors) > free_bytes * 0.5:
        return _core_vortex_gpu_lowmem(data_matrix, k_neighbors, logger=logger, device=device)

    _log(logger, f"Computing exact k-NN on GPU for {n_cells:,} cells (k={k_neighbors})...")

    distances, indices, _ref = knn_gpu(data_matrix, data_matrix, k_neighbors + 1, device=device)
    # Drop the self-match in column 0, matching the CPU _safe_tree_query contract.
    distances, indices = distances[:, 1:], indices[:, 1:]

    # knn_gpu returns squared distances; take the root so density matches the CPU path.
    kth_distances = distances[:, -1].clamp_min(0).sqrt()
    densities = 1.0 / (kth_distances + 1e-9)

    _log(logger, "Tracing density gradients to local maxima...")
    dev = densities.device
    pointers = torch.arange(n_cells, device=dev)
    if indices.shape[1] > 0:
        best_neighbors = indices.gather(1, densities[indices].argmax(1, keepdim=True)).squeeze(1)
        higher_density_mask = densities[best_neighbors] > densities
        pointers[higher_density_mask] = best_neighbors[higher_density_mask]

    while True:
        next_pointers = pointers[pointers]
        if torch.equal(next_pointers, pointers):
            break
        pointers = next_pointers

    _unique_roots, assignments = torch.unique(pointers, return_inverse=True)
    result = assignments.cpu().numpy().astype(int)
    del distances, indices, densities, pointers, assignments
    torch.cuda.empty_cache()
    return result


def run_core_vortex(
    data_matrix,
    k_neighbors=35,
    min_cluster_size=1,
    logger=None,
    query_workers=None,
    use_gpu=None,
    gpu_device=0,
):
    """Execute the core X-shift / Vortex density-gradient clustering algorithm."""
    if data_matrix.ndim != 2:
        raise ValueError("data_matrix must be 2-dimensional")

    n_cells = data_matrix.shape[0]
    if n_cells == 0:
        raise ValueError("Cannot cluster an empty matrix")
    if n_cells == 1:
        return np.zeros(1, dtype=int)

    k_neighbors = max(1, min(int(k_neighbors), n_cells - 1))

    if _resolve_use_gpu(use_gpu):
        try:
            cluster_assignments = _core_vortex_gpu(
                data_matrix, k_neighbors, logger=logger, device=gpu_device
            )
            cluster_assignments = _merge_small_clusters(
                cluster_assignments,
                data_matrix,
                min_cluster_size=min_cluster_size,
                logger=logger,
            )
            _log(logger, f"Core Vortex clustering produced {len(np.unique(cluster_assignments))} clusters.")
            return cluster_assignments
        except Exception as exc:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _log(logger, f"GPU Vortex step failed ({exc}); falling back to the CPU KD-tree path.")

    _log(logger, f"Building KD-tree for {n_cells:,} cells (k={k_neighbors})...")

    tree = cKDTree(data_matrix)
    distances, indices = _safe_tree_query(tree, data_matrix, k_neighbors, query_workers=query_workers)
    kth_distances = distances[:, -1]
    densities = 1.0 / (kth_distances + 1e-9)

    _log(logger, "Tracing density gradients to local maxima...")
    pointers = np.arange(n_cells, dtype=int)

    if indices.shape[1] > 0:
        best_positions = np.argmax(densities[indices], axis=1)
        best_neighbors = indices[np.arange(n_cells), best_positions]
        higher_density_mask = densities[best_neighbors] > densities
        pointers[higher_density_mask] = best_neighbors[higher_density_mask]

    roots = _resolve_pointer_roots(pointers)
    unique_roots = np.unique(roots)
    root_to_cluster = {root: cluster_id for cluster_id, root in enumerate(unique_roots)}
    cluster_assignments = np.array([root_to_cluster[root] for root in roots], dtype=int)

    cluster_assignments = _merge_small_clusters(
        cluster_assignments,
        data_matrix,
        min_cluster_size=min_cluster_size,
        logger=logger,
    )
    _log(logger, f"Core Vortex clustering produced {len(np.unique(cluster_assignments))} clusters.")
    return cluster_assignments


def _evaluate_single_vortex_k(task):
    k_neighbors, matrix_sketch, score_indices, score_matrix, min_cluster_size, use_gpu, gpu_device = task
    started = time.time()
    labels = run_core_vortex(
        matrix_sketch,
        k_neighbors=k_neighbors,
        min_cluster_size=min_cluster_size,
        logger=None,
        query_workers=1,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_clusters = int(len(unique_labels))
    score_labels = labels[score_indices]

    silhouette = np.nan
    dbi = np.nan
    if 1 < len(np.unique(score_labels)) < len(score_labels):
        try:
            silhouette = float(silhouette_score(score_matrix, score_labels))
        except Exception:
            silhouette = np.nan
        try:
            dbi = float(davies_bouldin_score(score_matrix, score_labels))
        except Exception:
            dbi = np.nan

    return {
        "k_neighbors": int(k_neighbors),
        "Num_Clusters": num_clusters,
        "Min_Cluster_Size": int(counts.min()),
        "Median_Cluster_Size": float(np.median(counts)),
        "Max_Cluster_Size": int(counts.max()),
        "Largest_Cluster_Fraction": float(counts.max() / counts.sum()),
        "Silhouette_Score": silhouette,
        "Davies_Bouldin_Index": dbi,
        "Runtime_Seconds": round(time.time() - started, 2),
    }


def evaluate_vortex_k_grid(
    df,
    norm_cols,
    k_values,
    sketch_size=30000,
    subsample_method="geosketch",
    min_cluster_size=1,
    random_seed=42,
    silhouette_sample_size=5000,
    n_jobs=None,
    logger=None,
    use_gpu=None,
    gpu_device=0,
    max_geosketch_input=GEOSKETCH_MAX_INPUT_ROWS,
):
    """Evaluate a grid of K values on a sketch to guide X-shift parameter choice."""
    available_cols = [col for col in norm_cols if col in df.columns]
    if not available_cols:
        raise ValueError("No normalized marker columns available for K evaluation.")

    matrix = df[available_cols].fillna(0).astype(float).values
    sketch_size = min(int(sketch_size), len(df))
    use_gpu = _resolve_use_gpu(use_gpu)
    _log(logger, f"Preparing Vortex K-scan on {sketch_size:,} cells...")
    sketch_indices = geometric_sketch_selection(
        matrix,
        n_select=sketch_size,
        random_seed=random_seed,
        method=subsample_method,
        max_geosketch_input=max_geosketch_input,
        logger=logger,
    )
    matrix_sketch = matrix[sketch_indices]

    score_sample_size = min(int(silhouette_sample_size), matrix_sketch.shape[0])
    rng = np.random.default_rng(random_seed)
    if score_sample_size < matrix_sketch.shape[0]:
        score_indices = rng.choice(matrix_sketch.shape[0], score_sample_size, replace=False)
        score_matrix = matrix_sketch[score_indices]
    else:
        score_indices = np.arange(matrix_sketch.shape[0])
        score_matrix = matrix_sketch

    unique_k_values = sorted({int(value) for value in k_values if int(value) > 0})
    if n_jobs is None:
        if use_gpu:
            # One CUDA context per worker would serialize on the device anyway and
            # multiplies GPU memory use, so the GPU scan runs the candidates serially.
            n_jobs = 1
        else:
            cpu_count = os.cpu_count() or 1
            n_jobs = min(len(unique_k_values), max(1, cpu_count - 1))

    tasks = [
        (k_neighbors, matrix_sketch, score_indices, score_matrix, min_cluster_size, use_gpu, gpu_device)
        for k_neighbors in unique_k_values
    ]

    if use_gpu:
        _log(logger, f"Running Vortex K-scan on GPU ({gpu_device_name()}), serial over {len(tasks)} candidates...")

    if n_jobs > 1 and len(tasks) > 1:
        _log(logger, f"Running Vortex K-scan with multiprocessing across {n_jobs} workers...")
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(_evaluate_single_vortex_k, tasks))
        except Exception as exc:
            _log(logger, f"Parallel K-scan unavailable ({exc}); falling back to serial execution.")
            results = [_evaluate_single_vortex_k(task) for task in tasks]
    else:
        results = [_evaluate_single_vortex_k(task) for task in tasks]

    results = sorted(results, key=lambda row: row["k_neighbors"])
    for row in results:
        silhouette = row["Silhouette_Score"]
        silhouette_text = f"{silhouette:.4f}" if pd.notna(silhouette) else "nan"
        _log(
            logger,
            f"K-scan complete for k={row['k_neighbors']}: clusters={row['Num_Clusters']}, silhouette={silhouette_text}",
        )

    return pd.DataFrame(results)


def recommend_vortex_k(
    results_df,
    target_n_clusters=None,
    max_largest_cluster_fraction=0.45,
    min_target_fraction=0.5,
    max_target_fraction=2.0,
):
    """Recommend a Vortex K value using a simple, notebook-friendly heuristic.

    Notes
    -----
    The current Vortex approximation can occasionally return candidates where a
    single cluster absorbs most cells. Those runs are usually under-segmented
    even if the total cluster count is near the requested target.

    The staged policy is:
    1. Prefer candidates that satisfy both the largest-cluster cap and a broad
       target-cluster band.
    2. If none exist, keep the largest-cluster cap and drop the target band.
    3. If still none exist, fall back to the target band only.
    4. If needed, fall back to all valid candidates.

    Within the active stage, closeness to the target cluster count is used
    first, and silhouette / DBI act only as tie-breakers.
    """
    if results_df is None or results_df.empty:
        raise ValueError("results_df is empty")

    df = results_df.copy()
    df["__silhouette"] = df["Silhouette_Score"].fillna(-1.0)
    df["__dbi"] = df["Davies_Bouldin_Index"].fillna(np.inf)
    df["__valid"] = df["Num_Clusters"] >= 2
    df["__largest_cluster_fraction"] = pd.to_numeric(df["Largest_Cluster_Fraction"], errors="coerce").fillna(np.inf)

    balance_cap = None if max_largest_cluster_fraction is None else float(max_largest_cluster_fraction)
    if balance_cap is None:
        df["__within_balance_cap"] = True
    else:
        df["__within_balance_cap"] = df["__largest_cluster_fraction"] <= balance_cap

    target_band = None
    if target_n_clusters is not None:
        target_n_clusters = int(target_n_clusters)
        target_min = max(2, int(np.floor(target_n_clusters * float(min_target_fraction))))
        target_max = max(target_min, int(np.ceil(target_n_clusters * float(max_target_fraction))))
        df["__target_diff"] = (df["Num_Clusters"] - target_n_clusters).abs()
        df["__within_target_band"] = df["Num_Clusters"].between(target_min, target_max)
        target_band = (target_min, target_max)
    else:
        df["__target_diff"] = np.nan
        df["__within_target_band"] = True

    stage_filters: list[tuple[str, pd.Series]] = []
    if target_n_clusters is not None:
        stage_filters.extend(
            [
                ("balanced_and_target_band", df["__valid"] & df["__within_balance_cap"] & df["__within_target_band"]),
                ("balanced_only", df["__valid"] & df["__within_balance_cap"]),
                ("target_band_only", df["__valid"] & df["__within_target_band"]),
            ]
        )
    stage_filters.append(("all_valid", df["__valid"]))

    selection_stage = "all_valid"
    used_balance_cap = None
    used_target_band = None
    selected_df = None
    for stage_name, mask in stage_filters:
        if mask.any():
            selected_df = df.loc[mask].copy()
            selection_stage = stage_name
            if stage_name in {"balanced_and_target_band", "balanced_only"} and balance_cap is not None:
                used_balance_cap = balance_cap
            if stage_name in {"balanced_and_target_band", "target_band_only"}:
                used_target_band = target_band
            break

    if selected_df is None or selected_df.empty:
        raise ValueError("No valid Vortex K candidates are available")

    if target_n_clusters is not None:
        sort_cols = ["__valid", "__target_diff", "__largest_cluster_fraction", "__silhouette", "__dbi"]
        ascending = [False, True, True, False, True]
    else:
        sort_cols = ["__valid", "__largest_cluster_fraction", "__silhouette", "__dbi"]
        ascending = [False, True, False, True]

    best_row = selected_df.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
    for temp_col in [
        "__silhouette",
        "__dbi",
        "__valid",
        "__target_diff",
        "__largest_cluster_fraction",
        "__within_balance_cap",
        "__within_target_band",
    ]:
        best_row.pop(temp_col, None)
    best_row["Selection_Balance_Cap"] = used_balance_cap
    best_row["Selection_Target_Band"] = used_target_band
    best_row["Selection_Stage"] = selection_stage
    if selection_stage != "balanced_and_target_band":
        best_row["Selection_Warning"] = (
            "No candidate satisfied both the dominant-cluster cap and the preferred target-cluster band. "
            "The recommendation is a fallback and may still be biologically unsatisfactory."
        )
    return best_row


def create_vortex_k_diagnostic_plot(
    results_df,
    plots_dir,
    filename="vortex_k_diagnostics.png",
):
    """Create a diagnostic figure summarizing K-scan behavior."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    results_sorted = results_df.sort_values("k_neighbors")

    axes[0].plot(results_sorted["k_neighbors"], results_sorted["Num_Clusters"], marker="o", linewidth=2)
    axes[0].set_title("Cluster Count vs K")
    axes[0].set_xlabel("K neighbors")
    axes[0].set_ylabel("Clusters")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_sorted["k_neighbors"], results_sorted["Silhouette_Score"], marker="o", color="#2E86AB", linewidth=2)
    axes[1].set_title("Silhouette vs K")
    axes[1].set_xlabel("K neighbors")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results_sorted["k_neighbors"], results_sorted["Davies_Bouldin_Index"], marker="o", color="#C73E1D", linewidth=2)
    axes[2].set_title("Davies-Bouldin vs K")
    axes[2].set_xlabel("K neighbors")
    axes[2].set_ylabel("DBI")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Vortex (X-shift) K Selection Diagnostics", fontsize=14)
    fig.tight_layout()

    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _project_umap_from_subsample(
    matrix_subsample,
    subsample_labels,
    all_assignments,
    subsample_indices,
    random_seed,
    umap_neighbors,
    logger=None,
):
    all_umap_coords = np.full((len(all_assignments), 2), np.nan, dtype=float)
    if sc is None:
        _log(logger, "Scanpy unavailable. Skipping Vortex UMAP generation.")
        return all_umap_coords

    _log(logger, "Generating UMAP embedding on the Vortex sketch...")
    adata = sc.AnnData(matrix_subsample.astype(np.float32, copy=False))
    max_pcs = min(30, adata.n_vars, max(1, adata.n_obs - 1))
    sc.tl.pca(adata, n_comps=max_pcs, svd_solver="randomized", random_state=random_seed)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(int(umap_neighbors), max(1, adata.n_obs - 1)),
        n_pcs=max_pcs,
        use_rep="X_pca",
        random_state=random_seed,
    )
    sc.tl.umap(adata, random_state=random_seed)

    all_umap_coords[subsample_indices] = adata.obsm["X_umap"]
    for cluster_id in np.unique(subsample_labels):
        cluster_mask = subsample_labels == cluster_id
        centroid = adata.obsm["X_umap"][cluster_mask].mean(axis=0)
        missing_mask = (all_assignments == cluster_id) & np.isnan(all_umap_coords[:, 0])
        all_umap_coords[missing_mask] = centroid

    return all_umap_coords


def intelligent_vortex_clustering(df, norm_cols, vortex_params, logger=None):
    """Run scalable Vortex clustering with sketching, KNN backfilling, and optional UMAP."""
    available_cols = [col for col in norm_cols if col in df.columns]
    if not available_cols:
        raise ValueError("No normalized columns available for Vortex clustering.")

    matrix = df[available_cols].fillna(0).astype(float).values
    total_cells = matrix.shape[0]

    k_neighbors = int(vortex_params.get("k_neighbors", 35))
    subsample_size = min(int(vortex_params.get("subsample_size", total_cells)), total_cells)
    subsample_method = vortex_params.get("subsample_method", "geosketch")
    random_seed = int(vortex_params.get("random_seed", 42))
    knn_neighbors = int(vortex_params.get("knn_neighbors", 15))
    min_cluster_size = int(vortex_params.get("min_cluster_size", 1))
    generate_umap = bool(vortex_params.get("generate_umap", True))
    umap_neighbors = int(vortex_params.get("umap_neighbors", 15))

    use_gpu = _resolve_use_gpu(vortex_params.get("use_gpu"))
    gpu_device = int(vortex_params.get("gpu_device", 0))
    max_geosketch_input = vortex_params.get("max_geosketch_input", GEOSKETCH_MAX_INPUT_ROWS)

    _log(logger, f"Starting Vortex clustering for {total_cells:,} cells...")
    _log(logger, f"Using k={k_neighbors}, subsample_size={subsample_size:,}, method={subsample_method}.")
    if use_gpu:
        _log(logger, f"GPU acceleration enabled: {gpu_device_name()}")

    start_time = time.time()
    subsample_indices = geometric_sketch_selection(
        matrix,
        n_select=subsample_size,
        random_seed=random_seed,
        method=subsample_method,
        max_geosketch_input=max_geosketch_input,
        logger=logger,
    )
    matrix_subsample = matrix[subsample_indices]

    subsample_clusters = run_core_vortex(
        matrix_subsample,
        k_neighbors=k_neighbors,
        min_cluster_size=min_cluster_size,
        logger=logger,
        use_gpu=use_gpu,
        gpu_device=gpu_device,
    )

    all_assignments = np.full(total_cells, -1, dtype=int)
    all_assignments[subsample_indices] = subsample_clusters

    unassigned_mask = all_assignments == -1
    if unassigned_mask.any():
        _log(logger, f"Assigning {int(unassigned_mask.sum()):,} remaining cells via KNN...")
        all_assignments[unassigned_mask] = _knn_backfill(
            matrix_subsample,
            subsample_clusters,
            matrix[unassigned_mask],
            knn_neighbors=knn_neighbors,
            use_gpu=use_gpu,
            gpu_device=gpu_device,
            logger=logger,
        )

    # Optional merge of the fine Vortex clusters into a manageable number of
    # meta-clusters. The density-gradient step routinely returns several hundred
    # clusters, which is impractical to annotate by hand; merging by marker
    # profile keeps distinct populations apart while bringing the count down to
    # something comparable with the other phenotyping methods.
    meta_clusters = int(vortex_params.get("meta_clusters", 0) or 0)
    if meta_clusters >= 2:
        from spacemap.ph_accel_utils import meta_cluster_labels

        all_assignments, _mapping = meta_cluster_labels(
            all_assignments, matrix, meta_clusters, logger=logger
        )
        subsample_clusters = all_assignments[subsample_indices]

    if generate_umap:
        all_umap_coords = _project_umap_from_subsample(
            matrix_subsample=matrix_subsample,
            subsample_labels=subsample_clusters,
            all_assignments=all_assignments,
            subsample_indices=subsample_indices,
            random_seed=random_seed,
            umap_neighbors=umap_neighbors,
            logger=logger,
        )
    else:
        all_umap_coords = np.full((total_cells, 2), np.nan, dtype=float)

    cluster_counts = pd.Series(all_assignments).value_counts().sort_index()
    diagnostics = {
        "k_neighbors": k_neighbors,
        "subsample_size": subsample_size,
        "subsample_method": subsample_method,
        "runtime_seconds": round(time.time() - start_time, 2),
        "n_clusters": int(cluster_counts.shape[0]),
        "cluster_sizes": cluster_counts,
        "min_cluster_size": int(cluster_counts.min()),
        "median_cluster_size": float(cluster_counts.median()),
        "max_cluster_size": int(cluster_counts.max()),
    }
    _log(
        logger,
        "Vortex clustering complete: "
        f"clusters={diagnostics['n_clusters']}, "
        f"min={diagnostics['min_cluster_size']}, "
        f"median={diagnostics['median_cluster_size']:.1f}, "
        f"max={diagnostics['max_cluster_size']}.",
    )

    return all_assignments, all_umap_coords, diagnostics


def launch_vortex_phenotype_assignment_app(
    char_df,
    reordered_cluster_ids,
    meta_cluster_phenotypes,
):
    """Launch the shared SPACEMAP phenotype studio using Vortex cluster aliases."""
    from spacemap.ph_leiden_utils import launch_phenotype_assignment_app as _shared_launch_phenotype_assignment_app

    working = char_df.copy()
    if "vortex_cluster" in working.columns:
        working = working.rename(columns={"vortex_cluster": "leiden_cluster"})
    return _shared_launch_phenotype_assignment_app(
        char_df=working,
        reordered_cluster_ids=reordered_cluster_ids,
        meta_cluster_phenotypes=meta_cluster_phenotypes,
    )


def create_vortex_qc_plots(
    data,
    plots_dir,
    cluster_col="vortex_cluster",
    phenotype_col=None,
    max_plot_cells=200000,
    random_seed=42,
    filename="vortex_qc_dashboard.png",
):
    """Create a compact QC dashboard with cluster-size and UMAP summaries."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cluster_col not in data.columns:
        raise ValueError(f"'{cluster_col}' column not found in data")

    working = data.copy()
    counts = working[cluster_col].value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    top_n = min(25, len(counts))

    axes[0, 0].bar(range(top_n), counts.iloc[:top_n].values, color="#2E86AB")
    axes[0, 0].set_title("Largest Cluster Sizes")
    axes[0, 0].set_xlabel("Ranked clusters")
    axes[0, 0].set_ylabel("Cells")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].hist(counts.values, bins=min(30, max(10, len(counts))), color="#F18F01", edgecolor="white")
    axes[0, 1].set_title("Cluster Size Distribution")
    axes[0, 1].set_xlabel("Cells per cluster")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    if {"umap_x", "umap_y"}.issubset(working.columns):
        if len(working) > max_plot_cells:
            rng = np.random.default_rng(random_seed)
            plot_indices = rng.choice(len(working), max_plot_cells, replace=False)
            plot_df = working.iloc[plot_indices].copy()
        else:
            plot_df = working.copy()

        scatter = axes[1, 0].scatter(
            plot_df["umap_x"],
            plot_df["umap_y"],
            c=plot_df[cluster_col],
            cmap="Spectral",
            s=0.7,
            alpha=0.7,
            rasterized=True,
        )
        axes[1, 0].set_title("UMAP Colored by Vortex Cluster")
        axes[1, 0].set_xlabel("UMAP 1")
        axes[1, 0].set_ylabel("UMAP 2")
        axes[1, 0].grid(True, alpha=0.2)
        fig.colorbar(scatter, ax=axes[1, 0], fraction=0.046, pad=0.04)

        if phenotype_col and phenotype_col in plot_df.columns:
            unique_phenotypes = plot_df[phenotype_col].fillna("Unassigned").astype(str).unique().tolist()
            colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(unique_phenotypes))))
            phenotype_colors = dict(zip(unique_phenotypes, colors))
            for phenotype in unique_phenotypes:
                mask = plot_df[phenotype_col].fillna("Unassigned").astype(str) == phenotype
                axes[1, 1].scatter(
                    plot_df.loc[mask, "umap_x"],
                    plot_df.loc[mask, "umap_y"],
                    s=0.7,
                    alpha=0.7,
                    color=phenotype_colors[phenotype],
                    label=phenotype,
                    rasterized=True,
                )
            axes[1, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            axes[1, 1].set_title("UMAP Colored by Phenotype")
        else:
            cumulative = counts.cumsum() / counts.sum()
            axes[1, 1].plot(np.arange(1, len(cumulative) + 1), cumulative.values, color="#A23B72", linewidth=2)
            axes[1, 1].set_title("Cumulative Cell Coverage by Cluster")
            axes[1, 1].set_xlabel("Top N clusters")
            axes[1, 1].set_ylabel("Fraction of cells")
            axes[1, 1].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("UMAP 1" if phenotype_col and phenotype_col in plot_df.columns else axes[1, 1].get_xlabel())
        axes[1, 1].set_ylabel("UMAP 2" if phenotype_col and phenotype_col in plot_df.columns else axes[1, 1].get_ylabel())
        if phenotype_col and phenotype_col in plot_df.columns:
            axes[1, 1].grid(True, alpha=0.2)
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "UMAP coordinates not available", ha="center", va="center", fontsize=12)
        cumulative = counts.cumsum() / counts.sum()
        axes[1, 1].plot(np.arange(1, len(cumulative) + 1), cumulative.values, color="#A23B72", linewidth=2)
        axes[1, 1].set_title("Cumulative Cell Coverage by Cluster")
        axes[1, 1].set_xlabel("Top N clusters")
        axes[1, 1].set_ylabel("Fraction of cells")
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Vortex (X-shift) Quality Control Dashboard", fontsize=15)
    fig.tight_layout()
    out_path = plots_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def apply_vortex_phenotypes(
    data,
    phenotype_map,
    output_base_dir,
    cell_id_column,
    cluster_col="vortex_cluster",
    method_prefix="vortex",
):
    """Apply Vortex cluster phenotypes and update the shared unified output file."""
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if cluster_col not in data.columns:
        raise ValueError(f"'{cluster_col}' column is required to apply Vortex phenotypes")
    if cell_id_column not in data.columns:
        raise ValueError(f"'{cell_id_column}' column required for proper data merging")

    working = data.copy()
    phenotype_col = f"phenotype_{method_prefix}"
    working[phenotype_col] = working[cluster_col].map(phenotype_map).fillna("Unassigned")

    summary = working.groupby(phenotype_col).agg(
        cell_count=(phenotype_col, "count"),
        num_clusters=(cluster_col, "nunique"),
    )
    summary["percentage"] = (summary["cell_count"] / len(working) * 100).round(2)
    summary = summary.sort_values("cell_count", ascending=False)

    unified_path = output_base_dir / "all_samples_combined_corrected_phenotyped.csv"
    method_specific_cols = [cluster_col, phenotype_col]
    method_specific_cols.extend([col for col in ["umap_x", "umap_y"] if col in working.columns])

    if unified_path.exists():
        existing = pd.read_csv(unified_path)
        if cell_id_column in existing.columns and len(existing) > 0:
            print(f"Merging {method_prefix} results with existing unified file...")
            existing = existing.set_index(cell_id_column)
            indexed_working = working.set_index(cell_id_column)
            for col in method_specific_cols:
                if col in indexed_working.columns:
                    existing[col] = indexed_working[col]
                    print(f" Added/updated column: {col}")
            if cell_id_column in existing.columns:
                existing = existing.drop(columns=[cell_id_column])
            unified_df = existing.reset_index()
        else:
            print(f"Creating new unified file with {method_prefix} results...")
            essential_cols = [cell_id_column] + [col for col in working.columns if not col.endswith("_norm")]
            unified_df = working[essential_cols].copy()
    else:
        print(f"Creating new unified file with {method_prefix} results...")
        essential_cols = [cell_id_column] + [col for col in working.columns if not col.endswith("_norm")]
        unified_df = working[essential_cols].copy()

    unified_df.to_csv(unified_path, index=False)
    print(f"Unified file saved: {unified_path} ({len(unified_df):,} rows)")

    summary_file = output_base_dir / f"{method_prefix}_phenotype_summary.csv"
    summary.to_csv(summary_file)

    mapping_file = output_base_dir / f"{method_prefix}_phenotype_mapping.csv"
    pd.DataFrame(list(phenotype_map.items()), columns=["cluster_id", "phenotype"]).to_csv(mapping_file, index=False)

    return {
        "unified_file": unified_path,
        "summary_file": summary_file,
        "mapping_file": mapping_file,
        "summary_df": summary,
        "data": working,
    }


launch_phenotype_assignment_app = launch_vortex_phenotype_assignment_app
apply_phenotypes = apply_vortex_phenotypes


__all__ = [
    "GEOSKETCH_AVAILABLE",
    "apply_flexible_normalization",
    "load_or_compute_normalized",
    "characterize_clusters",
    "create_cluster_heatmap",
    "geometric_sketch_selection",
    "run_core_vortex",
    "evaluate_vortex_k_grid",
    "recommend_vortex_k",
    "create_vortex_k_diagnostic_plot",
    "intelligent_vortex_clustering",
    "launch_vortex_phenotype_assignment_app",
    "launch_phenotype_assignment_app",
    "create_vortex_qc_plots",
    "apply_vortex_phenotypes",
    "apply_phenotypes",
]