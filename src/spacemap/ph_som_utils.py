"""
Self-Organizing Map (SOM) based phenotyping utilities for the SPACEMAP pipeline.

This module provides core functions for performing cell phenotyping using SOM clustering.
It includes utilities for SOM training, meta-clustering, cluster characterization, 
and interactive phenotype annotation.
"""

import os
# FIX: Set Protobuf implementation to python to avoid descriptor conflict warnings/errors
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Helper for flushed printing (important for Streamlit visibility)
def _print_flush(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# Scientific computing
from sklearn.cluster import AgglomerativeClustering

# SOM implementation
try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    print("Warning: minisom not installed. SOM functionality will be limited.")
    MINISOM_AVAILABLE = False
    MiniSom = None

# Geosketch implementation (optional but recommended for large datasets)
try:
    import geosketch
    GEOSKETCH_AVAILABLE = True
except ImportError:
    # print("Warning: geosketch not installed. Falling back to random sampling.") # reduced noise
    GEOSKETCH_AVAILABLE = False
    geosketch = None

warnings.filterwarnings('ignore')


# ============================================================================
# SOM TRAINING AND CLUSTER ANALYSIS
# ============================================================================


def train_som_advanced(clustering_data_scaled, data_df, cell_id_column, 
                       adaptive_config=None, mode_params=None, som_x_dim=None, som_y_dim=None,
                       subsample_size=100000, subsample_method='geosketch', random_seed=42, eval_size=4000,
                       second_stage_scaling=False, qe_scaling_strategy='none',
                       early_stop=True, early_stop_min_improvement=0.01, early_stop_min_epoch_fraction=0.3):
    """Advanced SOM training with adaptive grid sizing, mode-aware parameters, and comprehensive metrics.
    
    This function provides a complete SOM training pipeline with:
    1. Adaptive grid sizing based on data size and target cells per node
    2. Mode-aware parameter selection (fast/balanced/high_quality)
    3. Comprehensive quality metrics including normalized QE and KMeans baseline comparison
    4. Cell_ID anchoring for guaranteed correct cluster assignments
    5. Professional training progress monitoring
    
    Parameters
    ----------
    clustering_data_scaled : np.ndarray
        Scaled input data for training and assignment
    data_df : pd.DataFrame
        Original DataFrame containing cell_id column for proper anchoring
    cell_id_column : str
        Name of the cell_id column for proper anchoring (required parameter)
    adaptive_config : dict, optional
        Configuration for adaptive grid sizing with keys:
        - auto: bool, whether to use adaptive sizing
        - target_cells_per_node: int, target cells per SOM node
        - min_grid, max_grid: int, grid size bounds
        - mode: str, training mode ('fast', 'balanced', 'high_quality')
        - base_epochs: int, base number of training epochs
    mode_params : dict, optional
        Mode-specific parameters with keys:
        - iter_factor: float, iteration scaling factor
        - start_sigma: float, initial sigma value
        - lr: float, learning rate
        - batch_size: int, training batch size
    som_x_dim, som_y_dim : int, optional
        Fixed SOM grid dimensions (overridden if adaptive_config['auto'] is True)
    subsample_size : int
        Maximum number of cells to use for SOM training
    random_seed : int
        Random seed for reproducibility
    eval_size : int
        Number of cells to use for quality evaluation metrics
    second_stage_scaling : bool
        Whether to apply additional scaling before training
    qe_scaling_strategy : str
        Scaling strategy ('none', 'minmax', 'unit_variance')
    
    Returns
    -------
    dict
        Comprehensive results containing:
        - som: trained MiniSom object
        - som_clusters: cluster assignments for all cells
        - som_clusters_df: DataFrame with cell_id anchoring
        - quantization_errors: QE progression during training
        - quantization_errors_normalized: normalized QE progression
        - quantization_errors_kmeans_ratio: QE vs KMeans baseline ratios
        - topographic_errors: topographic error progression
        - grid_size: final grid dimensions
        - som_node_info: node statistics and coordinates
        - approx_pairwise_mean: scale reference for normalization
        - kmeans_qe_baseline: KMeans baseline for comparison
        - node_occupancy: occupancy statistics
        - training_summary: training configuration and results
    """
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from scipy.spatial.distance import cdist
    
    if not MINISOM_AVAILABLE or MiniSom is None:
        raise ImportError("MiniSom not available. Install minisom or provide fallback.")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Default configurations
    if adaptive_config is None:
        adaptive_config = {
            'auto': False,
            'target_cells_per_node': 1800,
            'min_grid': 40,
            'max_grid': 70,
            'mode': 'balanced',
            'base_epochs': 80
        }
    
    if mode_params is None:
        mode_params = {
            'iter_factor': 1.0,
            'start_sigma': 1.5,
            'lr': 0.10,
            'batch_size': 4096
        }
    
    # Prepare training subset
    _print_flush("Loading clustering matrix for SOM training subset...")
    n_training = min(subsample_size, len(clustering_data_scaled))
    
    # Prepare training subset
    _print_flush("Loading clustering matrix for SOM training subset...")
    n_training = min(subsample_size, len(clustering_data_scaled))
    
    # Enable Geometric Sketching if needed
    use_geosketch = (subsample_method == 'geosketch') and (n_training < len(clustering_data_scaled))

    if use_geosketch:
        _print_flush(f"Applying Geometric Sketching to select balanced subset of {n_training} cells...")
        try:
            # gs expects (n_samples, n_features); marker dimensionality (~50) is
            # small enough to sketch the scaled data directly without PCA.
            from spacemap.ph_accel_utils import geometric_sketch

            train_idx = geometric_sketch(
                clustering_data_scaled, n_training, random_seed=random_seed
            )
            _print_flush("Geosketch sampling complete.")
        except Exception as e:
            _print_flush(f"Geosketch failed ({e}), falling back to uniform random sampling.")
            train_idx = np.random.choice(len(clustering_data_scaled), n_training, replace=False)
    else:
        if subsample_method == 'random' and (n_training < len(clustering_data_scaled)):
            _print_flush("Using Uniform Random Sampling (fast mode).")

        train_idx = np.random.choice(len(clustering_data_scaled), n_training, replace=False)

    training_data = clustering_data_scaled[train_idx]
    _print_flush(f"Training subset: {n_training:,} cells (of {len(clustering_data_scaled):,})")
    
    # Adaptive grid sizing
    if adaptive_config.get('auto', False):
        target_cells_per_node = adaptive_config['target_cells_per_node']
        est_nodes = max(1, int(n_training / max(50, target_cells_per_node)))
        requested_side = max(1, int(np.sqrt(est_nodes)))
        min_grid = int(adaptive_config['min_grid'])
        max_grid = int(adaptive_config['max_grid'])
        side = max(min_grid, min(max_grid, requested_side))
        som_x_dim = side
        som_y_dim = side
        actual_cells_per_node = n_training / (som_x_dim * som_y_dim)
        _print_flush(f"Adaptive grid selected: {som_x_dim}x{som_y_dim} (nodes={som_x_dim*som_y_dim} ~ {actual_cells_per_node:.0f} cells/node)")
        if side != requested_side:
            bound_name = "Minimum Grid Side" if side == min_grid and requested_side < min_grid else "Maximum Grid Side"
            _print_flush(
                f"NOTE: Target Cells/Node={target_cells_per_node} implied a {requested_side}x{requested_side} grid, "
                f"but the {bound_name} bound ({side}) overrode it. The grid actually holds "
                f"~{actual_cells_per_node:.0f} cells/node, not {target_cells_per_node}. "
                f"Lower the bound (or turn off Auto Grid Size) if you want the target honoured."
            )
    else:
        if som_x_dim is None:
            som_x_dim = 60
        if som_y_dim is None:
            som_y_dim = 60
        _print_flush(f"Using fixed grid: {som_x_dim}x{som_y_dim}")
    
    # Mode parameter injection
    start_sigma = mode_params['start_sigma']
    som_sigma = start_sigma
    som_lr = mode_params['lr']
    batch_size = mode_params['batch_size']
    
    # Epoch computation scaling with grid complexity
    base_epochs = adaptive_config.get('base_epochs', 80)
    complexity_factor = (som_x_dim * som_y_dim) / 2500  # baseline 50x50
    som_iterations = int(base_epochs * mode_params['iter_factor'] / (1 + 0.3 * (complexity_factor - 1)))
    som_iterations = max(30, som_iterations)
    _print_flush(f"Planned epochs: {som_iterations} (mode={adaptive_config.get('mode', 'balanced')} complexity_factor={complexity_factor:.2f})")
    
    # Optional second-stage scaling
    if second_stage_scaling and qe_scaling_strategy != 'none':
        from sklearn.preprocessing import MinMaxScaler
        if qe_scaling_strategy == 'minmax':
            training_data = MinMaxScaler().fit_transform(training_data)
            _print_flush("Applied MinMax scaling (SECOND_STAGE_SCALING)")
        elif qe_scaling_strategy == 'unit_variance':
            var = training_data.var(axis=0, ddof=1)
            var[var == 0] = 1.0
            training_data = training_data / np.sqrt(var)
            _print_flush("Applied unit variance scaling (SECOND_STAGE_SCALING)")
    
    # Evaluation subset
    eval_size = min(eval_size, len(training_data))
    qe_eval_data = training_data[np.random.choice(len(training_data), eval_size, replace=False)]
    _print_flush(f"Eval subset: {eval_size} cells")
    
    # KMeans baseline
    _print_flush("Computing KMeans baseline...")
    n_kmeans_clusters = som_x_dim * som_y_dim
    kmeans_sample_size = min(15000, len(training_data))
    km_sample = training_data[np.random.choice(len(training_data), kmeans_sample_size, replace=False)]
    mbk = MiniBatchKMeans(n_clusters=n_kmeans_clusters, random_state=random_seed, batch_size=4096, n_init=1, max_iter=35)
    mbk.fit(km_sample)
    mbk_eval = mbk.predict(qe_eval_data)
    kmeans_qe_baseline = np.mean(np.linalg.norm(qe_eval_data - mbk.cluster_centers_[mbk_eval], axis=1))
    _print_flush(f"KMeans baseline QE: {kmeans_qe_baseline:.3f}")
    
    # Initialize SOM - use fast random init for 'fast' mode, PCA for others
    import time as _time
    from sklearn.decomposition import PCA
    
    use_pca_init = adaptive_config.get('mode') != 'fast'
    
    som = MiniSom(
        x=som_x_dim,
        y=som_y_dim,
        input_len=training_data.shape[1],
        sigma=som_sigma,
        learning_rate=som_lr,
        neighborhood_function='gaussian',
        topology='rectangular',
        activation_distance='euclidean',
        random_seed=random_seed
    )
    
    if use_pca_init:
        _print_flush("Initializing SOM with PCA weights (sklearn-optimized)...")
        _pca_start = _time.time()
        
        # Use sklearn PCA which is MUCH faster than MiniSom's pca_weights_init
        # MiniSom computes full covariance matrix and eigenvectors on all data
        # sklearn uses efficient randomized SVD
        pca_subsample_size = min(10000, len(training_data))
        pca_data = training_data[np.random.choice(len(training_data), pca_subsample_size, replace=False)]
        
        pca = PCA(n_components=2, random_state=random_seed)
        pca.fit(pca_data)
        
        # Initialize weights along first two principal components
        pc1, pc2 = pca.components_[0], pca.components_[1]
        data_mean = pca_data.mean(axis=0)
        
        for i, c1 in enumerate(np.linspace(-1, 1, som_x_dim)):
            for j, c2 in enumerate(np.linspace(-1, 1, som_y_dim)):
                som._weights[i, j] = data_mean + c1 * pc1 * np.sqrt(pca.explained_variance_[0]) + \
                                                 c2 * pc2 * np.sqrt(pca.explained_variance_[1])
        
        pca_elapsed = _time.time() - _pca_start
        _print_flush(f"SOM PCA init done in {pca_elapsed:.1f}s (sklearn-optimized)")
    else:
        _print_flush("Initializing SOM with random weights (fast mode)...")
        _pca_start = _time.time()
        som.random_weights_init(training_data[:min(5000, len(training_data))])
        _print_flush(f"SOM random init done in {_time.time()-_pca_start:.1f}s")
    
    _print_flush(f"SOM ready (sigma={som_sigma}, lr={som_lr})")
    
    # Scale references
    pair_ref_size = min(2500, len(training_data))
    ref_sample = training_data[np.random.choice(len(training_data), pair_ref_size, replace=False)]
    center = ref_sample.mean(axis=0)
    mean_center_dist = np.mean(np.linalg.norm(ref_sample - center, axis=1))
    approx_pairwise_mean = np.sqrt(2) * mean_center_dist
    _print_flush(f"Approx scale: pairwise_ref={approx_pairwise_mean:.3f}")
    
    # Training containers
    quantization_errors = []
    quantization_errors_normalized = []
    quantization_errors_kmeans_ratio = []
    topographic_errors = []
    checkpoints = []
    
    # Advanced training with custom implementation (restored from backup)
    _print_flush("Starting SOM training loop...")
    
    weights = som._weights
    map_shape = weights.shape[:2]
    coords = np.array([(i, j) for i in range(map_shape[0]) for j in range(map_shape[1])], dtype=np.int32)
    coord_i = coords[:, 0]
    coord_j = coords[:, 1]
    Delta = np.zeros_like(weights)
    
    steps_per_epoch = int(np.ceil(len(training_data) / batch_size))
    final_sigma = 0.55 if adaptive_config.get('mode') != 'high_quality' else 0.50
    final_lr = 0.02 if adaptive_config.get('mode') != 'high_quality' else 0.025
    
    # Cosine schedules
    def schedule(p, start, end):
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * p))
    
    # Fast metrics helpers
    def fast_quantization_error(data, weights_flat):
        dists = cdist(data, weights_flat, metric='euclidean')
        return dists.min(axis=1).mean()

    def fast_topographic_error(data, weights_flat, map_shape):
        dists = cdist(data, weights_flat, metric='euclidean')
        top2_flat_indices = np.argpartition(dists, 1, axis=1)[:, :2]
        row_indices = np.arange(len(data))[:, None]
        col_dists = dists[row_indices, top2_flat_indices]
        sorted_order = np.argsort(col_dists, axis=1)
        top2_sorted = top2_flat_indices[row_indices, sorted_order]
        
        bmu1 = top2_sorted[:, 0]
        bmu2 = top2_sorted[:, 1]
        
        x1, y1 = bmu1 // map_shape[1], bmu1 % map_shape[1]
        x2, y2 = bmu2 // map_shape[1], bmu2 % map_shape[1]
        
        chebyshev_dist = np.maximum(np.abs(x1 - x2), np.abs(y1 - y2))
        errors = np.sum(chebyshev_dist > 1)
        return errors / len(data)

    # Metric logging helper
    def record(tag):
        weights_flat = weights.reshape(-1, weights.shape[2])
        qe = fast_quantization_error(qe_eval_data, weights_flat)
        nqe = qe / approx_pairwise_mean
        ratio = qe / kmeans_qe_baseline
        te = fast_topographic_error(qe_eval_data, weights_flat, map_shape)
        quantization_errors.append(qe)
        quantization_errors_normalized.append(nqe)
        quantization_errors_kmeans_ratio.append(ratio)
        topographic_errors.append(te)
        checkpoints.append(tag)
        return qe, nqe, ratio, te
    
    print_interval = max(1, som_iterations // 35)
    prev_nqe = []
    
    # Main training loop
    for epoch in range(som_iterations):
        p = epoch / max(1, som_iterations - 1)
        som._sigma = schedule(p, som_sigma, final_sigma)
        som._learning_rate = schedule(p, som_lr, final_lr)
        
        perm = np.random.permutation(len(training_data))
        epoch_data = training_data[perm]
        
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, len(epoch_data))
            if s >= e:
                continue
            batch = epoch_data[s:e]
            flat_w = weights.reshape(-1, weights.shape[2])
            dists = cdist(batch, flat_w, metric='euclidean')
            bmu_idx = np.argmin(dists, axis=1)
            unique_bmu, inverse = np.unique(bmu_idx, return_inverse=True)
            
            sigma_sq = som._sigma ** 2
            lr = som._learning_rate
            
            # Vectorized target calculation
            counts = np.bincount(inverse)
            targets = np.zeros((len(unique_bmu), batch.shape[1]), dtype=batch.dtype)
            for d in range(batch.shape[1]):
                targets[:, d] = np.bincount(inverse, weights=batch[:, d]) / counts
                
            # Vectorized neighborhood calculation
            bi = (unique_bmu // map_shape[1])[:, None]
            bj = (unique_bmu % map_shape[1])[:, None]
            dist2 = (coord_i[None, :] - bi) ** 2 + (coord_j[None, :] - bj) ** 2
            neighs_flat = np.exp(-dist2 / (2 * sigma_sq))
            neighs = neighs_flat.reshape(len(unique_bmu), map_shape[0], map_shape[1])
            
            # Vectorized Delta calculation
            sum_neighs = neighs.sum(axis=0)
            term1 = np.tensordot(neighs, targets, axes=(0, 0))
            term2 = sum_neighs[..., None] * weights
            Delta = lr * (term1 - term2)
            weights += Delta
        
        if epoch % print_interval == 0 or epoch == som_iterations - 1:
            qe, nqe, ratio, te = record(f"E{epoch}")
            prev_nqe.append(nqe)
            _print_flush(f" Epoch {epoch:3d}/{som_iterations-1} | sigma={som._sigma:.3f} lr={som._learning_rate:.4f} QE={qe:.3f} nQE={nqe:.3f} r={ratio:.2f} TE={te:.3f}")
            # Lightweight early plateau detection
            if early_stop and len(prev_nqe) > 6 and epoch > som_iterations * early_stop_min_epoch_fraction:
                recent = prev_nqe[-5:]
                improv = (recent[0] - recent[-1]) / max(1e-9, recent[0])
                if improv < early_stop_min_improvement:
                    _print_flush(" Early stop: nQE plateau (<1% improvement)")
                    break
    
    final_qe = quantization_errors[-1]
    final_nqe = quantization_errors_normalized[-1]
    final_te = topographic_errors[-1]
    final_ratio = quantization_errors_kmeans_ratio[-1]
    _print_flush(f"Final metrics: QE={final_qe:.3f} nQE={final_nqe:.3f} ratio={final_ratio:.2f} TE={final_te:.3f}")
    
    # Project all cells (Vectorized Optimization)
    _print_flush("Projecting all cells to SOM nodes (Vectorized)...")
    try:
        from scipy.spatial.distance import cdist
        weights_flat = som._weights.reshape(-1, som._weights.shape[2])
        # Chunk processing to prevent OOM
        chunk_size = 50000 
        n_samples = len(clustering_data_scaled)
        som_clusters = np.zeros(n_samples, dtype=int)
        
        for i in range(0, n_samples, chunk_size):
            chunk = clustering_data_scaled[i : min(i + chunk_size, n_samples)]
            dists = cdist(chunk, weights_flat, metric='euclidean')
            som_clusters[i : i + len(chunk)] = np.argmin(dists, axis=1)
            if (i // chunk_size) % 10 == 0:
                _print_flush(f"  Projected {min(i + chunk_size, n_samples):,}/{n_samples:,} cells...")
            
    except Exception as e:
        _print_flush(f"Vectorized projection failed ({e}), falling back to iterative...")
        winner_coords = [som.winner(v) for v in clustering_data_scaled]
        som_clusters = np.array([c[0] * som_y_dim + c[1] for c in winner_coords])
    
    som_clusters_df = pd.DataFrame({cell_id_column: data_df[cell_id_column], 'som_cluster': som_clusters})
    
    # Node occupancy stats
    unique_nodes, node_counts = np.unique(som_clusters, return_counts=True)
    occupied = len(unique_nodes)
    empty = som_x_dim * som_y_dim - occupied
    median_cells = int(np.median(node_counts))
    _print_flush(f"Node occupancy: occupied={occupied} empty={empty} median_cells/node={median_cells}")
    
    # Compute node weight statistics
    node_weight_means = som._weights.reshape(som_x_dim * som_y_dim, -1)
    node_weight_std = np.zeros_like(node_weight_means)
    
    som_node_info = []
    for idx, node_id in enumerate(unique_nodes):
        x = node_id // som_y_dim
        y = node_id % som_y_dim
        som_node_info.append({
            'node_cluster_id': idx,
            'n_nodes': 1,
            'node_coords': [(x, y)],
            'mean_weights': node_weight_means[node_id],
            'std_weights': node_weight_std[node_id]
        })
    
    return {
        'som': som,
        'som_clusters': som_clusters,
        'som_clusters_df': som_clusters_df,
        'quantization_errors': quantization_errors,
        'quantization_errors_normalized': quantization_errors_normalized,
        'quantization_errors_kmeans_ratio': quantization_errors_kmeans_ratio,
        'topographic_errors': topographic_errors,
        'checkpoints': checkpoints,
        'grid_size': (som_x_dim, som_y_dim),
        'som_node_info': som_node_info,
        'approx_pairwise_mean': approx_pairwise_mean,
        'kmeans_qe_baseline': kmeans_qe_baseline,
        'node_occupancy': dict(occupied=occupied, empty=empty, median=median_cells),
        'training_summary': {
            'n_training': n_training,
            'eval_size': eval_size,
            'grid_size': (som_x_dim, som_y_dim),
            'final_qe': final_qe,
            'final_nqe': final_nqe,
            'final_te': final_te,
            'final_ratio': final_ratio,
            'mode': adaptive_config.get('mode', 'balanced'),
            'epochs_completed': len(checkpoints)
        }
    }


def characterize_som_clusters(data, som_clusters, marker_cols):
    """Compute per-cluster mean & std for markers."""
    data = data.copy()
    data['som_cluster'] = som_clusters
    cluster_records = []
    for cid in np.unique(som_clusters):
        subset = data[data['som_cluster'] == cid]
        record = {
            'som_cluster': cid,
            'cell_count': len(subset),
            'percentage': len(subset) / len(data) * 100
        }
        for m in marker_cols:
            if m in subset.columns:
                record[f"{m}_mean"] = subset[m].mean()
                record[f"{m}_std"] = subset[m].std()
        cluster_records.append(record)
    char_df = pd.DataFrame(cluster_records).sort_values('cell_count', ascending=False)
    return char_df


def _weighted_marker_means(sub, marker_cols, count_col='cell_count'):
    """Aggregate SOM-node marker means into a meta-cluster profile.

    Weighted by node occupancy: a node holding 7 cells must not count as much
    as one holding 5,000. An unweighted mean lets a handful of sparse outlier
    nodes dominate the profile and flip the phenotype call for the whole
    meta-cluster.
    """
    weights = pd.to_numeric(sub[count_col], errors='coerce').fillna(0.0).astype(float).values
    total = weights.sum()
    if total <= 0:
        return {col: float(pd.to_numeric(sub[col], errors='coerce').fillna(0.0).mean()) for col in marker_cols}
    return {
        col: float(np.average(pd.to_numeric(sub[col], errors='coerce').fillna(0.0).values, weights=weights))
        for col in marker_cols
    }


def meta_cluster_clusters(cluster_char_df, target_n=100, method='ward', metric='euclidean', marker_suffix='_mean',
                          prune_tiny_clusters=False, min_cluster_cells=None):
    """Reduce SOM clusters to meta-clusters via hierarchical clustering.

    ``prune_tiny_clusters`` force-merges under-sized meta-clusters into their
    nearest surviving neighbour. It defaults to False because pruning happens
    before the user ever sees the merge heatmap, so a genuinely rare phenotype
    would be absorbed with no way to overrule it. Enable it only when tiny
    outlier nodes are actively getting in the way.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    if len(cluster_char_df) <= target_n:
        cluster_char_df['meta_cluster'] = cluster_char_df['som_cluster']
        meta_map = dict(zip(cluster_char_df['som_cluster'], cluster_char_df['meta_cluster']))
        meta_df = cluster_char_df.copy()
        return meta_df, meta_map

    marker_cols = [c for c in cluster_char_df.columns if c.endswith(marker_suffix)]
    
    # Robust error handling: check if marker_cols is empty
    if not marker_cols:
        _print_flush(f"WARNING: No columns found ending with '{marker_suffix}'")
        _print_flush(f"Available columns: {list(cluster_char_df.columns)}")
        # Fallback: try to find any numeric columns that could be marker data
        numeric_cols = cluster_char_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['som_cluster', 'cell_count', 'percentage', 'meta_cluster']
        marker_cols = [c for c in numeric_cols if c not in exclude_cols]
        _print_flush(f"Fallback to numeric columns: {marker_cols[:5]}..." if marker_cols else "No fallback columns found")
        
        if not marker_cols:
            # Last resort: assign sequential meta-clusters
            _print_flush("WARNING: No marker columns found for meta-clustering. Using sequential assignment.")
            n_clusters = len(cluster_char_df)
            cluster_char_df = cluster_char_df.copy()
            cluster_char_df['meta_cluster'] = list(range(min(target_n, n_clusters)))[:n_clusters] if n_clusters > target_n else list(range(n_clusters))
            meta_map = dict(zip(cluster_char_df['som_cluster'], cluster_char_df['meta_cluster']))
            return cluster_char_df, meta_map
    
    feats = cluster_char_df[marker_cols].fillna(0)
    
    # Check if we have any valid data
    if feats.empty or feats.shape[1] == 0:
        raise ValueError(f"No valid feature data for meta-clustering. marker_cols={marker_cols}")
    
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    # Use appropriate clustering method with correct parameters
    if method == 'ward':
        model = AgglomerativeClustering(n_clusters=target_n, linkage='ward')
    elif method == 'complete':
        model = AgglomerativeClustering(n_clusters=target_n, linkage='complete', metric=metric)
    elif method == 'single':
        model = AgglomerativeClustering(n_clusters=target_n, linkage='single', metric=metric)
    else:  # Default to average
        model = AgglomerativeClustering(n_clusters=target_n, linkage='average', metric=metric)
    labels = model.fit_predict(feats_scaled)
    cluster_char_df = cluster_char_df.copy()
    cluster_char_df['meta_cluster'] = labels
    meta_map = dict(zip(cluster_char_df['som_cluster'], cluster_char_df['meta_cluster']))
    # Build meta characterization
    meta_list = []
    for mc in np.unique(labels):
        clust_ids = cluster_char_df[cluster_char_df['meta_cluster'] == mc]['som_cluster'].tolist()
        sub = cluster_char_df[cluster_char_df['meta_cluster'] == mc]
        rec = {
            'meta_cluster': mc,
            'cell_count': sub['cell_count'].sum(),
            'percentage': sub['percentage'].sum(),
            'som_clusters': clust_ids,
            'n_som_clusters': len(clust_ids)
        }
        rec.update(_weighted_marker_means(sub, marker_cols))
        meta_list.append(rec)
    meta_df = pd.DataFrame(meta_list).sort_values('cell_count', ascending=False)
    
    # --- PRUNING TINY CLUSTERS ---
    # Many users report seeing tiny clusters (<0.01% or <20 cells) alongside massive ones.
    # While scientifically truthful (outlier nodes), this is bad for phenotyping.
    # We will force-merge clusters smaller than 0.1% of the average cluster size into their nearest neighbor.
    
    if not prune_tiny_clusters:
        _print_flush(
            f"Tiny-cluster pruning disabled; keeping all {len(meta_df)} meta-clusters so rare "
            "populations survive to the merge heatmap."
        )
        return meta_df, meta_map

    try:
        total_cells = meta_df['cell_count'].sum()
        avg_cells = total_cells / target_n
        if min_cluster_cells is not None:
            min_threshold = float(min_cluster_cells)
        else:
            min_threshold = max(20, avg_cells * 0.05) # e.g. if avg is 1000, min is 50. Floor at 20 cells.

        tiny_clusters = meta_df[meta_df['cell_count'] < min_threshold]['meta_cluster'].values

        if len(tiny_clusters) > 0 and len(tiny_clusters) < len(meta_df):
            _print_flush(f"Pruning {len(tiny_clusters)} tiny meta-clusters (size < {min_threshold:.1f})...")

            # Get valid (large) clusters centroids
            valid_df = meta_df[~meta_df['meta_cluster'].isin(tiny_clusters)]
            # Match the metric space used by the Ward step above, otherwise a
            # high-dynamic-range marker decides the merge on its own.
            valid_feats = scaler.transform(valid_df[marker_cols].fillna(0))
            valid_ids = valid_df['meta_cluster'].values

            # Get tiny clusters centroids
            tiny_df = meta_df[meta_df['meta_cluster'].isin(tiny_clusters)]

            if len(valid_df) > 0:
                # Re-map each tiny cluster to nearest valid cluster
                updates = {}
                from scipy.spatial.distance import cdist

                for _, row in tiny_df.iterrows():
                    vec = scaler.transform(row[marker_cols].fillna(0).values.reshape(1, -1))
                    dists = cdist(vec, valid_feats, metric='euclidean') # Always use euclidean for nearest-neighbor cleanup
                    nearest_idx = np.argmin(dists)
                    nearest_id = valid_ids[nearest_idx]
                    updates[row['meta_cluster']] = nearest_id
                
                # Apply updates to meta_map
                for som_node, meta_id in list(meta_map.items()):
                    if meta_id in updates:
                        meta_map[som_node] = updates[meta_id]
                
                # Re-generate meta_df with merged clusters
                # (Recursive call or re-aggregation inline)
                # Inline re-aggregation for speed:
                cluster_char_df['meta_cluster'] = cluster_char_df['som_cluster'].map(meta_map)
                
                meta_list_v2 = []
                for mc in cluster_char_df['meta_cluster'].unique():
                    sub = cluster_char_df[cluster_char_df['meta_cluster'] == mc]
                    rec = {
                        'meta_cluster': mc,
                        'cell_count': sub['cell_count'].sum(),
                        'percentage': sub['percentage'].sum(),
                        'som_clusters': sub['som_cluster'].tolist(),
                        'n_som_clusters': len(sub)
                    }
                    rec.update(_weighted_marker_means(sub, marker_cols))
                    meta_list_v2.append(rec)
                meta_df = pd.DataFrame(meta_list_v2).sort_values('cell_count', ascending=False)
                _print_flush(f"Pruning complete. Reduced to {len(meta_df)} clusters.")
                
    except Exception as e:
        _print_flush(f"Warning: Tiny cluster pruning failed ({e}). Keeping original clusters.")

    return meta_df, meta_map


def create_meta_cluster_heatmap(char_df, plots_dir, title="Meta-Cluster Expression Heatmap", filename="meta_cluster_expression_heatmap.png"):
    """Generate clustered heatmap for meta-cluster marker expression."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    marker_cols = [c for c in char_df.columns if c.endswith('_mean')]
    if 'som_cluster' not in char_df.columns:
        # assume meta clusters; rename temporarily for uniform index naming
        working = char_df.rename(columns={'meta_cluster': 'som_cluster'})
    else:
        working = char_df.copy()
    heat = working.set_index('som_cluster')[marker_cols].copy()
    clean = {c: c.replace('_corr_norm_mean','').replace('_corr_mean','').replace('_mean','') for c in marker_cols}
    heat.rename(columns=clean, inplace=True)
    heat = heat.T
    # z-score rows
    heat_z = heat.apply(lambda x: (x - x.mean())/ (x.std() if x.std() else 1), axis=1).fillna(0)
    import seaborn as sns, matplotlib.pyplot as plt
    g = sns.clustermap(heat_z, method='ward', metric='euclidean', cmap='bwr', vmin=-3, vmax=3,
                       linewidths=0.3, row_cluster=True, col_cluster=True,
                       dendrogram_ratio=(0.05,0.2), cbar_pos=(0.02,0.02,0.01,0.12),
                       figsize=(max(8, heat_z.shape[1]*0.3), max(6, heat_z.shape[0]*0.3)))
    g.fig.suptitle(f"{title} (n={len(working)})", y=1.02)
    g.ax_heatmap.set_xlabel('Meta-Clusters')
    g.ax_heatmap.set_ylabel('Markers')
    g.ax_heatmap.yaxis.tick_right()
    g.ax_heatmap.yaxis.set_label_position('right')
    # relabel x with cell counts if available
    if 'cell_count' in working.columns:
        sizes = working.set_index('som_cluster')['cell_count']
        order = [int(t.get_text()) for t in g.ax_heatmap.get_xticklabels()]
        newlabs = [f"{cid}\n({sizes.get(cid,0):,})" for cid in order]
        g.ax_heatmap.set_xticklabels(newlabs, rotation=90)
    out_path = plots_dir / filename
    g.fig.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    # Return ordered cluster IDs (meta-cluster IDs) in displayed order
    order_ids = [int(t.get_text().split('\n')[0]) for t in g.ax_heatmap.get_xticklabels()]
    return order_ids, out_path


def launch_som_phenotype_assignment_app(char_df, reordered_cluster_ids, meta_cluster_phenotypes):
    """Launch the professional phenotype assignment application (SOM-style interface).

    This is an exact replication of the original SOM phenotype assignment studio UI.
    It has been generalized to automatically work with either SOM or Leiden clusters
    by detecting an available cluster ID column (prefers 'som_cluster', falls back
    to 'leiden_cluster'). To preserve a byte-for-byte identical interface layout,
    only minimal internal adaptation (column aliasing) was added.

    Parameters
    ----------
    char_df : pd.DataFrame
        Cluster characterization dataframe containing '*_mean' marker columns and a
        cluster ID column ('som_cluster' or 'leiden_cluster').
    reordered_cluster_ids : list[int]
        Cluster IDs in the order displayed in the prior heatmap (maintains visual parity).
    meta_cluster_phenotypes : dict[int,str]
        Pre-existing cluster -> phenotype mapping to preload (can be empty dict).

    Returns
    -------
    dict[int,str]
        Mapping cluster_id -> phenotype label.
    """
    # Detect and normalize cluster ID column name to 'som_cluster' (internal expectation)
    detected_cluster_col = None
    for cand in ['som_cluster', 'leiden_cluster']:
        if cand in char_df.columns:
            detected_cluster_col = cand
            break
    if detected_cluster_col is None:
        raise ValueError("Expected either 'som_cluster' or 'leiden_cluster' column in characterization dataframe.")
    # If using Leiden, create a temporary alias so the original UI code (which expects som_cluster) remains untouched.
    if detected_cluster_col != 'som_cluster':
        char_df = char_df.rename(columns={detected_cluster_col: 'som_cluster'})
    # (From this point forward the code is copied verbatim from the original SOM implementation.)

    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import seaborn as sns  # noqa: F401 (kept for parity & potential extension)
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import dendrogram, linkage

    class PhenotypeAssignmentApp:
        def __init__(self, char_df, reordered_cluster_ids, meta_cluster_phenotypes):
            self.root = tk.Tk()
            self.root.title("SPACEMAP V4 - Phenotype Assignment Studio")
            self.root.geometry("1900x1100")
            self.root.configure(bg='#f0f0f0')
            self.char_df = char_df
            self.reordered_cluster_ids = reordered_cluster_ids
            self.meta_cluster_phenotypes = meta_cluster_phenotypes.copy()
            self.colors = {
                'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#F18F01',
                'warning': '#C73E1D', 'assigned': '#FF6B6B', 'unassigned': '#4ECDC4',
                'selected': '#FFE66D', 'background': '#f8f9fa', 'card': '#ffffff'
            }
            self.selected_clusters = []
            self.current_cluster_order = self.reordered_cluster_ids
            self.assignments = []
            self.editing_assignment = None
            self.total_clusters = len(self.reordered_cluster_ids)
            self.assigned_clusters = set(self.meta_cluster_phenotypes.keys())
            self.unassigned_clusters = set(self.reordered_cluster_ids) - self.assigned_clusters
            self.load_existing_assignments()
            self.setup_gui()
            self.plot_heatmap()
            self.update_statistics()

        def load_existing_assignments(self):
            phenotype_groups = {}
            for cluster_id, phenotype in self.meta_cluster_phenotypes.items():
                phenotype_groups.setdefault(phenotype, []).append(cluster_id)
            for phenotype, clusters in phenotype_groups.items():
                total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(clusters)]['cell_count'])
                self.assignments.append({'clusters': clusters, 'phenotype': phenotype, 'cell_count': total_cells})

        def setup_gui(self):
            style = ttk.Style(); style.theme_use('clam')
            style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=self.colors['primary'])
            style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['secondary'])
            style.configure('Stats.TLabel', font=('Segoe UI', 11), foreground=self.colors['success'])
            style.configure('Card.TFrame', background=self.colors['card'], relief='raised', borderwidth=1)
            main_container = ttk.Frame(self.root, style='Card.TFrame'); main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
            header = ttk.Frame(main_container); header.pack(fill=tk.X, pady=(0, 15))
            ttk.Label(header, text="Phenotype Assignment Studio", style='Title.TLabel').pack(side=tk.LEFT)
            self.stats_frame = ttk.Frame(header); self.stats_frame.pack(side=tk.RIGHT)
            self.paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL); self.paned_window.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            self.left_panel = ttk.LabelFrame(self.paned_window, text="Interactive Cluster Heatmap", style='Card.TFrame'); self.paned_window.add(self.left_panel, weight=3)
            self.right_panel = ttk.LabelFrame(self.paned_window, text="Assignment Control Panel", style='Card.TFrame'); self.right_panel.configure(padding="10"); self.paned_window.add(self.right_panel, weight=1)
            self.setup_heatmap_panel(); self.setup_control_panel()

        def setup_heatmap_panel(self):
            controls_frame = ttk.Frame(self.left_panel); controls_frame.pack(fill=tk.X, padx=5, pady=5)
            self.cluster_x = tk.BooleanVar(value=True); self.cluster_y = tk.BooleanVar(value=True)
            ttk.Checkbutton(controls_frame, text="Cluster X-axis", variable=self.cluster_x).pack(side=tk.LEFT, padx=(0, 15))
            ttk.Checkbutton(controls_frame, text="Cluster Y-axis", variable=self.cluster_y).pack(side=tk.LEFT, padx=(0, 15))
            ttk.Button(controls_frame, text="Update Heatmap", command=self.plot_heatmap).pack(side=tk.LEFT, padx=(15, 0))
            legend_frame = ttk.Frame(controls_frame); legend_frame.pack(side=tk.RIGHT)
            ttk.Label(legend_frame, text="Selected", foreground='#DAA520', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            ttk.Label(legend_frame, text="Assigned", foreground='#DC143C', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            ttk.Label(legend_frame, text="Unassigned", foreground='#228B22', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            self.heatmap_fig = Figure(figsize=(16, 12), facecolor='white')
            self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, self.left_panel)
            self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        def setup_control_panel(self):
            selection_frame = ttk.LabelFrame(self.right_panel, text="Current Selection"); selection_frame.pack(fill=tk.X, pady=(0, 10))
            self.selected_label = ttk.Label(selection_frame, text="No clusters selected", foreground=self.colors['secondary'], font=('Arial', 10, 'bold')); self.selected_label.pack(pady=5)
            assignment_frame = ttk.LabelFrame(self.right_panel, text="Phenotype Assignment"); assignment_frame.pack(fill=tk.X, pady=(0, 10))
            ttk.Label(assignment_frame, text="Phenotype Name:").pack(anchor=tk.W, pady=(5, 0))
            self.phenotype_var = tk.StringVar(); self.phenotype_entry = ttk.Entry(assignment_frame, textvariable=self.phenotype_var, font=('Arial', 11)); self.phenotype_entry.pack(fill=tk.X, pady=(0, 5))
            buttons_frame = ttk.Frame(assignment_frame); buttons_frame.pack(fill=tk.X, pady=5)
            self.assign_btn = ttk.Button(buttons_frame, text="Add Assignment", command=self.add_assignment); self.assign_btn.pack(side=tk.LEFT, padx=(0, 5))
            self.update_btn = ttk.Button(buttons_frame, text="Update", command=self.update_assignment, state='disabled'); self.update_btn.pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(buttons_frame, text="Clear", command=self.clear_selection).pack(side=tk.RIGHT)
            quick_frame = ttk.Frame(assignment_frame); quick_frame.pack(fill=tk.X, pady=5)
            ttk.Button(quick_frame, text="Select Unassigned", command=self.select_unassigned).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_frame, text="Generate Heatmap", command=self.show_phenotype_heatmap).pack(side=tk.RIGHT)
            table_frame = ttk.LabelFrame(self.right_panel, text="Assignment Table"); table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            table_container = ttk.Frame(table_frame); table_container.pack(fill=tk.BOTH, expand=True, pady=5)
            columns = ('ID', 'Phenotype', 'Clusters', 'Cells')
            # Enable extended (multi) selection explicitly
            self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=12, selectmode='extended')
            self.tree.heading('ID', text='#'); self.tree.heading('Phenotype', text='Phenotype'); self.tree.heading('Clusters', text='Count'); self.tree.heading('Cells', text='Cells')
            self.tree.column('ID', width=30, anchor='center'); self.tree.column('Phenotype', width=120, anchor='w'); self.tree.column('Clusters', width=60, anchor='center'); self.tree.column('Cells', width=80, anchor='center')
            scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview); self.tree.configure(yscrollcommand=scrollbar.set)
            self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.tree.bind('<Button-1>', self.on_assignment_click); self.tree.bind('<Double-1>', self.on_assignment_double_click)
            # Add Ctrl+A select-all behavior
            self.tree.bind('<Control-a>', self.on_select_all_assignments)
            table_controls = ttk.Frame(table_frame); table_controls.pack(fill=tk.X, pady=(0, 5))
            ttk.Button(table_controls, text="Delete Selected", command=self.delete_assignment).pack(side=tk.LEFT)
            ttk.Button(table_controls, text="Save & Close", command=self.save_and_close).pack(side=tk.RIGHT)
            self.update_assignment_table()

        def update_statistics(self):
            for widget in self.stats_frame.winfo_children(): widget.destroy()
            assigned_count = len(self.assigned_clusters); unassigned_count = len(self.unassigned_clusters)
            total_cells_assigned = sum(self.char_df[self.char_df['som_cluster'].isin(self.assigned_clusters)]['cell_count'])
            total_cells = self.char_df['cell_count'].sum()
            ttk.Label(self.stats_frame, text=f"Total Clusters: {self.total_clusters} | Assigned: {assigned_count} | Unassigned: {unassigned_count}", style='Stats.TLabel').pack()
            ttk.Label(self.stats_frame, text=f"Cells Assigned: {total_cells_assigned:,}/{total_cells:,} ({100*total_cells_assigned/total_cells:.1f}%)", style='Stats.TLabel').pack()

        def plot_heatmap(self):
            self.heatmap_fig.clear()
            marker_cols = [c for c in self.char_df.columns if c.endswith('_mean')]
            heatmap_data = self.char_df.set_index('som_cluster')[marker_cols].copy()
            clean = {col: col.replace('_corr_norm_mean','').replace('_corr_mean','').replace('_mean','') for col in marker_cols}
            heatmap_data.rename(columns=clean, inplace=True)
            heatmap_data = heatmap_data.T
            heatmap_z = heatmap_data.apply(lambda x: (x - x.mean())/x.std(), axis=1).fillna(0)
            if self.cluster_x.get() or self.cluster_y.get():
                data_for_clustering = heatmap_z.copy()
                if self.cluster_x.get():
                    col_linkage = linkage(data_for_clustering.T, method='ward', metric='euclidean')
                    col_order = dendrogram(col_linkage, no_plot=True)['leaves']
                    self.current_cluster_order = [heatmap_z.columns[i] for i in col_order]
                else:
                    self.current_cluster_order = sorted(heatmap_z.columns)
                if self.cluster_y.get():
                    row_linkage = linkage(data_for_clustering, method='ward', metric='euclidean')
                    row_order = dendrogram(row_linkage, no_plot=True)['leaves']
                    marker_order = [heatmap_z.index[i] for i in row_order]
                else:
                    marker_order = list(heatmap_z.index)
                heatmap_z_ordered = heatmap_z.loc[marker_order, self.current_cluster_order]
            else:
                self.current_cluster_order = sorted(heatmap_z.columns)
                heatmap_z_ordered = heatmap_z[self.current_cluster_order]
            ax = self.heatmap_fig.add_subplot(111)
            im = ax.imshow(heatmap_z_ordered.values, cmap='bwr', aspect='auto', vmin=-3, vmax=3)
            for i, cluster_id in enumerate(self.current_cluster_order):
                if cluster_id in self.selected_clusters:
                    ax.axvline(x=i-0.5, color='#FFD700', linewidth=4, alpha=0.9); ax.axvline(x=i+0.5, color='#FFD700', linewidth=4, alpha=0.9)
                elif cluster_id in self.assigned_clusters:
                    ax.axvline(x=i-0.5, color='#FF4444', linewidth=3, alpha=0.7); ax.axvline(x=i+0.5, color='#FF4444', linewidth=3, alpha=0.7)
                elif cluster_id in self.unassigned_clusters:
                    ax.axvline(x=i-0.5, color='#44AA44', linewidth=2, alpha=0.6); ax.axvline(x=i+0.5, color='#44AA44', linewidth=2, alpha=0.6)
            ax.set_title('Interactive Phenotype Assignment - Click Clusters to Select\nYellow: Selected | Red: Assigned | Green: Unassigned', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Cluster Numbers', fontsize=12, fontweight='bold'); ax.set_ylabel('Markers', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(self.current_cluster_order)))
            ax.set_xticklabels([str(int(c)) for c in self.current_cluster_order], rotation=90, fontsize=9, ha='center')
            ax.set_yticks(range(len(heatmap_z_ordered.index))); ax.set_yticklabels(heatmap_z_ordered.index, fontsize=9)
            cbar = self.heatmap_fig.colorbar(im, ax=ax, shrink=0.9, pad=0.005, aspect=40, fraction=0.03); cbar.set_label('Z-score', fontsize=11)
            self.heatmap_fig.tight_layout(pad=0.5, rect=(0.02, 0.05, 0.98, 0.95))
            def on_click(event):
                if event.inaxes == ax and event.xdata is not None:
                    cluster_idx = int(round(event.xdata))
                    if 0 <= cluster_idx < len(self.current_cluster_order):
                        cluster_id = self.current_cluster_order[cluster_idx]
                        if cluster_id in self.assigned_clusters and cluster_id not in self.selected_clusters:
                            assigned_phenotype = self.meta_cluster_phenotypes.get(cluster_id, "Unknown")
                            self.highlight_assignment_by_cluster(cluster_id)
                            messagebox.showinfo("Cluster Already Assigned", f"Cluster {cluster_id} is already assigned to: '{assigned_phenotype}'\n\nThe assignment has been highlighted in the table.\nDouble-click the assignment to edit it.")
                            return
                        if cluster_id in self.selected_clusters:
                            self.selected_clusters.remove(cluster_id)
                        else:
                            self.selected_clusters.append(cluster_id)
                        self.update_selected_display(); self.plot_heatmap()
            self.heatmap_canvas.mpl_connect('button_press_event', on_click); self.heatmap_canvas.draw()

        # (Alias original method names for parity with SOM code)
        def update_selected_display(self): self._update_selected_display()
        def _update_selected_display(self):
            if self.selected_clusters:
                if self.editing_assignment is not None:
                    text = f"EDITING: {self.selected_clusters} (Click Update to save)"; color = self.colors['warning']
                else:
                    text = f"Selected: {self.selected_clusters}"; color = self.colors['secondary']
                self.selected_label.config(text=text, foreground=color)
            else:
                self.selected_label.config(text="No clusters selected", foreground='gray')

        def clear_selection(self): self._clear_selection()
        def _clear_selection(self):
            self.selected_clusters.clear(); self.editing_assignment = None
            self.assign_btn.config(state='normal'); self.update_btn.config(state='disabled')
            self.phenotype_var.set(""); self.update_selected_display(); self.plot_heatmap()
            for item in self.tree.selection(): self.tree.selection_remove(item)

        def select_unassigned(self):
            self.selected_clusters = list(self.unassigned_clusters); self.update_selected_display(); self.plot_heatmap()

        def add_assignment(self): self._add_assignment()
        def _add_assignment(self):
            if not self.selected_clusters:
                messagebox.showwarning("Warning", "Please select clusters first"); return
            phenotype = self.phenotype_var.get().strip()
            if not phenotype:
                messagebox.showwarning("Warning", "Please enter a phenotype name"); return
            conflicts = [c for c in self.selected_clusters if c in self.assigned_clusters]
            if conflicts:
                if not messagebox.askyesno("Cluster Conflict", f"Clusters {conflicts} are already assigned.\n\nDo you want to reassign them?"):
                    return
                self.remove_clusters_from_assignments(conflicts)
            total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(self.selected_clusters)]['cell_count'])
            self.assignments.append({'clusters': list(self.selected_clusters), 'phenotype': phenotype, 'cell_count': total_cells})
            for cluster_id in self.selected_clusters: self.meta_cluster_phenotypes[cluster_id] = phenotype
            self.assigned_clusters.update(self.selected_clusters); self.unassigned_clusters -= set(self.selected_clusters)
            self.clear_selection(); self.update_assignment_table(); self.update_statistics()

        # Method aliases for backward compatibility
        def assign_btn_callback(self): self._add_assignment()
        def add_assignment_public(self): self._add_assignment()
        def add_assignment_alias(self): self._add_assignment()

        def update_assignment(self):
            if self.editing_assignment is None: return
            phenotype = self.phenotype_var.get().strip()
            if not phenotype:
                messagebox.showwarning("Warning", "Please enter a phenotype name"); return
            old_assignment = self.assignments[self.editing_assignment]; old_clusters = set(old_assignment['clusters']); new_clusters = set(self.selected_clusters)
            self.assigned_clusters -= old_clusters; self.unassigned_clusters |= old_clusters
            for cluster_id in old_clusters:
                if cluster_id in self.meta_cluster_phenotypes: del self.meta_cluster_phenotypes[cluster_id]
            conflicts = [c for c in new_clusters if c in self.assigned_clusters]
            if conflicts:
                if not messagebox.askyesno("Cluster Conflict", f"Clusters {conflicts} are already assigned.\n\nDo you want to reassign them?"): return
                self.remove_clusters_from_assignments(conflicts)
            total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(self.selected_clusters)]['cell_count'])
            self.assignments[self.editing_assignment] = {'clusters': list(self.selected_clusters), 'phenotype': phenotype, 'cell_count': total_cells}
            for cluster_id in self.selected_clusters: self.meta_cluster_phenotypes[cluster_id] = phenotype
            self.assigned_clusters.update(self.selected_clusters); self.unassigned_clusters -= set(self.selected_clusters)
            self.clear_selection(); self.update_assignment_table(); self.update_statistics()

        def remove_clusters_from_assignments(self, clusters_to_remove):
            for i, assignment in enumerate(list(self.assignments)):
                old_clusters = set(assignment['clusters']); new_clusters = old_clusters - set(clusters_to_remove)
                if new_clusters != old_clusters:
                    if new_clusters:
                        total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(new_clusters)]['cell_count'])
                        self.assignments[i] = {'clusters': list(new_clusters), 'phenotype': assignment['phenotype'], 'cell_count': total_cells}
                    else:
                        self.assignments.pop(i)
            for cluster_id in clusters_to_remove:
                if cluster_id in self.meta_cluster_phenotypes: del self.meta_cluster_phenotypes[cluster_id]
            self.assigned_clusters -= set(clusters_to_remove); self.unassigned_clusters |= set(clusters_to_remove)

        def update_assignment_table(self):
            for item in self.tree.get_children(): self.tree.delete(item)
            for i, assignment in enumerate(self.assignments):
                values = (i + 1, assignment['phenotype'][:15] + ('...' if len(assignment['phenotype']) > 15 else ''), len(assignment['clusters']), f"{assignment['cell_count']:,}")
                self.tree.insert('', tk.END, values=values)

        def on_assignment_click(self, event):
            # Support multi-selection: aggregate clusters from all selected assignments
            sel = self.tree.selection()
            if not sel:
                return
            # During single-click editing state, only update selection display (don't trigger edit mode)
            selected_clusters = set()
            for item in sel:
                values = self.tree.item(item)['values']
                if not values:
                    continue
                assignment_idx = int(values[0]) - 1
                if 0 <= assignment_idx < len(self.assignments):
                    selected_clusters.update(self.assignments[assignment_idx]['clusters'])
            self.selected_clusters = sorted(selected_clusters)
            # Exit editing mode if multi-selection >1
            if len(sel) > 1 and self.editing_assignment is not None:
                self.editing_assignment = None
                self.assign_btn.config(state='normal'); self.update_btn.config(state='disabled')
            self.update_selected_display(); self.plot_heatmap()

        def on_select_all_assignments(self, event=None):
            # Select every assignment row (Ctrl+A) and update selection clusters
            for item in self.tree.get_children():
                self.tree.selection_add(item)
            self.on_assignment_click(event)
            return 'break'

        def on_assignment_double_click(self, event):
            item = self.tree.selection()[0] if self.tree.selection() else None
            if item:
                values = self.tree.item(item)['values']; assignment_idx = int(values[0]) - 1; assignment = self.assignments[assignment_idx]
                self.editing_assignment = assignment_idx; self.selected_clusters = list(assignment['clusters']); self.phenotype_var.set(assignment['phenotype'])
                self.assign_btn.config(state='disabled'); self.update_btn.config(state='normal'); self.update_selected_display(); self.plot_heatmap()

        def highlight_assignment_by_cluster(self, cluster_id):
            for i, assignment in enumerate(self.assignments):
                if cluster_id in assignment['clusters']:
                    children = self.tree.get_children()
                    if i < len(children):
                        self.tree.selection_set(children[i]); self.tree.focus(children[i]); self.tree.see(children[i])
                    break

        def delete_assignment(self):
            selected_items = list(self.tree.selection())
            if not selected_items:
                messagebox.showwarning("Warning", "Please select assignment(s) to delete"); return
            # Collect indices & phenotypes
            to_delete_indices = []
            phenos = []
            clusters_to_unassign = set()
            for item in selected_items:
                values = self.tree.item(item)['values']
                if not values:
                    continue
                idx = int(values[0]) - 1
                if 0 <= idx < len(self.assignments):
                    to_delete_indices.append(idx)
                    phenos.append(self.assignments[idx]['phenotype'])
                    clusters_to_unassign.update(self.assignments[idx]['clusters'])
            if not to_delete_indices:
                return
            to_delete_indices = sorted(set(to_delete_indices), reverse=True)
            confirm_msg = (
                f"Delete {len(to_delete_indices)} assignment(s)?\n\n" +
                ("Phenotypes: " + ", ".join(phenos[:10]) + (" ..." if len(phenos) > 10 else "")) +
                f"\nClusters to unassign: {len(clusters_to_unassign)}"
            )
            if not messagebox.askyesno("Confirm Delete", confirm_msg):
                return
            # Perform deletions
            for idx in to_delete_indices:
                assignment = self.assignments[idx]
                for cluster_id in assignment['clusters']:
                    if cluster_id in self.meta_cluster_phenotypes:
                        del self.meta_cluster_phenotypes[cluster_id]
                self.assigned_clusters -= set(assignment['clusters'])
                self.unassigned_clusters |= set(assignment['clusters'])
                self.assignments.pop(idx)
            self.clear_selection(); self.update_assignment_table(); self.update_statistics(); self.plot_heatmap()

        def show_phenotype_heatmap(self):
            if not self.meta_cluster_phenotypes:
                messagebox.showinfo("Info", "No phenotype assignments to display"); return
            heatmap_window = tk.Toplevel(self.root); heatmap_window.title("Phenotype Expression Heatmap"); heatmap_window.geometry("1200x800")
            pheno_df = self.char_df.copy(); pheno_df['phenotype'] = pheno_df['som_cluster'].map(self.meta_cluster_phenotypes).fillna('Unassigned')
            marker_cols = [col for col in pheno_df.columns if col.endswith('_mean')]
            pheno_means = pheno_df.groupby('phenotype')[marker_cols].mean()
            clean_names = {col: col.replace('_corr_norm_mean','').replace('_corr_mean','').replace('_mean','') for col in marker_cols}
            pheno_means = pheno_means.rename(columns=clean_names).T
            pheno_means_zscore = pheno_means.apply(lambda x: (x - x.mean()) / x.std(), axis=1).fillna(0)
            fig = Figure(figsize=(12, 8), facecolor='white'); ax = fig.add_subplot(111)
            im = ax.imshow(pheno_means_zscore.values, cmap='bwr', aspect='auto', vmin=-3, vmax=3)
            ax.set_title('Phenotype Expression Heatmap', fontsize=16, fontweight='bold', pad=20); ax.set_xlabel('Phenotypes', fontsize=14); ax.set_ylabel('Markers', fontsize=14)
            pheno_counts = pheno_df.groupby('phenotype')['cell_count'].sum()
            x_labels = [f"{pheno}\n({pheno_counts.get(pheno, 0):,})" for pheno in pheno_means_zscore.columns]
            ax.set_xticks(range(len(pheno_means_zscore.columns))); ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
            ax.set_yticks(range(len(pheno_means_zscore.index))); ax.set_yticklabels(pheno_means_zscore.index, fontsize=10)
            cbar = fig.colorbar(im, ax=ax, shrink=0.7); cbar.set_label('Z-score', fontsize=12)
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FCTA
            canvas = _FCTA(fig, heatmap_window); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas.draw()

        def save_and_close(self):
            if not self.meta_cluster_phenotypes:
                messagebox.showwarning("Warning", "No assignments made"); return
            summary = f"ASSIGNMENT SUMMARY\n{'='*50}\n\n"
            summary += f"Total assignments: {len(self.assignments)}\n"
            summary += f"Clusters assigned: {len(self.assigned_clusters)}/{self.total_clusters}\n"
            summary += f"Clusters remaining: {len(self.unassigned_clusters)}\n\n"
            summary += "PHENOTYPE BREAKDOWN:\n"
            for assignment in self.assignments:
                summary += f"   {assignment['phenotype']}: {len(assignment['clusters'])} clusters, {assignment['cell_count']:,} cells\n"
            if self.unassigned_clusters:
                summary += f"\nUnassigned clusters: {sorted(self.unassigned_clusters)}\n"
            summary += "\nReady to proceed to Step 5!"
            messagebox.showinfo("Assignment Complete", summary); self.root.destroy()

        # Backwards-compatible alias names (some notebook edits may have referenced these)
        add_assignment = add_assignment  # noqa

        # Maintain SOM method names used in original code
        def run(self): self.root.mainloop()

    print("="*80); print("LAUNCHING SPACEMAP V4 - PHENOTYPE ASSIGNMENT STUDIO"); print("="*80)
    print("\nPROFESSIONAL FEATURES:\n  * Interactive cluster selection with visual feedback\n  * Red = assigned, Green = unassigned, Yellow = selected\n  * Real-time statistics & progress tracking\n  * Assignment editing & management\n  * Phenotype heatmap generation\n  * Professional control panel")
    app = PhenotypeAssignmentApp(char_df, reordered_cluster_ids, meta_cluster_phenotypes); app.run()
    print("Application closed!")
    if app.meta_cluster_phenotypes:
        print(f"Total assignments saved: {len(app.meta_cluster_phenotypes)}\nReady to proceed to Step 5!")
    else:
        print("No assignments were made.")
    return app.meta_cluster_phenotypes


def apply_som_phenotypes(data, som_clusters, meta_cluster_map, meta_cluster_phenotypes, output_base_dir, cell_id_column, method_prefix='som'):
    """Apply SOM->meta-cluster->phenotype mapping & persist unified + summary files.
    
    Only adds SOM-specific columns to the unified file, preserves existing data.

    Parameters
    ----------
    data : pd.DataFrame (must include som_cluster column or som_clusters param provided)
    som_clusters : np.ndarray
        SOM cluster assignments (if not already in data['som_cluster']).
    meta_cluster_map : dict
        som_cluster -> meta_cluster mapping.
    meta_cluster_phenotypes : dict
        meta_cluster -> phenotype label.
    output_base_dir : Path-like
        Root outputs directory (default_config.output_dir).
    method_prefix : str
        Prefix for phenotype column (e.g. 'som', future 'leiden').
    cell_id_column : str
        Name of the cell ID column for proper data merging (required parameter).
    """
    output_base_dir = Path(output_base_dir)
    phenotyping_dir = output_base_dir
    phenotyping_dir.mkdir(exist_ok=True)
    
    data = data.copy()
    if 'som_cluster' not in data.columns:
        data['som_cluster'] = som_clusters

    # Add meta_cluster assignments
    data['meta_cluster'] = data['som_cluster'].map(meta_cluster_map)
    
    som_to_meta = meta_cluster_map
    final_map = {}
    for sc in data['som_cluster'].unique():
        meta = som_to_meta.get(sc)
        if meta is None:
            final_map[sc] = 'Unassigned'
        else:
            final_map[sc] = meta_cluster_phenotypes.get(meta, 'Unassigned')
    phenotype_col = f'phenotype_{method_prefix}'
    data[phenotype_col] = data['som_cluster'].map(final_map)

    summary = data.groupby(phenotype_col).agg({
        'som_cluster':'nunique', phenotype_col:'count'
    }).rename(columns={'som_cluster':'num_som_clusters', phenotype_col:'cell_count'})
    summary['percentage'] = (summary['cell_count']/len(data)*100).round(2)
    summary = summary.sort_values('cell_count', ascending=False)

    unified_path = phenotyping_dir / 'all_samples_combined_corrected_phenotyped.csv'
    
    # Define SOM-specific columns to add (no normalized markers)
    som_specific_cols = ['som_cluster', 'meta_cluster', phenotype_col]
    
    # Ensure cell_id is available for proper merging
    if cell_id_column not in data.columns:
        raise ValueError(f"{cell_id_column} column required for proper data merging")
    
    if unified_path.exists():
        existing = pd.read_csv(unified_path)
        if cell_id_column in existing.columns and len(existing) > 0:
            print(f"Merging SOM results with existing unified file...")
            
            # Use cell_id-based merge to properly anchor results
            existing = existing.set_index(cell_id_column)
            data_indexed = data.set_index(cell_id_column)
            
            # Only add/update SOM-specific columns
            for col in som_specific_cols:
                if col in data_indexed.columns:
                    existing[col] = data_indexed[col]
                    print(f" Added/updated column: {col}")
            
            # Reset index to get cell_id column back as a column (avoid duplicates)
            if cell_id_column in existing.columns:
                existing = existing.drop(columns=[cell_id_column])
            unified_df = existing.reset_index()
        else:
            print(f"Creating new unified file with SOM results...")
            # Filter data to only include essential columns (no normalized markers)
            essential_cols = [cell_id_column] + [c for c in data.columns if not c.endswith('_norm')]
            unified_df = data[essential_cols].copy()
    else:
        print(f"Creating new unified file with SOM results...")
        # Filter data to only include essential columns (no normalized markers)
        essential_cols = [cell_id_column] + [c for c in data.columns if not c.endswith('_norm')]
        unified_df = data[essential_cols].copy()
    
    unified_df.to_csv(unified_path, index=False)
    print(f"Unified file saved: {unified_path} ({len(unified_df):,} rows)")

    # summary + mappings
    summary_file = phenotyping_dir / f'{method_prefix}_phenotype_summary.csv'
    summary.to_csv(summary_file)
    mapping_file = phenotyping_dir / f'{method_prefix}_phenotype_mapping.csv'
    pd.DataFrame(list(meta_cluster_phenotypes.items()), columns=['meta_cluster', phenotype_col]).to_csv(mapping_file, index=False)
    som_meta_file = phenotyping_dir / 'som_to_meta_cluster_mapping.csv'
    pd.DataFrame(list(meta_cluster_map.items()), columns=['som_cluster','meta_cluster']).to_csv(som_meta_file, index=False)

    return {
        'unified_file': unified_path,
        'summary_file': summary_file,
        'mapping_file': mapping_file,
        'som_meta_file': som_meta_file,
        'summary_df': summary,
        'data': data
    }

