"""
Batch correction utilities for SPACEMAP V7 pipeline.

This module provides utilities for:
- Batch correction using SOM clustering and ComBat
- Quality control plotting
- Marker selection and validation
"""

import os
# Fix for protobuf compatibility which causes issues with UMAP/TensorFlow
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# UMAP (optional)
# Lazy loaded to avoid protobuf issues
UMAP_AVAILABLE = True 
umap = None

# SOM implementation
try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    print("Warning: minisom not installed. SOM functionality will be limited.")
    MINISOM_AVAILABLE = False
    MiniSom = None

# Combat batch correction
try:
    from combat.pycombat import pycombat
    COMBAT_AVAILABLE = True
except ImportError:
    print("Warning: pycombat not installed. Using alternative batch correction.")
    COMBAT_AVAILABLE = False
    pycombat = None


class BatchCorrector:
    """Handles batch correction using direct or SOM-clustered ComBat correction (cyCombine approach).
    
    This implementation matches the R cyCombine batch_correct() function:
    - Pre-normalization with StandardScaler (matching R's norm_method="scale")
    - SOM-based micro-clustering to group phenotypically similar cells
    - Parametric/non-parametric ComBat correction within each micro-cluster
    - Post-correction clipping to ensure non-negative values
    """
    
    def __init__(self, markers, normalization_method='scale', som_grid_size=(8, 8),
                 som_iterations=100000, random_seed=42, verbose=True,
                 mode='som_clustered', min_cells_per_cluster=10, covariates=None, parametric=True,
                 arcsinh_cofactor=5.0, correction_mode='monotonic'):
        """
        Parameters
        ----------
        correction_mode : {'monotonic', 'per_cluster'}
            How the estimated batch effect is *applied* to cells.

            ``'per_cluster'`` is the literal cyCombine translation: ComBat runs
            independently inside every SOM node and its output is written back
            per node. This makes the correction non-monotonic within a batch --
            two cells from the same batch with the *same* input value land in
            different nodes and come out with *different* corrected values. On
            markers with a large true-negative population (a spike at zero,
            which is typical of imaging/segmentation data) that reorders cells
            and blurs the negative/positive boundary that defines a phenotype.

            ``'monotonic'`` (default) keeps the SOM for what it is actually
            needed for -- estimating the batch effect within matched cell
            populations, so a batch that simply contains more of one cell type
            is not mistaken for a batch that is brighter -- but then applies the
            result as a single affine transform per (batch, marker). Cell
            ranking within a batch is preserved exactly, so no marker can be
            reordered by the correction.

            Use ``'per_cluster'`` to reproduce results from an earlier run.
        """
        self.markers = markers
        self.normalization_method = normalization_method  # 'scale', 'none'
        self.random_seed = random_seed
        self.verbose = verbose
        self.som_grid_size = som_grid_size
        self.som_iterations = som_iterations
        self.mode = 'som_clustered'  # cyCombine mode
        self.min_cells_per_cluster = min_cells_per_cluster
        self.covariates = covariates or []
        self.parametric = parametric
        self.arcsinh_cofactor = arcsinh_cofactor
        if correction_mode not in ('monotonic', 'per_cluster'):
            raise ValueError("correction_mode must be 'monotonic' or 'per_cluster'")
        self.correction_mode = correction_mode
        
        # Store normalization parameters for reference
        self.scaler = None
        self.normalization_params = {}
        self.run_summary = {}
        
    def correct_batch_effects(self, data):
        """
        Perform batch correction using ComBat (matching R cyCombine approach).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data with marker columns and batch information
            
        Returns:
        --------
        pandas.DataFrame
            Data with batch-corrected marker columns (suffix '_corr') in capped ArcSinh space
        """
        print(f"Starting batch correction ({self.mode} mode)...")
        print(f"  ArcSinh cofactor: {self.arcsinh_cofactor}")
        print(f"  Normalization method: {self.normalization_method}")
        
        # Prepare data and apply ArcSinh transformation (matching R cyCombine workflow)
        if self.verbose:
            print(f"Applying ArcSinh transformation with cofactor {self.arcsinh_cofactor}...")
        # Force numeric coercion to avoid string division TypeErrors
        numeric_marker_data = data[self.markers].apply(pd.to_numeric, errors='coerce').fillna(0)
        marker_data = np.arcsinh(numeric_marker_data / self.arcsinh_cofactor)
        batches = data['batch'].values if 'batch' in data.columns else data['batch_id'].values
        
        # Check batch distribution
        unique_batches = np.unique(batches)
        batch_counts = pd.Series(batches).value_counts()
        print(f"Batch distribution: {len(unique_batches)} batches")
        if self.verbose:
            for batch, count in batch_counts.items():
                print(f"  {batch}: {count:,} cells")
        
        # Initialize run summary
        self.run_summary = {
            'mode': self.mode,
            'n_cells_input': len(data),
            'n_markers': len(self.markers),
            'markers': self.markers,
            'covariates': self.covariates,
            'som': {
                'grid_size': self.som_grid_size,
                'n_clusters': self.som_grid_size[0] * self.som_grid_size[1]
            },
            'combat': {
                'n_clusters_total': 0,
                'n_clusters_skipped': 0,
                'n_clusters_covar_fallback': 0,
                'covariates': self.covariates,
                'skip_details': []
            },
            'capping': {
                'n_negative_clipped': 0,
                'n_capped': 0
            }
        }
        
        # Apply ComBat correction (cyCombine SOM-clustered approach)
        corrected_data = self._apply_som_clustered_combat_correction(marker_data, batches, data)
        
        # Combine with original data
        result = data.copy()
        
        # Add corrected markers (in capped ArcSinh space)
        for marker in self.markers:
            result[f"{marker}_corr"] = corrected_data[marker]
        
        print(f"ComBat correction completed for {len(self.markers)} markers (output is in capped ArcSinh space)")
        return result
    

    def _apply_som_clustered_combat_correction(self, marker_data, batches, full_df=None):
        """Apply SOM-clustered batch correction (matching R cyCombine approach)."""
        unique_batches = np.unique(batches)
        
        # Step 1: Normalize data individually for each batch (matching R cyCombine for clustering)
        normalized_data = marker_data.copy()
        if self.normalization_method == 'scale':
            if self.verbose:
                print("Applying per-batch Z-score standardization for SOM clustering...")
            for b in unique_batches:
                batch_mask = (batches == b)
                if batch_mask.any():
                    b_data = marker_data.loc[batch_mask]
                    b_mean = b_data.mean()
                    b_std = b_data.std()
                    b_std = b_std.replace(0, 1.0) # Avoid division by zero
                    normalized_data.loc[batch_mask] = (b_data - b_mean) / b_std
        elif self.normalization_method == 'rank':
            if self.verbose:
                print("Applying per-batch rank transformation for SOM clustering...")
            try:
                from scipy.stats import rankdata
                for b in unique_batches:
                    batch_mask = (batches == b)
                    if batch_mask.any():
                        b_data = marker_data.loc[batch_mask]
                        # Rank across cells (axis=0) for each marker, then scale to 0-1
                        ranked = b_data.apply(lambda x: rankdata(x) / len(x), axis=0)
                        normalized_data.loc[batch_mask] = ranked
            except ImportError:
                print("WARNING: scipy is required for rank normalization. Falling back to pandas rank.")
                for b in unique_batches:
                    batch_mask = (batches == b)
                    if batch_mask.any():
                        b_data = marker_data.loc[batch_mask]
                        ranked = b_data.rank(method='average', pct=True)
                        normalized_data.loc[batch_mask] = ranked
            
        # Step 2: Perform clustering
        n_rows, n_cols = self.som_grid_size
        n_clusters = n_rows * n_cols
        
        cluster_labels = np.zeros(len(marker_data), dtype=int)
        
        if MINISOM_AVAILABLE and MiniSom is not None:
            try:
                if self.verbose:
                    print(f"Training SOM grid of size {n_rows}x{n_cols} ({n_clusters} nodes) for {self.som_iterations} iterations...")
                normalized_values = normalized_data.values
                som = MiniSom(
                    n_rows, n_cols, len(self.markers),
                    sigma=1.0, learning_rate=0.5,
                    random_seed=self.random_seed
                )
                som.pca_weights_init(normalized_values)
                # train batch is fast and stable
                som.train(normalized_values, self.som_iterations, verbose=False)
                
                # Assign labels
                for idx, x in enumerate(normalized_values):
                    winner = som.winner(x)
                    cluster_labels[idx] = winner[0] * n_cols + winner[1]
            except Exception as e:
                print(f"SOM training failed: {e}. Falling back to KMeans clustering.")
                cluster_labels = self._run_kmeans_clustering(normalized_data, n_clusters)
        else:
            if self.verbose:
                print("MiniSom not available. Falling back to KMeans clustering.")
            cluster_labels = self._run_kmeans_clustering(normalized_data, n_clusters)
            
        # Store cluster labels on full_df for logging/plotting
        if full_df is not None:
            full_df['som_cluster'] = cluster_labels
            
        # Step 3: apply the correction.
        if self.correction_mode == 'monotonic':
            corrected = self._apply_monotonic_correction(marker_data, batches, cluster_labels, n_clusters)
        else:
            corrected = self._apply_per_cluster_combat(marker_data, batches, cluster_labels, n_clusters, full_df)

        # Step 4: Cap corrected values per marker to the min/max range of input values
        if self.verbose:
            print("Capping corrected values to input ranges...")
        n_capped = 0
        for col in corrected.columns:
            min_val = max(0.0, marker_data[col].min())
            max_val = marker_data[col].max()
            out_of_range = (corrected[col] < min_val) | (corrected[col] > max_val)
            n_capped += out_of_range.sum()
            corrected[col] = corrected[col].clip(lower=min_val, upper=max_val)
        self.run_summary['capping']['n_negative_clipped'] = int(n_capped)
        return corrected

    def _apply_monotonic_correction(self, marker_data, batches, cluster_labels, n_clusters):
        """Estimate the batch effect within matched populations, apply it monotonically.

        For each batch and each SOM node, the offset of that batch from the node's
        overall mean is the batch effect *for that cell population*. Averaging those
        per-node offsets across nodes -- weighted by how big each node is overall,
        never by how much of it this particular batch happens to contain -- yields a
        composition-controlled estimate: a batch made mostly of one cell type does not
        get that cell type's profile subtracted away.

        The averaged shift (and optional scale) is then applied once to every cell in
        the batch. Because that is a single affine map per (batch, marker), the order
        of cells within a batch is untouched: a cell that was the dimmest for a marker
        stays the dimmest. Any biological difference *within* a batch therefore
        survives the correction intact.
        """
        values = marker_data.to_numpy(dtype=float)
        corrected = values.copy()
        unique_batches = np.unique(batches)

        # Node sizes over the whole dataset -- the composition-neutral weights.
        node_sizes = np.bincount(cluster_labels, minlength=n_clusters).astype(float)
        node_index = [np.flatnonzero(cluster_labels == n) for n in range(n_clusters)]
        node_means = np.full((n_clusters, values.shape[1]), np.nan)
        node_stds = np.full((n_clusters, values.shape[1]), np.nan)
        for n, idx in enumerate(node_index):
            if idx.size:
                node_means[n] = values[idx].mean(axis=0)
                node_stds[n] = values[idx].std(axis=0)

        n_uncorrected = 0
        for b in unique_batches:
            batch_idx = np.flatnonzero(batches == b)
            batch_nodes = cluster_labels[batch_idx]

            shifts = np.zeros((n_clusters, values.shape[1]))
            scales = np.ones((n_clusters, values.shape[1]))
            weights = np.zeros(n_clusters)

            for n in range(n_clusters):
                cells = batch_idx[batch_nodes == n]
                # Both this batch's slice and the node overall must be big enough for
                # the offset to mean anything; thin nodes contribute nothing.
                if cells.size < self.min_cells_per_cluster or node_index[n].size < self.min_cells_per_cluster:
                    continue
                shifts[n] = values[cells].mean(axis=0) - node_means[n]
                batch_std = values[cells].std(axis=0)
                scales[n] = np.where(batch_std > 1e-6, node_stds[n] / np.maximum(batch_std, 1e-6), 1.0)
                weights[n] = node_sizes[n]

            if weights.sum() <= 0:
                # No node had enough of this batch to estimate anything: leave it alone
                # rather than apply a shift derived from noise.
                n_uncorrected += 1
                continue

            weights = weights / weights.sum()
            shift = (shifts * weights[:, None]).sum(axis=0)
            # Geometric mean for the scale factor, since scales are ratios.
            scale = np.exp((np.log(np.clip(scales, 1e-3, 1e3)) * weights[:, None]).sum(axis=0))

            batch_mean = values[batch_idx].mean(axis=0)
            corrected[batch_idx] = (values[batch_idx] - batch_mean) * scale + batch_mean - shift

        self.run_summary['combat']['n_clusters_total'] = n_clusters
        self.run_summary['combat']['n_batches_uncorrected'] = int(n_uncorrected)
        if self.verbose:
            print(f"Monotonic correction applied to {len(unique_batches) - n_uncorrected}/{len(unique_batches)} batches.")
        return pd.DataFrame(corrected, index=marker_data.index, columns=marker_data.columns)

    def _apply_per_cluster_combat(self, marker_data, batches, cluster_labels, n_clusters, full_df):
        """Legacy path: ComBat inside each SOM node, written back per node.

        Kept so earlier runs can be reproduced exactly. See ``correction_mode`` in
        ``__init__`` for why this is no longer the default.
        """
        # Skipped cells remain uncorrected (kept as input values)
        corrected = marker_data.copy()
        
        n_total_nodes = n_clusters
        n_skipped = 0
        n_covar_fallback = 0
        skip_details = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_cells_n = int(cluster_mask.sum())
            
            if cluster_cells_n == 0:
                n_skipped += 1
                continue
                
            cluster_batches = batches[cluster_mask]
            unique_cluster_batches = np.unique(cluster_batches)
            
            # Check cell count threshold
            if cluster_cells_n < self.min_cells_per_cluster:
                n_skipped += 1
                skip_details.append({
                    'cluster_id': cluster_id,
                    'reason': f'Too few cells ({cluster_cells_n} < {self.min_cells_per_cluster})',
                    'cells': cluster_cells_n
                })
                continue
                
            # Check batch diversity
            if len(unique_cluster_batches) < 2:
                n_skipped += 1
                skip_details.append({
                    'cluster_id': cluster_id,
                    'reason': f'Only 1 batch present ({unique_cluster_batches[0]})',
                    'cells': cluster_cells_n
                })
                continue
                
            # ComBat operates on the unscaled ArcSinh values (marker_data), not the per-batch Z-scored values
            cluster_data = marker_data.loc[cluster_mask].copy()
            
            try:
                batch_counts = pd.Series(cluster_batches).value_counts()
                has_single_sample_batch = (batch_counts == 1).any()
                
                # Build covariates if specified
                mod = []
                if self.covariates and full_df is not None:
                    for c in self.covariates:
                        if c in full_df.columns:
                            col_vals = full_df.loc[cluster_mask, c]
                            if col_vals.nunique() > 1 and not col_vals.isna().any():
                                if pd.api.types.is_numeric_dtype(col_vals):
                                    mod.append(col_vals.values.tolist())
                                else:
                                    codes, _ = pd.factorize(col_vals)
                                    mod.append(codes.tolist())
                                    
                if not COMBAT_AVAILABLE:
                    raise ImportError("pycombat not available")
                    
                # Setup kwargs
                kwargs = {
                    'par_prior': self.parametric,
                }
                if has_single_sample_batch:
                    kwargs['mean_only'] = True
                if mod:
                    kwargs['mod'] = mod
                    
                cluster_data_T = cluster_data.T
                corrected_cluster_T = pycombat(cluster_data_T, cluster_batches, **kwargs)
                corrected_cluster = corrected_cluster_T.T
                
                # Assign back
                corrected.loc[cluster_mask, :] = corrected_cluster.values
                
            except Exception as e:
                # If pycombat fails inside cluster, do z-score alignment in ArcSinh space as fallback
                cluster_corrected = self._fallback_zscore_alignment(cluster_data, cluster_batches)
                corrected.loc[cluster_mask, :] = cluster_corrected.values
                
        # Save cluster metrics in run summary
        self.run_summary['combat']['n_clusters_total'] = n_total_nodes
        self.run_summary['combat']['n_clusters_skipped'] = n_skipped
        self.run_summary['combat']['n_clusters_covar_fallback'] = n_covar_fallback
        self.run_summary['combat']['skip_details'] = skip_details

        return corrected
        
    def _fallback_zscore_alignment(self, data, batches):
        """Align distributions across batches using z-scores as fallback."""
        corrected = data.copy()
        unique_batches = np.unique(batches)
        for col in data.columns:
            global_mean = data[col].mean()
            global_std = data[col].std() if data[col].std() > 0 else 1.0
            for batch in unique_batches:
                batch_mask = batches == batch
                batch_vals = data.loc[batch_mask, col]
                if len(batch_vals) > 1:
                    b_mean = batch_vals.mean()
                    b_std = batch_vals.std()
                    if b_std > 0:
                        corrected.loc[batch_mask, col] = ((batch_vals - b_mean) / b_std) * global_std + global_mean
        return corrected
        
    def _run_kmeans_clustering(self, data, n_clusters):
        """KMeans fallback when MiniSom is not available."""
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_seed, batch_size=2048)
        return kmeans.fit_predict(data)


def extract_batch_id_from_filename(filename, strategy, filename_suffix=None, pattern=None):
    """Extract batch_id from filename based on strategy."""
    import re
    
    if strategy == 'filename_full':
        return filename[:-4] if filename.endswith('.csv') else filename
    elif strategy == 'filename_prefix':
        if filename.endswith('.csv'):
            filename = filename[:-4]
        if filename_suffix:
            if filename.endswith(filename_suffix.replace('.csv', '')):
                return filename.replace(filename_suffix.replace('.csv', ''), '')
        for suffix in ['.ome_cell_fully_annotated', '.ome_cell_features_annotated', '_fully_annotated']:
            if suffix in filename:
                return filename.split(suffix)[0]
        return filename
    elif strategy == 'filename_custom' and pattern:
        match = re.search(pattern, filename)
        return match.group(1) if match else filename
    else:
        return filename

def assign_universal_cell_ids(data_frames, cell_id_column='cell_id'):
    """Assign universal unique cell IDs to all cells."""
    cell_id_counter = 1
    modified_frames = []
    for df in data_frames:
        df_copy = df.copy()
        df_copy[cell_id_column] = range(cell_id_counter, cell_id_counter + len(df_copy))
        cell_id_counter += len(df_copy)
        modified_frames.append(df_copy)
    return modified_frames, cell_id_counter - 1


def preserve_filtered_cells_in_output(combined_data, corrected_data, markers, cell_id_column='cell_id'):
    """Preserve all cells in output with NA values for filtered cells."""
    complete_data = combined_data.copy()
    corrected_marker_cols = [f"{marker}_corr" for marker in markers]
    for col in corrected_marker_cols:
        complete_data[col] = np.nan
    if cell_id_column in corrected_data.columns:
        for col in corrected_marker_cols:
            if col in corrected_data.columns:
                correction_map = corrected_data.set_index(cell_id_column)[col]
                complete_data.loc[complete_data[cell_id_column].isin(correction_map.index), col] = \
                    complete_data.loc[complete_data[cell_id_column].isin(correction_map.index), cell_id_column].map(correction_map)
    return complete_data


def create_batch_correction_plots(data, markers, output_dir, dpi=300, cofactor=5.0):
    """Create comprehensive quality control plots for batch correction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-coerce markers and corrected markers to numeric in a copy to prevent TypeErrors
    data = data.copy()
    for m in markers:
        if m in data.columns:
            data[m] = pd.to_numeric(data[m], errors='coerce').fillna(0)
        corr_m = f"{m}_corr"
        if corr_m in data.columns:
            data[corr_m] = pd.to_numeric(data[corr_m], errors='coerce').fillna(0)
            
    # 1. Density plots for each marker
    print("Creating density plots...")
    _create_density_plots(data, markers, output_dir, dpi, cofactor)
    
    # 2. PCA plots
    print("Creating PCA plots...")
    _create_pca_plots(data, markers, output_dir, dpi, cofactor)
    
    # 3. UMAP plots
    print("Creating UMAP plots...")
    _create_umap_plots(data, markers, output_dir, dpi, cofactor)
    
    # 4. Batch integration metrics
    print("Creating batch integration plots...")
    _create_batch_integration_plots(data, markers, output_dir, dpi, cofactor)


def _create_density_plots(data, markers, output_dir, dpi, cofactor=5.0):
    """Create density plots comparing before/after correction."""
    n_markers = len(markers)
    n_cols = 3  # Reduced columns to give more space for legends
    n_rows = (n_markers + n_cols - 1) // n_cols
    
    # Increased figure width to accommodate legends on the right
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, marker in enumerate(markers):
        ax = axes[i]
        
        # Get data for this marker and add small constant to avoid log(0)
        marker_data = data[marker].copy()
        min_positive = marker_data[marker_data > 0].min() if (marker_data > 0).any() else 1
        log_offset = min_positive / 10  # Small offset to handle zeros
        marker_data = marker_data + log_offset
        
        # Plot original data with log10 bins
        log_bins = np.logspace(np.log10(marker_data.min()), np.log10(marker_data.max()), 50)
        
        for batch in data['batch_id'].unique():
            batch_data = data[data['batch_id'] == batch]
            batch_marker_data = batch_data[marker] + log_offset
            ax.hist(batch_marker_data, alpha=0.4, bins=log_bins, density=True, 
                   label=f"{batch} (original)", histtype='step', linewidth=2)
        
        # Plot corrected data (transform back to linear scale for comparison on original axes)
        corrected_col = f"{marker}_corr"
        if corrected_col in data.columns:
            corrected_linear = np.sinh(data[corrected_col].copy()) * cofactor
            corrected_data = corrected_linear + log_offset
            corrected_log_bins = np.logspace(np.log10(corrected_data.min()), np.log10(corrected_data.max()), 50)
            
            for batch in data['batch_id'].unique():
                batch_data = data[data['batch_id'] == batch]
                batch_corrected_linear = np.sinh(batch_data[corrected_col].copy()) * cofactor
                batch_corrected_data = batch_corrected_linear + log_offset
                ax.hist(batch_corrected_data, alpha=0.4, bins=corrected_log_bins, density=True,
                       label=f"{batch} (corrected)", histtype='step', linewidth=2, linestyle='--')
        
        # Set log scale and formatting
        ax.set_xscale('log')
        ax.set_title(f"{marker} - Before/After Correction", fontsize=12, fontweight='bold')
        ax.set_xlabel("Expression (log10 scale)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Place legend outside the plot area on the right
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True, 
                 fancybox=True, shadow=True)
    
    # Remove empty subplots
    for i in range(len(markers), len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legends
    plt.savefig(output_dir / "density_plots.png", dpi=dpi, bbox_inches='tight')
    plt.close()
 
 
def _create_pca_plots(data, markers, output_dir, dpi, cofactor=5.0):
    """Create PCA plots showing batch separation."""
    # Original data PCA (apply ArcSinh to match corrected space)
    original_data = data[markers].fillna(0)
    original_data_arcsinh = np.arcsinh(original_data / cofactor)
    pca = PCA(n_components=2)
    pca_original = pca.fit_transform(StandardScaler().fit_transform(original_data_arcsinh))
    
    # Corrected data PCA (already in ArcSinh space)
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]
    if corrected_markers:
        corrected_data = data[corrected_markers].fillna(0)
        pca_corrected = pca.fit_transform(StandardScaler().fit_transform(corrected_data))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original PCA
    scatter = axes[0].scatter(pca_original[:, 0], pca_original[:, 1], 
                             c=pd.Categorical(data['batch_id']).codes, 
                             alpha=0.6, s=1)
    axes[0].set_title("PCA - Original Data (ArcSinh space)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    
    # Corrected PCA
    if corrected_markers:
        scatter = axes[1].scatter(pca_corrected[:, 0], pca_corrected[:, 1], 
                                 c=pd.Categorical(data['batch_id']).codes, 
                                 alpha=0.6, s=1)
        axes[1].set_title("PCA - Corrected Data")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pca_plots.png", dpi=dpi, bbox_inches='tight')
    plt.close()
 
 
def _create_umap_plots(data, markers, output_dir, dpi, cofactor=5.0):
    """Create UMAP plots showing batch integration."""
    try:
        import umap
    except ImportError:
        print("UMAP not available, skipping UMAP plots...")
        return
 
    # Subsample for UMAP if needed
    if len(data) > 50000:
        plot_data = data.sample(n=50000, random_state=42)
    else:
        plot_data = data
    
    # Original data UMAP (apply ArcSinh)
    original_data = plot_data[markers].fillna(0)
    original_data_arcsinh = np.arcsinh(original_data / cofactor)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_original = reducer.fit_transform(StandardScaler().fit_transform(original_data_arcsinh))
    
    # Corrected data UMAP (already in ArcSinh space)
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in plot_data.columns]
    if corrected_markers:
        corrected_data = plot_data[corrected_markers].fillna(0)
        umap_corrected = reducer.fit_transform(StandardScaler().fit_transform(corrected_data))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original UMAP
    scatter = axes[0].scatter(umap_original[:, 0], umap_original[:, 1], 
                             c=pd.Categorical(plot_data['batch_id']).codes, 
                             alpha=0.6, s=1)
    axes[0].set_title("UMAP - Original Data (ArcSinh space)")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    
    # Corrected UMAP
    if corrected_markers:
        scatter = axes[1].scatter(umap_corrected[:, 0], umap_corrected[:, 1], 
                                 c=pd.Categorical(plot_data['batch_id']).codes, 
                                 alpha=0.6, s=1)
        axes[1].set_title("UMAP - Corrected Data")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
    
    plt.tight_layout()
    plt.savefig(output_dir / "umap_plots.png", dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Save UMAP data as CSV files
    print("Saving UMAP data as CSV files...")
    
    # Prepare columns to save (cell_id, metadata, markers)
    base_columns = ['cell_id', 'batch_id', 'file_prefix']
    marker_columns = markers + [f"{m}_corr" for m in markers if f"{m}_corr" in plot_data.columns]
    save_columns = base_columns + marker_columns
    
    # Ensure all columns exist in plot_data
    available_columns = [col for col in save_columns if col in plot_data.columns]
    
    # Save original UMAP data
    umap_original_df = plot_data[available_columns].copy()
    umap_original_df['UMAP1'] = umap_original[:, 0]
    umap_original_df['UMAP2'] = umap_original[:, 1]
    umap_original_df['data_type'] = 'original'
    
    original_csv_path = output_dir / "umap_original_data.csv"
    umap_original_df.to_csv(original_csv_path, index=False)
    print(f"  Saved original UMAP data: {original_csv_path}")
    
    # Save corrected UMAP data
    if corrected_markers:
        umap_corrected_df = plot_data[available_columns].copy()
        umap_corrected_df['UMAP1'] = umap_corrected[:, 0]
        umap_corrected_df['UMAP2'] = umap_corrected[:, 1]
        umap_corrected_df['data_type'] = 'corrected'
        
        corrected_csv_path = output_dir / "umap_corrected_data.csv"
        umap_corrected_df.to_csv(corrected_csv_path, index=False)
        print(f"  Saved corrected UMAP data: {corrected_csv_path}")


def _create_batch_integration_plots(data, markers, output_dir, dpi, cofactor=5.0):
    """Create plots showing batch integration metrics."""
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]
    if not corrected_markers:
        return
    
    # Calculate batch integration metrics (using ArcSinh original vs corrected)
    original_data = data[markers].fillna(0)
    original_data_arcsinh = np.arcsinh(original_data / cofactor)
    corrected_data = data[corrected_markers].fillna(0)
    
    # Silhouette scores
    batches_encoded = pd.Categorical(data['batch_id']).codes
    
    try:
        # Original silhouette score
        original_silhouette = silhouette_score(
            StandardScaler().fit_transform(original_data), 
            batches_encoded
        )
        
        # Corrected silhouette score
        corrected_silhouette = silhouette_score(
            StandardScaler().fit_transform(corrected_data), 
            batches_encoded
        )
        
        # Plot silhouette scores
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scores = [original_silhouette, corrected_silhouette]
        labels = ['Original', 'Corrected']
        
        bars = ax.bar(labels, scores)
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Batch Integration: Lower is Better')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / "batch_integration_metrics.png", dpi=dpi, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not calculate batch integration metrics: {e}")


def discover_data_and_markers(input_dir, exclude_patterns=None):
    """
    Discover available CSV files and extract potential marker columns.
    
    Parameters:
    -----------
    input_dir : Path
        Directory containing CSV files
    exclude_patterns : list, optional
        List of patterns to exclude from marker detection.
        If None, uses minimal default exclusions ['cell_id', 'batch_id'].
        Use [] for no exclusions at all.
        
    Returns:
    --------
    dict
        Dictionary containing discovered information
    """
    # Find all CSV files
    csv_files = list(input_dir.glob("*_fully_annotated.csv"))
    if not csv_files:
        csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        return {
            'csv_files': [],
            'potential_markers': [],
            'annotation_columns': [],
            'spatial_columns': [],
            'file_info': []
        }
    
    print(f"Found {len(csv_files)} CSV files:")
    
    # Display files with their potential batch identifiers
    file_info = []
    for file in csv_files:
        filename = file.name
        # Extract potential batch ID from filename
        filename_no_ext = filename[:-4] if filename.endswith('.csv') else filename
        
        # Try different extraction strategies
        if '.ome_cell_features_annotated' in filename_no_ext:
            batch_id = filename_no_ext.split('.ome_cell_features_annotated')[0]
        elif '_fully_annotated' in filename_no_ext:
            batch_id = filename_no_ext.split('_fully_annotated')[0]
        else:
            batch_id = filename_no_ext
            
        file_info.append({'file': filename, 'suggested_batch_id': batch_id})
        print(f"  {filename} -> suggested batch_id: '{batch_id}'")
    
    # Load first file to discover available columns
    print(f"\nAnalyzing columns from first file: {csv_files[0].name}")
    sample_df = pd.read_csv(csv_files[0], nrows=1000, low_memory=False)
    
    all_columns = list(sample_df.columns)
    print(f"Total columns found: {len(all_columns)}")
    
    # Identify potential marker columns - configurable exclusions
    if exclude_patterns is None:
        # Default minimal exclusions - only ID columns
        exclude_patterns = ['cell_id', 'batch_id']
    # If exclude_patterns is an empty list [], no exclusions will be applied
    
    potential_markers = []
    
    for col in all_columns:
        if sample_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            if not any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                potential_markers.append(col)
    
    print(f"\nPotential marker columns ({len(potential_markers)}):")
    for i, marker in enumerate(potential_markers):
        print(f"  {i+1:2d}. {marker}")
    
    # Check for common annotation columns
    annotation_columns = [col for col in all_columns if any(term in col.lower() for term in ['tissue', 'annotation', 'region', 'class'])]
    print(f"\nAnnotation/filtering columns found ({len(annotation_columns)}):")
    for col in annotation_columns:
        unique_vals = sample_df[col].unique()[:10]
        print(f"  {col}: {unique_vals}")
    
    # Check for spatial columns - very specific patterns to avoid catching marker proteins
    spatial_columns = [col for col in all_columns if any(term == col.lower() or term in col.lower() for term in ['x_centroid', 'y_centroid', 'centroid_x', 'centroid_y', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'solidity', 'extent', 'orientation', 'perimeter', 'area'])]
    print(f"\nSpatial/morphology columns found ({len(spatial_columns)}):")
    for col in spatial_columns:
        print(f"  {col}")
    
    return {
        'csv_files': csv_files,
        'potential_markers': potential_markers,
        'annotation_columns': annotation_columns,
        'spatial_columns': spatial_columns,
        'file_info': file_info
    }


def perform_quality_control_analysis(data, markers, qc_params, qc_dir):
    """
    Perform comprehensive quality control analysis on the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    markers : list
        List of marker columns
    qc_params : dict
        QC parameters
    qc_dir : Path
        Directory to save QC outputs
        
    Returns:
    --------
    pandas.DataFrame, dict
        QC-filtered data and filtering statistics
    """
    print("Performing quality control analysis...")
    
    # Initialize quality column
    data = data.copy()
    data['quality'] = 1
    
    # Track filtering statistics
    initial_cell_count = len(data)
    filtering_stats = {'initial_cells': initial_cell_count}
    
    # Create working copy for QC analysis
    qc_data = data[markers].copy()
    
    # Z-score normalization for outlier detection
    print("  Computing z-scores for outlier detection...")
    qc_data_zscore = qc_data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # Generate QC histograms
    if qc_params.get('generate_histograms', True):
        _generate_qc_histograms(qc_data_zscore, markers, qc_params, qc_dir)
    
    # Outlier detection
    print("  Detecting outlier cells...")
    outlier_mask, filtering_stats = _detect_outliers(
        qc_data_zscore, markers, qc_params, filtering_stats
    )
    
    # Apply area filtering if enabled
    if qc_params.get('enable_area_filter', False):
        area_outlier_mask = _apply_area_filtering(data, qc_params, filtering_stats)
        outlier_mask |= area_outlier_mask
    
    # Update quality column
    data.loc[outlier_mask, 'quality'] = 0
    
    # Generate QC summary plots
    if qc_params.get('generate_plots', True):
        _generate_qc_summary_plots(data, qc_data_zscore, markers, qc_params, qc_dir, filtering_stats)
    
    # Calculate final statistics
    final_cell_count = (data['quality'] == 1).sum()
    filtering_stats['final_cells'] = final_cell_count
    filtering_stats['total_filtered'] = initial_cell_count - final_cell_count
    filtering_stats['pass_rate'] = final_cell_count / initial_cell_count
    
    # Print and save QC summary
    _print_and_save_qc_summary(filtering_stats, qc_dir)
    
    return data, filtering_stats


def _generate_qc_histograms(qc_data_zscore, markers, qc_params, qc_dir):
    """Generate QC histograms for markers."""
    print("  Generating QC histograms...")
    histogram_subset = qc_params.get('histogram_subset', 10)
    markers_to_plot = markers[:histogram_subset] if histogram_subset else markers
    
    # Create subplots for histograms
    n_markers = len(markers_to_plot)
    n_cols = min(4, n_markers)
    n_rows = (n_markers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_markers == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, marker in enumerate(markers_to_plot):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[0, col] if n_cols > 1 else axes[0]
        
        # Plot histogram of z-scores
        ax.hist(qc_data_zscore[marker].dropna(), bins=50, alpha=0.7, density=True)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Mean')
        ax.axvline(x=2, color='orange', linestyle='--', alpha=0.5, label='2 SD')
        ax.axvline(x=-2, color='orange', linestyle='--', alpha=0.5)
        ax.set_title(f'{marker} (Z-score)')
        ax.set_xlabel('Z-score')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    # Remove extra subplots
    for i in range(n_markers, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            fig.delaxes(axes[row, col])
        else:
            fig.delaxes(axes[0, col] if n_cols > 1 else axes[0])
    
    plt.tight_layout()
    plot_dpi = qc_params.get('plot_dpi', 300)
    plt.savefig(qc_dir / "marker_zscore_histograms.png", dpi=plot_dpi, bbox_inches='tight')
    plt.close()
    print(f"    Saved histograms to: {qc_dir / 'marker_zscore_histograms.png'}")


def _detect_outliers(qc_data_zscore, markers, qc_params, filtering_stats):
    """Detect outlier cells based on z-score thresholds."""
    # Calculate percentile thresholds
    percentile_high = qc_params.get('percentile_high', 99.0)
    percentile_low = qc_params.get('percentile_low', 1.0)
    max_outlier_fraction = qc_params.get('max_outlier_fraction', 0.5)
    
    high_thresholds = qc_data_zscore.quantile(percentile_high / 100)
    low_thresholds = qc_data_zscore.quantile(percentile_low / 100)
    
    # Count outliers per cell
    high_outliers = (qc_data_zscore > high_thresholds).sum(axis=1)
    low_outliers = (qc_data_zscore < low_thresholds).sum(axis=1)
    
    # Calculate outlier fractions
    n_markers = len(markers)
    high_outlier_fraction = high_outliers / n_markers
    low_outlier_fraction = low_outliers / n_markers
    
    # Apply outlier filtering
    outlier_mask = pd.Series(False, index=qc_data_zscore.index)
    
    if qc_params.get('filter_high_outliers', True):
        high_outlier_mask = high_outlier_fraction > max_outlier_fraction
        outlier_mask |= high_outlier_mask
        filtering_stats['high_outliers'] = high_outlier_mask.sum()
        print(f"    High outlier cells (>{max_outlier_fraction*100}% markers > {percentile_high}th percentile): {high_outlier_mask.sum():,}")
    
    if qc_params.get('filter_low_outliers', True):
        low_outlier_mask = low_outlier_fraction > max_outlier_fraction
        outlier_mask |= low_outlier_mask
        filtering_stats['low_outliers'] = low_outlier_mask.sum()
        print(f"    Low outlier cells (>{max_outlier_fraction*100}% markers < {percentile_low}th percentile): {low_outlier_mask.sum():,}")
    
    return outlier_mask, filtering_stats


def _apply_area_filtering(data, qc_params, filtering_stats):
    """Apply area-based filtering if enabled."""
    area_column = qc_params.get('area_column', 'area')
    
    if area_column not in data.columns:
        return pd.Series(False, index=data.index)
    
    print("  Applying area-based filtering...")
    area_percentile_low = qc_params.get('area_percentile_low', 1.0)
    area_percentile_high = qc_params.get('area_percentile_high', 99.0)
    
    area_low = data[area_column].quantile(area_percentile_low / 100)
    area_high = data[area_column].quantile(area_percentile_high / 100)
    
    area_outlier_mask = (data[area_column] < area_low) | (data[area_column] > area_high)
    filtering_stats['area_outliers'] = area_outlier_mask.sum()
    print(f"    Area outlier cells (< {area_low:.1f} or > {area_high:.1f}): {area_outlier_mask.sum():,}")
    
    return area_outlier_mask


def _generate_qc_summary_plots(data, qc_data_zscore, markers, qc_params, qc_dir, filtering_stats):
    """Generate comprehensive QC summary plots."""
    print("  Generating QC summary plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Outlier fraction distribution
    max_outlier_fraction = qc_params.get('max_outlier_fraction', 0.5)
    high_outlier_fraction = (qc_data_zscore > qc_data_zscore.quantile(0.99)).sum(axis=1) / len(markers)
    low_outlier_fraction = (qc_data_zscore < qc_data_zscore.quantile(0.01)).sum(axis=1) / len(markers)
    
    axes[0, 0].hist(high_outlier_fraction, bins=50, alpha=0.7, label='High outliers', color='red')
    axes[0, 0].hist(low_outlier_fraction, bins=50, alpha=0.7, label='Low outliers', color='blue')
    axes[0, 0].axvline(x=max_outlier_fraction, color='black', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Fraction of markers as outliers')
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title('Outlier Fraction Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Quality distribution by batch
    quality_by_batch = data.groupby('batch_id')['quality'].agg(['count', 'sum']).reset_index()
    quality_by_batch['pass_rate'] = quality_by_batch['sum'] / quality_by_batch['count']
    
    x_pos = range(len(quality_by_batch))
    axes[0, 1].bar(x_pos, quality_by_batch['pass_rate'], alpha=0.7)
    axes[0, 1].set_xlabel('Batch')
    axes[0, 1].set_ylabel('Quality pass rate')
    axes[0, 1].set_title('Quality Pass Rate by Batch')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(quality_by_batch['batch_id'], rotation=45)
    
    # Plot 3: Area distribution (if available)
    area_column = qc_params.get('area_column', 'area')
    if area_column in data.columns:
        axes[1, 0].hist(data[area_column], bins=50, alpha=0.7)
        if qc_params.get('enable_area_filter', False):
            area_low = data[area_column].quantile(qc_params.get('area_percentile_low', 1.0) / 100)
            area_high = data[area_column].quantile(qc_params.get('area_percentile_high', 99.0) / 100)
            axes[1, 0].axvline(x=area_low, color='red', linestyle='--', label=f'{qc_params.get("area_percentile_low", 1.0)}th percentile')
            axes[1, 0].axvline(x=area_high, color='red', linestyle='--', label=f'{qc_params.get("area_percentile_high", 99.0)}th percentile')
            axes[1, 0].legend()
        axes[1, 0].set_xlabel(f'{area_column}')
        axes[1, 0].set_ylabel('Cell count')
        axes[1, 0].set_title('Cell Area Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'Area column not available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Cell Area Distribution - N/A')
    
    # Plot 4: Quality summary statistics
    total_cells = len(data)
    pass_cells = (data['quality'] == 1).sum()
    fail_cells = total_cells - pass_cells
    
    axes[1, 1].pie([pass_cells, fail_cells], labels=['Pass QC', 'Fail QC'], 
                   colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title(f'Quality Control Summary\n({pass_cells:,} / {total_cells:,} cells pass)')
    
    plt.tight_layout()
    plot_dpi = qc_params.get('plot_dpi', 300)
    plt.savefig(qc_dir / "quality_control_summary.png", dpi=plot_dpi, bbox_inches='tight')
    plt.close()
    print(f"    Saved QC summary to: {qc_dir / 'quality_control_summary.png'}")


def _print_and_save_qc_summary(filtering_stats, qc_dir):
    """Print and save QC summary statistics."""
    print(f"\nQuality Control Summary:")
    print(f"  Initial cells: {filtering_stats['initial_cells']:,}")
    print(f"  Final cells (quality=1): {filtering_stats['final_cells']:,}")
    print(f"  Filtered cells: {filtering_stats['total_filtered']:,}")
    print(f"  Pass rate: {filtering_stats['pass_rate']:.1%}")
    
    # Save QC statistics
    qc_stats_df = pd.DataFrame([filtering_stats])
    qc_stats_df.to_csv(qc_dir / "quality_control_stats.csv", index=False)
    print(f"  QC statistics saved to: {qc_dir / 'quality_control_stats.csv'}")


def load_and_combine_data(csv_files, metadata_df, use_metadata, batch_id_strategy, 
                         filename_suffix, batch_id_pattern, assign_universal_cell_id, 
                         cell_id_column):
    """
    Load and combine all CSV files with proper batch assignment and cell IDs.
    
    Parameters:
    -----------
    csv_files : list
        List of CSV file paths
    metadata_df : pandas.DataFrame or None
        Metadata dataframe
    use_metadata : bool
        Whether to use metadata for batch assignment
    batch_id_strategy : str
        Strategy for batch ID extraction
    filename_suffix : str
        Suffix for filename-based extraction
    batch_id_pattern : str
        Pattern for custom extraction
    assign_universal_cell_id : bool
        Whether to assign universal cell IDs
    cell_id_column : str
        Name of cell ID column
        
    Returns:
    --------
    pandas.DataFrame, int
        Combined data and total cell count
    """
    print(f"Loading {len(csv_files)} CSV files...")
    
    all_data_frames = []
    
    for i, file in enumerate(csv_files):
        print(f"  Loading {file.name}...")
        try:
            df = pd.read_csv(file, low_memory=False)
            
            # Extract batch_id based on strategy
            if use_metadata and batch_id_strategy == 'metadata':
                batch_id = _extract_batch_id_from_metadata(file, metadata_df)
            else:
                batch_id = extract_batch_id_from_filename(file.name, batch_id_strategy, filename_suffix, batch_id_pattern)
            
            print(f"    Assigned batch_id: '{batch_id}'")
            
            # Add batch information
            df['file_prefix'] = batch_id
            df['batch_id'] = batch_id
            df['batch'] = file.name
            
            all_data_frames.append(df)
            
            if i == 0:
                print(f"    First file shape: {df.shape}")
                print(f"    Columns: {len(df.columns)}")
                
        except Exception as e:
            print(f"    Error loading {file.name}: {e}")
            continue
    
    if not all_data_frames:
        raise ValueError("No data files could be loaded")
    
    # Assign universal cell IDs to ALL cells
    if assign_universal_cell_id:
        print("Assigning universal cell IDs to all cells...")
        all_data_frames, total_cells = assign_universal_cell_ids(all_data_frames, cell_id_column)
        print(f"  Assigned unique cell IDs from 1 to {total_cells}")
    
    # Combine all data
    print("Combining data...")
    combined_data = pd.concat(all_data_frames, ignore_index=True)
    
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Unique batches: {combined_data['batch_id'].nunique()}")
    print(f"Batch distribution:")
    for batch_id, count in combined_data['batch_id'].value_counts().items():
        print(f"  {batch_id}: {count:,} cells")
    
    if assign_universal_cell_id:
        print(f"Universal cell IDs: {combined_data[cell_id_column].min()} to {combined_data[cell_id_column].max()}")
        # Verify uniqueness
        if combined_data[cell_id_column].nunique() != len(combined_data):
            raise ValueError("Cell IDs are not unique!")
        else:
            print("  Cell ID uniqueness verified")
    
    return combined_data


def _extract_batch_id_from_metadata(file, metadata_df):
    """Extract batch_id from metadata file."""
    filename = file.name
    if filename.endswith('.csv'):
        filename = filename[:-4]
    if '.ome_cell_features_annotated' in filename:
        prefix = filename.split('.ome_cell_features_annotated')[0]
    else:
        prefix = filename
    
    print(f"    Extracted prefix: '{prefix}'")
    
    # Find matching metadata row
    metadata_row = metadata_df[metadata_df['file_prefix'] == prefix]
    if metadata_row.empty:
        print(f"    Warning: No metadata found for prefix '{prefix}'. Using filename-based batch_id.")
        return extract_batch_id_from_filename(file.name, 'filename_prefix', '.ome_cell_fully_annotated.csv')
    else:
        if len(metadata_row) > 1:
            print(f"    Warning: Multiple metadata rows for prefix '{prefix}'. Using first.")
            metadata_row = metadata_row.iloc[0]
        else:
            metadata_row = metadata_row.iloc[0]
        
        batch_id = metadata_row['batch_id']
        print(f"    Using metadata batch_id: '{batch_id}'")
        return batch_id


def apply_marker_selection_strategy(potential_markers, strategy, manual_markers=None, exclude_markers=None):
    """
    Apply marker selection strategy to determine final markers for correction.
    
    Parameters:
    -----------
    potential_markers : list
        List of discovered potential markers
    strategy : str
        Selection strategy ('auto_all', 'auto_filtered', 'manual')
    manual_markers : list, optional
        Manual marker list for 'manual' strategy
    exclude_markers : list, optional
        Markers to exclude for 'auto_filtered' strategy
        
    Returns:
    --------
    list
        Final list of markers for correction
    """
    if strategy == 'auto_all':
        final_markers = potential_markers.copy()
        print(f"Using all detected markers ({len(final_markers)})")
        
    elif strategy == 'auto_filtered':
        exclude_markers = exclude_markers or []
        final_markers = [m for m in potential_markers if m not in exclude_markers]
        print(f"Using filtered detected markers ({len(final_markers)} of {len(potential_markers)})")
        excluded = [m for m in potential_markers if m in exclude_markers]
        if excluded:
            print(f"  Excluded: {excluded}")
            
    elif strategy == 'manual':
        if manual_markers is None:
            raise ValueError("manual_markers must be provided for manual strategy")
        
        # Check which manual markers are available
        available_manual = [m for m in manual_markers if m in potential_markers]
        missing_manual = [m for m in manual_markers if m not in potential_markers]
        
        final_markers = available_manual
        print(f"Using manual marker selection ({len(available_manual)} of {len(manual_markers)} specified)")
        
        if missing_manual:
            print(f"  Missing markers: {missing_manual}")
            
    else:
        raise ValueError(f"Unknown marker selection strategy: {strategy}")
    
    return final_markers


def select_and_validate_markers(marker_selection_strategy, manual_markers, potential_markers=None):
    """
    Select and validate markers for batch correction.
    
    Parameters:
    -----------
    marker_selection_strategy : str
        'manual' or 'auto_all'
    manual_markers : list
        List of manually specified markers
    potential_markers : list, optional
        List of markers detected from data discovery
        
    Returns:
    --------
    list
        Final list of validated markers for batch correction
    """
    try:
        if marker_selection_strategy == 'manual':
            if potential_markers is not None:
                # Check which manual markers are available
                available_manual = [m for m in manual_markers if m in potential_markers]
                missing_manual = [m for m in manual_markers if m not in potential_markers]
                final_markers = available_manual
                print(f"Using manual marker selection ({len(available_manual)} of {len(manual_markers)} specified)")
                if missing_manual:
                    print(f"  Missing markers: {missing_manual}")
            else:
                # No potential markers available - use all manual markers
                print("Data discovery not run - using all manual markers")
                final_markers = manual_markers.copy()
        else:  # auto_all
            if potential_markers is not None:
                final_markers = potential_markers.copy()
                print(f"Using all detected markers ({len(final_markers)})")
            else:
                # Fallback to manual if no potential markers
                print("No potential markers available - falling back to manual selection")
                final_markers = manual_markers.copy()
            
    except Exception as e:
        print(f"Error in marker selection: {e}")
        print("Falling back to manual marker selection...")
        final_markers = manual_markers.copy()
    
    print(f"\nFinal marker list for batch correction ({len(final_markers)}):")
    for i, marker in enumerate(final_markers):
        print(f"  {i+1:2d}. {marker}")
    
    return final_markers


def apply_data_filters(data, filter_config):
    """
    Apply various filtering criteria to the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    filter_config : dict
        Configuration for filters
        
    Returns:
    --------
    pandas.DataFrame
        Filtered data
    """
    filtered_data = data.copy()
    initial_count = len(filtered_data)
    
    # Apply quality filtering
    if filter_config.get('enable_qc', False):
        quality_count = (filtered_data['quality'] == 1).sum()
        filtered_data = filtered_data[filtered_data['quality'] == 1].copy()
        print(f"  Quality filter: {len(filtered_data):,} of {initial_count:,} cells retained")

    # Apply tissue filtering
    tissue_filter = filter_config.get('tissue_filter', False)
    tissue_column = filter_config.get('tissue_column', 'tissue')
    tissue_value = filter_config.get('tissue_value', 1)
    
    if tissue_filter and tissue_column in filtered_data.columns:
        tissue_before = len(filtered_data)
        filtered_data = filtered_data[filtered_data[tissue_column] == tissue_value].copy()
        print(f"  Tissue filter ({tissue_column}=={tissue_value}): {len(filtered_data):,} of {tissue_before:,} cells retained")

    # Apply additional custom filters
    additional_filters = filter_config.get('additional_filters', {})
    for filter_col, filter_values in additional_filters.items():
        if filter_col in filtered_data.columns:
            before_count = len(filtered_data)
            if isinstance(filter_values, list):
                filtered_data = filtered_data[filtered_data[filter_col].isin(filter_values)].copy()
            else:
                filtered_data = filtered_data[filtered_data[filter_col] == filter_values].copy()
            print(f"  {filter_col} filter: {len(filtered_data):,} of {before_count:,} cells retained")
        else:
            print(f"  Warning: Filter column '{filter_col}' not found in data")

    # Final data summary
    print(f"\nFiltered dataset summary:")
    print(f"  Total cells for batch correction: {len(filtered_data):,}")
    print(f"  Cells filtered out: {initial_count - len(filtered_data):,}")
    print(f"  Retention rate: {len(filtered_data)/initial_count:.1%}")
    print(f"  Unique batches: {filtered_data['batch_id'].nunique()}")
    print(f"  Batch distribution:")
    for batch_id, count in filtered_data['batch_id'].value_counts().items():
        print(f"    {batch_id}: {count:,} cells")

    # Check for batches with insufficient cells
    min_cells_per_batch = filter_config.get('min_cells_per_batch', 100)
    small_batches = filtered_data['batch_id'].value_counts()
    small_batches = small_batches[small_batches < min_cells_per_batch]

    if len(small_batches) > 0:
        print(f"\nWarning: Small batches detected (< {min_cells_per_batch} cells):")
        for batch_id, count in small_batches.items():
            print(f"  {batch_id}: {count} cells")
        print("Consider combining small batches or adjusting filtering criteria.")
    
    return filtered_data


def validate_markers_availability(markers, data_columns):
    """
    Validate that markers are available in the data.
    
    Parameters:
    -----------
    markers : list
        List of marker names
    data_columns : list
        Available columns in the data
        
    Returns:
    --------
    list
        Available markers
    """
    available_markers = [m for m in markers if m in data_columns]
    
    print(f"Available markers for correction: {len(available_markers)}")
    if len(available_markers) != len(markers):
        missing = set(markers) - set(available_markers)
        print(f"  Missing markers: {list(missing)}")

    if not available_markers:
        raise ValueError("No markers available for correction")
    
    return available_markers


def save_batch_corrected_files(combined_data_corrected, output_dir, markers, cell_id_column='cell_id'):
    """
    Save batch-corrected files (combined and individual).
    
    Parameters:
    -----------
    combined_data_corrected : pandas.DataFrame
        Complete corrected dataset
    output_dir : Path
        Output directory
    markers : list
        List of marker names
    cell_id_column : str
        Name of cell ID column
        
    Returns:
    --------
    None
    """
    # Save combined corrected data (with ALL cells)
    output_file = output_dir / "all_samples_combined_corrected.csv"
    print(f"Saving combined corrected data to: {output_file}")
    combined_data_corrected.to_csv(output_file, index=False)
    
    # Save individual corrected files
    print("Saving individual corrected files...")
    for file_prefix_val, batch_df in combined_data_corrected.groupby('file_prefix'):
        output_filename = output_dir / f"{file_prefix_val}_corrected.csv"
        batch_df_copy = batch_df.copy()
        
        # Remove batch processing columns from individual files
        cols_to_remove = ['source_file', 'batch', 'som_cluster']
        batch_df_copy.drop(columns=cols_to_remove, inplace=True, errors='ignore')
        
        batch_df_copy.to_csv(output_filename, index=False)
        
        # Count cells with and without correction for this batch
        corrected_count = batch_df_copy[[f"{m}_corr" for m in markers if f"{m}_corr" in batch_df_copy.columns]].notna().all(axis=1).sum()
        total_count = len(batch_df_copy)
        
        print(f"  Saved: {output_filename}")
        print(f"    Total cells: {total_count:,} | Corrected: {corrected_count:,} | Filtered: {total_count - corrected_count:,}")


def verify_cell_id_integrity(original_data, final_data, cell_id_column='cell_id'):
    """
    Verify that all original cell IDs are preserved in the final data.
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original data
    final_data : pandas.DataFrame
        Final processed data
    cell_id_column : str
        Name of cell ID column
        
    Returns:
    --------
    bool
        True if integrity is verified
    """
    original_cell_ids = set(original_data[cell_id_column])
    final_cell_ids = set(final_data[cell_id_column])
    
    if original_cell_ids == final_cell_ids:
        print("Cell ID integrity verified - all original cells preserved")
        return True
    else:
        missing_ids = original_cell_ids - final_cell_ids
        extra_ids = final_cell_ids - original_cell_ids
        if missing_ids:
            print(f"Warning: Missing cell IDs: {len(missing_ids)}")
        if extra_ids:
            print(f"Warning: Extra cell IDs: {len(extra_ids)}")
        return False


def analyze_correction_magnitude(data, markers, batch_col='batch_id'):
    """
    Analyze batch correction magnitude for each marker and batch.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with original and corrected markers
    markers : list
        List of marker names
    batch_col : str
        Batch column name
        
    Returns:
    --------
    pandas.DataFrame
        Analysis results with correction magnitudes
    """
    print("Analyzing batch correction magnitude...")
    
    results = []
    
    for marker in markers:
        original_col = marker
        corrected_col = f"{marker}_corr"
        
        if corrected_col not in data.columns:
            continue
            
        # Calculate correction magnitude per batch
        for batch in data[batch_col].unique():
            batch_data = data[data[batch_col] == batch]
            
            if len(batch_data) < 2:
                continue
                
            original_vals = batch_data[original_col].dropna()
            corrected_vals = batch_data[corrected_col].dropna()
            
            if len(original_vals) == 0 or len(corrected_vals) == 0:
                continue
            
            # Calculate various correction metrics
            mean_original = original_vals.mean()
            mean_corrected = corrected_vals.mean()
            
            # Correction magnitude (fold change)
            correction_magnitude = abs(mean_corrected - mean_original) / (abs(mean_original) + 1e-6)
            
            # Variance change
            var_original = original_vals.var()
            var_corrected = corrected_vals.var()
            variance_change = (var_corrected - var_original) / (var_original + 1e-6)
            
            # Standard deviation change
            std_original = original_vals.std()
            std_corrected = corrected_vals.std()
            
            results.append({
                'marker': marker,
                'batch': batch,
                'n_cells': len(batch_data),
                'mean_original': mean_original,
                'mean_corrected': mean_corrected,
                'correction_magnitude': correction_magnitude,
                'variance_original': var_original,
                'variance_corrected': var_corrected,
                'variance_change': variance_change,
                'std_original': std_original,
                'std_corrected': std_corrected
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        print(f"  Analyzed {len(results_df)} marker-batch combinations")
        print(f"  Correction magnitude range: {results_df['correction_magnitude'].min():.3f} - {results_df['correction_magnitude'].max():.3f}")
        print(f"  Mean correction magnitude: {results_df['correction_magnitude'].mean():.3f}")
        
        # Flag high correction magnitudes
        high_correction = results_df[results_df['correction_magnitude'] > 1.0]
        if len(high_correction) > 0:
            print(f"  WARNING: {len(high_correction)} marker-batch combinations have >100% correction magnitude:")
            for _, row in high_correction.head(5).iterrows():
                print(f"    {row['marker']} in {row['batch']}: {row['correction_magnitude']:.1f}x")
    
    return results_df


def calculate_batch_variance_preservation(data, markers, batch_col='batch_id'):
    """
    Calculate how much biological variance is preserved after batch correction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with original and corrected markers
    markers : list
        List of marker names
    batch_col : str
        Batch column name
        
    Returns:
    --------
    dict
        Variance preservation metrics
    """
    print("Calculating variance preservation...")
    
    metrics = {}
    
    for marker in markers:
        original_col = marker
        corrected_col = f"{marker}_corr"
        
        if corrected_col not in data.columns:
            continue
            
        # Get clean data
        mask = data[original_col].notna() & data[corrected_col].notna()
        clean_data = data[mask]
        
        if len(clean_data) < 10:
            continue
            
        # Calculate within-batch variance (biological variance)
        within_batch_var_original = 0
        within_batch_var_corrected = 0
        total_cells = 0
        
        for batch in clean_data[batch_col].unique():
            batch_data = clean_data[clean_data[batch_col] == batch]
            if len(batch_data) < 2:
                continue
                
            within_batch_var_original += batch_data[original_col].var() * len(batch_data)
            within_batch_var_corrected += batch_data[corrected_col].var() * len(batch_data)
            total_cells += len(batch_data)
        
        if total_cells > 0:
            within_batch_var_original /= total_cells
            within_batch_var_corrected /= total_cells
            
            # Variance preservation ratio
            preservation_ratio = within_batch_var_corrected / (within_batch_var_original + 1e-6)
            
            metrics[marker] = {
                'within_batch_var_original': within_batch_var_original,
                'within_batch_var_corrected': within_batch_var_corrected,
                'preservation_ratio': preservation_ratio,
                'variance_loss': 1 - preservation_ratio
            }
    
    if metrics:
        preservation_ratios = [m['preservation_ratio'] for m in metrics.values()]
        variance_losses = [m['variance_loss'] for m in metrics.values()]
        
        print(f"  Analyzed {len(metrics)} markers")
        print(f"  Mean variance preservation: {np.mean(preservation_ratios):.1%}")
        print(f"  Mean variance loss: {np.mean(variance_losses):.1%}")
        
        # Flag markers with high variance loss
        high_loss_markers = [marker for marker, m in metrics.items() if m['variance_loss'] > 0.3]
        if high_loss_markers:
            print(f"  WARNING: {len(high_loss_markers)} markers lost >30% variance:")
            for marker in high_loss_markers[:5]:
                loss = metrics[marker]['variance_loss']
                print(f"    {marker}: {loss:.1%} variance loss")
    
    return metrics


def detect_overcorrected_cells(data, markers, batch_col='batch_id', magnitude_threshold=2.0, percentile_threshold=95.0):
    """
    Detect cells that may have been overcorrected during batch correction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data with original and corrected markers
    markers : list
        List of marker names
    batch_col : str
        Batch column name
    magnitude_threshold : float
        Threshold for flagging large corrections
        
    Returns:
    --------
    pandas.DataFrame
        Data with overcorrection flags
    """
    from scipy.spatial.distance import mahalanobis
    from scipy.linalg import pinv
    
    print("Detecting overcorrected cells...")
    
    data = data.copy()
    data['overcorrected'] = False
    data['correction_distance'] = 0.0
    
    # Calculate correction distances for each cell
    available_markers = [m for m in markers if f"{m}_corr" in data.columns]
    
    if len(available_markers) < 2:
        print("  Insufficient corrected markers for overcorrection detection")
        return data
    
    original_cols = available_markers
    corrected_cols = [f"{m}_corr" for m in available_markers]
    
    # Get clean data
    original_data = data[original_cols].fillna(0)
    corrected_data = data[corrected_cols].fillna(0)
    
    # Calculate correction vectors
    correction_vectors = corrected_data.values - original_data.values
    
    # Calculate covariance matrix of correction vectors
    try:
        cov_matrix = np.cov(correction_vectors.T)
        cov_inv = pinv(cov_matrix)
        
        # Calculate mahalanobis distance for each cell's correction
        distances = []
        for i in range(len(correction_vectors)):
            dist = mahalanobis(correction_vectors[i], np.zeros(len(available_markers)), cov_inv)
            distances.append(dist)
        
        data['correction_distance'] = distances
        
        # Flag cells with high correction distances
        distance_threshold = np.percentile(distances, percentile_threshold)
        data['overcorrected'] = data['correction_distance'] > distance_threshold
        
        overcorrected_count = data['overcorrected'].sum()
        print(f"  Flagged {overcorrected_count:,} cells ({overcorrected_count/len(data):.1%}) as potentially overcorrected")
        
        if overcorrected_count > 0:
            print(f"  Correction distance threshold: {distance_threshold:.2f}")
            print(f"  Max correction distance: {data['correction_distance'].max():.2f}")
            
            # Show batch distribution of overcorrected cells
            overcorrected_by_batch = data[data['overcorrected']].groupby(batch_col).size()
            print(f"  Overcorrected cells by batch:")
            for batch, count in overcorrected_by_batch.items():
                total_in_batch = (data[batch_col] == batch).sum()
                print(f"    {batch}: {count:,} / {total_in_batch:,} ({count/total_in_batch:.1%})")
        
    except Exception as e:
        print(f"  Error in overcorrection detection: {e}")
        data['overcorrected'] = False
        data['correction_distance'] = 0.0
    
    return data


def create_post_correction_diagnostic_plots(data, markers, output_dir, max_cells_plot=50000):
    """
    Create comprehensive diagnostic plots for post-correction assessment.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Batch-corrected data
    markers : list
        List of marker names
    output_dir : Path
        Output directory for plots
    max_cells_plot : int
        Maximum cells to include in plots for performance
        
    Returns:
    --------
    dict
        Dictionary containing plot file paths and key metrics
    """
    print("Creating post-correction diagnostic plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Subsample for plotting if needed
    if len(data) > max_cells_plot:
        plot_data = data.sample(n=max_cells_plot, random_state=42)
        print(f"  Subsampled to {max_cells_plot:,} cells for plotting")
    else:
        plot_data = data.copy()
    
    # Analyze correction magnitude
    correction_analysis = analyze_correction_magnitude(plot_data, markers)
    
    # Calculate variance preservation
    variance_metrics = calculate_batch_variance_preservation(plot_data, markers)
    
    # Create UMAP plots for assessment
    umap_plots = _create_post_correction_umap_plots(plot_data, markers, output_dir)
    
    # Create correction magnitude plots
    correction_plots = _create_correction_magnitude_plots(correction_analysis, output_dir)
    
    # Create variance preservation plots
    variance_plots = _create_variance_preservation_plots(variance_metrics, output_dir)
    
    # Summary metrics for decision making
    summary_metrics = {
        'total_cells': len(data),
        'plot_cells': len(plot_data),
        'mean_correction_magnitude': correction_analysis['correction_magnitude'].mean() if len(correction_analysis) > 0 else 0,
        'high_correction_markers': len(correction_analysis[correction_analysis['correction_magnitude'] > 1.0]) if len(correction_analysis) > 0 else 0,
        'mean_variance_preservation': np.mean([m['preservation_ratio'] for m in variance_metrics.values()]) if variance_metrics else 1.0,
        'high_variance_loss_markers': len([m for m in variance_metrics.values() if m['variance_loss'] > 0.3]) if variance_metrics else 0
    }
    
    # Save summary metrics
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(output_dir / "post_correction_summary_metrics.csv", index=False)
    
    print(f"  Diagnostic plots saved to: {output_dir}")
    print(f"  Mean correction magnitude: {summary_metrics['mean_correction_magnitude']:.2f}")
    print(f"  Mean variance preservation: {summary_metrics['mean_variance_preservation']:.1%}")
    print(f"  Markers with high correction (>100%): {summary_metrics['high_correction_markers']}")
    print(f"  Markers with high variance loss (>30%): {summary_metrics['high_variance_loss_markers']}")
    
    return {
        'summary_metrics': summary_metrics,
        'correction_analysis': correction_analysis,
        'variance_metrics': variance_metrics,
        'plot_dir': output_dir
    }


def _create_post_correction_umap_plots(data, markers, output_dir):
    """Create UMAP plots for post-correction assessment."""
    try:
        import umap
    except ImportError:
        print("  UMAP not available, skipping UMAP plots")
        return None
    
    print("  Creating post-correction UMAP plots...")
    
    # Prepare data
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]
    if len(corrected_markers) < 2:
        print("  Insufficient corrected markers for UMAP")
        return None
    
    # Scale data
    corrected_data = data[corrected_markers].fillna(0)
    scaled_data = StandardScaler().fit_transform(corrected_data)
    
    # Compute UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(scaled_data)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Colored by batch
    scatter1 = axes[0, 0].scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                  c=pd.Categorical(data['batch_id']).codes, 
                                  alpha=0.6, s=1, cmap='tab10')
    axes[0, 0].set_title('UMAP - Colored by Batch')
    axes[0, 0].set_xlabel('UMAP1')
    axes[0, 0].set_ylabel('UMAP2')
    
    # Plot 2: Colored by overcorrection (if available)
    if 'overcorrected' in data.columns:
        colors = ['blue' if not oc else 'red' for oc in data['overcorrected']]
        axes[0, 1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors, alpha=0.6, s=1)
        axes[0, 1].set_title('UMAP - Overcorrected Cells (Red)')
    else:
        axes[0, 1].scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=1)
        axes[0, 1].set_title('UMAP - All Cells')
    axes[0, 1].set_xlabel('UMAP1')
    axes[0, 1].set_ylabel('UMAP2')
    
    # Plot 3: Density plot
    axes[1, 0].hexbin(umap_coords[:, 0], umap_coords[:, 1], gridsize=50, cmap='Blues')
    axes[1, 0].set_title('UMAP - Cell Density')
    axes[1, 0].set_xlabel('UMAP1')
    axes[1, 0].set_ylabel('UMAP2')
    
    # Plot 4: Correction distance (if available)
    if 'correction_distance' in data.columns:
        scatter4 = axes[1, 1].scatter(umap_coords[:, 0], umap_coords[:, 1], 
                                      c=data['correction_distance'], alpha=0.6, s=1, cmap='viridis')
        plt.colorbar(scatter4, ax=axes[1, 1])
        axes[1, 1].set_title('UMAP - Correction Distance')
    else:
        axes[1, 1].scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=1)
        axes[1, 1].set_title('UMAP - All Cells')
    axes[1, 1].set_xlabel('UMAP1')
    axes[1, 1].set_ylabel('UMAP2')
    
    plt.tight_layout()
    plot_path = output_dir / "post_correction_umap_assessment.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save UMAP coordinates for interactive analysis
    umap_df = data[['cell_id', 'batch_id']].copy()
    umap_df['UMAP1'] = umap_coords[:, 0]
    umap_df['UMAP2'] = umap_coords[:, 1]
    if 'overcorrected' in data.columns:
        umap_df['overcorrected'] = data['overcorrected']
    if 'correction_distance' in data.columns:
        umap_df['correction_distance'] = data['correction_distance']
    
    umap_csv_path = output_dir / "post_correction_umap_coordinates.csv"
    umap_df.to_csv(umap_csv_path, index=False)
    
    print(f"    UMAP plots saved: {plot_path}")
    print(f"    UMAP coordinates saved: {umap_csv_path}")
    
    return {'plot_path': plot_path, 'coordinates_path': umap_csv_path}


def _create_correction_magnitude_plots(correction_analysis, output_dir):
    """Create correction magnitude analysis plots."""
    if len(correction_analysis) == 0:
        print("  No correction analysis data for plotting")
        return None
    
    print("  Creating correction magnitude plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Correction magnitude by marker
    marker_correction = correction_analysis.groupby('marker')['correction_magnitude'].mean().sort_values(ascending=False)
    axes[0, 0].bar(range(len(marker_correction)), marker_correction.values)
    axes[0, 0].set_xticks(range(len(marker_correction)))
    axes[0, 0].set_xticklabels(marker_correction.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Mean Correction Magnitude')
    axes[0, 0].set_title('Correction Magnitude by Marker')
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% threshold')
    axes[0, 0].legend()
    
    # Plot 2: Correction magnitude by batch
    batch_correction = correction_analysis.groupby('batch')['correction_magnitude'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(batch_correction)), batch_correction.values)
    axes[0, 1].set_xticks(range(len(batch_correction)))
    axes[0, 1].set_xticklabels(batch_correction.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Mean Correction Magnitude')
    axes[0, 1].set_title('Correction Magnitude by Batch')
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='100% threshold')
    axes[0, 1].legend()
    
    # Plot 3: Distribution of correction magnitudes
    axes[1, 0].hist(correction_analysis['correction_magnitude'], bins=50, alpha=0.7, density=True)
    axes[1, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='100% threshold')
    axes[1, 0].set_xlabel('Correction Magnitude')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Correction Magnitudes')
    axes[1, 0].legend()
    
    # Plot 4: Variance change vs correction magnitude
    axes[1, 1].scatter(correction_analysis['correction_magnitude'], 
                       correction_analysis['variance_change'], alpha=0.6, s=10)
    axes[1, 1].set_xlabel('Correction Magnitude')
    axes[1, 1].set_ylabel('Variance Change')
    axes[1, 1].set_title('Variance Change vs Correction Magnitude')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = output_dir / "correction_magnitude_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Correction magnitude plots saved: {plot_path}")
    return {'plot_path': plot_path}


def _create_variance_preservation_plots(variance_metrics, output_dir):
    """Create variance preservation analysis plots."""
    if not variance_metrics:
        print("  No variance metrics for plotting")
        return None
    
    print("  Creating variance preservation plots...")
    
    # Convert to DataFrame for easier plotting
    variance_df = pd.DataFrame.from_dict(variance_metrics, orient='index')
    variance_df['marker'] = variance_df.index
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Variance preservation by marker
    variance_df_sorted = variance_df.sort_values('preservation_ratio')
    axes[0].barh(range(len(variance_df_sorted)), variance_df_sorted['preservation_ratio'])
    axes[0].set_yticks(range(len(variance_df_sorted)))
    axes[0].set_yticklabels(variance_df_sorted['marker'])
    axes[0].set_xlabel('Variance Preservation Ratio')
    axes[0].set_title('Variance Preservation by Marker')
    axes[0].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='70% threshold')
    axes[0].legend()
    
    # Plot 2: Variance loss distribution
    axes[1].hist(variance_df['variance_loss'], bins=20, alpha=0.7, density=True)
    axes[1].axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='30% loss threshold')
    axes[1].set_xlabel('Variance Loss')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Variance Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plot_path = output_dir / "variance_preservation_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Variance preservation plots saved: {plot_path}")
    return {'plot_path': plot_path}


def apply_post_correction_qc_filters(data, qc_thresholds):
    """
    Apply post-correction QC filters based on user-defined thresholds.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Batch-corrected data with diagnostic metrics
    qc_thresholds : dict
        QC thresholds for filtering
        
    Returns:
    --------
    pandas.DataFrame, dict
        Filtered data and filtering statistics
    """
    print("Applying post-correction QC filters...")
    
    filtered_data = data.copy()
    initial_count = len(filtered_data)
    filtering_stats = {'initial_cells': initial_count}
    
    # Filter overcorrected cells
    if qc_thresholds.get('filter_overcorrected', False) and 'overcorrected' in filtered_data.columns:
        overcorrected_count = filtered_data['overcorrected'].sum()
        filtered_data = filtered_data[~filtered_data['overcorrected']].copy()
        filtering_stats['overcorrected_removed'] = overcorrected_count
        print(f"  Removed {overcorrected_count:,} overcorrected cells")
    
    # Filter by correction distance threshold
    if qc_thresholds.get('max_correction_distance') and 'correction_distance' in filtered_data.columns:
        max_distance = qc_thresholds['max_correction_distance']
        high_distance_count = (filtered_data['correction_distance'] > max_distance).sum()
        filtered_data = filtered_data[filtered_data['correction_distance'] <= max_distance].copy()
        filtering_stats['high_correction_distance_removed'] = high_distance_count
        print(f"  Removed {high_distance_count:,} cells with correction distance > {max_distance}")
    
    # Filter outliers from UMAP coordinates if available
    if qc_thresholds.get('filter_umap_outliers', False):
        umap_coords_file = qc_thresholds.get('umap_coordinates_file')
        if umap_coords_file and Path(umap_coords_file).exists():
            umap_df = pd.read_csv(umap_coords_file)
            
            # Define UMAP outliers (cells far from main cluster)
            umap1_lower = np.percentile(umap_df['UMAP1'], qc_thresholds.get('umap_percentile_low', 1))
            umap1_upper = np.percentile(umap_df['UMAP1'], qc_thresholds.get('umap_percentile_high', 99))
            umap2_lower = np.percentile(umap_df['UMAP2'], qc_thresholds.get('umap_percentile_low', 1))
            umap2_upper = np.percentile(umap_df['UMAP2'], qc_thresholds.get('umap_percentile_high', 99))
            
            # Find outlier cell IDs
            umap_outliers = umap_df[
                (umap_df['UMAP1'] < umap1_lower) | (umap_df['UMAP1'] > umap1_upper) |
                (umap_df['UMAP2'] < umap2_lower) | (umap_df['UMAP2'] > umap2_upper)
            ]['cell_id']
            
            # Filter from main data
            outlier_count = filtered_data['cell_id'].isin(umap_outliers).sum()
            filtered_data = filtered_data[~filtered_data['cell_id'].isin(umap_outliers)].copy()
            filtering_stats['umap_outliers_removed'] = outlier_count
            print(f"  Removed {outlier_count:,} UMAP outlier cells")
    
    # Calculate final statistics
    final_count = len(filtered_data)
    filtering_stats['final_cells'] = final_count
    filtering_stats['total_removed'] = initial_count - final_count
    filtering_stats['retention_rate'] = final_count / initial_count
    
    print(f"\nPost-correction QC summary:")
    print(f"  Initial cells: {initial_count:,}")
    print(f"  Final cells: {final_count:,}")
    print(f"  Removed: {initial_count - final_count:,}")
    print(f"  Retention rate: {final_count/initial_count:.1%}")
    
    return filtered_data, filtering_stats


# =============================================================================
# DIAGNOSTIC VISUALIZATION FUNCTIONS
# =============================================================================

def generate_diagnostic_visualizations(
    corrected_data,
    original_markers,
    corrected_markers,
    batch_col='batch_id',
    max_cells_for_umap=50000,
    umap_params=None,
    figure_size=(1200, 500),
    show_marker_distributions=True,
    markers_to_visualize=None,
    marker_transform='log10',
    histogram_bins=50,
    verbose=True,
    cofactor=5.0
):
    """
    Generate comprehensive diagnostic visualizations for batch correction quality assessment.
    
    Parameters:
    -----------
    corrected_data : pandas.DataFrame
        Data with both original and corrected marker columns
    original_markers : list
        List of original marker column names
    corrected_markers : list
        List of corrected marker column names (with '_corr' suffix)
    batch_col : str
        Column name containing batch identifiers
    max_cells_for_umap : int
        Maximum number of cells to use for UMAP computation
    umap_params : dict
        UMAP parameters (n_neighbors, min_dist, random_state)
    figure_size : tuple
        Default figure size (width, height)
    show_marker_distributions : bool
        Whether to generate marker distribution histograms (before/after comparison)
    markers_to_visualize : list or None
        Specific markers to show in histograms. If None, shows all markers
    marker_transform : str
        Transformation for marker histograms: 'log10', 'arcsinh', or 'none'
    histogram_bins : int
        Number of bins for marker distribution histograms
    verbose : bool
        Print progress messages
    cofactor : float
        Cofactor used for ArcSinh transformation
        
    Returns:
    --------
    dict : Dictionary containing:
        - umap_data: DataFrame with UMAP coordinates
        - statistics: Dictionary of correction statistics
        - recommendations: Dictionary of recommended QC parameters
    """
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        print("Error: plotly is required for diagnostic visualizations")
        return None
    
    try:
        from umap import UMAP
    except ImportError:
        print("Error: umap-learn is required for diagnostic visualizations")
        return None
    
    if verbose:
        print("="*80)
        print("GENERATING DIAGNOSTIC VISUALIZATIONS FOR QC THRESHOLD DETERMINATION")
        print("="*80)
        print(f"Analyzing {len(corrected_data):,} batch-corrected cells\n")
    
    # Set default UMAP parameters
    if umap_params is None:
        umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42}
    
    # Subsample if needed
    if len(corrected_data) > max_cells_for_umap:
        if verbose:
            print(f"Subsampling to {max_cells_for_umap:,} cells for UMAP computation...")
        sample_indices = np.random.choice(corrected_data.index, size=max_cells_for_umap, replace=False)
        umap_data = corrected_data.loc[sample_indices].copy()
    else:
        umap_data = corrected_data.copy()
    
    # Compute UMAP on original and corrected data
    if verbose:
        print("1. Computing UMAP embeddings...")
        print(f"   Using {len(original_markers)} markers for UMAP")
    
    # UMAP on original data (apply ArcSinh transform to match corrected space)
    if verbose:
        print("   Computing UMAP on ORIGINAL (uncorrected) data (ArcSinh space)...")
    umap_reducer_orig = UMAP(n_components=2, **umap_params)
    original_data_arcsinh = np.arcsinh(umap_data[original_markers].values / cofactor)
    umap_orig = umap_reducer_orig.fit_transform(original_data_arcsinh)
    umap_data['UMAP1_original'] = umap_orig[:, 0]
    umap_data['UMAP2_original'] = umap_orig[:, 1]
    
    # UMAP on corrected data (already in ArcSinh space)
    if verbose:
        print("   Computing UMAP on CORRECTED (batch-corrected) data...")
    umap_reducer_corr = UMAP(n_components=2, **umap_params)
    umap_corr = umap_reducer_corr.fit_transform(umap_data[corrected_markers].values)
    umap_data['UMAP1'] = umap_corr[:, 0]
    umap_data['UMAP2'] = umap_corr[:, 1]
    
    # Compute correction distances (both in ArcSinh space)
    if verbose:
        print("   Computing correction distances...")
    original_arcsinh_all = np.arcsinh(umap_data[original_markers].values / cofactor)
    correction_deltas = umap_data[corrected_markers].values - original_arcsinh_all
    correction_distances = np.linalg.norm(correction_deltas, axis=1)
    umap_data['correction_distance'] = correction_distances
    
    if verbose:
        print(f"   UMAP computation complete")
        print(f"   Original UMAP range: X=[{umap_data['UMAP1_original'].min():.1f}, {umap_data['UMAP1_original'].max():.1f}], "
              f"Y=[{umap_data['UMAP2_original'].min():.1f}, {umap_data['UMAP2_original'].max():.1f}]")
        print(f"   Corrected UMAP range: X=[{umap_data['UMAP1'].min():.1f}, {umap_data['UMAP1'].max():.1f}], "
              f"Y=[{umap_data['UMAP2'].min():.1f}, {umap_data['UMAP2'].max():.1f}]")
        print(f"   Correction distance: mean={correction_distances.mean():.2f}, "
              f"median={np.median(correction_distances):.2f}, "
              f"99th percentile={np.percentile(correction_distances, 99):.2f}\n")
    
    # Generate visualizations
    _plot_before_after_umap(umap_data, batch_col, figure_size, verbose)
    _plot_correction_distance_distribution(correction_distances, verbose)
    _plot_umap_with_correction_overlay(umap_data, batch_col, verbose)
    
    # Calculate suggested thresholds
    statistics, recommendations = _calculate_diagnostic_statistics(umap_data, correction_distances, batch_col, verbose)
    
    _plot_umap_with_thresholds(umap_data, recommendations, batch_col, verbose)
    
    # Calculate and display EMD and Silhouette metrics (cyCombine alignment)
    if verbose:
        print("Generating Earth Mover's Distance (EMD) and Silhouette score comparisons...")
    
    emd_df = calculate_batch_emd(corrected_data, original_markers, batch_col, cofactor=cofactor)
    sil_dict = calculate_batch_silhouette(corrected_data, original_markers, batch_col, sample_size=2000)
    
    if verbose:
        print(f"   Silhouette score Before: {sil_dict['silhouette_before']:.3f}")
        print(f"   Silhouette score After: {sil_dict['silhouette_after']:.3f}")
        if not emd_df.empty:
            mean_reduction = emd_df['Reduction'].mean()
            print(f"   Average EMD reduction across markers: {mean_reduction:.1%}")
            
    # Plotly EMD Grouped Bar Chart
    if not emd_df.empty:
        fig_emd = go.Figure()
        fig_emd.add_trace(go.Bar(
            x=emd_df['Marker'],
            y=emd_df['EMD_Before'],
            name='Before Correction (ArcSinh)',
            marker_color='indianred'
        ))
        fig_emd.add_trace(go.Bar(
            x=emd_df['Marker'],
            y=emd_df['EMD_After'],
            name='After Correction (ArcSinh)',
            marker_color='mediumseagreen'
        ))
        fig_emd.update_layout(
            title="Earth Mover's Distance (EMD) per Marker: Before vs. After Correction (Lower is Better)",
            xaxis_title="Marker",
            yaxis_title="Average Pairwise EMD",
            barmode='group',
            height=450,
            width=1200,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        fig_emd.show()
        
    # Plotly Silhouette Bar Chart
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Bar(
        x=['Before Correction', 'After Correction'],
        y=[sil_dict['silhouette_before'], sil_dict['silhouette_after']],
        marker_color=['indianred', 'mediumseagreen'],
        text=[f"{sil_dict['silhouette_before']:.3f}", f"{sil_dict['silhouette_after']:.3f}"],
        textposition='auto',
        width=0.4
    ))
    fig_sil.update_layout(
        title="Batch Identity Silhouette Score Comparison (Lower is Better: Closer to 0 indicates better batch mixing)",
        yaxis_title="Silhouette Score",
        height=400,
        width=600,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    fig_sil.show()
    
    # Generate marker distribution plots (before/after comparison)
    if show_marker_distributions:
        _plot_marker_distributions_before_after(
            umap_data=umap_data,
            original_markers=original_markers,
            corrected_markers=corrected_markers,
            batch_col=batch_col,
            markers_to_show=markers_to_visualize,
            transformation=marker_transform,
            n_bins=histogram_bins,
            verbose=verbose,
            cofactor=cofactor
        )
    
    # Print summary
    _print_diagnostic_summary(umap_data, statistics, recommendations, verbose)
    
    return {
        'umap_data': umap_data,
        'statistics': statistics,
        'recommendations': recommendations
    }


def _plot_before_after_umap(umap_data, batch_col, figure_size, verbose):
    """Generate before/after UMAP comparison plot."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    if verbose:
        print("2. Generating Before/After UMAP comparison...")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('BEFORE Batch Correction (Original)', 'AFTER Batch Correction'),
        horizontal_spacing=0.1
    )
    
    # Get unique batches and create color map with enough colors
    unique_batches = sorted(umap_data[batch_col].unique())
    n_batches = len(unique_batches)
    
    # Extend color palette if needed
    all_colors = (px.colors.qualitative.Plotly + 
                  px.colors.qualitative.D3 + 
                  px.colors.qualitative.Set3 + 
                  px.colors.qualitative.Set2)
    
    # Cycle through colors if we have more batches than colors
    colors = [all_colors[i % len(all_colors)] for i in range(n_batches)]
    batch_color_map = dict(zip(unique_batches, colors))
    
    for batch in unique_batches:
        batch_data = umap_data[umap_data[batch_col] == batch]
        fig.add_trace(
            go.Scatter(
                x=batch_data['UMAP1_original'], y=batch_data['UMAP2_original'],
                mode='markers', name=f'Batch {batch}',
                marker=dict(size=3, color=batch_color_map[batch], opacity=0.6),
                showlegend=True, legendgroup=str(batch)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=batch_data['UMAP1'], y=batch_data['UMAP2'],
                mode='markers', name=f'Batch {batch}',
                marker=dict(size=3, color=batch_color_map[batch], opacity=0.6),
                showlegend=False, legendgroup=str(batch)
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="UMAP1", row=1, col=1)
    fig.update_xaxes(title_text="UMAP1", row=1, col=2)
    fig.update_yaxes(title_text="UMAP2", row=1, col=1)
    fig.update_yaxes(title_text="UMAP2", row=1, col=2)
    fig.update_layout(
        title_text="Batch Correction Quality Assessment: UMAP Before vs After",
        height=500, width=1200, showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    fig.show()
    
    if verbose:
        print("   Before/After UMAP plot generated\n")


def _plot_correction_distance_distribution(correction_distances, verbose):
    """Generate correction distance distribution plot."""
    import plotly.graph_objects as go
    
    if verbose:
        print("3. Generating correction distance distribution...")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=correction_distances, nbinsx=50, name='All cells',
        opacity=0.7, marker_color='steelblue'
    ))
    
    percentiles = [90, 95, 99]
    percentile_values = [np.percentile(correction_distances, p) for p in percentiles]
    
    for p, val in zip(percentiles, percentile_values):
        fig.add_vline(
            x=val, line_dash="dash", line_color="red",
            annotation_text=f"{p}th percentile: {val:.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Correction Distance Distribution (Euclidean distance in marker space)",
        xaxis_title="Correction Distance", yaxis_title="Number of Cells",
        height=400, width=900, showlegend=True
    )
    fig.show()
    
    if verbose:
        print(f"   Correction Distance Statistics:")
        print(f"   - Mean: {correction_distances.mean():.2f}")
        print(f"   - Median: {np.median(correction_distances):.2f}")
        print(f"   - Std Dev: {correction_distances.std():.2f}")
        for p, val in zip(percentiles, percentile_values):
            print(f"   - {p}th percentile: {val:.2f}")
        print(f"\n   RECOMMENDATION: Set OVERCORRECTION_PERCENTILE_THRESHOLD between 95-99")
        print(f"      to filter cells with extreme correction distances.\n")


def _plot_umap_with_correction_overlay(umap_data, batch_col, verbose):
    """Generate UMAP with correction distance overlay."""
    import plotly.express as px
    
    if verbose:
        print("4. Generating UMAP with correction distance overlay...")
    
    fig = px.scatter(
        umap_data, x='UMAP1', y='UMAP2', color='correction_distance',
        color_continuous_scale='Viridis',
        title='Corrected Data UMAP colored by Correction Distance (Darker = higher correction)',
        labels={'correction_distance': 'Correction Distance'},
        hover_data=[batch_col, 'correction_distance']
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(height=600, width=900)
    fig.show()
    
    if verbose:
        print("   Correction distance overlay plot generated\n")


def _calculate_diagnostic_statistics(umap_data, correction_distances, batch_col, verbose):
    """Calculate statistics and recommendations."""
    if verbose:
        print("5. Calculating diagnostic statistics and recommendations...")
    
    umap1_percentiles = np.percentile(umap_data['UMAP1'], [1, 5, 95, 99])
    umap2_percentiles = np.percentile(umap_data['UMAP2'], [1, 5, 95, 99])
    
    statistics = {
        'total_cells': len(umap_data),
        'n_batches': umap_data[batch_col].nunique(),
        'correction_distance_mean': correction_distances.mean(),
        'correction_distance_median': np.median(correction_distances),
        'correction_distance_std': correction_distances.std(),
        'correction_distance_99p': np.percentile(correction_distances, 99),
        'umap1_range': (umap_data['UMAP1'].min(), umap_data['UMAP1'].max()),
        'umap2_range': (umap_data['UMAP2'].min(), umap_data['UMAP2'].max())
    }
    
    recommendations = {
        'umap1_min': np.floor(umap1_percentiles[0]),
        'umap1_max': np.ceil(umap1_percentiles[3]),
        'umap2_min': np.floor(umap2_percentiles[0]),
        'umap2_max': np.ceil(umap2_percentiles[3]),
        'overcorrection_threshold': 99,
        'filter_umap_outliers': False
    }
    
    # Check outlier fraction
    outlier_fraction = ((umap_data['UMAP1'] < recommendations['umap1_min']) | 
                       (umap_data['UMAP1'] > recommendations['umap1_max']) |
                       (umap_data['UMAP2'] < recommendations['umap2_min']) | 
                       (umap_data['UMAP2'] > recommendations['umap2_max'])).sum() / len(umap_data)
    
    recommendations['outlier_fraction'] = outlier_fraction
    recommendations['filter_umap_outliers'] = outlier_fraction > 0.01
    
    if verbose:
        print(f"   UMAP1 percentiles: 1%={umap1_percentiles[0]:.1f}, 5%={umap1_percentiles[1]:.1f}, "
              f"95%={umap1_percentiles[2]:.1f}, 99%={umap1_percentiles[3]:.1f}")
        print(f"   UMAP2 percentiles: 1%={umap2_percentiles[0]:.1f}, 5%={umap2_percentiles[1]:.1f}, "
              f"95%={umap2_percentiles[2]:.1f}, 99%={umap2_percentiles[3]:.1f}")
        print(f"   Outlier fraction: {outlier_fraction*100:.2f}%\n")
    
    return statistics, recommendations


def _plot_umap_with_thresholds(umap_data, recommendations, batch_col, verbose):
    """Generate UMAP with suggested threshold boxes."""
    import plotly.express as px
    
    if verbose:
        print("6. Generating UMAP with suggested filtering thresholds...")
    
    fig = px.scatter(
        umap_data, x='UMAP1', y='UMAP2', color=batch_col,
        title='Corrected Data UMAP with Suggested Filtering Thresholds (Red box shows potential outlier regions)',
        hover_data=[batch_col, 'correction_distance']
    )
    
    fig.add_shape(
        type="rect",
        x0=recommendations['umap1_min'], x1=recommendations['umap1_max'],
        y0=recommendations['umap2_min'], y1=recommendations['umap2_max'],
        line=dict(color="green", width=3, dash="dash"),
        label=dict(text="Suggested KEEP region", textposition="top center")
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.update_layout(height=700, width=1000)
    fig.show()
    
    if verbose:
        print(f"\n   SUGGESTED UMAP THRESHOLDS (based on 1st and 99th percentiles):")
        print(f"      UMAP1_MIN = {recommendations['umap1_min']:.0f}")
        print(f"      UMAP1_MAX = {recommendations['umap1_max']:.0f}")
        print(f"      UMAP2_MIN = {recommendations['umap2_min']:.0f}")
        print(f"      UMAP2_MAX = {recommendations['umap2_max']:.0f}\n")


def _plot_batch_specific_umaps(umap_data, batch_col, verbose):
    """Generate batch-specific UMAP panels - DEPRECATED."""
    # This function is kept for backward compatibility but not called by default
    pass


def _plot_marker_distributions_before_after(
    umap_data, 
    original_markers, 
    corrected_markers,
    batch_col,
    markers_to_show=None,
    transformation='log10',
    n_bins=50,
    verbose=True,
    cofactor=5.0
):
    """
    Generate before/after histograms for selected markers showing all batches overlaid.
    
    Parameters:
    -----------
    umap_data : pd.DataFrame
        Data with original and corrected markers
    original_markers : list
        List of original marker names
    corrected_markers : list
        List of corrected marker names (with '_corr' suffix)
    batch_col : str
        Column name for batch identifier
    markers_to_show : list or None
        Specific markers to visualize. If None, shows all markers
    transformation : str
        'log10', 'arcsinh', or 'none' for x-axis transformation
    n_bins : int
        Number of bins for histogram
    verbose : bool
        Print progress messages
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    if verbose:
        print("7. Generating marker distribution comparisons (Before/After batch correction)...")
    
    # Determine which markers to plot
    if markers_to_show is None:
        markers_list = original_markers
    else:
        markers_list = [m for m in markers_to_show if m in original_markers]
    
    if len(markers_list) == 0:
        print("   Warning: No valid markers to plot")
        return
    
    if verbose:
        print(f"   Plotting {len(markers_list)} markers with {transformation} transformation")
    
    # Get unique batches and create color map
    unique_batches = sorted(umap_data[batch_col].unique())
    n_batches = len(unique_batches)
    
    # Extend color palette if needed
    all_colors = (px.colors.qualitative.Plotly + 
                  px.colors.qualitative.D3 + 
                  px.colors.qualitative.Set3 + 
                  px.colors.qualitative.Set2)
    colors = [all_colors[i % len(all_colors)] for i in range(n_batches)]
    batch_color_map = dict(zip(unique_batches, colors))
    
    # Create subplots for each marker (2 columns: before and after)
    n_markers = len(markers_list)
    fig = make_subplots(
        rows=n_markers, cols=2,
        subplot_titles=None,  # We'll add custom titles
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        row_heights=[1]*n_markers
    )
    
    # Add custom column headers
    fig.add_annotation(
        text="<b>BEFORE Batch Correction</b>",
        xref="paper", yref="paper",
        x=0.25, y=1.02,
        showarrow=False,
        font=dict(size=14)
    )
    fig.add_annotation(
        text="<b>AFTER Batch Correction</b>",
        xref="paper", yref="paper",
        x=0.75, y=1.02,
        showarrow=False,
        font=dict(size=14)
    )
    
    # Helper function for transformation
    # Get axis label suffix
    if transformation == 'log10':
        x_label_suffix = " (log10)"
    elif transformation == 'arcsinh':
        x_label_suffix = f" (arcsinh, cofactor={cofactor})"
    else:
        x_label_suffix = ""
    
    # Plot each marker
    for marker_idx, marker in enumerate(markers_list):
        row = marker_idx + 1
        corrected_marker = f"{marker}_corr"
        
        # Get data for this marker
        original_data = umap_data[marker].values
        corrected_data = umap_data[corrected_marker].values
        batches = umap_data[batch_col].values
        
        # Apply transformation (aligning spaces properly)
        if transformation == 'arcsinh':
            # Transform original to ArcSinh, corrected is already ArcSinh
            original_transformed = np.arcsinh(original_data / cofactor)
            corrected_transformed = corrected_data
        elif transformation == 'log10':
            # Convert corrected to linear, apply log10(x + 1) to both
            corrected_linear = np.sinh(corrected_data) * cofactor
            original_transformed = np.log10(original_data + 1)
            corrected_transformed = np.log10(corrected_linear + 1)
        else: # 'none' (linear scale)
            # Convert corrected to linear, keep original linear
            original_transformed = original_data
            corrected_transformed = np.sinh(corrected_data) * cofactor
        
        # Calculate histogram range (common for both before/after)
        all_values = np.concatenate([original_transformed, corrected_transformed])
        x_min, x_max = np.percentile(all_values[np.isfinite(all_values)], [0.1, 99.9])
        x_range = [x_min, x_max]
        
        # Plot BEFORE (left column) - overlay all batches
        for batch in unique_batches:
            batch_mask = batches == batch
            batch_data = original_transformed[batch_mask]
            batch_data = batch_data[np.isfinite(batch_data)]
            
            fig.add_trace(
                go.Histogram(
                    x=batch_data,
                    name=str(batch),
                    marker_color=batch_color_map[batch],
                    opacity=0.5,
                    nbinsx=n_bins,
                    histnorm='probability density',
                    showlegend=(marker_idx == 0),  # Only show legend for first marker
                    legendgroup=str(batch)
                ),
                row=row, col=1
            )
        
        # Plot AFTER (right column) - overlay all batches
        for batch in unique_batches:
            batch_mask = batches == batch
            batch_data = corrected_transformed[batch_mask]
            batch_data = batch_data[np.isfinite(batch_data)]
            
            fig.add_trace(
                go.Histogram(
                    x=batch_data,
                    name=str(batch),
                    marker_color=batch_color_map[batch],
                    opacity=0.5,
                    nbinsx=n_bins,
                    histnorm='probability density',
                    showlegend=False,
                    legendgroup=str(batch)
                ),
                row=row, col=2
            )
        
        # Update axes for this marker row
        fig.update_xaxes(title_text=f"{marker}{x_label_suffix}", row=row, col=1, range=x_range)
        fig.update_xaxes(title_text=f"{marker}{x_label_suffix}", row=row, col=2, range=x_range)
        fig.update_yaxes(title_text="Density", row=row, col=1)
        fig.update_yaxes(title_text="Density", row=row, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Marker Distributions Before vs After Batch Correction (All Batches Overlaid)",
        height=300 * n_markers,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=9)
        ),
        barmode='overlay'
    )
    
    fig.show()
    
    if verbose:
        print(f"   Marker distribution plots generated for {len(markers_list)} markers")
        print(f"   Transformation applied: {transformation}")
        print(f"   Look for convergence of batch distributions after correction\n")


def _print_diagnostic_summary(umap_data, statistics, recommendations, verbose):
    """Print diagnostic summary and recommendations."""
    if not verbose:
        return
    
    print("="*80)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print(f"\nBATCH CORRECTION QUALITY:")
    print(f"   Total cells analyzed: {statistics['total_cells']:,}")
    print(f"   Number of batches: {statistics['n_batches']}")
    print(f"   Correction distance range: [{statistics['correction_distance_mean']:.2f}]")
    print(f"   High correction cells (>99th percentile): " + 
          f"{(umap_data['correction_distance'] > statistics['correction_distance_99p']).sum():,}")
    
    print(f"\nRECOMMENDED STEP 5 QC PARAMETERS:")
    print(f"   Based on the visualizations above, consider these settings:\n")
    print(f"   # Overcorrection filtering")
    print(f"   FILTER_OVERCORRECTED_CELLS = True")
    print(f"   OVERCORRECTION_PERCENTILE_THRESHOLD = {recommendations['overcorrection_threshold']}  # Adjust between 95-99 based on correction distance plot\n")
    
    if recommendations['filter_umap_outliers']:
        print(f"   # UMAP outlier filtering (WARNING: {recommendations['outlier_fraction']*100:.1f}% cells are potential outliers)")
        print(f"   FILTER_UMAP_OUTLIERS = True")
        print(f"   USE_ABSOLUTE_UMAP_FILTERING = True")
        print(f"   UMAP1_MIN = {recommendations['umap1_min']:.0f}")
        print(f"   UMAP1_MAX = {recommendations['umap1_max']:.0f}")
        print(f"   UMAP2_MIN = {recommendations['umap2_min']:.0f}")
        print(f"   UMAP2_MAX = {recommendations['umap2_max']:.0f}")
    else:
        print(f"   # UMAP outlier filtering (Only {recommendations['outlier_fraction']*100:.2f}% potential outliers - filtering may not be needed)")
        print(f"   FILTER_UMAP_OUTLIERS = False  # Optional - very few outliers detected")
    
    print(f"\n   # Correction distance filtering (usually disabled)")
    print(f"   FILTER_HIGH_CORRECTION_DISTANCE = False  # Enable if you see extreme corrections\n")
    
    print(f"\nNEXT STEPS:")
    print(f"   1. Review the UMAP plots above - check if batches are well-mixed")
    print(f"   2. Examine the correction distance distribution")
    print(f"   3. Look for obvious outlier clusters in the UMAP")
    print(f"   4. Adjust Step 5 parameters if needed based on these visualizations")
    print(f"   5. Proceed to Step 5 to apply QC filters")
    print("="*80)


def calculate_batch_emd(data, markers, batch_col, cofactor=5.0):
    """
    Calculate the average pairwise Earth Mover's Distance (EMD) between batches for each marker,
    both before batch correction (in ArcSinh space) and after batch correction (in ArcSinh space).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing original markers and corrected markers (with '_corr' suffix)
    markers : list
        List of original marker names
    batch_col : str
        Column name for batch ID
    cofactor : float
        Cofactor used for ArcSinh transformation of original data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ['Marker', 'EMD_Before', 'EMD_After', 'Reduction']
    """
    from scipy.stats import wasserstein_distance
    import pandas as pd
    import numpy as np
    
    batches = data[batch_col].dropna().unique()
    if len(batches) < 2:
        return pd.DataFrame(columns=['Marker', 'EMD_Before', 'EMD_After', 'Reduction'])
        
    results = []
    # Subsample to speed up EMD if many cells
    data_subset = data[[batch_col] + list(markers) + [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]].copy()
    if len(data_subset) > 5000:
        data_subset = data_subset.sample(n=5000, random_state=42)
        
    for marker in markers:
        corr_marker = f"{marker}_corr"
        if corr_marker not in data.columns:
            continue
            
        # Get ArcSinh transformed original values (force numeric to avoid string division TypeError)
        data_subset[marker] = pd.to_numeric(data_subset[marker], errors='coerce').fillna(0)
        data_subset[corr_marker] = pd.to_numeric(data_subset[corr_marker], errors='coerce').fillna(0)
        
        orig_vals = np.arcsinh(data_subset[marker] / cofactor).values
        corr_vals = data_subset[corr_marker].values
        
        # Calculate pairwise EMD before and after
        emd_before_list = []
        emd_after_list = []
            
        for i in range(len(batches)):
            for j in range(i + 1, len(batches)):
                b1 = batches[i]
                b2 = batches[j]
                
                b1_mask = data_subset[batch_col] == b1
                b2_mask = data_subset[batch_col] == b2
                
                if b1_mask.sum() > 5 and b2_mask.sum() > 5:
                    # Before (ArcSinh transformed)
                    b1_orig = np.arcsinh(data_subset.loc[b1_mask, marker] / cofactor)
                    b2_orig = np.arcsinh(data_subset.loc[b2_mask, marker] / cofactor)
                    emd_b = wasserstein_distance(b1_orig, b2_orig)
                    emd_before_list.append(emd_b)
                    
                    # After (already in ArcSinh space)
                    b1_corr = data_subset.loc[b1_mask, corr_marker]
                    b2_corr = data_subset.loc[b2_mask, corr_marker]
                    emd_a = wasserstein_distance(b1_corr, b2_corr)
                    emd_after_list.append(emd_a)
                    
        avg_emd_before = np.mean(emd_before_list) if emd_before_list else 0.0
        avg_emd_after = np.mean(emd_after_list) if emd_after_list else 0.0
        reduction = (avg_emd_before - avg_emd_after) / avg_emd_before if avg_emd_before > 0 else 0.0
        
        results.append({
            'Marker': marker,
            'EMD_Before': avg_emd_before,
            'EMD_After': avg_emd_after,
            'Reduction': reduction
        })
        
    return pd.DataFrame(results)


def calculate_batch_silhouette(data, markers, batch_col, sample_size=2000):
    """
    Calculate the Silhouette score for batch mixing before and after correction.
    Lower (closer to 0 or negative) indicates better mixing (less distinct clustering by batch).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with original and corrected markers
    markers : list
        List of original marker names
    batch_col : str
        Column name for batch ID
    sample_size : int
        Number of cells to subsample for computation
        
    Returns:
    --------
    dict
        Dictionary containing silhouette score before and after correction
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]
    if not corrected_markers:
        return {'silhouette_before': 0.0, 'silhouette_after': 0.0}
        
    sample_data = data.sample(n=min(len(data), sample_size), random_state=42) if len(data) > sample_size else data
    batches_encoded = pd.Categorical(sample_data[batch_col]).codes
    
    # Check if there are at least 2 batches in sample
    if len(np.unique(batches_encoded)) < 2:
        return {'silhouette_before': 0.0, 'silhouette_after': 0.0}
        
    # Before correction: scale the original data (coerce to numeric to prevent TypeError)
    orig_numeric = sample_data[markers].apply(pd.to_numeric, errors='coerce').fillna(0)
    orig_data_scaled = StandardScaler().fit_transform(orig_numeric)
    sil_before = silhouette_score(orig_data_scaled, batches_encoded)
    
    # After correction: corrected markers are in ArcSinh space, so Z-score them as well to compare fairly (coerce to numeric)
    corr_numeric = sample_data[corrected_markers].apply(pd.to_numeric, errors='coerce').fillna(0)
    corr_data_scaled = StandardScaler().fit_transform(corr_numeric)
    sil_after = silhouette_score(corr_data_scaled, batches_encoded)
    
    return {
        'silhouette_before': sil_before,
        'silhouette_after': sil_after
    }


def generate_qc_plots_emd_silhouette(emd_df, sil_dict, output_dir):
    """Plot EMD reduction and Silhouette score comparison, saving to output_dir."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # EMD Bar Plot
    if not emd_df.empty:
        plt.figure(figsize=(12, 6))
        # Melt dataframe for sns.barplot
        melted = emd_df.melt(id_vars='Marker', value_vars=['EMD_Before', 'EMD_After'], 
                             var_name='Timepoint', value_name='Average EMD')
        melted['Timepoint'] = melted['Timepoint'].map({'EMD_Before': 'Before Correction', 'EMD_After': 'After Correction'})
        sns.barplot(data=melted, x='Marker', y='Average EMD', hue='Timepoint', palette=['indianred', 'mediumseagreen'])
        plt.xticks(rotation=90)
        plt.title("Earth Mover's Distance (EMD) per Marker: Before vs. After Correction\n(Lower is better: indicates batches are aligned)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "qc_emd_comparison.png", dpi=300)
        plt.close()
        
    # Silhouette Comparison Bar Plot
    plt.figure(figsize=(6, 5))
    scores = [sil_dict['silhouette_before'], sil_dict['silhouette_after']]
    labels = ['Before Correction', 'After Correction']
    colors = ['indianred', 'mediumseagreen']
    bars = plt.bar(labels, scores, color=colors, width=0.5)
    plt.ylabel('Silhouette Score')
    plt.title('Batch Identity Silhouette Score\n(Lower is better: indicates better mixing of batches)', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_dir / "qc_silhouette_comparison.png", dpi=300)
    plt.close()

def calculate_emd_reduction(uncorrected_df, corrected_df, markers, batch_col='batch', cofactor=5.0):
    """
    Calculate the Earth Mover's Distance (EMD) reduction between batches.
    Follows cyCombine methodology: 1 - (EMD_corrected / EMD_uncorrected)
    Values closer to 1 indicate more technical variance removed.
    """
    try:
        from scipy.stats import wasserstein_distance
        import itertools
        import pandas as pd
        import numpy as np
        
        batches = uncorrected_df[batch_col].unique()
        if len(batches) < 2:
            return 0.0
            
        batch_pairs = list(itertools.combinations(batches, 2))
        
        emd_uncorr_total = []
        emd_corr_total = []
        
        for marker in markers:
            corr_col = f"{marker}_corr" if f"{marker}_corr" in corrected_df.columns else marker
            
            for b1, b2 in batch_pairs:
                uncorr_b1_raw = uncorrected_df[uncorrected_df[batch_col] == b1][marker].dropna()
                uncorr_b2_raw = uncorrected_df[uncorrected_df[batch_col] == b2][marker].dropna()
                
                corr_b1 = corrected_df[corrected_df[batch_col] == b1][corr_col].dropna()
                corr_b2 = corrected_df[corrected_df[batch_col] == b2][corr_col].dropna()
                
                if len(uncorr_b1_raw) > 0 and len(uncorr_b2_raw) > 0 and len(corr_b1) > 0 and len(corr_b2) > 0:
                    # Apply ArcSinh transformation to uncorrected linear values to match scale
                    uncorr_b1 = np.arcsinh(pd.to_numeric(uncorr_b1_raw, errors='coerce').fillna(0) / cofactor)
                    uncorr_b2 = np.arcsinh(pd.to_numeric(uncorr_b2_raw, errors='coerce').fillna(0) / cofactor)
                    
                    emd_uncorr_total.append(wasserstein_distance(uncorr_b1, uncorr_b2))
                    emd_corr_total.append(wasserstein_distance(corr_b1, corr_b2))
                    
        mean_emd_uncorr = np.mean(emd_uncorr_total) if emd_uncorr_total else 0.0
        mean_emd_corr = np.mean(emd_corr_total) if emd_corr_total else 0.0
        
        if mean_emd_uncorr == 0:
            return 0.0
            
        return 1.0 - (mean_emd_corr / mean_emd_uncorr)
        
    except ImportError:
        print("scipy is required to calculate EMD.")
        return 0.0

def calculate_mad_score(uncorrected_df, corrected_df, markers, batch_col='batch', cofactor=5.0):
    """
    Calculate the Median Absolute Deviation (MAD) score.
    Follows cyCombine methodology.
    Values closer to 0 indicate better preservation of biological variance.
    """
    import pandas as pd
    import numpy as np
    
    batches = uncorrected_df[batch_col].unique()
    mad_diffs = []
    
    for marker in markers:
        corr_col = f"{marker}_corr" if f"{marker}_corr" in corrected_df.columns else marker
        
        for b in batches:
            uncorr_b_raw = uncorrected_df[uncorrected_df[batch_col] == b][marker].dropna()
            corr_b = corrected_df[corrected_df[batch_col] == b][corr_col].dropna()
            
            if len(uncorr_b_raw) > 0 and len(corr_b) > 0:
                # Apply ArcSinh transformation to uncorrected linear values to match scale
                uncorr_b = np.arcsinh(pd.to_numeric(uncorr_b_raw, errors='coerce').fillna(0) / cofactor)
                uncorr_mad = np.median(np.abs(uncorr_b - np.median(uncorr_b)))
                corr_mad = np.median(np.abs(corr_b - np.median(corr_b)))
                
                if uncorr_mad > 0:
                    mad_diff = np.abs(uncorr_mad - corr_mad) / uncorr_mad
                    mad_diffs.append(mad_diff)
                    
    if mad_diffs:
        return np.mean(mad_diffs)
    return 0.0
