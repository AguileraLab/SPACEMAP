"""
Phenotyping utilities for SPACEMAP V4 pipeline.

This module provides utilities for:
- Batch correction using SOM clustering and ComBat
- Phenotyping methods (Leiden, SOM, Threshold, SCIMAP)
- Quality control plotting
- Data validation and preprocessing
"""

import os
# FIX: Set Protobuf implementation to python
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import scanpy as sc

# Scientific computing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
# import umap  # Lazy loaded to avoid protobuf issues
from scipy.stats import percentileofscore

# SOM implementation
try:
    from minisom import MiniSom
    MINISOM_AVAILABLE = True
except ImportError:
    MINISOM_AVAILABLE = False
    MiniSom = None

# Combat batch correction
try:
    from combat.pycombat import pycombat
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    pycombat = None

# Scanpy for Leiden clustering
try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    pass

# Geometric sketching (density-aware subsampling)
try:
    import geosketch
    GEOSKETCH_AVAILABLE = True
except ImportError:
    GEOSKETCH_AVAILABLE = False
    geosketch = None

# SCIMAP for phenotyping
try:
    import sys
    import io
    _orig_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        import scimap as sm
    finally:
        sys.stdout = _orig_stdout
except ImportError:
    pass


class BatchCorrector:
    """Handles batch correction using SOM-based clustering and ComBat correction."""
    
    def __init__(self, markers, som_grid_size=(8, 8), som_iterations=100000, random_seed=42, verbose=True):
        self.markers = markers
        self.som_grid_size = som_grid_size
        self.som_iterations = som_iterations
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.verbose = verbose
        
    def correct_batch_effects(self, data):
        """
        Perform batch correction using SOM clustering and ComBat.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data with marker columns and batch information
            
        Returns:
        --------
        pandas.DataFrame
            Data with batch-corrected marker columns (suffix '_corr')
        """
        print("Starting batch correction...")
        
        # Force numeric type coercion to avoid scaling/transform TypeErrors
        marker_data = data[self.markers].apply(pd.to_numeric, errors='coerce').fillna(0)
        batches = data['batch'].values
        
        # Scale data for SOM
        scaled_data = self.scaler.fit_transform(marker_data)
        
        # Perform SOM clustering
        print("Performing SOM clustering...")
        som_clusters = self._perform_som_clustering(scaled_data)
        
        # Apply ComBat correction per cluster
        print("Applying per-cluster batch correction...")
        corrected_data = self._apply_combat_correction(marker_data, batches, som_clusters)
        
        # Combine with original data
        result = data.copy()
        result['som_cluster'] = som_clusters
        
        # Add corrected markers
        for marker in self.markers:
            result[f"{marker}_corr"] = corrected_data[marker]
        
        return result
    
    def _perform_som_clustering(self, scaled_data):
        """Perform SOM clustering on scaled data."""
        if not MINISOM_AVAILABLE or MiniSom is None:
            print("MiniSom not available, using KMeans clustering instead...")
            from sklearn.cluster import KMeans
            
            # Use KMeans as fallback
            n_clusters = self.som_grid_size[0] * self.som_grid_size[1]
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
            clusters = kmeans.fit_predict(scaled_data)
            return clusters
        
        # Initialize SOM
        som = MiniSom(
            self.som_grid_size[0], 
            self.som_grid_size[1], 
            scaled_data.shape[1],
            sigma=0.3,
            learning_rate=0.5,
            random_seed=self.random_seed
        )
        
        # Train SOM
        som.train(scaled_data, self.som_iterations)
        
        # Get cluster assignments
        clusters = []
        for i in range(len(scaled_data)):
            winner = som.winner(scaled_data[i])
            cluster_id = winner[0] * self.som_grid_size[1] + winner[1]
            clusters.append(cluster_id)
        
        return np.array(clusters)
    
    def _apply_combat_correction(self, marker_data, batches, clusters):
        """Apply ComBat correction within each cluster."""
        corrected_data = marker_data.copy()
        unique_clusters = np.unique(clusters)
        
        if not COMBAT_AVAILABLE:
            print("pycombat not available, using z-score normalization per batch/cluster...")
            
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                cluster_data = marker_data[cluster_mask]
                cluster_batches = batches[cluster_mask]
                
                # Skip if cluster has only one batch
                if len(np.unique(cluster_batches)) <= 1:
                    continue
                
                # Apply z-score normalization per batch within cluster
                for batch in np.unique(cluster_batches):
                    batch_mask = cluster_batches == batch
                    if np.sum(batch_mask) > 1:  # Need at least 2 samples
                        batch_data = cluster_data[batch_mask]
                        
                        # Z-score normalize
                        for col in batch_data.columns:
                            mean_val = batch_data[col].mean()
                            std_val = batch_data[col].std()
                            
                            if std_val > 0:
                                normalized = (batch_data[col] - mean_val) / std_val
                                corrected_data.loc[cluster_mask & (batches == batch), col] = normalized
            
            return corrected_data
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = marker_data[cluster_mask]
            cluster_batches = batches[cluster_mask]
            
            # Skip if cluster has only one batch
            if len(np.unique(cluster_batches)) <= 1:
                continue
            
            # Apply ComBat correction
            try:
                # Check if any batch has only one sample
                batch_counts = pd.Series(cluster_batches).value_counts()
                has_single_sample_batch = (batch_counts == 1).any()
                
                # Suppress pycombat output if not verbose
                import sys
                import os
                from contextlib import redirect_stdout, redirect_stderr
                
                if not self.verbose:
                    # Redirect output to suppress pycombat messages
                    with open(os.devnull, 'w') as devnull:
                        with redirect_stdout(devnull), redirect_stderr(devnull):
                            if has_single_sample_batch:
                                corrected_cluster = pycombat(
                                    cluster_data.T,
                                    cluster_batches,
                                    mean_only=True
                                ).T
                            else:
                                corrected_cluster = pycombat(
                                    cluster_data.T,
                                    cluster_batches
                                ).T
                else:
                    # Use mean_only=True if there are single-sample batches
                    if has_single_sample_batch:
                        corrected_cluster = pycombat(
                            cluster_data.T,
                            cluster_batches,
                            mean_only=True
                        ).T
                    else:
                        corrected_cluster = pycombat(
                            cluster_data.T,
                            cluster_batches
                        ).T
                
                # Cap values to original range
                for col in corrected_cluster.columns:
                    original_min = cluster_data[col].min()
                    original_max = cluster_data[col].max()
                    corrected_cluster[col] = np.clip(
                        corrected_cluster[col], 
                        original_min, 
                        original_max
                    )
                
                corrected_data.loc[cluster_mask] = corrected_cluster
                
            except Exception as e:
                print(f"Warning: ComBat failed for cluster {cluster_id}: {e}")
                continue
        
        return corrected_data


def create_batch_correction_plots(data, markers, output_dir, dpi=300):
    """Create comprehensive quality control plots for batch correction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Density plots for each marker
    print("Creating density plots...")
    _create_density_plots(data, markers, output_dir, dpi)
    
    # 2. PCA plots
    print("Creating PCA plots...")
    _create_pca_plots(data, markers, output_dir, dpi)
    
    # 3. UMAP plots
    print("Creating UMAP plots...")
    _create_umap_plots(data, markers, output_dir, dpi)
    
    # 4. Batch integration metrics
    print("Creating batch integration plots...")
    _create_batch_integration_plots(data, markers, output_dir, dpi)


def _create_density_plots(data, markers, output_dir, dpi):
    """Create density plots comparing before/after correction."""
    n_markers = len(markers)
    n_cols = 4
    n_rows = (n_markers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, marker in enumerate(markers):
        ax = axes[i]
        
        # Plot original data
        for batch in data['batch_id'].unique():
            batch_data = data[data['batch_id'] == batch]
            ax.hist(batch_data[marker], alpha=0.3, bins=50, density=True, 
                   label=f"{batch} (original)")
        
        # Plot corrected data
        corrected_col = f"{marker}_corr"
        if corrected_col in data.columns:
            for batch in data['batch_id'].unique():
                batch_data = data[data['batch_id'] == batch]
                ax.hist(batch_data[corrected_col], alpha=0.3, bins=50, density=True,
                       label=f"{batch} (corrected)", linestyle='--')
        
        ax.set_title(f"{marker} - Before/After Correction")
        ax.set_xlabel("Expression")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend()
    
    # Remove empty subplots
    for i in range(len(markers), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_dir / "density_plots.png", dpi=dpi, bbox_inches='tight')
    plt.close()


def _create_pca_plots(data, markers, output_dir, dpi):
    """Create PCA plots showing batch separation."""
    # Original data PCA
    original_data = data[markers].fillna(0)
    pca = PCA(n_components=2)
    pca_original = pca.fit_transform(StandardScaler().fit_transform(original_data))
    
    # Corrected data PCA
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]
    if corrected_markers:
        corrected_data = data[corrected_markers].fillna(0)
        pca_corrected = pca.fit_transform(StandardScaler().fit_transform(corrected_data))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original PCA
    scatter = axes[0].scatter(pca_original[:, 0], pca_original[:, 1], 
                             c=pd.Categorical(data['batch_id']).codes, 
                             alpha=0.6, s=1)
    axes[0].set_title("PCA - Original Data")
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


def _create_umap_plots(data, markers, output_dir, dpi):
    """Create UMAP plots showing batch integration."""
    try:
        import umap
    except ImportError:
        print("Warning: umap not installed. Skipping UMAP plots.")
        return

    # Subsample for UMAP if needed
    if len(data) > 50000:
        plot_data = data.sample(n=50000, random_state=42)
    else:
        plot_data = data
    
    # Original data UMAP
    original_data = plot_data[markers].fillna(0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_original = reducer.fit_transform(StandardScaler().fit_transform(original_data))
    
    # Corrected data UMAP
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in plot_data.columns]
    if corrected_markers:
        corrected_data = plot_data[corrected_markers].fillna(0)
        umap_corrected = reducer.fit_transform(StandardScaler().fit_transform(corrected_data))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original UMAP
    scatter = axes[0].scatter(umap_original[:, 0], umap_original[:, 1], 
                             c=pd.Categorical(plot_data['batch_id']).codes, 
                             alpha=0.6, s=1)
    axes[0].set_title("UMAP - Original Data")
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


def _create_batch_integration_plots(data, markers, output_dir, dpi):
    """Create plots showing batch integration metrics."""
    corrected_markers = [f"{m}_corr" for m in markers if f"{m}_corr" in data.columns]
    if not corrected_markers:
        return
    
    # Calculate batch integration metrics
    original_data = data[markers].fillna(0)
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


# Additional phenotyping utility functions
def load_threshold_config(threshold_file):
    """Load threshold configuration from CSV file."""
    thresholds = pd.read_csv(threshold_file)
    return thresholds

def apply_threshold_phenotyping(data, thresholds, use_median=True):
    """Apply threshold-based phenotyping to data."""
    phenotyped_data = data.copy()
    
    for _, row in thresholds.iterrows():
        marker = row['marker']
        
        if use_median and 'median_threshold' in row:
            threshold = row['median_threshold']
        else:
            # Use first available threshold
            threshold_cols = [col for col in row.index if col.startswith('roi_')]
            if threshold_cols:
                threshold = row[threshold_cols[0]]
            else:
                continue
        
        # Apply threshold
        phenotyped_data[f"{marker}_positive"] = (data[marker] > threshold).astype(int)
    
    return phenotyped_data

def create_phenotype_combinations(data, phenotype_definitions):
    """Create phenotype combinations based on marker positivity."""
    phenotyped_data = data.copy()
    
    for _, row in phenotype_definitions.iterrows():
        phenotype_name = row['phenotype_name']
        
        # Create boolean mask for this phenotype
        mask = pd.Series(True, index=data.index)
        
        for marker in row.index:
            if marker in ['phenotype_name', 'hierarchy_level']:
                continue
                
            if f"{marker}_positive" in data.columns:
                expected_value = row[marker]
                if expected_value == 1:
                    mask &= (data[f"{marker}_positive"] == 1)
                elif expected_value == 0:
                    mask &= (data[f"{marker}_positive"] == 0)
        
        phenotyped_data[f"phenotype_{phenotype_name}"] = mask.astype(int)
    
    return phenotyped_data

def apply_flexible_normalization(df, marker_cols, group_cols, suffix,
                                 apply_z1=True, apply_z2=True, apply_cdf=True, apply_nlog=True, epsilon=1e-9):
    """
    Apply a flexible sequence of normalization steps to marker data.
    Aligned with comprehensive_phenotyping_gemini_v8_ML.py reference implementation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - marker_cols (list): List of marker columns to normalize.
    - group_cols (list or None): Columns to group by for normalization. If None, normalize globally.
    - suffix (str): Suffix to add to the new normalized columns (e.g., '_norm').
    - apply_z1 (bool): Apply marker-wise Z-score.
    - apply_z2 (bool): Apply cell-wise Z-score.
    - apply_cdf (bool): Apply CDF transformation.
    - apply_nlog (bool): Apply -Log transformation.
    - epsilon (float): Small value to prevent division by zero or log(0).

    Returns:
    - pd.DataFrame: DataFrame with ONLY the new normalized columns.
    """
    from scipy.stats import norm
    
    # Ensure all marker columns exist
    missing_markers = [m for m in marker_cols if m not in df.columns]
    if missing_markers:
        raise ValueError(f"Required marker columns not found in DataFrame: {missing_markers}")

    # Ensure grouping columns exist if specified
    valid_group_cols = []
    if group_cols:
        valid_group_cols = [col for col in group_cols if col in df.columns]
        missing_group_cols = [col for col in group_cols if col not in df.columns]
        if missing_group_cols:
            print(f"Warning: Grouping columns not found, will normalize across all data: {missing_group_cols}")
            valid_group_cols = []

    # Select only necessary columns for processing
    cols_to_process = list(set(marker_cols + valid_group_cols))
    df_proc = df[cols_to_process].copy()

    if not any([apply_z1, apply_z2, apply_cdf, apply_nlog]):
        print("No normalization steps selected. Returning empty DataFrame.")
        return pd.DataFrame(index=df.index)

    new_col_names = [col + suffix for col in marker_cols]

    # Define the normalization function to apply to each group (or the whole df)
    def normalize_subset(subset_df):
        subset_df = subset_df.copy()
        subset_marker_cols = [mc for mc in marker_cols if mc in subset_df.columns]
        if not subset_marker_cols: 
            return pd.DataFrame(columns=new_col_names, index=subset_df.index)

        # Use .values for numpy operations, ensure float type
        current_data = subset_df[subset_marker_cols].astype(float).values

        if subset_df.shape[0] < 2 and (apply_z1 or apply_z2):
            # Handle small groups
            final_data = np.zeros((subset_df.shape[0], len(subset_marker_cols)), dtype=float)
        else:
            processed_data = current_data.copy()
            if apply_z1:
                means_z1 = np.nanmean(processed_data, axis=0)
                stds_z1 = np.nanstd(processed_data, axis=0)
                stds_z1[stds_z1 == 0] = epsilon
                processed_data = (processed_data - means_z1) / stds_z1
            if apply_z2:
                means_z2 = np.nanmean(processed_data, axis=1, keepdims=True)
                stds_z2 = np.nanstd(processed_data, axis=1, keepdims=True)
                stds_z2[stds_z2 == 0] = epsilon
                processed_data = (processed_data - means_z2) / stds_z2
            if apply_cdf:
                processed_data = norm.cdf(processed_data)
            if apply_nlog:
                processed_data = -np.log(np.maximum(1 - processed_data, epsilon))

            # Handle potential NaNs/Infs
            final_data = np.nan_to_num(processed_data, nan=0.0, 
                                     posinf=np.finfo(processed_data.dtype).max, 
                                     neginf=np.finfo(processed_data.dtype).min)

        # Create DataFrame for results
        subset_new_col_names = [m + suffix for m in subset_marker_cols]
        subset_result_df = pd.DataFrame(final_data, index=subset_df.index, columns=subset_new_col_names)
        return subset_result_df

    # Perform normalization
    if valid_group_cols:
        results_list = [normalize_subset(group) for _, group in df_proc.groupby(valid_group_cols, group_keys=False)]
        if results_list:
            normalized_df = pd.concat(results_list).sort_index()
        else:
            normalized_df = pd.DataFrame(index=df_proc.index, columns=new_col_names)
    else:
        normalized_df = normalize_subset(df_proc)

    # Final check for NaNs
    valid_new_cols = [col for col in new_col_names if col in normalized_df.columns]
    if valid_new_cols and normalized_df[valid_new_cols].isnull().values.any():
        print(f"Warning: NaNs detected after normalization. Filling with 0.")
        normalized_df[valid_new_cols] = normalized_df[valid_new_cols].fillna(0)

    return normalized_df[valid_new_cols]


def load_or_compute_normalized(df, marker_cols, group_cols, suffix,
                               normalization_flags, cache_path=None):
    """
    Load normalized markers from a cache CSV if available and valid; otherwise compute
    using apply_flexible_normalization and optionally save to cache.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing marker_cols.
    marker_cols : list[str]
        Marker columns to normalize (e.g., ['CD3e_corr', ...]).
    group_cols : list[str] | None
        Grouping columns for normalization.
    suffix : str
        Suffix for normalized columns (e.g., '_norm').
    normalization_flags : dict
        Keys: 'z1','z2','cdf','nlog' booleans controlling steps.
    cache_path : Path-like | None
        If provided, read/write normalized cache CSV at this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with ONLY normalized columns, aligned to df.index order.
    """
    from pathlib import Path

    desired_cols = [f"{m}{suffix}" for m in marker_cols]

    # Try cache first
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path)
                # Heuristic: if cache has same number of rows and all desired columns, accept
                if len(cached) == len(df) and all(c in cached.columns for c in desired_cols):
                    # Ensure order aligns with current df order; assume same source/order
                    return cached[desired_cols].copy()
            except Exception:
                pass  # Fall back to compute

    # Compute
    norm_df = apply_flexible_normalization(
        df=df,
        marker_cols=marker_cols,
        group_cols=group_cols,
        suffix=suffix,
        apply_z1=normalization_flags.get('z1', True),
        apply_z2=normalization_flags.get('z2', False),
        apply_cdf=normalization_flags.get('cdf', False),
        apply_nlog=normalization_flags.get('nlog', False)
    )

    # Save cache if requested
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Write only normalized columns in order
            norm_df[desired_cols].to_csv(cache_path, index=False)
        except Exception:
            pass

    return norm_df[desired_cols]


def assign_phenotypes_to_clusters(cluster_means_df, marker_suffix='_norm', phenotype_definitions=None, 
                                phenotype_hierarchy=None, expression_threshold=0.1):
    """
    Assign phenotypes to clusters based on marker expression patterns.
    Aligned with comprehensive_phenotyping_gemini_v8_ML.py reference implementation.
    
    Parameters:
    - cluster_means_df: DataFrame with clusters as index and marker values as columns
    - marker_suffix: Suffix used in marker column names (e.g., '_norm')
    - phenotype_definitions: Dict with phenotype definitions
    - phenotype_hierarchy: Dict with phenotype hierarchy
    - expression_threshold: Threshold for considering a marker as positive
    
    Returns:
    - DataFrame with cluster IDs and assigned phenotypes
    """
    if phenotype_definitions is None:
        # Default phenotype definitions
        phenotype_definitions = {
            "T_cells": [["CD45", "CD3e"]],
            "CD4_T_cells": [["CD45", "CD3e", "CD4"]],
            "CD8_T_cells": [["CD45", "CD3e", "CD8"]],
            "Treg": [["CD45", "CD3e", "CD4", "FoxP3"]],
            "B_cells": [["CD45", "CD20"]],
            "Macrophages": [["CD45", "CD68"]],
            "Dendritic_cells": [["CD45", "CD11c", "HLA-DR"]],
            "Epithelial_cells": [["E-cadherin"], ["Pan-Cytokeratin"]],
            "Endothelial_cells": [["CD31"]],
            "Stromal_cells": [["Vimentin"], ["aSMA"]]
        }
    
    print(f"Assigning phenotypes to {len(cluster_means_df)} clusters...")
    
    # Create phenotype assignment DataFrame
    cluster_phenotypes = pd.DataFrame(index=cluster_means_df.index)
    cluster_phenotypes['phenotype'] = 'Unknown'
    cluster_phenotypes['confidence'] = 0.0
    
    # For each cluster, check which phenotype definitions it matches
    for cluster_id in cluster_means_df.index:
        cluster_expression = cluster_means_df.loc[cluster_id]
        best_phenotype = 'Unknown'
        best_score = 0.0
        
        for phenotype, or_conditions in phenotype_definitions.items():
            phenotype_score = 0.0
            
            # Check each OR condition (list of markers that must ALL be positive)
            for and_condition in or_conditions:
                # Check if ALL markers in this AND condition are positive
                condition_met = True
                condition_score = 0.0
                
                for marker in and_condition:
                    marker_col = f"{marker}{marker_suffix}"
                    if marker_col in cluster_expression.index:
                        expr_val = cluster_expression[marker_col]
                        if expr_val > expression_threshold:
                            condition_score += expr_val
                        else:
                            condition_met = False
                            break
                    else:
                        condition_met = False
                        break
                
                # If this AND condition is met, use it for scoring
                if condition_met:
                    phenotype_score = max(phenotype_score, condition_score)
            
            # Update best phenotype if this one scores higher
            if phenotype_score > best_score:
                best_score = phenotype_score
                best_phenotype = phenotype
        
        cluster_phenotypes.loc[cluster_id, 'phenotype'] = best_phenotype
        cluster_phenotypes.loc[cluster_id, 'confidence'] = best_score
    
    return cluster_phenotypes



def find_roi_threshold_file(input_csv_path, threshold_dir):
    """
    Find the appropriate ROI threshold CSV file based on the input CSV filename.
    Aligned with comprehensive_phenotyping_gemini_v8_ML.py reference implementation.
    """
    import re
    
    print(f"Searching for ROI threshold file corresponding to '{os.path.basename(input_csv_path)}' in '{threshold_dir}'...")

    if not os.path.isdir(threshold_dir):
        raise FileNotFoundError(f"Threshold directory not found at: {threshold_dir}")

    # Extract base name from input CSV filename
    input_filename_base = os.path.basename(input_csv_path)
    filename_no_ext, _ = os.path.splitext(input_filename_base)
    delimiters = ['.ome_cell_features', '.ome', '_cell_features', '_annotated', '_processed', '_summary', '_16bit', '_corrected_with_raw', '_corrected']
    base_name = None
    
    for delim in delimiters:
        if delim in filename_no_ext:
            potential_base, _, _ = filename_no_ext.partition(delim)
            if potential_base and potential_base != filename_no_ext:
                base_name = potential_base
                print(f"  Extracted base name using delimiter '{delim}': '{base_name}'")
                break
    
    if base_name is None:
        base_name = filename_no_ext
        print(f"  No specific delimiter found. Using full name before extension as base: '{base_name}'")
    
    base_name = base_name.rstrip('._ ')
    if not base_name:
        print("Warning: Base name extraction resulted in an empty string.")
        base_name = None

    threshold_file_path = None
    if base_name:
        # Search for the file using common patterns
        potential_thresh_files = [
            os.path.join(threshold_dir, f"{base_name}.ome.tiff_-_Merged_Image.csv"),
            os.path.join(threshold_dir, f"{base_name}_-_Merged_Image.csv"),
            os.path.join(threshold_dir, f"{base_name}.csv"),
        ]
        for potential_path in potential_thresh_files:
            if os.path.exists(potential_path):
                threshold_file_path = potential_path
                print(f"  Found matching ROI threshold file: {threshold_file_path}")
                break

    # Fallback: look for any file matching patterns
    if threshold_file_path is None:
        print("  No exact match found. Searching for any file matching common patterns...")
        potential_files = [f for f in os.listdir(threshold_dir) if f.endswith('.csv') and 'Merged_Image' in f]
        if not potential_files:
            potential_files = [f for f in os.listdir(threshold_dir) if f.endswith('.csv')]

        if not potential_files:
            raise FileNotFoundError(f"No CSV files found in threshold directory: {threshold_dir}")

        threshold_file_path = os.path.join(threshold_dir, potential_files[0])
        print(f"  Using first found threshold file: {threshold_file_path}")
        if len(potential_files) > 1:
            print(f"  WARNING: Found multiple potential threshold files. Using the first one: {potential_files[0]}")

    return threshold_file_path


def manual_rescale_scimap(adata, gates=None, apply_thresholds=True):
    """
    Manual rescaling function for SCIMAP aligned with reference SCIMAP_v1_working.py implementation.
    Maps expression values to a [0, 1] range based on provided gates or scales to 0.5-1.0 range when thresholds are disabled.
    
    Parameters:
    - adata (anndata.AnnData): The AnnData object with log-transformed data
    - gates (pd.Series): Series with markers as index and log-transformed gate values (required if apply_thresholds=True)
    - apply_thresholds (bool): If True, use gates for scaling. If False, scale to 0.5-1.0 range directly
    
    Returns:
    - np.ndarray: The rescaled data array
    """
    data = adata.X.copy()
    
    if not apply_thresholds:
        # -----------------------------------------------------------------------
        # WARNING: Running without thresholds is strongly discouraged for IMC/IF
        # data. This mode uses 0 as the log-space gate anchor:
        #   log1p(expr) == 0 (i.e., raw count = 0)  -> rescaled to 0.0  (<gate)
        #   log1p(expr) >  0 (any positive count)   -> rescaled to (0.5, 1.0]
        # For reliable phenotyping, set APPLY_THRESHOLDS = True.
        # -----------------------------------------------------------------------
        print(
            "  WARNING: Scaling WITHOUT thresholds. Gate anchor = 0 (log-space).\n"
            "  Any cell with log1p(expr) > 0 will be marked positive (>0.5).\n"
            "  For reliable IMC/IF phenotyping, set APPLY_THRESHOLDS = True."
        )
        if hasattr(data, "toarray"):
            data = data.toarray()
        data = data.astype(np.float64)

        for i, marker_name in enumerate(adata.var_names):
            col = data[:, i]
            max_val = float(col.max())
            pos_mask = col > 0.0

            col_out = col.copy()
            if pos_mask.any() and max_val > 0.0:
                # Linear map: 0 -> 0.5,  max_val -> 1.0
                col_out[pos_mask] = 0.5 + 0.5 * (col[pos_mask] / max_val)
            col_out[~pos_mask] = 0.0  # true negatives stay below gate
            data[:, i] = np.clip(col_out, 0.0, 1.0)

            pos_frac = pos_mask.mean() * 100.0
            print(f"    -> {marker_name}: log1p max={max_val:.3f}, {pos_frac:.1f}% cells >0 (mapped to >0.5)")

        return data
    
    else:
        if gates is None:
            raise ValueError("gates parameter is required when apply_thresholds=True")

        print("  Scaling data using provided gates (SCIMAP method):")
        if hasattr(data, "toarray"):
            data = data.toarray()
        data = data.astype(np.float64)
        max_values = data.max(axis=0)

        for i, marker_name in enumerate(adata.var_names):
            if marker_name not in gates.index:
                print(f"    -> WARNING: No gate found for marker {marker_name}. Leaving unscaled.")
                continue

            gate_value = float(gates.loc[marker_name])
            max_val = float(max_values[i])
            marker_column = data[:, i]

            below_gate_mask = marker_column < gate_value
            above_gate_mask = marker_column >= gate_value

            # Below gate -> [0, 0.5)
            data[below_gate_mask, i] = 0.5 * (marker_column[below_gate_mask] / (gate_value + 1e-9))

            # Above gate -> [0.5, 1.0]
            denominator = (max_val - gate_value) if (max_val - gate_value) > 0 else 1e-9
            data[above_gate_mask, i] = 0.5 + 0.5 * (
                (marker_column[above_gate_mask] - gate_value) / denominator
            )

            pos_frac = above_gate_mask.mean() * 100.0
            print(f"    -> {marker_name}: gate={gate_value:.4f}, {pos_frac:.1f}% cells above gate")

        return np.clip(data, 0.0, 1.0)


def create_phenotype_strategy_csv(output_path, phenotype_definitions=None):
    """
    Create a phenotype strategy CSV file for SCIMAP aligned with reference implementation.
    
    Parameters:
    - output_path: Path where to save the CSV file
    - phenotype_definitions: Dict with phenotype definitions (optional)
    """
    if phenotype_definitions is None:
        # Default SCIMAP phenotype strategy from reference
        phenotype_strategy = pd.DataFrame({
            'phenotype': [
                'T_cells', 'CD4_T_cells', 'CD8_T_cells', 'Treg', 'B_cells', 
                'Macrophages', 'Dendritic_cells', 'Epithelial_cells', 
                'Endothelial_cells', 'Stromal_cells'
            ],
            'pos': [
                'CD45,CD3e', 'CD45,CD3e,CD4', 'CD45,CD3e,CD8', 'CD45,CD3e,CD4,FoxP3',
                'CD45,CD20', 'CD45,CD68', 'CD45,CD11c,HLA-DR', 'Pan-Cytokeratin',
                'CD31', 'Vimentin'
            ],
            'neg': [
                '', '', '', '', '', '', '', 'CD45', 'CD45', 'CD45,Pan-Cytokeratin'
            ]
        })
    else:
        # Convert custom phenotype definitions to SCIMAP format
        phenotypes = []
        pos_markers = []
        neg_markers = []
        
        for pheno_name, definition in phenotype_definitions.items():
            phenotypes.append(pheno_name)
            if 'positive' in definition:
                pos_markers.append(','.join(definition['positive']))
            else:
                pos_markers.append('')
            if 'negative' in definition:
                neg_markers.append(','.join(definition['negative']))
            else:
                neg_markers.append('')
        
        phenotype_strategy = pd.DataFrame({
            'phenotype': phenotypes,
            'pos': pos_markers,
            'neg': neg_markers
        })
    
    # Save the strategy file
    phenotype_strategy.to_csv(output_path, index=False)
    print(f"Phenotype strategy file created: {output_path}")
    
    return phenotype_strategy


def create_phenotype_strategy_df(phenotype_definitions):
    """
    Convert a phenotype definition dictionary into a SCIMAP-compatible strategy DataFrame.
    The output DataFrame will have 'phenotype', 'pos', and 'neg' columns.
    Handles OR conditions by creating numbered sub-phenotypes.
    
    Returns
    -------
    tuple
        - pd.DataFrame: The SCIMAP-compatible strategy DataFrame.
        - dict: A mapping of sub-phenotypes to their parent phenotype.
    """
    import pandas as pd
    strategy_list = []
    sub_pheno_map = {}  # To map sub-phenotypes back to their parent
    
    for phenotype, marker_groups in phenotype_definitions.items():
        if len(marker_groups) == 1:
            # Single definition (AND only)
            strategy_list.append({
                'phenotype': phenotype,
                'pos': ','.join(marker_groups[0]),
                'neg': ''
            })
        else:
            # Multiple definitions (OR logic)
            for i, marker_group in enumerate(marker_groups):
                sub_phenotype = f"{phenotype}_{i+1}"
                sub_pheno_map[sub_phenotype] = phenotype
                strategy_list.append({
                    'phenotype': sub_phenotype,
                    'pos': ','.join(marker_group),
                    'neg': ''
                })
                
    return pd.DataFrame(strategy_list), sub_pheno_map

def resolve_sub_phenotypes(phenotype_series, sub_pheno_map):
    """
    Combine numbered sub-phenotypes (e.g., 'Pheno_1', 'Pheno_2') into a single parent phenotype
    using a provided mapping.
    """
    # Replace sub-phenotypes with their parent phenotype based on the map
    # Any phenotype not in the map will remain unchanged.
    return phenotype_series.replace(sub_pheno_map)


def load_polygon_threshold_file(threshold_dir, base_prefix, stat='median'):
    """Load a threshold file of form <base_prefix>.ome.tiff_-_Merged_Image.csv and compute per-marker thresholds.

    Parameters
    ----------
    threshold_dir : Path-like
    base_prefix : str
        Inferred base (e.g. P11_roi_1)
    stat : {'median','mean','min'}
        Aggregation across Polygon_* columns.

    Returns
    -------
    dict(marker -> threshold_value) or empty dict if file not found.
    """
    from pathlib import Path
    import pandas as pd
    threshold_dir = Path(threshold_dir)
    fname = f"{base_prefix}.ome.tiff_-_Merged_Image.csv"
    fpath = threshold_dir / fname
    if not fpath.exists():
        return {}
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return {}
    polygon_cols = [c for c in df.columns if c.lower().startswith('polygon')]
    if not polygon_cols:
        return {}
    agg_func = {'median': pd.Series.median, 'mean': pd.Series.mean, 'min': pd.Series.min}.get(stat, pd.Series.median)
    thresholds = {}
    for _, row in df.iterrows():
        marker = row.get('Marker')
        if marker is None:
            continue
        values = pd.to_numeric(row[polygon_cols], errors='coerce').dropna()
        if len(values) == 0:
            continue
        thresholds[marker] = float(agg_func(values))
    return thresholds


def subtract_thresholds_per_batch(df, markers, threshold_dir, stat='median', batch_col='batch_id', out_suffix='_thr'):
    """For each batch, load its threshold file and create threshold-subtracted marker columns.

    New columns: <marker><out_suffix>. Value = max(raw - threshold, 0).
    Missing thresholds default to 0.
    """
    from collections import defaultdict
    thresholds_cache = {}
    new_cols = []
    for batch in df[batch_col].unique():
        base = _infer_threshold_base(str(batch))
        if base not in thresholds_cache:
            thresholds_cache[base] = load_polygon_threshold_file(threshold_dir, base, stat=stat)
        th_map = thresholds_cache[base]
        mask = df[batch_col] == batch
        for m in markers:
            raw_vals = df.loc[mask, m] if m in df.columns else None
            if raw_vals is None:
                continue
            thr = th_map.get(m, 0.0)
            col_name = f"{m}{out_suffix}"
            df.loc[mask, col_name] = (raw_vals - thr).clip(lower=0)
            if col_name not in new_cols:
                new_cols.append(col_name)
    return df, new_cols


def build_phenotype_indicator_matrix(df, phenotype_definitions, marker_suffix='_thr', min_positive=0.0):
    """Create binary phenotype indicator columns (1/0) based on thresholded marker columns.

    phenotype_definitions: dict {phenotype: [ [AND set], [AND set 2], ... ] }
    A phenotype is positive if ANY AND set has all markers > 0.
    Adds columns with phenotype names; returns list of column names.
    """
    pheno_cols = []
    for pheno, or_conditions in phenotype_definitions.items():
        col = []
        for idx in df.index:
            positive = False
            for and_set in or_conditions:
                ok = True
                for marker in and_set:
                    col_name = f"{marker}{marker_suffix}"
                    if col_name not in df.columns or df.at[idx, col_name] <= 0:
                        ok = False
                        break
                if ok:
                    positive = True
                    break
            col.append(1 if positive else 0)
        df[pheno] = col
        pheno_cols.append(pheno)
    return df, pheno_cols


def _phenotype_marker_sets(phenotype_definitions):
    sets = {}
    for pheno, or_conditions in phenotype_definitions.items():
        # union of markers across all AND sets
        mset = set()
        for and_set in or_conditions:
            mset.update(and_set)
        sets[pheno] = mset
    return sets


def resolve_phenotype_ties(df, phenotype_definitions, pheno_cols, marker_suffix='_thr', norm_suffix='_thr_norm', group_col='batch_id', normalization_flags=None):
    """Assign phenotype_thr using tie-break & subset logic.

    Rules:
    - Single positive -> that phenotype
    - Subset (superset marker set) wins (largest marker set that is superset of others)
    - Else choose phenotype whose unique (non-shared) marker among candidates has highest normalized value
      (uses columns marker+norm_suffix). If normalization not available, falls back to raw thresholded markers.
    - If still tie -> first phenotype (sorted) deterministic.
    - No positives -> 'unclassified'
    """
    marker_sets = _phenotype_marker_sets(phenotype_definitions)
    result = []
    # Precompute available normalized columns detection
    use_norm = any(c.endswith(norm_suffix) for c in df.columns)
    for idx, row in df.iterrows():
        positives = [p for p in pheno_cols if row[p] == 1]
        if not positives:
            result.append('unclassified')
            continue
        if len(positives) == 1:
            result.append(positives[0])
            continue
        # Subset rule: choose phenotype with largest marker set that is superset of others it overlaps with
        positives_sorted = sorted(positives, key=lambda p: len(marker_sets[p]), reverse=True)
        top = positives_sorted[0]
        # Check if its marker set is superset of all others' sets (strictly)
        if all(marker_sets[top].issuperset(marker_sets[p]) for p in positives if p != top) and any(marker_sets[top] != marker_sets[p] for p in positives if p != top):
            result.append(top)
            continue
        # Tie-break via normalized values
        # Determine union & unique markers
        other_union = {}
        scores = {}
        # Precompute marker sets per phenotype limited to markers actually positive (>0) in the row
        pos_marker_sets = {}
        for p in positives:
            # choose the first satisfied AND set to define markers considered
            chosen = None
            for and_set in phenotype_definitions[p]:
                if all(row.get(f"{m}{marker_suffix}", 0) > 0 for m in and_set):
                    chosen = set(and_set)
                    break
            if chosen is None:
                chosen = marker_sets[p]
            pos_marker_sets[p] = chosen
        all_markers_union = set().union(*pos_marker_sets.values())
        for p in positives:
            shared = set()
            for q in positives:
                if q == p:
                    continue
                shared.update(pos_marker_sets[p].intersection(pos_marker_sets[q]))
            unique = pos_marker_sets[p] - shared
            if not unique:
                unique = pos_marker_sets[p]
            # Score = max normalized (or raw thresholded) among unique markers
            vals = []
            for m in unique:
                col_norm = f"{m}{norm_suffix}" if use_norm else f"{m}{marker_suffix}"
                if col_norm in df.columns:
                    vals.append(row[col_norm])
            scores[p] = max(vals) if vals else 0
        # Pick phenotype with highest score
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        result.append(best)
    df['phenotype_thr'] = result
    return df


def threshold_based_phenotyping(
    df,
    markers,
    phenotype_definitions,
    thresholds_dir,
    threshold_stat='median',
    batch_col='batch_id',
    normalization_flags=None,
    apply_normalization=True,
    marker_suffix='_thr',
    norm_suffix='_thr_norm'
):
    """High-level convenience pipeline performing threshold subtraction, indicator creation,
    optional normalization, and tie-breaking to assign phenotype_thr.
    Returns updated df and metadata dict.
    """
    if normalization_flags is None:
        normalization_flags = dict(z1=True, z2=False, cdf=False, nlog=False)
    # 1. Threshold subtraction
    df, thr_cols = subtract_thresholds_per_batch(df, markers, thresholds_dir, stat=threshold_stat, batch_col=batch_col, out_suffix=marker_suffix)
    # 2. Phenotype indicator columns
    df, pheno_cols = build_phenotype_indicator_matrix(df, phenotype_definitions, marker_suffix=marker_suffix)
    # 3. Normalization (applied to thresholded marker cols)
    if apply_normalization and any(normalization_flags.values()):
        norm_df = apply_flexible_normalization(
            df=df,
            marker_cols=[f"{m}{marker_suffix}" for m in markers if f"{m}{marker_suffix}" in df.columns],
            group_cols=[batch_col],
            suffix=norm_suffix,
            apply_z1=normalization_flags.get('z1', True),
            apply_z2=normalization_flags.get('z2', False),
            apply_cdf=normalization_flags.get('cdf', False),
            apply_nlog=normalization_flags.get('nlog', False)
        )
        df = pd.concat([df, norm_df], axis=1)
    # 4. Tie-breaking + subset rules
    df = resolve_phenotype_ties(
        df,
        phenotype_definitions=phenotype_definitions,
        pheno_cols=pheno_cols,
        marker_suffix=marker_suffix,
        norm_suffix=norm_suffix,
        group_col=batch_col,
        normalization_flags=normalization_flags
    )
    meta = {
        'threshold_columns': thr_cols,
        'phenotype_indicator_columns': pheno_cols,
        'normalized': any(c.endswith(norm_suffix) for c in df.columns)
    }
    return df, meta


# =============================================================================
# GEOMETRIC SKETCHING + LEIDEN HYPERPARAMETER OPTIMIZATION (HPO)
# =============================================================================

def create_geometric_sketch(
    data: pd.DataFrame,
    marker_cols,
    sketch_size: int = 50000,
    max_pcs: int = 50,
    random_seed: int = 42,
    scale: bool = True,
    float32: bool = True,
    verbose: bool = True,
):
    """Create a density-aware geometric sketch of the dataset for HPO.

    Workflow:
      1. Scale marker matrix (optional)
      2. Run PCA (scanpy / anndata backed) up to ``max_pcs``
      3. Apply geometric sketching on the PCA embedding to select ``sketch_size`` indices
      4. Return an AnnData object containing only the sketched cells with their PCA embedding

    Falls back to uniform random sampling if geosketch is unavailable.
    """
    import time
    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        raise ImportError("scanpy is required for sketch creation.")

    start = time.time()
    available = [m for m in marker_cols if m in data.columns]
    if not available:
        raise ValueError("No marker columns found for sketch creation.")
    mat = data[available].values
    if float32:
        mat = mat.astype('float32', copy=False)
    adata = ad.AnnData(mat)
    adata.var_names = available
    adata.obs_names = data.index.astype(str)
    if scale:
        sc.pp.scale(adata, max_value=10)
    # PCA (randomized usually faster on large n_obs)
    solver = 'randomized' if adata.n_obs > 20000 else 'arpack'
    sc.tl.pca(adata, n_comps=min(max_pcs, len(available), max(1, adata.n_obs - 1)), svd_solver=solver, random_state=random_seed)
    pca_emb = adata.obsm['X_pca']
    sketch_size = min(sketch_size, adata.n_obs)
    if sketch_size < adata.n_obs:
        if verbose:
            print(f"Applying geometric sketching to select {sketch_size} of {adata.n_obs} cells ...")
        from spacemap.ph_accel_utils import geometric_sketch

        sketch_indices = geometric_sketch(pca_emb, sketch_size, random_seed=random_seed)
    else:
        sketch_indices = np.arange(adata.n_obs)
    adata_sketch = adata[sketch_indices].copy()
    # Keep original index for traceability
    adata_sketch.obs['orig_index'] = adata.obs_names[sketch_indices]
    if verbose:
        print(f"Sketch created in {time.time()-start:.1f}s (cells: {adata_sketch.n_obs}).")
    return adata_sketch


def leiden_hpo(
    adata_sketch,
    pca_values,
    neighbor_values,
    resolution_values,
    random_seed: int = 42,
    metrics=('silhouette', 'davies_bouldin'),
    verbose: bool = True,
):
    """Perform systematic hyperparameter search over Leiden parameters on a sketched AnnData.

    Parameters
    ----------
    adata_sketch : AnnData
        Output of ``create_geometric_sketch`` (must already have full PCA embedding in ``.obsm['X_pca']``).
    pca_values, neighbor_values, resolution_values : iterable
        Search grids for n_pcs, n_neighbors, and resolution respectively.
    metrics : tuple
        Subset of ('silhouette','davies_bouldin') to compute.

    Returns
    -------
    pd.DataFrame with one row per parameter combination.
    """
    import scanpy as sc
    from copy import deepcopy
    results = []
    # Ensure unique & sorted for deterministic behavior
    pca_values = sorted(set(int(x) for x in pca_values))
    neighbor_values = sorted(set(int(x) for x in neighbor_values))
    resolution_values = sorted(set(float(x) for x in resolution_values))

    max_pca_available = adata_sketch.obsm['X_pca'].shape[1]
    if max(pca_values) > max_pca_available:
        raise ValueError(f"Requested max PCA components {max(pca_values)} exceeds available ({max_pca_available}). Increase max_pcs in create_geometric_sketch().")

    if verbose:
        total = len(pca_values) * len(neighbor_values) * len(resolution_values)
        print(f"Starting Leiden HPO search across {total} combinations ...")

    # To minimize recomputation, outer loops over n_pcs -> n_neighbors -> resolutions
    for n_pcs in pca_values:
        # Pre-slice embedding for metric evaluation
        pca_slice = adata_sketch.obsm['X_pca'][:, :n_pcs]
        for n_neighbors in neighbor_values:
            # Fresh copy for neighbors graph per (n_pcs, n_neighbors)
            base = adata_sketch.copy()
            sc.pp.neighbors(base, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=random_seed)
            for resolution in resolution_values:
                import time
                t0 = time.time()
                temp = base.copy()  # lightweight (views) but ensures isolation of cluster labels
                sc.tl.leiden(temp, resolution=resolution, random_state=random_seed, key_added='leiden_clusters')
                labels = temp.obs['leiden_clusters']
                n_clusters = labels.nunique()
                # Metric calculation
                if n_clusters > 1:
                    sil_val = silhouette_score(pca_slice, labels) if 'silhouette' in metrics else np.nan
                    db_val = davies_bouldin_score(pca_slice, labels) if 'davies_bouldin' in metrics else np.nan
                else:
                    sil_val = -1.0 if 'silhouette' in metrics else np.nan
                    db_val = float('inf') if 'davies_bouldin' in metrics else np.nan
                runtime = time.time() - t0
                results.append({
                    'PCA_n': n_pcs,
                    'n_neighbors': n_neighbors,
                    'Resolution': resolution,
                    'Num_Clusters': int(n_clusters),
                    'Silhouette_Score': float(sil_val),
                    'Davies_Bouldin_Index': float(db_val),
                    'runtime_seconds': round(runtime, 2)
                })
                if verbose:
                    print(f"  PCA={n_pcs:>2} k={n_neighbors:>2} res={resolution:>4}: clusters={n_clusters:>3} sil={sil_val:6.3f} db={db_val:6.3f} ({runtime:4.1f}s)")
    return pd.DataFrame(results)


def recommend_leiden_parameters(hpo_df: pd.DataFrame):
    """Recommend optimal Leiden parameters based on HPO results.

    Strategy: sort by Silhouette (desc) then Davies-Bouldin (asc), then moderate
    cluster counts (prefer >=3). Returns dict with best row info.
    """
    if hpo_df.empty:
        return {}
    df = hpo_df.copy()
    # Penalize pathological cluster counts (1 or extremely high > 5000) by adjusting metrics
    df['__sil_adj'] = df['Silhouette_Score']
    df.loc[df['Num_Clusters'] <= 1, '__sil_adj'] = -1
    df.loc[df['Num_Clusters'] > 5000, '__sil_adj'] = -1
    df_sorted = df.sort_values(['__sil_adj', 'Davies_Bouldin_Index'], ascending=[False, True])
    best = df_sorted.iloc[0].to_dict()
    best.pop('__sil_adj', None)
    return best


# =============================================================================
# THRESHOLD-BASED PHENOTYPING FUNCTIONS
# =============================================================================

def compute_batch_polygon_thresholds(polygon_dir, batch_ids, markers, stat='median'):
    """
    Compute polygon thresholds for each batch using specified statistic.
    
    Parameters
    ----------
    polygon_dir : Path
        Directory containing polygon threshold CSV files
    batch_ids : list
        List of batch identifiers to process
    markers : list
        List of marker names to extract thresholds for
    stat : str
        Statistic to compute ('median', 'mean', 'min')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with batch_id column and threshold values for each marker
    """
    import pandas as pd
    from pathlib import Path
    
    polygon_dir = Path(polygon_dir)
    results = []
    
    for batch_id in batch_ids:
        # Find matching threshold file using batch prefix
        batch_prefix = batch_id.split('.ome')[0] if '.ome' in batch_id else batch_id
        threshold_file = polygon_dir / f"{batch_prefix}.ome.tiff_-_Merged_Image.csv"
        
        if not threshold_file.exists():
            print(f"Warning: Threshold file not found for batch {batch_id}: {threshold_file}")
            continue
            
        try:
            # Load threshold data
            thr_df = pd.read_csv(threshold_file)
            
            # Verify Marker column exists
            if 'Marker' not in thr_df.columns:
                print(f"Warning: 'Marker' column not found in {threshold_file}")
                continue
            
            # Extract polygon columns (exclude 'Marker' column)
            polygon_cols = [col for col in thr_df.columns if col.startswith('Polygon_')]
            if not polygon_cols:
                print(f"Warning: No 'Polygon_' columns found in {threshold_file}")
                continue
            
            # Initialize result row
            result_row = {'batch_id': batch_id}
            
            # Compute statistic for each marker
            for marker in markers:
                marker_row = thr_df[thr_df['Marker'] == marker]
                if marker_row.empty:
                    print(f"Warning: Marker {marker} not found in {threshold_file}")
                    result_row[marker] = np.nan
                else:
                    # Extract polygon values and compute statistic
                    polygon_values = marker_row[polygon_cols].iloc[0].values
                    polygon_values = polygon_values[~pd.isna(polygon_values)]  # Remove NaNs
                    
                    if len(polygon_values) > 0:
                        if stat == 'median':
                            result_row[marker] = np.median(polygon_values)
                        elif stat == 'mean':
                            result_row[marker] = np.mean(polygon_values)
                        elif stat == 'min':
                            result_row[marker] = np.min(polygon_values)
                        else:
                            raise ValueError(f"Unknown statistic: {stat}")
                    else:
                        result_row[marker] = np.nan
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing {threshold_file}: {e}")
            continue
    
    return pd.DataFrame(results)


def apply_threshold_subtraction(df, thresholds_df, batch_col, markers, suffix='_thr', reduction_percent=0.0):
    """
    Apply threshold subtraction to raw markers, creating new thresholded columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with raw marker columns and batch identifiers
    thresholds_df : pd.DataFrame
        Threshold values per batch (from compute_batch_polygon_thresholds)
    batch_col : str
        Column name containing batch identifiers
    markers : list
        List of marker names to threshold
    suffix : str
        Suffix to add to thresholded marker columns
    reduction_percent : float
        Percentage to reduce thresholds (0-100). For example, 10.0 will reduce 
        thresholds by 10% (multiply by 0.9) to account for threshold estimation error.
        
    Returns
    -------
    tuple
        (df_with_thresholded_markers, list_of_thresholded_column_names)
    """
    import pandas as pd
    import numpy as np
    
    df_result = df.copy()
    thresholded_cols = []
    
    # Calculate reduction factor
    reduction_factor = 1.0 - (reduction_percent / 100.0)
    if reduction_percent > 0:
        print(f"Applying {reduction_percent}% threshold reduction (factor: {reduction_factor:.3f})")
    
    # Create mapping of batch_id to thresholds
    batch_thresholds = {}
    for _, row in thresholds_df.iterrows():
        batch_id = row[batch_col]
        batch_thresholds[batch_id] = {marker: row[marker] for marker in markers if marker in row}
    
    # Apply threshold subtraction for each marker
    for marker in markers:
        thr_col_name = f"{marker}{suffix}"
        thresholded_cols.append(thr_col_name)
        
        # Initialize thresholded column
        df_result[thr_col_name] = 0.0
        
        # Apply batch-specific thresholds
        for batch_id in df_result[batch_col].unique():
            batch_mask = df_result[batch_col] == batch_id
            
            if batch_id in batch_thresholds and marker in batch_thresholds[batch_id]:
                threshold = batch_thresholds[batch_id][marker]
                if not pd.isna(threshold):
                    # Apply reduction to threshold
                    adjusted_threshold = threshold * reduction_factor
                    # Subtract adjusted threshold and floor at 0
                    raw_values = df_result.loc[batch_mask, marker]
                    thresholded_values = raw_values - adjusted_threshold
                    thresholded_values = np.maximum(thresholded_values, 0)
                    df_result.loc[batch_mask, thr_col_name] = thresholded_values
                else:
                    print(f"Warning: NaN threshold for {marker} in batch {batch_id}")
            else:
                print(f"Warning: No threshold found for {marker} in batch {batch_id}")
    
    return df_result, thresholded_cols


def build_phenotype_boolean_matrix(df, phenotype_definitions, marker_suffix='_thr', min_positive=0.0):
    """
    Build boolean matrix for phenotype assignments based on marker definitions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with thresholded marker columns
    phenotype_definitions : dict
        Dict mapping phenotype names to list of marker lists (AND within, OR across)
    marker_suffix : str
        Suffix of marker columns to use
    min_positive : float
        Minimum value to consider a marker positive
        
    Returns
    -------
    pd.DataFrame
        Boolean matrix with phenotype columns
    """
    import pandas as pd
    import numpy as np
    
    bool_matrix = pd.DataFrame(index=df.index)
    
    for pheno, or_conditions in phenotype_definitions.items():
        col = []
        for idx in df.index:
            positive = False
            for and_set in or_conditions:
                ok = True
                for marker in and_set:
                    col_name = f"{marker}{marker_suffix}"
                    if col_name not in df.columns or df.at[idx, col_name] <= min_positive:
                        ok = False
                        break
                if ok:
                    positive = True
                    break
            col.append(1 if positive else 0)
        df[pheno] = col
        bool_matrix[pheno] = col
    return df, bool_matrix


def normalize_thresholded_markers(df, markers, batch_col, marker_suffix='_thr', method='z1'):
    """
    Apply normalization to thresholded markers for tie-breaking.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with thresholded marker columns
    markers : list
        List of base marker names
    batch_col : str
        Column for grouping normalization
    marker_suffix : str
        Suffix of thresholded marker columns
    method : str
        Normalization method ('z1' for Z-score)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized marker columns
    """
    import pandas as pd
    
    # Get thresholded marker columns
    thr_marker_cols = [f"{marker}{marker_suffix}" for marker in markers]
    available_cols = [col for col in thr_marker_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError("No thresholded marker columns found")
    
    # Apply normalization using existing function
    norm_df = apply_flexible_normalization(
        df=df,
        marker_cols=available_cols,
        group_cols=[batch_col],
        suffix='_norm',
        apply_z1=(method == 'z1'),
        apply_z2=False,
        apply_cdf=False,
        apply_nlog=False
    )
    
    return norm_df


def resolve_threshold_phenotype_conflicts(pheno_bool_df, df, markers, norm_df=None, 
                                        marker_suffix='_thr', phenotype_definitions=None):
    """
    Resolve phenotype conflicts using hierarchical rules and normalized tie-breaking.
    
    Parameters
    ----------
    pheno_bool_df : pd.DataFrame
        Boolean matrix from build_phenotype_boolean_matrix
    df : pd.DataFrame
        Original data with thresholded markers
    markers : list
        List of base marker names
    norm_df : pd.DataFrame, optional
        Normalized marker data for tie-breaking
    marker_suffix : str
        Suffix of thresholded marker columns
    phenotype_definitions : dict
        Phenotype definitions for subset detection
        
    Returns
    -------
    pd.Series
        Final phenotype assignments
    """
    import pandas as pd
    import numpy as np
    
    assignments = pd.Series(['unclassified'] * len(pheno_bool_df), index=pheno_bool_df.index)
    
    # Define subset relationships (hardcoded for now, could be inferred)
    subset_relationships = {
        'Treg': ['Helper T'],
        'Helper T': [],
        'Cytotoxic T': [],
        'Plasma B cell': ['B cell'],
        'B cell': []
    }
    
    for idx in pheno_bool_df.index:
        # Get all positive phenotypes for this cell
        positive_phenotypes = []
        for phenotype in pheno_bool_df.columns:
            if pheno_bool_df.loc[idx, phenotype]:
                positive_phenotypes.append(phenotype)
        
        if len(positive_phenotypes) == 0:
            assignments.loc[idx] = 'unclassified'
        elif len(positive_phenotypes) == 1:
            assignments.loc[idx] = positive_phenotypes[0]
        else:
            # Multiple phenotypes - need conflict resolution
            
            # 1. Check for subset relationships (subset wins)
            subset_winner = None
            for phenotype in positive_phenotypes:
                if phenotype in subset_relationships:
                    parents = subset_relationships[phenotype]
                    if all(parent in positive_phenotypes for parent in parents):
                        subset_winner = phenotype
                        break
            
            if subset_winner:
                assignments.loc[idx] = subset_winner
            else:
                # 2. Use normalized marker values for tie-breaking
                if norm_df is not None and phenotype_definitions is not None:
                    best_phenotype = None
                    best_score = -np.inf
                    
                    for phenotype in positive_phenotypes:
                        # Get unique markers for this phenotype (excluding shared ones)
                        phenotype_markers = set()
                        for marker_list in phenotype_definitions.get(phenotype, []):
                            phenotype_markers.update(marker_list)
                        
                        # Remove markers shared with other competing phenotypes
                        unique_markers = phenotype_markers.copy()
                        for other_phenotype in positive_phenotypes:
                            if other_phenotype != phenotype:
                                other_markers = set()
                                for marker_list in phenotype_definitions.get(other_phenotype, []):
                                    other_markers.update(marker_list)
                                unique_markers = unique_markers - other_markers
                        
                        # Calculate score based on unique normalized markers
                        if unique_markers:
                            scores = []
                            for marker in unique_markers:
                                # Use normalized columns if available, otherwise fall back to thresholded
                                norm_col = f"{marker}_thr_norm"
                                thr_col = f"{marker}{marker_suffix}"
                                
                                if norm_df is not None and norm_col in norm_df.columns:
                                    scores.append(norm_df.loc[idx, norm_col])
                                elif thr_col in df.columns:
                                    scores.append(df.loc[idx, thr_col])
                            
                            if scores:
                                phenotype_score = max(scores)  # Use highest unique marker
                                if phenotype_score > best_score:
                                    best_score = phenotype_score
                                    best_phenotype = phenotype
                    
                    if best_phenotype:
                        assignments.loc[idx] = best_phenotype
                    else:
                        # Fallback to first phenotype if tie-breaking fails
                        assignments.loc[idx] = positive_phenotypes[0]
                else:
                    # No normalization data - use first phenotype
                    assignments.loc[idx] = positive_phenotypes[0]
    
    return assignments


def generate_threshold_heatmap(df, assignment_col, markers, marker_suffix='_thr', 
                             figsize=(12, 8), save_path=None):
    """
    Generate phenotype × marker heatmap for threshold-based results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with phenotype assignments and marker values
    assignment_col : str
        Column containing phenotype assignments
    markers : list
        List of marker names
    marker_suffix : str
        Suffix to add to marker columns
    figsize : tuple
        Figure size
    save_path : str or Path
        Path to save the figure
        
    Returns
    -------
    matplotlib.axes.Axes
        Heatmap axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Get marker columns
    marker_cols = [f"{marker}{marker_suffix}" for marker in markers]
    available_cols = [col for col in marker_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError("No marker columns found")
    
    # Compute mean expression per phenotype
    phenotype_means = df.groupby(assignment_col)[available_cols].mean()
    
    # Clean up column names for display
    display_cols = [col.replace(marker_suffix, '') for col in available_cols]
    phenotype_means.columns = display_cols
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn for better-looking heatmap (matching Leiden style)
    sns.heatmap(
        phenotype_means.T,  # Transpose for markers on y-axis
        annot=True,
        fmt='.2f',
        cmap='bwr',  # Blue-white-red colormap (matches Leiden)
        center=0,
        vmin=-2,
        vmax=2,
        ax=ax,
        cbar_kws={'label': 'Mean Expression'}
    )
    
    ax.set_xlabel('Phenotypes')
    ax.set_ylabel('Markers')
    ax.set_title('Phenotype × Marker Expression Profile')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax


# =============================================================================
# SCIMAP-STYLE PHENOTYPING FUNCTIONS (from SCIMAP_v1_working.py reference)
# =============================================================================

def find_roi_threshold_file(batch_id, threshold_dir):
    """
    Find corresponding threshold file for a given batch/ROI ID.
    Based on the reference SCIMAP implementation naming pattern.
    """
    import os
    from pathlib import Path
    
    threshold_dir = Path(threshold_dir)
    
    # Extract prefix from batch_id (remove .ome_cell_features_16bit_annotated.csv)
    if batch_id.endswith('.ome_cell_features_16bit_annotated.csv'):
        prefix = batch_id.replace('.ome_cell_features_16bit_annotated.csv', '')
    else:
        prefix = str(batch_id)
    
    # Look for threshold file with pattern: prefix.ome.tiff_-_Merged_Image.csv
    threshold_filename = f"{prefix}.ome.tiff_-_Merged_Image.csv"
    threshold_path = threshold_dir / threshold_filename
    
    if threshold_path.exists():
        return threshold_path
    
    # Alternative patterns
    alt_patterns = [
        f"{prefix}_thresholds.csv",
        f"{prefix}.csv",
        f"{prefix}_Merged_Image.csv"
    ]
    
    for pattern in alt_patterns:
        alt_path = threshold_dir / pattern
        if alt_path.exists():
            return alt_path
    
    return None


def create_phenotype_strategy_csv(output_path):
    """
    Create default SCIMAP phenotype strategy CSV file from reference implementation.
    
    Parameters
    - output_path: Path where to save the CSV file
    """
    phenotype_strategy = pd.DataFrame({
        'phenotype': [
            'T_cells', 'CD4_T_cells', 'CD8_T_cells', 'Treg', 'B_cells', 
            'Macrophages', 'Dendritic_cells', 'Epithelial_cells', 
            'Endothelial_cells', 'Stromal_cells'
        ],
        'pos': [
            'CD45,CD3e', 'CD45,CD3e,CD4', 'CD45,CD3e,CD8', 'CD45,CD3e,CD4,FoxP3',
            'CD45,CD20', 'CD45,CD68', 'CD45,CD11c,HLA-DR', 'Pan-Cytokeratin',
            'CD31', 'Vimentin'
        ],
        'neg': [
            '', '', '', '', '', '', '', 'CD45', 'CD45', 'CD45,Pan-Cytokeratin'
        ]
    })
    
    phenotype_strategy.to_csv(output_path, index=False)
    print(f"Phenotype strategy file created: {output_path}")
    
    return phenotype_strategy


def load_and_process_thresholds_scimap(threshold_dir, batch_ids, markers):
    """
    Load and process threshold files for SCIMAP analysis.
    Based on the reference SCIMAP implementation.
    
    Parameters
    ----------
    threshold_dir : Path
        Directory containing threshold CSV files
    batch_ids : list
        List of batch/ROI identifiers
    markers : list
        List of marker names
        
    Returns
    -------
    dict
        Dictionary mapping batch_id to median thresholds
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    threshold_dir = Path(threshold_dir)
    batch_thresholds = {}
    
    print(f"Processing thresholds for {len(batch_ids)} batches...")
    
    for batch_id in batch_ids:
        threshold_file = find_roi_threshold_file(batch_id, threshold_dir)
        
        if threshold_file is None:
            print(f"  Warning: No threshold file found for batch {batch_id}")
            continue
            
        try:
            threshold_df = pd.read_csv(threshold_file)
            
            # Find polygon columns
            polygon_cols = [col for col in threshold_df.columns if col.startswith('Polygon_')]
            if not polygon_cols:
                print(f"  Warning: No polygon columns found in {threshold_file}")
                continue
            
            # Get marker column (first column)
            marker_col = threshold_df.columns[0]
            threshold_df = threshold_df.rename(columns={marker_col: 'Marker'})
            
            # Filter to available markers
            threshold_df = threshold_df[threshold_df['Marker'].isin(markers)].copy()
            
            # Calculate median threshold per marker
            threshold_df['median_threshold'] = threshold_df[polygon_cols].median(axis=1)
            
            # Create marker -> threshold mapping
            thresholds = threshold_df.set_index('Marker')['median_threshold'].to_dict()
            batch_thresholds[batch_id] = thresholds
            
            print(f"  Processed {len(thresholds)} thresholds for batch: {batch_id}")
            
        except Exception as e:
            print(f"Error processing threshold file {threshold_file}: {e}")
            continue
    
    return batch_thresholds


def apply_scimap_phenotyping(df, phenotype_strategy_df, gate_value=0.5):
    """
    Apply SCIMAP-style phenotyping based on strategy DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with rescaled marker columns
    phenotype_strategy_df : pd.DataFrame
        Phenotype strategy with columns: phenotype, marker, gate, strategy
    gate_value : float
        Default gate value for positive/negative calls
        
    Returns
    -------
    pd.DataFrame
        DataFrame with phenotype assignments added
    """
    import pandas as pd
    import numpy as np
    
    df_result = df.copy()
    
    # Get unique phenotypes
    phenotypes = phenotype_strategy_df['phenotype'].unique()
    
    # Initialize phenotype scores
    for phenotype in phenotypes:
        df_result[f'{phenotype}_score'] = 0.0
        df_result[f'{phenotype}_positive'] = False
    
    # Apply phenotype strategy
    for phenotype in phenotypes:
        phenotype_rules = phenotype_strategy_df[phenotype_strategy_df['phenotype'] == phenotype]
        
        # Calculate phenotype score for each cell
        scores = []
        positives = []
        
        for _, rule in phenotype_rules.iterrows():
            marker = rule['marker']
            gate = rule.get('gate', gate_value)
            strategy = rule.get('strategy', 'pos')
            
            if marker not in df_result.columns:
                print(f"  Warning: Marker {marker} not found for phenotype {phenotype}")
                continue
            
            marker_values = df_result[marker]
            
            if strategy == 'pos':
                # Positive gate
                marker_score = (marker_values > gate).astype(float)
            else:
                # Negative gate
                marker_score = (marker_values <= gate).astype(float)
            
            scores.append(marker_score)
        
        if scores:
            # Calculate overall phenotype score (all markers must be positive)
            combined_score = np.column_stack(scores).min(axis=1)
            df_result[f'{phenotype}_score'] = combined_score
            df_result[f'{phenotype}_positive'] = combined_score > 0
    
    # Assign primary phenotype (highest scoring)
    phenotype_cols = [f'{phenotype}_score' for phenotype in phenotypes]
    available_cols = [col for col in phenotype_cols if col in df_result.columns]
    
    if available_cols:
        # Get highest scoring phenotype for each cell
        max_scores = df_result[available_cols].max(axis=1)
        primary_phenotypes = df_result[available_cols].idxmax(axis=1)
        
        # Only assign if score > 0
        primary_phenotypes = primary_phenotypes.where(max_scores > 0, 'Unknown')
        primary_phenotypes = primary_phenotypes.str.replace('_score', '')
        
        df_result['phenotype_scimap'] = primary_phenotypes
    else:
        df_result['phenotype_scimap'] = 'Unknown'
    
    return df_result


def create_scimap_heatmap(df, assignment_col='phenotype_scimap', markers=None, 
                         marker_suffix='', figsize=(12, 8), save_path=None):
    """
    Create phenotype × marker heatmap for SCIMAP results matching Leiden style.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with phenotype assignments and marker columns
    assignment_col : str
        Column containing phenotype assignments
    markers : list
        List of marker names
    marker_suffix : str
        Suffix to add to marker columns (e.g., '_rescaled')
    figsize : tuple
        Figure size
    save_path : str or Path
        Path to save the figure
        
    Returns
    -------
    matplotlib.axes.Axes
        Heatmap axes object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Get marker columns
    if markers is None:
        marker_cols = [col for col in df.columns if col.endswith(marker_suffix)]
    else:
        marker_cols = [f"{marker}{marker_suffix}" for marker in markers]
        
    available_cols = [col for col in marker_cols if col in df.columns]
    
    if not available_cols:
        raise ValueError("No marker columns found")
    
    # Compute mean expression per phenotype
    phenotype_means = df.groupby(assignment_col)[available_cols].mean()
    
    # Clean up column names for display
    display_cols = [col.replace(marker_suffix, '') for col in available_cols]
    phenotype_means.columns = display_cols
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn for better-looking heatmap (matching Leiden style)
    sns.heatmap(
        phenotype_means.T,  # Transpose for markers on y-axis
        annot=True,
        fmt='.2f',
        cmap='bwr',  # Blue-white-red colormap (matches Leiden)
        center=0,
        vmin=-2,
        vmax=2,
        ax=ax,
        cbar_kws={'label': 'Mean Expression'}
    )
    
    ax.set_xlabel('Phenotypes')
    ax.set_ylabel('Markers')
    ax.set_title('Phenotype × Marker Expression Profile (SCIMAP)')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return ax


def _infer_threshold_base(filename_str):
    """
    Infers the base name for a threshold file from a batch/ROI identifier.
    e.g., 'P1_P3_roi_1.ome_cell_features_16bit_annotated.csv' -> 'P1_P3_roi_1'
    """
    import re
    # This regex will match the common prefixes like PXX_PXX_roi_X etc.
    match = re.match(r'([a-zA-Z0-9_]+)\.ome_cell_features', filename_str)
    if match:
        return match.group(1)
    # Fallback for other naming conventions
    return filename_str.split('.')[0]


# ============================================================================
# REFACTORED SOM PHENOTYPING HELPERS (for Step6B & future Step6A parity)
# ============================================================================

def prepare_som_input(df, clustering_markers_corr, normalization_flags, group_cols, subsample_for_plots=100000):
    """Prepare data & (optionally) normalized markers for SOM clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Input corrected dataframe.
    clustering_markers_corr : list[str]
        List of corrected marker column names (e.g. CD3e_corr).
    normalization_flags : dict
        Keys: z1, z2, cdf, nlog (booleans) controlling apply_flexible_normalization.
    group_cols : list[str]
        Grouping columns for normalization.
    subsample_for_plots : int
        Max number of cells to keep for downstream quick plotting convenience.

    Returns
    -------
    dict with:
        data (pd.DataFrame) possibly extended with normalized cols
        clustering_data_scaled (np.ndarray)
        available_markers (list) (the actual columns used for clustering)
        scaler (StandardScaler)
    """
    from sklearn.preprocessing import StandardScaler

    available_markers = [m for m in clustering_markers_corr if m in df.columns]
    if not available_markers:
        raise ValueError("No clustering markers available in dataframe.")

    # Apply flexible normalization if any flag is True
    any_norm = any(normalization_flags.values())
    data = df.copy()
    if any_norm:
        norm_df = apply_flexible_normalization(
            df=data,
            marker_cols=available_markers,
            group_cols=group_cols,
            suffix='_norm',
            apply_z1=normalization_flags.get('z1', True),
            apply_z2=normalization_flags.get('z2', False),
            apply_cdf=normalization_flags.get('cdf', False),
            apply_nlog=normalization_flags.get('nlog', False)
        )
        norm_cols = [c for c in norm_df.columns if c in norm_df]
        data = pd.concat([data, norm_df], axis=1)
        # Use normalized markers preferentially
        normalized_markers = [f"{m}_norm" for m in available_markers]
        usable_norm = [m for m in normalized_markers if m in data.columns]
        if usable_norm:
            available_markers = usable_norm

    clustering_data = data[available_markers].fillna(0).astype(float)
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    return {
        'data': data,
        'clustering_data_scaled': clustering_data_scaled,
        'available_markers': available_markers,
        'scaler': scaler
    }


def train_som(clustering_data_scaled, x_dim, y_dim, iterations, sigma=1.5, lr=0.05, subsample_limit=100000, random_seed=42, early_stop_patience=5):
    """Train a MiniSom (with early stopping) and return assignments & QE history."""
    if not MINISOM_AVAILABLE or MiniSom is None:
        raise ImportError("MiniSom not available. Install minisom or provide fallback.")

    data_for_training = clustering_data_scaled
    if len(clustering_data_scaled) > subsample_limit:
        idx = np.random.choice(len(clustering_data_scaled), subsample_limit, replace=False)
        data_for_training = clustering_data_scaled[idx]

    som = MiniSom(
        x=x_dim,
        y=y_dim,
        input_len=clustering_data_scaled.shape[1],
        sigma=sigma,
        learning_rate=lr,
        random_seed=random_seed
    )
    som.random_weights_init(data_for_training)

    quantization_errors = []
    for i in range(iterations):
        som.train_random(data_for_training, num_iteration=1, verbose=False)
        if i % 5 == 0 or i == iterations - 1:
            qe = som.quantization_error(data_for_training)
            quantization_errors.append(qe)
        if len(quantization_errors) > early_stop_patience:
            recent = np.diff(quantization_errors[-early_stop_patience:])
            if np.mean(recent) > -0.001:
                break

    # Assign ALL cells to clusters
    winner_coords = np.array([som.winner(v) for v in clustering_data_scaled]).T
    bmu_indices = np.ravel_multi_index(winner_coords, (x_dim, y_dim))
    return {
        'som': som,
        'som_clusters': bmu_indices,
        'quantization_errors': quantization_errors,
        'grid_size': (x_dim, y_dim)
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


def meta_cluster_clusters(cluster_char_df, target_n=100, method='ward', metric='euclidean', marker_suffix='_mean'):
    """Reduce SOM clusters to meta-clusters via hierarchical clustering."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    if len(cluster_char_df) <= target_n:
        cluster_char_df['meta_cluster'] = cluster_char_df['som_cluster']
        meta_map = dict(zip(cluster_char_df['som_cluster'], cluster_char_df['meta_cluster']))
        meta_df = cluster_char_df.copy()
        return meta_df, meta_map

    marker_cols = [c for c in cluster_char_df.columns if c.endswith(marker_suffix)]
    feats = cluster_char_df[marker_cols].fillna(0)
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    model = AgglomerativeClustering(n_clusters=target_n, linkage=method, metric=metric)
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
        for col in marker_cols:
            rec[col] = sub[col].mean()
        meta_list.append(rec)
    meta_df = pd.DataFrame(meta_list).sort_values('cell_count', ascending=False)
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
    g = sns.clustermap(heat_z, method='ward', metric='euclidean', cmap='bwr', vmin=-2, vmax=2,
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
    g.fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    # Return ordered cluster IDs (meta-cluster IDs) in displayed order
    order_ids = [int(t.get_text().split('\n')[0]) for t in g.ax_heatmap.get_xticklabels()]
    return order_ids, out_path


def launch_som_phenotype_assignment_app(char_df, reordered_cluster_ids, meta_cluster_phenotypes):
    """Launch the professional phenotype assignment application."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import seaborn as sns
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
                self.assignments.append({
                    'clusters': clusters, 'phenotype': phenotype, 'cell_count': total_cells
                })

        def setup_gui(self):
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=self.colors['primary'])
            style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), foreground=self.colors['secondary'])
            style.configure('Stats.TLabel', font=('Segoe UI', 11), foreground=self.colors['success'])
            style.configure('Card.TFrame', background=self.colors['card'], relief='raised', borderwidth=1)
            
            main_container = ttk.Frame(self.root, style='Card.TFrame')
            main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
            
            header = ttk.Frame(main_container)
            header.pack(fill=tk.X, pady=(0, 15))
            
            title_label = ttk.Label(header, text="Phenotype Assignment Studio", style='Title.TLabel')
            title_label.pack(side=tk.LEFT)
            
            self.stats_frame = ttk.Frame(header)
            self.stats_frame.pack(side=tk.RIGHT)
            
            self.paned_window = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
            self.paned_window.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
            
            self.left_panel = ttk.LabelFrame(self.paned_window, text="Interactive Cluster Heatmap", style='Card.TFrame')
            self.paned_window.add(self.left_panel, weight=3)
            
            self.right_panel = ttk.LabelFrame(self.paned_window, text="Assignment Control Panel", style='Card.TFrame')
            self.right_panel.configure(padding="10")
            self.paned_window.add(self.right_panel, weight=1)
            
            self.setup_heatmap_panel()
            self.setup_control_panel()

        def setup_heatmap_panel(self):
            controls_frame = ttk.Frame(self.left_panel)
            controls_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.cluster_x = tk.BooleanVar(value=True)
            self.cluster_y = tk.BooleanVar(value=True)
            
            ttk.Checkbutton(controls_frame, text="Cluster X-axis", variable=self.cluster_x).pack(side=tk.LEFT, padx=(0, 15))
            ttk.Checkbutton(controls_frame, text="Cluster Y-axis", variable=self.cluster_y).pack(side=tk.LEFT, padx=(0, 15))
            ttk.Button(controls_frame, text="Update Heatmap", command=self.plot_heatmap).pack(side=tk.LEFT, padx=(15, 0))
            
            legend_frame = ttk.Frame(controls_frame)
            legend_frame.pack(side=tk.RIGHT)
            
            ttk.Label(legend_frame, text="Selected", foreground='#DAA520', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            ttk.Label(legend_frame, text="Assigned", foreground='#DC143C', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            ttk.Label(legend_frame, text="Unassigned", foreground='#228B22', font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            
            self.heatmap_fig = Figure(figsize=(16, 12), facecolor='white')
            self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, self.left_panel)
            self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        def setup_control_panel(self):
            selection_frame = ttk.LabelFrame(self.right_panel, text="Current Selection")
            selection_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.selected_label = ttk.Label(selection_frame, text="No clusters selected", 
                                           foreground=self.colors['secondary'], font=('Arial', 10, 'bold'))
            self.selected_label.pack(pady=5)
            
            assignment_frame = ttk.LabelFrame(self.right_panel, text="Phenotype Assignment")
            assignment_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(assignment_frame, text="Phenotype Name:").pack(anchor=tk.W, pady=(5, 0))
            self.phenotype_var = tk.StringVar()
            self.phenotype_entry = ttk.Entry(assignment_frame, textvariable=self.phenotype_var, font=('Arial', 11))
            self.phenotype_entry.pack(fill=tk.X, pady=(0, 5))
            
            buttons_frame = ttk.Frame(assignment_frame)
            buttons_frame.pack(fill=tk.X, pady=5)
            
            self.assign_btn = ttk.Button(buttons_frame, text="Add Assignment", command=self.add_assignment)
            self.assign_btn.pack(side=tk.LEFT, padx=(0, 5))
            
            self.update_btn = ttk.Button(buttons_frame, text="Update", command=self.update_assignment, state='disabled')
            self.update_btn.pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Button(buttons_frame, text="Clear", command=self.clear_selection).pack(side=tk.RIGHT)
            
            quick_frame = ttk.Frame(assignment_frame)
            quick_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(quick_frame, text="Select Unassigned", command=self.select_unassigned).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(quick_frame, text="Generate Heatmap", command=self.show_phenotype_heatmap).pack(side=tk.RIGHT)
            
            table_frame = ttk.LabelFrame(self.right_panel, text="Assignment Table")
            table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            table_container = ttk.Frame(table_frame)
            table_container.pack(fill=tk.BOTH, expand=True, pady=5)
            
            columns = ('ID', 'Phenotype', 'Clusters', 'Cells')
            self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=12)
            
            self.tree.heading('ID', text='#')
            self.tree.heading('Phenotype', text='Phenotype')
            self.tree.heading('Clusters', text='Count')
            self.tree.heading('Cells', text='Cells')
            
            self.tree.column('ID', width=30, anchor='center')
            self.tree.column('Phenotype', width=120, anchor='w')
            self.tree.column('Clusters', width=60, anchor='center')
            self.tree.column('Cells', width=80, anchor='center')
            
            scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
            self.tree.configure(yscrollcommand=scrollbar.set)
            
            self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.tree.bind('<Button-1>', self.on_assignment_click)
            self.tree.bind('<Double-1>', self.on_assignment_double_click)
            
            table_controls = ttk.Frame(table_frame)
            table_controls.pack(fill=tk.X, pady=(0, 5))
            
            ttk.Button(table_controls, text="Delete Selected", command=self.delete_assignment).pack(side=tk.LEFT)
            ttk.Button(table_controls, text="Save & Close", command=self.save_and_close).pack(side=tk.RIGHT)
            
            self.update_assignment_table()

        def update_statistics(self):
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            assigned_count = len(self.assigned_clusters)
            unassigned_count = len(self.unassigned_clusters)
            total_cells_assigned = sum(self.char_df[self.char_df['som_cluster'].isin(self.assigned_clusters)]['cell_count'])
            total_cells = self.char_df['cell_count'].sum()
            
            stats_text = f"Total Clusters: {self.total_clusters} | Assigned: {assigned_count} | Unassigned: {unassigned_count}"
            stats_label = ttk.Label(self.stats_frame, text=stats_text, style='Stats.TLabel')
            stats_label.pack()
            
            cells_text = f"Cells Assigned: {total_cells_assigned:,}/{total_cells:,} ({100*total_cells_assigned/total_cells:.1f}%)"
            cells_label = ttk.Label(self.stats_frame, text=cells_text, style='Stats.TLabel')
            cells_label.pack()

        def plot_heatmap(self):
            self.heatmap_fig.clear()
            
            marker_cols = [c for c in self.char_df.columns if c.endswith('_mean')]
            heatmap_data = self.char_df.set_index('som_cluster')[marker_cols].copy()
            
            clean = {col: col.replace('_corr_norm_mean','').replace('_corr_mean','').replace('_mean','') 
                     for col in marker_cols}
            heatmap_data.rename(columns=clean, inplace=True)
            
            heatmap_data = heatmap_data.T
            heatmap_z = heatmap_data.apply(lambda x: (x - x.mean())/x.std(), axis=1).fillna(0)
            
            if self.cluster_x.get() or self.cluster_y.get():
                data_for_clustering = heatmap_z.copy()
                
                if self.cluster_x.get():
                    col_linkage = linkage(data_for_clustering.T, method='ward', metric='euclidean')
                    col_dendrogram = dendrogram(col_linkage, no_plot=True)
                    col_order = col_dendrogram['leaves']
                    self.current_cluster_order = [heatmap_z.columns[i] for i in col_order]
                else:
                    self.current_cluster_order = sorted(heatmap_z.columns)
                
                if self.cluster_y.get():
                    row_linkage = linkage(data_for_clustering, method='ward', metric='euclidean')
                    row_dendrogram = dendrogram(row_linkage, no_plot=True)
                    row_order = row_dendrogram['leaves']
                    marker_order = [heatmap_z.index[i] for i in row_order]
                else:
                    marker_order = list(heatmap_z.index)
                
                heatmap_z_ordered = heatmap_z.loc[marker_order, self.current_cluster_order]
            else:
                self.current_cluster_order = sorted(heatmap_z.columns)
                heatmap_z_ordered = heatmap_z[self.current_cluster_order]
            
            ax = self.heatmap_fig.add_subplot(111)
            im = ax.imshow(heatmap_z_ordered.values, cmap='bwr', aspect='auto', vmin=-2, vmax=2)
            
            for i, cluster_id in enumerate(self.current_cluster_order):
                if cluster_id in self.selected_clusters:
                    ax.axvline(x=i-0.5, color='#FFD700', linewidth=4, alpha=0.9)
                    ax.axvline(x=i+0.5, color='#FFD700', linewidth=4, alpha=0.9)
                elif cluster_id in self.assigned_clusters:
                    ax.axvline(x=i-0.5, color='#FF4444', linewidth=3, alpha=0.7)
                    ax.axvline(x=i+0.5, color='#FF4444', linewidth=3, alpha=0.7)
                elif cluster_id in self.unassigned_clusters:
                    ax.axvline(x=i-0.5, color='#44AA44', linewidth=2, alpha=0.6)
                    ax.axvline(x=i+0.5, color='#44AA44', linewidth=2, alpha=0.6)
            
            ax.set_title('Interactive Phenotype Assignment - Click Clusters to Select\\n' +
                        'Yellow: Selected | Red: Assigned | Green: Unassigned', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Cluster Numbers', fontsize=12, fontweight='bold')
            ax.set_ylabel('Markers', fontsize=12, fontweight='bold')
            
            ax.set_xticks(range(len(self.current_cluster_order)))
            tick_labels = [str(int(cluster_id)) for cluster_id in self.current_cluster_order]
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=9, ha='center')
            
            ax.set_yticks(range(len(heatmap_z_ordered.index)))
            ax.set_yticklabels(heatmap_z_ordered.index, fontsize=9)
            
            cbar = self.heatmap_fig.colorbar(im, ax=ax, shrink=0.9, pad=0.005, aspect=40, fraction=0.03)
            cbar.set_label('Z-score', fontsize=11)
            
            self.heatmap_fig.tight_layout(pad=0.5, rect=[0.02, 0.05, 0.98, 0.95])
            
            def on_click(event):
                if event.inaxes == ax and event.xdata is not None:
                    cluster_idx = int(round(event.xdata))
                    if 0 <= cluster_idx < len(self.current_cluster_order):
                        cluster_id = self.current_cluster_order[cluster_idx]
                        
                        if cluster_id in self.assigned_clusters and cluster_id not in self.selected_clusters:
                            assigned_phenotype = self.meta_cluster_phenotypes.get(cluster_id, "Unknown")
                            self.highlight_assignment_by_cluster(cluster_id)
                            messagebox.showinfo("Cluster Already Assigned", 
                                              f"Cluster {cluster_id} is already assigned to: '{assigned_phenotype}'\\n\\n" +
                                              "The assignment has been highlighted in the table.\\n" +
                                              "Double-click the assignment to edit it.")
                            return
                        
                        if cluster_id in self.selected_clusters:
                            self.selected_clusters.remove(cluster_id)
                        else:
                            self.selected_clusters.append(cluster_id)
                        
                        self.update_selected_display()
                        self.plot_heatmap()
            
            self.heatmap_canvas.mpl_connect('button_press_event', on_click)
            self.heatmap_canvas.draw()

        def update_selected_display(self):
            if self.selected_clusters:
                if self.editing_assignment is not None:
                    text = f"EDITING: {self.selected_clusters} (Click Update to save)"
                    color = self.colors['warning']
                else:
                    text = f"Selected: {self.selected_clusters}"
                    color = self.colors['secondary']
                self.selected_label.config(text=text, foreground=color)
            else:
                self.selected_label.config(text="No clusters selected", foreground='gray')

        def clear_selection(self):
            self.selected_clusters.clear()
            self.editing_assignment = None
            self.assign_btn.config(state='normal')
            self.update_btn.config(state='disabled')
            self.phenotype_var.set("")
            self.update_selected_display()
            self.plot_heatmap()
            for item in self.tree.selection():
                self.tree.selection_remove(item)

        def select_unassigned(self):
            self.selected_clusters = list(self.unassigned_clusters)
            self.update_selected_display()
            self.plot_heatmap()

        def add_assignment(self):
            if not self.selected_clusters:
                messagebox.showwarning("Warning", "Please select clusters first")
                return
            
            phenotype = self.phenotype_var.get().strip()
            if not phenotype:
                messagebox.showwarning("Warning", "Please enter a phenotype name")
                return
            
            conflicts = [c for c in self.selected_clusters if c in self.assigned_clusters]
            if conflicts:
                if not messagebox.askyesno("Cluster Conflict", 
                                         f"Clusters {conflicts} are already assigned.\\n\\n" +
                                         "Do you want to reassign them?"):
                    return
                self.remove_clusters_from_assignments(conflicts)
            
            total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(self.selected_clusters)]['cell_count'])
            
            assignment = {
                'clusters': list(self.selected_clusters),
                'phenotype': phenotype,
                'cell_count': total_cells
            }
            self.assignments.append(assignment)
            
            for cluster_id in self.selected_clusters:
                self.meta_cluster_phenotypes[cluster_id] = phenotype
            
            self.assigned_clusters.update(self.selected_clusters)
            self.unassigned_clusters -= set(self.selected_clusters)
            
            self.clear_selection()
            self.update_assignment_table()
            self.update_statistics()

        def update_assignment(self):
            if self.editing_assignment is None:
                return
            
            phenotype = self.phenotype_var.get().strip()
            if not phenotype:
                messagebox.showwarning("Warning", "Please enter a phenotype name")
                return
            
            old_assignment = self.assignments[self.editing_assignment]
            old_clusters = set(old_assignment['clusters'])
            new_clusters = set(self.selected_clusters)
            
            self.assigned_clusters -= old_clusters
            self.unassigned_clusters |= old_clusters
            
            for cluster_id in old_clusters:
                if cluster_id in self.meta_cluster_phenotypes:
                    del self.meta_cluster_phenotypes[cluster_id]
            
            conflicts = [c for c in new_clusters if c in self.assigned_clusters]
            if conflicts:
                if not messagebox.askyesno("Cluster Conflict", 
                                         f"Clusters {conflicts} are already assigned.\\n\\n" +
                                         "Do you want to reassign them?"):
                    return
                self.remove_clusters_from_assignments(conflicts)
            
            total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(self.selected_clusters)]['cell_count'])
            self.assignments[self.editing_assignment] = {
                'clusters': list(self.selected_clusters),
                'phenotype': phenotype,
                'cell_count': total_cells
            }
            
            for cluster_id in self.selected_clusters:
                self.meta_cluster_phenotypes[cluster_id] = phenotype
            
            self.assigned_clusters.update(self.selected_clusters)
            self.unassigned_clusters -= set(self.selected_clusters)
            
            self.clear_selection()
            self.update_assignment_table()
            self.update_statistics()

        def remove_clusters_from_assignments(self, clusters_to_remove):
            for i, assignment in enumerate(self.assignments):
                old_clusters = set(assignment['clusters'])
                new_clusters = old_clusters - set(clusters_to_remove)
                
                if new_clusters != old_clusters:
                    if new_clusters:
                        total_cells = sum(self.char_df[self.char_df['som_cluster'].isin(new_clusters)]['cell_count'])
                        self.assignments[i] = {
                            'clusters': list(new_clusters),
                            'phenotype': assignment['phenotype'],
                            'cell_count': total_cells
                        }
                    else:
                        self.assignments.pop(i)
                        break
            
            for cluster_id in clusters_to_remove:
                if cluster_id in self.meta_cluster_phenotypes:
                    del self.meta_cluster_phenotypes[cluster_id]
            
            self.assigned_clusters -= set(clusters_to_remove)
            self.unassigned_clusters |= set(clusters_to_remove)

        def update_assignment_table(self):
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for i, assignment in enumerate(self.assignments):
                values = (
                    i + 1,
                    assignment['phenotype'][:15] + ('...' if len(assignment['phenotype']) > 15 else ''),
                    len(assignment['clusters']),
                    f"{assignment['cell_count']:,}"
                )
                self.tree.insert('', tk.END, values=values)

        def on_assignment_click(self, event):
            item = self.tree.selection()[0] if self.tree.selection() else None
            if item:
                values = self.tree.item(item)['values']
                assignment_idx = int(values[0]) - 1
                assignment = self.assignments[assignment_idx]
                
                self.selected_clusters = list(assignment['clusters'])
                self.update_selected_display()
                self.plot_heatmap()

        def on_assignment_double_click(self, event):
            item = self.tree.selection()[0] if self.tree.selection() else None
            if item:
                values = self.tree.item(item)['values']
                assignment_idx = int(values[0]) - 1
                assignment = self.assignments[assignment_idx]
                
                self.editing_assignment = assignment_idx
                self.selected_clusters = list(assignment['clusters'])
                self.phenotype_var.set(assignment['phenotype'])
                
                self.assign_btn.config(state='disabled')
                self.update_btn.config(state='normal')
                
                self.update_selected_display()
                self.plot_heatmap()

        def highlight_assignment_by_cluster(self, cluster_id):
            for i, assignment in enumerate(self.assignments):
                if cluster_id in assignment['clusters']:
                    children = self.tree.get_children()
                    if i < len(children):
                        self.tree.selection_set(children[i])
                        self.tree.focus(children[i])
                        self.tree.see(children[i])
                    break

        def delete_assignment(self):
            selected_items = self.tree.selection()
            if not selected_items:
                messagebox.showwarning("Warning", "Please select an assignment to delete")
                return
            
            item = selected_items[0]
            values = self.tree.item(item)['values']
            assignment_idx = int(values[0]) - 1
            assignment = self.assignments[assignment_idx]
            
            if messagebox.askyesno("Confirm Delete", 
                                  f"Delete assignment '{assignment['phenotype']}'?\\n\\n" +
                                  f"This will unassign {len(assignment['clusters'])} clusters."):
                
                for cluster_id in assignment['clusters']:
                    if cluster_id in self.meta_cluster_phenotypes:
                        del self.meta_cluster_phenotypes[cluster_id]
                
                self.assigned_clusters -= set(assignment['clusters'])
                self.unassigned_clusters |= set(assignment['clusters'])
                
                self.assignments.pop(assignment_idx)
                
                self.clear_selection()
                self.update_assignment_table()
                self.update_statistics()
                self.plot_heatmap()

        def show_phenotype_heatmap(self):
            if not self.meta_cluster_phenotypes:
                messagebox.showinfo("Info", "No phenotype assignments to display")
                return
            
            heatmap_window = tk.Toplevel(self.root)
            heatmap_window.title("Phenotype Expression Heatmap")
            heatmap_window.geometry("1200x800")
            
            pheno_df = self.char_df.copy()
            pheno_df['phenotype'] = pheno_df['som_cluster'].map(self.meta_cluster_phenotypes).fillna('Unassigned')
            
            marker_cols = [col for col in pheno_df.columns if col.endswith('_mean')]
            pheno_means = pheno_df.groupby('phenotype')[marker_cols].mean()
            
            clean_names = {col: col.replace('_corr_norm_mean','').replace('_corr_mean','').replace('_mean','') 
                          for col in marker_cols}
            pheno_means = pheno_means.rename(columns=clean_names).T
            pheno_means_zscore = pheno_means.apply(lambda x: (x - x.mean()) / x.std(), axis=1).fillna(0)
            
            fig = Figure(figsize=(12, 8), facecolor='white')
            ax = fig.add_subplot(111)
            
            im = ax.imshow(pheno_means_zscore.values, cmap='bwr', aspect='auto', vmin=-2, vmax=2)
            
            ax.set_title('Phenotype Expression Heatmap', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Phenotypes', fontsize=14)
            ax.set_ylabel('Markers', fontsize=14)
            
            pheno_counts = pheno_df.groupby('phenotype')['cell_count'].sum()
            x_labels = [f"{pheno}\\n({pheno_counts.get(pheno, 0):,})" for pheno in pheno_means_zscore.columns]
            ax.set_xticks(range(len(pheno_means_zscore.columns)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=11)
            
            ax.set_yticks(range(len(pheno_means_zscore.index)))
            ax.set_yticklabels(pheno_means_zscore.index, fontsize=10)
            
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label('Z-score', fontsize=12)
            
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, heatmap_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()

        def save_and_close(self):
            if not self.meta_cluster_phenotypes:
                messagebox.showwarning("Warning", "No assignments made")
                return
            
            summary = f"ASSIGNMENT SUMMARY\\n{'='*50}\\n\\n"
            summary += f"Total assignments: {len(self.assignments)}\\n"
            summary += f"Clusters assigned: {len(self.assigned_clusters)}/{self.total_clusters}\\n"
            summary += f"Clusters remaining: {len(self.unassigned_clusters)}\\n\\n"
            
            summary += "PHENOTYPE BREAKDOWN:\\n"
            for assignment in self.assignments:
                summary += f"   {assignment['phenotype']}: {len(assignment['clusters'])} clusters, {assignment['cell_count']:,} cells\\n"
            
            if self.unassigned_clusters:
                summary += f"\\nUnassigned clusters: {sorted(self.unassigned_clusters)}\\n"
            
            summary += "\\nReady to proceed to Step 5!"
            
            messagebox.showinfo("Assignment Complete", summary)
            self.root.destroy()

        def run(self):
            self.root.mainloop()

    print("="*80)
    print("LAUNCHING SPACEMAP V4 - PHENOTYPE ASSIGNMENT STUDIO")
    print("="*80)
    print("")
    print("PROFESSIONAL FEATURES:")
    print("  * Interactive cluster selection with visual feedback")
    print("  * Red highlighting for assigned clusters")
    print("  * Green highlighting for unassigned clusters") 
    print("  * Yellow highlighting for selected clusters")
    print("  * Real-time statistics and progress tracking")
    print("  * Assignment editing and management")
    print("  * Interactive assignment table")
    print("  * Phenotype heatmap generation")
    print("  * Professional control panel")
    print("")
    print("INTERACTION GUIDE:")
    print("   Click clusters to select/deselect")
    print("   Click assigned clusters to see assignment info")
    print("   Single-click table rows to highlight clusters")
    print("   Double-click table rows to edit assignments")
    print("   Use 'Select Unassigned' for quick selection")
    print("")
    print("Launching application...")
    
    app = PhenotypeAssignmentApp(char_df, reordered_cluster_ids, meta_cluster_phenotypes)
    app.run()
    
    print("Application closed!")
    if app.meta_cluster_phenotypes:
        print(f"Total assignments saved: {len(app.meta_cluster_phenotypes)}")
        print("Ready to proceed to Step 5!")
    else:
        print("No assignments were made.")
    
    return app.meta_cluster_phenotypes



def apply_scimap_phenotypes(
    data,
    phenotype_assignments,
    output_base_dir,
    method_prefix='scimap',
    cell_id_col='CellID'
):
    """
    Apply SCIMAP phenotype assignments to unified dataset (compatible with Leiden/SOM pattern).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with cell data (should exclude normalized columns)
    phenotype_assignments : pd.Series or dict
        Phenotype assignments indexed by cell position or dict mapping
    output_base_dir : Path
        Base output directory for saving files
    method_prefix : str
        Method prefix for output files ('scimap')
    cell_id_col : str
        Name of cell ID column for proper anchoring
        
    Returns
    -------
    dict
        Dictionary with results including unified_file, summary info, etc.
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    if output_base_dir is None:
        raise ValueError("output_base_dir must be provided")
    
    output_base_dir = Path(output_base_dir)
    phenotyping_dir = output_base_dir
    phenotyping_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate cell ID column
    if cell_id_col not in data.columns:
        raise ValueError(f"Cell ID column '{cell_id_col}' not found in data")
    
    print(f"Applying {method_prefix} phenotypes to unified dataset...")
    
    phenotype_col = f'phenotype_{method_prefix}'
    
    # Create working copy and add phenotype assignments
    result_data = data.copy()
    
    # Handle different types of phenotype_assignments input
    if isinstance(phenotype_assignments, pd.Series):
        result_data[phenotype_col] = phenotype_assignments.reindex(data.index)
    elif isinstance(phenotype_assignments, dict):
        result_data[phenotype_col] = pd.Series(phenotype_assignments).reindex(data.index)
    elif isinstance(phenotype_assignments, (list, np.ndarray)):
        if len(phenotype_assignments) != len(data):
            raise ValueError(f"Phenotype assignments length {len(phenotype_assignments)} doesn't match data length {len(data)}")
        result_data[phenotype_col] = phenotype_assignments
    else:
        raise ValueError("phenotype_assignments must be Series, dict, list, or array")
    
    # Filter out any normalized columns that might have slipped in
    norm_cols = [c for c in result_data.columns if c.endswith('_norm')]
    if norm_cols:
        result_data = result_data.drop(columns=norm_cols)
        print(f"Filtered out {len(norm_cols)} normalized columns")
    
    # Define unified output file
    unified_file = phenotyping_dir / 'all_samples_combined_corrected_phenotyped.csv'
    
    # Handle existing unified file using cell_id-based merging (like Leiden/SOM)
    if unified_file.exists():
        print(f"Updating existing unified file: {unified_file}")
        existing_df = pd.read_csv(unified_file)
        
        if cell_id_col not in existing_df.columns:
            raise ValueError(f"Existing unified file missing '{cell_id_col}' column")
        
        # Remove existing SCIMAP column if present
        if phenotype_col in existing_df.columns:
            existing_df = existing_df.drop(columns=[phenotype_col])
            print(f"Removed existing {phenotype_col} column")
        
        # Prepare update data with only cell_id and new phenotype column
        update_data = result_data[[cell_id_col, phenotype_col]].copy()
        
        # Merge based on cell_id
        existing_df = existing_df.set_index(cell_id_col)
        update_data = update_data.set_index(cell_id_col)
        
        # Update with new phenotype column
        existing_df[phenotype_col] = update_data[phenotype_col]
        
        # Reset index to get cell_id back as column (avoid duplicates)
        if cell_id_col in existing_df.columns:
            existing_df = existing_df.drop(columns=[cell_id_col])
        unified_df = existing_df.reset_index()
        
        print(f"Updated {len(unified_df):,} cells with {phenotype_col} assignments")
    else:
        # Create new unified file
        print(f"Creating new unified file: {unified_file}")
        unified_df = result_data.copy()
    
    # Save unified file
    unified_df.to_csv(unified_file, index=False)
    print(f"Saved unified file: {unified_file}")
    
    # Generate summary statistics
    phenotype_counts = result_data[phenotype_col].value_counts()
    summary_df = pd.DataFrame({
        'cell_count': phenotype_counts,
        'percentage': (phenotype_counts / len(result_data) * 100).round(1)
    })
    
    # Save summary file
    summary_file = phenotyping_dir / f'{method_prefix}_phenotype_summary.csv'
    summary_df.to_csv(summary_file)
    
    # Create mapping file (for compatibility)
    mapping_df = pd.DataFrame({
        'phenotype': summary_df.index,
        'cell_count': summary_df['cell_count'].values
    })
    mapping_file = phenotyping_dir / f'{method_prefix}_phenotype_mapping.csv'
    mapping_df.to_csv(mapping_file, index=False)
    
    print(f"Generated summary files:")
    print(f"  Summary: {summary_file}")
    print(f"  Mapping: {mapping_file}")
    
    return {
        'data': unified_df,
        'unified_file': unified_file,
        'summary_file': summary_file,
        'summary_df': summary_df,
        'mapping_file': mapping_file,
        'phenotype_col': phenotype_col
    }


def apply_som_phenotypes(data, som_clusters, meta_cluster_map, meta_cluster_phenotypes, output_base_dir, method_prefix='som'):
    """Apply SOM->meta-cluster->phenotype mapping & persist unified + summary files.

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
    """
    output_base_dir = Path(output_base_dir)
    phenotyping_dir = output_base_dir / 'phenotyping'
    phenotyping_dir.mkdir(exist_ok=True)
    if 'som_cluster' not in data.columns:
        data = data.copy()
        data['som_cluster'] = som_clusters

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
    if unified_path.exists():
        existing = pd.read_csv(unified_path)
        cell_id_col = 'CellID'  # Use standard CellID column
        if len(existing) == len(data) and cell_id_col in existing.columns and cell_id_col in data.columns:
            # FIXED: Use ID-based merge to properly anchor results to correct cells
            existing = existing.set_index(cell_id_col)
            data_indexed = data.set_index(cell_id_col)
            
            # Update columns using ID-based alignment
            existing['som_cluster'] = data_indexed['som_cluster']
            existing[phenotype_col] = data_indexed[phenotype_col]
            
            # Reset index to get cell_id back as a column (avoid duplicates)
            if cell_id_col in existing.columns:
                existing = existing.drop(columns=[cell_id_col])
            unified_df = existing.reset_index()
        else:
            # If lengths mismatch or no 'id' column, trust the new data completely for this run
            unified_df = data.copy()
    else:
        unified_df = data.copy()
    unified_df.to_csv(unified_path, index=False)

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

