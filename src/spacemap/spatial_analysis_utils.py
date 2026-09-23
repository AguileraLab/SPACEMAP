"""
Spatial analysis utilities for SPACEMAP pipeline.

This module provides functions for analyzing spatial distribution patterns
of identified cell clusters/clones, including spatial metrics, neighbor
analysis, and statistical tests for spatial randomness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import distance_matrix, cKDTree
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


def compute_clone_spatial_metrics(df, clone_col, group_cols, spatial_cols=['x_centroid', 'y_centroid']):
    """
    Compute spatial metrics for each clone within each ROI.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level data with clone assignments and spatial coordinates
    clone_col : str
        Column name containing clone/cluster IDs
    group_cols : list
        Columns defining independent spatial regions (e.g., ['batch_id', 'tissue_roi', 'tumor_roi'])
    spatial_cols : list
        Column names for x and y coordinates
    
    Returns
    -------
    pd.DataFrame
        Spatial metrics for each clone in each ROI
    """
    
    results = []
    
    for group_keys, group_df in df.groupby(group_cols):
        # Get unique clones in this ROI
        clones = group_df[clone_col].unique()
        
        # Create group identifier
        if isinstance(group_keys, tuple):
            group_id = "_".join(str(k) for k in group_keys)
            group_dict = dict(zip(group_cols, group_keys))
        else:
            group_id = str(group_keys)
            group_dict = {group_cols[0]: group_keys}
        
        # Overall ROI metrics
        roi_area = compute_convex_hull_area(group_df[spatial_cols].values)
        roi_cell_count = len(group_df)
        
        for clone in clones:
            clone_df = group_df[group_df[clone_col] == clone]
            
            if len(clone_df) < 2:
                continue
            
            # Extract coordinates
            coords = clone_df[spatial_cols].values
            
            # Basic metrics
            cell_count = len(clone_df)
            proportion = cell_count / roi_cell_count
            
            # Spatial dispersion metrics
            centroid = coords.mean(axis=0)
            distances_from_centroid = np.sqrt(((coords - centroid)**2).sum(axis=1))
            mean_distance_from_centroid = distances_from_centroid.mean()
            std_distance_from_centroid = distances_from_centroid.std()
            
            # Spatial extent
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            spatial_extent = np.sqrt(x_range**2 + y_range**2)
            
            # Clone area (convex hull)
            if len(coords) >= 3:
                clone_area = compute_convex_hull_area(coords)
                area_ratio = clone_area / roi_area if roi_area > 0 else 0
                density = cell_count / clone_area if clone_area > 0 else 0
            else:
                clone_area = 0
                area_ratio = 0
                density = 0
            
            # Nearest neighbor distances (within clone)
            if len(coords) >= 2:
                dist_matrix = distance_matrix(coords, coords)
                np.fill_diagonal(dist_matrix, np.inf)
                nn_distances = dist_matrix.min(axis=1)
                mean_nn_distance = nn_distances.mean()
                std_nn_distance = nn_distances.std()
            else:
                mean_nn_distance = 0
                std_nn_distance = 0
            
            # Compile results
            result = {
                **group_dict,
                'clone': clone,
                'cell_count': cell_count,
                'proportion': proportion,
                'centroid_x': centroid[0],
                'centroid_y': centroid[1],
                'mean_dist_from_centroid': mean_distance_from_centroid,
                'std_dist_from_centroid': std_distance_from_centroid,
                'spatial_extent': spatial_extent,
                'clone_area': clone_area,
                'area_ratio': area_ratio,
                'density': density,
                'mean_nn_distance': mean_nn_distance,
                'std_nn_distance': std_nn_distance,
                'x_range': x_range,
                'y_range': y_range
            }
            
            results.append(result)
    
    return pd.DataFrame(results)


def compute_convex_hull_area(coords):
    """Compute area of convex hull around points."""
    try:
        from scipy.spatial import ConvexHull
        if len(coords) < 3:
            return 0
        hull = ConvexHull(coords)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0


def test_spatial_randomness(df, clone_col, group_cols, spatial_cols=['x_centroid', 'y_centroid'], 
                            n_simulations=1000, random_seed=42):
    """
    Test whether clone spatial distribution is random or clustered/dispersed.
    
    Uses Monte Carlo simulation to compare observed nearest neighbor distances
    to expected distribution under complete spatial randomness (CSR).
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level data
    clone_col : str
        Clone/cluster column name
    group_cols : list
        Spatial region grouping columns
    spatial_cols : list
        Coordinate column names
    n_simulations : int
        Number of Monte Carlo simulations
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Statistical test results for each clone in each ROI
    """
    
    np.random.seed(random_seed)
    results = []
    
    for group_keys, group_df in df.groupby(group_cols):
        # Create group identifier
        if isinstance(group_keys, tuple):
            group_dict = dict(zip(group_cols, group_keys))
        else:
            group_dict = {group_cols[0]: group_keys}
        
        clones = group_df[clone_col].unique()
        
        # Get spatial bounds for simulation
        x_min, x_max = group_df[spatial_cols[0]].min(), group_df[spatial_cols[0]].max()
        y_min, y_max = group_df[spatial_cols[1]].min(), group_df[spatial_cols[1]].max()
        
        for clone in clones:
            clone_df = group_df[group_df[clone_col] == clone]
            
            if len(clone_df) < 2:
                continue
            
            # Observed nearest neighbor distances
            coords = clone_df[spatial_cols].values
            obs_nn_dist = compute_mean_nn_distance(coords)
            
            # Simulate under CSR
            n_cells = len(clone_df)
            simulated_nn_dists = []
            
            for _ in range(n_simulations):
                # Generate random points in same spatial extent
                random_x = np.random.uniform(x_min, x_max, n_cells)
                random_y = np.random.uniform(y_min, y_max, n_cells)
                random_coords = np.column_stack([random_x, random_y])
                
                sim_nn_dist = compute_mean_nn_distance(random_coords)
                simulated_nn_dists.append(sim_nn_dist)
            
            simulated_nn_dists = np.array(simulated_nn_dists)
            
            # Calculate statistics
            expected_mean = simulated_nn_dists.mean()
            expected_std = simulated_nn_dists.std()
            
            # Z-score
            z_score = (obs_nn_dist - expected_mean) / expected_std if expected_std > 0 else 0
            
            # P-value (two-tailed)
            p_value = (np.abs(simulated_nn_dists - expected_mean) >= np.abs(obs_nn_dist - expected_mean)).mean()
            
            # Classification
            if p_value < 0.05:
                if obs_nn_dist < expected_mean:
                    pattern = 'Clustered'
                else:
                    pattern = 'Dispersed'
            else:
                pattern = 'Random'
            
            result = {
                **group_dict,
                'clone': clone,
                'cell_count': len(clone_df),
                'observed_nn_dist': obs_nn_dist,
                'expected_nn_dist': expected_mean,
                'expected_std': expected_std,
                'z_score': z_score,
                'p_value': p_value,
                'spatial_pattern': pattern
            }
            
            results.append(result)
    
    return pd.DataFrame(results)


def compute_mean_nn_distance(coords):
    """Compute mean nearest neighbor distance."""
    if len(coords) < 2:
        return 0
    dist_matrix = distance_matrix(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    nn_distances = dist_matrix.min(axis=1)
    return nn_distances.mean()


def compute_clone_interactions(df, clone_col, group_cols, spatial_cols=['x_centroid', 'y_centroid'],
                               interaction_radius=50):
    """
    Compute spatial interactions between different clones.
    
    For each cell, counts how many neighbors of each clone type are within
    the interaction radius.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level data
    clone_col : str
        Clone/cluster column name
    group_cols : list
        Spatial region grouping columns
    spatial_cols : list
        Coordinate column names
    interaction_radius : float
        Radius for defining spatial neighbors (in same units as coordinates)
    
    Returns
    -------
    pd.DataFrame
        Interaction matrix showing neighbor counts between clone pairs
    """
    
    results = []
    
    for group_keys, group_df in df.groupby(group_cols):
        if isinstance(group_keys, tuple):
            group_dict = dict(zip(group_cols, group_keys))
        else:
            group_dict = {group_cols[0]: group_keys}
        
        # Build spatial tree for efficient neighbor search
        coords = group_df[spatial_cols].values
        clones = group_df[clone_col].values
        tree = cKDTree(coords)
        
        # Get unique clones
        unique_clones = group_df[clone_col].unique()
        
        # Initialize interaction matrix
        interaction_matrix = pd.DataFrame(0, 
                                         index=unique_clones, 
                                         columns=unique_clones)
        
        # Count interactions
        for i, (coord, clone) in enumerate(zip(coords, clones)):
            # Find neighbors within radius
            neighbors = tree.query_ball_point(coord, interaction_radius)
            neighbors = [n for n in neighbors if n != i]  # Exclude self
            
            # Count neighbors by clone type
            if neighbors:
                neighbor_clones = clones[neighbors]
                for neighbor_clone in neighbor_clones:
                    interaction_matrix.loc[clone, neighbor_clone] += 1
        
        # Convert to long format
        for clone_a in unique_clones:
            for clone_b in unique_clones:
                result = {
                    **group_dict,
                    'clone_a': clone_a,
                    'clone_b': clone_b,
                    'interaction_count': interaction_matrix.loc[clone_a, clone_b],
                    'cells_clone_a': (group_df[clone_col] == clone_a).sum(),
                    'cells_clone_b': (group_df[clone_col] == clone_b).sum()
                }
                results.append(result)
    
    return pd.DataFrame(results)


def compute_distance_from_boundary(df, group_cols, spatial_cols=['x_centroid', 'y_centroid']):
    """
    Compute distance of each cell from ROI boundary.
    
    Useful for determining if clones are enriched at tumor periphery vs core.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level data
    group_cols : list
        Spatial region grouping columns
    spatial_cols : list
        Coordinate column names
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'distance_from_boundary' column
    """
    
    df = df.copy()
    df['distance_from_boundary'] = np.nan
    
    for group_keys, group_df in df.groupby(group_cols):
        idx = group_df.index
        coords = group_df[spatial_cols].values
        
        # Compute convex hull as boundary approximation
        try:
            from scipy.spatial import ConvexHull
            if len(coords) >= 3:
                hull = ConvexHull(coords)
                boundary_coords = coords[hull.vertices]
                
                # For each cell, find distance to nearest boundary point
                for i, coord in enumerate(coords):
                    distances = np.sqrt(((boundary_coords - coord)**2).sum(axis=1))
                    df.loc[idx[i], 'distance_from_boundary'] = distances.min()
        except Exception:
            # If convex hull fails, use distance from edges
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            for i, coord in enumerate(coords):
                dist_to_edges = [
                    coord[0] - x_min,
                    x_max - coord[0],
                    coord[1] - y_min,
                    y_max - coord[1]
                ]
                df.loc[idx[i], 'distance_from_boundary'] = min(dist_to_edges)
    
    return df


def test_clone_boundary_enrichment(df, clone_col, boundary_threshold=100):
    """
    Test whether clones are enriched at boundary vs core.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'distance_from_boundary' column
    clone_col : str
        Clone/cluster column name
    boundary_threshold : float
        Distance threshold to define boundary region
    
    Returns
    -------
    pd.DataFrame
        Statistical test results for each clone
    """
    
    if 'distance_from_boundary' not in df.columns:
        raise ValueError("Run compute_distance_from_boundary() first")
    
    results = []
    clones = df[clone_col].unique()
    
    for clone in clones:
        clone_df = df[df[clone_col] == clone]
        other_df = df[df[clone_col] != clone]
        
        # Count cells in boundary vs core
        clone_boundary = (clone_df['distance_from_boundary'] <= boundary_threshold).sum()
        clone_core = (clone_df['distance_from_boundary'] > boundary_threshold).sum()
        other_boundary = (other_df['distance_from_boundary'] <= boundary_threshold).sum()
        other_core = (other_df['distance_from_boundary'] > boundary_threshold).sum()
        
        # Chi-square test
        contingency_table = np.array([[clone_boundary, clone_core],
                                     [other_boundary, other_core]])
        
        if contingency_table.min() > 0:
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Odds ratio
            odds_ratio = (clone_boundary * other_core) / (clone_core * other_boundary) if clone_core > 0 and other_boundary > 0 else np.nan
            
            # Enrichment classification
            if p_value < 0.05:
                if odds_ratio > 1:
                    enrichment = 'Boundary-enriched'
                else:
                    enrichment = 'Core-enriched'
            else:
                enrichment = 'No enrichment'
        else:
            chi2, p_value, odds_ratio = np.nan, np.nan, np.nan
            enrichment = 'Insufficient data'
        
        result = {
            'clone': clone,
            'cells_boundary': clone_boundary,
            'cells_core': clone_core,
            'proportion_boundary': clone_boundary / (clone_boundary + clone_core) if (clone_boundary + clone_core) > 0 else 0,
            'chi2': chi2,
            'p_value': p_value,
            'odds_ratio': odds_ratio,
            'enrichment': enrichment
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def plot_clone_spatial_distribution(df, clone_col, group_cols, spatial_cols=['x_centroid', 'y_centroid'],
                                    output_dir=None, max_plots=20, figsize=(8, 8)):
    """
    Plot spatial distribution of clones for each ROI.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cell-level data
    clone_col : str
        Clone/cluster column name
    group_cols : list
        Spatial region grouping columns
    spatial_cols : list
        Coordinate column names
    output_dir : Path-like
        Directory to save plots
    max_plots : int
        Maximum number of plots to generate
    figsize : tuple
        Figure size
    
    Returns
    -------
    None
        Saves plots to output directory
    """
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    groups = list(df.groupby(group_cols))
    
    if len(groups) > max_plots:
        print(f"Warning: {len(groups)} ROIs found, only plotting first {max_plots}")
        groups = groups[:max_plots]
    
    for group_keys, group_df in groups:
        if isinstance(group_keys, tuple):
            group_id = "_".join(str(k) for k in group_keys)
        else:
            group_id = str(group_keys)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cells colored by clone
        clones = group_df[clone_col].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(clones)))
        
        for clone, color in zip(clones, colors):
            clone_df = group_df[group_df[clone_col] == clone]
            ax.scatter(clone_df[spatial_cols[0]], 
                      clone_df[spatial_cols[1]],
                      c=[color], 
                      label=f'Clone {clone} (n={len(clone_df)})',
                      s=5, alpha=0.6)
        
        ax.set_xlabel(spatial_cols[0])
        ax.set_ylabel(spatial_cols[1])
        ax.set_title(f'Spatial Distribution: {group_id}\n{len(group_df):,} cells, {len(clones)} clones')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_dir is not None:
            filename = f'spatial_distribution_{group_id}.png'
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_clone_metrics_summary(spatial_metrics_df, output_dir=None):
    """
    Create summary visualizations of clone spatial metrics.
    
    Parameters
    ----------
    spatial_metrics_df : pd.DataFrame
        Output from compute_clone_spatial_metrics()
    output_dir : Path-like
        Directory to save plots
    
    Returns
    -------
    None
        Displays or saves summary plots
    """
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Clone size distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cell count per clone
    axes[0, 0].hist(spatial_metrics_df['cell_count'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Cells per Clone')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Clone Size Distribution')
    
    # Spatial dispersion
    axes[0, 1].hist(spatial_metrics_df['mean_dist_from_centroid'], bins=30, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Mean Distance from Centroid')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Spatial Dispersion')
    
    # Clone density
    axes[1, 0].hist(spatial_metrics_df['density'], bins=30, edgecolor='black', color='green')
    axes[1, 0].set_xlabel('Cell Density (cells/area)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Clone Density Distribution')
    
    # Nearest neighbor distances
    axes[1, 1].hist(spatial_metrics_df['mean_nn_distance'], bins=30, edgecolor='black', color='red')
    axes[1, 1].set_xlabel('Mean Nearest Neighbor Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Within-Clone Cell Spacing')
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(output_dir / 'clone_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
