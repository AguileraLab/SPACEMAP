"""
Spatial microenvironment analysis utilities for SPACEMAP pipeline.

This module provides functions for analyzing spatial relationships between cells
and tissue regions (tumor beds, epithelia beds) to study microenvironment composition
and cellular gradients.

Key functions:
- compute_tumor_bed(): Assign tumor_bed status (1=tumor, 0=normal) based on cell type
- compute_distance_tumor(): Calculate signed distances to tumor regions
- compute_distance_epithelia(): Calculate signed distances to epithelia regions
- run_spatial_beds_analysis(): Main orchestration function
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_tumor_bed(group_df: pd.DataFrame, na_fill_value: int = 999) -> Tuple[pd.Series, Dict]:
    """
    Assign tumor_bed status for all cells in a group based on cell type and proximity.
    
    Algorithm:
    - Tumor cells (tumor==1) → tumor_bed = 1
    - Epithelia cells (epithelia==1) → tumor_bed = 0
    - Stroma cells: assign based on proximity (closer to tumor or epithelia)
    - Unassigned cells (no tumor or epithelia in group): filled with na_fill_value
    
    Parameters
    ----------
    group_df : pd.DataFrame
        Single (batch_id, tissue_roi) group with columns:
        'tumor', 'epithelia', 'stroma', 'x_centroid', 'y_centroid'
    na_fill_value : int
        Value to fill for cells that cannot be assigned (default: 999)
    
    Returns
    -------
    tumor_bed : pd.Series
        tumor_bed assignments for each cell in the group (1, 0, or na_fill_value)
    stats : Dict
        Summary statistics for the group
    """
    tumor_bed = pd.Series(np.nan, index=group_df.index, dtype='float64')
    
    # Identify cell types
    tumor_mask = group_df['tumor'] == 1
    epithelia_mask = group_df['epithelia'] == 1
    stroma_mask = (group_df['stroma'] == 1) & ~tumor_mask & ~epithelia_mask
    
    tumor_indices = group_df[tumor_mask].index
    epithelia_indices = group_df[epithelia_mask].index
    stroma_indices = group_df[stroma_mask].index
    
    n_tumor = len(tumor_indices)
    n_epithelia = len(epithelia_indices)
    n_stroma = len(stroma_indices)
    
    # Assign tumor_bed for tumor cells
    if n_tumor > 0:
        tumor_bed.loc[tumor_indices] = 1.0
    
    # Assign tumor_bed for epithelia cells
    if n_epithelia > 0:
        tumor_bed.loc[epithelia_indices] = 0.0
    
    # Assign tumor_bed for stroma cells based on proximity
    if n_stroma > 0 and (n_tumor > 0 or n_epithelia > 0):
        stroma_coords = group_df.loc[stroma_indices, ['x_centroid', 'y_centroid']].values
        
        # Compute distance to tumor
        dist_t = np.full(n_stroma, np.inf)
        if n_tumor > 0:
            tumor_coords = group_df.loc[tumor_indices, ['x_centroid', 'y_centroid']].values
            dist_t = cdist(stroma_coords, tumor_coords, metric='euclidean').min(axis=1)
        
        # Compute distance to epithelia
        dist_e = np.full(n_stroma, np.inf)
        if n_epithelia > 0:
            epithelia_coords = group_df.loc[epithelia_indices, ['x_centroid', 'y_centroid']].values
            dist_e = cdist(stroma_coords, epithelia_coords, metric='euclidean').min(axis=1)
        
        # Assign based on closer bed (convert boolean to float)
        tumor_bed.loc[stroma_indices] = (dist_t < dist_e).astype(float)
    
    # Fill remaining NAs with na_fill_value
    tumor_bed = tumor_bed.fillna(float(na_fill_value))
    
    stats = {
        'n_tumor': n_tumor,
        'n_epithelia': n_epithelia,
        'n_stroma': n_stroma,
        'n_tumor_bed': (tumor_bed == 1.0).sum(),
        'n_normal_bed': (tumor_bed == 0.0).sum(),
        'n_unassigned': (tumor_bed == float(na_fill_value)).sum()
    }
    
    return tumor_bed, stats


def compute_distance_tumor(group_df: pd.DataFrame, tumor_bed_col: str) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Compute signed distances to tumor regions within tumor_bed == 1 cells.
    
    Algorithm:
    - Stroma cells (stroma==1): positive distance to nearest tumor
    - Tumor cells (tumor==1): negative distance to nearest stroma (boundary distance)
    - Other cells: NaN
    
    Also assigns bed_roi labels (t_{roi_number}) based on closest tumor ROI.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        Single (batch_id, tissue_roi) group
    tumor_bed_col : str
        Column name containing tumor_bed assignments
    
    Returns
    -------
    distance_tumor : pd.Series
        Signed distances (NaN for cells outside tumor bed)
    bed_roi : pd.Series
        ROI labels t_1, t_2, etc. (NaN for cells without tumor ROI)
    stats : Dict
        Summary statistics
    """
    distance_tumor = pd.Series(np.nan, index=group_df.index, dtype='float64')
    bed_roi = pd.Series(np.nan, index=group_df.index, dtype='object')
    
    # Filter to tumor_bed == 1
    tumor_bed_mask = group_df[tumor_bed_col] == 1.0
    if tumor_bed_mask.sum() == 0:
        return distance_tumor, bed_roi, {'n_computed': 0, 'n_stroma': 0, 'n_tumor': 0}
    
    # Identify cell types within tumor bed
    tumor_mask = tumor_bed_mask & (group_df['tumor'] == 1)
    stroma_mask = tumor_bed_mask & (group_df['stroma'] == 1)
    
    tumor_indices = group_df[tumor_mask].index
    stroma_indices = group_df[stroma_mask].index
    
    n_tumor = len(tumor_indices)
    n_stroma = len(stroma_indices)
    n_computed = 0
    
    # Process stroma cells (positive distances)
    if n_stroma > 0 and n_tumor > 0:
        stroma_coords = group_df.loc[stroma_indices, ['x_centroid', 'y_centroid']].values
        tumor_coords = group_df.loc[tumor_indices, ['x_centroid', 'y_centroid']].values
        tumor_rois = group_df.loc[tumor_indices, 'tumor_roi'].values
        
        distances = cdist(stroma_coords, tumor_coords, metric='euclidean')
        min_distances = distances.min(axis=1)
        min_indices = distances.argmin(axis=1)
        
        distance_tumor.loc[stroma_indices] = min_distances
        closest_rois = tumor_rois[min_indices]
        bed_roi.loc[stroma_indices] = ['t_' + str(int(r)) if pd.notna(r) and r > 0 else np.nan 
                                        for r in closest_rois]
        n_computed += n_stroma
    
    # Process tumor cells (negative distances)
    if n_tumor > 0 and n_stroma > 0:
        tumor_coords = group_df.loc[tumor_indices, ['x_centroid', 'y_centroid']].values
        stroma_coords = group_df.loc[stroma_indices, ['x_centroid', 'y_centroid']].values
        tumor_rois = group_df.loc[tumor_indices, 'tumor_roi'].values
        
        distances = cdist(tumor_coords, stroma_coords, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        distance_tumor.loc[tumor_indices] = -min_distances
        bed_roi.loc[tumor_indices] = ['t_' + str(int(r)) if pd.notna(r) and r > 0 else np.nan 
                                       for r in tumor_rois]
        n_computed += n_tumor
    elif n_tumor > 0:
        # Tumor cells without stroma neighbors: assign own ROI with NaN distance
        tumor_rois = group_df.loc[tumor_indices, 'tumor_roi'].values
        bed_roi.loc[tumor_indices] = ['t_' + str(int(r)) if pd.notna(r) and r > 0 else np.nan 
                                       for r in tumor_rois]
    
    stats = {
        'n_computed': n_computed,
        'n_stroma': n_stroma,
        'n_tumor': n_tumor,
        'n_with_distance': distance_tumor.notna().sum(),
        'n_positive': (distance_tumor > 0).sum(),
        'n_negative': (distance_tumor < 0).sum()
    }
    
    return distance_tumor, bed_roi, stats


def compute_distance_epithelia(group_df: pd.DataFrame, tumor_bed_col: str) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Compute signed distances to epithelia regions within tumor_bed == 0 cells.
    
    Algorithm:
    - Stroma cells (stroma==1): positive distance to nearest epithelia
    - Epithelia cells (epithelia==1): negative distance to nearest stroma (boundary distance)
    - Other cells: NaN
    
    Also assigns bed_roi labels (e_{roi_number}) based on closest epithelia ROI.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        Single (batch_id, tissue_roi) group
    tumor_bed_col : str
        Column name containing tumor_bed assignments
    
    Returns
    -------
    distance_epithelia : pd.Series
        Signed distances (NaN for cells outside normal bed)
    bed_roi : pd.Series
        ROI labels e_1, e_2, etc. (NaN for cells without epithelia ROI)
    stats : Dict
        Summary statistics
    """
    distance_epithelia = pd.Series(np.nan, index=group_df.index, dtype='float64')
    bed_roi = pd.Series(np.nan, index=group_df.index, dtype='object')
    
    # Filter to tumor_bed == 0
    normal_bed_mask = group_df[tumor_bed_col] == 0.0
    if normal_bed_mask.sum() == 0:
        return distance_epithelia, bed_roi, {'n_computed': 0, 'n_stroma': 0, 'n_epithelia': 0}
    
    # Identify cell types within normal bed
    epithelia_mask = normal_bed_mask & (group_df['epithelia'] == 1)
    stroma_mask = normal_bed_mask & (group_df['stroma'] == 1)
    
    epithelia_indices = group_df[epithelia_mask].index
    stroma_indices = group_df[stroma_mask].index
    
    n_epithelia = len(epithelia_indices)
    n_stroma = len(stroma_indices)
    n_computed = 0
    
    # Process stroma cells (positive distances)
    if n_stroma > 0 and n_epithelia > 0:
        stroma_coords = group_df.loc[stroma_indices, ['x_centroid', 'y_centroid']].values
        epithelia_coords = group_df.loc[epithelia_indices, ['x_centroid', 'y_centroid']].values
        epithelia_rois = group_df.loc[epithelia_indices, 'epithelia_roi'].values
        
        distances = cdist(stroma_coords, epithelia_coords, metric='euclidean')
        min_distances = distances.min(axis=1)
        min_indices = distances.argmin(axis=1)
        
        distance_epithelia.loc[stroma_indices] = min_distances
        closest_rois = epithelia_rois[min_indices]
        bed_roi.loc[stroma_indices] = ['e_' + str(int(r)) if pd.notna(r) and r > 0 else np.nan 
                                        for r in closest_rois]
        n_computed += n_stroma
    
    # Process epithelia cells (negative distances)
    if n_epithelia > 0 and n_stroma > 0:
        epithelia_coords = group_df.loc[epithelia_indices, ['x_centroid', 'y_centroid']].values
        stroma_coords = group_df.loc[stroma_indices, ['x_centroid', 'y_centroid']].values
        epithelia_rois = group_df.loc[epithelia_indices, 'epithelia_roi'].values
        
        distances = cdist(epithelia_coords, stroma_coords, metric='euclidean')
        min_distances = distances.min(axis=1)
        
        distance_epithelia.loc[epithelia_indices] = -min_distances
        bed_roi.loc[epithelia_indices] = ['e_' + str(int(r)) if pd.notna(r) and r > 0 else np.nan 
                                           for r in epithelia_rois]
        n_computed += n_epithelia
    elif n_epithelia > 0:
        # Epithelia cells without stroma neighbors: assign own ROI with NaN distance
        epithelia_rois = group_df.loc[epithelia_indices, 'epithelia_roi'].values
        bed_roi.loc[epithelia_indices] = ['e_' + str(int(r)) if pd.notna(r) and r > 0 else np.nan 
                                           for r in epithelia_rois]
    
    stats = {
        'n_computed': n_computed,
        'n_stroma': n_stroma,
        'n_epithelia': n_epithelia,
        'n_with_distance': distance_epithelia.notna().sum(),
        'n_positive': (distance_epithelia > 0).sum(),
        'n_negative': (distance_epithelia < 0).sum()
    }
    
    return distance_epithelia, bed_roi, stats


def run_spatial_beds_analysis(
    input_file: Path,
    output_file: Path,
    required_cols: list = None,
    verbose: bool = True,
    na_fill_value: int = 999
) -> Dict:
    """
    Main orchestration function for spatial beds analysis.
    
    Processes each (batch_id, tissue_roi) group independently and computes:
    - tumor_bed: Binary assignment (1=tumor region, 0=normal epithelia region, na_fill_value=unassigned)
    - distance_tumor: Signed distance to tumor boundary (positive=approaching, negative=within)
    - distance_epithelia: Signed distance to epithelia boundary
    - bed_roi: ROI label (t_N for tumor, e_N for epithelia)
    
    Parameters
    ----------
    input_file : Path
        Path to unified CSV file
    output_file : Path
        Path to save results
    required_cols : list, optional
        Required columns (default: standard spatial columns)
    verbose : bool
        Print progress messages
    na_fill_value : int
        Value to fill for unassigned tumor_bed cells (default: 999)
    
    Returns
    -------
    results : Dict
        Comprehensive summary statistics
    """
    if required_cols is None:
        required_cols = ['batch_id', 'tissue_roi', 'x_centroid', 'y_centroid',
                        'tumor', 'epithelia', 'stroma', 'tumor_roi', 'epithelia_roi']
    
    # Load data
    if verbose:
        print("Loading unified CSV file...")
    
    df = pd.read_csv(input_file, low_memory=False)
    
    if verbose:
        print(f"Loaded: {len(df):,} cells")
    
    # Verify required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop existing output columns to allow reruns
    output_cols = ['tumor_bed', 'distance_tumor', 'distance_epithelia', 'bed_roi']
    df = df.drop(columns=[col for col in output_cols if col in df.columns], errors='ignore')
    
    # Initialize output columns
    df['tumor_bed'] = pd.NA
    df['distance_tumor'] = pd.NA
    df['distance_epithelia'] = pd.NA
    df['bed_roi'] = pd.NA
    
    # Group processing
    groups = list(df.groupby(['batch_id', 'tissue_roi']))
    n_groups = len(groups)
    
    if verbose:
        print(f"\nProcessing {n_groups} (batch_id × tissue_roi) groups...")
    
    # Initialize global statistics
    global_stats = {
        'tumor_bed': {'n_tumor_bed': 0, 'n_normal_bed': 0, 'n_unassigned': 0},
        'distance_tumor': {'n_computed': 0, 'n_positive': 0, 'n_negative': 0},
        'distance_epithelia': {'n_computed': 0, 'n_positive': 0, 'n_negative': 0},
        'bed_roi': {'n_tumor_rois': 0, 'n_epithelia_rois': 0, 'n_labeled': 0}
    }
    
    # Process groups
    for i, (group_key, group_df) in enumerate(groups):
        if verbose and (i + 1) % 50 == 0 or (i + 1) == n_groups:
            print(f"  Processed {i + 1}/{n_groups} groups...")
        
        group_indices = group_df.index
        batch_id, tissue_roi = group_key
        
        # Step 1: Compute tumor_bed
        tumor_bed, tb_stats = compute_tumor_bed(group_df, na_fill_value=na_fill_value)
        df.loc[group_indices, 'tumor_bed'] = tumor_bed
        
        # Accumulate statistics
        for key, val in tb_stats.items():
            if key in global_stats['tumor_bed']:
                global_stats['tumor_bed'][key] += val
        
        # Step 2: Compute distance_tumor and bed_roi for tumor bed
        group_df_with_bed = group_df.copy()
        group_df_with_bed['tumor_bed'] = tumor_bed
        
        dist_tumor, bed_roi_t, dt_stats = compute_distance_tumor(
            group_df_with_bed, 'tumor_bed'
        )
        df.loc[group_indices, 'distance_tumor'] = dist_tumor
        
        # Update bed_roi where tumor_bed == 1
        tumor_bed_mask = (tumor_bed == 1)
        bed_roi_update = bed_roi_t[tumor_bed_mask]
        df.loc[group_indices[tumor_bed_mask], 'bed_roi'] = bed_roi_update
        
        for key in ['n_positive', 'n_negative']:
            if key in dt_stats:
                global_stats['distance_tumor'][key] += dt_stats[key]
        global_stats['distance_tumor']['n_computed'] += dt_stats['n_computed']
        
        # Step 3: Compute distance_epithelia and bed_roi for normal bed
        dist_epi, bed_roi_e, de_stats = compute_distance_epithelia(
            group_df_with_bed, 'tumor_bed'
        )
        df.loc[group_indices, 'distance_epithelia'] = dist_epi
        
        # Update bed_roi where tumor_bed == 0
        normal_bed_mask = (tumor_bed == 0)
        bed_roi_update = bed_roi_e[normal_bed_mask]
        df.loc[group_indices[normal_bed_mask], 'bed_roi'] = bed_roi_update
        
        for key in ['n_positive', 'n_negative']:
            if key in de_stats:
                global_stats['distance_epithelia'][key] += de_stats[key]
        global_stats['distance_epithelia']['n_computed'] += de_stats['n_computed']
    
    if verbose:
        print(f"\nCompleted all {n_groups} groups")
    
    # Update bed_roi statistics
    n_tumor_rois = df[df['bed_roi'].notna()]['bed_roi'].str.startswith('t_').sum()
    n_epithelia_rois = df[df['bed_roi'].notna()]['bed_roi'].str.startswith('e_').sum()
    global_stats['bed_roi']['n_tumor_rois'] = n_tumor_rois
    global_stats['bed_roi']['n_epithelia_rois'] = n_epithelia_rois
    global_stats['bed_roi']['n_labeled'] = df['bed_roi'].notna().sum()
    
    # Save results
    df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\nSaved results to: {output_file}")
    
    return {'data': df, 'statistics': global_stats}
