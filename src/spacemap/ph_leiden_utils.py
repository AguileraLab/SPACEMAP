"""
Leiden clustering utilities for the SPACEMAP pipeline.

This module provides a streamlined set of functions for performing 
Leiden-based cell phenotyping, including data normalization, clustering, 
hyperparameter optimization, and interactive annotation.
"""

# CRITICAL: Protobuf fix MUST be before ANY other imports
# This prevents "Descriptors cannot be created directly" errors
import os

# Force Python implementation before any protobuf-using library loads
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# If already set but to something else, warn
elif os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] != "python":
    print("Warning: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION already set to", 
          os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"], flush=True)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import norm

# Scanpy for Leiden clustering
try:
    import scanpy as sc
    import anndata as ad
except ImportError:
    print("Warning: scanpy not installed. Leiden clustering will be unavailable.")

# Geometric sketching (density-aware subsampling)
try:
    import geosketch
    GEOSKETCH_AVAILABLE = True
except ImportError:
    GEOSKETCH_AVAILABLE = False
    geosketch = None
    print("Warning: geosketch not installed. Falling back to uniform random sampling for sketching.")

# Lazy import for umap to avoid tensorflow/protobuf issues at startup
umap = None
def get_umap():
    global umap
    if umap is None:
        try:
            import umap
        except ImportError:
            print("Warning: umap-learn not installed. Install for UMAP sketching.")
            umap = None
    return umap
    
    
def apply_flexible_normalization(df, marker_cols, group_cols, suffix,
                                 apply_z1=True, apply_z2=False, apply_cdf=False, apply_nlog=False, epsilon=1e-9):
    """
    Apply a flexible sequence of normalization steps to marker data.

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
                processed_data = np.clip(processed_data, -10, 10)  # Clip to prevent extreme values/overflow
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
                               normalization_flags, cache_path=None, id_col=None):
    """
    Load normalized markers from a cache CSV if available and valid; otherwise compute
    using apply_flexible_normalization and optionally save to cache.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing marker_cols and optionally id_col.
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
    id_col : str | None
        If provided, include this column in the returned DataFrame for stable merging.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized columns and optionally the id_col for stable merging.
    """
    from pathlib import Path

    desired_cols = [f"{m}{suffix}" for m in marker_cols]
    
    # Determine return columns (include id_col if specified)
    return_cols = desired_cols.copy()
    if id_col is not None and id_col in df.columns:
        return_cols = [id_col] + desired_cols

    # Try cache first
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path)
                # Check if cache has the required structure
                has_desired_cols = all(c in cached.columns for c in desired_cols)
                has_id_col = id_col is None or id_col in cached.columns
                
                if has_desired_cols and has_id_col:
                    if id_col is not None and id_col in cached.columns:
                        # Use ID-based merge for stable data alignment
                        result = pd.merge(df[[id_col]], cached, on=id_col, how='left')
                        return result[return_cols]
                    else:
                        # Fallback: assume row order matches (risky)
                        if len(cached) == len(df):
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

    # Save cache if requested (ALWAYS include ID column for stable anchoring)
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Include ID column in cache for stable data alignment
            cache_data = norm_df[desired_cols].copy()
            if id_col is not None and id_col in df.columns:
                cache_data = pd.concat([df[[id_col]], cache_data], axis=1)
            cache_data.to_csv(cache_path, index=False)
        except Exception:
            pass

    # Return normalized columns with optional id_col for stable merging
    if id_col is not None and id_col in df.columns:
        result_df = df[[id_col]].copy()
        result_df = pd.concat([result_df, norm_df[desired_cols]], axis=1)
        return result_df
    else:
        return norm_df[desired_cols]


def characterize_clusters(data, marker_cols, clusters=None, cluster_col=None):
    """Characterize clusters by computing mean marker expression & cell counts.

    Backward compatible enhancement: the original implementation required a
    ``clusters`` Series. The notebook (Step6A) calls this function with the
    signature ``characterize_clusters(data=df, cluster_col='leiden_cluster', marker_cols=norm_cols)``.
    To avoid breaking earlier code we now support EITHER:
        1. Passing a clusters Series via ``clusters`` (old style), OR
        2. Passing a column name via ``cluster_col`` (new convenience style).

    Parameters
    ----------
    data : pd.DataFrame
        Cell-level dataframe containing marker columns and (optionally) a cluster column.
    marker_cols : list[str]
        Marker columns to summarize (already normalized / prepared for characterization).
    clusters : pd.Series, optional
        Cluster assignments aligned to ``data.index``. Mutually exclusive with ``cluster_col``.
    cluster_col : str, optional
        Name of the column in ``data`` containing cluster IDs. Mutually exclusive with ``clusters``.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns:
            - <cluster_col> (or clusters.name / 'cluster_id')
            - cell_count
            - <marker>_mean for each marker in ``marker_cols``

    Raises
    ------
    ValueError
        If neither or both of ``clusters`` / ``cluster_col`` are provided, or inputs misaligned.
    """
    # Validate mutually exclusive parameters
    if (clusters is None and cluster_col is None) or (clusters is not None and cluster_col is not None):
        raise ValueError("Provide exactly one of 'clusters' (Series) or 'cluster_col' (str).")

    # Resolve clusters Series
    if cluster_col is not None:
        if cluster_col not in data.columns:
            raise ValueError(f"cluster_col '{cluster_col}' not found in data columns")
        clusters = data[cluster_col]
    else:
        # Ensure provided Series length/index alignment; reindex if needed
        if not clusters.index.equals(data.index):
            common_index = data.index.intersection(clusters.index)
            if len(common_index) == 0:
                raise ValueError("Data and clusters have no common indices.")
            print(f"Warning: Data and clusters indices differ. Aligning on {len(common_index)} common cells.")
            data = data.loc[common_index]
            clusters = clusters.loc[common_index]

    # Check marker columns
    missing_markers = [m for m in marker_cols if m not in data.columns]
    if missing_markers:
        raise ValueError(f"Marker columns missing for characterization: {missing_markers}")

    # Prepare processing frame
    df_proc = data[marker_cols].copy()
    cluster_col_name = cluster_col if cluster_col is not None else (clusters.name or 'cluster_id')
    df_proc[cluster_col_name] = clusters.values  # ensure alignment after any reindex

    # Compute per-cluster means
    mean_expr_df = df_proc.groupby(cluster_col_name, observed=True)[marker_cols].mean()
    mean_expr_df.columns = [f"{col}_mean" for col in mean_expr_df.columns]

    # Cell counts
    cell_counts = df_proc.groupby(cluster_col_name, observed=True).size().to_frame('cell_count')

    # Combine
    char_df = mean_expr_df.join(cell_counts).reset_index()
    return char_df


def create_cluster_heatmap(char_df, plots_dir, cluster_col, title="Cluster Expression Heatmap", filename="cluster_expression_heatmap.png"):
    """Generate clustered heatmap for cluster marker expression."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    marker_cols = [c for c in char_df.columns if c.endswith('_mean')]
    
    working = char_df.copy()
    heat = working.set_index(cluster_col)[marker_cols].copy()
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
    g.ax_heatmap.set_xlabel('Clusters')
    g.ax_heatmap.set_ylabel('Markers')
    g.ax_heatmap.yaxis.tick_right()
    g.ax_heatmap.yaxis.set_label_position('right')
    # relabel x with cell counts if available
    if 'cell_count' in working.columns:
        sizes = working.set_index(cluster_col)['cell_count']
        order = [int(t.get_text()) for t in g.ax_heatmap.get_xticklabels()]
        newlabs = [f"{cid}\n({sizes.get(cid,0):,})" for cid in order]
        g.ax_heatmap.set_xticklabels(newlabs, rotation=90)
    out_path = plots_dir / filename
    g.fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    # Return ordered cluster IDs in displayed order
    order_ids = [int(t.get_text().split('\n')[0]) for t in g.ax_heatmap.get_xticklabels()]
    return order_ids, out_path


def launch_phenotype_assignment_app(char_df, reordered_cluster_ids, meta_cluster_phenotypes):
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
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import seaborn as sns  # noqa: F401 (kept for parity & potential extension)
    from scipy.cluster.hierarchy import dendrogram, linkage

    class PhenotypeAssignmentApp:
        def __init__(self, char_df, reordered_cluster_ids, meta_cluster_phenotypes):
            self.root = tk.Tk()
            self.root.title("SPACEMAP - Phenotype Assignment Studio")
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
            self.heatmap_fig.tight_layout(pad=0.5, rect=[0.02, 0.05, 0.98, 0.95])
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

    print("="*80); print("LAUNCHING SPACEMAP - PHENOTYPE ASSIGNMENT STUDIO"); print("="*80)
    print("\nPROFESSIONAL FEATURES:\n  * Interactive cluster selection with visual feedback\n  * Red = assigned, Green = unassigned, Yellow = selected\n  * Real-time statistics & progress tracking\n  * Assignment editing & management\n  * Phenotype heatmap generation\n  * Professional control panel")
    app = PhenotypeAssignmentApp(char_df, reordered_cluster_ids, meta_cluster_phenotypes); app.run()
    print("Application closed!")
    if app.meta_cluster_phenotypes:
        print(f"Total assignments saved: {len(app.meta_cluster_phenotypes)}\nReady to proceed to Step 5!")
    else:
        print("No assignments were made.")
    return app.meta_cluster_phenotypes



def apply_phenotypes(data, cluster_col, phenotype_map, output_base_dir, method_prefix, 
                       cell_id_column, cluster_to_meta_map=None):
    """Apply cluster->phenotype mapping & persist unified + summary files.

    Only adds clustering-specific columns to the unified file, preserves existing data.
    
    Parameters
    ----------
    data : pd.DataFrame
    cluster_col : str
        Column name for the cluster IDs (e.g., 'som_cluster', 'leiden_cluster').
    phenotype_map : dict
        Mapping from cluster ID (or meta cluster ID) to phenotype label.
    output_base_dir : Path-like
        Root outputs directory.
    method_prefix : str
        Prefix for phenotype column (e.g., 'som', 'leiden').
    cell_id_column : str
        Name of the cell ID column for proper data merging (required parameter).
    cluster_to_meta_map : dict, optional
        Mapping from base cluster to meta cluster, if applicable (e.g., for SOM).
    """
    output_base_dir = Path(output_base_dir)
    phenotyping_dir = output_base_dir
    phenotyping_dir.mkdir(exist_ok=True)
    
    data = data.copy()
    phenotype_col_name = f'phenotype_{method_prefix}'

    if cluster_to_meta_map:
        # E.g., SOM workflow: som_cluster -> meta_cluster -> phenotype
        meta_cluster_col = 'meta_cluster' # temp column
        data[meta_cluster_col] = data[cluster_col].map(cluster_to_meta_map)
        data[phenotype_col_name] = data[meta_cluster_col].map(phenotype_map).fillna('Unassigned')
    else:
        # E.g., Leiden workflow: leiden_cluster -> phenotype
        # With the improved clustering approach, all cells should have valid cluster IDs
        data[phenotype_col_name] = data[cluster_col].map(phenotype_map).fillna('Unassigned')
        
        # Verify no invalid cluster assignments remain (quality check)
        invalid_clusters = data[cluster_col].isna().sum() + (data[cluster_col] < 0).sum()
        if invalid_clusters > 0:
            print(f"WARNING: Found {invalid_clusters} cells with invalid cluster assignments. "
                  f"These will be marked as 'Unassigned'. Consider re-running clustering.")

    summary = data.groupby(phenotype_col_name).agg(
        cell_count=(phenotype_col_name, 'count'),
        num_clusters=(cluster_col, 'nunique')
    )
    summary['percentage'] = (summary['cell_count'] / len(data) * 100).round(2)
    summary = summary.sort_values('cell_count', ascending=False)

    unified_path = phenotyping_dir / 'all_samples_combined_corrected_phenotyped.csv'
    
    # Define method-specific columns to add (no normalized markers)
    method_specific_cols = [cluster_col, phenotype_col_name]
    
    # Add embedding columns for Leiden method
    if method_prefix == 'leiden':
        embedding_cols = [c for c in ['umap_x', 'umap_y'] if c in data.columns]
        method_specific_cols.extend(embedding_cols)
    
    # Add meta_cluster for SOM method
    if cluster_to_meta_map and 'meta_cluster' in data.columns:
        method_specific_cols.append('meta_cluster')
    
    # Ensure configurable cell_id column is available for proper merging
    if cell_id_column not in data.columns:
        raise ValueError(f"'{cell_id_column}' column required for proper data merging")
    
    if unified_path.exists():
        existing = pd.read_csv(unified_path)
        if cell_id_column in existing.columns and len(existing) > 0:
            print(f"Merging {method_prefix} results with existing unified file...")
            
            # Use cell_id-based merge to properly anchor results
            existing = existing.set_index(cell_id_column)
            data_indexed = data.set_index(cell_id_column)
            
            # Only add/update method-specific columns
            for col in method_specific_cols:
                if col in data_indexed.columns:
                    existing[col] = data_indexed[col]
                    print(f" Added/updated column: {col}")
            
            # Reset index to get cell_id column back as a column (avoid duplicates)
            if cell_id_column in existing.columns:
                existing = existing.drop(columns=[cell_id_column])
            unified_df = existing.reset_index()
        else:
            print(f"Creating new unified file with {method_prefix} results...")
            # Filter data to only include essential columns (no normalized markers)
            essential_cols = [cell_id_column] + [c for c in data.columns if not c.endswith('_norm')] 
            unified_df = data[essential_cols].copy()
    else:
        print(f"Creating new unified file with {method_prefix} results...")
        # Filter data to only include essential columns (no normalized markers)  
        essential_cols = [cell_id_column] + [c for c in data.columns if not c.endswith('_norm')]
        unified_df = data[essential_cols].copy()
    
    unified_df.to_csv(unified_path, index=False)
    print(f"Unified file saved: {unified_path} ({len(unified_df):,} rows)")

    # Save summary and mapping files
    summary_file = phenotyping_dir / f'{method_prefix}_phenotype_summary.csv'
    summary.to_csv(summary_file)
    
    mapping_file = phenotyping_dir / f'{method_prefix}_phenotype_mapping.csv'
    pd.DataFrame(list(phenotype_map.items()), columns=['cluster_id', 'phenotype']).to_csv(mapping_file, index=False)
    
    if cluster_to_meta_map:
        som_meta_file = phenotyping_dir / 'som_to_meta_cluster_mapping.csv'
        pd.DataFrame(list(cluster_to_meta_map.items()), columns=['som_cluster', 'meta_cluster']).to_csv(som_meta_file, index=False)

    return {
        'unified_file': unified_path,
        'summary_file': summary_file,
        'mapping_file': mapping_file,
        'summary_df': summary,
        'data': data
    }


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
    """Create a density-aware geometric sketch of the dataset for HPO."""
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
    mat = np.nan_to_num(mat, nan=0.0)  # New: Explicitly handle any remaining NaNs
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


def _binned_sketch_selection(pca_coords, n_select, random_seed):
    """Coarse fallback sketch: per-PC binning plus a uniform top-up.

    Only used when geosketch is unavailable. It bins the first three PCs
    independently, which typically supplies a few thousand cells; the rest of
    the sketch is drawn uniformly at random, so rare populations are retained
    at roughly their background rate. Prefer geometric_sketch_selection.
    """
    import numpy as np

    np.random.seed(random_seed)
    n_cells, n_dims = pca_coords.shape

    if n_select >= n_cells:
        return np.arange(n_cells)

    # Start with random selection, then optimize for coverage
    # This is a simplified version of geometric sketching

    # Initialize with stratified random sampling across PCA space
    selected_indices = []
    
    # Divide each PC dimension into bins and sample from each
    n_bins_per_dim = max(2, int(np.sqrt(n_select)))
    
    for dim in range(min(3, n_dims)):  # Use first 3 PCs for binning
        pc_values = pca_coords[:, dim]
        bins = np.linspace(pc_values.min(), pc_values.max(), n_bins_per_dim + 1)
        
        for i in range(n_bins_per_dim):
            bin_mask = (pc_values >= bins[i]) & (pc_values < bins[i + 1])
            if i == n_bins_per_dim - 1:  # Include upper bound in last bin
                bin_mask = (pc_values >= bins[i]) & (pc_values <= bins[i + 1])
            
            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) > 0:
                # Sample from this bin
                n_from_bin = max(1, len(bin_indices) // n_bins_per_dim)
                sampled = np.random.choice(bin_indices, 
                                         size=min(n_from_bin, len(bin_indices)), 
                                         replace=False)
                selected_indices.extend(sampled)
    
    # Remove duplicates and random sample to exact size
    selected_indices = list(set(selected_indices))
    
    if len(selected_indices) > n_select:
        selected_indices = np.random.choice(selected_indices, size=n_select, replace=False)
    elif len(selected_indices) < n_select:
        # Fill remaining with random selection from unselected cells
        remaining_indices = np.setdiff1d(np.arange(n_cells), selected_indices)
        n_additional = n_select - len(selected_indices)
        if len(remaining_indices) > 0:
            additional = np.random.choice(remaining_indices,
                                        size=min(n_additional, len(remaining_indices)),
                                        replace=False)
            selected_indices.extend(additional)

    return np.array(selected_indices[:n_select])


def geometric_sketch_selection(pca_coords, n_select, random_seed, logger=None):
    """Density-aware sketch of PCA space (Hie et al. geometric sketching).

    Geosketch covers the *geometry* of the data rather than its density, so
    rare populations enter the sketch at far above their background rate. That
    matters here because anything missing from the sketch never gets its own
    Leiden cluster - the k-NN backfill can only assign such cells to whichever
    cluster happens to be nearest.

    Falls back to _binned_sketch_selection when geosketch is unavailable.
    """
    import numpy as np

    pca_coords = np.asarray(pca_coords)
    n_cells = pca_coords.shape[0]

    if n_select >= n_cells:
        return np.arange(n_cells)

    try:
        from spacemap.ph_accel_utils import geometric_sketch

        # Use every component handed in. Truncating here would discard the
        # very direction that separates a rare population, and the caller
        # already controls the width through the n_pcs setting.
        emb = np.ascontiguousarray(pca_coords, dtype=np.float64)
        selected = np.asarray(
            geometric_sketch(emb, int(n_select), random_seed=random_seed, logger=logger),
            dtype=int,
        )
        if selected.size:
            return selected
    except Exception as exc:
        message = f"Geometric sketching failed ({exc}); falling back to binned sketching."
        if logger is not None:
            logger.warning(message)
        else:
            print(f"Warning: {message}")

    return _binned_sketch_selection(pca_coords, n_select, random_seed)


def intelligent_leiden_clustering(df, norm_cols, leiden_params, logger):
    """
    Perform intelligent subsampling-based Leiden clustering for large datasets.
    
    Args:
        df: DataFrame with cell data
        norm_cols: List of normalized marker column names
        leiden_params: Dict with clustering parameters
        logger: Logger instance
    
    ``leiden_params['compute_umap']`` controls the UMAP step and defaults to
    True, so existing callers (including the notebooks) are unaffected. UMAP is
    only used for visualisation - it never affects the cluster assignments - and
    on large datasets it costs more than the clustering itself, so callers that
    do not need it can pass False and get None back for the coordinates.

    Returns:
        Tuple of (cluster_assignments, umap_coords or None)
    """
    import gc
    import numpy as np
    import scanpy as sc
    from datetime import datetime
    
    start_time = time.time()
    
    # Extract parameters
    n_pcs = leiden_params['n_pcs']
    n_neighbors = leiden_params['n_neighbors']
    resolution = leiden_params['resolution']
    subsample_size = leiden_params['subsample_size']
    random_seed = leiden_params['random_seed']
    
    logger.info("Creating AnnData object for intelligent subsampling...")
    obs_cols_to_include = ['batch_id', 'tissue_roi', 'x_centroid', 'y_centroid', 
                          'area', 'epithelia', 'stroma', 'tumor']
    obs_meta = df[[col for col in obs_cols_to_include if col in df.columns]].copy()
    
    try:
        # Create full AnnData object
        X = df[norm_cols].fillna(0).astype(float)
        adata_full = sc.AnnData(X=X.values, obs=obs_meta.copy())
        adata_full.var_names = norm_cols
        if 'label' in df.columns and df['label'].nunique() == len(df):
            adata_full.obs_names = df['label'].astype(str).values
        else:
            adata_full.obs_names_make_unique()
        logger.info(f"Full AnnData object created with shape: {adata_full.shape}")
        
        # Determine subsample size (respect user-configured value)
        base_sample_size = min(int(subsample_size), len(df))
        adaptive_sample_size = base_sample_size
        logger.info(
            "Using intelligent subsampling: %s cells (%.1f%% of dataset) [requested=%s]",
            f"{adaptive_sample_size:,}",
            adaptive_sample_size / len(df) * 100,
            f"{int(subsample_size):,}",
        )
        
        # STEP 1: GEOMETRIC SKETCHING FOR REPRESENTATIVE SUBSAMPLING
        logger.info("Step 1: Computing initial PCA for geometric sketching...")
        
        # Use a smaller initial sample for PCA computation
        initial_pca_size = min(50000, len(df))
        np.random.seed(random_seed)
        initial_idx = np.random.choice(len(df), initial_pca_size, replace=False)
        adata_init = adata_full[initial_idx].copy()
        
        # Compute PCA on initial sample
        max_pcs = min(adata_init.shape) - 1
        actual_n_pcs = min(n_pcs, max_pcs)
        if actual_n_pcs < n_pcs:
            logger.warning(f"Adjusted PCA dimensions: using {actual_n_pcs} PCs")
        
        sc.tl.pca(adata_init, n_comps=actual_n_pcs, svd_solver='randomized', random_state=random_seed)
        
        # Apply PCA transformation to full dataset
        logger.info("Applying PCA transformation to full dataset...")
        X_centered = adata_full.X - np.mean(adata_init.X, axis=0)
        adata_full.obsm['X_pca'] = X_centered @ adata_init.varm['PCs']
        
        # Geometric sketching: select cells that are well-distributed in PCA space
        logger.info("Performing geometric sketching for representative cell selection...")
        pca_coords = adata_full.obsm['X_pca']
        
        # Batch-aware stratified sampling if available
        if 'batch_id' in df.columns:
            logger.info("Using batch-aware stratified sampling...")
            batches = df['batch_id'].unique()
            subsample_indices = []
            cells_per_batch = adaptive_sample_size // len(batches)
            remainder = adaptive_sample_size % len(batches)
            
            for i, batch in enumerate(batches):
                batch_mask = df['batch_id'] == batch
                batch_indices = np.where(batch_mask)[0]
                
                if len(batch_indices) > 0:
                    batch_sample_size = cells_per_batch + (1 if i < remainder else 0)
                    batch_sample_size = min(batch_sample_size, len(batch_indices))
                    
                    if len(batch_indices) <= batch_sample_size:
                        subsample_indices.extend(batch_indices)
                    else:
                        batch_pca = pca_coords[batch_indices]
                        selected = geometric_sketch_selection(
                            batch_pca, batch_sample_size, random_seed, logger=logger
                        )
                        subsample_indices.extend(batch_indices[selected])

            subsample_indices = np.array(subsample_indices)
        else:
            logger.info("Using geometric sketching on full dataset...")
            subsample_indices = geometric_sketch_selection(
                pca_coords, adaptive_sample_size, random_seed, logger=logger
            )
        
        logger.info(f"Selected {len(subsample_indices):,} representative cells for clustering")
        
        # STEP 2: PERFORM CLUSTERING ON REPRESENTATIVE SUBSAMPLE
        logger.info("Step 2: Clustering representative subsample...")
        adata_subsample = adata_full[subsample_indices].copy()

        # Keep the projected PCA basis instead of refitting one on the subsample.
        # Step 3 extends labels with a k-NN fitted in adata_full's projected
        # space, so refitting here would define clusters in one basis and assign
        # the remaining cells in a different one, blurring cluster boundaries.
        adata_subsample.obsm['X_pca'] = np.asarray(adata_full.obsm['X_pca'])[subsample_indices]

        # Build neighborhood graph and perform clustering
        logger.info(f"Building k-NN graph ({n_neighbors} neighbors) on subsample...")
        sc.pp.neighbors(adata_subsample, n_neighbors=n_neighbors, n_pcs=actual_n_pcs,
                       use_rep='X_pca', random_state=random_seed, method='umap')
        
        logger.info(f"Running Leiden clustering (resolution={resolution}) on subsample...")
        sc.tl.leiden(adata_subsample, resolution=resolution, random_state=random_seed, 
                    key_added='leiden', n_iterations=10)
        
        n_clusters_subsample = adata_subsample.obs['leiden'].nunique()
        logger.info(f"Found {n_clusters_subsample} clusters in representative subsample")
        
        # STEP 3: ASSIGN ALL REMAINING CELLS USING k-NN IN PCA SPACE
        logger.info("Step 3: Assigning remaining cells using k-NN classification...")
        
        # Initialize cluster assignments for all cells
        all_cluster_assignments = np.full(len(df), -1, dtype=int)
        all_cluster_assignments[subsample_indices] = adata_subsample.obs['leiden'].astype(int).values
        
        # Find unassigned cells
        unassigned_mask = all_cluster_assignments == -1
        n_unassigned = unassigned_mask.sum()
        
        if n_unassigned > 0:
            logger.info(f"Assigning {n_unassigned:,} remaining cells using k-NN classification...")
            
            # Prepare training data (subsample) and test data (unassigned)
            X_train = adata_full.obsm['X_pca'][subsample_indices]
            y_train = adata_subsample.obs['leiden'].astype(int).values
            X_test = adata_full.obsm['X_pca'][unassigned_mask]

            # Distance-weighted k-NN transfer. Runs on the GPU when one is
            # available (exact brute-force search, same result as sklearn) and
            # falls back to KNeighborsClassifier otherwise.
            from spacemap.ph_accel_utils import knn_backfill

            y_pred = knn_backfill(
                X_train,
                y_train,
                X_test,
                knn_neighbors=min(15, len(X_train)),
                use_gpu=leiden_params.get('use_gpu'),
                logger=logger,
            )
            all_cluster_assignments[unassigned_mask] = y_pred
            
            logger.info(f"Successfully assigned all {n_unassigned:,} remaining cells")
        
        # Verify no cells remain unassigned
        still_unassigned = (all_cluster_assignments == -1).sum()
        if still_unassigned > 0:
            logger.error(f"CRITICAL ERROR: {still_unassigned} cells remain unassigned!")
            raise ValueError(f"Clustering failed - {still_unassigned} cells unassigned")

        # Optional merge into a fixed number of meta-clusters, so Leiden can be
        # brought to the same granularity as the other phenotyping methods
        # before per-cluster phenotype assignment.
        meta_clusters = int(leiden_params.get('meta_clusters', 0) or 0)
        if meta_clusters >= 2:
            from spacemap.ph_accel_utils import meta_cluster_labels

            merge_matrix = adata_full.obsm['X_pca'] if 'X_pca' in adata_full.obsm else adata_full.X
            all_cluster_assignments, _mapping = meta_cluster_labels(
                all_cluster_assignments,
                np.asarray(merge_matrix),
                meta_clusters,
                logger=logger,
            )
        
        # STEP 4: COMPUTE UMAP FOR VISUALIZATION (optional)
        if not leiden_params.get('compute_umap', True):
            logger.info(
                "Step 4: skipping UMAP (visualisation only; enable 'Compute UMAP embedding' to generate it)."
            )
            return all_cluster_assignments, None

        logger.info("Step 4: Computing UMAP embedding for visualization...")

        # Compute UMAP on subsample
        sc.tl.umap(adata_subsample, random_state=random_seed)
        
        # Initialize UMAP coordinates for all cells
        all_umap_coords = np.full((len(df), 2), np.nan)
        all_umap_coords[subsample_indices] = adata_subsample.obsm['X_umap']
        
        # FIXED: Assign UMAP coordinates to ALL remaining cells using simple cluster means
        n_missing_umap = np.isnan(all_umap_coords[:, 0]).sum()
        
        if n_missing_umap > 0:
            logger.info(f"Assigning UMAP coordinates to {n_missing_umap:,} cells using cluster centroids...")
            
            # Simple approach: Calculate cluster centroids directly from existing UMAP coordinates
            # Get the subsample UMAP coordinates and their cluster assignments
            subsample_umap = np.array(adata_subsample.obsm['X_umap'])  # Convert to numpy array
            subsample_clusters = np.array(adata_subsample.obs['leiden'].astype(int))  # Convert to numpy array
            
            logger.info(f"Computing centroids from {len(subsample_umap)} subsample cells with {len(np.unique(subsample_clusters))} clusters")
            
            # Calculate cluster centroids using numpy operations (more robust than pandas)
            unique_clusters = np.unique(subsample_clusters)
            cluster_centroids = {}
            
            for cluster_id in unique_clusters:
                cluster_mask = subsample_clusters == cluster_id
                cluster_coords = subsample_umap[cluster_mask]
                centroid = np.mean(cluster_coords, axis=0)
                cluster_centroids[cluster_id] = centroid
                logger.info(f"Cluster {cluster_id}: {cluster_mask.sum()} cells, centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}]")
            
            logger.info(f"Successfully computed {len(cluster_centroids)} cluster centroids")
            
            # Assign coordinates to missing cells based on their cluster assignments
            assigned_count = 0
            for cluster_id, centroid in cluster_centroids.items():
                # Find cells assigned to this cluster that are missing UMAP coordinates
                cluster_mask = all_cluster_assignments == cluster_id
                missing_umap_mask = cluster_mask & np.isnan(all_umap_coords[:, 0])
                n_to_assign = missing_umap_mask.sum()
                
                if n_to_assign > 0:
                    all_umap_coords[missing_umap_mask] = centroid
                    assigned_count += n_to_assign
                    logger.info(f"Assigned {n_to_assign:,} cells to cluster {cluster_id} centroid")
            
            # Verify assignment worked
            n_still_missing = np.isnan(all_umap_coords[:, 0]).sum()
            logger.info(f"Successfully assigned UMAP coordinates to {assigned_count:,} cells")
            logger.info(f"Still missing UMAP coordinates: {n_still_missing:,} cells")
        
        # Clean up intermediate objects
        del adata_init, adata_subsample
        gc.collect()
        
        # Final summary
        total_duration = time.time() - start_time
        end_time = datetime.now()
        logger.info(f"=== LEIDEN CLUSTERING FINISHED ===")
        logger.info(f"Total time: {total_duration/60:.1f} minutes ({total_duration:.1f}s)")
        logger.info(f"Finished at: {end_time.strftime('%H:%M:%S')}")
        logger.info(f"Final clusters: {len(np.unique(all_cluster_assignments))}")
        logger.info(f"Cells processed: {len(df):,}")
        logger.info(f"Cluster range: {all_cluster_assignments.min()} to {all_cluster_assignments.max()}")
        
        return all_cluster_assignments, all_umap_coords
        
    except Exception as e:
        logger.error(f"ERROR in clustering pipeline: {e}")
        raise
    finally:
        # Clean up
        if 'adata_full' in locals():
            del adata_full
        gc.collect()


# =============================================================================
# FEATURE VARIABILITY DISCOVERY
# =============================================================================

def apply_filter_for_variability(df, filter_criteria):
    """
    Apply filtering with OR/AND logic for variability analysis.
    Same logic as advanced filtering but simplified for variability discovery.
    """
    if not filter_criteria:
        return df
    
    original_count = len(df)
    or_conditions = filter_criteria.pop('OR_CONDITIONS', None)
    and_conditions = {k: v for k, v in filter_criteria.items() if v is not None}
    
    result_df = df.copy()
    
    # Apply OR conditions
    if or_conditions:
        or_masks = []
        for condition in or_conditions:
            condition_mask = pd.Series([True] * len(result_df), index=result_df.index)
            for column, value in condition.items():
                if column not in result_df.columns:
                    condition_mask = pd.Series([False] * len(result_df), index=result_df.index)
                    break
                if isinstance(value, list):
                    col_mask = result_df[column].isin(value)
                else:
                    col_mask = result_df[column] == value
                condition_mask = condition_mask & col_mask
            or_masks.append(condition_mask)
        
        if or_masks:
            combined_or_mask = or_masks[0]
            for mask in or_masks[1:]:
                combined_or_mask = combined_or_mask | mask
            result_df = result_df[combined_or_mask].copy()
    
    # Apply AND conditions
    if and_conditions:
        for column, value in and_conditions.items():
            if column not in result_df.columns:
                continue
            if isinstance(value, list):
                result_df = result_df[result_df[column].isin(value)].copy()
            else:
                result_df = result_df[result_df[column] == value].copy()
    
    return result_df


def discover_variable_features(df, filter_criteria, group_column, group_values, 
                               corr_markers, non_corr_markers, output_dir, 
                               normalization_method='arcsinh', cofactor=5):
    """
    Discover the most variable markers within filtered cell populations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with all cells
    filter_criteria : dict
        Base filtering criteria (AND conditions applied to all groups)
    group_column : str
        Column name to split analysis by (e.g., 'phenotype_ml_mr')
    group_values : list
        List of values in group_column to analyze separately
    corr_markers : list
        List of corrected marker names (will add _corr suffix)
    non_corr_markers : list
        List of non-corrected markers (used as-is)
    output_dir : Path
        Directory to save outputs
    normalization_method : str
        'arcsinh', 'zscore', or 'none'
    cofactor : float
        Cofactor for arcsinh transformation
        
    Returns
    -------
    dict
        Dictionary with group names as keys and variability DataFrames as values
    """
    from joblib import Parallel, delayed
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare marker lists
    all_markers = [f"{m}_corr" for m in corr_markers] + non_corr_markers
    
    # Apply base filters
    df_filtered = apply_filter_for_variability(df, filter_criteria.copy())
    print(f"Base filtering: {len(df_filtered):,}/{len(df):,} cells retained")
    
    results = {}
    
    for group_val in group_values:
        print(f"\n{'='*60}")
        print(f"Analyzing variability for: {group_column} == {group_val}")
        print(f"{'='*60}")
        
        # Filter to specific group
        df_group = df_filtered[df_filtered[group_column] == group_val].copy()
        print(f"Group cells: {len(df_group):,}")
        
        if len(df_group) < 10:
            print(f"Warning: Too few cells ({len(df_group)}) for {group_val}. Skipping.")
            continue
        
        # Check which markers are available
        available_markers = [m for m in all_markers if m in df_group.columns]
        if not available_markers:
            print(f"Warning: No markers found for {group_val}. Skipping.")
            continue
        
        print(f"Available markers: {len(available_markers)}")
        
        # Normalize markers
        df_norm = df_group.copy()
        
        if normalization_method == 'arcsinh':
            # Arcsinh transformation for corrected markers
            corr_available = [m for m in available_markers if m.endswith('_corr')]
            if corr_available:
                df_norm[corr_available] = np.arcsinh(df_norm[corr_available].values / cofactor)
            
            # Z-score for non-corrected markers
            non_corr_available = [m for m in available_markers if not m.endswith('_corr')]
            if non_corr_available:
                scaler = StandardScaler()
                df_norm[non_corr_available] = scaler.fit_transform(df_norm[non_corr_available].values)
                
        elif normalization_method == 'zscore':
            scaler = StandardScaler()
            df_norm[available_markers] = scaler.fit_transform(df_norm[available_markers].values)
        
        # Compute variability (standard deviation) in parallel
        def compute_std(feat):
            vals = df_norm[feat].dropna().values
            return np.std(vals) if len(vals) > 1 else 0
        
        print(f"Computing variability across {len(available_markers)} markers...")
        variability = Parallel(n_jobs=-1)(delayed(compute_std)(feat) for feat in available_markers)
        
        # Create results dataframe
        var_df = pd.DataFrame({
            'marker': available_markers,
            'std_dev': variability,
            'group': group_val,
            'n_cells': len(df_group)
        })
        var_df = var_df.sort_values('std_dev', ascending=False)
        
        # Save results
        safe_name = str(group_val).replace(' ', '_').replace('/', '_')
        csv_path = output_dir / f'variability_{group_column}_{safe_name}.csv'
        var_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        
        # Store results
        results[group_val] = var_df
        
        # Display top markers
        print(f"\nTop 10 most variable markers in {group_val}:")
        print(var_df[['marker', 'std_dev']].head(10).to_string(index=False))
    
    return results


def plot_variability_results(variability_results, output_dir, top_n=20):
    """
    Generate plots for feature variability analysis.
    
    Parameters
    ----------
    variability_results : dict
        Dictionary from discover_variable_features()
    output_dir : Path
        Directory to save plots
    top_n : int
        Number of top features to plot
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'variability_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for group_name, var_df in variability_results.items():
        print(f"\nGenerating plots for: {group_name}")
        
        # Clean group name for filenames
        safe_name = str(group_name).replace(' ', '_').replace('/', '_')
        
        # 1. Bar plot of top variable markers
        top_markers = var_df.head(top_n).copy()
        
        plt.figure(figsize=(12, max(8, top_n * 0.4)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_markers)))
        plt.barh(range(len(top_markers)), top_markers['std_dev'].values, color=colors)
        plt.yticks(range(len(top_markers)), top_markers['marker'].values)
        plt.xlabel('Standard Deviation (Variability)', fontsize=12)
        plt.ylabel('Marker', fontsize=12)
        plt.title(f'Top {top_n} Most Variable Markers\n{group_name} (n={var_df["n_cells"].iloc[0]:,} cells)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        bar_path = plots_dir / f'variability_bar_{safe_name}.png'
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {bar_path}")
        
        # 2. Waterfall plot showing all markers
        plt.figure(figsize=(14, 8))
        plt.plot(range(len(var_df)), var_df['std_dev'].values, 'o-', linewidth=2, markersize=6)
        
        # Annotate top 10 markers
        for i, row in var_df.head(10).iterrows():
            idx = var_df.index.get_loc(i)
            plt.annotate(row['marker'], 
                        xy=(idx, row['std_dev']),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=8, rotation=15,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Marker Rank', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title(f'Marker Variability Ranking\n{group_name} (n={var_df["n_cells"].iloc[0]:,} cells)',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        waterfall_path = plots_dir / f'variability_waterfall_{safe_name}.png'
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {waterfall_path}")


def plot_combined_variability_comparison(variability_results, output_dir, top_n=15):
    """
    Create comparative plots across all groups.
    
    Parameters
    ----------
    variability_results : dict
        Dictionary from discover_variable_features()
    output_dir : Path
        Directory to save plots
    top_n : int
        Number of top features to compare
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'variability_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if len(variability_results) < 2:
        print("Need at least 2 groups for comparison plots")
        return
    
    # Combine all results
    all_data = []
    for group_name, var_df in variability_results.items():
        df_temp = var_df.copy()
        df_temp['group'] = str(group_name)
        all_data.append(df_temp)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 1. Heatmap of top markers across groups
    # Get union of top markers from each group
    top_markers_set = set()
    for var_df in variability_results.values():
        top_markers_set.update(var_df.head(top_n)['marker'].tolist())
    
    # Create pivot table
    pivot_data = combined_df[combined_df['marker'].isin(top_markers_set)].pivot(
        index='marker', columns='group', values='std_dev'
    )
    
    plt.figure(figsize=(max(10, len(variability_results) * 2), max(8, len(top_markers_set) * 0.3)))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', 
               cbar_kws={'label': 'Standard Deviation'})
    plt.title(f'Marker Variability Comparison Across Groups\n(Top {top_n} markers from each group)',
             fontsize=14, fontweight='bold')
    plt.xlabel('Group', fontsize=12)
    plt.ylabel('Marker', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = plots_dir / 'variability_comparison_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison heatmap saved: {heatmap_path}")
    
    # 2. Side-by-side bar plot comparison
    fig, axes = plt.subplots(1, len(variability_results), 
                             figsize=(6 * len(variability_results), 10),
                             sharey=True)
    
    if len(variability_results) == 1:
        axes = [axes]
    
    for ax, (group_name, var_df) in zip(axes, variability_results.items()):
        top_markers = var_df.head(top_n).copy()
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_markers)))
        
        ax.barh(range(len(top_markers)), top_markers['std_dev'].values, color=colors)
        ax.set_yticks(range(len(top_markers)))
        ax.set_yticklabels(top_markers['marker'].values)
        ax.set_xlabel('Std Dev', fontsize=10)
        ax.set_title(f'{group_name}\n(n={var_df["n_cells"].iloc[0]:,})', 
                    fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    axes[0].set_ylabel('Marker', fontsize=12)
    plt.suptitle(f'Top {top_n} Variable Markers Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = plots_dir / 'variability_comparison_bars.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison bars saved: {comparison_path}")






