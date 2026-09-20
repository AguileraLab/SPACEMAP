"""ph_subpheno_utils.py

Objective sub-phenotyping: find which markers actually carry structure inside a
single parent phenotype, cluster on those markers only, and keep the subclusters
that survive a reproducibility check.

The problem this solves
-----------------------
Re-standardising every marker inside one phenotype rescales flat markers to unit
variance, which turns their measurement noise into apparent signal. A marker that
is uniformly negative in that phenotype (because it belongs to a different
lineage) contributes as much to the distance metric as a marker with real graded
biology, and the clustering then partitions noise.

The fix is to choose the feature space per phenotype, from the data:

1. ``score_markers`` measures, for every marker, how much cells of one phenotype
   genuinely differ from each other in it, after removing batch effects.
2. ``recommend_markers`` turns those numbers into a keep/drop call.
3. ``build_feature_matrix`` standardises only the surviving markers.
4. ``subcluster`` runs Leiden, SOM or Vortex on that matrix, GPU-accelerated
   where a CUDA device is present.
5. ``validate_subclusters`` reports whether each subcluster reproduces across
   batches and is separated from its nearest sibling.

Empirically (2.8M cells, 44 batches, 35 markers) the markers that collapse are
the ones belonging to *other* lineages -- not the markers used to gate the
phenotype. Gating markers frequently carry the functional axis itself
(HLA-DR/CD163 within Macrophage, aSMA/FN1/Podoplanin within Fibroblast), so they
must never be dropped on the grounds of having been used for phenotyping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .ph_accel_utils import (
        GPU_AVAILABLE,
        gpu_device_name,
        geometric_sketch,
        knn_backfill,
        knn_gpu,
        meta_cluster_labels,
        resolve_use_gpu,
    )
except ImportError:  # direct-module import (tests, notebooks)
    from ph_accel_utils import (  # type: ignore[no-redef]
        GPU_AVAILABLE,
        gpu_device_name,
        geometric_sketch,
        knn_backfill,
        knn_gpu,
        meta_cluster_labels,
        resolve_use_gpu,
    )

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency of the GUI
    torch = None

__all__ = [
    "DEFAULT_EXCLUDED_MARKERS",
    "COLLAPSED_SD_RATIO",
    "ENRICHED_SD_RATIO",
    "MAX_BATCH_ETA2",
    "BIMODAL_BC",
    "LINEAGE_SPECIFICITY",
    "HOME_GAP",
    "marker_reference_stats",
    "score_markers",
    "recommend_markers",
    "build_feature_matrix",
    "subcluster",
    "validate_subclusters",
    "cluster_stability",
    "CLUSTERING_METHODS",
    "gpu_status",
]

# Markers never worth clustering on. DAPI is a nuclear counterstain, not biology,
# and it is the single most batch-driven channel in practice.
DEFAULT_EXCLUDED_MARKERS = ("DAPI",)

# A marker whose spread inside the phenotype is below this fraction of its
# spread across all cells has effectively collapsed: standardising it would
# amplify noise.
COLLAPSED_SD_RATIO = 0.40

# At or above this, the marker varies *more* inside the phenotype than across the
# whole dataset -- the signature of a real within-phenotype axis.
ENRICHED_SD_RATIO = 1.00

# Fraction of a marker's variance explained by batch, above which its spread is
# a batch artefact rather than biology.
MAX_BATCH_ETA2 = 0.25

# Sarle's bimodality coefficient. Above this the distribution is not a single
# mode, i.e. the marker splits the population rather than grading it.
BIMODAL_BC = 0.555

# How concentrated a marker's expression is across phenotypes, in global SD units:
# (highest phenotype mean - median phenotype mean). Lineage markers score high
# (CD20 4.7, CD8 4.2, CD3e 2.3); markers every cell type carries at some rate
# score low (Ki67 1.1, PCNA 1.0, gH2AX 0.3). Only the former can plausibly be
# spillover from a neighbouring cell.
LINEAGE_SPECIFICITY = 1.5

# How far below its home phenotype this marker sits here, in global SD units.
HOME_GAP = 1.0

CLUSTERING_METHODS = ("Leiden", "SOM", "Vortex")


def gpu_status() -> tuple[bool, str]:
    """(available, human-readable description) for the CUDA device."""
    if not GPU_AVAILABLE:
        return False, "No CUDA device detected - running on CPU."
    return True, f"GPU acceleration available: {gpu_device_name()}"


def _log(logger, message: str) -> None:
    if logger is not None:
        logger.info(message)


# ---------------------------------------------------------------------------
# 1. Marker discovery
# ---------------------------------------------------------------------------


def marker_reference_stats(
    df: pd.DataFrame, marker_cols: list[str], phenotype_col: str
) -> dict:
    """Dataset-wide reference numbers needed to score any single phenotype.

    Returns the global SD per marker (the yardstick for `sd_ratio`) and the
    per-phenotype mean of every marker, which is what lets `score_markers`
    recognise a marker as foreign to the phenotype being examined.
    """
    values = df[marker_cols]
    global_sd = values.std()
    phenotype_means = values.groupby(df[phenotype_col].astype(str)).mean()
    return {
        "global_sd": global_sd,
        "phenotype_means": phenotype_means,
        # Concentrated across phenotypes = a lineage marker; flat = a marker every
        # cell type carries. Only the first kind can be spillover.
        "lineage_specificity": (phenotype_means.max() - phenotype_means.median()) / global_sd,
        "phenotype_max": phenotype_means.max(),
    }


def score_markers(
    subset: pd.DataFrame,
    marker_cols: list[str],
    *,
    reference: dict | None = None,
    batch_col: str | None = None,
    phenotype: str | None = None,
) -> pd.DataFrame:
    """Score every marker by the structure it carries inside `subset`.

    `subset` must already be restricted to the cells of one parent phenotype.
    All statistics are computed in two vectorised passes so this stays usable on
    hundreds of thousands of cells.

    Columns returned
    ----------------
    mean, sd_in         mean and SD of the marker inside this phenotype
    sd_ratio            sd_in / SD across the whole dataset. < COLLAPSED_SD_RATIO
                        means the marker is flat here; >= ENRICHED_SD_RATIO means
                        it varies more here than anywhere else.
    batch_eta2          fraction of the spread explained by batch
    sd_bio              spread left after removing batch differences
    bimodality          Sarle's coefficient; > BIMODAL_BC = splits rather than grades
    pct_positive        fraction of cells above this marker's dataset-wide median
    foreign             the phenotype sits in the lower half of this marker's
                        per-phenotype means, i.e. the marker belongs elsewhere
    score               sd_bio * (1 - batch_eta2): batch-free usable spread
    """
    values = subset[marker_cols].astype("float32")
    n = len(values)
    if n < 4:
        raise ValueError("Need at least 4 cells to score markers.")

    grand = values.mean()
    sd_in = values.std()

    if batch_col and batch_col in subset.columns:
        keys = subset[batch_col].astype(str).values
        group_means = values.groupby(keys).mean()
        group_sizes = pd.Series(keys).value_counts().sort_index()
        group_means = group_means.sort_index()
        ss_between = (group_means.sub(grand, axis=1) ** 2).mul(group_sizes.values, axis=0).sum()
        ss_total = ((values - grand) ** 2).sum()
        eta2 = (ss_between / ss_total.replace(0, np.nan)).clip(0, 1).fillna(0.0)
        dof = max(n - len(group_sizes), 1)
        sd_bio = np.sqrt((ss_total - ss_between).clip(lower=0) / dof)
    else:
        eta2 = pd.Series(0.0, index=marker_cols)
        sd_bio = sd_in

    # Sarle's bimodality coefficient from the moment definitions.
    skew, kurt = values.skew(), values.kurt()
    bimodality = (skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))

    reference = reference or {}
    global_sd = reference.get("global_sd")
    sd_ratio = (sd_in / global_sd).replace([np.inf, -np.inf], np.nan) if global_sd is not None else pd.Series(np.nan, index=marker_cols)

    pheno_means = reference.get("phenotype_means")
    if pheno_means is not None and phenotype is not None and phenotype in pheno_means.index:
        # Rank 1 = the phenotype with the highest mean for that marker.
        ranks = pheno_means.rank(ascending=False, axis=0)
        n_phenos = len(pheno_means)
        foreign = (ranks.loc[phenotype] > n_phenos / 2.0).reindex(marker_cols).fillna(False)
    else:
        foreign = pd.Series(False, index=marker_cols)

    if pheno_means is not None:
        # Median across phenotypes approximates the dataset-wide centre without
        # re-reading every cell.
        centre = pheno_means.median()
        pct_positive = (values > centre).mean()
    else:
        pct_positive = pd.Series(np.nan, index=marker_cols)

    lineage_spec = reference.get("lineage_specificity")
    lineage_spec = pd.Series(lineage_spec).reindex(marker_cols) if lineage_spec is not None else pd.Series(np.nan, index=marker_cols)
    phenotype_max = reference.get("phenotype_max")
    if phenotype_max is not None and global_sd is not None:
        home_gap = ((pd.Series(phenotype_max).reindex(marker_cols) - grand.reindex(marker_cols))
                    / pd.Series(global_sd).reindex(marker_cols))
    else:
        home_gap = pd.Series(np.nan, index=marker_cols)

    scores = pd.DataFrame(
        {
            "marker": [str(c) for c in marker_cols],
            "mean": grand.values,
            "sd_in": sd_in.values,
            "sd_ratio": pd.Series(sd_ratio).reindex(marker_cols).values,
            "batch_eta2": pd.Series(eta2).reindex(marker_cols).values,
            "sd_bio": pd.Series(sd_bio).reindex(marker_cols).values,
            "bimodality": bimodality.values,
            "pct_positive": pd.Series(pct_positive).reindex(marker_cols).values,
            "foreign": pd.Series(foreign).reindex(marker_cols).fillna(False).values,
            "lineage_spec": lineage_spec.values,
            "home_gap": home_gap.values,
        }
    )
    scores["score"] = scores.sd_bio * (1.0 - scores.batch_eta2.fillna(0.0))
    return scores.sort_values("score", ascending=False).reset_index(drop=True)


def recommend_markers(
    scores: pd.DataFrame,
    *,
    excluded: tuple[str, ...] = DEFAULT_EXCLUDED_MARKERS,
    collapsed_below: float = COLLAPSED_SD_RATIO,
    enriched_at: float = ENRICHED_SD_RATIO,
    max_batch_eta2: float = MAX_BATCH_ETA2,
    lineage_specificity: float = LINEAGE_SPECIFICITY,
    home_gap: float = HOME_GAP,
    max_markers: int | None = None,
) -> pd.DataFrame:
    """Turn marker scores into a keep/drop call with a stated reason.

    Adds `verdict` (enriched / graded / collapsed), `flag` (why a marker was
    rejected, if it was) and `recommended`. Ordering of the checks matters: a
    marker is only kept once it has passed all of them.

    Deliberately absent: any rule about whether the marker was used to define the
    phenotype. Gating markers routinely carry the functional axis inside their own
    phenotype, so excluding them on principle discards real biology.
    """
    out = scores.copy()
    base = out.marker.str.replace(r"_corr$", "", regex=True)
    excluded_lower = {str(name).strip().lower() for name in excluded}

    ratio = out.sd_ratio.fillna(1.0)
    out["verdict"] = np.where(
        ratio >= enriched_at, "enriched",
        np.where(ratio < collapsed_below, "collapsed", "graded"),
    )

    flag = pd.Series("", index=out.index, dtype=object)
    flag[base.str.strip().str.lower().isin(excluded_lower)] = "excluded marker"
    flag[(flag == "") & (out.verdict == "collapsed")] = "flat in this phenotype"
    flag[(flag == "") & (out.batch_eta2.fillna(0) > max_batch_eta2)] = "batch-driven"
    # Spillover looks like this: a marker restricted to some other lineage, sitting
    # far below its home phenotype here, yet switching cleanly on and off. All
    # three must hold. Requiring lineage specificity is what stops universal
    # functional markers -- Ki67, PCNA, gH2AX -- from being mistaken for
    # contamination just because this phenotype cycles less than others.
    spillover = (
        (flag == "")
        & (out.lineage_spec.fillna(0) >= lineage_specificity)
        & (out.home_gap.fillna(0) >= home_gap)
        & (out.bimodality.fillna(0) > BIMODAL_BC)
    )
    flag[spillover] = "possible spillover"
    out["flag"] = flag

    out["recommended"] = out.flag == ""
    if max_markers is not None:
        keep = out.index[out.recommended][:max_markers]
        out["recommended"] = out.index.isin(keep)
        out.loc[(out.flag == "") & ~out.recommended, "flag"] = "below marker cap"
    return out


# ---------------------------------------------------------------------------
# 2. Feature space
# ---------------------------------------------------------------------------


def build_feature_matrix(
    subset: pd.DataFrame,
    marker_cols: list[str],
    *,
    batch_col: str | None = None,
    clip: float | None = 5.0,
) -> np.ndarray:
    """Standardise the chosen markers for clustering.

    Values are z-scored within batch when a batch column is given, so a residual
    per-batch offset cannot masquerade as a subcluster. No inverse-arcsinh is
    applied: re-expanding to the raw scale restores the heavy tail and lets a
    handful of bright cells dominate the distance metric, which is a display
    choice rather than a modelling one.
    """
    values = subset[marker_cols].astype("float32")

    if batch_col and batch_col in subset.columns:
        keys = subset[batch_col].astype(str).values
        grouped = values.groupby(keys)
        centred = values - grouped.transform("mean")
        spread = grouped.transform("std").replace(0, np.nan)
        scaled = centred / spread
    else:
        spread = values.std().replace(0, np.nan)
        scaled = (values - values.mean()) / spread

    matrix = scaled.to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if clip:
        np.clip(matrix, -abs(clip), abs(clip), out=matrix)
    return matrix


def _pca(matrix: np.ndarray, n_components: int, use_gpu: bool, seed: int) -> np.ndarray:
    """PCA on the GPU when available, otherwise scikit-learn."""
    n_components = int(max(2, min(n_components, min(matrix.shape) - 1)))
    if use_gpu and torch is not None:
        try:
            device = torch.device("cuda:0")
            tensor = torch.as_tensor(matrix, device=device, dtype=torch.float32)
            tensor = tensor - tensor.mean(0, keepdim=True)
            # q slightly above n_components improves the randomised estimate.
            _u, _s, v = torch.pca_lowrank(tensor, q=min(n_components + 10, min(tensor.shape)))
            out = (tensor @ v[:, :n_components]).cpu().numpy()
            del tensor, v
            torch.cuda.empty_cache()
            return np.ascontiguousarray(out)
        except Exception:
            pass
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=seed).fit_transform(matrix)


def _knn(matrix: np.ndarray, k: int, use_gpu: bool, device: int = 0):
    """Exact k-NN, GPU-first. Returns (indices, distances) without the self-match."""
    k = int(min(k, len(matrix) - 1))
    if use_gpu:
        try:
            sq_dist, indices, _ref = knn_gpu(matrix, matrix, k + 1, device=device)
            dist = sq_dist.clamp_min(0).sqrt()
            out = indices[:, 1:].cpu().numpy(), dist[:, 1:].cpu().numpy()
            del sq_dist, indices, dist
            if torch is not None:
                torch.cuda.empty_cache()
            return out
        except Exception:
            pass
    from sklearn.neighbors import NearestNeighbors

    finder = NearestNeighbors(n_neighbors=k + 1).fit(matrix)
    dist, indices = finder.kneighbors(matrix)
    return indices[:, 1:], dist[:, 1:]


# ---------------------------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------------------------


def _leiden(matrix: np.ndarray, params: dict, use_gpu: bool, seed: int, logger) -> np.ndarray:
    import igraph as ig
    import leidenalg

    n_neighbors = int(params.get("n_neighbors", 15))
    resolution = float(params.get("resolution", 1.0))
    coords = _pca(matrix, int(params.get("n_pcs", 20)), use_gpu, seed)
    _log(logger, f"Leiden: PCA -> {coords.shape[1]} components, building {n_neighbors}-NN graph")
    indices, distances = _knn(coords, n_neighbors, use_gpu)

    edges, weights = _shared_neighbor_graph(indices, float(params.get("min_jaccard", 0.0625)))
    graph = ig.Graph(n=indices.shape[0], edges=edges, directed=False)
    graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights="weight",
        seed=int(seed),
    )
    return np.asarray(partition.membership, dtype=int)


def _shared_neighbor_graph(indices: np.ndarray, min_jaccard: float = 0.0625):
    """Weight each k-NN edge by how much the two cells' neighbourhoods overlap.

    This is the PhenoGraph/Seurat construction. A raw k-NN graph gives almost
    every edge the same strength, so Leiden carves it into many equal-sized blobs
    regardless of resolution; Jaccard weighting makes edges within a genuine
    community much stronger than edges bridging two, which is what lets the
    resolution parameter behave meaningfully on cytometry data.
    """
    from scipy.sparse import csr_matrix

    n, k = indices.shape
    rows = np.repeat(np.arange(n), k)
    adjacency = csr_matrix(
        (np.ones(n * k, dtype=np.float32), (rows, indices.ravel())), shape=(n, n)
    )
    adjacency = adjacency.maximum(adjacency.T)

    intersection = (adjacency @ adjacency.T).tocoo()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    union = degree[intersection.row] + degree[intersection.col] - intersection.data
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection.data / union, 0.0)

    keep = (intersection.row < intersection.col) & (jaccard >= min_jaccard)
    edges = list(zip(intersection.row[keep].tolist(), intersection.col[keep].tolist()))
    return edges, jaccard[keep].tolist()


def _som(matrix: np.ndarray, params: dict, use_gpu: bool, seed: int, logger) -> np.ndarray:
    """Self-organising map, then merge the nodes into the requested number of clusters.

    Over-clustering onto a grid and merging afterwards is more stable than trying
    to hit a cluster count directly, which is why the SOM path is offered here.
    """
    from minisom import MiniSom

    grid = int(params.get("grid", 10))
    iterations = int(params.get("iterations", 10000))

    rng = np.random.default_rng(seed)
    train = matrix
    max_train = int(params.get("train_size", 100000))
    if len(train) > max_train:
        train = train[rng.choice(len(train), max_train, replace=False)]

    som = MiniSom(grid, grid, matrix.shape[1], sigma=1.0, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(train)
    _log(logger, f"SOM: training {grid}x{grid} grid for {iterations:,} iterations")
    som.train_random(train, iterations, verbose=False)

    weights = som.get_weights().reshape(-1, matrix.shape[1]).astype(np.float32)
    # Assigning every cell to its best-matching unit is a 1-NN query, which is
    # exactly what the GPU path accelerates.
    if use_gpu:
        try:
            _d, indices, _ref = knn_gpu(weights, matrix, 1, device=0)
            nodes = indices[:, 0].cpu().numpy()
        except Exception:
            nodes = np.array([np.argmin(((weights - row) ** 2).sum(1)) for row in matrix])
    else:
        from sklearn.neighbors import NearestNeighbors

        nodes = NearestNeighbors(n_neighbors=1).fit(weights).kneighbors(matrix, return_distance=False)[:, 0]

    # Merging to `n_meta_clusters` happens once, in subcluster(), for every method.
    return np.asarray(nodes, dtype=int)


def _vortex(matrix: np.ndarray, params: dict, use_gpu: bool, seed: int, logger) -> np.ndarray:
    try:
        from .ph_vortex_utils import run_core_vortex
    except ImportError:
        from ph_vortex_utils import run_core_vortex  # type: ignore[no-redef]

    labels = run_core_vortex(
        matrix,
        k_neighbors=int(params.get("k_neighbors", 35)),
        min_cluster_size=int(params.get("min_cluster_size", 50)),
        logger=logger,
        use_gpu=use_gpu,
    )
    return np.asarray(labels, dtype=int)


_METHODS = {"leiden": _leiden, "som": _som, "vortex": _vortex}


def subcluster(
    matrix: np.ndarray,
    method: str = "Leiden",
    params: dict | None = None,
    *,
    use_gpu: bool | None = None,
    seed: int = 42,
    subsample_size: int | None = 150000,
    logger=None,
) -> np.ndarray:
    """Cluster `matrix` with the chosen method and return one label per row.

    Above `subsample_size` rows the work is done on a geometric sketch -- which
    preserves rare populations far better than uniform sampling -- and the labels
    are transferred to the remaining cells by k-NN. Both steps run on the GPU when
    one is present.
    """
    params = dict(params or {})
    key = str(method).strip().lower()
    if key not in _METHODS:
        raise ValueError(f"Unknown clustering method: {method!r}. Expected one of {CLUSTERING_METHODS}.")

    use_gpu = resolve_use_gpu(use_gpu)
    n_cells = len(matrix)
    _log(logger, f"Subclustering {n_cells:,} cells with {method} (GPU={'on' if use_gpu else 'off'})")

    target = int(params.get("n_meta_clusters", 0) or 0)

    def _run(block: np.ndarray) -> np.ndarray:
        """Cluster, then merge down to the requested granularity.

        Deliberately over-clustering and merging by marker profile is more stable
        than tuning a resolution until the count looks right: the fine clusters
        are found by the data, and only clusters that genuinely look alike are
        combined.
        """
        labels = _METHODS[key](block, params, use_gpu, seed, logger)
        n_found = len(np.unique(labels))
        if target and n_found > target:
            _log(logger, f"Merging {n_found} clusters down to {target} by marker profile")
            labels, _mapping = meta_cluster_labels(labels, block, target, logger=logger)
        return np.asarray(labels, dtype=int)

    if subsample_size and n_cells > subsample_size:
        _log(logger, f"Sketching {subsample_size:,} of {n_cells:,} cells")
        sketch_idx = np.asarray(
            geometric_sketch(matrix, int(subsample_size), random_seed=seed, logger=logger)
        )
        sketch_labels = _run(matrix[sketch_idx])

        labels = np.empty(n_cells, dtype=int)
        labels[sketch_idx] = sketch_labels
        rest = np.setdiff1d(np.arange(n_cells), sketch_idx, assume_unique=False)
        if len(rest):
            _log(logger, f"Transferring labels to the remaining {len(rest):,} cells")
            labels[rest] = knn_backfill(
                matrix[sketch_idx], sketch_labels, matrix[rest],
                knn_neighbors=15, use_gpu=use_gpu, logger=logger,
            )
        return labels

    return _run(matrix)


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------


def validate_subclusters(
    labels: np.ndarray,
    matrix: np.ndarray,
    marker_names: list[str],
    *,
    batch_labels: np.ndarray | None = None,
    min_cells: int = 100,
    min_batch_fraction: float = 0.5,
    min_effect_size: float = 0.5,
) -> pd.DataFrame:
    """Report whether each subcluster is real, and why.

    Three independent checks:

    * **size** - a subcluster below `min_cells` cannot be characterised.
    * **reproducibility** - present in at least `min_batch_fraction` of batches.
      With dozens of batches, a cluster confined to one is an artefact of that
      slide, not a cell state.
    * **separation** - the largest absolute Cohen's d against the pooled other
      cells must reach `min_effect_size` on at least one marker, otherwise the
      cluster has no marker profile that distinguishes it.
    """
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_batches = len(np.unique(batch_labels)) if batch_labels is not None else 0

    rows = []
    for cluster in unique:
        mask = labels == cluster
        size = int(mask.sum())
        others = ~mask

        if others.sum() == 0 or size == 0:
            best_marker, best_d = "", 0.0
        else:
            mean_in, mean_out = matrix[mask].mean(0), matrix[others].mean(0)
            var_in, var_out = matrix[mask].var(0), matrix[others].var(0)
            pooled = np.sqrt((var_in + var_out) / 2.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(pooled > 1e-9, (mean_in - mean_out) / pooled, 0.0)
            best = int(np.argmax(np.abs(d)))
            best_marker, best_d = marker_names[best], float(d[best])

        if batch_labels is not None and n_batches:
            present = len(np.unique(np.asarray(batch_labels)[mask]))
            batch_frac = present / n_batches
        else:
            present, batch_frac = 0, 1.0

        reasons = []
        if size < min_cells:
            reasons.append("too few cells")
        if batch_frac < min_batch_fraction:
            reasons.append("not reproducible across batches")
        if abs(best_d) < min_effect_size:
            reasons.append("no separating marker")

        rows.append(
            {
                "cluster": cluster,
                "cells": size,
                "pct": round(100.0 * size / len(labels), 2),
                "batches_present": present,
                "batch_fraction": round(batch_frac, 3),
                "top_marker": best_marker,
                "effect_size": round(best_d, 3),
                "verdict": "keep" if not reasons else "review",
                "reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values("cells", ascending=False).reset_index(drop=True)


def cluster_stability(
    matrix: np.ndarray,
    method: str,
    params: dict,
    *,
    n_repeats: int = 3,
    fraction: float = 0.8,
    seed: int = 42,
    use_gpu: bool | None = None,
    logger=None,
) -> float:
    """Mean pairwise agreement of repeated runs on overlapping subsamples.

    Uses the adjusted Rand index over the cells common to each pair of runs. It
    answers the question a resolution slider cannot: would this partition survive
    if the experiment were repeated. Roughly, >= 0.7 is stable, < 0.5 means the
    structure is being imposed rather than found.

    Returns NaN when the runs collapse to a single cluster, i.e. there is no
    structure here to be stable about.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n_cells = len(matrix)
    take = max(int(n_cells * fraction), 2)

    runs = []
    for repeat in range(n_repeats):
        idx = rng.choice(n_cells, take, replace=False)
        labels = subcluster(
            matrix[idx], method, params,
            use_gpu=use_gpu, seed=seed + repeat, subsample_size=None, logger=logger,
        )
        runs.append(pd.Series(labels, index=idx))

    scores = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            shared = runs[i].index.intersection(runs[j].index)
            if len(shared) <= 10:
                continue
            left, right = runs[i][shared], runs[j][shared]
            # Two runs that both collapse to a single cluster score a perfect ARI
            # while telling us nothing, so a degenerate partition is reported as
            # "no structure" rather than as perfect stability.
            if left.nunique() < 2 or right.nunique() < 2:
                continue
            scores.append(adjusted_rand_score(left, right))
    return float(np.mean(scores)) if scores else float("nan")
