"""Shared acceleration primitives for the phenotyping methods.

This module holds the pieces that every clustering method in tab 7 needs and
that used to be either slow or duplicated:

* GPU detection and an exact brute-force k-NN (used by Vortex and Leiden),
* :func:`knn_backfill`, the distance-weighted label transfer from a clustered
  sketch to the remaining cells,
* :func:`geosketch_gs_fast`, a vectorized replacement for ``geosketch.gs``,
* :func:`read_csv_fast`, a multithreaded CSV reader.

Everything degrades gracefully: with no GPU, no torch, or no pyarrow the CPU
implementations are used and results are unchanged.
"""

import numpy as np

try:
    import geosketch

    GEOSKETCH_AVAILABLE = True
except ImportError:
    GEOSKETCH_AVAILABLE = False
    geosketch = None

# Mirrors the detection pattern used by segmentation_cellpose.py: torch is an
# existing dependency, so the accelerated path needs no new package.
try:
    import torch

    GPU_AVAILABLE = bool(torch.cuda.is_available())
except Exception:  # torch missing or CUDA driver mismatch
    torch = None
    GPU_AVAILABLE = False


def gpu_device_name():
    """Human-readable CUDA device name, or None when no GPU is usable."""
    if not GPU_AVAILABLE or torch is None:
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def resolve_use_gpu(use_gpu):
    """Resolve the tri-state GPU flag (None = auto) against actual availability."""
    if use_gpu is None:
        return GPU_AVAILABLE
    return bool(use_gpu) and GPU_AVAILABLE


def _log(logger, message):
    if logger is not None:
        logger.info(message)
    else:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# GPU k-nearest neighbours
# ---------------------------------------------------------------------------

def _gpu_chunk_rows(n_ref, dtype_bytes=4, budget_fraction=0.25):
    """Rows per query chunk so the (chunk x n_ref) distance block fits comfortably."""
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:
        free_bytes = 8 * 1024 ** 3
    budget = max(int(free_bytes * budget_fraction), 256 * 1024 ** 2)
    rows = int(budget // max(1, n_ref * dtype_bytes))
    return int(np.clip(rows, 256, 16384))


def knn_gpu(reference, query, k, device=0):
    """Exact brute-force k-NN on the GPU.

    Returns ``(squared_distances, indices, reference_tensor)`` with neighbours
    sorted ascending. Uses the ``||a-b||^2 = ||a||^2 + ||b||^2 - 2ab`` expansion
    so the heavy term is a single matmul, which is what makes this fast on
    tensor-core hardware. Being brute force, it is exact -- there is no
    approximation trade-off relative to the CPU KD-tree path.
    """
    dev = torch.device(f"cuda:{device}")
    ref_t = torch.as_tensor(np.ascontiguousarray(reference), device=dev, dtype=torch.float32)
    ref_sq = (ref_t * ref_t).sum(1)

    n_query = query.shape[0]
    k = int(min(k, ref_t.shape[0]))
    out_d = torch.empty((n_query, k), device=dev, dtype=torch.float32)
    out_i = torch.empty((n_query, k), device=dev, dtype=torch.int64)

    chunk = _gpu_chunk_rows(ref_t.shape[0])
    for start in range(0, n_query, chunk):
        end = min(start + chunk, n_query)
        q = torch.as_tensor(np.ascontiguousarray(query[start:end]), device=dev, dtype=torch.float32)
        dist = ref_sq.unsqueeze(0) - 2.0 * (q @ ref_t.T)
        dist += (q * q).sum(1, keepdim=True)
        out_d[start:end], out_i[start:end] = torch.topk(dist, k, dim=1, largest=False, sorted=True)
        del q, dist

    return out_d, out_i, ref_t


def knn_backfill(
    reference_matrix,
    reference_labels,
    query_matrix,
    knn_neighbors=15,
    use_gpu=None,
    gpu_device=0,
    logger=None,
):
    """Distance-weighted k-NN label transfer from a clustered sketch to the rest.

    Matches ``sklearn.neighbors.KNeighborsClassifier(weights="distance")``
    semantics; the GPU path performs the same vote over an exact brute-force
    neighbour search. Falls back to sklearn on any GPU failure.
    """
    from sklearn.neighbors import KNeighborsClassifier

    reference_labels = np.asarray(reference_labels)
    n_neighbors = int(min(knn_neighbors, len(reference_matrix)))

    if resolve_use_gpu(use_gpu):
        try:
            # Labels may be arbitrary integers; compact them so they can index
            # the vote matrix, then map back afterwards.
            classes, compact = np.unique(reference_labels, return_inverse=True)
            distances, indices, _ref = knn_gpu(
                reference_matrix, query_matrix, n_neighbors, device=gpu_device
            )
            labels_t = torch.as_tensor(compact, device=distances.device, dtype=torch.int64)
            neighbour_labels = labels_t[indices]
            # knn_gpu returns squared distances; sklearn weights by 1/distance.
            weights = 1.0 / distances.clamp_min(1e-12).sqrt()
            votes = torch.zeros(
                (neighbour_labels.shape[0], len(classes)), device=distances.device
            )
            votes.scatter_add_(1, neighbour_labels, weights)
            winner = votes.argmax(1).cpu().numpy()
            del distances, indices, neighbour_labels, weights, votes
            torch.cuda.empty_cache()
            return classes[winner]
        except Exception as exc:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _log(logger, f"GPU KNN backfill failed ({exc}); falling back to CPU.")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)
    knn.fit(reference_matrix, reference_labels)
    return knn.predict(query_matrix)


# ---------------------------------------------------------------------------
# Fast geometric sketching
# ---------------------------------------------------------------------------

def _greedy_interval_labels(sorted_values, unit):
    """Greedy 1-D covering intervals over pre-sorted values.

    Reproduces geosketch's inner loop -- open a new interval as soon as a value
    exceeds ``interval_start + unit`` -- but iterates over intervals instead of
    over samples, so the cost scales with the (small) interval count rather than
    with the number of cells.
    """
    n = sorted_values.shape[0]
    labels = np.empty(n, dtype=np.int64)
    position = 0
    interval = 0
    while position < n:
        end = int(np.searchsorted(sorted_values, sorted_values[position] + unit, side="right"))
        if end <= position:
            end = position + 1
        labels[position:end] = interval
        position = end
        interval += 1
    return labels, interval


def geosketch_gs_fast(X, N, seed=None, alpha=0.1, max_iter=200):
    """Vectorized drop-in replacement for ``geosketch.gs`` (replace=False).

    Implements the same geometric plaid covering: binary-search a box side length
    until the number of occupied boxes is within ``alpha`` of N, then sample cells
    by repeated random sweeps over the occupied boxes.

    The reference implementation rebuilds the grid with per-sample Python loops on
    every search iteration, and its sampling stage calls ``list(valid_grids)``
    inside a loop over N -- which is O(N^2) and is what makes a 500k sketch take
    roughly an hour. This version is statistically equivalent but vectorized;
    measured on 5.6M cells it matches the reference sketch's density-flattening
    (log-density spread 0.206 vs 0.207, p90/p10 2.65 vs 2.65) in ~19s vs ~3400s.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    if N >= n_samples:
        return np.arange(n_samples)

    X = X - X.min(0)
    max_value = X.max()
    if max_value > 0:
        X = X / max_value
    X_ptp = np.ptp(X, axis=0)

    order = np.argsort(X, axis=0)
    sorted_cols = np.take_along_axis(X, order, axis=0)

    low_unit, high_unit = 0.0, float(max(X_ptp))
    unit = (low_unit + high_unit) / 4.0

    cell_ids = np.zeros(n_samples, dtype=np.int64)
    for _ in range(max_iter):
        # Mixed-radix encoding of the per-dimension interval indices, so each
        # occupied box becomes one integer key instead of a Python tuple.
        keys = np.zeros(n_samples, dtype=np.int64)
        radix = 1
        for d in range(n_features):
            if X_ptp[d] <= unit:
                continue
            labels_sorted, n_intervals = _greedy_interval_labels(sorted_cols[:, d], unit)
            labels = np.empty(n_samples, dtype=np.int64)
            labels[order[:, d]] = labels_sorted
            if n_intervals > 1 and radix > (2 ** 62) // n_intervals:
                # Grid too fine to keep encoding; combine what we have so far and
                # continue from a fresh radix so keys can never silently wrap.
                keys = np.unique(keys, return_inverse=True)[1]
                radix = int(keys.max()) + 1 if keys.size else 1
            keys = keys + labels * radix
            radix *= max(1, n_intervals)

        cell_ids = np.unique(keys, return_inverse=True)[1]
        n_boxes = int(cell_ids.max()) + 1 if cell_ids.size else 0

        if n_boxes > N * (1 + alpha):
            low_unit = unit
            unit = (unit + high_unit) / 2.0 if high_unit is not None else unit * 2.0
        elif n_boxes < N * (1 - alpha):
            high_unit = unit
            unit = (unit + low_unit) / 2.0 if low_unit is not None else unit / 2.0
        else:
            break

        if high_unit is not None and low_unit is not None and high_unit - low_unit < 1e-20:
            break

    # Sample without replacement: repeated random sweeps over occupied boxes,
    # taking one random unused cell from each box per sweep. This reproduces the
    # reference behaviour (refill the candidate set once every box has been hit)
    # without rebuilding a Python set on every draw.
    n_boxes = int(cell_ids.max()) + 1
    box_order = np.argsort(cell_ids, kind="stable")
    box_starts = np.searchsorted(cell_ids[box_order], np.arange(n_boxes + 1))

    members = [box_order[box_starts[b]:box_starts[b + 1]] for b in range(n_boxes)]
    for m in members:
        rng.shuffle(m)

    counts = np.zeros(n_boxes, dtype=np.int64)
    sizes = np.array([len(m) for m in members], dtype=np.int64)
    picked = []
    remaining = int(N)
    while remaining > 0:
        live = np.flatnonzero(counts < sizes)
        if live.size == 0:
            break
        rng.shuffle(live)
        take = live[:remaining]
        picked.append(
            np.fromiter((members[b][counts[b]] for b in take), dtype=np.int64, count=take.size)
        )
        counts[take] += 1
        remaining -= take.size

    result = np.concatenate(picked) if picked else np.empty(0, dtype=np.int64)
    return np.sort(result)


def geometric_sketch(embedding, n_select, random_seed=42, logger=None, use_fast=True):
    """Select ``n_select`` rows of ``embedding`` by geometric sketching.

    Prefers the vectorized implementation; falls back to the reference geosketch
    package, then to a uniform random sample.
    """
    embedding = np.asarray(embedding)
    n_cells = embedding.shape[0]
    if n_select >= n_cells:
        return np.arange(n_cells)

    if use_fast:
        try:
            selected = geosketch_gs_fast(embedding, int(n_select), seed=random_seed)
            if selected.size:
                _log(
                    logger,
                    f"Geometric sketching selected {selected.size:,} of {n_cells:,} cells "
                    f"using {embedding.shape[1]} dimensions.",
                )
                return selected
        except Exception as exc:
            _log(logger, f"Fast geometric sketching failed ({exc}); trying the reference implementation.")

    if GEOSKETCH_AVAILABLE and geosketch is not None:
        try:
            emb = np.ascontiguousarray(embedding, dtype=np.float64)
            return np.asarray(
                geosketch.gs(emb, int(n_select), replace=False, seed=random_seed), dtype=int
            )
        except Exception as exc:
            _log(logger, f"Geosketch failed ({exc}); using a uniform random sample.")

    rng = np.random.default_rng(random_seed)
    return rng.choice(n_cells, int(n_select), replace=False)


# ---------------------------------------------------------------------------
# Meta-clustering
# ---------------------------------------------------------------------------

def meta_cluster_labels(labels, matrix, target_n, linkage="ward", logger=None):
    """Merge fine-grained clusters into ``target_n`` meta-clusters.

    Density- and graph-based clustering can return far more clusters than there
    are cell types, which makes manual annotation impractical and leaves many
    clusters too small to characterise confidently. This merges clusters whose
    mean marker profiles are closest, using hierarchical clustering on the
    cluster centroids -- the same operation the SOM path performs on its nodes,
    so every method can be brought to a comparable granularity before the
    per-cluster phenotype assignment step.

    Merging happens in profile space, so clusters are only combined when they
    look alike across all markers; it does not simply absorb small clusters
    into big ones.

    Parameters
    ----------
    labels : array of int
        Cluster assignment per cell.
    matrix : 2-D array
        The feature matrix the clustering was computed on (cells x markers).
    target_n : int
        Desired number of meta-clusters. Values < 2, or >= the current cluster
        count, leave the labels untouched.

    Returns
    -------
    (new_labels, mapping) where mapping is {original_cluster: meta_cluster}.
    """
    from sklearn.cluster import AgglomerativeClustering

    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_clusters = len(unique)
    target_n = int(target_n or 0)

    if target_n < 2 or target_n >= n_clusters:
        return labels, {int(c): int(c) for c in unique}

    # Centroid per cluster, in the same space the clustering used.
    centroids = np.stack([matrix[labels == c].mean(axis=0) for c in unique])
    centroids = np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)

    model = AgglomerativeClustering(n_clusters=target_n, linkage=linkage)
    groups = model.fit_predict(centroids)

    mapping = {int(c): int(g) for c, g in zip(unique, groups)}
    remapped = np.array([mapping[int(c)] for c in labels], dtype=int)
    _log(
        logger,
        f"Meta-clustering merged {n_clusters} clusters into {len(np.unique(remapped))} "
        f"using {linkage} linkage on cluster marker profiles.",
    )
    return remapped, mapping


# ---------------------------------------------------------------------------
# Phenotype recommendation
# ---------------------------------------------------------------------------

#: Suffixes the pipeline appends to marker columns. Stripped when matching a
#: signature written with plain marker names against profile column names.
MARKER_SUFFIXES = ("_corr_norm", "_norm", "_corr", "_unmixed", "_resolve")


def strip_marker_suffix(marker):
    """Return a marker name with any known pipeline suffix removed."""
    name = str(marker)
    for suffix in MARKER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_phenotype_signatures(text, *, return_problems=False):
    """Parse legacy marker lists or grouped AND/OR rules, without flattening.

    One phenotype per line, markers separated by commas, a trailing ``-`` or a
    leading ``!`` marking a negative marker. ``+`` is optional and ignored::

        B: CD20+, CD3e-
        CD4: CD3e, CD4, CD8-, FoxP3-
        Treg: FoxP3, CD3e, CD4, CD8-

    Grouped rules use ``Name = [A, B] OR [C]; SUPPORT=[D]; NOT=[E]`` and
    return dictionaries with alternatives/support/excluded lists. Legacy lists
    retain the ``(positive, negative)`` tuple API and scoring. Invalid lines are
    skipped; pass ``return_problems=True`` to also get line-numbered diagnostics.
    """
    import re
    from spacemap_gui.resolve_rule_parser import parse_resolve_rule_text

    signatures = {}
    problems = []
    seen = set()
    for number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Reuse RESOLVE's grammar, but resolve actual marker columns later so
        # corrected/normalized suffixes and case-insensitive names still work.
        if any(char in line for char in "=[]") or re.search(r"\b(?:AND|OR|SUPPORT|NOT)\b", line, re.I):
            rows, errors = parse_resolve_rule_text(line)
            problems.extend(error.replace("line 1:", f"line {number}:", 1) for error in errors)
            if rows:
                row = rows[0]
                name = row.pop("Phenotype")
                if name.lower() in seen:
                    problems.append(f"line {number}: duplicate phenotype '{name}' skipped")
                else:
                    signatures[name] = row
                    seen.add(name.lower())
            continue
        if ":" not in line:
            problems.append(f"line {number}: expected 'Name: markers' or 'Name = [markers]'")
            continue
        name, _, marker_text = line.partition(":")
        name = name.strip()
        if not name:
            problems.append(f"line {number}: no phenotype name")
            continue
        if name.lower() in seen:
            problems.append(f"line {number}: duplicate phenotype '{name}' skipped")
            continue
        positive, negative = [], []
        for token in marker_text.split(","):
            token = token.strip()
            if not token:
                continue
            if token.startswith("!"):
                negative.append(token[1:].strip())
            elif token.endswith("-"):
                negative.append(token[:-1].strip())
            else:
                positive.append(token.rstrip("+").strip())
        positive = [m for m in positive if m]
        negative = [m for m in negative if m]
        if positive or negative:
            signatures[name] = (positive, negative)
            seen.add(name.lower())
        else:
            problems.append(f"line {number}: '{name}' has no usable markers")
    return (signatures, problems) if return_problems else signatures


def derive_signatures_from_reference(
    matrix,
    marker_names,
    labels,
    n_positive=3,
    n_negative=2,
    negative_cutoff=-0.25,
    min_cells=15,
    exclude=("Others", "Unassigned", "unclassified", "nan", "None"),
):
    """Learn phenotype signatures from manually annotated cells.

    For each labelled phenotype this takes the markers that separate it most
    strongly from the *other* phenotypes -- not simply its brightest markers,
    since a marker that is high everywhere identifies nothing. The same
    discriminative margin, in reverse, supplies the negative markers that
    separate otherwise-similar populations.

    Only a small annotated subset is needed: the signature depends on the mean
    profile per phenotype, which stabilises after a few dozen cells each.

    Returns ``{phenotype: (positive_markers, negative_markers)}``.
    """
    import pandas as pd

    matrix = np.asarray(matrix, dtype=float)
    labels = pd.Series(labels).astype("object")
    marker_names = [str(m) for m in marker_names]

    valid = labels.notna()
    excluded = {str(e).lower() for e in exclude}
    valid &= ~labels.astype(str).str.lower().isin(excluded)
    if not valid.any():
        return {}

    idx = np.flatnonzero(valid.to_numpy())
    y = labels.to_numpy()[idx]

    # Z-score per marker so the margins below are comparable across markers.
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    z = (matrix - matrix.mean(axis=0)) / std

    groups = [g for g in pd.unique(y) if (y == g).sum() >= min_cells]
    if len(groups) < 2:
        return {}
    profiles = pd.DataFrame(
        {g: z[idx[y == g]].mean(axis=0) for g in groups}, index=marker_names
    )

    signatures = {}
    for g in groups:
        others = profiles.drop(columns=[g]).mean(axis=1)
        margin = (profiles[g] - others).sort_values(ascending=False)
        positive = [m for m in margin.head(int(n_positive)).index if margin[m] > 0]
        negative = [
            m for m in margin.tail(int(n_negative)).index if margin[m] < negative_cutoff
        ]
        if positive:
            signatures[str(g)] = (positive, negative)
    return signatures


def signatures_to_text(signatures):
    """Render a signature dict back into the editable plain-text format."""
    lines = []
    for name in sorted(signatures):
        if isinstance(signatures[name], dict):
            rule = signatures[name]
            def group(markers):
                return "[" + ", ".join(markers) + "]"
            alternatives = " OR ".join(group(markers) for markers in rule["alternatives"])
            lines.append(f"{name} = {alternatives}; SUPPORT={group(rule['support'])}; NOT={group(rule['excluded'])}")
            continue
        positive, negative = signatures[name]
        markers = list(positive) + [f"{m}-" for m in negative]
        lines.append(f"{name}: {', '.join(markers)}")
    return "\n".join(lines)


def learn_phenotype_profiles(matrix, marker_names, labels, min_cells=15,
                             n_subprofiles=5, min_cells_per_subprofile=40, silenced=None,
                             exclude=("Others", "Unassigned", "unclassified", "nan", "None")):
    """Learn each phenotype's full marker profile from manually annotated cells.

    Unlike a signature -- which names a handful of markers and ignores the rest --
    this keeps the expected level of *every* marker, plus how much that level
    varies within the phenotype. A cluster can then be judged on its whole
    fingerprint rather than on three or four markers, which is what makes it
    noticeably more accurate.

    A cell type is rarely one uniform population: fibroblasts may be aSMA-high in
    one region and FN1-high in another, and averaging them into a single profile
    describes neither well. ``n_subprofiles`` can therefore learn several
    alternative fingerprints per cell type, and a cluster matching *any* of them
    is given that cell type.

    It defaults to 5, which held-out testing on a 2.8M-cell dataset put ahead of a
    single averaged profile by about one point on one reference and two on another.
    A cell type only gets as many fingerprints as it has cells to support -- one per
    ``min_cells_per_subprofile`` -- so small populations are never over-split.

    Choosing a different count per cell type was tried, both from how internally
    varied each looks and from held-out accuracy, and both did worse than one shared
    value: a few hundred annotated cells per type is too few to make that many
    separate choices reliably.

    ``silenced`` drops specific fingerprints by ``(phenotype, index)``. Annotated
    populations sometimes contain artefacts -- cells labelled epithelial whose
    strongest marker is FoxP3 are more likely segmentation spill-over from an
    adjacent Treg than real biology -- and silencing such a fingerprint stops it
    attracting clusters. Automatic removal was tested and made things slightly
    worse, because those cells are still real members of the annotation; the
    judgement of which are artefacts is left to the user.

    Returns ``{phenotype: {"mean", "sd", "n", "prototypes"}}`` over ``marker_names``,
    where ``prototypes`` holds the alternative fingerprints (at least one).
    """
    from sklearn.cluster import KMeans
    import pandas as pd

    matrix = np.asarray(matrix, dtype=float)
    labels = pd.Series(labels).astype("object")
    excluded = {str(e).lower() for e in exclude}
    keep = labels.notna().to_numpy()
    keep &= ~labels.astype(str).str.lower().isin(excluded).to_numpy()
    if not keep.any():
        return {}

    rows = np.flatnonzero(keep)
    y = labels.to_numpy()[rows]

    # Z-score so every marker contributes on a comparable scale.
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    z = (matrix - matrix.mean(axis=0)) / std

    profiles = {}
    for group in pd.unique(y):
        members = rows[y == group]
        if members.size < int(min_cells):
            continue
        block = z[members]
        # Split into alternative fingerprints only where there are enough cells to
        # estimate each one; otherwise a "sub-profile" is just noise.
        n_sub = int(max(1, min(int(n_subprofiles), members.size // int(min_cells_per_subprofile))))
        sizes = [int(members.size)]
        if n_sub > 1:
            try:
                km = KMeans(n_clusters=n_sub, n_init=5, random_state=0).fit(block)
                prototypes = [np.asarray(c, dtype=float) for c in km.cluster_centers_]
                sizes = np.bincount(km.labels_, minlength=n_sub).tolist()
            except Exception:
                prototypes = [block.mean(axis=0)]
        else:
            prototypes = [block.mean(axis=0)]

        # Record what leads each fingerprint and how many cells back it, so the
        # user can judge which are real and which look like spill-over.
        details = []
        for i, centre in enumerate(prototypes):
            order = np.argsort(-centre)[:5]
            top = int(order[0])
            details.append({
                "index": i,
                "top_marker": str(marker_names[top]) if top < len(marker_names) else "?",
                "top_value": float(centre[top]),
                "n_cells": int(sizes[i]) if i < len(sizes) else int(members.size),
                # The leading marker alone can mislead: a pattern headed by a foreign
                # marker may still carry this cell type's own markers just below it,
                # which is the difference between spill-over on a real cell and a
                # genuine mislabel. Keep the top five so that is visible.
                "top_markers": [
                    (str(marker_names[j]), float(centre[j]))
                    for j in order if j < len(marker_names)
                ],
            })

        drop = {i for (g, i) in (silenced or set()) if str(g) == str(group)}
        kept = [p for i, p in enumerate(prototypes) if i not in drop]
        if not kept:                       # never silence a cell type entirely
            kept = [block.mean(axis=0)]

        profiles[str(group)] = {
            "mean": block.mean(axis=0),
            "sd": block.std(axis=0),
            "n": int(members.size),
            "prototypes": kept,
            "details": details,
            "silenced": sorted(drop),
        }
    return profiles


#: Marks a pattern the user switched off, as distinct from one that ignores no markers.
OFF = object()


def normalize_prototype_markers(value):
    """Accept per-pattern marker edits in any of the forms they can arrive in.

    The editable pattern table produces ``{phenotype: {index: [markers]}}``, which
    survives being written to the project file. Older sessions used a
    ``{(phenotype, index): {...}}`` mapping, and saving that turns the tuple key
    into its own string representation, so both are decoded here rather than
    letting a restored project fail to load.

    Returns ``{(phenotype, index): {marker, ...}}``.
    """
    import ast as _ast

    out = {}
    if not value:
        return out

    def _store(name, index, markers):
        try:
            index = int(index)
        except (TypeError, ValueError):
            return
        if markers is None:
            out[(str(name), index)] = OFF          # pattern switched off
            return
        if isinstance(markers, (str, bytes)):
            markers = [markers]
        try:
            out[(str(name), index)] = {str(m) for m in markers}
        except TypeError:
            return

    for key, markers in dict(value).items():
        # Nested form: {phenotype: {index: [markers]}}
        if isinstance(markers, dict):
            for index, inner in markers.items():
                _store(key, index, inner)
            continue
        # Tuple key, still intact
        if isinstance(key, (tuple, list)) and len(key) == 2:
            _store(key[0], key[1], markers)
            continue
        # Tuple key that has been through a text round-trip
        if isinstance(key, str) and key.startswith("(") and key.endswith(")"):
            try:
                parsed = _ast.literal_eval(key)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
                _store(parsed[0], parsed[1], markers)
            continue
    return out


def match_phenotype_profiles(cluster_profiles, phenotype_profiles, marker_names,
                             use_range=False, range_floor=0.25, masked_markers=None,
                             prototype_markers=None):
    """Assign each cluster to the phenotype whose learned profile it best matches.

    Distance is the mean absolute difference across all markers between the
    cluster's profile and the phenotype's expected profile. With ``use_range``
    each marker is divided by how much that phenotype varies in it, so a marker
    the phenotype holds tightly counts for more than one it scatters across.

    ``prototype_markers`` ({(phenotype, index): {marker, ...}}) names the markers a
    single pattern should IGNORE, which is what the editable pattern table writes
    when a marker is deleted from a row. An entry of ``None`` switches the pattern
    off altogether.

    An ignored marker is not simply dropped from the average: doing that would let
    a pattern with fewer active markers score a smaller mean distance than its
    rivals and win every cluster. Instead its contribution is replaced by the mean
    contribution of the markers that remain, so ignoring a marker makes it carry no
    information for that cell type without making the cell type easier to match.

    ``masked_markers`` ({phenotype: {marker, ...}}) drops individual markers from a
    single cell type's comparison. This is for markers that appear in an annotated
    population without belonging to it -- FoxP3 showing up in epithelial cells that
    sit next to a Treg, say. Left in, the learned profile expects that marker, and
    a genuinely epithelial cluster *without* the neighbour's signal is penalised for
    lacking it. Masking removes the marker from that cell type's distance only, so
    the cell type is judged on the markers that actually define it. Other cell types
    still use the marker normally, so a real Treg is unaffected.

    Every cluster necessarily has a nearest phenotype, so there is no "matched
    nothing" case to patch over -- the margin to the runner-up serves directly as
    the confidence.

    Returns ``(assignment, confidence, ordered_phenotypes, distances)``.
    """
    if not phenotype_profiles:
        return {}, {}, [], None

    columns = [str(c) for c in cluster_profiles.columns]
    lookup = {c: i for i, c in enumerate(columns)}
    for i, c in enumerate(columns):
        lookup.setdefault(strip_marker_suffix(c), i)
        lookup.setdefault(strip_marker_suffix(c).lower(), i)

    order = []
    for name in marker_names:
        key = str(name)
        idx = lookup.get(key)
        if idx is None:
            idx = lookup.get(strip_marker_suffix(key).lower())
        order.append(idx)
    usable = [i for i, idx in enumerate(order) if idx is not None]
    if not usable:
        return {}, {}, [], None

    values = cluster_profiles.to_numpy(dtype=float)[:, [order[i] for i in usable]]
    phenotypes = sorted(phenotype_profiles)
    distances = np.zeros((values.shape[0], len(phenotypes)))
    masked_markers = {str(k): {str(v) for v in vals} for k, vals in (masked_markers or {}).items()}
    prototype_markers = normalize_prototype_markers(prototype_markers)
    marker_at = {p: str(marker_names[i]) for p, i in enumerate(usable)}

    def _ignored_mask(ignore):
        """Boolean mask over comparison positions for markers to disregard."""
        mask = np.zeros(len(usable), dtype=bool)
        if not ignore:
            return mask
        for p in range(len(usable)):
            name_full = marker_at[p]
            if name_full in ignore or strip_marker_suffix(name_full) in ignore:
                mask[p] = True
        return mask

    def _distance(observed, centre, spread, ignore_mask):
        """Mean deviation across ALL markers, with ignored ones made neutral.

        Every cell type is scored over the same number of markers, so one that
        disregards a few cannot win simply by averaging over fewer terms.
        """
        if spread is not None:
            deviation = np.abs(observed - centre) / (spread + range_floor)
        else:
            deviation = np.abs(observed - centre)
        if ignore_mask.any():
            keep = ~ignore_mask
            if not keep.any():
                return np.full(deviation.shape[0], np.inf)
            neutral = deviation[:, keep].mean(axis=1, keepdims=True)
            deviation = np.where(ignore_mask[None, :], neutral, deviation)
        return deviation.mean(axis=1)

    for j, name in enumerate(phenotypes):
        entry = phenotype_profiles[name]
        # A cell type may have several alternative patterns; matching any one of
        # them counts, so keep the closest.
        candidates = entry.get("prototypes") or [entry["mean"]]
        drop = masked_markers.get(name, set())

        spread = np.asarray(entry["sd"], dtype=float)[usable] if use_range else None
        per_prototype = []
        for p_index, candidate in enumerate(candidates):
            ignore = prototype_markers.get((name, p_index))
            if ignore is OFF:
                continue                      # pattern switched off entirely
            mask = _ignored_mask((ignore or set()) | drop)
            centre = np.asarray(candidate, dtype=float)[usable]
            per_prototype.append(_distance(values, centre, spread, mask))

        if not per_prototype:                 # every pattern switched off; fall back
            centre = np.asarray(entry["mean"], dtype=float)[usable]
            per_prototype = [_distance(values, centre, spread, np.zeros(len(usable), dtype=bool))]
        distances[:, j] = np.min(np.stack(per_prototype, axis=1), axis=1)

    best = distances.argmin(axis=1)
    ordered = np.sort(distances, axis=1)
    margin = (ordered[:, 1] - ordered[:, 0]) if distances.shape[1] > 1 else np.full(len(values), 1.0)

    assignment, confidence, matched = {}, {}, {}
    marker_labels = [str(marker_names[i]) for i in usable]
    for i, cluster in enumerate(cluster_profiles.index):
        name = phenotypes[best[i]]
        assignment[cluster] = name
        confidence[cluster] = float(margin[i])
        # Record which of the phenotype's fingerprints matched, and what it expects,
        # so the caller can show the comparison that drove the decision.
        entry = phenotype_profiles[name]
        candidates = entry.get("prototypes") or [entry["mean"]]
        per = [np.abs(values[i] - np.asarray(c, dtype=float)[usable]).mean() for c in candidates]
        pick = int(np.argmin(per))
        matched[cluster] = {
            "phenotype": name,
            "variant": pick + 1,
            "n_variants": len(candidates),
            "expected": np.asarray(candidates[pick], dtype=float)[usable],
            "observed": values[i],
            "markers": marker_labels,
        }
    return assignment, confidence, phenotypes, distances, matched


def recommend_phenotypes(
    cluster_profiles,
    signatures,
    seed_labels=None,
    propagate=True,
    min_score=0.0,
    phenotype_profiles=None,
    profile_marker_names=None,
    use_range=False,
    masked_markers=None,
    prototype_markers=None,
):
    """Propose a phenotype for every cluster, with a confidence and its reasoning.

    Every cluster is scored against every signature at once and keeps its best
    match -- there is no sequential assignment, so the order clusters are
    considered in cannot change the result. Confidence is the margin between the
    best and the runner-up phenotype, which is what makes a review queue useful:
    sorting by it ascending puts the genuinely ambiguous clusters first and
    leaves the obvious ones to be approved in bulk.

    Parameters
    ----------
    cluster_profiles : pandas.DataFrame
        Clusters (rows) x markers (columns), holding mean z-scored expression.
    signatures : dict
        Legacy ``{phenotype: (positive_markers, negative_markers)}`` or grouped
        rules from :func:`parse_phenotype_signatures`. For grouped rules, AND is
        the minimum marker expression and OR is the maximum alternative score.
        High NOT expression subtracts its maximum positive value. SUPPORT adds
        25% of its mean positive expression only to an already eligible score.
        These are graded cluster-expression scores, not per-cell boolean gates.
    seed_labels : dict, optional
        Known ``{cluster: phenotype}`` (e.g. majority vote over manually
        annotated cells). These are trusted over the signature score.
    propagate : bool
        Give clusters that match no signature the phenotype of the most similar
        already-decided cluster, so none are left unassigned by default.
    min_score : float
        A signature match below this score is treated as no match.

    Returns
    -------
    pandas.DataFrame indexed by cluster with columns: phenotype, confidence,
    source ('reference' | 'signature' | 'similarity'), evidence, runner_up.
    """
    import pandas as pd

    profiles = cluster_profiles.copy()
    profiles.columns = [str(c) for c in profiles.columns]
    seed_labels = {k: v for k, v in (seed_labels or {}).items()}

    # Signatures are written with bare marker names ("CD20") while the profile
    # columns usually carry pipeline suffixes ("CD20_corr_norm"). Resolve through
    # a suffix-insensitive lookup so either form matches, preferring an exact hit.
    lookup = {c: c for c in profiles.columns}
    for column in profiles.columns:
        cleaned = strip_marker_suffix(column)
        lookup.setdefault(cleaned, column)
        lookup.setdefault(cleaned.lower(), column)
    lookup.update({c.lower(): c for c in profiles.columns})

    def _resolve(markers):
        resolved = []
        for marker in markers:
            key = str(marker).strip()
            column = lookup.get(key) or lookup.get(key.lower()) or lookup.get(strip_marker_suffix(key).lower())
            if column is not None and column not in resolved:
                resolved.append(column)
        return resolved

    # Score every cluster against every signature in one pass.
    scores = {}
    unresolved = set()
    for name, signature in (signatures or {}).items():
        if isinstance(signature, dict):
            alternatives = signature["alternatives"]
            support = signature.get("support", [])
            excluded = signature.get("excluded", [])
            markers = [m for group in alternatives for m in group] + support + excluded
            missing = {m for m in markers if not _resolve([m])}
            unresolved.update(missing)
            # Do not weaken an AND group or drop an exclusion to make it match.
            if any(marker in missing for marker in excluded):
                continue
            group_scores = [
                profiles[_resolve(group)].min(axis=1)
                for group in alternatives
                if group and not any(marker in missing for marker in group)
            ]
            if not group_scores:
                continue
            value = pd.concat(group_scores, axis=1).max(axis=1)
            neg = _resolve(excluded)
            if neg:
                value = value - profiles[neg].clip(lower=0).max(axis=1)
            sup = _resolve(support)
            if sup:
                bonus = 0.25 * profiles[sup].clip(lower=0).mean(axis=1)
                value = value + bonus.where(value > min_score, 0.0)
            scores[name] = value
            continue
        positive, negative = signature
        pos = _resolve(positive)
        neg = _resolve(negative)
        unresolved.update(
            m for m in list(positive) + list(negative)
            if not _resolve([m])
        )
        if not pos and not neg:
            continue
        value = profiles[pos].mean(axis=1) if pos else 0.0
        if neg:
            value = value - profiles[neg].mean(axis=1)
        scores[name] = value
    score_frame = pd.DataFrame(scores) if scores else pd.DataFrame(index=profiles.index)

    # Full-profile matching compares a cluster against the complete learned
    # fingerprint of each phenotype rather than the handful of markers a
    # signature names. It is more accurate, and because every cluster has a
    # nearest phenotype there is no "matched nothing" case to fall back from.
    profile_assignment, profile_confidence, profile_matched = {}, {}, {}
    uncovered_scores = None
    if phenotype_profiles and profile_marker_names:
        profile_assignment, profile_confidence, _names, _dist, profile_matched = match_phenotype_profiles(
            profiles, phenotype_profiles, profile_marker_names, use_range=use_range,
            masked_markers=masked_markers, prototype_markers=prototype_markers,
        )
        # Signatures naming a phenotype the reference does not contain are the only
        # way such a cell type can ever be proposed; keep their scores available.
        extra = [c for c in score_frame.columns if c not in phenotype_profiles]
        if extra:
            uncovered_scores = score_frame[extra]

    rows = []
    for cluster in profiles.index:
        top_marker_text = ", ".join(
            f"{m} {profiles.loc[cluster, m]:+.1f}"
            for m in profiles.loc[cluster].nlargest(3).index
        )
        if cluster in seed_labels:
            rows.append(
                dict(cluster=cluster, phenotype=seed_labels[cluster], confidence=float("inf"),
                     source="reference", evidence="majority of annotated cells in this cluster",
                     runner_up="")
            )
            continue
        if cluster in profile_assignment:
            # A phenotype the reference never annotated has no learned profile, so
            # profile matching can never propose it. Let a signature written for
            # such a phenotype win when it scores clearly positive and beats the
            # signature score of whatever the profile picked -- otherwise adding a
            # cell type by hand would have no effect.
            if uncovered_scores is not None:
                extra = uncovered_scores.loc[cluster].sort_values(ascending=False)
                best_extra, best_extra_score = extra.index[0], float(extra.iloc[0])
                assigned = profile_assignment[cluster]
                rival = (
                    float(score_frame.loc[cluster, assigned])
                    if (not score_frame.empty and assigned in score_frame.columns)
                    else 0.0
                )
                if best_extra_score > min_score and best_extra_score > rival:
                    second = float(extra.iloc[1]) if len(extra) > 1 else float("-inf")
                    rows.append(
                        dict(cluster=cluster, phenotype=best_extra,
                             confidence=best_extra_score - (second if np.isfinite(second) else 0.0),
                             source="signature",
                             evidence=f"signature for a phenotype absent from the reference; {top_marker_text}",
                             runner_up=assigned)
                    )
                    continue
            detail = profile_matched.get(cluster)
            if detail is not None:
                expected = detail["expected"]
                observed = detail["observed"]
                markers = detail["markers"]
                # Show the markers that most define this phenotype, with what the
                # cluster actually has beside what the phenotype expects.
                top = np.argsort(-np.abs(expected))[:4]
                comparison = " · ".join(
                    f"{markers[t]} {observed[t]:+.1f} (expect {expected[t]:+.1f})" for t in top
                )
                variant = (
                    f" [variant {detail['variant']} of {detail['n_variants']}]"
                    if detail["n_variants"] > 1 else ""
                )
                evidence_text = f"{comparison}{variant}"
            else:
                evidence_text = f"closest learned profile; {top_marker_text}"
            rows.append(
                dict(cluster=cluster, phenotype=profile_assignment[cluster],
                     confidence=profile_confidence.get(cluster, 0.0),
                     source="profile", evidence=evidence_text, runner_up="")
            )
            continue
        if not score_frame.empty:
            ordered = score_frame.loc[cluster].sort_values(ascending=False)
            best, best_score = ordered.index[0], float(ordered.iloc[0])
            second = float(ordered.iloc[1]) if len(ordered) > 1 else float("-inf")
            if best_score > min_score:
                rows.append(
                    dict(cluster=cluster, phenotype=best,
                         confidence=best_score - (second if np.isfinite(second) else 0.0),
                         source="signature", evidence=top_marker_text,
                         runner_up=ordered.index[1] if len(ordered) > 1 else "")
                )
                continue
        rows.append(dict(cluster=cluster, phenotype=None, confidence=0.0, source="unmatched",
                         evidence=top_marker_text, runner_up=""))

    result = pd.DataFrame(rows).set_index("cluster")
    # Surfaced so the caller can warn instead of silently scoring nothing.
    result.attrs["unresolved_markers"] = sorted(unresolved)
    result.attrs["scored_phenotypes"] = sorted(score_frame.columns)

    # The raw confidence values are not comparable across sources: a signature
    # margin is in z-score units and unbounded, a similarity is a cosine in
    # [0, 1], and a reference call has no margin at all. `certainty` maps all
    # three onto one 0-1 scale so a single review cutoff is meaningful.
    def _certainty(row):
        if row["source"] == "reference":
            return 1.0
        if row["source"] in ("signature", "profile"):
            margin = max(float(row["confidence"]), 0.0)
            return margin / (margin + 1.0)      # 1.0 -> 0.50, 3.0 -> 0.75
        if row["source"] == "similarity":
            return max(0.0, min(1.0, float(row["confidence"])))
        return 0.0

    # Clusters that matched nothing inherit the most similar decided cluster.
    if propagate:
        decided = result.index[result["phenotype"].notna()]
        undecided = result.index[result["phenotype"].isna()]
        if len(decided) and len(undecided):
            ref = profiles.loc[decided].to_numpy(dtype=float)
            ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
            for cluster in undecided:
                vec = profiles.loc[cluster].to_numpy(dtype=float)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                similarity = ref @ vec
                pick = int(np.argmax(similarity))
                source_cluster = decided[pick]
                result.loc[cluster, "phenotype"] = result.loc[source_cluster, "phenotype"]
                result.loc[cluster, "source"] = "similarity"
                result.loc[cluster, "confidence"] = float(similarity[pick])
                result.loc[cluster, "evidence"] = (
                    f"closest profile: cluster {source_cluster} (similarity {similarity[pick]:.2f})"
                )

    result["certainty"] = result.apply(_certainty, axis=1).round(3)
    return result


def seed_labels_from_reference(cluster_labels, reference_labels, min_purity=0.5,
                               min_cells=3, exclude=("Others", "Unassigned", "unclassified")):
    """Name clusters directly from cells that were annotated by hand.

    For every cluster holding annotated cells, the majority label wins provided
    it accounts for at least ``min_purity`` of them. Clusters whose annotated
    cells disagree too much are left undecided rather than given a coin-flip
    label, and clusters containing no annotated cells are simply absent.

    This is the single most accurate signal available: it uses real labels
    rather than inferring identity from marker profiles. On a 2.8M-cell
    benchmark it lifted phenotype F1 by about five points over signatures alone.

    Returns ``({cluster: phenotype}, stats)``.
    """
    import pandas as pd

    frame = pd.DataFrame({
        "cluster": np.asarray(cluster_labels),
        "label": pd.Series(reference_labels).astype("object").to_numpy(),
    })
    excluded = {str(e).lower() for e in exclude}
    frame = frame[frame["label"].notna()]
    frame = frame[~frame["label"].astype(str).str.lower().isin(excluded)]

    seeds, impure = {}, 0
    for cluster, group in frame.groupby("cluster"):
        counts = group["label"].astype(str).value_counts()
        if counts.sum() < int(min_cells):
            continue
        if counts.iloc[0] / counts.sum() >= float(min_purity):
            seeds[cluster] = counts.index[0]
        else:
            impure += 1
    stats = {
        "clusters_with_annotations": int(frame["cluster"].nunique()),
        "clusters_named": len(seeds),
        "clusters_too_mixed": impure,
        "annotated_cells_used": int(len(frame)),
    }
    return seeds, stats


# ---------------------------------------------------------------------------
# Assessment against a manual reference
# ---------------------------------------------------------------------------

def score_against_reference(cluster_labels, reference_labels, phenotype_map,
                            exclude=("Others", "Unassigned", "unclassified")):
    """Score an existing cluster->phenotype map against manually annotated cells.

    Reports macro F1 across the annotated phenotypes plus a per-phenotype
    breakdown. Macro (rather than weighted) F1 is the honest summary here: it
    gives a rare population the same weight as an abundant one, so a method that
    recovers only the common cell types cannot hide behind them.

    Annotated cells whose label is in ``exclude`` are reported separately rather
    than scored, since "Others" is a statement that the annotator could not
    assign a phenotype, not a phenotype to be predicted.

    IMPORTANT: this is only an unbiased estimate if ``phenotype_map`` was built
    without using these same annotations. If clusters were named from them, the
    result is circular; use :func:`cross_validate_reference_assignment` instead.
    """
    import pandas as pd
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    frame = pd.DataFrame({
        "cluster": [normalize_cluster_key(c) for c in np.asarray(cluster_labels)],
        "truth": pd.Series(reference_labels).astype("object").to_numpy(),
    })
    frame = frame[frame["truth"].notna()]
    if frame.empty:
        return None

    excluded = {str(e).lower() for e in exclude}
    is_excluded = frame["truth"].astype(str).str.lower().isin(excluded)
    held = frame[is_excluded]
    frame = frame[~is_excluded]
    if frame.empty:
        return None

    lookup = {normalize_cluster_key(k): v for k, v in (phenotype_map or {}).items()}
    predicted = frame["cluster"].map(lookup).fillna("Others").astype(str)
    truth = frame["truth"].astype(str)

    labels = sorted(truth.unique())
    macro = float(f1_score(truth, predicted, average="macro", labels=labels, zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    per_phenotype = pd.DataFrame({
        "phenotype": labels, "F1": f1, "precision": precision,
        "recall": recall, "annotated_cells": support,
    }).sort_values("F1")

    assigned = predicted != "Others"
    result = {
        "macro_f1": macro,
        "per_phenotype": per_phenotype,
        "n_scored": int(len(frame)),
        "coverage": float(assigned.mean()),
        "accuracy_of_assigned": float((predicted[assigned] == truth[assigned]).mean()) if assigned.any() else 0.0,
        "n_reference_others": int(len(held)),
    }
    if len(held):
        held_pred = held["cluster"].map(lookup).fillna("Others").astype(str)
        result["reference_others_given_a_phenotype"] = float((held_pred != "Others").mean())
    return result


def cross_validate_reference_assignment(
    cluster_labels, cluster_profiles, matrix, marker_names, reference_labels,
    n_splits=3, use_seeding=False, random_state=0, propagate=True,
    exclude=("Others", "Unassigned", "unclassified"),
    scoring_mode="profile", use_range=False, n_subprofiles=5,
):
    """Unbiased estimate of automatic phenotyping, via held-out annotations.

    The annotated cells are split into folds. Each fold in turn is held out: the
    marker signatures (and, when ``use_seeding`` is set, the cluster names taken
    from annotated cells) are built from the remaining folds only, then scored on
    the held-out fold. No cell ever contributes to both building and scoring, so
    the result estimates how the approach performs on cells nobody has annotated
    -- which is the situation for the rest of the dataset.

    ``use_seeding=False`` answers "how good are signatures learned from my
    reference?"; ``use_seeding=True`` additionally lets held-in annotations name
    clusters directly, which is the configuration used for the real assignment.
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    clusters = np.array([normalize_cluster_key(c) for c in np.asarray(cluster_labels)], dtype=object)
    truth_all = pd.Series(reference_labels).astype("object")
    mask = truth_all.notna().to_numpy()
    excluded = {str(e).lower() for e in exclude}
    mask &= ~truth_all.astype(str).str.lower().isin(excluded).to_numpy()
    idx = np.flatnonzero(mask)
    if idx.size < 30:
        return None
    y = truth_all.to_numpy()[idx].astype(str)

    # Folds need at least n_splits members of every class.
    counts = pd.Series(y).value_counts()
    n_splits = int(max(2, min(n_splits, counts.min())))

    fold_macro, all_truth, all_pred = [], [], []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for build_pos, test_pos in splitter.split(idx, y):
        build_rows, test_rows = idx[build_pos], idx[test_pos]

        # Everything the assignment relies on is rebuilt from the build folds
        # only, so the held-out cells never influence how they are judged.
        # Pass the FULL matrix with only the build-fold cells labelled. Handing in a
        # subset would z-score it on that subset's own statistics, putting the learned
        # profiles in a different space from the cluster profiles they are compared
        # against, which quietly distorts every distance.
        fold_labels = np.full(matrix.shape[0], None, dtype=object)
        fold_labels[build_rows] = y[build_pos]

        signatures = derive_signatures_from_reference(matrix, marker_names, fold_labels)
        learned = None
        if scoring_mode == "profile":
            learned = learn_phenotype_profiles(
                matrix, marker_names, fold_labels, n_subprofiles=n_subprofiles,
            )
        if not signatures and not learned:
            continue
        seeds = None
        if use_seeding:
            seeds, _ = seed_labels_from_reference(clusters[build_rows], y[build_pos])
        table = recommend_phenotypes(
            cluster_profiles, signatures, seed_labels=seeds, propagate=propagate,
            phenotype_profiles=learned,
            profile_marker_names=marker_names if learned else None,
            use_range=use_range,
        )
        mapping = {normalize_cluster_key(c): r["phenotype"]
                   for c, r in table.iterrows() if pd.notna(r["phenotype"])}

        pred = pd.Series(clusters[test_rows]).map(mapping).fillna("Others").astype(str).to_numpy()
        truth = y[test_pos]
        labels = sorted(set(truth))
        fold_macro.append(f1_score(truth, pred, average="macro", labels=labels, zero_division=0))
        all_truth.append(truth); all_pred.append(pred)

    if not fold_macro:
        return None
    truth = np.concatenate(all_truth); pred = np.concatenate(all_pred)
    labels = sorted(set(truth))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=labels, zero_division=0
    )
    assigned = pred != "Others"
    return {
        "macro_f1": float(np.mean(fold_macro)),
        "macro_f1_sd": float(np.std(fold_macro)),
        "n_splits": n_splits,
        "per_phenotype": pd.DataFrame({
            "phenotype": labels, "F1": f1, "precision": precision,
            "recall": recall, "annotated_cells": support,
        }).sort_values("F1"),
        "n_scored": int(len(truth)),
        "coverage": float(assigned.mean()),
        "accuracy_of_assigned": float((pred[assigned] == truth[assigned]).mean()) if assigned.any() else 0.0,
        "used_seeding": bool(use_seeding),
    }


def normalize_cluster_key(value):
    """Coerce a cluster id to a stable key (int where possible)."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return value


# ---------------------------------------------------------------------------
# Fast CSV reading
# ---------------------------------------------------------------------------

def read_csv_fast(path, logger=None, columns=None):
    """Read a table, preferring the Parquet fast copy and then pyarrow.

    On the multi-GB batch-corrected tables this is far faster than pandas'
    single-threaded parser and produces an equivalent frame -- verified
    column-for-column, including null counts, against ``pd.read_csv``.
    Passing *columns* reads only those, which is the difference between minutes
    and about a second on a 13 GB table.
    """
    import pandas as pd

    try:
        # The shared loader reads the Parquet fast copy when one is current,
        # which skips unwanted columns on disk rather than parsing past them,
        # and otherwise uses the same threaded pyarrow CSV reader as before.
        from .dataio import load as _dataio_load

        return _dataio_load(path, columns=list(columns) if columns else None)
    except Exception as exc:
        _log(logger, f"Shared loader unavailable ({exc}); using pandas.")
        return pd.read_csv(path, usecols=list(columns) if columns else None, low_memory=False)
