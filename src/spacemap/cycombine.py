"""cyCombine's batch correction, ported from its R code so that it gives the same result.

cyCombine (Pedersen et al., Nat Commun 2022; https://github.com/biosurf/cyCombine) corrects each marker in
three steps, all reproduced here from the R sources:

1. ``normalize`` (norm_method "scale"): every marker z-scored within each batch, only to group similar cells.
2. ``create_som``: a kohonen::som (8 x 8 rectangular grid, bubble neighbourhood, online training, rlen 10,
   alpha 0.05 to 0.01, radius from the 2/3 quantile of the grid distances to 0, seed 473) on the z-scores; each
   cell's nearest unit is its group. R's random numbers (Mersenne-Twister after ``set.seed``, ``sample`` with
   rejection sampling) are reproduced too, so the same seed gives the same groups as R.
3. ``correct_data`` (the table workflow, as in released cyCombine 0.2 and ``batch_correct_df`` of 0.3): within
   each group holding cells of two or more batches, sva::ComBat (parametric empirical Bayes by default) on the
   measured values (arcsinh), with the covariate and anchor checks cyCombine makes per group, then each marker
   clipped to the group's own input range. Groups holding one batch are left as they are.

Checked against R 4.5.2 with cyCombine 0.3.0, kohonen 3.0.13 and sva 3.48.0 (tests/test_cycombine.py).
Where R would give missing values, this port keeps numbers instead: a marker constant but not zero within a
batch is z-scored to 0 (R: NaN); a group ComBat cannot estimate (two cells of two batches, say) keeps its
measured values (R: NaN); non-parametric ComBat works in log-space (R's likelihoods underflow to 0/0 on groups
of thousands of cells).
"""

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a SPACEMAP dependency
    njit = None

_EPS = 1e-8  # kohonen: relative test of equality of distances
_DBL_MAX = np.finfo(np.float64).max
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1), R's fixup of 0 and 1


def _jit(function):
    return njit(cache=True)(function) if njit else function


# ------------------------------------------------------------------ R's random numbers


_UPPER, _LOWER, _MATRIX_A = np.uint64(0x80000000), np.uint64(0x7FFFFFFF), np.uint64(0x9908B0DF)
_MASK_B, _MASK_C, _MASK_32 = np.uint64(0x9D2C5680), np.uint64(0xEFC60000), np.uint64(0xFFFFFFFF)
_S1, _S7, _S11, _S15, _S18, _ONE = (np.uint64(v) for v in (1, 7, 11, 15, 18, 1))


@_jit
def _mt_next(mt, pos):
    """The next 32-bit Mersenne-Twister output (R's MT_genrand, before scaling)."""
    if pos[0] >= 624:
        for kk in range(624):
            y = (mt[kk] & _UPPER) | (mt[(kk + 1) % 624] & _LOWER)
            value = mt[(kk + 397) % 624] ^ (y >> _S1)
            if y & _ONE:
                value ^= _MATRIX_A
            mt[kk] = value
        pos[0] = 0
    y = mt[pos[0]]
    pos[0] += 1
    y ^= y >> _S11
    y ^= (y << _S7) & _MASK_B
    y ^= (y << _S15) & _MASK_C
    y &= _MASK_32
    y ^= y >> _S18
    return y


@_jit
def _unif(mt, pos):
    """R's unif_rand(): in (0, 1), never 0 or 1."""
    value = _mt_next(mt, pos) * 2.3283064365386963e-10
    if value <= 0.0:
        return 0.5 * _I2_32M1
    if 1.0 - value <= 0.0:
        return 1.0 - 0.5 * _I2_32M1
    return value


class RRandom:
    """R's default random numbers after ``set.seed(seed)``: Mersenne-Twister, rejection sampling."""

    def __init__(self, seed):
        value = int(seed) % 2**32
        for _ in range(50):  # R's initial scrambling
            value = (69069 * value + 1) % 2**32
        seeds = []
        for _ in range(625):
            value = (69069 * value + 1) % 2**32
            seeds.append(value)
        # seeds[0] is R's position slot, reset to 624 so the first draw generates a new block.
        self.mt = np.array(seeds[1:], dtype=np.uint64)
        self.pos = np.array([624], dtype=np.int64)

    def unif(self):
        return float(_unif(self.mt, self.pos))

    def _index(self, dn):
        """R_unif_index: an integer in [0, dn) by rejection sampling of random bits."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):
                v = 65536 * v + int(math.floor(self.unif() * 65536))
            v &= (1 << bits) - 1
            if dn > v:
                return v

    def sample(self, n, size):
        """``sample(1:n, size)`` (without replacement), 1-based, as R draws it."""
        if n > 1e7 and size <= n / 2:  # sample.int's hashing path (sample2)
            chosen, seen = [], set()
            while len(chosen) < size:
                v = self._index(n)
                if v not in seen:
                    seen.add(v)
                    chosen.append(v + 1)
            return np.array(chosen, dtype=np.int64)
        pool = list(range(n))
        chosen = []
        for _ in range(size):
            j = self._index(n)
            chosen.append(pool[j] + 1)
            n -= 1
            pool[j] = pool[n]
        return np.array(chosen, dtype=np.int64)


# ------------------------------------------------------------------ 1. normalize (scale within each batch)


def _r_sum(values):
    """R's sum(): accumulated in long double, in order."""
    return np.cumsum(values, dtype=np.longdouble)[-1]


def normalize(values, batches, norm_method="scale"):
    """Per-batch SOM input: R's scale(), rank / n (average ties), or unchanged values."""
    if norm_method not in ("scale", "rank", "none"):
        raise ValueError("SOM normalization must be scale, rank or none.")
    values = np.asarray(values, dtype=np.float64)
    batches = np.asarray(batches)
    out = values.copy()
    if norm_method == "none":
        return out
    for batch in np.unique(batches):
        rows = np.flatnonzero(batches == batch)
        for column in range(values.shape[1]):
            v = values[rows, column]
            total = _r_sum(v)
            if float(total) == 0:  # cyCombine leaves an all-zero marker as it is
                continue
            if norm_method == "rank":
                from scipy.stats import rankdata

                out[rows, column] = rankdata(v, method="average") / len(v)
                continue
            centred = v - float(total / np.longdouble(v.size))
            sd = math.sqrt(float(_r_sum(centred * centred)) / max(1, v.size - 1))
            out[rows, column] = centred / sd if sd > 0 else 0.0
    return out


# ------------------------------------------------------------------ 2. create_som (kohonen::som, online)


@_jit
def _nearest(x, codes, mt, pos):
    """kohonen's FindBestMatchingUnit: Euclidean distance; near-ties (1e-8 relative) broken at random."""
    nind = 1
    index = -1
    distance = _DBL_MAX
    for cd in range(codes.shape[0]):
        d = 0.0
        for j in range(x.shape[0]):
            t = x[j] - codes[cd, j]
            d += t * t
        dist = math.sqrt(d)
        if dist <= distance * (1 + _EPS):
            if dist < distance * (1 - _EPS):
                nind = 1
                index = cd
            else:
                nind += 1
                if nind * _unif(mt, pos) < 1.0:
                    index = cd
            distance = dist
    return index


@_jit
def _train(data, codes, neighbours, alphas, radii, rlen, mt, pos):
    """kohonen's RcppSupersom, online mode, one layer, bubble neighbourhood."""
    n = data.shape[0]
    total = rlen * n
    step = 0
    for _ in range(rlen):
        for _ in range(n):
            i = int(n * _unif(mt, pos))
            nearest = _nearest(data[i], codes, mt, pos)
            tmp = step / total
            threshold = radii[0] - (radii[0] - radii[1]) * tmp
            if threshold < 1.0:
                threshold = 0.5
            alpha = alphas[0] - (alphas[0] - alphas[1]) * tmp
            for cd in range(codes.shape[0]):
                if neighbours[cd, nearest] <= threshold:
                    for j in range(data.shape[1]):
                        codes[cd, j] += alpha * (data[i, j] - codes[cd, j])
            step += 1


@_jit
def _map(data, codes, mt, pos):
    labels = np.empty(data.shape[0], dtype=np.int64)
    for i in range(data.shape[0]):
        labels[i] = _nearest(data[i], codes, mt, pos) + 1
    return labels


def grid_distances(xdim, ydim):
    """kohonen's unit.distances for a rectangular, non-toroidal grid (the maximum of the x and y offsets)."""
    x, y = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    points = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)  # expand.grid: x varies fastest
    return np.abs(points[:, None, :] - points[None, :, :]).max(axis=2)


def create_som(normed, xdim=8, ydim=8, rlen=10, seed=473):
    """cyCombine's create_som (kohonen): each cell's SOM unit, 1-based, the same as R for the same seed."""
    data = np.ascontiguousarray(normed, dtype=np.float64)
    random = RRandom(seed)
    units = xdim * ydim
    if len(data) < units:
        raise ValueError(f"The {xdim} × {ydim} SOM needs at least {units} valid cells. Choose a smaller SOM grid.")
    codes = np.ascontiguousarray(data[random.sample(len(data), units) - 1])
    neighbours = grid_distances(xdim, ydim)
    radius = float(np.quantile(neighbours, 2 / 3))  # R's quantile type 7, over the whole matrix
    _train(
        data,
        codes,
        neighbours,
        np.array([0.05, 0.01]),
        np.array([radius, 0.0]),
        int(rlen),
        random.mt,
        random.pos,
    )
    return _map(data, codes, random.mt, random.pos)


# ------------------------------------------------------------------ 3. ComBat (sva::ComBat)


def _aprior(delta):
    m, s2 = delta.mean(), delta.var(ddof=1)
    return (2 * s2 + m**2) / s2


def _bprior(delta):
    m, s2 = delta.mean(), delta.var(ddof=1)
    return (m * s2 + m**3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2 + a - 1)


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4):
    n = sdat.shape[1]
    g_old, d_old = g_hat, d_hat
    change = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        while change > conv:
            g_new = _postmean(g_hat, g_bar, n, d_old, t2)
            sum2 = ((sdat - g_new[:, None]) ** 2).sum(axis=1)
            d_new = _postvar(sum2, n, a, b)
            change = max(np.max(np.abs(g_new - g_old) / g_old), np.max(np.abs(d_new - d_old) / d_old))
            g_old, d_old = g_new, d_new
    return g_new, d_new


def _int_eprior(sdat, g_hat, d_hat):
    """sva's int.eprior (non-parametric), with the likelihoods in log-space so large groups do not underflow."""
    g_star, d_star = [], []
    n = sdat.shape[1]
    for i in range(sdat.shape[0]):
        g, d = np.delete(g_hat, i), np.delete(d_hat, i)
        sum2 = ((sdat[i][None, :] - g[:, None]) ** 2).sum(axis=1)
        log_lh = -(n / 2) * np.log(2 * np.pi * d) - sum2 / (2 * d)
        weights = np.exp(log_lh - log_lh.max())
        g_star.append((g * weights).sum() / weights.sum())
        d_star.append((d * weights).sum() / weights.sum())
    return np.array(g_star), np.array(d_star)


def combat(dat, batch, mod=None, par_prior=True, mean_only=False, ref_batch=None):
    """sva::ComBat on *dat* (markers x cells) for the *batch* of each cell; *mod*: covariate columns (cells x k)."""
    dat = np.asarray(dat, dtype=np.float64)
    batch = np.asarray(batch).astype(str)
    levels = sorted(set(batch))
    # Markers constant within a batch (of two or more cells) are not adjusted.
    zero = sorted(
        {
            row
            for level in levels
            if (batch == level).sum() > 1
            for row in np.flatnonzero(dat[:, batch == level].var(axis=1, ddof=1) == 0)
        }
    )
    keep = np.setdiff1d(np.arange(dat.shape[0]), zero)
    if not keep.size:
        return dat.copy()
    original = dat
    dat = dat[keep]
    batches = [np.flatnonzero(batch == level) for level in levels]
    n_batches = np.array([len(rows) for rows in batches])
    if (n_batches == 1).any():
        mean_only = True
    n_batch, n_array = len(levels), len(batch)
    batchmod = np.zeros((n_array, n_batch))
    for index, rows in enumerate(batches):
        batchmod[rows, index] = 1.0
    ref = None
    if ref_batch is not None:
        if str(ref_batch) not in levels:
            raise ValueError("The reference batch is not one of the batches.")
        ref = levels.index(str(ref_batch))
        batchmod[:, ref] = 1.0
    design = batchmod if mod is None else np.column_stack([batchmod, np.asarray(mod, dtype=np.float64)])
    constant = np.all(design == 1, axis=0)
    if ref is not None:
        constant[ref] = False
    design = design[:, ~constant]
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("A covariate is confounded with batch.")
    b_hat = np.linalg.solve(design.T @ design, design.T @ dat.T)
    if ref is not None:
        grand_mean = b_hat[ref]
        rows = batches[ref]
        var_pooled = ((dat[:, rows] - (design[rows] @ b_hat).T) ** 2).mean(axis=1)
    else:
        grand_mean = (n_batches / n_array) @ b_hat[:n_batch]
        var_pooled = ((dat - (design @ b_hat).T) ** 2).mean(axis=1)
    covariates = design.copy()
    covariates[:, :n_batch] = 0
    stand_mean = grand_mean[:, None] + (covariates @ b_hat).T
    s_data = (dat - stand_mean) / np.sqrt(var_pooled)[:, None]
    batch_design = design[:, :n_batch]
    gamma_hat = np.linalg.solve(batch_design.T @ batch_design, batch_design.T @ s_data.T)
    if mean_only:
        delta_hat = np.ones((n_batch, dat.shape[0]))
    else:
        delta_hat = np.array([s_data[:, rows].var(axis=1, ddof=1) for rows in batches])
    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)
    gamma_star = np.empty_like(gamma_hat)
    delta_star = np.empty_like(delta_hat)
    for i, rows in enumerate(batches):
        if par_prior:
            if mean_only:
                gamma_star[i] = _postmean(gamma_hat[i], gamma_bar[i], 1, 1, t2[i])
                delta_star[i] = 1.0
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    a, b = _aprior(delta_hat[i]), _bprior(delta_hat[i])
                gamma_star[i], delta_star[i] = _it_sol(
                    s_data[:, rows], gamma_hat[i], delta_hat[i], gamma_bar[i], t2[i], a, b
                )
        else:
            gamma_star[i], delta_star[i] = _int_eprior(
                s_data[:, rows], gamma_hat[i], np.ones(dat.shape[0]) if mean_only else delta_hat[i]
            )
    if ref is not None:
        gamma_star[ref], delta_star[ref] = 0.0, 1.0
    adjusted = s_data.copy()
    for j, rows in enumerate(batches):
        adjusted[:, rows] = (adjusted[:, rows] - (batch_design[rows] @ gamma_star).T) / np.sqrt(delta_star[j])[:, None]
    adjusted = adjusted * np.sqrt(var_pooled)[:, None] + stand_mean
    if ref is not None:
        adjusted[:, batches[ref]] = dat[:, batches[ref]]
    out = original.copy()
    out[keep] = adjusted
    return out


# ------------------------------------------------------------------ correct_data (per SOM group)


def _dummies(codes, levels):
    """R's model.matrix(~ factor): an intercept, then one column per level after the first (unused levels too)."""
    return np.column_stack([np.ones(len(codes))] + [(codes == level).astype(np.float64) for level in levels[1:]])


def _confounded(batch, mod):
    """cyCombine's check_confound: whether the covariate model is confounded with batch."""
    levels = sorted(set(batch))
    batchmod = np.column_stack([(batch == level).astype(np.float64) for level in levels])
    design = np.column_stack([batchmod, mod])
    design = design[:, ~np.all(design == 1, axis=0)]
    return np.linalg.matrix_rank(design) < design.shape[1]


def _skewed(codes, levels):
    """cyCombine: a group almost all of one covariate level is treated as having one level."""
    counts = np.array([(codes == level).sum() for level in levels])
    return counts.sum() < counts.max() + len(levels) * 5


def correct_data(values, batches, labels, covar=None, anchor=None, parametric=True, ref_batch=None, report=None):
    """cyCombine's correct_data: ComBat within each label group, then each marker clipped to the group's range.

    *values*: cells x markers (arcsinh), *covar* and *anchor*: one value per cell or None. Returns (corrected,
    summary) where summary counts the groups corrected, left alone (one batch) and where a covariate was used.
    """
    values = np.asarray(values, dtype=np.float64)
    batches = np.asarray(batches).astype(str)
    labels = np.asarray(labels)
    corrected = values.copy()
    factors = {}
    for name, column in (("covar", covar), ("anchor", anchor)):
        if column is None:
            continue
        column = np.asarray(column).astype(str)
        levels = sorted(set(column))
        if len(levels) > 1:
            factors[name] = (column, levels)
    summary = {"groups": 0, "one_batch": 0, "covar_used": 0, "anchor_used": 0,
               "covar_omitted": 0, "one_batch_cells": 0, "cells_kept_as_measured": 0}
    groups = sorted(set(labels.tolist()))
    for number, label in enumerate(groups):
        rows = np.flatnonzero(labels == label)
        if len(set(batches[rows])) == 1:
            summary["one_batch"] += 1
            summary["one_batch_cells"] += len(rows)
            continue
        summary["groups"] += 1
        batch = batches[rows]
        use = {}
        for name, (column, levels) in factors.items():
            mod = _dummies(column[rows], levels)
            if not _confounded(batch, mod) and not _skewed(column[rows], levels):
                use[name] = mod
            elif name == "covar":
                summary["covar_omitted"] += 1
        if len(use) == 2 and _confounded(batch, np.column_stack([use["covar"], use["anchor"][:, 1:]])):
            use.pop("anchor")
        mod = None
        if use:
            mod = np.column_stack([use[name] if index == 0 else use[name][:, 1:] for index, name in enumerate(use)])
            for name in use:
                summary[name + "_used"] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            adjusted = combat(values[rows].T, batch, mod=mod, par_prior=parametric, ref_batch=ref_batch).T
        low, high = values[rows].min(axis=0), values[rows].max(axis=0)
        adjusted = np.clip(adjusted, low, high)
        # ComBat cannot estimate anything from a group like two cells of two batches: R returns missing values
        # there; those cells keep their measured values instead.
        missing = ~np.isfinite(adjusted)
        if missing.any():
            summary["cells_kept_as_measured"] += int(missing.any(axis=1).sum())
            adjusted = np.where(missing, values[rows], adjusted)
        corrected[rows] = adjusted
        if report and number % 8 == 0:
            report(f"ComBat in SOM group {number + 1} of {len(groups)}")
    return corrected, summary


def batch_correct(
    values,
    batches,
    covar=None,
    anchor=None,
    xdim=8,
    ydim=8,
    rlen=10,
    seed=473,
    parametric=True,
    ref_batch=None,
    labels=None,
    report=None,
    norm_method="scale",
):
    """cyCombine's batch_correct_df on *values* (cells x markers, already arcsinh-transformed).

    Returns (corrected values, SOM unit of each cell, summary).
    """
    values = np.asarray(values, dtype=np.float64)
    batches = np.asarray(batches).astype(str)
    if labels is None:
        if report:
            report(f"Normalizing markers within each batch ({norm_method}, for the SOM only)")
        normed = normalize(values, batches, norm_method=norm_method)
        if report:
            report(f"Training the {xdim} x {ydim} SOM ({rlen} passes over {len(values):,} cells)")
        labels = create_som(normed, xdim=xdim, ydim=ydim, rlen=rlen, seed=seed)
    corrected, summary = correct_data(
        values, batches, labels, covar=covar, anchor=anchor, parametric=parametric, ref_batch=ref_batch, report=report
    )
    return corrected, labels, summary
