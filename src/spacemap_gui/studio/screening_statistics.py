"""Declared multiplicity scopes and biological-sample resampling for screening."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

SCOPES = {
    "global": "Whole scan (all types and zones)",
    "family": "Each feature type (all zones)",
    "classic": "Each feature type and zone (Classic)",
}


def adjust(results, c):
    """Retain every scope; q always identifies the explicitly selected one."""
    result = results.copy()
    for scope, columns in [("global", []), ("family", ["family"]), ("classic", ["family", "zone"])]:
        if scope == "classic" and "comparison" in result:
            columns = ["comparison", *columns]
        result["q_" + scope] = np.nan
        result["n_tests_" + scope] = 0
        groups = result.groupby(columns, sort=False).groups.values() if columns else [result.index]
        for idx in groups:
            p = result.loc[idx, "p"]
            # Classic excludes untestable rows. Complete-search scopes retain
            # the declared combinations as p=1, never inventing a reported q.
            use = p.dropna() if scope == "classic" else p.fillna(1)
            if len(use):
                result.loc[use.index, "q_" + scope] = multipletests(use, method=c["correction"])[1]
            result.loc[idx, "n_tests_" + scope] = len(use)
            result.loc[p.index[p.isna()], "q_" + scope] = np.nan
    scope = c.get("correction_scope", "global")
    result["q"] = result["q_" + scope]
    result["correction_scope"] = scope
    result["correction_n"] = result["n_tests_" + scope]
    return result


def _fast_mannwhitney(matrix, groups, c):
    """Vectorize columns with matching missingness and SciPy auto-test method."""
    e = c["endpoint"]
    a, b = matrix[groups == e["group_a"]], matrix[groups == e["group_b"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        effect = np.nanmedian(b, axis=0) - np.nanmedian(a, axis=0)
    p = np.full(matrix.shape[1], np.nan)
    # Group equal coverage masks, then separate exact/asymptotic columns: SciPy
    # chooses auto globally for a vectorized batch if any column contains ties.
    masks = np.isfinite(matrix)
    _, inverse = np.unique(masks.T, axis=0, return_inverse=True)
    for key in np.unique(inverse):
        cols = np.flatnonzero(inverse == key)
        mask = masks[:, cols[0]]
        x = matrix[np.ix_(mask & (groups == e["group_a"]), cols)]
        y = matrix[np.ix_(mask & (groups == e["group_b"]), cols)]
        if min(len(x), len(y)) < c["min_samples"]:
            continue
        combined = np.sort(np.concatenate([x, y]), axis=0)
        ties = (np.diff(combined, axis=0) == 0).any(axis=0)
        exact = (~ties) & (min(len(x), len(y)) <= 8)
        for flag, method in [(True, "exact"), (False, "asymptotic")]:
            choose = exact == flag
            if choose.any():
                p[cols[choose]] = stats.mannwhitneyu(x[:, choose], y[:, choose], axis=0,
                    alternative="two-sided", method=method).pvalue
    return effect, p


def resample(results, merged, c, infer, report=lambda *args: None, subject_draws=None, return_draws=False):
    """Resample whole subjects coherently across every feature and distance.

    Classic subsets are without replacement. Bootstrap is with replacement,
    stratified by group for independent binary comparisons, paired otherwise.
    Recovery requires adjusted significance AND the original effect direction.
    Bootstrap percentile intervals are pointwise, not post-selection intervals.
    """
    mode = c.get("resampling", "off")
    if mode == "off" or c["endpoint"]["type"] == "describe":
        return results
    c = dict(c)
    e = c["endpoint"]
    n = int(c["resamples"])
    rng = np.random.default_rng(int(c["seed"]) + 137)
    result = results.copy()
    keys = list(zip(result.feature, result.zone))
    units = merged[["_sample", "_group"]].drop_duplicates().reset_index(drop=True)
    subjects = units._sample.unique()
    matrix = merged.pivot(index=["_sample", "_group"], columns=["feature", "zone"], values="value").reindex(
        index=pd.MultiIndex.from_frame(units), columns=pd.MultiIndex.from_tuples(keys)).to_numpy(float)
    effect_draws = np.full((n, len(result)), np.nan)
    p_draws = np.full_like(effect_draws, np.nan) if return_draws else None
    tested_draws = np.zeros((n, len(result)), bool)
    recovered = np.zeros_like(tested_draws)
    fast = e["type"] == "binary" and not e["paired"] and c["group_test"] == "mannwhitney"
    original_direction = np.sign(np.log(result.effect.to_numpy(float))) if e["type"] == "survival" else np.sign(result.effect.to_numpy(float))
    for iteration in range(n):
        if subject_draws is not None:
            available = set(subjects)
            chosen = [s for s in subject_draws[iteration] if s in available]
        elif mode == "subsample":
            # Share Classic's actual sampler, including its fraction/rounding.
            from spacemap_gui.tabs.tab8_analysis.feature_screening_core import _sample_bootstrap_unit_keys
            source = pd.DataFrame({"__bootstrap_key__": subjects, "stratum_label": "All"})
            chosen = _sample_bootstrap_unit_keys(source, float(c["sample_fraction"]), rng, min_units_per_stratum=2)
        elif e["type"] == "binary" and not e["paired"]:
            chosen = []
            for group in [e["group_a"], e["group_b"]]:
                ids = units.loc[units._group == group, "_sample"].to_numpy()
                chosen.extend(rng.choice(ids, len(ids), replace=True))
        else:
            chosen = rng.choice(subjects, len(subjects), replace=True).tolist()
        if not len(chosen):
            continue
        if fast:
            lookup = dict(zip(units._sample, range(len(units))))
            indices = [lookup[s] for s in chosen]
            effect, p = _fast_mannwhitney(matrix[indices], units._group.to_numpy()[indices], c)
            draw = result.assign(effect=effect, p=p)
            if not return_draws:
                draw = adjust(draw, c)
        else:
            # Assign new IDs to bootstrap copies; preserve both paired groups
            # and every feature/zone for each sampled subject.
            parts = [merged.loc[merged._sample == s].assign(_sample=str(j)) for j, s in enumerate(chosen)]
            frame = pd.concat(parts, ignore_index=True)
            outcome_cols = [e["group_col"]] if e["type"] == "binary" else [e["value_col"]] if e["type"] == "continuous" else [e["time_col"], e["event_col"]]
            outcomes = frame[["_sample", "_group", *outcome_cols]].drop_duplicates()
            draw, _ = infer(frame.drop(columns=outcome_cols), outcomes, c)
            draw = draw.set_index(["feature", "zone"]).reindex(pd.MultiIndex.from_tuples(keys))
            effect, p = draw.effect.to_numpy(float), draw.p.to_numpy(float)
        effect_draws[iteration] = effect
        if return_draws:
            p_draws[iteration] = p
        tested_draws[iteration] = np.isfinite(p)
        direction = np.sign(np.log(effect)) if e["type"] == "survival" else np.sign(effect)
        recovered[iteration] = (draw.q.to_numpy(float) < c["alpha"]) & (direction == original_direction) & tested_draws[iteration]
        if iteration == 0 or (iteration + 1) % max(1, n // 20) == 0:
            report(f"Sample {'subset stability' if mode == 'subsample' else 'bootstrap'}: {iteration+1}/{n}", .78 + .1 * (iteration+1)/n)
    if return_draws:
        return effect_draws, p_draws
    result["resampling"] = mode
    result["resamples"] = n
    result["valid_resamples"] = tested_draws.sum(axis=0)
    result["recovery"] = recovered.mean(axis=0)  # untestable repeats count as failures
    near = recovered.copy()
    for i, (feature, zone) in enumerate(keys):
        neighbours = [j for j, (f, z) in enumerate(keys) if f == feature and abs(z-zone) <= int(c["nearby_zones"])
                      and original_direction[j] == original_direction[i]]
        near[:, i] = recovered[:, neighbours].any(axis=1)
    result["nearby_recovery"] = near.mean(axis=0)
    result["ci_low"], result["ci_high"] = np.nan, np.nan
    if mode == "bootstrap":
        for i in range(len(result)):
            draws = effect_draws[tested_draws[:, i], i]
            if len(draws) >= max(20, .8 * n):
                result.loc[result.index[i], ["ci_low", "ci_high"]] = np.quantile(draws, [.025, .975])
    # An originally untestable comparison has no defensible reference direction.
    result.loc[result.p.isna(), ["recovery", "nearby_recovery", "ci_low", "ci_high"]] = np.nan
    return result


def resample_joint(results, comparisons, c, infer, report=lambda *args: None):
    """One subject draw across all selected questions, followed by joint FDR.

    comparisons maps IDs to (merged sample values, comparison settings).
    Multi-question bootstrap is unstratified at the independent-subject level,
    preserving each person's visits and membership of overlapping subgroups.
    """
    if c["resampling"] == "off":
        return results
    result = results.reset_index(drop=True).copy()
    n = int(c["resamples"])
    if len(result) * n > 20_000_000:
        raise ValueError("This resampling plan exceeds 20 million test-repeats. Reduce comparisons or repeat count.")
    subjects = np.array(sorted(set().union(*(set(frame._sample) for frame, _ in comparisons.values()))))
    rng = np.random.default_rng(int(c["seed"]) + 137)
    if c["resampling"] == "subsample":
        size = min(len(subjects), max(2, int(np.ceil(len(subjects)*c["sample_fraction"]))))
        draws = [rng.choice(subjects, size, replace=False).tolist() for _ in range(n)]
    else:
        draws = [rng.choice(subjects, len(subjects), replace=True).tolist() for _ in range(n)]
    effects = np.full((n, len(result)), np.nan)
    pvalues = np.full_like(effects, np.nan)
    direction = np.sign(result.effect.to_numpy(float))
    survival = np.zeros(len(result), bool)
    for number, (key, (merged, cfg)) in enumerate(comparisons.items()):
        if cfg["endpoint"]["type"] == "describe":
            continue
        indices = np.flatnonzero(result.comparison.to_numpy() == key)
        sub = result.iloc[indices]
        updated = dict(cfg, resampling=c["resampling"], resamples=n)
        progress = lambda message, *_, k=number: report(f"Comparison {k+1}/{len(comparisons)} · {message}", None)
        a, p = resample(sub, merged, updated, infer, progress, draws, True)
        effects[:, indices], pvalues[:, indices] = a, p
        if cfg["endpoint"]["type"] == "survival":
            survival[indices] = True
            direction[indices] = np.sign(np.log(sub.effect.to_numpy(float)))
    recovered = np.zeros(effects.shape, bool)
    for i in range(n):
        adjusted = adjust(result.assign(p=pvalues[i]), c)
        sign = np.sign(effects[i])
        sign[survival] = np.sign(np.log(effects[i, survival]))
        recovered[i] = (adjusted.q.to_numpy() < c["alpha"]) & (sign == direction) & np.isfinite(pvalues[i])
    result["resampling"], result["resamples"] = c["resampling"], n
    result["valid_resamples"] = np.isfinite(pvalues).sum(axis=0)
    result["recovery"] = recovered.mean(axis=0)
    result["nearby_recovery"] = result.recovery
    for _, indices in result.groupby(["comparison", "feature"]).groups.items():
        indices = np.array(list(indices))
        for i in indices:
            near = indices[(np.abs(result.loc[indices, "zone"].to_numpy()-result.loc[i, "zone"]) <= c["nearby_zones"]) & (direction[indices] == direction[i])]
            result.loc[i, "nearby_recovery"] = recovered[:, near].any(axis=1).mean()
    result["ci_low"], result["ci_high"] = np.nan, np.nan
    if c["resampling"] == "bootstrap":
        for i in range(len(result)):
            values = effects[np.isfinite(pvalues[:, i]), i]
            if len(values) >= max(20, .8*n):
                result.loc[i, ["ci_low", "ci_high"]] = np.quantile(values, [.025,.975])
    result.loc[result.p.isna(), ["recovery", "nearby_recovery", "ci_low", "ci_high"]] = np.nan
    return result
