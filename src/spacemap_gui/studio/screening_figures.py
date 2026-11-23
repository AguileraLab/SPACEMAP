"""Readable screening exports: distance effects, sample dots, and coverage."""
from pathlib import Path
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .screening import DIVERSITY, FAMILIES, diversity_metrics
from .screening_statistics import SCOPES


def zone_label(c, zone):
    if c["zone_source"] == "whole":
        return "Whole region"
    j = int(zone)
    lo = c["widths"][j-1] if j and c["zone_mode"] == "rings" else 0
    return f"{lo:g}–{c['widths'][j]:g} µm ({c['side']})"


def measurement(c, family, feature=None):
    label = _measurement(c, family, feature)
    if c["endpoint"].get("change_col"):
        label = "Change in " + label
    return label


def _measurement(c, family, feature=None):
    if family == "density":
        return "Cells / estimated mm²" if c["zone_source"] != "polygons" and c["options"]["density"].get("area_source") == "estimate" else "Cells / mm²"
    if family == "proportion":
        return "% of denominator cells"
    if family == "ratio":
        return "Log2 cell-count ratio" if c["options"][family]["log2"] else "Cell-count ratio"
    if family == "diversity":
        metrics = diversity_metrics(c)
        metric = json.loads(feature)[1] if feature else metrics[0] if len(metrics) == 1 else None
        return DIVERSITY.get(metric, "Diversity measure")
    if family == "nn_distance":
        return "Mean neighbour distance (µm)"
    return {"fraction": "% of geometric motifs", "count": "Number found", "zscore": "Random-label Z-score (Classic)",
            "enrichment": "Log2 observed / expected (pseudocount)"}[c["options"][family]["metric"]]


def statistics_text(row, c):
    def fmt(value):
        return f"{value:.3g}" if np.isfinite(value) else "unavailable"
    if c["endpoint"]["type"] == "describe":
        return f"Descriptive · {int(row.n_samples)} biological samples · no significance test"
    effect = f"Effect = {fmt(row.effect)}"
    text = f"{effect} · p = {fmt(row.p)} · q = {fmt(row.q)}\n{row.test} · {int(row.n_samples)} samples"
    if row.get("resampling") in ("subsample", "bootstrap"):
        text += f"\nRecovery: {fmt(100 * row.recovery)}% exact / {fmt(100 * row.nearby_recovery)}% nearby ({int(row.resamples)} repeats)"
        if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
            text += f"\nEffect 95% bootstrap interval: {fmt(row.ci_low)} to {fmt(row.ci_high)} (pointwise)"
    return text


def save(out, results, samples, c, palette=None):
    folder = Path(out) / "figures"
    folder.mkdir()
    files = []
    colors = ["#2878a5", "#c55c32"]
    palette = palette or {}
    panels = []
    for family in c["families"]:
        metrics = diversity_metrics(c) if family == "diversity" else [None]
        for metric in metrics:
            stem = f"diversity_{metric}" if family == "diversity" and len(metrics) > 1 else family
            panels.append((family, metric, stem))
    for family, metric, stem in panels:
        rows = results[results.family == family]
        if metric:
            rows = rows[rows.feature == json.dumps(["diversity", metric])]
        if rows.empty:
            continue
        best = rows.drop_duplicates("feature").head(10)
        figure = plt.figure(figsize=(14, 7.5 + .25 * len(best) + (1 if c.get("resampling", "off") != "off" else 0)), layout="constrained")
        grid = figure.add_gridspec(2, 2, height_ratios=[max(.55, .15 * len(best)), 1])
        heat = figure.add_subplot(grid[0, :])
        pivot = rows.pivot(index="feature", columns="zone", values="effect").reindex(best.feature)
        values = pivot.to_numpy(float)
        if c["endpoint"]["type"] == "survival":
            values = np.log2(values)
        if c["endpoint"]["type"] == "describe":
            pivot = samples[samples.feature.isin(rows.feature)].pivot_table(index="feature", columns="zone", values="value", aggfunc="median").reindex(index=best.feature, columns=rows.zone.unique())
            values = pivot.to_numpy(float)
        limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1
        limit = limit if limit > 1e-12 else 1
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#dddddd")
        im = heat.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        heat.set_yticks(range(len(best)), [textwrap.fill(v, 45) for v in best.label], fontsize=9)
        heat.set_xticks(range(len(pivot.columns)), [zone_label(c, j).split(" (")[0] for j in pivot.columns], rotation=35, ha="right", fontsize=9)
        for iy, feature in enumerate(best.feature):
            for ix, zone in enumerate(pivot.columns):
                q = rows[(rows.feature == feature) & (rows.zone == zone)].q.iloc[0]
                if np.isfinite(q) and q < c["alpha"]:
                    heat.text(ix, iy, "*", ha="center", va="center", fontsize=12,
                              color="white" if abs(values[iy, ix]) > .6 * limit else "black")
        e = c["endpoint"]
        meaning = (f"{e['group_b']} − {e['group_a']} (" + ("median paired difference)" if e["paired"] else "median difference)") if e["type"] == "binary" else
                   "Spearman correlation" if e["type"] == "continuous" else
                   "Log2 hazard ratio per SD" if e["type"] == "survival" else "Median across samples")
        heat.set_title("Effect across every scanned zone", loc="left", fontsize=11)
        cb = figure.colorbar(im, ax=heat, shrink=.75, pad=.02)
        cb.set_label(meaning, fontsize=9)
        selected = rows.iloc[0]
        part = samples[(samples.feature == selected.feature) & (samples.zone == selected.zone)].dropna(subset=["value"])
        if e["type"] == "binary" and e["paired"]:
            paired = part.groupby("_sample")._group.nunique()
            part = part[part._sample.isin(paired[paired == 2].index)]
        dots, curve = figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])
        rng = np.random.default_rng(47)
        if e["type"] in ("binary", "describe"):
            groups = [e["group_a"], e["group_b"]] if e["type"] == "binary" else ["All"]
            if e["type"] == "binary" and e["paired"]:
                paired = part.pivot(index="_sample", columns="_group", values="value").reindex(columns=groups).dropna()
                for _, row in paired.iterrows():
                    dots.plot([0, 1], row.to_numpy(), color="#aaaaaa", lw=.6, zorder=0)
            for i, group in enumerate(groups):
                values = part.loc[part._group == group, "value"].to_numpy()
                dots.scatter(i + rng.uniform(-.1, .1, len(values)), values, color=palette.get(group, colors[i % 2]), alpha=.8, s=28)
                if len(values):
                    dots.plot([i-.2, i+.2], [np.median(values)]*2, color="black", lw=2)
            dots.set_xticks(range(len(groups)), [textwrap.fill(g, 22) for g in groups])
        else:
            outcome = e["value_col"] if e["type"] == "continuous" else e["time_col"]
            dots.scatter(part[outcome], part.value, s=25, color=colors[0])
            dots.set_xlabel(outcome)
        dots.set_ylabel(measurement(c, family, selected.feature))
        dots.set_title(textwrap.fill(selected.label + " · " + zone_label(c, selected.zone), 50) + "\n" + statistics_text(selected, c), fontsize=10, pad=12)
        if e["type"] in ("binary", "describe"):
            dots.set_xticks(range(len(groups)), [textwrap.fill(g, 22) + f"\nn={int((part._group == g).sum())}" for g in groups])
        profiles = samples[samples.feature == selected.feature]
        group_order = [e["group_a"], e["group_b"]] if e["type"] == "binary" else ["All"]
        if e["type"] == "binary" and e["paired"]:
            paired = profiles.dropna(subset=["value"]).groupby(["zone", "_sample"])._group.nunique()
            valid_keys = paired[paired == 2].reset_index()[["zone", "_sample"]]
            profiles = profiles.merge(valid_keys, on=["zone", "_sample"])
        for i, name in enumerate(group_order):
            g = profiles[profiles._group == name]
            summary = g.groupby("zone").value.agg(median="median", count="count", q25=lambda x:x.quantile(.25), q75=lambda x:x.quantile(.75))
            x = summary.index.to_numpy()
            curve.plot(x, summary["median"], "o-", label=name, color=palette.get(name, colors[i % 2]), lw=1.8)
            curve.fill_between(x, summary.q25.to_numpy(float), summary.q75.to_numpy(float), color=palette.get(name, colors[i % 2]), alpha=.14)
        curve.set_xticks(range(len(pivot.columns)), [zone_label(c,j).split(" (")[0] for j in pivot.columns], rotation=35, ha="right", fontsize=8)
        curve.set_ylabel(measurement(c, family, selected.feature))
        curve.set_title("Across zones: median and middle 50% of samples", fontsize=10)
        curve.legend(frameon=False, fontsize=9)
        for ax in (heat, dots, curve):
            ax.spines[["top", "right"]].set_visible(False)
        title = FAMILIES[family] + (" · " + DIVERSITY[metric] if metric else "")
        comparison = "\n" + textwrap.fill(c["_comparison_label"], 100) if c.get("_comparison_label") else ""
        figure.suptitle(title + " · spatial feature screening" + comparison, fontsize=16, fontweight="bold")
        qualification = ("Area is estimated from cell positions; not measured tissue area.\n" if family == "density" and "estimated" in measurement(c, family) else "")
        qualification += ("Random labels test spatial co-occurrence; p and q here compare biological samples.\n" if family in ("dyad", "triad") and c["options"][family]["metric"] == "zscore" else "")
        scope = c.get("correction_scope", "global")
        total = c.get("_scan_hypotheses", len(results))
        correction = f"across all {total:,} planned tests" if scope == "global" else SCOPES[scope] + f" · {total:,} tests in the scan"
        if scope == "classic":
            qualification += "Classic q does not adjust for searching across distances or additional comparisons.\n"
        if c.get("resampling", "off") != "off":
            qualification += "Recovery = significant again with the same direction; not independent validation.\n"
        figure.supxlabel(qualification + f"Each dot is one {c['sample_col']} · * adjusted q < {c['alpha']:g} · {correction}\n"
                            "Grey = unmeasured · Shading is sample spread, not a confidence interval.\n"
                            "The displayed feature was selected from this scan; overlapping zones are not independent confirmations.", fontsize=9)
        for ext in ("png", "svg", "pdf"):
            name = stem + "." + ext
            figure.savefig(folder / name, dpi=200, bbox_inches="tight")
        plt.close(figure)
        files.append({"family": family, "name": title, "file": stem + ".png", **({"metric": metric} if metric else {})})
    return files
