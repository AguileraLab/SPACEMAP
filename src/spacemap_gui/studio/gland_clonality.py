"""Gland composition types and proximity-bed niches, shared by Studio and classic.

An anatomical region is identified by image, tissue piece, compartment and local
ID. No coordinate neighbour search, genotype claim, or positivity gate is made.
Only well-sampled regions fit the models. Smaller nonempty regions are retained
as smoothed predictions; absent beds are missing, never a fabricated niche.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
K_HELP = {
    "interior_k": "How many gland composition types to make. For example, K=6 groups glands into six types based on their interior cell subtypes. These are composition types, not confirmed genetic clones.",
    "exterior_k": "How many surrounding niche types to make. For example, K=4 groups exterior beds into four types based on their immune and stromal cell mix.",
    "review": "Tries K=2–8 separately for glands and niches, regardless of the manual K settings. Higher separation scores mean more distinct groups. Review does not change settings or cell labels; use a suggested K, then Fit to apply it.",
}
DEFAULTS = dict(image_col="batch_id", piece_col="", cell_id_col="cell_id",
                label_col="label", x_col="x_centroid", y_col="y_centroid",
                interior_col="phenotype_sub", exterior_col="phenotype_ml_mr_refined",
                interior_classes=[], exterior_classes=[], regions=[], scope=[],
                parent_col="", parent_values=[], min_gland=20, min_bed=10,
                prior=10., standardize=True, z_clip=3., interior_k=6, exterior_k=4,
                auto_k=False, auto_bed_k=False, seed=42,
                output_col="gland_clone", niche_col="gland_bed_niche",
                sample_col="", visit_col="", group_col="", stratum_col="",
                baseline="", followup="", group_a="", group_b="",
                min_samples=3)


def defaults(columns):
    """Suggestions only; every role and label remains editable."""
    c = json.loads(json.dumps(DEFAULTS))
    for key in (k for k in c if k.endswith("_col") and k not in ("output_col", "niche_col")):
        if c[key] not in columns:
            c[key] = ""
    for key, candidates in {"piece_col": ["tissue_roi"], "sample_col": ["patient", "sample_id"],
                             "visit_col": ["timepoint", "visit"], "group_col": ["response", "group"],
                             "stratum_col": ["arm"], "parent_col": ["phenotype_ml_mr_refined"]}.items():
        c[key] = next((n for n in candidates if n in columns), "")
    # A parent restriction is optional and must be explicitly selected.
    c["parent_col"] = ""
    for prefix, title in (("tumor", "Tumor"), ("epithelia", "Normal epithelium")):
        if prefix + "_roi" in columns:
            c["regions"].append(dict(name=title, roi_col=prefix + "_roi",
                bed_id_col=prefix + "_bed_id" if prefix + "_bed_id" in columns else "",
                bed_col=prefix + "_bed" if prefix + "_bed" in columns else "", bed_value="1"))
    c["scope"] = [r["name"] for r in c["regions"]]
    return c


def check(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError("Unknown gland-type setting.")
    c = {**json.loads(json.dumps(DEFAULTS)), **values}
    for key in (k for k in DEFAULTS if k.endswith("_col")):
        if not isinstance(c[key], str) or len(c[key]) > 256:
            raise ValueError("Choose a column for " + key)
    for key in ("standardize", "auto_k", "auto_bed_k"):
        if not isinstance(c[key], bool):
            raise ValueError("Choose on or off for " + key)
    for key, low, high in (("min_gland", 1, 100000), ("min_bed", 1, 100000),
                           ("interior_k", 1, 30), ("exterior_k", 1, 30),
                           ("min_samples", 2, 10000), ("seed", 0, 2**31 - 1)):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or not low <= c[key] <= high:
            raise ValueError(f"{key}: enter a whole number from {low} to {high}.")
    for key, low, high in (("prior", 0., 10000.), ("z_clip", .1, 100.)):
        try:
            c[key] = float(c[key])
        except (TypeError, ValueError):
            raise ValueError("Enter a number for " + key) from None
        if not np.isfinite(c[key]) or not low <= c[key] <= high:
            raise ValueError(f"{key}: enter a finite value from {low} to {high}.")
    for key in ("interior_classes", "exterior_classes", "scope", "parent_values"):
        if not isinstance(c[key], list) or any(not isinstance(v, str) or not v.strip() for v in c[key]) or len(c[key]) > 500:
            raise ValueError("Choose values for " + key)
        c[key] = list(dict.fromkeys(c[key]))
    for key in ("baseline", "followup", "group_a", "group_b"):
        if not isinstance(c[key], str) or len(c[key]) > 256:
            raise ValueError("Choose a value for " + key)
    if not isinstance(c["regions"], list) or not 1 <= len(c["regions"]) <= 20:
        raise ValueError("Add at least one gland annotation mapping.")
    names = []
    for r in c["regions"]:
        if not isinstance(r, dict) or set(r) - {"name", "roi_col", "bed_id_col", "bed_col", "bed_value"}:
            raise ValueError("Choose gland ID, bed ID and optional bed flag columns.")
        for key in ("name", "roi_col", "bed_id_col", "bed_col", "bed_value"):
            if not isinstance(r.get(key, ""), str):
                raise ValueError("Annotation roles must be column names.")
        if not r.get("name", "").strip() or not r.get("roi_col"):
            raise ValueError("Give each compartment a name and a gland ID column.")
        names.append(r["name"])
    if len(set(names)) != len(names) or not c["scope"] or set(c["scope"]) - set(names):
        raise ValueError("Choose distinct compartment names and a fitting scope.")
    out = [c["output_col"], c["niche_col"]]
    if len(set(out)) != 2 or any(not NAME.fullmatch(n) for n in out):
        raise ValueError("Give the two outputs distinct names using letters, numbers and underscores.")
    if set(out) & set(required(c)):
        raise ValueError("Output names must not replace an input column.")
    if c["parent_col"] and not c["parent_values"]:
        raise ValueError("Choose the parent cell types, or set Parent cells to All.")
    return c


def required(c):
    names = [c[k] for k in ("image_col", "piece_col", "cell_id_col", "label_col", "x_col", "y_col",
             "interior_col", "exterior_col", "parent_col", "sample_col", "visit_col", "group_col", "stratum_col")]
    names += [r.get(k, "") for r in c["regions"] for k in ("roi_col", "bed_id_col", "bed_col")]
    return list(dict.fromkeys(n for n in names if n))


def _strings(s):
    """Canonicalise numeric category IDs (1 and 1.0), retaining real text labels."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("Int8").astype("string")
    v = s.astype("string").str.strip()
    return v.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _ids(s):
    v = _strings(s)
    number = pd.to_numeric(v, errors="coerce")
    good = v.notna() & ~v.str.lower().isin(["", "nan", "none", "null", "<na>"])
    good &= number.isna() | number.gt(0)
    return v.where(good)


@dataclass
class Analysis:
    config: dict
    structures: pd.DataFrame
    assignments: pd.DataFrame
    models: dict
    profiles: dict
    notes: list


def index_structures(frame, c):
    """Return region metadata, count matrices and row→structure maps in O(cells)."""
    frame = frame.reset_index(drop=True)
    missing = set(required(c)) - set(frame)
    if missing:
        raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
    if not all(c[k] for k in ("image_col", "cell_id_col", "x_col", "y_col", "interior_col")):
        raise ValueError("Choose image, cell ID, X, Y and interior subtype columns.")
    # Image names must exactly match the shared image/table assignment; unlike
    # local numeric gland IDs, a name such as "1.0" must not become "1".
    image = frame[c["image_col"]].astype("string").str.strip()
    piece = _ids(frame[c["piece_col"]]) if c["piece_col"] else pd.Series("whole", index=frame.index)
    valid = image.notna() & image.ne("") & piece.notna()
    inside = [_ids(frame[r["roi_col"]]) for r in c["regions"]]
    overlap = sum(s.notna().astype(int) for s in inside) > 1
    if (overlap & valid).any():
        raise ValueError(f"{int((overlap & valid).sum()):,} cells belong to several mapped gland compartments. Resolve the overlapping annotation IDs first.")
    any_inside = pd.concat(inside, axis=1).notna().any(axis=1)
    types = _strings(frame[c["interior_col"]])
    exterior = _strings(frame[c["exterior_col"]]) if c["exterior_col"] else pd.Series(pd.NA, index=frame.index, dtype="string")
    parent = (_strings(frame[c["parent_col"]]).isin(c["parent_values"]) if c["parent_col"] else pd.Series(True, index=frame.index))
    int_labels = c["interior_classes"]
    ext_labels = c["exterior_classes"]
    if not int_labels:
        raise ValueError("Select the interior subtypes to count.")
    structures, int_counts, ext_counts = [], [], []
    interior_map = np.full(len(frame), -1, dtype=np.int64)
    exterior_map = np.full(len(frame), -1, dtype=np.int64)
    notes = []
    for r, ids in zip(c["regions"], inside):
        if r["name"] not in c["scope"]:
            continue
        mask = valid & ids.notna()
        keys = pd.MultiIndex.from_arrays([image[mask], piece[mask], ids[mask]], names=["image", "piece", "local_id"])
        codes, distinct = pd.factorize(keys, sort=True)
        if not len(distinct):
            continue
        offset = sum(len(s) for s in structures)
        interior_map[np.flatnonzero(mask)] = codes + offset
        meta = distinct.to_frame(index=False)
        meta.columns = ["image", "piece", "local_id"]
        meta["compartment"] = r["name"]
        meta["structure"] = [hashlib.sha256(json.dumps([*k, r["name"]], ensure_ascii=False).encode()).hexdigest()[:24] for k in distinct]
        here = frame.loc[mask].copy()
        here["__structure"] = codes
        for key, role in (("sample", "sample_col"), ("visit", "visit_col"), ("group", "group_col"), ("stratum", "stratum_col")):
            if c[role]:
                vals = _strings(here[c[role]])
                groups = vals.groupby(codes)
                if groups.nunique(dropna=False).gt(1).any():
                    raise ValueError(f"{c[role]} changes within a gland. Check gland IDs, image and tissue-piece columns.")
                meta[key] = groups.first().reindex(range(len(meta))).to_numpy()
            else:
                meta[key] = ""
        meta["interior_total"] = np.bincount(codes, minlength=len(meta))
        xy = here[[c["x_col"], c["y_col"]]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col, name in zip(xy, ("x", "y")):
            grouped = xy[col].groupby(codes)
            meta[name] = grouped.mean().reindex(range(len(meta))).to_numpy()
            meta[name+"0"] = grouped.min().reindex(range(len(meta))).to_numpy()
            meta[name+"1"] = grouped.max().reindex(range(len(meta))).to_numpy()
        def counts(mapping, values, labels, eligibility):
            result = np.zeros((len(meta), len(labels)), dtype=np.int64)
            good = (mapping >= offset) & (mapping < offset + len(meta)) & eligibility
            lab = pd.Categorical(values[good], categories=labels).codes
            rows = mapping[good] - offset
            keep = lab >= 0
            np.add.at(result, (rows[keep], lab[keep]), 1)
            return result
        ic = counts(interior_map, types, int_labels, parent.to_numpy())
        if r.get("bed_id_col") and ext_labels and c["exterior_col"]:
            bed = _ids(frame[r["bed_id_col"]])
            bedmask = valid & ~any_inside & bed.notna()
            if r.get("bed_col"):
                bedmask &= _strings(frame[r["bed_col"]]).eq(r.get("bed_value", "1")).fillna(False)
            lookup = pd.MultiIndex.from_arrays([image[bedmask], piece[bedmask], bed[bedmask]])
            match = distinct.get_indexer(lookup)
            rows = np.flatnonzero(bedmask)
            matched = match >= 0
            if np.any(exterior_map[rows[matched]] >= 0):
                raise ValueError("Exterior cells have overlapping bed assignments. Choose disjoint proximity-bed mappings.")
            exterior_map[rows[matched]] = match[matched] + offset
            if (~matched).sum():
                notes.append(f"{r['name']}: {int((~matched).sum()):,} exterior cells have no matching gland in the same image and tissue piece; left unassigned.")
        ec = counts(exterior_map, exterior, ext_labels, np.ones(len(frame), dtype=bool))
        meta["n_interior"] = ic.sum(axis=1)
        meta["n_exterior"] = ec.sum(axis=1)
        structures.append(meta)
        int_counts.append(ic)
        ext_counts.append(ec)
    if not structures:
        raise ValueError("No glands have valid IDs in the selected images and tissue pieces.")
    if (~valid).sum():
        notes.append(f"{int((~valid).sum()):,} cells lack a valid image or tissue-piece ID and are excluded.")
    return pd.concat(structures, ignore_index=True), np.vstack(int_counts), np.vstack(ext_counts), interior_map, exterior_map, notes


def fit_composition(counts, labels, c, kind, report=lambda *a: None):
    """Count prior, size-weighted transform, weighted fit and reproducible K review."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits
    n = counts.sum(axis=1)
    fit = n >= c["min_gland" if kind == "interior" else "min_bed"]
    assigned = np.full(len(n), None, dtype=object)
    status = np.where(n > 0, "Small: review", "No selected cells").astype(object)
    status[fit] = "Fit"
    if not fit.any() or not n.sum() or not len(labels):
        return assigned, np.where(n > 0, "Too few to fit", status), None, None
    prior = counts.sum(axis=0) / n.sum()
    smooth = np.divide(counts + c["prior"] * prior, (n + c["prior"])[:, None],
                       out=np.zeros_like(counts, dtype=float), where=(n+c["prior"])[:, None] > 0)
    weights = np.log1p(n[fit])
    mu = np.average(smooth[fit], axis=0, weights=weights)
    sd = np.sqrt(np.average((smooth[fit]-mu)**2, axis=0, weights=weights))
    # A prior/n term can vary with structure size even when an observed subtype
    # is absent or constant in every fit gland. Do not standardize that artefact.
    observed_variable = np.ptp(counts[fit] / n[fit, None], axis=0) > 1e-8
    active = (sd > 1e-8) & observed_variable
    scale = np.where(active, sd, 1.)
    z = np.clip((smooth-mu)/scale, -c["z_clip"], c["z_clip"])
    z[:, ~active] = 0
    x = z if c["standardize"] else smooth
    x[:, ~active] = 0
    unique = len(np.unique(x[fit], axis=0))
    fixed = c["interior_k" if kind == "interior" else "exterior_k"]
    auto = c["auto_k" if kind == "interior" else "auto_bed_k"]
    if not auto and fixed > min(unique, int(fit.sum())):
        raise ValueError(f"{kind.title()}: K={fixed} needs {fixed} distinct well-sampled structures; found {min(unique, int(fit.sum()))}. Reduce K or use Auto.")
    candidates = list(range(2, min(8, unique, int(fit.sum())-1)+1)) if auto else [fixed]
    if not candidates:
        candidates = [1]
    diagnostics, best, score_best = [], None, -np.inf
    rng = np.random.default_rng(c["seed"])
    fit_x = x[fit]
    sample = rng.choice(len(fit_x), min(2000, len(fit_x)), replace=False)
    with threadpool_limits(limits=4):
        for k in candidates:
            report(f"Fitting {kind} types: K={k}…")
            model = KMeans(n_clusters=k, n_init=20, random_state=c["seed"]).fit(fit_x, sample_weight=weights)
            y = model.labels_[sample]
            score = float(silhouette_score(fit_x[sample], y)) if 1 < len(np.unique(y)) < len(y) else None
            diagnostics.append(dict(k=k, silhouette=score, inertia=float(model.inertia_)))
            rank = score if score is not None else -2.
            if best is None or rank > score_best:
                best, score_best = model, rank
        prediction = best.predict(x[n > 0])
    raw = np.divide(counts, n[:, None], out=np.zeros_like(counts, dtype=float), where=n[:, None] > 0)
    raw_means, z_means, titles = [], [], []
    # Cluster IDs are ordered by enrichment signature for reproducible readable names.
    original = sorted(range(best.n_clusters), key=lambda j: (int(np.argmax(best.cluster_centers_[j])), tuple(best.cluster_centers_[j])))
    order = {old: new for new, old in enumerate(original)}
    for old in original:
        member = fit.copy()
        member[fit] = best.labels_ == old
        w = np.log1p(n[member])
        rp = np.average(raw[member], axis=0, weights=w)
        zp = np.average(z[member], axis=0, weights=w)
        raw_means.append(rp.tolist()); z_means.append(zp.tolist())
        peak = int(np.argmax(zp)) if c["standardize"] else int(np.argmax(rp))
        top = np.argsort(-(zp if c["standardize"] else rp))[:min(2, len(labels))]
        signature = f"{labels[peak]} ({rp[peak]:.0%})" if kind == "interior" else " + ".join(labels[j] for j in top)
        titles.append(f"{'Gland' if kind == 'interior' else 'Niche'} {order[old]+1} · {signature}"[:128])
    assigned[n > 0] = [titles[order[int(j)]] for j in prediction]
    model = dict(features=labels, prior=prior.tolist(), prior_strength=c["prior"], mean=mu.tolist(), scale=scale.tolist(),
                 active=active.tolist(), standardized=c["standardize"], clip=c["z_clip"], centers=best.cluster_centers_.tolist(),
                 names={str(old): titles[order[old]] for old in original}, k=best.n_clusters,
                 silhouette=None if score_best == -2 else score_best, silhouette_n=len(sample),
                 diagnostics=diagnostics, fitted=int(fit.sum()), predicted_small=int(((n > 0) & ~fit).sum()))
    from importlib.metadata import version
    model["method"] = dict(schema=1, algorithm="KMeans", n_init=20, seed=c["seed"],
                           sample_weight="log1p(selected-cell count)", sklearn_version=version("scikit-learn"))
    profile = dict(rows=titles, columns=labels, raw=raw_means, z=z_means,
                   note="Colour: relative Z; text: log-size-weighted mean measured proportion in fit structures. Z is not marker positivity.")
    return assigned, status, model, profile


def analyse(frame, values, report=lambda *a: None):
    c = check(values)
    report("Indexing glands and their exterior beds…", .12)
    structures, ic, ec, imap, emap, notes = index_structures(frame, c)
    models, profiles = {}, {}
    for kind, counts, labels, output in (("interior", ic, c["interior_classes"], "gland_type"),
                                        ("exterior", ec, c["exterior_classes"], "niche")):
        report(f"Smoothing {kind} compositions…", .35 if kind == "interior" else .55)
        assigned, status, models[kind], profiles[kind] = fit_composition(counts, labels, c, kind, report)
        structures[output] = assigned
        structures[kind+"_status"] = status
        structures[kind+"_distance"] = np.nan
        model = models[kind]
        if model:
            n = counts.sum(axis=1)
            good = n > 0
            transformed = (counts[good] + c["prior"]*np.array(model["prior"])) / (n[good]+c["prior"])[:, None]
            if model["standardized"]:
                transformed = np.clip((transformed-np.array(model["mean"]))/np.array(model["scale"]), -model["clip"], model["clip"])
            transformed[:, ~np.array(model["active"])] = 0
            distance = ((transformed[:, None, :] - np.array(model["centers"])[None, :, :])**2).sum(axis=2).min(axis=1)**.5
            structures.loc[good, kind+"_distance"] = distance
        for j, label in enumerate(labels):
            structures[kind+"_count:"+label] = counts[:, j]
        if models[kind] is None:
            notes.append(f"{kind.title()}: no well-sampled nonempty structures to fit; no type was assigned.")
    assignments = pd.DataFrame(index=np.arange(len(frame)))
    for column, mapping, key in ((c["output_col"], imap, "gland_type"), (c["niche_col"], emap, "niche")):
        values_out = np.full(len(frame), None, dtype=object)
        good = mapping >= 0
        values_out[good] = structures[key].to_numpy()[mapping[good]]
        assignments[column] = values_out
    assignments["__gland"] = imap
    assignments["__bed"] = emap
    return Analysis(c, structures, assignments, models, profiles, notes)


def clinical(structures, c, *, compartment="", strata=False):
    """Independent-sample burdens; complete-pair deltas; one declared BH family.

    Each region contributes log1p(eligible cells). Empty compartments do not
    become zeros. A type absent in an otherwise measured sample does become 0.
    Visits remain distinct; pooled values average visits within each person.
    """
    from scipy.stats import mannwhitneyu
    if not c.get("sample_col") or not c.get("group_col"):
        return dict(values=[], tests=[], notes=["Choose independent sample and comparison columns to compare groups."])
    s = structures[structures.compartment.eq(compartment)].copy() if compartment else structures.copy()
    s = s[s["sample"].notna() & s["group"].notna() & s["sample"].ne("") & s["group"].ne("")]
    if c.get("visit_col"):
        s = s[s.visit.notna() & s.visit.ne("")]
    else:
        s["visit"] = "All"
    # A between-person comparison needs consistent group and arm per person.
    for column, source in (("group", c["group_col"]), ("stratum", c.get("stratum_col"))):
        if source and s.groupby("sample")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"{source} changes within an independent sample. This comparison assumes fixed groups; choose an appropriate sample/group column.")
    a, b = c["group_a"], c["group_b"]
    if not a or not b or a == b:
        return dict(values=[], tests=[], notes=["Choose two different comparison groups."])
    rows = []
    for kind, type_col, n_col in (("interior", "gland_type", "n_interior"), ("exterior", "niche", "n_exterior")):
        work = s[s[type_col].notna() & s[n_col].gt(0)].copy()
        if work.empty:
            continue
        work["weight"] = np.log1p(work[n_col])
        types = sorted(work[type_col].unique())
        for (sample, visit), g in work.groupby(["sample", "visit"], sort=True, observed=True):
            burden = g.groupby(type_col).weight.sum() / g.weight.sum() * 100
            for name in types:
                rows.append(dict(sample=sample, visit=visit, group=g.group.iloc[0], stratum=None if pd.isna(g.stratum.iloc[0]) else g.stratum.iloc[0],
                                 kind=kind, feature=name, value=float(burden.get(name, 0))))
    values = pd.DataFrame(rows)
    if values.empty:
        return dict(values=[], tests=[], notes=["No samples have eligible assigned structures and comparison metadata."])
    pools = [("All", values)]
    if strata and c.get("stratum_col"):
        pools += [(str(v), g) for v, g in values.dropna(subset=["stratum"]).groupby("stratum", observed=True) if str(v)]
    tests, display = [], []
    for stratum, pool in pools:
        preferred = [c["baseline"], c["followup"]]
        views = sorted([(str(visit), g) for visit, g in pool.groupby("visit", observed=True)],
                       key=lambda item: (preferred.index(item[0]) if item[0] in preferred else 2, item[0]))
        if pool.visit.nunique() > 1:
            pooled = pool.groupby(["sample", "group", "kind", "feature"], as_index=False).value.mean()
            views.append(("Pooled visits", pooled))
        if c["baseline"] and c["followup"] and c["baseline"] != c["followup"]:
            paired = pool.pivot(index=["sample", "group", "kind", "feature"], columns="visit", values="value")
            if {c["baseline"], c["followup"]} <= set(paired.columns):
                paired = paired.dropna(subset=[c["baseline"], c["followup"]])
                delta = (paired[c["followup"]] - paired[c["baseline"]]).rename("value").reset_index()
                views.append((f"Δ {c['followup']} − {c['baseline']}", delta))
        for view, g in views:
            d = g[g.group.isin([a, b])].copy()
            d["view"], d["stratum"] = view, stratum
            display.extend(d.to_dict("records"))
            for (kind, feature), f in d.groupby(["kind", "feature"], observed=True):
                av = f.loc[f.group.eq(a), "value"].to_numpy()
                bv = f.loc[f.group.eq(b), "value"].to_numpy()
                enough = min(len(av), len(bv)) >= c["min_samples"]
                med = lambda v: float(np.median(v)) if len(v) else None
                mean = lambda v: float(np.mean(v)) if len(v) else None
                tests.append(dict(kind=kind, feature=feature, view=view, stratum=stratum,
                    group_a=a, group_b=b, n_a=len(av), n_b=len(bv), median_a=med(av), median_b=med(bv),
                    mean_a=mean(av), mean_b=mean(bv), effect=med(bv)-med(av) if len(av) and len(bv) else None,
                    p=float(mannwhitneyu(av, bv, alternative="two-sided").pvalue) if enough else None,
                    state="Tested" if enough else "Too few independent samples"))
    # Include planned but untestable contrasts as p=1 in the correction family.
    ps = np.array([r["p"] if r["p"] is not None else 1. for r in tests])
    order = np.argsort(ps)
    qs = np.empty(len(ps))
    if len(ps):
        qs[order] = np.minimum(1., np.minimum.accumulate((ps[order]*len(ps)/np.arange(1, len(ps)+1))[::-1])[::-1])
    for row, q in zip(tests, qs):
        row["q"] = float(q) if row["p"] is not None else None
        row["higher"] = b if (row["effect"] or 0) > 0 else a if (row["effect"] or 0) < 0 else "Equal"
    return dict(values=display, tests=tests, notes=[f"Burden weights: log1p(selected cells) per structure. One value per independent sample and visit; complete pairs only for changes. FDR: all {len(tests)} planned types, views and strata in this comparison. These are exploratory comparisons, not independent validation."],
                trajectories=rows)


def settings(root, table, changes=None, expected=None):
    from spacemap import dataio
    from spacemap_gui import persistence
    from .settings import read_config, revision, project_lock
    key = "tab8_gland_types_" + hashlib.sha256(str(Path(table).resolve()).encode()).hexdigest()[:16]
    with project_lock(root):
        config = read_config(root)
        saved = (config.get("tab_configurations", {}).get("tab8_analysis") or {}).get(key)
        current = saved or defaults(dataio.columns(table))
        if changes is not None:
            if expected != revision(current):
                raise ValueError("Gland settings changed in another window. Reload them before saving.")
            current = check(changes)
            persistence.update_stored_config_key(config, key, current)
            persistence.save_project_config(str(root), config)
    return dict(values=current, revision=revision(current), columns=dataio.columns(table))


def options(table, column):
    from spacemap import dataio
    if not column or column not in dataio.columns(table):
        raise ValueError("Choose an existing column.")
    s = _strings(dataio.load(table, columns=[column])[column]).dropna()
    counts = s[s.ne("")].value_counts()
    if len(counts) > 500:
        raise ValueError("This column has more than 500 values. Choose a categorical column.")
    return dict(values=sorted(counts.index), counts={str(k): int(v) for k, v in counts.items()})


def prepare(root, table, values, images, overwrite=False, diagnose=False):
    from spacemap import table_collection
    from . import tasks
    c = check(values)
    if not isinstance(overwrite, bool) or not isinstance(diagnose, bool):
        raise ValueError("Choose whether to overwrite or fit K only.")
    if not images:
        raise ValueError("Tick at least one image.")
    if table_collection.is_collection(table) and (c["image_col"] != "batch_id" or c["cell_id_col"] != "cell_id"):
        raise ValueError("For separate tables use the shared batch_id and cell_id roles; source IDs are kept automatically.")
    if diagnose:
        c["auto_k"] = c["auto_bed_k"] = True
    names = [c["output_col"], c["niche_col"]]
    out = dict(root=str(root), table=str(table), result="gland_types", values=c, images=images,
               image_col=c["image_col"], selected_images=images, cell_id_col=c["cell_id_col"], markers=required(c),
               output_col=names[0], auxiliary_columns=names[1:], overwrite_columns=names if overwrite else [],
               diagnose=diagnose)
    # Diagnose has no column publication, and can inspect K even if outputs exist.
    tasks.validate_table({**out, "overwrite_columns": names} if diagnose else out)
    out["revision"] = tasks.source_revision(out)
    return out


def _json(path, value):
    from spacemap_workflows.leiden_job import write_json
    write_json(Path(path), value)


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def export_analysis(result, folder, palette=None):
    """Small region/model exports; never rewrite the active measurements."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result.structures.to_parquet(folder / "gland_structures.parquet", index=False)
    result.structures.to_csv(folder / "gland_structures.csv", index=False)
    _json(folder / "gland_models.json", result.models)
    _json(folder / "gland_settings.json", result.config)
    _json(folder / "gland_profiles.json", result.profiles)
    for kind, profile in result.profiles.items():
        if profile:
            for measure in ("raw", "z"):
                pd.DataFrame(profile[measure], index=profile["rows"], columns=profile["columns"]).to_csv(folder / f"{kind}_{measure}_profiles.csv")
            (folder / f"{kind}_profiles.png").write_bytes(profile_figure(profile))
        model = result.models.get(kind)
        if model and len(model.get("diagnostics", [])) > 1:
            pd.DataFrame(model["diagnostics"]).to_csv(folder / f"{kind}_k_review.csv", index=False)
            (folder / f"{kind}_k_review.png").write_bytes(k_review_figure(model, kind))
        distribution = silhouette_details(result.structures, model, kind)
        _json(folder / f"{kind}_silhouette.json", distribution)
        (folder / f"{kind}_silhouette.csv").write_text(silhouette_csv(distribution))
        (folder / f"{kind}_silhouette.png").write_bytes(silhouette_figure(distribution, palette))
    comparisons = None
    try:
        comparisons = clinical(result.structures, result.config, strata=True)
        _json(folder / "clinical.json", comparisons)
        pd.DataFrame(comparisons["tests"]).to_csv(folder / "clinical_tests.csv", index=False)
        pd.DataFrame(comparisons["values"]).to_csv(folder / "sample_burdens.csv", index=False)
    except ValueError as exc:
        result.notes.append("Clinical comparison: " + str(exc))
    return comparisons


def run(config, folder, report):
    from spacemap import dataio, table_collection, phenotype_presets
    from . import tasks
    c = config["values"]
    report("Reading selected cell columns…", .03)
    if table_collection.is_collection(config["table"]):
        frame = table_collection.load(config["table"], columns=required(c), images=config["images"])
    else:
        frame = dataio.load(config["table"], columns=required(c))
    image_names = frame[c["image_col"]].astype("string").str.strip()
    known = set(image_names.dropna())
    if set(config["images"]) - known:
        raise ValueError("Some selected image IDs do not match the chosen Image column.")
    frame = frame[image_names.isin(config["images"])].reset_index(drop=True)
    result = analyse(frame, c, report)
    if result.models["interior"] is None:
        raise ValueError("No glands meet the fitting minimum. Lower it or select more images/subtypes; nothing was saved.")
    names = [name for p in result.profiles.values() if p for name in p["rows"]]
    palette = phenotype_presets.ensure(names)
    report("Saving profiles and sample-level comparisons…", .8)
    export_analysis(result, folder, palette)
    # Retain a compact row map for exact image/Data links, including unselected
    # cell types inside a gland. Coordinates are for display; never re-match cells.
    link = pd.DataFrame({"cell_id": frame[c["cell_id_col"]].astype(str), "image": frame[c["image_col"]].astype("string").str.strip(),
                         "label": frame[c["label_col"]] if c["label_col"] else np.nan,
                         "x": pd.to_numeric(frame[c["x_col"]], errors="coerce"),
                         "y": pd.to_numeric(frame[c["y_col"]], errors="coerce"),
                         "gland": result.assignments["__gland"], "bed": result.assignments["__bed"]})
    link.to_parquet(Path(folder) / "gland_cell_links.parquet", index=False)
    outputs = {name: result.assignments[name] for name in (c["output_col"], c["niche_col"])}
    if config.get("diagnose"):
        if tasks.source_revision(config) != config["revision"]:
            raise ValueError("The source changed during K review. Repeat with the current data.")
        saved = {"table": config["table"], "columns": [], "cells": len(frame)}
    else:
        saved = tasks.save_result(config, outputs, frame[c["cell_id_col"]], report)
    summary = dict(**saved, folder=str(folder), images=config["images"], values=c, profiles=result.profiles,
                   models={kind: ({k: m[k] for k in ("k", "silhouette", "silhouette_n", "diagnostics", "fitted", "predicted_small")} if m else None) for kind, m in result.models.items()},
                   glands=len(result.structures), beds=int(result.structures.n_exterior.gt(0).sum()),
                   counts={str(k): int(v) for k, v in result.structures[result.structures.gland_type.notna()].groupby("image").size().items()},
                   palette=palette, notes=result.notes, diagnose=config.get("diagnose", False),
                   source_after=tasks.source_revision(config))
    _json(Path(folder) / "gland_result.json", summary)
    return summary


def saved_result(root, table, job_id):
    from . import jobs
    folder = jobs.directory(root, job_id)
    status = json.loads((folder / "status.json").read_text())
    if status.get("kind") != "gland_types" or status.get("status") != "done":
        raise ValueError("Choose a completed Gland types run.")
    result = json.loads((folder / "gland_result.json").read_text())
    if Path(result["table"]).resolve() != Path(table).resolve():
        raise ValueError("This run belongs to another cell table. Select its source table first.")
    # Derive from saved scores, including runs made before the chart existed.
    result["k_reviews"] = {kind: k_review(result["models"].get(kind)) for kind in ("interior", "exterior")}
    return result, folder


def runs(root, table):
    from . import jobs
    found = []
    for record in jobs.history(root, limit=200):
        r = record.get("result") or {}
        if record.get("kind") == "gland_types" and r.get("table") == str(table) and record.get("status") == "done":
            found.append(dict(id=record["id"], label=("K review · " if r.get("diagnose") else "") + record.get("started", record["id"]),
                              glands=r.get("glands"), beds=r.get("beds")))
    return found


def latest(root, table):
    """A small image-list summary, requested only after sources are displayed."""
    from . import jobs
    for record in jobs.history(root, limit=200):
        result = record.get("result") or {}
        if (record.get("kind") == "gland_types" and record.get("status") == "done"
                and result.get("table") == str(table) and not result.get("diagnose")):
            return {"counts": result.get("counts", {}), "job_id": record["id"]}
    return None


def structures(root, table, job_id, image=""):
    result, folder = saved_result(root, table, job_id)
    s = pd.read_parquet(folder / "gland_structures.parquet")
    if image:
        s = s[s.image.eq(image)]
    columns = [n for n in s if not n.startswith(("interior_count:", "exterior_count:"))]
    return dict(rows=records(s[columns]), images=result["images"])


def maps(root, table, job_id, image):
    result, folder = saved_result(root, table, job_id)
    if image not in result["images"]:
        raise ValueError("Choose an image included in this run.")
    link = pd.read_parquet(folder / "gland_cell_links.parquet", filters=[("image", "=", image)])
    count = len(link)
    # Display sampling alone; analysis and linked selections use every cell.
    link = link.sample(min(len(link), 25000), random_state=0).sort_index()
    link = link[np.isfinite(link.x) & np.isfinite(link.y)]
    s = pd.read_parquet(folder / "gland_structures.parquet")
    for role, col in (("gland", "gland_type"), ("bed", "niche")):
        good = link[role].ge(0)
        link[role+"_name"] = None
        link.loc[good, role+"_name"] = s.loc[link.loc[good, role], col].to_numpy()
        link[role+"_id"] = None
        link.loc[good, role+"_id"] = s.loc[link.loc[good, role], "structure"].to_numpy()
    return dict(points=records(link.drop(columns=["cell_id", "label"])), total=count,
                palette=result["palette"], image=image)


def select_cells(root, table, job_id, structure, role, channel):
    from . import tasks
    result, folder = saved_result(root, table, job_id)
    if result["source_after"] != tasks.source_revision({"table": str(table)}):
        raise ValueError("Cell data changed since this run. Rerun Gland types before linking its cells.")
    if role not in ("gland", "bed"):
        raise ValueError("Choose interior or exterior cells.")
    s = pd.read_parquet(folder / "gland_structures.parquet")
    selected = s.index[s.structure.eq(structure)]
    if len(selected) != 1:
        raise ValueError("Choose a structure from this result.")
    idx = selected[0]
    image = str(s.loc[idx, "image"])
    cells = pd.read_parquet(folder / "gland_cell_links.parquet", filters=[("image", "=", image), (role, "=", int(idx))])
    name = s.loc[idx, "gland_type" if role == "gland" else "niche"]
    column = None if result["diagnose"] else result["values"]["output_col" if role == "gland" else "niche_col"]
    return _cell_selection(table, result, cells, image, channel, name if pd.notna(name) else "Gland types", column)


def select_type_cells(root, table, job_id, image, kind, name, channel):
    """Select one fitted type in the open image; never choose a different image."""
    from . import tasks
    result, folder = saved_result(root, table, job_id)
    if result.get("diagnose"):
        raise ValueError("This is a K review. Fit the chosen K to assign types before showing them on the image.")
    if kind not in ("interior", "exterior") or name not in (result["profiles"].get(kind) or {}).get("rows", []):
        raise ValueError("Choose a gland type or niche from this run.")
    if result["source_after"] != tasks.source_revision({"table": str(table)}):
        raise ValueError("Cell data changed since this run. Rerun Gland types before linking its cells.")
    if image not in result["images"]:
        return dict(missing=True, message=f"{image} was not included in this run.", image=image)
    s = pd.read_parquet(folder / "gland_structures.parquet")
    role, name_col, output = ("gland", "gland_type", "output_col") if kind == "interior" else ("bed", "niche", "niche_col")
    matches = s[s.image.eq(image) & s[name_col].eq(name)]
    if matches.empty:
        return dict(missing=True, message=f"{name} is not present in {image}.", image=image)
    # Prefer a well-sampled example closest to its centre, within THIS image.
    ranked = matches.assign(_small=~matches[kind+"_status"].eq("Fit"))
    example = ranked.sort_values(["_small", kind+"_distance", "structure"], na_position="last").index[0]
    cells = pd.read_parquet(folder / "gland_cell_links.parquet",
                           filters=[("image", "=", image), (role, "in", [int(i) for i in matches.index])])
    zoom_cells = cells[cells[role].eq(example)]
    answer = _cell_selection(table, result, cells, image, channel, name, result["values"][output], zoom_cells)
    answer.update(structures=len(matches), example=str(s.loc[example, "structure"]))
    return answer


def _cell_selection(table, result, cells, image, channel, name, column, zoom_cells=None):
    from spacemap import dataio, table_collection
    from . import data_editor, flow_gating
    if cells.empty:
        raise ValueError("No cells are assigned to this part of the gland.")
    labels = pd.to_numeric(cells.label, errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(labels) & (labels > 0) & (labels == np.floor(labels)) & (labels <= 20000000)) or len(set(labels)) != len(labels):
        raise ValueError("Image linking needs unique positive segmentation labels within each image. Choose the correct Label column.")
    link_table = table_collection.source_for_image(table, image) if table_collection.is_collection(table) else table
    anchor_col = result["values"]["cell_id_col"]
    if table_collection.is_collection(table):
        anchor_col = next(r["anchor"] for r in table_collection.entries(table) if r["image"] == image)
    ids = dataio.load(link_table, columns=[anchor_col])[anchor_col].astype(str)
    if table_collection.is_collection(table):
        ids = json.dumps(image, ensure_ascii=False) + ":" + ids
    rows = pd.Index(ids).get_indexer(cells.cell_id)
    if (rows < 0).any():
        raise ValueError("Cell identities changed; rerun before linking cells.")
    selected_frame = pd.DataFrame({"_flow_table": str(link_table), "_flow_row": rows,
                                  "_flow_id": labels.astype(np.int64), "_flow_roi": image})
    selection = flow_gating.register_selection(channel, str(link_table), data_editor.stamp(link_table), selected_frame, name,
                                               colour=result.get("palette", {}).get(name))
    xy = (cells if zoom_cells is None else zoom_cells)[["x", "y"]].to_numpy(float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    bounds = None
    if len(xy):
        low, high = xy.min(axis=0)-20, xy.max(axis=0)+20
        bounds = [float(low[0]), float(low[1]), float(high[0]-low[0]), float(high[1]-low[1])]
    return dict(selection=selection, table=str(link_table), name=name, colour_by=column, bounds=bounds, image=image)


def profile_figure(profile):
    import io
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(max(7, len(profile["columns"])*.75), max(3.5, len(profile["rows"])*.55+2)), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    arr = np.array(profile["z"])
    limit = max(.1, float(np.abs(arr).max()))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i, row in enumerate(profile["raw"]):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if abs(arr[i,j]) > .6*limit else "black")
    ax.set_xticks(range(len(profile["columns"])), profile["columns"], rotation=40, ha="right")
    ax.set_yticks(range(len(profile["rows"])), profile["rows"])
    fig.colorbar(im, ax=ax, label="Relative Z (colour)")
    ax.set_title("Type profiles · measured proportions in cells (%)")
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=160)
    return stream.getvalue()


def k_review(model):
    """Describe saved K scores without refitting or treating a fixed K as a search."""
    rows = sorted((model or {}).get("diagnostics", []), key=lambda r: r["k"])
    valid = [r for r in rows if r.get("silhouette") is not None and np.isfinite(r["silhouette"])]
    best = max(valid, key=lambda r: r["silhouette"]) if len(valid) > 1 else None
    notes = []
    if not rows:
        notes.append("No well-sampled structures were available to fit this model.")
    elif not best:
        notes.append("Not enough scored K choices to recommend a number of types.")
    elif best["k"] == max(r["k"] for r in rows):
        notes.append("Best score is at the largest K tested; a higher K was not assessed.")
    return dict(rows=rows, suggested_k=best["k"] if best else None,
                score=best["silhouette"] if best else None, notes=notes,
                fitted=(model or {}).get("fitted", 0), scored=(model or {}).get("silhouette_n", 0))


def silhouette_details(structures, model, kind):
    """Score the saved fit sample, without refitting or using small predictions.

    Reproduce the exact transform and random sample used by fit_composition.
    Individual scores are unweighted, just like the mean in the K chart.
    The bounded sample keeps legacy-run diagnostics independent of cell tables.
    """
    from sklearn.metrics import silhouette_samples
    from threadpoolctl import threadpool_limits
    if kind not in ("interior", "exterior"):
        raise ValueError("Choose interior glands or exterior niches.")
    out = dict(schema=1, kind=kind, k=(model or {}).get("k"), fitted=0,
               scored=0, predicted_small=(model or {}).get("predicted_small", 0),
               mean=None, groups=[], points=[], state="unavailable", message="")
    if not model:
        out["message"] = "No fitted types are available for this compartment."
        return out
    fit = structures[structures[kind+"_status"].eq("Fit")]
    out["fitted"] = len(fit)
    if model["k"] < 2:
        out["message"] = "Silhouette needs at least two types; this run fitted K=1."
        return out
    sample = np.random.default_rng(model.get("method", {}).get("seed", 42)).choice(
        len(fit), min(2000, len(fit)), replace=False)
    selected = fit.iloc[sample]
    column = "gland_type" if kind == "interior" else "niche"
    labels = selected[column].astype(str).to_numpy()
    if not 1 < len(np.unique(labels)) < len(labels):
        out["message"] = "The saved score sample needs two types and more structures than types."
        return out
    counts = selected[[kind+"_count:"+name for name in model["features"]]].to_numpy(dtype=float)
    n = counts.sum(axis=1)
    strength = model["prior_strength"]
    x = (counts + strength*np.asarray(model["prior"])) / (n+strength)[:, None]
    if model["standardized"]:
        x = np.clip((x-np.asarray(model["mean"]))/np.asarray(model["scale"]), -model["clip"], model["clip"])
    x[:, ~np.asarray(model["active"], dtype=bool)] = 0
    with threadpool_limits(limits=4):
        scores = silhouette_samples(x, labels)
    out.update(state="ok", scored=len(scores), mean=float(scores.mean()))
    out["points"] = [dict(structure=str(row.structure), image=str(row.image),
                          type=str(name), selected_cells=int(count), silhouette=float(score))
                     for row, name, count, score in zip(selected.itertuples(), labels, n, scores)]
    for name in sorted(model["names"].values(), key=lambda s: int(s.split(" · ", 1)[0].split()[-1])):
        values = scores[labels == name]
        out["groups"].append(dict(name=name, fitted=int(fit[column].eq(name).sum()), scored=len(values),
            mean=float(values.mean()) if len(values) else None,
            quartiles=np.quantile(values, [.25, .5, .75]).tolist() if len(values) else [],
            negative=int((values < 0).sum()),
            negative_percent=float(100*(values < 0).mean()) if len(values) else None))
    return out


def silhouette_csv(data):
    """Export individual sampled scores with explicit scope and fitted K."""
    frame = pd.DataFrame(data["points"], columns=["structure", "image", "type", "selected_cells", "silhouette"])
    frame["kind"] = data["kind"]; frame["k"] = data["k"]
    frame["reference_structures"] = data["scored"]
    frame["fit_structures"] = data["fitted"]
    return frame.to_csv(index=False)


@lru_cache(maxsize=16)
def _saved_silhouette(folder, kind, stamps):
    folder = Path(folder)
    path = folder / f"{kind}_silhouette.json"
    if path.is_file():
        return json.loads(path.read_text())
    model = json.loads((folder / "gland_models.json").read_text()).get(kind)
    return silhouette_details(pd.read_parquet(folder / "gland_structures.parquet"), model, kind)


def saved_silhouette(root, table, job_id, kind):
    """On-demand legacy support; never touch cell tables or saved assignments."""
    if kind not in ("interior", "exterior"):
        raise ValueError("Choose interior glands or exterior niches.")
    _, folder = saved_result(root, table, job_id)
    files = [folder / "gland_models.json", folder / "gland_structures.parquet", folder / f"{kind}_silhouette.json"]
    stamps = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in files)
    return _saved_silhouette(str(folder.resolve()), kind, stamps)


def silhouette_figure(data, palette=None):
    """One row per type, individual structures plus median and middle 50%."""
    import io
    import textwrap
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    groups = data["groups"]
    fig = Figure(figsize=(10.8, max(3.5, 1.9+.8*len(groups))), layout="constrained")
    FigureCanvasAgg(fig); ax = fig.subplots()
    if groups:
        ax.axvspan(-1, 0, color="#cf6262", alpha=.08)
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        rng = np.random.default_rng(0)
        for i, group in enumerate(groups):
            values = [p["silhouette"] for p in data["points"] if p["type"] == group["name"]]
            colour = (palette or {}).get(group["name"], "#4477aa")
            ax.scatter(values, i+rng.uniform(-.17, .17, len(values)), s=12, alpha=.4, color=colour, edgecolors="none")
            if values:
                q1, med, q3 = group["quartiles"]
                ax.plot([q1, q3], [i, i], color=colour, linewidth=7, alpha=.9, solid_capstyle="butt")
                ax.plot([med, med], [i-.13, i+.13], color="#151d29", linewidth=2)
                note = f"n={group['scored']} · {group['negative_percent']:.1f}% < 0"
            else:
                note = "Not sampled"
            ax.text(1.02, i, note, transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.set_yticks(range(len(groups)), [textwrap.fill(g["name"], 31) for g in groups])
        ax.set_ylim(len(groups)-.6, -.6)
    else:
        ax.text(.5, .5, textwrap.fill(data["message"], 65), transform=ax.transAxes, ha="center", va="center")
        ax.set_yticks([])
    ax.set_xlim(-1, 1); ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_xlabel("Silhouette score · negative = closer to another type")
    title = "Gland types · individual gland scores" if data["kind"] == "interior" else "Niche types · individual bed scores"
    mean = f" · mean {data['mean']:.3f}" if data["mean"] is not None else ""
    ax.set_title(f"{title}\nSaved K={data['k']} · {data['scored']:,}/{data['fitted']:,} fit structures scored{mean}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Each dot = one scored structure. Thick line = middle 50%; tick = median.\n"
                  "Same sample as the K score; small predicted structures excluded. Not an accuracy measure.", fontsize=9)
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=160)
    return stream.getvalue()


def k_review_figure(model, kind):
    """Export the same saved-score bar chart used by Studio and Classic."""
    import io
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    review = k_review(model)
    fig = Figure(figsize=(7, 4.4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    valid = [r for r in review["rows"] if r.get("silhouette") is not None and np.isfinite(r["silhouette"])]
    if valid:
        scores = [r["silhouette"] for r in valid]
        ax.bar([r["k"] for r in valid], scores,
               color=["#29814d" if r["k"] == review["suggested_k"] else "#4477aa" for r in valid])
        for row in valid:
            ax.annotate(f"{row['silhouette']:.3f}", (row["k"], row["silhouette"]),
                        xytext=(0, 5 if row["silhouette"] >= 0 else -12), textcoords="offset points", ha="center", fontsize=10)
        low, high = min(0., *scores), max(.05, *scores)
        pad = max(.04, (high-low)*.18)
        ax.set_ylim(low-pad if low < 0 else 0, high+pad)
    else:
        ax.text(.5, .5, "Separation score unavailable", transform=ax.transAxes, ha="center")
    ax.set_xticks([r["k"] for r in review["rows"]])
    ax.set_xlabel("Number of types (K)"); ax.set_ylabel("Separation score (silhouette) · higher is better")
    title = "Gland types · interior" if kind == "interior" else "Niche types · exterior"
    subtitle = f"Suggested K = {review['suggested_k']} · best of the tested choices" if review["suggested_k"] else "No K recommendation"
    ax.set_title(title + "\n" + subtitle)
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    foot = "Suggested from the saved selection; not proof of biological types."
    if review["notes"]:
        foot += "\n" + "\n".join(review["notes"])
    fig.supxlabel(foot, fontsize=9)
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=160)
    return stream.getvalue()


def clinical_figure(comparison, view, stratum, palette):
    """One patient dot per row; effect direction names the actual groups."""
    import io
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    tests = [r for r in comparison["tests"] if r["view"] == view and r["stratum"] == stratum]
    if not tests:
        raise ValueError("No comparison for these visits and strata.")
    fig = Figure(figsize=(11, max(4, .75*len(tests)+1.8)), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    rng = np.random.default_rng(42)
    labels = []
    for i, row in enumerate(tests):
        vals = [v for v in comparison["values"] if v["view"] == view and v["stratum"] == stratum and v["kind"] == row["kind"] and v["feature"] == row["feature"]]
        for j, group in enumerate((row["group_a"], row["group_b"])):
            x = [v["value"] for v in vals if v["group"] == group]
            color = palette.get(group, ("#4477aa", "#ee6677")[j])
            y = i + (j-.5)*.28
            if x:
                ax.boxplot(x, positions=[y], widths=.2, vert=False, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, alpha=.3), medianprops=dict(color=color))
                ax.scatter(x, y+rng.uniform(-.07,.07,len(x)), s=18, color=color,
                           label=group if i == 0 else None, zorder=3)
        fmt = lambda v: "—" if v is None else f"{v:.3g}"
        labels.append(f"{row['feature']}\np={fmt(row['p'])} · q={fmt(row['q'])} · n={row['n_a']}/{row['n_b']}")
    ax.set_yticks(range(len(tests)), labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("Change in weighted burden (percentage points)" if view.startswith("Δ") else "Weighted burden (%)")
    ax.set_title(f"{view} · {stratum} · one dot per independent sample")
    ax.legend(loc="best"); ax.grid(axis="x", alpha=.2)
    stream = io.BytesIO(); fig.savefig(stream, format="png", dpi=150)
    return stream.getvalue()
