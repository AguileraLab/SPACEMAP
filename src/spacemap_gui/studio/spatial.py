"""Spatial analysis and Visualization for Studio, on the combined table, each result linked to the image's cells.

Reads only the columns asked for (from the table's Parquet form). What is computed here is shown beside the image
and drawn on it: a group's cells coloured or filtered, a density map, triangles of a triad, a tissue outline.

- ``composition``: cells of each group in each image (Proportions, where each neighbourhood is).
- ``profile``: each group's mean marker values (the heatmap).
- ``distance_profile``: share of each group in bands of distance from an annotation (Cell tagging's distances).
- ``agreement``: one phenotype column against a reference (the F1 paper figures' measures), cell by cell.
- jobs: ``density_job`` (cells per mm², classic's virtual tissue area), ``triads_job`` (classic's triad test),
  ``screening_job`` (classic's feature screening), ``umap_job`` (an embedding of a fair sample of cells).
"""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE = "batch_id"
POSITION = ("x_centroid", "y_centroid")


# ---------------------------------------------------------------- reading
def _file(table):
    """The table's Parquet form (itself or its twin) or None (a small CSV, read whole)."""
    from .data_view import _parquet_of

    return _parquet_of(table)


def columns(table):
    from spacemap import dataio

    return list(dataio.columns(str(table)))


def read(table, names, images=None):
    """A frame of *names* (and the image column), this image's rows only when *images* is given."""
    from spacemap import dataio

    names = list(dict.fromkeys([IMAGE, *names]))
    available = set(columns(table))
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError("The combined table has no " + ", ".join(missing[:4]) + " column.")
    frame = dataio.load(str(table), columns=names)
    if images:
        frame = frame[frame[IMAGE].astype(str).isin([str(i) for i in images])]
    return frame


def _text(series):
    return series.astype(object).where(series.notna(), None).map(lambda v: None if v is None else str(v))


def markers_of(table):
    """The marker columns: numbers that are not positions, ids, shape measures, distances, sizes or results."""
    skip = {"x_centroid", "y_centroid", "label", "cell_id", "area", "eccentricity", "solidity", "extent", "orientation",
            "perimeter", "major_axis_length", "minor_axis_length", "patient", IMAGE, "tissue_roi"}
    parquet = _file(table)
    if parquet is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.read_schema(parquet)
        numeric = {f.name for f in schema if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)}
    else:
        sample = pd.read_csv(table, nrows=500)
        numeric = {c for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])}
    names = []
    for name in columns(table):
        low = name.lower()
        if name not in numeric or low in skip or low.endswith(("_distance", "_roi", "_roi_size", "_bed_id", "_cn", "_um", "_px", "_index")) \
                or low == "tls" or low.startswith(("tls_", "phenotype", "cluster", "neighbo", "unnamed")):
            continue
        names.append(name)
    return names


# ---------------------------------------------------------------- quick results (shown at once)
def composition(table, column, images=None, top=40):
    """Cells of each group of *column* in each image: {"images", "groups", "counts": [[...]]} (groups by size)."""
    frame = read(table, [column], images)
    values = _text(frame[column]).fillna("Unassigned")
    counts = pd.crosstab(frame[IMAGE].astype(str), values)
    order = counts.sum(axis=0).sort_values(ascending=False).index[:top]
    counts = counts[order]
    return {"column": column, "images": counts.index.tolist(), "groups": [str(g) for g in counts.columns],
            "counts": counts.to_numpy().astype(int).tolist(), "cells": int(counts.to_numpy().sum())}


def profile(table, column, markers, images=None, per_group=20000, seed=0):
    """Each group's mean marker value (arcsinh, then z across all cells) for a heatmap; a fair sample per group."""
    if not markers:
        markers = markers_of(table)[:60]
    frame = read(table, [column, *markers], images)
    frame = frame[frame[column].notna()]
    groups = _text(frame[column])
    rng = np.random.default_rng(seed)
    keep = []
    for _, index in groups.groupby(groups).groups.items():
        index = np.asarray(index)
        keep.append(index if len(index) <= per_group else rng.choice(index, per_group, replace=False))
    frame = frame.loc[np.concatenate(keep)] if keep else frame.iloc[:0]
    values = np.arcsinh(frame[markers].apply(pd.to_numeric, errors="coerce").to_numpy(float) / 5.0)
    mean, spread = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
    spread[spread <= 0] = 1
    z = (values - mean) / spread
    labels = _text(frame[column]).to_numpy()
    rows = sorted(set(labels), key=lambda g: -int((labels == g).sum()))
    table_z = [np.nanmean(z[labels == g], axis=0) for g in rows]
    raw = [np.nanmean(values[labels == g], axis=0) for g in rows]
    clean = lambda m: np.round(np.nan_to_num(np.clip(np.asarray(m, float), -10, 10)), 3).tolist()
    return {"column": column, "markers": markers, "profile": {"rows": rows, "columns": markers, "values": clean(table_z), "raw": clean(raw)},
            "cells": {g: int((labels == g).sum()) for g in rows}}


def distance_profile(table, column, distance, images=None, step=25.0, limit=300.0, inside=True):
    """The share of each group of *column* in bands of *distance* (µm from an annotation's edge; negative inside),
    and each band's cells (to show on the image)."""
    frame = read(table, [column, distance], images)
    near = pd.to_numeric(frame[distance], errors="coerce")
    low = -limit if inside else 0.0
    edges = np.arange(low, limit + step, step)
    frame = frame[(near >= edges[0]) & (near <= edges[-1])]
    near = near.loc[frame.index]
    band = np.clip(np.digitize(near.to_numpy(), edges) - 1, 0, len(edges) - 2)
    groups = _text(frame[column]).fillna("Unassigned").to_numpy()
    names = pd.Series(groups).value_counts().index[:30].tolist()
    counts = np.zeros((len(names), len(edges) - 1), int)
    position = {g: i for i, g in enumerate(names)}
    for g, b in zip(groups, band):
        i = position.get(g)
        if i is not None:
            counts[i, b] += 1
    totals = counts.sum(axis=0)
    share = np.where(totals > 0, counts / np.maximum(totals, 1), 0.0)
    labels = [f"{a:g} to {b:g}" for a, b in zip(edges[:-1], edges[1:])]
    # Colour: how much more (or less) a group is in a band than overall (log2 of the ratio).
    overall = counts.sum(axis=1) / max(counts.sum(), 1)
    enrichment = np.log2((share + 1e-3) / (overall[:, None] + 1e-3))
    return {"column": column, "distance": distance, "edges": edges.tolist(), "bands": labels, "groups": names,
            "counts": counts.tolist(), "totals": totals.tolist(),
            "profile": {"rows": names, "columns": labels, "values": np.round(np.clip(enrichment, -3, 3), 3).tolist(),
                        "raw": np.round(share * 100, 2).tolist()}}


def agreement(table, reference, method, images=None):
    """*method* against *reference* cell by cell: the confusion table and each phenotype's precision, recall and F1."""
    frame = read(table, [reference, method], images)
    frame = frame[frame[reference].notna() & frame[method].notna()]
    truth, guess = _text(frame[reference]), _text(frame[method])
    confusion = pd.crosstab(truth, guess)
    names = sorted(set(confusion.index) | set(confusion.columns), key=lambda n: -int(truth.eq(n).sum()))
    confusion = confusion.reindex(index=names, columns=names, fill_value=0)
    rows = []
    for name in names:
        tp = int(confusion.loc[name, name])
        fp, fn = int(confusion[name].sum()) - tp, int(confusion.loc[name].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"phenotype": name, "cells": int(confusion.loc[name].sum()), "precision": round(precision, 3),
                     "recall": round(recall, 3), "f1": round(f1, 3)})
    total = int(confusion.to_numpy().sum())
    share = confusion.div(confusion.sum(axis=1).replace(0, 1), axis=0)
    return {"reference": reference, "method": method, "cells": total, "accuracy": round(float(np.trace(confusion.to_numpy())) / max(total, 1), 4),
            "macro_f1": round(float(np.mean([r["f1"] for r in rows])) if rows else 0.0, 4), "phenotypes": rows,
            "confusion": {"rows": names, "columns": names, "values": np.round(share.to_numpy() * 100, 2).tolist(),
                          "counts": confusion.to_numpy().astype(int).tolist()}}


# ---------------------------------------------------------------- jobs
def _out(root, kind):
    folder = Path(root) / "outputs/spatial_analysis" / kind / ("studio_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def density_job(settings, folder, report):
    """Cells per mm² of each group in each image (and region): the area is the cells' own tissue outline, as classic
    Tab 8 draws it (each cell widened by the spacing of its neighbours, holes below the size given filled). The
    outlines are kept as GeoJSON (one per image) to draw on the image."""
    from shapely.geometry import mapping

    from spacemap_gui.tabs.tab8_analysis.density_analysis import _virtual_tissue_geometry

    table, column = settings["table"], settings["column"]
    region = settings.get("region") or ""
    frame = read(table, [column, *POSITION, *([region] if region else [])], settings.get("images") or None)
    out = _out(settings["root"], "density")
    microns = float(settings.get("pixel_um", 0.5083))
    keys = [IMAGE, region] if region else [IMAGE]
    grouped = list(frame.groupby(keys, dropna=False))
    rows, outlines = [], {}
    for number, (key, part) in enumerate(grouped):
        key = key if isinstance(key, tuple) else (key,)
        name = str(key[0]) + (" · " + str(key[1]) if region else "")
        report(f"{key[0]}: tissue outline ({number + 1} of {len(grouped)})", number / max(len(grouped), 1))
        if len(part) < int(settings.get("min_cells", 20)):
            continue
        coords = part[list(POSITION)].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
        polygon, _, _ = _virtual_tissue_geometry(coords, float(settings.get("percentile", 99.0)), float(settings.get("min_hole_px2", 10000.0)))
        if polygon is None or polygon.is_empty:
            continue
        area = float(polygon.area) * (microns / 1000.0) ** 2
        counts = _text(part[column]).fillna("Unassigned").value_counts()
        for group, count in counts.items():
            rows.append({"image": str(key[0]), "region": str(key[1]) if region else "", "unit": name, "group": group,
                         "cells": int(count), "area_mm2": area, "density_per_mm2": count / area if area else np.nan})
        outlines.setdefault(str(key[0]), []).append({"type": "Feature", "geometry": mapping(polygon),
                                                    "properties": {"name": name, "area_mm2": round(area, 5)}})
    if not rows:
        raise ValueError("No image has enough cells with positions.")
    result = pd.DataFrame(rows)
    result.to_csv(out / "density_per_unit.csv", index=False)
    for image, features in outlines.items():
        (out / f"{image}_tissue_outline.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    wide = result.pivot_table(index="unit", columns="group", values="density_per_mm2", aggfunc="sum").fillna(0)
    order = wide.sum(axis=0).sort_values(ascending=False).index[:40]
    wide = wide[order]
    areas = result.groupby("unit")["area_mm2"].first()
    images = result.groupby("unit")["image"].first()
    log = np.log10(wide.to_numpy() + 1)
    z = (log - log.mean(axis=0)) / np.where(log.std(axis=0) > 0, log.std(axis=0), 1)
    return {"folder": str(out), "column": column, "units": wide.index.tolist(), "images": [images[u] for u in wide.index],
            "areas_mm2": [round(float(areas[u]), 5) for u in wide.index], "groups": [str(g) for g in wide.columns],
            "profile": {"rows": wide.index.tolist(), "columns": [str(g) for g in wide.columns],
                        "values": np.round(np.clip(z, -3, 3), 3).tolist(), "raw": np.round(wide.to_numpy(), 2).tolist()},
            "outlines": sorted(outlines)}


def triads_job(settings, folder, report):
    """Classic Tab 8's triad test: how often cells of phenotypes A, B and C sit together (each within the distance of
    the others), against the same cells with their phenotypes shuffled; each triangle found is kept (its three cells
    and positions) to draw on the image."""
    from spacemap_gui.tabs.tab8_analysis import triads

    chosen = settings["values"]
    column = chosen["phenotype_col"]
    region = chosen.get("roi_col") or ""
    batch = chosen.get("batch_col") or IMAGE
    frame = read(settings["table"], [column, *POSITION, "label", *([region] if region else []), *([batch] if batch != IMAGE else [])],
                 settings.get("images") or None)
    frame[list(POSITION)] = frame[list(POSITION)].apply(pd.to_numeric, errors="coerce")
    report("Counting triads and shuffling phenotypes…", 0.1)
    triad = tuple(str(p) for p in chosen["triad"])
    summary, per_region, null, instances = triads._detect_triad_enrichment(
        frame, triad, column, batch, region, "None", "", float(chosen["min_um"]), float(chosen["max_um"]),
        float(chosen["pixel_um"]), int(chosen["min_cells"]), int(chosen["n_permutations"]), int(chosen.get("seed", 0)), False,
        int(chosen.get("max_saved", 200000)))
    out = _out(settings["root"], "triads")
    if not instances.empty:
        labels = frame["label"].to_numpy()
        for corner in ("a", "b", "c"):
            index = instances[f"cell_{corner}_global_index"].to_numpy()
            instances[f"cell_{corner}_label"] = labels[frame.index.get_indexer(index)]
        instances["batch_id"] = instances["batch"]
        instances["x_centroid"], instances["y_centroid"] = instances["triad_centroid_x"], instances["triad_centroid_y"]
    points = out / f"triad_instances_{out.name}.csv"  # its own name: density maps list point tables by file name
    instances.to_csv(points, index=False)
    per_region.to_csv(out / "triad_counts_per_region.csv", index=False)
    (out / "triad_summary.json").write_text(json.dumps({**summary, "settings": chosen}, default=float, indent=2))
    hist, edges = np.histogram(null, bins=30) if len(null) else (np.array([]), np.array([]))
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in summary.items()}
    return {"folder": str(out), "table": str(points), "table_name": "Triads / " + points.name, "summary": clean, "triad": list(triad), "column": column,
            "per_image": per_region.rename(columns={"batch": "image"}).to_dict("records") if not per_region.empty else [],
            "null": {"counts": hist.astype(int).tolist(), "edges": np.round(edges, 2).tolist()}}


def triad_shapes(table, image, limit=5000):
    """The triangles of one image from a triad run's table, as closed rings (image pixels) to draw."""
    frame = pd.read_csv(table)
    frame = frame[frame["batch"].astype(str) == str(image)].head(limit)
    rings = [[[r.cell_a_x, r.cell_a_y], [r.cell_b_x, r.cell_b_y], [r.cell_c_x, r.cell_c_y], [r.cell_a_x, r.cell_a_y]]
             for r in frame.itertuples()]
    return {"rings": rings, "count": len(rings)}


SCREENING_DEFAULTS = {
    "outer_cap_mode": "percentile", "zone_geometry": "cumulative", "outer_cap_percentile": 95.0, "outer_cap_value": 300.0,
    "inner_cap_percentile": 5.0, "n_candidate_widths": 15, "round_width_to": 5.0, "max_width": 0.0, "exclude_negative_distances": True,
    "use_micron_per_pixel": True, "micron_per_pixel": 0.5083, "min_cells_per_zone": 20, "min_cells_for_feature": 10,
    "ratio_pseudocount": 1.0, "log_transform_ratios": True, "auto_interaction_distance": True, "auto_interaction_percentile": 95.0,
    "allow_self_interaction_features": False, "use_adaptive_interaction_radius": True, "adaptive_radius_direct_neighbours_only": True,
    "adaptive_radius_min_pct": 0.1, "adaptive_radius_max_pct": 99.0, "adaptive_radius_max_sample": 500,
    "dyad_triad_permutation_n_iter": 1000, "min_obs_motif_count": 3, "permutation_chunk_size": 64, "permutation_use_gpu": True,
    "n_jobs": 1, "dyad_min_dist_um": 0.0, "dyad_max_dist_um": 30.0, "triad_min_dist_um": 0.0, "triad_max_dist_um": 30.0,
    "triad_mode": "all", "triad_definitions": [], "nn_k": 3, "area_use_median": False, "area_percentile": 95.0, "min_hole_size_px2": 0.0,
    "patient_aggregation": "median", "split_method": "median", "quantile_low": 0.25, "quantile_high": 0.75, "custom_cutpoint": 0.0,
    "min_patients_per_group": 5, "show_ci": False, "show_excluded_km_curve": False, "volcano_label_discovery": False,
    "volcano_label_nominal": False, "volcano_label_other_top_n": 3, "time_unit": "Months", "fdr_mode": "per_family", "fdr_alpha": 0.05,
    "fdr_alpha_discovery": 0.10, "top_k_for_bootstrap": 5, "bootstrap_iterations": 30, "bootstrap_sampling_unit": "patient",
    "bootstrap_patient_fraction": 0.80, "bootstrap_random_seed": 20260407, "feature_bootstrap_neighbor_steps": 1,
    "feature_stability_top_n": 25, "min_patients_with_zone": 12}


def screening_config(table, chosen):
    """Classic Tab 8's feature screening settings from Studio's few choices (the rest are classic's defaults)."""
    names = columns(table)
    endpoint = chosen.get("endpoint", {})
    if endpoint.get("type") == "survival":
        endpoints = [{"enabled": True, "type": "survival", "label": endpoint.get("label") or "Survival", "time_col": endpoint["time_col"],
                      "censor_col": endpoint["censor_col"], "censor_flag_means_censored": bool(endpoint.get("censored_flag", False))}]
    else:
        endpoints = [{"enabled": True, "type": "binary", "multi_mode": False, "multi_cols": [], "label": endpoint.get("label") or "Response",
                      "group_col": endpoint["group_col"], "positive_value": str(endpoint.get("positive_value", ""))}]
    return {"source_csv": str(table),
            "columns": {"patient": chosen["patient_col"], "batch": chosen.get("batch_col") or IMAGE, "roi": chosen.get("roi_col") or IMAGE,
                        "x": "x_centroid", "y": "y_centroid", "phenotype": chosen["phenotype_col"], "distance": chosen["distance_col"],
                        "context_column": "", "context_values": [], "stratify": []},
            "selected_phenotypes": chosen.get("phenotypes") or [], "excluded_phenotypes": chosen.get("excluded") or [],
            "feature_families": chosen.get("families") or ["density", "proportion", "ratio", "dyad", "triad", "diversity", "nn_distance"],
            "row_filters": {}, "endpoints": endpoints,
            "screening": {**SCREENING_DEFAULTS, "micron_per_pixel": float(chosen.get("pixel_um", 0.5083))},
            "representatives": {"raw_image_dir": "", "max_examples_per_group": 3, "thumbnail_padding_px": 120, "thumbnail_max_dim": 700},
            "_available": names}


def screening_job(settings, folder, report):
    """Classic Tab 8's feature screening with Studio's choices; the run is saved where classic saves it."""
    from spacemap_gui.tabs.tab8_analysis import feature_screening_core as core

    config = screening_config(settings["table"], settings["values"])
    config.pop("_available", None)
    chosen = settings["values"]
    endpoint = chosen.get("endpoint", {})
    wanted = [chosen["patient_col"], chosen["phenotype_col"], chosen["distance_col"], "label", *POSITION,
              *[endpoint.get(k) for k in ("time_col", "censor_col", "group_col") if endpoint.get(k)],
              *[c for c in (chosen.get("batch_col"), chosen.get("roi_col")) if c]]
    frame = read(settings["table"], wanted, settings.get("images") or None)
    if not chosen.get("phenotypes"):
        config["selected_phenotypes"] = _text(frame[chosen["phenotype_col"]]).dropna().value_counts().index[:12].tolist()
    lines = []
    log = lambda message: lines.append(str(message))
    def progress(*args, **info):
        info = {**(args[0] if args and isinstance(args[0], dict) else {}), **info}
        report(str(info.get("message", "Screening…")), min(0.95, (info.get("current") or 0) / max(info.get("total") or 1, 1)))
    results = core.run_feature_screening_pipeline(frame, config, log_fn=log, progress_fn=progress)
    run = core.save_feature_screening_run(results, config, Path(settings["root"]) / "outputs/spatial_analysis/feature_screening", lines, log_fn=log)
    # The strongest features (each in its zone), with the phenotypes they are made of and the images where they are
    # highest and lowest, to show on the image.
    stats = results["stats_all"].sort_values(["p_value", "fdr_q"], na_position="last").drop_duplicates(["feature_name"]).head(30)
    features = pd.concat(results["combined_by_width"].values(), ignore_index=True) if results["combined_by_width"] else pd.DataFrame()
    hits = []
    for row in stats.itertuples():
        found = features[(features["feature_name"] == row.feature_name) & (features["zone_width"] == row.zone_width)] if not features.empty else features
        found = found.dropna(subset=["value"]).sort_values("value") if not found.empty else found
        ends = lambda part: [[str(b), round(float(v), 4)] for b, v in zip(part["batch"], part["value"])]
        effect = next((getattr(row, k) for k in ("hazard_ratio", "auc", "effect_size") if pd.notna(getattr(row, k, np.nan))), None)
        hits.append({"feature": row.feature_label, "family": row.feature_family, "zone": row.zone_label, "width": float(row.zone_width),
                     "p": float(row.p_value) if pd.notna(row.p_value) else None, "q": float(row.fdr_q) if pd.notna(row.fdr_q) else None,
                     "effect": float(effect) if effect is not None else None, "patients": int(row.n_patients),
                     "phenotypes": [p for p in str(row.feature_name).split("::")[1:] if p],
                     "highest": ends(found.tail(5).iloc[::-1]) if not found.empty else [], "lowest": ends(found.head(5)) if not found.empty else []})
    return {"run": Path(run).name, "folder": str(run), "selected_width": float(results["selected_width"]), "hits": hits,
            "phenotype_column": chosen["phenotype_col"], "distance_column": chosen["distance_col"]}


def umap_job(settings, folder, report):
    """A UMAP of a fair sample of cells (the same number from each image) on their markers (arcsinh, z-scored);
    each point keeps its image and cell number, so a selection is shown on the image."""
    import umap

    from sklearn.decomposition import PCA

    table, markers = settings["table"], settings.get("markers") or markers_of(settings["table"])[:60]
    report("Reading the cells…", 0.05)
    frame = read(table, [*markers, "label"], settings.get("images") or None)
    rng = np.random.default_rng(int(settings.get("seed", 0)))
    per_image = int(settings.get("per_image", 2000))
    keep = []
    for _, index in frame.groupby(frame[IMAGE].astype(str)).groups.items():
        index = np.asarray(index)
        keep.append(index if len(index) <= per_image else rng.choice(index, per_image, replace=False))
    frame = frame.loc[np.concatenate(keep)]
    values = np.arcsinh(frame[markers].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float) / 5.0)
    spread = values.std(axis=0)
    values = (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1)
    report(f"PCA and UMAP of {len(frame):,} cells…", 0.3)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=8):  # unlimited thread pools on a many-core node crash OpenBLAS
        pcs = PCA(n_components=min(20, values.shape[1]), random_state=0).fit_transform(values)
        embedding = umap.UMAP(n_neighbors=int(settings.get("n_neighbors", 15)), min_dist=float(settings.get("min_dist", 0.1)),
                              random_state=int(settings.get("seed", 0))).fit_transform(pcs)
    out = Path(settings["root"]) / "outputs/studio/umap"
    out.mkdir(parents=True, exist_ok=True)
    name = "umap_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(out / (name + ".npz"), xy=embedding.astype(np.float32), image=frame[IMAGE].astype(str).to_numpy().astype("U"),
                        label=pd.to_numeric(frame["label"], errors="coerce").fillna(-1).astype(np.int64).to_numpy(),
                        row=frame.index.to_numpy(np.int64))
    (out / (name + ".json")).write_text(json.dumps({"table": str(table), "markers": markers, "cells": int(len(frame)),
                                                   "per_image": per_image, "created": datetime.now().isoformat()}))
    return {"name": name, "cells": int(len(frame)), "markers": len(markers)}


def umap_points(root, name, table, column=""):
    """A saved UMAP's points with their image, cell number and (when asked) the group of *column*."""
    if not name.replace("_", "").isalnum():
        raise ValueError("Choose a saved UMAP.")
    path = Path(root) / "outputs/studio/umap" / (name + ".npz")
    if not path.is_file():
        raise ValueError("That UMAP is no longer saved.")
    with np.load(path, allow_pickle=False) as data:
        xy, image, label, row = data["xy"], data["image"], data["label"], data["row"]
    groups = None
    if column:
        values = read(table, [column])[column]
        groups = _text(values.reindex(row)).fillna("Unassigned").tolist()
    return {"x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(), "image": image.tolist(),
            "label": label.tolist(), "groups": groups, "column": column}


def saved_umaps(root):
    folder = Path(root) / "outputs/studio/umap"
    found = []
    for path in sorted(folder.glob("umap_*.json"), reverse=True) if folder.is_dir() else []:
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        found.append({"name": path.stem, **{k: info.get(k) for k in ("cells", "created")}})
    return found
