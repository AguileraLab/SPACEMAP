"""Studio spatial feature screening: explicit design, exact features, sample inference.

Research tables are read-only. Runs retain their complete search, denominators,
exclusions and nominal/adjusted tests. Feature selection never filters the
background population used to measure proportions or spatial composition.
"""
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations, permutations
import json
import hashlib
import math
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

FAMILIES = {"density": "Density", "proportion": "Proportion", "ratio": "Ratios",
            "diversity": "Diversity", "nn_distance": "Neighbour distance",
            "dyad": "Pairs (dyads)", "triad": "Triplets (triads)"}
DIVERSITY = {"shannon": "Shannon", "simpson": "Simpson", "richness": "Number of types", "evenness": "Evenness"}


class NoEligibleCells(ValueError):
    """A declared comparison has no measurable population after fixed filters."""


DEFAULTS = dict(sample_col="", image_col="batch_id", phenotype_col="", x_col="x_centroid", y_col="y_centroid",
    region_col="", phenotypes=[], filters=[], families=["proportion"],
    zone_source="distance", distance_col="", distance_unit="", coordinate_unit="px", pixel_um=0.0,
    zone_mode="cumulative", side="outside", widths=[50, 100, 150, 200, 250, 300],
    annotation_folder="", annotation_pattern="{image}.geojson", study_class="", target_class="",
    area_table="", min_cells=20, aggregation="median", min_samples=3,
    endpoint=dict(type="binary", group_col="", group_a="", group_b="", paired=False,
                  value_col="", time_col="", event_col="", censored=False,
                  change_col="", change_before="", change_after=""),
    split_by=[], split_depth=1, include_overall=True, extra_comparisons=[],
    change=dict(enabled=False, column="", before="", after=""),
    options=dict(density={"area_source": "polygons", "area_percentile": 95.0}, proportion={"denominator": "all", "labels": []},
                 ratio={"pseudocount": 0.5, "log2": True}, diversity={"metric": "shannon"},
                 nn_distance={"k": 1, "targets": "zone", "same_type": False},
                 dyad={"min_um": 0., "max_um": 30., "metric": "zscore", "same_type": False, "pseudocount": 0.5},
                 triad={"min_um": 0., "max_um": 30., "metric": "zscore", "same_type": False, "pseudocount": 0.5}),
    motif_permutations=1000, motif_min_count=3, motif_use_gpu=True,
    group_test="mannwhitney", correction="fdr_bh", correction_scope="family", alpha=0.05, seed=473, permutations=9999,
    resampling="off", resamples=100, sample_fraction=.8, nearby_zones=1)


def defaults(saved=None):
    value = deepcopy(DEFAULTS)
    # Previously saved Studio designs corrected the whole scan. Preserve that
    # meaning until the user explicitly chooses a different scope.
    if saved and "correction_scope" not in saved:
        value["correction_scope"] = "global"
    for key, item in (saved or {}).items():
        if key in ("endpoint", "options", "change"):
            if key in ("endpoint", "change"):
                value[key].update(item)
            else:
                for family, options in item.items():
                    if family in value[key]:
                        value[key][family].update(options)
        elif key in value:
            value[key] = deepcopy(item)
    # Old area-file settings remain usable without an extra selection.
    if value["area_table"] and "area_source" not in (saved or {}).get("options", {}).get("density", {}):
        value["options"]["density"]["area_source"] = "csv"
    return value


def diversity_metrics(c):
    """Keep legacy one-measure settings; an explicit empty selection stays empty."""
    opt = c["options"]["diversity"]
    return opt.get("metrics", [opt.get("metric", "shannon")])


def settings(root):
    from .settings import read_config, revision
    saved = read_config(root).get("tab_configurations", {}).get("tab8_analysis", {}).get("studio_screening", {})
    chosen = defaults(saved)
    for key in ("annotation_folder", "area_table"):
        chosen[key] = chosen[key].replace("${PROJECT_ROOT}", str(Path(root).resolve()))
    return {"values": chosen, "revision": revision(chosen)}


def save(root, chosen, expected):
    from spacemap_gui import persistence
    from .settings import read_config, project_lock
    if not isinstance(chosen, dict) or len(json.dumps(chosen)) > 100_000:
        raise ValueError("Invalid screening settings.")
    with project_lock(root):
        if settings(root)["revision"] != expected:
            raise ValueError("Screening settings changed in another window. Reopen this panel before saving.")
        config = read_config(root)
        config.setdefault("tab_configurations", {}).setdefault("tab8_analysis", {})["studio_screening"] = defaults(chosen)
        persistence.save_project_config(str(root), config)
    return settings(root)


def units(table):
    """Only declared units; a column name is not a unit of measurement."""
    from spacemap import table_collection
    if table_collection.is_collection(table):
        found = {}
        for row in table_collection.entries(table):
            for name, unit in units(row['path']).items():
                if name in found and found[name] != unit:
                    raise ValueError(f"{name} has different recorded units across image tables. Use consistent units before screening.")
                found[name] = unit
        return found
    result = {}
    if str(table).endswith(".parquet"):
        import pyarrow.parquet as pq
        metadata = pq.read_schema(table).metadata or {}
        try:
            result = json.loads(metadata.get(b"spacemap_column_units", b"{}"))
        except (ValueError, TypeError):
            pass
    return result


def check(chosen, names):
    c = defaults(chosen)
    from .screening_statistics import SCOPES
    if c["correction_scope"] not in SCOPES:
        raise ValueError("Choose which comparisons to correct together.")
    if c["resampling"] not in ("off", "subsample", "bootstrap"):
        raise ValueError("Choose Off, sample subsets, or bootstrap resampling.")
    for key, lo, hi in [("resamples", 20, 1000), ("nearby_zones", 0, 3)]:
        if not float(c[key]).is_integer() or not lo <= float(c[key]) <= hi:
            raise ValueError(f"{key} must be an integer between {lo} and {hi}.")
    if not .5 <= float(c["sample_fraction"]) < 1:
        raise ValueError("The subset fraction must be at least 0.5 and less than 1.")
    required = [c[k] for k in ("sample_col", "image_col", "phenotype_col", "x_col", "y_col")]
    from .screening_comparisons import validate_design
    required.extend(validate_design(c, names))
    if c["zone_source"] == "distance":
        required.append(c["distance_col"])
        if c["distance_unit"] not in ("px", "um"):
            raise ValueError("Choose the saved distance unit: pixels or µm.")
    if c["region_col"]:
        required.append(c["region_col"])
    endpoint = c["endpoint"]
    if endpoint.get("change_col"):
        required.append(endpoint["change_col"])
        if endpoint["type"] != "binary" or endpoint["paired"] or not endpoint.get("change_before") or not endpoint.get("change_after") or endpoint["change_before"] == endpoint["change_after"]:
            raise ValueError("Compare changes between independent groups using two different repeated observations.")
    if endpoint["type"] == "binary":
        required.append(endpoint["group_col"])
        if not endpoint["group_a"] or not endpoint["group_b"] or endpoint["group_a"] == endpoint["group_b"]:
            raise ValueError("Choose two different comparison groups.")
    elif endpoint["type"] == "continuous":
        required.append(endpoint["value_col"])
    elif endpoint["type"] == "survival":
        required.extend([endpoint["time_col"], endpoint["event_col"]])
    elif endpoint["type"] != "describe":
        raise ValueError("Choose a comparison type.")
    for rule in c["filters"]:
        required.append(rule.get("column", ""))
        if not rule.get("values"):
            raise ValueError("Choose values for each region filter, or remove it.")
    if any(not col or col not in names for col in required):
        raise ValueError("Choose valid columns for the sample, image, cell type, coordinates, regions and comparison.")
    if not c["phenotypes"] or len(c["phenotypes"]) > 40:
        raise ValueError("Choose between 1 and 40 cell types to screen.")
    if not c["families"] or set(c["families"]) - FAMILIES.keys():
        raise ValueError("Choose at least one of the seven feature types.")
    metrics = diversity_metrics(c)
    if not isinstance(metrics, list) or any(not isinstance(m, str) or m not in DIVERSITY for m in metrics) or len(set(metrics)) != len(metrics):
        raise ValueError("Choose valid, distinct diversity measures.")
    if "diversity" in c["families"] and not metrics:
        raise ValueError("Select at least one diversity measure, or untick Diversity.")
    for family, size in (("ratio", 2), ("dyad", 2), ("triad", 3)):
        if family in c["families"] and len(set(c["phenotypes"])) < size and not c["options"][family].get("same_type", False):
            raise ValueError(f"{FAMILIES[family]} needs at least {size} selected cell types, or repeated types enabled.")
    k = len(c["phenotypes"])
    planned = {"density": k, "proportion": k, "ratio": math.comb(k, 2), "diversity": len(metrics),
               "nn_distance": k*k if c["options"]["nn_distance"]["same_type"] else k*(k-1),
               "dyad": math.comb(k+1 if c["options"]["dyad"]["same_type"] else k, 2),
               "triad": math.comb(k+2 if c["options"]["triad"]["same_type"] else k, 3)}
    if sum(planned[f] for f in c["families"]) * (1 if c["zone_source"] == "whole" else len(c["widths"])) > 20000:
        raise ValueError("This design exceeds 20,000 comparisons. Narrow the cell types, zones or feature families before screening.")
    if c["zone_source"] not in ("distance", "polygons", "whole") or c["side"] not in ("outside", "inside", "both") or c["zone_mode"] not in ("cumulative", "rings"):
        raise ValueError("Choose valid zones and their side.")
    if c["coordinate_unit"] not in ("px", "um"):
        raise ValueError("Choose coordinate units.")
    widths = np.asarray(c["widths"], dtype=float)
    if c["zone_source"] != "whole" and (not len(widths) or len(widths) > 30 or not np.isfinite(widths).all() or (widths <= 0).any() or (np.diff(widths) <= 0).any()):
        raise ValueError("Enter up to 30 positive, increasing zone distances in µm.")
    if c["zone_source"] == "polygons" and (not c["annotation_folder"] or not c["study_class"] or not c["target_class"]):
        raise ValueError("Choose the annotation folder, sampled-region class and distance-target class.")
    density = c["options"]["density"]
    if density["area_source"] not in ("polygons", "estimate", "csv") or not 1 <= float(density["area_percentile"]) <= 100:
        raise ValueError("Choose a valid density area source and footprint percentile (1–100).")
    if "density" in c["families"] and c["zone_source"] != "polygons":
        if density["area_source"] == "polygons":
            raise ValueError("Density needs measured zone areas: choose Annotation polygons in Where to measure, or choose Cell footprint estimate / Area file in Density settings.")
        if density["area_source"] == "csv" and not c["area_table"]:
            raise ValueError("Choose an area file, or use Annotation polygons / Cell footprint estimate in Density settings.")
    for key, low, high in [("min_cells", 1, 1000000), ("min_samples", 2, 100000), ("permutations", 999, 99999), ("alpha", 0.0001, 0.5), ("pixel_um", 0, 1000)]:
        if not np.isfinite(float(c[key])) or not low <= float(c[key]) <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if c["correction"] not in ("fdr_bh", "fdr_by", "bonferroni") or c["aggregation"] not in ("median", "mean"):
        raise ValueError("Choose a valid multiplicity correction and image aggregation.")
    for family in ("dyad", "triad"):
        opt = c["options"][family]
        if not (0 <= float(opt["min_um"]) < float(opt["max_um"]) <= 1000) or opt["metric"] not in ("count", "fraction", "enrichment", "zscore") or not (0 < float(opt["pseudocount"]) <= 10):
            raise ValueError(f"Check {FAMILIES[family]} radius, measurement and pseudocount.")
    if c["group_test"] not in ("mannwhitney", "permutation"):
        raise ValueError("Choose Mann–Whitney or its permutation test for independent groups.")
    if not 99 <= int(c["motif_permutations"]) <= 99999 or not 0 <= int(c["motif_min_count"]) <= 1000000:
        raise ValueError("Use 99–99,999 random-label shuffles and a nonnegative minimum motif count.")
    nn = c["options"]["nn_distance"]
    if not 1 <= int(nn["k"]) <= 50 or nn["targets"] not in ("zone", "region"):
        raise ValueError("Choose 1–50 neighbours, in the zone or sampled region.")
    prop = c["options"]["proportion"]
    if prop["denominator"] not in ("all", "selected") or (prop["denominator"] == "selected" and not prop["labels"]):
        raise ValueError("Choose the proportion denominator cell types.")
    if prop["denominator"] == "selected" and not set(c["phenotypes"]) <= set(prop["labels"]):
        raise ValueError("Proportion targets must belong to the chosen denominator population.")
    if not 0 <= float(c["options"]["ratio"]["pseudocount"]) <= 10:
        raise ValueError("The ratio pseudocount must be 0–10.")
    return c, list(dict.fromkeys(required))


def _stamp(path):
    p = Path(path)
    st = p.stat()
    return {"path": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _table_stamp(path):
    from .tasks import source_revision
    return {**_stamp(path), "labels_and_results": source_revision({"table": str(path)})}


def output_base(root):
    from .settings import read_config
    from .steps import _folder
    from .folders import _mappings
    root = Path(root)
    return _folder(root, _mappings(root, read_config(root)), ("spatial_output_dir",), "outputs/spatial_analysis") / "feature_screening"


def prepare(root, table, chosen, images, image_entries=()):
    from spacemap import dataio
    from . import tls
    c, wanted = check(chosen, dataio.columns(table))
    from spacemap import table_collection
    if table_collection.is_collection(table):
        if c['image_col'] != 'batch_id':
            raise ValueError('Use batch_id as the Image column for separate image tables; it identifies their file assignments.')
        table_collection.validate(table, wanted, images)
    declared = units(table).get(c["distance_col"])
    canonical = {"px": "px", "pixel": "px", "pixels": "px", "um": "um", "µm": "um"}.get(str(declared))
    if c["zone_source"] == "distance" and canonical and canonical != c["distance_unit"]:
        raise ValueError(f"The table declares {c['distance_col']} in {canonical}. Use its recorded distance unit.")
    # Auto calibration is read once before submission, recorded for each image.
    if not images:
        raise ValueError("Tick the images to screen. An empty selection never means all images.")
    need_mpp = c["coordinate_unit"] == "px" or (c["zone_source"] == "distance" and c["distance_unit"] == "px") or c["zone_source"] == "polygons"
    calibration = tls.calibration(root, images, image_entries) if need_mpp and not c["pixel_um"] else {}
    mpp = {str(i): float(c["pixel_um"] or calibration.get(str(i), {}).get("mpp") or (0 if need_mpp else 1)) for i in images}
    if need_mpp and any(v <= 0 for v in mpp.values()):
        raise ValueError("Pixel calibration is missing for selected images. Enter µm per pixel in Coordinates and units.")
    from .colours import saved as saved_colours, key as colour_key
    from .settings import read_config
    colours = saved_colours(read_config(root))
    groups = [c["endpoint"]["group_a"], c["endpoint"]["group_b"]] if c["endpoint"]["type"] == "binary" else ["All"]
    palette = {g: colours.get(colour_key(g), ["#2878a5", "#c55c32"][i%2]) for i, g in enumerate(groups)}
    return {"root": str(root), "table": str(table), "values": c, "images": list(map(str, images)),
            "columns": wanted, "mpp": mpp, "source": _table_stamp(table), "palette": palette,
            "output_base": str(output_base(root))}


def _zone_index(distances, c):
    if c["zone_source"] == "whole":
        return np.zeros(len(distances), dtype=np.int64)
    d = np.asarray(distances, dtype=float).copy()
    valid = np.isfinite(d)
    if c["side"] == "inside":
        valid &= d <= 0
        d *= -1
    elif c["side"] == "outside":
        valid &= d >= 0
    else:
        d = np.abs(d)
    z = np.searchsorted(c["widths"], d, side="left").astype(np.int64)
    z[~valid | (z >= len(c["widths"]))] = -1
    return z


def _polygons(image, xy_um, mpp, c, include_geometry=False):
    """Area and cell membership use exactly the same clipped zone geometry."""
    import shapely
    from shapely.geometry import shape
    pattern = c["annotation_pattern"]
    if pattern.count("{image}") != 1:
        raise ValueError("Annotation filename must contain {image} once.")
    name = pattern.replace("{image}", image)
    if Path(name).name != name:
        raise ValueError("Annotation filenames must stay in the selected folder.")
    path = Path(c["annotation_folder"]) / name
    before = _stamp(path)
    contents = json.loads(path.read_text())
    classes = {}
    for f in contents.get("features", []):
        prop = f.get("properties") or {}
        cls = prop.get("classification") or {}
        label = cls.get("name", prop.get("name", "")) if isinstance(cls, dict) else str(cls)
        if label in (c["study_class"], c["target_class"]):
            geom = shape(f["geometry"])
            if not geom.is_valid or geom.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"{image}: invalid {label} polygon. Repair and save it before screening.")
            classes.setdefault(label, []).append(geom)
    if _stamp(path) != before:
        raise ValueError("Annotations changed while loading. Start again with the saved annotations.")
    if not classes.get(c["study_class"]) or not classes.get(c["target_class"]):
        raise ValueError(f"{image}: sampled-region or target annotation is missing; no substitute area was used.")
    study = shapely.union_all(classes[c["study_class"]])
    target = shapely.union_all(classes[c["target_class"]])
    points = shapely.points(xy_um / mpp)
    index = np.full(len(points), -1, dtype=np.int64)
    areas, geometries = [], []
    previous = shapely.GeometryCollection()
    for j, width in enumerate(c["widths"]):
        r = float(width) / mpp
        outer = target.buffer(r, quad_segs=32)
        inner = target.buffer(-r, quad_segs=32)
        band = (outer.difference(target) if c["side"] == "outside" else
                target.difference(inner) if c["side"] == "inside" else outer.difference(inner))
        cumulative = study.intersection(band)
        geometry = cumulative.difference(previous) if c["zone_mode"] == "rings" else cumulative
        mask = np.asarray(shapely.covers(cumulative, points)) & (index < 0)
        index[mask] = j
        areas.append(float(geometry.area) * mpp ** 2 / 1e6)
        geometries.append(geometry)
        previous = cumulative
    in_study = np.asarray(shapely.covers(study, points))
    result = (index, np.asarray(areas), in_study, before)
    return (*result, geometries) if include_geometry else result


def _measure(part, c, mpp, report):
    from . import screening_graph as graph
    image = str(part["_image"].iloc[0])
    xy = part[[c["x_col"], c["y_col"]]].to_numpy(float)
    if c["coordinate_unit"] == "px":
        xy *= mpp
    area = None
    geometry_stamp = None
    if c["zone_source"] == "polygons":
        zones, area, study, geometry_stamp = _polygons(image, xy, mpp, c)
        xy, part, zones = xy[study], part.iloc[np.flatnonzero(study)], zones[study]
    else:
        d = np.zeros(len(part)) if c["zone_source"] == "whole" else pd.to_numeric(part[c["distance_col"]], errors="coerce").to_numpy(float)
        if c["zone_source"] == "distance" and c["distance_unit"] == "px":
            d *= mpp
        zones = _zone_index(d, c)
    nz = 1 if c["zone_source"] == "whole" else len(c["widths"])
    rings = c["zone_mode"] == "rings"
    # Retain missing labels as an explicit background population. They count
    # toward all-cell proportions but not diversity of named cell types.
    labels = part[c["phenotype_col"]].astype("string").fillna("[Unassigned]").to_numpy(str)
    names = sorted(set(labels) | set(c["phenotypes"]) | set(c["options"]["proportion"]["labels"]))
    if len(names) > 50 and set(c["families"]) & {"dyad", "triad"}:
        raise ValueError("Motif screening supports up to 50 cell types. Choose a coarser cell-type column.")
    encoded = np.searchsorted(names, labels).astype(np.int64)
    lookup = {s: i for i, s in enumerate(names)}
    target_ids = [lookup[s] for s in c["phenotypes"]]
    zone_masks = [(zones == j) if rings else ((zones >= 0) & (zones <= j)) for j in range(nz)]
    counts = [np.bincount(encoded[mask], minlength=len(names)) for mask in zone_masks]
    out, audit, graphs, trees, nulls = [], [], {}, {}, {}
    if "density" in c["families"] and area is None and c["options"]["density"]["area_source"] == "estimate":
        from spacemap_gui.tabs.tab8_analysis.feature_screening_core import _compute_virtual_area_mm2
        report(f"{image}: estimating occupied area (Classic cell footprint)", None)
        # Coordinates are already µm; calibration=1 avoids applying pixel size twice.
        area = np.array([_compute_virtual_area_mm2(xy[mask], 1., 0., False,
                         c["options"]["density"]["area_percentile"])[0] for mask in zone_masks])
        audit.append({"image": image, "family": "density", "area_method": "Classic cell footprint estimate"})
    for family in c["families"]:
        opt = c["options"][family]
        if family in ("dyad", "triad"):
            key = (float(opt["min_um"]), float(opt["max_um"]))
            active = zones >= 0
            if key not in graphs:
                report(f"{image}: building {key[1]:g} µm neighbour graph", None)
                graphs[key] = graph.radius_edges(xy[active], *key)
            start = time.perf_counter()
            report(f"{image}: counting {FAMILIES[family].lower()} across {nz} zones", None)
            motif_counts, totals = graph.motif_counts(graphs[key], encoded[active], zones[active], len(names), nz,
                                                    triangles=family == "triad", rings=rings)
            audit.append({"image": image, "family": family, "cells": len(xy), "edges": len(graphs[key]),
                          "motifs_in_largest_zone": int(totals[-1]), "seconds": time.perf_counter() - start})
            definitions = graph.definitions(target_ids, 3 if family == "triad" else 2, opt["same_type"])
        for j, (mask, cnt) in enumerate(zip(zone_masks, counts)):
            n = int(mask.sum())
            base = dict(family=family, zone=j, cells=n, area_mm2=float(area[j]) if area is not None else np.nan)
            scores = {}
            if family in ("dyad", "triad") and opt["metric"] == "zscore" and n >= c["min_cells"]:
                cache_key = (*key, j)
                if cache_key not in nulls:
                    from spacemap_gui.tabs.tab8_analysis.feature_screening_core import _permutation_zscore_dyad_triad
                    # Both families reuse the same shuffles when their radii agree.
                    needs_triangles = any(f == "triad" and c["options"][f]["metric"] == "zscore" and
                                          (float(c["options"][f]["min_um"]), float(c["options"][f]["max_um"])) == key
                                          for f in c["families"])
                    selected = np.flatnonzero(mask[active])
                    mapping = np.full(int(active.sum()), -1, np.int64)
                    mapping[selected] = np.arange(len(selected))
                    edges = graphs[key]
                    edges = mapping[edges[(mapping[edges] >= 0).all(axis=1)]].astype(np.int32)
                    triangles = graph.triangle_array(edges, n) if needs_triangles else np.empty((0, 3), np.int32)
                    identity = [str(part[k].iloc[0]) for k in ("_sample", "_group", "_image", "_region")]
                    seed = (int(c["seed"]) + int.from_bytes(hashlib.sha256(json.dumps([identity, cache_key]).encode()).digest()[:4], "little")) % (2**32)
                    timing = dict(image=image, zone=j, family="random-label null", seed=seed, edges=len(edges), triangles=len(triangles))
                    start = time.perf_counter()
                    report(f"{image}: zone {j+1}/{nz} · {c['motif_permutations']} random-label shuffles", None)
                    nulls[cache_key] = _permutation_zscore_dyad_triad(labels[mask], edges, triangles,
                        int(c["motif_permutations"]), np.random.default_rng(seed),
                        min_obs_count=int(c["motif_min_count"]), use_gpu=bool(c["motif_use_gpu"]), diagnostics=timing)
                    timing["seconds"] = time.perf_counter() - start
                    audit.append(timing)
                scores = nulls[cache_key][int(family == "triad")]
            def add(parts, value, numerator=np.nan, denominator=np.nan, reason=""):
                out.append({**base, "feature": json.dumps([family, *parts]), "label": " / ".join(parts) or FAMILIES[family],
                            "value": value if n >= c["min_cells"] else np.nan, "numerator": numerator, "denominator": denominator,
                            "reason": "too few cells in zone" if n < c["min_cells"] else reason})
            if family == "density":
                for name in c["phenotypes"]:
                    a = base["area_mm2"]
                    add([name], cnt[lookup[name]] / a if a > 0 else np.nan, int(cnt[lookup[name]]), a)
            elif family == "proportion":
                den = n if opt["denominator"] == "all" else sum(cnt[lookup[s]] for s in opt["labels"])
                for name in c["phenotypes"]:
                    add([name], 100 * cnt[lookup[name]] / den if den else np.nan, int(cnt[lookup[name]]), den, "zero denominator" if not den else "")
            elif family == "ratio":
                for a, b in combinations(c["phenotypes"], 2):
                    num, den = cnt[lookup[a]] + opt["pseudocount"], cnt[lookup[b]] + opt["pseudocount"]
                    v = num / den if den else np.nan
                    if opt["log2"]:
                        v = np.log2(v) if v > 0 else np.nan
                    add([a, b], v, int(cnt[lookup[a]]), int(cnt[lookup[b]]), "undefined ratio" if not np.isfinite(v) else "")
            elif family == "diversity":
                selected = cnt[target_ids].astype(float)
                p = selected[selected > 0] / selected.sum() if selected.sum() else np.array([])
                richness = len(p)
                h = float(-np.sum(p * np.log(p)))
                values = {"shannon": h, "simpson": 1 - float(np.sum(p*p)), "richness": richness,
                          "evenness": h / np.log(richness) if richness > 1 else np.nan}
                for metric in diversity_metrics(c):
                    reason = "no selected types" if not richness else "evenness needs at least two types" if metric == "evenness" and richness < 2 else ""
                    add([metric], values[metric] if richness else np.nan, int(selected.sum()), n, reason)
                    out[-1]["label"] = DIVERSITY[metric]
            elif family == "nn_distance":
                for a, b in permutations(c["phenotypes"], 2) if not opt["same_type"] else [(a,b) for a in c["phenotypes"] for b in c["phenotypes"]]:
                    source = xy[mask & (labels == a)]
                    target = xy[(mask if opt["targets"] == "zone" else np.ones(len(mask), bool)) & (labels == b)]
                    k = int(opt["k"]) + int(a == b)
                    val = np.nan
                    if len(source) and len(target) >= k:
                        tree_key = (j if opt["targets"] == "zone" else -1, b)
                        if tree_key not in trees:
                            trees[tree_key] = cKDTree(target)
                        distances = trees[tree_key].query(source, k=list(range(1 + int(a == b), k + 1)))[0]
                        val = float(distances.mean())
                    add([a, b], val, len(source), len(target), "insufficient source/target cells" if not np.isfinite(val) else "")
            else:
                for motif in definitions:
                    code = 0
                    for label in sorted(motif):
                        code = code * len(names) + label
                    obs, total = int(motif_counts[j, code]), int(totals[j])
                    expected = total * graph.label_probability(cnt, motif)
                    if opt["metric"] == "zscore":
                        val = scores.get(tuple(sorted(names[i] for i in motif)), np.nan)
                    elif opt["metric"] == "count":
                        val = obs
                    elif opt["metric"] == "fraction":
                        val = 100 * obs / total if total else np.nan
                    else:
                        val = np.log2((obs + opt["pseudocount"]) / (expected + opt["pseudocount"])) if total and expected > 0 else np.nan
                    reason = "too few observed motifs or no variation under random labels" if opt["metric"] == "zscore" else "no motifs or absent types"
                    add([names[i] for i in motif], val, obs, total, reason if not np.isfinite(val) else "")
                    out[-1]["expected"] = expected
    return out, audit, geometry_stamp


def _sample_outcomes(frame, c):
    e = c["endpoint"]
    cols = ([e["group_col"]] if e["type"] == "binary" else [e["value_col"]] if e["type"] == "continuous" else
            [e["time_col"], e["event_col"]] if e["type"] == "survival" else [])
    frame["_sample"] = frame[c["sample_col"]].astype("string")
    frame["_group"] = frame[e["group_col"]].astype("string") if e["type"] == "binary" else "All"
    if e["type"] == "binary":
        frame = frame[frame["_group"].isin([e["group_a"], e["group_b"]])].copy()
        if not e["paired"] and (frame.groupby("_sample")["_group"].nunique() > 1).any():
            raise ValueError("A sample belongs to both comparison groups. Choose a paired comparison or filter to one observation per sample.")
    keys = ["_sample", "_group"]
    if cols and (frame.groupby(keys)[cols].nunique(dropna=False).max().max() > 1):
        raise ValueError("Outcome values vary within a sample. Filter to the intended observation or choose the correct sample column.")
    outcomes = frame.groupby(keys, dropna=False)[cols].first().reset_index() if cols else frame[keys].drop_duplicates()
    return frame, outcomes


def infer(samples, outcomes, c, publish=None):
    e = c["endpoint"]
    merged = samples.merge(outcomes, on=["_sample", "_group"], validate="many_to_one")
    rows = []
    rng = np.random.default_rng(c["seed"])
    previous_family = None
    for (family, feature, zone, label), part in merged.groupby(["family", "feature", "zone", "label"], sort=True):
        if publish and previous_family is not None and previous_family != family:
            partial = pd.DataFrame(rows)
            partial["q"] = np.nan
            publish(partial, merged, f"Compared {FAMILIES[previous_family]}; adjustment awaits the full scan")
        previous_family = family
        valid = part[np.isfinite(part["value"])].copy()
        row = dict(family=family, feature=feature, zone=int(zone), label=label, n_samples=valid["_sample"].nunique(),
                   n_a=0, n_b=0, median_a=np.nan, median_b=np.nan, effect=np.nan, p=np.nan, status="descriptive", test="none")
        if e["type"] == "binary":
            a = valid.loc[valid["_group"] == e["group_a"], "value"].to_numpy()
            b = valid.loc[valid["_group"] == e["group_b"], "value"].to_numpy()
            if e["paired"]:
                pivot = valid.pivot(index="_sample", columns="_group", values="value").reindex(columns=[e["group_a"], e["group_b"]]).dropna()
                a, b = pivot[e["group_a"]].to_numpy(), pivot[e["group_b"]].to_numpy()
            row.update(n_a=len(a), n_b=len(b), n_samples=len(a) if e["paired"] else len(a)+len(b), median_a=float(np.median(a)) if len(a) else np.nan,
                       median_b=float(np.median(b)) if len(b) else np.nan)
            row["effect"] = float(np.median(b - a)) if e["paired"] and len(a) else row["median_b"] - row["median_a"]
            if min(len(a), len(b)) < c["min_samples"]:
                row["status"] = "too few independent samples"
            elif np.ptp(np.r_[a, b]) == 0:
                row.update(p=1., status="constant", test="constant")
            elif e["paired"]:
                # Paired sign-flip test of the mean difference; labels exchange
                # only within the same biological unit. Exhaustive when small.
                result = stats.permutation_test((b - a,), lambda x, axis: np.mean(x, axis=axis), vectorized=True,
                    permutation_type="samples", n_resamples=int(c["permutations"]), random_state=rng, batch=256)
                row.update(p=float(result.pvalue), status="tested", test="paired permutation (mean difference)")
            elif c["group_test"] == "mannwhitney":
                # Same two-sided auto method as Classic's _screen_binary_feature.
                test = stats.mannwhitneyu(a, b, alternative="two-sided")
                row.update(p=float(test.pvalue), status="tested", test="Mann–Whitney (Classic)")
            else:
                # Permutation MW handles ties, zeros and small cohorts without
                # treating thousands of cells as independent replicates.
                # Rank once. Permuting the fixed pooled ranks gives exactly the
                # same rank-sum null, including ties, on supported SciPy versions.
                rank = stats.rankdata(np.r_[b, a])
                result = stats.permutation_test((rank[:len(b)], rank[len(b):]),
                    lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis), vectorized=True,
                    n_resamples=int(c["permutations"]), random_state=rng, batch=256)
                row.update(p=float(result.pvalue), status="tested", test="Mann–Whitney permutation")
        elif e["type"] == "continuous":
            y = pd.to_numeric(valid[e["value_col"]], errors="coerce")
            use = np.isfinite(y)
            x, y = valid.loc[use, "value"].to_numpy(), y[use].to_numpy()
            row["n_samples"] = len(x)
            if len(x) < c["min_samples"] or np.ptp(y) == 0 or np.ptp(x) == 0:
                row["status"] = "insufficient or constant values"
            else:
                rx, ry = stats.rankdata(x), stats.rankdata(y)
                rx, ry = rx - rx.mean(), ry - ry.mean()
                den = np.linalg.norm(rx) * np.linalg.norm(ry)
                observed = float(rx @ ry / den)
                hits = 0
                for _ in range(int(c["permutations"])):
                    hits += abs(float(rng.permutation(rx) @ ry / den)) >= abs(observed) - 1e-12
                row.update(effect=observed, p=(hits + 1) / (c["permutations"] + 1), status="tested", test="Spearman permutation")
        elif e["type"] == "survival":
            from lifelines import CoxPHFitter
            from lifelines.exceptions import ConvergenceError
            from lifelines.statistics import proportional_hazard_test
            fit = pd.DataFrame({"time": pd.to_numeric(valid[e["time_col"]], errors="coerce"),
                                "event": pd.to_numeric(valid[e["event_col"]], errors="coerce"), "feature": valid["value"]}).dropna()
            row["n_samples"] = len(fit)
            if not fit["event"].isin([0, 1]).all() or (fit["time"] <= 0).any():
                raise ValueError("Survival time must be positive and the event column must contain 0/1.")
            if e["censored"]:
                fit["event"] = 1 - fit["event"]
            if len(fit) < max(10, c["min_samples"]) or fit["event"].sum() < 5 or fit["feature"].std() == 0:
                row["status"] = "too few samples/events or constant values"
            else:
                try:
                    import warnings
                    fit["feature"] = (fit["feature"] - fit["feature"].mean()) / fit["feature"].std()
                    with warnings.catch_warnings(record=True) as caught:
                        model = CoxPHFitter().fit(fit, "time", "event")
                    if caught:
                        row["status"] = "Cox fit warning; inspect before inference"
                    else:
                        row.update(effect=float(model.hazard_ratios_["feature"]), p=float(model.summary.loc["feature", "p"]),
                                   ph_p=float(proportional_hazard_test(model, fit).summary.loc["feature", "p"]),
                                   status="tested", test="Cox HR per SD")
                except (ValueError, ArithmeticError, ConvergenceError) as exc:
                    row["status"] = "Cox fit failed: " + str(exc)[:100]
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No features could be measured. Check cell types, zones and filters.")
    from .screening_statistics import adjust
    result = adjust(result, c)
    return result.sort_values(["q", "p", "family", "label", "zone"], na_position="last"), merged


def compute(frame, c, mpp, report=lambda *args: None, publish=None, measurement_cache=None):
    c, _ = check(c, frame.columns)
    audit = {"input_cells": len(frame), "warnings": [], "timings": [], "annotations": []}
    if c["sample_col"] == c["image_col"]:
        audit["warnings"].append("Images were used as independent samples. Valid only if each image comes from a different independent biological sample.")
    if "density" in c["families"] and c["zone_source"] != "polygons" and c["options"]["density"]["area_source"] == "estimate":
        audit["warnings"].append("Density uses Classic's cell-footprint area estimate. Sparse cells, cell loss and zone boundaries can bias it; use measured annotation areas for tissue density.")
    if frame[c["sample_col"]].isna().any() or frame[c["image_col"]].isna().any():
        raise ValueError("Every analysed row needs a sample and image ID; fill or filter missing IDs first.")
    # Validate outcomes BEFORE region filters: a filter must not hide contradictory
    # sample metadata. Outcome rows retain samples with no eligible zone.
    frame, outcomes = _sample_outcomes(frame.copy(), c)
    frame["_image"] = frame[c["image_col"]].astype(str)
    for rule in c["filters"]:
        frame = frame[frame[rule["column"]].astype("string").isin(list(map(str, rule["values"])))].copy()
    for col in [c["x_col"], c["y_col"]]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid_xy = np.isfinite(frame[[c["x_col"], c["y_col"]]].to_numpy(float)).all(axis=1)
    audit["invalid_coordinates"] = int((~valid_xy).sum())
    frame = frame[valid_xy].copy()
    frame["_region"] = frame[c["region_col"]].astype("string") if c["region_col"] else "All"
    if c["region_col"]:
        audit["missing_region"] = int(frame["_region"].isna().sum())
        frame = frame[frame["_region"].notna()].copy()
    if frame.empty:
        raise NoEligibleCells("No cells remain after the region and coordinate checks.")
    audit["eligible_cells"] = len(frame)
    if c["zone_source"] == "polygons" and (c["region_col"] or (frame.groupby("_image")[["_sample", "_group"]].nunique() > 1).any().any()):
        raise ValueError("Polygon areas require one sample/group per image and no region-ID split. For shared images or separate pieces, use a zone-area CSV with distance-column zones.")
    keys = ["_sample", "_group", "_image", "_region"]
    change_col = c["endpoint"].get("change_col")
    if change_col:
        frame["_occasion"] = frame[change_col].astype("string")
        frame = frame[frame._occasion.isin([c["endpoint"]["change_before"], c["endpoint"]["change_after"]])].copy()
        if frame.empty:
            raise NoEligibleCells("No cells remain in the selected before/after observations.")
        if c["zone_source"] == "polygons" and (frame.groupby("_image")._occasion.nunique() > 1).any():
            raise ValueError("Polygon areas require one observation per image. Assign distinct image IDs to different visits.")
        keys.append("_occasion")
    groups = frame.groupby(keys, sort=True, observed=True)
    rows = []
    area_lookup = {}
    area_without_group = {}
    if "density" in c["families"] and c["options"]["density"]["area_source"] == "csv" and c["zone_source"] != "polygons":
        area_path = Path(c["area_table"])
        audit["area_source"] = _stamp(area_path)
        area_frame = pd.read_csv(area_path, dtype={"sample": str, "group": str, "image": str, "region": str})
        area_keys = ["sample", "group", "image", "region", "zone"]
        if not set(area_keys + ["area_mm2"]) <= set(area_frame) or area_frame.duplicated(area_keys).any():
            raise ValueError("Area CSV needs unique sample, group, image, region, zone (0-based) and area_mm2 columns.")
        area_lookup = area_frame.set_index(area_keys).area_mm2.to_dict()
        # A physical area does not change when the outcome question changes.
        # Reuse only unambiguous measured areas; conflicting group-specific
        # areas must still be supplied for the requested population.
        for area_key, values in area_frame.groupby(["sample", "image", "region", "zone"]):
            if values.area_mm2.nunique(dropna=False) == 1:
                area_without_group[area_key] = values.area_mm2.iloc[0]
    last_preview = 0.
    for i, (key, part) in enumerate(groups):
        report(f"{key[2]}: measuring image {i+1}/{len(groups)}", 0.05 + 0.65 * i / len(groups))
        if measurement_cache is None:
            measured, timing, annotation = _measure(part, c, float(mpp.get(str(key[2]), 1)), report)
        else:
            measured, timing, annotation = measurement_cache(part, c, float(mpp.get(str(key[2]), 1)), report)
        for r in measured:
            r.update(zip(keys, key))
            if r["family"] == "density" and area_lookup:
                fallback = area_without_group.get((key[0], key[2], key[3], r["zone"]), np.nan)
                a = float(area_lookup.get((*key[:4], r["zone"]), fallback))
                r["area_mm2"] = r["denominator"] = a
                r["value"] = r["numerator"] / a if a > 0 and r["cells"] >= c["min_cells"] else np.nan
        rows.extend(measured)
        audit["timings"].extend(timing)
        if annotation:
            audit["annotations"].append(annotation)
        if publish and not change_col and (i == 0 or time.monotonic() - last_preview > 5) and i < len(groups)-1:
            provisional = pd.DataFrame(rows)
            sample_keys = ["_sample", "_group", "family", "feature", "label", "zone"]
            preview_samples = provisional.groupby(sample_keys, as_index=False).agg(value=("value", c["aggregation"]))
            # No inference on an incomplete cohort. Density waits for its areas
            # to be joined; those rows explicitly remain unmeasured for now.
            descriptive = deepcopy(c)
            descriptive["endpoint"]["type"] = "describe"
            preview_stats, merged = infer(preview_samples, outcomes, descriptive)
            publish(preview_stats, merged, provisional, audit, f"Measuring images: {i+1}/{len(groups)} · descriptive preview")
            last_preview = time.monotonic()
    regions = pd.DataFrame(rows)
    if regions.empty:
        raise ValueError("The selected feature families need more cell types (pairs need two, triplets three, unless repeated types are enabled).")
    if "density" in c["families"]:
        den = regions["family"] == "density"
        a = pd.to_numeric(regions["area_mm2"], errors="coerce")
        if (den & ((a < 0) | ~np.isfinite(a))).any() and c["options"]["density"]["area_source"] != "estimate":
            raise ValueError("A density zone has no valid measured area. Supply all zone areas; zero area is unmeasured, not zero density.")
        ok = den & (a > 0) & (regions["cells"] >= c["min_cells"])
        regions.loc[ok, "value"] = regions.loc[ok, "numerator"] / a[ok]
        regions.loc[den, "denominator"] = a[den]
        regions.loc[den & (a == 0), "reason"] = "zero sampled area"
    # Each image/piece contributes once; repeated images never become subjects.
    sample_keys = ["_sample", "_group", "family", "feature", "label", "zone"]
    samples = regions.groupby(sample_keys + (["_occasion"] if change_col else []), as_index=False).agg(value=("value", c["aggregation"]),
                         n_regions=("value", "count"), cells=("cells", "sum"))
    if change_col:
        visits = samples.pivot(index=sample_keys, columns="_occasion", values="value").reindex(columns=[c["endpoint"]["change_before"], c["endpoint"]["change_after"]])
        samples = (visits.iloc[:, 1] - visits.iloc[:, 0]).rename("value").reset_index()
        totals = regions.groupby(sample_keys, as_index=False).agg(n_regions=("value", "count"), cells=("cells", "sum"))
        samples = samples.merge(totals, on=sample_keys)
        audit["warnings"].append(f"Comparing within-sample changes: {change_col} {c['endpoint']['change_after']} minus {c['endpoint']['change_before']}. Both observations are required.")
    all_features = regions[["family", "feature", "label", "zone"]].drop_duplicates()
    grid = outcomes[["_sample", "_group"]].merge(all_features, how="cross")
    samples = grid.merge(samples, on=sample_keys, how="left", validate="one_to_one")
    audit["missing_sample_features"] = int(samples["value"].isna().sum())
    audit["independent_samples"] = int(outcomes["_sample"].nunique())
    audit["images"] = int(frame["_image"].nunique())
    audit["sample_groups"] = {str(k): int(v) for k, v in outcomes.groupby("_group")["_sample"].nunique().items()}
    report("Comparing independent samples; correcting the complete search", 0.75)
    callback = (lambda results, merged, message: publish(results, merged, regions, audit, message)) if publish else None
    results, merged = infer(samples, outcomes, c, callback)
    if c["resampling"] != "off" and c["endpoint"]["type"] != "describe":
        from .screening_statistics import resample
        if publish:
            publish(results, merged, regions, audit, "Comparisons finished; checking sample stability")
        results = resample(results, merged, c, infer, report)
        audit["resampling"] = {"method": c["resampling"], "iterations": c["resamples"],
            "unit": c["sample_col"], "fraction": c["sample_fraction"] if c["resampling"] == "subsample" else 1,
            "nearby_zones": c["nearby_zones"], "same_direction_required": True}
    return results, samples, regions, audit, merged


def job(settings, folder, report):
    from spacemap import dataio
    from . import screening_figures
    start = time.perf_counter()
    c = settings["values"]
    if _table_stamp(settings["table"]) != settings["source"]:
        raise ValueError("Cell data changed after the scan was submitted. Start a new scan.")
    out = Path(settings.get("output_base") or output_base(settings["root"])) / (datetime.now(timezone.utc).strftime("studio_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    (out / "tables").mkdir(parents=True)
    index = Path(settings["root"]) / "outputs/studio/screening_runs"
    index.mkdir(parents=True, exist_ok=True)
    _atomic_json(index / (out.name + ".json"), {"folder": str(out.resolve())})
    if folder:
        _atomic_json(Path(folder) / "screening_progress.json", {"run": out.name})
    def preview(results, merged, regions, audit, message):
        if _table_stamp(settings["table"]) != settings["source"]:
            raise ValueError("Cell data changed during screening. Start a new scan.")
        payload = _display_result(settings, out, results, merged, regions, audit, [], partial=True, message=message)
        _atomic_json(out / "design.json", {"settings": c, "images": settings["images"], "mpp": settings["mpp"], "audit": audit})
        _atomic_json(out / "result.json", payload)
        report(message, None)
    frame = dataio.load(settings["table"], columns=settings["columns"])
    frame = frame[frame[c["image_col"]].astype(str).isin(settings["images"])].copy()
    from . import screening_comparisons
    if screening_comparisons.enabled(c):
        return screening_comparisons.run(settings, out, frame, report)
    results, samples, regions, audit, merged = compute(frame, c, settings["mpp"], report, preview)
    source_stamps = [*audit["annotations"], *([audit["area_source"]] if audit.get("area_source") else [])]
    if _table_stamp(settings["table"]) != settings["source"] or any(_stamp(s["path"]) != s for s in source_stamps):
        raise ValueError("Inputs changed during screening. Results were not published; rerun on the saved inputs.")
    results.to_csv(out / "tables/screening_results_all.csv", index=False)
    results.loc[results.q < c["alpha"]].to_csv(out / "tables/significant_features.csv", index=False)
    merged.to_csv(out / "tables/sample_features.csv", index=False)
    regions.to_csv(out / "tables/region_features.csv", index=False)
    report("Drawing feature summaries and sample comparisons", 0.9)
    figures = screening_figures.save(out, results, merged, c, settings.get("palette"))
    audit["seconds"] = time.perf_counter() - start
    audit["hypotheses"] = len(results)
    audit["source"] = settings["source"]
    (out / "design.json").write_text(json.dumps({"settings": c, "images": settings["images"], "mpp": settings["mpp"], "audit": audit}, indent=2))
    result = _display_result(settings, out, results, merged, regions, audit, figures)
    _atomic_json(out / "result.json", result)
    report("Finished. Tables and figures are ready.", 1.)
    return result


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _display_result(settings, out, results, merged, regions, audit, figures, partial=False, message=""):
    c = settings["values"]
    def clean(records):
        return json.loads(pd.DataFrame(records).to_json(orient="records"))
    hits = []
    by_feature = dict(tuple(regions.groupby(["feature", "zone"], sort=False)))
    observed = dict(tuple(merged.groupby(["feature", "zone"], sort=False)))
    # Keep each chosen family represented in the preview and ranked table.
    # Diversity's four scales all need representation, even if one metric fills
    # the first thirty ranks. Each selected metric has at most thirty zones.
    display = pd.concat([results[results.family != "diversity"].groupby("family", sort=False).head(30),
                         results[results.family == "diversity"]])
    for row in display.to_dict("records"):
        key = (row["feature"], row["zone"])
        part = by_feature[key].dropna(subset=["value"])
        observations = observed[key].dropna(subset=["value"])
        e = c["endpoint"]
        if e["type"] == "binary" and e["paired"]:
            paired = observations.groupby("_sample")._group.nunique()
            observations = observations[observations._sample.isin(paired[paired == 2].index)]
        elif e["type"] in ("continuous", "survival"):
            columns = [e["value_col"]] if e["type"] == "continuous" else [e["time_col"], e["event_col"]]
            valid = np.isfinite(observations[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
            observations = observations.loc[valid]
        part = part.merge(observations[["_sample", "_group"]], on=["_sample", "_group"], validate="many_to_one")
        image_values = part.groupby("_image")["value"].median().sort_values()
        row["highest"] = [[str(k), float(v)] for k, v in image_values.tail(5).iloc[::-1].items()]
        row["lowest"] = [[str(k), float(v)] for k, v in image_values.head(5).items()]
        # A change belongs to a subject's two visits, not one representative image.
        if e.get("change_col"):
            row["highest"], row["lowest"] = [], []
        row["phenotypes"] = c["phenotypes"] if row["family"] == "diversity" else [n for n in json.loads(row["feature"])[1:] if n in c["phenotypes"]]
        images_by_sample = dict(tuple(part.groupby(["_sample", "_group"])))
        row["observations"] = []
        for r in observations.to_dict("records"):
            images = images_by_sample[(r["_sample"], r["_group"])].groupby("_image").value.agg(c["aggregation"])
            # The first example is closest to the displayed sample value, rather
            # than whichever filename happens to sort first.
            order = (images - r["value"]).abs().sort_values(kind="stable").index.tolist()
            point = {"sample": str(r["_sample"]), "group": str(r["_group"]), "value": float(r["value"]), "images": order}
            if e.get("change_col"):
                part_sample = images_by_sample[(r["_sample"], r["_group"])]
                point["visits"] = []
                for visit in [e["change_before"], e["change_after"]]:
                    values = part_sample[part_sample._occasion.astype(str) == visit].groupby("_image").value.agg(c["aggregation"])
                    value = part_sample.loc[part_sample._occasion.astype(str) == visit, "value"].agg(c["aggregation"])
                    ordered = (values-value).abs().sort_values(kind="stable").index.tolist()
                    point["visits"].append({"name": visit, "value": float(value), "images": ordered})
                point["images"] = [image for visit in point["visits"] for image in visit["images"]]
            row["observations"].append(point)
        hits.append(row)
    return {"version": 2, "run": out.name, "folder": str(out), "table": str(settings["table"]), "source": settings["source"],
              "settings": c, "mpp": settings["mpp"], "hits": clean(hits), "tests": clean(results), "figures": figures,
              "tested": int(results.p.notna().sum()), "hypotheses": len(results), "significant": int((results.q < c["alpha"]).sum()),
              "samples": audit.get("independent_samples", int(merged._sample.nunique())), "cells": audit["eligible_cells"],
              "seconds": audit.get("seconds"), "partial": partial, "message": message,
              "warnings": audit.get("warnings", []), "sample_groups": audit.get("sample_groups", {}), "images": audit.get("images"),
              "revision": str(time.time_ns()), "palette": settings.get("palette", {})}


def runs(root, table):
    bases = {Path(root) / "outputs/spatial_analysis/feature_screening", output_base(root)}
    files = {p for base in bases for p in base.glob("*/result.json")}
    for receipt in (Path(root) / "outputs/studio/screening_runs").glob("*.json"):
        path = Path(json.loads(receipt.read_text())["folder"]) / "result.json"
        if path.exists():
            files.add(path)
    found = []
    seen = set()
    for file in sorted(files, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        result = json.loads(file.read_text())
        if result.get("table") == str(table) and result["run"] not in seen:
            seen.add(result["run"])
            found.append({k: result[k] for k in ("run", "tested", "significant", "samples", "seconds")})
    return found


def result(root, run, comparison=""):
    if Path(run).name != run or not run:
        raise ValueError("Choose a saved screening run.")
    folder = Path(root) / "outputs/spatial_analysis/feature_screening" / run
    receipt = Path(root) / "outputs/studio/screening_runs" / (run + ".json")
    if receipt.exists():
        folder = Path(json.loads(receipt.read_text())["folder"])
    saved = json.loads((folder / "result.json").read_text())
    if comparison and saved.get("comparisons"):
        item = next((p for p in saved["comparisons"] if p["id"] == comparison), None)
        if item is None or item["status"] in ("skipped", "ready"):
            reason = (item.get("reason") or "This comparison has not been measured yet.") if item else "Choose a comparison from this run."
            raise ValueError(reason)
        selected = json.loads((folder / comparison / "result.json").read_text())
        saved = dict(selected, comparisons=saved["comparisons"], scan_settings=saved["scan_settings"], scan_folder=str(folder),
                     revision=str(selected["revision"])+":"+str(saved["revision"]))
    # Older saved runs contain only a shortlist. Expose their complete original
    # tests without recalculating q or rewriting any recorded result.
    if not saved.get("partial") and "tests" not in saved:
        saved["tests"] = json.loads(pd.read_csv(Path(saved["folder"]) / "tables/screening_results_all.csv").to_json(orient="records"))
    return saved


def finding(root, run, feature, zone, comparison=""):
    """Load sample dots only for an explicitly opened finding, including old runs."""
    saved = result(root, run, comparison)
    cached = next((r for r in saved["hits"] if r["feature"] == feature and r["zone"] == zone), None)
    if cached:
        return cached
    selected = [r for r in saved.get("tests", []) if r["feature"] == feature and r["zone"] == zone]
    if not selected or saved.get("partial"):
        raise ValueError("Choose a completed finding from this saved run.")
    tables = Path(saved["folder"]) / "tables"
    # Chunked reads bound memory for large saved scans; no source cell-table scan.
    def read(name):
        parts = []
        for part in pd.read_csv(tables / name, chunksize=50000, dtype={"_sample": str, "_group": str, "_image": str, "_occasion": str}):
            parts.append(part[(part.feature == feature) & (part.zone == zone)])
        return pd.concat(parts, ignore_index=True)
    samples, regions = read("sample_features.csv"), read("region_features.csv")
    c = saved["settings"]
    # Legacy numeric/survival exports omit outcome columns. Eligibility was
    # already tested; do not invent outcome values while drawing their samples.
    e = c["endpoint"]
    extra = [e["value_col"]] if e["type"] == "continuous" else [e["time_col"], e["event_col"]] if e["type"] == "survival" else []
    if any(col not in samples for col in extra):
        raise ValueError("This old run did not save outcomes needed for interactive review. Its full tests remain downloadable; run again for linked plots.")
    payload = _display_result({"values": c, "table": saved["table"], "source": saved["source"], "mpp": saved["mpp"]},
        Path(saved["folder"]), pd.DataFrame(selected), samples, regions, {"eligible_cells": saved["cells"]}, [])
    return payload["hits"][0]


def context_cells(root, table, run, image, feature, zone, channel=None, comparison="", sample="", group="", occasion=""):
    """Explicit image review, bound to the run's source and exact selected zone."""
    from spacemap import dataio
    saved = result(root, run, comparison)
    if saved["source"] != _table_stamp(table):
        raise ValueError("Cell data changed since this scan. Run a new scan before linking its cells.")
    c = saved["settings"]
    if not any(r["feature"] == feature and r["zone"] == zone for r in saved.get("tests", saved["hits"])):
        raise ValueError("Choose a feature from this saved run.")
    _, names = check(c, dataio.columns(table))
    if "label" not in dataio.columns(table):
        raise ValueError("Image linking needs the segmentation label column; the exported measurements are still available.")
    from spacemap import table_collection
    link_table = table
    if table_collection.is_collection(table):
        link_table = table_collection.source_for_image(table, image)
        frame = table_collection.load(table, columns=list(dict.fromkeys([*names, 'label'])), images=[image])
    else:
        frame = dataio.load(table, columns=list(dict.fromkeys([*names, "label"])))
    frame = frame.reset_index(drop=True)
    frame = frame[frame[c["image_col"]].astype(str) == image].copy()
    for rule in c["filters"]:
        frame = frame[frame[rule["column"]].astype("string").isin(rule["values"])]
    if c["endpoint"]["type"] == "binary":
        e = c["endpoint"]
        frame = frame[frame[e["group_col"]].astype("string").isin([e["group_a"], e["group_b"]])]
        if e.get("change_col"):
            frame = frame[frame[e["change_col"]].astype("string").isin([e["change_before"], e["change_after"]])]
    if c["region_col"]:
        frame = frame[frame[c["region_col"]].notna()]
    if sample:
        frame = frame[frame[c["sample_col"]].astype("string") == sample]
    if group and c["endpoint"]["type"] == "binary":
        frame = frame[frame[c["endpoint"]["group_col"]].astype("string") == group]
    if occasion:
        column = c["endpoint"].get("change_col")
        if not column or occasion not in (c["endpoint"]["change_before"], c["endpoint"]["change_after"]):
            raise ValueError("Choose a before/after observation from this comparison.")
        frame = frame[frame[column].astype("string") == occasion]
    mpp = saved["mpp"].get(image)
    if mpp is None:
        raise ValueError("This image was not included in the scan.")
    xy = frame[[c["x_col"], c["y_col"]]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(xy).all(axis=1)
    frame, xy = frame.iloc[np.flatnonzero(valid)], xy[valid]
    geometry = None
    if c["zone_source"] == "polygons":
        if c["coordinate_unit"] == "px":
            xy *= mpp
        z, _, _, stamp, geometries = _polygons(image, xy, mpp, c, include_geometry=True)
        from shapely.geometry import mapping
        geometry = mapping(geometries[zone])
        design = json.loads((Path(saved["folder"]) / "design.json").read_text())
        if stamp not in design["audit"]["annotations"]:
            raise ValueError("These annotations changed since the scan. Rerun before linking cells.")
    else:
        d = np.zeros(len(frame)) if c["zone_source"] == "whole" else pd.to_numeric(frame[c["distance_col"]], errors="coerce").to_numpy(float)
        if c["zone_source"] == "distance" and c["distance_unit"] == "px":
            d *= mpp
        z = _zone_index(d, c)
    mask = z == zone if c["zone_mode"] == "rings" else ((z >= 0) & (z <= zone))
    parts = json.loads(feature)
    names = c["phenotypes"] if parts[0] == "diversity" else [n for n in parts[1:] if n in c["phenotypes"]]
    if names:
        mask &= frame[c["phenotype_col"]].isin(names).to_numpy()
    if _table_stamp(table) != saved["source"]:
        raise ValueError("Cell data changed while opening this result.")
    if channel:
        from . import data_editor, flow_gating
        ids = pd.to_numeric(frame.loc[mask, "label"], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(ids) & (ids > 0) & (ids == np.floor(ids)) & (ids <= 20000000)):
            raise ValueError("Set valid segmentation labels before linking cells.")
        selected = pd.DataFrame({"_flow_table": str(link_table), "_flow_row": frame.index[mask],
                                 "_flow_id": ids.astype(np.int64), "_flow_roi": image})
        if selected.duplicated(["_flow_roi", "_flow_id"]).any():
            raise ValueError("Segmentation labels repeat in this image. Check the image ID column.")
        selection = flow_gating.register_selection(channel, str(link_table), data_editor.stamp(link_table), selected, "Spatial screening")
        coords_px = (xy / mpp) if c["zone_source"] == "polygons" else xy if c["coordinate_unit"] == "px" else None
        bounds = None
        if coords_px is not None and mask.any():
            low, high = coords_px[mask].min(axis=0)-20, coords_px[mask].max(axis=0)+20
            bounds = [float(low[0]), float(low[1]), float(high[0]-low[0]), float(high[1]-low[1])]
        return {"selection": selection, "table": str(link_table), "name": "Spatial screening", "geometry": geometry, "bounds": bounds}
    return {"labels": frame.loc[mask, "label"].tolist(), "column": c["phenotype_col"]}
