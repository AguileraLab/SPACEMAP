"""Exploratory B/T aggregates, with Cell tagging's containment and distance conventions.

Boundaries are unions of cell-centred discs, clipped to tissue; holes are never
filled. Connected polygon parts are assessed independently. This is a candidate
detector, not a diagnosis of TLS or a maturation classifier.
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import mapping

DEFAULTS = dict(phenotype_col="", roi_col="tissue_roi", b_labels=["B"],
                t_labels=["CD4", "CD8", "Treg"], max_gap=30.0, min_cells=50,
                min_b=20, min_t=20, min_fraction=0.5, min_density=0.0,
                pixel_um=0.0, edge_trim=0.0, output_col="TLS")
PREFIX = re.compile(r"TLS(?:_[A-Za-z0-9]{1,40})?")
SUFFIXES = ("", "_id", "_size", "_distance")


def outputs(prefix):
    if not isinstance(prefix, str) or not PREFIX.fullmatch(prefix):
        raise ValueError("Use TLS or TLS_ followed by a short name (letters and numbers).")
    return [prefix + suffix for suffix in SUFFIXES]


def check(changes):
    if not isinstance(changes, dict) or set(changes) - set(DEFAULTS):
        raise ValueError("Unknown TLS candidate setting.")
    result = dict(changes)
    for key, value in result.items():
        if key in ("b_labels", "t_labels"):
            if not isinstance(value, list) or not value or len(value) > 100 or any(
                    not isinstance(v, str) or not v.strip() or len(v) > 256 for v in value):
                raise ValueError("Choose at least one phenotype for B cells and for T cells.")
            result[key] = list(dict.fromkeys(value))
        elif key in ("phenotype_col", "roi_col", "output_col"):
            if not isinstance(value, str) or len(value) > 256:
                raise ValueError("Choose a valid column name.")
            result[key] = value.strip()
            if key == "output_col":
                outputs(result[key])
        else:
            limits = {"max_gap": (1, 500), "min_cells": (2, 1000000), "min_b": (1, 1000000),
                      "min_t": (1, 1000000), "min_fraction": (0, 1), "min_density": (0, 1000000),
                      "pixel_um": (0, 1000), "edge_trim": (0, 250)}
            low, high = limits[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}.")
            if key in ("min_cells", "min_b", "min_t") and int(value) != value:
                raise ValueError("Cell counts must be whole numbers.")
    if "max_gap" in result and result.get("edge_trim", 0) >= result["max_gap"] / 2:
        raise ValueError("Trim edge must be smaller than half the Maximum gap.")
    return result


def values(config):
    saved = (config.get("tab_configurations", {}).get("tab8_analysis", {}).get("studio_tls") or {})
    return {**DEFAULTS, **check(saved)}


def save(root, changes, expected):
    from spacemap_gui import persistence
    from .settings import project_lock, read_config, revision

    changes = check(changes)
    with project_lock(root):
        config = read_config(root)
        if expected != revision(values(config)):
            raise ValueError("TLS settings changed in another window. Reopen the tab before saving.")
        chosen = check({**values(config), **changes})
        config.setdefault("tab_configurations", {}).setdefault("tab8_analysis", {})["studio_tls"] = chosen
        persistence.save_project_config(str(root), config)
    return chosen


def labels(table, column):
    from spacemap import dataio
    if column not in dataio.columns(table):
        return []
    return sorted(dataio.load(table, columns=[column])[column].dropna().astype(str).unique().tolist())


def _parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [p for g in getattr(geometry, "geoms", []) for p in _parts(g)]


def _discs(xy, radius):
    """Bounded intermediate memory, no all-pairs graph or image-sized raster."""
    chunks = [shapely.union_all(shapely.buffer(shapely.points(xy[i:i + 4096]), radius, quad_segs=12))
              for i in range(0, len(xy), 4096)]
    return shapely.union_all(chunks)


def detect_region(xy, phenotypes, chosen, microns, tissue=None):
    """Return candidate polygons (pixel coordinates) and measured cells.

    Disc overlap supplies the distance-limited connectivity: two centres more
    than max_gap apart cannot connect directly. No convex hull, dilation/closing
    across holes, or infinite-range nearest-neighbour link is used. Discs are a
    polygon approximation, making connectivity conservatively shorter by <0.3%.
    Tissue from annotations wins; without it, a union of all cells' discs is an
    explicitly approximate tissue support for the same clean-line distance rule.
    """
    points = shapely.points(xy)
    b = np.isin(phenotypes, chosen["b_labels"])
    t = np.isin(phenotypes, chosen["t_labels"])
    radius = float(chosen["max_gap"]) / (2 * microns)
    zeros = np.zeros(len(xy), dtype=float)
    result = {"": zeros.copy(), "_id": zeros.copy(), "_size": zeros.copy(),
              "_distance": np.full(len(xy), np.nan)}
    if b.sum() < chosen["min_b"] or t.sum() < chosen["min_t"]:
        return [], result
    # Seed with connected B-cell cores. Admit only T cells within one maximum
    # gap of a qualifying B core; do not propagate through long T-cell chains.
    # The final outline defines connectivity, avoiding disagreeing IDs when a
    # tissue hole splits a putative aggregate.
    support = tissue if tissue is not None else _discs(xy, radius)
    cores = _parts(_discs(xy[b], radius).intersection(support))
    if not cores:
        return [], result
    core_hits = shapely.STRtree(cores).query(points[b], predicate="intersects")
    core_counts = np.bincount(core_hits[1], minlength=len(cores))
    seed_rows = np.unique(core_hits[0][core_counts[core_hits[1]] >= chosen["min_b"]])
    if not len(seed_rows):
        return [], result
    seed = xy[b][seed_rows]
    from scipy.spatial import cKDTree
    near = cKDTree(seed).query(xy[t], k=1)[0] <= 2 * radius
    aggregate = _discs(np.concatenate([seed, xy[t][near]]), radius).intersection(support)
    trim = float(chosen.get("edge_trim", 0)) / microns
    if trim:
        aggregate = aggregate.buffer(-trim)
    parts = sorted(_parts(aggregate), key=lambda g: (g.bounds[1], g.bounds[0], -g.area))
    return measure_regions(xy, phenotypes, chosen, microns, parts, support)


def measure_regions(xy, phenotypes, chosen, microns, parts, support, *, manual=False):
    """The same containment, area and signed-distance calculation after detection or editing."""
    from spacemap.cell_spatial_annotation import _nearest_visible_distance, _region_hits

    points = shapely.points(xy)
    b, t = np.isin(phenotypes, chosen["b_labels"]), np.isin(phenotypes, chosen["t_labels"])
    result = {"": np.zeros(len(xy)), "_id": np.zeros(len(xy)), "_size": np.zeros(len(xy)),
              "_distance": np.full(len(xy), np.nan)}
    if not parts:
        return [], result
    hits = shapely.STRtree(parts).query(points, predicate="intersects")
    counts = np.bincount(hits[1], minlength=len(parts))
    bs = np.bincount(hits[1], weights=b[hits[0]], minlength=len(parts))
    ts = np.bincount(hits[1], weights=t[hits[0]], minlength=len(parts))
    regions = []
    for i, polygon in enumerate(parts):
        nb, nt, n = int(bs[i]), int(ts[i]), int(counts[i])
        area = float(polygon.area) * microns ** 2
        density = (nb + nt) * 1e6 / area if area else 0
        if not manual and (nb < chosen["min_b"] or nt < chosen["min_t"] or nb + nt < chosen["min_cells"]
                or (nb + nt) / max(n, 1) < chosen["min_fraction"] or density < chosen["min_density"]):
            continue
        regions.append(dict(id=len(regions) + 1, geom=polygon, area_um2=area,
                            cells=n, b_cells=nb, t_cells=nt,
                            lymphocyte_fraction=(nb + nt) / max(n, 1), density_per_mm2=density,
                            **({"reviewed": True} if manual else {})))
    if regions:
        positions, owners = _region_hits(points, regions)
        result[""][positions] = 1
        result["_id"][positions] = [regions[i]["id"] for i in owners]
        result["_size"][positions] = [regions[i]["area_um2"] for i in owners]
        cache = {}
        for start in range(0, len(points), 8192):
            positions, distance = _nearest_visible_distance(
                points[start:start + 8192], regions, support, 0.0, len(regions), probe_cache=cache)
            result["_distance"][start + positions] = distance * microns
    return regions, result


def _eligible(series):
    text = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return series.notna() & ~numeric.le(0) & ~text.isin(["", "nan", "none", "unassigned", "outside", "<na>"])


def calibration(root, images, image_entries=()):
    """MPP per image: loaded/project image metadata, then project calibration.

    Entries come from Studio's trusted image catalogue, never client paths.
    Never borrow the open image's MPP for a different ROI in the cohort.
    """
    from spacemap.viewer.sources import image_info
    from . import segmentation, tagging
    from .settings import read_config

    config = read_config(root)
    candidates = {}
    def add(sample, path):
        if sample and path:
            candidates.setdefault(tagging._sample_key(sample), []).append(Path(path))
    for entry in image_entries or []:
        add(entry.get("sample") or entry.get("name") or entry.get("path"), entry.get("path"))
    directory = segmentation.step_folders(root, config, config.get("directory_mappings") or {})["images"]
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.name.lower().endswith(segmentation.IMAGE_SUFFIXES):
                add(path.name, path)
    def positive(value):
        try:
            number = float(value)
            return number if np.isfinite(number) and number > 0 else None
        except (ValueError, TypeError):
            return None
    fallback = positive(config.get("mpp_codex"))
    found = {}
    for image in images:
        item = {"mpp": fallback, "source": "project" if fallback else "missing"}
        for path in dict.fromkeys(candidates.get(tagging._sample_key(image), [])):
            try:
                size = positive(image_info(str(path)).pixel_size_um)
            except (OSError, ValueError, RuntimeError):
                continue
            if size:
                item = {"mpp": size, "source": "image", "file": str(path)}
                break
        found[str(image)] = item
    return found


def prepare(root, table, chosen, images, overwrite=False, image_entries=()):
    """Freeze settings, pixel calibration and annotation inputs before the worker starts."""
    from spacemap import dataio
    from spacemap.viewer.sources import fingerprint
    from . import tagging, tasks
    from .settings import read_config

    chosen = {**DEFAULTS, **check(chosen)}
    if set(chosen["b_labels"]) & set(chosen["t_labels"]):
        raise ValueError("B-cell and T-cell phenotypes must be different.")
    required = ["batch_id", "x_centroid", "y_centroid", chosen["phenotype_col"]]
    if chosen["roi_col"]:
        required.append(chosen["roi_col"])
    missing = set(required) - set(dataio.columns(table))
    if missing:
        raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
    known = set(labels(table, chosen["phenotype_col"]))
    if not known.intersection(chosen["b_labels"]) or not known.intersection(chosen["t_labels"]):
        raise ValueError("Choose B-cell and T-cell labels present in the phenotype column.")
    selected = dataio.load(table, columns=["batch_id"])["batch_id"].dropna().astype(str).unique().tolist()
    if images:
        if set(images) - set(selected):
            raise ValueError("Some selected images are not in this table.")
        selected = [i for i in selected if i in images]
    if not selected:
        raise ValueError("No images to analyse.")
    config = read_config(root)
    state = tagging.values(root, config)
    exports, _ = tagging._exports_and_images(root)
    detected = calibration(root, selected, image_entries) if not chosen["pixel_um"] else {}
    sources, scales = {}, {}
    for image in selected:
        scale = chosen["pixel_um"] or detected[image]["mpp"]
        if not scale or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{image}: no pixel calibration. Set MPP below Maximum gap before running.")
        scales[image] = float(scale)
        match = (state.get(tagging.tab5.GEOJSON_MATCHES_KEY) or {}).get(image)
        path, _ = tagging.annotation_path(state, image, match, exports)
        if path and Path(path).is_file():
            sources[image] = {"path": str(path), "revision": fingerprint(Path(path))}
    names = outputs(chosen["output_col"])
    settings = dict(root=str(root), table=str(table), result="tls_candidates", values=chosen,
                    images=selected, image_col="batch_id", markers=required, cell_id_col="cell_id",
                    output_col=names[0], auxiliary_columns=names[1:],
                    overwrite_columns=names if overwrite else [], pixel_sizes=scales,
                    pixel_sources=detected or {i: {"mpp": chosen["pixel_um"], "source": "manual"} for i in selected},
                    tissue_sources=sources, tissue_label=state[tagging.tab5.TISSUE_LABEL_KEY])
    tasks.validate_table(settings)
    settings["revision"] = tasks.source_revision(settings)
    return settings


def run(settings, folder, report):
    from spacemap import dataio
    from spacemap.cell_spatial_annotation import load_annotation_polygons
    from spacemap.viewer.sources import fingerprint
    from . import tasks

    tasks.validate_table(settings)
    chosen = settings["values"]
    names = outputs(chosen["output_col"])
    report("Reading positions and phenotypes…", .03)
    review = settings.get("review")
    frame = dataio.load(settings["table"], columns=list(dict.fromkeys(["cell_id", *settings["markers"], *(names if review else [])]))).reset_index(drop=True)
    # Only a handful of columns for the cohort; geometry work is per image/region.
    tags = {name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan).copy()
            if review else np.full(len(frame), np.nan) for name in names}
    work = frame[frame.batch_id.astype(str).isin(settings["images"])].copy()
    coordinates = work[["x_centroid", "y_centroid"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    work[["x_centroid", "y_centroid"]] = coordinates
    valid = np.isfinite(coordinates).all(axis=1)
    if chosen["roi_col"]:
        valid &= _eligible(work[chosen["roi_col"]])
    work = work[valid]
    counts = {image: 0 for image in settings["images"]}
    measured = {image: 0 for image in settings["images"]}
    features, summaries, warnings = [], [], []
    if review:
        from . import jobs, tls_review
        parent = jobs.directory(settings["root"], review["parent"])
        prior = jobs.status(settings["root"], review["parent"])["result"]
        retained = lambda row: not (row["image"] == review["image"] and row["region"] == review["region"])
        summaries = [s for s in prior["structures"] if retained(s)]
        features = [f for f in json.loads((parent / "tls_candidates.geojson").read_text())["features"] if retained(f["properties"])]
        counts, measured, warnings = dict(prior["counts"]), dict(prior["measured"]), list(prior["warnings"])
        counts[review["image"]] = sum(s["image"] == review["image"] for s in summaries)
    sources = settings.get("tissue_sources", {})
    for number, image in enumerate(settings["images"]):
        if review and image != review["image"]:
            continue
        report(f"{image}: finding B/T aggregates ({number + 1}/{len(counts)})…", .08 + .83 * number / len(counts))
        image_rows = work[work.batch_id.astype(str).eq(image)]
        tissue = None
        if image in sources:
            source = sources[image]
            if fingerprint(Path(source["path"])) != source["revision"]:
                raise ValueError("Tissue annotations changed. Start again with the current annotations.")
            polygons = load_annotation_polygons(source["path"]).get(settings["tissue_label"].lower(), [])
            if polygons:
                tissue = shapely.union_all(polygons)
        if tissue is None:
            warnings.append(f"{image}: tissue gaps inferred from cells; no linked tissue polygon.")
        else:
            image_rows = image_rows[shapely.intersects(tissue, shapely.points(image_rows[["x_centroid", "y_centroid"]].to_numpy(float)))]
        grouped = image_rows.groupby(chosen["roi_col"], sort=True) if chosen["roi_col"] else [("image", image_rows)]
        for region, part in grouped:
            if review and str(region) != review["region"]:
                continue
            xy = part[["x_centroid", "y_centroid"]].to_numpy(float)
            phenotypes = part[chosen["phenotype_col"]].fillna("").astype(str).to_numpy()
            microns = settings["pixel_sizes"][image]
            if review:
                # Editing can include more T cells, but cannot bridge unsupported tissue gaps.
                # Scope the boundary to this region's cell support as well as available tissue.
                support = _discs(xy, chosen["max_gap"] / (2 * microns))
                if tissue is not None:
                    support = support.intersection(tissue)
                edited = shapely.union_all(tls_review.polygons(review["features"])).intersection(support)
                pieces = sorted(_parts(edited), key=lambda g: (g.bounds[1], g.bounds[0], -g.area))
                found, result = measure_regions(xy, phenotypes, chosen, microns, pieces, support, manual=True)
            else:
                found, result = detect_region(xy, phenotypes, chosen, microns, tissue)
            for suffix in SUFFIXES:
                tags[chosen["output_col"] + suffix][part.index.to_numpy()] = result[suffix]
            if not review:
                measured[image] += len(part)
            counts[image] += len(found)
            for item in found:
                row = {k: v for k, v in item.items() if k != "geom"}
                row.update(image=image, region=str(region), key=json.dumps([image, str(region), item["id"]]),
                           pixel_um=settings["pixel_sizes"][image], tissue_source="annotation" if tissue is not None else "cells")
                summaries.append(row)
                features.append(dict(type="Feature", geometry=mapping(item["geom"]),
                                     properties={**row, "classification": {"name": "TLS candidate"}}))
    # Artifacts remain private to the job until the guarded column save succeeds.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tls_candidates.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features)))
    fields = ["image", "region", "id", "key", "area_um2", "cells", "b_cells", "t_cells", "lymphocyte_fraction",
              "density_per_mm2", "pixel_um", "tissue_source", "reviewed"]
    pd.DataFrame(summaries, columns=fields).to_csv(folder / "tls_candidates.csv", index=False)
    (folder / "tls_method.json").write_text(json.dumps({"method": "manual_boundary_review_v1" if review else "b_core_discs_v2", "table": settings["table"],
        "parent_job": review["parent"] if review else None,
        "manual_scope": {k: review[k] for k in ("image", "region")} if review else None,
        "source_revision": settings["revision"], "images": settings["images"], "settings": chosen, "pixel_sizes": settings["pixel_sizes"],
        "pixel_sources": settings.get("pixel_sources", {}),
        "tissue_sources": sources, "warnings": warnings, "distance": "Cell tagging clean straight-line boundary distance; µm, negative inside",
        "boundary": ("User-reviewed polygons, unioned within one region and clipped to that region's cell support and linked tissue; no detection cutoffs reapplied"
                     if review else "Connected B-cell cores (at least min_b), then T cells within max_gap of core cells; union of discs radius max_gap/2, optionally inset by edge_trim, clipped to tissue; holes preserved")}, indent=2))
    for source in sources.values():
        if fingerprint(Path(source["path"])) != source["revision"]:
            raise ValueError("Tissue annotations changed during the run. No TLS columns were applied.")
    saved = tasks.save_result(settings, tags, frame.cell_id, report)
    return {**saved, "folder": str(folder), "images": settings["images"], "counts": counts, "measured": measured,
            "structures": summaries, "warnings": list(dict.fromkeys(warnings)), "settings": chosen,
            **({"reviewed": True, "parent_job": review["parent"]} if review else {}),
            "total": len(summaries), "units": {"size": "µm²", "distance": "µm"}}


def latest(root, table):
    from . import jobs
    from spacemap import dataio
    available = set(dataio.columns(table)) if table else set()
    # Stop at the first matching run rather than loading all job results (some
    # clustering reports are large). Completed TLS jobs need no process probe.
    paths = sorted(jobs.root(root).glob("*/status.json"), key=lambda p: p.parent.stat().st_mtime_ns, reverse=True)
    for path in paths:
        if not re.fullmatch(r"[0-9a-f]{32}", path.parent.name):
            continue
        try:
            job = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        result = job.get("result") or {}
        if (job.get("kind") == "tls_candidates" and job.get("status") == "done"
                and result.get("table") == str(table) and set(result.get("columns") or [None]) <= available):
            return {**result, "job_id": path.parent.name}
    return None
