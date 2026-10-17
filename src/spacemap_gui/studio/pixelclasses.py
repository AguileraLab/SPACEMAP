"""Studio's Annotation part for pixel classes (Tab 4 step 3), live on the image.

The marker stack is built once per image with Tab 4's ``build_feature_stack``
at a low pyramid level (about 1,200 pixels across) over the valid tissue Tab 4
saved, else the tissue found now. Choosing the number of classes re-runs Tab
4's ``run_pixel_clustering`` on a sample of pixels, which takes about a second,
and the class map is sent as a small PNG drawn over the image.

Classes are named into domains (Tab 4's ``annot_ta_domain_map``) and the map
can be coloured by domain. Exporting clusters the preview again the same way,
gives every pixel at Tab 4's export resolution the class of the nearest preview
class centre, so the names still fit, and writes Tab 4's files with Tab 4's
``export_cluster_partition`` and ``export_domain_partition``.
"""

from collections import OrderedDict
import base64
import functools
import io
import json
from pathlib import Path
import re
import threading
import time

import numpy as np

PREVIEW_SIDE = 1200
FIT_SAMPLE = 20000
UNASSIGNED = "Unassigned"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_ .+-]{0,59}$")
_prepared = OrderedDict()
_lock = threading.Lock()


def _saved(config, key, default):
    value = ((config.get("tab_configurations") or {}).get("tab4_annotation") or {}).get(key)
    return value if value not in (None, "", [], {}) else default


def settings(config):
    from spacemap.tissue_annotation_utils import DEFAULT_CLUSTERING_CONFIG

    saved = _saved(config, "annot_ta_cluster_config", {})
    return {**DEFAULT_CLUSTERING_CONFIG, **(saved if isinstance(saved, dict) else {})}


def tissue_mask(reader, config, factor, markers):
    """Tab 4's tissue at *factor* when no valid tissue has been saved yet."""
    from spacemap.tissue_annotation_utils import DEFAULT_TISSUE_CONFIG, detect_tissue_mask, split_he_and_codex_channels

    he, _ = split_he_and_codex_channels(list(reader.channel_names))
    values = {**DEFAULT_TISSUE_CONFIG, **_saved(config, "annot_ta_tissue_config", {}), "blur_downsample": factor}
    return detect_tissue_mask(reader, values, he_indices=list(he), codex_indices=list(markers))["tissue_mask"]


def prepare(image, config, valid_tissue=None, *, options=None):
    """Tab 4's marker stack over the valid tissue at the preview level; cached per image."""
    from spacemap.tissue_annotation_utils import (
        PyramidImageReader,
        build_feature_stack,
        load_geojson_mask,
        split_he_and_codex_channels,
    )
    from spacemap.viewer.sources import fingerprint

    values = settings(config)
    reader = PyramidImageReader(image)
    names = list(reader.channel_names)
    _, codex = split_he_and_codex_channels(names)
    chosen = [names.index(name) for name in _saved(config, "annot_ta_cluster_markers", []) if name in names]
    markers = chosen or list(codex)
    if options is not None:
        markers = [names.index(name) for name in options['markers']] if options.get('selected_only') else list(range(len(names)))
        values = {**values, 'clustering_downsample':options['scale'], 'spatial_features':options['features'],
                  'use_top_k_masking':options['top_k'], 'sigma_spatial_smoothing':options['smoothing']}
    if not markers:
        raise ValueError("This image has no markers to classify.")
    factor = max(int(values.get("clustering_downsample", 4)), 1)
    height, width = reader.full_shape
    while options is None and max(height, width) / factor > PREVIEW_SIDE:
        factor *= 2
    signature = json.dumps(
        {
            "image": str(image),
            "stamp": fingerprint(image),
            "markers": markers,
            "factor": factor,
            "values": {k: v for k, v in values.items() if k not in ("n_clusters", "random_seed")},
            "scope_path": str(valid_tissue),
            "valid": Path(valid_tissue).stat().st_mtime_ns if valid_tissue and Path(valid_tissue).is_file() else None,
        },
        sort_keys=True,
        default=str,
    )
    with _lock:
        if signature in _prepared:
            _prepared.move_to_end(signature)
            return _prepared[signature]
    started = time.perf_counter()
    shape = reader.target_shape(factor)
    mask = None
    if valid_tissue and Path(valid_tissue).is_file():
        mask = load_geojson_mask(str(valid_tissue), shape, float(factor))
    if mask is None:
        mask = tissue_mask(reader, config, factor, markers)
    if not mask.any():
        raise ValueError("The saved tissue has no pixels at this scale. Review Tissue or choose a finer scale.")
    stack = build_feature_stack(reader, markers, mask, {**values, "clustering_downsample": factor})
    prepared = {
        "signature": signature,
        "stack": stack,
        "values": values,
        "factor": factor,
        "full_shape": (height, width),
        "from_saved_tissue": bool(valid_tissue and Path(valid_tissue).is_file()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    with _lock:
        _prepared[signature] = prepared
        while len(_prepared) > 2:
            _prepared.popitem(last=False)
    return prepared


def _png(cluster_map, colours):
    """The class map as a PNG data URL; a class whose colour is None is left clear."""
    from PIL import Image

    rgba = np.zeros(cluster_map.shape + (4,), np.uint8)
    for index, colour in enumerate(colours):
        if not colour:
            continue
        colour = colour.lstrip("#")
        rgba[cluster_map == index] = (int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16), 255)
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _clusters(prepared, n_clusters):
    """Tab 4's clustering of the prepared stack; the same stack and count give the same classes."""
    from spacemap.tissue_annotation_utils import run_pixel_clustering

    last = prepared.get("last")
    if last and last["n_clusters"] == n_clusters:
        return last["found"]
    found = run_pixel_clustering(
        prepared["stack"], {**prepared["values"], "n_clusters": n_clusters, "kmeans_fit_sample": FIT_SAMPLE}
    )
    prepared["last"] = {"n_clusters": n_clusters, "found": found}
    return found


def _count(n_clusters):
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or not 2 <= n_clusters <= 40:
        raise ValueError("Choose between 2 and 40 classes.")
    return n_clusters


def check_names(n_clusters, domains, excluded):
    """{class: domain} and the removed classes, checked; blank and Unassigned names are dropped."""
    _count(n_clusters)
    if not isinstance(domains, dict) or not isinstance(excluded, list):
        raise ValueError("Send the domain of each class and the removed classes.")
    named = {}
    for key, name in domains.items():
        try:
            cluster = int(key)
        except (TypeError, ValueError):
            raise ValueError("Classes are numbered.") from None
        if not 0 <= cluster <= n_clusters or not isinstance(name, str):
            raise ValueError(f"There is no class {key}.")
        name = name.strip()
        if name and name != UNASSIGNED:
            if not NAME.match(name):
                raise ValueError(f"'{name}' cannot be a domain name: use letters, digits, spaces, _ . + or -.")
            named[cluster] = name
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= n_clusters for v in excluded):
        raise ValueError("Remove classes by number.")
    return named, sorted(set(excluded))


def domain_order(domains, excluded=()):
    """Domain names in Tab 4's export order: by the first class named into each."""
    order = []
    for cluster in sorted(domains):
        name = domains[cluster]
        if cluster not in excluded and name not in order:
            order.append(name)
    return order


def domain_colours(config, order):
    """Tab 4's colours for the domains: saved ones kept, new ones far from them."""
    from spacemap.tissue_annotation_utils import assign_distinct_colors

    saved = _saved(config, "annot_ta_domain_colors", {})
    return assign_distinct_colors(order, existing=saved if isinstance(saved, dict) else {}, seed=42)


@functools.lru_cache(maxsize=1)
def common_names():
    """Tab 4's catalogue of common domain names, read from its source (importing it loads Streamlit)."""
    import ast

    import spacemap_gui

    source = Path(spacemap_gui.__file__).parent / "tabs" / "tab4_annotation" / "s3_pixel_classification.py"
    try:
        for item in ast.parse(source.read_text()).body:
            target = getattr(item, "target", None) or (getattr(item, "targets", None) or [None])[0]
            if getattr(target, "id", None) == "COMMON_DOMAIN_LABELS":
                return tuple(name for group in ast.literal_eval(item.value).values() for name in group)
    except (OSError, SyntaxError, ValueError):
        pass
    return ()


def names(config, sample, n_clusters):
    """The saved names when they were made for this image and class count, else none."""
    made_for = _saved(config, "annot_ta_domain_map_image", {})
    saved = _saved(config, "annot_ta_domain_map", {})
    fits = isinstance(made_for, dict) and made_for.get("sample") == sample and made_for.get("n_clusters") == n_clusters
    domains, excluded = {}, []
    if fits:
        domains = {int(k): str(v) for k, v in saved.items() if str(k).lstrip("-").isdigit() and str(v) != UNASSIGNED}
        excluded = [int(v) for v in _saved(config, "annot_ta_excluded_clusters", []) if isinstance(v, int)]
    known = [str(v) for v in _saved(config, "annot_ta_domain_names", []) if str(v).strip()]
    return {
        "domains": {str(k): v for k, v in sorted(domains.items())},
        "excluded": excluded,
        "names": known,
        "common": [name for name in common_names() if name not in known],
        "made_for": made_for if isinstance(made_for, dict) and made_for else None,
    }


def save_names(root, sample, n_clusters, domains, excluded, expected):
    """Tab 4's domain map, removed classes, domain names and colours; the class count too."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    named, removed = check_names(n_clusters, domains, excluded)
    with project_lock(root):
        config = read_config(root)
        if expected is not None and expected != revision(names(config, sample, n_clusters)):
            raise ValueError("These names changed in another window. Reload them before saving.")
        order = domain_order(named, removed)
        known = [str(v) for v in _saved(config, "annot_ta_domain_names", []) if str(v).strip()]
        colours = dict(_saved(config, "annot_ta_domain_colors", {}) or {})
        colours.update(domain_colours(config, order))
        clustering = dict(_saved(config, "annot_ta_cluster_config", {}) or {})
        clustering["n_clusters"] = n_clusters
        updates = {
            "annot_ta_domain_map": {str(k): v for k, v in sorted(named.items())},
            "annot_ta_excluded_clusters": removed,
            "annot_ta_domain_names": known + [name for name in order if name not in known],
            "annot_ta_domain_colors": colours,
            "annot_ta_domain_map_image": {"sample": sample, "n_clusters": n_clusters},
            "annot_ta_cluster_config": clustering,
        }
        for key, value in updates.items():
            persistence.update_stored_config_key(config, key, value)
        persistence.save_project_config(str(root), config)
        saved = names(read_config(root), sample, n_clusters)
    return {**saved, "revision": revision(saved)}


def classify(image, config, n_clusters, valid_tissue=None, hidden=(), domains=None, excluded=()):
    """Tab 4's pixel clustering at the preview level with *n_clusters* classes.

    With *domains* ({class: name}) the map is coloured by domain: classes not
    named are grey and removed classes are left clear.
    """
    _count(n_clusters)
    before = time.perf_counter()
    prepared = prepare(image, config, valid_tissue)
    freshly = time.perf_counter() - before > 1.0
    started = time.perf_counter()
    stack = prepared["stack"]
    # Showing or hiding a class, or naming it, only redraws the map: the last clustering is kept.
    found = _clusters(prepared, n_clusters)
    cluster_map, colours = found["cluster_map"], list(found["cluster_colors"])
    labels = found["labels_valid"]
    z = stack["pixels_z_valid"]
    marker_names = stack["marker_names"]
    classes, profile = [], []
    for row in found["summary"].itertuples():
        members = labels == row.cluster
        means = z[members].mean(axis=0) if members.any() else np.zeros(len(marker_names))
        profile.append(np.round(means, 3).tolist())
        top = [marker_names[i] for i in np.argsort(means)[::-1][:3] if means[i] > 0]
        classes.append(
            {
                "id": int(row.cluster),
                "colour": colours[int(row.cluster)] if int(row.cluster) < len(colours) else "#888888",
                "percent": round(float(row.tissue_percent), 2),
                "top_markers": top,
                "low_signal": found["reserved_low_signal_cluster"] == int(row.cluster),
            }
        )
    hidden, legend = set(hidden), None
    if domains is None:
        painted = [None if index in hidden else colour for index, colour in enumerate(colours)]
    else:
        named, removed = check_names(n_clusters, domains, list(excluded))
        order = domain_order(named, removed)
        palette = domain_colours(config, order)
        painted = [
            None if index in removed or index in hidden else palette.get(named.get(index), "#555555")
            for index in range(len(colours))
        ]
        share = {item["id"]: item["percent"] for item in classes}
        legend = [
            {
                "name": name,
                "colour": palette[name],
                "classes": [c for c in sorted(named) if named[c] == name and c not in removed],
                "percent": round(sum(share.get(c, 0) for c in named if named[c] == name and c not in removed), 2),
            }
            for name in order
        ]
    height, width = prepared["full_shape"]
    return {
        "png": _png(cluster_map, painted),
        "image_width": width,
        "image_height": height,
        "classes": classes,
        "profile": {
            "rows": [str(item["id"]) for item in classes],
            "columns": list(marker_names),
            "values": profile,
            "note": "Mean of each marker (z-score over the tissue) in each class",
        },
        "domains": legend,
        "pixels": int(len(labels)),
        "ms": round(1000 * (time.perf_counter() - started)),
        "prepared_seconds": prepared["seconds"] if freshly else None,
        "from_saved_tissue": prepared["from_saved_tissue"],
        "n_clusters": n_clusters,
    }


def scaling(reader, stack, values):
    """The preview's per-marker normalisation in Tab 4's ``build_feature_stack``, to use at another resolution."""
    raw = reader.read_channels(stack["marker_indices"], stack["downsample"])
    percentile = float(values["percentile_threshold"])
    cofactors, lows, highs, means, spreads = [], [], [], [], []
    for channel, name in enumerate(stack["marker_names"]):
        cofactors.append(stack["cofactors"][name])
        plane = np.arcsinh(raw[channel] / cofactors[-1])
        lows.append(float(plane.min()))
        highs.append(float(np.percentile(plane, percentile)))
        column = stack["pixels_original_valid"][:, channel]
        means.append(float(column.mean()))
        spreads.append(float(column.std()))
    return {"cofactors": cofactors, "low": lows, "high": highs, "mean": means, "std": spreads}


def features_like(reader, stack, scale, mask, factor, values):
    """Tab 4's features at *factor* in the preview's units: its normalisation and Top-K, not new ones."""
    from skimage.filters import gaussian

    image = reader.read_channels(stack["marker_indices"], factor)
    count = image.shape[0]
    sigma = float(values["sigma_spatial_smoothing"])
    for channel in range(count):
        plane = image[channel]
        np.arcsinh(plane / scale["cofactors"][channel], out=plane)
        low, high = scale["low"][channel], scale["high"][channel]
        if high - low > 1e-8:
            plane -= low
            plane /= high - low
            np.clip(plane, 0.0, 1.0, out=plane)
        else:
            plane[:] = 0.0
        if sigma > 0:
            image[channel] = gaussian(plane, sigma=sigma, preserve_range=True).astype(np.float32)
    valid_indices = np.where(mask.reshape(-1))[0]
    if not valid_indices.size:
        raise ValueError("The valid tissue is empty at the export resolution.")
    original = image.reshape(count, -1).T[valid_indices]
    del image
    z = np.zeros_like(original, dtype=np.float32)
    for channel in range(count):
        if scale["std"][channel] > 1e-8:
            z[:, channel] = (original[:, channel] - scale["mean"][channel]) / scale["std"][channel]
    features = z
    if bool(values["use_top_k_masking"]):
        # Tab 4's Top-K masking, a block of pixels at a time: the same result in far less memory.
        k = int(min(max(stack["top_k_effective"], 1), count))
        features = np.zeros_like(z)
        for start in range(0, len(z), 400_000):
            block = z[start : start + 400_000]
            top = np.argsort(block, axis=1)[:, -k:]
            rows = np.arange(len(block))[:, None]
            features[start : start + 400_000][rows, top] = block[rows, top]
    return {
        "marker_names": stack["marker_names"],
        "shape": mask.shape,
        "valid_tissue_mask": mask,
        "valid_indices": valid_indices,
        "pixels_original_valid": original,
        "pixels_z_valid": z,
        "features": features,
    }


def _nearest(features, centres, chunk=400_000):
    """The nearest centre of each row, a block at a time to keep memory small."""
    labels = np.empty(len(features), np.int32)
    squared = (centres**2).sum(axis=1)
    for start in range(0, len(features), chunk):
        block = features[start : start + chunk]
        labels[start : start + chunk] = (squared[None, :] - 2.0 * block @ centres.T).argmin(axis=1)
    return labels


def export(image, config, output, sample, report=None, *, scope=None):
    """Tab 4 step 3's files for *image* with the classes and names saved in Studio."""
    from spacemap.tissue_annotation_utils import (
        PyramidImageReader,
        build_cluster_palette,
        export_cluster_partition,
        export_domain_partition,
        load_geojson_mask,
    )

    from . import annotation

    made_for = _saved(config, "annot_ta_domain_map_image", {})
    if not isinstance(made_for, dict) or made_for.get("sample") != sample:
        raise ValueError(f"Save the class names for {sample} in Studio first.")
    n_clusters = _count(made_for.get("n_clusters"))
    saved = names(config, sample, n_clusters)
    named, removed = check_names(n_clusters, saved["domains"], saved["excluded"])
    valid = Path(scope) if scope else annotation.analysis_tissue(output, sample)
    if report:
        report(f"{sample}: finding the {n_clusters} classes shown in Studio")
    prepared = prepare(image, config, valid)
    found = _clusters(prepared, n_clusters)
    values = prepared["values"]
    factor = max(int(values.get("clustering_downsample", 4)), 1)
    reserved = found["reserved_low_signal_cluster"]
    if factor == prepared["factor"]:
        stack, labels = prepared["stack"], found["labels_valid"]
    else:
        # The preview's class centres, applied to pixels put through the preview's normalisation.
        preview = prepared["stack"]["features"]
        centres = np.stack(
            [
                preview[found["labels_valid"] == c].mean(axis=0)
                if (found["labels_valid"] == c).any()
                else np.full(preview.shape[1], 1e6, np.float32)
                for c in range(n_clusters)
            ]
        ).astype(np.float32)
        reader = PyramidImageReader(image)
        scale = scaling(reader, prepared["stack"], values)
        shape = reader.target_shape(factor)
        mask = load_geojson_mask(str(valid), shape, float(factor)) if valid.is_file() else None
        if mask is None:
            mask = tissue_mask(reader, config, factor, prepared["stack"]["marker_indices"])
        if report:
            report(f"{sample}: reading {len(scale['cofactors'])} markers at 1/{factor} resolution")
        stack = features_like(reader, prepared["stack"], scale, mask, factor, values)
        if report:
            report(f"{sample}: giving {len(stack['features']):,} pixels their class")
        labels = _nearest(stack["features"], centres)
        if reserved is not None:
            threshold = float(np.quantile(preview.mean(axis=1), float(values["low_signal_quantile"])))
            labels[stack["features"].mean(axis=1) <= threshold] = reserved
    height, width = stack["shape"]
    cluster_map = np.full((height, width), -1, dtype=np.int32)
    cluster_map.flat[stack["valid_indices"]] = labels
    clustering = {
        "labels_valid": labels,
        "cluster_map": cluster_map,
        "cluster_colors": build_cluster_palette(n_clusters, include_low_signal=reserved is not None),
        "reserved_low_signal_cluster": reserved,
        "n_clusters": n_clusters,
    }
    settings = {**values, "clustering_downsample": factor}
    if report:
        report(f"{sample}: writing the class outlines")
    Path(output).mkdir(parents=True, exist_ok=True)
    clusters = export_cluster_partition(clustering, stack, settings, output, sample, excluded_clusters=removed)
    domains = None
    named = {c: name for c, name in named.items() if c not in removed}
    if named:
        if report:
            report(f"{sample}: writing the domains")
        domains = export_domain_partition(
            clusters,
            named,
            stack,
            clustering,
            settings,
            output,
            sample,
            domain_colors=domain_colours(config, domain_order(named, removed)),
        )
    counts = np.bincount(labels, minlength=len(clustering["cluster_colors"]))
    return {
        "files": [clusters["clusters_path"], clusters["merged_valid_path"]]
        + ([domains["domains_path"]] if domains else []),
        "classes": {int(c): round(100.0 * int(n) / max(len(labels), 1), 2) for c, n in enumerate(counts) if n},
        "domains": len(domain_order(named, removed)),
        "polygons": clusters["polygon_count"],
        "resolution_um": clusters["export_resolution_um"],
        "factor": factor,
    }
