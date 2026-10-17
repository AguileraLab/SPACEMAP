"""Studio's regions: polygons to review and name (pixel classes and marker regions), and what stands out in each.

A region's card lists the markers high (and low) inside it against the tissue: for every marker, the mean inside the
region (read at the finest pyramid level where the region is at most 2,048 pixels across) is set against the tissue's
pixels (read once per image at about 1,500 pixels across): z is how many robust spreads (1.4826 x MAD) the region's
mean lies above the tissue's median, and fold is the region's mean over the tissue's mean.
"""

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

REFERENCE_SIDE = 1500


def _stamp(path):
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


def _shapes(path):
    from shapely.geometry import shape

    data = json.loads(Path(path).read_text())
    features = data.get("features", []) if isinstance(data, dict) else data
    return [shape(f["geometry"]).buffer(0) for f in features if f.get("geometry")]


@lru_cache(maxsize=6)
def _reference(image, image_stamp, tissue, tissue_stamp):
    """Per marker: the median, robust spread and mean of the tissue's pixels (the whole image without a tissue)."""
    import rasterio.features
    from affine import Affine
    from spacemap.viewer.sources import image_info, open_source

    info = image_info(image)
    level = next(
        (
            i
            for i in range(len(info.levels) - 1, -1, -1)
            if max(info.levels[i].width, info.levels[i].height) >= REFERENCE_SIDE
        ),
        0,
    )
    size = info.levels[level]
    source = open_source(image)
    inside = None
    if tissue:
        shapes = _shapes(tissue)
        if shapes:
            inside = rasterio.features.rasterize(
                [(g, 1) for g in shapes],
                out_shape=(size.height, size.width),
                transform=Affine(size.downsample_x, 0, 0, 0, size.downsample_y, 0),
                fill=0,
                dtype="uint8",
            ).astype(bool)
            if not inside.any():
                inside = None
    stats = []
    for channel in range(len(info.channel_names)):
        data = source.read_region(level, channel, 0, 0, size.width, size.height).astype(np.float32)
        values = data[inside] if inside is not None else data.reshape(-1)
        median = float(np.median(values))
        spread = float(1.4826 * np.median(np.abs(values - median))) or float(values.std()) or 1.0
        stats.append((median, spread, float(values.mean())))
    return {
        "stats": stats,
        "names": list(info.channel_names),
        "tissue": inside is not None,
        "microns": round((info.pixel_size_um or 0) * size.downsample_x, 2),
    }


def profile(image, geometry, tissue=None):
    """Each marker's mean inside *geometry* (image pixels) against the tissue: robust z and fold, highest first."""
    from shapely.geometry import shape
    from spacemap.viewer.render import marker_pixels
    from spacemap.viewer.sources import image_info

    region = shape(geometry).buffer(0)
    if region.is_empty:
        raise ValueError("The region has no area.")
    tissue = str(tissue) if tissue and Path(tissue).is_file() else None
    reference = _reference(str(image), _stamp(image), tissue, _stamp(tissue) if tissue else 0)
    info = image_info(str(image))
    markers = []
    for channel, name in enumerate(reference["names"]):
        if info.rgb:
            break
        values, level = marker_pixels(str(image), info, channel, region=region)
        mean = float(np.asarray(values, np.float64).mean())
        median, spread, reference_mean = reference["stats"][channel]
        markers.append(
            {
                "name": name,
                "index": channel,
                "mean": round(mean, 2),
                "z": round((mean - median) / spread, 2),
                "fold": round(mean / reference_mean, 2) if reference_mean > 0 else 0.0,
            }
        )
    markers.sort(key=lambda item: -item["z"])
    return {
        "markers": markers,
        "reference": "tissue" if reference["tissue"] else "whole image",
        "microns": round((info.pixel_size_um or 0) * info.levels[level].downsample_x, 2) if markers else 0,
        "reference_microns": reference["microns"],
        "area_px": round(region.area, 1),
    }


# ---------------------------------------------------------------- marker regions
SETTINGS_KEY = "annot_studio_marker_regions"
DEFAULTS = {"rules": [], "sigma": 2.0, "within": "tissue", "min_area_um2": 100.0, "scale": 4, "name": "",
            "cutoff": "raw", "method": "Triangle", "cofactor": 5.0, "pool": "slide", "percentile": 95.0, "manual": 0.0}
# How a rule's level is read: raw intensity; the image's own background level by an Auto method, or as saved (the
# right panel's min, or the Background step's table), times the rule's factor; or a robust z-score of the arcsinh
# intensities inside the area (median and MAD there, so a few bright pixels do not move it); or two groups: the area's
# pixels (log intensities) split into background and signal by a two-group mixture, fitted per image or for all ROIs of
# a slide together, and a pixel counts when it is at least this sure to be signal.
CUTOFFS = ("raw", "auto", "saved", "z", "groups")
RULE_DEFAULT = {"raw": 100.0, "auto": 1.0, "saved": 1.0, "z": 1.5, "groups": 0.9}
POOLS = ("image", "slide")
SAMPLE = 300_000  # pixels per fit, shared between a slide's ROIs
SCALES = (1, 2, 4, 8, 16)
REGION_COLOUR = "#ffd600"


def settings(config):
    saved = next(
        (
            tab[SETTINGS_KEY]
            for tab in (config.get("tab_configurations") or {}).values()
            if isinstance(tab, dict) and isinstance(tab.get(SETTINGS_KEY), dict)
        ),
        {},
    )
    return {**DEFAULTS, **saved}


def check(values, markers=None):
    """The marker-region settings, cleaned: each rule a marker and the level it must be above (all rules at once)."""
    import re

    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError("Unknown marker-region setting.")
    values = {**DEFAULTS, **values}
    rules = []
    for rule in values["rules"] or []:
        if not isinstance(rule, dict) or not isinstance(rule.get("marker"), str) or not rule["marker"]:
            raise ValueError("Each rule needs a marker.")
        if markers is not None and rule["marker"] not in markers:
            raise ValueError(f"{rule['marker']} is not a marker of this image.")
        try:
            above = float(rule.get("above"))
        except (TypeError, ValueError):
            raise ValueError(f"Type the level {rule['marker']} must be above.") from None
        if isinstance(rule.get("above"), bool) or not math.isfinite(above):
            raise ValueError("Marker levels must be finite numbers.")
        if values.get("cutoff") in ("auto", "saved") and not 0 < above <= 100:
            raise ValueError(f"{rule['marker']}: the factor times its level is above 0 (1 is the level itself).")
        if values.get("cutoff") == "z" and not -10 <= above <= 20:
            raise ValueError(f"{rule['marker']}: the z-score is between -10 and 20 (1.5 is usual).")
        if values.get("cutoff") == "groups" and not 0.5 <= above <= 0.999:
            raise ValueError(f"{rule['marker']}: how sure is between 0.5 and 0.999 (0.9 is usual).")
        rules.append({"marker": rule["marker"], "above": above})
    try:
        sigma, smallest, scale = float(values["sigma"]), float(values["min_area_um2"]), int(values["scale"])
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Smoothing, smallest size and scale are numbers.") from None
    if (
        any(isinstance(values[key], bool) for key in ("sigma", "min_area_um2", "scale"))
        or not math.isfinite(smallest)
        or not 0 <= sigma <= 50
        or smallest < 0
        or scale not in SCALES
        or float(values["scale"]) != scale
        or values["within"] not in ("tissue", "image")
    ):
        raise ValueError(
            "Smoothing 0 to 50 px, smallest size 0 or more, scale 1, 2, 4, 8 or 16, within tissue or image."
        )
    from spacemap.background_detection import THRESHOLD_METHODS

    if values["cutoff"] not in CUTOFFS:
        raise ValueError("Choose the cutoff: raw intensity, Auto level, saved level, z-score or two groups.")
    if values["pool"] not in POOLS:
        raise ValueError("Fit two groups per image or per slide.")
    from spacemap.thresholding_engine import parameters
    threshold_params = parameters(values)
    if values["method"] not in THRESHOLD_METHODS:
        raise ValueError("Choose one of the Auto methods.")
    try:
        cofactor = float(values["cofactor"])
    except (TypeError, ValueError):
        raise ValueError("The arcsinh cofactor is a number.") from None
    if isinstance(values["cofactor"], bool) or not 0 < cofactor <= 100000:
        raise ValueError("The arcsinh cofactor is above 0 (5 is usual).")
    name = str(values.get("name") or "").strip()
    if name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_ .+()-]{0,59}", name):
        raise ValueError("Names use letters, numbers, spaces and _ . + ( ) - (up to 60).")
    return {
        "rules": rules,
        "sigma": sigma,
        "within": values["within"],
        "min_area_um2": smallest,
        "scale": scale,
        "name": name,
        "cutoff": values["cutoff"],
        "method": values["method"],
        **threshold_params,
        "cofactor": cofactor,
        "pool": values["pool"],
    }


def save(root, values):
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    values = check(values)
    with project_lock(root):
        config = read_config(root)
        persistence.update_stored_config_key(config, SETTINGS_KEY, values)
        persistence.save_project_config(str(root), config)
    return values


def saved_levels(root, config, mappings, image):
    """Each marker's background level saved for this image: its min in the right panel (set by Auto or by hand, or by
    the Background step's Detect), else the Background step's threshold table."""
    from spacemap.viewer import display

    from . import background

    levels = {}
    try:
        _, tables, _ = background.folders(root, mappings, config)
        table = background.table_path(tables, image)
        if table.is_file():
            levels.update(background.read_levels(table))
    except (OSError, ValueError, KeyError):
        pass
    try:
        shown = display.load(display.settings_path(root, image))
    except (OSError, ValueError):
        shown = {}
    for channel in shown.get("channels") or []:
        if channel.get("name") and isinstance(channel.get("low"), (int, float)) and math.isfinite(channel["low"]):
            levels[str(channel["name"])] = float(channel["low"])
    return levels


def _level(rule, data, inside, values, image, info, channel, shapes, levels):
    """The rule's cutoff in raw intensity for this image, and how it was found (shown to the user)."""
    from spacemap.viewer import render

    mode, above = values["cutoff"], rule["above"]
    if mode == "raw":
        return above, {"marker": rule["marker"], "raw": round(above, 2)}
    if mode == "auto":
        found = render.background_level(str(image), info, channel, values["method"], region=shapes or None, params=values)
        level = float(found["threshold"])
        return level * above, {"marker": rule["marker"], "level": round(level, 2), "raw": round(level * above, 2), "method": values["method"]}
    if mode == "saved":
        if rule["marker"] not in (levels or {}):
            raise ValueError(f"{Path(image).name}: no saved level for {rule['marker']}. Run Auto in the right panel or Detect "
                             "in the Background step first, or choose another cutoff.")
        level = float(levels[rule["marker"]])
        return level * above, {"marker": rule["marker"], "level": round(level, 2), "raw": round(level * above, 2)}
    # Robust z-score of arcsinh(intensity / cofactor) over the area's pixels: comparable between images.
    cofactor = values["cofactor"]
    pixels = np.arcsinh(data[inside] / cofactor) if inside.any() else np.arcsinh(data.ravel() / cofactor)
    middle = float(np.median(pixels))
    spread = float(np.median(np.abs(pixels - middle))) * 1.4826
    if not spread > 0:
        raise ValueError(f"{Path(image).name}: {rule['marker']} is flat inside the area, so it has no z-score.")
    raw = float(cofactor * np.sinh(middle + above * spread))
    return raw, {"marker": rule["marker"], "z": above, "raw": round(raw, 2), "median_raw": round(float(cofactor * np.sinh(middle)), 2)}


def describe(cutoffs):
    """The cutoffs used, in words: "CD31 above 312 (z 1.5)"."""
    parts = []
    for item in cutoffs:
        if "sure" in item:
            how = (f"two groups, {item['sure'] * 100:g}% sure, {item['fitted']} ROI{'s' if item['fitted'] != 1 else ''}, "
                   f"separation {item['separation']:g}" + ("; the groups overlap, a weak signal" if item["separation"] < 2 else ""))
        else:
            how = f"z {item['z']:g}" if "z" in item else (f"{item['method']} level {item['level']:g}" if "method" in item else
                                                        (f"saved level {item['level']:g}" if "level" in item else "raw"))
        parts.append(f"{item['marker']} above {item['raw']:g}" + ("" if how == "raw" else f" ({how})"))
    return "; ".join(parts)


def _grid(image, values, tissue):
    """The image at the chosen scale: its pyramid level, the grid, and the area (tissue or whole image) on it."""
    import rasterio.features
    from affine import Affine
    from shapely.geometry import box
    from shapely.ops import unary_union
    from spacemap.viewer.sources import image_info, open_source

    info = image_info(str(image))
    # Read a sufficiently fine pyramid plane, then use exactly the requested grid.
    level = max((i for i, size in enumerate(info.levels) if max(size.downsample_x, size.downsample_y) <= values['scale']), default=0)
    dx = dy = values['scale']
    shape = (math.ceil(info.height / dy), math.ceil(info.width / dx))
    transform = Affine(dx, 0, 0, 0, dy, 0)
    boundary = box(0, 0, info.width, info.height)
    shapes = []
    inside = np.ones(shape, bool)
    if values["within"] == "tissue":
        if not tissue or not Path(tissue).is_file():
            raise ValueError("No tissue for this image yet: detect it first (Tissue), or look in the whole image.")
        shapes = _shapes(tissue)
        boundary = unary_union(shapes).intersection(boundary)
        inside = (
            rasterio.features.rasterize(
                [(g, 1) for g in shapes], out_shape=shape, transform=transform, fill=0, dtype="uint8"
            ).astype(bool)
            if shapes
            else np.zeros(shape, bool)
        )
    return {"info": info, "names": list(info.channel_names), "level": level, "size": info.levels[level], "shape": shape,
            "transform": transform, "boundary": boundary, "shapes": shapes, "inside": inside, "source": open_source(str(image))}


def _plane(grid, channel, sigma):
    """One marker on the grid, smoothed."""
    import cv2
    from scipy import ndimage

    size = grid["size"]
    data = grid["source"].read_region(grid["level"], channel, 0, 0, size.width, size.height).astype(np.float32)
    if data.shape != grid["shape"]:
        data = cv2.resize(data, (grid["shape"][1], grid["shape"][0]), interpolation=cv2.INTER_AREA)
    if sigma > 0:
        data = ndimage.gaussian_filter(data, sigma)
    return data


_FITS = {}


def _stamp(path):
    try:
        found = Path(path).stat()
        return (found.st_mtime_ns, found.st_size)
    except (OSError, TypeError):
        return None


def _two_groups(x):
    """A two-group Gaussian mixture of 1-D values by expectation-maximisation, in plain numpy (scikit-learn's BLAS
    crashes in the server's worker threads): the best of three starts. Weights, means and standard deviations, the
    brighter group last."""
    fits = [_em(x, start) for start in ((30, 90), (50, 95), (20, 75))]
    return max(fits, key=lambda fit: fit[3])[:3]


def _em(x, start, rounds=500):
    weights = np.array([0.5, 0.5])
    means = np.percentile(x, start).astype(np.float64)
    spread = np.full(2, max(float(x.std()), 1e-3) / 2)
    previous = -np.inf
    for _ in range(rounds):
        density = weights / (spread * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x[:, None] - means) / spread) ** 2)
        total = density.sum(axis=1) + 1e-300
        share = density / total[:, None]
        counts = share.sum(axis=0) + 1e-12
        weights = counts / len(x)
        means = (share * x[:, None]).sum(axis=0) / counts
        spread = np.sqrt((share * (x[:, None] - means) ** 2).sum(axis=0) / counts).clip(1e-3)
        likelihood = float(np.log(total).sum())
        if likelihood - previous < 1e-7 * abs(likelihood):
            break
        previous = likelihood
    order = np.argsort(means)
    return weights[order], means[order], spread[order], likelihood


def _signal_share(fit, x):
    """How sure each value (rising) is to be in the brighter group. A brighter pixel is never less sure than a dimmer
    one, and one as bright as the signal group's middle is signal (a narrow signal group would otherwise hand the
    brightest pixels back to the background's wider tail)."""
    weights, means, spread = fit["weights"], fit["means"], fit["spread"]
    density = weights / spread * np.exp(-0.5 * ((x[:, None] - means) / spread) ** 2)
    sure = density[:, 1] / (density.sum(axis=1) + 1e-300)
    sure[x >= means[1]] = 1.0
    return np.maximum.accumulate(sure)


def fit_groups(pool, values, marker):
    """Two groups of a marker's pixels inside the areas of *pool* ([(image, area file)], one image or all ROIs of a
    slide): log intensities fitted by a two-group Gaussian mixture, the brighter group being the signal. Cached per
    files, marker, smoothing and scale."""
    key = (tuple((str(image), _stamp(image), str(area), _stamp(area)) for image, area in pool), marker,
           values["sigma"], values["scale"])
    fit = _FITS.get(key)
    if fit is not None:
        return fit
    rng = np.random.default_rng(0)
    share = max(SAMPLE // max(len(pool), 1), 20_000)
    parts, used = [], []
    for image, area in pool:
        grid = _grid(image, {**values, "within": "tissue" if area else "image"}, area)
        if marker not in grid["names"]:
            continue
        pixels = _plane(grid, grid["names"].index(marker), values["sigma"])[grid["inside"]]
        if pixels.size:
            parts.append(rng.choice(pixels, min(share, pixels.size), replace=False))
            used.append(Path(image).name)
    if not parts or sum(len(part) for part in parts) < 100:
        raise ValueError(f"{marker}: too few pixels inside the tissue to find two groups.")
    x = np.log1p(np.maximum(np.concatenate(parts), 0)).astype(np.float64)
    weights, means, spread = _two_groups(x)
    separation = (means[1] - means[0]) * math.sqrt(2) / math.sqrt(spread[0] ** 2 + spread[1] ** 2)
    fit = {"weights": weights, "means": means, "spread": spread, "from": float(means[0]), "to": float(x.max()),
           "separation": round(float(separation), 2), "signal_share": round(float(weights[1]), 3), "images": used}
    if len(_FITS) > 64:
        _FITS.clear()
    _FITS[key] = fit
    return fit


def groups_level(rule, pool, values):
    """The raw intensity above which a pixel is at least *rule['above']* sure to be in the signal group."""
    fit = fit_groups(pool, values, rule["marker"])
    grid = np.linspace(fit["from"], fit["to"], 4000)
    sure = _signal_share(fit, grid)
    hit = np.flatnonzero(sure >= rule["above"])
    if not hit.size:
        raise ValueError(f"{rule['marker']}: no pixel is {rule['above'] * 100:g}% sure to be signal. Lower How sure.")
    raw = float(np.expm1(grid[hit[0]]))
    return raw, {"marker": rule["marker"], "raw": round(raw, 2), "sure": rule["above"], "separation": fit["separation"],
                 "signal_share": fit["signal_share"], "fitted": len(fit["images"])}


def pool_for(root, mappings, image, values):
    """The ROIs a two-group fit uses for *image*: the image alone, or every image of its slide with an area (the
    Samples table, or the project's lineage, says which slide); None for the other cutoffs."""
    from . import annotation, annotation_inputs, samples

    if values["cutoff"] != "groups":
        return None
    output = annotation.all_folders(root, mappings)["output"]
    if values["pool"] == "image":
        return [(str(image), str(annotation_inputs.scope_path(root, image, output, "regions")))]
    try:
        frame, _ = samples._current(root, mappings)
        slide_of = {str(roi): str(slide) for roi, slide in zip(frame["roi"], frame["slide"]) if str(slide)}
    except (OSError, ValueError, KeyError):
        slide_of = {}
    slide = lambda path: slide_of.get(annotation.sample_id(Path(path).name)) or annotation.sample_id(Path(path).name)
    folder = Path(image).parent
    pool = []
    for path in sorted(folder.iterdir()):
        if not (path.is_file() and path.name.lower().endswith(annotation.IMAGE_EXTENSIONS)) or slide(path) != slide(image):
            continue
        try:
            area = annotation_inputs.scope_path(root, path, output, "regions")
        except (ValueError, OSError):
            continue
        if Path(area).is_file():
            pool.append((str(path), str(area)))
    if not any(Path(path).resolve() == Path(image).resolve() for path, _ in pool):
        pool.append((str(image), str(annotation_inputs.scope_path(root, image, output, "regions"))))
    return pool


def find(image, values, tissue=None, limit=5000, levels=None, pool=None):
    """Regions where every rule's marker (smoothed) is above its level, inside the tissue or the whole image, as
    GeoJSON features in image pixels (smallest first dropped below the size limit), top left first. The level is raw
    intensity, the image's Auto or saved level times a factor, a z-score of arcsinh intensities inside the area, or the
    two-group split of the area's pixels (of *pool*, the slide's ROIs, when given) (see CUTOFFS); each image gets its
    own, so one setting works across images."""
    import rasterio.features
    from scipy import ndimage
    from shapely.geometry import mapping, shape
    from .annotation_edits import polygons

    values = check(values)
    if not values["rules"]:
        raise ValueError("Add at least one marker and the level it must be above.")
    grid = _grid(image, values, tissue)
    info, names, inside = grid["info"], grid["names"], grid["inside"]
    missing = [rule["marker"] for rule in values["rules"] if rule["marker"] not in names]
    if missing:
        raise ValueError("Not a marker of this image: " + ", ".join(missing))
    dx = dy = values['scale']
    transform, boundary, shapes = grid["transform"], grid["boundary"], grid["shapes"]
    mask = inside.copy()
    cutoffs = []
    for rule in values["rules"]:
        channel = names.index(rule["marker"])
        data = _plane(grid, channel, values["sigma"])
        # The level comes from the area only (the clean tissue, once artifacts are subtracted from it).
        if values["cutoff"] == "groups":
            cutoff, used = groups_level(rule, pool or [(str(image), str(tissue) if tissue else None)], values)
        else:
            cutoff, used = _level(rule, data, inside, values, image, info, channel, shapes, levels)
        cutoffs.append(used)
        mask &= data > cutoff
    microns = info.pixel_size_um or 0.5
    smallest_px = values["min_area_um2"] / (microns * microns) / (dx * dy)  # at this level
    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        mask = np.isin(labels, np.flatnonzero(sizes >= max(smallest_px, 1)) + 1)
    features = []
    for geometry, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        polygon = shape(geometry).simplify(max(dx, 1.0) * 0.75).intersection(boundary)
        features.extend(p for p in polygons(polygon) if p.area * microns * microns >= values['min_area_um2'])
    features.sort(
        key=lambda p: (round(p.centroid.y / max(dy * 64, 1)), p.centroid.x)
    )  # rows top to bottom, then left to right
    if len(features) > limit:
        raise ValueError(
            f"{len(features):,} regions: too many to review. Raise the levels, the smoothing or the smallest size."
        )
    named = values["name"]
    out = []
    for number, polygon in enumerate(features, 1):
        label = named or f"Region {number}"
        properties = {
            "name": label,
            "label": label,
            "source": "marker_regions",
            "number": number,
            "classification": {"name": named or "Marker_Region", "color": [255, 214, 0]},
            "cluster_colour": REGION_COLOUR,
            "marker_rules": values['rules'],
            "region_settings": values,
            "source_annotation": str(tissue) if tissue else None,
        }
        if named:
            properties["named"] = True
        out.append({"type": "Feature", "geometry": mapping(polygon), "properties": properties})
    return out, {
        "level_microns": round(microns * dx, 2),
        "regions": len(out),
        "area_um2": round(sum(p.area for p in features) * microns * microns, 1),
        "scale": values['scale'], "grid_shape": list(mask.shape), "cutoffs": cutoffs,
    }


def write(path, features, expected=None):
    from . import annotation_edits

    found = annotation_edits.replace(
        path, features, expected if expected is not None else annotation_edits.revision(path), "Detected marker regions"
    )
    return len(found["features"])


def detect_many(items, report):
    done = []
    for number, item in enumerate(items):
        report(f"{Path(item['image']).name}: finding marker regions", number / len(items))
        features, stats = find(item["image"], item["values"], item.get("tissue"), levels=item.get("levels"), pool=item.get("pool"))
        write(item["target"], features, item.get("revision"))
        report(f"{Path(item['image']).name}: {stats['regions']} regions; {describe(stats['cutoffs'])}", (number + 1) / len(items))
        if item.get('root'):
            register_result(item)
        done.append({"image": Path(item["image"]).name, **stats})
    return done


def target(output, image, values, selection):
    """Separate named threshold results and results derived from another layer."""
    from hashlib import sha256
    from . import annotation
    if not values['name'] and selection.get('mode') != 'file':
        return annotation.marker_regions(output, annotation.sample_id(image))
    identity = json.dumps({'name': values['name'], 'parent': selection}, sort_keys=True)
    return Path(output) / (annotation.sample_id(image)+'_marker_regions_'+sha256(identity.encode()).hexdigest()[:12]+'.geojson')


def register_result(item):
    from . import annotation_layers
    annotation_layers.register(item['root'], item['image'], item['target'],
        name='Marker regions'+(' · '+item['values']['name'] if item['values']['name'] else ''),
        kind='regions', parent=item.get('selection'), settings=item['values'])


# ---------------------------------------------------------------- class regions (pixel classes as polygons)
def _hex(colour_rgb):
    """QuPath's packed colour (a signed integer) as #rrggbb."""
    value = int(colour_rgb) & 0xFFFFFF
    return "#%06x" % value


def group_of(properties):
    """A class region's group: its cluster, and its name when it was named apart from the cluster (a split)."""
    cluster = properties.get("cluster")
    return f"{cluster}|{properties['name']}" if properties.get("named") else str(cluster)


def class_regions(clusters_path, domains=None, smallest_px=0.0):
    """Tab 4's class outlines (one feature per class) as one polygon per connected piece, each with its cluster, its
    cluster's colour and, when the cluster is named into a domain, that name. Top left first."""
    from shapely.geometry import mapping, shape

    domains = {int(k): str(v) for k, v in (domains or {}).items()}
    data = json.loads(Path(clusters_path).read_text())
    pieces = []
    for feature in data.get("features", []) if isinstance(data, dict) else data:
        properties = feature.get("properties") or {}
        if properties.get("clusterId") is None or not feature.get("geometry"):
            continue
        cluster = int(properties["clusterId"])
        colour = _hex((properties.get("classification") or {}).get("colorRGB", -256))
        geometry = shape(feature["geometry"]).buffer(0)
        for part in getattr(geometry, "geoms", [geometry]):
            if part.geom_type == "Polygon" and not part.is_empty and part.area >= smallest_px:
                pieces.append((cluster, colour, part))
    height = max((p.bounds[3] - p.bounds[1] for _, _, p in pieces), default=1.0) or 1.0
    pieces.sort(key=lambda item: (int(item[2].centroid.y // height), item[2].centroid.x))
    features = []
    for number, (cluster, colour, polygon) in enumerate(pieces, 1):
        name = domains.get(cluster)
        properties = {
            "cluster": cluster,
            "cluster_colour": colour,
            "number": number,
            "source": "pixel_classes",
            "name": name or f"Cluster {cluster}",
            "label": name or f"Cluster {cluster}",
            "classification": {
                "name": name or f"Cluster_{cluster}",
                "color": [int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)],
            },
        }
        if name:
            properties["named"] = True
        features.append({"type": "Feature", "geometry": mapping(polygon), "properties": properties})
    return features


def group_profile(features, stack, factor):
    """The heatmap of class regions by group: each marker's mean z-score (the pixel classes' own features, over the
    tissue) in the pixels each group covers, and each group's share of those pixels."""
    import rasterio.features
    from affine import Affine
    from shapely.geometry import shape

    height, width = stack["shape"]
    groups, shapes = [], []
    for feature in features:
        properties = feature.get("properties") or {}
        if properties.get("cluster") is None or not feature.get("geometry"):
            continue
        key = group_of(properties)
        if key not in groups:
            groups.append(key)
        shapes.append((shape(feature["geometry"]), groups.index(key) + 1))
    if not shapes:
        return {"rows": [], "columns": list(stack["marker_names"]), "values": [], "share": []}
    raster = rasterio.features.rasterize(
        shapes, out_shape=(height, width), transform=Affine(factor, 0, 0, 0, factor, 0), fill=0, dtype="int32"
    )
    labels = raster.reshape(-1)[stack["valid_indices"]]
    z = stack["pixels_z_valid"]
    counts = np.bincount(labels, minlength=len(groups) + 1)
    values = []
    for index in range(1, len(groups) + 1):
        members = labels == index
        values.append(np.round(z[members].mean(axis=0), 3).tolist() if members.any() else [0.0] * z.shape[1])
    total = max(int(counts[1:].sum()), 1)
    return {
        "rows": groups,
        "columns": list(stack["marker_names"]),
        "values": values,
        "share": [round(100.0 * int(counts[i + 1]) / total, 2) for i in range(len(groups))],
        "note": "Mean of each marker (z-score over the tissue) in the pixels each group of regions covers",
    }
