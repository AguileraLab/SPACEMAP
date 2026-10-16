"""Studio's Annotation parts for artifacts: blood clots, blur, bright spots and lint, live.

Preparing an image reads Tab 4's signal maps once at a preview resolution
(about 2,000 pixels across) and scores it with Tab 4's own detectors. A slider
then only re-thresholds those scores with Tab 4's formulas and clean-up, which
takes milliseconds; ``tests/test_studio_artifacts.py`` checks the live masks
equal Tab 4's ``detect_artifacts`` at the same resolution and value. Blood
clots run Tab 4's ``detect_blood_clot`` itself, at half the preview resolution.
Applying runs Tab 4's steps 1 and 2 at their normal resolution and writes
Tab 4's files, the valid tissue with every artifact taken out.

Each slider is one of Tab 4's own settings, saved where Tab 4 keeps it, so
classic Tab 4 reproduces what Studio shows. Blur uses Tab 4's overall
sharpness test with a manual cut (blur_evidence_rule "pooled_only").
"""

from collections import OrderedDict
from pathlib import Path
import json
import threading
import time

import numpy as np

PREVIEW_SIDE = 2400  # about 2,000 px across: 1/8 of a 17,000 px ROI, ~4 um per pixel
# part -> (Tab 4 config, setting moved by the slider, switch, slider range, step)
PARTS = {
    "clot": ("annot_ta_tissue_config", "clot_eosin_ratio", "detect_blood_clot", (1.0, 4.0), 0.05),
    "blur": ("annot_ta_artifact_config", "blur_ratio_threshold", "detect_blur", (0.0, 1.0), 0.01),
    "bright": (
        "annot_ta_artifact_config",
        "bright_flat_mean_percentile",
        "detect_bright_artifacts",
        (90.0, 99.99),
        0.05,
    ),
    "lint": ("annot_ta_artifact_config", "lint_robust_z_threshold", "detect_filamentous_lint", (3.0, 40.0), 0.5),
}
COLOURS = {"clot": "#e6194b", "blur": "#ffff00", "bright": "#ff00ff", "lint": "#00ffff"}

_prepared = OrderedDict()
_lock = threading.Lock()


def _saved(config, key):
    value = ((config.get("tab_configurations") or {}).get("tab4_annotation") or {}).get(key)
    return dict(value) if isinstance(value, dict) else {}


def artifact_settings(config):
    from spacemap.tissue_annotation_utils import DEFAULT_ARTIFACT_CONFIG

    return {**DEFAULT_ARTIFACT_CONFIG, **_saved(config, "annot_ta_artifact_config")}


def values(config):
    """The four sliders and their switches, as Tab 4 has them saved."""
    from . import annotation

    tissue, artifact = annotation.settings(config), artifact_settings(config)
    saved = _saved(config, "annot_ta_artifact_config")
    result = {}
    for part, (where, key, switch, _, _) in PARTS.items():
        source = tissue if where == "annot_ta_tissue_config" else artifact
        result[part] = {"value": float(source[key]), "on": bool(source.get(switch, True))}
    # Blur has no switch in Tab 4; Studio stores it off as a zero cut.
    manual_blur = str(saved.get("blur_threshold_mode", artifact["blur_threshold_mode"])) == "manual"
    result["blur"]["on"] = not (manual_blur and float(artifact["blur_ratio_threshold"]) == 0.0)
    result["blur"]["auto"] = not manual_blur
    return result


def check(changes):
    if not isinstance(changes, dict) or set(changes) - set(PARTS):
        raise ValueError("Choose clot, blur, bright or lint.")
    for part, change in changes.items():
        low, high = PARTS[part][3]
        if not isinstance(change, dict) or set(change) - {"value", "on"}:
            raise ValueError(f"Send a value and on/off for {part}.")
        value = change.get("value")
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high
        ):
            raise ValueError(f"The {part} value must be between {low} and {high}.")
        if "on" in change and not isinstance(change["on"], bool):
            raise ValueError(f"Choose on or off for {part}.")
    return changes


def save(root, changes, expected):
    """Save the sliders into Tab 4's own settings, so classic Tab 4 reproduces them."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config, revision

    check(changes)
    with project_lock(root):
        config = read_config(root)
        if expected != revision(values(config)):
            raise ValueError("These settings changed in another window. Reload them before saving.")
        stored = {
            "annot_ta_tissue_config": _saved(config, "annot_ta_tissue_config"),
            "annot_ta_artifact_config": _saved(config, "annot_ta_artifact_config"),
        }
        current = values(config)
        for part, change in changes.items():
            where, key, switch, _, _ = PARTS[part]
            value = float(change.get("value", current[part]["value"]))
            on = change.get("on", current[part]["on"])
            if part == "blur":
                stored[where].update(
                    blur_threshold_mode="manual",
                    blur_evidence_rule="pooled_only",
                    blur_sensitivity=1.0,
                    blur_ratio_threshold=value if on else 0.0,
                )
            else:
                stored[where][key] = value
                stored[where][switch] = bool(on)
        for where, value in stored.items():
            persistence.update_stored_config_key(config, where, value)
        persistence.save_project_config(str(root), config)
        return values(read_config(root))


def _preview_downsample(reader, base):
    height, width = reader.full_shape
    downsample = max(int(base), 1)
    while max(height, width) / downsample > PREVIEW_SIDE:
        downsample *= 2
    return downsample


def _markers(names, config):
    """Tab 4 step 2's markers: the saved choice, else every marker that is not H&E."""
    from spacemap.tissue_annotation_utils import split_he_and_codex_channels

    he, codex = split_he_and_codex_channels(names)
    chosen = [
        names.index(n)
        for n in ((config.get("tab_configurations") or {}).get("tab4_annotation") or {}).get(
            "annot_ta_artifact_markers"
        )
        or []
        if n in names
    ]
    return list(he), chosen or list(codex)


def prepare(image, config, boundary=None, *, scale=None):
    """Tab 4's maps and scores for *image* at the preview resolution; cached per image and settings."""
    from spacemap.tissue_annotation_utils import (
        PyramidImageReader,
        detect_artifacts,
        detect_tissue_mask,
        load_geojson_mask,
    )
    from spacemap.viewer.sources import fingerprint

    from . import annotation

    tissue_cfg = annotation.settings(config)
    user = _saved(config, "annot_ta_artifact_config")
    signature = json.dumps(
        {
            "image": str(image),
            "scale": scale,
            "stamp": fingerprint(image),
            "tissue": {k: v for k, v in tissue_cfg.items() if not k.startswith("clot_") and k != "detect_blood_clot"},
            "artifact": {
                k: v for k, v in user.items() if not k.startswith(("blur_ratio", "bright_flat_mean", "lint_robust"))
            },
            "boundary": Path(boundary).stat().st_mtime_ns if boundary and Path(boundary).is_file() else None,
        },
        sort_keys=True,
        default=str,
    )
    with _lock:
        if signature in _prepared:
            _prepared.move_to_end(signature)
            return _prepared[signature]
    started = time.perf_counter()
    reader = PyramidImageReader(image)
    names = list(reader.channel_names)
    he, markers = _markers(names, config)
    tissue_he, _ = annotation.channels(names, config)
    downsample = int(scale) if scale is not None else _preview_downsample(reader, tissue_cfg.get("blur_downsample", 2))
    maps = detect_tissue_mask(
        reader,
        {**tissue_cfg, "blur_downsample": downsample, "detect_blood_clot": False},
        he_indices=tissue_he or he,
        codex_indices=markers,
        need_codex_maps=True,
    )
    tissue = maps["tissue_mask"]
    if boundary and Path(boundary).is_file():
        # Tab 4 step 2 reloads the boundary from disk so corrections made in QuPath count.
        loaded = load_geojson_mask(str(boundary), tissue.shape, float(downsample))
        if loaded.any():
            tissue = loaded
    inputs = {
        "tissue_mask": tissue,
        "codex_projection": maps["codex_projection"],
        "he_gray": maps["he_gray"],
        "codex_robust_max": maps["codex_robust_max"],
        "channel_positive_count": maps["channel_positive_count"],
        "n_codex_channels": len(markers),
        "microns_per_pixel": float(tissue_cfg.get("microns_per_pixel", 0.5083)),
        "downsample": downsample,
        "border_px": int(maps.get("border_px", 0)),
        "channel_focus_grid": maps.get("channel_focus_grid"),
        "channel_signal_grid": maps.get("channel_signal_grid"),
    }
    settings = artifact_settings(config)
    automatic = detect_artifacts(
        config={
            **settings,
            "blur_threshold_mode": "auto",
            "write_review_manifest": False,
            "detect_filamentous_lint": False,
            "detect_bright_artifacts": False,
        },
        **inputs,
    )
    prepared = {
        "inputs": inputs,
        "user": user,
        "settings": settings,
        "tissue_cfg": tissue_cfg,
        "he_rgb": maps.get("he_rgb"),
        "downsample": downsample,
        "shape": tissue.shape,
        "codex_focus": automatic["codex_focus"],
        "evaluate": automatic["evaluate_mask"],
        "auto_blur": float(automatic["stats"]["blur_threshold_used"]),
        "blur_found": bool(automatic["stats"]["blur_population_found"]),
        "seconds": round(time.perf_counter() - started, 1),
    }
    prepared.update(_blur_scores(prepared))
    prepared.update(_bright_scores(prepared))
    with _lock:
        _prepared[signature] = prepared
        while len(_prepared) > 2:
            _prepared.popitem(last=False)
    return prepared


def _lengths(prepared):
    from spacemap.tissue_annotation_utils import _scale_um

    pixel_um = prepared["inputs"]["microns_per_pixel"] * prepared["downsample"]

    def length_px(key):
        return _scale_um(prepared["user"], key, pixel_um) / pixel_um

    def area_px(key):
        return int(np.ceil(float(prepared["settings"][key]) / pixel_um**2))

    return pixel_um, length_px, area_px


def _blur_scores(prepared):
    """Tab 4's pooled sharpness arms (see detect_artifacts), kept for live cuts."""
    from spacemap.tissue_annotation_utils import _focus_map, _odd_kernel

    settings, user = prepared["settings"], prepared["user"]
    _, length_px, _ = _lengths(prepared)
    tissue, evaluate = prepared["inputs"]["tissue_mask"], prepared["evaluate"]
    codex_focus, he_gray = prepared["codex_focus"], prepared["inputs"]["he_gray"]
    he_focus = (
        _focus_map(
            he_gray,
            pre_sigma_px=max(length_px("noise_smoothing_um"), 0.0),
            ksize=_odd_kernel(length_px("focus_detail_um"), 1, 31),
            window_px=_odd_kernel(length_px("focus_window_um"), 3, 999),
            post_sigma_px=max(length_px("region_smoothing_um"), 0.0),
        )
        if he_gray is not None
        else None
    )
    structure = None
    if he_focus is not None:
        inside = he_focus[tissue]
        floor = (
            float(np.percentile(inside, float(settings.get("he_structure_percentile", 15.0)))) if inside.size else 0.0
        )
        structure = he_focus >= floor
    if "blur_reference_mode" in user:
        mode = str(user["blur_reference_mode"])
    elif "use_he_reference" in user:
        mode = "he" if bool(user["use_he_reference"]) else "self"
    else:
        mode = str(settings["blur_reference_mode"])
    if he_focus is None and mode in ("he", "he_and_self", "he_or_self"):
        mode = "self"
    percentile = float(
        user.get("sharp_reference_percentile")
        or user.get("self_reference_percentile")
        or settings["sharp_reference_percentile"]
    )
    domain = evaluate if evaluate.any() else tissue

    def anchored(score):
        inside = score[domain]
        return score / max(float(np.percentile(inside, percentile)) if inside.size else 0.0, 1e-12)

    ratio_he = ratio_self = None
    if mode in ("he", "he_and_self", "he_or_self"):
        inside = he_focus[domain]
        he_floor = max(float(np.percentile(inside, 5.0)) if inside.size else 0.0, 1e-8)
        ratio_he = anchored(codex_focus / np.maximum(he_focus, he_floor))
    if mode in ("self", "he_and_self", "he_or_self"):
        ratio_self = anchored(codex_focus)
    return {"ratio_he": ratio_he, "ratio_self": ratio_self, "he_structure": structure, "blur_mode": mode}


def _bright_scores(prepared):
    import cv2

    from spacemap.tissue_annotation_utils import _odd_kernel

    _, length_px, _ = _lengths(prepared)
    window = _odd_kernel(length_px("focus_window_um"), 3, 999)
    local_mean = cv2.boxFilter(
        np.asarray(prepared["inputs"]["codex_projection"], dtype=np.float32),
        ddepth=-1,
        ksize=(window, window),
        borderType=cv2.BORDER_REFLECT,
    )
    tissue = prepared["inputs"]["tissue_mask"]
    return {
        "local_mean": local_mean,
        "mean_values": local_mean[tissue],
        "focus_values": prepared["codex_focus"][tissue],
    }


def blur_mask(prepared, cut):
    """Tab 4's blur at a manual cut, overall sharpness test only (as Studio saves it)."""
    import cv2

    from spacemap.tissue_annotation_utils import _disk_kernel, _remove_small_components

    tissue, evaluate = prepared["inputs"]["tissue_mask"], prepared["evaluate"]
    if cut <= 0:
        return np.zeros_like(tissue, dtype=bool)
    _, length_px, area_px = _lengths(prepared)
    threshold = float(cut) * float(prepared["settings"].get("blur_sensitivity", 1.0))
    arms = []
    if prepared["ratio_he"] is not None:
        arm = (prepared["ratio_he"] < threshold) & evaluate
        if prepared["he_structure"] is not None:
            arm &= prepared["he_structure"]
        arms.append(arm)
    if prepared["ratio_self"] is not None:
        arms.append((prepared["ratio_self"] < threshold) & evaluate)
    if not arms:
        return np.zeros_like(tissue, dtype=bool)
    seed = (
        arms[0] & arms[1] if prepared["blur_mode"] == "he_and_self" and len(arms) == 2 else np.logical_or.reduce(arms)
    )
    mask = seed.astype(np.uint8)
    close_px, open_px = int(round(length_px("fill_holes_um"))), int(round(length_px("remove_specks_um")))
    if close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk_kernel(close_px))
    if open_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disk_kernel(open_px))
    core = _remove_small_components((mask > 0) & tissue, area_px("min_area_microns2"), connectivity=8)
    pixel_um = prepared["inputs"]["microns_per_pixel"] * prepared["downsample"]
    expand = int(np.ceil(float(prepared["settings"]["expand_microns"]) / pixel_um))
    if float(prepared["settings"]["expand_microns"]) > 0 and expand > 0 and core.any():
        core = (cv2.dilate(core.astype(np.uint8), _disk_kernel(expand)) > 0) & tissue
    return core


def bright_mask(prepared, percentile):
    """Tab 4's flat bright glare at a brightness percentile."""
    import cv2

    from spacemap.tissue_annotation_utils import _disk_kernel, _remove_small_components

    settings = prepared["settings"]
    tissue = prepared["inputs"]["tissue_mask"]
    _, length_px, area_px = _lengths(prepared)
    means, focus = prepared["mean_values"], prepared["focus_values"]
    breadth = np.zeros_like(tissue, dtype=bool)
    count = prepared["inputs"]["channel_positive_count"]
    if bool(settings["use_breadth_signal"]) and count is not None and prepared["inputs"]["n_codex_channels"] > 0:
        share = count.astype(np.float32) / float(prepared["inputs"]["n_codex_channels"])
        breadth = (share >= float(settings["bright_breadth_fraction"])) & tissue
    spread = float(np.percentile(means, 99.0) - np.percentile(means, 1.0)) if means.size else 0.0
    if spread <= 1e-6:
        glare = np.zeros_like(tissue, dtype=bool)
    else:
        glare = (
            (prepared["local_mean"] >= float(np.percentile(means, float(percentile))))
            & (
                prepared["codex_focus"]
                <= float(np.percentile(focus, float(settings["bright_flat_texture_percentile"])))
            )
            & tissue
        )
    mask = (breadth | glare).astype(np.uint8)
    close_px, open_px = int(round(length_px("bright_fill_holes_um"))), int(round(length_px("bright_remove_specks_um")))
    if close_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _disk_kernel(close_px))
    if open_px > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _disk_kernel(open_px))
    core = _remove_small_components((mask > 0) & tissue, area_px("min_area_microns2_bright"), connectivity=8)
    pixel_um = prepared["inputs"]["microns_per_pixel"] * prepared["downsample"]
    expand = int(np.ceil(float(settings["bright_expand_microns"]) / pixel_um))
    if float(settings["bright_expand_microns"]) > 0 and expand > 0 and core.any():
        core = (cv2.dilate(core.astype(np.uint8), _disk_kernel(expand)) > 0) & tissue
    return core


def lint_mask(prepared, z, *, domain=None):
    """Tab 4's filamentous lint on the glass at a brightness (robust z) cut; see detect_artifacts."""
    import cv2
    from skimage.morphology import skeletonize

    from spacemap.tissue_annotation_utils import _disk_kernel

    settings = prepared["settings"]
    inputs = prepared["inputs"]
    tissue, robust = inputs["tissue_mask"], inputs["codex_robust_max"]
    lint = np.zeros_like(tissue, dtype=bool)
    if robust is None:
        return lint
    pixel_um, length_px, _ = _lengths(prepared)
    pixel_area = pixel_um**2
    exclusion = int(np.ceil(float(settings["lint_he_exclusion_margin_um"]) / pixel_um))
    outside = ~(cv2.dilate(tissue.astype(np.uint8), _disk_kernel(exclusion)) > 0)
    # Studio's combined review can inspect lint on tissue as well as glass.
    # Classic Tab 4 retains its outside-tissue behavior when no domain is given.
    candidate = (robust >= float(z)) & (outside if domain is None else domain)
    border = inputs["border_px"]
    if border > 0:
        candidate[:border, :] = candidate[-border:, :] = False
        candidate[:, :border] = candidate[:, -border:] = False
    candidate = candidate.astype(np.uint8)
    close_px, open_px = int(round(length_px("lint_join_gaps_um"))), int(round(length_px("lint_remove_specks_um")))
    if close_px > 0:
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, _disk_kernel(close_px))
    if open_px > 0:
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, _disk_kernel(open_px))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    aspect_min = float(settings["lint_min_aspect_ratio"])
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label, :5])
        area_um2 = area * pixel_area
        if area_um2 < float(settings["lint_min_area_microns2"]) or area_um2 > float(settings["lint_max_area_microns2"]):
            continue
        roi = labels[y : y + h, x : x + w] == label
        contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        side_a, side_b = cv2.minAreaRect(max(contours, key=cv2.contourArea))[1]
        major, minor = max(side_a, side_b, 1.0), max(min(side_a, side_b), 1.0)
        skeleton = float(skeletonize(roi).sum())
        width_px = area / max(skeleton, 1.0)
        length_um = max(major, skeleton) * pixel_um
        width_um = width_px * pixel_um
        seam = w / max(h, 1) >= float(settings["lint_seam_min_span_ratio"]) and width_um <= float(
            settings["lint_seam_max_width_microns"]
        )
        if (
            length_um >= float(settings["lint_min_length_microns"])
            and float(settings["lint_min_width_microns"]) <= width_um <= float(settings["lint_max_width_microns"])
            and major / minor >= aspect_min
            and skeleton / max(width_px, 1.0) >= aspect_min
            and not seam
        ):
            lint[y : y + h, x : x + w][roi] = True
    expand = int(np.ceil(float(settings["lint_expand_microns"]) / pixel_um))
    if float(settings["lint_expand_microns"]) > 0 and expand > 0 and lint.any():
        lint = cv2.dilate(lint.astype(np.uint8), _disk_kernel(expand)) > 0
    return lint


def clot_mask(prepared, ratio):
    """Tab 4's blood-clot detector at an eosin ratio, at half the preview resolution."""
    import cv2

    from spacemap.tissue_annotation_utils import detect_blood_clot

    he_rgb = prepared["he_rgb"]
    tissue = prepared["inputs"]["tissue_mask"]
    if he_rgb is None:
        return np.zeros_like(tissue, dtype=bool), 2
    small = cv2.resize(
        he_rgb, (max(he_rgb.shape[1] // 2, 1), max(he_rgb.shape[0] // 2, 1)), interpolation=cv2.INTER_AREA
    )
    small_tissue = (
        cv2.resize(tissue.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    )
    found = detect_blood_clot(
        small,
        small_tissue,
        {**prepared["tissue_cfg"], "clot_eosin_ratio": float(ratio)},
        prepared["inputs"]["microns_per_pixel"],
        prepared["downsample"] * 2,
    )
    return found["clot_mask"], 2


def _clot_scores(prepared):
    """Tab 4's eosin ratio (eosin over its tissue reference) at the clot preview resolution."""
    import cv2

    from spacemap.tissue_annotation_utils import detect_blood_clot

    if "clot_ratio" not in prepared:
        he_rgb, tissue = prepared["he_rgb"], prepared["inputs"]["tissue_mask"]
        if he_rgb is None:
            raise ValueError("This image has no H&E layers, so blood clots cannot be measured.")
        small = cv2.resize(
            he_rgb, (max(he_rgb.shape[1] // 2, 1), max(he_rgb.shape[0] // 2, 1)), interpolation=cv2.INTER_AREA
        )
        inside = (
            cv2.resize(tissue.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        )
        found = detect_blood_clot(
            small, inside, prepared["tissue_cfg"], prepared["inputs"]["microns_per_pixel"], prepared["downsample"] * 2
        )
        eosin = np.asarray(found["eosin"], dtype=np.float32)
        percentile = float(prepared["tissue_cfg"].get("clot_eosin_reference_percentile", 25.0))
        reference = max(float(np.percentile(eosin[inside], percentile)) if inside.any() else 0.0, 1e-9)
        prepared["clot_ratio"], prepared["clot_tissue"] = eosin / reference, inside
    return prepared["clot_ratio"], prepared["clot_tissue"]


def _raster(polygons, shape, scale):
    import cv2

    mask = np.zeros(shape, np.uint8)
    for ring in polygons or []:
        points = np.round(np.asarray(ring, dtype=np.float64) / scale).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask > 0


def suggest(image, config, part, examples, clean=None, boundary=None):
    """A slider value that catches the drawn example, using the same score the slider moves.

    *examples* are polygons over the artifact, *clean* optional polygons over good
    tissue (or clean glass for lint); with both, the cut is placed between them.
    """
    import cv2

    if part not in PARTS:
        raise ValueError("Choose clot, blur, bright or lint.")
    prepared = prepare(image, config, boundary)
    tissue = prepared["inputs"]["tissue_mask"]
    scale = prepared["downsample"]
    low_is_artifact = False
    if part == "blur":
        arms = [arm for arm in (prepared["ratio_he"], prepared["ratio_self"]) if arm is not None]
        if not arms:
            raise ValueError("Blur cannot be measured here.")
        combine = np.maximum if prepared["blur_mode"] == "he_and_self" and len(arms) == 2 else np.minimum
        score = combine.reduce(arms) if len(arms) > 1 else arms[0]
        score = score / float(prepared["settings"].get("blur_sensitivity", 1.0))
        domain, low_is_artifact, where = prepared["evaluate"], True, "tissue with marker signal"
    elif part == "bright":
        score, domain, where = prepared["local_mean"], tissue, "tissue"
    elif part == "lint":
        exclusion = int(
            np.ceil(
                float(prepared["settings"]["lint_he_exclusion_margin_um"])
                / (prepared["inputs"]["microns_per_pixel"] * scale)
            )
        )
        from spacemap.tissue_annotation_utils import _disk_kernel

        score = prepared["inputs"]["codex_robust_max"]
        domain = ~(cv2.dilate(tissue.astype(np.uint8), _disk_kernel(exclusion)) > 0)
        where = "the glass outside the tissue"
    else:
        score, domain = _clot_scores(prepared)
        scale, where = scale * 2, "tissue"
    marked = _raster(examples, score.shape, scale) & domain
    if not marked.any():
        raise ValueError(f"Draw the example over {where}.")
    values = score[marked]
    example = float(np.percentile(values, 75 if low_is_artifact else 25))
    cut = example
    if clean:
        good = _raster(clean, score.shape, scale) & domain
        if good.any():
            other = float(np.percentile(score[good], 10 if low_is_artifact else 90))
            if (other > example) if low_is_artifact else (other < example):
                cut = (example + other) / 2
    if part == "bright":
        cut = 100.0 * float((prepared["mean_values"] < cut).mean())
    low, high = PARTS[part][3]
    step = PARTS[part][4]
    value = float(np.clip(round(cut / step) * step, low, high))
    caught_mask = {"blur": blur_mask, "bright": bright_mask, "lint": lint_mask}.get(part)
    mask = caught_mask(prepared, value) if caught_mask else clot_mask(prepared, value)[0]
    return {
        "part": part,
        "value": round(value, 4),
        "example_pixels": int(marked.sum()),
        "caught": round(float(mask[marked].mean()), 3) if mask.shape == marked.shape else None,
    }


def rings(mask, scale, limit=4000):
    """Outlines (with holes) of a mask, in full-resolution image pixels."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:limit]:
        if len(contour) < 3:
            continue
        simple = cv2.approxPolyDP(contour, 0.8, True)[:, 0, :]
        if len(simple) >= 3:
            polygons.append([[round(float(x) * scale, 1), round(float(y) * scale, 1)] for x, y in simple])
    return polygons


def preview(image, config, part, value, boundary=None):
    """One part's live mask for a slider value, as outlines and its share of the tissue."""
    import cv2

    if part not in PARTS:
        raise ValueError("Choose clot, blur, bright or lint.")
    before = time.perf_counter()
    prepared = prepare(image, config, boundary)
    prepared_now = time.perf_counter() - before > 1.0
    started = time.perf_counter()
    scale = prepared["downsample"]
    if part == "blur":
        mask = blur_mask(prepared, value)
    elif part == "bright":
        mask = bright_mask(prepared, value)
    elif part == "lint":
        mask = lint_mask(prepared, value)
    else:
        mask, factor = clot_mask(prepared, value)
        scale *= factor
    tissue = prepared["inputs"]["tissue_mask"]
    if mask.shape != tissue.shape:
        tissue = (
            cv2.resize(tissue.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        )
    inside = mask & tissue
    count = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    return {
        "part": part,
        "outlines": rings(mask, scale),
        "percent": round(100.0 * float(inside.sum()) / max(int(tissue.sum()), 1), 2),
        "regions": int(count),
        "ms": round(1000 * (time.perf_counter() - started)),
        "prepared_seconds": prepared["seconds"] if prepared_now else None,
        "auto_blur": prepared["auto_blur"] if part == "blur" else None,
        "blur_found": prepared["blur_found"] if part == "blur" else None,
        "colour": COLOURS[part],
        "tissue_outline": rings(prepared["inputs"]["tissue_mask"], prepared["downsample"]) if part == "blur" else None,
    }


def apply(image, config, output, sample, report=None):
    """Tab 4 steps 1 and 2 at their resolution with the saved sliders; writes Tab 4's files."""
    from spacemap.tissue_annotation_utils import (
        PyramidImageReader,
        detect_artifacts,
        detect_tissue_mask,
        mask_to_geojson_features,
        qupath_color_from_hex,
        region_features_from_labels,
        write_feature_collection,
    )
    from spacemap_gui.tabs.tab4_annotation.s2_artifact_removal import ARTIFACT_EXPORTS

    from . import annotation

    tissue_cfg = annotation.settings(config)
    settings = artifact_settings(config)
    reader = PyramidImageReader(image)
    names = list(reader.channel_names)
    he, markers = _markers(names, config)
    tissue_he, _ = annotation.channels(names, config)
    if report:
        report(f"{sample}: reading the markers")
    maps = detect_tissue_mask(
        reader,
        {**tissue_cfg, "sample_id": sample},
        he_indices=tissue_he or he,
        codex_indices=markers,
        need_codex_maps=True,
    )
    downsample = float(tissue_cfg.get("blur_downsample", 2))
    simplify = float(tissue_cfg.get("boundary_simplify_px", 2.0))
    output = Path(output)
    (output / annotation.INTERMEDIATE).mkdir(parents=True, exist_ok=True)
    written = []
    # Step 1: the boundary (clots excluded when that is on) and the clot layer.
    boundary_features = mask_to_geojson_features(
        maps["tissue_mask"],
        scale_factor=downsample,
        class_name="Tissue_Boundary",
        color_rgb=-16711936,
        simplify_px=simplify,
        merge=False,
        locked=True,
    )
    write_feature_collection(boundary_features, annotation.boundary(output, sample))
    written.append(str(annotation.boundary(output, sample)))
    if tissue_cfg.get("detect_blood_clot"):
        clot = maps.get("clot") or {}
        regions, labels = clot.get("regions") or [], clot.get("region_labels")
        features = (
            region_features_from_labels(
                labels,
                regions,
                downsample,
                "Blood_Clot",
                color_rgb=int(qupath_color_from_hex(COLOURS["clot"])),
                simplify_px=simplify,
            )
            if labels is not None and regions
            else []
        )
        write_feature_collection(features, annotation.clot(output, sample))
        written.append(str(annotation.clot(output, sample)))
    if report:
        report(f"{sample}: finding blur, bright spots and lint")
    # Step 2: artifacts, and the valid tissue with them taken out.
    artifacts = detect_artifacts(
        tissue_mask=maps["tissue_mask"],
        codex_projection=maps["codex_projection"],
        he_gray=maps["he_gray"],
        codex_robust_max=maps["codex_robust_max"],
        channel_positive_count=maps["channel_positive_count"],
        n_codex_channels=len(markers),
        config={**settings, "write_review_manifest": False},
        microns_per_pixel=float(tissue_cfg.get("microns_per_pixel", 0.5083)),
        downsample=int(downsample),
        border_px=int(maps.get("border_px", 0)),
        marker_names=[names[i] for i in markers],
        channel_focus_grid=maps.get("channel_focus_grid"),
        channel_signal_grid=maps.get("channel_signal_grid"),
        sample_id=sample,
    )
    for mask_key, suffix, class_name, hex_color, tolerance, region_key in ARTIFACT_EXPORTS:
        path = (
            Path(annotation.valid_tissue(output, sample))
            if suffix == "valid_tissue"
            else output / annotation.INTERMEDIATE / f"{sample}_{suffix}.geojson"
        )
        color = int(qupath_color_from_hex(hex_color))
        simplify_px = tolerance if tolerance is not None else float(settings.get("boundary_simplify_px", 2.0))
        if region_key is not None:
            features = region_features_from_labels(
                artifacts["region_labels"][region_key],
                artifacts["regions"][region_key],
                downsample,
                class_name,
                color_rgb=color,
                simplify_px=simplify_px,
            )
        else:
            mask = artifacts.get(mask_key)
            features = (
                mask_to_geojson_features(
                    mask, downsample, class_name, color_rgb=color, simplify_px=simplify_px, merge=True
                )
                if mask is not None and mask.any()
                else []
            )
        write_feature_collection(features, path)
        written.append(str(path))
    annotation.subtract_drawn(output, sample)  # the artifacts drawn by hand too
    stats = {key: value for key, value in artifacts["stats"].items() if isinstance(value, (int, float, bool, str))}
    return {"written": written, "stats": stats}


def detect_editable(image, config, output, sample, expected, report=None):
    """Run classic detectors in a scratch folder; keep their polygons without changing tissue."""
    import tempfile

    from . import annotation, annotation_edits

    target = annotation.drawn_artifacts(output, sample)
    previous, _ = annotation_edits.read(target, kind="artifacts")
    # Shapes drawn by hand survive a new detection. Old automatic results are replaced.
    manual = [f for f in previous if f["properties"].get("source") != "auto_artifact"]
    with tempfile.TemporaryDirectory(prefix="spacemap-artifacts-") as temporary:
        found = apply(image, config, temporary, sample, report)
        automatic = []
        for name in found["written"]:
            path = Path(name)
            if path.name.endswith(("_valid_tissue.geojson", "_tissue_boundary.geojson", "_detected_artifacts.geojson")):
                continue  # union file duplicates the individual artifact masks
            for feature in annotation_edits.read(path, kind="artifacts")[0]:
                feature["properties"]["source"] = "auto_artifact"
                automatic.append(feature)
    result = annotation_edits.replace(target, [*manual, *automatic], expected, "Detected artifacts", kind="artifacts")
    return {"sample": sample, "regions": len(result["features"]), "files": [str(target)]}
