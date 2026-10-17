"""Studio's Registration step: Tab 1's CODEX and H&E pairs and their alignment.

Pairs are found as Tab 1 finds them, with Tab 1's saved choices: its naming rule (a prefix and suffix
stripped from each file name, e.g. ``P11_codex`` and ``P11_he`` both become ``P11``) and, for each CODEX
image, Auto, Skip or any H&E image. A pair is registered by Tab 1's own steps: prepare the H&E at the
CODEX resolution, extract the CODEX reference (anchor) channel, align with palom, and merge both into
``<registered>/<sample>_merged.ome.tiff``.
"""

import os
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".qptiff", ".svs")
CODEX_MPP, HE_MPP = 0.5083, 0.2495  # Tab 1's fallbacks when a file records no pixel size
KEYPOINTS = 10000
INTERMEDIATE = "intermediate_images"
PREVIOUS = "previous"  # a pair registered again: its earlier result is moved here

# Tab 1's naming rule, with its defaults, and its per-CODEX-image choice.
RULE = {
    "tab1_batch_codex_prefix": "",
    "tab1_batch_codex_suffix": "_codex",
    "tab1_batch_he_prefix": "",
    "tab1_batch_he_suffix": "_he",
}
OVERRIDE = "tab1_pair_override_"
EXTRA_DIRS = "tab1_extra_codex_dirs"  # more CODEX folders: CODEX (2), CODEX (3), ... aligned to CODEX (1)
EXTRA_CHOICES = "tab1_extra_codex_choices"  # {CODEX (1) file: [<Auto>, <Skip> or a file, one per extra folder]}
PAIR_MARKERS = "tab1_pair_markers"  # {CODEX (1) file: {"images": [per CODEX image], "he": {...}}}
HE_CHANNELS = (("HE_Red", "#ff0000"), ("HE_Green", "#00ff00"), ("HE_Blue", "#0000ff"))
AUTO, SKIP = "<Auto>", "<Skip>"
FOLDER_KEYS = {
    "codex": ("batch_codex_dir",),
    "he": ("batch_he_dir",),
    "registered": ("tab1_batch_output_dir", "registration_output_dir"),
}
FOLDER_DEPTH = 4


def _path(root, value):
    return Path(str(value).replace("${PROJECT_ROOT}", str(root))).expanduser()


def _mapped(root, mappings, keys, fallback):
    value = next((mappings[key] for key in keys if (mappings or {}).get(key)), None)
    return _path(root, value) if value else Path(root) / fallback


def folders(root, mappings):
    """(CODEX, H&E, registered) folders as Tab 1 uses them."""
    return (
        _mapped(root, mappings, ("batch_codex_dir",), "data/codex"),
        _mapped(root, mappings, ("batch_he_dir",), "data/he"),
        _mapped(root, mappings, ("tab1_batch_output_dir", "registration_output_dir"), "outputs/registered"),
    )


def _tab1(config):
    return ((config or {}).get("tab_configurations") or {}).get("tab1_registration") or {}


def rule(config):
    saved = _tab1(config)
    return {key: str(saved[key]) if saved.get(key) is not None else default for key, default in RULE.items()}


def _images(folder):
    from spacemap.background_subtraction_utils import is_supported_image_file

    if not folder.is_dir():
        return []
    return sorted(entry.name for entry in os.scandir(folder) if entry.is_file() and is_supported_image_file(entry.name))


def plan(root, mappings, config):
    """Tab 1's pairs for the project (Tab 1's ``find_pairs``, with its saved naming rule and choices).

    Each row: the sample name, the CODEX file, the H&E file (None when skipped, unmatched or a duplicate
    sample name), the automatic match, the saved choice, whether the name follows the rule, whether
    another CODEX image gives the same sample, and whether its merged image exists.
    """
    from spacemap_gui.tabs.tab1_registration import find_pairs

    codex_dir, he_dir, registered = folders(root, mappings)
    chosen_rule = rule(config)
    saved = _tab1(config)
    codex_names, he_names = _images(codex_dir), _images(he_dir)
    same = codex_dir.is_dir() and he_dir.is_dir() and codex_dir.resolve() == he_dir.resolve()
    choices = {name: saved[OVERRIDE + name] for name in codex_names if isinstance(saved.get(OVERRIDE + name), str)}
    found = find_pairs(codex_names, he_names, *(chosen_rule[key] for key in RULE), choices, same_folder=same)
    from spacemap_gui.tabs.tab1_registration import _build_role_file_map

    extra_dirs = extra_folders(root, config)
    extra_files = [_images(folder) for folder in extra_dirs]
    extra_maps = [_build_role_file_map(names, *(chosen_rule[key] for key in RULE)) for names in extra_files]
    extra_saved = saved.get(EXTRA_CHOICES) or {}
    edited = saved.get(PAIR_MARKERS) or {}
    for row in found["rows"]:
        output = output_for(registered, row["sample"])
        row.update(output=output.name, registered=output.exists(), edited=row["codex"] in edited)
        chosen = list(extra_saved.get(row["codex"]) or [])
        row["extra"] = []
        for k, (names, mapped) in enumerate(zip(extra_files, extra_maps)):
            choice = chosen[k] if k < len(chosen) and (chosen[k] in (AUTO, SKIP) or chosen[k] in names) else AUTO
            auto = mapped.get(row["sample"])
            row["extra"].append({"auto": auto, "choice": choice, "file": None if choice == SKIP else auto if choice == AUTO else choice})
    listed = {row["codex"] for row in found["rows"]}
    return {
        "folders": {kind: str(folder) for kind, folder in zip(FOLDER_KEYS, (codex_dir, he_dir, registered))},
        "found": {"codex": codex_dir.is_dir(), "he": he_dir.is_dir()},
        "rule": chosen_rule,
        "rows": found["rows"],
        "he_files": found["he_files"],
        "unpaired_he": found["unpaired_he"],
        "ignored_codex": [name for name in codex_names if name not in listed],
        "ignored_he": [name for name in he_names if name not in found["he_files"]],
        "extra_folders": [str(folder) for folder in extra_dirs],
        "extra_files": extra_files,
    }


def extra_folders(root, config):
    base = Path(root)
    return [Path(str(value).replace("${PROJECT_ROOT}", str(base))) for value in (_tab1(config).get(EXTRA_DIRS) or []) if value]


def pairing_revision(found):
    """What another window must not have changed before a save: the folders, the rule and the choices."""
    from .settings import revision

    return revision({"folders": found["folders"], "rule": found["rule"], "extra": found.get("extra_folders", []),
                     "choices": [(row["codex"], row["choice"], [item["choice"] for item in row.get("extra", [])]) for row in found["rows"]]})


def pairs(root, mappings, config=None):
    """sample -> {"codex": path, "he": path} for every pair Tab 1 would register."""
    from .settings import read_config

    config = read_config(root) if config is None else config
    codex_dir, he_dir, _ = folders(root, mappings)
    found = plan(root, mappings, config)
    edits = _tab1(config).get(PAIR_MARKERS) or {}
    return {
        row["sample"]: {
            "codex": str(codex_dir / row["codex"]),
            "he": str(he_dir / row["he"]),
            "extra": [str(Path(folder) / item["file"]) for folder, item in zip(found["extra_folders"], row["extra"]) if item["file"]],
            "markers": edits.get(row["codex"]) or {},
        }
        for row in found["rows"]
        if row["he"]
    }


def _shown(root, folder):
    """A folder as the project records it: relative to the project when inside it."""
    try:
        return str(Path(folder).resolve().relative_to(Path(root).resolve())) or "."
    except ValueError:
        return str(folder)


def image_folders(root):
    """The project's folders holding images (a few levels deep), relative to the project: the choices for a folder."""
    from spacemap.background_subtraction_utils import is_supported_image_file

    root = Path(root).resolve()
    found = []
    for current, names, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        names[:] = sorted(name for name in names if not name.startswith(".") and name != INTERMEDIATE)
        if depth >= FOLDER_DEPTH:
            names[:] = []
        if any(is_supported_image_file(name) for name in files):
            found.append(str(Path(current).relative_to(root)))
    return sorted(found)


def save_pairing(root, changes, expected):
    """Save Tab 1's naming rule, per-image choices and folders in the project, as classic Tab 1 would.

    *changes*: any of the rule's keys; ``choices`` {CODEX file: "<Auto>", "<Skip>" or an H&E file};
    ``folders`` {"codex"/"he"/"registered": a folder's path, absolute or inside the project; CODEX and
    H&E folders must exist, the registered folder may be new}. *expected* is the revision the window last read.
    """
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if not isinstance(changes, dict) or set(changes) - set(RULE) - {"choices", "folders", "extra_folders", "extra_choices"}:
        raise ValueError("Only the naming rule, the pair choices and the folders can be changed here.")
    for key in set(changes) & set(RULE):
        value = changes[key]
        if not isinstance(value, str) or len(value) > 64 or any(mark in value for mark in "/\\"):
            raise ValueError("A prefix or suffix is short text, without / or \\.")
    folders_changed = changes.get("folders") or {}
    if not isinstance(folders_changed, dict) or set(folders_changed) - set(FOLDER_KEYS):
        raise ValueError("Choose the CODEX, H&E or registered folder.")
    base = Path(root).resolve()
    resolved = {}
    for kind, value in folders_changed.items():
        if not isinstance(value, str) or not value.strip() or len(value) > 1024:
            raise ValueError("Type the folder's path.")
        target = Path(value.strip()).expanduser()
        target = (target if target.is_absolute() else base / target).resolve()
        if kind != "registered" and not target.is_dir():
            raise ValueError(f"{target} is not a folder.")
        if kind == "registered" and not target.parent.is_dir():
            raise ValueError(f"{target.parent} is not a folder.")
        resolved[kind] = target
    extra = changes.get("extra_folders")
    if extra is not None:
        if not isinstance(extra, list) or len(extra) > 8 or not all(isinstance(v, str) and v.strip() for v in extra):
            raise ValueError("Type each extra CODEX folder's path.")
        extra = [(Path(v.strip()).expanduser() if Path(v.strip()).expanduser().is_absolute() else base / v.strip()).resolve() for v in extra]
        missing = [str(folder) for folder in extra if not folder.is_dir()]
        if missing:
            raise ValueError(f"{missing[0]} is not a folder.")
    with project_lock(root):
        config = read_config(root)
        mappings = config.get("directory_mappings") or {}
        current = plan(root, mappings, config)
        if expected != pairing_revision(current):
            raise ValueError("The pairs changed in another window. Reload them before saving.")
        choices = changes.get("choices") or {}
        if not isinstance(choices, dict):
            raise ValueError("Choose Auto, Skip or an H&E image for each CODEX image.")
        codex_files = {row["codex"] for row in current["rows"]}
        for codex_file, choice in choices.items():
            if codex_file not in codex_files:
                raise ValueError(f"{codex_file} is not one of the CODEX images.")
            if choice not in (AUTO, SKIP) and choice not in current["he_files"]:
                raise ValueError(f"{choice} is not one of the H&E images.")
        for key in set(changes) & set(RULE):
            persistence.update_stored_config_key(config, key, changes[key].strip())
        for codex_file, choice in choices.items():
            persistence.update_stored_config_key(config, OVERRIDE + codex_file, choice)
        if extra is not None:
            persistence.update_stored_config_key(config, EXTRA_DIRS, [
                "${PROJECT_ROOT}/" + folder.relative_to(base).as_posix() if folder == base or base in folder.parents else str(folder)
                for folder in extra])
        extra_choices = changes.get("extra_choices") or {}
        if extra_choices:
            if not isinstance(extra_choices, dict) or not all(
                    codex in codex_files and isinstance(values, list) and all(isinstance(v, str) for v in values)
                    for codex, values in extra_choices.items()):
                raise ValueError("Choose Auto, Skip or an image for each extra CODEX folder.")
            stored = dict(_tab1(config).get(EXTRA_CHOICES) or {})
            stored.update(extra_choices)
            persistence.update_stored_config_key(config, EXTRA_CHOICES, stored)
        for kind, target in resolved.items():
            inside = target == base or base in target.parents
            value = "${PROJECT_ROOT}/" + target.relative_to(base).as_posix() if inside else str(target)
            for key in FOLDER_KEYS[kind]:
                persistence.update_stored_config_key(config, key, value)
        persistence.save_project_config(str(root), config)
        config = read_config(root)
        return plan(root, config.get("directory_mappings") or {}, config)


def feature_matches(root, mappings, config, sample):
    """The newest matched-points figure of *sample*'s registration (palom's coarse alignment), or None.

    Tab 1 names its intermediate files after the H&E image (``<H&E name>_he_resized[_rotN]``), so the
    figure is found by that name in the registered folder's intermediate images.
    """
    from spacemap.registration import _image_base_name

    row = next((row for row in plan(root, mappings, config)["rows"] if row["sample"] == sample), None)
    if row is None or not row["he"]:
        raise ValueError(f"No pair {sample}.")
    _, _, registered = folders(root, mappings)
    base = _image_base_name(row["he"], suffix_to_strip="_he")
    folder = registered / INTERMEDIATE
    found = sorted(folder.glob(f"{base}_he*_feature_matches.png"), key=lambda path: path.stat().st_mtime) if folder.is_dir() else []
    return found[-1] if found else None


def output_for(registered_dir, sample):
    return Path(registered_dir) / f"{sample}_merged.ome.tiff"


def set_aside(output):
    """Move an earlier result into ``previous/`` beside it (kept, never deleted) before registering again."""
    output = Path(output)
    if not output.exists():
        return None
    folder = output.parent / PREVIOUS
    folder.mkdir(exist_ok=True)
    target, number = folder / output.name, 1
    while target.exists():
        number += 1
        target = folder / output.name.replace("_merged", f"_merged_{number}", 1)
    output.rename(target)
    return target


def detected_pixel_size(path):
    """The image's pixel size in µm as its file records it, or None."""
    from spacemap.background_subtraction_utils import get_image_mpp

    try:
        value = get_image_mpp(str(path)).get("codex")
        return round(float(value), 6) if value else None
    except Exception:
        return None


def pixel_size(path, fallback):
    from spacemap.background_subtraction_utils import get_image_mpp

    try:
        value = get_image_mpp(str(path)).get("codex")
        return float(value) if value else fallback
    except Exception:  # unreadable metadata: Tab 1 also falls back
        return fallback


def check(options):
    anchor = options.get("anchor") or None  # empty means Auto (DAPI), as in Tab 1
    rotation = options.get("rotation", 0)
    keypoints = options.get("keypoints", KEYPOINTS)
    if anchor is not None and (not isinstance(anchor, str) or not 0 < len(anchor) <= 128):
        raise ValueError("Choose the anchor marker by name.")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Rotate the H&E by 0, 90, 180 or 270 degrees.")
    if isinstance(keypoints, bool) or not isinstance(keypoints, int) or not 500 <= keypoints <= 100000:
        raise ValueError("Keypoints must be between 500 and 100,000.")
    return {"anchor": anchor or None, "rotation": rotation, "keypoints": keypoints}


def register(items, options, report):
    """Register each pair with Tab 1's functions; a pair already merged is kept, unless it is to be registered again."""
    written = []
    for index, item in enumerate(items):
        output = Path(item["output"])
        if output.exists() and not item.get("again"):
            report(f"{item['sample']}: already registered, kept", (index + 1) / len(items))
            continue
        earlier = set_aside(output) if item.get("again") else None
        try:
            written.append(_register_one(item, options, index, len(items), report))
        except BaseException:
            if earlier:  # registering again failed: the earlier result goes back in place of any partial one
                output.unlink(missing_ok=True)
                earlier.rename(output)
            raise
    return written


def _layer_config(edits):
    """A merge channel configuration from a pair's saved layer choices (kept layers, new names and colours)."""
    if not edits:
        return None
    return {
        "keep_indices": [int(i) for i in edits.get("keep", [])] or None,
        "rename_by_index": {int(i): name for i, name in (edits.get("rename") or {}).items()},
        "colors_by_index": {int(i): colour for i, colour in (edits.get("colors") or {}).items()},
    }


def _register_one(item, options, index, count, report):
    """One sample: the H&E (and any extra CODEX images) aligned to CODEX (1), then merged, as Tab 1 does."""
    from spacemap.registration import (
        extract_dapi_channel,
        merge_multi_round_ome_tiff,
        preprocess_he_image,
        register_codex_to_codex,
        register_images,
    )

    output = Path(item["output"])
    step = lambda text, fraction, index=index: report(f"{item['sample']}: {text}", (index + fraction) / count)  # noqa: E731
    intermediate = output.parent / INTERMEDIATE
    intermediate.mkdir(parents=True, exist_ok=True)
    markers = item.get("markers") or {}
    images = markers.get("images") or []
    anchors = [(images[i].get("anchor") if i < len(images) else None) or options.get("anchor") for i in range(1 + len(item.get("extra") or []))]
    chosen_mpp = markers.get("mpp") or {}
    codex_mpp = chosen_mpp.get("codex") or pixel_size(item["codex"], CODEX_MPP)
    he_mpp = chosen_mpp.get("he") or pixel_size(item["he"], HE_MPP)
    step(f"preparing the H&E ({he_mpp:g} µm/px to the CODEX's {codex_mpp:g})", 0.05)
    resized = preprocess_he_image(
        item["he"],
        str(intermediate),
        {"hne": he_mpp, "codex": codex_mpp},
        pre_rotation_deg=float(markers.get("he_rotation", options["rotation"]) or 0),
    )
    step(f"extracting the anchor channel ({anchors[0] or 'Auto'})", 0.15)
    anchor = extract_dapi_channel(item["codex"], str(intermediate), reference_channel=anchors[0])
    step("aligning the H&E with palom", 0.25)
    registered = register_images(
        resized,
        anchor,
        str(intermediate),
        {"hne": codex_mpp, "codex": codex_mpp},
        reference_channel=anchors[0],
        n_keypoints=int(options["keypoints"]),
    )
    aligned = []
    for number, extra in enumerate(item.get("extra") or [], start=2):
        step(f"aligning CODEX ({number}) {Path(extra).name}", 0.4 + 0.3 * (number - 2) / max(len(item["extra"]), 1))
        moving_anchor = extract_dapi_channel(extra, str(intermediate), reference_channel=anchors[number - 1])
        target = intermediate / f"{item['sample']}_codex{number}_registered_to_codex1.ome.tiff"
        register_codex_to_codex(
            anchor, moving_anchor, extra, str(target), {"codex_ref": codex_mpp, "codex_mov": pixel_size(extra, CODEX_MPP)},
            overwrite=True, fixed_reference_channel=anchors[0], moving_reference_channel=anchors[number - 1],
            n_keypoints=int(options["keypoints"]),
        )
        if not target.exists():
            raise RuntimeError(f"{item['sample']}: CODEX ({number}) {Path(extra).name} could not be aligned to CODEX (1).")
        aligned.append(str(target))
    step("merging CODEX and H&E", 0.8)
    provenance = {
        "CODEX": Path(item["codex"]).name,
        "H&E": Path(item["he"]).name,
        "Anchor": anchors[0] or "Auto (DAPI)",
        "H&E rotation": f"{markers.get('he_rotation', options['rotation'])} degrees",
        "Keypoints": str(options["keypoints"]),
        "CODEX pixel size (µm)": f"{codex_mpp:g}",
        "H&E pixel size (µm)": f"{he_mpp:g}",
    }
    for number, extra in enumerate(item.get("extra") or [], start=2):
        provenance[f"CODEX ({number})"] = f"{Path(extra).name} (anchor {anchors[number - 1] or 'Auto'})"
    configs = [_layer_config(images[i] if i < len(images) else None) for i in range(1 + len(aligned))]
    configs.append(_layer_config(markers.get("he")))
    merge_multi_round_ome_tiff([item["codex"], *aligned, registered], str(output), codex_mpp,
                               configs if any(configs) else None, provenance=provenance)
    if not output.exists():
        raise RuntimeError(f"{item['sample']}: registration finished without writing {output.name}.")
    return str(output)


def quality(path, side=2048):
    """How well a registered image's H&E sits on its CODEX: DAPI against the H&E's nuclear stain (haematoxylin).

    Read at a small size (the longest side about *side* pixels), inside the tissue: the correlation of the two
    (1 is a perfect match) and the shift that would match them best (near 0 when the registration is right).
    """
    from skimage.color import rgb2hed
    from skimage.registration import phase_cross_correlation

    from spacemap.viewer import render
    from spacemap.viewer.sources import image_info

    info = image_info(str(path))
    names = list(info.channel_names)
    wanted = ["DAPI", "HE_Red", "HE_Green", "HE_Blue"]
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError("This image has no " + ", ".join(missing) + ": open a registered image (sample_merged).")
    z = max(0, render.max_zoom(info) - max(0, int(np.log2(side / render.TILE))))
    scale = 2**z
    columns = int(np.ceil(info.width / (render.TILE * scale)))
    rows = int(np.ceil(info.height / (render.TILE * scale)))

    def whole(channel):
        return np.vstack(
            [
                np.hstack([render.raw_tile(str(path), info, z, x, y, channel) for x in range(columns)])
                for y in range(rows)
            ]
        ).astype(np.float64)

    dapi = whole(names.index("DAPI"))
    rgb = np.dstack([whole(names.index(name)) for name in wanted[1:]])
    rgb = rgb / (255.0 if rgb.max() <= 255 else rgb.max())  # the H&E as 8-bit colour, as it is stored
    nuclei = rgb2hed(np.clip(rgb, 0, 1))[..., 0]
    tissue = rgb.mean(axis=2) < 0.9  # not the near-white glass
    if tissue.sum() < 100:
        raise ValueError("Too little tissue at this size to judge the registration.")
    r = float(np.corrcoef(dapi[tissue], nuclei[tissue])[0, 1])
    shift, _, _ = phase_cross_correlation(
        np.where(tissue, nuclei, 0), np.where(tissue, dapi, 0), upsample_factor=4, normalization=None
    )
    mpp = pixel_size(path, CODEX_MPP)
    shift_um = float(np.hypot(*shift) * scale * mpp)
    return {
        "correlation": round(r, 3),
        "shift_um": round(shift_um, 1),
        "verdict": "good" if r >= 0.3 and shift_um <= 10 else "check",
        "scale": scale,
    }



def _hex(color):
    if isinstance(color, str):
        return ("#" + color.lstrip("#"))[:7].lower()
    return "#%02x%02x%02x" % tuple(int(v) for v in color[:3])


def pair_markers(root, mappings, config, sample):
    """One pair's layers for its registered image: each CODEX image's markers (kept, new name, colour, anchor)
    and the H&E's three layers, with the pair's saved choices (all kept, names and colours as read, anchor DAPI)."""
    from spacemap.background_subtraction_utils import get_channel_info

    found = plan(root, mappings, config)
    row = next((item for item in found["rows"] if item["sample"] == sample), None)
    if row is None:
        raise ValueError(f"No pair {sample}.")
    codex_dir, _, _ = folders(root, mappings)
    saved = (_tab1(config).get(PAIR_MARKERS) or {}).get(row["codex"]) or {}
    paths = [codex_dir / row["codex"]] + [Path(folder) / item["file"] for folder, item in zip(found["extra_folders"], row["extra"]) if item["file"]]
    images = []
    for number, path in enumerate(paths):
        names, colors, meta, _ = get_channel_info(str(path))
        meta = list(meta) + [{}] * (len(names) - len(meta))
        edits = (saved.get("images") or [{}] * len(paths))[number] if number < len(saved.get("images") or []) else {}
        keep = set(edits.get("keep", range(len(names))))
        anchor = edits.get("anchor") or next((name for name in names if "dapi" in name.lower()), names[0] if names else None)
        images.append({
            "label": f"CODEX ({number + 1})" if len(paths) > 1 else "CODEX", "file": path.name, "anchor": anchor,
            "channels": [{"index": i, "name": name, "color": _hex(color), "keep": i in keep,
                          "rename": (edits.get("rename") or {}).get(str(i), ""), "new_color": (edits.get("colors") or {}).get(str(i), ""),
                          "dye": str((info or {}).get("dye") or "-"), "exposure": str((info or {}).get("exposure") or "-")}
                         for i, (name, color, info) in enumerate(zip(names, colors, meta))],
        })
    he = saved.get("he") or {}
    he_keep = set(he.get("keep", range(3)))
    images.append({"label": "H&E", "file": row["he"], "anchor": None, "channels": [
        {"index": i, "name": name, "color": color, "keep": i in he_keep, "rename": (he.get("rename") or {}).get(str(i), ""),
         "new_color": (he.get("colors") or {}).get(str(i), ""), "dye": "-", "exposure": "-"} for i, (name, color) in enumerate(HE_CHANNELS)]})
    _, he_dir, _ = folders(root, mappings)
    saved_mpp = saved.get("mpp") or {}
    mpp = {}
    for kind, path, fallback in (("codex", paths[0], CODEX_MPP), ("he", he_dir / row["he"] if row["he"] else None, HE_MPP)):
        found_mpp = detected_pixel_size(path) if path else None
        mpp[kind] = {"detected": found_mpp, "fallback": fallback, "saved": saved_mpp.get(kind), "used": saved_mpp.get(kind) or found_mpp or fallback}
    return {"sample": sample, "codex": row["codex"], "edited": bool(saved), "images": images, "mpp": mpp}


def save_pair_rotation(root, sample, degrees):
    """The H&E's turn for this pair (set on the image in Compare), used as its pre-rotation when registering."""
    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    if isinstance(degrees, bool) or not isinstance(degrees, (int, float)) or not -360 <= degrees <= 360:
        raise ValueError("Turn the H&E by -360 to 360 degrees.")
    with project_lock(root):
        config = read_config(root)
        row = next((item for item in plan(root, config.get("directory_mappings") or {}, config)["rows"] if item["sample"] == sample), None)
        if row is None:
            raise ValueError(f"No pair {sample}.")
        stored = dict(_tab1(config).get(PAIR_MARKERS) or {})
        stored[row["codex"]] = {**(stored.get(row["codex"]) or {}), "he_rotation": round(float(degrees), 2)}
        persistence.update_stored_config_key(config, PAIR_MARKERS, stored)
        persistence.save_project_config(str(root), config)
    return {"sample": sample, "rotation": round(float(degrees), 2)}


def save_pair_markers(root, sample, edits):
    """Save one pair's layer choices: {"images": [{"keep", "rename", "colors", "anchor"} per CODEX image], "he": {...}}."""
    import re

    from spacemap_gui import persistence

    from .settings import project_lock, read_config

    def clean(item, anchor):
        if not isinstance(item, dict) or set(item) - {"keep", "rename", "colors", "anchor"}:
            raise ValueError("Send each image's kept layers, names and colours.")
        keep = item.get("keep", [])
        if not isinstance(keep, list) or not keep or not all(isinstance(i, int) and 0 <= i < 1000 for i in keep):
            raise ValueError("Keep at least one layer of each image.")
        rename = {str(k): str(v).strip() for k, v in (item.get("rename") or {}).items() if str(v).strip()}
        colors = {str(k): str(v).lower() for k, v in (item.get("colors") or {}).items() if v}
        if any(len(v) > 64 for v in rename.values()) or any(not re.fullmatch(r"#[0-9a-f]{6}", v) for v in colors.values()):
            raise ValueError("Names are short; colours are #rrggbb.")
        cleaned = {"keep": sorted(set(keep)), "rename": rename, "colors": colors}
        if anchor:
            if not isinstance(item.get("anchor"), str) or not item["anchor"]:
                raise ValueError("Choose each CODEX image's anchor.")
            cleaned["anchor"] = item["anchor"]
        return cleaned

    if not isinstance(edits, dict) or set(edits) - {"images", "he", "mpp"} or not isinstance(edits.get("images"), list):
        raise ValueError("Send the pair's layers.")
    cleaned = {"images": [clean(item, True) for item in edits["images"]], "he": clean(edits.get("he") or {"keep": [0, 1, 2]}, False)}
    mpp = edits.get("mpp") or {}
    if not isinstance(mpp, dict) or set(mpp) - {"codex", "he"} or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.01 <= v <= 50 for v in mpp.values() if v is not None):
        raise ValueError("A pixel size is 0.01 to 50 µm per pixel.")
    mpp = {k: float(v) for k, v in mpp.items() if v is not None}
    if mpp:
        cleaned["mpp"] = mpp
    with project_lock(root):
        config = read_config(root)
        mappings = config.get("directory_mappings") or {}
        row = next((item for item in plan(root, mappings, config)["rows"] if item["sample"] == sample), None)
        if row is None:
            raise ValueError(f"No pair {sample}.")
        stored = dict(_tab1(config).get(PAIR_MARKERS) or {})
        if "he_rotation" in (stored.get(row["codex"]) or {}):
            cleaned["he_rotation"] = stored[row["codex"]]["he_rotation"]
        stored[row["codex"]] = cleaned
        persistence.update_stored_config_key(config, PAIR_MARKERS, stored)
        persistence.save_project_config(str(root), config)
        config = read_config(root)
        return pair_markers(root, config.get("directory_mappings") or {}, config, sample)


QUICK_WIDTH = 1500  # the CODEX's width in the small copies Quick align matches


def _small_copy(path, mpp, target_mpp, channel=None):
    """One channel (or an RGB image as grey) of *path* at *target_mpp* µm/px, from its nearest smaller level."""
    import cv2
    import tifffile
    import zarr

    with tifffile.TiffFile(str(path)) as tif:
        series = max(tif.series, key=lambda s: int(np.prod(s.shape)))
        axes, width = series.axes, series.shape[series.axes.index("X")]
        levels = list(series.levels) or [series]
        chosen = 0
        for index, level in enumerate(levels):
            if mpp * width / level.shape[axes.index("X")] <= target_mpp:
                chosen = index
        level = levels[chosen]
        data = np.asarray(zarr.open(series.aszarr(level=chosen), mode="r")[...])
    rgb = "S" in axes and data.shape[axes.index("S")] in (3, 4)
    picked = []
    for axis, name in enumerate(axes):
        if name in "YX" or (rgb and name == "S"):
            picked.append(slice(None))
        elif name in "CI" and not rgb:
            picked.append(channel or 0)
        else:
            picked.append(0)
    plane = data[tuple(picked)].astype(np.float32)
    if rgb:
        plane = plane.mean(axis=-1 if axes.index("S") > axes.index("X") else 0)
    level_mpp = mpp * width / level.shape[axes.index("X")]
    factor = level_mpp / target_mpp
    size = (max(1, int(round(plane.shape[1] * factor))), max(1, int(round(plane.shape[0] * factor))))
    return cv2.resize(plane, size, interpolation=cv2.INTER_AREA), width, int(series.shape[axes.index("Y")])


def quick_align(root, mappings, config, sample):
    """Where the H&E sits on the CODEX, found in seconds from small copies (palom's coarse matching, as registering starts).

    Returns the H&E's turn (degrees), its size against the pixel sizes (zoom, 1 when they are right), and the
    top-left of its box in CODEX pixels, as the viewer places the H&E over the CODEX (turned about its centre).
    """
    from palom.register import feature_based_registration

    from spacemap.background_subtraction_utils import get_channel_info

    details = pair_markers(root, mappings, config, sample)
    codex_dir, he_dir, _ = folders(root, mappings)
    row = next(row for row in plan(root, mappings, config)["rows"] if row["sample"] == sample)
    if not row["he"]:
        raise ValueError(f"{sample} has no H&E image.")
    codex_mpp, he_mpp = details["mpp"]["codex"]["used"], details["mpp"]["he"]["used"]
    names = get_channel_info(str(codex_dir / row["codex"]))[0]
    anchor = details["images"][0]["anchor"]
    import tifffile

    with tifffile.TiffFile(str(codex_dir / row["codex"])) as tif:
        series = max(tif.series, key=lambda s: int(np.prod(s.shape)))
        codex_width = series.shape[series.axes.index("X")]
    target = codex_mpp * codex_width / QUICK_WIDTH
    reference, _, _ = _small_copy(codex_dir / row["codex"], codex_mpp, target, names.index(anchor) if anchor in names else 0)
    moving, he_width, he_height = _small_copy(he_dir / row["he"], he_mpp, target)
    matrix = np.asarray(feature_based_registration(reference, moving, n_keypoints=4000, plot_match_result=False, auto_mask=True))
    if not np.isfinite(matrix).all() or np.allclose(matrix, np.eye(3)[:2]):
        raise ValueError("Quick align found too few matching points: turn the H&E by eye first, then try again.")
    linear = matrix[:, :2]
    if np.linalg.det(linear) < 0:
        raise ValueError("The H&E looks mirrored against the CODEX: Quick align cannot place it.")
    zoom = float(np.sqrt(abs(np.linalg.det(linear))))
    angle = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    factor = target / codex_mpp  # CODEX pixels per small-copy pixel
    shift = matrix[:, 2] * factor
    half = np.array([he_width, he_height]) * (he_mpp / codex_mpp) * zoom / 2  # half the H&E's box, in CODEX pixels
    turn = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))], [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
    corner = shift + turn @ half - half  # the box's top-left when it is turned about its centre
    return {"sample": sample, "rotation": round(angle, 2), "zoom": round(zoom, 4), "x": round(float(corner[0]), 1),
            "y": round(float(corner[1]), 1), "pixel_sizes_ok": bool(0.8 <= zoom <= 1.25)}
