"""Studio's Cellpose training: tiles corrected on the image make a model better, round by round.

The user's workflow (as Cellpose's own trainer, with validation):

* A training session names the model and its base (built-in or a saved model);
  the session's current model starts as the base.
* Tiles of a chosen size are placed on the tissue, each for training or for
  validation. The current model predicts a tile's cells, the user corrects them
  (delete wrong ones, draw missing ones), and the tile is kept: its picture
  (``_img.tif``: cell-border markers combined, empty, nuclear, as Studio gives
  Cellpose), the corrected cells (``_masks.tif``, the ground truth) and the
  model's prediction before the corrections (``_pred.tif``).
* Train continues from the current model on every training tile (the Cellpose
  GUI's settings: SGD, learning rate 0.1, weight decay 0.0001, 100 epochs) and
  saves the next version, ``<name>_v<n>``, which becomes the current model.
* Validate compares each validation tile's prediction with its corrections:
  average precision at IoU 0.5 (Cellpose's metric), precision, recall, F1 and
  the cell counts. Validation tiles can then join training.
* Score all versions runs the base model and every version on the same
  validation tiles, so the rounds can be compared: is training still helping?

Everything is in the project: ``outputs/segmentation/cellpose_training/<set>/``
(tiles, ``examples.json``, ``session.json``) and ``outputs/segmentation/
cellpose_models/models/`` (Cellpose's own place for trained models).
"""

from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,50}")
ROLES = ("train", "validate")
TILE_SIZES = (256, 512, 768, 1024)
REGION_LIMIT = 1024
# The Cellpose GUI's training settings (cellpose/gui/gui.py).
TRAINING = {"learning_rate": 0.1, "weight_decay": 0.0001, "n_epochs": 100, "SGD": True}


def folders(root, training_set="studio"):
    if not NAME.fullmatch(training_set):
        raise ValueError("Name the training set with letters, numbers, _ . or -.")
    base = Path(root) / "outputs" / "segmentation"
    return base / "cellpose_training" / training_set, base / "cellpose_models"


def _read(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    temporary.replace(path)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def session(root, training_set="studio"):
    folder, _ = folders(root, training_set)
    return _read(folder / "session.json", None)


def start(root, name, base, training_set="studio"):
    """A new session: *name* for its versions, trained from *base* (built-in model or a model file)."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError("Name the model with letters, numbers, _ . or - (up to 50).")
    if not isinstance(base, str) or not base or len(base) > 1024:
        raise ValueError("Choose the model to start from.")
    base = checked_base(base)
    folder, models = folders(root, training_set)
    if (models / "models" / f"{name}_v1").exists():
        raise ValueError("Models with this name exist. Choose a new name so they are kept.")
    value = {"name": name, "base": base, "current": base, "versions": [], "started": _now()}
    _write(folder / "session.json", value)
    return value


def checked_base(base):
    """Resolve saved weights explicitly; Cellpose otherwise falls back for unknown names."""
    from .segmentation import BUILTIN

    if base in BUILTIN:
        return base
    path = Path(base).expanduser()
    if not path.is_file():
        raise ValueError("The selected model file is missing. Choose an available model in Start from.")
    return str(path.resolve())


def starting_models(root, config, mappings):
    """Use Segmentation's catalogue, including the configured model folder and current custom path."""
    from . import segmentation

    places = segmentation.step_folders(root, config, mappings)
    choices = segmentation.model_choices(root, places['models'])
    current, extra = segmentation.current_model(segmentation.values(root, config), choices)
    if extra:
        choices.append(extra)
    found = []
    seen = set()
    selected = ''
    for item in choices:
        base = str(Path(item['path']).expanduser().resolve()) if item['path'] else item['id']
        if item['id'] == current:
            selected = base
        if base in seen:
            continue
        seen.add(base)
        found.append({**item, 'base': base, 'available': not item['path'] or Path(base).is_file(),
                      'modified': Path(base).stat().st_mtime_ns if item['path'] and Path(base).is_file() else 0})
    # Actual file times also order imported models and v9/v10 correctly without metadata.
    groups = {'Trained in this project': 0, 'Models folder': 1, 'Other': 2, 'Cellpose': 3}
    found.sort(key=lambda item: (groups.get(item['group'], 2), -item['modified'], item['name']))
    # Keep a missing selected path visible so Start reports it, rather than
    # silently changing the user's intended starting weights.
    if not selected:
        selected = next((item['base'] for item in found if item['available']), 'cyto')
    return {'starting_models': found, 'selected_base': selected}


def examples(root, training_set="studio"):
    """The tiles (newest first), the session and the models trained here."""
    folder, models = folders(root, training_set)
    entries = sorted(_read(folder / "examples.json", []), key=lambda item: item["created"], reverse=True)
    trained = []
    saved = models / "models"  # where Cellpose's train_seg puts them
    if saved.is_dir():
        for path in sorted(saved.iterdir()):
            if path.is_file() and not path.name.endswith((".json", ".npy", ".tmp")):
                trained.append({"name": path.name, "path": str(path), **_read(path.with_name(path.name + ".json"), {})})
    counts = {role: sum(item.get("role", "train") == role for item in entries) for role in ROLES}
    return {
        "scores": _scores(folder, entries),
        "examples": entries,
        "cells": sum(item["cells"] for item in entries),
        "counts": counts,
        "models": trained,
        "session": session(root, training_set),
        "set": training_set,
        "tile_sizes": list(TILE_SIZES),
    }


def masks_from(polygons, bounds):
    """Label image of *polygons* (image pixels) inside *bounds* (x, y, w, h); later cells win overlaps."""
    from skimage.draw import polygon as fill

    x, y, width, height = (int(round(v)) for v in bounds)
    masks = np.zeros((height, width), np.uint16)
    label = 0
    for ring in polygons or []:
        points = np.asarray(ring, dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            continue
        rows, cols = fill(points[:, 1] - y, points[:, 0] - x, shape=masks.shape)
        if len(rows) < 4:
            continue
        label += 1
        masks[rows, cols] = label
    # A cell hidden entirely by later ones is dropped, and labels are made consecutive.
    present = np.unique(masks[masks > 0])
    remap = np.zeros(label + 1, np.uint16)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.uint16)
    return remap[masks]


def check_region(bounds, width, height):
    if not (
        isinstance(bounds, list)
        and len(bounds) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bounds)
    ):
        raise ValueError("Place the tile first.")
    x, y, w, h = bounds
    if w < 32 or h < 32:
        raise ValueError("Use a larger tile (at least 32 pixels each way).")
    if w > REGION_LIMIT or h > REGION_LIMIT:
        raise ValueError(f"Use a smaller tile (at most {REGION_LIMIT} pixels each way).")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("The tile must lie inside the image.")
    return [int(round(v)) for v in bounds]


def save_example(
    root, image, names, bounds, polygons, settings, training_set="studio", role="train", predicted=None, model=""
):
    """Keep a corrected tile: its picture, the corrected cells, and what the model predicted before."""
    import tifffile

    from . import segmentation

    if role not in ROLES:
        raise ValueError("A tile is for training or for validation.")
    if not isinstance(polygons, list) or not polygons or len(polygons) > 20000:
        raise ValueError("Keep at least one cell in the tile.")
    if predicted is not None and (not isinstance(predicted, list) or len(predicted) > 20000):
        raise ValueError("The prediction could not be read.")
    folder, _ = folders(root, training_set)
    nuclear, membranes = segmentation.channels(names, settings)
    planes = segmentation.read_crop(image, names, [nuclear, *membranes], bounds)
    nucleus = planes[0].astype(np.float32)
    borders = np.max(np.stack(planes[1:]), axis=0).astype(np.float32)
    picture = np.stack((borders, np.zeros_like(borders), nucleus))  # (3, h, w), as Cellpose is given it
    masks = masks_from(polygons, bounds)
    cells = int(masks.max())
    if not cells:
        raise ValueError("None of the cells is inside the tile.")
    x, y, w, h = bounds
    name = f"{Path(image).name.split('.')[0]}_x{x}_y{y}_w{w}_h{h}"
    folder.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(folder / f"{name}_img.tif", picture)
    tifffile.imwrite(folder / f"{name}_masks.tif", masks)
    if predicted is not None:
        tifffile.imwrite(folder / f"{name}_pred.tif", masks_from(predicted, bounds))
    entries = [item for item in _read(folder / "examples.json", []) if item["name"] != name]
    entries.append(
        {
            "name": name,
            "image": Path(image).name,
            "bounds": list(bounds),
            "role": role,
            "model": str(model or ""),
            "predicted": None if predicted is None else len(predicted),
            "nuclear": nuclear,
            "membranes": membranes,
            "cells": cells,
            "created": _now(),
        }
    )
    _write(folder / "examples.json", entries)
    return {**examples(root, training_set), "saved": name, "saved_cells": cells}


def delete_example(root, name, training_set="studio"):
    folder, _ = folders(root, training_set)
    entries = _read(folder / "examples.json", [])
    if name not in {item["name"] for item in entries}:
        raise ValueError("There is no such tile.")
    for suffix in ("_img.tif", "_masks.tif", "_pred.tif"):
        (folder / f"{name}{suffix}").unlink(missing_ok=True)
    _write(folder / "examples.json", [item for item in entries if item["name"] != name])
    return examples(root, training_set)


def set_role(root, names, role, training_set="studio"):
    """Move tiles to training or validation (validated tiles can join training)."""
    if role not in ROLES:
        raise ValueError("A tile is for training or for validation.")
    folder, _ = folders(root, training_set)
    entries = _read(folder / "examples.json", [])
    known = {item["name"] for item in entries}
    if not isinstance(names, list) or not names or not set(names) <= known:
        raise ValueError("Choose tiles of this training set.")
    for item in entries:
        if item["name"] in names:
            item["role"] = role
    _write(folder / "examples.json", entries)
    return examples(root, training_set)


def compare(truth, predicted):
    """How a prediction matches the corrected cells at IoU 0.5 (Cellpose's matching)."""
    from cellpose import metrics

    precision, tp, fp, fn = (value[0][0] for value in metrics.average_precision([truth], [predicted], threshold=[0.5]))
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "ap": round(float(precision), 4)}


def summary(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "ap": round(tp / max(tp + fp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
    }


def validate(root, training_set="studio"):
    """Each validation tile's prediction against its corrections, and all of them together."""
    import tifffile

    folder, _ = folders(root, training_set)
    tiles = [item for item in _read(folder / "examples.json", []) if item.get("role") == "validate"]
    rows, total = [], {"tp": 0, "fp": 0, "fn": 0}
    for item in tiles:
        prediction = folder / f"{item['name']}_pred.tif"
        if not prediction.is_file():
            continue
        truth, predicted = tifffile.imread(folder / f"{item['name']}_masks.tif"), tifffile.imread(prediction)
        found = compare(truth, predicted)
        for key in total:
            total[key] += found[key]
        rows.append(
            {
                "name": item["name"],
                "image": item["image"],
                "model": item.get("model", ""),
                "corrected": int(truth.max()),
                "predicted": int(predicted.max()),
                **summary({key: found[key] for key in total}),
            }
        )
    if not rows:
        raise ValueError("Keep at least one corrected validation tile first.")
    return {"tiles": rows, "overall": summary(total)}


def _load(folder, entries):
    import tifffile

    pictures = [tifffile.imread(folder / f"{item['name']}_img.tif") for item in entries]
    labels = [tifffile.imread(folder / f"{item['name']}_masks.tif") for item in entries]
    return pictures, labels


def _model(base, gpu):
    """A Cellpose model to train or score; on the CPU without MKLDNN, which cannot train."""
    from cellpose import models

    base = checked_base(base or 'cyto')
    if Path(base).is_file():
        return models.CellposeModel(gpu=gpu, pretrained_model=str(base), mkldnn=False)
    return models.CellposeModel(gpu=gpu, model_type=base or "cyto", mkldnn=False)


def measure(model, pictures, labels):
    """How *model* finds the corrected cells of the tiles, all cells together."""
    total = {"tp": 0, "fp": 0, "fn": 0}
    for picture, truth in zip(pictures, labels):
        found = compare(truth, model.eval(picture, channels=[1, 3], channel_axis=0)[0])
        for key in total:
            total[key] += found[key]
    return summary(total)


def score(model, pictures, labels):
    """Average precision at IoU 0.5 of *model* over the tiles, all cells together."""
    return measure(model, pictures, labels)["ap"]


def _lineup(current):
    """The base model, then every version, as (name, model)."""
    base = current["base"]
    return [(Path(base).name if "/" in base else base, base)] + [
        (item["name"], item["path"]) for item in current["versions"]
    ]


def _scores(folder, entries):
    """The last scores of every version, and whether they still match the tiles and versions."""
    found, current = _read(folder / "scores.json", None), _read(folder / "session.json", None)
    if not found or not current or found.get("session") != current.get("started"):
        return None
    tiles = sorted(item["name"] for item in entries if item.get("role") == "validate")
    found["current"] = found.get("tiles") == tiles and [row["name"] for row in found.get("rows", [])] == [
        name for name, _ in _lineup(current)
    ]
    return found


def score_versions(settings, folder, report):
    """The job: the base model and every version on the same validation tiles, to compare the rounds."""
    import torch

    root, training_set = Path(settings["project_root"]), settings.get("set", "studio")
    source, _ = folders(root, training_set)
    current = session(root, training_set)
    if not current:
        raise ValueError("Start a training session first.")
    check = sorted(
        (item for item in _read(source / "examples.json", []) if item.get("role") == "validate"),
        key=lambda item: item["name"],
    )
    if not check:
        raise ValueError("Keep at least one corrected validation tile first.")
    pictures, truths = _load(source, check)
    gpu = bool(settings.get("gpu", True)) and torch.cuda.is_available()
    lineup, rows = _lineup(current), []
    for index, (name, path) in enumerate(lineup):
        report(
            f"Scoring {name} on {len(check)} validation tiles ({index + 1} of {len(lineup)})…",
            0.05 + 0.9 * index / len(lineup),
        )
        rows.append({"name": name, "model": path, **measure(_model(path, gpu), pictures, truths)})
    result = {
        "session": current["started"],
        "tiles": [item["name"] for item in check],
        "cells": int(sum(int(truth.max()) for truth in truths)),
        "rows": rows,
        "scored": _now(),
    }
    _write(source / "scores.json", result)
    return result


def train(settings, folder, report):
    """The job: the next version of the session's model, from the current one, on every training tile."""
    import torch
    from cellpose import train as cellpose_train

    root, training_set = Path(settings["project_root"]), settings.get("set", "studio")
    source, models_folder = folders(root, training_set)
    current = session(root, training_set)
    if not current:
        raise ValueError("Start a training session first.")
    entries = sorted(_read(source / "examples.json", []), key=lambda item: item["name"])
    fit = [item for item in entries if item.get("role", "train") == "train"]
    check = [item for item in entries if item.get("role") == "validate"]
    if not fit:
        raise ValueError("Keep at least one corrected training tile first.")
    version = len(current["versions"]) + 1
    name = f"{current['name']}_v{version}"
    if (models_folder / "models" / name).exists():
        raise ValueError(f"{name} exists already. Start a session with a new name.")
    report(f"Reading {len(fit)} training and {len(check)} validation tiles…", 0.02)
    pictures, labels = _load(source, fit)
    checks, truths = _load(source, check) if check else ([], [])
    gpu = bool(settings.get("gpu", True)) and torch.cuda.is_available()
    report(f"Loading {Path(current['current']).name} on {'GPU' if gpu else 'CPU'}…", 0.05)
    model = _model(current["current"], gpu)
    before = score(model, checks, truths) if checks else None
    epochs = int(settings.get("n_epochs", TRAINING["n_epochs"]))
    report(f"Training {name} for {epochs} epochs on {len(fit)} tiles…", 0.1)
    models_folder.mkdir(parents=True, exist_ok=True)
    path, losses = cellpose_train.train_seg(
        model.net,
        train_data=pictures,
        train_labels=labels,
        channels=[1, 3],
        channel_axis=0,
        normalize=True,
        min_train_masks=0,
        save_path=str(models_folder),
        nimg_per_epoch=max(8, len(fit)),
        learning_rate=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
        n_epochs=epochs,
        SGD=TRAINING["SGD"],
        model_name=name,
    )[:2]
    trained = _model(str(path), gpu)
    after = None
    if checks:
        report("Scoring it on the validation tiles…", 0.93)
        after = score(trained, checks, truths)
    note = {
        "version": version,
        "from": current["current"],
        "base": current["base"],
        "training_tiles": len(fit),
        "training_cells": sum(item["cells"] for item in fit),
        "validation_tiles": len(check),
        "epochs": epochs,
        "ap_before": before,
        "ap_after": after,
        "diameter": round(float(trained.net.diam_labels.item()), 2),
        "final_loss": round(float(losses[-1]), 4) if len(losses) else None,
        "created": _now(),
        "set": training_set,
    }
    _write(Path(str(path) + ".json"), note)
    latest = session(root, training_set) or current
    latest["current"] = str(path)
    latest["versions"] = [*latest.get("versions", []), {"name": name, "path": str(path), **note}]
    _write(source / "session.json", latest)
    return {"model": str(path), "name": name, **note}


def tile_outlines(root, image_name, training_set="studio", limit=20000):
    """The corrected cells of the kept tiles on one image, as outlines in image pixels (to show them on the image)."""
    import tifffile
    from skimage.measure import find_contours, regionprops

    folder, _ = folders(root, training_set)
    rings = []
    for entry in _read(folder / "examples.json", []):
        if entry.get("image") != image_name:
            continue
        path = folder / f"{entry['name']}_masks.tif"
        if not path.is_file():
            continue
        masks = tifffile.imread(path)
        x0, y0 = entry["bounds"][0], entry["bounds"][1]
        for region in regionprops(masks):
            top, left, bottom, right = region.bbox
            crop = np.pad(region.image, 1).astype(float)
            contours = find_contours(crop, 0.5)
            if not contours:
                continue
            ring = max(contours, key=len)[::2]  # every second point: light enough to draw thousands of cells
            rings.append([[round(x0 + left + col - 1, 1), round(y0 + top + row - 1, 1)] for row, col in ring])
            if len(rings) >= limit:
                return rings
    return rings
