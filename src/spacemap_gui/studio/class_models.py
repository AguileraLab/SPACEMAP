"""Shared classes: one pixel classification for many images, so a class means the same marker profile everywhere.

A class model is fitted once on a fair sample of pixels from each chosen image (each image prepared as Classification
does it: its markers scaled within its own Work within area), then every pixel of every image goes to its nearest
class. The class numbers, names, colours, merges and exclusions belong to the model, so naming a class on one image
names it on all of them. Each image's run is saved like an ordinary classification run (it opens with its classes,
and Save polygons works as before), and records the model it came from.

Checks kept with each image: the share of its pixels far from every class (a tissue the other images do not have),
and, per class, how far the image's mean profile is from the model's (the same name meaning something else here).

Files: ``outputs/studio/class_models/<id>.json`` (settings, names, checks) and ``<id>.npz`` (centres, profiles).
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import uuid

import numpy as np

from . import annotation_edits as edits

PER_IMAGE = 20_000   # pixels sampled from each image for the fit
FAR = 99.0           # a pixel farther from its class than this percentile of the fit's distances does not fit well


def folder(root):
    return Path(root) / "outputs/studio/class_models"


def _identifier(value):
    if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Choose a saved class model.")
    return value


def read(root, model_id):
    """The model's record and its arrays (centres, z profiles)."""
    model_id = _identifier(model_id)
    path = folder(root) / (model_id + ".json")
    if not path.is_file():
        raise ValueError("That class model is no longer in this project.")
    record = json.loads(path.read_text())
    with np.load(folder(root) / (model_id + ".npz")) as data:
        arrays = {name: data[name] for name in data.files}
    return record, arrays


def listing(root):
    """The project's class models, newest first, and the one chosen ("" = each image its own classes)."""
    home = folder(root)
    models = []
    for path in home.glob("*.json") if home.is_dir() else []:
        if path.name == "selected.json":
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        models.append({k: record.get(k) for k in ("id", "name", "created", "n_clusters", "images", "options", "scale_by")})
    models.sort(key=lambda m: m.get("created") or "", reverse=True)
    chosen = ""
    try:
        chosen = json.loads((home / "selected.json").read_text()).get("id", "")
    except (OSError, ValueError):
        pass
    if chosen and not any(m["id"] == chosen for m in models):
        chosen = ""
    return {"models": models, "selected": chosen}


def select(root, model_id):
    """Choose the class model Classify uses ("" for each image its own classes)."""
    if model_id:
        read(root, model_id)
    home = folder(root)
    home.mkdir(parents=True, exist_ok=True)
    edits._atomic(home / "selected.json", json.dumps({"id": model_id or ""}).encode())
    return listing(root)


def save_review(root, model_id, state):
    """The class names, colours, merges and exclusions of the model: every image classified with it shows them."""
    from .settings import project_lock

    model_id = _identifier(model_id)
    path = folder(root) / (model_id + ".json")
    with project_lock(root):
        record = json.loads(path.read_text())
        record["review"] = {k: state.get(k) for k in ("groups", "names", "colors", "excluded", "original_names")}
        edits._atomic(path, json.dumps(record, indent=2).encode())


def review_of(root, model_id):
    record, _ = read(root, model_id)
    return dict(record.get("review") or {})


def _prepare(root, config, image, output, chosen):
    """The image prepared as Classification does it, inside its Work within area."""
    from . import annotation_inputs as inputs, pixelclasses

    scope = inputs.scope_path(root, image, output, "classes")
    if not Path(scope).is_file():
        raise ValueError("no saved Tissue or Work within annotation")
    return scope, pixelclasses.prepare(image, config, scope, options=chosen)


def _columns(have, names):
    """The positions of *names* among *have* (an image missing one of them cannot use the model)."""
    have = list(have)
    missing = [name for name in names if name not in have]
    if missing:
        raise ValueError("it lacks " + ", ".join(missing[:5]) + (" and more" if len(missing) > 5 else ""))
    return [have.index(name) for name in names]


def slide_of(image):
    """An image's slide: its name before "_roi" (P15_P16_roi_2 -> P15_P16), else the image itself."""
    import re

    from . import annotation

    sample = annotation.sample_id(Path(image).name)
    found = re.match(r"^(.*?)_roi", sample)
    return found.group(1) if found else sample


def _slide_scale(root, config, images, output, chosen, report=lambda *a: None):
    """One scaling of each marker for all *images* of a slide: the median arcsinh cofactor, the lowest low, the
    median high of the images' own scalings, then the mean and spread of their tissue pixels in that scaling."""
    from spacemap.tissue_annotation_utils import PyramidImageReader

    from . import pixelclasses

    found, names = [], None
    for image in images:
        _, prepared = _prepare(root, config, image, output, chosen)
        reader = PyramidImageReader(image)
        own = pixelclasses.scaling(reader, prepared["stack"], prepared["values"])
        if names is None:
            names = list(prepared["stack"]["marker_names"])
        order = [list(prepared["stack"]["marker_names"]).index(n) for n in names]
        found.append((image, reader, prepared, {k: [own[k][i] for i in order] for k in own}))
    scale = {"cofactors": np.median([f[3]["cofactors"] for f in found], axis=0).tolist(), "low": np.min([f[3]["low"] for f in found], axis=0).tolist(),
             "high": np.median([f[3]["high"] for f in found], axis=0).tolist(), "mean": [0.0] * len(names), "std": [1.0] * len(names)}
    total, squares, count = np.zeros(len(names)), np.zeros(len(names)), 0
    for image, reader, prepared, _ in found:
        report(f"{Path(image).name}: scaling within its slide", None)
        stack = _in_scale(reader, prepared, scale)
        values = stack["pixels_original_valid"].astype(np.float64)
        total += values.sum(axis=0); squares += (values ** 2).sum(axis=0); count += len(values)
    mean = total / max(count, 1)
    scale.update(mean=mean.tolist(), std=np.sqrt(np.maximum(squares / max(count, 1) - mean ** 2, 1e-12)).tolist(), marker_names=names)
    return scale


def _in_scale(reader, prepared, scale):
    """The image's features in another scaling (marker intensity; Top-K as Classification sets it)."""
    from . import pixelclasses

    stack = prepared["stack"]
    order = [list(scale.get("marker_names") or stack["marker_names"]).index(n) for n in stack["marker_names"]] if scale.get("marker_names") else None
    own = {k: [scale[k][i] for i in order] if order else scale[k] for k in ("cofactors", "low", "high", "mean", "std")}
    return pixelclasses.features_like(reader, stack, own, stack["valid_tissue_mask"], prepared["factor"], prepared["values"])


def _features(stack):
    return list(stack["feature_names"]) if stack.get("feature_names") else list(stack["marker_names"])


def _low_signal(features, values):
    if str(values.get("low_signal_handling")) != "reserve":
        return np.zeros(len(features), bool)
    mean = features.mean(axis=1)
    return mean <= float(np.quantile(mean, float(values["low_signal_quantile"])))


def fit(root, config, images, output, chosen, report=lambda *a: None, name="", scale_by="image"):
    """A class model from a fair sample of each image's pixels. Returns its record (the images are not yet classified).
    *scale_by* "slide": each marker scaled once per slide (all its images together) instead of per image, so a slide
    stained brighter than the others is brought to the same levels."""
    from sklearn.cluster import MiniBatchKMeans

    from spacemap.tissue_annotation_utils import DEFAULT_CLUSTERING_CONFIG, build_cluster_palette

    from . import classification

    chosen = classification.options(chosen)
    if scale_by not in ("image", "slide"):
        raise ValueError("Scale within each image or each slide.")
    if scale_by == "slide" and chosen["features"]:
        raise ValueError("Scaling per slide works with marker intensity only: untick Local texture and Marker edges.")
    slide_scales = {}
    if scale_by == "slide":
        from collections import defaultdict

        groups = defaultdict(list)
        for image in images:
            groups[slide_of(image)].append(image)
        for number, (slide, members) in enumerate(groups.items()):
            report(f"{Path(members[0]).name}: scaling slide {slide} ({number + 1} of {len(groups)})", 0.3 * number / len(groups))
            try:
                slide_scales[slide] = _slide_scale(root, config, members, output, chosen)
            except (ValueError, OSError) as error:
                report(f"{Path(members[0]).name}: slide {slide} not scaled: {error}", None)
    samples, profiles, used, skipped, feature_names, marker_names, values = [], [], [], [], None, None, None
    rng = np.random.default_rng(42)
    for number, image in enumerate(images):
        label = Path(image).name
        report(f"{label}: reading pixels ({number + 1} of {len(images)})", 0.5 * number / len(images))
        try:
            _, prepared = _prepare(root, config, image, output, chosen)
        except (ValueError, OSError) as error:
            skipped.append(f"{label}: {error}")
            continue
        stack = prepared["stack"]
        if scale_by == "slide":
            if slide_of(image) not in slide_scales:
                skipped.append(f"{label}: its slide could not be scaled")
                continue
            from spacemap.tissue_annotation_utils import PyramidImageReader

            stack = {**stack, **_in_scale(PyramidImageReader(image), prepared, slide_scales[slide_of(image)])}
        if feature_names is None:
            feature_names = _features(stack)
            marker_names = list(stack["marker_names"])
            values = {**DEFAULT_CLUSTERING_CONFIG, **prepared["values"]}
        try:
            columns, markers = _columns(_features(stack), feature_names), _columns(stack["marker_names"], marker_names)
        except ValueError as error:
            skipped.append(f"{label}: {error}")
            continue
        features = stack["features"][:, columns]
        keep = np.flatnonzero(~_low_signal(features, values))
        pick = keep if len(keep) <= PER_IMAGE else rng.choice(keep, PER_IMAGE, replace=False)
        samples.append(features[pick].astype(np.float32))
        profiles.append(stack["pixels_z_valid"][:, markers][pick].astype(np.float32))
        used.append(str(image))
    if not used:
        raise ValueError("No ticked image could be read: " + "; ".join(skipped[:4]))
    pooled, pooled_z = np.concatenate(samples), np.concatenate(profiles)
    n = int(chosen["n_clusters"])
    if len(pooled) < n:
        raise ValueError("There are fewer pixels than classes. Use fewer classes or a finer Scale.")
    report(f"Finding {n} shared classes in {len(pooled):,} pixels from {len(used)} images", 0.55)
    kmeans = MiniBatchKMeans(n_clusters=n, batch_size=int(values["kmeans_batch_size"]), random_state=int(values["random_seed"]),
                             n_init=int(values["kmeans_n_init"]), max_iter=int(values["kmeans_max_iter"]))
    with _limited():
        kmeans.fit(pooled)
        centres = kmeans.cluster_centers_.astype(np.float32)
        labels, distances = _assign(pooled, centres)
    counts = np.bincount(labels, minlength=n)
    profile = np.stack([pooled_z[labels == i].mean(axis=0) if counts[i] else np.zeros(pooled_z.shape[1], np.float32) for i in range(n)])
    reserve = str(values.get("low_signal_handling")) == "reserve"
    model_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc)
    record = {"id": model_id, "name": name or "Shared classes " + created.astimezone().strftime("%Y-%m-%d %H:%M"),
              "created": created.isoformat(), "n_clusters": n, "options": chosen, "images": len(used), "fitted_on": used,
              "skipped": skipped, "feature_names": feature_names, "marker_names": marker_names,
              "cluster_colors": list(build_cluster_palette(n, include_low_signal=reserve)), "reserve_low_signal": reserve,
              "low_signal_quantile": float(values["low_signal_quantile"]), "far_distance": float(np.percentile(distances, FAR)),
              "sample_per_image": PER_IMAGE, "pooled_pixels": int(len(pooled)), "review": {}, "scale_by": scale_by,
              "slide_scales": slide_scales}
    home = folder(root)
    home.mkdir(parents=True, exist_ok=True)
    with (home / (model_id + ".npz")).open("wb") as handle:
        np.savez_compressed(handle, centres=centres, profile=profile.astype(np.float32), counts=counts)
    edits._atomic(home / (model_id + ".json"), json.dumps(record, indent=2).encode())
    return record


def _limited():
    """Classification's guard for the maths libraries: four threads, one classification at a time (unlimited thread
    pools on a many-core node crash OpenBLAS)."""
    from contextlib import ExitStack

    from threadpoolctl import threadpool_limits

    from .classification import _compute_lock

    stack = ExitStack()
    stack.enter_context(_compute_lock)
    stack.enter_context(threadpool_limits(limits=4))
    return stack


def _assign(features, centres, execution=None):
    """The nearest class of each row and its distance."""
    from .classification import _nearest_auto

    labels = _nearest_auto(features, centres, execution if execution is not None else {})
    squared = np.empty(len(features), np.float32)
    for start in range(0, len(features), 200_000):
        block = features[start:start + 200_000]
        squared[start:start + 200_000] = ((block - centres[labels[start:start + 200_000]]) ** 2).sum(axis=1)
    return labels, np.sqrt(np.maximum(squared, 0))


def apply(root, config, image, output, model_id):
    """Classify *image* with the model: (prepared, found, chosen, seconds) like ``classification.compute``, its run
    marked with the model and carrying its fit checks."""
    from . import classification

    record, arrays = read(root, model_id)
    chosen = classification.options(record["options"])
    started = time.perf_counter()
    scope, prepared = _prepare(root, config, image, output, chosen)
    stack = prepared["stack"]
    if record.get("scale_by") == "slide":
        # The model's scaling of this image's slide; a slide it was not fitted on is scaled from its images here.
        from spacemap.tissue_annotation_utils import PyramidImageReader

        slide = slide_of(image)
        scale = record["slide_scales"].get(slide) or _slide_scale(root, config, [image], output, chosen)
        stack = {**stack, **_in_scale(PyramidImageReader(image), prepared, scale)}
    features = stack["features"][:, _columns(_features(stack), record["feature_names"])]
    markers = _columns(stack["marker_names"], record["marker_names"])
    n = int(record["n_clusters"])
    low = _low_signal(features, {"low_signal_handling": "reserve" if record["reserve_low_signal"] else "include",
                                 "low_signal_quantile": record["low_signal_quantile"]})
    execution = {"device": "cpu"}
    labels = np.full(len(features), n, np.int32)
    with _limited():
        fitted, distances = _assign(np.ascontiguousarray(features[~low], dtype=np.float32), arrays["centres"], execution)
    labels[~low] = fitted
    size = n + (1 if record["reserve_low_signal"] else 0)
    counts = np.bincount(labels, minlength=size)
    z, original = stack["pixels_z_valid"][:, markers], stack["pixels_original_valid"][:, markers]
    sums = lambda a: np.column_stack([np.bincount(labels, weights=a[:, i], minlength=size) for i in range(a.shape[1])])
    z_sums = sums(z)
    cluster_map = np.full(stack["shape"], -1, np.int32)
    cluster_map.flat[stack["valid_indices"]] = labels
    # Checks: pixels that fit no class well; classes whose profile here differs from the model's.
    far = float((distances > record["far_distance"]).mean()) if len(distances) else 0.0
    differ = []
    for i in range(n):
        if counts[i] < 50:
            continue
        here = z_sums[i] / counts[i]
        gap = float(np.sqrt(((here - arrays["profile"][i]) ** 2).mean()))
        differ.append({"id": i, "share": round(100 * counts[i] / max(int(counts.sum()), 1), 2), "gap": round(gap, 3)})
    run_id = sha256((prepared["signature"] + model_id).encode()).hexdigest()[:24]
    found = {"run_id": run_id, "cluster_map": cluster_map, "cluster_colors": record["cluster_colors"], "counts": counts,
             "z_sums": z_sums, "raw_sums": sums(original), "execution": execution,
             "check": {"far_percent": round(100 * far, 2), "classes": differ},
             "provenance": {"scope": str(scope), "scope_revision": edits.revision(scope), "image": str(image),
                            "signature": prepared["signature"], "markers": record["marker_names"],
                            "feature_names": record["feature_names"], "cofactors": stack["cofactors"],
                            "settings": prepared["values"], "class_model": model_id, "class_model_name": record["name"],
                            "normalization_scope": "slide" if record.get("scale_by") == "slide" else "annotation"}}
    # The page's class lists read the model's marker order.
    prepared = {**prepared, "stack": {**stack, "marker_names": record["marker_names"], "feature_names": record["feature_names"],
                                      "pixels_z_valid": z, "pixels_original_valid": original}}
    return prepared, found, chosen, time.perf_counter() - started


def classify(root, config, image, output, model_id):
    """Classify and keep the run (with the model's names), as a batch job does for each image."""
    from . import classification

    prepared, found, chosen, seconds = apply(root, config, image, output, model_id)
    classification.persist(root, image, prepared, found)
    state = classification.review(review_of(root, model_id), np.flatnonzero(found["counts"]).tolist())
    classification.save_review(root, image, found, chosen, state)
    return found, seconds


def flagged(check, threshold=1.0):
    """The classes of an image whose profile differs from the model's by more than *threshold* (z units)."""
    return [item for item in check.get("classes", []) if item["gap"] > threshold]
