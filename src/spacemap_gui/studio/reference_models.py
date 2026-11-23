"""Saved suggestion models, trained explicitly or by opt-in background updates."""

from datetime import datetime, timezone
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import threading
import time

import numpy as np

from spacemap import dataio, userpaths
from spacemap.viewer import cells, consensus_model, marker_transforms, reference
from spacemap_workflows.leiden_job import write_json
from .settings import project_lock


def directory():
    return userpaths.models_dir() / "studio_reference"


def marker_revision(settings):
    return hashlib.sha256(repr(marker_transforms.revision(settings)).encode()).hexdigest()


def model_path(model_id):
    if not isinstance(model_id, str) or not re.fullmatch(r"[0-9a-f]{32}", model_id):
        raise ValueError("Choose a listed saved model.")
    return directory() / (model_id + ".joblib")


def metadata(model_id):
    path = model_path(current_model_id(model_id)).with_suffix(".json")
    if not path.exists():
        raise ValueError("This saved model is no longer available. Choose another model.")
    return json.loads(path.read_text())


def _auto_index():
    path = directory() / ".auto_models.json"
    return json.loads(path.read_text()) if path.exists() else {"current": {}, "retired": {}}


def current_model_id(model_id):
    """Follow a saved automatic selection after replacement; imports stay fixed."""
    path = model_path(model_id)  # Validate before using identifiers from storage.
    if path.is_file() and path.with_suffix(".json").is_file():
        return model_id
    index = _auto_index()
    slot = index["retired"].get(model_id)
    current = index["current"].get(slot, model_id)
    model_path(current)
    return current


def _source(model_id):
    from spacemap.viewer.sources import fingerprint

    # Read metadata and its artifact together. A background replacement cannot
    # remove either half while a cell request loads it.
    with project_lock(directory()):
        model_id = current_model_id(model_id)
        meta = metadata(model_id)
        path = model_path(model_id)
        stamp = fingerprint(path)
        return model_id, meta, {**_load(str(path), stamp), "_file_stamp": stamp}


def _automatic_slot(meta):
    if (not meta.get("auto_train") or meta.get("imported") or meta.get("imported_from")
            or meta.get("persistent_version")):
        return None
    if not meta.get("table_path") or not meta.get("column"):
        return None
    key = [str(Path(meta["table_path"]).resolve()), meta["column"]]
    if meta.get("base_model_id"):
        key.append(meta["base_model_id"])
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def _replace_automatic(meta):
    """Under the models lock, retain one successful automatic fit per source.

    Artifacts use new IDs to invalidate prediction caches. One small index lets
    older saved selections follow the latest model without keeping old files.
    """
    slot = _automatic_slot(meta)
    if slot is None:
        return meta
    candidates = {}
    for path in directory().glob("*.json"):
        if not re.fullmatch(r"[0-9a-f]{32}", path.stem):
            continue
        try:
            value = json.loads(path.read_text())
            if _automatic_slot(value) == slot and path.with_suffix(".joblib").is_file():
                candidates[path.stem] = value
        except (OSError, ValueError):
            continue
    # A slower, earlier fit must not replace a more recent successful run.
    winner = max(candidates, key=lambda key: (
        candidates[key].get("training_started", candidates[key].get("trained", "")), key))
    retired = set(candidates) - {winner}
    index = _auto_index()
    index["current"][slot] = winner
    index["retired"].update({key: slot for key in retired})
    try:
        write_json(directory() / ".auto_models.json", index)
    except BaseException:
        # Publication did not commit: keep the previous usable models and
        # discard only this run's new files.
        path = model_path(meta["model_id"])
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        raise
    for key in retired:
        path = model_path(key)
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
    # Reduced-feature caches are derived from their parent and retire with it.
    for path in (directory() / "adapted").glob("*.json"):
        try:
            parent = json.loads(path.read_text()).get("adapted_from")
        except (OSError, ValueError):
            continue
        if parent in index["retired"] and not model_path(parent).exists():
            path.with_suffix(".joblib").unlink(missing_ok=True)
            path.unlink(missing_ok=True)
    if retired:
        _load.cache_clear()
        with _ADAPT_LOCK:
            for key, job in list(_ADAPT_JOBS.items()):
                if key[0] == str(directory()) and key[1] in retired and job.done():
                    del _ADAPT_JOBS[key]
        for key in list(_MAPS):
            if key[2] in retired:
                del _MAPS[key]
    return candidates[winner]


def _save_model(model_id, artifact, meta):
    import joblib

    with project_lock(directory()):
        path = model_path(model_id)
        temporary = path.with_suffix(".tmp")
        try:
            joblib.dump(artifact, temporary, compress=3)
            temporary.chmod(0o600)
            os.replace(temporary, path)
            write_json(path.with_suffix(".json"), meta)
        except BaseException:
            temporary.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            raise
        # Nothing old is removed until the replacement is completely written.
        return _replace_automatic(meta) if meta.get("auto_train") else meta


def available(table):
    root = directory()
    if not root.exists():
        return []
    snapshot = []
    with project_lock(root):
        index = _auto_index()
        replaced = {}
        for old_id, slot in index["retired"].items():
            replaced.setdefault(index["current"].get(slot), []).append(old_id)
        paths = [path for path in root.glob("*.json") if re.fullmatch(r"[0-9a-f]{32}", path.stem)]
        for path in sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True)[:100]:
            try:
                if path.with_suffix(".joblib").is_file():
                    snapshot.append((path.stem, json.loads(path.read_text())))
            except (OSError, ValueError):
                continue
    models = []
    from . import project_files
    table_root = next((p for p in Path(table).resolve().parents if project_files.index(p).exists()), None)
    for model_id, value in snapshot:
        if project_files.excluded(model_path(model_id), table_root):
            continue
        try:
            if value.get("table_path") == str(Path(table).resolve()) or _fits(value, table):
                entry = {key: item for key, item in value.items() if key not in {"table_path", "revision"}}
                entry["model_id"] = model_id
                entry["replaces"] = replaced.get(model_id, [])
                entry["same_table"] = value.get("table_path") == str(Path(table).resolve())
                if value.get("table_path") != str(Path(table).resolve()):
                    entry["missing_markers"] = [m for m in value["markers"] if m not in _present(value, table)]
                models.append(entry)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(models, key=lambda entry:not entry["same_table"])


SAMPLE_LIMIT = 20000


def training_percent(value=70):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 50 <= value <= 90 or int(value) != value:
        raise ValueError("Choose a training percentage from 50 to 90; the rest is validation.")
    return int(value)


def training_sample(values, targets, limit=SAMPLE_LIMIT, seed=42, rows=None):
    """Up to *limit* training cells, every phenotype kept in proportion (at least its smaller share), stored with the
    model so it can be refitted to fewer markers in another project."""
    rng = np.random.default_rng(seed)
    chosen = []
    classes, counts = np.unique(targets, return_counts=True)
    for name, count in zip(classes, counts):
        members = np.flatnonzero(targets == name)
        take = min(count, max(50, int(round(limit * count / len(targets)))))
        chosen.append(rng.choice(members, size=take, replace=False))
    chosen = np.sort(np.concatenate(chosen))[:limit] if chosen else np.empty(0, np.int64)
    return {"values": np.asarray(values[chosen], np.float32), "targets": np.asarray(targets[chosen]),
            **({"rows":np.asarray(rows)[chosen]} if rows is not None else {})}


def _present(meta, table):
    try:
        columns = set(dataio.columns(table))
    except (OSError, ValueError):
        return []
    return [marker for marker in meta.get("markers", []) if marker in columns]


def _fits(meta, table):
    """Saved transferable models use matching named features; partial panels
    need retained examples and permission to refit to the available features."""
    if not (meta.get("imported") or meta.get("adaptable")):
        return False
    present = _present(meta, table)
    return bool(present) and (len(present) == len(meta.get("markers", [])) or
                             bool(meta.get("adaptable") and meta.get("allow_missing", True)))


def adapt(model_id, present, *, _snapshot=None):
    """The model refitted to the markers *present*, from its stored training sample, with both versions scored on the
    same held-out cells. Cached per marker set."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from spacemap.viewer.sources import fingerprint

    model_id, meta, source = _snapshot or _source(model_id)
    markers = list(meta["markers"])
    keep = [m for m in markers if m in set(present)]
    if not keep:
        raise ValueError("This table has none of the model's markers.")
    key = hashlib.sha256(json.dumps([model_id, source["_file_stamp"], keep, "holdout-v3"]).encode()).hexdigest()[:32]
    target = directory() / "adapted" / (key + ".joblib")
    with project_lock(directory()):
        if target.is_file() and target.with_suffix(".json").is_file():
            saved = json.loads(target.with_suffix(".json").read_text())
            if saved.pop("sklearn", None) == consensus_model._version():
                return _load(str(target), fingerprint(target)), saved
    sample = source.get("training_sample")
    if not sample:
        raise ValueError("This model was trained before models kept their examples, so it needs all its markers; missing: "
                         + ", ".join(m for m in markers if m not in keep) + ". Retrain it in its project to make it adaptable.")
    values, targets = sample["values"], sample["targets"]
    columns = [markers.index(m) for m in keep]
    full, reduced = clone(source["model"]).set_params(n_jobs=2), clone(source["model"]).set_params(n_jobs=2)
    validation = source.get("validation_sample")
    if validation is not None:
        # The retained examples contain only training cells. The saved validation
        # cells never enter this fit, including when adapting to a smaller panel.
        reduced.fit(values[:, columns], targets)
        full_score = float(balanced_accuracy_score(validation["targets"], source["model"].predict(validation["values"]))) if len(validation['targets']) else None
        reduced_score = float(balanced_accuracy_score(validation["targets"], reduced.predict(validation["values"][:, columns]))) if len(validation['targets']) else None
    else:
        # Legacy models retain all examples. Keep their original internal check
        # distinct from the permanently held-out validation in newer models.
        _, counts = np.unique(targets, return_counts=True)
        train, test = train_test_split(np.arange(len(targets)), test_size=0.25,
                                       stratify=targets if counts.min() >= 2 else None, random_state=42)
        full.fit(values[train], targets[train])
        reduced.fit(values[train][:, columns], targets[train])
        full_score = float(balanced_accuracy_score(targets[test], full.predict(values[test])))
        reduced_score = float(balanced_accuracy_score(targets[test], reduced.predict(values[test][:, columns])))
    reduced.fit(values[:, columns], targets)
    reduced.n_jobs = 1
    missing = [m for m in markers if m not in keep]
    provenance = {**source["provenance"], "markers": keep, "missing_markers": missing, "adapted_from": model_id,
                  "held_out_balanced_accuracy": reduced_score, "full_model_accuracy_same_cells": full_score,
                  "note": f"Refitted without {', '.join(missing) or 'no markers'} from the model's stored training cells; "
                          + (f"held-out agreement {reduced_score:.0%} (the full model scores {full_score:.0%} on the same cells). " if reduced_score is not None else "No saved validation cells. ") +
                          "This tests the stored training examples, not accuracy in the new tissue."}
    # Preserve independent votes for labelled cells in the original table.
    if "rows" in sample:
        current = np.asarray(sample["rows"], np.int64) >= 0
        sample_rows = np.asarray(sample["rows"], np.int64)[current]
        order = np.argsort(sample_rows)
        trained_rows, oob = sample_rows[order], reduced.oob_decision_function_[current][order]
    else:
        # Older artifacts cannot map sampled examples back to rows. Do not
        # present their in-sample predictions as independent review evidence.
        trained_rows = source["rows"]
        oob = np.full((len(trained_rows), len(reduced.classes_)), np.nan)
    artifact = {"revision": key, "model": reduced, "rows": trained_rows,
                "oob": oob, "provenance": provenance,
                "training_origin": source.get("training_origin")}
    with project_lock(directory()):
        # A fit already in progress may finish for its caller, but must not
        # recreate a cache after its automatic parent has been replaced.
        if model_path(model_id).is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            joblib.dump(artifact, temporary, compress=3)
            os.replace(temporary, target)
            write_json(target.with_suffix(".json"), {**provenance, "sklearn": consensus_model._version()})
    return artifact, provenance


_ADAPT_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reference-panel")
_ADAPT_LOCK = threading.Lock()
_ADAPT_JOBS = {}


def _adapt_ready(model_id, present, retry=False, snapshot=None):
    """Refit once per panel off the request thread; cell clicks poll progress."""
    key = (str(directory()), model_id, tuple(present))
    with _ADAPT_LOCK:
        job = _ADAPT_JOBS.get(key)
        if job is None or (retry and job.done() and job.exception() is not None):
            if sum(not work.done() for work in _ADAPT_JOBS.values()) >= 4:
                return None
            for old in list(_ADAPT_JOBS):
                if len(_ADAPT_JOBS) <= 24:
                    break
                if _ADAPT_JOBS[old].done():
                    del _ADAPT_JOBS[old]
            job = _ADAPT_POOL.submit(adapt, model_id, present, _snapshot=snapshot)
            _ADAPT_JOBS[key] = job
    return job.result()[0] if job.done() else None


# ---------------------------------------------------------------- another project's scale
# A model trained on another project's cells (another scanner, stain batch or exposure) sees this project's marker
# values on a different scale. Each marker is first put on the training cells' scale by quantile matching: this
# table's 1st..99th percentiles are mapped onto the training sample's, between them linearly.
QUANTILES = np.linspace(0.01, 0.99, 99)
_MAPS = {}


class ScaleMatched:
    """A model that puts each marker on its training scale before predicting."""

    def __init__(self, model, maps):
        self.model, self.maps = model, maps
        self.classes_ = model.classes_

    def predict_proba(self, values):
        values = np.array(values, dtype=float, copy=True)
        for column, pair in enumerate(self.maps):
            if pair is not None:
                values[:, column] = _mapped(values[:, column], *pair)
        return self.model.predict_proba(values)


def _mapped(values, source, target):
    """Piecewise-linear quantile map; beyond the ends it continues the end segments, so the brightest and dimmest
    cells keep their order instead of being flattened."""
    out = np.interp(values, source, target)
    low_slope = (target[1] - target[0]) / max(source[1] - source[0], 1e-12)
    high_slope = (target[-1] - target[-2]) / max(source[-1] - source[-2], 1e-12)
    low, high = values < source[0], values > source[-1]
    out[low] = target[0] + (values[low] - source[0]) * low_slope
    out[high] = target[-1] + (values[high] - source[-1]) * high_slope
    return out


def _row_count(table):
    import pyarrow.parquet as pq

    for candidate in (Path(table), dataio.fast_copy_path(table)):
        if candidate and Path(candidate).is_file() and str(candidate).endswith(".parquet"):
            return pq.ParquetFile(candidate).metadata.num_rows
    return len(dataio.load(table, columns=[dataio.columns(table)[0]]))


def scale_maps(settings, model_id, present, limit=20000, *, _snapshot=None):
    """Per marker (in *present*'s order): (this table's quantiles, the training cells' quantiles), or None when a
    marker cannot be matched (no spread). None overall when the model kept no training cells."""
    from spacemap.viewer.sources import fingerprint

    model_id, meta, source = _snapshot or _source(model_id)
    key = (str(settings["table"]), fingerprint(settings["table"]), model_id, tuple(present), marker_revision(settings))
    if key in _MAPS:
        return _MAPS[key]
    sample = source.get("training_sample")
    if not sample:
        return None
    trained = np.asarray(sample["values"], float)[:, [list(meta["markers"]).index(m) for m in present]]
    count = _row_count(settings["table"])
    rows = np.sort(np.random.default_rng(0).choice(count, size=min(count, limit), replace=False))
    here = np.asarray(consensus_model.selected_values({**settings, "markers": list(present)}, rows), float)
    maps = []
    for column in range(len(present)):
        # Size, shape and H&E channels must retain their measured scale.
        from .reference_features import kind
        if meta.get("feature_kinds", {}).get(present[column], kind(present[column])) != "Markers":
            maps.append(None)
            continue
        mine, theirs = here[:, column], trained[:, column]
        mine, theirs = mine[np.isfinite(mine)], theirs[np.isfinite(theirs)]
        if len(mine) < 50 or len(theirs) < 50 or np.ptp(mine) <= 0 or np.ptp(theirs) <= 0:
            maps.append(None)
            continue
        source_q, target_q = np.quantile(mine, QUANTILES), np.quantile(theirs, QUANTILES)
        source_q = np.maximum.accumulate(source_q + np.arange(len(source_q)) * 1e-9)  # strictly rising for interp
        maps.append((source_q, target_q))
    if len(_MAPS) > 32:
        _MAPS.clear()
    _MAPS[key] = maps
    return maps


def checked_rows(settings):
    with reference.database(settings["table"]) as db:
        if not db or not db.execute("SELECT name FROM sqlite_master WHERE name='checked'").fetchone():
            return []
        checked = list(
            db.execute(
                "SELECT roi,label FROM checked WHERE column_name=? AND checked=1", (settings["training_column"],)
            )
        )
    rows = []
    for roi in sorted({row[0] for row in checked}):
        _, positions = cells.assignment_rows(
            settings["table"],
            roi,
            [row[1] for row in checked if row[0] == roi],
            roi_column=settings["roi_column"],
            label_column=settings["label_column"],
        )
        rows.extend(positions.tolist())
    return rows


def train(settings, folder, report):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from .tasks import source_revision
    from spacemap.viewer.sources import fingerprint
    from . import reference_examples

    reference_examples.require_reusable(settings)
    started = datetime.now(timezone.utc).isoformat()
    percent = training_percent(settings.get("training_percent", 70))
    column = settings["training_column"]
    if column not in dataio.columns(settings["table"]):
        raise ValueError("Choose an existing reference column.")
    revision = marker_revision(settings)
    reviews = reference.path_for(settings["table"])
    review_stamp = fingerprint(reviews) if reviews.exists() else None
    if settings.get("checked_only"):
        settings = {**settings, "checked_rows": checked_rows(settings)}
    if settings.get('selected_images') is not None and settings.get('roi_column'):
        from . import phenotyping_images
        selected = np.flatnonzero(phenotyping_images.mask(dataio.load(settings['table'], columns=[settings['roi_column']]), settings))
        if settings.get('checked_rows') is not None:
            selected = np.intersect1d(selected, settings['checked_rows'])
        settings = {**settings, 'checked_rows':selected}
    report("Choosing labelled examples, leaving out cells marked Needs review…", 0.1)
    minimum_classes = 1 if settings.get('base_model_id') and settings.get('include_saved_examples') else 2
    try:
        rows, targets, counts, flagged = consensus_model._sample(settings, lambda message: report(message),
            limit=10000 if settings.get("auto_train") else consensus_model.MAX_EXAMPLES, minimum_classes=minimum_classes)
    except ValueError as exc:
        if "two phenotypes" in str(exc):
            raise ValueError("Label at least five cells per phenotype." if minimum_classes == 1 else
                             "Label at least five cells for each of two phenotypes.") from exc
        raise
    values = consensus_model.selected_values(settings, rows, streaming=True)
    valid = np.isfinite(values).all(axis=1)
    missing = int((~valid).sum())
    rows, targets, values = rows[valid], targets[valid], values[valid]
    classes, numbers = np.unique(targets, return_counts=True)
    keep = np.isin(targets, classes[numbers >= 5])
    # _sample omits singleton labels; still report them to the user.
    excluded = sorted(set(counts) - set(classes[numbers >= 5]))
    rows, targets, values = rows[keep], targets[keep], values[keep]
    if len(np.unique(targets)) < minimum_classes:
        raise ValueError(
            "Label at least five cells per phenotype."
            if minimum_classes == 1 else "Label at least five cells for each of two phenotypes."
        )
    training, testing = train_test_split(
        np.arange(len(rows)),
        test_size=max(len(np.unique(targets)), int(np.ceil((100 - percent) * len(rows) / 100))),
        stratify=targets,
        random_state=42,
    )
    # consensus_model.predict uses searchsorted for the training-row/OOB lookup.
    training, testing = np.sort(training), np.sort(testing)
    from . import reference_splits
    split = reference_splits.make(settings, rows, targets, training, testing)
    candidates = [("Random Forest", RandomForestClassifier), ("Extra Trees", ExtraTreesClassifier)]
    base = None
    parent = None
    if settings.get("base_model_id"):
        from sklearn.base import clone
        _, base_meta, base_artifact = _source(settings["base_model_id"])
        base = base_artifact["model"]
        if not isinstance(base, (RandomForestClassifier, ExtraTreesClassifier)):
            raise ValueError("Choose a Random Forest or Extra Trees reference model.")
        if base_meta["sklearn"] != consensus_model._version():
            raise ValueError("The imported model's software version does not match.")
        candidates = [(base_meta["method"], type(base))]
        parent = (base_meta, base_artifact)
    fit_markers, fit_values, fit_targets, fit_rows, bank, replayed = reference_examples.combine(
        settings, values, targets, rows, training, parent)
    if parent is not None and settings.get('include_saved_examples'):
        lost = set(map(str, parent[1]['model'].classes_)) - set(map(str, fit_targets))
        from spacemap.phenotype_policy import allowed
        lost = {name for name in lost if allowed([name], settings.get('excluded_training_phenotypes', []))[0]}
        if lost:
            raise ValueError('Training would lose saved phenotypes: ' + ', '.join(sorted(lost)) +
                             '. Keep their labels, or choose New model.')
    validation_bank = reference_examples.validation_bank(settings, values, targets, rows, testing, parent)
    if len(np.unique(fit_targets)) < 2:
        raise ValueError('Combined training needs at least two phenotypes. Add another phenotype with five usable labels.')
    feature_positions = [settings['markers'].index(m) for m in fit_markers]
    validation = {"values": values[testing][:, feature_positions], "targets": targets[testing], "rows": rows[testing]}
    sample = training_sample(fit_values, fit_targets, rows=fit_rows)
    if settings.get("auto_train"):
        candidates = candidates[:1]
    saved = []
    for index, (name, constructor) in enumerate(candidates):
        report("Comparing " + name + " on held-out reference cells…", 0.25 + 0.25 * index)
        model = constructor(
            n_estimators=64 if settings.get("auto_train") else 96,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
            bootstrap=True,
            oob_score=True,
        )
        if base is not None:
            # A new classifier is fitted from explicit examples, never mutated trees.
            model = clone(base).set_params(n_jobs=2, bootstrap=True, oob_score=True, warm_start=False)
        model.fit(fit_values, fit_targets)
        predicted = model.predict(validation['values'])
        score = float(balanced_accuracy_score(targets[testing], predicted))
        per_class = {
            str(label): float(np.mean(predicted[targets[testing] == label] == label)) for label in np.unique(targets)
        }
        report("Preparing " + name + " for fast cell suggestions…", 0.4 + 0.25 * index)
        model.n_jobs = 1
        model_id = uuid.uuid4().hex
        provenance = {
            "method": name,
            "column": column,
            "source": "table",
            "roi": "all ROIs in this table",
            "table": Path(settings["table"]).name,
            "markers": fit_markers,
            "name": settings.get('model_name', '').strip() or (Path(settings.get('project_root', folder)).name + ' · ' + column),
            "model_timezone": settings.get("model_timezone", "UTC"),
            "project_key": reference_examples.source_key(settings.get('project_root', folder)),
            "classes": [str(c) for c in model.classes_],
            "saved_example_cells": replayed,
            "total_fit_cells": len(fit_targets),
            "include_saved_examples": bool(settings.get('include_saved_examples')),
            "excluded_training_phenotypes": settings.get('excluded_training_phenotypes', []),
            "archived_features": sorted({m for group in bank for m in group['markers']}),
            "unused_features": [m for m in settings['markers'] if m not in fit_markers],
            "feature_kinds": settings.get("feature_kinds", {}),
            "allow_missing": settings.get("allow_missing", True),
            "examples": len(rows),
            "labelled_available": sum(counts.values()),
            "sampling_limit_per_class": consensus_model.PER_CLASS,
            "training_cells": len(training),
            "training_percent": percent,
            "validation_percent": 100 - percent,
            "validation_held_out": True,
            "split_id": split['id'],
            "split_seed": 42,
            "flagged_excluded": flagged,
            "missing_excluded": missing,
            "needs_examples": excluded,
            "revision": marker_revision({**settings, 'markers': fit_markers}),
            "held_out_balanced_accuracy": score,
            "held_out_cells": len(testing),
            "held_out_macro_f1": float(f1_score(targets[testing], predicted, average="macro", zero_division=0)),
            "held_out_per_class": per_class,
            "checked_only": bool(settings.get("checked_only")),
            "auto_train": bool(settings.get("auto_train")),
            "persistent_version": bool(settings.get("project_root")),
            "label_revision": settings["revision"],
            "model_id": model_id,
            "trained": datetime.now(timezone.utc).isoformat(),
            "training_started": started,
            "note": "Agreement with held-out reference labels; neighbouring cells may be related. This is not independent biological validation.",
        }
        provenance["marker_importance"] = {m: round(float(w), 4) for m, w in zip(fit_markers, model.feature_importances_)}
        provenance["adaptable"] = True  # its training sample lets it be refitted to fewer markers in another project
        if settings.get("base_model_id"):
            provenance["base_model_id"] = settings["base_model_id"]
            provenance['phenotype_groups'] = base_meta.get('phenotype_groups', {})
            provenance['keep_detailed_predictions'] = base_meta.get('keep_detailed_predictions', True)
        if settings.get("training_sources"):
            provenance["training_images"] = [Path(item["table"]).name for item in settings["training_sources"]]
        provenance["training_scope"] = 'Current labels + saved training examples' if replayed else "Current project's labelled cells only"
        artifact = {
            "revision": model_id,
            "model": model,
            "rows": rows[training],
            "oob": model.oob_decision_function_[replayed:].copy(),
            "provenance": provenance,
            "training_sample": {**sample, "markers": fit_markers},
            "example_archive": bank,
            "validation_sample": validation,
            "validation_archive": validation_bank,
            "training_origin": settings.get("training_origin"),
        }
        saved.append(
            (
                model_id,
                artifact,
                {
                    **provenance,
                    "table_path": str(Path(settings["table"]).resolve()),
                    "revision": marker_revision({**settings, 'markers': fit_markers}),
                    "sklearn": consensus_model._version(),
                    **({"training_sources": [{**s, 'markers': fit_markers,
                        'marker_revision': marker_revision({**s, 'markers': fit_markers})}
                        for s in settings['training_sources']]} if settings.get('training_sources') else {}),
                },
            )
        )
    if (
        marker_revision(settings) != revision
        or (not settings.get("auto_train") and (source_revision(settings) != settings["revision"]
            or (fingerprint(reviews) if reviews.exists() else None) != review_stamp))
    ):
        raise ValueError(
            "Measurements, review flags or reference labels changed during training. Train again with the current labels."
        )
    report("Saving models for later use…", 0.96)
    if settings.get("training_sources"):
        from .reference_collection import check_sources
        check_sources(settings["training_sources"])
    models = [_save_model(model_id, artifact, meta) for model_id, artifact, meta in saved]
    reference_splits.save(settings.get('project_root', folder), split, models)
    if settings.get('project_root'):
        from . import reference_setup
        reference_setup.publish_training(settings['project_root'], models, settings.get('base_model_id', ''),
                                         settings.get('models_folder'), automatic=bool(settings.get('auto_train')))
    return {"table": str(settings["table"]), "models": models, "examples": len(rows),
            "split_id": models[0].get("split_id", split['id']), "auto_train": bool(settings.get("auto_train"))}


@lru_cache(maxsize=6)
def _load(path, stamp):
    import joblib

    return joblib.load(path)


def _for_table(artifact, table):
    """Use independent votes from this table's training rows, even in a pooled model."""
    origin = artifact.get("training_origin")
    if not origin:
        return artifact
    positions = np.asarray(artifact["rows"], np.int64)
    matches = np.asarray(origin["tables"])[positions] == str(Path(table).resolve())
    local_rows = np.asarray(origin["rows"], np.int64)[positions[matches]]
    order = np.argsort(local_rows)
    return {**artifact, "rows": local_rows[order] if len(order) else np.array([-1]),
            "oob": artifact["oob"][matches][order] if len(order) else np.zeros((1, len(artifact["model"].classes_)))}


def _scan_predictor(model_id, present, own, allow_missing, snapshot=None, table=None):
    """Score unlabelled cells by their measured panel, inside the scan worker."""
    def predict(artifact, values, rows):
        out = consensus_model.predict(artifact, values, rows)
        if not allow_missing or not len(values):
            return out
        masks, assignments = np.unique(np.isfinite(values), axis=0, return_inverse=True)
        for number, mask in enumerate(masks):
            if mask.all() or not mask.any():
                continue
            keep = [name for name, usable in zip(present, mask) if usable]
            reduced = _adapt_ready(model_id, keep, snapshot=snapshot)
            while reduced is None:
                time.sleep(.05)
                reduced = _adapt_ready(model_id, keep, snapshot=snapshot)
            if own and table:
                reduced = _for_table(reduced, table)
            if not own:
                reduced = {**reduced, "rows":np.array([-1]), "oob":np.zeros((1,len(reduced["model"].classes_)))}
                if isinstance(artifact["model"], ScaleMatched):
                    reduced = {**reduced, "model":ScaleMatched(reduced["model"],
                        [pair for pair, usable in zip(artifact["model"].maps, mask) if usable])}
            group = np.flatnonzero(assignments == number)
            probabilities = consensus_model.predict(reduced, values[group][:, mask], rows[group])
            target = {name:i for i,name in enumerate(artifact["model"].classes_)}
            out[group] = 0
            for j,name in enumerate(reduced["model"].classes_):
                out[group,target[name]] = probabilities[:,j]
        return out
    return predict


def request(settings, model_id, *, label=None, prepare=False, scan=False, exclude=(), retry=False, cancel=False):
    snapshot = _source(model_id)
    model_id, meta, artifact = snapshot
    source = next((item for item in meta.get("training_sources", []) if item["table"] == str(Path(settings["table"]).resolve())), None)
    own = bool(source) or meta.get("table_path") == str(Path(settings["table"]).resolve())
    if not own and not _fits(meta, settings["table"]):
        raise ValueError("This dataset does not match this model's features. Choose another saved model or retrain with shared features.")
    source_settings = {**settings, "markers":meta["markers"]}
    present = _present(meta, settings["table"])
    if meta["sklearn"] != consensus_model._version():
        raise ValueError("The model software version changed. Train a fresh reference model.")
    expected_revision = source["marker_revision"] if source else meta["revision"]
    if own and marker_revision(source_settings) != expected_revision:
        raise ValueError("The marker measurements changed since training. Train a fresh reference model.")
    rows, values = None, None
    if not prepare and not scan:
        _, rows = cells.assignment_rows(settings["table"], settings["roi"], [label],
            roi_column=settings["roi_column"], label_column=settings["label_column"])
        if len(rows) != 1:
            raise ValueError("This cell has no unique matching row.")
        values = consensus_model.selected_values({**settings, "markers":present}, rows)
        finite = np.isfinite(values[0])
        present = [name for name, usable in zip(present, finite) if usable]
        values = values[:, finite]
    if not present:
        raise ValueError("This cell/table has none of this model's selected features measured. Choose another model or training feature list.")
    missing = [name for name in meta["markers"] if name not in present]
    if missing and (not meta.get("allow_missing", True) or not meta.get("adaptable")):
        raise ValueError("Missing model features: " + ", ".join(missing) + ". Enable available-feature models in Model training and train again.")
    settings = {**settings, "markers":present}
    if missing:
        artifact = _adapt_ready(model_id, present, retry=retry, snapshot=snapshot)
        if artifact is None:
            return {"status":"preparing", "message":"Preparing a model using " + str(len(present)) + "/" +
                    str(len(meta["markers"])) + " features. Missing: " + ", ".join(missing) + ". This panel is prepared once."}
    from .mr_ml import panel_assessment
    assessed = panel_assessment(snapshot, present, artifact)['phenotypes']
    panel_check = {'suggested': [p['phenotype'] for p in assessed if p['suggested']],
                   'needs_checking': [p['phenotype'] for p in assessed if not p['suggested']]}
    artifact = {**artifact, 'provenance': {**artifact['provenance'], 'panel_check': panel_check,
                'needs_examples': artifact['provenance'].get('needs_examples', []) if own else []}}
    if not own:
        # Another table's cells: no cell here was a training example, so its out-of-bag votes do not apply.
        # (row -1 matches no cell; an empty list would break the lookup).
        artifact = {**artifact, "rows": np.array([-1], dtype=np.int64), "oob": np.zeros((1, len(artifact["model"].classes_)))}
    elif source:
        artifact = _for_table(artifact, settings["table"])
    if not own and settings.get("match_scale", True):
        maps = scale_maps(settings, model_id, present, _snapshot=snapshot)
        if maps and any(pair is not None for pair in maps):
            matched = sum(pair is not None for pair in maps)
            artifact = {**artifact, "model": ScaleMatched(artifact["model"], maps), "revision": str(artifact["revision"]) + "-scaled",
                        "provenance": {**artifact["provenance"], "scale_matched": matched,
                                       "scale_note": f"{matched} of {len(maps)} markers were put on the training cells' scale "
                                                     "(quantile matching), as this project was measured differently."}}
    from spacemap.phenotype_policy import prediction_positions
    blocked = settings.get('excluded_prediction_phenotypes', [])
    permitted = prediction_positions(artifact['model'].classes_, blocked, meta.get('phenotype_groups', {}))
    if blocked:
        artifact = {**artifact, 'revision': str(artifact['revision']) + '-policy-' +
                    hashlib.sha256(json.dumps(sorted(blocked)).encode()).hexdigest()[:16],
                    'provenance': {**artifact['provenance'], 'excluded_prediction_phenotypes': blocked}}
    if prepare:
        return {"status": "ready", "message": "Saved reference model ready" + (" (marker values put on its training scale)."
                if artifact["provenance"].get("scale_matched") else "."), "provenance": artifact["provenance"]}
    if scan:
        predictor = _scan_predictor(model_id, present, own, meta.get('adaptable') and meta.get('allow_missing', True), snapshot, settings['table'])
        with consensus_model._LOCK:
            return consensus_model._scan(
                settings, artifact, exclude, cancel, retry, lambda: not own or marker_revision(source_settings) == expected_revision,
                predictor=lambda a, v, r: predictor(a, v, r)[:, permitted]
            )
    probabilities = consensus_model.predict(artifact, values, rows)[0]
    if not np.isfinite(probabilities).all():
        raise ValueError("This labelled cell has no independent model votes. Retrain the reference model to review it; other cells can still receive suggestions.")
    detail_probabilities = [{"phenotype": str(name), "probability": float(probabilities[i])}
                            for i, name in enumerate(artifact['model'].classes_) if i in permitted]
    from .mr_ml import prediction_groups
    names, positions = prediction_groups(artifact['model'].classes_, meta.get('phenotype_groups', {}), blocked=blocked)
    probabilities = np.asarray([probabilities[p].sum() for p in positions])
    order = np.argsort(-probabilities, kind="stable")[:3]
    return {
        "status": "ready",
        "label": int(label),
        "choices": [
            {"phenotype": str(names[i]), "probability": float(probabilities[i])} for i in order
        ],
        "provenance": {**artifact["provenance"], "features_used":len(present), "features_total":len(meta["markers"]),
                       "missing_markers":missing},
        "probabilities": [{"phenotype": str(name), "probability": float(probabilities[i])}
                          for i, name in enumerate(names)],
        "detailed_probabilities": detail_probabilities,
    }
