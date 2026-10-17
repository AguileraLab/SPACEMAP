"""Exact cell selections; only small summaries cross the viewer connection.

Rows are anchored by table fingerprint plus ROI and segmentation ID. Transforms
use float64 and the same population standard deviations as marker-bar views.
"""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import uuid

import numpy as np
import pandas as pd

from spacemap import dataio
from spacemap.viewer import cells, consensus_model, marker_transforms
from spacemap.viewer.sources import fingerprint

_LOCK = threading.RLock()
_COMPUTE = threading.BoundedSemaphore(2)
_ITEMS = OrderedDict()
_LIMIT = 12


def validate_rule(rule, markers):
    if not isinstance(rule, dict) or set(rule) - {"conditions", "join", "scope"}:
        raise ValueError("Choose conditions, All/Any and an image scope.")
    conditions = rule.get("conditions")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 20:
        raise ValueError("Use between 1 and 20 conditions.")
    if rule.get("join", "all") not in {"all", "any"}:
        raise ValueError("Choose All or Any conditions.")
    if rule.get("scope", "image") not in {"image", "project"}:
        raise ValueError("Choose this image or all ROIs.")
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {"marker", "transform", "op", "value"}:
            raise ValueError("Each condition needs a marker, scale, comparison and value.")
        if condition["marker"] not in markers:
            raise ValueError("Unknown marker: " + str(condition["marker"]))
        if condition["transform"] not in {"Raw", "Arcsinh", "Z1", "Z2"}:
            raise ValueError("Choose Raw, Arcsinh, Z1 or Z2.")
        if condition["op"] not in {">", ">=", "<", "<=", "=="}:
            raise ValueError("Choose a listed comparison.")
        value = condition["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("Enter a finite number for each condition.")
    return {"conditions": deepcopy(conditions), "join": rule.get("join", "all"), "scope": rule.get("scope", "image")}


def _revision(settings):
    return hashlib.sha256(repr(marker_transforms.revision(settings)).encode()).hexdigest()


def remember(item):
    token = uuid.uuid4().hex
    with _LOCK:
        _ITEMS[token] = item
        while len(_ITEMS) > _LIMIT:
            _ITEMS.popitem(last=False)
    return token


def get(token, *, channel=None, table=None):
    with _LOCK:
        item = _ITEMS.get(token)
        if item is None:
            raise ValueError("This selection expired. Apply the rule again.")
        if channel is not None and item["channel"] != channel:
            raise ValueError("This selection belongs to another viewer.")
        if table is not None and item["settings"]["table"] != table:
            raise ValueError("This selection belongs to another cell table.")
        _ITEMS.move_to_end(token)
    if _revision(item["settings"]) != item["revision"]:
        raise ValueError("The marker measurements changed. Apply the selection again.")
    return item


def _transforms(raw, needed=None):
    needed = needed or {"Raw", "Arcsinh", "Z1", "Z2"}
    result = {"Raw": raw}
    if "Arcsinh" in needed:
        result["Arcsinh"] = np.arcsinh(raw / 5.0)
    finite = np.isfinite(raw)
    if "Z1" in needed:
        counts = finite.sum(axis=0)
        mean = np.where(finite, raw, 0).sum(axis=0) / np.maximum(counts, 1)
        sd = np.sqrt(np.where(finite, (raw - mean) ** 2, 0).sum(axis=0) / np.maximum(counts, 1))
        sd[sd == 0] = 1e-9
        result["Z1"] = (raw - mean) / sd
    if "Z2" in needed:
        count = finite.sum(axis=1, keepdims=True)
        mean = np.where(finite, raw, 0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
        sd = np.sqrt(np.where(finite, (raw - mean) ** 2, 0).sum(axis=1, keepdims=True) / np.maximum(count, 1))
        sd[sd == 0] = 1e-9
        result["Z2"] = (raw - mean) / sd
    return result


def evaluate(channel, settings, rule):
    rule = validate_rule(rule, settings["markers"])
    revision = _revision(settings)
    with _COMPUTE:
        if rule["scope"] == "image":
            rois = [settings["roi"]]
        else:
            rois = (
                dataio.load(settings["table"], columns=[settings["roi_column"]])[settings["roi_column"]]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        groups = []
        total = 0
        hist_samples = [[] for _ in rule["conditions"]]
        needed = {condition["transform"] for condition in rule["conditions"]}
        measured = (
            settings["markers"]
            if "Z2" in needed
            else list(dict.fromkeys(condition["marker"] for condition in rule["conditions"]))
        )
        reading = {**settings, "markers": measured}
        for roi in rois:
            positions, ids = cells.roi_index(
                settings["table"], roi, roi_column=settings["roi_column"], label_column=settings["label_column"]
            )
            if not len(positions):
                continue
            if (ids <= 0).any() or pd.Series(ids).duplicated().any():
                raise ValueError("Each ROI needs unique, positive cell numbers.")
            raw = consensus_model.selected_values(reading, positions, dtype=np.float64)
            views = _transforms(raw, needed)
            tests = []
            for i, condition in enumerate(rule["conditions"]):
                values = views[condition["transform"]][:, measured.index(condition["marker"])]
                operation = {
                    ">": np.greater,
                    ">=": np.greater_equal,
                    "<": np.less,
                    "<=": np.less_equal,
                    "==": np.equal,
                }[condition["op"]]
                tests.append(np.isfinite(values) & operation(values, condition["value"]))
                # A bounded, evenly spaced display sample; selection uses every cell.
                remaining = max(0, 4096 - sum(len(part) for part in hist_samples[i]))
                sample = values[np.linspace(0, len(values) - 1, min(remaining, len(values)), dtype=int)]
                hist_samples[i].append(sample[np.isfinite(sample)])
            mask = np.logical_and.reduce(tests) if rule["join"] == "all" else np.logical_or.reduce(tests)
            groups.append({"roi": roi, "rows": positions[mask].copy(), "ids": ids[mask].copy()})
            total += len(positions)
        if _revision(settings) != revision:
            raise ValueError("Measurements changed while selecting. Try again.")
        histograms = []
        for condition, parts in zip(rule["conditions"], hist_samples):
            sample = np.concatenate(parts) if parts else np.array([])
            counts, edges = np.histogram(sample, bins=24)
            histograms.append(
                {
                    "marker": condition["marker"],
                    "transform": condition["transform"],
                    "counts": counts.tolist(),
                    "edges": edges.tolist(),
                    "sampled": len(sample),
                }
            )
        item = {
            "channel": channel,
            "settings": dict(settings),
            "rule": rule,
            "revision": revision,
            "groups": groups,
            "total": total,
            "histograms": histograms,
        }
        token = remember(item)
    return summary(token, item)


def summary(token, item=None):
    item = item or get(token)
    return {
        "id": token,
        "count": sum(len(group["ids"]) for group in item["groups"]),
        "total": item["total"],
        "by_roi": {group["roi"]: len(group["ids"]) for group in item["groups"]},
        "rule": item["rule"],
        "histograms": item["histograms"],
        "note": "Selection uses every matching cell. Histograms show up to 4,096 sampled cells per condition. Z1 compares cells within each ROI; Z2 compares all measured image markers within each cell; Arcsinh uses cofactor 5.",
    }


def refine(token, channel, roi, ids, mode):
    item = get(token, channel=channel)
    if mode not in {"add", "remove", "toggle"}:
        raise ValueError("Choose Add or Remove.")
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    found, rows = cells.assignment_rows(
        item["settings"]["table"],
        roi,
        ids,
        roi_column=item["settings"]["roi_column"],
        label_column=item["settings"]["label_column"],
    )
    if set(found) != set(ids):
        raise ValueError("Some cells do not belong to this ROI.")
    groups = [
        {"roi": group["roi"], "rows": group["rows"].copy(), "ids": group["ids"].copy()} for group in item["groups"]
    ]
    group = next((value for value in groups if value["roi"] == roi), None)
    if group is None:
        raise ValueError("Apply the rule to this image before refining it.")
    mapping = dict(zip(group["ids"], group["rows"]))
    for label, row in zip(found, rows):
        if mode == "remove" or mode == "toggle" and label in mapping:
            mapping.pop(label, None)
        else:
            mapping[label] = row
    group["ids"] = np.array(sorted(mapping), dtype=np.uint32)
    group["rows"] = np.array([mapping[label] for label in group["ids"]], dtype=np.int64)
    updated = {**item, "groups": groups}
    new = remember(updated)
    return summary(new, updated)


def tile_mask(token, table, roi, labels):
    item = get(token, table=table)
    group = next((group for group in item["groups"] if group["roi"] == roi), None)
    if group is None or not len(group["ids"]):
        return np.zeros_like(labels)
    if "sorted_ids" not in group:
        group["sorted_ids"] = np.sort(group["ids"])
    ids = group["sorted_ids"]
    positions = np.searchsorted(ids, labels)
    return np.where((positions < len(ids)) & (ids[np.minimum(positions, len(ids) - 1)] == labels), labels, 0)


_HEAT_STATS = cells._Cache(8)


def heatmap(token, channel):
    """Exact Z1 means, per ROI and marker, in bounded batches."""
    item = get(token, channel=channel)
    settings = item["settings"]
    width = len(settings["markers"])
    sums = np.zeros((2, width))
    finite_counts = np.zeros((2, width), dtype=np.int64)
    counts = [0, 0]
    with _COMPUTE:
        for group in item["groups"]:
            rows, _ = cells.roi_index(
                settings["table"],
                group["roi"],
                roi_column=settings["roi_column"],
                label_column=settings["label_column"],
            )
            key = (item["revision"], group["roi"])
            stats = _HEAT_STATS.get(key)
            if stats is None:
                stats = marker_transforms.statistics({**settings, "roi": group["roi"]})
                _HEAT_STATS.put(key, stats)
            mean, scale, _ = stats
            selected = np.sort(group["rows"])
            for start in range(0, len(rows), 65536):
                positions = rows[start : start + 65536]
                raw = consensus_model.selected_values(settings, positions, dtype=np.float64)
                values = (raw - mean[0]) / scale[0]
                places = np.searchsorted(selected, positions)
                matches = (
                    (places < len(selected)) & (selected[np.minimum(places, len(selected) - 1)] == positions)
                    if len(selected)
                    else np.zeros(len(positions), bool)
                )
                for index, mask in enumerate((matches, ~matches)):
                    chosen = values[mask]
                    finite = np.isfinite(chosen)
                    sums[index] += np.where(finite, chosen, 0).sum(axis=0)
                    finite_counts[index] += finite.sum(axis=0)
                    counts[index] += int(mask.sum())
    means = np.divide(sums, finite_counts, out=np.full_like(sums, np.nan), where=finite_counts > 0)
    order = np.argsort(-np.nan_to_num(np.abs(means[0] - means[1]), nan=-1), kind="stable")
    if _revision(settings) != item["revision"]:
        raise ValueError("Measurements changed while comparing. Apply the rule again.")
    return {
        "markers": [settings["markers"][i] for i in order],
        "groups": ["Selected", "Other cells"],
        "counts": counts,
        "values": [[float(v) if np.isfinite(v) else None for v in row[order]] for row in means],
        "sampled": counts,
        "note": "Exact mean Z1, using all selected and other cells in this scope. Z1 compares each marker within its ROI. Markers are ordered by the largest difference. Missing measurements are left out per marker.",
    }


def reference_summary(settings, column):
    values = dataio.load(settings["table"], columns=[settings["roi_column"], column])
    names = values[column].fillna("").astype(str).str.strip()
    valid = names.ne("")
    frame = pd.DataFrame({"roi": values.loc[valid, settings["roi_column"]].astype(str), "phenotype": names[valid]})
    counts = frame.groupby(["roi", "phenotype"], sort=True).size()
    return {
        "column": column,
        "total": int(valid.sum()),
        "rows": [
            {"roi": roi, "phenotype": phenotype, "cells": int(count)} for (roi, phenotype), count in counts.items()
        ],
    }


def library(project):
    path = Path(project) / "outputs/studio/rules.json"
    if not path.exists():
        return []
    if path.stat().st_size > 128000:
        raise ValueError("The saved rule library is too large.")
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError("The saved rule library could not be read.")
    return value


def save_rule(project, name, rule, markers):
    from .settings import project_lock
    from spacemap_workflows.leiden_job import write_json

    if not isinstance(name, str) or not 0 < len(name.strip()) <= 100:
        raise ValueError("Enter a short rule name.")
    checked = validate_rule(rule, markers)
    with project_lock(project):
        existing = library(project)
        if any(entry["name"].casefold() == name.strip().casefold() for entry in existing):
            raise ValueError("That rule name already exists. Use a new name to keep both versions.")
        if len(existing) >= 100:
            raise ValueError("This library already has 100 rules.")
        existing.append({"name": name.strip(), "rule": checked})
        path = Path(project) / "outputs/studio/rules.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, existing)
    return existing
