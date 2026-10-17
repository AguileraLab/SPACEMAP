"""Explicit model inputs, independent of image visibility and clustering panels."""

import hashlib
import json
from pathlib import Path
import re

from spacemap_workflows.leiden_job import write_json
from .settings import project_lock, read_config


def _path(root):
    return Path(root) / "outputs" / "studio" / "reference_features.json"


def _key(table):
    return hashlib.sha256(str(Path(table).resolve()).encode()).hexdigest()


def saved(root, table):
    path = _path(root)
    return (json.loads(path.read_text()) if path.exists() else {}).get(_key(table))


def kind(name):
    text = name.lower()
    if re.match(r"^(he[_ -]|h&e|hematoxylin|haematoxylin|eosin)", text):
        return "H&E"
    if any(word in text for word in ("area", "eccentricity", "solidity", "perimeter", "axis", "extent", "orientation", "circularity", "roundness", "diameter")):
        return "Morphology"
    return "Markers"


def options(channel):
    from spacemap_gui.viewer import registry
    from spacemap.viewer.sources import image_info
    from . import data_view, manual_reference, markers

    project = registry.session_project(channel) or {}
    table, keys, roles = manual_reference.context(channel)
    view = data_view._table(table)
    config = read_config(project["root"])
    suffix = markers.suffix_of(config, view.names)
    spec = (registry.session_view(channel, mark_seen=False) or {}).get("spec") or {}
    image = next((layer for layer in spec.get("layers", []) if layer.get("type") == "image"), {})
    measured = [c.get("name") for c in image.get("channels", []) if c.get("name") in view.names]
    image_path = registry.resolve(image.get("token", ""))
    if not measured and image_path:
        measured = [name for name in image_info(image_path).channel_names if name in view.names]
    paired = [m["name"] for m in markers.panel(view.names, suffix) if m.get("raw") and m.get("corrected")]
    # Keep the full dataset panel even when a different ROI has fewer channels.
    measured = list(dict.fromkeys(measured + paired))
    excluded = set(keys.values()) | set(roles.values()) | {"index", "original_label", "x_centroid", "y_centroid", "z_centroid"}
    numeric = [name for name in view.names if view.kinds[name] == "value" and name not in excluded
               and not name.lower().startswith(("phenotype_", "leiden_", "som_", "vortex_", "umap_"))]
    entries = [{"name":name, "group":kind(name) if name in measured or kind(name) != "Markers"
                else "Markers" if name.removesuffix(suffix) in measured else "Other numeric"} for name in numeric]
    defaults = [name for name in measured if name in numeric]
    record = saved(project["root"], table)
    # Previous Auto-train settings are already an explicit full training panel.
    if record is None:
        from . import auto_reference
        previous = auto_reference._read(project["root"]).get(_key(table), {}).get("settings", {})
        defaults = previous.get("markers") or defaults
    chosen = record["features"] if record is not None else defaults
    return {"table":str(table), "features":entries, "selected":chosen, "saved":record is not None,
            "missing":[name for name in chosen if name not in numeric],
            "sets":markers.saved_sets(project["root"], config, suffix),
            "allow_missing":True if record is None else record.get("allow_missing", True)}


def save(channel, features, allow_missing=True):
    from spacemap_gui.viewer import registry
    project = registry.session_project(channel) or {}
    if project.get("read_only"):
        raise ValueError("This project is read-only.")
    found = options(channel)
    available = {item["name"] for item in found["features"]}
    if (not isinstance(features, list) or not features or len(features) > 512
            or any(not isinstance(name, str) or name not in available for name in features)):
        raise ValueError("Choose at least one listed measurement for training. IDs and phenotype labels cannot be features.")
    if not isinstance(allow_missing, bool):
        raise ValueError("Choose whether to use available features when some are missing.")
    features = list(dict.fromkeys(features))
    with project_lock(project["root"]):
        path = _path(project["root"])
        records = json.loads(path.read_text()) if path.exists() else {}
        records[_key(found["table"])] = {"features":features, "allow_missing":allow_missing}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, records)
    return {**found, "selected":features, "missing":[], "saved":True, "allow_missing":allow_missing}
