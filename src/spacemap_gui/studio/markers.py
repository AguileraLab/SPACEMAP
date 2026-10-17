"""The markers a clustering method can use, by name, as measured (raw) or batch corrected, and the saved sets.

The combined table holds each marker as measured (``CD45``) and, after batch
correction, corrected (``CD45`` + Tab 6's suffix, ``_corr`` by default). Studio
lists each marker once, by its name, and a method's settings keep the real
columns (``CD45_corr`` when corrected values are chosen), as classic Tab 7 does.

Saved sets come from two places: the marker sets of the image viewer's
right-hand pane (``outputs/viewer/display/marker_sets.json``) and classic Tab 7's
named marker panels (``tab7_marker_workspace``); both are given by marker name.
"""

import json
from pathlib import Path
import re

DEFAULT_SUFFIX = "_corr"
SUFFIX = re.compile(r"_[A-Za-z0-9_]{1,30}")
# Classic Tab 7's columns that are not markers (leiden_clustering.py).
NOT_MARKERS = {
    "cell_id",
    "batch_id",
    "tissue",
    "x_centroid",
    "y_centroid",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "major_axis",
    "minor_axis",
    "orientation",
    "label",
    "index",
    "major_axis_length",
    "minor_axis_length",
    "original_filename",
    "corrected",
    "tls",
}
NOT_PREFIXES = ("umap_", "leiden_", "phenotype_", "som_", "vortex_", "cn_", "meta_", "tls_")


def suffix_of(config, columns=()):
    """Tab 6's corrected suffix, else the one the table uses."""
    saved = ((config.get("tab_configurations") or {}).get("tab6_batch_correction") or {}).get("bc_config")
    value = (saved if isinstance(saved, dict) else {}).get("corrected_suffix")
    if isinstance(value, str) and SUFFIX.fullmatch(value):
        return value
    names = set(columns)
    for candidate in (DEFAULT_SUFFIX, "_corrected", "_batch_corrected"):
        if any(name.endswith(candidate) and name[: -len(candidate)] in names for name in names):
            return candidate
    return DEFAULT_SUFFIX


def _is_marker(name):
    lowered = name.lower()
    return lowered not in NOT_MARKERS and not lowered.startswith(NOT_PREFIXES) and not lowered.endswith("_norm")


def panel(columns, suffix):
    """Each marker once: [{"name", "raw", "corrected"}], in the table's order."""
    names = [str(c) for c in columns if _is_marker(str(c))]
    present = set(names)
    found = {}
    for name in names:
        base = name[: -len(suffix)] if suffix and name.endswith(suffix) and len(name) > len(suffix) else name
        item = found.setdefault(base, {"name": base, "raw": False, "corrected": False})
        if base == name:
            item["raw"] = True
        else:
            item["corrected"] = True
    return [item for item in found.values() if item["raw"] or item["name"] + suffix in present]


def base_of(column, suffix):
    return column[: -len(suffix)] if suffix and column.endswith(suffix) and len(column) > len(suffix) else column


def columns_for(names, source, suffix, available):
    """The real columns for marker *names*: corrected ones when *source* is "corrected" (and they exist)."""
    present = set(available)
    out = []
    for name in names:
        column = name + suffix if source == "corrected" else name
        if column not in present:
            raise ValueError(f"The table has no {column}.")
        out.append(column)
    return out


def saved_sets(root, config, suffix):
    """[{"name", "from", "markers"}]: the viewer's marker sets and classic Tab 7's marker panels, by marker name."""
    sets = []
    path = Path(root) / "outputs" / "viewer" / "display" / "marker_sets.json"
    try:
        data = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    for name, markers in ((data.get("sets") if isinstance(data, dict) else None) or {}).items():
        names = [str(item.get("name")) for item in markers or [] if isinstance(item, dict) and item.get("name")]
        if names:
            sets.append({"name": str(name), "from": "Image viewer", "markers": names})
    workspace = ((config.get("tab_configurations") or {}).get("tab7_phenotyping") or {}).get("tab7_marker_workspace")
    panels = (workspace if isinstance(workspace, dict) else {}).get("marker_panels")
    for name, columns in (panels if isinstance(panels, dict) else {}).items():
        names = list(dict.fromkeys(base_of(str(column), suffix) for column in columns or []))
        if names:
            sets.append({"name": str(name), "from": "Tab 7", "markers": names})
    return sets


def choices(root, config, columns, current=()):
    """What the page needs: the markers, the suffix, the saved sets, and the current choice by name and source."""
    suffix = suffix_of(config, columns)
    markers = panel(columns, suffix)
    current = [str(c) for c in current or []]
    corrected = bool(current) and all(c.endswith(suffix) for c in current)
    any_corrected = any(item["corrected"] for item in markers)
    grouping = [name for name in ("batch_id", "slide", "slide_id") if name in set(columns)]
    return {
        "suffix": suffix,
        "markers": markers,
        # What normalisation can work within: each ROI (batch_id) or each slide, when the table has them.
        "groups": grouping,
        # Phenotype and cluster columns, for figures.
        "phenotypes": [str(c) for c in columns if str(c).startswith("phenotype_")],
        "sets": saved_sets(root, config, suffix),
        "source": "corrected" if corrected or (not current and any_corrected) else "raw",
        "chosen": [base_of(c, suffix) for c in current],
    }
