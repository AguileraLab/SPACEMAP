"""User-wide phenotype names and colours, independent of project cell labels."""

from functools import lru_cache
from contextlib import contextmanager
import colorsys
import hashlib
import json
import os
import re
import tempfile
import threading

from . import userpaths

_OLD_BASE_COLOURS = {
    "B":"#24306e", "CD4":"#e78ac3", "CD8":"#e31a1c", "DC":"#2ca02c",
    "Endothelial":"#006d6f", "Epithelial":"#7b3294", "Fibroblast":"#737373",
    "IMC":"#348abd", "Macrophage":"#9ecae1", "Plasma":"#b65cff", "Treg":"#ff9800", "Others":"#d9d9d9",
}
# Palette from the user's phenotype legend (Figure 2's phenotype palette).
BASE_COLOURS = {
    "B":"#262a78", "CD4":"#e78ac3", "CD8":"#d62728", "DC":"#2ca02c",
    "Endothelial":"#006d5b", "Epithelial":"#7a3e85", "Fibroblast":"#7f7f7f",
    "IMC":"#377eb8", "Macrophage":"#9ecae1", "Plasma":"#c77cff", "Treg":"#ff9800", "Others":"#d9d9d9",
}
ALIASES = {"B cell":"B", "Plasma B cell":"Plasma", "CAF":"Fibroblast", "Endothelia":"Endothelial",
           "Epithelia":"Epithelial", "Mac":"Macrophage", "Other":"Others"}
DEFAULT_PHENOTYPE_COLORS = {**BASE_COLOURS, **{alias:BASE_COLOURS[name] for alias,name in ALIASES.items()}}
_LOCK = threading.RLock()


def path():
    return userpaths.config_dir() / "phenotype_presets.json"


def canonical(name):
    name = str(name).strip()
    known = {value.casefold(): value for value in BASE_COLOURS}
    known.update({alias.casefold(): value for alias, value in ALIASES.items()})
    return known.get(name.casefold(), name)


def _validate(entries, allow_auto=False):
    if not isinstance(entries, list) or len(entries) > 200:
        raise ValueError("Use a list of at most 200 phenotype names and colours.")
    seen, checked = set(), []
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"name", "color"}:
            raise ValueError("Each phenotype needs a name and colour.")
        name, colour = item["name"], item["color"]
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 128 or re.search(r"[\x00-\x1f]", name):
            raise ValueError("Use phenotype names of 1–128 characters without line breaks.")
        name = canonical(name)
        key = re.sub(r"[^a-z0-9_]+", "_", name.casefold()).strip("_")
        if key in seen or not key:
            raise ValueError("Use distinct phenotype names.")
        if not (allow_auto and colour is None) and (not isinstance(colour, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", colour)):
            raise ValueError(f"{name}: choose a colour as #rrggbb.")
        seen.add(key)
        checked.append({"name":name, "color":colour.lower() if colour else None})
    return checked


def signature():
    target = path()
    try:
        stat = target.stat()
        return str(target), stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return str(target), None, None


@lru_cache(maxsize=8)
def _read(filename, stamp, size):
    if stamp is None:
        return {"entries":[{"name":name, "color":colour} for name,colour in BASE_COLOURS.items()], "assigned":{}}
    with open(filename) as handle:
        data = json.load(handle)
    entries = _validate(data["entries"])
    # Upgrade the earlier approximate defaults, retaining deliberate user edits.
    if data.get("version", 1) < 2:
        for item in entries:
            if item["color"] == _OLD_BASE_COLOURS.get(item["name"]):
                item["color"] = BASE_COLOURS[item["name"]]
    return {"entries":entries, "assigned":dict(data.get("assigned") or {})}


def read():
    entries = [dict(item) for item in _read(*signature())["entries"]]
    revision = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"entries":entries, "revision":revision}


@contextmanager
def _locked():
    import fcntl
    from .sharing import open_lock
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open_lock(target.with_suffix(".lock")) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _write(entries, assigned):
    target, temporary = path(), None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, prefix=".phenotypes-", delete=False) as handle:
            temporary = handle.name
            json.dump({"version":2, "entries":entries, "assigned":assigned}, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    _read.cache_clear()


def _palette(data):
    values = {**BASE_COLOURS, **data["assigned"]}
    values.update({item["name"]:item["color"] for item in data["entries"]})
    # A single case-insensitive identity per phenotype, including known aliases.
    return {canonical(name).casefold():colour for name,colour in values.items()}


def _new_colours(count, avoid):
    """Choose unused colours far apart in Lab; never cycle the old 20/60 colours."""
    import numpy as np
    from .acc_utils import _srgb_to_lab, default_palette
    taken = set(c.lower() for c in avoid)
    candidates = set(default_palette(60))
    hues = max(72, (len(taken) + count) // 6 + 1)
    for hue in range(hues):
        for light in (.38, .55, .72):
            for saturation in (.5, .75, 1):
                rgb = colorsys.hls_to_rgb(hue / hues, light, saturation)
                candidates.add("#%02x%02x%02x" % tuple(round(c * 255) for c in rgb))
    candidates = sorted(candidates - taken)
    labs = np.array([_srgb_to_lab(c) for c in candidates])
    distance = np.full(len(candidates), np.inf)
    for colour in taken:
        distance = np.minimum(distance, ((labs - _srgb_to_lab(colour))**2).sum(axis=1))
    chosen = []
    for _ in range(count):
        index = int(np.argmax(distance))
        chosen.append(candidates[index])
        distance = np.minimum(distance, ((labs - labs[index])**2).sum(axis=1))
        distance[index] = -1
    return chosen


def ensure(names):
    """Resolve names globally, durably allocating new colours only once.

    Allocations are separate from the editable vocabulary so viewing cluster
    numbers or ROI categories does not add them to the phenotype dropdown.
    """
    names = list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))
    data = _read(*signature())
    known = _palette(data)
    pending = {canonical(name).casefold():canonical(name) for name in names if canonical(name).casefold() not in known}
    if pending:
        with _locked():
            data = _read(*signature())
            known = _palette(data)
            pending = {key:name for key,name in sorted(pending.items()) if key not in known}
            if pending:
                assigned = dict(data["assigned"])
                for name, colour in zip(pending.values(), _new_colours(len(pending), known.values())):
                    assigned[name] = colour
                _write(data["entries"], assigned)
                known = _palette(_read(*signature()))
    return {name:known[canonical(name).casefold()] for name in names}


def save(entries, revision):
    checked = _validate(entries, allow_auto=True)
    with _locked():
        if read()["revision"] != revision:
            raise ValueError("The phenotype preset changed. Reopen it before saving.")
        data = _read(*signature())
        known = _palette(data)
        automatic = []
        for item in checked:
            if item["color"] is None:
                item["color"] = known.get(item["name"].casefold())
                if item["color"] is None:
                    automatic.append(item)
        avoid = [*known.values(), *(item["color"] for item in checked if item["color"])]
        for item, colour in zip(automatic, _new_colours(len(automatic), avoid) if automatic else []):
            item["color"] = colour
        assigned = dict(data["assigned"])
        # Forget older spellings so a case-only edit cannot leave two colours.
        for item in checked:
            assigned = {name:c for name,c in assigned.items() if canonical(name).casefold() != item["name"].casefold()}
            assigned[item["name"]] = item["color"]
        _write(checked, assigned)
        return read()


def update(changes):
    """Explicit colour edits are shared by all projects, including aliases."""
    checked = _validate([{"name":name, "color":colour} for name,colour in changes.items()])
    with _locked():
        data = _read(*signature())
        entries = [dict(item) for item in data["entries"]]
        by_key = {item["name"].casefold():item for item in entries}
        assigned = dict(data["assigned"])
        for item in checked:
            key = item["name"].casefold()
            if key in by_key:
                by_key[key]["color"] = item["color"]
            elif len(entries) < 200:
                entries.append(item)
            assigned = {name:c for name,c in assigned.items() if canonical(name).casefold() != key}
            assigned[item["name"]] = item["color"]
        _write(entries, assigned)
    return colours()


def colours():
    data = _read(*signature())
    known = _palette(data)
    names = {*BASE_COLOURS, *ALIASES, *data["assigned"], *(item["name"] for item in data["entries"])}
    return {name:known[canonical(name).casefold()] for name in names}
