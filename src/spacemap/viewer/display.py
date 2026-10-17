"""How each marker of an image is shown, saved with the project.

The viewer stores, per image, every marker's colour, display range and
visibility, plus an optional new name. Other steps read it back: the minimum a
marker was set to is a ready-made background threshold, and renamed markers
can be carried into ROI extraction.

    {"channels": [{"index": 0, "name": "DAPI", "label": "DAPI", "color": "#4f7dff",
                   "low": 120.0, "high": 1800.0, "visible": true}, ...],
     "saved_at": "2026-09-24T21:40:00+00:00"}
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
from pathlib import Path

_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def settings_path(root, image) -> Path:
    """Keep existing files; new files distinguish equal names in different folders."""
    from .view_state import path_for

    folder = Path(root) / 'outputs' / 'viewer' / 'display'
    scoped = folder / path_for(root, image).name
    legacy = folder / f'{Path(image).name}.json'
    if scoped.is_file() or not legacy.is_file():
        return scoped
    try:
        owner = json.loads(legacy.read_text()).get('image')
    except (OSError, ValueError, AttributeError):
        return legacy  # preserve/report an unreadable old file rather than hiding it
    return legacy if not owner or Path(owner).resolve() == Path(image).resolve() else scoped


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean(channels) -> list[dict]:
    """Keep only well-formed channel entries."""
    out = []
    for item in channels or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(item.get("name") or "")
        label = str(item.get("label") or name).strip() or name
        color = str(item.get("color") or "")
        out.append(
            {
                "index": index,
                "name": name,
                "label": label,
                "color": color.lower() if _COLOR.match(color) else "#ffffff",
                "low": _number(item.get("low")),
                "high": _number(item.get("high")),
                "visible": bool(item.get("visible")),
                "custom": bool(item.get("custom")),  # a colour chosen by the user; else the image file's colour is used
            }
        )
    return out


def load(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"channels": []}
    try:
        data = json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return {"channels": []}
    return {
        "channels": clean(data.get("channels")),
        "saved_at": data.get("saved_at"),
        "auto": clean_auto(data.get("auto")),
    }


def clean_auto(options) -> dict:
    """Only bounded method/scope preferences belong in the display settings."""
    from spacemap.thresholding_engine import THRESHOLD_METHODS, parameters

    options = options if isinstance(options, dict) else {}
    method = options.get("method", "Triangle")
    extra = {}
    if isinstance(options.get("display"), dict):
        settings = options["display"]
        upper = float(settings.get("upper_percentile", 99.8))
        extra["display"] = {"adjust": settings.get("adjust") if settings.get("adjust") in ("both", "lower", "upper") else "both",
                            "upper_percentile": upper if 0 < upper <= 100 else 99.8}
    if isinstance(options.get("histogram"), dict):
        settings = options["histogram"]
        cofactor = float(settings.get("cofactor", 5))
        extra["histogram"] = {"scale": settings.get("scale") if settings.get("scale") in ("Linear", "Log10", "Arcsinh") else "Linear",
                              "y": settings.get("y") if settings.get("y") in ("log", "count", "fraction") else "log",
                              "cofactor": cofactor if 0 < cofactor <= 1_000_000 else 5}
    return {
        "method": method if method in THRESHOLD_METHODS else "Triangle",
        "scope": str(options.get("scope") or "")[:2048],
        **({"parameters": parameters(options.get("parameters"))} if "parameters" in options else {}),
        **({"native": bool(options["native"])} if "native" in options else {}),
        **extra,
    }


def save(path: str | Path, channels, auto=None, stamp=None, image=None) -> dict:
    from . import view_state

    path = Path(path)
    with view_state.locked(path):
        data = {
            "channels": clean(channels),
            "auto": clean_auto(auto if auto is not None else load(path).get("auto")),
        }
        owner = image or (view_state.load(path) or {}).get('image')
        if owner:
            data['image'] = str(Path(owner).resolve())
        return view_state.write(path, data, stamp)


def minimums(path: str | Path) -> dict[str, float]:
    """Marker name -> the lowest value shown, for markers whose minimum was set."""
    return {c["name"]: c["low"] for c in load(path)["channels"] if c["name"] and c["low"] is not None}


def renamed(path: str | Path) -> dict[str, str]:
    """Original marker name -> new name, for markers given a new name in the viewer."""
    return {c["name"]: c["label"] for c in load(path)["channels"] if c["name"] and c["label"] != c["name"]}


# ---------------------------------------------------------------------------
# Marker sets: named groups of markers with their colours, ranges and names
# (for example "T cells" or "B cells"), kept per project and applied by marker
# name, so a set works on every image that has those markers.
# ---------------------------------------------------------------------------


def sets_path(display_path: str | Path) -> Path:
    """The project's marker sets live beside its per-image display settings."""
    return Path(display_path).parent / "marker_sets.json"


def load_sets(path: str | Path) -> dict[str, list[dict]]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return {}
    sets = data.get("sets") if isinstance(data, dict) else None
    if not isinstance(sets, dict):
        return {}
    return {str(name): _clean_set(markers) for name, markers in sets.items() if str(name).strip()}


def _clean_set(markers) -> list[dict]:
    out = []
    for item in markers or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        color = str(item.get("color") or "")
        out.append(
            {
                "name": str(item["name"]),
                "label": str(item.get("label") or item["name"]),
                "color": color.lower() if _COLOR.match(color) else "#ffffff",
                "low": _number(item.get("low")),
                "high": _number(item.get("high")),
            }
        )
    return out


_SETS_LOCK = threading.Lock()  # two saves at once (Save pressed twice) must not share a file or lose a set


def _write_sets(path: Path, sets: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(json.dumps({"sets": sets}, indent=1))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_set(path: str | Path, name: str, markers) -> dict[str, list[dict]]:
    """Add or replace the set called *name*; returns every set."""
    name = str(name or "").strip()
    if not name:
        raise ValueError("Give the set a name.")
    cleaned = _clean_set(markers)
    if not cleaned:
        raise ValueError("Tick at least one marker to save as a set.")
    with _SETS_LOCK:
        sets = load_sets(path)
        sets[name] = cleaned
        _write_sets(Path(path), sets)
    return sets


def delete_set(path: str | Path, name: str) -> dict[str, list[dict]]:
    with _SETS_LOCK:
        sets = load_sets(path)
        sets.pop(str(name), None)
        _write_sets(Path(path), sets)
    return sets
