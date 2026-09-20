"""Make ``<project>.json`` a complete, drop-in portable record of a project.

Historically a project's settings were split across four places: the project
config, the per-project section of ``interface_state.json`` (tabs 8 and 9), the
``outputs/phenotyping/cache/*_params.json`` files (tab 7), and the
directory-scoped ``interface_settings.json``. Handing a collaborator the project
JSON therefore handed them roughly 86% of the project.

This module folds the strays into the config on save and writes them back out on
load, and rewrites paths that live inside the project as ``${PROJECT_ROOT}/...``
so the same file works from a different checkout on a different machine.

Safety rules that matter:

* Restoring never deletes a local file that the bundle does not mention.
* Restoring never overwrites a local file that is *newer* than the config it
  came from, so reloading a project cannot silently revert edits you made after
  the last save.
* Absolute paths outside the project (raw image archives, shared drives) are
  left untouched and reported, because only a human can remap those.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "BUNDLE_KEY",
    "BUNDLE_VERSION",
    "PROJECT_ROOT_TOKEN",
    "portablize",
    "localize",
    "external_paths",
    "collect_bundle",
    "apply_bundle",
    "PHENOTYPING_CACHE_RELPATH",
]

#: Stand-in for the project root inside a saved config.
PROJECT_ROOT_TOKEN = "${PROJECT_ROOT}"

#: Where the folded-in state lives inside the project config.
BUNDLE_KEY = "portable_state"
BUNDLE_VERSION = 1

PHENOTYPING_CACHE_RELPATH = "outputs/phenotyping/cache"

#: Params files are small settings blobs; anything larger is a result snapshot
#: that has no business travelling inside a config file.
MAX_PARAMS_FILE_BYTES = 2 * 1024 * 1024


def _real(path: Any) -> str:
    try:
        return os.path.realpath(str(path))
    except Exception:
        return str(path)


def portablize(value: Any, project_root: str | Path) -> Any:
    """Rewrite every path inside `project_root` as ``${PROJECT_ROOT}/...``.

    Walks nested dicts and lists. Paths outside the project are returned
    unchanged: they are real external references, not relocatable ones.
    """
    root = _real(project_root)

    def convert(item: Any) -> Any:
        if isinstance(item, str):
            if not item.startswith(("/", "\\")):
                return item
            resolved = _real(item)
            if resolved == root:
                return PROJECT_ROOT_TOKEN
            if resolved.startswith(root + os.sep):
                return PROJECT_ROOT_TOKEN + "/" + os.path.relpath(resolved, root)
            return item
        if isinstance(item, dict):
            return {
                (convert(key) if isinstance(key, str) else key): convert(sub)
                for key, sub in item.items()
            }
        if isinstance(item, list):
            return [convert(sub) for sub in item]
        return item

    return convert(value)


def localize(value: Any, project_root: str | Path) -> Any:
    """Expand ``${PROJECT_ROOT}`` back to a real path. Inverse of `portablize`."""
    root = str(project_root).rstrip("/\\")

    def convert(item: Any) -> Any:
        if isinstance(item, str):
            return item.replace(PROJECT_ROOT_TOKEN, root) if PROJECT_ROOT_TOKEN in item else item
        if isinstance(item, dict):
            return {
                (convert(key) if isinstance(key, str) else key): convert(sub)
                for key, sub in item.items()
            }
        if isinstance(item, list):
            return [convert(sub) for sub in item]
        return item

    return convert(value)


def external_paths(value: Any, project_root: str | Path) -> list[str]:
    """Absolute paths pointing outside the project, deduplicated and sorted.

    These are what a collaborator has to remap by hand after importing; the tab 0
    import summary shows them rather than pretending the project is fully portable.
    """
    root = _real(project_root)
    found: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, str):
            if item.startswith("/") and not item.startswith(PROJECT_ROOT_TOKEN):
                resolved = _real(item)
                if resolved != root and not resolved.startswith(root + os.sep):
                    found.add(item)
        elif isinstance(item, dict):
            for key, sub in item.items():
                walk(key)
                walk(sub)
        elif isinstance(item, list):
            for sub in item:
                walk(sub)

    walk(value)
    return sorted(found)


def _read_params_files(project_root: str | Path) -> dict[str, Any]:
    """Every tab-parameter JSON in the phenotyping cache directory."""
    cache_dir = Path(project_root) / PHENOTYPING_CACHE_RELPATH
    params: dict[str, Any] = {}
    if not cache_dir.is_dir():
        return params
    for path in sorted(cache_dir.glob("*.json")):
        try:
            if path.stat().st_size > MAX_PARAMS_FILE_BYTES:
                continue
            params[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A corrupt or unreadable params file must not block saving the project.
            continue
    return params


def collect_bundle(
    project_root: str | Path,
    interface_state: dict[str, Any] | None = None,
    directory_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gather everything that lives outside the project config, path-portable.

    `interface_state` is this project's section of interface_state.json (the
    tab 8 / tab 9 per-CSV configs). `directory_settings` is the subset of
    interface_settings.json whose keys point inside the project.
    """
    bundle = {
        "version": BUNDLE_VERSION,
        "interface_state": interface_state or {},
        "tab_params": _read_params_files(project_root),
        "directory_settings": directory_settings or {},
    }
    return portablize(bundle, project_root)


def apply_bundle(
    project_root: str | Path,
    bundle: dict[str, Any] | None,
    config_mtime: float | None = None,
) -> dict[str, Any]:
    """Write the bundle's tab-params files out and return the interface state.

    Returns the (localized) interface-state section for the caller to install.

    A params file is written when it is missing, or when its content differs and
    the project config is at least as new as the file. That last condition is
    what stops a stale bundle from reverting parameters you changed after the
    project was last saved.
    """
    if not isinstance(bundle, dict):
        return {}

    local = localize(bundle, project_root)
    cache_dir = Path(project_root) / PHENOTYPING_CACHE_RELPATH
    params = local.get("tab_params") or {}

    if params:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return local.get("interface_state") or {}

    for name, payload in params.items():
        # Never let a config reach outside its own cache directory.
        if os.path.basename(name) != name or not name.endswith(".json"):
            continue
        target = cache_dir / name
        try:
            serialized = json.dumps(payload, indent=2)
            if target.exists():
                current = target.read_text(encoding="utf-8")
                if current == serialized:
                    continue
                if config_mtime is not None and target.stat().st_mtime > config_mtime:
                    # Local file is newer than the saved config: keep the local edits.
                    continue
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(serialized, encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            continue

    return local.get("interface_state") or {}
