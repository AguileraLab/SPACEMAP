"""Inventory and safe removal of everything the GUI writes outside your results.

Three kinds of file accumulate while the app runs, and they are emphatically not
equally disposable:

``recomputed``
    Pure derived data (JIT caches, downloaded model weights, normalisation
    caches). Deleting costs time on the next run and nothing else.

``results``
    Large parquet snapshots backing the "Results & Phenotype Refinement" panels.
    Every tab guards its read with an existence check, so deleting these makes
    the panel disappear until the step is re-run -- it does not raise.

``settings``
    The ``*_params.json`` files holding each tab's chosen parameters. These are
    *not* caches. Deleting them resets tabs to defaults, so they are inventoried
    separately and never included in a "clear caches" sweep.

Nothing here ever touches the project config, the project index, or any analysis
output.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CacheEntry",
    "CATEGORY_LABELS",
    "scan_caches",
    "clear_entry",
    "format_size",
    "user_cache_roots",
]

CATEGORY_LABELS = {
    "recomputed": "Rebuilt automatically",
    "results": "Re-run the step to restore",
    "settings": "Tab parameters (not a cache)",
}


@dataclass
class CacheEntry:
    key: str
    label: str
    category: str
    description: str
    paths: list[Path] = field(default_factory=list)
    #: When set, only these files inside `paths` are removed, and the directory stays.
    patterns: tuple[str, ...] = ()

    @property
    def files(self) -> list[Path]:
        found: list[Path] = []
        for path in self.paths:
            if not path.exists():
                continue
            if self.patterns:
                for pattern in self.patterns:
                    found.extend(p for p in path.glob(pattern) if p.is_file())
            elif path.is_dir():
                found.extend(p for p in path.rglob("*") if p.is_file())
            else:
                found.append(path)
        return found

    @property
    def size_bytes(self) -> int:
        total = 0
        for file in self.files:
            try:
                total += file.stat().st_size
            except OSError:
                continue
        return total

    @property
    def file_count(self) -> int:
        return len(self.files)


def format_size(size_bytes: int) -> str:
    for unit, limit in (("B", 1024), ("KB", 1024 ** 2), ("MB", 1024 ** 3)):
        if size_bytes < limit:
            scaled = size_bytes if unit == "B" else size_bytes / (limit / 1024)
            return f"{scaled:.0f} {unit}" if unit == "B" else f"{scaled:.2f} {unit}"
    return f"{size_bytes / (1024 ** 3):.2f} GB"


def user_cache_roots(env: dict | None = None) -> dict[str, Path]:
    """Machine-level cache directories, honouring the same env vars the app reads."""
    import os

    env = env if env is not None else os.environ
    base = Path(env.get("SPACEMAP_CACHE_DIR", Path.home() / ".cache" / "spacemap"))
    return {
        "numba": Path(env.get("NUMBA_CACHE_DIR", base / "numba_cache")),
        "cellpose": Path(env.get("CELLPOSE_LOCAL_MODELS_PATH", base / "cellpose_cache")),
    }


def scan_caches(project_root: str | Path | None, env: dict | None = None) -> list[CacheEntry]:
    """Inventory every removable artefact for one project plus the machine caches.

    Only the given project is inspected: a sweep must never reach into a project
    the user does not currently have open.
    """
    roots = user_cache_roots(env)
    entries: list[CacheEntry] = [
        CacheEntry(
            key="numba",
            label="Numba JIT cache",
            category="recomputed",
            description="Compiled kernels. Rebuilt on first use; the next run is slower once.",
            paths=[roots["numba"]],
        ),
        CacheEntry(
            key="cellpose",
            label="Cellpose model weights",
            category="recomputed",
            description="Downloaded again automatically. Needs network access to restore.",
            paths=[roots["cellpose"]],
        ),
    ]

    if project_root:
        phenotyping = Path(project_root) / "outputs" / "phenotyping"
        entries.extend([
            CacheEntry(
                key="norm_cache",
                label="Normalisation cache",
                category="recomputed",
                description="Pre-computed normalised markers. Recomputed on demand.",
                paths=[phenotyping / "leiden_cache"],
            ),
            CacheEntry(
                key="temp_results",
                label="Intermediate result tables",
                category="results",
                description=(
                    "Parquet snapshots behind the interactive results panels. Deleting hides "
                    "those panels until you re-run the step; assignments already saved to the "
                    "unified CSV are unaffected."
                ),
                paths=[phenotyping / "temp"],
                patterns=("*.parquet",),
            ),
            CacheEntry(
                key="cached_results",
                label="Saved run snapshots",
                category="results",
                description="Parquet results kept alongside tab settings so a previous run can be reloaded.",
                paths=[phenotyping / "cache"],
                patterns=("*.parquet",),
            ),
            CacheEntry(
                key="tab_params",
                label="Tab parameter files",
                category="settings",
                description=(
                    "Each tab's saved settings. Deleting resets those tabs to defaults - "
                    "it does not delete data, but you will have to re-enter your parameters."
                ),
                paths=[phenotyping / "cache"],
                patterns=("*.json",),
            ),
        ])

    return entries


def clear_entry(entry: CacheEntry) -> tuple[int, int]:
    """Delete an entry's files. Returns (files_removed, bytes_freed).

    Pattern-scoped entries delete matching files and leave the directory in
    place, because tabs recreate the directory but not necessarily the path to it.
    """
    files = entry.files
    freed = entry.size_bytes
    removed = 0

    if entry.patterns:
        for file in files:
            try:
                file.unlink()
                removed += 1
            except OSError:
                continue
        return removed, freed

    for path in entry.paths:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                removed += sum(1 for p in path.rglob("*") if p.is_file())
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed, freed
