#!/usr/bin/env python3
"""Check release archives for the assets a clone-free installation needs."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check_wheel(path: Path) -> None:
    widget = "spacemap_gui/tabs/tab7_phenotyping/heatmap_click_selector/frontend/build/"
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for required in ("spacemap_gui/app.py", "spacemap/model_catalog.json",
                         "spacemap/marker_configs.json", widget + "index.html",
                         widget + "THIRD_PARTY_NOTICES.txt"):
            if required not in names:
                raise ValueError(f"{path.name}: missing {required}")
        manifest = json.loads(archive.read("spacemap/marker_configs.json"))
        if not manifest.get("profiles", {}).get("default_phenotyping", {}).get("corr_markers"):
            raise ValueError("Packaged marker defaults are empty")
        html = archive.read(widget + "index.html").decode()
        for asset in re.findall(r'(?:src|href)="([^"]+)"', html):
            if asset.startswith(("http:", "https:", "/")) or widget + asset.removeprefix("./") not in names:
                raise ValueError(f"Missing or external widget asset: {asset}")
        if not any(name.startswith(widget + "assets/plotly.min-") for name in names):
            raise ValueError("The heatmap's local Plotly library is missing")
        entries = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        if "spacemap = spacemap_gui.launch:main" not in archive.read(entries).decode():
            raise ValueError("The spacemap command is missing")
        licenses = [name for name in names if ".dist-info/" in name and name.endswith("/LICENSE")]
        if not licenses or b"academic research use only" not in archive.read(licenses[0]):
            raise ValueError("The UT Southwestern license is missing")
        if any("node_modules" in PurePosixPath(name).parts for name in names):
            raise ValueError("Build-time node_modules must not be packaged")
    print(f"OK: {path.name}: GUI, marker defaults, local widget assets, notices and license")


def check_sdist(path: Path) -> None:
    with tarfile.open(path) as archive:
        names = {str(PurePosixPath(name).relative_to(PurePosixPath(name).parts[0]))
                 for name in archive.getnames()}
    for required in ("LICENSE", "CITATION.cff", "environment.yml", "Dockerfile", "docker-compose.yml",
                     "requirements-lock.txt", "marker_configs.json", "docs/first-project.md",
                     "docs/installation.md", "scripts/check_release_artifacts.py",
                     "scripts/check_runtime.py", "scripts/check_container.py", "scripts/check_installed_app.py"):
        if required not in names:
            raise ValueError(f"{path.name}: missing {required}")
    print(f"OK: {path.name}: installation and source-release files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels, sources = list(args.directory.glob("*.whl")), list(args.directory.glob("*.tar.gz"))
    if not wheels or not sources:
        parser.error("Build a wheel and source archive first: python -m build")
    for path in wheels:
        check_wheel(path)
    for path in sources:
        check_sdist(path)


if __name__ == "__main__":
    main()
