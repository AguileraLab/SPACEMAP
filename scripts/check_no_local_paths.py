#!/usr/bin/env python3
"""Fail when a committed file contains an absolute path from someone's machine.

An absolute path that works on one workstation is a silent bug for everyone
else: the GUI offers a directory nobody has, a notebook fails on its first
cell, and a cluster mount point tells the world where the data lives.

Everything the application writes should come from ``spacemap.userpaths``, and
everything a notebook reads should be built from ``DATA_DIR``. This script
enforces that in CI.

    python scripts/check_no_local_paths.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Service accounts created by this project. They are fixed, published names
#: rather than a person's login, so a path under them is portable.
SERVICE_ACCOUNTS = ("spacemap",)

_NOT_SERVICE = "(?!(?:" + "|".join(SERVICE_ACCOUNTS) + ")/)"

#: Paths that point at one person's machine or one site's storage.
FORBIDDEN = [
    (re.compile(r"/home[0-9]*/" + _NOT_SERVICE + r"[a-z][a-z0-9_.-]*/", re.IGNORECASE), "a home directory"),
    (re.compile(r"/Users/(?!<)[a-z][a-z0-9_.-]*/", re.IGNORECASE), "a macOS home directory"),
    (re.compile(r"[A-Z]:\\\\?Users\\\\?", re.IGNORECASE), "a Windows home directory"),
    (re.compile(r"/(?:[A-Za-z0-9_.-]+/)?(?:work|scratch)/(?!<)[A-Za-z0-9_.-]+/"), "a site-specific work directory"),
    (re.compile(r"\bs\d{6}\b"), "a personal account identifier"),
    (re.compile(r"OneDrive", re.IGNORECASE), "a personal cloud folder"),
    (re.compile(r"\bNucleus[A-Za-z]*[0-9]+\b", re.IGNORECASE), "an internal compute host"),
    (re.compile(r"/cm/shared/apps/"), "a site-specific application directory"),
]

#: Directories that are never scanned.
SKIP_DIRECTORIES = {
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "node_modules", "build", "dist", ".venv", "venv", "site-packages",
}

#: Files that legitimately mention these patterns: this checker itself, and the
#: documentation that explains the rule.
#: Files that legitimately quote these patterns because they document the rule
#: itself, or record history such as a module that used to carry a site name.
ALLOWLIST = {
    "scripts/check_no_local_paths.py",
    "CONTRIBUTING.md",
    "docs/LEDGER.md",
    "docs/handover.md",
    "docs/publishing-changes.md",
    "docs/working-with-this-repository.md",
}

# Personal account identifiers are forbidden in every publishable file,
# including documentation exceptions, binary assets and compiled GUI files.
ACCOUNT_IDENTIFIER = re.compile(rb"s[0-9]{6}", re.IGNORECASE)

#: Other path patterns are checked in these text formats.
EXTENSIONS = {
    ".py", ".ipynb", ".md", ".txt", ".toml", ".yml", ".yaml", ".json",
    ".sh", ".cfg", ".ini", ".in", ".dockerfile", ".js", ".mjs", ".ts", ".tsx", ".html", ".css", ".svg", ".cff",
}

EXTRA_FILES = {"Dockerfile", "docker-compose.yml", ".gitignore", ".gitmessage"}


def _is_scannable(path: Path) -> bool:
    # Compiled frontend assets are tracked release files and need the same
    # checks as their sources. Only fallback directory discovery skips builds.
    if any(part in (SKIP_DIRECTORIES - {"build", "dist"}) for part in path.parts):
        return False
    if path.as_posix() in ALLOWLIST:
        return False
    return path.suffix.lower() in EXTENSIONS or path.name in EXTRA_FILES


def _tracked_files() -> list[Path] | None:
    """Files Git actually tracks, or None when this is not a repository.

    Scanning the working tree instead would flag the user's own data sitting
    beside the code -- a project folder full of absolute paths is entirely
    legitimate there, and only becomes a problem if it is committed.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    root = Path(".").resolve()
    problems: list[str] = []

    candidates = _tracked_files()
    if candidates is None:
        candidates = [item.relative_to(root) for item in root.rglob("*") if item.is_file()]

    for relative in sorted(candidates):
        path = root / relative
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        account = ACCOUNT_IDENTIFIER.search(data)
        if account:
            problems.append(f"{relative}: personal account identifier -- {account.group().decode()!r}")
            continue
        if not _is_scannable(relative):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern, description in FORBIDDEN:
                match = pattern.search(line)
                if match:
                    problems.append(
                        f"{relative}:{number}: {description} -- {match.group(0)!r}\n"
                        f"    {line.strip()[:140]}"
                    )
                    break

    if problems:
        print("Machine-specific paths found in committed files:\n")
        for problem in problems:
            print(f"  {problem}")
        print(
            "\nBuild application paths from spacemap.userpaths, and notebook "
            "paths from DATA_DIR.\nSee CONTRIBUTING.md > House rules."
        )
        return 1

    print("No machine-specific paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
