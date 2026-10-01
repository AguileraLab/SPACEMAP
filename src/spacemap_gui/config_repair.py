"""Shrink a project file that data has leaked into, without loading it.

A project file records settings and should be well under 1 MB. When a
session value holding per-cell tables slipped past the size checks, the file
grew to tens of GB. Loading such a file needs more memory than most nodes
have, so this module reads it as a stream instead and writes a copy without
the data-sized entries. Every setting is kept.

It relies on the layout the project file is always written in
(``json.dumps(..., indent=2)``): each entry starts on its own line at a fixed
indent, and a multi-line value ends with a bracket at that same indent. Data
inside a dropped entry is skipped with a fast byte search, not parsed, so a
44 GB file takes about as long as reading it once.
"""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path

#: Settings (entries inside a tab) larger than this in the file are data.
SETTING_LIMIT = 1024 * 1024
#: Any other entry larger than this is data. Directory lists and the audit log
#: stay far below it.
SECTION_LIMIT = 16 * 1024 * 1024
#: Sections whose entries are checked one by one rather than as a whole: the
#: tab settings, and the bundle of settings kept outside the file.
_PORTABLE = (("portable_state",), ("portable_state", "interface_state"), ("portable_state", "tab_params"))


def _descend(here: tuple) -> bool:
    return (here[0] == "tab_configurations" and len(here) <= 2) or here in _PORTABLE


_CHUNK = 64 * 1024 * 1024
_KEY = re.compile(rb'^( *)("(?:[^"\\]|\\.)*"): ')


class _Reader:
    """Lines of a file, plus a fast skip to a given closing line."""

    def __init__(self, handle):
        self.handle = handle
        self.buffer = b""
        self.eof = False

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.handle.read(_CHUNK)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def line(self) -> bytes | None:
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                result, self.buffer = self.buffer[:end + 1], self.buffer[end + 1:]
                return result
            if not self._fill():
                result, self.buffer = self.buffer, b""
                return result or None

    def skip_through(self, closer: bytes) -> int:
        """Consume up to and including the line that is exactly *closer* (optionally with a comma)."""
        pattern = re.compile(rb"\n" + re.escape(closer) + rb",?\r?\n")
        skipped = 0
        self.buffer = b"\n" + self.buffer  # so a closer on the very next line matches
        while True:
            found = pattern.search(self.buffer)
            if found:
                skipped += found.end() - 1
                self.buffer = self.buffer[found.end():]
                return skipped
            keep = len(closer) + 4
            skipped += max(0, len(self.buffer) - keep)
            self.buffer = self.buffer[-keep:]
            if not self._fill():
                raise ValueError("the file ends inside an entry")


def _read_dict(reader: _Reader, indent: int, path: tuple, removed: list) -> tuple[list[bytes], bytes]:
    """Entries of a dict whose keys sit at *indent* spaces; returns (entries, closing line)."""
    closing = b" " * (indent - 2) + b"}"
    entries: list[bytes] = []
    while True:
        raw = reader.line()
        if raw is None:
            raise ValueError("the file ends inside a section")
        line = raw.rstrip(b"\r\n")
        if line.rstrip(b",") == closing:
            return entries, line.rstrip(b",")
        match = _KEY.match(line)
        if not match or len(match.group(1)) != indent:
            raise ValueError(f"unexpected layout at indent {indent}")
        key = json.loads(match.group(2))
        here = path + (key,)
        rest = line[match.end():]
        if _descend(here) and rest == b"{":
            inner, close = _read_dict(reader, indent + 2, here, removed)
            body = b",\n".join(inner)
            entries.append(line + b"\n" + (body + b"\n" if body else b"") + close)
            continue
        limit = SETTING_LIMIT if len(path) >= 2 or path == ("portable_state",) else SECTION_LIMIT
        opener = rest.rstrip(b",")
        if opener in (b"{", b"["):
            closer = b" " * indent + (b"}" if opener == b"{" else b"]")
            lines, size = [line], len(raw)
            while True:
                raw = reader.line()
                if raw is None:
                    raise ValueError("the file ends inside an entry")
                inner_line = raw.rstrip(b"\r\n")
                size += len(raw)
                lines.append(inner_line)
                if inner_line.rstrip(b",") == closer:
                    lines[-1] = closer
                    entries.append(b"\n".join(lines))
                    break
                if size > limit:  # data, not a setting: skip the rest unread
                    size += reader.skip_through(closer)
                    removed.append(("/".join(here), size))
                    break
        elif len(raw) > limit:
            removed.append(("/".join(here), len(raw)))
        else:
            entries.append(line.rstrip(b","))


def shrink_project_file(path: str | Path) -> tuple[Path, list[tuple[str, int]]]:
    """Rewrite *path* without its data-sized entries; keep the original beside it.

    Returns (backup path, [(entry, bytes removed)]). The rewritten file is
    checked to be valid JSON before it replaces the original, which is kept
    as ``<name>.oversized-<timestamp>`` for the user to delete. Raises
    ``ValueError`` if the file is not in the expected layout; nothing is
    changed then.
    """
    path = Path(path)
    removed: list[tuple[str, int]] = []
    temporary = path.with_name(f".{path.name}.shrink.tmp")
    try:
        with open(path, "rb") as handle:
            reader = _Reader(handle)
            first = reader.line()
            if first is None or first.strip() != b"{":
                raise ValueError("not a SPACEMAP project file")
            entries, _ = _read_dict(reader, 2, (), removed)
        with open(temporary, "wb") as out:
            out.write(b"{\n" + b",\n".join(entries) + b"\n}\n")
        with open(temporary, "r", encoding="utf-8") as check:
            json.load(check)
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))  # stays as private as the original
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    backup = path.with_name(f"{path.name}.oversized-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.replace(path, backup)
    os.replace(temporary, path)
    return backup, removed
