"""Projects shared by a lab group: files SPACEMAP writes stay readable and writable by the project's group.

A project folder is shared through its Unix group (for example ``Aguilera_lab``). Three things used to break that:

* the default umask (022) made new files writable by their creator only, so a collaborator's SPACEMAP could not open
  the lock and log files the first user had made ("Permission denied: ....json.lock");
* files written atomically (a private temporary file moved into place) ended up readable by their creator only;
* nothing put the group back on files that already had the wrong permissions.

So SPACEMAP runs with umask 007 (owner and group read and write, others nothing), locks fall back to a read-only
handle when they cannot be written (``flock`` works on it), atomically written files get group access, and opening a
project repairs the files the current user owns in it. Only a file's owner can change its permissions, so each
collaborator's SPACEMAP repairs their own files; ``spacemap share-project <folder>`` does it in one go.
Set ``SPACEMAP_KEEP_UMASK=1`` to keep your own umask.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import threading
import time

GROUP_UMASK = 0o007
_REPAIRED: set[str] = set()
_REPAIR_GUARD = threading.Lock()


def apply_group_umask() -> None:
    """New files and folders: owner and group read and write, others nothing."""
    if os.environ.get("SPACEMAP_KEEP_UMASK"):
        return
    os.umask(GROUP_UMASK)


def share(path, *, group=None) -> bool:
    """Give the project's group read and write on *path* if the current user owns it (folders also pass their group
    on to new files). The group is the parent folder's when *group* is not given. Returns whether anything changed."""
    path = Path(path)
    try:
        info = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
        return False
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISDIR(info.st_mode):
        wanted = mode | 0o070 | stat.S_ISGID
    else:
        wanted = mode | 0o060 | (0o010 if mode & 0o100 else 0)
    changed = False
    try:
        if wanted != mode:
            os.chmod(path, wanted)
            changed = True
        if group is None:
            group = os.stat(path.parent).st_gid
        if group != info.st_gid and group in os.getgroups():
            os.chown(path, -1, group)
            changed = True
    except OSError:
        return changed
    return changed


def open_lock(path):
    """Open a lock file for ``fcntl.flock``: written when possible, read-only when another member made it without
    group write (a shared lock works on a read-only handle too)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = open(path, "a")
    except PermissionError:
        return open(path, "r")
    share(path)
    return handle


def repair(root, *, budget_seconds: float = 120.0) -> dict:
    """Give the group access to every file and folder the current user owns under *root* (not following links).
    Stops after *budget_seconds*; returns counts."""
    root = Path(root)
    started = time.monotonic()
    try:
        group = os.stat(root).st_gid
    except OSError:
        return {"checked": 0, "changed": 0, "complete": False}
    checked = changed = 0
    stack = [root]
    while stack:
        if time.monotonic() - started > budget_seconds:
            return {"checked": checked, "changed": changed, "complete": False}
        folder = stack.pop()
        if share(folder, group=group):
            changed += 1
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            checked += 1
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and share(entry.path, group=group):
                    changed += 1
            except OSError:
                continue
    return {"checked": checked, "changed": changed, "complete": True}


def repair_in_background(root) -> None:
    """Repair a project once per process, without holding up whoever opened it."""
    key = str(Path(root).resolve())
    with _REPAIR_GUARD:
        if key in _REPAIRED:
            return
        _REPAIRED.add(key)
    threading.Thread(target=repair, args=(key,), name="spacemap-share-project", daemon=True).start()
