"""Shared lab projects: files SPACEMAP writes stay readable and writable by the project's group."""

import fcntl
import os
import stat

import pytest

from spacemap import sharing


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_share_gives_the_group_access_and_folders_pass_it_on(tmp_path):
    folder = tmp_path / "outputs"
    folder.mkdir(mode=0o700)
    os.chmod(folder, 0o700)
    private = folder / "view.json"
    private.write_text("{}")
    os.chmod(private, 0o600)
    assert sharing.share(folder) and sharing.share(private)
    assert _mode(folder) & 0o070 == 0o070 and _mode(folder) & stat.S_ISGID
    assert _mode(private) & 0o060 == 0o060 and not _mode(private) & 0o007  # others are not given access
    assert not sharing.share(private)  # nothing left to change


def test_a_lock_made_without_group_write_still_locks(tmp_path):
    lock = tmp_path / "view.json.lock"
    lock.write_text("")
    os.chmod(lock, 0o444)  # what a collaborator's old lock looks like to us: not writable
    if os.access(lock, os.W_OK):
        pytest.skip("running as a user who can write anything")
    with sharing.open_lock(lock) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)  # a read-only handle is enough to lock
        fcntl.flock(handle, fcntl.LOCK_UN)
    fresh = tmp_path / "deeper" / "new.lock"
    with sharing.open_lock(fresh) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
    assert _mode(fresh) & 0o060 == 0o060


def test_repair_walks_the_project_and_the_umask_keeps_others_out(tmp_path):
    project = tmp_path / "project"
    (project / "outputs" / "viewer").mkdir(parents=True)
    for path in (project / "outputs", project / "outputs" / "viewer"):
        os.chmod(path, 0o700)
    for name in ("a.json", "outputs/viewer/b.json.lock"):
        (project / name).write_text("")
        os.chmod(project / name, 0o600)
    os.symlink(project / "a.json", project / "link.json")
    done = sharing.repair(project)
    assert done["complete"] and done["changed"] >= 4
    assert all(_mode(project / name) & 0o060 == 0o060 for name in ("a.json", "outputs/viewer/b.json.lock"))
    assert _mode(project / "outputs" / "viewer") & stat.S_ISGID
    previous = os.umask(0o022)
    try:
        sharing.apply_group_umask()
        assert os.umask(0o022) == 0o007
    finally:
        os.umask(previous)
