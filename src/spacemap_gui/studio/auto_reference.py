"""Opt-in reference training, debounced across Studio windows and saved per table.

Studio's existing job poll advances the queue. Workers keep running when the
viewer closes; a pending update resumes when that table is opened again.
"""

import hashlib
import json
from pathlib import Path
import time

from spacemap.viewer import reference
from spacemap.viewer.sources import fingerprint
from spacemap_workflows.leiden_job import write_json
from . import jobs, manual_reference, tasks
from .settings import project_lock

QUIET_SECONDS = 15
MIN_INTERVAL = 60


def _path(root):
    return Path(root) / "outputs" / "studio" / "auto_reference.json"


def _read(root):
    path = _path(root)
    return json.loads(path.read_text()) if path.exists() else {}


def _context(channel):
    from spacemap_gui.viewer import registry
    project = registry.session_project(channel) or {}
    table, _, _ = manual_reference.context(channel)
    key = hashlib.sha256(str(Path(table).resolve()).encode()).hexdigest()
    return project, key


def _view(item):
    settings = item.get("settings", {})
    return {"enabled":bool(item.get("enabled")), "column":settings.get("training_column", ""),
            "table":str(settings.get("table", "")), "checked_only":bool(settings.get("checked_only")),
            "state":item.get("state", "off"), "message":item.get("message", ""),
            "job_id":item.get("job_id"), "completed":item.get("completed")}


def configure(channel, column, enabled, checked_only=False):
    if not isinstance(enabled, bool) or not isinstance(checked_only, bool):
        raise ValueError("Choose whether Auto-train and Only checked are on or off.")
    project, key = _context(channel)
    if project.get("read_only"):
        raise ValueError("This project is read-only.")
    settings = manual_reference.training_settings(channel, column, checked_only) if enabled else None
    with project_lock(project["root"]):
        saved = _read(project["root"])
        item = saved.setdefault(key, {})
        if enabled:
            settings["table"] = str(settings["table"])
            # Changing the source does not change the manual-label destination.
            old_settings = {k:v for k,v in item.get("settings", {}).items() if k != "revision"}
            if old_settings != {k:v for k,v in settings.items() if k != "revision"}:
                item.pop("completed", None)
                item.pop("job_id", None)
            if item.get("settings") != settings or not item.get("enabled"):
                item.update(stamp=None, attempted=None, dirty_since=time.time())
            item.update(settings=settings, enabled=True, state="waiting", message="Waiting for a short pause in labelling…")
        else:
            item.update(enabled=False, state="off", message="Auto-train is off. Saved models remain available.")
        _path(project["root"]).parent.mkdir(parents=True, exist_ok=True)
        write_json(_path(project["root"]), saved)
        return _view(item)


def status(channel):
    project, key = _context(channel)
    return _view(_read(project["root"]).get(key, {}))


def _stamp(settings):
    reviews = reference.path_for(settings["table"])
    return tasks.source_revision(settings) + ":" + (fingerprint(reviews) if reviews.exists() else "")


def tick(channel, now=None):
    """At most one launch per stable revision, under the shared project lock."""
    project, key = _context(channel)
    root = project["root"]
    if not _path(root).exists():
        return {"enabled":False, "state":"off"}
    now = time.time() if now is None else now
    with project_lock(root, tolerate_read_only=True):
        saved = _read(root)
        item = saved.get(key, {})
        if not item.get("enabled") or project.get("read_only"):
            return _view(item)
        before = json.dumps(item, sort_keys=True)
        settings = dict(item["settings"])
        stamp = _stamp(settings)
        if stamp != item.get("stamp"):
            item.update(stamp=stamp, dirty_since=now)
        recent = jobs.recent(root)
        active = next((job for job in recent if job["status"] in jobs.ACTIVE), None)
        own = next((job for job in recent if job["id"] == item.get("job_id")), None)
        if own and own["status"] == "done":
            item["completed"] = {"id":own["id"], "result":own.get("result", {})}
        if active:
            item.update(state="training" if active["id"] == item.get("job_id") else "waiting",
                        message="Updating suggestions…" if active["id"] == item.get("job_id") else "Waiting for the current job…")
        elif stamp == item.get("attempted"):
            if own and own["status"] == "done":
                waiting = (own.get("result") or {}).get("waiting")
                item.update(state="waiting" if waiting else "ready", message=waiting or "Suggestions are up to date.")
            elif own and own["status"] in {"failed", "cancelled", "stopped"}:
                item.update(state="error", message=own.get("message") or "Training stopped. Edit labels or switch Auto-train off and on to retry.")
        elif now - item.get("dirty_since", now) < QUIET_SECONDS or now - item.get("started", 0) < MIN_INTERVAL:
            item.update(state="waiting", message="Labels saved. Suggestions will update after a short pause…")
        else:
            # No full cell-table scan in the polling path. The isolated worker
            # checks example counts, flags and missing values before fitting.
            settings.update(revision=tasks.source_revision(settings), auto_train=True)
            try:
                job = jobs.start(root, "train_reference_model", settings)
                item.update(job_id=job["id"], attempted=stamp, started=now, state="training", message="Updating suggestions…")
            except ValueError as exc:
                item.update(state="waiting", message=str(exc))
        if json.dumps(item, sort_keys=True) != before:
            write_json(_path(root), saved)
        return _view(item)
