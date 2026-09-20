"""Read-only monitoring helpers shared by every background action type."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .paths import ASSISTANT_DATA_DIR, is_assistant_read_allowed, is_within


ACTIONS_DIR = ASSISTANT_DATA_DIR / "actions"


def _valid_action_directory(action_id: str) -> Path | None:
    try:
        parsed = uuid.UUID(str(action_id))
    except (ValueError, TypeError, AttributeError):
        return None
    if str(parsed) != str(action_id):
        return None
    path = (ACTIONS_DIR / str(parsed)).resolve()
    return path if is_within(path, ACTIONS_DIR) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_all_actions(limit: int = 50) -> list[dict[str, Any]]:
    if not ACTIONS_DIR.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for plan_path in ACTIONS_DIR.glob("*/plan.json"):
        directory = _valid_action_directory(plan_path.parent.name)
        if directory is None:
            continue
        plan = _read_json(plan_path)
        status = _read_json(directory / "status.json")
        created = str(plan.get("created_at") or status.get("created_at") or "")
        rows.append(
            {
                "action_id": plan_path.parent.name,
                "action_type": str(plan.get("action_type") or "unknown"),
                "created_at": created,
                "state": str(status.get("state") or "unknown"),
                "stage": str(status.get("stage") or ""),
                "message": str(status.get("message") or ""),
                "elapsed_seconds": status.get("elapsed_seconds"),
            }
        )
    return sorted(rows, key=lambda item: item["created_at"], reverse=True)[: max(1, int(limit))]


def action_details(action_id: str) -> dict[str, Any]:
    directory = _valid_action_directory(action_id)
    if directory is None:
        return {}
    return {
        "plan": _read_json(directory / "plan.json"),
        "status": _read_json(directory / "status.json"),
        "report": _read_json(directory / "report.json"),
        "directory": str(directory),
    }


def action_log(action_id: str, max_characters: int = 80_000) -> str:
    directory = _valid_action_directory(action_id)
    if directory is None:
        return "Invalid action number."
    try:
        text = (directory / "worker.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if text:
        return text[-max(2_000, int(max_characters)) :]
    status = _read_json(directory / "status.json")
    if status:
        return f"{status.get('stage', 'Waiting')}: {status.get('message', '')}".strip()
    return "The worker has not written a message yet."


def progress_events(action_id: str, limit: int = 200) -> list[dict[str, Any]]:
    directory = _valid_action_directory(action_id)
    if directory is None:
        return []
    try:
        lines = (directory / "progress.jsonl").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines[-max(1, int(limit)) :]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _planned_output_paths(value: Any, *, key: str = "") -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            paths.extend(_planned_output_paths(child, key=str(child_key).lower()))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_planned_output_paths(child, key=key))
    elif isinstance(value, str) and value.startswith("/"):
        output_words = ("output", "final", "partial", "staging", "intermediate", "backup", "qc")
        if any(word in key for word in output_words):
            path = Path(value).resolve(strict=False)
            if is_assistant_read_allowed(path):
                paths.append(path)
    return paths


def action_file_progress(action_id: str) -> list[dict[str, Any]]:
    details = action_details(action_id)
    plan = details.get("plan") or {}
    unique: dict[str, Path] = {}
    for path in _planned_output_paths(plan):
        if not plan.get("overwrite") and f".before-{action_id[:8]}" in path.name:
            continue
        unique[str(path)] = path
    staging_dir = plan.get("staging_dir")
    if staging_dir:
        candidate = Path(str(staging_dir)).resolve(strict=False)
        if is_assistant_read_allowed(candidate) and candidate.is_dir():
            for path in candidate.glob("*"):
                if path.is_file():
                    unique[str(path.resolve())] = path.resolve()
    rows: list[dict[str, Any]] = []
    for path in unique.values():
        try:
            exists = path.exists()
            is_file = exists and path.is_file()
            is_directory = exists and path.is_dir()
            size = path.stat().st_size if is_file else None
        except OSError:
            exists, is_file, is_directory, size = False, False, False, None
        if "/staging/" in str(path) or path.name == "staging":
            location = "Temporary"
        elif ".before-" in path.name:
            location = "Backup"
        elif "qc" in path.name.lower():
            location = "QC"
        else:
            location = "Final"
        rows.append(
            {
                "file": path.name,
                "location": location,
                "state": "Folder" if is_directory else ("Writing/ready" if exists else "Waiting"),
                "size": human_size(size) if size is not None else "—",
                "path": str(path),
            }
        )
    return sorted(rows, key=lambda item: (item["state"] == "Waiting", item["file"]))


def human_size(value: int | None) -> str:
    if value is None:
        return "—"
    size = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
