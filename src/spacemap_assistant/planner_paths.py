"""Local path aliases used when a hosted model plans a local SPACEMAP job."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .general_action import GeneralActionError
from .online_access import extract_prompt_paths
from .paths import require_assistant_read_path, require_assistant_write_path


def online_action_content(prompt: str) -> tuple[str, dict[str, str]]:
    """Hide directory names while giving a hosted planner stable path aliases."""
    sanitized = str(prompt).replace("\\_", "_")
    aliases: dict[str, str] = {}
    descriptions: list[str] = []
    for index, raw in enumerate(extract_prompt_paths(prompt), start=1):
        alias = f"LOCAL_PATH_{index}"
        candidate = Path(raw).expanduser().resolve(strict=False)
        if candidate.exists():
            candidate = require_assistant_read_path(candidate, must_exist=True)
            kind = "existing folder" if candidate.is_dir() else f"existing {candidate.suffix or 'data'} file"
        else:
            candidate = require_assistant_write_path(candidate)
            kind = "requested new output path"
        aliases[alias] = str(candidate)
        sanitized = sanitized.replace(raw, alias)
        descriptions.append(f"- {alias}: {kind}")
    if aliases:
        sanitized += (
            "\n\nLOCAL PATH ALIASES (only these descriptions are sent; file contents stay local):\n"
            + "\n".join(descriptions)
            + "\nUse an existing file/folder alias under inputs. Use the requested output alias under output_dir."
        )
    return sanitized, aliases


def restore_online_job_paths(payload: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    """Resolve only aliases created locally; reject model-invented filesystem paths."""
    def restore(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): restore(child) for key, child in value.items()}
        if isinstance(value, list):
            return [restore(child) for child in value]
        if not isinstance(value, str):
            return value
        key = value.strip().strip("[]")
        if key in aliases:
            return aliases[key]
        if value.strip().startswith("/"):
            raise GeneralActionError("The hosted planner invented a local path, so the plan was rejected.")
        return value

    restored = restore(dict(payload))
    inputs = restored.get("inputs")
    if not isinstance(inputs, dict):
        raise GeneralActionError("The hosted plan did not provide named local inputs.")
    for value in inputs.values():
        if str(value) not in aliases.values():
            raise GeneralActionError("Every hosted-plan input must use a local path alias.")
    if str(restored.get("output_dir") or "") not in aliases.values():
        raise GeneralActionError("The hosted plan output must use a local path alias.")
    return restored
