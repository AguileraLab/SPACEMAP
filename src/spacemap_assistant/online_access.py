"""Owner-controlled, bounded data sharing with hosted AI providers."""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .paths import (
    AI_ROOT,
    ASSISTANT_DATA_DIR,
    ASSISTANT_PROJECTS_DIR,
    SOURCE_ROOT,
    AssistantPathError,
    is_within,
    require_assistant_read_path,
    require_assistant_write_path,
)


ONLINE_ACCESS_PATH = ASSISTANT_DATA_DIR / "config" / "online_data_access.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".qptiff"}
TABLE_SUFFIXES = {".csv", ".tsv"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | TABLE_SUFFIXES


@dataclass(frozen=True)
class OnlineShareLimits:
    max_files: int = 4
    csv_rows: int = 200
    image_max_size: int = 1024

    def normalized(self) -> "OnlineShareLimits":
        return OnlineShareLimits(
            max_files=min(8, max(1, int(self.max_files))),
            csv_rows=min(1000, max(25, int(self.csv_rows))),
            image_max_size=min(2048, max(256, int(self.image_max_size))),
        )


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _empty_config() -> dict[str, Any]:
    return {"schema_version": 1, "grants": [], "limits": asdict(OnlineShareLimits())}


def _load_config() -> dict[str, Any]:
    try:
        payload = json.loads(ONLINE_ACCESS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_config()
    if not isinstance(payload, dict):
        return _empty_config()
    grants = payload.get("grants") if isinstance(payload.get("grants"), list) else []
    clean_grants = []
    for item in grants:
        if not isinstance(item, dict) or not str(item.get("path") or "").strip():
            continue
        clean_grants.append(
            {
                "id": str(item.get("id") or ""),
                "path": str(_resolved(str(item["path"]))),
                "created_at": str(item.get("created_at") or ""),
            }
        )
    limits = payload.get("limits") if isinstance(payload.get("limits"), dict) else {}
    normalized = OnlineShareLimits(
        max_files=limits.get("max_files", 4),
        csv_rows=limits.get("csv_rows", 200),
        image_max_size=limits.get("image_max_size", 1024),
    ).normalized()
    return {"schema_version": 1, "grants": clean_grants, "limits": asdict(normalized)}


def _write_config(payload: dict[str, Any]) -> None:
    target = require_assistant_write_path(ONLINE_ACCESS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def list_online_access_grants() -> list[dict[str, Any]]:
    return list(_load_config()["grants"])


def load_online_share_limits() -> OnlineShareLimits:
    return OnlineShareLimits(**_load_config()["limits"]).normalized()


def save_online_share_limits(
    max_files: int,
    csv_rows: int,
    image_max_size: int,
) -> OnlineShareLimits:
    limits = OnlineShareLimits(max_files, csv_rows, image_max_size).normalized()
    payload = _load_config()
    payload["limits"] = asdict(limits)
    _write_config(payload)
    return limits


def grant_online_access(path: str | Path) -> dict[str, Any]:
    """Allow bounded hosted-model evidence from an exact owner-approved data root."""
    if not str(path).strip():
        raise AssistantPathError("Enter an exact data folder path.")
    candidate = require_assistant_read_path(path, must_exist=True)
    if not candidate.is_dir():
        raise AssistantPathError("Online data access must name a folder, not one file.")
    inside_application = is_within(candidate, AI_ROOT) or is_within(candidate, SOURCE_ROOT)
    if inside_application and not is_within(candidate, ASSISTANT_PROJECTS_DIR):
        raise AssistantPathError(
            "Only project data can be shared from inside the application folder. "
            "Source, configuration, memory, credentials, models, and runtime files stay private."
        )
    if any(part in {".git", ".codex", ".agents"} for part in candidate.parts):
        raise AssistantPathError("Application and agent-control folders cannot be shared online.")
    if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
        raise AssistantPathError("Choose a data folder, not an application source folder.")
    payload = _load_config()
    canonical = str(candidate)
    for item in payload["grants"]:
        if item["path"] == canonical:
            return item
    record = {
        "id": str(uuid.uuid4()),
        "path": canonical,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    payload["grants"].append(record)
    _write_config(payload)
    return record


def revoke_online_access(grant_id: str) -> dict[str, Any]:
    payload = _load_config()
    removed = next(
        (item for item in payload["grants"] if item["id"] == str(grant_id)), None
    )
    if removed is None:
        raise AssistantPathError("That online data grant no longer exists.")
    payload["grants"] = [
        item for item in payload["grants"] if item["id"] != str(grant_id)
    ]
    _write_config(payload)
    return removed


def is_online_share_allowed(path: str | Path) -> bool:
    candidate = _resolved(path)
    # The Owner already controls the projects directory as the built-in data area.
    # Selecting an online model plus the session consent is the explicit approval
    # to send a bounded preview/sample from this tree. All descendants inherit it.
    if is_within(candidate, ASSISTANT_PROJECTS_DIR):
        return True
    return any(
        is_within(candidate, item["path"]) for item in list_online_access_grants()
    )


def require_online_share_path(path: str | Path, *, must_exist: bool = True) -> Path:
    candidate = require_assistant_read_path(path, must_exist=must_exist)
    if not is_online_share_allowed(candidate):
        raise AssistantPathError(
            "The selected model is online, but this file has not been approved for online review. "
            "Local access and online sharing are separate. Add a parent data folder under "
            "Admin Access > Online data folders once; every subfolder below it is included."
        )
    return candidate


def extract_prompt_paths(prompt: str) -> list[str]:
    cleaned = str(prompt).replace("\\_", "_")
    raw = re.findall(r"(?<![A-Za-z0-9])(/[^\s`\"']+)", cleaned)
    paths: list[str] = []
    for value in raw:
        candidate = value.rstrip(".,;:?!)]}")
        if candidate not in paths:
            paths.append(candidate)
    return paths


def plan_online_evidence(prompt: str) -> list[dict[str, str]]:
    """Validate named files/folders without opening scientific file contents."""
    limits = load_online_share_limits()
    planned: list[dict[str, str]] = []
    for raw in extract_prompt_paths(prompt):
        source = require_online_share_path(raw, must_exist=True)
        candidates = [source] if source.is_file() else sorted(source.iterdir())
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            require_online_share_path(candidate, must_exist=True)
            kind = "image preview" if candidate.suffix.lower() in IMAGE_SUFFIXES else "table sample"
            planned.append({"path": str(candidate), "kind": kind})
            if len(planned) >= limits.max_files:
                return planned
    return planned


def _image_data_url(path: Path, *, max_size: int, require_he: bool) -> tuple[str, tuple[int, int]]:
    if path.suffix.lower() in {".tif", ".tiff", ".qptiff"}:
        from spacemap_gui.tabs.tab9_visualization.overlay_core import load_base_image

        image, _full_shape, _level = load_base_image(
            path,
            channels=None,
            max_size=max_size,
            require_he=require_he,
        )
    else:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", image.size


def _table_sample(path: Path, *, max_rows: int) -> tuple[str, int]:
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=separator)
        for row in reader:
            rows.append([str(value)[:500] for value in row[:100]])
            if len(rows) >= max_rows + 1:
                break
    output = io.StringIO()
    writer = csv.writer(output, delimiter=separator, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()[:200_000], max(0, len(rows) - 1)


def build_online_evidence_content(
    prompt: str,
    *,
    model_supports_images: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a bounded, sanitized multimodal user message for an approved request."""
    limits = load_online_share_limits()
    planned = plan_online_evidence(prompt)
    if extract_prompt_paths(prompt) and not planned:
        raise AssistantPathError(
            "No supported image or CSV/TSV file was found at the approved path."
        )
    sanitized = str(prompt).replace("\\_", "_")
    for index, raw in enumerate(extract_prompt_paths(prompt), start=1):
        sanitized = sanitized.replace(raw, f"[OWNER-APPROVED LOCAL DATA {index}]")
    blocks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    require_he = bool(re.search(r"\b(?:h&e|h\s+and\s+e|he)\b", prompt, re.IGNORECASE))
    table_sections: list[str] = []
    for index, item in enumerate(planned, start=1):
        path = Path(item["path"])
        if item["kind"] == "image preview":
            if not model_supports_images:
                raise AssistantPathError(
                    "The selected NVIDIA model does not accept images. Choose Muse Glimmer, "
                    "Gemma, or DiffusionGemma."
                )
            data_url, size = _image_data_url(
                path,
                max_size=limits.image_max_size,
                require_he=require_he and path.suffix.lower() in {".tif", ".tiff", ".qptiff"},
            )
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            manifest.append(
                {"path": str(path), "kind": "resized_png", "width": size[0], "height": size[1]}
            )
        else:
            sample, rows = _table_sample(path, max_rows=limits.csv_rows)
            table_sections.append(
                f"\nApproved table sample {index} ({rows} data rows; local filename hidden):\n```\n{sample}\n```"
            )
            manifest.append({"path": str(path), "kind": "table_sample", "rows": rows})
    blocks.append(
        {
            "type": "text",
            "text": sanitized + "".join(table_sections),
        }
    )
    return blocks, manifest


def build_online_preview_content(
    preview_paths: list[str | Path],
    prompt: str,
    *,
    model_supports_images: bool,
    context_note: str = "",
    max_size: int = 1024,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Send only locally generated review PNGs, never their whole-slide source."""
    if not model_supports_images:
        raise AssistantPathError(
            "The selected NVIDIA model does not accept images. Choose Muse Glimmer, Gemma, "
            "or DiffusionGemma."
        )
    sanitized = str(prompt).replace("\\_", "_")
    for index, raw in enumerate(extract_prompt_paths(prompt), start=1):
        sanitized = sanitized.replace(raw, f"[LOCAL SOURCE {index} OMITTED]")
    blocks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for raw_path in preview_paths[:8]:
        path = require_online_share_path(raw_path, must_exist=True)
        if path.suffix.casefold() != ".png":
            raise AssistantPathError("Online visual QC accepts only locally generated PNG previews.")
        data_url, size = _image_data_url(
            path,
            max_size=min(1024, max(256, int(max_size))),
            require_he=False,
        )
        blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        manifest.append(
            {
                "path": str(path),
                "kind": "generated_qc_preview",
                "width": size[0],
                "height": size[1],
            }
        )
    if not blocks:
        raise AssistantPathError("No generated review previews were available.")
    blocks.append(
        {
            "type": "text",
            "text": (
                sanitized
                + (
                    "\n\n" + str(context_note).strip()
                    if str(context_note).strip()
                    else (
                        "\n\nThe attached files are small SPACEMAP-generated review previews. "
                        "Image 1 is the overview; later images are numbered polygon tiles. "
                        "Give the assessment now. Do not ask for another confirmation and do not provide code."
                    )
                )
            ),
        }
    )
    return blocks, manifest
