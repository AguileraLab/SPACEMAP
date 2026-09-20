"""A narrow, approved action for registering one CODEX/H&E image pair.

The language model never calls the scientific functions itself.  This module
validates a typed plan, records it, and starts a fixed worker only after the UI
has collected explicit approval.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import (
    AI_ROOT,
    SOURCE_ROOT,
    ASSISTANT_DATA_DIR,
    action_workspace_root,
    is_within,
    require_assistant_read_path,
    require_assistant_write_path,
)


ACTIONS_DIR = ASSISTANT_DATA_DIR / "actions"
SUPPORTED_TIFF_SUFFIXES = (".qptiff", ".tif", ".tiff", ".ome.tif", ".ome.tiff")
REFERENCE_TERMS = ("dapi", "hoechst", "hoescht", "nuclear", "nuclei")


class RegistrationActionError(RuntimeError):
    """A safe, user-readable registration action failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    target = require_assistant_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_action_id(action_id: str) -> str:
    try:
        parsed = uuid.UUID(str(action_id))
    except (ValueError, TypeError, AttributeError) as error:
        raise RegistrationActionError("The registration action number is invalid.") from error
    if str(parsed) != str(action_id):
        raise RegistrationActionError("The registration action number is invalid.")
    return str(parsed)


def action_directory(action_id: str) -> Path:
    return require_assistant_write_path(ACTIONS_DIR / _validated_action_id(action_id))


def _plan_digest(action_id: str) -> str:
    plan_path = action_directory(action_id) / "plan.json"
    try:
        return hashlib.sha256(plan_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RegistrationActionError("The saved registration plan is missing.") from error


def _is_supported_tiff(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_TIFF_SUFFIXES)


def _validate_inputs(codex_path: str | Path, he_path: str | Path, output_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    codex = require_assistant_read_path(codex_path, must_exist=True)
    he = require_assistant_read_path(he_path, must_exist=True)
    output = require_assistant_write_path(output_dir)
    if not codex.is_file() or not _is_supported_tiff(codex):
        raise RegistrationActionError("The CODEX input must be an existing TIFF or QPTIFF file.")
    if not he.is_file() or not _is_supported_tiff(he):
        raise RegistrationActionError("The H&E input must be an existing TIFF or QPTIFF file.")
    if codex == he:
        raise RegistrationActionError("The CODEX and H&E inputs must be different files.")

    output_project = action_workspace_root(output)
    try:
        output_relative = output.relative_to(output_project)
    except ValueError as error:
        raise RegistrationActionError("The output folder is outside the selected AI project.") from error
    if not output_relative.parts or output_relative.parts[0].lower() != "outputs":
        raise RegistrationActionError("The action output folder must be inside the project's outputs folder.")
    if is_within(output, codex.parent) or is_within(output, he.parent):
        raise RegistrationActionError("The output folder cannot be inside an input image folder.")
    return codex, he, output, output_project


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _plain_stem(path: Path) -> str:
    name = path.name
    for suffix in sorted(SUPPORTED_TIFF_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"(?i)(?:[_\-. ](?:codex|h&e|hne|he))$", "", name).strip("_.- ")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "registered_image"


def _pair_prefix(codex: Path, he: Path) -> str:
    codex_stem = _plain_stem(codex)
    he_stem = _plain_stem(he)
    if codex_stem.lower() == he_stem.lower():
        return codex_stem
    return f"{codex_stem}_to_{he_stem}"[:120]


def inspect_image(path: Path, role: str) -> dict[str, Any]:
    """Read only the image header and channel metadata."""
    from spacemap.background_subtraction_utils import get_channel_info, get_image_mpp

    names, _colors, _metadata, _mpp = get_channel_info(str(path))
    detected_mpp = get_image_mpp(str(path)).get("codex")
    if role == "he" and len(names) <= 1:
        names = ["HE_Red", "HE_Green", "HE_Blue"]
    return {
        "channels": [str(name) for name in names],
        "mpp": None if detected_mpp is None else float(detected_mpp),
    }


@dataclass(frozen=True)
class RegistrationPlan:
    schema_version: int
    action_id: str
    action_type: str
    created_at: str
    project_root: str
    codex_path: str
    he_path: str
    output_dir: str
    prefix: str
    codex_mpp: float
    he_mpp: float
    reference_channel: str
    reference_channel_index: int
    n_keypoints: int
    pre_rotation_deg: float
    overwrite: bool
    existing_output_signature: dict[str, int] | None
    backup_output: str
    codex_channels: list[str]
    he_channels: list[str]
    codex_signature: dict[str, int]
    he_signature: dict[str, int]
    intermediate_dir: str
    partial_output: str
    final_output: str
    conversation_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_registration_plan(
    codex_path: str | Path,
    he_path: str | Path,
    output_dir: str | Path,
    *,
    reference_channel: str = "",
    codex_mpp: float | None = None,
    he_mpp: float | None = None,
    n_keypoints: int = 10_000,
    pre_rotation_deg: float = 0.0,
    overwrite: bool = False,
    conversation_id: str = "",
) -> RegistrationPlan:
    """Inspect inputs, create an immutable plan, and save it for review."""
    codex, he, output, project_root = _validate_inputs(codex_path, he_path, output_dir)
    keypoints = int(n_keypoints)
    rotation = float(pre_rotation_deg)
    if not 1_000 <= keypoints <= 100_000:
        raise RegistrationActionError("Registration keypoints must be between 1,000 and 100,000.")
    if rotation not in {-270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0}:
        raise RegistrationActionError("H&E pre-rotation must be a 90-degree step.")

    codex_info = inspect_image(codex, "codex")
    he_info = inspect_image(he, "he")
    resolved_codex_mpp = float(codex_mpp if codex_mpp is not None else codex_info["mpp"] or 0)
    resolved_he_mpp = float(he_mpp if he_mpp is not None else he_info["mpp"] or 0)
    if resolved_codex_mpp <= 0 or resolved_he_mpp <= 0:
        raise RegistrationActionError(
            "Microns-per-pixel (MPP) could not be read. Enter valid CODEX and H&E MPP values."
        )

    channel_names = codex_info["channels"]
    requested = str(reference_channel).strip()
    if requested:
        matching = [
            (index, name)
            for index, name in enumerate(channel_names)
            if name.lower() == requested.lower()
        ]
        if not matching:
            raise RegistrationActionError(f"Reference channel '{requested}' was not found in the CODEX image.")
        reference_index, reference_name = matching[0]
    else:
        reference_index, reference_name = 0, channel_names[0] if channel_names else "Channel 0"
        for index, name in enumerate(channel_names):
            if any(term in name.lower() for term in REFERENCE_TERMS):
                reference_index, reference_name = index, name
                break

    action_id = str(uuid.uuid4())
    prefix = _pair_prefix(codex, he)
    run_output_dir = output / "_spacemap_ai_actions" / action_id
    partial = output / f".{prefix}.spacemap-ai-{action_id}.partial.ome.tiff"
    final = output / f"{prefix}_merged.ome.tiff"
    replace_existing = bool(overwrite)
    if final.exists() and not replace_existing:
        raise RegistrationActionError(
            "The final output already exists. Choose the separate 'replace existing output' "
            f"option if you want a new plan to replace it: {final}"
        )
    existing_output_signature = _file_signature(final) if final.exists() else None
    backup = output / f"{prefix}_merged.before-{action_id[:8]}.ome.tiff"

    plan = RegistrationPlan(
        schema_version=1,
        action_id=action_id,
        action_type="register_codex_he_pair",
        created_at=utc_now(),
        project_root=str(project_root.resolve()),
        codex_path=str(codex),
        he_path=str(he),
        output_dir=str(output),
        prefix=prefix,
        codex_mpp=resolved_codex_mpp,
        he_mpp=resolved_he_mpp,
        reference_channel=reference_name,
        reference_channel_index=int(reference_index),
        n_keypoints=keypoints,
        pre_rotation_deg=rotation,
        overwrite=replace_existing,
        existing_output_signature=existing_output_signature,
        backup_output=str(backup),
        codex_channels=channel_names,
        he_channels=he_info["channels"],
        codex_signature=_file_signature(codex),
        he_signature=_file_signature(he),
        intermediate_dir=str(run_output_dir / "intermediate"),
        partial_output=str(partial),
        final_output=str(final),
        conversation_id=str(conversation_id).strip(),
    )
    directory = action_directory(action_id)
    _atomic_json(directory / "plan.json", plan.as_dict())
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "planned",
            "stage": "Waiting for user approval",
            "created_at": plan.created_at,
            "updated_at": utc_now(),
        },
    )
    return plan


def load_registration_plan(action_id: str, *, verify_inputs: bool = False) -> RegistrationPlan:
    directory = action_directory(action_id)
    try:
        payload = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        plan = RegistrationPlan(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RegistrationActionError("The saved registration plan is missing or invalid.") from error
    if plan.action_id != action_id or plan.action_type != "register_codex_he_pair":
        raise RegistrationActionError("The saved registration plan failed its safety check.")
    codex, he, output, project_root = _validate_inputs(plan.codex_path, plan.he_path, plan.output_dir)
    if str(project_root.resolve()) != plan.project_root:
        raise RegistrationActionError("The saved project boundary no longer matches the plan.")
    for generated in (plan.intermediate_dir, plan.partial_output, plan.final_output, plan.backup_output):
        if not is_within(generated, output):
            raise RegistrationActionError("A generated output path escaped the approved output folder.")
    if verify_inputs:
        if _file_signature(codex) != plan.codex_signature or _file_signature(he) != plan.he_signature:
            raise RegistrationActionError("An input image changed after the plan was created. Build a new plan.")
    return plan


def load_action_status(action_id: str) -> dict[str, Any]:
    try:
        return json.loads((action_directory(action_id) / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"action_id": action_id, "state": "unknown", "stage": "Status is unavailable"}


def list_registration_actions(limit: int = 25) -> list[dict[str, Any]]:
    """List recent durable registration actions without opening scientific images."""
    if not ACTIONS_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for plan_path in ACTIONS_DIR.glob("*/plan.json"):
        try:
            action_id = _validated_action_id(plan_path.parent.name)
            plan = load_registration_plan(action_id)
            status = load_action_status(action_id)
        except (RegistrationActionError, OSError, ValueError):
            continue
        items.append(
            {
                "action_id": action_id,
                "created_at": plan.created_at,
                "prefix": plan.prefix,
                "state": status.get("state", "unknown"),
                "final_output": plan.final_output,
            }
        )
    return sorted(items, key=lambda item: item["created_at"], reverse=True)[: max(1, int(limit))]


def launch_registration_action(action_id: str) -> int:
    """Record approval and start the fixed worker without a shell."""
    plan = load_registration_plan(action_id, verify_inputs=True)
    directory = action_directory(action_id)
    status = load_action_status(action_id)
    if status.get("state") != "planned":
        raise RegistrationActionError("This plan is not waiting for approval and cannot be started again.")
    final_output = Path(plan.final_output)
    if final_output.exists() and not plan.overwrite:
        raise RegistrationActionError("The final output now exists, so this action was stopped.")
    if plan.overwrite:
        if plan.existing_output_signature is None and final_output.exists():
            raise RegistrationActionError("An unreviewed output appeared after planning. Build a new plan.")
        if plan.existing_output_signature is not None:
            if not final_output.exists() or _file_signature(final_output) != plan.existing_output_signature:
                raise RegistrationActionError("The existing output changed after review. Build a new plan.")

    approved_at = utc_now()
    _atomic_json(
        directory / "approval.json",
        {
            "action_id": action_id,
            "approved_at": approved_at,
            "approval_method": "SPACEMAP AI review checkbox and start button",
            "plan_path": str(directory / "plan.json"),
            "plan_sha256": _plan_digest(action_id),
            "replace_existing_output": plan.overwrite,
        },
    )
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "launching",
            "stage": "Starting background worker",
            "created_at": plan.created_at,
            "approved_at": approved_at,
            "updated_at": utc_now(),
        },
    )

    source_root = SOURCE_ROOT
    environment = os.environ.copy()
    previous_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(source_root) + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
    environment["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        "-m",
        "spacemap_assistant.registration_worker",
        "--action-id",
        action_id,
    ]
    log_path = directory / "worker.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(AI_ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _atomic_json(
        directory / "launcher.json",
        {
            "action_id": action_id,
            "pid": int(process.pid),
            "launched_at": utc_now(),
            "worker": "spacemap_assistant.registration_worker",
            "log_path": str(log_path),
        },
    )
    return int(process.pid)


def recent_action_log(action_id: str, max_characters: int = 30_000) -> str:
    try:
        content = (action_directory(action_id) / "worker.log").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return "The worker has not written a log yet."
    return content[-max(1_000, int(max_characters)) :]


def parse_registration_request(prompt: str) -> dict[str, Any] | None:
    """Extract only clear absolute paths from a natural-language registration request."""
    cleaned = str(prompt).replace("\\_", "_")
    if "register" not in cleaned.lower():
        return None
    raw_paths = re.findall(r"(?<![A-Za-z0-9])(/[^\s`\"']+)", cleaned)
    paths = [value.rstrip(".,;:?!)]}") for value in raw_paths]
    image_paths = [value for value in paths if _is_supported_tiff(Path(value))]
    directory_paths = [value for value in paths if value not in image_paths]
    if len(image_paths) < 2:
        return None

    codex = next((value for value in image_paths if "/codex/" in value.lower()), image_paths[0])
    he = next((value for value in image_paths if "/he/" in value.lower() or "/hne/" in value.lower()), "")
    if not he:
        he = next((value for value in image_paths if value != codex), "")
    output = next((value for value in directory_paths if "/outputs/" in value.lower()), "")
    if not he or not output:
        return None
    overwrite_requested = any(term in cleaned.lower() for term in ("overwrite", "over-write", "replace existing"))
    return {
        "codex_path": codex,
        "he_path": he,
        "output_dir": output,
        "overwrite": overwrite_requested,
    }
