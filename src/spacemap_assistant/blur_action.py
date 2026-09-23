"""Checked background action for SPACEMAP blur-parameter experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import (
    SOURCE_ROOT,
    ASSISTANT_DATA_DIR,
    action_workspace_root,
    is_within,
    require_assistant_read_path,
    require_assistant_write_path,
)
from .registration_action import _file_signature, utc_now


ACTIONS_DIR = ASSISTANT_DATA_DIR / "actions"
SUPPORTED_TIFF_SUFFIXES = (".ome.tiff", ".ome.tif", ".qptiff", ".tiff", ".tif")


class BlurActionError(RuntimeError):
    """A short, user-readable blur action error."""


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
        raise BlurActionError("The blur experiment number is invalid.") from error
    if str(parsed) != str(action_id):
        raise BlurActionError("The blur experiment number is invalid.")
    return str(parsed)


def action_directory(action_id: str) -> Path:
    return require_assistant_write_path(ACTIONS_DIR / _validated_action_id(action_id))


def _sample_id(path: Path) -> str:
    lowered = path.name.casefold()
    for suffix in SUPPORTED_TIFF_SUFFIXES:
        if lowered.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _path_signature(path: Path | None) -> dict[str, int] | None:
    return _file_signature(path) if path is not None else None


@dataclass(frozen=True)
class BlurPlan:
    schema_version: int
    action_id: str
    action_type: str
    created_at: str
    user_request: str
    project_root: str
    image_path: str
    tissue_boundary_path: str
    project_config_path: str
    output_parent: str
    run_output_dir: str
    maximum_trials: int
    microns_per_pixel: float | None
    image_signature: dict[str, int]
    boundary_signature: dict[str, int]
    project_config_signature: dict[str, int] | None
    provider: str
    model_id: str
    # Kept on the durable action so its progress, result, and previews can be
    # recovered by the same chat after Streamlit reruns or model switches.
    conversation_id: str = ""
    request_message_id: str = ""
    artifact_kind: str = "blur"
    automatic_visual_review: bool = True
    online_visual_review_approved: bool = False
    include_scale_bars: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_blur_plan(
    image_path: str | Path,
    tissue_boundary_path: str | Path,
    output_parent: str | Path,
    *,
    project_config_path: str | Path | None = None,
    maximum_trials: int = 10,
    microns_per_pixel: float | None = None,
    user_request: str = "",
    provider: str = "local",
    model_id: str = "",
    conversation_id: str = "",
    request_message_id: str = "",
    artifact_kind: str = "blur",
    automatic_visual_review: bool = True,
    online_visual_review_approved: bool = False,
    include_scale_bars: bool = False,
) -> BlurPlan:
    kind = str(artifact_kind).strip().casefold()
    if kind not in {"blur", "bright"}:
        raise BlurActionError("This artifact experiment supports blur or bright glare/debris.")
    image = require_assistant_read_path(image_path, must_exist=True)
    boundary = require_assistant_read_path(tissue_boundary_path, must_exist=True)
    output = require_assistant_write_path(output_parent)
    project_root = action_workspace_root(output)
    relative = output.relative_to(project_root)
    if not relative.parts or relative.parts[0].casefold() != "outputs":
        raise BlurActionError("Choose an output folder inside the project's outputs folder.")
    if not image.is_file() or not image.name.casefold().endswith(SUPPORTED_TIFF_SUFFIXES):
        raise BlurActionError("The artifact input must be an existing TIFF or QPTIFF image.")
    if not boundary.is_file() or boundary.suffix.casefold() not in {".geojson", ".json"}:
        raise BlurActionError("The tissue boundary must be an existing GeoJSON file.")
    project_config: Path | None = None
    if project_config_path:
        project_config = require_assistant_read_path(project_config_path, must_exist=True)
        if not project_config.is_file() or project_config.suffix.casefold() != ".json":
            raise BlurActionError("The project settings file must be an existing JSON file.")
    limit = int(maximum_trials)
    if not 1 <= limit <= 20:
        raise BlurActionError("Choose between 1 and 20 artifact trials.")
    mpp = None if microns_per_pixel is None else float(microns_per_pixel)
    if mpp is not None and mpp <= 0:
        raise BlurActionError("Microns per pixel must be greater than zero.")

    action_id = str(uuid.uuid4())
    run_output = output / f"_spacemap_ai_{kind}_runs" / f"{_sample_id(image)}_{action_id[:8]}"
    if not is_within(run_output, output):
        raise BlurActionError("The artifact run folder escaped the selected output folder.")
    plan = BlurPlan(
        schema_version=1,
        action_id=action_id,
        action_type=("optimize_blur_detection" if kind == "blur" else "optimize_artifact_detection"),
        created_at=utc_now(),
        user_request=str(user_request).strip()[:20_000],
        project_root=str(project_root),
        image_path=str(image),
        tissue_boundary_path=str(boundary),
        project_config_path=str(project_config or ""),
        output_parent=str(output),
        run_output_dir=str(run_output),
        maximum_trials=limit,
        microns_per_pixel=mpp,
        image_signature=_file_signature(image),
        boundary_signature=_file_signature(boundary),
        project_config_signature=_path_signature(project_config),
        provider=str(provider),
        model_id=str(model_id),
        conversation_id=str(conversation_id).strip(),
        request_message_id=str(request_message_id).strip(),
        artifact_kind=kind,
        automatic_visual_review=bool(automatic_visual_review),
        online_visual_review_approved=bool(online_visual_review_approved),
        include_scale_bars=bool(include_scale_bars),
    )
    directory = action_directory(action_id)
    _atomic_json(directory / "plan.json", plan.as_dict())
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "planned",
            "stage": "Waiting for approval",
            "message": "The inputs and output folder were checked.",
            "created_at": plan.created_at,
            "updated_at": utc_now(),
        },
    )
    return plan


def load_blur_plan(action_id: str, *, verify_inputs: bool = False) -> BlurPlan:
    directory = action_directory(action_id)
    try:
        plan = BlurPlan(**json.loads((directory / "plan.json").read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BlurActionError("The saved blur plan is missing or invalid.") from error
    if plan.action_id != action_id or plan.action_type not in {
        "optimize_blur_detection",
        "optimize_artifact_detection",
    }:
        raise BlurActionError("The saved artifact plan failed its check.")
    image = require_assistant_read_path(plan.image_path, must_exist=True)
    boundary = require_assistant_read_path(plan.tissue_boundary_path, must_exist=True)
    output = require_assistant_write_path(plan.output_parent)
    if action_workspace_root(output) != Path(plan.project_root).resolve():
        raise BlurActionError("The selected project boundary changed after planning.")
    if not is_within(plan.run_output_dir, output):
        raise BlurActionError("The saved artifact output escaped the selected output folder.")
    config = None
    if plan.project_config_path:
        config = require_assistant_read_path(plan.project_config_path, must_exist=True)
    if verify_inputs:
        if _file_signature(image) != plan.image_signature:
            raise BlurActionError("The image changed after planning. Build a new plan.")
        if _file_signature(boundary) != plan.boundary_signature:
            raise BlurActionError("The tissue boundary changed after planning. Build a new plan.")
        if _path_signature(config) != plan.project_config_signature:
            raise BlurActionError("The project settings changed after planning. Build a new plan.")
    return plan


def load_action_status(action_id: str) -> dict[str, Any]:
    try:
        return json.loads((action_directory(action_id) / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"action_id": action_id, "state": "unknown", "stage": "Status unavailable"}


def list_blur_actions(limit: int = 25) -> list[dict[str, Any]]:
    if not ACTIONS_DIR.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in ACTIONS_DIR.glob("*/plan.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("action_type") not in {
                "optimize_blur_detection",
                "optimize_artifact_detection",
            }:
                continue
            action_id = _validated_action_id(path.parent.name)
            plan = load_blur_plan(action_id)
            status = load_action_status(action_id)
        except (BlurActionError, OSError, ValueError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "action_id": action_id,
                "created_at": plan.created_at,
                "image_name": Path(plan.image_path).name,
                "trial_count": plan.maximum_trials,
                "state": str(status.get("state") or "unknown"),
                "artifact_kind": plan.artifact_kind,
            }
        )
    return sorted(rows, key=lambda item: item["created_at"], reverse=True)[: max(1, int(limit))]


def _worker_command(plan: BlurPlan, directory: Path) -> list[str]:
    executable = Path(sys.executable).resolve()
    if not Path("/usr/bin/bwrap").is_file():
        return [
            str(executable),
            "-m",
            "spacemap_assistant.blur_worker",
            "--action-id",
            plan.action_id,
        ]
    return [
        "/usr/bin/bwrap",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        plan.project_root,
        plan.project_root,
        "--bind",
        str(directory),
        str(directory),
        "--bind",
        str(ASSISTANT_DATA_DIR / "database"),
        str(ASSISTANT_DATA_DIR / "database"),
        "--chdir",
        plan.project_root,
        "--setenv",
        "HOME",
        str(directory / "runtime_home"),
        "--setenv",
        "MPLCONFIGDIR",
        str(directory / "runtime_home" / ".matplotlib"),
        "--setenv",
        "NUMBA_CACHE_DIR",
        str(directory / "runtime_home" / ".numba"),
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONPATH",
        str(SOURCE_ROOT),
        str(executable),
        "-m",
        "spacemap_assistant.blur_worker",
        "--action-id",
        plan.action_id,
    ]


def launch_blur_action(action_id: str, *, approval_method: str) -> int:
    plan = load_blur_plan(action_id, verify_inputs=True)
    directory = action_directory(action_id)
    if load_action_status(action_id).get("state") != "planned":
        raise BlurActionError("This artifact plan is not waiting for approval.")
    if Path(plan.run_output_dir).exists():
        raise BlurActionError("The unique run folder already exists. Build a new plan.")
    (directory / "runtime_home").mkdir(parents=True, exist_ok=True)
    approval = {
        "action_id": action_id,
        "approved_at": utc_now(),
        "approval_method": str(approval_method),
        "plan_sha256": hashlib.sha256((directory / "plan.json").read_bytes()).hexdigest(),
    }
    _atomic_json(directory / "approval.json", approval)
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "launching",
            "stage": f"Starting {plan.artifact_kind} experiment",
            "message": "The checked background worker is starting.",
            "created_at": plan.created_at,
            "updated_at": utc_now(),
        },
    )
    log_handle = (directory / "worker.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            _worker_command(plan, directory),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env={
                **os.environ,
                "PYTHONPATH": str(SOURCE_ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    finally:
        log_handle.close()
    _atomic_json(directory / "process.json", {"pid": process.pid, "started_at": utc_now()})
    return int(process.pid)


_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(/[A-Za-z0-9_.~+@%:=,()\[\]/\\-]+)", re.MULTILINE
)


def _prompt_paths(prompt: str) -> list[Path]:
    paths: list[Path] = []
    for match in _PATH_PATTERN.finditer(str(prompt).replace("\\_", "_")):
        raw = match.group(1).rstrip(".,;:!?)]}\"")
        candidate = Path(raw).expanduser().resolve(strict=False)
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _infer_project_root(path: Path) -> Path | None:
    parts = path.parts
    lowered = [item.casefold() for item in parts]
    if "projects" not in lowered:
        return None
    index = len(lowered) - 1 - lowered[::-1].index("projects")
    if len(parts) <= index + 1:
        return None
    return Path(*parts[: index + 2])


def _configured_annotation_output(project_root: Path, config: Path | None) -> Path:
    fallback = project_root / "outputs" / "processed_annotations"
    if config is None:
        return fallback
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        raw = str((payload.get("directory_mappings") or {}).get("annot_ta_output_dir") or "")
    except (OSError, TypeError, json.JSONDecodeError):
        return fallback
    if not raw:
        return fallback
    return Path(raw.replace("${PROJECT_ROOT}", str(project_root))).expanduser().resolve(strict=False)


def parse_artifact_optimization_request(prompt: str) -> dict[str, Any] | None:
    """Translate blur or bright-glare optimization language without asking a model for code."""
    text = str(prompt).replace("\\_", "_")
    lowered = text.casefold()
    if any(
        word in lowered
        for word in ("bright glare", "glare and debris", "bright spot", "bright artifact", "bright debris")
    ):
        artifact_kind = "bright"
    elif any(word in lowered for word in ("blur", "blurry", "out of focus", "out-of-focus")):
        artifact_kind = "blur"
    else:
        return None
    action_words = (
        "detect",
        "test",
        "optimize",
        "optimise",
        "best parameter",
        "best setting",
        "parameter search",
    )
    if not any(word in lowered for word in action_words):
        return None
    paths = _prompt_paths(text)
    images = [path for path in paths if path.name.casefold().endswith(SUPPORTED_TIFF_SUFFIXES)]
    explicit_geojson = [path for path in paths if path.suffix.casefold() in {".geojson", ".json"}]
    if not images:
        return None
    image = images[0]
    project_root = _infer_project_root(image)
    config = None
    if project_root is not None:
        candidate = project_root / f"{project_root.name}.json"
        if candidate.is_file():
            config = candidate
    directory_paths = [path for path in paths if path not in images and path not in explicit_geojson]
    output = directory_paths[-1] if directory_paths else None
    if output is None and project_root is not None:
        output = _configured_annotation_output(project_root, config)
    if output is None:
        return None
    sample = _sample_id(image)
    boundary = next(
        (path for path in explicit_geojson if "tissue_boundary" in path.name.casefold()),
        None,
    )
    search_roots = [output]
    if project_root is not None:
        processed = project_root / "outputs" / "processed_annotations"
        search_roots.extend(
            [
                processed / "stage0_tissue_annotation",
                processed,
                *(path for path in processed.glob("*") if path.is_dir()),
            ]
        )
    if boundary is None:
        candidates = [root / f"{sample}_tissue_boundary.geojson" for root in search_roots]
        boundary = next((path for path in candidates if path.is_file()), candidates[0])
    trial_match = re.search(r"(?:test|try|run)\s+(\d{1,2})\s+(?:settings|configurations|trials)", lowered)
    maximum_trials = min(20, max(1, int(trial_match.group(1)))) if trial_match else 10
    mpp_match = re.search(r"(?:mpp|microns?\s+per\s+pixel|pixel\s+is)\D{0,20}(\d+(?:\.\d+)?)", lowered)
    return {
        "image_path": str(image),
        "tissue_boundary_path": str(boundary),
        "output_parent": str(output),
        "project_config_path": str(config or ""),
        "maximum_trials": maximum_trials,
        "microns_per_pixel": float(mpp_match.group(1)) if mpp_match else None,
        "user_request": text,
        "artifact_kind": artifact_kind,
        "include_scale_bars": bool(re.search(r"\bscale[\s-]*bar\b", lowered)),
    }


def parse_blur_optimization_request(prompt: str) -> dict[str, Any] | None:
    """Compatibility wrapper for older callers that specifically request blur."""
    draft = parse_artifact_optimization_request(prompt)
    return draft if draft is not None and draft.get("artifact_kind") == "blur" else None
