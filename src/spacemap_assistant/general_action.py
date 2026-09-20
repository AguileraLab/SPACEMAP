"""Protected general SPACEMAP job plans generated from read-only source evidence."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
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
from .registration_action import _file_signature, utc_now


ACTIONS_DIR = ASSISTANT_DATA_DIR / "actions"
MAX_CODE_CHARACTERS = 60_000
MAX_INPUTS = 50
ALLOWED_IMPORT_ROOTS = {
    "anndata",
    "collections",
    "cv2",
    "itertools",
    "json",
    "math",
    "matplotlib",
    "numpy",
    "pandas",
    "pathlib",
    "scanpy",
    "scipy",
    "seaborn",
    "skimage",
    "sklearn",
    "spacemap",
    "spacemap_gui",
    "statistics",
    "time",
    "tifffile",
}
FORBIDDEN_IMPORT_ROOTS = {
    "asyncio",
    "ctypes",
    "http",
    "importlib",
    "multiprocessing",
    "os",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "help",
    "input",
    "open",
    "quit",
}
RESERVED_GLOBALS = {"INPUTS", "OUTPUT_DIR", "PARAMETERS"}


class GeneralActionError(RuntimeError):
    """A clear validation or execution-plan failure."""


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
        raise GeneralActionError("The AI job number is invalid.") from error
    if str(parsed) != str(action_id):
        raise GeneralActionError("The AI job number is invalid.")
    return str(parsed)


def action_directory(action_id: str) -> Path:
    return require_assistant_write_path(ACTIONS_DIR / _validated_action_id(action_id))


def _plan_digest(action_id: str) -> str:
    try:
        return hashlib.sha256((action_directory(action_id) / "plan.json").read_bytes()).hexdigest()
    except OSError as error:
        raise GeneralActionError("The saved AI job plan is missing.") from error


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if path.is_file():
        return {"kind": "file", **_file_signature(path)}
    try:
        children = sum(1 for _ in path.iterdir())
    except OSError:
        children = -1
    return {
        "kind": "directory",
        "mtime_ns": int(stat.st_mtime_ns),
        "child_count": children,
    }


def _assignment_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _assignment_names(item)}
    return set()


def validate_generated_code(code: str, project_root: Path) -> None:
    """Reject shell/network/dynamic-code features before the sandbox sees a job."""
    text = str(code)
    if not text.strip() or len(text) > MAX_CODE_CHARACTERS:
        raise GeneralActionError("The generated job code is empty or too large.")
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as error:
        raise GeneralActionError(f"The generated job code is not valid Python: {error}") from error
    referenced_output = False
    referenced_spacemap = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                    raise GeneralActionError(f"Import '{alias.name}' is not allowed in a scientific job.")
                referenced_spacemap = referenced_spacemap or root in {"spacemap", "spacemap_gui"}
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if node.level or root in FORBIDDEN_IMPORT_ROOTS or root not in ALLOWED_IMPORT_ROOTS:
                raise GeneralActionError(f"Import '{node.module}' is not allowed in a scientific job.")
            referenced_spacemap = referenced_spacemap or root in {"spacemap", "spacemap_gui"}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise GeneralActionError(f"Call '{node.func.id}' is not allowed in a scientific job.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise GeneralActionError("Double-underscore object access is not allowed in a scientific job.")
        elif isinstance(node, ast.While):
            raise GeneralActionError("Open-ended while loops are not allowed. Use a clear bounded trial list.")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise GeneralActionError("Global and nonlocal statements are not allowed in a scientific job.")
        elif isinstance(node, ast.Assign):
            assigned = {name for target in node.targets for name in _assignment_names(target)}
            if assigned & RESERVED_GLOBALS:
                raise GeneralActionError("The job cannot replace its protected input/output variables.")
        elif isinstance(node, ast.AnnAssign):
            if _assignment_names(node.target) & RESERVED_GLOBALS:
                raise GeneralActionError("The job cannot replace its protected input/output variables.")
        elif isinstance(node, ast.AugAssign):
            if _assignment_names(node.target) & RESERVED_GLOBALS:
                raise GeneralActionError("The job cannot replace its protected input/output variables.")
        elif isinstance(node, ast.Name) and node.id == "OUTPUT_DIR":
            referenced_output = True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if value.startswith("/"):
                literal_path = Path(value).resolve(strict=False)
                if not is_within(literal_path, project_root):
                    raise GeneralActionError(
                        "Generated code contains an absolute path outside the selected project. "
                        "Use the checked INPUTS values instead."
                    )
    if not referenced_spacemap:
        raise GeneralActionError("The job does not call or import SPACEMAP scientific code.")
    if not referenced_output:
        raise GeneralActionError("The job must save reviewable results under OUTPUT_DIR.")


def _registered_tools_in_code(code: str) -> list[str]:
    """Return catalog tool names referenced by ordinary import/call syntax."""
    from spacemap_workflows.registry import list_tools

    callable_to_name = {item.callable_path: item.name for item in list_tools()}
    tree = ast.parse(str(code), mode="exec")
    imports: dict[str, str] = {}
    instances: dict[str, str] = {}
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                callable_path = f"{node.module}:{alias.name}"
                imports[alias.asname or alias.name] = callable_path
                if callable_path in callable_to_name:
                    used.add(callable_to_name[callable_path])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            continue
        imported = imports.get(value.func.id, "")
        if ":" not in imported:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for name in _assignment_names(target):
                instances[name] = imported
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            callable_path = imports.get(node.func.id, "")
            if callable_path in callable_to_name:
                used.add(callable_to_name[callable_path])
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            owner = instances.get(node.func.value.id) or imports.get(node.func.value.id, "")
            if ":" in owner:
                module, symbol = owner.split(":", 1)
                callable_path = f"{module}:{symbol}.{node.func.attr}"
            else:
                callable_path = f"{owner}:{node.func.attr}" if owner else ""
            if callable_path in callable_to_name:
                used.add(callable_to_name[callable_path])
    return sorted(used)


def _validate_parameter_paths(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _validate_parameter_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_parameter_paths(child)
    elif isinstance(value, str) and value.strip().startswith("/"):
        require_assistant_read_path(value, must_exist=False)


@dataclass(frozen=True)
class GeneralPlan:
    schema_version: int
    action_id: str
    action_type: str
    created_at: str
    title: str
    user_request: str
    project_root: str
    inputs: dict[str, str]
    input_signatures: dict[str, dict[str, Any]]
    parameters: dict[str, Any]
    code: str
    code_sha256: str
    output_dir: str
    staging_dir: str
    backup_dir: str
    overwrite: bool
    existing_output_signature: dict[str, Any] | None
    max_runtime_seconds: int
    registered_tools: list[str] = field(default_factory=list)
    planner_provider: str = "local"
    planner_model: str = ""
    requested_output_dir: str = ""
    conversation_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_general_plan(
    *,
    title: str,
    user_request: str,
    inputs: dict[str, str],
    parameters: dict[str, Any] | None,
    code: str,
    output_dir: str | Path,
    overwrite: bool = False,
    max_runtime_seconds: int = 43_200,
    planner_provider: str = "local",
    planner_model: str = "",
    conversation_id: str = "",
) -> GeneralPlan:
    if not isinstance(inputs, dict) or not 1 <= len(inputs) <= MAX_INPUTS:
        raise GeneralActionError("A scientific AI job needs between 1 and 50 named inputs.")
    resolved_inputs: dict[str, Path] = {}
    for name, value in inputs.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", str(name)):
            raise GeneralActionError(f"Input name '{name}' is invalid.")
        path = require_assistant_read_path(value, must_exist=True)
        resolved_inputs[str(name)] = path

    requested_output = require_assistant_write_path(output_dir)
    project_root = action_workspace_root(requested_output)
    relative = requested_output.relative_to(project_root)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "outputs":
        raise GeneralActionError("Choose a named subfolder inside the project's outputs folder.")
    replacing = bool(overwrite)
    overwrite_words = ("overwrite", "over-write", "replace existing", "replace the existing")
    if replacing and not any(word in str(user_request).lower() for word in overwrite_words):
        raise GeneralActionError(
            "The generated job requested replacement, but the user's instruction did not explicitly ask to overwrite."
        )
    if requested_output.exists() and not requested_output.is_dir():
        raise GeneralActionError("The general AI job output must be a folder, not a file.")
    action_id = str(uuid.uuid4())
    final_output = requested_output
    if requested_output.exists() and not replacing:
        safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", str(title)).strip("_")[:48]
        final_output = require_assistant_write_path(
            requested_output / "_spacemap_ai_runs" / f"{safe_title or 'job'}_{action_id[:8]}"
        )
    if any(is_within(final_output, path) for path in resolved_inputs.values() if path.is_dir()):
        raise GeneralActionError("The output folder cannot be inside an input folder.")

    validate_generated_code(code, project_root)
    try:
        json.dumps(parameters or {}, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GeneralActionError("Job parameters must contain simple JSON values.") from error
    _validate_parameter_paths(parameters or {})
    registered_tools = _registered_tools_in_code(code)
    if not registered_tools:
        raise GeneralActionError(
            "The job does not call a registered public SPACEMAP function. Ask the planner to use one "
            "of the exact functions shown in the tool catalog."
        )
    runtime = int(max_runtime_seconds)
    if not 60 <= runtime <= 172_800:
        raise GeneralActionError("Maximum runtime must be between 1 minute and 48 hours.")

    staging = project_root / "outputs" / ".spacemap_ai_staging" / action_id
    backup = final_output.with_name(f"{final_output.name}.before-{action_id[:8]}")
    plan = GeneralPlan(
        schema_version=1,
        action_id=action_id,
        action_type="general_spacemap_job",
        created_at=utc_now(),
        title=str(title).strip()[:200] or "SPACEMAP AI job",
        user_request=str(user_request).strip()[:20_000],
        project_root=str(project_root),
        inputs={name: str(path) for name, path in resolved_inputs.items()},
        input_signatures={name: _path_signature(path) for name, path in resolved_inputs.items()},
        parameters=dict(parameters or {}),
        code=str(code),
        code_sha256=hashlib.sha256(str(code).encode("utf-8")).hexdigest(),
        output_dir=str(final_output),
        staging_dir=str(staging),
        backup_dir=str(backup),
        overwrite=replacing,
        existing_output_signature=_path_signature(final_output) if final_output.exists() else None,
        max_runtime_seconds=runtime,
        registered_tools=registered_tools,
        planner_provider=str(planner_provider),
        planner_model=str(planner_model),
        requested_output_dir=str(requested_output),
        conversation_id=str(conversation_id).strip(),
    )
    directory = action_directory(action_id)
    _atomic_json(directory / "plan.json", plan.as_dict())
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "planned",
            "stage": "Waiting for approval",
            "created_at": plan.created_at,
            "updated_at": utc_now(),
        },
    )
    return plan


def load_general_plan(action_id: str, *, verify_inputs: bool = False) -> GeneralPlan:
    directory = action_directory(action_id)
    try:
        plan = GeneralPlan(**json.loads((directory / "plan.json").read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise GeneralActionError("The saved AI job plan is missing or invalid.") from error
    if plan.action_id != action_id or plan.action_type != "general_spacemap_job":
        raise GeneralActionError("The saved AI job failed its safety check.")
    project_root = Path(plan.project_root).resolve()
    if action_workspace_root(project_root) != project_root:
        raise GeneralActionError("The saved AI project boundary is invalid.")
    validate_generated_code(plan.code, project_root)
    if hashlib.sha256(plan.code.encode("utf-8")).hexdigest() != plan.code_sha256:
        raise GeneralActionError("The generated code changed after the plan was saved.")
    for path in (plan.output_dir, plan.staging_dir, plan.backup_dir):
        if not is_within(path, project_root):
            raise GeneralActionError("A generated path escaped the selected AI project.")
    if verify_inputs:
        for name, value in plan.inputs.items():
            path = require_assistant_read_path(value, must_exist=True)
            if _path_signature(path) != plan.input_signatures[name]:
                raise GeneralActionError(f"Input '{name}' changed after planning. Build a new plan.")
    return plan


def load_action_status(action_id: str) -> dict[str, Any]:
    try:
        return json.loads((action_directory(action_id) / "status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"action_id": action_id, "state": "unknown", "stage": "Status unavailable"}


def list_general_actions(limit: int = 25) -> list[dict[str, Any]]:
    if not ACTIONS_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for plan_path in ACTIONS_DIR.glob("*/plan.json"):
        payload: dict[str, Any]
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("action_type") != "general_spacemap_job":
            continue
        try:
            action_id = _validated_action_id(plan_path.parent.name)
            plan = load_general_plan(action_id)
            status = load_action_status(action_id)
        except (GeneralActionError, OSError, ValueError):
            continue
        items.append(
            {
                "action_id": action_id,
                "created_at": plan.created_at,
                "title": plan.title,
                "state": status.get("state", "unknown"),
                "output_dir": plan.output_dir,
            }
        )
    return sorted(items, key=lambda item: item["created_at"], reverse=True)[: max(1, int(limit))]


def _bubblewrap_command(plan: GeneralPlan, directory: Path) -> list[str]:
    executable = Path(sys.executable).resolve()
    if not Path("/usr/bin/bwrap").is_file():
        raise GeneralActionError("The protected job sandbox (bubblewrap) is not installed.")
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
        "spacemap_assistant.general_worker",
        "--action-id",
        plan.action_id,
    ]


def launch_general_action(action_id: str, *, approval_method: str) -> int:
    plan = load_general_plan(action_id, verify_inputs=True)
    directory = action_directory(action_id)
    if load_action_status(action_id).get("state") != "planned":
        raise GeneralActionError("This AI job is not waiting for approval and cannot start again.")
    final = Path(plan.output_dir)
    if final.exists() and not plan.overwrite:
        raise GeneralActionError("The output folder now exists, so the job was stopped.")
    if plan.overwrite and plan.existing_output_signature is not None:
        if not final.exists() or _path_signature(final) != plan.existing_output_signature:
            raise GeneralActionError("The existing output folder changed after review.")

    approved_at = utc_now()
    _atomic_json(
        directory / "approval.json",
        {
            "action_id": action_id,
            "approved_at": approved_at,
            "approval_method": str(approval_method),
            "plan_sha256": _plan_digest(action_id),
            "overwrite": plan.overwrite,
        },
    )
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": "launching",
            "stage": "Starting protected worker",
            "created_at": plan.created_at,
            "approved_at": approved_at,
            "updated_at": utc_now(),
        },
    )
    environment = os.environ.copy()
    source_root = SOURCE_ROOT
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(source_root) + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = _bubblewrap_command(plan, directory)
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
            "worker": "spacemap_assistant.general_worker inside bubblewrap",
            "log_path": str(log_path),
        },
    )
    return int(process.pid)


def parse_job_envelope(answer: str) -> tuple[str, dict[str, Any] | None]:
    """Remove and decode one model-generated job envelope from a chat answer."""
    text = str(answer)
    pattern = re.compile(r"<spacemap_job>\s*(\{.*?\})\s*</spacemap_job>", re.DOTALL)
    match = pattern.search(text)
    if match is None:
        if "<spacemap_job>" in text:
            raise GeneralActionError(
                "The model's internal job plan was incomplete, so nothing was prepared or run."
            )
        return text, None
    clean_answer = pattern.sub("", text).strip()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise GeneralActionError(f"The model's job plan was not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GeneralActionError("The model's job plan must be a JSON object.")
    return clean_answer, payload
