"""Sandboxed worker for a reviewed AI-generated SPACEMAP Python job."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .general_action import (
    ALLOWED_IMPORT_ROOTS,
    FORBIDDEN_CALLS,
    GeneralActionError,
    _atomic_json,
    _path_signature,
    _plan_digest,
    action_directory,
    load_general_plan,
)
from .paths import ASSISTANT_DATA_DIR, is_within
from .registration_action import utc_now
from .storage import AssistantStore


def _append_event(
    action_id: str,
    stage: str,
    message: str,
    *,
    state: str = "running",
    started: float,
) -> None:
    directory = action_directory(action_id)
    event = {
        "created_at": utc_now(),
        "state": state,
        "stage": stage,
        "message": str(message),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    with (directory / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str) + "\n")
    _atomic_json(
        directory / "status.json",
        {
            "action_id": action_id,
            "state": state,
            "stage": stage,
            "message": str(message),
            "elapsed_seconds": event["elapsed_seconds"],
            "updated_at": event["created_at"],
        },
    )
    print(f"[{event['created_at']}] {stage}: {message}", flush=True)


def _write_requested(arguments: tuple[Any, ...]) -> tuple[Path | None, bool]:
    if not arguments:
        return None, False
    raw_path = arguments[0]
    if isinstance(raw_path, int):
        return None, False
    try:
        path = Path(os.fsdecode(os.fspath(raw_path))).resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return None, False
    mode = arguments[1] if len(arguments) > 1 else "r"
    if isinstance(mode, str):
        writes = any(flag in mode for flag in ("w", "a", "x", "+"))
    elif isinstance(mode, int):
        writes = bool(
            mode
            & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
    else:
        writes = False
    return path, writes


def _audit_path(raw_path: Any) -> Path | None:
    if isinstance(raw_path, int):
        return None
    try:
        return Path(os.fsdecode(os.fspath(raw_path))).resolve(strict=False)
    except (TypeError, ValueError, OSError, UnicodeError):
        return None


def _install_runtime_guard(initial_write_roots: list[Path]) -> list[Path]:
    """Enforce the reviewed write roots even inside the filesystem sandbox."""
    allowed_write_roots = [path.resolve(strict=False) for path in initial_write_roots]

    def allowed(path: Path) -> bool:
        return any(is_within(path, root) for root in allowed_write_roots)

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event == "open":
            path, writes = _write_requested(arguments)
            if writes and path is not None and not allowed(path):
                raise PermissionError(f"AI job write blocked outside its temporary output: {path}")
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown"}:
            if arguments:
                path = _audit_path(arguments[0])
                if path is not None and not allowed(path):
                    raise PermissionError(f"AI job filesystem change blocked: {path}")
        elif event in {"os.rename", "os.replace"}:
            for raw in arguments[:2]:
                path = _audit_path(raw)
                if path is not None and not allowed(path):
                    raise PermissionError(f"AI job move blocked: {path}")
        elif event.startswith("subprocess") or event in {"os.system", "os.posix_spawn"}:
            raise PermissionError("AI-generated jobs cannot start shell commands or child processes.")
        elif event in {"socket.bind", "socket.connect", "socket.connect_ex", "socket.getaddrinfo"}:
            raise PermissionError("AI-generated scientific jobs cannot use the network.")

    sys.addaudithook(audit)
    return allowed_write_roots


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    root = str(name).split(".", 1)[0]
    if level or root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import '{name}' is not allowed in this AI job.")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    values = dict(vars(builtins))
    for name in FORBIDDEN_CALLS:
        values.pop(name, None)
    values["__import__"] = _safe_import
    return values


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("The AI job reached its reviewed maximum runtime.")


def _artifact_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        return "ai_generated_figure"
    if name.endswith((".tif", ".tiff", ".qptiff")):
        return "ai_generated_image"
    if name.endswith((".csv", ".parquet", ".tsv")):
        return "ai_generated_table"
    if name.endswith((".json", ".geojson")):
        return "ai_generated_data"
    return "ai_generated_artifact"


def _staged_files(staging: Path) -> list[Path]:
    files: list[Path] = []
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise GeneralActionError(f"Symbolic links are not allowed in AI job output: {path}")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(staging).parts):
            files.append(path)
    return sorted(files)


def run_general_action(action_id: str) -> int:
    started = time.monotonic()
    directory = action_directory(action_id)
    try:
        plan = load_general_plan(action_id, verify_inputs=True)
        try:
            approval = json.loads((directory / "approval.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GeneralActionError("This AI job has no valid approval record.") from error
        if approval.get("action_id") != action_id or approval.get("plan_sha256") != _plan_digest(action_id):
            raise GeneralActionError("The AI job no longer matches the approved plan.")

        project_root = Path(plan.project_root).resolve()
        staging = Path(plan.staging_dir).resolve()
        final = Path(plan.output_dir).resolve()
        backup = Path(plan.backup_dir).resolve()
        runtime_home = directory / "runtime_home"
        if staging.exists():
            raise GeneralActionError("This job already has a temporary output folder and will not reuse it.")
        if final.exists() and not plan.overwrite:
            raise GeneralActionError("The final output folder exists; replacement was not approved.")
        if plan.overwrite and plan.existing_output_signature is not None:
            if not final.exists() or _path_signature(final) != plan.existing_output_signature:
                raise GeneralActionError("The existing output changed after approval.")
            if backup.exists():
                raise GeneralActionError("The planned backup folder already exists.")
        staging.mkdir(parents=True, exist_ok=False)
        runtime_home.mkdir(parents=True, exist_ok=True)
        allowed_write_roots = _install_runtime_guard([staging, directory, Path("/tmp")])

        _append_event(
            action_id,
            "Starting",
            "Protected filesystem is active; SPACEMAP source is read-only.",
            started=started,
        )
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(plan.max_runtime_seconds)
        globals_dict: dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "__name__": "__spacemap_ai_job__",
            "INPUTS": MappingProxyType(dict(plan.inputs)),
            "PARAMETERS": MappingProxyType(dict(plan.parameters)),
            "OUTPUT_DIR": staging,
        }
        _append_event(
            action_id,
            "Running scientific code",
            plan.title,
            started=started,
        )
        compiled = compile(plan.code, f"<spacemap-ai-job-{action_id[:8]}>", "exec")
        exec(compiled, globals_dict, globals_dict)
        signal.alarm(0)

        files = _staged_files(staging)
        if not files:
            raise GeneralActionError(
                "The job finished without saving any reviewable file under OUTPUT_DIR."
            )
        _append_event(
            action_id,
            "Validating outputs",
            f"Found {len(files)} generated file(s).",
            started=started,
        )
        manifests = [
            {
                "relative_path": str(path.relative_to(staging)),
                "size_bytes": int(path.stat().st_size),
                "kind": _artifact_kind(path),
            }
            for path in files
        ]
        if any(item["size_bytes"] <= 0 for item in manifests):
            raise GeneralActionError("At least one generated output file is empty.")

        # The generated code has finished. The trusted publisher may now update
        # the approved final output directory inside this project.
        allowed_write_roots.extend([project_root, ASSISTANT_DATA_DIR / "database"])
        final.parent.mkdir(parents=True, exist_ok=True)
        moved_old = False
        if final.exists():
            os.replace(final, backup)
            moved_old = True
        try:
            os.replace(staging, final)
        except Exception:
            if moved_old and backup.exists() and not final.exists():
                os.replace(backup, final)
            raise

        report = {
            "action_id": action_id,
            "state": "completed",
            "title": plan.title,
            "user_request": plan.user_request,
            "started_at": plan.created_at,
            "completed_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "inputs": plan.inputs,
            "parameters": plan.parameters,
            "registered_tools": plan.registered_tools,
            "planner": {"provider": plan.planner_provider, "model": plan.planner_model},
            "output_dir": str(final),
            "requested_output_dir": plan.requested_output_dir or plan.output_dir,
            "backup_dir": str(backup) if backup.exists() else None,
            "files": manifests,
            "sandbox": {
                "filesystem_root": "read-only",
                "writable_project": plan.project_root,
                "source_editing": "blocked by read-only mount",
                "subprocess_and_network": "blocked for generated code",
            },
        }
        _atomic_json(directory / "report.json", report)
        _atomic_json(final / "run_report.json", report)
        _append_event(
            action_id,
            "Complete",
            f"AI job finished with {len(files)} reviewable file(s).",
            state="completed",
            started=started,
        )

        store = AssistantStore()
        for item in manifests[:500]:
            path = final / item["relative_path"]
            store.register_artifact(
                path,
                kind=item["kind"],
                caption=f"{plan.title}: {path.name}",
                source_page="assistant",
                source_command="general_spacemap_job",
                metadata={
                    "action_id": action_id,
                    "conversation_id": plan.conversation_id,
                    "output_dir": str(final),
                },
            )
        store.register_artifact(
            final / "run_report.json",
            kind="ai_job_report",
            caption=f"Run report: {plan.title}",
            source_page="assistant",
            source_command="general_spacemap_job",
            metadata={"action_id": action_id, "conversation_id": plan.conversation_id},
        )
        if plan.conversation_id and store.conversation_exists(plan.conversation_id):
            store.add_message(
                plan.conversation_id,
                "assistant",
                (
                    f"Background SPACEMAP job `{action_id[:8]}` finished with "
                    f"**{len(files)} reviewable file(s)**. Output folder: `{final}`"
                ),
                metadata={
                    "source": "background_action_completed",
                    "action_id": action_id,
                    "page_context": "assistant",
                },
            )
        parameter_summary = json.dumps(plan.parameters, ensure_ascii=False, default=str)
        store.add_memory_if_new(
            (
                f"Reusable SPACEMAP recipe: {plan.title}. Registered tools: "
                f"{', '.join(plan.registered_tools)}. Parameters: {parameter_summary[:1200]}. "
                "This job completed and produced non-empty outputs; scientific quality still needs "
                "the recorded QC or user review."
            ),
            scope="project",
            kind="recipe_note",
            status="proposed",
            confidence=0.8,
            evidence=[f"action:{action_id}", f"report:{final / 'run_report.json'}"],
            source="completed_protected_job",
        )
        store.log_activity(
            "general_action_completed",
            f"Protected AI job {action_id[:8]} completed.",
            details={"title": plan.title, "output_dir": str(final), "files": len(files)},
        )
        return 0
    except Exception as error:
        signal.alarm(0)
        traceback.print_exc()
        _append_event(
            action_id,
            "Failed",
            str(error),
            state="failed",
            started=started,
        )
        try:
            AssistantStore().log_activity(
                "general_action_failed",
                f"Protected AI job {action_id[:8]} failed.",
                level="ERROR",
                details={"error": str(error)},
            )
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one protected SPACEMAP AI job.")
    parser.add_argument("--action-id", required=True)
    arguments = parser.parse_args(argv)
    return run_general_action(arguments.action_id)


if __name__ == "__main__":
    sys.exit(main())
