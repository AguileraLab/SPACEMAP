"""Background worker for one approved CODEX/H&E registration plan."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .paths import require_assistant_write_path
from .registration_action import (
    RegistrationActionError,
    _atomic_json,
    _file_signature,
    _plan_digest,
    action_directory,
    load_registration_plan,
    utc_now,
)
from .storage import AssistantStore


def _append_event(action_id: str, stage: str, message: str, *, state: str = "running", started: float) -> None:
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


def _normalize_preview(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D preview plane, got {data.shape}.")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def _create_qc_preview(plan: Any, registered_he: Path, qc_path: Path) -> dict[str, Any]:
    import cv2
    from PIL import Image, ImageDraw
    from spacemap.background_subtraction_utils import load_preview_channels_compatible

    codex_data, _shape, _rgb = load_preview_channels_compatible(
        plan.codex_path, [plan.reference_channel_index], max_dim=1100
    )
    dapi = _normalize_preview(codex_data[plan.reference_channel_index])

    he_data, _he_shape, he_is_rgb = load_preview_channels_compatible(
        str(registered_he), [0, 1, 2], max_dim=1100
    )
    if he_is_rgb:
        he_rgb = np.asarray(he_data["rgb"])[..., :3]
        he_rgb = np.stack([_normalize_preview(he_rgb[..., idx]) for idx in range(3)], axis=-1)
    else:
        planes = [_normalize_preview(he_data[index]) for index in sorted(he_data)[:3]]
        while len(planes) < 3:
            planes.append(planes[-1])
        he_rgb = np.stack(planes[:3], axis=-1)
    if he_rgb.shape[:2] != dapi.shape:
        he_rgb = cv2.resize(he_rgb, (dapi.shape[1], dapi.shape[0]), interpolation=cv2.INTER_AREA)

    he_gray = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2GRAY)
    dapi_edge = cv2.Sobel(dapi, cv2.CV_32F, 1, 1, ksize=3)
    he_edge = cv2.Sobel(he_gray, cv2.CV_32F, 1, 1, ksize=3)
    if float(np.std(dapi_edge)) > 0 and float(np.std(he_edge)) > 0:
        edge_correlation = float(np.corrcoef(dapi_edge.ravel(), he_edge.ravel())[0, 1])
    else:
        edge_correlation = 0.0

    dapi_rgb = np.stack([dapi, dapi, dapi], axis=-1)
    overlay = np.zeros_like(he_rgb)
    overlay[..., 0] = he_gray
    overlay[..., 1] = dapi
    overlay[..., 2] = he_gray
    panels = np.concatenate([dapi_rgb, he_rgb, overlay], axis=1)
    title_height = 38
    canvas = Image.new("RGB", (panels.shape[1], panels.shape[0] + title_height), "white")
    canvas.paste(Image.fromarray(panels), (0, title_height))
    draw = ImageDraw.Draw(canvas)
    panel_width = dapi.shape[1]
    draw.text((10, 11), f"CODEX {plan.reference_channel}", fill="black")
    draw.text((panel_width + 10, 11), "Registered H&E", fill="black")
    draw.text((panel_width * 2 + 10, 11), "Overlay (green=CODEX, magenta=H&E)", fill="black")
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(qc_path)
    return {
        "preview_path": str(qc_path),
        "edge_correlation": round(edge_correlation, 5),
        "metric_note": "Descriptive edge correlation only; a person must still review the overlay.",
    }


def run_registration_action(action_id: str) -> int:
    started = time.monotonic()
    directory = action_directory(action_id)
    try:
        plan = load_registration_plan(action_id, verify_inputs=True)
        try:
            approval = json.loads((directory / "approval.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RegistrationActionError("This action has no valid user approval record.") from error
        if approval.get("action_id") != action_id or approval.get("plan_sha256") != _plan_digest(action_id):
            raise RegistrationActionError("The plan does not match the user-approved plan.")
        output_dir = require_assistant_write_path(plan.output_dir)
        intermediate_dir = require_assistant_write_path(plan.intermediate_dir)
        partial_output = require_assistant_write_path(plan.partial_output)
        final_output = require_assistant_write_path(plan.final_output)
        backup_output = require_assistant_write_path(plan.backup_output)
        if final_output.exists() and not plan.overwrite:
            raise RegistrationActionError("The final output exists. The worker will not overwrite it.")
        if plan.overwrite:
            if plan.existing_output_signature is None and final_output.exists():
                raise RegistrationActionError("An unreviewed output appeared after planning.")
            if plan.existing_output_signature is not None:
                if not final_output.exists() or _file_signature(final_output) != plan.existing_output_signature:
                    raise RegistrationActionError("The existing output changed after approval.")
            if backup_output.exists():
                raise RegistrationActionError("The planned backup output already exists.")
        if partial_output.exists():
            raise RegistrationActionError("This action already has a partial output and will not overwrite it.")

        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / "plan.json", intermediate_dir.parent / "approved_plan.json")

        _append_event(action_id, "Starting", "Loading the reviewed SPACEMAP registration functions.", started=started)
        from spacemap.background_subtraction_utils import get_channel_info, get_image_mpp
        from spacemap.registration import (
            extract_dapi_channel,
            merge_multi_round_ome_tiff,
            preprocess_he_image,
            register_images,
        )

        _append_event(action_id, "1/4 H&E preparation", "Resizing H&E to the CODEX pixel size.", started=started)
        he_resized = Path(
            preprocess_he_image(
                plan.he_path,
                str(intermediate_dir),
                {"hne": plan.he_mpp, "codex": plan.codex_mpp},
                overwrite=False,
                pre_rotation_deg=plan.pre_rotation_deg,
            )
        )
        if not he_resized.exists():
            raise RuntimeError("H&E preparation did not produce an image.")

        _append_event(action_id, "2/4 Reference channel", f"Using CODEX channel {plan.reference_channel}.", started=started)
        reference_path = Path(
            extract_dapi_channel(
                plan.codex_path,
                str(intermediate_dir),
                overwrite=False,
                reference_channel=plan.reference_channel,
                reference_channel_index=plan.reference_channel_index,
            )
        )
        if not reference_path.exists():
            raise RuntimeError("Reference-channel extraction did not produce an image.")

        _append_event(action_id, "3/4 Registration", "Aligning H&E to CODEX.", started=started)
        registered_he = Path(
            register_images(
                str(he_resized),
                str(reference_path),
                str(intermediate_dir),
                {"hne": plan.codex_mpp, "codex": plan.codex_mpp},
                overwrite=False,
                reference_channel=plan.reference_channel if reference_path.resolve() == Path(plan.codex_path).resolve() else None,
                reference_channel_index=plan.reference_channel_index if reference_path.resolve() == Path(plan.codex_path).resolve() else 0,
                n_keypoints=plan.n_keypoints,
            )
        )
        if not registered_he.exists():
            raise RuntimeError("Registration did not produce the registered H&E image.")

        _append_event(action_id, "4/4 Merge", "Writing CODEX plus registered H&E to a temporary final stack.", started=started)
        merge_multi_round_ome_tiff(
            [plan.codex_path, str(registered_he)],
            str(partial_output),
            plan.codex_mpp,
            channel_configs=None,
            overwrite=False,
        )
        if not partial_output.exists() or partial_output.stat().st_size <= 0:
            raise RuntimeError("The merge did not produce a valid output file.")
        partial_channels, _partial_colors, _partial_metadata, _partial_mpp = get_channel_info(str(partial_output))
        expected_channels = len(plan.codex_channels) + 3
        if len(partial_channels) < expected_channels:
            raise RuntimeError(
                f"The merged output has {len(partial_channels)} channels; at least {expected_channels} were expected."
            )
        # Preserve an explicitly approved replacement so it remains recoverable.
        if final_output.exists():
            os.link(final_output, backup_output)
            final_output.unlink()
        try:
            # A hard link publishes without silently overwriting another process's file.
            os.link(partial_output, final_output)
        except Exception:
            if backup_output.exists() and not final_output.exists():
                os.link(backup_output, final_output)
            raise
        partial_output.unlink()

        _append_event(action_id, "QC", "Checking metadata and creating a review overlay.", started=started)
        channel_names, _colors, _metadata, mpp = get_channel_info(str(final_output))
        detected_mpp = get_image_mpp(str(final_output)).get("codex")
        qc_path = intermediate_dir.parent / "registration_qc.png"
        qc = _create_qc_preview(plan, registered_he, qc_path)
        report = {
            "action_id": action_id,
            "state": "completed",
            "started_at": plan.created_at,
            "completed_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "inputs": {"codex": plan.codex_path, "he": plan.he_path},
            "output": str(final_output),
            "output_size_bytes": final_output.stat().st_size,
            "channel_count": len(channel_names),
            "channels": channel_names,
            "mpp": detected_mpp if detected_mpp is not None else mpp,
            "reference_channel": plan.reference_channel,
            "registration_settings": {
                "n_keypoints": plan.n_keypoints,
                "pre_rotation_deg": plan.pre_rotation_deg,
                "overwrite": plan.overwrite,
            },
            "replaced_output_backup": str(backup_output) if backup_output.exists() else None,
            "qc": qc,
            "intermediate_dir": str(intermediate_dir),
        }
        _atomic_json(directory / "report.json", report)
        _atomic_json(intermediate_dir.parent / "run_report.json", report)
        _append_event(
            action_id,
            "Complete",
            f"Registration finished. Human QC review is still required: {final_output}",
            state="completed",
            started=started,
        )
        store = AssistantStore()
        store.register_artifact(
            final_output,
            kind="registered_ome_tiff",
            caption=f"Registered CODEX + H&E stack for {plan.prefix}",
            source_page="tab1",
            source_command="registration.codex_he_pair",
            metadata={
                "action_id": action_id,
                "conversation_id": plan.conversation_id,
                "qc_preview": str(qc_path),
            },
        )
        store.register_artifact(
            qc_path,
            kind="registration_qc_image",
            caption=f"Registration QC overlay for {plan.prefix}",
            source_page="tab1",
            source_command="registration.codex_he_pair",
            metadata={
                "action_id": action_id,
                "conversation_id": plan.conversation_id,
                "final_output": str(final_output),
            },
        )
        if plan.conversation_id and store.conversation_exists(plan.conversation_id):
            store.add_message(
                plan.conversation_id,
                "assistant",
                (
                    f"Background registration `{action_id[:8]}` finished. Output: "
                    f"`{final_output}`. The registration QC preview is available in this chat and "
                    "under Actions."
                ),
                metadata={
                    "source": "background_action_completed",
                    "action_id": action_id,
                    "page_context": "tab1",
                },
            )
        store.log_activity(
            "registration_completed",
            f"Approved registration {action_id[:8]} completed.",
            details={"output": str(final_output), "elapsed_seconds": report["elapsed_seconds"]},
        )
        return 0
    except Exception as error:
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
                "registration_failed",
                f"Approved registration {action_id[:8]} failed: {error}",
                level="error",
            )
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one approved SPACEMAP AI registration action.")
    parser.add_argument("--action-id", required=True)
    arguments = parser.parse_args(argv)
    return run_registration_action(arguments.action_id)


if __name__ == "__main__":
    sys.exit(main())
