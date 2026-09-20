"""Background worker for one approved ROI extraction plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .paths import require_assistant_write_path
from .registration_action import _file_signature, utc_now
from .roi_action import (
    ROIActionError,
    _atomic_json,
    _plan_digest,
    action_directory,
    load_roi_plan,
)
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


def _create_qc_preview(plan: Any, qc_path: Path) -> None:
    from PIL import Image, ImageDraw
    from spacemap.background_subtraction_utils import load_preview_channels_compatible

    loaded, full_shape, is_rgb = load_preview_channels_compatible(
        plan.image_path, [0], max_dim=1200
    )
    if is_rgb:
        rgb = np.asarray(loaded["rgb"])[..., :3]
        rgb = np.stack([_normalize_preview(rgb[..., index]) for index in range(3)], axis=-1)
    else:
        plane = _normalize_preview(loaded[0])
        rgb = np.stack([plane, plane, plane], axis=-1)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    preview_height, preview_width = rgb.shape[:2]
    full_height, full_width = full_shape
    scale_x = preview_width / max(1, full_width)
    scale_y = preview_height / max(1, full_height)
    colors = ["#00FF66", "#FFCC00", "#00CCFF", "#FF55DD", "#FF6644"]
    for index, roi in enumerate(plan.rois):
        x1, y1, x2, y2 = roi["bbox"]
        box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), str(roi["label"]), fill=color)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(qc_path)


def _rollback_publication(published: list[Path], restored: list[tuple[Path, Path]]) -> None:
    for final in reversed(published):
        final.unlink(missing_ok=True)
    for backup, final in reversed(restored):
        if backup.exists() and not final.exists():
            os.link(backup, final)


def run_roi_action(action_id: str) -> int:
    started = time.monotonic()
    directory = action_directory(action_id)
    try:
        plan = load_roi_plan(action_id, verify_inputs=True)
        try:
            approval = json.loads((directory / "approval.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ROIActionError("This ROI action has no valid approval record.") from error
        if approval.get("action_id") != action_id or approval.get("plan_sha256") != _plan_digest(action_id):
            raise ROIActionError("The ROI plan does not match the approved plan.")

        output_dir = require_assistant_write_path(plan.output_dir)
        staging_dir = require_assistant_write_path(plan.staging_dir)
        qc_path = require_assistant_write_path(plan.qc_preview)
        if staging_dir.exists() and any(staging_dir.iterdir()):
            raise ROIActionError("This action already has temporary ROI files and will not reuse them.")
        output_dir.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        for roi in plan.rois:
            final = require_assistant_write_path(roi["output_path"])
            backup = require_assistant_write_path(roi["backup_path"])
            expected = roi["existing_output_signature"]
            if final.exists() and not plan.overwrite:
                raise ROIActionError("An ROI output exists. This plan did not approve replacement.")
            if plan.overwrite and expected is not None:
                if not final.exists() or _file_signature(final) != expected:
                    raise ROIActionError("An existing ROI output changed after approval.")
                if backup.exists():
                    raise ROIActionError("A planned ROI backup name already exists.")
            elif plan.overwrite and final.exists():
                raise ROIActionError("An unreviewed ROI output appeared after approval.")

        _append_event(
            action_id,
            "Starting",
            f"Extracting {len(plan.rois)} ROI(s) with {len(plan.channels)} channel(s).",
            started=started,
        )
        from spacemap.registration import process_single_image_rois

        last_progress = -1

        def progress(completed: int, total: int) -> None:
            nonlocal last_progress
            if completed == last_progress:
                return
            last_progress = completed
            _append_event(
                action_id,
                "Extracting ROIs",
                f"Completed {completed} of {total} ROI crops.",
                started=started,
            )

        process_single_image_rois(
            plan.image_path,
            plan.boundaries_path,
            str(staging_dir),
            {"codex": plan.mpp},
            progress_callback=progress,
            channel_config={"keep": plan.channels, "rename": {}, "colors": {}},
        )

        _append_event(action_id, "Validation", "Checking temporary ROI outputs.", started=started)
        from spacemap.background_subtraction_utils import get_channel_info

        staged: list[tuple[Path, Path, Path, dict[str, Any]]] = []
        for roi in plan.rois:
            final = require_assistant_write_path(roi["output_path"])
            temporary = staging_dir / final.name
            backup = require_assistant_write_path(roi["backup_path"])
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError(f"SPACEMAP did not produce ROI output: {temporary.name}")
            channel_names, _colors, _metadata, _mpp = get_channel_info(str(temporary))
            if len(channel_names) != len(plan.channels):
                raise RuntimeError(
                    f"{temporary.name} has {len(channel_names)} channels; {len(plan.channels)} were expected."
                )
            staged.append((temporary, final, backup, roi))

        _append_event(action_id, "Publishing", "Moving validated ROI files into the final folder.", started=started)
        published: list[Path] = []
        restored: list[tuple[Path, Path]] = []
        try:
            for temporary, final, backup, _roi in staged:
                if final.exists():
                    os.link(final, backup)
                    final.unlink()
                    restored.append((backup, final))
                os.link(temporary, final)
                published.append(final)
            for temporary, _final, _backup, _roi in staged:
                temporary.unlink()
        except Exception:
            _rollback_publication(published, restored)
            raise

        qc_error = None
        try:
            _append_event(action_id, "QC", "Creating an ROI boundary preview.", started=started)
            _create_qc_preview(plan, qc_path)
        except Exception as error:
            qc_error = str(error)
            print(f"QC preview warning: {error}", flush=True)

        outputs = [
            {
                "label": roi["label"],
                "path": str(final),
                "size_bytes": final.stat().st_size,
                "bbox": roi["bbox"],
                "backup": str(backup) if backup.exists() else None,
            }
            for _temporary, final, backup, roi in staged
        ]
        report = {
            "action_id": action_id,
            "state": "completed",
            "started_at": plan.created_at,
            "completed_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "input_image": plan.image_path,
            "boundaries": plan.boundaries_path,
            "output_dir": plan.output_dir,
            "roi_count": len(outputs),
            "channel_count": len(plan.channels),
            "channels": plan.channels,
            "mpp": plan.mpp,
            "outputs": outputs,
            "qc": {
                "preview_path": str(qc_path) if qc_path.is_file() else None,
                "warning": qc_error,
                "note": "The preview is a quick boundary check. A person must still review the cropped ROIs.",
            },
        }
        _atomic_json(directory / "report.json", report)
        _atomic_json(output_dir / "_spacemap_ai_actions" / action_id / "run_report.json", report)
        _append_event(
            action_id,
            "Complete",
            f"ROI extraction finished. {len(outputs)} output file(s) need human QC.",
            state="completed",
            started=started,
        )

        store = AssistantStore()
        for output in outputs:
            store.register_artifact(
                Path(output["path"]),
                kind="roi_ome_tiff",
                caption=f"Extracted ROI: {output['label']}",
                source_page="tab3",
                source_command="roi.extract",
                metadata={
                    "action_id": action_id,
                    "conversation_id": plan.conversation_id,
                    "bbox": output["bbox"],
                },
            )
        if qc_path.is_file():
            store.register_artifact(
                qc_path,
                kind="roi_extraction_qc_image",
                caption=f"ROI boundary preview for {Path(plan.image_path).name}",
                source_page="tab3",
                source_command="roi.extract",
                metadata={
                    "action_id": action_id,
                    "conversation_id": plan.conversation_id,
                    "roi_count": len(outputs),
                },
            )
        if plan.conversation_id and store.conversation_exists(plan.conversation_id):
            store.add_message(
                plan.conversation_id,
                "assistant",
                (
                    f"Background ROI extraction `{action_id[:8]}` finished with "
                    f"**{len(outputs)} output image(s)**. Output folder: `{plan.output_dir}`"
                ),
                metadata={
                    "source": "background_action_completed",
                    "action_id": action_id,
                    "page_context": "tab3",
                },
            )
        store.log_activity(
            "roi_extraction_completed",
            f"Approved ROI extraction {action_id[:8]} completed.",
            details={"output_dir": plan.output_dir, "roi_count": len(outputs)},
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
                "roi_extraction_failed",
                f"ROI extraction {action_id[:8]} failed.",
                level="ERROR",
                details={"error": str(error)},
            )
        except Exception:
            pass
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one approved SPACEMAP AI ROI action.")
    parser.add_argument("--action-id", required=True)
    arguments = parser.parse_args(argv)
    return run_roi_action(arguments.action_id)


if __name__ == "__main__":
    sys.exit(main())
