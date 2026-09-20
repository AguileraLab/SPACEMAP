"""Fixed worker for checked SPACEMAP blur and bright-artifact experiments."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from spacemap_workflows.annotation import (
    run_blur_parameter_search,
    run_bright_parameter_search,
)

from .blur_action import action_directory, load_blur_plan
from .model_client import (
    LOCAL_PROVIDER,
    NVIDIA_MODEL_CATALOG,
    NVIDIA_PROVIDER,
    LocalModelClient,
    ModelUnavailable,
    NvidiaModelClient,
    load_model_settings,
)
from .online_access import build_online_preview_content
from .paths import require_assistant_write_path
from .preview_action import (
    PreviewActionError,
    generate_annotation_qc_previews,
    generate_scale_bar_previews,
)
from .registration_action import utc_now
from .storage import AssistantStore


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    target = require_assistant_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _automatic_visual_review(
    plan: Any,
    result: dict[str, Any],
    previews: list[dict[str, Any]],
    update: Any,
) -> dict[str, Any]:
    """Ask the selected visual model to assess only the generated QC PNGs."""
    if not plan.automatic_visual_review:
        return {"state": "not_requested", "answer": ""}
    if not previews:
        return {
            "state": "not_available",
            "answer": "",
            "error": "No local candidate previews were created.",
        }
    provider = str(plan.provider or LOCAL_PROVIDER)
    model_id = str(plan.model_id or "")
    if provider == NVIDIA_PROVIDER and not plan.online_visual_review_approved:
        return {
            "state": "waiting_for_online_approval",
            "answer": "",
            "error": (
                "The parameter tests are complete, but this run was not approved to send its small "
                "QC PNGs to NVIDIA. The source TIFF was not sent."
            ),
        }
    if provider == NVIDIA_PROVIDER and not bool(
        (NVIDIA_MODEL_CATALOG.get(model_id) or {}).get("supports_images")
    ):
        return {
            "state": "model_has_no_vision",
            "answer": "",
            "error": f"{model_id or 'The selected NVIDIA model'} does not accept images.",
        }

    selected_name = str(result.get("selected_trial") or "")
    selected = next(
        (
            row
            for row in (result.get("comparison") or [])
            if str(row.get("trial") or "") == selected_name
        ),
        {},
    )
    if str(plan.artifact_kind) == "bright":
        comparison_keys = (
            "trial",
            "polygon_count",
            "bright_percent",
            "brightness_percentile",
            "texture_percentile",
            "stability_iou",
            "machine_screen_score",
        )
        review_rules = (
            "A true artifact must be unusually bright and smooth or structureless. Organized cells, "
            "glands, nuclei, tissue-shaped signal, or coherent marker patterns indicate a false positive."
        )
        final_request = (
            "End with the exact Brightness percentile and Texture percentile you recommend from the "
            "tested settings."
        )
        artifact_name = "bright glare/debris"
    else:
        comparison_keys = (
            "trial",
            "polygon_count",
            "blurry_percent",
            "sensitivity_level",
            "evidence_rule",
            "separation_gap",
            "stability_iou",
            "machine_screen_score",
        )
        review_rules = (
            "True blur shows a real loss of fine detail relative to nearby tissue in H&E and supported "
            "CODEX markers. Empty, dark, unstained, folded, or edge regions alone are false-positive evidence."
        )
        final_request = "End with the exact tested blur settings you recommend."
        artifact_name = "blur"
    short_table = [
        {key: row.get(key) for key in comparison_keys}
        for row in (result.get("comparison") or [])[:10]
    ]
    instruction = (
        f"Review SPACEMAP's provisional {artifact_name} polygons now. {review_rules} The attached "
        "images contain the same candidates in H&E/brightfield-like and CODEX views; red outlines "
        "mark detections. For each numbered candidate, write one verdict: likely true artifact, "
        "likely false positive, or uncertain, followed by one short visual reason. Then say whether "
        f"the provisional setting is acceptable, too permissive, or too strict. {final_request} "
        "Do not propose another plan, do not ask a question, and do not claim that thumbnail QC is "
        "final pathology approval.\n\n"
        f"Provisional trial: {selected_name}\n"
        f"Provisional statistics and settings: {json.dumps(selected, default=str)}\n"
        f"All trial summaries: {json.dumps(short_table, default=str)}"
    )
    review_paths = [str(item.get("preview_path") or "") for item in previews[:8]]
    update("AI visual review", f"Sending {len(review_paths)} small QC preview(s) to {model_id or provider}.")
    settings = load_model_settings()
    try:
        if provider == NVIDIA_PROVIDER:
            content, manifest = build_online_preview_content(
                review_paths,
                instruction,
                model_supports_images=True,
            )
            client: LocalModelClient | NvidiaModelClient = NvidiaModelClient(
                model_id=model_id,
                timeout_seconds=settings.timeout_seconds,
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the visual QC reviewer inside SPACEMAP. Judge only the supplied "
                        "small previews and the recorded measurements."
                    ),
                },
                {"role": "user", "content": content},
            ]
            answer = client.complete(
                messages,
                temperature=0.0,
                max_tokens=max(1024, settings.max_tokens),
                enable_thinking=False,
            )
        else:
            blocks: list[dict[str, Any]] = []
            manifest = []
            for index, raw_path in enumerate(review_paths, start=1):
                path = Path(raw_path)
                data = path.read_bytes()
                if len(data) > 20 * 1024 * 1024:
                    raise ValueError(f"QC preview {path.name} is larger than 20 MB.")
                blocks.append(
                    {
                        "type": "text",
                        "text": (
                            f"Image {index}: {str(previews[index - 1].get('qc_modality') or 'image').upper()} "
                            f"{previews[index - 1].get('qc_kind') or 'preview'}"
                        ),
                    }
                )
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")
                        },
                    }
                )
                manifest.append({"path": str(path), "kind": "generated_qc_preview"})
            blocks.append({"type": "text", "text": instruction})
            client = LocalModelClient(timeout_seconds=settings.timeout_seconds)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the visual QC reviewer inside SPACEMAP. Judge only the supplied "
                        "small previews and the recorded measurements."
                    ),
                },
                {"role": "user", "content": blocks},
            ]
            answer = client.complete(
                messages,
                temperature=0.0,
                max_tokens=max(1024, settings.max_tokens),
            )
        if not answer.strip():
            update("AI visual review", "The model returned no text; retrying the same review once.")
            if provider == NVIDIA_PROVIDER:
                answer = client.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=max(1024, settings.max_tokens),
                    enable_thinking=False,
                )
            else:
                answer = client.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=max(1024, settings.max_tokens),
                )
        if not answer.strip():
            return {
                "state": "empty_model_answer",
                "provider": provider,
                "model": model_id,
                "answer": "",
                "evidence": manifest,
                "error": "The selected visual model returned no text.",
            }
        return {
            "state": "completed",
            "provider": provider,
            "model": model_id,
            "answer": answer.strip(),
            "evidence": manifest,
        }
    except (ModelUnavailable, OSError, TypeError, ValueError) as error:
        return {
            "state": "failed",
            "provider": provider,
            "model": model_id,
            "answer": "",
            "error": f"{type(error).__name__}: {error}",
        }


def run(action_id: str) -> None:
    plan = load_blur_plan(action_id, verify_inputs=True)
    directory = action_directory(action_id)
    started = time.monotonic()

    def update(stage: str, message: str) -> None:
        elapsed = round(time.monotonic() - started, 3)
        event = {
            "action_id": action_id,
            "time": utc_now(),
            "created_at": utc_now(),
            "stage": str(stage),
            "message": str(message),
            "elapsed_seconds": elapsed,
        }
        with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
        _atomic_json(
            directory / "status.json",
            {
                **event,
                "state": "running",
                "created_at": plan.created_at,
                "updated_at": event["time"],
            },
        )
        print(f"[{event['time']}] [{stage}] {message}", flush=True)

    try:
        artifact_kind = plan.artifact_kind
        artifact_label = "bright glare/debris" if artifact_kind == "bright" else "blur"
        update("Starting", "Loading the checked SPACEMAP Annotation settings.")
        runner = run_bright_parameter_search if artifact_kind == "bright" else run_blur_parameter_search
        result = runner(
            plan.image_path,
            plan.tissue_boundary_path,
            plan.run_output_dir,
            project_config_path=plan.project_config_path or None,
            microns_per_pixel=plan.microns_per_pixel,
            maximum_trials=plan.maximum_trials,
            progress=update,
        )
        next_state = str(result.get("next_state") or "")
        review_previews: list[dict[str, Any]] = []
        preview_errors: list[str] = []
        candidate_path = str((result.get("outputs") or {}).get("provisional_geojson") or "")
        if next_state == "visual_review_required" and candidate_path:
            update(
                "Preparing visual review",
                "Building local H&E and CODEX diagnostic views for the candidate polygons.",
            )
            preview_root = Path(plan.run_output_dir) / "visual_review_previews"
            for mode in ("he", "codex"):
                try:
                    bundle = generate_annotation_qc_previews(
                        plan.image_path,
                        [candidate_path],
                        preferred_mode=mode,
                        overview_size=1024,
                        tile_size=768,
                        max_tiles=3,
                        output_dir=preview_root / mode,
                    )
                    previews = list(bundle.get("previews") or [])
                    if plan.include_scale_bars:
                        previews = generate_scale_bar_previews(
                            previews,
                            mpp=plan.microns_per_pixel,
                            position="bottom-left",
                        )
                    for preview in previews:
                        review_previews.append({**preview, "qc_modality": mode})
                except (PreviewActionError, OSError, TypeError, ValueError) as error:
                    preview_errors.append(f"{mode}: {error}")
            result["review_previews"] = review_previews
            result["review_preview_errors"] = preview_errors
            review_manifest = Path(plan.run_output_dir) / "visual_review_preview_manifest.json"
            _atomic_json(
                review_manifest,
                {
                    "action_id": action_id,
                    "candidate_geojson": candidate_path,
                    "previews": review_previews,
                    "errors": preview_errors,
                    "status": "ready_for_model_or_human_review" if review_previews else "preview_failed",
                },
            )
            result.setdefault("outputs", {})["review_preview_manifest"] = str(review_manifest)
        visual_review = {"state": "not_available", "answer": ""}
        if next_state == "visual_review_required":
            visual_review = _automatic_visual_review(
                plan,
                result,
                review_previews,
                update,
            )
            result["visual_review"] = visual_review
            if visual_review.get("state") == "completed":
                update("AI visual review", "The selected model completed the candidate assessment.")
            else:
                update(
                    "AI visual review",
                    str(visual_review.get("error") or "The visual assessment was not completed."),
                )
        report = {
            "schema": "spacemap.artifact_action.report.v1",
            "action_id": action_id,
            "artifact_kind": artifact_kind,
            "state": "completed",
            "completed_at": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "result": result,
            "scientific_status": next_state or "unknown",
        }
        _atomic_json(directory / "report.json", report)
        store = AssistantStore()
        for kind, path_key, caption in (
            (f"{artifact_kind}_comparison", "comparison_csv", f"{artifact_label.title()} parameter comparison"),
            (f"{artifact_kind}_candidate", "provisional_geojson", f"Provisional {artifact_label} polygons"),
            (f"{artifact_kind}_run_manifest", "manifest_path", f"{artifact_label.title()} experiment manifest"),
        ):
            value = result.get("outputs", {}).get(path_key) or result.get(path_key)
            if value and Path(str(value)).is_file():
                store.register_artifact(
                    str(value),
                    kind=kind,
                    caption=caption,
                    source_page="tab4",
                    source_command=f"annotation.{artifact_kind}_optimize",
                    metadata={
                        "action_id": action_id,
                        "conversation_id": plan.conversation_id,
                        "model": plan.model_id,
                        "provider": plan.provider,
                    },
                )
        for preview in review_previews:
            store.register_artifact(
                preview["preview_path"],
                kind="assistant_image_preview",
                caption=(
                    f"{artifact_label.title()} QC {preview.get('qc_modality', 'image')} · "
                    f"{preview.get('qc_kind', 'preview')}"
                ),
                source_page="tab4",
                source_command=plan.user_request,
                metadata={
                    **preview,
                    "action_id": action_id,
                    "conversation_id": plan.conversation_id,
                    "request_message_id": plan.request_message_id or f"action:{action_id}",
                    "annotation_kind": artifact_kind,
                },
            )
        elapsed = report["elapsed_seconds"]
        trial_count = len(result.get("comparison") or [])
        if next_state == "no_candidates_detected":
            message = (
                f"Tested {trial_count} configurations; all returned zero {artifact_label} polygons. "
                "No best setting was selected and no visual review was performed."
            )
        else:
            message = (
                f"Tested {trial_count} configurations. The best machine-screened candidate and "
                f"{len(review_previews)} local QC preview(s) are ready."
            )
            if visual_review.get("state") == "completed":
                message += " The selected AI completed its visual assessment."
            else:
                message += " The AI visual assessment still needs attention."
        selected = next(
            (
                item
                for item in (result.get("comparison") or [])
                if item.get("trial") == result.get("selected_trial")
            ),
            {},
        )
        if next_state != "no_candidates_detected":
            store.add_memory_if_new(
                (
                    f"Reusable {artifact_label}-detection experiment result: provisional trial "
                    f"{result.get('selected_trial')} ranked highest after {trial_count} "
                    f"tests. Settings: {json.dumps(selected, ensure_ascii=False, default=str)[:1200]}. "
                    "This remains provisional until its polygons receive visual review."
                ),
                scope="project",
                kind="recipe_note",
                status="proposed",
                confidence=0.7,
                evidence=[f"action:{action_id}", f"manifest:{result.get('manifest_path', '')}"],
                source=f"completed_{artifact_kind}_experiment",
            )
        _atomic_json(
            directory / "status.json",
            {
                "action_id": action_id,
                "state": "completed",
                "stage": "Parameter search complete",
                "message": message,
                "created_at": plan.created_at,
                "updated_at": utc_now(),
                "elapsed_seconds": elapsed,
                "output_dir": plan.run_output_dir,
                "next_state": next_state,
                "review_preview_count": len(review_previews),
            },
        )
        if plan.conversation_id and store.conversation_exists(plan.conversation_id):
            if next_state == "no_candidates_detected":
                chat_answer = (
                    f"Background {artifact_label} job `{action_id[:8]}` finished its {trial_count} tests. "
                    "Every test returned 0 polygons, so I did not select a best setting or claim a "
                    f"visual review. Run folder: `{plan.run_output_dir}`"
                )
            else:
                chat_answer = (
                    f"Background {artifact_label} job `{action_id[:8]}` finished {trial_count} tests. The "
                    f"provisional setting is **{result.get('selected_trial')}**. I prepared "
                    f"{len(review_previews)} local H&E/CODEX review image(s)."
                )
                if visual_review.get("state") == "completed":
                    chat_answer += (
                        "\n\n**AI visual assessment**\n\n"
                        + str(visual_review.get("answer") or "")
                        + "\n\nThis remains thumbnail-level QC; approve or correct it before treating the GeoJSON as final."
                    )
                else:
                    chat_answer += (
                        "\n\nThe visual model did not complete the assessment: `"
                        + str(visual_review.get("error") or visual_review.get("state") or "unknown")
                        + "` The parameter results and review images are still available in this chat."
                    )
            store.add_message(
                plan.conversation_id,
                "assistant",
                chat_answer,
                metadata={
                    "source": "background_action_completed",
                    "action_id": action_id,
                    "page_context": "tab4",
                },
            )
            store.log_activity(
                "background_action_chat_updated",
                f"Posted completion for {artifact_label} action {action_id[:8]} to its chat.",
                details={"action_id": action_id, "conversation_id": plan.conversation_id},
            )
        print(f"[COMPLETE] {message}", flush=True)
    except BaseException as error:
        elapsed = round(time.monotonic() - started, 3)
        _atomic_json(
            directory / "status.json",
            {
                "action_id": action_id,
                "state": "failed",
                "stage": f"{plan.artifact_kind.title()} experiment failed",
                "message": f"{type(error).__name__}: {error}",
                "created_at": plan.created_at,
                "updated_at": utc_now(),
                "elapsed_seconds": elapsed,
            },
        )
        print(f"[FAILED] {type(error).__name__}: {error}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-id", required=True)
    arguments = parser.parse_args()
    run(arguments.action_id)


if __name__ == "__main__":
    main()
