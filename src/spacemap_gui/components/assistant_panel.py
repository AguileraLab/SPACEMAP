"""Full assistant workspace and compact app-wide chat panel."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import streamlit as st

from spacemap_assistant.cluster import (
    ClusterError,
    ResourceRequest,
    WORKLOAD_PROFILES,
    create_probe_script,
    query_my_jobs,
    query_partitions,
    submit_probe_script,
    validate_probe_script,
)
from spacemap_assistant.action_monitor import (
    action_details,
    action_file_progress,
    list_all_actions,
    progress_events,
)
from spacemap_assistant.context import build_page_context, build_workspace_context
from spacemap_assistant.code_index import source_context_for_prompt
from spacemap_assistant.cluster_labeling import prepare_cluster_labeling_evidence
from spacemap_assistant.general_action import (
    GeneralActionError,
    build_general_plan,
    launch_general_action,
    load_action_status as load_general_action_status,
    load_general_plan,
    parse_job_envelope,
)
from spacemap_assistant.blur_action import (
    BlurActionError,
    build_blur_plan,
    launch_blur_action,
    load_action_status as load_blur_action_status,
    load_blur_plan,
    parse_artifact_optimization_request,
)
from spacemap_assistant.instructions import (
    CORE_SAFETY_POLICY,
    build_system_prompt,
    load_instruction_profile,
    reset_instruction_profile,
    save_instruction_profile,
)
from spacemap_assistant.memory import suggest_memories
from spacemap_assistant.model_client import (
    ACTION_PLAN_MAX_TOKENS,
    COMPLEX_ACTION_PLAN_MAX_TOKENS,
    DEFAULT_NVIDIA_MODEL_ID,
    LOCAL_PROVIDER,
    MODEL_ID,
    NVIDIA_MODEL_CATALOG,
    NVIDIA_PROVIDER,
    LocalModelClient,
    ModelUnavailable,
    NvidiaModelClient,
    completion_budget,
    delete_nvidia_api_key,
    load_model_settings,
    load_provider_settings,
    nvidia_api_key_saved,
    save_nvidia_api_key,
    save_model_settings,
    save_provider_settings,
    test_nvidia_connection,
)
from spacemap_assistant.model_runtime import (
    ModelRuntimeError,
    runtime_status,
)
from spacemap_assistant.online_access import (
    build_online_evidence_content,
    build_online_preview_content,
    extract_prompt_paths,
    grant_online_access,
    list_online_access_grants,
    load_online_share_limits,
    plan_online_evidence,
    revoke_online_access,
    save_online_share_limits,
)
from spacemap_assistant.registration_action import (
    RegistrationActionError,
    build_registration_plan,
    launch_registration_action,
    load_action_status as load_registration_action_status,
    load_registration_plan,
    parse_registration_request,
)
from spacemap_assistant.roi_action import (
    ROIActionError,
    build_roi_plan,
    launch_roi_action,
    load_action_status as load_roi_action_status,
    load_roi_plan,
    parse_roi_extraction_request,
)
from spacemap_assistant.paths import (
    AI_ROOT,
    ASSISTANT_DATA_DIR,
    ASSISTANT_PROJECTS_DIR,
    AssistantPathError,
    grant_admin_access,
    list_admin_access_grants,
    require_assistant_read_path,
    require_assistant_write_path,
    revoke_admin_access,
)
from spacemap_assistant.preview_action import (
    HE_CHANNELS,
    PreviewActionError,
    generate_he_previews,
    generate_annotation_qc_previews,
    generate_marker_previews,
    generate_non_he_previews,
    generate_overlay_previews,
    generate_scale_bar_previews,
    is_multi_step_scientific_request,
    match_requested_channels,
    parse_image_preview_request,
    parse_visual_assessment_request,
    preview_channel_names,
)
from spacemap_assistant.planner_paths import online_action_content, restore_online_job_paths
from spacemap_assistant.storage import AssistantStore
from spacemap_gui.process_log import PROCESS_LOGS_KEY
from spacemap_gui.workflow_guidance import PIPELINE_GUIDES
from spacemap_gui.components.assistant_actions import (
    BLUR_ACTION_ID_KEY,
    GENERAL_ACTION_ID_KEY,
    LIVE_ACTION_KEY,
    REGISTRATION_ACTION_ID_KEY,
    ROI_ACTION_ID_KEY,
    preapproval_enabled,
    render_registration_actions,
    set_registration_draft,
    set_roi_draft,
)
from spacemap_workflows.registry import tool_manifest


CONVERSATION_KEY = "_spacemap_ai_conversation_id"
MAX_CHAT_MESSAGES = 40
CLUSTER_PARTITIONS_KEY = "_spacemap_ai_cluster_partitions"
CLUSTER_JOBS_KEY = "_spacemap_ai_cluster_jobs"
CLUSTER_TERMINAL_KEY = "_spacemap_ai_cluster_terminal"
CLUSTER_SCRIPT_KEY = "_spacemap_ai_cluster_script"
CHAT_UPLOAD_COUNTER_KEY = "_spacemap_ai_chat_upload_counter"
NVIDIA_SESSION_CONSENT_KEY = "_spacemap_ai_nvidia_session_consent"
PENDING_ONLINE_SHARE_KEY = "_spacemap_ai_pending_online_share"
GEMMA_NOTICE = (
    "Configured model: Google Gemma 4 12B. It runs locally and offline. "
    "Follow your own institution's policy for the data you place in it."
)

PROVIDER_LABELS = {
    LOCAL_PROVIDER: "Local Gemma (offline and private)",
    NVIDIA_PROVIDER: "NVIDIA hosted API (online prototype)",
}


def _store() -> AssistantStore:
    return AssistantStore()


def _conversation_id(store: AssistantStore) -> str:
    value = str(st.session_state.get(CONVERSATION_KEY) or "")
    if not value or not store.conversation_exists(value):
        value = store.create_conversation(
            project_root=str(st.session_state.get("active_project_root") or "")
        )
        st.session_state[CONVERSATION_KEY] = value
    return value


def _context_page(page_id: str) -> str:
    if page_id in {"assistant", "assistant_action"}:
        return str(st.session_state.get("_spacemap_last_pipeline_tab") or "tab0")
    return page_id


def _current_context(page_id: str) -> dict[str, Any]:
    target = _context_page(page_id)
    context = build_page_context(
        target,
        st.session_state,
        guide=PIPELINE_GUIDES.get(target),
    )
    context["workspace_state"] = build_workspace_context(st.session_state)
    if page_id == "assistant_action":
        action_id = str(st.session_state.get(LIVE_ACTION_KEY) or "")
        details = action_details(action_id) if action_id else {}
        plan = details.get("plan") or {}
        status = details.get("status") or {}
        if plan:
            context["selected_action"] = {
                "action_id": action_id,
                "action_type": plan.get("action_type"),
                "title": plan.get("title"),
                "user_request": plan.get("user_request"),
                "state": status.get("state"),
                "stage": status.get("stage"),
                "message": status.get("message"),
                "inputs": plan.get("inputs"),
                "parameters": plan.get("parameters"),
                "image_path": plan.get("image_path"),
                "boundaries_path": plan.get("boundaries_path") or plan.get("tissue_boundary_path"),
                "output": plan.get("run_output_dir") or plan.get("output_dir") or plan.get("final_output"),
            }
    return context


def _online_safe_context(context: dict[str, Any]) -> dict[str, Any]:
    """Return only generic workflow help; never include project state or paths."""
    return {
        "page_id": str(context.get("page_id") or ""),
        "page_title": str(context.get("page_title") or ""),
        "registered_sections": [str(item) for item in (context.get("registered_sections") or [])],
        "purpose": str(context.get("purpose") or ""),
        "before_run": str(context.get("before_run") or ""),
        "next_step": str(context.get("next_step") or ""),
    }


def _online_prompt_safety_issue(prompt: str) -> str | None:
    """Catch obvious private-data cases before a hosted request leaves the workstation."""
    text = str(prompt)
    sensitive_terms = (
        "medical record number",
        "patient name",
        "date of birth",
        "protected health information",
        "subject identifier",
    )
    if any(term in text.lower() for term in sensitive_terms):
        return "This message may contain protected or identifying information."
    return None


def _online_safe_thread_history(
    store: AssistantStore,
    conversation_id: str,
    current_message_id: str,
) -> list[dict[str, str]]:
    """Keep hosted chat continuity inside this thread while omitting paths and risky text."""
    result: list[dict[str, str]] = []
    remaining = 6_000
    messages = store.get_messages(conversation_id, limit=12)
    for item in reversed(messages):
        if str(item.get("id") or "") == str(current_message_id):
            continue
        role = str(item.get("role") or "")
        if role not in {"user", "assistant"}:
            continue
        source = str((item.get("metadata") or {}).get("source") or "")
        if role == "assistant" and source in {
            "he_previews_generated",
            "image_previews_generated",
            "image_preview_failed",
            "image_preview_needs_channels",
            "image_preview_needs_source",
        }:
            continue
        content = str(item.get("content") or "").replace("\\_", "_")
        if _online_prompt_safety_issue(content):
            continue
        content = re.sub(
            r"(?<![A-Za-z0-9])(/[^\s`\"']+)",
            "[LOCAL PATH OMITTED]",
            content,
        )
        content = _redact_log_text(content).strip()
        if not content:
            continue
        if result and len(content) > remaining:
            break
        result.append({"role": role, "content": content[:remaining]})
        remaining -= min(len(content), remaining)
        if remaining <= 0 or len(result) >= 8:
            break
    return list(reversed(result))


def _online_system_prompt(
    context: dict[str, Any],
    active_visual_context: str = "",
    *,
    registered_tools: str = "",
    allow_action_plan: bool = False,
) -> str:
    safe_context = json.dumps(_online_safe_context(context), ensure_ascii=False, indent=2)
    visual_text = (
        "\n\nActive visual state from this chat only (local paths omitted):\n"
        + active_visual_context
        if active_visual_context
        else ""
    )
    action_text = (
        "\n\nThe user asked for execution. You are the planner; local SPACEMAP is the executor. "
        "Use only a registered callable below. Return a short explanation plus exactly one complete "
        "<spacemap_job> JSON object using the same format described below. Local path aliases such as "
        "LOCAL_PATH_1 must be copied exactly into inputs and output_dir; never invent a local path. "
        "Code reads only INPUTS and PARAMETERS, writes only below OUTPUT_DIR, imports a registered "
        "SPACEMAP function, and contains no absolute path. Do not claim the work ran. Required JSON keys: "
        "title, inputs, parameters, code, output_dir, overwrite, max_runtime_seconds. If an essential "
        "scientific input is missing, ask one short question and do not emit a job.\n\n"
        "Registered tools relevant to this request:\n"
        + registered_tools
        if allow_action_plan
        else ""
    )
    advisory_text = (
        "You are advisory only: do not claim that you ran an analysis or changed SPACEMAP. "
        if not allow_action_plan
        else "You may plan a local action, but you cannot execute it yourself. "
    )
    return (
        "You are the online consulting model for SPACEMAP, a spatial pathology analysis UI. "
        "Use simple language and give the direct answer first. "
        + advisory_text
        + "When the current message includes Owner-approved "
        "image previews or table samples, analyze only that bounded evidence and say that it is a preview "
        "or sample, not the whole original file. Local registered SPACEMAP tools, not this hosted model, "
        "perform approved work. State uncertainty for scientific labels and recommendations.\n\n"
        "Generic workflow context (contains no project paths or project settings):\n"
        + safe_context
        + visual_text
        + action_text
    )


def _answer_is_waiting_for_input(answer: str) -> bool:
    lowered = str(answer).casefold()
    phrases = (
        "please provide",
        "i still need",
        "which file",
        "which column",
        "what output",
        "need the full path",
        "cannot choose without",
        "before i can prepare",
    )
    return any(phrase in lowered for phrase in phrases)


def _advisory_answer_needs_repair(answer: str) -> bool:
    """Reject empty placeholders and leaked model-internal tool thoughts."""
    text = str(answer).strip()
    if text.casefold() in {"", "none", "null", "n/a"}:
        return True
    leaked_call = re.search(
        r"(?im)^\s*(?:thought\s+call|tool\s+call|function\s+call)\s*:",
        text,
    )
    raw_tool_block = re.search(
        r"(?is)<(?:tool_call|function_call)>.*?</(?:tool_call|function_call)>",
        text,
    )
    return bool(leaked_call or raw_tool_block)


def _cluster_text_only_messages(
    messages: list[dict[str, Any]],
    prompt: str,
    evidence_note: str,
) -> list[dict[str, Any]]:
    """Replace only the newest user payload; preserve this chat's history."""
    result = [dict(item) for item in messages]
    replacement = (
        str(prompt)
        + str(evidence_note)
        + "\n\nThe visual endpoint was unavailable. Use the exact numeric marker and spatial "
        "evidence above. Clearly mark any conclusion that would still benefit from image review."
    )
    for index in range(len(result) - 1, -1, -1):
        if result[index].get("role") == "user":
            result[index]["content"] = replacement
            break
    return result


def _requested_context_page(prompt: str, current_page_id: str) -> str:
    """Use the workflow named by the user instead of an unrelated visible page."""
    text = str(prompt).lower()
    keyword_pages = (
        ("tab1", ("register", "registration", "align codex", "align h&e", "align he")),
        ("tab2", ("background subtract", "background detection", "remove background")),
        ("tab3", ("roi", "region of interest")),
        ("tab4", ("artifact", "pixel classification", "tissue detection", "annotation")),
        ("tab5", ("segment", "segmentation", "cellpose", "deepcell")),
        ("tab6", ("batch correction", "correct batch")),
        ("tab7", ("phenotyp", "leiden", "clustering", "manual reference")),
        ("tab8", ("spatial analysis", "neighborhood", "cell interaction", "triad", "clonality")),
        ("tab9", ("visualization", "heatmap", "umap", "figure")),
    )
    for page_id, keywords in keyword_pages:
        if any(keyword in text for keyword in keywords):
            return page_id
    return _context_page(current_page_id)


def _deterministic_quick_reply(prompt: str, context: dict[str, Any]) -> str | None:
    normalized = prompt.lower().strip(" ?.!")
    page = context.get("page_title", "this page")
    if context.get("page_id") == "tab1" and "register" in normalized and any(
        phrase in normalized for phrase in ("what do i need", "explain", "how do i")
    ):
        settings = context.get("current_settings") or {}
        configured = []
        for label, key in (
            ("CODEX folder", "batch_codex_dir"),
            ("H&E folder", "batch_he_dir"),
            ("output folder", "tab1_batch_output_dir"),
        ):
            value = settings.get(key)
            if value:
                configured.append(f"- {label}: `{value}`")
        configured_text = "\n\nCurrent registered settings:\n\n" + "\n".join(configured) if configured else ""
        return (
            "For one CODEX + H&E registration, you need: the CODEX image, the matching H&E image, "
            "correct MPP values for both, a nuclear reference channel such as DAPI, and an output folder. "
            "SPACEMAP prepares the H&E, selects/extracts the reference, aligns H&E to CODEX, then writes "
            "`<pair>_merged.ome.tiff`. Review the pair, MPP, orientation, reference channel, and final overlay."
            + configured_text
        )
    if normalized in {"what am i seeing", "explain this page"}:
        purpose = context.get("purpose") or "This page is part of the SPACEMAP workflow."
        sections = ", ".join(context.get("registered_sections") or [])
        response = f"You are viewing **{page}**. {purpose}"
        if sections:
            response += f"\n\nAvailable sections include: {sections}."
        return response
    if normalized in {"what should i do next", "what is next"}:
        before = context.get("before_run") or "Review the current inputs and warnings."
        next_step = context.get("next_step") or "Continue after the current page is complete."
        return f"Before running anything: {before}\n\nAfter this page: {next_step}"
    if normalized in {"check my current settings", "check my settings"}:
        settings = context.get("current_settings") or {}
        if not settings:
            return f"I do not see any simple saved settings for **{page}** yet."
        preview = list(settings.items())[:12]
        lines = "\n".join(f"- `{key}`: {value}" for key, value in preview)
        extra = len(settings) - len(preview)
        if extra > 0:
            lines += f"\n- …and {extra} more settings."
        return f"I can see these current **{page}** settings:\n\n{lines}"
    return None


def _request_with_recent_path_context(
    prompt: str,
    store: AssistantStore,
    conversation_id: str,
) -> str:
    """Join a path-only follow-up to the recent user request."""
    clean = str(prompt).strip()
    has_absolute_path = bool(re.search(r"(?<![A-Za-z0-9])/[\w./\\-]+", clean))
    looks_like_followup = has_absolute_path and (
        clean.startswith("/")
        or len(clean.split()) <= 12
        or any(
            phrase in clean.lower()
            for phrase in ("source image", "registered image", "use this image", "this file")
        )
    )
    if not looks_like_followup:
        return clean
    user_messages = [
        item["content"]
        for item in store.get_messages(conversation_id, limit=10)
        if item["role"] == "user"
    ]
    recent = user_messages[-8:]
    for index in range(len(recent) - 2, -1, -1):
        older = recent[index].lower()
        is_action_anchor = (
            "register" in older
            or (
                "roi" in older
                and any(word in older for word in ("extract", "crop", "create"))
            )
            or (
                "cluster" in older
                and any(
                    word in older
                    for word in ("marker", "expression", "heatmap", "pixel classification")
                )
                and any(
                    word in older
                    for word in ("label", "name", "identify", "phenotype", "annotate")
                )
            )
        )
        if is_action_anchor:
            return "\n".join(recent[index:])
    return clean


def _artifact_draft_with_thread_context(
    prompt: str,
    store: AssistantStore,
    conversation_id: str,
) -> dict[str, Any] | None:
    """Understand short artifact follow-ups using only this open chat thread."""
    clean = _request_with_recent_path_context(prompt, store, conversation_id)
    direct = parse_artifact_optimization_request(clean)
    if direct is not None:
        return direct
    lowered = str(prompt).casefold()
    followup_cues = (
        "brightfield",
        "false positive",
        "true positive",
        "detected region",
        "detected spot",
        "representative tile",
        "best setting",
        "best parameter",
        "marker that",
        "visual review",
    )
    if not any(cue in lowered for cue in followup_cues):
        return None
    messages = store.get_messages(conversation_id, limit=20)
    for message in reversed(messages[:-1]):
        if message.get("role") != "user":
            continue
        earlier = str(message.get("content") or "").strip()
        if not earlier:
            continue
        combined = f"{earlier}\n\nFollow-up instruction:\n{prompt}"
        draft = parse_artifact_optimization_request(combined)
        if draft is not None:
            return draft
    return None


def _is_plain_approval(prompt: str) -> bool:
    normalized = re.sub(r"[^a-z ]+", " ", str(prompt).lower())
    normalized = " ".join(normalized.split())
    return normalized in {
        "approve",
        "approved",
        "yes approve",
        "approve it",
        "go ahead",
        "start it",
        "run it",
    }


def _launch_chat_approved_action(
    prompt: str,
    store: AssistantStore,
    conversation_id: str,
) -> bool:
    """Launch the newest exact plan prepared in this Streamlit session."""
    if not _is_plain_approval(prompt):
        return False
    candidates: list[tuple[str, str, str]] = []
    action_specs = (
        (
            "blur experiment",
            BLUR_ACTION_ID_KEY,
            load_blur_action_status,
            load_blur_plan,
        ),
        (
            "roi",
            ROI_ACTION_ID_KEY,
            load_roi_action_status,
            load_roi_plan,
        ),
        (
            "registration",
            REGISTRATION_ACTION_ID_KEY,
            load_registration_action_status,
            load_registration_plan,
        ),
        (
            "general",
            GENERAL_ACTION_ID_KEY,
            load_general_action_status,
            load_general_plan,
        ),
    )
    for action_type, state_key, status_loader, plan_loader in action_specs:
        action_id = str(st.session_state.get(state_key) or "")
        if not action_id:
            continue
        try:
            if status_loader(action_id).get("state") == "planned":
                candidates.append((str(plan_loader(action_id).created_at), action_type, action_id))
        except (
            BlurActionError,
            GeneralActionError,
            ROIActionError,
            RegistrationActionError,
            OSError,
            ValueError,
        ):
            continue
    if not candidates:
        answer = (
            "I do not have a checked plan waiting in this app session. Please give me the input and "
            "output paths again, or open **Actions** and select a saved waiting plan."
        )
        store.add_message(conversation_id, "assistant", answer, metadata={"source": "approval_missing"})
        return True

    _created_at, action_type, action_id = max(candidates)
    try:
        if action_type == "blur experiment":
            pid = launch_blur_action(action_id, approval_method="User typed approve in chat")
        elif action_type == "roi":
            pid = launch_roi_action(action_id, approval_method="User typed approve in chat")
        elif action_type == "registration":
            pid = launch_registration_action(action_id)
        else:
            pid = launch_general_action(action_id, approval_method="User typed approve in chat")
        st.session_state[LIVE_ACTION_KEY] = action_id
        answer = (
            f"Approved. I started the checked {action_type} job `{action_id[:8]}` as process {pid}. "
            "Open **Actions** to watch its progress, previews, and results."
        )
        store.log_activity(
            "chat_action_approved",
            f"User approved {action_type} action {action_id[:8]} in chat.",
            details={"action_id": action_id, "pid": pid},
        )
    except (
        BlurActionError,
        GeneralActionError,
        ROIActionError,
        RegistrationActionError,
        OSError,
        ValueError,
    ) as error:
        answer = f"I could not start that plan because its final safety check failed:\n\n`{error}`"
        store.log_activity("chat_action_approval_failed", str(error), level="warning")
    store.add_message(
        conversation_id,
        "assistant",
        answer,
        metadata={"source": "chat_approval", "action_id": action_id},
    )
    return True


def _set_action_session_id(action_type: str, action_id: str) -> None:
    if action_type in {"optimize_blur_detection", "optimize_artifact_detection"}:
        st.session_state[BLUR_ACTION_ID_KEY] = action_id
    elif action_type == "extract_rois":
        st.session_state[ROI_ACTION_ID_KEY] = action_id
    elif action_type == "register_codex_he_pair":
        st.session_state[REGISTRATION_ACTION_ID_KEY] = action_id
    elif action_type == "general_spacemap_job":
        st.session_state[GENERAL_ACTION_ID_KEY] = action_id
    st.session_state[LIVE_ACTION_KEY] = action_id
    st.session_state["_spacemap_ai_action_notebook_pending_id"] = action_id


def _start_checked_action(action_type: str, action_id: str, *, approval_method: str) -> int:
    if action_type in {"optimize_blur_detection", "optimize_artifact_detection"}:
        return launch_blur_action(action_id, approval_method=approval_method)
    if action_type == "extract_rois":
        return launch_roi_action(action_id, approval_method=approval_method)
    if action_type == "register_codex_he_pair":
        return launch_registration_action(action_id)
    if action_type == "general_spacemap_job":
        return launch_general_action(action_id, approval_method=approval_method)
    raise ValueError("This action type cannot be started.")


def _latest_pending_or_running_action() -> dict[str, Any] | None:
    actions = list_all_actions(limit=30)
    return next((item for item in actions if item.get("state") in {"launching", "running"}), None) or next(
        (item for item in actions if item.get("state") == "planned"),
        None,
    )


def _render_chat_action_banner(page_id: str) -> None:
    action = _latest_pending_or_running_action()
    if action is None:
        return
    action_id = str(action["action_id"])
    action_type = str(action["action_type"])
    _set_action_session_id(action_type, action_id)
    label = {
        "optimize_blur_detection": "blur experiment",
        "optimize_artifact_detection": "artifact experiment",
        "extract_rois": "ROI extraction",
        "register_codex_he_pair": "registration",
        "general_spacemap_job": "protected AI",
    }.get(action_type, action_type)
    state = str(action.get("state") or "")
    if state == "planned":
        st.warning(
            f"**NOT RUNNING — WAITING FOR APPROVAL** · {label} plan `{action_id[:8]}`. "
            "You may review it in Actions or start the checked plan here."
        )
        if st.button(
            "Start This Checked Plan",
            type="primary",
            width="stretch",
            key=f"assistant_chat_start_{page_id}_{action_id}",
        ):
            try:
                pid = _start_checked_action(
                    action_type,
                    action_id,
                    approval_method="Direct Start This Checked Plan button in chat",
                )
                AssistantStore().log_activity(
                    "chat_action_approved",
                    f"User started checked {label} action {action_id[:8]} from Chat.",
                    details={"action_id": action_id, "pid": pid},
                )
                st.success(f"Started process {pid}. Its Action Notebook will track it.")
                st.rerun()
            except (
                BlurActionError,
                GeneralActionError,
                ROIActionError,
                RegistrationActionError,
                OSError,
                ValueError,
            ) as error:
                st.error(str(error))
    else:
        st.info(
            f"**RUNNING** · {label} job `{action_id[:8]}` · "
            f"{action.get('stage') or 'Starting'}"
        )


def _chat_action_id(store: AssistantStore, conversation_id: str) -> str:
    """Recover the action owned by this chat, even after a model switch or rerun."""
    messages = store.get_messages(conversation_id, limit=80)
    for message in reversed(messages):
        action_id = str((message.get("metadata") or {}).get("action_id") or "")
        if action_id:
            return action_id
    for state_key in (
        BLUR_ACTION_ID_KEY,
        ROI_ACTION_ID_KEY,
        REGISTRATION_ACTION_ID_KEY,
        GENERAL_ACTION_ID_KEY,
    ):
        action_id = str(st.session_state.get(state_key) or "")
        if action_id:
            return action_id
    for action in list_all_actions(limit=50):
        details = action_details(str(action.get("action_id") or ""))
        if str((details.get("plan") or {}).get("conversation_id") or "") == conversation_id:
            return str(action.get("action_id") or "")
    # Compatibility for actions created before conversation_id was stored in
    # the plan: match an exact input or run path mentioned in this chat.
    chat_text = "\n".join(str(message.get("content") or "") for message in messages)
    normalized_chat_text = re.sub(
        # Sites that expose one share under two prefixes can add a rewrite
        # here; by default no path rewriting is performed.
        r"^$",
        "",
        chat_text,
    )
    for action in list_all_actions(limit=50):
        details = action_details(str(action.get("action_id") or ""))
        plan = details.get("plan") or {}
        anchors = (
            str(plan.get("image_path") or ""),
            str(plan.get("run_output_dir") or ""),
            str(plan.get("final_output") or ""),
        )
        if any(
            anchor
            and anchor
            in normalized_chat_text
            for anchor in anchors
        ):
            return str(action.get("action_id") or "")
    return ""


def _completed_blur_answer(action_id: str, details: dict[str, Any]) -> str:
    status = details.get("status") or {}
    report = details.get("report") or {}
    result = report.get("result") or {}
    rows = list(result.get("comparison") or [])
    count = len(rows)
    nonempty = [row for row in rows if int(row.get("polygon_count") or 0) > 0]
    output_dir = str(status.get("output_dir") or (details.get("plan") or {}).get("run_output_dir") or "")
    artifact_kind = str((details.get("plan") or {}).get("artifact_kind") or "blur")
    artifact_label = "bright glare/debris" if artifact_kind == "bright" else "blur"
    if rows and not nonempty:
        gaps = [float(row.get("separation_gap")) for row in rows if row.get("separation_gap") is not None]
        gap_note = f" The largest measured blur-population gap was {max(gaps):.3f}." if gaps else ""
        return (
            f"Job `{action_id[:8]}` finished the machine tests, but it did **not** finish the "
            f"scientific {artifact_label} review. All **{count}** settings returned **0 polygons**, so there is "
            f"no valid best setting to recommend.{gap_note} No visual review was performed because "
            "there were no candidate regions to inspect."
            + (
                " A new blur run will also test the upstream population-separation setting "
                "(2.5 and 2.0), instead of repeating only downstream sensitivity changes."
                if artifact_kind == "blur"
                else ""
            )
            + f"\n\nRun folder: `{output_dir}`"
        )
    selected_name = str(result.get("selected_trial") or "")
    selected = next((row for row in rows if str(row.get("trial")) == selected_name), {})
    polygon_count = int(selected.get("polygon_count") or 0)
    preview_count = len(result.get("review_previews") or [])
    preview_note = (
        f" I also prepared {preview_count} local QC preview(s)."
        if preview_count
        else " The visual review has not been performed yet."
    )
    return (
        f"Job `{action_id[:8]}` finished **{count}** parameter tests. The provisional machine "
        f"candidate is **{selected_name or 'not recorded'}** with **{polygon_count} polygon(s)**."
        f"{preview_note} A model or person must review those polygons before the GeoJSON is final.\n\n"
        f"Run folder: `{output_dir}`"
    )


def _plain_action_status_answer(
    prompt: str,
    store: AssistantStore,
    conversation_id: str,
) -> str | None:
    normalized = str(prompt).lower()
    status_question = any(
        phrase in normalized
        for phrase in (
            "are you running",
            "is it running",
            "did you start",
            "has it started",
            "what is the job status",
            "have you finished",
            "are you finished",
            "did it finish",
            "is it finished",
            "is it done",
            "have you done the visual review",
        )
    )
    if not status_question:
        return None
    action_id = _chat_action_id(store, conversation_id)
    if not action_id:
        return "I do not see a background action connected to this chat."
    details = action_details(action_id)
    plan = details.get("plan") or {}
    status = details.get("status") or {}
    state = str(status.get("state") or "unknown")
    action_type = str(plan.get("action_type") or "job")
    action = {
        "action_id": action_id,
        "action_type": action_type,
        "state": state,
        "stage": status.get("stage"),
    }
    if state == "planned":
        return (
            f"No. Plan `{action['action_id'][:8]}` is checked but **waiting for approval**. "
            "Use **Start This Checked Plan** in Chat, type **approve**, or start it from Actions."
        )
    if state in {"launching", "running"}:
        return (
            f"No, it is not finished yet. Job `{action_id[:8]}` is **running**. Current step: "
            f"{action.get('stage') or 'Starting'}. Open **Actions** to watch its progress and live results."
        )
    if state == "failed":
        return f"Job `{action_id[:8]}` failed: {status.get('message') or 'No error detail was saved.'}"
    if state == "cancelled":
        return f"Job `{action_id[:8]}` was stopped. {status.get('message') or ''}".strip()
    if state == "completed" and action_type in {
        "optimize_blur_detection",
        "optimize_artifact_detection",
    }:
        return _completed_blur_answer(action_id, details)
    if state == "completed":
        return (
            f"Yes. Job `{action_id[:8]}` completed. "
            f"{status.get('message') or 'Its report and outputs are available in Actions.'}"
        )
    return f"Job `{action_id[:8]}` has state **{state}**. {status.get('message') or ''}".strip()


def _recent_preview_states(
    store: AssistantStore,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Return only the latest complete visual state from the current chat thread."""
    candidates = []
    for item in store.list_artifacts(kind="assistant_image_preview", limit=200):
        metadata = item.get("metadata") or {}
        if str(metadata.get("conversation_id") or "") != conversation_id:
            continue
        preview_path = str(item.get("path") or metadata.get("preview_path") or "")
        source_path = str(metadata.get("source_path") or "")
        if preview_path and source_path and Path(preview_path).is_file() and Path(source_path).exists():
            candidates.append((item, metadata, preview_path))
    if not candidates:
        return []
    newest_request = str(candidates[0][1].get("request_message_id") or "")
    selected = [
        value
        for value in candidates
        if newest_request
        and str(value[1].get("request_message_id") or "") == newest_request
    ] or [candidates[0]]
    # Reconcile old results created by earlier, looser text routing.  A named
    # marker view is more specific than an accidental generic H&E view unless
    # the command explicitly requested H&E. This repairs state without deleting
    # the historical artifact or consulting another chat.
    if len(selected) > 1:
        command = str(selected[0][0].get("source_command") or "")
        parsed = parse_image_preview_request(command) if command else None
        requested_modes = set((parsed or {}).get("modes") or [])
        marker_items = [
            value for value in selected if str(value[1].get("preview_mode") or "") == "markers"
        ]
        if requested_modes:
            selected = [
                value
                for value in selected
                if str(value[1].get("preview_mode") or "") in requested_modes
                or str(value[1].get("preview_mode") or "") == "markers"
            ]
        elif marker_items:
            selected = marker_items
    newest_per_view = []
    seen_views = set()
    for value in selected:
        metadata = value[1]
        key = (
            str(Path(str(metadata.get("source_path") or "")).expanduser().resolve(strict=False)),
            str(metadata.get("preview_mode") or ""),
            tuple(str(item) for item in (metadata.get("channels") or [])),
            str(metadata.get("qc_kind") or ""),
            int(metadata.get("candidate") or 0),
        )
        if key in seen_views:
            continue
        seen_views.add(key)
        newest_per_view.append(value)
    selected = newest_per_view
    states = []
    for item, metadata, preview_path in selected:
        state = dict(metadata)
        state["preview_path"] = preview_path
        state["source_command"] = str(item.get("source_command") or "")
        states.append(state)
    return states


def _recent_preview_source(store: AssistantStore, conversation_id: str) -> str:
    """Return the newest source from this chat; cache and other chats are ignored."""
    states = _recent_preview_states(store, conversation_id)
    return str(states[0].get("source_path") or "") if states else ""


def _recent_annotation_qc_bundle(
    store: AssistantStore,
    conversation_id: str,
) -> dict[str, Any] | None:
    """Recover the active annotation-review evidence from this chat only."""
    candidates = []
    for item in store.list_artifacts(kind="assistant_image_preview", limit=300):
        metadata = item.get("metadata") or {}
        if str(metadata.get("conversation_id") or "") != conversation_id:
            continue
        if not str(metadata.get("qc_kind") or ""):
            continue
        path = Path(str(item.get("path") or metadata.get("preview_path") or ""))
        source = Path(str(metadata.get("source_path") or ""))
        if path.is_file() and source.is_file():
            candidates.append((item, metadata, path))
    if not candidates:
        return None
    request_id = str(candidates[0][1].get("request_message_id") or "")
    selected = [
        value
        for value in candidates
        if str(value[1].get("request_message_id") or "") == request_id
    ]
    if not selected:
        return None
    selected.sort(
        key=lambda value: (
            0 if value[1].get("qc_kind") == "overview" else 1,
            int(value[1].get("candidate") or 0),
        )
    )
    previews = []
    for item, metadata, path in selected:
        preview = dict(metadata)
        preview["preview_path"] = str(path)
        preview["source_command"] = str(item.get("source_command") or "")
        previews.append(preview)
    overlay_paths = []
    for source in previews[0].get("overlay_sources") or []:
        if isinstance(source, dict) and str(source.get("path") or ""):
            overlay_path = Path(str(source["path"]))
            try:
                current = overlay_path.stat()
            except OSError:
                return None
            expected_size = source.get("size")
            expected_mtime = source.get("mtime_ns")
            if expected_size is not None and int(expected_size) != int(current.st_size):
                return None
            if expected_mtime is not None and int(expected_mtime) != int(current.st_mtime_ns):
                return None
            overlay_paths.append(str(overlay_path))
    if not overlay_paths:
        return None
    return {
        "request": {
            "source_path": str(previews[0].get("source_path") or ""),
            "geojson_paths": overlay_paths,
            "preferred_mode": (
                "he" if previews[0].get("preview_mode") == "qc_he" else "automatic"
            ),
            "max_size": 1024,
        },
        "previews": previews,
        "summary": dict(previews[0].get("qc_summary") or {}),
    }


def _is_annotation_qc_followup(prompt: str) -> bool:
    """Recognize a continuation of an active visual annotation review."""
    normalized = " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(prompt).casefold()).split()
    )
    if normalized in {
        "proceed",
        "go ahead",
        "do it",
        "continue",
        "assess it",
        "check it",
        "check them",
    }:
        return True
    return bool(
        re.search(
            r"\b(?:blood clot|clot|polygon|annotation|visual assessment|visual review|false positive)\b",
            normalized,
        )
        and re.search(
            r"\b(?:assess|assessment|check|review|inspect|evaluate|think|correct|wrong|suspicious)\b",
            normalized,
        )
    )


def _same_resolved_path(left: str, right: str) -> bool:
    """Treat configured filesystem aliases as one source."""
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return str(left) == str(right)


def _visual_state_context(states: list[dict[str, Any]], *, online_safe: bool = False) -> str:
    """Describe the thread's active visual state without consulting other chats or cache."""
    views = []
    for state in states[:8]:
        source = str(state.get("source_path") or "")
        raw_scale = state.get("scale_bar") or {}
        scale_bar = (
            {
                key: raw_scale.get(key)
                for key in ("source_mpp", "length_um", "position", "color")
                if raw_scale.get(key) is not None
            }
            if isinstance(raw_scale, dict)
            else {}
        )
        overlays = [
            str(item.get("path") or "")
            for item in (state.get("overlay_sources") or [])
            if isinstance(item, dict)
        ]
        views.append(
            {
                "source": Path(source).name if online_safe else source,
                "view": str(state.get("preview_mode") or ""),
                "channels": [str(item) for item in (state.get("channels") or [])],
                "overlays": [Path(item).name if online_safe else item for item in overlays],
                "scale_bar": scale_bar or None,
            }
        )
    return json.dumps({"views": views}, ensure_ascii=False, indent=2) if views else ""


def _submit_prompt(
    prompt: str,
    page_id: str,
    store: AssistantStore,
    conversation_id: str,
    *,
    live: bool = False,
    uploaded_images: list[Any] | None = None,
    record_user_message: bool = True,
    online_share_approved: bool = False,
    existing_user_message_id: str | None = None,
) -> None:
    clean_prompt = str(prompt).strip()
    if not clean_prompt:
        return
    created_action_id = ""
    provider_settings = load_provider_settings()
    online_provider = provider_settings.provider == NVIDIA_PROVIDER
    active_model_id = provider_settings.nvidia_model_id if online_provider else MODEL_ID
    settings = load_model_settings()
    response_token_limit, is_execution_request, budget_kind = completion_budget(
        clean_prompt,
        settings.max_tokens,
    )
    requested_page = _requested_context_page(clean_prompt, page_id)
    # The Action Notebook is itself meaningful context. A short instruction
    # such as "make trial 4 stricter" must stay attached to the selected job,
    # while the notebook still inherits the last visible pipeline tab.
    context = _current_context(page_id if page_id == "assistant_action" else requested_page)
    image_files = list(uploaded_images or [])[:4]
    user_uploaded_images = bool(image_files)
    image_names = [str(getattr(item, "name", "uploaded image")) for item in image_files]
    user_message_id = str(existing_user_message_id or "")
    if record_user_message:
        user_message_id = store.add_message(
            conversation_id,
            "user",
            clean_prompt,
            metadata={
                "page_context": context.get("page_id"),
                "assistant_mode": "planning_with_reviewed_actions",
                "image_names": image_names,
            },
        )
        suggested_memory_ids = []
        for suggestion in suggest_memories(clean_prompt):
            memory_id = store.add_memory_if_new(
                suggestion.content,
                scope="project",
                kind=suggestion.kind,
                status="proposed",
                confidence=suggestion.confidence,
                evidence=[
                    f"conversation:{conversation_id}",
                    f"message:{user_message_id}",
                ],
                source="automatic_chat_suggestion",
            )
            if memory_id:
                suggested_memory_ids.append(memory_id)
        if suggested_memory_ids:
            store.log_activity(
                "memory_suggested",
                f"Saved {len(suggested_memory_ids)} possible reusable lesson(s) for review.",
                details={"conversation_id": conversation_id, "memory_ids": suggested_memory_ids},
            )
        store.log_activity(
            "chat_request",
            f"Question received from {context.get('page_title', page_id)}",
            details={"conversation_id": conversation_id, "page_id": context.get("page_id")},
        )
    if _launch_chat_approved_action(clean_prompt, store, conversation_id):
        return
    status_answer = _plain_action_status_answer(clean_prompt, store, conversation_id)
    if status_answer is not None:
        store.add_message(
            conversation_id,
            "assistant",
            status_answer,
            metadata={"source": "recorded_action_status"},
        )
        return

    workflow_request = is_multi_step_scientific_request(clean_prompt)
    thread_visuals = (
        [] if workflow_request else _recent_preview_states(store, conversation_id)
    )
    # A request to assess an annotation is not a plain display request, even if
    # it also says "show" or "overlay". Give scientific QC priority so the
    # selected model receives the evidence and question.
    direct_qc_request = (
        None if workflow_request else parse_visual_assessment_request(clean_prompt)
    )
    preview_request = (
        None if direct_qc_request is not None else parse_image_preview_request(clean_prompt)
    )
    if preview_request is not None:
        explicit_source = str(preview_request.get("source_path") or "")
        recent_visuals = thread_visuals
        source_path = explicit_source or (
            str(recent_visuals[0].get("source_path") or "") if recent_visuals else ""
        )
        if not source_path:
            answer = (
                "I understand that you want an image view or edit, but this chat thread has no recent "
                "image to reuse. Give me the TIFF path once; later requests in this chat can reuse its "
                "complete visual state."
            )
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={"source": "image_preview_needs_source", "page_context": "tab9"},
            )
            return
        try:
            available_channels = preview_channel_names(source_path)
            requested_channels = match_requested_channels(
                str(preview_request.get("request_text") or clean_prompt),
                available_channels,
            )
            modes = list(preview_request.get("modes") or [])
            geojson_paths = list(preview_request.get("geojson_paths") or [])
            scale_bar = preview_request.get("scale_bar")
            same_thread_source = any(
                _same_resolved_path(str(item.get("source_path") or ""), source_path)
                for item in recent_visuals
            )
            reuse_thread_visual = bool(
                recent_visuals
                and same_thread_source
                and not modes
                and not requested_channels
            )
            if not modes and not requested_channels and geojson_paths and not reuse_thread_visual:
                modes.append(
                    "he" if all(name in available_channels for name in HE_CHANNELS) else "non_he"
                )
            if not modes and not requested_channels and not reuse_thread_visual:
                shown_channels = ", ".join(available_channels[:30])
                extra = max(0, len(available_channels) - 30)
                answer = (
                    "I found the image, but this chat has no matching prior visual state and I could "
                    "not identify which channel you want. Available "
                    f"channels include: {shown_channels}"
                    + (f", and {extra} more." if extra else ".")
                )
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"source": "image_preview_needs_channels", "page_context": "tab9"},
                )
                return
            previews = list(recent_visuals) if reuse_thread_visual else []
            with st.spinner("Reading small pyramid layers and making the requested thumbnails…"):
                if "he" in modes:
                    previews.extend(
                        generate_he_previews(
                            source_path,
                            max_size=int(preview_request["max_size"]),
                        )
                    )
                if "non_he" in modes:
                    previews.extend(
                        generate_non_he_previews(
                            source_path,
                            channel_names=available_channels,
                            max_size=int(preview_request["max_size"]),
                        )
                    )
                if requested_channels:
                    previews.extend(
                        generate_marker_previews(
                            source_path,
                            requested_channels,
                            channel_names=available_channels,
                            max_size=int(preview_request["max_size"]),
                        )
                    )
                if geojson_paths:
                    previews = generate_overlay_previews(
                        previews,
                        geojson_paths,
                        prompt=clean_prompt,
                    )
                if scale_bar:
                    previews = generate_scale_bar_previews(
                        previews,
                        mpp=scale_bar.get("mpp"),
                        length_um=scale_bar.get("length_um"),
                        position=str(scale_bar.get("position") or "bottom-left"),
                    )
            for preview in previews:
                preview_mode = str(preview.get("preview_mode") or "he")
                channel_text = " + ".join(str(item) for item in preview.get("channels") or [])
                preview_label = {
                    "he": "H&E",
                    "non_he": "CODEX composite",
                    "markers": channel_text or "Marker channels",
                }.get(preview_mode, channel_text or "Image")
                if preview.get("overlay_sources"):
                    preview_label += " + overlays"
                if preview.get("scale_bar"):
                    preview_label += " + scale bar"
                store.register_artifact(
                    preview["preview_path"],
                    kind="assistant_image_preview",
                    caption=f"{preview_label} · {Path(preview['source_path']).name}",
                    source_page="tab9",
                    source_command=clean_prompt,
                    metadata={
                        **preview,
                        "conversation_id": conversation_id,
                        "request_message_id": user_message_id,
                    },
                )
            output_dir = Path(previews[0]["preview_path"]).parent
            preview_lines = []
            for preview in previews:
                label = {
                    "he": "H&E",
                    "non_he": "Representative CODEX composite",
                    "markers": "Requested channel view",
                }.get(str(preview.get("preview_mode") or ""), "Image view")
                preview_lines.append(
                    f"- **{label}:** {' + '.join(preview.get('channels') or [])}"
                )
                overlay_stats = preview.get("overlay_stats") or {}
                if overlay_stats:
                    preview_lines[-1] += (
                        f"; drew {int(overlay_stats.get('cell_polygons') or 0):,} cell polygon(s) "
                        f"and {int(overlay_stats.get('tissue_polygons') or 0):,} annotation polygon(s)"
                    )
                scale_settings = preview.get("scale_bar") or {}
                if scale_settings:
                    preview_lines[-1] += (
                        f"; added a {float(scale_settings['length_um']):g} µm white scale bar "
                        f"at {str(scale_settings['position']).replace('-', ' ')}"
                    )
            answer = (
                f"Done. I displayed **{len(previews)} requested thumbnail(s)** below.\n\n"
                + "\n".join(preview_lines)
                + "\n\nI read only the needed channels from a small pyramid level; the full image was not loaded.\n\n"
                f"Saved previews: `{output_dir}`\n\n"
                "This display was made by SPACEMAP's local image tool. The selected AI model was not called."
            )
            source = "image_previews_generated"
            level = "info"
            details = {
                "conversation_id": conversation_id,
                "request_message_id": user_message_id,
                "previews": previews,
            }
        except (PreviewActionError, AssistantPathError, OSError, ValueError) as error:
            answer = f"I could not make the requested image view:\n\n`{error}`"
            source = "image_preview_failed"
            level = "warning"
            details = {"conversation_id": conversation_id, "error": str(error)}
        store.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={"source": source, "page_context": "tab9"},
        )
        store.log_activity(source, answer.split("\n", 1)[0], level=level, details=details)
        return

    local_qc_note = ""
    qc_summary: dict[str, Any] = {}
    qc_previews: list[dict[str, Any]] = []
    # Annotation review is prepared locally for every model. Online models may
    # receive only these small generated previews, never the source TIFF.
    if clean_prompt:
        qc_request = direct_qc_request
        reusable_qc: dict[str, Any] | None = None
        recent_qc = _recent_annotation_qc_bundle(store, conversation_id)
        if qc_request is None and _is_annotation_qc_followup(clean_prompt):
            reusable_qc = recent_qc
            if reusable_qc is not None:
                qc_request = dict(reusable_qc["request"])
        elif qc_request is not None and recent_qc is not None:
            recent_request = recent_qc["request"]
            same_source = _same_resolved_path(
                str(qc_request.get("source_path") or ""),
                str(recent_request.get("source_path") or ""),
            )
            requested_overlays = {
                str(Path(path).expanduser().resolve(strict=False))
                for path in qc_request.get("geojson_paths") or []
            }
            recent_overlays = {
                str(Path(path).expanduser().resolve(strict=False))
                for path in recent_request.get("geojson_paths") or []
            }
            if same_source and requested_overlays == recent_overlays:
                reusable_qc = recent_qc
        if qc_request is not None:
            try:
                if reusable_qc is not None:
                    qc_previews = list(reusable_qc["previews"])
                    qc_summary = dict(reusable_qc["summary"])
                else:
                    with st.spinner(
                        "Reading the annotation and building local review tiles…"
                    ):
                        qc_bundle = generate_annotation_qc_previews(
                            qc_request["source_path"],
                            list(qc_request["geojson_paths"]),
                            preferred_mode=str(qc_request["preferred_mode"]),
                            overview_size=int(qc_request["max_size"]),
                            tile_size=768,
                            max_tiles=6,
                        )
                    qc_previews = list(qc_bundle["previews"])
                    qc_summary = dict(qc_bundle["summary"])
                for preview in qc_previews:
                    kind_label = (
                        "QC overview"
                        if preview.get("qc_kind") == "overview"
                        else f"QC candidate {preview.get('candidate')}"
                    )
                    store.register_artifact(
                        preview["preview_path"],
                        kind="assistant_image_preview",
                        caption=f"{kind_label} · {Path(preview['source_path']).name}",
                        source_page="tab4",
                        source_command=clean_prompt,
                        metadata={
                            **preview,
                            "conversation_id": conversation_id,
                            "request_message_id": user_message_id,
                        },
                    )
                image_files.extend(Path(item["preview_path"]) for item in qc_previews)
                image_files = image_files[:8]
                image_names = [
                    str(item.name if isinstance(item, Path) else getattr(item, "name", "image"))
                    for item in image_files
                ]
                area_text = (
                    f"{float(qc_summary['total_area_mm2']):.4f} mm²"
                    if qc_summary.get("total_area_mm2") is not None
                    else f"{float(qc_summary['total_area_px2']):,.0f} source pixels"
                )
                annotation_kind = ""
                for item in qc_previews:
                    explicit_kind = str(item.get("annotation_kind") or "").casefold()
                    overlay_names = [
                        Path(str(source.get("path") or "")).name.casefold()
                        for source in item.get("overlay_sources") or []
                        if isinstance(source, dict)
                    ]
                    if explicit_kind in {"blur", "bright"}:
                        annotation_kind = explicit_kind
                        break
                    if any("blur" in name for name in overlay_names):
                        annotation_kind = "blur"
                        break
                    if any("bright" in name for name in overlay_names):
                        annotation_kind = "bright"
                        break
                if annotation_kind == "blur":
                    evidence_header = (
                    "- The supplied images include local H&E and representative CODEX overviews and "
                    "numbered polygon tiles. A red outline and faint red fill mark each proposed "
                    "blurry region. Compare fine detail inside each outline with nearby tissue.\n"
                    "- Reject low-signal, naturally empty, dark, folded, or edge regions unless the "
                    "surrounding image evidence truly shows loss of focus.\n"
                    )
                    assessment_target = (
                        "true blur from low signal, empty space, folds, tissue edges, and other artifacts"
                    )
                elif annotation_kind == "bright":
                    evidence_header = (
                        "- The supplied images include local H&E and CODEX overviews plus numbered "
                        "tiles. The CODEX view uses DAPI and the strongest marker evidence saved for "
                        "the detected region. A red outline marks the proposed bright artifact.\n"
                        "- True glare/debris is unusually bright and smooth or structureless. "
                        "Organized cellular or tissue-shaped marker signal is evidence of a false positive.\n"
                    )
                    assessment_target = (
                        "true glare/debris from structured biological marker expression, ordinary bright "
                        "cells, tissue edges, and naturally low-texture tissue"
                    )
                else:
                    evidence_header = (
                        "- Image 1 is the whole H&E overview. Later images are numbered polygon tiles "
                        "with nearby tissue for comparison.\n"
                        "- A red outline and faint red fill mark the proposed blood-clot polygon.\n"
                        "- SPACEMAP's blood-clot function uses virtual H&E and requires both eosin-rich "
                        "pink material and low local hematoxylin/nuclear texture.\n"
                    )
                    assessment_target = (
                        "true clot from pink but nucleated muscle, collagen, blur, empty space, and edge artifacts"
                    )
                local_qc_note = (
                    "\n\nLOCAL SPACEMAP QC EVIDENCE (created automatically; full TIFF stayed local):\n"
                    f"- The GeoJSON contains {int(qc_summary['feature_count'])} feature(s) and "
                    f"{int(qc_summary['polygon_count'])} polygon component(s), totaling {area_text}.\n"
                    f"{evidence_header}"
                    "Give the visual assessment now. Do not propose a plan, ask for confirmation, or "
                    "emit a job or code. Assess every supplied candidate tile. Distinguish "
                    f"{assessment_target}. Report likely correct, likely false positive, or uncertain "
                    "for each candidate, with a short visual reason. This is thumbnail-level QC, not "
                    "final pathology approval."
                )
                store.log_activity(
                    "local_annotation_qc_prepared",
                    f"Prepared {len(qc_previews)} local visual QC image(s) for Gemma.",
                    details={
                        "conversation_id": conversation_id,
                        "summary": qc_summary,
                        "previews": [item["preview_path"] for item in qc_previews],
                    },
                )
            except (PreviewActionError, AssistantPathError, OSError, TypeError, ValueError) as error:
                if "no polygons" in str(error).casefold():
                    answer = (
                        "I checked the GeoJSON directly. It currently contains **0 polygons**, so "
                        "there is no annotation region to assess in this file. The latest detector "
                        "may have found no candidate or replaced an older result."
                    )
                    event = "local_annotation_qc_empty"
                else:
                    answer = f"I could not prepare the local visual QC evidence:\n\n`{error}`"
                    event = "local_annotation_qc_failed"
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"source": event, "page_context": "tab4"},
                )
                store.log_activity(event, str(error), level="warning")
                return

    # Resolve a path-only follow-up from this conversation before choosing an
    # evidence adapter. This keeps the second message (for example, a CSV path)
    # attached to the user's earlier cluster-labeling question, without looking
    # at another chat or stale preview cache.
    action_request = _request_with_recent_path_context(clean_prompt, store, conversation_id)
    cluster_evidence = prepare_cluster_labeling_evidence(action_request, st.session_state)
    if cluster_evidence is not None:
        if cluster_evidence.get("error"):
            answer = str(cluster_evidence["error"])
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={"source": "pixel_cluster_evidence_missing", "page_context": "tab4"},
            )
            store.log_activity("pixel_cluster_evidence_missing", answer, level="warning")
            return
        local_qc_note = str(cluster_evidence.get("note") or "")
        cluster_images = [Path(path) for path in cluster_evidence.get("images") or []]
        if cluster_images:
            local_qc_note += (
                "\nLOCAL IMAGE LEGEND: Image 1 is the balanced marker-expression heatmap; "
                "Image 2 is the spatial cluster map."
            )
        image_files.extend(cluster_images)
        qc_previews = [
            {"preview_path": str(path), "source_path": str(path), "qc_kind": "cluster_evidence"}
            for path in cluster_images
        ]
        store.log_activity(
            "pixel_cluster_evidence_prepared",
            f"Prepared current Pixel Classification evidence for {len(cluster_evidence.get('clusters') or [])} clusters.",
            details={
                "conversation_id": conversation_id,
                "assignment_csv": str(cluster_evidence.get("assignment_csv") or ""),
                "profile_csv": str(cluster_evidence.get("profile_csv") or ""),
                "figures": [str(path) for path in cluster_images],
            },
        )
        # A complete 20-cluster table needs more space than an ordinary chat
        # answer. This exception affects only model text generation; it does
        # not load tables or image outputs into GPU memory all at once.
        response_token_limit = max(response_token_limit, 4_096 if online_provider else 2_048)
        budget_kind = "pixel_cluster_labeling"

    quick_reply = None if image_files else _deterministic_quick_reply(clean_prompt, context)
    if quick_reply is not None:
        store.add_message(conversation_id, "assistant", quick_reply, metadata={"source": "page_context"})
        store.log_activity("chat_response", "Answered from the registered page context.")
        return

    blur_draft = _artifact_draft_with_thread_context(
        clean_prompt,
        store,
        conversation_id,
    )
    if blur_draft is not None:
        blur_draft["provider"] = provider_settings.provider
        blur_draft["model_id"] = active_model_id
        blur_draft["conversation_id"] = conversation_id
        blur_draft["request_message_id"] = user_message_id
        blur_draft["automatic_visual_review"] = True
        blur_draft["online_visual_review_approved"] = bool(
            online_provider and st.session_state.get(NVIDIA_SESSION_CONSENT_KEY)
        )
        artifact_label = (
            "bright-glare/debris" if blur_draft.get("artifact_kind") == "bright" else "blur"
        )
        try:
            with st.spinner("Checking the Annotation inputs and building the parameter tests…"):
                plan = build_blur_plan(**blur_draft)
            created_action_id = plan.action_id
            st.session_state[BLUR_ACTION_ID_KEY] = plan.action_id
            st.session_state[LIVE_ACTION_KEY] = plan.action_id
            if preapproval_enabled():
                pid = launch_blur_action(
                    plan.action_id,
                    approval_method="Session pre-approval switch",
                )
                answer = (
                    f"I checked the image, tissue boundary, and saved Annotation settings, then started "
                    f"**{plan.maximum_trials} {artifact_label} tests** as process {pid}. Open **Actions** "
                    "to watch the parameters, progress, previews, and tables together."
                )
                source = "blur_action_started"
            else:
                answer = (
                    f"I built checked {artifact_label} experiment `{plan.action_id[:8]}` with "
                    f"**{plan.maximum_trials} settings**. Type **approve** here, or review it under "
                    "**Actions**. Every trial goes into a new folder; the current annotation "
                    "will not be replaced."
                )
                if plan.online_visual_review_approved:
                    answer += (
                        " Starting this plan also allows only its small generated QC PNGs to be "
                        "reviewed by the selected NVIDIA model; the source TIFF stays local."
                    )
                source = "blur_action_planned"
            store.log_activity(
                source,
                f"{artifact_label.title()} experiment {plan.action_id[:8]} translated from chat.",
                details={"action_id": plan.action_id, "output_dir": plan.run_output_dir},
            )
        except (BlurActionError, OSError, ValueError) as error:
            answer = f"I could not prepare the {artifact_label} tests:\n\n`{error}`"
            source = "blur_action_validation_failed"
            store.log_activity(source, str(error), level="warning")
        store.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={
                "source": source,
                "page_context": "tab4",
                "action_id": created_action_id,
            },
        )
        return

    roi_draft = parse_roi_extraction_request(action_request)
    if roi_draft is not None:
        set_roi_draft(roi_draft)
        try:
            with st.spinner("Checking the image and every ROI boundary…"):
                plan = build_roi_plan(**roi_draft, conversation_id=conversation_id)
            created_action_id = plan.action_id
            st.session_state[ROI_ACTION_ID_KEY] = plan.action_id
            st.session_state[LIVE_ACTION_KEY] = plan.action_id
            if preapproval_enabled():
                pid = launch_roi_action(plan.action_id, approval_method="Session pre-approval switch")
                answer = (
                    f"I checked the paths and {len(plan.rois)} ROI boundaries, created the missing output "
                    f"folder plan, and started ROI extraction in the background as process {pid}. "
                    "Open **Actions > ROI Extraction** to see progress and the read-only worker terminal."
                )
                source = "roi_action_started"
                event = "roi_extraction_started"
            else:
                replacement = (
                    " You asked to replace same-named files; old files will be backed up."
                    if plan.overwrite
                    else " If a same-named output appears, the job will stop."
                )
                answer = (
                    f"I accepted these paths as one allowed SPACEMAP_AI project and built checked plan "
                    f"`{plan.action_id[:8]}`. I found **{len(plan.rois)} ROIs**, **{len(plan.channels)} "
                    f"channels**, and an image size of **{plan.image_width:,} × {plan.image_height:,}**. "
                    "Type **approve** here to start this exact plan, or review it under "
                    "**Actions > ROI Extraction** first."
                    + replacement
                )
                source = "roi_action_planned"
                event = "roi_extraction_planned"
        except (ROIActionError, OSError, ValueError) as error:
            answer = f"I could not prepare ROI extraction because a safety or input check failed:\n\n`{error}`"
            source = "roi_action_validation_failed"
            event = "roi_extraction_validation_failed"
        store.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={
                "source": source,
                "page_context": "tab3",
                "action_id": created_action_id,
            },
        )
        store.log_activity(event, "ROI request translated into a checked action.", details=roi_draft)
        return

    lowered_action_request = action_request.lower()
    if (
        "roi" in lowered_action_request
        and any(word in lowered_action_request for word in ("extract", "crop", "create"))
        and any(word in lowered_action_request for word in (".geojson", ".json"))
        and not any(word in lowered_action_request for word in (".tif", ".tiff", ".qptiff"))
    ):
        answer = (
            "The GeoJSON and output paths are inside the allowed **SPACEMAP_AI/projects** folder. "
            "I still need the registered TIFF image that should be cropped. Send its full path in "
            "your next message; I will keep this request and combine the paths."
        )
        store.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={"source": "roi_action_needs_image", "page_context": "tab3"},
        )
        return

    registration_draft = parse_registration_request(action_request)
    if registration_draft is not None:
        set_registration_draft(registration_draft)
        if preapproval_enabled():
            try:
                plan = build_registration_plan(
                    **registration_draft,
                    conversation_id=conversation_id,
                )
                created_action_id = plan.action_id
                st.session_state[REGISTRATION_ACTION_ID_KEY] = plan.action_id
                pid = launch_registration_action(plan.action_id)
                answer = (
                    f"I checked the paths and settings and started registration in the background as process {pid}. "
                    "Open **Actions > Registration** to see progress and QC."
                )
                source = "registration_action_started"
                event = "registration_started"
            except (RegistrationActionError, OSError, ValueError) as error:
                answer = f"I could not start registration because a safety or input check failed:\n\n`{error}`"
                source = "registration_action_validation_failed"
                event = "registration_validation_failed"
        else:
            replacement = " You asked to replace an existing result if needed; that choice is visible in the draft." if registration_draft.get("overwrite") else " If the final file already exists, I will ask before replacing it."
            answer = (
                "I found a clear CODEX + H&E registration request and prepared the action fields. "
                "Open **Actions > Registration**, build the checked plan, then approve it to run. "
                "You can turn on session pre-approval there if you want clear requests to start automatically."
                + replacement
            )
            source = "registration_action_draft"
            event = "registration_draft"
        store.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={
                "source": source,
                "page_context": "tab1",
                "action_id": created_action_id,
            },
        )
        store.log_activity(
            event,
            "Registration request translated into a checked action.",
            details=registration_draft,
        )
        return

    profile = load_instruction_profile()
    memories: list[dict[str, Any]] = []
    online_path_aliases: dict[str, str] = {}
    if online_provider:
        if not nvidia_api_key_saved():
            answer = (
                "Your question was saved, but no NVIDIA API key is saved. Open "
                "**AI setup > Answer with an online model instead** to enter it."
            )
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={"model": active_model_id, "error": True},
            )
            store.log_activity("nvidia_key_missing", "Online question was kept local; no API call was made.", level="warning")
            return
        if not bool(st.session_state.get(NVIDIA_SESSION_CONSENT_KEY)):
            answer = (
                "Your question was saved, but nothing was sent online. Turn on the NVIDIA session "
                "confirmation above the Chat box, then send the question again."
            )
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={"model": active_model_id, "error": True},
            )
            store.log_activity("nvidia_consent_missing", "Online question was kept local; no API call was made.", level="warning")
            return
        safety_issue = _online_prompt_safety_issue(clean_prompt)
        if safety_issue:
            answer = (
                f"I kept this question local and did not call NVIDIA. {safety_issue} "
                "Use **Local Gemma** for this message, or rewrite it with public, non-identifying text only."
            )
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={"model": active_model_id, "error": True},
            )
            store.log_activity("nvidia_privacy_block", "A hosted request was blocked before transmission.", level="warning")
            return
        if user_uploaded_images:
            answer = (
                "I kept the uploaded image local and did not call NVIDIA. Online image upload is "
                "disabled in this first safety version. Use **Local Gemma** for images."
            )
            store.add_message(
                conversation_id,
                "assistant",
                answer,
                metadata={"model": active_model_id, "error": True},
            )
            store.log_activity("nvidia_image_block", "Hosted image upload was blocked before transmission.", level="warning")
            return
        named_paths = extract_prompt_paths(clean_prompt)
        evidence_manifest: list[dict[str, Any]] = []
        supports_online_images = bool(NVIDIA_MODEL_CATALOG[active_model_id]["supports_images"])
        if cluster_evidence is not None:
            table_evidence_paths = [
                str(path)
                for path in (
                    cluster_evidence.get("assignment_csv"),
                    cluster_evidence.get("profile_csv"),
                    cluster_evidence.get("spatial_csv"),
                )
                if path
            ]
            # The exact marker values replace the heatmap pixels online. Send
            # only the smaller spatial map as visual evidence; this keeps the
            # request below institutional proxy and hosted-model limits.
            online_cluster_previews = [
                item
                for item in qc_previews
                if "cluster_map" in Path(str(item.get("preview_path") or "")).stem
            ][:1]
            evidence_paths = [
                *table_evidence_paths,
                *(
                    [str(online_cluster_previews[0]["preview_path"])]
                    if supports_online_images and online_cluster_previews
                    else []
                ),
            ]
            try:
                evidence_plan = plan_online_evidence("\n".join(evidence_paths))
                if not evidence_plan:
                    raise AssistantPathError("The current Pixel Classification evidence is not approved for online review.")
            except (AssistantPathError, FileNotFoundError, OSError, ValueError) as error:
                answer = f"I kept the cluster evidence local and did not call NVIDIA. `{error}`"
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "error": True},
                )
                store.log_activity("nvidia_cluster_evidence_block", str(error), level="warning")
                return
            if not preapproval_enabled() and not online_share_approved:
                st.session_state[PENDING_ONLINE_SHARE_KEY] = {
                    "prompt": clean_prompt,
                    "page_id": page_id,
                    "conversation_id": conversation_id,
                    "user_message_id": user_message_id,
                    "files": evidence_plan,
                }
                answer = (
                    "I found the marker and spatial evidence behind the current Pixel Classification tab. "
                    "Nothing has been sent online yet. Choose **Approve Online Share** below, or use Auto "
                    "approval for this session."
                )
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "source": "online_share_waiting"},
                )
                store.log_activity(
                    "nvidia_cluster_share_waiting",
                    "Owner approval is required before sending current-tab cluster evidence.",
                    details={"conversation_id": conversation_id, "files": evidence_plan},
                )
                return
            if supports_online_images and online_cluster_previews:
                try:
                    online_cluster_note = str(cluster_evidence.get("note") or "") + (
                        "\nONLINE IMAGE LEGEND: Image 1 is the spatial cluster map. The exact numeric "
                        "marker-expression values above replace the heatmap image."
                    )
                    user_content, evidence_manifest = build_online_preview_content(
                        [str(item["preview_path"]) for item in online_cluster_previews],
                        clean_prompt,
                        model_supports_images=True,
                        context_note=online_cluster_note,
                        max_size=640,
                    )
                    evidence_manifest.extend(
                        {"path": path, "kind": "derived_cluster_table_summary"}
                        for path in table_evidence_paths
                    )
                except (AssistantPathError, FileNotFoundError, OSError, ValueError) as error:
                    answer = f"I could not prepare the cluster evidence for NVIDIA: `{error}`"
                    store.add_message(
                        conversation_id,
                        "assistant",
                        answer,
                        metadata={"model": active_model_id, "error": True},
                    )
                    store.log_activity("nvidia_cluster_prepare_failed", str(error), level="warning")
                    return
            else:
                user_content = clean_prompt + str(cluster_evidence.get("note") or "")
                evidence_manifest = [
                    {"path": str(path), "kind": "local_cluster_summary_source"}
                    for path in table_evidence_paths
                ]
        elif qc_previews:
            evidence_plan = [
                {"path": str(item["preview_path"]), "kind": "generated QC preview"}
                for item in qc_previews
            ]
            if not preapproval_enabled() and not online_share_approved:
                st.session_state[PENDING_ONLINE_SHARE_KEY] = {
                    "prompt": clean_prompt,
                    "page_id": page_id,
                    "conversation_id": conversation_id,
                    "user_message_id": user_message_id,
                    "files": evidence_plan,
                }
                answer = (
                    f"I created **{len(evidence_plan)} local annotation review image(s)**. Nothing "
                    "has been sent online yet. Choose **Approve Online Share** below, or use Auto "
                    "approval for this session."
                )
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "source": "online_share_waiting"},
                )
                store.log_activity(
                    "nvidia_file_share_waiting",
                    "Owner approval is required before sending generated QC previews.",
                    details={"conversation_id": conversation_id, "files": evidence_plan},
                )
                return
            try:
                user_content, evidence_manifest = build_online_preview_content(
                    [str(item["preview_path"]) for item in qc_previews],
                    clean_prompt + local_qc_note,
                    model_supports_images=supports_online_images,
                )
            except (AssistantPathError, FileNotFoundError, OSError, ValueError) as error:
                answer = f"I could not prepare the local review previews for NVIDIA: `{error}`"
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "error": True},
                )
                store.log_activity("nvidia_file_prepare_failed", str(error), level="warning")
                return
        elif is_execution_request:
            try:
                user_content, online_path_aliases = online_action_content(clean_prompt)
            except (AssistantPathError, FileNotFoundError, OSError, ValueError) as error:
                answer = f"I could not prepare the local path aliases for planning: `{error}`"
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "error": True},
                )
                store.log_activity("nvidia_action_alias_failed", str(error), level="warning")
                return
        elif named_paths:
            try:
                evidence_plan = plan_online_evidence(clean_prompt)
                if not evidence_plan:
                    raise AssistantPathError(
                        "No supported image or CSV/TSV file was found at that path."
                    )
            except (AssistantPathError, FileNotFoundError, OSError, ValueError) as error:
                answer = f"I kept this request local and did not call NVIDIA. `{error}`"
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "error": True},
                )
                store.log_activity("nvidia_file_access_block", str(error), level="warning")
                return
            if not preapproval_enabled() and not online_share_approved:
                st.session_state[PENDING_ONLINE_SHARE_KEY] = {
                    "prompt": clean_prompt,
                    "page_id": page_id,
                    "conversation_id": conversation_id,
                    "user_message_id": user_message_id,
                    "files": evidence_plan,
                }
                answer = (
                    f"I found **{len(evidence_plan)} approved local file(s)** for online review. "
                    "Nothing has been sent yet. Review the list below and choose **Approve Online Share** "
                    "or **Cancel**."
                )
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "source": "online_share_waiting"},
                )
                store.log_activity(
                    "nvidia_file_share_waiting",
                    "Owner approval is required before sending bounded file evidence.",
                    details={"conversation_id": conversation_id, "files": evidence_plan},
                )
                return
            try:
                online_content, evidence_manifest = build_online_evidence_content(
                    clean_prompt,
                    model_supports_images=bool(
                        NVIDIA_MODEL_CATALOG[active_model_id]["supports_images"]
                    ),
                )
            except (AssistantPathError, FileNotFoundError, OSError, ValueError) as error:
                answer = f"I could not prepare the approved online evidence: `{error}`"
                store.add_message(
                    conversation_id,
                    "assistant",
                    answer,
                    metadata={"model": active_model_id, "error": True},
                )
                store.log_activity("nvidia_file_prepare_failed", str(error), level="warning")
                return
            user_content: Any = online_content
        else:
            user_content = clean_prompt
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": _online_system_prompt(
                    context,
                    _visual_state_context(thread_visuals, online_safe=True),
                    registered_tools=tool_manifest(clean_prompt, limit=12)[:24_000],
                    allow_action_plan=is_execution_request,
                ),
            },
        ]
        messages.extend(
            _online_safe_thread_history(store, conversation_id, user_message_id)
        )
        messages.append({"role": "user", "content": user_content})
    else:
        memories = store.relevant_verified_memories(clean_prompt)
        memory_text = "\n".join(f"- {item['content']}" for item in memories)
        prior_messages = store.relevant_prior_user_messages(
            clean_prompt,
            exclude_conversation_id=conversation_id,
            limit=4,
        )
        prior_chat_text = "\n".join(
            f"- [{item['created_at'][:10]} · {item['title'][:60]}] {item['content']}"
            for item in prior_messages
        )
        context_text = json.dumps(context, ensure_ascii=False, default=str, indent=2)
        allow_action_plan = bool(is_execution_request and not local_qc_note)
        if allow_action_plan:
            try:
                source_context = source_context_for_prompt(clean_prompt)
            except (OSError, ValueError, SyntaxError):
                source_context = ""
            registered_tools = tool_manifest(clean_prompt, limit=12)[:24_000]
        else:
            source_context = ""
            registered_tools = ""
        system_prompt = build_system_prompt(
            profile,
            page_context=context_text[:24_000],
            active_visual_context=_visual_state_context(thread_visuals)[:8_000],
            verified_memories=memory_text[:8_000],
            prior_chat_context=prior_chat_text[:4_000],
            source_context=source_context[:24_000],
            registered_tools=registered_tools,
            allow_action_plan=allow_action_plan,
        )
        messages = [{"role": "system", "content": system_prompt}]
        history = store.get_messages(conversation_id, limit=24)
        bounded_history: list[dict[str, Any]] = []
        remaining_characters = 20_000
        for item in reversed(history):
            if item["role"] not in {"user", "assistant"}:
                continue
            source = str((item.get("metadata") or {}).get("source") or "")
            if item["role"] == "assistant" and source in {
                "he_previews_generated",
                "image_previews_generated",
                "image_preview_failed",
                "image_preview_needs_channels",
                "image_preview_needs_source",
            }:
                continue
            content = str(item["content"])
            if bounded_history and len(content) > remaining_characters:
                break
            bounded_history.append({"role": item["role"], "content": content[:remaining_characters]})
            remaining_characters -= min(len(content), remaining_characters)
            if remaining_characters <= 0:
                break
        messages.extend(reversed(bounded_history))

    if image_files and not online_provider and messages and messages[-1]["role"] == "user":
        blocks: list[dict[str, Any]] = []
        try:
            for upload in image_files:
                if isinstance(upload, Path):
                    local_path = require_assistant_read_path(upload, must_exist=True)
                    raw = local_path.read_bytes()
                    upload_name = local_path.name
                    mime = "image/png"
                else:
                    raw = upload.getvalue()
                    upload_name = str(getattr(upload, "name", "Image"))
                    mime = str(getattr(upload, "type", "") or "image/png")
                if len(raw) > 20 * 1024 * 1024:
                    raise ValueError(f"{upload_name} is larger than 20 MB.")
                if not mime.startswith("image/"):
                    raise ValueError(f"{upload_name} is not an image.")
                encoded = base64.b64encode(raw).decode("ascii")
                blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            blocks.append({"type": "text", "text": clean_prompt + local_qc_note})
            messages[-1]["content"] = blocks
        except (OSError, ValueError) as error:
            answer = f"I could not use the uploaded image: {error}"
            store.add_message(conversation_id, "assistant", answer, metadata={"model": active_model_id, "error": True})
            store.log_activity("model_image_rejected", answer, level="warning")
            return

    if online_provider:
        response_token_limit = min(
            response_token_limit,
            int(NVIDIA_MODEL_CATALOG[active_model_id]["max_output_tokens"]),
        )
        budget_kind = "online_action_plan" if is_execution_request else "online_consultation"
    answer = ""
    try:
        if online_provider:
            client: LocalModelClient | NvidiaModelClient = NvidiaModelClient(
                model_id=active_model_id,
                timeout_seconds=settings.timeout_seconds,
            )
        else:
            _ensure_local_model_running()
            client = LocalModelClient(timeout_seconds=settings.timeout_seconds)
        if is_execution_request and not online_provider:
            health = client.health()
            server_limit = int(health.get("max_new_tokens") or 2048)
            response_token_limit = min(response_token_limit, server_limit)
        stream_placeholder = None
        if live and not is_execution_request:
            with st.chat_message("assistant"):
                stream_placeholder = st.empty()
                last_draw = 0.0
                if online_provider:
                    pieces = client.stream_complete(
                        messages,
                        temperature=settings.temperature,
                        max_tokens=response_token_limit,
                        enable_thinking=False,
                    )
                else:
                    pieces = client.stream_complete(
                        messages,
                        temperature=settings.temperature,
                        max_tokens=response_token_limit,
                    )
                for piece in pieces:
                    answer += piece
                    now = time.monotonic()
                    if now - last_draw >= 0.08:
                        stream_placeholder.markdown(answer + "▌")
                        last_draw = now
                stream_placeholder.markdown(
                    answer or "The model returned no visible text. Retrying once…"
                )
        elif is_execution_request:
            with st.spinner(
                "Building and checking the action plan privately. Internal job instructions are hidden…"
            ):
                answer = client.complete(
                    messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                )
        else:
            if online_provider:
                answer = client.complete(
                    messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                    enable_thinking=False,
                )
            else:
                answer = client.complete(
                    messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                )
        if not answer.strip():
            store.log_activity(
                "model_empty_response_retry",
                f"{active_model_id} returned no visible text; retrying once.",
                level="warning",
            )
            if online_provider and cluster_evidence is not None:
                messages = _cluster_text_only_messages(
                    messages,
                    clean_prompt,
                    str(cluster_evidence.get("note") or ""),
                )
                answer = client.complete(
                    messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                    enable_thinking=False,
                )
                store.log_activity(
                    "nvidia_cluster_text_fallback",
                    "The visual request returned no text; retried with the exact compact tables only.",
                    details={"model": active_model_id},
                )
            elif online_provider:
                answer = client.complete(
                    messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                    enable_thinking=False,
                )
            else:
                answer = client.complete(
                    messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                )
            if stream_placeholder is not None:
                stream_placeholder.markdown(
                    answer
                    or "The selected model returned an empty answer twice. Try again or switch models."
                )
        answer = answer.strip() or (
            "The selected model returned an empty answer twice. Try again or switch models."
        )
        if not is_execution_request and _advisory_answer_needs_repair(answer):
            # Some OpenAI-compatible model servers expose a model's internal
            # tool thought as ordinary text. Advisory chat has no executable
            # tool call at this point, so ask once for the actual user-facing
            # answer and never display the leaked pseudo-call as the result.
            repair_messages = [
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": (
                        "That was not a user-facing answer. It was empty or contained an internal "
                        "tool thought. Do not call, name, or simulate a tool. Use the current-tab "
                        "evidence already supplied and answer the user's question directly now. "
                        "If evidence is truly missing, name only the exact missing field."
                    ),
                },
            ]
            if online_provider:
                repaired = client.complete(
                    repair_messages,
                    temperature=0.0,
                    max_tokens=response_token_limit,
                    enable_thinking=False,
                )
            else:
                repaired = client.complete(
                    repair_messages,
                    temperature=0.0,
                    max_tokens=response_token_limit,
                )
            if repaired.strip() and not _advisory_answer_needs_repair(repaired):
                answer = repaired.strip()
                store.log_activity(
                    "advisory_answer_repaired",
                    "Replaced an empty or leaked internal tool thought with a direct answer.",
                    details={"model": active_model_id, "provider": provider_settings.provider},
                )
            else:
                answer = (
                    "The selected model did not produce a usable answer. Its response was empty or "
                    "contained an internal tool call, so SPACEMAP hid it. Please retry or switch models."
                )
            if stream_placeholder is not None:
                stream_placeholder.markdown(answer)
        if is_execution_request and not local_qc_note:
            # General SPACEMAP work uses the automatic source/tool registry. A
            # small model may first return prose or truncate its JSON. Give it
            # up to three short validator-guided attempts; this applies to every
            # registered backend tool and is not tied to one user phrase.
            for repair_attempt in range(1, 4):
                plan_error = ""
                try:
                    _unused_text, first_job = parse_job_envelope(answer)
                except GeneralActionError as error:
                    first_job = None
                    plan_error = str(error)
                if first_job is not None or _answer_is_waiting_for_input(answer):
                    break
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "The SPACEMAP action validator could not read your executable action"
                            + (f": {plan_error}. " if plan_error else ". ")
                            + "Use the registered tool catalog and source signatures already supplied. "
                            "If the required inputs are present, return one complete, compact "
                            "<spacemap_job> JSON object now. Use exact registered callables, keep the "
                            "Python short, read paths only from INPUTS, and write outputs only below "
                            "OUTPUT_DIR. Do not repeat the prose plan. If one essential input is truly "
                            "missing, ask only for that input."
                        ),
                    },
                ]
                if online_provider:
                    repaired = client.complete(
                        repair_messages,
                        temperature=0.0,
                        max_tokens=response_token_limit,
                        enable_thinking=False,
                    )
                else:
                    repaired = client.complete(
                        repair_messages,
                        temperature=0.0,
                        max_tokens=response_token_limit,
                    )
                if repaired.strip():
                    answer = repaired.strip()
                    store.log_activity(
                        "action_plan_repaired",
                        f"The universal planner used validator repair attempt {repair_attempt}.",
                        details={
                            "model": active_model_id,
                            "provider": provider_settings.provider,
                            "attempt": repair_attempt,
                            "validator_error": plan_error,
                        },
                    )
                else:
                    break
        if not local_qc_note:
            try:
                plan = None
                clean_answer = answer
                for validation_attempt in range(1, 3):
                    clean_answer, job_payload = parse_job_envelope(answer)
                    if job_payload is None:
                        break
                    restored_payload = (
                        restore_online_job_paths(job_payload, online_path_aliases)
                        if online_provider
                        else job_payload
                    )
                    try:
                        plan = build_general_plan(
                            title=str(restored_payload.get("title") or "SPACEMAP AI job"),
                            user_request=clean_prompt,
                            inputs=restored_payload.get("inputs") or {},
                            parameters=restored_payload.get("parameters") or {},
                            code=str(restored_payload.get("code") or ""),
                            output_dir=str(restored_payload.get("output_dir") or ""),
                            overwrite=bool(restored_payload.get("overwrite", False)),
                            max_runtime_seconds=int(restored_payload.get("max_runtime_seconds", 43_200)),
                            planner_provider=provider_settings.provider,
                            planner_model=active_model_id,
                            conversation_id=conversation_id,
                        )
                        break
                    except (GeneralActionError, OSError, TypeError, ValueError) as validation_error:
                        if validation_attempt >= 2:
                            raise
                        validator_messages = [
                            *messages,
                            {"role": "assistant", "content": answer},
                            {
                                "role": "user",
                                "content": (
                                    "The protected SPACEMAP validator rejected that job: "
                                    f"{validation_error}. Correct only the job. Return one complete, "
                                    "compact <spacemap_job> JSON object using exact registered functions "
                                    "and the exact source signatures already supplied. Keep every input "
                                    "path in inputs, every setting in parameters, and every output under "
                                    "OUTPUT_DIR."
                                ),
                            },
                        ]
                        if online_provider:
                            repaired = client.complete(
                                validator_messages,
                                temperature=0.0,
                                max_tokens=response_token_limit,
                                enable_thinking=False,
                            )
                        else:
                            repaired = client.complete(
                                validator_messages,
                                temperature=0.0,
                                max_tokens=response_token_limit,
                            )
                        if not repaired.strip():
                            raise validation_error
                        answer = repaired.strip()
                        store.log_activity(
                            "action_plan_validation_repair",
                            "The universal planner corrected a job rejected by the protected validator.",
                            details={
                                "model": active_model_id,
                                "provider": provider_settings.provider,
                                "validator_error": str(validation_error),
                            },
                        )
                if plan is not None:
                    created_action_id = plan.action_id
                    st.session_state[GENERAL_ACTION_ID_KEY] = plan.action_id
                    st.session_state[LIVE_ACTION_KEY] = plan.action_id
                    if preapproval_enabled():
                        pid = launch_general_action(
                            plan.action_id,
                            approval_method="Session pre-approval switch",
                        )
                        action_note = (
                            f"I created a protected plan and started it as process {pid}. "
                            "Open **Actions** to watch the real progress, previews, and results."
                        )
                        store.log_activity(
                            "general_action_started",
                            f"Protected AI job {plan.action_id[:8]} started from chat.",
                            details={"title": plan.title, "output_dir": plan.output_dir, "pid": pid},
                        )
                    else:
                        action_note = (
                            "I created a protected job plan. Open **Actions > All Other Tools** to review the "
                            "inputs, parameters, output folder, and exact code before starting it."
                        )
                        store.log_activity(
                            "general_action_planned",
                            f"Protected AI job {plan.action_id[:8]} prepared for review.",
                            details={"title": plan.title, "output_dir": plan.output_dir},
                        )
                    answer = (clean_answer + "\n\n" + action_note).strip()
            except (GeneralActionError, OSError, TypeError, ValueError) as error:
                try:
                    clean_answer, _payload = parse_job_envelope(answer)
                except GeneralActionError:
                    clean_answer = re.sub(
                        r"<spacemap_job>.*?(?:</spacemap_job>|$)",
                        "",
                        answer,
                        flags=re.DOTALL,
                    ).strip()
                answer = (
                    clean_answer
                    + "\n\nI did not run the proposed job because its safety check failed: "
                    + f"`{error}`"
                ).strip()
        store.log_activity(
            "model_response",
            (
                "NVIDIA answered from a privacy-limited workflow context."
                if online_provider
                else "Gemma answered from the registered page context."
            ),
            details={
                "model": active_model_id,
                "provider": provider_settings.provider,
                "image_count": len(image_files),
                "completion_budget": response_token_limit,
                "budget_kind": budget_kind,
            },
        )
    except ModelUnavailable as error:
        recovered_from_online = False
        recovery_error = str(error)
        if online_provider and cluster_evidence is not None:
            text_messages = _cluster_text_only_messages(
                messages,
                clean_prompt,
                str(cluster_evidence.get("note") or ""),
            )
            try:
                fallback_client = NvidiaModelClient(
                    model_id=active_model_id,
                    timeout_seconds=min(90, settings.timeout_seconds),
                )
                recovered = fallback_client.complete(
                    text_messages,
                    temperature=settings.temperature,
                    max_tokens=response_token_limit,
                    enable_thinking=False,
                ).strip()
                if not recovered:
                    raise ModelUnavailable("The compact NVIDIA retry also returned no visible text.")
                answer = (
                    "*The NVIDIA visual request failed, so I retried with the exact marker and spatial "
                    "tables only.*\n\n" + recovered
                )
                recovered_from_online = True
                store.log_activity(
                    "nvidia_cluster_recovered_text_only",
                    "Recovered the cluster answer with a compact text-only NVIDIA request.",
                    details={"model": active_model_id, "original_error": str(error)},
                )
            except (ModelUnavailable, OSError, ValueError) as fallback_error:
                recovery_error = str(fallback_error)
                try:
                    local_client = LocalModelClient(timeout_seconds=settings.timeout_seconds)
                    recovered = local_client.complete(
                        text_messages,
                        temperature=settings.temperature,
                        max_tokens=min(2_048, response_token_limit),
                    ).strip()
                    if not recovered:
                        raise ModelUnavailable("Local Gemma also returned no visible text.")
                    answer = (
                        "*NVIDIA did not answer, so local Gemma completed this request using the same "
                        "marker and spatial tables. No additional data was sent online.*\n\n" + recovered
                    )
                    recovered_from_online = True
                    store.log_activity(
                        "nvidia_cluster_recovered_locally",
                        "Recovered the cluster answer with local Gemma after NVIDIA failed.",
                        details={
                            "model": active_model_id,
                            "original_error": str(error),
                            "online_fallback_error": str(fallback_error),
                        },
                    )
                except (ModelUnavailable, OSError, ValueError) as local_error:
                    recovery_error = f"{fallback_error}; local fallback: {local_error}"
            if recovered_from_online and stream_placeholder is not None:
                stream_placeholder.markdown(answer)
        if online_provider and not recovered_from_online:
            answer = (
                "I saved your question, but the NVIDIA hosted model did not answer. Open "
                "**AI setup > Answer with an online model instead** to check the key and test the connection.\n\n"
                f"Connection detail: `{recovery_error}`"
            )
        elif not online_provider:
            if qc_previews:
                answer = (
                    f"I completed the local preparation and displayed {len(qc_previews)} annotation "
                    "review image(s), but Gemma's visual judgment failed. The files do not need to be "
                    "prepared again; use **Retry Last Request** after the model is ready.\n\n"
                    f"Connection detail: `{error}`"
                )
            else:
                answer = (
                    "I saved your question, but the local model could not answer. "
                    f"**AI setup** (Tab 0 > AI Assistant, or Setup on this page) shows what is missing.\n\n"
                    f"Detail: `{error}`"
                )
        if not recovered_from_online:
            store.log_activity("model_unavailable", recovery_error, level="warning")
    store.add_message(
        conversation_id,
        "assistant",
        answer,
        metadata={
            "model": active_model_id,
            "memory_count": len(memories),
            "instruction_version": profile.version,
            "page_context": context.get("page_id"),
            "image_names": image_names,
            "online_evidence_count": len(evidence_manifest) if online_provider else 0,
            "action_id": created_action_id,
        },
    )
    if online_provider and evidence_manifest:
        store.log_activity(
            "nvidia_file_share_attempted",
            f"Included {len(evidence_manifest)} bounded owner-approved evidence item(s) in the NVIDIA request.",
            details={"conversation_id": conversation_id, "items": evidence_manifest},
        )


def _render_messages(store: AssistantStore, conversation_id: str, *, compact: bool) -> None:
    messages = store.get_messages(conversation_id, limit=10 if compact else MAX_CHAT_MESSAGES)
    if not messages:
        st.info("Ask about the current page, settings, results, or what to do next.")
        return
    for message in messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_conversation_previews(
    store: AssistantStore,
    conversation_id: str,
    *,
    compact: bool,
) -> None:
    """Show the newest requested preview set directly in Chat and the side panel."""
    candidates = [
        item
        for item in store.list_artifacts(kind="assistant_image_preview", limit=200)
        if str((item.get("metadata") or {}).get("conversation_id") or "") == conversation_id
    ]
    if not candidates:
        return
    newest_request = str((candidates[0].get("metadata") or {}).get("request_message_id") or "")
    latest_user_message = next(
        (
            item
            for item in reversed(store.get_messages(conversation_id, limit=20))
            if item.get("role") == "user"
        ),
        None,
    )
    latest_completed_action = next(
        (
            str((item.get("metadata") or {}).get("action_id") or "")
            for item in reversed(store.get_messages(conversation_id, limit=30))
            if item.get("role") == "assistant"
            and (item.get("metadata") or {}).get("source") == "background_action_completed"
        ),
        "",
    )
    # Do not leave an old image result under an unrelated newer answer. Preview
    # files remain in chat history/storage, but the active gallery belongs only
    # to the latest user request.
    request_is_latest = bool(
        latest_user_message and newest_request == str(latest_user_message.get("id") or "")
    )
    action_is_latest = bool(
        latest_completed_action
        and any(
            str((item.get("metadata") or {}).get("action_id") or "")
            == latest_completed_action
            for item in candidates
        )
    )
    if not request_is_latest and not action_is_latest:
        return
    if action_is_latest and not request_is_latest:
        artifacts = [
            item
            for item in candidates
            if str((item.get("metadata") or {}).get("action_id") or "")
            == latest_completed_action
        ]
    else:
        artifacts = [
            item
            for item in candidates
            if str((item.get("metadata") or {}).get("request_message_id") or "") == newest_request
        ]
    active_paths = {
        str(item.get("preview_path") or "")
        for item in _recent_preview_states(store, conversation_id)
    }
    active_artifacts = [item for item in artifacts if str(item.get("path") or "") in active_paths]
    if active_artifacts:
        artifacts = active_artifacts
    existing: list[tuple[dict[str, Any], Path]] = []
    seen: set[str] = set()
    for artifact in artifacts:
        path = Path(str(artifact.get("path") or ""))
        if path.is_file() and str(path) not in seen:
            seen.add(str(path))
            existing.append((artifact, path))
    if not existing:
        return
    existing.sort(key=lambda pair: str((pair[0].get("metadata") or {}).get("source_path") or ""))

    st.markdown("#### Requested image previews")
    displayed_modes = {
        str((artifact.get("metadata") or {}).get("preview_mode") or "he")
        for artifact, _path in existing
    }
    if displayed_modes == {"he"}:
        st.caption("H&E · HE_Red + HE_Green + HE_Blue")
    elif displayed_modes == {"non_he"}:
        st.caption("Representative CODEX/non-H&E composite")
    elif displayed_modes == {"markers"}:
        st.caption("Requested image channel view")
    elif displayed_modes <= {"qc_he", "qc_image"}:
        st.caption("Annotation QC · overview and numbered polygon tiles")
    else:
        st.caption("Requested H&E and CODEX/channel views")
    columns = [st.container()] if compact else list(st.columns(min(2, len(existing))))
    for index, (artifact, path) in enumerate(existing):
        metadata = artifact.get("metadata") or {}
        source_name = Path(str(metadata.get("source_path") or path.name)).name
        target = columns[0] if compact else columns[index % len(columns)]
        with target:
            st.image(str(path), caption=source_name, width="stretch")
            st.caption(
                f"Preview {metadata.get('preview_width', '?')} × {metadata.get('preview_height', '?')} · "
                f"pyramid level {metadata.get('pyramid_level', '?')}"
            )


def _render_online_session_notice() -> None:
    settings = load_provider_settings()
    if settings.provider != NVIDIA_PROVIDER:
        st.caption("Model: Local Gemma · offline")
        return
    model_label = str(NVIDIA_MODEL_CATALOG[settings.nvidia_model_id]["label"])
    st.warning(
        f"Online model: **{model_label}**. Your question, a small generic tab description, and up to "
        "8 recent messages from this open chat thread will go to NVIDIA. Local paths are removed. "
        "A file is sent only when you name it in the message and its "
        "folder has separate Owner-approved online access. Large images become small PNG previews; "
        "tables become limited row samples. Source code, credentials, saved memory, and other chat threads are not sent."
    )
    st.checkbox(
        "For this session, I will send only public, synthetic, deidentified, or institutionally approved data—no confidential data or PHI.",
        key=NVIDIA_SESSION_CONSENT_KEY,
    )
    st.caption(
        "Action approval mode also controls file sharing: Ask waits for your button click; Auto sends "
        "the bounded copy after all checks. Change it under Actions."
    )
    if not nvidia_api_key_saved():
        st.caption("No NVIDIA API key is saved yet. Add it under Model.")


def _quick_model_options() -> list[str]:
    """Return stable values for the chat-level model selector."""
    return [LOCAL_PROVIDER, *(f"{NVIDIA_PROVIDER}:{model_id}" for model_id in NVIDIA_MODEL_CATALOG)]


def _quick_model_value() -> str:
    settings = load_provider_settings()
    if settings.provider == NVIDIA_PROVIDER:
        return f"{NVIDIA_PROVIDER}:{settings.nvidia_model_id}"
    return LOCAL_PROVIDER


def _quick_model_label(value: str) -> str:
    if value == LOCAL_PROVIDER:
        return "Local · Google Gemma 4 12B"
    _provider, model_id = value.split(":", 1)
    return f"Online · {NVIDIA_MODEL_CATALOG[model_id]['label']}"


def _last_user_request(store: AssistantStore, conversation_id: str) -> str:
    """Return the newest user request in this chat only."""
    messages = store.get_messages(conversation_id, limit=200)
    return next(
        (str(item["content"]) for item in reversed(messages) if item["role"] == "user"),
        "",
    )


def _render_quick_model_switch(
    page_id: str,
    store: AssistantStore,
    conversation_id: str,
    *,
    compact: bool,
) -> None:
    """Switch models inside chat without changing the active conversation."""
    current_value = _quick_model_value()
    options = _quick_model_options()
    location = "side" if compact else "full"
    with st.container(border=True):
        if compact:
            st.caption("Model for this chat")
            selector_area = st.container()
            retry_area = st.container()
        else:
            selector_area, retry_area = st.columns([3, 1])
        with selector_area:
            selected = st.selectbox(
                "Model for this chat",
                options,
                index=options.index(current_value),
                format_func=_quick_model_label,
                key=f"assistant_quick_model_{page_id}_{location}_{current_value}",
                label_visibility="collapsed" if compact else "visible",
                help=(
                    "Changing this keeps the same saved chat, earlier messages, and chat-linked images. "
                    "API models still require NVIDIA consent and approved data-sharing folders."
                ),
            )
        if selected != current_value:
            previous_label = _quick_model_label(current_value)
            if selected == LOCAL_PROVIDER:
                saved = save_provider_settings(
                    LOCAL_PROVIDER,
                    load_provider_settings().nvidia_model_id,
                )
            else:
                _provider, model_id = selected.split(":", 1)
                saved = save_provider_settings(NVIDIA_PROVIDER, model_id)
            store.log_activity(
                "chat_model_switched",
                f"Changed chat model from {previous_label} to {_quick_model_label(selected)}.",
                details={
                    "conversation_id": conversation_id,
                    "provider": saved.provider,
                    "model": (
                        saved.nvidia_model_id
                        if saved.provider == NVIDIA_PROVIDER
                        else MODEL_ID
                    ),
                },
            )
            st.rerun()

        last_request = _last_user_request(store, conversation_id)
        with retry_area:
            retry = st.button(
                "Retry Last Request",
                width="stretch",
                disabled=not bool(last_request),
                key=f"assistant_retry_last_{page_id}_{location}_{current_value}",
                help=(
                    "Ask the selected model to answer your newest request again. You do not need to "
                    "copy it. The same chat history remains available."
                ),
            )
        st.caption("Switching models does not start a new chat or erase its context.")
    if retry and last_request:
        _submit_prompt(last_request, page_id, store, conversation_id, live=True)
        st.rerun()


def _render_pending_online_share(
    store: AssistantStore,
    conversation_id: str,
) -> None:
    pending = st.session_state.get(PENDING_ONLINE_SHARE_KEY)
    if not isinstance(pending, dict) or pending.get("conversation_id") != conversation_id:
        return
    files = list(pending.get("files") or [])
    with st.container(border=True):
        st.markdown("#### Online file review is waiting for you")
        st.caption("Nothing in this list has been sent yet.")
        st.dataframe(
            [
                {"Local file": item.get("path", ""), "What will be sent": item.get("kind", "")}
                for item in files
            ],
            hide_index=True,
            width="stretch",
        )
        approve, cancel = st.columns(2)
        with approve:
            if st.button(
                "Approve Online Share",
                type="primary",
                width="stretch",
                key=f"approve_online_share_{conversation_id}",
            ):
                request = dict(pending)
                st.session_state.pop(PENDING_ONLINE_SHARE_KEY, None)
                _submit_prompt(
                    str(request.get("prompt") or ""),
                    str(request.get("page_id") or "assistant"),
                    store,
                    conversation_id,
                    live=True,
                    record_user_message=False,
                    online_share_approved=True,
                    existing_user_message_id=str(request.get("user_message_id") or ""),
                )
                st.rerun()
        with cancel:
            if st.button(
                "Cancel",
                width="stretch",
                key=f"cancel_online_share_{conversation_id}",
            ):
                st.session_state.pop(PENDING_ONLINE_SHARE_KEY, None)
                store.add_message(
                    conversation_id,
                    "assistant",
                    "Cancelled. No file content was sent online.",
                    metadata={"source": "online_share_cancelled"},
                )
                store.log_activity(
                    "nvidia_file_share_cancelled",
                    "Owner cancelled the pending online file share.",
                    details={"conversation_id": conversation_id},
                )
                st.rerun()


def _start_new_chat(store: AssistantStore) -> None:
    conversation_id = store.create_conversation(
        project_root=str(st.session_state.get("active_project_root") or "")
    )
    st.session_state[CONVERSATION_KEY] = conversation_id
    st.session_state[CHAT_UPLOAD_COUNTER_KEY] = int(
        st.session_state.get(CHAT_UPLOAD_COUNTER_KEY, 0)
    ) + 1
    store.log_activity(
        "chat_created",
        f"Started new conversation {conversation_id[:8]}.",
        details={"conversation_id": conversation_id},
    )


def _render_conversation_controls(
    page_id: str,
    store: AssistantStore,
    conversation_id: str,
    *,
    compact: bool,
) -> None:
    current = store.get_conversation(conversation_id) or {}
    columns = st.columns([1, 2])
    with columns[0]:
        if st.button(
            "New Chat",
            icon=":material/add:",
            width="stretch",
            key=f"assistant_new_chat_{page_id}_{'compact' if compact else 'full'}",
        ):
            _start_new_chat(store)
            st.rerun()
    with columns[1]:
        title = str(current.get("title") or "New conversation")
        st.caption(f"Current chat: {title[:80]}")

    with st.expander("Chat History", expanded=False):
        conversations = store.list_conversations(limit=100)
        labels = {
            (
                f"{item['title'][:70]} · {int(item.get('message_count') or 0)} messages · "
                f"{item['updated_at'][:19]} · {item['id'][:8]}"
            ): item["id"]
            for item in conversations
        }
        options = list(labels)
        index = next(
            (position for position, label in enumerate(options) if labels[label] == conversation_id),
            0,
        )
        selected = st.selectbox(
            "Saved chats",
            options,
            index=index,
            key=f"assistant_chat_history_{page_id}_{'compact' if compact else 'full'}",
        )
        if st.button(
            "Open Selected Chat",
            width="stretch",
            key=f"assistant_open_chat_{page_id}_{'compact' if compact else 'full'}",
        ):
            st.session_state[CONVERSATION_KEY] = labels[selected]
            st.rerun()
        delete_confirmed = st.checkbox(
            "I want to delete this saved chat",
            key=(
                f"assistant_delete_chat_confirm_{page_id}_"
                f"{'compact' if compact else 'full'}_{labels[selected]}"
            ),
        )
        delete_exports = st.checkbox(
            "Also delete Markdown/JSONL copies previously saved in SPACEMAP_AI",
            key=(
                f"assistant_delete_chat_exports_{page_id}_"
                f"{'compact' if compact else 'full'}_{labels[selected]}"
            ),
        )
        if st.button(
            "Delete Selected Chat",
            width="stretch",
            disabled=not delete_confirmed,
            key=(
                f"assistant_delete_chat_{page_id}_"
                f"{'compact' if compact else 'full'}_{labels[selected]}"
            ),
        ):
            selected_id = labels[selected]
            removed_exports = (
                store.delete_conversation_exports(selected_id) if delete_exports else []
            )
            deleted = store.delete_conversation(selected_id)
            store.log_activity(
                "chat_deleted",
                f"Deleted saved chat {selected_id[:8]} and its gallery links.",
                details={
                    "conversation_id": selected_id,
                    "export_files_deleted": len(removed_exports),
                    **deleted,
                },
            )
            if selected_id == conversation_id:
                _start_new_chat(store)
            st.success("The chat was deleted. Scientific output files were not deleted.")
            st.rerun()
        st.caption(
            "Chats are saved automatically. Deleting a chat removes its messages and gallery links, "
            "but never deletes TIFF, CSV, PNG, or other scientific files."
        )
        st.code(
            f"Chat database: {store.database_path}\n"
            f"Saved exports: {ASSISTANT_DATA_DIR / 'exports'}",
            language="text",
        )


def _render_chat(
    page_id: str,
    *,
    compact: bool = False,
    show_online_consent: bool = True,
) -> None:
    store = _store()
    conversation_id = _conversation_id(store)
    _render_conversation_controls(page_id, store, conversation_id, compact=compact)
    _render_chat_action_banner(page_id)
    _render_quick_model_switch(
        page_id,
        store,
        conversation_id,
        compact=compact,
    )
    if show_online_consent:
        _render_online_session_notice()
    elif load_provider_settings().provider == NVIDIA_PROVIDER:
        consent = bool(st.session_state.get(NVIDIA_SESSION_CONSENT_KEY))
        st.caption(
            "Online session approval is on." if consent else
            "Online session approval is off. Turn it on in the main Chat tab before sending data online."
        )
    _render_messages(store, conversation_id, compact=compact)
    _render_conversation_previews(store, conversation_id, compact=compact)
    _render_pending_online_share(store, conversation_id)

    if compact:
        with st.form(f"assistant_compact_chat_{page_id}", clear_on_submit=True):
            prompt = st.text_area(
                "Message",
                placeholder="Ask about this page…",
                height=80,
                key=f"assistant_compact_prompt_{page_id}",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", width="stretch")
        if submitted and prompt.strip():
            _submit_prompt(prompt, page_id, store, conversation_id, live=True)
            st.rerun()
    else:
        counter = int(st.session_state.get(CHAT_UPLOAD_COUNTER_KEY, 0))
        online_provider = load_provider_settings().provider == NVIDIA_PROVIDER
        uploads = st.file_uploader(
            "Optional images for local Gemma to review",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            key=f"assistant_chat_images_{counter}",
            help="Use up to 4 PNG/JPEG previews or small TIFF images. Large whole-slide images should be converted to QC previews first.",
            disabled=online_provider,
        )
        prompt = st.chat_input("Ask SPACEMAP AI…", key="assistant_workspace_chat_input")
        if prompt:
            _submit_prompt(
                str(prompt),
                page_id,
                store,
                conversation_id,
                live=True,
                uploaded_images=list(uploads or []),
            )
            st.session_state[CHAT_UPLOAD_COUNTER_KEY] = counter + 1
            st.rerun()


def _render_chat_exports() -> None:
    store = _store()
    conversation_id = _conversation_id(store)
    messages = store.get_messages(conversation_id, limit=100_000)
    if not messages:
        st.caption("The conversation can be exported after the first message.")
        return
    markdown = store.conversation_markdown(conversation_id)
    jsonl = store.conversation_jsonl(conversation_id)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download Markdown",
            markdown,
            file_name=f"spacemap_ai_chat_{conversation_id[:8]}.md",
            mime="text/markdown",
            width="stretch",
        )
    with col2:
        st.download_button(
            "Download JSONL",
            jsonl,
            file_name=f"spacemap_ai_chat_{conversation_id[:8]}.jsonl",
            mime="application/x-ndjson",
            width="stretch",
        )
    with col3:
        if st.button("Save Both in SPACEMAP_AI", width="stretch"):
            paths = store.save_conversation_exports(conversation_id, "spacemap_ai_chat")
            st.success("Saved: " + ", ".join(str(path) for path in paths.values()))


def _render_recent_artifacts(conversation_id: str) -> None:
    store = _store()
    artifacts = [
        item
        for item in store.list_artifacts(limit=500)
        if str((item.get("metadata") or {}).get("conversation_id") or "")
        == conversation_id
    ][:12]
    if not artifacts:
        return
    with st.expander("Images and outputs from this chat", expanded=False):
        st.caption("These gallery items belong only to the open chat.")
        shown = 0
        for artifact in artifacts:
            path = Path(str(artifact.get("path") or ""))
            if not path.is_file():
                continue
            caption = str(artifact.get("caption") or path.name)
            if artifact.get("kind") == "registration_qc_image" or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(path), caption=caption, width="stretch")
                shown += 1
            elif path.suffix.lower() in {".csv", ".tsv"}:
                st.markdown(f"**{caption}**")
                try:
                    import pandas as pd

                    separator = "\t" if path.suffix.lower() == ".tsv" else ","
                    preview = pd.read_csv(path, sep=separator, nrows=25)
                    st.dataframe(preview, hide_index=True, width="stretch")
                except Exception:
                    st.code(str(path), language="text")
                shown += 1
            else:
                st.markdown(f"**{caption}**")
                st.code(str(path), language="text")
            if shown >= 4:
                break
        if st.button(
            "Clear Preview Thumbnails from This Chat",
            width="stretch",
            key=f"assistant_clear_previews_{conversation_id}",
        ):
            removed = store.delete_conversation_artifacts(
                conversation_id,
                kind="assistant_image_preview",
            )
            store.log_activity(
                "chat_previews_cleared",
                f"Cleared {removed} preview link(s) from chat {conversation_id[:8]}.",
                details={"conversation_id": conversation_id, "removed": removed},
            )
            st.success("Preview thumbnails were cleared from this chat. Image files were kept.")
            st.rerun()


def _render_instruction_editor() -> None:
    profile = load_instruction_profile()
    st.markdown("### Positive and negative instructions")
    st.caption(
        "Positive instructions tell the AI what to do. Negative instructions tell it what to avoid. "
        "They are added to every local model request."
    )
    with st.form("assistant_instruction_form"):
        positive = st.text_area(
            "Positive instructions",
            value=profile.positive,
            height=230,
            help="Examples: use simple language; explain evidence; show a plan first.",
        )
        negative = st.text_area(
            "Negative instructions",
            value=profile.negative,
            height=230,
            help="Examples: do not overwrite files; do not guess labels; do not hide failed QC.",
        )
        saved = st.form_submit_button("Save Instructions", type="primary")
    if saved:
        try:
            new_profile = save_instruction_profile(positive, negative)
            st.success(f"Instructions saved as version {new_profile.version}.")
        except ValueError as error:
            st.error(str(error))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset to Safe Defaults", width="stretch"):
            reset_instruction_profile()
            st.rerun()
    with col2:
        instruction_json = json.dumps(
            {
                "positive": profile.positive,
                "negative": profile.negative,
                "version": profile.version,
                "updated_at": profile.updated_at,
            },
            indent=2,
        )
        st.download_button(
            "Export Instructions",
            instruction_json,
            file_name="spacemap_ai_instructions.json",
            mime="application/json",
            width="stretch",
        )
    with st.expander("Fixed safety policy", expanded=False):
        st.caption("This part cannot be edited from chat or from the instruction boxes.")
        st.code(CORE_SAFETY_POLICY, language="text")


def _render_memory() -> None:
    store = _store()
    st.markdown("### Assistant memory")
    st.info(
        "Chats are saved automatically. When your message looks like a reusable preference or lesson, "
        "it is also saved here as **Proposed**. Proposed items are not used until you press "
        "**Accept & Use**. This prevents one mistake from becoming a permanent rule."
    )
    with st.expander("How memory works", expanded=True):
        st.markdown(
            "1. You teach the Assistant in chat, for example: `Remember that CD3 is our T-cell marker.`\n"
            "2. The lesson appears below as **Proposed**.\n"
            "3. Read it. You can edit it, accept it, or reject it.\n"
            "4. Only **Verified** memory is added to future relevant questions.\n"
            "5. **Rejected / old** memory is kept for the audit record but is not used.\n\n"
            "Normal tab settings do not need to fill memory: the Assistant reads the current page and "
            "project settings directly. Memory is for reusable knowledge, preferences, labels, and lessons."
        )

    if st.button("Find Useful Lessons in All Chats", width="stretch"):
        added = 0
        for message in store.list_user_messages(limit=5_000):
            for suggestion in suggest_memories(message["content"]):
                memory_id = store.add_memory_if_new(
                    suggestion.content,
                    scope="project",
                    kind=suggestion.kind,
                    status="proposed",
                    confidence=suggestion.confidence,
                    evidence=[
                        f"conversation:{message['conversation_id']}",
                        f"message:{message['id']}",
                    ],
                    source="chat_history_suggestion",
                )
                added += int(memory_id is not None)
        if added:
            st.success(f"Found {added} new possible lesson(s). Review them below.")
        else:
            st.info("No new clear, reusable lessons were found in saved chats.")

    with st.form("assistant_add_memory", clear_on_submit=True):
        content = st.text_area("New lesson or preference", height=100)
        col1, col2, col3 = st.columns(3)
        with col1:
            scope = st.selectbox("Scope", ["project", "lab", "run"])
        with col2:
            kind = st.selectbox("Type", ["lesson", "preference", "marker_alias", "recipe_note"])
        with col3:
            status = st.selectbox("Status", ["proposed", "verified"])
        add = st.form_submit_button("Add Memory")
    if add and content.strip():
        store.add_memory(content, scope=scope, kind=kind, status=status, source="user")
        st.success("Memory added.")
        st.rerun()

    memories = store.list_memories()
    if not memories:
        st.info("No memories have been saved yet.")
    else:
        proposed_count = sum(item["status"] == "proposed" for item in memories)
        verified_count = sum(item["status"] == "verified" for item in memories)
        rejected_count = sum(item["status"] == "superseded" for item in memories)
        counts = st.columns(3)
        counts[0].metric("Needs review", proposed_count)
        counts[1].metric("In use", verified_count)
        counts[2].metric("Rejected / old", rejected_count)
        display = [
            {
                "ID": item["id"][:8],
                "Scope": item["scope"],
                "Type": item["kind"],
                "Status": item["status"],
                "Memory": item["content"],
                "Updated": item["updated_at"],
            }
            for item in memories
        ]
        st.dataframe(display, hide_index=True, width="stretch")
        selectable = {f"{item['id'][:8]} — {item['content'][:70]}": item["id"] for item in memories}
        selected_label = st.selectbox("Memory to review", list(selectable))
        selected_id = selectable[selected_label]
        selected = next(item for item in memories if item["id"] == selected_id)
        scopes = ["project", "lab", "run"]
        kinds = ["lesson", "preference", "marker_alias", "recipe_note"]
        with st.form(f"assistant_review_memory_{selected_id}"):
            edited_content = st.text_area("Memory text", value=selected["content"], height=100)
            col1, col2 = st.columns(2)
            with col1:
                edited_scope = st.selectbox(
                    "Scope",
                    scopes,
                    index=scopes.index(selected["scope"]) if selected["scope"] in scopes else 0,
                )
            with col2:
                edited_kind = st.selectbox(
                    "Type",
                    kinds,
                    index=kinds.index(selected["kind"]) if selected["kind"] in kinds else 0,
                )
            buttons = st.columns(3)
            save_edit = buttons[0].form_submit_button("Save Edit", width="stretch")
            accept = buttons[1].form_submit_button("Accept & Use", type="primary", width="stretch")
            reject = buttons[2].form_submit_button("Reject", width="stretch")
        if save_edit or accept:
            store.update_memory(
                selected_id,
                edited_content,
                scope=edited_scope,
                kind=edited_kind,
            )
        if accept:
            store.set_memory_status(selected_id, "verified")
            st.success("Memory accepted. The Assistant can now use it when relevant.")
            st.rerun()
        if reject:
            store.set_memory_status(selected_id, "superseded")
            st.info("Memory rejected. It will not be used.")
            st.rerun()
        if save_edit:
            st.success("Memory updated. Its current review status was kept.")
            st.rerun()
    with st.expander("Transfer Memory", expanded=False):
        st.caption(
            "Export makes a small portable file. Imported memories return as Proposed so you can "
            "check them before a new Agent uses them."
        )
        st.download_button(
            "Export Memory JSONL",
            store.export_memories_jsonl(),
            file_name="spacemap_ai_memory.jsonl",
            mime="application/x-ndjson",
            width="stretch",
        )
        memory_upload = st.file_uploader(
            "Import Memory JSONL",
            type=["jsonl"],
            key="assistant_memory_import",
        )
        if st.button(
            "Import as Proposed",
            disabled=memory_upload is None,
            width="stretch",
        ):
            try:
                imported = store.import_memories_jsonl(memory_upload.getvalue().decode("utf-8"))
                st.success(f"Imported {imported} new memory item(s) for review.")
            except (UnicodeDecodeError, ValueError) as error:
                st.error(str(error))


def _render_provider_controls() -> None:
    current = load_provider_settings()
    st.markdown("### Choose the chat model")
    st.info(
        "**Local Gemma** is private and uses your GPU. **NVIDIA hosted API** runs on NVIDIA's "
        "computers, so your local GPU is not used for the answer."
    )
    provider_options = [LOCAL_PROVIDER, NVIDIA_PROVIDER]
    model_options = list(NVIDIA_MODEL_CATALOG)
    with st.form("assistant_provider_settings"):
        provider = st.radio(
            "Provider",
            provider_options,
            index=provider_options.index(current.provider),
            format_func=lambda value: PROVIDER_LABELS[value],
        )
        nvidia_model_id = st.selectbox(
            "NVIDIA model",
            model_options,
            index=model_options.index(current.nvidia_model_id)
            if current.nvidia_model_id in model_options
            else model_options.index(DEFAULT_NVIDIA_MODEL_ID),
            format_func=lambda value: str(NVIDIA_MODEL_CATALOG[value]["label"]),
            help="This choice is used only when NVIDIA hosted API is selected.",
        )
        st.caption(str(NVIDIA_MODEL_CATALOG[nvidia_model_id]["description"]))
        save_choice = st.form_submit_button("Save Model Choice", type="primary")
    if save_choice:
        saved = save_provider_settings(provider, nvidia_model_id)
        _store().log_activity(
            "model_provider_saved",
            f"Chat provider changed to {PROVIDER_LABELS[saved.provider]}.",
            details={"provider": saved.provider, "model": saved.nvidia_model_id},
        )
        st.success(f"Saved: {PROVIDER_LABELS[saved.provider]}")
        st.rerun()

    st.markdown("### NVIDIA API key")
    if nvidia_api_key_saved():
        st.success("An NVIDIA API key is saved in a private local file (owner-only permission).")
    else:
        st.warning("No NVIDIA API key is saved.")
    with st.form("assistant_nvidia_key", clear_on_submit=True):
        api_key = st.text_input(
            "Enter or replace NVIDIA API key",
            type="password",
            placeholder="nvapi-…",
            help="The key is never written to chat, activity logs, or chat exports.",
        )
        save_key = st.form_submit_button("Save API Key")
    if save_key:
        try:
            save_nvidia_api_key(api_key)
            _store().log_activity("nvidia_api_key_saved", "An NVIDIA API key was saved securely.")
            st.success("API key saved.")
            st.rerun()
        except (OSError, ValueError) as error:
            st.error(str(error))

    selected = load_provider_settings()
    if st.button(
        "Test NVIDIA Connection (small request, uses one API call)",
        disabled=not nvidia_api_key_saved(),
        width="stretch",
    ):
        started = time.monotonic()
        try:
            with st.spinner("Testing without extended reasoning; this may wait up to 3 minutes…"):
                test_nvidia_connection(selected.nvidia_model_id)
            elapsed = time.monotonic() - started
            _store().log_activity(
                "nvidia_connection_test",
                "The selected NVIDIA hosted model answered the connection test.",
                details={"model": selected.nvidia_model_id, "elapsed_seconds": round(elapsed, 2)},
            )
            st.success(f"Connection works. The selected NVIDIA model answered in {elapsed:.1f} seconds.")
        except (ModelUnavailable, OSError, ValueError) as error:
            _store().log_activity("nvidia_connection_failed", str(error), level="warning")
            st.error(f"Connection failed: {error}")
    st.caption(
        "The test disables extended reasoning and asks for a very short answer. A GPU does "
        "not speed this up because the model runs on NVIDIA's hosted service."
    )

    with st.expander("Remove the saved NVIDIA key", expanded=False):
        st.caption("This removes only the saved key. It does not remove models, chats, or SPACEMAP data.")
        if st.button("Remove Saved API Key", disabled=not nvidia_api_key_saved(), width="stretch"):
            try:
                delete_nvidia_api_key()
                _store().log_activity("nvidia_api_key_removed", "The saved NVIDIA API key was removed.")
                st.success("Saved NVIDIA API key removed.")
                st.rerun()
            except OSError as error:
                st.error(str(error))

    st.error(
        "NVIDIA's trial service must not receive confidential data, PHI, personal data, or sensitive "
        "human-subject research data. Follow your institution's approval process before "
        "using real study content."
    )
    st.caption(
        "Online mode sends the current question plus bounded recent context from this open chat. "
        "Project files, local paths, saved memory, other chats, and source code stay local unless an "
        "Owner-approved bounded file preview is explicitly requested."
    )


def _render_model_settings() -> None:
    provider = load_provider_settings()
    st.markdown("### Answer settings")
    if provider.provider == NVIDIA_PROVIDER:
        st.info(f"Current model: {NVIDIA_MODEL_CATALOG[provider.nvidia_model_id]['label']} · online")
    else:
        st.info(GEMMA_NOTICE)
    settings = load_model_settings()
    with st.form("gemma_model_settings"):
        temperature = st.slider(
            "Answer creativity",
            min_value=0.0,
            max_value=1.5,
            value=float(settings.temperature),
            step=0.05,
            help="Lower is more repeatable. 0.2 is a good scientific default.",
        )
        max_tokens = st.number_input(
            "Maximum normal chat tokens",
            min_value=32,
            max_value=2048,
            value=int(settings.max_tokens),
            step=32,
        )
        timeout_seconds = st.number_input(
            "Maximum wait (seconds)",
            min_value=30,
            max_value=1800,
            value=int(settings.timeout_seconds),
            step=30,
        )
        if st.form_submit_button("Save Answer Settings", type="primary"):
            saved = save_model_settings(float(temperature), int(max_tokens), int(timeout_seconds))
            st.success(f"Saved. Normal answers are limited to {saved.max_tokens} tokens.")

    st.info(
        f"Scientific action plans automatically receive up to {ACTION_PLAN_MAX_TOKENS:,} tokens. "
        f"Complex comparisons and parameter searches receive up to "
        f"{COMPLEX_ACTION_PLAN_MAX_TOKENS:,}. This does not limit worker logs, images, tables, "
        "TIFF files, CSV files, or other scientific outputs. Complete Pixel Classification reviews "
        "receive up to 2,048 local or 4,096 online answer tokens so all clusters can be reported. "
        "Muse receives at least 1,024 generation tokens because it uses part of them for hidden reasoning."
    )

    st.markdown("### Local storage")
    st.code(
        f"SPACEMAP AI root: {AI_ROOT}\n"
        f"Assistant data: {ASSISTANT_DATA_DIR}",
        language="text",
    )
    st.info("The current manual SPACEMAP tools keep their existing behavior. These limits apply to AI tools only.")


def _render_admin_access() -> None:
    st.markdown("### Administrator folder access")
    st.warning(
        "This controls AI tools, not the normal SPACEMAP buttons. Chat cannot add or remove access. "
        "Only use Read + write for a specific data project you are willing to let approved jobs change."
    )
    st.markdown("#### Built-in access")
    st.dataframe(
        [
            {"Folder": str(AI_ROOT), "Permission": "Read only", "Can revoke": "No"},
            {"Folder": str(ASSISTANT_DATA_DIR), "Permission": "Read + write", "Can revoke": "No"},
            {"Folder": str(ASSISTANT_PROJECTS_DIR), "Permission": "Read + write", "Can revoke": "No"},
        ],
        hide_index=True,
        width="stretch",
    )

    grants = list_admin_access_grants()
    st.markdown("#### Extra access granted by administrator")
    if grants:
        st.dataframe(
            [
                {
                    "ID": item["id"][:8],
                    "Folder": item["path"],
                    "Permission": "Read + write" if item["mode"] == "read_write" else "Read only",
                    "Added": item["created_at"],
                }
                for item in grants
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No extra folders are allowed.")

    with st.form("assistant_admin_access_add", clear_on_submit=True):
        folder = st.text_input(
            "Exact folder to allow",
            placeholder="/path/to/a_data_project",
        )
        permission_label = st.selectbox("Permission", ["Read only", "Read + write"])
        confirmed = st.checkbox(
            "I understand that approved AI actions can use this whole folder and its subfolders."
        )
        add_access = st.form_submit_button("Grant Folder Access", type="primary")
    if add_access:
        if not confirmed:
            st.error("Confirm the administrator access statement first.")
        else:
            try:
                mode = "read_write" if permission_label == "Read + write" else "read"
                record = grant_admin_access(folder, mode)
                AssistantStore().log_activity(
                    "admin_access_granted",
                    f"Administrator granted {permission_label.lower()} access to {record['path']}.",
                    details=record,
                )
                st.success("Folder access granted.")
                st.rerun()
            except AssistantPathError as error:
                st.error(str(error))

    if grants:
        grant_labels = {
            f"{item['path']} · {'Read + write' if item['mode'] == 'read_write' else 'Read only'} · {item['id'][:8]}": item["id"]
            for item in grants
        }
        selected_grant = st.selectbox("Access to revoke", list(grant_labels))
        if st.button("Revoke Selected Access", width="stretch"):
            try:
                removed = revoke_admin_access(grant_labels[selected_grant])
                AssistantStore().log_activity(
                    "admin_access_revoked",
                    f"Administrator revoked access to {removed['path']}.",
                    details=removed,
                )
                st.success("Folder access revoked. New plans cannot use it.")
                st.rerun()
            except AssistantPathError as error:
                st.error(str(error))

    with st.expander("Test a path", expanded=False):
        test_path = st.text_input("Path to test", key="assistant_admin_test_path")
        test_mode = st.radio("Test permission", ["Read", "Write"], horizontal=True)
        if st.button("Check Path Access", width="stretch"):
            try:
                if test_mode == "Read":
                    checked = require_assistant_read_path(test_path)
                else:
                    checked = require_assistant_write_path(test_path)
                st.success(f"Allowed: {checked}")
            except (AssistantPathError, OSError, ValueError) as error:
                st.error(str(error))

    st.caption(
        "Application source, model files, environments, and Git metadata remain protected. "
        "For an external runnable project, grant its project folder and save results under its outputs/ folder."
    )

    st.divider()
    st.markdown("### Owner-approved online data folders")
    st.warning(
        "The built-in SPACEMAP_AI/projects folder and all of its subfolders are already available for "
        "bounded online review when the Owner selects an online model and confirms the online session. "
        "Use this list only to approve another data folder outside Projects."
    )
    st.caption(
        "Images are resized and converted to PNG locally. CSV/TSV files send only a limited sample. "
        "The original files stay where they are."
    )
    online_grants = list_online_access_grants()
    if online_grants:
        st.dataframe(
            [
                {
                    "ID": item["id"][:8],
                    "Folder": item["path"],
                    "Subfolders": "Included",
                    "Added": item["created_at"],
                }
                for item in online_grants
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No folder is allowed to send file evidence to an online model.")

    with st.form("assistant_online_access_add", clear_on_submit=True):
        online_folder = st.text_input(
            "Exact data folder to allow online",
            placeholder=str(ASSISTANT_PROJECTS_DIR / "a_public_project"),
        )
        online_confirmed = st.checkbox(
            "I am the Owner and confirm this folder contains only public, deidentified, synthetic, or institutionally approved data."
        )
        add_online = st.form_submit_button("Allow Bounded Online Review", type="primary")
    if add_online:
        if not online_confirmed:
            st.error("Confirm the Owner data statement first.")
        else:
            try:
                record = grant_online_access(online_folder)
                AssistantStore().log_activity(
                    "online_access_granted",
                    f"Owner allowed bounded online review from {record['path']}.",
                    details=record,
                )
                st.success("Online data-folder access was added.")
                st.rerun()
            except (AssistantPathError, OSError, ValueError) as error:
                st.error(str(error))

    if online_grants:
        online_labels = {
            f"{item['path']} · {item['id'][:8]}": item["id"] for item in online_grants
        }
        selected_online = st.selectbox("Online access to revoke", list(online_labels))
        if st.button("Revoke Selected Online Access", width="stretch"):
            try:
                removed = revoke_online_access(online_labels[selected_online])
                AssistantStore().log_activity(
                    "online_access_revoked",
                    f"Owner revoked online data access to {removed['path']}.",
                    details=removed,
                )
                st.success("Online access was revoked immediately for new requests.")
                st.rerun()
            except (AssistantPathError, OSError, ValueError) as error:
                st.error(str(error))

    limits = load_online_share_limits()
    with st.form("assistant_online_share_limits"):
        max_files = st.number_input(
            "Maximum files sent in one request",
            min_value=1,
            max_value=8,
            value=limits.max_files,
        )
        csv_rows = st.number_input(
            "Maximum CSV/TSV data rows per file",
            min_value=25,
            max_value=1000,
            value=limits.csv_rows,
            step=25,
        )
        image_max_size = st.number_input(
            "Maximum image preview side (pixels)",
            min_value=256,
            max_value=2048,
            value=limits.image_max_size,
            step=128,
        )
        if st.form_submit_button("Save Online Sharing Limits"):
            saved = save_online_share_limits(max_files, csv_rows, image_max_size)
            st.success(
                f"Saved: {saved.max_files} files, {saved.csv_rows} table rows, "
                f"{saved.image_max_size}px image previews."
            )


def _append_compute_terminal(text: str) -> None:
    existing = str(st.session_state.get(CLUSTER_TERMINAL_KEY) or "")
    combined = (existing + "\n\n" + str(text).strip()).strip()
    st.session_state[CLUSTER_TERMINAL_KEY] = combined[-30_000:]


def _refresh_cluster() -> None:
    partitions, partition_terminal = query_partitions()
    jobs, jobs_terminal = query_my_jobs()
    st.session_state[CLUSTER_PARTITIONS_KEY] = partitions
    st.session_state[CLUSTER_JOBS_KEY] = jobs
    _append_compute_terminal(partition_terminal + "\n\n" + jobs_terminal)


def _ensure_local_model_running() -> None:
    """Start the chosen local model when it is downloaded but stopped, and wait for it.

    Asking a question is enough: nobody has to open the setup and press Start
    first. Raises ModelUnavailable with the reason when it cannot start.
    """
    from spacemap_assistant.model_runtime import ensure_running

    status = runtime_status()
    if status.ready:
        return
    if not status.installed:
        raise ModelUnavailable(status.message)
    with st.status(f"Starting {status.message.split(' is ')[0]} (about a minute on a GPU)...") as box:
        try:
            ensure_running(on_wait=lambda current, seconds: box.update(
                label=f"{current.message} {seconds:.0f} s"))
        except ModelRuntimeError as error:
            box.update(label="The local model could not start.", state="error")
            raise ModelUnavailable(str(error)) from error
        box.update(label="The local model is ready.", state="complete")


def _partition_rows(partitions: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "Partition": item.name,
            "Idle nodes": item.idle_nodes,
            "Total nodes": item.total_nodes,
            "CPU/node": item.cpus_per_node,
            "RAM/node (GB)": round(item.memory_mib / 1024, 1),
            "GPU/node": item.gpu_per_node,
            "State": ", ".join(item.states),
        }
        for item in partitions
    ]


def _render_cluster_controls() -> None:
    st.markdown("### Cluster compute")
    st.caption(
        "The assistant can read cluster availability, recommend GPU or high-memory resources, and create a fixed safe probe. "
        "It cannot accept arbitrary terminal commands."
    )
    if st.button("Check cluster now", width="stretch"):
        try:
            _refresh_cluster()
            _store().log_activity("cluster_checked", "Cluster partitions and current jobs were refreshed.")
            st.success("Cluster information refreshed.")
        except ClusterError as error:
            _append_compute_terminal(f"Cluster check failed: {error}")
            st.error(str(error))

    partitions = st.session_state.get(CLUSTER_PARTITIONS_KEY, [])
    jobs = st.session_state.get(CLUSTER_JOBS_KEY, [])
    if partitions:
        with st.expander("Available partitions", expanded=False):
            st.dataframe(_partition_rows(partitions), hide_index=True, width="stretch")
    if jobs:
        with st.expander("My active cluster jobs", expanded=False):
            st.dataframe(jobs, hide_index=True, width="stretch")

    st.markdown("#### Plan a resource request")
    profile_name = st.selectbox("Type of work", list(WORKLOAD_PROFILES), key="assistant_compute_profile")
    profile = WORKLOAD_PROFILES[profile_name]
    input_key = re.sub(r"[^a-z0-9]+", "_", profile_name.lower()).strip("_")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cpus = st.number_input("CPUs", 1, 256, profile.cpus, key=f"compute_cpus_{input_key}")
    with col2:
        memory_gb = st.number_input("RAM (GB)", 1, 1024, profile.memory_gb, key=f"compute_ram_{input_key}")
    with col3:
        gpu_count = st.number_input("GPUs", 0, 4, profile.gpu_count, key=f"compute_gpu_{input_key}")
    with col4:
        hours = st.number_input("Hours", 1, 168, profile.hours, key=f"compute_hours_{input_key}")
    available_names = [item.name for item in partitions]
    partition_choice = st.selectbox(
        "Partition",
        ["auto", *available_names] if available_names else ["auto"],
        help="Auto chooses a suitable currently available partition.",
    )

    if st.button("Create Safe Resource Probe", type="primary", width="stretch"):
        try:
            if not partitions:
                _refresh_cluster()
                partitions = st.session_state.get(CLUSTER_PARTITIONS_KEY, [])
            request = ResourceRequest(
                workload=profile.workload,
                cpus=int(cpus),
                memory_gb=int(memory_gb),
                gpu_count=int(gpu_count),
                hours=int(hours),
                partition=partition_choice,
            )
            path, selected, reason = create_probe_script(request, partitions)
            validation = validate_probe_script(path)
            st.session_state[CLUSTER_SCRIPT_KEY] = str(path)
            _append_compute_terminal(validation)
            _store().log_activity(
                "cluster_probe_created",
                f"Safe probe created for {selected.name}.",
                details={"script": str(path), "reason": reason},
            )
            st.success(reason)
        except (ClusterError, OSError, ValueError) as error:
            _append_compute_terminal(f"Probe creation failed: {error}")
            st.error(str(error))

    script_path = str(st.session_state.get(CLUSTER_SCRIPT_KEY) or "")
    if script_path:
        st.caption(f"Prepared probe: {script_path}")
        with st.expander("Review exact Slurm job file", expanded=False):
            try:
                review_path = require_assistant_write_path(Path(script_path))
                validate_probe_script(review_path)
                st.code(review_path.read_text(encoding="utf-8"), language="bash")
            except (ClusterError, OSError, ValueError) as error:
                st.warning(str(error))
        confirmed = st.checkbox(
            "I reviewed this fixed resource probe and want to submit it to the cluster.",
            key="assistant_cluster_submit_confirmation",
        )
        if st.button("Submit probe to cluster", disabled=not confirmed, width="stretch"):
            try:
                job_id, terminal = submit_probe_script(script_path)
                _append_compute_terminal(terminal)
                _store().log_activity(
                    "cluster_probe_submitted",
                    f"Cluster resource probe submitted as Slurm job {job_id}.",
                    details={"job_id": job_id, "script": script_path},
                )
                st.success(f"Probe submitted. Slurm job number: {job_id}")
                _refresh_cluster()
            except (ClusterError, OSError, ValueError) as error:
                _append_compute_terminal(f"Probe submission failed: {error}")
                st.error(str(error))

    st.markdown("#### Visible compute terminal")
    st.caption("This is a read-only view of commands chosen by SPACEMAP AI and their output. There is no command input box.")
    terminal = str(st.session_state.get(CLUSTER_TERMINAL_KEY) or "No cluster checks have run in this app session.")
    st.code(terminal, language="text", wrap_lines=True, height=340)


def _redact_log_text(value: str) -> str:
    text = str(value).replace("\x1b", "")
    text = re.sub(r"(?i)(api[_-]?key|password|token|secret)\s*[=:]\s*\S+", r"\1=[REDACTED]", text)
    return text


def _activity_lines(store: AssistantStore, detailed: bool) -> list[str]:
    lines: list[str] = []
    for item in store.list_activity(limit=300):
        line = f"{item['created_at']} [{item['level'].upper()}] {item['event']}: {item['message']}"
        if detailed and item.get("details"):
            line += " | " + json.dumps(item["details"], default=str, sort_keys=True)
        lines.append(_redact_log_text(line))

    process_logs = st.session_state.get(PROCESS_LOGS_KEY, {})
    if isinstance(process_logs, dict):
        for page, records in process_logs.items():
            if not isinstance(records, list):
                continue
            for item in records[-100:]:
                if not isinstance(item, dict):
                    continue
                line = (
                    f"{item.get('Timestamp', '')} [{str(item.get('Status', 'info')).upper()}] "
                    f"{page}/{item.get('Event', 'event')}: {item.get('Message', '')}"
                )
                if detailed and item.get("Details") not in {None, "", "{}"}:
                    line += " | " + str(item["Details"])
                lines.append(_redact_log_text(line))
    return sorted(lines)[-400:]


def _render_activity_console(*, expanded: bool) -> None:
    store = _store()
    with st.expander("Activity Console", expanded=expanded):
        col1, col2 = st.columns([1, 3])
        with col1:
            view = st.radio("View", ["Simple", "Detailed"], horizontal=True, key="assistant_console_view")
        with col2:
            st.caption("Read-only. Use approved buttons—not terminal commands—to control future jobs.")
        lines = _activity_lines(store, detailed=view == "Detailed")
        console_text = "\n".join(lines) if lines else "No assistant or pipeline activity has been recorded yet."
        st.code(console_text, language="text", line_numbers=False, wrap_lines=True, height=320)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Console Log",
                console_text,
                file_name="spacemap_ai_activity.log",
                mime="text/plain",
                width="stretch",
            )
        with col2:
            if st.button("Refresh Console", width="stretch"):
                st.rerun()


@st.fragment(run_every=2.0)
def _render_compact_live_action_status() -> None:
    actions = list_all_actions(limit=10)
    active = next(
        (item for item in actions if item.get("state") in {"launching", "running"}),
        None,
    )
    if active is None:
        return
    st.markdown(
        f"**Working:** {active['action_type']} · {active.get('stage') or 'Starting'} · "
        f"job `{active['action_id'][:8]}`"
    )
    if active.get("message"):
        st.caption(str(active["message"]))
    events = progress_events(active["action_id"], limit=4)
    if events:
        st.code(
            "\n".join(
                f"{item.get('stage', '')}: {item.get('message', '')}" for item in events[-4:]
            ),
            language="text",
        )
    files = [item for item in action_file_progress(active["action_id"]) if item["state"] != "Waiting"]
    if files:
        st.caption(" · ".join(f"{item['file']}: {item['size']}" for item in files[:3]))


def render_assistant_launcher(page_id: str, assistant_page: Any) -> None:
    """Render the compact top-right assistant panel on normal pipeline pages."""
    context = _current_context(page_id)
    st.markdown(
        """
        <style>
        div[data-testid="stPopoverBody"] { min-width: min(460px, 92vw); }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.popover("AI Assistant", icon=":material/chat:", width="stretch"):
        st.caption(f"Current context: {context.get('page_title', page_id)} · Planning mode")
        _render_compact_live_action_status()
        quick = st.columns(2)
        store = _store()
        conversation_id = _conversation_id(store)
        with quick[0]:
            if st.button("What am I seeing?", key=f"assistant_explain_{page_id}", width="stretch"):
                _submit_prompt("What am I seeing?", page_id, store, conversation_id)
                st.rerun()
        with quick[1]:
            if st.button("What is next?", key=f"assistant_next_{page_id}", width="stretch"):
                _submit_prompt("What should I do next?", page_id, store, conversation_id)
                st.rerun()
        _render_chat(page_id, compact=True)
        st.page_link(assistant_page, label="Open Full Assistant Workspace", icon=":material/smart_toy:", width="stretch")


def render_assistant_workspace() -> None:
    """Render the full, unnumbered SPACEMAP AI control workspace."""
    page_id = "assistant"
    context = _current_context(page_id)
    st.markdown("## SPACEMAP AI Assistant")
    st.caption(
        f"Safe assistant and reviewed actions · Current scientific context: {context.get('page_title')}"
    )

    # One Setup tab holds every setting (models, online provider, answers,
    # cluster, folder access, activity); Tab 0 shows the same page.
    chat_tab, actions_tab, instructions_tab, memory_tab, setup_tab = st.tabs(
        ["Chat", "Actions", "Instructions & Safety", "Memory", "Setup"]
    )
    with chat_tab:
        _render_chat(page_id, compact=False)
        with st.expander("Save or export this conversation", expanded=False):
            _render_chat_exports()
        _render_recent_artifacts(_conversation_id(_store()))
    with actions_tab:
        render_registration_actions(
            chat_renderer=lambda: _render_chat(
                "assistant_action",
                compact=True,
                show_online_consent=False,
            )
        )
    with instructions_tab:
        _render_instruction_editor()
    with memory_tab:
        _render_memory()
    with setup_tab:
        from spacemap_gui.components.ai_setup import render_ai_setup

        render_ai_setup("assistant")
