"""Reviewed scientific actions shown in the SPACEMAP AI workspace."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import streamlit as st

from spacemap_assistant.action_monitor import (
    action_details,
    action_file_progress,
    action_log,
    list_all_actions,
    progress_events,
)
from spacemap_assistant.action_notebook import (
    action_notebook_json,
    action_title,
    collect_action_outputs,
    notebook_action_id,
    read_table_preview,
    save_action_notebook,
)
from spacemap_assistant.general_action import (
    GeneralActionError,
    action_directory as general_action_directory,
    launch_general_action,
    list_general_actions,
    load_action_status as load_general_action_status,
    load_general_plan,
)
from spacemap_assistant.blur_action import (
    BlurActionError,
    action_directory as blur_action_directory,
    launch_blur_action,
    list_blur_actions,
    load_action_status as load_blur_action_status,
    load_blur_plan,
)
from spacemap_assistant.paths import ASSISTANT_PROJECTS_DIR
from spacemap_assistant.registration_action import (
    RegistrationActionError,
    action_directory,
    build_registration_plan,
    launch_registration_action,
    list_registration_actions,
    load_action_status,
    load_registration_plan,
    recent_action_log,
)
from spacemap_assistant.roi_action import (
    ROIActionError,
    action_directory as roi_action_directory,
    build_roi_plan,
    launch_roi_action,
    list_roi_actions,
    load_action_status as load_roi_action_status,
    load_roi_plan,
    recent_action_log as recent_roi_action_log,
)
from spacemap_assistant.storage import AssistantStore
from spacemap_workflows.registry import list_tools, refresh_tool_registry, tool_counts
from spacemap_workflows.annotation import default_blur_trials, default_bright_trials


REGISTRATION_DRAFT_KEY = "_spacemap_ai_registration_draft"
REGISTRATION_ACTION_ID_KEY = "_spacemap_ai_registration_action_id"
REGISTRATION_CODEX_WIDGET = "assistant_action_codex_path"
REGISTRATION_HE_WIDGET = "assistant_action_he_path"
REGISTRATION_OUTPUT_WIDGET = "assistant_action_output_dir"
REGISTRATION_OVERWRITE_WIDGET = "assistant_action_overwrite"
ROI_DRAFT_KEY = "_spacemap_ai_roi_draft"
ROI_ACTION_ID_KEY = "_spacemap_ai_roi_action_id"
ROI_IMAGE_WIDGET = "assistant_action_roi_image_path"
ROI_BOUNDARIES_WIDGET = "assistant_action_roi_boundaries_path"
ROI_OUTPUT_WIDGET = "assistant_action_roi_output_dir"
ROI_MPP_WIDGET = "assistant_action_roi_mpp"
ROI_OVERWRITE_WIDGET = "assistant_action_roi_overwrite"
ACTION_PREAPPROVAL_KEY = "_spacemap_ai_session_action_preapproval"
ACTION_APPROVAL_MODE_KEY = "_spacemap_ai_session_action_approval_mode"
LIVE_ACTION_KEY = "_spacemap_ai_live_action_id"
GENERAL_ACTION_ID_KEY = "_spacemap_ai_general_action_id"
BLUR_ACTION_ID_KEY = "_spacemap_ai_blur_action_id"
ACTION_NOTEBOOK_KEY = "_spacemap_ai_action_notebook_id"
ACTION_NOTEBOOK_PENDING_KEY = "_spacemap_ai_action_notebook_pending_id"
ACTION_NOTEBOOK_LAST_ACTIVE_KEY = "_spacemap_ai_action_notebook_last_active_id"
ACTION_IMPORTED_NOTEBOOK_KEY = "_spacemap_ai_imported_action_notebook"


def preapproval_enabled() -> bool:
    """Return the session-only safe-action permission."""
    mode = st.session_state.get(ACTION_APPROVAL_MODE_KEY)
    if mode is not None:
        return str(mode) == "auto"
    return bool(st.session_state.get(ACTION_PREAPPROVAL_KEY, False))


@st.fragment(run_every=2.0)
def _render_live_action_terminal() -> None:
    """Auto-refresh the durable stdout/progress view without rerunning the full app."""
    actions = list_all_actions(limit=30)
    if not actions:
        st.caption("No background action has been planned yet.")
        return
    labels = {
        (
            f"{item['state'].upper()} · {item['action_type']} · "
            f"{item['created_at'][:19]} · {item['action_id'][:8]}"
        ): item["action_id"]
        for item in actions
    }
    current = str(st.session_state.get(LIVE_ACTION_KEY) or "")
    options = list(labels)
    index = next((position for position, label in enumerate(options) if labels[label] == current), 0)
    selected = st.selectbox("Background job", options, index=index, key="_live_action_selector")
    action_id = labels[selected]
    st.session_state[LIVE_ACTION_KEY] = action_id
    row = next(item for item in actions if item["action_id"] == action_id)

    columns = st.columns(4)
    columns[0].metric("State", _status_label(str(row["state"])))
    columns[1].metric("Stage", str(row["stage"] or "—"))
    elapsed = row.get("elapsed_seconds")
    columns[2].metric("Elapsed", f"{float(elapsed):,.1f} s" if elapsed is not None else "—")
    columns[3].metric("Action", action_id[:8])
    if row.get("message"):
        st.caption(str(row["message"]))

    files = action_file_progress(action_id)
    if files:
        st.dataframe(files, hide_index=True, width="stretch", height=min(250, 36 + 35 * len(files)))

    events = progress_events(action_id, limit=100)
    event_text = "\n".join(
        f"{item.get('created_at', '')}  {item.get('stage', '')}: {item.get('message', '')}"
        for item in events[-30:]
    )
    stdout = action_log(action_id, max_characters=60_000)
    if stdout and stdout not in event_text:
        event_text = (event_text + "\n" + stdout).strip()
    st.code(event_text or "Waiting for the worker's first message…", language="text", wrap_lines=True, height=360)
    st.caption("Live, read-only worker output. It refreshes every 2 seconds and accepts no commands.")


def set_registration_draft(draft: dict[str, object]) -> None:
    """Move a clear natural-language request into editable action fields."""
    clean = dict(draft)
    st.session_state[REGISTRATION_DRAFT_KEY] = clean
    st.session_state[REGISTRATION_CODEX_WIDGET] = str(clean.get("codex_path") or "")
    st.session_state[REGISTRATION_HE_WIDGET] = str(clean.get("he_path") or "")
    st.session_state[REGISTRATION_OUTPUT_WIDGET] = str(clean.get("output_dir") or "")
    st.session_state[REGISTRATION_OVERWRITE_WIDGET] = bool(clean.get("overwrite", False))


def set_roi_draft(draft: dict[str, object]) -> None:
    """Move a clear natural-language ROI request into editable action fields."""
    clean = dict(draft)
    st.session_state[ROI_DRAFT_KEY] = clean
    st.session_state[ROI_IMAGE_WIDGET] = str(clean.get("image_path") or "")
    st.session_state[ROI_BOUNDARIES_WIDGET] = str(clean.get("boundaries_path") or "")
    st.session_state[ROI_OUTPUT_WIDGET] = str(clean.get("output_dir") or "")
    st.session_state[ROI_OVERWRITE_WIDGET] = bool(clean.get("overwrite", False))


def _suggested_paths() -> tuple[str, str, str]:
    project = ASSISTANT_PROJECTS_DIR / "AI_test"
    return (
        str(project / "data" / "codex" / "P11_codex.qptiff"),
        str(project / "data" / "he" / "P11_he.qptiff"),
        str(project / "outputs" / "AI_registered"),
    )


def _status_label(state: str) -> str:
    return {
        "planned": "Waiting for approval",
        "launching": "Starting",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "cancelled": "Stopped",
    }.get(state, state.title() or "Unknown")


def _render_saved_action(action_id: str) -> None:
    try:
        plan = load_registration_plan(action_id)
    except RegistrationActionError as error:
        st.error(str(error))
        return
    status = load_action_status(action_id)
    state = str(status.get("state") or "unknown")

    st.markdown("### Reviewed registration plan")
    cols = st.columns(4)
    cols[0].metric("State", _status_label(state))
    cols[1].metric("CODEX MPP", f"{plan.codex_mpp:g}")
    cols[2].metric("H&E MPP", f"{plan.he_mpp:g}")
    cols[3].metric("Anchor", plan.reference_channel)
    st.code(
        f"CODEX: {plan.codex_path}\n"
        f"H&E:   {plan.he_path}\n"
        f"Final: {plan.final_output}\n"
        f"Temporary work: {plan.intermediate_dir}",
        language="text",
    )
    st.caption(
        f"Plan {plan.action_id[:8]} · {len(plan.codex_channels)} CODEX channels · "
        f"{len(plan.he_channels)} H&E channels · {plan.n_keypoints:,} keypoints · "
        f"rotation {plan.pre_rotation_deg:g}°"
    )
    if plan.overwrite:
        st.warning(
            "Replacement was requested and is part of this plan. The old result will be preserved "
            f"as a recoverable backup: {plan.backup_output}"
        )
    else:
        st.info("This plan will stop if a result with the same final name appears.")

    if state == "planned":
        if preapproval_enabled():
            st.success("Session pre-approval is on. This registered action may start after validation.")
            confirmed = True
        else:
            confirmed = st.checkbox(
                "I reviewed these exact files, settings, and output. Start this registration.",
                key=f"assistant_registration_approval_{action_id}",
            )
        if st.button(
            "Start Approved Registration",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=f"assistant_registration_start_{action_id}",
        ):
            try:
                pid = launch_registration_action(action_id)
                AssistantStore().log_activity(
                    "registration_started",
                    f"Approved registration {action_id[:8]} started in the background.",
                    details={"pid": pid, "final_output": plan.final_output},
                )
                st.success(f"Registration started in the background as process {pid}.")
                st.rerun()
            except (RegistrationActionError, OSError, ValueError) as error:
                st.error(str(error))
    else:
        message = str(status.get("message") or status.get("stage") or "")
        if state == "completed":
            st.success(message or "Registration completed.")
        elif state == "failed":
            st.error(message or "Registration failed. Review the worker log below.")
        else:
            st.info(message or "Registration is running in the background.")
        if status.get("elapsed_seconds") is not None:
            st.caption(f"Elapsed time: {float(status['elapsed_seconds']):,.1f} seconds")

    directory = action_directory(action_id)
    report_path = directory / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            qc = report.get("qc") or {}
            preview_path = Path(str(qc.get("preview_path") or ""))
            if preview_path.is_file():
                st.markdown("#### Registration QC preview")
                st.image(str(preview_path), width="stretch")
                st.caption(str(qc.get("metric_note") or "Human review is required."))
            st.download_button(
                "Download Run Report",
                json.dumps(report, indent=2),
                file_name=f"registration_{action_id[:8]}_report.json",
                mime="application/json",
            )
        except (OSError, json.JSONDecodeError):
            st.warning("The run report could not be read.")

    with st.expander("Visible registration terminal", expanded=state in {"running", "failed"}):
        st.caption("Read-only. This shows the fixed worker's real messages; it does not accept commands.")
        st.code(recent_action_log(action_id), language="text", wrap_lines=True, height=340)
        if st.button("Refresh Registration Status", key=f"refresh_registration_{action_id}"):
            st.rerun()
        st.download_button(
            "Download Worker Log",
            recent_action_log(action_id, max_characters=2_000_000),
            file_name=f"registration_{action_id[:8]}.log",
            mime="text/plain",
            key=f"download_registration_log_{action_id}",
        )


def _render_registration_content() -> None:
    default_codex, default_he, default_output = _suggested_paths()
    st.session_state.setdefault(REGISTRATION_CODEX_WIDGET, default_codex)
    st.session_state.setdefault(REGISTRATION_HE_WIDGET, default_he)
    st.session_state.setdefault(REGISTRATION_OUTPUT_WIDGET, default_output)
    st.session_state.setdefault(REGISTRATION_OVERWRITE_WIDGET, False)
    st.text_input("CODEX image", key=REGISTRATION_CODEX_WIDGET)
    st.text_input("H&E image", key=REGISTRATION_HE_WIDGET)
    st.text_input("Output folder", key=REGISTRATION_OUTPUT_WIDGET)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        reference_channel = st.text_input("Reference channel", value="", help="Leave blank to auto-detect DAPI.")
    with col2:
        keypoints = st.number_input("Keypoints", 1_000, 100_000, 10_000, 1_000)
    with col3:
        rotation = st.selectbox("H&E rotation", [0, 90, 180, 270, -90, -180, -270])
    with col4:
        replace_existing = st.checkbox(
            "Replace existing result",
            key=REGISTRATION_OVERWRITE_WIDGET,
            help="Only enable this when you intend to replace the current final output.",
        )
    if replace_existing:
        st.warning("The old final output will be kept under a unique backup name; it will not be lost.")

    if st.button("Inspect Images and Build Plan", type="primary", width="stretch"):
        try:
            with st.spinner("Reading image headers and building the registration plan…"):
                plan = build_registration_plan(
                    st.session_state[REGISTRATION_CODEX_WIDGET],
                    st.session_state[REGISTRATION_HE_WIDGET],
                    st.session_state[REGISTRATION_OUTPUT_WIDGET],
                    reference_channel=reference_channel,
                    n_keypoints=int(keypoints),
                    pre_rotation_deg=float(rotation),
                    overwrite=bool(replace_existing),
                )
            st.session_state[REGISTRATION_ACTION_ID_KEY] = plan.action_id
            AssistantStore().log_activity(
                "registration_planned",
                f"Registration plan {plan.action_id[:8]} created for review.",
                details={"codex": plan.codex_path, "he": plan.he_path, "output": plan.final_output},
            )
            if preapproval_enabled():
                pid = launch_registration_action(plan.action_id)
                AssistantStore().log_activity(
                    "registration_started",
                    f"Pre-approved registration {plan.action_id[:8]} started in the background.",
                    details={"pid": pid, "final_output": plan.final_output},
                )
                st.success(f"Validated plan and started registration as process {pid}.")
            else:
                st.success("Plan created. Review it below before starting.")
            st.rerun()
        except (RegistrationActionError, OSError, ValueError) as error:
            st.error(str(error))

    recent_actions = list_registration_actions()
    if recent_actions:
        options = {
            f"{item['created_at'][:19]} · {item['prefix']} · {_status_label(str(item['state']))} · {item['action_id'][:8]}": item["action_id"]
            for item in recent_actions
        }
        current = str(st.session_state.get(REGISTRATION_ACTION_ID_KEY) or "")
        labels = list(options)
        current_index = next((index for index, label in enumerate(labels) if options[label] == current), 0)
        selected = st.selectbox("Saved registration actions", labels, index=current_index)
        st.session_state[REGISTRATION_ACTION_ID_KEY] = options[selected]

    action_id = str(st.session_state.get(REGISTRATION_ACTION_ID_KEY) or "")
    if action_id:
        _render_saved_action(action_id)
    else:
        st.info("No action has been planned in this app session yet.")


def _suggested_roi_paths() -> tuple[str, str, str]:
    project = ASSISTANT_PROJECTS_DIR / "AI_test"
    return (
        str(project / "outputs" / "AI_registered" / "P11_merged.ome.tiff"),
        str(project / "data" / "P11_roi_bounding_boxes_coords.geojson"),
        str(project / "outputs" / "rois"),
    )


def _render_saved_roi_action(action_id: str) -> None:
    try:
        plan = load_roi_plan(action_id)
    except ROIActionError as error:
        st.error(str(error))
        return
    status = load_roi_action_status(action_id)
    state = str(status.get("state") or "unknown")

    st.markdown("### Reviewed ROI extraction plan")
    columns = st.columns(4)
    columns[0].metric("State", _status_label(state))
    columns[1].metric("ROIs", len(plan.rois))
    columns[2].metric("Channels", len(plan.channels))
    columns[3].metric("MPP", f"{plan.mpp:g}")
    st.code(
        f"Image:      {plan.image_path}\n"
        f"Boundaries: {plan.boundaries_path}\n"
        f"Output:     {plan.output_dir}\n"
        f"Image size: {plan.image_width:,} × {plan.image_height:,} pixels",
        language="text",
    )
    with st.expander("Exact ROI output files", expanded=False):
        st.code("\n".join(str(item["output_path"]) for item in plan.rois), language="text")
    if plan.overwrite:
        st.warning("Replacement was requested. Every old ROI file will receive a unique backup name.")
    else:
        st.info("This action will stop if an output file with the same name appears.")

    if state == "planned":
        if preapproval_enabled():
            st.success("Session pre-approval is on. This registered action may start after validation.")
            confirmed = True
            approval_method = "Session pre-approval switch"
        else:
            confirmed = st.checkbox(
                "I reviewed these exact files and boundaries. Start ROI extraction.",
                key=f"assistant_roi_approval_{action_id}",
            )
            approval_method = "SPACEMAP AI review checkbox and start button"
        if st.button(
            "Start Approved ROI Extraction",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=f"assistant_roi_start_{action_id}",
        ):
            try:
                pid = launch_roi_action(action_id, approval_method=approval_method)
                AssistantStore().log_activity(
                    "roi_extraction_started",
                    f"Approved ROI extraction {action_id[:8]} started.",
                    details={"pid": pid, "output_dir": plan.output_dir},
                )
                st.success(f"ROI extraction started in the background as process {pid}.")
                st.rerun()
            except (ROIActionError, OSError, ValueError) as error:
                st.error(str(error))
    else:
        message = str(status.get("message") or status.get("stage") or "")
        if state == "completed":
            st.success(message or "ROI extraction completed.")
        elif state == "failed":
            st.error(message or "ROI extraction failed. Review the worker log below.")
        else:
            st.info(message or "ROI extraction is running in the background.")
        if status.get("elapsed_seconds") is not None:
            st.caption(f"Elapsed time: {float(status['elapsed_seconds']):,.1f} seconds")

    directory = roi_action_directory(action_id)
    report_path = directory / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            preview_path = Path(str((report.get("qc") or {}).get("preview_path") or ""))
            if preview_path.is_file():
                st.markdown("#### ROI boundary preview")
                st.image(str(preview_path), width="stretch")
                st.caption(str((report.get("qc") or {}).get("note") or "Human review is required."))
            st.download_button(
                "Download ROI Run Report",
                json.dumps(report, indent=2),
                file_name=f"roi_extraction_{action_id[:8]}_report.json",
                mime="application/json",
            )
        except (OSError, json.JSONDecodeError):
            st.warning("The ROI run report could not be read.")

    with st.expander("Visible ROI worker terminal", expanded=state in {"running", "failed"}):
        st.caption("Read-only. It shows the fixed ROI worker's real messages and accepts no commands.")
        st.code(recent_roi_action_log(action_id), language="text", wrap_lines=True, height=340)
        if st.button("Refresh ROI Status", key=f"refresh_roi_{action_id}"):
            st.rerun()
        st.download_button(
            "Download ROI Worker Log",
            recent_roi_action_log(action_id, max_characters=2_000_000),
            file_name=f"roi_extraction_{action_id[:8]}.log",
            mime="text/plain",
            key=f"download_roi_log_{action_id}",
        )


def _render_roi_content() -> None:
    default_image, default_boundaries, default_output = _suggested_roi_paths()
    st.session_state.setdefault(ROI_IMAGE_WIDGET, default_image)
    st.session_state.setdefault(ROI_BOUNDARIES_WIDGET, default_boundaries)
    st.session_state.setdefault(ROI_OUTPUT_WIDGET, default_output)
    st.session_state.setdefault(ROI_MPP_WIDGET, 0.0)
    st.session_state.setdefault(ROI_OVERWRITE_WIDGET, False)
    st.text_input("Registered image", key=ROI_IMAGE_WIDGET)
    st.text_input("ROI boundaries (GeoJSON)", key=ROI_BOUNDARIES_WIDGET)
    st.text_input("ROI output folder", key=ROI_OUTPUT_WIDGET)
    column1, column2 = st.columns(2)
    with column1:
        mpp = st.number_input(
            "MPP override (0 = read from image)",
            min_value=0.0,
            step=0.001,
            format="%.4f",
            key=ROI_MPP_WIDGET,
        )
    with column2:
        overwrite = st.checkbox(
            "Replace existing ROI outputs",
            key=ROI_OVERWRITE_WIDGET,
            help="Only enable this when you intend to replace same-named ROI files.",
        )
    if overwrite:
        st.warning("Old ROI files will be preserved under unique backup names.")

    if st.button("Inspect Boundaries and Build ROI Plan", type="primary", width="stretch"):
        try:
            with st.spinner("Reading the image header and checking every ROI boundary…"):
                plan = build_roi_plan(
                    st.session_state[ROI_IMAGE_WIDGET],
                    st.session_state[ROI_BOUNDARIES_WIDGET],
                    st.session_state[ROI_OUTPUT_WIDGET],
                    mpp=float(mpp) if float(mpp) > 0 else None,
                    overwrite=bool(overwrite),
                )
            st.session_state[ROI_ACTION_ID_KEY] = plan.action_id
            AssistantStore().log_activity(
                "roi_extraction_planned",
                f"ROI plan {plan.action_id[:8]} created.",
                details={
                    "image": plan.image_path,
                    "boundaries": plan.boundaries_path,
                    "output": plan.output_dir,
                    "roi_count": len(plan.rois),
                },
            )
            if preapproval_enabled():
                pid = launch_roi_action(plan.action_id, approval_method="Session pre-approval switch")
                st.success(f"Validated the ROI plan and started process {pid}.")
            else:
                st.success("ROI plan created. Review it below before starting.")
            st.rerun()
        except (ROIActionError, OSError, ValueError) as error:
            st.error(str(error))

    recent_actions = list_roi_actions()
    if recent_actions:
        options = {
            f"{item['created_at'][:19]} · {item['image_name']} · {item['roi_count']} ROIs · "
            f"{_status_label(str(item['state']))} · {item['action_id'][:8]}": item["action_id"]
            for item in recent_actions
        }
        current = str(st.session_state.get(ROI_ACTION_ID_KEY) or "")
        labels = list(options)
        index = next((idx for idx, label in enumerate(labels) if options[label] == current), 0)
        selected = st.selectbox("Saved ROI actions", labels, index=index)
        st.session_state[ROI_ACTION_ID_KEY] = options[selected]

    action_id = str(st.session_state.get(ROI_ACTION_ID_KEY) or "")
    if action_id:
        _render_saved_roi_action(action_id)
    else:
        st.info("No ROI extraction has been planned in this app session yet.")


def _render_saved_general_action(action_id: str) -> None:
    try:
        plan = load_general_plan(action_id)
    except GeneralActionError as error:
        st.error(str(error))
        return
    status = load_general_action_status(action_id)
    state = str(status.get("state") or "unknown")

    st.markdown("### Protected SPACEMAP job")
    columns = st.columns(4)
    columns[0].metric("State", _status_label(state))
    columns[1].metric("Inputs", len(plan.inputs))
    columns[2].metric("Max time", f"{plan.max_runtime_seconds / 3600:g} h")
    columns[3].metric("Job", plan.action_id[:8])
    st.markdown(f"**{plan.title}**")
    st.caption(plan.user_request)
    if plan.registered_tools:
        st.caption("Registered SPACEMAP tools: " + ", ".join(plan.registered_tools))
    st.code(
        "\n".join(f"{name}: {path}" for name, path in plan.inputs.items())
        + f"\nOutput folder: {plan.output_dir}",
        language="text",
    )
    if plan.parameters:
        with st.expander("Parameters and hypothesis settings", expanded=True):
            st.json(plan.parameters)
    with st.expander("Exact generated Python to run", expanded=state == "planned"):
        st.caption(
            "This code can read the SPACEMAP source and call its functions. The operating-system sandbox "
            "mounts application source read-only and exposes only a temporary output folder for writing."
        )
        st.code(plan.code, language="python", line_numbers=True)
    if plan.overwrite:
        st.warning(f"Replacement was requested. The current output folder will be moved to: {plan.backup_dir}")
    else:
        st.info("The job stops if its requested final output folder already exists.")

    if state == "planned":
        if preapproval_enabled():
            confirmed = True
            method = "Session pre-approval switch"
            st.success("Session pre-approval is on. The protected job may start after final validation.")
        else:
            confirmed = st.checkbox(
                "I reviewed the inputs, output, parameters, and exact generated code. Start this job.",
                key=f"assistant_general_approval_{action_id}",
            )
            method = "Exact generated-code review checkbox and start button"
        if st.button(
            "Start Protected SPACEMAP Job",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=f"assistant_general_start_{action_id}",
        ):
            try:
                pid = launch_general_action(action_id, approval_method=method)
                st.session_state[LIVE_ACTION_KEY] = action_id
                AssistantStore().log_activity(
                    "general_action_started",
                    f"Protected AI job {action_id[:8]} started.",
                    details={"pid": pid, "title": plan.title, "output_dir": plan.output_dir},
                )
                st.success(f"Protected job started as process {pid}.")
                st.rerun()
            except (GeneralActionError, OSError, ValueError) as error:
                st.error(str(error))
    elif state == "completed":
        st.success(str(status.get("message") or "The protected job completed."))
    elif state == "failed":
        st.error(str(status.get("message") or "The protected job failed. See its Action Notebook technical log."))
    else:
        st.info(str(status.get("message") or "The protected job is running."))

    report_path = general_action_directory(action_id) / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            st.download_button(
                "Download AI Job Report",
                json.dumps(report, indent=2),
                file_name=f"spacemap_ai_job_{action_id[:8]}_report.json",
                mime="application/json",
            )
        except (OSError, json.JSONDecodeError):
            st.warning("The AI job report could not be read.")


def _render_general_actions() -> None:
    st.caption(
        "The selected model may choose from the full indexed SPACEMAP tool catalog and prepare a "
        "bounded job. The source tree is read-only while the job runs."
    )
    actions = list_general_actions()
    if not actions:
        st.info(
            "No protected job has been prepared yet. Ask Chat to perform a SPACEMAP task and include "
            "the exact input and output paths."
        )
        return
    options = {
        f"{item['created_at'][:19]} · {item['title']} · {_status_label(str(item['state']))} · "
        f"{item['action_id'][:8]}": item["action_id"]
        for item in actions
    }
    current = str(st.session_state.get(GENERAL_ACTION_ID_KEY) or "")
    labels = list(options)
    index = next((position for position, label in enumerate(labels) if options[label] == current), 0)
    selected = st.selectbox("Saved protected jobs", labels, index=index)
    action_id = options[selected]
    st.session_state[GENERAL_ACTION_ID_KEY] = action_id
    _render_saved_general_action(action_id)


def _render_saved_blur_action(action_id: str) -> None:
    try:
        plan = load_blur_plan(action_id)
    except BlurActionError as error:
        st.error(str(error))
        return
    status = load_blur_action_status(action_id)
    state = str(status.get("state") or "unknown")
    artifact_label = "Bright glare/debris" if plan.artifact_kind == "bright" else "Blur"
    st.markdown(f"### {artifact_label} parameter experiment")
    columns = st.columns(4)
    columns[0].metric("State", _status_label(state))
    columns[1].metric("Trials", plan.maximum_trials)
    columns[2].metric("MPP", f"{plan.microns_per_pixel:g}" if plan.microns_per_pixel else "From project")
    columns[3].metric("Job", plan.action_id[:8])
    st.code(
        f"Image: {plan.image_path}\n"
        f"Tissue boundary: {plan.tissue_boundary_path}\n"
        f"Project settings: {plan.project_config_path or 'SPACEMAP defaults'}\n"
        f"New run folder: {plan.run_output_dir}",
        language="text",
    )
    st.caption(
        "Every tested setting is kept in this new run folder. The source image and the current "
        "annotation are not replaced."
    )
    if plan.provider == "nvidia" and plan.online_visual_review_approved:
        st.info(
            "Starting this checked plan also permits the selected NVIDIA model to review only the "
            "small QC PNGs created by this run. The full TIFF is never sent."
        )
    if state == "planned":
        if preapproval_enabled():
            confirmed = True
            method = "Session pre-approval switch"
            st.success("Auto approval is on. This checked experiment may start.")
        else:
            confirmed = st.checkbox(
                "I reviewed the image, tissue boundary, and output folder. Start the tests.",
                key=f"assistant_blur_approval_{action_id}",
            )
            method = "Artifact plan review checkbox and start button"
        if st.button(
            f"Start {artifact_label} Tests",
            type="primary",
            disabled=not confirmed,
            width="stretch",
            key=f"assistant_blur_start_{action_id}",
        ):
            try:
                pid = launch_blur_action(action_id, approval_method=method)
                st.session_state[LIVE_ACTION_KEY] = action_id
                AssistantStore().log_activity(
                    "artifact_action_started",
                    f"{artifact_label} experiment {action_id[:8]} started.",
                    details={"pid": pid, "output_dir": plan.run_output_dir},
                )
                st.success(f"{artifact_label} tests started in the background as process {pid}.")
                st.rerun()
            except (BlurActionError, OSError, ValueError) as error:
                st.error(str(error))
    elif state == "completed":
        if str(status.get("next_state") or "") == "no_candidates_detected":
            st.warning(str(status.get("message") or f"No {artifact_label} candidates were detected."))
            st.caption("No setting was selected, and there were no polygons available for visual review.")
        else:
            st.success(str(status.get("message") or "The parameter tests completed."))
            st.warning(
                "The selected result is provisional. Review its polygons before publishing it as the "
                "final Annotation result."
            )
    elif state == "failed":
        st.error(str(status.get("message") or f"The {artifact_label} experiment failed."))
    else:
        st.info(str(status.get("message") or f"The {artifact_label} experiment is running."))

    report_path = blur_action_directory(action_id) / "report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = report.get("result") or {}
            outputs = result.get("outputs") or {}
            comparison = list(result.get("comparison") or [])
            if comparison and not any(int(item.get("polygon_count") or 0) for item in comparison):
                st.error(
                    f"All {len(comparison)} tested settings produced zero polygons. This run has no "
                    "best result, even if it was created by an older app version that labelled one trial provisional."
                )
            st.code(
                f"Comparison: {outputs.get('comparison_csv', '')}\n"
                f"Provisional polygons: {outputs.get('provisional_geojson', '')}",
                language="text",
            )
            previews = [
                item for item in (result.get("review_previews") or [])
                if Path(str(item.get("preview_path") or "")).is_file()
            ]
            if previews:
                st.markdown("#### Local visual-review evidence")
                st.image(
                    [str(item["preview_path"]) for item in previews],
                    caption=[
                        f"{str(item.get('qc_modality') or 'image').upper()} · "
                        f"{item.get('qc_kind') or 'preview'}"
                        for item in previews
                    ],
                    width="stretch",
                )
            st.download_button(
                "Download Artifact Run Report",
                json.dumps(report, indent=2),
                file_name=f"artifact_experiment_{action_id[:8]}_report.json",
                mime="application/json",
            )
        except (OSError, json.JSONDecodeError):
            st.warning("The artifact run report could not be read.")


def _render_blur_actions() -> None:
    st.caption(
        "Ask Chat to optimize blurry regions or bright glare/debris. The worker uses the real "
        "Annotation code and your saved project settings."
    )
    actions = list_blur_actions()
    if not actions:
        st.info("No artifact experiment has been prepared yet.")
        return
    options = {
        f"{item['created_at'][:19]} · {item.get('artifact_kind', 'blur')} · {item['image_name']} · {item['trial_count']} trials · "
        f"{_status_label(str(item['state']))} · {item['action_id'][:8]}": item["action_id"]
        for item in actions
    }
    current = str(st.session_state.get(BLUR_ACTION_ID_KEY) or "")
    labels = list(options)
    index = next((position for position, label in enumerate(labels) if options[label] == current), 0)
    selected = st.selectbox("Saved artifact experiments", labels, index=index)
    action_id = options[selected]
    st.session_state[BLUR_ACTION_ID_KEY] = action_id
    _render_saved_blur_action(action_id)


def _action_label(item: dict[str, object]) -> str:
    details = action_details(str(item["action_id"]))
    title = action_title(details.get("plan") or {})
    return (
        f"{_status_label(str(item.get('state') or 'unknown'))} · {title} · "
        f"{str(item.get('created_at') or '')[:19]} · {str(item['action_id'])[:8]}"
    )


def _progress_fraction(status: dict[str, object], events: list[dict[str, object]]) -> float:
    state = str(status.get("state") or "")
    if state == "completed":
        return 1.0
    messages = [
        str(status.get("stage") or ""),
        str(status.get("message") or ""),
        *(f"{item.get('stage', '')} {item.get('message', '')}" for item in events[-10:]),
    ]
    for message in reversed(messages):
        for pattern in (r"trial\s+(\d+)\s*/\s*(\d+)", r"channel\s+(\d+)\s*/\s*(\d+)", r"completed\s+(\d+)\s+of\s+(\d+)", r"(\d+)\s*/\s*(\d+)"):
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match and int(match.group(2)) > 0:
                return min(0.98, max(0.01, int(match.group(1)) / int(match.group(2))))
    return 0.03 if state in {"launching", "running"} else 0.0


def _input_rows(plan: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    nested_inputs = plan.get("inputs")
    if isinstance(nested_inputs, dict):
        for name, value in nested_inputs.items():
            rows.append({"Input": str(name).replace("_", " ").title(), "Path": str(value)})
    for key in (
        "codex_path",
        "he_path",
        "image_path",
        "boundaries_path",
        "tissue_boundary_path",
        "project_config_path",
    ):
        value = plan.get(key)
        if value:
            rows.append({"Input": key.removesuffix("_path").replace("_", " ").title(), "Path": str(value)})
    return rows


def _parameter_rows(plan: dict[str, object]) -> list[dict[str, object]]:
    action_type = str(plan.get("action_type") or "")
    if action_type in {"optimize_blur_detection", "optimize_artifact_detection"}:
        trials = default_bright_trials() if action_type == "optimize_artifact_detection" else default_blur_trials()
        rows: list[dict[str, object]] = []
        for number, trial in enumerate(trials[: int(plan.get("maximum_trials") or len(trials))], start=1):
            rows.append({"Trial": number, "Name": trial["name"], **dict(trial.get("overrides") or {})})
        return rows
    if action_type in {"registration", "register_codex_he_pair"}:
        return [
            {
                "CODEX MPP": plan.get("codex_mpp"),
                "H&E MPP": plan.get("he_mpp"),
                "Reference channel": plan.get("reference_channel"),
                "Keypoints": plan.get("n_keypoints"),
                "Rotation": plan.get("pre_rotation_deg"),
                "Overwrite": plan.get("overwrite"),
            }
        ]
    if action_type == "extract_rois":
        return [
            {
                "ROIs": len(plan.get("rois") or []),
                "Channels": len(plan.get("channels") or []),
                "MPP": plan.get("mpp"),
                "Overwrite": plan.get("overwrite"),
            }
        ]
    parameters = plan.get("parameters")
    if isinstance(parameters, dict):
        return [{"Parameter": key, "Value": value} for key, value in parameters.items()]
    return []


def _start_selected_action(action_id: str, action_type: str) -> None:
    method = "Session auto approval" if preapproval_enabled() else "Action Notebook approval"
    if action_type in {"registration", "register_codex_he_pair"}:
        pid = launch_registration_action(action_id)
    elif action_type == "extract_rois":
        pid = launch_roi_action(action_id, approval_method=method)
    elif action_type in {"optimize_blur_detection", "optimize_artifact_detection"}:
        pid = launch_blur_action(action_id, approval_method=method)
    else:
        pid = launch_general_action(action_id, approval_method=method)
    st.session_state[LIVE_ACTION_KEY] = action_id
    AssistantStore().log_activity(
        "action_notebook_started",
        f"Action {action_id[:8]} started from its notebook.",
        details={"pid": pid, "action_type": action_type},
    )


def _render_plan_column(action_id: str, plan: dict[str, object], status: dict[str, object]) -> None:
    st.markdown("#### Plan and parameters")
    request = str(plan.get("user_request") or "")
    if request:
        st.caption("What you asked")
        st.write(request)
    inputs = _input_rows(plan)
    if inputs:
        st.caption("Inputs")
        st.dataframe(inputs, hide_index=True, width="stretch")
    output = plan.get("run_output_dir") or plan.get("output_dir") or plan.get("final_output")
    if output:
        st.caption("Output")
        st.code(str(output), language="text", wrap_lines=True)

    parameters = _parameter_rows(plan)
    if parameters:
        st.caption("Settings to use or test")
        st.dataframe(parameters, hide_index=True, width="stretch", height=min(390, 36 + 35 * len(parameters)))
    tools = plan.get("registered_tools") or []
    if tools:
        st.caption("SPACEMAP tools selected for this job")
        st.write(", ".join(str(item) for item in tools))
    if plan.get("code"):
        with st.expander("Reproducible Python prepared for this action", expanded=False):
            st.code(str(plan["code"]), language="python", line_numbers=True)

    if str(status.get("state") or "") == "planned":
        if preapproval_enabled():
            approved = True
            st.info("Auto approval is on. You can start this checked plan now.")
        else:
            approved = st.checkbox(
                "I reviewed this plan. Start it.",
                key=f"action_notebook_approve_{action_id}",
            )
        if st.button(
            "Start Action",
            type="primary",
            width="stretch",
            disabled=not approved,
            key=f"action_notebook_start_{action_id}",
        ):
            try:
                _start_selected_action(action_id, str(plan.get("action_type") or ""))
                st.success("The action started in the background.")
                st.rerun()
            except (RegistrationActionError, ROIActionError, BlurActionError, GeneralActionError, OSError, ValueError) as error:
                st.error(str(error))


def _render_results_column(action_id: str, details: dict[str, object]) -> None:
    report = details.get("report") or {}
    status = details.get("status") or {}
    state = str(status.get("state") or "unknown")
    st.markdown("#### Results and previews")
    message = str(status.get("message") or "")
    if state == "completed":
        st.success(message or "The action completed.")
    elif state == "failed":
        st.error(message or "The action failed. Open the technical log below.")
    elif state in {"running", "launching"}:
        st.info(message or "The action is running. Results appear here as they are created.")
    else:
        st.caption(message or "Results will appear after the action starts.")

    result = report.get("result") if isinstance(report, dict) else {}
    result = result if isinstance(result, dict) else {}
    comparison = result.get("comparison") or (report.get("comparison") if isinstance(report, dict) else None) or []
    if comparison:
        st.caption("Compared settings")
        st.dataframe(comparison, hide_index=True, width="stretch", height=315)
        selected = result.get("selected_trial")
        if selected:
            st.success(f"Best machine-screened setting: {selected}")

    visual_review = result.get("visual_review") or {}
    if isinstance(visual_review, dict):
        answer = str(visual_review.get("answer") or "").strip()
        if answer:
            with st.expander("AI visual assessment", expanded=True):
                st.markdown(answer)
        elif visual_review.get("error"):
            st.warning(f"Visual assessment was not completed: {visual_review['error']}")

    outputs = collect_action_outputs(details)
    images = outputs["images"]
    # Old artifact runs created decorated and undecorated copies. Prefer the
    # neutral copy unless this exact action explicitly requested scale bars.
    plan = details.get("plan") or {}
    plain = [path for path in images if "scale_bar" not in path.stem.casefold()]
    if plain and not bool(plan.get("include_scale_bars")):
        images = plain
    if images:
        st.caption("Figures created by this action")
        st.image([str(path) for path in images[:8]], caption=[path.name for path in images[:8]], width="stretch")
        if len(images) > 8:
            st.caption(f"Showing 8 of {len(images)} previews. The notebook export records up to 12.")

    for path in outputs["tables"][:4]:
        rows = read_table_preview(path)
        with st.expander(f"Table · {path.name}", expanded=not bool(comparison)):
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch", height=300)
            else:
                st.caption("The table is empty or still being written.")
            st.caption(str(path))

    file_rows = action_file_progress(action_id)
    if file_rows:
        with st.expander("Output files", expanded=not bool(report)):
            st.dataframe(file_rows, hide_index=True, width="stretch", height=min(300, 36 + 35 * len(file_rows)))


@st.fragment(run_every=2.0)
def _render_action_notebook_workspace(action_id: str) -> None:
    details = action_details(action_id)
    plan = details.get("plan") or {}
    status = details.get("status") or {}
    if not plan:
        st.error("This action record is missing.")
        return
    events = progress_events(action_id, limit=200)
    state = str(status.get("state") or "unknown")
    stage = str(status.get("stage") or "Waiting")
    elapsed = status.get("elapsed_seconds")

    metrics = st.columns([1, 2, 1, 1])
    metrics[0].metric("State", _status_label(state))
    metrics[1].metric("Current step", stage)
    metrics[2].metric("Elapsed", f"{float(elapsed):,.1f} s" if elapsed is not None else "—")
    metrics[3].metric("Action", action_id[:8])
    progress = _progress_fraction(status, events)
    st.progress(progress, text=f"Overall progress · {round(progress * 100):d}%")

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        _render_plan_column(action_id, plan, status)
    with right:
        _render_results_column(action_id, details)

    with st.expander("Technical log", expanded=state == "failed"):
        event_text = "\n".join(
            f"{item.get('created_at') or item.get('time') or ''}  {item.get('stage', '')}: {item.get('message', '')}"
            for item in events[-60:]
        )
        worker_text = action_log(action_id, max_characters=80_000)
        text = event_text
        if worker_text and worker_text not in text:
            text = (text + "\n" + worker_text).strip()
        st.code(text or "Waiting for the first worker message…", language="text", wrap_lines=True, height=300)
        st.download_button(
            "Download Technical Log",
            text,
            file_name=f"SPACEMAP_action_{action_id[:8]}.log",
            mime="text/plain",
            key=f"action_notebook_log_{action_id}",
        )


def _render_imported_notebook(payload: bytes) -> None:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        st.error("This is not a readable notebook file.")
        return
    st.info("This notebook's live action record is not on this computer. Showing its saved copy.")
    for cell in document.get("cells") or []:
        source = "".join(cell.get("source") or [])
        if cell.get("cell_type") == "markdown":
            st.markdown(source)
        elif cell.get("cell_type") == "code":
            if source.strip():
                with st.expander("Saved notebook code", expanded=False):
                    st.code(source, language="python")
            for output in cell.get("outputs") or []:
                data = output.get("data") or {}
                encoded = data.get("image/png") or data.get("image/jpeg")
                if encoded:
                    import base64

                    st.image(base64.b64decode(encoded))
                text = output.get("text") or data.get("text/plain")
                if text:
                    st.code("".join(text) if isinstance(text, list) else str(text), language="text")


def _render_tool_catalog() -> None:
    with st.expander("Available SPACEMAP tools", expanded=False):
        counts = tool_counts()
        st.caption(
            f"{counts['total']:,} functions are indexed from the current SPACEMAP source. "
            "You normally do not need to choose them—the AI selects the relevant ones."
        )
        if st.button("Refresh tool list", key="_spacemap_ai_refresh_tool_registry"):
            counts = refresh_tool_registry()
            st.success(f"Tool list refreshed: {counts['total']:,} functions.")
        query = st.text_input("Find a tool (optional)", key="_spacemap_ai_tool_search")
        rows = []
        for item in list_tools():
            searchable = f"{item.name} {item.callable_path} {item.summary}".casefold()
            if query.strip() and query.casefold() not in searchable:
                continue
            rows.append({"Area": item.page, "Tool": item.name, "Purpose": item.summary})
            if len(rows) >= 100:
                break
        st.dataframe(rows, hide_index=True, width="stretch", height=320)


def render_registration_actions(chat_renderer: Callable[[], None] | None = None) -> None:
    """Render one action as a simple, live, notebook-like workspace."""
    st.markdown("## Action Notebook")
    st.caption(
        "One job, one status, and one place for its inputs, tested settings, figures, tables, and record."
    )
    approval, history = st.columns([1, 2])
    with approval:
        st.radio(
            "Before actions run",
            options=["ask", "auto"],
            format_func=lambda value: "Ask me" if value == "ask" else "Run checked plans automatically",
            key=ACTION_APPROVAL_MODE_KEY,
            horizontal=True,
            help="Overwrite still happens only when your request clearly permits it.",
        )

    actions = list_all_actions(limit=200)
    if not actions:
        st.info("No action exists yet. Ask the AI in Chat to perform a SPACEMAP task.")
        if chat_renderer is not None:
            with st.expander("Chat with SPACEMAP AI", expanded=True):
                chat_renderer()
        _render_tool_catalog()
        return

    action_ids = [str(item["action_id"]) for item in actions]
    pending = str(st.session_state.pop(ACTION_NOTEBOOK_PENDING_KEY, "") or "")
    active = next((str(item["action_id"]) for item in actions if item.get("state") in {"launching", "running"}), "")
    last_active = str(st.session_state.get(ACTION_NOTEBOOK_LAST_ACTIVE_KEY) or "")
    if pending in action_ids:
        st.session_state[ACTION_NOTEBOOK_KEY] = pending
    elif str(st.session_state.get(ACTION_NOTEBOOK_KEY) or "") not in action_ids:
        st.session_state[ACTION_NOTEBOOK_KEY] = active or action_ids[0]
    elif active and active != last_active:
        st.session_state[ACTION_NOTEBOOK_KEY] = active
    if active:
        st.session_state[ACTION_NOTEBOOK_LAST_ACTIVE_KEY] = active

    with history:
        selected_id = st.selectbox(
            "Open current or previous action",
            action_ids,
            format_func=lambda value: _action_label(next(item for item in actions if item["action_id"] == value)),
            key=ACTION_NOTEBOOK_KEY,
        )
    st.session_state[LIVE_ACTION_KEY] = selected_id

    details = action_details(selected_id)
    st.markdown(f"### {action_title(details.get('plan') or {})}")
    _render_action_notebook_workspace(selected_id)

    with st.expander("Save or open an Action Notebook", expanded=False):
        st.caption(
            "The notebook saves this job's inputs, settings, result tables, small figures, and log. "
            "Large microscopy images remain as file paths, so the notebook stays small."
        )
        notebook = action_notebook_json(selected_id)
        buttons = st.columns(2)
        with buttons[0]:
            st.download_button(
                "Download .ipynb",
                notebook,
                file_name=f"SPACEMAP_action_{selected_id[:8]}.ipynb",
                mime="application/x-ipynb+json",
                width="stretch",
            )
        with buttons[1]:
            if st.button("Save beside the action record", width="stretch"):
                try:
                    saved = save_action_notebook(selected_id)
                    st.success(f"Saved: {saved}")
                except (OSError, ValueError) as error:
                    st.error(str(error))
        uploaded = st.file_uploader("Open a saved .ipynb", type=["ipynb"], key="_spacemap_ai_action_notebook_upload")
        if uploaded is not None:
            payload = uploaded.getvalue()
            try:
                imported_id = notebook_action_id(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                imported_id = ""
            if imported_id in action_ids:
                if st.button("Open this run", key="_spacemap_ai_open_imported_action"):
                    st.session_state[ACTION_NOTEBOOK_PENDING_KEY] = imported_id
                    st.rerun()
                st.caption(f"This notebook belongs to saved action {imported_id[:8]}.")
            else:
                _render_imported_notebook(payload)

    if chat_renderer is not None:
        with st.expander(
            "Chat with SPACEMAP AI about this action",
            expanded=True,
        ):
            st.caption(
                f"Current action: {selected_id[:8]}. Ask for an explanation or request changed settings here. "
                "A changed plan is checked and saved as a new action; running records are never silently rewritten."
            )
            chat_renderer()

    _render_tool_catalog()
