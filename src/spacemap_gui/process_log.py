"""Shared timestamped process logging for the SPACEMAP interface."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


PROCESS_LOGS_KEY = "_spacemap_process_logs"
MAX_PROCESS_LOG_ENTRIES = 1_000
CSV_FIELDS = (
    "Timestamp",
    "Context",
    "Event",
    "Status",
    "Message",
    "Duration (s)",
    "Details",
)


def get_process_log_context(context: str) -> str:
    """Return the top-level pipeline context for a tab or subtab context."""
    return str(context).split(":", 1)[0]


def process_logging_enabled(context: str) -> bool:
    """Events follow the tab's single "Record run log" switch."""
    from spacemap_gui import run_log

    return run_log.enabled(get_process_log_context(context))


def get_process_log_path(project_root: Path, context: str) -> Path:
    """Return the durable CSV path for one pipeline page's process log."""
    page_context = get_process_log_context(context)
    return project_root / "logs" / "process_logs" / f"{page_context}_process_log.csv"


def build_process_log_record(
    context: str,
    event: str,
    *,
    message: str = "",
    status: str = "info",
    duration_s: float | None = None,
    details: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, str]:
    """Build one stable, CSV-ready process log record."""
    recorded_at = timestamp or datetime.now()
    return {
        "Timestamp": recorded_at.strftime("%Y-%m-%d %H:%M:%S"),
        "Context": str(context),
        "Event": str(event),
        "Status": str(status),
        "Message": str(message),
        "Duration (s)": "" if duration_s is None else f"{float(duration_s):.3f}",
        "Details": json.dumps(details or {}, default=str, sort_keys=True),
    }


def _get_page_records(context: str) -> list[dict[str, str]]:
    page_context = get_process_log_context(context)
    logs = st.session_state.setdefault(PROCESS_LOGS_KEY, {})
    records = logs.setdefault(page_context, [])
    return records if isinstance(records, list) else []


def _append_record_to_csv(record: Mapping[str, str], project_root: Path, context: str) -> None:
    log_path = get_process_log_path(project_root, context)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        with open(log_path, "a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})
    except OSError:
        # Logging must never prevent the scientific operation from continuing.
        pass


def record_process_event(
    context: str,
    event: str,
    *,
    message: str = "",
    status: str = "info",
    duration_s: float | None = None,
    details: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, str] | None:
    """Record a timestamped interface event and append it to the project CSV log."""
    if not process_logging_enabled(context):
        return None

    record = build_process_log_record(
        context,
        event,
        message=message,
        status=status,
        duration_s=duration_s,
        details=details,
    )
    records = _get_page_records(context)
    records.append(record)
    if len(records) > MAX_PROCESS_LOG_ENTRIES:
        del records[:-MAX_PROCESS_LOG_ENTRIES]

    resolved_project_root = project_root or st.session_state.get("active_project_root")
    if resolved_project_root:
        _append_record_to_csv(record, Path(resolved_project_root), context)
    return record


def _records_to_csv(records: list[Mapping[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field, "") for field in CSV_FIELDS})
    return buffer.getvalue()


def render_process_log_panel(context: str, project_root: Path | None = None) -> None:
    """The tab's run log: recent runs with their time and resources, then events."""
    from spacemap_gui import run_log

    page_context = get_process_log_context(context)
    resolved_project_root = project_root or st.session_state.get("active_project_root")
    with st.expander("Run log", expanded=False):
        if not run_log.enabled(page_context):
            st.caption(f"'{run_log.LABEL}' is off for this tab, so new runs are not being recorded.")
        if resolved_project_root:
            runs = run_log.read_rows(resolved_project_root, page_context, limit=50)
            if runs:
                st.markdown("**Recent runs on this tab**")
                shown = ["started_at", "action", "status", "duration_s", "cpu_time_s",
                         "peak_ram_gb", "peak_gpu_memory_gb", "settings_changed_json"]
                st.dataframe([{k: row.get(k, "") for k in shown} for row in reversed(runs)],
                             hide_index=True, width="stretch")
            path = run_log.log_path(resolved_project_root)
            st.caption(f"All tabs, full detail: `{path}`")
            if path.exists():
                st.download_button(
                    "Download Run Log CSV",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/csv",
                    key=f"{page_context}_run_log_download",
                    width="stretch",
                )

        records = _get_page_records(page_context)
        if not records:
            st.caption("No step events have been recorded in this session yet.")
            return
        st.markdown("**Step events in this session**")
        st.dataframe(records[-200:], hide_index=True, width="stretch")
        st.download_button(
            "Download Step Events CSV",
            data=_records_to_csv(records),
            file_name=f"{page_context}_process_log.csv",
            mime="text/csv",
            key=f"{page_context}_process_log_download",
            width="stretch",
        )
        if resolved_project_root:
            st.caption(f"Project copy: {get_process_log_path(Path(resolved_project_root), page_context)}")
        if st.button("Clear Step Events Shown Here", key=f"{page_context}_process_log_clear", width="stretch"):
            st.session_state.setdefault(PROCESS_LOGS_KEY, {})[page_context] = []
            st.rerun()
