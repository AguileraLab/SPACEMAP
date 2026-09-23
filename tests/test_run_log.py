"""One run log for every tab: one switch, one CSV, time and resources per action."""

from __future__ import annotations

import csv
import json
from datetime import datetime

import pytest
import streamlit as st

from spacemap_gui import run_log


@pytest.fixture
def project(tmp_path):
    st.session_state.clear()
    st.session_state["active_project_root"] = str(tmp_path)
    yield tmp_path
    st.session_state.clear()


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_tab_uses_the_same_switch_and_it_starts_on(project):
    assert run_log.switch_key("tab7:leiden") == "tab7_create_log"
    assert run_log.enabled("tab7")
    st.session_state["tab7_create_log"] = False
    assert not run_log.enabled("tab7:leiden")
    assert run_log.enabled("tab1")


def test_a_pressed_button_is_recorded_with_time_and_resources(project):
    st.session_state["leiden_stop_job_btn"] = True   # the button pressed in this rerun
    st.session_state["leiden_resolution"] = 1.5
    with run_log.page_run("tab7"):
        sum(range(100_000))

    rows = _rows(run_log.log_path(project))
    assert len(rows) == 1
    row = rows[0]
    assert row["tab"] == "Tab 7 Phenotyping"
    assert row["action"] == "Stop this run"
    assert row["status"] == "completed"
    assert float(row["duration_s"]) >= 0
    assert float(row["peak_ram_gb"]) > 0
    assert json.loads(row["settings_json"])["leiden_resolution"] == 1.5


def test_quick_page_updates_without_a_button_are_not_logged(project):
    with run_log.page_run("tab3"):
        pass
    assert not run_log.log_path(project).exists()


def test_settings_changed_since_the_previous_run_are_listed(project):
    st.session_state["leiden_stop_job_btn"] = True
    st.session_state["leiden_resolution"] = 1.5
    with run_log.page_run("tab7"):
        pass
    st.session_state["leiden_resolution"] = 2.0
    with run_log.page_run("tab7"):
        pass
    last = _rows(run_log.log_path(project))[-1]
    assert json.loads(last["settings_changed_json"]) == {"leiden_resolution": [1.5, 2.0]}


def test_a_switched_off_tab_records_nothing(project):
    st.session_state["tab7_create_log"] = False
    st.session_state["leiden_stop_job_btn"] = True
    with run_log.page_run("tab7"):
        pass
    assert not run_log.log_path(project).exists()


def test_a_failure_is_recorded_and_still_raised(project):
    st.session_state["leiden_stop_job_btn"] = True
    with pytest.raises(ValueError):
        with run_log.page_run("tab7"):
            raise ValueError("bad input")
    row = _rows(run_log.log_path(project))[-1]
    assert row["status"] == "failed" and "bad input" in row["error"]


def test_background_jobs_can_write_a_row_without_a_session(tmp_path):
    usage = run_log.ResourceMonitor().start().stop()
    path = run_log.record_run(tmp_path, "tab7:leiden", "Leiden clustering (background run)",
                              started_at=datetime.now().astimezone(), usage=usage,
                              settings={"resolution": 1.5})
    rows = run_log.read_rows(tmp_path, "tab7")
    assert path and rows[-1]["action"] == "Leiden clustering (background run)"
