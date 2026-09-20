"""Switching to a smaller unified CSV must not crash a data-bound widget.

Reproduces the reported failure: a subsample number_input remembered 150000 from
a large CSV, then the user switched to one with 5576 rows and Streamlit raised
StreamlitValueAboveMaxError before the tab could render.
"""

import pytest
import streamlit as st

from spacemap_gui.persistence import record_latest_unified_csv
from spacemap_gui.ui_utils import clamp_numeric_widget_state


@pytest.fixture(autouse=True)
def clean_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_a_value_above_the_new_maximum_is_pulled_down():
    st.session_state["umap_subsample"] = 150000

    result = clamp_numeric_widget_state("umap_subsample", 0, 5576)

    assert result == 5576
    assert st.session_state["umap_subsample"] == 5576


def test_a_value_below_the_new_minimum_is_pulled_up():
    st.session_state["k"] = 1

    assert clamp_numeric_widget_state("k", 2, 10) == 2
    assert st.session_state["k"] == 2


def test_a_value_already_in_range_is_untouched():
    st.session_state["k"] = 42

    assert clamp_numeric_widget_state("k", 0, 100) == 42
    assert st.session_state["k"] == 42


def test_integer_bounds_keep_an_integer_result():
    st.session_state["k"] = 150000.0

    result = clamp_numeric_widget_state("k", 0, 5576)

    assert isinstance(result, int) and result == 5576


def test_float_bounds_keep_a_float_result():
    st.session_state["k"] = 99.0

    result = clamp_numeric_widget_state("k", 0.0, 1.5)

    assert isinstance(result, float) and result == pytest.approx(1.5)


def test_an_unset_key_falls_back_without_writing_state():
    assert clamp_numeric_widget_state("missing", 0, 100, fallback=7) == 7
    assert "missing" not in st.session_state, "no stored value means nothing to correct"


def test_a_non_numeric_stored_value_is_discarded():
    st.session_state["k"] = "not a number"

    assert clamp_numeric_widget_state("k", 0, 100, fallback=5) == 5
    assert "k" not in st.session_state


def test_an_impossible_range_drops_the_stale_value():
    """An empty table can produce max < min; nothing can satisfy that."""
    st.session_state["k"] = 10

    assert clamp_numeric_widget_state("k", 5, 0, fallback=None) is None
    assert "k" not in st.session_state


def test_the_reported_scenario_end_to_end():
    """Large CSV -> small CSV -> the widget's stored value is renderable."""
    big_rows, small_rows = 150000, 5576

    clamp_numeric_widget_state("umap_subsample", 0, big_rows, fallback=big_rows)
    st.session_state["umap_subsample"] = big_rows          # user ran on the big file

    value = clamp_numeric_widget_state("umap_subsample", 0, small_rows)

    assert 0 <= value <= small_rows
    assert 0 <= st.session_state["umap_subsample"] <= small_rows


# ---------------------------------------------------------------------------
# Latest unified CSV
# ---------------------------------------------------------------------------


def test_switching_csv_records_it_on_the_project():
    st.session_state["active_project"] = {"unified_csv_filename": "old.csv"}

    record_latest_unified_csv("new.csv")

    assert st.session_state["active_project"]["unified_csv_filename"] == "new.csv"


def test_recording_never_writes_the_tab0_widget_key():
    """That key belongs to a Tab 0 widget; writing it from another page raises."""
    st.session_state["active_project"] = {"unified_csv_filename": "old.csv"}

    record_latest_unified_csv("new.csv")

    assert "proj_unified_csv_filename" not in st.session_state


def test_recording_is_a_no_op_without_a_project():
    record_latest_unified_csv("new.csv")   # must not raise

    assert "active_project" not in st.session_state


def test_blank_names_are_ignored():
    st.session_state["active_project"] = {"unified_csv_filename": "keep.csv"}

    record_latest_unified_csv("")
    record_latest_unified_csv(None)

    assert st.session_state["active_project"]["unified_csv_filename"] == "keep.csv"


def test_every_data_bound_widget_site_is_guarded():
    """Each widget whose bounds come from the data must clamp first."""
    from pathlib import Path

    guarded = {
        "src/spacemap_gui/tabs/tab9_visualization/umap_explorer.py": 2,
        "src/spacemap_gui/tabs/tab7_phenotyping/visualization.py": 2,
        "src/spacemap_gui/tabs/tab8_analysis/feature_screening.py": 1,
        "src/spacemap_gui/tabs/tab9_visualization/f1_strict_validation_ui.py": 1,
        "src/spacemap_gui/tabs/tab9_visualization/f1_paper_figures.py": 1,
        "src/spacemap_gui/tabs/tab6_batch_correction.py": 1,
    }
    for path, expected in guarded.items():
        source = Path(path).read_text()
        calls = source.count("clamp_numeric_widget_state(")
        # one import line plus the expected number of call sites
        assert calls >= expected, f"{path}: expected >= {expected} guarded widgets, found {calls}"
