"""Intermediate images must land in a subfolder, not loose in the output directory."""

import pytest

from spacemap_gui.tabs.tab1_registration import (
    DEFAULT_INTERMEDIATE_SUBDIR,
    _intermediate_subdir_name,
)


@pytest.fixture
def session(monkeypatch):
    import streamlit as st

    store = {}
    monkeypatch.setattr(st, "session_state", store, raising=False)
    return store


@pytest.mark.parametrize("stored", [None, "", "   ", "/", "\\"])
def test_blank_or_missing_falls_back_to_default(session, stored):
    """A blank field — including one persisted by an older project bundle — uses the default."""
    if stored is not None:
        session["tab1_batch_intermediate_subdir"] = stored

    assert _intermediate_subdir_name("tab1_batch_intermediate_subdir") == DEFAULT_INTERMEDIATE_SUBDIR
    assert DEFAULT_INTERMEDIATE_SUBDIR == "intermediate_images"


def test_user_supplied_name_is_respected(session):
    session["multi_intermediate_subdir"] = "  scratch/  "

    assert _intermediate_subdir_name("multi_intermediate_subdir") == "scratch"
