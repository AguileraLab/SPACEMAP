"""Saving in the ROI tab.

"Save Pair Settings" was folded into Save Project, so committing a pair must gather the
same state the old button did — and must not discard settings written by older versions
of the tab, which would silently reset an existing project.
"""

import json

import pytest

from spacemap_gui.tabs.tab3_roi_extraction import (
    _commit_all_pair_settings,
    _commit_pair_settings,
    save_drawn_rois_for_extraction,
)


IMAGE = "canine_merged.ome.tiff"


@pytest.fixture
def session(monkeypatch):
    import streamlit as st

    store = {}
    monkeypatch.setattr(st, "session_state", store, raising=False)
    return store


def test_commit_gathers_markers_names_and_boundaries(session):
    session[f"tab3_m_exp_map_{IMAGE}"] = {"DAPI": True, "CD8": False}
    session[f"tab3_m_vis_{IMAGE}"] = {0: True}
    session[f"tab3_m_rename_{IMAGE}"] = {"CD8": "CD8a"}
    session[f"tab3_m_colors_{IMAGE}"] = {"DAPI": (0, 0, 255)}
    session[f"tab3_boundary_method_{IMAGE}"] = "Predefined GeoJSON boundaries"
    session["tab3_multi_roi_labels"] = {IMAGE: {0: "LN1"}}

    settings = _commit_pair_settings(IMAGE)

    assert settings["export_map"] == {"DAPI": True, "CD8": False}
    assert settings["rename_map"] == {"CD8": "CD8a"}
    assert settings["color_map"] == {"DAPI": (0, 0, 255)}
    assert settings["boundary_method"] == "Predefined GeoJSON boundaries"
    assert settings["roi_labels"] == {0: "LN1"}
    assert session["tab3_multi_pair_settings"][IMAGE] is settings


def test_commit_preserves_settings_from_older_projects(session):
    """An old project holds keys this version never writes; they must survive."""
    session["tab3_multi_pair_settings"] = {IMAGE: {"legacy_field": "keep me", "export_map": {"DAPI": True}}}
    session[f"tab3_m_rename_{IMAGE}"] = {"CD8": "CD8a"}

    settings = _commit_pair_settings(IMAGE)

    assert settings["legacy_field"] == "keep me"
    assert settings["export_map"] == {"DAPI": True}
    assert settings["rename_map"] == {"CD8": "CD8a"}


def test_commit_on_untouched_image_is_harmless(session):
    """An image the user never opened has no widget state and must not be corrupted."""
    settings = _commit_pair_settings("never_opened.ome.tiff")

    assert settings == {}


def test_commit_all_walks_every_pair(session):
    session["tab3_m_rename_a.tiff"] = {"X": "Y"}
    session["tab3_m_rename_b.tiff"] = {"P": "Q"}

    _commit_all_pair_settings([{"image": "a.tiff"}, {"image": "b.tiff"}, {"image": None}])

    assert session["tab3_multi_pair_settings"]["a.tiff"]["rename_map"] == {"X": "Y"}
    assert session["tab3_multi_pair_settings"]["b.tiff"]["rename_map"] == {"P": "Q"}


def _rect(left, top, width, height):
    return {"type": "rect", "left": left, "top": top, "width": width, "height": height}


def test_save_rois_overwrites_the_named_file(tmp_path):
    """Save ROIs replaces the file the boundaries came from."""
    source = tmp_path / "canine.geojson"
    source.write_text('{"type": "FeatureCollection", "features": []}')

    written = save_drawn_rois_for_extraction(
        objects=[_rect(0, 0, 10, 10)], labels=["LN1"], scale_x=1.0, scale_y=1.0,
        image_path="canine_merged.ome.tiff", geojson_dir=tmp_path, output_path=source,
    )

    assert written == source
    assert len(json.loads(source.read_text())["features"]) == 1


def test_save_rois_without_target_derives_the_name(tmp_path):
    written = save_drawn_rois_for_extraction(
        objects=[_rect(0, 0, 10, 10)], labels=["LN1"], scale_x=1.0, scale_y=1.0,
        image_path="canine_merged.ome.tiff", geojson_dir=tmp_path,
    )

    assert written.name == "canine_roi_bounding_boxes_coords.geojson"


def test_saved_rois_carry_names_qupath_can_read(tmp_path):
    written = save_drawn_rois_for_extraction(
        objects=[_rect(0, 0, 10, 10)], labels=["LN1"], scale_x=1.0, scale_y=1.0,
        image_path="canine_merged.ome.tiff", geojson_dir=tmp_path,
    )

    props = json.loads(written.read_text())["features"][0]["properties"]
    assert props["label"] == "LN1"
    assert props["name"] == "LN1"
