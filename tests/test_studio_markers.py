"""Markers by name, raw or batch corrected, and the saved sets of the viewer and Tab 7."""

import json

import pytest

from spacemap_gui.studio import markers

COLUMNS = [
    "cell_id",
    "batch_id",
    "area",
    "x_centroid",
    "CD45",
    "CD45_corr",
    "CD3",
    "CD3_corr",
    "PanCK",
    "phenotype_leiden",
    "CD3_norm",
    "umap_1",
]


def test_each_marker_is_listed_once_with_what_it_has():
    found = markers.panel(COLUMNS, "_corr")
    assert found == [
        {"name": "CD45", "raw": True, "corrected": True},
        {"name": "CD3", "raw": True, "corrected": True},
        {"name": "PanCK", "raw": True, "corrected": False},
    ]
    assert markers.columns_for(["CD45", "CD3"], "corrected", "_corr", COLUMNS) == ["CD45_corr", "CD3_corr"]
    with pytest.raises(ValueError, match="PanCK_corr"):
        markers.columns_for(["PanCK"], "corrected", "_corr", COLUMNS)


def test_the_suffix_is_tab6s_or_the_tables():
    config = {"tab_configurations": {"tab6_batch_correction": {"bc_config": {"corrected_suffix": "_bc"}}}}
    assert markers.suffix_of(config, ["CD3", "CD3_bc"]) == "_bc"
    assert markers.suffix_of({}, ["CD3", "CD3_corrected"]) == "_corrected"
    assert markers.suffix_of({}, ["CD3"]) == "_corr"
    assert markers.panel(["CD3", "CD3_bc"], "_bc") == [{"name": "CD3", "raw": True, "corrected": True}]


def test_sets_come_from_the_viewer_and_tab7_and_the_choice_is_read_back(tmp_path):
    display = tmp_path / "outputs/viewer/display"
    display.mkdir(parents=True)
    (display / "marker_sets.json").write_text(
        json.dumps({"sets": {"phenotyping_markers": [{"name": "CD45"}, {"name": "CD3"}]}})
    )
    config = {
        "tab_configurations": {
            "tab7_phenotyping": {"tab7_marker_workspace": {"marker_panels": {"all_corrected": ["CD45_corr", "PanCK"]}}}
        }
    }
    found = markers.choices(tmp_path, config, COLUMNS, current=["CD45_corr", "CD3_corr"])
    assert found["source"] == "corrected" and found["chosen"] == ["CD45", "CD3"] and found["suffix"] == "_corr"
    assert found["sets"] == [
        {"name": "phenotyping_markers", "from": "Image viewer", "markers": ["CD45", "CD3"]},
        {"name": "all_corrected", "from": "Tab 7", "markers": ["CD45", "PanCK"]},
    ]
    assert markers.choices(tmp_path, {}, COLUMNS, current=["CD45"])["source"] == "raw"
    assert markers.choices(tmp_path, {}, ["CD45"], current=[])["source"] == "raw"
