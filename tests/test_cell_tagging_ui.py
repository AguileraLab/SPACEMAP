"""CSV execution and form regression tests for standalone spatial tagging."""

import importlib
from pathlib import Path
import sys
import types

import pandas as pd
import pytest
from shapely.geometry import box

from test_cell_spatial_annotation import spatial, _feature, _geojson


@pytest.fixture
def tagging(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.syspath_prepend(str(source))
    # Use the real focused backend, without the package's optional GPU imports.
    if "spacemap" not in sys.modules:
        package = types.ModuleType("spacemap")
        package.__path__ = [str(source / "spacemap")]
        monkeypatch.setitem(sys.modules, "spacemap", package)
    monkeypatch.setitem(sys.modules, "spacemap.cell_spatial_annotation", spatial)
    # ``from spacemap import cell_spatial_annotation`` reads the package attribute, which an earlier
    # import of the real module has already set: point it at the same module.
    monkeypatch.setattr(sys.modules["spacemap"], "cell_spatial_annotation", spatial, raising=False)
    return importlib.import_module("spacemap_gui.tabs.cell_tagging_ui")


def _state(tagging, overwrite):
    return {
        tagging.RULES_KEY: spatial.rules_to_records([
            spatial.SpatialFeatureRule("tumor", distance=True),
        ]),
        tagging.BED_LABELS_KEY: ["tumor", "normal", "reactive"],
        tagging.OVERWRITE_COLUMNS_KEY: overwrite,
        tagging.MIN_BED_CELLS_KEY: 1,
    }


def test_tagging_requires_restart_if_old_backend_is_cached(tagging, monkeypatch):
    monkeypatch.delattr(spatial, "CLEAN_PATH_ENGINE")
    with pytest.raises(RuntimeError, match="Restart Streamlit"):
        tagging._validate_distance_engine(True)
    assert tagging._validate_distance_engine(False) == "Euclidean / legacy"


@pytest.mark.parametrize("overwrite", [True, False])
def test_merged_overwrite_clears_unmatched_failed_and_missing_id_rows(
    tmp_path, tagging, monkeypatch, overwrite,
):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
    ])
    malformed = tmp_path / "broken.geojson"
    malformed.write_text("invalid json")
    monkeypatch.setattr(tagging.st, "session_state", _state(tagging, overwrite))
    monkeypatch.setattr(tagging, "match_geojson", lambda key, *_: {
        "good": str(annotation), "failed": str(malformed),
    }.get(key))
    frame = pd.DataFrame({
        "image": ["missing", "good", "failed", "good", None],
        "x_centroid": [20, 20, 20, 150, 20], "y_centroid": [20] * 5,
        "tumor_bed": [999] * 5, "tumor_bed_id": [999] * 5,
        "tumor_distance": [999] * 5, "tumor": [999] * 5,
        "unselected_bed": [42] * 5, "cell_id": list("abcde"),
    })
    source = tmp_path / "cells.csv"
    destination = tmp_path / "tagged.csv"
    frame.to_csv(source, index=False)
    tagged, records = tagging.run_merged(
        str(source), "image", str(tmp_path), str(destination), "x_centroid", "y_centroid"
    )
    assert tagged["cell_id"].tolist() == list("abcde")
    assert tagged["unselected_bed"].eq(42).all()
    existing = ["tumor_bed", "tumor_bed_id", "tumor_distance", "tumor"]
    if overwrite:
        assert tagged.loc[[0, 2, 4], existing].isna().all().all()
        assert tagged.loc[3, existing[:-1]].isna().all()
        assert tagged.loc[3, "tumor"] == 0
        assert tagged.loc[1, "tumor_bed"] == 1
    else:
        assert tagged[existing].eq(999).all().all()
    assert tagged.loc[1, "normal_bed"] == 0
    assert tagged.loc[[0, 2, 3, 4], "normal_bed"].isna().all()
    assert any(r["Status"].startswith("failed") for r in records)
    assert pd.read_csv(source)["tumor_bed"].eq(999).all()
    pd.testing.assert_frame_equal(pd.read_csv(destination), tagged, check_dtype=False)


@pytest.mark.parametrize("overwrite", [True, False])
def test_per_image_passes_bed_selection_and_overwrite_to_engine(tmp_path, tagging, monkeypatch, overwrite):
    annotation = _geojson(tmp_path, [
        _feature("tissue", box(0, 0, 100, 100)),
        _feature("tumor", box(10, 10, 30, 30)),
    ])
    monkeypatch.setattr(tagging.st, "session_state", _state(tagging, overwrite))
    monkeypatch.setattr(tagging, "match_geojson", lambda *_: str(annotation))
    frame = pd.DataFrame({"x_centroid": [20, 150], "y_centroid": [20, 20], "tumor_bed": [999, 999]})
    source = tmp_path / "sample_cell_features.csv"
    frame.to_csv(source, index=False)
    records = tagging.run_per_image(
        str(tmp_path), [source.name], str(tmp_path), str(tmp_path / "out"),
        "_cell_features", "_tagged", "x_centroid", "y_centroid",
    )
    assert records[0]["Status"] == "ok"
    tagged = pd.read_csv(tmp_path / "out" / "sample_cell_features_tagged.csv")
    assert "reactive_bed" in tagged and "normal_bed_id" in tagged
    if overwrite:
        assert tagged.loc[0, "tumor_bed"] == 1
        assert pd.isna(tagged.loc[1, "tumor_bed"])
    else:
        assert tagged["tumor_bed"].eq(999).all()


def test_saved_pair_settings_migrate_to_bed_selection(tagging, monkeypatch):
    rules = [spatial.SpatialFeatureRule("tumor", bed=True, bed_target="normal")]
    state = {tagging.RULES_KEY: spatial.rules_to_records(rules)}
    monkeypatch.setattr(tagging.st, "session_state", state)
    assert tagging.selected_bed_labels() == ["tumor", "normal"]
    state[tagging.BED_LABELS_KEY] = []
    assert tagging.selected_bed_labels() == []  # Explicitly disabling survives reload.


def test_rules_form_saves_three_beds_and_previews_columns(tagging, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from spacemap_gui import persistence

    monkeypatch.setattr(persistence, "save_active_project_state", lambda *_: None)
    app = AppTest.from_string('''
from spacemap_gui.tabs.cell_tagging_ui import _render_rules_form
_render_rules_form(["tissue", "tumor", "normal", "reactive"], "tissue")
''').run(timeout=15)
    assert not app.exception
    app.multiselect[0].set_value(["tumor", "normal", "reactive"])
    app.checkbox(key="seg_tag_use_gpu_distances_widget").check()
    app.button[0].click().run(timeout=15)
    assert not app.exception
    assert app.session_state[tagging.BED_LABELS_KEY] == ["tumor", "normal", "reactive"]
    assert app.session_state[tagging.USE_GPU_KEY] is True
    assert any("reactive_bed" in caption.value and "normal_bed" in caption.value for caption in app.caption)
