"""Panel edits preserve Tab 7's file format, history and unsaved raw thresholds."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap.resolve_live import LiveResolve, Panel, Settings
from spacemap.viewer import live_resolve, panels
from spacemap_gui.viewer import registry
from test_viewer import client  # noqa: F401


def payload():
    return {
        "criteria_cols": ["Criteria 1", "Criteria 2"],
        "table": [
            {"Phenotype": "A", "Criteria 1": ["M0"], "Criteria 2": [], "Supportive": [], "Must not have": []},
            {"Phenotype": "B", "Criteria 1": ["M1"], "Criteria 2": [], "Supportive": [], "Must not have": []},
        ],
    }


def test_panel_save_keeps_order_and_an_exact_backup_and_rejects_stale_edits(tmp_path):
    path = tmp_path / "panel.json"
    original = json.dumps(payload())
    path.write_text(original)
    initial = panels.read(path)
    changed = payload()
    changed["table"].reverse()
    changed["table"][0]["Supportive"] = ["M0"]
    saved = panels.save(path, changed, initial["revision"], ["M0", "M1"])
    assert (tmp_path / saved["backup"]).read_text() == original
    assert list(Panel.from_payload(saved["panel"]).definitions) == ["B", "A"]
    with pytest.raises(ValueError, match="changed elsewhere"):
        panels.save(path, payload(), initial["revision"], ["M0", "M1"])
    assert panels.read(path)["revision"] == saved["revision"]


@pytest.mark.parametrize("change", ["duplicate", "unknown", "invalid-list", "empty"])
def test_invalid_panel_edits_do_not_touch_the_original(tmp_path, change):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload()))
    initial = panels.read(path)
    changed = payload()
    if change == "duplicate":
        changed["table"][1]["Phenotype"] = "A"
    elif change == "unknown":
        changed["table"][0]["Criteria 1"] = ["MISSING"]
    elif change == "invalid-list":
        changed["table"][0]["Criteria 1"] = "M0"
    else:
        changed["table"] = []
    with pytest.raises(ValueError):
        panels.save(path, changed, initial["revision"], ["M0", "M1"])
    assert panels.read(path) == initial
    assert not list(tmp_path.glob("*.before-viewer-*"))


def test_panel_routes_refresh_live_assignments_and_keep_unsaved_thresholds(client, tmp_path):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload()))
    table = tmp_path / "table.csv"
    pd.DataFrame(
        {"batch_id": ["roi"] * 4, "label": range(1, 5), "M0": [10, 50, 100, 200], "M1": [20, 60, 100, 100]}
    ).to_csv(table, index=False)
    token = registry.register_object("resolve", table=str(table), roi="roi", panel=str(path))
    initial = client.get(f"/resolve/{token}/panel").json()
    assert "M0" in initial["markers"]
    client.post(f"/resolve/{token}/open", json={})
    client.post(f"/resolve/{token}/thresholds", json={"thresholds": {"M0": 42.375}})
    changed = initial["panel"]
    changed["table"][0]["Phenotype"] = "Renamed A"
    response = client.post(f"/resolve/{token}/panel", json={"panel": changed, "revision": initial["revision"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert "Renamed A" in result["live"]["phenotypes"]
    assert next(m for m in result["live"]["markers"] if m["marker"] == "M0")["threshold"] == 42.375
    assert (tmp_path / result["backup"]).exists()
    assert client.post(f"/resolve/{token}/panel", json={"panel": changed, "revision": "stale"}).status_code == 400


def test_reading_panel_never_starts_the_resolve_engine(client, tmp_path, monkeypatch):
    import builtins

    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload()))
    table = tmp_path / "table.csv"
    pd.DataFrame({"batch_id": ["roi"], "label": [1], "M0": [5]}).to_csv(table, index=False)
    token = registry.register_object("resolve", table=str(table), roi="roi", panel=str(path))
    original = builtins.__import__

    def import_without_engine(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "spacemap.viewer" and "live_resolve" in (fromlist or ()):
            raise AssertionError("Reading panel names must not load the RESOLVE engine.")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_engine)
    response = client.get(f"/resolve/{token}/panel")
    assert response.status_code == 200, response.text
    assert response.json()["panel"] == payload()


def test_near_threshold_highlight_includes_boundaries_and_excludes_invalid_cells():
    raw = np.array([[89.9], [90], [100], [110], [110.1], [np.nan]])
    engine = LiveResolve(raw, ["M0"], Panel({"A": [["M0"]]}), Settings())
    session = live_resolve.Session(
        spec={},
        roi="roi",
        markers=["M0"],
        labels=np.arange(1, 7),
        raw=raw,
        engine=engine,
        thresholds={"M0": 100},
        source={"M0": "saved"},
    )
    live_resolve.assign(session, {})
    regular, _ = live_resolve.phenotype_lut(session)
    highlighted, _ = live_resolve.phenotype_lut(session, near=True)
    np.testing.assert_array_equal(highlighted[[2, 3, 4]], np.tile([255, 212, 0, 255], (3, 1)))
    assert highlighted[1, 3] == highlighted[5, 3] == 50
    assert highlighted[6, 3] == 0
    assert regular[1, 3] == 255
