"""Commands validate project names, preserve cell identity and support guarded Undo."""

import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio, labels
from spacemap.viewer import cells, display, reference
from spacemap_gui.studio import commands as bus, schemas
from spacemap_gui.studio.names import NameChoice, match
from spacemap_gui.viewer import registry


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "on")
    from test_viewer_cell_profiles import marker_image

    root = tmp_path / "study"
    roi = root / "outputs/rois"
    roi.mkdir(parents=True)
    image = roi / "P11_roi_2.ome.tiff"
    marker_image(image, np.ones((2, 64, 64), np.uint16))
    table = root / "cells.csv"
    pd.DataFrame(
        {
            "batch_id": ["P11_roi_2"] * 4 + ["other"],
            "label": [1, 2, 3, 4, 1],
            "x_centroid": [1.0, 2.0, 3.0, 4.0, 100.0],
            "y_centroid": [5.0, 6.0, 7.0, 8.0, 100.0],
            "M0": [1.0, 2.0, 3.0, 4.0, 100.0],
            "M1": [3.0, 2.0, 1.0, 4.0, 200.0],
            "phenotype_ml_consensus": ["CD8", "CD8", "B", "B", "Other"],
            "phenotype_ref1": [None] * 5,
        }
    ).to_csv(table, index=False)
    dataio.ensure_fast_copy(table, force=True)
    token = registry.register(image)
    cell = registry.register_object("cells", image=str(image), roi="P11_roi_2", tables={"t0": str(table)})
    display_path = root / "display/image.json"
    display_path.parent.mkdir()
    display.save_set(
        display.sets_path(display_path), "Immune", [{"name": "M1", "color": "#00ff00", "low": 3, "high": 20}]
    )
    disp = registry.register_object("display", path=str(display_path), image=str(image))
    panel = root / "panel.json"
    panel.write_text(
        json.dumps(
            {
                "criteria_cols": ["Criteria 1"],
                "table": [{"Phenotype": "T", "Criteria 1": ["M0"]}, {"Phenotype": "B", "Criteria 1": ["M1"]}],
            }
        )
    )
    thresholds = root / "thresholds.csv"
    pd.DataFrame([{"batch_id": "P11_roi_2", "M0": 2, "M1": 2}]).to_csv(thresholds, index=False)
    resolve = registry.register_object(
        "resolve", cells=cell, table=str(table), roi="P11_roi_2", panel=str(panel), thresholds=str(thresholds)
    )
    spec = {
        "id": "command-view",
        "title": "P11",
        "layers": [
            {
                "id": "image",
                "type": "image",
                "token": token,
                "display_token": disp,
                "channels": [
                    {"index": 0, "name": "M0", "visible": True, "low": 1, "high": 10, "color": "#4f7dff"},
                    {"index": 1, "name": "M1", "visible": False, "low": 1, "high": 10, "color": "#ff0000"},
                ],
            },
            {
                "id": "cells",
                "type": "cells",
                "token": cell,
                "image_token": token,
                "resolve": resolve,
                "table": "t0",
                "tables": [{"id": "t0", "name": "Cells"}],
                "color_by": "phenotype_ml_consensus",
            },
        ],
    }
    channel = "bus-" + uuid.uuid4().hex
    registry.publish(
        channel, spec, project={"root": str(root), "mappings": {}, "unified": str(table), "colors": {}, "tab": "tab7"}
    )
    return channel, table, root, spec


def run(setup, name, args=None, **kwargs):
    return bus.execute(setup[0], {"name": name, "args": args or {}, **kwargs})


def test_names_report_matches_and_never_guess_ambiguous():
    corrections = []
    assert match("P11 roi 2", ["P11_roi_2.ome.tiff"], "image", corrections=corrections) == "P11_roi_2.ome.tiff"
    assert corrections
    with pytest.raises(NameChoice):
        match("CD3", ["CD3e", "CD3d"], "marker")
    with pytest.raises(NameChoice):
        match("none", ["CD3e"], "marker")


@pytest.mark.parametrize(
    "name,args",
    [
        ("show_markers", {"markers": ["m0"]}),
        ("apply_marker_set", {"set": "immune"}),
        ("colour_by", {"column": "phenotype ml consensus"}),
        ("filter_cells", {"column": "phenotype_ml_consensus", "phenotypes": ["cd8"]}),
        ("density", {"phenotype": "CD8"}),
        ("zoom_to_cell", {"cell": 3}),
        ("count_cells", {}),
        ("set_threshold", {"marker": "M0", "value": 3}),
        ("save_image", {}),
    ],
)
def test_each_view_command_has_result_and_history(setup, name, args):
    result = run(setup, name, args)
    assert result["status"] == "done", result
    assert result["message"] and result["command_id"]
    if name != 'count_cells':  # a read-only count does not publish a new view
        assert result['spec']['command']['name'] == name  # polling must retain the command's intent
    assert result["undo"] == (name not in {"count_cells", "save_image"})
    records = [json.loads(line) for line in (setup[2] / "logs/studio_commands.jsonl").read_text().splitlines()]
    assert records[-1]["name"] == name and records[-1]["status"] == "done"
    if result["undo"]:
        undone = run(setup, "undo", {"command_id": result["command_id"]})
        assert undone["status"] == "done", undone


def test_counts_scope_and_other_roi_values(setup):
    assert run(setup, "count_cells", {"phenotype": "cd8"})["data"]["total"] == 2
    assert run(setup, "count_cells", {"phenotype": "Other", "scope": "project"})["data"]["total"] == 1
    assert run(setup, "count_cells", {"scope": "project"})["data"]["total"] == 5


def test_assign_requires_confirmation_and_undo_is_guarded(setup):
    args = {"column": "phenotype_ref1", "phenotype": "T", "cells": [1, 2]}
    assert run(setup, "assign_phenotype", args)["status"] == "needs_confirmation"
    assert labels.load(setup[1]).empty
    result = run(setup, "assign_phenotype", args, confirmed=True)
    assert result["status"] == "done", result
    assert result["data"]["count"] == 2
    assert pd.isna(cells.roi_values(str(setup[1]), "other", "phenotype_ref1")[1]).all()
    undo = run(setup, "undo", {"command_id": result["command_id"]}, confirmed=True)
    assert undo["status"] == "done", undo
    assert pd.isna(cells.roi_values(str(setup[1]), "P11_roi_2", "phenotype_ref1")[1]).all()
    changed = run(setup, "assign_phenotype", args, confirmed=True)
    reference.save(str(setup[1]), "P11_roi_2", "phenotype_ref1", {1: "B"}, "newer-reference")
    assert run(setup, "undo", {"command_id": changed["command_id"]}, confirmed=True)["status"] == "error"


def test_save_threshold_undo_preserves_newer_files(setup):
    old = (setup[2] / "thresholds.csv").read_bytes()
    run(setup, "set_threshold", {"marker": "M0", "value": 4})
    assert run(setup, "save_thresholds")["status"] == "needs_confirmation"
    saved = run(setup, "save_thresholds", confirmed=True)
    assert saved["status"] == "done", saved
    assert (setup[2] / "thresholds.csv").read_bytes() != old
    assert run(setup, "undo", {"command_id": saved["command_id"]}, confirmed=True)["status"] == "done"
    assert (setup[2] / "thresholds.csv").read_bytes() == old
    newer = run(setup, "save_thresholds", confirmed=True)
    (setup[2] / "thresholds.csv").write_text("newer data")
    assert run(setup, "undo", {"command_id": newer["command_id"]}, confirmed=True)["status"] == "error"


def test_duplicate_request_is_applied_once_and_source_column_is_protected(setup):
    args = {"column": "phenotype_ref1", "phenotype": "T", "cells": [1]}
    first = run(setup, "assign_phenotype", args, confirmed=True, request_id="same-save-request")
    assert first["status"] == "done", first
    again = run(setup, "assign_phenotype", args, confirmed=True, request_id="same-save-request")
    assert again["replayed"] and again["command_id"] == first["command_id"]
    assert run(setup, "assign_phenotype", {**args, "column": "label"}, confirmed=True)["status"] == "error"
    assert run(setup, "assign_phenotype", {**args, "cells": [1000]}, confirmed=True)["status"] == "error"


@pytest.mark.parametrize(
    "name,args",
    [
        ("open_image", {"image": "missing"}),
        ("show_markers", {"markers": ["missing"]}),
        ("apply_marker_set", {"set": "missing"}),
        ("colour_by", {"column": "missing"}),
        ("filter_cells", {"column": "phenotype_ml_consensus", "phenotypes": ["missing"]}),
        ("density", {"phenotype": "missing"}),
        ("zoom_to_cell", {"cell": 9999}),
        ("count_cells", {"column": "missing"}),
        ("set_threshold", {"marker": "missing", "value": 2}),
        ("undo", {"command_id": "missing-command"}),
    ],
)
def test_wrong_names_and_ids_do_not_change_the_view(setup, name, args):
    before = registry.session_view(setup[0], mark_seen=False)
    result = run(setup, name, args)
    assert result["status"] in {"error", "needs_choice"}, result
    assert registry.session_view(setup[0], mark_seen=False) == before


def test_schema_rejects_extra_fields_nan_and_bad_types():
    for name, args in [
        ("open_image", {"image": "valid", "code": "print(1)"}),
        ("set_threshold", {"marker": "M0", "value": float("nan")}),
        ("zoom_to_cell", {"cell": True}),
        ("show_markers", {"markers": "M0"}),
    ]:
        with pytest.raises(ValueError):
            schemas.validate(name, args)


def test_open_image_matches_project_names(setup):
    result = run(setup, "open_image", {"image": "P11 roi 2"})
    assert result["status"] == "done", result
    assert result["corrections"] and result["spec"]["layers"][0]["type"] == "image"


def test_readonly_blocks_assign_and_undo_writes(setup):
    made = run(setup, "assign_phenotype", {"column": "phenotype_ref1", "phenotype": "T", "cells": [1]}, confirmed=True)
    assert made["status"] == "done"
    view = registry.session_view(setup[0], mark_seen=False)
    registry.publish(setup[0], view["spec"], project={**registry.session_project(setup[0]), "read_only": True})
    assert run(setup, "undo", {"command_id": made["command_id"]}, confirmed=True)["status"] == "error"
    assert (
        run(setup, "assign_phenotype", {"column": "phenotype_ref1", "phenotype": "B", "cells": [1]}, confirmed=True)[
            "status"
        ]
        == "error"
    )


def test_source_table_change_blocks_old_label_undo(setup):
    made = run(setup, "assign_phenotype", {"column": "phenotype_ref1", "phenotype": "T", "cells": [1]}, confirmed=True)
    frame = pd.read_csv(setup[1])
    frame.loc[0, "phenotype_ref1"] = "T"
    frame.to_csv(setup[1], index=False)
    assert run(setup, "undo", {"command_id": made["command_id"]}, confirmed=True)["status"] == "error"


def test_stale_view_is_rejected_and_retried_receipt_never_rewinds(setup):
    version = registry.session_view(setup[0], mark_seen=False)["version"]
    first = run(setup, "show_markers", {"markers": ["M1"]}, version=version, request_id="first-action")
    assert first["status"] == "done"
    assert run(setup, "show_markers", {"markers": ["M0"]}, version=version)["status"] == "error"
    latest = run(setup, "show_markers", {"markers": ["M0"]})
    again = run(setup, "show_markers", {"markers": ["M1"]}, request_id="first-action")
    assert again["replayed"] and again["version"] == latest["version"] and again["effects"] == []


def test_client_view_is_limited_to_registered_layers_and_undo_preserves_zoom(setup):
    view = {
        "id": setup[3]["id"],
        "layers": [
            {"id": "image", "channels": [{"index": 99, "name": "made-up"}]},
            {"id": "extra", "token": "unregistered"},
        ],
        "bounds": [2, 3, 40, 50],
    }
    result = run(setup, "show_markers", {"markers": ["M1"]}, view=view)
    assert result["status"] == "done"
    undone = run(setup, "undo", {"command_id": result["command_id"]})
    assert undone["spec"]["bounds"] == view["bounds"]
    assert all(layer["id"] != "extra" for layer in undone["spec"]["layers"])
    assert all(channel["index"] < 2 for channel in undone["spec"]["layers"][0]["channels"])


def test_command_and_ask_routes(setup):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api

    with TestClient(api.app) as client:
        assert client.get("/studio/missing/commands").status_code == 404
        assert len(client.get(f"/studio/{setup[0]}/commands").json()["commands"]) == len(schemas.COMMANDS)
        result = client.post(
            f"/studio/{setup[0]}/command", json={"name": "count_cells", "args": {"phenotype": "CD8"}}
        ).json()
        assert result["status"] == "done" and result["data"]["total"] == 2
        proposal = client.post(f"/studio/{setup[0]}/ask", json={"text": "show M0", "use_model": False}).json()
        assert proposal["command"]["args"]["markers"] == ["M0"]
        assert client.post("/studio/missing/ask", json={"text": "show CD8"}).status_code == 400


def test_ambiguous_images_do_not_change_view(setup, monkeypatch):
    from spacemap_gui.viewer import ui

    monkeypatch.setattr(
        ui,
        "window_images",
        lambda _: [
            {"id": "a", "name": "P11.ome.tiff", "sample": "P11", "group": "ROIs"},
            {"id": "b", "name": "P11.ome.tiff", "sample": "P11", "group": "Registered"},
        ],
    )
    before = registry.session_view(setup[0], mark_seen=False)
    result = run(setup, "open_image", {"image": "P11"})
    assert result["status"] == "needs_choice" and len(result["choices"]) == 2
    assert registry.session_view(setup[0], mark_seen=False) == before


def test_explicit_current_image_preserves_other_view_preferences(setup):
    view = {"id": setup[3]["id"], "layers": [{"id": "cells", "visible": False, "color_by": "phenotype_ref1"}]}
    result = run(setup, "show_markers", {"image": "P11 roi 2", "markers": ["M1"]}, view=view)
    assert result["status"] == "done", result
    layer = next(item for item in result["spec"]["layers"] if item["type"] == "cells")
    assert not layer["visible"] and layer["color_by"] == "phenotype_ref1"


def saved_resolve_workspace(context):
    """A classic-compatible project panel, independent of the open image layer."""
    _, table, root, _ = context
    cache = root / "outputs/phenotyping/cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "resolve_pheno_panel.json").write_bytes((root / "panel.json").read_bytes())
    pd.DataFrame([{"batch_id": roi, "M0": 2, "M1": 2} for roi in pd.read_csv(table).batch_id.unique()]).to_csv(cache / "resolve_thresholds_table.csv", index=False)


def test_job_commands_validate_inputs_and_use_registered_tables(setup, monkeypatch):
    from spacemap_gui.studio import jobs

    frame = pd.read_csv(setup[1])
    frame["cell_id"] = range(len(frame))
    frame.to_csv(setup[1], index=False)
    dataio.ensure_fast_copy(setup[1], force=True)
    saved_resolve_workspace(setup)
    captured = []

    def start(project, kind, settings, **kwargs):
        captured.append((project, kind, settings))
        return {"id": "a" * 32, "status": "starting"}

    monkeypatch.setattr(jobs, "start", start)
    for name, args in [
        ("run_resolve_all", {"column": "phenotype_new"}),
        ("run_mr_ml", {"column": "phenotype_mr_test", "reference_column": "phenotype_ref1"}),
    ]:
        assert run(setup, name, args)["status"] == "needs_confirmation"
        result = run(setup, name, args, confirmed=True)
        assert result["status"] == "done", result
        assert result["data"]["job"]["status"] == "starting"
        assert captured[-1][2]["table"] == str(setup[1])
    assert run(setup, "run_resolve_all", {"column": "phenotype_ml_consensus"}, confirmed=True)["status"] == "error"
    assert (
        run(setup, "run_mr_ml", {"column": "phenotype_new", "reference_column": "not_present"}, confirmed=True)[
            "status"
        ]
        == "needs_choice"
    )


def test_completed_job_layer_and_undo(setup, monkeypatch):
    from spacemap_gui.studio import jobs

    monkeypatch.setattr(
        jobs,
        "status",
        lambda root, id: {"status": "done", "result": {"table": str(setup[1]), "column": "phenotype_ml_consensus"}},
    )
    result = run(setup, "show_job_result", {"job_id": "a" * 32})
    assert result["status"] == "done" and result["spec"]["layers"][-1]["color_by"] == "phenotype_ml_consensus"
    assert len(run(setup, "undo", {"command_id": result["command_id"]})["spec"]["layers"]) == 2


def test_classic_assistant_counts_in_chat_and_opens_visual_results(setup, monkeypatch):
    from spacemap_gui.studio import classic
    from spacemap_gui.viewer import ui

    state = {}
    monkeypatch.setattr(classic.st, "session_state", state)
    monkeypatch.setattr(ui, "studio_link", lambda page: "/studio")
    monkeypatch.setattr(ui, "_session_id", lambda: setup[0])
    monkeypatch.setattr(ui, "viewer_window_open", lambda: False)
    messages = []

    class Store:
        def add_message(self, conversation, role, text, metadata=None):
            messages.append(text)

    assert classic.submit("count CD8 cells", "tab7", Store(), "chat")
    assert "CD8: 2" in messages[-1] and ui._OPEN_KEY not in state
    assert classic.submit("show markers M1", "tab7", Store(), "chat")
    assert state[ui._OPEN_KEY] and state["_viewer_command_result"]["effects"][0]["type"] == "channels"
    assert not classic.submit("show me the registration parameters", "tab1", Store(), "chat")
    assert not classic.submit("load the project", "tab0", Store(), "chat")


def test_classic_unknown_marker_names_offer_choices_without_changing_view(setup, monkeypatch):
    from spacemap_gui.studio import classic
    from spacemap_gui.viewer import ui

    state = {}
    monkeypatch.setattr(classic.st, "session_state", state)
    monkeypatch.setattr(ui, "studio_link", lambda page: "/studio")
    monkeypatch.setattr(ui, "_session_id", lambda: setup[0])

    class Store:
        def add_message(self, *args, **kwargs):
            pass

    before = registry.session_view(setup[0], mark_seen=False)
    assert classic.submit("show markers M", "tab7", Store(), "chat")
    assert state[classic.PENDING]["result"]["status"] == "needs_choice"
    assert registry.session_view(setup[0], mark_seen=False) == before


def test_reference_training_command_and_ambiguous_reference_names(setup, monkeypatch):
    from spacemap_gui.studio import jobs

    # A training source must contain categorical labels; the shared fixture is blank.
    frame = pd.read_csv(setup[1])
    frame["phenotype_ref1"] = ["CD8", "CD8", "B", "B", None]
    frame.to_csv(setup[1], index=False)
    dataio.ensure_fast_copy(setup[1], force=True)
    calls = []
    monkeypatch.setattr(
        jobs,
        "start",
        lambda root, kind, settings, **kw: calls.append((kind, settings)) or {"id": "b" * 32, "status": "starting"},
    )
    result = run(
        setup, "train_reference_model", {"reference_column": "phenotype ref1", "checked_only": True}, confirmed=True
    )
    assert result["status"] == "done" and calls[0][0] == "train_reference_model"
    assert calls[0][1]["training_column"] == "phenotype_ref1" and calls[0][1]["checked_only"]
    assert (
        run(setup, "train_reference_model", {"reference_column": "phenotype"}, confirmed=True)["status"]
        == "needs_choice"
    )
    assert run(setup, "train_reference_model", {"reference_column": "phenotype_ref1"})["status"] == "needs_confirmation"


def test_a_steps_settings_change_through_its_own_checks_and_undo(setup):
    from spacemap_gui.studio import leiden, step_settings

    root = setup[2]
    asked = run(setup, "change_settings", {"method": "leiden", "changes": {"resolution": 1.5}})
    assert asked["status"] == "needs_confirmation"
    done = run(setup, "change_settings", {"method": "leiden", "changes": {"resolution": 1.5}}, confirmed=True)
    assert done["status"] == "done" and done["data"]["settings"] == {"method": "leiden", "changes": {"resolution": 1.5}}
    assert leiden.values(root)["resolution"] == 1.5 and step_settings.current(root, "leiden")["resolution"] == 1.5
    for wrong in (
        {"method": "leiden", "changes": {"resolution": 99}},
        {"method": "leiden", "changes": {"colour": 1}},
        {"method": "scimap", "changes": {"a": 1}},
    ):
        assert run(setup, "change_settings", wrong, confirmed=True)["status"] == "error"
    assert run(setup, "undo", {"command_id": done["command_id"]}, confirmed=True)["status"] == "done"
    assert leiden.values(root)["resolution"] == leiden.DEFAULTS["resolution"]
    assert step_settings.current(root, "batch_correction")["correction_mode"] == "cycombine"


def test_a_triad_table_is_drawn_as_a_density_map(setup):
    channel, table, root, spec = setup
    triads = root / "outputs/spatial_analysis/triad/triad_instances_local.csv"
    triads.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "batch_id": "P11_roi_2",
            "triad_label": ["T::B::T", "T::B::T", "B::B::B"],
            "triad_centroid_x": [2.0, 3, 4],
            "triad_centroid_y": [5.0, 6, 7],
        }
    ).to_csv(triads, index=False)
    cells_layer = next(layer for layer in spec["layers"] if layer["type"] == "cells")
    old = {key: value for key, value in registry.resolve_object(cells_layer["token"], "cells").items() if key != "kind"}
    cells_layer["token"] = registry.register_object("cells", **{**old, "tables": {"t0": str(table), "p0": str(triads)}})
    cells_layer["density_tables"] = [
        {"id": "t0", "name": "Cells"},
        {"id": "p0", "name": "Triads / triad_instances_local.csv"},
    ]
    registry.publish(channel, spec, project=registry.session_project(channel))
    found = run(setup, "density", {"table": "Triads / triad_instances_local.csv", "phenotype": "T::B::T"})
    assert found["status"] == "done" and found["data"]["count"] == 2, found["message"]
    added = found["effects"][0]["layer"]
    assert (
        added["type"] == "density"
        and added["table"] == "p0"
        and added["fcol"] == "triad_label"
        and added["fvals"] == ["T::B::T"]
    )
    assert run(setup, "density", {"table": "Triads / triad_instances_local.csv", "phenotype": "nothing like it"})[
        "status"
    ] in {
        "needs_choice",
        "error",
    }
