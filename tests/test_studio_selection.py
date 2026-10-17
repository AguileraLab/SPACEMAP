import json
from pathlib import Path

import numpy as np
import pytest

from spacemap import dataio, labels
from spacemap_gui.studio import selection
from test_studio_commands import setup, run  # noqa: F401


def rule(marker="M0", value=2, scale="Raw", scope="image"):
    return {
        "conditions": [{"marker": marker, "transform": scale, "op": ">", "value": value}],
        "scope": scope,
        "join": "all",
    }


@pytest.mark.parametrize(
    "scale,value,expected", [("Raw", 2, 2), ("Arcsinh", float(np.arcsinh(2 / 5)), 2), ("Z1", 0, 2), ("Z2", 0, 1)]
)
def test_rules_and_scales_exact(setup, scale, value, expected):
    result = run(setup, "select_cells", {"rule": rule(value=value, scale=scale)})
    assert result["status"] == "done", result
    assert result["data"]["selection"]["count"] == expected
    assert result["spec"]["layers"][-1]["selection"] == result["data"]["selection"]["id"]
    assert result["spec"]["layers"][-1]["color_by"] == ""
    assert run(setup, "undo", {"command_id": result["command_id"]})["status"] == "done"


def test_join_all_any_and_invalid_names(setup):
    value = rule()
    value["conditions"].append({"marker": "M1", "transform": "Raw", "op": ">", "value": 2})
    assert run(setup, "select_cells", {"rule": value})["data"]["selection"]["count"] == 1
    value["join"] = "any"
    assert run(setup, "select_cells", {"rule": value})["data"]["selection"]["count"] == 3
    assert run(setup, "select_cells", {"rule": rule("missing")})["status"] == "needs_choice"
    assert run(setup, "select_cells", {"rule": rule(value=float("nan"))})["status"] == "error"


def test_refine_click_draw_and_restore(setup):
    run(setup, "select_cells", {"rule": rule()})
    assert run(setup, "refine_selection", {"cells": [1], "mode": "add"})["data"]["selection"]["count"] == 3
    assert run(setup, "refine_selection", {"cells": [4], "mode": "remove"})["data"]["selection"]["count"] == 2
    result = run(setup, "refine_selection", {"points": [[0, 4], [3, 4], [3, 8], [0, 8]], "mode": "add"})
    assert result["status"] == "done", result
    assert result["data"]["selection"]["count"] == 3
    assert run(setup, "refine_selection", {"cells": [999], "mode": "add"})["status"] == "error"


def test_all_rois_exact_identity_assign_and_guarded_undo(setup):
    before = Path(setup[1]).read_bytes()
    result = run(setup, "select_cells", {"rule": rule(scope="project")})
    assert result["data"]["selection"]["count"] == 3
    assigned = run(setup, "assign_selection", {"column": "phenotype_ref1", "phenotype": "chosen"}, confirmed=True)
    assert assigned["status"] == "done", assigned
    assert dataio.load(setup[1], columns=["phenotype_ref1"])["phenotype_ref1"].fillna("").tolist() == [
        "",
        "",
        "chosen",
        "chosen",
        "chosen",
    ]
    assert Path(setup[1]).read_bytes() == before
    assert run(setup, "undo", {"command_id": assigned["command_id"]}, confirmed=True)["status"] == "done"
    assert dataio.load(setup[1], columns=["phenotype_ref1"])["phenotype_ref1"].fillna("").tolist() == [""] * 5
    assigned = run(setup, "assign_selection", {"column": "phenotype_ref1", "phenotype": "chosen"}, confirmed=True)
    from spacemap.viewer import reference

    reference.save(setup[1], "other", "phenotype_ref1", {1: "newer"}, "external-request")
    assert run(setup, "undo", {"command_id": assigned["command_id"]}, confirmed=True)["status"] == "error"
    assert dataio.load(setup[1], columns=["phenotype_ref1"])["phenotype_ref1"].fillna("").tolist() == [
        "",
        "",
        "chosen",
        "chosen",
        "newer",
    ]


def test_library_reapply_heatmap_and_layer_mask(setup):
    result = run(setup, "select_cells", {"rule": rule()})
    token = result["data"]["selection"]["id"]
    masked = selection.tile_mask(token, str(setup[1]), "P11_roi_2", np.array([[0, 1, 2], [3, 4, 999]]))
    assert masked.tolist() == [[0, 0, 0], [3, 4, 0]]
    heat = run(setup, "selection_heatmap")["data"]["heatmap"]
    assert heat["counts"] == [2, 2]
    values = np.array([[1, 3], [2, 2], [3, 1], [4, 4]], dtype=float)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    expected = dict(zip(["M0", "M1"], z[2:].mean(axis=0)))
    np.testing.assert_allclose(heat["values"][0], [expected[marker] for marker in heat["markers"]])
    assert run(setup, "save_rule", {"rule_name": "Immune"}, confirmed=True)["status"] == "done"
    assert run(setup, "save_rule", {"rule_name": "Immune"}, confirmed=True)["status"] == "error"
    assert run(setup, "apply_rule", {"rule_name": "immune", "scope": "project"})["data"]["selection"]["count"] == 3
    assert json.loads((setup[2] / "outputs/studio/rules.json").read_text())[0]["name"] == "Immune"
    with pytest.raises(ValueError, match="another viewer"):
        selection.get(token, channel="other-viewer")


def test_source_change_rejects_selection_and_assignment(setup):
    result = run(setup, "select_cells", {"rule": rule()})
    labels.append(setup[1], [{"row": 0, "label": 1, "column": "M0", "value": 100}])
    assert (
        run(setup, "assign_selection", {"column": "phenotype_ref1", "phenotype": "bad"}, confirmed=True)["status"]
        == "error"
    )
    with pytest.raises(ValueError, match="changed"):
        selection.get(result["data"]["selection"]["id"])


def test_selection_retry_and_interrupted_undo_are_safe(setup, monkeypatch):
    from spacemap.viewer import reference

    run(setup, "select_cells", {"rule": rule(scope="project")})
    assigned = run(
        setup,
        "assign_selection",
        {"column": "phenotype_ref1", "phenotype": "chosen"},
        confirmed=True,
        request_id="selection-save-once",
    )
    assert run(
        setup,
        "assign_selection",
        {"column": "phenotype_ref1", "phenotype": "chosen"},
        confirmed=True,
        request_id="selection-save-once",
    )["replayed"]
    with bus_database() as db:
        undo = json.loads(
            db.execute(
                "SELECT undo FROM commands WHERE channel=? AND id=?", (setup[0], assigned["command_id"])
            ).fetchone()[0]
        )
    append = labels.append
    calls = []

    def interrupted(*args, **kwargs):
        append(*args, **kwargs)
        if not calls:
            calls.append(True)
            raise OSError("interrupted after append")

    monkeypatch.setattr(labels, "append", interrupted)
    with pytest.raises(OSError):
        reference.undo_groups(
            undo["table"], undo["column"], undo["previous"], undo["expected"], "retry-undo", **undo["keys"]
        )
    assert (
        reference.undo_groups(
            undo["table"], undo["column"], undo["previous"], undo["expected"], "retry-undo", **undo["keys"]
        )["restored"]
        == 3
    )
    assert dataio.load(setup[1], columns=["phenotype_ref1"])["phenotype_ref1"].fillna("").tolist() == [""] * 5


def bus_database():
    from spacemap_gui.studio.commands import _database

    return _database()


def test_reference_summary_includes_phenotype_and_roi(setup):
    run(setup, "select_cells", {"rule": rule(scope="project")})
    preview = run(setup, "assign_selection", {"phenotype": "T"})
    assert (
        preview["status"] == "needs_confirmation"
        and "phenotype_ref1" in preview["message"]
        and "3 selected cells" in preview["message"]
    )
    result = run(setup, "assign_selection", {"phenotype": "T"}, confirmed=True)
    assert result["data"]["reference_summary"]["rows"] == [
        {"roi": "P11_roi_2", "phenotype": "T", "cells": 2},
        {"roi": "other", "phenotype": "T", "cells": 1},
    ]
