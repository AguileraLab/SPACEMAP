"""The merged IDs/table survive adoption and do not depend on the image's local cell table."""

import pandas as pd
import pytest

from spacemap import dataio, pending_columns
from spacemap_gui.studio import jobs, leiden, phenotyping_inputs as inputs, projects, tasks
from spacemap_gui.studio.settings import read_config, revision
from spacemap_gui.viewer import registry
from test_studio_commands import run, setup  # noqa: F401


def combined(context):
    channel, local, root, _ = context
    frame = pd.read_csv(local)
    frame.insert(0, "cell_id", [f"cell_{i}" for i in range(len(frame))])
    frame["slide"] = "S1"
    frame["slide_id"] = "P1T1"
    path = root / "outputs/batch_corrected/new.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    inputs.use_table(channel, path)
    return path, frame


def test_adoption_is_persistent_and_csv_caches_are_not_inputs(setup):  # noqa: F811
    path, _ = combined(setup)
    channel, local, root, _ = setup
    assert registry.session_project(channel)["unified"] == str(path)
    config = read_config(root)
    assert config["tab_configurations"]["tab7_phenotyping"]["global_pheno_file"] == path.name
    assert projects._context(root)["unified"] == str(path)
    assert registry.take_table_choice(channel) == {"root": str(root), "table": str(path)}
    assert registry.take_table_choice(channel) is None
    # The displayed image still points at local cells: that must not control the next clustering run.
    cell = registry.session_view(channel)["spec"]["layers"][1]
    assert registry.resolve_object(cell["token"], "cells")["tables"]["t0"] == str(local)
    csv = path.parent / "older.csv"
    csv.write_text("cell_id\na\n")
    csv.with_suffix(".csv.parquet").touch()
    assert inputs.table_files(path.parent) == [str(csv), str(path)]


@pytest.mark.parametrize("empty_view", [False, True])
def test_leiden_uses_combined_ids_even_with_a_different_or_absent_image(setup, monkeypatch, empty_view):  # noqa: F811
    path, frame = combined(setup)
    channel, _, root, _ = setup
    if empty_view:
        registry.publish(channel, {"id": "empty", "layers": []}, project=registry.session_project(channel))
    leiden.save(root, {"markers": ["M0", "M1"]}, revision(leiden.values(root)))
    started = []
    monkeypatch.setattr(jobs, "start", lambda root, kind, settings, **kwargs: started.append(settings) or {"id": "x"})
    result = run(setup, "run_leiden", confirmed=True)
    assert result["status"] == "done", result
    assert started[0]["table"] == str(path)
    assert started[0]["pipeline"]["full_input_path"] == str(path)
    assert started[0]["cell_id_col"] == "cell_id"
    assert len(dataio.load(started[0]["table"], columns=["cell_id"])) == len(frame)


def test_roles_use_actual_columns_and_result_coverage_refreshes(setup, monkeypatch):  # noqa: F811
    path, frame = combined(setup)
    channel, _, root, _ = setup
    frame = frame.rename(columns={"batch_id": "image_name", "slide": "physical_slide", "cell_id": "global_cell"})
    frame.to_parquet(path, index=False)
    roles = {"roi": "image_name", "slide": "physical_slide", "cell_id": "global_cell"}
    inputs.save_roles(root, str(path), roles)
    info = inputs.plan(root, {}, str(path))
    assert info["roles"] == roles and info["cells"] == 5
    assert len(info["rows"]) == 2 and info["rows"][0]["slide"] == "S1"
    assert info["result_columns"] == ["phenotype_ml_consensus", "phenotype_ref1"]
    for values, expected in [(["T", None, None, None, None], 1), (["T"] * 5, 4)]:
        pending_columns.save(path, {"phenotype_test_clusters": values}, anchor_column="global_cell", anchor_values=frame.global_cell)
        found = inputs.plan(root, {}, str(path))
        row = next(row for row in found["rows"] if row["roi"] == "P11_roi_2")
        assert row["results"]["phenotype_test_clusters"] == expected
    leiden.save(root, {"markers": ["M0", "M1"]}, revision(leiden.values(root)))
    started = []
    monkeypatch.setattr(jobs, "start", lambda root, kind, settings, **kwargs: started.append(settings) or {"id": "x"})
    assert run(setup, "run_leiden", confirmed=True)["status"] == "done"
    assert started[0]["pipeline"]["cell_id_col"] == "global_cell" and started[0]["roi_col"] == "image_name"
    with pytest.raises(ValueError, match="repeated"):
        inputs.save_roles(root, str(path), {**roles, "cell_id": "label"})


@pytest.mark.parametrize("bad_ids", [["a", "a", "c", "d", "e"], ["a", "b", "c", "d", None], ["a", "b", "c", "d", " "]])
def test_duplicate_or_missing_merged_ids_are_rejected_before_a_run(setup, bad_ids):  # noqa: F811
    path, frame = combined(setup)
    frame["cell_id"] = bad_ids
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing or repeated"):
        tasks.validate_table({"table": str(path), "output_col": "phenotype_new", "markers": ["M0"]})


def test_changing_folders_does_not_silently_select_another_table(setup):  # noqa: F811
    path, _ = combined(setup)
    channel, _, root, _ = setup
    empty = root / "empty"
    empty.mkdir()
    inputs.save_folders(root, {"tables": str(empty)})
    found = inputs.plan(root, {}, str(path))
    assert found["path"] == str(path) and found["tables"] == []
    assert registry.session_project(channel)["unified"] == str(path)
    with pytest.raises(ValueError, match="file"):
        inputs.save_folders(root, {"tables": str(path)})


def test_api_adopts_correction_and_disambiguates_slide_columns(setup, monkeypatch):  # noqa: F811
    from starlette.testclient import TestClient
    from spacemap_gui.viewer import api

    path, _ = combined(setup)
    channel, local, root, _ = setup
    registry.session_use_table(channel, str(local))
    monkeypatch.setattr(jobs, "status", lambda *args: {"kind": "correct_batches", "status": "done", "result": {"output": str(path)}})
    client = TestClient(api.app)
    response = client.post(f"/studio/{channel}/correction/use", json={"job_id": "x"})
    assert response.status_code == 200, response.text
    assert projects._context(root)["unified"] == str(path)
    plan = client.get(f"/studio/{channel}/phenotyping_inputs").json()
    assert plan["roles"] == {"roi": "batch_id", "slide": "slide", "cell_id": "cell_id"}
    markers = client.get(f"/studio/{channel}/marker_choices?method=leiden").json()
    assert markers["roles"]["slide"] == "slide" and markers["groups"] == ["batch_id", "slide", "slide_id"]
    assert not {"slide", "slide_id"} & {m["name"] for m in markers["markers"]}


def test_classic_receives_the_studio_table_choice_once(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.viewer import ui

    path, _ = combined(setup)
    channel, _, root, _ = setup
    state = {ui._SESSION_KEY: channel, "active_project_root": str(root), "active_project": {},
             "global_pheno_file": "old.csv", "bc_last_run_file": "old.csv"}
    monkeypatch.setattr(ui.st, "session_state", state)
    ui.apply_studio_table_choice()
    assert state["global_pheno_file"] == path.name and state["global_pheno_dir"] == str(path.parent)
    assert state["active_project"]["tab_configurations"]["tab7_phenotyping"]["global_pheno_file"] == path.name
    state["global_pheno_file"] = "a_later_classic_choice.csv"
    ui.apply_studio_table_choice()
    assert state["global_pheno_file"] == "a_later_classic_choice.csv"


def test_ambiguous_image_folder_keeps_table_controls_usable(setup):  # noqa: F811
    path, _ = combined(setup)
    _, _, root, _ = setup
    images = root / "outputs/rois"
    (images / "P11_roi_2_bg_subtracted.ome.tiff").touch()
    plan = inputs.plan(root, {}, str(path))
    assert plan["path"] == str(path) and plan["cells"] == 5 and plan["columns"]
    assert any("Two ROI images" in text for text in plan["warnings"])
