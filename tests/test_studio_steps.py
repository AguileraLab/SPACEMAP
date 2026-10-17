import json

import pytest
from starlette.testclient import TestClient

from spacemap_gui.studio.steps import work_list, overview, LAYOUT
from spacemap_gui.viewer import registry, api


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    root = tmp_path / "study"
    root.mkdir()
    registry.publish("steps-test", {"id": "empty", "layers": []}, project={"root": str(root), "mappings": {}})
    return root


def test_pairs_and_saved_output_states_come_from_disk(project):
    for folder in ["data/codex", "data/he", "outputs/registered"]:
        (project / folder).mkdir(parents=True)
    (project / "data/codex/P11_codex.qptiff").touch()
    (project / "data/he/P11_he.qptiff").touch()
    (project / "data/codex/P12_codex.qptiff").touch()
    first = work_list("steps-test", "tab1")
    rows = {row["name"]: row for row in first["items"]}
    assert rows["P11"]["state"] == "not_run" and rows["P12"]["state"] == "needs_review"
    (project / "outputs/registered/P11_merged.ome.tiff").touch()
    assert next(row for row in work_list("steps-test", "tab1")["items"] if row["name"] == "P11")["state"] == "done"


@pytest.mark.parametrize("step", list(LAYOUT))
def test_each_step_offers_a_lightweight_work_list(project, step):
    result = work_list("steps-test", step)
    assert result["step"] == step and "items" in result and "outputs" in result
    assert result["actions"]["run"] == ("resolve_all" if step == "tab7" else "classic")


def test_rail_states_summarise_each_step(project):
    # The Studio rail shows done, partly done or not run for every step.
    for folder in ["data/codex", "data/he", "outputs/registered"]:
        (project / folder).mkdir(parents=True)
    for name in [
        "data/codex/P11_codex.qptiff",
        "data/he/P11_he.qptiff",
        "data/codex/P12_codex.qptiff",
        "data/he/P12_he.qptiff",
    ]:
        (project / name).touch()
    assert overview("steps-test")["states"]["tab1"] == "none"
    (project / "outputs/registered/P11_merged.ome.tiff").touch()
    assert overview("steps-test")["states"]["tab1"] == "partial"
    (project / "outputs/registered/P12_merged.ome.tiff").touch()
    states = overview("steps-test")["states"]
    assert states["tab1"] == "done" and set(states) == set(LAYOUT)
    with TestClient(api.app) as client:
        assert client.get("/studio/steps-test/overview").json()["states"]["tab1"] == "done"
        assert client.get("/studio/missing/overview").json()["states"]["tab1"] == "unknown"


def test_states_follow_real_project_file_names(project):
    # INNATE_all_09262026: feature tables are "<roi>.ome_cell_features.csv", ROIs are cut
    # from registered images when background removal was skipped, and the per-ROI
    # tables batch correction read were cleaned up after combining.
    for folder in [
        "outputs/registered",
        "outputs/background_subtracted",
        "outputs/rois",
        "outputs/segmentation",
        "outputs/batch_corrected",
    ]:
        (project / folder).mkdir(parents=True)
    (project / "outputs/registered/P11_merged.ome.tiff").touch()
    (project / "outputs/registered/P12_merged.ome.tiff").touch()
    (project / "outputs/rois/P11_roi_1.ome.tiff").touch()
    (project / "outputs/segmentation/P11_roi_1.ome_cell_features.csv").touch()
    (project / "outputs/batch_corrected/combined.csv").touch()
    rois = {row["name"]: row["state"] for row in work_list("steps-test", "tab3")["items"]}
    assert rois == {"P11": "done", "P12": "not_run"}
    assert {row["name"]: row["state"] for row in work_list("steps-test", "tab5")["items"]} == {"P11_roi_1": "done"}
    # Tab 6 reads Tab 5's annotated tables (outputs/annotated_csv); cleaned up, the combined table is listed.
    assert {row["name"]: row["state"] for row in work_list("steps-test", "tab6")["items"]} == {"combined": "done"}
    (project / "outputs/annotated_csv").mkdir()
    (project / "outputs/annotated_csv/P11_roi_1.ome_cell_features_annotated.csv").touch()
    assert {row["name"]: row["state"] for row in work_list("steps-test", "tab6")["items"]} == {"P11_roi_1": "done"}
    (project / "outputs/annotated_csv/P11_roi_1.ome_cell_features_annotated.csv").unlink()
    (project / "outputs/segmentation/P11_roi_1.ome_cell_features.csv").unlink()
    assert {row["name"]: row["state"] for row in work_list("steps-test", "tab6")["items"]} == {"combined": "done"}


def test_background_states_follow_tab2_files(project):
    # Tab 2 measures the ROI images; levels live in data/thresholds, images in background_subtracted.
    for folder in ["outputs/rois", "data/thresholds", "outputs/background_subtracted"]:
        (project / folder).mkdir(parents=True)
    for name in ["P11_roi_1", "P11_roi_2", "P12_roi_1"]:
        (project / f"outputs/rois/{name}.ome.tiff").touch()
    (project / "data/thresholds/P11_roi_1.ome.tiff_-_Merged_Image.csv").write_text("Marker,Polygon_1\nM0,1\n")
    (project / "data/thresholds/P11_roi_2.ome.tiff_-_Merged_Image.csv").write_text("Marker,Polygon_1\nM0,1\n")
    (project / "outputs/background_subtracted/P11_roi_2_bg_subtracted.ome.tiff").touch()
    rows = {row["name"]: row["state_label"] for row in work_list("steps-test", "tab2")["items"]}
    assert rows == {"P11_roi_1": "Levels saved", "P11_roi_2": "Background removed", "P12_roi_1": "Not run"}
    assert overview("steps-test")["states"]["tab2"] == "partial"


def test_roi_states_follow_tab3_files(project):
    for folder in ["outputs/registered", "data/boundaries", "outputs/rois"]:
        (project / folder).mkdir(parents=True)
    for name in ["P11", "P12", "P13"]:
        (project / f"outputs/registered/{name}_merged.ome.tiff").touch()
    feature = {
        "type": "Feature",
        "properties": {"name": "roi_1"},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]]},
    }
    two = {"type": "FeatureCollection", "features": [feature, {**feature, "properties": {"name": "roi_2"}}]}
    for name in ["P11", "P12"]:
        (project / f"data/boundaries/{name}_roi_bounding_boxes_coords.geojson").write_text(json.dumps(two))
    for name in ["P11_roi_1", "P11_roi_2", "P12_roi_1"]:
        (project / f"outputs/rois/{name}.ome.tiff").touch()
    rows = {row["name"]: row["state_label"] for row in work_list("steps-test", "tab3")["items"]}
    assert rows == {"P11": "2 ROIs extracted", "P12": "1 of 2 ROIs extracted", "P13": "Not run"}


def test_routes_reject_unregistered_and_readonly_writes(project):
    current = registry.session_view("steps-test", mark_seen=False)
    registry.publish("steps-test", current["spec"], project={"root": str(project), "read_only": True})
    with TestClient(api.app) as client:
        assert client.get("/studio/steps-test/step/tab7").status_code == 200
        assert client.get("/studio/missing/step/tab7").status_code == 400
        assert client.get("/studio/steps-test/step/not-a-step").status_code == 400
        assert client.post("/studio/steps-test/settings", json={"changes": {}, "revision": ""}).status_code == 400
        assert client.post("/studio/steps-test/jobs/" + "a" * 32, json={}).status_code == 400


def test_failed_run_marks_only_its_recorded_input(project):
    import csv
    from spacemap_gui.run_log import TAB_LABELS

    folder = project / "outputs/rois"
    folder.mkdir(parents=True)
    (folder / "roi1.ome.tiff").touch()
    (folder / "roi2.ome.tiff").touch()
    (project / "logs").mkdir()
    with (project / "logs/run_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tab", "status", "action", "error", "settings_json"])
        writer.writeheader()
        writer.writerow(
            {
                "tab": TAB_LABELS["tab5"],
                "status": "failed",
                "action": "segment",
                "error": "Missing nuclear channel",
                "settings_json": json.dumps({"input_file": str(folder / "roi1.ome.tiff")}),
            }
        )
    result = work_list("steps-test", "tab5")
    states = {row["name"]: row["state"] for row in result["items"]}
    assert states == {"roi1": "failed", "roi2": "not_run"}
    assert result["last_run"]["message"] == "Missing nuclear channel"


def test_phenotyping_run_in_the_classic_tab_counts_as_done(project, monkeypatch):
    # Without reading the table, PDAC's ROIs all showed "Not run".
    import pandas as pd

    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    (project / "outputs/rois").mkdir(parents=True)
    for name in ("s1r1", "s2r1", "s3r1"):
        (project / f"outputs/rois/{name}.ome.tiff").touch()
    table = project / "cells.csv"
    pd.DataFrame(
        {
            "cell_id": ["c0", "c1", "c2", "c3"],
            "batch_id": ["s1r1", "s1r1", "s2r1", "s3r1"],
            "label": [1, 2, 1, 1],
            "phenotype_leiden": ["A", "B", None, None],
            "phenotype_som": ["x", None, "y", None],
            "phenotype_ref1": [None, None, None, "T"],
        }
    ).to_csv(table, index=False)
    registry.publish(
        "steps-test",
        {"id": "empty", "layers": []},
        project={"root": str(project), "mappings": {}, "unified": str(table)},
    )
    rows = {row["name"]: row for row in work_list("steps-test", "tab7")["items"]}
    assert rows["s1r1"]["state"] == "done" and rows["s1r1"]["methods"] == ["phenotype_leiden", "phenotype_som"]
    assert rows["s2r1"]["state_label"] == "Phenotyped (1 method)"
    assert rows["s3r1"]["state"] == "not_run"  # only reference labels: not a method's result
    # A result column a Studio job saved beside the table (not in the file) counts too, and does not break
    # the check: it used to raise, so nothing was cached and every step list re-read the whole table.
    from spacemap import pending_columns

    pending_columns.save(
        table,
        {"phenotype_ml_mr_studio": ["T", "", None, "B"]},
        anchor_column="cell_id",
        anchor_values=["c0", "c1", "c2", "c3"],
    )
    rows = {row["name"]: row for row in work_list("steps-test", "tab7")["items"]}
    assert rows["s3r1"]["methods"] == ["phenotype_ml_mr_studio"]
    assert rows["s1r1"]["methods"] == ["phenotype_leiden", "phenotype_som", "phenotype_ml_mr_studio"]


def test_project_status_lists_samples_folders_and_the_file(project, monkeypatch):
    from spacemap import ai_settings
    from spacemap_gui.studio import assistant
    from spacemap_gui.studio.steps import project_status

    (project / "outputs/rois").mkdir(parents=True)
    (project / "outputs/rois/P11_roi_2.ome.tiff").touch()
    (project / "study.json").write_text("{}")
    found = project_status("steps-test")
    assert found["project"]["name"] == "study" and found["project"]["file"] == "study.json"
    assert [step["id"] for step in found["steps"]][:2] == ["tab1", "tab2"]
    names = {item["name"]: item["steps"] for item in found["samples"]}
    assert names["P11_roi_2"]["tab4"]["state"] == "not_run" and "tab1" not in names["P11_roi_2"]
    folders = {(item["step"], item["label"]): item for item in found["folders"]}
    assert folders[("tab3", "ROI images")] | {"files": 1} == folders[("tab3", "ROI images")]
    assert folders[("tab3", "ROI images")]["path"] == "outputs/rois" and folders[("tab3", "ROI images")]["exists"]
    assert not folders[("tab1", "Registered images")]["exists"]
    monkeypatch.setattr(ai_settings, "is_enabled", lambda: False)
    assert assistant.status()["enabled"] is False


def test_step_folders_follow_tab0s_defaults(tmp_path):
    from spacemap_gui.tabs.tab0_project import get_default_directory_mappings

    defaults = get_default_directory_mappings(str(tmp_path), str(tmp_path / "codex"), str(tmp_path / "he"))
    chosen_by_user = {"batch_codex_dir", "batch_he_dir"}  # picked when the project is made
    checked = 0
    for inputs, outputs in LAYOUT.values():
        for _, keys, fallback in inputs + outputs:
            if keys and keys[0] in defaults and keys[0] not in chosen_by_user:
                assert str(tmp_path / fallback) == defaults[keys[0]], keys[0]
                checked += 1
    assert checked >= 15


def test_a_row_opens_the_file_it_lists_after_a_folder_changes_in_studio(project):
    """The registration folder changed in Studio (project file), not in the session: both follow the project file."""
    for folder in ("outputs/registered", "output/registered_new"):
        (project / folder).mkdir(parents=True)
        (project / folder / "P11_merged.ome.tiff").write_bytes(b"x")
    (project / "study.json").write_text(json.dumps({"project_name": "study", "tab_configurations": {},
                                                    "directory_mappings": {"tab1_batch_output_dir": "${PROJECT_ROOT}/output/registered_new"}}))
    row = next(item for item in work_list("steps-test", "tab3")["items"] if item["name"] == "P11")
    opened = registry.session_images("steps-test")[row["image"]["id"]]["path"]
    assert opened.endswith("output/registered_new/P11_merged.ome.tiff")
