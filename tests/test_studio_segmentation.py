"""Studio's Segmentation step uses Tab 5's settings and functions."""

import json
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from spacemap_gui.studio import segmentation
from spacemap_gui.viewer import api
from test_studio_commands import run, setup  # noqa: F401  (fixture)


def test_settings_follow_tab5_and_are_checked(tmp_path):
    config = {"tab_configurations": {"tab5_segmentation": {"tab4_model_type": "nuclei", "tab4_flow": 0.4}}}
    values = segmentation.values(tmp_path, config)
    assert values["tab4_model_type"] == "nuclei" and values["tab4_flow"] == 0.4 and values["tab4_diameter"] == 0
    assert values["tab4_custom_model"].endswith("data/seg_model/CP_20241014_094617")
    for wrong in ({"tab4_model_type": "unet"}, {"tab4_flow": 9}, {"tab4_diameter": -1}, {"tab4_tile_size": 10}):
        with pytest.raises(ValueError):
            segmentation.check(wrong)
    cfg = segmentation.config_for({**values, "tab4_model_type": "cyto", "tab4_diameter": 30})
    assert cfg.cellpose_model_type == "cyto" and cfg.cellpose_model_path == "" and cfg.diameter == 30


def test_nuclear_marker_stands_in_when_no_border_marker_is_chosen():
    names = ["DAPI", "CD44", "CD31"]
    blank = dict(segmentation.DEFAULTS)
    assert segmentation.channels(names, blank) == ("DAPI", ["DAPI"])
    chosen = {**blank, "tab4_last_membrane_names": ["CD44", "Missing"]}
    assert segmentation.channels(names, chosen) == ("DAPI", ["CD44"])


def test_outlines_are_polygons_in_image_pixels():
    masks = np.zeros((40, 40), np.int32)
    masks[5:15, 5:15] = 1
    masks[20:30, 22:35] = 2
    polygons = segmentation.outlines(masks, 100, 200)
    assert len(polygons) == 2
    xs = [p[0] for p in polygons[1]]
    ys = [p[1] for p in polygons[1]]
    assert min(xs) == 122 and max(xs) == 134 and min(ys) == 220 and max(ys) == 229


def test_settings_route_saves_in_the_project(setup):  # noqa: F811
    root = setup[2]
    (root / f"{root.name}.json").write_text(json.dumps({"tab_configurations": {}}))
    with TestClient(api.app) as client:
        route = f"/studio/{setup[0]}/segmentation_settings"
        current = client.get(route).json()
        saved = client.post(
            route,
            json={
                "changes": {"tab4_model_type": "nuclei", "tab4_last_membrane_names": ["M1"]},
                "revision": current["revision"],
            },
        ).json()
        assert saved["values"]["tab4_model_type"] == "nuclei"
        stored = json.loads((root / f"{root.name}.json").read_text())["tab_configurations"]["tab5_segmentation"]
        assert stored["tab4_model_type"] == "nuclei" and stored["tab4_last_membrane_names"] == ["M1"]
        stale = client.post(route, json={"changes": {"tab4_flow": 0.5}, "revision": current["revision"]})
        assert stale.status_code == 400 and "another window" in stale.json()["error"]


def test_preview_route_clips_and_centres_the_area(setup, monkeypatch):  # noqa: F811
    seen = {}

    def fake(image, names, bounds, settings):
        seen.update(bounds=bounds, names=names, model=settings["tab4_model_type"])
        return {"count": 0, "outlines": []}

    monkeypatch.setattr(segmentation, "preview", fake)
    with TestClient(api.app) as client:
        route = f"/studio/{setup[0]}/segment_preview"
        assert (
            client.post(route, json={"bounds": [-10, -10, 50, 40], "settings": {"tab4_model_type": "cp"}}).status_code
            == 200
        )
        assert seen["bounds"] == [0, 0, 40, 30] and seen["names"] == ["M0", "M1"] and seen["model"] == "cp"
        assert client.post(route, json={"bounds": [0, 0, 10]}).status_code == 400
        assert client.post(route, json={"bounds": [0, 0, 10, 10], "settings": {"tab4_tile_size": 1}}).status_code == 400


def test_segment_images_starts_a_job_with_saved_settings(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs

    started = {}
    monkeypatch.setattr(
        jobs,
        "start",
        lambda root, kind, settings, request_id=None: started.update(kind=kind, settings=settings) or {"id": "x"},
    )
    result = run(setup, "segment_images", {}, confirmed=True)
    assert result["status"] == "done", result
    assert started["kind"] == "segment_images"
    assert started["settings"]["images"][0]["image"].endswith("P11_roi_2.ome.tiff")
    assert started["settings"]["images"][0]["outputs"].endswith("outputs/segmentation")
    assert started["settings"]["values"]["tab4_model_type"] == "cyto"


def test_a_saved_custom_model_path_is_read_back(tmp_path):
    import json

    from spacemap_gui.studio.settings import read_config, revision

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"tab_configurations": {}}))
    model = str(root / "outputs/segmentation/cellpose_models/models/studio_v1")
    before = segmentation.values(root, read_config(root))
    after = segmentation.save(root, {"tab4_model_type": "custom", "tab4_custom_model": model}, revision(before))
    stored = json.loads((root / "study.json").read_text())
    # Where classic Tab 5 keeps it, relative to the project so the project can move.
    assert (
        stored["directory_mappings"]["tab4_custom_model"]
        == "${PROJECT_ROOT}/outputs/segmentation/cellpose_models/models/studio_v1"
    )
    assert after["tab4_model_type"] == "custom" and after["tab4_custom_model"] == model
    (root / "study.json").write_text(
        json.dumps(
            {"directory_mappings": {"tab4_custom_model": "${PROJECT_ROOT}/models/mine"}, "tab_configurations": {}}
        )
    )
    assert segmentation.values(root, read_config(root))["tab4_custom_model"] == str(root / "models/mine")


def _square(name, x, size=20, kind="tissue"):
    return {"type": "Feature", "properties": {"id": name, "name": name, "classification": {"name": kind}},
            "geometry": {"type": "Polygon", "coordinates": [[[x, 0], [x + size, 0], [x + size, size], [x, size], [x, 0]]]}}


def test_segment_inside_a_saved_annotation_or_the_picked_polygons(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs
    from spacemap_gui.studio.settings import read_config, revision
    from spacemap_gui.viewer import registry

    root = setup[2]
    started = {}
    monkeypatch.setattr(jobs, "start", lambda root, kind, settings, request_id=None: started.update(settings=settings) or {"id": "x"})
    tissue = root / "outputs/processed_annotations/stage0_tissue_annotation/P11_roi_2_valid_tissue.geojson"
    tissue.parent.mkdir(parents=True)
    tissue.write_text(json.dumps({"type": "FeatureCollection", "features": [_square("tissue_1", 0), _square("tissue_2", 30)]}))
    segmentation.save(root, {"tab4_mask_layer": "tissue"}, revision(segmentation.values(root, read_config(root))))

    assert run(setup, "segment_images", {}, confirmed=True)["status"] == "done"
    area = started["settings"]["images"][0]["area"]
    assert area == {"name": "tissue", "pending": str(tissue.resolve())}  # the image's own tissue; cut out in the job
    area = {"name": "tissue", "path": segmentation.area_path(root, read_config(root), {}, "P11_roi_2.ome.tiff", area["pending"])}
    assert area["path"] == str(tissue.resolve())

    config = segmentation.config_for(started["settings"]["values"])
    notes = segmentation.region(config, "P11_roi_2.ome.tiff", started["settings"]["values"], area)
    assert config.use_input_geojson and config.input_geojson_path == area["path"] and config.mask_class_names == []
    assert "artifact" in config.mask_exclude_class_names and "inside tissue" in notes[-1]  # artifacts still cut out
    with pytest.raises(ValueError, match="no saved tissue"):
        segmentation.region(segmentation.config_for(started["settings"]["values"]), "P11_roi_2.ome.tiff", started["settings"]["values"])

    tissue.unlink()
    missing = run(setup, "segment_images", {}, confirmed=True)
    assert missing["status"] != "done" and "No saved annotation to segment in for P11_roi_2.ome.tiff (tissue)" in json.dumps(missing)

    # Picked polygons: only those, on this image; clots are kept when they are what was picked.
    drawn = root / "outputs/viewer/shapes/clots.geojson"
    drawn.parent.mkdir(parents=True)
    drawn.write_text(json.dumps({"type": "FeatureCollection", "features": [_square("a", 0, kind="Blood_Clot"), _square("b", 30, kind="Blood_Clot")]}))
    layer = registry.register_object("shapes", path=str(drawn), image=str(root / "outputs/rois/P11_roi_2.ome.tiff"), editable=True)
    assert run(setup, "segment_images", {"shape": layer, "ids": ["a"]}, confirmed=True)["status"] == "done"
    area = started["settings"]["images"][0]["area"]
    assert area["exact"] and len(json.loads(open(area["path"]).read())["features"]) == 1
    config = segmentation.config_for(started["settings"]["values"])
    segmentation.region(config, "P11_roi_2.ome.tiff", started["settings"]["values"], area)
    assert config.mask_exclude_class_names == []
    assert run(setup, "segment_images", {"scope": "all", "shape": layer, "ids": ["a"]}, confirmed=True)["status"] != "done"


def test_the_step_lists_images_models_and_runs_batches(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs
    from spacemap_gui.studio.settings import read_config

    root = setup[2]
    started = {}
    monkeypatch.setattr(jobs, "start", lambda root, kind, settings, request_id=None: started.update(settings=settings) or {"id": "x"})
    models = root / "data/seg_model"
    models.mkdir(parents=True)
    (models / "CP_lab_model").write_bytes(b"0" * 1_100_000)
    (models / "notes.txt").write_text("not a model")
    found = segmentation.plan(root, read_config(root), {})
    assert [row["image"] for row in found["rows"]] == ["P11_roi_2.ome.tiff"] and not found["rows"][0]["segmented"]
    groups = {item["group"] for item in found["models"]}
    assert groups == {"Cellpose", "Models folder"} and found["model"] == "cyto"
    segmentation.save_plan(root, {"model": "folder:CP_lab_model"})
    assert segmentation.values(root, read_config(root))["tab4_custom_model"].endswith("data/seg_model/CP_lab_model")
    assert segmentation.plan(root, read_config(root), {})["model"] == "folder:CP_lab_model"
    with pytest.raises(ValueError, match="not in the list"):
        segmentation.save_plan(root, {"model": "folder:notes.txt"})

    # A ticked batch; a segmented image is kept unless Segment again, which moves its results to previous/ first.
    output = root / "outputs/segmentation"
    output.mkdir(parents=True)
    (output / "P11_roi_2.ome_cell_features.csv").write_text("label,x\n1,2\n2,3\n")
    assert segmentation.plan(root, read_config(root), {})["rows"][0]["cells"] == 2
    kept = run(setup, "segment_images", {"scope": "images", "images": ["P11_roi_2.ome.tiff"]}, confirmed=True)
    assert "segmented already" in kept["message"] and "settings" not in started
    segmentation.save_plan(root, {"areas": {"P11_roi_2.ome.tiff": segmentation.WHOLE}})
    again = run(setup, "segment_images", {"scope": "images", "images": ["P11_roi_2.ome.tiff"], "again": True, "tag": False}, confirmed=True)
    assert again["status"] == "done", again
    assert started["settings"]["images"][0]["area"] == {"whole": True} and started["settings"]["values"]["tab4_spatial_enabled"] is False
    assert not (output / "P11_roi_2.ome_cell_features.csv").exists() and list((output / "previous").rglob("*_cell_features.csv"))


def test_training_lists_and_starts_from_saved_and_folder_models(setup):
    import os
    from spacemap_gui.studio import training

    channel, _, root, _ = setup
    models = root / 'data/seg_model'
    models.mkdir(parents=True)
    old = models / 'older_v9'
    newest = models / 'latest_v26'
    for path, modified in ((old, 10), (newest, 20)):
        path.write_bytes(b'0' * 1_100_000)
        os.utime(path, (modified, modified))
    trained = root / 'outputs/segmentation/cellpose_models/models/local_v10'
    trained.parent.mkdir(parents=True)
    trained.write_bytes(b'weights')
    (root / f'{root.name}.json').write_text(json.dumps({'tab_configurations': {}}))
    segmentation.save_plan(root, {'model': 'folder:latest_v26'})
    with TestClient(api.app) as client:
        route = f'/studio/{channel}/training'
        found = client.get(route).json()
        assert found['selected_base'] == str(newest)
        options = found['starting_models']
        assert [i['base'] for i in options if i['group'] == 'Models folder'] == [str(newest), str(old)]
        assert any(i['base'] == str(trained) for i in options)
        started = client.post(route + '/start', json={'name': 'improved', 'base': str(newest)})
        assert started.status_code == 200 and started.json()['current'] == str(newest)
        assert training.session(root)['base'] == str(newest)
        newest.unlink()
        unavailable = client.get(route).json()
        assert unavailable['selected_base'] == str(newest)
        assert any(i['base'] == str(newest) and not i['available'] for i in unavailable['starting_models'])
        missing = client.post(route + '/start', json={'name': 'wrong', 'base': str(newest)})
        assert missing.status_code == 400 and 'missing' in missing.json()['error']
        assert training.session(root)['name'] == 'improved'


def test_saved_model_weights_are_used_without_builtin_fallback(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from spacemap_gui.studio import training

    seen = []
    monkeypatch.setitem(sys.modules, 'cellpose', SimpleNamespace(models=SimpleNamespace(
        CellposeModel=lambda **kwargs: seen.append(kwargs) or kwargs)))
    path = tmp_path / 'trained_v26'
    path.write_bytes(b'weights')
    training._model(str(path), False)
    assert seen[-1] == {'gpu': False, 'pretrained_model': str(path), 'mkldnn': False}
    training._model('cyto', False)
    assert seen[-1]['model_type'] == 'cyto' and 'pretrained_model' not in seen[-1]
    path.unlink()
    with pytest.raises(ValueError, match='missing'):
        training._model(str(path), False)
    assert len(seen) == 2


def test_the_job_prepares_each_area_and_skips_one_it_cannot(tmp_path, monkeypatch):
    """Selected hands the job each image's chosen area; the job cuts it out, and an image whose area is gone is left
    out (said) while the others are segmented."""
    from spacemap_gui.studio import tasks

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"project_name": "study", "directory_mappings": {}, "tab_configurations": {}}))
    images = []
    for name in ("a.ome.tiff", "b.ome.tiff"):
        image = root / name
        image.write_bytes(b"x")
        images.append(image)
    area = root / "area.geojson"
    area.write_text("{}")
    seen = {}
    monkeypatch.setattr(segmentation, "area_path", lambda r, c, m, name, pending: pending if Path(pending).is_file() else (_ for _ in ()).throw(ValueError("gone")))
    monkeypatch.setattr(segmentation, "run", lambda items, values, report: seen.setdefault("items", items) and [{"image": i["image"], "cells": 1} for i in items])
    result = tasks.segment_images({"root": str(root), "values": {}, "images": [
        {"image": str(images[0]), "area": {"name": "tissue", "pending": str(area)}},
        {"image": str(images[1]), "area": {"name": "tissue", "pending": str(root / "missing.geojson")}}]}, None, lambda *a: None)
    assert [i["area"] for i in seen["items"]] == [{"name": "tissue", "path": str(area)}]
    assert result["skipped"] == [{"image": "b.ome.tiff", "error": "gone"}] and result["count"] == 1
