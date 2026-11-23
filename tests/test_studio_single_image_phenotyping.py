"""Segmentation/tagging tables work without Batch correction or an image column."""

import uuid

import pandas as pd
import pytest

from spacemap import dataio
from spacemap.viewer import cells
from spacemap_gui.studio import cell_links, jobs, leiden, manual_reference, phenotyping_inputs as inputs
from spacemap_gui.studio.settings import revision
from spacemap_gui.viewer import registry
from test_studio_commands import run, setup  # noqa: F401


@pytest.fixture
def single(setup):  # noqa: F811
    channel, local, root, spec = setup
    frame = pd.read_csv(local).iloc[:4].drop(columns="batch_id")
    frame["tissue_roi"] = [0, 1, 1, 2]
    path = root / "P11_roi_2.ome_cell_features_tagged.parquet"
    frame.to_parquet(path, index=False)
    inputs.use_table(channel, path)
    return channel, path, root, spec


def test_counts_and_results_use_image_filename_not_tissue_pieces(single):
    _, path, root, _ = single
    (root / "outputs/rois/second.ome.tiff").touch()
    plan = inputs.plan(root, {}, str(path))
    assert plan["roles"] == {"roi": "", "slide": "", "cell_id": "label"}
    assert plan["cells"] == 4 and plan["single_image"] == "P11_roi_2"
    assert [row["roi"] for row in plan["rows"]] == ["P11_roi_2", "second"]
    assert plan["rows"][0]["cells"] == 4 and plan["rows"][1]["cells"] is None
    assert plan["rows"][0]["results"]["phenotype_ml_consensus"] == 4
    assert not plan["warnings"]


def test_tissue_mapping_can_be_cleared_and_unknown_image_is_explicit(single):
    _, path, root, _ = single
    inputs.save_roles(root, path, {"roi": "tissue_roi", "slide": "", "cell_id": "label", "label": ""})
    wrong = inputs.plan(root, {}, path)
    assert len(wrong["rows"]) == 1 and wrong["rows"][0]["cells"] is None
    assert "No image column" in wrong["warnings"][0]
    inputs.save_roles(root, path, {"roi": "", "slide": "", "cell_id": "label"})
    assert inputs.plan(root, {}, path)["rows"][0]["cells"] == 4
    generic = root / "measurements.parquet"
    pd.read_parquet(path).to_parquet(generic, index=False)
    assert inputs.single_image(root, generic) == ""
    inputs.save_roles(root, generic, {"roi": "", "slide": "", "cell_id": "label", "image": "P11_roi_2"})
    assert inputs.single_image(root, generic) == "P11_roi_2"


def test_link_checks_image_and_coordinates_without_roi_column(single):
    _, path, root, spec = single
    token = spec["layers"][1]["token"]
    found = cell_links.bind(root, str(path), token)
    assert found["linked"] and found["keys"]["roi_column"] == ""
    ids, values = cells.roi_values(str(path), found["roi"], "M0", roi_column="")
    assert ids.tolist() == [1, 2, 3, 4] and values.tolist() == [1, 2, 3, 4]
    other = registry.register_object("cells", roi="second", tables={})
    assert not cell_links.bind(root, str(path), other)["linked"]
    frame = pd.read_parquet(path)
    frame.loc[3, "x_centroid"] += 100
    frame.to_parquet(path, index=False)
    assert not cell_links.bind(root, str(path), token)["linked"]


def test_manual_labels_and_undo_use_local_cell_ids(single):
    channel, path, _, _ = single
    manual_reference.create(channel, "phenotype_ref1")
    info = manual_reference.options(channel)
    base = {"table": str(path), "revision": info["revision"], "column": "phenotype_ref1", "phenotype": "B"}
    request = uuid.uuid4().hex
    manual_reference.assign(channel, {**base, "request_id": request, "selection": {"rows": [0, 2]}})
    assert dataio.load(path, columns=["phenotype_ref1"]).phenotype_ref1.fillna("").tolist() == ["B", "", "B", ""]
    manual_reference.undo(channel, request)
    assert dataio.load(path, columns=["phenotype_ref1"]).phenotype_ref1.isna().all()
    with pytest.raises(ValueError, match="another image"):
        manual_reference.assign(channel, {**base, "request_id": uuid.uuid4().hex, "selection": {"roi": "second", "cells": [1]}})


def test_leiden_can_start_from_raw_table(single, monkeypatch):
    _, path, root, _ = single
    leiden.save(root, {"markers": ["M0", "M1"]}, revision(leiden.values(root)))
    started = []
    monkeypatch.setattr(jobs, "start", lambda root, kind, settings, **kwargs: started.append(settings) or {"id": "test"})
    result = run(single, "run_leiden", confirmed=True)
    assert result["status"] == "done", result
    assert started[0]["table"] == str(path) and started[0]["cell_id_col"] == "label"
    assert started[0]["roi_col"] == ""


def test_data_api_reports_single_image_identity(single):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    channel, _, _, _ = single
    client = TestClient(app)
    response = client.post(f"/studio/{channel}/data/query", json={"source": {"combined": True}})
    assert response.status_code == 200, response.text
    assert response.json()["identity"]["0"]["roi"] == "P11_roi_2"
    response = client.post(f"/studio/{channel}/data/groups", json={"source": {"combined": True}, "labels": [1], "column": "phenotype_ml_consensus"})
    assert response.status_code == 200, response.text
    assert response.json()["groups"] == ["CD8"]


def test_subphenotype_filters_use_active_image_table_without_correction(single):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    channel, path, _, _ = single
    client = TestClient(app)
    settings = client.get(f'/studio/{channel}/subpheno_settings')
    assert settings.status_code == 200, settings.text
    assert 'tissue_roi' in settings.json()['columns']
    response = client.get(f'/studio/{channel}/phenotype_values', params={'column': 'tissue_roi'})
    assert response.status_code == 200, response.text
    assert {r['name']: r['cells'] for r in response.json()['values']} == {'0': 1, '1': 2, '2': 1}
    labels = dataio.load(path).phenotype_ml_consensus.dropna().unique().tolist()
    response = client.post(f'/studio/{channel}/subpheno_selection', json={
        'parent_col': 'phenotype_ml_consensus', 'parent_values': labels,
        'filters': [{'column': 'tissue_roi', 'op': 'in', 'values': ['1']}]})
    assert response.status_code == 200, response.text
    assert response.json()['cells'] == 2
    bad = client.post(f'/studio/{channel}/subpheno_selection', json={
        'parent_col': 'phenotype_ml_consensus', 'parent_values': labels,
        'filters': [{'column': 'tissue_roi', 'op': 'anything', 'values': ['1']}]})
    assert bad.status_code == 400


def test_pairing_and_counts_cover_all_images_and_can_be_corrected(single):
    channel, first, root, _ = single
    (root / 'outputs/rois/second.ome.tiff').touch()
    second = root / 'second_cell_features_tagged.csv'
    frame = pd.read_parquet(first).iloc[:2]
    frame.to_csv(second, index=False)
    assert inputs.image_tables(root, first)['second'] == str(second)
    assert inputs.paired_summary(root, first, 'second')['cells'] == 2
    alternate = root / 'manually_named.csv'
    frame.iloc[:1].to_csv(alternate, index=False)
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    client = TestClient(app)
    route = f'/studio/{channel}/phenotyping_inputs'
    response = client.post(route, json={'pair': {'image': 'second', 'table': str(alternate)}})
    assert response.status_code == 200, response.text
    assert inputs.single_image(root, alternate) == 'second'
    assert inputs.image_tables(root, first)['second'] == str(alternate)
    assert client.get(route, params={'image': 'second'}).json()['cells'] == 1
    assert inputs.column_roles(root, alternate)['roi'] == ''
    # Explicitly unpairing stays unpaired, rather than silently restoring the filename match.
    assert client.post(route, json={'pair': {'image': 'second', 'table': ''}}).status_code == 200
    assert inputs.image_tables(root, first)['second'] == ''
    assert client.get(route, params={'image': 'second'}).status_code == 400


def test_ambiguous_pairing_is_not_guessed(single):
    _, first, root, _ = single
    pd.read_parquet(first).to_csv(root / 'P11_roi_2_cell_features.csv', index=False)
    assert inputs.image_tables(root, first)['P11_roi_2'] == ''
    inputs.save_pair(root, first, 'P11_roi_2', str(first))
    assert inputs.image_tables(root, first)['P11_roi_2'] == str(first)


def test_annotation_bed_is_not_offered_as_phenotype_training_labels(single):
    channel, path, _, _ = single
    frame = pd.read_parquet(path)
    frame['bed'] = ['tumor', 'stroma', 'tumor', 'stroma']
    frame['phenotype_named'] = ['B', 'T', 'B', 'T']
    frame.to_parquet(path, index=False)
    found = manual_reference.options(channel)
    assert 'bed' in found['columns']  # it remains available as table data
    assert 'bed' not in found['training_columns']
    assert 'bed' in found['other_training_columns']  # still usable by explicit choice
    assert 'phenotype_named' in found['training_columns']
