"""The merged rows link to masks by ROI/local label, independently of global cell IDs."""

import pandas as pd

from spacemap import pending_columns
from spacemap_gui.studio import cell_links, phenotyping_inputs
from spacemap_gui.viewer import registry
from test_studio_commands import setup  # noqa: F401


def combined(context):
    channel, local, root, _ = context
    frame = pd.read_csv(local).rename(columns={"batch_id":"roi_name", "label":"object_number"})
    frame["global_id"] = [f"cell_{i}" for i in range(len(frame))]
    frame["physical_slide"] = "S1"
    path = root / "combined.parquet"
    frame.to_parquet(path, index=False)
    phenotyping_inputs.use_table(channel, path)
    phenotyping_inputs.save_roles(root, str(path), {"roi":"roi_name", "slide":"physical_slide", "cell_id":"global_id", "label":"object_number"})
    return path, frame


def test_link_keeps_local_tables_and_uses_custom_identity_columns(setup):  # noqa: F811
    path, frame = combined(setup)
    channel, local, root, _ = setup
    token = registry.session_view(channel)["spec"]["layers"][1]["token"]
    link = cell_links.bind(root, str(path), token)
    assert link["linked"] and link["token"] == token
    registered = registry.resolve_object(token, "cells")
    assert registered["tables"]["t0"] == str(local)
    bound = cell_links.table_layer(registered, link["table"])
    assert bound["roi_column"] == "roi_name" and bound["label_column"] == "object_number"
    assert "roi_column" not in cell_links.table_layer(registered, "t0")
    pending_columns.save(path, {"phenotype_clusters":["A","B","A","B","Other"]},
                         anchor_column="global_id", anchor_values=frame.global_id)
    from spacemap_gui.viewer.api import _cell_lut
    lut, _ = _cell_lut(bound, link["table"], "phenotype_clusters", frozenset({"B"}), "viridis")
    assert lut[1, 3] > 0 and lut[2, 3] == 0


def test_link_rejects_repeated_local_ids_and_changed_segmentation(setup):  # noqa: F811
    path, frame = combined(setup)
    channel, _, root, _ = setup
    token = registry.session_view(channel)["spec"]["layers"][1]["token"]
    original = frame.copy()
    frame.loc[1,"object_number"] = frame.loc[0,"object_number"]
    frame.to_parquet(path, index=False)
    found = cell_links.bind(root, str(path), token)
    assert not found["linked"] and "unique" in found["reason"]
    original.loc[:3,"x_centroid"] += 50
    original.to_parquet(path, index=False)
    found = cell_links.bind(root, str(path), token)
    assert not found["linked"] and "another segmentation" in found["reason"]


def test_data_api_links_hidden_identity_columns_and_selected_groups(setup):  # noqa: F811
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    path, frame = combined(setup)
    channel = setup[0]
    pending_columns.save(path, {"phenotype_clusters":["A","B","A","B","Other"]},
                         anchor_column="global_id", anchor_values=frame.global_id)
    client = TestClient(app)
    def request(action, **extra):
        response = client.post(f"/studio/{channel}/data/{action}", json={"source":{"combined":True},
            "rows":{"this_image":True,"column":"roi_name","value":"P11_roi_2"}, **extra})
        assert response.status_code == 200, response.text
        return response.json()
    page = request("query", columns=["M0"], selection_column="phenotype_clusters")
    assert page["total"] == 4 and page["shown"] == ["M0"]
    assert page["identity"]["0"] == {"label":1,"roi":"P11_roi_2","cell_id":"cell_0","group":"A"}
    assert request("groups", labels=[1,2], column="phenotype_clusters")["groups"] == ["A","B"]
    assert request("query", filters=[{"col":"phenotype_clusters","values":["A"]}])["matched"] == 2
    assert request("find", label=3)["offset"] == 2
    assert request("find", label=1, cell_roi="other", rows={"this_image":False})["offset"] == 4
    assert request("find", label=1, cell_roi="other", rows={"this_image":False}, sort={"col":"M0", "desc":True})["offset"] == 0
    assert request("find", label=1, cell_roi="missing", rows={"this_image":False})["offset"] is None
    whole = request("info", rows={"this_image":False})
    assert whole["image_column"] == whole["roi_column"] == "roi_name"
    assert request("info", rows={"this_image":False, "column":"physical_slide"})["roi_column"] == "roi_name"


def test_image_filename_roi_links_without_changing_local_reference(setup, monkeypatch):  # noqa: F811
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    from spacemap_gui.studio import reference_models

    path, frame = combined(setup)
    channel, _, root, _ = setup
    frame.loc[frame.roi_name == "P11_roi_2", "roi_name"] = "P11_roi_2.ome.tiff"
    frame.to_parquet(path, index=False)
    token = registry.session_view(channel)["spec"]["layers"][1]["token"]
    link = cell_links.bind(root, str(path), token)
    assert link["linked"] and link["keys"]["roi"] == "P11_roi_2.ome.tiff"
    assert registry.resolve_object(token, "cells")["roi"] == "P11_roi_2"
    monkeypatch.setattr(reference_models, "request", lambda settings, *args, **kwargs: settings)
    response = TestClient(app).post(f"/cells/{token}/suggestions", json={
        "t":link["table"], "mode":"saved_reference", "label":1, "model_id":"test"})
    assert response.status_code == 200, response.text
    assert response.json()["roi"] == "P11_roi_2.ome.tiff"
    assert response.json()["label_column"] == "object_number"


def test_ambiguous_roi_names_do_not_link(setup):  # noqa: F811
    path, frame = combined(setup)
    channel, _, root, _ = setup
    frame.loc[:1, "roi_name"] = "P11_roi_2.ome.tiff"
    frame.loc[2:3, "roi_name"] = "P11_roi_2.tiff"
    frame.to_parquet(path, index=False)
    token = registry.session_view(channel)["spec"]["layers"][1]["token"]
    link = cell_links.bind(root, str(path), token)
    assert not link["linked"] and "uniquely" in link["reason"]
