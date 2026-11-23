"""The merged rows link to masks by ROI/local label, independently of global cell IDs."""

import pandas as pd

from spacemap import pending_columns
from spacemap_gui.studio import cell_links, phenotyping_inputs
from spacemap_gui.viewer import registry
from test_studio_commands import setup  # noqa: F401


def test_table_stays_available_when_a_cell_layer_cannot_reconnect(setup, monkeypatch):
    channel, table, _, _ = setup
    token = registry.session_view(channel)['spec']['layers'][1]['token']
    found = cell_links.context(channel, ['expired-cell-token', token])
    assert found['table']['path'] == str(table)
    assert 'phenotype_ml_consensus' in found['table']['columns']
    assert not found['layers']['expired-cell-token']['linked']
    assert found['layers'][token]['linked']

    def broken(*args):
        raise OSError('Cell geometry is unavailable')
    monkeypatch.setattr(cell_links, 'bind', broken)
    found = cell_links.context(channel, [token])
    assert found['table']['path'] == str(table)
    assert found['layers'][token]['reason'] == 'Cell geometry is unavailable'


def test_link_revision_tracks_saved_values_without_reading_cell_rows(setup, monkeypatch):
    from spacemap import labels
    from spacemap_gui.studio import data_view

    path, frame = combined(setup)
    channel = setup[0]
    def revision():
        return cell_links.context(channel)['table']['revision']
    before = revision()
    assert revision() == before
    # A schema-only check must also notice overwriting an existing column.
    monkeypatch.setattr(data_view, '_table', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Full table read')))
    labels.append(path, [{'row': 0, 'column': 'phenotype_ref1', 'value': 'T'}])
    labelled = revision()
    assert labelled != before and revision() == labelled
    pending_columns.save(path, {'phenotype_clusters': ['A'] * len(frame)},
                         anchor_column='global_id', anchor_values=frame.global_id)
    predicted = revision()
    assert predicted != labelled
    pending_columns.save(path, {'phenotype_clusters': ['B'] * len(frame)},
                         anchor_column='global_id', anchor_values=frame.global_id)
    overwritten = revision()
    assert overwritten != predicted and revision() == overwritten
    frame.loc[0, 'M0'] = 123
    frame.to_parquet(path, index=False)
    assert revision() != overwritten


def test_conflicting_data_draft_does_not_break_saved_image_links(setup):
    from spacemap_gui.studio import data_view
    from spacemap_gui.viewer.api import app
    from starlette.testclient import TestClient
    path, frame = combined(setup)
    channel, _, root, _ = setup
    token = registry.session_view(channel)['spec']['layers'][1]['token']
    data_view.edit(path, [{'row':0,'col':'M0','value':99}])
    pending_columns.save(path, {'M0':[77]*len(frame), 'phenotype_new':['T']*len(frame)},
                         anchor_column='global_id',anchor_values=frame.global_id)
    found = cell_links.context(channel,[token])
    assert found['layers'][token]['linked']
    assert 'phenotype_new' in found['table']['columns']
    response = TestClient(app).post(f'/studio/{channel}/data/query',json={'source':{'combined':True}})
    assert response.status_code == 409
    assert response.json()['code'] == 'data_external_change'


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


def test_link_recovers_when_the_fast_table_finishes_without_source_changes(setup, monkeypatch):
    from spacemap_gui.studio import data_view

    path, _ = combined(setup)
    channel, _, root, _ = setup
    token = registry.session_view(channel)['spec']['layers'][1]['token']
    read = data_view._table
    ready = False

    def preparing(table, *args, **kwargs):
        if str(table) == str(path) and not ready:
            raise data_view.Preparing('Making a fast copy of the cell table')
        return read(table, *args, **kwargs)

    monkeypatch.setattr(data_view, '_table', preparing)
    first = cell_links.context(channel, [token])['layers'][token]
    assert not first['linked'] and first['retryable']
    ready = True  # the CSV is unchanged; only its fast copy became available
    second = cell_links.context(channel, [token])['layers'][token]
    assert second['linked'] and 'same label and position' in second['reason']
