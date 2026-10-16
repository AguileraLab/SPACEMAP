import uuid

import pandas as pd
import pytest

from spacemap import dataio, pending_columns
from spacemap.viewer import cells, reference
from spacemap_gui.studio import manual_reference as manual, cluster_names
from spacemap_gui.viewer import registry
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def study(setup):  # noqa: F811
    channel, table, root, spec = setup
    frame = pd.read_csv(table)
    frame["cell_id"] = [101, 102, 103, 104, 201]
    frame["prediction"] = ["T", None, "B", "B", "T"]
    frame["M0_corr"] = frame["M0"] / 5
    path = table.with_suffix(".parquet")
    frame.to_parquet(path, index=False)
    project = registry.session_project(channel)
    registry.publish(channel, spec, project={**project, "unified":str(path)})
    return channel, path, spec


def assign(study, selection, value="T", column="my labels", request_id=None):
    info = manual.options(study[0], column)
    return manual.assign(study[0], {"table":info["table"], "revision":info["revision"], "selection":selection,
        "column":column,"phenotype":value,"request_id":request_id or uuid.uuid4().hex})


def test_custom_column_selection_identity_undo_and_native_parquet_commit(study):
    channel, path, _ = study
    original = path.read_bytes()
    created = manual.create(channel, "my labels")
    assert created["column"] == "my labels" and "my labels" in cluster_names.reference_columns(path)
    assert dataio.load(path, columns=["my labels"])["my labels"].isna().all()
    saved = assign(study, {"rows":[0,4]})  # local label 1, two different ROIs
    assert saved["assigned"] == 2
    assert cells.values_at_rows(path, range(5), "my labels").tolist() == ["T", None, None, None, "T"]
    assert path.read_bytes() == original
    manual.undo(channel, saved["request_id"])
    assert dataio.load(path, columns=["my labels"])["my labels"].isna().all()
    assign(study, {"cells":[2], "roi":"P11_roi_2"}, "CD4")
    pending_columns.commit(path)
    frame = pd.read_parquet(path)
    assert frame["my labels"].fillna("").tolist() == ["", "CD4", "", "", ""]
    assert "my labels" in reference.columns(path)
    assert reference.phenotype_names(path, "my labels") == ["CD4"]
    assert reference.options([path], path)["reference_columns"] == ["my labels", "phenotype_ref1"]


def test_filtered_assignment_and_clear_use_all_matches_with_guarded_undo(study):
    saved = assign(study, {"filters":[{"col":"M0","lo":2,"hi":4}]})
    assert saved["assigned"] == 3
    newer = assign(study, {"rows":[2]}, "B")
    with pytest.raises(ValueError, match="changed again"):
        manual.undo(study[0], saved["request_id"])
    manual.undo(study[0], newer["request_id"])
    cleared = assign(study, {"rows":[1,2,3]}, "")
    assert cleared["assigned"] == 3
    assert manual.options(study[0], "my labels")["counts"]["groups"] == []


def test_request_retry_is_idempotent_and_other_columns_are_protected(study):
    saved = assign(study, {"rows":[0]}, request_id="manual-retry-1")
    assert assign(study, {"rows":[0]}, request_id="manual-retry-1") == saved
    for column in ("M0", "cell_id", "batch_id", "prediction", "a\nb", "../reference"):
        with pytest.raises(ValueError):
            manual.create(study[0], column)
    with pytest.raises(ValueError, match="valid rows"):
        assign(study, {"rows":[-1]})
    with pytest.raises(ValueError, match="unique matching row"):
        assign(study, {"cells":[999], "roi":"P11_roi_2"})


def test_stale_selection_cannot_edit_new_table(study):
    info = manual.options(study[0])
    with pytest.raises(ValueError, match="table changed"):
        manual.assign(study[0], {"table":info["table"],"revision":"old","column":"my labels", "phenotype":"T", "selection":{"rows":[0]}})


def test_f1_ignores_unlabelled_reference_but_penalizes_missing_predictions(study):
    assign(study, {"rows":[0,1]}, "T")
    assign(study, {"rows":[4]}, "B")
    found = manual.evaluate(study[1], "my labels", "prediction")
    assert found["cells"] == 3 and found["unlabelled"] == 1
    assert found["macro_f1"] == pytest.approx(.25) and found["accuracy"] == pytest.approx(1/3)
    assert "not an independent test" in found["note"]
    with pytest.raises(ValueError, match="different prediction"):
        manual.evaluate(study[1], "my labels", "my labels")


def test_reference_training_can_use_custom_column_without_open_image(study):
    manual.create(study[0], "my labels")
    registry.publish(study[0], {"id":"no-image", "layers":[]}, project=registry.session_project(study[0]))
    settings = manual.training_settings(study[0], "my labels")
    assert settings["markers"] == ["M0"] and settings["training_column"] == "my labels"
    assert settings["roi_column"] == "batch_id" and settings["revision"]


def test_reference_training_resolves_image_markers_without_channel_names(study):
    manual.create(study[0], "my labels")
    for layer in study[2]["layers"]:
        for channel in layer.get("channels", []):
            channel.pop("name", None)
    registry.publish(study[0], study[2], project=registry.session_project(study[0]))
    settings = manual.training_settings(study[0], "my labels")
    assert settings["markers"] == ["M0", "M1"]


def test_training_accepts_another_phenotype_column_but_not_identifiers(study):
    settings = manual.training_settings(study[0], "prediction")
    assert settings["training_column"] == "prediction"
    assert "prediction" not in reference.columns(study[1])
    for column in ("cell_id", "batch_id", "M0"):
        with pytest.raises(ValueError, match="phenotype column"):
            manual.training_settings(study[0], column)


def test_manual_api_supports_no_image_and_rejects_read_only_writes(study):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    url = f"/studio/{study[0]}/manual_reference"
    with TestClient(app) as client:
        assert client.post(url, json={"action":"create","column":"my labels"}).status_code == 200
        info = client.get(url).json()
        result = client.post(url, json={"action":"assign","table":info["table"],"revision":info["revision"],
            "column":"my labels","phenotype":"T","selection":{"rows":[0,4]},"request_id":"api-label-01"})
        assert result.status_code == 200 and result.json()["assigned"] == 2
        project = registry.session_project(study[0])
        registry.publish(study[0], study[2], project={**project,"read_only":True})
        assert client.post(url, json={"action":"create","column":"blocked"}).status_code == 400


def test_auto_training_settings_survive_requests_and_respect_read_only(study):
    from starlette.testclient import TestClient
    from spacemap_gui.viewer.api import app
    url = f"/studio/{study[0]}/auto_reference"
    with TestClient(app) as client:
        assert not client.get(url).json()["enabled"]
        result = client.post(url, json={"column":"prediction", "enabled":True, "checked_only":False})
        assert result.status_code == 200
        assert client.get(url).json()["column"] == "prediction"
        assert client.get(url).json()["enabled"]
        assert client.post(url, json={"enabled":False}).json()["state"] == "off"
        project = registry.session_project(study[0])
        registry.publish(study[0], study[2], project={**project,"read_only":True})
        assert client.post(url, json={"column":"prediction", "enabled":True}).status_code == 400
