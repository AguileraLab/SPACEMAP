"""Studio and classic share saved keys without overwriting active widget edits."""

from copy import deepcopy
import json

import pytest

from spacemap_gui.studio import settings
from spacemap_gui import persistence


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACEMAP_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(persistence.st, "session_state", {})
    root = tmp_path / "study"
    root.mkdir()
    persistence.save_project_config(
        str(root),
        {
            "project_root": str(root),
            "tab_configurations": {
                "tab7_phenotyping": {"resolve_config": {"output_col": "phenotype_old", "norm_z1": True}}
            },
        },
    )
    return root


def test_studio_saves_classic_keys_and_classic_picks_them_up(project):
    config = settings.read_config(project)
    old = deepcopy(config["tab_configurations"]["tab7_phenotyping"]["resolve_config"])
    session = {"resolve_config": old, "resolve_output_col": old["output_col"]}
    before = settings.get(project)
    saved = settings.save(project, {"output_col": "phenotype_new"}, before["revision"])
    assert saved["values"]["output_col"] == "phenotype_new"
    assert settings.pull_into_classic(project, config, session) == ["resolve_config"]
    assert session["resolve_config"]["output_col"] == session["resolve_output_col"] == "phenotype_new"
    assert settings._resolve_path(project).exists()


def test_local_edits_survive_while_unedited_settings_update(project):
    config = settings.read_config(project)
    old = deepcopy(config["tab_configurations"]["tab7_phenotyping"]["resolve_config"])
    session = {"resolve_config": {**old, "output_col": "phenotype_local"}, "resolve_output_col": "phenotype_typing"}
    settings.save(project, {"output_col": "phenotype_remote", "norm_z1": False}, settings.get(project)["revision"])
    settings.pull_into_classic(project, config, session)
    assert session["resolve_config"]["output_col"] == "phenotype_local"
    assert session["resolve_output_col"] == "phenotype_typing"
    assert session["resolve_config"]["norm_z1"] is False


def test_classic_save_is_visible_in_studio_and_stale_studio_form_is_rejected(project):
    before = settings.get(project)
    config = settings.read_config(project)
    persistence.update_stored_config_key(
        config, "resolve_config", {"output_col": "phenotype_classic", "norm_z1": False}
    )
    persistence.save_project_config(str(project), config)
    assert settings.get(project)["values"]["output_col"] == "phenotype_classic"
    with pytest.raises(ValueError, match="another window"):
        settings.save(project, {"output_col": "phenotype_stale"}, before["revision"])


@pytest.mark.parametrize(
    "changes",
    [
        {"data": [0] * 100000},
        {"output_col": "label"},
        {"tie_score_cofactor": float("nan")},
        {"norm_z1": "yes"},
        {"norm_nlog": True, "norm_cdf": False},
    ],
)
def test_settings_reject_invalid_or_data_sized_input(project, changes):
    before = settings.config_path(project).read_bytes()
    with pytest.raises(ValueError):
        settings.save(project, changes, settings.get(project)["revision"])
    assert settings.config_path(project).read_bytes() == before


def test_changed_settings_rebuild_live_engine_and_keep_raw_gate(tmp_path):
    import pandas as pd
    from spacemap.viewer import live_resolve

    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"criteria_cols": ["Criteria 1"], "table": [{"Phenotype": "A", "Criteria 1": ["M0"]}]}))
    table = tmp_path / "cells.csv"
    pd.DataFrame({"batch_id": ["roi"] * 3, "label": [1, 2, 3], "M0": [10.0, 20.0, 30.0]}).to_csv(table, index=False)
    spec = {"table": str(table), "roi": "roi", "panel": str(panel)}
    first = live_resolve.open_session(spec)
    live_resolve.assign(first, {"M0": 17.375})
    path = tmp_path / "resolve_params.json"
    path.write_text(json.dumps({"tie_score_transform": "asinh_difference", "tie_score_cofactor": 7.0}))
    live_resolve.invalidate_settings(path)
    second = live_resolve.open_session(spec)
    assert second is not first and second.thresholds["M0"] == 17.375
    assert second.engine.settings.tie_score_cofactor == 7.0


def test_classic_save_and_sync_are_reentrant_under_shared_lock(project):
    # Studio's save calls classic persistence while holding this same file lock.
    with settings.project_lock(project):
        persistence.save_project_config(str(project), settings.read_config(project))
    assert settings.read_config(project)["tab_configurations"]


def test_studio_segmentation_settings_reach_an_open_classic_tab5(project):
    from spacemap_gui.studio import segmentation

    config = settings.read_config(project)
    session = {"tab4_model_type": "cyto", "tab4_flow": 0.6}
    config["tab_configurations"]["tab5_segmentation"] = {"tab4_model_type": "cyto", "tab4_flow": 0.8}
    current = segmentation.values(project, settings.read_config(project))
    segmentation.save(project, {"tab4_model_type": "nuclei", "tab4_flow": 0.4}, settings.revision(current))
    changed = settings.pull_into_classic(project, config, session)
    assert session["tab4_model_type"] == "nuclei", "An unedited classic value takes Studio's change"
    assert session["tab4_flow"] == 0.6 and "tab4_flow" not in changed, "A value being edited in classic is kept"
