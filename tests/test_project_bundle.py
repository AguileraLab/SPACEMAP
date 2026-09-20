"""<project>.json must be a complete, relocatable record of a project."""

import json
import os
import time
from pathlib import Path

import pytest
import streamlit as st

from spacemap_gui import persistence
from spacemap_gui.project_bundle import (
    BUNDLE_KEY,
    PROJECT_ROOT_TOKEN,
    apply_bundle,
    collect_bundle,
    external_paths,
    localize,
    portablize,
)

EXTERNAL = "/archive/shared/raw/codex"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project with settings in all four historical locations."""
    monkeypatch.setattr(persistence, "INTERFACE_STATE_FILE", tmp_path / "interface_state.json")
    monkeypatch.setattr(persistence, "PERSISTENT_SETTINGS_FILE", tmp_path / "interface_settings.json")
    persistence._RESTORE_POINTS_TAKEN.clear()
    st.session_state.clear()

    root = tmp_path / "site_a" / "MyProject"
    cache = root / "outputs" / "phenotyping" / "cache"
    cache.mkdir(parents=True)
    (cache / "tools_params.json").write_text(json.dumps({
        "tools_pg_geojson_dir": str(root / "outputs" / "segmentation"),
        "tools_pg_png_dpi": 300,
    }))
    (cache / "leiden_params.json").write_text(json.dumps({"resolution": 1.4}))

    config = {
        "project_name": "MyProject",
        "project_root": str(root),
        "unified_csv_filename": "unified.csv",
        "directory_mappings": {
            "bc_output_dir": str(root / "outputs" / "batch_corrected"),
            "batch_codex_dir": EXTERNAL,          # genuinely external
        },
        "tab_configurations": {"tab7_phenotyping": {"viz_seg_dir": str(root / "outputs" / "segmentation")}},
        "audit_log": [],
    }
    (root / "MyProject.json").write_text(json.dumps(config))

    st.session_state["active_project_root"] = str(root)
    persistence.update_interface_state({"tab9_csv_configs": {str(root / "a.csv"): {"dpi": 300}}})

    yield {"root": root, "config": config, "cache": cache, "tmp": tmp_path}
    st.session_state.clear()


# ---------------------------------------------------------------------------
# Path portability
# ---------------------------------------------------------------------------


def test_paths_inside_the_project_become_relocatable(tmp_path):
    root = tmp_path / "proj"
    (root / "outputs").mkdir(parents=True)

    result = portablize({"a": str(root / "outputs"), "b": str(root), "c": EXTERNAL}, root)

    assert result["a"] == f"{PROJECT_ROOT_TOKEN}/outputs"
    assert result["b"] == PROJECT_ROOT_TOKEN
    assert result["c"] == EXTERNAL, "external paths must survive untouched"


def test_portablize_walks_nested_structures(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    payload = {"list": [str(root / "x"), {"deep": str(root / "y")}], "n": 3, "flag": True}

    result = portablize(payload, root)

    assert result["list"][0] == f"{PROJECT_ROOT_TOKEN}/x"
    assert result["list"][1]["deep"] == f"{PROJECT_ROOT_TOKEN}/y"
    assert result["n"] == 3 and result["flag"] is True


def test_localize_is_the_inverse(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    payload = {"a": str(root / "outputs" / "x"), "b": EXTERNAL}

    assert localize(portablize(payload, root), root) == payload


def test_localize_targets_the_new_root(tmp_path):
    old, new = tmp_path / "site_a", tmp_path / "site_b"
    old.mkdir()

    portable = portablize({"d": str(old / "outputs")}, old)

    assert localize(portable, new)["d"] == str(new / "outputs")


def test_external_paths_are_reported_for_the_importer(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    payload = {"inside": str(root / "a"), "outside": EXTERNAL, "other": "/mnt/scratch"}

    reported = external_paths(payload, root)

    assert reported == [EXTERNAL, "/mnt/scratch"]


def test_relative_values_are_left_alone(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()

    assert portablize({"pattern": "*.geojson", "n": 5}, root) == {"pattern": "*.geojson", "n": 5}


# ---------------------------------------------------------------------------
# Collect / apply
# ---------------------------------------------------------------------------


def test_bundle_collects_every_stray_location(project):
    bundle = collect_bundle(
        project["root"],
        interface_state={"tab9_csv_configs": {"x": 1}},
        directory_settings={"sec": {"a": 1}},
    )

    assert set(bundle["tab_params"]) == {"tools_params.json", "leiden_params.json"}
    assert bundle["tab_params"]["leiden_params.json"]["resolution"] == 1.4
    assert bundle["interface_state"] == {"tab9_csv_configs": {"x": 1}}
    assert bundle["directory_settings"] == {"sec": {"a": 1}}


def test_dict_keys_holding_paths_are_converted(tmp_path):
    """tab8/tab9 configs are keyed by absolute CSV path, not just valued by one."""
    root = tmp_path / "proj"
    root.mkdir()
    payload = {"tab9_csv_configs": {str(root / "a.csv"): {"dpi": 300}}}

    portable = portablize(payload, root)
    assert list(portable["tab9_csv_configs"]) == [f"{PROJECT_ROOT_TOKEN}/a.csv"]

    moved = tmp_path / "elsewhere"
    assert list(localize(portable, moved)["tab9_csv_configs"]) == [str(moved / "a.csv")]


def test_bundled_params_have_portable_paths(project):
    bundle = collect_bundle(project["root"])

    stored = bundle["tab_params"]["tools_params.json"]["tools_pg_geojson_dir"]
    assert stored == f"{PROJECT_ROOT_TOKEN}/outputs/segmentation"


def test_apply_writes_params_at_the_new_root(project, tmp_path):
    bundle = collect_bundle(project["root"])
    new_root = tmp_path / "site_b" / "MyProject"
    new_root.mkdir(parents=True)

    interface_state = apply_bundle(new_root, bundle)

    written = json.loads((new_root / "outputs/phenotyping/cache/tools_params.json").read_text())
    assert written["tools_pg_geojson_dir"] == str(new_root / "outputs" / "segmentation")
    assert written["tools_pg_png_dpi"] == 300
    assert interface_state == {}


def test_apply_never_reverts_a_file_newer_than_the_config(project):
    """Reloading a project must not undo edits made after the last save."""
    bundle = collect_bundle(project["root"])
    target = project["cache"] / "leiden_params.json"
    target.write_text(json.dumps({"resolution": 9.9}))
    stale_config_time = target.stat().st_mtime - 100

    apply_bundle(project["root"], bundle, config_mtime=stale_config_time)

    assert json.loads(target.read_text())["resolution"] == 9.9, "local edit must win"


def test_apply_overwrites_when_the_config_is_newer(project):
    bundle = collect_bundle(project["root"])
    target = project["cache"] / "leiden_params.json"
    target.write_text(json.dumps({"resolution": 9.9}))
    newer_config_time = target.stat().st_mtime + 100

    apply_bundle(project["root"], bundle, config_mtime=newer_config_time)

    assert json.loads(target.read_text())["resolution"] == 1.4


def test_apply_never_deletes_unmentioned_files(project):
    bundle = collect_bundle(project["root"])
    extra = project["cache"] / "handmade.json"
    bundle["tab_params"].pop("leiden_params.json")
    extra.write_text("{}")

    apply_bundle(project["root"], bundle)

    assert extra.exists()
    assert (project["cache"] / "leiden_params.json").exists()


def test_apply_rejects_filenames_that_escape_the_cache_directory(project, tmp_path):
    victim = tmp_path / "escaped.json"
    bundle = {"version": 1, "tab_params": {"../../../escaped.json": {"x": 1}}, "interface_state": {}}

    apply_bundle(project["root"], bundle)

    assert not victim.exists(), "a config must not write outside its cache directory"


def test_apply_tolerates_a_missing_bundle(project):
    assert apply_bundle(project["root"], None) == {}
    assert apply_bundle(project["root"], {}) == {}


# ---------------------------------------------------------------------------
# Full round trip
# ---------------------------------------------------------------------------


def test_a_collaborator_can_relocate_the_whole_project(project, tmp_path):
    """Save at one root, hand over only the JSON, load at another root."""
    enriched = persistence.attach_portable_state(str(project["root"]), project["config"])

    serialized = json.dumps(enriched)
    assert str(project["root"]) not in serialized.replace(f'"project_root": "{project["root"]}"', ""), \
        "no absolute in-project path may remain in the shared file"
    assert EXTERNAL in serialized, "external references must be kept for the importer to remap"

    # Collaborator side: a different root, with only the JSON present.
    new_root = tmp_path / "site_b" / "MyProject"
    new_root.mkdir(parents=True)
    (new_root / "MyProject.json").write_text(serialized)
    st.session_state["active_project_root"] = str(new_root)

    localized = persistence.restore_portable_state(json.loads(serialized), str(new_root))

    assert localized["directory_mappings"]["bc_output_dir"] == str(new_root / "outputs" / "batch_corrected")
    assert localized["directory_mappings"]["batch_codex_dir"] == EXTERNAL
    assert localized["tab_configurations"]["tab7_phenotyping"]["viz_seg_dir"] == str(new_root / "outputs" / "segmentation")

    restored = json.loads((new_root / "outputs/phenotyping/cache/tools_params.json").read_text())
    assert restored["tools_pg_geojson_dir"] == str(new_root / "outputs" / "segmentation")
    assert restored["tools_pg_png_dpi"] == 300

    st.session_state["active_project_root"] = str(new_root)
    assert persistence.load_interface_state()["tab9_csv_configs"], "tab 8/9 configs must travel too"


def test_loading_a_config_without_a_bundle_still_works(project):
    """Existing projects predate the bundle and must load unchanged."""
    plain = dict(project["config"])
    plain.pop(BUNDLE_KEY, None)

    localized = persistence.restore_portable_state(plain, str(project["root"]))

    assert localized["directory_mappings"]["bc_output_dir"] == str(project["root"] / "outputs" / "batch_corrected")


def test_external_paths_surface_for_the_import_summary(project):
    enriched = persistence.attach_portable_state(str(project["root"]), project["config"])

    assert EXTERNAL in external_paths(enriched["directory_mappings"], project["root"])


# ---------------------------------------------------------------------------
# Restore points
# ---------------------------------------------------------------------------


def test_only_one_restore_point_per_session(project):
    first = persistence.create_restore_point(project["root"], once_per_session=True)
    second = persistence.create_restore_point(project["root"], once_per_session=True)

    assert first is not None and first.exists()
    assert second is None, "a session takes one snapshot, not one per save"


def test_explicit_restore_points_always_write(project):
    persistence.create_restore_point(project["root"], once_per_session=True)
    manual = persistence.create_restore_point(project["root"], label="manual")

    assert manual is not None and "manual" in manual.name


def test_restore_points_are_capped(project):
    for index in range(8):
        path = persistence.restore_point_dir(project["root"])
        path.mkdir(parents=True, exist_ok=True)
        (path / f"2026010{index}_120000_autosave.json").write_text("{}")

    persistence.prune_restore_points(project["root"], keep=5)
    remaining = persistence.list_restore_points(project["root"])

    assert len(remaining) == 5
    assert remaining[0].name.startswith("20260103"), "the oldest go first"


def test_a_corrupt_config_never_evicts_a_good_restore_point(project):
    (project["root"] / "MyProject.json").write_text("{ not json")

    assert persistence.create_restore_point(project["root"], label="manual") is None
    assert persistence.list_restore_points(project["root"]) == []


def test_no_bak_file_is_written_on_save(project):
    persistence.save_project_config(str(project["root"]), project["config"])

    assert not list(project["root"].glob("*.json.bak")), "per-save .bak copies are gone"
    assert persistence.list_restore_points(project["root"]), "replaced by a restore point"
