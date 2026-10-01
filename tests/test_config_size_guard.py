"""The project file holds settings, never data.

A per-cell table nested inside a session value (Tab 8's neighbourhood result)
passed the old per-setting size check, because the check measured the table's
short printed summary rather than the one-object-per-row JSON actually
written. The project file grew to 44 GB and took the interface down on load.
These tests pin every layer of the fix.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from spacemap_gui import config_repair, persistence


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "INTERFACE_STATE_FILE", tmp_path / "interface_state.json")
    monkeypatch.setattr(persistence, "PERSISTENT_SETTINGS_FILE", tmp_path / "interface_settings.json")
    root = tmp_path / "PROJ"
    root.mkdir()
    return root


def _cells(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"batch_id": ["s1r1"] * rows, "label": np.arange(rows),
                         "freq_T": np.random.default_rng(0).random(rows), "CN_cluster": np.arange(rows) % 7})


def test_a_table_is_measured_as_it_is_written_even_when_nested():
    nested = {"signature": "x", "cluster_df": _cells(20_000), "freq_cols": ["freq_T"]}
    assert len(json.dumps(nested, default=str)) < 5_000  # what the old check saw
    assert persistence.estimate_json_bytes(nested) > 1_000_000  # what gets written
    small = pd.DataFrame({"marker": ["CD3", "CD8"], "threshold": [0.4, 0.7]})
    written = len(json.dumps(persistence.sanitize_json_obj(small)))
    assert abs(persistence.estimate_json_bytes(small) - written) < 64


def test_measuring_a_huge_table_stops_early():
    huge = {"cluster_df": _cells(3_000_000), "arrays": [np.zeros(10_000_000)]}
    started = time.perf_counter()
    size = persistence.estimate_json_bytes(huge, limit=persistence._MAX_CONFIG_VALUE_BYTES)
    assert size > persistence._MAX_CONFIG_VALUE_BYTES
    assert time.perf_counter() - started < 1.0


def test_result_holders_are_never_settings():
    for key in ("tab8_nb_result", "tab8_triad_instances_df", "tab8_triad_p1_summary_df",
                "tab8_triad_run_df_inputs", "tools_edit_csv_df", "anything_result"):
        assert persistence.is_transient_key(key), key
    assert not persistence.is_transient_key("resolve_threshold_table")


def test_a_nested_table_is_not_stored_but_small_settings_are():
    config = {"tab_configurations": {}}
    persistence.update_stored_config_key(config, "tab8_nb_custom", {"cluster_df": _cells(50_000)})
    persistence.update_stored_config_key(config, "tab8_nb_k", 20)
    table = pd.DataFrame({"marker": ["CD3"], "threshold": [0.4]})
    persistence.update_stored_config_key(config, "resolve_threshold_table", table)
    stored = {key for tab in config["tab_configurations"].values() for key in tab}
    assert "tab8_nb_custom" not in stored
    assert {"tab8_nb_k", "resolve_threshold_table"} <= stored


def test_saving_leaves_data_out_and_keeps_every_setting(isolated):
    config = {
        "project_name": "PROJ",
        "tab_configurations": {
            "tab8": {"tab8_nb_k": 20, "tab8_leaked": {"cluster_df": _cells(40_000)}},
            "tab7": {"resolve_threshold_table": {"__type__": "DataFrame", "data": [{"marker": "CD3", "threshold": 0.4}]}},
        },
        "huge_section": ["x" * 100] * 400_000,  # 40 MB of something that is not a setting
    }
    persistence.save_project_config(str(isolated), config)
    written = isolated / "PROJ.json"
    assert written.stat().st_size < 1_000_000
    saved = json.loads(written.read_text())
    assert saved["tab_configurations"]["tab8"] == {"tab8_nb_k": 20}
    assert saved["tab_configurations"]["tab7"]["resolve_threshold_table"]["data"][0]["marker"] == "CD3"
    assert "huge_section" not in saved


def _pretty(config: dict) -> str:
    return json.dumps(config, indent=2)


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_an_oversized_file_is_cleaned_by_streaming(tmp_path, position, monkeypatch):
    monkeypatch.setattr(config_repair, "SETTING_LIMIT", 50_000)
    monkeypatch.setattr(config_repair, "_CHUNK", 4096)  # force the skip across many chunks
    records = persistence.sanitize_json_obj(_cells(5_000))
    tab8 = {"tab8_a": 1, "tab8_b": {"nested": [1, 2, {"deep": "x"}]}, "tab8_c": "text, with comma"}
    items = list(tab8.items())
    index = {"first": 0, "middle": 1, "last": 3}[position]
    items.insert(index, ("tab8_nb_result", {"cluster_df": records, "signature": "s"}))
    config = {
        "project_name": "PROJ",
        "directory_mappings": {"a": "${PROJECT_ROOT}/x"},
        "tab_configurations": {"tab7": {"k": [1, 2]}, "tab8": dict(items), "tab9": {}},
        "audit_log": [{"event": "saved"}],
        "portable_state": {"version": 1, "interface_state": {"csv1": {"a": 1}}, "tab_params": {}},
    }
    path = tmp_path / "PROJ.json"
    path.write_text(_pretty(config))
    path.chmod(0o600)
    backup, removed = config_repair.shrink_project_file(path)
    assert path.stat().st_mode & 0o777 == 0o600
    cleaned = json.loads(path.read_text())
    expected = json.loads(_pretty(config))
    del expected["tab_configurations"]["tab8"]["tab8_nb_result"]
    assert cleaned == expected
    assert [entry for entry, _ in removed] == ["tab_configurations/tab8/tab8_nb_result"]
    assert backup.exists() and json.loads(backup.read_text()) == json.loads(_pretty(config))


def test_a_file_in_another_layout_is_left_alone(tmp_path):
    path = tmp_path / "PROJ.json"
    path.write_text(json.dumps({"tab_configurations": {"tab8": {"a": 1}}}))  # one line, not indented
    with pytest.raises(ValueError):
        config_repair.shrink_project_file(path)
    assert json.loads(path.read_text()) == {"tab_configurations": {"tab8": {"a": 1}}}
    assert not list(tmp_path.glob("*.oversized-*"))


def test_loading_an_oversized_project_cleans_it_first(isolated, monkeypatch):
    monkeypatch.setattr(persistence, "_SHRINK_ON_LOAD_BYTES", 100_000)
    monkeypatch.setattr(config_repair, "SETTING_LIMIT", 50_000)
    config = {"project_name": "PROJ", "project_root": str(isolated.resolve()),
              "tab_configurations": {"tab8": {"tab8_nb_k": 20,
                                              "tab8_nb_result": {"cluster_df": persistence.sanitize_json_obj(_cells(5_000))}}}}
    (isolated / "PROJ.json").write_text(_pretty(config))
    loaded = persistence.load_project_config(str(isolated))
    assert loaded["tab_configurations"]["tab8"] == {"tab8_nb_k": 20}
    assert (isolated / "PROJ.json").stat().st_size < 100_000
    assert len(list(isolated.glob("PROJ.json.oversized-*"))) == 1
