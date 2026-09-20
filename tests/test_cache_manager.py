"""The cache sweep must free space without ever touching data or other projects."""

import json

import pytest

from spacemap_gui.cache_manager import (
    CATEGORY_LABELS,
    clear_entry,
    format_size,
    scan_caches,
    user_cache_roots,
)


@pytest.fixture
def project(tmp_path):
    """A project with every kind of file the app writes, plus a second project."""
    root = tmp_path / "projA"
    phen = root / "outputs" / "phenotyping"
    (phen / "cache").mkdir(parents=True)
    (phen / "temp").mkdir(parents=True)
    (phen / "leiden_cache").mkdir(parents=True)

    (phen / "cache" / "leiden_results_1.parquet").write_bytes(b"x" * 4000)
    (phen / "cache" / "tools_params.json").write_text(json.dumps({"a": 1}))
    (phen / "cache" / "leiden_params.json").write_text(json.dumps({"b": 2}))
    (phen / "temp" / "leiden_temp_1.parquet").write_bytes(b"y" * 8000)
    (phen / "leiden_cache" / "res_abc.parquet").write_bytes(b"z" * 2000)

    # Real analysis output that must survive every sweep
    (root / "outputs" / "batch_corrected").mkdir(parents=True)
    unified = root / "outputs" / "batch_corrected" / "unified.csv"
    unified.write_text("cell_id\n1\n")
    config = root / "projA.json"
    config.write_text(json.dumps({"project_name": "projA"}))

    other = tmp_path / "projB" / "outputs" / "phenotyping" / "cache"
    other.mkdir(parents=True)
    (other / "leiden_results_1.parquet").write_bytes(b"q" * 5000)

    env = {
        "SPACEMAP_CACHE_DIR": str(tmp_path / "usercache"),
        "NUMBA_CACHE_DIR": str(tmp_path / "usercache" / "numba_cache"),
        "CELLPOSE_LOCAL_MODELS_PATH": str(tmp_path / "usercache" / "cellpose_cache"),
    }
    numba = tmp_path / "usercache" / "numba_cache"
    numba.mkdir(parents=True)
    (numba / "kernel.nbi").write_bytes(b"n" * 1000)

    return {"root": root, "phen": phen, "env": env, "unified": unified,
            "config": config, "other": other, "tmp": tmp_path}


def _by_key(entries):
    return {e.key: e for e in entries}


def test_scan_finds_each_category(project):
    entries = _by_key(scan_caches(project["root"], project["env"]))

    assert entries["numba"].category == "recomputed"
    assert entries["temp_results"].category == "results"
    assert entries["tab_params"].category == "settings"
    assert set(CATEGORY_LABELS) == {"recomputed", "results", "settings"}


def test_sizes_and_counts_are_reported(project):
    entries = _by_key(scan_caches(project["root"], project["env"]))

    assert entries["temp_results"].size_bytes == 8000
    assert entries["temp_results"].file_count == 1
    assert entries["cached_results"].size_bytes == 4000
    assert entries["tab_params"].file_count == 2
    assert entries["numba"].size_bytes == 1000


def test_parquet_and_json_in_one_folder_are_not_double_counted(project):
    """cache/ holds both snapshots and settings; each entry must claim only its own."""
    entries = _by_key(scan_caches(project["root"], project["env"]))

    snapshots = {p.name for p in entries["cached_results"].files}
    settings = {p.name for p in entries["tab_params"].files}

    assert snapshots == {"leiden_results_1.parquet"}
    assert settings == {"tools_params.json", "leiden_params.json"}
    assert not snapshots & settings


def test_clearing_results_leaves_settings_and_data_intact(project):
    entries = _by_key(scan_caches(project["root"], project["env"]))

    removed, freed = clear_entry(entries["temp_results"])

    assert (removed, freed) == (1, 8000)
    assert not list((project["phen"] / "temp").glob("*.parquet"))
    assert (project["phen"] / "temp").is_dir(), "the directory itself must remain"
    assert (project["phen"] / "cache" / "tools_params.json").exists()
    assert project["unified"].read_text() == "cell_id\n1\n"
    assert project["config"].exists()


def test_clearing_settings_leaves_snapshots(project):
    entries = _by_key(scan_caches(project["root"], project["env"]))

    clear_entry(entries["tab_params"])

    assert not list((project["phen"] / "cache").glob("*.json"))
    assert (project["phen"] / "cache" / "leiden_results_1.parquet").exists()


def test_a_whole_directory_entry_is_emptied_but_recreated(project):
    entries = _by_key(scan_caches(project["root"], project["env"]))

    removed, freed = clear_entry(entries["norm_cache"])

    assert (removed, freed) == (1, 2000)
    assert (project["phen"] / "leiden_cache").is_dir()
    assert not list((project["phen"] / "leiden_cache").iterdir())


def test_a_sweep_never_reaches_another_project(project):
    for entry in scan_caches(project["root"], project["env"]):
        clear_entry(entry)

    survivor = project["other"] / "leiden_results_1.parquet"
    assert survivor.exists(), "another project's files must be untouched"
    assert survivor.stat().st_size == 5000


def test_clearing_is_idempotent(project):
    entries = _by_key(scan_caches(project["root"], project["env"]))

    clear_entry(entries["temp_results"])
    removed, freed = clear_entry(entries["temp_results"])

    assert (removed, freed) == (0, 0)


def test_missing_directories_are_reported_as_empty(tmp_path):
    entries = _by_key(scan_caches(tmp_path / "nonexistent", {"SPACEMAP_CACHE_DIR": str(tmp_path / "none")}))

    assert entries["temp_results"].size_bytes == 0
    assert entries["temp_results"].file_count == 0
    assert clear_entry(entries["temp_results"]) == (0, 0)


def test_scan_without_a_project_returns_only_machine_caches(project):
    keys = {e.key for e in scan_caches(None, project["env"])}

    assert keys == {"numba", "cellpose"}


def test_user_cache_roots_follow_the_environment(tmp_path):
    roots = user_cache_roots({"NUMBA_CACHE_DIR": str(tmp_path / "nb")})

    assert roots["numba"] == tmp_path / "nb"
    assert "cellpose" in roots


def test_size_formatting():
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.00 KB"
    assert format_size(5 * 1024 ** 2) == "5.00 MB"
    assert format_size(3 * 1024 ** 3) == "3.00 GB"
