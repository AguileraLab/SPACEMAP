"""Studio's neighbourhoods run Tab 8's features and clustering, then save the names, in new columns."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import leiden, neighbourhoods, tasks


@pytest.fixture
def table(tmp_path):
    """Two ROIs, each with a tumour half (x < 500) and an immune half."""
    rng = np.random.default_rng(7)
    parts = []
    for roi in ("R1", "R2"):
        x, y = rng.uniform(0, 1000, 800), rng.uniform(0, 1000, 800)
        immune = np.where(rng.random(800) < 0.85, "T cell", "Macrophage")
        tumour = np.where(rng.random(800) < 0.9, "Tumor", "T cell")
        parts.append(
            pd.DataFrame(
                {
                    "batch_id": roi,
                    "x_centroid": x,
                    "y_centroid": y,
                    "phenotype_resolve": np.where(x < 500, tumour, immune),
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame.insert(0, "cell_id", [f"c{i}" for i in range(len(frame))])
    path = tmp_path / "combined.csv"
    frame.to_csv(path, index=False)
    return path


def job(path, **changes):
    chosen = {
        **neighbourhoods.values({}),
        "phenotype_col": "phenotype_resolve",
        "roi_col": "",
        "n_neighborhoods": 2,
        **changes,
    }
    settings = {
        "table": str(path),
        "markers": ["phenotype_resolve", "x_centroid", "y_centroid"],
        "output_col": chosen["output_col"] + "_cn",
        "names_col": chosen["output_col"],
        "cell_id_col": "cell_id",
        "result": "neighbourhoods",
        "values": chosen,
    }
    settings["revision"] = tasks.source_revision(settings)
    return settings


def test_the_neighbourhoods_split_tumour_from_immune(table, tmp_path):
    found = neighbourhoods.run(job(table), tmp_path / "job", lambda *args: None)
    assert found["column"] == "neighborhood_cn" and found["phenotype_column"] == "neighborhood"
    assert len(found["clusters"]) == 2 and {item["top_markers"][0] for item in found["clusters"]} == {"Tumor", "T cell"}
    # Where each neighbourhood is: its cells per ROI, most first, adding up to its size.
    where = found["where"]
    assert set(where) == {item["name"] for item in found["clusters"]}
    for item in found["clusters"]:
        assert {place["roi"] for place in where[item["name"]]} <= {"R1", "R2"}
        assert sum(place["cells"] for place in where[item["name"]]) == item["cells"]
    profile = found["profile"]
    assert profile["aggregation"] == "mean" and profile["raw_label"] == "Mean neighbourhood share"
    assert sorted(profile["rows"]) == ["CN 0", "CN 1"] and set(profile["columns"]) == {"Macrophage", "T cell", "Tumor"}
    shares = dict(zip(profile["rows"], profile["raw"]))
    assert sum(shares["CN 0"]) == pytest.approx(1.0, abs=1e-3)
    frame = dataio.load(table, columns=["x_centroid", "neighborhood_cn"])
    tumour = next(item["name"] for item in found["clusters"] if item["top_markers"][0] == "Tumor")
    assert (frame.loc[frame["x_centroid"] < 400, "neighborhood_cn"] == tumour).mean() > 0.95
    names = {item["name"]: ("Tumour core" if item["name"] == tumour else "Immune zone") for item in found["clusters"]}
    naming = {
        "table": str(table),
        "markers": found["markers"],
        "output_col": "neighborhood",
        "clusters_col": "neighborhood_cn",
        "cell_id_col": "cell_id",
        "result": "neighbourhoods",
        "names": leiden.check_names(names, found["clusters"]),
    }
    naming["revision"] = tasks.source_revision(naming)
    named = leiden.name_clusters(naming, tmp_path / "job2", lambda *args: None)
    assert named["column"] == "neighborhood" and set(named["phenotypes"]) == {"Tumour core", "Immune zone"}
    # The name column must be new: a second run with the same name is refused.
    with pytest.raises(ValueError, match="already exists"):
        tasks.validate_table({**naming, "revision": None})


def test_settings_are_tab8s_keys(tmp_path):
    from spacemap_gui.studio.settings import read_config, revision

    root = tmp_path / "study"
    root.mkdir()
    (root / "study.json").write_text(json.dumps({"tab_configurations": {"tab8_analysis": {"tab8_nb_k": 15}}}))
    before = neighbourhoods.values(read_config(root))
    assert before["k_neighbors"] == 15 and before["n_neighborhoods"] == 10 and before["roi_col"] == "tissue_roi"
    neighbourhoods.save(root, {"phenotype_col": "phenotype_resolve", "n_neighborhoods": 6}, revision(before))
    stored = json.loads((root / "study.json").read_text())["tab_configurations"]["tab8_analysis"]
    assert stored["tab8_nb_pheno_col"] == "phenotype_resolve" and stored["tab8_nb_kmeans_k"] == 6
    for wrong in ({"k_neighbors": 2}, {"normalize": 1}, {"output_col": "phenotype_x"}, {"shape": 1}):
        with pytest.raises(ValueError):
            neighbourhoods.check(wrong)


@pytest.mark.parametrize("name", ["neighborhood", "neighbourhood", "neighborhood_cohort", "neighbourhoods_2"])
def test_result_names_save_and_validate(table, tmp_path, name):
    from spacemap_gui.studio.settings import revision

    root = tmp_path / "project"
    root.mkdir()
    (root / "project.json").write_text("{}")
    saved = neighbourhoods.save(root, {"output_col": " " + name + " "}, revision(neighbourhoods.values({})))
    assert saved["output_col"] == name
    assert tasks.validate_table(job(table, output_col=name))


def test_only_the_chosen_images_get_neighbourhoods(table, tmp_path):
    settings = {**job(table, output_col="neighborhood_one"), "images": ["R2"], "image_col": "batch_id"}
    found = neighbourhoods.run(settings, tmp_path / "job", lambda *args: None)
    frame = dataio.load(table, columns=["batch_id", "neighborhood_one_cn"])
    assert found["images"] == ["R2"] and set(found["where"]) and all(place["roi"] == "R2" for places in found["where"].values() for place in places)
    assert frame.loc[frame["batch_id"] == "R1", "neighborhood_one_cn"].isna().all() and frame.loc[frame["batch_id"] == "R2", "neighborhood_one_cn"].notna().all()
    from spacemap_gui.studio import phenotyping_overview

    assert phenotyping_overview.results(str(table))["neighborhood_one_cn"] == {"R2": 800}
    with pytest.raises(ValueError, match="None of the chosen images"):
        neighbourhoods.run({**job(table, output_col="neighborhood_none"), "images": ["R9"], "image_col": "batch_id"}, tmp_path / "job2", lambda *args: None)


def test_older_neighbourhood_profiles_are_described_as_mean_shares(tmp_path):
    from spacemap_gui.studio import chat
    from test_studio_chat import _job

    _job(tmp_path, "neighbourhoods", {"table": "cells.csv", "column": "neighborhood_cn",
        "phenotype_column": "neighborhood", "clusters": [{"name": "CN 0", "cells": 100, "top_markers": ["B"]}],
        "profile": {"columns": ["B", "CD4"], "rows": ["CN 0"], "values": [[1., -1.]], "raw": [[.8, .2]]}})
    profile = chat.results({"root": str(tmp_path)}, cached_only=True)[0]["profile"]
    assert profile["aggregation"] == "mean"
    assert profile["means_before_cluster_zscore"] == {"CN 0": [.8, .2]}
    assert "local phenotype fractions" in profile["summary_scale"]
    assert "rankings can differ" in profile["summary_scale"]
