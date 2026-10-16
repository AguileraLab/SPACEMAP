"""Heatmap values must describe the clustering's actual analysis scale, not another transform."""
import time

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from spacemap_gui.studio import cluster_profiles as cp, classification, clustering, leiden, jobs, chat
from spacemap_gui.viewer.api import app
from test_studio_chat import _job
from test_studio_commands import setup  # noqa: F401


@pytest.fixture
def frame():
    # A few bright cells deliberately distinguish the mean from the median.
    return pd.DataFrame({"leiden_cluster": [0, 0, 0, 1, 1, 1, 2, 2, 2, -1],
        "A_corr": [1, 1, 20, 5, 6, 7, 3, 3, 3, 900],
        "B_corr": [3, 4, 5, 4, 4, 50, 2, 2, 3, 800],
        "batch_id": ["R1"] * 5 + ["R2"] * 5})


@pytest.mark.parametrize("method", ["leiden", "som", "vortex"])
@pytest.mark.parametrize("flags", [{}, {"norm_z1": True},
    {"norm_z1": True, "norm_z2": True, "norm_cdf": True, "norm_nlog": True}])
def test_uses_exact_selected_transformations_and_no_extra_arcsinh(frame, method, flags):
    from spacemap.ph_leiden_utils import apply_flexible_normalization
    markers = ["A_corr", "B_corr"]
    config = {"markers_selected": markers, "cofactor": 5., "norm_sinh": False, "batch_id_col": "batch_id", **flags}
    columns = [m + "_norm" for m in markers]
    if flags:
        frame[columns] = apply_flexible_normalization(frame, markers, ["batch_id"], "_norm",
            apply_z1=flags.get("norm_z1", False), apply_z2=flags.get("norm_z2", False),
            apply_cdf=flags.get("norm_cdf", False), apply_nlog=flags.get("norm_nlog", False))
    else:
        frame[columns] = frame[markers].to_numpy()
    if method == "leiden":
        clusters, profile = leiden.describe(frame, columns, config)
    else:
        clusters, profile = clustering.describe(frame, columns, markers, "leiden_cluster", config)
    assert profile["aggregation"] == "median" and profile["columns"] == markers
    assert "Low expression" not in profile["rows"]
    assert next(c for c in clusters if c["name"] == "Low expression")["cells"] == 1
    valid = frame[frame.leiden_cluster >= 0]
    for summary in ("median", "mean"):
        expected = valid.groupby("leiden_cluster")[columns].agg(summary)
        z = ((expected - expected.mean()) / expected.std()).fillna(0)
        np.testing.assert_allclose(profile["summaries"][summary]["raw"], expected, atol=.000051)
        np.testing.assert_allclose(profile["summaries"][summary]["values"], z, atol=.00051)
    if not flags:
        assert profile["raw"][0][0] == 1.0  # arcsinh(1/5) would be 0.1987
        assert "no additional transformation" in profile["note"]
    else:
        assert "Z1 within batch_id" in profile["note"]
    assert "Corrected (arcsinh)" in profile["note"]
    assert ("−log(1−CDF)" in profile["note"]) == bool(flags.get("norm_nlog"))


def test_mean_matches_classic_heatmap(frame):
    from spacemap_gui.tabs.tab7_phenotyping.common import build_leiden_heatmap_figure
    frame = frame[frame.leiden_cluster >= 0].copy()
    frame["A_corr_norm"], frame["B_corr_norm"] = frame.A_corr, frame.B_corr
    columns = ["A_corr_norm", "B_corr_norm"]
    _, profile = cp.build(frame, columns, "leiden_cluster")
    figure, *_ = build_leiden_heatmap_figure("summary-regression", frame, columns, False, False, False)
    assert figure is not None
    np.testing.assert_allclose(profile["summaries"]["mean"]["values"], np.asarray(figure.data[0].z).T, atol=.00051)
    assert profile["raw"] != profile["summaries"]["mean"]["raw"]
    assert profile["top_markers"] != profile["summaries"]["mean"]["top_markers"]


def test_constant_and_single_group_remain_finite():
    frame = pd.DataFrame({"group": [0, 0, None], "A_norm": [3., 3., 99.]})
    _, profile = cp.build(frame, ["A_norm"], "group")
    assert profile["raw"] == [[3.]] and profile["values"] == [[0.]]
    with pytest.raises(ValueError, match="missing"):
        cp.build(frame, ["B_norm"], "group")


@pytest.mark.parametrize("method", sorted(cp.METHODS))
def test_job_preference_is_shared_by_ui_top_markers_and_ai(tmp_path, frame, method):
    clusters, profile = cp.build(frame, ["A_corr", "B_corr"], "leiden_cluster")
    result = {"table": str(tmp_path / "cells.csv"), "column": "phenotype_clusters", "cells": len(frame),
              "markers": ["A_corr", "B_corr"], "clusters": clusters, "profile": profile}
    job_id = _job(tmp_path, method, result)
    assert jobs.status(tmp_path, job_id)["result"]["profile"]["aggregation"] == "median"
    cp.preference(tmp_path, method, "mean")
    found = jobs.status(tmp_path, job_id)["result"]
    assert found["profile"]["aggregation"] == "mean"
    assert found["profile"]["raw"] == profile["summaries"]["mean"]["raw"]
    assert found["clusters"][0]["top_markers"] == profile["summaries"]["mean"]["top_markers"].get(clusters[0]["name"], [])
    if method in chat.CLUSTERING:
        evidence = chat.results({"root": str(tmp_path), "unified": result["table"]}, cached_only=True)
        assert evidence[0]["profile"]["aggregation"] == "mean"
        assert "means_before_cluster_zscore" in evidence[0]["profile"]
        assert "medians_before_cluster_zscore" not in evidence[0]["profile"]
    cp.preference(tmp_path, method, "median")
    assert jobs.status(tmp_path, job_id)["result"]["profile"]["raw"] == profile["raw"]


def test_legacy_leiden_rebuilds_from_saved_run_without_reclustering(tmp_path, frame):
    markers = ["A_corr", "B_corr"]
    for m in markers:
        frame[m + "_norm"] = frame[m] * 2
    result = {"column": "phenotype_clusters", "table": "cells.parquet", "clusters": [{"name": "Cluster 0", "cells": 3}],
              "profile": {"rows": ["Cluster 0"], "columns": markers, "values": [[9, 9]], "raw": [[9, 9]]}}
    job_id = _job(tmp_path, "leiden", result, {"pipeline": {"markers_selected": markers, "norm_z1": True}})
    folder = tmp_path / "outputs/studio/jobs" / job_id
    frame.to_parquet(folder / "results.parquet")
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    for _ in range(100):
        found = jobs.status(tmp_path, job_id)["result"]
        if not found.get("profile_pending"):
            break
        assert found["profile"] is None and not any(c["top_markers"] for c in found["clusters"])
        time.sleep(.01)
    assert found["profile"]["version"] == cp.VERSION
    assert found["profile"]["raw"][0][0] == 2.
    assert "Z1" in found["profile"]["note"]
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before


def test_summary_api_validates_and_persists(setup):  # noqa: F811
    channel, _, root, _ = setup
    client = TestClient(app)
    url = f"/studio/{channel}/heatmap_summary"
    for body in ({}, {"method": "leiden", "aggregation": "sum"}, {"method": "unknown", "aggregation": "mean"}):
        assert client.post(url, json=body).status_code == 400
    assert client.post(url, json={"method": "leiden", "aggregation": "mean"}).json() == {"aggregation": "mean"}
    assert cp.preference(root, "leiden") == "mean"


def test_classification_default_and_merged_medians_use_all_pixels(tmp_path):
    z = np.array([[0., 1], [0, 2], [100, 3], [10, 8], [10, 8], [10, 8], [10, 8]], dtype=np.float32)
    ids = np.array([0, 0, 0, 1, 1, 1, 1])
    prepared = {"factor": 2, "stack": {"marker_names": ["A", "B"], "pixels_z_valid": z, "pixels_original_valid": z / 100}}
    found = {"run_id": "a" * 24, "cluster_map": ids.reshape(1, -1), "counts": np.bincount(ids),
        "z_sums": np.array([z[ids == i].sum(0) for i in range(2)]),
        "raw_sums": np.array([(z[ids == i] / 100).sum(0) for i in range(2)]),
        "cluster_colors": ["#ff0000", "#0000ff"], "execution": {}, "provenance": {}}
    first = classification.render(prepared, found, {})
    assert first["profile"]["aggregation"] == "median"
    assert first["profile"]["values"][0] == [0., 2.]
    merged = classification.render(prepared, found, {"groups": [[0, 1]]})
    np.testing.assert_allclose(merged["profile"]["values"], [np.median(z, axis=0)])
    assert merged["profile"]["values"][0][0] == 10.  # not median([0, 10]) == 5
    mean = classification.render(prepared, found, {"groups": [[0, 1]], "aggregation": "mean"})
    np.testing.assert_allclose(mean["profile"]["values"], [z.mean(0)], atol=.000051)
    assert merged["png"] == mean["png"]  # heatmap choice does not move pixels
    classification.persist(tmp_path, "image.ome.tiff", prepared, found)
    reopened = classification.load(tmp_path, "image.ome.tiff", found["run_id"])
    saved = classification.render(*reopened, {"groups": [[0, 1]]})
    assert saved["profile"]["values"] == merged["profile"]["values"]
    assert saved["profile"]["raw"] == merged["profile"]["raw"]
    assert "before Z-score" in saved["profile"]["raw_label"]
    # Old files with only sums must never pretend to contain medians.
    old = {"factor": 2, "stack": {"marker_names": ["A", "B"]}}
    old_found = {k: v for k, v in found.items() if k != "_profile_medians"}
    legacy = classification.render(old, old_found, {})
    assert legacy["profile"]["aggregation"] == "mean"
    assert legacy["profile"]["available_summaries"] == ["mean"]
    assert "Older run" in legacy["profile"]["note"]
