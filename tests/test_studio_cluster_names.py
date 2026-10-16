"""Cluster name suggestions from reference cells and from the reference model's marker vote."""

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import cluster_names


@pytest.fixture
def table(tmp_path, monkeypatch):
    """Three populations (CD3, PanCK, CD68 high); clusters match them; some cells labelled by hand."""
    monkeypatch.setenv("SPACEMAP_FAST_COPY", "off")
    rng = np.random.default_rng(1)
    parts = []
    for cluster, (marker, name) in enumerate((("CD3", "T cell"), ("PanCK", "Tumour"), ("CD68", "Macrophage"))):
        frame = pd.DataFrame({m: rng.gamma(2.0, 20.0, 300) for m in ("CD3", "PanCK", "CD68")})
        frame[marker] += rng.gamma(20.0, 60.0, 300)
        frame["phenotype_leiden_clusters"] = f"Cluster {cluster}"
        labels = np.array([""] * 300, dtype=object)
        labels[:12] = name  # a few reference cells in their own cluster
        frame["phenotype_ref1"] = labels
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    combined.loc[320:322, "phenotype_ref1"] = "T cell"  # three T cell references fell in the tumour cluster
    combined.insert(0, "cell_id", [f"c{i}" for i in range(len(combined))])
    path = tmp_path / "combined.csv"
    combined.to_csv(path, index=False)
    return path


def test_each_cluster_gets_its_reference_share_and_a_marker_vote(table):
    assert cluster_names.reference_columns(table) == ["phenotype_ref1"]
    found = cluster_names.suggest(table, "phenotype_leiden_clusters", "phenotype_ref1", ["CD3", "PanCK", "CD68"])
    assert found["by_markers"] and found["labelled"] == 36 + 3
    assert {item["label"]: item["cells"] for item in found["labels"]} == {"T cell": 15, "Tumour": 12, "Macrophage": 12}
    t_cells = found["clusters"]["Cluster 0"]
    assert t_cells["top"][0] == {"label": "T cell", "cells": 12, "of_label": 0.8, "of_cluster": 1.0}
    assert t_cells["suggestions"][0]["label"] == "T cell" and t_cells["suggestions"][0]["by_markers"] > 0.5
    tumour = found["clusters"]["Cluster 1"]
    # Of its own total, 20% of the T cell references are in the tumour cluster; its best suggestion stays Tumour.
    assert {item["label"]: item["of_label"] for item in tumour["top"]} == {"Tumour": 1.0, "T cell": 0.2}
    assert tumour["suggestions"][0]["label"] == "Tumour"
    assert found["clusters"]["Cluster 2"]["suggestions"][0]["label"] == "Macrophage"


def test_without_reference_cells_it_says_so(table):
    import pandas as pd

    frame = pd.read_csv(table)
    frame["phenotype_ref1"] = ""
    frame.to_csv(table, index=False)
    with pytest.raises(ValueError, match="no labelled cells"):
        cluster_names.suggest(table, "phenotype_leiden_clusters", "phenotype_ref1", ["CD3"])


def test_custom_reference_and_pending_labels_refresh_cluster_suggestions(table):
    from spacemap import labels
    from spacemap.viewer import reference
    frame = pd.read_csv(table).rename(columns={"phenotype_ref1":"my_labels"})
    frame.to_csv(table, index=False)
    reference.register_column(table, "my_labels")
    assert cluster_names.reference_columns(table) == ["my_labels"]
    first = cluster_names.suggest(table, "phenotype_leiden_clusters", "my_labels", ["CD3"])
    labels.append(table, [{"row":12, "column":"my_labels", "value":"T cell"}])
    assert cluster_names.suggest(table, "phenotype_leiden_clusters", "my_labels", ["CD3"], cached_only=True) is None
    second = cluster_names.suggest(table, "phenotype_leiden_clusters", "my_labels", ["CD3"])
    assert second["labelled"] == first["labelled"] + 1


def test_prompt_cache_lookup_does_not_load_or_train_on_cell_rows(table, monkeypatch):
    from spacemap import dataio
    def unexpected(*args, **kwargs):
        pytest.fail("Prompt preparation must not load rows for a reference model")
    monkeypatch.setattr(dataio, "load", unexpected)
    assert cluster_names.suggest(table, "phenotype_leiden_clusters", "phenotype_ref1", ["CD3"], cached_only=True) is None


def test_reference_model_reads_marker_rows_only_for_its_sample(table, monkeypatch):
    from spacemap import dataio
    from spacemap_gui.studio import data_view
    monkeypatch.setattr(cluster_names, "SAMPLE", 10)
    monkeypatch.setattr(cluster_names, "PER_LABEL", 5)
    load, measured = dataio.load, data_view._table(str(table))
    take, reads = measured.take, []
    def track_load(path, *args, **kwargs):
        assert not {"CD3", "CD68", "PanCK"}.intersection(kwargs.get("columns") or [])
        return load(path, *args, **kwargs)
    def track_take(rows, columns):
        reads.append((len(rows), columns))
        return take(rows, columns)
    monkeypatch.setattr(dataio, "load", track_load)
    monkeypatch.setattr(measured, "take", track_take)
    found = cluster_names.suggest(table, "phenotype_leiden_clusters", "phenotype_ref1", ["CD3", "PanCK", "CD68"])
    assert found["by_markers"] and reads and reads[0][0] <= 45
