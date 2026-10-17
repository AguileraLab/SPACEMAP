"""Spatial analysis and Visualization on the combined table: composition, heatmap profile, distance bands,
method agreement, density with tissue outlines, triads (drawable triangles) and UMAP points linked to cells."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import spatial


@pytest.fixture
def table(tmp_path):
    """Two images; in each, tumour cells on the left (inside the annotation: negative distance) and T cells on the right."""
    rng = np.random.default_rng(3)
    parts = []
    for roi in ("R1", "R2"):
        x, y = rng.uniform(0, 1000, 1500), rng.uniform(0, 1000, 1500)
        tumour = x < 500
        kind = np.where(tumour, np.where(rng.random(1500) < .9, "Tumor", "T cell"), np.where(rng.random(1500) < .85, "T cell", "Macrophage"))
        reference = np.where(rng.random(1500) < .9, kind, "Macrophage")
        parts.append(pd.DataFrame({"batch_id": roi, "label": np.arange(1, 1501), "x_centroid": x, "y_centroid": y,
                                   "phenotype_ml_consensus": kind, "phenotype_ref1": reference, "tumor_distance": (x - 500) * 0.5,
                                   "CD3": np.where(kind == "T cell", 800, 50) + rng.normal(0, 20, 1500),
                                   "PanCK": np.where(kind == "Tumor", 900, 40) + rng.normal(0, 20, 1500)}))
    path = tmp_path / "combined.csv"
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    return path


def test_quick_results(table):
    assert spatial.markers_of(table) == ["CD3", "PanCK"]
    found = spatial.composition(table, "phenotype_ml_consensus")
    assert found["images"] == ["R1", "R2"] and found["groups"][0] in ("T cell", "Tumor") and found["cells"] == 3000
    profile = spatial.profile(table, "phenotype_ml_consensus", ["CD3", "PanCK"])["profile"]
    row = dict(zip(profile["rows"], profile["values"]))
    assert row["T cell"][0] > 0.5 and row["Tumor"][1] > 0.5
    bands = spatial.distance_profile(table, "phenotype_ml_consensus", "tumor_distance", step=50, limit=250)
    tumour = bands["profile"]["values"][bands["groups"].index("Tumor")]
    assert tumour[0] > 0 and tumour[-1] < 0  # more tumour inside (negative), less far outside
    agree = spatial.agreement(table, "phenotype_ref1", "phenotype_ml_consensus", ["R1"])
    assert 0.8 < agree["accuracy"] < 1 and {p["phenotype"] for p in agree["phenotypes"]} == {"Tumor", "T cell", "Macrophage"}


def test_density_triads_and_umap(table, tmp_path):
    report = lambda *a: None
    density = spatial.density_job({"root": str(tmp_path), "table": str(table), "column": "phenotype_ml_consensus", "pixel_um": 1.0}, None, report)
    assert density["units"] == ["R1", "R2"] and 0.8 < density["areas_mm2"][0] < 1.1  # about the 1 mm² square
    outline = json.loads((Path(density["folder"]) / "R1_tissue_outline.geojson").read_text())
    assert outline["features"][0]["properties"]["area_mm2"] > 0.8
    triads = spatial.triads_job({"root": str(tmp_path), "table": str(table), "images": ["R1"], "values": {"phenotype_col": "phenotype_ml_consensus",
                                 "triad": ["Tumor", "Tumor", "T cell"], "min_um": 0, "max_um": 40, "pixel_um": 1.0, "min_cells": 20, "n_permutations": 20}}, None, report)
    assert triads["summary"]["observed_total_count"] > 0 and triads["per_image"][0]["image"] == "R1"
    assert triads["table_name"].startswith("Triads / triad_instances_studio_")
    shapes = spatial.triad_shapes(triads["table"], "R1", limit=10)
    assert shapes["count"] == 10 and len(shapes["rings"][0]) == 4
    made = spatial.umap_job({"root": str(tmp_path), "table": str(table), "markers": ["CD3", "PanCK"], "per_image": 300}, None, report)
    points = spatial.umap_points(tmp_path, made["name"], table, "phenotype_ml_consensus")
    assert len(points["x"]) == 600 and set(points["image"]) == {"R1", "R2"} and points["groups"][0] in ("Tumor", "T cell", "Macrophage")
    assert spatial.saved_umaps(tmp_path)[0]["name"] == made["name"]


def test_screening_settings_are_classic(table):
    config = spatial.screening_config(table, {"patient_col": "batch_id", "phenotype_col": "phenotype_ml_consensus", "distance_col": "tumor_distance",
                                              "endpoint": {"type": "binary", "group_col": "phenotype_ref1", "positive_value": "Tumor"}})
    assert config["columns"]["distance"] == "tumor_distance" and config["endpoints"][0]["type"] == "binary"
    assert config["screening"]["n_candidate_widths"] == 15 and "triad" in config["feature_families"]


def test_feature_screening_finds_the_planted_difference(tmp_path):
    """16 patients: responders have more T cells beside the tumour. Classic's screening, run by Studio, finds a T cell
    or macrophage feature first and names the images where it is highest and lowest."""
    rng = np.random.default_rng(1)
    parts = []
    for p in range(16):
        n, responder = 300, p % 2 == 0
        x, y = rng.uniform(0, 800, n), rng.uniform(0, 800, n)
        kind = np.where(x < 400, "Tumor", np.where(rng.random(n) < (.5 if responder else .15), "T cell", "Macrophage"))
        parts.append(pd.DataFrame({"batch_id": f"P{p}_roi_1", "patient": f"P{p}", "label": np.arange(1, n + 1), "x_centroid": x, "y_centroid": y,
                                   "phenotype_ml_consensus": kind, "tumor_distance": (x - 400) * 0.5, "response": "yes" if responder else "no"}))
    table = tmp_path / "combined.csv"
    pd.concat(parts).to_csv(table, index=False)
    found = spatial.screening_job({"root": str(tmp_path), "table": str(table), "values": {"patient_col": "patient", "phenotype_col": "phenotype_ml_consensus",
                                   "distance_col": "tumor_distance", "endpoint": {"type": "binary", "group_col": "response", "positive_value": "yes"}, "pixel_um": 0.5}},
                                  None, lambda *a: None)
    best = found["hits"][0]
    assert best["p"] < 0.01 and set(best["phenotypes"]) & {"T cell", "Macrophage"} and len(best["highest"]) == 5
    assert (Path(found["folder"]) / "tables" / "screening_results_all.csv").is_file()
