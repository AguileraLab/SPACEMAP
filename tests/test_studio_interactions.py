"""Studio's cell-cell interactions run Tab 8's test; classic Tab 8's triads and screening runs are read for Studio."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap_gui.studio import interactions, tab8_results
from test_studio_neighbourhoods import table  # noqa: F401  (fixture: tumour and immune halves in two ROIs)


def test_neighbours_of_the_same_half_are_enriched(table, tmp_path):  # noqa: F811
    chosen = {
        **interactions.values({}),
        "phenotype_col": "phenotype_resolve",
        "roi_col": "",
        "radius_um": 20.0,
        "pixel_um": 1.0,
        "n_permutations": 60,
    }
    found = interactions.run({"table": str(table), "values": chosen}, tmp_path / "job", lambda *args: None)
    rows = found["profile"]["rows"]
    z = pd.DataFrame(found["profile"]["raw"], index=rows, columns=rows)
    assert set(rows) == {"Tumor", "T cell", "Macrophage"} and found["meta"]["n_regions_used"] == 2
    assert z.loc["Tumor", "Tumor"] > 3 and z.loc["Tumor", "Macrophage"] < -3
    assert found["pairs"][0]["z"] == max(found["pairs"], key=lambda pair: abs(pair["z"]))["z"]
    assert (tmp_path / "job" / "cci_all_pairs.csv").is_file() and found["column"] == "phenotype_resolve"
    json.dumps(found, allow_nan=False)


def test_settings_are_tab8s_keys():
    for wrong in ({"radius_um": 0}, {"n_permutations": 10.5}, {"colour": "x"}, {"phenotype_col": 3}):
        with pytest.raises(ValueError):
            interactions.check(wrong)
    config = {
        "tab_configurations": {"tab8_analysis": {"tab8_cci_radius_um": 12.0, "tab8_cci_pheno_col": "phenotype_x"}}
    }
    assert interactions.values(config)["radius_um"] == 12.0 and interactions.values(config)["n_permutations"] == 500


def test_classic_tab8_results_are_read(tmp_path):
    triad = tmp_path / "outputs/spatial_analysis/triad"
    triad.mkdir(parents=True)
    pd.DataFrame(
        {"batch": "R1", "triad_label": ["A::B::C"] * 3 + ["A::A::B"], "triad_centroid_x": 1.0, "triad_centroid_y": 2.0}
    ).to_csv(triad / "triad_instances_local.csv", index=False)
    run = tmp_path / "outputs/spatial_feature_screening/feature_screening_1"
    (run / "tables").mkdir(parents=True)
    (run / "figures").mkdir()
    pd.DataFrame(
        {
            "feature_label": ["f1", "f2", "f3"],
            "endpoint_label": "OS",
            "p_value": [0.2, 0.001, 0.04],
            "fdr_q": [0.3, 0.01, 0.06],
        }
    ).to_csv(run / "tables" / "screening_results_all.csv", index=False)
    from PIL import Image

    Image.new("RGB", (2400, 1200), "white").save(run / "figures" / "forest_plot.png")
    found = tab8_results.triads(tmp_path)
    assert found == [
        {
            "name": "Triads / triad_instances_local.csv",
            "triads": 4,
            "types": [{"label": "A::B::C", "count": 3}, {"label": "A::A::B", "count": 1}],
        }
    ]
    screening = tab8_results.screening(tmp_path)
    assert screening["runs"] == ["feature_screening_1"] and screening["tested"] == 3 and screening["significant"] == 1
    assert [row["feature_label"] for row in screening["hits"]] == ["f2", "f3", "f1"]
    assert screening["figures"][0]["name"] == "forest plot" and screening["figures"][0]["image"].startswith(
        "data:image/png;base64,"
    )
    assert tab8_results.screening(tmp_path / "none") == {"runs": []}


def test_only_the_chosen_images_are_tested(table, tmp_path):  # noqa: F811
    chosen = {**interactions.values({}), "phenotype_col": "phenotype_resolve", "batch_col": "batch_id", "roi_col": "",
              "radius_um": 20.0, "pixel_um": 1.0, "n_permutations": 30}
    found = interactions.run({"table": str(table), "values": chosen, "images": ["R1"], "image_col": "batch_id"}, tmp_path / "job", lambda *args: None)
    assert found["images"] == ["R1"] and found["meta"]["n_regions_used"] == 1
