"""Raw per-image measurements do not require corrected, batch or slide metadata."""
import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import clustering, consensus, leiden, mr_ml, tasks


@pytest.fixture
def raw_table(tmp_path):
    rng = np.random.default_rng(12)
    population = np.repeat(np.arange(3), 80)
    values = rng.gamma(2, 3, (240, 3)) + np.eye(3)[population] * 100
    frame = pd.DataFrame(values, columns=["CD3", "PanCK", "CD68"])
    frame.insert(0, "label", np.arange(1, 241))
    table = tmp_path / "sample.ome_cell_features_tagged.csv"
    frame.to_csv(table, index=False)
    return tmp_path, table


@pytest.mark.parametrize("method", ["leiden", "som", "vortex"])
@pytest.mark.parametrize("filter_present", [False, True])
def test_real_clustering_raw_single_image_with_cached_optional_columns(raw_table, method, filter_present):
    root, table = raw_table
    if filter_present:
        frame = pd.read_csv(table)
        frame["corrected"] = np.arange(len(frame)) % 2
        frame.to_csv(table, index=False)
    markers = ["CD3", "PanCK", "CD68"]
    defaults = leiden.values(root) if method == "leiden" else clustering.values(root, method)
    config = {**defaults, "markers": markers, "cell_id_col": "label", "batch_id_col": "slide",
              "enable_filter": True, "filter_col": "corrected", "filter_val": "1",
              "enable_low_expr_filter": False, "use_gpu": False, "use_intelligent": False,
              "subsample_method": "random", "n_pcs": 2, "n_neighbors": 10,
              "grid_dim": 3, "n_meta_target": 3, "mode": "fast", "auto_select_k": False,
              "vortex_k": 15, "output_col": "phenotype_" + method}
    settings = {"table": str(table), "project_root": str(root), "markers": markers,
                "cell_id_col": "label", "roi_col": "", "method": method, "cofactor": 5,
                "output_col": config["output_col"] + "_clusters", "phenotype_col": config["output_col"]}
    if method == "leiden":
        settings["pipeline"] = leiden.pipeline_settings(config, table, 5)
    else:
        settings["config"] = config
    settings["revision"] = tasks.source_revision(settings)
    messages = []
    run = leiden.run if method == "leiden" else clustering.run
    found = run(settings, root / "job", lambda text, *_: messages.append(text))
    result = dataio.load(table)[settings["output_col"]]
    assert len(result) == 240
    assert result.notna().sum() == (120 if filter_present else 240)
    if filter_present:
        assert result.iloc[::2].isna().all() and result.iloc[1::2].notna().all()
    else:
        assert any("saved filter corrected is unavailable" in message for message in messages)
    assert any("Normalising across this table: slide is unavailable" in message for message in messages)
    assert found["where"] == {}
    assert set(markers) <= set(found["profile"]["columns"])
    # Neither input measurements nor fabricated metadata are written back.
    assert not {"batch_id", "slide", "cell_id"} & set(pd.read_csv(table).columns)


@pytest.mark.parametrize("module", [consensus, mr_ml])
def test_ml_defaults_do_not_require_tissue_but_preserve_explicit_filters(tmp_path, module):
    assert not module.values(tmp_path)["enable_filter"]
    cache = module.cache_path(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"enable_filter": True, "filter_col": "chosen_tissue"}))
    assert module.values(tmp_path)["enable_filter"]
    assert module.values(tmp_path)["filter_col"] == "chosen_tissue"


def test_consensus_accepts_raw_columns_without_batch_slide_or_tissue(raw_table):
    root, table = raw_table
    frame = pd.read_csv(table)
    for name in ["phenotype_a", "phenotype_b"]:
        frame[name] = np.repeat(["T", "Tumour", "Myeloid"], 80)
    frame.to_csv(table, index=False)
    chosen = {**consensus.values(root), "markers": ["CD3", "PanCK", "CD68"],
              "phenotyping_methods": ["phenotype_a", "phenotype_b"], "min_consensus_methods": 2,
              "n_estimators": 20, "cpu_workers": 2}
    consensus.check_run(chosen, dataio.columns(table))
    settings = {"table": str(table), "project_root": str(root), "markers": chosen["markers"],
                "cell_id_col": "label", "config": chosen, "output_col": chosen["output_phenotype_col"]}
    settings["revision"] = tasks.source_revision(settings)
    consensus.run(settings, root / "job", lambda *_: None)
    assert dataio.load(table)[settings["output_col"]].notna().all()


def test_mr_ml_accepts_raw_reference_cells_without_batch_slide_or_tissue(raw_table):
    root, table = raw_table
    frame = pd.read_csv(table)
    frame["phenotype_ref1"] = np.repeat(["T", "Tumour", "Myeloid"], 80)
    frame.to_csv(table, index=False)
    chosen = {**mr_ml.values(root), "markers": ["CD3", "PanCK", "CD68"], "n_estimators": 20, "cpu_workers": 2}
    settings = mr_ml.job_settings(root, table, {"cell_id": "label", "roi": "", "slide": ""}, chosen)
    assert settings["cell_id_col"] == "label" and not settings["roi_column"]
    assert not settings["config"]["enable_filter"]
    tasks.mr_ml(settings, root / "job", lambda *_: None)
    assert dataio.load(table)[settings["output_col"]].notna().all()
