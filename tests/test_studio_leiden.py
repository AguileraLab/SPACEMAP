"""Studio's Leiden runs Tab 7's pipeline, saves the clusters, then the names, in new columns."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import leiden, tasks
from test_studio_commands import run, setup  # noqa: F401  (fixture)


@pytest.fixture
def project(tmp_path):
    """Three clear populations (CD3, PanCK and CD68 high) in two ROIs."""
    rng = np.random.default_rng(4)
    rows = []
    for roi in ("R1", "R2"):
        for population, marker in enumerate(("CD3", "PanCK", "CD68")):
            cells = 200
            frame = pd.DataFrame({name: rng.gamma(2.0, 20.0, cells) for name in ("CD3", "PanCK", "CD68", "CD20")})
            frame[marker] += rng.gamma(20.0, 60.0, cells)
            frame["batch_id"] = roi
            frame["label"] = np.arange(cells) + population * cells
            rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    table.insert(0, "cell_id", [f"cell_{i}" for i in range(len(table))])
    root = tmp_path / "study"
    (root / "outputs/batch_corrected").mkdir(parents=True)
    path = root / "outputs/batch_corrected/combined.csv"
    table.to_csv(path, index=False)
    return root, path


def settings_for(root, path, **changes):
    values = {**leiden.values(root), "markers": ["CD3", "PanCK", "CD68", "CD20"], "use_gpu": False, **changes}
    values["subsample_size"] = 100000
    return {
        "table": str(path),
        "markers": values["markers"],
        "output_col": values["output_col"] + "_clusters",
        "phenotype_col": values["output_col"],
        "cell_id_col": "cell_id",
        "project_root": str(root),
        "pipeline": leiden.pipeline_settings(values, path, 5.0),
    }


def test_settings_are_tab7s_file(project):
    from spacemap_gui.studio.settings import revision

    root, _ = project
    cache = leiden.cache_path(root)
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"resolution": 1.2, "min_marker_sum": -0.5, "input_file": "kept.csv"}))
    before = leiden.values(root)
    assert before["resolution"] == 1.2 and before["low_expr_threshold"] == -0.5 and before["n_neighbors"] == 15
    saved = leiden.save(root, {"markers": ["CD3"], "resolution": 0.8, "low_expr_threshold": -1.0}, revision(before))
    stored = json.loads(cache.read_text())
    assert stored["input_file"] == "kept.csv" and stored["markers"] == ["CD3"] and stored["min_marker_sum"] == -1.0
    assert saved["resolution"] == 0.8
    for wrong in ({"output_col": "leiden"}, {"resolution": 9}, {"markers": []}, {"n_neighbors": 2.5}, {"colour": 1}):
        with pytest.raises(ValueError):
            leiden.check(wrong)
    with pytest.raises(ValueError, match="another window"):
        leiden.save(root, {"resolution": 1.0}, revision(before))


def test_the_request_has_tab7s_keys(project):
    root, path = project
    request = leiden.pipeline_settings(leiden.values(root), path, 5.0)
    classic = """full_input_path markers_selected cell_id_col batch_id_col enable_filter filter_col filter_val norm_z1
        norm_z2 norm_cdf norm_nlog norm_sinh low_expr_metric enable_low_expr_filter low_expr_threshold resolution
        n_neighbors n_pcs use_intelligent subsample_size random_seed compute_umap meta_clusters use_gpu
        output_col_name cofactor fresh_assignment_mode""".split()
    assert set(request) == set(classic)


def test_leiden_finds_the_populations_then_names_are_saved(project):
    root, path = project
    settings = settings_for(root, path, resolution=0.3, enable_low_expr_filter=False)
    settings["revision"] = tasks.source_revision(settings)
    messages = []
    found = leiden.run(settings, root / "job", lambda text, share=None: messages.append(text))
    assert found["column"] == "phenotype_leiden_clusters" and found["cells"] == 1200 and messages
    profile = found["profile"]
    assert profile["rows"] == [item["name"] for item in found["clusters"]]
    assert profile["columns"] == ["CD3", "PanCK", "CD68", "CD20"] and len(profile["values"][0]) == 4
    cd3 = next(i for i, item in enumerate(found["clusters"]) if item["top_markers"][0] == "CD3")
    assert max(profile["values"][cd3]) == profile["values"][cd3][0]  # the CD3 cluster stands out in CD3
    tops = {item["top_markers"][0] for item in found["clusters"]}
    assert {"CD3", "PanCK", "CD68"} <= tops, found["clusters"]
    frame = dataio.load(path, columns=["cell_id", "phenotype_leiden_clusters"])
    assert frame["phenotype_leiden_clusters"].str.startswith("Cluster ").all()
    names = {item["name"]: item["top_markers"][0] + " cells" for item in found["clusters"]}
    naming = {
        "table": str(path),
        "markers": found["markers"],
        "output_col": "phenotype_leiden",
        "clusters_col": found["column"],
        "cell_id_col": "cell_id",
        "names": leiden.check_names(names, found["clusters"]),
    }
    naming["revision"] = tasks.source_revision(naming)
    named = leiden.name_clusters(naming, root / "job2", lambda text, share=None: None)
    table = dataio.load(path, columns=["CD3", "phenotype_leiden_clusters", "phenotype_leiden"])
    assert named["column"] == "phenotype_leiden" and set(named["phenotypes"]) <= set(names.values())
    assert (table["phenotype_leiden"] == table["phenotype_leiden_clusters"].map(names)).all()
    medians = table.groupby("phenotype_leiden")["CD3"].median()
    assert medians["CD3 cells"] > 5 * medians["PanCK cells"]
    with pytest.raises(ValueError):
        leiden.check_names({"Cluster 99": "T cells"}, found["clusters"])


def test_the_commands_check_before_starting(setup, monkeypatch):  # noqa: F811
    from spacemap_gui.studio import jobs
    from spacemap_gui.studio.settings import revision

    started = []
    monkeypatch.setattr(jobs, "start", lambda *args, **kwargs: started.append(args) or {"id": "x"})
    context = setup
    result = run(context, "run_leiden", {}, confirmed=True)
    assert "Choose the markers" in result["message"] and not started
    root = context[2]
    leiden.save(root, {"markers": ["M0", "M1"]}, revision(leiden.values(root)))
    result = run(context, "run_leiden", {}, confirmed=True)
    assert "cell_id" in result["message"] and not started
    result = run(context, "name_clusters", {"job_id": "0" * 32, "names": {"Cluster 0": "T"}}, confirmed=True)
    assert result["status"] != "done" and not started


def test_transformation_and_normalisation_are_tab7s_settings(project):
    from spacemap_gui.studio import clustering
    from spacemap_gui.studio.settings import revision

    root, _ = project
    saved = leiden.save(
        root, {"norm_z2": True, "norm_cdf": True, "batch_id_col": "slide"}, revision(leiden.values(root))
    )
    stored = json.loads(leiden.cache_path(root).read_text())
    assert stored["norm_z2"] is True and stored["batch_id_col"] == "slide" and saved["norm_z1"] is True
    request = leiden.pipeline_settings(saved, "t.csv", 5.0)
    assert request["norm_z2"] and request["norm_cdf"] and request["batch_id_col"] == "slide"
    with pytest.raises(ValueError, match="CDF before"):
        leiden.save(root, {"norm_cdf": False, "norm_nlog": True}, revision(leiden.values(root)))
    with pytest.raises(ValueError):
        leiden.check({"batch_id_col": "../x"})
    som = clustering.save(
        root, "som", {"norm_nlog": True, "norm_cdf": True, "batch_id_col": ""}, revision(clustering.values(root, "som"))
    )
    assert som["norm_nlog"] and som["batch_id_col"] == ""
