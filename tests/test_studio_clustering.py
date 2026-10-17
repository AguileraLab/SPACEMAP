"""Studio's SOM runs classic Tab 7's SOM steps as a job; its groups come with top markers, heatmap and where."""

import json

import pytest

from spacemap import dataio
from spacemap_gui.studio import clustering, tasks
from test_studio_leiden import project  # noqa: F401  (fixture: CD3, PanCK and CD68 populations in R1 and R2)

MARKERS = ["CD3", "PanCK", "CD68", "CD20"]


def som_job(root, path, **changes):
    config = {
        **clustering.values(root, "som"),
        "markers": MARKERS,
        "grid_dim": 4,
        "n_meta_target": 3,
        "mode": "fast",
        "subsample_size": 5000,
        "subsample_method": "random",
        "enable_low_expr_filter": False,
        **changes,
    }
    settings = {
        "table": str(path),
        "markers": MARKERS,
        "output_col": config["output_col"] + "_clusters",
        "phenotype_col": config["output_col"],
        "cell_id_col": "cell_id",
        "project_root": str(root),
        "config": config,
        "cofactor": 5.0,
    }
    settings["revision"] = tasks.source_revision(settings)
    return settings


def test_som_settings_are_tab7s_file(project):  # noqa: F811
    from spacemap_gui.studio.settings import revision

    root, _ = project
    cache = clustering.cache_path(root, "som")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"grid_dim": 12, "meta_method": "ward", "input_file": "kept.csv"}))
    before = clustering.values(root, "som")
    assert before["grid_dim"] == 12 and before["meta_method"] == "ward" and before["n_meta_target"] == 45
    clustering.save(root, "som", {"markers": ["CD3"], "n_meta_target": 20, "mode": "fast"}, revision(before))
    stored = json.loads(cache.read_text())
    assert stored["input_file"] == "kept.csv" and stored["n_meta_target"] == 20 and stored["mode"] == "fast"
    for wrong in ({"mode": "fastest"}, {"grid_dim": 1}, {"output_col": "som"}, {"meta_method": "ward"}):
        with pytest.raises(ValueError):
            clustering.check("som", wrong)
    with pytest.raises(ValueError, match="another window"):
        clustering.save(root, "som", {"grid_dim": 10}, revision(before))


def test_som_finds_the_populations(project):  # noqa: F811
    root, path = project
    found = clustering.run(som_job(root, path), root / "job", lambda *args: None)
    assert found["column"] == "phenotype_som_clusters" and found["cells"] == 1200
    assert len(found["clusters"]) == 3 and found["profile"]["columns"] == MARKERS
    tops = {item["top_markers"][0] for item in found["clusters"]}
    assert tops == {"CD3", "PanCK", "CD68"}, found["clusters"]
    frame = dataio.load(path, columns=["cell_id", "CD3", "PanCK", "phenotype_som_clusters"])
    cd3 = next(item["name"] for item in found["clusters"] if item["top_markers"][0] == "CD3")
    assert frame.groupby("phenotype_som_clusters")["CD3"].median()[cd3] > 5 * frame["PanCK"].median() / 5
    assert set(found["where"]) == {item["name"] for item in found["clusters"]}
    assert all({place["roi"] for place in places} <= {"R1", "R2"} for places in found["where"].values())


def test_dim_cells_form_the_low_expression_group(project):  # noqa: F811
    root, path = project
    found = clustering.run(
        som_job(root, path, enable_low_expr_filter=True, low_expr_threshold=-0.2, output_col="phenotype_som_dim"),
        root / "job",
        lambda *args: None,
    )
    names = {item["name"] for item in found["clusters"]}
    assert "Low expression" in names and sum(item["cells"] for item in found["clusters"]) == 1200


def test_vortex_finds_the_populations_with_tab7s_settings_file(project):  # noqa: F811
    from spacemap_gui.studio.settings import revision

    root, path = project
    assert clustering.values(root, "vortex")["low_expr_threshold"] == -0.75  # classic's default cut
    clustering.save(root, "vortex", {"markers": MARKERS, "vortex_k": 30}, revision(clustering.values(root, "vortex")))
    assert json.loads(clustering.cache_path(root, "vortex").read_text())["vortex_k"] == 30
    config = {
        **clustering.values(root, "vortex"),
        "auto_select_k": False,
        "use_gpu": False,
        "subsample_method": "random",
        "enable_low_expr_filter": False,
    }
    settings = {
        "table": str(path),
        "markers": MARKERS,
        "output_col": "phenotype_vortex_clusters",
        "phenotype_col": "phenotype_vortex",
        "cell_id_col": "cell_id",
        "project_root": str(root),
        "config": config,
        "method": "vortex",
        "cofactor": 5.0,
    }
    settings["revision"] = tasks.source_revision(settings)
    found = clustering.run(settings, root / "job", lambda *args: None)
    assert found["column"] == "phenotype_vortex_clusters" and found["k"] == 30 and found["cells"] == 1200
    tops = {item["top_markers"][0] for item in found["clusters"] if item["cells"] > 100}
    assert {"CD3", "PanCK", "CD68"} <= tops, found["clusters"]
    saved = dataio.load(path, columns=["phenotype_vortex_clusters"])["phenotype_vortex_clusters"]
    assert saved.str.startswith("Cluster ").all()
