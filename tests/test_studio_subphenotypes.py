"""Studio's sub-phenotyping: classic's marker discovery and sub-clustering, one phenotype at a time."""

import json

import numpy as np
import pandas as pd
import pytest

from spacemap import dataio
from spacemap_gui.studio import subphenotypes, tasks
from test_studio_leiden import project  # noqa: F401  (fixture: CD3, PanCK and CD68 populations in R1 and R2)

MARKERS = ["CD3", "PanCK", "CD68", "CD20"]


@pytest.fixture
def parents(project):  # noqa: F811
    """The phenotype of each cell, and half the T cells in a second state (CD20 high)."""
    root, path = project
    frame = pd.read_csv(path)
    frame["phenotype_parent"] = np.array(["T cell", "Tumour", "Macrophage"])[
        np.argmax(frame[["CD3", "PanCK", "CD68"]].to_numpy(), axis=1)
    ]
    t_cells = np.flatnonzero(frame["phenotype_parent"] == "T cell")
    rng = np.random.default_rng(9)
    frame.loc[t_cells[::2], "CD20"] += rng.gamma(20.0, 60.0, len(t_cells[::2]))
    frame.to_csv(path, index=False)
    return root, path, t_cells


def job(root, path, mode, **changes):
    chosen = {
        **subphenotypes.values(root),
        "parent_col": "phenotype_parent",
        "parent_values": ["T cell"],
        "markers": MARKERS,
        "method": "Leiden",
        "resolution": 0.3,
        "n_meta_clusters": 2,
        "use_gpu": False,
        **changes,
    }
    settings = {
        "table": str(path),
        "markers": MARKERS,
        "output_col": "phenotype_sub_clusters",
        "phenotype_col": "phenotype_sub",
        "cell_id_col": "cell_id",
        "project_root": str(root),
        "config": chosen,
        "mode": mode,
    }
    settings["revision"] = tasks.source_revision(settings)
    return settings


def test_markers_are_found_inside_the_phenotype_then_it_is_split_on_them(parents, tmp_path):
    root, path, t_cells = parents
    found = subphenotypes.run(job(root, path, "discover"), tmp_path / "find", lambda *args: None)
    rows = {row["marker"]: row for row in found["discovery"]["T cell"]["markers"]}
    assert found["discovery"]["T cell"]["cells"] == len(t_cells)
    assert rows["CD20"]["recommended"] and not rows["PanCK"]["recommended"] and rows["PanCK"]["flag"]
    split = subphenotypes.run(job(root, path, "run"), tmp_path / "run", lambda *args: None)
    assert split["column"] == "phenotype_sub_clusters" and len(split["clusters"]) == 2
    assert {item["name"] for item in split["clusters"]} == {"T_cell_0", "T_cell_1"}
    # With two clusters all increased markers tie at z=1/sqrt(2); their exact order is not biological evidence.
    assert any("CD20" in item["top_markers"] for item in split["clusters"]) and split["markers_used"]["T cell"]
    assert split["profile"]["aggregation"] == "median"
    cd20 = split["profile"]["columns"].index("CD20")
    assert max(row[cd20] for row in split["profile"]["values"]) > 0
    labels = dataio.load(path, columns=["phenotype_parent", "phenotype_sub_clusters"])
    assert labels.loc[labels.phenotype_parent != "T cell", "phenotype_sub_clusters"].isna().all()
    assert labels.loc[labels.phenotype_parent == "T cell", "phenotype_sub_clusters"].notna().all()
    assert "of" in split["summary"] and split["checks"] and split["where"]
    assert split["cells"] == len(t_cells)
    json.dumps(split, allow_nan=False)  # the Jobs list sends it to the page as strict JSON


def test_settings_are_tab7s_file(parents):
    from spacemap_gui.studio.settings import revision

    root, _, _ = parents
    saved = subphenotypes.save(
        root,
        {"parent_values": ["T cell"], "chosen_markers": {"T cell": ["CD20", "CD3"]}},
        revision(subphenotypes.values(root)),
    )
    stored = json.loads(subphenotypes.cache_path(root).read_text())
    assert stored["chosen_markers"] == {"T cell": ["CD20", "CD3"]} and saved["method"] == "Leiden"
    for wrong in ({"method": "kmeans"}, {"studio_output_col": "sub"}, {"chosen_markers": ["CD3"]}, {"resolution": 9}):
        with pytest.raises(ValueError):
            subphenotypes.check(wrong)
